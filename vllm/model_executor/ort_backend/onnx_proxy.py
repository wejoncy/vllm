import sys
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
import numpy as np
from pathlib import Path
import tempfile

from vllm.model_executor.input_c_metadata import ConvertInputMetadataToC, GetAddrForCStruct, InputMetadata, set_current_input_metadata
from vllm.model_executor.parallel_utils.parallel_state import (get_tensor_model_parallel_group,
                                                               get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.sequence import SamplerOutput
from vllm.logger import init_logger

from vllm.file_baton import FileBaton
from vllm import _C 


KVCache = Tuple[torch.Tensor, torch.Tensor]
logger = init_logger(__name__)

MODEL_BASE_PATH = os.getenv(
    'MODEL_BASE_PATH', f'{tempfile.gettempdir()}/vllm/export_onnx')    

import onnxruntime  # noqa


class Identify(nn.Module):
    def __init__(self):
        super(Identify, self).__init__()

    def forward(self, lm_head_weight, hidden_states, input_metadata):
        return hidden_states


torch_dtype_to_numpy_dict = {
    torch.bool: np.bool_,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
}


class AutoONNXForCausalLM:
    def __init__(self, config, quant_config=None):
        self.config = config
        self.quant_config = quant_config
        self.model = None
        self.dummpy_inputs = None
        kv_dtype, scale_dtype = (torch.float16, torch.float16)
        self.none_tensor_k = torch.empty((1, 1, 1, 1, 1), dtype=kv_dtype).cuda()
        self.none_tensor_v = torch.empty((1, 1, 1, 1), dtype=kv_dtype).cuda()

        self.lm_head_weight = None
        self.onnx_filepath = None
        self.lmhead_filepath = None
        self.do_export = True
        self.ort_hidden_states = None
        self.ort_session = None
        self.output_name = None # for onnx model outside of vllm exporter
        self.input_keycache_name = None
        self.input_valuecache_name = None
        self.run_options = None
        self.has_bind_cache_kv  = False
        self.metadata_pointer_tensor = torch.empty((1), dtype=torch.int64, device='cpu')
        self.enable_ort = config.backend == 'ort'
        self.set_model(None, config.name_or_path or config.auto_map['AutoConfig'].split('--')[0])

    def check_input_output(self):
        self.output_name = self.ort_session.get_outputs()[0].name
        if len(self.ort_session.get_outputs()) > 1:
            assert self.output_name == "last_hidden_state"
        all_input_names = [i.name for i in self.ort_session.get_inputs()]
        all_input_names.remove("input_ids")
        all_input_names.remove("position_ids")
        if 'input_metadata' in all_input_names:
            all_input_names.remove("input_metadata")
        assert len(all_input_names) % 2 == 0
        import re
        all_input_names = list(set([re.sub(r'[0-9]+', '{idx}', i) for i in all_input_names]))
        assert len(all_input_names) == 2
        self.input_keycache_name, self.input_valuecache_name = (all_input_names[
            0], all_input_names[1]) if 'key' in all_input_names[0] else (all_input_names[1],all_input_names[0])

    def init_ort_session(self):
        if self.enable_ort and not self.do_export and self.ort_session is None:
            os.environ['LOCAL_WORLD_SIZE'] = str(get_tensor_model_parallel_world_size())
            os.environ['LOCAL_RANK'] = str(get_tensor_model_parallel_rank())
            provider_opt = {"device_id": get_tensor_model_parallel_rank(),
                            #"enable_cuda_graph": "true",
                            #"has_user_compute_stream" : "true",
                            #"user_compute_stream" : str(torch.cuda.current_stream().cuda_stream)
                            }
            self.model and self.model.to('cpu')
            self.model = None
            torch.cuda.empty_cache()
            session_options = onnxruntime.SessionOptions()
            session_options.register_custom_ops_library(_C.__file__)
            # Check whether GPU (NVIDIA/AMD) is available
            ep = "CUDAExecutionProvider"
            if torch.version.hip is not None:
                ep = "ROCMExecutionProvider"

            self.ort_session = onnxruntime.InferenceSession(
                self.onnx_filepath, providers=[(ep, provider_opt)], sess_options=session_options)
            self.check_input_output()

            self.ort_binding = self.ort_session.io_binding()
            self.has_position_ids_inputs = "position_ids" in [i.name for i in self.ort_session.get_inputs()]
            self.run_options = onnxruntime.RunOptions()
            self.run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")

    def set_model(self, model: nn.Module, model_path_or_name: str):
        self.model = model
        if self.onnx_filepath is None:
            import re
            while model_path_or_name[-1] == '/':
                model_path_or_name = model_path_or_name[:-1]
            model_path_or_name = model_path_or_name.split('/')[-1]
            model_path_or_name = re.sub(r'[^0-9a-zA-Z]', '_', model_path_or_name)

            onnx_model_name = model_path_or_name
            onnx_filepath = f"{MODEL_BASE_PATH}/{onnx_model_name}.onnx"
            if get_tensor_model_parallel_world_size() > 1:
                onnx_filepath = onnx_filepath.replace(".onnx", f"_rank_{get_tensor_model_parallel_rank()}.onnx")
            self.onnx_filepath = Path(onnx_filepath)
            baton = FileBaton(os.path.join('/tmp/', 'vllm_ort_lock'))
            with baton.exclude_lock():
                self.onnx_filepath.parent.exists() or self.onnx_filepath.parent.mkdir(parents=True, exist_ok=True)
        self.do_export = not self.onnx_filepath.exists()
        if self.do_export:
            logger.warn(f'{self.onnx_filepath} is not exist, will export onnx model on the fly............')

        self.init_ort_session()

    def export_onnx(self, ort_backend, input_ids: torch.Tensor,
                    positions: torch.Tensor,
                    input_metadata: InputMetadata,
                    kv_caches: List[KVCache],):
        if not self.do_export:
            return
        torch_module = ort_backend.torch_module
        def _export_onnx(input_ids, positions, input_metadata: InputMetadata, kv_caches):
            assert isinstance(input_metadata, InputMetadata)
            self.dummpy_inputs = (input_ids, positions, kv_caches, input_metadata
            ) if self.dummpy_inputs is None else self.dummpy_inputs
         
            layer_num = self.config.num_hidden_layers
            seq_dim = len(input_ids.shape) - 1
            onnx_dynamic_axes = {"input_ids": {seq_dim: "seq_len"}, "position_ids": {seq_dim: "seq_len"}}
            if seq_dim>0:
                onnx_dynamic_axes["input_ids"][0] = "batch_size"
                onnx_dynamic_axes["position_ids"][0] = "batch_size"
            onnx_inp_names = ["input_ids", "position_ids"]
            for i in range(layer_num):
                onnx_inp_names.append(f'key_cache.{i}')
                onnx_inp_names.append(f'value_cache.{i}')
                onnx_dynamic_axes[onnx_inp_names[-2]] = {0: "num_blocks",
                                                         1: "num_heads", 2: "head_size_x", 3: "block_size", 4: "x"}
                onnx_dynamic_axes[onnx_inp_names[-1]] = {0: "num_blocks",
                                                         1: "num_heads", 2: "head_size", 3: "block_size"}
                
            onnx_inp_names.append("input_metadata")
            onnx_inp_names = tuple(onnx_inp_names)

            onnx_out_names = ("last_hidden_state",)
            onnx_inputs = list(self.dummpy_inputs)
            for i in range(len(onnx_inputs)):
                if type(onnx_inputs[i]) is InputMetadata:
                    onnx_inputs[i] = torch.tensor([0], dtype=torch.int64)
                elif isinstance(onnx_inputs[i], list) and len(onnx_inputs[i]) == layer_num:
                    cache_ins = (self.none_tensor_k, self.none_tensor_v)
                    onnx_inputs[i] = [cache_ins for _ in range(layer_num)]

            import shutil
            rank = get_tensor_model_parallel_rank()
            onnx_path = Path(self.onnx_filepath)

            tmp_onnx = onnx_path.parent/f'tmp_{rank}'/onnx_path.name
            tmp_onnx.parent.exists() and shutil.rmtree(tmp_onnx.parent)
            tmp_onnx.parent.mkdir(parents=True)

            torch.onnx.export(model=torch_module, args=tuple(onnx_inputs), f=str(tmp_onnx), verbose=False, opset_version=17,
                              input_names=onnx_inp_names, output_names=onnx_out_names, dynamic_axes=onnx_dynamic_axes)

            onnx_path.exists() and onnx_path.unlink()
            (onnx_path.parent/onnx_path.with_suffix('.data').name).exists() and (
                onnx_path.parent/onnx_path.with_suffix('.data').name).unlink()

            import onnx
            onnx_model = onnx.load(str(tmp_onnx))
            assert onnx_model.graph.output[0].name == "last_hidden_state"
            del onnx_model.graph.output[1:]
            onnx.save_model(onnx_model, str(self.onnx_filepath), save_as_external_data=True, all_tensors_to_one_file=True,
                            location=onnx_path.with_suffix('.data').name, size_threshold=1024, convert_attribute=False)

            logger.info(f" rank:{rank} export onnx success. ------------")
            if get_tensor_model_parallel_world_size() > 1:
                torch.distributed.barrier(group=get_tensor_model_parallel_group())
            logger.info(f"all {get_tensor_model_parallel_world_size()} ranks export onnx success. ----------")

            self.do_export = False
            self.init_ort_session()

        sampler = torch_module.sampler
        torch_module.sampler = Identify()
        if input_ids.dim() > 1:
            input_ids = input_ids[:1,:]
            positions = positions[:1,:]
        with torch.no_grad():
            _export_onnx(input_ids, positions, input_metadata, kv_caches, )
        torch_module.sampler = sampler
        if hasattr(torch_module, 'lm_head_weight'):
            lm_head_weight = torch_module.lm_head_weight.required_grad_(False)
            lm_head_bias = None
        else:
            lm_head_weight = torch_module.lm_head.weight
            lm_head_bias = torch_module.lm_head.bias
            lm_head_weight.requires_grad = False
            if lm_head_bias is not None:
                lm_head_bias.requires_grad = False
        ort_backend.lm_head_weight = lm_head_weight.to(input_ids.device)
        ort_backend.lm_head_bias = lm_head_bias.to(input_ids.device) if lm_head_bias is not None else None
        ort_backend.torch_module = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
        kv_caches: List[KVCache],
    ) -> SamplerOutput:
        assert self.ort_session.get_inputs()[0].type == 'tensor(int64)' and input_ids.is_contiguous() and positions.is_contiguous()

        Y_shape = torch.Size((*input_ids.shape, self.config.hidden_size))
        if self.ort_hidden_states is None or self.ort_hidden_states.numel() < Y_shape.numel():
            self.ort_hidden_states = torch.empty((torch.prod(torch.tensor(
                Y_shape)),), dtype=torch.float16, device=input_ids.device).contiguous()

        self.ort_binding.bind_input(
            name='input_ids',
            device_type=input_ids.device.type,
            device_id=input_ids.device.index,
            element_type=np.int64,
            shape=tuple(input_ids.shape),
            buffer_ptr=input_ids.data_ptr(),
        )
        if self.has_position_ids_inputs:
            self.ort_binding.bind_input(
                name='position_ids',
                device_type=positions.device.type,
                device_id=positions.device.index,
                element_type=np.int64,
                shape=tuple(positions.shape),
                buffer_ptr=positions.data_ptr(),
            )
        input_metadata.max_prompt_len = input_ids.shape[-1]
        input_metadata.num_prompts = input_ids.shape[0]
        set_current_input_metadata(input_metadata)
        if self.has_bind_cache_kv == False:
            self.ort_binding.bind_input(
                name='input_metadata',
                device_type='cuda',
                device_id=positions.device.index,
                element_type=np.int64,
                shape=tuple(self.metadata_pointer_tensor.shape),
                buffer_ptr=self.metadata_pointer_tensor.data_ptr(),
            )
            kv_cache_dtype, scale_dtype = (torch_dtype_to_numpy_dict[torch.float16], None)
            for i in range(self.config.num_hidden_layers):
                k_tensor = kv_caches[i][0] if kv_caches[i][0] is not None else self.none_tensor_k
                v_tensor = kv_caches[i][1] if kv_caches[i][0] is not None else self.none_tensor_v
                self.ort_binding.bind_input(
                    name=self.input_keycache_name.format(idx=i),
                    device_type=k_tensor.device.type,
                    device_id=k_tensor.device.index,
                    element_type=kv_cache_dtype,
                    shape=tuple(k_tensor.shape),
                    buffer_ptr=k_tensor.data_ptr(),
                )
                self.ort_binding.bind_input(
                    name=self.input_valuecache_name.format(idx=i),
                    device_type=v_tensor.device.type,
                    device_id=v_tensor.device.index,
                    element_type=kv_cache_dtype,
                    shape=tuple(v_tensor.shape),
                    buffer_ptr=v_tensor.data_ptr(),
                )
            self.has_bind_cache_kv = kv_caches[0][0] is not None

        self.ort_binding.bind_output(
            name=self.output_name,
            device_type=self.ort_hidden_states.device.type,
            device_id=self.ort_hidden_states.device.index,
            element_type=np.float16,
            shape=tuple(Y_shape),
            buffer_ptr=self.ort_hidden_states.data_ptr(),
        )
        self.ort_session.run_with_iobinding(self.ort_binding, run_options=self.run_options)
        return self.ort_hidden_states[:Y_shape.numel()].view(Y_shape)