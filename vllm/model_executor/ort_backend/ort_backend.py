
from typing import List, Optional, Tuple

import torch
from torch import nn

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.sequence import SamplerOutput
from .onnx_proxy import AutoONNXForCausalLM

KVCache = Tuple[torch.Tensor, torch.Tensor]


class ORTBackend(nn.Module):
    def __init__(
        self,
        torch_model_create_func: callable,
        config,
        quant_config: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.lm_head_weight = None

        self.ort_model = AutoONNXForCausalLM(config, quant_config)

        self.torch_module = torch_model_create_func() if self.ort_model.do_export else None
        self.config = config


        from vllm.model_executor.layers.sampler import Sampler
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> SamplerOutput:
        self.ort_model.export_onnx(self, input_ids, positions, input_metadata, kv_caches, )
        hidden_states = self.ort_model.forward(input_ids, positions, input_metadata, kv_caches)
        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        next_tokens = self.sampler(self.lm_head_weight, hidden_states,
                                   sampling_metadata)
        return next_tokens
        
    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        from pathlib import Path
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        lmhead_weight_path = Path(self.ort_model.onnx_filepath).parent / \
            f'{self.ort_model.onnx_filepath.stem}_lm_head_rank_{tensor_model_parallel_rank}.pt'
        torch_module = self.torch_module

        def retrieve_model(torch_module):
            for name, mod in torch_module.named_children():
                if name != 'sampler':
                    return name, mod
            return "model", None

        if self.ort_model.enable_ort and not self.ort_model.do_export:
            lm_head_weight = torch.load(lmhead_weight_path)
            if self.lm_head_weight is None:
                self.lm_head_weight = lm_head_weight
            else:
                self.lm_head_weight.data = lm_head_weight.data
            ret = None
        else:
            ret = torch_module.load_weights(model_name_or_path, cache_dir,
                                            load_format, revision)
        if self.ort_model.enable_ort and self.ort_model.do_export:
            self.ort_model.set_model(retrieve_model(torch_module)[1], model_name_or_path)
            if hasattr(torch_module, 'lm_head'):
                torch.save(torch_module.lm_head.weight.data, lmhead_weight_path)
            else:
                torch.save(torch_module.lm_head_weight.data, lmhead_weight_path)
