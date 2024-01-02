"""Inference-only Mixtral model."""
import argparse
import os
from multiprocessing import Process, set_start_method
from vllm.worker.worker import _init_distributed_environment
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size, get_tensor_model_parallel_group)
from vllm.config import ParallelConfig
from transformers import MixtralConfig
from torch import nn
import math
import shutil
import torch.nn.functional as F
import transformers
from typing import List, Optional, Tuple
import sys
import numpy as np
from modeling_mixtral import MixtralForCausalLM

import torch
from pathlib import Path
torch.zeros(1).cuda()

CUR_PATH = Path(__file__).parent.absolute()

def init_test_distributed_environment(tensor_parallel_size: int, rank: int,
                                      distributed_init_port: str = "51409"):
    parallel_config = ParallelConfig(1, tensor_parallel_size,
                                     worker_use_ray=True)
    distributed_init_method = f"tcp://127.0.0.1:{distributed_init_port}"
    torch.cuda.set_device(rank)
    _init_distributed_environment(
        parallel_config, rank, distributed_init_method)

def export_onnx_no_tp(model: torch.nn.Module, onnx_path_str: str, sample_inputs: tuple, with_past: bool = False, opset=16) -> Path:
    from onnxruntime.transformers import large_model_exporter

    # since onnxruntime 1.7
    sample_inputs = (sample_inputs.input_ids, sample_inputs.attention_mask)
    sample_inputs_tp = list(sample_inputs)
    if sample_inputs_tp[1] is None:
        sample_inputs_tp[1] = torch.ones_like(sample_inputs_tp[0])
    model = large_model_exporter.move_to_appropriate_device(model, sample_inputs_tp)

    sample_inputs = large_model_exporter.adapt_inputs_to_device(
        sample_inputs_tp, next(model.parameters()).device)
    model.device = next(model.parameters()).device
    # input_keys would be usesful if the model has some special inputs
    input_keys, onnx_inputs, past_key_values = large_model_exporter.retrieve_onnx_inputs(model, sample_inputs, with_past)

    onnx_io_tuple = large_model_exporter.fetch_onnx_inputs_outputs_name(model, onnx_inputs, input_keys, past_key_values, with_past, False)

    onnx_model_name = "mixtral_rank0.onnx"
    onnx_path: Path = Path(onnx_path_str).absolute()
    onnx_path_enc = onnx_path / onnx_model_name if onnx_path.suffix != ".onnx" else onnx_path
    onnx_path_enc.parent.mkdir(parents=True, exist_ok=True)

    large_model_exporter.do_export_internal(
        model, onnx_io_tuple, onnx_inputs, onnx_path_enc, opset)
    if not with_past:
        return onnx_path_enc

    onnx_io_tuple = large_model_exporter.fetch_onnx_inputs_outputs_name(model, onnx_inputs, input_keys, past_key_values, with_past, True)
    # workaround for attention_mask
    onnx_inputs[1] = onnx_inputs[1].long()

    onnx_model_name = "mixtral_with_past_rank0.onnx"
    onnx_path_dec = onnx_path_enc.parent / onnx_model_name

    large_model_exporter.do_export_internal(
        model, onnx_io_tuple, onnx_inputs, onnx_path_dec, opset)

    onnx_path_one_for_all = onnx_path_enc.parent / "model_one_for_all.onnx"
    #merge_decoders(onnx_path_enc, onnx_path_dec, save_path=onnx_path_one_for_all)
    return onnx_path_one_for_all

def export_onnx(model_name, tensor_parallel_size, rank, onnx_path):
    torch.set_default_dtype(torch.float16)
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    inputs = tokenizer("once upon a time ", return_tensors="pt").to("cuda")
    torch.cuda.set_device(rank)
    model = MixtralForCausalLM(config)
    model.load_weights(model_name)
    model.eval()
    if tensor_parallel_size == 1:
        return export_onnx_no_tp(model, f"{onnx_path}/onnx_models_int4",inputs,True)
    model.to("cuda")
    out, past_key_values = model(**inputs)
    onnx_pt_inputs = (inputs.input_ids, inputs.attention_mask, None, None)
    tmp_onnx = Path(f"{onnx_path}/tmp{rank}/mixtral_rank{rank}.onnx")
    tmp_onnx.parent.exists() and shutil.rmtree(tmp_onnx.parent)
    tmp_onnx.parent.mkdir(exist_ok=True)
    onnx_model_path = Path(f"{onnx_path}/onnx_models/mixtral_rank{rank}.onnx").absolute()
    onnx_model_path.parent.mkdir(exist_ok=True)
    onnx_model_path.exists() and onnx_model_path.unlink()
    (onnx_model_path.parent/onnx_model_path.with_suffix('.data').name).exists() and (
        onnx_model_path.parent/onnx_model_path.with_suffix('.data').name).unlink()
    onnx_inp_names = ("input_ids","attention_mask")
    onnx_out_names = ("last_hidden_state",)
    for layer_idx in range(model.config.num_hidden_layers):
        onnx_out_names = onnx_out_names + \
            (f"past_key.{layer_idx}",
             f"past_value.{layer_idx}")
    torch.onnx.export(model=model, args=tuple(onnx_pt_inputs), f=str(tmp_onnx), verbose=False, opset_version=17,
                      input_names=tuple(onnx_inp_names), output_names=tuple(onnx_out_names),
                      dynamic_axes={"input_ids": {
                          0: "batch_size", 1: "seq_len"}},
                      autograd_inlining=False)
    torch.distributed.barrier(group=get_tensor_model_parallel_group())
    import onnx
    onnx_model = onnx.load(str(tmp_onnx))
    onnx.save_model(onnx_model, str(onnx_model_path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=tmp_onnx.with_suffix('.data').name, size_threshold=1024, convert_attribute=False)

    onnx_pt_inputs = (inputs.input_ids, inputs.attention_mask, None, past_key_values)
    tmp_onnx = Path(f"{onnx_path}/tmp{rank}/mixtral_with_past_rank{rank}.onnx")
    tmp_onnx.parent.exists() and shutil.rmtree(tmp_onnx.parent)
    tmp_onnx.parent.mkdir(exist_ok=True)
    onnx_pastmodel_path = Path(
        f"{onnx_path}/onnx_models/mixtral_with_past_rank{rank}.onnx").absolute()
    onnx_pastmodel_path.exists() and onnx_pastmodel_path.unlink()
    (onnx_pastmodel_path.parent/onnx_pastmodel_path.with_suffix('.data').name).exists() and (
        onnx_pastmodel_path.parent/onnx_pastmodel_path.with_suffix('.data').name).unlink()
    onnx_inp_names = ("input_ids", "attention_mask")
    for layer_idx in range(model.config.num_hidden_layers):
        onnx_inp_names = onnx_inp_names + \
            (f"present_key.{layer_idx}", f"present_values.{layer_idx}")
    dynamic_axes = {"input_ids": {0: "batch_size", 1: "seq_len"}}
    for layer_idx in range(model.config.num_hidden_layers):
        dynamic_axes[f"present_key.{layer_idx}"] = {
            0: "batch_size", 2: "seq_len", 1: "num_heads", 3: "head_dim"}
        dynamic_axes[f"present_values.{layer_idx}"] = {
            0: "batch_size", 2: "seq_len", 1: "num_heads", 3: "head_dim"}
    torch.onnx.export(model=model, args=tuple(onnx_pt_inputs), f=str(tmp_onnx), verbose=False, opset_version=17,
                      input_names=tuple(onnx_inp_names), output_names=tuple(onnx_out_names),
                      dynamic_axes=dynamic_axes,
                      autograd_inlining=False)
    torch.distributed.barrier(group=get_tensor_model_parallel_group())
    import onnx
    onnx_model = onnx.load(str(tmp_onnx))
    onnx.save_model(onnx_model, str(onnx_pastmodel_path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=tmp_onnx.with_suffix('.data').name, size_threshold=1024, convert_attribute=False)


def infer_model(model_name, tensor_parallel_size, rank, model_or_sess, onnx_path):
    middle_onnx_path = "onnx_models" if tensor_parallel_size > 1 else "onnx_models_int4"

    torch.set_default_dtype(torch.float16)
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    inputs = tokenizer("once upon a time ", return_tensors="pt").to("cuda")

    if not isinstance(model_or_sess, nn.Module):
        attention_mask = inputs.attention_mask.cpu().numpy()
        onnx_inputs = {"input_ids": inputs.input_ids.cpu().numpy(), "attention_mask":inputs.attention_mask.cpu().numpy()}
        ortout = model_or_sess.run(None, onnx_inputs)
        out = torch.from_numpy(ortout[0]).cuda()
        onnx_model_path = Path(
            f"{onnx_path}/{middle_onnx_path}/mixtral_with_past_rank{rank}.onnx").absolute()
        import onnxruntime
        from vllm import paged_attn
        session_options = onnxruntime.SessionOptions()
        session_options.register_custom_ops_library(paged_attn.__file__)
        provider_opt = {"device_id": rank, }
        model_or_sess = onnxruntime.InferenceSession(str(onnx_model_path), providers=[(
            "CUDAExecutionProvider", provider_opt)], sess_options=session_options)
    else:
        outputs = model_or_sess(**inputs)
        out, past_key_values = outputs.values()
    gen_ids = []
    while len(gen_ids) < 100:
        next_id = out[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids.append(next_id)
        if isinstance(model_or_sess, nn.Module):
            outputs = model_or_sess(next_id, past_key_values=past_key_values)
            out, past_key_values = outputs.values()
        else:
            attention_mask = np.concatenate([attention_mask,np.ones((attention_mask.shape[0],1),attention_mask.dtype)], axis=-1)
            onnx_inputs = {"input_ids": next_id.cpu().numpy(), "attention_mask":attention_mask}
            for layer_idx in range(config.num_hidden_layers):
                onnx_inputs[f"present_key.{layer_idx}"] = ortout[1+layer_idx*2]
                onnx_inputs[f"present_values.{layer_idx}"] = ortout[1+layer_idx*2+1]

            ortout = model_or_sess.run(None, onnx_inputs)
            out = torch.from_numpy(ortout[0]).cuda()
    gen_ids = torch.cat(gen_ids, dim=-1)
    if rank == 0:
        print(tokenizer.decode(gen_ids[0]))


def test_infer_model(model_name, tensor_parallel_size, rank, onnx_path):
    middle_onnx_path = "onnx_models" if tensor_parallel_size > 1 else "onnx_models_int4"
    torch.set_default_dtype(torch.float16)
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True)

    test_torch = False
    os.environ['LOCAL_WORLD_SIZE'] = str(tensor_parallel_size)
    os.environ['LOCAL_RANK'] = str(rank)
    torch.cuda.set_device(rank)
    if test_torch:
        model = MixtralForCausalLM(config)
        model.load_weights(model_name)
        model.eval()
        model.to("cuda")
        #out, past_key_values = model(**inputs)
        # del model,out
        # torch.cuda.empty_cache()
    else:
        import onnxruntime
        from vllm import paged_attn
        session_options = onnxruntime.SessionOptions()
        session_options.register_custom_ops_library(paged_attn.__file__)
        provider_opt = {"device_id": rank, }
        onnx_model_path = Path(
            f"{onnx_path}/{middle_onnx_path}/mixtral_rank{rank}.onnx").absolute()
        sess = onnxruntime.InferenceSession(str(onnx_model_path), providers=[(
            "CUDAExecutionProvider", provider_opt)], sess_options=session_options)

    infer_model(model_name, tensor_parallel_size, rank, model if test_torch else sess, onnx_path)


def process_entry(tensor_parallel_size, rank, cli_args):
    init_test_distributed_environment(tensor_parallel_size, rank)

    if cli_args.export_onnx:
        export_onnx(cli_args.model, tensor_parallel_size, rank, cli_args.onnx_path)
    
    if cli_args.infer:
        test_infer_model(cli_args.model, tensor_parallel_size, rank, cli_args.onnx_path)


def main(cli_args):
    tensor_parallel_size = cli_args.tensor_parallel_size
    if tensor_parallel_size == 1:
        process_entry(tensor_parallel_size, 0, cli_args)
    else:
        set_start_method("spawn", force=True)

        processes = []
        for rank in range(tensor_parallel_size):
            p = Process(target=process_entry,
                        args=(tensor_parallel_size, rank, cli_args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

if __name__ == "__main__":
    import sys
    sys.argv = ["", "-i=torch", "-e"]
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e', '--export_onnx',  action="store_true", default=False, help='export onnx models')
    parser.add_argument('--onnx_path', type=str, default=CUR_PATH,  help='onnx path')
    parser.add_argument("-m", '--model', type=str, default=f'{CUR_PATH}/Mixtral-8x7B-Instruct-v0.1/',  help='onnx path where load from or saved to')
    parser.add_argument('-tp', '--tensor_parallel_size', type=int, default=1, choices=[1,2,4,8],  help='tp size')
    parser.add_argument('-i', '--infer', type=str, default="", choices=["", "torch", "ort"],  help='the backend to run models')

    cli_args = parser.parse_args()
    main(cli_args)

