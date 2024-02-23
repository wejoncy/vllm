import os
import sys

import json

ORT_PATH="/ws/code/onnxruntime"

sys.path.insert(0, f"{ORT_PATH}/onnxruntime/python/tools/kernel_explorer/kernels")
sys.path.insert(0, f"{ORT_PATH}/build_rocm/Release")
os.environ["KERNEL_EXPLORER_BUILD_DIR"] = f"{ORT_PATH}/build_rocm/Release"


import multiprocessing as mp
from multiprocessing import Pool, current_process


def profile(name, *args, **kwargs):
    import kernel_explorer as ke

    #ke.set_ort_severity(0)
    #ke.set_ort_verbosity(1000)
    ke.set_return_tuning_results()
    ke.set_dispatchable_pattern("*Tunable*")
    print(os.environ["HIP_VISIBLE_DEVICES"])
    if name == "gemm":
        from gemm_test import profile_with_args as profile

        return profile(*args, **kwargs)
    elif name == "softmax":
        from softmax_test import profile_with_args as profile

        return profile(*args, **kwargs)
    else:
        return []


def get_llama_gemm_configs(model):
    '''
    Get GEMM configs for a given model, the model structure is LLama family
    '''
    # TODO: need to support TP
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model)
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_size = hidden_size // num_heads
    qkv_proj = (hidden_size, (num_heads + 2 * num_kv_heads) * head_size)
    attn_out = (hidden_size, hidden_size)
    mlp1 = (hidden_size, intermediate_size * 2)
    mlp2 = (intermediate_size, hidden_size)

    configs = []

    M_max = 2048
    m = 1
    while m <= M_max:
        configs.append(("gemm", "float16", False, False, m, qkv_proj[1], qkv_proj[0]))
        configs.append(("gemm", "float16", False, False, m, attn_out[1], attn_out[0]))
        configs.append(("gemm", "float16", False, False, m, mlp1[1], mlp1[0]))
        configs.append(("gemm", "float16", False, False, m, mlp2[1], mlp2[0]))
        m += 1

    return configs


if __name__ == "__main__":
    import torch
    #configs = [
    #    #("gemm", "float16", transA, transB, m, n, k),
    #    ("gemm", "float16", False, False, 18, 4096, 4096),
    #    #("softmax", 1, 1024, False, "float16"),
    #]
    # configs = get_llama_gemm_configs("meta-llama/Llama-2-7b-hf")
    # configs = get_llama_gemm_configs("meta-llama/Llama-2-13b-hf")
    # configs = get_llama_gemm_configs("mistralai/Mistral-7B-v0.1")
    configs = get_llama_gemm_configs("microsoft/phi-2")
    save_res_file = "tuning-result-phi2-all-2048-size.json"

    mp.set_start_method("spawn")

    NUM_GPUS=12
    START_GPU=4

    def init():
        pidx = int(current_process()._identity[0]) - 1
        os.environ["HIP_VISIBLE_DEVICES"] = str(pidx % NUM_GPUS + START_GPU)
    
    with Pool(processes=NUM_GPUS, initializer=init) as pool:
        ret = pool.starmap(profile, configs, chunksize=1)

    from pprint import pprint
    from onnxruntime.tools.offline_tuning import Merger

    m = Merger()
    for tr in ret:
        m.merge(tr)

    with open(save_res_file, 'w') as fp:
       json.dump(m.get_merged(), fp)
    #pprint(m.get_merged())

