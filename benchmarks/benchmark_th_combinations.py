import subprocess
import os
import sys
import time
import re
import concurrent
import pandas as pd

import torch

def benchmark_throughput_in_parallel(model:str, input_len:int, output_len:int, num_prompt:int, cuda_device_id:int, with_ort:bool=False, backend='vllm'):
    my_env = os.environ.copy()
    free, total = torch.cuda.mem_get_info(cuda_device_id)
    while (total - free) / 1024/1024 > 500:
        time.sleep(10)
        free, total = torch.cuda.mem_get_info(cuda_device_id)
        print(f"waiting for memory of gpu{cuda_device_id} to be available")
    print(f"gpu{cuda_device_id} is used for {model} {input_len} {output_len} {num_prompt} {with_ort} {backend}")
    my_env["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
    command = [sys.executable, f"{os.path.dirname(__file__)}/benchmark_throughput.py", f"--backend={backend}",
                f"--model={model}",
                f"--input-len={input_len}",
                f"--output-len={output_len}",
                f"--num-prompt={num_prompt}",
                "--max-model-len=2400",
                # "--enforce-eager",
                ]
    if with_ort:
        command.append("--with-ort")
    handle = subprocess.Popen(
        command, stdout=subprocess.PIPE, env=my_env)
    handle.wait()
    output = handle.stdout.read().decode("utf-8").split("\n")
    print(f"gpu {cuda_device_id} done for {model} {input_len} {output_len} {num_prompt} {with_ort} {backend}")
    return [(input_len, output_len, num_prompt, with_ort, backend), output]

def benchmark_throughput(model: str, input_lens: list, output_lens: list, num_prompts: list, gpu_ids:list, with_ort: bool = False, backend='vllm'):
    outputs = []
    jobs = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
        for input_len in input_lens:
            for output_len in output_lens:
                for num_prompt in num_prompts:
                    job = executor.submit(benchmark_throughput_in_parallel, model,
                                          input_len, output_len, num_prompt, gpu_ids[len(jobs) % len(gpu_ids)], with_ort, backend)
                    jobs.append(job)
                    time.sleep(20)
    outputs = [job.result() for job in jobs]
    return outputs


def parse_metrics_for_th(outputs: list):
    metric_outs = []
    for output in outputs:
        (input_len, output_len, num_prompt, with_ort, backend), item_output = output
        item_output = [item for item in item_output if 'Throughput' in item][0]
        metric_out = re.findall(
            r'Throughput: ([0-9.]+) requests/s, ([0-9.]+) tokens/s', item_output)[0]
        metric_out = [float(item) for item in metric_out]
        metric_out = dict(zip(['requests/s', 'tokens/s'], metric_out))
        metric_out['input_len'] = input_len
        metric_out['output_len'] = output_len
        metric_out['num_prompt'] = num_prompt
        metric_out['with_ort'] = with_ort
        metric_out['backend'] = backend
        metric_outs.append(metric_out)
    return metric_outs


def benchmark_throughput_phi2(gpu_ids):
    input_lens = [16, 64, 256, 1024, 2048]
    output_lens = [128, 256]
    num_prompts = [1, 4, 16, 32, 64, 128]

    basename = "benchmark_throughput_phi2.csv"
    df = 0
    if not os.path.exists(basename.replace('.csv', '_ort_vllm.csv')):
        outputs = benchmark_throughput(
            "../phi-2/", input_lens, output_lens, num_prompts, gpu_ids, with_ort=True)
        metric_outs_ort_vllm = parse_metrics_for_th(outputs)
        df = pd.DataFrame(metric_outs_ort_vllm)
        df.to_csv(basename.replace('.csv', '_ort_vllm.csv'))
    else:
        print(f"skip {basename}")

    if not os.path.exists(basename.replace('.csv', '_vllm.csv')):
        outputs = benchmark_throughput(
            "../phi-2/", input_lens, output_lens, num_prompts)
        metric_outs_vllm = parse_metrics_for_th(outputs)
        df = pd.DataFrame(metric_outs_vllm)
        df.to_csv(basename.replace('.csv', '_vllm.csv'))
    else:
        print(f"skip {basename}")

    if not os.path.exists(basename.replace('.csv', '_mii.csv')):
        outputs = benchmark_throughput(
            "../phi-2/", input_lens, output_lens, num_prompts, backend="mii")
        metric_outs_mii = parse_metrics_for_th(outputs)
        df = pd.DataFrame(metric_outs_mii)
        df.to_csv(basename.replace('.csv', '_mii.csv'))
    else:
        print(f"skip {basename}")
    return df


if __name__ == "__main__":
    benchmark_throughput_phi2([0,1,2])
    print('done')
