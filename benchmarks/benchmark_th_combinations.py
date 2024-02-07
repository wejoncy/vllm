import subprocess
import os
import sys
import time
import re
import concurrent
import pandas as pd
import socket
import torch


def launch_vllm_server(model: str, cuda_device_id: int, port:int, with_ort: bool = False):
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
    command = [sys.executable, "-m", "vllm.entrypoints.api_server",
                f"--model={model}", "--swap-space=16", "--disable-log-requests",
                f"--port={port}"]
    if with_ort:
        command.append("--backend=ort")
    print(f"running CUDA_VISIBLE_DEVICES={cuda_device_id} {' '.join(command)}")
    handle = subprocess.Popen(command, stdout=subprocess.PIPE, env=my_env)
    return handle

def check_server_is_ready(ip:str, port:int):
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
    sock.settimeout(5)  
    try:  
        result = sock.connect_ex((ip, port))  
        return result == 0
    except Exception as e:  
        print(f"error: {str(e)}")  
    finally:  
        sock.close()

def get_free_port():  
    sock = socket.socket()
    sock.bind(('', 0))
    ip, port = sock.getsockname()
    sock.close()
    return port

def benchmark_serving_in_parallel(model:str, num_prompt:int, cuda_device_id:int, with_ort:bool=False, backend='vllm'):
    my_env = os.environ.copy()
    free, total = torch.cuda.mem_get_info(cuda_device_id)
    while (total - free) / 1024/1024 > 500:
        time.sleep(10)
        free, total = torch.cuda.mem_get_info(cuda_device_id)
        print(f"waiting for memory of gpu{cuda_device_id} to be available")
    print(f"gpu{cuda_device_id} is used for {model} {num_prompt} {with_ort} {backend}")
    my_env["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
    server_port = get_free_port()
    server_handle = launch_vllm_server(model, cuda_device_id, server_port, with_ort)
    n_wait = 0
    while not check_server_is_ready("localhost", server_port):
        if server_handle.poll() is not None:
            print(f"failed to launch server for {model} {num_prompt} {with_ort} {backend}")
            return
        time.sleep(1)
        print(f"\rwaiting for server to be ready"+"."*n_wait, end='')
        n_wait+=1
    print("server is ready")
    command = [sys.executable, f"{os.path.dirname(__file__)}/benchmark_serving.py", f"--backend={backend}",
               f"--tokenizer={model}",
               "--dataset=./ShareGPT_V3_unfiltered_cleaned_split.json",
                f"--num-prompt={num_prompt}",
                "--request-rate=1.0",
                f"--port={server_port}"
                ]

    print(f"running CUDA_VISIBLE_DEVICES={cuda_device_id} {' '.join(command)}")
    client_handle = subprocess.Popen(command, stdout=subprocess.PIPE, env=my_env)
    client_ret = client_handle.communicate()
    server_handle.terminate()
    server_handle.communicate()
    if client_handle.returncode != 0:
        print(f"gpu {cuda_device_id} failed for {model} {num_prompt} {with_ort} {backend}")
        return
    output = client_ret[0].decode("utf-8").split('\n')
    output_dict = {}
    for o in output:
        if ':' in o:
            k,v = o.split(':')
            output_dict[k] = float(v.strip().split(' ')[0])
    print(
        f"gpu {cuda_device_id} done for {model}  {num_prompt} {with_ort} {backend}, {output_dict}")

    return [(num_prompt, with_ort, backend), output_dict]

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
    print(f"running CUDA_VISIBLE_DEVICES={cuda_device_id} {command}")
    handle = subprocess.Popen(command, stdout=subprocess.PIPE, env=my_env)
    handle.wait()
    if handle.returncode != 0:
        print(f"gpu {cuda_device_id} failed for {model} {input_len} {output_len} {num_prompt} {with_ort} {backend}")
    output = handle.stdout.read().decode("utf-8").split("\n")
    print(f"gpu {cuda_device_id} done for {model} {input_len} {output_len} {num_prompt} {with_ort} {backend}")
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
    metric_outs = parse_metrics_for_th(output)
    return [(input_len, output_len, num_prompt, with_ort, backend), metric_outs]

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
    outputs = [job.result() for job in jobs]
    return outputs



def benchmark_throughput_phi2(gpu_ids):
    input_lens = [16, 64, 256, 1024, 2048]
    output_lens = [128, 256]
    num_prompts = [1, 4, 16, 32, 64, 128]

    basename = "benchmark_throughput_phi2.csv"
    df = 0
    if not os.path.exists(basename.replace('.csv', '_ort_vllm.csv')):
        metric_outs_ort_vllm = benchmark_throughput(
            "../phi-2/", input_lens, output_lens, num_prompts, gpu_ids, with_ort=True)
        df = pd.DataFrame(metric_outs_ort_vllm)
        df.to_csv(basename.replace('.csv', '_ort_vllm.csv'))
    else:
        print(f"skip {basename}")

    if not os.path.exists(basename.replace('.csv', '_vllm.csv')):
        metric_outs_vllm = benchmark_throughput(
            "../phi-2/", input_lens, output_lens, num_prompts, gpu_ids)
        df = pd.DataFrame(metric_outs_vllm)
        df.to_csv(basename.replace('.csv', '_vllm.csv'))
    else:
        print(f"skip {basename}")

    if not os.path.exists(basename.replace('.csv', '_mii.csv')):
        metric_outs_mii = benchmark_throughput(
            "../phi-2/", input_lens, output_lens, num_prompts, gpu_ids, backend="mii")
        df = pd.DataFrame(metric_outs_mii)
        df.to_csv(basename.replace('.csv', '_mii.csv'))
    else:
        print(f"skip {basename}")
    return df

def benchmark_rps_phi2():
    out = benchmark_serving_in_parallel("../phi-2/", 1, 0, with_ort=True)


if __name__ == "__main__":
    benchmark_rps_phi2()
    benchmark_throughput_phi2([0,1,2])
    print('done')
