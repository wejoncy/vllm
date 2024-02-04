import argparse

import subprocess
import os
import sys
import re
import pandas as pd


def benchmark_throughput(model: str, input_lens: list, output_lens: list, num_prompts: list, with_ort: bool = False, backend='vllm'):
    outputs = []
    for input_len in input_lens:
        for output_len in output_lens:
            for num_prompt in num_prompts:
                command = [sys.executable, f"{os.path.dirname(__file__)}/benchmark_throughput.py", f"--backend={backend}",
                           f"--model={model}",
                           f"--input-len={input_len}",
                           f"--output-len={output_len}",
                           f"--num-prompt={num_prompt}"]
                if with_ort:
                    command.append("--with-ort")
                handle = subprocess.Popen(command, stdout=subprocess.PIPE)
                handle.wait()
                output = handle.stdout.read().decode("utf-8").split("\n")
                outputs.append(
                    [(input_len, output_len, num_prompt, with_ort, backend), output])
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


def benchmark_throughput_main(args: argparse.Namespace):
    input_lens = [16, 64, 256, 1024, 2048]
    output_lens = [128, 256]
    num_prompts = [1, 4, 16, 32, 64, 128]

    basename = "benchmark_throughput_phi2.csv"
    df = None
    if not os.path.exists(basename.replace('.csv', '_ort_vllm.csv')):
        outputs = benchmark_throughput(
            args.model, input_lens, output_lens, num_prompts, with_ort=True)
        metric_outs_ort_vllm = parse_metrics_for_th(outputs)
        df = pd.DataFrame(metric_outs_ort_vllm)
        df.to_csv(basename.replace('.csv', '_ort_vllm.csv'))

    if not os.path.exists(basename.replace('.csv', '_vllm.csv')):
        outputs = benchmark_throughput(
            args.model, input_lens, output_lens, num_prompts)
        metric_outs_vllm = parse_metrics_for_th(outputs)
        df = pd.DataFrame(metric_outs_vllm)
        df.to_csv(basename.replace('.csv', '_vllm.csv'))

    if not os.path.exists(basename.replace('.csv', '_mii.csv')):
        outputs = benchmark_throughput(
            args.model, input_lens, output_lens, num_prompts, backend="mii")
        metric_outs_mii = parse_metrics_for_th(outputs)
        df = pd.DataFrame(metric_outs_mii)
        df.to_csv(basename.replace('.csv', '_mii.csv'))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--model",
                        type=str,
                        default="microsoft/phi-2")
    args = parser.parse_args()

    benchmark_throughput_main(args)
    print('done')
