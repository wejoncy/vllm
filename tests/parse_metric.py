import pandas
import numpy as np
import re
import sys
import os

def parse_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    
    res_list = []
    rps = 1.0
    for line in lines:
        if 'Total time:' in line:
            out = re.findall(r'Total time: ([0-9.]+) s', line)[0]
            res_list.append([rps])
            rps+=0.5
            res_list[-1].append(float(out))
        elif 'Throughput:' in line:
            out = re.findall(r'Throughput: ([0-9.]+) requests/s', line)[0]
            res_list[-1].append(float(out))
        elif 'Average latency:' in line:
            out = re.findall(r'Average latency: ([0-9.]+) s', line)[0]
            res_list[-1].append(float(out))
        elif 'Average latency per token:' in line:
            out = re.findall(r'Average latency per token: ([0-9.]+) s', line)[0]
            res_list[-1].append(float(out))
        elif 'Average latency per output token:' in line:
            out = re.findall(r'Average latency per output token: ([0-9.]+) s', line)[0]
            res_list[-1].append(float(out))
        
    df = pandas.DataFrame(res_list, columns=['RPS','Total time','Throughput','Average latency','Average latency per token','Average latency per output token'])
    return df


if __name__ == "__main__":
    if len(sys.argv) == 3:
        base_name = os.path.basename(sys.argv[1])
        print(base_name, '\n', parse_file(sys.argv[1]))
        base_name = os.path.basename(sys.argv[2])
        print(base_name, '\n', parse_file(sys.argv[2]))
    else:
        print('Usage: python parse_metric.py <file1> <file2>')
        sys.exit(1)


