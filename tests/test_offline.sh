#！/bin/bash
# Path: tests/test_offline.sh
export NCCL_DEBUG=WARN

if [ x"$1" != x"" ];then
    gpu=V100
else
    gpu=A100
fi

CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_throughput.py --backend=vllm --model=../Llama-2-7b-hf/ --with-ort --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json   -tp=1&
CUDA_VISIBLE_DEVICES=1 python benchmarks/benchmark_throughput.py --backend=vllm --model=../Llama-2-7b-hf/ --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json -tp=1&
wait

if [ "$gpu" = "V100" ]; then
    CUDA_VISIBLE_DEVICES=0,1 python benchmarks/benchmark_throughput.py --backend=vllm --model=../Llama-2-13b-hf/ --with-ort --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json   -tp=1
    CUDA_VISIBLE_DEVICES=0,1 python benchmarks/benchmark_throughput.py --backend=vllm --model=../Llama-2-13b-hf/ --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json -tp=1
else
    CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_throughput.py --backend=vllm --model=../Llama-2-13b-hf/ --with-ort --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json   -tp=1&
    CUDA_VISIBLE_DEVICES=1 python benchmarks/benchmark_throughput.py --backend=vllm --model=../Llama-2-13b-hf/ --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json -tp=1&
    wait
fi



if [ "$gpu" = "A100" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmarks/benchmark_throughput.py --backend=vllm --model=../Llama-2-70b-hf/ --with-ort --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json   -tp=4
    # fail for cuda-graph
    CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmarks/benchmark_throughput.py --backend=vllm --model=../Llama-2-70b-hf/ --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json -tp=4 --enforce-eager
fi
