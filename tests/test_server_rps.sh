#！/bin/bash

set -e -x

export NCCL_DEBUG=WARN

function SendRequests() { #model_name
    model_name=$1
    port=$2

    start=1
    end=7
    interval=5

    logname=serving_${model_name##*/}_$port.log
    echo "port = "$port | tee  $logname

    for ((i=$start*10; i<=$end*10; i+=$interval)); do
        rps_number=$(echo "scale=1; $i/10" | bc)
        echo "rps = "$rps_number | tee -a  $logname
        python benchmarks/benchmark_serving.py --backend vllm --tokenizer ${model_name} \
        --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json  --request-rate $rps_number --port $port |tee -a  $logname
    done
}

function LaunchServer() { #model_name
    model_name=$1
    backend=$2
    port=$3
    cuda_device=$4
    TP=$(echo "$cuda_device"|grep , -o |wc -l)
    TP=`expr $TP + 1`
    extra_flag=""
    if [ "$TP" -gt "1" ]; then
         extra_flag="--enforce-eager"
    fi
    CUDA_VISIBLE_DEVICES=${cuda_device} python -m vllm.entrypoints.api_server \
    --model ${model_name} --swap-space 16 --disable-log-requests --backend=$backend \
    --port=$port -tp=${TP} ${extra_flag} >> server_$backend.log 2>&1  &
    echo $!
}

function WaitPortReady(){
    port=$1
    while ! nc -z localhost $port; do   
        sleep 3 # wait for 3 seconds before check again
    done
    echo "port ${port} is ready"

}

function BenchmarkModel() { #model_name
    pids=()
    model_name=$1

    ports=(8001 8000)
    backends=(ort torch)
    cuda_devices=("0" "1")

    for idx in $(seq 0 1);
    do
	    backend=${backends[idx]}
	    cuda_device=${cuda_devices[idx]}
	    port=${ports[idx]}

        cpid=$(LaunchServer $model_name $backend $port ${cuda_device})
        pids+=( "$cpid" )
    done

    for idx in $(seq 0 1);
    do
	port=${ports[idx]}
        WaitPortReady $port
    done
    
    SendRequests $model_name ${ports[0]}  & pids+=( "$!" )
    SendRequests $model_name ${ports[1]}
    wait ${pids[2]}
    kill -9 ${pids[0]}
    kill -9 ${pids[1]}
    #kill -9 $(ps aux | grep "vllm.entrypoints.api_server" | grep -v grep | awk '{print $2}')
}

function BenchmarkModelWithTP() { #model_name
    pids=()
    model_name=$1
    TP=$2
    if [ "$TP" = "2" ];then
        cuda_device="0,1"
    else
        cuda_device="0,1,2,3"
    fi
    backend=ort
    port=8001
    cpid=$(LaunchServer $model_name $backend $port ${cuda_device})
    
    WaitPortReady $port
    SendRequests $model_name $port
    pkill -9 -P ${cpid}
    
    sleep 10
    backend=torch
    port=8000
    cpid=$(LaunchServer $model_name $backend $port ${cuda_device})
    WaitPortReady $port
    SendRequests $model_name $port
    pkill -9 -P ${cpid}
}

#benchmark three models one by one 
#BenchmarkModel ../Llama-2-7b-hf
#BenchmarkModel ../Llama-2-13b-hf
#BenchmarkModelWithTP ../Llama-2-70b-hf 4
BenchmarkModelWithTP ../Llama-2-13b-hf 2
