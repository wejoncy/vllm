#!/bin/bash

# This script is used to prepare the package and data for the test.
# It will download the package and data from the public storage account.
# It will also build the package and install it to the current python environment.
if [ ! -d "/home/aiscuser/work" ];then
    mkdir /home/aiscuser/work
fi
cd  /home/aiscuser/work

function download_llama() {
    cd  /home/aiscuser/work
    if [ ! -f azcopy ];then
        wget https://aka.ms/downloadazcopy-v10-linux -O /tmp/az.tar && tar xf /tmp/az.tar -C /tmp/ --strip-components 1 && cp /tmp/azcopy ./
    fi
    
    if [ ! -f "Llama-2-70b-hf" ];then
        ./azcopy copy --recursive=true  "https://singdata.blob.core.windows.net/singularity/Llama-2-70b-hf?$KEY" ./
    fi
    if [ ! -f "Llama-2-13b-hf" ];then
        ./azcopy copy --recursive=true  "https://singdata.blob.core.windows.net/singularity/Llama-2-13b-hf?$KEY" ./
    fi
    if [ ! -f "Llama-2-7b-hf" ];then
        ./azcopy copy --recursive=true  "https://singdata.blob.core.windows.net/singularity/Llama-2-7b-hf?sp=rwl&$KEY" ./
    fi
}


function prepare_vllm() {
    cd  /home/aiscuser/work
    git clone --recurse https://github.com/facebookresearch/xformers.git
    cd xformers 
    python setup.py develop
    
    cd  /home/aiscuser/work
    git clone https://github.com/wejoncy/vllm.git
    cd vllm && git checkout ort
    python setup.py develop
    if [ ! -f "ShareGPT_V3_unfiltered_cleaned_split.json" ];then
        wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    fi
}
download_llama &
prepare_vllm &

cd  /home/aiscuser/work
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime 
bash build.sh --cmake_generator "Ninja" --config Release --cmake_extra_defines CMAKE_EXPORT_COMPILE_COMMANDS=ON --skip_tests  --build_wheel --use_cuda  --cudnn_home /usr/lib/x86_64-linux-gnu  --cuda_home /usr/local/cuda  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="80" --enable_nccl  --disable_types  float8

pip uninstall xformers onnxruntime-training -y
pip install build/Linux/Release/dist/onnxruntime_gpu-1.17.0-cp38-cp38-linux_x86_64.whl
cd  /home/aiscuser/work/xformers
python setup.py develop

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
sudo apt install netcat bc lrzsz
wait
