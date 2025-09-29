#!/bin/bash

# network
source ~/clash.sh
bash ~/clash-for-linux-backup/start.sh
proxy_on

# env
source ~/anaconda3/etc/profile.d/conda.sh
conda config --append envs_dirs ~/.conda/envs
conda activate deepspeed
echo -e "\e[31m$CONDA_DEFAULT_ENV\e[0m"

export WANDB_API_KEY=""

export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HUGGINGFACE_HUB_TOKEN=""
export HF_ENDPOINT="https://hf-mirror.com"
# export HF_ENDPOINT="https://huggingface.co"
export HF_HOME="~/.cache/huggingface"
export PYTHONPATH='.'

cd ~/work/dev6/DiffusionAsShader/thirdparty/x-flux


GPU_IDS="all"
NUM_PROCESSES=4
PORT=29502

# Single GPU uncompiled training
ACCELERATE_CONFIG_FILE="~/work/dev6/DiffusionAsShader/accelerate_configs/gpu4.yaml"

accelerate launch \
        --config_file $ACCELERATE_CONFIG_FILE \
        --gpu_ids $GPU_IDS \
        --num_processes $NUM_PROCESSES \
        --main_process_port $PORT \
        train_flux_deepspeed_controlnet.py \
        --config "train_configs/test_depth_controlnet.yaml"


# GPU_IDS="0"
# NUM_PROCESSES=1
# PORT=29502

# # Single GPU uncompiled training
# ACCELERATE_CONFIG_FILE="~/work/dev6/DiffusionAsShader/accelerate_configs/gpu2.yaml"

# accelerate launch \
#         --config_file $ACCELERATE_CONFIG_FILE \
#         --num_processes $NUM_PROCESSES \
#         --main_process_port $PORT \
#         train_flux_deepspeed_controlnet.py \
#         --config "train_configs/test_depth_controlnet.yaml"
