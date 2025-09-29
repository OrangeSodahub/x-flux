#!/bin/bash

# network
# source ~/clash.sh
# bash ~/clash-for-linux-backup/start.sh
# proxy_on

# env
# source ~/anaconda3/etc/profile.d/conda.sh
# conda config --append envs_dirs ~/.conda/envs
# conda activate deepspeed
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

cd ~/work/dev6/DiffusionAsShader/thirdparty/x-flux/


python3 main_depth.py \
    --prompt "A photo of real-wolrd robotics manipulation." \
    --depth ~/work/dev6/DiffusionAsShader/thirdparty/x-flux/results_infer_demo_0/frame_0000.npy \
    --control_type depth \
    --local_path ~/work/dev6/DiffusionAsShader/thirdparty/x-flux/saves_depth/checkpoint-97500/controlnet.bin \
    --use_controlnet \
    --model_type flux-dev \
    --width 480 \
    --height 320 \
    --timestep_to_start_cfg 1 \
    --num_steps 25 \
    --true_gs 3.5 \
    --guidance 3 \
    --num_images_per_prompt 10 \
    --save_path results_infer_demo_0