#!/bin/bash

# deepseed training
echo "Starting reward model training with DeepSpeed..."

# export CUDA_VISIBLE_DEVICES=5,7

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTITED_DEBUG=DETAIL
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_OPS=0
rm -rf ~/.cache/torch_extensions


export RANK=${RANK:-0}
export LOCAL_RANK=${LOCAL_RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

# rn no deepspeed :( 
# to add back, do     # --deepspeed deepspeed_config_2.json \
python -m accelerate.commands.launch rm_training_corrected.py \
    --num_processes $RANK \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --max_length 2048 \
    --gradient_checkpointing \
    --num_train_epochs 1 \
    --learning_rate 3e-5 \
    --output_path ./models/llama3_rm_fair_deepspeed

echo "Training completed!"