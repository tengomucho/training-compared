#!/bin/bash

# Variables for training
NUM_EPOCHS=3
BS=1
GRADIENT_ACCUMULATION_STEPS=8
LOGGING_STEPS=2
MODEL_NAME=${1:-"Qwen/Qwen3-8B"}
OUTPUT_DIR="$(echo $MODEL_NAME | cut -d'/' -f2)-finetuned"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# For testing purposes, limit steps
MAX_STEPS=${2:-5}  # Remove this line for full training

# Distributed training configuration
NUM_PROCESSES=4  # Number of GPU processes to spawn
MASTER_PORT=29500  # Port for process communication (change if port is in use)

echo "MODEL_NAME: $MODEL_NAME"
echo "MAX_STEPS: $MAX_STEPS"
echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "Starting distributed training with accelerate launch and ZeRO-1 optimization..."

# Launch distributed training with accelerate launch using ZeRO-1 config
# --config_file: Use the accelerate config file for ZeRO-1 optimization
accelerate launch \
  --config_file accelerate_config.yaml \
  finetune_qwen3.py \
  --model_id $MODEL_NAME \
  --num_train_epochs $NUM_EPOCHS \
  --do_train \
  --max_steps $MAX_STEPS \
  --per_device_train_batch_size $BS \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --learning_rate 8e-4 \
  --bf16 \
  --logging_steps $LOGGING_STEPS \
  --output_dir $OUTPUT_DIR \
  --lr_scheduler_type "cosine" \
  --overwrite_output_dir