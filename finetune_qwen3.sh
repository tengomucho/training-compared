#!/bin/bash

# Variables for training
NUM_EPOCHS=3
BS=1
GRADIENT_ACCUMULATION_STEPS=8
LOGGING_STEPS=2
MODEL_NAME=${$1:-"Qwen/Qwen3-8B"}
OUTPUT_DIR="$(echo $MODEL_NAME | cut -d'/' -f2)-finetuned"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# For testing purposes, limit steps
MAX_STEPS=${$2:-5}  # Remove this line for full training

echo "MODEL_NAME: $MODEL_NAME"
echo "MAX_STEPS: $MAX_STEPS"

python finetune_qwen3.py \
  --model_id $MODEL_NAME \
  --num_train_epochs $NUM_EPOCHS \
  --do_train \
  --max_steps $MAX_STEPS \
  --per_device_train_batch_size $BS \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --learning_rate 8e-4 \
  --logging_steps $LOGGING_STEPS \
  --output_dir $OUTPUT_DIR \
  --lr_scheduler_type "cosine" \
  --overwrite_output_dir \
  --save_strategy "steps" \
  --save_steps 50 \
  --eval_strategy "no" \
  --warmup_steps 10 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --dataloader_num_workers 0
