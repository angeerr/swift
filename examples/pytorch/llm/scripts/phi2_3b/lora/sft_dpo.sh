#!/bin/bash
set -e

swift dpo \
    --model_type phi2-3b \
    --sft_type full \
    --template_type default \
    --batch_size 16 \
    --train_dataset_sample 41600 \
    --eval_steps 100 \
    --output_dir output \
    --optim lion_32bit \
    --num_train_epochs 1 \
    --max_length 512 \
    --learning_rate 2e-6 \
    --weight_decay 0.01 \
    --use_flash_attn true \
    --save_only_model true \
    --dataset hh-rlhf-harmless-base \
    --agma_gradient_accumulation_steps 16 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing false 

