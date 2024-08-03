#!/bin/bash
CUDA_VISIBLE_DEVICES=2 \
swift dpo \
    --model_type phi2-3b \
    --sft_type full \
    --template_type default \
    --batch_size 2 \
    --train_dataset_sample 41600 \
    --eval_steps 800 \
    --output_dir output \
    --optim AGMA \
    --num_train_epochs 1 \
    --max_length 512 \
    --learning_rate 2e-6 \
    --weight_decay 0.01 \
    --use_flash_attn true \
    --save_only_model true \
    --dataset hh-rlhf-harmless-base \
    --gradient_accumulation_steps 32 \
    --gradient_checkpointing false
