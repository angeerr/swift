#!/bin/bash
set -e

nohup swift pretrain \
    --model_type gpt2-medium \
    --sft_type full \
    --template_type default \
    --batch_size 64 \
    --eval_steps 10 \
    --output_dir output \
    --optim adamw_torch \
    --num_train_epochs 1 \
    --max_length 512 \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --adam_beta1 0.9\
    --adam_beta2 0.95\
    --use_flash_attn true \
    --save_only_model true \
    --dataset wikipedia \
    --gradient_accumulation_steps 1 \
    --agma_gradient_accumulation_steps 32 \
    --dataloader_num_workers 8 \
    --gradient_checkpointing false 
    