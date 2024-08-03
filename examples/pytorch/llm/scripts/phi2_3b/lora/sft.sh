#!/bin/bash
"""
echo "Running training with optimizer: AGMA"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup swift sft \
    --model_type phi2-3b \
    --sft_type full \
    --template_type default \
    --batch_size 32 \
    --train_dataset_sample 41600 \
    --eval_steps 10 \
    --output_dir /mnt/nvme1n1/phi2_output/AGMA \
    --optim AGMA \
    --num_train_epochs 1 \
    --max_length 512 \
    --learning_rate 2e-6 \
    --weight_decay 0.01 \
    --use_flash_attn true \
    --save_only_model true \
    --dataset alpaca-en \
    --agma_gradient_accumulation_steps 4 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing false \
    > /mnt/nvme1n1/phi2_output/AGMA_training.log 2>&1

# 使用优化器 AGMA_Lion
echo "Running training with optimizer: AGMA_Lion"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup swift sft \
    --model_type phi2-3b \
    --sft_type full \
    --template_type default \
    --batch_size 32 \
    --train_dataset_sample 41600 \
    --eval_steps 10 \
    --output_dir /mnt/nvme1n1/phi2_output/AGMA_Lion \
    --optim AGMA_Lion \
    --num_train_epochs 1 \
    --max_length 512 \
    --learning_rate 2e-6 \
    --weight_decay 0.01 \
    --use_flash_attn true \
    --save_only_model true \
    --dataset alpaca-en \
    --agma_gradient_accumulation_steps 4 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing false \
    > /mnt/nvme1n1/phi2_output/AGMA_Lion_training.log 2>&1

# 使用优化器 adamw_torch
echo "Running training with optimizer: adamw_torch"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup swift sft \
    --model_type phi2-3b \
    --sft_type full \
    --template_type default \
    --batch_size 32 \
    --train_dataset_sample 41600 \
    --eval_steps 10 \
    --output_dir /mnt/nvme1n1/phi2_output/adamw_torch \
    --optim adamw_torch \
    --num_train_epochs 1 \
    --max_length 512 \
    --learning_rate 2e-6 \
    --weight_decay 0.01 \
    --use_flash_attn true \
    --save_only_model true \
    --dataset alpaca-en \
    --agma_gradient_accumulation_steps 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing false \
    > /mnt/nvme1n1/phi2_output/adamw_torch_training.log 2>&1

# 使用优化器 lion_32bit
echo "Running training with optimizer: lion_32bit"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup swift sft \
    --model_type phi2-3b \
    --sft_type full \
    --template_type default \
    --batch_size 32 \
    --train_dataset_sample 41600 \
    --eval_steps 10 \
    --output_dir /mnt/nvme1n1/phi2_output/lion_32bit \
    --optim lion_32bit \
    --num_train_epochs 1 \
    --max_length 512 \
    --learning_rate 2e-6 \
    --weight_decay 0.01 \
    --use_flash_attn true \
    --save_only_model true \
    --dataset alpaca-en \
    --agma_gradient_accumulation_steps 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing false \
    > /mnt/nvme1n1/phi2_output/lion_32bit_training.log 2>&1


echo "Running training with dpo lion_32bit"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup swift dpo \
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
    --agma_gradient_accumulation_steps 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing false 
    > /mnt/nvme1n1/phi2_output/dpo_lion_8.log 2>&1
"""
echo "Running training with dpo lion_pma"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup swift dpo \
    --model_type phi2-3b \
    --sft_type full \
    --template_type default \
    --batch_size 16 \
    --train_dataset_sample 41600 \
    --eval_steps 100 \
    --output_dir output \
    --optim AGMA_Lion \
    --num_train_epochs 1 \
    --max_length 512 \
    --learning_rate 2e-6 \
    --weight_decay 0.01 \
    --use_flash_attn true \
    --save_only_model true \
    --dataset hh-rlhf-harmless-base \
    --agma_gradient_accumulation_steps 8 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing false 
    > /mnt/nvme1n1/phi2_output/dpo_lion_pma_8.log 2>&1