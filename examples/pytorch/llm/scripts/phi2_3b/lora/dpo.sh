#!/bin/bash
set -e
optimizers=("AGMA" "AGMA_Lion" "adamw_torch" "lion_32bit")
for optim in "${optimizers[@]}"; do
    echo "Running training with optimizer: ${optim}"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup swift dpo \
        --model_type phi2-3b \
        --sft_type full \
        --template_type default \
        --batch_size 16 \
        --train_dataset_sample 41600 \
        --eval_steps 10 \
        --output_dir /mnt/nvme1n1/phi2_output/${optim} \
        --optim ${optim} \
        --num_train_epochs 1 \
        --max_length 512 \
        --learning_rate 2e-6 \
        --weight_decay 0.01 \
        --use_flash_attn true \
        --save_only_model true \
        --dataset hh-rlhf-harmless-base \
        --gradient_accumulation_steps 1 \
        --agma_gradient_accumulation_steps 1 \
        --dataloader_num_workers 8 \
        --gradient_checkpointing false \
        > /mnt/nvme1n1/phi2_output/${optim}_training.log 2>&1 &
done
wait
