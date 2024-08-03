#!/bin/bash
steps_values=(2)
for GA_steps in "${steps_values[@]}"; do
    echo "Running training with --agma_gradient_accumulation_steps=${GA_steps}"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup swift dpo \
        --model_type qwen1half-0_5b-chat \
        --sft_type full \
        --template_type default \
        --batch_size 8 \
        --train_dataset_sample 41600 \
        --eval_steps 100 \
        --output_dir output \
        --optim AGMA \
        --num_train_epochs 1 \
        --max_length 512 \
        --learning_rate 2e-6 \
        --weight_decay 0.01 \
        --use_flash_attn false \
        --save_only_model true \
        --dataset hh-rlhf-harmless-base \
        --agma_gradient_accumulation_steps ${GA_steps} \
        --gradient_accumulation_steps 1 \
        --gradient_checkpointing false \
        > output_pma_${GA_steps}.log 2>&1 
done
