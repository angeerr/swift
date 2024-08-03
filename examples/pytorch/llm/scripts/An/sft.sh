#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup swift sft \
    --model_type llama3-8b-instruct \
    --model_revision master \
    --sft_type lora \
    --tuner_backend peft \
    --template_type AUTO \
    --dtype AUTO \
    --output_dir output \
    --dataset blossom-math-zh \
    --train_dataset_sample -1 \
    --num_train_epochs 1 \
    --max_length 2048 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 10 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --output_dir /mnt/nvme1n1/rebuttal \
    --agma_gradient_accumulation_steps 4 \
    --gradient_accumulation_steps 1 \
    --optim AGMA \
    > /mnt/nvme1n1/phi2_output/rebuttal_AGMA_training.log 2>&1