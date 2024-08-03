# Experimental environment: A10
# 8GB GPU memory
CUDA_VISIBLE_DEVICES=2 \
nohup swift eval \
    --ckpt_dir /mnt/nvme1n1/phi2_output/adamw_torch/phi2-3b/v0-20240520-224619/checkpoint-319/ \
    --eval_dataset mmlu \
    --eval_limit 10 \
    --load_dataset_config true \
    --use_flash_attn false \
    --max_new_tokens 4096 \
    --temperature 0.1 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --merge_lora false \
    --truncation_strategy truncation_left > nohup_adamw_4.out 2>&1 &
