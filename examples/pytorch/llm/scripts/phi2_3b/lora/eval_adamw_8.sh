# Experimental environment: A10
# 8GB GPU memory
CUDA_VISIBLE_DEVICES=3 \
swift eval \
    --ckpt_dir /home/nlp_class/swift/examples/pytorch/llm/scripts/phi2_3b/lora/output/phi2-3b/sft_adamw_8/checkpoint-159 \
    --eval_dataset mmlu \
    --eval_limit 10 \
    --load_dataset_config true \
    --use_flash_attn false \
    --max_new_tokens 4096 \
    --temperature 0.1 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --merge_lora false \
    --truncation_strategy truncation_left > nohup_adamw_8.out 2>&1 &
