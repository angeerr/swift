# Experimental environment: A10
# 8GB GPU memory
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift eval \
    --model_type phi2-3b \
    --eval_dataset mmlu \
    --eval_limit 10 \
    --use_flash_attn false \
    --max_new_tokens 1024 \
    --temperature 0.1 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --merge_lora false \
    --truncation_strategy truncation_left \
