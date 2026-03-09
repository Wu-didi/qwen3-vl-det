export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="outputs/qwen3vl2b_lora/checkpoint-2500"

# 自动用模型路径生成输出目录，避免覆盖
OUTPUT_DIR="eval_results/$(echo $MODEL_PATH | tr '/' '_')"

# 基础模型评估
# MODEL_PATH="./model_cache/Qwen/Qwen3-VL-2B-Instruct"

python scripts/evaluate.py  --model_path $MODEL_PATH \
                            --test_data data/hefei_last_dataset/sft_output/test.jsonl \
                            --output_dir $OUTPUT_DIR --coco_map
