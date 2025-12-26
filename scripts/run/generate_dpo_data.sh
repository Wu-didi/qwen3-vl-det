#!/bin/bash
# 生成 DPO 偏好数据

set -e

#==========================================
# 参数配置
#==========================================
INPUT_DATA="data/hefei_last_dataset/qwen_data/train.json"
OUTPUT_DATA="data/dpo_data/train.json"
METHOD="perturb"  # perturb (快速) 或 generate (使用模型)

# 如果使用 generate 方法，需要指定模型路径
MODEL_PATH="./model_cache/Qwen/Qwen3-VL-2B-Instruct"

# 最大样本数 (可选，用于测试)
# MAX_SAMPLES=1000

#==========================================
cd "$(dirname "$0")/../.."

echo "=========================================="
echo "生成 DPO 偏好数据"
echo "=========================================="
echo "输入: $INPUT_DATA"
echo "输出: $OUTPUT_DATA"
echo "方法: $METHOD"
echo "=========================================="

CMD="python scripts/data/generate_dpo_data.py \
    --input $INPUT_DATA \
    --output $OUTPUT_DATA \
    --method $METHOD"

if [ "$METHOD" = "generate" ]; then
    CMD="$CMD --model_path $MODEL_PATH"
fi

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

echo ""
eval $CMD

echo ""
echo "=========================================="
echo "完成！现在可以运行 DPO 训练:"
echo "  bash scripts/run/train_dpo.sh"
echo "=========================================="
