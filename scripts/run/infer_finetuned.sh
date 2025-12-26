#!/bin/bash
# 使用微调模型进行推理

set -e

# 默认参数
MODEL_PATH="${MODEL_PATH:-outputs/qwen3vl_lora}"
BASE_MODEL="${BASE_MODEL:-}"
IMAGE_PATH="${1:-}"

# 切换到项目根目录
cd "$(dirname "$0")/../.."

# 激活虚拟环境（如果存在）
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# 检查参数
if [ -z "$IMAGE_PATH" ]; then
    echo "用法: $0 <图片路径>"
    echo ""
    echo "示例:"
    echo "  $0 test.jpg"
    echo "  MODEL_PATH=outputs/qwen3vl_grpo $0 test.jpg"
    echo ""
    echo "环境变量:"
    echo "  MODEL_PATH  - 微调模型路径 (默认: outputs/qwen3vl_lora)"
    echo "  BASE_MODEL  - 基础模型路径 (可选，自动从配置读取)"
    exit 1
fi

echo "=========================================="
echo "微调模型推理"
echo "=========================================="
echo "微调模型: $MODEL_PATH"
echo "图片: $IMAGE_PATH"
echo "=========================================="

# 构建命令
CMD="python scripts/inference/inference_finetuned.py \
    --model_path $MODEL_PATH \
    --image_path $IMAGE_PATH"

if [ -n "$BASE_MODEL" ]; then
    CMD="$CMD --base_model_path $BASE_MODEL"
fi

eval $CMD
