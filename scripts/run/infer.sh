#!/bin/bash
# 单张图片推理脚本

set -e

# 默认参数
MODEL_PATH="${MODEL_PATH:-./model_cache/Qwen/Qwen3-VL-2B-Instruct}"
IMAGE_PATH="${1:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
DTYPE="${DTYPE:-bfloat16}"

# 切换到项目根目录
cd "$(dirname "$0")/../.."

# 激活虚拟环境（如果存在）
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# 检查参数
if [ -z "$IMAGE_PATH" ]; then
    echo "用法: $0 <图片路径> [选项]"
    echo ""
    echo "示例:"
    echo "  $0 test.jpg"
    echo "  $0 test.jpg --output result.jpg"
    echo "  MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct $0 test.jpg"
    echo ""
    echo "环境变量:"
    echo "  MODEL_PATH  - 模型路径 (默认: ./model_cache/Qwen/Qwen3-VL-2B-Instruct)"
    echo "  OUTPUT_PATH - 输出图片路径 (可选，绘制检测框)"
    echo "  DTYPE       - 数据类型 (默认: bfloat16)"
    exit 1
fi

echo "=========================================="
echo "交通设备异常检测推理"
echo "=========================================="
echo "模型: $MODEL_PATH"
echo "图片: $IMAGE_PATH"
echo "=========================================="

# 构建命令
CMD="python scripts/inference/infer.py \
    --model $MODEL_PATH \
    --image $IMAGE_PATH \
    --dtype $DTYPE \
    --json"

if [ -n "$OUTPUT_PATH" ]; then
    CMD="$CMD --output $OUTPUT_PATH"
fi

eval $CMD
