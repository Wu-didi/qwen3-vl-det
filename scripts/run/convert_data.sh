#!/bin/bash
# CVAT 数据转换脚本

set -e

# 默认参数
CVAT_DIR="${CVAT_DIR:-data/annotations}"
OUTPUT_DIR="${OUTPUT_DIR:-data/qwen_data}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
ONLY_ABNORMAL="${ONLY_ABNORMAL:-false}"

# 切换到项目根目录
cd "$(dirname "$0")/../.."

# 激活虚拟环境（如果存在）
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "=========================================="
echo "CVAT 数据转换为 Qwen-VL 格式"
echo "=========================================="
echo "输入目录: $CVAT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "训练集比例: $TRAIN_RATIO"
echo "验证集比例: $VAL_RATIO"
echo "仅异常样本: $ONLY_ABNORMAL"
echo "=========================================="

# 构建命令
CMD="python scripts/data/cvat_to_qwenvl.py \
    --cvat-dir $CVAT_DIR \
    --output-dir $OUTPUT_DIR \
    --train-ratio $TRAIN_RATIO \
    --val-ratio $VAL_RATIO"

if [ "$ONLY_ABNORMAL" = "true" ]; then
    CMD="$CMD --only-abnormal"
fi

eval $CMD

echo ""
echo "转换完成！输出文件:"
ls -la $OUTPUT_DIR/
