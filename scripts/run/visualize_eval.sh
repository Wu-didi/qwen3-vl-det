#!/bin/bash
# 可视化评估结果脚本
#
# 用法:
#   ./scripts/run/visualize_eval.sh                    # 可视化单个模型结果
#   EVAL_DIR=eval_results/lora ./scripts/run/visualize_eval.sh
#   PLOT_TYPE=per_class ./scripts/run/visualize_eval.sh

set -e

# 默认参数
EVAL_DIR="${EVAL_DIR:-eval_results}"
OUTPUT="${OUTPUT:-plots/eval_plots.png}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.5}"
PLOT_TYPE="${PLOT_TYPE:-all}"

echo "=========================================="
echo "可视化评估结果"
echo "=========================================="
echo "评估目录: $EVAL_DIR"
echo "输出文件: $OUTPUT"
echo "IoU 阈值: $IOU_THRESHOLD"
echo "图表类型: $PLOT_TYPE"
echo "=========================================="

# 创建输出目录
mkdir -p "$(dirname "$OUTPUT")"

# 运行可视化
python scripts/visualize_eval_results.py \
    --eval_dir "$EVAL_DIR" \
    --output "$OUTPUT" \
    --iou_threshold "$IOU_THRESHOLD" \
    --plot_type "$PLOT_TYPE"

echo ""
echo "=========================================="
echo "可视化完成！图表已保存到: $OUTPUT"
echo "=========================================="
