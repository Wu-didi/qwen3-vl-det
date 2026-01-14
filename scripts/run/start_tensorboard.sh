#!/bin/bash
# 启动 TensorBoard 查看训练曲线

LOGDIR=${1:-"outputs/qwen3vl_grpo_trl"}
PORT=${PORT:-6006}

echo "=========================================="
echo "启动 TensorBoard"
echo "=========================================="
echo "日志目录: $LOGDIR"
echo "访问地址: http://localhost:$PORT"
echo ""
echo "主要指标说明:"
echo "  - reward: 平均奖励 (越高越好)"
echo "  - reward_std: 奖励标准差"
echo "  - kl: KL 散度 (过高说明模型偏离太远)"
echo "  - clip_ratio: 裁剪比率 (过高说明更新太激进)"
echo "  - rewards/format_reward: 格式正确性"
echo "  - rewards/bbox_iou_reward: 检测框 IoU"
echo "  - rewards/category_match_reward: 类别匹配"
echo "  - rewards/status_accuracy_reward: 状态准确率"
echo "=========================================="
echo "按 Ctrl+C 停止"
echo ""

tensorboard --logdir=$LOGDIR --port=$PORT --bind_all
