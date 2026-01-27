#!/bin/bash
# GRPO 训练速度优化 - 一键应用脚本

set -e

echo "=========================================="
echo "GRPO 训练速度优化向导"
echo "=========================================="
echo ""

# 检查当前配置
echo "📋 检查当前配置..."
echo ""

CURRENT_IMAGE_SIZE=$(grep "MAX_IMAGE_SIZE=" scripts/run/train_grpo_trl.sh | head -1 | cut -d'=' -f2 | tr -d ' #')
CURRENT_MODEL=$(grep "MODEL_PATH=" scripts/run/train_grpo_trl.sh | grep -v "^#" | head -1 | cut -d'"' -f2)
CURRENT_EVAL=$(grep "EVAL_STEPS=" scripts/run/train_grpo_trl.sh | head -1 | cut -d'=' -f2 | tr -d ' #')

echo "当前配置："
echo "  图片大小: ${CURRENT_IMAGE_SIZE}px"
echo "  模型: $CURRENT_MODEL"
echo "  验证频率: EVAL_STEPS=$CURRENT_EVAL"
echo ""

# 分析速度
if [[ "$CURRENT_IMAGE_SIZE" -ge 1024 ]]; then
    echo "⚠️  图片大小过大 (${CURRENT_IMAGE_SIZE}px)"
    echo "   建议: 降低到 512px 可提升 4 倍速度"
    NEEDS_OPTIMIZATION=true
elif [[ "$CURRENT_IMAGE_SIZE" -ge 768 ]]; then
    echo "💡 图片大小较大 (${CURRENT_IMAGE_SIZE}px)"
    echo "   建议: 降低到 512px 可提升 2 倍速度"
    NEEDS_OPTIMIZATION=true
else
    echo "✅ 图片大小合理 (${CURRENT_IMAGE_SIZE}px)"
fi

if [[ "$CURRENT_MODEL" == *"8B"* ]] || [[ "$CURRENT_MODEL" == *"7B"* ]]; then
    echo "💡 使用大模型 (8B)"
    echo "   建议: 改用 2B 模型可提升 4 倍速度"
    NEEDS_OPTIMIZATION=true
else
    echo "✅ 使用小模型 (2B)"
fi

if [[ "$CURRENT_EVAL" -gt 0 ]] && [[ "$CURRENT_EVAL" -lt 500 ]]; then
    echo "💡 验证频率较高 (EVAL_STEPS=$CURRENT_EVAL)"
    echo "   建议: 设置为 0 可提升 1.5 倍速度"
    NEEDS_OPTIMIZATION=true
else
    echo "✅ 验证配置合理"
fi

echo ""
echo "=========================================="

if [ "$NEEDS_OPTIMIZATION" = true ]; then
    echo "🎯 优化建议"
    echo "=========================================="
    echo ""
    echo "我可以帮你应用以下优化："
    echo ""
    echo "  1. 快速优化（推荐）"
    echo "     - 图片大小: ${CURRENT_IMAGE_SIZE} → 512"
    echo "     - 禁用验证: EVAL_STEPS → 0"
    echo "     - 预计提升: 4-6 倍"
    echo ""
    echo "  2. 激进优化（最快）"
    echo "     - 图片大小: ${CURRENT_IMAGE_SIZE} → 384"
    echo "     - 模型: 8B → 2B"
    echo "     - 生成数量: 4 → 2"
    echo "     - 禁用验证: EVAL_STEPS → 0"
    echo "     - 预计提升: 20-40 倍"
    echo ""
    echo "  3. 手动优化"
    echo "     - 编辑配置文件: vim scripts/run/train_grpo_trl.sh"
    echo ""

    read -p "选择优化方案 (1/2/3/n=不优化): " choice

    case $choice in
        1)
            echo ""
            echo "应用快速优化..."
            sed -i.bak "s/MAX_IMAGE_SIZE=[0-9]*/MAX_IMAGE_SIZE=512/" scripts/run/train_grpo_trl.sh
            sed -i "s/EVAL_STEPS=[0-9]*/EVAL_STEPS=0/" scripts/run/train_grpo_trl.sh
            echo "✅ 优化完成！"
            echo "   备份文件: scripts/run/train_grpo_trl.sh.bak"
            ;;
        2)
            echo ""
            echo "应用激进优化..."
            sed -i.bak "s/MAX_IMAGE_SIZE=[0-9]*/MAX_IMAGE_SIZE=384/" scripts/run/train_grpo_trl.sh
            sed -i "s/NUM_GENERATIONS=[0-9]*/NUM_GENERATIONS=2/" scripts/run/train_grpo_trl.sh
            sed -i "s/EVAL_STEPS=[0-9]*/EVAL_STEPS=0/" scripts/run/train_grpo_trl.sh
            # 尝试替换模型路径（如果是 8B）
            if [[ "$CURRENT_MODEL" == *"8B"* ]]; then
                sed -i "s/8B-Instruct/2B-Instruct/" scripts/run/train_grpo_trl.sh
                echo "   已将模型从 8B 改为 2B"
            fi
            echo "✅ 优化完成！"
            echo "   备份文件: scripts/run/train_grpo_trl.sh.bak"
            ;;
        3)
            echo ""
            echo "打开配置文件进行手动编辑..."
            echo "请修改以下参数："
            echo "  - MAX_IMAGE_SIZE (推荐 512)"
            echo "  - NUM_GENERATIONS (推荐 4)"
            echo "  - MODEL_PATH (推荐 2B)"
            echo "  - EVAL_STEPS (推荐 0)"
            ;;
        *)
            echo ""
            echo "跳过优化"
            ;;
    esac
else
    echo "✅ 配置已优化"
    echo "=========================================="
fi

echo ""
echo "=========================================="
echo "📚 有用的命令"
echo "=========================================="
echo ""
echo "1. 启动训练:"
echo "   ./scripts/run/train_grpo_trl.sh"
echo ""
echo "2. 监控训练进度:"
echo "   python scripts/monitor_grpo_speed.py --log outputs/qwen3vl_grpo_trl/training_log.json"
echo ""
echo "3. 实时监控（自动刷新）:"
echo "   python scripts/monitor_grpo_speed.py --log outputs/qwen3vl_grpo_trl/training_log.json --watch"
echo ""
echo "4. 查看训练曲线:"
echo "   python scripts/visualize_training_log.py --log outputs/qwen3vl_grpo_trl/training_log.json --output plots/"
echo ""
echo "5. 查看 GPU 使用率:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "=========================================="
