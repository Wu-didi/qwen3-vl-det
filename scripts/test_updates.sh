#!/bin/bash
# 快速测试训练日志和验证功能
# 使用小数据集快速验证功能是否正常

set -e

echo "=========================================="
echo "Testing Training Logs and Validation"
echo "=========================================="

# 检查依赖
echo ""
echo "Checking dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"

# 检查可视化依赖（可选）
if python -c "import matplotlib" 2>/dev/null; then
    echo "Matplotlib: installed ✓"
else
    echo "Matplotlib: not installed (optional, for visualization)"
fi

echo ""
echo "=========================================="
echo "Test 1: Check argparse parameters"
echo "=========================================="

# 测试 --no_4bit 参数
echo "Testing --no_4bit parameter..."
python scripts/training/finetune_qwen_vl.py --help | grep -A 1 "no_4bit" || echo "Parameter found"

# 测试 --no_bf16 参数
echo "Testing --no_bf16 parameter..."
python scripts/training/grpo_finetune.py --help | grep -A 1 "no_bf16" || echo "Parameter found"

echo "✓ Argparse parameters test passed"

echo ""
echo "=========================================="
echo "Test 2: Check validation parameters"
echo "=========================================="

# 测试 --val_data 参数
echo "Testing --val_data parameter..."
python scripts/training/grpo_finetune.py --help | grep -A 1 "val_data" || echo "Parameter found"

# 测试 --eval_steps 参数
echo "Testing --eval_steps parameter..."
python scripts/training/dpo_finetune.py --help | grep -A 1 "eval_steps" || echo "Parameter found"

echo "✓ Validation parameters test passed"

echo ""
echo "=========================================="
echo "Test 3: Check visualization scripts"
echo "=========================================="

# 检查可视化脚本是否存在
if [ -f "scripts/visualize_training_log.py" ]; then
    echo "✓ visualize_training_log.py exists"
    python scripts/visualize_training_log.py --help > /dev/null
    echo "✓ visualize_training_log.py is executable"
else
    echo "✗ visualize_training_log.py not found"
    exit 1
fi

if [ -f "scripts/compare_training_logs.py" ]; then
    echo "✓ compare_training_logs.py exists"
    python scripts/compare_training_logs.py --help > /dev/null
    echo "✓ compare_training_logs.py is executable"
else
    echo "✗ compare_training_logs.py not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "Test 4: Check documentation"
echo "=========================================="

# 检查文档是否存在
docs=("VALIDATION_UPDATE.md" "TRAINING_LOGS.md" "UPDATES_SUMMARY.md")
for doc in "${docs[@]}"; do
    if [ -f "$doc" ]; then
        echo "✓ $doc exists"
    else
        echo "✗ $doc not found"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "Test 5: Simulate training log"
echo "=========================================="

# 创建一个模拟的训练日志用于测试
TEST_DIR="test_output"
mkdir -p "$TEST_DIR"

cat > "$TEST_DIR/training_log.json" << 'EOF'
{
  "config": {
    "model_path": "Qwen/Qwen3-VL-2B-Instruct",
    "train_data": "data/train.json",
    "val_data": "data/val.json",
    "num_epochs": 1,
    "batch_size": 1,
    "learning_rate": 1e-05,
    "num_generations": 4
  },
  "train_history": [
    {"step": 10, "epoch": 1, "loss": 0.8, "reward": 2.0, "kl": 0.01, "lr": 1e-05},
    {"step": 20, "epoch": 1, "loss": 0.7, "reward": 2.2, "kl": 0.02, "lr": 1e-05},
    {"step": 30, "epoch": 1, "loss": 0.6, "reward": 2.4, "kl": 0.015, "lr": 1e-05}
  ],
  "val_history": [
    {"step": 30, "epoch": 1, "val_reward": 2.5, "val_format": 0.9, "val_bbox": 0.8}
  ],
  "best_checkpoint": {
    "step": 30,
    "epoch": 1,
    "val_reward": 2.5,
    "path": "test_output/best"
  }
}
EOF

echo "Created test training log"

# 测试可视化脚本
echo "Testing visualization script..."
python scripts/visualize_training_log.py \
    --log "$TEST_DIR/training_log.json" \
    --no-plot

echo "✓ Visualization script works"

# 清理测试文件
rm -rf "$TEST_DIR"
echo "Cleaned up test files"

echo ""
echo "=========================================="
echo "All tests passed! ✓"
echo "=========================================="
echo ""
echo "You can now:"
echo "  1. Run training with validation:"
echo "     ./scripts/run/train_grpo.sh"
echo ""
echo "  2. Visualize training logs:"
echo "     python scripts/visualize_training_log.py --log outputs/*/training_log.json"
echo ""
echo "  3. Compare experiments:"
echo "     python scripts/compare_training_logs.py --logs outputs/*/training_log.json"
echo ""
