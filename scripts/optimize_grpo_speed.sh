#!/bin/bash
# GRPO 训练速度优化配置
# 针对不同场景的推荐配置

echo "=========================================="
echo "GRPO 训练速度优化建议"
echo "=========================================="

cat << 'EOF'

## 🐌 为什么 GRPO 训练慢？

1. **在线生成响应**：每个样本需要生成多个响应（默认 4 个）
2. **多次前向传播**：
   - 生成 4 个响应：4 次生成
   - 计算 policy log probs：4 次前向
   - 计算 reference log probs：4 次前向
   - 总共约 12 次前向传播 vs 监督学习的 1 次

3. **图片 token 数量**：
   - Qwen-VL 图片 token 与图片大小成平方关系
   - 1024px 图片 ≈ 4000+ tokens
   - 512px 图片 ≈ 1000 tokens
   - 384px 图片 ≈ 600 tokens

## ⚡ 优化配置（按场景）

### 场景 1: 快速实验/调试（最快）
MAX_IMAGE_SIZE=384
NUM_GENERATIONS=2
LORA_R=32
EVAL_STEPS=0
MODEL_PATH="Qwen3-VL-2B-Instruct"

预计速度：~30-60 秒/样本（2B 模型）
适用：快速验证想法、调试代码

### 场景 2: 平衡速度和效果（推荐）
MAX_IMAGE_SIZE=512
NUM_GENERATIONS=4
LORA_R=64
EVAL_STEPS=0
MODEL_PATH="Qwen3-VL-2B-Instruct"

预计速度：~60-120 秒/样本（2B 模型）
适用：正常训练

### 场景 3: 追求最佳效果（慢）
MAX_IMAGE_SIZE=768
NUM_GENERATIONS=6
LORA_R=64
EVAL_STEPS=200
MODEL_PATH="Qwen3-VL-8B-Instruct"

预计速度：~300-600 秒/样本（8B 模型）
适用：最终模型训练

### 场景 4: 你当前的配置（非常慢）
MAX_IMAGE_SIZE=1024  # ⚠️ 太大！
NUM_GENERATIONS=4
LORA_R=64
EVAL_STEPS=200
MODEL_PATH="Qwen3-VL-8B-Instruct"

预计速度：~600-1200 秒/样本（8B 模型）
问题：图片太大 + 8B 模型 + 频繁验证

## 📊 速度对比（相对于基准）

配置项                    | 速度影响 | 建议
-------------------------|---------|-----
MAX_IMAGE_SIZE=384       | 1x      | 快速实验
MAX_IMAGE_SIZE=512       | 2x      | 推荐
MAX_IMAGE_SIZE=768       | 4x      | 高质量
MAX_IMAGE_SIZE=1024      | 8x      | ⚠️ 太慢
NUM_GENERATIONS=2        | 1x      | 最快
NUM_GENERATIONS=4        | 2x      | 推荐
NUM_GENERATIONS=6        | 3x      | 高质量
2B 模型                  | 1x      | 推荐
8B 模型                  | 4x      | 高质量
LORA_R=32               | 1x      | 快速
LORA_R=64               | 1.2x    | 推荐
EVAL_STEPS=0            | 1x      | 训练时禁用
EVAL_STEPS=200          | 1.5x    | 需要验证时

## 🎯 立即优化建议

### 1. 降低图片大小（最重要！）
sed -i 's/MAX_IMAGE_SIZE=1024/MAX_IMAGE_SIZE=512/' scripts/run/train_grpo_trl.sh

### 2. 禁用训练时验证
sed -i 's/EVAL_STEPS=200/EVAL_STEPS=0/' scripts/run/train_grpo_trl.sh

### 3. 减少生成数量（可选）
sed -i 's/NUM_GENERATIONS=4/NUM_GENERATIONS=2/' scripts/run/train_grpo_trl.sh

### 4. 使用 2B 模型测试（可选）
sed -i 's/Qwen3-VL-8B-Instruct/Qwen3-VL-2B-Instruct/' scripts/run/train_grpo_trl.sh

## 💡 其他优化技巧

### 1. 使用更小的 LoRA rank
LORA_R=32  # 从 64 降到 32，速度提升 ~20%

### 2. 减少 max_completion_length
MAX_COMPLETION_LENGTH=256  # 从 512 降到 256

### 3. 增大 gradient_accumulation
GRADIENT_ACCUMULATION=8  # 从 4 增到 8，减少优化步骤

### 4. 使用更少的训练数据
# 先用 10% 数据快速验证效果
head -n 100 train.json > train_small.json

## 📈 预期训练时间估算

假设有 1000 个训练样本：

配置                          | 每样本时间 | 总时间（1 epoch）
-----------------------------|-----------|------------------
当前配置（1024px + 8B）       | ~10 分钟  | ~7 天
优化后（512px + 8B）          | ~2 分钟   | ~1.5 天
推荐配置（512px + 2B）        | ~1 分钟   | ~17 小时
快速配置（384px + 2B）        | ~30 秒    | ~8 小时

## 🔍 监控训练速度

### 查看当前速度
tail -f outputs/qwen3vl_grpo_trl/training_log.json | grep step

### 计算平均速度
python << 'PYTHON'
import json
import time

log_file = "outputs/qwen3vl_grpo_trl/training_log.json"
try:
    with open(log_file) as f:
        log = json.load(f)

    history = log.get("train_history", [])
    if len(history) >= 2:
        # 估算每步时间
        steps = [h["step"] for h in history]
        # 假设每步处理 1 个样本（batch_size=1, gradient_accumulation=4）
        samples_per_step = 4  # gradient_accumulation

        print(f"已完成 {len(history)} 个记录点")
        print(f"最新 step: {history[-1]['step']}")
        print(f"估算：每 {samples_per_step} 个样本记录一次")
except FileNotFoundError:
    print("训练日志文件不存在，训练可能还未开始")
PYTHON

## ⚠️ 注意事项

1. **图片大小 vs 检测精度**
   - 512px 对大多数场景足够
   - 只有需要检测小目标时才用 768px+
   - 1024px 通常没必要，性价比低

2. **NUM_GENERATIONS vs 训练效果**
   - 2: 最快，但可能不稳定
   - 4: 推荐，平衡速度和效果
   - 6+: 更稳定，但收益递减

3. **验证频率**
   - 训练时可以禁用验证（EVAL_STEPS=0）
   - 训练完成后单独运行验证
   - 或者设置更大的 EVAL_STEPS（如 500）

4. **模型选择**
   - 先用 2B 模型验证效果
   - 确认有效后再用 8B 模型
   - 8B 模型效果提升通常 < 20%，但时间增加 4 倍

## 🚀 快速应用优化

# 备份当前配置
cp scripts/run/train_grpo_trl.sh scripts/run/train_grpo_trl.sh.backup

# 应用推荐配置
cat > scripts/run/train_grpo_trl_fast.sh << 'SCRIPT'
#!/bin/bash
# GRPO 快速训练配置

set -e

CUDA_DEVICES=7
MAX_IMAGE_SIZE=512                 # 从 1024 降到 512（速度提升 4 倍）
BATCH_SIZE=1
NUM_GENERATIONS=4                  # 保持 4 个生成
GRADIENT_ACCUMULATION=4
LORA_R=64
NUM_EPOCHS=1
LEARNING_RATE=5e-6

TEMPERATURE=0.7
BETA=0.5

MODEL_PATH="./model_cache/Qwen/Qwen3-VL-8B-Instruct"
SFT_MODEL_PATH="/mnt/home/wudidi/code_v5/qwen3-vl-det/outputs/qwen3vl_lora_exp3"
TRAIN_DATA="data/hefei_last_dataset/qwen_data/train.json"
VAL_DATA=""  # 训练时禁用验证
OUTPUT_DIR="outputs/qwen3vl_grpo_trl_fast"

LORA_ALPHA=16
LORA_DROPOUT=0.1
MAX_COMPLETION_LENGTH=512
MAX_PROMPT_LENGTH=1024
SAVE_STEPS=200
EVAL_STEPS=0                       # 禁用验证以加速
LOGGING_STEPS=10

DISABLE_4BIT=false
DISABLE_BF16=false

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/../.."

echo "=========================================="
echo "GRPO 快速训练配置"
echo "=========================================="
echo "优化项："
echo "  - 图片大小: 1024 -> 512 (速度提升 4x)"
echo "  - 禁用验证: EVAL_STEPS=0"
echo "  - 预计速度: ~2 分钟/样本 (8B 模型)"
echo "=========================================="

CMD="python scripts/training/grpo_finetune_trl.py \
    --model_path $MODEL_PATH \
    --train_data $TRAIN_DATA \
    --output_dir $OUTPUT_DIR \
    --max_image_size $MAX_IMAGE_SIZE \
    --batch_size $BATCH_SIZE \
    --num_generations $NUM_GENERATIONS \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --temperature $TEMPERATURE \
    --beta $BETA \
    --num_epochs $NUM_EPOCHS \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS"

if [ "$DISABLE_4BIT" = "true" ]; then
    CMD="$CMD --no_4bit"
fi

if [ "$DISABLE_BF16" = "true" ]; then
    CMD="$CMD --no_bf16"
fi

if [ -n "$SFT_MODEL_PATH" ]; then
    CMD="$CMD --sft_model_path $SFT_MODEL_PATH"
fi

eval $CMD
SCRIPT

chmod +x scripts/run/train_grpo_trl_fast.sh

echo ""
echo "✅ 已创建优化配置: scripts/run/train_grpo_trl_fast.sh"
echo ""
echo "使用方法："
echo "  ./scripts/run/train_grpo_trl_fast.sh"
echo ""

EOF
