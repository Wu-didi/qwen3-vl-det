#!/bin/bash
# GRPO 强化学习微调脚本

set -e

#==========================================
# 常用参数 - 直接修改这里
#==========================================
CUDA_DEVICES=0                    # GPU 编号
MAX_IMAGE_SIZE=1024                # 图片最大边长 (256/384/512/768)
BATCH_SIZE=1                      # 批次大小 (GRPO建议保持1，用gradient_accumulation代替)
NUM_GENERATIONS=4                 # 每样本生成数 (2/4/6)
GRADIENT_ACCUMULATION=4           # 梯度累积 (相当于扩大batch)
LORA_R=64                         # LoRA rank (8/16/32/64)
NUM_EPOCHS=1                      # 训练轮数
LEARNING_RATE=5e-6                # 学习率 (降低，原 1e-5)

#==========================================
# 路径配置
#==========================================
MODEL_PATH="./model_cache/Qwen/Qwen3-VL-2B-Instruct"
SFT_MODEL_PATH="./outputs/qwen3vl_lora"   # SFT 微调后的模型路径 (留空则从基础模型开始)
TRAIN_DATA="data/hefei_last_dataset/qwen_data/train.json"
VAL_DATA="data/hefei_last_dataset/qwen_data/val.json"  # 验证集路径 (留空则不验证)
OUTPUT_DIR="outputs/qwen3vl_grpo"

#==========================================
# GRPO 参数 (一般不需要改)
#==========================================
TEMPERATURE=0.7
KL_COEF=0.5                       # 增大 KL 系数，防止模型偏离太远 (原 0.1)

#==========================================
# 其他参数
#==========================================
LORA_ALPHA=16
LORA_DROPOUT=0.1
MAX_LENGTH=2048
DISABLE_4BIT=false                # 设为 true 关闭 4bit (默认开启)
SAVE_STEPS=200                    # 每处理多少个样本保存一次检查点
EVAL_STEPS=200                    # 每多少步验证一次 (0 表示不验证)
LOGGING_STEPS=10

#==========================================
# 环境设置
#==========================================
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/../.."

echo "=========================================="
echo "GRPO 强化学习微调"
echo "=========================================="
echo "GPU: $CUDA_DEVICES"
echo "基础模型: $MODEL_PATH"
if [ -n "$SFT_MODEL_PATH" ]; then
    echo "SFT模型: $SFT_MODEL_PATH (在SFT基础上继续训练)"
else
    echo "SFT模型: 无 (从基础模型开始)"
fi
echo "输出: $OUTPUT_DIR"
echo "------------------------------------------"
echo "图片大小: ${MAX_IMAGE_SIZE}px"
echo "Batch Size: $BATCH_SIZE (建议保持1)"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "Num Generations: $NUM_GENERATIONS"
echo "LoRA R: $LORA_R"
echo "=========================================="

CMD="python scripts/training/grpo_finetune.py \
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
    --kl_coef $KL_COEF \
    --num_epochs $NUM_EPOCHS \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS"

if [ "$DISABLE_4BIT" = "true" ]; then
    CMD="$CMD --no_4bit"
fi

if [ -n "$SFT_MODEL_PATH" ]; then
    CMD="$CMD --sft_model_path $SFT_MODEL_PATH"
fi

if [ -n "$VAL_DATA" ] && [ -f "$VAL_DATA" ]; then
    CMD="$CMD --val_data $VAL_DATA"
fi

echo ""
eval $CMD
