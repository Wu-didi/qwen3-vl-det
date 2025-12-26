#!/bin/bash
# DPO 直接偏好优化微调脚本

set -e

#==========================================
# 常用参数 - 直接修改这里
#==========================================
CUDA_DEVICES=0                    # GPU 编号
MAX_IMAGE_SIZE=512                # 图片最大边长 (256/384/512/768)
BATCH_SIZE=1                      # 批次大小
GRADIENT_ACCUMULATION=8           # 梯度累积
LORA_R=64                         # LoRA rank (8/16/32/64)
NUM_EPOCHS=1                      # 训练轮数
LEARNING_RATE=1e-5                # 学习率
BETA=0.1                          # DPO 温度参数

#==========================================
# 路径配置
#==========================================
MODEL_PATH="./model_cache/Qwen/Qwen3-VL-2B-Instruct"
TRAIN_DATA="data/dpo_data/train.json"
OUTPUT_DIR="outputs/qwen3vl_dpo"

#==========================================
# 其他参数
#==========================================
LORA_ALPHA=16
LORA_DROPOUT=0.1
MAX_LENGTH=2048
USE_4BIT=true
SAVE_STEPS=500
LOGGING_STEPS=10

#==========================================
# 环境设置
#==========================================
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/../.."

echo "=========================================="
echo "DPO 直接偏好优化微调"
echo "=========================================="
echo "GPU: $CUDA_DEVICES"
echo "模型: $MODEL_PATH"
echo "输出: $OUTPUT_DIR"
echo "------------------------------------------"
echo "图片大小: ${MAX_IMAGE_SIZE}px"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "LoRA R: $LORA_R"
echo "Beta: $BETA"
echo "=========================================="

# 检查训练数据是否存在
if [ ! -f "$TRAIN_DATA" ]; then
    echo ""
    echo "错误: 训练数据不存在: $TRAIN_DATA"
    echo ""
    echo "请先生成 DPO 偏好数据:"
    echo "  python scripts/data/generate_dpo_data.py \\"
    echo "      --input data/hefei_last_dataset/qwen_data/train.json \\"
    echo "      --output data/dpo_data/train.json"
    echo ""
    exit 1
fi

CMD="python scripts/training/dpo_finetune.py \
    --model_path $MODEL_PATH \
    --train_data $TRAIN_DATA \
    --output_dir $OUTPUT_DIR \
    --max_image_size $MAX_IMAGE_SIZE \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --beta $BETA \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS"

if [ "$USE_4BIT" = "true" ]; then
    CMD="$CMD --use_4bit"
fi

echo ""
eval $CMD
