#!/bin/bash
# GRPO 强化学习微调脚本 (TRL 版本)
# 基于 TRL 的 GRPOTrainer，更稳定的训练过程

set -e

#==========================================
# 常用参数 - 直接修改这里
#==========================================
CUDA_DEVICES=0                    # GPU 编号
MAX_IMAGE_SIZE=1024                # 图片最大边长
BATCH_SIZE=1                      # 批次大小
NUM_GENERATIONS=4                 # 每样本生成数
GRADIENT_ACCUMULATION=4           # 梯度累积
LORA_R=64                         # LoRA rank
NUM_EPOCHS=1                      # 训练轮数
LEARNING_RATE=5e-6                # 学习率 (GRPO 建议较低)

#==========================================
# GRPO 参数 (关键参数)
#==========================================
TEMPERATURE=0.7                   # 生成温度
BETA=0.5                          # KL 惩罚系数 (防止模型偏离太远)
                                  # 0.1: 轻度约束，允许较大变化
                                  # 0.5: 中度约束，推荐起始值
                                  # 1.0: 强约束，保守更新

#==========================================
# 路径配置
#==========================================
MODEL_PATH="./model_cache/Qwen/Qwen3-VL-2B-Instruct"
SFT_MODEL_PATH="./outputs/qwen3vl_lora"   # SFT 模型路径 (留空从基础模型开始)
TRAIN_DATA="data/hefei_last_dataset/qwen_data/train.json"
OUTPUT_DIR="outputs/qwen3vl_grpo_trl"

#==========================================
# 其他参数
#==========================================
LORA_ALPHA=16
LORA_DROPOUT=0.1
MAX_COMPLETION_LENGTH=512
MAX_PROMPT_LENGTH=1024
SAVE_STEPS=200
LOGGING_STEPS=10

# 日志选项
USE_WANDB=false                   # 是否使用 wandb (需要 pip install wandb)
WANDB_PROJECT="qwen-vl-grpo"      # wandb 项目名

#==========================================
# 环境设置
#==========================================
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/../.."

echo "=========================================="
echo "GRPO 强化学习微调 (TRL 版本)"
echo "=========================================="
echo "GPU: $CUDA_DEVICES"
echo "基础模型: $MODEL_PATH"
if [ -n "$SFT_MODEL_PATH" ] && [ -d "$SFT_MODEL_PATH" ]; then
    echo "SFT模型: $SFT_MODEL_PATH (在SFT基础上继续训练)"
else
    echo "SFT模型: 无 (从基础模型开始)"
    SFT_MODEL_PATH=""
fi
echo "输出: $OUTPUT_DIR"
echo "------------------------------------------"
echo "图片大小: ${MAX_IMAGE_SIZE}px"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "Num Generations: $NUM_GENERATIONS"
echo "Learning Rate: $LEARNING_RATE"
echo "Beta (KL coef): $BETA"
echo "LoRA R: $LORA_R"
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
    --logging_steps $LOGGING_STEPS \
    --use_4bit \
    --bf16"

if [ -n "$SFT_MODEL_PATH" ]; then
    CMD="$CMD --sft_model_path $SFT_MODEL_PATH"
fi

if [ "$USE_WANDB" = "true" ]; then
    CMD="$CMD --use_wandb --wandb_project $WANDB_PROJECT"
fi

echo ""
echo "训练完成后，可以使用以下命令查看训练曲线："
echo "  tensorboard --logdir $OUTPUT_DIR"
echo "  或"
echo "  python scripts/visualize_training.py --logdir $OUTPUT_DIR --export"
echo ""
eval $CMD
