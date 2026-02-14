#!/bin/bash
# GRPO 强化学习微调脚本 (TRL 版本)
# 基于 TRL 的 GRPOTrainer，更稳定的训练过程

set -e

#==========================================
# 常用参数 - 直接修改这里
#==========================================
CUDA_DEVICES=7                    # GPU 编号
MAX_IMAGE_SIZE=512                 # 图片最大边长 (推荐 384-512，1024 太大会很慢！)
BATCH_SIZE=1                      # 批次大小
NUM_GENERATIONS=4                 # 每样本生成数 (可改为 2 加速，但效果可能略差)
GRADIENT_ACCUMULATION=4           # 梯度累积
LORA_R=64                         # LoRA rank (可改为 32 加速)
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

# 奖励函数方案: risk_aware(推荐, 可写论文) | legacy(旧版)
REWARD_SCHEME="${REWARD_SCHEME:-risk_aware}"
REWARD_MATCH_IOU="${REWARD_MATCH_IOU:-0.5}"
REWARD_HALLUCINATION_UNIT_PENALTY="${REWARD_HALLUCINATION_UNIT_PENALTY:-0.35}"
REWARD_NO_DET_MISSING_PENALTY="${REWARD_NO_DET_MISSING_PENALTY:-0.2}"
REWARD_OMISSION_PENALTY="${REWARD_OMISSION_PENALTY:-1.0}"
# 风险感知奖励权重 (仅 REWARD_SCHEME=risk_aware 生效)
REWARD_W_FORMAT="${REWARD_W_FORMAT:-0.2}"
REWARD_W_SET_F1="${REWARD_W_SET_F1:-3.0}"
REWARD_W_IOU="${REWARD_W_IOU:-2.0}"
REWARD_W_COUNT="${REWARD_W_COUNT:-1.2}"
REWARD_W_RISK="${REWARD_W_RISK:-2.5}"
REWARD_W_ANOMALY="${REWARD_W_ANOMALY:-2.0}"

#==========================================
# 路径配置
#==========================================
MODEL_PATH="./model_cache/Qwen/Qwen3-VL-8B-Instruct"
SFT_MODEL_PATH="/mnt/home/wudidi/code_v5/qwen3-vl-det/outputs/qwen3vl_lora_exp3"   # SFT 模型路径 (留空从基础模型开始)
TRAIN_DATA="data/hefei_last_dataset/qwen_data/train.json"
VAL_DATA="data/hefei_last_dataset/qwen_data/val.json"  # 验证集路径 (留空则不验证)
OUTPUT_DIR="outputs/qwen3vl_grpo_trl"

#==========================================
# 其他参数
#==========================================
LORA_ALPHA=16
LORA_DROPOUT=0.1
MAX_COMPLETION_LENGTH=512
MAX_PROMPT_LENGTH=1024
SAVE_STEPS=200
EVAL_STEPS=0                       # 每多少步验证一次 (设为 0 禁用验证以加速训练)
LOGGING_STEPS=10

# 量化和精度选项
DISABLE_4BIT=false                # 设为 true 关闭 4bit (默认开启)
DISABLE_BF16=false                # 设为 true 关闭 bf16 (默认开启)

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
echo "Reward scheme: $REWARD_SCHEME"
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
    --reward_scheme $REWARD_SCHEME \
    --reward_match_iou $REWARD_MATCH_IOU \
    --reward_hallucination_unit_penalty $REWARD_HALLUCINATION_UNIT_PENALTY \
    --reward_no_detection_missing_penalty $REWARD_NO_DET_MISSING_PENALTY \
    --reward_omission_penalty $REWARD_OMISSION_PENALTY \
    --reward_w_format $REWARD_W_FORMAT \
    --reward_w_set_f1 $REWARD_W_SET_F1 \
    --reward_w_iou $REWARD_W_IOU \
    --reward_w_count $REWARD_W_COUNT \
    --reward_w_risk $REWARD_W_RISK \
    --reward_w_anomaly $REWARD_W_ANOMALY \
    --num_epochs $NUM_EPOCHS \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS"

# 量化和精度选项 (默认都开启)
if [ "$DISABLE_4BIT" = "true" ]; then
    CMD="$CMD --no_4bit"
fi

if [ "$DISABLE_BF16" = "true" ]; then
    CMD="$CMD --no_bf16"
fi

if [ -n "$SFT_MODEL_PATH" ]; then
    CMD="$CMD --sft_model_path $SFT_MODEL_PATH"
fi

if [ -n "$VAL_DATA" ] && [ -f "$VAL_DATA" ]; then
    CMD="$CMD --val_data $VAL_DATA"
fi

if [ "$USE_WANDB" = "true" ]; then
    CMD="$CMD --use_wandb --wandb_project $WANDB_PROJECT"
fi

echo ""
echo "训练完成后，可以使用以下命令查看训练曲线："
echo "  tensorboard --logdir $OUTPUT_DIR"
echo "  或"
echo "  python scripts/visualize_training_log.py --log $OUTPUT_DIR/training_log.json --output $OUTPUT_DIR/plots"
echo ""
eval $CMD
