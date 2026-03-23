#!/bin/bash
# GRPO 强化学习微调脚本 (TRL 版本)
# 基于 TRL 的 GRPOTrainer，更稳定的训练过程

set -e

#==========================================
# 常用参数 - 直接修改这里
#==========================================
CUDA_DEVICES=6
MAX_IMAGE_SIZE=1024
BATCH_SIZE=2
NUM_GENERATIONS=4
GRADIENT_ACCUMULATION=4
LORA_R=64
NUM_EPOCHS=1
LEARNING_RATE=5e-6

#==========================================
# GRPO 参数 (关键参数)
#==========================================
TEMPERATURE=0.7
BETA=0.0                   # 关键改动: 先关 KL，避免往 base model 拉回去
REF_MODEL_MODE="auto"
REF_USE_4BIT=true

# 关键改动: 改成更贴近 AP50 的简单奖励
REWARD_SCHEME="simple"
REWARD_MATCH_IOU=0.5
REWARD_HALLUCINATION_UNIT_PENALTY=0.35
REWARD_NO_DET_MISSING_PENALTY=0.2
REWARD_OMISSION_PENALTY=1.0
REWARD_W_FORMAT=0.2
REWARD_W_SET_F1=3.0
REWARD_W_IOU=2.0
REWARD_W_COUNT=1.5
REWARD_W_RISK=2.5
REWARD_W_ANOMALY=2.0
REWARD_W_RECALL=2.0
REWARD_W_COMPLETENESS=1.5

#==========================================
# 路径配置
#==========================================
MODEL_PATH="./model_cache/Qwen/Qwen3-VL-4B-Instruct"
SFT_MODEL_PATH="outputs/qwen3vl4b_lora"
OUTPUT_DIR="outputs/qwen3vl4b_grpo_trl_exp6"
LOG_DIR=""
DATA_FORMAT="auto"
TRAIN_DATA="data/hefei_last_dataset/rft_output_aligned/train.jsonl"
VAL_DATA="data/hefei_last_dataset/rft_output_aligned/val.jsonl"

#==========================================
# 其他参数
#==========================================
LORA_ALPHA=16
LORA_DROPOUT=0.1
MAX_COMPLETION_LENGTH=512
MAX_PROMPT_LENGTH=1024
SAVE_STEPS=200
EVAL_STEPS=0
LOGGING_STEPS=10

# 量化和精度选项
DISABLE_4BIT=true
DISABLE_BF16=false

# 日志选项
USE_WANDB=false
WANDB_PROJECT="qwen-vl-grpo"

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
echo "训练数据: ${TRAIN_DATA:-<missing>}"
echo "验证数据: ${VAL_DATA:-<none>}"
echo "输出: $OUTPUT_DIR"
echo "日志: ${LOG_DIR:-logs/$(basename "$OUTPUT_DIR")}"
echo "------------------------------------------"
echo "图片大小: ${MAX_IMAGE_SIZE}px"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "Num Generations: $NUM_GENERATIONS"
echo "Learning Rate: $LEARNING_RATE"
echo "Beta (KL coef): $BETA"
echo "Reference model mode: $REF_MODEL_MODE"
echo "LoRA R: $LORA_R"
echo "Reward scheme: $REWARD_SCHEME"
echo "Data format: $DATA_FORMAT"
echo "=========================================="

CMD="python scripts/training/rft/grpo_finetune_trl.py \
    --model_path $MODEL_PATH \
    --train_data $TRAIN_DATA \
    --data_format $DATA_FORMAT \
    --output_dir $OUTPUT_DIR \
    --max_image_size $MAX_IMAGE_SIZE \
    --batch_size $BATCH_SIZE \
    --num_generations $NUM_GENERATIONS \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --temperature $TEMPERATURE \
    --beta $BETA \
    --ref_model_mode $REF_MODEL_MODE \
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
    --reward_w_recall $REWARD_W_RECALL \
    --reward_w_completeness $REWARD_W_COMPLETENESS \
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

if [ "$REF_USE_4BIT" = "false" ]; then
    CMD="$CMD --no_ref_4bit"
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

if [ -n "$LOG_DIR" ]; then
    CMD="$CMD --log_dir $LOG_DIR"
fi

_effective_log_dir="${LOG_DIR:-logs/$(basename "$OUTPUT_DIR")}"
echo ""
echo "训练完成后，可以使用以下命令查看训练曲线："
echo "  tensorboard --logdir ${_effective_log_dir}/runs"
echo "  或"
echo "  python scripts/visualize_training_log.py --log ${_effective_log_dir}/training_log.json --output ${_effective_log_dir}/plots"
echo ""
eval $CMD
