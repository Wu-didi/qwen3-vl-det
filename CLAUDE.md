# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Traffic equipment anomaly detection service using Qwen3-VL/Qwen2.5-VL vision-language models. Provides REST API and Gradio web interface for detecting anomalies in traffic infrastructure images.

## Commands

### Shell 脚本 (推荐)
```bash
# 启动 API 服务
./scripts/run/start_api.sh
PORT=8080 ./scripts/run/start_api.sh           # 自定义端口
RELOAD=true ./scripts/run/start_api.sh         # 开发模式

# 启动 Gradio Web UI
./scripts/run/start_gradio.sh
SHARE=true ./scripts/run/start_gradio.sh       # 公开分享

# 下载模型
./scripts/run/download_model.sh
MODEL=Qwen/Qwen3-VL-8B-Instruct ./scripts/run/download_model.sh

# 单张图片推理
./scripts/run/infer.sh test.jpg
OUTPUT_PATH=result.jpg ./scripts/run/infer.sh test.jpg

# 微调模型推理
./scripts/run/infer_finetuned.sh test.jpg
MODEL_PATH=outputs/qwen3vl_grpo ./scripts/run/infer_finetuned.sh test.jpg

# 数据转换 (CVAT -> Qwen-VL)
CVAT_DIR=data/annotations OUTPUT_DIR=data/qwen_data ./scripts/run/convert_data.sh

# LoRA 监督微调
./scripts/run/train_lora.sh
TRAIN_DATA=data/qwen_data/train.json NUM_EPOCHS=5 ./scripts/run/train_lora.sh

# GRPO 强化学习微调
./scripts/run/train_grpo.sh
TRAIN_DATA=data/qwen_data/train.json NUM_GENERATIONS=6 ./scripts/run/train_grpo.sh

# 评估模型
./scripts/run/evaluate.sh
MODEL_PATH=outputs/qwen3vl_lora TEST_DATA=data/qwen_data/test.json ./scripts/run/evaluate.sh
COMPUTE_MAP=true ./scripts/run/evaluate.sh  # 计算 mAP

# 可视化评估结果
./scripts/run/visualize_eval.sh
EVAL_DIR=eval_results/lora ./scripts/run/visualize_eval.sh

# 比较多个模型
./scripts/run/compare_models.sh
EVAL_DIRS="eval_results/base eval_results/lora" LABELS="Base LoRA" ./scripts/run/compare_models.sh
```

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Download model (first time setup)
python scripts/download_model.py --model Qwen/Qwen3-VL-2B-Instruct

# Start REST API server with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start Gradio web interface
python gradio_app.py  # http://localhost:7860

# Run benchmark
python scripts/benchmark.py --image-dir ./examples/sample_images
```

### Fine-tuning
```bash
# Install fine-tuning dependencies
pip install -r requirements_finetune.txt

# Convert CVAT annotations to Qwen-VL format
python scripts/data/cvat_to_qwenvl.py \
    --cvat-dir data/annotations \
    --output-dir data/qwen_data

# Run LoRA fine-tuning (QLoRA by default)
python scripts/training/finetune_qwen_vl.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --train_data data/qwen_data/train.json \
    --val_data data/qwen_data/val.json \
    --output_dir outputs/qwen3vl_lora

# Run GRPO (reinforcement learning) fine-tuning
python scripts/training/grpo_finetune.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --train_data data/qwen_data/train.json \
    --val_data data/qwen_data/val.json \
    --output_dir outputs/qwen3vl_grpo \
    --num_generations 4 \
    --kl_coef 0.1 \
    --eval_steps 200

# Run DPO (direct preference optimization) fine-tuning
python scripts/training/dpo_finetune.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --train_data data/dpo_data/train.json \
    --val_data data/dpo_data/val.json \
    --output_dir outputs/qwen3vl_dpo \
    --beta 0.1 \
    --eval_steps 500

# Inference with fine-tuned model
python scripts/inference/inference_finetuned.py \
    --model_path outputs/qwen3vl_lora \
    --image test.jpg

# Quick inference (standalone script)
python scripts/inference/infer.py \
    --model ./model_cache/Qwen/Qwen3-VL-2B-Instruct \
    --image test.jpg \
    --output result.jpg

# Visualize training logs
python scripts/visualize_training_log.py \
    --log outputs/qwen3vl_grpo/training_log.json \
    --output plots/

# Evaluate model on test set
python scripts/evaluate.py \
    --model_path outputs/qwen3vl_lora \
    --test_data data/qwen_data/test.json \
    --output_dir eval_results/

# Evaluate with multiple IoU thresholds (compute mAP)
python scripts/evaluate.py \
    --model_path outputs/qwen3vl_lora \
    --test_data data/qwen_data/test.json \
    --iou_thresholds 0.5 0.75 0.9 \
    --output_dir eval_results/

# Visualize evaluation results
python scripts/visualize_eval_results.py \
    --eval_dir eval_results/ \
    --output plots/eval_plots.png

# Compare multiple models
python scripts/visualize_eval_results.py \
    --eval_dirs eval_results/base eval_results/lora eval_results/grpo \
    --labels "Base" "LoRA" "GRPO" \
    --output plots/model_comparison.png
```

### Docker
```bash
docker-compose up -d
docker-compose logs -f traffic-detector
```

### Testing
```bash
pytest tests/ -v

# Test single detection via curl
curl -X POST "http://localhost:8000/api/v1/detect/upload" \
  -F "file=@test_image.jpg"
```

## Architecture

```
app/                         # FastAPI 应用
├── main.py                  # 入口, lifespan 管理
├── api/routes.py            # API 端点
├── core/
│   ├── config.py            # Pydantic settings
│   └── detector.py          # 核心检测逻辑
├── models/
│   ├── qwen_vl.py           # Qwen-VL 模型封装
│   └── prompts.py           # 检测提示词模板
├── schemas/detection.py     # Pydantic 数据模型
└── utils/
    ├── image_utils.py       # 图片加载工具
    └── visualization.py     # 检测框绘制

scripts/                     # 脚本工具
├── run/                     # 启动脚本
│   ├── start_api.sh         # 启动 API 服务
│   ├── start_gradio.sh      # 启动 Gradio UI
│   ├── train_lora.sh        # LoRA 微调
│   ├── train_grpo.sh        # GRPO 微调
│   ├── infer.sh             # 推理
│   ├── infer_finetuned.sh   # 微调模型推理
│   ├── convert_data.sh      # 数据转换
│   ├── download_model.sh    # 下载模型
│   ├── evaluate.sh          # 模型评估
│   ├── visualize_eval.sh    # 可视化评估结果
│   └── compare_models.sh    # 比较多个模型
├── training/                # 训练脚本
│   ├── finetune_qwen_vl.py  # LoRA/QLoRA 监督微调
│   ├── grpo_finetune.py     # GRPO 强化学习微调
│   └── dpo_finetune.py      # DPO 微调
├── data/                    # 数据处理
│   └── cvat_to_qwenvl.py    # CVAT -> Qwen-VL 格式转换
├── inference/               # 推理脚本
│   ├── infer.py             # 独立推理脚本
│   └── inference_finetuned.py # 微调模型推理
├── evaluate.py              # 目标检测评估（Precision/Recall/F1/mAP）
├── visualize_eval_results.py # 评估结果可视化
├── visualize_training_log.py # 训练日志可视化
├── benchmark.py             # 性能测试
└── download_model.py        # 模型下载

docs/                        # 文档
├── EVALUATION.md            # 评估指南（详细说明）
└── data_preparation_plan.md # 数据准备计划

tests/                       # 测试用例
examples/                    # 示例图片
gradio_app.py                # Gradio Web UI
```

## Key Design Decisions

1. **Model Loading**: Global singleton pattern via `get_model()` and `get_detector()` to avoid reloading
2. **Prompt Engineering**: Structured prompts in `prompts.py` guide model to output JSON with bbox coordinates
3. **Coordinates**: Bounding boxes use normalized 0-1000 scale (not pixels) for resolution independence
4. **Response Parsing**: `detector._extract_json()` handles JSON in markdown code blocks or raw format
5. **Fine-tuning**: QLoRA (4-bit) by default for memory efficiency; targets q/k/v/o/gate/up/down projections
6. **GRPO Training**: Reward based on format correctness, bbox IoU, category accuracy; uses KL divergence penalty
7. **Validation**: All training scripts support validation on held-out data; best models saved based on validation metrics
8. **Training Logs**: All training scripts automatically save detailed logs to `training_log.json` for analysis and visualization
9. **Usability Check**: GRPO training includes periodic usability self-checks to verify actual model output quality beyond reward metrics

## Training & Validation

### Validation Support

All training scripts now support validation during training:

- **LoRA/QLoRA** (`finetune_qwen_vl.py`): Validates every 500 steps, tracks validation loss
- **GRPO** (`grpo_finetune.py`): Validates every 200 steps, tracks reward/format/bbox/category metrics
- **DPO** (`dpo_finetune.py`): Validates every 500 steps, tracks loss/accuracy/reward_margin

### Usability Self-Check (GRPO Only)

GRPO training includes a **usability self-check** mechanism to prevent "fake training" where reward increases but model outputs remain unusable.

**What it does**:
- Every N steps (default: 200), randomly samples 8 examples
- Generates responses and parses them using actual detection logic
- Computes real-world metrics:
  - Parse success rate (can extract valid boxes?)
  - Format completeness (has box tags, numbering, status?)
  - Average IoU (localization accuracy)
  - Category accuracy (correct classification)
  - Anomaly detection accuracy (normal vs abnormal)

**Why it matters**:
- Reward curves can be misleading - model might optimize reward without producing usable outputs
- Usability check validates that the model actually generates parseable, accurate detections
- Catches issues like: format degradation, coordinate drift, category confusion

**Configuration**:
```bash
# Enable usability check (default)
python scripts/training/grpo_finetune.py \
    --train_data data.json \
    --usability_check_steps 200 \
    --usability_check_samples 8

# Disable usability check
python scripts/training/grpo_finetune.py \
    --train_data data.json \
    --usability_check_steps 0

# In shell scripts
USABILITY_CHECK_STEPS=200 ./scripts/run/train_grpo.sh
```

**Log output**:
Usability metrics are saved to `training_log.json` under `usability_history`:
- `usability_parse_success_rate`: % of samples with valid box extraction
- `usability_has_box_rate`: % with `<box>` tags
- `usability_has_numbered_rate`: % with numbered format
- `usability_has_status_rate`: % with status field
- `usability_avg_iou`: Average IoU with ground truth
- `usability_category_accuracy`: Category matching accuracy
- `usability_anomaly_accuracy`: Anomaly detection accuracy

### Training Logs

All training scripts automatically save detailed logs to `training_log.json`:

**Log Contents**:
- Training configuration (hyperparameters, model path, etc.)
- Training history (loss, reward, learning rate per step)
- Validation history (validation metrics per eval step)
- Best checkpoint information (step, metrics, path)

**Visualization**:
```bash
# View training summary
python scripts/visualize_training_log.py --log outputs/qwen3vl_grpo/training_log.json

# Generate training curves
python scripts/visualize_training_log.py \
    --log outputs/qwen3vl_grpo/training_log.json \
    --output plots/

# Export to CSV
python scripts/visualize_training_log.py \
    --log outputs/qwen3vl_grpo/training_log.json \
    --export-csv
```

See [TRAINING_LOGS.md](TRAINING_LOGS.md) for detailed documentation.

### Best Model Selection

- **LoRA**: Uses Hugging Face Trainer's built-in checkpointing (saves last 3 checkpoints)
- **GRPO**: Saves best model based on `val_reward` to `{output_dir}/best/`
- **DPO**: Saves best model based on `val_accuracy` to `{output_dir}/best/`

### Disabling Validation

```bash
# Method 1: Don't provide --val_data
python scripts/training/grpo_finetune.py --train_data data.json --output_dir outputs/

# Method 2: Set --eval_steps 0
python scripts/training/grpo_finetune.py --train_data data.json --val_data val.json --eval_steps 0

# Method 3: In shell scripts, set VAL_DATA="" or EVAL_STEPS=0
```

### Quantization Control

All training scripts now use `--no_*` flags to disable default-enabled features:

```bash
# Disable 4-bit quantization (use full precision LoRA)
python scripts/training/finetune_qwen_vl.py --train_data data.json --no_4bit

# Disable bfloat16 (use float16 instead)
python scripts/training/finetune_qwen_vl.py --train_data data.json --no_bf16

# Disable gradient checkpointing (faster but more VRAM)
python scripts/training/finetune_qwen_vl.py --train_data data.json --no_gradient_checkpointing

# In shell scripts
DISABLE_4BIT=true ./scripts/run/train_lora.sh
```

## Detection Categories

- `traffic_sign` - Traffic signs (damaged, missing, blocked, faded, tilted)
- `traffic_light` - Traffic lights (malfunction, bulb broken, not lit, color abnormal)
- `road_facility` - Road facilities (guardrail, road surface, markings, manhole)
- `guidance_screen` - Electronic displays (display error, black screen, garbled)
- `height_limit` - Height limiters (structure damaged, marking unclear)
- `cabinet` - Control cabinets (door open, damaged)

## Environment Variables

Key configuration via `.env`:
- `MODEL_TYPE` - Predefined model (e.g., `Qwen/Qwen3-VL-2B-Instruct`)
- `MODEL_NAME` - Custom local path (overrides MODEL_TYPE when set)
- `DEVICE` - `cuda` or `cpu`
- `TORCH_DTYPE` - `bfloat16`, `float16`, or `float32`
- `CONFIDENCE_THRESHOLD` - Filter detections below this value (default 0.5)
