# GRPO Training Quick Start (After Bug Fixes)

This guide shows how to use the fixed GRPO training implementation.

## Prerequisites

```bash
# Install dependencies
pip install -r requirements_finetune.txt

# Verify fixes are applied
python -c "
import re
pattern = r'<box>\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*,\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*</box>'
test = '<box>(100.5, 200.3), (300.7, 400.9)</box>'
match = re.search(pattern, test)
if match:
    print('✅ Box extraction fix verified')
else:
    print('❌ Fix not applied - please update code')
"
```

## Training with TRL (Recommended)

### Basic Training

```bash
python scripts/training/grpo_finetune_trl.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --train_data data/qwen_data/train.json \
    --val_data data/qwen_data/val.json \
    --output_dir outputs/qwen3vl_grpo \
    --num_generations 4 \
    --temperature 0.7 \
    --beta 0.1 \
    --num_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --eval_steps 200
```

### Using Shell Script

```bash
# Set environment variables
export MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
export TRAIN_DATA="data/qwen_data/train.json"
export VAL_DATA="data/qwen_data/val.json"
export NUM_GENERATIONS=4
export TEMPERATURE=0.7

# Run training
./scripts/run/train_grpo.sh
```

### Continue from SFT Model

```bash
python scripts/training/grpo_finetune_trl.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --sft_model_path outputs/qwen3vl_lora \
    --train_data data/qwen_data/train.json \
    --output_dir outputs/qwen3vl_grpo_from_sft \
    --num_generations 4
```

## Training with Custom Trainer

### Basic Training

```bash
python scripts/training/grpo_finetune.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --train_data data/qwen_data/train.json \
    --val_data data/qwen_data/val.json \
    --output_dir outputs/qwen3vl_grpo_custom \
    --num_generations 4 \
    --temperature 0.7 \
    --kl_coef 0.1
```

## Monitoring Training

### Check Training Logs

```bash
# View training log
cat outputs/qwen3vl_grpo/training_log.json | jq '.train_history[-5:]'

# Check validation results
cat outputs/qwen3vl_grpo/training_log.json | jq '.val_history[-3:]'
```

### Expected Metrics (After Fixes)

**Good Training**:
```json
{
  "step": 100,
  "loss": 0.456,
  "reward": 2.1,
  "rewards/format_reward/mean": 1.0,
  "rewards/bbox_iou_reward/mean": 0.45,
  "rewards/category_match_reward/mean": 0.6,
  "rewards/status_accuracy_reward/mean": 0.7
}
```

**Bad Training (Gaming)**:
```json
{
  "step": 100,
  "loss": 0.234,
  "reward": 1.2,
  "rewards/format_reward/mean": 0.95,
  "rewards/bbox_iou_reward/mean": 0.05,  ⚠️ Too low!
  "rewards/category_match_reward/mean": 0.1,  ⚠️ Too low!
  "rewards/status_accuracy_reward/mean": 0.15  ⚠️ Too low!
}
```

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir outputs/qwen3vl_grpo

# Or use the script
./scripts/run/start_tensorboard.sh
```

**Key Metrics to Watch**:
- `reward`: Should increase over time (target: 3-5)
- `rewards/bbox_iou_reward/mean`: Should be > 0.3 (not stuck at 0)
- `rewards/format_reward/mean`: Should be close to 1.0 (binary gate)
- `kl`: Should stay < 0.5 (not diverging too much)

## Hyperparameter Tuning

### Recommended Settings

| Parameter | Small Dataset (<1K) | Medium Dataset (1K-10K) | Large Dataset (>10K) |
|-----------|---------------------|-------------------------|----------------------|
| `num_generations` | 3-4 | 4-6 | 4-6 |
| `temperature` | 0.7-0.8 | 0.7-0.9 | 0.8-1.0 |
| `beta` (KL coef) | 0.05-0.1 | 0.1-0.2 | 0.1-0.2 |
| `learning_rate` | 5e-6 | 5e-6 | 1e-5 |
| `batch_size` | 1 | 1-2 | 2-4 |
| `gradient_accumulation_steps` | 4-8 | 4-8 | 4-8 |

### Troubleshooting

**Issue**: Rewards not increasing

```bash
# Check if format reward is binary (0 or 1)
cat outputs/qwen3vl_grpo/training_log.json | \
  jq '.train_history[] | select(.step % 50 == 0) | {step, format: .["rewards/format_reward/mean"]}'

# Should see values close to 0.0 or 1.0, not 0.5, 0.7, etc.
```

**Issue**: BBox reward always 0

```bash
# Check model outputs
python scripts/inference/infer.py \
    --model outputs/qwen3vl_grpo/checkpoint-200 \
    --image examples/sample_images/test.jpg

# Verify box format matches pattern
# Should see: <box>(x1,y1),(x2,y2)</box>
```

**Issue**: Training too slow

```bash
# Reduce image size
export MAX_IMAGE_SIZE=384  # Default: 512

# Reduce generations
export NUM_GENERATIONS=3  # Default: 4

# Use smaller model
export MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"  # Not 8B
```

## Evaluation

### Evaluate Trained Model

```bash
python scripts/evaluate.py \
    --model_path outputs/qwen3vl_grpo/best \
    --test_data data/qwen_data/test.json \
    --output_dir eval_results/grpo

# With mAP calculation
python scripts/evaluate.py \
    --model_path outputs/qwen3vl_grpo/best \
    --test_data data/qwen_data/test.json \
    --iou_thresholds 0.5 0.75 0.9 \
    --output_dir eval_results/grpo
```

### Visualize Results

```bash
# Visualize evaluation results
python scripts/visualize_eval_results.py \
    --eval_dir eval_results/grpo \
    --output plots/grpo_eval.png

# Compare with baseline
python scripts/visualize_eval_results.py \
    --eval_dirs eval_results/base eval_results/lora eval_results/grpo \
    --labels "Base" "LoRA" "GRPO" \
    --output plots/comparison.png
```

## Inference

### Single Image

```bash
python scripts/inference/inference_finetuned.py \
    --model_path outputs/qwen3vl_grpo/best \
    --image test.jpg \
    --output result.jpg
```

### Batch Inference

```bash
# Create inference script
cat > batch_infer.sh << 'SCRIPT'
#!/bin/bash
MODEL_PATH="outputs/qwen3vl_grpo/best"
INPUT_DIR="test_images"
OUTPUT_DIR="results"

mkdir -p "$OUTPUT_DIR"

for img in "$INPUT_DIR"/*.jpg; do
    basename=$(basename "$img" .jpg)
    python scripts/inference/inference_finetuned.py \
        --model_path "$MODEL_PATH" \
        --image "$img" \
        --output "$OUTPUT_DIR/${basename}_result.jpg"
done
SCRIPT

chmod +x batch_infer.sh
./batch_infer.sh
```

## Best Practices

### 1. Start with SFT

```bash
# Step 1: LoRA fine-tuning (SFT)
./scripts/run/train_lora.sh

# Step 2: GRPO on top of SFT
python scripts/training/grpo_finetune_trl.py \
    --sft_model_path outputs/qwen3vl_lora \
    --train_data data/qwen_data/train.json \
    --output_dir outputs/qwen3vl_grpo_from_sft
```

### 2. Use Validation Set

Always provide `--val_data` to:
- Monitor overfitting
- Save best checkpoint based on validation reward
- Track generalization

### 3. Monitor Reward Breakdown

Check that all reward components are contributing:
```bash
# Good: All rewards > 0
format: 1.0, bbox: 0.6, category: 0.8, status: 0.7

# Bad: Only format is high
format: 1.0, bbox: 0.05, category: 0.1, status: 0.1
```

### 4. Adjust Temperature

- **Low temp (0.5-0.7)**: More deterministic, faster convergence
- **High temp (0.8-1.0)**: More exploration, better diversity

Start with 0.7, increase if rewards plateau.

### 5. Use Gradient Accumulation

Instead of large batch size:
```bash
# ❌ Bad: Large batch size (OOM)
--batch_size 8

# ✅ Good: Gradient accumulation
--batch_size 1 --gradient_accumulation_steps 8
```

## Common Pitfalls

### ❌ Don't: Use old checkpoints

Old checkpoints (before bug fixes) have incorrect gradients. Always restart training.

### ❌ Don't: Set num_generations < 3

GRPO needs multiple generations for advantage estimation. Minimum: 3, recommended: 4-6.

### ❌ Don't: Ignore validation metrics

If validation reward decreases while training reward increases → overfitting.

### ❌ Don't: Use very high KL coefficient

High `beta` (>0.3) prevents exploration. Start with 0.1.

### ✅ Do: Check reward breakdown

Ensure bbox/category/status rewards are non-zero, not just format.

### ✅ Do: Use early stopping

Save best checkpoint based on validation reward, not training loss.

### ✅ Do: Visualize outputs

Regularly check model outputs to ensure they're improving:
```bash
python scripts/inference/infer.py \
    --model outputs/qwen3vl_grpo/checkpoint-200 \
    --image test.jpg
```

## Example Training Pipeline

```bash
#!/bin/bash
set -e

# 1. Prepare data
python scripts/data/cvat_to_qwenvl.py \
    --cvat-dir data/annotations \
    --output-dir data/qwen_data

# 2. LoRA fine-tuning (optional but recommended)
./scripts/run/train_lora.sh

# 3. GRPO training
python scripts/training/grpo_finetune_trl.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --sft_model_path outputs/qwen3vl_lora \
    --train_data data/qwen_data/train.json \
    --val_data data/qwen_data/val.json \
    --output_dir outputs/qwen3vl_grpo \
    --num_generations 4 \
    --temperature 0.7 \
    --beta 0.1 \
    --num_epochs 1 \
    --eval_steps 200

# 4. Evaluate
python scripts/evaluate.py \
    --model_path outputs/qwen3vl_grpo/best \
    --test_data data/qwen_data/test.json \
    --iou_thresholds 0.5 0.75 0.9 \
    --output_dir eval_results/grpo

# 5. Visualize
python scripts/visualize_eval_results.py \
    --eval_dir eval_results/grpo \
    --output plots/grpo_eval.png

echo "✅ Training pipeline complete!"
```

## Resources

- **Bug Fixes Documentation**: `GRPO_BUGFIXES.md`
- **Training Logs**: `outputs/qwen3vl_grpo/training_log.json`
- **Evaluation Guide**: `docs/EVALUATION.md`
- **Test Suite**: `/tmp/test_grpo_fixes.py`

## Support

If you encounter issues:

1. Check `GRPO_BUGFIXES.md` for known issues
2. Verify fixes are applied (run test suite)
3. Check training logs for reward breakdown
4. Visualize model outputs to debug
