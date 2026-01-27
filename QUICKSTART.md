# 快速开始指南

## 本次更新内容

✅ **修复 argparse 参数问题**：现在可以正确关闭 4bit、bf16 等默认开启的功能
✅ **添加验证逻辑**：GRPO 和 DPO 训练支持验证集评估和最佳模型保存
✅ **添加训练日志记录**：所有训练自动保存详细日志到 JSON 文件
✅ **添加可视化工具**：查看训练曲线、对比多个实验

---

## 快速验证更新

运行测试脚本确认所有功能正常：

```bash
bash scripts/test_updates.sh
```

预期输出：
```
==========================================
All tests passed! ✓
==========================================
```

---

## 使用示例

### 1. 训练参数控制

```bash
# 默认使用 4bit QLoRA（推荐）
python scripts/training/finetune_qwen_vl.py \
    --train_data data/qwen_data/train.json \
    --output_dir outputs/qwen3vl_lora

# 关闭 4bit，使用全精度 LoRA
python scripts/training/finetune_qwen_vl.py \
    --train_data data/qwen_data/train.json \
    --no_4bit \
    --output_dir outputs/qwen3vl_lora_fp16

# 关闭 bf16，使用 fp16
python scripts/training/finetune_qwen_vl.py \
    --train_data data/qwen_data/train.json \
    --no_bf16 \
    --output_dir outputs/qwen3vl_lora_fp16

# Shell 脚本方式
DISABLE_4BIT=true ./scripts/run/train_lora.sh
```

### 2. 带验证集训练

```bash
# GRPO 训练（带验证）
python scripts/training/grpo_finetune.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --train_data data/qwen_data/train.json \
    --val_data data/qwen_data/val.json \
    --eval_steps 200 \
    --output_dir outputs/qwen3vl_grpo

# DPO 训练（带验证）
python scripts/training/dpo_finetune.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --train_data data/dpo_data/train.json \
    --val_data data/dpo_data/val.json \
    --eval_steps 500 \
    --output_dir outputs/qwen3vl_dpo

# Shell 脚本方式
./scripts/run/train_grpo.sh  # 已配置验证集
```

### 3. 查看训练日志

```bash
# 打印训练摘要
python scripts/visualize_training_log.py \
    --log outputs/qwen3vl_grpo/training_log.json

# 生成训练曲线图
python scripts/visualize_training_log.py \
    --log outputs/qwen3vl_grpo/training_log.json \
    --output plots/

# 导出为 CSV
python scripts/visualize_training_log.py \
    --log outputs/qwen3vl_grpo/training_log.json \
    --export-csv
```

### 4. 对比多个实验

```bash
# 对比不同实验
python scripts/compare_training_logs.py \
    --logs outputs/exp1/training_log.json \
           outputs/exp2/training_log.json \
           outputs/exp3/training_log.json \
    --output comparison/ \
    --html

# 查看生成的 HTML 报告
open comparison/comparison.html
```

### 5. 使用最佳模型推理

```bash
# 使用验证集上表现最好的模型
python scripts/inference/inference_finetuned.py \
    --model_path outputs/qwen3vl_grpo/best \
    --image test.jpg
```

---

## 训练输出目录结构

```
outputs/qwen3vl_grpo/
├── training_log.json          # ✨ 训练日志（新增）
├── best/                       # ✨ 最佳验证模型（新增）
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
├── checkpoint-samples-200/     # 定期检查点
├── checkpoint-samples-400/
├── final/                      # 最终模型
└── grpo_config.json           # 训练配置
```

---

## 训练日志示例

`training_log.json` 包含完整的训练历史：

```json
{
  "config": {
    "model_path": "Qwen/Qwen3-VL-2B-Instruct",
    "num_epochs": 1,
    "learning_rate": 1e-05,
    "num_generations": 4
  },
  "train_history": [
    {
      "step": 10,
      "loss": 0.5432,
      "reward": 2.3456,
      "kl": 0.0123
    }
  ],
  "val_history": [
    {
      "step": 200,
      "val_reward": 2.5678,
      "val_format": 0.92,
      "val_bbox": 0.78
    }
  ],
  "best_checkpoint": {
    "step": 400,
    "val_reward": 2.7890,
    "path": "outputs/qwen3vl_grpo/best"
  }
}
```

---

## 验证日志示例

训练过程中会定期输出验证结果：

```
2026-01-25 10:30:00 - INFO - Running validation...
Validation: 100%|████████| 50/50 [00:45<00:00,  1.11it/s]
2026-01-25 10:30:45 - INFO - Validation results:
  reward=2.3456, format=0.92, bbox=0.78, category=0.85
2026-01-25 10:30:45 - INFO - New best validation reward: 2.3456
2026-01-25 10:30:45 - INFO - Checkpoint saved to outputs/qwen3vl_grpo/best
```

---

## 常见问题

### Q: 如何禁用验证？

**A**: 有两种方法：

```bash
# 方法1: 不提供 --val_data
python scripts/training/grpo_finetune.py \
    --train_data data.json \
    --output_dir outputs/

# 方法2: 设置 --eval_steps 0
python scripts/training/grpo_finetune.py \
    --train_data data.json \
    --val_data val.json \
    --eval_steps 0
```

### Q: 如何关闭 4bit 量化？

**A**: 使用 `--no_4bit` 参数：

```bash
python scripts/training/finetune_qwen_vl.py \
    --train_data data.json \
    --no_4bit

# 或在 shell 脚本中
DISABLE_4BIT=true ./scripts/run/train_lora.sh
```

### Q: 验证会增加多少训练时间？

**A**: 验证 50 个样本大约需要 30-60 秒，取决于：
- 模型大小（2B vs 8B）
- 生成长度（max_new_tokens）
- GPU 性能

可以通过调整 `eval_steps` 来平衡验证频率和训练速度。

### Q: 如何查看所有可用参数？

**A**: 使用 `--help` 查看：

```bash
python scripts/training/grpo_finetune.py --help
python scripts/training/dpo_finetune.py --help
python scripts/training/finetune_qwen_vl.py --help
```

### Q: 训练日志文件在哪里？

**A**: 在输出目录的 `training_log.json`：

```bash
outputs/qwen3vl_grpo/training_log.json
outputs/qwen3vl_dpo/training_log.json
outputs/qwen3vl_lora/training_log.json
```

---

## 完整训练流程示例

```bash
# 1. 准备数据
python scripts/data/cvat_to_qwenvl.py \
    --cvat-dir data/annotations \
    --output-dir data/qwen_data

# 2. LoRA 监督微调
python scripts/training/finetune_qwen_vl.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --train_data data/qwen_data/train.json \
    --val_data data/qwen_data/val.json \
    --output_dir outputs/qwen3vl_lora \
    --num_epochs 3

# 3. GRPO 强化学习微调（在 LoRA 基础上）
python scripts/training/grpo_finetune.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --sft_model_path outputs/qwen3vl_lora \
    --train_data data/qwen_data/train.json \
    --val_data data/qwen_data/val.json \
    --output_dir outputs/qwen3vl_grpo \
    --num_generations 4 \
    --eval_steps 200

# 4. 查看训练日志
python scripts/visualize_training_log.py \
    --log outputs/qwen3vl_grpo/training_log.json \
    --output plots/

# 5. 对比 LoRA 和 GRPO
python scripts/compare_training_logs.py \
    --logs outputs/qwen3vl_lora/training_log.json \
           outputs/qwen3vl_grpo/training_log.json \
    --output comparison/ \
    --html

# 6. 使用最佳模型推理
python scripts/inference/inference_finetuned.py \
    --model_path outputs/qwen3vl_grpo/best \
    --image test.jpg
```

---

## 监控训练进度

### 实时监控（推荐）

在另一个终端运行：

```bash
# 每 60 秒刷新一次训练摘要
watch -n 60 "python scripts/visualize_training_log.py \
    --log outputs/qwen3vl_grpo/training_log.json --no-plot"
```

### 查看最新日志

```bash
# 查看最后 20 行训练日志
tail -n 20 outputs/qwen3vl_grpo/training_log.json

# 使用 jq 格式化查看
cat outputs/qwen3vl_grpo/training_log.json | jq .
```

---

## 详细文档

- **验证逻辑详细说明**：[VALIDATION_UPDATE.md](VALIDATION_UPDATE.md)
- **训练日志详细说明**：[TRAINING_LOGS.md](TRAINING_LOGS.md)
- **完整更新总结**：[UPDATES_SUMMARY.md](UPDATES_SUMMARY.md)
- **项目文档**：[CLAUDE.md](CLAUDE.md)

---

## 需要帮助？

1. 运行测试脚本：`bash scripts/test_updates.sh`
2. 查看详细文档：上述 markdown 文件
3. 查看命令帮助：`python script.py --help`

---

**更新日期**：2026-01-25
**版本**：v2.0
