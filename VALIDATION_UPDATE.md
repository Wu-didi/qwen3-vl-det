# 验证逻辑更新说明

## 概述

为 GRPO 和 DPO 训练脚本添加了完整的验证逻辑，现在所有训练脚本都支持在训练过程中定期在验证集上评估模型性能。

## 修改的文件

### 1. Python 训练脚本

#### `scripts/training/grpo_finetune.py`
**新增功能**：
- 添加 `--val_data` 参数：指定验证集路径（可选）
- 添加 `--eval_steps` 参数：每 N 步验证一次（默认 200，设为 0 禁用）
- 添加 `GRPOTrainer.evaluate()` 方法：在验证集上评估模型
- 训练时自动保存最佳验证 reward 的模型到 `best/` 目录

**验证指标**：
- `val_reward`: 平均奖励分数
- `val_format`: 格式正确性
- `val_bbox`: 边界框准确性（IoU）
- `val_category`: 类别匹配率
- `val_completeness`: 检测完整性

#### `scripts/training/dpo_finetune.py`
**新增功能**：
- 添加 `--val_data` 参数：指定验证集路径（可选）
- 添加 `--eval_steps` 参数：每 N 步验证一次（默认 500，设为 0 禁用）
- 添加 `DPOTrainer.evaluate()` 方法：在验证集上评估模型
- 训练时自动保存最佳验证准确率的模型到 `best/` 目录

**验证指标**：
- `val_loss`: DPO 损失
- `val_accuracy`: 偏好准确率（chosen > rejected 的比例）
- `val_reward_margin`: 奖励差值

#### `scripts/training/grpo_finetune_trl.py`
**新增功能**：
- 添加 `--val_data` 参数：指定验证集路径（可选）
- 添加 `--eval_steps` 参数：每 N 步验证一次（默认 200，设为 0 禁用）
- 使用 TRL 的内置验证机制（通过 `eval_dataset` 和 `eval_strategy`）

### 2. Shell 启动脚本

#### `scripts/run/train_grpo.sh`
**新增配置**：
```bash
VAL_DATA="data/hefei_last_dataset/qwen_data/val.json"  # 验证集路径
EVAL_STEPS=200                                          # 验证频率
```

#### `scripts/run/train_dpo.sh`
**新增配置**：
```bash
VAL_DATA="data/dpo_data/val.json"  # 验证集路径
EVAL_STEPS=500                      # 验证频率
```

#### `scripts/run/train_grpo_trl.sh`
**新增配置**：
```bash
VAL_DATA="data/hefei_last_dataset/qwen_data/val.json"  # 验证集路径
EVAL_STEPS=200                                          # 验证频率
```

## 使用方法

### 1. 使用 Python 脚本直接训练

#### GRPO 训练（带验证）
```bash
python scripts/training/grpo_finetune.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --train_data data/qwen_data/train.json \
    --val_data data/qwen_data/val.json \
    --eval_steps 200 \
    --output_dir outputs/qwen3vl_grpo
```

#### DPO 训练（带验证）
```bash
python scripts/training/dpo_finetune.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --train_data data/dpo_data/train.json \
    --val_data data/dpo_data/val.json \
    --eval_steps 500 \
    --output_dir outputs/qwen3vl_dpo
```

#### 禁用验证
```bash
# 方法1: 不提供 --val_data
python scripts/training/grpo_finetune.py \
    --train_data data/qwen_data/train.json \
    --output_dir outputs/qwen3vl_grpo

# 方法2: 设置 --eval_steps 0
python scripts/training/grpo_finetune.py \
    --train_data data/qwen_data/train.json \
    --val_data data/qwen_data/val.json \
    --eval_steps 0 \
    --output_dir outputs/qwen3vl_grpo
```

### 2. 使用 Shell 脚本训练

#### 修改配置文件
编辑 `scripts/run/train_grpo.sh`：
```bash
# 启用验证
VAL_DATA="data/hefei_last_dataset/qwen_data/val.json"
EVAL_STEPS=200

# 禁用验证（留空或设为 0）
VAL_DATA=""
# 或
EVAL_STEPS=0
```

#### 运行训练
```bash
./scripts/run/train_grpo.sh
```

## 验证逻辑说明

### 验证时机
- 每 `eval_steps` 步执行一次验证
- 验证时模型切换到 `eval()` 模式
- 验证完成后恢复 `train()` 模式

### 验证样本数量
为了节省时间，验证时默认只评估 50 个样本：
- GRPO: `self.evaluate(val_dataset, num_samples=50)`
- DPO: `self.evaluate(val_dataset, num_samples=50)`

可以修改代码中的 `num_samples` 参数来调整验证样本数量。

### 最佳模型保存
- GRPO: 根据 `val_reward` 保存最佳模型
- DPO: 根据 `val_accuracy` 保存最佳模型
- 最佳模型保存在 `{output_dir}/best/` 目录

### 输出目录结构
```
outputs/qwen3vl_grpo/
├── best/                          # 最佳验证模型
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
├── checkpoint-samples-200/        # 定期检查点
├── checkpoint-samples-400/
├── final/                         # 最终模型
├── training_log.json              # 训练日志
└── grpo_config.json
```

## 验证日志示例

### GRPO 验证日志
```
2026-01-25 10:30:00 - INFO - Running validation...
Validation: 100%|████████| 50/50 [00:45<00:00,  1.11it/s]
2026-01-25 10:30:45 - INFO - Validation results: reward=2.3456, format=0.92, bbox=0.78, category=0.85
2026-01-25 10:30:45 - INFO - New best validation reward: 2.3456
2026-01-25 10:30:45 - INFO - Checkpoint saved to outputs/qwen3vl_grpo/best
```

### DPO 验证日志
```
2026-01-25 10:30:00 - INFO - Running validation...
Validation: 100%|████████| 50/50 [00:30<00:00,  1.67it/s]
2026-01-25 10:30:30 - INFO - Validation results: loss=0.4523, accuracy=0.78, reward_margin=1.2345
2026-01-25 10:30:30 - INFO - New best validation accuracy: 0.78
2026-01-25 10:30:30 - INFO - Checkpoint saved to outputs/qwen3vl_dpo/best
```

## 注意事项

1. **验证集格式**：验证集格式必须与训练集相同
   - GRPO: 需要 `conversations` 格式（与 LoRA 训练相同）
   - DPO: 需要 `chosen`/`rejected` 格式

2. **显存占用**：验证时会生成响应，需要额外显存
   - 如果显存不足，可以减少 `num_samples` 或增大 `eval_steps`

3. **训练时间**：验证会增加训练时间
   - 50 个样本的验证大约需要 30-60 秒（取决于模型大小和生成长度）
   - 可以通过调整 `eval_steps` 来平衡验证频率和训练速度

4. **最佳模型选择**：
   - GRPO: 选择 `val_reward` 最高的模型
   - DPO: 选择 `val_accuracy` 最高的模型
   - 如果验证集不具代表性，最佳模型可能不是最终模型

## 与 LoRA 训练的对比

| 特性 | LoRA (finetune_qwen_vl.py) | GRPO | DPO |
|------|---------------------------|------|-----|
| 验证支持 | ✅ 已有 | ✅ 新增 | ✅ 新增 |
| 验证指标 | Loss | Reward, Format, BBox, Category | Loss, Accuracy, Margin |
| 验证频率 | 每 500 步 | 每 200 步（可配置） | 每 500 步（可配置） |
| 最佳模型 | ❌ 无 | ✅ 基于 val_reward | ✅ 基于 val_accuracy |
| 验证样本 | 全部 | 50 个（可配置） | 50 个（可配置） |

## 后续优化建议

1. **Early Stopping**：添加早停机制，当验证指标不再提升时自动停止训练
2. **验证样本数量配置**：将 `num_samples` 作为命令行参数
3. **更多验证指标**：添加 F1 score、mAP 等检测任务常用指标
4. **验证结果可视化**：保存验证集的预测结果，便于分析
5. **学习率调度**：根据验证指标动态调整学习率
