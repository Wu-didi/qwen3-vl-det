# 训练日志记录功能

## 概述

所有训练脚本现在都会自动保存详细的训练日志到 JSON 文件，包括：
- 训练配置
- 训练历史（loss、reward、learning rate 等）
- 验证历史（validation metrics）
- 最佳检查点信息

## 日志文件位置

训练日志保存在输出目录的 `training_log.json` 文件中：

```
outputs/qwen3vl_grpo/
├── training_log.json          # 训练日志
├── best/                       # 最佳模型
├── checkpoint-samples-200/     # 定期检查点
└── final/                      # 最终模型
```

## 日志格式

### LoRA/QLoRA 训练日志

```json
{
  "config": {
    "model_path": "Qwen/Qwen3-VL-2B-Instruct",
    "train_data": "data/qwen_data/train.json",
    "val_data": "data/qwen_data/val.json",
    "num_epochs": 3,
    "batch_size": 2,
    "learning_rate": 0.0002,
    "lora_r": 64,
    "use_4bit": true,
    "bf16": true
  },
  "train_history": [
    {
      "step": 10,
      "epoch": 0.1,
      "loss": 1.2345,
      "learning_rate": 0.00002
    },
    {
      "step": 20,
      "epoch": 0.2,
      "loss": 1.1234,
      "learning_rate": 0.00004
    }
  ],
  "val_history": [
    {
      "step": 500,
      "epoch": 1.0,
      "eval_loss": 0.9876
    }
  ],
  "final_metrics": {
    "best_metric": 0.9876,
    "best_checkpoint": "outputs/qwen3vl_lora/checkpoint-1500"
  }
}
```

### GRPO 训练日志

```json
{
  "config": {
    "model_path": "Qwen/Qwen3-VL-2B-Instruct",
    "train_data": "data/qwen_data/train.json",
    "val_data": "data/qwen_data/val.json",
    "num_generations": 4,
    "temperature": 0.7,
    "kl_coef": 0.1
  },
  "train_history": [
    {
      "step": 10,
      "epoch": 1,
      "samples": 40,
      "loss": 0.5432,
      "reward": 2.3456,
      "kl": 0.0123,
      "lr": 0.00001
    }
  ],
  "val_history": [
    {
      "step": 200,
      "epoch": 1,
      "samples": 800,
      "val_reward": 2.5678,
      "val_format": 0.92,
      "val_bbox": 0.78,
      "val_category": 0.85,
      "val_completeness": 0.88
    }
  ],
  "best_checkpoint": {
    "step": 400,
    "epoch": 1,
    "samples": 1600,
    "val_reward": 2.7890,
    "path": "outputs/qwen3vl_grpo/best"
  }
}
```

### DPO 训练日志

```json
{
  "config": {
    "model_path": "Qwen/Qwen3-VL-2B-Instruct",
    "train_data": "data/dpo_data/train.json",
    "val_data": "data/dpo_data/val.json",
    "beta": 0.1
  },
  "train_history": [
    {
      "step": 10,
      "epoch": 1,
      "loss": 0.4523,
      "accuracy": 0.75,
      "reward_margin": 1.2345
    }
  ],
  "val_history": [
    {
      "step": 500,
      "epoch": 1,
      "val_loss": 0.3456,
      "val_accuracy": 0.82,
      "val_reward_margin": 1.5678
    }
  ],
  "best_checkpoint": {
    "step": 1000,
    "epoch": 1,
    "val_accuracy": 0.85,
    "val_loss": 0.3123,
    "val_reward_margin": 1.6789,
    "path": "outputs/qwen3vl_dpo/best"
  }
}
```

## 使用方法

### 1. 查看训练日志摘要

```bash
python scripts/visualize_training_log.py --log outputs/qwen3vl_grpo/training_log.json
```

输出示例：
```
============================================================
Training Summary
============================================================

Configuration:
  Model: Qwen/Qwen3-VL-2B-Instruct
  Output: outputs/qwen3vl_grpo
  Epochs: 1
  Batch Size: 1
  Learning Rate: 1e-05
  Gradient Accumulation: 4

Training History: 150 entries
  Final Step: 150
  Final Epoch: 1
  Final loss: 0.4523
  Final reward: 2.7890
  Final kl: 0.0234
  Final lr: 1.00e-05

Validation History: 5 entries
  Final Val Step: 150
  val_reward: 2.8901
  val_format: 0.94
  val_bbox: 0.82
  val_category: 0.88

Best Checkpoint:
  Step: 120
  Epoch: 1
  val_reward: 2.8901
  Path: outputs/qwen3vl_grpo/best
============================================================
```

### 2. 生成训练曲线图

```bash
# 生成并保存图片
python scripts/visualize_training_log.py \
    --log outputs/qwen3vl_grpo/training_log.json \
    --output plots/

# 生成的图片：
# plots/loss_curve.png
# plots/reward_curve.png
# plots/kl_curve.png
# plots/lr_curve.png
```

### 3. 导出为 CSV 格式

```bash
python scripts/visualize_training_log.py \
    --log outputs/qwen3vl_grpo/training_log.json \
    --export-csv \
    --output exports/

# 生成的文件：
# exports/train_history.csv
# exports/val_history.csv
```

### 4. 只打印摘要（不生成图片）

```bash
python scripts/visualize_training_log.py \
    --log outputs/qwen3vl_grpo/training_log.json \
    --no-plot
```

## 训练曲线示例

### LoRA 训练曲线
- **Loss Curve**: 训练和验证损失随步数变化
- **Learning Rate Curve**: 学习率调度曲线（warmup + decay）

### GRPO 训练曲线
- **Loss Curve**: 策略损失
- **Reward Curve**: 平均奖励分数（训练集和验证集）
- **KL Divergence Curve**: 与参考模型的 KL 散度
- **Format/BBox/Category Curves**: 各项子指标的变化

### DPO 训练曲线
- **Loss Curve**: DPO 损失
- **Accuracy Curve**: 偏好准确率（chosen > rejected 的比例）
- **Reward Margin Curve**: 奖励差值

## 日志记录时机

### 训练日志
- **LoRA**: 每 `logging_steps` 步记录一次（默认 10 步）
- **GRPO**: 每 `logging_steps` 步记录一次（默认 10 步）
- **DPO**: 每 `logging_steps` 步记录一次（默认 10 步）

### 验证日志
- **LoRA**: 每 `eval_steps` 步验证一次（默认 500 步）
- **GRPO**: 每 `eval_steps` 步验证一次（默认 200 步）
- **DPO**: 每 `eval_steps` 步验证一次（默认 500 步）

### 日志保存
- 验证后立即保存（包含最新的验证结果）
- 训练结束时保存最终日志

## 与 TensorBoard 的对比

| 特性 | training_log.json | TensorBoard |
|------|-------------------|-------------|
| 格式 | JSON（易于解析） | 二进制事件文件 |
| 可读性 | 高（文本格式） | 需要 TensorBoard 查看 |
| 可编程性 | 高（Python dict） | 需要 TensorBoard API |
| 存储大小 | 小 | 较大 |
| 实时查看 | 否 | 是 |
| 离线分析 | 是 | 需要启动服务 |
| 配置信息 | 完整保存 | 部分保存 |
| 最佳模型信息 | 保存 | 不保存 |

**建议**：
- 使用 `training_log.json` 进行离线分析和自动化处理
- 使用 TensorBoard 进行实时监控（如果需要）
- 两者可以同时使用，互为补充

## 编程访问日志

### Python 示例

```python
import json

# 加载日志
with open('outputs/qwen3vl_grpo/training_log.json', 'r') as f:
    log = json.load(f)

# 访问配置
config = log['config']
print(f"Model: {config['model_path']}")
print(f"Learning Rate: {config['learning_rate']}")

# 访问训练历史
train_history = log['train_history']
final_loss = train_history[-1]['loss']
final_reward = train_history[-1]['reward']
print(f"Final Loss: {final_loss:.4f}")
print(f"Final Reward: {final_reward:.4f}")

# 访问验证历史
val_history = log['val_history']
if val_history:
    best_val_reward = max(entry['val_reward'] for entry in val_history)
    print(f"Best Val Reward: {best_val_reward:.4f}")

# 访问最佳检查点
if 'best_checkpoint' in log:
    best = log['best_checkpoint']
    print(f"Best Checkpoint: {best['path']}")
    print(f"Best Val Reward: {best['val_reward']:.4f}")
```

### 绘制自定义曲线

```python
import json
import matplotlib.pyplot as plt

# 加载日志
with open('outputs/qwen3vl_grpo/training_log.json', 'r') as f:
    log = json.load(f)

# 提取数据
train_history = log['train_history']
steps = [entry['step'] for entry in train_history]
rewards = [entry['reward'] for entry in train_history]

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(steps, rewards, linewidth=2)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Training Reward Curve')
plt.grid(True, alpha=0.3)
plt.savefig('reward_curve.png', dpi=150)
```

## 故障排查

### 日志文件不存在
- 确认训练已经开始并至少运行了一个 logging step
- 检查输出目录是否正确
- 查看训练日志中是否有错误信息

### 日志文件为空或不完整
- 训练可能被中断
- 检查磁盘空间是否充足
- 查看训练进程是否正常结束

### 验证历史为空
- 确认提供了 `--val_data` 参数
- 确认 `--eval_steps > 0`
- 检查验证集文件是否存在

### 图片生成失败
- 确认安装了 matplotlib: `pip install matplotlib`
- 检查输出目录是否有写权限
- 查看错误信息

## 最佳实践

1. **定期备份日志**：训练日志包含完整的训练历史，建议定期备份
2. **版本控制**：将训练配置和日志一起保存，便于复现实验
3. **对比实验**：使用日志文件对比不同超参数的效果
4. **自动化分析**：编写脚本自动分析多个实验的日志
5. **监控训练**：定期查看日志摘要，及时发现训练问题

## 示例工作流

```bash
# 1. 启动训练
./scripts/run/train_grpo.sh

# 2. 训练过程中查看摘要（另一个终端）
watch -n 60 "python scripts/visualize_training_log.py \
    --log outputs/qwen3vl_grpo/training_log.json --no-plot"

# 3. 训练结束后生成完整报告
python scripts/visualize_training_log.py \
    --log outputs/qwen3vl_grpo/training_log.json \
    --output reports/grpo_experiment_1/ \
    --export-csv

# 4. 对比多个实验
python scripts/compare_experiments.py \
    --logs outputs/*/training_log.json \
    --output comparison_report.html
```

## 未来改进

- [ ] 添加实时日志流式更新
- [ ] 支持多实验对比可视化
- [ ] 添加更多统计指标（均值、方差、趋势等）
- [ ] 支持导出为 HTML 报告
- [ ] 集成到 Web UI 中实时查看
