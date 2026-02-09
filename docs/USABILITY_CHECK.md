# Usability Self-Check for GRPO Training

## 问题背景

在 GRPO 强化学习训练中，仅依赖 reward 曲线可能会被"假训练"误导：

- **Reward 在上升**：模型可能在优化 reward 函数
- **但输出不可用**：格式错误、坐标乱码、类别混乱

这种情况下，tensorboard 看着很好，但模型实际上不能用。

## 解决方案：可用性自检

每隔 N 步（默认 200），自动进行可用性自检：

1. **随机抽样**：从训练集/验证集随机抽取 8 个样本
2. **实际推理**：调用 `generate_responses()` 生成输出
3. **解析验证**：使用真实的解析逻辑（`parse_box_format()`）提取检测结果
4. **统计指标**：计算实际可用性指标

## 可用性指标

### 格式完整性
- `usability_parse_success_rate`: 成功解析出完整检测结果的比例
- `usability_has_box_rate`: 包含 `<box>` 标签的比例
- `usability_has_numbered_rate`: 包含序号格式（`1. xxx`）的比例
- `usability_has_status_rate`: 包含状态字段的比例

### 检测准确性
- `usability_avg_iou`: 预测框与真实框的平均 IoU
- `usability_category_accuracy`: 类别匹配准确率（IoU > 0.5 时）
- `usability_anomaly_accuracy`: 异常状态判断准确率

### 检测完整性
- `usability_avg_pred_boxes`: 平均预测框数量
- `usability_avg_gt_boxes`: 平均真实框数量

## 使用方法

### 1. 启用可用性自检（默认）

```bash
# 使用 shell 脚本（推荐）
./scripts/run/train_grpo.sh

# 或直接调用 Python
python scripts/training/grpo_finetune.py \
    --train_data data/qwen_data/train.json \
    --val_data data/qwen_data/val.json \
    --output_dir outputs/qwen3vl_grpo \
    --usability_check_steps 200 \
    --usability_check_samples 8
```

### 2. 自定义检查频率

```bash
# 每 100 步检查一次，抽样 16 个样本
USABILITY_CHECK_STEPS=100 USABILITY_CHECK_SAMPLES=16 ./scripts/run/train_grpo.sh

# 或
python scripts/training/grpo_finetune.py \
    --train_data data.json \
    --usability_check_steps 100 \
    --usability_check_samples 16
```

### 3. 禁用可用性自检

```bash
# 设置 steps 为 0
USABILITY_CHECK_STEPS=0 ./scripts/run/train_grpo.sh

# 或
python scripts/training/grpo_finetune.py \
    --train_data data.json \
    --usability_check_steps 0
```

## 查看结果

### 1. 训练时实时查看

训练过程中会打印可用性自检结果：

```
Usability check results:
  Parse success: 87.5% (7/8)
  Has <box>: 100.0%
  Has numbered: 100.0%
  Has status: 87.5%
  Avg IoU: 0.723 (n=14)
  Category accuracy: 85.7% (12/14)
  Anomaly accuracy: 92.9% (13/14)
  Avg pred boxes: 1.9, Avg GT boxes: 2.0
```

### 2. 查看训练日志

所有可用性自检结果保存在 `training_log.json` 的 `usability_history` 字段：

```bash
# 查看摘要
python scripts/visualize_training_log.py \
    --log outputs/qwen3vl_grpo/training_log.json

# 生成可视化图表
python scripts/visualize_training_log.py \
    --log outputs/qwen3vl_grpo/training_log.json \
    --output plots/

# 导出 CSV
python scripts/visualize_training_log.py \
    --log outputs/qwen3vl_grpo/training_log.json \
    --export-csv
```

### 3. 可视化图表

`visualize_training_log.py` 会生成以下图表：

- `training_metrics.png`: 训练指标（loss, reward, KL, LR）
- `validation_metrics.png`: 验证指标（val_reward, format, bbox, category）
- `usability_metrics.png`: 可用性指标（6 个子图）
- `reward_vs_usability.png`: **Reward vs Usability 对比图**（最重要）

**重点关注 `reward_vs_usability.png`**：
- 左轴：Validation Reward（绿色）
- 右轴：Parse Success（蓝色）、Avg IoU（橙色）、Category Acc（红色）

**健康的训练**：Reward 和 Usability 指标同步上升
**假训练**：Reward 上升但 Usability 指标停滞或下降

## 典型问题诊断

### 问题 1：Parse Success Rate 低

**症状**：`usability_parse_success_rate < 50%`

**原因**：
- 模型输出格式崩溃
- 没有生成 `<box>` 标签
- 坐标格式错误

**解决**：
- 检查 reward 函数的 format 权重是否太低
- 增加 `reward_format_weight`（默认 0.2）
- 检查训练数据格式是否正确

### 问题 2：IoU 很低

**症状**：`usability_avg_iou < 0.3`

**原因**：
- 坐标预测不准确
- 模型在随机猜测位置

**解决**：
- 增加 `reward_bbox_weight`（默认 3.0）
- 检查训练数据的坐标是否正确
- 降低学习率，避免过度更新

### 问题 3：Category Accuracy 低

**症状**：`usability_category_accuracy < 50%`

**原因**：
- 类别预测混乱
- 模型没有学会区分不同设备

**解决**：
- 增加 `reward_category_weight`（默认 2.0）
- 检查训练数据的类别标注是否一致
- 增加训练样本多样性

### 问题 4：Reward 上升但 Usability 不变

**症状**：Reward 曲线上升，但所有 usability 指标停滞

**原因**：
- **典型的假训练**
- 模型在优化 reward 函数的漏洞，而不是真正学习

**解决**：
- 检查 reward 函数设计是否合理
- 增加 KL 系数（`kl_coef`），防止模型偏离太远
- 降低学习率
- 检查参考模型是否正确加载

## 最佳实践

1. **始终启用可用性自检**：默认配置（200 steps, 8 samples）已经足够
2. **优先看 Usability 指标**：比 Reward 更能说明模型是否可用
3. **对比 Reward vs Usability**：两者应该同步上升
4. **早期停止**：如果 Usability 指标不上升，及时停止训练
5. **保存最佳模型**：基于 `val_reward` 保存，但部署前验证 usability 指标

## 性能影响

- **额外时间**：每次自检约 10-30 秒（取决于样本数和模型大小）
- **频率建议**：200 steps（默认）平衡了监控频率和训练效率
- **显存占用**：可忽略（推理时使用 `@torch.no_grad()`）

## 示例输出

```
Step 200 (samples: 800): loss=0.3421, reward=2.1543, kl=0.0234, lr=5.00e-06
Running usability check on 8 samples...
Usability check results:
  Parse success: 87.5% (7/8)
  Has <box>: 100.0%
  Has numbered: 100.0%
  Has status: 87.5%
  Avg IoU: 0.723 (n=14)
  Category accuracy: 85.7% (12/14)
  Anomaly accuracy: 92.9% (13/14)
  Avg pred boxes: 1.9, Avg GT boxes: 2.0
```

## 总结

可用性自检是防止"假训练"的关键机制：

- ✅ **真实验证**：使用实际解析逻辑，不依赖 reward 函数
- ✅ **早期发现**：及时发现格式崩溃、坐标错误等问题
- ✅ **可视化对比**：Reward vs Usability 图表直观展示训练质量
- ✅ **低开销**：每 200 步仅需 10-30 秒

**建议**：始终启用可用性自检，优先关注 usability 指标而非 reward。
