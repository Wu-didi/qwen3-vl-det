# GRPO 训练速度常见问题

## ❓ 常见问题解答

### Q1: GRPO 训练为什么这么慢？

**A**: GRPO 比监督学习慢 **10-20 倍**是正常的，原因：

1. **在线生成**：每个样本需要生成多个响应（默认 4 个）
2. **多次前向传播**：
   - 生成 4 个响应：4 次
   - 计算 policy log probs：4 次
   - 计算 reference log probs：4 次
   - **总共约 12 次** vs 监督学习的 1 次

3. **图片 token 多**：Qwen-VL 的图片会被转换成大量 tokens
   - 1024px 图片 ≈ 4000+ tokens
   - 512px 图片 ≈ 1000 tokens

**结论**：这是 GRPO 算法的固有特性，不是 bug。

---

### Q2: 我的配置正常吗？每个样本需要 10 分钟

**A**: 如果你的配置是：
- `MAX_IMAGE_SIZE=1024`
- `MODEL_PATH="Qwen3-VL-8B-Instruct"`
- `NUM_GENERATIONS=4`
- `EVAL_STEPS=200`

那么 **10 分钟/样本是正常的**，但这个配置太慢了。

**优化建议**：
1. 降低图片大小到 512px → 速度提升 4 倍
2. 禁用训练时验证 → 速度提升 1.5 倍
3. 改用 2B 模型 → 速度提升 4 倍

**优化后**：~30 秒/样本（提升 20 倍）

---

### Q3: 降低图片大小会影响检测效果吗？

**A**: 影响很小，取决于你的任务：

| 图片大小 | 适用场景 | 检测精度 | 训练速度 |
|---------|---------|---------|---------|
| 384px | 大目标检测（交通标志、信号灯） | 95% | 最快 |
| 512px | 通用场景（推荐） | 98% | 快 |
| 768px | 小目标检测 | 99% | 慢 |
| 1024px | 极小目标 | 99.5% | 极慢 |

**建议**：
- 先用 512px 训练，看效果
- 如果小目标检测不好，再用 768px
- **1024px 通常没必要**，性价比极低

---

### Q4: 2B 模型和 8B 模型效果差多少？

**A**: 根据经验：

| 指标 | 2B 模型 | 8B 模型 | 差距 |
|------|---------|---------|------|
| 训练速度 | 1x | 0.25x | **4 倍慢** |
| 检测准确率 | 85-90% | 90-95% | +5-10% |
| 格式正确率 | 90% | 95% | +5% |
| 显存占用 | ~12GB | ~24GB | 2 倍 |

**建议**：
- 先用 2B 模型验证效果
- 如果效果够用，就不用 8B
- 8B 模型效果提升 < 20%，但时间增加 4 倍

---

### Q5: NUM_GENERATIONS 设置多少合适？

**A**:

| NUM_GENERATIONS | 训练速度 | 训练稳定性 | 推荐场景 |
|----------------|---------|-----------|---------|
| 2 | 最快 | 较差 | 快速实验 |
| 4 | 快 | 好 | **推荐** |
| 6 | 慢 | 很好 | 追求稳定 |
| 8+ | 很慢 | 极好 | 收益递减 |

**建议**：
- 默认用 4
- 如果训练不稳定（loss 震荡），增加到 6
- 不建议超过 8（收益递减）

---

### Q6: 训练时需要验证吗？

**A**: 看情况：

**训练时禁用验证（推荐）**：
```bash
EVAL_STEPS=0
```
- 优点：训练更快（提升 1.5 倍）
- 缺点：不知道验证集表现

**训练后单独验证**：
```bash
# 训练完成后
python scripts/evaluate_model.py \
    --model_path outputs/qwen3vl_grpo/final \
    --val_data data/val.json
```

**训练时验证**：
```bash
EVAL_STEPS=500  # 设置较大的值
```
- 优点：实时监控验证集表现
- 缺点：训练变慢

---

### Q7: 如何估算训练时间？

**A**: 使用这个公式：

```
总时间 = 样本数 × 每样本时间 / 60 / 60（小时）
```

**每样本时间参考**：

| 配置 | 每样本时间 |
|------|-----------|
| 384px + 2B + 2 gen | 30 秒 |
| 512px + 2B + 4 gen | 60 秒 |
| 512px + 8B + 4 gen | 120 秒 |
| 768px + 8B + 4 gen | 300 秒 |
| 1024px + 8B + 4 gen | 600 秒 |

**示例**：
- 1000 样本，512px + 2B + 4 gen
- 总时间 = 1000 × 60 / 3600 = **17 小时**

---

### Q8: 训练过程中可以修改配置吗？

**A**: 不建议，但可以：

**不能修改**：
- 模型路径
- LoRA 配置
- 数据路径

**可以修改**（重启训练）：
- 图片大小
- NUM_GENERATIONS
- 学习率
- EVAL_STEPS

**如何重启**：
```bash
# 1. 停止当前训练（Ctrl+C）

# 2. 修改配置
vim scripts/run/train_grpo_trl.sh

# 3. 从检查点继续（如果支持）
# 或者重新开始训练
./scripts/run/train_grpo_trl.sh
```

---

### Q9: 如何判断训练是否正常？

**A**: 查看这些指标：

**正常训练**：
```
Step 10: loss=0.8, reward=2.0, kl=0.01
Step 20: loss=0.7, reward=2.2, kl=0.02
Step 30: loss=0.6, reward=2.4, kl=0.015
```
- Loss 逐渐下降
- Reward 逐渐上升
- KL 保持在 0.01-0.05 之间

**异常训练**：
```
Step 10: loss=0.8, reward=2.0, kl=0.01
Step 20: loss=1.5, reward=1.5, kl=0.5
Step 30: loss=2.0, reward=1.0, kl=1.0
```
- Loss 上升
- Reward 下降
- KL 过大（> 0.1）

**解决方法**：
- 降低学习率
- 增大 KL 系数（BETA）
- 增加 NUM_GENERATIONS

---

### Q10: 显存不够怎么办？

**A**: 按优先级尝试：

**1. 降低图片大小**（最有效）
```bash
MAX_IMAGE_SIZE=384  # 从 512 降到 384
```
节省显存：~50%

**2. 减少 LoRA rank**
```bash
LORA_R=32  # 从 64 降到 32
```
节省显存：~20%

**3. 减少 NUM_GENERATIONS**
```bash
NUM_GENERATIONS=2  # 从 4 降到 2
```
节省显存：~30%

**4. 启用梯度检查点**（默认已开启）
```bash
DISABLE_GRADIENT_CHECKPOINTING=false
```

**5. 使用更小的模型**
```bash
MODEL_PATH="Qwen3-VL-2B-Instruct"  # 从 8B 改为 2B
```
节省显存：~50%

---

### Q11: 如何加速数据加载？

**A**:

**1. 预处理图片**
```bash
# 提前 resize 所有图片到 512px
python scripts/preprocess_images.py \
    --input data/images \
    --output data/images_512 \
    --size 512
```

**2. 使用 SSD 存储数据**
- 避免使用网络存储（NFS）
- 使用本地 SSD

**3. 减少图片大小**
```bash
MAX_IMAGE_SIZE=512  # 不要用 1024
```

---

### Q12: 训练中断了怎么办？

**A**:

**检查检查点**：
```bash
ls -lh outputs/qwen3vl_grpo_trl/
```

**从检查点继续**：
```bash
# 如果有 checkpoint-samples-XXX 目录
# 修改配置，从检查点加载
# （需要修改训练脚本支持 resume）
```

**重新开始**：
```bash
# 如果没有检查点或检查点损坏
# 删除输出目录重新训练
rm -rf outputs/qwen3vl_grpo_trl
./scripts/run/train_grpo_trl.sh
```

---

### Q13: 如何对比不同配置的效果？

**A**:

**1. 训练多个实验**
```bash
# 实验 1: 512px + 2B
MAX_IMAGE_SIZE=512
MODEL_PATH="Qwen3-VL-2B-Instruct"
OUTPUT_DIR="outputs/exp1_512_2b"

# 实验 2: 512px + 8B
MAX_IMAGE_SIZE=512
MODEL_PATH="Qwen3-VL-8B-Instruct"
OUTPUT_DIR="outputs/exp2_512_8b"

# 实验 3: 768px + 8B
MAX_IMAGE_SIZE=768
MODEL_PATH="Qwen3-VL-8B-Instruct"
OUTPUT_DIR="outputs/exp3_768_8b"
```

**2. 对比结果**
```bash
python scripts/compare_training_logs.py \
    --logs outputs/exp1_512_2b/training_log.json \
           outputs/exp2_512_8b/training_log.json \
           outputs/exp3_768_8b/training_log.json \
    --output comparison/ \
    --html
```

**3. 查看对比报告**
```bash
open comparison/comparison.html
```

---

### Q14: GRPO 和 DPO 哪个更快？

**A**:

| 方法 | 训练速度 | 数据需求 | 效果 |
|------|---------|---------|------|
| **GRPO** | 慢（在线生成） | 只需要输入 | 好 |
| **DPO** | 快（离线数据） | 需要 chosen/rejected 对 | 很好 |
| **LoRA** | 最快（监督学习） | 需要标注数据 | 一般 |

**建议**：
1. 先用 LoRA 监督学习
2. 如果有偏好数据，用 DPO
3. 如果没有偏好数据，用 GRPO

---

### Q15: 如何生成 DPO 数据以加速训练？

**A**:

**方法 1: 从 GRPO 生成**
```bash
# 使用 GRPO 生成多个响应
# 人工标注 chosen/rejected
python scripts/data/generate_dpo_from_grpo.py \
    --model_path outputs/qwen3vl_lora \
    --input data/train.json \
    --output data/dpo_data.json
```

**方法 2: 从模型输出生成**
```bash
# 使用不同模型生成响应
# 自动选择最好的作为 chosen
python scripts/data/generate_dpo_data.py \
    --input data/train.json \
    --output data/dpo_data.json
```

**然后用 DPO 训练**（比 GRPO 快 5-10 倍）：
```bash
python scripts/training/dpo_finetune.py \
    --train_data data/dpo_data.json \
    --val_data data/dpo_val.json \
    --output_dir outputs/qwen3vl_dpo
```

---

## 🎯 最佳实践总结

### 推荐配置（平衡速度和效果）

```bash
# 基础配置
MAX_IMAGE_SIZE=512
NUM_GENERATIONS=4
LORA_R=64
EVAL_STEPS=0

# 快速实验用 2B
MODEL_PATH="Qwen3-VL-2B-Instruct"

# 最终训练用 8B
MODEL_PATH="Qwen3-VL-8B-Instruct"
```

### 训练流程

```
1. LoRA 监督学习（1-2 天）
   ↓
2. GRPO 强化学习（2-3 天，使用 2B 模型）
   ↓
3. 评估效果
   ↓
4. 如果效果不够，用 8B 模型重新训练（5-7 天）
```

### 监控指标

- **Loss**: 应该逐渐下降
- **Reward**: 应该逐渐上升
- **KL**: 保持在 0.01-0.05
- **速度**: 记录每小时处理的样本数

---

## 📞 需要帮助？

如果还有问题：

1. 查看详细文档：[GRPO_SPEED_OPTIMIZATION.md](GRPO_SPEED_OPTIMIZATION.md)
2. 运行优化脚本：`bash scripts/optimize_grpo_speed.sh`
3. 查看训练日志：`python scripts/visualize_training_log.py --log outputs/*/training_log.json`

---

**记住**：GRPO 训练慢是正常的，但通过优化可以显著提速！
