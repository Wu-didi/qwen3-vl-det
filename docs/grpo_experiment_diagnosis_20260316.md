# GRPO 实验排查记录（2026-03-16）

## 现象

- `qwen3vl4b_lora` 的 `map50_95 = 0.3550`，明显高于 GRPO：
  - `qwen3vl4b_grpo_trl_final`: `map50_95 = 0.1453`
  - `qwen3vl4b_grpo_trl_exp3_final`: `map50_95 = 0.1487`
  - `qwen3vl8b_grpo_trl_exp3_final`: `map50_95 = 0.1525`
- GRPO 模型平均输出极短：
  - LoRA `avg_response_tokens = 246.09`
  - 4B GRPO `avg_response_tokens = 20.74 / 20.88`
  - 8B GRPO `avg_response_tokens = 23.57`
- GRPO 明显少报框：
  - LoRA `avg_pred_boxes = 2.23`
  - 4B GRPO `avg_pred_boxes ≈ 1.61`
  - 8B GRPO `avg_pred_boxes ≈ 1.56`
- `backpack_box` 在 GRPO 中几乎完全不出框：
  - LoRA `num_pred = 411`
  - 4B GRPO `num_pred = 0`
  - 8B GRPO `num_pred = 1`

## 根因

### 1. `risk_aware` 奖励函数只支持旧 `<box>` 文本格式

旧实现里，`scripts/training/rft/rewarding.py` 的格式门控和 GT 解析只认：

- `1. 类别`
- `状态：...`
- `<box>(x1,y1),(x2,y2)</box>`

但当前仓库同时存在新 JSON 检测格式：

```json
{"detections":[{"device_type":"...","state":"...","bbox_1000":[...]}]}
```

一旦 GRPO 训练数据或模型输出使用 JSON 格式，正样本大面积被判成“格式无效”，奖励直接变成 0。

这会把策略推向一个错误最优解：

- 少输出
- 输出空结果
- 输出很短的保守答案

和当前日志完全一致。

### 2. 数据加载和 `reward_scheme` 被错误绑定

旧逻辑里：

- `reward_scheme == new_json` 才走 `load_grpo_dataset`
- 其他 reward（包括 `risk_aware`）强制走 `load_and_prepare_dataset`

这意味着只要选了 `risk_aware`，即使你给的是结构化 `ground_truth` 数据，也不会走正确的数据链路。

### 3. 训练日志已经体现了奖励失真

- `logs/qwen3vl4b_grpo_trl/training_log.json`
  - `882` 个 reward 记录里，`99.32%` 为 `0`
- `logs/qwen3vl8b_grpo_trl_exp3/training_log.json`
  - `882` 个 reward 记录里，`100%` 为 `0`

这不是“RL 还没调好”，而是奖励链路本身已经失真。

## 已修复

### 1. `risk_aware` 奖励同时支持两种格式

已修改 `scripts/training/rft/rewarding.py`：

- 支持从 JSON completion 中解析 `detections`
- 支持从 JSON assistant / `ground_truth` 中解析 GT
- JSON 空检测 `{"detections":[]}` 现在会被识别为合法的 no-object 响应
- 类别匹配增加了中英文别名归一化，兼容旧文本和新结构化格式混用

### 2. 数据格式选择改为独立配置

已修改 `scripts/training/rft/grpo_finetune_trl.py`：

- 新增 `--data_format auto|conversation|grpo`
- 默认 `auto`，根据样本字段自动识别
- `reward_scheme` 不再决定数据加载器
- `new_json` 奖励如果配错数据格式，会直接报错而不是静默训练

### 3. 旧 conversation 数据也会尽量提取结构化 GT

已修改 `scripts/training/rft/data_utils.py`：

- `load_and_prepare_dataset` 现在支持 JSON / JSONL
- 会尝试从 assistant 文本中提取 JSON `ground_truth`
- collator 会把 `ground_truth` 一并传给 reward 函数

### 4. 训练脚本默认值更稳

已修改 `scripts/run/train_grpo_trl.sh`：

- 增加 `DATA_FORMAT=auto`
- 默认优先使用 `data/hefei_last_dataset/rft_output/*.jsonl`
- 如果结构化数据不存在，再回退到旧 `qwen_data/*.json`

## 回归验证

新增测试：

- `tests/test_grpo_rewarding.py`

覆盖了以下回归场景：

- JSON completion + JSON reference 可以拿到正奖励
- JSON 空检测会被当成合法 no-object 响应
- 旧 `<box>` completion 可以和结构化 `ground_truth` 正常匹配

## 建议的下一轮实验

建议先重跑一版最小修复验证，不要同时继续堆 reward 权重：

```bash
DATA_FORMAT=auto \
REWARD_SCHEME=risk_aware \
TRAIN_DATA=data/hefei_last_dataset/rft_output/train.jsonl \
VAL_DATA=data/hefei_last_dataset/rft_output/val.jsonl \
bash scripts/run/train_grpo_trl.sh
```

优先观察三件事：

- 训练中 `reward=0` 的比例是否明显下降
- `avg_response_tokens` 是否回升到至少能完整输出多框 JSON
- `avg_pred_boxes` 是否接近 GT 平均框数
