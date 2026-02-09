# 模型评估指南

本项目提供了完整的目标检测评估工具，支持计算标准检测指标。

## 评估指标

### 支持的指标

1. **Precision (精确率)**: TP / (TP + FP)
   - 预测为正例中真正为正例的比例
   - 衡量模型预测的准确性

2. **Recall (召回率)**: TP / (TP + FN)
   - 真实正例中被正确预测的比例
   - 衡量模型的检测完整性

3. **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
   - Precision 和 Recall 的调和平均
   - 综合评估模型性能

4. **mAP (mean Average Precision)**
   - 多个 IoU 阈值下的平均精确率
   - 目标检测的标准评估指标

5. **IoU (Intersection over Union)**
   - 预测框和真实框的重叠度
   - 用于判断检测是否正确

6. **Per-Class Metrics**
   - 每个类别的独立指标
   - 分析不同类别的检测性能

## 快速开始

### 1. 评估单个模型

```bash
# 使用 shell 脚本（推荐）
./scripts/run/evaluate.sh

# 自定义参数
MODEL_PATH=outputs/qwen3vl_lora \
TEST_DATA=data/qwen_data/test.json \
OUTPUT_DIR=eval_results/lora \
./scripts/run/evaluate.sh

# 或直接使用 Python
python scripts/evaluate.py \
    --model_path outputs/qwen3vl_lora \
    --test_data data/qwen_data/test.json \
    --output_dir eval_results/lora \
    --iou_threshold 0.5
```

### 2. 计算 mAP

```bash
# 计算多个 IoU 阈值的 mAP
COMPUTE_MAP=true ./scripts/run/evaluate.sh

# 或指定自定义阈值
python scripts/evaluate.py \
    --model_path outputs/qwen3vl_lora \
    --test_data data/qwen_data/test.json \
    --iou_thresholds 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 \
    --output_dir eval_results/
```

### 3. 可视化结果

```bash
# 可视化单个模型的评估结果
./scripts/run/visualize_eval.sh

# 自定义参数
EVAL_DIR=eval_results/lora \
OUTPUT=plots/lora_eval.png \
./scripts/run/visualize_eval.sh

# 生成所有类型的图表
PLOT_TYPE=all ./scripts/run/visualize_eval.sh
```

### 4. 比较多个模型

```bash
# 比较多个模型的性能
./scripts/run/compare_models.sh

# 自定义模型和标签
EVAL_DIRS="eval_results/base eval_results/lora eval_results/grpo" \
LABELS="Base LoRA GRPO" \
OUTPUT=plots/comparison.png \
./scripts/run/compare_models.sh
```

## 详细用法

### 评估脚本参数

```bash
python scripts/evaluate.py \
    --model_path <模型路径> \
    --test_data <测试数据路径> \
    --output_dir <输出目录> \
    --iou_threshold <IoU阈值> \
    --iou_thresholds <多个IoU阈值> \
    --max_samples <最大样本数> \
    --no_4bit  # 禁用4-bit量化
```

**参数说明**:
- `--model_path`: 模型路径（基础模型或微调后的 LoRA 模型）
- `--test_data`: 测试数据 JSON 文件路径（Qwen-VL 格式）
- `--output_dir`: 评估结果保存目录
- `--iou_threshold`: 单个 IoU 阈值（默认 0.5）
- `--iou_thresholds`: 多个 IoU 阈值，用于计算 mAP
- `--max_samples`: 限制评估样本数量（用于快速测试）
- `--no_4bit`: 禁用 4-bit 量化（需要更多显存）

### 可视化脚本参数

```bash
python scripts/visualize_eval_results.py \
    --eval_dir <评估结果目录> \
    --output <输出图片路径> \
    --iou_threshold <IoU阈值> \
    --plot_type <图表类型>
```

**图表类型**:
- `metrics_by_iou`: 不同 IoU 阈值下的指标曲线
- `per_class`: 每个类别的指标柱状图
- `confusion`: TP/FP/FN 统计和指标雷达图
- `all`: 生成所有类型的图表（默认）

### 模型比较参数

```bash
python scripts/visualize_eval_results.py \
    --eval_dirs <多个评估目录> \
    --labels <模型标签> \
    --output <输出图片路径> \
    --iou_threshold <IoU阈值>
```

## 输出文件

### 评估结果文件

评估完成后，会在输出目录生成以下文件：

```
eval_results/
├── eval_results_iou0.50.json    # IoU=0.5 的详细结果
├── eval_results_iou0.75.json    # IoU=0.75 的详细结果
├── eval_results_iou0.90.json    # IoU=0.9 的详细结果
└── eval_summary.json             # 总结（包含 mAP）
```

### 结果文件格式

```json
{
  "iou_threshold": 0.5,
  "overall": {
    "precision": 0.8523,
    "recall": 0.7891,
    "f1_score": 0.8193,
    "tp": 342,
    "fp": 59,
    "fn": 91,
    "iou_mean": 0.7234,
    "iou_std": 0.1456
  },
  "per_class": {
    "交通信号灯": {
      "precision": 0.9012,
      "recall": 0.8456,
      "f1_score": 0.8725,
      "tp": 123,
      "fp": 13,
      "fn": 22
    },
    "交通诱导屏": {
      "precision": 0.8234,
      "recall": 0.7123,
      "f1_score": 0.7634,
      "tp": 89,
      "fp": 19,
      "fn": 36
    }
  }
}
```

## 评估流程

### 完整评估流程示例

```bash
# 1. 准备测试数据
# 确保测试数据格式正确（Qwen-VL 格式）

# 2. 评估基础模型
MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct \
TEST_DATA=data/qwen_data/test.json \
OUTPUT_DIR=eval_results/base \
COMPUTE_MAP=true \
./scripts/run/evaluate.sh

# 3. 评估 LoRA 微调模型
MODEL_PATH=outputs/qwen3vl_lora \
TEST_DATA=data/qwen_data/test.json \
OUTPUT_DIR=eval_results/lora \
COMPUTE_MAP=true \
./scripts/run/evaluate.sh

# 4. 评估 GRPO 微调模型
MODEL_PATH=outputs/qwen3vl_grpo \
TEST_DATA=data/qwen_data/test.json \
OUTPUT_DIR=eval_results/grpo \
COMPUTE_MAP=true \
./scripts/run/evaluate.sh

# 5. 可视化各个模型的结果
EVAL_DIR=eval_results/base OUTPUT=plots/base_eval.png ./scripts/run/visualize_eval.sh
EVAL_DIR=eval_results/lora OUTPUT=plots/lora_eval.png ./scripts/run/visualize_eval.sh
EVAL_DIR=eval_results/grpo OUTPUT=plots/grpo_eval.png ./scripts/run/visualize_eval.sh

# 6. 比较所有模型
EVAL_DIRS="eval_results/base eval_results/lora eval_results/grpo" \
LABELS="Base LoRA GRPO" \
OUTPUT=plots/model_comparison.png \
./scripts/run/compare_models.sh
```

## 评估指标解读

### Precision vs Recall 权衡

- **高 Precision, 低 Recall**: 模型保守，只预测有把握的目标，漏检较多
- **低 Precision, 高 Recall**: 模型激进，预测很多目标，误检较多
- **高 Precision, 高 Recall**: 理想情况，模型性能优秀
- **F1-Score**: 综合考虑两者，适合作为主要评估指标

### IoU 阈值选择

- **IoU=0.5**: 宽松标准，常用于初步评估
- **IoU=0.75**: 严格标准，要求更精确的定位
- **IoU=0.5:0.95**: COCO 标准，计算 mAP@[.5:.95]

### Per-Class 分析

通过每个类别的指标，可以发现：
- 哪些类别检测效果好/差
- 是否存在类别不平衡问题
- 需要针对性优化的类别

## 常见问题

### Q: 评估速度慢怎么办？

```bash
# 使用较少的样本快速测试
MAX_SAMPLES=50 ./scripts/run/evaluate.sh

# 或直接指定
python scripts/evaluate.py \
    --model_path outputs/qwen3vl_lora \
    --test_data data/qwen_data/test.json \
    --max_samples 50
```

### Q: 显存不足怎么办？

```bash
# 禁用 4-bit 量化可能需要更多显存，建议保持启用
# 如果仍然不足，可以减小图片尺寸（需要修改代码中的 max_image_size）
```

### Q: 如何评估特定类别？

目前评估脚本会自动计算所有类别的指标。如需只评估特定类别，可以：
1. 过滤测试数据，只保留特定类别的样本
2. 查看输出的 per_class 结果

### Q: mAP 和 Precision 有什么区别？

- **Precision**: 单个 IoU 阈值下的精确率
- **mAP**: 多个 IoU 阈值下的平均精确率，更全面地评估模型性能

## 进阶用法

### 自定义 IoU 阈值

```python
# 在代码中修改 DetectionEvaluator 的 iou_threshold
evaluator = DetectionEvaluator(iou_threshold=0.6)
```

### 添加新的评估指标

可以在 `scripts/evaluate.py` 中扩展 `EvalMetrics` 类：

```python
@dataclass
class EvalMetrics:
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    # 添加新指标
    specificity: float = 0.0
    accuracy: float = 0.0
```

### 导出详细的预测结果

修改评估脚本，保存每个样本的预测和真实标注，用于错误分析。

## 参考资料

- [COCO Detection Evaluation](https://cocodataset.org/#detection-eval)
- [Object Detection Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)
- [mAP Calculation](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)
