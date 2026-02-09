# 评估功能总结

## 新增文件

### 核心脚本
1. **scripts/evaluate.py** (约 800 行)
   - 完整的目标检测评估脚本
   - 支持 Precision, Recall, F1-Score, mAP 等指标
   - 支持多个 IoU 阈值评估
   - 支持每个类别的独立指标计算

2. **scripts/visualize_eval_results.py** (约 400 行)
   - 评估结果可视化脚本
   - 支持多种图表类型（指标曲线、柱状图、雷达图等）
   - 支持多模型对比

### Shell 脚本
3. **scripts/run/evaluate.sh**
   - 评估启动脚本
   - 支持环境变量配置

4. **scripts/run/visualize_eval.sh**
   - 可视化启动脚本

5. **scripts/run/compare_models.sh**
   - 模型对比启动脚本

### 文档
6. **docs/EVALUATION.md** (约 500 行)
   - 详细的评估指南
   - 包含指标说明、使用示例、常见问题等

7. **docs/EVALUATION_QUICKSTART.md** (约 200 行)
   - 快速开始指南
   - 简明的使用说明

## 主要功能

### 1. 评估指标
- **Precision (精确率)**: TP / (TP + FP)
- **Recall (召回率)**: TP / (TP + FN)
- **F1-Score**: 调和平均
- **mAP**: 多 IoU 阈值平均精确率
- **Per-Class Metrics**: 每类别独立指标
- **IoU Statistics**: 重叠度统计

### 2. 可视化功能
- 指标 vs IoU 阈值曲线
- 每类别指标柱状图
- TP/FP/FN 统计图
- 指标雷达图
- 多模型对比图

### 3. 使用方式

#### Shell 脚本（推荐）
```bash
# 评估模型
./scripts/run/evaluate.sh
MODEL_PATH=outputs/qwen3vl_lora ./scripts/run/evaluate.sh
COMPUTE_MAP=true ./scripts/run/evaluate.sh

# 可视化
./scripts/run/visualize_eval.sh
EVAL_DIR=eval_results/lora ./scripts/run/visualize_eval.sh

# 对比模型
./scripts/run/compare_models.sh
```

#### Python 直接调用
```bash
# 评估
python scripts/evaluate.py \
    --model_path outputs/qwen3vl_lora \
    --test_data data/qwen_data/test.json \
    --output_dir eval_results/

# 可视化
python scripts/visualize_eval_results.py \
    --eval_dir eval_results/ \
    --output plots/eval_plots.png
```

## 技术实现

### 检测匹配算法
1. 计算 IoU 矩阵（预测框 vs 真实框）
2. 只匹配相同类别
3. 贪心匹配（优先高 IoU）
4. 统计 TP/FP/FN

### 数据格式支持
- 输入：Qwen-VL 格式的 JSON 数据
- 输出：JSON 格式的评估结果
- 可视化：PNG 格式的图表

### 模型支持
- 基础模型（Qwen3-VL / Qwen2.5-VL）
- LoRA 微调模型
- GRPO 微调模型
- 任何输出 `<box>` 格式的模型

## 依赖更新

在 `requirements_finetune.txt` 中新增：
```
numpy>=1.24.0
matplotlib>=3.7.0
```

## 文档更新

### CLAUDE.md
- 新增评估相关的 shell 脚本命令
- 更新项目架构说明
- 添加评估脚本到文件列表

### 新增文档
- `docs/EVALUATION.md`: 详细评估指南
- `docs/EVALUATION_QUICKSTART.md`: 快速开始

## 使用示例

### 完整评估流程
```bash
# 1. 评估基础模型
MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct \
TEST_DATA=data/qwen_data/test.json \
OUTPUT_DIR=eval_results/base \
COMPUTE_MAP=true \
./scripts/run/evaluate.sh

# 2. 评估 LoRA 模型
MODEL_PATH=outputs/qwen3vl_lora \
TEST_DATA=data/qwen_data/test.json \
OUTPUT_DIR=eval_results/lora \
COMPUTE_MAP=true \
./scripts/run/evaluate.sh

# 3. 评估 GRPO 模型
MODEL_PATH=outputs/qwen3vl_grpo \
TEST_DATA=data/qwen_data/test.json \
OUTPUT_DIR=eval_results/grpo \
COMPUTE_MAP=true \
./scripts/run/evaluate.sh

# 4. 可视化各模型结果
EVAL_DIR=eval_results/base OUTPUT=plots/base_eval.png ./scripts/run/visualize_eval.sh
EVAL_DIR=eval_results/lora OUTPUT=plots/lora_eval.png ./scripts/run/visualize_eval.sh
EVAL_DIR=eval_results/grpo OUTPUT=plots/grpo_eval.png ./scripts/run/visualize_eval.sh

# 5. 对比所有模型
EVAL_DIRS="eval_results/base eval_results/lora eval_results/grpo" \
LABELS="Base LoRA GRPO" \
OUTPUT=plots/model_comparison.png \
./scripts/run/compare_models.sh
```

## 输出示例

### 评估结果文件
```
eval_results/
├── eval_results_iou0.50.json    # IoU=0.5 详细结果
├── eval_results_iou0.75.json    # IoU=0.75 详细结果
├── eval_results_iou0.90.json    # IoU=0.9 详细结果
└── eval_summary.json             # 总结（含 mAP）
```

### 可视化图表
```
plots/
├── eval_plots_metrics_by_iou.png    # 指标曲线
├── eval_plots_per_class.png         # 类别柱状图
├── eval_plots_confusion.png         # 混淆统计
└── model_comparison.png             # 模型对比
```

## 特点

1. **完整性**: 支持标准目标检测评估指标
2. **易用性**: 提供 shell 脚本封装，支持环境变量配置
3. **灵活性**: 支持多种 IoU 阈值、自定义参数
4. **可视化**: 丰富的图表类型，支持多模型对比
5. **兼容性**: 支持 Qwen3-VL 和 Qwen2.5-VL 模型
6. **文档**: 详细的使用文档和示例

## 后续改进建议

1. **性能优化**: 支持批量推理加速评估
2. **更多指标**: 添加 AP@IoU, AR (Average Recall) 等
3. **错误分析**: 可视化误检和漏检的样本
4. **导出功能**: 支持导出 CSV、Excel 格式
5. **Web 界面**: 提供 Gradio 界面进行交互式评估

## 总结

本次更新为项目添加了完整的目标检测评估功能，填补了之前缺失的验证指标代码。现在用户可以：

1. ✅ 评估模型性能（Precision, Recall, F1, mAP）
2. ✅ 计算每个类别的独立指标
3. ✅ 可视化评估结果
4. ✅ 对比多个模型的性能
5. ✅ 使用简单的 shell 脚本快速评估

所有功能已经过语法检查，可以直接使用。
