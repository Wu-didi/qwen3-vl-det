# 交通设备异常检测服务

基于 Qwen3-VL / Qwen2.5-VL 视觉语言模型的交通设备异常检测服务，提供 REST API 和 Web 界面。

## 功能特性

- **多类型检测**：支持 6 大类交通设备异常检测
  - 交通标志：损坏、缺失、遮挡、褪色、倾斜
  - 交通信号灯：故障、灯泡损坏、不亮、颜色异常
  - 道路设施：护栏损坏、路面破损、标线磨损、井盖缺失
  - 诱导屏：显示故障、黑屏、乱码
  - 限高架：结构损坏、标识不清
  - 机柜：柜门未关闭、损坏

- **多模型支持**：自动适配 Qwen3-VL 和 Qwen2.5-VL 系列模型
- **多输入方式**：Base64 编码、图片 URL、文件上传
- **结构化输出**：JSON 格式结果，包含类别、置信度、边界框坐标和描述
- **可视化**：在图片上绘制检测框和标签
- **Web 界面**：基于 Gradio 的交互式检测界面

## 支持的模型

| 模型 | 显存需求 | 说明 |
|------|---------|------|
| Qwen/Qwen3-VL-2B-Instruct | ~6GB | 轻量版，适合 RTX 3060/4060 |
| Qwen/Qwen3-VL-4B-Instruct | ~10GB | 小型版，适合 RTX 3070/4070 |
| Qwen/Qwen3-VL-8B-Instruct | ~18GB | 标准版，适合 RTX 3090/4090 |
| Qwen/Qwen2.5-VL-3B-Instruct | ~8GB | Qwen2.5 轻量版 |
| Qwen/Qwen2.5-VL-7B-Instruct | ~16GB | Qwen2.5 标准版 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt

# Gradio 界面需要额外安装
pip install gradio
```

### 2. 下载模型

**方式一：使用脚本下载（需要网络访问 Hugging Face）**

```bash
python scripts/download_model.py --model Qwen/Qwen3-VL-2B-Instruct
```

**方式二：手动下载**

从 [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct) 或 [Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) 下载模型到 `model_cache/Qwen/Qwen3-VL-2B-Instruct/` 目录。

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```bash
# 使用预定义模型（需网络下载）
MODEL_TYPE=Qwen/Qwen3-VL-2B-Instruct

# 或使用本地模型路径
MODEL_TYPE=custom
MODEL_NAME=./model_cache/Qwen/Qwen3-VL-2B-Instruct

# 其他配置
MODEL_CACHE_DIR=./model_cache
DEVICE=cuda
TORCH_DTYPE=bfloat16
CONFIDENCE_THRESHOLD=0.5
```
export CUDA_VISIBLE_DEVICES=7
### 4. 启动服务

**REST API 服务**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Gradio Web 界面**

```bash
python gradio_app.py
# 访问 http://localhost:7860
```

### 5. Docker 部署

```bash
docker-compose up -d
```

## 训练（SFT / GRPO）

### 1. 安装训练依赖

```bash
pip install -r requirements_finetune.txt
```

> `requirements_finetune.txt` 已包含 `trl`，用于 `GRPOTrainer` 训练。

### 2. 数据格式要求（重要）

- 训练数据采用 Qwen-VL 对话格式（`image + conversations`），可用 `scripts/data/cvat_to_qwenvl.py` 转换。
- 对于“图中无相关设备”的样本，`assistant` 建议明确输出：`未检测到相关设备。`
- GRPO 奖励已使用严格格式门控：格式不合法时，其它奖励项不会生效。

### 3. SFT（LoRA/QLoRA）

```bash
./scripts/run/train_lora.sh
```

### 4. GRPO（推荐 TRL 版本）

```bash
./scripts/run/train_grpo_trl.sh
```

可选：自定义实现版本

```bash
./scripts/run/train_grpo.sh
```

### 5. 训练日志可视化

```bash
python scripts/visualize_training_log.py \
  --log outputs/qwen3vl_grpo_trl/training_log.json \
  --output outputs/qwen3vl_grpo_trl/plots
```

## 评估（可用于论文）

检测评估与 VLM 质量评估统一在 `scripts/evaluate.py` 中，结果会写入 `eval_summary.json`。

```bash
python scripts/evaluate.py \
  --model_path outputs/qwen3vl_lora \
  --test_data data/qwen_data/test.json \
  --coco_map \
  --output_dir eval_results/lora
```

支持的核心指标：
- 检测指标：`AP50` / `AP75` / `mAP50-95`（COCO/YOLO 风格）
- VLM 指标：`parse_success_rate`、`strict_format_rate`、`hallucination_rate`
- 论文常用补充：`exact_match_rate`、`token_f1`、`ROUGE-L F1`、`BLEU-1/4`
- 任务相关补充：`anomaly_precision/recall/f1`、`count_mae/rmse`、时延 `latency_ms_mean/p50/p95`

详细说明见：`docs/EVALUATION.md`

## 使用方式

### Web 界面（推荐新手使用）

启动 Gradio 界面后访问 http://localhost:7860

1. 上传图片
2. （可选）选择检测类别
3. 调整置信度阈值
4. 点击「开始检测」
5. 查看检测结果和可视化

### REST API

#### 健康检查

```bash
curl http://localhost:8000/api/v1/health
```

#### 文件上传检测

```bash
curl -X POST "http://localhost:8000/api/v1/detect/upload" \
  -F "file=@test_image.jpg"
```

#### JSON 请求检测

```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "categories": ["traffic_sign", "traffic_light"],
    "quick_mode": false
  }'
```

#### 检测并返回可视化图片

```bash
curl -X POST "http://localhost:8000/api/v1/detect/visualize" \
  -F "file=@test_image.jpg" \
  -o result.jpg
```

#### 批量检测

```bash
curl -X POST "http://localhost:8000/api/v1/detect/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      {"image_url": "https://example.com/1.jpg", "image_id": "img1"},
      {"image_url": "https://example.com/2.jpg", "image_id": "img2"}
    ]
  }'
```

## 响应格式

```json
{
  "has_anomaly": true,
  "detections": [
    {
      "category": "traffic_sign",
      "anomaly_type": "damaged",
      "confidence": 0.85,
      "bbox": [120, 80, 350, 280],
      "description": "交通标志牌右上角有明显破损"
    }
  ],
  "summary": "检测到1处交通标志异常",
  "image_id": "test_image",
  "processing_time_ms": 1234.5
}
```

**说明**：
- `bbox`: 边界框坐标 `[x1, y1, x2, y2]`，归一化到 0-1000 范围
- `confidence`: 置信度，范围 0-1
- `category`: 检测类别（traffic_sign, traffic_light, road_facility, guidance_screen, height_limit, cabinet）

## 配置项

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `MODEL_TYPE` | 预定义模型类型 | `Qwen/Qwen3-VL-8B-Instruct` |
| `MODEL_NAME` | 自定义模型路径（设置后覆盖 MODEL_TYPE） | - |
| `MODEL_CACHE_DIR` | 模型缓存目录 | `./model_cache` |
| `DEVICE` | 推理设备 | `cuda` |
| `TORCH_DTYPE` | 模型精度（float16/bfloat16/float32） | `bfloat16` |
| `MAX_NEW_TOKENS` | 最大生成 token 数 | `2048` |
| `CONFIDENCE_THRESHOLD` | 置信度过滤阈值 | `0.5` |
| `MAX_IMAGE_SIZE` | 图片最大边长（超过会缩放） | `1280` |
| `API_HOST` | API 监听地址 | `0.0.0.0` |
| `API_PORT` | API 监听端口 | `8000` |
| `LOG_LEVEL` | 日志级别 | `INFO` |

## 项目结构

```
qwen3-vl-det/
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── api/routes.py        # API 路由定义
│   ├── core/
│   │   ├── config.py        # 配置管理（Pydantic Settings）
│   │   └── detector.py      # 检测器核心逻辑
│   ├── models/
│   │   ├── qwen_vl.py       # Qwen-VL 模型封装（支持 Qwen3/2.5）
│   │   └── prompts.py       # 检测提示词模板
│   ├── schemas/detection.py # Pydantic 数据模型
│   └── utils/
│       ├── image_utils.py   # 图片加载工具
│       └── visualization.py # 检测结果可视化
├── gradio_app.py            # Gradio Web 界面
├── scripts/
│   ├── download_model.py    # 模型下载脚本
│   └── benchmark.py         # 性能测试脚本
├── tests/                   # 测试用例
├── requirements.txt         # Python 依赖
├── .env.example             # 环境变量示例
├── Dockerfile
└── docker-compose.yml
```

## 硬件要求

| 模型大小 | 最低显存 | 推荐 GPU |
|---------|---------|----------|
| 2B | 6GB | RTX 3060, RTX 4060 |
| 4B | 10GB | RTX 3070, RTX 4070 |
| 7-8B | 16-18GB | RTX 3090, RTX 4090 |
| 32B+ | 40GB+ | A100, 多卡并行 |

- **内存**：16GB+（推荐 32GB）
- **存储**：10-50GB（取决于模型大小）

## 常见问题

### Q: 无法连接 Hugging Face 下载模型？

使用本地模型路径：
```bash
MODEL_TYPE=custom
MODEL_NAME=./model_cache/Qwen/Qwen3-VL-2B-Instruct
```

### Q: 显存不足？

1. 使用更小的模型（如 2B 版本）
2. 设置 `TORCH_DTYPE=float16`
3. 减小 `MAX_IMAGE_SIZE`

### Q: 检测结果不准确？

1. 调低 `CONFIDENCE_THRESHOLD`
2. 使用更大的模型
3. 确保图片清晰度足够

## License

MIT License
