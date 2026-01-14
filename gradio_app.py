"""Gradio web interface for traffic equipment anomaly detection."""

import json
import os
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

# 全局模型实例
_model = None
_processor = None
_current_model_name = None

# 基础模型列表
BASE_MODELS = {
    "Qwen3-VL-2B": "./model_cache/Qwen/Qwen3-VL-2B-Instruct",
    "Qwen3-VL-4B": "./model_cache/Qwen/Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B": "./model_cache/Qwen/Qwen3-VL-8B-Instruct",
    "Qwen2.5-VL-7B": "./model_cache/Qwen/Qwen2.5-VL-7B-Instruct",
}

# 微调模型目录
FINETUNED_MODELS_DIR = "./outputs"


def scan_finetuned_models() -> dict:
    """扫描 outputs 目录下的微调模型"""
    finetuned_models = {}

    if not os.path.exists(FINETUNED_MODELS_DIR):
        return finetuned_models

    for name in os.listdir(FINETUNED_MODELS_DIR):
        model_path = os.path.join(FINETUNED_MODELS_DIR, name)
        config_path = os.path.join(model_path, "finetune_config.json")
        adapter_config_path = os.path.join(model_path, "adapter_config.json")

        # 检查是否有 LoRA 配置文件
        if os.path.isdir(model_path) and (
            os.path.exists(config_path) or os.path.exists(adapter_config_path)
        ):
            # 使用 🔧 标识微调模型
            display_name = f"🔧 {name} (LoRA)"
            finetuned_models[display_name] = model_path

    return finetuned_models


def get_available_models() -> dict:
    """获取所有可用模型（基础模型 + 微调模型）"""
    models = BASE_MODELS.copy()
    models.update(scan_finetuned_models())
    return models


# 可用模型列表（动态扫描）
AVAILABLE_MODELS = get_available_models()

# 类别颜色
CATEGORY_COLORS = {
    "traffic_sign": (255, 0, 0),
    "traffic_light": (0, 255, 0),
    "road_facility": (0, 0, 255),
    "guidance_screen": (255, 165, 0),
    "height_limit": (128, 0, 128),
    "cabinet": (255, 192, 203),
}

CATEGORY_NAMES = {
    "traffic_sign": "交通标志",
    "traffic_light": "信号灯",
    "road_facility": "道路设施",
    "guidance_screen": "诱导屏",
    "height_limit": "限高架",
    "cabinet": "机柜",
}


def is_finetuned_model(model_choice: str) -> bool:
    """判断是否为微调模型"""
    return model_choice.startswith("🔧")


def get_base_model_path(finetuned_path: str) -> str:
    """从微调模型配置中获取基础模型路径"""
    config_path = os.path.join(finetuned_path, "finetune_config.json")

    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
                return config.get("model_path", "./model_cache/Qwen/Qwen3-VL-2B-Instruct")
        except (json.JSONDecodeError, KeyError):
            pass

    # 尝试从 adapter_config.json 读取
    adapter_config_path = os.path.join(finetuned_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        try:
            with open(adapter_config_path) as f:
                config = json.load(f)
                return config.get("base_model_name_or_path", "./model_cache/Qwen/Qwen3-VL-2B-Instruct")
        except (json.JSONDecodeError, KeyError):
            pass

    return "./model_cache/Qwen/Qwen3-VL-2B-Instruct"


def get_model_class(model_path: str):
    """根据模型路径返回对应的模型类"""
    from transformers import Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

    model_path_lower = model_path.lower()
    if "qwen3" in model_path_lower:
        return Qwen3VLForConditionalGeneration
    else:
        return Qwen2_5_VLForConditionalGeneration


def load_model(model_choice: str) -> str:
    """加载选定的模型（支持基础模型和微调模型）"""
    global _model, _processor, _current_model_name

    import torch
    from transformers import AutoProcessor

    # 刷新模型列表
    global AVAILABLE_MODELS
    AVAILABLE_MODELS = get_available_models()

    if model_choice not in AVAILABLE_MODELS:
        return f"❌ 未知模型: {model_choice}"

    model_path = AVAILABLE_MODELS[model_choice]

    # 如果已经加载了相同的模型，直接返回
    if _current_model_name == model_choice and _model is not None:
        return f"✅ 模型 {model_choice} 已加载，无需重复加载"

    # 释放旧模型
    if _model is not None:
        del _model
        del _processor
        _model = None
        _processor = None
        torch.cuda.empty_cache()

    try:
        if is_finetuned_model(model_choice):
            # 加载微调模型
            from peft import PeftModel

            base_model_path = get_base_model_path(model_path)
            print(f"Loading base model from: {base_model_path}")
            print(f"Loading LoRA weights from: {model_path}")

            # 加载基础模型
            model_class = get_model_class(base_model_path)
            base_model = model_class.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

            # 加载 LoRA 权重并合并
            _model = PeftModel.from_pretrained(base_model, model_path)
            _model = _model.merge_and_unload()

            # 优先从微调模型加载 processor，失败则从基础模型加载
            try:
                _processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            except Exception:
                _processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

            _current_model_name = model_choice
            return f"✅ 微调模型 {model_choice} 加载成功！\n   基础模型: {base_model_path}"

        else:
            # 加载基础模型
            model_class = get_model_class(model_path)
            _processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

            _model = model_class.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

            _current_model_name = model_choice
            return f"✅ 模型 {model_choice} 加载成功！"

    except Exception as e:
        _model = None
        _processor = None
        _current_model_name = None
        return f"❌ 加载失败: {str(e)}"


def get_model_status() -> str:
    """获取当前模型状态"""
    if _current_model_name:
        if is_finetuned_model(_current_model_name):
            return f"当前模型: {_current_model_name} (微调)"
        return f"当前模型: {_current_model_name}"
    return "未加载模型"


def refresh_model_list():
    """刷新模型列表"""
    global AVAILABLE_MODELS
    AVAILABLE_MODELS = get_available_models()
    choices = list(AVAILABLE_MODELS.keys())
    return gr.update(choices=choices)


def parse_box_format(text: str) -> dict:
    """
    解析 <box>(x1,y1),(x2,y2)</box> 格式的检测结果
    返回与 JSON 格式兼容的结构
    """
    import re
    detections = []

    # 按序号分割每个检测项
    items = re.split(r'(?=\d+\.\s+)', text)

    for item in items:
        if not item.strip():
            continue

        # 提取类别 (序号后面的第一行)
        cat_match = re.match(r'(\d+)\.\s*([^\n]+)', item)
        if not cat_match:
            continue

        category = cat_match.group(2).strip()

        # 提取状态
        status_match = re.search(r'状态[：:]\s*([^\n]+)', item)
        status = status_match.group(1).strip() if status_match else "正常"

        # 提取原因
        reason_match = re.search(r'原因[：:]\s*([^\n]+)', item)
        reason = reason_match.group(1).strip() if reason_match else ""

        # 提取坐标
        box_match = re.search(r'<box>\s*\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)\s*</box>', item)
        if not box_match:
            continue

        x1, y1, x2, y2 = int(box_match.group(1)), int(box_match.group(2)), int(box_match.group(3)), int(box_match.group(4))

        # 判断是否异常
        is_anomaly = any(kw in status for kw in ["异常", "全灭", "损坏", "故障", "破损", "不亮", "错误", "黑屏", "全亮"])

        detections.append({
            "category": category,
            "anomaly_type": status,
            "confidence": 0.9 if is_anomaly else 0.8,
            "bbox": [x1, y1, x2, y2],
            "description": reason if reason else status,
        })

    has_anomaly = any("异常" in d.get("anomaly_type", "") for d in detections)

    return {
        "has_anomaly": has_anomaly,
        "detections": detections,
        "summary": f"检测到 {len(detections)} 个目标" + ("，存在异常" if has_anomaly else "，均正常")
    }


def draw_detections_on_image(image: Image.Image, detections: list, has_anomaly: bool) -> Image.Image:
    """在图片上绘制检测框，异常目标框内红色高亮"""
    img = image.copy().convert("RGBA")
    width, height = img.size

    draw = ImageDraw.Draw(img)

    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 20)
        except:
            font = ImageFont.load_default()

    for det in detections:
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        # 判断坐标类型
        max_coord = max(bbox)
        if max_coord > 1000:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        else:
            x1 = int(bbox[0] * width / 1000)
            y1 = int(bbox[1] * height / 1000)
            x2 = int(bbox[2] * width / 1000)
            y2 = int(bbox[3] * height / 1000)

        # 确保坐标有效
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width-1, x2), min(height-1, y2)

        # 确保框有效尺寸
        if x2 <= x1 or y2 <= y1:
            continue

        category = det.get("category", "unknown")
        anomaly_type = det.get("anomaly_type", "")
        is_abnormal = anomaly_type.lower() not in ["normal", "正常", ""]

        # 异常目标：红色高亮填充 + 红色边框
        if is_abnormal:
            # 在目标框内绘制红色半透明填充（60% 透明度）
            fill_overlay = Image.new("RGBA", (x2-x1, y2-y1), (255, 0, 0, int(255 * 0.6)))
            img.paste(fill_overlay, (x1, y1), fill_overlay)
            color = (255, 0, 0)  # 红色边框
        else:
            # 正常目标：使用类别颜色
            color = CATEGORY_COLORS.get(category, (0, 255, 0))

        # 绘制边界框
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 标签文本
        cat_name = CATEGORY_NAMES.get(category, category)
        confidence = det.get("confidence", 0)
        label = f"{cat_name}: {anomaly_type} ({confidence:.0%})"

        # 绘制标签背景
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_y = max(0, y1 - text_h - 6)

        draw.rectangle([x1, label_y, x1 + text_w + 6, label_y + text_h + 6], fill=color)
        draw.text((x1 + 3, label_y + 3), label, fill=(255, 255, 255), font=font)

    return img.convert("RGB")


def detect(image: Image.Image, prompt: str) -> tuple[Image.Image | None, str, str]:
    """执行检测"""
    import torch
    import re

    if image is None:
        return None, "{}", "请上传图片"

    if _model is None or _processor is None:
        return None, "{}", "❌ 请先加载模型"

    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 处理输入
    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(_model.device)

    # 推理
    with torch.no_grad():
        output_ids = _model.generate(**inputs, max_new_tokens=2048, do_sample=False)

    # 解码
    output_ids_trimmed = output_ids[0][inputs.input_ids.shape[1]:]
    result_text = _processor.decode(output_ids_trimmed, skip_special_tokens=True)

    # 尝试解析结果 (支持两种格式)
    json_data = None

    # 1. 先尝试解析 JSON 格式
    patterns = [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```", r"\{[\s\S]*\}"]
    for pattern in patterns:
        matches = re.findall(pattern, result_text)
        for match in matches:
            try:
                parsed = json.loads(match)
                # 确保是字典格式
                if isinstance(parsed, dict):
                    json_data = parsed
                    break
            except:
                continue
        if json_data:
            break

    # 2. 如果 JSON 解析失败，尝试解析 <box> 格式 (微调模型输出)
    if not json_data and "<box>" in result_text:
        json_data = parse_box_format(result_text)

    # 3. 如果都失败，返回原始文本
    if not json_data or not json_data.get("detections"):
        return None, result_text, "无法解析检测结果，原始输出如上"

    # 绘制检测框
    has_anomaly = json_data.get("has_anomaly", False)
    detections = json_data.get("detections", [])

    # 检查是否有真正的异常（非 normal）
    real_anomaly = any(
        d.get("anomaly_type", "").lower() not in ["normal", "正常", ""]
        for d in detections
    )

    annotated_image = draw_detections_on_image(image, detections, real_anomaly)

    # 格式化 JSON
    json_str = json.dumps(json_data, ensure_ascii=False, indent=2)

    # 生成摘要
    summary = json_data.get("summary", "")
    if detections:
        summary += f"\n\n检测到 {len(detections)} 个目标"
        if real_anomaly:
            summary = "⚠️ 发现异常！\n\n" + summary

    return annotated_image, json_str, summary


# 默认检测 prompt (JSON 格式 - 适合基础模型)
DEFAULT_PROMPT_JSON = """请检测图片中的交通设备异常，包括：交通标志、信号灯、道路设施、诱导屏、限高架、机柜等。

以JSON格式输出：
```json
{
  "has_anomaly": true/false,
  "detections": [
    {
      "category": "类别(traffic_sign/traffic_light/road_facility等)",
      "anomaly_type": "异常类型(damaged/normal等)",
      "confidence": 0.9,
      "bbox": [x1, y1, x2, y2],
      "description": "描述"
    }
  ],
  "summary": "总结"
}
```

bbox为像素坐标，请确保边界框紧密包围目标。"""

# 微调模型 prompt (box 格式 - 适合 LoRA 微调模型)
DEFAULT_PROMPT_BOX = """请检测图像中的交通设备。需要检测的设备类型包括：交通信号灯、交通诱导屏、限高架、机柜、背包箱。

请按以下格式输出每个检测到的设备：
1. 设备类型
2. 状态（正常/异常状态）
3. 如果异常，说明可能的原因
4. 位置坐标：<box>(x1,y1),(x2,y2)</box>（坐标范围0-1000）

如果没有检测到任何设备，请回复"未检测到相关设备"。"""

# 默认使用微调模型的 prompt
DEFAULT_PROMPT = DEFAULT_PROMPT_BOX


# 构建界面
with gr.Blocks(title="交通设备异常检测", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚦 交通设备异常检测系统\n基于 Qwen-VL 视觉语言模型（支持基础模型和 LoRA 微调模型）")

    # 模型加载区
    with gr.Row():
        with gr.Column(scale=2):
            model_dropdown = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value=list(AVAILABLE_MODELS.keys())[0] if AVAILABLE_MODELS else None,
                label="选择模型（🔧 表示微调模型）",
            )
        with gr.Column(scale=1):
            with gr.Row():
                refresh_btn = gr.Button("🔄 刷新列表", variant="secondary", size="sm")
                load_btn = gr.Button("📥 加载模型", variant="primary")
        with gr.Column(scale=2):
            model_status = gr.Textbox(label="模型状态", value="未加载模型", interactive=False)

    gr.Markdown("---")

    # 检测区
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="上传图片", type="pil", height=400)
            prompt_input = gr.Textbox(
                label="检测提示词",
                value=DEFAULT_PROMPT,
                lines=8,
            )
            detect_btn = gr.Button("🔍 开始检测", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(label="检测结果", type="pil", height=400)
            summary_output = gr.Textbox(label="检测摘要", lines=4)
            json_output = gr.Code(label="JSON 结果", language="json", lines=10)

    # 事件绑定
    refresh_btn.click(
        fn=refresh_model_list,
        inputs=[],
        outputs=[model_dropdown],
    )

    load_btn.click(
        fn=load_model,
        inputs=[model_dropdown],
        outputs=[model_status],
    )

    detect_btn.click(
        fn=detect,
        inputs=[input_image, prompt_input],
        outputs=[output_image, json_output, summary_output],
    )


if __name__ == "__main__":
    print("启动交通设备异常检测系统...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
