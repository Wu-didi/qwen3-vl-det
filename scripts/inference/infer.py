#!/usr/bin/env python3
"""
Qwen3-VL 推理脚本

用法:
    python infer.py --image path/to/image.jpg --prompt "描述这张图片"
    python infer.py --image path/to/image.jpg  # 使用默认检测 prompt
    python infer.py --image path/to/image.jpg --output result.jpg  # 保存可视化结果
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration


# 默认检测 prompt
DEFAULT_PROMPT = """分析这张图片，检测是否存在交通设备异常（如交通标志损坏、信号灯故障、道路设施破损等）。

以JSON格式输出：
```json
{
  "has_anomaly": true/false,
  "detections": [
    {
      "category": "类别",
      "anomaly_type": "异常类型",
      "confidence": 0.0-1.0,
      "bbox": [x1, y1, x2, y2],
      "description": "描述"
    }
  ],
  "summary": "总结"
}
```

注意：bbox坐标为归一化坐标，范围 0-1000。"""

# 类别颜色映射
CATEGORY_COLORS = {
    "traffic_sign": (255, 0, 0),      # 红
    "traffic_light": (0, 255, 0),     # 绿
    "road_facility": (0, 0, 255),     # 蓝
    "guidance_screen": (255, 165, 0), # 橙
    "height_limit": (128, 0, 128),    # 紫
    "cabinet": (255, 192, 203),       # 粉
}

# 类别中文名
CATEGORY_NAMES = {
    "traffic_sign": "交通标志",
    "traffic_light": "信号灯",
    "road_facility": "道路设施",
    "guidance_screen": "诱导屏",
    "height_limit": "限高架",
    "cabinet": "机柜",
}


def load_model(model_path: str, device: str = "cuda", dtype: str = "bfloat16"):
    """加载模型和处理器

    Args:
        model_path: 模型路径或 Hugging Face 模型名
        device: 设备 (cuda/cpu)
        dtype: 数据类型 (float16/bfloat16/float32)

    Returns:
        model, processor
    """
    print(f"正在加载模型: {model_path}")
    start = time.time()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # 根据模型路径选择模型类
    model_path_lower = model_path.lower()
    if "qwen3-vl" in model_path_lower or "qwen3_vl" in model_path_lower:
        print("使用 Qwen3-VL 模型类")
        model_class = Qwen3VLForConditionalGeneration
    else:
        print("使用 Qwen2.5-VL 模型类")
        model_class = Qwen2_5_VLForConditionalGeneration

    model = model_class.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"模型加载完成，耗时: {time.time() - start:.2f}s")
    return model, processor


def inference(
    model,
    processor,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 2048,
) -> str:
    """执行推理

    Args:
        model: 模型
        processor: 处理器
        image_path: 图片路径
        prompt: 提示词
        max_new_tokens: 最大生成 token 数

    Returns:
        生成的文本
    """
    # 加载图片
    image = Image.open(image_path).convert("RGB")

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
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        truncation=True,  # Enable truncation for safety
        max_length=2048,  # Reasonable default for inference
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # 生成
    print("正在推理...")
    start = time.time()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # 解码输出
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print(f"推理完成，耗时: {time.time() - start:.2f}s")
    return output_text


def extract_json(text: str) -> dict | None:
    """从文本中提取 JSON"""
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"\{[\s\S]*\}",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    return None


def draw_detections(image: Image.Image, detections: list, line_width: int = 3) -> Image.Image:
    """在图片上绘制检测框

    Args:
        image: PIL Image
        detections: 检测结果列表
        line_width: 线宽

    Returns:
        绘制后的图片
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()

    for det in detections:
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        # 判断坐标类型：如果最大值超过 1000，认为是像素坐标
        max_coord = max(bbox)
        if max_coord > 1000:
            # 像素坐标，直接使用
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        else:
            # 归一化坐标 (0-1000)，转换为像素坐标
            x1 = int(bbox[0] * width / 1000)
            y1 = int(bbox[1] * height / 1000)
            x2 = int(bbox[2] * width / 1000)
            y2 = int(bbox[3] * height / 1000)

        # 确保坐标在图片范围内
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        # 获取颜色
        category = det.get("category", "unknown")
        color = CATEGORY_COLORS.get(category, (128, 128, 128))

        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # 准备标签文本
        cat_name = CATEGORY_NAMES.get(category, category)
        anomaly_type = det.get("anomaly_type", "")
        confidence = det.get("confidence", 0)
        label = f"{cat_name}: {anomaly_type} ({confidence:.0%})"

        # 计算文本大小
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # 绘制标签背景
        label_y = max(0, y1 - text_height - 6)
        draw.rectangle(
            [x1, label_y, x1 + text_width + 6, label_y + text_height + 6],
            fill=color,
        )

        # 绘制标签文字
        draw.text((x1 + 3, label_y + 3), label, fill=(255, 255, 255), font=font)

    return img


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 推理脚本")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="./model_cache/Qwen/Qwen3-VL-2B-Instruct",
        help="模型路径或 HuggingFace 模型名",
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="图片路径",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=DEFAULT_PROMPT,
        help="提示词（默认为异常检测 prompt）",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出图片路径（绘制检测框后保存）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="模型数据类型",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="尝试解析并格式化输出为 JSON",
    )

    args = parser.parse_args()

    # 检查图片路径
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"错误: 图片不存在: {args.image}")
        return 1

    # 加载模型
    model, processor = load_model(args.model, dtype=args.dtype)

    # 执行推理
    result = inference(
        model,
        processor,
        args.image,
        args.prompt,
        args.max_tokens,
    )

    print("\n" + "=" * 50)
    print("输出结果:")
    print("=" * 50)

    # 解析 JSON
    json_data = extract_json(result)

    if args.json or args.output:
        if json_data:
            print(json.dumps(json_data, ensure_ascii=False, indent=2))
        else:
            print(result)
            print("\n[警告] 无法解析 JSON，跳过可视化")
    else:
        print(result)

    # 绘制检测框并保存
    if args.output and json_data:
        detections = json_data.get("detections", [])
        if detections:
            print(f"\n检测到 {len(detections)} 个目标，正在绘制...")
            image = Image.open(args.image).convert("RGB")
            result_image = draw_detections(image, detections)

            # 确定输出路径
            output_path = args.output
            result_image.save(output_path)
            print(f"可视化结果已保存到: {output_path}")
        else:
            print("\n未检测到目标，无需绘制")
    elif args.output and not json_data:
        print("\n[警告] 无法解析检测结果，跳过可视化")

    return 0


if __name__ == "__main__":
    exit(main())
