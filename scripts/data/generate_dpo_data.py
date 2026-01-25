#!/usr/bin/env python3
"""
生成 DPO 偏好数据

从现有的 SFT 数据生成 DPO 需要的偏好对 (chosen/rejected)。

方法:
1. 使用原始标注作为 chosen (正确响应)
2. 生成 rejected 响应:
   - 方法 A: 使用模型生成 (可能出错)
   - 方法 B: 扰动原始标注 (修改坐标/类别)
   - 方法 C: 混合方法

Usage:
    # 使用扰动方法 (快速，不需要模型)
    python scripts/data/generate_dpo_data.py \
        --input data/hefei_last_dataset/qwen_data/train.json \
        --output data/dpo_data/train.json \
        --method perturb

    # 使用模型生成 (慢，但更真实)
    python scripts/data/generate_dpo_data.py \
        --input data/hefei_last_dataset/qwen_data/train.json \
        --output data/dpo_data/train.json \
        --method generate \
        --model_path ./model_cache/Qwen/Qwen3-VL-2B-Instruct
"""

import os
import re
import json
import random
import argparse
import logging
from typing import Dict, List, Optional
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_box_format(text: str) -> List[Dict]:
    """解析 <box>(x1,y1),(x2,y2)</box> 格式"""
    detections = []

    item_pattern = r'(\d+)\.\s*([^\n]+).*?(?:状态[：:]\s*([^\n]+))?.*?<box>\s*\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)\s*</box>'
    matches = re.findall(item_pattern, text, re.DOTALL)

    for match in matches:
        try:
            category = match[1].strip()
            status = match[2].strip() if match[2] else "正常"
            x1, y1, x2, y2 = int(match[3]), int(match[4]), int(match[5]), int(match[6])

            detections.append({
                "category": category,
                "status": status,
                "bbox": [x1, y1, x2, y2]
            })
        except (ValueError, IndexError):
            continue

    return detections


def perturb_bbox(bbox: List[int], max_shift: int = 100) -> List[int]:
    """扰动边界框坐标"""
    x1, y1, x2, y2 = bbox

    # 随机移动坐标
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)

    new_x1 = max(0, min(1000, x1 + shift_x))
    new_y1 = max(0, min(1000, y1 + shift_y))
    new_x2 = max(0, min(1000, x2 + shift_x + random.randint(-50, 50)))
    new_y2 = max(0, min(1000, y2 + shift_y + random.randint(-50, 50)))

    # 确保 x2 > x1, y2 > y1
    if new_x2 <= new_x1:
        new_x2 = new_x1 + 50
    if new_y2 <= new_y1:
        new_y2 = new_y1 + 50

    return [new_x1, new_y1, new_x2, new_y2]


def perturb_category(category: str) -> str:
    """扰动类别名称"""
    categories = [
        "交通信号灯", "交通信号灯（车行灯）", "交通信号灯（人行灯）",
        "交通诱导屏", "限高架", "机柜", "背包箱",
        "路灯", "电线杆", "其他设备"
    ]

    # 返回一个不同的类别
    other_categories = [c for c in categories if c not in category]
    if other_categories:
        return random.choice(other_categories)
    return category


def perturb_status(status: str) -> str:
    """扰动状态"""
    if "正常" in status:
        return random.choice(["异常", "损坏", "故障", "全灭"])
    else:
        return "正常"


def generate_rejected_by_perturbation(chosen_text: str) -> str:
    """通过扰动生成 rejected 响应"""
    detections = parse_box_format(chosen_text)

    if not detections:
        # 如果原文没有检测结果，生成一个假的检测
        return "检测到以下交通设备：\n\n1. 交通信号灯\n   - 状态：正常\n   - 位置：<box>(100,100),(200,200)</box>"

    # 随机选择扰动方式
    perturbation_type = random.choice([
        "wrong_bbox",      # 错误的边界框
        "wrong_category",  # 错误的类别
        "wrong_status",    # 错误的状态
        "missing",         # 漏检
        "extra",           # 误检
    ])

    if perturbation_type == "wrong_bbox":
        # 修改边界框坐标
        result_lines = ["检测到以下交通设备：\n"]
        for i, det in enumerate(detections, 1):
            new_bbox = perturb_bbox(det["bbox"])
            line = f"{i}. {det['category']}\n   - 状态：{det['status']}\n   - 位置：<box>({new_bbox[0]},{new_bbox[1]}),({new_bbox[2]},{new_bbox[3]})</box>"
            result_lines.append(line)
        return "\n".join(result_lines)

    elif perturbation_type == "wrong_category":
        # 修改类别
        result_lines = ["检测到以下交通设备：\n"]
        for i, det in enumerate(detections, 1):
            new_category = perturb_category(det["category"])
            bbox = det["bbox"]
            line = f"{i}. {new_category}\n   - 状态：{det['status']}\n   - 位置：<box>({bbox[0]},{bbox[1]}),({bbox[2]},{bbox[3]})</box>"
            result_lines.append(line)
        return "\n".join(result_lines)

    elif perturbation_type == "wrong_status":
        # 修改状态
        result_lines = ["检测到以下交通设备：\n"]
        for i, det in enumerate(detections, 1):
            new_status = perturb_status(det["status"])
            bbox = det["bbox"]
            line = f"{i}. {det['category']}\n   - 状态：{new_status}\n   - 位置：<box>({bbox[0]},{bbox[1]}),({bbox[2]},{bbox[3]})</box>"
            result_lines.append(line)
        return "\n".join(result_lines)

    elif perturbation_type == "missing":
        # 漏掉一些检测
        if len(detections) > 1:
            kept = random.sample(detections, max(1, len(detections) - 1))
        else:
            # 如果只有一个检测，返回"未检测到"
            return "未检测到相关设备。"

        result_lines = ["检测到以下交通设备：\n"]
        for i, det in enumerate(kept, 1):
            bbox = det["bbox"]
            line = f"{i}. {det['category']}\n   - 状态：{det['status']}\n   - 位置：<box>({bbox[0]},{bbox[1]}),({bbox[2]},{bbox[3]})</box>"
            result_lines.append(line)
        return "\n".join(result_lines)

    else:  # extra
        # 添加额外的误检
        result_lines = ["检测到以下交通设备：\n"]
        for i, det in enumerate(detections, 1):
            bbox = det["bbox"]
            line = f"{i}. {det['category']}\n   - 状态：{det['status']}\n   - 位置：<box>({bbox[0]},{bbox[1]}),({bbox[2]},{bbox[3]})</box>"
            result_lines.append(line)

        # 添加一个假的检测
        fake_bbox = [random.randint(0, 500), random.randint(0, 500),
                     random.randint(500, 1000), random.randint(500, 1000)]
        fake_category = random.choice(["交通信号灯", "机柜", "背包箱", "限高架"])
        line = f"{len(detections) + 1}. {fake_category}\n   - 状态：正常\n   - 位置：<box>({fake_bbox[0]},{fake_bbox[1]}),({fake_bbox[2]},{fake_bbox[3]})</box>"
        result_lines.append(line)

        return "\n".join(result_lines)


def generate_rejected_by_model(
    image_path: str,
    prompt: str,
    model,
    processor,
    max_new_tokens: int = 512,
) -> str:
    """使用模型生成 rejected 响应"""
    from PIL import Image

    try:
        image = Image.open(image_path).convert("RGB")
    except:
        return None

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        truncation=True,  # Enable truncation for safety
        max_length=2048,  # Reasonable default for data generation
        return_tensors="pt",
    ).to(model.device)

    import torch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,  # 高温度增加多样性
            top_p=0.95,
        )

    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return response


def convert_sft_to_dpo(
    input_path: str,
    output_path: str,
    method: str = "perturb",
    model_path: str = None,
    max_samples: int = None,
):
    """将 SFT 数据转换为 DPO 格式"""

    # 加载 SFT 数据
    with open(input_path, 'r', encoding='utf-8') as f:
        sft_data = json.load(f)

    logger.info(f"Loaded {len(sft_data)} samples from {input_path}")

    if max_samples:
        sft_data = sft_data[:max_samples]
        logger.info(f"Using first {max_samples} samples")

    # 如果使用模型生成方法，加载模型
    model = None
    processor = None
    if method == "generate":
        if not model_path:
            raise ValueError("--model_path is required for 'generate' method")

        logger.info(f"Loading model from {model_path}")
        import torch
        from transformers import AutoProcessor

        model_path_lower = model_path.lower()
        if "qwen3" in model_path_lower:
            from transformers import Qwen3VLForConditionalGeneration
            model_class = Qwen3VLForConditionalGeneration
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_class = Qwen2_5_VLForConditionalGeneration

        model = model_class.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        processor = AutoProcessor.from_pretrained(model_path)

    # 转换数据
    dpo_data = []
    for item in tqdm(sft_data, desc="Generating DPO data"):
        image_path = item["image"]
        conversations = item["conversations"]

        # ✅ 修复：使用 from 字段，支持多轮对话
        user_messages = []
        assistant_messages = []

        for conv in conversations:
            role = conv.get("from", "user")
            # 标准化角色名称
            if role in ["human", "user"]:
                role = "user"
            elif role in ["gpt", "assistant"]:
                role = "assistant"

            text = conv.get("value", "")

            if role == "user":
                user_messages.append(text)
            elif role == "assistant":
                assistant_messages.append(text)

        # 使用最后一条消息
        if not user_messages or not assistant_messages:
            logger.warning(f"Skipping sample: missing user or assistant messages")
            continue

        user_msg = user_messages[-1]
        assistant_msg = assistant_messages[-1]

        # chosen = 原始标注
        chosen = assistant_msg

        # 生成 rejected
        prompt = user_msg.replace("<image>\n", "").replace("<image>", "")

        if method == "perturb":
            rejected = generate_rejected_by_perturbation(chosen)
        elif method == "generate":
            rejected = generate_rejected_by_model(
                image_path, prompt, model, processor
            )
            if rejected is None:
                continue
            # 如果生成的和 chosen 完全一样，使用扰动方法
            if rejected.strip() == chosen.strip():
                rejected = generate_rejected_by_perturbation(chosen)
        else:
            raise ValueError(f"Unknown method: {method}")

        dpo_item = {
            "image": image_path,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
        dpo_data.append(dpo_item)

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(dpo_data)} DPO samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate DPO preference data")
    parser.add_argument("--input", type=str, required=True,
                        help="Input SFT data path")
    parser.add_argument("--output", type=str, required=True,
                        help="Output DPO data path")
    parser.add_argument("--method", type=str, default="perturb",
                        choices=["perturb", "generate"],
                        help="Method to generate rejected responses")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model path (required for 'generate' method)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")

    args = parser.parse_args()

    convert_sft_to_dpo(
        args.input,
        args.output,
        args.method,
        args.model_path,
        args.max_samples,
    )


if __name__ == "__main__":
    main()
