#!/usr/bin/env python3
"""
Inference script for fine-tuned Qwen-VL model.

Supports both Qwen3-VL and Qwen2.5-VL models.

Usage:
    python scripts/inference_finetuned.py --model_path outputs/qwen3vl_lora --image_path test.jpg
"""

import argparse
import json
import torch
from PIL import Image
from transformers import AutoProcessor
from peft import PeftModel


def get_model_class(model_path: str):
    """根据模型路径返回对应的模型类"""
    model_path_lower = model_path.lower()
    if "qwen3" in model_path_lower:
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration


def load_model(model_path: str, base_model_path: str = None):
    """Load fine-tuned model with LoRA weights."""

    # Determine base model path
    if base_model_path is None:
        # Try to read from config
        config_path = f"{model_path}/finetune_config.json"
        try:
            with open(config_path) as f:
                config = json.load(f)
                base_model_path = config.get("model_path", "Qwen/Qwen3-VL-2B-Instruct")
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not read config file: {e}")
            base_model_path = "Qwen/Qwen3-VL-2B-Instruct"

    print(f"Loading base model from {base_model_path}")

    # 动态选择模型类
    model_class = get_model_class(base_model_path)
    print(f"Using model class: {model_class.__name__}")

    # Load base model
    model = model_class.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA weights
    print(f"Loading LoRA weights from {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()  # Merge LoRA weights for faster inference

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model.eval()
    return model, processor


def detect_anomalies(
    model,
    processor,
    image_path: str,
    prompt: str = None
) -> str:
    """Run anomaly detection on an image."""

    # Default prompt
    if prompt is None:
        prompt = """请检测图像中的交通设备。需要检测的设备类型包括：交通信号灯、交通诱导屏、限高架、机柜、背包箱。

请按以下格式输出每个检测到的设备：
1. 设备类型
2. 状态（正常/异常状态）
3. 如果异常，说明可能的原因
4. 位置坐标：<box>(x1,y1),(x2,y2)</box>（坐标范围0-1000）

如果没有检测到任何设备，请回复"未检测到相关设备"。"""

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Format messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Process
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        truncation=True,  # Enable truncation for safety
        max_length=2048,  # Reasonable default for inference
        return_tensors="pt",
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Decode
    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return response


def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Qwen2-VL")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned model")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Path to base model (optional)")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt (optional)")

    args = parser.parse_args()

    # Load model
    model, processor = load_model(args.model_path, args.base_model_path)

    # Run detection
    print(f"\nProcessing: {args.image_path}")
    print("-" * 50)

    result = detect_anomalies(model, processor, args.image_path, args.prompt)

    print(result)
    print("-" * 50)


if __name__ == "__main__":
    main()
