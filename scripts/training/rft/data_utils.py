"""RFT 数据加载与 batch 组装工具。"""

import json
import logging
import os

from datasets import Dataset
from PIL import Image


logger = logging.getLogger(__name__)


def load_grpo_dataset(
    data_path: str,
    processor,
    max_image_size: int = 512,
) -> Dataset:
    """加载 GRPO 专用格式数据集（rft_output/*.jsonl）。

    期望每行字段：image, prompt, ground_truth, difficulty
    输出 dataset 字段：prompt, image_path, ground_truth, difficulty, max_image_size
    ground_truth / difficulty 保留为 dict，由 collator 原样传入 reward 函数。
    """
    with open(data_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            raw_data = json.load(f)
        else:
            raw_data = [json.loads(line) for line in f if line.strip()]

    logger.info("Loaded %d samples from %s", len(raw_data), data_path)

    processed_data = []
    skipped = 0

    for idx, item in enumerate(raw_data):
        image_path = item.get("image", "")
        if not os.path.exists(image_path):
            skipped += 1
            continue

        prompt_text = item.get("prompt", "")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        processed_data.append(
            {
                "prompt": prompt,
                "image_path": image_path,
                "ground_truth": item.get("ground_truth", {"detections": []}),
                "difficulty": item.get("difficulty", {}),
                "max_image_size": max_image_size,
            }
        )

    logger.info("Processed %d samples, skipped %d", len(processed_data), skipped)
    return Dataset.from_list(processed_data)


def create_grpo_data_collator(processor):
    """支持 GRPO 格式的 collator，将 ground_truth/difficulty 原样传入 batch。"""

    def collate_fn(features):
        images = []
        for f in features:
            img = load_image_lazy(f["image_path"], f.get("max_image_size", 512))
            images.append([img])

        return {
            "prompt": [f["prompt"] for f in features],
            "images": images,
            "ground_truth": [f["ground_truth"] for f in features],
            "difficulty": [f["difficulty"] for f in features],
        }

    return collate_fn


def load_and_prepare_dataset(
    data_path: str,
    processor,
    max_image_size: int = 512,
) -> Dataset:
    """加载并整理 GRPO 数据集（图像懒加载模式）。"""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    logger.info("Loaded %d samples from %s", len(raw_data), data_path)

    processed_data = []
    skipped = 0

    for idx, item in enumerate(raw_data):
        image_path = item.get("image", "")
        conversations = item.get("conversations", [])

        if len(conversations) < 2:
            skipped += 1
            continue

        if not os.path.exists(image_path):
            skipped += 1
            continue

        user_messages = []
        assistant_messages = []

        for conv in conversations:
            role = conv.get("from", "user")
            if role in ["human", "user"]:
                role = "user"
            elif role in ["gpt", "assistant"]:
                role = "assistant"

            text = conv.get("value", "").replace("<image>\n", "").replace("<image>", "").strip()

            if role == "user":
                user_messages.append(text)
            elif role == "assistant":
                assistant_messages.append(text)

        if not user_messages or not assistant_messages:
            logger.warning("Sample %d: missing user or assistant messages, skipping", idx)
            skipped += 1
            continue

        user_msg = user_messages[-1]
        assistant_msg = assistant_messages[-1]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_msg},
                ],
            }
        ]

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        processed_data.append(
            {
                "prompt": prompt,
                "image_path": image_path,
                "assistant": assistant_msg,
                "max_image_size": max_image_size,
            }
        )

        if (idx + 1) % 1000 == 0:
            logger.info("Processed %d/%d samples...", idx + 1, len(raw_data))

    logger.info("Processed %d samples, skipped %d", len(processed_data), skipped)

    return Dataset.from_list(processed_data)


def load_image_lazy(image_path: str, max_image_size: int = 512) -> Image.Image:
    """按需读取并缩放图像。"""
    try:
        image = Image.open(image_path).convert("RGB")
        if max_image_size and max(image.size) > max_image_size:
            ratio = max_image_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
            image = image.resize(new_size, resample)
        return image
    except Exception as exc:
        logger.warning("Failed to load image %s: %s", image_path, exc)
        return Image.new("RGB", (224, 224), color="white")


def create_data_collator(processor):
    """创建支持图像懒加载的 collator。"""

    def collate_fn(features):
        images = []
        for f in features:
            img = load_image_lazy(f["image_path"], f.get("max_image_size", 512))
            images.append([img])

        return {
            "prompt": [f["prompt"] for f in features],
            "images": images,
            "assistant": [f["assistant"] for f in features],
        }

    return collate_fn
