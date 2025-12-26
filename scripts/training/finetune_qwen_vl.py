#!/usr/bin/env python3
"""
Qwen-VL LoRA Fine-tuning Script for Traffic Equipment Anomaly Detection.

Supports both Qwen3-VL and Qwen2.5-VL models.

Usage:
    python scripts/finetune_qwen_vl.py \
        --model_path Qwen/Qwen3-VL-2B-Instruct \
        --train_data data/hefei_last_dataset/qwen_data/train.json \
        --val_data data/hefei_last_dataset/qwen_data/val.json \
        --output_dir outputs/qwen3vl_lora
"""

import os
import json
import argparse
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# 延迟导入模型类，根据模型路径动态选择
def get_model_class(model_path: str):
    """根据模型路径返回对应的模型类"""
    model_path_lower = model_path.lower()
    if "qwen3" in model_path_lower:
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    else:
        # Qwen2.5-VL 或其他版本
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    """Fine-tuning configuration."""
    # Model
    model_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    use_4bit: bool = True  # QLoRA
    use_8bit: bool = False

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Data
    train_data: str = "data/hefei_last_dataset/qwen_data/train.json"
    val_data: str = "data/hefei_last_dataset/qwen_data/val.json"
    max_length: int = 2048
    max_image_size: int = 512  # 图片最大边长，影响显存占用和检测精度

    # Training
    output_dir: str = "outputs/qwen3vl_lora"
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # Other
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True


class TrafficAnomalyDataset(Dataset):
    """Dataset for traffic equipment anomaly detection."""

    def __init__(
        self,
        data_path: str,
        processor,
        max_length: int = 2048,
        max_image_size: int = 512,
    ):
        self.processor = processor
        self.max_length = max_length
        self.max_image_size = max_image_size

        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        logger.info(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def _find_assistant_start(self, input_ids: torch.Tensor) -> int:
        """找到 assistant 回复开始的位置"""
        # Qwen-VL 的 assistant 标记通常是 <|im_start|>assistant
        # 我们需要找到最后一个 assistant 标记的位置
        tokenizer = self.processor.tokenizer

        # 尝试多种可能的 assistant 标记
        assistant_tokens_candidates = [
            "assistant",
            "<|im_start|>assistant",
            "Assistant",
        ]

        input_ids_list = input_ids.tolist()

        # 方法1: 查找 "assistant" 文本对应的 token 序列
        for candidate in assistant_tokens_candidates:
            try:
                candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
                # 在 input_ids 中查找这个序列
                for i in range(len(input_ids_list) - len(candidate_ids) + 1):
                    if input_ids_list[i:i+len(candidate_ids)] == candidate_ids:
                        # 返回 assistant 标记之后的位置
                        return i + len(candidate_ids)
            except Exception:
                continue

        # 方法2: 使用特殊 token ID
        # Qwen 模型通常使用 <|im_start|> 和 <|im_end|> 作为消息边界
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        if im_start_id is not None and im_start_id != tokenizer.unk_token_id:
            # 找到倒数第二个 <|im_start|>（最后一个是 assistant 的开始）
            positions = [i for i, x in enumerate(input_ids_list) if x == im_start_id]
            if len(positions) >= 1:
                # 最后一个 <|im_start|> 后面应该是 assistant
                last_start = positions[-1]
                # 跳过 <|im_start|>assistant\n
                # 通常格式是 <|im_start|>assistant\n 内容 <|im_end|>
                for i in range(last_start + 1, min(last_start + 10, len(input_ids_list))):
                    # 找到换行符后的位置
                    token_text = tokenizer.decode([input_ids_list[i]])
                    if '\n' in token_text:
                        return i + 1
                return last_start + 3  # 默认跳过几个 token

        # 方法3: 回退策略 - 假设前 60% 是 prompt
        return int(len(input_ids_list) * 0.6)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.data[idx]

        # Get image path and conversations
        image_path = item["image"]
        conversations = item["conversations"]

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
            # 限制图片大小以减少显存占用 (Qwen-VL 图片 token 很多)
            if self.max_image_size and max(image.size) > self.max_image_size:
                ratio = self.max_image_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.LANCZOS)
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            # Return a placeholder if image fails to load
            image = Image.new("RGB", (224, 224), color="white")

        # Extract user and assistant messages
        user_msg = conversations[0]["value"]
        assistant_msg = conversations[1]["value"]

        # Format as chat messages for Qwen-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_msg.replace("<image>\n", "").replace("<image>", "")},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_msg},
                ],
            }
        ]

        # Process with Qwen-VL processor
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            truncation=False,  # 不截断，避免图像 token 丢失
            return_tensors="pt",
        )

        # Squeeze batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Create labels with proper masking
        labels = inputs["input_ids"].clone()

        # 1. Mask padding tokens
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        # 2. Mask user/system part - only train on assistant response
        assistant_start = self._find_assistant_start(inputs["input_ids"])
        labels[:assistant_start] = -100

        inputs["labels"] = labels

        return inputs


def create_model_and_processor(config: FinetuneConfig):
    """Create model and processor with LoRA."""
    logger.info(f"Loading model from {config.model_path}")

    # 动态选择模型类
    model_class = get_model_class(config.model_path)
    logger.info(f"Using model class: {model_class.__name__}")

    # Quantization config for QLoRA
    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif config.use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model
    model = model_class.from_pretrained(
        config.model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(
        config.model_path,
        trust_remote_code=True,
    )

    # Prepare for k-bit training if using quantization
    if config.use_4bit or config.use_8bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing
        )

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


class VLDataCollator:
    """Data collator for Qwen-VL models.

    处理 VL 模型特殊的张量结构，包括：
    - input_ids: 文本 token IDs
    - attention_mask: 注意力掩码
    - pixel_values: 图像像素值
    - image_grid_thw: 图像网格信息 (Qwen2-VL/Qwen3-VL)
    - labels: 训练标签
    """

    def __init__(self, processor, pad_token_id: int = None):
        self.processor = processor
        self.pad_token_id = pad_token_id or processor.tokenizer.pad_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}

        for key in features[0].keys():
            values = [f[key] for f in features]

            if not isinstance(values[0], torch.Tensor):
                batch[key] = values
                continue

            # 特殊处理不同类型的张量
            if key in ["input_ids", "attention_mask", "labels"]:
                # 需要 padding 到相同长度
                max_len = max(v.shape[0] for v in values)
                padded_values = []
                for v in values:
                    pad_len = max_len - v.shape[0]
                    if pad_len > 0:
                        if key == "input_ids":
                            pad_value = self.pad_token_id
                        elif key == "attention_mask":
                            pad_value = 0
                        else:  # labels
                            pad_value = -100
                        padding = torch.full((pad_len,), pad_value, dtype=v.dtype)
                        v = torch.cat([v, padding])
                    padded_values.append(v)
                batch[key] = torch.stack(padded_values)
            elif key == "pixel_values":
                # pixel_values 可能有不同的形状，需要特殊处理
                # 通常形状是 [num_patches, channels, height, width]
                # 如果所有样本的 patch 数量相同，可以直接 stack
                # 否则需要 padding 或使用 list
                try:
                    batch[key] = torch.stack(values)
                except RuntimeError:
                    # 形状不一致，使用 concatenate 并记录边界
                    batch[key] = torch.cat(values, dim=0)
                    # 需要额外记录每个样本的 patch 数量
                    batch["pixel_values_lengths"] = torch.tensor([v.shape[0] for v in values])
            elif key == "image_grid_thw":
                # image_grid_thw 记录每个图像的网格信息
                # 形状通常是 [num_images, 3]
                try:
                    batch[key] = torch.stack(values)
                except RuntimeError:
                    batch[key] = torch.cat(values, dim=0)
            else:
                # 其他张量尝试 stack
                try:
                    batch[key] = torch.stack(values)
                except RuntimeError:
                    # 如果失败，保持为 list
                    batch[key] = values

        return batch


def train(config: FinetuneConfig):
    """Main training function."""
    logger.info("Starting fine-tuning...")

    # Set seed
    torch.manual_seed(config.seed)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(config.output_dir, "finetune_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=2, default=str)

    # Create model and processor
    model, processor = create_model_and_processor(config)

    # Create datasets
    train_dataset = TrafficAnomalyDataset(
        config.train_data,
        processor,
        config.max_length,
        config.max_image_size
    )

    val_dataset = None
    if config.val_data and os.path.exists(config.val_data):
        val_dataset = TrafficAnomalyDataset(
            config.val_data,
            processor,
            config.max_length,
            config.max_image_size
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if val_dataset else None,
        eval_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        save_total_limit=3,
        bf16=config.bf16,
        fp16=not config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb etc.
        seed=config.seed,
    )

    # Data collator
    data_collator = VLDataCollator(processor)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving model to {config.output_dir}")
    trainer.save_model()
    processor.save_pretrained(config.output_dir)

    logger.info("Fine-tuning completed!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL for traffic anomaly detection")

    # Model arguments
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="Path to pretrained model")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--use_8bit", action="store_true",
                        help="Use 8-bit quantization")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")

    # Data arguments
    parser.add_argument("--train_data", type=str,
                        default="data/hefei_last_dataset/qwen_data/train.json",
                        help="Path to training data")
    parser.add_argument("--val_data", type=str,
                        default="data/hefei_last_dataset/qwen_data/val.json",
                        help="Path to validation data")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--max_image_size", type=int, default=512,
                        help="Maximum image size (longest edge). Larger = better quality but more VRAM")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="outputs/qwen3vl_lora",
                        help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N steps")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Use gradient checkpointing")

    args = parser.parse_args()

    # Create config from args
    config = FinetuneConfig(
        model_path=args.model_path,
        use_4bit=args.use_4bit and not args.use_8bit,
        use_8bit=args.use_8bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        train_data=args.train_data,
        val_data=args.val_data,
        max_length=args.max_length,
        max_image_size=args.max_image_size,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        seed=args.seed,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    train(config)


if __name__ == "__main__":
    main()
