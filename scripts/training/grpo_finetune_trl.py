#!/usr/bin/env python3
"""
GRPO Fine-tuning Script using TRL's GRPOTrainer.

Based on: https://github.com/2U1/Qwen-VL-Series-Finetune

Usage:
    python scripts/training/grpo_finetune_trl.py \
        --model_path Qwen/Qwen3-VL-2B-Instruct \
        --train_data data/qwen_data/train.json \
        --output_dir outputs/qwen3vl_grpo
"""

import os
import re
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoConfig,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import GRPOConfig as TRLGRPOConfig
from datasets import Dataset

# Import custom trainer - handle relative import
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from qwen_grpo_trainer import QwenVLGRPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_model_class(model_path: str):
    """Get the appropriate model class based on model path."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "").lower()

    if model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    elif model_type == "qwen2_5_vl":
        return Qwen2_5_VLForConditionalGeneration
    else:
        # Default fallback
        if "qwen3" in model_path.lower():
            from transformers import Qwen3VLForConditionalGeneration
            return Qwen3VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration


# ============ Reward Functions ============

def format_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Reward function that checks if the completion has proper format.

    Expected format:
    1. 设备类别
       状态：正常/异常
       <box>(x1,y1),(x2,y2)</box>
    """
    rewards = []

    for completion in completions:
        reward = 0.0

        # Check for box format
        has_box = bool(re.search(r'<box>\s*\(\d+\s*,\s*\d+\)\s*,\s*\(\d+\s*,\s*\d+\)\s*</box>', completion))

        # Check for numbered list format
        has_numbered = bool(re.search(r'\d+\.\s+\S+', completion))

        # Check for status
        has_status = bool(re.search(r'状态[：:]\s*\S+', completion))

        if has_box and has_numbered and has_status:
            reward = 1.0  # Perfect format
        elif has_box and (has_numbered or has_status):
            reward = 0.7  # Good format
        elif has_box:
            reward = 0.5  # Has detection
        elif completion.strip():
            reward = 0.1  # Has some output
        else:
            reward = 0.0  # Empty

        rewards.append(reward)

    return rewards


def bbox_iou_reward(completions: List[str], assistant: List[str], **kwargs) -> List[float]:
    """
    Reward function based on IoU between predicted and ground truth bboxes.

    Args:
        completions: Model generated outputs
        assistant: Ground truth responses from dataset
    """
    rewards = []

    for completion, gt_response in zip(completions, assistant):
        pred_boxes = _extract_boxes(completion)
        gt_boxes = _extract_boxes(gt_response)

        if not gt_boxes:
            # No ground truth boxes - reward if also no predictions
            reward = 1.0 if not pred_boxes else 0.3
        elif not pred_boxes:
            # Has ground truth but no predictions - penalty
            reward = 0.0
        else:
            # Compute best matching IoU
            ious = []
            for gt_box in gt_boxes:
                best_iou = 0.0
                for pred_box in pred_boxes:
                    iou = _compute_iou(pred_box, gt_box)
                    best_iou = max(best_iou, iou)
                ious.append(best_iou)

            # Average IoU as reward
            reward = sum(ious) / len(ious) if ious else 0.0

        rewards.append(reward)

    return rewards


def category_match_reward(completions: List[str], assistant: List[str], **kwargs) -> List[float]:
    """
    Reward function based on category matching.
    """
    rewards = []

    for completion, gt_response in zip(completions, assistant):
        pred_cats = _extract_categories(completion)
        gt_cats = _extract_categories(gt_response)

        if not gt_cats:
            reward = 1.0 if not pred_cats else 0.5
        elif not pred_cats:
            reward = 0.0
        else:
            # Check how many GT categories are matched
            matches = 0
            for gt_cat in gt_cats:
                for pred_cat in pred_cats:
                    if gt_cat in pred_cat or pred_cat in gt_cat:
                        matches += 1
                        break

            reward = matches / len(gt_cats)

        rewards.append(reward)

    return rewards


def status_accuracy_reward(completions: List[str], assistant: List[str], **kwargs) -> List[float]:
    """
    Reward function for status (正常/异常) accuracy.
    """
    rewards = []
    anomaly_keywords = ["异常", "全灭", "损坏", "故障", "破损", "不亮", "错误", "黑屏", "全亮"]

    for completion, gt_response in zip(completions, assistant):
        pred_statuses = _extract_statuses(completion)
        gt_statuses = _extract_statuses(gt_response)

        if not gt_statuses:
            reward = 1.0 if not pred_statuses else 0.5
        elif not pred_statuses:
            reward = 0.0
        else:
            # Check if anomaly detection matches
            pred_has_anomaly = any(
                any(kw in s for kw in anomaly_keywords) for s in pred_statuses
            )
            gt_has_anomaly = any(
                any(kw in s for kw in anomaly_keywords) for s in gt_statuses
            )

            if pred_has_anomaly == gt_has_anomaly:
                reward = 1.0
            else:
                reward = 0.0

        rewards.append(reward)

    return rewards


# ============ Helper Functions ============

def _extract_boxes(text: str) -> List[List[int]]:
    """Extract bounding boxes from text."""
    boxes = []
    pattern = r'<box>\s*\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)\s*</box>'
    for match in re.finditer(pattern, text):
        try:
            box = [int(match.group(i)) for i in range(1, 5)]
            boxes.append(box)
        except ValueError:
            continue
    return boxes


def _extract_categories(text: str) -> List[str]:
    """Extract categories from numbered list."""
    categories = []
    pattern = r'\d+\.\s*([^\n]+)'
    for match in re.finditer(pattern, text):
        cat = match.group(1).strip()
        # Remove status if on same line
        if "状态" in cat:
            cat = cat.split("状态")[0].strip()
        if cat:
            categories.append(cat)
    return categories


def _extract_statuses(text: str) -> List[str]:
    """Extract status values from text."""
    statuses = []
    pattern = r'状态[：:]\s*([^\n]+)'
    for match in re.finditer(pattern, text):
        statuses.append(match.group(1).strip())
    return statuses


def _compute_iou(box1: List[int], box2: List[int]) -> float:
    """Compute IoU between two boxes."""
    if len(box1) != 4 or len(box2) != 4:
        return 0.0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# ============ Data Processing ============

def load_and_prepare_dataset(
    data_path: str,
    processor,
    max_image_size: int = 512,
) -> Dataset:
    """
    Load dataset and prepare for GRPO training.
    Uses lazy loading - images are loaded on-demand during training.

    Returns HuggingFace Dataset with columns:
    - prompt: The text prompt (with image placeholder)
    - image_path: Path to image (loaded lazily)
    - assistant: Ground truth response (for reward computation)
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    logger.info(f"Loaded {len(raw_data)} samples from {data_path}")

    processed_data = []
    skipped = 0

    for idx, item in enumerate(raw_data):
        image_path = item.get("image", "")
        conversations = item.get("conversations", [])

        if len(conversations) < 2:
            skipped += 1
            continue

        # Check image exists (don't load yet)
        if not os.path.exists(image_path):
            skipped += 1
            continue

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

            text = conv.get("value", "").replace("<image>\n", "").replace("<image>", "").strip()

            if role == "user":
                user_messages.append(text)
            elif role == "assistant":
                assistant_messages.append(text)

        # 使用最后一条消息（对于单轮检测任务）
        if not user_messages or not assistant_messages:
            logger.warning(f"Sample {idx}: missing user or assistant messages, skipping")
            skipped += 1
            continue

        user_msg = user_messages[-1]
        assistant_msg = assistant_messages[-1]

        # Build chat messages for prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_msg},
                ],
            }
        ]

        # Apply chat template
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        processed_data.append({
            "prompt": prompt,
            "image_path": image_path,  # Store path, load lazily
            "assistant": assistant_msg,  # Ground truth for reward
            "max_image_size": max_image_size,
        })

        # Progress log
        if (idx + 1) % 1000 == 0:
            logger.info(f"Processed {idx + 1}/{len(raw_data)} samples...")

    logger.info(f"Processed {len(processed_data)} samples, skipped {skipped}")

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(processed_data)
    return dataset


def load_image_lazy(image_path: str, max_image_size: int = 512) -> Image.Image:
    """Load and resize image on-demand."""
    try:
        image = Image.open(image_path).convert("RGB")
        if max_image_size and max(image.size) > max_image_size:
            ratio = max_image_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            # Handle Pillow version compatibility
            try:
                resample = Image.Resampling.LANCZOS  # Pillow 10+
            except AttributeError:
                resample = Image.LANCZOS  # Pillow 9-
            image = image.resize(new_size, resample)
        return image
    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        return Image.new("RGB", (224, 224), color="white")


def create_data_collator(processor):
    """Create a custom data collator for multimodal GRPO with lazy image loading."""

    def collate_fn(features):
        """Collator with lazy image loading."""
        # Load images lazily
        images = []
        for f in features:
            img = load_image_lazy(f["image_path"], f.get("max_image_size", 512))
            images.append([img])  # List format for batch processing

        batch = {
            "prompt": [f["prompt"] for f in features],
            "images": images,
            "assistant": [f["assistant"] for f in features],
        }
        return batch

    return collate_fn


# ============ Model Creation ============

def create_model_and_processor(
    model_path: str,
    sft_model_path: str = "",
    use_4bit: bool = True,
    bf16: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
):
    """Create model with optional LoRA and SFT weights."""
    logger.info(f"Loading model from {model_path}")

    model_class = get_model_class(model_path)
    logger.info(f"Using model class: {model_class.__name__}")

    # Quantization config
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # Check flash attention availability
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        logger.info("Using Flash Attention 2")
    except ImportError:
        attn_impl = "sdpa"
        logger.info("Flash Attention not available, using SDPA")

    # Load model
    model = model_class.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Prepare for k-bit training
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    # Configure LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load SFT weights if provided
    if sft_model_path and os.path.exists(sft_model_path):
        from peft import PeftModel
        logger.info(f"Loading SFT LoRA weights from {sft_model_path}")
        model = PeftModel.from_pretrained(model, sft_model_path, is_trainable=True)
        logger.info("SFT LoRA weights loaded, model is already a PEFT model")
        # Keep peft_config for trainer to recognize this is a PEFT model
        # The trainer will detect the existing adapter and not apply a new one

    return model, processor, peft_config


# ============ Main Training ============

def main():
    import argparse

    parser = argparse.ArgumentParser(description="GRPO fine-tuning with TRL")

    # Model arguments
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--sft_model_path", type=str, default="",
                        help="Path to SFT LoRA model to continue from")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization (default: enabled, use --no_4bit to disable)")
    parser.add_argument("--no_4bit", dest="use_4bit", action="store_false",
                        help="Disable 4-bit quantization")
    parser.set_defaults(use_4bit=True)
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 precision (default: enabled, use --no_bf16 to disable)")
    parser.add_argument("--no_bf16", dest="bf16", action="store_false",
                        help="Disable bfloat16 precision")
    parser.set_defaults(bf16=True)

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # Data arguments
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default="",
                        help="Path to validation data (optional)")
    parser.add_argument("--max_image_size", type=int, default=512)

    # GRPO arguments
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty coefficient")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="outputs/qwen3vl_grpo_trl")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200,
                        help="Evaluate every N steps (0 to disable)")

    # Logging
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="qwen-vl-grpo",
                        help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name for logging")

    # Other
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup run name
    if args.run_name is None:
        import datetime
        args.run_name = f"grpo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Set seed
    torch.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, "training_config.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Create model and processor
    model, processor, peft_config = create_model_and_processor(
        model_path=args.model_path,
        sft_model_path=args.sft_model_path,
        use_4bit=args.use_4bit,
        bf16=args.bf16,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Load dataset
    dataset = load_and_prepare_dataset(
        data_path=args.train_data,
        processor=processor,
        max_image_size=args.max_image_size,
    )

    # Load validation dataset (if provided)
    eval_dataset = None
    if args.val_data and os.path.exists(args.val_data):
        logger.info(f"Loading validation data from {args.val_data}")
        eval_dataset = load_and_prepare_dataset(
            data_path=args.val_data,
            processor=processor,
            max_image_size=args.max_image_size,
        )
    else:
        logger.info("No validation data provided, skipping validation")

    # Define reward functions
    reward_funcs = [
        format_reward,
        bbox_iou_reward,
        category_match_reward,
        status_accuracy_reward,
    ]

    # Reward weights (format:1, bbox:2, category:1, status:1)
    reward_weights = [1.0, 2.0, 1.0, 1.0]

    # Setup logging
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args),
            )
            report_to = ["tensorboard", "wandb"]
            logger.info(f"W&B logging enabled: {args.wandb_project}/{args.run_name}")
        except ImportError:
            logger.warning("wandb not installed, falling back to tensorboard only")
            report_to = ["tensorboard"]
    else:
        report_to = ["tensorboard"]

    # Configure GRPO training
    logger.info("Configuring GRPO training...")
    training_args = TRLGRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset and args.eval_steps > 0 else "no",
        save_total_limit=3,
        bf16=args.bf16,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # GRPO specific
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,  # Still used internally
        beta=args.beta,  # KL coefficient
        temperature=args.temperature,
        reward_weights=reward_weights,  # Weights for each reward function
        # Logging
        report_to=report_to,
        logging_first_step=True,
        seed=args.seed,
    )

    # Create trainer with custom Qwen-VL support
    logger.info("Creating trainer...")
    trainer = QwenVLGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
    )

    # Set custom data collator after initialization
    trainer.data_collator = create_data_collator(processor)

    # Train
    logger.info("Starting GRPO training with TRL...")
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final"))
    processor.save_pretrained(os.path.join(args.output_dir, "final"))

    # 保存训练日志
    training_log = {
        "config": vars(args),
        "train_history": [],
        "val_history": [],
        "final_metrics": {},
    }

    # 从 trainer 的日志历史中提取训练指标
    if hasattr(trainer.state, 'log_history'):
        for log_entry in trainer.state.log_history:
            # 训练日志（包含 reward 等 GRPO 特有指标）
            if 'loss' in log_entry or 'reward' in log_entry:
                entry = {
                    "step": log_entry.get('step', 0),
                    "epoch": log_entry.get('epoch', 0),
                }
                # 添加所有可用的指标
                for key in ['loss', 'reward', 'kl', 'learning_rate']:
                    if key in log_entry:
                        entry[key] = log_entry[key]

                if 'eval_' not in str(log_entry):
                    training_log["train_history"].append(entry)

            # 验证日志
            if 'eval_reward' in log_entry or 'eval_loss' in log_entry:
                val_entry = {
                    "step": log_entry.get('step', 0),
                    "epoch": log_entry.get('epoch', 0),
                }
                for key in log_entry:
                    if key.startswith('eval_'):
                        val_entry[key] = log_entry[key]
                training_log["val_history"].append(val_entry)

    # 保存最终指标
    if hasattr(trainer.state, 'best_metric'):
        training_log["final_metrics"]["best_metric"] = trainer.state.best_metric
    if hasattr(trainer.state, 'best_model_checkpoint'):
        training_log["final_metrics"]["best_checkpoint"] = trainer.state.best_model_checkpoint

    # 保存到 JSON 文件
    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2, default=str)
    logger.info(f"Training log saved to {log_path}")

    logger.info(f"Training complete! Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
