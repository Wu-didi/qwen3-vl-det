#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) Fine-tuning Script for Qwen-VL.

DPO 直接从偏好数据学习，不需要在线生成，比 GRPO 快很多。

Usage:
    python scripts/training/dpo_finetune.py \
        --model_path Qwen/Qwen3-VL-2B-Instruct \
        --train_data data/dpo_data/train.json \
        --output_dir outputs/qwen3vl_dpo

References:
    - DPO Paper: https://arxiv.org/abs/2305.18290
"""

import os
import re
import json
import time
import argparse
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_model_class(model_path: str):
    """根据模型路径返回对应的模型类"""
    model_path_lower = model_path.lower()
    if "qwen3" in model_path_lower:
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration


@dataclass
class DPOConfig:
    """DPO training configuration."""
    # Model
    model_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    use_4bit: bool = True
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
    train_data: str = "data/dpo_data/train.json"
    val_data: str = ""  # 验证集路径（可选）
    max_length: int = 2048
    max_image_size: int = 512

    # DPO specific
    beta: float = 0.1  # DPO temperature, 控制偏好强度

    # Training
    output_dir: str = "outputs/qwen3vl_dpo"
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500  # 每多少步验证一次
    max_grad_norm: float = 1.0

    # Other
    seed: int = 42
    bf16: bool = True


class DPODataset(Dataset):
    """DPO 训练数据集

    数据格式:
    {
        "image": "path/to/image.jpg",
        "prompt": "用户提示",
        "chosen": "好的响应",
        "rejected": "差的响应"
    }
    """

    def __init__(self, data_path: str, processor, max_length: int = 2048, max_image_size: int = 512):
        self.processor = processor
        self.max_length = max_length
        self.max_image_size = max_image_size

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        logger.info(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.data[idx]

        image_path = item["image"]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
            if self.max_image_size and max(image.size) > self.max_image_size:
                ratio = self.max_image_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.LANCZOS)
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            image = Image.new("RGB", (224, 224), color="white")

        return {
            "image": image,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }


class DPOTrainer:
    """DPO 训练器"""

    def __init__(
        self,
        model,
        ref_model,
        processor,
        config: DPOConfig,
    ):
        self.model = model
        self.ref_model = ref_model
        self.processor = processor
        self.config = config

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.global_step = 0
        self.best_val_accuracy = 0.0  # 记录最佳验证准确率

    def compute_log_probs(
        self,
        model,
        image: Image.Image,
        prompt: str,
        response: str,
    ) -> torch.Tensor:
        """计算给定响应的 log 概率"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=False,  # Single sample, no padding needed
            truncation=True,  # Enable truncation to prevent OOM
            max_length=self.config.max_length,  # Use configured max_length
        ).to(model.device)

        # Forward pass
        outputs = model(**inputs, labels=inputs["input_ids"])

        # 返回平均 log 概率
        return -outputs.loss  # loss 是负 log likelihood

    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 DPO 损失

        L_DPO = -log(sigmoid(β * (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))
        """
        # 计算 log ratio
        policy_log_ratio = policy_chosen_logps - policy_rejected_logps
        ref_log_ratio = ref_chosen_logps - ref_rejected_logps

        # DPO loss
        logits = self.config.beta * (policy_log_ratio - ref_log_ratio)
        loss = -F.logsigmoid(logits)

        return loss

    def dpo_step(
        self,
        image: Image.Image,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict[str, float]:
        """执行单个 DPO 更新步骤"""
        # 计算 policy model 的 log probs
        policy_chosen_logps = self.compute_log_probs(
            self.model, image, prompt, chosen
        )
        policy_rejected_logps = self.compute_log_probs(
            self.model, image, prompt, rejected
        )

        # 计算 reference model 的 log probs (不需要梯度)
        with torch.no_grad():
            ref_chosen_logps = self.compute_log_probs(
                self.ref_model, image, prompt, chosen
            )
            ref_rejected_logps = self.compute_log_probs(
                self.ref_model, image, prompt, rejected
            )

        # 计算 DPO 损失
        loss = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        )

        # 反向传播
        loss.backward()

        # 计算一些统计信息
        with torch.no_grad():
            chosen_reward = self.config.beta * (policy_chosen_logps - ref_chosen_logps)
            rejected_reward = self.config.beta * (policy_rejected_logps - ref_rejected_logps)
            reward_margin = chosen_reward - rejected_reward
            accuracy = (reward_margin > 0).float()

        return {
            "loss": loss.item(),
            "chosen_reward": chosen_reward.item(),
            "rejected_reward": rejected_reward.item(),
            "reward_margin": reward_margin.item(),
            "accuracy": accuracy.item(),
        }

    @torch.no_grad()
    def evaluate(self, val_dataset: DPODataset, num_samples: int = None) -> Dict[str, float]:
        """在验证集上评估模型

        Args:
            val_dataset: 验证数据集
            num_samples: 评估的样本数量（None 表示全部）

        Returns:
            验证指标字典
        """
        logger.info("Running validation...")
        self.model.eval()

        # 限制验证样本数量以节省时间
        if num_samples is None:
            num_samples = len(val_dataset)
        num_samples = min(num_samples, len(val_dataset))

        val_stats = []

        for i in tqdm(range(num_samples), desc="Validation"):
            sample = val_dataset[i]
            image = sample["image"]
            prompt = sample["prompt"]
            chosen = sample["chosen"]
            rejected = sample["rejected"]

            # 计算 chosen 和 rejected 的 log probs
            policy_chosen_logps = self.compute_log_probs(
                self.model, image, prompt, chosen
            )
            policy_rejected_logps = self.compute_log_probs(
                self.model, image, prompt, rejected
            )

            # 计算 reference model 的 log probs
            ref_chosen_logps = self.compute_log_probs(
                self.ref_model, image, prompt, chosen
            )
            ref_rejected_logps = self.compute_log_probs(
                self.ref_model, image, prompt, rejected
            )

            # 计算 DPO 损失
            loss = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
            )

            # 计算统计信息
            chosen_reward = self.config.beta * (policy_chosen_logps - ref_chosen_logps)
            rejected_reward = self.config.beta * (policy_rejected_logps - ref_rejected_logps)
            reward_margin = chosen_reward - rejected_reward
            accuracy = (reward_margin > 0).float()

            val_stats.append({
                "loss": loss.item(),
                "accuracy": accuracy.item(),
                "reward_margin": reward_margin.item(),
            })

        # 计算平均指标
        avg_stats = {
            "val_loss": sum(s["loss"] for s in val_stats) / len(val_stats),
            "val_accuracy": sum(s["accuracy"] for s in val_stats) / len(val_stats),
            "val_reward_margin": sum(s["reward_margin"] for s in val_stats) / len(val_stats),
        }

        logger.info(
            f"Validation results: loss={avg_stats['val_loss']:.4f}, "
            f"accuracy={avg_stats['val_accuracy']:.2f}, "
            f"reward_margin={avg_stats['val_reward_margin']:.4f}"
        )

        self.model.train()
        return avg_stats

    def train(self, dataset: DPODataset, val_dataset: DPODataset = None):
        """训练循环"""
        logger.info("Starting DPO training...")

        def collate_fn(batch):
            return {
                "image": [item["image"] for item in batch],
                "prompt": [item["prompt"] for item in batch],
                "chosen": [item["chosen"] for item in batch],
                "rejected": [item["rejected"] for item in batch],
            }

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        self.model.train()
        accumulated_steps = 0
        epoch_stats = []

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

            for batch_idx, batch in enumerate(pbar):
                if batch_idx == 0:
                    logger.info("Processing first batch...")

                batch_stats = []
                for i in range(len(batch["image"])):
                    image = batch["image"][i]
                    prompt = batch["prompt"][i]
                    chosen = batch["chosen"][i]
                    rejected = batch["rejected"][i]

                    stats = self.dpo_step(image, prompt, chosen, rejected)
                    batch_stats.append(stats)
                    accumulated_steps += 1

                avg_stats = {
                    "loss": sum(s["loss"] for s in batch_stats) / len(batch_stats),
                    "accuracy": sum(s["accuracy"] for s in batch_stats) / len(batch_stats),
                    "reward_margin": sum(s["reward_margin"] for s in batch_stats) / len(batch_stats),
                }
                epoch_stats.append(avg_stats)

                if batch_idx == 0:
                    logger.info(f"First batch: loss={avg_stats['loss']:.4f}, acc={avg_stats['accuracy']:.2f}")

                # 梯度累积
                if accumulated_steps % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.global_step % self.config.logging_steps == 0:
                        recent = epoch_stats[-self.config.logging_steps:]
                        avg_loss = sum(s["loss"] for s in recent) / len(recent)
                        avg_acc = sum(s["accuracy"] for s in recent) / len(recent)
                        logger.info(
                            f"Step {self.global_step}: loss={avg_loss:.4f}, accuracy={avg_acc:.2f}"
                        )

                    # ✅ 新增：验证逻辑
                    if val_dataset and self.config.eval_steps > 0 and self.global_step % self.config.eval_steps == 0:
                        val_stats = self.evaluate(val_dataset, num_samples=50)  # 验证 50 个样本

                        # 保存最佳模型
                        if val_stats["val_accuracy"] > self.best_val_accuracy:
                            self.best_val_accuracy = val_stats["val_accuracy"]
                            logger.info(f"New best validation accuracy: {self.best_val_accuracy:.2f}")
                            self.save_checkpoint("best")

                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(f"checkpoint-{self.global_step}")

                pbar.set_postfix({
                    "loss": f"{avg_stats['loss']:.4f}",
                    "acc": f"{avg_stats['accuracy']:.2f}",
                })

        self.save_checkpoint("final")
        logger.info("DPO training completed!")

    def save_checkpoint(self, name: str):
        """保存检查点"""
        save_path = os.path.join(self.config.output_dir, name)
        os.makedirs(save_path, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

        config_dict = {
            k: v for k, v in self.config.__dict__.items()
            if not k.startswith('_')
        }
        with open(os.path.join(save_path, "dpo_config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        logger.info(f"Checkpoint saved to {save_path}")


def create_model_and_processor(config: DPOConfig):
    """创建模型和处理器"""
    logger.info(f"Loading model from {config.model_path}")

    model_class = get_model_class(config.model_path)
    logger.info(f"Using model class: {model_class.__name__}")

    # 量化配置
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

    model = model_class.from_pretrained(
        config.model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        config.model_path,
        trust_remote_code=True,
    )

    if config.use_4bit or config.use_8bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def create_reference_model(config: DPOConfig):
    """创建参考模型 (frozen)"""
    logger.info("Loading reference model...")

    model_class = get_model_class(config.model_path)

    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    ref_model = model_class.from_pretrained(
        config.model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    return ref_model


def main():
    parser = argparse.ArgumentParser(description="DPO fine-tuning for Qwen-VL")

    # Model arguments
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--no_4bit", action="store_false", dest="use_4bit", default=True,
                        help="Disable 4-bit quantization (QLoRA). Default: enabled")
    parser.add_argument("--use_8bit", action="store_true", default=False,
                        help="Use 8-bit quantization")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # Data arguments
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default="",
                        help="Path to validation data (optional)")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_image_size", type=int, default=512)

    # DPO arguments
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO temperature (控制偏好强度)")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="outputs/qwen3vl_dpo")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N steps (0 to disable)")

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_bf16", action="store_false", dest="bf16", default=True,
                        help="Disable bfloat16. Default: enabled")

    args = parser.parse_args()

    config = DPOConfig(
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
        beta=args.beta,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        seed=args.seed,
        bf16=args.bf16,
    )

    torch.manual_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    # 创建模型
    model, processor = create_model_and_processor(config)
    ref_model = create_reference_model(config)

    # 创建数据集
    dataset = DPODataset(
        config.train_data,
        processor,
        config.max_length,
        config.max_image_size
    )

    # 加载验证集（如果提供）
    val_dataset = None
    if config.val_data and os.path.exists(config.val_data):
        logger.info(f"Loading validation data from {config.val_data}")
        val_dataset = DPODataset(
            config.val_data,
            processor,
            config.max_length,
            config.max_image_size
        )
    else:
        logger.info("No validation data provided, skipping validation")

    # 创建训练器
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        processor=processor,
        config=config,
    )

    # 开始训练
    trainer.train(dataset, val_dataset=val_dataset)


if __name__ == "__main__":
    main()
