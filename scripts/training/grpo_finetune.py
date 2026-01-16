#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) Fine-tuning Script for Qwen-VL.

GRPO 是一种高效的强化学习微调方法，通过组内相对优势来优化策略，
不需要单独的奖励模型。

Usage:
    python scripts/grpo_finetune.py \
        --model_path Qwen/Qwen3-VL-2B-Instruct \
        --train_data data/qwen_data/train.json \
        --output_dir outputs/qwen3vl_grpo

References:
    - DeepSeekMath: https://arxiv.org/abs/2402.03300
    - TRL GRPO: https://huggingface.co/docs/trl/grpo_trainer
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
    GenerationConfig,
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
class GRPOConfig:
    """GRPO training configuration."""
    # Model
    model_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    sft_model_path: str = ""  # SFT 微调后的 LoRA 模型路径 (可选)
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
    train_data: str = "data/qwen_data/train.json"
    max_length: int = 2048
    max_new_tokens: int = 512  # 检测 JSON 输出通常不需要太长
    max_image_size: int = 512  # 图片最大边长

    # GRPO specific
    num_generations: int = 4  # 每个样本生成的响应数量 (G)
    temperature: float = 0.7  # 生成时的温度
    kl_coef: float = 0.1  # KL 散度系数
    clip_range: float = 0.2  # PPO-style clipping

    # Training
    output_dir: str = "outputs/qwen3vl_grpo"
    num_epochs: int = 1
    batch_size: int = 1  # 每个 batch 的样本数
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 200
    max_grad_norm: float = 1.0

    # Reward weights
    reward_format_weight: float = 1.0  # 格式正确性权重
    reward_bbox_weight: float = 2.0    # 边界框准确性权重
    reward_category_weight: float = 1.5  # 类别准确性权重
    reward_confidence_weight: float = 0.5  # 置信度校准权重

    # Other
    seed: int = 42
    bf16: bool = True


class RewardCalculator:
    """计算检测结果的奖励分数"""

    def __init__(self, config: GRPOConfig):
        self.config = config

    def parse_box_format(self, text: str) -> List[Dict]:
        """
        解析 <box>(x1,y1),(x2,y2)</box> 格式的检测结果

        返回: [{"category": "设备类型", "status": "正常/异常", "bbox": [x1,y1,x2,y2]}, ...]
        """
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

            # 提取坐标
            box_match = re.search(r'<box>\s*\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)\s*</box>', item)
            if not box_match:
                continue

            x1, y1, x2, y2 = int(box_match.group(1)), int(box_match.group(2)), int(box_match.group(3)), int(box_match.group(4))

            # 判断是否异常
            is_anomaly = any(kw in status for kw in ["异常", "全灭", "损坏", "故障", "破损", "不亮", "错误", "黑屏", "全亮"])

            detections.append({
                "category": category,
                "status": status,
                "is_anomaly": is_anomaly,
                "bbox": [x1, y1, x2, y2]
            })

        # 如果上面没匹配到，尝试只提取 box 坐标
        if not detections:
            box_pattern = r'<box>\s*\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)\s*</box>'
            simple_matches = re.findall(box_pattern, text)
            for match in simple_matches:
                try:
                    x1, y1, x2, y2 = int(match[0]), int(match[1]), int(match[2]), int(match[3])
                    detections.append({
                        "category": "unknown",
                        "status": "unknown",
                        "is_anomaly": False,
                        "bbox": [x1, y1, x2, y2]
                    })
                except ValueError:
                    continue

        return detections

    def compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个边界框的 IoU"""
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

    def compute_reward(
        self,
        generated_text: str,
        ground_truth: Dict,
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算生成结果的奖励分数

        Args:
            generated_text: 模型生成的文本
            ground_truth: 真实标注 (包含 detections 等字段)

        Returns:
            total_reward: 总奖励分数
            reward_breakdown: 各项奖励分解
        """
        rewards = {
            "format": 0.0,
            "bbox": 0.0,
            "category": 0.0,
            "completeness": 0.0,
        }

        # 解析生成的文本 (支持 <box> 格式)
        pred_detections = self.parse_box_format(generated_text)
        gt_detections = ground_truth.get("detections", [])

        # 1. 格式正确性奖励
        has_box = len(pred_detections) > 0
        has_structure = "检测到" in generated_text or "未检测到" in generated_text

        if has_box or has_structure:
            rewards["format"] = 1.0
        elif generated_text.strip():
            rewards["format"] = 0.3  # 有输出但格式不对
        else:
            rewards["format"] = -1.0  # 空输出

        # 如果没有真实检测
        if not gt_detections:
            if not pred_detections:
                # 正确预测没有设备
                rewards["bbox"] = 1.0
                rewards["category"] = 1.0
                rewards["completeness"] = 1.0
            else:
                # 预测了不存在的设备 (false positive)
                rewards["bbox"] = -0.3
                rewards["category"] = -0.3
                rewards["completeness"] = 0.5
        else:
            # 2. 边界框准确性奖励 (基于最佳匹配 IoU)
            if pred_detections:
                ious = []
                category_matches = []

                for gt_det in gt_detections:
                    gt_bbox = gt_det.get("bbox", [])
                    gt_category = gt_det.get("category", "")

                    best_iou = 0.0
                    best_cat_match = False

                    for pred_det in pred_detections:
                        pred_bbox = pred_det.get("bbox", [])
                        pred_category = pred_det.get("category", "")

                        if gt_bbox and pred_bbox:
                            iou = self.compute_iou(gt_bbox, pred_bbox)
                            if iou > best_iou:
                                best_iou = iou
                                # 类别模糊匹配
                                best_cat_match = (
                                    gt_category in pred_category or
                                    pred_category in gt_category or
                                    gt_category == pred_category
                                )

                    ious.append(best_iou)
                    category_matches.append(best_cat_match)

                # 平均 IoU 作为 bbox 奖励
                avg_iou = sum(ious) / len(ious) if ious else 0.0
                rewards["bbox"] = avg_iou * 2 - 1  # 映射到 [-1, 1]

                # 类别匹配率作为 category 奖励
                cat_match_rate = sum(category_matches) / len(category_matches) if category_matches else 0.0
                rewards["category"] = cat_match_rate * 2 - 1

                # 完整性奖励 (检测到的比例)
                completeness = min(len(pred_detections) / len(gt_detections), 1.0)
                rewards["completeness"] = completeness * 2 - 1
            else:
                # 没有预测任何检测 (false negative)
                rewards["bbox"] = -1.0
                rewards["category"] = -1.0
                rewards["completeness"] = -1.0

        # 计算总奖励
        weights = [
            self.config.reward_format_weight,
            self.config.reward_bbox_weight,
            self.config.reward_category_weight,
            self.config.reward_confidence_weight,  # 用于 completeness
        ]
        total = sum(r * w for r, w in zip(rewards.values(), weights))

        return total, rewards


class GRPODataset(Dataset):
    """GRPO 训练数据集"""

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
        conversations = item["conversations"]

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
            # 限制图片大小以减少显存占用
            if self.max_image_size and max(image.size) > self.max_image_size:
                ratio = self.max_image_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.LANCZOS)
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            image = Image.new("RGB", (224, 224), color="white")

        # Extract messages
        user_msg = conversations[0]["value"]
        assistant_msg = conversations[1]["value"]

        # 解析 ground truth
        ground_truth = self._parse_ground_truth(assistant_msg)

        return {
            "image": image,
            "prompt": user_msg.replace("<image>\n", "").replace("<image>", ""),
            "ground_truth": ground_truth,
            "reference_response": assistant_msg,
        }

    def _parse_ground_truth(self, response: str) -> Dict:
        """从参考响应中解析 ground truth (支持 <box> 格式)"""
        detections = []

        # 按序号分割每个检测项
        items = re.split(r'(?=\d+\.\s+)', response)

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

            # 提取坐标
            box_match = re.search(r'<box>\s*\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)\s*</box>', item)
            if not box_match:
                continue

            x1, y1, x2, y2 = int(box_match.group(1)), int(box_match.group(2)), int(box_match.group(3)), int(box_match.group(4))

            is_anomaly = any(kw in status for kw in ["异常", "全灭", "损坏", "故障", "破损", "不亮", "错误", "黑屏", "全亮"])

            detections.append({
                "category": category,
                "status": status,
                "is_anomaly": is_anomaly,
                "bbox": [x1, y1, x2, y2]
            })

        # 如果没匹配到完整格式，尝试只提取 box
        if not detections:
            box_pattern = r'<box>\s*\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)\s*</box>'
            simple_matches = re.findall(box_pattern, response)
            for match in simple_matches:
                try:
                    x1, y1, x2, y2 = int(match[0]), int(match[1]), int(match[2]), int(match[3])
                    detections.append({
                        "category": "unknown",
                        "bbox": [x1, y1, x2, y2]
                    })
                except ValueError:
                    continue

        has_anomaly = any(d.get("is_anomaly", False) for d in detections)

        return {
            "has_anomaly": has_anomaly,
            "detections": detections,
            "summary": response[:200]
        }


class GRPOTrainer:
    """GRPO 训练器"""

    def __init__(
        self,
        model,
        ref_model,
        processor,
        config: GRPOConfig,
        reward_calculator: RewardCalculator,
        total_steps: int,  # ✅ 新增：用于计算 warmup
    ):
        self.model = model
        self.ref_model = ref_model  # 参考模型（frozen）
        self.processor = processor
        self.config = config
        self.reward_calculator = reward_calculator

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # ✅ 新增：Warmup scheduler
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,  # 从 10% 学习率开始
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        logger.info(f"Warmup scheduler: {warmup_steps} steps (ratio={config.warmup_ratio})")

        self.global_step = 0

    @torch.no_grad()
    def generate_responses(
        self,
        image: Image.Image,
        prompt: str,
        num_generations: int,
    ) -> List[str]:
        """为单个样本生成多个响应"""
        # 生成时需要切换到 eval 模式，避免 gradient checkpointing 干扰
        was_training = self.model.training
        self.model.eval()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.model.device)

        # 批量生成所有响应 (更高效)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=0.9,
            num_return_sequences=num_generations,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        # 解码所有响应
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        responses = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # 恢复训练模式
        if was_training:
            self.model.train()

        return responses

    def _find_assistant_start(self, input_ids: torch.Tensor) -> int:
        """找到 assistant 回复开始的位置（与 SFT 相同的改进逻辑）"""
        tokenizer = self.processor.tokenizer
        input_ids_list = input_ids.tolist() if input_ids.dim() == 1 else input_ids[0].tolist()

        # 方法1: 使用 <|im_start|> 特殊 token
        try:
            im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
            if im_start_id is not None and im_start_id != tokenizer.unk_token_id:
                positions = [i for i, x in enumerate(input_ids_list) if x == im_start_id]

                if len(positions) >= 2:
                    last_start = positions[-1]
                    for offset in range(1, min(8, len(input_ids_list) - last_start)):
                        idx = last_start + offset
                        token_text = tokenizer.decode([input_ids_list[idx]], skip_special_tokens=False)
                        if '\n' in token_text:
                            return idx + 1
                    return last_start + 3
        except Exception:
            pass

        # 方法2: 搜索完整的 assistant prompt
        try:
            assistant_prompt = "<|im_start|>assistant\n"
            assistant_ids = tokenizer.encode(assistant_prompt, add_special_tokens=False)
            for i in range(len(input_ids_list) - len(assistant_ids) + 1):
                if input_ids_list[i:i+len(assistant_ids)] == assistant_ids:
                    return i + len(assistant_ids)
        except Exception:
            pass

        # 回退策略
        return int(len(input_ids_list) * 0.5)

    def compute_log_probs(
        self,
        model,
        image: Image.Image,
        prompt: str,
        response: str,
    ) -> torch.Tensor:
        """计算给定响应的 log 概率（修复版：正确 mask prompt + 返回总和）"""
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
            padding=True,
            truncation=False,
        ).to(model.device)

        # ✅ 关键修复1: 创建 labels 并 mask prompt 部分
        labels = inputs["input_ids"].clone()

        # 找到 assistant 开始位置
        assistant_start = self._find_assistant_start(inputs["input_ids"][0])

        # Mask prompt 部分（只训练 assistant 的回复）
        labels[0, :assistant_start] = -100

        # Mask padding tokens
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        # Forward pass
        outputs = model(**inputs, labels=labels)

        # ✅ 关键修复2: 计算有效 token 数量并返回总 log prob
        valid_tokens = (labels != -100).sum().float()

        # outputs.loss 是平均 negative log likelihood
        # 我们需要总的 log probability
        total_log_prob = -outputs.loss * valid_tokens

        return total_log_prob

    def grpo_step(
        self,
        image: Image.Image,
        prompt: str,
        ground_truth: Dict,
    ) -> Dict[str, float]:
        """执行单个 GRPO 更新步骤（修复版：添加 PPO clipping）"""
        step_start = time.time()

        # 1. 生成多个响应
        gen_start = time.time()
        responses = self.generate_responses(
            image, prompt, self.config.num_generations
        )
        gen_time = time.time() - gen_start

        # 2. 计算每个响应的奖励
        rewards = []
        reward_breakdowns = []
        for response in responses:
            reward, breakdown = self.reward_calculator.compute_reward(
                response, ground_truth
            )
            rewards.append(reward)
            reward_breakdowns.append(breakdown)

        rewards = torch.tensor(rewards, device=self.model.device)

        # 3. 计算组内相对优势 (Group Relative Advantage)
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        advantages = (rewards - mean_reward) / std_reward

        # ✅ 修复1: 先计算所有响应的初始 log probs（用于 PPO clipping）
        logprob_start = time.time()
        with torch.no_grad():
            old_log_probs = []
            ref_log_probs = []
            for response in responses:
                # 当前策略的初始 log prob
                old_log_prob = self.compute_log_probs(
                    self.model, image, prompt, response
                )
                old_log_probs.append(old_log_prob)

                # 参考策略的 log prob
                ref_log_prob = self.compute_log_probs(
                    self.ref_model, image, prompt, response
                )
                ref_log_probs.append(ref_log_prob)

        # 4. 计算策略损失（带 PPO clipping）
        total_loss = 0.0
        kl_divs = []

        for i, (response, advantage) in enumerate(zip(responses, advantages)):
            # 当前策略的 log 概率（可训练）
            log_prob = self.compute_log_probs(
                self.model, image, prompt, response
            )

            # ✅ 修复2: 计算 importance ratio
            old_log_prob = old_log_probs[i]
            ratio = torch.exp(log_prob - old_log_prob)

            # ✅ 修复3: PPO clipping
            clip_range = self.config.clip_range
            clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)

            # PPO 损失：取 min 以限制更新幅度
            # 当 advantage > 0: 限制 ratio 不超过 1+clip（防止过度增大概率）
            # 当 advantage < 0: 限制 ratio 不低于 1-clip（防止过度减小概率）
            surr1 = ratio * advantage
            surr2 = clipped_ratio * advantage
            policy_loss = -torch.min(surr1, surr2)  # ✅ 使用 min，加负号变成损失

            # ✅ 修复4: KL 散度惩罚（相对于参考模型）
            ref_log_prob = ref_log_probs[i]
            kl_div = log_prob - ref_log_prob
            kl_divs.append(kl_div.item())

            # 总损失 = 策略损失 + KL 惩罚
            loss = policy_loss + self.config.kl_coef * kl_div
            total_loss = total_loss + loss

        logprob_time = time.time() - logprob_start

        # 平均损失
        total_loss = total_loss / self.config.num_generations

        # ✅ 新增：梯度累积损失缩放
        total_loss = total_loss / self.config.gradient_accumulation_steps

        # 5. 反向传播
        backward_start = time.time()
        total_loss.backward()
        backward_time = time.time() - backward_start

        total_time = time.time() - step_start
        if self.global_step % 10 == 0:
            logger.info(
                f"Step timing: gen={gen_time:.1f}s, logprob={logprob_time:.1f}s, "
                f"backward={backward_time:.1f}s, total={total_time:.1f}s"
            )

        # 返回统计信息
        return {
            "loss": total_loss.item(),
            "mean_reward": mean_reward.item(),
            "std_reward": std_reward.item(),
            "max_reward": rewards.max().item(),
            "min_reward": rewards.min().item(),
            "mean_kl": sum(kl_divs) / len(kl_divs),
        }

    def train(self, dataset: GRPODataset):
        """训练循环（修复版：正确的梯度累积顺序）"""
        logger.info("Starting GRPO training...")

        # 自定义 collate_fn，保留 PIL Image 不做转换
        def collate_fn(batch):
            return {
                "image": [item["image"] for item in batch],
                "prompt": [item["prompt"] for item in batch],
                "ground_truth": [item["ground_truth"] for item in batch],
                "reference_response": [item["reference_response"] for item in batch],
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
        total_samples = 0

        # ✅ 修复: 在训练开始前 zero_grad
        self.optimizer.zero_grad()

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

            for batch_idx, batch in enumerate(pbar):
                if batch_idx == 0:
                    logger.info("Processing first batch...")

                # 遍历 batch 中的每个样本
                batch_stats = []
                for i in range(len(batch["image"])):
                    image = batch["image"][i]
                    prompt = batch["prompt"][i]
                    ground_truth = batch["ground_truth"][i]

                    # GRPO step（会调用 backward）
                    stats = self.grpo_step(image, prompt, ground_truth)
                    batch_stats.append(stats)
                    accumulated_steps += 1
                    total_samples += 1

                    # 按样本数保存检查点
                    if total_samples % self.config.save_steps == 0:
                        self.save_checkpoint(f"checkpoint-samples-{total_samples}")

                # 记录 batch 平均统计
                avg_stats = {
                    "loss": sum(s["loss"] for s in batch_stats) / len(batch_stats),
                    "mean_reward": sum(s["mean_reward"] for s in batch_stats) / len(batch_stats),
                    "mean_kl": sum(s.get("mean_kl", 0) for s in batch_stats) / len(batch_stats),
                }
                epoch_stats.append(avg_stats)

                if batch_idx == 0:
                    logger.info(
                        f"First batch completed: loss={avg_stats['loss']:.4f}, "
                        f"reward={avg_stats['mean_reward']:.4f}, kl={avg_stats['mean_kl']:.4f}"
                    )

                # ✅ 修复: 梯度累积逻辑
                if accumulated_steps % self.config.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                    # 执行优化步骤
                    self.optimizer.step()

                    # ✅ 新增：更新学习率（warmup）
                    self.scheduler.step()

                    # ✅ 关键: step 后立即 zero_grad
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # 日志
                    if self.global_step % self.config.logging_steps == 0:
                        recent_stats = epoch_stats[-self.config.logging_steps*self.config.gradient_accumulation_steps:]
                        if recent_stats:
                            avg_loss = sum(s["loss"] for s in recent_stats) / len(recent_stats)
                            avg_reward = sum(s["mean_reward"] for s in recent_stats) / len(recent_stats)
                            avg_kl = sum(s.get("mean_kl", 0) for s in recent_stats) / len(recent_stats)
                            current_lr = self.scheduler.get_last_lr()[0]
                            logger.info(
                                f"Step {self.global_step} (samples: {total_samples}): "
                                f"loss={avg_loss:.4f}, reward={avg_reward:.4f}, kl={avg_kl:.4f}, lr={current_lr:.2e}"
                            )

                    # 清理显存
                    if self.global_step % 5 == 0:
                        torch.cuda.empty_cache()

                # 更新进度条
                pbar.set_postfix({
                    "loss": f"{avg_stats['loss']:.4f}",
                    "reward": f"{avg_stats['mean_reward']:.4f}",
                    "kl": f"{avg_stats['mean_kl']:.4f}",
                    "samples": total_samples,
                })

        # 保存最终模型
        self.save_checkpoint("final")
        logger.info("GRPO training completed!")

    def save_checkpoint(self, name: str):
        """保存检查点"""
        save_path = os.path.join(self.config.output_dir, name)
        os.makedirs(save_path, exist_ok=True)

        # 保存 LoRA 权重
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

        # 保存配置
        config_path = os.path.join(save_path, "grpo_config.json")
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=2, default=str)

        logger.info(f"Checkpoint saved to {save_path}")


def create_model_and_processor(config: GRPOConfig):
    """创建模型和处理器"""
    logger.info(f"Loading base model from {config.model_path}")

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

    # 加载基础模型
    model = model_class.from_pretrained(
        config.model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        config.model_path,
        trust_remote_code=True,
    )

    # 准备 k-bit 训练
    if config.use_4bit or config.use_8bit:
        model = prepare_model_for_kbit_training(model)

    # 如果提供了 SFT 模型路径，先加载 SFT 的 LoRA 权重
    if config.sft_model_path and os.path.exists(config.sft_model_path):
        from peft import PeftModel
        logger.info(f"Loading SFT LoRA weights from {config.sft_model_path}")

        # 加载 SFT LoRA 权重
        model = PeftModel.from_pretrained(model, config.sft_model_path, is_trainable=True)
        logger.info("SFT LoRA weights loaded, continuing GRPO training on top of SFT")

        # 打印可训练参数
        model.print_trainable_parameters()
    else:
        # 从头开始配置 LoRA
        logger.info("No SFT model provided, starting GRPO from base model")
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


def create_reference_model(config: GRPOConfig):
    """创建参考模型 (frozen)"""
    logger.info("Loading reference model...")

    model_class = get_model_class(config.model_path)

    # 量化配置
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

    # 如果提供了 SFT 模型路径，参考模型也需要加载 SFT 权重
    if config.sft_model_path and os.path.exists(config.sft_model_path):
        from peft import PeftModel
        logger.info(f"Loading SFT LoRA weights for reference model from {config.sft_model_path}")
        ref_model = PeftModel.from_pretrained(ref_model, config.sft_model_path)
        ref_model = ref_model.merge_and_unload()  # 合并权重以提高推理速度
        logger.info("Reference model: SFT weights merged")

    # 冻结参考模型
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    return ref_model


def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning for Qwen-VL")

    # Model arguments
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--sft_model_path", type=str, default="",
                        help="Path to SFT fine-tuned LoRA model (optional, for continuing from SFT)")
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--use_8bit", action="store_true")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # Data arguments
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_image_size", type=int, default=512,
                        help="Maximum image size (longest edge)")

    # GRPO arguments
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--kl_coef", type=float, default=0.1)

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="outputs/qwen3vl_grpo")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (建议保持1，用gradient_accumulation代替)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)

    args = parser.parse_args()

    # 创建配置
    config = GRPOConfig(
        model_path=args.model_path,
        sft_model_path=args.sft_model_path,
        use_4bit=args.use_4bit and not args.use_8bit,
        use_8bit=args.use_8bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        train_data=args.train_data,
        max_length=args.max_length,
        max_image_size=args.max_image_size,
        num_generations=args.num_generations,
        temperature=args.temperature,
        kl_coef=args.kl_coef,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        bf16=args.bf16,
    )

    # 设置随机种子
    torch.manual_seed(config.seed)

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(config.output_dir, "grpo_config.json"), 'w') as f:
        json.dump(vars(config), f, indent=2, default=str)

    # 创建模型
    model, processor = create_model_and_processor(config)
    ref_model = create_reference_model(config)

    # 创建数据集
    dataset = GRPODataset(config.train_data, processor, config.max_length, config.max_image_size)

    # ✅ 计算总步数（用于 warmup scheduler）
    total_samples = len(dataset) * config.num_epochs
    total_steps = total_samples // config.gradient_accumulation_steps
    logger.info(f"Total training steps: {total_steps} (samples: {total_samples})")

    # 创建奖励计算器
    reward_calculator = RewardCalculator(config)

    # 创建训练器
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        processor=processor,
        config=config,
        reward_calculator=reward_calculator,
        total_steps=total_steps,  # ✅ 传递 total_steps
    )

    # 开始训练
    trainer.train(dataset)


if __name__ == "__main__":
    main()
