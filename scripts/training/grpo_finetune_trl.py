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
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

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

NO_DETECTION_PATTERNS = [
    "未检测到相关设备",
    "未检测到设备",
    "未检测到目标",
    "没有检测到",
    "no relevant equipment",
    "no equipment detected",
]

BOX_PATTERN = (
    r"<box>\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*,\s*"
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*</box>"
)

ANOMALY_KEYWORDS = ["异常", "全灭", "损坏", "故障", "破损", "不亮", "错误", "黑屏", "全亮"]


@dataclass
class ParsedDetection:
    """Structured detection parsed from text."""
    category: str
    status: str
    is_anomaly: bool
    bbox: List[int]


@dataclass
class RiskRewardConfig:
    """Config for risk-aware GRPO rewards."""
    match_iou_threshold: float = 0.5
    hallucination_unit_penalty: float = 0.35
    no_detection_missing_penalty: float = 0.2
    omission_penalty: float = 1.0


RISK_REWARD_CFG = RiskRewardConfig()


def _is_no_detection_response(text: str) -> bool:
    """Check whether the model explicitly says no equipment was detected."""
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in NO_DETECTION_PATTERNS)


def _has_structured_detection_format(text: str) -> bool:
    """Check whether completion follows numbered + status + box format."""
    has_box = bool(re.search(BOX_PATTERN, text))
    has_numbered = bool(re.search(r"\d+\.\s+\S+", text))
    has_status = bool(re.search(r"状态[：:]\s*\S+", text))
    return has_box and has_numbered and has_status


def _is_format_valid(completion: str, gt_response: Optional[str] = None) -> bool:
    """
    Strict format gate used by all reward functions.

    Valid cases:
    1. Standard structured detections (numbered + status + box).
    2. For no-box ground truth samples, explicit "no detection" response.
    """
    if _has_structured_detection_format(completion):
        return True

    if gt_response is None:
        return False

    gt_boxes = _extract_boxes(gt_response)
    if not gt_boxes and _is_no_detection_response(completion):
        return True

    return False


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

def format_reward(
    completions: List[str],
    assistant: Optional[List[str]] = None,
    **kwargs,
) -> List[float]:
    """
    Reward function that checks if the completion has proper format.
    Acts as a gating function: returns 0 if format is invalid, 1 if valid.

    Expected format:
    1. 设备类别
       状态：正常/异常
       <box>(x1,y1),(x2,y2)</box>
    """
    rewards = []

    for idx, completion in enumerate(completions):
        gt_response = assistant[idx] if assistant and idx < len(assistant) else None
        rewards.append(1.0 if _is_format_valid(completion, gt_response) else 0.0)

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
        # Strict gate: invalid format gets zero reward for all branches.
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_boxes = _extract_boxes(completion)
        gt_boxes = _extract_boxes(gt_response)

        if not gt_boxes:
            # No GT boxes: only explicit no-detection output gets full credit.
            if not pred_boxes and _is_no_detection_response(completion):
                reward = 1.0
            else:
                reward = 0.0
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
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_cats = _extract_categories(completion)
        gt_cats = _extract_categories(gt_response)

        if not gt_cats:
            reward = 1.0 if (not pred_cats and _is_no_detection_response(completion)) else 0.0
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
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_statuses = _extract_statuses(completion)
        gt_statuses = _extract_statuses(gt_response)

        if not gt_statuses:
            reward = 1.0 if (not pred_statuses and _is_no_detection_response(completion)) else 0.0
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


def set_f1_reward(completions: List[str], assistant: List[str], **kwargs) -> List[float]:
    """Set-level detection reward based on one-to-one TP/FP/FN F1."""
    rewards: List[float] = []

    for completion, gt_response in zip(completions, assistant):
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_dets = _extract_detections(completion)
        gt_dets = _extract_detections(gt_response)
        pred_count = len(pred_dets)
        gt_count = len(gt_dets)

        if gt_count == 0:
            if pred_count == 0 and _is_no_detection_response(completion):
                rewards.append(1.0)
            elif pred_count == 0:
                rewards.append(max(0.0, 1.0 - RISK_REWARD_CFG.no_detection_missing_penalty))
            else:
                rewards.append(0.0)
            continue

        matches = _match_detections(
            pred_dets,
            gt_dets,
            iou_threshold=RISK_REWARD_CFG.match_iou_threshold,
            require_category=True,
        )
        tp = len(matches)
        fp = max(pred_count - tp, 0)
        fn = max(gt_count - tp, 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rewards.append(_safe_f1(precision, recall))

    return rewards


def localization_quality_reward(
    completions: List[str], assistant: List[str], **kwargs
) -> List[float]:
    """Localization quality using IoU over matched pairs."""
    rewards: List[float] = []
    iou_thr = RISK_REWARD_CFG.match_iou_threshold
    denom = max(1e-6, 1.0 - iou_thr)

    for completion, gt_response in zip(completions, assistant):
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_dets = _extract_detections(completion)
        gt_dets = _extract_detections(gt_response)

        if not gt_dets:
            if not pred_dets and _is_no_detection_response(completion):
                rewards.append(1.0)
            elif not pred_dets:
                rewards.append(max(0.0, 1.0 - RISK_REWARD_CFG.no_detection_missing_penalty))
            else:
                rewards.append(0.0)
            continue

        matches = _match_detections(
            pred_dets,
            gt_dets,
            iou_threshold=iou_thr,
            require_category=True,
        )

        if not matches:
            rewards.append(0.0)
            continue

        mean_iou = sum(m["iou"] for m in matches) / len(matches)
        norm_iou = max(0.0, min(1.0, (mean_iou - iou_thr) / denom))
        rewards.append(norm_iou)

    return rewards


def count_alignment_reward(completions: List[str], assistant: List[str], **kwargs) -> List[float]:
    """Reward count consistency and penalize severe over/under-prediction."""
    rewards: List[float] = []

    for completion, gt_response in zip(completions, assistant):
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_count = len(_extract_detections(completion))
        gt_count = len(_extract_detections(gt_response))

        if gt_count == 0:
            if pred_count == 0 and _is_no_detection_response(completion):
                rewards.append(1.0)
            elif pred_count == 0:
                rewards.append(max(0.0, 1.0 - RISK_REWARD_CFG.no_detection_missing_penalty))
            else:
                penalty = min(1.0, pred_count * RISK_REWARD_CFG.hallucination_unit_penalty)
                rewards.append(-penalty)
            continue

        if _is_no_detection_response(completion) and pred_count == 0:
            rewards.append(-min(1.0, RISK_REWARD_CFG.omission_penalty))
            continue

        ratio_err = abs(pred_count - gt_count) / max(gt_count, 1)
        rewards.append(max(-1.0, min(1.0, 1.0 - ratio_err)))

    return rewards


def risk_control_reward(completions: List[str], assistant: List[str], **kwargs) -> List[float]:
    """
    Risk-aware reward:
    - penalize hallucinations on empty-GT samples
    - penalize omissions on positive samples
    """
    rewards: List[float] = []

    for completion, gt_response in zip(completions, assistant):
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_dets = _extract_detections(completion)
        gt_dets = _extract_detections(gt_response)
        pred_count = len(pred_dets)
        gt_count = len(gt_dets)
        no_detection = _is_no_detection_response(completion)

        if gt_count == 0:
            if pred_count == 0 and no_detection:
                rewards.append(1.0)
            elif pred_count == 0:
                rewards.append(max(0.0, 1.0 - RISK_REWARD_CFG.no_detection_missing_penalty))
            else:
                halluc_penalty = min(1.0, pred_count * RISK_REWARD_CFG.hallucination_unit_penalty)
                rewards.append(-halluc_penalty)
            continue

        if pred_count == 0:
            rewards.append(-min(1.0, RISK_REWARD_CFG.omission_penalty))
            continue

        matches = _match_detections(
            pred_dets,
            gt_dets,
            iou_threshold=RISK_REWARD_CFG.match_iou_threshold,
            require_category=False,
        )
        covered_gt = {m["gt_idx"] for m in matches}
        covered_pred = {m["pred_idx"] for m in matches}

        omission_rate = 1.0 - (len(covered_gt) / max(gt_count, 1))
        hallucinated = max(pred_count - len(covered_pred), 0)
        reward = (
            1.0
            - RISK_REWARD_CFG.omission_penalty * omission_rate
            - min(1.0, hallucinated * RISK_REWARD_CFG.hallucination_unit_penalty)
        )
        if no_detection:
            reward -= RISK_REWARD_CFG.omission_penalty

        rewards.append(max(-1.0, min(1.0, reward)))

    return rewards


def anomaly_instance_f1_reward(
    completions: List[str], assistant: List[str], **kwargs
) -> List[float]:
    """
    Instance-level anomaly reward on matched detections.
    Emphasizes anomaly recall under one-to-one localization/category matches.
    """
    rewards: List[float] = []

    for completion, gt_response in zip(completions, assistant):
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_dets = _extract_detections(completion)
        gt_dets = _extract_detections(gt_response)
        no_detection = _is_no_detection_response(completion)

        if not gt_dets:
            if not pred_dets and no_detection:
                rewards.append(1.0)
            elif not pred_dets:
                rewards.append(max(0.0, 1.0 - RISK_REWARD_CFG.no_detection_missing_penalty))
            else:
                false_alarm = sum(1 for det in pred_dets if det.is_anomaly)
                rewards.append(-min(1.0, 0.5 * false_alarm))
            continue

        matches = _match_detections(
            pred_dets,
            gt_dets,
            iou_threshold=RISK_REWARD_CFG.match_iou_threshold,
            require_category=True,
        )
        if not matches:
            gt_has_anomaly = any(det.is_anomaly for det in gt_dets)
            rewards.append(-1.0 if gt_has_anomaly else 0.0)
            continue

        matched_pred_indices = {m["pred_idx"] for m in matches}
        gt_anomaly_total = sum(1 for det in gt_dets if det.is_anomaly)
        pred_anomaly_total = 0
        tp_anomaly = 0

        for m in matches:
            pred_anomaly = pred_dets[m["pred_idx"]].is_anomaly
            gt_anomaly = gt_dets[m["gt_idx"]].is_anomaly
            if pred_anomaly:
                pred_anomaly_total += 1
            if pred_anomaly and gt_anomaly:
                tp_anomaly += 1

        unmatched_pred_anomaly = sum(
            1
            for idx, det in enumerate(pred_dets)
            if idx not in matched_pred_indices and det.is_anomaly
        )
        fp_anomaly = max(pred_anomaly_total - tp_anomaly, 0) + unmatched_pred_anomaly
        fn_anomaly = max(gt_anomaly_total - tp_anomaly, 0)

        if gt_anomaly_total == 0:
            rewards.append(1.0 if fp_anomaly == 0 else max(-1.0, 1.0 - 0.5 * fp_anomaly))
            continue

        if no_detection and len(pred_dets) == 0:
            rewards.append(-1.0)
            continue

        precision = tp_anomaly / (tp_anomaly + fp_anomaly) if (tp_anomaly + fp_anomaly) > 0 else 0.0
        recall = tp_anomaly / (tp_anomaly + fn_anomaly) if (tp_anomaly + fn_anomaly) > 0 else 0.0
        # Slightly favor recall to reduce anomaly misses.
        reward = 0.4 * precision + 0.6 * recall
        rewards.append(max(-1.0, min(1.0, reward)))

    return rewards


# ============ Helper Functions ============

def _extract_boxes(text: str) -> List[List[int]]:
    """Extract bounding boxes from text. Supports integers, floats, and negative numbers."""
    boxes = []
    for match in re.finditer(BOX_PATTERN, text):
        try:
            # Convert to int (round floats)
            box = [int(float(match.group(i))) for i in range(1, 5)]
            boxes.append(box)
        except (ValueError, TypeError):
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


def _normalize_category(category: str) -> str:
    """Normalize category text for fuzzy matching."""
    return re.sub(r"\s+", "", category.strip().lower())


def _is_anomaly_status(status: str) -> bool:
    """Detect anomaly status by keywords."""
    return any(kw in status for kw in ANOMALY_KEYWORDS)


def _category_match(pred_category: str, gt_category: str) -> bool:
    """Fuzzy category match."""
    pred = _normalize_category(pred_category)
    gt = _normalize_category(gt_category)
    if not pred or not gt:
        return False
    if pred == "unknown" or gt == "unknown":
        return False
    return pred == gt or pred in gt or gt in pred


def _extract_detections(text: str) -> List[ParsedDetection]:
    """Parse structured detections with category/status/bbox."""
    detections: List[ParsedDetection] = []
    items = re.split(r"(?=\d+\.\s+)", text)

    for item in items:
        if not item.strip():
            continue

        cat_match = re.match(r"(\d+)\.\s*([^\n]+)", item)
        if not cat_match:
            continue

        category = cat_match.group(2).strip()
        status_match = re.search(r"状态[：:]\s*([^\n]+)", item)
        status = status_match.group(1).strip() if status_match else "正常"

        box_match = re.search(BOX_PATTERN, item)
        if not box_match:
            continue

        try:
            bbox = [int(float(box_match.group(i))) for i in range(1, 5)]
        except (ValueError, TypeError):
            continue

        detections.append(
            ParsedDetection(
                category=category,
                status=status,
                is_anomaly=_is_anomaly_status(status),
                bbox=bbox,
            )
        )

    # Fallback: box-only parsing
    if not detections:
        for box in _extract_boxes(text):
            detections.append(
                ParsedDetection(
                    category="unknown",
                    status="unknown",
                    is_anomaly=False,
                    bbox=box,
                )
            )

    return detections


def _match_detections(
    pred_dets: List[ParsedDetection],
    gt_dets: List[ParsedDetection],
    iou_threshold: float,
    require_category: bool = True,
) -> List[Dict[str, Any]]:
    """
    Greedy one-to-one matching by IoU.
    Returns matched pairs with pred_idx/gt_idx/iou/category_ok.
    """
    candidates: List[Tuple[float, int, int, bool]] = []
    for pred_idx, pred_det in enumerate(pred_dets):
        for gt_idx, gt_det in enumerate(gt_dets):
            iou = _compute_iou(pred_det.bbox, gt_det.bbox)
            if iou < iou_threshold:
                continue
            category_ok = _category_match(pred_det.category, gt_det.category)
            if require_category and not category_ok:
                continue
            # Prefer higher IoU, then category-consistent matches.
            score = iou + (0.01 if category_ok else 0.0)
            candidates.append((score, pred_idx, gt_idx, category_ok))

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_pred = set()
    used_gt = set()
    matches: List[Dict[str, Any]] = []

    for _, pred_idx, gt_idx, category_ok in candidates:
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        iou = _compute_iou(pred_dets[pred_idx].bbox, gt_dets[gt_idx].bbox)
        matches.append(
            {
                "pred_idx": pred_idx,
                "gt_idx": gt_idx,
                "iou": iou,
                "category_ok": category_ok,
            }
        )

    return matches


def _safe_f1(precision: float, recall: float) -> float:
    """F1 helper."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


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
    parser.add_argument(
        "--reward_scheme",
        type=str,
        default="risk_aware",
        choices=["risk_aware", "legacy"],
        help="Reward design: risk_aware (recommended) or legacy",
    )
    parser.add_argument(
        "--reward_match_iou",
        type=float,
        default=0.5,
        help="IoU threshold for one-to-one matching in risk-aware rewards",
    )
    parser.add_argument(
        "--reward_hallucination_unit_penalty",
        type=float,
        default=0.35,
        help="Per-instance hallucination penalty (risk-aware rewards)",
    )
    parser.add_argument(
        "--reward_no_detection_missing_penalty",
        type=float,
        default=0.2,
        help="Penalty when GT is empty but model does not output explicit no-detection text",
    )
    parser.add_argument(
        "--reward_omission_penalty",
        type=float,
        default=1.0,
        help="Penalty strength for missing detections on positive samples",
    )
    parser.add_argument("--reward_w_format", type=float, default=0.2)
    parser.add_argument("--reward_w_set_f1", type=float, default=3.0)
    parser.add_argument("--reward_w_iou", type=float, default=2.0)
    parser.add_argument("--reward_w_count", type=float, default=1.2)
    parser.add_argument("--reward_w_risk", type=float, default=2.5)
    parser.add_argument("--reward_w_anomaly", type=float, default=2.0)

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

    # Configure risk-aware reward parameters (used when reward_scheme=risk_aware)
    global RISK_REWARD_CFG
    RISK_REWARD_CFG = RiskRewardConfig(
        match_iou_threshold=max(0.1, min(0.95, float(args.reward_match_iou))),
        hallucination_unit_penalty=max(0.0, float(args.reward_hallucination_unit_penalty)),
        no_detection_missing_penalty=max(0.0, float(args.reward_no_detection_missing_penalty)),
        omission_penalty=max(0.0, float(args.reward_omission_penalty)),
    )

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
    if args.reward_scheme == "legacy":
        reward_funcs = [
            format_reward,
            bbox_iou_reward,
            category_match_reward,
            status_accuracy_reward,
        ]
        reward_weights = [0.2, 3.0, 2.0, 2.0]
        logger.info("Using LEGACY reward scheme")
    else:
        reward_funcs = [
            format_reward,
            set_f1_reward,
            localization_quality_reward,
            count_alignment_reward,
            risk_control_reward,
            anomaly_instance_f1_reward,
        ]
        reward_weights = [
            args.reward_w_format,
            args.reward_w_set_f1,
            args.reward_w_iou,
            args.reward_w_count,
            args.reward_w_risk,
            args.reward_w_anomaly,
        ]
        logger.info("Using RISK-AWARE reward scheme")
        logger.info(
            "Risk-aware reward cfg: match_iou=%.2f, halluc_penalty=%.2f, "
            "no_det_missing_penalty=%.2f, omission_penalty=%.2f",
            RISK_REWARD_CFG.match_iou_threshold,
            RISK_REWARD_CFG.hallucination_unit_penalty,
            RISK_REWARD_CFG.no_detection_missing_penalty,
            RISK_REWARD_CFG.omission_penalty,
        )
        logger.info("Risk-aware reward weights: %s", reward_weights)

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
