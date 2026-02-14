#!/usr/bin/env python3
"""
目标检测评估脚本 - 计算标准检测指标

支持的指标:
- AP50 (IoU=0.50)
- AP75 (IoU=0.75)
- mAP50-95 (COCO 风格，IoU=0.50:0.05:0.95)
- Precision / Recall / F1-Score
- AP per class
- IoU 分布统计
- VLM 质量指标（EM/Token-F1/ROUGE/BLEU/异常检测与计数误差等）

Usage:
    # 评估微调后的模型
    python scripts/evaluate.py \
        --model_path outputs/qwen3vl_lora \
        --test_data data/qwen_data/test.json \
        --output_dir eval_results/

    # 评估基础模型
    python scripts/evaluate.py \
        --model_path Qwen/Qwen3-VL-2B-Instruct \
        --test_data data/qwen_data/test.json \
        --iou_threshold 0.5

    # 计算 COCO 风格 mAP
    python scripts/evaluate.py \
        --model_path outputs/qwen3vl_lora \
        --test_data data/qwen_data/test.json \
        --coco_map
"""

import os
import re
import json
import time
import math
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

COCO_IOU_THRESHOLDS = [round(0.5 + i * 0.05, 2) for i in range(10)]
NO_OBJECT_PATTERNS = [
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
class Detection:
    """单个检测结果"""
    category: str
    bbox: List[float]  # [x1, y1, x2, y2] in 0-1000 scale
    confidence: float = 1.0
    status: str = "unknown"
    is_anomaly: bool = False


@dataclass
class EvalMetrics:
    """评估指标"""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    ap: float = 0.0  # Average Precision
    tp: int = 0  # True Positives
    fp: int = 0  # False Positives
    fn: int = 0  # False Negatives
    iou_mean: float = 0.0
    iou_std: float = 0.0
    num_pred: int = 0
    num_gt: int = 0


@dataclass
class VLMEvalMetrics:
    """VLM 常规质量指标（与检测 AP 互补）"""
    parse_success_rate: float = 0.0
    strict_format_rate: float = 0.0
    no_object_rejection_rate: float = 0.0
    hallucination_rate: float = 0.0
    false_rejection_rate: float = 0.0
    omission_rate: float = 0.0
    anomaly_accuracy: float = 0.0
    anomaly_precision: float = 0.0
    anomaly_recall: float = 0.0
    anomaly_f1: float = 0.0
    category_precision: float = 0.0
    category_recall: float = 0.0
    category_f1: float = 0.0
    exact_match_rate: float = 0.0
    token_f1: float = 0.0
    rouge_l_f1: float = 0.0
    bleu1: float = 0.0
    bleu4: float = 0.0
    count_mae: float = 0.0
    count_rmse: float = 0.0
    avg_response_tokens: float = 0.0
    avg_pred_boxes: float = 0.0
    avg_gt_boxes: float = 0.0
    latency_ms_mean: float = 0.0
    latency_ms_p50: float = 0.0
    latency_ms_p95: float = 0.0


class DetectionParser:
    """解析模型输出的检测结果"""

    @staticmethod
    def parse_box_format(text: str) -> List[Detection]:
        """
        解析 <box>(x1,y1),(x2,y2)</box> 格式的检测结果

        返回: List[Detection]
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

            # 提取置信度（可选）
            confidence = 1.0
            conf_match = re.search(
                r'(?:confidence|置信度)\s*[：:]\s*([01](?:\.\d+)?|\d{1,3}(?:\.\d+)?%)',
                item,
                flags=re.IGNORECASE,
            )
            if conf_match:
                conf_text = conf_match.group(1).strip()
                try:
                    if conf_text.endswith("%"):
                        confidence = float(conf_text[:-1]) / 100.0
                    else:
                        confidence = float(conf_text)
                        if confidence > 1.0:
                            confidence = confidence / 100.0
                    confidence = float(np.clip(confidence, 0.0, 1.0))
                except ValueError:
                    confidence = 1.0

            # 提取坐标 (支持浮点数和负数)
            box_match = re.search(
                BOX_PATTERN,
                item
            )
            if not box_match:
                continue

            x1 = int(float(box_match.group(1)))
            y1 = int(float(box_match.group(2)))
            x2 = int(float(box_match.group(3)))
            y2 = int(float(box_match.group(4)))

            # 判断是否异常
            is_anomaly = any(
                kw in status
                for kw in ["异常", "全灭", "损坏", "故障", "破损", "不亮", "错误", "黑屏", "全亮"]
            )

            detections.append(Detection(
                category=category,
                bbox=[x1, y1, x2, y2],
                confidence=confidence,
                status=status,
                is_anomaly=is_anomaly,
            ))

        # 如果上面没匹配到，尝试只提取 box 坐标 (支持浮点数和负数)
        if not detections:
            simple_matches = re.findall(BOX_PATTERN, text)
            for match in simple_matches:
                try:
                    x1 = int(float(match[0]))
                    y1 = int(float(match[1]))
                    x2 = int(float(match[2]))
                    y2 = int(float(match[3]))
                    detections.append(Detection(
                        category="unknown",
                        bbox=[x1, y1, x2, y2],
                        status="unknown",
                    ))
                except (ValueError, TypeError):
                    continue

        return detections


def normalize_text(text: str) -> str:
    """Normalize text for VLM quality metrics."""
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize_text_for_metrics(text: str) -> List[str]:
    """
    Tokenize text for BLEU/ROUGE:
    - Use whitespace tokens when spaces exist.
    - Fallback to char-level tokens for Chinese-like text.
    """
    normalized = normalize_text(text)
    if not normalized:
        return []
    if " " in normalized:
        return [tok for tok in normalized.split(" ") if tok]
    return [ch for ch in normalized if not ch.isspace()]


def extract_ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    """Extract n-gram counts."""
    counts: Dict[Tuple[str, ...], int] = defaultdict(int)
    if n <= 0 or len(tokens) < n:
        return counts
    for i in range(len(tokens) - n + 1):
        counts[tuple(tokens[i : i + n])] += 1
    return counts


def compute_corpus_bleu(
    pred_token_lists: List[List[str]],
    ref_token_lists: List[List[str]],
    max_n: int = 4,
) -> float:
    """Compute smoothed corpus BLEU score."""
    if not pred_token_lists or not ref_token_lists:
        return 0.0

    total_pred_len = sum(len(tokens) for tokens in pred_token_lists)
    total_ref_len = sum(len(tokens) for tokens in ref_token_lists)
    if total_pred_len == 0:
        return 0.0

    if total_pred_len > total_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (total_ref_len / max(total_pred_len, 1)))

    log_p_sum = 0.0
    for n in range(1, max_n + 1):
        clipped_total = 0
        pred_total = 0

        for pred_tokens, ref_tokens in zip(pred_token_lists, ref_token_lists):
            pred_counts = extract_ngram_counts(pred_tokens, n)
            ref_counts = extract_ngram_counts(ref_tokens, n)

            pred_total += sum(pred_counts.values())
            for gram, cnt in pred_counts.items():
                clipped_total += min(cnt, ref_counts.get(gram, 0))

        # Add-1 smoothing
        p_n = (clipped_total + 1.0) / (pred_total + 1.0)
        log_p_sum += math.log(p_n)

    return float(bp * math.exp(log_p_sum / max_n))


def lcs_length(a: List[str], b: List[str]) -> int:
    """Compute LCS length with O(min(m,n)) memory."""
    if len(a) < len(b):
        a, b = b, a

    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def compute_rouge_l_f1(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    """Compute ROUGE-L F1 for a single sample."""
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compute_token_f1(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    """Compute token-level F1 (SQuAD-style overlap)."""
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum(min(cnt, ref_counts.get(tok, 0)) for tok, cnt in pred_counts.items())
    if overlap <= 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def is_no_object_response(text: str) -> bool:
    """Check whether text explicitly indicates no object/equipment."""
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in NO_OBJECT_PATTERNS)


def is_strict_detection_format(text: str) -> bool:
    """Check strict structured detection format."""
    has_box = bool(re.search(BOX_PATTERN, text))
    has_numbered = bool(re.search(r"\d+\.\s+\S+", text))
    has_status = bool(re.search(r"状态[：:]\s*\S+", text))
    return has_box and has_numbered and has_status


def compute_vlm_metrics(samples_for_eval: List[Dict]) -> VLMEvalMetrics:
    """Compute VLM-oriented quality metrics for paper reporting."""
    total = len(samples_for_eval)
    if total == 0:
        return VLMEvalMetrics()

    parse_success = 0
    strict_format = 0
    no_object_total = 0
    no_object_rejected = 0
    hallucinations = 0
    non_empty_total = 0
    false_rejections = 0
    omissions = 0
    anomaly_correct = 0
    anomaly_tp = 0
    anomaly_fp = 0
    anomaly_fn = 0

    category_tp = 0
    category_fp = 0
    category_fn = 0

    exact_match = 0
    token_f1_scores: List[float] = []
    rouge_scores: List[float] = []
    pred_token_lists: List[List[str]] = []
    ref_token_lists: List[List[str]] = []
    response_tokens_total = 0

    pred_boxes_total = 0
    gt_boxes_total = 0
    count_abs_errors: List[float] = []
    count_sq_errors: List[float] = []

    latency_ms: List[float] = []

    for sample in samples_for_eval:
        pred_text = sample.get("pred_text", "")
        gt_text = sample.get("gt_text", "")
        pred_dets: List[Detection] = sample.get("pred_dets", [])
        gt_dets: List[Detection] = sample.get("gt_dets", [])

        gt_empty = len(gt_dets) == 0
        no_object_pred = is_no_object_response(pred_text)
        strict = is_strict_detection_format(pred_text)

        if strict:
            strict_format += 1

        # Parse success: parsed detections or explicit no-object declaration for empty GT.
        parsed_ok = (len(pred_dets) > 0) or (gt_empty and no_object_pred)
        if parsed_ok:
            parse_success += 1

        if gt_empty:
            no_object_total += 1
            if no_object_pred and len(pred_dets) == 0:
                no_object_rejected += 1
            if len(pred_dets) > 0:
                hallucinations += 1
        else:
            non_empty_total += 1
            if no_object_pred:
                false_rejections += 1
            if len(pred_dets) == 0:
                omissions += 1

        gt_has_anomaly = any(det.is_anomaly for det in gt_dets)
        pred_has_anomaly = any(det.is_anomaly for det in pred_dets)
        if gt_has_anomaly == pred_has_anomaly:
            anomaly_correct += 1
        if pred_has_anomaly and gt_has_anomaly:
            anomaly_tp += 1
        elif pred_has_anomaly and not gt_has_anomaly:
            anomaly_fp += 1
        elif (not pred_has_anomaly) and gt_has_anomaly:
            anomaly_fn += 1

        gt_cats = {DetectionEvaluator._normalize_category(det.category) for det in gt_dets}
        pred_cats = {DetectionEvaluator._normalize_category(det.category) for det in pred_dets}
        category_tp += len(gt_cats & pred_cats)
        category_fp += len(pred_cats - gt_cats)
        category_fn += len(gt_cats - pred_cats)

        pred_tokens = tokenize_text_for_metrics(pred_text)
        ref_tokens = tokenize_text_for_metrics(gt_text)
        if normalize_text(pred_text) == normalize_text(gt_text):
            exact_match += 1
        token_f1_scores.append(compute_token_f1(pred_tokens, ref_tokens))
        pred_token_lists.append(pred_tokens)
        ref_token_lists.append(ref_tokens)
        rouge_scores.append(compute_rouge_l_f1(pred_tokens, ref_tokens))
        response_tokens_total += len(pred_tokens)

        pred_boxes_total += len(pred_dets)
        gt_boxes_total += len(gt_dets)
        count_error = float(len(pred_dets) - len(gt_dets))
        count_abs_errors.append(abs(count_error))
        count_sq_errors.append(count_error * count_error)

        if "latency_ms" in sample:
            latency_ms.append(float(sample["latency_ms"]))

    category_precision = (
        category_tp / (category_tp + category_fp) if (category_tp + category_fp) > 0 else 0.0
    )
    category_recall = (
        category_tp / (category_tp + category_fn) if (category_tp + category_fn) > 0 else 0.0
    )
    category_f1 = (
        2 * category_precision * category_recall / (category_precision + category_recall)
        if (category_precision + category_recall) > 0
        else 0.0
    )
    anomaly_precision = (
        anomaly_tp / (anomaly_tp + anomaly_fp) if (anomaly_tp + anomaly_fp) > 0 else 0.0
    )
    anomaly_recall = (
        anomaly_tp / (anomaly_tp + anomaly_fn) if (anomaly_tp + anomaly_fn) > 0 else 0.0
    )
    anomaly_f1 = (
        2 * anomaly_precision * anomaly_recall / (anomaly_precision + anomaly_recall)
        if (anomaly_precision + anomaly_recall) > 0
        else 0.0
    )

    if latency_ms:
        latency_arr = np.array(latency_ms, dtype=np.float32)
        latency_mean = float(np.mean(latency_arr))
        latency_p50 = float(np.percentile(latency_arr, 50))
        latency_p95 = float(np.percentile(latency_arr, 95))
    else:
        latency_mean = 0.0
        latency_p50 = 0.0
        latency_p95 = 0.0

    return VLMEvalMetrics(
        parse_success_rate=parse_success / total,
        strict_format_rate=strict_format / total,
        no_object_rejection_rate=(
            no_object_rejected / no_object_total if no_object_total > 0 else 0.0
        ),
        hallucination_rate=(
            hallucinations / no_object_total if no_object_total > 0 else 0.0
        ),
        false_rejection_rate=(
            false_rejections / non_empty_total if non_empty_total > 0 else 0.0
        ),
        omission_rate=(
            omissions / non_empty_total if non_empty_total > 0 else 0.0
        ),
        anomaly_accuracy=anomaly_correct / total,
        anomaly_precision=anomaly_precision,
        anomaly_recall=anomaly_recall,
        anomaly_f1=anomaly_f1,
        category_precision=category_precision,
        category_recall=category_recall,
        category_f1=category_f1,
        exact_match_rate=exact_match / total,
        token_f1=float(np.mean(token_f1_scores)) if token_f1_scores else 0.0,
        rouge_l_f1=float(np.mean(rouge_scores)) if rouge_scores else 0.0,
        bleu1=compute_corpus_bleu(pred_token_lists, ref_token_lists, max_n=1),
        bleu4=compute_corpus_bleu(pred_token_lists, ref_token_lists, max_n=4),
        count_mae=float(np.mean(count_abs_errors)) if count_abs_errors else 0.0,
        count_rmse=float(np.sqrt(np.mean(count_sq_errors))) if count_sq_errors else 0.0,
        avg_response_tokens=response_tokens_total / total,
        avg_pred_boxes=pred_boxes_total / total,
        avg_gt_boxes=gt_boxes_total / total,
        latency_ms_mean=latency_mean,
        latency_ms_p50=latency_p50,
        latency_ms_p95=latency_p95,
    )


class DetectionEvaluator:
    """目标检测评估器（COCO/YOLO 风格 AP 计算）"""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.parser = DetectionParser()

    @staticmethod
    def compute_iou(box1: List[float], box2: List[float]) -> float:
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

    @staticmethod
    def _normalize_category(category: str) -> str:
        """归一化类别名称（用于匹配）"""
        return category.strip().lower()

    @staticmethod
    def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
        """
        计算 AP（COCO/YOLO 常见 101-point interpolation）。
        """
        if recalls.size == 0 or precisions.size == 0:
            return 0.0

        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        # Precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        recall_points = np.linspace(0.0, 1.0, 101)
        ap = 0.0
        for r in recall_points:
            valid = np.where(mrec >= r)[0]
            ap += np.max(mpre[valid]) if valid.size > 0 else 0.0
        return float(ap / len(recall_points))

    def _evaluate_single_class(
        self,
        preds: List[Dict],
        gt_by_image: Dict[str, List[List[float]]],
        iou_threshold: float,
    ) -> Tuple[EvalMetrics, List[float], bool]:
        """
        评估单个类别（按 confidence 排序，计算 AP/PR/F1）。
        """
        num_gt = sum(len(v) for v in gt_by_image.values())
        num_pred = len(preds)

        if num_pred > 0:
            preds_sorted = sorted(preds, key=lambda x: x["confidence"], reverse=True)
        else:
            preds_sorted = []

        tp_flags = np.zeros(num_pred, dtype=np.float32)
        fp_flags = np.zeros(num_pred, dtype=np.float32)
        matched_ious: List[float] = []

        # 每个 IoU 阈值下，一张图中的每个 GT 最多匹配一次
        matched = {
            image_id: np.zeros(len(boxes), dtype=bool)
            for image_id, boxes in gt_by_image.items()
        }

        for i, pred in enumerate(preds_sorted):
            image_id = pred["image_id"]
            pred_bbox = pred["bbox"]
            gt_boxes = gt_by_image.get(image_id, [])

            if not gt_boxes:
                fp_flags[i] = 1.0
                continue

            ious = np.array([self.compute_iou(pred_bbox, gt_bbox) for gt_bbox in gt_boxes])
            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])

            if best_iou >= iou_threshold and not matched[image_id][best_idx]:
                tp_flags[i] = 1.0
                matched[image_id][best_idx] = True
                matched_ious.append(best_iou)
            else:
                fp_flags[i] = 1.0

        tp_cum = np.cumsum(tp_flags)
        fp_cum = np.cumsum(fp_flags)

        if num_pred > 0:
            precisions_curve = tp_cum / (tp_cum + fp_cum + 1e-9)
        else:
            precisions_curve = np.array([], dtype=np.float32)

        if num_gt > 0:
            recalls_curve = tp_cum / (num_gt + 1e-9)
            ap = self._compute_ap(recalls_curve, precisions_curve)
        else:
            recalls_curve = np.array([], dtype=np.float32)
            ap = 0.0

        tp = int(tp_cum[-1]) if tp_cum.size > 0 else 0
        fp = int(fp_cum[-1]) if fp_cum.size > 0 else 0
        fn = max(num_gt - tp, 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        iou_mean = float(np.mean(matched_ious)) if matched_ious else 0.0
        iou_std = float(np.std(matched_ious)) if matched_ious else 0.0

        metrics = EvalMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            ap=ap,
            tp=tp,
            fp=fp,
            fn=fn,
            iou_mean=iou_mean,
            iou_std=iou_std,
            num_pred=num_pred,
            num_gt=num_gt,
        )
        return metrics, matched_ious, num_gt > 0

    def evaluate_dataset(
        self,
        samples: List[Dict],
        iou_threshold: float,
    ) -> Dict[str, Dict]:
        """
        评估整个数据集，在指定 IoU 阈值下输出 overall/per-class 指标。
        """
        class_preds: Dict[str, List[Dict]] = defaultdict(list)
        class_gts: Dict[str, Dict[str, List[List[float]]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for sample in samples:
            image_id = sample["image_id"]
            for det in sample["pred_dets"]:
                cat = self._normalize_category(det.category)
                class_preds[cat].append(
                    {
                        "image_id": image_id,
                        "bbox": det.bbox,
                        "confidence": float(det.confidence),
                    }
                )
            for det in sample["gt_dets"]:
                cat = self._normalize_category(det.category)
                class_gts[cat][image_id].append(det.bbox)

        all_classes = sorted(set(class_preds.keys()) | set(class_gts.keys()))
        per_class_metrics: Dict[str, EvalMetrics] = {}

        total_tp = total_fp = total_fn = 0
        total_num_pred = total_num_gt = 0
        all_matched_ious: List[float] = []
        class_aps_with_gt: List[float] = []

        for cat in all_classes:
            metrics, matched_ious, has_gt = self._evaluate_single_class(
                class_preds.get(cat, []),
                class_gts.get(cat, {}),
                iou_threshold,
            )
            per_class_metrics[cat] = metrics

            total_tp += metrics.tp
            total_fp += metrics.fp
            total_fn += metrics.fn
            total_num_pred += metrics.num_pred
            total_num_gt += metrics.num_gt
            all_matched_ious.extend(matched_ious)
            if has_gt:
                class_aps_with_gt.append(metrics.ap)

        overall_precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        overall_recall = (
            total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        )
        overall_f1 = (
            2 * overall_precision * overall_recall / (overall_precision + overall_recall)
            if (overall_precision + overall_recall) > 0
            else 0.0
        )
        overall_ap = float(np.mean(class_aps_with_gt)) if class_aps_with_gt else 0.0
        overall_iou_mean = (
            float(np.mean(all_matched_ious)) if all_matched_ious else 0.0
        )
        overall_iou_std = float(np.std(all_matched_ious)) if all_matched_ious else 0.0

        overall_metrics = EvalMetrics(
            precision=overall_precision,
            recall=overall_recall,
            f1_score=overall_f1,
            ap=overall_ap,
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            iou_mean=overall_iou_mean,
            iou_std=overall_iou_std,
            num_pred=total_num_pred,
            num_gt=total_num_gt,
        )

        return {
            "overall": overall_metrics,
            "per_class": per_class_metrics,
        }


class ModelInference:
    """模型推理"""

    def __init__(
        self,
        model_path: str,
        use_4bit: bool = True,
        bf16: bool = True,
        max_image_size: int = 512,
    ):
        self.model_path = model_path
        self.max_image_size = max_image_size

        logger.info(f"Loading model from {model_path}")

        model_class = get_model_class(model_path)

        # 量化配置
        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        # 加载模型
        self.model = model_class.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.model.eval()
        logger.info("Model loaded successfully")

    @torch.no_grad()
    def predict(self, image: Image.Image, prompt: str) -> str:
        """对单张图片进行推理"""
        # 限制图片大小
        if self.max_image_size and max(image.size) > self.max_image_size:
            ratio = self.max_image_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
            image = image.resize(new_size, resample)

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
            padding=False,
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # 评估时使用贪心解码
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response


def load_test_data(data_path: str) -> List[Dict]:
    """加载测试数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} test samples from {data_path}")
    return data


def parse_ground_truth(conversations: List[Dict]) -> Tuple[str, List[Detection], str]:
    """从 conversations 中解析 prompt、ground truth detections 和 reference text"""
    # 提取 user 和 assistant 消息
    user_msg = ""
    assistant_msg = ""

    for conv in conversations:
        role = conv.get("from", "user")
        if role in ["human", "user"]:
            role = "user"
        elif role in ["gpt", "assistant"]:
            role = "assistant"

        text = conv.get("value", "").replace("<image>\n", "").replace("<image>", "")

        if role == "user":
            user_msg = text
        elif role == "assistant":
            assistant_msg = text

    # 解析 ground truth
    parser = DetectionParser()
    gt_dets = parser.parse_box_format(assistant_msg)

    return user_msg, gt_dets, assistant_msg


def run_evaluation(
    model_path: str,
    test_data_path: str,
    output_dir: Optional[str] = None,
    iou_thresholds: List[float] = None,
    use_4bit: bool = True,
    max_samples: Optional[int] = None,
):
    """运行完整评估"""
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    inference = ModelInference(
        model_path=model_path,
        use_4bit=use_4bit,
    )

    # 加载测试数据
    test_data = load_test_data(test_data_path)
    if max_samples:
        test_data = test_data[:max_samples]

    # 先统一推理一次，再在不同 IoU 阈值上复用预测结果评估
    parser = DetectionParser()
    samples_for_eval = []

    for idx, sample in enumerate(tqdm(test_data, desc="Running inference")):
        image_path = sample["image"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            continue

        prompt, gt_dets, gt_text = parse_ground_truth(sample["conversations"])

        infer_start = time.time()
        pred_text = inference.predict(image, prompt)
        latency = (time.time() - infer_start) * 1000.0

        pred_dets = parser.parse_box_format(pred_text)

        image_id = sample.get("id") or f"{Path(image_path).stem}_{idx}"
        samples_for_eval.append(
            {
                "image_id": str(image_id),
                "pred_dets": pred_dets,
                "gt_dets": gt_dets,
                "pred_text": pred_text,
                "gt_text": gt_text,
                "latency_ms": latency,
            }
        )

    if not samples_for_eval:
        logger.error("No valid samples were evaluated.")
        return

    metrics_by_iou: Dict[float, Dict] = {}

    for iou_threshold in iou_thresholds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating with IoU threshold = {iou_threshold:.2f}")
        logger.info(f"{'='*60}")

        evaluator = DetectionEvaluator(iou_threshold=iou_threshold)
        eval_result = evaluator.evaluate_dataset(samples_for_eval, iou_threshold)
        metrics = eval_result["overall"]
        class_metrics = eval_result["per_class"]

        logger.info(f"\nOverall Metrics (IoU={iou_threshold:.2f}):")
        logger.info(f"  AP:        {metrics.ap:.4f}")
        logger.info(f"  Precision: {metrics.precision:.4f}")
        logger.info(f"  Recall:    {metrics.recall:.4f}")
        logger.info(f"  F1-Score:  {metrics.f1_score:.4f}")
        logger.info(f"  TP: {metrics.tp}, FP: {metrics.fp}, FN: {metrics.fn}")
        logger.info(f"  IoU Mean:  {metrics.iou_mean:.4f} ± {metrics.iou_std:.4f}")

        logger.info(f"\nPer-Class Metrics (IoU={iou_threshold:.2f}):")
        for cat, cat_metrics in sorted(class_metrics.items()):
            logger.info(f"  {cat}:")
            logger.info(f"    AP:        {cat_metrics.ap:.4f}")
            logger.info(f"    Precision: {cat_metrics.precision:.4f}")
            logger.info(f"    Recall:    {cat_metrics.recall:.4f}")
            logger.info(f"    F1-Score:  {cat_metrics.f1_score:.4f}")
            logger.info(
                f"    TP: {cat_metrics.tp}, FP: {cat_metrics.fp}, FN: {cat_metrics.fn}, "
                f"GT: {cat_metrics.num_gt}, Pred: {cat_metrics.num_pred}"
            )

        metrics_by_iou[iou_threshold] = eval_result

        if output_dir:
            result_file = os.path.join(output_dir, f"eval_results_iou{iou_threshold:.2f}.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "iou_threshold": iou_threshold,
                        "overall": {
                            "ap": metrics.ap,
                            "precision": metrics.precision,
                            "recall": metrics.recall,
                            "f1_score": metrics.f1_score,
                            "tp": metrics.tp,
                            "fp": metrics.fp,
                            "fn": metrics.fn,
                            "num_gt": metrics.num_gt,
                            "num_pred": metrics.num_pred,
                            "iou_mean": metrics.iou_mean,
                            "iou_std": metrics.iou_std,
                        },
                        "per_class": {
                            cat: {
                                "ap": m.ap,
                                "precision": m.precision,
                                "recall": m.recall,
                                "f1_score": m.f1_score,
                                "tp": m.tp,
                                "fp": m.fp,
                                "fn": m.fn,
                                "num_gt": m.num_gt,
                                "num_pred": m.num_pred,
                                "iou_mean": m.iou_mean,
                                "iou_std": m.iou_std,
                            }
                            for cat, m in class_metrics.items()
                        },
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            logger.info(f"Results saved to {result_file}")

    # 汇总 COCO/YOLO 风格指标
    iou_thresholds_sorted = sorted(metrics_by_iou.keys())

    def nearest_threshold(target: float) -> float:
        return min(iou_thresholds_sorted, key=lambda x: abs(x - target))

    t50 = nearest_threshold(0.5)
    t75 = nearest_threshold(0.75)
    ap50 = metrics_by_iou[t50]["overall"].ap
    ap75 = metrics_by_iou[t75]["overall"].ap
    map50_95 = float(np.mean([metrics_by_iou[t]["overall"].ap for t in iou_thresholds_sorted]))

    logger.info(f"\n{'='*60}")
    logger.info("COCO/YOLO Style Summary")
    logger.info(f"{'='*60}")
    logger.info(f"AP50 (IoU={t50:.2f}): {ap50:.4f}")
    logger.info(f"AP75 (IoU={t75:.2f}): {ap75:.4f}")
    logger.info(f"mAP50-95:           {map50_95:.4f}")

    # 计算 VLM 常规指标（与 AP 互补）
    vlm_metrics = compute_vlm_metrics(samples_for_eval)
    logger.info(f"\n{'='*60}")
    logger.info("VLM Quality Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Parse success rate:    {vlm_metrics.parse_success_rate:.4f}")
    logger.info(f"Strict format rate:    {vlm_metrics.strict_format_rate:.4f}")
    logger.info(f"No-object reject rate: {vlm_metrics.no_object_rejection_rate:.4f}")
    logger.info(f"Hallucination rate:    {vlm_metrics.hallucination_rate:.4f}")
    logger.info(f"False reject rate:     {vlm_metrics.false_rejection_rate:.4f}")
    logger.info(f"Omission rate:         {vlm_metrics.omission_rate:.4f}")
    logger.info(f"Anomaly accuracy:      {vlm_metrics.anomaly_accuracy:.4f}")
    logger.info(
        f"Anomaly P/R/F1:        "
        f"{vlm_metrics.anomaly_precision:.4f}/"
        f"{vlm_metrics.anomaly_recall:.4f}/"
        f"{vlm_metrics.anomaly_f1:.4f}"
    )
    logger.info(f"Category F1:           {vlm_metrics.category_f1:.4f}")
    logger.info(f"Exact match rate:      {vlm_metrics.exact_match_rate:.4f}")
    logger.info(f"Token F1:              {vlm_metrics.token_f1:.4f}")
    logger.info(f"ROUGE-L F1:            {vlm_metrics.rouge_l_f1:.4f}")
    logger.info(f"BLEU-1 / BLEU-4:       {vlm_metrics.bleu1:.4f} / {vlm_metrics.bleu4:.4f}")
    logger.info(
        f"Count MAE / RMSE:      "
        f"{vlm_metrics.count_mae:.4f} / {vlm_metrics.count_rmse:.4f}"
    )
    logger.info(
        f"Latency ms (mean/p50/p95): "
        f"{vlm_metrics.latency_ms_mean:.1f}/{vlm_metrics.latency_ms_p50:.1f}/{vlm_metrics.latency_ms_p95:.1f}"
    )

    if output_dir:
        summary_file = os.path.join(output_dir, "eval_summary.json")
        per_class_summary = {}
        all_classes = sorted(
            {
                cat
                for per_iou in metrics_by_iou.values()
                for cat in per_iou["per_class"].keys()
            }
        )

        for cat in all_classes:
            ap_list = []
            for iou in iou_thresholds_sorted:
                class_metric = metrics_by_iou[iou]["per_class"].get(cat)
                if class_metric and class_metric.num_gt > 0:
                    ap_list.append(class_metric.ap)

            metric50 = metrics_by_iou[t50]["per_class"].get(cat, EvalMetrics())
            metric75 = metrics_by_iou[t75]["per_class"].get(cat, EvalMetrics())
            per_class_summary[cat] = {
                "ap50": metric50.ap,
                "ap75": metric75.ap,
                "map50_95": float(np.mean(ap_list)) if ap_list else 0.0,
                "precision@50": metric50.precision,
                "recall@50": metric50.recall,
                "f1@50": metric50.f1_score,
                "num_gt": metric50.num_gt,
                "num_pred": metric50.num_pred,
            }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_path": model_path,
                    "test_data": test_data_path,
                    "num_samples": len(samples_for_eval),
                    "iou_thresholds": iou_thresholds_sorted,
                    "ap50": ap50,
                    "ap75": ap75,
                    "map50_95": map50_95,
                    "overall@50": {
                        "precision": metrics_by_iou[t50]["overall"].precision,
                        "recall": metrics_by_iou[t50]["overall"].recall,
                        "f1_score": metrics_by_iou[t50]["overall"].f1_score,
                    },
                    "results_by_iou": {
                        str(iou): {
                            "ap": metrics_by_iou[iou]["overall"].ap,
                            "precision": metrics_by_iou[iou]["overall"].precision,
                            "recall": metrics_by_iou[iou]["overall"].recall,
                            "f1_score": metrics_by_iou[iou]["overall"].f1_score,
                        }
                        for iou in iou_thresholds_sorted
                    },
                    "vlm_metrics": {
                        "parse_success_rate": vlm_metrics.parse_success_rate,
                        "strict_format_rate": vlm_metrics.strict_format_rate,
                        "no_object_rejection_rate": vlm_metrics.no_object_rejection_rate,
                        "hallucination_rate": vlm_metrics.hallucination_rate,
                        "false_rejection_rate": vlm_metrics.false_rejection_rate,
                        "omission_rate": vlm_metrics.omission_rate,
                        "anomaly_accuracy": vlm_metrics.anomaly_accuracy,
                        "anomaly_precision": vlm_metrics.anomaly_precision,
                        "anomaly_recall": vlm_metrics.anomaly_recall,
                        "anomaly_f1": vlm_metrics.anomaly_f1,
                        "category_precision": vlm_metrics.category_precision,
                        "category_recall": vlm_metrics.category_recall,
                        "category_f1": vlm_metrics.category_f1,
                        "exact_match_rate": vlm_metrics.exact_match_rate,
                        "token_f1": vlm_metrics.token_f1,
                        "rouge_l_f1": vlm_metrics.rouge_l_f1,
                        "bleu1": vlm_metrics.bleu1,
                        "bleu4": vlm_metrics.bleu4,
                        "count_mae": vlm_metrics.count_mae,
                        "count_rmse": vlm_metrics.count_rmse,
                        "avg_response_tokens": vlm_metrics.avg_response_tokens,
                        "avg_pred_boxes": vlm_metrics.avg_pred_boxes,
                        "avg_gt_boxes": vlm_metrics.avg_gt_boxes,
                        "latency_ms_mean": vlm_metrics.latency_ms_mean,
                        "latency_ms_p50": vlm_metrics.latency_ms_p50,
                        "latency_ms_p95": vlm_metrics.latency_ms_p95,
                    },
                    "per_class": per_class_summary,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate object detection model on test set"
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model (base model or fine-tuned LoRA)"
    )
    parser.add_argument(
        "--no_4bit",
        action="store_false",
        dest="use_4bit",
        default=True,
        help="Disable 4-bit quantization"
    )

    # Data arguments
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for quick testing)"
    )

    # Evaluation arguments
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="Single IoU threshold for evaluation (default: 0.5)"
    )
    parser.add_argument(
        "--iou_thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Custom IoU thresholds (e.g., 0.5 0.75 0.9)"
    )
    parser.add_argument(
        "--coco_map",
        action="store_true",
        help="Use COCO-style IoU thresholds: 0.50:0.05:0.95"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results"
    )

    args = parser.parse_args()

    # 确定 IoU 阈值
    if args.coco_map:
        iou_thresholds = COCO_IOU_THRESHOLDS
    elif args.iou_thresholds:
        iou_thresholds = args.iou_thresholds
    else:
        iou_thresholds = [args.iou_threshold]

    iou_thresholds = sorted(set(float(x) for x in iou_thresholds))

    # 运行评估
    run_evaluation(
        model_path=args.model_path,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        iou_thresholds=iou_thresholds,
        use_4bit=args.use_4bit,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
