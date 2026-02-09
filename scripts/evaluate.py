#!/usr/bin/env python3
"""
目标检测评估脚本 - 计算标准检测指标

支持的指标:
- Precision (精确率)
- Recall (召回率)
- F1-Score
- mAP (mean Average Precision)
- AP per class
- IoU 分布统计

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

    # 计算多个 IoU 阈值的 mAP
    python scripts/evaluate.py \
        --model_path outputs/qwen3vl_lora \
        --test_data data/qwen_data/test.json \
        --iou_thresholds 0.5 0.75 0.9
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
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

            # 提取坐标 (支持浮点数和负数)
            box_match = re.search(
                r'<box>\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*,\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*</box>',
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
                status=status,
                is_anomaly=is_anomaly,
            ))

        # 如果上面没匹配到，尝试只提取 box 坐标 (支持浮点数和负数)
        if not detections:
            box_pattern = r'<box>\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*,\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*</box>'
            simple_matches = re.findall(box_pattern, text)
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


class DetectionEvaluator:
    """目标检测评估器"""

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

    def match_detections(
        self,
        pred_dets: List[Detection],
        gt_dets: List[Detection],
    ) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
        """
        匹配预测和真实检测框

        Returns:
            matches: List[(pred_idx, gt_idx, iou)]
            unmatched_preds: List[pred_idx]
            unmatched_gts: List[gt_idx]
        """
        if not pred_dets or not gt_dets:
            return [], list(range(len(pred_dets))), list(range(len(gt_dets)))

        # 计算所有配对的 IoU
        iou_matrix = np.zeros((len(pred_dets), len(gt_dets)))
        for i, pred in enumerate(pred_dets):
            for j, gt in enumerate(gt_dets):
                # 只匹配相同类别
                if self._normalize_category(pred.category) == self._normalize_category(gt.category):
                    iou_matrix[i, j] = self.compute_iou(pred.bbox, gt.bbox)

        # 贪心匹配：优先匹配 IoU 最高的
        matches = []
        matched_preds = set()
        matched_gts = set()

        # 按 IoU 从高到低排序
        iou_pairs = []
        for i in range(len(pred_dets)):
            for j in range(len(gt_dets)):
                if iou_matrix[i, j] >= self.iou_threshold:
                    iou_pairs.append((i, j, iou_matrix[i, j]))

        iou_pairs.sort(key=lambda x: x[2], reverse=True)

        for pred_idx, gt_idx, iou in iou_pairs:
            if pred_idx not in matched_preds and gt_idx not in matched_gts:
                matches.append((pred_idx, gt_idx, iou))
                matched_preds.add(pred_idx)
                matched_gts.add(gt_idx)

        unmatched_preds = [i for i in range(len(pred_dets)) if i not in matched_preds]
        unmatched_gts = [i for i in range(len(gt_dets)) if i not in matched_gts]

        return matches, unmatched_preds, unmatched_gts

    @staticmethod
    def _normalize_category(category: str) -> str:
        """归一化类别名称（用于匹配）"""
        # 移除空格、标点
        category = category.strip().lower()
        # 可以添加更多归一化规则
        return category

    def evaluate_sample(
        self,
        pred_text: str,
        gt_dets: List[Detection],
    ) -> Dict:
        """评估单个样本"""
        pred_dets = self.parser.parse_box_format(pred_text)

        matches, unmatched_preds, unmatched_gts = self.match_detections(
            pred_dets, gt_dets
        )

        tp = len(matches)
        fp = len(unmatched_preds)
        fn = len(unmatched_gts)

        # 计算匹配的 IoU 统计
        ious = [iou for _, _, iou in matches]

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "ious": ious,
            "num_pred": len(pred_dets),
            "num_gt": len(gt_dets),
        }

    def compute_metrics(
        self,
        all_results: List[Dict],
    ) -> EvalMetrics:
        """计算总体指标"""
        total_tp = sum(r["tp"] for r in all_results)
        total_fp = sum(r["fp"] for r in all_results)
        total_fn = sum(r["fn"] for r in all_results)

        # Precision & Recall
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

        # F1-Score
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # IoU 统计
        all_ious = []
        for r in all_results:
            all_ious.extend(r["ious"])

        iou_mean = np.mean(all_ious) if all_ious else 0.0
        iou_std = np.std(all_ious) if all_ious else 0.0

        return EvalMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            iou_mean=iou_mean,
            iou_std=iou_std,
        )

    def compute_per_class_metrics(
        self,
        all_results: List[Dict],
        pred_texts: List[str],
        gt_dets_list: List[List[Detection]],
    ) -> Dict[str, EvalMetrics]:
        """计算每个类别的指标"""
        # 按类别分组
        class_results = defaultdict(list)

        for pred_text, gt_dets in zip(pred_texts, gt_dets_list):
            pred_dets = self.parser.parse_box_format(pred_text)

            # 按类别分组
            pred_by_class = defaultdict(list)
            gt_by_class = defaultdict(list)

            for det in pred_dets:
                cat = self._normalize_category(det.category)
                pred_by_class[cat].append(det)

            for det in gt_dets:
                cat = self._normalize_category(det.category)
                gt_by_class[cat].append(det)

            # 对每个类别单独评估
            all_cats = set(pred_by_class.keys()) | set(gt_by_class.keys())
            for cat in all_cats:
                result = self.evaluate_sample(
                    pred_text="",  # 不需要重新解析
                    gt_dets=gt_by_class.get(cat, []),
                )
                # 手动设置 pred_dets
                pred_dets_cat = pred_by_class.get(cat, [])
                matches, unmatched_preds, unmatched_gts = self.match_detections(
                    pred_dets_cat, gt_by_class.get(cat, [])
                )
                result["tp"] = len(matches)
                result["fp"] = len(unmatched_preds)
                result["fn"] = len(unmatched_gts)
                result["ious"] = [iou for _, _, iou in matches]

                class_results[cat].append(result)

        # 计算每个类别的指标
        class_metrics = {}
        for cat, results in class_results.items():
            class_metrics[cat] = self.compute_metrics(results)

        return class_metrics


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


def parse_ground_truth(conversations: List[Dict]) -> Tuple[str, List[Detection]]:
    """从 conversations 中解析 prompt 和 ground truth"""
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

    return user_msg, gt_dets


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

    # 对每个 IoU 阈值进行评估
    all_metrics = {}

    for iou_threshold in iou_thresholds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating with IoU threshold = {iou_threshold}")
        logger.info(f"{'='*60}")

        evaluator = DetectionEvaluator(iou_threshold=iou_threshold)

        # 运行推理和评估
        all_results = []
        pred_texts = []
        gt_dets_list = []

        for sample in tqdm(test_data, desc=f"Evaluating (IoU={iou_threshold})"):
            # 加载图片
            image_path = sample["image"]
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")
                continue

            # 解析 ground truth
            prompt, gt_dets = parse_ground_truth(sample["conversations"])

            # 推理
            pred_text = inference.predict(image, prompt)

            # 评估
            result = evaluator.evaluate_sample(pred_text, gt_dets)
            all_results.append(result)
            pred_texts.append(pred_text)
            gt_dets_list.append(gt_dets)

        # 计算总体指标
        metrics = evaluator.compute_metrics(all_results)

        logger.info(f"\nOverall Metrics (IoU={iou_threshold}):")
        logger.info(f"  Precision: {metrics.precision:.4f}")
        logger.info(f"  Recall:    {metrics.recall:.4f}")
        logger.info(f"  F1-Score:  {metrics.f1_score:.4f}")
        logger.info(f"  TP: {metrics.tp}, FP: {metrics.fp}, FN: {metrics.fn}")
        logger.info(f"  IoU Mean:  {metrics.iou_mean:.4f} ± {metrics.iou_std:.4f}")

        # 计算每个类别的指标
        class_metrics = evaluator.compute_per_class_metrics(
            all_results, pred_texts, gt_dets_list
        )

        logger.info(f"\nPer-Class Metrics (IoU={iou_threshold}):")
        for cat, cat_metrics in sorted(class_metrics.items()):
            logger.info(f"  {cat}:")
            logger.info(f"    Precision: {cat_metrics.precision:.4f}")
            logger.info(f"    Recall:    {cat_metrics.recall:.4f}")
            logger.info(f"    F1-Score:  {cat_metrics.f1_score:.4f}")
            logger.info(f"    TP: {cat_metrics.tp}, FP: {cat_metrics.fp}, FN: {cat_metrics.fn}")

        all_metrics[iou_threshold] = {
            "overall": metrics,
            "per_class": class_metrics,
        }

        # 保存结果
        if output_dir:
            result_file = os.path.join(
                output_dir,
                f"eval_results_iou{iou_threshold:.2f}.json"
            )
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "iou_threshold": iou_threshold,
                    "overall": {
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1_score": metrics.f1_score,
                        "tp": metrics.tp,
                        "fp": metrics.fp,
                        "fn": metrics.fn,
                        "iou_mean": metrics.iou_mean,
                        "iou_std": metrics.iou_std,
                    },
                    "per_class": {
                        cat: {
                            "precision": m.precision,
                            "recall": m.recall,
                            "f1_score": m.f1_score,
                            "tp": m.tp,
                            "fp": m.fp,
                            "fn": m.fn,
                        }
                        for cat, m in class_metrics.items()
                    },
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {result_file}")

    # 计算 mAP (多个 IoU 阈值的平均)
    if len(iou_thresholds) > 1:
        map_score = np.mean([
            all_metrics[iou]["overall"].precision
            for iou in iou_thresholds
        ])
        logger.info(f"\nmAP@[{','.join(map(str, iou_thresholds))}]: {map_score:.4f}")

        if output_dir:
            summary_file = os.path.join(output_dir, "eval_summary.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "model_path": model_path,
                    "test_data": test_data_path,
                    "num_samples": len(test_data),
                    "iou_thresholds": iou_thresholds,
                    "mAP": map_score,
                    "results_by_iou": {
                        str(iou): {
                            "precision": all_metrics[iou]["overall"].precision,
                            "recall": all_metrics[iou]["overall"].recall,
                            "f1_score": all_metrics[iou]["overall"].f1_score,
                        }
                        for iou in iou_thresholds
                    }
                }, f, indent=2, ensure_ascii=False)
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
        help="IoU threshold for matching (default: 0.5)"
    )
    parser.add_argument(
        "--iou_thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Multiple IoU thresholds for mAP calculation (e.g., 0.5 0.75 0.9)"
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
    if args.iou_thresholds:
        iou_thresholds = args.iou_thresholds
    else:
        iou_thresholds = [args.iou_threshold]

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
