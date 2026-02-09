#!/usr/bin/env python3
"""
可视化评估结果脚本

Usage:
    # 可视化单个评估结果
    python scripts/visualize_eval_results.py \
        --eval_dir eval_results/ \
        --output plots/eval_plots.png

    # 比较多个模型的评估结果
    python scripts/visualize_eval_results.py \
        --eval_dirs eval_results/base_model eval_results/lora_model eval_results/grpo_model \
        --labels "Base" "LoRA" "GRPO" \
        --output plots/model_comparison.png
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_eval_results(eval_dir: str) -> Dict:
    """加载评估结果"""
    results = {}

    # 加载所有 IoU 阈值的结果
    for result_file in Path(eval_dir).glob("eval_results_iou*.json"):
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            iou_threshold = data["iou_threshold"]
            results[iou_threshold] = data

    # 加载 summary（如果存在）
    summary_file = Path(eval_dir) / "eval_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            results["summary"] = json.load(f)

    return results


def plot_metrics_by_iou(
    results: Dict,
    output_path: str,
    title: str = "Detection Metrics vs IoU Threshold"
):
    """绘制不同 IoU 阈值下的指标曲线"""
    iou_thresholds = sorted([k for k in results.keys() if isinstance(k, float)])

    if not iou_thresholds:
        logger.warning("No IoU threshold results found")
        return

    precisions = []
    recalls = []
    f1_scores = []

    for iou in iou_thresholds:
        overall = results[iou]["overall"]
        precisions.append(overall["precision"])
        recalls.append(overall["recall"])
        f1_scores.append(overall["f1_score"])

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(iou_thresholds, precisions, marker='o', label='Precision', linewidth=2)
    ax.plot(iou_thresholds, recalls, marker='s', label='Recall', linewidth=2)
    ax.plot(iou_thresholds, f1_scores, marker='^', label='F1-Score', linewidth=2)

    ax.set_xlabel('IoU Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {output_path}")
    plt.close()


def plot_per_class_metrics(
    results: Dict,
    output_path: str,
    iou_threshold: float = 0.5,
    title: str = "Per-Class Detection Metrics"
):
    """绘制每个类别的指标柱状图"""
    if iou_threshold not in results:
        logger.warning(f"IoU threshold {iou_threshold} not found in results")
        return

    per_class = results[iou_threshold]["per_class"]

    if not per_class:
        logger.warning("No per-class results found")
        return

    categories = list(per_class.keys())
    precisions = [per_class[cat]["precision"] for cat in categories]
    recalls = [per_class[cat]["recall"] for cat in categories]
    f1_scores = [per_class[cat]["f1_score"] for cat in categories]

    # 创建图表
    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f"{title} (IoU={iou_threshold})", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {output_path}")
    plt.close()


def plot_confusion_stats(
    results: Dict,
    output_path: str,
    iou_threshold: float = 0.5,
    title: str = "Detection Statistics"
):
    """绘制 TP/FP/FN 统计"""
    if iou_threshold not in results:
        logger.warning(f"IoU threshold {iou_threshold} not found in results")
        return

    overall = results[iou_threshold]["overall"]

    # 总体统计
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：TP/FP/FN 柱状图
    categories = ['True Positive', 'False Positive', 'False Negative']
    values = [overall["tp"], overall["fp"], overall["fn"]]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']

    ax1.bar(categories, values, color=colors, alpha=0.8)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Detection Counts', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # 在柱子上显示数值
    for i, v in enumerate(values):
        ax1.text(i, v + max(values) * 0.02, str(v), ha='center', fontsize=11, fontweight='bold')

    # 右图：Precision/Recall/F1 雷达图
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [overall["precision"], overall["recall"], overall["f1_score"]]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]  # 闭合
    angles += angles[:1]

    ax2 = plt.subplot(122, projection='polar')
    ax2.plot(angles, values, 'o-', linewidth=2, color='#3498db')
    ax2.fill(angles, values, alpha=0.25, color='#3498db')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics, fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.set_title('Metrics Radar', fontsize=13, fontweight='bold', pad=20)
    ax2.grid(True)

    plt.suptitle(f"{title} (IoU={iou_threshold})", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {output_path}")
    plt.close()


def plot_model_comparison(
    eval_dirs: List[str],
    labels: List[str],
    output_path: str,
    iou_threshold: float = 0.5,
):
    """比较多个模型的性能"""
    all_results = []

    for eval_dir in eval_dirs:
        results = load_eval_results(eval_dir)
        all_results.append(results)

    # 提取指标
    precisions = []
    recalls = []
    f1_scores = []

    for results in all_results:
        if iou_threshold in results:
            overall = results[iou_threshold]["overall"]
            precisions.append(overall["precision"])
            recalls.append(overall["recall"])
            f1_scores.append(overall["f1_score"])
        else:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)

    # 创建图表
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Model Comparison (IoU={iou_threshold})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])

    # 在柱子上显示数值
    for i in range(len(labels)):
        ax.text(i - width, precisions[i] + 0.02, f'{precisions[i]:.3f}',
                ha='center', fontsize=9)
        ax.text(i, recalls[i] + 0.02, f'{recalls[i]:.3f}',
                ha='center', fontsize=9)
        ax.text(i + width, f1_scores[i] + 0.02, f'{f1_scores[i]:.3f}',
                ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Comparison plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation results")

    # Single model visualization
    parser.add_argument(
        "--eval_dir",
        type=str,
        help="Directory containing evaluation results"
    )

    # Multi-model comparison
    parser.add_argument(
        "--eval_dirs",
        type=str,
        nargs="+",
        help="Multiple evaluation directories for comparison"
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        help="Labels for each model in comparison"
    )

    # Common arguments
    parser.add_argument(
        "--output",
        type=str,
        default="eval_plots.png",
        help="Output plot file path"
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for visualization (default: 0.5)"
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=["metrics_by_iou", "per_class", "confusion", "comparison", "all"],
        default="all",
        help="Type of plot to generate"
    )

    args = parser.parse_args()

    # Model comparison mode
    if args.eval_dirs:
        if not args.labels or len(args.labels) != len(args.eval_dirs):
            parser.error("--labels must be provided and match the number of --eval_dirs")

        plot_model_comparison(
            eval_dirs=args.eval_dirs,
            labels=args.labels,
            output_path=args.output,
            iou_threshold=args.iou_threshold,
        )
        return

    # Single model mode
    if not args.eval_dir:
        parser.error("Either --eval_dir or --eval_dirs must be provided")

    results = load_eval_results(args.eval_dir)

    if not results:
        logger.error(f"No evaluation results found in {args.eval_dir}")
        return

    # 创建输出目录
    output_dir = Path(args.output).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # 生成图表
    if args.plot_type in ["metrics_by_iou", "all"]:
        output_path = args.output.replace(".png", "_metrics_by_iou.png")
        plot_metrics_by_iou(results, output_path)

    if args.plot_type in ["per_class", "all"]:
        output_path = args.output.replace(".png", "_per_class.png")
        plot_per_class_metrics(results, output_path, args.iou_threshold)

    if args.plot_type in ["confusion", "all"]:
        output_path = args.output.replace(".png", "_confusion.png")
        plot_confusion_stats(results, output_path, args.iou_threshold)

    logger.info("Visualization complete!")


if __name__ == "__main__":
    main()
