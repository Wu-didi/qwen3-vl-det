#!/usr/bin/env python3
"""
可视化训练日志脚本

支持显示：
- 训练历史（loss, reward, KL divergence, learning rate）
- 验证历史（validation metrics）
- 可用性自检历史（usability metrics）

Usage:
    # 查看训练摘要
    python scripts/visualize_training_log.py --log outputs/qwen3vl_grpo/training_log.json

    # 生成可视化图表
    python scripts/visualize_training_log.py \
        --log outputs/qwen3vl_grpo/training_log.json \
        --output plots/

    # 导出 CSV
    python scripts/visualize_training_log.py \
        --log outputs/qwen3vl_grpo/training_log.json \
        --export-csv
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import pandas as pd


def load_training_log(log_path: str) -> Dict[str, Any]:
    """加载训练日志"""
    with open(log_path, 'r') as f:
        return json.load(f)


def print_summary(log: Dict[str, Any]):
    """打印训练摘要"""
    print("=" * 80)
    print("Training Log Summary")
    print("=" * 80)

    # 配置信息
    config = log.get("config", {})
    print("\n[Configuration]")
    print(f"  Model: {config.get('model_path', 'N/A')}")
    print(f"  Output: {config.get('output_dir', 'N/A')}")
    print(f"  Epochs: {config.get('num_epochs', 'N/A')}")
    print(f"  Batch size: {config.get('batch_size', 'N/A')}")
    print(f"  Gradient accumulation: {config.get('gradient_accumulation_steps', 'N/A')}")
    print(f"  Learning rate: {config.get('learning_rate', 'N/A')}")
    print(f"  LoRA r: {config.get('lora_r', 'N/A')}")
    print(f"  Num generations: {config.get('num_generations', 'N/A')}")
    print(f"  KL coefficient: {config.get('kl_coef', 'N/A')}")

    # 训练历史
    train_history = log.get("train_history", [])
    if train_history:
        print("\n[Training History]")
        print(f"  Total steps: {len(train_history)}")
        last_entry = train_history[-1]
        print(f"  Final step: {last_entry.get('step', 'N/A')}")
        print(f"  Final loss: {last_entry.get('loss', 'N/A'):.4f}")
        print(f"  Final reward: {last_entry.get('reward', 'N/A'):.4f}")
        print(f"  Final KL: {last_entry.get('kl', 'N/A'):.4f}")
        print(f"  Final LR: {last_entry.get('lr', 'N/A'):.2e}")

    # 验证历史
    val_history = log.get("val_history", [])
    if val_history:
        print("\n[Validation History]")
        print(f"  Total validations: {len(val_history)}")
        last_val = val_history[-1]
        print(f"  Final val_reward: {last_val.get('val_reward', 'N/A'):.4f}")
        print(f"  Final val_format: {last_val.get('val_format', 'N/A'):.4f}")
        print(f"  Final val_bbox: {last_val.get('val_bbox', 'N/A'):.4f}")
        print(f"  Final val_category: {last_val.get('val_category', 'N/A'):.4f}")

    # 可用性自检历史
    usability_history = log.get("usability_history", [])
    if usability_history:
        print("\n[Usability Check History]")
        print(f"  Total checks: {len(usability_history)}")
        last_check = usability_history[-1]
        print(f"  Parse success rate: {last_check.get('usability_parse_success_rate', 'N/A'):.1%}")
        print(f"  Has box rate: {last_check.get('usability_has_box_rate', 'N/A'):.1%}")
        print(f"  Has numbered rate: {last_check.get('usability_has_numbered_rate', 'N/A'):.1%}")
        print(f"  Has status rate: {last_check.get('usability_has_status_rate', 'N/A'):.1%}")
        print(f"  Avg IoU: {last_check.get('usability_avg_iou', 'N/A'):.3f}")
        print(f"  Category accuracy: {last_check.get('usability_category_accuracy', 'N/A'):.1%}")
        print(f"  Anomaly accuracy: {last_check.get('usability_anomaly_accuracy', 'N/A'):.1%}")

    # 最佳检查点
    best_checkpoint = log.get("best_checkpoint")
    if best_checkpoint:
        print("\n[Best Checkpoint]")
        print(f"  Step: {best_checkpoint.get('step', 'N/A')}")
        print(f"  Epoch: {best_checkpoint.get('epoch', 'N/A')}")
        print(f"  Val reward: {best_checkpoint.get('val_reward', 'N/A'):.4f}")
        print(f"  Path: {best_checkpoint.get('path', 'N/A')}")

    print("=" * 80)


def plot_training_curves(log: Dict[str, Any], output_dir: str):
    """生成训练曲线图"""
    os.makedirs(output_dir, exist_ok=True)

    train_history = log.get("train_history", [])
    val_history = log.get("val_history", [])
    usability_history = log.get("usability_history", [])

    if not train_history:
        print("No training history to plot")
        return

    # 转换为 DataFrame
    train_df = pd.DataFrame(train_history)
    val_df = pd.DataFrame(val_history) if val_history else None
    usability_df = pd.DataFrame(usability_history) if usability_history else None

    # 1. 训练指标（loss, reward, KL）
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)

    # Loss
    axes[0, 0].plot(train_df['step'], train_df['loss'], label='Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Reward
    axes[0, 1].plot(train_df['step'], train_df['reward'], label='Reward', alpha=0.7, color='green')
    if val_df is not None and 'val_reward' in val_df.columns:
        axes[0, 1].plot(val_df['step'], val_df['val_reward'], label='Val Reward', alpha=0.7, color='orange', marker='o')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].set_title('Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # KL Divergence
    if 'kl' in train_df.columns:
        axes[1, 0].plot(train_df['step'], train_df['kl'], label='KL Divergence', alpha=0.7, color='red')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('KL Divergence')
        axes[1, 0].set_title('KL Divergence')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Learning Rate
    if 'lr' in train_df.columns:
        axes[1, 1].plot(train_df['step'], train_df['lr'], label='Learning Rate', alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/training_metrics.png")
    plt.close()

    # 2. 验证指标
    if val_df is not None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Validation Metrics', fontsize=16)

        metrics = ['val_reward', 'val_format', 'val_bbox', 'val_category']
        titles = ['Validation Reward', 'Format Score', 'BBox Score', 'Category Score']
        colors = ['green', 'blue', 'orange', 'red']

        for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
            if metric in val_df.columns:
                ax = axes[idx // 2, idx % 2]
                ax.plot(val_df['step'], val_df[metric], label=title, alpha=0.7, color=color, marker='o')
                ax.set_xlabel('Step')
                ax.set_ylabel(metric)
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'validation_metrics.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/validation_metrics.png")
        plt.close()

    # 3. 可用性自检指标
    if usability_df is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Usability Check Metrics', fontsize=16)

        metrics = [
            ('usability_parse_success_rate', 'Parse Success Rate', 'green'),
            ('usability_has_box_rate', 'Has Box Rate', 'blue'),
            ('usability_has_numbered_rate', 'Has Numbered Rate', 'orange'),
            ('usability_has_status_rate', 'Has Status Rate', 'red'),
            ('usability_avg_iou', 'Average IoU', 'purple'),
            ('usability_category_accuracy', 'Category Accuracy', 'brown'),
        ]

        for idx, (metric, title, color) in enumerate(metrics):
            if metric in usability_df.columns:
                ax = axes[idx // 3, idx % 3]
                ax.plot(usability_df['step'], usability_df[metric], label=title, alpha=0.7, color=color, marker='o')
                ax.set_xlabel('Step')
                ax.set_ylabel(metric)
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'usability_metrics.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/usability_metrics.png")
        plt.close()

    # 4. 综合对比图（Reward vs Usability）
    if val_df is not None and usability_df is not None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.suptitle('Reward vs Usability Metrics', fontsize=16)

        # 双 Y 轴
        ax1 = ax
        ax2 = ax1.twinx()

        # Reward (左轴)
        if 'val_reward' in val_df.columns:
            ax1.plot(val_df['step'], val_df['val_reward'], label='Val Reward', alpha=0.7, color='green', marker='o', linewidth=2)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Reward', color='green')
        ax1.tick_params(axis='y', labelcolor='green')
        ax1.grid(True, alpha=0.3)

        # Usability metrics (右轴)
        if 'usability_parse_success_rate' in usability_df.columns:
            ax2.plot(usability_df['step'], usability_df['usability_parse_success_rate'], label='Parse Success', alpha=0.7, color='blue', marker='s', linewidth=2)
        if 'usability_avg_iou' in usability_df.columns:
            ax2.plot(usability_df['step'], usability_df['usability_avg_iou'], label='Avg IoU', alpha=0.7, color='orange', marker='^', linewidth=2)
        if 'usability_category_accuracy' in usability_df.columns:
            ax2.plot(usability_df['step'], usability_df['usability_category_accuracy'], label='Category Acc', alpha=0.7, color='red', marker='d', linewidth=2)
        ax2.set_ylabel('Usability Metrics', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reward_vs_usability.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/reward_vs_usability.png")
        plt.close()


def export_csv(log: Dict[str, Any], output_dir: str):
    """导出 CSV 文件"""
    os.makedirs(output_dir, exist_ok=True)

    train_history = log.get("train_history", [])
    val_history = log.get("val_history", [])
    usability_history = log.get("usability_history", [])

    if train_history:
        train_df = pd.DataFrame(train_history)
        train_csv = os.path.join(output_dir, 'train_history.csv')
        train_df.to_csv(train_csv, index=False)
        print(f"Exported: {train_csv}")

    if val_history:
        val_df = pd.DataFrame(val_history)
        val_csv = os.path.join(output_dir, 'val_history.csv')
        val_df.to_csv(val_csv, index=False)
        print(f"Exported: {val_csv}")

    if usability_history:
        usability_df = pd.DataFrame(usability_history)
        usability_csv = os.path.join(output_dir, 'usability_history.csv')
        usability_df.to_csv(usability_csv, index=False)
        print(f"Exported: {usability_csv}")


def main():
    parser = argparse.ArgumentParser(description="Visualize training logs")
    parser.add_argument("--log", type=str, required=True, help="Path to training_log.json")
    parser.add_argument("--output", type=str, default="", help="Output directory for plots (optional)")
    parser.add_argument("--export-csv", action="store_true", help="Export history to CSV files")

    args = parser.parse_args()

    # 加载日志
    log = load_training_log(args.log)

    # 打印摘要
    print_summary(log)

    # 生成图表
    if args.output:
        print(f"\nGenerating plots to {args.output}...")
        plot_training_curves(log, args.output)

    # 导出 CSV
    if args.export_csv:
        csv_dir = args.output if args.output else os.path.dirname(args.log)
        print(f"\nExporting CSV files to {csv_dir}...")
        export_csv(log, csv_dir)


if __name__ == "__main__":
    main()
