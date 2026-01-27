#!/usr/bin/env python3
"""
训练日志可视化脚本

读取 training_log.json 并生成训练曲线图

Usage:
    python scripts/visualize_training_log.py --log outputs/qwen3vl_grpo/training_log.json
    python scripts/visualize_training_log.py --log outputs/qwen3vl_grpo/training_log.json --output plots/
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


def load_training_log(log_path: str) -> Dict:
    """加载训练日志"""
    with open(log_path, 'r') as f:
        return json.load(f)


def print_summary(log: Dict):
    """打印训练摘要"""
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)

    # 配置信息
    if 'config' in log:
        config = log['config']
        print("\nConfiguration:")
        print(f"  Model: {config.get('model_path', 'N/A')}")
        print(f"  Output: {config.get('output_dir', 'N/A')}")
        print(f"  Epochs: {config.get('num_epochs', 'N/A')}")
        print(f"  Batch Size: {config.get('batch_size', 'N/A')}")
        print(f"  Learning Rate: {config.get('learning_rate', 'N/A')}")
        print(f"  Gradient Accumulation: {config.get('gradient_accumulation_steps', 'N/A')}")

    # 训练历史
    if 'train_history' in log and log['train_history']:
        train_history = log['train_history']
        print(f"\nTraining History: {len(train_history)} entries")

        # 最后一条训练记录
        last_train = train_history[-1]
        print(f"  Final Step: {last_train.get('step', 'N/A')}")
        print(f"  Final Epoch: {last_train.get('epoch', 'N/A')}")

        # 打印所有可用的指标
        metrics = {k: v for k, v in last_train.items() if k not in ['step', 'epoch', 'samples']}
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  Final {key}: {value:.4f}")
            else:
                print(f"  Final {key}: {value}")

    # 验证历史
    if 'val_history' in log and log['val_history']:
        val_history = log['val_history']
        print(f"\nValidation History: {len(val_history)} entries")

        # 最后一条验证记录
        last_val = val_history[-1]
        print(f"  Final Val Step: {last_val.get('step', 'N/A')}")

        # 打印所有验证指标
        val_metrics = {k: v for k, v in last_val.items() if k not in ['step', 'epoch', 'samples']}
        for key, value in val_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    # 最佳检查点
    if 'best_checkpoint' in log and log['best_checkpoint']:
        best = log['best_checkpoint']
        print("\nBest Checkpoint:")
        print(f"  Step: {best.get('step', 'N/A')}")
        print(f"  Epoch: {best.get('epoch', 'N/A')}")
        for key, value in best.items():
            if key not in ['step', 'epoch', 'path', 'samples']:
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        if 'path' in best:
            print(f"  Path: {best['path']}")

    print("="*60 + "\n")


def plot_training_curves(log: Dict, output_dir: str = None):
    """绘制训练曲线"""
    if not HAS_MATPLOTLIB:
        print("Skipping plots: matplotlib not installed")
        return

    train_history = log.get('train_history', [])
    val_history = log.get('val_history', [])

    if not train_history:
        print("No training history to plot")
        return

    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 提取训练指标
    train_steps = [entry['step'] for entry in train_history]

    # 确定有哪些指标
    available_metrics = set()
    for entry in train_history:
        available_metrics.update(entry.keys())
    available_metrics.discard('step')
    available_metrics.discard('epoch')
    available_metrics.discard('samples')

    # 为每个指标绘制曲线
    for metric in available_metrics:
        train_values = [entry.get(metric) for entry in train_history if metric in entry]

        if not train_values or all(v is None for v in train_values):
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(train_steps[:len(train_values)], train_values, label=f'Train {metric}', linewidth=2)

        # 如果有验证数据，也绘制
        if val_history:
            val_metric_name = f'val_{metric}' if not metric.startswith('val_') else metric
            val_steps = []
            val_values = []
            for entry in val_history:
                if val_metric_name in entry:
                    val_steps.append(entry['step'])
                    val_values.append(entry[val_metric_name])

            if val_values:
                plt.plot(val_steps, val_values, label=f'Val {metric}',
                        linewidth=2, linestyle='--', marker='o', markersize=6)

        plt.xlabel('Step', fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.title(f'Training Curve: {metric.replace("_", " ").title()}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if output_dir:
            output_path = os.path.join(output_dir, f'{metric}_curve.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot: {output_path}")
        else:
            plt.show()

        plt.close()


def export_csv(log: Dict, output_dir: str):
    """导出训练日志为 CSV 格式"""
    import csv

    os.makedirs(output_dir, exist_ok=True)

    # 导出训练历史
    if 'train_history' in log and log['train_history']:
        train_csv_path = os.path.join(output_dir, 'train_history.csv')
        with open(train_csv_path, 'w', newline='') as f:
            if log['train_history']:
                writer = csv.DictWriter(f, fieldnames=log['train_history'][0].keys())
                writer.writeheader()
                writer.writerows(log['train_history'])
        print(f"Exported training history to: {train_csv_path}")

    # 导出验证历史
    if 'val_history' in log and log['val_history']:
        val_csv_path = os.path.join(output_dir, 'val_history.csv')
        with open(val_csv_path, 'w', newline='') as f:
            if log['val_history']:
                writer = csv.DictWriter(f, fieldnames=log['val_history'][0].keys())
                writer.writeheader()
                writer.writerows(log['val_history'])
        print(f"Exported validation history to: {val_csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize training logs")
    parser.add_argument("--log", type=str, required=True,
                        help="Path to training_log.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for plots (default: show plots)")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export logs to CSV format")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting (only print summary)")

    args = parser.parse_args()

    # 加载日志
    print(f"Loading training log from: {args.log}")
    log = load_training_log(args.log)

    # 打印摘要
    print_summary(log)

    # 绘制曲线
    if not args.no_plot:
        plot_training_curves(log, args.output)

    # 导出 CSV
    if args.export_csv:
        output_dir = args.output or os.path.dirname(args.log)
        export_csv(log, output_dir)


if __name__ == "__main__":
    main()
