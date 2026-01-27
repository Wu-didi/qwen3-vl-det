#!/usr/bin/env python3
"""
对比多个训练实验的日志

Usage:
    python scripts/compare_training_logs.py \
        --logs outputs/exp1/training_log.json outputs/exp2/training_log.json \
        --output comparison.html
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


def load_logs(log_paths: List[str]) -> Dict[str, Dict]:
    """加载多个训练日志"""
    logs = {}
    for path in log_paths:
        name = Path(path).parent.name
        with open(path, 'r') as f:
            logs[name] = json.load(f)
    return logs


def print_comparison_table(logs: Dict[str, Dict]):
    """打印对比表格"""
    print("\n" + "="*80)
    print("Training Comparison")
    print("="*80)

    # 表头
    exp_names = list(logs.keys())
    print(f"\n{'Metric':<30} " + " ".join(f"{name:>15}" for name in exp_names))
    print("-"*80)

    # 配置对比
    print("\nConfiguration:")
    config_keys = ['model_path', 'num_epochs', 'batch_size', 'learning_rate',
                   'num_generations', 'temperature', 'kl_coef', 'beta']

    for key in config_keys:
        values = []
        for name in exp_names:
            config = logs[name].get('config', {})
            value = config.get(key, 'N/A')
            if isinstance(value, float):
                values.append(f"{value:.2e}")
            else:
                values.append(str(value)[:15])

        if any(v != 'N/A' for v in values):
            print(f"  {key:<28} " + " ".join(f"{v:>15}" for v in values))

    # 最终指标对比
    print("\nFinal Training Metrics:")
    for name in exp_names:
        train_history = logs[name].get('train_history', [])
        if train_history:
            last = train_history[-1]
            print(f"\n  {name}:")
            for key, value in last.items():
                if key not in ['step', 'epoch', 'samples']:
                    if isinstance(value, float):
                        print(f"    {key:<26} {value:>15.4f}")

    # 验证指标对比
    print("\nBest Validation Metrics:")
    for name in exp_names:
        val_history = logs[name].get('val_history', [])
        if val_history:
            # 找到最佳验证结果
            best = val_history[-1]  # 简化：使用最后一个
            print(f"\n  {name}:")
            for key, value in best.items():
                if key not in ['step', 'epoch', 'samples']:
                    if isinstance(value, float):
                        print(f"    {key:<26} {value:>15.4f}")

    # 最佳检查点对比
    print("\nBest Checkpoints:")
    for name in exp_names:
        best = logs[name].get('best_checkpoint')
        if best:
            print(f"\n  {name}:")
            print(f"    Step: {best.get('step', 'N/A')}")
            print(f"    Epoch: {best.get('epoch', 'N/A')}")
            for key, value in best.items():
                if key not in ['step', 'epoch', 'path', 'samples']:
                    if isinstance(value, float):
                        print(f"    {key:<26} {value:>15.4f}")

    print("\n" + "="*80 + "\n")


def plot_comparison(logs: Dict[str, Dict], output_dir: str):
    """绘制对比曲线"""
    if not HAS_MATPLOTLIB:
        print("Skipping plots: matplotlib not installed")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 确定所有实验共有的指标
    all_metrics = set()
    for log in logs.values():
        train_history = log.get('train_history', [])
        if train_history:
            for entry in train_history:
                all_metrics.update(entry.keys())

    all_metrics.discard('step')
    all_metrics.discard('epoch')
    all_metrics.discard('samples')

    # 为每个指标绘制对比曲线
    for metric in all_metrics:
        plt.figure(figsize=(12, 6))

        has_data = False
        for name, log in logs.items():
            train_history = log.get('train_history', [])
            steps = []
            values = []

            for entry in train_history:
                if metric in entry and entry[metric] is not None:
                    steps.append(entry['step'])
                    values.append(entry[metric])

            if values:
                plt.plot(steps, values, label=name, linewidth=2, marker='o',
                        markersize=3, markevery=max(1, len(steps)//20))
                has_data = True

        if not has_data:
            plt.close()
            continue

        plt.xlabel('Step', fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.title(f'Comparison: {metric.replace("_", " ").title()}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f'comparison_{metric}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot: {output_path}")
        plt.close()

    # 绘制验证指标对比
    val_metrics = set()
    for log in logs.values():
        val_history = log.get('val_history', [])
        if val_history:
            for entry in val_history:
                val_metrics.update(entry.keys())

    val_metrics.discard('step')
    val_metrics.discard('epoch')
    val_metrics.discard('samples')

    for metric in val_metrics:
        plt.figure(figsize=(12, 6))

        has_data = False
        for name, log in logs.items():
            val_history = log.get('val_history', [])
            steps = []
            values = []

            for entry in val_history:
                if metric in entry and entry[metric] is not None:
                    steps.append(entry['step'])
                    values.append(entry[metric])

            if values:
                plt.plot(steps, values, label=name, linewidth=2, marker='o', markersize=6)
                has_data = True

        if not has_data:
            plt.close()
            continue

        plt.xlabel('Step', fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.title(f'Validation Comparison: {metric.replace("_", " ").title()}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f'comparison_{metric}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved validation comparison plot: {output_path}")
        plt.close()


def generate_html_report(logs: Dict[str, Dict], output_path: str, plot_dir: str):
    """生成 HTML 对比报告"""
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Training Comparison Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 8px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .metric-name {
            font-weight: bold;
            color: #333;
        }
        .best-value {
            background-color: #c8e6c9;
            font-weight: bold;
        }
        .plot {
            margin: 20px 0;
            text-align: center;
        }
        .plot img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .experiment-card {
            background-color: #f9f9f9;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            border-left: 4px solid #4CAF50;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Training Comparison Report</h1>
        <p>Generated on: """ + str(Path(output_path).stat().st_mtime if Path(output_path).exists() else "N/A") + """</p>
"""

    # 配置对比表
    html += "<h2>Configuration Comparison</h2>\n<table>\n<tr><th>Parameter</th>"
    exp_names = list(logs.keys())
    for name in exp_names:
        html += f"<th>{name}</th>"
    html += "</tr>\n"

    config_keys = ['model_path', 'num_epochs', 'batch_size', 'learning_rate',
                   'num_generations', 'temperature', 'kl_coef', 'beta']

    for key in config_keys:
        values = []
        for name in exp_names:
            config = logs[name].get('config', {})
            value = config.get(key, 'N/A')
            values.append(value)

        if any(v != 'N/A' for v in values):
            html += f"<tr><td class='metric-name'>{key}</td>"
            for value in values:
                if isinstance(value, float):
                    html += f"<td>{value:.4f}</td>"
                else:
                    html += f"<td>{value}</td>"
            html += "</tr>\n"

    html += "</table>\n"

    # 最终指标对比
    html += "<h2>Final Training Metrics</h2>\n"
    for name in exp_names:
        train_history = logs[name].get('train_history', [])
        if train_history:
            last = train_history[-1]
            html += f"<div class='experiment-card'><h3>{name}</h3><table>\n"
            for key, value in last.items():
                if key not in ['step', 'epoch', 'samples']:
                    if isinstance(value, float):
                        html += f"<tr><td class='metric-name'>{key}</td><td>{value:.4f}</td></tr>\n"
            html += "</table></div>\n"

    # 绘图
    if plot_dir and os.path.exists(plot_dir):
        html += "<h2>Training Curves</h2>\n"
        for img_file in sorted(Path(plot_dir).glob("comparison_*.png")):
            rel_path = os.path.relpath(img_file, os.path.dirname(output_path))
            html += f"<div class='plot'><img src='{rel_path}' alt='{img_file.stem}'></div>\n"

    html += """
    </div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"\nHTML report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare multiple training experiments")
    parser.add_argument("--logs", nargs='+', required=True,
                        help="Paths to training_log.json files")
    parser.add_argument("--output", type=str, default="comparison",
                        help="Output directory or HTML file path")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting")
    parser.add_argument("--html", action="store_true",
                        help="Generate HTML report")

    args = parser.parse_args()

    # 加载日志
    print(f"Loading {len(args.logs)} training logs...")
    logs = load_logs(args.logs)

    # 打印对比表格
    print_comparison_table(logs)

    # 确定输出路径
    if args.output.endswith('.html'):
        output_html = args.output
        output_dir = os.path.dirname(output_html) or '.'
        plot_dir = os.path.join(output_dir, 'plots')
    else:
        output_dir = args.output
        plot_dir = os.path.join(output_dir, 'plots')
        output_html = os.path.join(output_dir, 'comparison.html')

    # 绘制对比曲线
    if not args.no_plot:
        os.makedirs(plot_dir, exist_ok=True)
        plot_comparison(logs, plot_dir)

    # 生成 HTML 报告
    if args.html:
        os.makedirs(output_dir, exist_ok=True)
        generate_html_report(logs, output_html, plot_dir if not args.no_plot else None)


if __name__ == "__main__":
    main()
