#!/usr/bin/env python3
"""
可视化 GRPO 训练曲线

Usage:
    # 启动 TensorBoard
    python scripts/visualize_training.py --logdir outputs/qwen3vl_grpo_trl

    # 或者导出图片
    python scripts/visualize_training.py --logdir outputs/qwen3vl_grpo_trl --export
"""

import os
import argparse
from pathlib import Path


def start_tensorboard(logdir: str, port: int = 6006):
    """启动 TensorBoard 服务"""
    print(f"启动 TensorBoard...")
    print(f"日志目录: {logdir}")
    print(f"访问地址: http://localhost:{port}")
    print("-" * 50)
    print("按 Ctrl+C 停止")

    os.system(f"tensorboard --logdir={logdir} --port={port} --bind_all")


def export_plots(logdir: str, output_dir: str = None):
    """导出训练曲线为图片"""
    try:
        import matplotlib.pyplot as plt
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("请安装依赖: pip install matplotlib tensorboard")
        return

    if output_dir is None:
        output_dir = os.path.join(logdir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    # 查找 events 文件
    events_files = list(Path(logdir).rglob("events.out.tfevents.*"))
    if not events_files:
        print(f"未找到 TensorBoard events 文件: {logdir}")
        return

    print(f"找到 {len(events_files)} 个 events 文件")

    # 加载所有 events
    all_scalars = {}

    for events_file in events_files:
        ea = event_accumulator.EventAccumulator(str(events_file.parent))
        ea.Reload()

        tags = ea.Tags().get('scalars', [])
        for tag in tags:
            events = ea.Scalars(tag)
            if tag not in all_scalars:
                all_scalars[tag] = []
            all_scalars[tag].extend([(e.step, e.value) for e in events])

    if not all_scalars:
        print("未找到任何 scalar 数据")
        return

    # 按类别分组绘图
    categories = {
        'reward': ['reward', 'reward_std'],
        'rewards_detail': [k for k in all_scalars.keys() if k.startswith('rewards/')],
        'kl_clip': ['kl', 'clip_ratio'],
        'loss': [k for k in all_scalars.keys() if 'loss' in k.lower()],
        'lr': [k for k in all_scalars.keys() if 'lr' in k.lower() or 'learning_rate' in k.lower()],
    }

    plt.style.use('seaborn-v0_8-darkgrid')

    for cat_name, tags in categories.items():
        available_tags = [t for t in tags if t in all_scalars]
        if not available_tags:
            continue

        fig, axes = plt.subplots(len(available_tags), 1, figsize=(12, 4*len(available_tags)))
        if len(available_tags) == 1:
            axes = [axes]

        for ax, tag in zip(axes, available_tags):
            data = sorted(all_scalars[tag], key=lambda x: x[0])
            steps = [d[0] for d in data]
            values = [d[1] for d in data]

            ax.plot(steps, values, linewidth=1.5)
            ax.set_xlabel('Step')
            ax.set_ylabel(tag)
            ax.set_title(tag)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{cat_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")

    # 创建综合图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Reward 曲线
    if 'reward' in all_scalars:
        data = sorted(all_scalars['reward'], key=lambda x: x[0])
        axes[0, 0].plot([d[0] for d in data], [d[1] for d in data], 'b-', linewidth=1.5)
        axes[0, 0].set_title('Reward (越高越好)')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)

    # Reward std
    if 'reward_std' in all_scalars:
        data = sorted(all_scalars['reward_std'], key=lambda x: x[0])
        axes[0, 1].plot([d[0] for d in data], [d[1] for d in data], 'g-', linewidth=1.5)
        axes[0, 1].set_title('Reward Std (组内方差)')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Std')
        axes[0, 1].grid(True, alpha=0.3)

    # KL
    if 'kl' in all_scalars:
        data = sorted(all_scalars['kl'], key=lambda x: x[0])
        axes[1, 0].plot([d[0] for d in data], [d[1] for d in data], 'r-', linewidth=1.5)
        axes[1, 0].set_title('KL Divergence (越低越稳定)')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('KL')
        axes[1, 0].grid(True, alpha=0.3)

    # Clip ratio
    clip_tags = [k for k in all_scalars.keys() if 'clip' in k.lower()]
    if clip_tags:
        for tag in clip_tags[:3]:  # 最多显示3条
            data = sorted(all_scalars[tag], key=lambda x: x[0])
            axes[1, 1].plot([d[0] for d in data], [d[1] for d in data], linewidth=1.5, label=tag)
        axes[1, 1].set_title('Clip Ratio')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('GRPO Training Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()

    summary_path = os.path.join(output_dir, "training_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存综合图: {summary_path}")

    print(f"\n所有图片已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="可视化 GRPO 训练曲线")
    parser.add_argument("--logdir", type=str, default="outputs/qwen3vl_grpo_trl",
                        help="TensorBoard 日志目录")
    parser.add_argument("--port", type=int, default=6006,
                        help="TensorBoard 端口")
    parser.add_argument("--export", action="store_true",
                        help="导出图片而不是启动 TensorBoard")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="图片输出目录 (默认为 logdir/plots)")

    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        print(f"目录不存在: {args.logdir}")
        return

    if args.export:
        export_plots(args.logdir, args.output_dir)
    else:
        start_tensorboard(args.logdir, args.port)


if __name__ == "__main__":
    main()
