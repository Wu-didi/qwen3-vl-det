#!/usr/bin/env python3
"""
GRPO 训练速度监控工具

实时监控训练速度，估算剩余时间，提供优化建议

Usage:
    python scripts/monitor_grpo_speed.py --log outputs/qwen3vl_grpo_trl/training_log.json
    python scripts/monitor_grpo_speed.py --log outputs/qwen3vl_grpo_trl/training_log.json --watch
"""

import os
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path


def load_log(log_path):
    """加载训练日志"""
    try:
        with open(log_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def analyze_speed(log):
    """分析训练速度"""
    if not log:
        return None

    train_history = log.get('train_history', [])
    if len(train_history) < 2:
        return None

    # 获取配置
    config = log.get('config', {})
    gradient_accumulation = config.get('gradient_accumulation_steps', 4)
    batch_size = config.get('batch_size', 1)
    samples_per_step = gradient_accumulation * batch_size

    # 计算速度
    total_steps = train_history[-1]['step']
    total_samples = total_steps * samples_per_step

    # 估算时间（假设均匀分布）
    # 这里我们无法获取真实时间，只能根据记录点数量估算
    num_logs = len(train_history)
    logging_steps = config.get('logging_steps', 10)

    return {
        'total_steps': total_steps,
        'total_samples': total_samples,
        'num_logs': num_logs,
        'samples_per_step': samples_per_step,
        'logging_steps': logging_steps,
        'config': config,
    }


def print_status(log_path, clear_screen=False):
    """打印训练状态"""
    if clear_screen:
        os.system('clear' if os.name != 'nt' else 'cls')

    print("=" * 70)
    print("GRPO 训练速度监控")
    print("=" * 70)
    print(f"日志文件: {log_path}")
    print(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    log = load_log(log_path)

    if not log:
        print("❌ 无法加载训练日志")
        print()
        print("可能原因：")
        print("  1. 训练还未开始")
        print("  2. 日志文件路径不正确")
        print("  3. 日志文件格式错误")
        return

    # 分析速度
    speed_info = analyze_speed(log)

    if not speed_info:
        print("⏳ 训练刚开始，数据不足")
        print()
        print("建议：等待 10-20 分钟后再查看")
        return

    # 显示配置
    config = speed_info['config']
    print("📋 训练配置")
    print("-" * 70)
    print(f"  模型: {config.get('model_path', 'N/A')}")
    print(f"  图片大小: {config.get('max_image_size', 'N/A')}px")
    print(f"  生成数量: {config.get('num_generations', 'N/A')}")
    print(f"  Batch Size: {config.get('batch_size', 'N/A')}")
    print(f"  梯度累积: {config.get('gradient_accumulation_steps', 'N/A')}")
    print(f"  LoRA Rank: {config.get('lora_r', 'N/A')}")
    print()

    # 显示进度
    print("📊 训练进度")
    print("-" * 70)
    print(f"  已完成步数: {speed_info['total_steps']}")
    print(f"  已处理样本: {speed_info['total_samples']}")
    print(f"  记录点数量: {speed_info['num_logs']}")
    print()

    # 显示最新指标
    train_history = log.get('train_history', [])
    if train_history:
        latest = train_history[-1]
        print("📈 最新指标")
        print("-" * 70)
        for key, value in latest.items():
            if key not in ['step', 'epoch', 'samples']:
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        print()

    # 显示验证结果
    val_history = log.get('val_history', [])
    if val_history:
        latest_val = val_history[-1]
        print("✅ 最新验证结果")
        print("-" * 70)
        for key, value in latest_val.items():
            if key not in ['step', 'epoch', 'samples']:
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        print()

    # 显示最佳检查点
    best_checkpoint = log.get('best_checkpoint')
    if best_checkpoint:
        print("🏆 最佳检查点")
        print("-" * 70)
        print(f"  Step: {best_checkpoint.get('step', 'N/A')}")
        print(f"  Epoch: {best_checkpoint.get('epoch', 'N/A')}")
        for key, value in best_checkpoint.items():
            if key not in ['step', 'epoch', 'path', 'samples']:
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
        print(f"  路径: {best_checkpoint.get('path', 'N/A')}")
        print()

    # 优化建议
    print("💡 优化建议")
    print("-" * 70)

    suggestions = []

    # 检查图片大小
    max_image_size = config.get('max_image_size', 512)
    if max_image_size >= 1024:
        suggestions.append("⚠️  图片大小过大 (1024px)，建议降低到 512px 以提升 4 倍速度")
    elif max_image_size >= 768:
        suggestions.append("💡 图片大小较大 (768px)，可降低到 512px 以提升 2 倍速度")

    # 检查模型大小
    model_path = config.get('model_path', '')
    if '8B' in model_path or '7B' in model_path:
        suggestions.append("💡 使用大模型 (8B)，可改用 2B 模型以提升 4 倍速度")

    # 检查生成数量
    num_generations = config.get('num_generations', 4)
    if num_generations >= 6:
        suggestions.append("💡 生成数量较多 (6+)，可降低到 4 以提升速度")

    # 检查验证频率
    eval_steps = config.get('eval_steps', 0)
    if eval_steps > 0 and eval_steps < 500:
        suggestions.append("💡 验证频率较高，可设置 EVAL_STEPS=0 以提升 1.5 倍速度")

    # 检查 LoRA rank
    lora_r = config.get('lora_r', 64)
    if lora_r >= 64:
        suggestions.append("💡 LoRA rank 较大 (64+)，可降低到 32 以提升 ~20% 速度")

    if suggestions:
        for suggestion in suggestions:
            print(f"  {suggestion}")
    else:
        print("  ✅ 配置已优化，无明显瓶颈")

    print()
    print("=" * 70)


def watch_mode(log_path, interval=60):
    """监控模式"""
    print("进入监控模式（按 Ctrl+C 退出）")
    print(f"刷新间隔: {interval} 秒")
    print()

    try:
        while True:
            print_status(log_path, clear_screen=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n监控已停止")


def main():
    parser = argparse.ArgumentParser(description="GRPO 训练速度监控")
    parser.add_argument("--log", type=str, required=True,
                        help="训练日志文件路径")
    parser.add_argument("--watch", action="store_true",
                        help="监控模式（实时刷新）")
    parser.add_argument("--interval", type=int, default=60,
                        help="监控模式刷新间隔（秒）")

    args = parser.parse_args()

    if args.watch:
        watch_mode(args.log, args.interval)
    else:
        print_status(args.log)


if __name__ == "__main__":
    main()
