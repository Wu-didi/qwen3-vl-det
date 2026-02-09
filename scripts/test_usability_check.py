#!/usr/bin/env python3
"""
测试可用性自检功能

这个脚本创建一个模拟的训练日志，用于测试可视化功能。
"""

import json
import os
import random

def create_mock_training_log(output_path: str):
    """创建模拟的训练日志"""

    # 模拟配置
    config = {
        "model_path": "Qwen/Qwen3-VL-2B-Instruct",
        "output_dir": "outputs/qwen3vl_grpo",
        "num_epochs": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-6,
        "lora_r": 64,
        "num_generations": 4,
        "kl_coef": 0.5,
        "usability_check_steps": 200,
        "usability_check_samples": 8,
    }

    # 模拟训练历史（1000 步）
    train_history = []
    for step in range(1, 1001):
        # 模拟 loss 下降
        loss = 0.5 * (1 - step / 1000) + 0.1 + random.uniform(-0.02, 0.02)
        # 模拟 reward 上升
        reward = 1.0 + 2.0 * (step / 1000) + random.uniform(-0.1, 0.1)
        # 模拟 KL 散度
        kl = 0.05 + random.uniform(-0.01, 0.01)
        # 模拟学习率（warmup）
        if step < 100:
            lr = 5e-6 * (step / 100)
        else:
            lr = 5e-6

        train_history.append({
            "step": step,
            "epoch": 1,
            "samples": step * 4,
            "loss": loss,
            "reward": reward,
            "kl": kl,
            "lr": lr,
        })

    # 模拟验证历史（每 200 步）
    val_history = []
    for step in range(200, 1001, 200):
        progress = step / 1000
        val_history.append({
            "step": step,
            "epoch": 1,
            "samples": step * 4,
            "val_reward": 1.0 + 2.0 * progress + random.uniform(-0.1, 0.1),
            "val_format": 0.5 + 0.4 * progress + random.uniform(-0.05, 0.05),
            "val_bbox": 0.3 + 0.5 * progress + random.uniform(-0.05, 0.05),
            "val_category": 0.4 + 0.5 * progress + random.uniform(-0.05, 0.05),
            "val_completeness": 0.5 + 0.4 * progress + random.uniform(-0.05, 0.05),
        })

    # 模拟可用性自检历史（每 200 步）
    usability_history = []
    for step in range(200, 1001, 200):
        progress = step / 1000

        # 模拟两种情况：
        # 1. 健康训练：usability 指标随 reward 上升
        # 2. 假训练：usability 指标停滞

        # 这里模拟健康训练
        usability_history.append({
            "step": step,
            "epoch": 1,
            "samples": step * 4,
            "dataset": "val",
            "usability_parse_success_rate": 0.5 + 0.4 * progress + random.uniform(-0.05, 0.05),
            "usability_has_box_rate": 0.6 + 0.35 * progress + random.uniform(-0.05, 0.05),
            "usability_has_numbered_rate": 0.7 + 0.25 * progress + random.uniform(-0.05, 0.05),
            "usability_has_status_rate": 0.6 + 0.3 * progress + random.uniform(-0.05, 0.05),
            "usability_avg_iou": 0.3 + 0.5 * progress + random.uniform(-0.05, 0.05),
            "usability_category_accuracy": 0.4 + 0.5 * progress + random.uniform(-0.05, 0.05),
            "usability_anomaly_accuracy": 0.5 + 0.4 * progress + random.uniform(-0.05, 0.05),
            "usability_avg_pred_boxes": 1.5 + 0.5 * progress + random.uniform(-0.1, 0.1),
            "usability_avg_gt_boxes": 2.0,
        })

    # 最佳检查点
    best_checkpoint = {
        "step": 1000,
        "epoch": 1,
        "samples": 4000,
        "val_reward": 2.95,
        "path": "outputs/qwen3vl_grpo/best",
    }

    # 组装日志
    log = {
        "config": config,
        "train_history": train_history,
        "val_history": val_history,
        "usability_history": usability_history,
        "best_checkpoint": best_checkpoint,
    }

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(log, f, indent=2)

    print(f"Created mock training log: {output_path}")
    print(f"  Train history: {len(train_history)} steps")
    print(f"  Val history: {len(val_history)} checkpoints")
    print(f"  Usability history: {len(usability_history)} checks")


def main():
    # 创建模拟日志
    output_path = "test_outputs/mock_training_log.json"
    create_mock_training_log(output_path)

    print("\nTest the visualization script:")
    print(f"  python scripts/visualize_training_log.py --log {output_path}")
    print(f"  python scripts/visualize_training_log.py --log {output_path} --output test_outputs/plots/")
    print(f"  python scripts/visualize_training_log.py --log {output_path} --export-csv")


if __name__ == "__main__":
    main()
