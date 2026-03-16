#!/usr/bin/env python3
"""
Qwen-VL SFT（LoRA/QLoRA）训练入口脚本。

这个文件只负责两件事：
1. 解析命令行参数。
2. 把参数组装成配置对象并调用训练主流程。

真正的训练逻辑已经拆分到同目录模块中：
- sft_config.py: 训练配置 dataclass
- sft_dataset.py: 数据读取与标签掩码逻辑
- sft_model.py: 模型/处理器加载与 LoRA 挂载
- sft_collator.py: 多模态 batch 拼接与 padding
- sft_train.py: Trainer 训练编排与日志落盘

Usage:
    python scripts/training/sft/finetune_qwen_vl.py \
        --model_path Qwen/Qwen3-VL-2B-Instruct \
        --train_data data/hefei_last_dataset/qwen_data/train.json \
        --val_data data/hefei_last_dataset/qwen_data/val.json \
        --output_dir outputs/qwen3vl_lora
"""

import argparse
import logging

# 兼容两种运行方式：
# 1) 直接脚本运行：python scripts/training/sft/finetune_qwen_vl.py
# 2) 模块方式运行：python -m scripts.training.sft.finetune_qwen_vl
# 这里用 try/except 避免相对导入路径问题。
try:
    from sft_config import FinetuneConfig
    from sft_train import train
except ImportError:  # pragma: no cover
    from .sft_config import FinetuneConfig
    from .sft_train import train


# 统一日志格式，方便训练期间排查问题。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-VL for traffic anomaly detection"
    )

    # ------------------------------------------------------------------
    # 1) 模型与量化相关参数
    # ------------------------------------------------------------------
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Path to pretrained model",
    )
    # 默认开启 4bit；传入 --no_4bit 才关闭。
    parser.add_argument(
        "--no_4bit",
        action="store_false",
        dest="use_4bit",
        default=True,
        help="Disable 4-bit quantization (QLoRA). Default: enabled",
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        default=False,
        help="Use 8-bit quantization",
    )

    # ------------------------------------------------------------------
    # 2) LoRA 超参数
    # ------------------------------------------------------------------
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout",
    )

    # ------------------------------------------------------------------
    # 3) 数据与序列长度参数
    # ------------------------------------------------------------------
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/hefei_last_dataset/qwen_data/train.json",
        help="Path to training data",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="data/hefei_last_dataset/qwen_data/val.json",
        help="Path to validation data",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max_image_size",
        type=int,
        default=512,
        help="Maximum image size (longest edge). Larger = better quality but more VRAM",
    )

    # ------------------------------------------------------------------
    # 4) 训练调度参数
    # ------------------------------------------------------------------
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/qwen3vl_lora",
        help="Output directory for model weights and checkpoints",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Directory for training logs (training_log.json, finetune_config.json). "
             "Defaults to logs/<output_dir_basename> if not specified.",
    )
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps",
    )

    # ------------------------------------------------------------------
    # 5) 精度/稳定性参数
    # ------------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # 默认使用 bf16；传入 --no_bf16 则使用 fp16。
    parser.add_argument(
        "--no_bf16",
        action="store_false",
        dest="bf16",
        default=True,
        help="Disable bfloat16. Default: enabled",
    )
    # 默认开启梯度检查点；传入 --no_gradient_checkpointing 可关闭。
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_false",
        dest="gradient_checkpointing",
        default=True,
        help="Disable gradient checkpointing. Default: enabled",
    )

    return parser.parse_args()


def build_config(args) -> FinetuneConfig:
    """将 argparse 结果映射为 FinetuneConfig。"""
    # 量化模式优先级说明：
    # - 默认 use_4bit=True
    # - 如果显式传入 --use_8bit，则强制关闭 4bit，避免二者冲突。
    return FinetuneConfig(
        model_path=args.model_path,
        use_4bit=args.use_4bit and not args.use_8bit,
        use_8bit=args.use_8bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        train_data=args.train_data,
        val_data=args.val_data,
        max_length=args.max_length,
        max_image_size=args.max_image_size,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        seed=args.seed,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
    )


def main():
    """入口函数：解析参数 -> 构建配置 -> 启动训练。"""
    args = parse_args()
    config = build_config(args)
    train(config)


if __name__ == "__main__":
    main()
