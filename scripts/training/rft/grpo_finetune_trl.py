#!/usr/bin/env python3
"""
使用 TRL 的 GRPOTrainer 进行 Qwen-VL 强化学习微调。

当前文件只保留“训练编排”和“参数解析”，核心逻辑已拆分：
- rewarding.py: 奖励函数与文本解析
- data_utils.py: 数据加载与图像懒加载 collator
- model_utils.py: 模型与处理器构建
- qwen_grpo_trainer.py: 多模态 GRPOTrainer 适配
"""

import json
import logging
import os

import torch
from trl import GRPOConfig as TRLGRPOConfig

# 兼容脚本运行与模块运行两种导入方式。
try:
    from data_utils import create_data_collator, create_grpo_data_collator, load_and_prepare_dataset, load_grpo_dataset
    from grpo_reward_functions import build_reward_funcs
    from model_utils import create_model_and_processor
    from qwen_grpo_trainer import QwenVLGRPOTrainer
    from rewarding import (
        RiskRewardConfig,
        anomaly_instance_f1_reward,
        bbox_iou_reward,
        category_match_reward,
        count_alignment_reward,
        format_reward,
        get_risk_reward_config,
        localization_quality_reward,
        risk_control_reward,
        set_f1_reward,
        set_risk_reward_config,
        status_accuracy_reward,
    )
except ImportError:  # pragma: no cover
    from .data_utils import create_data_collator, create_grpo_data_collator, load_and_prepare_dataset, load_grpo_dataset
    from .grpo_reward_functions import build_reward_funcs
    from .model_utils import create_model_and_processor
    from .qwen_grpo_trainer import QwenVLGRPOTrainer
    from .rewarding import (
        RiskRewardConfig,
        anomaly_instance_f1_reward,
        bbox_iou_reward,
        category_match_reward,
        count_alignment_reward,
        format_reward,
        get_risk_reward_config,
        localization_quality_reward,
        risk_control_reward,
        set_f1_reward,
        set_risk_reward_config,
        status_accuracy_reward,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数。"""
    import argparse

    parser = argparse.ArgumentParser(description="GRPO fine-tuning with TRL")

    # -------------------- 模型参数 --------------------
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument(
        "--sft_model_path",
        type=str,
        default="",
        help="Path to SFT LoRA model to continue from",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization (default: enabled, use --no_4bit to disable)",
    )
    parser.add_argument(
        "--no_4bit",
        dest="use_4bit",
        action="store_false",
        help="Disable 4-bit quantization",
    )
    parser.set_defaults(use_4bit=True)
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 precision (default: enabled, use --no_bf16 to disable)",
    )
    parser.add_argument(
        "--no_bf16",
        dest="bf16",
        action="store_false",
        help="Disable bfloat16 precision",
    )
    parser.set_defaults(bf16=True)

    # -------------------- LoRA 参数 --------------------
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # -------------------- 数据参数 --------------------
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default="", help="Path to validation data (optional)")
    parser.add_argument("--max_image_size", type=int, default=512)

    # -------------------- GRPO 参数 --------------------
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient")
    parser.add_argument(
        "--reward_scheme",
        type=str,
        default="new_json",
        choices=["new_json", "risk_aware", "legacy"],
        help="Reward design: new_json (recommended, for rft_output data), risk_aware, or legacy",
    )
    parser.add_argument(
        "--reward_match_iou",
        type=float,
        default=0.5,
        help="IoU threshold for one-to-one matching in risk-aware rewards",
    )
    parser.add_argument(
        "--reward_hallucination_unit_penalty",
        type=float,
        default=0.35,
        help="Per-instance hallucination penalty (risk-aware rewards)",
    )
    parser.add_argument(
        "--reward_no_detection_missing_penalty",
        type=float,
        default=0.2,
        help="Penalty when GT is empty but model does not output explicit no-detection text",
    )
    parser.add_argument(
        "--reward_omission_penalty",
        type=float,
        default=1.0,
        help="Penalty strength for missing detections on positive samples",
    )
    parser.add_argument("--reward_w_format", type=float, default=0.2)
    parser.add_argument("--reward_w_set_f1", type=float, default=3.0)
    parser.add_argument("--reward_w_iou", type=float, default=2.0)
    parser.add_argument("--reward_w_count", type=float, default=1.2)
    parser.add_argument("--reward_w_risk", type=float, default=2.5)
    parser.add_argument("--reward_w_anomaly", type=float, default=2.0)

    # -------------------- 训练参数 --------------------
    parser.add_argument("--output_dir", type=str, default="outputs/qwen3vl_grpo_trl")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluate every N steps (0 to disable)")

    # -------------------- 日志参数 --------------------
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="qwen-vl-grpo", help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for logging")

    # -------------------- 其他参数 --------------------
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def build_reward_bundle(args):
    """根据 reward_scheme 构建奖励函数与权重。"""
    if args.reward_scheme == "new_json":
        # 使用 grpo_reward_functions.py 中的新版 JSON 格式 reward（配合 rft_output 数据）
        reward_funcs = build_reward_funcs(as_list=True)
        reward_weights = [0.2, 0.2, 0.6, 0.25, 0.25, 0.4, -0.05]
        logger.info("Using NEW_JSON reward scheme (grpo_reward_functions.py)")
    elif args.reward_scheme == "legacy":
        reward_funcs = [
            format_reward,
            bbox_iou_reward,
            category_match_reward,
            status_accuracy_reward,
        ]
        reward_weights = [0.2, 3.0, 2.0, 2.0]
        logger.info("Using LEGACY reward scheme")
    else:
        reward_funcs = [
            format_reward,
            set_f1_reward,
            localization_quality_reward,
            count_alignment_reward,
            risk_control_reward,
            anomaly_instance_f1_reward,
        ]
        reward_weights = [
            args.reward_w_format,
            args.reward_w_set_f1,
            args.reward_w_iou,
            args.reward_w_count,
            args.reward_w_risk,
            args.reward_w_anomaly,
        ]
        cfg = get_risk_reward_config()
        logger.info("Using RISK-AWARE reward scheme")
        logger.info(
            "Risk-aware reward cfg: match_iou=%.2f, halluc_penalty=%.2f, "
            "no_det_missing_penalty=%.2f, omission_penalty=%.2f",
            cfg.match_iou_threshold,
            cfg.hallucination_unit_penalty,
            cfg.no_detection_missing_penalty,
            cfg.omission_penalty,
        )
        logger.info("Risk-aware reward weights: %s", reward_weights)

    return reward_funcs, reward_weights


def build_report_to(args):
    """配置日志后端。"""
    if args.use_wandb:
        try:
            import wandb

            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args),
            )
            logger.info("W&B logging enabled: %s/%s", args.wandb_project, args.run_name)
            return ["tensorboard", "wandb"]
        except ImportError:
            logger.warning("wandb not installed, falling back to tensorboard only")
            return ["tensorboard"]
    return ["tensorboard"]


def save_training_log(trainer, args):
    """从 trainer.state 中提取并保存训练日志。"""
    training_log = {
        "config": vars(args),
        "train_history": [],
        "val_history": [],
        "final_metrics": {},
    }

    if hasattr(trainer.state, "log_history"):
        for log_entry in trainer.state.log_history:
            if "loss" in log_entry or "reward" in log_entry:
                entry = {
                    "step": log_entry.get("step", 0),
                    "epoch": log_entry.get("epoch", 0),
                }
                for key in ["loss", "reward", "kl", "learning_rate"]:
                    if key in log_entry:
                        entry[key] = log_entry[key]

                if "eval_" not in str(log_entry):
                    training_log["train_history"].append(entry)

            if "eval_reward" in log_entry or "eval_loss" in log_entry:
                val_entry = {
                    "step": log_entry.get("step", 0),
                    "epoch": log_entry.get("epoch", 0),
                }
                for key in log_entry:
                    if key.startswith("eval_"):
                        val_entry[key] = log_entry[key]
                training_log["val_history"].append(val_entry)

    if hasattr(trainer.state, "best_metric"):
        training_log["final_metrics"]["best_metric"] = trainer.state.best_metric
    if hasattr(trainer.state, "best_model_checkpoint"):
        training_log["final_metrics"]["best_checkpoint"] = trainer.state.best_model_checkpoint

    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2, default=str)
    logger.info("Training log saved to %s", log_path)


def main():
    args = parse_args()

    if args.run_name is None:
        import datetime

        args.run_name = f"grpo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    torch.manual_seed(args.seed)

    set_risk_reward_config(
        RiskRewardConfig(
            match_iou_threshold=max(0.1, min(0.95, float(args.reward_match_iou))),
            hallucination_unit_penalty=max(0.0, float(args.reward_hallucination_unit_penalty)),
            no_detection_missing_penalty=max(0.0, float(args.reward_no_detection_missing_penalty)),
            omission_penalty=max(0.0, float(args.reward_omission_penalty)),
        )
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    model, processor, peft_config = create_model_and_processor(
        model_path=args.model_path,
        sft_model_path=args.sft_model_path,
        use_4bit=args.use_4bit,
        bf16=args.bf16,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # 根据 reward_scheme 选择数据加载方式
    use_grpo_format = args.reward_scheme == "new_json"
    _load_dataset = load_grpo_dataset if use_grpo_format else load_and_prepare_dataset
    _collator = create_grpo_data_collator if use_grpo_format else create_data_collator

    dataset = _load_dataset(
        data_path=args.train_data,
        processor=processor,
        max_image_size=args.max_image_size,
    )

    eval_dataset = None
    if args.val_data and os.path.exists(args.val_data):
        logger.info("Loading validation data from %s", args.val_data)
        eval_dataset = _load_dataset(
            data_path=args.val_data,
            processor=processor,
            max_image_size=args.max_image_size,
        )
    else:
        logger.info("No validation data provided, skipping validation")

    reward_funcs, reward_weights = build_reward_bundle(args)
    report_to = build_report_to(args)

    logger.info("Configuring GRPO training...")
    training_args = TRLGRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset and args.eval_steps > 0 else "no",
        save_total_limit=3,
        bf16=args.bf16,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
        temperature=args.temperature,
        reward_weights=reward_weights,
        report_to=report_to,
        logging_first_step=True,
        seed=args.seed,
    )

    logger.info("Creating trainer...")
    trainer = QwenVLGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
    )

    trainer.data_collator = _collator(processor)

    logger.info("Starting GRPO training with TRL...")
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    save_training_log(trainer, args)
    logger.info("Training complete! Model saved to %s", final_dir)


if __name__ == "__main__":
    main()
