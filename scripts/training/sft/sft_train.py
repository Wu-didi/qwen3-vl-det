import json
import logging
import os

import torch
from transformers import Trainer, TrainingArguments

# 兼容脚本运行和模块运行两种导入路径。
try:
    from sft_collator import VLDataCollator
    from sft_config import FinetuneConfig
    from sft_dataset import TrafficAnomalyDataset
    from sft_model import create_model_and_processor
except ImportError:  # pragma: no cover
    from .sft_collator import VLDataCollator
    from .sft_config import FinetuneConfig
    from .sft_dataset import TrafficAnomalyDataset
    from .sft_model import create_model_and_processor


logger = logging.getLogger(__name__)


def train(config: FinetuneConfig):
    """SFT 主训练流程。"""
    logger.info("Starting fine-tuning...")

    # 1) 固定随机种子，提升可复现性。
    torch.manual_seed(config.seed)

    # 2) 创建输出目录。
    os.makedirs(config.output_dir, exist_ok=True)

    # 3) 保存本次训练配置，便于回溯。
    config_path = os.path.join(config.output_dir, "finetune_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2, default=str)

    # 4) 构建模型与处理器。
    model, processor = create_model_and_processor(config)

    # 5) 构建训练集。
    train_dataset = TrafficAnomalyDataset(
        config.train_data,
        processor,
        config.max_length,
        config.max_image_size,
    )

    # 6) 可选构建验证集（仅当路径存在时启用）。
    val_dataset = None
    if config.val_data and os.path.exists(config.val_data):
        val_dataset = TrafficAnomalyDataset(
            config.val_data,
            processor,
            config.max_length,
            config.max_image_size,
        )

    # 7) 组装 Hugging Face TrainingArguments。
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if val_dataset else None,
        eval_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        save_total_limit=3,
        bf16=config.bf16,
        fp16=not config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
        seed=config.seed,
    )

    # 8) 多模态自定义 collator，处理文本与图像张量拼接。
    data_collator = VLDataCollator(processor)

    # 9) 创建 Trainer。
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # 10) 启动训练。
    logger.info("Starting training...")
    trainer.train()

    # 11) 保存最终模型与处理器。
    logger.info("Saving model to %s", config.output_dir)
    trainer.save_model()
    processor.save_pretrained(config.output_dir)

    # 12) 组装训练日志结构。
    training_log = {
        "config": vars(config),
        "train_history": [],
        "val_history": [],
        "final_metrics": {},
    }

    # 13) 从 trainer.state.log_history 抽取训练/验证曲线。
    if hasattr(trainer.state, "log_history"):
        for log_entry in trainer.state.log_history:
            if "loss" in log_entry and "epoch" in log_entry:
                training_log["train_history"].append(
                    {
                        "step": log_entry.get("step", 0),
                        "epoch": log_entry.get("epoch", 0),
                        "loss": log_entry.get("loss", 0),
                        "learning_rate": log_entry.get("learning_rate", 0),
                    }
                )
            elif "eval_loss" in log_entry:
                training_log["val_history"].append(
                    {
                        "step": log_entry.get("step", 0),
                        "epoch": log_entry.get("epoch", 0),
                        "eval_loss": log_entry.get("eval_loss", 0),
                    }
                )

    # 14) 补充最优模型相关字段（若 Trainer 提供）。
    if hasattr(trainer.state, "best_metric"):
        training_log["final_metrics"]["best_metric"] = trainer.state.best_metric
    if hasattr(trainer.state, "best_model_checkpoint"):
        training_log["final_metrics"]["best_checkpoint"] = trainer.state.best_model_checkpoint

    # 15) 写入 training_log.json。
    log_path = os.path.join(config.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2, default=str)
    logger.info("Training log saved to %s", log_path)

    logger.info("Fine-tuning completed!")
