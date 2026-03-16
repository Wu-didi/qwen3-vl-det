"""SFT 训练配置定义。

该文件的职责是集中管理默认超参数，避免散落在多个文件中。
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class FinetuneConfig:
    """LoRA/QLoRA 监督微调配置。"""

    # ------------------------------------------------------------------
    # 模型与量化
    # ------------------------------------------------------------------
    # 预训练模型路径（本地目录或 Hugging Face 模型名）。
    model_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    # 是否启用 4bit 量化（QLoRA）。
    use_4bit: bool = True
    # 是否启用 8bit 量化（与 4bit 二选一）。
    use_8bit: bool = False

    # ------------------------------------------------------------------
    # LoRA 参数
    # ------------------------------------------------------------------
    # LoRA rank，越大表达能力越强，同时显存占用也更高。
    lora_r: int = 64
    # LoRA 缩放参数。
    lora_alpha: int = 16
    # LoRA dropout，有助于提升泛化。
    lora_dropout: float = 0.1
    # 需要注入 LoRA 的线性层名称。
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # ------------------------------------------------------------------
    # 数据参数
    # ------------------------------------------------------------------
    # 训练/验证数据，支持 JSON array (.json) 或 JSON Lines (.jsonl) 格式。
    train_data: str = "data/hefei_last_dataset/sft_output/train.jsonl"
    val_data: str = "data/hefei_last_dataset/sft_output/val.jsonl"
    # 文本 token 最大长度，超出会截断。
    max_length: int = 2048
    # 图像最长边缩放上限，控制显存与视觉 token 数。
    max_image_size: int = 512

    # ------------------------------------------------------------------
    # 训练参数
    # ------------------------------------------------------------------
    # 输出目录（checkpoint、模型权重）。
    output_dir: str = "outputs/qwen3vl_lora"
    # 日志目录（training_log.json、finetune_config.json）。
    # 留空时自动推导为 logs/<output_dir 的最后一级目录名>。
    log_dir: str = ""
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # ------------------------------------------------------------------
    # 其他运行参数
    # ------------------------------------------------------------------
    seed: int = 42
    # 是否使用 bfloat16；若关闭则训练逻辑会退回 fp16。
    bf16: bool = True
    # 是否启用梯度检查点，用于降低显存。
    gradient_checkpointing: bool = True
