"""RFT 模型构建工具（Qwen-VL + LoRA/QLoRA）。"""

import logging
import os

import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration


logger = logging.getLogger(__name__)


def get_model_class(model_path: str):
    """根据配置自动选择 Qwen3-VL 或 Qwen2.5-VL 模型类。"""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "").lower()

    if model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    if model_type == "qwen2_5_vl":
        return Qwen2_5_VLForConditionalGeneration

    if "qwen3" in model_path.lower():
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    return Qwen2_5_VLForConditionalGeneration


def create_model_and_processor(
    model_path: str,
    sft_model_path: str = "",
    use_4bit: bool = True,
    bf16: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
):
    """创建模型与处理器，并准备 PEFT 训练。"""
    logger.info("Loading model from %s", model_path)

    model_class = get_model_class(model_path)
    logger.info("Using model class: %s", model_class.__name__)

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    try:
        import flash_attn  # noqa: F401

        attn_impl = "flash_attention_2"
        logger.info("Using Flash Attention 2")
    except ImportError:
        attn_impl = "sdpa"
        logger.info("Flash Attention not available, using SDPA")

    model = model_class.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    if sft_model_path and os.path.exists(sft_model_path):
        from peft import PeftModel

        logger.info("Loading SFT LoRA weights from %s", sft_model_path)
        model = PeftModel.from_pretrained(model, sft_model_path, is_trainable=True)
        logger.info("SFT LoRA weights loaded, model is already a PEFT model")

    return model, processor, peft_config
