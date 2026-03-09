import logging
from importlib.metadata import PackageNotFoundError, version

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig

# 同时兼容脚本运行和模块运行。
try:
    from sft_config import FinetuneConfig
except ImportError:  # pragma: no cover
    from .sft_config import FinetuneConfig


logger = logging.getLogger(__name__)


def ensure_bitsandbytes_available(config: FinetuneConfig) -> None:
    """在启用 4bit/8bit 量化前检查 bitsandbytes 依赖。"""
    if not (config.use_4bit or config.use_8bit):
        return

    try:
        bnb_version = version("bitsandbytes")
        logger.info("bitsandbytes version: %s", bnb_version)
    except PackageNotFoundError as exc:
        quant_mode = "4bit" if config.use_4bit else "8bit"
        raise RuntimeError(
            f"bitsandbytes is required for {quant_mode} quantization. "
            "Install it with `python -m pip install bitsandbytes` "
            "or disable 4bit with `--no_4bit`."
        ) from exc


def get_model_class(model_path: str):
    """根据模型路径选择 Qwen3-VL 或 Qwen2.5-VL 的模型类。"""
    model_path_lower = model_path.lower()
    if "qwen3" in model_path_lower:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration

    from transformers import Qwen2_5_VLForConditionalGeneration

    return Qwen2_5_VLForConditionalGeneration


def create_model_and_processor(config: FinetuneConfig):
    """创建模型与处理器，并挂载 LoRA 适配器。"""
    logger.info("Loading model from %s", config.model_path)
    ensure_bitsandbytes_available(config)

    # 1) 动态选择模型类，支持不同 Qwen-VL 版本。
    model_class = get_model_class(config.model_path)
    logger.info("Using model class: %s", model_class.__name__)

    # 2) 配置量化参数（4bit / 8bit）。
    bnb_config = None
    if config.use_4bit:
        # QLoRA 默认配置：nf4 + double quant + bf16/fp16 compute dtype。
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif config.use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # 3) 加载基础模型。
    model = model_class.from_pretrained(
        config.model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 4) 加载处理器（文本 tokenizer + 视觉处理）。
    processor = AutoProcessor.from_pretrained(
        config.model_path,
        trust_remote_code=True,
    )

    # 5) 量化训练前准备（冻结/类型处理/梯度检查点配合）。
    if config.use_4bit or config.use_8bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )

    # 6) 构建 LoRA 配置并注入。
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # 打印可训练参数比例，便于确认 LoRA 挂载是否成功。
    model.print_trainable_parameters()

    return model, processor
