#!/usr/bin/env python3
"""Script to download and cache the Qwen VL model."""

import argparse
import os
import sys
from enum import Enum


class ModelSource(str, Enum):
    """Model download source."""
    MODELSCOPE = "modelscope"
    HUGGINGFACE = "huggingface"


# ModelScope 模型名称映射
MODELSCOPE_MODELS = {
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3-vl-32b": "Qwen/Qwen3-VL-32B-Instruct",
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    "qwen2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
}

# HuggingFace 模型名称映射
HUGGINGFACE_MODELS = {
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3-vl-32b": "Qwen/Qwen3-VL-32B-Instruct",
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    "qwen2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
}

MODEL_INFO = {
    "qwen3-vl-2b": (6, "轻量版，适合 RTX 3060/4060"),
    "qwen3-vl-4b": (10, "小型版，适合 RTX 3070/4070"),
    "qwen3-vl-8b": (18, "标准版，适合 RTX 3090/4090"),
    "qwen3-vl-32b": (70, "大型版，需要多卡"),
    "qwen2.5-vl-3b": (8, "Qwen2.5 轻量版"),
    "qwen2.5-vl-7b": (16, "Qwen2.5 标准版，更稳定"),
    "qwen2.5-vl-32b": (70, "Qwen2.5 大型版"),
    "qwen2.5-vl-72b": (150, "Qwen2.5 超大版，需要多卡"),
}


def list_models() -> None:
    """Print available models."""
    print("\n可用模型列表:")
    print("-" * 60)
    print(f"{'模型类型':<18} {'显存需求':<10} {'说明'}")
    print("-" * 60)
    for key, (vram, desc) in MODEL_INFO.items():
        print(f"{key:<18} {vram:>4} GB     {desc}")
    print("-" * 60)


def download_from_modelscope(
    model_name: str,
    cache_dir: str,
) -> str:
    """Download model from ModelScope.

    Returns:
        Local path to the downloaded model.
    """
    from modelscope import snapshot_download

    print(f"从 ModelScope 下载模型: {model_name}")
    print(f"缓存目录: {cache_dir}")

    local_path = snapshot_download(
        model_name,
        cache_dir=cache_dir,
    )

    print(f"模型已下载到: {local_path}")
    return local_path


def download_from_huggingface(
    model_name: str,
    cache_dir: str,
    use_auth_token: bool = False,
) -> None:
    """Download model from HuggingFace."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"从 HuggingFace 下载模型: {model_name}")
    print(f"缓存目录: {cache_dir}")

    print("\n正在下载 processor...")
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        token=use_auth_token if use_auth_token else None,
    )
    print("Processor 下载成功!")

    print("\n正在下载模型 (可能需要较长时间)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        token=use_auth_token if use_auth_token else None,
    )
    print("模型下载成功!")


def download_model(
    model_type: str = "qwen3-vl-8b",
    model_name: str | None = None,
    cache_dir: str = "./model_cache",
    source: ModelSource = ModelSource.MODELSCOPE,
    use_auth_token: bool = False,
) -> None:
    """Download model from specified source.

    Args:
        model_type: Predefined model type.
        model_name: Custom model name (overrides model_type).
        cache_dir: Directory to cache downloaded files.
        source: Download source (modelscope or huggingface).
        use_auth_token: Whether to use auth token (HuggingFace only).
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Resolve model name based on source
    if model_name:
        actual_model_name = model_name
    else:
        models_map = MODELSCOPE_MODELS if source == ModelSource.MODELSCOPE else HUGGINGFACE_MODELS
        if model_type not in models_map:
            print(f"错误: 未知的模型类型 '{model_type}'")
            list_models()
            sys.exit(1)
        actual_model_name = models_map[model_type]

    try:
        if source == ModelSource.MODELSCOPE:
            local_path = download_from_modelscope(actual_model_name, cache_dir)
            print(f"\n下载完成!")
            print(f"模型路径: {local_path}")
            print(f"\n请在 .env 文件中设置:")
            print(f"MODEL_NAME={local_path}")
        else:
            download_from_huggingface(actual_model_name, cache_dir, use_auth_token)
            print(f"\n模型文件已缓存到: {cache_dir}")

        print("\n现在可以启动服务了。")

    except ImportError as e:
        if source == ModelSource.MODELSCOPE:
            print(f"错误: 缺少 modelscope 包。{e}")
            print("请安装: pip install modelscope")
        else:
            print(f"错误: 缺少必要的包。{e}")
            print("请安装: pip install transformers accelerate")
        sys.exit(1)
    except Exception as e:
        print(f"下载模型时出错: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="下载 Qwen VL 模型用于交通设备异常检测"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用的模型",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="qwen3-vl-2b",
        choices=list(MODEL_INFO.keys()),
        help="模型类型 (默认: qwen3-vl-2b)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="自定义模型名称或路径 (会覆盖 --type)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./model_cache",
        help="模型缓存目录 (默认: ./model_cache)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="modelscope",
        choices=["modelscope", "huggingface"],
        help="下载源 (默认: modelscope，国内推荐)",
    )
    parser.add_argument(
        "--auth",
        action="store_true",
        help="使用认证令牌 (仅 HuggingFace)",
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    download_model(
        model_type=args.type,
        model_name=args.model,
        cache_dir=args.cache_dir,
        source=ModelSource(args.source),
        use_auth_token=args.auth,
    )


if __name__ == "__main__":
    main()
