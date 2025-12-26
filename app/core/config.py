"""Configuration management for the detection service."""

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelType(str, Enum):
    """Supported model types."""

    # Qwen3-VL series
    QWEN3_VL_2B = "Qwen/Qwen3-VL-2B-Instruct"
    QWEN3_VL_4B = "Qwen/Qwen3-VL-4B-Instruct"
    QWEN3_VL_8B = "Qwen/Qwen3-VL-8B-Instruct"
    QWEN3_VL_32B = "Qwen/Qwen3-VL-32B-Instruct"

    # Qwen2.5-VL series
    QWEN2_5_VL_3B = "Qwen/Qwen2.5-VL-3B-Instruct"
    QWEN2_5_VL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"
    QWEN2_5_VL_32B = "Qwen/Qwen2.5-VL-32B-Instruct"
    QWEN2_5_VL_72B = "Qwen/Qwen2.5-VL-72B-Instruct"

    # Custom/local model
    CUSTOM = "custom"


# Model info: (VRAM requirement GB, description)
MODEL_INFO = {
    ModelType.QWEN3_VL_2B: (6, "轻量版，适合 RTX 3060/4060"),
    ModelType.QWEN3_VL_4B: (10, "小型版，适合 RTX 3070/4070"),
    ModelType.QWEN3_VL_8B: (18, "标准版，适合 RTX 3090/4090"),
    ModelType.QWEN3_VL_32B: (70, "大型版，需要多卡"),
    ModelType.QWEN2_5_VL_3B: (8, "Qwen2.5 轻量版"),
    ModelType.QWEN2_5_VL_7B: (16, "Qwen2.5 标准版，更稳定"),
    ModelType.QWEN2_5_VL_32B: (70, "Qwen2.5 大型版"),
    ModelType.QWEN2_5_VL_72B: (150, "Qwen2.5 超大版，需要多卡"),
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Model settings
    model_type: ModelType = Field(
        default=ModelType.QWEN3_VL_8B,
        description="Predefined model type to use",
    )
    model_name: str | None = Field(
        default=None,
        description="Custom model name/path (overrides model_type if set)",
    )

    @property
    def effective_model_name(self) -> str:
        """Get the effective model name to use."""
        if self.model_name:
            return self.model_name
        return self.model_type.value

    @property
    def is_qwen3_vl(self) -> bool:
        """Check if using Qwen3-VL model series."""
        model_name = self.effective_model_name.lower()
        return "qwen3-vl" in model_name or "qwen3_vl" in model_name
    model_cache_dir: str = Field(
        default="./model_cache",
        description="Directory to cache downloaded models",
    )
    device: str = Field(
        default="cuda",
        description="Device to run inference on (cuda/cpu)",
    )
    torch_dtype: Literal["float16", "bfloat16", "float32"] = Field(
        default="bfloat16",
        description="Torch dtype for model weights",
    )
    max_new_tokens: int = Field(
        default=2048,
        description="Maximum number of new tokens to generate",
    )

    # API settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=1, description="Number of API workers")
    api_reload: bool = Field(default=False, description="Enable auto-reload")

    # Detection settings
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for detections",
    )
    max_image_size: int = Field(
        default=1280,
        description="Maximum image dimension (will resize if larger)",
    )
    enable_bbox: bool = Field(
        default=True,
        description="Enable bounding box detection",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
