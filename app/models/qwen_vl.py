"""Qwen-VL model wrapper for inference."""

import logging
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class QwenVLModel:
    """Wrapper for Qwen3-VL model inference."""

    def __init__(self, settings: Settings | None = None):
        """Initialize the model.

        Args:
            settings: Application settings. If None, uses default settings.
        """
        self.settings = settings or get_settings()
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> None:
        """Load the model and processor."""
        if self._loaded:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading model: {self.settings.effective_model_name}")

        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[self.settings.torch_dtype]

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.settings.effective_model_name,
            cache_dir=self.settings.model_cache_dir,
            trust_remote_code=True,
        )

        # Select model class based on model type
        if self.settings.is_qwen3_vl:
            logger.info("Using Qwen3-VL model class")
            model_class = Qwen3VLForConditionalGeneration
        else:
            logger.info("Using Qwen2.5-VL model class")
            model_class = Qwen2_5_VLForConditionalGeneration

        # Load model
        self.model = model_class.from_pretrained(
            self.settings.effective_model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            cache_dir=self.settings.model_cache_dir,
            trust_remote_code=True,
        )

        self._loaded = True
        logger.info("Model loaded successfully")

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False
        torch.cuda.empty_cache()
        logger.info("Model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate response for an image and prompt.

        Args:
            image: PIL Image to process.
            prompt: Text prompt for the model.
            max_new_tokens: Maximum tokens to generate. Uses settings default if None.

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        max_tokens = max_new_tokens or self.settings.max_new_tokens

        # Build conversation format for Qwen3-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process inputs
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        # Decode output (remove input tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text

    def generate_batch(
        self,
        images: list[Image.Image],
        prompts: list[str],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Generate responses for multiple images.

        Args:
            images: List of PIL Images.
            prompts: List of prompts (one per image or single prompt for all).
            max_new_tokens: Maximum tokens to generate.

        Returns:
            List of generated text responses.
        """
        if len(prompts) == 1:
            prompts = prompts * len(images)

        if len(images) != len(prompts):
            raise ValueError("Number of images must match number of prompts")

        results = []
        for image, prompt in zip(images, prompts):
            result = self.generate(image, prompt, max_new_tokens)
            results.append(result)

        return results


# Global model instance
_model_instance: QwenVLModel | None = None


def get_model() -> QwenVLModel:
    """Get the global model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = QwenVLModel()
    return _model_instance
