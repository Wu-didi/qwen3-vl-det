#!/usr/bin/env python3
"""
Custom GRPO Trainer for Qwen-VL models with proper multimodal support.

Based on: https://github.com/2U1/Qwen-VL-Series-Finetune
"""

import re
import torch
from typing import Any, Dict, List, Optional
from contextlib import nullcontext

from transformers.trainer import Trainer
from trl import GRPOTrainer
from trl.data_utils import is_conversational
from trl.trainer.utils import pad, selective_log_softmax
from accelerate.utils import gather_object, is_peft_model


def _identity_collator(features):
    """Identity collator that passes data through unchanged."""
    return features


class QwenVLGRPOTrainer(GRPOTrainer):
    """
    Custom GRPO Trainer for Qwen-VL multimodal models.

    Extends TRL's GRPOTrainer with proper image handling for VLM generation
    and log probability computation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override data_collator to prevent processing
        self.data_collator = _identity_collator

    def _set_signature_columns_if_needed(self):
        """Set signature columns for GRPO with multimodal inputs."""
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt", "assistant", "image", "images", "image_path",
                "max_image_size", "video", "videos"
            ]

    def _generate_single_turn(self, prompts: list):
        """Override to include images in generation for multimodal models."""
        from trl.models.utils import unwrap_model_for_generation

        device = self.accelerator.device

        # Get stored images
        images = getattr(self, '_current_images', None)

        # Build processor kwargs - don't truncate for multimodal inputs
        processor_kwargs = {
            "text": prompts,
            "return_tensors": "pt",
            "padding": True,
            "padding_side": "left",
            "add_special_tokens": False,
        }

        # Add images if available
        if images is not None:
            processor_kwargs["images"] = images

        # Process inputs
        generate_inputs = self.processing_class(**processor_kwargs)
        generate_inputs = Trainer._prepare_inputs(self, generate_inputs)

        # Generate completions
        with (
            unwrap_model_for_generation(
                self.model_wrapped, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model,
            torch.no_grad(),
        ):
            prompt_completion_ids = unwrapped_model.generate(
                **generate_inputs,
                generation_config=self.generation_config,
                disable_compile=True
            )

        # Extract prompt and completion ids
        prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]
        completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]

        # Return format for _generate_single_turn (4 values):
        # prompt_ids, completion_ids, logprobs, extra_fields
        return prompt_ids, completion_ids, None, {}

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """
        Override to handle multimodal inputs properly.
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Handle different input formats
        if isinstance(inputs, dict):
            if "prompt" in inputs:
                bsz = len(inputs["prompt"])
                inputs = [
                    {k: (v[i] if v is not None else None) for k, v in inputs.items()}
                    for i in range(bsz)
                ]
            else:
                raise ValueError("Expected inputs with 'prompt' key")
        elif not isinstance(inputs, list):
            raise TypeError(f"Expected list[dict] or dict, got {type(inputs).__name__}")

        prompts = [x["prompt"] for x in inputs]

        # Extract images
        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None
                      for example in inputs]
        else:
            images = None

        if images is not None and all(img_list is None or img_list == [] for img_list in images):
            images = None

        # Store images for use in _generate_single_turn
        self._current_images = images

        # Generate completions - TRL 0.26+ returns 7 values
        gen_output = self._generate(prompts)
        # Unpack: prompt_ids, completion_ids, tool_mask, completions, total_tokens, logprobs, extra_fields
        prompt_ids_list = gen_output[0]
        completion_ids_list = gen_output[1]
        # tool_mask = gen_output[2]  # Not used for VLM
        # completions_raw = gen_output[3]  # Not used, we decode ourselves
        # total_tokens = gen_output[4]  # Not used
        sampling_logprobs = gen_output[5] if len(gen_output) > 5 else None
        extra_fields = gen_output[6] if len(gen_output) > 6 else {}

        # Clear stored images
        self._current_images = None

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        # Mask truncated completions
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt and completion
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs
        forward_kwargs = {}
        if images is not None:
            processor_kwargs = dict(
                text=prompts,
                padding=True,
                return_tensors="pt",
            )
            processor_kwargs["images"] = images

            prompt_inputs = self.processing_class(**processor_kwargs)
            prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
            forward_kwargs = {
                k: v for k, v in prompt_inputs.items()
                if k not in ["input_ids", "attention_mask"]
            }

        with torch.no_grad():
            # Check if need old_per_token_logps
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    **forward_kwargs,
                )
            else:
                old_per_token_logps = None

            # Compute reference model log probs
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            **forward_kwargs,
                        )
            else:
                ref_per_token_logps = None

        # Decode completions
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text, strict=True):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Calculate rewards
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # Apply weights and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards and advantages
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        # Scale rewards
        if self.scale_rewards in ["group", "none"]:
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        else:
            std_rewards = torch.ones_like(rewards)

        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        # Slice to local part
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log metrics
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)

        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())

        # Log texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))

        # Build output with all required keys for TRL 0.26+
        # Note: num_items_in_batch must be a 0-dim tensor to survive shuffle_sequence_dict
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": torch.tensor(len(prompts) * self.num_generations, device=device),
        }

        # Add old_per_token_logps if computed
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps

        # Add ref_per_token_logps if computed
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps

        # Add multimodal inputs
        for key in ["pixel_values", "image_grid_thw", "pixel_attention_mask", "image_sizes"]:
            if key in forward_kwargs:
                output[key] = forward_kwargs[key]

        if num_images is not None:
            output["num_images"] = num_images

        return output

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        **kwargs,
    ):
        """Compute log-probs with multimodal inputs."""
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []

        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            model_inputs = {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
            }

            # Handle image inputs
            if image_grid_thw is not None and pixel_values is not None and num_images is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([
                    torch.tensor([0], device=rows_per_sample.device),
                    rows_per_sample.cumsum(0)
                ])
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]

                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]

            # Add other kwargs
            for key in ["pixel_attention_mask", "image_sizes"]:
                if key in kwargs and kwargs[key] is not None:
                    model_inputs[key] = kwargs[key][start : start + batch_size]

            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False

            logits = model(**model_inputs).logits
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    from trl.trainer.utils import entropy_from_logits
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies
