#!/usr/bin/env python3
"""
面向 Qwen-VL 的自定义 GRPO Trainer。

这个实现基于 TRL 的 `GRPOTrainer`，核心目标是补齐多模态训练链路：
1. 生成阶段把图像一并送入模型；
2. 计算 log-prob 时正确处理 `pixel_values / image_grid_thw`；
3. 与 TRL 0.26+ 输出字段格式兼容。

参考实现：
https://github.com/2U1/Qwen-VL-Series-Finetune
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
    """恒等 collator：不做任何处理，原样透传。"""
    return features


class QwenVLGRPOTrainer(GRPOTrainer):
    """Qwen-VL 多模态 GRPO 训练器。

    与原生 TRL GRPOTrainer 的主要差异：
    - 支持图像输入参与采样与前向计算；
    - 兼容 Qwen-VL 的 patch/grid 组织方式；
    - 按 GRPO 需求返回完整字段（advantages、old/ref logps 等）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 覆盖为恒等 collator，避免默认 collator 破坏图像字段结构。
        self.data_collator = _identity_collator

    def _set_signature_columns_if_needed(self):
        """声明 Trainer 可接受的输入字段（含多模态字段）。"""
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt", "assistant", "image", "images", "image_path",
                "max_image_size", "video", "videos"
            ]

    def _generate_single_turn(self, prompts: list):
        """单轮生成：在文本 prompt 的基础上附带图像输入。"""
        from trl.models.utils import unwrap_model_for_generation

        device = self.accelerator.device

        # 从当前 batch 上下文中取出图像（由上层函数预先写入）。
        images = getattr(self, '_current_images', None)

        # 构建 processor 输入参数。
        # 注意：生成时需要左侧 padding，保证右端 token 对齐。
        processor_kwargs = {
            "text": prompts,
            "return_tensors": "pt",
            "padding": True,
            "padding_side": "left",
            "add_special_tokens": False,
        }

        if images is not None:
            processor_kwargs["images"] = images

        # 编码输入并迁移到正确设备。
        generate_inputs = self.processing_class(**processor_kwargs)
        generate_inputs = Trainer._prepare_inputs(self, generate_inputs)

        # 进入“仅生成、不计算梯度”上下文。
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

        # 按 prompt 长度切分 completion。
        prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # 仅保留每条 completion 的首个 EOS 之前内容。
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # 转成 python list，兼容 TRL 后续流水线。
        prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]
        completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]

        # 返回格式需匹配 TRL 的 _generate_single_turn 协议。
        # (prompt_ids, completion_ids, sampling_logprobs, extra_fields)
        return prompt_ids, completion_ids, None, {}

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """生成 completion 并计算奖励/优势，完整支持多模态输入。"""
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # 兼容 dict 与 list[dict] 两种入参格式。
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

        # 从样本里提取图像字段，统一成 list-of-list 格式。
        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None
                      for example in inputs]
        else:
            images = None

        if images is not None and all(img_list is None or img_list == [] for img_list in images):
            images = None

        # 临时缓存图像，供 _generate_single_turn 使用。
        self._current_images = images

        # TRL 0.26+ 返回 7 元组。
        gen_output = self._generate(prompts)
        prompt_ids_list = gen_output[0]
        completion_ids_list = gen_output[1]
        sampling_logprobs = gen_output[5] if len(gen_output) > 5 else None
        extra_fields = gen_output[6] if len(gen_output) > 6 else {}

        # 生成完成后立刻清理缓存，避免跨 batch 串扰。
        self._current_images = None

        # list[token_ids] -> padded tensor
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        # 可选：把“被截断而未到 EOS”的 completion mask 掉。
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # 拼接 prompt + completion，供 log-prob 计算。
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        # 每个样本包含的图像数量（用于切分 patch 行）。
        num_images = [len(img_list) for img_list in images] if images is not None else None

        # 准备多模态前向额外字段。
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
            # 在部分调度配置下，需要显式记录 old_logps。
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

            # 计算参考策略 log-prob（用于 KL 项）。
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
                    # 无显式 ref_model 时，临时关闭 adapter 作为参考策略。
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

        # 解码文本用于日志与奖励函数输入。
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text, strict=True):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # 逐奖励函数计算得分。
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # 按权重合成总奖励。
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # 组内中心化优势：A = r - mean(r_group)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        # 奖励缩放（group / batch / none）。
        if self.scale_rewards in ["group", "none"]:
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        else:
            std_rewards = torch.ones_like(rewards)

        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        # 只保留当前进程负责的切片。
        local_n = len(prompts) * self.num_generations
        process_slice = slice(
            self.accelerator.process_index * local_n,
            (self.accelerator.process_index + 1) * local_n,
        )
        advantages = advantages[process_slice]

        # 记录指标。
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)

        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())

        # 记录文本日志（用于可视化/诊断）。
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))

        # 组装 TRL 训练所需输出字典。
        # 注意：num_items_in_batch 必须是 0 维 tensor，才能通过 TRL 的序列打乱逻辑。
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": torch.tensor(len(prompts) * self.num_generations, device=device),
        }

        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps

        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps

        # 透传多模态字段，供后续 loss 前向使用。
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
        """计算每 token 的 log-prob（以及可选熵），支持多模态切片。"""
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

            # 多图像场景下，pixel_values 是按 patch 行展平的，需要按样本切片。
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

            # 透传其他可选视觉字段。
            for key in ["pixel_attention_mask", "image_sizes"]:
                if key in kwargs and kwargs[key] is not None:
                    model_inputs[key] = kwargs[key][start : start + batch_size]

            # 某些模型支持 logits_to_keep（可减少显存开销）。
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
