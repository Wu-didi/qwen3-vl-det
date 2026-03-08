from typing import Dict, List

import torch


class VLDataCollator:
    """Qwen-VL 训练用数据整理器。

    目标：把 Dataset 返回的多个样本拼成一个 batch，同时保证：
    - 文本相关张量（input_ids/attention_mask/labels）按最长序列右侧补齐。
    - 图像相关张量（pixel_values/image_grid_thw）按 Qwen-VL 预期方式拼接。
    """

    def __init__(self, processor, pad_token_id: int = None):
        # 若未显式传入 pad_token_id，则用 tokenizer 的默认值。
        self.processor = processor
        self.pad_token_id = pad_token_id or processor.tokenizer.pad_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """将样本列表拼接为 batch 字典。"""
        batch = {}

        # 遍历样本中的每个字段，逐字段做拼接。
        for key in features[0].keys():
            values = [f[key] for f in features]

            # 非 Tensor 字段原样保留为列表。
            if not isinstance(values[0], torch.Tensor):
                batch[key] = values
                continue

            # ----------------------------------------------------------
            # 1) 文本字段：按最长长度 padding
            # ----------------------------------------------------------
            if key in ["input_ids", "attention_mask", "labels"]:
                max_len = max(v.shape[0] for v in values)
                padded_values = []
                for v in values:
                    pad_len = max_len - v.shape[0]
                    if pad_len > 0:
                        # 不同字段使用不同 padding 值。
                        if key == "input_ids":
                            pad_value = self.pad_token_id
                        elif key == "attention_mask":
                            pad_value = 0
                        else:
                            # labels 的 padding 必须是 -100，才能被 loss 忽略。
                            pad_value = -100
                        padding = torch.full((pad_len,), pad_value, dtype=v.dtype)
                        v = torch.cat([v, padding])
                    padded_values.append(v)
                batch[key] = torch.stack(padded_values)

            # ----------------------------------------------------------
            # 2) 视觉 patch：直接拼接
            # ----------------------------------------------------------
            elif key == "pixel_values":
                # Qwen-VL 通常按 patch 维度拼接，后续通过 image_grid_thw 还原分组。
                batch[key] = torch.cat(values, dim=0)

            # ----------------------------------------------------------
            # 3) 图像网格信息：统一成 (N, 3) 后拼接
            # ----------------------------------------------------------
            elif key == "image_grid_thw":
                processed_values = []
                for v in values:
                    # 某些样本可能是 (3,)；先扩成 (1, 3) 再 cat。
                    if v.dim() == 1:
                        v = v.unsqueeze(0)
                    processed_values.append(v)
                batch[key] = torch.cat(processed_values, dim=0)

            # ----------------------------------------------------------
            # 4) 其他张量：优先 stack，失败则保留 list
            # ----------------------------------------------------------
            else:
                try:
                    batch[key] = torch.stack(values)
                except RuntimeError:
                    batch[key] = values

        return batch
