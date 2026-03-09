import json
import logging
import os
import re
from typing import Any, Dict

from PIL import Image
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class TrafficAnomalyDataset(Dataset):
    """交通设备异常检测数据集。

    输入数据要求为 Qwen-VL 对话格式，每条样本至少包含：
    - image: 图像路径
    - conversations: 对话列表（通常 user + assistant）

    本类的核心职责：
    1. 读取图像与对话并构造成 Qwen-VL 输入。
    2. 生成 labels，并把非 assistant 回复区域 mask 成 -100。
    """

    def __init__(
        self,
        data_path: str,
        processor,
        max_length: int = 2048,
        max_image_size: int = 512,
    ):
        # 处理器负责 chat template + tokenize + 图像预处理。
        self.processor = processor
        # 文本最大 token 长度（超出截断）。
        self.max_length = max_length
        # 图像最长边限制，用于控制视觉 token 数量和显存。
        self.max_image_size = max_image_size

        # 一次性加载数据到内存，便于随机索引。
        # 同时支持 JSON array (.json) 和 JSON Lines (.jsonl) 两种格式。
        with open(data_path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                # JSON array 格式
                self.data = json.load(f)
            else:
                # JSONL 格式：每行一个 JSON 对象
                self.data = [json.loads(line) for line in f if line.strip()]

        logger.info("Loaded %d samples from %s", len(self.data), data_path)

    def __len__(self):
        """返回样本数量。"""
        return len(self.data)

    def _find_assistant_start(self, input_ids: torch.Tensor) -> int:
        """定位 assistant 回复起始 token 位置。

        这是 label mask 的关键：
        - 起始位置之前全部置为 -100（不参与 loss）
        - 起始位置及之后才参与训练

        由于不同版本 tokenizer 的行为可能不同，这里采用三级回退策略。
        """
        tokenizer = self.processor.tokenizer
        input_ids_list = input_ids.tolist() if input_ids.dim() == 1 else input_ids[0].tolist()

        # 方法 1：基于 <|im_start|> 特殊 token（优先，通常最稳）。
        try:
            im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
            if im_start_id is not None and im_start_id != tokenizer.unk_token_id:
                # 找到所有 <|im_start|> 位置，通常最后一个对应 assistant 段落起点。
                positions = [i for i, token_id in enumerate(input_ids_list) if token_id == im_start_id]

                if len(positions) >= 2:
                    last_start = positions[-1]
                    # 在后续少量 token 中寻找换行，跳过 "assistant\n" 头部。
                    for offset in range(1, min(8, len(input_ids_list) - last_start)):
                        idx = last_start + offset
                        token_text = tokenizer.decode([input_ids_list[idx]], skip_special_tokens=False)
                        if "\n" in token_text:
                            return idx + 1
                    # 如果没有找到换行，则使用经验偏移值。
                    return last_start + 3
        except Exception as exc:
            logger.debug("Method 1 failed: %s", exc)

        # 方法 2：直接匹配 "<|im_start|>assistant\n" 的 token 序列。
        try:
            assistant_prompt = "<|im_start|>assistant\n"
            assistant_ids = tokenizer.encode(assistant_prompt, add_special_tokens=False)

            for i in range(len(input_ids_list) - len(assistant_ids) + 1):
                if input_ids_list[i : i + len(assistant_ids)] == assistant_ids:
                    return i + len(assistant_ids)
        except Exception as exc:
            logger.debug("Method 2 failed: %s", exc)

        # 方法 3：解码后用正则找 "assistant\n" 文本位置，再反推 token 数。
        try:
            full_text = tokenizer.decode(input_ids_list, skip_special_tokens=False)
            assistant_positions = [m.start() for m in re.finditer(r"assistant\s*\n", full_text)]

            if assistant_positions:
                last_assistant_text_pos = assistant_positions[-1]
                prefix_text = full_text[:last_assistant_text_pos]
                prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
                return len(prefix_ids) + 3
        except Exception as exc:
            logger.debug("Method 3 failed: %s", exc)

        # 最后回退：若三种方法都失败，取序列中点作为保守估计。
        fallback_pos = int(len(input_ids_list) * 0.5)
        logger.warning(
            "Failed to find assistant start position accurately, using fallback: %d. "
            "This may affect training quality. Sequence length: %d",
            fallback_pos,
            len(input_ids_list),
        )
        return fallback_pos

    def __getitem__(self, idx) -> Dict[str, Any]:
        """按索引返回单条训练样本（已 tokenized + labels）。"""
        item = self.data[idx]

        # ------------------------------
        # 1) 基础字段校验
        # ------------------------------
        if "image" not in item:
            raise ValueError(f"Sample {idx}: missing 'image' field")
        if "conversations" not in item:
            raise ValueError(f"Sample {idx}: missing 'conversations' field")
        if len(item["conversations"]) < 2:
            raise ValueError(
                f"Sample {idx}: conversations must have at least 2 messages (user + assistant)"
            )

        image_path = item["image"]
        conversations = item["conversations"]

        # ------------------------------
        # 2) 图像读取与缩放
        # ------------------------------
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_path).convert("RGB")

            # 若图像过大，按比例缩放到 max_image_size。
            if self.max_image_size and max(image.size) > self.max_image_size:
                ratio = self.max_image_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.LANCZOS)
        except Exception as exc:
            raise RuntimeError(f"Sample {idx}: Failed to load image {image_path}: {exc}")

        # ------------------------------
        # 3) 构建 chat template 消息
        # ------------------------------
        messages = []
        # 只在第一条 user 消息注入图像，避免重复图像 token。
        first_user_msg = True

        for conv in conversations:
            # role 兼容多种标注写法。
            role = conv.get("from", "user")
            if role in ["human", "user"]:
                role = "user"
            elif role in ["gpt", "assistant"]:
                role = "assistant"
            elif role == "system":
                role = "system"
            else:
                logger.warning("Sample %d: unknown role '%s', treating as user", idx, role)
                role = "user"

            # 文本里若有 <image> 占位符，去掉，图像由 content.image 单独提供。
            text = conv.get("value", "")
            text = text.replace("<image>\n", "").replace("<image>", "")

            if role == "user" and first_user_msg:
                content = [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ]
                first_user_msg = False
            else:
                content = [{"type": "text", "text": text}]

            messages.append({"role": role, "content": content})

        # ------------------------------
        # 4) 应用 chat template + tokenizer
        # ------------------------------
        # `apply_chat_template` 的作用：将之前构建的 `messages` 列表格式化为模型期望的对话文本模板。
        # - 插入必要的对话/role 标记（如 <|im_start|>/assistant 等），以匹配训练时的输入格式；
        # - 将图像与文本在模板中正确占位（实际图像通过 processor 的 images 参数传入）；
        # - 这里指定 `tokenize=False` 表示只返回拼接好的字符串，由后续的 processor 调用负责分词和张量化；
        # - `add_generation_prompt=False` 禁止在末尾追加生成提示（训练时通常不需要额外的生成提示）。
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # processor 返回 batch 维，单样本下挤掉第 0 维。
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # ------------------------------
        # 5) 构建 labels 并做 mask
        # ------------------------------
        labels = inputs["input_ids"].clone()

        # padding token 不参与损失。
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        # 只训练 assistant 回复，前缀（system/user）全部 mask。
        assistant_start = self._find_assistant_start(inputs["input_ids"])
        labels[:assistant_start] = -100

        inputs["labels"] = labels
        return inputs
