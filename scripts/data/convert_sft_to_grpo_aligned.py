#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert SFT JSONL files into GRPO-style JSONL while preserving the SFT prompt text.

The output keeps the structured fields preferred by GRPO reward functions:
    - image
    - prompt
    - ground_truth
    - difficulty

Unlike the original CVAT -> GRPO conversion, this script reuses the exact prompt
text from the SFT samples so GRPO continues from the same instruction format.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


NORMALIZED_TINY_SHORT_SIDE = 16
NORMALIZED_TINY_AREA = 16 * 16


def strip_image_tokens(text: str) -> str:
    """Remove SFT-only image placeholders and surrounding whitespace."""
    return text.replace("<image>\n", "").replace("<image>", "").strip()


def extract_last_prompt(conversations: Iterable[Dict[str, Any]]) -> str:
    """Return the final user prompt from an SFT conversation."""
    prompt = ""
    for item in conversations:
        role = item.get("from", "")
        if role in {"user", "human"}:
            prompt = strip_image_tokens(str(item.get("value", "")))
    return prompt


def parse_ground_truth_json(text: str) -> Dict[str, Any]:
    """Parse the assistant JSON response into structured ground truth."""
    clean = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    try:
        payload = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            raise ValueError("assistant message does not contain JSON ground truth")
        payload = json.loads(match.group())

    if not isinstance(payload, dict) or not isinstance(payload.get("detections"), list):
        raise ValueError("assistant message JSON must contain a detections list")
    return payload


def build_difficulty(ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the GRPO difficulty dictionary from normalized detections.

    The original CVAT -> GRPO pipeline computed some values in absolute pixels.
    Here we recompute a lightweight equivalent from bbox_1000 so the aligned
    dataset stays self-contained.
    """
    detections = ground_truth.get("detections", [])
    areas: List[int] = []
    short_sides: List[int] = []

    for det in detections:
        bbox = det.get("bbox_1000", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        areas.append(width * height)
        short_sides.append(min(width, height))

    num_targets = len(detections)
    num_abnormal = sum(1 for det in detections if det.get("state") != "normal")

    return {
        "num_targets": num_targets,
        "num_abnormal": num_abnormal,
        "num_device_types": len({det.get("device_type") for det in detections}),
        "min_box_area": min(areas) if areas else None,
        "min_short_side": min(short_sides) if short_sides else None,
        "has_tiny_object": any(
            side <= NORMALIZED_TINY_SHORT_SIDE or area <= NORMALIZED_TINY_AREA
            for side, area in zip(short_sides, areas, strict=True)
        ),
        # These flags are not recoverable from SFT JSONL alone; keep them explicit.
        "has_ignore_region": False,
        "has_distractor": False,
        "empty_scene": num_targets == 0,
    }


def convert_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one SFT sample into aligned GRPO format."""
    conversations = sample.get("conversations", [])
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("sample is missing conversations")

    prompt = extract_last_prompt(conversations)
    if not prompt:
        raise ValueError("sample is missing a usable user prompt")

    assistant_text = ""
    for item in conversations:
        role = item.get("from", "")
        if role in {"assistant", "gpt"}:
            assistant_text = str(item.get("value", ""))

    if not assistant_text:
        raise ValueError("sample is missing an assistant response")

    ground_truth = parse_ground_truth_json(assistant_text)
    return {
        "image": sample["image"],
        "prompt": prompt,
        "ground_truth": ground_truth,
        "difficulty": build_difficulty(ground_truth),
    }


def convert_file(input_path: Path, output_path: Path) -> Dict[str, int]:
    """Convert a single split file and preserve sample order."""
    count = 0
    empty = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            sample = json.loads(line)
            converted = convert_sample(sample)
            dst.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1
            if converted["difficulty"]["empty_scene"]:
                empty += 1

    return {"samples": count, "empty_scenes": empty}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SFT JSONL to prompt-aligned GRPO JSONL")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/hefei_last_dataset/sft_output",
        help="Directory containing SFT train/val/test JSONL files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/hefei_last_dataset/rft_output_aligned",
        help="Directory to write aligned GRPO JSONL files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    summary: Dict[str, Any] = {
        "source_dir": str(input_dir),
        "output_dir": str(output_dir),
        "note": "GRPO-format data with prompts copied from SFT samples",
        "splits": {},
    }

    for split in ("train", "val", "test"):
        input_path = input_dir / f"{split}.jsonl"
        output_path = output_dir / f"{split}.jsonl"
        stats = convert_file(input_path, output_path)
        summary["splits"][split] = stats

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
