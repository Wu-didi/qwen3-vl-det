#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Directly convert Stage-1 CVAT XML annotations to minimal GRPO-ready jsonl files.

Output fields per sample:
- image
- prompt
- ground_truth
- difficulty

Usage:
    # 单个 XML
    python convert_cvat_to_grpo.py \
        --xml /path/to/annotations_lcx.xml \
        --output_dir /path/to/out_grpo

    # 合并目录下所有 annotations_*.xml（推荐）
    python convert_cvat_to_grpo.py \
        --xml_dir data/hefei_last_dataset/hefei_stage1_cvat_data \
        --output_dir data/hefei_last_dataset/rft_output \
        --image_root data/hefei_last_dataset/hefei_stage1_cvat_data
"""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional


DEVICE_TYPE_MAP = {
    "traffic-signal-system": "traffic_signal",
    "traffic-guidance-system": "guidance_screen",
    "restricted-elevated": "height_limit_bar",
    "cabinet": "cabinet",
    "backpack-box": "backpack_box",
}

SIGNAL_SUBTYPE_MAP = {
    "Vehicle-traffic-signal-lights": "vehicle_signal",
    "Pedestrian-traffic-signal-lights": "pedestrian_signal",
}

STATE_MAP_COMMON = {
    "normal": "normal",
    "abnormal": "abnormal",
}

STATE_MAP_SIGNAL = {
    "normal": "normal",
    "all-off": "all-off",
    "all-on": "all-on",
    "abnormal": "abnormal",
}

STATE_MAP_GUIDANCE = {
    "normal": "normal",
    "black-screen": "black-screen",
    "abnormal": "abnormal",
}

TARGET_LABELS = set(DEVICE_TYPE_MAP.keys())
IGNORE_LABELS = {"ignore"}
DISTRACTOR_LABELS = {
    "off-site",
    "off-site-other",
    "Gun-type-Camera",
    "Dome-Camera",
    "Flashlight",
    "b-Flashlight",
}

PROMPT = (
    "Detect all target traffic devices in the image and output a JSON object with "
    "\"detections\" as a list. Each detection must contain: device_type, sub_type, "
    "state, and bbox_1000. The bbox_1000 field must be in [x1, y1, x2, y2] format "
    "with coordinates normalized to 0-1000. Target device types include traffic "
    "signals, guidance screens, height limit bars, cabinets, and backpack boxes. "
    "If no target device is present, output {\"detections\":[]}."
)


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--xml", type=str, help="Path to a single CVAT XML file")
    group.add_argument("--xml_dir", type=str, help="Directory containing annotations_*.xml files (all merged)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--image_root", type=str, default=None, help="Root prefix to prepend to image paths")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    return parser.parse_args()


def extract_stratum(image_name: str) -> str:
    """从图片路径提取 category/state 分层 key，用于分层划分。"""
    parts = Path(image_name).parts
    start = 0
    for i, p in enumerate(parts):
        if p == "hefei-dataset":
            start = i + 1
    if start < len(parts) and parts[start] == "hefei-dataset":
        start += 1
    stratum_parts = parts[start:-1]
    return "/".join(stratum_parts) if stratum_parts else "unknown"


def stratified_split_all(
    image_keys: List[str],
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, str]:
    """按分层精确分配 train/val/test，保证每个 stratum 都按比例划分。"""
    strata: Dict[str, List[str]] = defaultdict(list)
    for key in image_keys:
        strata[extract_stratum(key)].append(key)

    assignments: Dict[str, str] = {}
    for stratum, keys in sorted(strata.items()):
        keys_sorted = sorted(keys)
        n = len(keys_sorted)
        n_train = max(1, int(round(n * train_ratio))) if n > 1 else 1
        n_val = max(0, int(round(n * val_ratio))) if n > 2 else 0
        for i, key in enumerate(keys_sorted):
            if i < n_train:
                assignments[key] = "train"
            elif i < n_train + n_val:
                assignments[key] = "val"
            else:
                assignments[key] = "test"
    return assignments


def read_attr(box_elem, attr_name):
    for a in box_elem.findall("attribute"):
        if a.attrib.get("name") == attr_name:
            return (a.text or "").strip()
    return None


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def to_bbox_abs(box_elem):
    x1 = float(box_elem.attrib["xtl"])
    y1 = float(box_elem.attrib["ytl"])
    x2 = float(box_elem.attrib["xbr"])
    y2 = float(box_elem.attrib["ybr"])
    return [x1, y1, x2, y2]


def abs_to_1000(bbox, width, height):
    x1, y1, x2, y2 = bbox
    out = [
        int(round(clip(x1 / width * 1000.0, 0.0, 1000.0))),
        int(round(clip(y1 / height * 1000.0, 0.0, 1000.0))),
        int(round(clip(x2 / width * 1000.0, 0.0, 1000.0))),
        int(round(clip(y2 / height * 1000.0, 0.0, 1000.0))),
    ]
    out[0], out[2] = min(out[0], out[2]), max(out[0], out[2])
    out[1], out[3] = min(out[1], out[3]), max(out[1], out[3])
    return out


def bbox_stats(bbox):
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    short_side = min(w, h)
    return w, h, area, short_side


def map_target_box(label, box_elem):
    device_type = DEVICE_TYPE_MAP[label]
    sub_type = None
    state = None

    if label == "traffic-signal-system":
        raw_subtype = read_attr(box_elem, "class")
        raw_state = read_attr(box_elem, "state")
        sub_type = SIGNAL_SUBTYPE_MAP.get(raw_subtype, None)
        state = STATE_MAP_SIGNAL.get(raw_state, "abnormal")
    elif label == "traffic-guidance-system":
        raw_state = read_attr(box_elem, "state")
        state = STATE_MAP_GUIDANCE.get(raw_state, "abnormal")
    else:
        raw_state = read_attr(box_elem, "state")
        state = STATE_MAP_COMMON.get(raw_state, "abnormal")

    return {
        "device_type": device_type,
        "sub_type": sub_type,
        "state": state,
    }


def build_sample(image_elem, image_root: Optional[str] = None):
    image_name = image_elem.attrib["name"]
    image_path = str(Path(image_root) / image_name) if image_root else image_name
    width = int(float(image_elem.attrib["width"]))
    height = int(float(image_elem.attrib["height"]))

    detections = []
    num_ignore = 0
    num_distractor = 0
    areas = []
    short_sides = []

    for box in image_elem.findall("box"):
        label = box.attrib.get("label")
        bbox_abs = to_bbox_abs(box)

        if label in TARGET_LABELS:
            mapped = map_target_box(label, box)
            bbox_1000 = abs_to_1000(bbox_abs, width, height)
            detections.append({
                "device_type": mapped["device_type"],
                "sub_type": mapped["sub_type"],
                "state": mapped["state"],
                "bbox_1000": bbox_1000,
            })
            _, _, area, short_side = bbox_stats(bbox_abs)
            areas.append(area)
            short_sides.append(short_side)

        elif label in IGNORE_LABELS:
            num_ignore += 1
        elif label in DISTRACTOR_LABELS:
            num_distractor += 1

    detections.sort(key=lambda d: (
        d["device_type"],
        d["bbox_1000"][0], d["bbox_1000"][1], d["bbox_1000"][2], d["bbox_1000"][3]
    ))

    tiny_thr_short = 8.0
    tiny_thr_area = 64.0
    num_targets = len(detections)
    num_abnormal = sum(1 for d in detections if d["state"] != "normal")
    has_tiny_object = any((s < tiny_thr_short) or (a < tiny_thr_area) for s, a in zip(short_sides, areas))

    return {
        "image": image_path,
        "prompt": PROMPT,
        "ground_truth": {
            "detections": detections
        },
        "difficulty": {
            "num_targets": num_targets,
            "num_abnormal": num_abnormal,
            "num_device_types": len(set(d["device_type"] for d in detections)),
            "min_box_area": (min(areas) if areas else None),
            "min_short_side": (min(short_sides) if short_sides else None),
            "has_tiny_object": has_tiny_object,
            "has_ignore_region": num_ignore > 0,
            "has_distractor": num_distractor > 0,
            "empty_scene": num_targets == 0,
        }
    }


def schema():
    return {
        "prompt_format": {
            "top_level_key": "detections",
            "bbox_format": "bbox_1000",
            "bbox_order": "[x1, y1, x2, y2]",
            "bbox_range": [0, 1000],
        },
        "device_type": [
            "traffic_signal",
            "guidance_screen",
            "height_limit_bar",
            "cabinet",
            "backpack_box",
        ],
        "sub_type": [
            "vehicle_signal",
            "pedestrian_signal",
            None,
        ],
        "state_by_type": {
            "traffic_signal": ["normal", "all-off", "all-on", "abnormal"],
            "guidance_screen": ["normal", "black-screen", "abnormal"],
            "height_limit_bar": ["normal", "abnormal"],
            "cabinet": ["normal", "abnormal"],
            "backpack_box": ["normal", "abnormal"],
        },
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有 XML 文件
    if args.xml_dir:
        xml_paths = sorted(Path(args.xml_dir).glob("annotations_*.xml"))
        if not xml_paths:
            raise FileNotFoundError(f"No annotations_*.xml found in {args.xml_dir}")
        print(f"Found {len(xml_paths)} XML files: {[p.name for p in xml_paths]}")
    else:
        xml_paths = [Path(args.xml)]

    # --- Pass 1: 收集所有图片 key，计算分层划分 ---
    all_entries = []  # (xml_stem, image_elem)
    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        xml_stem = xml_path.stem
        for image_elem in tree.getroot().findall(".//image"):
            all_entries.append((xml_stem, image_elem))

    all_keys = [elem.attrib.get("name", "") for _, elem in all_entries]
    split_map = stratified_split_all(all_keys, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    # --- Pass 2: 构建样本并分配 split ---
    split_rows: Dict[str, list] = {"train": [], "val": [], "test": []}
    counters: Dict[str, Counter] = {
        "images": Counter(),
        "empty_scenes": Counter(),
        "target_objects": Counter(),
        "state_counts": Counter(),
    }

    for _, image_elem in all_entries:
        image_key = image_elem.attrib.get("name", "")
        subset = image_elem.attrib.get("subset", "").lower()
        split = subset if subset in {"train", "val", "test"} else split_map.get(image_key, "train")

        row = build_sample(image_elem, image_root=args.image_root)
        split_rows[split].append(row)

        counters["images"][split] += 1
        if row["difficulty"]["empty_scene"]:
            counters["empty_scenes"][split] += 1
        for det in row["ground_truth"]["detections"]:
            counters["target_objects"][det["device_type"]] += 1
            counters["state_counts"][det["state"]] += 1

    for split in ("train", "val", "test"):
        write_jsonl(output_dir / f"{split}.jsonl", split_rows[split])

    with open(output_dir / "schema.json", "w", encoding="utf-8") as f:
        json.dump(schema(), f, ensure_ascii=False, indent=2)

    summary = {
        "xml_files": [str(p) for p in xml_paths],
        "num_images": len(all_entries),
        "splits": {k: len(v) for k, v in split_rows.items()},
        "empty_scenes": dict(counters["empty_scenes"]),
        "target_object_counts": dict(counters["target_objects"]),
        "state_counts": dict(counters["state_counts"]),
        "split_rule": "stratified by category×state (directory path)",
        "prompt": PROMPT,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

