#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert Stage-1 CVAT XML annotations to SFT JSONL files.

Output format
-------------
Each sample is a single JSON object in JSONL format:
{
  "id": "stage1_000001",
  "image": "relative/or/absolute/path.jpg",
  "conversations": [
    {"from": "user", "value": "<image> ..."},
    {"from": "assistant", "value": "{\"detections\": [...]}"}
  ],
  "meta": {... optional metadata ...}
}

Design choices
--------------
- Gold-only: only uses CVAT human annotations.
- Structured assistant output: assistant always emits a JSON string.
- Empty-scene support: images with no valid target objects become {"detections": []}.
- Deterministic split: split by md5(image_key) for reproducibility.
- Tiny-box filtering is supported but disabled by default.
- Detection order is fixed for more stable SFT supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET


PROMPT_STAGE1 = (
    "<image>\n"
    "Detect traffic-related devices in the image and output their device type, state, and location. "
    "The target device categories are: traffic_signal, guidance_screen, height_limit_bar, cabinet, and backpack_box.\n\n"
    "Output JSON only, using the following format: "
    '{"detections": [{"device_type": "...", "sub_type": "...", "state": "...", "bbox_1000": [x1, y1, x2, y2]}]}.\n'
    "Rules:\n"
    "1. bbox_1000 must be the normalized [x1, y1, x2, y2] coordinates scaled to 0-1000.\n"
    "2. device_type must use one of these fixed labels: traffic_signal, guidance_screen, height_limit_bar, cabinet, backpack_box.\n"
    "3. sub_type should be null for non-signal devices. For traffic signals, use vehicle_signal or pedestrian_signal.\n"
    "4. state must use the closed-set label defined for that device.\n"
    "5. If no target device is present, output {\"detections\": []}.\n"
    "6. Output JSON only. Do not include any extra text."
)

STAGE1_TARGET_LABELS = {
    "traffic-signal-system",
    "traffic-guidance-system",
    "restricted-elevated",
    "cabinet",
    "backpack-box",
}


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

STAGE1_IGNORE_LABELS = {
    "ignore",
    "off-site",
    "Gun-type-Camera",
    "Dome-Camera",
    "Flashlight",
    "off-site-other",
    "b-Flashlight",
}

DEVICE_ORDER = {
    "traffic_signal": 0,
    "guidance_screen": 1,
    "height_limit_bar": 2,
    "cabinet": 3,
    "backpack_box": 4,
}


@dataclass
class Box:
    label: str
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    occluded: int
    source: str
    attrs: Dict[str, str]


def safe_int(x: str, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def safe_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def normalize_box_to_1000(box_xyxy_abs: Tuple[float, float, float, float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = box_xyxy_abs
    x1 = max(0.0, min(x1, width))
    x2 = max(0.0, min(x2, width))
    y1 = max(0.0, min(y1, height))
    y2 = max(0.0, min(y2, height))
    if width <= 0 or height <= 0:
        return [0, 0, 0, 0]
    return [
        int(round(x1 / width * 1000)),
        int(round(y1 / height * 1000)),
        int(round(x2 / width * 1000)),
        int(round(y2 / height * 1000)),
    ]


def parse_box(elem: ET.Element) -> Box:
    attrs: Dict[str, str] = {}
    for attr in elem.findall("attribute"):
        name = attr.attrib.get("name", "").strip()
        value = (attr.text or "").strip()
        if name:
            attrs[name] = value
        elif value:
            attrs["value"] = value

    return Box(
        label=elem.attrib["label"],
        xtl=safe_float(elem.attrib.get("xtl", "0")),
        ytl=safe_float(elem.attrib.get("ytl", "0")),
        xbr=safe_float(elem.attrib.get("xbr", "0")),
        ybr=safe_float(elem.attrib.get("ybr", "0")),
        occluded=safe_int(elem.attrib.get("occluded", "0")),
        source=elem.attrib.get("source", "unknown"),
        attrs=attrs,
    )


def box_is_tiny(box: Box, min_short_side: float = 0.0, min_area: float = 0.0) -> bool:
    w = max(0.0, box.xbr - box.xtl)
    h = max(0.0, box.ybr - box.ytl)
    if min_short_side > 0 and min(w, h) < min_short_side:
        return True
    if min_area > 0 and (w * h) < min_area:
        return True
    return False


def deterministic_split(image_key: str, train_ratio: float, val_ratio: float) -> str:
    assert 0 < train_ratio < 1
    assert 0 <= val_ratio < 1
    assert train_ratio + val_ratio < 1
    h = hashlib.md5(image_key.encode("utf-8")).hexdigest()
    score = int(h[:8], 16) / 0xFFFFFFFF
    if score < train_ratio:
        return "train"
    if score < train_ratio + val_ratio:
        return "val"
    return "test"


def extract_stratum(image_name: str) -> str:
    """Extract category+state stratum from image path for stratified splitting.

    Image paths look like:
      hefei-dataset/hefei-dataset/<category>/[sub_type/]<state>/<filename>
    We use the parent directory of the image file as the stratum key,
    which encodes both category and state (e.g. "backpack-box/bad",
    "traffic-signal-system/Pedestrian-traffic-signal-lights/bad").
    """
    parts = Path(image_name).parts
    # Drop leading 'hefei-dataset/hefei-dataset' prefix if present
    start = 0
    for i, p in enumerate(parts):
        if p == "hefei-dataset":
            start = i + 1
    # Skip the second 'hefei-dataset' if present
    if start < len(parts) and parts[start] == "hefei-dataset":
        start += 1
    # Everything from start to the second-to-last part is the stratum
    stratum_parts = parts[start:-1]
    return "/".join(stratum_parts) if stratum_parts else "unknown"


def stratified_split_all(
    image_keys: List[str],
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, str]:
    """Assign train/val/test to all images with per-stratum stratification.

    Within each stratum (category × state), images are sorted by name and
    then sliced exactly at train_ratio / val_ratio boundaries so that every
    stratum contributes proportionally to all three splits.
    """
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


def maybe_join_image_path(image_name: str, image_root: Optional[str]) -> str:
    if not image_root:
        return image_name
    return str(Path(image_root) / image_name)


def json_dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def detection_sort_key(det: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    bbox = det["bbox_1000"]
    x1, y1, x2, y2 = bbox
    return (
        DEVICE_ORDER.get(det["device_type"], 999),
        x1,
        y1,
        x2,
        y2,
    )


def build_stage1_sample(
    image_elem: ET.Element,
    image_root: Optional[str],
    keep_meta: bool,
    min_short_side: float,
    min_area: float,
) -> Dict[str, Any]:
    image_name = image_elem.attrib["name"]
    width = safe_int(image_elem.attrib["width"])
    height = safe_int(image_elem.attrib["height"])

    detections: List[Dict[str, Any]] = []
    ignored = 0

    for box_elem in image_elem.findall("box"):
        box = parse_box(box_elem)
        if box.label in STAGE1_IGNORE_LABELS:
            ignored += 1
            continue
        if box.label not in STAGE1_TARGET_LABELS:
            continue
        if box_is_tiny(box, min_short_side=min_short_side, min_area=min_area):
            continue

        det: Dict[str, Any] = {
            "device_type": DEVICE_TYPE_MAP[box.label],
            "sub_type": None,
            "state": box.attrs.get("state", "normal"),
            "bbox_1000": normalize_box_to_1000((box.xtl, box.ytl, box.xbr, box.ybr), width, height),
        }
        if box.label == "traffic-signal-system":
            raw_sub_type = box.attrs.get("class", None)
            det["sub_type"] = SIGNAL_SUBTYPE_MAP.get(raw_sub_type, raw_sub_type)

        detections.append(det)

    detections.sort(key=detection_sort_key)

    sample = {
        "id": f"stage1_{image_elem.attrib.get('id', '')}",
        "image": maybe_join_image_path(image_name, image_root),
        "conversations": [
            {"from": "user", "value": PROMPT_STAGE1},
            {"from": "assistant", "value": json_dumps_compact({"detections": detections})},
        ],
    }

    if keep_meta:
        sample["meta"] = {
            "width": width,
            "height": height,
            "num_detections": len(detections),
            "num_ignored_boxes": ignored,
            "image_name": image_name,
        }
    return sample


def convert(
    xml_path: str,
    output_dir: str,
    image_root: Optional[str],
    train_ratio: float,
    val_ratio: float,
    keep_meta: bool,
    min_short_side: float,
    min_area: float,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    images = root.findall(".//image")

    split_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    stats = Counter()

    for image_elem in images:
        sample = build_stage1_sample(
            image_elem=image_elem,
            image_root=image_root,
            keep_meta=keep_meta,
            min_short_side=min_short_side,
            min_area=min_area,
        )

        image_key = image_elem.attrib.get("name", sample["id"])
        subset = image_elem.attrib.get("subset")
        if subset in {"train", "val", "test"}:
            split = subset
        else:
            split = deterministic_split(image_key, train_ratio=train_ratio, val_ratio=val_ratio)

        split_to_records[split].append(sample)
        stats[f"{split}_images"] += 1

        assistant_payload = json.loads(sample["conversations"][1]["value"])
        stats[f"{split}_targets"] += len(assistant_payload["detections"])
        if len(assistant_payload["detections"]) == 0:
            stats[f"{split}_empty"] += 1

    written = {}
    for split, records in split_to_records.items():
        out_path = output_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        written[split] = str(out_path)

    summary = {
        "xml_path": xml_path,
        "num_images": len(images),
        "written_files": written,
        "stats": dict(stats),
        "split_rule": {
            "subset_priority": "use image subset attr if present and in {train,val,test}; otherwise deterministic md5 split",
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": round(1.0 - train_ratio - val_ratio, 6),
        },
        "filters": {
            "min_short_side": min_short_side,
            "min_area": min_area,
        },
        "sorting": "detections sorted by device_type priority, then bbox x1, y1, x2, y2",
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def convert_multi(
    xml_paths: List[str],
    output_dir: str,
    image_root: Optional[str],
    train_ratio: float,
    val_ratio: float,
    keep_meta: bool,
    min_short_side: float,
    min_area: float,
) -> None:
    """Merge multiple CVAT XML files and write unified JSONL splits.

    Split strategy (stratified):
      - If an image has a subset attribute in {train, val, test}, honour it directly.
      - Otherwise use stratified_split_all: images are grouped by their
        category × state stratum (extracted from the directory path), sorted
        deterministically, and sliced at the exact ratio boundaries so every
        stratum is proportionally represented in all three splits.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # --- Pass 1: collect all (xml_path, image_elem) pairs and their keys ---
    all_entries: List[Tuple[str, ET.Element, str]] = []  # (xml_stem, elem, image_key)
    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        xml_stem = Path(xml_path).stem
        for image_elem in tree.getroot().findall(".//image"):
            image_key = image_elem.attrib.get("name", "")
            all_entries.append((xml_stem, image_elem, image_key))

    # Pre-compute stratified split for images without an explicit subset attr
    free_keys = [
        key for _, elem, key in all_entries
        if elem.attrib.get("subset") not in {"train", "val", "test"}
    ]
    split_map = stratified_split_all(free_keys, train_ratio=train_ratio, val_ratio=val_ratio)

    # --- Pass 2: build samples and assign splits ---
    split_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    stats: Counter = Counter()
    stratum_split_counts: Dict[str, Counter] = defaultdict(Counter)

    for xml_stem, image_elem, image_key in all_entries:
        sample = build_stage1_sample(
            image_elem=image_elem,
            image_root=image_root,
            keep_meta=keep_meta,
            min_short_side=min_short_side,
            min_area=min_area,
        )
        if keep_meta:
            sample["meta"]["source_xml"] = xml_stem
            sample["meta"]["stratum"] = extract_stratum(image_key)

        subset = image_elem.attrib.get("subset")
        if subset in {"train", "val", "test"}:
            split = subset
        else:
            split = split_map.get(image_key, "train")

        split_to_records[split].append(sample)
        stats[f"{split}_images"] += 1
        assistant_payload = json.loads(sample["conversations"][1]["value"])
        stats[f"{split}_targets"] += len(assistant_payload["detections"])
        if len(assistant_payload["detections"]) == 0:
            stats[f"{split}_empty"] += 1
        stratum_split_counts[extract_stratum(image_key)][split] += 1

    written = {}
    for split, records in split_to_records.items():
        out_path = output_dir_path / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        written[split] = str(out_path)

    # Build per-stratum distribution table for the summary
    stratum_table = {
        stratum: dict(counts)
        for stratum, counts in sorted(stratum_split_counts.items())
    }

    summary = {
        "xml_files": [str(p) for p in xml_paths],
        "num_xml_files": len(xml_paths),
        "num_images": len(all_entries),
        "written_files": written,
        "stats": dict(stats),
        "split_rule": {
            "method": "stratified by category×state stratum (directory path)",
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": round(1.0 - train_ratio - val_ratio, 6),
        },
        "stratum_distribution": stratum_table,
        "filters": {
            "min_short_side": min_short_side,
            "min_area": min_area,
        },
        "sorting": "detections sorted by device_type priority, then bbox x1, y1, x2, y2",
    }

    summary_path = output_dir_path / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    summary_path = output_dir_path / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Stage-1 CVAT XML to SFT JSONL.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--xml", type=str, help="Path to a single CVAT XML annotation file.")
    group.add_argument(
        "--xml_dir",
        type=str,
        help="Directory containing multiple CVAT XML files (annotations_*.xml). All will be merged.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save JSONL outputs.")
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Optional root prefix to prepend to image paths stored in XML. "
             "For hefei_stage1_cvat_data, set this to the hefei_stage1_cvat_data directory.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio for images without subset attr.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Val split ratio for images without subset attr.")
    parser.add_argument("--keep_meta", action="store_true", help="Keep width/height and extra metadata in each sample.")
    parser.add_argument(
        "--min_short_side",
        type=float,
        default=0.0,
        help="Optional filter: drop boxes whose short side is smaller than this threshold (pixels).",
    )
    parser.add_argument(
        "--min_area",
        type=float,
        default=0.0,
        help="Optional filter: drop boxes whose area is smaller than this threshold (pixels^2).",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.xml_dir:
        xml_paths = sorted(Path(args.xml_dir).glob("annotations_*.xml"))
        if not xml_paths:
            raise FileNotFoundError(f"No annotations_*.xml files found in {args.xml_dir}")
        print(f"Found {len(xml_paths)} XML files: {[p.name for p in xml_paths]}")
        convert_multi(
            xml_paths=[str(p) for p in xml_paths],
            output_dir=args.output_dir,
            image_root=args.image_root,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            keep_meta=args.keep_meta,
            min_short_side=args.min_short_side,
            min_area=args.min_area,
        )
    else:
        convert(
            xml_path=args.xml,
            output_dir=args.output_dir,
            image_root=args.image_root,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            keep_meta=args.keep_meta,
            min_short_side=args.min_short_side,
            min_area=args.min_area,
        )


if __name__ == "__main__":
    main()
