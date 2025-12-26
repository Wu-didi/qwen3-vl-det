#!/usr/bin/env python3
"""
CVAT XML to Qwen-VL fine-tuning format converter.

Converts CVAT annotations to Qwen-VL conversation format for anomaly detection task.
"""

import os
import json
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class BoundingBox:
    """Bounding box annotation."""
    label: str
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    state: str = "normal"
    attributes: Dict[str, str] = field(default_factory=dict)

    def to_normalized(self, width: int, height: int, scale: int = 1000) -> Tuple[int, int, int, int]:
        """Convert to normalized coordinates (0-scale)."""
        x1 = int(self.xtl / width * scale)
        y1 = int(self.ytl / height * scale)
        x2 = int(self.xbr / width * scale)
        y2 = int(self.ybr / height * scale)
        return (x1, y1, x2, y2)


@dataclass
class ImageAnnotation:
    """Image with its annotations."""
    image_id: str
    image_path: str
    width: int
    height: int
    boxes: List[BoundingBox] = field(default_factory=list)


# Category mappings (CVAT label -> Chinese description)
CATEGORY_NAMES = {
    "traffic-signal-system": "交通信号灯",
    "traffic-guidance-system": "交通诱导屏",
    "restricted-elevated": "限高架",
    "cabinet": "机柜",
    "backpack-box": "背包箱",
    "off-site": "非现场系统",
    "Gun-type-Camera": "枪机摄像头",
    "Dome-Camera": "球机摄像头",
    "Flashlight": "闪光灯",
    "b-Flashlight": "爆闪灯",
    "off-site-other": "其他设备",
    "ignore": "忽略区域",
}

# State mappings (CVAT state -> Chinese description)
STATE_NAMES = {
    "normal": "正常",
    "abnormal": "异常",
    "all-off": "全灭",
    "all-on": "全亮",
    "black-screen": "黑屏",
}

# Abnormal reason descriptions (category, state) -> reason
ABNORMAL_REASONS = {
    # 交通信号灯
    ("traffic-signal-system", "all-off"): "信号灯不亮，可能存在电源故障、灯泡损坏或控制器故障",
    ("traffic-signal-system", "all-on"): "信号灯全亮，控制系统可能发生故障",
    ("traffic-signal-system", "abnormal"): "信号灯显示异常，可能存在灯泡部分损坏或颜色显示错误",
    # 交通诱导屏
    ("traffic-guidance-system", "black-screen"): "诱导屏黑屏不显示，可能存在电源故障或显示模块损坏",
    ("traffic-guidance-system", "abnormal"): "诱导屏显示异常，可能存在显示内容错乱或部分像素损坏",
    # 限高架
    ("restricted-elevated", "abnormal"): "限高架异常，可能存在结构损坏、标识缺失或倾斜变形",
    # 机柜
    ("cabinet", "abnormal"): "机柜异常，可能存在柜门未关闭、外壳破损或倾斜",
    # 背包箱
    ("backpack-box", "abnormal"): "背包箱异常，可能存在箱门未关闭、外壳破损或安装松动",
}

# Abnormal states that need detection
ABNORMAL_STATES = {"abnormal", "all-off", "all-on", "black-screen"}

# Main equipment categories (exclude sub-categories of off-site)
MAIN_CATEGORIES = {
    "traffic-signal-system",
    "traffic-guidance-system",
    "restricted-elevated",
    "cabinet",
    "backpack-box",
}


def parse_cvat_xml(xml_path: str) -> List[ImageAnnotation]:
    """Parse CVAT XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = []

    for image_elem in root.findall(".//image"):
        image_id = image_elem.get("id")
        image_path = image_elem.get("name")
        width = int(image_elem.get("width"))
        height = int(image_elem.get("height"))

        boxes = []
        for box_elem in image_elem.findall("box"):
            label = box_elem.get("label")
            xtl = float(box_elem.get("xtl"))
            ytl = float(box_elem.get("ytl"))
            xbr = float(box_elem.get("xbr"))
            ybr = float(box_elem.get("ybr"))

            # Parse attributes
            attributes = {}
            state = "normal"
            for attr_elem in box_elem.findall("attribute"):
                attr_name = attr_elem.get("name")
                attr_value = attr_elem.text
                attributes[attr_name] = attr_value
                if attr_name == "state":
                    state = attr_value

            box = BoundingBox(
                label=label,
                xtl=xtl,
                ytl=ytl,
                xbr=xbr,
                ybr=ybr,
                state=state,
                attributes=attributes
            )
            boxes.append(box)

        annotation = ImageAnnotation(
            image_id=image_id,
            image_path=image_path,
            width=width,
            height=height,
            boxes=boxes
        )
        annotations.append(annotation)

    return annotations


def filter_valid_annotations(annotations: List[ImageAnnotation],
                              require_abnormal: bool = False,
                              abnormal_ratio: float = 1.0) -> List[ImageAnnotation]:
    """
    Filter annotations to keep images with main equipment.

    Args:
        annotations: List of image annotations
        require_abnormal: If True, only keep images with at least one abnormal equipment
        abnormal_ratio: If require_abnormal is False, ratio of abnormal samples to keep
                       (1.0 = keep all, 0.5 = keep 50% of normal samples)

    Returns:
        Filtered list of annotations
    """
    abnormal_samples = []
    normal_samples = []

    for ann in annotations:
        # Check if image has main category equipment
        has_main_equipment = any(box.label in MAIN_CATEGORIES for box in ann.boxes)
        if not has_main_equipment:
            continue

        # Check if image has any abnormal equipment
        has_abnormal = any(
            box.label in MAIN_CATEGORIES and box.state in ABNORMAL_STATES
            for box in ann.boxes
        )

        if has_abnormal:
            abnormal_samples.append(ann)
        else:
            normal_samples.append(ann)

    print(f"Found {len(abnormal_samples)} images with abnormal equipment")
    print(f"Found {len(normal_samples)} images with only normal equipment")

    if require_abnormal:
        return abnormal_samples

    # Balance the dataset if needed
    if abnormal_ratio < 1.0 and normal_samples:
        num_normal = int(len(normal_samples) * abnormal_ratio)
        normal_samples = random.sample(normal_samples, num_normal)
        print(f"Sampled {num_normal} normal images for balance")

    all_samples = abnormal_samples + normal_samples
    print(f"Total samples after filtering: {len(all_samples)}")
    return all_samples


def format_bbox_qwen(x1: int, y1: int, x2: int, y2: int) -> str:
    """Format bbox in Qwen-VL style: <box>(x1,y1),(x2,y2)</box>"""
    return f"<box>({x1},{y1}),({x2},{y2})</box>"


def format_ref_qwen(text: str, x1: int, y1: int, x2: int, y2: int) -> str:
    """Format reference with bbox in Qwen-VL style: <ref>text</ref><box>...</box>"""
    return f"<ref>{text}</ref><box>({x1},{y1}),({x2},{y2})</box>"


def create_detection_prompt() -> str:
    """Create the detection prompt for Qwen-VL."""
    return """请检测图像中的交通设备。需要检测的设备类型包括：交通信号灯、交通诱导屏、限高架、机柜、背包箱。

请按以下格式输出每个检测到的设备：
1. 设备类型
2. 状态（正常/异常状态）
3. 如果异常，说明可能的原因
4. 位置坐标：<box>(x1,y1),(x2,y2)</box>（坐标范围0-1000）

如果没有检测到任何设备，请回复"未检测到相关设备"。"""


def get_abnormal_reason(label: str, state: str) -> str:
    """Get the abnormal reason description for a given category and state."""
    reason = ABNORMAL_REASONS.get((label, state))
    if reason:
        return reason
    # Fallback generic reason
    category_name = CATEGORY_NAMES.get(label, label)
    state_name = STATE_NAMES.get(state, state)
    return f"{category_name}出现{state_name}状态，需要检修"


def create_response_for_annotation(ann: ImageAnnotation) -> str:
    """Create detection response for an annotation, including both normal and abnormal equipment."""
    equipment_items = []

    for box in ann.boxes:
        if box.label not in MAIN_CATEGORIES:
            continue
        if box.label == "ignore":
            continue

        coords = box.to_normalized(ann.width, ann.height)
        category_name = CATEGORY_NAMES.get(box.label, box.label)
        state_name = STATE_NAMES.get(box.state, box.state)

        # Add extra info for traffic signals
        extra_info = ""
        if box.label == "traffic-signal-system":
            if "class" in box.attributes:
                cls = box.attributes["class"]
                if cls == "Vehicle-traffic-signal-lights":
                    extra_info = "（车行灯）"
                elif cls == "Pedestrian-traffic-signal-lights":
                    extra_info = "（人行灯）"

        if box.state in ABNORMAL_STATES:
            # Abnormal equipment - include reason
            reason = get_abnormal_reason(box.label, box.state)
            item = {
                "category": f"{category_name}{extra_info}",
                "state": state_name,
                "reason": reason,
                "bbox": format_bbox_qwen(*coords),
                "is_abnormal": True
            }
        else:
            # Normal equipment - no reason needed
            item = {
                "category": f"{category_name}{extra_info}",
                "state": "正常",
                "reason": None,
                "bbox": format_bbox_qwen(*coords),
                "is_abnormal": False
            }
        equipment_items.append(item)

    if not equipment_items:
        return "未检测到相关设备。"

    response = "检测到以下交通设备：\n"
    for i, item in enumerate(equipment_items, 1):
        response += f"\n{i}. {item['category']}\n"
        response += f"   - 状态：{item['state']}\n"
        if item['is_abnormal'] and item['reason']:
            response += f"   - 原因：{item['reason']}\n"
        response += f"   - 位置：{item['bbox']}"

    return response


def convert_to_qwenvl_format(annotations: List[ImageAnnotation],
                             image_base_path: str) -> List[Dict]:
    """
    Convert annotations to Qwen-VL fine-tuning format.

    Output format:
    {
        "id": "unique_id",
        "conversations": [
            {"from": "user", "value": "<image>prompt"},
            {"from": "assistant", "value": "response with <box> tags"}
        ],
        "image": "path/to/image"
    }
    """
    samples = []
    prompt = create_detection_prompt()

    for ann in annotations:
        # Construct full image path
        image_path = os.path.join(image_base_path, ann.image_path)

        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        response = create_response_for_annotation(ann)

        sample = {
            "id": f"traffic_anomaly_{ann.image_id}",
            "conversations": [
                {
                    "from": "user",
                    "value": f"<image>\n{prompt}"
                },
                {
                    "from": "assistant",
                    "value": response
                }
            ],
            "image": image_path
        }
        samples.append(sample)

    return samples


def split_dataset(samples: List[Dict],
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1,
                  seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset into train/val/test sets."""
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_set = shuffled[:train_end]
    val_set = shuffled[train_end:val_end]
    test_set = shuffled[val_end:]

    return train_set, val_set, test_set


def analyze_annotations(annotations: List[ImageAnnotation]) -> Dict:
    """Analyze annotation statistics."""
    stats = {
        "total_images": len(annotations),
        "category_counts": defaultdict(int),
        "state_counts": defaultdict(int),
        "abnormal_by_category": defaultdict(int),
    }

    for ann in annotations:
        for box in ann.boxes:
            stats["category_counts"][box.label] += 1
            if box.label in MAIN_CATEGORIES:
                stats["state_counts"][box.state] += 1
                if box.state in ABNORMAL_STATES:
                    stats["abnormal_by_category"][box.label] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert CVAT annotations to Qwen-VL format")
    parser.add_argument("--cvat-dir", type=str, required=True,
                        help="Directory containing CVAT XML files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for converted data")
    parser.add_argument("--only-abnormal", action="store_true",
                        help="Only include images with abnormal equipment")
    parser.add_argument("--normal-sample-ratio", type=float, default=1.0,
                        help="Ratio of normal-only images to keep (1.0=all, 0.5=half)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all XML files
    xml_files = list(Path(args.cvat_dir).glob("annotations_*.xml"))
    print(f"Found {len(xml_files)} CVAT XML files")

    # Parse all annotations
    all_annotations = []
    for xml_file in xml_files:
        print(f"Parsing {xml_file.name}...")
        annotations = parse_cvat_xml(str(xml_file))
        all_annotations.extend(annotations)

    print(f"\nTotal images parsed: {len(all_annotations)}")

    # Analyze statistics
    stats = analyze_annotations(all_annotations)
    print("\n=== Annotation Statistics ===")
    print(f"Total images: {stats['total_images']}")
    print("\nCategory counts:")
    for cat, count in sorted(stats['category_counts'].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    print("\nState distribution (main categories):")
    for state, count in sorted(stats['state_counts'].items(), key=lambda x: -x[1]):
        print(f"  {state}: {count}")
    print("\nAbnormal counts by category:")
    for cat, count in sorted(stats['abnormal_by_category'].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Filter valid annotations (images with main equipment)
    filtered_annotations = filter_valid_annotations(
        all_annotations,
        require_abnormal=args.only_abnormal,
        abnormal_ratio=args.normal_sample_ratio
    )

    # Convert to Qwen-VL format
    image_base_path = args.cvat_dir
    samples = convert_to_qwenvl_format(filtered_annotations, image_base_path)
    print(f"\nConverted {len(samples)} samples to Qwen-VL format")

    # Split dataset
    train_set, val_set, test_set = split_dataset(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1 - args.train_ratio - args.val_ratio,
        seed=args.seed
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train_set)}")
    print(f"  Val: {len(val_set)}")
    print(f"  Test: {len(test_set)}")

    # Save datasets
    train_path = os.path.join(args.output_dir, "train.json")
    val_path = os.path.join(args.output_dir, "val.json")
    test_path = os.path.join(args.output_dir, "test.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_set, f, ensure_ascii=False, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)

    print(f"\nSaved datasets to {args.output_dir}")

    # Save statistics
    stats_path = os.path.join(args.output_dir, "stats.json")
    # Convert defaultdict to regular dict for JSON serialization
    stats_serializable = {
        k: dict(v) if isinstance(v, defaultdict) else v
        for k, v in stats.items()
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_serializable, f, ensure_ascii=False, indent=2)

    # Show sample output
    if samples:
        print("\n=== Sample Output ===")
        sample = samples[0]
        print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
