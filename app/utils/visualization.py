"""Visualization utilities for detection results."""

import io
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.schemas.detection import Detection, DetectionResult

# Color mapping for different categories
CATEGORY_COLORS = {
    "traffic_sign": (255, 0, 0),  # Red
    "traffic_light": (0, 255, 0),  # Green
    "road_facility": (0, 0, 255),  # Blue
    "guidance_screen": (255, 165, 0),  # Orange
    "height_limit": (128, 0, 128),  # Purple
    "cabinet": (255, 192, 203),  # Pink
    "unknown": (128, 128, 128),  # Gray
}

# Chinese labels for categories
CATEGORY_LABELS = {
    "traffic_sign": "交通标志",
    "traffic_light": "信号灯",
    "road_facility": "道路设施",
    "guidance_screen": "诱导屏",
    "height_limit": "限高架",
    "cabinet": "机柜",
    "unknown": "未知",
}


def get_category_color(category: str) -> Tuple[int, int, int]:
    """Get color for a category.

    Args:
        category: Category name.

    Returns:
        RGB color tuple.
    """
    return CATEGORY_COLORS.get(category, CATEGORY_COLORS["unknown"])


def draw_detection_on_image(
    image: Image.Image,
    detection: Detection,
    line_width: int = 3,
    font_size: int = 20,
) -> Image.Image:
    """Draw a single detection on an image.

    Args:
        image: PIL Image to draw on.
        detection: Detection to visualize.
        line_width: Width of bounding box lines.
        font_size: Font size for labels.

    Returns:
        Image with detection drawn.
    """
    if detection.bbox is None:
        return image

    # Make a copy
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Convert normalized coordinates to pixel coordinates
    width, height = img.size
    x1 = int(detection.bbox[0] * width / 1000)
    y1 = int(detection.bbox[1] * height / 1000)
    x2 = int(detection.bbox[2] * width / 1000)
    y2 = int(detection.bbox[3] * height / 1000)

    # Get color
    color = get_category_color(detection.category)

    # Draw bounding box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

    # Prepare label text
    category_label = CATEGORY_LABELS.get(detection.category, detection.category)
    label = f"{category_label}: {detection.anomaly_type} ({detection.confidence:.2f})"

    # Try to load a font that supports Chinese
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Get text bounding box
    text_bbox = draw.textbbox((x1, y1), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Draw label background
    label_y = max(0, y1 - text_height - 4)
    draw.rectangle(
        [x1, label_y, x1 + text_width + 4, label_y + text_height + 4],
        fill=color,
    )

    # Draw label text
    draw.text((x1 + 2, label_y + 2), label, fill=(255, 255, 255), font=font)

    return img


def visualize_detections(
    image: Image.Image,
    result: DetectionResult,
    line_width: int = 3,
    font_size: int = 20,
    show_summary: bool = True,
) -> Image.Image:
    """Visualize all detections on an image.

    Args:
        image: PIL Image to draw on.
        result: DetectionResult with detections.
        line_width: Width of bounding box lines.
        font_size: Font size for labels.
        show_summary: Whether to show summary text.

    Returns:
        Image with all detections drawn.
    """
    img = image.copy()

    # Draw each detection
    for detection in result.detections:
        img = draw_detection_on_image(img, detection, line_width, font_size)

    # Add summary at the bottom if requested
    if show_summary and result.summary:
        img = _add_summary_bar(img, result.summary, font_size)

    return img


def _add_summary_bar(
    image: Image.Image,
    summary: str,
    font_size: int = 20,
) -> Image.Image:
    """Add a summary bar at the bottom of the image.

    Args:
        image: PIL Image.
        summary: Summary text.
        font_size: Font size.

    Returns:
        Image with summary bar.
    """
    # Create new image with extra height for summary
    bar_height = font_size + 20
    new_img = Image.new("RGB", (image.width, image.height + bar_height), (0, 0, 0))
    new_img.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_img)

    # Try to load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Truncate summary if too long
    max_chars = image.width // (font_size // 2)
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."

    # Draw summary
    draw.text((10, image.height + 5), summary, fill=(255, 255, 255), font=font)

    return new_img


def create_comparison_image(
    original: Image.Image,
    annotated: Image.Image,
) -> Image.Image:
    """Create side-by-side comparison of original and annotated images.

    Args:
        original: Original image.
        annotated: Image with annotations.

    Returns:
        Combined comparison image.
    """
    # Resize to same height if different
    if original.height != annotated.height:
        ratio = original.height / annotated.height
        new_width = int(annotated.width * ratio)
        annotated = annotated.resize((new_width, original.height), Image.Resampling.LANCZOS)

    # Create combined image
    combined = Image.new("RGB", (original.width + annotated.width + 10, original.height))
    combined.paste(original, (0, 0))
    combined.paste(annotated, (original.width + 10, 0))

    return combined


def save_visualization(
    image: Image.Image,
    path: str,
    format: str = "JPEG",
    quality: int = 95,
) -> None:
    """Save visualization to file.

    Args:
        image: PIL Image to save.
        path: Output file path.
        format: Image format.
        quality: JPEG quality (1-100).
    """
    if format.upper() == "JPEG":
        image.save(path, format=format, quality=quality)
    else:
        image.save(path, format=format)


def image_to_bytes(
    image: Image.Image,
    format: str = "JPEG",
    quality: int = 95,
) -> bytes:
    """Convert PIL Image to bytes.

    Args:
        image: PIL Image.
        format: Output format.
        quality: JPEG quality.

    Returns:
        Image bytes.
    """
    buffer = io.BytesIO()
    if format.upper() == "JPEG":
        image.save(buffer, format=format, quality=quality)
    else:
        image.save(buffer, format=format)
    return buffer.getvalue()
