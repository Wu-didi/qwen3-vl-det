#!/usr/bin/env python3
"""Benchmark script for testing detection performance."""

import argparse
import os
import time
from pathlib import Path

from PIL import Image


def run_benchmark(
    image_dir: str,
    num_iterations: int = 10,
    quick_mode: bool = False,
) -> None:
    """Run performance benchmark.

    Args:
        image_dir: Directory containing test images.
        num_iterations: Number of iterations per image.
        quick_mode: Use quick detection mode.
    """
    from app.core.config import get_settings
    from app.core.detector import get_detector
    from app.utils.image_utils import resize_image

    settings = get_settings()
    detector = get_detector()

    # Ensure model is loaded
    detector.ensure_model_loaded()

    # Find test images
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_paths.extend(Path(image_dir).glob(ext))

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_paths)} images")
    print(f"Running {num_iterations} iterations per image")
    print(f"Quick mode: {quick_mode}")
    print("-" * 50)

    total_time = 0
    total_detections = 0

    for image_path in image_paths:
        print(f"\nImage: {image_path.name}")

        # Load and resize image
        image = Image.open(image_path).convert("RGB")
        image = resize_image(image, settings.max_image_size)

        image_times = []
        for i in range(num_iterations):
            start = time.time()
            result = detector.detect(
                image=image,
                quick_mode=quick_mode,
                image_id=str(image_path),
            )
            elapsed = (time.time() - start) * 1000
            image_times.append(elapsed)
            total_detections += len(result.detections)

        avg_time = sum(image_times) / len(image_times)
        min_time = min(image_times)
        max_time = max(image_times)
        total_time += sum(image_times)

        print(f"  Avg: {avg_time:.1f}ms, Min: {min_time:.1f}ms, Max: {max_time:.1f}ms")
        print(f"  Detections: {len(result.detections)}")

    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Total iterations: {len(image_paths) * num_iterations}")
    print(f"  Total detections: {total_detections}")
    print(f"  Total time: {total_time:.1f}ms")
    print(
        f"  Avg per image: {total_time / (len(image_paths) * num_iterations):.1f}ms"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark detection performance")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="./examples/sample_images",
        help="Directory containing test images",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations per image (default: 10)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick detection mode",
    )

    args = parser.parse_args()

    run_benchmark(
        image_dir=args.image_dir,
        num_iterations=args.iterations,
        quick_mode=args.quick,
    )


if __name__ == "__main__":
    main()
