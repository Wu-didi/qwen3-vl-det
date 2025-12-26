"""Tests for detector module."""

import pytest
from PIL import Image

from app.schemas.detection import BoundingBox, Detection, DetectionResult


def test_bounding_box_from_list():
    """Test BoundingBox creation from list."""
    bbox = BoundingBox.from_list([100, 200, 300, 400])
    assert bbox.x1 == 100
    assert bbox.y1 == 200
    assert bbox.x2 == 300
    assert bbox.y2 == 400


def test_bounding_box_to_pixel_coords():
    """Test coordinate conversion."""
    bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400)
    # For a 1000x1000 image, normalized coords equal pixel coords
    x1, y1, x2, y2 = bbox.to_pixel_coords(1000, 1000)
    assert x1 == 100
    assert y1 == 200
    assert x2 == 300
    assert y2 == 400

    # For a 2000x2000 image, coords should be doubled
    x1, y1, x2, y2 = bbox.to_pixel_coords(2000, 2000)
    assert x1 == 200
    assert y1 == 400
    assert x2 == 600
    assert y2 == 800


def test_detection_model():
    """Test Detection model."""
    det = Detection(
        category="traffic_sign",
        anomaly_type="damaged",
        confidence=0.85,
        bbox=[100, 200, 300, 400],
        description="Test detection",
    )
    assert det.category == "traffic_sign"
    assert det.confidence == 0.85
    bbox = det.get_bbox()
    assert bbox is not None
    assert bbox.x1 == 100


def test_detection_result_model():
    """Test DetectionResult model."""
    result = DetectionResult(
        has_anomaly=True,
        detections=[
            Detection(
                category="traffic_sign",
                anomaly_type="damaged",
                confidence=0.85,
                bbox=[100, 200, 300, 400],
                description="Test",
            )
        ],
        summary="Found 1 anomaly",
    )
    assert result.has_anomaly is True
    assert len(result.detections) == 1
