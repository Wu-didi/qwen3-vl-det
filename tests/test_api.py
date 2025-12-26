"""Tests for API endpoints."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client():
    """Create test client."""
    from app.main import app

    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


def test_root(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["service"] == "Traffic Equipment Anomaly Detection"


def test_health(client):
    """Test health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_categories(client):
    """Test categories endpoint."""
    response = client.get("/api/v1/categories")
    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert "anomaly_types" in data
    assert len(data["categories"]) > 0


# Note: Detection tests require model to be loaded
# These are integration tests that should be run with the model available

# @pytest.mark.integration
# def test_detect_upload(client, sample_image):
#     """Test detection from file upload."""
#     response = client.post(
#         "/api/v1/detect/upload",
#         files={"file": ("test.jpg", sample_image, "image/jpeg")},
#     )
#     assert response.status_code == 200
#     data = response.json()
#     assert "has_anomaly" in data
#     assert "detections" in data
#     assert "summary" in data
