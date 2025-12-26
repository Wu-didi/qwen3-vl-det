"""Pydantic models for API request/response schemas."""

from typing import Annotated

from pydantic import BaseModel, Field

from app.models.prompts import AnomalyCategory, AnomalyType


class BoundingBox(BaseModel):
    """Bounding box coordinates (normalized 0-1000)."""

    x1: Annotated[float, Field(ge=0, le=1000, description="Left x coordinate")]
    y1: Annotated[float, Field(ge=0, le=1000, description="Top y coordinate")]
    x2: Annotated[float, Field(ge=0, le=1000, description="Right x coordinate")]
    y2: Annotated[float, Field(ge=0, le=1000, description="Bottom y coordinate")]

    def to_list(self) -> list[float]:
        """Convert to list format."""
        return [self.x1, self.y1, self.x2, self.y2]

    def to_pixel_coords(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert normalized coords to pixel coordinates.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            Tuple of (x1, y1, x2, y2) in pixel coordinates.
        """
        return (
            int(self.x1 * width / 1000),
            int(self.y1 * height / 1000),
            int(self.x2 * width / 1000),
            int(self.y2 * height / 1000),
        )

    @classmethod
    def from_list(cls, coords: list[float]) -> "BoundingBox":
        """Create from list format."""
        if len(coords) != 4:
            raise ValueError("Bounding box must have 4 coordinates")
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])


class Detection(BaseModel):
    """Single anomaly detection result."""

    category: str = Field(description="Anomaly category (e.g., traffic_sign)")
    anomaly_type: str = Field(description="Type of anomaly (e.g., damaged)")
    confidence: Annotated[
        float, Field(ge=0.0, le=1.0, description="Detection confidence score")
    ]
    bbox: list[float] | None = Field(
        default=None, description="Bounding box [x1, y1, x2, y2] normalized 0-1000"
    )
    description: str = Field(description="Detailed description of the anomaly")

    def get_bbox(self) -> BoundingBox | None:
        """Get bounding box as BoundingBox object."""
        if self.bbox is None:
            return None
        return BoundingBox.from_list(self.bbox)


class DetectionResult(BaseModel):
    """Complete detection result for an image."""

    has_anomaly: bool = Field(description="Whether any anomaly was detected")
    detections: list[Detection] = Field(
        default_factory=list, description="List of detected anomalies"
    )
    summary: str = Field(description="Summary of detection results")
    image_id: str | None = Field(default=None, description="Optional image identifier")
    processing_time_ms: float | None = Field(
        default=None, description="Processing time in milliseconds"
    )


class DetectionRequest(BaseModel):
    """Request model for single image detection."""

    image_base64: str | None = Field(
        default=None, description="Base64 encoded image data"
    )
    image_url: str | None = Field(default=None, description="URL of the image")
    image_id: str | None = Field(default=None, description="Optional image identifier")
    categories: list[str] | None = Field(
        default=None,
        description="Specific categories to detect. If None, detects all.",
    )
    quick_mode: bool = Field(
        default=False, description="Use quick mode for faster inference"
    )

    def model_post_init(self, __context) -> None:
        """Validate that either image_base64 or image_url is provided."""
        if self.image_base64 is None and self.image_url is None:
            raise ValueError("Either image_base64 or image_url must be provided")


class BatchDetectionRequest(BaseModel):
    """Request model for batch image detection."""

    images: list[DetectionRequest] = Field(
        description="List of images to process", min_length=1, max_length=10
    )


class BatchDetectionResult(BaseModel):
    """Response model for batch detection."""

    results: list[DetectionResult] = Field(description="List of detection results")
    total_images: int = Field(description="Total number of images processed")
    total_anomalies: int = Field(description="Total number of anomalies detected")
    total_processing_time_ms: float = Field(
        description="Total processing time in milliseconds"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether the model is loaded")
    model_name: str = Field(description="Name of the loaded model")
    device: str = Field(description="Device being used for inference")


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(description="Error message")
    detail: str | None = Field(default=None, description="Detailed error information")
