"""API routes for traffic equipment anomaly detection."""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image

from app.core.config import Settings, get_settings
from app.core.detector import TrafficAnomalyDetector, get_detector
from app.models.prompts import AnomalyCategory
from app.models.qwen_vl import QwenVLModel, get_model
from app.schemas.detection import (
    BatchDetectionRequest,
    BatchDetectionResult,
    DetectionRequest,
    DetectionResult,
    ErrorResponse,
    HealthResponse,
)
from app.utils.image_utils import (
    load_image_from_base64,
    load_image_from_url,
    resize_image,
)
from app.utils.visualization import image_to_bytes, visualize_detections

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["detection"])


def get_model_dep() -> QwenVLModel:
    """Dependency for getting the model."""
    return get_model()


def get_detector_dep() -> TrafficAnomalyDetector:
    """Dependency for getting the detector."""
    return get_detector()


def get_settings_dep() -> Settings:
    """Dependency for getting settings."""
    return get_settings()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    model: Annotated[QwenVLModel, Depends(get_model_dep)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> HealthResponse:
    """Check service health status."""
    return HealthResponse(
        status="healthy" if model.is_loaded else "model_not_loaded",
        model_loaded=model.is_loaded,
        model_name=settings.effective_model_name,
        device=settings.device,
    )


@router.post("/load-model")
async def load_model(
    model: Annotated[QwenVLModel, Depends(get_model_dep)],
) -> dict:
    """Manually load the model."""
    if model.is_loaded:
        return {"message": "Model already loaded"}

    try:
        model.load()
        return {"message": "Model loaded successfully"}
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.post(
    "/detect",
    response_model=DetectionResult,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def detect_anomalies(
    request: DetectionRequest,
    detector: Annotated[TrafficAnomalyDetector, Depends(get_detector_dep)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> DetectionResult:
    """Detect traffic equipment anomalies in a single image.

    Accepts either base64 encoded image or image URL.
    """
    # Load image
    try:
        if request.image_base64:
            image = load_image_from_base64(request.image_base64)
        elif request.image_url:
            image = load_image_from_url(request.image_url)
        else:
            raise HTTPException(
                status_code=400, detail="Either image_base64 or image_url is required"
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Resize if needed
    image = resize_image(image, settings.max_image_size)

    # Parse categories
    categories = None
    if request.categories:
        try:
            categories = [AnomalyCategory(c) for c in request.categories]
        except ValueError:
            # If invalid category, detect all
            categories = None

    # Run detection
    try:
        result = detector.detect(
            image=image,
            categories=categories,
            quick_mode=request.quick_mode,
            image_id=request.image_id,
        )
        return result
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.post(
    "/detect/upload",
    response_model=DetectionResult,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def detect_from_upload(
    file: UploadFile = File(...),
    image_id: Annotated[str | None, Query(description="Image identifier")] = None,
    categories: Annotated[
        str | None, Query(description="Comma-separated categories to detect")
    ] = None,
    quick_mode: Annotated[bool, Query(description="Use quick mode")] = False,
    detector: TrafficAnomalyDetector = Depends(get_detector_dep),
    settings: Settings = Depends(get_settings_dep),
) -> DetectionResult:
    """Detect anomalies from uploaded image file."""
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WebP.",
        )

    # Load image
    try:
        contents = await file.read()
        import io

        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")

    # Resize if needed
    image = resize_image(image, settings.max_image_size)

    # Parse categories
    cat_list = None
    if categories:
        try:
            cat_list = [AnomalyCategory(c.strip()) for c in categories.split(",")]
        except ValueError:
            cat_list = None

    # Run detection
    try:
        result = detector.detect(
            image=image,
            categories=cat_list,
            quick_mode=quick_mode,
            image_id=image_id or file.filename,
        )
        return result
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.post(
    "/detect/visualize",
    responses={
        200: {"content": {"image/jpeg": {}}},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def detect_and_visualize(
    file: UploadFile = File(...),
    categories: Annotated[
        str | None, Query(description="Comma-separated categories")
    ] = None,
    quick_mode: Annotated[bool, Query(description="Use quick mode")] = False,
    detector: TrafficAnomalyDetector = Depends(get_detector_dep),
    settings: Settings = Depends(get_settings_dep),
) -> Response:
    """Detect anomalies and return annotated image."""
    # Validate and load image
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        contents = await file.read()
        import io

        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")

    image = resize_image(image, settings.max_image_size)

    # Parse categories
    cat_list = None
    if categories:
        try:
            cat_list = [AnomalyCategory(c.strip()) for c in categories.split(",")]
        except ValueError:
            cat_list = None

    # Run detection
    try:
        result = detector.detect(
            image=image,
            categories=cat_list,
            quick_mode=quick_mode,
            image_id=file.filename,
        )
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    # Visualize results
    annotated = visualize_detections(image, result)
    image_bytes = image_to_bytes(annotated)

    return Response(content=image_bytes, media_type="image/jpeg")


@router.post(
    "/detect/batch",
    response_model=BatchDetectionResult,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def detect_batch(
    request: BatchDetectionRequest,
    detector: Annotated[TrafficAnomalyDetector, Depends(get_detector_dep)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> BatchDetectionResult:
    """Detect anomalies in multiple images."""
    start_time = time.time()
    results = []
    total_anomalies = 0

    for img_request in request.images:
        # Load image
        try:
            if img_request.image_base64:
                image = load_image_from_base64(img_request.image_base64)
            elif img_request.image_url:
                image = load_image_from_url(img_request.image_url)
            else:
                results.append(
                    DetectionResult(
                        has_anomaly=False,
                        detections=[],
                        summary="错误: 未提供图片",
                        image_id=img_request.image_id,
                    )
                )
                continue
        except ValueError as e:
            results.append(
                DetectionResult(
                    has_anomaly=False,
                    detections=[],
                    summary=f"错误: {str(e)}",
                    image_id=img_request.image_id,
                )
            )
            continue

        # Resize
        image = resize_image(image, settings.max_image_size)

        # Parse categories
        categories = None
        if img_request.categories:
            try:
                categories = [AnomalyCategory(c) for c in img_request.categories]
            except ValueError:
                categories = None

        # Detect
        try:
            result = detector.detect(
                image=image,
                categories=categories,
                quick_mode=img_request.quick_mode,
                image_id=img_request.image_id,
            )
            results.append(result)
            total_anomalies += len(result.detections)
        except Exception as e:
            results.append(
                DetectionResult(
                    has_anomaly=False,
                    detections=[],
                    summary=f"检测失败: {str(e)}",
                    image_id=img_request.image_id,
                )
            )

    total_time = (time.time() - start_time) * 1000

    return BatchDetectionResult(
        results=results,
        total_images=len(results),
        total_anomalies=total_anomalies,
        total_processing_time_ms=total_time,
    )


@router.get("/categories")
async def list_categories() -> dict:
    """List all supported anomaly categories and types."""
    from app.models.prompts import AnomalyCategory, AnomalyType

    return {
        "categories": [
            {"value": c.value, "label": c.name}
            for c in AnomalyCategory
        ],
        "anomaly_types": [
            {"value": t.value, "label": t.name}
            for t in AnomalyType
        ],
    }


@router.get("/models")
async def list_models(
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> dict:
    """List all supported models and current selection."""
    from app.core.config import MODEL_INFO, ModelType

    models = []
    for model_type in ModelType:
        if model_type == ModelType.CUSTOM:
            continue
        vram, desc = MODEL_INFO.get(model_type, (0, ""))
        models.append({
            "type": model_type.name,
            "name": model_type.value,
            "vram_gb": vram,
            "description": desc,
        })

    return {
        "available_models": models,
        "current_model_type": settings.model_type.name,
        "current_model_name": settings.effective_model_name,
    }
