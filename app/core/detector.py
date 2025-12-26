"""Core detection logic for traffic equipment anomaly detection."""

import json
import logging
import re
import time
from typing import Any

from PIL import Image

from app.core.config import Settings, get_settings
from app.models.prompts import AnomalyCategory, get_detection_prompt
from app.models.qwen_vl import QwenVLModel, get_model
from app.schemas.detection import Detection, DetectionResult

logger = logging.getLogger(__name__)


class TrafficAnomalyDetector:
    """Detector for traffic equipment anomalies using Qwen3-VL."""

    def __init__(
        self,
        model: QwenVLModel | None = None,
        settings: Settings | None = None,
    ):
        """Initialize the detector.

        Args:
            model: Qwen-VL model instance. If None, uses global instance.
            settings: Application settings. If None, uses default settings.
        """
        self.model = model or get_model()
        self.settings = settings or get_settings()

    def ensure_model_loaded(self) -> None:
        """Ensure the model is loaded."""
        if not self.model.is_loaded:
            self.model.load()

    def detect(
        self,
        image: Image.Image,
        categories: list[AnomalyCategory] | None = None,
        quick_mode: bool = False,
        image_id: str | None = None,
    ) -> DetectionResult:
        """Detect anomalies in an image.

        Args:
            image: PIL Image to analyze.
            categories: Specific categories to detect. If None, detects all.
            quick_mode: Use quick mode for faster inference.
            image_id: Optional identifier for the image.

        Returns:
            DetectionResult with detected anomalies.
        """
        self.ensure_model_loaded()

        start_time = time.time()

        # Get appropriate prompt
        prompt = get_detection_prompt(categories, quick_mode)

        # Run inference
        try:
            response = self.model.generate(image, prompt)
            logger.debug(f"Model response: {response}")
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            return DetectionResult(
                has_anomaly=False,
                detections=[],
                summary=f"检测失败: {str(e)}",
                image_id=image_id,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Parse response
        result = self._parse_response(response)
        result.image_id = image_id
        result.processing_time_ms = (time.time() - start_time) * 1000

        # Filter by confidence threshold
        if self.settings.confidence_threshold > 0:
            result.detections = [
                d
                for d in result.detections
                if d.confidence >= self.settings.confidence_threshold
            ]
            result.has_anomaly = len(result.detections) > 0

        return result

    def detect_batch(
        self,
        images: list[Image.Image],
        categories: list[AnomalyCategory] | None = None,
        quick_mode: bool = False,
        image_ids: list[str] | None = None,
    ) -> list[DetectionResult]:
        """Detect anomalies in multiple images.

        Args:
            images: List of PIL Images.
            categories: Specific categories to detect.
            quick_mode: Use quick mode.
            image_ids: Optional list of image identifiers.

        Returns:
            List of DetectionResult for each image.
        """
        if image_ids is None:
            image_ids = [None] * len(images)

        results = []
        for image, image_id in zip(images, image_ids):
            result = self.detect(image, categories, quick_mode, image_id)
            results.append(result)

        return results

    def _parse_response(self, response: str) -> DetectionResult:
        """Parse model response to DetectionResult.

        Args:
            response: Raw model response text.

        Returns:
            Parsed DetectionResult.
        """
        # Try to extract JSON from response
        json_data = self._extract_json(response)

        if json_data is None:
            # Fallback: try to infer from text
            return self._parse_text_response(response)

        try:
            has_anomaly = json_data.get("has_anomaly", False)
            summary = json_data.get("summary", "")
            detections = []

            for det in json_data.get("detections", []):
                detection = Detection(
                    category=det.get("category", "unknown"),
                    anomaly_type=det.get("anomaly_type", "unknown"),
                    confidence=float(det.get("confidence", 0.5)),
                    bbox=det.get("bbox"),
                    description=det.get("description", ""),
                )
                detections.append(detection)

            return DetectionResult(
                has_anomaly=has_anomaly,
                detections=detections,
                summary=summary,
            )
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return self._parse_text_response(response)

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        """Extract JSON object from text.

        Args:
            text: Text that may contain JSON.

        Returns:
            Parsed JSON dict or None if not found.
        """
        # Try to find JSON block in markdown code blocks
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",  # ```json ... ```
            r"```\s*([\s\S]*?)\s*```",  # ``` ... ```
            r"\{[\s\S]*\}",  # Raw JSON object
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                json_str = match if isinstance(match, str) else match
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue

        return None

    def _parse_text_response(self, response: str) -> DetectionResult:
        """Parse non-JSON text response.

        Args:
            response: Raw text response.

        Returns:
            DetectionResult inferred from text.
        """
        response_lower = response.lower()

        # Check for anomaly indicators
        anomaly_keywords = [
            "异常",
            "损坏",
            "故障",
            "缺失",
            "破损",
            "不亮",
            "倾斜",
            "遮挡",
            "褪色",
            "乱码",
            "黑屏",
            "未关闭",
            "damaged",
            "broken",
            "missing",
            "malfunction",
        ]

        normal_keywords = [
            "正常",
            "未检测到异常",
            "没有异常",
            "状态良好",
            "no anomaly",
            "normal",
        ]

        has_anomaly = any(kw in response_lower for kw in anomaly_keywords)
        is_normal = any(kw in response_lower for kw in normal_keywords)

        if is_normal and not has_anomaly:
            has_anomaly = False

        return DetectionResult(
            has_anomaly=has_anomaly,
            detections=[],
            summary=response[:500] if len(response) > 500 else response,
        )


# Global detector instance
_detector_instance: TrafficAnomalyDetector | None = None


def get_detector() -> TrafficAnomalyDetector:
    """Get the global detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = TrafficAnomalyDetector()
    return _detector_instance
