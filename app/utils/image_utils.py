"""Image processing utilities."""

import base64
import io
import logging
from urllib.parse import urlparse

import httpx
from PIL import Image

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def load_image_from_base64(base64_str: str) -> Image.Image:
    """Load image from base64 encoded string.

    Args:
        base64_str: Base64 encoded image data.

    Returns:
        PIL Image object.

    Raises:
        ValueError: If decoding fails.
    """
    try:
        # Handle data URL format (e.g., "data:image/jpeg;base64,...")
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]

        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return image.convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}")


def load_image_from_url(url: str, timeout: float = 30.0) -> Image.Image:
    """Load image from URL.

    Args:
        url: URL of the image.
        timeout: Request timeout in seconds.

    Returns:
        PIL Image object.

    Raises:
        ValueError: If loading fails.
    """
    try:
        # Validate URL
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme: {parsed.scheme}")

        # Fetch image
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()

        image = Image.open(io.BytesIO(response.content))
        return image.convert("RGB")
    except httpx.HTTPError as e:
        raise ValueError(f"Failed to fetch image from URL: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load image from URL: {e}")


def load_image_from_path(path: str) -> Image.Image:
    """Load image from file path.

    Args:
        path: Path to the image file.

    Returns:
        PIL Image object.

    Raises:
        ValueError: If loading fails.
    """
    try:
        image = Image.open(path)
        return image.convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image from path: {e}")


def resize_image(
    image: Image.Image,
    max_size: int | None = None,
    maintain_aspect: bool = True,
) -> Image.Image:
    """Resize image if it exceeds maximum size.

    Args:
        image: PIL Image to resize.
        max_size: Maximum dimension. Uses settings default if None.
        maintain_aspect: Whether to maintain aspect ratio.

    Returns:
        Resized PIL Image.
    """
    if max_size is None:
        max_size = get_settings().max_image_size

    width, height = image.size

    if width <= max_size and height <= max_size:
        return image

    if maintain_aspect:
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
    else:
        new_width = min(width, max_size)
        new_height = min(height, max_size)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 string.

    Args:
        image: PIL Image to convert.
        format: Output format (JPEG, PNG, etc.).

    Returns:
        Base64 encoded string.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_image_info(image: Image.Image) -> dict:
    """Get image information.

    Args:
        image: PIL Image.

    Returns:
        Dict with width, height, mode, and format.
    """
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format,
    }
