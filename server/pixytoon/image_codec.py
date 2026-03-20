"""Image encode/decode/resize utilities for the diffusion engine."""

from __future__ import annotations

from base64 import b64decode, b64encode
from io import BytesIO

from PIL import Image


def round8(v: int) -> int:
    """Round to nearest multiple of 8 (SD1.5 VAE requirement)."""
    return ((v + 4) // 8) * 8


def decode_b64_image(data: str) -> Image.Image:
    """Decode a base64-encoded PNG into a PIL Image (RGB)."""
    try:
        raw = b64decode(data)
        return Image.open(BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}") from e


def encode_image_b64(image: Image.Image) -> str:
    """Encode a PIL Image to base64 PNG string."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    return b64encode(buf.getvalue()).decode("ascii")


def resize_to_target(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize image to target dimensions if sizes differ (LANCZOS)."""
    if image.size != (width, height):
        return image.resize((width, height), Image.LANCZOS)
    return image
