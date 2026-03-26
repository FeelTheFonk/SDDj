"""QR Code image generation + scan validation for ControlNet conditioning."""

import logging

import qrcode
from PIL import Image

log = logging.getLogger("sddj.qrcode")

_EC_MAP = {
    "L": qrcode.constants.ERROR_CORRECT_L,
    "M": qrcode.constants.ERROR_CORRECT_M,
    "Q": qrcode.constants.ERROR_CORRECT_Q,
    "H": qrcode.constants.ERROR_CORRECT_H,
}

# QR v40 capacity at EC=H (bytes). Content exceeding this will still
# work if a lower EC level is selected, but we use this as the hard cap
# since SDDj defaults to EC=H.
MAX_QR_CONTENT_BYTES = 1273


def generate_qr_image(
    content: str,
    target_width: int = 768,
    target_height: int = 768,
    error_correction: str = "H",
    module_size: int = 16,
) -> Image.Image:
    """Generate QR code on #808080 gray canvas for ControlNet conditioning.

    Design per QR Code Monster v2 model card:
      - Gray (#808080) background for seamless blending
      - Module size 16px (training distribution)
      - NEAREST resampling preserves sharp module edges
      - 4-block quiet zone per ISO 18004
      - 85% canvas fill leaves scanner-friendly margin
    """
    ec = _EC_MAP.get(error_correction.upper(), qrcode.constants.ERROR_CORRECT_H)
    qr = qrcode.QRCode(
        version=None,
        error_correction=ec,
        box_size=module_size,
        border=4,
    )
    qr.add_data(content)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

    canvas = Image.new("RGB", (target_width, target_height), (128, 128, 128))
    qr_w, qr_h = qr_img.size
    max_dim = int(min(target_width, target_height) * 0.85)
    if max(qr_w, qr_h) > max_dim:
        scale = max_dim / max(qr_w, qr_h)
        qr_img = qr_img.resize(
            (int(qr_w * scale), int(qr_h * scale)), Image.NEAREST
        )
        qr_w, qr_h = qr_img.size

    x = (target_width - qr_w) // 2
    y = (target_height - qr_h) // 2
    canvas.paste(qr_img, (x, y))
    return canvas


def validate_qr_scannable(
    image: Image.Image,
    expected_content: str,
) -> bool:
    """Verify generated image contains a scannable QR matching expected content.

    Uses OpenCV QRCodeDetector (opencv-python-headless already a project dep).
    """
    import cv2
    import numpy as np

    arr = np.array(image.convert("RGB"))
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(arr)
    if data and data.strip() == expected_content.strip():
        return True
    log.warning("QR scan validation failed: decoded=%r expected=%r", data, expected_content)
    return False


def generate_and_validate(
    content: str,
    target_width: int = 768,
    target_height: int = 768,
    error_correction: str = "H",
    module_size: int = 16,
) -> tuple[Image.Image, bool]:
    """Generate QR control image and validate it scans correctly.

    Returns (image, is_valid).
    """
    img = generate_qr_image(content, target_width, target_height,
                            error_correction, module_size)
    is_valid = validate_qr_scannable(img, content)
    return img, is_valid
