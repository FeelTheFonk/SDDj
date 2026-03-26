"""Tests for QR Code generation and scan validation."""

from __future__ import annotations

import pytest
from PIL import Image

from sddj.qrcode_generator import (
    generate_qr_image,
    validate_qr_scannable,
    generate_and_validate,
    MAX_QR_CONTENT_BYTES,
)


class TestGenerateQrImage:
    def test_returns_pil_image(self):
        img = generate_qr_image("https://example.com")
        assert isinstance(img, Image.Image)

    def test_correct_size(self):
        img = generate_qr_image("test", target_width=512, target_height=512)
        assert img.size == (512, 512)

    def test_correct_mode_rgb(self):
        img = generate_qr_image("test")
        assert img.mode == "RGB"

    def test_default_size_768(self):
        img = generate_qr_image("test")
        assert img.size == (768, 768)

    def test_non_square(self):
        img = generate_qr_image("test", target_width=512, target_height=768)
        assert img.size == (512, 768)

    def test_error_correction_levels(self):
        for ec in ("L", "M", "Q", "H"):
            img = generate_qr_image("test", error_correction=ec)
            assert img.size == (768, 768)

    def test_module_size_variations(self):
        for ms in (4, 8, 16, 32):
            img = generate_qr_image("test", module_size=ms)
            assert isinstance(img, Image.Image)

    def test_long_url(self):
        url = "https://example.com/" + "a" * 200
        img = generate_qr_image(url)
        assert isinstance(img, Image.Image)

    def test_short_text(self):
        img = generate_qr_image("hi")
        assert isinstance(img, Image.Image)

    def test_gray_background(self):
        """Canvas corners should be gray (#808080) since QR is centered."""
        img = generate_qr_image("test", target_width=1024, target_height=1024)
        corner = img.getpixel((0, 0))
        assert corner == (128, 128, 128)

    def test_1024_resolution(self):
        img = generate_qr_image("test", target_width=1024, target_height=1024)
        assert img.size == (1024, 1024)


class TestValidateQrScannable:
    def test_validates_correct_content(self):
        """Generated QR control image should be scannable."""
        content = "https://github.com"
        img = generate_qr_image(content, target_width=768, target_height=768)
        assert validate_qr_scannable(img, content) is True

    def test_rejects_wrong_content(self):
        content = "https://github.com"
        img = generate_qr_image(content)
        assert validate_qr_scannable(img, "https://wrong.com") is False

    def test_blank_image_fails(self):
        blank = Image.new("RGB", (512, 512), (128, 128, 128))
        assert validate_qr_scannable(blank, "anything") is False


class TestGenerateAndValidate:
    def test_returns_tuple(self):
        img, is_valid = generate_and_validate("https://example.com")
        assert isinstance(img, Image.Image)
        assert isinstance(is_valid, bool)

    def test_valid_url(self):
        _, is_valid = generate_and_validate("https://example.com")
        assert is_valid is True


class TestConstants:
    def test_max_content_bytes(self):
        assert MAX_QR_CONTENT_BYTES == 1273
