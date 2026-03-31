"""Tests for post-processing pipeline — pixelation, quantize, palette, dither."""

from __future__ import annotations

import numpy as np
from PIL import Image

from sddj.postprocess import apply
from sddj.protocol import (
    DitherMode,
    PaletteMode,
    PaletteSpec,
    PixelateMethod,
    PixelateSpec,
    PostProcessSpec,
    QuantizeMethod,
)


def _make_test_image(w=128, h=128, mode="RGBA"):
    """Create a colorful test image."""
    arr = np.random.randint(0, 255, (h, w, 4 if mode == "RGBA" else 3), dtype=np.uint8)
    if mode == "RGBA":
        arr[:, :, 3] = 255
    return Image.fromarray(arr)


def _make_alpha_test_image(w=64, h=64):
    """Create image with mixed transparent/opaque regions."""
    arr = np.random.randint(0, 255, (h, w, 4), dtype=np.uint8)
    # Top half opaque, bottom half transparent
    arr[:h // 2, :, 3] = 255
    arr[h // 2:, :, 3] = 0
    return Image.fromarray(arr)


class TestPixelation:
    def test_pixelate_enabled(self):
        np.random.seed(42)
        img = _make_test_image(512, 512)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=True, target_size=64),
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        assert result.size == (64, 64)

    def test_pixelate_disabled(self):
        np.random.seed(42)
        img = _make_test_image(128, 128)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        assert result.size == (128, 128)

    def test_pixelate_rectangular(self):
        np.random.seed(42)
        img = _make_test_image(256, 128)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=True, target_size=64),
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        w, h = result.size
        assert max(w, h) == 64

    def test_pixelate_box_method(self):
        """BOX pixelation should produce different results than NEAREST."""
        np.random.seed(42)
        img = _make_test_image(256, 256)
        spec_nearest = PostProcessSpec(
            pixelate=PixelateSpec(enabled=True, target_size=32, method=PixelateMethod.NEAREST),
            dither=DitherMode.NONE,
        )
        spec_box = PostProcessSpec(
            pixelate=PixelateSpec(enabled=True, target_size=32, method=PixelateMethod.BOX),
            dither=DitherMode.NONE,
        )
        result_nearest = apply(img, spec_nearest)
        result_box = apply(img, spec_box)
        assert result_nearest.size == (32, 32)
        assert result_box.size == (32, 32)
        # BOX averaging produces smoother results — verify images differ
        arr_n = np.array(result_nearest)
        arr_b = np.array(result_box)
        assert not np.array_equal(arr_n, arr_b)

    def test_pixelate_box_default_nearest(self):
        """Default method should be NEAREST for backward compatibility."""
        spec = PixelateSpec(enabled=True, target_size=64)
        assert spec.method == PixelateMethod.NEAREST


class TestQuantization:
    def test_kmeans_reduces_colors(self):
        np.random.seed(42)
        img = _make_test_image(64, 64)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            quantize_enabled=True,
            quantize_method=QuantizeMethod.KMEANS,
            quantize_colors=8,
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        colors = set()
        for px in result.getdata():
            colors.add(px[:3])
        assert len(colors) <= 10  # Allow some slack for alpha compositing

    def test_octree_method(self):
        np.random.seed(42)
        img = _make_test_image(64, 64)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            quantize_enabled=True,
            quantize_method=QuantizeMethod.OCTREE,
            quantize_colors=16,
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        assert result.size == (64, 64)

    def test_median_cut_method(self):
        np.random.seed(42)
        img = _make_test_image(64, 64)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            quantize_enabled=True,
            quantize_method=QuantizeMethod.MEDIAN_CUT,
            quantize_colors=16,
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        assert result.size == (64, 64)

    def test_octree_lab_method(self):
        """OCTREE_LAB should produce CIELAB-perceptual palettes."""
        np.random.seed(42)
        img = _make_test_image(64, 64)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            quantize_enabled=True,
            quantize_method=QuantizeMethod.OCTREE_LAB,
            quantize_colors=8,
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        assert result.size == (64, 64)
        colors = set()
        for px in result.getdata():
            colors.add(px[:3])
        assert len(colors) <= 10

    def test_octree_lab_with_alpha(self):
        """OCTREE_LAB must preserve alpha channel."""
        np.random.seed(42)
        img = _make_alpha_test_image(64, 64)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            quantize_enabled=True,
            quantize_method=QuantizeMethod.OCTREE_LAB,
            quantize_colors=8,
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        assert result.mode == "RGBA"
        arr = np.array(result)
        # Bottom half alpha should still be 0
        assert np.all(arr[32:, :, 3] == 0)


class TestDithering:
    def test_floyd_steinberg(self):
        np.random.seed(42)
        img = _make_test_image(64, 64)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            quantize_enabled=True,
            quantize_colors=8,
            dither=DitherMode.FLOYD_STEINBERG,
        )
        result = apply(img, spec)
        assert result.size == (64, 64)

    def test_bayer_4x4(self):
        np.random.seed(42)
        img = _make_test_image(64, 64)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            quantize_enabled=True,
            quantize_colors=8,
            dither=DitherMode.BAYER_4X4,
        )
        result = apply(img, spec)
        assert result.size == (64, 64)

    def test_no_dither(self):
        np.random.seed(42)
        img = _make_test_image(64, 64)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        assert result.size == (64, 64)

    def test_bayer_2_colors(self):
        """v0.7.9: Bayer dithering with 2-color palette should not crash."""
        np.random.seed(42)
        img = _make_test_image(32, 32)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            quantize_enabled=True,
            quantize_colors=2,
            dither=DitherMode.BAYER_4X4,
            palette=PaletteSpec(
                mode=PaletteMode.CUSTOM,
                colors=["#000000", "#FFFFFF"],
            ),
        )
        result = apply(img, spec)
        assert result.size == (32, 32)

    def test_floyd_steinberg_alpha_aware(self):
        """Alpha-aware FS dithering must not dither transparent pixels."""
        np.random.seed(42)
        img = _make_alpha_test_image(32, 32)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            quantize_enabled=True,
            quantize_colors=4,
            dither=DitherMode.FLOYD_STEINBERG,
            remove_bg=True,
        )
        result = apply(img, spec)
        assert result.mode == "RGBA"

    def test_bayer_alpha_aware(self):
        """Alpha-aware Bayer dithering must handle transparent regions."""
        np.random.seed(42)
        img = _make_alpha_test_image(32, 32)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            quantize_enabled=True,
            quantize_colors=4,
            dither=DitherMode.BAYER_8X8,
            remove_bg=True,
        )
        result = apply(img, spec)
        assert result.mode == "RGBA"

    def test_dither_single_color_palette_noop(self):
        """Dithering with 1-color palette must not crash (degenerate case)."""
        np.random.seed(42)
        img = _make_test_image(16, 16)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            quantize_enabled=False,
            dither=DitherMode.FLOYD_STEINBERG,
            palette=PaletteSpec(
                mode=PaletteMode.CUSTOM,
                colors=["#FF0000"],
            ),
        )
        result = apply(img, spec)
        assert result.size == (16, 16)


class TestPalette:
    def test_auto_palette(self):
        np.random.seed(42)
        img = _make_test_image(64, 64)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            palette=PaletteSpec(mode=PaletteMode.AUTO),
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        assert result.size == (64, 64)

    def test_custom_palette(self):
        np.random.seed(42)
        img = _make_test_image(64, 64)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            palette=PaletteSpec(
                mode=PaletteMode.CUSTOM,
                colors=["#FF0000", "#00FF00", "#0000FF", "#FFFFFF", "#000000"],
            ),
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        assert result.size == (64, 64)

    def test_custom_palette_enforces_colors(self):
        """Custom palette must snap all pixels to palette colors."""
        np.random.seed(42)
        img = _make_test_image(32, 32)
        target_colors = ["#FF0000", "#00FF00", "#0000FF"]
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            palette=PaletteSpec(
                mode=PaletteMode.CUSTOM,
                colors=target_colors,
            ),
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        expected = {(255, 0, 0), (0, 255, 0), (0, 0, 255)}
        actual_colors = set()
        for px in result.getdata():
            actual_colors.add(px[:3])
        assert actual_colors.issubset(expected)


class TestFullPipeline:
    def test_complete_postprocess_pipeline(self):
        np.random.seed(42)
        img = _make_test_image(512, 512)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=True, target_size=64),
            quantize_enabled=True,
            quantize_method=QuantizeMethod.KMEANS,
            quantize_colors=16,
            dither=DitherMode.FLOYD_STEINBERG,
            palette=PaletteSpec(mode=PaletteMode.AUTO),
            remove_bg=False,
        )
        result = apply(img, spec)
        assert result.size == (64, 64)
        assert result.mode == "RGBA"

    def test_passthrough_no_processing(self):
        """With all processing disabled, image must be returned untouched (pixel-perfect)."""
        np.random.seed(42)
        img = _make_test_image(128, 128)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            quantize_enabled=False,
            dither=DitherMode.NONE,
            palette=PaletteSpec(mode=PaletteMode.AUTO),
            remove_bg=False,
        )
        result = apply(img, spec)
        assert result.size == (128, 128)
        assert result is img

    def test_default_spec_is_passthrough(self):
        """Default PostProcessSpec must not alter the image (raw SD output)."""
        np.random.seed(42)
        img = _make_test_image(128, 128)
        spec = PostProcessSpec()
        result = apply(img, spec)
        assert result is img

    def test_quantize_disabled_preserves_all_colors(self):
        """With quantize_enabled=False, all original colors must survive."""
        np.random.seed(42)
        img = _make_test_image(64, 64)
        original_colors = set(img.convert("RGBA").getdata())
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            quantize_enabled=False,
            dither=DitherMode.NONE,
            palette=PaletteSpec(mode=PaletteMode.AUTO),
            remove_bg=False,
        )
        result = apply(img, spec)
        assert result is img
        result_colors = set(result.convert("RGBA").getdata())
        assert original_colors == result_colors

    def test_full_pipeline_box_octreelab_fs(self):
        """Full pipeline: BOX + OCTREE_LAB + Floyd-Steinberg + custom palette."""
        np.random.seed(42)
        img = _make_test_image(256, 256)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=True, target_size=32, method=PixelateMethod.BOX),
            quantize_enabled=True,
            quantize_method=QuantizeMethod.OCTREE_LAB,
            quantize_colors=8,
            dither=DitherMode.FLOYD_STEINBERG,
            palette=PaletteSpec(
                mode=PaletteMode.CUSTOM,
                colors=["#1a1c2c", "#5d275d", "#b13e53", "#ef7d57",
                         "#ffcd75", "#a7f070", "#38b764", "#257179"],
            ),
            remove_bg=False,
        )
        result = apply(img, spec)
        assert result.size == (32, 32)
        assert result.mode == "RGBA"
        # All output colors must be from the 8-color palette
        expected = {
            (0x1a, 0x1c, 0x2c), (0x5d, 0x27, 0x5d), (0xb1, 0x3e, 0x53), (0xef, 0x7d, 0x57),
            (0xff, 0xcd, 0x75), (0xa7, 0xf0, 0x70), (0x38, 0xb7, 0x64), (0x25, 0x71, 0x79),
        }
        actual_colors = set()
        for px in result.getdata():
            actual_colors.add(px[:3])
        assert actual_colors.issubset(expected)
