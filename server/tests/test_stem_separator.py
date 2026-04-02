"""Tests for stem_separator — demucs/roformer availability check."""

from __future__ import annotations

import pytest

from sddj.stem_separator import (
    STEM_NAMES,
    DEMUCS_STEM_NAMES,
    ROFORMER_STEM_NAMES,
    StemSeparator,
    is_available,
)


class TestIsAvailable:
    def test_returns_bool(self):
        result = is_available()
        assert isinstance(result, bool)


class TestStemSeparator:
    def test_init(self):
        sep = StemSeparator(model_name="htdemucs", device="cpu")
        assert sep._model_name == "htdemucs"
        assert sep._device == "cpu"

    def test_is_available_method(self):
        sep = StemSeparator()
        assert isinstance(sep.is_available(), bool)

    def test_unload_when_not_loaded(self):
        sep = StemSeparator()
        sep.unload()  # should not raise

    def test_separate_file_not_found(self):
        sep = StemSeparator()
        if not sep.is_available():
            pytest.skip("demucs/roformer not installed")
        with pytest.raises(FileNotFoundError):
            sep.separate("/nonexistent/file.wav")


class TestStemBackendDispatch:

    def test_default_backend_is_demucs(self):
        from sddj.stem_separator import StemSeparator, _DemucsBackend
        sep = StemSeparator()
        backend = sep._get_backend()
        assert isinstance(backend, _DemucsBackend)

    def test_roformer_fallback_to_demucs(self):
        """When roformer requested but not installed, falls back to demucs."""
        from unittest.mock import patch
        from sddj.stem_separator import StemSeparator, _DemucsBackend
        with patch("sddj.stem_separator.settings") as mock_settings:
            mock_settings.stem_backend = "roformer"
            mock_settings.stem_device = "cpu"
            mock_settings.stem_model = "htdemucs"
            with patch("sddj.stem_separator._is_roformer_available", return_value=False):
                sep = StemSeparator()
                backend = sep._get_backend()
                assert isinstance(backend, _DemucsBackend)


class TestStemNames:
    def test_demucs_stems(self):
        assert set(DEMUCS_STEM_NAMES) == {"drums", "bass", "vocals", "other"}

    def test_roformer_stems(self):
        assert set(ROFORMER_STEM_NAMES) == {"drums", "bass", "vocals", "other", "guitar", "piano"}

    def test_stem_names_is_superset(self):
        """STEM_NAMES (public alias) contains all stems from both backends."""
        assert set(DEMUCS_STEM_NAMES).issubset(set(STEM_NAMES))
        assert set(ROFORMER_STEM_NAMES).issubset(set(STEM_NAMES))
