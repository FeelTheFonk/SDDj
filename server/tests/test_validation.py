"""Tests for input validation utilities."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from sddj.validation import validate_resource_name, validate_path_in_sandbox


class TestValidateResourceName:
    @pytest.mark.parametrize("name", [
        "pixel_art", "my-preset", "Test Preset", "lora_v1.2",
        "a", "CamelCase", "with spaces",
    ])
    def test_valid_names(self, name):
        validate_resource_name(name, "test")  # Should not raise

    @pytest.mark.parametrize("name", [
        "", "../evil", "..\\bad", "../../etc/passwd",
        "bad/name", "bad:name", "bad\x00name",
    ])
    def test_invalid_names(self, name):
        with pytest.raises(ValueError):
            validate_resource_name(name, "test")

    def test_too_long_name(self):
        with pytest.raises(ValueError):
            validate_resource_name("a" * 257, "test")

    def test_max_length_ok(self):
        validate_resource_name("a" * 256, "test")  # Should not raise

    def test_dotdot_in_name(self):
        with pytest.raises(ValueError):
            validate_resource_name("foo..bar", "test")


class TestValidatePathInSandbox:
    def test_valid_path_inside_sandbox(self, tmp_path):
        child = tmp_path / "subdir" / "file.txt"
        child.parent.mkdir(parents=True, exist_ok=True)
        child.touch()
        validate_path_in_sandbox(child, tmp_path)  # Should not raise

    def test_path_traversal_rejected(self, tmp_path):
        evil = tmp_path / ".." / "etc" / "passwd"
        with pytest.raises(ValueError, match="escapes sandbox"):
            validate_path_in_sandbox(evil, tmp_path)

    def test_symlink_escape_rejected(self, tmp_path):
        target = Path(tempfile.gettempdir()) / "outside_sandbox.txt"
        target.touch(exist_ok=True)
        try:
            link = tmp_path / "sneaky_link"
            link.symlink_to(target)
            with pytest.raises(ValueError, match="escapes sandbox"):
                validate_path_in_sandbox(link, tmp_path)
        finally:
            target.unlink(missing_ok=True)

    def test_same_as_sandbox_is_valid(self, tmp_path):
        validate_path_in_sandbox(tmp_path, tmp_path)  # Should not raise
