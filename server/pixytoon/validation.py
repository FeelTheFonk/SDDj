"""Shared input validation utilities."""

from __future__ import annotations

import re

_SAFE_NAME = re.compile(r'^[\w\-. ]+$')


def validate_resource_name(name: str, kind: str) -> None:
    """Reject names with path traversal characters."""
    if not name or not _SAFE_NAME.match(name) or '..' in name:
        raise ValueError(f"Invalid {kind} name: {name!r}")
