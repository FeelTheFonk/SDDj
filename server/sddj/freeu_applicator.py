"""FreeU v2 application — single source of truth for all pipelines."""

from __future__ import annotations

import logging

from .config import settings

log = logging.getLogger("sddj.freeu")

_FREEU_APPLIED: set[int] = set()


def apply_freeu(pipe) -> None:
    """Apply FreeU v2 to any diffusers pipeline if enabled in settings."""
    if not settings.enable_freeu:
        return
    pipe_id = id(pipe)
    if pipe_id in _FREEU_APPLIED:
        return
    try:
        pipe.enable_freeu(
            s1=settings.freeu_s1,
            s2=settings.freeu_s2,
            b1=settings.freeu_b1,
            b2=settings.freeu_b2,
        )
        _FREEU_APPLIED.add(pipe_id)
    except Exception as e:
        log.warning("FreeU v2 unavailable for %s: %s", type(pipe).__name__, e)
