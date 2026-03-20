"""DeepCache lifecycle management — setup and context manager for toggle."""

from __future__ import annotations

import logging
from contextlib import contextmanager

from .config import settings

log = logging.getLogger("pixytoon.deepcache")


def create_helper(pipe):
    """Create and enable a DeepCacheSDHelper for the given pipeline.

    Returns the helper on success, None if DeepCache is unavailable.
    """
    if not settings.enable_deepcache:
        return None
    try:
        from DeepCache import DeepCacheSDHelper
        helper = DeepCacheSDHelper(pipe=pipe)
        helper.set_params(
            cache_interval=settings.deepcache_interval,
            cache_branch_id=settings.deepcache_branch,
        )
        helper.enable()
        log.info("DeepCache enabled (interval=%d)", settings.deepcache_interval)
        return helper
    except Exception as e:
        log.warning("DeepCache unavailable: %s", e)
        return None


@contextmanager
def suspended(helper):
    """Context manager: disable DeepCache on enter, re-enable on exit.

    Raises if disable fails — AnimateDiff/chain MUST run without DeepCache
    (5D latent shapes are incompatible with DeepCache's 4D caching).
    """
    if helper is None:
        yield
        return

    try:
        helper.disable()
        log.info("DeepCache temporarily disabled")
    except Exception:
        log.warning("Failed to disable DeepCache — proceeding anyway")
        yield
        return

    try:
        yield
    finally:
        try:
            helper.enable()
            log.info("DeepCache re-enabled")
        except Exception as e:
            log.warning("Failed to re-enable DeepCache: %s", e)
