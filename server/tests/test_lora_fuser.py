"""Tests for LoRA fuser lifecycle — snapshot, restore, hotswap, dynamo reset."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch
from collections import OrderedDict

import pytest


@pytest.fixture
def mock_settings():
    """Provide mock settings with all required fields."""
    with patch("sddj.lora_fuser.settings") as ms:
        ms.enable_torch_compile = True
        ms.enable_lora_hotswap = True
        ms.loras_dir = MagicMock()
        yield ms


@pytest.fixture
def mock_pipe():
    """Provide a mock pipeline with UNet state dict.

    _orig_mod is set to the unet itself so _get_raw_module() is a no-op
    (MagicMock always returns True for hasattr, so we must be explicit).
    """
    pipe = MagicMock()
    # Simulate a small UNet state dict
    import torch
    state = OrderedDict({
        "conv.weight": torch.randn(3, 3),
        "conv.bias": torch.randn(3),
    })
    pipe.unet._orig_mod = pipe.unet  # _get_raw_module passthrough
    pipe.unet.state_dict.return_value = state
    _unet_param = torch.randn(3, 3, device="cpu")
    pipe.unet.parameters.side_effect = lambda: iter([_unet_param])
    pipe.unet.named_parameters.side_effect = lambda: iter([
        ("conv.weight", _unet_param),
    ])
    pipe.unet.named_buffers.side_effect = lambda: iter([])
    # text_encoder
    te_state = OrderedDict({"embed.weight": torch.randn(4, 4)})
    pipe.text_encoder._orig_mod = pipe.text_encoder
    pipe.text_encoder.state_dict.return_value = te_state
    _te_param = torch.randn(4, 4, device="cpu")
    pipe.text_encoder.parameters.side_effect = lambda: iter([_te_param])
    pipe.text_encoder.named_parameters.side_effect = lambda: iter([
        ("embed.weight", _te_param),
    ])
    pipe.text_encoder.named_buffers.side_effect = lambda: iter([])
    return pipe


def test_snapshot_captured_once(mock_settings, mock_pipe):
    """Weight snapshot is captured only once, before first style LoRA fuse."""
    from sddj.lora_fuser import LoRAFuser
    fuser = LoRAFuser()
    fuser._ensure_snapshot(mock_pipe)
    assert fuser._original_unet_state is not None
    first_snapshot = fuser._original_unet_state

    fuser._ensure_snapshot(mock_pipe)
    assert fuser._original_unet_state is first_snapshot  # same object — no re-capture


def test_restore_weights(mock_settings, mock_pipe):
    """Restore loads the snapshot back into UNet."""
    from sddj.lora_fuser import LoRAFuser
    fuser = LoRAFuser()
    fuser._ensure_snapshot(mock_pipe)
    fuser._restore_weights(mock_pipe)
    mock_pipe.unet.load_state_dict.assert_called_once()


def test_restore_uses_assign_false(mock_settings, mock_pipe):
    """Restore must use assign=False (default) to preserve tensor identity for torch.compile."""
    from sddj.lora_fuser import LoRAFuser
    fuser = LoRAFuser()
    fuser._ensure_snapshot(mock_pipe)
    fuser._restore_weights(mock_pipe)
    # load_state_dict called without assign=True — preserves compiled graph refs
    call_kwargs = mock_pipe.unet.load_state_dict.call_args
    assert "assign" not in (call_kwargs.kwargs or {}), \
        "load_state_dict must NOT pass assign=True (breaks torch.compile tensor refs)"


def test_restore_no_snapshot(mock_settings, mock_pipe):
    """Restore with no snapshot is a no-op."""
    from sddj.lora_fuser import LoRAFuser
    fuser = LoRAFuser()
    fuser._restore_weights(mock_pipe)  # should not raise
    mock_pipe.unet.load_state_dict.assert_not_called()


def test_needs_dynamo_reset_with_hotswap(mock_settings):
    """Dynamo reset NOT needed when hotswap is enabled."""
    from sddj.lora_fuser import LoRAFuser
    mock_settings.enable_lora_hotswap = True
    mock_settings.enable_torch_compile = True
    fuser = LoRAFuser()
    assert fuser._needs_dynamo_reset() is False


def test_needs_dynamo_reset_without_hotswap(mock_settings):
    """Dynamo reset IS needed when hotswap is disabled."""
    from sddj.lora_fuser import LoRAFuser
    mock_settings.enable_lora_hotswap = False
    mock_settings.enable_torch_compile = True
    fuser = LoRAFuser()
    assert fuser._needs_dynamo_reset() is True


def test_needs_dynamo_reset_no_compile(mock_settings):
    """Dynamo reset NOT needed when torch.compile disabled."""
    from sddj.lora_fuser import LoRAFuser
    mock_settings.enable_lora_hotswap = False
    mock_settings.enable_torch_compile = False
    fuser = LoRAFuser()
    assert fuser._needs_dynamo_reset() is False


def test_get_raw_module_unwraps_compiled(mock_settings):
    """_get_raw_module returns _orig_mod from OptimizedModule."""
    from sddj.lora_fuser import _get_raw_module
    inner = MagicMock(name="raw_module")
    wrapper = MagicMock(name="compiled_module")
    wrapper._orig_mod = inner
    assert _get_raw_module(wrapper) is inner


def test_get_raw_module_passthrough(mock_settings):
    """_get_raw_module returns module as-is when not compiled."""
    from sddj.lora_fuser import _get_raw_module
    module = MagicMock(spec=[])  # no _orig_mod
    assert _get_raw_module(module) is module


def test_snapshot_captures_text_encoder(mock_settings, mock_pipe):
    """Snapshot captures both UNet and text_encoder states."""
    from sddj.lora_fuser import LoRAFuser

    fuser = LoRAFuser()
    fuser._ensure_snapshot(mock_pipe)

    assert fuser._original_unet_state is not None
    assert fuser._original_te_state is not None
    assert "conv.weight" in fuser._original_unet_state
    assert "embed.weight" in fuser._original_te_state


def test_restore_weights_both_modules(mock_settings, mock_pipe):
    """Restore loads snapshot into both UNet and text_encoder."""
    from sddj.lora_fuser import LoRAFuser

    fuser = LoRAFuser()
    fuser._ensure_snapshot(mock_pipe)
    fuser._restore_weights(mock_pipe)

    mock_pipe.unet.load_state_dict.assert_called_once()
    mock_pipe.text_encoder.load_state_dict.assert_called_once()
