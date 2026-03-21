"""Tests for protocol models — validation, enums, conversions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pixytoon.protocol import (
    Action,
    AnimationRequest,
    CleanupResponse,
    DitherMode,
    ErrorResponse,
    GenerateRequest,
    GenerationMode,
    ListResponse,
    PaletteMode,
    PixelateSpec,
    PongResponse,
    PostProcessSpec,
    PresetDeletedResponse,
    PresetResponse,
    PresetSavedResponse,
    ProgressResponse,
    PromptResultResponse,
    QuantizeMethod,
    RealtimeFrameRequest,
    RealtimeReadyResponse,
    RealtimeResultResponse,
    RealtimeStartRequest,
    RealtimeUpdateRequest,
    Request,
    ResultResponse,
    SeedStrategy,
)


class TestAction:
    def test_all_actions_exist(self):
        expected = {
            "generate", "generate_animation", "cancel",
            "list_loras", "list_palettes", "list_controlnets", "list_embeddings",
            "ping",
            "realtime_start", "realtime_frame", "realtime_update", "realtime_stop",
            "generate_prompt", "list_presets", "get_preset", "save_preset", "delete_preset",
            "cleanup",
        }
        actual = {a.value for a in Action}
        assert expected == actual

    def test_action_from_string(self):
        assert Action("generate") == Action.GENERATE
        assert Action("cleanup") == Action.CLEANUP


class TestGenerateRequest:
    def test_defaults(self):
        req = GenerateRequest(prompt="test")
        assert req.width == 512
        assert req.height == 512
        assert req.steps == 8
        assert req.seed == -1
        assert req.mode == GenerationMode.TXT2IMG

    def test_img2img_requires_source(self):
        with pytest.raises(ValidationError, match="source_image"):
            GenerateRequest(prompt="test", mode="img2img")

    def test_inpaint_requires_mask(self):
        with pytest.raises(ValidationError, match="mask_image"):
            GenerateRequest(prompt="test", mode="inpaint", source_image="base64data")

    def test_controlnet_requires_control_image(self):
        with pytest.raises(ValidationError, match="control_image"):
            GenerateRequest(prompt="test", mode="controlnet_canny")

    def test_valid_img2img(self):
        req = GenerateRequest(prompt="test", mode="img2img", source_image="data")
        assert req.source_image == "data"

    def test_size_bounds(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", width=32)
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", width=4096)

    def test_steps_bounds(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", steps=0)
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", steps=101)

    def test_cfg_bounds(self):
        req = GenerateRequest(prompt="test", cfg_scale=0.0)
        assert req.cfg_scale == 0.0
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", cfg_scale=31.0)


class TestAnimationRequest:
    def test_defaults(self):
        req = AnimationRequest(prompt="test")
        assert req.frame_count == 8
        assert req.frame_duration_ms == 100
        assert req.denoise_strength == 0.30

    def test_seed_strategies(self):
        for s in ("fixed", "increment", "random"):
            req = AnimationRequest(prompt="test", seed_strategy=s)
            assert req.seed_strategy == SeedStrategy(s)


class TestRealtimeModels:
    def test_start_defaults(self):
        req = RealtimeStartRequest(prompt="test")
        assert req.steps == 4
        assert req.cfg_scale == 2.5
        assert req.denoise_strength == 0.5

    def test_frame_with_roi(self):
        req = RealtimeFrameRequest(
            image="b64data", frame_id=5,
            roi_x=10, roi_y=20, roi_w=100, roi_h=50,
            mask="maskb64",
        )
        assert req.roi_x == 10
        assert req.roi_w == 100
        assert req.mask == "maskb64"

    def test_frame_without_roi(self):
        req = RealtimeFrameRequest(image="b64data")
        assert req.roi_x is None
        assert req.mask is None

    def test_update_partial(self):
        req = RealtimeUpdateRequest(denoise_strength=0.7)
        assert req.prompt is None
        assert req.denoise_strength == 0.7


class TestPostProcessSpec:
    def test_defaults(self):
        pp = PostProcessSpec()
        assert pp.pixelate.enabled is True
        assert pp.quantize_colors == 32
        assert pp.dither == DitherMode.NONE
        assert pp.palette.mode == PaletteMode.AUTO

    def test_custom_palette(self):
        pp = PostProcessSpec(palette={"mode": "custom", "colors": ["#FF0000"]})
        assert pp.palette.mode == PaletteMode.CUSTOM
        assert pp.palette.colors == ["#FF0000"]


class TestRequestConversions:
    def test_to_generate_request(self):
        req = Request(
            action="generate", prompt="hello",
            width=768, height=512, steps=12,
        )
        gen = req.to_generate_request()
        assert isinstance(gen, GenerateRequest)
        assert gen.prompt == "hello"
        assert gen.width == 768

    def test_to_animation_request(self):
        req = Request(
            action="generate_animation", prompt="anim",
            frame_count=16, seed_strategy="random",
        )
        anim = req.to_animation_request()
        assert isinstance(anim, AnimationRequest)
        assert anim.frame_count == 16

    def test_to_realtime_start(self):
        req = Request(
            action="realtime_start", prompt="live",
            steps=4, cfg_scale=2.5,
        )
        rt = req.to_realtime_start()
        assert isinstance(rt, RealtimeStartRequest)
        assert rt.steps == 4

    def test_to_realtime_frame(self):
        req = Request(
            action="realtime_frame", image="b64",
            frame_id=3, roi_x=10, roi_y=20, roi_w=64, roi_h=64,
        )
        rf = req.to_realtime_frame()
        assert isinstance(rf, RealtimeFrameRequest)
        assert rf.roi_x == 10
        assert rf.frame_id == 3

    def test_to_realtime_update(self):
        req = Request(action="realtime_update", prompt="new", steps=6)
        ru = req.to_realtime_update()
        assert isinstance(ru, RealtimeUpdateRequest)
        assert ru.prompt == "new"
        assert ru.steps == 6


class TestResponseModels:
    def test_progress(self):
        r = ProgressResponse(step=3, total=8)
        assert r.type == "progress"

    def test_result(self):
        r = ResultResponse(image="b64", seed=42, time_ms=1000, width=512, height=512)
        assert r.type == "result"

    def test_error(self):
        r = ErrorResponse(code="CANCELLED", message="User cancelled")
        assert r.type == "error"

    def test_pong(self):
        assert PongResponse().type == "pong"

    def test_list(self):
        r = ListResponse(list_type="loras", items=["lora1", "lora2"])
        assert len(r.items) == 2

    def test_realtime_ready(self):
        r = RealtimeReadyResponse()
        assert r.type == "realtime_ready"

    def test_realtime_result_with_roi(self):
        r = RealtimeResultResponse(
            image="b64", latency_ms=50, frame_id=1,
            width=512, height=512, roi_x=10, roi_y=20,
        )
        assert r.roi_x == 10

    def test_prompt_result(self):
        r = PromptResultResponse(prompt="test prompt", components={"style": "pixel art"})
        assert r.type == "prompt_result"
        assert r.components["style"] == "pixel art"

    def test_preset_response(self):
        r = PresetResponse(name="test", data={"steps": 8})
        assert r.type == "preset"

    def test_preset_saved(self):
        r = PresetSavedResponse(name="test")
        assert r.type == "preset_saved"

    def test_preset_deleted(self):
        r = PresetDeletedResponse(name="test")
        assert r.type == "preset_deleted"

    def test_cleanup_response(self):
        r = CleanupResponse(message="Done", freed_mb=128.5)
        assert r.type == "cleanup_done"
        assert r.freed_mb == 128.5
