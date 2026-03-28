"""Tests for random prompt schedule generation."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from sddj.prompt_generator import PromptGenerator
from sddj.prompt_schedule import (
    PromptSchedule,
    ScheduleRandomProfile,
    _RANDOM_PROFILES,
    randomize_schedule,
    schedule_to_dsl,
)


# ─── Fixtures ────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def deterministic_seed():
    random.seed(42)
    yield


@pytest.fixture
def gen(tmp_prompts_dir: Path) -> PromptGenerator:
    return PromptGenerator(tmp_prompts_dir)


# ─── TestRandomizeSchedule ───────────────────────────────────


class TestRandomizeSchedule:
    def test_returns_valid_dict(self, gen: PromptGenerator):
        result = randomize_schedule(100, 24.0, "dynamic", gen, randomness=10)
        assert isinstance(result, dict)
        assert "keyframes" in result
        assert isinstance(result["keyframes"], list)
        assert len(result["keyframes"]) > 0

    def test_keyframes_start_at_zero(self, gen: PromptGenerator):
        for profile in _RANDOM_PROFILES:
            result = randomize_schedule(100, 24.0, profile, gen, randomness=10)
            assert result["keyframes"][0]["frame"] == 0, f"profile={profile}"

    def test_keyframes_ascending_order(self, gen: PromptGenerator):
        for profile in _RANDOM_PROFILES:
            result = randomize_schedule(100, 24.0, profile, gen, randomness=15)
            frames = [kf["frame"] for kf in result["keyframes"]]
            assert frames == sorted(frames), f"profile={profile}"

    def test_no_e004_violations(self, gen: PromptGenerator):
        """transition_frames must not exceed gap to previous keyframe."""
        for profile in _RANDOM_PROFILES:
            result = randomize_schedule(100, 24.0, profile, gen, randomness=15)
            kfs = result["keyframes"]
            for i in range(1, len(kfs)):
                gap = kfs[i]["frame"] - kfs[i - 1]["frame"]
                tf = kfs[i].get("transition_frames", 0)
                assert tf < gap, (
                    f"profile={profile}, kf[{i}]: tf={tf} >= gap={gap}"
                )

    def test_weight_in_range(self, gen: PromptGenerator):
        for profile in _RANDOM_PROFILES:
            result = randomize_schedule(100, 24.0, profile, gen, randomness=15)
            for kf in result["keyframes"]:
                w = kf.get("weight", 1.0)
                assert 0.1 <= w <= 5.0, f"profile={profile}, weight={w}"
                we = kf.get("weight_end")
                if we is not None:
                    assert 0.1 <= we <= 5.0

    def test_denoise_in_range(self, gen: PromptGenerator):
        result = randomize_schedule(100, 24.0, "chaos", gen, randomness=20)
        for kf in result["keyframes"]:
            ds = kf.get("denoise_strength")
            if ds is not None:
                assert 0.0 <= ds <= 1.0

    def test_cfg_in_range(self, gen: PromptGenerator):
        result = randomize_schedule(100, 24.0, "chaos", gen, randomness=20)
        for kf in result["keyframes"]:
            cfg = kf.get("cfg_scale")
            if cfg is not None:
                assert 1.0 <= cfg <= 30.0

    def test_steps_in_range(self, gen: PromptGenerator):
        result = randomize_schedule(100, 24.0, "chaos", gen, randomness=20)
        for kf in result["keyframes"]:
            steps = kf.get("steps")
            if steps is not None:
                assert 1 <= steps <= 150

    def test_no_frame_beyond_total(self, gen: PromptGenerator):
        for total in [10, 50, 100, 256]:
            for profile in _RANDOM_PROFILES:
                result = randomize_schedule(total, 24.0, profile, gen, randomness=15)
                for kf in result["keyframes"]:
                    assert kf["frame"] < total, f"total={total}, profile={profile}"

    def test_locked_subject_in_prompts(self, gen: PromptGenerator):
        result = randomize_schedule(
            100, 24.0, "dynamic", gen,
            randomness=10,
            locked_fields={"subject": "dragon"},
        )
        for kf in result["keyframes"]:
            assert "dragon" in kf["prompt"].lower(), (
                f"locked subject 'dragon' missing from: {kf['prompt']}"
            )

    def test_randomness_zero_copies_base_prompt(self, gen: PromptGenerator):
        base = "a serene mountain lake"
        result = randomize_schedule(
            100, 24.0, "cinematic", gen,
            randomness=0, base_prompt=base,
        )
        for kf in result["keyframes"]:
            assert kf["prompt"] == base

    def test_all_profiles_valid_output(self, gen: PromptGenerator):
        for name in _RANDOM_PROFILES:
            result = randomize_schedule(100, 24.0, name, gen, randomness=10)
            sched = PromptSchedule.from_dict(result)
            vr = sched.validate(100)
            assert vr.valid, f"profile={name}, errors={vr.errors}"

    def test_unknown_profile_defaults_dynamic(self, gen: PromptGenerator):
        result = randomize_schedule(100, 24.0, "nonexistent_profile", gen, randomness=10)
        kf_count = len(result["keyframes"])
        dyn = _RANDOM_PROFILES["dynamic"]
        assert dyn.kf_count[0] <= kf_count <= dyn.kf_count[1]

    def test_negative_per_keyframe(self, gen: PromptGenerator):
        result = randomize_schedule(100, 24.0, "dynamic", gen, randomness=10)
        for kf in result["keyframes"]:
            assert "negative_prompt" in kf


# ─── TestScheduleToDsl ───────────────────────────────────────


class TestScheduleToDsl:
    def test_roundtrip_via_parser(self, gen: PromptGenerator):
        from sddj.dsl_parser import parse as dsl_parse

        result = randomize_schedule(100, 24.0, "dynamic", gen, randomness=10)
        dsl_text = schedule_to_dsl(result["keyframes"])
        parsed = dsl_parse(dsl_text, 100, 24.0, default_prompt="")
        assert parsed.schedule is not None
        assert len(parsed.schedule.keyframes) == len(result["keyframes"])
        for orig, parsed_kf in zip(result["keyframes"], parsed.schedule.keyframes):
            assert orig["frame"] == parsed_kf.frame

    def test_empty_returns_empty(self):
        assert schedule_to_dsl([]) == ""

    def test_all_directives_present(self):
        kfs = [{
            "frame": 0,
            "prompt": "test prompt",
            "negative_prompt": "bad things",
            "transition": "ease_in_out",
            "transition_frames": 8,
            "weight": 1.30,
            "weight_end": 0.80,
            "denoise_strength": 0.45,
            "cfg_scale": 6.5,
            "steps": 10,
        }]
        dsl = schedule_to_dsl(kfs)
        assert "[0]" in dsl
        assert "test prompt" in dsl
        assert "-- bad things" in dsl
        assert "transition: ease_in_out" in dsl
        assert "blend: 8" in dsl
        assert "weight: 1.30->0.80" in dsl
        assert "denoise: 0.45" in dsl
        assert "cfg: 6.5" in dsl
        assert "steps: 10" in dsl

    def test_weight_end_format(self):
        kfs = [{"frame": 0, "prompt": "x", "weight": 1.20, "weight_end": 1.50}]
        dsl = schedule_to_dsl(kfs)
        assert "weight: 1.20->1.50" in dsl


# ─── TestStress ──────────────────────────────────────────────


class TestStress:
    def test_100_random_all_valid(self, gen: PromptGenerator):
        profiles = list(_RANDOM_PROFILES.keys())
        for i in range(100):
            random.seed(i)
            profile = random.choice(profiles)
            total = random.randint(4, 500)
            rnd = random.randint(0, 20)
            result = randomize_schedule(total, 24.0, profile, gen, randomness=rnd)
            sched = PromptSchedule.from_dict(result)
            vr = sched.validate(total)
            assert vr.valid, (
                f"i={i}, profile={profile}, total={total}, rnd={rnd}, "
                f"errors={vr.errors}"
            )


# ─── TestEdgeCases ───────────────────────────────────────────


class TestEdgeCases:
    def test_total_frames_1(self, gen: PromptGenerator):
        result = randomize_schedule(1, 24.0, "dynamic", gen, randomness=10)
        assert len(result["keyframes"]) == 1
        assert result["keyframes"][0]["frame"] == 0
        assert result["keyframes"][0]["transition"] == "hard_cut"

    def test_total_frames_2(self, gen: PromptGenerator):
        result = randomize_schedule(2, 24.0, "dynamic", gen, randomness=10)
        kfs = result["keyframes"]
        assert len(kfs) <= 2
        for i in range(1, len(kfs)):
            tf = kfs[i].get("transition_frames", 0)
            gap = kfs[i]["frame"] - kfs[i - 1]["frame"]
            assert tf < gap

    def test_total_frames_3(self, gen: PromptGenerator):
        result = randomize_schedule(3, 24.0, "dynamic", gen, randomness=10)
        assert len(result["keyframes"]) >= 1
        sched = PromptSchedule.from_dict(result)
        assert sched.validate(3).valid

    def test_very_large_frames(self, gen: PromptGenerator):
        result = randomize_schedule(10800, 24.0, "chaos", gen, randomness=20)
        chaos = _RANDOM_PROFILES["chaos"]
        kf_count = len(result["keyframes"])
        assert kf_count <= chaos.kf_count[1]
        sched = PromptSchedule.from_dict(result)
        assert sched.validate(10800).valid


# ─── TestProfiles ────────────────────────────────────────────


class TestProfiles:
    def test_all_profiles_exist(self):
        expected = {"gentle", "dynamic", "rhythmic", "cinematic", "dreamy", "chaos", "minimal"}
        assert expected == set(_RANDOM_PROFILES.keys())

    def test_gentle_mostly_blends(self, gen: PromptGenerator):
        blend_count = 0
        total_transitions = 0
        for _ in range(50):
            result = randomize_schedule(100, 24.0, "gentle", gen, randomness=10)
            for kf in result["keyframes"][1:]:
                total_transitions += 1
                if kf["transition"] in ("blend", "ease_in_out"):
                    blend_count += 1
        if total_transitions > 0:
            ratio = blend_count / total_transitions
            assert ratio > 0.6, f"blend ratio = {ratio:.2f}"

    def test_chaos_has_param_overrides(self, gen: PromptGenerator):
        has_override = 0
        total_kf = 0
        for _ in range(30):
            result = randomize_schedule(100, 24.0, "chaos", gen, randomness=20)
            for kf in result["keyframes"]:
                total_kf += 1
                if kf.get("denoise_strength") or kf.get("cfg_scale") or kf.get("steps"):
                    has_override += 1
        if total_kf > 0:
            ratio = has_override / total_kf
            assert ratio > 0.3, f"override ratio = {ratio:.2f}"

    def test_cinematic_has_weight_end(self, gen: PromptGenerator):
        found = False
        for _ in range(30):
            result = randomize_schedule(100, 24.0, "cinematic", gen, randomness=15)
            for kf in result["keyframes"]:
                if kf.get("weight_end") is not None:
                    found = True
                    break
            if found:
                break
        assert found, "cinematic profile never produced weight_end"

    def test_minimal_exactly_2_keyframes(self, gen: PromptGenerator):
        for _ in range(20):
            result = randomize_schedule(100, 24.0, "minimal", gen, randomness=10)
            assert len(result["keyframes"]) == 2


# ─── TestProtocol ────────────────────────────────────────────


class TestProtocol:
    def test_action_exists(self):
        from sddj.protocol import Action
        assert hasattr(Action, "RANDOMIZE_SCHEDULE")
        assert Action.RANDOMIZE_SCHEDULE.value == "randomize_schedule"

    def test_request_schedule_profile(self):
        from sddj.protocol import Request, Action
        req = Request(action=Action.RANDOMIZE_SCHEDULE, schedule_profile="chaos")
        assert req.schedule_profile == "chaos"

    def test_response_serialization(self):
        from sddj.protocol import RandomizedScheduleResponse
        resp = RandomizedScheduleResponse(
            dsl_text="[0]\ntest\n",
            keyframes=[{"frame": 0, "prompt": "test"}],
            profile="dynamic",
            keyframe_count=1,
        )
        data = resp.model_dump()
        assert data["type"] == "randomized_schedule"
        assert data["profile"] == "dynamic"
        assert data["keyframe_count"] == 1
        assert len(data["keyframes"]) == 1
