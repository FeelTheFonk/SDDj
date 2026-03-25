"""Tests for expression_presets, new math functions, choreography, and slot inversion."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sddj.audio_analyzer import AudioAnalysis
from sddj.expression_presets import (
    CHOREOGRAPHY_PRESETS,
    EXPRESSION_PRESETS,
    detect_conflicts,
    get_choreography_preset,
    get_expression_preset,
    list_choreography_presets,
    list_expression_presets,
)
from sddj.modulation_engine import (
    ExpressionEvaluator,
    ModulationEngine,
    ModulationSlot,
    PRESETS,
    TARGET_RANGES,
)


# ─── Helpers ────────────────────────────────────────────────

def _make_analysis(n_frames=100, fps=24.0):
    """Create a synthetic AudioAnalysis with all required features."""
    rng = np.random.default_rng(42)
    features = {
        "global_rms": np.clip(rng.random(n_frames).astype(np.float32), 0, 1),
        "global_onset": np.clip(rng.random(n_frames).astype(np.float32), 0, 1),
        "global_centroid": np.clip(rng.random(n_frames).astype(np.float32), 0, 1),
        "global_beat": (rng.random(n_frames) > 0.7).astype(np.float32),
        "global_low": np.clip(rng.random(n_frames).astype(np.float32), 0, 1),
        "global_mid": np.clip(rng.random(n_frames).astype(np.float32), 0, 1),
        "global_high": np.clip(rng.random(n_frames).astype(np.float32), 0, 1),
        "global_sub_bass": np.clip(rng.random(n_frames).astype(np.float32), 0, 1),
        "global_bass": np.clip(rng.random(n_frames).astype(np.float32), 0, 1),
        "global_low_mid": np.clip(rng.random(n_frames).astype(np.float32), 0, 1),
        "global_upper_mid": np.clip(rng.random(n_frames).astype(np.float32), 0, 1),
        "global_presence": np.clip(rng.random(n_frames).astype(np.float32), 0, 1),
        "global_brilliance": np.clip(rng.random(n_frames).astype(np.float32), 0, 1),
        "global_air": np.clip(rng.random(n_frames).astype(np.float32), 0, 1),
        "global_ultrasonic": np.clip(rng.random(n_frames).astype(np.float32), 0, 1),
        "global_spectral_contrast": np.clip(rng.random(n_frames).astype(np.float32), 0, 1),
        "global_spectral_flatness": np.clip(rng.random(n_frames).astype(np.float32), 0, 1),
        "global_spectral_bandwidth": np.clip(rng.random(n_frames).astype(np.float32), 0, 1),
        "global_spectral_rolloff": np.clip(rng.random(n_frames).astype(np.float32), 0, 1),
        "global_spectral_flux": np.clip(rng.random(n_frames).astype(np.float32), 0, 1),
        "global_chroma_energy": np.clip(rng.random(n_frames).astype(np.float32), 0, 1),
    }
    return AudioAnalysis(
        fps=fps, duration=n_frames / fps, total_frames=n_frames,
        sample_rate=44100, audio_path="test.wav",
        features=features, bpm=120.0,
    )


# ─── New Math Functions ────────────────────────────────────

class TestNewMathFunctions:
    @pytest.fixture
    def ev(self):
        return ExpressionEvaluator()

    # Easing
    def test_easeIn_boundaries(self, ev):
        assert ev.evaluate("easeIn(0.0)", {}) == pytest.approx(0.0)
        assert ev.evaluate("easeIn(1.0)", {}) == pytest.approx(1.0)
        assert ev.evaluate("easeIn(0.5)", {}) == pytest.approx(0.25)

    def test_easeOut_boundaries(self, ev):
        assert ev.evaluate("easeOut(0.0)", {}) == pytest.approx(0.0)
        assert ev.evaluate("easeOut(1.0)", {}) == pytest.approx(1.0)
        assert ev.evaluate("easeOut(0.5)", {}) == pytest.approx(0.75)

    def test_easeInOut_boundaries(self, ev):
        assert ev.evaluate("easeInOut(0.0)", {}) == pytest.approx(0.0)
        assert ev.evaluate("easeInOut(1.0)", {}) == pytest.approx(1.0)
        assert ev.evaluate("easeInOut(0.5)", {}) == pytest.approx(0.5)

    def test_easeInCubic(self, ev):
        assert ev.evaluate("easeInCubic(0.5)", {}) == pytest.approx(0.125)

    def test_easeOutCubic(self, ev):
        assert ev.evaluate("easeOutCubic(0.5)", {}) == pytest.approx(0.875)

    # Animation easing
    def test_bounce_at_zero(self, ev):
        assert ev.evaluate("bounce(0.0)", {}) == pytest.approx(0.0, abs=0.01)

    def test_bounce_decays(self, ev):
        b_early = ev.evaluate("bounce(0.2)", {})
        b_late = ev.evaluate("bounce(0.9)", {})
        assert b_early > b_late  # bounce decays over time

    def test_elastic_boundaries(self, ev):
        assert ev.evaluate("elastic(0.0)", {}) == pytest.approx(0.0)
        assert ev.evaluate("elastic(1.0)", {}) == pytest.approx(1.0)

    # Utility
    def test_sign(self, ev):
        assert ev.evaluate("sign(-5.0)", {}) == -1.0
        assert ev.evaluate("sign(3.0)", {}) == 1.0
        assert ev.evaluate("sign(0.0)", {}) == 0.0

    def test_mix_is_lerp(self, ev):
        assert ev.evaluate("mix(0, 10, 0.3)", {}) == pytest.approx(3.0)
        assert ev.evaluate("mix(0, 10, 0.5)", {}) == ev.evaluate("lerp(0, 10, 0.5)", {})

    def test_remap(self, ev):
        assert ev.evaluate("remap(5, 0, 10, 0, 100)", {}) == pytest.approx(50.0)
        assert ev.evaluate("remap(0, 0, 10, 100, 200)", {}) == pytest.approx(100.0)
        assert ev.evaluate("remap(10, 0, 10, 100, 200)", {}) == pytest.approx(200.0)

    def test_remap_equal_bounds(self, ev):
        """Edge case: equal input range should not crash."""
        result = ev.evaluate("remap(5, 5, 5, 0, 100)", {})
        assert isinstance(result, float)

    def test_step(self, ev):
        assert ev.evaluate("step(0.55, 4)", {}) == pytest.approx(0.5)
        assert ev.evaluate("step(0.99, 10)", {}) == pytest.approx(0.9)

    def test_step_zero_n(self, ev):
        """Edge case: step with n=0 returns x unchanged."""
        assert ev.evaluate("step(0.5, 0)", {}) == pytest.approx(0.5)

    def test_fract(self, ev):
        assert ev.evaluate("fract(3.7)", {}) == pytest.approx(0.7)
        assert ev.evaluate("fract(1.0)", {}) == pytest.approx(0.0)

    def test_pingpong(self, ev):
        assert ev.evaluate("pingpong(0.5, 1.0)", {}) == pytest.approx(0.5)
        assert ev.evaluate("pingpong(1.5, 1.0)", {}) == pytest.approx(0.5)

    def test_pingpong_zero_length(self, ev):
        """Edge case: zero length should not crash."""
        assert ev.evaluate("pingpong(0.5, 0.0)", {}) == pytest.approx(0.0)

    def test_hash1d(self, ev):
        result = ev.evaluate("hash1d(42.0)", {})
        assert 0.0 <= result <= 1.0

    def test_smoothnoise(self, ev):
        result = ev.evaluate("smoothnoise(3.5)", {})
        assert isinstance(result, float)

    def test_atan2(self, ev):
        assert ev.evaluate("atan2(1.0, 0.0)", {}) == pytest.approx(math.pi / 2)


# ─── Slot Inversion ────────────────────────────────────────

class TestSlotInversion:
    def test_inverted_slot_maps_zero_to_max(self):
        engine = ModulationEngine()
        analysis = AudioAnalysis(
            fps=24.0, duration=1/24, total_frames=1,
            sample_rate=22050, audio_path="test.wav",
            features={"global_rms": np.array([0.0], dtype=np.float32)},
        )
        slot = ModulationSlot(
            source="global_rms", target="denoise_strength",
            min_val=0.2, max_val=0.8, invert=True,
        )
        schedule = engine.compute_schedule(analysis, [slot])
        # Source=0 inverted→1, mapped to max
        assert schedule.get_params(0)["denoise_strength"] == pytest.approx(0.8)

    def test_inverted_slot_maps_one_to_min(self):
        engine = ModulationEngine()
        analysis = AudioAnalysis(
            fps=24.0, duration=1/24, total_frames=1,
            sample_rate=22050, audio_path="test.wav",
            features={"global_rms": np.array([1.0], dtype=np.float32)},
        )
        slot = ModulationSlot(
            source="global_rms", target="denoise_strength",
            min_val=0.2, max_val=0.8, invert=True,
        )
        schedule = engine.compute_schedule(analysis, [slot])
        # Source=1 inverted→0, mapped to min
        assert schedule.get_params(0)["denoise_strength"] == pytest.approx(0.2)

    def test_non_inverted_unchanged(self):
        """Default invert=False should behave as before."""
        engine = ModulationEngine()
        analysis = AudioAnalysis(
            fps=24.0, duration=1/24, total_frames=1,
            sample_rate=22050, audio_path="test.wav",
            features={"global_rms": np.array([0.0], dtype=np.float32)},
        )
        slot = ModulationSlot(
            source="global_rms", target="denoise_strength",
            min_val=0.2, max_val=0.8, invert=False,
        )
        schedule = engine.compute_schedule(analysis, [slot])
        assert schedule.get_params(0)["denoise_strength"] == pytest.approx(0.2)


# ─── Expression Presets ────────────────────────────────────

class TestExpressionPresets:
    def test_all_presets_evaluate_without_error(self):
        """Every expression preset must evaluate without raising."""
        ev = ExpressionEvaluator()
        analysis = _make_analysis(n_frames=10)
        variables = {
            "t": 5.0, "max_f": 10.0, "fps": 24.0, "s": 5.0 / 24.0, "bpm": 120.0,
        }
        for name, arr in analysis.features.items():
            variables[name] = float(arr[5])

        for preset_name, preset in EXPRESSION_PRESETS.items():
            for target, expr in preset["targets"].items():
                try:
                    result = ev.evaluate(expr, variables)
                    assert isinstance(result, (int, float)), \
                        f"Preset {preset_name!r} target {target!r} returned non-numeric"
                except Exception as e:
                    pytest.fail(f"Preset {preset_name!r} target {target!r} failed: {e}")

    def test_all_presets_have_valid_targets(self):
        for name, preset in EXPRESSION_PRESETS.items():
            for target in preset["targets"]:
                assert target in TARGET_RANGES, \
                    f"Preset {name!r} has invalid target: {target!r}"

    def test_all_presets_have_category(self):
        for name, preset in EXPRESSION_PRESETS.items():
            assert "category" in preset, f"Preset {name!r} missing category"
            assert preset["category"] in (
                "rhythmic", "temporal", "spectral", "easing", "camera",
            ), f"Preset {name!r} has unknown category: {preset['category']!r}"

    def test_all_presets_have_description(self):
        for name, preset in EXPRESSION_PRESETS.items():
            assert "description" in preset, f"Preset {name!r} missing description"
            assert len(preset["description"]) > 10, \
                f"Preset {name!r} has too short description"

    def test_list_expression_presets_returns_categories(self):
        result = list_expression_presets()
        assert isinstance(result, dict)
        assert "rhythmic" in result
        assert "spectral" in result
        assert len(result) >= 5

    def test_get_expression_preset_found(self):
        preset = get_expression_preset("bpm_pulse")
        assert preset is not None
        assert "targets" in preset

    def test_get_expression_preset_not_found(self):
        assert get_expression_preset("nonexistent") is None


# ─── Choreography Presets ──────────────────────────────────

class TestChoreographyPresets:
    def test_all_choreographies_have_valid_targets(self):
        for name, choreo in CHOREOGRAPHY_PRESETS.items():
            for target in choreo.get("expressions", {}):
                assert target in TARGET_RANGES, \
                    f"Choreography {name!r} has invalid expression target: {target!r}"
            for slot in choreo.get("slots", []):
                assert slot["target"] in TARGET_RANGES, \
                    f"Choreography {name!r} has invalid slot target: {slot['target']!r}"

    def test_all_choreographies_evaluate_without_error(self):
        ev = ExpressionEvaluator()
        variables = {
            "t": 50.0, "max_f": 100.0, "fps": 24.0, "s": 50.0 / 24.0, "bpm": 120.0,
        }
        analysis = _make_analysis(n_frames=10)
        for name, arr in analysis.features.items():
            variables[name] = float(arr[5])

        for choreo_name, choreo in CHOREOGRAPHY_PRESETS.items():
            for target, expr in choreo.get("expressions", {}).items():
                try:
                    result = ev.evaluate(expr, variables)
                    assert isinstance(result, (int, float)), \
                        f"Choreography {choreo_name!r} target {target!r} returned non-numeric"
                except Exception as e:
                    pytest.fail(f"Choreography {choreo_name!r} target {target!r} failed: {e}")

    def test_all_choreographies_have_description(self):
        for name, choreo in CHOREOGRAPHY_PRESETS.items():
            assert "description" in choreo
            assert len(choreo["description"]) > 10

    def test_all_choreographies_have_expressions(self):
        """Every choreography must have at least 2 expression targets (coordinated)."""
        for name, choreo in CHOREOGRAPHY_PRESETS.items():
            expr = choreo.get("expressions", {})
            assert len(expr) >= 2, \
                f"Choreography {name!r} has only {len(expr)} expression targets"

    def test_choreography_produces_valid_schedule(self):
        """Integration test: choreography slots + expressions → valid schedule."""
        engine = ModulationEngine()
        analysis = _make_analysis(n_frames=50)
        choreo = CHOREOGRAPHY_PRESETS["wandering_voyage"]
        slots = [ModulationSlot(**s) for s in choreo["slots"]]
        schedule = engine.compute_schedule(analysis, slots, choreo["expressions"])
        assert schedule.total_frames == 50
        for i in range(50):
            params = schedule.get_params(i)
            assert len(params) > 0
            for key, val in params.items():
                assert key in TARGET_RANGES, f"Unknown target {key} at frame {i}"
                lo, hi = TARGET_RANGES[key]
                assert lo <= val <= hi, \
                    f"Frame {i}: {key}={val} outside [{lo}, {hi}]"

    def test_list_choreography_presets(self):
        result = list_choreography_presets()
        assert len(result) >= 7
        for item in result:
            assert "name" in item
            assert "description" in item

    def test_get_choreography_preset(self):
        assert get_choreography_preset("orbit_journey") is not None
        assert get_choreography_preset("nonexistent") is None


# ─── Conflict Detection ───────────────────────────────────

class TestConflictDetection:
    def test_no_conflicts(self):
        conflicts = detect_conflicts(["denoise_strength"], ["motion_x"])
        assert conflicts == []

    def test_single_conflict(self):
        conflicts = detect_conflicts(
            ["denoise_strength", "cfg_scale"],
            ["denoise_strength", "motion_x"],
        )
        assert conflicts == ["denoise_strength"]

    def test_multiple_conflicts(self):
        conflicts = detect_conflicts(
            ["motion_x", "motion_y"],
            ["motion_x", "motion_y", "motion_zoom"],
        )
        assert set(conflicts) == {"motion_x", "motion_y"}


# ─── New Presets in PRESETS Dict ───────────────────────────

class TestNewPresets:
    def test_voyage_presets_exist(self):
        for name in ("voyage_serene", "voyage_exploratory",
                     "voyage_dramatic", "voyage_psychedelic"):
            assert name in PRESETS, f"Missing preset: {name}"
            slots = ModulationEngine.get_preset(name)
            assert len(slots) >= 3, f"Preset {name!r} has too few slots"

    def test_rest_aware_presets_exist(self):
        for name in ("intelligent_drift", "reactive_pause"):
            assert name in PRESETS, f"Missing preset: {name}"
            slots = ModulationEngine.get_preset(name)
            assert len(slots) >= 3

    def test_voyage_presets_have_motion(self):
        motion_set = {"motion_x", "motion_y", "motion_zoom", "motion_rotation",
                      "motion_tilt_x", "motion_tilt_y"}
        for name in ("voyage_serene", "voyage_exploratory",
                     "voyage_dramatic", "voyage_psychedelic"):
            slots = ModulationEngine.get_preset(name)
            targets = {s.target for s in slots}
            assert targets & motion_set, f"Preset {name!r} has no motion targets"

    def test_voyage_psychedelic_has_palette_shift(self):
        slots = ModulationEngine.get_preset("voyage_psychedelic")
        targets = {s.target for s in slots}
        assert "palette_shift" in targets

    def test_new_presets_denoise_floor(self):
        """All new presets must respect denoise min_val >= 0.30."""
        new_presets = [
            "voyage_serene", "voyage_exploratory", "voyage_dramatic",
            "voyage_psychedelic", "intelligent_drift", "reactive_pause",
        ]
        for name in new_presets:
            for slot_dict in PRESETS[name]:
                if slot_dict["target"] == "denoise_strength":
                    assert slot_dict["min_val"] >= 0.30, \
                        f"Preset {name!r} denoise min_val below 0.30"


# ─── Regression: Existing Expressions Still Work ──────────

class TestRegressionExistingExpressions:
    """Ensure adding new functions doesn't break expressions from docs/existing usage."""

    def test_doc_example_expressions(self):
        ev = ExpressionEvaluator()
        variables = {
            "t": 5, "max_f": 100, "fps": 24.0, "s": 5/24,
            "bpm": 120.0, "global_rms": 0.5, "global_onset": 0.3,
            "global_beat": 0.8, "global_centroid": 0.6,
            "global_low": 0.4, "global_mid": 0.5, "global_high": 0.3,
        }
        # Expressions from AUDIO-REACTIVITY.md
        exprs = [
            "0.2 + 0.3 * global_rms",
            "where(global_beat > 0.5, 0.6, 0.3)",
            "smoothstep(0.2, 0.8, global_rms) * 0.5 + 0.2",
            "lerp(0.2, 0.7, t / max_f)",
            "clamp(global_onset * 2, 0.2, 0.9)",
            "sin(t * 0.1) * 2 + 3",
            "0.3 + 0.4 * abs(sin(s * 3.14159 * bpm / 60))",
        ]
        for expr in exprs:
            result = ev.evaluate(expr, variables)
            assert isinstance(result, float), f"Expression {expr!r} failed"


# ─── New Validation Tests ─────────────────────────────────

class TestPresetValidation:
    """Validate all expression presets parse and evaluate without error."""

    @pytest.fixture()
    def evaluator(self) -> ExpressionEvaluator:
        return ExpressionEvaluator()

    @pytest.fixture()
    def dummy_vars(self) -> list[str]:
        return [
            "global_rms", "global_onset", "global_beat", "global_low",
            "global_mid", "global_high", "global_centroid",
            "global_spectral_flux", "global_spectral_contrast",
            "global_chroma_energy", "bpm",
        ]

    def test_all_presets_parse(self, evaluator, dummy_vars):
        """Every expression in every preset must parse without error."""
        for name, preset in EXPRESSION_PRESETS.items():
            for target, expr in preset["targets"].items():
                err = evaluator.validate(expr, dummy_vars)
                assert err is None, f"Preset '{name}' target '{target}' failed: {err}"

    def test_all_presets_evaluate(self, evaluator):
        """Every expression must evaluate to a finite float."""
        variables = {
            "global_rms": 0.5, "global_onset": 0.3, "global_beat": 0.7,
            "global_low": 0.4, "global_mid": 0.5, "global_high": 0.6,
            "global_centroid": 0.5, "global_spectral_flux": 0.3,
            "global_spectral_contrast": 0.4, "global_chroma_energy": 0.5,
            "t": 50, "max_f": 100, "fps": 24.0, "s": 2.0, "bpm": 120.0,
        }
        for name, preset in EXPRESSION_PRESETS.items():
            for target, expr in preset["targets"].items():
                val = evaluator.evaluate(expr, variables)
                assert isinstance(val, float), f"Preset '{name}' target '{target}' returned {type(val)}"
                assert val == val, f"Preset '{name}' target '{target}' returned NaN"

    def test_easing_functions_resolve(self, evaluator):
        """Easing functions used in presets must be registered in the evaluator."""
        variables = {"t": 50, "max_f": 100, "fps": 24.0, "s": 2.0}
        easing_expressions = [
            ("easeIn", "easeIn(0.5)"),
            ("easeOut", "easeOut(0.5)"),
            ("easeInOut", "easeInOut(0.5)"),
            ("bounce", "bounce(0.5)"),
            ("elastic", "elastic(0.5)"),
        ]
        for name, expr in easing_expressions:
            val = evaluator.evaluate(expr, variables)
            assert isinstance(val, float), f"Easing '{name}' failed"
            assert val == val, f"Easing '{name}' returned NaN"

    def test_all_denoise_min_val_above_floor(self):
        """All denoise_strength slots must have min_val >= 0.30 (Hyper-SD quality floor)."""
        for name, choreo in CHOREOGRAPHY_PRESETS.items():
            for slot in choreo.get("slots", []):
                if slot["target"] == "denoise_strength":
                    assert slot["min_val"] >= 0.30, (
                        f"Choreography '{name}' denoise min_val={slot['min_val']} < 0.30"
                    )

