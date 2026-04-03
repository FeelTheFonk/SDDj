#!/usr/bin/env python3
"""SDDj Pipeline Quality Diagnostics — Prompt Adherence Isolation Tests.

Systematic A/B testing to identify which inference pipeline component
degrades prompt adherence. Tests SageAttention, DeepCache, FreeU, and
their interactions with torch.compile + Hyper-SD.

STOP THE SERVER BEFORE RUNNING (frees GPU memory).

Usage:
    cd server
    uv run python -m sddj.diagnostics.pipeline_quality

Results saved to: server/diagnostics_output/
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s %(levelname)-6s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sddj.diagnostics")

# ── Output directory ────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "diagnostics_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Test parameters ─────────────────────────────────────────
SEED = 42
STEPS = 8
CFG = 5.0
WIDTH = 512
HEIGHT = 512
CLIP_SKIP = 2

# Two maximally different prompts to test discriminability
PROMPT_A = "a bright red dragon breathing fire over a medieval castle, pixel art, game sprite"
PROMPT_B = "a tiny blue fish swimming in a green ocean with coral, pixel art, game sprite"
NEGATIVE = "blurry, lowres, bad quality"


def _mse(img_a: Image.Image, img_b: Image.Image) -> float:
    """Mean Squared Error between two PIL images."""
    a = np.asarray(img_a, dtype=np.float32)
    b = np.asarray(img_b, dtype=np.float32)
    if a.shape != b.shape:
        return float("inf")
    return float(np.mean((a - b) ** 2))


def _ssim_approx(img_a: Image.Image, img_b: Image.Image) -> float:
    """Simplified Structural Similarity Index (luminance channel)."""
    a = np.asarray(img_a.convert("L"), dtype=np.float64)
    b = np.asarray(img_b.convert("L"), dtype=np.float64)
    c1 = 6.5025  # (0.01 * 255)^2
    c2 = 58.5225  # (0.03 * 255)^2
    mu_a = a.mean()
    mu_b = b.mean()
    sig_a = a.var()
    sig_b = b.var()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    ssim = ((2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (sig_a + sig_b + c2)
    )
    return float(ssim)


def _compute_clip_score(pipe, prompt: str, image: Image.Image) -> float | None:
    """Compute CLIP text-image similarity using the pipeline's own text encoder.

    Uses a lightweight approach: encode the prompt via the pipeline's CLIP
    text encoder and compare with a CLIP image encoding.
    Returns None if CLIPModel is not available.
    """
    try:
        from transformers import CLIPModel, CLIPProcessor

        model_id = "openai/clip-vit-base-patch32"
        clip_model = CLIPModel.from_pretrained(model_id, local_files_only=True).to("cpu")
        clip_processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)

        inputs = clip_processor(
            text=[prompt], images=[image.convert("RGB")],
            return_tensors="pt", padding=True,
        )
        with torch.inference_mode():
            outputs = clip_model(**inputs)
        score = outputs.logits_per_text.item() / 100.0  # Normalize to ~0-1
        del clip_model, clip_processor
        gc.collect()
        return score
    except Exception as e:
        log.debug("CLIP score unavailable: %s", e)
        return None


# ── SageAttention Probe (no pipeline needed) ────────────────

def test_1_sageattn_probe():
    """Test which head dimensions SageAttention supports on this GPU."""
    log.info("=" * 60)
    log.info("TEST 1: SageAttention Head Dimension Probe")
    log.info("=" * 60)

    results = {}
    try:
        from sageattention import sageattn
    except ImportError:
        log.warning("sageattention not installed — skipping probe")
        return {"error": "sageattention not installed"}

    import torch.nn.functional as F
    native_sdpa = F.scaled_dot_product_attention

    head_dims = [32, 40, 64, 80, 96, 128, 160, 192, 256]
    batch, heads, seq_len = 1, 8, 77

    for hd in head_dims:
        q = torch.randn(batch, heads, seq_len, hd, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, heads, seq_len, hd, device="cuda", dtype=torch.float16)
        v = torch.randn(batch, heads, seq_len, hd, device="cuda", dtype=torch.float16)

        try:
            with torch.inference_mode():
                out_sage = sageattn(q, k, v)
                out_native = native_sdpa(q, k, v)
                max_diff = (out_sage.float() - out_native.float()).abs().max().item()
                mean_diff = (out_sage.float() - out_native.float()).abs().mean().item()
            results[hd] = {
                "supported": True,
                "max_abs_diff": max_diff,
                "mean_abs_diff": mean_diff,
            }
            log.info(
                "  head_dim=%3d: OK  (max_diff=%.6f, mean_diff=%.6f)",
                hd, max_diff, mean_diff,
            )
        except Exception as e:
            results[hd] = {"supported": False, "error": str(e)}
            log.info("  head_dim=%3d: FAIL (%s)", hd, e)

        del q, k, v
        torch.cuda.empty_cache()

    # SD1.5 head dims: 40 (320ch/8h), 80 (640ch/8h), 160 (1280ch/8h), 64 (text encoder)
    sd15_dims = {40: "UNet 320ch", 80: "UNet 640ch", 160: "UNet 1280ch", 64: "TextEncoder"}
    log.info("\n  SD1.5 relevant head dimensions:")
    for hd, desc in sd15_dims.items():
        r = results.get(hd, {})
        status = "OK" if r.get("supported") else "FALLBACK"
        log.info("    %s (head_dim=%d): %s", desc, hd, status)

    return results


# ── Text Encoder Embedding Consistency ──────────────────────

def test_2_embedding_consistency(pipe):
    """Compare text encoder output with SageAttention vs native SDPA."""
    log.info("=" * 60)
    log.info("TEST 2: Text Encoder Embedding Consistency")
    log.info("=" * 60)

    import torch.nn.functional as F
    from ..pipeline_factory import _original_sdpa

    prompt = PROMPT_A
    results = {}

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    tokens = tokenizer(
        prompt, padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(text_encoder.device)

    # 1. Encode with current attention (SageAttention if active)
    with torch.inference_mode():
        embed_current = text_encoder(tokens)[0].clone()

    # 2. Temporarily restore native SDPA and encode again
    if _original_sdpa is not None:
        saved_sdpa = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = _original_sdpa
        with torch.inference_mode():
            embed_native = text_encoder(tokens)[0].clone()
        F.scaled_dot_product_attention = saved_sdpa

        diff = (embed_current.float() - embed_native.float())
        max_diff = diff.abs().max().item()
        mean_diff = diff.abs().mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            embed_current.float().view(1, -1),
            embed_native.float().view(1, -1),
        ).item()

        results = {
            "sage_vs_native_max_diff": max_diff,
            "sage_vs_native_mean_diff": mean_diff,
            "sage_vs_native_cosine_sim": cos_sim,
        }
        log.info("  Embedding difference (SageAttention vs native SDPA):")
        log.info("    Max absolute diff:  %.8f", max_diff)
        log.info("    Mean absolute diff: %.8f", mean_diff)
        log.info("    Cosine similarity:  %.8f", cos_sim)

        if cos_sim < 0.999:
            log.warning("  ⚠ SIGNIFICANT embedding divergence! Cosine sim < 0.999")
        elif cos_sim < 0.9999:
            log.info("  → Moderate embedding divergence (cosine < 0.9999)")
        else:
            log.info("  → Embeddings are nearly identical")
    else:
        log.info("  SageAttention not active — no comparison needed")
        results = {"sage_active": False}

    return results


# ── Pipeline Generation Helpers ─────────────────────────────

def _generate_one(pipe, prompt: str, negative: str, seed: int,
                   steps: int = STEPS, cfg: float = CFG) -> Image.Image:
    """Generate a single image, returning PIL Image."""
    gen = torch.Generator("cuda").manual_seed(seed)
    with torch.inference_mode():
        torch.compiler.cudagraph_mark_step_begin()
        output = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=WIDTH,
            height=HEIGHT,
            generator=gen,
            clip_skip=CLIP_SKIP,
            output_type="pil",
        )
    return output.images[0]


# ── Prompt Discriminability Test ────────────────────────────

def test_3_prompt_discriminability(pipe):
    """Generate with two very different prompts — are results different?

    If outputs are nearly identical despite different prompts,
    the pipeline is ignoring prompts.
    """
    log.info("=" * 60)
    log.info("TEST 3: Prompt Discriminability")
    log.info("=" * 60)
    log.info("  Prompt A: %s", PROMPT_A[:60])
    log.info("  Prompt B: %s", PROMPT_B[:60])

    img_a = _generate_one(pipe, PROMPT_A, NEGATIVE, SEED)
    img_b = _generate_one(pipe, PROMPT_B, NEGATIVE, SEED)

    img_a.save(OUTPUT_DIR / "test3_prompt_a.png")
    img_b.save(OUTPUT_DIR / "test3_prompt_b.png")

    mse = _mse(img_a, img_b)
    ssim = _ssim_approx(img_a, img_b)

    log.info("  MSE between prompt A and B: %.2f", mse)
    log.info("  SSIM between prompt A and B: %.4f", ssim)

    if mse < 100:
        log.warning("  ⚠ CRITICAL: MSE < 100 — images nearly identical despite different prompts!")
        log.warning("  → Pipeline is likely IGNORING the prompt")
        verdict = "FAIL_PROMPTS_IGNORED"
    elif mse < 500:
        log.warning("  ⚠ WARNING: MSE < 500 — images suspiciously similar")
        verdict = "WARN_LOW_DISCRIMINABILITY"
    else:
        log.info("  → Prompts produce visually distinct images (MSE > 500)")
        verdict = "PASS"

    # CLIP scores if available
    clip_a = _compute_clip_score(pipe, PROMPT_A, img_a)
    clip_b = _compute_clip_score(pipe, PROMPT_B, img_b)
    clip_cross_a = _compute_clip_score(pipe, PROMPT_B, img_a)  # B's prompt for A's image

    if clip_a is not None:
        log.info("  CLIP score (prompt_A ↔ image_A): %.4f", clip_a)
        log.info("  CLIP score (prompt_B ↔ image_B): %.4f", clip_b)
        log.info("  CLIP score (prompt_B ↔ image_A): %.4f (cross, should be lower)", clip_cross_a)

    return {
        "mse": mse,
        "ssim": ssim,
        "verdict": verdict,
        "clip_a_match": clip_a,
        "clip_b_match": clip_b,
        "clip_cross": clip_cross_a,
    }


# ── Component Isolation A/B Tests ───────────────────────────

def test_4_component_isolation(pipe, deepcache_helper):
    """A/B test each component to measure its impact on output.

    For each component, generate with it ON (baseline) and OFF,
    then measure how much the output changes.
    Large change → component significantly affects output.
    """
    log.info("=" * 60)
    log.info("TEST 4: Component Isolation A/B Tests")
    log.info("=" * 60)

    import torch.nn.functional as F
    from ..pipeline_factory import _original_sdpa
    from ..config import settings

    prompt = PROMPT_A
    results = {}

    # Capture current SDPA at function entry (may be SageAttention wrapper)
    sage_active = _original_sdpa is not None
    saved_sdpa = F.scaled_dot_product_attention if sage_active else None

    # ── 4a: Baseline (current production config) ────────────
    log.info("  [4a] Generating BASELINE (all optimizations active)...")
    img_baseline = _generate_one(pipe, prompt, NEGATIVE, SEED)
    img_baseline.save(OUTPUT_DIR / "test4a_baseline.png")

    clip_baseline = _compute_clip_score(pipe, prompt, img_baseline)
    if clip_baseline is not None:
        log.info("       CLIP score: %.4f", clip_baseline)

    # ── 4b: SageAttention OFF ───────────────────────────────
    if sage_active:
        log.info("  [4b] Generating with NATIVE SDPA (SageAttention OFF)...")
        F.scaled_dot_product_attention = _original_sdpa
        try:
            img_sdpa = _generate_one(pipe, prompt, NEGATIVE, SEED)
            img_sdpa.save(OUTPUT_DIR / "test4b_native_sdpa.png")
            mse_sage = _mse(img_baseline, img_sdpa)
            ssim_sage = _ssim_approx(img_baseline, img_sdpa)
            clip_sdpa = _compute_clip_score(pipe, prompt, img_sdpa)
            log.info("       MSE vs baseline: %.2f", mse_sage)
            log.info("       SSIM vs baseline: %.4f", ssim_sage)
            if clip_sdpa is not None:
                log.info("       CLIP score: %.4f (baseline: %.4f)", clip_sdpa, clip_baseline)
            results["sage_off"] = {
                "mse_vs_baseline": mse_sage,
                "ssim_vs_baseline": ssim_sage,
                "clip_score": clip_sdpa,
            }
        finally:
            F.scaled_dot_product_attention = saved_sdpa
    else:
        log.info("  [4b] SKIPPED — SageAttention not active")

    # ── 4c: DeepCache OFF ───────────────────────────────────
    if deepcache_helper is not None:
        log.info("  [4c] Generating with DEEPCACHE OFF...")
        try:
            deepcache_helper.disable()
            img_nodc = _generate_one(pipe, prompt, NEGATIVE, SEED)
            img_nodc.save(OUTPUT_DIR / "test4c_no_deepcache.png")
            mse_dc = _mse(img_baseline, img_nodc)
            ssim_dc = _ssim_approx(img_baseline, img_nodc)
            clip_nodc = _compute_clip_score(pipe, prompt, img_nodc)
            log.info("       MSE vs baseline: %.2f", mse_dc)
            log.info("       SSIM vs baseline: %.4f", ssim_dc)
            if clip_nodc is not None:
                log.info("       CLIP score: %.4f (baseline: %.4f)", clip_nodc, clip_baseline)
            results["deepcache_off"] = {
                "mse_vs_baseline": mse_dc,
                "ssim_vs_baseline": ssim_dc,
                "clip_score": clip_nodc,
            }
        finally:
            deepcache_helper.enable()
    else:
        log.info("  [4c] SKIPPED — DeepCache not active")

    # ── 4d: FreeU OFF ───────────────────────────────────────
    log.info("  [4d] Generating with FREEU OFF...")
    try:
        pipe.disable_freeu()
        img_nofreeu = _generate_one(pipe, prompt, NEGATIVE, SEED)
        img_nofreeu.save(OUTPUT_DIR / "test4d_no_freeu.png")
        mse_fu = _mse(img_baseline, img_nofreeu)
        ssim_fu = _ssim_approx(img_baseline, img_nofreeu)
        clip_nofreeu = _compute_clip_score(pipe, prompt, img_nofreeu)
        log.info("       MSE vs baseline: %.2f", mse_fu)
        log.info("       SSIM vs baseline: %.4f", ssim_fu)
        if clip_nofreeu is not None:
            log.info("       CLIP score: %.4f (baseline: %.4f)", clip_nofreeu, clip_baseline)
        results["freeu_off"] = {
            "mse_vs_baseline": mse_fu,
            "ssim_vs_baseline": ssim_fu,
            "clip_score": clip_nofreeu,
        }
    except Exception as e:
        log.warning("       FreeU disable failed: %s", e)
    finally:
        if settings.enable_freeu:
            pipe.enable_freeu(
                s1=settings.freeu_s1, s2=settings.freeu_s2,
                b1=settings.freeu_b1, b2=settings.freeu_b2,
            )

    # ── 4e: DeepCache + FreeU OFF ──────────────────────────
    log.info("  [4e] Generating with DEEPCACHE + FREEU OFF...")
    dc_disabled = False
    fu_disabled = False
    try:
        if deepcache_helper is not None:
            deepcache_helper.disable()
            dc_disabled = True
        pipe.disable_freeu()
        fu_disabled = True

        img_minimal = _generate_one(pipe, prompt, NEGATIVE, SEED)
        img_minimal.save(OUTPUT_DIR / "test4e_no_dc_no_freeu.png")
        mse_min = _mse(img_baseline, img_minimal)
        ssim_min = _ssim_approx(img_baseline, img_minimal)
        clip_minimal = _compute_clip_score(pipe, prompt, img_minimal)
        log.info("       MSE vs baseline: %.2f", mse_min)
        log.info("       SSIM vs baseline: %.4f", ssim_min)
        if clip_minimal is not None:
            log.info("       CLIP score: %.4f (baseline: %.4f)", clip_minimal, clip_baseline)
        results["dc_plus_freeu_off"] = {
            "mse_vs_baseline": mse_min,
            "ssim_vs_baseline": ssim_min,
            "clip_score": clip_minimal,
        }
    finally:
        if dc_disabled and deepcache_helper is not None:
            deepcache_helper.enable()
        if fu_disabled and settings.enable_freeu:
            pipe.enable_freeu(
                s1=settings.freeu_s1, s2=settings.freeu_s2,
                b1=settings.freeu_b1, b2=settings.freeu_b2,
            )

    # ── 4f: SageAttention + DeepCache + FreeU OFF ──────────
    if sage_active:
        log.info("  [4f] Generating with ALL OPTIMIZATIONS OFF (SDPA + no DC + no FreeU)...")
        dc_disabled = False
        fu_disabled = False
        try:
            F.scaled_dot_product_attention = _original_sdpa
            if deepcache_helper is not None:
                deepcache_helper.disable()
                dc_disabled = True
            pipe.disable_freeu()
            fu_disabled = True

            img_bare = _generate_one(pipe, prompt, NEGATIVE, SEED)
            img_bare.save(OUTPUT_DIR / "test4f_all_off.png")
            mse_bare = _mse(img_baseline, img_bare)
            ssim_bare = _ssim_approx(img_baseline, img_bare)
            clip_bare = _compute_clip_score(pipe, prompt, img_bare)
            log.info("       MSE vs baseline: %.2f", mse_bare)
            log.info("       SSIM vs baseline: %.4f", ssim_bare)
            if clip_bare is not None:
                log.info("       CLIP score: %.4f (baseline: %.4f)", clip_bare, clip_baseline)
            results["all_off"] = {
                "mse_vs_baseline": mse_bare,
                "ssim_vs_baseline": ssim_bare,
                "clip_score": clip_bare,
            }
        finally:
            F.scaled_dot_product_attention = saved_sdpa
            if dc_disabled and deepcache_helper is not None:
                deepcache_helper.enable()
            if fu_disabled and settings.enable_freeu:
                pipe.enable_freeu(
                    s1=settings.freeu_s1, s2=settings.freeu_s2,
                    b1=settings.freeu_b1, b2=settings.freeu_b2,
                )

    # ── 4g: Prompt discriminability with ALL OFF ────────────
    if sage_active:
        log.info("  [4g] Prompt discriminability with ALL OPTIMIZATIONS OFF...")
        dc_disabled = False
        fu_disabled = False
        try:
            F.scaled_dot_product_attention = _original_sdpa
            if deepcache_helper is not None:
                deepcache_helper.disable()
                dc_disabled = True
            pipe.disable_freeu()
            fu_disabled = True

            img_bare_a = _generate_one(pipe, PROMPT_A, NEGATIVE, SEED)
            img_bare_b = _generate_one(pipe, PROMPT_B, NEGATIVE, SEED)
            img_bare_a.save(OUTPUT_DIR / "test4g_all_off_prompt_a.png")
            img_bare_b.save(OUTPUT_DIR / "test4g_all_off_prompt_b.png")
            mse_disc = _mse(img_bare_a, img_bare_b)
            log.info("       MSE between prompt A/B (all off): %.2f", mse_disc)
            results["all_off_discriminability_mse"] = mse_disc
        finally:
            F.scaled_dot_product_attention = saved_sdpa
            if dc_disabled and deepcache_helper is not None:
                deepcache_helper.enable()
            if fu_disabled and settings.enable_freeu:
                pipe.enable_freeu(
                    s1=settings.freeu_s1, s2=settings.freeu_s2,
                    b1=settings.freeu_b1, b2=settings.freeu_b2,
                )

    results["clip_baseline"] = clip_baseline
    return results


# ── DeepCache Interaction Test ──────────────────────────────

def test_5_deepcache_detail(pipe, deepcache_helper):
    """Test DeepCache impact at different intervals.

    interval=1 means no caching (every step is full).
    Compare with interval=3 (current) to see how much caching degrades output.
    """
    log.info("=" * 60)
    log.info("TEST 5: DeepCache Interval Impact")
    log.info("=" * 60)

    if deepcache_helper is None:
        log.info("  SKIPPED — DeepCache not active")
        return {}

    prompt = PROMPT_A
    results = {}

    # Current config (interval=3)
    log.info("  [interval=3] Generating (current config)...")
    img_i3 = _generate_one(pipe, prompt, NEGATIVE, SEED)
    img_i3.save(OUTPUT_DIR / "test5_deepcache_i3.png")

    # Disable DeepCache (equivalent to interval=infinity, i.e. full UNet every step)
    log.info("  [no cache] Generating (DeepCache off = full UNet every step)...")
    deepcache_helper.disable()
    try:
        img_full = _generate_one(pipe, prompt, NEGATIVE, SEED)
        img_full.save(OUTPUT_DIR / "test5_deepcache_off.png")
    finally:
        deepcache_helper.enable()

    mse = _mse(img_i3, img_full)
    ssim = _ssim_approx(img_i3, img_full)
    log.info("  MSE (cached vs full):  %.2f", mse)
    log.info("  SSIM (cached vs full): %.4f", ssim)

    clip_cached = _compute_clip_score(pipe, prompt, img_i3)
    clip_full = _compute_clip_score(pipe, prompt, img_full)
    if clip_cached is not None:
        log.info("  CLIP score cached: %.4f, full: %.4f", clip_cached, clip_full)
        delta = (clip_full or 0) - (clip_cached or 0)
        if delta > 0.02:
            log.warning("  ⚠ DeepCache significantly degrades prompt adherence (CLIP delta: +%.4f)", delta)

    results = {
        "mse_cached_vs_full": mse,
        "ssim_cached_vs_full": ssim,
        "clip_cached": clip_cached,
        "clip_full": clip_full,
    }
    return results


# ── Main Entry Point ────────────────────────────────────────

def main():
    log.info("SDDj Pipeline Quality Diagnostics")
    log.info("=" * 60)
    log.info("Output directory: %s", OUTPUT_DIR)

    if not torch.cuda.is_available():
        log.error("CUDA not available — cannot run diagnostics")
        sys.exit(1)

    props = torch.cuda.get_device_properties(0)
    log.info("GPU: %s (sm%d%d, %d SMs, %.1f GB)",
             props.name,
             *torch.cuda.get_device_capability(),
             props.multi_processor_count,
             props.total_memory / 1024**3)

    all_results = {}

    # ── Test 1: SageAttention probe (lightweight) ───────────
    t0 = time.perf_counter()
    all_results["sageattn_probe"] = test_1_sageattn_probe()
    log.info("  Test 1 completed in %.1fs\n", time.perf_counter() - t0)

    # ── Load pipeline ───────────────────────────────────────
    log.info("Loading pipeline for generation tests...")
    log.info("(This uses the same loading path as the production server)")

    from ..config import settings
    from .. import pipeline_factory
    from ..freeu_applicator import apply_freeu
    from ..embedding_blend import bump_model_generation

    bump_model_generation()

    if settings.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    pipe = pipeline_factory.load_base_pipeline()
    pipeline_factory.setup_attention(pipe)
    pipeline_factory.setup_vae(pipe)
    pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)
    pipeline_factory.setup_hyper_sd(pipe)
    apply_freeu(pipe)

    # Quantization + compile
    pipeline_factory.apply_unet_quantization(pipe)
    pipeline_factory.apply_torch_compile(pipe)
    pipeline_factory.apply_vae_compile(pipe)

    # DeepCache
    from .. import deepcache_manager
    dc_helper = deepcache_manager.create_helper(pipe)

    # Default style LoRA
    from ..lora_manager import list_loras
    from ..lora_fuser import LoRAFuser
    from ..embedding_blend import clear_embedding_cache

    fuser = LoRAFuser()
    lora_name = settings.default_style_lora
    if lora_name == "auto":
        available = list_loras()
        if available:
            lora_name = available[0]
        else:
            lora_name = None

    if lora_name:
        fuser.set_lora(pipe, lora_name, settings.default_style_lora_weight)
        clear_embedding_cache()
        log.info("Style LoRA loaded: %s (weight=%.2f)", lora_name, settings.default_style_lora_weight)

    # TI embeddings
    from ..ti_manager import list_embeddings, resolve_embedding_path
    for name in list_embeddings():
        try:
            path = resolve_embedding_path(name)
            pipe.load_textual_inversion(str(path), token=name)
        except Exception:
            pass

    # Warmup
    log.info("Warmup (first compile)...")
    with torch.inference_mode():
        gen = torch.Generator("cuda").manual_seed(0)
        torch.compiler.cudagraph_mark_step_begin()
        pipe(
            prompt="warmup", negative_prompt="warmup",
            num_inference_steps=STEPS, guidance_scale=CFG,
            width=WIDTH, height=HEIGHT, generator=gen,
            clip_skip=CLIP_SKIP, output_type="latent",
        )
    if dc_helper is not None:
        dc_helper.cached_output = {}
        dc_helper.start_timestep = None
        dc_helper.cur_timestep = 0
    log.info("Pipeline ready.\n")

    # ── Test 2: Embedding consistency ───────────────────────
    t0 = time.perf_counter()
    all_results["embedding_consistency"] = test_2_embedding_consistency(pipe)
    log.info("  Test 2 completed in %.1fs\n", time.perf_counter() - t0)

    # ── Test 3: Prompt discriminability ─────────────────────
    t0 = time.perf_counter()
    all_results["prompt_discriminability"] = test_3_prompt_discriminability(pipe)
    log.info("  Test 3 completed in %.1fs\n", time.perf_counter() - t0)

    # ── Test 4: Component isolation ─────────────────────────
    t0 = time.perf_counter()
    all_results["component_isolation"] = test_4_component_isolation(pipe, dc_helper)
    log.info("  Test 4 completed in %.1fs\n", time.perf_counter() - t0)

    # ── Test 5: DeepCache detail ────────────────────────────
    t0 = time.perf_counter()
    all_results["deepcache_detail"] = test_5_deepcache_detail(pipe, dc_helper)
    log.info("  Test 5 completed in %.1fs\n", time.perf_counter() - t0)

    # ── Final Report ────────────────────────────────────────
    log.info("=" * 60)
    log.info("DIAGNOSTIC REPORT")
    log.info("=" * 60)

    # Report summary
    probe = all_results.get("sageattn_probe", {})
    hd160 = probe.get(160, {})
    if hd160.get("supported"):
        log.info("SageAttention: head_dim=160 SUPPORTED (max_diff=%.6f)", hd160["max_abs_diff"])
    else:
        log.warning("SageAttention: head_dim=160 NOT SUPPORTED → mixed attention in UNet")

    embed = all_results.get("embedding_consistency", {})
    if "sage_vs_native_cosine_sim" in embed:
        cos = embed["sage_vs_native_cosine_sim"]
        if cos < 0.999:
            log.warning("Text embeddings DIVERGE (cos=%.6f) → SageAttention changes prompt encoding!", cos)
        else:
            log.info("Text embeddings consistent (cos=%.6f)", cos)

    disc = all_results.get("prompt_discriminability", {})
    log.info("Prompt discriminability: %s (MSE=%.2f)", disc.get("verdict", "?"), disc.get("mse", 0))

    iso = all_results.get("component_isolation", {})
    components = []
    for key, label in [("sage_off", "SageAttention"), ("deepcache_off", "DeepCache"),
                        ("freeu_off", "FreeU"), ("dc_plus_freeu_off", "DC+FreeU"),
                        ("all_off", "All optimizations")]:
        data = iso.get(key, {})
        mse = data.get("mse_vs_baseline")
        clip = data.get("clip_score")
        if mse is not None:
            components.append((label, mse, clip))

    if components:
        log.info("\nComponent impact ranking (MSE vs baseline — higher = more impact):")
        components.sort(key=lambda x: x[1], reverse=True)
        for label, mse, clip in components:
            clip_str = f", CLIP={clip:.4f}" if clip is not None else ""
            log.info("  %-20s MSE=%8.2f%s", label, mse, clip_str)
        clip_bl = iso.get("clip_baseline")
        if clip_bl is not None:
            log.info("  %-20s CLIP=%8.4f (reference)", "Baseline", clip_bl)

    dc = all_results.get("deepcache_detail", {})
    if dc.get("clip_full") and dc.get("clip_cached"):
        delta = dc["clip_full"] - dc["clip_cached"]
        if delta > 0.01:
            log.warning("\nDeepCache reduces prompt adherence (CLIP delta: +%.4f without cache)", delta)

    # Save full results as JSON
    report_path = OUTPUT_DIR / "diagnostics_report.json"

    def _serialize(obj):
        if isinstance(obj, float):
            return round(obj, 8)
        if isinstance(obj, np.floating):
            return round(float(obj), 8)
        return obj

    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_serialize)

    log.info("\nFull report saved to: %s", report_path)
    log.info("Images saved to: %s", OUTPUT_DIR)
    log.info("\nCompare images visually — look at:")
    log.info("  test4a_baseline.png        → Current production output")
    log.info("  test4b_native_sdpa.png     → Without SageAttention")
    log.info("  test4c_no_deepcache.png    → Without DeepCache")
    log.info("  test4d_no_freeu.png        → Without FreeU")
    log.info("  test4f_all_off.png         → Bare pipeline (most faithful to prompt)")
    log.info("  test4g_all_off_prompt_*.png → Discriminability with bare pipeline")


if __name__ == "__main__":
    main()
