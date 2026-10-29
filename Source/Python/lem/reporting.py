# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Pure data-formatting helpers for replay_train reporting.

Extracted from replay_train.py — builds JSONL records and log message strings.
No model, optimizer, or training infrastructure imports.
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger("replay_train")

# ── Timing key orderings (shared between stall + inference report) ──────────

CPU_TIMING_KEYS = [
    "sync", "read", "build", "compute", "gpu_sync", "gc", "action", "ctx", "body", "store",
]
GPU_TIMING_KEYS = ["encode", "vision", "forward", "motor", "gaze", "voice"]

ASYNC_OP_ORDER = [
    "sample", "zero_grad", "gather_primary", "encode_batch", "build_context",
    "forward", "pred_heads", "gather_target", "encode_target", "wellness",
    "motor", "gaze", "voice", "imagination", "loss", "backward", "pace", "clip_grad",
    "optimizer_step",
]


# ── Shared helper ───────────────────────────────────────────────────────────

def format_timing_breakdown(
    timings: dict[str, float],
    frame_time_ms: float,
) -> str:
    """Format CPU + GPU timing keys into ``" | "``-joined breakdown string.

    ``rest`` = *frame_time_ms* minus sum of CPU keys only (GPU keys overlap
    with ``gpu_sync`` and are informational).
    """
    parts: list[str] = []
    for k in CPU_TIMING_KEYS + GPU_TIMING_KEYS:
        if k in timings:
            parts.append(f"{k}={timings[k]:.1f}")
    cpu_sum = sum(timings.get(k, 0) for k in CPU_TIMING_KEYS)
    rest = frame_time_ms - cpu_sum
    parts.append(f"rest={rest:.1f}")
    return " | ".join(parts)


# ── Stall logging ──────────────────────────────────────────────────────────

def log_stall(
    inference_step: int,
    inference_time_ms: float,
    timings: dict[str, float],
    train_op: str,
    gpu_busy: bool,
    gc_event: str | None,
    alloc_delta: int | None,
    retry_delta: int,
    build_detail: dict[str, float] | None,
) -> None:
    """Emit a ``log.warning`` for a slow (>60 ms) inference frame."""
    breakdown = format_timing_breakdown(timings, inference_time_ms)
    alloc_str = ""
    if alloc_delta is not None and alloc_delta != 0:
        alloc_str = f" | alloc_delta={alloc_delta / 1024:.0f}KB"
    if retry_delta > 0:
        alloc_str += f" | alloc_retries={retry_delta}"
    build_str = ""
    if build_detail:
        build_str = " | build_detail={" + ", ".join(
            f"{k}:{v:.1f}" for k, v in build_detail.items()) + "}"
    log.warning(
        f"  STALL frame {inference_step}: {inference_time_ms:.1f}ms | "
        + breakdown
        + f" | train_op={train_op} | gpu_busy={gpu_busy}"
        + alloc_str + build_str + (gc_event or ""))


# ── Inference reporting ────────────────────────────────────────────────────

def format_inference_log(
    inference_step: int,
    fps: float,
    avg_ms: float,
    buffer_size: int,
    buffer_capacity: int,
    buffer_fill_pct: float,
    frame_number: int,
    phase_name: str,
    motor_confidence: float,
    has_fresh_vision: bool,
    timing_str: str,
) -> tuple[str, str]:
    """Return two log-message strings for the periodic inference report."""
    main = (
        f"[Inference {inference_step}] {fps:.1f} FPS ({avg_ms:.1f} ms) | "
        f"buffer={buffer_size}/{buffer_capacity} "
        f"({buffer_fill_pct:.0%}) | "
        f"frame={frame_number} | "
        f"phase={phase_name} | "
        f"confidence={motor_confidence:.3f}")
    detail = f"  timing(ms): {timing_str} | fresh_vision={has_fresh_vision}"
    return main, detail


def build_inference_stats_record(
    inference_step: int,
    frame_number: int,
    fps: float,
    avg_ms: float,
    buffer_fill: float,
    buffer_size: int,
    phase_name: str,
    motor_confidence: float,
    has_fresh_vision: bool,
    source_mask: object,
    stall_count: int,
    timings: dict[str, float] | None,
    vram_mb: dict[str, float] | None,
    peak_vram_mb: float | None,
    alloc_retries: int | None,
    transfer_times: dict[str, float] | None,
    gpu_info: dict | None,
) -> dict:
    """Build the ``type: inference`` JSONL stats record."""
    rec: dict = {
        "type": "inference",
        "inference_step": inference_step,
        "timestamp": time.time(),
        "frame_number": frame_number,
        "fps": round(fps, 1),
        "frame_time_ms": round(avg_ms, 1),
        "buffer_fill": buffer_fill,
        "buffer_size": buffer_size,
        "phase": phase_name,
        "motor_confidence": round(motor_confidence, 4),
        "has_fresh_vision": has_fresh_vision,
        "sources": source_mask,
        "stall_count": stall_count,
    }
    if timings:
        rec["inference_timings"] = {k: round(v, 1) for k, v in timings.items()}
    if vram_mb is not None:
        rec["vram_mb"] = vram_mb
    if peak_vram_mb is not None:
        rec["peak_vram_mb"] = peak_vram_mb
    if alloc_retries is not None:
        rec["alloc_retries"] = alloc_retries
    if transfer_times is not None:
        rec["transfer_times_ms"] = transfer_times
    if gpu_info is not None:
        rec["gpu_info"] = gpu_info
    return rec


# ── Sync training reporting ────────────────────────────────────────────────

def format_sync_training_log(
    training_step: int,
    avg_loss: float,
    avg_grad: float,
    batch_size: int,
    avg_train_ms: float,
    loss_bd: dict | None,
) -> str:
    """Return the ``──── [Train …] ────`` log string.

    *loss_bd* may contain raw ``torch.Tensor`` values — ``.item()`` is called
    here so the caller doesn't need to pre-serialize for the log line.
    """
    bd_str = ""
    if loss_bd is not None:
        parts: list[str] = []
        for key in ("prediction", "motor", "wellness", "gaze", "voice",
                     "vicreg", "imagination", "latent_reg"):
            if key == "prediction" and isinstance(loss_bd.get(key), dict):
                for mod, val in loss_bd[key].items():
                    parts.append(f"pred_{mod}={val.item():.4f}")
            elif key in loss_bd:
                parts.append(f"{key}={loss_bd[key].item():.4f}")
        bd_str = " | " + " ".join(parts)
    return (
        f"──── [Train {training_step}] loss={avg_loss:.4f} | "
        f"grad={avg_grad:.3f} | "
        f"B={batch_size} | "
        f"{avg_train_ms:.0f} ms/step{bd_str} ────")


def build_sync_training_stats_record(
    training_step: int,
    avg_loss: float,
    avg_grad: float,
    avg_train_ms: float,
    batch_size: int,
    buffer_fill: float,
    phase_name: str,
    train_mean_pe: float | None,
    loss_breakdown: dict[str, float] | None,
    step_timings: dict[str, float] | None,
    priority_beta: float | None,
    per_sample_mse_stats: dict[str, float] | None,
    grad_norms: dict | None,
    encoder_activations: dict | None,
    temporal_attention: dict | None,
    coherence_score: float,
    peak_vram_mb: float | None,
) -> dict:
    """Build the ``type: training`` JSONL stats record (sync path)."""
    rec: dict = {
        "type": "training",
        "training_step": training_step,
        "timestamp": time.time(),
        "avg_loss": round(avg_loss, 6),
        "avg_grad_norm": round(avg_grad, 4),
        "train_time_ms": round(avg_train_ms, 1),
        "batch_size": batch_size,
        "buffer_fill": buffer_fill,
        "phase": phase_name,
        "train_mean_pe": round(train_mean_pe, 6) if train_mean_pe is not None else None,
    }
    if loss_breakdown:
        rec["loss_breakdown"] = loss_breakdown
    if step_timings:
        rec["step_timings"] = {k: round(v, 1) for k, v in step_timings.items()}
    if priority_beta is not None:
        rec["priority_beta"] = round(priority_beta, 4)
    if per_sample_mse_stats is not None:
        rec["per_sample_mse"] = per_sample_mse_stats
    if grad_norms:
        rec["grad_norms"] = grad_norms
    if encoder_activations:
        rec["encoder_activations"] = encoder_activations
    if temporal_attention:
        rec["temporal_attention"] = temporal_attention
    rec["coherence_score"] = round(coherence_score, 4)
    if peak_vram_mb is not None:
        rec["peak_vram_mb"] = peak_vram_mb
    return rec


# ── Async training reporting ───────────────────────────────────────────────

def format_async_training_log(
    training_step: int,
    avg_loss: float,
    avg_grad: float,
    batch_size: int,
    avg_train_ms: float,
    step_timings_accum: dict[str, list[float]] | None,
) -> list[str]:
    """Return 1-2 log-message strings for the async training report."""
    msgs = [
        f"[AsyncTrain {training_step}] loss={avg_loss:.4f} | "
        f"grad={avg_grad:.3f} | B={batch_size} | "
        f"{avg_train_ms:.0f} ms/step"
    ]
    if step_timings_accum:
        timing_parts: list[str] = []
        for op in ASYNC_OP_ORDER:
            vals = step_timings_accum.get(op)
            if vals:
                avg = sum(vals) / len(vals)
                timing_parts.append(f"{op}={avg:.1f}")
        if timing_parts:
            msgs.append(f"  train_timing(ms): {' | '.join(timing_parts)}")
    return msgs


def build_async_training_stats_record(
    training_step: int,
    avg_loss: float,
    avg_grad: float,
    avg_train_ms: float,
    batch_size: int,
    buffer_fill: float,
    phase_name: str,
    loss_breakdown: dict | None,
    train_mean_pe: float | None,
    step_timings: dict[str, float] | None,
    grad_norms: dict | None,
    encoder_activations: dict | None,
    temporal_attention: dict | None,
    coherence_score: float,
    peak_vram_mb: float | None,
) -> dict:
    """Build the ``type: training`` JSONL stats record (async path)."""
    rec: dict = {
        "type": "training",
        "training_step": training_step,
        "timestamp": time.time(),
        "avg_loss": round(avg_loss, 6),
        "avg_grad_norm": round(avg_grad, 4),
        "train_time_ms": round(avg_train_ms, 1),
        "batch_size": batch_size,
        "buffer_fill": buffer_fill,
        "phase": phase_name,
        "async": True,
    }
    if loss_breakdown is not None:
        rec["loss_breakdown"] = loss_breakdown
    if train_mean_pe is not None:
        rec["train_mean_pe"] = round(train_mean_pe, 6)
    if step_timings:
        rec["step_timings"] = step_timings
    if grad_norms is not None:
        rec["grad_norms"] = grad_norms
    if encoder_activations is not None:
        rec["encoder_activations"] = encoder_activations
    if temporal_attention is not None:
        rec["temporal_attention"] = temporal_attention
    rec["coherence_score"] = round(coherence_score, 4)
    if peak_vram_mb is not None:
        rec["peak_vram_mb"] = peak_vram_mb
    return rec


# ── Metrics record ─────────────────────────────────────────────────────────

def build_metrics_record(
    training_step: int,
    inference_step: int,
    inference_fps: float,
    inference_ms: float,
    train_ms: float,
    vram_mb: float,
    peak_vram_mb: float,
    buffer_fill: float,
    phase_name: str,
    green_contexts: bool,
    async_training: bool,
    batch_size: int,
    step_timings: dict[str, float] | None,
) -> dict:
    """Build the persistent metrics JSONL record."""
    rec: dict = {
        "step": training_step,
        "timestamp": time.time(),
        "inference_fps": round(inference_fps, 1),
        "inference_ms": round(inference_ms, 1),
        "train_ms": round(train_ms, 1),
        "train_steps_per_s": round(1000 / train_ms, 2) if train_ms > 0 else 0,
        "vram_mb": round(vram_mb, 0),
        "peak_vram_mb": round(peak_vram_mb, 0),
        "buffer_fill": round(buffer_fill, 3),
        "inference_step": inference_step,
        "phase": phase_name,
        "green_contexts": green_contexts,
        "async_training": async_training,
        "batch_size": batch_size,
    }
    if step_timings:
        rec["step_timings"] = step_timings
    return rec


# ── Summary / final ────────────────────────────────────────────────────────

def format_summary_log(
    inference_step: int,
    inf_fps: float,
    train_fps: float,
    buffer_size: int,
    buffer_capacity: int,
    total_written: int,
    vram_mb: float,
    peak_mb: float,
) -> str:
    """Return the periodic 1000-step summary log string."""
    return (
        f"[Summary {inference_step}] inference={inf_fps:.1f} FPS | "
        f"training={train_fps:.1f} steps/s | "
        f"buffer={buffer_size}/{buffer_capacity} "
        f"({total_written} total stored) | "
        f"VRAM={vram_mb:.0f}/{peak_mb:.0f} MB")


def format_final_summary(
    training_steps: int,
    inference_frames: int,
    avg_train_ms: float,
    total_written: int,
    buffer_size: int,
    buffer_capacity: int,
    peak_vram_mb: float | None,
    device_name: str | None,
) -> list[str]:
    """Return 2-3 log-message strings for the end-of-run summary."""
    msgs = [
        f"Done. {training_steps} training steps "
        f"({inference_frames} inference frames), "
        f"avg train={avg_train_ms:.1f} ms/step",
        f"Buffer: {total_written} total frames stored, "
        f"final fill={buffer_size}/{buffer_capacity}",
    ]
    if peak_vram_mb is not None and device_name is not None:
        msgs.append(f"Peak VRAM ({device_name}): {peak_vram_mb:.0f} MB")
    return msgs
