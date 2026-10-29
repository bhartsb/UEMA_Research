# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Manifest system — reads pipeline_manifest.json, writes tokenizer_manifest.json.

The pipeline manifest describes the raw buffer layout and modality metadata.
The tokenizer manifest describes the normalized output tensors and their semantics.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import TokenizerConfig

log = logging.getLogger(__name__)


def load_pipeline_manifest(path: str) -> Optional[dict]:
    """Load pipeline_manifest.json if it exists. Returns None on error."""
    p = Path(path)
    if not p.exists():
        log.info("Pipeline manifest not found at '%s' — using defaults", path)
        return None
    try:
        with open(p) as f:
            manifest = json.load(f)
        log.info("Pipeline manifest loaded: %d top-level keys", len(manifest))
        return manifest
    except Exception as e:
        log.warning("Failed to load pipeline manifest '%s': %s", path, e)
        return None


def apply_manifest_overrides(config: TokenizerConfig, manifest: dict) -> TokenizerConfig:
    """Override TokenizerConfig fields from pipeline manifest values."""
    vision = manifest.get("vision", {})
    if "panorama_width" in vision:
        config.panorama_width = vision["panorama_width"]
    if "panorama_height" in vision:
        config.panorama_height = vision["panorama_height"]
    if "fovea_width" in vision:
        config.fovea_width = vision["fovea_width"]
    if "fovea_height" in vision:
        config.fovea_height = vision["fovea_height"]
    if "fovea_fov_degrees" in vision:
        config.fovea_fov_deg = vision["fovea_fov_degrees"]
    if "panorama_fov_h_degrees" in vision:
        config.panorama_fov_h_deg = vision["panorama_fov_h_degrees"]
    if "panorama_fov_v_degrees" in vision:
        config.panorama_fov_v_deg = vision["panorama_fov_v_degrees"]

    audio = manifest.get("audio", {})
    if "sample_rate" in audio:
        config.audio_sample_rate = audio["sample_rate"]

    # GPU interop (D3D12→CUDA zero-copy)
    gpu_interop = manifest.get("gpu_interop", {})
    if gpu_interop:
        config.gpu_interop_available = True
        config.adapter_luid_low = int(gpu_interop.get("adapter_luid_low", 0))
        config.adapter_luid_high = int(gpu_interop.get("adapter_luid_high", 0))
        config.fence_handle_name = gpu_interop.get("fence_name", "")
        config.texture_alloc_sizes = gpu_interop.get("allocation_sizes", {})
        config.parity_handle_names = gpu_interop.get("parity_handles", {})
        log.info("GPU interop available: adapter LUID=0x%08X%08X, fence=%s",
                 config.adapter_luid_high, config.adapter_luid_low, config.fence_handle_name)

    log.info("Config updated from manifest: pano=%dx%d, fovea=%dx%d (%.1f deg), sr=%d",
             config.panorama_width, config.panorama_height,
             config.fovea_width, config.fovea_height,
             config.fovea_fov_deg, config.audio_sample_rate)

    return config


def build_tokenizer_manifest(config: TokenizerConfig) -> dict:
    """Build the tokenizer manifest dict describing all output shapes and semantics."""
    ps = config.patch_size
    pano_crop_w = (config.panorama_width // ps) * ps
    pano_crop_h = (config.panorama_height // ps) * ps
    pano_cols = pano_crop_w // ps
    pano_rows = pano_crop_h // ps
    pano_patches = pano_cols * pano_rows

    fovea_cols = config.fovea_width // ps
    fovea_rows = config.fovea_height // ps
    fovea_patches = fovea_cols * fovea_rows

    total_patches = (pano_patches + fovea_patches) * 2

    return {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_manifest": config.pipeline_manifest_path,
        "frame_metadata": {
            "fields": [
                "frame_number", "timestamp", "wall_clock", "dt",
                "gaze_azimuth_deg", "gaze_elevation_deg", "modality_valid_mask",
            ],
            "note": "dt is variable - no fixed tick rate",
        },
        "modalities": {
            "vision": {
                "patches_shape": [total_patches, ps, ps, 3],
                "angular_pos_shape": [total_patches, 2],
                "dtype": "float32",
                "device": config.device,
                "panorama_patches_per_eye": pano_patches,
                "fovea_patches_per_eye": fovea_patches,
                "eyes": 2,
                "patch_size": ps,
                "panorama_fov_h_deg": config.panorama_fov_h_deg,
                "panorama_fov_v_deg": config.panorama_fov_v_deg,
                "fovea_fov_deg": config.fovea_fov_deg,
                "color_space": "RGB",
                "normalization": "[0.0, 1.0]",
                "overlap_kept": True,
                "angular_encoding": "azimuth_elevation_radians",
                "concatenation_order": [
                    "panorama_left", "panorama_right",
                    "fovea_left", "fovea_right",
                ],
            },
            "audio": {
                "spectral_shape": [config.fft_n_bins, 2],
                "spectral_source": "ue_cpp_fft",
                "metadata_shape": [5],
                "metadata_fields": [
                    "ILD_dB_norm", "ITD_ms_norm", "azimuth_norm",
                    "front_back_confidence", "left_rms",
                ],
                "dtype": "float32",
                "fft_window_size": config.fft_window_size,
                "fft_n_bins": config.fft_n_bins,
                "sample_rate": config.audio_sample_rate,
                "spectral_normalization": "log_db_-80_to_0_mapped_0_to_1",
            },
            "proprioception": {
                "shape": [23, 12],
                "fields_per_bone": [
                    "quat_x", "quat_y", "quat_z", "quat_w",
                    "pos_x", "pos_y", "pos_z",
                    "vel_x", "vel_y", "vel_z",
                    "fatigue", "bone_index",
                ],
                "dtype": "float32",
                "n_bones": 23,
                "transform_space": "root_relative",
                "position_range_cm": config.position_range_cm,
                "velocity_range_cms": config.velocity_range_cms,
                "note": "n_bones will increase when full skeleton is wired",
            },
            "motor": {
                "handles_shape": [23, 10],
                "energy_shape": [3],
                "fields_per_handle": [
                    "target_pos_x", "target_pos_y", "target_pos_z",
                    "target_rot_x", "target_rot_y", "target_rot_z", "target_rot_w",
                    "radius", "strength", "is_active",
                ],
                "energy_fields": ["ph_work", "pcc_holding", "total"],
                "dtype": "float32",
            },
            "touch": {
                "status": "stub",
                "impulses_shape": [16, 8],
                "collisions_shape": [16, 8],
                "dtype": "float32",
                "note": "Will be replaced with fixed sensor array when tactile system is complete",
            },
            "voice": {
                "shape": [56],
                "fields": [
                    "freq_norm", "tenseness", "tongue_index_norm", "tongue_diam_norm",
                    "vibrato_amount_norm", "vibrato_freq_norm", "intensity", "loudness",
                    "velum_norm", "auto_wobble", "always_voice", "synth_time_wrapped",
                    *[f"diameter_{i}" for i in range(44)],
                ],
                "dtype": "float32",
            },
            "intermind": {
                "status": "stub",
                "shape": [3],
                "dtype": "float32",
                "note": "Zero vector until Many Minds is implemented",
            },
        },
    }


def write_tokenizer_manifest(config: TokenizerConfig, output_path: Optional[str] = None) -> dict:
    """Build and write tokenizer_manifest.json. Returns the manifest dict."""
    manifest = build_tokenizer_manifest(config)

    path = output_path or config.output_manifest_path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, 'w') as f:
        json.dump(manifest, f, indent=2)

    log.info("Tokenizer manifest written to '%s'", path)
    return manifest
