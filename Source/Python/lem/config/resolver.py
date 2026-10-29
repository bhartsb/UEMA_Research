# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
resolve(raw_config) -> dict

The most critical function in the LEM config system.

Takes a raw master_config dict (merged with defaults) and returns a fully-resolved
config with all underscore-prefixed derived fields computed. Pure function, no side
effects.

Derived fields computed:
  _active_modalities          — list of enabled modality names (order-preserving)
  _prediction_head_modalities — _active minus 'motor'
  _seq_length                 — n_active * context_length
  _n_modalities               — len(_active_modalities)
  _d_output                   — per-encoder: always == d_model
  _d_internal                 — per-encoder: resolved from d_internal or d_model
  _input_dim                  — proprioception: pipeline.input_dim (explicit) or formula fallback
  _input_dim                  — motor_controller: d_model * n_active + n_handles * features_per_handle
  _embed_input_dim            — intermind: d_model + features_per_agent
  _n_patches                  — vision: computed from panorama + fovea resolution
  _patch_dim                  — vision: patch_size^2 * 3
  _input_dim                  — wellness.approximate: d_model * (n_active - 1)
  SHM layout                  — per-modality offsets with 256-byte alignment
"""

from __future__ import annotations

import copy
import math
from typing import Any

from .defaults import get_defaults

# Modalities the approximate wellness evaluator (Judge B) never reads. Written here as well as in
# lem.model.wellness (APPROXIMATE_EXCLUDED) because this module must not import torch;
# tests/test_vestibular_sense.py asserts the two are equal.
APPROXIMATE_EXCLUDED = ("intermind", "vestibular")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override values win."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _align_256(size: int) -> int:
    """Align size up to 256-byte boundary."""
    return (size + 255) & ~255



def _compute_n_patches(panorama_resolution: list, fovea_resolution: list,
                       patch_size: int) -> int:
    """Compute total patch count from panorama and fovea resolution."""
    pano_w, pano_h = panorama_resolution
    fovea_w, fovea_h = fovea_resolution

    pano_cols = pano_w // patch_size
    pano_rows = pano_h // patch_size
    patches_per_pano = pano_cols * pano_rows

    fovea_cols = fovea_w // patch_size
    fovea_rows = fovea_h // patch_size
    patches_per_fovea = fovea_cols * fovea_rows

    # 2 eyes * (pano + fovea)
    total = 2 * (patches_per_pano + patches_per_fovea)
    return total


def resolve(raw_config: dict) -> dict:
    """
    Compute all underscore-prefixed derived fields from raw config.

    Pure function. No side effects. No file I/O.
    """
    # Step 0: Merge defaults with user config (user wins)
    defaults = get_defaults()
    cfg = _deep_merge(defaults, raw_config)

    # Step 0b: THE VESTIBULAR SENSE TAKES THE INTERMIND PLACEHOLDER'S SLOT
    # (docs/gain_of_function_v3/DESIGN.md D4, --vestibular-sense on). Modality order IS the
    # model's input layout: the forward model's modality embedding row, the motor
    # controller's input columns and every saved context are indexed by position in
    # _active_modalities. A modality the defaults do not know is appended AFTER them, so
    # without this step "vestibular" would land after "voice" and move to a different slot
    # the day voice is switched on. It is placed directly after "intermind", so with
    # intermind disabled it takes intermind's index (5 on every bridge run). A config without
    # a "vestibular" entry - every launch without the flag - is not touched at all.
    if "vestibular" in cfg.get("modalities", {}) and "intermind" in cfg["modalities"]:
        _order = [k for k in cfg["modalities"] if k != "vestibular"]
        _order.insert(_order.index("intermind") + 1, "vestibular")
        cfg["modalities"] = {k: cfg["modalities"][k] for k in _order}

    body = cfg["body"]
    modalities = cfg["modalities"]
    model = cfg["model"]
    enc_defaults = cfg["encoder_defaults"]
    d_model = model["d_model"]
    context_length = model["context_length"]

    # =========================================================================
    # Step 1: Filter active modalities (order-preserving)
    # =========================================================================
    active = [name for name in modalities if modalities[name].get("enabled", False)]
    cfg["_active_modalities"] = active
    cfg["_n_modalities"] = len(active)

    # =========================================================================
    # Step 2: Prediction head modalities (active minus motor)
    # =========================================================================
    cfg["_prediction_head_modalities"] = [m for m in active if m != "motor"]

    # =========================================================================
    # Step 3: Merge encoder_defaults into each active modality encoder
    # =========================================================================
    merge_keys = [
        "bias", "dropout", "activation", "norm_type",
        "pool_type", "pool_n_heads", "ffn_expansion", "d_internal",
    ]
    for mod_name in modalities:
        enc = modalities[mod_name].get("encoder", {})
        for key in merge_keys:
            if key not in enc:
                enc[key] = enc_defaults[key]
        modalities[mod_name]["encoder"] = enc

    # =========================================================================
    # Step 4: Resolve d_internal and _d_output for each encoder
    # =========================================================================
    for mod_name in modalities:
        enc = modalities[mod_name]["encoder"]
        if enc.get("d_internal") is None:
            enc["d_internal"] = d_model
        enc["_d_internal"] = enc["d_internal"]
        enc["_d_output"] = d_model

    # =========================================================================
    # Step 4b: Resolve per-modality encoder internal dims
    #
    # Explicit values are used as-is. null → d_model.
    # =========================================================================
    for mod_name in modalities:
        enc = modalities[mod_name]["encoder"]
        enc_type = enc.get("type", "")

        if enc_type in ("cross_patch_attention", "cross_handle_attention"):
            if enc.get("per_patch_hidden") is None:
                enc["per_patch_hidden"] = d_model
            # slot count for the optional per-patch identity embedding
            pipe = modalities[mod_name].get("pipeline", {})
            enc["_n_patches_for_pos"] = int(pipe.get("n_sensors", pipe.get("n_handles", 0)) or 0)

        if enc_type == "conv1d_stack":
            if enc.get("conv_channels") is None:
                enc["conv_channels"] = [d_model, d_model, d_model]

        if enc_type == "mlp_project":
            if enc.get("hidden_dims") is None:
                enc["hidden_dims"] = [d_model, d_model]

        if enc_type == "swin_patch":
            if enc.get("angular_pos_hidden") is None:
                enc["angular_pos_hidden"] = d_model

    # =========================================================================
    # Step 5: Vision-specific derived fields
    # =========================================================================
    if "vision" in modalities:
        vis = modalities["vision"]
        vis_pipe = vis["pipeline"]
        vis_enc = vis["encoder"]
        patch_size = vis_enc["patch_size"]

        pano_res = vis_pipe["panorama_resolution"]
        fovea_res = vis_pipe["fovea_resolution"]

        vis_enc["_n_patches"] = _compute_n_patches(pano_res, fovea_res, patch_size)
        vis_enc["_patch_dim"] = patch_size * patch_size * 3  # RGB

    # =========================================================================
    # Step 6: Proprioception _input_dim
    # =========================================================================
    if "proprioception" in modalities:
        # Use explicit pipeline.input_dim if set (matches current tokenizer output).
        # Otherwise fall back to full-skeleton formula for future use.
        explicit_dim = modalities["proprioception"].get("pipeline", {}).get("input_dim")
        if explicit_dim is not None:
            proprio_dim = explicit_dim
        else:
            n_joints = body["n_joints"]
            n_handles = body["n_handles"]
            n_bodysets = body["n_bodysets"]
            # joints*3 (rotation euler) + joints*3 (translation) + handles*3 (position)
            # + handles*7 (quaternion+pos offset) + bodysets (fatigue) + bodysets (energy) + 1 (global)
            proprio_dim = (n_joints * 3 + n_joints * 3 + n_handles * 3
                           + n_handles * 7 + n_bodysets + n_bodysets + 1)
        modalities["proprioception"]["encoder"]["_input_dim"] = proprio_dim

    # =========================================================================
    # Step 7: Intermind _embed_input_dim
    # =========================================================================
    if "intermind" in modalities:
        features_per_agent = modalities["intermind"]["pipeline"].get(
            "features_per_agent", 3)
        modalities["intermind"]["encoder"]["_embed_input_dim"] = d_model + features_per_agent

    # =========================================================================
    # Step 8: Voice _input_dim (if present)
    # =========================================================================
    if "voice" in modalities:
        voice_input = modalities["voice"]["pipeline"].get("input_dim", 56)
        modalities["voice"]["encoder"]["_input_dim"] = voice_input

    # =========================================================================
    # Step 8a: Vestibular _input_dim (if present): the width of the body's
    # vestibular / internal-state vector, mimo_bridge/vestibular_adapter.VESTIBULAR_DIM,
    # which the modality's config carries as pipeline.input_dim
    # =========================================================================
    if "vestibular" in modalities:
        modalities["vestibular"]["encoder"]["_input_dim"] = int(
            modalities["vestibular"]["pipeline"]["input_dim"])

    # =========================================================================
    # Step 8b: Audio — derive spatial_branch_input_dim from pipeline.spatial_features
    # =========================================================================
    if "audio" in modalities:
        audio_enc = modalities["audio"]["encoder"]
        audio_pipe = modalities["audio"].get("pipeline", {})
        if "spatial_branch_input_dim" not in audio_enc:
            audio_enc["spatial_branch_input_dim"] = audio_pipe.get("spatial_features", 2)

    # =========================================================================
    # Step 9: Forward model derived fields
    # =========================================================================
    n_active = len(active)
    fwd = model["forward_model"]
    fwd["_seq_length"] = n_active * context_length
    fwd["_n_modalities"] = n_active

    # =========================================================================
    # Step 10: Motor controller _input_dim
    # =========================================================================
    mc = model["motor_controller"]
    # Input = pooled latents (d_model * n_active) + flat handle state (n_handles * features_per_handle).
    # features_per_handle is the FULL tokenizer output per handle (10: pos+rot+radius+strength+is_active).
    # output_per_handle is what the controller PREDICTS (9: pos+rot+radius+strength).
    motor_input_features = modalities.get("motor", {}).get("pipeline", {}).get(
        "features_per_handle", mc["output_per_handle"])
    wellness_grad_dim = body["n_handles"] * mc["output_per_handle"]  # 23 * 9 = 207
    mc["_input_dim"] = d_model * n_active + body["n_handles"] * motor_input_features + wellness_grad_dim

    # =========================================================================
    # Step 11: Wellness approximate _input_dim
    # =========================================================================
    # n_sensory = active minus intermind (stub modality) and minus vestibular: the approximate
    # evaluator keeps its five-token input when the vestibular sense takes intermind's slot
    # (DESIGN.md D4), so a checkpoint's evaluator loads unchanged. The same two names are
    # lem.model.wellness.APPROXIMATE_EXCLUDED; a test holds the two lists together.
    n_sensory = len([m for m in active if m not in APPROXIMATE_EXCLUDED])
    wellness = model["wellness"]
    wellness["approximate"]["_input_dim"] = d_model * n_sensory

    # =========================================================================
    # Step 12: SHM layout (per-modality offsets with 256-byte alignment)
    # =========================================================================
    shm_layout = {}
    current_offset = 0  # After SHM header (handled separately)

    # Define SHM slot sizes for each modality
    shm_slot_sizes = {}
    for mod_name in active:
        mod_cfg = modalities[mod_name]
        pipe = mod_cfg.get("pipeline", {})
        fmt = pipe.get("format", "")

        if mod_name == "vision":
            # Vision is GPU-resident, separate SHM region
            pano_res = pipe["panorama_resolution"]
            fovea_res = pipe["fovea_resolution"]
            pano_bytes = pano_res[0] * pano_res[1] * 4  # BGRA8
            fovea_bytes = fovea_res[0] * fovea_res[1] * 4
            shm_slot_sizes[mod_name] = 2 * (pano_bytes + fovea_bytes)  # Both eyes
        elif mod_name == "audio":
            shm_slot_sizes[mod_name] = 8192  # Fixed audio slice size
        elif mod_name == "proprioception":
            shm_slot_sizes[mod_name] = 4096
        elif mod_name == "motor":
            shm_slot_sizes[mod_name] = 2048
        elif mod_name == "tactile":
            n_sensors = pipe.get("n_sensors", 256)
            features = pipe.get("features_per_sensor", 8)
            raw_size = n_sensors * features * 4 + 16  # Data + header
            shm_slot_sizes[mod_name] = _align_256(raw_size)
        elif mod_name == "intermind":
            n_agents = pipe.get("n_agents", 5)
            features = pipe.get("features_per_agent", 3)
            raw_size = n_agents * (d_model + features) * 4 + 16
            shm_slot_sizes[mod_name] = _align_256(raw_size)
        elif mod_name == "voice":
            shm_slot_sizes[mod_name] = 256
        else:
            # Generic: estimate from pipeline config
            raw_size = 1024
            shm_slot_sizes[mod_name] = _align_256(raw_size)

    # Compute aligned offsets
    for mod_name in active:
        aligned_size = _align_256(shm_slot_sizes[mod_name])
        shm_layout[mod_name] = {
            "offset": current_offset,
            "size": shm_slot_sizes[mod_name],
            "aligned_size": aligned_size,
        }
        current_offset += aligned_size

    # Double buffering: total = sum * 2
    cfg["_shm_layout"] = shm_layout
    cfg["_shm_total_per_slot"] = current_offset
    cfg["_shm_total"] = current_offset * 2  # Double buffer

    return cfg
