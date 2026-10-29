# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
emit_model_receipt(config, registry, fwd, motor, wellness) -> dict

Emits model_receipt.json from training init. Contains:
  - version, active_modalities, encoders_built
  - forward_model details, motor_controller details
  - prediction_head_modalities, total_params
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch.nn as nn


def emit_model_receipt(config: dict,
                       registry: Optional[nn.Module] = None,
                       forward_model: Optional[nn.Module] = None,
                       motor_controller: Optional[nn.Module] = None,
                       wellness: Optional[nn.Module] = None,
                       gaze_head: Optional[nn.Module] = None,
                       voice_head: Optional[nn.Module] = None,
                       prediction_heads: Optional[nn.Module] = None,
                       output_path: Optional[str] = None) -> dict:
    """
    Build model receipt from resolved config and built modules.

    Args:
        config: Fully-resolved config dict.
        registry: EncoderRegistry module (or None for config-only receipt).
        forward_model: ForwardModel module (or None).
        motor_controller: MotorController module (or None).
        wellness: WellnessEvaluators module (or None).
        output_path: If provided, write receipt JSON to this path.

    Returns:
        Model receipt dict.
    """
    d_model = config["model"]["d_model"]
    active = config["_active_modalities"]

    # Encoders built
    encoders_built = {}
    for mod_name in active:
        enc_cfg = config["modalities"][mod_name]["encoder"]
        encoders_built[mod_name] = {
            "type": enc_cfg["type"],
            "d_output": enc_cfg["_d_output"],
            "d_internal": enc_cfg.get("_d_internal", d_model),
        }

    # Total params
    total_params = 0
    for module in [registry, forward_model, motor_controller, wellness, gaze_head, voice_head]:
        if module is not None:
            total_params += sum(p.numel() for p in module.parameters())

    receipt = {
        "version": config.get("version", "unknown"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # the numerical regime every measured dtype on the page depends on
        "precision": config["model"].get("precision", "fp32"),
        "active_modalities": list(active),
        "encoders_built": encoders_built,
        "forward_model": {
            "_seq_length": config["model"]["forward_model"]["_seq_length"],
            "_n_modalities": config["model"]["forward_model"]["_n_modalities"],
            "n_layers": config["model"]["forward_model"]["n_layers"],
            "n_heads": config["model"]["forward_model"]["n_heads"],
            # WHICH stages were built and how wide the FFN is. The explainer drew
            # three stage boxes and "x4 expansion" from fixed strings, so an ablation
            # run would have rendered a core the model did not have, with nothing in
            # the receipt that could contradict it.
            "ffn_expansion": config["model"]["forward_model"]["ffn_expansion"],
            "stages": dict(config["model"]["forward_model"]["stages"]),
        },
        "motor_controller": {
            "_input_dim": config["model"]["motor_controller"]["_input_dim"],
            "output_per_handle": config["model"]["motor_controller"]["output_per_handle"],
            "n_layers": config["model"]["motor_controller"]["n_layers"],
            "hidden_dim": config["model"]["motor_controller"]["hidden_dim"],
            "recursive": config["model"]["motor_controller"]["recursive"],
            "adaptive": config["model"]["motor_controller"].get("adaptive", False),
            "max_depth": config["model"]["motor_controller"].get("max_depth", config["model"]["motor_controller"]["n_layers"]),
        },
        "gaze_head": {
            "enabled": config["model"].get("gaze_head", {}).get("enabled", False),
            "output_dim": config["model"].get("gaze_head", {}).get("output_dim", 2),
            "params": sum(p.numel() for p in gaze_head.parameters()) if gaze_head is not None else 0,
        },
        "voice_head": {
            "enabled": config["model"].get("voice_head", {}).get("enabled", False),
            "output_dim": (voice_head.mlp[-1].out_features if voice_head is not None
                           else config["model"].get("voice_head", {}).get("output_dim", 80)),
            "params": sum(p.numel() for p in voice_head.parameters()) if voice_head is not None else 0,
        },
        "prediction_head_modalities": list(config["_prediction_head_modalities"]),
        "wellness": {
            "approximate_input_dim": config["model"]["wellness"]["approximate"]["_input_dim"],
        },
        "total_params": total_params,
    }

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)

    return receipt
