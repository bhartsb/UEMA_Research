# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
PlaceholderSystem — feeds synthetic data for modalities with data_source='synthetic'.

Prevents the model from learning to ignore blocked modalities.
Reads resolved config, builds one generator per synthetic modality.
Each generator produces tensors matching the exact shape the encoder expects.
"""

from __future__ import annotations

from typing import Dict

import torch

from .strategies import (
    STRATEGY_REGISTRY,
    GaussianStrategy,
    SparseContactStrategy,
    ReplayBufferStrategy,
)


class PlaceholderSystem:
    """
    Manages synthetic data generation for blocked modalities.

    Built from resolved config. Only builds generators for modalities
    where data_source='synthetic'.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        self._generators: Dict[str, object] = {}
        self._shapes: Dict[str, list] = {}
        self._modality_configs: Dict[str, dict] = {}

        d_model = config["model"]["d_model"]
        body = config["body"]

        for mod_name in config["_active_modalities"]:
            mod_cfg = config["modalities"][mod_name]
            if mod_cfg["data_source"] != "synthetic":
                continue

            placeholder_cfg = mod_cfg["placeholder"]
            strategy_name = placeholder_cfg["strategy"]
            pipe_cfg = mod_cfg.get("pipeline", {})
            enc_cfg = mod_cfg.get("encoder", {})

            # Determine shape for this modality
            shape = self._resolve_shape(mod_name, pipe_cfg, enc_cfg, body, d_model)
            self._shapes[mod_name] = shape
            self._modality_configs[mod_name] = mod_cfg

            # Build strategy
            if strategy_name == "gaussian":
                self._generators[mod_name] = GaussianStrategy(
                    shape=shape,
                    mean=placeholder_cfg.get("mean", 0.0),
                    std=placeholder_cfg.get("std", 0.1),
                    clip=placeholder_cfg.get("clip", (0.0, 1.0)),   # P012: default keeps today's draw
                    device=device,
                )
            elif strategy_name == "sparse_contact":
                self._generators[mod_name] = SparseContactStrategy(
                    shape=shape,
                    n_active=placeholder_cfg.get("n_active", 3),
                    force_mean=placeholder_cfg.get("force_mean", 0.3),
                    force_std=placeholder_cfg.get("force_std", 0.15),
                    temporal_correlation=placeholder_cfg.get("temporal_correlation", 0.0),
                    device=device,
                )
            elif strategy_name == "replay_buffer":
                gen = ReplayBufferStrategy(
                    shape=shape,
                    device=device,
                    jitter_std=placeholder_cfg.get("jitter_std", 0.01),
                )
                self._generators[mod_name] = gen
            else:
                raise ValueError(
                    f"Unknown placeholder strategy '{strategy_name}' "
                    f"for modality '{mod_name}'"
                )

    @staticmethod
    def _resolve_shape(mod_name: str, pipe_cfg: dict, enc_cfg: dict,
                       body: dict, d_model: int) -> list[int]:
        """Determine the tensor shape for a modality from config."""
        if mod_name == "vision":
            n_patches = enc_cfg.get("_n_patches", 32506)
            patch_dim = enc_cfg.get("_patch_dim", 768)
            return [n_patches, patch_dim]
        elif mod_name == "tactile":
            return list(pipe_cfg.get("shape", [
                pipe_cfg.get("n_sensors", 256),
                pipe_cfg.get("features_per_sensor", 8),
            ]))
        elif mod_name == "audio":
            return [pipe_cfg.get("fft_n_bins", 256), pipe_cfg.get("n_channels", 2)]
        elif mod_name == "proprioception":
            return [enc_cfg.get("_input_dim", 527)]
        elif mod_name == "motor":
            return [pipe_cfg.get("n_handles", body["n_handles"]),
                    pipe_cfg["features_per_handle"]]
        elif mod_name == "intermind":
            n_agents = pipe_cfg.get("n_agents", body["n_agents"])
            embed_dim = enc_cfg.get("_embed_input_dim", d_model + 3)
            return [n_agents, embed_dim]
        elif mod_name == "voice":
            return [enc_cfg.get("_input_dim", pipe_cfg.get("input_dim", 56))]
        else:
            raise ValueError(f"Unknown modality for shape resolution: {mod_name}")

    @property
    def synthetic_modalities(self) -> list[str]:
        return list(self._generators.keys())

    def fill_frame(self, frame: dict,
                   batch_size: int = 1) -> Dict[str, str]:
        """
        Fill synthetic data into frame dict for synthetic modalities.

        Args:
            frame: Mutable dict of per-modality tensors. Synthetic modalities
                   will be added/overwritten.
            batch_size: Batch size for generated tensors.

        Returns:
            source_mask entries: {modality: "synthetic"} for each filled modality.
        """
        source_entries = {}
        for mod_name, gen in self._generators.items():
            frame[mod_name] = gen.generate(batch_size)
            source_entries[mod_name] = "synthetic"
        return source_entries
