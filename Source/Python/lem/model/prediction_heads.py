# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
PredictionHeads — one per active modality EXCEPT motor.

Each head: learned query + MHA + 2-layer MLP.
Auto-built from resolved config. When a modality is disabled, its head
disappears. Motor NEVER gets a prediction head.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """Single prediction head for one modality."""

    def __init__(self, d_model: int, n_heads: int,
                 hidden_expansion: int, n_mlp_layers: int):
        super().__init__()

        hidden_dim = d_model * hidden_expansion

        # Learned query
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # MHA: query attends to last-step modality tokens
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True,
        )

        # MLP
        layers = []
        in_dim = d_model
        for i in range(n_mlp_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, d_model))
        self.mlp = nn.Sequential(*layers)

    def forward(self, last_step: torch.Tensor) -> torch.Tensor:
        """
        Args:
            last_step: [B, n_mod, d_model] — last timestep tokens from forward model.
        Returns:
            [B, d_model] — predicted next-frame embedding for this modality.
        """
        B = last_step.shape[0]
        q = self.query.expand(B, -1, -1)
        q = self.norm(q)
        out, _ = self.attn(q, last_step, last_step)
        out = out.squeeze(1)  # [B, d_model]
        out = self.mlp(out)
        return out


class GazeHead(nn.Module):
    """
    Gaze target output: [B, 2] (azimuth, elevation in radians).

    Learned query attends to forward model last-step tokens,
    then MLP projects to 2D gaze, scaled by tanh to range limits.
    """

    def __init__(self, d_model: int, hidden_dim: int, n_heads: int,
                 azimuth_range: list, elevation_range: list):
        super().__init__()

        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

        # Range scaling: tanh output * scale + offset
        az_lo, az_hi = azimuth_range
        el_lo, el_hi = elevation_range
        self.register_buffer("az_scale", torch.tensor((az_hi - az_lo) / 2.0))
        self.register_buffer("az_offset", torch.tensor((az_hi + az_lo) / 2.0))
        self.register_buffer("el_scale", torch.tensor((el_hi - el_lo) / 2.0))
        self.register_buffer("el_offset", torch.tensor((el_hi + el_lo) / 2.0))

    def forward(self, last_step: torch.Tensor) -> torch.Tensor:
        """
        Args:
            last_step: [B, n_mod, d_model] from forward model.
        Returns:
            [B, 2] — (azimuth, elevation) in radians.
        """
        B = last_step.shape[0]
        q = self.query.expand(B, -1, -1)
        q = self.norm(q)
        out, _ = self.attn(q, last_step, last_step)
        out = out.squeeze(1)  # [B, d_model]
        raw = self.mlp(out)   # [B, 2]

        # Scale tanh output to configured ranges
        az = torch.tanh(raw[:, 0:1]) * self.az_scale + self.az_offset
        el = torch.tanh(raw[:, 1:2]) * self.el_scale + self.el_offset
        return torch.cat([az, el], dim=-1)  # [B, 2]


class VoiceHead(nn.Module):
    """
    Action-slot voice output: [B, output_dim] (default 80), each value 0 to 1. 80 is the
    width the shared-memory action slot reserves (lem/action/writer.py OFF_VOICE = 856,
    float32[80]); the vocal-tract parameter set this repo defines is 56 wide
    (tokenizer/modalities/voice.py: 11 named parameters, a wrapped synth clock, 44 tract
    diameters), so 24 outputs carry no named vocal-tract meaning. Nothing reads the slot
    yet: no UE code reads VoiceParams. (audit P248)

    Learned query attends to forward model last-step tokens,
    then MLP projects to output_dim with sigmoid activation.
    """

    def __init__(self, d_model: int, hidden_dim: int, n_heads: int, output_dim: int = 80):
        super().__init__()
        self.output_dim = int(output_dim)

        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, last_step: torch.Tensor) -> torch.Tensor:
        """
        Args:
            last_step: [B, n_mod, d_model] from forward model.
        Returns:
            [B, output_dim] -- action-slot voice numbers in [0, 1].
        """
        B = last_step.shape[0]
        q = self.query.expand(B, -1, -1)
        q = self.norm(q)
        out, _ = self.attn(q, last_step, last_step)
        out = out.squeeze(1)  # [B, d_model]
        return torch.sigmoid(self.mlp(out))  # [B, 80]


class PredictionHeads(nn.Module):
    """
    Collection of prediction heads — one per active modality except motor.

    Built dynamically from resolved config.
    """

    def __init__(self, config: dict):
        super().__init__()

        model_cfg = config["model"]
        heads_cfg = model_cfg["prediction_heads"]
        d_model = model_cfg["d_model"]

        self._head_modalities = list(config["_prediction_head_modalities"])

        heads = {}
        for mod_name in self._head_modalities:
            heads[mod_name] = PredictionHead(
                d_model=d_model,
                n_heads=heads_cfg["n_heads"],
                hidden_expansion=heads_cfg["hidden_expansion"],
                n_mlp_layers=heads_cfg["n_mlp_layers"],
            )

        self.heads = nn.ModuleDict(heads)
        if 'vision' in self.heads:
            from .input_conditioning import install
            install(self.heads['vision'].mlp[0])

        # RESIDUAL PREDICTION (2026-08-28, default off). With this on, a head predicts the
        # CHANGE and forward() adds the current token, so "nothing happens" is the zero
        # output instead of a mapping the network must learn and keep re-learning.
        #
        # Why it is a candidate at all, measured on H34's final checkpoint: persistence is
        # nearly as good as the model on every sense, so most of the target is reproducible
        # by copying the present and the action's marginal value is a few percent of the
        # loss; a two-layer MLP with 1,547 examples extracts 9.2% from the executed order
        # in predicting the CHANGE while the 3.5M model moves its prediction 0.0018 when
        # the commanded rotation is swapped; and the motor token's gradient is 23x smaller
        # than proprioception's. Predicting the change removes the copyable part, so what
        # the action explains becomes the whole task rather than a few percent of it.
        self.residual = bool(model_cfg.get("residual_prediction", False))

    @property
    def head_modalities(self) -> list[str]:
        return list(self._head_modalities)

    def _residual_base(self, current, mod_name, who):
        """The current token for this modality, or a loud failure.

        A caller that forgets `current` while the flag is on would train the model on
        changes and measure it on absolutes - the model trained one way and evaluated
        another, which is exactly how the copy-degeneration and the centimetre units
        defect each cost runs. So this raises rather than silently falling back.
        """
        if current is None:
            raise RuntimeError(
                "residual_prediction is ON and prediction heads were called without the "
                "current tokens (caller: %s). Every site that turns last_step into a "
                "prediction must pass current={modality: [B, d_model]} - the RAW encoder "
                "tokens the target is measured against, not the forward model's "
                "current_flat, which carries the modality embedding." % who)
        t = current.get(mod_name) if isinstance(current, dict) else None
        if t is None:
            raise RuntimeError(
                "residual_prediction is ON and no current token was given for %r "
                "(caller: %s)" % (mod_name, who))
        return t.reshape(t.shape[0], -1)

    def forward(self, last_step: torch.Tensor,
                current: Dict[str, torch.Tensor] = None,
                who: str = "unnamed caller") -> Dict[str, torch.Tensor]:
        """
        Args:
            last_step: [B, n_mod, d_model] from forward model.
            current: with residual_prediction on, {modality: [B, d_model]} raw encoder
                tokens for THIS moment. The head's output is added to them.
            who: names the call site in the error if `current` is missing.
        Returns:
            {modality_name: [B, d_model]} predictions for each head modality.
        """
        predictions = {}
        extra = self._all_slot_outputs(last_step)
        for mod_name, head in self.heads.items():
            out = head(last_step)
            if self.residual:
                out = out + self._residual_base(current, mod_name, who)
            if extra is not None:
                out = out + extra[mod_name].to(out.dtype)
            predictions[mod_name] = out
        return predictions

    def _all_slot_outputs(self, last_step):
        """--heads-all-slot-path on (lem/model/heads_all_slot_path.py): each head's all-slot path output, {head:
        [B, d_model]} float32, added to the head's prediction after its own output and the current token. None in
        every build without the path (the attribute is not there), and while the path is bypassed (its switch check)."""
        path = self.__dict__.get("all_slot_path")
        if path is None or path.bypass:
            return None
        return path(last_step)

    def forward_stacked(self, last_step: torch.Tensor,
                        current: Dict[str, torch.Tensor] = None,
                        who: str = "unnamed caller"
                        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Like forward(), but also returns a stacked [B, n_heads, d_model] tensor.

        Callers that need tensor-indexed access (loss_manager, imagination) use
        the stacked tensor; callers that need dict access (wellness, diagnostics)
        use the dict. Both share the same underlying computation -- no extra work.

        Returns:
            (stacked, predictions_dict):
                stacked: [B, n_heads, d_model] in _head_modalities order.
                predictions_dict: {mod_name: [B, d_model]} same as forward().
        """
        preds = []
        predictions = {}
        extra = self._all_slot_outputs(last_step)
        for mod_name in self._head_modalities:
            p = self.heads[mod_name](last_step)
            if self.residual:
                p = p + self._residual_base(current, mod_name, who)
            if extra is not None:
                p = p + extra[mod_name].to(p.dtype)
            preds.append(p)
            predictions[mod_name] = p
        stacked = torch.stack(preds, dim=1)  # [B, n_heads, d_model]
        return stacked, predictions
