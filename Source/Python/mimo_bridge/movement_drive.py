# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Movement-hunger drive (US11537850B2 wellness modulation, adapted to the bridge).

The patent's mechanism: stimulation deprivation is not neutral - absent pleasure within a
time window, a pain effect activates, and pain ATTENUATES sensor resolution while pleasure
amplifies it. Here the stimulation is the body's own movement, measured body-locally
(per-bone velocity minus the body-mean velocity, so being carried or falling does not
count - only flexing does), gated by how self-caused the movement is (body-schema
coherence).

Three routed effects, each behind its own gate in train_mimo:

  Stage 1 (--movement-drive): a movement_state wellness sub-signal. Mild bounded bonus
      while moving; a stillness DEBT that grows the longer the body stays still and
      punishes superlinearly. It joins the goal-ring score, so remembered moments of
      movement feel better and the intention planner starts steering toward them.
  Stage 2 (--movement-sensor-mod): sensor-resolution modulation. Prolonged stillness
      attenuates tactile and proprioception gain (numbness, floored - no sensor death at
      this scale); movement restores and mildly amplifies. Numbness raises prediction
      error, which re-arms the curiosity gate: boredom -> fading senses -> hunger.
  Stage 3 (--movement-param-mod): parameter modulation. Stillness debt raises babble
      noise (restless fidgeting); movement mildly raises the learning-rate multiplier,
      stillness lowers it ("walking helps thinking"), both tightly bounded.

All state is per-env and reset with episodes. Everything is no_grad: the drive shapes
what the model FEELS and EXPERIENCES, never backpropagates through it.
"""
from __future__ import annotations

from typing import Optional

import torch


class MovementDrive:
    def __init__(self, n_envs: int, device, dt: float = 0.05,
                 m_ref_cm_s: float = 12.0,       # local speed that counts as "clearly moving"
                 still_thresh: float = 0.25,     # fraction of m_ref below which we call it still
                 debt_rise_s: float = 150.0,     # stillness time to reach full debt (~2.5 min)
                 debt_fall_s: float = 30.0,      # movement time to clear full debt
                 bonus: float = 0.35, penalty: float = 0.5,
                 gain_floor: float = 0.3, gain_ceil: float = 1.3):
        self.device = device
        self.dt = dt
        self.m_ref = m_ref_cm_s
        self.still_thresh = still_thresh
        self.rise = dt / debt_rise_s
        self.fall = dt / debt_fall_s
        self.bonus = bonus
        self.penalty = penalty
        self.gain_floor = gain_floor
        self.gain_ceil = gain_ceil
        from tokenizer.config import TokenizerConfig
        self.vel_range = float(TokenizerConfig().velocity_range_cms)   # tokenizer norm -> cm/s
        self.debt = torch.zeros(n_envs, device=device)     # [B] in [0, 1]
        self._prev_v: torch.Tensor | None = None           # [B, 23, 3] for smoothness
        self._smooth = torch.ones(n_envs, device=device)   # EMA of direction persistence
        self.last = {}

    def reset(self, env_idx: Optional[torch.Tensor] = None):
        if env_idx is None:
            self.debt.zero_()
            self._prev_v = None
            self._smooth = torch.ones_like(self._smooth)
        else:
            self.debt[env_idx] = 0.0

    @torch.no_grad()
    def update(self, proprio: torch.Tensor, self_factor: float = 1.0) -> dict:
        """proprio: [B, 23, 12] tokenized bones; fields 7:10 are world velocity in the
        tokenizer's normalized units (cm/s / velocity_range_cms) - converted back to cm/s."""
        B = proprio.shape[0]
        if self.debt.shape[0] != B:
            self.debt = torch.zeros(B, device=proprio.device)
        v = proprio[:, :, 7:10].float() * self.vel_range   # [B, 23, 3] world vel, cm/s
        local = v - v.mean(dim=1, keepdim=True)            # body-local: flexing, not travel
        speed = local.norm(dim=-1).mean(dim=1)             # [B] mean local speed, cm/s
        m = (speed / self.m_ref).clamp(0.0, 3.0)
        # smoothness: do the limbs keep going the way they were going? Twitching reverses
        # direction constantly (cos ~ -1..0); a sustained stroke holds it (cos ~ 1). The
        # bonus is gated by this, so vibrating cannot score as movement (video finding
        # 2026-08-21: bodies jittered - raw speed was rewarded regardless of coherence).
        if self._prev_v is not None and self._prev_v.shape == v.shape:
            num = (v * self._prev_v).sum(-1)
            den = v.norm(dim=-1) * self._prev_v.norm(dim=-1) + 1e-6
            step_cos = (num / den).mean(dim=1)                       # [B] in [-1, 1]
            self._smooth = 0.9 * self._smooth + 0.1 * step_cos.clamp(-1, 1)
        self._prev_v = v.clone()
        smooth01 = ((self._smooth + 1.0) / 2.0).clamp(0.0, 1.0)      # [B] in [0, 1]
        sf = float(min(max(self_factor, 0.0), 1.0))
        moving = m > self.still_thresh
        self.debt = torch.where(moving,
                                (self.debt - self.fall * m).clamp(min=0.0),
                                (self.debt + self.rise).clamp(max=1.0))
        wellness = (0.5 + self.bonus * torch.tanh(m) * sf * smooth01
                    - self.penalty * self.debt.pow(2)).clamp(0.0, 1.0)      # [B]
        gain = (1.0 + 0.25 * torch.tanh(m) - 0.7 * self.debt.pow(2)).clamp(self.gain_floor, self.gain_ceil)
        noise_mult = float((1.0 + 1.5 * self.debt).mean())                  # restlessness
        lr_mult = float((1.0 + 0.10 * torch.tanh(m) - 0.15 * self.debt.pow(2)).mean().clamp(0.85, 1.10))
        self.last = {"local_speed": float(speed.mean()), "m_norm": float(m.mean()),
                     "smoothness": float(smooth01.mean()),
                     "stillness_debt": float(self.debt.mean()),
                     "movement_wellness": float(wellness.mean()),
                     "sensor_gain": float(gain.mean()),
                     "noise_mult": noise_mult, "lr_mult": lr_mult, "self_factor": sf}
        return {"wellness": wellness, "gain": gain, "noise_mult": noise_mult,
                "lr_mult": lr_mult, "m": m, "debt": self.debt.clone()}
