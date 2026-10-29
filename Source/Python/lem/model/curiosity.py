# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Curiosity system — UEMA Spec §21.

Modulates wellness gradient based on prediction error magnitude.
Drive output in [0, 1]: peaks in goldilocks zone (learnable PE range),
zero when overwhelmed (noise) or bored (mastered).
Curiosity amplifies the wellness signal — it never inverts or suppresses it.
Safety (withdrawal from pain, joint limits) is handled unconditionally
by the wellness evaluators themselves.
"""


class CuriositySystem:
    """PE-driven curiosity: a triangle over one scalar prediction error.

    Two consumers, each behind its own flag. Under --curiosity-goal the felt
    value stored with a goal-ring moment is multiplied by (1 + drive), which is
    the only live effect when --wellness-mode is baseline. Under --wellness-mode
    direct or forward the drive also scales the wellness loss.
    """

    def __init__(self, pe_min: float = 0.1, pe_max: float = 0.4):
        if pe_min <= 0 or pe_max <= pe_min:
            raise ValueError(f"Need 0 < pe_min < pe_max, got ({pe_min}, {pe_max})")
        self.pe_min = pe_min
        self.pe_max = pe_max
        self._mid = (pe_min + pe_max) / 2

    def compute_drive(self, pe: float) -> float:
        """Compute curiosity drive from scalar prediction error.

        Returns a value in [0.0, 1.0]:
          - Bored   (pe <= pe_min): 0 — nothing to learn, mastery plateau
          - Goldilocks (pe_min < pe < pe_max): positive, peaks at 1.0 at midpoint
          - Overwhelmed (pe >= pe_max): 0 — pure noise, can't extract signal

        Curiosity is an amplifier on the wellness gradient, not a safety gate.
        The wellness evaluators (tactile_comfort, motor_state) handle safety
        unconditionally from frame zero.
        """
        if pe <= self.pe_min:
            raw = pe / self.pe_min - 1.0
        elif pe >= self.pe_max:
            raw = 1.0 - pe / self.pe_max
        else:
            # Half the band width, not the full width: the midpoint is half a band
            # from each edge, so dividing by the full width leaves the tent at 0.5
            # where the outer branches are already 0 - a 0.5 step at both edges.
            raw = 1.0 - abs(pe - self._mid) / ((self.pe_max - self.pe_min) / 2.0)
        return max(0.0, min(1.0, raw))

    def zone(self, pe: float) -> str:
        """Return the curiosity zone name for logging."""
        if pe <= self.pe_min:
            return "bored"
        elif pe >= self.pe_max:
            return "overwhelmed"
        return "goldilocks"
