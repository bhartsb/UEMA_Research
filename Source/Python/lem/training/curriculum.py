# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Developmental Curriculum Manager — 6-phase curriculum with handle restrictions,
mastery-based advancement, and per-phase imagination/context control.

UEMA Spec v3.4 §9.4: Developmental Motor Constraints
UEMA Spec v3.4 §15.1: Mastery Detection

When --curriculum is enabled, this replaces the static step-based phase lookup
from _get_developmental_phase() in harness.py.  When --curriculum is off,
harness._get_developmental_phase() continues to work unchanged.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

HANDLE_NAMES: list[str] = [
    "wrist_L", "thumb_L", "index_L", "middle_L", "ring_L", "pinky_L",
    "wrist_R", "thumb_R", "index_R", "middle_R", "ring_R", "pinky_R",
    "head", "upperarm_L", "lowerarm_L", "upperarm_R", "lowerarm_R",
    "thigh_L", "calf_L", "thigh_R", "calf_R", "foot_L", "foot_R",
    "gaze",   # the eyes: a real body part from 2026-08-24 (see lem/model/body_sets.py)
]
# POSITIONAL COUPLING (measured 2026-09-15). This list is 24 long and limits_for() builds
# handle_mask from it BY INDEX, but a body may carry more handles: the MIMo trunk_v1 layout
# has 26 (... gaze, lower_trunk, chest), and mimo_bridge/body_layout.py extend_limits() pads
# the mask to the body's length with True. That is harmless only while every phase activates
# every handle, which is today's table - the padded mask measured 26 entries, all True. If a
# phase ever masks handles again, these two lists have to be kept in index lockstep by hand,
# and the trunk handles would stay force-enabled whatever the phase said. active_handle_indices
# is NOT dead weight: lem/async_trainer.py and lem/training_process.py both read it at a phase
# change, and lem/tests/test_curriculum.py requires it in the dict. The lockstep is checked by
# name for a manager given the body's handle_names (train_mimo passes them, 2026-09-23): a body
# whose first 24 rows are not these parts in this order is refused.

ALL_HANDLE_INDICES: list[int] = list(range(len(HANDLE_NAMES)))

PHASE_ORDER: list[str] = [
    "neonate", "early_infant", "late_infant", "toddler", "child", "mature",
]


# ──────────────────────────────────────────────────────────────────────────────
# Phase definitions (spec v3.4 §9.4)
#
# All phases activate ALL handles (2026-08-24). The earlier wrists-then-fingers unlock
# schedule was removed: corticospinal wiring to individual fingers exists at birth, weak
# and noisy, not absent, so the ability should be PRESENT and unmastered rather than
# forbidden. Whole-hand behaviour early comes from the finger-coupling tendons in the
# body XML, weak distal strength, and babble - learned individuation, not permitted
# individuation. Unlocking handles mid-training also changes the action space under a
# trained forward model, the same defect as changing the physics late (cf. spec 9.5,
# "Bootstrap Without Scaffold"). rom/strength/noise limits stay: weak-but-present is
# anatomy; an absent handle is a mask.
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PhaseDefinition:
    """Immutable definition of a single curriculum phase."""
    name: str
    rom_limit: float
    strength_limit: float
    exploration_noise: float
    context_length: int
    imagination_enabled: bool
    imagination_steps: int
    active_handle_indices: tuple[int, ...]


PHASE_DEFINITIONS: dict[str, PhaseDefinition] = {
    "neonate": PhaseDefinition(
        name="neonate",
        rom_limit=0.05, strength_limit=0.02,
        exploration_noise=0.01, context_length=8,
        imagination_enabled=False, imagination_steps=0,
        active_handle_indices=tuple(ALL_HANDLE_INDICES),
    ),
    "early_infant": PhaseDefinition(
        name="early_infant",
        rom_limit=0.3, strength_limit=0.15,
        exploration_noise=0.008, context_length=8,
        imagination_enabled=False, imagination_steps=0,
        active_handle_indices=tuple(ALL_HANDLE_INDICES),
    ),
    "late_infant": PhaseDefinition(
        name="late_infant",
        rom_limit=0.5, strength_limit=0.3,
        exploration_noise=0.005, context_length=16,
        imagination_enabled=True, imagination_steps=3,
        active_handle_indices=tuple(ALL_HANDLE_INDICES),
    ),
    "toddler": PhaseDefinition(
        name="toddler",
        rom_limit=0.7, strength_limit=0.6,
        exploration_noise=0.003, context_length=24,
        imagination_enabled=True, imagination_steps=3,
        active_handle_indices=tuple(ALL_HANDLE_INDICES),
    ),
    "child": PhaseDefinition(
        name="child",
        rom_limit=1.0, strength_limit=0.9,
        exploration_noise=0.001, context_length=32,
        imagination_enabled=True, imagination_steps=3,
        active_handle_indices=tuple(ALL_HANDLE_INDICES),
    ),
    "mature": PhaseDefinition(
        name="mature",
        rom_limit=1.0, strength_limit=1.0,
        exploration_noise=0.0, context_length=32,
        imagination_enabled=True, imagination_steps=3,
        active_handle_indices=tuple(ALL_HANDLE_INDICES),
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Mastery tracker (spec v3.4 §15.1)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MasteryTracker:
    """Rolling window tracker for mastery-based early advancement.

    Criteria (all must hold simultaneously over the full window):
      - Mean PE < 0.05
      - PE variance < 0.01
      - 95% of steps have PE < 0.05 (consistency)
    """
    window_size: int = 50_000
    pe_threshold: float = 0.05
    variance_threshold: float = 0.01
    consistency_threshold: float = 0.95
    _pe_history: deque = field(default_factory=lambda: deque(maxlen=50_000))

    def __post_init__(self):
        self._pe_history = deque(maxlen=self.window_size)

    def record(self, mean_pe: float) -> None:
        self._pe_history.append(mean_pe)

    def is_mastered(self) -> bool:
        if len(self._pe_history) < self.window_size:
            return False
        vals = self._pe_history
        n = len(vals)
        s = sum(vals)
        mean = s / n
        if mean >= self.pe_threshold:
            return False
        variance = sum((v - mean) ** 2 for v in vals) / n
        if variance >= self.variance_threshold:
            return False
        under = sum(1 for v in vals if v < self.pe_threshold)
        if under / n < self.consistency_threshold:
            return False
        return True

    def stats(self) -> dict:
        """Return current mastery statistics for reporting."""
        if not self._pe_history:
            return {
                "pe_mean": None, "pe_variance": None,
                "consistency": None, "window_fill": 0.0,
            }
        vals = self._pe_history
        n = len(vals)
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / n
        under = sum(1 for v in vals if v < self.pe_threshold)
        return {
            "pe_mean": mean,
            "pe_variance": variance,
            "consistency": under / n,
            "window_fill": n / self.window_size,
        }

    def state_dict(self) -> dict:
        return {
            "pe_history": list(self._pe_history),
            "window_size": self.window_size,
        }

    def load_state_dict(self, state: dict) -> None:
        self.window_size = state.get("window_size", self.window_size)
        self._pe_history = deque(
            state.get("pe_history", []), maxlen=self.window_size)


# ──────────────────────────────────────────────────────────────────────────────
# Curriculum manager
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# ONE SCHEDULE (2026-08-27). Two copies of this table existed: the definitions above,
# which the trainer follows through the CurriculumManager, and a hand-written four-phase
# table in lem/config/defaults.py keyed by absolute step, which LemLite.dev_limits() and
# the causal test read. They disagreed on names, values and clock. Everything now derives
# from PHASE_DEFINITIONS through the functions below. The values themselves are a first
# draft to be revisited from data after the next run (operator decision).
#
# THE RAMP IS LIVE ON THE MIMo BRIDGE (corrected 2026-09-15; this comment used to say the
# body stayed infant-sized and the strength/range ramp was not relevant here, and both
# halves were wrong). With --age-binding curriculum, PHASE_AGE drives grow_scene() ->
# MIMo's adjust_mimo_to_age at growth model "v2" and the worker loads a regrown body,
# 1 month at neonate to 24 months at mature (mimo_bridge/env.py). rom_limit and
# strength_limit clamp EVERY executed order: reach = 40 cm x rom_limit, so one order may
# move a handle 2 cm at neonate and 40 cm at mature, and the strength channel is clamped
# to 0.02 of full push at neonate and 1.00 at mature (mimo_bridge/lem_model.py).
#
# BABBLE FLOOR. exploration_noise is never returned below BABBLE_FLOOR. Measured across
# seven runs: at the table's 0.0 (mature) the body froze and nothing could be learned
# about the action; at 0.05 it moved (0.51 rad, 27 cm/s). Movement is a wellness term,
# so movement must exist. Revisit when intention is on - then the intent to move to
# maintain wellness should carry it, and this floor can be questioned.
# ──────────────────────────────────────────────────────────────────────────────
BABBLE_FLOOR: float = 0.05


def limits_for(pd: "PhaseDefinition") -> dict:
    """The dev_limits dict for one phase definition, floor applied. The ONLY producer."""
    mask = [False] * len(HANDLE_NAMES)
    for idx in pd.active_handle_indices:
        mask[idx] = True
    return {
        "phase_name": pd.name,
        "rom_limit": pd.rom_limit,
        "strength_limit": pd.strength_limit,
        "exploration_noise": max(float(pd.exploration_noise), BABBLE_FLOOR),
        "context_length": pd.context_length,
        "active_handle_indices": list(pd.active_handle_indices),
        "imagination_enabled": pd.imagination_enabled,
        "imagination_steps": pd.imagination_steps,
        "handle_mask": mask,
    }


def phase_index_at_step(step: int, steps_per_phase: int, start_phase: str = "neonate") -> int:
    """Where a step-count clock puts the phase: start index + step // steps_per_phase,
    clamped to the first and last phase. This is the manager's own rule with mastery off.

    The lower clamp matters because callers may pass a step measured from a phase start
    (harness._get_developmental_phase subtracts training.phase_start_global): a query for
    a step BEFORE that start would otherwise floor-divide to a negative index, and
    PHASE_ORDER[-1] is 'mature' - the strongest phase - returned for the earliest step."""
    base = PHASE_ORDER.index(start_phase)
    if steps_per_phase is None or steps_per_phase <= 0:
        return len(PHASE_ORDER) - 1
    return min(len(PHASE_ORDER) - 1,
               max(0, base + int(step) // int(steps_per_phase)))


def limits_at_step(step: int, steps_per_phase: int, start_phase: str = "neonate") -> dict:
    """The limits a run with this steps_per_phase is under at `step` (mastery off)."""
    return limits_for(PHASE_DEFINITIONS[PHASE_ORDER[phase_index_at_step(step, steps_per_phase, start_phase)]])


def phases_table(steps_per_phase: int) -> dict:
    """The config-file form of the same schedule: {name: {start_step, end_step, ...}}."""
    out = {}
    for i, name in enumerate(PHASE_ORDER):
        lim = limits_for(PHASE_DEFINITIONS[name])
        out[name] = {
            "start_step": i * int(steps_per_phase),
            "end_step": None if i == len(PHASE_ORDER) - 1 else (i + 1) * int(steps_per_phase),
            "rom_limit": lim["rom_limit"],
            "strength_limit": lim["strength_limit"],
            "exploration_noise": lim["exploration_noise"],
            "context_length": lim["context_length"],
        }
    return out


class CurriculumManager:
    """Manages 6-phase developmental curriculum.

    Usage::

        cm = CurriculumManager(steps_per_phase=50000)
        # Each training step:
        dev_limits = cm.get_dev_limits()         # backward-compatible dict
        result = cm.step(mean_pe=current_pe)     # advance + check transition
        if result["phase_changed"]:
            # handle context resize, imagination toggle, etc.
    """

    def __init__(
        self,
        steps_per_phase: int = 50_000,
        mastery_advancement: bool = False,
        start_phase: str = "neonate",
        report_every: int = 10_000,
        mastery_window: int = 50_000,
        handle_names: Optional[Sequence[str]] = None,
    ):
        if start_phase not in PHASE_ORDER:
            raise ValueError(
                f"Unknown phase: {start_phase!r}. "
                f"Must be one of {PHASE_ORDER}")

        # The body this run drives, by name (fix-rows-r2, 2026-09-23). The handle counts in
        # the progress line, the phase-change line and transition_log used to count this
        # module's 24 HANDLE_NAMES, so every H84 segment (trunk_v1, 26 handles, all 26 on)
        # logged 'handles=24/24'. limits_for builds the phase mask over HANDLE_NAMES by
        # position and body_layout.extend_limits pads it with True for the rows past them,
        # so that mask is the right one only if the body's first rows ARE those parts in
        # that order: checked here by name, refused otherwise. Omitted = the 24 (lem/ runs).
        names = tuple(HANDLE_NAMES if handle_names is None else handle_names)
        if len(names) < len(HANDLE_NAMES):
            raise ValueError(
                f"the body has {len(names)} handles, fewer than the {len(HANDLE_NAMES)} "
                f"the curriculum's mask covers")
        for row, (want, got) in enumerate(zip(HANDLE_NAMES, names)):
            if want != got:
                raise ValueError(
                    f"body handle row {row} is {got!r}, but the curriculum's mask puts "
                    f"{want!r} there: the phase mask would switch the wrong part")
        self._handle_names: tuple = names

        self._steps_per_phase = steps_per_phase
        self._mastery_advancement = mastery_advancement
        self._report_every = report_every

        self._phase_idx: int = PHASE_ORDER.index(start_phase)
        self._phase_step: int = 0
        self._global_step: int = 0
        self._phase_start_global: int = 0

        self._mastery = MasteryTracker(window_size=mastery_window)

        # 10K progress report accumulators
        self._report_pe_accum: float = 0.0
        self._report_pe_count: int = 0

        logger.info(
            f"CurriculumManager: start_phase={start_phase}, "
            f"steps_per_phase={steps_per_phase}, "
            f"mastery_advancement={mastery_advancement}, "
            f"body handles={len(names)}{self._always_on_note()}")

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def phase_name(self) -> str:
        return PHASE_ORDER[self._phase_idx]

    @property
    def phase_def(self) -> PhaseDefinition:
        return PHASE_DEFINITIONS[self.phase_name]

    @property
    def phase_step(self) -> int:
        return self._phase_step

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def is_final_phase(self) -> bool:
        return self._phase_idx >= len(PHASE_ORDER) - 1

    @property
    def phase_progress(self) -> float:
        """Progress within current phase [0, 1]."""
        if self._steps_per_phase <= 0:
            return 1.0
        return min(1.0, self._phase_step / self._steps_per_phase)

    # ── Core API ────────────────────────────────────────────────────────

    def get_dev_limits(self) -> dict:
        """Return dev_limits dict compatible with _get_developmental_phase().

        Produced by limits_for() - the one producer - so the trainer, LemLite.dev_limits()
        and the causal test cannot disagree. Keys: phase_name, rom_limit, strength_limit,
        exploration_noise (floored), context_length, active_handle_indices,
        imagination_enabled, imagination_steps, handle_mask.
        """
        return limits_for(self.phase_def)
    def step(self, mean_pe: Optional[float] = None) -> dict:
        """Advance by one training step.

        Args:
            mean_pe: Mean prediction error for this step (for mastery tracking
                     and progress reports).

        Returns:
            dict with:
              phase_changed: bool
              progress_report: Optional[str] — emitted every report_every steps
              transition_log: Optional[dict] — emitted on phase change
        """
        self._phase_step += 1
        self._global_step += 1

        result: dict = {
            "phase_changed": False,
            "progress_report": None,
            "transition_log": None,
        }

        # Feed PE
        if mean_pe is not None:
            self._mastery.record(mean_pe)
            self._report_pe_accum += mean_pe
            self._report_pe_count += 1

        # 10K progress report
        if (self._report_every > 0
                and self._phase_step % self._report_every == 0):
            result["progress_report"] = self._make_progress_report()

        # Check phase transition
        if not self.is_final_phase:
            should_advance = False
            reason = ""

            if self._phase_step >= self._steps_per_phase:
                should_advance = True
                reason = "step_threshold"
            elif (self._mastery_advancement
                  and self._phase_step % 1000 == 0
                  and self._mastery.is_mastered()):
                should_advance = True
                reason = "mastery"

            if should_advance:
                result["transition_log"] = self._advance_phase(reason)
                result["phase_changed"] = True

        return result

    # ── Internal ────────────────────────────────────────────────────────

    def _active_body_handles(self, pd: PhaseDefinition) -> int:
        """How many of the body's handles phase `pd` leaves switched on, counted on the
        mask training uses: limits_for's mask over HANDLE_NAMES plus one True for every
        body row past them - the padding mimo_bridge.body_layout.extend_limits applies
        (no lem/ module depends on mimo_bridge, so the rule is repeated here)."""
        mask = limits_for(pd)["handle_mask"]
        return sum(mask) + (len(self._handle_names) - len(mask))

    def _always_on_note(self) -> str:
        """' (lower_trunk, chest always on)' when the body has handles past the
        curriculum's list - no phase can switch those off - and '' when it has none."""
        extra = self._handle_names[len(HANDLE_NAMES):]
        return f" ({', '.join(extra)} always on)" if extra else ""

    def _advance_phase(self, reason: str) -> dict:
        """Transition to the next phase."""
        old_name = self.phase_name
        old_def = self.phase_def
        mastery_stats = self._mastery.stats()

        next_name = PHASE_ORDER[self._phase_idx + 1]
        next_def = PHASE_DEFINITIONS[next_name]

        transition_log = {
            "timestamp": time.time(),
            "global_step": self._global_step,
            "from_phase": old_name,
            "to_phase": next_name,
            "reason": reason,
            "phase_steps": self._phase_step,
            "mastery_stats": mastery_stats,
            "old_context_length": old_def.context_length,
            "new_context_length": next_def.context_length,
            "old_handles": self._active_body_handles(old_def),
            "new_handles": self._active_body_handles(next_def),
            "imagination_change": (
                f"{old_def.imagination_enabled}->{next_def.imagination_enabled}"),
        }

        # Advance
        self._phase_idx += 1
        self._phase_step = 0
        self._phase_start_global = self._global_step

        # Reset mastery tracker — new phase has new PE distribution
        self._mastery = MasteryTracker(
            window_size=self._mastery.window_size)

        # Reset report accumulators
        self._report_pe_accum = 0.0
        self._report_pe_count = 0

        logger.info(
            f"=== PHASE TRANSITION: {old_name} -> {self.phase_name} "
            f"(reason={reason}, global_step={self._global_step}) ===")
        logger.info(
            f"  ROM: {old_def.rom_limit} -> {next_def.rom_limit} | "
            f"Strength: {old_def.strength_limit} -> {next_def.strength_limit} | "
            f"Context: {old_def.context_length} -> {next_def.context_length} | "
            f"Handles: {self._active_body_handles(old_def)} -> "
            f"{self._active_body_handles(next_def)} | "
            f"Imagination: {old_def.imagination_enabled} -> "
            f"{next_def.imagination_enabled}")

        return transition_log

    def _make_progress_report(self) -> str:
        """Build periodic progress report string."""
        avg_pe = (self._report_pe_accum / self._report_pe_count
                  if self._report_pe_count > 0 else None)
        pd = self.phase_def

        pe_str = f"avg_PE={avg_pe:.5f}" if avg_pe is not None else "avg_PE=N/A"
        report = (
            f"[Curriculum {self.phase_name} "
            f"step {self._phase_step}/{self._steps_per_phase} "
            f"({self.phase_progress:.0%})] {pe_str}")

        if self._mastery_advancement:
            ms = self._mastery.stats()
            if ms["pe_mean"] is not None:
                report += (
                    f" | mastery: mean={ms['pe_mean']:.5f}, "
                    f"var={ms['pe_variance']:.6f}, "
                    f"consistency={ms['consistency']:.2%}, "
                    f"window={ms['window_fill']:.0%}")
            else:
                report += " | mastery: window filling..."

        report += (
            f" | handles={self._active_body_handles(pd)}/{len(self._handle_names)}"
            f"{self._always_on_note()}, "
            f"ctx={pd.context_length}, "
            f"imag={'on' if pd.imagination_enabled else 'off'}")

        # Reset accumulators for next reporting window
        self._report_pe_accum = 0.0
        self._report_pe_count = 0

        logger.info(report)
        return report

    # ── Checkpoint save/load ────────────────────────────────────────────

    def state_dict(self) -> dict:
        """Serialize curriculum state for checkpoint."""
        return {
            "phase_idx": self._phase_idx,
            "phase_step": self._phase_step,
            "global_step": self._global_step,
            "phase_start_global": self._phase_start_global,
            "steps_per_phase": self._steps_per_phase,
            "mastery_advancement": self._mastery_advancement,
            "report_every": self._report_every,
            "mastery": self._mastery.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore curriculum state from checkpoint."""
        self._phase_idx = state["phase_idx"]
        self._phase_step = state["phase_step"]
        self._global_step = state["global_step"]
        self._phase_start_global = state.get("phase_start_global", 0)
        self._steps_per_phase = state.get(
            "steps_per_phase", self._steps_per_phase)
        self._mastery_advancement = state.get(
            "mastery_advancement", self._mastery_advancement)
        self._report_every = state.get(
            "report_every", self._report_every)
        if "mastery" in state:
            self._mastery.load_state_dict(state["mastery"])

        logger.info(
            f"Curriculum restored: phase={self.phase_name}, "
            f"phase_step={self._phase_step}, "
            f"global_step={self._global_step}")
