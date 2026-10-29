# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Training helpers — developmental phases and precision mapping.

Used by live_test.py and live_test_lite.py.
"""

from __future__ import annotations

from typing import Optional

import torch


def _get_developmental_phase(config: dict, step: int) -> dict:
    """Developmental limits in force at training STEP `step` (a step, not a phase index).

    One schedule: lem.training.curriculum. With training.steps_per_phase in the config
    (every generated default carries it; the MIMo trainer overwrites it with the run's
    flag) the answer is limits_at_step(), which is exactly what the CurriculumManager
    returns at that step with mastery advancement off. Configs without it fall back to
    their own developmental_phases table, read generically by start_step.

    training.start_phase names the phase the run is in at training.phase_start_global,
    and the step count is measured from that global step. A CurriculumManager counts
    inside its phase (phase_step), so without the offset a resumed run's absolute step
    is divided by steps_per_phase from zero and lands in a later phase than the manager
    is in - measured on a checkpoint saved at 'child' resumed at global step 3,500,000
    with steps_per_phase 3000: manager 'child' (strength limit 0.90) against this
    function's 'mature' (strength limit 1.00), the same disagreement whether or not
    --start-phase matched. phase_start_global defaults to 0, which is what a run that
    started at step 0 in its start_phase has, so configs written before this behave
    exactly as they did.
    """
    tr = config.get("training", {})
    spp = tr.get("steps_per_phase")
    if spp:
        from lem.training.curriculum import limits_at_step
        lim = limits_at_step(int(step) - int(tr.get("phase_start_global", 0) or 0),
                             int(spp), str(tr.get("start_phase", "neonate")))
        lim = dict(lim)
        lim.setdefault("context_length", config["model"]["context_length"])
        return lim
    phases = tr["developmental_phases"]
    default_ctx = config["model"]["context_length"]
    ordered = sorted(phases.items(), key=lambda kv: -int(kv[1].get("start_step", 0)))
    for phase_name, phase in ordered:
        if step >= int(phase.get("start_step", 0)):
            end = phase.get("end_step")
            if end is None or step < end:
                return {
                    "phase_name": phase_name,
                    "rom_limit": phase["rom_limit"],
                    "strength_limit": phase["strength_limit"],
                    "exploration_noise": phase.get("exploration_noise", 0.0),
                    "context_length": phase.get("context_length", default_ctx),
                }
    first_name, first = ordered[-1]
    return {
        "phase_name": first_name,
        "rom_limit": first["rom_limit"],
        "strength_limit": first["strength_limit"],
        "exploration_noise": first.get("exploration_noise", 0.0),
        "context_length": first.get("context_length", default_ctx),
    }

def _get_precision_dtype(precision: str) -> torch.dtype:
    """Map config precision string to torch dtype."""
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }.get(precision, torch.float16)
