# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The two readers of the developmental schedule must return the same limits.

The CurriculumManager counts steps INSIDE its phase; harness._get_developmental_phase
divides a step count by steps_per_phase. Without training.phase_start_global naming the
global step the current phase began at, a resumed run's absolute step put the fallback
in a later phase than the manager was in - and the fallback is what
LemLite.dev_limits(step) answers, which run_mode_a and run_mode_b read every step.
"""
import pytest

from lem.training.curriculum import (CurriculumManager, PHASE_ORDER,
                                     limits_at_step, phase_index_at_step)
from lem.training.harness import _get_developmental_phase

SPP = 3000     # every H8x run's --steps-per-phase


def _config(start_phase, phase_start_global=None):
    training = {"steps_per_phase": SPP, "start_phase": start_phase}
    if phase_start_global is not None:
        training["phase_start_global"] = int(phase_start_global)
    return {"model": {"context_length": 32}, "training": training}


def _manager(saved_phase, saved_phase_step, global_step):
    cm = CurriculumManager(steps_per_phase=SPP, mastery_advancement=False,
                           start_phase=saved_phase, report_every=10 ** 9)
    cm._phase_step = int(saved_phase_step)
    cm._global_step = int(global_step)
    return cm


# saved phase, phase_step, the --start-phase flag, the global step of the resume
RESUMES = [("child", 1200, "neonate", 3_500_000),
           ("late_infant", 400, "neonate", 900_000),
           ("mature", 0, "neonate", 4_083_925),
           ("late_infant", 400, "late_infant", 9_000),
           ("child", 1200, "child", 3_500_000)]


@pytest.mark.parametrize("saved,phase_step,flag,global_step", RESUMES)
def test_anchored_config_makes_both_readers_return_one_answer(
        saved, phase_step, flag, global_step):
    cm = _manager(saved, phase_step, global_step)
    cfg = _config(cm.phase_name, global_step - cm.phase_step)
    theirs = _get_developmental_phase(cfg, global_step)
    ours = cm.get_dev_limits()
    assert theirs["phase_name"] == ours["phase_name"]
    assert theirs["rom_limit"] == ours["rom_limit"]
    assert theirs["strength_limit"] == ours["strength_limit"]


@pytest.mark.parametrize("saved,phase_step,flag,global_step", RESUMES)
def test_the_launch_flag_and_the_absolute_step_do_not_make_them_agree(
        saved, phase_step, flag, global_step):
    """The shape of the defect: writing the FLAG and clocking on the ABSOLUTE step.

    The one case that agrees does so only because a 4-million-step absolute clock at
    steps_per_phase 3000 clamps to the last phase, which is also where the manager sits.
    """
    cm = _manager(saved, phase_step, global_step)
    theirs = _get_developmental_phase(_config(flag), global_step)
    ours = cm.get_dev_limits()
    agree = (theirs["phase_name"] == ours["phase_name"]
             and theirs["strength_limit"] == ours["strength_limit"])
    assert agree == (ours["phase_name"] == PHASE_ORDER[-1])


def test_the_fallback_tracks_the_manager_across_a_later_transition():
    """The anchor is written once at launch and the clock carries it forward: the
    manager rolls a phase every steps_per_phase, and so does the offset step count."""
    cm = _manager("late_infant", 0, 1_000_000)
    cfg = _config(cm.phase_name, 1_000_000)
    for k in range(1, len(PHASE_ORDER) - PHASE_ORDER.index("late_infant")):
        step = 1_000_000 + k * SPP
        while cm.global_step < step:
            cm.step(mean_pe=0.5)
        assert _get_developmental_phase(cfg, step)["phase_name"] == cm.phase_name


def test_a_step_before_the_phase_start_clamps_to_the_first_phase():
    """A step measured from the phase start can be negative, and a negative index into
    a 6-entry PHASE_ORDER either counts from the END - PHASE_ORDER[-1] is 'mature', the
    STRONGEST phase, returned for the earliest step - or raises IndexError once it goes
    past -6. LemLite.dev_limits(0) takes exactly that path when the config is anchored
    at a mid-lineage step, so the index is clamped at both ends."""
    assert phase_index_at_step(-1, SPP, "neonate") == 0
    assert phase_index_at_step(-SPP * 10, SPP, "child") == 0
    assert limits_at_step(-SPP * 10, SPP, "child")["phase_name"] == PHASE_ORDER[0]
    assert limits_at_step(-1, SPP, "late_infant")["phase_name"] == "early_infant"
    # 3,500,000 steps before the phase start is index 4 - 1167: an IndexError before
    # the clamp, the weakest phase after it
    assert _get_developmental_phase(_config("child", 3_500_000), 0)["phase_name"] == "neonate"


def test_configs_without_the_anchor_are_unchanged():
    """phase_start_global absent means 0, which is what a run that started at step 0 in
    its start_phase has - every config written before the anchor existed."""
    for step in (0, 3, 15_000, 897_695, 10 ** 9):
        assert (_get_developmental_phase(_config("neonate"), step)
                == _get_developmental_phase(_config("neonate", 0), step))
