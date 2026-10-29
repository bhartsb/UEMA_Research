# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""What a curriculum phase change does, and what the record says it did.

Four separate things were invisible or wrong at a phase boundary:
  * the type "phase_transition" record was written BEFORE the rebuild decision was
    computed, so it read the same whether the bodies were rebuilt and the context
    buffer, pending transitions, goal ring and replay episode were dropped, or nothing
    was touched at all;
  * model.config carried the LAUNCH FLAG and an absolute-step clock, so
    LemLite.dev_limits(step) - what run_mode_a and run_mode_b read every step - could
    return different rom/strength limits from the CurriculumManager the trainer reads;
  * the trainer marked the replay episode boundary itself AND through
    reset_transition_state, advancing the episode id twice for one boundary;
  * one model step's length in the body's own time was a hardcoded 0.01 s per env step
    in two places, while the authority (env.dt) was exported nowhere.
"""
import inspect
import json

import numpy as np
import pytest

from mimo_bridge import train_mimo as TM
from mimo_bridge.lem_model import LemLite
from lem.training.curriculum import CurriculumManager, PHASE_ORDER
from lem.training.harness import _get_developmental_phase

SPP = 3000     # every H8x run's --steps-per-phase


def _args(*extra):
    return TM.parse_args(["--out", "unused", "--steps-per-phase", str(SPP), *extra])


class _ConfigOnlyModel:
    """Just the config dict: bind_schedule_to_config touches nothing else."""
    def __init__(self):
        self.config = {"model": {"context_length": 32}, "training": {}}


# ───────────────────────────────────────────── what a phase change actually changes
def test_the_runs_own_flags_rebuild_the_bodies_at_every_transition():
    """H80 through H83 all ran --age-binding curriculum --delay-binding on
    --scene-binding phase. Under those flags the MIMo age changes at all five
    transitions and the delays change at all five, so the drop DOES happen."""
    args = _args("--age-binding", "curriculum", "--delay-binding", "on",
                 "--scene-binding", "phase")
    kw = TM.env_kwargs_for(args, PHASE_ORDER[0], "out")
    for i in range(1, len(PHASE_ORDER)):
        changes, new_kw = TM.rebuild_changes(args, PHASE_ORDER[i], "out", kw)
        assert changes, f"{PHASE_ORDER[i-1]} -> {PHASE_ORDER[i]} reported no rebuild"
        assert "age" in changes
        kw.update(new_kw)
    assert kw["age"] == TM.PHASE_AGE["mature"]


def test_with_every_binding_off_nothing_is_rebuilt_and_nothing_is_dropped():
    """The configuration in which the record used to be silently wrong: the limits still
    go from rom 0.05 to rom 1.0 while the context buffer, the pending transitions and
    the goal ring all survive, because no env kwarg differs."""
    args = _args("--age-binding", "none", "--delay-binding", "off",
                 "--scene-binding", "fixed")
    kw = TM.env_kwargs_for(args, PHASE_ORDER[0], "out")
    for i in range(1, len(PHASE_ORDER)):
        changes, _ = TM.rebuild_changes(args, PHASE_ORDER[i], "out", kw)
        assert changes == {}
    first = CurriculumManager(steps_per_phase=SPP, start_phase="neonate",
                              report_every=10 ** 9).get_dev_limits()
    last = CurriculumManager(steps_per_phase=SPP, start_phase="mature",
                             report_every=10 ** 9).get_dev_limits()
    assert first["rom_limit"] == 0.05 and last["rom_limit"] == 1.0


def test_the_record_says_whether_the_bodies_were_rebuilt():
    rec = TM.phase_transition_record({"from_phase": "neonate", "to_phase": "early_infant"},
                                     step=3000, changes={}, timestamp=1.0)
    assert rec["type"] == "phase_transition" and rec["rebuilt"] is False
    assert rec["changes"] == {} and rec["step"] == 3000
    assert rec["from_phase"] == "neonate"          # the manager's own fields survive
    rec2 = TM.phase_transition_record(None, step=6000,
                                      changes={"age": 8.0, "sensory_delay": 2},
                                      timestamp=2.0)
    assert rec2["rebuilt"] is True and rec2["changes"]["age"] == 8.0
    json.dumps(rec2)                                # the stream takes it as written


def test_the_trainer_writes_the_record_after_deciding_and_logs_both_outcomes():
    src = inspect.getsource(TM.main)
    write = src.index("phase_transition_record(")
    decide = src.index("changes, new_kw = rebuild_changes(")
    assert decide < write, "the rebuild decision must be taken before the record"
    assert "bodies rebuilt: %s" in src


# ─────────────────────────────────────────── one schedule, two readers, one answer
@pytest.mark.parametrize("saved,phase_step,flag,global_step",
                         [("child", 1200, "neonate", 3_500_000),
                          ("late_infant", 400, "neonate", 900_000),
                          ("late_infant", 400, "late_infant", 9_000),
                          ("mature", 0, "neonate", 4_083_925)])
def test_the_trainers_config_write_makes_dev_limits_answer_what_the_manager_answers(
        saved, phase_step, flag, global_step):
    cm = CurriculumManager(steps_per_phase=SPP, mastery_advancement=False,
                           start_phase=saved, report_every=10 ** 9)
    cm._phase_step = phase_step
    cm._global_step = global_step
    model = _ConfigOnlyModel()
    TM.bind_schedule_to_config(model, cm, SPP, global_step)
    assert model.config["training"]["start_phase"] == cm.phase_name
    assert model.config["training"]["phase_start_global"] == global_step - phase_step
    theirs = _get_developmental_phase(model.config, global_step)
    ours = cm.get_dev_limits()
    assert theirs["phase_name"] == ours["phase_name"]
    assert theirs["rom_limit"] == ours["rom_limit"]
    assert theirs["strength_limit"] == ours["strength_limit"]


def test_the_trainer_rebinds_the_schedule_after_a_phase_roll():
    """An overseer 'advance_phase' rolls the manager early; without a rebind the two
    readers part company again at the next step."""
    src = inspect.getsource(TM.main)
    assert src.count("bind_schedule_to_config(") == 2


# ─────────────────────────────────── one boundary, one replay episode-id increment
def test_reset_transition_state_is_the_single_owner_of_the_replay_boundary():
    m = LemLite(device="cpu", seed=0, imagination=False,
                overrides={"training": {"steps_per_phase": SPP}})
    m.enable_replay(2, size_per_env=32)
    before = m.replay._episode_id
    m.reset_transition_state()
    assert m.replay._episode_id == before + 1


def test_the_trainer_does_not_mark_the_replay_boundary_itself():
    src = inspect.getsource(TM.main)
    assert "replay.mark_reset()" not in src, (
        "reset_transition_state already marks it; a second call advances the episode "
        "id twice for one boundary")
    # all three teleport sites still reset the model's carried state
    assert src.count("model.reset_transition_state()") == 3


# ──────────────────────────────────────────── one moment in the body's own time
def test_one_model_step_is_fifty_milliseconds_exactly():
    """0.005 s MuJoCo timestep x frame_skip 2 x --action-repeat 5. Each product is exact
    in binary floating point, so this is 0.050000 s, not an approximation."""
    assert TM.ENV_DT_ASSUMED_S == 0.01
    args = _args("--action-repeat", "5")
    assert TM._transition_seconds(args) == 0.05
    assert TM._transition_seconds(args, 0.005 * 2) == 0.05
    assert TM._transition_seconds(args, 0.01) == TM._transition_seconds(args)
    # a different scene timestep moves it, which is the whole point of measuring
    assert TM._transition_seconds(args, 0.02) == 0.1
    assert TM._transition_seconds(_args("--action-repeat", "1"), 0.01) == 0.01


def test_the_measured_env_step_is_exported_and_used():
    src = inspect.getsource(TM.main)
    assert 'env_dt = float(frames[0].dt)' in src
    assert 'model.receipt["env_dt_s"] = env_dt' in src
    assert "_tsec = _transition_seconds(args, env_dt)" in src
    assert "* 0.01" not in src, "no hardcoded env-step length may remain in the loop"


# ────────────────────────── the curiosity drive's steps and the causal error's steps
def test_the_fallback_steps_are_the_ones_the_causal_error_mean_drops():
    class _R:
        def __init__(self, cp):
            self.causal_pe = cp
    assert TM.curiosity_used_fallback(_R(float("nan"))) is True
    assert TM.curiosity_used_fallback(_R(None)) is True
    assert TM.curiosity_used_fallback(_R(0.0)) is False
    assert TM.curiosity_used_fallback(_R(0.37)) is False
    # the trainer's own causal_pe accumulator uses the same finiteness test, so the
    # counter and the divisor it explains cannot drift apart
    src = inspect.getsource(TM.main)
    assert "_cp is not None and _cp == _cp" in src
    assert '"curiosity_drive_fallback_steps"' in src


# ─────────────────────── which order the body got, on a scheduled rest window
def test_the_executed_order_is_offered_to_the_explainer_when_it_can_take_it():
    def new_style(model, frames, res, *, run, step, held_order=None,
                  order_sent=None, rest_window=False):
        pass

    def old_style(model, frames, res, *, run, step, held_order=None):
        pass

    sent = np.zeros((2, 26, 9), dtype=np.float32)
    fields = TM.explain_rest_fields(new_style, sent, True)
    assert fields["rest_window"] is True and fields["order_sent"] is sent
    assert TM.explain_rest_fields(old_style, sent, True) == {}
    assert TM.explain_rest_fields(None, sent, True) == {}
    off = TM.explain_rest_fields(new_style, sent, False)
    assert off["rest_window"] is False


def test_the_trainer_offers_the_order_the_body_was_actually_stepped_with():
    src = inspect.getsource(TM.main)
    assert "explain_rest_fields(build_snapshot, send_motor, scheduled_hold)" in src
    # send_motor is the array src.step receives; held_order stays the proposal
    assert "frames = src.step(send_motor," in src
    assert "held_order=held_motor" in src
