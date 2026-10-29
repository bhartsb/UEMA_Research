# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The swap test's second phase, its size-matched and shifted-own arms, and the side heads' size-matched control
(2026-10-02, docs/gain_of_function_v4/agents/swap_test_phases_2026-10-02.md).

CPU only, small fixtures, no MuJoCo world and no checkpoint. Each check that guards a reading also runs against a
planted defect of that reading (phase off by one, the absolute command in place of the size-matched one, the noise
not added, the added passes watched by the liveness repair) and must catch it."""
import contextlib
import math

import numpy as np
import pytest
import torch

from mimo_bridge import prediction_audit as PA
from mimo_bridge.prediction_audit import (accumulate, accumulate_rows, audit_blocks, command_spread_std,
                                          command_spread_update, paired_errors, paired_rows, shifted_own_order,
                                          summarize, summarize_hold_first, summarize_rows)


# ------------------------------------------------------------------------------ the phase rule
def latch_steps(hold, n, first=0):
    """The trainer's latch rule (train_mimo.py: `if step % hold == 0 or held_motor is None`) from model step
    `first` (a start or a resume) to n: the model steps on which a new command is latched."""
    held = None
    out = set()
    for step in range(first, n):
        if step % hold == 0 or held is None:
            out.add(step)
            held = step
    return out


def phase_violations(blocks_fn, hold, every=20, n=1200, first=0):
    """What is wrong with a phase rule under the trainer's latch rule: every 'hold_first' model step s must grade
    the first model step of a hold (s - 1 latched a command), one per `every` model steps when hold <= every (the
    first latch at or after each multiple of `every`), every latch when hold > every; 'every_20' exactly on the
    multiples of `every`."""
    bad = []
    latches = latch_steps(hold, n, first)
    first_steps = [s for s in range(first + 1, n) if "hold_first" in blocks_fn(s, hold)]
    for s in first_steps:
        if (s - 1) not in latches:
            bad.append("model step %d grades a transition that is not a hold's first (no latch at %d)" % (s, s - 1))
    if hold <= every:
        for k in range((first + every - 1) // every, (n - hold - 1) // every):
            want = min(L for L in latches if L >= k * every and L % hold == 0) + 1
            got = [s for s in first_steps if k * every <= s - 1 < (k + 1) * every]
            if got != [want]:
                bad.append("period %d: graded %s, the hold's first model step is %d" % (k, got, want))
    else:
        regular = sorted(L + 1 for L in latches if L % hold == 0 and L + 1 < n)
        if first_steps != regular:
            bad.append("with a hold longer than %d every latch is graded once: %s against %s"
                       % (every, first_steps[:5], regular[:5]))
    for s in range(first, n):
        if ("every_20" in blocks_fn(s, hold)) != (s % every == 0):
            bad.append("model step %d: the existing block's rule changed" % s)
            break
    return bad


@pytest.mark.parametrize("hold", [1, 2, 3, 4, 5, 7, 10, 20, 25, 40])
def test_the_hold_first_phase_follows_the_trainers_latch_rule(hold):
    assert phase_violations(audit_blocks, hold) == []
    # a resume or a reset latches on its first model step whatever its number: the rule still names only
    # hold-first transitions
    assert phase_violations(audit_blocks, hold, first=37) == []


def test_with_motor_hold_5_the_phase_is_model_steps_one_after_a_multiple_of_20():
    firsts = [s for s in range(0, 200) if "hold_first" in audit_blocks(s, 5)]
    assert firsts == list(range(1, 200, 20))
    assert [s for s in range(0, 200) if "every_20" in audit_blocks(s, 5)] == list(range(0, 200, 20))
    # --intention-every never enters: the plan reaches the body only at a latch
    assert all(audit_blocks(s, 5) == audit_blocks(s, 5, 20) for s in range(200))
    # another hold: 3 (does not divide 20): the first latch at or after 20k, plus one
    assert [s for s in range(0, 70) if "hold_first" in audit_blocks(s, 3)] == [1, 22, 43, 61]


def test_the_phase_check_catches_a_phase_off_by_one():
    def off_by_one(step, hold, every=20):            # planted defect: one model step late
        out = list(audit_blocks(step, hold, every))
        if "hold_first" in out:
            out.remove("hold_first")
        if step >= 2 and (step - 2) % hold == 0 and (step - 2) % every < hold:
            out.append("hold_first")
        return tuple(out)

    def latch_step_itself(step, hold, every=20):     # planted defect: grading the latch step, the last of a hold
        out = [b for b in audit_blocks(step, hold, every) if b != "hold_first"]
        if step % hold == 0 and step % every < hold:
            out.append("hold_first")
        return tuple(out)
    for planted in (off_by_one, latch_step_itself):
        for hold in (3, 5):
            assert phase_violations(planted, hold), planted.__name__


# ------------------------------------------------------------------------------ the readings' arithmetic
def test_each_arm_reading_is_one_minus_the_summed_errors_ratio_and_the_old_keys_do_not_move():
    current = torch.zeros(2, 4)
    actual = torch.ones(2, 4)
    own = torch.full((2, 4), 0.9)           # own error 0.01 per number
    absolute = torch.full((2, 4), 0.5)      # 0.25
    sized = torch.full((2, 4), 0.8)         # 0.04
    shifted = torch.full((2, 4), 0.95)      # 0.0025: the shifted copy predicts better
    old = paired_errors(own, current, actual, absolute)
    new = paired_errors(own, current, actual, absolute, size_matched=sized, shifted_own=shifted)
    assert {k: v for k, v in new.items() if k in old} == old
    assert abs(new["size_matched_action_mse"] - 0.04) < 1e-7 and abs(new["shifted_own_action_mse"] - 0.0025) < 1e-7
    assert abs(new["size_matched_shift_mse"] - 0.01) < 1e-7       # ||own - sized||^2 = 0.1^2
    sums = {}
    accumulate(sums, {"s": new})
    r = summarize(sums)["modalities"]["s"]
    assert abs(r["improvement_over_shuffled"] - (1 - 0.01 / 0.25)) < 1e-6
    assert abs(r["improvement_over_size_matched"] - (1 - 0.01 / 0.04)) < 1e-6
    assert abs(r["improvement_over_shifted_own"] - (1 - 0.01 / 0.0025)) < 1e-6     # below 0
    assert abs(r["sensitivity_size_matched"] - math.sqrt(0.01 / 0.81)) < 1e-6      # change 0.9^2
    # a window sums the errors before the one ratio: 0.04 and 0.36 against own 0.01 and 0.01
    sums = {}
    accumulate(sums, {"s": paired_errors(own, current, actual, absolute, size_matched=sized)})
    accumulate(sums, {"s": paired_errors(own, current, actual, absolute, size_matched=torch.full((2, 4), 0.4))})
    r = summarize(sums)["modalities"]["s"]
    assert abs(r["improvement_over_size_matched"] - (1 - 0.02 / 0.40)) < 1e-6
    assert r["size_matched_transitions"] == 4 and "shifted_own_transitions" not in r
    # an arm with a non-finite value is left out, never read as an error
    bad = paired_errors(own, current, actual, absolute, size_matched=torch.full((2, 4), float("nan")))
    assert "size_matched_action_mse" not in bad
    # the block carries the definitions
    s = summarize(sums)
    assert "absolute" in s["arms"]["shuffled"].lower() and "one segment" in s["arms"]["shuffled"]
    assert "shuffled_order" in s["arms"]["size_matched"] and "multiple of 20" in s["sampled_at"]


def test_per_body_rows_pool_with_and_without_body_2_and_the_hold_first_block_says_its_phase():
    B = 3
    current = torch.zeros(B, 2)
    actual = torch.ones(B, 2)
    own = torch.tensor([[0.9, 0.9], [0.5, 0.5], [0.9, 0.9]])
    sized = torch.tensor([[0.8, 0.8], [0.0, 0.0], [0.7, 0.7]])
    rows = paired_rows(own, current, actual, None, size_matched=sized)
    assert torch.allclose(rows["size_matched_action_mse"], torch.tensor([0.04, 1.0, 0.09]))
    sums = {}
    accumulate_rows(sums, {"s": rows})
    out = summarize_rows(sums, excluded=(1,))
    pooled = out["senses"]["s"]["all_bodies"]
    assert abs(pooled["improvement_over_size_matched"] - (1 - (0.01 + 0.25 + 0.01) / (0.04 + 1.0 + 0.09))) < 1e-6
    without = out["senses"]["s"]["without_body_1"]
    assert abs(without["improvement_over_size_matched"] - (1 - 0.02 / 0.13)) < 1e-6
    assert out["senses"]["s"]["per_body"]["size_matched_transitions"] == [1, 1, 1]
    hf = summarize_hold_first(sums)
    assert "first model step of a hold" in hf["sampled_at"] and "size_matched" in hf["arms"]
    assert "multiple of 20" in summarize_rows(sums)["sampled_at"]


# ------------------------------------------------------------------------------ the size-matched command
GAZE = 23


def _commands(B=4, H=26, seed=0):
    g = torch.Generator().manual_seed(seed)
    cmd = torch.randn(B, H, 9, generator=g) * 20.0
    q = torch.randn(B, H, 4, generator=g)
    cmd[..., 3:7] = q / q.norm(dim=-1, keepdim=True)
    cmd[..., 7] = 0.0
    cmd[..., 8] = torch.rand(B, H, generator=g)
    from mimo_bridge.lem_model import LemLite
    az_el = (torch.rand(B, 2, generator=g) - 0.5) * 0.8
    cmd = torch.as_tensor(LemLite.gaze_to_handle(cmd.numpy(), az_el.numpy(), convention="legacy"))
    ach = torch.zeros(B, H, 10)
    ach[..., 0:3] = torch.randn(B, H, 3, generator=g) * 0.1
    q0 = torch.randn(B, H, 4, generator=g)
    ach[..., 3:7] = q0 / q0.norm(dim=-1, keepdim=True)
    eye = (torch.rand(B, 2, generator=g) - 0.5) * 40.0          # degrees
    return cmd, ach, eye


def size_matched_violations(builder, convention="legacy"):
    """What is wrong with a size-matched builder: the 25 body rows must equal contact_head.shuffled_order; the eyes'
    row must be the other body's asked eye turn (command minus its eyes' angles) added to this body's eyes' angles;
    and the result must differ from the absolute roll."""
    from lem.model.contact_head import shuffled_order, order_features
    from lem.model.efference import gaze_order_features
    from mimo_bridge.gaze_reflex import AZ_RANGE, EL_RANGE
    cmd, ach, eye = _commands()
    if convention == "physical":
        from mimo_bridge.lem_model import LemLite
        az_el = gaze_order_features(cmd, GAZE, "legacy")[:, 9:11]
        cmd = torch.as_tensor(LemLite.gaze_to_handle(cmd.numpy(), az_el.numpy(), convention="physical"))
    out = builder(cmd, ach, GAZE, eye, convention)
    bad = []
    body = [i for i in range(cmd.shape[1]) if i != GAZE]
    ref = shuffled_order(cmd, ach, GAZE)
    if not torch.equal(out[:, body], ref[:, body]):
        bad.append("the body rows are not contact_head.shuffled_order")
    if not torch.allclose(order_features(out, ach, GAZE), order_features(cmd.roll(1, 0), ach.roll(1, 0), GAZE),
                          atol=1e-4):
        bad.append("the asked moves and turns are not the other body's")
    feats = gaze_order_features(out, GAZE, convention)[:, 9:11]
    own = gaze_order_features(cmd, GAZE, convention)[:, 9:11]
    e = eye * (math.pi / 180.0)
    want = (e + (own.roll(1, 0) - e.roll(1, 0)))
    want = torch.stack([want[:, 0].clamp(*AZ_RANGE), want[:, 1].clamp(*EL_RANGE)], dim=-1)
    if not torch.allclose(feats, want, atol=1e-5):
        bad.append("the eyes' row is not the other body's asked eye turn applied to this body's eyes")
    if not torch.equal(out[:, GAZE, [0, 1, 2, 7, 8]], cmd.roll(1, 0)[:, GAZE, [0, 1, 2, 7, 8]]):
        bad.append("the eyes' row's other numbers are not the other body's")
    if torch.allclose(out, cmd.roll(1, 0)):
        bad.append("the command is the absolute roll")
    return bad


@pytest.mark.parametrize("convention", ["legacy", "physical"])
def test_the_size_matched_command_is_the_contact_heads_control_with_the_eyes_matched(convention):
    from lem.model.efference import size_matched_other_order

    def build(cmd, ach, gi, eye, conv):
        return size_matched_other_order(cmd, ach, gi, eye_deg=eye, convention=conv)
    assert size_matched_violations(build, convention) == []


def test_the_size_matched_check_catches_the_absolute_roll():
    def absolute(cmd, ach, gi, eye, conv):           # planted defect: the old control
        return cmd.roll(1, dims=0).clone()

    def contact_only(cmd, ach, gi, eye, conv):       # planted defect: the eyes' row left as it stands
        from lem.model.contact_head import shuffled_order
        return shuffled_order(cmd, ach, gi)
    assert any("shuffled_order" in v or "absolute" in v for v in size_matched_violations(absolute))
    assert any("eyes' row" in v for v in size_matched_violations(contact_only))


def test_without_eye_angles_the_eyes_row_stays_the_other_bodys():
    from lem.model.contact_head import shuffled_order
    from lem.model.efference import size_matched_other_order
    cmd, ach, _ = _commands()
    assert torch.equal(size_matched_other_order(cmd, ach, GAZE), shuffled_order(cmd, ach, GAZE))


# ------------------------------------------------------------------------------ the shifted-own arm
def test_the_command_spread_is_each_columns_standard_deviation_and_ignores_quaternion_sign():
    g = torch.Generator().manual_seed(4)
    B, H = 8, 26
    sd = torch.linspace(0.5, 10.0, H * 9).reshape(H, 9)
    state = None
    for i in range(600):
        x = torch.randn(B, H, 9, generator=g) * sd + 3.0
        x[..., 3:7] = torch.tensor([0.0, 0.0, 0.6, 0.8]) * (1.0 if i % 2 else -1.0)   # one orientation, two signs
        state = command_spread_update(state, x)
        if i == 198:
            assert command_spread_std(state) is None           # fewer than 200 model steps of stream
    std = command_spread_std(state).reshape(H, 9).float()
    keep = torch.ones(9, dtype=torch.bool)
    keep[3:7] = False
    rel = (std[:, keep] / sd[:, keep] - 1.0).abs()
    assert float(rel.max()) < 0.15, float(rel.max())           # 4,800 values per column
    assert float(std[:, 3:7].max()) < 1e-6                     # q and -q are one orientation


def shifted_violations(fn):
    """What is wrong with a shifted-own builder: deterministic per (seed, model step), different between model
    steps, the noise on each non-quaternion column one standard deviation of that column, unit quaternions, and no
    draw from the global random stream."""
    bad = []
    B, H = 8, 26
    exec_t = torch.zeros(B, H, 9)
    exec_t[..., 6] = 1.0
    exec_t[..., 0:3] = 5.0
    sd = torch.linspace(0.1, 4.0, H * 9, dtype=torch.float64)
    rng = torch.get_rng_state()
    a = fn(exec_t, sd, 77, 0)
    if not torch.equal(torch.get_rng_state(), rng):
        bad.append("the global random stream was drawn from")
    if not torch.equal(a, fn(exec_t, sd, 77, 0)):
        bad.append("not a function of the seed and the model step")
    if torch.equal(a, fn(exec_t, sd, 78, 0)):
        bad.append("the same noise on two model steps")
    draws = torch.stack([fn(exec_t, sd, s, 0) - exec_t for s in range(400)])     # [400, B, H, 9]
    emp = draws.reshape(-1, H, 9).std(dim=0)
    want = sd.reshape(H, 9).float()
    cols = [0, 1, 2, 7, 8]
    if not torch.allclose(emp[:, cols], want[:, cols], rtol=0.12):
        bad.append("the noise is not one standard deviation per column")
    if not torch.allclose(a[..., 3:7].norm(dim=-1), torch.ones(B, H), atol=1e-5):
        bad.append("a quaternion is not of length 1")
    return bad


def test_the_shifted_own_command_adds_one_standard_deviation_per_column():
    assert shifted_violations(shifted_own_order) == []


def test_the_shifted_check_catches_noise_not_added():
    def no_noise(exec_t, std, step, seed):           # planted defect
        return exec_t.clone()

    def half(exec_t, std, step, seed):               # planted defect: the August size
        return shifted_own_order(exec_t, std, step, seed, size=0.5)
    assert shifted_violations(no_noise)
    assert any("one standard deviation" in v for v in shifted_violations(half))


# ------------------------------------------------------------------------------ the learner
@pytest.fixture(scope="module")
def fx():
    import importlib
    return importlib.import_module("mimo_bridge.tests.swap_test_fixture")


def _learner(fx, **kw):
    import random
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    m, a = fx.build(fx.argv(**kw))
    return m, a


def test_the_hold_first_rows_and_the_arms_appear_on_their_model_steps(fx):
    m, a = _learner(fx)
    w = fx.SyntheticWorld(a.n_envs, seed=1)
    seen = {}

    def on_step(step, model, res):
        seen[step] = (bool(model.prediction_audit_hold_first_rows),
                      sorted(model.prediction_audit_hold_first_rows.get("proprioception", {})),
                      sorted((model.last_prediction_audit.get("proprioception") or {})))
    fx.drive(m, w, 42, on_step=on_step)
    assert [s for s, v in seen.items() if v[0]] == [1, 21, 41]
    keys = seen[21][1]
    for k in ("shuffled_action_mse", "size_matched_action_mse", "size_matched_own_prediction_mse"):
        assert k in keys
    assert "shifted_own_transitions" not in keys                 # fewer than 200 model steps of stream
    assert "size_matched_action_mse" in seen[20][2] and "size_matched_action_mse" not in seen[21][2]
    assert m._command_spread["steps"] == 41
    ef = m.efference_window_summary()
    assert ef["size_matched_eye_as_stands"] == 0.0
    for k in ("body_cm_size_matched_next", "body_cm_size_matched_latch", "body_deg_size_matched_next"):
        assert ef[k] is not None, k
    assert all(v is not None for v in ef["gaze_deg_size_matched"])
    assert ef["gaze_copy"]["vision_err_copy_size_matched"] is not None
    assert "absolute" in ef["controls"]["shuffled"].lower()


def test_the_swap_arms_put_the_size_matched_command_into_the_motor_token(fx, monkeypatch):
    from lem.model import efference as EF
    from mimo_bridge.lem_model import command_for_encoder
    m, a = _learner(fx)
    w = fx.SyntheticWorld(a.n_envs, seed=1)
    made, motor_rows = [], []
    real_builder = EF.size_matched_other_order
    real_encode = m.encoder_registry.encode

    def spy_builder(*args, **kw):
        out = real_builder(*args, **kw)
        made.append(out.clone())
        return out

    def spy_encode(frame, mask=None):
        if set(frame) == {"motor"}:
            motor_rows.append(frame["motor"].clone())
        return real_encode(frame, mask)

    def on_step(step, model, res):
        if step == 20:                                       # model step 21 grades a hold's first model step
            monkeypatch.setattr(EF, "size_matched_other_order", spy_builder)
            monkeypatch.setattr(model.encoder_registry, "encode", spy_encode)
    fx.drive(m, w, 22, on_step=on_step)
    assert made, "the size-matched builder was not called"
    wants = [command_for_encoder(x) for x in made]
    hits = [r for r in motor_rows for want in wants
            if r.shape[-1] == 19 and r.shape[:-1] == want.shape[:-1] and torch.equal(r[..., 10:], want)]
    assert hits, "no motor token was built from the size-matched command"


def test_the_episode_companion_carries_the_command_spread_and_an_old_one_restores(fx):
    from mimo_bridge import continuation_state as CS
    m, a = _learner(fx)
    w = fx.SyntheticWorld(a.n_envs, seed=1)
    fx.drive(m, w, 6)
    state = CS.capture_learner(m, include_replay=False)
    assert "_command_spread" in state["data"] and state["data"]["_command_spread"]["steps"] == 5
    assert "prediction_audit_hold_first_rows" not in state["data"]          # a public name: never saved
    m2, _ = _learner(fx)
    CS.restore_learner(m2, state)
    assert torch.equal(m2._command_spread["mean"], m._command_spread["mean"])
    old = dict(state, data={k: v for k, v in state["data"].items() if k != "_command_spread"})
    m3, _ = _learner(fx)
    CS.restore_learner(m3, old)                                # a companion saved before this code
    assert m3._command_spread is None


def _run(fx, n, added=True, watch_added=False, monkeypatch=None):
    from lem.model import liveness as LV
    PA.ADDED_READINGS = added
    try:
        if watch_added:                               # planted defect: the added passes join the watching window
            monkeypatch.setattr(LV, "unwatched", lambda model: contextlib.nullcontext())
        m, a = _learner(fx)
        fx.plant_dead_units(m)
        rows = fx.drive(m, fx.SyntheticWorld(a.n_envs, seed=1), n)
        return rows, fx.revivals(m), fx.state_digest(m)
    finally:
        PA.ADDED_READINGS = True
        if watch_added:
            monkeypatch.undo()


def test_the_added_readings_change_no_loss_weight_command_or_random_stream(fx, monkeypatch):
    """A learner with the added readings and one without train bit for bit alike over 45 model steps (3 hold-first
    and 2 existing-block audits, the side heads graded every model step) while the liveness repair revives units
    (fixture.plant_dead_units), and the comparison catches the added passes joining the repair's watching window."""
    on, rev_on, d_on = _run(fx, 45)
    off, rev_off, d_off = _run(fx, 45, added=False)
    assert rev_on and rev_on == rev_off, "the liveness repair must revive units in this run"
    diff = [(a["step"], k) for a, b in zip(on, off) for k in a if a[k] != b.get(k)]
    assert diff == [] and d_on == d_off
    planted, _, d_planted = _run(fx, 45, watch_added=True, monkeypatch=monkeypatch)
    assert any(a != b for a, b in zip(on, planted)) or d_planted != d_on
