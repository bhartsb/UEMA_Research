# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The switch-jump measurement (docs/gain_of_function_v4/motor_fixes_switch_jump.py; review of the first
branch, finding 3; the deploy branch's review, findings 3, 4 and 5) on a small CPU learner and synthetic
rings: its replay forward reproduces the replay lesson's own per-sense errors on the same windows, each
order form changes exactly the fields its fix changes, the order centre is converted by the production
conversion (radius entries under F4, nothing under F1), an F1 form's body head reads the rows the ledger
keeps, the replay contract is the replay lesson's own, and the declared safe-jump rule - 400 replays of
the pre-deploy record, each window moved by the jumps of one measured block - stops where the kill-line
watcher would, refuses what it cannot judge, and decides together only on the combined form too.
"""
import importlib.util
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

DOCS = Path(__file__).resolve().parents[4] / "docs" / "gain_of_function_v4"


def _load():
    sys.path.insert(0, str(DOCS))
    spec = importlib.util.spec_from_file_location("motor_fixes_switch_jump", DOCS / "motor_fixes_switch_jump.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SJ = _load()


def _names():
    from mimo_bridge.body_layout import names_for
    return list(names_for("trunk_v1"))


NAMES = _names()
GI = NAMES.index("gaze")


def _learner(no_vision=True, **kw):
    from mimo_bridge.lem_model import LemLite
    torch.set_num_threads(1)
    torch.manual_seed(0)
    return LemLite(device="cpu", seed=0, imagination=False, no_vision=no_vision, next_frame_loss=True,
                   body_layout="trunk_v1", efference_heads=True, next_frame_norm="persistence",
                   persistent_memory="on", overrides={"model": {"context_length": 4}}, **kw)


def _filled_ring(m, n_envs=2, size=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    m.enable_replay(n_envs, size_per_env=size)
    rp = m.replay
    H = len(m.handle_names)
    rp.tokens.copy_(torch.randn(rp.tokens.shape, generator=g).half())
    ach = torch.randn(n_envs, size, H, 10, generator=g) * 0.1
    q = torch.randn(n_envs, size, H, 4, generator=g)
    ach[..., 3:7] = q / q.norm(dim=-1, keepdim=True)
    rp.achieved.copy_(ach.half())
    exe = torch.randn(n_envs, size, H, 9, generator=g) * 20.0
    q = torch.randn(n_envs, size, H, 4, generator=g)
    exe[..., 3:7] = q / q.norm(dim=-1, keepdim=True)
    exe[..., 7:9] = torch.rand(n_envs, size, H, 2, generator=g)
    rp.executed.copy_(exe.half())
    rp.has_executed.fill_(True)
    rp.valid.fill_(True)
    rp.episode.zero_()
    rp.action_contract.fill_(1)
    if rp.memory_states is not None:
        rp.memory_states.copy_(torch.randn(rp.memory_states.shape, generator=g) * 0.1)
    rp.ptr = size
    rp.count = size
    return rp


def test_the_replicated_forward_reproduces_the_replay_lessons_own_errors():
    m = _learner()
    _filled_ring(m)
    torch.manual_seed(3)
    w = m.replay.sample_windows(8, m.ctx_len, m.device)
    assert w is not None and w["memory_state"] is not None
    # the gaze copy reads only once the heads' running means are set (as after a lived moment)
    m.efference.ensure_centres(torch.randn(2, m.n_mod, m.d_model), torch.randn(2, len(NAMES), 9))
    theirs = SJ.replay_terms_of_the_model(m, w)
    _, _, se, keep = SJ.replay_forward(m, w, w["executed"])
    mine = {s: float((se[s] * keep[s].float()).sum() / keep[s].float().sum().clamp(min=1.0)) for s in se}
    assert theirs and set(theirs) == set(mine)
    for s in theirs:
        assert abs(mine[s] - theirs[s]) <= 1e-5 * max(1.0, abs(theirs[s])), (s, mine[s], theirs[s])
    # and the replay lesson's weights were put back: a second reading is the same
    assert SJ.replay_terms_of_the_model(m, w) == theirs
    # the orders reach the forward: another form changes the errors
    other = w["executed"].clone()
    other[..., 7] = 0.0
    _, _, se2, _ = SJ.replay_forward(m, w, other)
    assert any(not torch.equal(se[s], se2[s]) for s in se)


def test_each_form_changes_exactly_the_fields_of_its_fix():
    from mimo_bridge.action_decoder import position_handles
    g = torch.Generator().manual_seed(1)
    exe = torch.randn(3, 5, len(NAMES), 9, generator=g)
    pose = torch.randn(3, 5, len(NAMES), 10, generator=g) * 0.1
    stored = SJ.form_orders(exe, pose, "as_stored", NAMES, GI)
    assert torch.equal(stored, exe)
    f1 = SJ.form_orders(exe, pose, "f1", NAMES, GI)
    diff = (f1 != exe)
    assert torch.equal(f1[..., GI, 3:5], -exe[..., GI, 3:5])
    diff[..., GI, 3:5] = False
    assert not diff.any(), "F1 changes the eye row's quaternion x and y only"
    for form, rows in (("f4", "all"), ("f4_no_head_pelvis", "no_head_pelvis")):
        f4 = SJ.form_orders(exe, pose, form, NAMES, GI)
        live = [i for i, n in enumerate(NAMES) if n in position_handles(rows)]
        dead = [i for i in range(len(NAMES)) if i not in live]
        assert torch.equal(f4[..., live, 0:3], exe[..., live, 0:3])
        assert torch.allclose(f4[..., dead, 0:3], pose[..., dead, 0:3] * 200.0)
        assert float(f4[..., 7].abs().max()) == 0.0
        assert torch.equal(f4[..., 3:7], exe[..., 3:7]) and torch.equal(f4[..., 8], exe[..., 8])
        both = SJ.form_orders(exe, pose, "f1_" + form, NAMES, GI)
        want = f4.clone()
        want[..., GI, 3:5] = -want[..., GI, 3:5]
        assert torch.equal(both, want)
    assert SJ.form_settings("f1_f4_no_head_pelvis") == ("physical", True, "no_head_pelvis")
    # the plain f4 form keeps the rows of the run the ring was stored under
    assert SJ.form_settings("f4", "no_head_pelvis") == ("legacy", True, "no_head_pelvis")
    assert torch.equal(SJ.form_orders(exe, pose, "f4", NAMES, GI, "no_head_pelvis"),
                       SJ.form_orders(exe, pose, "f4_no_head_pelvis", NAMES, GI))


def test_the_forms_equal_what_the_learner_would_have_recorded():
    """F4's form of a stored order is LemLite.executed_fields_only on the pose at the order's latch; F1's
    is the row the stored-row rewrite at the switch writes (LemLite.mirror_eye_rows)."""
    from mimo_bridge.lem_model import LemLite
    g = torch.Generator().manual_seed(2)
    exe = torch.randn(2, len(NAMES), 9, generator=g) * 10.0
    pose = torch.randn(2, len(NAMES), 10, generator=g) * 0.1
    for rows in ("all", "no_head_pelvis"):
        form = "f4" if rows == "all" else "f4_no_head_pelvis"
        assert torch.allclose(SJ.form_orders(exe, pose, form, NAMES, GI),
                              LemLite.executed_fields_only(exe, pose, rows), atol=1e-5)
    assert torch.equal(SJ.form_orders(exe, pose, "f1", NAMES, GI), LemLite.mirror_eye_rows(exe, GI))


def test_the_latch_position_is_where_each_order_first_appears():
    T, H = 14, len(NAMES)
    exe = np.zeros((T, H, 9), dtype=np.float32)
    for t in range(T):
        exe[t, :, 0] = t // 5                          # a new order every 5 moments
        exe[t, GI, 3] = t                              # the eyes' row changes every moment
    hx = np.ones(T, dtype=bool)
    ep = np.zeros(T, dtype=np.int32)
    ep[12:] = 1                                        # a reset inside the third order
    got = SJ.latch_positions(exe, hx, ep, GI)
    assert got.tolist() == [0] * 5 + [5] * 5 + [10, 10, 12, 12]
    hx[6] = False                                      # an unlabelled moment starts a new run after it
    got = SJ.latch_positions(exe, hx, ep, GI)
    assert got[7] == 7 and got[6] == 5


def test_the_order_centre_is_converted_by_the_loaders_own_conversion():
    """The radius entries under F4; the eye entries stay under F1 (the controller writes the same row at the
    switch: the deploy branch's review, finding 3a; since the fourth review's fix round the live switch leaves
    them as saved too, LemLite.CENTRE_LEFT_AS_SAVED)."""
    H = len(NAMES)
    centre = torch.linspace(-1.0, 1.0, H * 9)
    radius = [h * 9 + 7 for h in range(H)]
    assert torch.equal(SJ.converted_centre(centre, 0.5, "as_stored", NAMES, GI), centre)
    assert torch.equal(SJ.converted_centre(centre, 0.5, "f1", NAMES, GI), centre)
    want = centre.clone()
    want[radius] = 0.0
    for form in ("f4", "f4_no_head_pelvis", "f1_f4", "f1_f4_no_head_pelvis"):
        assert torch.equal(SJ.converted_centre(centre, 0.5, form, NAMES, GI), want), form
    assert torch.equal(centre, torch.linspace(-1.0, 1.0, H * 9)), "the saved centre is not touched"


def test_an_f1_forms_body_head_reads_the_rows_the_ledger_keeps():
    enc = {f: "enc " + f for f in SJ.FORMS}
    orders = {f: "orders " + f for f in SJ.FORMS}
    centres = {f: "centre " + f for f in SJ.FORMS}
    for form, base in (("as_stored", "as_stored"), ("f4", "f4"), ("f4_no_head_pelvis", "f4_no_head_pelvis"),
                       ("f1", "as_stored"), ("f1_f4", "f4"), ("f1_f4_no_head_pelvis", "f4_no_head_pelvis")):
        assert SJ.body_head_inputs(form, enc, orders, centres) == (enc[base], orders[base], centres[base])
    # every F1 form's base is measured before it
    for form, base in SJ.BODY_BASE.items():
        assert SJ.FORMS.index(base) < SJ.FORMS.index(form)


def test_a_setting_the_run_already_has_is_not_measured_as_a_switch():
    forms, left = SJ.applicable_forms(dict(eye_row_convention="legacy", executed_fields_only=False))
    assert forms == list(SJ.FORMS) and not left
    forms, left = SJ.applicable_forms(dict(eye_row_convention="legacy", executed_fields_only=True))
    assert forms == ["as_stored", "f1"] and set(left) == {"f4", "f4_no_head_pelvis", "f1_f4", "f1_f4_no_head_pelvis"}
    forms, left = SJ.applicable_forms(dict(eye_row_convention="physical", executed_fields_only=False))
    assert forms == ["as_stored", "f4", "f4_no_head_pelvis"]
    assert all("already has" in why for why in left.values())


def test_the_moments_studied_carry_the_replay_lessons_own_contract():
    """eligible_positions keeps the windows LemLite.replay_update samples under the learner's settings
    (limb_target_control.replay_contract), with every context order stored."""
    from mimo_bridge.limb_target_control import POSITION_ROWS_CONTRACT_BIT
    m = SimpleNamespace(hand_target_mode="world", parent_relative_targets="off", position_rows="all")
    assert SJ.replay_contract_of(m) is None
    m.position_rows = "no_head_pelvis"
    assert SJ.replay_contract_of(m) == POSITION_ROWS_CONTRACT_BIT
    B, T, L = 2, 12, 3
    rp = dict(size=T, ptr=0, count=T, episode=torch.zeros(B, T, dtype=torch.long),
              has_executed=torch.ones(B, T, dtype=torch.bool),
              action_contract=torch.full((B, T), POSITION_ROWS_CONTRACT_BIT, dtype=torch.long))
    rp["action_contract"][1, 5] = 0                    # one moment of one body stored under 'all'
    got = SJ.eligible_positions(rp, L, SJ.replay_contract_of(m))
    # a window j reads moments j-L..j-1 for its context and step-from moment: 5 is in j = 6, 7, 8
    assert got.tolist() == [3, 4, 5, 9, 10, 11]
    assert SJ.eligible_positions(rp, L, None).tolist() == list(range(L, T))
    # the eye row's contract (the fourth review's fix round): under 'physical' the eye bit is required too
    from mimo_bridge.limb_target_control import EYE_ROW_CONTRACT_BIT
    m.eye_row_convention = "physical"
    assert SJ.replay_contract_of(m) == POSITION_ROWS_CONTRACT_BIT | EYE_ROW_CONTRACT_BIT
    assert SJ.eligible_positions(rp, L, SJ.replay_contract_of(m)).tolist() == []


def test_the_eye_angles_are_read_from_the_achieved_eye_row():
    from mimo_bridge.lem_model import LemLite
    rng = np.random.default_rng(4)
    for _ in range(20):
        h, v = rng.uniform(-0.7, 0.7), rng.uniform(-0.8, 0.5)
        row = LemLite.gaze_to_handle(np.zeros((1, len(NAMES), 9)), np.array([[h, v]]), convention="physical")
        ach = torch.zeros(1, len(NAMES), 10)
        ach[0, GI, 3:7] = torch.as_tensor(row[0, GI, 3:7])
        got = SJ.eye_angles_deg(ach[:, GI])[0]
        assert abs(float(got[0]) - math.degrees(h)) < 1e-3 and abs(float(got[1]) - math.degrees(v)) < 1e-3


def test_the_heads_readings_pair_each_body_with_another_body_of_the_same_moment():
    m = _learner()
    B, k = 4, 3
    R = B * k
    g = torch.Generator().manual_seed(5)
    lat = torch.randn(R, m.n_mod, m.d_model, generator=g)
    order = torch.randn(R, len(NAMES), 9, generator=g)
    other = torch.randn(R, len(NAMES), 9, generator=g)
    ach0 = torch.randn(R, len(NAMES), 10, generator=g) * 0.1
    ach1 = ach0 + 0.01
    seen, seen_gaze = [], []
    real, real_gaze = m.efference.body, m.efference.gaze

    def body(latent, o, a, record=False):
        seen.append(o.clone())
        return real(latent, o, a, record=record)

    def gaze(latent, o, a, head_motion=None, record=False):
        seen_gaze.append(o.clone())
        return real_gaze(latent, o, a, head_motion=head_motion, record=record)
    m.efference.body, m.efference.gaze = body, gaze

    def rolled(x):
        return x.reshape(k, B, *x.shape[1:]).roll(1, dims=1).reshape(x.shape)
    SJ.head_readings(m, lat, order, ach0, ach1, B)
    own, shuf = seen
    assert torch.equal(own, order) and torch.equal(shuf, rolled(order))
    # the body head can read other orders than the gaze head (an F1 form's body head)
    seen.clear(), seen_gaze.clear()
    SJ.head_readings(m, lat, order, ach0, ach1, B, body_order_enc=other)
    assert torch.equal(seen[0], other) and torch.equal(seen[1], rolled(other))
    assert torch.equal(seen_gaze[0], order) and torch.equal(seen_gaze[1], rolled(order))
    # a centre handed in is used for the reading and put back after it
    before = m.efference.order_centre.clone()
    SJ.head_readings(m, lat, order, ach0, ach1, B, centre=torch.zeros_like(before) + 3.0)
    assert torch.equal(m.efference.order_centre, before)


# ------------------------------------------------------------------ the safe-jump rule
def _windows(values, field="causal_pe", folder="seg40_resume_x", first=1000):
    """One launch's stats records, 100 steps apart; `values` is causal_pe or the body-head margin."""
    recs = []
    for i, v in enumerate(values):
        r = {"type": "mimo", "step": first + 100 * i,
             "efference": {"body_cm_err_next": 0.4, "body_cm_shuffled_next": 0.7,
                           "gaze_deg_err": [2.0 + 0.01 * (i % 3 - 1)], "gaze_deg_shuffled": [3.4]},
             "causal_pe": 0.016, "cmd_response_cos_held": 0.4}
        if field == "causal_pe":
            r["causal_pe"] = v
        else:
            r["efference"]["body_cm_err_next"] = r["efference"]["body_cm_shuffled_next"] - v
        recs.append(r)
    return [(folder, recs)]


def _blocks(n, body=0.0, gaze=0.0, causal=0.0):
    return [(body, gaze, causal)] * n


def test_a_jump_every_block_shares_stops_every_replay_or_none():
    # causal_pe windows (the first is the launch's own and not judged): a jump of 0.0045 puts the three
    # middle windows over 0.031 in a row, a jump of 0.001 puts none over
    win = _windows([0.020, 0.020, 0.027, 0.0275, 0.0275, 0.020])
    out = SJ.safe_verdict(win, _blocks(25, causal=0.0045), replays=40)
    assert out["safe"] is False and out["unshifted_record_stops"] is None
    assert out["stops_independent"] == 40 and out["stops_sequential"] == out["sequential_starts"] == 25
    assert out["stop_share_per_100"] == 100.0
    # the judged windows 1 to 5 hold three runs of 3; the middle one crosses with every run of 3 blocks
    assert out["pairs"] == 23 * 3 and out["crossing_pairs"] == 23
    assert out["stops_by_line"]["causal error"] == dict(independent=40, sequential=25, crossing_pairs=23)
    assert out["stops_by_line"]["body head"] == dict(independent=0, sequential=0, crossing_pairs=0)
    out = SJ.safe_verdict(win, _blocks(25, causal=0.001), replays=40)
    assert out["safe"] is True and out["stop_share_per_100"] == 0.0 and out["crossing_pairs"] == 0
    # zero jumps are the unmoved record, which never stopped the watcher
    assert SJ.safe_verdict(win, _blocks(25), replays=40)["stop_share_per_100"] == 0.0
    # a margin moves down by a harmful jump (the head's own error moves up)
    win = _windows([0.30, 0.30, 0.10, 0.10, 0.10, 0.30], field="body")
    assert SJ.safe_verdict(win, _blocks(25, body=-0.11), replays=40)["safe"] is False
    assert SJ.safe_verdict(win, _blocks(25, body=-0.05), replays=40)["safe"] is True


def test_a_jump_that_varies_window_to_window_stops_where_its_mean_would_not():
    """The deploy branch's review, finding 4: a mean jump the constant-jump rule passed stopped the
    watcher in 90 to 236 of 400 replays once each window carried its own jump."""
    win = _windows([0.10] * 40, field="body")
    # the mean jump -0.07 leaves every margin at 0.03 cm, above the line; half the blocks move -0.12
    blocks = _blocks(20, body=-0.02) + _blocks(20, body=-0.12)
    assert SJ.safe_verdict(win, _blocks(40, body=-0.07))["safe"] is True
    out = SJ.safe_verdict(win, blocks)
    # 39 judged windows, each past the line with probability 1/2: no three in a row in about 4 of
    # every 100 replays
    assert out["safe"] is False and out["stops_independent"] > 300, out
    assert out["stops_by_line"]["body head"]["independent"] > 300
    assert out["stops_by_line"]["causal error"] == dict(independent=0, sequential=0, crossing_pairs=0)


def test_the_rule_refuses_what_it_cannot_judge_and_names_its_line(monkeypatch):
    win = _windows([0.020] * 8)
    out = SJ.safe_verdict(win, _blocks(SJ.MIN_BLOCKS - 1))
    assert out["safe"] is None and "refused" in out["why"]
    # non-finite blocks do not count
    out = SJ.safe_verdict(win, _blocks(SJ.MIN_BLOCKS) + [(float("nan"), 0.0, 0.0)])
    assert out["blocks"] == SJ.MIN_BLOCKS and out["safe"] is True
    # a record that already stops the watcher is never safe
    bad = _windows([0.040] * 5)
    out = SJ.safe_verdict(bad, _blocks(SJ.MIN_BLOCKS))
    assert out["safe"] is False and out["unshifted_record_stops"]
    # the line: fewer than 5 of every 100 replays (19 of 400 passes, 20 does not)
    for stops, safe in ((19, True), (20, False)):
        monkeypatch.setattr(SJ, "replay_stops", lambda *a, _s=stops, **k: _s)
        assert SJ.safe_verdict(win, _blocks(SJ.MIN_BLOCKS))["safe"] is safe


def test_the_replays_are_reproducible_and_judge_launches_as_the_watcher_does():
    win = _windows([0.10] * 12, field="body") + _windows([0.10] * 12, field="body", folder="seg40_resume_y",
                                                          first=5000)
    blocks = _blocks(10, body=-0.02) + _blocks(10, body=-0.12)
    compact = [(f, [SJ.WATCH.compact(r) for r in rs]) for f, rs in win]
    a = SJ.replay_stops(compact, blocks, replays=50, seed=3)
    assert a == SJ.replay_stops(compact, blocks, replays=50, seed=3)
    # the short record a stop writes (30 steps here) is neither a window nor a break: two full
    # windows past the line, the short record past it, then a full window under it - no stop
    recs = _windows([0.30, 0.05, 0.05, 0.05, 0.30, 0.30], field="body")[0][1]
    steps = [1000, 1100, 1200, 1230, 1330, 1430]
    for r, s in zip(recs, steps):
        r["step"] = s
    got = SJ.safe_verdict([("seg40_resume_x", recs)], _blocks(25, body=-0.10), replays=5)
    assert got["safe"] is True and got["stop_share_per_100"] == 0.0 and got["crossing_pairs"] == 0
    # the all-pairings test counts runs of judged windows as the watcher does: the short record is skipped
    # without breaking a run, and the record 100 steps after it is a full window
    runs = SJ.judged_window_runs([("seg40_resume_x", recs)])
    assert [[c["step"] for c in run] for _, run in runs] == [[1100, 1200, 1330], [1200, 1330, 1430]]
    # judged as a window, the short record would have completed three in a row
    recs[3]["step"] = 1300
    recs[4]["step"], recs[5]["step"] = 1400, 1500
    assert SJ.safe_verdict([("seg40_resume_x", recs)], _blocks(25, body=-0.10), replays=5)["safe"] is False


def test_jumps_that_come_in_runs_are_judged_in_their_recorded_order():
    """The second review, finding 4: drawing every window's jump independently passed 4 consecutive
    blocks of a large jump (on the real record 2 of 400 replays stopped, against 179 of 400 with the
    blocks in their recorded order). The rule now also replays the blocks in their recorded order from
    random starts, and the verdict reads the larger count."""
    win = _windows([0.25] * 60, field="body")
    runs = [((-0.30 if 30 <= b < 34 else 0.0), 0.0, 0.0) for b in range(60)]
    out = SJ.safe_verdict(win, runs)
    # every start but the two that put the run of 4 across the launch's first (unjudged) window stops it
    assert out["stops_independent"] < 20 and out["stops_sequential"] == 58 and out["sequential_starts"] == 60, out
    assert out["safe"] is False and out["decided_by_measure"] == "sequential" and "sequential" in out["why"]
    # 2 runs of 3 blocks hold 3 of the 4, each crossing with every one of the 57 runs of judged windows
    assert out["crossing_pairs"] == 2 * 57 and out["pairs"] == 58 * 57
    # the same values with no persistence (every 15th block): no 3 in a row in the recorded order
    spread = [((-0.30 if b % 15 == 0 else 0.0), 0.0, 0.0) for b in range(60)]
    out = SJ.safe_verdict(win, spread)
    assert out["stops_sequential"] == 0 and out["crossing_pairs"] == 0 and out["safe"] is True
    # enumerated over every start: the same count every time, one outcome per start
    compact = [(f, [SJ.WATCH.compact(r) for r in rs]) for f, rs in win]
    assert SJ.replay_stops_sequential(compact, runs) == (58, 60)


def test_persistent_jumps_too_small_to_stop_anything_are_safe_whatever_their_autocorrelation():
    """The third review, finding C: the correlation refusal (the blocks worth fewer than 20 independent
    ones) withheld verdicts from persistent jumps too small to stop anything. It is gone; the lag-1
    autocorrelations are reported; the all-pairings test judges what decides a stop."""
    win = _windows([0.30] * 60, field="body")
    drift = [(-0.05 + 0.04 * math.sin(2 * math.pi * b / 60), 0.0, 0.0) for b in range(60)]
    out = SJ.safe_verdict(win, drift, replays=40)
    assert out["safe"] is True and out["crossing_pairs"] == 0 and out["stop_share_per_100"] == 0.0
    assert out["lag1_autocorrelation"]["body_margin"] > 0.9 and "effective_blocks" not in out
    assert not hasattr(SJ, "effective_blocks")
    # a jump every block shares reads no correlation
    assert SJ.lag1_autocorrelations([(-0.07, 0.0, 0.0)] * 60)[0] == 0.0


def test_a_crossing_pair_under_the_stop_line_refuses_the_verdict_and_names_it():
    """The third review's rule: below the stop line, a pairing of a run of 3 blocks with a run of 3 judged
    windows that crosses means the verdict depends on how often the two meet in a watch, which the ring
    cannot tell: refused, naming the crossing runs."""
    # one low stretch of 3 windows in the record, one run of 3 large jumps in the ring
    win = _windows([0.40] * 20 + [0.10] * 3 + [0.40] * 37, field="body")
    blocks = [((-0.15 if 30 <= b < 33 else 0.0), 0.0, 0.0) for b in range(60)]
    out = SJ.safe_verdict(win, blocks)
    assert out["crossing_pairs"] == 1 and out["stops_sequential"] == 1 and out["sequential_starts"] == 60
    assert out["stops_independent"] < 20
    assert out["safe"] is None and out["why"].startswith("refused")
    assert out["crossing_runs"] == [dict(first_block=30, launch="seg40_resume_x",
                                         window_steps=[3000, 3100, 3200], lines=["body head"])]
    assert "blocks 30 to 32 with the windows at steps 3000/3100/3200" in out["why"]
    # the record's own low stretch meets no large run of blocks: SAFE
    out = SJ.safe_verdict(win, [((-0.15 if b in (30, 32) else 0.0), 0.0, 0.0) for b in range(60)])
    assert out["safe"] is True and out["crossing_pairs"] == 0


def test_the_verdict_names_the_watch_it_assumes_and_what_a_longer_one_does():
    """The second review, point 7: the rule replays the pre-deploy record once, while the watcher watches
    until it writes a stop; a SAFE form's chance of a stop grows with the watch."""
    win = _windows([0.30] * 60, field="body")
    out = SJ.safe_verdict(win, _blocks(25), replays=40)
    w = out["watch"]
    assert w["windows"] == 60 and w["steps"] == 6000
    assert abs(w["hours"] - 6000 / 0.502 / 3600.0) < 0.06
    assert w["stop_chance_per_100_over"]["48 hours"] == 0.0 and "watch as long as the replayed record" in out["why"]
    # p = 5 in 100 over the record: a watch k times as long stops with chance 1 - 0.95^k
    w = SJ.watch_told(win, 20, 400)
    k = 48.0 / (w["steps"] / SJ.PACE_STEPS_PER_S / 3600.0)
    assert abs(w["stop_chance_per_100_over"]["48 hours"] - round(100.0 * (1 - 0.95 ** k), 1)) < 1e-9
    assert w["stop_chance_per_100_over"]["96 hours"] > w["stop_chance_per_100_over"]["48 hours"]
    # the deploy's own watch, 6,000 updates (the operator's decision of 2026-09-25): here as long as the record
    assert SJ.DEPLOY_WATCH_STEPS == 6000
    assert w["stop_chance_per_100_over"]["the deploy's watch of 6000 updates"] == 5.0
    longer = SJ.watch_told(_windows([0.30] * 240, field="body"), 20, 400)
    assert longer["stop_chance_per_100_over"]["the deploy's watch of 6000 updates"] == round(100 * (1 - 0.95 ** 0.25), 1)


def _following(live_body, live_gaze, stored_offset=(0.0, 0.0), sign=1.0, n_blocks=25, per_block=4, jumps=None):
    """Readings whose as-stored block margins follow the live windows over the same steps: block b (steps
    1000 + 100 b ...) holds the live window recorded at step 1100 + 100 b, moved by stored_offset
    (sign -1: mirrored about the live mean, so they fall when the live ones rise); each form moved by its
    (body, gaze, causal) jump."""
    steps = [1000 + 100 * b + i for b in range(n_blocks) for i in range(per_block)]
    lb, lg = np.asarray(live_body, float), np.asarray(live_gaze, float)
    per = {}
    for f, (jb_, jg, jc) in dict({"as_stored": (0.0, 0.0, 0.0)}, **(jumps or {})).items():
        bm = [(lb.mean() + sign * (lb[b + 1] - lb.mean())) + stored_offset[0] + jb_ for b in range(n_blocks)
              for _ in range(per_block)]
        gm = [(lg.mean() + sign * (lg[b + 1] - lg.mean())) + stored_offset[1] + jg for b in range(n_blocks)
              for _ in range(per_block)]
        per[f] = dict(step=list(steps), body_err=[0.7 - m for m in bm], body_shuf=[0.7] * len(steps),
                      gaze_err=[3.4 - m for m in gm], gaze_shuf=[3.4] * len(steps), causal=[0.016 + jc] * len(steps))
    return per


def _live(body, gaze, first=1000):
    """One launch's stats records with these body-head and gaze-head margins (100 steps apart)."""
    recs = []
    for i, (b, g) in enumerate(zip(body, gaze)):
        recs.append({"type": "mimo", "step": first + 100 * i,
                     "efference": {"body_cm_err_next": 0.7 - b, "body_cm_shuffled_next": 0.7,
                                   "gaze_deg_err": [3.4 - g], "gaze_deg_shuffled": [3.4]},
                     "causal_pe": 0.016, "cmd_response_cos_held": 0.4})
    return [("seg40_resume_x", recs)]


def test_the_head_margins_must_follow_the_live_gauge_and_scale_the_jumps(monkeypatch):
    """The third review, finding D: the two-sided level line (the as-stored margin within one live standard
    deviation of the live one) refused a readout whose jump carries over. Now the gap is recorded; a verdict
    is refused only when the as-stored block margins are not shown to rise and fall with the live windows over
    the same steps (correlation over at least 20 pairs not above twice the spread of shuffled pairings: the
    fourth review, finding 3) or sit at or under the kill line; and each head's jumps are scaled up by the live
    mean over the as-stored mean, never down."""
    rng = np.random.default_rng(2)
    lb = 0.30 + 0.05 * np.sin(np.arange(30) / 3.0) + rng.normal(0, 0.005, 30)
    lg = 1.40 + 0.10 * np.cos(np.arange(30) / 4.0) + rng.normal(0, 0.01, 30)
    live = _live(lb, lg)
    # the readout follows, 0.10 cm and 0.20 degrees under the live gauge: passed, jumps scaled up
    per = _following(lb, lg, stored_offset=(-0.10, -0.20))
    m = SJ.margin_fidelity(live, per["as_stored"])
    assert m["passed"] and m["body_margin_cm"]["pairs"] == 24 and m["body_margin_cm"]["correlation_with_live"] > 0.99
    b = m["body_margin_cm"]
    assert b["scale"] == pytest.approx(b["live_mean"] / b["as_stored_mean"]) and b["scale"] > 1.3
    assert b["difference_in_live_sds"] > 1, "a gap the old line refused is recorded, not refused"
    # the readout over the live gauge: never scaled down
    m = SJ.margin_fidelity(live, _following(lb, lg, stored_offset=(0.10, 0.20))["as_stored"])
    assert m["passed"] and m["body_margin_cm"]["scale"] == 1.0 and m["gaze_margin_deg"]["scale"] == 1.0
    # the readout falls when the live gauge rises: refused
    m = SJ.margin_fidelity(live, _following(lb, lg, sign=-1.0)["as_stored"])
    assert not m["passed"] and "not shown to rise and fall" in m["body_margin_cm"]["why"]
    # at or under the kill line: refused
    m = SJ.margin_fidelity(live, _following(lb, lg, stored_offset=(-0.50, 0.0))["as_stored"])
    assert not m["body_margin_cm"]["passed"] and "kill line" in m["body_margin_cm"]["why"]
    # too few live windows over the ring's steps: refused
    m = SJ.margin_fidelity(_live(lb[:12], lg[:12]), per["as_stored"])
    assert not m["passed"] and "under the 20" in m["body_margin_cm"]["why"]
    # through the rule: a body jump of -0.10 cm is SAFE unscaled on this record and NOT SAFE scaled by the
    # live over as-stored ratio (the record's lowest three windows sit 0.125 to 0.14 cm over the line)
    low = lb.copy()
    low[10:13] = [0.13, 0.125, 0.14]
    live = _live(low, lg)
    per = _following(low, lg, stored_offset=(-0.12, 0.0), jumps={"f4": (-0.10, 0.0, 0.0)})
    out = _summarise(monkeypatch, {}, live, margins="real", per=per,
                     settings=dict(eye_row_convention="physical", executed_fields_only=False))
    v = out["verdicts"]["f4"]
    assert v["jump_scale"]["body_margin_cm"] > 1.5 and v["safe"] is False, v
    assert out["fidelity"]["margins"]["passed"] is True
    unscaled = SJ.safe_verdict(live, [(-0.10, 0.0, 0.0)] * 25)
    assert unscaled["safe"] is True


def _unrelated(lb, lg, seed, n_blocks=25, per_block=4, spread=(0.05, 0.10)):
    """Readings whose as-stored block margins are UNRELATED to the live windows: each block's margins drawn at
    random about the live means (a readout that reads another quantity with the same level and spread)."""
    rng = np.random.default_rng(seed)
    steps = [1000 + 100 * b + i for b in range(n_blocks) for i in range(per_block)]
    bb = np.mean(lb) + rng.normal(0, spread[0], n_blocks)
    gb = np.mean(lg) + rng.normal(0, spread[1], n_blocks)
    bm = [bb[b] for b in range(n_blocks) for _ in range(per_block)]
    gm = [gb[b] for b in range(n_blocks) for _ in range(per_block)]
    return dict(step=steps, body_err=[0.7 - x for x in bm], body_shuf=[0.7] * len(steps),
                gaze_err=[3.4 - x for x in gm], gaze_shuf=[3.4] * len(steps), causal=[0.016] * len(steps))


def test_a_readout_unrelated_to_the_live_gauge_is_refused(monkeypatch):
    """The fourth review, finding 3 (planted fault): the head-margin check passed a readout whose block margins
    correlate with the live windows above 0, so a readout unrelated to the live gauge passed half the time. Now
    it must exceed twice the spread of the correlations shuffled pairings give, computed in the job."""
    # 200 planted readouts: the spread to about 4 in 100 (raising=False: the file also runs on the code before
    # the fix, which fails on how often it passes an unrelated readout, not on a missing name)
    monkeypatch.setattr(SJ, "FIDELITY_SHUFFLES", 300, raising=False)
    rng = np.random.default_rng(2)
    lb = 0.30 + 0.05 * np.sin(np.arange(30) / 3.0) + rng.normal(0, 0.005, 30)
    lg = 1.40 + 0.10 * np.cos(np.arange(30) / 4.0) + rng.normal(0, 0.01, 30)
    live = _live(lb, lg)
    rows = [SJ.margin_fidelity(live, _unrelated(lb, lg, seed))["body_margin_cm"] for seed in range(200)]
    rs = np.asarray([b["correlation_with_live"] for b in rows])
    passed = [b["passed"] for b in rows]
    old_rule = float((rs > 0).mean())           # the line before the fix: correlation above 0
    assert 0.35 < old_rule < 0.65, old_rule
    assert float(np.mean(passed)) <= 0.06, "an unrelated readout passed %.2f of the time" % float(np.mean(passed))
    for b in rows:
        assert b["correlation_line"] == pytest.approx(2.0 * b["shuffled_spread"])
        assert 0.15 < b["shuffled_spread"] < 0.30, "about 1 / sqrt(24) at 24 pairs"
    # one planted readout the old line passed: correlation above 0, under twice the shuffled spread
    seed = next(s for s, r in enumerate(rs) if 0.05 < r < 0.2)
    m = SJ.margin_fidelity(live, _unrelated(lb, lg, seed))
    assert not m["passed"] and "not shown to rise and fall" in m["body_margin_cm"]["why"]
    assert "twice the spread" in m["body_margin_cm"]["why"]


def test_the_shuffled_spread_is_measured_on_the_pairs_it_judges():
    a = np.arange(40.0)
    s = SJ.shuffled_correlation_spread(a, a[::-1] ** 1.5)
    assert abs(s - 1.0 / math.sqrt(39)) < 0.03
    assert SJ.shuffled_correlation_spread(a, np.ones(40)) == 0.0
    assert SJ.shuffled_correlation_spread(a, a, seed=1) == SJ.shuffled_correlation_spread(a, a, seed=1)


def _per(jumps, n_blocks=25, per_block=4):
    """Per-variant readings over n_blocks blocks of 100 steps: as_stored at fixed levels, each form moved
    by its (body margin, gaze margin, causal_pe) jump."""
    steps = [1000 + 100 * b + i for b in range(n_blocks) for i in range(per_block)]
    n = len(steps)
    per = {}
    for f, (jb, jg, jc) in dict({"as_stored": (0.0, 0.0, 0.0)}, **jumps).items():
        per[f] = dict(step=list(steps), body_err=[0.4 - jb] * n, body_shuf=[0.7] * n,
                      gaze_err=[2.0 - jg] * n, gaze_shuf=[3.4] * n, causal=[0.016 + jc] * n)
    return per, n


def _summarise(monkeypatch, jumps, windows, fidelity=0.0, settings=None, left_out=None, world=True,
               world_jumps=None, margins=None, per=None):
    """summarise on synthetic readings. An F1 form's second body-head reading (the rows a new order carries right
    after the switch) takes world_jumps[form] when given, else the form's own jumps; world=False: no recording
    of the changes the switch makes (--eye-rows). margins: the head margins' fidelity (margin_fidelity) as
    given (a dict), 'real' for the job's own on these readings, None for a passing check with no scaling
    (the synthetic readings hold every block at one level, so there is nothing for the live windows to
    follow). per: the readings, when the test builds them."""
    monkeypatch.setattr(SJ, "pre_deploy_windows", lambda segments=None: windows)
    if margins != "real":
        stub = margins or {"passed": True, "body_margin_cm": {"passed": True, "scale": 1.0},
                           "gaze_margin_deg": {"passed": True, "scale": 1.0}}
        monkeypatch.setattr(SJ, "margin_fidelity", lambda windows, base: stub)
    jumps = dict(jumps)
    if world:
        for f in [f for f in list(jumps) if f.startswith("f1")]:
            jumps[f + SJ.WORLD] = (world_jumps or {}).get(f, jumps[f])
    if per is None:
        per, n = _per(jumps)
    else:
        n = len(per["as_stored"]["step"])
    fid = None if fidelity is None else dict(largest_relative_difference=fidelity)
    cover = (dict(available=True, orders_given_a_change=8 * n) if world
             else dict(available=False, why="no --eye-rows: run motor_fixes_eye_row_split.py --eye-rows-out on the "
                                            "same checkpoint first"))
    return SJ.summarise(SimpleNamespace(stride=1, pre_deploy_segments="40"), None, per, {}, {}, fid,
                        np.arange(n), None, "ck", "args", 0.0, time.time(), 8,
                        settings or dict(eye_row_convention="legacy", executed_fields_only=False), left_out,
                        world=cover)


def test_the_combined_form_decides_the_deploy(monkeypatch):
    """The operator's decision of 2026-09-25 (WORKLOG b28dcdbc2): F4 and F1 are one semantic correction and
    switch together; the combined form f1_f4 gates the deploy, the single forms are diagnostics. The deploy
    branch's review, finding 4: f1 and f4 each move the body-head margin by -0.07 cm and pass; together they
    move it -0.14 cm, past the line on the record's lowest three windows."""
    win = _windows([0.30] * 5 + [0.10] * 3 + [0.30] * 20, field="body")
    out = _summarise(monkeypatch, {"f1": (-0.07, 0.0, 0.0), "f4": (-0.07, 0.0, 0.0),
                                   "f1_f4": (-0.14, 0.0, 0.0)}, win)
    v = out["verdicts"]
    assert v["f1"]["safe"] is True and v["f4"]["safe"] is True and v["f1_f4"]["safe"] is False
    rule = out["deploy_rule"]
    assert rule["deploy_allowed"] is False and rule["gating_form"] == "f1_f4"
    assert rule["gating_verdict"]["safe"] is False and rule["gating_verdict"]["stop_share_per_100"] >= 5.0
    assert rule["switches"] == ["--executed-fields-only on", "--eye-row-convention physical"]
    assert rule["diagnostic_verdicts"] == {"f4": True, "f1": True}
    assert "b28dcdbc2" in rule["rule"] and "together_allowed" not in rule and "f4_alone_allowed" not in rule
    assert "f3_or_limb_flags_in_the_same_relaunch" in rule
    # the block jumps the replays drew from are in the output
    blocks = out["forms"]["f1_f4"]["jump_blocks"]
    assert len(blocks) == 25 and all(abs(b["body_margin_cm"] + 0.14) < 1e-9 for b in blocks)
    assert out["pre_deploy_full_windows"] == 27
    # the combined form SAFE: the deploy is allowed, whatever a single form reads
    out = _summarise(monkeypatch, {"f1": (-0.14, 0.0, 0.0), "f4": (-0.02, 0.0, 0.0),
                                   "f1_f4": (-0.04, 0.0, 0.0)}, win)
    assert out["verdicts"]["f1"]["safe"] is False
    assert out["deploy_rule"]["deploy_allowed"] is True and out["deploy_rule"]["diagnostic_verdicts"]["f1"] is False


def test_the_gating_form_follows_what_the_run_already_has():
    assert SJ.gating_form(dict(eye_row_convention="legacy", executed_fields_only=False)) == "f1_f4"
    assert SJ.gating_form(dict(eye_row_convention="legacy", executed_fields_only=True)) == "f1"
    assert SJ.gating_form(dict(eye_row_convention="physical", executed_fields_only=False)) == "f4"
    assert SJ.gating_form(dict(eye_row_convention="physical", executed_fields_only=True)) is None
    assert SJ.switches_of("f1") == ["--eye-row-convention physical"]
    assert SJ.switches_of("f4") == ["--executed-fields-only on"] and SJ.switches_of(None) == []


def test_no_verdict_without_a_passing_fidelity_check(monkeypatch):
    win = _windows([0.28, 0.30, 0.32] * 10, field="body")
    jumps = {"f1": (0.0, 0.0, 0.0), "f4": (0.0, 0.0, 0.0), "f1_f4": (0.0, 0.0, 0.0)}
    for fidelity, passed in ((None, False), (2e-3, False), (5e-4, True)):
        out = _summarise(monkeypatch, jumps, win, fidelity=fidelity)
        assert out["fidelity"]["passed"] is passed
        for v in out["verdicts"].values():
            assert (v["safe"] is True) if passed else (v["safe"] is None and "fidelity" in v["why"])
        assert out["deploy_rule"]["deploy_allowed"] is passed
    # a setting the run already has: its forms carry the reason, never a verdict; the other's form decides
    out = _summarise(monkeypatch, {"f1": (0.0, 0.0, 0.0)}, win,
                     settings=dict(eye_row_convention="legacy", executed_fields_only=True),
                     left_out={"f4": "not measured: already on"})
    assert out["verdicts"]["f4"] == {"safe": None, "why": "not measured: already on"}
    assert out["deploy_rule"]["already_on"] == {"executed_fields_only": True}
    assert out["deploy_rule"]["gating_form"] == "f1" and out["deploy_rule"]["deploy_allowed"] is True
    assert out["deploy_rule"]["switches"] == ["--eye-row-convention physical"]


def test_an_f1_verdict_is_the_worse_of_its_two_body_head_readings(monkeypatch):
    """The four F1 follow-ups, item 3: the body head's rows at the live switch differ from the rows as stored;
    the F1 verdict reads the body head both ways (the rows as stored, and the rows a new order carries right after
    the switch) and the worse decides."""
    win = _windows([0.30] * 5 + [0.10] * 3 + [0.30] * 20, field="body")
    jumps = {"f1": (-0.02, 0.0, 0.0), "f4": (-0.02, 0.0, 0.0), "f1_f4": (-0.04, 0.0, 0.0)}
    # the combined form's rows as stored pass; with the rows after the switch its body head's margin falls past
    # the line: the gate is NOT SAFE, decided by the second reading
    out = _summarise(monkeypatch, jumps, win, world_jumps={"f1_f4": (-0.14, 0.0, 0.0)})
    v = out["verdicts"]["f1_f4"]
    assert v["safe"] is False and v["decided_by"] == "rows_after_switch"
    assert v["readings"]["rows_as_stored"]["safe"] is True and v["readings"]["rows_after_switch"]["safe"] is False
    assert out["verdicts"]["f1"]["safe"] is True and "f1_f4" + SJ.WORLD not in out["verdicts"]
    assert out["deploy_rule"]["deploy_allowed"] is False
    assert out["deploy_rule"]["gating_verdict"]["decided_by"] == "rows_after_switch"
    assert out["forms"]["f1_f4" + SJ.WORLD]["body_margin_cm"]["jump"] == pytest.approx(-0.14)
    # both readings pass: SAFE, and the deploy is allowed
    out = _summarise(monkeypatch, jumps, win)
    assert out["verdicts"]["f1_f4"]["safe"] is True and out["deploy_rule"]["deploy_allowed"] is True
    assert out["rows_after_switch_reading"]["available"] is True


def test_without_the_recording_of_the_rows_after_the_switch_the_f1_verdicts_are_refused(monkeypatch):
    win = _windows([0.28, 0.30, 0.32] * 10, field="body")
    out = _summarise(monkeypatch, {"f1": (0.0, 0.0, 0.0), "f4": (0.0, 0.0, 0.0), "f1_f4": (0.0, 0.0, 0.0)}, win,
                     world=False)
    for f in ("f1", "f1_f4"):
        v = out["verdicts"][f]
        assert v["safe"] is None and "needs the recording" in v["why"] and "--eye-rows" in v["why"]
        assert v["first_reading"]["safe"] is True
    assert out["verdicts"]["f4"]["safe"] is True
    assert out["deploy_rule"]["deploy_allowed"] is False and out["deploy_rule"]["gating_verdict"]["safe"] is None


def test_the_two_readings_combine_worst_first():
    safe = lambda s, n, pairs=0: dict(safe=s, stop_share_per_100=n, crossing_pairs=pairs, why="w%s" % n)  # noqa: E731
    c = SJ.combine_f1_readings(safe(True, 0.75), safe(True, 1.75))
    assert c["safe"] is True and c["stop_share_per_100"] == 1.75 and c["decided_by"] == "rows_after_switch"
    c = SJ.combine_f1_readings(safe(False, 7.5), safe(True, 10.0))
    assert c["safe"] is False and c["decided_by"] == "rows_as_stored"
    c = SJ.combine_f1_readings(safe(None, 0.0, 3), safe(True, 0.5))
    assert c["safe"] is None and c["decided_by"] == "rows_as_stored"
    c = SJ.combine_f1_readings(safe(None, 1.0, 2), safe(None, 1.0, 9))
    assert c["decided_by"] == "rows_after_switch", "the same share: the more crossing pairs decide"
    c = SJ.combine_f1_readings(safe(False, 6.25), safe(False, 7.75))
    assert c["safe"] is False and c["stop_share_per_100"] == 7.75
    assert set(c["readings"]) == {"rows_as_stored", "rows_after_switch"}


def _eye_legacy_orders(az, el, H=None):
    from mimo_bridge.lem_model import LemLite
    H = len(NAMES) if H is None else H
    base = np.random.default_rng(3).normal(size=(len(az), H, 9)).astype(np.float32)
    return torch.as_tensor(LemLite.gaze_to_handle(base, np.stack([az, el], -1), convention="legacy"))


def test_the_rows_after_the_switch_move_each_stored_elevation_by_its_recorded_change():
    """The second reading's rows (the fourth review's fix round): each stored ('legacy') row's azimuth kept, its
    elevation sent moved by the recorded change and clipped to the eye's range, written in 'physical'; nothing
    else in the order changes."""
    from mimo_bridge.gaze_reflex import EL_RANGE
    from mimo_bridge.lem_model import LemLite
    az = np.array([0.2, -0.3, 0.1, 0.0])
    el = np.array([0.1, -0.4, 0.5, -0.8])
    delta = np.array([-0.2, 0.3, 0.3, 1.9])                     # the last two leave the range: clipped
    orders = _eye_legacy_orders(az, el)
    got, clipped = SJ.rows_after_switch(orders, delta, "legacy", GI)
    want_el = np.clip(el + delta, *EL_RANGE)
    want = LemLite.gaze_to_handle(orders.numpy(), np.stack([az, want_el], -1), convention="physical")
    np.testing.assert_allclose(got[:, GI, 3:7].numpy(), want[:, GI, 3:7], atol=2e-6)
    assert clipped == 2
    other = torch.ones(len(NAMES), dtype=torch.bool)
    other[GI] = False
    assert torch.equal(got[:, other], orders[:, other]) and torch.equal(got[:, GI, 0:3], orders[:, GI, 0:3])
    # no change: the stored row written in 'physical' (the elevation it sent, the other convention's row)
    same, _ = SJ.rows_after_switch(orders, np.zeros(4), "legacy", GI)
    np.testing.assert_allclose(same[:, GI, 3:5].numpy(), -orders[:, GI, 3:5].numpy(), atol=2e-6)


def _eye_rows_file(path, B, steps=12, seed=5, drop_body=None, no_planner_body=None):
    """A --eye-rows recording: per body, delta_el (b*10 + the step, in hundredths of a radian, so the test can
    read which one an order took) and author (the planner every 4th step)."""
    rec = {}
    for b in range(B):
        if b == drop_body:
            continue
        author = np.array([1.0 if i % 4 == 0 and b != no_planner_body else 0.0 for i in range(steps)])
        rec["b%d__delta_el" % b] = (10.0 * b + np.arange(steps)) / 100.0
        rec["b%d__author" % b] = author
        rec["b%d__el_sent" % b] = np.zeros(steps)
    np.savez(str(path), **rec)
    return str(path)


def test_the_recorded_changes_pair_each_body_and_author_in_recorded_order(tmp_path):
    """EYE_ROWS_PAIRING: an order stored at one of the planner's decision steps takes that body's recorded planner
    changes in order, every other order the body's other changes in order, wrapping round."""
    path = _eye_rows_file(tmp_path / "rows.npz", B=2, steps=8)
    er = SJ.EyeRows(path, 2, intention_every=10)
    assert er.available and er.fallback == []
    # steps 20 and 30 are the planner's decision steps (multiples of 10); 21, 22 are not
    got = er.changes([20, 21, 22, 30]).numpy() * 100.0
    # body 0: planner changes at recorded steps 0 and 4 (0, 4), others 1, 2, 3, 5...; body 1 the same + 10
    np.testing.assert_allclose(got, [0, 10, 1, 11, 2, 12, 4, 14])
    got = er.changes([40, 41]).numpy() * 100.0                   # the planner's list wraps round (0, 4, 0)
    np.testing.assert_allclose(got, [0, 10, 3, 13])
    rec = er.record()
    assert rec["orders_given_a_change"] == 12 and rec["per_body"][0]["planner_steps"] == 2
    assert "not that order's own change" in rec["pairing"]
    # without the planner's cadence (intention off) every order takes the other steps' changes
    er = SJ.EyeRows(path, 2, intention_every=None)
    np.testing.assert_allclose(er.changes([20, 30]).numpy() * 100.0, [1, 11, 2, 12])
    # a body missing from the recording: not available, with the reason
    er = SJ.EyeRows(_eye_rows_file(tmp_path / "rows2.npz", B=3, drop_body=2), 3, intention_every=10)
    assert er.available is False and "[2]" in er.why
    # a body without planner steps takes its whole recording for them, and is named
    er = SJ.EyeRows(_eye_rows_file(tmp_path / "rows3.npz", B=2, no_planner_body=1), 2, intention_every=10)
    assert er.available and er.fallback == [1]


def test_the_gate_exit_code_reads_the_deploy_rule():
    assert SJ.gate_exit_code({"deploy_rule": {"deploy_allowed": True}}, True) == 0
    assert SJ.gate_exit_code({"deploy_rule": {"deploy_allowed": False}}, True) == 1
    assert SJ.gate_exit_code({"deploy_rule": {"deploy_allowed": False}}, False) == 0
    assert SJ.gate_exit_code({}, True) == 1


def _synthetic_companion_ring(m, B=4, T=40, seed=7, ctrl=True, history=True):
    """A replay ring as the episode companion carries it (vars of ReplayMemory), with orders held for
    5 moments by every body together (the trainer's latch) and the eyes' row changing every moment.
    ctrl: the model's gaze command beside every order (ReplayMemory.ctrl_gaze), else none.
    history: the persistent memory's history beside it (load_all keeps the companion's in
    'memory_history'), which the F1 forms' memory path is built from."""
    g = torch.Generator().manual_seed(seed)
    m.enable_replay(B, size_per_env=T)
    rp = m.replay
    H = len(NAMES)
    rp.tokens.copy_(torch.randn(rp.tokens.shape, generator=g).half())
    ach = torch.randn(B, T, H, 10, generator=g) * 0.1
    q = torch.randn(B, T, H, 4, generator=g)
    ach[..., 3:7] = q / q.norm(dim=-1, keepdim=True)
    rp.achieved.copy_(ach.half())
    exe = torch.zeros(B, T, H, 9)
    for s in range(0, T, 5):
        o = torch.randn(B, 1, H, 9, generator=g) * 20.0
        qq = torch.randn(B, 1, H, 4, generator=g)
        o[..., 3:7] = qq / qq.norm(dim=-1, keepdim=True)
        o[..., 7:9] = torch.rand(B, 1, H, 2, generator=g)
        exe[:, s:s + 5] = o
    eye = torch.randn(B, T, 4, generator=g)
    exe[:, :, GI, 3:7] = eye / eye.norm(dim=-1, keepdim=True)
    rp.executed.copy_(exe.half())
    rp.has_executed.fill_(True)
    rp.valid.fill_(True)
    rp.episode.zero_()
    rp.memory_states.copy_(torch.randn(rp.memory_states.shape, generator=g) * 0.1)
    if ctrl:
        rp.ctrl_gaze.copy_((torch.rand(rp.ctrl_gaze.shape, generator=g) - 0.5) * 0.8)
        rp.has_ctrl_gaze.fill_(True)
    rp.ptr, rp.count = T, T
    ring = dict(vars(rp))
    ring["step"] = 5000
    if history:
        ring["memory_history"] = _history_state(m, B=B)
    m.replay = None
    return ring


def test_the_job_runs_end_to_end_on_a_small_learner_with_the_gaze_copy(monkeypatch, tmp_path):
    """SJ.run on a small learner with vision (so the gaze copy into vision is part of the forward) and
    a synthetic companion ring: the replication matches the replay lesson, every form and centre
    variant is read on the same moments, the as-stored jumps are exactly 0, the forms with F4 move the
    body head, an F1 form's body head reads the rows the ledger keeps while its gaze head reads the
    rewritten rows under 'physical', the second reading gives each stored order a recorded change (the
    ring holds no gaze command: the deploy's case), the controllability tracker grades the full-tenure
    orders, and every form gets a verdict from the watcher's judgement."""
    m = _learner(no_vision=False)
    assert m.efference.vision_idx is not None
    m.efference.ensure_centres(torch.randn(2, m.n_mod, m.d_model), torch.randn(2, len(NAMES), 9))
    ring = _synthetic_companion_ring(m, ctrl=False)
    rows = _eye_rows_file(tmp_path / "eye_rows.npz", B=4, steps=12)
    monkeypatch.setattr(SJ, "load_all", lambda ck, dev: (m, ring, "synthetic", "synthetic args", 0.0,
                                                          {"intention": "on", "intention_every": 5}))
    monkeypatch.setattr(SJ, "pre_deploy_windows", lambda segments=None: _windows([0.016] * 8))
    # the ring's 40 steps hold no live stats window: the head-margin check has its own test above
    monkeypatch.setattr(SJ, "margin_fidelity", lambda windows, base: {
        "passed": True, "body_margin_cm": {"passed": True}, "gaze_margin_deg": {"passed": True}})
    monkeypatch.setattr(SJ, "BLOCK_STEPS", 1)          # 36 moments, 36 blocks: enough for a verdict
    monkeypatch.setattr(SJ, "REPLAYS", 40)
    calls = []
    real = SJ.head_readings

    def spy(model, last_step, order_enc, ach0, ach1, bodies, centre=None, body_order_enc=None):
        body = order_enc if body_order_enc is None else body_order_enc
        calls.append((model.efference.eye_row_convention, order_enc[:, GI, 3:5].clone(), body[:, GI, 3:5].clone()))
        return real(model, last_step, order_enc, ach0, ach1, bodies, centre=centre, body_order_enc=body_order_enc)
    monkeypatch.setattr(SJ, "head_readings", spy)
    out = SJ.run(SimpleNamespace(checkpoint="x", device="cpu", stride=1, max_moments=0,
                                 moments_per_batch=2, progress=False, pre_deploy_segments="40", eye_rows=rows))
    # the deploy's case (the fourth review, finding 1): a ring with no gaze command gets F1 verdicts, where the
    # code before the fix refused them for want of the command
    assert "first_reading" not in out["verdicts"]["f1_f4"], out["verdicts"]["f1_f4"].get("why")
    fid = out["fidelity"]
    assert fid["replay_update_terms"] and "vision" in fid["replicated_terms"]
    assert fid["largest_relative_difference"] < 1e-5 and fid["passed"], fid
    world = {f + SJ.WORLD for f in SJ.FORMS if f.startswith("f1")}
    variants = set(SJ.FORMS) | {"f4_centre_as_saved", "f4_no_head_pelvis_centre_as_saved"} | world
    assert set(out["forms"]) == variants
    wr = out["rows_after_switch_reading"]
    assert wr["available"] and wr["orders_given_a_change"] == 36 * 4 and wr["intention_every"] == 5
    assert wr["source"] == rows and "not that order's own change" in wr["pairing"]
    # the second reading changes only the body head: the gaze head and the senses read the F1 form's forward
    for f in ("f1", "f1_f4"):
        assert out["forms"][f + SJ.WORLD]["causal_pe"]["level"] == out["forms"][f]["causal_pe"]["level"]
        assert out["forms"][f + SJ.WORLD]["gaze_margin_deg"]["level"] == out["forms"][f]["gaze_margin_deg"]["level"]
        assert out["forms"][f + SJ.WORLD]["body_margin_cm"]["level"] != out["forms"][f]["body_margin_cm"]["level"]
    stored = out["forms"]["as_stored"]
    assert stored["body_margin_cm"]["jump"] == 0.0 and stored["causal_pe"]["jump"] == 0.0
    assert out["moments"] == 40 - 4
    assert out["forms"]["f4"]["body_margin_cm"]["jump"] != 0.0
    assert out["forms"]["f4"]["controllability"]["orders"] > 0
    assert out["forms"]["f1"]["controllability"]["orders"] > 0
    assert out["forms"]["f1"]["gaze_margin_deg"]["jump"] != 0.0
    # the centre variants change only the body head: the senses are read from the same forward
    assert out["forms"]["f4_centre_as_saved"]["causal_pe"]["level"] == out["forms"]["f4"]["causal_pe"]["level"]
    # per batch, 11 readings (6 forms, 2 centre variants, 3 second readings), in the order the loop makes
    # them: the body head reads the stored eye rows, except an F1 form's second reading, which reads the
    # rows after the switch (each stored elevation moved by a recorded change: neither the stored rows nor
    # their full mirror); the gaze head reads the rewritten rows exactly when the heads' convention is
    # 'physical'
    order = ["as_stored", "f4", "f4_centre_as_saved", "f4_no_head_pelvis", "f4_no_head_pelvis_centre_as_saved",
             "f1", "f1" + SJ.WORLD, "f1_f4", "f1_f4" + SJ.WORLD, "f1_f4_no_head_pelvis",
             "f1_f4_no_head_pelvis" + SJ.WORLD]
    per_batch = len(order)
    assert per_batch == len(variants) and len(calls) % per_batch == 0 and calls
    for i in range(0, len(calls), per_batch):
        batch = dict(zip(order, calls[i:i + per_batch]))
        own = batch["as_stored"][1]
        assert sum(conv == "physical" for conv, _, _ in batch.values()) == 6
        for name, (conv, gaze_xy, body_xy) in batch.items():
            assert torch.equal(gaze_xy, -own if conv == "physical" else own), name
            if name.endswith(SJ.WORLD):
                assert not torch.equal(body_xy, own) and not torch.equal(body_xy, -own), name
            else:
                assert torch.equal(body_xy, own), name
    assert set(out["verdicts"]) == set(SJ.FORMS) - {"as_stored"}
    # every form is judged by the rule: the independent replays, the sequential replay from every start,
    # the all-pairings test, the watch it assumes, the heads' jump scale; an F1 form by both of its body
    # head's readings
    assert all(v["replays"] == 40 and {"stops_independent", "stops_sequential", "sequential_starts",
                                       "crossing_pairs", "pairs", "stop_share_per_100", "watch",
                                       "jump_scale"} <= set(v) for v in out["verdicts"].values()), out["verdicts"]
    assert all(v["safe"] in (True, False, None) and v["why"] for v in out["verdicts"].values())
    assert all(set(out["verdicts"][f]["readings"]) == {"rows_as_stored", "rows_after_switch"}
               for f in ("f1", "f1_f4", "f1_f4_no_head_pelvis"))
    assert set(out["deploy_rule"]) >= {"deploy_allowed", "gating_form", "gating_verdict", "switches",
                                       "diagnostic_verdicts", "rule"}
    assert out["deploy_rule"]["gating_form"] == "f1_f4"
    assert out["deploy_rule"]["deploy_allowed"] is (out["verdicts"]["f1_f4"]["safe"] is True)
    assert out["run_settings"]["position_rows"] == "all"
    assert m.efference.eye_row_convention == "legacy", "the heads' own convention is put back"


def test_end_to_end_without_the_recording_the_f1_verdicts_are_refused(monkeypatch, tmp_path):
    """Without --eye-rows, or with a recording that lacks a body, the F1 verdicts are refused with the reason; a
    ring that holds the model's gaze command changes nothing (the switch and the job no longer read it)."""
    m = _learner()
    m.efference.ensure_centres(torch.randn(2, m.n_mod, m.d_model), torch.randn(2, len(NAMES), 9))
    ring = _synthetic_companion_ring(m, ctrl=True)
    monkeypatch.setattr(SJ, "load_all", lambda ck, dev: (m, ring, "synthetic", "synthetic args", 0.0, {}))
    monkeypatch.setattr(SJ, "pre_deploy_windows", lambda segments=None: _windows([0.016] * 8))
    monkeypatch.setattr(SJ, "margin_fidelity", lambda windows, base: {
        "passed": True, "body_margin_cm": {"passed": True}, "gaze_margin_deg": {"passed": True}})
    monkeypatch.setattr(SJ, "BLOCK_STEPS", 1)
    monkeypatch.setattr(SJ, "REPLAYS", 40)
    for rows, why in ((None, "no --eye-rows"),
                      (_eye_rows_file(tmp_path / "three.npz", B=4, drop_body=3), "holds no body [3]")):
        out = SJ.run(SimpleNamespace(checkpoint="x", device="cpu", stride=1, max_moments=0, moments_per_batch=2,
                                     progress=False, pre_deploy_segments="40", eye_rows=rows))
        assert not any(f.endswith(SJ.WORLD) for f in out["forms"])
        assert out["rows_after_switch_reading"]["available"] is False
        for f in ("f1", "f1_f4", "f1_f4_no_head_pelvis"):
            v = out["verdicts"][f]
            assert v["safe"] is None and "needs the recording" in v["why"] and why in v["why"] and "first_reading" in v
        assert out["verdicts"]["f4"]["replays"] == 40
        assert out["deploy_rule"]["deploy_allowed"] is False and out["deploy_rule"]["gating_form"] == "f1_f4"


def test_block_standard_errors_use_100_step_blocks():
    steps = np.arange(1000, 1400)
    vals = np.where((steps // 100) % 2 == 0, 1.0, 3.0)
    mean, se = SJ.block_se(vals, steps)
    assert mean == 2.0 and abs(se - np.std([1, 3, 1, 3], ddof=1) / 2.0) < 1e-12
    assert SJ.block_means(vals, steps) == {10: 1.0, 11: 3.0, 12: 1.0, 13: 3.0}


def test_the_job_needs_the_deploys_own_checkpoint():
    with pytest.raises(SystemExit):
        SJ.main(["--out", "x.json"])


# ------------------------------------------------------------------ the memory's path (the four F1 follow-ups, item 4)
def _history_state(m, B=4, n=6, seed=9, spacing=0.3):
    """A persistent-memory history of B bodies written by the memory's own writer (MemoryHistory.observe)
    with random events of the memory's input width, `spacing` seconds apart; its state_dict, as the
    episode companion carries it (learner['memory'])."""
    from lem.model.persistent_memory import MemoryHistory
    g = torch.Generator().manual_seed(seed)
    h = MemoryHistory(capacity=int(m.memory_history.capacity), interval_s=float(m.memory_history.interval_s))
    for t in range(n):
        f = torch.randn(B, int(m.persistent_memory.input_size), generator=g) * 0.3
        h.observe(f, torch.full((B,), spacing * t), m.persistent_memory)
    return h.state_dict()


def _open_readout(m, seed=4):
    """The memory's readout starts at 0 (its bias is then the same for every history); give it weights so
    the history's change reaches the bias."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.persistent_memory.readout.weight.copy_(torch.randn(m.persistent_memory.readout.weight.shape, generator=g) * 0.2)


def test_the_memory_bias_change_is_the_one_the_live_switch_makes():
    """SJ.memory_bias_change on a history equals bias(after the trainer's own switch rewrote it) minus
    bias(before), through the memory's own encoder and readout."""
    from mimo_bridge.train_mimo import switch_stored_eye_rows
    m = _learner()
    _open_readout(m)
    m.memory_history.load_state_dict(_history_state(m, B=2), m.device)
    snap = m.memory_history.snapshot()
    ev0 = snap["events"].clone()
    got = SJ.memory_bias_change(m, snap)
    assert torch.equal(snap["events"], ev0), "the snapshot handed in is not touched"
    pm = m.persistent_memory
    with torch.no_grad():
        before = pm.bias(pm.encode(**snap)[0])
    m.motor_interface_change = {"eye_row_convention": ["legacy", "physical"]}
    m.motor_interface_conversion = {"changes": dict(m.motor_interface_change)}
    switch_stored_eye_rows(m, None)
    with torch.no_grad():
        after = pm.bias(pm.encode(**m.memory_history.snapshot())[0])
    assert got.shape == before.shape and float(got.abs().max()) > 1e-6, "the eye columns reach the bias"
    assert torch.allclose(got, (after - before).float(), atol=1e-6)


def test_the_memory_path_leaves_the_repairs_event_centre_as_the_live_switch_leaves_it():
    """Since the fourth review's fix round the live switch leaves the repair's running mean of events as saved
    (LemLite.CENTRE_LEFT_AS_SAVED); the memory path's change is the history's rewrite alone, and the centre is not
    touched by it, set or not."""
    from mimo_bridge.train_mimo import switch_stored_eye_rows
    m = _learner()
    _open_readout(m)
    width = int(m.persistent_memory.input_size)
    centre = torch.linspace(-1.0, 1.0, width)
    m.persistent_memory.register_buffer("event_centre", centre.clone())
    m.persistent_memory.register_buffer("event_centre_weight", torch.ones((), dtype=torch.float64))
    m.memory_history.load_state_dict(_history_state(m, B=2), m.device)
    snap = m.memory_history.snapshot()
    got = SJ.memory_bias_change(m, snap)
    assert torch.equal(m.persistent_memory.event_centre, centre)
    # the live switch on the same learner leaves the centre too, and records why
    m.motor_interface_change = {"eye_row_convention": ["legacy", "physical"]}
    m.motor_interface_conversion = {"changes": dict(m.motor_interface_change)}
    switch_stored_eye_rows(m, None)
    assert torch.equal(m.persistent_memory.event_centre, centre)
    rec = m.motor_interface_conversion["event_centre_eye_shift"]
    assert rec["shift"] is None and "left as saved at the switch" in rec["left_as_saved"]
    assert float(got.abs().max()) > 1e-6, "the history's eye columns still reach the bias"


def test_the_memory_path_from_the_checkpoint_history_and_from_a_recorded_one(tmp_path):
    m = _learner()
    _open_readout(m)
    B = 4
    rp = dict(tokens=torch.zeros(B, 3, 2, 4), memory_history=_history_state(m, B=B))
    mp = SJ.memory_path(m, rp)
    assert mp["available"] and mp["source"].startswith("checkpoint")
    assert len(mp["per_body"]) == B and all(p.shape == (1, m.n_mod, m.d_model) for p in mp["per_body"])
    assert all(s["moments"] == 1 and s["change_rms"] > 0 for s in mp["sizes"])
    assert "cannot show how the change varies" in mp["meaning"]
    # no history: not available, with the reason; no memory: nothing to count
    assert SJ.memory_path(m, dict(tokens=rp["tokens"]))["available"] is False
    assert SJ.memory_path(SimpleNamespace(persistent_memory=None), rp) is None
    # a recorded history: every body's changes in their recorded order
    rec = {"b%d" % b: np.random.default_rng(b).normal(size=(5, m.n_mod, m.d_model)).astype(np.float32)
           for b in range(B)}
    path = tmp_path / "memory_path.npz"
    np.savez(str(path), **rec)
    mp = SJ.memory_path(m, rp, recorded=str(path))
    assert mp["available"] and mp["source"].startswith("recorded")
    assert torch.equal(mp["per_body"][2], torch.as_tensor(rec["b2"]))
    np.savez(str(path), **{k: v for k, v in rec.items() if k != "b3"})
    mp = SJ.memory_path(m, rp, recorded=str(path))
    assert mp["available"] is False and "[3]" in mp["why"]
    # each window takes its body's change k for its k-th analysed position, wrapping round
    per_body = [torch.arange(3, dtype=torch.float32).reshape(3, 1, 1) + 10 * b for b in range(2)]
    shift = SJ.memory_shift_for(np.array([7, 8, 9, 10]), 7, per_body, 2)
    assert shift.reshape(-1).tolist() == [0, 10, 1, 11, 2, 12, 0, 10]


def test_the_replay_forward_adds_the_memory_change_to_the_memory_bias():
    m = _learner()
    rp = _filled_ring(m, n_envs=2, size=40)
    w = rp.sample_windows(4, m.ctx_len, torch.device("cpu"))
    R = int(w["tokens"].shape[0])
    _, _, se0, _ = SJ.replay_forward(m, w, w["executed"])
    _, _, se_zero, _ = SJ.replay_forward(m, w, w["executed"], memory_shift=torch.zeros(R, m.n_mod, m.d_model))
    _, _, se1, _ = SJ.replay_forward(m, w, w["executed"], memory_shift=torch.ones(R, m.n_mod, m.d_model))
    assert all(torch.allclose(se0[k], se_zero[k], atol=1e-7, rtol=0) for k in se0)
    assert any(not torch.allclose(se0[k], se1[k], atol=1e-6, rtol=0) for k in se0)


def test_the_job_counts_the_memorys_path_in_the_f1_forms_end_to_end(monkeypatch, tmp_path):
    """With the checkpoint's own history the F1 forms read the memory bias changed by the switch, the other
    forms do not; without a history the F1 verdicts are refused with the reason."""
    m = _learner()
    _open_readout(m)
    m.efference.ensure_centres(torch.randn(2, m.n_mod, m.d_model), torch.randn(2, len(NAMES), 9))
    ring = _synthetic_companion_ring(m)
    ring["memory_history"] = _history_state(m, B=4)
    rows = _eye_rows_file(tmp_path / "eye_rows.npz", B=4)
    monkeypatch.setattr(SJ, "load_all", lambda ck, dev: (m, ring, "synthetic", "synthetic args", 0.0, {}))
    monkeypatch.setattr(SJ, "pre_deploy_windows", lambda segments=None: _windows([0.016] * 8))
    monkeypatch.setattr(SJ, "margin_fidelity", lambda windows, base: {
        "passed": True, "body_margin_cm": {"passed": True}, "gaze_margin_deg": {"passed": True}})
    monkeypatch.setattr(SJ, "BLOCK_STEPS", 1)
    monkeypatch.setattr(SJ, "REPLAYS", 40)
    seen = []
    real = SJ.replay_forward

    def spy(model, w, executed, memory_shift=None):
        seen.append(memory_shift is not None and float(memory_shift.abs().sum()) > 0)
        return real(model, w, executed, memory_shift=memory_shift)
    monkeypatch.setattr(SJ, "replay_forward", spy)
    a = SimpleNamespace(checkpoint="x", device="cpu", stride=1, max_moments=0, moments_per_batch=2,
                        progress=False, pre_deploy_segments="40", memory_path=None, eye_rows=rows)
    out = SJ.run(a)
    mp = out["memory_path"]
    assert mp["available"] and mp["source"].startswith("checkpoint") and len(mp["sizes"]) == 4
    # per batch: the fidelity check's forward first, then one forward per form; the F1 forms carry the change
    forms = [f for f in SJ.FORMS]
    body = seen[1:]
    assert len(body) % len(forms) == 0
    for i in range(0, len(body), len(forms)):
        assert body[i:i + len(forms)] == [f.startswith("f1") for f in forms]
    assert out["verdicts"]["f1"]["safe"] in (True, False, None) and "memory" not in out["verdicts"]["f1"]["why"]
    # without a history: the F1 verdicts are refused, F4's stand
    ring.pop("memory_history")
    seen.clear()
    out = SJ.run(a)
    assert out["memory_path"]["available"] is False
    for f in ("f1", "f1_f4", "f1_f4_no_head_pelvis"):
        assert out["verdicts"][f]["safe"] is None and "memory" in out["verdicts"][f]["why"]
    assert out["verdicts"]["f4"]["replays"] == 40
