# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The regression pack (mimo_bridge/tools/regression_pack.py).

Every reading is checked on data whose answer is known: a token that never changes reads 0 and an unrelated
token every moment reads its own spread; fingers that arrive at their wrist-relative goal close 100% of the turn
even while the wrist turns, fingers that stay put close 0%; the model part's construction equals production's
own replay lesson on the same window, reads exactly 0 when every body was given the same order (the control),
and counts a body only where its senses were real; the stats records of an abandoned future are cut; the
observer's occasions and body-steps follow the tracker's rules and the body-2 exclusion; the pooling with and
without body 2, the no-change lines, the verdicts and the table's words.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimo_bridge.tools import regression_pack as RP

TRUNK = ('wrist_L', 'thumb_L', 'index_L', 'middle_L', 'ring_L', 'pinky_L', 'wrist_R', 'thumb_R', 'index_R',
         'middle_R', 'ring_R', 'pinky_R', 'head', 'upperarm_L', 'lowerarm_L', 'upperarm_R', 'lowerarm_R', 'thigh_L',
         'calf_L', 'thigh_R', 'calf_R', 'foot_L', 'foot_R', 'gaze', 'lower_trunk', 'chest')
MODS = ["vision", "tactile", "audio", "proprioception", "motor", "vestibular"]


def _ring(B=3, N=2000, d=8, H=26):
    """A synthetic replay ring in the companion's layout (fp16 storage, full, oldest slot 0)."""
    achieved = torch.zeros(B, N, H, 10)
    achieved[..., 6] = 1.0
    executed = torch.zeros(B, N, H, 9)
    executed[..., 6] = 1.0
    executed[..., 8] = 0.5
    has = torch.ones(B, N, dtype=torch.bool)
    has[:, -1] = False                       # the newest moment's outgoing order is not back-filled yet
    return dict(mods=list(MODS), tokens=torch.zeros(B, N, len(MODS), d).half(), achieved=achieved.half(),
                executed=executed.half(), has_executed=has, valid=torch.ones(B, N, len(MODS), dtype=torch.bool),
                episode=torch.zeros(B, N, dtype=torch.int32), ptr=N, count=N, size=N, n_envs=B,
                memory_states=None, self_states=None)


# ================================================================================ token spread (readings 3-4)
def test_a_token_that_never_changes_reads_zero_and_an_unrelated_token_reads_its_spread():
    ring = _ring(B=3, N=6000)
    g = torch.Generator().manual_seed(0)
    k = MODS.index("proprioception")
    ring["tokens"][1, :, k] = torch.randn(6000, 8, generator=g).half()          # unrelated every moment
    alt = torch.tensor([0.3, -0.3]).repeat(3000)                                # +-0.3 alternating
    ring["tokens"][2, :, k] = alt[:, None].expand(6000, 8).half()
    ring["tokens"][0, :, k] = 0.7                                               # the same every moment
    sp = RP.token_spreads(ring)
    p = sp["proprioception"]
    assert len(p[0]["windows"]) == 6, "six 50-s windows of 1,000 moments in a full 6,000-moment ring"
    assert p[0]["mean"] == 0.0
    assert 0.85 < p[1]["mean"] < 1.15
    # every 5th moment of an alternating sequence: 200 samples of one value, since 5 is odd they alternate
    assert abs(p[2]["mean"] - 0.3 * math.sqrt(200 / 199)) < 1e-3
    assert sp["vestibular"][0]["mean"] == 0.0


def test_a_window_crossing_an_episode_reset_is_left_out():
    ring = _ring(B=1, N=2000)
    ring["episode"][0, 1500:] = 1
    sp = RP.token_spreads(ring)["proprioception"][0]
    assert len(sp["windows"]) == 1, "the window 1,000-1,999 crosses the reset at 1,500"


def test_the_spread_follows_the_ring_in_time_order_after_a_wrap():
    ring = _ring(B=1, N=2000)
    k = MODS.index("proprioception")
    # the newest 1,000 moments sit in slots 700-1,699 once the write head is at 1,700 (ptr 3,700)
    ring["ptr"] = 3700
    order = RP.chronological(ring)
    assert int(order[0]) == 1700 and int(order[-1]) == 1699
    ring["tokens"][0, :, k] = 0.0
    ring["tokens"][0, 700:1700:2, k] = 1.0
    sp = RP.token_spreads(ring)["proprioception"][0]
    assert sp["windows"][-1] > 0.4 and sp["windows"][0] == 0.0


# ============================================================================================ tenures
def test_an_order_is_one_tenure_while_its_limb_rows_hold_and_the_eyes_move():
    ring = _ring(B=1, N=20)
    ex = ring["executed"][0].float()
    gi = TRUNK.index("gaze")
    for t in range(20):
        ex[t, gi, 3] = 0.01 * t                         # the eye row changes every moment
        ex[t, 0, 0] = float(t // 5)                     # a new limb order every 5 moments
    pairs = RP.tenures(ex, ring["has_executed"][0], ring["episode"][0], gi)
    # the first order began before the ring (moment 0 has no predecessor), the last has not ended
    assert pairs == [(5, 9), (10, 14)]


def test_an_episode_reset_or_an_unlabelled_moment_inside_an_order_drops_it():
    ring = _ring(B=1, N=30)
    ex = ring["executed"][0].float()
    for t in range(30):
        ex[t, 0, 0] = float(t // 5)
    ep = ring["episode"][0].clone()
    has = ring["has_executed"][0].clone()
    has[12] = False                                     # inside the order of moments 10-14
    ep[21:] = 1                                         # a reset inside the order of moments 20-24
    assert RP.tenures(ex, has, ep, TRUNK.index("gaze")) == [(5, 9), (15, 19)]


# ======================================================================================= hand following
def _axis_angle(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    h = math.radians(deg) / 2
    return np.r_[axis * math.sin(h), math.cos(h)]


def _qmul(a, b):
    av, aw, bv, bw = np.asarray(a[:3]), a[3], np.asarray(b[:3]), b[3]
    return np.r_[aw * bv + bw * av + np.cross(av, bv), aw * bw - np.dot(av, bv)]


def _hand_ring(arrive: bool, tenures=6, wrist_turn_deg=10.0, half=False):
    """tenures + 2 orders of 5 moments each (the first and the last are not whole in the ring, so `tenures` are
    scored). Order k asks each finger for its wrist-relative pose R30 (k even) or the identity (k odd), while
    the wrist turns 10 degrees about the world vertical per order and is ordered exactly where it ends.
    `arrive`: the fingers end each order at their goal, else they never move relative to the wrist. Stored in
    float32 unless `half` (the companion's fp16)."""
    K = tenures + 2
    N = 5 * K + 1
    ring = _ring(B=1, N=N)
    ex = ring["executed"][0].float().numpy()
    ach = ring["achieved"][0].float().numpy()
    R30 = _axis_angle([1, 0, 0], 30)
    I = np.array([0, 0, 0, 1.0])
    fingers = {s: [TRUNK.index(d + "_" + s) for d in ("thumb", "index", "middle", "ring", "pinky")] for s in "LR"}
    wrists = {s: TRUNK.index("wrist_" + s) for s in "LR"}
    for k in range(K + 1):
        qw = _axis_angle([0, 0, 1], wrist_turn_deg * k)
        rel_at = (R30 if k % 2 else I) if arrive else I
        for t in range(5 * k, min(N, 5 * k + 5)):
            for s in "LR":
                ach[t, wrists[s], 3:7] = qw
                for f in fingers[s]:
                    ach[t, f, 3:7] = _qmul(qw, rel_at)
        if k < K:
            qw_end = _axis_angle([0, 0, 1], wrist_turn_deg * (k + 1))
            goal = R30 if k % 2 == 0 else I
            for t in range(5 * k, 5 * k + 5):
                for s in "LR":
                    ex[t, wrists[s], 3:7] = qw_end
                    for f in fingers[s]:
                        ex[t, f, 3:7] = _qmul(qw_end, goal)
                ex[t, 0, 0] = k + 1                       # a new order every 5 moments
    cast = (lambda x: x.half()) if half else (lambda x: x)
    ring["executed"] = cast(torch.tensor(ex, dtype=torch.float32)[None])
    ring["achieved"] = cast(torch.tensor(ach, dtype=torch.float32)[None])
    ring["has_executed"][0, :] = True
    ring["has_executed"][0, -1] = False
    return ring


def test_fingers_that_arrive_relative_to_a_turning_wrist_close_the_whole_turn():
    out = RP.hand_following(_hand_ring(arrive=True), TRUNK)[0]
    f = out["fingers"]
    assert out["orders"] == 6, "the ring's first and last orders are not whole"
    assert f["part_orders"] == 60, "10 fingers x 6 orders"
    assert f["asked_deg"] == pytest.approx(60 * 30.0, rel=1e-4)
    assert f["left_deg"] < 1e-2 and f["closer"] == 60
    w = out["wrists"]
    assert w["part_orders"] == 12 and w["asked_deg"] == pytest.approx(120.0, rel=1e-4) and w["left_deg"] < 1e-2


def test_the_companions_half_precision_leaves_under_half_a_degree():
    f = RP.hand_following(_hand_ring(arrive=True, half=True), TRUNK)[0]["fingers"]
    assert f["asked_deg"] == pytest.approx(60 * 30.0, rel=2e-3)
    assert f["left_deg"] < 0.5 * f["part_orders"] and f["closer"] == f["part_orders"]


def test_fingers_that_stay_put_relative_to_the_wrist_close_nothing_although_the_wrist_moved_them():
    out = RP.hand_following(_hand_ring(arrive=False), TRUNK)[0]
    f = out["fingers"]
    # the orders back to the start ask for no turn (below the scorable floor) and are left out
    assert f["part_orders"] == 30
    assert f["left_deg"] == pytest.approx(f["asked_deg"], rel=2e-3)
    assert f["closer"] == 0
    pooled = RP.pooled(_ck(hands=[out]), ())
    assert pooled["hand_following"] == pytest.approx(0.0, abs=2e-3)


def test_the_hand_chance_line_is_the_following_functions_own():
    from lem.model.motor_synergy import following_counts
    ring = _hand_ring(arrive=True)
    out = RP.hand_following(ring, TRUNK)[0]
    ex, ach = ring["executed"][0].float(), ring["achieved"][0].float()
    wr = torch.tensor([TRUNK.index("wrist_L"), TRUNK.index("wrist_R")])
    s = torch.tensor([5 * k for k in range(1, 7)])
    c = following_counts(ex[s], ach[s], ach[s + 5], wr)
    assert out["wrists"]["chance"] == pytest.approx(float(c["chance"].sum()), rel=1e-6)


# =============================================================================== the model part (1-2)
@pytest.fixture(scope="module")
def lite():
    """A small real LemLite (128 wide, context 4) with vision, the gaze copy, the persistent memory and a
    4-body replay ring of 40 random moments; the gaze copy's centre set from one real latent."""
    torch.set_num_threads(2)
    from mimo_bridge.lem_model import LemLite, command_for_encoder
    torch.manual_seed(0)
    m = LemLite(device="cpu", seed=0, imagination=False, body_layout="trunk_v1", persistent_memory="on",
                self_state="learn", efference_heads=True, next_frame_loss=True)
    B, H = 4, len(m.handle_names)
    m.enable_replay(n_envs=B, size_per_env=40)
    g = torch.Generator().manual_seed(1)
    mem = m.replay.memory_states.shape[-1]
    for t in range(40):
        toks = {k: torch.randn(B, 1, m.d_model, generator=g) for k in m.active}
        ach = torch.zeros(B, H, 10)
        ach[..., 0:3] = torch.randn(B, H, 3, generator=g) * 0.05
        q = torch.randn(B, H, 4, generator=g)
        ach[..., 3:7] = q / q.norm(dim=-1, keepdim=True)
        order = torch.zeros(B, H, 9)
        order[..., 0:3] = torch.randn(B, H, 3, generator=g) * 10.0
        q = torch.randn(B, H, 4, generator=g)
        order[..., 3:7] = q / q.norm(dim=-1, keepdim=True)
        order[..., 7] = 1.0
        order[..., 8] = 0.5
        m.replay.add(toks, ach, {"rom_limit": 1.0, "strength_limit": 1.0}, executed=order,
                     memory_state=torch.randn(B, mem, generator=g))
    ring = dict(vars(m.replay))
    w = RP.gather_windows(ring, [2], m.ctx_len)
    with torch.no_grad():
        f = RP._replay_forward(m, w, w["executed"][:, m.ctx_len - 1])
        m.efference.ensure_centres(f["last"], command_for_encoder(w["executed"][:, m.ctx_len - 1]))
    return m, ring


def test_the_model_part_builds_production_replay_lessons_own_prediction(lite):
    m, ring = lite
    L = m.ctx_len
    w = RP.gather_windows(ring, [3], L)
    R = w["tokens"].shape[0]
    with torch.no_grad():
        mine = RP._replay_forward(m, w, w["executed"][:, L - 1])
    seen = {}

    def keep_last(_mod, _inp, out):             # a forward hook that returns None leaves the output alone
        if "last" not in seen:
            seen["last"] = out[1].detach().clone()

    def keep_preds(_mod, _inp, out):
        if "preds" not in seen:
            seen["preds"] = {k: v.detach().clone() for k, v in out.items()}
    h1 = m.forward_model.register_forward_hook(keep_last)
    h2 = m.prediction_heads.register_forward_hook(keep_preds)
    window = {k: w[k] for k in ("tokens", "achieved", "valid", "executed", "has_executed", "memory_state", "self_state")}
    window.update(rom=torch.ones(R), strength=torch.ones(R))
    saved = [p.detach().clone() for p in m.params]
    real = m.replay.sample_windows
    m.replay.sample_windows = lambda *a, **k: window
    try:
        assert m.replay_update(batch=R) is not None
    finally:
        m.replay.sample_windows = real
        h1.remove()
        h2.remove()
        with torch.no_grad():
            for p, q in zip(m.params, saved):
                p.copy_(q)
    assert torch.allclose(mine["last"], seen["last"], atol=1e-6)
    for k, v in seen["preds"].items():
        assert torch.allclose(mine["preds"][k].float(), v.float(), atol=1e-6), k


def test_every_body_given_the_same_order_reads_exactly_zero(lite):
    m, ring = lite
    same = dict(ring)
    ex = ring["executed"].clone()
    ex[:] = ex[0:1]                                    # body 0's order everywhere
    same["executed"] = ex
    r = RP.model_readings(SimpleNamespace(model=m), same, n_slots=6)
    assert r["gaze_copy_active"] is True
    for b in r["per_body"]:
        assert b["body_rows"] > 0 and b["eye_rows"] > 0
        assert b["body_shift_sq"] == 0.0
        assert b["eye_err_own"] == b["eye_err_other"]
    p = RP.pooled(_ck(model=r, bodies=4), ())
    assert p["body_action_sensitivity"] == 0.0 and p["eye_vs_other_body"] == 0.0


def test_different_orders_move_the_prediction_and_the_pooling_adds_bodies(lite):
    m, ring = lite
    r = RP.model_readings(SimpleNamespace(model=m), ring, n_slots=6)
    assert all(b["body_shift_sq"] > 0 for b in r["per_body"])
    assert r["slots"] == sorted(r["slots"]) and len(r["slots"]) >= 2
    ck = _ck(model=r, bodies=4)
    s = {k: sum(b[k] for b in r["per_body"]) for k in r["per_body"][0]}
    p = RP.pooled(ck, ())
    assert p["body_action_sensitivity"] == pytest.approx(math.sqrt(s["body_shift_sq"] / s["body_change_sq"]))
    # the eye headline is the own command against the eyes held, anchored to the picture's own change; the other
    # body's command and the ratio of errors are kept beside it
    assert p["eye_prediction"] == pytest.approx((s["eye_err_still"] - s["eye_err_own"]) / s["eye_err_unchanged"])
    assert p["eye_vs_other_body"] == pytest.approx((s["eye_err_other"] - s["eye_err_own"]) / s["eye_err_unchanged"])
    assert p["eye_share_of_error"] == pytest.approx(1 - s["eye_err_own"] / s["eye_err_other"])
    # the per-window-end sums add up to the per-body sums
    for k in RP.MODEL_KEYS:
        assert float(np.asarray(r["per_slot"][k]).sum()) == pytest.approx(s[k])


# ================================================================ review fix 3: the eyes-held control
def _linear_copy_sums(g, n=4000, d=192, seed=0):
    """The review's linear model of the gaze copy (section 1): the eye-driven change e and the rest n are N(0, 1);
    the copy with response g leaves (1 - g) e + n with the own command, e - g e' + n with another body's (e' that
    body's eye-driven change) and e + n with the eyes held (no movement commanded) - which is also the unchanged
    picture's error. Summed squared errors per number, as model_readings sums them."""
    r = np.random.default_rng(seed)
    e, e2, rest = (r.standard_normal((n, d)) for _ in range(3))

    def err(x):
        return float(np.square(x).mean(-1).sum())
    return dict(eye_err_own=err((1 - g) * e + rest), eye_err_other=err(e - g * e2 + rest), eye_err_still=err(e + rest),
                eye_err_unchanged=err(e + rest), eye_rows=n, body_shift_sq=0.0, body_change_sq=1.0, body_err_own=1.0,
                body_err_other=1.0, body_err_unchanged=1.0, body_rows=n)


def _model_part(sums_by_body):
    """A model part in model_readings' layout with one window end."""
    return dict(per_body=sums_by_body, per_slot={k: [[b[k] for b in sums_by_body]] for k in RP.MODEL_KEYS},
                gaze_copy_active=True, variants={})


def test_the_eye_headline_falls_when_the_gaze_copy_over_reacts():
    """Fails on the pack's first eye control: against another body's command the headline rose to 199% at twice the
    right response and 298% at three times (the review's simulation), though the three-times copy predicts the
    eye-driven change worse than assuming the eyes stay still."""
    read = {g: RP.pooled(_ck(model=_model_part([_linear_copy_sums(g)]), bodies=1), ()) for g in (1.0, 2.0, 3.0)}
    assert read[1.0]["eye_prediction"] == pytest.approx(0.5, abs=0.02)
    assert read[2.0]["eye_prediction"] == pytest.approx(0.0, abs=0.02)
    assert read[3.0]["eye_prediction"] == pytest.approx(-1.5, abs=0.05)
    assert read[2.0]["eye_prediction"] < read[1.0]["eye_prediction"] and read[3.0]["eye_prediction"] < 0
    # beside it, the other-body form rises with the over-reaction (why it is not the headline)
    assert read[1.0]["eye_vs_other_body"] < read[2.0]["eye_vs_other_body"] < read[3.0]["eye_vs_other_body"]


def test_the_eyes_held_command_is_the_eye_row_of_the_moment_before(lite):
    m, ring = lite
    gi = list(m.handle_names).index("gaze")
    own, prev = torch.randn(3, len(m.handle_names), 9), torch.randn(3, len(m.handle_names), 9)
    still = RP.eyes_still_order(own, prev, gi)
    assert torch.equal(still[:, gi, 3:7], prev[:, gi, 3:7])
    keep = [i for i in range(9) if not 3 <= i < 7]
    assert torch.equal(still[:, gi, keep], own[:, gi, keep]) and torch.equal(still[:, :gi], own[:, :gi])
    # eyes whose command never changes: the eyes-held command IS the own command, so the headline reads exactly 0
    held = dict(ring)
    ex = ring["executed"].clone()
    ex[:, :, gi, 3:7] = torch.tensor([0.0, 0.1, 0.0, 0.995]).half()
    held["executed"] = ex
    r = RP.model_readings(SimpleNamespace(model=m), held, n_slots=6)
    for b in r["per_body"]:
        assert b["eye_rows"] > 0 and b["eye_err_still"] == b["eye_err_own"]
    assert RP.pooled(_ck(model=r, bodies=4), ())["eye_prediction"] == 0.0


# ======================================================= review fix 4b: body 2 never serves as a control
def test_the_control_body_is_the_previous_kept_body():
    assert RP.partners(8) == [7, 0, 1, 2, 3, 4, 5, 6], "the live audit's order.roll(1)"
    assert RP.partners(8, [2]) == [7, 0, 1, 1, 3, 4, 5, 6], "body 3's control is body 1 when body 2 is left out"
    assert RP.partners(4, [0]) == [3, 3, 1, 2]


def test_body_2s_orders_do_not_enter_the_readings_without_body_2(lite):
    """Fails on the pack's first partner rule: body 3's control order was body 2's (order.roll(1)), so the readings
    'without body 2' moved when only body 2's orders changed."""
    m, ring = lite
    r0 = RP.model_readings(SimpleNamespace(model=m), ring, n_slots=6)
    other = dict(ring)
    ex = ring["executed"].clone()
    g = torch.Generator().manual_seed(9)
    ex[2, :, :, 0:3] = (torch.randn(ex.shape[1], ex.shape[2], 3, generator=g) * 30.0).half()
    other["executed"] = ex
    r1 = RP.model_readings(SimpleNamespace(model=m), other, n_slots=6)
    p0, p1 = RP.pooled(_ck(model=r0, bodies=4), [2]), RP.pooled(_ck(model=r1, bodies=4), [2])
    assert p0["body_action_sensitivity"] == pytest.approx(p1["body_action_sensitivity"], rel=1e-6)
    assert p0["eye_vs_other_body"] == pytest.approx(p1["eye_vs_other_body"], rel=1e-6)
    # ...while the readings with every body do move (body 2's order is body 3's control there)
    a0, a1 = RP.pooled(_ck(model=r0, bodies=4), ()), RP.pooled(_ck(model=r1, bodies=4), ())
    assert a0["body_action_sensitivity"] != pytest.approx(a1["body_action_sensitivity"], rel=1e-6)
    assert r0["variants"]["2"]["bodies"] == [3]


def test_a_body_is_counted_only_where_its_sense_was_real(lite):
    m, ring = lite
    part = dict(ring)
    v = ring["valid"].clone()
    v[1, :, ring["mods"].index("vision")] = False
    part["valid"] = v
    r = RP.model_readings(SimpleNamespace(model=m), part, n_slots=6)
    assert r["per_body"][1]["eye_rows"] == 0 and r["per_body"][1]["body_rows"] > 0
    assert r["per_body"][0]["eye_rows"] > 0


def test_the_gaze_copy_without_its_centre_leaves_the_eye_reading_unmeasured(lite):
    m, ring = lite
    w0 = m.efference.centre_weight
    m.efference.centre_weight = 0.0
    try:
        r = RP.model_readings(SimpleNamespace(model=m), ring, n_slots=3)
    finally:
        m.efference.centre_weight = w0
    assert r["gaze_copy_active"] is False
    assert RP.pooled(_ck(model=r, bodies=4), ())["eye_prediction"] is None


# ==================================================================================== the stats records
def _run(saved: Path, name: str, launch_step: int, t0: float, steps, extra=None):
    d = saved / "experiments" / name
    d.mkdir(parents=True)
    rows = [dict(type="launch", timestamp=t0, step=launch_step)]
    rows += [dict(type="mimo", step=s, timestamp=t0 + i, **(extra or {})) for i, s in enumerate(steps)]
    (d / "replay_training_stats.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return d


def test_a_later_run_cuts_the_abandoned_future_of_an_earlier_one(tmp_path):
    saved = tmp_path / "Saved" / "mimo_bridge"
    _run(saved, RP.LINEAGE + "_segA", 1000, 100.0, range(1100, 2100, 100), dict(src="A"))
    _run(saved, RP.LINEAGE + "_segB", 1500, 200.0, range(1600, 2600, 100), dict(src="B"))
    _run(saved, "H99_other_lineage", 0, 50.0, range(100, 3000, 100))
    recs = RP.stats_records(saved, 1000, 2500)
    steps = [r["step"] for r in recs]
    assert steps == list(range(1100, 2600, 100)), "one record per step: A up to 1,500, then B"
    assert all(r["src"] == ("A" if r["step"] <= 1500 else "B") for r in recs)
    assert RP.stats_records(saved, 2000, 2200)[0]["src"] == "B"


def test_the_live_gauges_are_pooled_by_their_own_counts():
    recs = [dict(step=100, efference=dict(gaze_copy=dict(vision_err_with_copy=0.9, vision_err_copy_shuffled=1.0, events=100),
                                          gaze_deg_err=[1, 2, 3.0], gaze_deg_do_nothing=[2, 4, 6.0],
                                          gaze_deg_shuffled=[2, 4, 4.0], body_cm_err_latch=1.0, body_cm_shuffled_latch=2.0),
                 prediction_audit=dict(modalities=dict(
                     vision=dict(transitions=800, prediction_mse=0.02, persistence_mse=0.01),
                     proprioception=dict(sensitivity=0.02, shuffled_transitions=40, matched_prediction_mse=0.5,
                                         shuffled_action_mse=1.0))),
                 token_temporal_std=dict(proprioception=0.1, vestibular=0.0),
                 token_temporal_std_bodies=dict(proprioception=[0.1, 0.4], vestibular=[0.0, 0.0]),
                 hand_control=dict(joint_error_deg=10.0, body_observations=800),
                 synergy=dict(following_by_author=dict(planner=dict(part_orders=100, closer=60, chance_count=30.0)))),
            dict(step=200, efference=dict(gaze_copy=dict(vision_err_with_copy=0.5, vision_err_copy_shuffled=1.0, events=300),
                                          gaze_deg_err=[1, 2, 1.0], gaze_deg_do_nothing=[2, 4, 2.0],
                                          gaze_deg_shuffled=[2, 4, 4.0]),
                 prediction_audit=dict(modalities=dict(
                     vision=dict(transitions=800, prediction_mse=0.04, persistence_mse=0.01),
                     proprioception=dict(sensitivity=0.04, shuffled_transitions=40, matched_prediction_mse=1.0,
                                         shuffled_action_mse=1.0))),
                 token_temporal_std=dict(proprioception=0.3, vestibular=0.0),
                 token_temporal_std_bodies=dict(proprioception=[0.3, 0.2], vestibular=[0.0, 0.0]),
                 hand_control=dict(joint_error_deg=20.0, body_observations=200))]
    r = RP.records_readings(recs)
    assert r["eye_view_share_live"] == pytest.approx(1 - (0.9 * 100 + 0.5 * 300) / (1.0 * 400))
    assert r["eye_aim_150ms_share_live"] == pytest.approx(1 - 4.0 / 8.0)
    assert r["eye_aim_150ms_vs_shuffled_live"] == pytest.approx(1 - 4.0 / 8.0)
    assert r["vision_error_vs_unchanged_live"] == pytest.approx(0.06 / 0.02)
    assert r["body_sensitivity_live"] == pytest.approx(0.03)
    assert r["body_improvement_over_shuffled_live"] == pytest.approx(1 - 1.5 / 2.0)
    assert r["efference_body_vs_shuffled_live"] == pytest.approx(0.5)
    assert r["proprioception_spread_body0_live"] == pytest.approx(0.2)
    assert r["proprioception_spread_bodies_live"] == pytest.approx([0.2, 0.3])
    assert r["hand_joint_error_deg_live"] == pytest.approx((10 * 800 + 20 * 200) / 1000)
    assert r["following_closer_per_100_live"] == 60 and r["following_chance_per_100_live"] == 30


# ========================================================================================== the observer
def _obs(acts, exposure_steps=3000, bodies=range(8), baseline_occasions=100, runs=("runA",)):
    """A synthetic observer state: `acts` = [(cap, sub, step, body)], every body exposed from step 0 for
    `exposure_steps` steps in bins of 100 (all authored spontaneously), babble 100 occasions of every row in
    20,000 body-steps."""
    from mimo_bridge.tools.observer import store as ST
    cat = dict(capabilities=[dict(id="G01", name="Fixation", observable=True), dict(id="H05", name="Finger moves", observable=True),
                             dict(id="P11", name="Climb", observable=False)])
    store = ST.LineageStore(None, {})
    for cap, sub, step, body in acts:
        store.dets.add(dict(cap=cap, sub=sub, step0=step, step1=step + 2, body=body, pid=1, run=runs[0],
                            provenance="spontaneous", evidence={}))
    store.dets.freeze()
    bins = {str(b): dict(steps=100, by={"spontaneous": 100}, abs=100, abs_by={"spontaneous": 100})
            for b in range(exposure_steps // 100)}
    exposures = {runs[0]: {str(body): {"1": bins} for body in bodies}}
    return dict(cat=cat, store=store, live=np.ones(len(store.dets), dtype=bool), exposures=exposures,
                cuts_by_run={runs[0]: {}}, line={},
                baseline=dict(steps=20000, caps={"G01": dict(occasions=baseline_occasions, count=150, steps=20000),
                                                 "H05": dict(occasions=baseline_occasions, count=150, steps=20000)}))


def test_occasions_are_counted_per_body_and_part_with_the_one_second_gap():
    acts = [("H05", "L:index", 100, 0), ("H05", "L:index", 110, 0),       # one occasion (10 steps apart)
            ("H05", "L:index", 140, 0),                                     # a new one (30 steps after 110)
            ("H05", "R:index", 105, 0),                                     # another part: its own occasion
            ("H05", "L:index", 100, 1),                                     # another body: its own occasion
            ("G01", "ball", 200, 2)]
    w = RP.observer_window(_obs(acts), 0, 3000)
    assert w["rows"]["H05"]["acts"] == 5 and w["rows"]["H05"]["occasions"] == 4
    assert w["body_steps"] == 8 * 3000
    assert w["rows"]["H05"]["per_1000"] == pytest.approx(1000 * 4 / 24000)
    assert w["rows"]["H05"]["babble_per_1000"] == pytest.approx(1000 * 100 / 20000)
    assert w["rows"]["H05"]["vs_babble"] == pytest.approx((4 / 24000) / (100 / 20000))
    assert "P11" not in w["rows"], "a capability this world cannot show is not counted"
    wo = RP.observer_window(_obs(acts), 0, 3000, exclude=[2])
    assert wo["rows"]["G01"]["occasions"] == 0 and wo["body_steps"] == 7 * 3000


def test_the_window_counts_only_its_own_steps_rounded_to_the_exposure_bins():
    acts = [("G01", "ball", 1050, 0), ("G01", "ball", 2990, 0), ("G01", "ball", 3050, 0)]
    w = RP.observer_window(_obs(acts, exposure_steps=4000), 1025, 3025)
    assert (w["lo"], w["hi"]) == (1000, 3000)
    assert w["rows"]["G01"]["occasions"] == 2 and w["body_steps"] == 8 * 2000


def test_an_abandoned_future_counts_no_body_steps():
    o = _obs([], exposure_steps=3000)
    o["cuts_by_run"] = {"runA": {"0": {"1": 1000}}}
    o["line"] = {"runA": 2000}
    w = RP.observer_window(o, 0, 3000)
    assert w["body_steps_per_body"]["0"] == 1000 and w["body_steps_per_body"]["1"] == 2000


def test_a_row_that_became_rarer_is_named_and_an_unchanged_one_is_not():
    before = [("H05", "L:index", s, b) for b in range(8) for s in range(0, 3000, 30)]    # 100 per body
    before += [("G01", "ball", s, b) for b in range(8) for s in range(0, 3000, 300)]     # 10 per body
    after = [("H05", "L:index", 3000 + s, b) for b in range(8) for s in range(0, 3000, 300)]
    after += [("G01", "ball", 3000 + s, b) for b in range(8) for s in range(0, 3000, 300)]
    o = _obs(before + after, exposure_steps=6000)
    ch = RP.observer_change(RP.observer_window(o, 0, 3000), RP.observer_window(o, 3000, 6000))
    assert ch["fell"] == ["H05"] and ch["rose"] == [] and ch["tested"] == 2


def _windows(counts_by_row, exposure=21000.0):
    """Observer windows in observer_window's layout from occasion counts: {row: [occasions in each window]}."""
    n = len(next(iter(counts_by_row.values())))
    return [dict(lo=3000 * i, hi=3000 * (i + 1), body_steps=exposure,
                 rows={c: dict(name=c, occasions=int(ks[i]), per_1000=1000.0 * int(ks[i]) / exposure,
                               babble_per_1000=None, vs_babble=None) for c, ks in counts_by_row.items()})
            for i in range(n)]


def test_a_row_falls_only_beyond_its_own_measured_no_change_scatter():
    """Fails on the pack's first observer rule (beyond counting chance AND beyond the row's largest change between
    two no-change windows), which flagged a fallen row in 16 of 40 no-change comparisons on H84's windows: here 20
    rows whose rates wander between windows (ln sd 0.6 around 20 occasions, as repeat contact with the same object
    read 24, 5, 10, 37, 19), each window held out in turn and judged against each other one in both directions."""
    rng = np.random.default_rng(250925)
    rows = {"R%02d" % i: rng.poisson(20.0 * np.exp(rng.normal(0.0, 0.6, 5))) for i in range(20)}
    ws = _windows(rows)
    comps, fallen = 0, 0
    for i, held in enumerate(ws):
        lines = RP.observer_row_lines([w for j, w in enumerate(ws) if j != i])
        for j, w in enumerate(ws):
            if j == i:
                continue
            for a, b in ((w, held), (held, w)):
                comps += 1
                fallen += bool(RP.observer_change(a, b, row_lines=lines)["fell"])
    assert comps == 40
    assert fallen <= 2, "%d of 40 no-change comparisons read a fallen row" % fallen
    loo = RP.observer_leave_one_out(ws)
    assert loo["comparisons"] == 40 and loo["with_a_fallen_row"] == fallen and loo["line"] == max(loo["fallen_counts"])


def test_a_real_fall_beyond_the_rows_scatter_is_named_and_a_wandering_row_is_not():
    steady = {"H05": [200, 210, 190, 205, 195], "R02": [24, 5, 10, 37, 19]}
    lines = RP.observer_row_lines(_windows(steady))
    assert lines["H05"]["extra_var"] == pytest.approx(0.0), "counting chance explains a steady row's spread"
    assert lines["R02"]["extra_var"] > 0.3, "a wandering row carries its own extra scatter"
    before, after = _windows({"H05": [200], "R02": [24]}), _windows({"H05": [60], "R02": [5]})
    ch = RP.observer_change(before[0], after[0], row_lines=lines)
    assert ch["fell"] == ["H05"], "a steady row that fell to under a third fell"
    assert ch["rows"]["R02"]["fell"] is False, "24 -> 5 is inside R02's own no-change scatter"
    assert RP.observer_change(before[0], after[0], row_lines={})["fell"] == [], "a row without a scatter is not judged"


# ================================================================================ pooling and verdicts
def _ck(model=None, hands=None, bodies=None, spread=None):
    n = bodies or (len(hands) if hands else 8)
    hands = hands or [dict(orders=1, fingers=dict(part_orders=10, closer=6, chance=3.0, asked_deg=100.0, left_deg=60.0, moved_deg=50.0),
                           wrists=dict(part_orders=2, closer=1, chance=1.0, asked_deg=20.0, left_deg=10.0, moved_deg=10.0))
                      for _ in range(n)]
    spread = spread or {s: [dict(windows=[0.3], ends_back=[0], mean=0.3) for _ in range(n)]
                        for s in ("proprioception", "vestibular")}
    return dict(bodies=n, model=model, spread=spread, hands=hands)


def test_body_2_is_left_out_of_every_pooled_reading():
    spread = {"proprioception": [dict(windows=[], mean=v) for v in (0.4, 0.4, 0.05, 0.4)],
              "vestibular": [dict(windows=[], mean=0.0) for _ in range(4)]}
    hands = [dict(orders=1, fingers=dict(part_orders=10, closer=10, chance=3.0, asked_deg=100.0, left_deg=0.0, moved_deg=1.0),
                  wrists=dict(part_orders=0, closer=0, chance=0.0, asked_deg=0.0, left_deg=0.0, moved_deg=0.0))
             for _ in range(4)]
    hands[2]["fingers"].update(closer=0, left_deg=100.0)
    ck = _ck(hands=hands, spread=spread)
    all8, wo = RP.pooled(ck), RP.pooled(ck, [2])
    assert all8["proprioception_variation"] == pytest.approx((0.4 * 3 + 0.05) / 4)
    assert wo["proprioception_variation"] == pytest.approx(0.4)
    assert all8["proprioception_variation_deaf_bodies"] == [2] and wo["proprioception_variation_deaf_bodies"] == []
    assert all8["vestibular_variation_deaf_bodies"] == [0, 1, 2, 3], "0% is under the balance line of 1.6%"
    assert all8["hand_following"] == pytest.approx(1 - 100 / 400) and wo["hand_following"] == 1.0
    assert [r["body"] for r in RP.per_body_table(ck)] == [0, 1, 2, 3]
    assert RP.per_body_table(ck)[2]["hand_following"] == 0.0


def test_the_verdict_reads_the_worse_direction_beyond_the_no_change_line():
    assert RP.verdict("hand_following", 0.50, 0.40, 0.05)["verdict"] == "REGRESSED"
    assert RP.verdict("hand_following", 0.50, 0.47, 0.05)["verdict"] == "within noise"
    assert RP.verdict("hand_following", 0.50, 0.60, 0.05)["verdict"] == "IMPROVED"
    assert RP.verdict("hand_following", 0.50, 0.60, None)["verdict"].startswith("no verdict")
    assert RP.verdict("hand_following", None, 0.60, 0.05)["verdict"].startswith("unmeasured")


def _fake_compare(monkeypatch, tmp_path, lines):
    cks = {"A": _ck(), "B": _ck(hands=[dict(orders=1, fingers=dict(part_orders=10, closer=2, chance=3.0, asked_deg=100.0,
                                                                   left_deg=90.0, moved_deg=10.0),
                                           wrists=dict(part_orders=2, closer=1, chance=1.0, asked_deg=20.0, left_deg=10.0,
                                                       moved_deg=10.0)) for _ in range(8)])}
    monkeypatch.setattr(RP, "checkpoint_readings", lambda ck, **k: cks[ck])
    monkeypatch.setattr(RP, "stats_records", lambda saved, lo, hi: [])
    acts = [("H05", "L:index", s, b) for b in range(8) for s in range(0, 3000, 30)]
    acts += [("H05", "L:index", 3000 + s, b) for b in range(8) for s in range(0, 3000, 300)]
    o = _obs(acts, exposure_steps=6000)
    o.update(store_generation=1, baseline_path="x", detector_version="v", rules_version="r")
    monkeypatch.setattr(RP, "observer_state", lambda saved: o)
    before = dict(kind="checkpoint", lo=0, hi=3000, checkpoint="A", checkpoint_step=3000)
    after = dict(kind="checkpoint", lo=3000, hi=6000, checkpoint="B", checkpoint_step=6000)
    return RP.compare(before, after, saved=tmp_path, lines=lines)


def _repeat_lines(sd=0.01):
    """A noise file's lines: every ring reading's repeat (sd 0.01 at the fakes' own sample sizes: 80 or 70 finger
    orders, one 50-s window), the observer rows' scatter and the count line 0."""
    n = {"hand_following": (80, 70), "proprioception_variation": (1, 1), "vestibular_variation": (1, 1),
         "eye_prediction": (1, 1), "body_action_sensitivity": (1, 1)}
    rows = {"H05": dict(extra_var=0.0, windows=5), "G01": dict(extra_var=0.0, windows=5)}
    return dict(repeat={k: {v: dict(sd=sd, n=m[i], dof=19) for i, v in enumerate(("all", "without_body_2"))}
                        for k, m in n.items()},
                observer_counts={"all": 0, "without_body_2": 0}, observer_rows={"all": rows, "without_body_2": rows})


def test_compare_judges_every_headline_without_body_2_and_names_the_rows_that_fell(monkeypatch, tmp_path):
    lines = _repeat_lines()
    rep = _fake_compare(monkeypatch, tmp_path, lines)
    hf = rep["headlines"]["hand_following"]["without_body_2"]
    assert hf["line"] == pytest.approx(RP.t_quantile(19) * math.sqrt(2) * 0.01), "t x sqrt(sd_b^2 + sd_a^2)"
    h = rep["headlines"]
    assert set(h) == set(RP.HEADLINES)
    assert h["hand_following"]["without_body_2"]["verdict"] == "REGRESSED"
    assert h["proprioception_variation"]["without_body_2"]["verdict"] == "within noise"
    assert h["eye_prediction"]["without_body_2"]["verdict"].startswith("unmeasured"), "no model part in this fake"
    assert h["observer_counts"]["without_body_2"]["value"] == 1
    assert h["observer_counts"]["without_body_2"]["fell"][0][0] == "H05"
    assert h["observer_counts"]["without_body_2"]["verdict"] == "REGRESSED"
    text = RP.table(rep)
    for name in RP.HEADLINES:
        assert ("observer behaviour counts" if name == "observer_counts" else name.replace("_", " ")) in text
    assert "body 3" in text and "numbered from 0" in text
    low = text.lower()
    for banned in ("honest", "truthful", "barely", "slightly", "significantly", "substantially", "roughly"):
        assert banned not in low, banned


def test_without_a_measured_line_no_verdict_is_given(monkeypatch, tmp_path):
    rep = _fake_compare(monkeypatch, tmp_path, {})
    for name in RP.HEADLINES:
        v = rep["headlines"][name]["without_body_2"]["verdict"]
        assert v.startswith("no verdict") or v.startswith("unmeasured"), (name, v)


def _spread_ck(window_values, bodies=8):
    """A checkpoint whose proprioception token spread read `window_values` in its six 50-s windows, every body."""
    per = [dict(windows=list(window_values), ends_back=[1000 * (len(window_values) - 1 - i) for i in range(len(window_values))],
                mean=float(np.mean(window_values))) for _ in range(bodies)]
    return dict(_ck(spread={"proprioception": per, "vestibular": [dict(windows=[0.0] * 6, ends_back=list(range(6)), mean=0.0)
                                                                   for _ in range(bodies)]}), step=415025)


def test_the_noise_is_a_repeat_on_one_checkpoint(monkeypatch, tmp_path):
    """Fails on the pack's first noise rule, whose proprioception line (2.56 points) was the spread BETWEEN six
    checkpoints 15,800 updates apart and equalled the fitted training trend over them: the noise is now the reading's
    scatter inside one checkpoint (its six 50-s windows), and a compare scales it to the side's own sample."""
    vals = [0.21, 0.23, 0.22, 0.24, 0.20, 0.22]
    monkeypatch.setattr(RP, "checkpoint_readings", lambda c, **k: _spread_ck(vals))
    monkeypatch.setattr(RP, "stats_records", lambda saved, lo, hi: [])
    o = _obs([("G01", "ball", s, b) for b in range(8) for s in range(0, 15000, 300)], exposure_steps=15000)
    monkeypatch.setattr(RP, "observer_state", lambda saved: o)
    r = RP.noise("c", 0, 15000, saved=tmp_path, window=3000)
    ref = r["lines"]["repeat"]["proprioception_variation"]["without_body_2"]
    assert ref["sd"] == pytest.approx(np.std(vals, ddof=1) / math.sqrt(6)) and ref["n"] == 6 and ref["dof"] == 5
    assert ref["same_size_line"] == pytest.approx(RP.t_quantile(5) * math.sqrt(2) * ref["sd"])
    assert r["lines"]["repeat"]["vestibular_variation"]["all"]["sd"] == 0.0
    assert r["lines"]["observer_counts"]["all"] == 0, "steady occasions: no row fell in the leave-one-out comparisons"
    assert r["observer"]["all"]["leave_one_out"]["comparisons"] == 5 * 4 * 2
    assert r["lines"]["observer_rows"]["all"]["G01"]["occasions"] == [80] * 5
    assert r["lines"]["observer_rows"]["all"]["G01"]["extra_var"] == pytest.approx(0.0)
    assert r["version"] == RP.PACK_VERSION


def test_the_sampling_sd_is_the_batches_spread_over_the_root_of_their_number():
    """Readings 1-2 on BATCHES contiguous groups of window ends: the sd of the whole sample's reading."""
    S = 40
    r = np.random.default_rng(3)
    shift = r.uniform(0.0, 2.0, S)
    rows = [dict(eye_err_own=1.0, eye_err_still=1.1, eye_err_other=1.2, eye_err_unchanged=2.0, eye_rows=1.0,
                 body_shift_sq=float(x), body_change_sq=100.0, body_err_own=1.0, body_err_other=1.0,
                 body_err_unchanged=1.0, body_rows=1.0) for x in shift]
    m = dict(per_body=[{k: float(sum(x[k] for x in rows)) for k in RP.MODEL_KEYS}],
             per_slot={k: [[x[k]] for x in rows] for k in RP.MODEL_KEYS}, gaze_copy_active=True, variants={})
    p = RP.pooled(_ck(model=m, bodies=1), ())
    parts = np.array_split(np.arange(S), RP.BATCHES)
    vals = [math.sqrt(shift[q].sum() / (100.0 * len(q))) for q in parts]
    assert p["body_action_sensitivity_sd"] == pytest.approx(np.std(vals, ddof=1) / math.sqrt(RP.BATCHES))
    assert p["body_action_sensitivity_dof"] == RP.BATCHES - 1 and p["body_action_sensitivity_n"] == S


def test_the_model_sample_is_big_enough_for_the_change_it_must_show():
    """Fails on the pack's first sample (24 window ends per body): with one 24-end sample's sd measured at 1.96
    points (ten samples at 415,025 spread 6.03), a same-size compare's line for body action sensitivity was 7.3
    points against the 1-point total loss it must show; the arithmetic of MODEL_SLOTS puts it under 1."""
    sd24 = RP.SAMPLE_SD_24["body_action_sensitivity"]
    assert sd24 == pytest.approx(0.0196, abs=1e-4)
    need = RP.slots_needed(sd24, RP.DETECT["body_action_sensitivity"])
    assert RP.MODEL_SLOTS >= need
    line = RP.t_quantile(RP.BATCHES - 1) * math.sqrt(2) * sd24 * math.sqrt(24 / RP.MODEL_SLOTS)
    assert line < RP.DETECT["body_action_sensitivity"]
    assert RP.MODEL_SLOTS <= 6000 - 129 - 1, "the ring holds 5,870 usable window ends per body"


def test_a_line_combines_both_sides_sampling_noise():
    b = dict(hand_following=0.3, hand_following_sd=0.004, hand_following_n=1000, hand_following_dof=19)
    a = dict(hand_following=0.3, hand_following_sd=0.003, hand_following_n=250, hand_following_dof=9)
    ln = RP.line_of("hand_following", "without_body_2", b, a, {})
    assert ln["dof"] == 9 and ln["line"] == pytest.approx(RP.t_quantile(9) * math.sqrt(0.004 ** 2 + 0.003 ** 2))
    # the noise file's repeat, scaled to a side's smaller sample, wins when it is larger than the side's own
    ref = dict(repeat=dict(hand_following=dict(without_body_2=dict(sd=0.004, n=1000, dof=19))))
    ln2 = RP.line_of("hand_following", "without_body_2", b, a, ref)
    assert ln2["sd_after"] == pytest.approx(0.004 * math.sqrt(1000 / 250))
    assert RP.line_of("hand_following", "all", {}, a, {})["line"] is None, "a side without any sd gives no line"


def test_a_noise_file_of_another_pack_version_gives_no_lines(tmp_path):
    old = tmp_path / "old.json"
    old.write_text(json.dumps(dict(version="regression_pack_v1", lines={"hand_following": {"all": 0.01}})), encoding="utf-8")
    assert RP.load_lines(old) == {}
    new = tmp_path / "new.json"
    new.write_text(json.dumps(dict(version=RP.PACK_VERSION, lines={"x": 1})), encoding="utf-8")
    assert RP.load_lines(new) == {"x": 1}


def test_a_side_is_a_checkpoint_a_bare_step_or_a_step_range(tmp_path, monkeypatch):
    monkeypatch.setattr(RP, "ARCHIVE", tmp_path / "no_archive")      # never the real D: archive in a test
    saved = tmp_path / "Saved" / "mimo_bridge"
    ck = saved / "experiments" / (RP.LINEAGE + "_seg1") / "checkpoints" / "ckpt_step_5000"
    ck.mkdir(parents=True)
    for f in ("model.safetensors", "episode_state.json"):
        (ck / f).write_text("x", encoding="utf-8")
    s = RP.parse_side(str(ck), 3000, saved)
    assert (s["lo"], s["hi"], s["checkpoint_step"]) == (2000, 5000, 5000)
    r = RP.parse_side("1000-6000", 3000, saved)
    assert (r["lo"], r["hi"]) == (1000, 6000)
    assert r["checkpoint"] is not None and r["checkpoint_step"] == 5000
    assert RP.parse_side("5000", 3000, saved)["checkpoint"] == str(ck)
    with pytest.raises(FileNotFoundError):
        RP.parse_side("4000", 3000, saved)
    with pytest.raises(ValueError):
        RP.parse_side("6000-1000", 3000, saved)


# ================================================== review fix 4a: the after side holds only post-switch updates
def test_the_after_side_starts_at_the_switch_and_reads_only_the_moments_after_it(tmp_path, monkeypatch):
    """Fails on the pack's first sides: a checkpoint's window was the 3,000 steps ending at it and its ring readings
    used all 6,000 moments, so after a 1,742-update watch 71% of what the after side read came before the switch."""
    monkeypatch.setattr(RP, "ARCHIVE", tmp_path / "no_archive")
    saved = tmp_path / "Saved" / "mimo_bridge"
    ck = saved / "experiments" / (RP.LINEAGE + "_seg1") / "checkpoints" / "ckpt_step_6742"
    ck.mkdir(parents=True)
    for f in ("model.safetensors", "episode_state.json"):
        (ck / f).write_text("x", encoding="utf-8")
    s = RP.parse_side("6742", 3000, saved, switch_step=5000)
    assert (s["lo"], s["hi"], s["newest"], s["switch_step"]) == (5000, 6742, 1742, 5000)
    with pytest.raises(ValueError):
        RP.parse_side("6742", 3000, saved, switch_step=7000)
    assert RP.parse_side("6742", 3000, saved)["newest"] is None


def test_the_before_and_after_windows_may_not_overlap(tmp_path):
    before = dict(kind="checkpoint", lo=2000, hi=5000, checkpoint=None, checkpoint_step=5000)
    after = dict(kind="checkpoint", lo=3742, hi=6742, checkpoint=None, checkpoint_step=6742)
    with pytest.raises(ValueError, match="overlap"):
        RP.compare(before, after, saved=tmp_path, lines={})


def test_every_ring_reading_after_a_switch_reads_only_the_newest_moments():
    ring = _ring(B=2, N=6000)
    k = MODS.index("proprioception")
    g = torch.Generator().manual_seed(0)
    ring["tokens"][:, :4000, k] = torch.randn(2, 4000, 8, generator=g).half()   # before the switch: varying
    ring["tokens"][:, 4000:, k] = 0.5                                             # after it: one constant token
    whole = RP.token_spreads(ring)["proprioception"][0]
    after = RP.token_spreads(ring, newest=2000)["proprioception"][0]
    assert len(whole["windows"]) == 6 and whole["mean"] > 0.3
    assert len(after["windows"]) == 2 and after["mean"] == 0.0 and after["ends_back"] == [1000, 0]
    # every model window lies inside the newest moments, its context included
    backs = RP.model_slots(ring, 128, 1280, newest=1742)
    assert backs and max(backs) + 129 <= 1742 and min(backs) >= 1
    assert RP.model_slots(ring, 128, 1280, newest=100) == [], "a watch shorter than one window reads nothing"


def test_hand_following_after_a_switch_counts_only_orders_begun_after_it():
    ring = _hand_ring(arrive=True, tenures=6)
    N = int(ring["count"])
    whole = RP.hand_following(ring, TRUNK)[0]
    late = RP.hand_following(ring, TRUNK, newest=N - 20)[0]
    assert whole["orders"] == 6 and late["orders"] == 3, "orders begun at moments 20, 25 and 30 of 41"
    # the batch sums add up to the whole
    for part in ("fingers", "wrists"):
        for key in ("part_orders", "asked_deg", "left_deg"):
            assert sum(b[key] for b in whole["batches"][part]) == pytest.approx(whole[part][key], rel=1e-9)


# ============================================================== review fix 6: the cache honours --slots
def test_a_cache_written_for_another_sample_size_is_read_again(monkeypatch, tmp_path):
    """Fails on the pack's first cache, keyed on the checkpoint alone: measuring again with more moments silently
    reused the 24-moment sums."""
    ck = tmp_path / "run" / "checkpoints" / "ckpt_step_5000"
    ck.mkdir(parents=True)
    cache = tmp_path / "cache"
    cache.mkdir()
    for name in ("run_ckpt_step_5000.json", RP.cache_name(ck, 24, None)):
        (cache / name).write_text(json.dumps(dict(version=RP.PACK_VERSION, newest=None, model=dict(n_slots=24))),
                                  encoding="utf-8")

    class Loaded(Exception):
        pass

    def load(_ck):
        raise Loaded()
    monkeypatch.setattr(RP, "load_ring", load)
    assert RP.checkpoint_readings(ck, cache_dir=cache, n_slots=24)["model"]["n_slots"] == 24
    with pytest.raises(Loaded):
        RP.checkpoint_readings(ck, cache_dir=cache, n_slots=48)
    with pytest.raises(Loaded):
        RP.checkpoint_readings(ck, cache_dir=cache, n_slots=24, newest=1742)
    assert RP.PACK_VERSION == "regression_pack_v2"
