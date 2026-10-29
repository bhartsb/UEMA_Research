# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The wellness channels (docs/gain_of_function_v3/DESIGN.md D6-D9, lem/model/wellness_channels.py):
controllability against known effects, stillness never counted, its shuffled-order control; learning
progress cancelling a world that gets easier and reading an improving model as positive; the
normaliser keeping a 100x-wider channel from deciding alone, the weights realising the priority
order, the integrity gate excluding a painful moment, the state surviving a save; the consequence
count reading only world contact under the exclude rule."""
import math

import pytest
import torch

from lem.model import wellness_channels as WC
from mimo_bridge.lem_model import consequence_kinds


# ---------------------------------------------------------------------------- agreement
def test_agreement_is_cosine_times_size_ratio():
    e = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
    a = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    g = WC.agreement(e, a)
    assert g.tolist() == pytest.approx([1.0, -1.0, 0.0, 0.5, 0.0])


def _tracker(B=2, floors=None):
    H = 4          # handles 0-2 body, 3 = gaze
    return WC.ControllabilityTracker(H, gaze_idx=3, hold=2,
                                     floors=floors or {"command": 0.5, "efference": 0.5},
                                     ema_orders=1)


def _pose(B, H=4):
    a = torch.zeros(B, H, 10)
    a[..., 6] = 1.0                 # identity quaternion (x y z w)
    return a


def _qz(deg):
    """x y z w quaternion of a turn of `deg` degrees about z."""
    h = math.radians(deg) / 2.0
    return torch.tensor([0.0, 0.0, math.sin(h), math.cos(h)])


def _qx(deg):
    h = math.radians(deg) / 2.0
    return torch.tensor([math.sin(h), 0.0, 0.0, math.cos(h)])


def _order(B, H=4):
    o = torch.zeros(B, H, 9)
    o[..., 6] = 1.0
    o[..., 8] = 0.5
    return o


def _run_order(tr, begin, end, order, author="planner", start=10, B=2, eff=None, next_id=2):
    """Order 1 is in force for steps start+1 .. start+hold; order `next_id` appears after."""
    tr.step(start, _order(B), 0, begin, "reflex", None)
    tr.step(start + 1, order, 1, begin, author, None)          # the order appears: begin = prev
    tr.step(start + 2, order, 1, end, author, None)
    return tr.step(start + 3, order, next_id, end, author, eff)   # hold 2 moments -> graded


def test_the_command_source_grades_a_followed_order_near_one_and_stillness_not_at_all():
    B = 2
    tr = _tracker(B)
    begin = _pose(B)
    # body 0 is told to turn handle 0 by 40 degrees about z (3.6 tenure spreads) and does
    # exactly that; body 1 is told to turn it 2 degrees (0.18 spreads) - under the 0.5 floor,
    # so it is never counted
    o = _order(B)
    o[0, 0, 3:7] = _qz(40.0)
    o[1, 0, 3:7] = _qz(2.0)
    end = begin.clone()
    end[0, 0, 3:7] = _qz(40.0)
    end[1, 0, 3:7] = _qz(2.0)
    rec = _run_order(tr, begin, end, o)
    assert rec is not None and rec["source"] == "command"
    assert rec["agreement"][0] == pytest.approx(1.0, abs=1e-4)
    assert rec["agreement"][1] is None, "an order under the stillness floor must not be counted"
    v = tr.value(B, "cpu")
    assert v[0].item() == pytest.approx(1.0, abs=1e-4) and math.isnan(v[1].item())
    s = tr.summary(B)
    assert s["measured_share"] == 0.5 and s["agreement_mean"] == pytest.approx(1.0, abs=1e-4)
    assert s["by_author"] == {"planner": pytest.approx(1.0, abs=1e-4)}


def test_a_half_followed_order_reads_half_and_a_named_position_is_not_the_command():
    B = 1
    tr = _tracker(B)
    begin = _pose(B)
    o = _order(B)
    o[0, 0, 3:7] = _qz(40.0)
    o[0, 0, 0:3] = 30.0              # a position far off: the decoder clips it; not counted
    end = begin.clone()
    end[0, 0, 3:7] = _qz(20.0)       # turned half as far, the right way
    rec = _run_order(tr, begin, end, o, B=B)
    assert rec["agreement"][0] == pytest.approx(0.5, abs=1e-3)


def test_the_shuffled_order_control_reads_low_when_each_body_did_its_own_order():
    B = 2
    tr = _tracker(B)
    begin = _pose(B)
    o = _order(B)
    o[0, 0, 3:7] = _qz(40.0)         # body 0: about z
    o[1, 0, 3:7] = _qx(40.0)         # body 1: about x
    end = begin.clone()
    end[0, 0, 3:7] = _qz(40.0)
    end[1, 0, 3:7] = _qx(40.0)
    _run_order(tr, begin, end, o)
    s = tr.summary(B)
    assert s["agreement_mean"] == pytest.approx(1.0, abs=1e-4)
    assert s["control_mean"] == pytest.approx(0.0, abs=1e-4),         "another body's order, taken from this body's pose, must not match this body's result"
    assert s["difference_mean"] == pytest.approx(1.0, abs=1e-4)


def test_the_efference_source_is_used_when_its_tenure_is_fresh():
    B = 2
    tr = _tracker(B)
    begin = _pose(B)
    o = _order(B)
    o[:, 0, 3:7] = _qz(40.0)
    end = begin.clone()
    end[:, 0, 3:7] = _qz(40.0)
    exp_deg = torch.zeros(B, 4, 3)
    exp_deg[:, 0, 2] = -40.0      # the head expected the opposite turn
    eff = {"step": 13, "order_began": 10, "expected_cm": torch.zeros(B, 4, 3),
           "expected_deg": exp_deg}
    rec = _run_order(tr, begin, end, o, eff=eff)
    assert rec["source"] == "efference"
    assert rec["agreement"][0] == pytest.approx(-1.0, abs=1e-4)
    s = tr.summary(B)
    assert s["command_agreement_mean"] == pytest.approx(1.0, abs=1e-4)


def test_with_the_heads_on_an_order_without_their_record_is_skipped_not_judged_by_command():
    B = 2
    tr = WC.ControllabilityTracker(4, gaze_idx=3, hold=2,
                                   floors={"command": 0.5, "efference": 0.5}, ema_orders=1,
                                   expect_efference=True)
    begin = _pose(B)
    o = _order(B)
    o[:, 0, 3:7] = _qz(40.0)
    end = begin.clone()
    end[:, 0, 3:7] = _qz(40.0)
    assert _run_order(tr, begin, end, o, eff=None) is None
    s = tr.summary(B)
    # measured_share is a share of ORDERS: the two orders the head left unjudged count, unmeasured
    assert s["efference_missing"] == 1 and s["orders"] == 2 and s["measured_share"] == 0.0
    assert s["source"] is None and s["agreement_mean"] is None


def test_an_order_of_the_wrong_length_or_across_a_reset_is_not_graded():
    B = 1
    tr = _tracker(B)
    p = _pose(B)
    o = _order(B)
    tr.step(0, o, 1, p, "planner", None)
    tr.step(1, o, 2, p, "planner", None)       # order 2 begins (pending from step 0)
    tr.forget_pending()                        # a reset: order 2 must never be graded
    assert tr.step(2, o, 3, p, "planner", None) is None
    assert tr.step(3, o, 4, p, "planner", None) is None      # order 4 begins, pending from 2
    tr.step(4, o, 4, p, "planner", None)
    tr.step(5, o, 4, p, "planner", None)
    assert tr.step(6, o, 5, p, "planner", None) is None      # ran 3 moments, not 2
    assert tr.summary(B)["tenure_mismatch"] == 1


def test_the_evidence_gate_opens_only_when_agreement_beats_its_control():
    B = 2
    tr = _tracker(B)
    ok, mean, line, why = tr.gate()
    assert not ok and mean is None
    # each body does ITS OWN order: agreement 1, the shuffled-order control 0 -> opens
    for k in range(WC.CONTROL_GATE_READINGS + 1):      # one reading per order
        begin = _pose(B)
        o = _order(B)
        o[0, 0, 3:7] = _qz(40.0 + k)
        o[1, 0, 3:7] = _qx(40.0 + k)
        end = begin.clone()
        end[0, 0, 3:7] = _qz(40.0 + k)
        end[1, 0, 3:7] = _qx(40.0 + k)
        _run_order(tr, begin, end, o, start=10 * (k + 1), next_id=2 + 10 * k)
    ok, mean, line, _ = tr.gate()
    assert ok and mean > line
    # both bodies turn about z whatever they are told: the result matches another body's
    # expectation as well as its own -> the gate shuts
    tr2 = _tracker(B)
    for k in range(WC.CONTROL_GATE_READINGS + 1):      # one reading per order
        begin = _pose(B)
        o = _order(B)
        o[:, 0, 3:7] = _qz(40.0)
        end = begin.clone()
        end[:, 0, 3:7] = _qz(10.0 + k)
        _run_order(tr2, begin, end, o, start=10 * (k + 1), next_id=2 + 10 * k)
    assert not tr2.gate()[0]
    s = tr2.summary(B)
    assert s["gate_open"] is False and "It is shut" in s["gate_why"]
    assert s["difference_mean"] == pytest.approx(0.0, abs=1e-6)


def test_controllability_state_round_trips():
    B = 2
    tr = _tracker(B)
    tr.ema_m = torch.tensor([0.3, 0.1])
    tr.ema_w = torch.tensor([0.5, 0.2])
    tr.gate_readings.extend([0.1, -0.2, 0.3])
    tr2 = _tracker(B)
    tr2.load_state_dict(tr.state_dict())
    assert torch.equal(tr2.value(B, "cpu"), tr.value(B, "cpu"))
    assert list(tr2.gate_readings) == [0.1, -0.2, 0.3]


# ---------------------------------------------------------------------------- learning progress
def test_the_ratio_cancels_a_world_that_gets_easier():
    lp = WC.LearningProgressTracker(fast=10, slow=100)
    # the model always makes 40% of doing-nothing's error; the world gets 10x easier at moment 300
    for t in range(330):
        k = 1.0 if t < 300 else 0.1
        lp.update("vision", torch.tensor([0.4 * k]), torch.tensor([1.0 * k]))
    pr = lp.progress("vision")
    assert abs(float(pr[0])) < 1e-4, "an easier world is not learning progress"
    # the raw-error version would have read the drop as large progress
    s = lp.state["vision"]
    raw_fast, raw_slow = float(s["mf"] / s["wf"]), float(s["ms"] / s["ws"])
    assert (raw_slow - raw_fast) / raw_slow > 0.5


def test_an_improving_model_reads_positive_and_a_worsening_one_negative():
    lp = WC.LearningProgressTracker(fast=10, slow=200)
    for t in range(400):
        r_good = max(0.1, 0.9 - 0.002 * t)          # the model's share of doing-nothing's error falls
        r_bad = min(0.9, 0.1 + 0.002 * t)
        lp.update("proprioception", torch.tensor([r_good, r_bad]), torch.tensor([1.0, 1.0]))
    pr = lp.progress("proprioception")
    assert float(pr[0]) > 0.2 and float(pr[1]) < -0.2
    ch = lp.channel(2)
    assert float(ch[0]) == pytest.approx(float(pr[0]))


def test_learning_progress_round_trips_and_reports():
    lp = WC.LearningProgressTracker(fast=5, slow=50)
    for t in range(30):
        lp.update("tactile", torch.tensor([0.5, 0.2]), torch.tensor([1.0, 1.0]))
        lp.tick_window(2)
    lp2 = WC.LearningProgressTracker(fast=5, slow=50)
    lp2.load_state_dict(lp.state_dict())
    assert torch.equal(lp2.progress("tactile"), lp.progress("tactile"))
    s = lp.summary(2)
    assert set(s["per_sense"]) == {"tactile"} and len(s["per_body"]) == 2
    assert s["ratio_fast"]["tactile"] == pytest.approx(0.35, abs=1e-4)


# ---------------------------------------------------------------------------- the normaliser
def _feed(norm, names, ch):
    for t in range(ch.shape[0]):
        for ci, n in enumerate(names):
            if n != WC.PAIN_ROW:
                norm.update(n, ch[t, ci])


def test_a_channel_with_100x_the_spread_does_not_dominate_after_normalisation():
    g = torch.Generator().manual_seed(0)
    T, B = 400, 4
    names = ["consequence", "prediction_error", WC.PAIN_ROW]
    ch = torch.zeros(T, 3, B)
    ch[:, 0] = torch.randn(T, B, generator=g) * 1.0          # wide channel
    ch[:, 1] = torch.randn(T, B, generator=g) * 0.01         # 100x narrower
    ch[:, 2] = 0.1                                            # pain, under any line
    norm = WC.ChannelNormalizer(horizon=100000)
    _feed(norm, names, ch)
    w = {"consequence": 0.5, "prediction_error": 0.5}
    floors = {"consequence": 1e-6, "prediction_error": 1e-6}
    ranked, contrib, gated, ag = WC.rank_values(ch, names, w, norm, floors, gate_line=0.9)
    sh = WC.RankShares()
    sh.add(contrib, ranked, gated, ag, names)
    vs = sh.summary()["variance_share"]
    assert abs(vs["consequence"] - 0.5) < 0.1 and abs(vs["prediction_error"] - 0.5) < 0.1
    # the raw sum, for contrast: the wide channel carries almost all of it
    raw = ch[:, 0] + ch[:, 1]
    share_raw = float(((ch[:, 0] - ch[:, 0].mean(0)) * (raw - raw.mean(0))).mean()
                      / ((raw - raw.mean(0)) ** 2).mean())
    assert share_raw > 0.99


def test_the_weights_realise_the_priority_order():
    g = torch.Generator().manual_seed(1)
    T, B = 300, 2
    names = [c for c in WC.CHANNEL_ORDER if WC.DEFAULT_WEIGHTS[c] > 0] + [WC.PAIN_ROW]
    C = len(names)
    ch = torch.randn(T, C, B, generator=g)
    ch[:, -1] = 0.0
    norm = WC.ChannelNormalizer(horizon=100000)
    _feed(norm, names, ch)
    floors = {n: 1e-6 for n in names}
    ranked, contrib, gated, ag = WC.rank_values(ch, names, WC.DEFAULT_WEIGHTS, norm, floors, 0.9)
    sh = WC.RankShares()
    sh.add(contrib, ranked, gated, ag, names)
    vs = sh.summary()["variance_share"]
    # independent unit-spread channels: each carries its weight squared over the sum of squares
    assert vs["integrity"] > vs["controllability"] > vs["learning_progress"] > vs["coherence"]
    assert vs["coherence"] > vs["agency"]
    tot = sum(WC.DEFAULT_WEIGHTS[n] ** 2 for n in names if n != WC.PAIN_ROW)
    assert vs["integrity"] == pytest.approx(4.0 / tot, abs=0.08)


def test_the_integrity_gate_excludes_a_painful_moment_however_good_it_is():
    names = ["consequence", WC.PAIN_ROW]
    ch = torch.zeros(3, 2, 1)
    ch[:, 0, 0] = torch.tensor([0.0, 3.0, 1.0])        # moment 1 has by far the most consequence
    ch[:, 1, 0] = torch.tensor([0.1, 0.95, 0.2])       # ... and pain over the line
    norm = WC.ChannelNormalizer(horizon=10)
    _feed(norm, names, ch)
    ranked, contrib, gated, ag = WC.rank_values(ch, names, {"consequence": 0.5}, norm,
                                                {"consequence": 0.1}, gate_line=0.9)
    assert gated[:, 0].tolist() == [False, True, False]
    assert int(ranked[:, 0].argmax()) == 2
    # every candidate over the line: the least painful is chosen
    ch2 = ch.clone()
    ch2[:, 1, 0] = torch.tensor([0.99, 0.95, 0.97])
    r2, _, g2, ag2 = WC.rank_values(ch2, names, {"consequence": 0.5}, norm, {"consequence": 0.1}, 0.9)
    assert bool(ag2[0]) and int(r2[:, 0].argmax()) == 1


def test_an_unmeasured_value_counts_as_the_centre():
    names = ["controllability", WC.PAIN_ROW]
    ch = torch.tensor([[[0.5], [0.0]], [[float("nan")], [0.0]]])
    norm = WC.ChannelNormalizer(horizon=10)
    norm.update("controllability", torch.tensor([0.0, 1.0]))
    ranked, contrib, _, _ = WC.rank_values(ch, names, {"controllability": 1.0}, norm,
                                           {"controllability": 0.1}, 0.9)
    assert float(contrib[1, 0, 0]) == 0.0 and torch.isfinite(ranked).all()


def test_the_normaliser_is_corrected_for_starting_at_zero_and_round_trips():
    norm = WC.ChannelNormalizer(horizon=2000)
    norm.update("capability", torch.tensor([0.6, 0.7]))
    c, s = norm.centre_scale("capability")
    assert c == pytest.approx(0.65) and s == pytest.approx(0.05)
    n2 = WC.ChannelNormalizer(horizon=2000)
    n2.load_state_dict(norm.state_dict())
    assert n2.centre_scale("capability") == norm.centre_scale("capability")
    st = WC.WellnessChannelState(True, None, WC.LearningProgressTracker())
    st.normalizer = norm
    st2 = WC.WellnessChannelState(True, None, WC.LearningProgressTracker())
    assert st2.load_state_dict(st.state_dict()) == ["normalizer", "learning_progress"]
    assert st2.normalizer.centre_scale("capability") == norm.centre_scale("capability")


def test_weights_parse_and_refuse_unknown_names():
    w = WC.parse_weights("integrity=3, efficiency=0.1")
    assert w["integrity"] == 3.0 and w["efficiency"] == 0.1 and w["consequence"] == 0.5
    with pytest.raises(ValueError):
        WC.parse_weights("integrity=2,wellbeing=1")


# ---------------------------------------------------------------------------- consequence
def test_the_exclude_rule_counts_world_contact_only():
    self_press = {"hand_pressing": True, "hand_pressing_world": False, "hand_pressing_self": True}
    world_press = {"hand_pressing": True, "hand_pressing_world": True, "hand_pressing_self": False}
    assert consequence_kinds(self_press) == 1.0 and consequence_kinds(self_press, world_only=True) == 0.0
    assert consequence_kinds(world_press, world_only=True) == 1.0
    # a hand touching the mobile still counts under both rules, the view too
    t = {"mobile_touched_hand": True, "hand_pressing": True, "hand_pressing_world": False,
         "gaze_on_ball": True}
    assert consequence_kinds(t) == 2.0 and consequence_kinds(t, world_only=True) == 2.0
    # a report without the split pays no hand press under the exclude rule
    assert consequence_kinds({"hand_pressing": True}, world_only=True) == 0.0


# ============================================================================ lane-b3-wellness
# the b3 review's findings (docs/gain_of_function_v3/reviews/b3-wellness.md), one test each
def _q_axis(axis, deg):
    axis = torch.as_tensor(axis, dtype=torch.float32)
    axis = axis / axis.norm()
    h = math.radians(deg) / 2.0
    return torch.cat([axis * math.sin(h), torch.tensor([math.cos(h)])])


def _qmul(a, b):                     # x y z w
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def _eff(B, H, step, began, exp_deg, shuf_deg):
    return {"step": step, "order_began": began, "expected_cm": torch.zeros(B, H, 3),
            "expected_deg": exp_deg, "expected_shuffled_cm": torch.zeros(B, H, 3),
            "expected_shuffled_deg": shuf_deg}


def _drift_orders(tr, n_orders, head_uses_order, seed=0, B=4, H=4):
    """Each body's result: a 60-degree turn its own state predicts (an axis that differs body to
    body and order to order) followed by 6 degrees of the 60-degree turn it was ordered. The head
    expects the drift, plus the order's 6 degrees when head_uses_order; its shuffled-order
    expectation is the same drift plus the OTHER body's order. Returns the readings the first
    build's control (another body's expectation) would have given."""
    from lem.model.efference import pose_change
    g = torch.Generator().manual_seed(seed)
    old_control = []
    for k in range(n_orders):
        begin, o, end = _pose(B, H), _order(B, H), _pose(B, H)
        drift = torch.zeros(B, H, 3)
        ask = torch.zeros(B, H, 3)
        for b in range(B):
            for h in range(H - 1):
                d = torch.randn(3, generator=g)
                a = torch.randn(3, generator=g)
                d, a = d / d.norm(), a / a.norm()
                o[b, h, 3:7] = _q_axis(a, 60.0)
                end[b, h, 3:7] = _qmul(_q_axis(a, 6.0), _q_axis(d, 60.0))
                drift[b, h], ask[b, h] = d * 60.0, a * 6.0
        exp = drift + (ask if head_uses_order else 0.0)
        shuf = drift + (ask.roll(1, dims=0) if head_uses_order else 0.0)
        s0 = 100 + 10 * k
        rec = _run_order(tr, begin, end, o, start=s0, B=B, next_id=2 + 10 * k,
                         eff=_eff(B, H, s0 + 3, s0, exp, shuf))
        assert rec is not None and rec["source"] == "efference"
        E = WC.effect_vector(torch.zeros(B, H, 3), exp, tr.body_mask, tr.body_mask)
        cm, deg = pose_change(begin, end)
        A = WC.effect_vector(cm, deg, tr.body_mask, tr.body_mask)
        old_control.extend((WC.agreement(E, A) - WC.agreement(E.roll(1, dims=0), A)).tolist())
    return old_control


def test_an_order_blind_head_with_body_varying_drift_never_opens_the_gate():
    """review of b3, major 1. The head expects each body's own state-driven drift and nothing of
    the order. Scored against another body's expectation (the first build's control) it beat its
    control far past the opening line; scored against the SHUFFLED ORDER it scores exactly 0 above
    it on every reading, and the gate never opens."""
    tr = WC.ControllabilityTracker(4, gaze_idx=3, hold=2, floors={"command": 0.5, "efference": 0.5},
                                   ema_orders=1, expect_efference=True)
    opened = []
    old = []
    for k in range(WC.CONTROL_GATE_READINGS + 8):
        old += _drift_orders(tr, 1, head_uses_order=False, seed=k)
        opened.append(tr.gate()[0])
    s = tr.summary(4)
    assert not any(opened) and s["gate_open"] is False
    assert s["difference_mean"] == pytest.approx(0.0, abs=1e-6)
    assert s["agreement_mean"] > 0.8, "the drift-only head does agree with the body"
    mean = sum(old) / len(old)
    se = (sum((v - mean) ** 2 for v in old) / (len(old) - 1) / len(old)) ** 0.5
    assert mean > WC.CONTROL_GATE_OPEN_NOISE * se, "the first build's control would have opened"


def test_a_head_that_uses_the_order_opens_the_gate_against_the_shuffled_order():
    tr = WC.ControllabilityTracker(4, gaze_idx=3, hold=2, floors={"command": 0.5, "efference": 0.5},
                                   ema_orders=1, expect_efference=True)
    for k in range(WC.CONTROL_GATE_READINGS + 8):
        _drift_orders(tr, 1, head_uses_order=True, seed=100 + k)
    s = tr.summary(4)
    assert s["difference_mean"] > 0.0 and s["gate_open"] is True
    assert "It is open" in s["gate_why"]


def test_the_command_source_control_takes_the_other_order_from_this_bodys_pose():
    """Body 0 starts turned 90 degrees about z, body 1 at rest; both are told to reach the same
    world orientation, and both do. The other body's order taken from body 0's own pose asks for
    exactly body 0's turn, so the control matches as well as the order does: an order that any
    body would have been given from that pose is no evidence of control."""
    B = 2
    tr = _tracker(B)
    begin = _pose(B)
    begin[0, 0, 3:7] = _qz(90.0)
    o = _order(B)
    o[:, 0, 3:7] = _qz(130.0)
    end = begin.clone()
    end[:, 0, 3:7] = _qz(130.0)
    rec = _run_order(tr, begin, end, o)
    assert rec["agreement"][0] == pytest.approx(1.0, abs=1e-4)
    assert rec["control"][0] == pytest.approx(1.0, abs=1e-4)


def _gate_after(tr, values):
    tr.gate_readings.clear()
    tr.gate_readings.extend(values)
    tr._update_gate()
    return tr.gate()


def _readings(mean_in_se, n=WC.CONTROL_GATE_READINGS):
    """n readings whose mean stands `mean_in_se` of their own standard errors above 0."""
    base = [(-1.0) ** i for i in range(n)]
    m0 = sum(base) / n
    sd = (sum((v - m0) ** 2 for v in base) / (n - 1)) ** 0.5
    se = sd / n ** 0.5
    return [v + mean_in_se * se for v in base]


def test_the_gate_opens_above_three_noises_and_closes_below_one():
    """review of b3, minor 1: one line re-tested at every planning event opened on noise 2.7 times
    in 100. Two lines: shut until the advantage clears 3 times its noise, then open until it falls
    under 1 time."""
    tr = _tracker(2)
    assert not _gate_after(tr, _readings(2.0))[0], "2 noises does not open a shut gate"
    ok, mean, line, why = _gate_after(tr, _readings(3.5))
    assert ok and line == pytest.approx(mean / 3.5 * WC.CONTROL_GATE_CLOSE_NOISE, rel=1e-6)
    assert "3.5 times its own measurement noise" in why
    assert _gate_after(tr, _readings(2.0))[0], "2 noises keeps an open gate open"
    assert _gate_after(tr, _readings(1.2))[0]
    assert not _gate_after(tr, _readings(0.8))[0], "under 1 noise it closes"
    assert not _gate_after(tr, _readings(2.5))[0], "and it needs 3 again to reopen"
    # a resume re-proves: the readings come back, the gate starts shut and re-judges them
    _gate_after(tr, _readings(2.0))
    tr.gate_is_open = True
    tr2 = _tracker(2)
    tr2.load_state_dict(tr.state_dict())
    assert list(tr2.gate_readings) == list(tr.gate_readings) and not tr2.gate()[0]
    # readings taken against the first build's control are not carried
    sd = tr.state_dict()
    sd.pop("control_kind")
    tr3 = _tracker(2)
    tr3.load_state_dict(sd)
    assert len(tr3.gate_readings) == 0


# ---------------------------------------------------------------------------- the ring's detrend
def test_the_detrend_stores_a_steady_drift_as_the_same_value_every_moment():
    names = ["integrity", "capability", WC.PAIN_ROW, WC.HARM_ROW]
    state = {}
    rows = []
    for t in range(600):
        raw = torch.tensor([[0.9 - 0.0003 * t, 0.5], [0.2 + 0.0001 * t, 0.6],
                            [0.1 + 0.0003 * t, 0.2], [0.1 + 0.0003 * t, 0.2]])
        rows.append(WC.detrend_moment(state, names, raw, 20))
    S = torch.stack(rows)                                                # [T, C, B]
    ring = S[-200:]
    # the drift, 0.06 of integrity across a ring, is gone: the stored value does not move
    assert float(ring[:, 0, 0].max() - ring[:, 0, 0].min()) < 1e-4
    assert float(ring[:, 1, 1].abs().max()) < 1e-5, "a constant stores 0 (to float32 rounding)"
    pain = torch.stack([torch.tensor([0.1 + 0.0003 * t, 0.2]) for t in range(400, 600)])
    assert torch.allclose(ring[:, 2], pain) and torch.allclose(ring[:, 3], pain), \
        "the gate's rows are stored raw"
    assert torch.isnan(S[0, 0]).all(), "a body's first moment has no average yet"
    raw = torch.tensor([[0.3, 0.4], [0.5, 0.6], [0.1, 0.1], [0.1, 0.1]])
    assert torch.equal(WC.detrend_moment({}, names, raw, None), raw), "no horizon = raw"


def test_the_detrend_skips_an_unmeasured_value_and_keeps_its_average():
    names = ["controllability", WC.PAIN_ROW]
    state = {}
    for v in (0.2, 0.4):
        WC.detrend_moment(state, names, torch.tensor([[v], [0.0]]), 10)
    before = [t.clone() for t in state["controllability"]]
    out = WC.detrend_moment(state, names, torch.tensor([[float("nan")], [0.0]]), 10)
    assert torch.isnan(out[0]).all()
    assert all(torch.equal(a, b) for a, b in zip(before, state["controllability"]))
    out = WC.detrend_moment(state, names, torch.tensor([[0.3, 0.3], [0.0, 0.0]]), 10)
    assert torch.isnan(out[0]).all() and state["controllability"][0].shape[0] == 2, \
        "a batch of another size restarts the average"


def _replay(names, stream, weights, horizon, plan_every=10, ring=200, floors=None, gate=0.99):
    """The lem_model path on a stream [N, C, B] of raw channel vectors: store (detrend), learn the
    normaliser on the stored values, rank every `plan_every` moments once the ring is full, and
    count where the chosen goal sat. Returns the goal_age block."""
    norm = WC.ChannelNormalizer()
    shares = WC.RankShares()
    state, ringl = {}, []
    floors = floors or {n: 1e-6 for n in names}
    for t in range(stream.shape[0]):
        stored = WC.detrend_moment(state, names, stream[t], horizon)
        for ci, n in enumerate(names):
            if n not in WC.UNRANKED_ROWS:
                norm.update(n, stored[ci])
        ringl.append(stored)
        if len(ringl) > ring:
            ringl.pop(0)
        if len(ringl) >= ring and t % plan_every == 0:
            ranked, *_ = WC.rank_values(torch.stack(ringl), names, weights, norm, floors, gate)
            shares.add_goal(ranked.argmax(0).tolist(), len(ringl))
    return shares.summary()["goal_age"]


def _stream(N, B, seed, drift=False):
    """Channel values with no age in them - seeded noise around fixed levels and consequence events
    on 5 in 100 moments - or (drift=True) integrity falling in a straight line as skin load builds
    (0.06 across a ring against a moment-to-moment spread of 0.005) and learning progress rising
    in a straight line (0.01 across a ring), the shapes the b3 review measured inside a ring."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(N, dtype=torch.float32).view(N, 1)
    zero = torch.zeros(N, B)
    if drift:
        names = ["integrity", "learning_progress", "coherence", "prediction_error",
                 WC.PAIN_ROW, WC.HARM_ROW]
        integ = 0.3 - 0.0003 * t + 0.005 * torch.randn(N, B, generator=g)
        lp = 0.00005 * t + 0.0005 * torch.randn(N, B, generator=g)
        coh = 0.5 + 0.005 * torch.randn(N, B, generator=g)
        pe = 0.7 + 0.01 * torch.randn(N, B, generator=g)
        return names, torch.stack([integ, lp, coh, pe, zero, zero], dim=1)
    names = ["integrity", "learning_progress", "coherence", "consequence", WC.PAIN_ROW, WC.HARM_ROW]
    integ = 0.3 + 0.02 * torch.randn(N, B, generator=g)
    lp = 0.0005 * torch.randn(N, B, generator=g)
    coh = 0.5 + 0.005 * torch.randn(N, B, generator=g)
    cq = (torch.rand(N, B, generator=g) < 0.05).float()
    return names, torch.stack([integ, lp, coh, cq, zero, zero], dim=1)


# THE BAND A CHOICE BLIND TO AGE FALLS IN, on this replay: 10 goals in 100 in each end tenth of the
# ring, 20 together. MEASURED 2026-09-23 (lane-b3-wellness, scratchpad lane_b/calib_ageblind.py):
# the age-blind stream below at RING_DETREND_STEPS gave 18.5 to 21.0 in 100 at the two ends
# together across 30 seeds (stored raw: 18.9 to 20.6); the drifting stream gave 30.7 to 33.7
# stored raw (mean share back 0.72-0.74, pulled toward the oldest end) and 18.9 to 21.1 stored as
# distances from each body's running average.
AGE_BLIND_ENDS_PER_100 = (18.5, 21.1)


def test_an_age_blind_input_gives_an_age_blind_choice():
    names, stream = _stream(3000, 16, seed=3)
    ga = _replay(names, stream, WC.DEFAULT_WEIGHTS, WC.RING_DETREND_STEPS)
    ends = 100.0 * (ga["oldest_tenth_share"] + ga["newest_tenth_share"])
    assert AGE_BLIND_ENDS_PER_100[0] <= ends <= AGE_BLIND_ENDS_PER_100[1], ga
    assert 0.4 <= ga["mean_share_back"] <= 0.6, ga


def test_a_trending_channel_does_not_pull_the_goal_to_one_end():
    """Stored raw (the first build) the goals crowd toward the oldest end (31 in 100 at the two
    ends, mean share back 0.72); stored as distances from each body's running average the choice
    falls in the band an age-blind input gives."""
    names, stream = _stream(3000, 16, seed=4, drift=True)
    before = _replay(names, stream, WC.DEFAULT_WEIGHTS, None)
    assert 100.0 * (before["oldest_tenth_share"] + before["newest_tenth_share"]) > 28.0, before
    assert before["mean_share_back"] > 0.7, before
    ga = _replay(names, stream, WC.DEFAULT_WEIGHTS, WC.RING_DETREND_STEPS)
    ends = 100.0 * (ga["oldest_tenth_share"] + ga["newest_tenth_share"])
    assert AGE_BLIND_ENDS_PER_100[0] <= ends <= AGE_BLIND_ENDS_PER_100[1], ga


def test_the_goal_age_counts_where_each_goal_sat():
    sh = WC.RankShares()
    sh.add_goal([0, 199, 100, 180, 19], 200)
    g = sh.summary()["goal_age"]
    assert g["events"] == 5 and g["oldest_tenth_share"] == 0.4 and g["newest_tenth_share"] == 0.4
    assert g["mean_share_back"] == pytest.approx((1.0 + 0.0 + 99 / 199 + 19 / 199 + 180 / 199) / 5,
                                                 abs=1e-4)
    assert WC.RankShares().summary()["goal_age"] is None


# ---------------------------------------------------------------------------- the harm gate
def test_the_gate_reads_harm_the_limit_part_included_and_never_an_unmeasured_one():
    names = ["consequence", WC.PAIN_ROW, WC.HARM_ROW]
    ch = torch.zeros(3, 3, 1)
    ch[:, 0, 0] = torch.tensor([0.0, 3.0, 1.0])
    ch[:, 1, 0] = torch.tensor([0.1, 0.1, 0.1])            # the skin is fine everywhere
    ch[:, 2, 0] = torch.tensor([0.1, 0.95, float("nan")])  # 1: a limb jammed; 2: unmeasured
    norm = WC.ChannelNormalizer(horizon=10)
    _feed(norm, names, ch)
    ranked, _, gated, ag = WC.rank_values(ch, names, {"consequence": 0.5}, norm,
                                          {"consequence": 0.1}, gate_line=0.9)
    assert gated[:, 0].tolist() == [False, True, False]
    assert int(ranked[:, 0].argmax()) == 2
    ch[:, 2, 0] = torch.tensor([0.99, 0.95, 0.97])
    r2, _, g2, ag2 = WC.rank_values(ch, names, {"consequence": 0.5}, norm, {"consequence": 0.1}, 0.9)
    assert bool(ag2[0]) and int(r2[:, 0].argmax()) == 1 and torch.isinf(r2[0, 0]), \
        "every moment over the line: only the least harmful keeps its value"


# ---------------------------------------------------------------------------- learning progress
def test_a_freshly_started_sense_sits_out_for_the_holdout_then_starts_fresh():
    lp = WC.LearningProgressTracker(fast=5, slow=50)
    for t in range(20):
        lp.update("vestibular", torch.tensor([1.0]), torch.tensor([1.0]))
        lp.update("vision", torch.tensor([0.5]), torch.tensor([1.0]))
    lp.set_holdout(["vestibular"], 3)
    assert "vestibular" not in lp.state, "averages of the replaced network are dropped"
    counted = []
    for t in range(5):
        lp.tick_holdout()
        lp.update("vestibular", torch.tensor([0.01]), torch.tensor([1.0]))
        counted.append("vestibular" in lp.state)
    assert counted == [False, False, False, True, True], "exactly 3 steps out, counted from the 4th"
    assert "vision" in lp.state and not lp.holdout
    lp.set_holdout(["efference_body"], 7)
    lp2 = WC.LearningProgressTracker(fast=5, slow=50)
    lp2.load_state_dict(lp.state_dict())
    assert lp2.holdout == {"efference_body": 7}
    assert lp.summary(1)["held_out_steps"] == {"efference_body": 7}


def test_the_cost_window_is_per_model_step_not_per_timed_call():
    """The channels' per-step work is timed in two places each model step (the observation and
    the ring's record); the window divides by model steps, so a flags-on step time and a flags-off
    one compare step for step (machine.wellness_channels_ms)."""
    st = WC.WellnessChannelState(True, None, None)
    for _ in range(4):
        st.add_cost(1.0, new_step=True)      # _observe_wellness_channels
        st.add_cost(2.0)                     # _rank_store_moment
    st.add_cost(10.0, planning=True)
    w = st.cost_window()
    assert w == {"per_step_ms": 3.0, "planning_ms": 10.0, "steps": 4, "planning_events": 1}
    assert st.cost_window()["steps"] == 0, "each call starts a fresh window"
    from mimo_bridge.machine_stats import MachineSampler
    m = MachineSampler(gpu_reader=lambda: None).sample(model_ms=100.0, env_ms=10.0, wellness=w)
    assert m["wellness_channels_ms"] == 5.5 and m["wellness_rank_ms_per_plan"] == 10.0
