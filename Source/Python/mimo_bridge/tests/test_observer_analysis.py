# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The developmental observer's analysis (mimo_bridge/tools/observer): what each layer decides.

Pinned here, on synthetic inputs where the right answer is known exactly:
  * relaunch ordering and abandoned futures (stream.py);
  * the primitive helpers' runs and turns (primitives.py);
  * every tracker stage's rule: unmeasured without its control, reached only against chance by
    its statistical test, generalization across bodies and contexts, retention and regression;
  * the novelty miner keeps recurring UNEXPLAINED patterns and drops explained ones, and drops a
    run random-order babble makes as often;
  * the primitives: a short break in one contact is one touch, an impact is a force RISING by a
    body weight (not a resting load), a loss of support is outside force held low, a weight
    shift is the centre of mass moving and staying moved;
  * the fast lane separates a brief peak from a gradual one of the same total;
  * the evidence join reads the contact head against its base rate, the efference expectation
    against the shuffled control, and the planner's decisiveness;
  * the record store reads whole gzip members only, and its saves are generations;
  * a recorded crib world digests end to end into a report with every catalog row; a digest
    stopped between writing and committing never counts a window twice; small windows lose no
    chunk and emit no act twice; a finished run is sealed; an analysis of another detector
    version is moved aside.
The scripted validation worlds (tools/observer/validate.py) cover the detectors on real physics.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mimo_bridge.tools.observer import catalog as CAT
from mimo_bridge.tools.observer import evidence as EV
from mimo_bridge.tools.observer import fastlane as FL
from mimo_bridge.tools.observer import novelty as NV
from mimo_bridge.tools.observer import primitives as P
from mimo_bridge.tools.observer import stream as ST
from mimo_bridge.tools.observer import tracker as TR


# ------------------------------------------------------------------ stream
def _touch(p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_bytes(b"")


def test_segments_are_ordered_by_process_start_and_a_resume_abandons_the_rolled_back_future(tmp_path):
    b0 = tmp_path / "observer" / "world" / "b0"
    # process 900 started first and reached step 499; process 100 (a smaller pid, started later)
    # resumed from checkpoint 300: steps 300..499 of process 900 are an abandoned future
    _touch(b0 / "header_900_1000.json")
    _touch(b0 / "header_100_2000.json")
    for first, pid, seq in ((0, 900, 0), (100, 900, 1), (200, 900, 2), (300, 900, 3), (400, 900, 4),
                            (300, 100, 0), (400, 100, 1)):
        _touch(b0 / ("chunk_%09d_%d_%05d.npz" % (first, pid, seq)))
    files = ST.RunFiles(str(tmp_path))
    segs = files.world_segments(0)
    assert [s.pid for s in segs] == [900, 100], "by the process's first header time, not pid"
    assert [c.seq for c in segs[0].chunks] == [0, 1, 2, 3, 4]
    cuts = ST.abandoned_from(segs)
    assert cuts == {900: 300} and 100 not in cuts
    live = ST.live_mask(np.array([299, 300, 450]), 900, cuts)
    assert live.tolist() == [True, False, False]
    assert ST.live_mask(np.array([300, 450]), 100, cuts).tolist() == [True, True]


# ------------------------------------------------------------------ primitives helpers
def test_intervals_break_at_discontinuities_and_turns_find_monotone_runs():
    mask = np.array([0, 1, 1, 1, 0, 1, 1], dtype=bool)
    cont = np.array([1, 1, 1, 0, 1, 1, 1], dtype=bool)
    assert P._intervals(mask, cont) == [(1, 2), (3, 3), (5, 6)]
    x = np.array([0, 5, 12, 20, 20, 8, 1, 1, 18], dtype=float)
    turns = P._turns(x, np.ones(len(x), dtype=bool), 10.0)
    # detector v9: a turn starts where its movement starts - the LAST record of the level it leaves
    # (the fall leaves 20 after record 4, the rise leaves 1 after record 7); until v8 the FIRST
    # record of that level (3 and 6), counting the stillness at the extreme into the next turn
    assert [(a, b) for a, b, _ in turns] == [(0, 3), (4, 6), (7, 8)]
    assert [round(c) for _, _, c in turns] == [20, -19, 17]
    # a change below the minimum is not a turn
    assert P._turns(np.array([0, 4, 8, 3.0]), np.ones(4, dtype=bool), 10.0) == []


# ------------------------------------------------------------------ tracker
def _det(step, body=0, episode=None, prov="spontaneous", pred=None, intent=None, posture="back",
         objects=(), key="k", measures=None, present=True):
    return dict(cap="X", sub="", key=key, body=body, step0=step, step1=step + 2,
                episode=episode or "e%d:%d" % (body, step), provenance=prov, live=True, scripted=False,
                caregiver_touch=False, caregiver_present=present, posture=posture, objects=list(objects),
                measures=measures or {"m": 1.0},
                evidence=dict(prediction=None if pred is None else dict(kind="efference", verdict=pred),
                              intention=None if intent is None else dict(verdict=intent)))


def _exposure(steps, by=None, windows=None, bodies=8):
    return dict(steps=steps, by_author=by or {"spontaneous": steps}, windows=windows or {}, bodies=bodies)


RARE = dict(count=0, occasions=0, steps=100000)     # a babble that never made the act in 100,000 steps


def _spread(n, start=0, gap=100, **kw):
    """n acts on separate occasions (gap steps apart on one body)."""
    return [_det(start + gap * i, **kw) for i in range(n)]


def test_without_a_baseline_every_stage_against_babble_is_unmeasured_never_failed():
    out = TR.evaluate(_spread(5), _exposure(10000), None, None)
    assert out["stages"]["observed_once"]["reached"] is None
    assert out["stages"]["repeated"]["reached"] is None
    assert "unmeasured" in out["stages"]["repeated"]["note"]


def test_repeated_counts_occasions_against_babbles_own_occasions():
    base = dict(count=10, occasions=10, steps=100000)
    few = _spread(3, gap=1000)                          # 3 against 10 in ten times the steps
    many = _spread(12, gap=500)
    r_few = TR.evaluate(few, _exposure(10000), base, None)["stages"]["repeated"]
    r_many = TR.evaluate(many, _exposure(10000), base, None)["stages"]["repeated"]
    assert r_few["reached"] is False and r_few["p_value"] > 0.05
    assert r_many["reached"] is True and r_many["p_value"] < 0.05 and r_many["first_step"] is not None
    # conditional binomial: 12 of the 22 occasions on the side that holds 1/11 of the steps
    assert r_many["p_value"] == pytest.approx(TR.two_sample_above(12, 10000, 10, 100000))
    loud = dict(count=1000, occasions=1000, steps=100000)
    assert TR.evaluate(many, _exposure(10000), loud, None)["stages"]["repeated"]["reached"] is False
    # a burst is one occasion: 12 acts inside 0.6 s never repeat
    burst = [_det(i) for i in range(12)]
    rb = TR.evaluate(burst, _exposure(10000), RARE, None)["stages"]["repeated"]
    assert rb["occasions"] == 1 and rb["reached"] is False


def test_a_babble_that_never_made_it_still_carries_its_own_uncertainty():
    base = dict(count=0, occasions=0, steps=3000)
    two = TR.evaluate(_spread(2), _exposure(1000), base, None)["stages"]["repeated"]
    three = TR.evaluate(_spread(3), _exposure(1000), base, None)["stages"]["repeated"]
    assert two["reached"] is False                       # fewer than 3 occasions
    # all 3 occasions on the side holding a quarter of the steps: 0.25 ** 3
    assert three["p_value"] == pytest.approx(0.25 ** 3) and three["reached"] is True


def test_observed_once_needs_a_share_of_occasions_beyond_babbles_95th_percentile_above_five_percent():
    base = dict(count=50, occasions=50, steps=10000, measures={"m": [0.1, 0.5, 2.0]},
                measured_occasions=50, measures_occasion={"m": [0.1, 0.5, 2.0]})
    q = ("m", "higher")
    weak = [_det(1, measures={"m": 1.0})]
    assert TR.evaluate(weak, _exposure(100), base, q)["stages"]["observed_once"]["reached"] is False
    lucky = [_det(1, measures={"m": 1.0}), _det(30, measures={"m": 3.0})]     # 1 of 2: babble does 5%
    assert TR.evaluate(lucky, _exposure(100), base, q)["stages"]["observed_once"]["reached"] is False
    # a burst is one piece of evidence (tracker_rules_v4): four big acts inside one second are one
    # occasion, judged by its first act
    burst = [_det(1, measures={"m": 1.0})] + [_det(5 + i, measures={"m": 3.0}) for i in range(3)]
    b1 = TR.evaluate(burst, _exposure(100), base, q)["stages"]["observed_once"]
    assert b1["reached"] is False and b1["beyond"] == 0 and b1["measured"] == 1
    strong = [_det(1, measures={"m": 1.0})] + [_det(30 * (i + 1), measures={"m": 3.0}) for i in range(3)]
    s = TR.evaluate(strong, _exposure(100), base, q)["stages"]["observed_once"]
    assert s["reached"] is True and s["beyond"] == 3 and s["measured"] == 4 and s["by"] == "size"
    # dated where the running test began to pass for good, not at the first occasion beyond: after
    # steps 1 and 30 (1 of 2 beyond, doubled p 0.195) it had not; from step 60 (2 of 3, doubled p
    # 0.0145) it passed at every occasion
    assert s["first_step"] == 60
    assert 2 * TR.binom_above(1, 2, 0.05) > 0.05 and 2 * TR.binom_above(2, 3, 0.05) < 0.05
    # a babble summary without the occasion-level line (written before v4) cannot judge size
    old = dict(count=50, occasions=50, steps=10000, measures={"m": [0.1, 0.5, 2.0]})
    s0 = TR.evaluate(strong, _exposure(100), old, q)["stages"]["observed_once"]
    assert s0["share_p"] is None and s0["measure"] is None and s0["by"] != "size"
    # no bigger than babble's range but far more often: seen by frequency
    freq = [_det(i * 50, measures={"m": 1.0}) for i in range(20)]
    s2 = TR.evaluate(freq, _exposure(1000), base, q)["stages"]["observed_once"]
    assert s2["reached"] is True and s2["by"] == "frequency" and s2["beyond"] == 0
    # dated at the occasion from which the running rate test passed at every later one, and babble's
    # count scaled to the same body-steps rides along for the page (50 in 10,000 steps: 5 in 1,000)
    assert s2["first_step"] > 0 and s2["expected_by_chance"] == pytest.approx(5.0)
    # an act without the measure is no evidence either way
    blank = [_det(1, measures={"x": 9.0}) for _ in range(5)]
    b = TR.evaluate(blank, _exposure(100), base, q)["stages"]["observed_once"]
    assert b["reached"] is False and b["measured"] == 0


def test_the_ladder_is_in_order_a_later_stage_waits_for_the_earlier():
    yes = _spread(9, pred=True) + [_det(950, pred=False)]
    out = TR.evaluate(yes, _exposure(10000), None, None)["stages"]
    pr = out["predicted_before_action"]
    assert pr["reached"] is False and pr["waits_for"] == "repeated" and pr["predicted"] == 9


def test_a_later_stage_is_reached_no_earlier_than_the_one_it_waits_for():
    yes = _spread(9, pred=True) + [_det(950, pred=False)]
    st = TR.evaluate(yes, _exposure(10000), RARE, None)["stages"]
    assert st["predicted_before_action"]["first_step"] >= st["repeated"]["first_step"]


def test_predicted_before_action_is_a_binomial_test_against_one_half():
    yes = _spread(9, pred=True) + [_det(950, pred=False)]
    coin = [_det(i * 100, pred=bool(i % 2)) for i in range(10)]
    none = _spread(10)
    ex = _exposure(10000)
    assert TR.evaluate(yes, ex, RARE, None)["stages"]["predicted_before_action"]["reached"] is True
    assert TR.evaluate(coin, ex, RARE, None)["stages"]["predicted_before_action"]["reached"] is False
    assert TR.evaluate(none, ex, RARE, None)["stages"]["predicted_before_action"]["reached"] is None


def test_selected_intentionally_needs_decisive_choices_and_more_acts_while_the_planner_leads():
    ex = _exposure(2000, by={"planner-selected": 1000, "spontaneous": 1000})
    planner = [_det(i * 100, prov="planner-selected", intent=True) for i in range(10)]
    babble = [_det(1000 + i * 100, prov="spontaneous") for i in range(2)]
    out = TR.evaluate(planner + babble, ex, RARE, None)["stages"]["selected_intentionally"]
    assert out["reached"] is True and out["decisive"] == 10
    even = [_det(i * 100, prov="planner-selected", intent=True) for i in range(5)] + \
           [_det(1000 + i * 100, prov="spontaneous") for i in range(5)]
    assert TR.evaluate(even, ex, RARE, None)["stages"]["selected_intentionally"]["reached"] is False
    tied = [_det(i * 100, prov="planner-selected", intent=False) for i in range(10)]
    t = TR.evaluate(tied, ex, RARE, None)["stages"]["selected_intentionally"]
    assert t["reached"] is False and t.get("p_value") is None, "no decisive choice: the rate is not tested"
    no_planner = TR.evaluate(babble, _exposure(1000), RARE, None)["stages"]["selected_intentionally"]
    assert no_planner["reached"] is False and "no command" in no_planner["note"]


def test_generalized_needs_three_bodies_and_two_contexts_after_repeating():
    base = dict(count=1, occasions=1, steps=1000000)
    one_body = [_det(i * 50, body=0, episode="e%d" % i) for i in range(10)]
    many = [_det(i * 50, body=i % 4, episode="e%d" % i, posture="back" if i % 2 else "side") for i in range(10)]
    assert TR.evaluate(one_body, _exposure(10000), base, None)["stages"]["generalized"]["reached"] is False
    assert TR.evaluate(many, _exposure(10000), base, None)["stages"]["generalized"]["reached"] is True
    # the ladder holds: on four bodies from two starts but not repeating more than babble, it waits
    # for repeated and keeps its numbers; when it does repeat it is dated no earlier than repeated,
    # and repeated no earlier than seen
    loud = dict(count=5000, occasions=5000, steps=100000)
    wait = TR.evaluate(many, _exposure(10000), loud, None)["stages"]
    assert wait["generalized"]["reached"] is False and wait["generalized"]["waits_for"] == "repeated"
    assert wait["generalized"]["bodies"] == 4 and wait["retained"]["waits_for"] == "repeated"
    st = TR.evaluate(many, _exposure(10000), base, None)["stages"]
    assert st["generalized"]["first_step"] >= st["repeated"]["first_step"] >= st["observed_once"]["first_step"]


def test_retained_across_later_windows_and_a_drop_marks_regression():
    base = dict(count=1, steps=10 ** 7)
    W = TR.RET_WINDOW
    windows = {str(w): 8 * W for w in range(4)}
    keep = [_det(w * W + i * 100, body=i % 4, episode="e%d_%d" % (w, i)) for w in range(4) for i in range(20)]
    r = TR.evaluate(keep, _exposure(32 * W, windows=windows), base, None)
    assert r["stages"]["retained"]["reached"] is True and not r["regressed"]
    fade = [d for d in keep if d["step0"] < 2 * W or d["step0"] % (7 * 100) == 0]
    r2 = TR.evaluate(fade, _exposure(32 * W, windows=windows), base, None)
    assert r2["regressed"] is True and r2["stages"]["retained"]["reached"] is False


def test_separate_occasions_are_acts_a_second_apart_on_one_body():
    steps = np.array([0, 5, 10, 40, 41, 100])
    assert TR.occasions(steps, np.zeros(6, dtype=int)) == 3       # 0-10, 40-41, 100
    # every body counts its own: body 0 acts at 0, 10, 41 (two occasions), body 1 at 5, 40, 100 (three)
    assert TR.occasions(steps, np.array([0, 1, 0, 1, 0, 1])) == 5
    few = [_det(i, episode="one") for i in range(0, 30, 2)]       # 15 acts inside 1.5 s
    base = dict(count=1, steps=10 ** 7)
    r = TR.evaluate(few, _exposure(10000), base, None)["stages"]["repeated"]
    assert r["reached"] is False and r["occasions"] == 1


def test_occasions_are_counted_per_body_and_part_so_two_hands_never_merge():
    """Both hands spreading every 15 steps (babble's pattern) are two parts that never pause 1 s;
    merged per body they were ONE occasion, which made a learner that pauses look above babble."""
    steps = np.arange(0, 300, 15)
    both = np.r_[steps, steps + 1]
    bodies = np.zeros(both.size, dtype=np.int64)
    hands = np.r_[np.zeros(steps.size), np.ones(steps.size)].astype(np.int64)
    assert TR.occasions(both, bodies) == 1, "merged per body: one burst"
    assert TR.occasions(both, bodies, hands) == 2, "per hand: one burst each"
    # a row's evaluation counts per part (its sub-row), the same way babble's summary does
    acts = [dict(_det(t), sub=hand) for t in range(0, 1000, 100) for hand in ("L", "R")]
    out = TR.evaluate(acts, _exposure(10000), RARE, None)["stages"]["repeated"]
    assert out["occasions"] == 20, "both hands at the same steps: ten occasions each"


def test_more_often_than_babble_needs_more_acts_as_well_as_more_occasions():
    """A learner that pauses more than babble has more separate occasions without doing more; both
    counts must beat babble's (review 2026-09-24: finger spreading, 1,025 occasions against 894 but
    2,389 acts against 3,279)."""
    base = dict(count=400, occasions=40, steps=10000)             # babble: 10 acts to an occasion
    pausing = [_det(i * 25) for i in range(80)]                    # 80 acts, each its own occasion
    st = TR.evaluate(pausing, _exposure(10000), base, None)["stages"]
    assert st["repeated"]["occasions"] == 80 and st["repeated"]["reached"] is False
    assert st["observed_once"]["reached"] is False
    assert TR.two_sample_above(80, 10000, 40, 10000) < 0.001, "occasions alone would have passed"


def test_many_tests_at_once_use_the_false_discovery_line():
    # Benjamini-Hochberg: sorted p against k/m * 5% -> 0.001 and 0.02 pass, 0.04 (alone it would) not;
    # the line is 5% x 2 / 4 (tracker_rules_v5: q k / m, not the largest passing p-value)
    assert TR.fdr_line([0.3, 0.04, 0.001, 0.02]) == pytest.approx(0.025)
    assert TR.fdr_line([0.2, 0.5]) == 0.0 and TR.fdr_line([]) == 0.0
    cat = dict(capabilities=[dict(id="C%d" % i, name="c%d" % i, group="g", quality=None) for i in range(20)])
    base = {"C%d" % i: dict(count=10, steps=100000) for i in range(20)}
    dets = []
    for i in range(20):                  # expected 1 by chance in 10,000 steps: C0 30 acts, C1 4
        n = 30 if i == 0 else 4 if i == 1 else 3          # (p 0.019 alone), the other 18 three (p 0.08)
        dets += [dict(_det(j * 100), cap="C%d" % i) for j in range(n)]
    out = TR.track_all(dets, {"0": dict(steps=10000, by_author={"spontaneous": 10000}, windows={})}, base, cat)
    assert out["C0"]["stages"]["repeated"]["reached"] is True
    alone = TR.evaluate([dict(_det(j * 100), cap="C1") for j in range(4)], _exposure(10000), base["C1"], None)
    assert alone["stages"]["repeated"]["reached"] is True, "4 against 1 expected passes when tested alone"
    r1 = out["C1"]["stages"]["repeated"]
    assert r1["reached"] is False and r1["tests_together"] == 20 and r1["alpha"] < 0.05


def test_scripted_probe_and_caregiver_acts_never_count():
    dets = [_det(1), dict(_det(2), scripted=True), dict(_det(3), provenance="probe-forced"),
            dict(_det(4), caregiver_touch=True), dict(_det(5), live=False)]
    out = TR.evaluate(dets, _exposure(100), None, None)
    assert out["counted"] == 1 and out["scripted"] == 1 and out["caregiver_involved"] == 1


def test_every_catalog_row_is_tracked_and_the_unobservable_ones_say_why():
    cat = CAT.load()
    assert len([c for c in cat["capabilities"] if not c["id"].startswith("HX")]) == 52
    out = TR.track_all([], {"0": dict(steps=100, by_author={}, windows={})}, {}, cat)
    assert set(out) == {c["id"] for c in cat["capabilities"]}
    for cid in ("P11", "P12", "O09", "O10"):
        assert out[cid]["observable"] is False and out[cid]["why_not"]
        assert all(s["reached"] is None for s in out[cid]["stages"].values())


# ------------------------------------------------------------------ novelty
def _ep(i, body, tokens, caps=(), step=None):
    return dict(id="ep%d" % i, body=body, step0=step if step is not None else i * 100, step1=(step or i * 100) + 20,
                tokens=list(tokens), caps=list(caps), live=True, caregiver_touch=False, authors={2: 20})


def test_novelty_keeps_recurring_unexplained_patterns_and_drops_explained_ones():
    new = ["touch:hand_R_palm~mobile", "object_moves:mobile", "curl:R:open"]
    known = ["curl:R:close", "curl:R:open"]
    eps = [_ep(i, i % 3, new) for i in range(6)]
    eps += [_ep(10 + i, i % 3, known, caps=["H04"]) for i in range(6)]
    eps += [_ep(20 + i, 0, ["touch:foot_L~floor", "head_turn"]) for i in range(6)]   # one body only
    cands = NV.mine(eps)
    toks = [c["tokens"] for c in cands]
    assert any(t == new or t == new[:2] for t in toks)
    assert not any(t == known for t in toks), "H04 explains a close-open curl"
    assert not any("touch:foot_L~floor" in t for t in toks), "a pattern on one body is not general"
    top = cands[0]
    assert top["world_effect"] and top["label"].startswith("novel candidate #")
    assert NV.candidate_id(top["tokens"]) == top["id"], "a stable id from the tokens"
    o12 = NV.o12_detections(cands, {e["id"]: e for e in eps})
    assert o12 and all(d["cap"] == "O12" and d["provenance"] == "spontaneous" for d in o12)


def test_novelty_drops_a_run_random_order_babble_makes_as_often():
    new = ["touch:hand_R_palm~mobile", "object_moves:mobile"]
    eps = [_ep(i, i % 3, new) for i in range(6)]
    rid = NV.candidate_id(tuple(new))
    kept = NV.mine(eps, baseline=dict(runs={}, steps=10000), exposure_steps=1000)
    top = [c for c in kept if c["id"] == rid]
    # babble never made it: the chance that all 6 fall on the side with 1/11 of the steps
    assert top and top[0]["expected_by_chance"] == 0 and top[0]["p_value"] == pytest.approx((1 / 11) ** 6)
    loud = NV.mine(eps, baseline=dict(runs={rid: 60}, steps=10000), exposure_steps=1000)
    assert not [c for c in loud if c["id"] == rid], "babble makes it 6 times per 1000 steps too"
    unmeasured = NV.mine(eps)
    assert [c for c in unmeasured if c["id"] == rid][0]["chance_measured"] is False


# ------------------------------------------------------------------ primitives on made-up tracks
class _FakeTrack:
    def __init__(self, n, weight=100.0):
        self.n = n
        self.step = np.arange(n)
        self.sim_time = 0.05 * np.arange(n)
        self.continuous = np.ones(n, dtype=bool)
        self.continuous[0] = False
        self.header = dict(weight_n=weight)
        self.fast = np.full((n, 5, 8), np.nan)
        self.fast[:, 0, :5] = 9.81
        self.fast[:, 1, :5] = 0.2


def test_a_short_break_in_one_contact_is_the_same_touch():
    on = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1], dtype=bool)
    cont = np.ones(9, dtype=bool)
    assert P.merge_breaks(on, cont, 2).tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 1]
    cut = cont.copy()
    cut[2] = False                                  # a break across a record gap stays a break
    assert P.merge_breaks(on, cut, 2).tolist() == on.tolist()


def test_an_impact_is_a_rising_force_and_a_resting_load_is_none():
    tr = _FakeTrack(20)
    tr.fast[:, 3, :5] = 60.0                        # the largest resting contact: 0.6 body weights
    tr.fast[:, 2, :5] = 100.0                       # outside force = the body weight
    assert [e for e in P.transients(tr) if e["kind"] == "impact"] == []
    tr.fast[10, 3, 2] = 170.0                       # +1.1 body weights within one 10 ms frame
    hits = [e for e in P.transients(tr) if e["kind"] == "impact"]
    assert [e["i0"] for e in hits] == [10] and hits[0]["peak"] == pytest.approx(1.1)


def test_a_loss_of_support_is_outside_force_held_low_not_a_frame_swing():
    tr = _FakeTrack(20)
    tr.fast[:, 3, :5] = 50.0
    tr.fast[:, 2, :5] = 100.0
    tr.fast[5, 2, 1] = 10.0                         # one frame down: a solver swing
    assert [e for e in P.transients(tr) if e["kind"] == "support_loss"] == []
    tr.fast[8, 2, 3:5] = 10.0                       # 30 ms down across two steps
    tr.fast[9, 2, 0] = 10.0
    hits = [e for e in P.transients(tr) if e["kind"] == "support_loss"]
    assert len(hits) == 1 and (hits[0]["i0"], hits[0]["i1"]) == (8, 9) and hits[0]["low_ms"] == 30.0


def test_a_weight_shift_is_the_centre_of_mass_moving_and_staying_moved():
    tr = _FakeTrack(40)
    tr.com = np.zeros((40, 3))
    tr.trunk_lat = np.tile([0.0, 1.0, 0.0], (40, 1))
    tr.com[10:, 1] = np.minimum(0.03, 0.01 * np.arange(1, 31))    # 3 cm toward the body's left, kept
    ev = P.support_shifts(tr, {})
    assert len(ev) == 1 and ev[0]["lateral_m"] > 0.019 and ev[0]["i0"] < 12
    wob = _FakeTrack(40)
    wob.com = np.zeros((40, 3))
    wob.trunk_lat = tr.trunk_lat
    wob.com[10:13, 1] = [0.01, 0.025, 0.005]                       # out and straight back
    assert P.support_shifts(wob, {}) == []


# ------------------------------------------------------------------ record store
def test_the_store_reads_whole_gzip_members_and_leaves_a_torn_one(tmp_path):
    from mimo_bridge.tools.observer import store as STO
    path = str(tmp_path / "detections.jsonl.gz")
    STO.append_records(path, [dict(a=1), dict(a=2)], json.dumps)
    STO.append_records(path, [dict(a=3)], json.dumps)
    whole = os.path.getsize(path)
    rows, used = STO.read_records(path)
    assert [r["a"] for r in rows] == [1, 2, 3] and used == whole
    import gzip
    torn = gzip.compress(b'{"a": 4}\n')
    with open(path, "ab") as f:
        f.write(torn[: len(torn) // 2])
    rows, used = STO.read_records(path)
    assert [r["a"] for r in rows] == [1, 2, 3] and used == whole, "the torn member is left for later"
    assert STO.truncate(path, whole) and os.path.getsize(path) == whole


def test_the_store_saves_in_generations_and_a_cut_off_save_keeps_the_last(tmp_path):
    from mimo_bridge.tools.observer import store as STO
    a = tmp_path / "run1"
    a.mkdir()
    STO.append_records(str(a / "detections.jsonl.gz"),
                       [dict(cap="H01", sub="L", step0=i, body=0, pid=1, measures={"m": 1.0}) for i in range(5)],
                       json.dumps)
    folder = str(tmp_path / "store")
    s1 = STO.LineageStore(folder, {"H01": "m"})
    assert s1.update("run1", str(a)) == 5 and s1.save(force=True)
    s2 = STO.LineageStore(folder, {"H01": "m"})
    assert len(s2.dets) == 5 and s2.offsets == s1.offsets
    # a save cut off after its data files, before its meta: the last complete save stands
    with open(os.path.join(folder, "store_dets_%d.npz" % (s1.generation + 1)), "wb") as f:
        f.write(b"half a file")
    s3 = STO.LineageStore(folder, {"H01": "m"})
    assert len(s3.dets) == 5
    # a store that read further than a run's file now reaches says so
    STO.truncate(str(a / "detections.jsonl.gz"), 0)
    with pytest.raises(STO.StoreMismatch):
        s3.update("run1", str(a))


# ------------------------------------------------------------------ fast lane
def test_the_fast_lane_separates_a_brief_peak_that_a_slow_average_does_not():
    brief = np.zeros(400)
    brief[100] = 50.0
    gradual = np.zeros(400)
    gradual[100:150] = 1.0
    cmp = FL.brief_vs_gradual(brief, gradual, slow_window=200)
    assert cmp["peak_ratio"] == pytest.approx(50.0)
    assert cmp["slow_ratio"] == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    x = 0.01 * rng.random(600)
    x[300] = 5.0
    sp = FL.spikes(x, 0.01)
    assert len(sp) == 1 and sp[0]["peak"] == 300 and sp[0]["recovered"]


# ------------------------------------------------------------------ evidence
class _Lane:
    def __init__(self):
        self.senses = ["vision", "tactile", "audio", "proprioception", "vestibular"]
        self.handles = ["wrist_R", "index_R"]
        self.rows, self.contact, self.eff, self.licenses = {}, {}, {}, []


def test_the_evidence_join_reads_each_prediction_against_its_control():
    lane = _Lane()
    q = np.zeros((2, 3, 3), dtype=np.float32)
    q[1, 2, 1] = 0.30                              # body 1, patch 2, the 'self' bit
    base = np.full((3, 3), 0.05, dtype=np.float32)
    lane.contact[10] = dict(q=q, base=base, valid=np.ones(2, dtype=bool))
    exp = np.zeros((2, 2, 3), dtype=np.float32)
    ach = np.zeros((2, 2, 3), dtype=np.float32)
    sh = np.zeros((2, 2, 3), dtype=np.float32)
    ach[1, 1] = [1.0, 0, 0]
    exp[1, 1] = [0.9, 0, 0]
    sh[1, 1] = [0.1, 0, 0]
    lane.eff[25] = dict(began=20, exp_cm=exp, exp_deg=exp, ach_cm=ach, ach_deg=ach, sh_cm=sh, sh_deg=sh)
    prev = np.zeros((2, 2, 3), dtype=np.float32)
    prev[1, 1] = [0.4, 0, 0]                        # the previous order moved the finger 0.4 cm
    lane.eff[20] = dict(began=15, exp_cm=prev, exp_deg=prev, ach_cm=prev, ach_deg=prev, sh_cm=prev, sh_deg=prev)
    header = dict(patch_bodies=[5, 6, 7], entities=[dict(kind="self", group="finger_R_index", bodies=[7]),
                                                    dict(kind="self", group="finger_R_thumb", bodies=[8])])
    chain = [dict(type="latch", step=8, order_seq=4, reached_body=True,
                  plan=dict(pred_wellness_gaps=dict(outcome=["tied", "best"], gap_to_best_alt=[0.0, 0.1],
                                                    tie_band=0.0734)))]
    j = EV.Joiner(lane, chain, lane.handles, header)

    class _Tr:
        step = np.arange(100)
    d = dict(cap="B02", step0=10, step1=12, i0=10, parts=["finger_R_index"], contact_entities=[0, 1],
             author="planner", order_seq=4)
    pred = j.contact_prediction(d, 1, _Tr())
    assert pred["ratio"] == pytest.approx(6.0, rel=1e-3) and pred["verdict"] is True and pred["bit"] == "self"
    m = j.movement_prediction(dict(cap="H05", step0=22, parts=["finger_R_index"]), 1, _Tr())
    assert m["verdict"] is True and m["own_order_error_cm"] < m["shuffled_order_error_cm"]
    assert m["persistence_error_cm"] == pytest.approx(0.6, rel=1e-5)   # |0.4 - 1.0|
    # a learner no better than "keep doing what it did" is not credited
    lane.eff[20]["ach_cm"] = ach.copy()
    j2 = EV.Joiner(lane, chain, lane.handles, header)
    assert j2.movement_prediction(dict(cap="H05", step0=22, parts=["finger_R_index"]), 1, _Tr())["verdict"] is False
    # an act whose order left no planner record gets no intention verdict of another order
    assert j.intention(dict(d, order_seq=5), 1)["verdict"] is None
    assert j.intention(d, 1)["verdict"] is True
    assert j.intention(d, 0)["verdict"] is False and "tied" in j.intention(d, 0)["note"]
    assert j.intention(dict(d, author="reflex_noise"), 1)["verdict"] is False


# ------------------------------------------------------------------ end to end on a real world
def test_a_recorded_crib_world_digests_into_a_report_with_every_catalog_row(tmp_path, monkeypatch):
    """Record a crib world under random orders through the body's own controller, digest it with
    the daemon, and build the lineage report: every capability row is present, the babble acts
    are recorded but never counted, exposure matches the records, and a second digest adds
    nothing (finalized steps are never re-emitted)."""
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.tools.observer import daemon as DM
    run = tmp_path / "H99_test_seg1"
    src = SingleEnvSource(scene="crib", vision=False, age=24, run_dir=str(tmp_path / "scene"), seed=4)
    rng = np.random.default_rng(0)
    try:
        src.observer_open(str(run / "observer" / "world"), run=run.name, chunk_steps=40)
        src.reset()
        H = src.n_handles
        motor = None
        for step in range(130):
            if step % 5 == 0:
                motor = np.zeros((1, H, 9))
                motor[..., 0:3] = np.nan
                q = rng.normal(size=(H, 4))
                motor[0, :, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
                motor[..., 7] = 1.0
                motor[0, :, 8] = rng.uniform(0, 0.3, H)
            for rep in range(5):
                src.step(motor, None, None, render=False,
                         observe=dict(step=step, rep=rep, reps=5, author="random_actions",
                                      latched=step % 5 == 0, order_seq=step // 5, bout=True, rest=False))
    finally:
        src.close()
    monkeypatch.setattr(DM, "SAVED", str(tmp_path / "saved"))
    monkeypatch.setattr(DM, "SEAL_QUIET_S", 0)        # this test seals the moment it finishes recording
    import shutil
    small = tmp_path / "H99_test_seg2"             # the same recording, digested in small windows
    shutil.copytree(run / "observer" / "world", small / "observer" / "world")
    dg = DM.RunDigest(str(run))
    s1 = dg.digest(final=False)
    assert not s1["errors"], s1["errors"]
    assert not dg.sealed, "the active run is never sealed"
    s2 = dg.digest(final=True)
    assert dg.sealed and s2["pending"] == 0, "a finished run with nothing left is sealed"
    s3 = dg.digest(final=True)
    assert s3["detections"] == 0 and s3["episodes"] == 0 and s3["chunks"] == 0, "sealed: never digested again"
    analysis = run / "observer" / "analysis"
    dets = DM.read_jsonl(str(analysis / "detections.jsonl"))
    eps = DM.read_jsonl(str(analysis / "episodes.jsonl"))
    assert dets and eps
    # a window written but not committed (the digester stopped in between) is cut away on restart
    committed = (analysis / "detections.jsonl.gz").stat().st_size
    from mimo_bridge.tools.observer import store as STO
    STO.append_records(str(analysis / "detections.jsonl.gz"), dets[:3], json.dumps)
    again = DM.RunDigest(str(run))
    assert (analysis / "detections.jsonl.gz").stat().st_size == committed
    assert again.sealed and len(DM.read_jsonl(str(analysis / "detections.jsonl"))) == len(dets)
    # small windows: every chunk done, no act emitted twice, the same exposure
    monkeypatch.setattr(DM, "MAX_WINDOW_CHUNKS", 3)     # 40-step chunks: the first window finalizes one
    dg_small = DM.RunDigest(str(small))
    beats = []
    dg_small.beat = lambda r, b: beats.append(b)
    assert not dg_small.digest(final=True)["errors"] and dg_small.sealed
    small_dets = DM.read_jsonl(str(small / "observer" / "analysis" / "detections.jsonl"))
    keys = [(d["cap"], d.get("sub"), d.get("key"), d["step0"]) for d in small_dets]
    assert len(keys) == len(set(keys)), "no act emitted twice across windows"
    assert len(beats) >= 2, "a heartbeat after every window"
    assert sum(r["steps"] for p in dg_small.exposure["0"].values() for r in p.values()) == 130
    assert 0.5 * len(dets) <= len(small_dets) <= 1.5 * len(dets)
    # an analysis of another detector version is moved aside, never mixed
    with open(small / "observer" / "analysis" / "progress.json", encoding="utf-8") as f:
        prog = json.load(f)
    prog["state"]["detector_version"] = "observer_detectors_v0"
    with open(small / "observer" / "analysis" / "progress.json", "w", encoding="utf-8") as f:
        json.dump(prog, f)
    DM.RunDigest(str(small))
    moved = [p for p in (small / "observer").iterdir() if p.name.startswith("analysis_observer_detectors_v0")]
    assert moved and (moved[0] / "moved_aside.json").exists()
    assert not (small / "observer" / "analysis" / "detections.jsonl.gz").exists()
    assert all(d["provenance"] == "baseline" and d["body"] == 0 for d in dets)
    assert len({(d["cap"], d["step0"], d["sub"], d.get("key")) for d in dets}) == len(dets)
    steps = sum(r["steps"] for r in dg.exposure["0"][next(iter(dg.exposure["0"]))].values())
    assert steps == 130
    out = str(tmp_path / "saved" / "observer" / "H99_test" / "development.json")
    rep = DM.build_report([str(run)], out, {str(run): dg}, render_replays=False)
    assert set(rep["capabilities"]) == {c["id"] for c in CAT.load()["capabilities"]}
    assert all(r["counted"] == 0 for r in rep["capabilities"].values()), "babble is never counted"
    assert rep["baseline"]["available"] is False
    with open(out, encoding="utf-8") as f:
        assert json.load(f)["counts"]["detections"] == len(dets)
    # the same report from the lineage store the service keeps
    from mimo_bridge.tools.observer import store as STO
    st = STO.LineageStore(str(tmp_path / "store"), DM.quality_map(CAT.load()))
    st.update(dg.run, dg.out, dg.committed)
    rep2 = DM.build_report([str(run)], out, {str(run): dg}, render_replays=False, store=st)
    assert rep2["counts"] == rep["counts"]
    assert {k: v["counted"] for k, v in rep2["capabilities"].items()} == \
        {k: v["counted"] for k, v in rep["capabilities"].items()}


# ------------------------------------------------------------------ the page's sentences (view.py)
def test_the_page_names_parts_and_tokens_in_words():
    from mimo_bridge.tools.observer import view as V
    assert V.token_words("digit:L:middle:straighten") == "the left middle finger straightens"
    assert V.token_words("touch:foot_L~foot_R") == "the left foot touches the right foot"
    assert V.token_words("touch:arm_R_lower~finger_L_thumb") == "the right forearm touches the left thumb"
    assert V.token_words("approach:R:target") == "the right hand moves toward the ball"
    assert V.token_words("posture:back>side") == "the body moves from its back to its side"
    assert V.token_words("something:new") == "something new"      # an unknown token stays readable


def test_every_fast_lane_count_stands_beside_babble_and_a_limp_body():
    from mimo_bridge.tools.observer import view as V
    total = {"support_drop": dict(samples=95000, spikes=6494, spikes_per_10000=683.6),
             "head_jolt": dict(samples=95000, spikes=4043, spikes_per_10000=425.6, largest=66.75,
                               mean_recovery_s=0.031)}
    chance = {"random_orders": {"support_drop": dict(spikes_per_10000=837.0), "head_jolt": dict(spikes_per_10000=708.0)},
              "limp": {"support_drop": dict(spikes_per_10000=0.0), "head_jolt": dict(spikes_per_10000=0.0)}}
    lines = V.fast_lines(total, chance, 8)
    jolt = next(line for line in lines if "acceleration" in line)
    assert "426 per 100 s of one body's life" in jolt and "random flailing 708" in jolt and "no commands 0." in jolt
    assert "6.8 g" in jolt
    drop = next(line for line in lines if "support" in line)
    assert "684 per 100 s" in drop and "random flailing 837" in drop
    assert "random flailing not measured" in V.fast_lines(total, {}, 8)[0]
    assert lines[-1].startswith("No goal line is set"), "a floor is not a goal: the page says none is set"


def test_a_stage_cell_says_its_count_against_babbles_in_the_same_time():
    from mimo_bridge.tools.observer import view as V
    st = dict(reached=True, first_step=383823, by="frequency", occasions=209, expected_by_chance=152.73)
    assert V.stage_text("observed_once", st, {})["text"] == ("Passed at step 383,823: more often than random flailing "
                                                             "on both counts (209 occasions; flailing: 153 in the same time)")
    # a size pass names its quantity, its line and the direction that counts
    size = dict(reached=True, first_step=5, by="size", beyond=233, measured=1183, measure="amount_deg",
                direction="higher", beyond_line=34.71)
    assert V.stage_text("observed_once", size, {})["text"] == (
        "Passed at step 5: bigger than random flailing (233 of 1,183 occasions started with an amount over 34.7°, "
        "which only 5 in 100 flailing occasions do)")
    # repeats after a frequency pass says it is the same count, not a second finding
    rep = dict(reached=True, first_step=6, occasions=209, expected_by_chance=152.73)
    assert "the same counts as Seen" in V.stage_text("repeated", rep, {}, dict(stages=dict(observed_once=st)))["text"]
    # more occasions but fewer single acts than babble: the cell says which count fell short
    paused = dict(reached=False, occasions=1025, expected_by_chance=894, acts=2389, expected_acts=3279,
                  babble_occasions=894)
    assert V.stage_text("repeated", paused, {})["text"].startswith(
        "Not yet: 1,025 occasions against flailing's 894 in the same time. The count holding it back is single acts: "
        "2,389 against flailing's 3,279")
    # above babble but short of the line says so; under 3 occasions says 3 are needed
    close = dict(reached=False, occasions=22, expected_by_chance=8.9)
    assert V.stage_text("repeated", close, {})["text"] == ("Not yet: 22 occasions against flailing's 8.9 in the same "
                                                           "time; not past its pass line yet.")
    close.update(babble_occasions=89, p_value=0.01, alpha=0.004)          # with babble's own count: the distance
    assert "passing needs about" in V.stage_text("repeated", close, {})["text"]
    assert V.stage_text("repeated", dict(reached=False, occasions=1), {})["text"] == \
        "Not yet: 1 occasion; passing needs at least 3."
    # no acts is not a missing record; a tied planner makes Chosen unreachable, and the cell says why
    assert V.stage_text("predicted_before_action", dict(reached=False, judged=0), {})["text"] == \
        "Not yet: no acts to judge yet."
    tied = V.stage_text("selected_intentionally", dict(reached=False, decisive=0), {},
                        dict(planner_tied=True, choices_judged=1024, choices_best=0))
    assert tied["text"] == ("Not yet: a choice counts as chosen only when the chosen command's predicted wellness beats "
                            "every alternative's by more than the tie band, the evaluator's own error on a difference "
                            "between two candidates; in the current run that happened in 0 of 1,024 choices. On H84 the "
                            "band has been wider than the candidates' predicted spread in every report window since "
                            "Sep 23, so this count is no evidence about the forward model, and it does not show that "
                            "the model learned no intentional behavior: the planner selects by predicted goal distance, "
                            "recorded beside each act of the hands alone.")
    gen = dict(reached=False, waits_for="repeated", bodies=8, contexts=2)
    assert V.stage_text("generalized", gen, {})["text"] == \
        "Not judged until Repeats passes. On 8 bodies, from 2 starting situations; it would pass today."
    wait = dict(reached=False, waits_for="repeated", predicted=47, judged=60)
    assert V.stage_text("predicted_before_action", wait, {})["text"].startswith(
        "Not judged until Repeats passes. 47 of 60 occasions predicted; it would not pass today; passing needs about")
    # a touch row carries its measured chance: the same verdict at moments with no touch starting
    touch = dict(reached=False, predicted=162, judged=208, null_rate=0.47, null_moments=2990)
    assert V.stage_text("predicted_before_action", touch, {})["text"].startswith(
        "Not yet: 162 of 208 occasions predicted; with no touch starting, the same test passes 47 in 100; passing needs")
    assert V.stage_text("repeated", dict(reached=None), {})["state"] == "unmeasured"


def test_the_headline_says_simulated_life_is_summed_over_the_bodies(tmp_path):
    from mimo_bridge.tools.observer import view as V
    saved = tmp_path / "Saved" / "mimo_bridge"
    (saved / "observer" / "H99").mkdir(parents=True)
    run = saved / "experiments" / "H99_seg1"
    (saved / "training_continuation.json").write_text(json.dumps(dict(active_run=str(run))), encoding="utf-8")
    rep = dict(capabilities={}, exposure={"0": dict(steps=9600), "1": dict(steps=9600)},
               baseline=dict(available=True, steps=20000), counts=dict(live_detections=0))
    (saved / "observer" / "H99" / "development.json").write_text(json.dumps(rep), encoding="utf-8")
    page = V.build(str(tmp_path))
    assert "16 minutes of simulated life summed over 2 bodies (8 minutes each)" in page["headline"]


def test_a_stage_that_passed_and_later_fails_is_recorded_as_lost(tmp_path):
    """Each report recomputes the stages; the history keeps a loss visible until the stage returns,
    and a new detector version is a fresh start, not a loss."""
    from mimo_bridge.tools.observer import daemon as DM

    def rows(reached):
        return {"G01": dict(name="Independent eye movement", stages=dict(
            repeated=dict(reached=reached, first_step=100 if reached else None, p_value=0.01, alpha=0.02)))}
    assert DM.stage_history(str(tmp_path), rows(True), 150) == []
    lost = DM.stage_history(str(tmp_path), rows(False), 300)
    assert lost == [dict(step=300, cap="G01", stage="repeated")]
    assert DM.stage_history(str(tmp_path), rows(False), 400) == lost, "lost once, dated once"
    assert DM.stage_history(str(tmp_path), rows(True), 500) == []
    with open(tmp_path / "stage_history.jsonl", encoding="utf-8") as f:
        changes = [json.loads(line)["change"] for line in f]
    assert changes == ["lost", "back"]
    # another detector version or new counting rules: the old verdicts are not the new rule's losses,
    # and the history says the record restarted
    state = json.loads((tmp_path / "stage_state.json").read_text(encoding="utf-8"))
    state["version"] = "observer_analysis_v0/observer_detectors_v0/tracker_rules_v1/"
    (tmp_path / "stage_state.json").write_text(json.dumps(state), encoding="utf-8")
    assert DM.stage_history(str(tmp_path), rows(False), 600) == []
    with open(tmp_path / "stage_history.jsonl", encoding="utf-8") as f:
        assert json.loads(f.readlines()[-1])["change"] == "reset"


def test_the_page_keeps_detected_apart_from_beyond_babble_and_names_sub_rows_in_words():
    from mimo_bridge.tools.observer import view as V
    row = dict(id="H01", name="Whole-hand close after touch", counted=34, exposure_steps=22000,
               baseline_rate_per_1000_steps=1.0, stages=dict(observed_once=dict(reached=False)))
    v = V.row_view(row, {})
    assert v["summary"].startswith("Detected 34 times") and "random flailing would produce about 22" in v["summary"]
    eyes = V.row_view(dict(id="G01", name="Independent eye movement", counted=3, exposure_steps=1000, scripted=1727,
                           stages={}), {})
    assert "world's own eye movers" in eyes["notes"][0] and "toy" not in eyes["notes"][0]
    assert V.sub_label("B02", "finger_L_index+finger_L_thumb") == "left index finger and left thumb"
    assert V.sub_label("G01", "L") == "left eye" and V.sub_label("P06", "toward_R") == "toward the right"
    assert V.sub_label("G02", "art") == "painted wall panels" and V.sub_label("P05", "L:arm") == "left arm"
    assert V.measure_words("min_distance_cm") == ("minimum distance", " cm")


def test_the_milestone_line_names_the_closest_behaviour_and_what_it_lacks():
    """The outside review's milestone: beyond babble, repeats, predicted, chosen - all four at once."""
    from mimo_bridge.tools.observer import view as V
    reached = dict(reached=True)
    caps = {"H05": dict(name="Individual finger isolation", stages=dict(observed_once=reached, repeated=reached),
                        subs={"L:thumb": dict(stages=dict(observed_once=reached, repeated=reached,
                                                          predicted_before_action=reached))}),
            "G01": dict(name="Independent eye movement", stages=dict(observed_once=reached))}
    m = V.milestone(caps)
    assert m["done"] == [] and m["best"] == 3
    line = V.milestone_line(m)
    assert "Individual finger isolation (left thumb)" in line and "missing: Chosen" in line
    caps["H05"]["subs"]["L:thumb"]["stages"]["selected_intentionally"] = reached
    assert "passes all four" in V.milestone_line(V.milestone(caps))


def test_a_run_its_recorders_wrote_recently_is_not_sealed_and_new_files_reopen_a_sealed_one(tmp_path, monkeypatch):
    """Segment 39's first run was sealed at 3:38 PM PDT on 2026-09-24 with 0 chunks, two minutes after
    it started, so its steps never reached the report: a seal needs a quiet recorder, and a seal the
    recorders break by writing again is lifted."""
    import time
    from mimo_bridge.tools.observer import daemon as DM
    run = tmp_path / "H99_test_seg1"
    body = run / "observer" / "world" / "b0"
    body.mkdir(parents=True)
    (body / "header_1_1.json").write_text("{}", encoding="utf-8")
    dg = DM.RunDigest(str(run))
    dg.digest(final=True)
    assert not dg.sealed, "a run written seconds ago was sealed"
    monkeypatch.setattr(DM, "SEAL_QUIET_S", 0)
    old = time.time() - 5
    os.utime(body / "header_1_1.json", (old, old))
    dg.digest(final=True)
    assert dg.sealed
    assert not dg.reopen_if_written(), "nothing new was written"
    (body / "status_1.json").write_text("{}", encoding="utf-8")
    assert dg.reopen_if_written() and not dg.sealed, "a file written after the seal must lift it"
