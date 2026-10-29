# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The wellness channels inside the live model and the trainer (DESIGN.md D6-D9).

1. With the measurement-only flags on (--controllability, --learning-progress under the raw
   ranking), no order, loss, wellness value, weight or random stream changes: they measure.
2. Under --wellness-rank normalized the goal is the best NORMALISED moment and every block reads
   sane values: variance shares adding to 1, controllability and learning progress in [-1, 1].
3. The trainer writes the blocks only when asked (one-body micro-run, the form
   test_causal_chain._micro_run established).
Slow: real bodies on the CPU."""
import json
import math
import os

import numpy as np
import pytest
import torch


def _rng():
    return torch.get_rng_state().clone(), np.random.get_state()


def _set_rng(s):
    torch.set_rng_state(s[0])
    np.random.set_state(s[1])


def _build(**kw):
    from mimo_bridge.lem_model import LemLite
    return LemLite(device="cpu", seed=0, intention=True, intention_every=5, imagination=False,
                   no_vision=True, next_frame_loss=True, smooth_actions=0.35,
                   consequence_weight=0.5, curiosity_goal=True,
                   overrides={"training": {"steps_per_phase": 3000}}, **kw)


@pytest.mark.slow
def test_the_measurement_flags_change_no_order_loss_or_weight(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    torch.set_num_threads(1)
    a = _build()
    b = _build(controllability=True, learning_progress=True)
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "wc"), tag="_wc", vision=False, seed=5)
    try:
        frames = src.reset()
        dev = dict(a.dev_limits(10 ** 9))
        dev["exploration_noise"] = 0.05
        prev, prev_id, held, seq = None, None, None, 0
        for i in range(27):
            s0 = _rng()
            ra = a.step(frames, dev, train=True, step_index=i, executed_action=prev,
                        executed_order_id=prev_id)
            sa = _rng()
            _set_rng(s0)
            rb = b.step(frames, dev, train=True, step_index=i, executed_action=prev,
                        executed_order_id=prev_id)
            sb = _rng()
            assert torch.equal(sa[0], sb[0]), f"step {i}: a global random stream moved"
            assert np.array_equal(ra.motor_for_action, rb.motor_for_action), f"step {i}: order changed"
            assert ra.loss == rb.loss and ra.wellness == rb.wellness, f"step {i}"
            if i % 5 == 0 or held is None:
                held, seq = ra.motor_for_action, seq + 1
            frames = src.step(held, ra.gaze_out, None, render=False)
            prev, prev_id = held, seq
        for pa, pb in zip(a.params, b.params):
            assert torch.equal(pa, pb), "a measurement flag changed a weight"
    finally:
        src.close()
    c = b.controllability_window_summary()
    assert c["orders"] >= 4 and c["source"] == "command"
    assert c["measured_share"] is not None and 0.0 <= c["measured_share"] <= 1.0
    if c["agreement_mean"] is not None:
        assert -1.0 <= c["agreement_mean"] <= 1.0
    lp = b.learning_progress_window_summary()
    assert set(lp["per_sense"]) >= {"proprioception", "tactile"}
    assert lp["value_mean"] is not None and -1.0 <= lp["value_mean"] <= 1.0
    assert a.controllability_window_summary() is None and a.wellness_rank_window_summary() is None


@pytest.mark.slow
def test_the_normalised_ranking_chooses_goals_and_reports_sane_blocks(tmp_path):
    from lem.model import wellness_channels as WC
    from mimo_bridge.datasource import SingleEnvSource
    torch.set_num_threads(1)
    m = _build(controllability=True, learning_progress=True, wellness_rank="normalized",
               curiosity_mode="additive", consequence_self_contact="exclude")
    assert m.rank_channel_names[-2:] == [WC.PAIN_ROW, WC.HARM_ROW]
    assert {"integrity", "controllability", "learning_progress", "consequence",
            "curiosity"} <= set(m.rank_channel_names)
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "wn"), tag="_wn", vision=False, seed=7)
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9))
        dev["exploration_noise"] = 0.05
        prev, prev_id, held, seq, plans = None, None, None, 0, 0
        for i in range(36):
            r = m.step(frames, dev, train=False, step_index=i, executed_action=prev,
                       executed_order_id=prev_id)
            if m.last_intention is not None and m.last_intention.get("step") == i:
                plans += 1
                # the goal's place in the ring rides the intention record (the explainer reads it)
                gi, ga = m.last_intention["goal_ring_index"], m.last_intention["goal_age_steps"]
                assert len(gi) == 1 and 0 <= gi[0] < len(m.goal_ring)
                assert ga[0] == len(m.goal_ring) - 1 - gi[0]
            assert len(m._goal_ring_ch) == len(m.goal_ring)
            if i % 5 == 0 or held is None:
                held, seq = r.motor_for_action, seq + 1
            frames = src.step(held, r.gaze_out, None, render=False)
            prev, prev_id = held, seq
    finally:
        src.close()
    assert plans >= 2
    w = m.wellness_rank_window_summary()
    assert w["planning_events"] >= 2
    vs = w["variance_share"]
    assert vs is not None and abs(sum(vs.values()) - 1.0) < 1e-3
    assert abs(sum(w["winner_margin_share"].values()) - 1.0) < 1e-3
    assert 0.0 <= w["integrity_gated_share"] <= 1.0
    assert set(w["raw"]) >= {"integrity", "pain", "harm", "consequence"}
    assert w["goal_age"]["events"] == plans and 0.0 <= w["goal_age"]["mean_share_back"] <= 1.0
    assert abs(sum(w["weight_share_by_channel"].values()) - 1.0) < 1e-3
    assert w["weight_share_by_channel"]["controllability"] == 0.0, "its gate is shut"
    assert w["detrend_steps"] == WC.RING_DETREND_STEPS
    assert all(math.isfinite(v) for v in w["scale"].values() if v is not None)
    assert "controllability" in w["planner_blind"] and "integrity" in w["planner_partly"]
    assert m.last_consequence.get("world_only") == 1.0
    # the stored raw value is untouched by the ranking: still the plain sum plus drives
    assert all(torch.isfinite(v).all() for _, v in m.goal_ring)
    # the episode companion carries the new working state (capture raises on anything it cannot
    # serialise), and a ring restored WITHOUT channel records restarts instead of misaligning
    from mimo_bridge import continuation_state as CS
    state = CS.capture_learner(m, include_replay=False)
    assert len(state["data"]["_goal_ring_ch"]) == len(m.goal_ring)
    state["data"].pop("_goal_ring_ch")            # a companion saved before this code
    m2 = _build(controllability=True, learning_progress=True, wellness_rank="normalized",
                curiosity_mode="additive", consequence_self_contact="exclude")
    CS.restore_learner(m2, state)
    assert len(m2.goal_ring) == len(m.goal_ring) and len(m2._goal_ring_ch) == 0
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "wn2"), tag="_wn2", vision=False, seed=8)
    try:
        f2 = src.reset()
        m2.step(f2, dev, train=False, step_index=100, executed_action=None)
    finally:
        src.close()
    assert m2._rank_ring_cleared == 1 and len(m2.goal_ring) == 1 == len(m2._goal_ring_ch)


def test_the_bad_combinations_are_refused():
    with pytest.raises(ValueError, match="curiosity-mode additive"):
        _build(wellness_rank="normalized")
    with pytest.raises(ValueError, match="intention"):
        from mimo_bridge.lem_model import LemLite
        LemLite(device="cpu", no_vision=True, wellness_rank="normalized")
    with pytest.raises(ValueError):
        _build(wellness_rank="sorted")


def _micro_run(tmp_path, name, extra):
    from mimo_bridge import train_mimo as TM
    out = str(tmp_path / name)
    os.makedirs(out, exist_ok=True)
    argv = ["--n-envs", "1", "--no-vision", "--minutes", "5", "--scene", "reach", "--out", out,
            "--video-every-min", "0", "--replay", "off", "--curriculum", "on",
            "--next-frame-loss", "on", "--intention", "on", "--intention-every", "5",
            "--report-every", "10", "--max-steps", "31", "--consequence-w", "0.5"] + extra
    assert TM.main(argv) == 0, "trainer micro-run failed"
    recs = [json.loads(x) for x in open(os.path.join(out, "replay_training_stats.jsonl"),
                                        encoding="utf-8") if x.strip()]
    return out, [r for r in recs if r.get("type") == "mimo"]


@pytest.mark.slow
def test_the_trainer_writes_the_blocks_only_when_asked(tmp_path):
    _, mimo = _micro_run(tmp_path, "on", ["--controllability", "on", "--learning-progress", "on",
                                          "--wellness-rank", "normalized", "--curiosity-mode",
                                          "additive", "--consequence-self-contact", "exclude"])
    assert len(mimo) >= 3
    last = mimo[-1]
    assert "controllability" in last and "learning_progress" in last and "wellness_rank" in last
    assert last["wellness_rank"]["mode"] == "normalized"
    assert any(r["wellness_rank"]["planning_events"] for r in mimo)
    assert last["consequence"]["world_only"] == 1.0
    assert len(json.dumps(last)) < 200_000
    # the channels' own time rides the machine block, so a proof can compare flags on and off
    assert last["machine"]["wellness_channels_ms"] is not None
    assert last["machine"]["wellness_channels_ms"] <= last["machine"]["model_ms"]
    assert any(r["wellness_rank"]["goal_age"] for r in mimo)
    _, off = _micro_run(tmp_path, "off", [])
    for k in ("controllability", "learning_progress", "wellness_rank"):
        assert k not in off[-1]
    assert "world_only" not in (off[-1].get("consequence") or {})
    assert "wellness_channels_ms" not in off[-1]["machine"]


# ============================================================================ lane-b3-wellness
_ALL_ON = dict(controllability=True, learning_progress=True, wellness_rank="normalized",
               curiosity_mode="additive", consequence_self_contact="exclude")


def _drive(m, frames_fn, n, start=0, train=False, hold=5, srcs=None, dev=None):
    """Step a model n times on the given sources, latching an order every `hold` steps."""
    prev = prev_id = held = None
    seq = 0
    frames = frames_fn()
    for i in range(start, start + n):
        r = m.step(frames, dev, train=train, step_index=i, executed_action=prev,
                   executed_order_id=prev_id)
        if i % hold == 0 or held is None:
            held, seq = r.motor_for_action, seq + 1
        gz = np.asarray(r.gaze_out)
        frames = [s.step(held[j:j + 1], gz[j:j + 1], None, render=False)[0]
                  for j, s in enumerate(srcs)]
        prev, prev_id = held, seq
    return frames


@pytest.mark.slow
def test_a_resume_into_a_build_with_another_channel_list_restarts_the_ring_and_steps(tmp_path):
    """review of b3, major 2: the channel list rode the episode companion and overwrote the list
    the run's flags set; restoring a list that named controllability into a build without it
    failed at the first step. Now the list is public (never captured) and the ring carries the
    list its rows are in, so a resume into another list restarts the ring and steps."""
    from lem.model import wellness_channels as WC
    from mimo_bridge import continuation_state as CS
    from mimo_bridge.datasource import SingleEnvSource
    torch.set_num_threads(1)
    m = _build(**_ALL_ON)
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "a"), tag="_la", vision=False, seed=11)
    try:
        dev = dict(m.dev_limits(10 ** 9))
        _drive(m, src.reset, 26, srcs=[src], dev=dev)
    finally:
        src.close()
    state = CS.capture_learner(m, include_replay=False)
    assert "_rank_channel_names" not in state["data"] and "rank_channel_names" not in state["data"]
    assert state["data"]["_goal_ring_ch_names"] == m.rank_channel_names
    assert len(state["data"]["_goal_ring_ch"]) == len(m.goal_ring) > 20
    # a companion written by the first b3 build also carried the list: it is ignored
    state["data"]["_rank_channel_names"] = ["integrity", "controllability", "pain"]
    other = _build(wellness_rank="normalized", curiosity_mode="additive",
                   consequence_self_contact="exclude")           # controllability, LP off
    mine = list(other.rank_channel_names)
    assert "controllability" not in mine
    CS.restore_learner(other, state)
    assert other.rank_channel_names == mine and other._rank_channel_names == mine
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "b"), tag="_lb", vision=False, seed=12)
    try:
        f = src.reset()
        other.step(f, dev, train=False, step_index=100, executed_action=None)
    finally:
        src.close()
    assert other._rank_ring_cleared == 1
    assert len(other.goal_ring) == 1 == len(other._goal_ring_ch)
    assert other._goal_ring_ch[-1].shape[0] == len(mine)
    # the other way: a list with MORE channels, restored into the build that has them, keeps its ring
    m2 = _build(**_ALL_ON)
    CS.restore_learner(m2, CS.capture_learner(m, include_replay=False))
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "c"), tag="_lc", vision=False, seed=13)
    try:
        f = src.reset()
        n0 = len(m2.goal_ring)
        m2.step(f, dev, train=False, step_index=101, executed_action=None)
    finally:
        src.close()
    assert m2._rank_ring_cleared == 0 and len(m2.goal_ring) == min(200, n0 + 1)


@pytest.mark.slow
def test_two_bodies_training_pair_every_graded_order_with_the_efference_lessons_record(tmp_path):
    """review of b3, major 3: the pairing of a graded order with the efference head's tenure
    record (its step and the step the order began) was tested only in the tracker's own
    numbering. Two real bodies, training on, the heads on: every order the tracker grades is
    judged against the record _efference_lesson wrote for that same order at that same step, and
    the record carries the head's shuffled-order expectation the control is built from."""
    from mimo_bridge.datasource import SingleEnvSource
    torch.set_num_threads(2)
    m = _build(controllability=True, learning_progress=True, efference_heads=True)
    assert m.wellness_channels.control.expect_efference
    srcs = [SingleEnvSource(scene="reach", run_dir=str(tmp_path / f"e{b}"), tag=f"_le{b}",
                            vision=False, seed=20 + b) for b in range(2)]
    graded = []
    try:
        frames = [s.reset()[0] for s in srcs]
        dev = dict(m.dev_limits(10 ** 9))
        dev["exploration_noise"] = 0.05
        prev = prev_id = held = None
        seq = 0
        for i in range(33):
            r = m.step(frames, dev, train=True, step_index=i, executed_action=prev,
                       executed_order_id=prev_id)
            last = m.wellness_channels.control.last
            if last.get("step") == i:
                eff = m.last_efference_tenure
                graded.append((i, dict(last), None if eff is None else
                               (eff["step"], eff["order_began"],
                                "expected_shuffled_deg" in eff)))
            if i % 5 == 0 or held is None:
                held, seq = r.motor_for_action, seq + 1
            gz = np.asarray(r.gaze_out)
            frames = [s.step(held[j:j + 1], gz[j:j + 1], None, render=False)[0]
                      for j, s in enumerate(srcs)]
            prev, prev_id = held, seq
    finally:
        for s in srcs:
            s.close()
    assert len(graded) >= 4, graded
    for i, last, eff in graded:
        assert last["source"] == "efference"
        assert eff is not None and eff[0] == i and eff[1] == last["order_began"] and eff[2], (i, last, eff)
    c = m.controllability_window_summary()
    assert c["efference_missing"] == 0 and c["source"] == "efference" and c["orders"] == 2 * len(graded)


def test_goal_locality_falls_back_to_the_ranking_when_every_moment_in_reach_is_gated():
    """review of b3, minor: with every moment inside the reach ball over the harm line, the
    largest-value pick over minus infinity returned ring index 0 - the oldest moment, over the
    line and possibly outside the ball. The ball now counts as empty and the ranking's own pick,
    which respects the gate, stands."""
    torch.manual_seed(0)
    m = _build(goal_locality=True, **_ALL_ON)
    T, B = 30, 2
    toks = {k: torch.zeros(T, B, 4) for k in m.active}
    for k in m.active:
        toks[k][:, :, 0] = torch.arange(T, dtype=torch.float32).view(T, 1)   # moment t sits at t
    cur = {k: toks[k][-1].clone() for k in m.active}                        # now = the newest
    wl = torch.zeros(T, B)
    wl[-16:] = float("-inf")               # everything within reach of now is over the harm line
    wl[3] = 1.0                             # the ranking's pick: moment 3, out of reach
    best = wl.argmax(0)
    m._gl_reach_ema = torch.full((B,), 30.0)       # reach: the newest 12 moments (distance t^2/4)
    m._gl_instant_ema = torch.full((B,), 1.0)
    goal, info = m._gl_select(cur, wl, toks, best, step=100, allowed=torch.isfinite(wl))
    assert info["fallbacks_total"] == B
    assert m._gl_goal_index == [3, 3] and m._gl_goal_age == [T - 1 - 3] * B
    for k in m.active:
        assert torch.equal(goal[k], toks[k][3])
    # without the gate's mask (the first build) the pick was an over-the-line moment in reach
    m._gl_clear()
    m._gl_reach_ema = torch.full((B,), 30.0)
    m._gl_instant_ema = torch.full((B,), 1.0)
    goal2, _ = m._gl_select(cur, wl, toks, best, step=100)
    assert m._gl_goal_index == [0, 0], "the first build's pick: the oldest moment, out of reach"


def test_a_missing_touch_reading_is_unmeasured_not_painless():
    """review of b3, minor: _channel_pain returned 0 when no skin features existed, so integrity
    read its best value and the gate could not fire."""
    from lem.model import wellness_channels as WC
    from mimo_bridge.lem_model import LemLite

    class Stub:
        device = torch.device("cpu")
    p = LemLite._channel_pain(Stub(), {}, {}, 3)
    assert p.shape == (3,) and torch.isnan(p).all()
    m = _build(**_ALL_ON)
    m.curiosity_measured = False
    B = 2
    frames = [object()] * B
    g = {k: torch.full((B, 1), 0.5) for k in ("coherence", "prediction_error", "tactile_comfort",
                                               "capability", "motor_state")}
    g["motor_state"] = torch.full((B, 1), 0.98)
    v = m._wellness_channel_vector(B, {"grounded": g}, torch.full((B,), float("nan")), None,
                                   None, torch.zeros(B), 0.0, frames)
    names = m.rank_channel_names
    assert torch.isnan(v[names.index("integrity")]).all()
    assert torch.isnan(v[names.index(WC.HARM_ROW)]).all()
    v2 = m._wellness_channel_vector(B, {"grounded": g}, torch.tensor([0.3, 0.01]), None,
                                    None, torch.zeros(B), 0.0, frames)
    assert v2[names.index(WC.HARM_ROW)].tolist() == pytest.approx([0.3, 0.02], abs=1e-6)
    assert v2[names.index("integrity")].tolist() == pytest.approx([0.7, 0.98], abs=1e-6)


def test_a_load_that_starts_a_network_fresh_holds_its_learning_progress_out():
    """review of b2, finding 5: the keys of every network a load started fresh sit out of learning
    progress for LP_HOLDOUT_STEPS model steps."""
    from lem.model import wellness_channels as WC
    m = _build(learning_progress=True)
    keys = m._lp_hold_out_fresh({"reinitialized_components": ["efference_heads", "encoder:vision",
                                                              "gaze_head"]})
    assert keys == ["efference_body", "efference_gaze"] + (["vision"] if "vision" in m.pred_mods
                                                           else [])
    assert all(m.wellness_channels.lp.holdout[k] == WC.LP_HOLDOUT_STEPS for k in keys)
    assert _build()._lp_hold_out_fresh({"reinitialized_components": ["efference_heads"]}) == []
