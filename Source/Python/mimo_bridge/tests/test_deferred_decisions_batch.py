# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The deferred-decisions batch (operator-ordered 2026-09-02): items 1, 3, 4, 5, 6.

1  an order id stamps every latched order - a bit-identically re-issued order still
   ends the previous tenure (float equality could not see it); the ledger carries it.
3  the latent regularizer's running statistics ride runtime_state.pt across resumes.
4  the stats fields mean what their names say: tenure fields are per-EVENT means,
   agency/td_gate/causal_pe/curiosity/imagination are window means, train_mean_pe is
   real-sources-only; and the dashboard renders the action-knowledge gauges.
5  body-schema EMA evidence rides runtime_state.pt (when the feature is on).
6  defaults.py's source scales match master_config.yaml (the yaml always won when
   present; a yaml-less loader would have trained against noise at 0.1).
"""
import inspect
import os

import numpy as np
import pytest
import torch

from mimo_bridge.lem_model import LemLite
from mimo_bridge import lem_model as LM
from mimo_bridge import train_mimo as TM
from mimo_bridge.transition import TransitionV1, Pending


def _model(**kw):
    kw.setdefault("next_frame_loss", True)
    return LemLite(device="cpu", seed=0, imagination=False,
                   action_tenure_loss=True, action_tenure_token_reach=True,
                   agency_drive=True, intention=True, intention_execute=False,
                   overrides={"training": {"steps_per_phase": 3000}}, **kw)


def _constant_exec(B=1):
    a = torch.zeros(B, 24, 9)
    a[..., 6] = 1.0     # identity rotation
    a[..., 8] = 0.5     # constant strength
    a[..., 0] = 0.05    # constant position ask
    return a


# ===================================================================== item 1
def test_item1_identical_reissued_order_ends_a_tenure_when_stamped(tmp_path):
    """The exec passed in is bit-identical on every step; only the id changes every
    5 steps. With ids the tenure lesson must fire; without them float equality sees
    one endless order and the lesson must never fire (the documented fallback)."""
    from mimo_bridge.datasource import SingleEnvSource

    def drive(with_ids):
        m = _model()
        src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / f"i1{int(with_ids)}"),
                              tag=f"_i1{int(with_ids)}", vision=False, seed=5)
        try:
            frames = src.reset()
            dev = dict(m.dev_limits(10 ** 9))
            exec_const = _constant_exec()
            fired = 0
            for i in range(16):
                oid = (i // 5) + 1 if with_ids else None
                res = m.step(frames, dev, train=True, step_index=i,
                             executed_action=exec_const.clone(),
                             executed_order_id=oid)
                if "tenure" in res.breakdown:
                    fired += 1
                for _ in range(5):
                    frames = src.step(res.motor_for_action, res.gaze_out, None,
                                      render=False)
            return fired
        finally:
            src.close()

    assert drive(with_ids=True) >= 1, (
        "bit-identical re-issued orders with fresh ids never ended a tenure - the "
        "id path is not wired")
    assert drive(with_ids=False) == 0, (
        "sanity broken: without ids, float equality should merge identical orders "
        "(the documented fallback) - if this fires, the test no longer proves the "
        "id is what made the difference")


def test_item1_the_ledger_carries_the_order_id():
    p = Pending(env_id=0, episode_id=1, frame_t=10, state_t={},
                achieved_t=np.zeros((24, 10)), context_t=None,
                proposed_action=np.zeros((24, 9)))
    tr = p.complete(executed_action=np.zeros((24, 9)), state_t1={}, frame_t1=11,
                    duration_s=0.25, episode_id=1, order_id=7)
    assert tr.order_id == 7
    assert "order_id" in {f.name for f in TransitionV1.__dataclass_fields__.values()}


def test_item1_trainer_and_battery_stamp_orders():
    src = inspect.getsource(TM)
    assert "order_seq += 1" in src and "prev_order_id = order_seq" in src
    assert "executed_order_id=prev_order_id" in src
    assert "order_id=prev_order_id" in src            # into the ledger
    # teleports drop the id with the action (both reset paths)
    assert src.count("prev_order_id = None") >= 3     # init + 2 teleport paths
    from mimo_bridge.tools import causal_test as CT
    csrc = inspect.getsource(CT._collect)
    assert "executed_order_id=prev_oid" in csrc and "order_seq += 1" in csrc


def test_item1_reset_clears_the_id():
    m = _model()
    m._tenure_prev_order_id = 42
    m.reset_transition_state()
    assert m._tenure_prev_order_id is None
    m._tenure_prev_order_id = 42
    m.reset_context(2)
    assert m._tenure_prev_order_id is None


# ===================================================================== items 3 + 5
def test_item3_running_statistics_ride_the_checkpoint(tmp_path):
    m = _model(checkpoint_dir=str(tmp_path / "ck"))
    assert m.latent_regularizer is not None, "this test needs the regularizer on"
    m.latent_regularizer.running_mean = torch.full_like(
        m.latent_regularizer.running_mean, 0.25)
    m.latent_regularizer.running_var = torch.full_like(
        m.latent_regularizer.running_var, 2.0)
    m.latent_regularizer._initialized = True
    m._vis_adv_ema = 0.0123
    m._agency_pay_ema = torch.tensor([0.5])
    p = m.save_checkpoint(4321)
    assert os.path.isfile(os.path.join(p, "runtime_state.pt"))
    m2 = _model(checkpoint_dir=str(tmp_path / "ck"))
    m2.load_checkpoint(p, load_optimizer=False)
    assert float(m2.latent_regularizer.running_mean.mean()) == pytest.approx(0.25)
    assert float(m2.latent_regularizer.running_var.mean()) == pytest.approx(2.0)
    assert m2.latent_regularizer._initialized is True
    assert m2._vis_adv_ema == pytest.approx(0.0123)
    assert torch.allclose(m2._agency_pay_ema.cpu(), torch.tensor([0.5]))


def test_item3_legacy_json_still_loads(tmp_path):
    import json as _json
    m = _model(checkpoint_dir=str(tmp_path / "ckL"))
    p = m.save_checkpoint(99)
    os.remove(os.path.join(p, "runtime_state.pt"))
    receipt_path=os.path.join(p,'model_receipt.json')
    with open(receipt_path) as f:receipt=_json.load(f)
    receipt.pop('runtime_state_required',None)  # a genuine older checkpoint lacked this contract
    with open(receipt_path,'w') as f:_json.dump(receipt,f)
    with open(os.path.join(p, "runtime_state.json"), "w") as f:
        _json.dump({"vis_adv_ema": 0.9, "touch_adv_ema": 0.8,
                    "hand_motion_ema": 0.7, "agency_pay_ema": None}, f)
    m2 = _model(checkpoint_dir=str(tmp_path / "ckL"))
    m2.load_checkpoint(p, load_optimizer=False)
    assert m2._vis_adv_ema == pytest.approx(0.9)
    assert m2._touch_adv_ema == pytest.approx(0.8)


def test_item5_body_schema_evidence_rides_when_enabled(tmp_path):
    m = _model(checkpoint_dir=str(tmp_path / "ckB"), body_schema_updates=True)
    with torch.no_grad():
        for prm in m._schemas[0].parameters():
            prm.add_(1.0)
        for buf in m._schemas[0].buffers():
            if buf.is_floating_point():
                buf.add_(1.0)
    ref = {k: v.clone() for k, v in m._schemas[0].state_dict().items()}
    p = m.save_checkpoint(777)
    m2 = _model(checkpoint_dir=str(tmp_path / "ckB"), body_schema_updates=True)
    m2.load_checkpoint(p, load_optimizer=False)
    got = m2._schemas[0].state_dict()
    assert all(torch.allclose(ref[k], got[k].cpu()) for k in ref), (
        "body-schema evidence did not survive the round trip")


def test_item5_disabled_feature_saves_no_schemas(tmp_path):
    m = _model(checkpoint_dir=str(tmp_path / "ckN"))
    p = m.save_checkpoint(11)
    rt = torch.load(os.path.join(p, "runtime_state.pt"), weights_only=False)
    assert "body_schemas" not in rt


# ===================================================================== item 4
def test_item4_producer_semantics_pinned():
    src = inspect.getsource(TM)
    # tenure fields accumulate on EVENT steps only and divide by event counts
    assert 'if "tenure" in res.breakdown:' in src
    assert 'if "tenure_tok" in res.breakdown:' in src
    assert 'v2 / acc["tn_n"]' in src and 'v2 / acc["tt_n"]' in src
    assert 'v2 / n for k2, v2 in acc.items() if k2.startswith("tn_")' not in src
    # snapshots became window means
    assert '"curiosity_drive": res.curiosity_drive' not in src
    assert '"imagination_loss": res.imagination_loss' not in src
    assert 'acc["cur"] / acc["cur_n"]' in src
    assert 'acc["cpe"] / acc["cpe_n"]' in src
    # agency and gate are per-key window means
    assert '"agency": (dict(getattr(model, "last_agency"' not in src
    assert 'agN_' in src and 'gtN_' in src
    # train_mean_pe is real-sources-only; the stopgap duplicate is retired
    assert '"pe_real_mean"' not in src
    assert 'train_mean_pe=sum(acc["pe"].values())' not in src
    i = src.index("train_mean_pe=")
    assert '"real"' in src[i:i + 400]


def test_item4_dashboard_renders_the_action_knowledge_rows():
    from mimo_bridge.tools import dashboard
    # every new row label resolves to a popup (the dashboard's own standing rule)
    for label in ("Order-to-outcome matching (rank-1)",
                  "Hand-regard gate on vision's matching share",
                  "Tenure pose lesson (250 ms order horizon)",
                  "Agency drive payments",
                  "Expectation gate (top-down blend into perception)"):
        assert dashboard.doc_for(label) is not None, f"no popup for: {label}"
    src = inspect.getsource(dashboard.describe_run)
    for key in ("tenure_pe_pose", "rank1", "vis_gate", "agency_g", "td_gate_g",
                "tenure_events", "rank1_win"):
        assert f'info["{key}"]' in src, f"describe_run does not read {key}"


def test_item4_dashboard_rows_survive_an_old_log():
    """Fields guarded on presence: an H61-era record (no agency.motion, no td_gate,
    snapshot semantics) must not crash the renderer's row construction logic."""
    from mimo_bridge.tools import dashboard
    src = inspect.getsource(dashboard)
    i = src.index("action knowledge: what the model knows")
    # widened 2026-09-08: the row block grew past 5000 chars; widened again 2026-09-23
    # (verify-fix-dashboards): the section-11 rewrite of the rows put "if _tg:" at 14,193
    blk = src[i:i + 16000]
    for guard in ('if i.get("rank1") is not None', 'if i.get("vis_gate") is not None',
                  'if i.get("tenure_pe_pose") is not None',
                  '_ag.get("pay") is not None', "if _tg:"):
        assert guard in blk, f"missing presence guard: {guard}"


# ===================================================================== item 6
def test_item6_defaults_match_the_yaml():
    from lem.config.defaults import get_defaults
    d = get_defaults()["training"]["loss"]["source_scales"]
    m = _model()
    y = m.config["training"]["loss"]["source_scales"]
    assert d == y == {"real": 1.0, "synthetic": 0.0, "stale": 0.3}, (
        f"defaults {d} vs resolved yaml {y}: the fallback and the yaml must agree - "
        f"a yaml-less loader trains with whatever the fallback says")


# ===================================================================== gap 1: earned horizon
def _imag_model(**kw):
    kw.setdefault("next_frame_loss", True)
    # a caller-supplied overrides dict wins wholesale (it must then carry its own
    # steps_per_phase) - passing both explicitly was a TypeError
    kw.setdefault("overrides", {"training": {"steps_per_phase": 3000}})
    return LemLite(device="cpu", seed=0, imagination=True,
                   action_tenure_loss=True, action_tenure_token_reach=True,
                   agency_drive=True, intention=True, intention_execute=False, **kw)


def test_gap1_the_license_accumulates_from_the_imagination_chain(tmp_path):
    """Per-depth win-vs-do-nothing EMAs must exist and stay in [0,1] after a real
    drive with the chain on."""
    from mimo_bridge.datasource import SingleEnvSource
    m = _imag_model(plan_horizon="earned")
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "g1"), tag="_g1",
                          vision=False, seed=5)
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9))
        dev["exploration_noise"] = 0.05
        held = prev = None
        for i in range(12):
            res = m.step(frames, dev, train=True, step_index=i, executed_action=prev)
            if i % 5 == 0 or held is None:
                held = res.motor_for_action
            for _ in range(5):
                frames = src.step(held, res.gaze_out, None, render=False)
            prev = held
        assert m._plan_win_ema is not None, "the chain ran but no license EMAs exist"
        assert len(m._plan_win_ema) == m.imagination_steps
        assert all(0.0 <= w <= 1.0 for w in m._plan_win_ema)
        assert m.last_plan_license.get("win_emas") is not None
    finally:
        src.close()


def test_gap1_the_horizon_follows_the_license(tmp_path):
    """With depths 1 and 2 licensed (win EMA > 0.5) and depth 3 not, an earned-mode
    planning event must aim 3 steps ahead (1 + 2 licensed); a fixed-mode model must
    stay at the config constant 1."""
    from mimo_bridge.datasource import SingleEnvSource

    def drive_to_plan(mode):
        m = _imag_model(plan_horizon=mode)
        m._plan_win_ema = [0.9, 0.8, 0.2]
        src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / f"g1h{mode}"),
                              tag=f"_gh{mode[:3]}", vision=False, seed=7)
        try:
            frames = src.reset()
            dev = dict(m.dev_limits(10 ** 9))
            dev["exploration_noise"] = 0.05
            held = prev = None
            for i in range(30):
                res = m.step(frames, dev, train=True, step_index=i, executed_action=prev)
                # the license updates from the live chain would drag the hand-set
                # EMAs; re-pin them so the assertion tests the horizon logic alone
                m._plan_win_ema = [0.9, 0.8, 0.2]
                if m.last_intention is not None:
                    return m.last_intention.get("horizon")
                if i % 5 == 0 or held is None:
                    held = res.motor_for_action
                for _ in range(5):
                    frames = src.step(held, res.gaze_out, None, render=False)
                prev = held
            return None
        finally:
            src.close()

    assert drive_to_plan("earned") == 3
    assert drive_to_plan("fixed") == 1


def test_gap1_license_persists_across_resume(tmp_path):
    m = _imag_model(checkpoint_dir=str(tmp_path / "ckg"))
    m._plan_win_ema = [0.7, 0.6, 0.1]
    p = m.save_checkpoint(555)
    m2 = _imag_model(checkpoint_dir=str(tmp_path / "ckg"))
    m2.load_checkpoint(p, load_optimizer=False)
    assert m2._plan_win_ema == pytest.approx([0.7, 0.6, 0.1])


def _gl_fixture():
    """A model with goal locality on, plus a fabricated 30-entry ring laid out on a
    line: entry j sits at position j (128-dim, one active direction), the body at
    position 29. Entry 0 (far, distance 29^2-scaled) carries the highest wellness;
    entry 25 (near) the best reachable one. The self-measured 14-step reach then
    spans exactly 14 positions, so the ball holds entries 15..29."""
    import torch as th
    m = LemLite(device="cpu", seed=0, imagination=True, intention=True,
                intention_execute=False, goal_locality=True,
                overrides={"training": {"steps_per_phase": 3000}})
    T, B, d = 30, 1, 128
    toks = {mod: th.zeros(T, B, d) for mod in m.active}
    for j in range(T):
        for mod in m.active:
            toks[mod][j, 0, 0] = float(j)
    wl = th.zeros(T, B)
    wl[0, 0] = 10.0          # global best-felt: the FAR moment
    wl[25, 0] = 5.0          # best reachable moment
    cur = {mod: toks[mod][T - 1].clone() for mod in m.active}
    return m, cur, wl, toks


def test_goal_locality_picks_best_felt_within_reach():
    import torch as th
    m, cur, wl, toks = _gl_fixture()
    best_global = wl.argmax(dim=0)
    goal, info = m._gl_select(cur, wl, toks, best_global, step=1000)
    assert goal is not None and not info.get("warming")
    # the far global best (position 0) is outside the self-measured 14-step ball;
    # the pick must be entry 25 (position 25.0 on the line), not 0.0
    assert float(goal["vision"][0, 0]) == 25.0
    assert info["new_picks"] == 1 and info["fallbacks_total"] == 0
    # commitment: an identical second event keeps the held goal, no re-pick
    goal2, info2 = m._gl_select(cur, wl, toks, best_global, step=1005)
    assert float(goal2["vision"][0, 0]) == 25.0 and info2["new_picks"] == 0


def test_goal_locality_escape_and_arrival():
    import torch as th
    m, cur, wl, toks = _gl_fixture()
    best_global = wl.argmax(dim=0)
    m._gl_select(cur, wl, toks, best_global, step=1000)          # holds entry 25
    # the body teleports far from the held goal -> escape -> fresh pick near the
    # new position (the ring still spans 0..29; from position 300 nothing is
    # within reach, so the fallback global argmax fires and is counted)
    far = {mod: toks[mod][0].clone() for mod in m.active}
    for mod in m.active:
        far[mod][0, 0] = 300.0
    _g, info = m._gl_select(far, wl, toks, best_global, step=1010)
    assert info["fallbacks_total"] >= 1
    # arrival: standing ON the held goal (within the one-step line) releases it
    m2, cur2, wl2, toks2 = _gl_fixture()
    m2._gl_select(cur2, wl2, toks2, wl2.argmax(dim=0), step=1000)
    at_goal = {mod: toks2[mod][25].clone() for mod in m2.active}
    _g2, info2 = m2._gl_select(at_goal, wl2, toks2, wl2.argmax(dim=0), step=1005)
    assert info2["arrived_now"] == 1 and info2["arrivals_total"] == 1


def test_gap1_license_migrates_across_chain_depth_change(tmp_path):
    """A resume that deepens the chain keeps the earned depth EMAs and opens the new
    depths at 0; a resume that shallows keeps the surviving prefix. Before this fix the
    loader discarded ANY mismatched list (cold zeros: ~69 consecutive wins per depth to
    re-cross 0.5, paid for a config change). Context 8 on BOTH sides: the lite default
    ctx 4 clamps imagination_steps to 3 (deepening could never happen), and save/load
    must share ctx or the temporal embedding shapes refuse the weights."""
    _ctx = {"model": {"context_length": 8}, "training": {"steps_per_phase": 3000}}
    m = _imag_model(checkpoint_dir=str(tmp_path / "ckm"), overrides=_ctx)
    assert m.imagination_steps == 3
    m._plan_win_ema = [0.7, 0.6, 0.1]
    p = m.save_checkpoint(556)
    deep = _imag_model(checkpoint_dir=str(tmp_path / "ckm"),
                       overrides={"model": {"context_length": 8},
                                  "training": {"steps_per_phase": 3000,
                                               "imagination": {"steps": 6}}})
    assert deep.imagination_steps == 6, "ctx clamp ate the deepening - test is void"
    deep.load_checkpoint(p, load_optimizer=False)
    assert deep._plan_win_ema == pytest.approx([0.7, 0.6, 0.1, 0.0, 0.0, 0.0])
    shallow = _imag_model(checkpoint_dir=str(tmp_path / "ckm"),
                          overrides={"model": {"context_length": 8},
                                     "training": {"steps_per_phase": 3000,
                                                  "imagination": {"steps": 2}}})
    shallow.load_checkpoint(p, load_optimizer=False)
    assert shallow._plan_win_ema == pytest.approx([0.7, 0.6])


def test_gap1_pure_out_excludes_the_latent_penalty():
    from lem import live_test as LT
    src = inspect.getsource(LT._run_imagination_compiled)
    # anchor on the FLOAT append: a None slot-holder append now precedes it, and the
    # guarantee guarded here is about the graded loss value specifically
    i = src.index("pure_out.append(float")
    assert src.index("latent_regularizer is not None") > i, (
        "pure_out must be appended BEFORE the latent penalty joins step_loss")


def test_gap1_flag_and_stats_plumbed():
    src = inspect.getsource(TM)
    assert '"--plan-horizon"' in src and 'plan_horizon=args.plan_horizon' in src
    assert '"plan_license"' in src
    with pytest.raises(ValueError):
        LemLite(device="cpu", seed=0, imagination=False, plan_horizon="sideways")


# ===================================================================== gap 2: the mobile
def test_gap2_scene_gains_the_mobile_only_when_asked():
    from mimo_bridge.env import scene_xml
    on = scene_xml("crib", mobile=True)
    off = scene_xml("crib", mobile=False)
    # a REAL mobile: a spinnable four-arm frame with five dangling shapes
    assert 'name="mobile_swing"' in on and 'name="mobile_spin"' in on
    for el in ("mobile_el_e_g", "mobile_el_w_g", "mobile_el_n_g",
               "mobile_el_s_g", "mobile_el_c_g"):
        assert f'name="{el}"' in on, f"missing element {el}"
    assert 'damping="0.015"' in on                       # top joint, probe-tuned
    assert 'damping="0.010"' in on                       # spin, probe-tuned ambient settle
    assert "mobile" not in off.lower() or 'name="mobile' not in off


def test_gap2_env_reports_strikes_and_speed(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    src = SingleEnvSource(scene="crib", run_dir=str(tmp_path / "mob"), tag="_mob",
                          vision=False, seed=3, mobile=True)
    try:
        frames = src.reset()
        null = np.zeros((24, 9), dtype=np.float32)
        null[:, 6] = 1.0
        seen_speed = rest_contact = 0
        for i in range(30):
            frames = src.step(torch.from_numpy(null).unsqueeze(0))
            t = frames[0].meta.get("task") or {}
            if "mobile_speed_cms" in t:
                seen_speed += 1
            rest_contact += bool(t.get("mobile_touched"))
        assert seen_speed >= 25, "mobile speed meta absent"
        assert rest_contact == 0, "the bob must not touch a limp body at rest"
    finally:
        src.close()


def test_the_reach_gauge_says_how_close_the_hands_came_to_the_mobile(tmp_path):
    """Hand strikes on the mobile read 0 for whole runs, and 0 cannot tell a hand that came
    within a finger's width from one that never left the floor. The gauge is the gap between
    the nearest hand surface and the nearest mobile surface, 0 when they touch. Measured
    2026-09-20 in a fresh crib scene: 0.190 m with the body at rest, 0.052 m at the best of
    172 commanded postures, and the lowest mobile part hangs 0.200 m above the floor."""
    from mimo_bridge.datasource import SingleEnvSource
    src = SingleEnvSource(scene="crib", run_dir=str(tmp_path / "reach"), tag="_reach",
                          vision=False, seed=3, mobile=True)
    try:
        frames = src.reset()
        null = np.zeros((24, 9), dtype=np.float32)
        null[:, 6] = 1.0
        frames = src.step(torch.from_numpy(null).unsqueeze(0))
        t = frames[0].meta.get("task") or {}
        gap = t.get("mobile_hand_gap_m")
        assert gap is not None, "the reach gauge is missing from the task report"
        assert 0.0 < gap < 1.5, gap                      # a real separation, in metres
        assert not t.get("mobile_touched")                # and nothing is touching at rest
        env = src.env                                     # carry a hand up to an element
        m, d = env.model, env.data
        hand = env._hand_geoms[0]
        d.qpos[:3] += d.geom_xpos[m.geom("mobile_el_c_g").id] - d.geom_xpos[hand]
        env._mj.mj_forward(m, d)
        close = env._collect().info['task']['mobile_hand_gap_m']
        assert close < gap, (close, gap)
        assert close == 0.0, close                        # surfaces overlapping reads zero
    finally:
        src.close()


def test_the_closest_approach_gauge_starts_each_window_empty():
    """The window accumulator is reset by zeroing every number in it, and min(0.0, x) is
    0.0 - so a closest-approach gauge left at 0.0 would report a touch in every window
    after the first one that saw one. It starts at infinity instead, and the stats row
    prints nothing when no gap was measured. Measured on the live run 2026-09-20: the
    first window after the gauge shipped read a closest approach of 0.000 m with a hand
    strike rate of 0.0017, and the window mean was 0.168 m."""
    src = inspect.getsource(TM)
    assert 'acc["mobile_gap_min"] = float("inf")' in src, "the reset leaves it at 0.0"
    assert 'mobile_gap_min=float("inf")' in src, "it does not start empty either"
    assert 'np.isfinite(acc.get("mobile_gap_min", np.inf))' in src, "an empty window must print nothing"


def test_gap2_trainer_plumbs_and_reports():
    src = inspect.getsource(TM)
    assert '"--mobile"' in src
    assert 'mobile=args.mobile == "on"' in src
    for key in ('"mobile_speed"', '"mobile_hit"', '"mobile_hand"'):
        assert key in src, f"trainer drops {key}"


def test_gap3_dashboard_shows_the_license():
    from mimo_bridge.tools import dashboard
    assert dashboard.doc_for("Plan horizon license (earned lookahead)") is not None
    src = inspect.getsource(dashboard.describe_run)
    assert 'info["plan_lic"]' in src


# ===================================================================== vision gate widening
def test_selfmotion_gate_opens_on_a_moving_mobile_without_the_hand():
    """H64 measured the hand-only gate open on 2.75% of tenure events while the
    struck mobile swung in view all night. In self-motion mode the cached
    visibility flag must go true for a body whose mobile moves above the 2 cm/s
    ambient floor even with the hand out of view; in hand mode it must not."""
    for mode, expect in (("self-motion", True), ("hand", False)):
        m = LemLite(device="cpu", seed=0, imagination=False, next_frame_loss=True,
                    action_tenure_loss=True, action_tenure_token_reach=True,
                    vision_gate=mode,
                    overrides={"training": {"steps_per_phase": 3000}})

        class _F:
            meta = {"task": {"hand_in_view": False, "mobile_speed_cms": 10.0}}

        flags = [bool((f.meta.get("task") or {}).get("hand_in_view"))
                 or (m.vision_gate_mode == "self-motion"
                     and float((f.meta.get("task") or {}).get("mobile_speed_cms") or 0.0)
                     > m._selfmotion_floor_cms)
                 for f in [_F()]]
        assert flags[0] is expect
    with pytest.raises(ValueError):
        LemLite(device="cpu", seed=0, imagination=False, vision_gate="always")


def test_selfmotion_gate_still_blocks_still_mobile_and_no_hand():
    """The H55 protection survives: no hand, mobile at ambient drift -> no vision."""
    m = LemLite(device="cpu", seed=0, imagination=False, next_frame_loss=True,
                action_tenure_loss=True, action_tenure_token_reach=True,
                vision_gate="self-motion",
                overrides={"training": {"steps_per_phase": 3000}})
    t = {"hand_in_view": False, "mobile_speed_cms": 1.4}   # measured ambient
    open_ = bool(t.get("hand_in_view")) or float(t.get("mobile_speed_cms")) > m._selfmotion_floor_cms
    assert open_ is False
    src = inspect.getsource(LM.LemLite.step)
    assert 'self.vision_gate_mode == "self-motion"' in src
    tsrc = inspect.getsource(TM)
    assert '"--vision-gate"' in tsrc and "vision_gate=args.vision_gate" in tsrc


# ===================================================================== situation test
def test_situation_tool_measures_before_it_claims():
    """The situation-dependence test must carry its own floors: determinism (T1),
    slot invariance (T2), and the self-measured jitter floor (T4) that the
    relevant change (T3) must clear with a bootstrap interval - no chosen
    thresholds anywhere, and the caregiver is disabled so only the TEST moves
    the ball."""
    from mimo_bridge.tools import situation_test as ST
    src = inspect.getsource(ST)
    for marker in ("t1_determinism_floor", "t2_slot_effect",
                   "t3_relevant_change_cmd_delta", "t4_irrelevant_change_cmd_delta",
                   "np.percentile(ratios, [2.5, 97.5])",
                   'caregiver_mode="off", caregiver_seconds=0.0',
                   "randomize_target()"):
        assert marker in src, f"missing: {marker}"
    assert "motor_out" in src, "the measured command must be the pre-babble proposal"
    assert 'lo > 1.0' in src, "established requires the CI clear of 1.0x"


def test_situation_cmd_is_deterministic_and_slot_invariant():
    """T1/T2 on a fresh tiny model with synthetic situations: identical inputs give
    identical commands, and two situations swapped between batch slots carry
    their commands with them."""
    from mimo_bridge.tools.situation_test import _cmd
    m = LemLite(device="cpu", seed=0, imagination=False, next_frame_loss=True,
                overrides={"training": {"steps_per_phase": 3000}})
    g = torch.Generator().manual_seed(4)
    def situation(seed_shift):
        toks = {mm: torch.randn(1, 1, m.d_model, generator=g) for mm in m.active}
        ctx = torch.randn(1, (m.ctx_len - 1) * m.n_mod, m.d_model, generator=g)
        hs = torch.randn(1, 24, 10, generator=g) * 0.1
        return toks, ctx, hs
    ta, ca, ha = situation(0)
    tb, cb, hb = situation(1)
    c1 = _cmd(m, ta, ca, ha)
    c2 = _cmd(m, ta, ca, ha)
    assert torch.equal(c1, c2), "identical cached inputs must give identical commands"
    tok_ab = {k: torch.cat([ta[k], tb[k]]) for k in ta}
    tok_ba = {k: torch.cat([tb[k], ta[k]]) for k in ta}
    o_ab = _cmd(m, tok_ab, torch.cat([ca, cb]), torch.cat([ha, hb]))
    o_ba = _cmd(m, tok_ba, torch.cat([cb, ca]), torch.cat([hb, ha]))
    assert torch.allclose(o_ab[0], o_ba[1], atol=1e-5), "command did not travel with its situation"
    assert torch.allclose(o_ab[1], o_ba[0], atol=1e-5)
    assert not torch.allclose(o_ab[0], o_ab[1], atol=1e-4), (
        "two different situations gave the same command - the test would be vacuous")


def test_overseer_cites_the_situation_test():
    from mimo_bridge.tools import overseer as OV
    src = inspect.getsource(OV)
    assert "situation_test.json" in src
    assert "situation dependence untested" in src
    assert src.count("(situation dependence is not established)") == 1, (
        "the unconditional assertion must survive only in the historical comment")


def test_situation_capture_takes_numpy_commands():
    """The bridge's StepResult.motor_out is numpy; capture crashed on .detach()."""
    from mimo_bridge.tools import situation_test as ST
    src = inspect.getsource(ST)
    assert "torch.as_tensor(np.asarray(res.motor_out)" in src
    assert "res.motor_out.detach()" not in src


def test_battery_context_snapshot_survives_the_first_step():
    """E12 moved the snapshot before step(); a fresh model's buffer is None there
    (S_0's context is zeros). The battery crashed on its first transition."""
    from mimo_bridge.tools import causal_test as CT
    src = inspect.getsource(CT._collect)
    assert "if model.context_buffer is not None" in src
    m = LemLite(device="cpu", seed=0, imagination=False, next_frame_loss=True,
                overrides={"training": {"steps_per_phase": 3000}})
    assert m.context_buffer is None
    ctx = (m.context_buffer if m.context_buffer is not None
           else torch.zeros(1, (m.ctx_len - 1) * m.n_mod, m.d_model))
    assert ctx.shape == (1, (m.ctx_len - 1) * m.n_mod, m.d_model)
