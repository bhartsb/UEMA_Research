# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The fixes to the gain-of-function probes from their review (docs/gain_of_function_v3/reviews/
probes.md, 2026-09-23): snapshots of the upgraded model, the planner-off causal test, the
counterfactual binding score, the eye row of every scripted order, goal lines, readable numbers,
the load search, the watcher's memory of where it began, and the retry of failures written by
older code.

Fast tests use small fakes. The tests marked slow build the deploy's full flag set on H84's own
arguments (no checkpoint load) or load ckpt_step_344400. The checkpoint is found wherever the
archiver moved it (tests/saved_checkpoints.py), its run's args.json through its lineage.json, and
the tests are skipped only where they are nowhere."""
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimo_bridge import copied_execution as CE
from mimo_bridge import storage_locations as SL
from mimo_bridge.tests.saved_checkpoints import h84_344400
from mimo_bridge.tools import gof_probe_harness as H
from mimo_bridge.tools.check_no_bare_decimals import bare_decimals

ROOT = Path(__file__).resolve().parents[4]
H84_FOUND = h84_344400()
H84 = H84_FOUND.path
NEEDS_H84 = H84_FOUND.mark()


def _h84_args():
    """(args.json path or None, why not): an archived checkpoint has no args.json beside it, so the
    run's own is found through the checkpoint's lineage.json (storage_locations.resolve_run_args)."""
    if H84 is None:
        return None, H84_FOUND.reason
    try:
        return SL.resolve_run_args(H84), ""
    except FileNotFoundError as exc:
        return None, str(exc)


H84_ARGS, _NO_ARGS = _h84_args()
NEEDS_H84_ARGS = pytest.mark.skipif(H84_ARGS is None, reason="H84 segment 31's args.json: %s" % _NO_ARGS)
DEPLOY = dict(intention="on", causal_chain="on", efference_heads="on", contact_head="on",
              contact_control_wellness=0.5, wellness_rank="normalized", curiosity_mode="additive",
              controllability="on", learning_progress="on", vestibular_sense="on")


# The rest of DESIGN.md section 2's flags, each on at its deploy value when this train_mimo.py parses
# it (verify-fix-probes 2026-09-23: the set above left out --next-frame-norm persistence, whose
# LemLite.nf_norm the snapshot refused by name). Order matters: --synergy-babble needs the synergy.
LATER_FLAGS = (("motor_synergy", "--motor-synergy", "on"), ("synergy_babble", "--synergy-babble", "on"),
               ("smooth_source", "--smooth-source", "executed"),
               ("consequence_self_contact", "--consequence-self-contact", "exclude"),
               ("next_frame_norm", "--next-frame-norm", "persistence"),
               ("liveness_repair", "--liveness-repair", "on"))


def _deploy_flags():
    """The deploy's flag set: DEPLOY plus every flag of LATER_FLAGS this train_mimo.py parses."""
    from mimo_bridge import train_mimo as TM
    flags = dict(DEPLOY)
    for name, flag, value in LATER_FLAGS:
        try:
            TM.parse_args(["--out", "unused", flag, value])
        except SystemExit:
            continue
        if name == "synergy_babble" and flags.get("motor_synergy") != "on":
            continue
        flags[name] = value
    return flags


# ------------------------------------------------------------------ copied_execution
def _contact_tracker():
    from lem.model import contact_head as CH
    t = CH.ContactTracker()
    t.m[:] = np.random.default_rng(0).random(t.m.shape)
    t.w[:] = 1.0
    t.n[:] = 500
    for gate in t.gates.values():
        gate["readings"].extend([0.1, -0.2, 0.3])
    t.live["ctx1"] = torch.tensor([True, False, True])
    t.hist_head["any"].add(np.array([0.2, 0.9]), np.array([False, True]))
    return t


def _wellness_state():
    from lem.model import wellness_channels as WC
    ctl = WC.ControllabilityTracker(4, 3, hold=5, floors=dict(WC.CONTROL_FLOOR))
    ctl.ema_m, ctl.ema_w = torch.tensor([0.1, 0.2]), torch.tensor([0.5, 0.5])
    ctl.gate_readings.extend([0.01, 0.02])
    lp = WC.LearningProgressTracker()
    lp.update("vision", torch.tensor([0.2, 0.3]), torch.tensor([0.4, 0.5]))
    st = WC.WellnessChannelState(True, ctl, lp)
    st.normalizer.update("integrity", torch.tensor([0.9, 0.8]))
    return st


def _next_frame_norm():
    nfn = pytest.importorskip("lem.model.next_frame_norm", reason="b6's --next-frame-norm not in this tree")
    t = nfn.NextFrameNorm()
    t.update({"vision": 0.009, "tactile": 0.025})
    w, info = t.weights(["vision", "tactile"], {"vision": 0.01})
    t.add_live({"vision": 0.01, "tactile": 0.02}, w, info)
    return t


def test_the_next_frame_norm_is_copied_whole_by_a_probe_snapshot():
    """LemLite.nf_norm (--next-frame-norm persistence, DESIGN D12) is a plain object on the model
    under a public name; before 2026-09-23 capture() refused it by name, so no probe could snapshot
    a model built with every deploy flag on (verify-fix-probes, scratch all_flags_snapshot.py)."""
    obj = _next_frame_norm()
    found, saved = CE._capture_special(obj, "nf_norm")
    assert found, "NextFrameNorm is not on copied_execution's list of plain trackers"
    restored = CE._restore_special(None, saved)
    assert restored is not obj and CE.equal_tree(CE.plain_state(restored), CE.plain_state(obj))
    restored.update({"vision": 5.0})
    assert CE.equal_tree(CE.plain_state(CE._restore_special(None, saved)), CE.plain_state(obj))


@pytest.mark.parametrize("make", [_contact_tracker, _wellness_state])
def test_plain_trackers_are_copied_whole_and_restored_as_fresh_copies(make):
    obj = make()
    found, saved = CE._capture_special(obj, "tracker")
    assert found
    before = CE.plain_state(obj)
    restored = CE._restore_special(None, saved)
    assert restored is not obj and CE.equal_tree(CE.plain_state(restored), before)
    # a restored branch that changes its tracker never changes the snapshot
    if hasattr(restored, "gates"):
        restored.gates["self"]["readings"].append(9.0)
    elif hasattr(restored, "gate_readings"):
        restored.gate_readings.append(9.0)
    else:
        restored.normalizer.update("integrity", torch.tensor([0.1]))
    again = CE._restore_special(None, saved)
    assert CE.equal_tree(CE.plain_state(again), before)
    assert not CE.equal_tree(CE.plain_state(restored), before)


def test_a_tracker_holding_a_parameter_or_an_unknown_object_is_refused_by_name():
    obj = _contact_tracker()
    obj.live["ctx1"] = torch.nn.Parameter(torch.zeros(3))
    with pytest.raises(ValueError, match=r"tracker\.live\['ctx1'\]"):
        CE._capture_special(obj, "tracker")
    obj = _contact_tracker()
    obj.helper = object()
    with pytest.raises(ValueError, match=r"tracker\.helper \(object\)"):
        CE._capture_special(obj, "tracker")
    # an unlisted class is not special: capture() then refuses it by its attribute name
    assert CE._capture_special(SimpleNamespace(a=1), "x") == (False, None)


def test_a_generator_is_saved_as_its_state_and_draws_the_same_after_restore():
    g = torch.Generator(device="cpu").manual_seed(7)
    torch.rand(3, generator=g)
    found, saved = CE._capture_special(g, "gen")
    assert found and isinstance(saved, CE._SavedGenerator)
    first = torch.rand(4, generator=g)
    same = CE._restore_special(g, saved)
    assert same is g and torch.equal(torch.rand(4, generator=g), first)
    fresh = CE._restore_special(None, saved)
    assert torch.equal(torch.rand(4, generator=fresh), first)


# ------------------------------------------------------------------ the deploy's full flag set
def test_unmeasured_tracker_state_restores_without_accepting_nan_predictions():
    missing = torch.tensor([1., float('nan')])
    assert not CE.equal_tree(missing, missing.clone())
    assert CE.equal_tree(CE.plain_state(missing), CE.plain_state(missing.clone()))
    assert not CE.equal_tree(CE.plain_state(missing), CE.plain_state(torch.tensor([float('nan'), 1.])))


@pytest.mark.slow
@NEEDS_H84_ARGS
def test_the_harness_snapshots_and_restores_the_deploys_full_flag_set(tmp_path):
    """Built on H84's own arguments with every deploy flag on (the review's blocker: the snapshot
    refused contact_tracker and wellness_channels). One step, snapshot, three steps, restore, the
    same three steps: every prediction, order and tracker is identical."""
    from mimo_bridge.tools import causal_test as CT
    from mimo_bridge import train_mimo as TM
    torch.set_num_threads(4)
    saved, _p, _f = CT._cfg_from_run(str(H84))
    ns = CT._probe_configuration(saved, "cpu", 0, True, dict(_deploy_flags(), caregiver_record_dir=""))
    model = TM.build_learner(ns, None)
    comps = H.components(model)
    assert comps["contact_head"]["present"] and comps["efference"]["present"]
    assert model.wellness_channels is not None and model.contact_tracker is not None
    frozen = H.Frozen(model=model, args=ns, info={}, limits=dict(phase_name="mature", strength_limit=1.0,
                      rom_limit=1.0, age_months=24.0), limits_source="test", load={"reinitialized": []},
                      build_s=0.0, load_s=0.0)
    s = H.Session(frozen, seed=3, gaze_reflex=False, saccades=False, run_dir=str(tmp_path / "scene"))
    try:
        for _ in range(3):
            s.observe()
            s.act(s.hold_order())
        s.observe()
        snap = s.snapshot()
        assert set(snap["model"]["special"]) >= {"contact_tracker", "wellness_channels"}
        if getattr(model, "nf_norm", None) is not None:          # --next-frame-norm persistence (b6)
            assert "nf_norm" in snap["model"]["special"]
        at_snap = CE.trackers(model)
        order = s.rotated(s.hold_order(), "lowerarm_R", [0, 0, 1], 20.0)
        recs = []
        for rep in range(2):
            if rep:
                s.restore(snap)
                assert CE.equal_tree(CE.trackers(model), at_snap), CE.first_difference(CE.trackers(model), at_snap)
            out = []
            for k in range(3):
                s.act(order, new_order=(k == 0))
                s.observe()
                out.append(CE.output_record(model, s.res))
            recs.append(out)
            if not rep:
                assert not CE.equal_tree(CE.trackers(model), at_snap), "the trackers never moved: nothing tested"
        assert CE.equal_tree(recs[0], recs[1]), CE.first_difference(recs[0], recs[1])
    finally:
        s.close()


@pytest.mark.slow
@NEEDS_H84_ARGS
def test_the_causal_test_builds_the_deploys_flags_with_the_planner_off():
    """causal_test's default is the planner off; --wellness-rank normalized refuses a build
    without it, so a planner-off probe builds with the raw ranking."""
    from mimo_bridge.tools import causal_test as CT
    from mimo_bridge import train_mimo as TM
    saved, _p, _f = CT._cfg_from_run(str(H84))
    ns = CT._probe_configuration(dict(saved, **_deploy_flags()), "cpu", 0, False, None)
    assert ns.intention == "off" and ns.wellness_rank == "raw" and ns.agency_drive == "off"
    model = TM.build_learner(ns, None)
    assert model.intention_gen is None and model.wellness_rank_mode == "raw"
    on = CT._probe_configuration(dict(saved, **_deploy_flags()), "cpu", 0, True, None)
    assert on.wellness_rank == "normalized"


def test_planner_off_settings_follow_the_planner():
    from mimo_bridge.tools import causal_test as CT
    saved = {"wellness_rank": "normalized", "intention": "on", "agency_drive": "on"}
    assert CT._probe_configuration(saved, "cpu", 0, False, None).wellness_rank == "raw"
    assert CT._probe_configuration(saved, "cpu", 0, True, None).wellness_rank == "normalized"


# ------------------------------------------------------------------ the eye row, on the real checkpoint
@pytest.mark.slow
@NEEDS_H84
def test_a_body_arms_prediction_is_the_models_own_next_moment_lesson_prediction():
    """hold_order()'s eye row is the command act() sends, so predict(moment, order) equals the
    model's own delayed next-frame prediction (_nf_pred) after act(order) and observe() - exactly
    (the review measured the old row 161 degrees from the executed one)."""
    H84_FOUND.require()
    from mimo_bridge.lem_model import LemLite
    frozen = H.load_frozen(H84)
    starts = H.start_states(frozen, 1, gaze_reflex=False, saccades=False)
    session, snap = next(starts)
    try:
        session.restore(snap)
        gi = frozen.model.GAZE_IDX
        base = session.hold_order()
        want = LemLite.gaze_to_handle(base, session.eyes_rad().reshape(1, 2))
        assert np.array_equal(base[0, gi], want[0, gi])
        for order in (base, session.rotated(base, "lowerarm_R", [0, 0, 1], 20.0)):
            session.restore(snap)
            p = H.predict(frozen.model, session.moment(), order, want_eff=False)["preds"]
            session.act(order)
            assert np.array_equal(session.executed, order)
            session.observe()
            for m, v in frozen.model._nf_pred.items():
                if m in p:
                    assert torch.equal(v.detach().float().reshape(1, -1), p[m].reshape(1, -1)), m
    finally:
        starts.close()


# ------------------------------------------------------------------ numbers a person reads
def test_pct_writes_three_significant_figures_without_exponents():
    assert H.pct(7.314362812039169e-08) == "0.00000731%"
    assert H.pct(0.0013529) == "0.135%"
    assert H.pct(-0.198) == "-19.8%"
    assert H.pct(0.5) == "50%" and H.pct(1.0) == "100%" and H.pct(0.0) == "0%"
    assert H.pct(None) == "unmeasured"
    assert H.parts_in(7.3e-8) == "7 in every hundred million"
    assert H.parts_in(0.0014) == "1 in every thousand"
    assert H.parts_in(0.05) is None
    assert "wrong way" in H.share_text(-0.2)
    for v in (7.3e-8, 3e-6, 0.0014, 0.25, -0.198, 1.2):
        assert not bare_decimals(H.share_text(v)), H.share_text(v)


def test_headlines_carry_no_goal_until_one_is_measured_and_round_the_distance():
    h = H.headline("m", 7.314362812039169e-08, "share", 0.0, None, "text")
    assert h["goal"] is None and h["distance_to_goal"] is None and "unmeasured" in h["goal_note"]
    assert h["display"] == "0.00000731%"
    g = H.headline("m", 7.314362812039169e-08, "share", 0.0, 1.0, "text")
    assert g["distance_to_goal"] == 1.0                  # 0.9999999268563718 at 3 significant figures
    assert H.sig(0.123456) == 0.123 and H.sig(-1234.5) == -1230.0 and H.sig(None) is None


def test_the_probes_sentences_carry_no_bare_decimals_and_no_code_names():
    from mimo_bridge.tools import gof_probe_gaze as G, gof_probe_binding as B, gof_probe_posture as P
    findings = dict(vision_captured_share_150ms=7.3e-8, vision_captured_share_50ms_trained=3.0e-8,
                    vision_size_against_hold_150ms=5.0e-6,
                    discrimination_vision_150ms=dict(pairs=24, decided_pairs=0, correct=0, undecided_pairs=4,
                                                     tied_pairs=20),
                    clipped_commands=dict(n=2, of=16))
    reaf = [dict(ratio_imposed_over_self=1.003), dict(ratio_imposed_over_self=0.97)]
    lifts = [dict(lifting=dict(neck_torque_mean_nm=1.82), holding=dict(neck_torque_mean_nm=0.1))]
    texts = [G.lived_sentence(findings),
             B.lived_sentence(2, 2, 0.0014, -0.198, 0.02, (1, 2)),
             P.reafference_sentence(reaf),
             P.lived_sentence([3.4], P._token_sense(["proprioception"])[1], 5.8e-8, reaf, 0, 2, lifts)]
    for t in texts:
        assert not bare_decimals(t), (t, bare_decimals(t))
        for banned in ("token", "handle", "honest", "barely", "slightly", "roughly", "e-0"):
            assert banned not in t.lower(), (banned, t)
    assert "undecided" in texts[0] and "clipped" in texts[0]
    assert "no consistent difference" in texts[2]


def test_units_name_plain_things():
    import inspect
    from mimo_bridge.tools import gof_probe_gaze as G, gof_probe_body as Bo, gof_probe_binding as Bi, \
        gof_probe_posture as P
    for mod in (G, Bo, Bi, P):
        src = inspect.getsource(mod.run)
        assert "(0-1)" not in src and "token (0-1)" not in src and "handle's place" not in src, mod.__name__


# ------------------------------------------------------------------ components
def test_components_count_a_part_only_when_the_attribute_is_the_module():
    flag_only = SimpleNamespace(active=["vision"], motor_synergy=False, synergy=None, contact_head=None,
                                tenure_heads=None, persistent_memory=None)
    assert not H.components(flag_only)["synergy"]["present"]
    built = SimpleNamespace(active=["vision"], motor_synergy=True, synergy=torch.nn.Linear(2, 2),
                            contact_head=torch.nn.Linear(2, 2), tenure_heads=None, persistent_memory=None)
    c = H.components(built)
    assert c["synergy"] == dict(present=True, attr="synergy")
    assert c["contact_head"] == dict(present=True, attr="contact_head")


# ------------------------------------------------------------------ the burst merge
def test_the_burst_merge_is_the_trainers():
    tasks = [dict(self_contact_onsets=[dict(pair=["a", "b"])], contact_onsets_world=1, contact_patch_state=[0, 2, 0],
                  mobile_touched=True, roll_deg=10.0),
             dict(self_contact_onsets=[], contact_onsets_world=2, contact_patch_state=[1, 0, 0], roll_deg=20.0,
                  hand_view_deg=40.0)]
    m = H.merge_burst_tasks(tasks)
    assert m["self_contact_onsets"] == [dict(pair=["a", "b"])] and m["contact_onsets_world"] == 3
    assert m["contact_patch_state_burst"] == [1, 2, 0] and m["contact_patch_state"] == [1, 0, 0]
    assert m["mobile_touched"] is True and m["roll_deg"] == 15.0 and m["hand_view_deg"] == 40.0


# ------------------------------------------------------------------ the load search
def test_the_load_search_finds_the_smallest_torque_that_halves_the_turn():
    from mimo_bridge.tools import gof_probe_body as B
    turn = lambda t: -10.0 / (1.0 + t)                   # a turn the load slows: halved at 1 N m
    out = B.size_load(turn, -10.0)
    assert out["torque_nm"] is not None and 1.0 <= out["torque_nm"] <= 1.0 + 0.8 / 64 + 1e-9
    assert abs(out["loaded_deg"]) <= 5.0 + 1e-9
    stuck = B.size_load(lambda t: 10.0 - 1e-4 * t, 10.0)  # the load hardly changes the motion
    assert stuck["torque_nm"] is None and stuck["search"][-1][0] == pytest.approx(B.LOAD_TORQUE_MAX_NM)


# ------------------------------------------------------------------ the self-touch holding arm
class _FakeTouchSession:
    """Enough of a Session for binding.self_touch: the body is at `pose` until an arm moves it."""

    def __init__(self):
        self.start = np.zeros((26, 3))
        self.pose_now = self.start.copy()
        self.handle_names = ["h%d" % i for i in range(26)]
        self.dev = {"strength_limit": 1.0}
        self.clock = SimpleNamespace(model_step_seconds=0.05)
        self.env = SimpleNamespace(model=SimpleNamespace(body=lambda n: SimpleNamespace(id=0), nbody=0))

    def restore(self, snap):
        self.pose_now = self.start.copy()

    def hold_order(self):
        o = np.zeros((1, 26, 9), np.float32)
        o[0, :, 0:3] = self.pose_now
        return o

    def contact_between(self, a, b):
        return 0.0 if np.allclose(self.pose_now, self.start) else 1.0

    def physics_only(self, snap, order, steps, watch=None):
        out = []
        for _ in range(steps):
            self.pose_now = self.start + 5.0                 # the scripted order moves the body
            out.append(watch(self) if watch else None)
        return out


def test_the_self_touch_holding_arm_is_ordered_to_the_start_pose(monkeypatch):
    from mimo_bridge.tools import gof_probe_binding as B
    s = _FakeTouchSession()
    sent = []

    def fake_stepwise(session, snap, order, steps, senses, watch=None, **kw):
        sent.append(np.array(order, copy=True))
        session.pose_now = session.start + 5.0              # an arm leaves the body at the touching pose
        return [dict(step=k + 1, scores={}, share={}, cf={}, watch=(1.0 if watch else None)) for k in range(steps)]

    monkeypatch.setattr(B, "_stepwise", fake_stepwise)
    monkeypatch.setattr(B, "_groups", lambda session, kind: ({1}, {2}))
    monkeypatch.setattr(B, "_touch_order", lambda session, kind: np.full((1, 26, 9), 9.0, np.float32))
    comps = {"contact_head": {"present": False}}
    out = B.self_touch(s, {}, ["tactile"], comps)
    assert all(r["status"] == "measured" for r in out.values())
    holds = sent[1::2]                                      # moving, holding, moving, holding, ...
    assert len(holds) == 3
    for h in holds:
        assert np.array_equal(h[0, :, 0:3], s.start)         # the start pose, not the touching pose


# ------------------------------------------------------------------ the binding counterfactual
class _FakeBindingSession:
    """A one-number world: an order moves the state by `order[0,0,0]` per step; the fake model
    predicts half of that move plus a continuation of the last move (a model that extends motion)."""

    def __init__(self):
        self.x, self.last = 0.0, 0.0
        self.handle_names = ["h"]
        self.frames = [SimpleNamespace(meta={"task": {}})]
        self.model = None

    def snapshot(self):
        return dict(x=self.x, last=self.last)

    def restore(self, snap):
        self.x, self.last = snap["x"], snap["last"]

    def moment(self):
        return SimpleNamespace(tokens={"vision": torch.tensor([[self.x]])}, x=self.x, last=self.last)

    def exec_order(self, order, gaze_rad=None):
        return np.asarray(order, np.float32)

    def hold_order(self):
        return np.zeros((1, 1, 9), np.float32)

    def act(self, order, new_order=True, **kw):
        move = float(np.asarray(order)[0, 0, 0])
        self.last = move
        self.x += move

    def observe(self):
        pass

    def readout(self):
        return dict(tokens={"vision": torch.tensor([[self.x]])})


def test_the_binding_score_is_the_counterfactual_against_holding(monkeypatch):
    from mimo_bridge.tools import gof_probe_binding as B

    def fake_predict(model, mom, order, want_eff=True):
        move = float(np.asarray(order)[0, 0, 0])
        return {"preds": {"vision": torch.tensor([[mom.x + 0.5 * move + mom.last]])}}

    monkeypatch.setattr(H, "predict", fake_predict)
    s = _FakeBindingSession()
    s.last = 3.0                                             # the hand was already moving
    order = np.zeros((1, 1, 9), np.float32)
    order[0, 0, 0] = 2.0
    rows = B._stepwise(s, s.snapshot(), order, 3, ["vision"], counterfactual_until=2)
    assert rows[0]["cf"]["vision"]["captured"] == pytest.approx(0.5)    # half the order's effect
    assert rows[1]["cf"]["vision"]["captured"] == pytest.approx(0.5)
    assert rows[2]["cf"] == {}                                          # past counterfactual_until
    # the continuation share rewards extending the motion: 2.5 of the real 2 at the first step
    assert rows[0]["share"]["vision"]["captured"] == pytest.approx((0.5 * 2 + 3.0) / 2.0)


# ------------------------------------------------------------------ the gaze pairs and routes
def _gaze_res(noise_decided: bool):
    t = lambda *v: torch.tensor([list(v)], dtype=torch.float32)
    def arm(pred, real):
        return dict(target_deg=[0.0, 0.0], pred={"preds": {"vision": pred}}, pred_ablated={"preds": {"vision": pred}},
                    roll=[{"vision": pred}], roll_ablated=[{"vision": pred}],
                    actual=[dict(tokens={"vision": real}, eyes_deg=np.zeros(2))])
    arms = {"hold": arm(t(0.0, 0.0), t(0.0, 0.0)),
            "left": arm(t(1e-6, 0.0), t(1.0, 0.0)),        # predictions a millionth apart,
            "right": arm(t(0.0, 1e-6), t(0.0, 1.0))}       # outcomes a whole unit apart
    line = {"left|right": dict(main_norm=1.4e-6, spread=1e-6, line=2e-6, decided=noise_decided, readings=4)}
    return dict(mom=SimpleNamespace(tokens={"vision": t(0.0, 0.0)}), eyes0=np.zeros(2), arms=arms, pathway=None,
                noise={"50ms_trained": {"vision": line}, "50ms": {"vision": line}})


def test_a_gaze_pair_under_its_noise_line_is_undecided_not_correct():
    from mimo_bridge.tools import gof_probe_gaze as G
    under = G._score_state(_gaze_res(False), ["vision"], 1)
    assert under["discrimination"]["50ms"]["vision"] == [] and under["discrimination"]["50ms_undecided"]["vision"] == 1
    over = G._score_state(_gaze_res(True), ["vision"], 1)
    assert over["discrimination"]["50ms"]["vision"] == [True]
    # both routes are scored against holding the eyes still, by name
    assert set(over["sensitivity_vs_hold"]) == {"50ms_trained", "50ms"}
    row = over["sensitivity_vs_hold"]["50ms_trained"]["vision"][0]
    assert row["captured_share"] == pytest.approx(1e-6) and row["dot_pp"] == pytest.approx(1e-12)


def test_a_gaze_pair_counts_as_identical_only_when_its_predictions_are_equal():
    """verify-fix-probes 2026-09-23: the float tie test ran before the noise line and its count was
    written "gave identical predictions"; on ckpt_step_344400 21 of those 24 pairs' predictions
    differed and only 3 were exactly equal. Pairs whose predictions differ but sit under the noise
    line are undecided; only exactly equal predictions are identical."""
    from mimo_bridge.tools import gof_probe_gaze as G
    t = lambda *v: torch.tensor([list(v)], dtype=torch.float32)

    def arm(pred, real):
        return dict(target_deg=[0.0, 0.0], pred={"preds": {"vision": pred}}, pred_ablated={"preds": {"vision": pred}},
                    roll=[{"vision": pred}], roll_ablated=[{"vision": pred}],
                    actual=[dict(tokens={"vision": real}, eyes_deg=np.zeros(2))])
    arms = {"hold": arm(t(0.0, 0.0), t(0.0, 0.0)),
            "left": arm(t(1e-9, 0.0), t(1.0, 0.0)),        # differs from right by a billionth
            "right": arm(t(0.0, 1e-9), t(0.0, 1.0)),
            "up": arm(t(1e-9, 0.0), t(0.0, 0.5))}          # exactly equal to left's prediction
    under = {f"{x}|{y}": dict(main_norm=0.0, spread=1e-6, line=2e-6, decided=False, readings=4)
             for x, y in (("left", "right"), ("left", "up"), ("right", "up"))}
    res = dict(mom=SimpleNamespace(tokens={"vision": t(0.0, 0.0)}), eyes0=np.zeros(2), arms=arms, pathway=None,
               noise={"50ms_trained": {"vision": under}, "50ms": {"vision": under}})
    out = G._score_state(res, ["vision"], 1)["discrimination"]
    assert out["50ms_ties"]["vision"] == 1 and out["50ms_undecided"]["vision"] == 2 and out["50ms"]["vision"] == []
    s = G.lived_sentence(dict(discrimination_vision_150ms=dict(pairs=24, decided_pairs=0, correct=0,
                                                               undecided_pairs=21, tied_pairs=3)))
    assert "21 were undecided" in s and "3 gave exactly the same prediction" in s and not bare_decimals(s)


# ------------------------------------------------------------------ retries and the watcher's memory
def _info(tmp_path, step=7, run="R1"):
    return dict(run=run, name=f"ckpt_step_{step}", checkpoint=str(tmp_path / "ck" / f"ckpt_step_{step}"),
                lineage_step=step)


def test_a_failure_from_older_code_is_run_again_and_this_codes_failure_once_more(tmp_path, monkeypatch):
    """Rewritten 2026-09-23 (verify-fix-probes) to the new truth: a failure of the SAME code is run
    once more (the review named memory shortages and scene-file races, which need not repeat), and
    after SAME_CODE_ATTEMPTS failures in a row it waits for other code; before, one same-code
    failure ended that checkpoint's measurement."""
    root = tmp_path / "gof"
    info = _info(tmp_path)
    assert H.needs_run(info["checkpoint"], "gaze", root)
    monkeypatch.setitem(H._CODE, "commit", "old")
    fail = dict(headline=None, status="failed: MemoryError: out of memory")
    H.write_result(info, "gaze", 7, fail, root=root)
    line = H.read_index(root)[-1]
    assert line["harness_version"] == H.HARNESS_VERSION and line["code_commit"] == "old"
    assert H.SAME_CODE_ATTEMPTS == 2
    assert H.needs_run(info["checkpoint"], "gaze", root)             # a first same-code failure: once more
    H.write_result(info, "gaze", 7, fail, root=root)
    assert not H.needs_run(info["checkpoint"], "gaze", root)         # twice on this code: waits
    monkeypatch.setitem(H._CODE, "commit", "new")
    assert H.needs_run(info["checkpoint"], "gaze", root)             # fixed code: retried
    H.write_result(info, "gaze", 7, dict(headline=None, status="measured"), root=root)
    assert not H.needs_run(info["checkpoint"], "gaze", root)
    # failures of older code do not count against this code: one failure of it is run once more
    monkeypatch.setitem(H._CODE, "commit", "old")
    H.write_result(info, "binding", 9, fail, root=root)
    H.write_result(info, "binding", 9, fail, root=root)
    monkeypatch.setitem(H._CODE, "commit", "new")
    H.write_result(info, "binding", 9, fail, root=root)
    assert H.needs_run(info["checkpoint"], "binding", root)


def test_a_success_from_an_older_harness_is_measured_again(tmp_path, monkeypatch):
    root, info = tmp_path / "gof", _info(tmp_path)
    H.write_result(info, "gaze", 7, dict(headline=None, status="measured"), root=root)
    assert not H.needs_run(info["checkpoint"], "gaze", root)
    monkeypatch.setattr(H, "HARNESS_VERSION", H.HARNESS_VERSION + "_changed")
    assert H.needs_run(info["checkpoint"], "gaze", root)


def test_the_watcher_runs_a_same_code_failure_once_more_then_leaves_it(tmp_path, monkeypatch):
    from mimo_bridge.tools import gof_probe_watch as W
    root = tmp_path / "gof"
    ck = tmp_path / "exp" / "L_seg1" / "checkpoints" / "ckpt_step_20"
    monkeypatch.setattr(W, "complete_checkpoints", lambda e, p: [(20, ck)])
    monkeypatch.setattr(W, "newest_mtime", lambda p: 0.0)
    info = dict(run="L_seg1", name=ck.name, checkpoint=str(ck), lineage_step=20)
    monkeypatch.setitem(H._CODE, "commit", "c1")
    for probe in W.PROBES:
        H.write_result(info, probe, 7, dict(headline=None, status="measured"), root=root)
    assert W.pending(tmp_path / "exp", "L_", root, 10, time.time()) == []
    H.write_result(info, "gaze", 7, dict(headline=None, status="failed: MemoryError"), root=root)
    assert [s for s, _ in W.pending(tmp_path / "exp", "L_", root, 10, time.time())] == [20]
    H.write_result(info, "gaze", 7, dict(headline=None, status="failed: MemoryError"), root=root)
    assert W.pending(tmp_path / "exp", "L_", root, 10, time.time()) == []


def test_the_watcher_reads_the_commit_again_before_each_pass():
    """verify-fix-probes 2026-09-23: code_commit() is cached at a process's first call; a watcher
    judging with the commit it started on counts a newer suite child's failure as other code for
    ever. refresh_code_commit() reads git again, and the watcher's loop calls it every pass."""
    import inspect
    import subprocess
    from mimo_bridge.tools import gof_probe_watch as W
    saved = dict(H._CODE)
    try:
        H._CODE.clear()
        H._CODE["commit"] = "stale"
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
        # a pinned job's export holds no .git: its commit is the one pinned_job checked the export against
        assert H.refresh_code_commit()["commit"] == (head or H.pinned_commit())
    finally:
        H._CODE.clear()
        H._CODE.update(saved)
    loop = inspect.getsource(W.main).split("while True:", 1)[1]
    assert loop.index("H.refresh_code_commit()") < loop.index("pending(")


def test_the_watcher_remembers_where_it_began(tmp_path, monkeypatch):
    from mimo_bridge.tools import gof_probe_watch as W
    exp, root = tmp_path / "experiments", tmp_path / "gof"
    found = [(10, Path("a")), (20, Path("b"))]
    monkeypatch.setattr(W, "complete_checkpoints", lambda e, p: list(found))
    step, how = W.start_step_for(root, "L_", exp)
    assert step == 20 and "first start" in how
    W._status(root, state="idle", start_steps={"L_": dict(step=step, since="then", how=how)})
    found.append((30, Path("c")))                                      # written while the watcher was down
    step2, how2 = W.start_step_for(root, "L_", exp)
    assert step2 == 20 and "recorded" in how2                          # not 30: step 30 is still probed
    W._status(root, state="idle", start_steps={"M_": dict(step=5, since="now", how="x")})
    assert set(W.read_status(root)["start_steps"]) == {"L_", "M_"}     # other lineages kept
    assert W.start_step_for(root, "L_", exp, explicit=0)[0] == 0
    assert W.start_step_for(root, "N_", exp, include_existing=True)[0] == 0


# ------------------------------------------------------------------ both sides of a self-touch
def test_both_sides_are_scored_on_each_sides_newly_own_touched_patches():
    """_both_sides reads, on each side's patches that the world reported NEWLY touched by an own
    part at the onset (bit 2 of the burst's patch state, off the moment before), the contact head's
    own-part probability for the touching order against the hold order."""
    from mimo_bridge.tools import gof_probe_binding as B
    names = ["a_patch", "b_patch", "b_other", "elsewhere"]
    bodies = {1: "a_patch", 2: "b_patch", 3: "b_other"}
    s = SimpleNamespace(env=SimpleNamespace(model=SimpleNamespace(body=lambda b: SimpleNamespace(name=bodies[b]))))
    p = lambda *v: np.asarray(v, dtype=np.float32)
    row = dict(contacts=dict(order=dict(p_self=p(0.9, 0.2, 0.8, 0.0), patch_names=names),
                             hold=dict(p_self=p(0.1, 0.4, 0.1, 0.0), patch_names=names),
                             # b_other was already touched by an own part the moment before: not an onset
                             patch_state_before=[0, 0, 2, 0]),
               world_task=dict(contact_patch_state_burst=[2, 3, 2, 1]))
    out = B._both_sides(s, row, {1}, {2, 3})
    assert out["sides"]["a"]["patches_touched"] == ["a_patch"] and out["sides"]["a"]["order_above_hold"]
    assert out["sides"]["b"]["patches_touched"] == ["b_patch"]              # b_other excluded
    assert out["sides"]["b"]["order_above_hold"] is False                   # 0.2 against 0.4
    assert out["sides_scored"] == 2 and out["both_sides_foreseen"] is False
    # a side the world did not report newly own-touched is not scored, and nothing is decided
    row["world_task"] = dict(contact_patch_state_burst=[2, 1, 0, 0])
    out = B._both_sides(s, row, {1}, {2, 3})
    assert "no new own-part touch" in out["sides"]["b"]["status"] and out["both_sides_foreseen"] is None
    assert "unmeasured" in B._both_sides(s, dict(contacts=None), {1}, {2})["status"]


def test_a_side_whose_probabilities_differ_by_rounding_is_a_tie_not_foreseen():
    """verify-fix-probes 2026-09-23: a bare po > ph let a head whose answer does not depend on the
    order score a side on float rounding (the class of the review's gaze-pair finding)."""
    from mimo_bridge.tools import gof_probe_binding as B
    names = ["a_patch", "b_patch"]
    bodies = {1: "a_patch", 2: "b_patch"}
    s = SimpleNamespace(env=SimpleNamespace(model=SimpleNamespace(body=lambda b: SimpleNamespace(name=bodies[b]))))
    p = lambda *v: np.asarray(v, dtype=np.float64)
    row = dict(contacts=dict(order=dict(p_self=p(0.5 + 1e-9, 0.5 + 1e-9), patch_names=names),
                             hold=dict(p_self=p(0.5, 0.5), patch_names=names), patch_state_before=[0, 0]),
               world_task=dict(contact_patch_state_burst=[2, 2]))
    out = B._both_sides(s, row, {1}, {2})
    assert out["sides"]["a"]["order_above_hold"] is None and out["sides_tied"] == 2
    assert out["both_sides_foreseen"] is False and out["sides_scored"] == 2
    t = B.lived_sentence(2, 2, 0.001, 0.002, None, (0, 1))
    assert "1 of every 4" in t and not bare_decimals(t)


def test_the_body_sentence_anchors_the_tenure_heads_inputs_to_the_response_line():
    """verify-fix-probes 2026-09-23: the sentence printed the raw input range ("inputs from -32.9 to
    -7.1"), a unit a person cannot picture. fix-readout-explainer 2026-09-23: "rises above -3; the
    highest input any unit received was -7" is the same kind of number (the extended scanner's raw
    hidden-unit input rule). The sentence now says what the line means - below it GELU moves its
    output by less than 1.2% of a change in its input - and how near the nearest unit came, as a count."""
    from mimo_bridge.tools import gof_probe_body as Bo
    from mimo_bridge.tools.check_no_bare_decimals import readout_problems
    rows = [dict(units=256, live_share=0.0, input_min=-32.92, input_max=-7.13)]
    live = Bo._liveness_summary([dict(arms={"hold": dict(tenure_hidden=rows[0])})])
    text = live["lived"]
    assert live["units"] == 256 and "-32.9" not in text and "-7" not in text and "-3" not in text
    assert "less than 1.2% of the change" in text
    # at -7.13 GELU's output moves by 2.5e-11 of a change in its input: 3 in every hundred billion
    assert "3 in every hundred billion parts" in text, text
    assert not readout_problems(text), readout_problems(text)
    # the raw values stay in the file for a reader of the numbers
    assert live["input_max"] == -7.13 and live["live_input_line"] == H.GELU_LIVE_INPUT
    # a head with a unit above the line says so, in the same words
    rows = [dict(units=4, live_share=0.25, input_min=-9.0, input_max=-1.0)]
    text = Bo._liveness_summary([dict(arms={"hold": dict(tenure_hidden=rows[0])})])["lived"]
    assert "At least one unit left the flat end" in text and not readout_problems(text)
    # the efference head's sentence, the other liveness sentence of the body probe
    eff = Bo._efference_liveness({"net.0": 0.6, "net.2": 0.7})["lived"]
    assert "60% and 70%" in eff and "1.2%" in eff and not readout_problems(eff), eff


def test_the_response_line_is_where_gelu_moves_its_output_by_1_2_percent_of_a_change():
    """The share the liveness sentences print is GELU's slope at the harness's line, computed, not
    typed: |Phi(-3) - 3 phi(3)| = 0.0119. Below the line the slope only shrinks toward 0 (so 'input
    above the line' and 'output follows at least 1.2%' are one test there), and far above it the
    output follows the input fully (100%)."""
    assert abs(H.LIVE_FOLLOW - 0.011946) < 1e-6 and H.pct(H.LIVE_FOLLOW, 2) == "1.2%"
    below = [abs(H.gelu_slope(H.GELU_LIVE_INPUT - 0.25 * k)) for k in range(1, 60)]
    assert all(a > b for a, b in zip(below, below[1:])) and below[0] < H.LIVE_FOLLOW
    assert abs(H.gelu_slope(8.0) - 1.0) < 1e-9 and H.gelu_slope(0.0) == 0.5
    assert H.follow_text(-29.1) == "less than 1 in every trillion parts"
    assert H.follow_text(-5.5) == "6 in every ten million parts"


def test_the_liveness_words_never_say_every_responding_unit_moves_by_1_2_percent():
    """verify-fix-readout-explainer 2026-09-23. The 1.2% belongs to the LINE, not to every unit above
    it: GELU's slope passes through 0 at an input of -0.75, above the line, so a unit that counts as
    responding can move its output by less than 1.2% (measured: from -0.78 to -0.725 on a 0.001
    grid). The addendum said "a responding unit moves it by 1.2% or more" and "the 1.2% a responding
    unit reaches"; it now ties 1.2% to a unit at the line. And the nearest-unit sentence wrote "by
    only less than 1 in every trillion parts" for every nearest input below about -7.6."""
    import re
    from mimo_bridge.tools.check_no_bare_decimals import page_text, readout_problems
    assert abs(H.gelu_slope(-0.75)) < H.LIVE_FOLLOW and -0.75 > H.GELU_LIVE_INPUT
    add = (Path(__file__).resolve().parents[4] / "docs" / "gain_of_function_v3"
           / "UEMA_Embodied_Gain_of_Function_Addendum_v3.html")
    text = " ".join(page_text(add.read_text(encoding="utf-8")).split())
    claims = re.findall(r"[^.;]*responding unit[^.;]*1\.2%[^.;]*|[^.;]*1\.2%[^.;]*responding unit[^.;]*", text)
    assert not claims, claims
    assert text.count("1.2%") >= 3            # the addendum still gives the line's share
    for x in (-3.0, -5.5, -7.13, -7.7, -8.0, -29.0):
        s = H.nearest_unit_text(x)
        assert "only less than" not in s and not readout_problems(s), (x, s)
    assert H.nearest_unit_text(-8.0) == ("Even the unit nearest to responding moved its output by less "
                                         "than 1 in every trillion parts of a change in its input.")
    assert H.nearest_unit_text(-7.13) == ("Even the unit nearest to responding moved its output by only 3 "
                                          "in every hundred billion parts of a change in its input.")


# ------------------------------------------------------------------ suite.json follows the newest result
def test_suite_json_follows_a_probe_run_alone_and_the_suite_keeps_that_entry(tmp_path, monkeypatch):
    """A probe run on its own (not through the suite) puts its newest headline into suite.json, and
    a later suite run of another probe keeps it (2026-09-23: the v2 gaze and body results were run
    one at a time and suite.json went on listing their v1 headlines)."""
    from mimo_bridge.tools import gof_probe_suite as S
    root = tmp_path / "gof"
    info = _info(tmp_path)
    H.write_result(info, "gaze", 7, dict(headline=H.headline("g", 0.1, "share", 0.0, None, "x"),
                                         status="measured", elapsed_s=5.0), root=root)
    path = H.write_result(info, "gaze", 7, dict(headline=H.headline("g", 0.2, "share", 0.0, None, "y"),
                                                status="measured", elapsed_s=6.0), root=root)
    suite = root / info["run"] / info["name"] / "suite.json"
    entry = json.loads(suite.read_text(encoding="utf-8"))["probes"]["gaze"]
    assert entry["headline"]["value"] == 0.2 and entry["path"] == str(path)
    assert entry["harness_version"] == H.HARNESS_VERSION and entry["elapsed_s"] == 6.0
    monkeypatch.setattr(H, "resolve_checkpoint", lambda ck: info)

    def fake_module(name):
        probe = name.rsplit("_", 1)[1]

        def run(ckpt, root=None, **kw):
            body = dict(headline=H.headline(probe, 0.3, "share", 0.0, None, "z"), status="measured", elapsed_s=1.0)
            body["path"] = str(H.write_result(info, probe, S.ITEMS[probe], body, root=root))
            return body
        return SimpleNamespace(run=run)

    monkeypatch.setattr(S.importlib, "import_module", fake_module)
    S.run_suite("ck", root=root, probes=("binding",))
    s = json.loads(suite.read_text(encoding="utf-8"))
    assert s["probes"]["gaze"]["headline"]["value"] == 0.2                 # the lone run's entry kept
    b = s["probes"]["binding"]
    assert b["harness_version"] == H.HARNESS_VERSION and "peak_memory_gb" in b and b["time"]
