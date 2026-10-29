# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The gain-of-function probe harness, suite and watcher (DESIGN.md D15, section 9).

Fast tests use small fakes; the two marked slow ones load the real H84 checkpoint (ckpt_step_344400)
and its episode companion, found wherever the archiver moved them, and are skipped only when that
checkpoint is nowhere.
"""
import json
import math
import os
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimo_bridge.tools import gof_probe_harness as H


# ------------------------------------------------------------------ scoring
def test_captured_share_reads_the_share_of_the_real_change():
    a = torch.tensor([2.0, 0.0])
    assert H.captured_share(a, a) == pytest.approx(1.0)
    assert H.captured_share(0.5 * a, a) == pytest.approx(0.5)
    assert H.captured_share(torch.tensor([0.0, 3.0]), a) == pytest.approx(0.0)    # right angles
    assert H.captured_share(-a, a) == pytest.approx(-1.0)
    assert H.captured_share(a, torch.zeros(2)) is None                          # nothing happened


def test_pooled_share_is_not_ruled_by_a_tiny_arm():
    rows = [dict(d=1.0, n=1.0),                    # a real change, fully foreseen
            dict(d=-0.003, n=0.00001)]             # a hundredth of a cm, per-arm ratio -300
    assert H.pooled_share(rows, "d", "n") == pytest.approx(0.997 / 1.00001, rel=1e-6)
    assert H.pooled_share([], "d", "n") is None
    assert H.pooled_share([None, dict(d=None, n=1.0)], "d", "n") is None


def test_pairing_ties_are_not_wrong_answers():
    act_a, act_b = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    assert H.pairing_correct(act_a, act_b, act_a, act_b) is True
    assert H.pairing_correct(act_b, act_a, act_a, act_b) is False
    same = torch.tensor([0.5, 0.5])
    assert H.pairing_correct(same, same.clone(), act_a, act_b) is None       # identical predictions
    assert H.pairing_correct(act_a, act_b, act_a, act_a.clone()) is None     # identical outcomes


def test_better_of_calls_float_level_differences_a_tie():
    assert H.better_of("l", 2.9831981, "r", 2.9831981) == "tie"
    assert H.better_of("l", 3.0, "r", 3.0 + 1e-9) == "tie"
    assert H.better_of("l", 3.1, "r", 3.0) == "l"
    assert H.better_of("l", 3.0, "r", 3.1) == "r"


def test_headline_distance_and_pct_words():
    h = H.headline("m", 0.25, "share", 0.0, 1.0, "text")
    assert h["distance_to_goal"] == pytest.approx(0.75) and h["floor"] == 0.0
    assert H.headline("m", None, "share", 0.0, 1.0, "t")["distance_to_goal"] is None
    assert H.headline("m", 0.3, "share", 0.0, None, "t")["distance_to_goal"] is None
    # three significant figures, trailing zeros dropped, never an exponent (review 2026-09-23)
    assert H.pct(None) == "unmeasured" and H.pct(0.5) == "50%"
    assert H.pct(4e-8) == "0.000004%"                              # tiny shares stay readable


def test_token_scores_improvement_over_persistence():
    cur = {"v": torch.zeros(1, 2)}
    act = {"v": torch.tensor([[1.0, 0.0]])}
    exact = H.token_scores({"v": torch.tensor([[1.0, 0.0]])}, act, cur, ["v"])["v"]
    assert exact["improvement"] == pytest.approx(1.0)
    lazy = H.token_scores({"v": torch.zeros(1, 2)}, act, cur, ["v"])["v"]
    assert lazy["improvement"] == pytest.approx(0.0)


def test_rotation_helpers_turn_by_the_asked_angle():
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = H.quat_mul_xyzw(H.axis_angle_xyzw([0, 0, 1], 20.0), q0)
    assert H.rotation_deg_xyzw(q0, q1) == pytest.approx(20.0, abs=1e-6)
    assert H.rotation_deg_xyzw(q1, -q1) == pytest.approx(0.0, abs=1e-6)    # q and -q are one pose
    r0 = dict(pos_cm=np.zeros((2, 3)), quat=np.stack([q0, q0]))
    r1 = dict(pos_cm=np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]), quat=np.stack([q1, q0]))
    ch = H.body_change(r0, r1)
    assert ch["cm_norm"][0] == pytest.approx(5.0) and ch["deg"][0] == pytest.approx(20.0, abs=1e-6)
    assert ch["deg"][1] == pytest.approx(0.0, abs=1e-6)


def test_tenure_liveness_sees_a_dead_hidden_layer():
    head = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 3))
    with torch.no_grad():
        head[0].weight.zero_()
        head[0].bias.fill_(-10.0)
    live = H.tenure_liveness(head, torch.randn(1, 4))
    assert live["live_share"] == 0.0 and live["input_max"] == pytest.approx(-10.0)
    with torch.no_grad():
        head[0].bias.fill_(0.5)
    assert H.tenure_liveness(head, torch.randn(1, 4))["live_share"] == 1.0


# ------------------------------------------------------------------ the episode slice
def test_slice_batch_takes_one_body_everywhere():
    n = 3
    state = {
        "context_buffer": torch.arange(3 * 4).reshape(3, 4).float(),
        "memory": {"carry": torch.arange(3 * 2).reshape(1, 3, 2).float(),
                   "events": [torch.full((3, 5), float(i)) for i in range(4)]},
        "site_ids": (("a", "b"), ("c", "d"), ("e", "f")),
        "goal_ring": deque([({"v": torch.arange(3).float()}, torch.tensor([1.0, 2.0, 3.0]))], maxlen=5),
        "replay": {"n_envs": 3, "tokens": torch.zeros(3, 7)},
        "token_ring": deque([torch.zeros(192)], maxlen=4),        # not batched: kept whole
        "plan_win_ema": [0.1, 0.2, 0.3],                          # a list of 3 floats is not bodies
        "step": 12,
    }
    b = H.slice_batch(state, n, 1)
    assert torch.equal(b["context_buffer"], torch.tensor([[4.0, 5.0, 6.0, 7.0]]))
    assert torch.equal(b["memory"]["carry"], torch.tensor([[[2.0, 3.0]]]))
    assert b["memory"]["events"][2].shape == (1, 5)
    assert b["site_ids"] == (("c", "d"),)
    assert float(b["goal_ring"][0][0]["v"]) == 1.0 and b["goal_ring"].maxlen == 5
    assert b["replay"]["n_envs"] == 1 and b["replay"]["tokens"].shape == (1, 7)
    assert b["token_ring"][0].shape == (192,) and b["plan_win_ema"] == [0.1, 0.2, 0.3] and b["step"] == 12
    # the source is untouched
    assert state["context_buffer"].shape == (3, 4)


def test_sanitize_replaces_every_recording_tape(tmp_path):
    from mimo_bridge.event_tape import EventTape
    live = tmp_path / "live_recordings"
    tape = EventTape(live, tmp_path / "status.json")
    ep = {"worlds": [{"world": {"caregiver": {"events": tape, "time": 1.0}}},
                     {"world": {"caregiver": {"events": [1, 2]}}}],
          "other": [tape, (tape,)]}
    assert H.sanitize_episode(ep) == 2          # the tuple is read-only: counted where replaceable
    assert ep["worlds"][0]["world"]["caregiver"]["events"] == []
    assert ep["worlds"][1]["world"]["caregiver"]["events"] == [1, 2]
    assert ep["other"][0] == []
    assert not live.exists()                     # nothing was written into the recording folder


# ------------------------------------------------------------------ resolution and output
def _fake_checkpoint(tmp_path, run="R1", step=7, lineage_out=None, args=None, beside=True):
    run_dir = tmp_path / "experiments" / run
    ck = run_dir / "checkpoints" / f"ckpt_step_{step}"
    ck.mkdir(parents=True)
    (ck / "model.safetensors").write_bytes(b"x")
    if beside:
        (run_dir / "args.json").write_text(json.dumps(args or {"seed": 1}), encoding="utf-8")
    lineage = {"run": run, "step": step, "out": str(lineage_out or run_dir)}
    (ck / "lineage.json").write_text(json.dumps(lineage), encoding="utf-8")
    return ck


def test_resolve_checkpoint_reads_lineage_and_args(tmp_path):
    ck = _fake_checkpoint(tmp_path)
    info = H.resolve_checkpoint(ck)
    assert info["run"] == "R1" and info["lineage_step"] == 7 and info["args_found"]
    assert info["name"] == "ckpt_step_7"
    with pytest.raises(FileNotFoundError):
        H.resolve_checkpoint(tmp_path / "nowhere")


def test_write_result_and_index(tmp_path):
    root = tmp_path / "gof"
    info = dict(run="R1", name="ckpt_step_7", checkpoint=str(tmp_path / "ck"), lineage_step=7)
    head = H.headline("m", 0.1, "share", 0.0, 1.0, "lived")
    p = H.write_result(info, "gaze", 7, dict(headline=head, status="measured", arms={}, controls={},
                                            config={}, extra=np.float32(1.5)), root=root)
    rec = json.loads(Path(p).read_text(encoding="utf-8"))
    for key in ("probe", "item", "run", "checkpoint", "lineage_step", "time", "arms", "controls", "headline", "status"):
        assert key in rec
    assert rec["extra"] == 1.5
    rows = H.read_index(root)
    assert len(rows) == 1 and rows[0]["probe"] == "gaze" and rows[0]["path"] == str(p)
    assert H.indexed(info["checkpoint"], "gaze", root) and not H.indexed(info["checkpoint"], "body", root)
    # a re-run appends; latest_index keeps the newest line per checkpoint and probe
    H.write_result(info, "gaze", 7, dict(headline=H.headline("m", 0.2, "share", 0.0, 1.0, "again"),
                                         status="measured"), root=root)
    latest = H.latest_index(root)
    assert len(H.read_index(root)) == 2 and len(latest) == 1 and latest[0]["headline"]["value"] == 0.2


def test_components_detects_by_attribute():
    # LemLite.motor_synergy is the FLAG (a bool) and LemLite.synergy the network: a part counts only
    # when its attribute holds a module (review of the probes, 2026-09-23: this test used to pin the
    # flag's name with a stand-in object, so a flags-off build was reported as carrying a synergy)
    bare = SimpleNamespace(active=["vision", "motor"], tenure_heads=None, persistent_memory=None,
                           motor_synergy=False, synergy=None, contact_head=None)
    c = H.components(bare)
    assert not any(v["present"] for v in c.values())
    net = torch.nn.Linear(2, 2)
    full = SimpleNamespace(active=["vision", "vestibular"], efference=net, contact_head=net, motor_synergy=True,
                           synergy=net, tenure_heads={"pose": object()}, persistent_memory=object())
    c = H.components(full)
    assert all(v["present"] for v in c.values())
    assert c["synergy"]["attr"] == "synergy"
    assert H.absent("efference")["status"] == "unmeasured: component absent"


def test_ancestors_follow_resumed_from(tmp_path):
    a = _fake_checkpoint(tmp_path, run="R0", step=1)
    b = _fake_checkpoint(tmp_path, run="R1", step=2)
    lin = json.loads((b / "lineage.json").read_text()); lin["resumed_from"] = str(a)
    (b / "lineage.json").write_text(json.dumps(lin))
    c = _fake_checkpoint(tmp_path, run="R2", step=3)
    lin = json.loads((c / "lineage.json").read_text()); lin["resumed_from"] = str(b)
    (c / "lineage.json").write_text(json.dumps(lin))
    info = H.resolve_checkpoint(c)
    got = [Path(p).resolve() for p in H.ancestors(info)]
    assert got == [a.resolve(), b.resolve()]                 # oldest first
    none = H.earlier_checkpoint(H.resolve_checkpoint(a))
    assert none["checkpoint"] is None


# ------------------------------------------------------------------ suite and watcher
def test_suite_records_a_failing_probe_and_runs_the_rest(tmp_path, monkeypatch):
    from mimo_bridge.tools import gof_probe_suite as S
    info = dict(run="R1", name="ckpt_step_7", checkpoint=str(tmp_path / "ck"), lineage_step=7)
    monkeypatch.setattr(H, "resolve_checkpoint", lambda ck: info)
    calls = []

    def fake_module(name):
        probe = name.rsplit("_", 1)[1]

        def run(ckpt, root=None, **kw):
            calls.append(probe)
            if probe == "body":
                raise RuntimeError("boom")
            body = dict(headline=H.headline(probe, 0.0, "share", 0.0, 1.0, "x"), status="measured", arms={},
                        controls={}, config={})
            body["path"] = str(H.write_result(info, probe, S.ITEMS[probe], body, root=root))
            return body
        return SimpleNamespace(run=run)

    monkeypatch.setattr(S.importlib, "import_module", fake_module)
    out = S.run_suite("ck", root=tmp_path / "gof")
    assert calls == list(S.PROBES)
    assert out["probes"]["body"]["status"].startswith("failed: RuntimeError: boom")
    assert all(H.indexed(info["checkpoint"], p, tmp_path / "gof") for p in S.PROBES)   # failure indexed too
    assert (tmp_path / "gof" / "R1" / "ckpt_step_7" / "suite.json").exists()
    # a first failure of this code is run once more (verify-fix-probes 2026-09-23: a memory shortage
    # need not repeat); after H.SAME_CODE_ATTEMPTS failures of the same code it waits for other code
    again = S.run_suite("ck", root=tmp_path / "gof", skip_indexed=True)
    assert again["probes"]["body"]["status"].startswith("failed: RuntimeError: boom") and calls[-1] == "body"
    assert all(again["probes"][p]["status"].startswith("skipped") for p in S.PROBES if p != "body")
    third = S.run_suite("ck", root=tmp_path / "gof", skip_indexed=True)
    assert all(v["status"].startswith("skipped") for v in third["probes"].values())
    # a partial re-run keeps the other probes' entries in suite.json
    S.run_suite("ck", root=tmp_path / "gof", probes=("gaze",))
    summary = json.loads((tmp_path / "gof" / "R1" / "ckpt_step_7" / "suite.json").read_text(encoding="utf-8"))
    assert set(summary["probes"]) == set(S.PROBES) and len(summary["runs"]) == 4


def _complete(ck: Path, step: int):
    for f in ("model.safetensors", "optimizer.pt", "runtime_state.pt"):
        (ck / f).write_bytes(b"x")
    (ck / "meta.json").write_text(json.dumps({"step": step}), encoding="utf-8")
    (ck / "model_receipt.json").write_text(json.dumps({}), encoding="utf-8")


def test_watcher_picks_complete_aged_unindexed_new_checkpoints(tmp_path):
    from mimo_bridge.tools import gof_probe_watch as W
    exp = tmp_path / "experiments"
    old = _fake_checkpoint(tmp_path, run="L_seg1", step=10)
    _complete(old, 10)
    new = _fake_checkpoint(tmp_path, run="L_seg2", step=20)
    _complete(new, 20)
    partial = _fake_checkpoint(tmp_path, run="L_seg2", step=30)          # no optimizer: incomplete
    (partial / "meta.json").write_text(json.dumps({"step": 30}), encoding="utf-8")
    other = _fake_checkpoint(tmp_path, run="M_seg1", step=40)
    _complete(other, 40)
    found = [p.name for _, p in W.complete_checkpoints(exp, "L_")]
    assert found == ["ckpt_step_10", "ckpt_step_20"]
    root = tmp_path / "gof"
    start_step = 10                     # the watcher began when step 10 was the newest checkpoint
    now = time.time()
    assert W.pending(exp, "L_", root, start_step, now) == []            # too young
    for p in (old, new):
        for f in p.iterdir():
            os.utime(f, (now - 120, now - 120))
    todo = W.pending(exp, "L_", root, start_step, now)
    assert [p.name for _, p in todo] == ["ckpt_step_20"]                 # existing one left alone
    info = dict(run="L_seg2", name="ckpt_step_20", checkpoint=str(new), lineage_step=20)
    for probe in W.PROBES:
        H.write_result(info, probe, 7, dict(headline=None, status="measured"), root=root)
    assert W.pending(exp, "L_", root, start_step, now) == []             # all four indexed
    assert [p.name for _, p in W.pending(exp, "L_", root, 0, now)] == ["ckpt_step_10"]


# ------------------------------------------------------------------ the real checkpoint
# found wherever the archiver moved it (tests/saved_checkpoints.py); skipped only where it is nowhere
from mimo_bridge.tests.saved_checkpoints import h84_344400  # noqa: E402

H84_FOUND = h84_344400()
H84 = H84_FOUND.path
NEEDS_H84 = H84_FOUND.mark()


@pytest.mark.slow
@NEEDS_H84
def test_h84_episode_start_is_exact_and_on_the_trained_footing():
    """One body from the episode companion: the restored moment repeats exactly, the next-moment
    vision error sits at the live run's level, and the probe world records nothing."""
    H84_FOUND.require()
    frozen = H.load_frozen(H84)
    assert frozen.load["reinitialized"] == []
    comps = H.components(frozen.model)
    assert comps["tenure_pose_head"]["present"] and comps["persistent_memory"]["present"]
    starts = H.start_states(frozen, 1, gaze_reflex=False, saccades=False)   # held: dropping it closes the session
    session, snap = next(starts)
    try:
        assert session.start.startswith("episode companion")
        assert session.kw["caregiver_record_dir"] == "" and "gof_probes" in session.kw["run_dir"]
        session.restore(snap)
        mom = session.moment()
        assert mom.context_source.startswith("lesson context")
        order = session.hold_order()
        p1 = H.predict(frozen.model, mom, order)
        a1 = session.run_arm(snap, order, steps=2)
        session.restore(snap)
        p2 = H.predict(frozen.model, session.moment(), order)
        a2 = session.run_arm(snap, order, steps=2)
        for m in a1[0]["tokens"]:
            assert torch.equal(a1[-1]["tokens"][m], a2[-1]["tokens"][m])
        for m in p1["preds"]:
            assert torch.equal(p1["preds"][m], p2["preds"][m])
        vis = H.token_scores(p1["preds"], a1[0]["tokens"], mom.tokens, ["vision"])["vision"]
        # live segment 31 read 0.0081 in its first 100 steps; a fresh world read 0.21-0.24
        assert vis["pred_err"] < 0.05
    finally:
        starts.close()


@pytest.mark.slow
@NEEDS_H84
def test_eye_step_response_is_sampled_every_substep():
    """Physics only (no model): the run's eye servo told to jump; every jump is sampled at the
    physics substep and ends within 5% of its size by 400 ms (measured 2026-09-22: 130-195 ms)."""
    H84_FOUND.require()
    from mimo_bridge.tools import causal_test as CT
    from mimo_bridge.tools import gof_probe_gaze as G
    saved, _, _ = CT._cfg_from_run(str(H84))
    ns = CT._probe_configuration(saved, "cpu", 0, False, None)
    frozen = H.Frozen(model=None, args=ns, info={}, limits=dict(phase_name="mature", age_months=24.0,
                      strength_limit=1.0, rom_limit=1.0), limits_source="test", load={}, build_s=0.0, load_s=0.0)
    out = G.step_response(frozen, amplitudes=(5.0, 20.0))
    assert set(out["per_jump"]) == {"az_5", "az_20", "el_5", "el_20"}
    for r in out["per_jump"].values():
        assert r["sample_ms"] == pytest.approx(5.0, abs=0.01)
        assert r["final_error_deg"] < 0.05 * abs(r["jump_deg"])
        assert r["peak_speed_deg_s"] > 0
    assert out["summary"]["peak_speed_deg_s_max"] > out["per_jump"]["az_5"]["peak_speed_deg_s"]


def test_predict_uses_the_models_own_efference_entry_point():
    """With efference heads, predict() hands the model's predict_efference the same moment
    (tokens, context, pose, memory bias) and the order in execution units."""
    d = 4
    seen = {}

    class Enc:
        def encode(self, frame, mask):
            return {"motor": torch.zeros(1, 1, d)}

    class Heads:
        head_modalities = ("vision",)

        def __call__(self, last, current=None, who=None):
            return {m: v for m, v in (current or {}).items()}

    def fm(tokens, ctx, **kw):
        seen["fm_kw"] = kw
        return None, torch.zeros(1, len(tokens), d), None

    def predict_efference(order_cm, state=None):
        seen["order"] = order_cm
        seen["state"] = state
        return {"body_tenure_cm": torch.ones(1, 26, 3), "gaze_deg": torch.zeros(1, 3, 2),
                "handle_names": ["x"] * 26}

    model = SimpleNamespace(encoder_registry=Enc(), active=["vision", "motor"], d_model=d, forward_model=fm,
                            prediction_heads=Heads(), tenure_heads=None, GAZE_IDX=23,
                            efference=object(), predict_efference=predict_efference)
    mom = H.Moment(tokens={"vision": torch.ones(1, d)}, ctx=torch.zeros(1, 6, d), ach=torch.zeros(1, 26, 10),
                   mem=torch.full((1, 2, d), 0.25), context_source="test")
    order = np.zeros((1, 26, 9), np.float32)
    order[..., 6] = 1.0
    order[0, 3, 0] = 12.0                          # cm: execution units reach the model unchanged
    out = H.predict(model, mom, order)
    assert torch.equal(seen["fm_kw"]["memory_bias"], mom.mem)
    assert seen["state"].memory_bias is mom.mem and seen["state"].context is mom.ctx
    assert float(np.asarray(seen["order"])[0, 3, 0]) == 12.0
    assert out["eff"]["body_tenure_cm"].shape == (26, 3) and "handle_names" not in out["eff"]


def test_body_scoring_reads_the_efference_head_against_holding_still():
    """_score's efference branch: the head's predicted change relative to holding still,
    projected on the real change relative to holding still."""
    from mimo_bridge.tools import gof_probe_body as B
    names = ["a", "b"]
    q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (2, 1))

    def arm(real_cm, eff_cm):
        ro = dict(tokens={"proprioception": torch.zeros(1, 3)}, pos_cm=np.asarray(real_cm, float), quat=q)
        return dict(pred=dict(preds={"proprioception": torch.zeros(1, 3)},
                              eff={"body_tenure_cm": np.asarray(eff_cm, float), "body_tenure_deg": np.zeros((2, 3))}),
                    readouts=[ro], change=dict(cm=np.asarray(real_cm, float), cm_norm=np.linalg.norm(real_cm, axis=-1),
                                               deg=np.zeros(2)), lesson_50ms={})
    hold = arm([[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]])
    moved = arm([[2, 0, 0], [0, 0, 0]], [[1, 0, 0], [0, 0, 0]])     # foresaw half the real move
    res = dict(arms={"hold": hold, "elbow_a": moved}, null_pred={},
               plans={"elbow": dict(handle="a", turns_deg=[20.0, -20.0])})
    sc = B._score(res, names)["arms"]["elbow_a"]
    assert sc["dot_eff_pred_actual_cm2"] == pytest.approx(2.0) and sc["dot_eff_actual_actual_cm2"] == pytest.approx(4.0)
    assert H.pooled_share([sc], "dot_eff_pred_actual_cm2", "dot_eff_actual_actual_cm2") == pytest.approx(0.5)
    assert sc["efference_tenure_cm_error"] == pytest.approx(0.5)      # |1-2| on one of two handles
