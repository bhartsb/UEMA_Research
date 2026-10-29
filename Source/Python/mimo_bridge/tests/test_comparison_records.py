# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The comparison records (mimo_bridge/comparison_records.py; design note
docs/gain_of_function_v4/comparison_records/COMPARISON_RECORDS.md).

CPU only. Recorded inputs: comparison_records_lane_vision.npz (the learning-on measurement's model-lane vision column,
3 runs x 1,000 model steps x 8 bodies; made by docs/gain_of_function_v4/comparison_records/make_spike_fixture.py).
Stub inputs: hand-built rows. The learner: tests/swap_test_fixture.py (H84's learner flags at width 32, 3 bodies,
seeded synthetic frames), with a replay ring, the observer's per-body errors and the spike trigger.

What the tests show:
  - each record is written with its shape and its values (the lane's two new columns, the step and replay-pass lines,
    the spike event's inputs and weights), checked against an independent computation of the same numbers;
  - the spike rule fires once on the recorded finger run at model step 482,108 (body 0) and never on the two recorded
    control runs; on planted rows it fires at the third consecutive crossing and keeps its rate limit;
  - computing the records leaves the learner's losses, gradient norms, commands, random streams and every weight,
    buffer and Adam running gradient average bit for bit as they are without them (records on against off, over replay passes, lane
    rows, spike captures and a weights snapshot);
  - the bundle review's fixes of Oct 10: a line holding a non-finite number is written with it named and the writer
    stays on (FIX-1); held lines reach disk at every flush, the trainer's stats record (FIX-2); a record that fails is
    counted, recording continues and training is unchanged (FIX-3); a spike event is refused while the drive lacks
    its size plus the reserve (FIX-6).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mimo_bridge import comparison_records as CR
from mimo_bridge.observer_model_lane import ModelLane, SENSES, read_model_chunk

HERE = Path(__file__).resolve().parent
LANE = HERE / "comparison_records_lane_vision.npz"
LANE_SHA256 = "38e1055d76c7e5528df93c7adbe41ec9c894eede81d33be07c54d34ac14737e7"


# ------------------------------------------------------------------------------ the spike rule
def _rows(n_bodies=8, value=0.003):
    return np.full(n_bodies, value), np.ones(n_bodies)


def test_the_spike_rule_fires_at_the_third_consecutive_crossing_and_keeps_its_rate_limit(tmp_path):
    sp = CR.VisionSpike(str(tmp_path), line=0.14, run=3, max_events=3, min_gap=300)
    fired, captured = [], []

    def feed(step, over=(), value=0.25):
        e, ok = _rows()
        for b in over:
            e[b] = value
        if sp.observe(step, e, ok, e / 2):
            captured.append(step)
            sp.keep(step, dict(step=step))
        ev = sp.judge(step)
        if ev is not None:
            fired.append(ev)
            sp.pending = None

    for s in range(1000, 1010):
        feed(s, over=(0,) if 1003 <= s <= 1005 else ())
    assert [e["step"] for e in fired] == [1005] and fired[0]["bodies"] == [0] and fired[0]["first_step"] == 1003
    assert [s for s, _ in fired[0]["captures"]] == [1003, 1004, 1005]
    # one crossing, two crossings, a run broken by a body that was not graded, a run broken by a gap in model steps:
    # none fires
    for s in range(1400, 1420):
        feed(s, over=(1,) if s in (1401, 1404, 1405, 1408, 1409) else ())
    e, ok = _rows()
    e[2] = 0.3
    for s in (1430, 1431):
        sp.observe(s, e, ok)
        assert sp.judge(s) is None
    ok2 = ok.copy()
    ok2[2] = 0
    sp.observe(1432, e, ok2)                                        # body 2 not graded: its count ends
    assert sp.judge(1432) is None
    sp.observe(1440, e, ok)                                         # not the model step after 1432: counts restart
    sp.observe(1441, e, ok)
    assert sp.judge(1441) is None
    e_nan = e.copy()
    e_nan[2] = np.nan
    sp.observe(1442, e_nan, ok)
    assert sp.judge(1442) is None and int(sp.counts[2]) == 0
    assert len(fired) == 1
    # the rate limit: a run 100 model steps after the event is not kept (and nothing is captured for it)...
    n_cap = len(captured)
    for s in range(1100, 1106):
        feed(s, over=(3,) if 1101 <= s <= 1103 else ())
    assert len(fired) == 1 and sp.suppressed == 1 and len(captured) == n_cap
    # ...300 model steps after it is; then a third; a fourth is refused: 3 events per run at most
    for start in (1600, 1900, 2200):
        for s in range(start, start + 4):
            feed(s, over=(4,) if start <= s <= start + 2 else ())
    assert [e["step"] for e in fired] == [1005, 1602, 1902]
    assert sp.suppressed == 2 and not sp.allowed(10 ** 6)
    lines = [json.loads(x) for x in Path(sp.crossings.path).read_text(encoding="utf-8").splitlines()]
    kept = [x for x in lines if "kept" in x]
    assert [x["kept"] for x in kept] == [True, False, True, True, False]
    assert "fewer than 300" in kept[1]["why"] and "3 events kept already" in kept[-1]["why"]
    crossing = [x for x in lines if x.get("step") == 1003 and "bodies" in x][0]
    assert crossing["bodies"] == [0] and crossing["errors"] == [0.25] and crossing["persistence"] == [0.125]


def test_the_declared_spike_rule_on_the_recorded_runs_fires_once_in_the_finger_run_and_never_in_the_controls():
    """The rule's ground (comparison_records.SPIKE_GROUND), re-read from the recorded lane: the declared rule fires at
    model step 482,108 on body 0 in the finger run and nowhere in the two control runs; a single-transition line at the
    same height would fire in every run."""
    import hashlib
    assert hashlib.sha256(LANE.read_bytes()).hexdigest() == LANE_SHA256
    z = np.load(LANE)
    for run, want in (("control", []), ("finger", [(482108, [0])]), ("control_again", [])):
        sp = CR.VisionSpike(None, line=CR.SPIKE_LINE, run=CR.SPIKE_RUN, max_events=CR.SPIKE_MAX_SNAPSHOTS,
                            min_gap=CR.SPIKE_MIN_GAP_STEPS)
        got = []
        for step, row in zip(z[run + "_step"], z[run + "_vision"]):
            sp.observe(int(step), row, np.isfinite(row).astype(float))
            ev = sp.judge(int(step))
            if ev is not None:
                got.append((ev["step"], ev["bodies"]))
                sp.pending = None
        assert got == want, run
        single = CR.VisionSpike(None, line=CR.SPIKE_LINE, run=1, max_events=10 ** 6, min_gap=0)
        n = 0
        for step, row in zip(z[run + "_step"], z[run + "_vision"]):
            single.observe(int(step), row, np.isfinite(row).astype(float))
            n += single.judge(int(step)) is not None
        assert n >= 3, (run, n)
    # the ground's numbers, re-measured: the largest three-step minimum outside the event, the event's
    v = {r: np.nan_to_num(z[r + "_vision"].astype(np.float64), nan=-1.0) for r in ("control", "finger", "control_again")}
    s = {r: z[r + "_step"] for r in v}

    def best3(r, inside):
        out = -1.0
        for b in range(8):
            for i in range(2, len(s[r])):
                ins = r == "finger" and 482100 <= s[r][i] <= 482115
                if ins == inside:
                    out = max(out, float(v[r][i - 2:i + 1, b].min()))
        return out
    outside = max(best3(r, False) for r in v)
    assert abs(outside - CR.SPIKE_GROUND["three_step_minimum_max_outside_event"]["control"][0]) < 5e-5
    assert abs(best3("finger", True) - CR.SPIKE_GROUND["three_step_minimum_in_event"][0]) < 5e-5
    assert outside < CR.SPIKE_LINE < best3("finger", True)


# ------------------------------------------------------------------------------ the lane's columns (stub inputs)
class _StubModel:
    def __init__(self, B, row):
        self.observer_moment = row
        self.last_next_pe = {"vision": 0.5}


def test_the_lane_writes_the_persistence_and_no_copy_columns_and_reads_an_older_row_as_not_read(tmp_path):
    B = 3
    lane = ModelLane(str(tmp_path / "model"), run="t", n_envs=B, handle_names=["h"] * 26, chunk_steps=4)
    senses = ["vision", "proprioception"]
    per = np.array([[0.1, 0.2, 0.3], [0.01, 0.02, 0.03]], np.float32)
    pers = np.array([[0.4, 0.5, 0.6], [0.04, 0.05, 0.06]], np.float32)
    lane.add(0, _StubModel(B, dict(step=0, senses=senses, next_pe_per_body=per, persistence_per_body=pers,
                                   vision_without_copy_per_body=np.array([0.7, 0.8, 0.9], np.float32),
                                   gaze_copy_applied=True)),
             author="planner", latched=True, order_seq=1, rest=False, bout=True)
    # a row written by code before the records (only next_pe_per_body): NaN columns and applied -1
    lane.add(1, _StubModel(B, dict(step=1, senses=senses, next_pe_per_body=per)),
             author="planner", latched=False, order_seq=1, rest=False, bout=True)
    # a row whose records' columns cannot be read (FIX-3): they read NaN and -1, the failure is counted, the lane's
    # own column stands and the lane stays on
    lane.add(2, _StubModel(B, dict(step=2, senses=senses, next_pe_per_body=per, persistence_per_body="not numbers",
                                   gaze_copy_applied=True)),
             author="planner", latched=False, order_seq=1, rest=False, bout=True)
    assert lane.enabled and lane.error is None and CR.failures_of(lane)["lane_columns"]["count"] == 1
    lane.close()
    status = json.loads(sorted((tmp_path / "model").glob("status_*.json"))[0].read_text(encoding="utf-8"))
    assert status["comparison_records_failures"]["lane_columns"]["count"] == 1
    (chunk,) = sorted((tmp_path / "model").glob("chunk_*.npz"))
    z = read_model_chunk(str(chunk))
    assert np.array_equal(z["next_pe_body"][2, :, SENSES.index("vision")], per[0])
    assert np.isnan(z["persist_pe_body"][2]).all() and int(z["gaze_copy_applied"][2]) == -1
    z = {k: (v[:2] if k in ("persist_pe_body", "vision_no_copy_body", "gaze_copy_applied", "next_pe_body") else v)
         for k, v in z.items()}
    assert z["persist_pe_body"].shape == (2, B, len(SENSES)) and z["vision_no_copy_body"].shape == (2, B)
    v, p = SENSES.index("vision"), SENSES.index("proprioception")
    assert np.array_equal(z["persist_pe_body"][0, :, v], pers[0]) and np.array_equal(z["persist_pe_body"][0, :, p],
                                                                                      pers[1])
    assert np.isnan(z["persist_pe_body"][0, :, SENSES.index("audio")]).all()
    assert np.array_equal(z["vision_no_copy_body"][0], np.array([0.7, 0.8, 0.9], np.float32))
    assert z["gaze_copy_applied"].tolist() == [1, -1]
    assert np.isnan(z["persist_pe_body"][1]).all() and np.isnan(z["vision_no_copy_body"][1]).all()
    assert np.array_equal(z["next_pe_body"][1, :, v], per[0])
    header = json.loads(sorted((tmp_path / "model").glob("header_*.json"))[0].read_text(encoding="utf-8"))
    assert header["comparison_records"]["version"] == CR.RECORDS_VERSION


# ------------------------------------------------------------------------------ the replay windows (stub ring)
def test_each_window_is_named_by_its_body_and_model_steps_and_the_sampler_draws_no_more_than_before():
    from mimo_bridge.replay import ReplayMemory
    mods, d, H, size, ctx = ["vision", "motor"], 4, 26, 12, 3
    ring = ReplayMemory(mods, d, n_handles=H, size_per_env=size, n_envs=2)
    slots = np.full(size, -1, dtype=np.int64)
    for k in range(20):                         # 20 adds into 12 slots: the ring wraps; model steps 100..119
        slots[ring.ptr % size] = 100 + k
        tok = {m: torch.full((2, 1, d), float(100 + k) + (0.5 if m == "motor" else 0.0)) for m in mods}
        tok["vision"][1] += 1000.0                                    # body 1's tokens differ from body 0's
        ring.add(tok, torch.zeros(2, H, 10), {}, executed=torch.zeros(2, H, 9), action_contract=1)
    torch.manual_seed(5)
    w = ring.sample_windows(6, ctx, "cpu")
    rng_after = torch.get_rng_state()
    rec = CR.window_record(ring, slots, w["source_env"], w["source_slot"], ctx)
    assert len(rec["body"]) == len(w["tokens"]) == 6
    for i, (b, first, target) in enumerate(zip(rec["body"], rec["first"], rec["target"])):
        assert target - first == ctx                                     # ctx + 1 consecutive model steps
        want = torch.tensor([float(s) + 1000.0 * b for s in range(first, target + 1)])
        assert torch.equal(w["tokens"][i, :, 0, 0], want), (i, b, first)  # the tokens ARE those model steps' entries
    assert all(not r for r in rec["restored"]) and rec["contract_min"] == rec["contract_max"]
    # the same seed without the bookkeeping's names gives the same windows and leaves the stream in the same place:
    # the sampler's draws are exactly torch.randint's, as before (the names are read from what it chose)
    torch.manual_seed(5)
    w2 = ring.sample_windows(6, ctx, "cpu")
    assert torch.equal(w2["tokens"], w["tokens"]) and torch.equal(torch.get_rng_state(), rng_after)
    # a slot this process never wrote (a ring restored from a companion): the model step is read as newest - entries
    # since, and the window is marked restored
    held = slots.copy()
    held[:] = -1
    held[(ring.ptr - 1) % size] = 119
    rec2 = CR.window_record(ring, held, w["source_env"], w["source_slot"], ctx)
    assert rec2["first"] == rec["first"] and rec2["target"] == rec["target"] and all(rec2["restored"])


# ------------------------------------------------------------------------------ the learner
@pytest.fixture(scope="module")
def fx():
    import importlib
    return importlib.import_module("mimo_bridge.tests.swap_test_fixture")


def _learner(fx):
    import random
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    m, a = fx.build(fx.argv())
    m.enable_replay(a.n_envs, size_per_env=64)
    m.observer_per_body_errors = True
    return m, a


def _run(fx, tmp, n, recording, spike=None, lane=False, spies=None):
    """n model steps of the fixture learner with a replay pass every 2 model steps (the trainer's rule: after model
    step k when k + 1 is even), the comparison records' writers as the trainer runs them when recording."""
    CR.RECORDING = recording
    try:
        m, a = _learner(fx)
        recs = lane_obj = None
        if recording:
            recs = CR.ComparisonRecords(str(tmp / "comparison_records"), run="t", n_envs=a.n_envs, reserve_bytes=0)
            if spike is not None:
                recs.spike = CR.VisionSpike(recs.spike_folder, reserve_bytes=0, **spike)
            recs.attach(m)
        if lane:
            lane_obj = ModelLane(str(tmp / "observer" / "model"), run="t", n_envs=a.n_envs,
                                 handle_names=list(m.handle_names), chunk_steps=10)
        passes = []

        def on_step(step, model, res):
            if spies is not None:
                spies["steps"].append(dict(step=step, grad=float(res.grad_norm),
                                           clip=dict(model.clip_norms or {}),
                                           audit={k: {kk: vv.detach().clone() for kk, vv in v.items()}
                                                  for k, v in model.prediction_audit_rows.items()}))
            if recs is not None:
                recs.after_step(step, model, res, snapshot=lambda k: CR.snapshot_weights(
                    model, k, str(tmp / "comparison_records" / "vision_spike" / "checkpoints")))
            if lane_obj is not None:
                lane_obj.add(step, model, author="planner", latched=False, order_seq=0, rest=False, bout=True)
            if (step + 1) % 2 == 0:
                rr = model.replay_update(batch=4)
                passes.append(None if rr is None else float(rr["replay_loss"]).hex())
                if recs is not None:
                    recs.replay_pass(step + 1, rr)
        rows = fx.drive(m, fx.SyntheticWorld(a.n_envs, seed=1), n, on_step=on_step)
        if recs is not None:
            recs.close()
        if lane_obj is not None:
            lane_obj.close()
        return dict(rows=rows, passes=passes, digest=fx.state_digest(m), model=m, records=recs)
    finally:
        CR.RECORDING = True


SPIKE = dict(line=0.0, run=2, max_events=2, min_gap=6)        # a line every graded transition crosses: it fires


def test_the_records_change_no_loss_gradient_command_random_stream_or_weight(fx, tmp_path):
    """Records on (every lane column, every step and pass line, spike captures of every crossing and two weights
    snapshots) against records off: the same 30 model steps and 15 replay passes, bit for bit."""
    on = _run(fx, tmp_path / "on", 30, True, spike=SPIKE, lane=True)
    off = _run(fx, tmp_path / "off", 30, False, lane=True)
    diff = [(a["step"], k) for a, b in zip(on["rows"], off["rows"]) for k in a if a[k] != b.get(k)]
    assert diff == [] and on["passes"] == off["passes"] and on["digest"] == off["digest"]
    assert sum(p is not None for p in on["passes"]) >= 8, "the replay passes ran"
    ev = [json.loads(x) for x in Path(on["records"].events.path).read_text(encoding="utf-8").splitlines()]
    assert len(ev) == 2 and all("weights" in e and Path(e["weights"]).is_dir() for e in ev), ev
    # off computes none of them
    assert off["model"].gaze_copy_now is None and not (tmp_path / "off" / "comparison_records").exists()


def test_each_record_holds_the_numbers_an_independent_reading_gives(fx, tmp_path, monkeypatch):
    from lem.model import efference as EF
    pooled = {}
    real_gauge = EF._gauge_vision

    def gauge(model, before, after, target, shuffled, size_matched=None):
        pooled[int(model.step_count)] = float((before.detach().float() - target.detach().float()).pow(2).mean())
        return real_gauge(model, before, after, target, shuffled, size_matched)
    monkeypatch.setattr(EF, "_gauge_vision", gauge)
    from mimo_bridge.replay import ReplayMemory
    drawn = []
    real_sample = ReplayMemory.sample_windows

    def sample(self, *a, **k):
        w = real_sample(self, *a, **k)
        drawn.append(None if w is None else dict(tokens=w["tokens"].clone(), env=list(w["source_env"]),
                                                 slot=list(w["source_slot"])))
        return w
    monkeypatch.setattr(ReplayMemory, "sample_windows", sample)
    spies = dict(steps=[])
    run = _run(fx, tmp_path, 24, True, spike=SPIKE, lane=True, spies=spies)
    m = run["model"]
    B = 3
    # record a and b: the lane's columns
    z = [read_model_chunk(str(c)) for c in sorted((tmp_path / "observer" / "model").glob("chunk_*.npz"))]
    steps = np.concatenate([c["step"] for c in z])
    pers = np.concatenate([c["persist_pe_body"] for c in z])
    nocopy = np.concatenate([c["vision_no_copy_body"] for c in z])
    per = np.concatenate([c["next_pe_body"] for c in z])
    applied = np.concatenate([c["gaze_copy_applied"] for c in z])
    assert pers.shape == (24, B, len(SENSES)) and nocopy.shape == (24, B)
    checked = 0
    for s in spies["steps"]:
        i = int(np.nonzero(steps == s["step"])[0][0])
        for sense, row in s["audit"].items():
            if sense in SENSES and "persistence_mse" in row:
                # the audit's paired_rows computes the same persistence error by its own code
                assert np.allclose(pers[i, :, SENSES.index(sense)], row["persistence_mse"].numpy(), rtol=1e-6,
                                   atol=0), (s["step"], sense)
                checked += 1
    assert checked >= 40
    v = SENSES.index("vision")
    graded = np.isfinite(nocopy[:, 0])
    assert graded.sum() >= 15 and (applied[graded] >= 0).all()
    same = graded & (applied == 0)
    assert np.array_equal(nocopy[same], per[same, :, v])                 # the copy added nothing: equal
    # (the fixture's learner starts at step_count 0 and counts one per model step: the spy's key is the model step)
    gauged = [j for j, s in enumerate(steps) if graded[j] and int(s) in pooled]
    matched = [j for j in gauged if math.isclose(float(nocopy[j].mean()), pooled[int(steps[j])],
                                                 rel_tol=1e-5, abs_tol=1e-9)]
    assert gauged and len(matched) == len(gauged), "the per-body errors average to the gaze copy's own pooled gauge"
    # record c: the step lines and the pass lines
    lines = [json.loads(x) for x in Path(run["records"].steps.path).read_text(encoding="utf-8").splitlines()]
    assert [r["step"] for r in lines] == list(range(24))
    for r, s in zip(lines, spies["steps"]):
        assert r["grad_norm"] == s["grad"] and r["group_norms"] == s["clip"] and set(r["group_norms"]) >= {"core"}
        assert r["group_norms"]["core"] == s["grad"] or s["grad"] == 0.0
    passes = [json.loads(x) for x in Path(run["records"].passes.path).read_text(encoding="utf-8").splitlines()]
    real = [w for w in drawn if w is not None]
    assert len(passes) == len(real) >= 6
    ctx = m.ctx_len
    for p, w in zip(passes, real):
        wr = p["windows"]
        assert wr["body"] == w["env"] and p["stepped"] is True and p["grad_norm"] == p["group_norms"]["core"]
        assert set(p["terms"]) <= set(SENSES) and p["motor"] is not None
        # the pass's loss is its terms, each at its --next-frame-norm weight (1 without one), plus the motor term
        weights = p["term_weights"] or {}
        want = sum(t * weights.get(k, 1.0) for k, t in p["terms"].items()) + p["motor"]
        assert math.isclose(want, p["loss"], rel_tol=1e-5, abs_tol=1e-7), (p["step"], want, p["loss"])
        for i, (b, first, target) in enumerate(zip(wr["body"], wr["first"], wr["target"])):
            assert target - first == ctx and not wr["restored"][i] and target < p["step"]
            slot0 = int(np.nonzero(m.replay_slot_step == first)[0][0])
            idx = [(slot0 + j) % m.replay.size for j in range(ctx + 1)]
            ring = m.replay.tokens[b, idx].float()
            assert torch.equal(ring, w["tokens"][i].float()), (p["step"], i)   # the window is those entries
    # record d: the events' inputs and weights
    ev = [json.loads(x) for x in Path(run["records"].events.path).read_text(encoding="utf-8").splitlines()]
    assert len(ev) == 2 and ev[1]["step"] - ev[0]["step"] >= SPIKE["min_gap"]
    for e in ev:
        assert e["captured_steps"] == [e["step"] - 1, e["step"]] and e["inputs_bytes"] > 0
        data = torch.load(e["inputs"], weights_only=False)
        assert [t["step"] for t in data["transitions"]] == e["captured_steps"]
        t = data["transitions"][-1]
        for key in ("prev_tokens", "target_tokens", "context", "command", "pose", "prediction", "vision_without_copy",
                    "raw_prev", "raw_now", "errors", "persistence"):
            assert key in t, key
        assert tuple(t["context"].shape)[0] == B and tuple(t["command"].shape) == (B, 26, 9)
        assert tuple(t["pose"].shape) == (B, 26, 10) and t["prediction"]["vision"].shape[0] == B
        i = int(np.nonzero(steps == t["step"])[0][0])
        assert np.allclose(np.asarray(t["errors"]), per[i, :, v], rtol=1e-6, atol=0)
        ck = Path(e["weights"])
        assert (ck / "model.safetensors").is_file() and ck.name == "ckpt_step_%d" % e["step"]
    # the snapshot left the learner's own checkpoint manager where it was
    assert getattr(m, "checkpoints", None) is None or "vision_spike" not in str(m.checkpoints._checkpoint_dir)


def _jsonl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.is_file() else []


def _status(folder):
    return json.loads(sorted(Path(folder).glob("status_*.json"))[0].read_text(encoding="utf-8"))


def test_failing_records_are_counted_recording_continues_and_training_is_unchanged(fx, tmp_path, monkeypatch):
    """FIX-3 of the bundle review of Oct 10: the records a and b (lane_moment), the pass line (pass_record) and the
    spike capture all fail at every model step; the learner trains bit for bit as with no records, every failure is
    counted where the status file reads it, and the writers keep writing."""
    def broken(*a, **k):
        raise RuntimeError("planted: the record fails")

    def broken_keep(self, step, inputs):
        raise RuntimeError("planted: the capture fails")
    monkeypatch.setattr(CR, "lane_moment", broken)
    monkeypatch.setattr(CR, "pass_record", broken)
    monkeypatch.setattr(CR.VisionSpike, "keep", broken_keep)
    on = _run(fx, tmp_path / "on", 24, True, spike=SPIKE, lane=True)
    off = _run(fx, tmp_path / "off", 24, False, lane=True)
    diff = [(a["step"], k) for a, b in zip(on["rows"], off["rows"]) for k in a if a[k] != b.get(k)]
    assert diff == [] and on["passes"] == off["passes"] and on["digest"] == off["digest"]
    real_passes = sum(p is not None for p in on["passes"])
    assert real_passes >= 6
    failures = CR.failures_of(on["model"])
    assert failures["lane"]["count"] >= 20 and "planted" in failures["lane"]["first_error"]
    assert failures["replay_pass"]["count"] == real_passes
    spike = on["records"].spike
    assert spike.enabled and spike.failures >= 20 and "planted" in spike.error
    # recording continues: a step line for every model step, a line naming each failed pass, the lane's own column
    folder = tmp_path / "on" / "comparison_records"
    assert [r["step"] for r in _jsonl(on["records"].steps.path)] == list(range(24))
    lines = _jsonl(on["records"].passes.path)
    assert len(lines) == real_passes and all(x.get("record_failed") is True for x in lines)
    status = _status(folder)
    assert status["failures"]["learner"]["lane"]["count"] == failures["lane"]["count"]
    assert status["failures"]["learner"]["replay_pass"]["count"] == real_passes
    assert status["failures"]["spike"]["spike"]["count"] == spike.failures and status["spike"]["enabled"]
    z = [read_model_chunk(str(c)) for c in sorted((tmp_path / "on" / "observer" / "model").glob("chunk_*.npz"))]
    per = np.concatenate([c["next_pe_body"] for c in z])
    assert np.isfinite(per[:, :, SENSES.index("proprioception")]).sum() >= 20, "the step's own per-body errors stand"
    assert np.isnan(np.concatenate([c["persist_pe_body"] for c in z])).all()
    assert list((folder / "vision_spike").glob("spike_error_*.json"))


# ------------------------------------------------------------------------------ the writers (FIX-1, FIX-2, FIX-6)
def test_a_line_with_a_non_finite_number_is_written_with_it_named_and_the_writer_stays_on(tmp_path):
    """FIX-1 of the bundle review of Oct 10 (its verify_linewriter.py case): 5 good lines, one with a NaN weight and an
    infinite loss, one no JSON form can hold, 4 good lines. Before the fix the NaN line turned the writer off and the 9
    good lines were lost uncounted."""
    import os

    class NoForm:
        def __repr__(self):
            raise RuntimeError("planted: no form at all")
    w = CR.LineWriter(str(tmp_path), "replay_passes", reserve_bytes=0, flush_lines=100)
    for i in range(5):
        w.add(dict(step=i, loss=1.0, term_weights={"vision": 1.0}))
    w.add(dict(step=5, loss=float("inf"), term_weights={"vision": 1.0, "vestibular": float("nan")},
               terms=[np.float32(0.5)]))
    w.add(dict(step=6, odd=NoForm()))
    for i in range(7, 11):
        w.add(dict(step=i, loss=1.0, term_weights={"vision": 1.0}))
    w.flush()
    lines = _jsonl(w.path)
    assert [x["step"] for x in lines] == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]
    bad = lines[5]
    assert bad["loss"] is None and bad["term_weights"] == {"vision": 1.0, "vestibular": None} and bad["terms"] == [0.5]
    assert sorted(bad["nonfinite"]) == ["loss", "term_weights.vestibular"]
    assert all("nonfinite" not in x for i, x in enumerate(lines) if i != 5)
    st = w.status()
    assert st["enabled"] and st["written"] == 10 and st["repaired"] == 1
    assert st["unserialisable"] == 1 and st["dropped"] == 1 and "planted" in st["error"]
    err = json.loads((tmp_path / ("replay_passes_error_%d.json" % os.getpid())).read_text(encoding="utf-8"))
    assert err["step"] == 6 and err["status"]["unserialisable"] == 1


def test_a_replay_pass_line_with_a_non_finite_weight_is_written_and_the_next_lines_follow(fx, tmp_path, monkeypatch):
    """FIX-1 through the learner: the second pass line's record carries a NaN --next-frame-norm weight (the route of
    the review: a NaN replay term under the vestibular revival's ceiling); it is written with the weight named and
    every later pass still has its line."""
    real = CR.pass_record
    calls = []

    def planted(**kw):
        rec = real(**kw)
        calls.append(rec)
        if len(calls) == 2:
            rec["term_weights"] = dict(rec["term_weights"] or {}, vestibular=float("nan"))
        return rec
    monkeypatch.setattr(CR, "pass_record", planted)
    run = _run(fx, tmp_path, 24, True)
    lines = _jsonl(run["records"].passes.path)
    assert len(lines) == len(calls) >= 6
    assert lines[1]["term_weights"]["vestibular"] is None and "term_weights.vestibular" in lines[1]["nonfinite"]
    status = _status(tmp_path / "comparison_records")
    assert status["replay_passes"]["enabled"] and status["replay_passes"]["repaired"] >= 1
    assert status["replay_passes"]["written"] == len(calls) and status["replay_passes"]["dropped"] == 0


def test_held_lines_reach_disk_at_every_flush_with_the_status_file(tmp_path):
    """FIX-2: with FLUSH_LINES 100 nothing reaches disk on its own over 7 model steps; the trainer's stats record
    (ComparisonRecords.flush) writes every held step and pass line and the status file at once."""
    from types import SimpleNamespace
    recs = CR.ComparisonRecords(str(tmp_path / "cr"), run="t", n_envs=2, reserve_bytes=0)
    model = SimpleNamespace(clip_norms={"core": 0.5}, grad_clip=1.0, _nonfinite_steps=0)
    for k in range(7):
        recs.after_step(k, model, SimpleNamespace(loss=1.0, grad_norm=0.5))
        if (k + 1) % 2 == 0:
            recs.replay_pass(k + 1, {"replay_loss": 1.0, "record": dict(loss=1.0, stepped=True)})
    assert CR.FLUSH_LINES == 100 and not Path(recs.steps.path).exists() and not Path(recs.passes.path).exists()
    recs.flush()
    assert [x["step"] for x in _jsonl(recs.steps.path)] == list(range(7))
    assert [x["step"] for x in _jsonl(recs.passes.path)] == [2, 4, 6]
    st = _status(tmp_path / "cr")
    assert st["flushes"] == 1 and st["steps"]["written"] == 7 and st["replay_passes"]["written"] == 3
    assert st["steps"]["held"] == 0
    # a pass whose learner record failed is still a line
    recs.replay_pass(8, {"replay_loss": 1.0, "record": None})
    recs.close()
    assert _jsonl(recs.passes.path)[-1]["record_failed"] is True and _status(tmp_path / "cr")["flushes"] == 2


def test_a_spike_event_is_refused_when_the_drive_lacks_its_size_plus_the_reserve(tmp_path, monkeypatch):
    """FIX-6: no event is kept while the drive holds less than its estimated size (inputs' bytes + the weights'
    estimate) plus the reserve; the weights are refused alone when the inputs' write left too little; each refusal is
    the event's line; the estimate grows to the size of the last weights saved."""
    from types import SimpleNamespace
    recs = CR.ComparisonRecords(str(tmp_path / "cr"), run="t", n_envs=1, reserve_bytes=1000,
                                weights_bytes_estimate=5000)
    caps = [(7, dict(x=torch.zeros(100)))]                     # 400 bytes of numbers
    saved = []

    def snapshot(k):
        folder = tmp_path / ("w_%d" % k)
        folder.mkdir()
        (folder / "model.safetensors").write_bytes(b"0" * 7000)
        saved.append(k)
        return str(folder)
    frees = []

    def usage(path):
        return SimpleNamespace(free=frees.pop(0) if frees else 10 ** 12, total=0, used=0)
    monkeypatch.setattr(CR.shutil, "disk_usage", usage)
    frees[:] = [400 + 5000 + 1000 - 1]
    recs.write_event(dict(step=7, bodies=[0], first_step=7, captures=caps), None, snapshot)
    frees[:] = [400 + 5000 + 1000, 5000 + 1000 - 1]
    recs.write_event(dict(step=400, bodies=[0], first_step=400, captures=caps), None, snapshot)
    frees[:] = [400 + 5000 + 1000, 5000 + 1000]
    recs.write_event(dict(step=800, bodies=[0], first_step=800, captures=caps), None, snapshot)
    frees[:] = [400 + 7000 + 1000 - 1]
    recs.write_event(dict(step=1200, bodies=[0], first_step=1200, captures=caps), None, snapshot)
    ev = _jsonl(recs.events.path)
    assert [e["step"] for e in ev] == [7, 400, 800, 1200]
    assert "refused" in ev[0] and "inputs" not in ev[0] and ev[0]["inputs_bytes_estimate"] == 400
    assert not list(Path(recs.spike_folder).glob("inputs_step_7_*"))
    assert "refused" not in ev[1] and ev[1]["inputs_bytes"] > 0 and "weights_refused" in ev[1] and "weights" not in ev[1]
    assert ev[2]["weights"] and ev[2]["weights_bytes"] == 7000 and saved == [800]
    assert ev[3]["weights_bytes_estimate"] == 7000 and "refused" in ev[3]
    assert recs.refused_events == 2 and recs.refused_weights == 1
    recs.close()
    st = _status(tmp_path / "cr")
    assert st["refused_events"] == 2 and st["refused_weights"] == 1 and st["last_weights_bytes"] == 7000


# ------------------------------------------------------------------------------ preservation (NOTE-5, NOTE-4, NOTE-6)
def _policy(root: Path) -> Path:
    """A valid registered retention policy (lem/training/retention.py) at root/.checkpoint_retention.json, enrolling
    root/experiments as its run root with a budget of 1 byte: every snapshot it may retire, it retires."""
    for sub in ("experiments", "companions", "references"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    path = root / ".checkpoint_retention.json"
    path.write_text(json.dumps(dict(
        schema="checkpoint_retention_v1", checkpoint_run_roots=[str(root / "experiments")],
        companion_roots=[str(root / "companions")], reference_roots=[str(root / "references")],
        maximum_checkpoint_bytes=1, minimum_free_bytes=1, minimum_recovery_snapshots=2,
        audit_path=str(root / "retention_audit.jsonl"))), encoding="utf-8")
    return path


def test_spike_weights_survive_every_retention_rule(fx, tmp_path):
    """NOTE-5 of the bundle review: the configuration's keep_last_n (set to 1 here), a registered retention policy above
    the run folder, and the same model step saved by a second launch remove no spike snapshot; the same keep_last_n
    under the same policy does remove snapshots from a plain checkpoint manager's folder (the control)."""
    from lem.training import retention as RT
    from lem.training.checkpoint import CheckpointManager
    policy = _policy(tmp_path)
    m, a = _learner(fx)
    m.config["checkpoint"]["keep_last_n"] = 1
    run = tmp_path / "experiments" / "run"
    first = CR.ComparisonRecords(str(run / "comparison_records"), run="run", n_envs=a.n_envs, reserve_bytes=0)
    for k in (300, 600, 900):
        CR.snapshot_weights(m, k, first.weights_folder)
    import time as _time
    _time.sleep(0.01)
    second = CR.ComparisonRecords(str(run / "comparison_records"), run="run", n_envs=a.n_envs, reserve_bytes=0)
    assert second.launch != first.launch
    CR.snapshot_weights(m, 300, second.weights_folder)              # the same model step, a later launch
    # the control: a plain manager under the same policy and keep_last_n keeps 1 of 3
    plain = tmp_path / "plain_checkpoints"
    held = m.checkpoints
    m.checkpoints = CheckpointManager(m.config, str(plain))
    try:
        for k in (300, 600, 900):
            m.save_checkpoint(k)
    finally:
        m.checkpoints = held
    assert sorted(p.name for p in plain.glob("ckpt_step_*")) == ["ckpt_step_900"], "the rule bites a plain folder"
    # the registered policy's own pass, with a budget of 1 byte, reaches no spike folder
    p = RT.policy(policy)
    report = RT.maintain(policy, apply=True, processes=False)
    spike_root = run / "comparison_records" / "vision_spike" / "checkpoints"
    for launch in (first.launch, second.launch):
        assert not RT.owned_save_directory(p, spike_root / launch)
    assert sorted(p.name for p in (spike_root / first.launch).glob("ckpt_step_*")) == [
        "ckpt_step_300", "ckpt_step_600", "ckpt_step_900"]
    assert [p.name for p in (spike_root / second.launch).glob("ckpt_step_*")] == ["ckpt_step_300"]
    assert all((spike_root / first.launch / n / "model.safetensors").is_file() for n in ("ckpt_step_300", "ckpt_step_900"))
    assert report["retired"] == []


def test_an_inputs_file_is_complete_or_absent(tmp_path, monkeypatch):
    """NOTE-4: the inputs file is written under a temporary name and renamed when complete; a write that fails midway
    leaves no file under the final name, and the event's line names the error."""
    recs = CR.ComparisonRecords(str(tmp_path / "cr"), run="t", n_envs=1, reserve_bytes=0, weights_bytes_estimate=0)
    real_save = torch.save

    def half_save(obj, f, *a, **k):
        Path(f).write_bytes(b"half a file")
        raise OSError("planted: the process ended during the write")
    monkeypatch.setattr(torch, "save", half_save)
    recs.write_event(dict(step=5, bodies=[0], first_step=5, captures=[(5, dict(x=torch.zeros(4)))]), None, None)
    monkeypatch.setattr(torch, "save", real_save)
    recs.write_event(dict(step=400, bodies=[0], first_step=400, captures=[(400, dict(x=torch.zeros(4)))]), None, None)
    ev = _jsonl(recs.events.path)
    assert "planted" in ev[0]["error"] and "inputs" not in ev[0]
    folder = Path(recs.spike_folder)
    assert not (folder / ("inputs_step_5_%s.pt" % recs.launch)).exists()
    assert (folder / ("inputs_step_5_%s.pt.part" % recs.launch)).exists()
    assert Path(ev[1]["inputs"]).name == "inputs_step_400_%s.pt" % recs.launch and Path(ev[1]["inputs"]).is_file()
    assert torch.equal(torch.load(ev[1]["inputs"], weights_only=False)["transitions"][0]["x"], torch.zeros(4))
    assert not (folder / ("inputs_step_400_%s.pt.part" % recs.launch)).exists()


COMPARISON_PIN_STEPS = [481800, 484675]


def test_the_comparison_runs_pin_keeps_every_lane_and_world_chunk_of_the_run():
    """NOTE-6: the observer daemon's housekeeping removes digested world and model-lane chunks 72 hours after their
    file time unless a pin of Saved/mimo_bridge/observer/pins.json keeps them (tools/observer/pins.py). The pin the
    design note gives for each comparison run, steps 481,800 to 484,675 with no bodies named, is valid and keeps every
    lane chunk and every world chunk of every body of a run that records model steps 481,800 to 484,674 in chunks of
    100, with the daemon's own shifts for the lane."""
    from mimo_bridge.tools.observer import pins as PN
    from mimo_bridge.tools.observer import evidence as EV
    run = "H84_fresh_drivers_20260918_seg41j_0000"
    entry = dict(id="comparison_finger_481800", run=run, steps=list(COMPARISON_PIN_STEPS),
                 reason="matched finger comparison: the comparison records' lane columns and the world records",
                 owner="the coordinator")
    pins = PN.Pins(entries=[entry], now=0.0)
    assert not pins.invalid and not pins.blocks_all and pins.for_run("C:/x/" + run)
    first, last = 481800, 484674
    lane = {"chunk_%09d_1_%05d.npz" % (s, i): (s + EV.TRAINER_LANE_LAG, min(s + 99, last) + EV.TRAINER_LANE_LAG)
            for i, s in enumerate(range(first, last + 1, 100))}
    world = {"chunk_%09d_1_%05d.npz" % (s, i): (s, min(s + 99, last)) for i, s in enumerate(range(first, last + 1, 100))}
    kept_lane = PN.pinned_names(pins.for_run(run), lane, shift_lo=EV.TRAINER_LANE_LAG,
                                shift_hi=EV.TRAINER_LANE_LAG + PN.LANE_REACH_STEPS)
    assert set(kept_lane) == set(lane), sorted(set(lane) - set(kept_lane))
    for body in range(8):
        assert set(PN.pinned_names(pins.for_run(run), world, body=body)) == set(world)
