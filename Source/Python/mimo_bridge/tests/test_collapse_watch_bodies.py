# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The collapse watch on every body (LemLite._token_ring_bodies, token_temporal_std_bodies; the stats record's
token_temporal_std_bodies; docs/gain_of_function_v4/agents/proprio_token_2026-09-25.md).

What is pinned: body 0's entry is the existing collapse watch; the window is the watch's own (200 samples, every
5th step); a change of body count starts a new window; the episode companion carries it and a restore keeps it on
the CPU; a companion without a sense's deque gets one at the next sample (no KeyError); the trainer writes it into
every report; on a real body the step fills it on the sampling steps only; reading it every step changes no weight,
no optimizer tensor and no loss (the proof across the two code trees is proprio_token_bit_identity.py)."""
import inspect
from collections import deque

import numpy as np
import pytest
import torch


def _lite(**kw):
    torch.set_num_threads(1)
    from mimo_bridge.lem_model import LemLite
    return LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                   body_layout="trunk_v1", **kw)


def _fill(m, B=3, n=30, seed=0):
    """What LemLite.step appends on n sampling steps: body 0's token to token_ring, every body's to the new ring."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        for s in m.active:
            x = torch.randn(B, m.d_model, generator=g)
            m.token_ring[s].append(x[0].clone())
            m._token_ring_bodies[s].append(x.clone())


def test_body_zero_is_the_collapse_watch_and_every_body_is_read():
    m = _lite()
    assert set(m._token_ring_bodies) == set(m.active)
    assert all(r.maxlen == 200 for r in m._token_ring_bodies.values()), "the watch's own window"
    assert m.token_temporal_std_bodies() == {}, "fewer than 8 samples: no reading, as the watch"
    _fill(m, B=3, n=30)
    per = m.token_temporal_std_bodies()
    one = m.token_temporal_std()
    assert set(per) == set(one) == set(m.active)
    for s in m.active:
        assert len(per[s]) == 3
        assert per[s][0] == pytest.approx(one[s], rel=1e-6)
    # a body whose token never changes reads 0
    for s in m.active:
        for t in m._token_ring_bodies[s]:
            t[2] = 1.5
    assert all(m.token_temporal_std_bodies()[s][2] == 0.0 for s in m.active)


def test_a_change_of_body_count_starts_a_new_window():
    m = _lite()
    _fill(m, B=3, n=20)
    _fill(m, B=2, n=5, seed=1)
    assert m.token_temporal_std_bodies() == {}, "5 samples of the new body count: not yet a reading"
    _fill(m, B=2, n=5, seed=2)
    per = m.token_temporal_std_bodies()
    assert all(len(v) == 2 for v in per.values())


def test_the_companion_carries_it_and_a_restore_keeps_it_on_the_cpu():
    from mimo_bridge import continuation_state as CS
    m = _lite()
    _fill(m, B=3, n=12)
    state = CS.capture_learner(m, include_replay=False)
    assert "_token_ring_bodies" in state["data"]
    assert "_token_ring_bodies" in CS.CPU_DATA_NAMES, "kept on the CPU after a CUDA resume, as token_ring"
    new = _lite()
    CS.restore_learner(new, state)
    assert new.token_temporal_std_bodies() == m.token_temporal_std_bodies()
    assert all(t.device.type == "cpu" for r in new._token_ring_bodies.values() for t in r)


def test_a_companion_without_a_senses_deque_gets_one_at_the_next_sample():
    src = inspect.getsource(__import__("mimo_bridge.lem_model", fromlist=["LemLite"]).LemLite.step)
    assert "self._token_ring_bodies.setdefault(m, deque(maxlen=200)).append(" in src
    # the append sits beside token_ring's, on the same sampling steps
    block = src.split("if self.step_count % 5 == 0:")[1][:600]
    assert "self.token_ring[m].append(tokens_raw[m].detach().float().squeeze(1)[0].cpu())" in block
    assert "tokens_raw[m].detach().float().squeeze(1).cpu())" in block


def test_the_trainer_writes_it_into_every_report():
    from mimo_bridge import train_mimo as TM
    src = inspect.getsource(TM)
    assert '"token_temporal_std_bodies": model.token_temporal_std_bodies(),' in src


SIX = ("vision", "tactile", "audio", "proprioception", "motor", "vestibular")


def test_the_explainer_draws_the_quietest_body_per_sense_with_its_ranges():
    from mimo_bridge.tools import build_explainer as BX
    recs = []
    for i in range(6):
        recs.append({"type": "mimo", "step": 100 * (i + 1),
                     "token_temporal_std": {m: 0.4 for m in SIX},
                     "token_temporal_std_bodies": {m: [0.4, 0.09 + 0.001 * i, 0.42] for m in SIX}})
    out = BX.build_series({"mimo": recs, "training": [], "launch": []}, [], modalities=list(SIX))
    ch = next(c for c in out["charts"] if c["label"].startswith("7b"))
    assert ch["doc"] == "chart.collapse_bodies" and ch["unit"] == "%" and ch["scale"] == 100.0
    assert [s["key"] for s in ch["series"]] == ["token_temporal_std_bodies.%s.min" % m for m in SIX]
    assert ch["series"][0]["points"][0][1] == pytest.approx(0.09)
    doc = BX.DOC_CHARTS["chart.collapse_bodies"]["w"]
    for part in ("0% = that body's token is the same at every model step", "vision 32-51%", "tactile 17-50%",
                 "audio 34-48%", "proprioception 26-52%", "motor 47-68%", "proprioception 4-20%."):
        assert part in doc, part
    # the ranges split the windows at the hourly watcher's own lines, and the text names every line
    from mimo_bridge import deaf_senses as DS
    for sense, line in DS.LINES.items():
        named = DS.word(sense) + " " + DS.line_pct(line)
        assert named in doc or (sense == "vision" and "vision and proprioception " + DS.line_pct(line) in doc), sense
    # records from before the reading existed: the chart says the feature is off, it is not dropped
    old = [{k: v for k, v in r.items() if k != "token_temporal_std_bodies"} for r in recs]
    out2 = BX.build_series({"mimo": old, "training": [], "launch": []}, [], modalities=list(SIX))
    assert not any(c["label"].startswith("7b") for c in out2["charts"])
    assert not any(lab.startswith("7b") for lab in out2["dropped"])


def _drive(m, tmp_path, n, read_every_step=False):
    from mimo_bridge.datasource import SingleEnvSource
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "cw"), tag="_cw", vision=False, seed=5)
    losses = []
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9))
        dev["exploration_noise"] = 0.05
        held = prev = None
        for i in range(n):
            res = m.step(frames, dev, train=True, step_index=i, executed_action=prev)
            losses.append(float(res.loss))
            if read_every_step:
                m.token_temporal_std_bodies()
                m.token_temporal_std()
            if i % 5 == 0 or held is None:
                held = res.motor_for_action
            for _ in range(5):
                frames = src.step(held, res.gaze_out, None, render=False)
            prev = held
    finally:
        src.close()
    return losses


@pytest.mark.slow
def test_on_a_real_body_the_step_fills_it_and_reading_it_changes_no_training(tmp_path):
    from mimo_bridge.lem_model import LemLite
    kw = dict(device="cpu", seed=0, intention=False, imagination=False, next_frame_loss=True,
              overrides={"training": {"steps_per_phase": 3000}})
    torch.manual_seed(0)
    a = LemLite(**kw)
    del a._token_ring_bodies[a.active[-1]]           # as if restored from a companion without that sense
    torch.manual_seed(1)
    la = _drive(a, tmp_path, 45, read_every_step=True)
    torch.manual_seed(0)
    b = LemLite(**kw)
    torch.manual_seed(1)
    lb = _drive(b, tmp_path, 45, read_every_step=False)
    # the step appended on the watch's sampling steps (step_count 0, 5, ..., 40: 9 samples), every sense
    assert all(len(r) == 9 for r in a._token_ring_bodies.values()), {k: len(v) for k, v in a._token_ring_bodies.items()}
    assert set(a._token_ring_bodies) == set(a.active), "the missing sense's deque was made at the next sample"
    per, one = a.token_temporal_std_bodies(), a.token_temporal_std()
    assert set(per) == set(a.active) and all(len(v) == 1 for v in per.values())
    for s in a.active:
        assert per[s][0] == pytest.approx(one[s], rel=1e-6)
    # reading the per-body watch every step changed nothing the model learns
    assert la == lb
    for p, q in zip(a.params, b.params):
        assert torch.equal(p, q)
    for p, q in zip(a.params, b.params):
        sa, sb = a.optimizer.state.get(p, {}), b.optimizer.state.get(q, {})
        assert set(sa) == set(sb)
        for key in sa:
            if torch.is_tensor(sa[key]):
                assert torch.equal(sa[key], sb[key])
