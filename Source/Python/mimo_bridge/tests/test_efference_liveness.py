# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""DESIGN section 10 rule 4 for the efference heads (review of b1, MAJOR 1): on the checkpoint's own
lesson inputs, every hidden layer is live and every attention pool is spread, at the start and after
100 training steps.

THE INPUTS are docs/gain_of_function_v3/measurements/efference_lesson_inputs_ckpt_step_344400.npz:
what the efference lesson and the gaze copy received on 120 lived moments of ckpt_step_344400 (two
bodies of its own episode companion, living under the model's own behaviour, learning off; the
capture script is capture_efference_fixture.py beside the file; the measurements are
in efference_liveness_ckpt_step_344400.json): the latent of S_t-1 + A_t-1, the order, the
achieved pose before and after, the eye angles before and after, vision's next-moment prediction
before the gaze copy and the real vision token.

THE LINES. A layer is live when more than half its units' input rose above -3 on the moments
(DESIGN section 10, rules 2 and 4). A pool is spread when its entropy, per head, averages more than
half of the even-spread maximum over the six tokens (the same majority line; on six tokens a head at
that line gives about three quarters of its weight to one token). The pools' cosine bound
(lem.model.efference.TokenPool) keeps every head at or above that line at worst; the test checks what
the heads actually keep. THE CONTROL that shows the pool line catches the defect: the same heads with
the bound lifted fall under it on the same moments within the same 100 steps (and the first version's
pools, efference_v1, fell from 77 and 90 to 23 and 36 of every 100 parts of an even spread in the
lane's measurement, efference_liveness_ckpt_step_344400.json).
"""
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lem.model import efference as EF

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = (ROOT / "docs" / "gain_of_function_v3" / "measurements"
           / "efference_lesson_inputs_ckpt_step_344400.npz")
LIVE_LINE = 0.5
SPREAD_LINE = 0.5
STEPS = 100
BATCH = 8

pytestmark = pytest.mark.skipif(not FIXTURE.is_file(), reason="the efference fixture is not on disk")


def _data():
    fx = np.load(FIXTURE)
    t = {k: torch.from_numpy(fx[k]).float() for k in
         ("latent", "order", "ach0", "ach1", "eye0", "eye1", "vision_pred", "vision_target")}
    names = [str(x) for x in fx["handle_names"]]
    active = [str(x) for x in fx["active"]]
    return t, names, active


def _spread(h, x, which):
    pool = h.body_pool if which == "body" else h.gaze_pool
    with torch.no_grad():
        _, w = pool(x)
        p = w.float().clamp_min(1e-12)
        return float((-(p * p.log()).sum(-1) / np.log(p.shape[-1])).mean())


def _read(h, t):
    with torch.no_grad():
        h.body(t["latent"], t["order"], t["ach0"], record=True)
        h.gaze(t["latent"], t["order"], t["ach0"], record=True)
        h.vision_delta(t["latent"], t["order"], t["ach0"], record=True)
        rep = h.window_liveness()
        x = h.normalise(t["latent"])
    rep["spread_direct"] = {"body_pool": _spread(h, x, "body"), "gaze_pool": _spread(h, x, "gaze")}
    return rep


def _train(h, t, gi):
    S = EF.EFFERENCE_SCALES
    N, H = t["order"].shape[0], t["order"].shape[1]
    mask = torch.ones(H, dtype=torch.bool)
    mask[gi] = False
    cm_t, deg_t = EF.pose_change(t["ach0"], t["ach1"])
    t1 = t["eye1"] - t["eye0"]
    ps = list(h.parameters())
    opt = torch.optim.AdamW([{"params": [p for p in ps if p.dim() >= 2], "weight_decay": 0.01},
                             {"params": [p for p in ps if p.dim() < 2], "weight_decay": 0.0}],
                            lr=3e-4, betas=(0.9, 0.999))
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(N, generator=g)
    for s in range(STEPS):
        b = idx[(torch.arange(BATCH) + s * BATCH) % N]
        pb = h.body(t["latent"][b], t["order"][b], t["ach0"][b])
        lb = 0.5 * (EF.scaled_error(pb[..., 0:3], cm_t[b], S["pos_next_cm"], mask)
                    + EF.scaled_error(pb[..., 3:6], deg_t[b], S["rot_next_deg"], mask))
        lg = EF.scaled_error(h.gaze(t["latent"][b], t["order"][b], t["ach0"][b])[:, 0], t1[b],
                             S["gaze_deg_1"])
        lv = F.mse_loss(t["vision_pred"][b] + h.vision_delta(t["latent"][b], t["order"][b],
                                                              t["ach0"][b]),
                        t["vision_target"][b])
        loss = 0.1 * (lb + lg) + lv
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ps, 1.0)
        opt.step()
        h.update_centres(t["latent"][b], t["order"][b])


def _heads(t, names, active):
    torch.manual_seed(0)
    N, n_mod, d = t["latent"].shape
    h = EF.make_heads(d, len(names), names.index("gaze"), names.index("head"), n_mod,
                      active.index("vision"), 0)
    h.ensure_centres(t["latent"], t["order"])
    return h


def test_the_heads_are_live_and_spread_at_the_start_and_after_100_steps():
    """Measured when written (2026-09-23): live share 1.0 in every layer at the start and after;
    pool spread 0.99 and 0.99 at the start, 0.85 and 0.85 after 100 steps."""
    t, names, active = _data()
    h = _heads(t, names, active)
    raw = t["latent"].reshape(t["latent"].shape[0], -1)
    for when in ("start", "after"):
        if when == "after":
            _train(h, t, names.index("gaze"))
        rep = _read(h, t)
        assert set(rep["liveness"]) == set(EF.EfferenceHeads.LIVE_LAYERS), rep
        for layer, share in rep["liveness"].items():
            assert share > LIVE_LINE, (when, layer, share)
        for pool in EF.EfferenceHeads.POOLS:
            assert rep["pool_spread"][pool] > SPREAD_LINE, (when, pool, rep["pool_spread"])
            assert abs(rep["pool_spread"][pool] - rep["spread_direct"][pool]) < 1e-4
        # the centring works on these latents: of the raw latent, the part every moment shares is
        # most of its length (66 of 77 when measured); of the centred one, almost none
        assert rep["input_shared_share"]["latent"] < 0.05, rep["input_shared_share"]
    assert float(raw.mean(0).norm()) / float(raw.norm(dim=-1).mean()) > 0.8


def test_the_control_an_unbounded_pool_locks_on_the_same_moments():
    """The line has teeth: the same heads with the pools' cosine bound lifted to 20 (the control's
    setting; nothing in the model uses it) fall under the spread line within the same 100 steps
    (measured when written: 0.17 and 0.21), while every hidden layer stays live - so a live-share
    test alone would pass while the pools lock, which is how the first version's pools were missed."""
    t, names, active = _data()
    h = _heads(t, names, active)
    h.body_pool.bound = 20.0
    h.gaze_pool.bound = 20.0
    _train(h, t, names.index("gaze"))
    rep = _read(h, t)
    assert all(v > LIVE_LINE for v in rep["liveness"].values()), rep["liveness"]
    assert all(rep["pool_spread"][p] < SPREAD_LINE for p in EF.EfferenceHeads.POOLS), rep["pool_spread"]


def test_the_bound_keeps_every_head_at_the_spread_line_at_worst():
    """logit_bound: the most concentrated weighting a head can reach over six tokens (one token at
    the largest cosine, five at the smallest) keeps exactly the floor's share of the entropy."""
    t = EF.logit_bound(6)
    assert abs(t - 1.4225) < 1e-3
    p = torch.softmax(torch.tensor([t] + [-t] * 5, dtype=torch.float64), 0)
    share = float(-(p * p.log()).sum() / np.log(6))
    assert abs(share - EF.POOL_SPREAD_FLOOR) < 1e-6 and float(p[0]) < 0.78
