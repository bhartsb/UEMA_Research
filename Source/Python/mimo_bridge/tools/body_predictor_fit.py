# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Historical additive linear physical predictor, retained for reproducibility.

This feature map is NOT a state-conditioned action model: current pose only
enters as a displacement subtraction, and shared state terms cancel between
actions. See body_state_prediction.py for explicit state/action interactions.

The reviewer's 2026-09-10 instruction, executed under the declaration written
BEFORE this fit (Saved/mimo_bridge/body_predictor_study_2026-09-10/
declaration.md): a small ridge readout predicts each candidate's 50 ms body
POSITION change in centimetres from per-handle physical features; the claim is
judged on within-state ACTION EFFECTS (pairwise differences between arms), so
anything the arms share cancels and only command-caused difference counts.

Features per arm, per handle (physical units, no tokens anywhere):
  executed displacement request   command position - achieved position (cm)
  executed rotation               quaternion (4)
  radius, strength                as executed
  current velocity                proprioception velocity block
Targets: end achieved position - start achieved position, cm (24 handles x 3).

Gate (declared): evaluation-world effect-error ratio < 1.0 (1.0 = predicting
no difference between actions) in EVERY evaluation world AND below the 5th
percentile of eight shuffled-command fits per world. Training worlds pick the
ridge penalty by leave-one-world-out; evaluation worlds choose nothing.

python -m mimo_bridge.tools.body_predictor_fit --data <study_dir>
    [--train-worlds 6] [--pos-scale 200]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _features(row, arm, pos_scale):
    ach = row["physical_start"]["achieved"].float()          # (24, 10)
    pro = row["physical_start"]["proprioception"].float()    # (24, 12)
    cmd = row["commands"][arm].float()                       # (24, 9)
    cur_cm = ach[:, 0:3] * pos_scale
    disp = cmd[:, 0:3] - cur_cm                              # executed request, cm
    vel = pro[:, 7:10]                                       # normalized velocity, NOT cm/s
    f = torch.cat([disp, cmd[:, 3:7], cmd[:, 7:9], vel], dim=-1)   # (24, 12)
    return f.reshape(-1)                                     # (288,)


def _target(row, arm, pos_scale):
    end = row["physical_end"][arm]["achieved"].float()[:, 0:3]
    start = row["physical_start"]["achieved"].float()[:, 0:3]
    return ((end - start) * pos_scale).reshape(-1)           # (72,) cm


def _pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _fit_ridge(X, Y, lam):
    # Mixed centimetre/unit-quaternion features produce ill-conditioned normal
    # equations. Keep this historical feature map, but solve in float64.
    X, Y = X.double(), Y.double()
    Xb = torch.cat([X, torch.ones(X.shape[0], 1)], dim=-1)
    A = Xb.T @ Xb + lam * torch.eye(Xb.shape[1], dtype=X.dtype)
    return torch.linalg.solve(A, Xb.T @ Y)


def _predict(W, X):
    X = X.to(W)
    return torch.cat([X, torch.ones(X.shape[0], 1)], dim=-1) @ W


def _effect_ratio(rows, W, pos_scale):
    """Sum over states of ||pred_effect - true_effect||^2 / ||true_effect||^2."""
    if not rows or not torch.isfinite(W).all():
        raise ValueError("Missing outcomes or non-finite fitted weights")
    num = den = 0.0
    for row in rows:
        n = row["commands"].shape[0]
        feats = torch.stack([_features(row, a, pos_scale) for a in range(n)])
        tgts = torch.stack([_target(row, a, pos_scale) for a in range(n)])
        preds = _predict(W, feats)
        for i, j in _pairs(n):
            te = tgts[i] - tgts[j]
            pe = preds[i] - preds[j]
            num += float(((pe - te) ** 2).sum())
            den += float((te ** 2).sum())
    if not np.isfinite([num, den]).all() or den <= 0:
        raise ValueError("Non-finite or absent physical action effects")
    return num / den


def _dataset(rows, pos_scale, cmd_perm=None, rng=None):
    X, Y = [], []
    for row in rows:
        n = row["commands"].shape[0]
        order = list(range(n))
        if cmd_perm is not None:
            order = list(rng.permutation(n))
        for a in range(n):
            feats_row = dict(row)
            # shuffled control: commands reassigned across arms WITHIN the
            # state; outcomes keep their true arm - the command no longer
            # explains its own outcome
            src = order[a] if cmd_perm is not None else a
            f = _features(row, src, pos_scale)
            X.append(f)
            Y.append(_target(row, a, pos_scale))
    return torch.stack(X), torch.stack(Y)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--train-worlds", type=int, default=6)
    ap.add_argument("--pos-scale", type=float, default=200.0,
                    help="normalized achieved position -> cm (the command "
                         "contract's x200)")
    args = ap.parse_args(argv)
    rows = torch.load(args.data / "paired_frames.pt", weights_only=False)
    worlds = sorted({r["world"] for r in rows})
    train_w = [w for w in worlds if w < args.train_worlds]
    eval_w = [w for w in worlds if w >= args.train_worlds]
    if len(train_w) < 2 or not eval_w:
        raise ValueError("Need multiple training worlds and nonempty evaluation worlds")
    if len({(r['world'], r['event']) for r in rows}) != len(rows):
        raise ValueError("Duplicate physical decision states")
    by_w = {w: [r for r in rows if r["world"] == w] for w in worlds}
    ps = args.pos_scale

    # ridge penalty by leave-one-training-world-out; evaluation chooses nothing
    cv = {}
    for lam in LAMBDAS:
        errs = []
        for held in train_w:
            tr = [r for w in train_w if w != held for r in by_w[w]]
            X, Y = _dataset(tr, ps)
            W = _fit_ridge(X, Y, lam)
            errs.append(_effect_ratio(by_w[held], W, ps))
        cv[lam] = float(np.mean(errs))
    lam = min(cv, key=cv.get)

    tr = [r for w in train_w for r in by_w[w]]
    X, Y = _dataset(tr, ps)
    W = _fit_ridge(X, Y, lam)
    per_world = {w: _effect_ratio(by_w[w], W, ps) for w in eval_w}

    # Eight shuffled fits form an empirical comparison, NOT a calibrated 5%
    # significance test (the minimum exact permutation p-value would be 1/9).
    shuffled = {w: [] for w in eval_w}
    for seed in range(8):
        rng = np.random.default_rng(900 + seed)
        Xs, Ys = _dataset(tr, ps, cmd_perm=True, rng=rng)
        Ws = _fit_ridge(Xs, Ys, lam)
        for w in eval_w:
            shuffled[w].append(_effect_ratio(by_w[w], Ws, ps))
    sh_p5 = {w: float(np.percentile(shuffled[w], 5)) for w in eval_w}

    gate = all(per_world[w] < 1.0 and per_world[w] < sh_p5[w] for w in eval_w)
    out = {
        "declaration": str(args.data / "declaration.md") if (args.data / "declaration.md").is_file() else None,
        "historical_limitations": "Additive state terms cannot change action effects; raw physical replay "
                                  "was not retained in the original datasets. A fit failure is not "
                                  "proof that the physics or the live brain cannot learn action effects.",
        "checkpoint_data": str(args.data),
        "data_sha256": hashlib.sha256(
            open(args.data / "paired_frames.pt", "rb").read()).hexdigest(),
        "source_sha256": hashlib.sha256(
            open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "train_worlds": train_w, "eval_worlds": eval_w,
        "ridge_lambda": lam, "cv_curve": {str(k): v for k, v in cv.items()},
        "eval_effect_ratio_per_world": {str(w): per_world[w] for w in eval_w},
        "shuffled_ratios_per_world": {str(w): shuffled[w] for w in eval_w},
        "shuffled_p5_per_world": {str(w): sh_p5[w] for w in eval_w},
        "gate": "PASS" if gate else "FAIL",
        "gate_rule": "ratio < 1.0 AND < shuffled p5 in EVERY evaluation world",
        "units": "centimetres; ratio 1.0 = predicting no difference between "
                 "actions, 0 = exact",
    }
    p = args.data / "body_predictor_result.json"
    json.dump(out, open(p, "w"), indent=1)
    torch.save({"W": W, "lambda": lam,
                "feature_doc": "per handle: disp_cm(3) quat(4) rad+str(2) vel(3)",
                "data_sha256": out["data_sha256"]},
               args.data / "body_predictor_weights.pt")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("shuffled_ratios_per_world", "cv_curve")},
                     indent=1))
    print("written:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
