# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Locate usable action information with small readouts and training-world CV.

All readouts learn measured within-state outcome differences. They receive either
the command, its motor token, or the existing forward transformer's output. Their
penalty is selected using only training worlds. No production weights are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from mimo_bridge.tools.action_effect_learning import MODS, model, predict, paired_differences, write_json
from lem.model.encoder_registry import normalize_executed_command


def center(x):
    return x - x.mean(1, keepdim=True)


def fit_readout(x, y, penalty):
    xc, yc = center(x).flatten(0, 1).double(), center(y).flatten(0, 1).double()
    scale = xc.square().mean(0).sqrt().clamp_min(1e-7)
    xn = xc / scale
    # Dual form keeps even the transformer readout to a training-sample-sized solve.
    gram = xn @ xn.T
    weight = xn.T @ torch.linalg.solve(gram + len(xn) * penalty * torch.eye(len(xn)), yc)
    return weight / scale[:, None]


def scores(pred, y):
    p, t = paired_differences(pred), paired_differences(y)
    return float((p - t).square().mean() / t.square().mean())


def run(ckpt, folder):
    dec = json.loads((folder / "declaration.json").read_text())
    if hashlib.sha256((ckpt / "model.safetensors").read_bytes()).hexdigest() != dec["checkpoint_sha256"]:
        raise ValueError("readout checkpoint differs from collection")
    rows = torch.load(folder / "paired_frames.pt", weights_only=False)
    worlds = torch.tensor([r["world"] for r in rows])
    train_ids = dec["train_worlds"]
    test_ids = dec["heldout_worlds"]
    if set(train_ids) & set(test_ids):
        raise ValueError("training worlds overlap held-out worlds")
    m = model(ckpt)
    features = {"command": [], "motor_token": [], "forward_output": []}
    captured = {}
    fm_hook = m.forward_model.register_forward_hook(lambda module, args, out: captured.update(fm=out[1].detach()))
    motor_hook = m.encoder_registry.encoders["motor"].register_forward_hook(
        lambda module, args, out: captured.update(motor=out.detach()))
    try:
        with torch.no_grad():
            for offset in range(0, len(rows), 2):
                batch = rows[offset:offset+2]
                predict(m, batch)
                n, actions = len(batch), batch[0]["commands"].shape[0]
                features["forward_output"].append(captured["fm"].reshape(n, actions, -1))
                features["motor_token"].append(m.encoder_registry.output_norm(captured["motor"]).reshape(n, actions, -1))
                features["command"].append(normalize_executed_command(
                    torch.stack([r["commands"] for r in batch])).reshape(n, actions, -1))
    finally:
        fm_hook.remove()
        motor_hook.remove()
    features = {k: torch.cat(v).double() for k, v in features.items()}
    y = {k: torch.stack([r["target"][k] for r in rows]).double() for k in MODS}
    train = torch.tensor([int(w) in train_ids for w in worlds])
    penalties = (.01, .1, 1., 10., 100.)
    results = {"scale": "1 = predicts no action differences; 0 = predicts them exactly",
        "penalty_selection": "Leave-one-training-world-out; held-out worlds never select penalty",
        "limits": "Small linear readouts of fixed representations; these are not a deployed planner.",
        "features": {}}
    learned = {"checkpoint_sha256": dec["checkpoint_sha256"], "train_worlds": train_ids,
               "meaning": "Difference between two feature vectors times weight predicts their outcome difference.",
               "weights": {}}
    for feature, x in features.items():
        entry = {}
        for sense in MODS:
            cv = {}
            for penalty in penalties:
                fold_scores = []
                for v in train_ids:
                    fit = train & (worlds != v)
                    val = worlds == v
                    weight = fit_readout(x[fit], y[sense][fit], penalty)
                    fold_scores.append(scores(x[val] @ weight, y[sense][val]))
                cv[penalty] = sum(fold_scores) / len(fold_scores)
            penalty = min(cv, key=cv.get)
            weight = fit_readout(x[train], y[sense][train], penalty)
            learned["weights"][feature + "/" + sense] = weight.detach().clone()
            entry[sense] = {"penalty": penalty, "training_world_cv": cv,
                "heldout": {int(w): scores(x[worlds == w] @ weight, y[sense][worlds == w]) for w in test_ids},
                "train": {int(w): scores(x[worlds == w] @ weight, y[sense][worlds == w]) for w in train_ids}}
            if feature == "command" and sense == "proprioception":
                nulls = []
                rng = torch.Generator().manual_seed(4173)
                for replicate in range(8):
                    shuffled = y[sense].clone()
                    for i in torch.where(train)[0]:
                        shuffled[i] = shuffled[i, torch.randperm(shuffled.shape[1], generator=rng)]
                    null_cv = {}
                    for alpha in penalties:
                        per_world = []
                        for v in train_ids:
                            fit, val = train & (worlds != v), worlds == v
                            nw = fit_readout(x[fit], shuffled[fit], alpha)
                            per_world.append(scores(x[val] @ nw, shuffled[val]))
                        null_cv[alpha] = sum(per_world) / len(per_world)
                    nw = fit_readout(x[train], shuffled[train], min(null_cv, key=null_cv.get))
                    nulls.append({int(w): scores(x[worlds == w] @ nw, y[sense][worlds == w]) for w in test_ids})
                entry[sense]["shuffled_training_action_controls"] = nulls
        results["features"][feature] = entry
    write_json(folder / "readout_results.json", results)
    torch.save(learned, folder / "readout_weights.pt")
    print(json.dumps(results, indent=2, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    run(a.ckpt, a.out)


if __name__ == "__main__":
    main()
