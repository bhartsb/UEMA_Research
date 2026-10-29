# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Movement-sequence predictor (2026-09-10 declaration in docs/, committed
before collection): forecast the FIVE-FRAME sequence of raw body-position
changes during one held order from prior executed commands and observations
available by the decision, the current raw state, and the candidate
executed command.

Built ON the reviewer's verified machinery (body_state_prediction): same
kernel-ridge families, same leak-safe conventions, same world-split selection
and scoring. The sequence model is the state_action_kernel family with the
state vector ENRICHED by the history and touch features their review named
missing, and with per-frame path targets as supervision. The comparator is
their exact linear family refit on the same data (same budget - a stricter
comparison than their saved weights on foreign features).

Gates (declaration section 'Declared gates'): in EVERY evaluation world the
sequence model must beat the linear comparator on action differences, beat
"no movement" and "constant velocity" on absolute outcomes, and beat eight
shuffled-command fits (an empirical comparison, not a calibrated 5% test).
A learning curve describes this recipe's response to more training examples.
It cannot establish that sample count was the sole cause of earlier failures.

python -m mimo_bridge.tools.body_sequence_fit --data <study_dir> --out <new_dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mimo_bridge.tools.body_state_prediction import (  # noqa: E402
    tensor, effects, score, fit, predict, subset)
from tokenizer.modalities.touch import _FIELD_NORMS  # noqa: E402

PATH_FRAMES = 5


def extract_sequence(rows, horizon=0.25):
    """Their extract(), enriched: history + touch into the state; path targets.

    Every input field is measured at or before the decision. The last
    historical command ends at the current observation; it is not a future
    candidate command. New timestamped rows are checked below;
    targets come only from physical_paths."""
    states, commands, targets, velocities, floors, resting = [], [], [], [], [], []
    norms = torch.as_tensor(np.asarray(_FIELD_NORMS[:8]), dtype=torch.float64)
    for row in rows:
        bones = tensor(row["physical_start"]["bones"])
        pos, quat, vel = bones[:, 4:7], bones[:, :4], bones[:, 7:10]
        quat = quat * torch.where(quat[:, 3:] < 0, -1.0, 1.0)
        state = [torch.cat([quat, (pos - pos.mean(0)) / 200.0, vel / 200.0,
                            bones[:, 10:11]], -1).flatten()]
        # touch NOW: tokenizer-normalized fields, zero-padded layout
        state.append((tensor(row["physical_start"]["patches"])[:, :8]
                      / norms).flatten())
        # strictly-past history: per step, pose delta, velocity, executed
        # command relative position + strength
        h = row["history"]
        if h is None or len(h) != PATH_FRAMES:
            raise ValueError("history missing: collect with --history-steps 5")
        prev = None
        for e in h:
            if 'outcome_time' in e:
                start, end, now = e['command_start_time'],e['outcome_time'],row['sim_time']
                if not (np.isfinite([start,end,now]).all() and start < now
                        and end <= now+1e-7 and np.isclose(end-start,.05,atol=1e-7)):
                    raise ValueError('History includes a future or mis-timed observation')
            a = tensor(e["achieved"])
            hp = a[:, 0:3]                       # normalized pose
            hv = tensor(e["proprioception"])[:, 7:10]
            cmd = tensor(e["command"]).squeeze(0)
            rel = cmd[:, 0:3] / 200.0 - hp
            step_feat = [hp - (prev if prev is not None else hp), hv,
                         rel, cmd[:, 8:9]]
            prev = hp
            state.append(torch.cat(step_feat, -1).flatten())
        states.append(torch.cat(state))
        cmds = tensor(row["commands"]).clone()
        cmds[..., :3] = (cmds[..., :3] - pos[None]) / 200.0
        cmds[..., 3:7] *= torch.where(cmds[..., 6:7] < 0, -1.0, 1.0)
        commands.append(cmds.flatten(1))
        idx = [round((horizon / PATH_FRAMES) * (k + 1) / 0.05) - 1
               for k in range(PATH_FRAMES)]
        path = torch.stack([torch.cat([
            tensor(row["physical_paths"][a][i]["bones"])[:, 4:7] - pos
            for i in idx], -1).flatten() for a in range(4)])
        replay = torch.cat([tensor(row["physical_paths"][4][i]["bones"])[:, 4:7]
                            - pos for i in idx], -1).flatten()
        first = torch.cat([tensor(row["physical_paths"][0][i]["bones"])[:, 4:7]
                           - pos for i in idx], -1).flatten()
        floors.append(float((first - replay).square().mean()))
        targets.append(path)
        velocities.append(vel.flatten())
        resting.append(bool(row.get("resting_start")))
    data = dict(s=torch.stack(states), c=torch.stack(commands),
                y=torch.stack(targets), v=torch.stack(velocities),
                floor=np.array(floors),
                worlds=np.array([r["world"] for r in rows]),
                resting=np.array(resting), horizon=horizon)
    for k in ("s", "c", "y", "v"):
        if not torch.isfinite(data[k]).all():
            raise ValueError(f"non-finite {k}")
    return data


def absolute_scores(pred, data):
    """Absolute-outcome error vs 'no movement' and 'constant velocity',
    on the FINAL path frame (their review's absolute baselines)."""
    y_final = final_positions(data["y"])
    p_final = final_positions(pred)
    err = float((p_final - y_final).square().mean())
    still = float(y_final.square().mean())
    dt = data["horizon"]
    # Both tensors now refer to the SAME 24 body points at the final time.
    # The previous flat [-72:] slice mixed time/body indices and invalidated
    # that physical interpretation, including the constant-velocity comparison.
    cv = (data["v"] * dt).unsqueeze(1).expand_as(y_final)
    cvel = float((cv - y_final).square().mean())
    return {"model": err, "no_movement": still, "constant_velocity": cvel}


def final_positions(flat):
    """Extract time 4 from [body point, time, xyz] flattened path targets."""
    if flat.shape[-1] != 24 * PATH_FRAMES * 3:
        raise ValueError("Expected 24 body points by 5 times by xyz")
    if not torch.isfinite(flat).all():
        raise ValueError("Non-finite body path")
    return flat.reshape(*flat.shape[:-1],24,PATH_FRAMES,3)[..., -1, :].flatten(-2)


def prediction_gate(per_world, absolutes, shuffled):
    """Reject missing evidence and unresolved effects before any PASS."""
    if not per_world or set(per_world) != set(absolutes) or set(per_world) != set(shuffled):
        raise ValueError("Missing evaluation worlds")
    passed = []
    for world, values in per_world.items():
        absolute, null = absolutes[world], shuffled[world]
        required = [values['sequence'],values['linear_comparator'],
                    values['effect_mse_cm2'],values['replay_floor_max'],
                    absolute['model'],absolute['no_movement'],absolute['constant_velocity'],*null]
        if len(null) != 8 or not np.isfinite(required).all() or min(required) < 0:
            raise ValueError("Invalid prediction evidence")
        if values['effect_mse_cm2'] <= max(0.,10*values['replay_floor_max']):
            raise ValueError("Action differences unresolved against raw replay")
        passed.append(values['sequence'] < min(1.,values['linear_comparator'],*null)
                      and absolute['model'] < min(absolute['no_movement'],absolute['constant_velocity']))
    return bool(all(passed))


def world_select_fit(data, train_w, family, sizes=None):
    """Leave-one-training-world-out over this study's lambda/width grid."""
    lams = (1e-4, 1e-3, 1e-2, 1e-1)
    widths = (0.5, 1.0, 2.0) if family == "state_action_kernel" else (1.0,)
    best, best_err = None, None
    for lam in lams:
        for width in widths:
            errs = []
            for held in train_w:
                tr_i = np.where(np.isin(data["worlds"], [w for w in train_w
                                                         if w != held]))[0]
                ev_i = np.where(data["worlds"] == held)[0]
                if sizes is not None:
                    tr_i = tr_i[:sizes]
                m = fit(subset(data, tr_i), family, lam, width)
                errs.append(score(predict(m, subset(data, ev_i)),
                                  subset(data, ev_i)["y"]))
            e = float(np.mean(errs))
            if best_err is None or e < best_err:
                best, best_err = (lam, width), e
    tr_i = np.where(np.isin(data["worlds"], train_w))[0]
    if sizes is not None:
        tr_i = tr_i[:sizes]
    return fit(subset(data, tr_i), family, *best), best, best_err


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="New result directory; previous fits are never overwritten")
    ap.add_argument("--train-worlds", type=int, default=6)
    ap.add_argument("--extra-train-data", type=Path, nargs="*", default=[],
                    help="additional collections whose EVERY world joins "
                         "training (world ids offset by 100 per directory); "
                         "evaluation outcomes must not guide these additions; "
                         "otherwise reserve fresh confirmation worlds")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True,exist_ok=False)
    rows = torch.load(args.data / "paired_frames.pt", weights_only=False)
    for k, extra in enumerate(args.extra_train_data):
        for r in torch.load(extra / "paired_frames.pt", weights_only=False):
            r = dict(r)
            r["world"] = 100 * (k + 1) + r["world"]
            rows.append(r)
    data = extract_sequence(rows)
    worlds = sorted(set(data["worlds"].tolist()))
    train_w = [w for w in worlds if w < args.train_worlds or w >= 100]
    eval_w = [w for w in worlds if args.train_worlds <= w < 100]
    if len(train_w) < 3 or not eval_w:
        raise ValueError("Need training worlds and nonempty evaluation worlds")
    tr_i = np.where(np.isin(data["worlds"], train_w))[0]

    seq_model, seq_hp, seq_cv = world_select_fit(data, train_w,
                                                 "state_action_kernel")
    lin_model, lin_hp, lin_cv = world_select_fit(data, train_w, "linear")

    per_world, absolutes = {}, {}
    for w in eval_w:
        ev = subset(data, np.where(data["worlds"] == w)[0])
        ps, pl = predict(seq_model, ev), predict(lin_model, ev)
        per_world[w] = {"sequence": score(ps, ev["y"]),
                        "linear_comparator": score(pl, ev["y"]),
                        "replay_floor_mean": float(np.mean(ev["floor"])),
                        "replay_floor_max": float(np.max(ev["floor"])),
                        "effect_mse_cm2": float(effects(ev['y']).square().mean())}
        absolutes[w] = absolute_scores(ps, ev)
    shuffled = {w: [] for w in eval_w}
    for seed in range(8):
        m = fit(subset(data, tr_i), "state_action_kernel",
                seq_hp[0], seq_hp[1], shuffle_seed=700 + seed)
        for w in eval_w:
            ev = subset(data, np.where(data["worlds"] == w)[0])
            shuffled[w].append(score(predict(m, ev), ev["y"]))

    curve = {}
    for n in sorted({24, 48, 96, 192, len(tr_i)}):
        if n > len(tr_i):
            continue
        m, _, cv_err = world_select_fit(data, train_w,
                                        "state_action_kernel", sizes=n)
        curve[n] = {"train_cv": cv_err,
                    "eval": {w: score(predict(m, subset(
                        data, np.where(data["worlds"] == w)[0])),
                        subset(data, np.where(data["worlds"] == w)[0])["y"])
                        for w in eval_w}}

    gate = prediction_gate(per_world,absolutes,shuffled)
    out = {
        "declaration": "docs/body_sequence_study_2026-09-10/declaration.md (committed before collection)",
        "data_sha256": hashlib.sha256(
            open(args.data / "paired_frames.pt", "rb").read()).hexdigest(),
        "source_sha256": hashlib.sha256(
            open(__file__, "rb").read()).hexdigest(),
        "extra_training_sources": [{"path":str(p), "sha256":hashlib.sha256(
            (p/'paired_frames.pt').read_bytes()).hexdigest()} for p in args.extra_train_data],
        "train_worlds": train_w, "eval_worlds": eval_w,
        "resting_start_fraction": float(np.mean(data["resting"])),
        "sequence_hp": {"lam": seq_hp[0], "width": seq_hp[1], "cv": seq_cv},
        "linear_hp": {"lam": lin_hp[0], "cv": lin_cv},
        "per_world": {str(w): per_world[w] for w in eval_w},
        "absolute_final_frame": {str(w): absolutes[w] for w in eval_w},
        "shuffled_min_per_world": {str(w): float(min(shuffled[w]))
                                   for w in eval_w},
        "learning_curve": {str(k): v for k, v in curve.items()},
        "gate": "PASS" if gate else "FAIL",
        "gate_rule": "every evaluation world: sequence < linear comparator "
                     "(action effects) AND < no-movement AND < constant-velocity "
                     "(absolute, final frame) AND < all 8 shuffled fits",
        "scale": "action-effect score: 1 = predicting no difference between "
                 "actions, 0 = exact; absolute errors are cm squared per coordinate",
    }
    p = args.out / "sequence_result.json"
    json.dump(out, open(p, "w"), indent=1, default=float,allow_nan=False)
    torch.save({"sequence": seq_model, "linear": lin_model,
                "data_sha256": out["data_sha256"]},
               args.out / "sequence_models.pt")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("learning_curve",)}, indent=1, default=float))
    print("written:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
