# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Wellness-outcome predictor: steps 1-2 of the reviewer's choice-study design
(docs/sequence_confirmation_2026-09-11/next_wellness_test.md; declaration
committed in docs/wellness_choice_study_2026-09-11/declaration.md).

Targets per candidate arm, all grounded formulas (never tokens): the per-frame
tactile-comfort and posture sequence over the held order plus END capacity -
an uncomfortable movement cannot hide behind a comfortable ending. Historical
rows reconstruct intermediate tactile/posture from the retained per-frame
patches/bones through the SAME state_wellness preparation, and every row must
show exact parity between its reconstructed final frame and the stored end
wellness or it is rejected (never fabricated; intermediate capacity does not
exist historically and is not invented).

Two equal-budget alternatives, settings chosen by world-CV on the 24 training
worlds only, then FROZEN to disk:
  A  sensing + prior commands + candidate command (extract_sequence features)
  B  A + the body predictor's forecast path, produced for every training row
     by a body fit that LEFT THAT ROW'S WORLD OUT (their step 2: in-sample
     forecasts would be unrealistically easy; the recorded actual path is
     never substituted)

No gate here: selection only. The gate lives in the confirmation stage on
fresh worlds with recorded-before-reveal choices.

python -m mimo_bridge.tools.wellness_outcome_fit --primary <dir> --extra <dir> <dir>
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

from mimo_bridge.tools.body_state_prediction import fit, predict, subset  # noqa: E402
from mimo_bridge.tools.body_sequence_fit import extract_sequence, PATH_FRAMES  # noqa: E402
from mimo_bridge.tools.first_action_benefit import state_wellness  # noqa: E402
from lem.model.wellness import WellnessEvaluators, WellnessMode, MotorStateFormula  # noqa: E402


class _Meter:
    """state_wellness needs only .wellness with the grounded formulas."""

    def __init__(self):
        cfg = {"_active_modalities": ["vision", "tactile", "motor"],
               "model": {"d_model": 8,
                         "wellness": {"grounded": {"motor_state": {"input_dim": 216}},
                                      "approximate": {"_input_dim": 24,
                                                      "hidden_dims": [8],
                                                      "n_outputs": 5}}}}
        self.wellness = WellnessEvaluators(cfg, WellnessMode.DIRECT,
                                           MotorStateFormula.BOUNDARY)


class _Frame:
    def __init__(self, entry):
        self.patches = entry["patches"].numpy()
        self.bones = entry["bones"].numpy()
        self.tokenized = type("T", (), {"motor_capability": None})()


def component_targets(rows, meter):
    """[N, A, 2*PATH_FRAMES+1]: per-frame (tactile, posture) + end capacity.
    Parity rule: reconstructed final tactile/posture must equal the stored end
    wellness to 1e-6 on every arm, else the row is rejected (counted)."""
    out, rejected = [], 0
    for row in rows:
        arms = []
        ok = True
        for a in range(4):
            seq = []
            for entry in row["physical_paths"][a]:
                w = (entry.get("wellness")
                     or state_wellness(meter, _Frame(entry)))
                seq.append((w["tactile"], w["posture"]))
            endw = row["wellness"][a]
            if (abs(seq[-1][0] - endw["tactile"]) > 1e-6
                    or abs(seq[-1][1] - endw["posture"]) > 1e-6):
                ok = False
                break
            arms.append([v for pair in seq for v in pair] + [endw["capability"]])
        if ok:
            out.append(arms)
        else:
            rejected += 1
            out.append(None)
    return out, rejected


def leave_world_out_paths(data, worlds_of, hp):
    """Body-forecast features for every row, from a body fit trained WITHOUT
    that row's world (their step 2)."""
    paths = torch.zeros(data["y"].shape, dtype=torch.float64)
    for w in sorted(set(worlds_of.tolist())):
        tr_i = np.where(worlds_of != w)[0]
        ev_i = np.where(worlds_of == w)[0]
        m = fit(subset(data, tr_i), "state_action_kernel", hp["lam"], hp["width"])
        paths[ev_i] = predict(m, subset(data, ev_i))
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--extra", type=Path, nargs="*", default=[])
    ap.add_argument("--train-worlds", type=int, default=6,
                    help="primary-data worlds below this id train; its "
                         "evaluation worlds are NOT touched here at all")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rows = [r for r in torch.load(args.primary / "paired_frames.pt",
                                  weights_only=False)
            if r["world"] < args.train_worlds]
    for k, extra in enumerate(args.extra):
        for r in torch.load(extra / "paired_frames.pt", weights_only=False):
            r = dict(r)
            r["world"] = 100 * (k + 1) + r["world"]
            rows.append(r)
    meter = _Meter()
    targets, rejected = component_targets(rows, meter)
    keep = [i for i, t in enumerate(targets) if t is not None]
    rows = [rows[i] for i in keep]
    y_w = torch.tensor([targets[i] for i in keep], dtype=torch.float64)
    data = extract_sequence(rows)
    worlds = data["worlds"]

    # body-fit hyperparameters: the frozen sequence study's selection
    seq_hp = {"lam": 0.01, "width": 2.0}
    body_paths = leave_world_out_paths(data, worlds, seq_hp)

    def eval_alternative(with_paths):
        d = dict(data)
        if with_paths:
            # the arm's OWN forecast rides with the arm's command features -
            # in the shared state it would leak every arm's forecast to every
            # candidate and lose the per-candidate meaning
            d = dict(d, c=torch.cat([data["c"], body_paths / 10.0], -1))
        d["y"] = y_w                      # [N, arms, outputs] - fit() flattens
        best, best_err = None, None
        for lam in (1e-3, 1e-2, 1e-1):
            for width in (0.5, 1.0, 2.0):
                errs = []
                for held in sorted(set(worlds.tolist())):
                    tr_i = np.where(worlds != held)[0]
                    ev_i = np.where(worlds == held)[0]
                    m = fit(subset(d, tr_i), "state_action_kernel", lam, width)
                    pred = predict(m, subset(d, ev_i))
                    true = subset(d, ev_i)["y"]
                    # component action-effect error, their scale (1 = no
                    # difference between actions)
                    from mimo_bridge.tools.body_state_prediction import effects
                    num = float((effects(pred) - effects(true)).square().sum())
                    den = float(effects(true).square().sum())
                    errs.append(num / max(den, 1e-12))
                e = float(np.mean(errs))
                if best_err is None or e < best_err:
                    best, best_err = (lam, width), e
        tr_all = np.arange(len(rows))
        return fit(subset(d, tr_all), "state_action_kernel", *best), best, best_err

    model_a, hp_a, cv_a = eval_alternative(False)
    model_b, hp_b, cv_b = eval_alternative(True)

    # the FROZEN full-training body fit that confirmation will use for B
    body_full = fit(data, "state_action_kernel", seq_hp["lam"], seq_hp["width"])

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "declaration": "docs/wellness_choice_study_2026-09-11/declaration.md",
        "rows_used": len(rows), "rows_rejected_parity": rejected,
        "worlds": sorted(set(worlds.tolist())),
        "cv_effect_error": {"A_no_paths": cv_a, "B_with_paths": cv_b},
        "hp": {"A": hp_a, "B": hp_b, "body": seq_hp},
        "target_doc": "per-frame (tactile, posture) x5 + end capacity, 0-1 "
                      "component scales; action-effect CV score 1 = no "
                      "difference between actions",
        "frozen": True,
        "source_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
    }
    torch.save({"A": model_a, "B": model_b, "body_full": body_full,
                "manifest": manifest}, args.out / "frozen_selectors.pt")
    json.dump(manifest, open(args.out / "manifest.json", "w"), indent=1,
              default=float)
    print(json.dumps(manifest, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
