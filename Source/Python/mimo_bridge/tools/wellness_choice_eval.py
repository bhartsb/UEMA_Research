# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Wellness-choice confirmation (steps 3-5 of the reviewer's design; frozen
selectors only, fresh worlds only).

Each selector ranks the same four already-executed-form candidates from the
FROZEN models; every choice is computed from decision-time inputs and written
to disk BEFORE any outcome is read into the scoring pass. Scores are the
REALIZED grounded components (comfort + capacity + posture, 0-3 total) and
the worst intermediate comfort of the chosen branch. Comparators: the
planner's own choice (arm 0), the pose-hold candidate (arm 1), uniform
random, selector A (no body-path features), selector B (with them). The best
available candidate is reported as the ceiling only, never as a controller.

Verdict per the committed declaration: world-level paired benefit over BOTH
the planner's choice and hold, 95% CI above zero across worlds, benefit above
replay disagreement, worst intermediate comfort not measurably worse, and the
body-path claim only if B beats A. Worlds whose candidates do not differ
beyond replay error are reported as unable to answer.

python -m mimo_bridge.tools.wellness_choice_eval --data <confirmation_dir>
    --selectors <frozen_dir>
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

from mimo_bridge.tools.body_state_prediction import predict, subset  # noqa: E402
from mimo_bridge.tools.body_sequence_fit import extract_sequence, PATH_FRAMES  # noqa: E402

TAC_END, POST_END, CAP_END = 2 * (PATH_FRAMES - 1), 2 * (PATH_FRAMES - 1) + 1, 2 * PATH_FRAMES


def predicted_wellness(pred):
    """pred [..., 2*F+1] -> (total end wellness, predicted worst comfort)."""
    tac_seq = pred[..., 0:2 * PATH_FRAMES:2]
    total = pred[..., TAC_END] + pred[..., POST_END] + pred[..., CAP_END]
    return total, tac_seq.min(-1).values


def realized(row):
    tot = [w["tactile"] + w["capability"] + w["posture"] for w in row["wellness"]]
    worst = [min(e["wellness"]["tactile"] for e in row["physical_paths"][a])
             for a in range(4)]
    rep = row["wellness"][0]["tactile"] + row["wellness"][0]["capability"] \
        + row["wellness"][0]["posture"]
    rep_arm = row["physical_paths"][4][-1]["wellness"]
    rep_tot = rep_arm["tactile"] + rep_arm["capability"] + rep_arm["posture"]
    return tot, worst, abs(rep - rep_tot)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--selectors", type=Path, required=True)
    args = ap.parse_args(argv)
    rows = torch.load(args.data / "paired_frames.pt", weights_only=False)
    frozen = torch.load(args.selectors / "frozen_selectors.pt",
                        weights_only=False)
    data = extract_sequence(rows)
    body_paths = predict(frozen["body_full"], data)
    data_b = dict(data, c=torch.cat([data["c"], body_paths / 10.0], -1))

    # CHOICES FIRST, from inputs only, written before scoring
    choices = {}
    for name, model, d in (("A", frozen["A"], data),
                           ("B", frozen["B"], data_b)):
        pred = predict(model, d)                       # [N, arms, outputs]
        total, worst = predicted_wellness(pred)
        pick = total.argmax(-1)
        choices[name] = {"pick": pick.tolist(),
                         "predicted_total": total.tolist(),
                         "predicted_worst_comfort": worst.tolist()}
    stamp = {"selectors_sha256": hashlib.sha256(
        open(args.selectors / "frozen_selectors.pt", "rb").read()).hexdigest(),
        "data_sha256": hashlib.sha256(
            open(args.data / "paired_frames.pt", "rb").read()).hexdigest(),
        "choices": choices}
    json.dump(stamp, open(args.data / "choices_before_outcomes.json", "w"),
              indent=1)

    worlds = sorted({r["world"] for r in rows})
    per_world, unable = {}, []
    for w in worlds:
        wr = [(i, r) for i, r in enumerate(rows) if r["world"] == w]
        pol = {k: [] for k in ("A", "B", "planner", "hold", "random",
                               "ceiling", "worstA", "worstB")}
        reps, spreads = [], []
        for i, r in wr:
            tot, worst, rep = realized(r)
            reps.append(rep)
            spreads.append(max(tot) - min(tot))
            a_pick = choices["A"]["pick"][i]
            b_pick = choices["B"]["pick"][i]
            pol["A"].append(tot[a_pick])
            pol["B"].append(tot[b_pick])
            pol["planner"].append(tot[0])
            pol["hold"].append(tot[1])
            pol["random"].append(float(np.mean(tot)))
            pol["ceiling"].append(max(tot))
            pol["worstA"].append(worst[a_pick])
            pol["worstB"].append(worst[b_pick])
        rep90 = float(np.percentile(reps, 90))
        if float(np.mean(spreads)) <= 10 * max(rep90, 1e-9):
            unable.append(w)
            continue
        per_world[w] = {k: float(np.mean(v)) for k, v in pol.items()}
        per_world[w]["replay_p90"] = rep90

    def ci(diffs):
        from scipy.stats import t
        d = np.asarray(diffs, dtype=float)
        if len(d) < 2 or not np.isfinite(d).all():
            return {"mean": None, "ci95": [None, None], "valid": False}
        m, s = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
        spread = float(t.ppf(.975, len(d) - 1)) * s
        return {"mean": m, "ci95": [m - spread, m + spread], "valid": True}

    ws = sorted(per_world)
    report = {"worlds_answering": ws, "worlds_unable": unable,
              "per_world": {str(w): per_world[w] for w in ws}}
    for sel in ("A", "B"):
        vs_planner = ci([per_world[w][sel] - per_world[w]["planner"] for w in ws])
        vs_hold = ci([per_world[w][sel] - per_world[w]["hold"] for w in ws])
        capture = ci([(per_world[w][sel] - per_world[w]["planner"])
                      / max(per_world[w]["ceiling"] - per_world[w]["planner"], 1e-9)
                      for w in ws])
        worst_vs_planner = ci([per_world[w]["worst" + sel]
                               - np.mean([min(e["wellness"]["tactile"]
                                              for e in r["physical_paths"][0])
                                          for r in rows if r["world"] == w])
                               for w in ws])
        rep_max = max((per_world[w]["replay_p90"] for w in ws), default=None)
        passed = (all(c["valid"] for c in (vs_planner, vs_hold, worst_vs_planner))
                  and rep_max is not None and np.isfinite(rep_max)
                  and vs_planner["ci95"][0] > 0 and vs_hold["ci95"][0] > 0
                  and vs_planner["mean"] > rep_max
                  and worst_vs_planner["ci95"][0] >= -rep_max)
        report[sel] = {"benefit_vs_planner": vs_planner,
                       "benefit_vs_hold": vs_hold,
                       "capture_of_available": capture,
                       "worst_comfort_vs_planner": worst_vs_planner,
                       "replay_p90_max": rep_max,
                       "gate": "PASS" if passed else "FAIL"}
    if report.get("B", {}).get("gate") == "PASS":
        difference = ci([per_world[w]['B'] - per_world[w]['A'] for w in ws])
        report['body_path_comparison'] = difference
        report["body_path_claim"] = (
            "supported" if difference['valid'] and difference['ci95'][0] > 0 else
            "NOT supported: B has not demonstrated a paired advantage over A")
    report["scale"] = ("realized bodily wellness, 0-3 (comfort+capacity+"
                       "posture); benefits are paired world means; ceiling = "
                       "best available candidate, reported only")
    json.dump(report, open(args.data / "choice_report.json", "w"), indent=1,
              default=float)
    print(json.dumps(report, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
