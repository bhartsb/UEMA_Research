# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Section 30: how far ahead the learner's imagination is accurate enough to choose by.

"Back down legs first" cannot be coded as a goal; it can only emerge if the learner predicts far
enough ahead for a safer path to win. So the practical horizon is MEASURED, never assumed:

  per-depth accuracy     the planner's licence (lem_model.last_plan_license): for each imagined
                         depth (1, 2, 3 model steps = 50, 100, 150 ms), the share of moments the
                         imagined chain was closer to what happened than "nothing changes" was.
                         0.5 is that do-nothing line, a floor and never a goal: it says the chain
                         is informative, not that it can choose
  choosing by prediction the planner's evaluator ranks its chosen plan against four alternatives
                         (hold, reflex, two perturbations); 'best' means it beat them all by more
                         than the measured tie band (causal_chain.py). The share of 'best' is the
                         share of choices the prediction actually decided
  depth in use           the horizon each planning event ran at
  does depth change      whether a deeper horizon changes which plan is selected and improves the
  the choice             physical outcome needs a paired test from identical saved starts (depth 1
                         against depth 3, same candidates, executed); it is not measured here and
                         the report says so with that test named
"""
from __future__ import annotations

from typing import List

import numpy as np


def report(lane, chain: List[dict]) -> dict:
    out = dict(depths=[], licensed_depth=None, planner_events=0)
    lic = list(getattr(lane, "licenses", []) or []) if lane is not None else []
    if lic:
        step, last = lic[-1]
        win = list(last.get("win_emas") or [])
        out["depths"] = [dict(depth=i + 1, horizon_ms=50 * (i + 1), beats_do_nothing_share=float(w),
                              line=0.5, informative=bool(w > 0.5)) for i, w in enumerate(win)]
        out["licensed_depth"] = int(last.get("licensed", 0))
        out["licence_step"] = int(step)
        out["mode"] = last.get("mode")
        if last.get("mode") == "earned":
            # lem_model: an earned horizon plans 1 step ahead plus one per licensed depth
            out["planner_max_steps"] = max(1, min(1 + out["licensed_depth"], len(win)))
        hist = [(s, list(l.get("win_emas") or [])) for s, l in lic[-200:]]
        out["history"] = [dict(step=int(s), win=[float(x) for x in w]) for s, w in hist]
    lat = [r for r in chain if r.get("type") == "latch"]
    out["planner_events"] = len(lat)
    outcomes = []
    horizons = []
    for r in lat:
        plan = r.get("plan") or {}
        horizons.append(int(plan.get("horizon", 0) or 0))
        for o in ((plan.get("pred_wellness_gaps") or {}).get("outcome") or []):
            if o is not None:
                outcomes.append(o)
    if outcomes:
        n = len(outcomes)
        out["choices"] = dict(judged=n, best=sum(1 for o in outcomes if o == "best"),
                              tied=sum(1 for o in outcomes if o == "tied"),
                              behind=sum(1 for o in outcomes if o == "behind"))
        out["decided_share"] = out["choices"]["best"] / n
    if horizons:
        out["depth_in_use"] = dict(zip(*[list(map(int, a)) for a in np.unique(horizons, return_counts=True)]))
        out["deeper_than_one_share"] = float(np.mean(np.asarray(horizons) > 1))
    out["deeper_horizon_changes_choice"] = dict(
        measured=False,
        next_test=("paired planner comparison from identical saved starts: the same candidates "
                   "selected at depth 1 and at depth 3, both executed, physical outcome and "
                   "wellness compared; a candidate for tools/ability_rechecks once the planner's "
                   "decided share is above zero (below that, no depth can change a tied choice)"))
    return out
