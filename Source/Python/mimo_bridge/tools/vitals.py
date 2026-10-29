# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Vital signs for a RUNNING run: is the content sane, not just the process alive?

One band table, one evaluator, used by BOTH the overseer (alarms every pass) and the
dashboard (display). Two copies of the same bands would drift - the calibrate.py lesson:
a checker holding its own copy of a number ends up passing a value nobody runs.

Every band cites where its number came from. A band violation is an ALARM, never an
auto-stop: semantic checks can be wrong about a new experiment, and the cost of a false
stop is hours; the cost of a loud false alarm is a glance.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

# (key, extractor-path, lo, hi, why this band)
# extractor-path: dotted into the "mimo" stats record; "rate:" prefix = per minute of
# body time, computed from the change between the first and last record in the window.
BANDS = [
    ("saccades per baby-minute", "rate:task_saccades", 40.0, 300.0,
     "measured 120/min live; an infant ~60-180. Dead-at-reset read ~2 (bug 93aea0ec)"),
    ("eye pitch (deg, + is up)", "task_eye_pitch", 0.0, 95.0,
     "below zero = looking at the floor; the plank-posture bug read -48 for seven runs"),
    ("fresh vision fraction", "vision_fresh_fraction", 0.99, 1.0,
     "every model step must see a new image; 0.5 = the old every-other-frame repeat"),
    ("imagination alignment error", "wellness_align_err", 0.0, 0.30,
     "measured 0.28 the day the formulas landed, 0.04 after six hours; above 0.3 = broken"),
    ("limb follows commands (bout cosine)", "follow_bout|cmd_response_cos_held", -0.038, 1.0,
     "obedience judged on movement bouts only (follow_bout; measured mean 0.006 +/- 0.015 "
     "over 8h of _y, line = mean - 3x spread). The raw held cosine goes NEGATIVE by "
     "construction during pose-hold rests (the hold command opposes drift) and read "
     "-0.18..-0.24 on a healthy rest-era run; it is only the fallback for pre-rest "
     "stats files that lack follow_bout"),
    ("vision prediction error", "pe.vision", 0.0005, 0.8,
     "0 = the loss is disconnected; 0.8+ = collapse. The old 0.003 floor was set when "
     "healthy read 0.02-0.2; by 2026-09-08 healthy reads 0.0027-0.003 and brushed the "
     "floor. A disconnected loss reads 0.0000; 0.0005 sits 5x under measured healthy"),
    ("touch patches active", "contacts_per_env", 2.0, 80.0,
     "a crib body rests on the floor: 15-20 normal; 0 = touch pipeline dead"),
    ("commanded rotation (rad)", "cmd_mag_rad", 0.10, 2.0,
     "the SIZE of the orders being sent, upstream of whether the body follows them. "
     "Measured 0.42-0.68 across H20-H22 (planner on) and 0.03-0.08 across H23-H28 "
     "(planner off); body speed fell 8.7x with it. The floor sits below every run "
     "that moved and above every run that did not. Near zero = nothing is being asked "
     "of the body, so no causal test can resolve anything"),
    ("body speed (cm/s)", "movement.local_speed", 5.0, 120.0,
     "25-40 normal; near zero = frozen body, huge = physics blow-up"),
]


def _get(rec: dict, path: str):
    cur = rec
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur if isinstance(cur, (int, float)) else None


def evaluate(stats_path: str, window: int = 30) -> List[dict]:
    """Band-check the last `window` mimo records. Returns one row per band with a verdict:
    OK / LOW / HIGH / NO DATA. Missing instruments are reported, not skipped - a metric
    that silently vanishes is itself a finding (that is how half of them went unlogged)."""
    recs = []
    try:
        with open(stats_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if '"mimo"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("type") == "mimo":
                    recs.append(r)
    except OSError:
        return [dict(name="stats file", value=None, band=None, verdict="NO DATA",
                     why=f"cannot read {stats_path}")]
    recs = recs[-window:]
    out = []
    for name, path, lo, hi, why in BANDS:
        if path.startswith("rate:"):
            key = path[5:]
            vals = [( _get(r, key), r.get("step"), r.get("elapsed_s")) for r in recs]
            vals = [(v, s) for v, s, _ in vals if v is not None and s is not None]
            if len(vals) < 2 or vals[-1][1] == vals[0][1]:
                out.append(dict(name=name, value=None, band=[lo, hi], verdict="NO DATA", why=why))
                continue
            # cumulative counters reset when a run restarts into the same stats file; a
            # window spanning the reset made the rate NEGATIVE (-975/min seen live) and the
            # overseer alarmed on an artifact. Use only the monotone tail after the last
            # reset.
            last_reset = 0
            for j in range(1, len(vals)):
                if vals[j][0] < vals[j - 1][0]:
                    last_reset = j
            vals = vals[last_reset:]
            if len(vals) < 2 or vals[-1][1] == vals[0][1]:
                out.append(dict(name=name, value=None, band=[lo, hi], verdict="NO DATA", why=why))
                continue
            dv = vals[-1][0] - vals[0][0]
            dsteps = vals[-1][1] - vals[0][1]
            baby_min = dsteps * 5 * 0.01 / 60.0          # action_repeat 5; bands are coarse
            value = dv / max(baby_min, 1e-9)
        else:
            # "a|b" = alternatives: the first key with any data in the window wins for
            # ALL records (resolved per window, never per record - averaging two
            # different metrics into one number would blend their semantics)
            vs = []
            for alt in path.split("|"):
                vs = [_get(r, alt) for r in recs]
                vs = [v for v in vs if v is not None]
                if vs:
                    break
            if not vs:
                out.append(dict(name=name, value=None, band=[lo, hi], verdict="NO DATA", why=why))
                continue
            value = sum(vs[-10:]) / len(vs[-10:])
        verdict = "OK" if lo <= value <= hi else ("LOW" if value < lo else "HIGH")
        out.append(dict(name=name, value=round(float(value), 4), band=[lo, hi],
                        verdict=verdict, why=why))
    return out


def problems(rows: List[dict]) -> List[str]:
    return [f"{r['name']}: {r['verdict']}"
            + (f" at {r['value']}" if r["value"] is not None else "")
            + f" (band {r['band']}) - {r['why']}"
            for r in rows if r["verdict"] != "OK"]
