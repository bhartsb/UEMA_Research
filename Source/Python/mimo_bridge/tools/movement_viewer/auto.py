# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The automatic picks: which saved commands get pictures without anyone asking, and why.

The operator's question of October 18, 2026: how often, and when, to make these pictures ("when error is higher and
when it is lower and by dates"). The answer built here: every command is scored (index.py), so pictures are chosen from
the scores instead of at fixed times. INPUT: the index of every recording. OUTPUT: auto.json, a list of picks, each
{recording, body, step, time, reason, value}; the viewer service draws them (service.py) and the panel shows them.

THE RULES (each pick says which one chose it):
  hourly     per recording, body and clock hour of recording, among the planner's commands (author code 1; every author
             when the hour has none): the command with the smallest, the median and the largest HANDS error at its end
             (the mean of the left-hand and right-hand finger errors, degrees), the smallest and the largest WHOLE-BODY
             error (the whole-body orientation measure: the mean orientation error of the powered body parts among the
             15 scored ones, degrees), and one command drawn at random
             (numpy's generator seeded with the recording id, body and hour, so the same command is drawn every time)
  run start  per recording and body: the first planner command at or after its first model step + 100, + 1,000 and
             + 10,000 model steps (a repair or any other change starts a new run)
  new best   within each run, per body: an hour whose median HANDS error is the lowest (or the highest) of every
  new worst  earlier hour of the same run; the pick is that hour's median command
  test run   per test recording and body: its median and its largest HANDS error over the whole run
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, List

import numpy as np

from . import HOME, part_path, replace_file
from . import index as I

AUTO_FILE = HOME / "auto.json"
# the rules in plain sentences, for the page (the module doc says the same)
RULES = (
    "Every hour of every recording, for each body: the planner's command with the smallest, the middle and the largest "
    "hands distance at its end (the average of the two hands' finger distances from their asked-for angles, degrees), "
    "and with the smallest and the largest whole-body distance (the average angle of the powered body parts among the "
    "15 scored ones from their asked-for directions); and one command drawn at random.",
    "At each run's start: the first planner command 100, 1,000 and 10,000 model steps after it (a repair or any other "
    "change starts a new run).",
    "Within each run, for each body: an hour whose middle hands distance is lower (or higher) than every earlier hour "
    "of the same run (runs with other settings, and sibling runs restarted from one checkpoint, are never joined).",
    "In each test run, for each body: its middle and its largest hands distance over the whole run.",
)
RUN_OFFSETS = (100, 1000, 10000)


def _hands(cols: Dict[str, np.ndarray]) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.stack([cols["m_fingers_L_e"], cols["m_fingers_R_e"]]).astype(np.float64), axis=0)


def _seed(*parts) -> int:
    return int(hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def picks_for(rec: dict) -> List[dict]:
    out: List[dict] = []
    for b in sorted(int(k) for k in rec["bodies"]):
        cols = I.body_cols(rec["id"], b)
        if cols is None or not len(cols["step"]):
            continue
        t = cols["time"]
        step = cols["step"]
        planner = cols["author"] == 1
        hands = _hands(cols)
        whole = cols["m_body_turn_e"].astype(np.float64)

        def add(i, reason, value=None):
            out.append(dict(recording=rec["id"], label=rec["label"], kind=rec["kind"], body=b, step=int(step[i]),
                            time=float(t[i]), reason=reason, value=None if value is None or not np.isfinite(value)
                            else round(float(value), 3)))
        hours = np.floor(t / 3600.0).astype(np.int64)
        for hr in np.unique(hours):
            inhour = hours == hr
            pool = inhour & planner
            if not pool.any():
                pool = inhour
            when = time.strftime("%I %p", time.localtime(hr * 3600)).lstrip("0")
            for name, vals in (("hands", hands), ("whole body", whole)):
                ok = pool & np.isfinite(vals)
                if not ok.any():
                    continue
                idx = np.flatnonzero(ok)
                v = vals[idx]
                add(int(idx[np.argmin(v)]), "smallest %s distance in the hour from %s" % (name, when), float(v.min()))
                add(int(idx[np.argmax(v)]), "largest %s distance in the hour from %s" % (name, when), float(v.max()))
                if name == "hands":
                    k = int(idx[np.argsort(v, kind="stable")[len(v) // 2]])
                    add(k, "middle hands distance in the hour from %s" % when, float(vals[k]))
            idx = np.flatnonzero(pool)
            rng = np.random.default_rng(_seed(rec["id"], b, int(hr)))
            k = int(rng.choice(idx))
            add(k, "drawn at random in the hour from %s" % when, float(hands[k]))
        first = int(step.min())
        for off in RUN_OFFSETS:
            cand = np.flatnonzero(planner & (step >= first + off))
            if len(cand):
                add(int(cand[0]), "first planner command %s model steps after this run's start" % format(off, ","),
                    float(hands[cand[0]]))
        if rec["kind"] == "test":
            ok = np.isfinite(hands) & planner
            if ok.any():
                idx = np.flatnonzero(ok)
                v = hands[idx]
                add(int(idx[np.argsort(v, kind="stable")[len(v) // 2]]), "middle hands distance of this test run",
                    float(np.median(v)))
                add(int(idx[np.argmax(v)]), "largest hands distance of this test run", float(v.max()))
    return out


def records_by_body(recordings: List[dict]) -> List[dict]:
    """Within each recording (one run, one set of control settings), per body: an hour whose median hands error of the
    planner's commands is lower (new best) or higher (new worst) than every earlier hour of the same run. Runs are not
    joined into one timeline: sibling runs restarted from one checkpoint, and runs with other settings, are separate
    examples (the independent review of Oct 18, 2026, finding 3)."""
    out = []
    for rec in recordings:
        for b in sorted(int(k) for k in rec["bodies"]):
            cols = I.body_cols(rec["id"], b)
            if cols is None or not len(cols["step"]):
                continue
            hands = _hands(cols)
            ok = (cols["author"] == 1) & np.isfinite(hands)
            hours = np.floor(cols["time"] / 3600.0).astype(np.int64)
            lo, hi = np.inf, -np.inf
            for hr in np.unique(hours[ok]):
                idx = np.flatnonzero(ok & (hours == hr))
                v = hands[idx]
                med = float(np.median(v))
                k = int(idx[np.argsort(v, kind="stable")[len(v) // 2]])
                for kind_, better in (("new best", med < lo), ("new worst", med > hi)):
                    if better and np.isfinite(lo if kind_ == "new best" else hi):
                        out.append(dict(recording=rec["id"], label=rec["label"], kind=rec["kind"], body=b,
                                        step=int(cols["step"][k]), time=float(cols["time"][k]), value=round(med, 3),
                                        reason="%s hourly middle hands distance of this run for body %d" % (kind_, b)))
                lo, hi = min(lo, med), max(hi, med)
    return out


def compute(recordings: List[dict]) -> dict:
    picks: List[dict] = []
    for rec in recordings:
        picks += picks_for(rec)
    picks += records_by_body(recordings)
    merged: Dict[tuple, dict] = {}
    for p in picks:
        key = (p["recording"], p["body"], p["step"])
        if key in merged:
            merged[key]["reasons"].append(p["reason"])
        else:
            merged[key] = dict(p, reasons=[p["reason"]])
            merged[key].pop("reason")
    return dict(computed_at=time.time(), picks=sorted(merged.values(), key=lambda p: -p["time"]), rules=list(RULES))


def write(result: dict, path=AUTO_FILE) -> None:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    tmp = part_path(path)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f)
    replace_file(tmp, path)


def load(path=AUTO_FILE) -> dict:
    try:
        with open(str(path), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return dict(computed_at=None, picks=[])
