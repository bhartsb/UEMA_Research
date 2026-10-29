# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The meta-overseer: watches the overseer and pushes it to get better at the project's goal.

    python -m mimo_bridge.tools.meta_overseer            # loop every 15 minutes
    python -m mimo_bridge.tools.meta_overseer --once

The project's goal (from the Fast Gain-of-Function Plan): visible, measurable gain of
function in the baby avatar - prediction of its own senses, infant motor milestones
(rolling, reaching), telling situations apart, body maps from touch.

What the meta-overseer does every pass:
  1. AUDIT the overseer against reality:
       - every incident that actually happened (crashed run, stalled run, NaN, crash
         snapshot) - did the overseer warn or act BEFORE it, or did it miss it?
       - every intervention the overseer made (advance_phase, snapshot, stop, kill) -
         did it help? (curiosity revived? run continued? chain moved on?)
  2. SCORECARD the project: across runs in time order, are the goal numbers moving?
       rolling (deg), ball reach (cm and touch rate), the parallel-body
       action-signature ratio (an UNCALIBRATED proxy - not situational awareness,
       see scorecard()), prediction error, touch probe accuracy.
  3. TUNE the overseer with evidence: writes overseer_directives.json (thresholds the
     overseer reads at every pass). A missed incident tightens the relevant threshold;
     warnings that repeatedly come to nothing loosen it. All changes bounded and logged
     with the reason.
  4. NAME THE GAPS: failure signatures with no matching overseer check land in a "gaps"
     list on the dashboard - the work queue for improving the overseer itself (the
     thinking layer / the humans pick these up).

Everything goes to Saved/mimo_bridge/meta_overseer_log.jsonl and the dashboard's
meta-overseer panel.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
SAVED = os.path.join(REPO, "Saved", "mimo_bridge")
DIRECTIVES = os.path.join(SAVED, "overseer_directives.json")
LOG = os.path.join(SAVED, "meta_overseer_log.jsonl")

DEFAULT_DIRECTIVES = {"STALL_MIN": 10.0, "COLLAPSE_STD": 0.02, "MIN_RAM_GB": 2.5, "MIN_DISK_GB": 8.0}
BOUNDS = {"STALL_MIN": (4.0, 20.0), "COLLAPSE_STD": (0.005, 0.05), "MIN_RAM_GB": (1.5, 6.0), "MIN_DISK_GB": (4.0, 15.0)}

# failure signatures the overseer has (or should have) a check for
KNOWN_SIGNATURES = {
    r"WinError 1450|Insufficient system resources": "pipe/pool exhaustion (fixed in code: slim pipes)",
    r"Shadow framebuffer|GL": "graphics context exhaustion (fixed in code: shared GL context)",
    r"Could not allocate memory|_ArrayMemoryError|MemoryError": "out of memory (overseer RAM check)",
    r"did not answer for .* s \(hung\)": "hung worker (pipe deadline)",
    r"NaN": "NaN (overseer check)",
}


def read_jsonl(path, kind=None):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if kind is None or r.get("type") == kind:
                    out.append(r)
    except FileNotFoundError:
        pass
    return out


def all_runs():
    runs = []
    for base in (os.path.join(SAVED, "experiments"), os.path.join(SAVED, "runs")):
        for d in sorted(glob.glob(os.path.join(base, "*"))):
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "args.json")):
                runs.append(d)
    return runs


# ---------------------------------------------------------------- 1. audit
def audit():
    caught, missed, interventions = [], [], []
    for run in all_runs():
        name = os.path.basename(run)
        olog = read_jsonl(os.path.join(run, "overseer_log.jsonl"), "overseer")
        summ = None
        try:
            summ = json.load(open(os.path.join(run, "summary.json")))
        except Exception:
            pass
        # incident: the run crashed
        if summ and not summ.get("ok") and summ.get("exception"):
            sig = summ["exception"].strip().splitlines()[-1][:120]
            t_end = os.path.getmtime(os.path.join(run, "summary.json"))
            # AUDIT FIX: credit only INTERVENTION-class actions in a bounded window before
            # the crash - a routine hourly snapshot hours earlier is not "catching" it
            warned = any(0 <= t_end - o["timestamp"] < 900
                         and any(("snapshot_and_stop" in a or "sent stop" in a or "killed" in a)
                                 for a in (o.get("actions") or []))
                         for o in olog)
            (caught if warned else missed).append({"run": name, "incident": sig, "overseer_acted_before": warned, "when": t_end})
        # incident: stall kill by the overseer (counted as caught by definition)
        for o in olog:
            for a in o.get("actions") or []:
                interventions.append({"run": name, "time": o["timestamp"], "action": a})
        # outcome of advance_phase: curiosity before vs after
        recs = read_jsonl(os.path.join(run, "replay_training_stats.jsonl"), "mimo")
        cmds = read_jsonl(os.path.join(run, "replay_training_stats.jsonl"), "overseer_cmd")
        adv_times = sorted(c["timestamp"] for c in cmds if c.get("cmd") == "advance_phase")
        rapid = sum(1 for a, b in zip(adv_times, adv_times[1:]) if b - a < 900)
        if rapid:
            interventions.append({"run": name, "action": f"OVERSEER MISBEHAVIOR: advanced phases twice within 15 min ({rapid}x)"})
        for c in cmds:
            if c.get("cmd") == "advance_phase":
                before = [r.get("curiosity_drive", 0) for r in recs if r["timestamp"] < c["timestamp"]][-10:]
                after = [r.get("curiosity_drive", 0) for r in recs if r["timestamp"] > c["timestamp"]][:10]
                if after:
                    helped = float(np.mean(after)) > float(np.mean(before or [0])) + 0.05
                    interventions.append({"run": name, "action": "advance_phase outcome",
                                          "curiosity_before": round(float(np.mean(before or [0])), 2),
                                          "curiosity_after": round(float(np.mean(after)), 2),
                                          "helped": bool(helped)})
    return caught, missed, interventions


# ---------------------------------------------------------------- 2. scorecard
def scorecard():
    rows = []
    for run in all_runs():
        recs = read_jsonl(os.path.join(run, "replay_training_stats.jsonl"), "mimo")
        if not recs:
            continue
        last = recs[-1]
        t0 = recs[0].get("timestamp", os.path.getctime(run))
        pe = last.get("pe", {})
        real = [pe.get(m) for m in ("vision", "tactile", "proprioception") if pe.get(m) is not None]
        rows.append({"run": os.path.basename(run), "start": t0,
                     "roll_deg_max": last.get("roll_deg_max"),
                     "ball_touch_rate": last.get("task_ball_touch_rate"),
                     "ball_dist_cm": None if last.get("task_hand_ball_dist_m") is None else round(100 * last["task_hand_ball_dist_m"]),
                     "discrimination": last.get("motor_discrimination_ratio"),
                     "pe_real": None if not real else round(float(np.mean(real)), 4)})
    rows.sort(key=lambda r: r["start"])
    # The runs compared here are NOT matched: body count, scene, resumed checkpoint,
    # duration and flags all differ between the earliest and the newest. So the trend
    # states which DIRECTION the number moved and nothing about why. "improving" was a
    # claim the comparison cannot support, and for the discrimination ratio it is not
    # even a claim about the model: that ratio is between-body spread over within-body
    # variance, and it rises when a body simply moves LESS. It is renamed on the way
    # out so no consumer can print it as situational awareness.
    trend = {}
    for key, out_key in (("roll_deg_max", "roll_deg_max"),
                         ("discrimination", "action_signature_ratio")):
        vals = [(r["start"], r[key]) for r in rows if r[key] is not None]
        if len(vals) >= 2:
            trend[out_key] = ("higher than the earliest unmatched run" if vals[-1][1] > vals[0][1]
                              else "not higher than the earliest unmatched run")
    return rows[-8:], trend


# ---------------------------------------------------------------- 3. tune
def tune(caught, missed):
    d = dict(DEFAULT_DIRECTIVES)
    try:
        d.update(json.load(open(DIRECTIVES)))
    except Exception:
        pass
    changes = []
    # each incident may tighten the dials ONCE: only incidents newer than the last
    # directive update count (the old rule re-applied the same day-old crash every
    # pass, ratcheting MIN_RAM_GB to its ceiling and killing healthy launches)
    last_upd = 0.0
    try:
        last_upd = time.mktime(time.strptime(d.get("updated", ""), "%Y-%m-%d %H:%M"))
    except Exception:
        pass
    missed = [m for m in missed if m.get("when", time.time()) > last_upd]
    # a missed crash means the overseer was too slow or too lax: tighten the stall window
    if missed:
        new = max(BOUNDS["STALL_MIN"][0], d["STALL_MIN"] - 2.0)
        if new != d["STALL_MIN"]:
            changes.append(f"STALL_MIN {d['STALL_MIN']} -> {new} because {len(missed)} incident(s) happened without prior overseer action")
            d["STALL_MIN"] = new
    # memory incidents: raise the RAM floor
    if any("memory" in (m.get("incident") or "").lower() for m in missed):
        new = min(BOUNDS["MIN_RAM_GB"][1], d["MIN_RAM_GB"] + 1.0)
        if new != d["MIN_RAM_GB"]:
            changes.append(f"MIN_RAM_GB {d['MIN_RAM_GB']} -> {new} because an out-of-memory incident was missed")
            d["MIN_RAM_GB"] = new
    for k, (lo, hi) in BOUNDS.items():
        d[k] = float(min(hi, max(lo, d[k])))
    if changes:
        d["updated"] = time.strftime("%Y-%m-%d %H:%M")
        d["reasons"] = changes
        json.dump(d, open(DIRECTIVES, "w"), indent=1)
    return d, changes


# ---------------------------------------------------------------- 4. gaps
def gaps(missed):
    out = []
    for m in missed:
        sig = m.get("incident") or ""
        known = next((v for k, v in KNOWN_SIGNATURES.items() if re.search(k, sig)), None)
        if known:
            out.append(f"{m['run']}: {sig[:90]} -> covered by: {known}")
        else:
            out.append(f"{m['run']}: {sig[:90]} -> NO overseer check exists for this failure; a new check or code fix is needed")
    return out


def too_perfect(run):
    """Numbers that are impossibly good are usually self-grading. This one rule would have
    caught the gaze loss (0.0000 for two days: it graded its own echo) and the motor_state
    gauge (pegged at exactly 1.0 all run). Flags: any loss-breakdown term averaging below
    1e-3 for a whole run, and any wellness gauge sitting at exactly 0.0 or 1.0 throughout."""
    flags = []
    try:
        if json.load(open(os.path.join(run, "args.json"))).get("eval_only"):
            return []          # frozen runs learn nothing: ~0 losses are expected there
    except Exception:
        pass
    tr = read_jsonl(os.path.join(run, "replay_training_stats.jsonl"), "training")
    if len(tr) >= 10:
        keys = (tr[-1].get("loss_breakdown") or {}).keys()
        for k in keys:
            vals = [abs((r.get("loss_breakdown") or {}).get(k, 1)) for r in tr[-30:]]
            if vals and max(vals) < 1e-3:
                flags.append(f"loss term '{k}' ~0 all run: TOO PERFECT - check for self-grading")
    recs = read_jsonl(os.path.join(run, "replay_training_stats.jsonl"), "mimo")
    if len(recs) >= 10:
        for k in (recs[-1].get("wellness") or {}):
            vals = [(r.get("wellness") or {}).get(k) for r in recs[-30:] if (r.get("wellness") or {}).get(k) is not None]
            if vals and (all(v >= 0.999 for v in vals) or all(v <= 0.001 for v in vals)):
                flags.append(f"wellness gauge '{k}' pegged at {vals[-1]:.0f} all run: dead or saturated")
    return flags


def false_kills():
    """Every overseer stop is audited against what happened AFTER it: a resource stop whose
    shortage vanished within minutes was a false kill (this caught pattern stopped H5 and H6
    at launch). Also flags any directive sitting exactly at its bound - a pegged dial means
    the tuner ratcheted."""
    out = []
    try:
        d = json.load(open(DIRECTIVES))
        for k, (lo, hi) in BOUNDS.items():
            if float(d.get(k, (lo + hi) / 2)) in (lo, hi):
                out.append(f"directive {k} pegged at its bound {d[k]}: tuner may have ratcheted")
    except Exception:
        pass
    for run in all_runs():
        olog = read_jsonl(os.path.join(run, "overseer_log.jsonl"), "overseer")
        for o in olog:
            if not any("snapshot_and_stop" in a for a in (o.get("actions") or [])):
                continue
            reasons = " ".join((o.get("findings") or {}).get("cto") or [])
            if "RAM" in reasons:
                # AUDIT FIX: the old test measured how fast the trainer complied, not
                # whether the shortage was real. A RAM stop is suspect only when it hit
                # a run still in its LAUNCH window (workers spawning = transient dips).
                try:
                    started = os.path.getmtime(os.path.join(run, "args.json"))
                except OSError:
                    continue
                if 0 <= o["timestamp"] - started < 600:
                    out.append(f"{os.path.basename(run)}: RAM stop {o['timestamp']-started:.0f}s after "
                               f"launch - likely a startup-transient FALSE KILL")
    return out[-8:]


def watchdog_the_watchdog():
    """Is the overseer itself alive, current, and running its vitals checks?

    Learned the hard way: the overseer ran for half a day with no knowledge of the vitals
    bands sitting on its own disk, and nothing anywhere said so. So the meta-overseer now
    checks the checker: for each active run, the overseer must have reported recently, its
    report must carry the vitals marker (proof the running code knows about bands), and a
    VITALS violation that persists across three consecutive reports is escalated here
    rather than left scrolling by in a log nobody reads."""
    from mimo_bridge.tools.overseer import active_runs, read_tail
    problems, escalations = [], []
    for run in active_runs():
        path = os.path.join(run, "overseer_log.jsonl")
        reps = read_tail(path, "overseer") if os.path.exists(path) else []
        name = os.path.basename(run)
        if not reps:
            problems.append(f"{name}: overseer has never reported on this run")
            continue
        last = reps[-1]
        age_min = (time.time() - last.get("timestamp", 0)) / 60.0
        if age_min > 10:
            problems.append(f"{name}: overseer's last report is {age_min:.0f} min old")
        if not last.get("has_vitals"):
            problems.append(f"{name}: overseer is running STALE CODE (no vitals in its report)")
        persist = None
        tails = [set(x for x in (r.get("findings", {}).get("cto") or []) if x.startswith("VITALS:"))
                 for r in reps[-3:]]
        if len(tails) == 3 and (common := tails[0] & tails[1] & tails[2]):
            for c in sorted(common):
                escalations.append(f"{name}: unresolved for 3+ passes: {c}")
    return problems, escalations


def pass_once():
    caught, missed, interventions = audit()
    rows, trend = scorecard()
    directives, changes = tune(caught, missed)
    tp = []
    for run in all_runs()[-6:]:
        for f in too_perfect(run):
            tp.append(f"{os.path.basename(run)}: {f}")
    wd_problems, wd_escalations = watchdog_the_watchdog()
    rep = {"type": "meta_overseer", "timestamp": time.time(),
           "overseer_health": wd_problems, "vitals_escalations": wd_escalations,
           "too_perfect": tp[-8:], "false_kills": false_kills(),
           "audit": {"incidents_with_prior_action": len(caught), "incidents_missed": len(missed),
                     "missed": missed[-5:], "recent_interventions": interventions[-8:]},
           "scorecard": rows, "goal_trend": trend,
           "directives": directives, "directive_changes": changes,
           "gaps": gaps(missed)}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rep) + "\n")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=float, default=900.0)
    a = ap.parse_args()
    from mimo_bridge.tools.selfreload import stamp as _stamp, exec_if_changed as _reexec
    import sys as _sys
    _t0 = _stamp()
    while True:
        _reexec(_t0, "mimo_bridge.tools.meta_overseer", _sys.argv[1:], log=print)
        rep = pass_once()
        if a.once:
            print(json.dumps(rep, indent=1, default=str))
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()

