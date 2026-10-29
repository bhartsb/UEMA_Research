# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A small, fixed cross-capability regression pack for H84, run after every repair's watch.

    python -m mimo_bridge.tools.regression_pack compare --before <checkpoint | A-B> --after <checkpoint | A-B>
        [--switch-step S] [--window 3000] [--no-model] [--out DIR]
    python -m mimo_bridge.tools.regression_pack checkpoint <checkpoint> [<checkpoint> ...] [--after-switch S] [--out DIR]
    python -m mimo_bridge.tools.regression_pack noise --checkpoint C --observer-from A --observer-to B --out FILE

WHY (the operator's decision of 2026-09-25 7:47 PM PDT, docs/gain_of_function_v3/WORKLOG.md): "After every
repair watch, run a small fixed cross-capability regression pack, not only the metric directly associated with
that repair ... eye prediction, body action sensitivity, proprio variation, vestibular variation, hand-following,
observer behavior counts. Why? Because these failures appear coupled. A repair can improve the intended subsystem
while harming another." And: body 2 (numbered from 0, the dashboard's body 3; face down since Sept 19) is
excluded from learning-performance conclusions until its mechanics are understood, so every reading is given with
and without body 2, and the verdict is read on the reading without it.

THE SIX READINGS (the table's headline row for each; every one reads higher = better except the observer's
count of dropped rows). Each is measured on the checkpoint itself, so a comparison needs no running trainer:

 1 eye prediction      how much less next-picture prediction error the body's own eye command leaves than a
                       command that holds the eyes where they are, both sent through the same gaze copy (the
                       efference path that shifts the predicted picture with the eyes; lem.model.efference.
                       GazeCopy), as a share of the picture's own change (the error of predicting that the picture
                       stays the same), on the checkpoint's own replay ring: (error(eyes held) - error(own)) /
                       error(unchanged). The eyes-held command is the step-from order with its eye row replaced by
                       the eye row of the order in force one moment earlier: the command the eyes were last sent,
                       i.e. where they are (the ring keeps the eye command actually sent at every moment, written
                       by LemLite.gaze_to_handle in the model's own eye-row convention, and not the eye joints).
                       0% = the eye command adds nothing a still-eye command does not, 100% = its advantage is the
                       whole change of the picture, below 0% = the copy moves the predicted picture the wrong way
                       or too far. WHY THIS CONTROL (review of 2026-09-25, section 1): against ANOTHER body's eye
                       command the reading measures how much the prediction depends on the command, not whether it
                       is right - in a linear model of the copy the right response read 100%, twice it 199% and
                       three times 298%, although at three times the prediction of the eye-driven change is worse
                       than assuming the eyes stay still. Against the eyes held, the same model reads 50%, 0% and
                       -150%: an over-reacting copy reads LOWER. The other-body form stays beside it
                       (eye_vs_other_body), with the ratio of errors 1 - own / other (eye_share_of_error, the live
                       gauge's form). Production's live gauge (efference.gaze_copy: 1 - vision_err_with_copy /
                       vision_err_copy_shuffled) is given beside it for all 8 bodies (not kept per body).
 2 body action         how far the model's body-position (proprioception) prediction moves when the order it
   sensitivity         was given is swapped for another body's order at the same moment, as a share of how far
                       the prediction moves from the present at all: sqrt(sum |p(own) - p(other)|^2 /
                       sum |p(own) - now|^2) over the ring's moments (the prediction audit's `sensitivity`,
                       mimo_bridge/prediction_audit.py). 0% = the order is ignored, 100% = the order moves the
                       prediction as far as the whole predicted change. Beside it: whether the own order also
                       predicts better (1 - error(own) / error(other)); production's live audit for all 8 bodies.
                       The other order is the previous body's at the same moment AMONG THE BODIES KEPT (partners):
                       in the readings without body 2, body 3's control is body 1's order, never body 2's.
 3 proprioception      the collapse watch's quantity on every body (LemLite.token_temporal_std; per body on the
   variation           proprioception branch as token_temporal_std_bodies): the standard deviation over time of
                       a body's raw body-position token (200 samples, every 5th stored moment, 50 s), mean over
                       its 192 numbers, averaged over the ring's six 50-s windows. A token has root mean square
                       1 per number, so 0% = the same token every moment and 100% = an unrelated token every
                       moment. The deaf line of the per-body alert (mimo_bridge/deaf_senses.py on
                       fix/proprio-token-2026-09-25): 20%, from 480 fifty-second windows.
 4 vestibular          the same for the balance token. Its deaf line: 1.6% (the same measurement). Every body
   variation           read 0% from the Sept 23 deploy to Sept 25: the sense is under repair.
 5 hand following      how much of each order's finger turn the fingers closed by the end of the order's
                       tenure (250 ms), measured relative to the wrist (hand goals are joint goals around the
                       wrist, mimo_bridge/hand_target_control.py, so a moving wrist carries the goal with it):
                       1 - sum(turn left at the end) / sum(turn ordered), over every finger of every order in the
                       ring (lem.model.motor_synergy.following_counts on wrist-relative orientations). 0% = the
                       fingers ended no closer than they began, 100% = they arrived. Beside it: out of 100
                       finger orders, how many ended closer to the ordered turn, against the count a random
                       direction of the same size would reach (the function's exact chance line); the wrists'
                       own closure; production's live hand joint error for all 8 bodies (hand_control).
 6 observer behaviour  per catalog capability (docs/gain_of_function_v4/capability_catalog.json), separate
   counts              occasions per 1,000 body-steps in the window, beside babble's over its own body-steps
                       (the developmental observer's lineage store and its random-order babble baseline, counted
                       exactly as tools/observer/tracker.py counts occasions: per body and part, an act 1 s or
                       more after the previous one opens a new occasion). The headline counts the capability
                       rows whose occasion rate FELL between the two windows beyond THE ROW'S OWN MEASURED
                       NO-CHANGE SCATTER (observer_change): the fall in ln(occasions per body-step) against its
                       spread, which is counting chance at the two windows' own counts plus the row's extra
                       scatter measured over the no-change windows (the noise file's observer_rows), read on a t
                       distribution with the windows' degrees of freedom and judged with the Benjamini-Hochberg
                       line across rows at the observer's share of the pack's false-alarm rate. 0 = no behaviour
                       became rarer. WHY (review of 2026-09-25, section 3): occasion counts scatter far beyond
                       counting chance (repeat contact with the same object read 24, 5, 10, 37 and 19 occasions in
                       five no-change windows), and the first rule - beyond counting chance AND beyond the row's
                       largest no-change change - flagged a fallen row in 16 of 40 no-change comparisons.

BESIDE THE SIX (2026-09-26, the memory deploy's readings beyond vision): with the model part, `compare` and
`checkpoint` also read, on the same ring and model, attention spread per attention stage and head, unit liveness
per block and head, and what the memory's variation is worth to each sense's prediction, per body
(mimo_bridge/tools/memory_deploy_readings.py; --no-internals leaves them out). They are judged against their own
no-change lines (memory_deploy_readings.NOISE_FILE) in a second table; the six headlines are unchanged.

NOISE AND LINES (pack v2, 2026-09-26; the review of 2026-09-25, section 2). Each reading's no-change line is the
sampling noise of the two sides measured as REPEATS ON ONE CHECKPOINT, not the spread between checkpoints (which
mixed 15,800 updates of training drift into the proprioception line and the 24-moment sample's own noise into the
eye and body lines): each side's reading is also computed on BATCHES contiguous parts of its own sample (model
readings: the window ends; hand following: the orders by start; the token spreads: their 50-s windows), and its
sampling standard deviation is the batches' spread over the square root of their number (batch means, which keeps
the correlation between neighbouring moments inside each batch). The line is
    t(1 - FALSE_ALARM_PER_READING, dof) x sqrt(sd_before^2 + sd_after^2),
each side's sd the larger of its own batch reading and the noise file's repeat on 415,025 scaled to the side's
sample size (sd_ref x sqrt(n_ref / n)). FALSE_ALARM_PER_READING = 5% / 6 one-sided: a pair of checkpoints with no
change reads REGRESSED on any one reading 1 time in 120, on any of the six at most 1 time in 20. A change beyond
the line in the worse direction is REGRESSED, in the better IMPROVED, else "within noise"; a reading without a
measured sd says so and gives no verdict. The noise file (NOISE_FILE, written by the `noise` command) must carry
this pack version; a file of another version gives no lines.

AFTER A SWITCH (the review of 2026-09-25, section 1). The after side reads only what was lived after the repair
was switched on: --switch-step S (default: the before side's step) starts its records and observer window at S and
restricts every ring reading to the moments added after S (one moment per model step, LemLite.step ->
ReplayMemory.add), whole windows only; a before window that overlaps the after window is refused.

WHAT IS READ. The stats records (replay_training_stats.jsonl of every run of the lineage, a later run's first step
cutting an earlier run's abandoned future), the observer's lineage store and babble baseline, and each
checkpoint's episode companion (its replay ring: every body's raw tokens, pose and executed order at every model
step for the last 300 s). Readings 3-5 need no model. Readings 1-2 build the checkpoint's model on the CPU and run
the replay lesson's own construction (LemLite.replay_update; the audit's replay_inputs in
docs/gain_of_function_v3/audit_visual_regression_2026_09_24.py) with no optimizer: nothing trains, nothing is
written beside the pack's own output. The graphics card is hidden (CUDA_VISIBLE_DEVICES=-1).
"""
from __future__ import annotations

import os
if __name__ == "__main__":  # only when run as a program: importing never changes the devices (2026-09-26)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import argparse
import gc
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PACK_VERSION = "regression_pack_v2"   # v2 2026-09-26: the eyes-held control, lines from repeats on one checkpoint,
                                      # the after side after its switch only, kept-body partners, the observer's scatter
REPO = Path(__file__).resolve().parents[4]
MAIN_REPO = Path("C:/Users/bhart/Documents/UEMA_devA_private")
LINEAGE = "H84_fresh_drivers_20260918"
ARCHIVE = Path("D:/UEMA_Research/external/D/UEMA_recordings/H84_checkpoint_archive")
WINDOW_STEPS = 3000                 # the tracker's retention window and the ability scheduler's cadence
EXCLUDED_BODY = 2                   # face down since Sept 19 (the operator's decision of 2026-09-25 7:47 PM)
NOISE_FILE = REPO / "docs" / "gain_of_function_v3" / "measurements" / "regression_pack_noise_v2_2026-09-26.json"
BIN = 100                           # the observer's exposure bins (tools/observer/daemon.py BIN)
# THE FALSE-ALARM RATE the lines declare: 5% for the six readings together, one-sided, split evenly (Bonferroni)
FALSE_ALARM_FAMILY = 0.05
FALSE_ALARM_PER_READING = FALSE_ALARM_FAMILY / 6
# the batches each side's sample is cut into for its sampling sd (batch means): 20 gives the sd 19 degrees of
# freedom (t = 2.63 at 1 in 120, against 2.39 for a known sd)
BATCHES = 20
# THE MODEL SAMPLE (window ends per body for readings 1-2), from the measurement it must beat. MEASURED 2026-09-25
# (review of the pack, model_sampling_noise.py on 415,025): ten disjoint 24-moment samples per body (body 2 left
# out) spread 6.03 points for body action sensitivity and 6.95 for the eye reading of the first definition; the
# expected spread of ten normal draws is 3.078 sd, so one 24-moment sample's sd is 1.96 and 2.26 points. The change
# the body reading must show is a total loss from about 1% (the live audit's 0.90%, the pack's 24-moment 1.02%), so
# its line must stay under 1.0 point: t x sqrt(2) x 1.96 x sqrt(24 / N) <= 1.0 with t = 2.63 gives N >= 1,270. The
# ring holds 5,870 usable window ends per body (6,000 moments less the 129-moment window), so 1,280 ends lie 4.6
# moments apart, about one per 250-ms order. The same N puts the first eye definition's line at 1.14 points (the
# eyes-held definition's sd is unmeasured until the noise rerun). The arithmetic assumes the moments independent;
# the rerun's batch means measure it with their correlation (slots_needed, test_regression_pack). COST: the model
# part runs 10,240 windows at the measured 0.156 s each: about 27 minutes per checkpoint (it was 30 s at 24).
MODEL_SLOTS = 1280
SAMPLE_SD_24 = {"body_action_sensitivity": 6.03 / 3.078 / 100.0, "eye_vs_other_body": 6.95 / 3.078 / 100.0}
DETECT = {"body_action_sensitivity": 0.010}   # the change the line must be smaller than (a total loss from 1%)
MAX_BATCH = 64                      # lem.model.forward_model's no-grad token buffer
DEAF_LINES = {"proprioception": 0.20, "vestibular": 0.016}   # deaf_senses.LINES (proprioception branch)
SPREAD_WINDOW = 1000                # the collapse watch's window in stored moments (200 samples, every 5th)
SPREAD_EVERY = 5

HEADLINES = ("eye_prediction", "body_action_sensitivity", "proprioception_variation", "vestibular_variation",
             "hand_following", "observer_counts")
MEANING = {
    "eye_prediction": ("how much less next-picture prediction error the body's own eye command leaves than a "
                       "command that holds the eyes where they are, as a share of how much the picture changed",
                       "0% = the eye command adds nothing a still-eye command does not; 100% = its advantage is the "
                       "whole change of the picture; a gaze copy that over-reacts reads lower"),
    "body_action_sensitivity": ("how far the body-position prediction moves when the order is swapped for "
                                "another body's, as a share of the whole predicted change",
                                "0% = the order is ignored; 100% = the order moves it as far as the whole "
                                "predicted change"),
    "proprioception_variation": ("how much each body's body-position token changes over 50 s (mean over bodies)",
                                 "0% = the same token every moment; 100% = an unrelated token every moment; "
                                 "a body under 20% is deaf in this sense"),
    "vestibular_variation": ("how much each body's balance token changes over 50 s (mean over bodies)",
                             "0% = the same token every moment; 100% = an unrelated token every moment; "
                             "a body under 1.6% is deaf in this sense"),
    "hand_following": ("how much of each ordered finger turn (relative to the wrist) the fingers closed by the "
                       "end of the order's 250 ms", "0% = no closer than they began; 100% = arrived"),
    "observer_counts": ("capability rows whose occasions per body-step fell between the windows beyond the row's "
                        "own measured no-change scatter",
                        "0 = no behaviour became rarer; the count of rows tested is beside it"),
}
BETTER = {h: +1 for h in HEADLINES}
BETTER["observer_counts"] = -1


# =================================================================================================== paths
def saved_root(explicit: Optional[str] = None) -> Path:
    """Saved/mimo_bridge of this checkout, else of the main tree (a sparse worktree holds no Saved/: it is read
    by absolute path, never linked)."""
    if explicit:
        return Path(explicit)
    for root in (REPO, MAIN_REPO):
        p = root / "Saved" / "mimo_bridge"
        if (p / "experiments").is_dir():
            return p
    raise FileNotFoundError("no Saved/mimo_bridge/experiments under %s or %s" % (REPO, MAIN_REPO))


def checkpoint_step(ckpt) -> int:
    """The checkpoint's step: lineage.json 'step', else the folder's number."""
    p = Path(ckpt)
    try:
        step = json.loads((p / "lineage.json").read_text(encoding="utf-8")).get("step")
        if step is not None:
            return int(step)
    except (OSError, ValueError):
        pass
    m = re.search(r"ckpt_step_(\d+)", p.name)
    if not m:
        raise ValueError("not a checkpoint folder: %s" % p)
    return int(m.group(1))


def lineage_checkpoints(saved: Path, lineage: str = LINEAGE, roots: Optional[Sequence[Path]] = None) -> List[Tuple[int, Path]]:
    """Every complete checkpoint of the lineage on C: (the run folders) and D: (the archive), one per step, the
    C: copy first."""
    roots = list(roots) if roots is not None else [saved / "experiments", ARCHIVE]
    seen, out = set(), []
    for root in roots:
        if not Path(root).is_dir():
            continue
        for run in sorted(Path(root).iterdir()):
            if not run.name.startswith(lineage) or not (run / "checkpoints").is_dir():
                continue
            for ck in (run / "checkpoints").iterdir():
                m = re.fullmatch(r"ckpt_step_(\d+)", ck.name)
                if not m or int(m.group(1)) in seen:
                    continue
                if not all((ck / f).exists() for f in ("model.safetensors", "episode_state.json")):
                    continue
                seen.add(int(m.group(1)))
                out.append((int(m.group(1)), ck))
    return sorted(out)


def parse_side(text: str, window: int, saved: Path, switch_step: Optional[int] = None) -> dict:
    """--before / --after: a checkpoint folder or a bare step (the lineage's checkpoint of that step; its window
    is the `window` steps ending at its step), or a step range 'A-B' (the window (A, B]; its checkpoint is the
    lineage's newest one inside it, when there is one). switch_step (the after side): the window starts no earlier
    than the switch, and `newest` is how many of the checkpoint's ring moments were lived after it (one moment per
    model step: LemLite.step adds one to the replay ring per call); None = the whole ring."""
    m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", str(text))
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if hi <= lo:
            raise ValueError("a step range must end after it starts: %s" % text)
        inside = [(s, p) for s, p in lineage_checkpoints(saved) if lo < s <= hi]
        ck = inside[-1][1] if inside else None
        side = dict(kind="range", lo=lo, hi=hi, checkpoint=None if ck is None else str(ck),
                    checkpoint_step=None if ck is None else inside[-1][0])
    else:
        if re.fullmatch(r"\s*\d+\s*", str(text)):
            # a bare step: the lineage's checkpoint of that step, on C: or on the D: archive
            found = [p for s, p in lineage_checkpoints(saved) if s == int(text)]
            if not found:
                raise FileNotFoundError("the lineage has no checkpoint of step %s" % text)
            text = str(found[0])
        p = Path(text)
        if not (p / "model.safetensors").exists():
            raise FileNotFoundError("no checkpoint at %s" % text)
        step = checkpoint_step(p)
        side = dict(kind="checkpoint", lo=step - int(window), hi=step, checkpoint=str(p), checkpoint_step=step)
    return after_switch(side, switch_step)


def after_switch(side: dict, switch_step: Optional[int]) -> dict:
    """The side restricted to what came after the switch: its window's start moved up to the switch, and `newest`
    = the ring moments lived after it (the checkpoint's step minus the switch). Refuses a checkpoint from before
    the switch."""
    side = dict(side, switch_step=None if switch_step is None else int(switch_step), newest=None)
    if switch_step is None:
        return side
    s = int(switch_step)
    if side["hi"] <= s:
        raise ValueError("the after side ends at %s, not after the switch at %s" % (side["hi"], s))
    side["lo"] = max(int(side["lo"]), s)
    if side.get("checkpoint_step") is not None:
        if int(side["checkpoint_step"]) <= s:
            raise ValueError("the after side's checkpoint %s is not after the switch at %s" % (side["checkpoint_step"], s))
        side["newest"] = int(side["checkpoint_step"]) - s
    return side


# ============================================================================================ stats records
def _g(d, *path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _num(v) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _first_record(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    return json.loads(line)
                except ValueError:
                    continue
    except OSError:
        return None
    return None


def _last_step(path: Path) -> Optional[int]:
    """The step of the file's last mimo record (read from its last megabyte)."""
    last = None
    try:
        with path.open("rb") as f:
            size = path.stat().st_size
            f.seek(max(0, size - 2 ** 20))
            for line in f.read().decode("utf-8", "replace").splitlines():
                if '"mimo"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("type") == "mimo" and r.get("step") is not None:
                    last = int(r["step"])
    except OSError:
        return None
    return last


def stats_runs(saved: Path, lineage: str = LINEAGE) -> List[dict]:
    """Every run of the lineage with a stats file: its launch time, launch step and last recorded step, and its
    CUT: the launch step of the first run launched after it (the supervisor resumes a crash from the last
    checkpoint into a new run folder, so the older run's later records are an abandoned future)."""
    runs = []
    for run in sorted((saved / "experiments").iterdir()):
        path = run / "replay_training_stats.jsonl"
        if not run.name.startswith(lineage) or not path.exists():
            continue
        first = _first_record(path) or {}
        t0 = _num(first.get("timestamp"))
        s0 = first.get("step")
        if first.get("type") != "launch":
            s0 = (int(s0) - 100) if s0 is not None else None
        runs.append(dict(run=str(run), path=str(path), start=t0 if t0 is not None else path.stat().st_mtime,
                         first_step=None if s0 is None else int(s0), last_step=_last_step(path)))
    runs.sort(key=lambda r: r["start"])
    for i, r in enumerate(runs):
        later = [x["first_step"] for x in runs[i + 1:] if x["first_step"] is not None]
        r["cut"] = min(later) if later else None
    return runs


def stats_records(saved: Path, lo: int, hi: int, lineage: str = LINEAGE) -> List[dict]:
    """The lineage's report records ("mimo") with lo < step <= hi, each run's abandoned future cut, in step
    order."""
    out = []
    for r in stats_runs(saved, lineage):
        if r["last_step"] is None or r["first_step"] is None or r["last_step"] <= lo or r["first_step"] >= hi:
            continue
        with open(r["path"], "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if '"mimo"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("type") != "mimo" or rec.get("step") is None:
                    continue
                s = int(rec["step"])
                if not (lo < s <= hi) or (r["cut"] is not None and s > r["cut"]):
                    continue
                rec["_run"] = Path(r["run"]).name
                out.append(rec)
    out.sort(key=lambda x: int(x["step"]))
    return out


def _ratio_share(num: float, den: float) -> Optional[float]:
    return None if den is None or den <= 0 or num is None else 1.0 - num / den


def records_readings(records: List[dict]) -> dict:
    """The pack's readings from production's own report records (all 8 bodies pooled; the live gauges are not
    kept per body, except the collapse watch's per-body field once the proprioception branch is deployed)."""
    out = dict(records=len(records), steps=[int(records[0]["step"]), int(records[-1]["step"])] if records else None)
    # 1 the live gaze copy: vision's next-moment error with the own eye command against another body's
    w = ws = n = 0.0
    for r in records:
        gc_ = _g(r, "efference", "gaze_copy") or {}
        a, b, e = _num(gc_.get("vision_err_with_copy")), _num(gc_.get("vision_err_copy_shuffled")), _num(gc_.get("events"))
        if a is None or b is None or not e:
            continue
        w += a * e
        ws += b * e
        n += e
    out["eye_view_share_live"] = _ratio_share(w, ws) if n else None
    out["eye_view_events"] = int(n)
    # the eye-aim forecast 150 ms ahead (the ability scheduler's gauge eye_forecast_share_150ms)
    err = dn = sh = 0.0
    k = 0
    for r in records:
        e = _g(r, "efference") or {}
        a, b, c = (e.get(x) for x in ("gaze_deg_err", "gaze_deg_do_nothing", "gaze_deg_shuffled"))
        if not (isinstance(a, list) and isinstance(b, list) and len(a) > 2 and len(b) > 2):
            continue
        if _num(a[2]) is None or _num(b[2]) is None:
            continue
        err += a[2]
        dn += b[2]
        sh += (c[2] if isinstance(c, list) and len(c) > 2 and _num(c[2]) is not None else float("nan"))
        k += 1
    out["eye_aim_150ms_share_live"] = _ratio_share(err, dn) if k else None
    out["eye_aim_150ms_vs_shuffled_live"] = _ratio_share(err, sh) if k and math.isfinite(sh) else None
    # vision's next-picture error against the unchanged picture's (the prediction audit, all bodies)
    pm = pp = 0.0
    for r in records:
        v = _g(r, "prediction_audit", "modalities", "vision") or {}
        t_, a, b = _num(v.get("transitions")), _num(v.get("prediction_mse")), _num(v.get("persistence_mse"))
        if t_ and a is not None and b is not None:
            pm += a * t_
            pp += b * t_
    out["vision_error_vs_unchanged_live"] = (pm / pp) if pp > 0 else None
    # 2 the prediction audit's body-position sensitivity and its accuracy against another body's order
    sens, m_ = [], 0.0
    sh_, mt_ = 0.0, 0.0
    for r in records:
        p = _g(r, "prediction_audit", "modalities", "proprioception") or {}
        s = _num(p.get("sensitivity"))
        if s is not None:
            sens.append(s)
        sn, a, b = _num(p.get("shuffled_transitions")), _num(p.get("matched_prediction_mse")), _num(p.get("shuffled_action_mse"))
        if sn and a is not None and b is not None:
            mt_ += a * sn
            sh_ += b * sn
    out["body_sensitivity_live"] = float(np.mean(sens)) if sens else None
    out["body_improvement_over_shuffled_live"] = _ratio_share(mt_, sh_) if sh_ > 0 else None
    ea = eb = 0.0
    for r in records:
        e = _g(r, "efference") or {}
        a, b = _num(e.get("body_cm_err_latch")), _num(e.get("body_cm_shuffled_latch"))
        if a is not None and b is not None:
            ea += a
            eb += b
    out["efference_body_vs_shuffled_live"] = _ratio_share(ea, eb) if eb > 0 else None
    # 3-4 the collapse watch: body 0 always; every body once the per-body field is recorded
    for sense in ("proprioception", "vestibular"):
        v0 = [_num(_g(r, "token_temporal_std", sense)) for r in records]
        v0 = [x for x in v0 if x is not None]
        out[sense + "_spread_body0_live"] = float(np.mean(v0)) if v0 else None
        per = [(_g(r, "token_temporal_std_bodies") or {}).get(sense) for r in records]
        per = [np.asarray(x, dtype=float) for x in per if isinstance(x, list) and x]
        if per and len({len(x) for x in per}) == 1:
            out[sense + "_spread_bodies_live"] = [float(v) for v in np.mean(np.stack(per), 0)]
    # 5 the hands' joint error from their joint goals, and every part's following of its order
    je = jn = 0.0
    for r in records:
        h = r.get("hand_control") or {}
        a, n_ = _num(h.get("joint_error_deg")), _num(h.get("body_observations"))
        if a is not None and n_:
            je += a * n_
            jn += n_
    out["hand_joint_error_deg_live"] = (je / jn) if jn > 0 else None
    po = cl = ch = 0.0
    for r in records:
        for row in ((_g(r, "synergy", "following_by_author") or {}).values()):
            if isinstance(row, dict) and _num(row.get("part_orders")):
                po += row["part_orders"]
                cl += float(row.get("closer") or 0)
                ch += float(row.get("chance_count") or 0)
    out["following_closer_per_100_live"] = (100.0 * cl / po) if po else None
    out["following_chance_per_100_live"] = (100.0 * ch / po) if po else None
    return out


# ================================================================================================ observer
def _observer():
    from mimo_bridge.tools.observer import catalog as CAT, daemon as DM, store as ST, tracker as TR
    return CAT, DM, ST, TR


def load_store(folder: Path, quality: dict, tries: int = 5):
    """The lineage store as its last complete save names it, read-only. The digester writes a new generation's
    files, then its meta, then removes the old files; a read that meets a removed file reads the meta again."""
    CAT, DM, ST, TR = _observer()
    last = None
    for _ in range(int(tries)):
        try:
            meta = json.loads((folder / "store_meta.json").read_text(encoding="utf-8"))
            if meta.get("version") != ST.STORE_VERSION:
                raise ValueError("store version %s, the observer reads %s" % (meta.get("version"), ST.STORE_VERSION))
            s = ST.LineageStore(None, quality)
            if meta.get("quality_hash") != s.qhash:
                raise ValueError("the store was built with another catalog's measures")
            s.strings.items[:] = meta["strings"]
            s.strings.index = {x: i for i, x in enumerate(s.strings.items)}
            s.dets.load(str(folder / ("store_dets_%d.npz" % int(meta["generation"]))))
            s.offsets = meta["offsets"]
            return s, meta
        except (OSError, KeyError) as error:
            last = error
            time.sleep(1.0)
    raise RuntimeError("the observer's lineage store could not be read: %s" % last)


def observer_state(saved: Path, lineage: str = LINEAGE) -> dict:
    """Everything the window counts need, read once: the store, the live flag of every detection (the
    digester's own cuts), each run's exposure bins and cuts, the babble baseline and the catalog."""
    CAT, DM, ST, TR = _observer()
    cat = CAT.load()
    store, meta = load_store(saved / "observer" / lineage / "store", DM.quality_map(cat))
    states, exposures, header = {}, {}, None
    for run in sorted((saved / "experiments").iterdir()):
        a = run / "observer" / "analysis"
        if not run.name.startswith(lineage) or not (a / "state.json").exists():
            continue
        try:
            states[run.name] = json.loads((a / "state.json").read_text(encoding="utf-8"))
            exposures[run.name] = json.loads((a / "exposure.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        hs = sorted((run / "observer" / "world").glob("b*/header_*.json"), key=lambda p: p.stat().st_mtime)
        if hs:
            try:
                header = json.loads(hs[-1].read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
    cuts_by_run = {name: st.get("cuts", {}) for name, st in states.items()}
    line = DM.lineage_cuts(states)
    live = DM.live_rows(store, cuts_by_run, line)
    signature = DM.world_signature(header) if header else None
    base_path = saved / "observer" / "baselines" / str(signature) / "random_orders.json"
    base = {}
    if signature and base_path.exists():
        base = json.loads(base_path.read_text(encoding="utf-8"))
        if base.get("detector_version") not in (None, DM.DETECTOR_VERSION):
            base = {}
    return dict(cat=cat, store=store, store_generation=meta.get("generation"), live=live, states=states,
                exposures=exposures, cuts_by_run=cuts_by_run, line=line, signature=signature, baseline=base,
                baseline_path=str(base_path), detector_version=DM.DETECTOR_VERSION, rules_version=TR.RULES_VERSION)


def window_exposure(obs: dict, lo: int, hi: int, exclude: Iterable[int] = ()) -> Dict[int, float]:
    """Counted body-steps per body in [lo, hi) (bins of 100 steps; the tracker's exclusions: probe, baseline
    and rest steps; a bin in an abandoned future is not counted)."""
    CAT, DM, ST, TR = _observer()
    ex = set(int(b) for b in exclude)
    out: Dict[int, float] = {}
    for run, bodies in obs["exposures"].items():
        cuts = obs["cuts_by_run"].get(run) or {}
        line_cut = obs["line"].get(run)
        for body, pids in bodies.items():
            if int(body) in ex:
                continue
            for pid, bins in pids.items():
                cut = (cuts.get(body) or {}).get(pid)
                for b, row in bins.items():
                    start = int(b) * BIN
                    if not (lo <= start < hi):
                        continue
                    if cut is not None and start >= int(cut):
                        continue
                    if line_cut is not None and start >= int(line_cut):
                        continue
                    counted = row["steps"] - sum(v for k, v in row["by"].items() if k in TR.EXCLUDED)
                    out[int(body)] = out.get(int(body), 0.0) + float(counted)
    return out


def observer_window(obs: dict, lo: int, hi: int, exclude: Iterable[int] = ()) -> dict:
    """Per capability row: counted acts and separate occasions in [lo, hi), counted body-steps, occasions per
    1,000 body-steps, babble's over its own body-steps and the ratio of the two. lo and hi are rounded down to
    the exposure bins (100 steps) so the acts and the body-steps cover the same steps."""
    CAT, DM, ST, TR = _observer()
    lo, hi = (int(lo) // BIN) * BIN, (int(hi) // BIN) * BIN
    ex = set(int(b) for b in exclude)
    t = obs["store"].dets
    counted = TR.counted_mask(t, obs["live"])
    body = t["body"].astype(np.int64)
    sel = counted & (t["step0"] >= lo) & (t["step0"] < hi) & ~np.isin(body, list(ex))
    expo = window_exposure(obs, lo, hi, ex)
    E = float(sum(expo.values()))
    caps = t["cap"]
    base_caps = (obs["baseline"] or {}).get("caps") or {}
    rows = {}
    for cap in obs["cat"]["capabilities"]:
        if not cap.get("observable", True):
            continue
        cid = cap["id"]
        cid_i = t.s.index.get(cid)
        m = sel & (caps == cid_i) if cid_i is not None else np.zeros(len(t), dtype=bool)
        idx = np.nonzero(m)[0]
        occ = TR.occasions(t["step0"][idx], body[idx], t["sub"][idx]) if idx.size else 0
        b = base_caps.get(cid) or {}
        b_occ = b.get("occasions", b.get("count"))
        b_steps = float(b.get("steps") or (obs["baseline"] or {}).get("steps") or 0)
        rate = (1000.0 * occ / E) if E > 0 else None
        b_rate = (1000.0 * float(b_occ) / b_steps) if (b_occ is not None and b_steps > 0) else None
        rows[cid] = dict(name=cap["name"], acts=int(idx.size), occasions=int(occ), per_1000=rate,
                         babble_per_1000=b_rate, babble_occasions=None if b_occ is None else int(b_occ),
                         babble_steps=b_steps or None,
                         expected_from_babble=(float(b_occ) * E / b_steps) if (b_occ is not None and b_steps > 0) else None,
                         vs_babble=(rate / b_rate) if (rate is not None and b_rate) else None)
    return dict(lo=lo, hi=hi, body_steps=E, body_steps_per_body={str(k): v for k, v in sorted(expo.items())},
                excluded_bodies=sorted(ex), rows=rows)


def log_rate(occasions: int, body_steps: float) -> Optional[float]:
    """ln of a row's occasions per body-step, with half an occasion added so an empty window has a value."""
    return None if not body_steps > 0 else math.log((float(occasions) + 0.5) / float(body_steps))


# the observer's share of the pack's false-alarm rate: the Benjamini-Hochberg line across its rows at this level
# keeps the chance of any row reading "fell" between two no-change windows at this level when no row changed
OBSERVER_Q = FALSE_ALARM_PER_READING


def row_scatter_test(k1: int, e1: float, k2: int, e2: float, scatter: dict) -> Tuple[float, float, float]:
    """(p that the rate fell, p that it rose, the standardised change) for one row between two windows, judged
    against the row's MEASURED no-change scatter. The change is ln((k2 + 0.5) / e2) - ln((k1 + 0.5) / e1); its
    no-change variance is counting chance at the two windows (1 / (k + 0.5) for each, the delta method, with each
    window's count as the pooled rate predicts it, so the outcome itself does not widen the line) plus twice the
    row's extra variance (scatter["extra_var"]: the variance of its ln rate over the no-change windows beyond the
    counting chance of those windows' own counts). Read on Student's t with the no-change windows' degrees of
    freedom (windows - 1), because the extra variance is estimated from those few windows."""
    from scipy import stats
    k1, k2, e1, e2 = float(k1), float(k2), float(e1), float(e2)
    d = math.log((k2 + 0.5) / e2) - math.log((k1 + 0.5) / e1)
    k1h, k2h = (k1 + k2) * e1 / (e1 + e2), (k1 + k2) * e2 / (e1 + e2)
    var = 2.0 * float(scatter["extra_var"]) + 1.0 / (k1h + 0.5) + 1.0 / (k2h + 0.5)
    z = d / math.sqrt(var)
    dof = max(1, int(scatter["windows"]) - 1)
    return float(stats.t.cdf(z, dof)), float(stats.t.sf(z, dof)), z


def observer_change(before: dict, after: dict, q: float = OBSERVER_Q, row_lines: Optional[dict] = None) -> dict:
    """Per row, whether its occasions per body-step FELL (or rose) from `before` to `after`.
    With `row_lines` (observer_row_lines of the no-change windows; the noise file's observer_rows): each row is
    judged against its own measured no-change scatter (row_scatter_test), the Benjamini-Hochberg line across the
    rows judged at q; a row without a measured scatter is not judged. Without them (the record's plain test only):
    the exact conditional test of two Poisson counts (tracker.two_sample_above), which assumes counting chance is
    the whole scatter - MEASURED 2026-09-25: it called up to 6 of 23 rows fallen between two no-change windows,
    and the first rule built on it (beyond chance and beyond the row's largest no-change change) flagged a fallen
    row in 16 of 40 no-change comparisons."""
    CAT, DM, ST, TR = _observer()
    E1, E2 = float(before["body_steps"]), float(after["body_steps"])
    rows, drop_p, rise_p = {}, {}, {}
    for cid, a in before["rows"].items():
        b = after["rows"].get(cid)
        if b is None:
            continue
        k1, k2 = int(a["occasions"]), int(b["occasions"])
        l1, l2 = log_rate(k1, E1), log_rate(k2, E2)
        sc = (row_lines or {}).get(cid) or {}
        row = dict(name=a["name"], before_per_1000=a["per_1000"], after_per_1000=b["per_1000"],
                   babble_per_1000=a["babble_per_1000"], before_vs_babble=a["vs_babble"], after_vs_babble=b["vs_babble"],
                   before_occasions=k1, after_occasions=k2,
                   ln_change=(None if l1 is None or l2 is None else l2 - l1),
                   row_extra_sd_ln=(None if sc.get("extra_var") is None else math.sqrt(sc["extra_var"])))
        if E1 > 0 and E2 > 0 and (k1 + k2) >= TR.MIN_COUNT:
            if row_lines is None:
                drop_p[cid] = TR.two_sample_above(k1, E1, k2, E2)
                rise_p[cid] = TR.two_sample_above(k2, E2, k1, E1)
            elif sc.get("extra_var") is not None and int(sc.get("windows") or 0) >= 2:
                drop_p[cid], rise_p[cid], row["z"] = row_scatter_test(k1, E1, k2, E2, sc)
            if cid in drop_p:
                row.update(drop_p=drop_p[cid], rise_p=rise_p[cid])
        rows[cid] = row
    drop_line, rise_line = TR.fdr_line(list(drop_p.values()), q), TR.fdr_line(list(rise_p.values()), q)
    dropped = sorted(c for c, p in drop_p.items() if drop_line > 0 and p <= drop_line)
    rose = sorted(c for c, p in rise_p.items() if rise_line > 0 and p <= rise_line)
    for c in rows:
        rows[c]["fell"] = c in dropped
        rows[c]["rose"] = c in rose
    return dict(rows=rows, tested=len(drop_p), fell=dropped, rose=rose, drop_line=drop_line, rise_line=rise_line,
                fdr=q, body_steps=[E1, E2], row_lines_used=row_lines is not None,
                rule=("each row against its own measured no-change scatter" if row_lines is not None
                      else "counting chance only (the plain test; for the record)"))


def observer_row_lines(windows: List[dict]) -> Dict[str, dict]:
    """Each row's measured no-change scatter over the windows given: the variance of its ln(occasions per body-step)
    over the windows, the part of it counting chance explains at those windows' counts (the mean of 1 / (k + 0.5)),
    and the rest, the row's extra variance (0 when counting chance explains it all); with the occasions of every
    window and the largest change between two of them (line_ln, the first rule's line, for the record)."""
    out = {}
    for cid in (windows[0]["rows"] if windows else {}):
        ks = [int(w["rows"][cid]["occasions"]) for w in windows if w["body_steps"] > 0]
        vals = [log_rate(w["rows"][cid]["occasions"], w["body_steps"]) for w in windows]
        vals = [v for v in vals if v is not None]
        d = [abs(a - b) for i, a in enumerate(vals) for b in vals[i + 1:]]
        var = float(np.var(vals, ddof=1)) if len(vals) >= 2 else None
        chance = float(np.mean([1.0 / (k + 0.5) for k in ks])) if ks else None
        out[cid] = dict(line_ln=(max(d) if d else None), occasions=[w["rows"][cid]["occasions"] for w in windows],
                        windows=len(vals), var_ln=var, counting_var_ln=chance,
                        extra_var=(None if var is None else max(0.0, var - chance)))
    return out


def observer_leave_one_out(windows: List[dict], q: float = OBSERVER_Q) -> dict:
    """The rule measured on no-change windows: each window held out in turn, the rows' scatter measured on the
    others, the held-out window judged against each of the others in both directions. Returns every comparison's
    count of fallen rows, the share of comparisons with at least one (the rule's measured false-alarm rate, against
    the q it declares) and the count LINE: the most rows that fell in any of these no-change comparisons (a compare
    reads REGRESSED only above it)."""
    comps = []
    for i, held in enumerate(windows):
        rest = [w for j, w in enumerate(windows) if j != i]
        lines = observer_row_lines(rest)
        for j, w in enumerate(windows):
            if j == i:
                continue
            for a, b in ((w, held), (held, w)):
                ch = observer_change(a, b, q, row_lines=lines)
                comps.append(dict(before=[a["lo"], a["hi"]], after=[b["lo"], b["hi"]], fell=ch["fell"], rose=ch["rose"]))
    n = len(comps)
    fell = [len(c["fell"]) for c in comps]
    return dict(comparisons=n, with_a_fallen_row=int(sum(1 for x in fell if x)),
                share_with_a_fallen_row=(sum(1 for x in fell if x) / n) if n else None, declared_rate=q,
                line=(max(fell) if fell else None), fallen_counts=fell, pairs=comps)


# ================================================================================================ the ring
def load_ring(ckpt) -> dict:
    """The checkpoint's replay ring and the few other things the pack reads, from its episode companion
    (continuation_state.load: the receipt's hash checked, a moved companion found through the move records).
    The caregiver's recording tapes are replaced by plain lists first (gof_probe_harness.sanitize_episode):
    nothing here can write into a live recording folder. Everything else is dropped before returning."""
    from mimo_bridge import continuation_state
    from mimo_bridge.tools.gof_probe_harness import sanitize_episode
    t0 = time.time()
    state = continuation_state.load(str(ckpt))
    if state is None:
        raise ValueError("%s has no episode companion" % ckpt)
    sanitize_episode(state)
    rep = dict(state["learner"]["objects"]["replay"])
    keep = {k: rep[k] for k in ("mods", "tokens", "achieved", "executed", "has_executed", "valid", "episode",
                                "ptr", "count", "size", "n_envs", "memory_states", "self_states", "action_contract",
                                "rom")
            if k in rep}
    keep["companion_step"] = int(state["trainer"]["step"])
    keep["load_seconds"] = round(time.time() - t0, 1)
    del state, rep
    gc.collect()
    return keep


def chronological(ring: dict):
    """Ring slot indices, oldest first."""
    import torch
    n, ptr, count = int(ring["size"]), int(ring["ptr"]), int(ring["count"])
    if count < n:
        return torch.arange(count)
    return (torch.arange(n) + ptr % n) % n


def _first_readable(ring: dict, newest: Optional[int]) -> int:
    """The first chronological index a reading may use: 0 for the whole ring, else the start of the newest
    `newest` moments (those lived after the switch)."""
    n = int(chronological(ring).numel())
    return 0 if newest is None else max(0, n - int(newest))


def token_spreads(ring: dict, senses=("proprioception", "vestibular"), window: int = SPREAD_WINDOW,
                  every: int = SPREAD_EVERY, newest: Optional[int] = None) -> dict:
    """Per sense, per body: the collapse watch's quantity (std over time of the raw token, 200 samples every 5th
    moment over 1,000 moments, mean over the numbers) on each of the ring's whole 50-s windows, newest last, each
    with its end (moments before the newest), and their mean. The windows never cross an episode reset (a window
    that does is left out); with `newest`, only windows wholly inside the newest `newest` moments."""
    order = chronological(ring)
    mods = list(ring["mods"])
    ep = ring["episode"][:, order]
    out = {}
    N = int(order.numel())
    first = _first_readable(ring, newest)
    ends = [e for e in range(N - 1, window - 2, -window) if e - window + 1 >= first][::-1]
    for sense in senses:
        if sense not in mods:
            out[sense] = None
            continue
        k = mods.index(sense)
        tok = ring["tokens"][:, order, k].float()
        B = tok.shape[0]
        per_body = []
        for b in range(B):
            vals, at = [], []
            for e in ends:
                idx = list(range(e - window + 1, e + 1, every))
                if not bool((ep[b, idx] == ep[b, idx[-1]]).all()):
                    continue
                vals.append(float(tok[b, idx].std(0).mean()))
                at.append(N - 1 - e)
            per_body.append(dict(windows=vals, ends_back=at, mean=float(np.mean(vals)) if vals else None))
        out[sense] = per_body
    return out


def _quat_conj_mul(a, b):
    """conj(a) * b for x y z w quaternions (torch, broadcast): b expressed in a's frame."""
    import torch
    ax, ay, az, aw = (-a[..., 0], -a[..., 1], -a[..., 2], a[..., 3])
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + bw * ax + ay * bz - az * by,
                        aw * by + bw * ay + az * bx - ax * bz,
                        aw * bz + bw * az + ax * by - ay * bx,
                        aw * bw - ax * bx - ay * by - az * bz], -1)


def tenures(executed, has_executed, episode, gaze_row: int):
    """(start, end) moment pairs of every WHOLE order of one body: an order is the run of moments whose executed
    order stays the same in every row but the eyes' (the trainer rewrites the eye row every moment with the eye
    command actually sent). It counts only when a new order begins at both of its ends inside one episode, with
    every moment labelled: the ring's first order began before the ring did, its newest one has not ended, and
    production's own following gauge grades full executed tenures only. The order carries the body from `start`
    to `end + 1`."""
    import torch
    n = executed.shape[0]
    if n < 2:
        return []
    limbs = torch.cat([executed[:, :gaze_row], executed[:, gaze_row + 1:]], dim=1).float()
    has = torch.as_tensor(has_executed, dtype=torch.bool)
    ep = torch.as_tensor(episode)
    begins = torch.zeros(n, dtype=torch.bool)       # moment t is the first of a new order, known from t - 1
    begins[1:] = ((limbs[1:] != limbs[:-1]).reshape(n - 1, -1).any(-1) & (ep[1:] == ep[:-1]) & has[1:] & has[:-1])
    starts = [int(t) for t in torch.nonzero(begins).flatten()]
    out = []
    for t, nxt in zip(starts, starts[1:]):
        if bool(has[t:nxt].all()) and bool((ep[t:nxt + 1] == ep[t]).all()):
            out.append((t, nxt - 1))
    return out


HAND_KEYS = ("part_orders", "closer", "chance", "asked_deg", "left_deg", "moved_deg")


def hand_following(ring: dict, handle_names: Sequence[str], newest: Optional[int] = None,
                   batches: int = BATCHES) -> List[dict]:
    """Per body, over every order in the ring (with `newest`: every order begun inside the newest `newest`
    moments): the fingers' following of their ordered turn relative to the wrist (the hand goal's own frame), and
    the wrists' following of theirs in the world, with the random-axis chance line of
    lem.model.motor_synergy.following_counts. Sums, so bodies pool by adding; the same sums per batch (the readable
    moments cut into `batches` equal spans, an order in the span where it began) for the sampling sd."""
    import torch
    from lem.model.motor_synergy import following_counts
    names = list(handle_names)
    gi = names.index("gaze")
    order = chronological(ring)
    ex = ring["executed"][:, order].float()           # [B, N, H, 9] cm and x y z w
    ach = ring["achieved"][:, order].float()          # [B, N, H, 10] position / 200 and x y z w
    hx = ring["has_executed"][:, order]
    ep = ring["episode"][:, order]
    B, N = ex.shape[0], ex.shape[1]
    first = _first_readable(ring, newest)
    span = max(1, N - first)
    sides = [(names.index("wrist_" + s), [names.index(d + "_" + s) for d in ("thumb", "index", "middle", "ring", "pinky")])
             for s in ("L", "R")]
    fingers = [i for _, f in sides for i in f]
    wrists = [w for w, _ in sides]
    out = []
    for b in range(B):
        pairs = [p for p in tenures(ex[b], hx[b], ep[b], gi) if p[0] >= first]
        acc = {k: dict(part_orders=0, closer=0, chance=0.0, asked_deg=0.0, left_deg=0.0, moved_deg=0.0)
               for k in ("fingers", "wrists")}
        per = {k: [dict((x, 0.0) for x in HAND_KEYS) for _ in range(int(batches))] for k in ("fingers", "wrists")}
        if pairs:
            s = torch.tensor([p[0] for p in pairs])
            e = torch.tensor([p[1] + 1 for p in pairs])
            which = ((s - first) * int(batches) // span).clamp(0, int(batches) - 1)
            o, a0, a1 = ex[b, s], ach[b, s], ach[b, e]
            # the fingers relative to their wrist: conj(wrist) * finger, for the order and both poses
            ro, r0, r1 = o.clone(), a0.clone(), a1.clone()
            for w, fs in sides:
                for f in fs:
                    ro[:, f, 3:7] = _quat_conj_mul(o[:, w, 3:7], o[:, f, 3:7])
                    r0[:, f, 3:7] = _quat_conj_mul(a0[:, w, 3:7], a0[:, f, 3:7])
                    r1[:, f, 3:7] = _quat_conj_mul(a1[:, w, 3:7], a1[:, f, 3:7])
            for key, (oo, x0, x1, idx) in (("fingers", (ro, r0, r1, fingers)), ("wrists", (o, a0, a1, wrists))):
                c = following_counts(oo, x0, x1, torch.tensor(idx))
                acc[key].update(part_orders=int(c["closer"].numel()), closer=int(c["closer"].sum()),
                                chance=float(c["chance"].sum()), asked_deg=float(c["ordered_deg"].sum()),
                                left_deg=float(c["remaining_deg"].sum()), moved_deg=float(c["moved_deg"].sum()))
                # per batch: the same function on the orders begun in the batch's span (sums add up to the whole)
                for j in range(int(batches)):
                    sel = which == j
                    if not bool(sel.any()):
                        continue
                    cj = following_counts(oo[sel], x0[sel], x1[sel], torch.tensor(idx))
                    per[key][j].update(part_orders=float(cj["closer"].numel()), closer=float(cj["closer"].sum()),
                                       chance=float(cj["chance"].sum()), asked_deg=float(cj["ordered_deg"].sum()),
                                       left_deg=float(cj["remaining_deg"].sum()), moved_deg=float(cj["moved_deg"].sum()))
        out.append(dict(orders=len(pairs), batches=per, **acc))
    return out


# ============================================================================================ the model part
def _replay_forward(model, w: dict, step_order):
    """The replay lesson's construction of one batch of windows (LemLite.replay_update, without the loss and
    the optimizer; the same as audit_visual_regression_2026_09_24.replay_inputs): the context from the stored
    raw tokens plus the modality embedding, the motor token of the step-from moment and of every labelled
    context moment rebuilt from [achieved pose, executed order], the persistent memory's bias from the stored
    memory state. `step_order` is the executed order of the step-from moment (the real one or a swapped one).
    Returns the latent, the heads' predictions and the current and next tokens."""
    inp = _replay_inputs(model, w, step_order)
    bias, cur = inp["bias"], inp["cur"]
    _, last, _ = model.forward_model(cur, inp["ctx"], **({} if bias is None else {"memory_bias": bias}))
    preds = model.prediction_heads(last, current={m: v.squeeze(1) for m, v in cur.items()}, who="regression_pack")
    return dict(last=last, preds=preds, cur={m: v.squeeze(1) for m, v in cur.items()}, nxt=inp["nxt"], lab=inp["lab"])


def _replay_inputs(model, w: dict, step_order):
    """The inputs of _replay_forward's forward-model call: the current tokens `cur` ({sense: [R, 1, d]}), the
    context `ctx` ([R, (L - 1) x n_mod, d]), the bias the forward model adds after its final LayerNorm (`bias`:
    the memory's part `memory_bias` plus the self perception's part `self_bias`; None when the build has
    neither), the next tokens `nxt` and the step-from moment's label `lab`. Split out of _replay_forward on
    2026-09-26 so the memory deploy's readings can change the memory's part of the bias alone; _replay_forward's
    result is unchanged."""
    import torch
    from mimo_bridge.lem_model import MOTOR_ACHIEVED_F, command_for_encoder
    L, d = int(model.ctx_len), int(model.d_model)
    mods = list(w["mods"])
    toks, ach = w["tokens"], w["achieved"]
    R = toks.shape[0]
    fm = getattr(model.forward_model, "_orig_mod", model.forward_model)
    fm._refresh_embedding_cache()
    emb = fm._mod_emb_expanded
    ctx = toks[:, :L - 1] + emb.to(toks.dtype).unsqueeze(1)
    cur = {m: toks[:, L - 1, k].unsqueeze(1) for k, m in enumerate(mods)}
    nxt = {m: toks[:, L, k] for k, m in enumerate(mods)}

    def enc(a9, act):
        x = torch.cat([a9[..., :MOTOR_ACHIEVED_F], command_for_encoder(act)], dim=-1)
        flat = x.reshape(-1, x.shape[-2], x.shape[-1])
        return model.encoder_registry.encode({"motor": flat}, {"motor": "real"})["motor"].reshape(-1, d)

    lab = w["has_executed"][:, L - 1]
    mi = mods.index("motor")
    cur["motor"] = torch.where(lab.view(R, 1, 1), enc(ach[:, L - 1], step_order).reshape(R, 1, d), cur["motor"])
    ok = w["has_executed"][:, :L - 1]
    reb = enc(ach[:, :L - 1], w["executed"][:, :L - 1]).reshape(R, L - 1, d) + emb.to(toks.dtype)[:, mi].unsqueeze(1)
    ctx[:, :, mi] = torch.where(ok.unsqueeze(-1), reb, ctx[:, :, mi])
    mb = sb = None
    if getattr(model, "persistent_memory", None) is not None:
        if w.get("memory_state") is None:
            raise ValueError("the ring holds no memory state for its windows")
        mb = model.persistent_memory.bias(w["memory_state"])
    if getattr(model, "self_state_mode", None) == "perceive":
        if w.get("self_state") is None:
            raise ValueError("the ring holds no self perception for its windows")
        sb = model.self_state.residual(w["self_state"]).reshape(-1, model.n_mod, d)
    bias = sb if mb is None else (mb if sb is None else mb + sb)
    return dict(cur=cur, ctx=ctx.reshape(R, -1, d), bias=bias, memory_bias=mb, self_bias=sb, nxt=nxt, lab=lab)


def gather_windows(ring: dict, backs: Sequence[int], ctx_len: int) -> dict:
    """For each `back` (adds before the newest moment; the window ENDS there), every body's window of
    ctx_len + 1 consecutive moments, stacked slot-major: row = slot * B + body. Also each row's slot and body
    and whether the window stays inside one episode."""
    import torch
    need = int(ctx_len) + 1
    size, ptr, B = int(ring["size"]), int(ring["ptr"]), int(ring["tokens"].shape[0])
    rows_idx, bodies, slots = [], [], []
    for si, back in enumerate(backs):
        end = ptr - 1 - int(back)
        ix = torch.tensor([(end - need + 1 + j) % size for j in range(need)])
        for b in range(B):
            rows_idx.append(ix)
            bodies.append(b)
            slots.append(si)

    def g(t, dt=None):
        x = torch.stack([t[b, ix] for b, ix in zip(bodies, rows_idx)])
        return x if dt is None else x.to(dt)
    ms, ss = ring.get("memory_states"), ring.get("self_states")
    ep = g(ring["episode"])
    return dict(tokens=g(ring["tokens"], torch.float32), achieved=g(ring["achieved"], torch.float32),
                executed=g(ring["executed"], torch.float32), has_executed=g(ring["has_executed"]),
                valid=g(ring["valid"]), mods=list(ring["mods"]),
                memory_state=None if ms is None else torch.stack([ms[b, ix[-2]] for b, ix in zip(bodies, rows_idx)]).float(),
                self_state=None if ss is None else torch.stack([ss[b, ix[-2]] for b, ix in zip(bodies, rows_idx)]).float(),
                same_episode=(ep == ep[:, -1:]).all(dim=1), body=torch.tensor(bodies), slot=torch.tensor(slots))


def model_slots(ring: dict, ctx_len: int, n: int = MODEL_SLOTS, newest: Optional[int] = None) -> List[int]:
    """n window ends spread evenly over the ring (adds before the newest moment), newest first; a window never
    ends on the newest moment (its outgoing order is not back-filled yet). With `newest`, every moment of every
    window (its context included) lies inside the newest `newest` moments (those lived after the switch)."""
    count = int(ring["count"])
    need = int(ctx_len) + 1
    top = count - need - 1
    if newest is not None:
        top = min(top, int(newest) - need)
    if top < 1:
        return []
    n = min(int(n), top)
    return sorted({int(round(1 + i * (top - 1) / max(1, n - 1))) for i in range(n)})


def partners(B: int, exclude: Iterable[int] = ()) -> List[int]:
    """Each body's control body: the previous body, cyclically, among those NOT excluded (the live audit's
    order.roll(1) when nothing is excluded). An excluded body keeps b - 1 (its own rows are left out anyway)."""
    ex = set(int(b) for b in exclude)
    kept = [b for b in range(int(B)) if b not in ex]
    out = []
    for b in range(int(B)):
        if b in ex or len(kept) < 2:
            out.append((b - 1) % int(B))
        else:
            out.append(kept[kept.index(b) - 1])
    return out


def eyes_still_order(order, previous, gaze_idx: int):
    """The step-from order with its eye command replaced by the eye command of the order in force one moment
    earlier: the command the eyes were last sent, i.e. where they are, so no eye movement is commanded. The ring
    stores the eye command actually sent (the trainer's LemLite.gaze_to_handle, in the model's own eye-row
    convention), so copying the row keeps the convention. Only the command (x y z w, columns 3:7) is replaced."""
    out = order.clone()
    out[:, int(gaze_idx), 3:7] = previous[:, int(gaze_idx), 3:7]
    return out


MODEL_KEYS = ("eye_err_own", "eye_err_still", "eye_err_other", "eye_err_unchanged", "eye_rows",
              "body_shift_sq", "body_change_sq", "body_err_own", "body_err_other", "body_err_unchanged", "body_rows")


def _take(w: dict, idx) -> dict:
    """The rows `idx` of a gather_windows batch."""
    import torch
    R = int(w["tokens"].shape[0])
    return {k: (v[idx] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == R else v) for k, v in w.items()}


def _row_readings(model, w, real, own, other, lab_other, L: int, gaze_idx: int) -> Tuple[dict, object, object]:
    """Per row of a batch: the eye's errors with the own, the eyes-held and the partner's command through the gaze
    copy (the latent of the own order held), the unchanged picture's; the body's prediction moved by the partner's
    order; and which rows count for the eye and for the body."""
    from lem.model.efference import apply_gaze_copy
    from mimo_bridge.lem_model import command_for_encoder
    mods = list(w["mods"])
    vi, pi = mods.index("vision"), mods.index("proprioception")
    lab, lab_prev = w["has_executed"][:, L - 1], w["has_executed"][:, L - 2]
    still = eyes_still_order(own, w["executed"][:, L - 2], gaze_idx)
    ach = w["achieved"][:, L - 1]
    heads = model.prediction_heads
    v_own = apply_gaze_copy(heads, real["preds"], real["last"], command_for_encoder(own), ach, mask=lab)["vision"].float()
    v_still = apply_gaze_copy(heads, real["preds"], real["last"], command_for_encoder(still), ach,
                              mask=lab & lab_prev)["vision"].float()
    v_other = apply_gaze_copy(heads, real["preds"], real["last"], command_for_encoder(other), ach,
                              mask=lab_other)["vision"].float()
    swapped = _replay_forward(model, w, other)
    tv, cv = real["nxt"]["vision"].float(), real["cur"]["vision"].float()
    p_own = real["preds"]["proprioception"].float()
    p_other = swapped["preds"]["proprioception"].float()
    tp, cp = real["nxt"]["proprioception"].float(), real["cur"]["proprioception"].float()
    base = w["same_episode"] & lab & lab_other
    ok_v = base & lab_prev & w["valid"][:, L, vi] & w["valid"][:, L - 1, vi]
    ok_p = base & w["valid"][:, L, pi] & w["valid"][:, L - 1, pi]
    rows = dict(eye_err_own=(v_own - tv).square().mean(-1), eye_err_still=(v_still - tv).square().mean(-1),
                eye_err_other=(v_other - tv).square().mean(-1), eye_err_unchanged=(cv - tv).square().mean(-1),
                body_shift_sq=(p_own - p_other).square().mean(-1), body_change_sq=(p_own - cp).square().mean(-1),
                body_err_own=(p_own - tp).square().mean(-1), body_err_other=(p_other - tp).square().mean(-1),
                body_err_unchanged=(cp - tp).square().mean(-1))
    return rows, ok_v, ok_p


def model_readings(frozen, ring: dict, n_slots: int = MODEL_SLOTS, newest: Optional[int] = None,
                   exclusions: Sequence[Sequence[int]] = ((EXCLUDED_BODY,),)) -> dict:
    """Readings 1 and 2 per window end and per body on the checkpoint's own ring (sums, so bodies and window ends
    pool by adding):
      eye: vision's next-moment error with the gaze copy fed the body's own executed order, with the eyes held
        where they are (eyes_still_order: the headline's control) and with the partner's order at the same
        moment (beside it), the latent held the same; and the unchanged picture's error;
      body: the proprioception prediction with the body's own order and with the partner's (the prediction
        audit's control, lem_model.step): the squared move between them, the squared predicted change, and
        both errors.
    The partner is the previous body among those kept (partners): for each exclusion in `exclusions`, the bodies
    whose partner changes when it is left out (body 3 when body 2 is) are read again with their kept partner and
    stored under variants["2"], which pooled() uses for the readings without body 2. A row counts for a sense when
    its window stays inside one episode, it and its partner carry their step-from order (the eye also the order
    one moment earlier), and the sense was a real reading at the step-from and the target moment. With `newest`,
    only windows wholly inside the newest `newest` ring moments (lived after the switch)."""
    import torch
    model = frozen.model
    L = int(model.ctx_len)
    mods = list(ring["mods"])
    fm = getattr(model.forward_model, "_orig_mod", model.forward_model)
    if list(getattr(fm, "_active_modalities", mods)) != mods:
        raise ValueError("the ring's senses %s are not the forward model's %s" % (mods, list(fm._active_modalities)))
    B = int(ring["tokens"].shape[0])
    gi = list(model.handle_names).index("gaze")
    backs = model_slots(ring, L, n_slots, newest)
    S = len(backs)
    per_batch = max(1, MAX_BATCH // B)
    per_slot = {k: np.zeros((S, B)) for k in MODEL_KEYS}
    base_partner = partners(B)
    variants = {}
    for ex in exclusions:
        pv = partners(B, ex)
        changed = [b for b in range(B) if b not in set(ex) and pv[b] != base_partner[b]]
        variants[",".join(str(int(x)) for x in ex)] = dict(
            excluded=[int(x) for x in ex], partners=pv, bodies=changed,
            per_slot={k: np.zeros((S, len(changed))) for k in MODEL_KEYS})
    gcopy_active = bool(getattr(getattr(model, "efference", None), "centre_weight", 0.0) > 0.0)
    t0 = time.time()

    def add(target, cols, s0, w, rows, ok_v, ok_p, idx):
        for j, r in enumerate(idx):
            b = int(w["body"][r])
            if b not in cols:
                continue
            s_, c = s0 + int(w["slot"][r]), cols[b]
            if bool(ok_v[j]):
                for k in ("eye_err_own", "eye_err_still", "eye_err_other", "eye_err_unchanged"):
                    target[k][s_, c] += float(rows[k][j])
                target["eye_rows"][s_, c] += 1
            if bool(ok_p[j]):
                for k in ("body_shift_sq", "body_change_sq", "body_err_own", "body_err_other", "body_err_unchanged"):
                    target[k][s_, c] += float(rows[k][j])
                target["body_rows"][s_, c] += 1
    with torch.no_grad():
        for i in range(0, S, per_batch):
            w = gather_windows(ring, backs[i:i + per_batch], L)
            R = w["tokens"].shape[0]
            own = w["executed"][:, L - 1]
            lab = w["has_executed"][:, L - 1]
            real = _replay_forward(model, w, own)
            # rows are slot-major: the partner of row (slot, b) is row (slot, partner[b]) - AT THE SAME MOMENT
            src = w["slot"] * B + torch.as_tensor(base_partner)[w["body"]]
            rows, ok_v, ok_p = _row_readings(model, w, real, own, own[src], lab[src], L, gi)
            add(per_slot, {b: b for b in range(B)}, i, w, rows, ok_v, ok_p, list(range(R)))
            for key, var in variants.items():
                if not var["bodies"]:
                    continue
                idx = [r for r in range(R) if int(w["body"][r]) in var["bodies"]]
                sub = _take(w, torch.tensor(idx))
                src_v = w["slot"][idx] * B + torch.as_tensor(var["partners"])[w["body"][idx]]
                real_v = dict(last=real["last"][idx], preds={k: v[idx] for k, v in real["preds"].items()},
                              cur={k: v[idx] for k, v in real["cur"].items()}, nxt={k: v[idx] for k, v in real["nxt"].items()})
                rows_v, okv_v, okp_v = _row_readings(model, sub, real_v, own[idx], own[src_v], lab[src_v], L, gi)
                add(var["per_slot"], {b: c for c, b in enumerate(var["bodies"])}, i, w, rows_v, okv_v, okp_v, idx)
            del w, real
    per_body = [{k: float(per_slot[k][:, b].sum()) for k in MODEL_KEYS} for b in range(B)]
    for var in variants.values():
        var["per_slot"] = {k: v.tolist() for k, v in var["per_slot"].items()}
    return dict(per_body=per_body, per_slot={k: v.tolist() for k, v in per_slot.items()}, slots=backs,
                n_slots=int(n_slots), newest=None if newest is None else int(newest), variants=variants,
                seconds=round(time.time() - t0, 1), gaze_copy_active=gcopy_active,
                construction="the replay lesson's (LemLite.replay_update), fp32 on the CPU, no optimizer",
                partner="the previous body among those kept, at the same moment (partners)",
                eye_control="the eyes held: the step-from order with the eye command of one moment earlier")


# ======================================================================================== one checkpoint
def cache_name(ckpt: Path, n_slots: int, newest: Optional[int]) -> str:
    """The cache file of one checkpoint's readings: keyed on the model sample's size and on the moments read
    (the whole ring, or only the newest ones after a switch), so another --slots or switch never reads it."""
    return "%s_%s_slots%d_%s.json" % (ckpt.parent.parent.name, ckpt.name, int(n_slots),
                                      "whole_ring" if newest is None else "newest%d" % int(newest))


def checkpoint_readings(ckpt, *, model: bool = True, n_slots: int = MODEL_SLOTS, cache_dir: Optional[Path] = None,
                        internals: bool = False, newest: Optional[int] = None,
                        internal_slots: Optional[int] = None) -> dict:
    """Readings 1-5 of one checkpoint, per body (sums and per-body values, per window end and per batch for the
    sampling sd); cached by checkpoint, pack version, model sample size and moments read in cache_dir (a later
    compare reads the cache instead of the 600 MB companion). newest: read only the ring's newest `newest` moments
    (those lived after a switch). internals: also the memory deploy's readings (memory_deploy_readings.
    checkpoint_part: attention spread, unit liveness and the memory's worth, per body) on the same ring and model,
    at their own sample size (internal_slots, default memory_deploy_readings.SLOTS: their lines were measured on
    it), under "memory_deploy"; a cached checkpoint without them (or with another version, sample size or moments
    read) is read again."""
    from mimo_bridge.tools import memory_deploy_readings as MDR
    ckpt = Path(ckpt)
    step = checkpoint_step(ckpt)
    cache = None
    islots = MDR.SLOTS if internal_slots is None else int(internal_slots)

    def fits(got):
        m = got.get("model")
        return (got.get("version") == PACK_VERSION and got.get("newest") == newest
                and (not model or (m is not None and m.get("n_slots") == int(n_slots))))

    def has_internals(got):
        part = got.get("memory_deploy") or {}
        return (part.get("version") == MDR.READINGS_VERSION and part.get("n_slots") == islots
                and part.get("newest") == newest)
    if cache_dir is not None:
        cache = Path(cache_dir) / cache_name(ckpt, n_slots, newest)
        if cache.exists():
            try:
                got = json.loads(cache.read_text(encoding="utf-8"))
                if fits(got) and (has_internals(got) or not (internals and model)):
                    return got
            except (OSError, ValueError):
                pass
    t0 = time.time()
    ring = load_ring(ckpt)
    out = dict(version=PACK_VERSION, checkpoint=str(ckpt), step=step, companion_step=ring["companion_step"],
               ring_moments=int(ring["count"]), ring_ptr=int(ring["ptr"]), newest=newest,
               bodies=int(ring["tokens"].shape[0]), mods=list(ring["mods"]), companion_load_s=ring["load_seconds"])
    out["spread"] = token_spreads(ring, newest=newest)
    from mimo_bridge.body_layout import names_for
    handles = None
    if model:
        from mimo_bridge.tools import gof_probe_harness as H
        import torch
        torch.set_num_threads(4)
        t1 = time.time()
        frozen = H.load_frozen(str(ckpt))
        handles = list(frozen.model.handle_names)
        out["model_load_s"] = round(time.time() - t1, 1)
        out["model"] = model_readings(frozen, ring, n_slots, newest=newest)
        if internals:
            out["memory_deploy"] = MDR.checkpoint_part(frozen.model, ring, islots, ckpt=ckpt, newest=newest)
        del frozen
        gc.collect()
    else:
        out["model"] = None
    if handles is None:
        handles = list(names_for("trunk_v1"))
        if int(ring["achieved"].shape[2]) != len(handles):
            raise ValueError("the ring holds %d handles, trunk_v1 has %d" % (int(ring["achieved"].shape[2]), len(handles)))
    out["hands"] = hand_following(ring, handles, newest=newest)
    out["seconds"] = round(time.time() - t0, 1)
    out["peak_commit_gb"] = _peak_commit_gb()
    del ring
    gc.collect()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
        tmp.replace(cache)
    return out


def _peak_commit_gb() -> Optional[float]:
    """This process's peak private commit so far, decimal GB (Windows; None elsewhere), read as the probe suite
    reads it (gof_probe_suite._peak_commit_gb); the memory guard's own units are 1024-based GiB."""
    try:
        import psutil
        peak = getattr(psutil.Process().memory_info(), "peak_pagefile", None)
        return round(float(peak) / 1e9, 2) if peak is not None else None
    except Exception:                                  # noqa: BLE001 - a reading, never a failure
        return None


# ============================================================================== pooling with and without body 2
def _bodies(n: int, exclude: Iterable[int]) -> List[int]:
    ex = set(int(b) for b in exclude)
    return [b for b in range(int(n)) if b not in ex]


def model_values(s: dict, gaze_copy_active: bool = True) -> dict:
    """Readings 1-2 from summed rows (any set of bodies and window ends)."""
    out = {}
    eye_ok = bool(s["eye_rows"] and gaze_copy_active)
    un = s["eye_err_unchanged"]
    # THE EYE HEADLINE (pack v2): the own eye command's error advantage over a command that holds the eyes where they
    # are, as a share of the unchanged picture's error. Anchored to the picture's own change (measured 2026-09-25: the
    # model's error swung 1.0-3.1 times the unchanged picture's across six no-change checkpoints, which moves a ratio
    # of errors but cancels from a difference); against the eyes held (review 2026-09-25: against another body's
    # command an over-reacting gaze copy reads HIGHER - 199% at twice the right response)
    out["eye_prediction"] = ((s["eye_err_still"] - s["eye_err_own"]) / un) if (eye_ok and un > 0) else None
    out["eye_vs_other_body"] = ((s["eye_err_other"] - s["eye_err_own"]) / un) if (eye_ok and un > 0) else None
    out["eye_share_of_error"] = (1.0 - s["eye_err_own"] / s["eye_err_other"]) if (eye_ok and s["eye_err_other"] > 0) else None
    out["vision_error_vs_unchanged"] = (s["eye_err_own"] / un) if un > 0 else None
    out["body_action_sensitivity"] = math.sqrt(s["body_shift_sq"] / s["body_change_sq"]) if s["body_change_sq"] > 0 else None
    out["body_sensitivity_vs_actual_change"] = (math.sqrt(s["body_shift_sq"] / s["body_err_unchanged"])
                                                if s["body_err_unchanged"] > 0 else None)
    out["body_improvement_over_other_order"] = (1.0 - s["body_err_own"] / s["body_err_other"]) if s["body_err_other"] > 0 else None
    out["body_improvement_over_unchanged"] = (1.0 - s["body_err_own"] / s["body_err_unchanged"]) if s["body_err_unchanged"] > 0 else None
    out["eye_rows"], out["body_rows"] = int(s["eye_rows"]), int(s["body_rows"])
    return out


def model_slot_sums(m: dict, exclude: Iterable[int] = (), partners_of: Optional[Iterable[int]] = None) -> Dict[str, np.ndarray]:
    """Per window end, the model rows summed over the bodies kept; a body whose partner changes when `exclude` is
    left out (body 3 without body 2) is read from its kept-partner rows (model_readings' variants). partners_of:
    the exclusion that defines the partners, when it is not `exclude` (one body's own values: () = b - 1)."""
    ex = sorted(set(int(b) for b in exclude))
    pex = ex if partners_of is None else sorted(set(int(b) for b in partners_of))
    B = len(m["per_body"])
    keep = _bodies(B, ex)
    var = (m.get("variants") or {}).get(",".join(str(b) for b in pex)) if pex else None
    if pex and var is None and any(p != q for b, (p, q) in enumerate(zip(partners(B, pex), partners(B))) if b in keep):
        raise ValueError("these readings carry no kept-partner rows for the bodies without %s" % pex)
    out = {}
    for k in MODEL_KEYS:
        a = np.asarray(m["per_slot"][k], dtype=float)
        if a.size == 0:
            out[k] = np.zeros(0)
            continue
        cols = [b for b in keep if not (var and b in var["bodies"])]
        tot = a[:, cols].sum(axis=1)
        if var and var["bodies"]:
            tot = tot + np.asarray(var["per_slot"][k], dtype=float).sum(axis=1)
        out[k] = tot
    return out


def batch_sd(values: Sequence[Optional[float]]) -> Tuple[Optional[float], int]:
    """(the whole sample's sampling sd by batch means: the batches' standard deviation over the square root of their
    number, their number) from each batch's own reading; None with fewer than 2."""
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if len(v) < 2:
        return None, len(v)
    return float(np.std(v, ddof=1) / math.sqrt(len(v))), len(v)


def pooled(ck: dict, exclude: Iterable[int] = (), batches: int = BATCHES,
           partners_of: Optional[Iterable[int]] = None) -> dict:
    """The five checkpoint headlines pooled over the bodies kept (sums added, then the ratio), each with its
    sampling sd from repeats inside this checkpoint (`<name>_sd`, `<name>_n` the sample it was read on, `<name>_dof`):
    readings 1-2 on `batches` contiguous groups of window ends, hand following on the orders begun in `batches`
    equal spans, the token spreads on their 50-s windows."""
    out = {}
    m = ck.get("model")
    keep = _bodies(ck["bodies"], exclude)
    if m:
        s = model_slot_sums(m, exclude, partners_of)
        active = bool(m.get("gaze_copy_active"))
        out.update(model_values({k: float(v.sum()) for k, v in s.items()}, active))
        S = len(s["eye_rows"])
        parts = [p for p in np.array_split(np.arange(S), min(int(batches), S)) if len(p)] if S else []
        vals = [model_values({k: float(v[p].sum()) for k, v in s.items()}, active) for p in parts]
        for name, rows in (("eye_prediction", "eye_rows"), ("body_action_sensitivity", "body_rows")):
            sd, k = batch_sd([v[name] for v in vals])
            out[name + "_sd"], out[name + "_n"], out[name + "_dof"] = sd, out[rows], max(0, k - 1)
    else:
        out.update(eye_prediction=None, body_action_sensitivity=None)
    for sense, name in (("proprioception", "proprioception_variation"), ("vestibular", "vestibular_variation")):
        per = ck["spread"].get(sense)
        if not per:
            out[name] = None
            continue
        vals = [per[b]["mean"] for b in keep if per[b]["mean"] is not None]
        out[name] = float(np.mean(vals)) if vals else None
        low = min(((per[b]["mean"], b) for b in keep if per[b]["mean"] is not None), default=(None, None))
        out[name + "_lowest_body"] = dict(body=low[1], value=low[0])
        out[name + "_deaf_bodies"] = [b for b in keep if per[b]["mean"] is not None and per[b]["mean"] < DEAF_LINES[sense]]
        # the repeats: each 50-s window's value over the kept bodies that have it
        by_end: Dict[int, List[float]] = {}
        for b in keep:
            for v, e in zip(per[b]["windows"], per[b].get("ends_back") or []):
                by_end.setdefault(int(e), []).append(float(v))
        sd, k = batch_sd([float(np.mean(v)) for _e, v in sorted(by_end.items())])
        out[name + "_sd"], out[name + "_n"], out[name + "_dof"] = sd, k, max(0, k - 1)
    for part, name in (("fingers", "hand_following"), ("wrists", "wrist_following")):
        s = {k: sum(ck["hands"][b][part][k] for b in keep) for k in ("part_orders", "closer", "chance", "asked_deg", "left_deg")}
        out[name] = (1.0 - s["left_deg"] / s["asked_deg"]) if s["asked_deg"] > 0 else None
        out[name + "_closer_per_100"] = (100.0 * s["closer"] / s["part_orders"]) if s["part_orders"] else None
        out[name + "_chance_per_100"] = (100.0 * s["chance"] / s["part_orders"]) if s["part_orders"] else None
        out[name + "_part_orders"] = int(s["part_orders"])
        nb = min(len((ck["hands"][b].get("batches") or {}).get(part) or []) for b in keep) if keep else 0
        bat = []
        for j in range(nb):
            asked = sum(ck["hands"][b]["batches"][part][j]["asked_deg"] for b in keep)
            left = sum(ck["hands"][b]["batches"][part][j]["left_deg"] for b in keep)
            bat.append((1.0 - left / asked) if asked > 0 else None)
        sd, k = batch_sd(bat)
        out[name + "_sd"], out[name + "_n"], out[name + "_dof"] = sd, int(s["part_orders"]), max(0, k - 1)
    return out


def per_body_table(ck: dict) -> List[dict]:
    """Each body's own headline values (the same pooling on one body)."""
    return [dict(body=b, **{k: v for k, v in pooled(ck, [x for x in range(ck["bodies"]) if x != b], partners_of=()).items()
                            if k in ("eye_prediction", "body_action_sensitivity", "proprioception_variation",
                                     "vestibular_variation", "hand_following")}) for b in range(ck["bodies"])]


# ======================================================================================== compare and lines
def load_lines(path: Optional[Path] = None) -> dict:
    """The noise file's measurements (its `lines`: each reading's repeat on one checkpoint and the observer's rows
    and count line) when it was written by THIS pack version; {} otherwise (a file of another definition gives no
    lines, so no verdict reads a line measured for something else)."""
    p = Path(path) if path else NOISE_FILE
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if doc.get("version") != PACK_VERSION:
        return {}
    return doc.get("lines") or {}


def t_quantile(dof: int, rate: float = FALSE_ALARM_PER_READING) -> float:
    """Student's t at 1 - rate (one-sided) with `dof` degrees of freedom."""
    from scipy import stats
    return float(stats.t.ppf(1.0 - float(rate), max(1, int(dof))))


def slots_needed(sd_24: float, target: float, dof: int = BATCHES - 1) -> int:
    """How many window ends per body put a same-size compare's line under `target`, when one 24-end sample's
    sampling sd is sd_24 (independent moments: the sd falls as 1 / sqrt(N)): the smallest N with
    t x sqrt(2) x sd_24 x sqrt(24 / N) <= target."""
    return int(math.ceil(24.0 * (t_quantile(dof) * math.sqrt(2.0) * float(sd_24) / float(target)) ** 2))


def side_sd(name: str, variant: str, pooled_side: dict, lines: dict) -> Tuple[Optional[float], Optional[int], str]:
    """(sd, degrees of freedom, where from) of one side's reading: the larger of its own repeats (pooled's
    `<name>_sd`) and the noise file's repeat on one checkpoint scaled to this side's sample (sd_ref x sqrt(n_ref / n))."""
    own, n, dof = pooled_side.get(name + "_sd"), pooled_side.get(name + "_n"), pooled_side.get(name + "_dof")
    ref = ((lines.get("repeat") or {}).get(name) or {}).get(variant) or {}
    scaled = None
    if ref.get("sd") is not None and ref.get("n") and n:
        scaled = float(ref["sd"]) * math.sqrt(float(ref["n"]) / float(n))
    got = [(v, d, w) for v, d, w in ((own, dof, "this side's repeats"), (scaled, ref.get("dof"), "the noise file's repeat"))
           if v is not None]
    if not got:
        return None, None, "unmeasured"
    v, d, w = max(got, key=lambda x: x[0])
    return float(v), (None if d is None else int(d)), w


def line_of(name: str, variant: str, before: dict, after: dict, lines: dict) -> dict:
    """The no-change line of one reading for this pair: t(1 - FALSE_ALARM_PER_READING, dof) x sqrt(sd_b^2 + sd_a^2),
    dof the smaller of the two sides' (a side's sd from few repeats widens the line)."""
    sb, db, wb = side_sd(name, variant, before, lines)
    sa, da, wa = side_sd(name, variant, after, lines)
    if sb is None or sa is None:
        return dict(line=None, sd_before=sb, sd_after=sa)
    dof = min(x for x in (db, da) if x is not None) if (db is not None or da is not None) else BATCHES - 1
    t = t_quantile(dof)
    return dict(line=t * math.sqrt(sb * sb + sa * sa), sd_before=sb, sd_after=sa, dof=dof, t=t,
                sd_from=[wb, wa], false_alarm_per_reading=FALSE_ALARM_PER_READING)


def verdict(name: str, before, after, line) -> dict:
    """REGRESSED / IMPROVED / within noise for one headline; no verdict without both readings and a line."""
    if before is None or after is None:
        return dict(change=None, verdict="unmeasured: a reading is missing")
    change = float(after) - float(before)
    if line is None:
        return dict(change=change, verdict="no verdict: no noise line measured for this reading")
    good = BETTER[name] * change
    if good < -float(line):
        v = "REGRESSED"
    elif good > float(line):
        v = "IMPROVED"
    else:
        v = "within noise"
    return dict(change=change, line=float(line), verdict=v)


def compare(before: dict, after: dict, *, saved: Path, model: bool = True, cache_dir: Optional[Path] = None,
            lines: Optional[dict] = None, n_slots: int = MODEL_SLOTS, internals: bool = True,
            deploy_lines: Optional[dict] = None) -> dict:
    """The whole pack on two sides (parse_side). Every headline with and without body 2; the verdict is read on
    the reading without body 2 (the operator's decision: body 2 is excluded from learning-performance
    conclusions), the reading with it given beside. internals (with the model part): the memory deploy's
    readings beside the headlines (report["memory_deploy"], judged against deploy_lines, default
    memory_deploy_readings.NOISE_FILE)."""
    from mimo_bridge.tools import memory_deploy_readings as MDR
    lines = load_lines() if lines is None else lines
    internals = bool(internals and model)
    if int(before["hi"]) > int(after["lo"]):
        raise ValueError("the before window (%s, %s] and the after window (%s, %s] overlap: the after side must hold "
                         "only what came after the before side (give --switch-step)"
                         % (before["lo"], before["hi"], after["lo"], after["hi"]))
    t0 = time.time()
    report = dict(version=PACK_VERSION, generated=time.strftime("%Y-%m-%d %H:%M:%S"), before=before, after=after,
                  excluded_body=EXCLUDED_BODY, body_numbering="from 0 (the dashboard numbers from 1: body 2 here is its body 3)",
                  headlines={}, beside={}, observer=None, notes=[])
    sides = {}
    for key, side in (("before", before), ("after", after)):
        recs = stats_records(saved, side["lo"], side["hi"])
        ck = (checkpoint_readings(side["checkpoint"], model=model, n_slots=n_slots, cache_dir=cache_dir,
                                  internals=internals, newest=side.get("newest"))
              if side.get("checkpoint") else None)
        sides[key] = dict(records=records_readings(recs), checkpoint=ck,
                          pooled_all=pooled(ck) if ck else {}, pooled_without=pooled(ck, [EXCLUDED_BODY]) if ck else {},
                          per_body=per_body_table(ck) if ck else None)
    # the ring's own count of the moments added between the two checkpoints, beside the step count used for `newest`
    cb, ca = sides["before"]["checkpoint"], sides["after"]["checkpoint"]
    if cb and ca and cb.get("ring_ptr") is not None and ca.get("ring_ptr") is not None and after.get("newest") is not None:
        added, steps = int(ca["ring_ptr"]) - int(cb["ring_ptr"]), int(ca["step"]) - int(cb["step"])
        if added != steps:
            report["notes"].append("the ring added %d moments between the checkpoints over %d steps; the after side "
                                   "read its newest %d" % (added, steps, int(after["newest"])))
    obs = observer_state(saved)
    ow = {}
    for variant, ex in (("all", ()), ("without_body_2", (EXCLUDED_BODY,))):
        wb = observer_window(obs, before["lo"], before["hi"], ex)
        wa = observer_window(obs, after["lo"], after["hi"], ex)
        ow[variant] = dict(before=wb, after=wa,
                           change=observer_change(wb, wa, row_lines=(lines.get("observer_rows") or {}).get(variant)))
    report["observer"] = dict(store_generation=obs["store_generation"], baseline=obs["baseline_path"],
                              detector_version=obs["detector_version"], rules_version=obs["rules_version"], **ow)
    for name in HEADLINES:
        if name == "observer_counts":
            row = {}
            for variant in ("all", "without_body_2"):
                ch = ow[variant]["change"]
                line = (lines.get(name) or {}).get(variant)
                n_fell = len(ch["fell"])
                v = ("no verdict: no noise line measured for this reading" if (line is None or not ch["row_lines_used"])
                     else ("REGRESSED" if n_fell > float(line) else "within noise"))
                row[variant] = dict(value=n_fell, tested=ch["tested"], fell=[(c, ch["rows"][c]["name"]) for c in ch["fell"]],
                                    rose=[(c, ch["rows"][c]["name"]) for c in ch["rose"]], line=line, verdict=v)
            report["headlines"][name] = row
            continue
        row = {}
        for variant, key in (("all", "pooled_all"), ("without_body_2", "pooled_without")):
            b = sides["before"][key].get(name)
            a = sides["after"][key].get(name)
            ln = line_of(name, variant, sides["before"][key], sides["after"][key], lines)
            row[variant] = dict(before=b, after=a, **verdict(name, b, a, ln["line"]),
                                **{k: v for k, v in ln.items() if k != "line"})
        report["headlines"][name] = row
    report["beside"] = dict(before=dict(records=sides["before"]["records"], pooled_all=sides["before"]["pooled_all"],
                                        pooled_without=sides["before"]["pooled_without"], per_body=sides["before"]["per_body"]),
                            after=dict(records=sides["after"]["records"], pooled_all=sides["after"]["pooled_all"],
                                       pooled_without=sides["after"]["pooled_without"], per_body=sides["after"]["per_body"]))
    if internals:
        parts = [(sides[k]["checkpoint"] or {}).get("memory_deploy") for k in ("before", "after")]
        report["memory_deploy"] = MDR.compare_parts(parts[0], parts[1], deploy_lines)
    report["checkpoint_costs"] = {k: (None if not sides[k]["checkpoint"] else
                                      dict({x: sides[k]["checkpoint"].get(x) for x in ("seconds", "companion_load_s", "model_load_s", "peak_commit_gb")},
                                           memory_deploy_s=(sides[k]["checkpoint"].get("memory_deploy") or {}).get("seconds")))
                                  for k in ("before", "after")}
    report["seconds"] = round(time.time() - t0, 1)
    return report


# ================================================================================================ the table
def _pct(v, digits: int = 1) -> str:
    if v is None:
        return "unmeasured"
    return ("%." + str(digits) + "f%%") % (100.0 * float(v))


def _pts(v) -> str:
    if v is None:
        return "unmeasured"
    return "%+.1f points" % (100.0 * float(v))


def table(report: dict) -> str:
    """One short plain-language table: each headline with its meaning, both sides without and with body 2, the
    change, the no-change line and the verdict (read without body 2)."""
    b, a = report["before"], report["after"]

    def side(s):
        what = ("checkpoint %s" % format(s["checkpoint_step"], ",") if s.get("checkpoint_step") is not None else "no checkpoint")
        ring = ("its whole replay ring" if s.get("newest") is None
                else "the %s ring moments lived after the switch at %s" % (format(s["newest"], ","), format(s["switch_step"], ",")))
        return "%s, %s (records of steps %s-%s)" % (what, ring, format(s["lo"], ","), format(s["hi"], ","))
    out = ["Regression pack (%s), %s" % (report["version"], report["generated"]),
           "Before: " + side(b), "After:  " + side(a),
           "Body 2 is numbered from 0 (the dashboard's body 3); the verdict is read without it. A change counts only "
           "beyond the no-change line: the sampling noise of both sides, measured as repeats inside each checkpoint, "
           "at which two checkpoints with no change between them cross it 1 time in %d per reading (1 in %d across "
           "the six)." % (int(round(1.0 / FALSE_ALARM_PER_READING)), int(round(1.0 / FALSE_ALARM_FAMILY)))]
    out += ["Note: " + n for n in report.get("notes") or []]
    out += ["",
           "| reading | what the number is | before (without body 2 / all 8) | after (without body 2 / all 8) | change without body 2 | no-change line | verdict |",
           "|---|---|---|---|---|---|---|"]
    for name in HEADLINES:
        row = report["headlines"][name]
        what, scale = MEANING[name]
        if name == "observer_counts":
            w, al = row["without_body_2"], row["all"]
            fell = ", ".join(n for _c, n in w["fell"]) or "none"
            rose = ", ".join(n for _c, n in w["rose"]) or "none"
            out.append("| observer behaviour counts | %s (%s) | the before window is the reference | %d of %d rows "
                       "fell / %d with all 8 | fell: %s; rose: %s | %s rows | %s |" % (
                           what, scale, w["value"], w["tested"], al["value"], fell, rose,
                           "unmeasured" if w["line"] is None else "%g" % w["line"], w["verdict"]))
            continue
        w, al = row["without_body_2"], row["all"]
        out.append("| %s | %s (%s) | %s / %s | %s / %s | %s | %s | %s |" % (
            name.replace("_", " "), what, scale, _pct(w["before"]), _pct(al["before"]), _pct(w["after"]), _pct(al["after"]),
            _pts(w.get("change")), "unmeasured" if w.get("line") is None else "%.1f points" % (100.0 * w["line"]),
            w["verdict"]))
    rb, ra = report["beside"]["before"]["records"], report["beside"]["after"]["records"]
    out += ["", "Production's live gauges over the same windows (all 8 bodies; not kept per body):",
            "- eye command's share of the picture error removed (live gaze copy): %s before, %s after" % (
                _pct(rb.get("eye_view_share_live")), _pct(ra.get("eye_view_share_live"))),
            "- eye aim forecast 150 ms ahead, share of 'the eyes stay still' error removed: %s before, %s after" % (
                _pct(rb.get("eye_aim_150ms_share_live")), _pct(ra.get("eye_aim_150ms_share_live"))),
            "- body-position prediction's sensitivity to the order (live audit): %s before, %s after" % (
                _pct(rb.get("body_sensitivity_live")), _pct(ra.get("body_sensitivity_live"))),
            "- hands' mean joint error from their joint goals: %s before, %s after (0 degrees = on the goal)" % (
                "unmeasured" if rb.get("hand_joint_error_deg_live") is None else "%.1f degrees" % rb["hand_joint_error_deg_live"],
                "unmeasured" if ra.get("hand_joint_error_deg_live") is None else "%.1f degrees" % ra["hand_joint_error_deg_live"])]
    text = "\n".join(out) + "\n"
    if report.get("memory_deploy") is not None:
        from mimo_bridge.tools import memory_deploy_readings as MDR
        text += MDR.table(report["memory_deploy"])
    return text


# ================================================================================================ noise
def repeat_reference(ck: dict) -> dict:
    """Each ring reading's repeat on this one checkpoint (pooled's sd by batch means or over the 50-s windows), with
    and without body 2, and what it implies for a compare of two same-size samples: the line t x sqrt(2) x sd and,
    for body action sensitivity, whether that line is under the change it must show (DETECT)."""
    ref = {}
    for variant, ex in (("all", ()), ("without_body_2", (EXCLUDED_BODY,))):
        p = pooled(ck, ex)
        for name in HEADLINES[:-1]:
            sd, n, dof = p.get(name + "_sd"), p.get(name + "_n"), p.get(name + "_dof")
            row = dict(value=p.get(name), sd=sd, n=n, dof=dof)
            if sd is not None and dof:
                row["same_size_line"] = t_quantile(dof) * math.sqrt(2.0) * sd
                if name in DETECT:
                    row["must_show"] = DETECT[name]
                    row["line_under_what_it_must_show"] = bool(row["same_size_line"] < DETECT[name])
            ref.setdefault(name, {})[variant] = row
        if ck.get("model"):
            ref.setdefault("eye_vs_other_body", {})[variant] = dict(value=p.get("eye_vs_other_body"))
    return ref


def noise(checkpoint: str, observer_lo: int, observer_hi: int, *, saved: Path, window: int = WINDOW_STEPS,
          model: bool = True, cache_dir: Optional[Path] = None, n_slots: int = MODEL_SLOTS) -> dict:
    """The noise measurements the verdicts read (pack v2):
      repeat: every ring reading measured as repeats on ONE checkpoint (pooled's batch means of the model sample
        and of the orders, the token spreads' 50-s windows), so no training drift enters it; a compare scales the
        sd to its own sample size and takes the larger of that and its own sides' repeats (side_sd);
      observer_rows: each capability row's measured no-change scatter over consecutive `window`-step windows
        between observer_lo and observer_hi (observer_row_lines);
      observer_counts: the count line from the rule measured on those windows, each held out in turn
        (observer_leave_one_out), with the share of no-change comparisons that read a fallen row."""
    ck = checkpoint_readings(checkpoint, model=model, n_slots=n_slots, cache_dir=cache_dir)
    lines = dict(repeat=repeat_reference(ck))
    obs = observer_state(saved)
    wins = [(s, s + window) for s in range((int(observer_lo) // BIN) * BIN, int(observer_hi) - window + 1, window)]
    ocounts, orows = {}, {}
    for variant, ex in (("all", ()), ("without_body_2", (EXCLUDED_BODY,))):
        ws = [observer_window(obs, s, e, ex) for s, e in wins]
        orows[variant] = observer_row_lines(ws)
        loo = observer_leave_one_out(ws)
        plain = [len(observer_change(ws[i], ws[j])["fell"]) for i in range(len(ws)) for j in range(len(ws)) if i != j]
        ocounts[variant] = dict(windows=[[w["lo"], w["hi"], w["body_steps"]] for w in ws], leave_one_out=loo,
                                plain_test_fallen_counts=plain)
    lines["observer_counts"] = {v: ocounts[v]["leave_one_out"]["line"] for v in ocounts}
    lines["observer_rows"] = orows
    rec = [dict(lo=s, hi=e, **records_readings(stats_records(saved, s, e))) for s, e in wins]
    return dict(version=PACK_VERSION, generated=time.strftime("%Y-%m-%d %H:%M:%S"), checkpoint=str(checkpoint),
                step=ck["step"], n_slots=int(n_slots), batches=BATCHES, false_alarm_per_reading=FALSE_ALARM_PER_READING,
                lines=lines, observer=ocounts, records_windows=rec, window=window,
                observer_lo=observer_lo, observer_hi=observer_hi,
                cost=dict(seconds=ck.get("seconds"), companion_load_s=ck.get("companion_load_s"),
                          model_load_s=ck.get("model_load_s"), model_s=(ck.get("model") or {}).get("seconds"),
                          peak_commit_gb=ck.get("peak_commit_gb")),
                rule=("a reading's line is t(1 - %.5f, dof) x sqrt(sd_before^2 + sd_after^2), each side's sd the larger "
                      "of its own repeats and this file's repeat scaled to its sample; the observer's count line is the "
                      "most rows that fell in the leave-one-out no-change comparisons" % FALSE_ALARM_PER_READING))


# ====================================================================== the jump chain (the heavier part)
# The operator's shared review (relayed by the coordinator, 2026-09-25 evening): after the whole repair sequence
# the decisive checkpoint is whether the cleaner body gives a measurable jump along ONE chain - action-conditioned
# prediction, then planner candidate separation, then planner choice quality, then actual physical improvement
# over the learned reflex controller. Run once before the sequence and once after, never after every watch. Each
# link is read with the instrument that will read it after, and each has a declared JUMP rule with its arithmetic:
#   1 action-conditioned prediction  the persistent-memory author's watch readings (records, no model;
#                                    docs/gain_of_function_v4/persistent_memory_watch_readings.py on
#                                    fix/persistent-memory-2026-09-25): the median of the last 20 full reports;
#                                    a jump = after minus before beyond the record's largest no-change change of a
#                                    median of 20 reports (its median_of_20_change_abs_max), in the better direction
#   2 candidate separation           the same file: the planner candidates' predicted-wellness spread per body; the
#                                    same rule. Beside it the planner recheck's captured share (version 3 of
#                                    docs/gain_of_function_v4/probe_planner_candidates.py, the round link 3 reads):
#                                    over every pair of the five candidates from one saved start, sum of mean((pred
#                                    a - pred b) x (real a - real b)) / sum of mean((real a - real b)^2), pooled over
#                                    vision, tactile, audio and proprioception, planner route, 50 ms ahead; 0 = the
#                                    predicted difference between two candidates carries none of the real one, 1 =
#                                    all of it, below 0 = the wrong way. It replaced (jump_chain_v3, 2026-10-02) the
#                                    share of choices inside the evaluator's tie band: that band (0.0734) is 12 to 22
#                                    times the evaluator's predicted spread between the candidates in the nine
#                                    rechecks of 399,100 to 450,525, so every choice ties whatever the forward model
#                                    does and the tie share cannot move (the own-command analysis review, Oct 2)
#   3 planner choice quality         docs/gain_of_function_v4/probe_planner_candidates.py's result (the ability
#                                    scheduler's planner_candidates task): out of 100 pairs of candidates, how many
#                                    the forecast orders as their actual distances to the goal at the order's end
#                                    are ordered (50 = chance); a jump = a rise beyond the largest change between
#                                    the pre-repair rounds AND the after's 95% body interval above 50 - at the
#                                    pre-repair rounds' interval half-widths (7.9-15.7) the interval condition binds:
#                                    the reading must reach 58-66 (CHAIN_RANKING_BAR); beside it the probe's first
#                                    flat stage along the route
#   4 physical improvement           docs/gain_of_function_v4/probe_planner_vs_reflex.py's result (v2): how much
#     over the reflex controller     nearer its goal the planner's order ended than the REFLEX ORDER from the
#                                    identical saved state, as a share of the mean starting distance (so no start
#                                    with a small goal distance outweighs the rest); a jump = the after's 95% body
#                                    interval entirely above 0 and its mean above the before's interval, AND the
#                                    matched mirror control's interval above 0 (the planner's direction, not any
#                                    deviation, is what helps)
# Every reading with and without body 2 where the instrument keeps bodies apart; the jump is read without it.
# Each link's REACHABILITY ARITHMETIC (the style guide, "Declared criteria ship with reachability arithmetic") is
# CHAIN_REACHABILITY: the judged window and its lag, the noise on it, and the size of change the rule demands
# against what is measured today.
# v3 (2026-10-02): link 2 reads the planner recheck's captured share in place of the tie share (above)
CHAIN_VERSION = "jump_chain_v3"
# (section, name, compare, better): the watch file's readings the chain reads
CHAIN_WATCH = {
    "action_conditioned_proprioception": ("stats", "action_conditioned_proprioception", "diff", +1),
    "sensitivity_proprioception": ("stats", "sensitivity_proprioception", "ln", +1),
    "efference_action_conditioned": ("per_body", "efference_action_conditioned", "diff", +1),
    "candidate_spread": ("per_body", "candidate_spread", "ln", +1),
}
CHAIN_LINK_OF = {"action_conditioned_proprioception": 1, "sensitivity_proprioception": 1,
                 "efference_action_conditioned": 1, "candidate_spread": 2}
# link 2's captured share. Its noise: the largest change between the pooled readings of the nine rechecks of
# checkpoints 399,100 to 450,525 (Saved/mimo_bridge/ability_rechecks/planner_candidates_*/result.json,
# summary.forward_model.senses.planner["1"].judge_senses.captured_share: 0.0020 at 414,200 to 0.0171 at 450,525),
# which span training and repairs and so is wide. Vision carries all of it there (its own share 0.17 to 0.36;
# tactile, audio and proprioception within 0.0005 of 0), weighted by vision's part of the pooled real difference
# (0.007 to 0.081), which the saved worlds vary from round to round.
CHAIN_CAPTURED_NOISE = 0.0151
CHAIN_CAPTURED_PROBE = "planner_candidates_v3"      # the probe version whose rows carry the float32 floor
CHAIN_CAPTURED_SENSES = ("vision", "tactile", "audio", "proprioception")   # the probe's JUDGE_SENSES
# the evaluator's predicted spread between the five candidates (median over the starts) in those nine rechecks
# (summary.evaluator.candidate_spread_median.predicted: 0.0033 to 0.0062): the tie band is 11.9 to 22.5 times it
CHAIN_TIE_BAND_OVER_SPREAD = (11.9, 22.5)
# the largest change of the ranking between the three pre-repair planner_candidates_v2 rounds (408,450: 55.7;
# 411,125: 56.0; 414,200: 46.25 per 100 pairs; Saved/mimo_bridge/ability_rechecks/planner_candidates_*/result.json):
# used when no rounds are given to measure it again
CHAIN_RANKING_NOISE = 9.7
# the pre-repair rounds' 95% body interval half-widths (408,450: 47.5-63.3; 411,125: 40.5-71.9; 414,200: 33.75-58.1),
# so an interval clears 50 only at a reading of 50 + 7.9 to 50 + 15.7: the true bar of link 3 is 58-66, not the
# before + 9.7 = 55.95 the first record wrote (review of 2026-09-25, section 4)
CHAIN_RANKING_HALF_WIDTHS = (7.9, 15.7, 12.18)
TIE_BAND = 0.0734                       # mimo_bridge.causal_chain's tie band on the 0-5 wellness sum
CHAIN_REACHABILITY = {
    1: ("Judged window: the median of 20 full reports (one per 100 updates, so 2,000 updates); a change reaches half "
        "its weight in that median 1,000 updates after it, so the after reading is taken at least 2,000 updates after "
        "the last repair's switch. Noise: the record's largest no-change move of such a median since 380,700 "
        "(0.121 points for the real order's share of body-position error removed, a factor of 5.4 for the order's "
        "sensitivity, 12.8 points for the efference head without body 2), which includes four launches and so is "
        "wide. Demanded against today: the real order's share must rise from -0.004% to above +0.117%; the "
        "sensitivity from 0.90% to above 4.8%; the efference head from 39.7% to above 52.4%. No repair in the "
        "sequence declares an expected size for these; the memory repair's own goal (a sense's error lowered by the "
        "memory's variation) was never reached by any checkpoint before the memory saturated."),
    2: ("Judged window: for the candidate spread, as link 1 (with its lag); for the captured share (the planner "
        "recheck's, probe version 3), one planner-candidates round (8 bodies x 2 saved planning events, 10 pairs of "
        "candidates each), the round link 3 reads. Candidate spread: from 0.0038 to above 0.0061 (a factor of 1.62, "
        "its no-change line). Captured share noise: 1.51 in 100, the largest change between the nine rechecks of "
        "checkpoints 399,100 to 450,525 (pooled readings 0.2 to 1.7 in 100), and the after round's own body "
        "interval. Demanded against the before "
        "round at 415,025 (0.22 in 100 without body 2, interval 0.04 to 1.18): above 1.73 in 100, with the after's "
        "interval above 0.22 and its rows measured by version 3 (the float32 floor checked). Goal line: 100 in 100 "
        "(the predicted difference between two candidates equals the real one along its direction); at 415,025 the "
        "reading is 99.8 in 100 short of it. Vision carries all of today's share (its own share 22 in 100 at "
        "415,025, through the gaze copy; tactile, audio and proprioception within 5 in 10,000 of 0 in every "
        "recheck), and vision's real difference is 1 in 100 of the pooled real difference at 415,025 (0.7 to 8.1 in "
        "100 across the rechecks): at that mix a perfect vision forecast adds 0.8 in 100, under the jump line, so a "
        "jump needs tactile, audio or proprioception to carry part of their own difference (link 1's quantity) "
        "first. The share of choices inside the tie band is no longer read: the band (0.0734) is 12 to 22 times "
        "the evaluator's predicted spread between the candidates in those nine rechecks, so every choice ties "
        "whatever the forward model does."),
    3: ("Judged window: one planner-candidates round (8 bodies x 2 saved planning events, 10 pairs each) on the "
        "checkpoint at the end of the sequence. Noise: 9.7 in 100 (the largest change between the three pre-repair "
        "rounds) and the round's own body interval, whose half-width was 7.9-15.7 in 100. Demanded: from 46.25 to "
        "58-66 in 100 (the interval above 50 binds; before + 9.7 = 55.95 does not). The probe calls 80-90 in 100 a "
        "ranking a planner can use. Today the first flat stage along the route is the forward model (it captures "
        "0.2% of the between-candidate difference), so a rise needs link 1 to rise first."),
    4: ("Judged window: 7 bodies x 3 saved planning events (body 2 left out), each order held for its 250-ms tenure. "
        "Noise: the body-resampled interval of the planner-over-reflex reading, and the change between 414,200 and "
        "415,025 (both pause jobs; unmeasured until they run). Demanded: the planner's order ending nearer its own "
        "goal than the reflex order on average across bodies, with the interval above 0 and above the before's; at "
        "414,200 the planner's order closed less of its goal distance than the reflex order in 11 of 16 starts "
        "(9 of 14 without body 2), so today's reading is expected below 0."),
}


def _watch_value(doc: dict, section: str, name: str, variant: str) -> Tuple[Optional[float], Optional[float]]:
    """(the median of the last 20 full reports, the record's largest no-change change of a median of 20) for one
    reading of the watch file; a stats reading has one variant (all 8 bodies pooled)."""
    block = (doc.get(section) or {}).get(name) or {}
    if section == "per_body":
        block = block.get("all_bodies" if variant == "all" else "without_body_2") or {}
    return _num(block.get("baseline_last_20_median")), _num(block.get("median_of_20_change_abs_max"))


def chain_watch(before: dict, after: Optional[dict]) -> dict:
    out = {}
    for key, (section, name, compare, better) in CHAIN_WATCH.items():
        row = dict(link=CHAIN_LINK_OF[key], compare=compare, better="up" if better > 0 else "down",
                   pooled_only=(section == "stats"))
        for variant in ("all", "without_body_2"):
            if section == "stats" and variant == "without_body_2":
                row[variant] = dict(note="the stats record pools the 8 bodies; no body split exists")
                continue
            b, line = _watch_value(before, section, name, variant)
            a = None if after is None else _watch_value(after, section, name, variant)[0]
            cell = dict(before=b, line=line, after=a)
            if b is not None and line is not None:
                if compare == "ln":
                    cell["jump_needs"] = (b * math.exp(line) if better > 0 else b * math.exp(-line)) if b > 0 else None
                else:
                    cell["jump_needs"] = b + better * line
            if a is not None and b is not None and line is not None:
                ch = (math.log(a / b) if (compare == "ln" and a > 0 and b > 0) else (a - b))
                cell.update(change=ch, jump=bool(better * ch > line))
            row[variant] = cell
        out[key] = row
    return out


def _probe_module(name: str):
    import importlib.util
    path = REPO / "docs" / "gain_of_function_v4" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(folder: Path) -> List[dict]:
    rows = []
    with open(Path(folder) / "starts.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def captured_share_of(summary: dict) -> dict:
    """Link 2's reading from a probe_planner_candidates summary: the captured share pooled over the senses Judge B
    reads, planner route, 50 ms ahead (summary.forward_model.senses.planner["1"].judge_senses.captured_share), with
    its 95% body interval and measurability (True / False = the pooled real difference above / within its float32
    floor; None = rows measured before version 3, which recorded no floor), and each sense's own share beside it."""
    depth1 = ((((summary or {}).get("forward_model") or {}).get("senses") or {}).get("planner") or {}).get("1") or {}

    def block(name):
        cs = (depth1.get(name) or {}).get("captured_share") or {}
        return dict(value=_num(cs.get("value")), interval=cs.get("interval"), measurable=cs.get("measurable"),
                    bodies=cs.get("bodies"))
    out = block("judge_senses")
    out["senses"] = {m: block(m) for m in CHAIN_CAPTURED_SENSES if m in depth1}
    return out


def captured_cell(before: Optional[dict], after: Optional[dict], noise: float = CHAIN_CAPTURED_NOISE) -> dict:
    """Link 2's captured-share jump for one variant: the after's reading above the before's by more than `noise`,
    the after's 95% body interval above the before's reading (the rise exceeds the after round's own sampling), and
    the after measured by version 3 and measurable (its pooled real difference above the float32 floor)."""
    b, a = (before or {}).get("value"), (after or {}).get("value")
    cell = dict(before=b, after=a, before_interval=(before or {}).get("interval"),
                after_interval=(after or {}).get("interval"), after_measurable=(after or {}).get("measurable"),
                jump_needs=None if b is None else b + noise)
    if after is not None and after.get("measurable") is False:
        cell.update(jump=False, note="not measurable: the after round's real difference between candidates is "
                                     "within the float32 floor of the tokens")
    elif b is not None and a is not None:
        iv = after.get("interval")
        cell.update(change=a - b, jump=bool(a - b > noise and iv and iv[0] > b and after.get("measurable") is True))
        if after.get("measurable") is None:
            cell["note"] = ("the after round's rows were measured before %s (no float32 floor recorded), so no jump "
                            "is read" % CHAIN_CAPTURED_PROBE)
    return cell


def chain_candidates(folder: Path, draws: int = 5000) -> dict:
    """Link 3 from a probe_planner_candidates result folder: its first flat stage, the ranking of the goal distance
    at the order's end (and of wellness over the tenure), and the chosen order's actual wellness against the
    alternatives, for all bodies (the result's own summary) and without body 2 (the probe's summarise on its rows
    without body 2)."""
    P = _probe_module("probe_planner_candidates")
    result = json.loads((Path(folder) / "result.json").read_text(encoding="utf-8"))
    rows = _rows(folder)
    out = dict(folder=str(folder), checkpoint=result.get("checkpoint"), probe=result.get("probe"),
               script_sha256=result.get("script_sha256"), starts=result.get("evaluated_starts"),
               summarised_by=P.PROBE_VERSION)
    for variant, rs in (("all", rows), ("without_body_2", [r for r in rows if int(r["body"]) != EXCLUDED_BODY])):
        s = P.summarise(rs, draws=draws)
        v = P.decide(s)
        rk = s.get("ranking") or {}
        out[variant] = dict(
            # link 2: the planner recheck's captured share (module text, link 2) and each judged sense's own
            captured_share=captured_share_of(s),
            first_flat_stage=v["first_flat_stage_along_the_route"],
            forward_model_transfer=v["forward_model_transfer"], forward_model_captured_share=v["forward_model_captured_share"],
            ranking_goal_distance_end=dict((k, (rk.get("goal_distance_end") or {}).get(k))
                                           for k in ("per_100", "interval", "pairs_judged", "bodies")),
            ranking_wellness_tenure=dict((k, (rk.get("wellness_tenure") or {}).get(k))
                                         for k in ("per_100", "interval", "pairs_judged", "bodies")),
            chosen_minus_alternatives_wellness=(s.get("planner") or {}).get("chosen_actual_wellness_minus_alternatives"),
            chosen_rank_by_actual_goal_distance=((s.get("planner") or {}).get("chosen_rank_by_actual_goal_distance") or {}).get("value"))
    return out


REFLEX_KEYS = ("planner_over_reflex", "planner_over_mirror", "mirror_over_reflex", "starts_nearer_than_reflex_per_100",
               "mirror_matched", "mean_start_distance", "advantage_over_reflex_goal_closure",
               "planner_value_goal_closure", "planner_value_wellness_tenure", "advantage_over_reflex_wellness_tenure",
               "evaluator_agreement", "deviation", "physical_improvement_over_reflex")


def chain_reflex(folder: Path) -> dict:
    """Link 4 from a probe_planner_vs_reflex result folder (v2: the planner against the reflex order, the matched
    mirror as the control)."""
    result = json.loads((Path(folder) / "result.json").read_text(encoding="utf-8"))
    out = dict(folder=str(folder), checkpoint=result.get("checkpoint"), probe=result.get("probe"),
               starts=result.get("evaluated_starts"), skipped=len(result.get("skipped_starts") or []))
    for variant, key in (("all", "all_bodies"), ("without_body_2", "without_body_2")):
        s = (result.get("summary") or {}).get(key) or {}
        out[variant] = {k: s.get(k) for k in REFLEX_KEYS}
    return out


def watch_file(path: str, before_step: Optional[int] = None) -> dict:
    """A watch readings file, refused when its last report lies past `before_step` (a before side regenerated after
    a switch would move onto repaired reports), with its report span and SHA-256 for the record."""
    import hashlib
    raw = Path(path).read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    last = doc.get("last_report")
    if before_step is not None and (last is None or int(last) > int(before_step)):
        raise ValueError("the before watch file %s ends at report %s, past the before checkpoint %s"
                         % (path, last, before_step))
    doc["_pinned"] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), first_report=doc.get("first_report"),
                          last_report=last, last_20_reports=doc.get("last_20_reports"))
    return doc


def ranking_bar(before_per_100: Optional[float], noise: float, half_widths: Sequence[float]) -> Optional[List[float]]:
    """Link 3's bar as its two conditions give it: above before + noise AND an interval above 50, which at the
    measured half-widths needs a reading of 50 + half-width; the bar is the larger, over the half-widths' range."""
    if before_per_100 is None:
        return None
    return [max(before_per_100 + noise, 50.0 + min(half_widths)), max(before_per_100 + noise, 50.0 + max(half_widths))]


def chain(before: dict, after: Optional[dict] = None, ranking_noise: Optional[float] = None, draws: int = 5000,
          before_step: Optional[int] = None) -> dict:
    """The four links before (and after) the repair sequence. `before` / `after`: dict(watch=path to the watch
    readings JSON, candidates=probe_planner_candidates result folder, reflex=probe_planner_vs_reflex result folder),
    any of them missing. Each link's reading, its noise, its jump rule and its reachability arithmetic, with and
    without body 2. before_step: the before checkpoint; a before watch file with a later report is refused.
    `draws`: the body bootstrap's draws for the planner-candidates summaries (the probe's own default is 5,000)."""
    def load(side, key, fn):
        p = (side or {}).get(key)
        return None if not p else fn(p)
    rep = dict(version=CHAIN_VERSION, generated=time.strftime("%Y-%m-%d %H:%M:%S"), before=before, after=after,
               excluded_body=EXCLUDED_BODY, tie_band=TIE_BAND, before_step=before_step,
               reachability={str(k): v for k, v in CHAIN_REACHABILITY.items()})
    wb = load(before, "watch", lambda p: watch_file(p, before_step))
    wa = load(after, "watch", lambda p: watch_file(p))
    if wb is not None and wa is not None and wa.get("last_report") is not None and wb.get("last_report") is not None \
            and int(wa["last_report"]) <= int(wb["last_report"]):
        raise ValueError("the after watch file ends at report %s, not after the before file's %s"
                         % (wa["last_report"], wb["last_report"]))
    rep["watch_files"] = dict(before=None if wb is None else wb["_pinned"], after=None if wa is None else wa["_pinned"])
    rep["watch"] = None if wb is None else chain_watch(wb, wa)
    cb = load(before, "candidates", lambda p: chain_candidates(p, draws=draws))
    ca = load(after, "candidates", lambda p: chain_candidates(p, draws=draws))
    # link 2's probe reading: the planner recheck's captured share, from the same rounds as link 3
    link2 = dict(reading="the planner recheck's captured share (%s): over every pair of the five candidates from "
                         "one saved start, sum of mean((predicted a - predicted b) x (real a - real b)) / sum of "
                         "mean((real a - real b)^2), pooled over %s, planner route, 50 ms ahead; 0 = the predicted "
                         "difference between two candidates carries none of the real difference, 1 = all of it (the "
                         "goal line), below 0 = the wrong way" % (CHAIN_CAPTURED_PROBE, ", ".join(CHAIN_CAPTURED_SENSES)),
                 replaces="the share of planner choices inside the tie band, which the band (0.0734, %.1f to %.1f "
                          "times the evaluator's predicted spread between the candidates in the nine rechecks of "
                          "399,100 to 450,525) holds at 1 whatever the forward model does" % CHAIN_TIE_BAND_OVER_SPREAD,
                 noise=CHAIN_CAPTURED_NOISE,
                 rule="a jump: the after's captured share above the before's by more than %.4f (the largest change "
                      "between the nine rechecks of 399,100 to 450,525), its 95%% body interval above the before's "
                      "reading, and its rows measured by %s with the pooled real difference above the float32 "
                      "floor" % (CHAIN_CAPTURED_NOISE, CHAIN_CAPTURED_PROBE),
                 before_probe=None if cb is None else cb.get("probe"), after_probe=None if ca is None else ca.get("probe"))
    for variant in ("all", "without_body_2"):
        b = ((cb or {}).get(variant) or {}).get("captured_share")
        a = ((ca or {}).get(variant) or {}).get("captured_share")
        link2[variant] = captured_cell(b, a if ca is not None else None)
    rep["candidate_separation"] = link2
    noise = CHAIN_RANKING_NOISE if ranking_noise is None else float(ranking_noise)
    link3 = dict(before=cb, after=ca, noise_per_100=noise, interval_half_widths=list(CHAIN_RANKING_HALF_WIDTHS),
                 rule="a jump: the goal-distance ranking rises by more than %.1f per 100 pairs (the largest change "
                      "between the pre-repair rounds) and the after's 95%% body interval lies above 50 (chance); at the "
                      "pre-repair rounds' interval half-widths (%.1f-%.1f) that needs a reading of %.0f-%.0f" % (
                          noise, min(CHAIN_RANKING_HALF_WIDTHS), max(CHAIN_RANKING_HALF_WIDTHS),
                          50 + min(CHAIN_RANKING_HALF_WIDTHS), 50 + max(CHAIN_RANKING_HALF_WIDTHS)))
    for variant in ("all", "without_body_2"):
        b = ((cb or {}).get(variant) or {}).get("ranking_goal_distance_end") or {}
        a = ((ca or {}).get(variant) or {}).get("ranking_goal_distance_end") or {}
        cell = dict(jump_needs=ranking_bar(b.get("per_100"), noise, CHAIN_RANKING_HALF_WIDTHS))
        if b.get("per_100") is not None and a.get("per_100") is not None:
            iv = a.get("interval")
            cell.update(change=a["per_100"] - b["per_100"],
                        jump=bool(a["per_100"] - b["per_100"] > noise and iv and iv[0] > 50.0),
                        first_flat_stage=[cb[variant]["first_flat_stage"], ca[variant]["first_flat_stage"]])
        link3[variant] = cell
    rep["planner_choice_quality"] = link3
    rb, ra = load(before, "reflex", chain_reflex), load(after, "reflex", chain_reflex)
    link4 = dict(before=rb, after=ra,
                 rule="a jump: the after's planner_over_reflex (how much nearer its goal the planner's order ended than "
                      "the reflex order from the identical saved state, as a share of the mean starting distance) has "
                      "its 95% body interval entirely above 0 and its mean above the before's interval, AND the after's "
                      "planner_over_mirror (the same against the matched mirror) has its interval above 0")
    if rb and ra:
        for variant in ("all", "without_body_2"):
            b = rb[variant].get("planner_over_reflex") or {}
            a = ra[variant].get("planner_over_reflex") or {}
            m = ra[variant].get("planner_over_mirror") or {}
            if a.get("interval") and b.get("interval") and a.get("mean") is not None:
                link4[variant] = dict(change=a["mean"] - (b.get("mean") or 0.0),
                                      beats_reflex=bool(a["interval"][0] > 0 and a["mean"] > b["interval"][1]),
                                      beats_mirror=bool(m.get("interval") and m["interval"][0] > 0))
                link4[variant]["jump"] = bool(link4[variant]["beats_reflex"] and link4[variant]["beats_mirror"])
    rep["physical_improvement"] = link4
    return rep


def chain_table(rep: dict) -> str:
    """The four links in one short table (the jump read without body 2 where bodies are kept apart)."""
    out = ["Jump chain (%s), %s. Body 2 is numbered from 0 (the dashboard's body 3); the jump is read without it "
           "where the instrument keeps bodies apart." % (rep["version"], rep["generated"]), "",
           "| link | reading | before | after | jump needs | jump |", "|---|---|---|---|---|---|"]
    names = {"action_conditioned_proprioception": "1 action-conditioned prediction: share of the body-position "
                                                  "prediction error the real order removes against another body's "
                                                  "order (0 = none; all 8 pooled)",
             "sensitivity_proprioception": "1 how far another body's order moves the body-position prediction, as a "
                                           "share of the whole predicted change (all 8 pooled)",
             "efference_action_conditioned": "1 the efference head: share of each handle's movement error the real "
                                             "order removes against another body's (without body 2)",
             "candidate_spread": "2 candidate separation: predicted wellness spread of the planner's five candidates "
                                 "(0-5 scale; the evaluator's tie band is 0.0734; without body 2)"}

    def f(v, pct=False):
        if v is None:
            return "unmeasured"
        return ("%.2f%%" % (100 * v)) if pct else ("%.4g" % v)
    for key, row in (rep.get("watch") or {}).items():
        cell = row.get("without_body_2") if not row["pooled_only"] else row.get("all")
        cell = cell or {}
        pct = key in ("action_conditioned_proprioception", "sensitivity_proprioception", "efference_action_conditioned")
        needs = cell.get("jump_needs")
        needs = "unmeasured" if needs is None else ("%s %s" % ("above" if row["better"] == "up" else "below", f(needs, pct)))
        out.append("| %d | %s | %s | %s | %s | %s |" % (
            row["link"], names[key], f(cell.get("before"), pct), f(cell.get("after"), pct),
            needs, "unmeasured" if "jump" not in cell else ("YES" if cell["jump"] else "no")))
    l2 = rep.get("candidate_separation") or {}
    w2 = l2.get("without_body_2") or {}

    def per_100(v):
        text = "%.2f" % (100 * v)
        return "0.00" if text == "-0.00" else text

    def cs(side):
        r = ((side or {}).get("without_body_2") or {}).get("captured_share") or {}
        if r.get("measurable") is False:
            return "not measurable (real difference within the float32 floor)"
        if r.get("value") is None:
            return "unmeasured"
        senses = ", ".join("%s %s" % (m, "not measurable" if b.get("measurable") is False else
                                      ("unmeasured" if b.get("value") is None else per_100(b["value"])))
                           for m, b in (r.get("senses") or {}).items())
        version = "" if r.get("measurable") is not None else "; rows measured before %s" % CHAIN_CAPTURED_PROBE
        return "%s in 100 (%s; %s%s)" % (per_100(r["value"]), "interval unmeasured" if not r.get("interval") else
                                         "%s-%s" % tuple(per_100(x) for x in r["interval"]), senses, version)
    planner = rep.get("planner_choice_quality") or {}
    out.append("| 2 | candidate separation: the planner recheck's captured share (probe %s; it replaces the share of "
               "choices inside the tie band, which the band, %.0f to %.0f times the predicted spread between the "
               "candidates, holds at every choice whatever the forward model does): of the real difference between "
               "two candidates' next tokens, pooled over vision, tactile, audio and proprioception 50 ms ahead, the "
               "part the predicted difference carries along its direction (0 = none, 100 = all of it, the goal line; "
               "below 0 = the wrong way; without body 2) | %s | %s | %s | %s |" % (
                   CHAIN_CAPTURED_PROBE, CHAIN_TIE_BAND_OVER_SPREAD[0], CHAIN_TIE_BAND_OVER_SPREAD[1],
                   cs(planner.get("before")), cs(planner.get("after")),
                   "unmeasured" if w2.get("jump_needs") is None else
                   "above %s in 100 (a rise of more than %s) with the interval above the before's reading" % (
                       per_100(w2["jump_needs"]), per_100(l2.get("noise", CHAIN_CAPTURED_NOISE))),
                   "unmeasured" if "jump" not in w2 else ("YES" if w2["jump"] else "no")))
    l3 = rep.get("planner_choice_quality") or {}

    def rk(side, variant):
        r = ((side or {}).get(variant) or {}).get("ranking_goal_distance_end") or {}
        return "unmeasured" if r.get("per_100") is None else "%.1f in 100 (%s)" % (
            r["per_100"], "interval unmeasured" if not r.get("interval") else "%.1f-%.1f" % tuple(r["interval"]))
    w3 = l3.get("without_body_2") or {}
    bar = w3.get("jump_needs")
    out.append("| 3 | planner choice quality: out of 100 pairs of candidates, how many the forecast orders as their "
               "actual distances to the goal are ordered (50 = chance; without body 2) | %s | %s | %s (a rise of more "
               "than %.1f and an interval above 50) | %s |" % (
                   rk(l3.get("before"), "without_body_2"), rk(l3.get("after"), "without_body_2"),
                   "unmeasured" if not bar else "above %.1f-%.1f" % tuple(bar), l3.get("noise_per_100", 0.0),
                   "unmeasured" if "jump" not in w3 else ("YES" if w3["jump"] else "no")))
    l4 = rep.get("physical_improvement") or {}

    def pv(side, key="planner_over_reflex"):
        r = ((side or {}).get("without_body_2") or {}).get(key) or {}
        return "unmeasured" if r.get("mean") is None else "%+.2f%% (%s)" % (
            100 * r["mean"], "interval unmeasured" if not r.get("interval") else
            "%+.2f%% to %+.2f%%" % (100 * r["interval"][0], 100 * r["interval"][1]))
    out.append("| 4 | physical improvement over the reflex controller: how much nearer its own goal the planner's order "
               "ended than the reflex order from the identical saved state, as a share of the mean starting distance "
               "(0 = no nearer; without body 2); beside it the same against the matched mirror (the control) | %s; "
               "mirror %s | %s; mirror %s | an interval above 0 and a mean above the before's interval, and the "
               "mirror's interval above 0 | %s |" % (
                   pv(l4.get("before")), pv(l4.get("before"), "planner_over_mirror"),
                   pv(l4.get("after")), pv(l4.get("after"), "planner_over_mirror"),
                   "unmeasured" if "without_body_2" not in l4 else ("YES" if l4["without_body_2"]["jump"] else "no")))
    out += ["", "Reachability arithmetic of each rule:"]
    out += ["- link %s: %s" % (k, v) for k, v in sorted((rep.get("reachability") or {}).items())]
    return "\n".join(out) + "\n"


# ================================================================================================ the CLI
def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                   encoding="utf-8")
    tmp.replace(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mimo_bridge.tools.regression_pack", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="the pack on two checkpoints or step ranges: one JSON and one table")
    c.add_argument("--before", required=True)
    c.add_argument("--after", required=True)
    c.add_argument("--switch-step", type=int, default=None,
                   help="the update the repair was switched on: the after side reads only what came after it "
                        "(default: the before side's step)")
    c.add_argument("--window", type=int, default=WINDOW_STEPS)
    c.add_argument("--no-model", action="store_true", help="readings 1-2 left unmeasured (no model is built)")
    c.add_argument("--out", default=None)
    c.add_argument("--cache", default=None, help="checkpoint readings cache (default <out>/checkpoints)")
    c.add_argument("--saved", default=None)
    c.add_argument("--lines", default=None, help="the noise file (default %s)" % NOISE_FILE.name)
    c.add_argument("--slots", type=int, default=MODEL_SLOTS)
    c.add_argument("--no-internals", action="store_true",
                   help="leave out the memory deploy's readings (attention spread, unit liveness, the memory's worth)")
    c.add_argument("--deploy-lines", default=None, help="their noise file (default memory_deploy_readings.NOISE_FILE)")
    k = sub.add_parser("checkpoint", help="readings 1-5 of checkpoints into the cache (the pause job)")
    k.add_argument("checkpoints", nargs="+", help="checkpoint folders or bare steps of the lineage")
    k.add_argument("--out", required=True)
    k.add_argument("--after-switch", type=int, default=None,
                   help="read only the ring moments lived after this update (the watch-end side of a repair)")
    k.add_argument("--no-model", action="store_true")
    k.add_argument("--slots", type=int, default=MODEL_SLOTS)
    k.add_argument("--no-internals", action="store_true")
    k.add_argument("--saved", default=None)
    n = sub.add_parser("noise", help="measure the noise the verdicts read: repeats on one checkpoint, the observer's rows")
    n.add_argument("--checkpoint", required=True, help="a checkpoint folder or a bare step of the lineage")
    n.add_argument("--observer-from", type=int, required=True)
    n.add_argument("--observer-to", type=int, required=True)
    n.add_argument("--window", type=int, default=WINDOW_STEPS)
    n.add_argument("--out", required=True)
    n.add_argument("--cache", required=True)
    n.add_argument("--saved", default=None)
    n.add_argument("--no-model", action="store_true")
    n.add_argument("--slots", type=int, default=MODEL_SLOTS)
    j = sub.add_parser("chain", help="the jump chain's four links before (and after) the repair sequence")
    for side in ("before", "after"):
        j.add_argument("--%s-watch" % side, default=None, help="persistent_memory_watch_readings.py output")
        j.add_argument("--%s-candidates" % side, default=None, help="a probe_planner_candidates result folder")
        j.add_argument("--%s-reflex" % side, default=None, help="a probe_planner_vs_reflex result folder")
    j.add_argument("--before-step", type=int, default=None,
                   help="the before checkpoint's step: a before watch file with a later report is refused")
    j.add_argument("--ranking-noise", type=float, default=None)
    j.add_argument("--out", required=True)
    j.add_argument("--saved", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "chain":
        sides = {s: dict(watch=getattr(a, s + "_watch"), candidates=getattr(a, s + "_candidates"),
                         reflex=getattr(a, s + "_reflex")) for s in ("before", "after")}
        after = sides["after"] if any(sides["after"].values()) else None
        rep = chain(sides["before"], after, ranking_noise=a.ranking_noise, before_step=a.before_step)
        out = Path(a.out)
        # the watch files the links were read from, copied beside the report: a later regeneration cannot move them
        for s in ("before", "after"):
            if sides[s]["watch"]:
                out.mkdir(parents=True, exist_ok=True)
                (out / ("%s_watch.json" % s)).write_bytes(Path(sides[s]["watch"]).read_bytes())
        _write(out / "jump_chain.json", rep)
        text = chain_table(rep)
        (out / "jump_chain.md").write_text(text, encoding="utf-8")
        print(text, flush=True)
        return 0
    saved = saved_root(getattr(a, "saved", None))
    if a.cmd == "checkpoint":
        for text in a.checkpoints:
            side = parse_side(text, WINDOW_STEPS, saved, switch_step=a.after_switch)
            r = checkpoint_readings(side["checkpoint"], model=not a.no_model, n_slots=a.slots, cache_dir=Path(a.out),
                                    internals=not a.no_internals, newest=side.get("newest"))
            print(json.dumps(dict(checkpoint=side["checkpoint"], step=r["step"], newest=r.get("newest"),
                                  seconds=r.get("seconds"), companion_load_s=r.get("companion_load_s"),
                                  model_load_s=r.get("model_load_s"), model_s=(r.get("model") or {}).get("seconds"),
                                  peak_commit_gb=r.get("peak_commit_gb"),
                                  memory_deploy_s=(r.get("memory_deploy") or {}).get("seconds"))), flush=True)
        return 0
    if a.cmd == "noise":
        ck = parse_side(a.checkpoint, a.window, saved)["checkpoint"]
        r = noise(ck, a.observer_from, a.observer_to, saved=saved, window=a.window, model=not a.no_model,
                  cache_dir=Path(a.cache), n_slots=a.slots)
        _write(Path(a.out), r)
        print(json.dumps(dict(repeat=r["lines"]["repeat"], observer_counts=r["lines"]["observer_counts"],
                              cost=r["cost"]), indent=1), flush=True)
        return 0
    before = parse_side(a.before, a.window, saved)
    switch = int(before["hi"]) if a.switch_step is None else int(a.switch_step)
    after = parse_side(a.after, a.window, saved, switch_step=switch)
    out = Path(a.out) if a.out else saved.parent / "gof_observer_20260924" / "regression_pack" / time.strftime("%Y%m%d_%H%M%S")
    cache = Path(a.cache) if a.cache else out / "checkpoints"
    from mimo_bridge.tools import memory_deploy_readings as MDR
    report = compare(before, after, saved=saved, model=not a.no_model, cache_dir=cache,
                     lines=load_lines(Path(a.lines)) if a.lines else None, n_slots=a.slots,
                     internals=not a.no_internals,
                     deploy_lines=MDR.load_lines(Path(a.deploy_lines)) if a.deploy_lines else None)
    _write(out / "regression_pack.json", report)
    text = table(report)
    (out / "regression_pack.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print("written: %s" % (out / "regression_pack.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
