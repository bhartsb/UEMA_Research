"""The kill-line watcher of the motor fixes F1 (--eye-row-convention physical) and F4
(--executed-fields-only on), which deploy together: it judges the deploy plan's kill lines over the watch's
6,000 updates and writes the stop itself.

    python docs/gain_of_function_v4/motor_fixes_watch.py --deploy-run <run folder> --ensure
        [--baseline-run <pre-deploy run folder>] [--after-step N] [--poll 60] [--hours 0] [--fresh]
        [--watch-steps 6000] [--anchor-before <result.json>] [--anchor-floor <result.json>]
        [--anchor-readings <folder>[,<folder>...]]
    python docs/gain_of_function_v4/motor_fixes_watch.py --deploy-run <run folder> --end "<reason>"

--ensure starts the watcher as a detached process (its output in <deploy run>/motor_fixes_watch.log)
unless one for this deploy run is alive and polling; run it at the deploy and at every hourly check.
Without --ensure the watch runs in the calling process (--once: one poll). --end asks the watch to end
(at a revert): the watcher reads <deploy run>/motor_fixes_watch_end.json at its next poll, records phase
'ended' and exits (code 7), and --ensure starts none after it.

WHY A PROGRAM OF ITS OWN (review of 2026-09-25, finding 1). The hourly watcher
(docs/h84_launch_2026-09-18/hourly_watch.py, declared_criteria) compares ONE stats field with a
constant, as a mean over the last hour. The deploy's kill lines need the difference of two fields (a
head's error against its shuffled-order control), a count of consecutive 100-step windows, and the
observer's eye joint angles per block. The deploy plan is
docs/gain_of_function_v4/agents/motor_fixes_eye_fields_2026-09-25.md.

THE LINES (declared in the deploy plan before the deploy; crossing one is a stop):
  body head    efference.body_cm_err_next at or above efference.body_cm_shuffled_next: the efference
               body head predicts the next 50 ms no better from the body's own order than from another
               body's order (pre-deploy: below it in every window; smallest margin 0.097 cm over the
               353 full windows of segments 36-40, 0.070 cm on a 50-step stop record)
  gaze head    efference.gaze_deg_err[0] at or above efference.gaze_deg_shuffled[0], the same for the
               eyes at 50 ms (pre-deploy: below it in every window, smallest margin 0.95 degrees)
  causal error causal_pe above 0.031, the highest pre-deploy window (median 0.0165)
  following    cmd_response_cos_held below 0.335, the lowest pre-deploy window
  Each needs 3 CONSECUTIVE FULL windows past its line. A full window is a stats record whose step is
  --report-every (the run's own, 100) after the record before it in the same launch; the shorter record
  the trainer writes at every stop (25 to 75 steps: the steps since the last report) neither counts nor
  breaks a run (review of the deploy branch, finding 2: judged as a window, it could stop the run on a
  25-step reading, inside the ability scheduler's saved window). The first record of every launch at or
  after N is not judged either (N = the switch step): a launch rebuilds its context from the episode
  companion. A full window that lacks the value breaks the run; counts restart in each launch.
  eye stop     the vertical eye angle within 2 degrees of a joint stop in more than 5 of every 100
               observer records of a block of 1,700 model steps (all 8 bodies), blocks counted from N
               over the JOINED records of the deploy run and its continuations (review of the deploy
               branch, finding 1: cut per run folder, 46 to 54 of every 100 recorded steps fell in no
               judged block). A block is judged once the chain's observer records pass its end by one
               chunk (100 steps: every body's chunk holding the block's last step is then written), and
               judged once. A block complete in steps but under 2,000 records, and a stretch of the
               stats clock with no observer record, are printed: the line is blind there.

  anchored eye (the fourth review's finding 4, fixed in its fix round: no line above could see the eye skill
               F1 is for, which is worth about 1 in 100 of the picture's error). The quantity: how much less
               error the model's 50 ms picture prediction has when told the eye command actually sent than
               when told the eyes were held still (docs/gain_of_function_v3/probe_saved_gaze.py,
               summary.saved_starts.50ms.visual_prediction_mse.error_reduction_fraction; 0 in 100: the eye
               command tells the prediction nothing; the top of the scale, the share of the picture's change
               eye movement causes, is unmeasured). Every reading of a checkpoint of a watched run at or
               after N is judged once: the ability scheduler's own gaze test (the switch's flag change
               triggers it, within its 3,000-update ride-along), and the deploy's own readings
               (--anchor-readings: motor_fixes_eye_anchor.py). A reading under the LINE is a stop: the lower
               of the reading before the switch (--anchor-before, on the deploy stop's checkpoint) and the
               reading right after it with no training (--anchor-floor: the same checkpoint loaded
               'physical', its stored eye rows rewritten), less ANCHOR_DROP (3.1 in 100).
               REACHABILITY. The expected transient: at the switch the transformer's motor token reads the
               eye row's raw numbers, whose vertical part now means the mirrored elevation, while the gaze
               copy and the gaze head read the elevation through the convention; so the reading may drop at
               once, by up to the motor token's share of the skill, and recover as the lessons relearn (replay
               from post-switch moments only: the eye row's replay contract). --anchor-floor measures that
               drop before any learning, so the line sits 3.1 in 100 under it and does not depend on the
               recovery's time constant, which is unmeasured; a further fall while relearning, past the line,
               is a stop. The noise on the judged reading: two readings of the lineage 3,075 to 4,600 steps
               apart differ with a standard deviation of 1.57 in 100 (the 10
               successive differences of the scheduler's 11 readings from 380,550 to 414,200: 1.35, 3.15,
               1.51, 0.21, 1.24, -0.22, 2.61, 0.97, 0.47, 0.87, 1.27 in 100); the line sits 1.96 of those
               below, so a healthy reading falls under it 2.5 times in 100, and a watch that judges two
               readings (the scheduler's triggered one and the end reading) stops falsely about 5 times in 100.
               WHAT IT CAN SEE: a reading 3.1 in 100 under both references. The whole skill before the switch
               is about 1 in 100 (median of the last five readings 0.97), so this line cannot see the skill's
               loss alone: it sees the eye command turning into misleading information for the picture
               prediction. The loss of the skill itself is judged by paired arms from one checkpoint (the
               trial's L4, docs/gain_of_function_v4/agents/eye_fixes_review4_2026-09-26.md, question 4).
               Without --anchor-floor and --anchor-before the line judges nothing and every poll says so.

THE TIMELINE. The deploy run and every continuation of it form one chain: a folder beside it whose
continuation_receipt.json names a watched run as its parent_run (the supervisor's relaunch), or whose
args.json resumes from a watched run's checkpoint (a relaunch by hand, the way training continues
after a stop that requires repair: the supervisor never relaunches such a run). Each run owns the
steps from its own start (N for the deploy run, the resumed checkpoint's step for a continuation) to
the next run's start: a crash resume abandons the older run's steps after that checkpoint, and they
are not judged. A relaunch whose args.json sets either deploy setting (--eye-row-convention,
--executed-fields-only) otherwise than the deploy run's is a REVERT: it is not followed, nor anything
continuing it, and every poll prints it as NOT FOLLOWED (the third review, point G: a revert relaunched
from a watched run's checkpoint was followed and judged). End the watch with --end when reverting. F4 and
F1 deploy together (the operator's decision of 2026-09-25), so one watcher watches the combined form; the
single status file the dashboards read is that watcher's.

THE STOP, and when it waits. The hourly watcher's three writes (stop_run): the supervisor pause in
Saved/mimo_bridge/supervisor_control.json (read, given cmd pause, and replaced with every other key
kept; an UNREADABLE file is left as it is, since its other keys cannot be kept), <run>/stop_reason.json
with resume_policy repair_required, and <run>/control.json {"cmd": "snapshot_and_stop"} as UTF-8
without a byte-order mark (no auto_resume: the consumed stop also forbids an automatic relaunch), each
carrying the stop's id and this program's name, the two run files never replacing a file of that name.
Written into the ACTIVE run only, when it is one of the watched runs. NEVER while the ability scheduler
holds a saved window ('preparing', 'saving', 'ready') or a review hold ('held for review'), or its
status file is unreadable (a reader can meet it mid-replace): a changed supervisor control cancels the
window's remaining comparisons, and during a review hold it records the failed round as reviewed (the
first review's finding 2, reproduced). A task the scheduler runs outside a saved window is no hold (the
second review, finding 6). Nor while supervisor_control.json is unreadable or carries no cmd (the
supervisor reads both as a pause), or the active run holds an unconsumed control.json (a coordinated
pause's own stop), or a supervisor pause is in force while no trainer of the active run runs. A PLAIN
pause while the active run's trainer runs (neither a scheduler window nor a coordinated pause, which have
their own marks above: the deploy procedure's own steps 4 to 6 run the deploy trainer under the
coordinator's pause) is no hold (the third review, finding E): the stop's control.json and
stop_reason.json are written and the pause is kept as it is (the stop's record says pause_kept; such a
stop is never 'released'). While a hold stands, the stop WAITS: it is printed first in every poll's
lines and by --ensure, with what it waits for and since when, and written to the status file the
dashboards read (Saved/mimo_bridge/motor_fixes_watch_status.json); it is written at the first poll where
none of the holds stands.

A WRITTEN STOP IS CONFIRMED (the second review, finding 2; the third review, finding A). The watch goes
on polling after the write. The stop is CONFIRMED when no trainer of a watched run launched at or before
it remains and either a control_done_*.json holding its id is in a watched run (the trainer renames
control.json so when it reads it), or it LANDED IN A STOPPED RUN: no trainer read it, no such trainer
ran on 2 polls in a row ending at least 60 s after the write, and a stop_reason.json (repair_required)
is in a run it was written to. The second case is the common one after a stop that waited for a hold
ending with training stopped (a scheduler window, the pause runner's pause): the watcher's first poll
after the hold clears comes before the supervisor's relaunch in 37 to 53 of every 60 cases (the third
review, measured), and the stop's pause then keeps the supervisor from relaunching. Until confirmed, at
every poll it is written into each watched run that runs a trainer launched at or before the stop was
written, and into the active run when that is such a run: the supervisor may relaunch a continuation
between the watcher's read of the active run and its write, and the ability scheduler's snapshot
request (written with a replace) may replace the watcher's control.json, the trainer then reading that
instead. Never over a control.json already there, and never into a run launched after the stop (a
continuation someone started after it is their decision). The stop is RELEASED when the supervisor pause
it wrote was replaced (someone lifted or rewrote it). A confirmed or released stop does NOT end the
watch (the third review, finding A: it did, and a deploy continued after the stop then ran unwatched
until the next hourly --ensure): the stop's crossing run is not judged again, and every continuation is
judged from its first full windows.

ONE WATCHER, RESTART AND STAYING ALIVE (the first review's finding 6; the second review's finding 3).
One watcher per deploy run: the process holds <deploy run>/motor_fixes_watch.lock (the OwnerLock the
supervisor and the ability scheduler use) for its life, and a second one ends at once. The state file
<deploy run>/motor_fixes_watch.json is written at every poll MERGED with what it holds (the union of the
written stops and the judged eye blocks; a stop in both keeps every field either set), and read back at
every poll: the heartbeat, the process id, every stop written with its confirmation, every eye block
judged. A watcher started again for the same deploy run reads it: the windows of a written stop's
CROSSING run at or before the stop's step are not judged again (only that run's: a continuation from an
older checkpoint re-runs those steps and is judged) and judged blocks (keyed by their first step and the
runs whose records they held) are not judged again, so a restart after a stop that is continued (by the
supervisor or by hand, above) never stops the continuation on the old crossing (--fresh ignores the
file). Nothing else keeps this program alive (the supervisor restarts only hourly_watch.py), so the
procedure runs --ensure at the deploy, at every hourly check and after a continued stop: it starts a
detached watcher unless one holds the lock, and names a watcher that holds it without polling for 5
polls (it is not replaced while it lives). --hours 0 (the default) watches until the watch is complete.

ONE WATCH OF 6,000 UPDATES (the operator's decision of 2026-09-25, WORKLOG b28dcdbc2: F4 and F1 are one
semantic correction and switch together at one restart; one deploy stop, one watch). --watch-steps
(default 6,000, the replay ring's turnover) bounds the watch: the stats windows after N + 6,000 are
outside it, and so are the eye blocks that start at or after it (the last block inside starts before it
and ends up to 1,700 steps after, judged once its records are in). When every window inside is judged and
every eye block inside is judged, or settled blind because training ran a whole block past its end without
its observer records, the phase is 'complete', the state file keeps it (with the blind blocks and every
stop written during the watch), the watch ends, and --ensure starts no watcher for a complete watch. The
kill lines are judged on the running deploy, which carries both switches: the combined form the
switch-jump job's gate measured.

THE HOURLY WATCHER'S OWN-BASELINE RULES (review of the first branch, finding 1). With --baseline-run,
the pre-deploy run's watcher_baselines.json is copied into each watched run that has none yet (marked
copied_from), so the hourly watcher judges the switch's first hours against the levels before it.
Only into runs that begin before N + 6,000 (the replay ring's turnover). The hourly watcher writes its
own file at its first check when it loaded none, so the copy must land before it starts; the
procedure starts this watcher before the supervisor is reattached, and the supervisor starts the
hourly watcher only then.

ONE SWITCH AT A TIME (review of the deploy branch, finding 5). The decoder's position rows
(--position-rows, audit F3) and the limb flags (--parent-relative-targets, --reach-projection) change
the body's control or the ledger's rows in ways the switch-jump job cannot measure on stored outcomes,
and the following line's pre-deploy band does not hold under them: the deploy plan switches them in a
relaunch of their own. At every poll the deploy run's args.json is compared with the args.json of the
run it resumed from (through the checkpoint it names, wherever the archiver moved it; else
--baseline-run's), and a change of any of the three is printed and kept in the state file as NOT
COVERED. It is not a kill line and writes no stop.

MEMORY (review finding 7): each stats record is kept as the eight numbers the lines read.
NUMBERS A PERSON READS (DESIGN.md section 11): margins in cm and degrees, the causal error and the
following reading as a percentage of their line, the eye-stop share as records per 100.
Tests: tests/test_motor_fixes_watch.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import motor_fixes_baselines as MB  # noqa: E402  (also puts Source/Python on the path)

SAVED = ROOT / "Saved" / "mimo_bridge"
REPORT_EVERY = 100
CONSECUTIVE = 3
BODY_LINE_CM = 0.0            # margin = shuffled - error; at or under 0 = no better than another order
GAZE_LINE_DEG = 0.0
CAUSAL_PE_LINE = 0.031
FOLLOW_LINE = 0.335
EYE_STOP_LINE_PER_100 = MB.EYE_STOP_LINE_PER_100
EYE_BLOCK = MB.BLOCK
CHUNK_STEPS = 100             # the observer recorder writes one chunk per body per 100 steps
REPLAY_TURNOVER_STEPS = 6000
# THE DEPLOY'S ONE WATCH (the operator's decision of 2026-09-25, WORKLOG b28dcdbc2: F4 and F1 switch together,
# one deploy stop, one watch): the 6,000 updates after the switch, the replay ring's turnover
WATCH_STEPS = REPLAY_TURNOVER_STEPS
STATE_NAME = "motor_fixes_watch.json"
LOG_NAME = "motor_fixes_watch.log"
LOCK_NAME = "motor_fixes_watch.lock"          # in the deploy run: one watcher per deploy run
STATUS_NAME = "motor_fixes_watch_status.json"  # in Saved/mimo_bridge: what the dashboards read
WRITER = "docs/gain_of_function_v4/motor_fixes_watch.py"
LINES = ("body head", "gaze head", "causal error", "following")
EYE_CROSSING = "the chain's observer records"   # the crossing 'run' of an eye-stop stop (no run folder)
ANCHOR_CROSSING = "the anchored eye readings"   # the crossing 'run' of an anchored-eye stop (a checkpoint's reading)
# THE ANCHORED EYE LINE (the fourth review's finding 4): the reading's noise and the line's distance under the
# lower of the two references, in the reading's own units (a share of the picture error; x 100 to print)
ANCHOR_NOISE = 0.01566        # sd of the difference of two successive lineage readings, 3,075 to 4,600 steps apart
ANCHOR_Z = 1.96               # one-sided: 2.5 healthy readings in 100 fall under the line
ANCHOR_DROP = ANCHOR_Z * ANCHOR_NOISE          # 0.0307: 3.1 in 100
ANCHOR_KEY_PREFIX = "anchor:"                   # judged readings are kept among judged_blocks with this prefix
# the ability scheduler's gaze test receipts (tools/ability_rechecks.py: <saved>/ability_rechecks/<task>_<step>_
# attempt<k>_<ns>/result.json)
SCHEDULER_GAZE_GLOB = "ability_rechecks/gaze_prediction_*/result.json"
# the ability scheduler's window states that hold training (tools/ability_comparison_window.py)
SCHEDULER_HOLDS = ("preparing", "saving", "ready", "held for review")
ALIVE_POLLS = 5               # --ensure: a watcher that polled within this many polls is alive
# a stop that landed in a stopped run (confirm_stop; the third review, finding A): confirmed after this many
# polls in a row with no trainer launched at or before it running, the last at least QUIET_SECONDS after the
# write. The supervisor launches 37 to 53 s after a hold clears and writes its receipt 0.1 s before the
# launch (measured by the third review on the 20 newest continuations of H84), so a relaunch still in flight
# at the write is running, and found, by then.
QUIET_POLLS = 2
QUIET_SECONDS = 60.0
END_NAME = "motor_fixes_watch_end.json"       # in the deploy run: --end ends the watch (a revert)
# the deploy's two settings (train_mimo.py args.json keys): a relaunch that reverts either is not the deploy's
# continuation (the third review, point G)
DEPLOY_SETTINGS = ("eye_row_convention", "executed_fields_only")


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def compact(r):
    """The fields the lines read, from a stats record (a compact record passes unchanged): 0.44 MB of
    parsed record becomes eight numbers (review finding 7)."""
    if r.get("_compact"):
        return r
    if r.get("type") == "motor_interface_change":
        return {"_compact": True, "type": r.get("type"), "step": r.get("step"),
                "timestamp": r.get("timestamp"), "changes": r.get("changes")}
    eff = r.get("efference") or {}

    def first(v):
        return v[0] if isinstance(v, list) and v else None
    return {"_compact": True, "type": r.get("type"), "step": r.get("step"),
            "body_err": eff.get("body_cm_err_next"), "body_shuf": eff.get("body_cm_shuffled_next"),
            "gaze_err": first(eff.get("gaze_deg_err")), "gaze_shuf": first(eff.get("gaze_deg_shuffled")),
            "causal_pe": r.get("causal_pe"), "cos_held": r.get("cmd_response_cos_held")}


def line_values(r):
    """{line: (value, crossed)} of one stats record; value None when the record lacks it. The two head
    lines' values are margins (the control's error minus the head's own), crossed at or under 0."""
    c = compact(r)
    out = {}
    for line, e, s, edge in (("body head", c["body_err"], c["body_shuf"], BODY_LINE_CM),
                             ("gaze head", c["gaze_err"], c["gaze_shuf"], GAZE_LINE_DEG)):
        m = s - e if _num(e) and _num(s) else None
        out[line] = (m, m is not None and m <= edge)
    v = c["causal_pe"] if _num(c["causal_pe"]) else None
    out["causal error"] = (v, v is not None and v > CAUSAL_PE_LINE)
    v = c["cos_held"] if _num(c["cos_held"]) else None
    out["following"] = (v, v is not None and v < FOLLOW_LINE)
    return out


def describe(line, value):
    """A line's value for a person (DESIGN section 11): real units or a percentage of its line."""
    if value is None:
        return "unmeasured"
    if line in ("body head", "gaze head"):
        unit, nd = ("cm", 3) if line == "body head" else ("degrees", 2)
        if value == 0:
            return ("the error with the body's own order equals the error with another body's order "
                    "(equal or over crosses)")
        return ("the error with the body's own order %.*f %s %s the error with another body's order "
                "(equal or over crosses)" % (nd, abs(value), unit, "under" if value > 0 else "over"))
    if line == "causal error":
        return "%.0f%% of its kill line (over 100%% crosses)" % (100.0 * value / CAUSAL_PE_LINE)
    return "%.0f%% of its kill line (under 100%% crosses)" % (100.0 * value / FOLLOW_LINE)


def judge_windows(runs_records, after_step, report_every=REPORT_EVERY, consecutive=CONSECUTIVE,
                  spans=None, judged_after=None, closed=None):
    """runs_records: [(run name, [stats records in file order])]. Returns (stop reason or None, report).

    spans: {run name: (start step, end step or None)}: a run's launch began at its start (N, or the
    checkpoint a continuation resumed from) and the next run's launch at its end; the run owns the
    records after its start up to and including its end (a record at step C covers the 100 steps
    before C, all before the checkpoint at C; a crash resume abandons the older run's records after
    it). Without a span a run starts at after_step and owns every later record. judged_after: {run
    name: a written stop's step} - that run's windows at or before it are not judged again, and only
    that run's (the deploy branch's second review, point 7: a continuation from an OLDER checkpoint
    re-runs those steps, and they are judged); a plain number applies to every run. closed: {run name:
    the step of the written stop that closed it} - a written stop's CROSSING run is not judged again at
    all once the stop is written (the third review, finding A: the watch goes on after a stop, and the
    windows a waiting stop let that run go on writing belong to the crossing it was stopped for).
    Per launch: the first record is not judged; a record is a FULL window when its step is report_every
    after the previous record's (the launch's start for the first); a shorter record neither counts nor
    breaks a run; every line counts consecutive full windows past it, and the window that makes a count
    reach `consecutive` is a stop."""
    report = {"runs": [], "judged_windows": 0, "short_records": 0}
    for run, records in runs_records:
        if closed and run in closed:
            report["runs"].append(dict(run=run, windows=sum(1 for r in records if r.get("type") == "mimo"),
                                       judged=0, short=0, counts={k: 0 for k in LINES}, last={},
                                       last_step=None, closed_by_stop_at=closed[run]))
            continue
        lo, hi = (spans or {}).get(run, (after_step, None))
        lo = int(after_step) if lo is None else int(lo)
        done = judged_after.get(run) if isinstance(judged_after, dict) else judged_after
        floor = max(int(after_step), int(done)) if done is not None else int(after_step)
        rows = sorted((compact(r) for r in records if r.get("type") == "mimo"
                       and isinstance(r.get("step"), int) and r["step"] > max(lo, int(after_step))
                       and (hi is None or r["step"] <= hi)), key=lambda r: r["step"])
        counts = {k: 0 for k in LINES}
        last, prev, judged, short = {}, lo, 0, 0
        for i, r in enumerate(rows):
            gap = r["step"] - prev
            prev = r["step"]
            if i == 0:
                continue                           # the launch's own first window
            if gap != int(report_every):
                short += 1                         # a stop's short record: neither counts nor breaks
                continue
            if r["step"] <= floor:
                continue                           # judged before a written stop
            judged += 1
            for line, (value, crossed) in line_values(r).items():
                counts[line] = counts[line] + 1 if crossed else 0
                last[line] = value
                if counts[line] >= consecutive:
                    report["judged_windows"] += judged
                    report["short_records"] += short
                    report["runs"].append(dict(run=run, windows=len(rows), judged=judged, short=short,
                                               counts=dict(counts), last=last, last_step=r["step"]))
                    return ("motor fixes kill line crossed: %s, %d full windows running up to step %d in %s; "
                            "last window: %s" % (line, counts[line], r["step"], run,
                                                 describe(line, value))), report
        report["judged_windows"] += judged
        report["short_records"] += short
        report["runs"].append(dict(run=run, windows=len(rows), judged=judged, short=short,
                                   counts=dict(counts), last={k: last.get(k) for k in LINES},
                                   last_step=rows[-1]["step"] if rows else None))
    return None, report


def stop_step_of(report):
    """The step of the window that completed a crossing (the last judged run's last step)."""
    runs = report.get("runs") or []
    return runs[-1].get("last_step") if runs else None


def crossing_run_of(report):
    """The run whose windows completed a crossing (the last judged run)."""
    runs = report.get("runs") or []
    return runs[-1].get("run") if runs else None


# ------------------------------------------------------------------ the eye-stop line, over the chain
def chain_eye_series(spans_in_order, after_step):
    """The observer records of every watched run on the chain's timeline, joined: each run's records
    from its start (the first model step it ran: N, or the checkpoint step it resumed from) up to the
    next run's start, excluded (the trainer records step k before it counts it done, so a launch that
    saved at step C recorded steps up to C - 1, and the launch resumed from C records C first). (steps,
    horizontal deg, vertical deg, horizontal range, vertical range, the run name of every record) or
    None."""
    S, H, V, R, rh, rv = [], [], [], [], None, None
    for run, (lo, hi) in spans_in_order:
        got = MB.eye_series(run, max(int(after_step), int(lo if lo is not None else after_step)))
        if got is None:
            continue
        s, h, v, rh_, rv_ = got
        keep = np.ones(len(s), dtype=bool) if hi is None else s < hi
        if keep.any():
            S.append(s[keep]), H.append(h[keep]), V.append(v[keep])
            R.append(np.full(int(keep.sum()), Path(str(run)).name, dtype=object))
            rh, rv = rh_, rv_
    if not S:
        return None
    S, H, V, R = np.concatenate(S), np.concatenate(H), np.concatenate(V), np.concatenate(R)
    order = np.argsort(S, kind="stable")
    return S[order], H[order], V[order], rh, rv, R[order]


def block_key(first_step, runs=None):
    """A judged eye block's key: its first step and the runs whose records it held. A continuation from
    an older checkpoint re-runs a block with other records, so that block is judged again (the deploy
    branch's second review, point 7)."""
    return "%d@%s" % (int(first_step), "+".join(sorted(runs))) if runs else str(int(first_step))


def judge_eye_blocks(series, after_step, judged=None, block=EYE_BLOCK, line=EYE_STOP_LINE_PER_100,
                     min_records=MB.MIN_BLOCK_RECORDS, stats_clock=None, label="the chain", before=None):
    """The eye-stop line on the chain's joined observer records. Blocks start at after_step; a block
    is complete once the records pass its end by one chunk; each complete block with enough records is
    judged once: a block under the line has its key (block_key: its first step and the runs whose
    records it held) recorded in `judged`; a block over it is returned as the stop (its key in the row's
    'stop_key') and recorded only when the stop is written, so a stop that has to wait is found again
    at the next poll. before: the watch's end (N + --watch-steps): blocks that start at or after it are
    outside the watch and neither judged nor returned. Returns (stop reason or None, rows, blind): rows =
    every block with its state; blind = the stretches the line could not judge."""
    judged = judged if judged is not None else set()
    blind = []
    if series is None:
        if stats_clock is not None and stats_clock - int(after_step) >= block:
            blind.append("no observer record since step %d while training reached step %d: the eye-stop "
                         "line is blind" % (int(after_step), int(stats_clock)))
        return None, [], blind
    S, H, V, rh, rv = series[:5]
    R = series[5] if len(series) > 5 else None
    top = int(S.max())
    rows = MB.eye_blocks(S, H, V, rh, rv, int(after_step), block=block, min_records=min_records)
    if before is not None:
        rows = [b for b in rows if b["first_step"] < int(before)]
    why = None
    for b in rows:
        end = b["first_step"] + block - 1
        b["complete"] = bool(top >= end + CHUNK_STEPS)
        v = b.get("vertical_near_stop_per_100")
        if not b["complete"]:
            continue
        if v is None:
            b["judged"] = False
            blind.append("the block of steps %d to %d holds %d observer records, under the %d a block "
                         "needs: not judged" % (b["first_step"], end, b["records"], min_records))
            continue
        b["judged"] = True
        runs = None
        if R is not None:
            inside = (S >= b["first_step"]) & (S <= end)
            runs = sorted(set(R[inside].tolist()))
        key = block_key(b["first_step"], runs)
        b["key"] = key
        if key in judged:
            continue
        if v > line:
            if why is None:
                why = ("motor fixes kill line crossed: eye stop - in the block of steps %d to %d of %s the "
                       "vertical eye angle sat within %g degrees of a joint stop in %.1f of every 100 "
                       "records (the line: more than %g)"
                       % (b["first_step"], end, label, MB.STOP_BAND_DEG, v, line))
                b["stop_step"], b["stop_key"] = end, key
            continue
        judged.add(key)
    clock_in_watch = stats_clock if before is None or stats_clock is None else min(int(stats_clock), int(before))
    if clock_in_watch is not None and clock_in_watch - top >= block:
        blind.append("the newest observer record is at step %d while training reached step %d: the "
                     "eye-stop line is blind from there" % (top, int(clock_in_watch)))
    return why, rows, blind


def anchor_value(result):
    """The anchored eye reading of a probe_saved_gaze.py result (a dict): how much less error the 50 ms picture
    prediction has with the eye command sent than with the eyes held still, as a share (x 100: in 100), or None."""
    try:
        v = result["summary"]["saved_starts"]["50ms"]["visual_prediction_mse"]["error_reduction_fraction"]
    except (KeyError, TypeError):
        return None
    return float(v) if _num(v) else None


def checkpoint_of(path):
    """(the run folder's name, the step) of a checkpoint path (<run>/checkpoints/ckpt_step_<k>, wherever the
    archiver moved it), or (None, None)."""
    p = Path(str(path or ""))
    name = p.name
    if not name.startswith("ckpt_step_") or not name[len("ckpt_step_"):].isdigit():
        return None, None
    run = p.parent.parent.name if p.parent.name == "checkpoints" else None
    return run, int(name[len("ckpt_step_"):])


def anchor_readings(saved, folders=()):
    """Every anchored eye reading on disk: the ability scheduler's gaze test results (SCHEDULER_GAZE_GLOB under
    `saved`) and every result.json under the deploy's own readings folders. [dict(key, path, run, step, value)],
    each keyed by its result file."""
    paths = sorted(Path(saved).glob(SCHEDULER_GAZE_GLOB))
    for f in folders or ():
        if f and Path(f).is_dir():
            paths += sorted(Path(f).glob("**/result.json"))
    out = []
    for p in paths:
        r = _read_json(p)
        if not isinstance(r, dict):
            continue
        run, step = checkpoint_of(r.get("checkpoint"))
        value = anchor_value(r)
        if run is None or value is None:
            continue
        out.append(dict(key=ANCHOR_KEY_PREFIX + str(Path(p).resolve()), path=str(p), run=run, step=step,
                        value=value))
    return out


def anchor_references(before_path, floor_path):
    """(the line or None, the record): the lower of the reading before the switch and the reading right after
    it with no training, less ANCHOR_DROP; None with the reason when either reading is missing or unreadable."""
    before = _read_json(before_path) if before_path else None
    floor = _read_json(floor_path) if floor_path else None
    b = anchor_value(before) if isinstance(before, dict) else None
    f = anchor_value(floor) if isinstance(floor, dict) else None
    rec = dict(before=b, before_path=before_path, floor=f, floor_path=floor_path, drop=ANCHOR_DROP,
               noise=ANCHOR_NOISE, z=ANCHOR_Z)
    if b is None or f is None:
        rec["why_blind"] = ("the anchored eye line judges nothing: %s" % "; ".join(
            x for x in ("no readable reading before the switch (--anchor-before)" if b is None else None,
                        "no readable reading right after it with no training (--anchor-floor)" if f is None
                        else None) if x))
        return None, rec
    rec["line"] = min(b, f) - ANCHOR_DROP
    return rec["line"], rec


def judge_anchor(readings, runs, N, line, judged, references=None):
    """The anchored eye line: each reading of a checkpoint of a watched run (`runs`, by folder name) at or after
    step N, judged once (its key kept in `judged` when it is at or over the line; a reading under it is returned
    as the stop, its key in the row's 'stop_key', and kept only when the stop is written). Returns (stop reason
    or None, rows)."""
    names = {Path(str(r)).name for r in runs}
    rows, why = [], None
    for r in sorted(readings, key=lambda x: (x["step"], x["key"])):
        if r["run"] not in names or r["step"] < int(N):
            continue
        row = dict(r)
        row["judged_before"] = r["key"] in judged
        rows.append(row)
        if line is None or row["judged_before"]:
            continue
        if r["value"] < line:
            if why is None:
                ref = references or {}
                why = ("motor fixes kill line crossed: anchored eye - at checkpoint %d of %s the model's 50 ms picture "
                       "prediction had %.1f in 100 less error with the eye command sent than with the eyes held "
                       "still, under the line of %.1f in 100 (the lower of the reading before the switch, %s in 100, "
                       "and right after it with no training, %s in 100, less %.1f: 2.5 healthy readings in 100 fall "
                       "under it)" % (r["step"], r["run"], 100.0 * r["value"], 100.0 * line,
                                      "%.1f" % (100.0 * ref["before"]) if _num(ref.get("before")) else "unread",
                                      "%.1f" % (100.0 * ref["floor"]) if _num(ref.get("floor")) else "unread",
                                      100.0 * ANCHOR_DROP))
                row["stop_key"] = r["key"]
            continue
        judged.add(r["key"])
    return why, rows


def watch_complete(N, watch_steps, stats_clock, eye_rows, block=EYE_BLOCK, anchor_rows=None, anchor_line=None):
    """(complete, what is still awaited, the eye blocks settled blind): the watch of `watch_steps` updates
    after N is complete when the stats clock has reached N + watch_steps (every full window of the watch
    has been judged) and every eye block that starts before N + watch_steps is settled: judged (a block
    under the line, or one with too few records, said blind), or blind because training ran a whole block
    past its end while its observer records never completed it (the eye_blind lines say so). The last
    such block ends up to one block after the watch's end and is judged once the records pass that end.
    With the anchored eye line (anchor_line not None) the watch also awaits a judged reading of a checkpoint
    at or after N + watch_steps (the end reading: the procedure's pause job at the watch's end)."""
    if not watch_steps:
        return False, "no watch length (--watch-steps 0): the watch ends only with a stop", []
    end = int(N) + int(watch_steps)
    waiting = []
    if stats_clock is None or int(stats_clock) < end:
        waiting.append("the stats windows up to step %d (the stats clock is at %s)" % (end, stats_clock))
    if anchor_line is not None and not any(r["step"] >= end for r in (anchor_rows or [])):
        waiting.append("an anchored eye reading of a checkpoint at or after step %d (the end reading)" % end)
    rows = {b["first_step"]: b for b in (eye_rows or [])}
    open_blocks, blind = [], []
    for f in range(int(N), end, int(block)):
        b = rows.get(f) or {}
        if b.get("complete") and b.get("judged") is not None:
            continue
        if stats_clock is not None and int(stats_clock) >= f + 2 * int(block):
            blind.append(f)
            continue
        open_blocks.append(f)
    if open_blocks:
        waiting.append("the eye block%s from step %s (judged once the observer records pass its end)"
                       % ("s" if len(open_blocks) > 1 else "", ", ".join(str(f) for f in open_blocks)))
    return (not waiting), "; ".join(waiting), blind


# ------------------------------------------------------------------ the runs, the files, the stop
def _same(a, b):
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def parent_of(run):
    """(the run folder `run` continues, the checkpoint it resumed from), or (None, None). The
    supervisor's relaunch writes continuation_receipt.json (parent_run, checkpoint); a relaunch by hand -
    the way training continues after a stop that requires repair, since the supervisor never relaunches
    such a run - names the checkpoint in its args.json (--resume), whose run folder is the parent."""
    r = _read_json(Path(run) / "continuation_receipt.json")
    if isinstance(r, dict) and r.get("parent_run"):
        return str(r["parent_run"]), (str(r["checkpoint"]) if r.get("checkpoint") else None)
    a = _read_json(Path(run) / "args.json")
    if isinstance(a, dict) and a.get("resume"):
        ck = Path(str(a["resume"]))
        if ck.parent.name == "checkpoints":
            return str(ck.parent.parent), str(ck)
    return None, None


def _is_run(a, b):
    """The same run folder: the same path, or the same folder name (a checkpoint the archiver moved
    names its run folder by name on another drive; run folder names carry their launch time)."""
    return _same(a, b) or Path(str(a)).name == Path(str(b)).name


def reverts_of(run, deploy_run):
    """{setting: [the deploy's, this run's]} for each deploy setting (DEPLOY_SETTINGS) this run's args.json
    sets otherwise than the deploy run's: a relaunch that reverts the deploy (the third review, point G: a
    revert relaunched from a watched run's checkpoint was followed and judged). Empty when either args.json
    is unreadable (a launch whose trainer has not written its args.json yet is followed)."""
    mine = _read_json(Path(run) / "args.json")
    deploy = _read_json(Path(deploy_run) / "args.json")
    if not isinstance(mine, dict) or not isinstance(deploy, dict):
        return {}
    return {k: [deploy.get(k), mine.get(k)] for k in DEPLOY_SETTINGS
            if k in deploy and mine.get(k) != deploy.get(k)}


def watched_runs(deploy_run, not_followed=None):
    """The deploy run, then every continuation of it beside it (parent_of naming a watched run), in the
    order found (a parent before its continuation). A continuation that reverts a deploy setting
    (reverts_of) is not followed, nor anything continuing it; not_followed (a dict) gets {its name: the
    settings it reverts}."""
    deploy_run = Path(deploy_run)
    runs = [deploy_run]
    try:
        siblings = sorted(p for p in deploy_run.parent.iterdir() if p.is_dir())
    except OSError:
        return runs
    parents = {}
    for p in siblings:
        parent, _ = parent_of(p)
        if parent:
            parents[p] = parent
    grew = True
    while grew:
        grew = False
        for p, parent in parents.items():
            if not any(_same(p, r) for r in runs) and any(_is_run(parent, r) for r in runs):
                reverted = reverts_of(p, deploy_run)
                if reverted:
                    if not_followed is not None:
                        not_followed[p.name] = reverted
                    continue
                runs.append(p)
                grew = True
    return runs


def first_step_of(run, deploy_run, after_step):
    """The model step a watched run began at: N for the deploy run, the resumed checkpoint's step for a
    continuation (parent_of), None when unknown."""
    if _same(run, deploy_run):
        return int(after_step)
    name = Path(str(parent_of(run)[1] or "")).name
    return int(name[len("ckpt_step_"):]) if name.startswith("ckpt_step_") and name[10:].isdigit() else None


def chain_spans(runs, deploy_run, after_step):
    """[(run, (start, end))] in chain order: each run owns the steps from its start to the next run's
    start (the last run: no end). A run whose start is unknown starts where the one before it ended."""
    starts = [first_step_of(r, deploy_run, after_step) for r in runs]
    for i in range(1, len(starts)):
        if starts[i] is None:
            starts[i] = starts[i - 1]
    # chronological by start (a run found later can have begun earlier: two relaunches of one parent)
    order = sorted(range(len(runs)), key=lambda i: (starts[i], i))
    runs, starts = [runs[i] for i in order], [starts[i] for i in order]
    out = []
    for i, r in enumerate(runs):
        end = starts[i + 1] if i + 1 < len(runs) else None
        out.append((r, (starts[i], end)))
    return out


class StatsReader:
    """The compact form (compact) of every mimo and motor_interface_change record of each run's stats
    file, read incrementally (a partly written last line is left for the next poll)."""

    def __init__(self):
        self.records, self.offsets = {}, {}

    def read(self, run):
        path = Path(run) / "replay_training_stats.jsonl"
        key = str(path)
        recs = self.records.setdefault(key, [])
        try:
            with open(path, "rb") as fh:
                fh.seek(self.offsets.get(key, 0))
                data = fh.read()
        except OSError:
            return recs
        end = data.rfind(b"\n")
        if end < 0:
            return recs
        self.offsets[key] = self.offsets.get(key, 0) + end + 1
        for line in data[:end].decode("utf-8", "replace").splitlines():
            if '"mimo"' not in line and '"motor_interface_change"' not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("type") in ("mimo", "motor_interface_change"):
                recs.append(compact(r))
        return recs


def switch_step(records):
    """N: the step of the first motor_interface_change record, or None."""
    for r in records:
        if r.get("type") == "motor_interface_change" and isinstance(r.get("step"), int):
            return int(r["step"])
    return None


def report_every_of(run):
    """The run's own --report-every (args.json), 100 when it is not recorded."""
    args = _read_json(Path(run) / "args.json", {}) or {}
    v = args.get("report_every")
    return int(v) if isinstance(v, int) and v > 0 else REPORT_EVERY


def ensure_baselines(run, baseline_run, begins, after_step, turnover=REPLAY_TURNOVER_STEPS):
    """Copy the pre-deploy run's watcher_baselines.json into `run` when it has none and begins before
    N + turnover. Returns what was done, for the state file."""
    if baseline_run is None:
        return "no --baseline-run: the hourly watcher measures its own first hour"
    dest = Path(run) / "watcher_baselines.json"
    if dest.exists():
        base = _read_json(dest, {}) or {}
        return ("present, copied from %s" % base["copied_from"] if base.get("copied_from")
                else "present, measured by the hourly watcher itself (not replaced)")
    if begins is None or begins >= int(after_step) + int(turnover):
        return "not copied: the run begins at or after N + %d, past the replay ring's turnover" % turnover
    src = _read_json(Path(baseline_run) / "watcher_baselines.json")
    if not isinstance(src, dict) or not src:
        return "not copied: %s holds no readable watcher_baselines.json" % baseline_run
    out = dict(src, copied_from=str(Path(baseline_run) / "watcher_baselines.json"),
               copied_at=time.time(), copied_because=(
                   "motor fixes deploy: the hourly watcher's erosion rules are judged against the levels "
                   "before the switch, not against a first hour that holds the switch's transient"))
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    tmp.replace(dest)
    return "copied from %s" % baseline_run


# the settings the deploy plan switches in a relaunch of their own, never beside F1 or F4, with their
# defaults (train_mimo.py; the deploy branch's review, finding 5)
SEPARATE_RELAUNCH = {"position_rows": "all", "parent_relative_targets": "off", "reach_projection": "off"}


def same_relaunch_changes(deploy_run, baseline_run=None):
    """{flag: [before, after]} for each SEPARATE_RELAUNCH flag the deploy relaunch changed: the deploy
    run's args.json against the args.json of the run it resumed from (the checkpoint its --resume names,
    wherever the archiver moved it), else against --baseline-run's. A sentence when neither is readable."""
    args = _read_json(Path(deploy_run) / "args.json")
    if not isinstance(args, dict):
        return "unchecked: %s holds no readable args.json" % Path(deploy_run).name
    before = None
    if args.get("resume"):
        try:
            from mimo_bridge.storage_locations import resolve_run_args
            before = _read_json(resolve_run_args(str(args["resume"])))
        except Exception:
            before = None
    if not isinstance(before, dict) and baseline_run is not None:
        before = _read_json(Path(baseline_run) / "args.json")
    if not isinstance(before, dict):
        return "unchecked: no readable args.json for the run the deploy resumed from"
    out = {}
    for flag, default in SEPARATE_RELAUNCH.items():
        was, now = before.get(flag, default), args.get(flag, default)
        if was != now:
            out[flag] = [was, now]
    return out


def active_run(saved):
    st = _read_json(Path(saved) / "training_continuation.json", {}) or {}
    return st.get("active_run")


def _read_checked(path):
    """(value, None) for a readable JSON file; (None, 'missing') when there is no such file; (None,
    'unreadable') when it exists but cannot be read or parsed (caught mid-replace, or written as UTF-16
    by PowerShell)."""
    p = Path(path)
    if not p.exists():
        return None, "missing"
    try:
        return json.loads(p.read_text(encoding="utf-8-sig")), None
    except (OSError, ValueError):
        return None, "unreadable"


def write_hold(saved, run, trainer_running=False):
    """Why no control file may be written now, or None (the deploy branch's first review, finding 2 and
    point 10; its second review, finding 6 and point 7; its third review, finding E):
      - the ability scheduler holds a saved window ('preparing', 'saving', 'ready') or a review hold; its
        status file UNREADABLE is a hold too (a reader can meet it mid-replace); a task it runs outside
        a saved window is no hold (the scheduler reconciles a running task before any pause check, and a
        stop does not disturb it);
      - supervisor_control.json is unreadable or carries no cmd (the supervisor reads both as a pause);
      - the run holds a control.json no trainer has consumed yet (a coordinated pause's own stop);
      - a supervisor pause is in force - UNLESS it is a PLAIN pause while a trainer of the run runs
        (trainer_running: the third review, finding E). A scheduler window and a coordinated pause are
        recognised by their own marks above; a plain pause with the watched trainer running is neither
        (the deploy procedure's own steps 4 to 6 run the deploy trainer under the coordinator's pause),
        and a stop written then disturbs nothing: stop_run writes control.json and stop_reason.json and
        keeps the pause as it is (plain_pause, keep_pause)."""
    status, err = _read_checked(Path(saved) / "ability_rechecks" / "status.json")
    if err == "unreadable" or (err is None and not isinstance(status, dict)):
        return "the ability scheduler's status.json is unreadable (it may be mid-replace): read as a hold"
    window = ((status or {}).get("comparison_window") or {}).get("state")
    if window in SCHEDULER_HOLDS:
        return "the ability scheduler holds its saved window (state '%s')" % window
    control, err = _read_checked(Path(saved) / "supervisor_control.json")
    if err == "unreadable" or (err is None and not isinstance(control, dict)):
        return "supervisor_control.json is unreadable: read as a pause, as the supervisor reads it"
    if isinstance(control, dict) and "cmd" not in control:
        return "supervisor_control.json carries no cmd: read as a pause, as the supervisor reads it"
    if run is not None and (Path(run) / "control.json").exists():
        return "%s holds a control.json no trainer has consumed yet" % Path(run).name
    if isinstance(control, dict) and control.get("cmd") == "pause" and not trainer_running:
        return ("a supervisor pause is in force (%s) and no trainer of the watched run runs"
                % str(control.get("reason", "no reason given"))[:160])
    return None


def plain_pause(saved):
    """The reason of the supervisor pause in force, when there is one (write_hold has let a stop through
    it only as a plain pause while the watched trainer runs), else None."""
    control, err = _read_checked(Path(saved) / "supervisor_control.json")
    if err is None and isinstance(control, dict) and control.get("cmd") == "pause":
        return str(control.get("reason", "no reason given"))[:160]
    return None


def _create_exclusive(path, text):
    """Write `text` to `path` only when no file of that name exists, atomically: a temporary file moved
    onto the name by a call that fails when the name exists (os.rename on Windows, os.link elsewhere).
    Another program's control.json is never replaced (the ability scheduler's snapshot request is
    written with a replace; the confirmation below covers the case where it replaces this one)."""
    path = Path(path)
    tmp = path.with_name("%s.%d.motor_watch.tmp" % (path.name, os.getpid()))
    tmp.write_text(text, encoding="utf-8")
    try:
        if os.name == "nt":
            os.rename(tmp, path)
        else:
            os.link(tmp, path)
            tmp.unlink()
        return True
    except FileExistsError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def write_stop_files(run, why, stop_id):
    """<run>/control.json {"cmd": "snapshot_and_stop"} (no auto_resume) and then <run>/stop_reason.json
    (resume_policy repair_required), UTF-8 without a byte-order mark, each carrying the stop's id and
    this program's name; neither replaces a file of that name, and nothing more is written when the
    control.json could not be. Returns (control written, reason written)."""
    run = Path(run)
    control = _create_exclusive(run / "control.json", json.dumps(
        {"cmd": "snapshot_and_stop", "reason": why, "writer": WRITER, "stop_id": stop_id}))
    if not control:
        return False, False
    reason = _create_exclusive(run / "stop_reason.json", json.dumps(
        {"reason": why, "resume_policy": "repair_required", "timestamp": time.time(), "writer": WRITER,
         "stop_id": stop_id}))
    if control:
        print("STOP RULE REACHED: %s -> snapshot_and_stop written to %s" % (why, run / "control.json"),
              flush=True)
    return control, reason


def write_pause(saved, why, stop_id):
    """The supervisor pause of a stop: supervisor_control.json read, given cmd pause, the reason, the time,
    this program's name and the stop's id, and replaced atomically with every other key kept (the
    archiver switch). An UNREADABLE file is left as it is - its other keys cannot be kept - and
    stop_reason.json then keeps the supervisor from relaunching the stopped run on its own (the deploy
    branch's second review, point 7). Returns None when written, else why not."""
    path = Path(saved) / "supervisor_control.json"
    control, err = _read_checked(path)
    if err == "unreadable" or (err is None and not isinstance(control, dict)):
        return ("supervisor_control.json is unreadable: left as it is (its other keys cannot be kept); "
                "stop_reason.json keeps the supervisor from relaunching the stopped run")
    control = dict(control or {})
    control.update(cmd="pause", reason=why, timestamp=time.time(), writer=WRITER, stop_id=stop_id)
    try:
        tmp = path.with_suffix(".motor_watch.tmp")
        tmp.write_text(json.dumps(control), encoding="utf-8")
        tmp.replace(path)
    except OSError as error:
        return "the supervisor pause could not be written (%s)" % error
    return None


def stop_run(run, saved, why, stop_id=None, keep_pause=None):
    """The hourly watcher's three writes (hourly_watch.stop_run) for `run`: control.json first (created
    exclusively: a control.json that appeared since the hold was read is never replaced, and then
    nothing else is written, so the watcher's own pause cannot hold its own stop), then stop_reason.json
    (write_stop_files) and the supervisor pause (write_pause). keep_pause: the reason of a plain
    supervisor pause in force while the watched trainer runs (write_hold; the third review, finding E):
    that pause is kept as it is and none is written (the record's pause_kept); control.json stops the
    trainer and stop_reason.json keeps the supervisor from relaunching when the pause is lifted. Returns
    the stop's record."""
    stop_id = stop_id or new_stop_id()
    control, reason = write_stop_files(run, why, stop_id)
    if not control:
        pause_note = "not written: the stop waits"
    elif keep_pause is not None:
        pause_note = ("not written: a plain supervisor pause was in force while the trainer ran (%s); kept as "
                      "it is" % keep_pause)
    else:
        pause_note = write_pause(saved, why, stop_id)
    return dict(stop_id=stop_id, time=time.time(), run=str(run), reason=why, written=bool(control),
                written_to=[str(run)] if control else [],
                pause_written=pause_note is None, pause_kept=keep_pause if control else None,
                notes=[n for n in (pause_note,
                                   None if control else "%s already held a control.json: not replaced; the "
                                   "stop is written at the next poll" % Path(run).name,
                                   None if reason else "%s already held a stop_reason.json: kept"
                                   % Path(run).name) if n])


def new_stop_id():
    return "%d-%d" % (int(time.time() * 1000), os.getpid())


# ------------------------------------------------------------------ a written stop, confirmed
def consumed_in(run, stop_id):
    """The control_done_*.json in `run` that holds this stop (the trainer renames control.json so when it
    reads it), or None."""
    for p in sorted(Path(run).glob("control_done_*.json")):
        d = _read_json(p, {})
        if isinstance(d, dict) and stop_id is not None and d.get("stop_id") == stop_id:
            return str(p)
    return None


def launched_at(run):
    """When a run's launch began: the supervisor's continuation receipt's time (written before it
    launches), else the time the trainer wrote the run's args.json (at its start), else None."""
    r = _read_json(Path(run) / "continuation_receipt.json")
    if isinstance(r, dict) and _num(r.get("timestamp")):
        return float(r["timestamp"])
    try:
        return os.path.getmtime(Path(run) / "args.json")
    except OSError:
        return None


def trainers_of(run):
    """The process ids of the trainers running for `run` (the supervisor's own matching,
    training_continuation.processes), or None when processes cannot be listed."""
    try:
        from mimo_bridge.tools.training_continuation import processes
        return [p.pid for p in processes("trainer", run)]
    except Exception:
        return None


def reason_in_place(stop):
    """The stop_reason.json (resume_policy repair_required) in a run this stop was written to, or None:
    the file that keeps the supervisor from relaunching that run (training_continuation.pause_reason)."""
    for r in stop.get("written_to") or []:
        d = _read_json(Path(r) / "stop_reason.json")
        if isinstance(d, dict) and d.get("resume_policy") == "repair_required":
            return str(Path(r) / "stop_reason.json")
    return None


def confirm_stop(stop, runs, saved, now=None):
    """One poll of a written stop (the deploy branch's second review, finding 2; its third review,
    finding A). The stop is CONFIRMED when no trainer of a watched run launched at or before the stop
    remains and either
      - READ: a control_done_*.json holding its id is in a watched run (the trainer renames control.json
        so when it reads it); or
      - LANDED IN A STOPPED RUN: no trainer read it, no such trainer ran on QUIET_POLLS polls in a row
        ending at least QUIET_SECONDS after the write, and a stop_reason.json (repair_required) is in a
        run it was written to. The case: a stop that waited for a hold which ends with training stopped
        (a scheduler window, the pause runner's pause) is written at the watcher's first poll after the
        hold clears, before the supervisor's relaunch (37 to 53 s after, measured by the third review;
        the watcher polls every 60 s), so it lands in the stopped run; its pause then keeps the
        supervisor from relaunching, and no trainer ever reads it. A relaunch that raced the write
        writes its receipt 0.1 s before it launches and is found as a watched run launched before the
        stop: it is chased, below.
    A trainer launched AFTER the stop (a continuation someone started after it: their decision) neither
    receives it nor keeps it from being confirmed. Until confirmed, at every poll the stop is written
    into each watched run that runs a trainer launched at or before the stop was written, and into the
    active run when that is such a watched run (the supervisor may have relaunched a continuation between
    the watcher's read of the active run and its write; the ability scheduler's snapshot request may
    have replaced the watcher's control.json, and the trainer read that instead) - never over a
    control.json already there. RELEASED when the supervisor pause the stop wrote was replaced (someone
    lifted or rewrote it); a stop written under a pause it kept (write_hold: a plain pause while a
    watched trainer runs) wrote no pause, and is never released. After a stop is confirmed or released
    the watch GOES ON (the third review, finding A): its crossing run is not judged again, and a
    continuation is judged from its first full windows. Returns the phase: 'confirmed', 'released' or
    'confirming'; the stop's record is updated in place."""
    now = time.time() if now is None else float(now)
    sid = stop.get("stop_id")
    consumed = {Path(r).name: c for r in runs for c in [consumed_in(r, sid)] if c}
    live, unknown = {}, []
    for r in runs:
        t = trainers_of(r)
        if t is None:
            unknown.append(Path(r).name)
        elif t:
            live[str(r)] = t
    wrote = float(stop.get("time") or 0.0)

    def before_stop(r):
        started = launched_at(r)
        return started is None or started <= wrote
    early = {r: t for r, t in live.items() if before_stop(r)}
    stop["consumed_in"] = sorted(consumed.values())
    stop["trainers_running"] = {Path(r).name: t for r, t in live.items()}
    stop["trainers_unlisted"] = unknown
    stop["checked"] = now
    if consumed and not early and not unknown:
        stop["confirmed"] = now
        stop["confirmed_how"] = ("read: by the trainer of %s, and no trainer of a watched run launched at or "
                                 "before the stop remains" % ", ".join(sorted(consumed)))
        return "confirmed"
    in_place = reason_in_place(stop)
    if not consumed and not early and not unknown and in_place:
        stop["quiet_polls"] = int(stop.get("quiet_polls") or 0) + 1
        if stop["quiet_polls"] >= QUIET_POLLS and now - wrote >= QUIET_SECONDS:
            stop["confirmed"] = now
            stop["confirmed_how"] = (
                "landed in a stopped run: no trainer read it, no trainer of a watched run launched at or before "
                "the stop ran on %d polls in a row (%.0f s after the write), and %s keeps the supervisor from "
                "relaunching that run" % (stop["quiet_polls"], now - wrote, in_place))
            return "confirmed"
    else:
        stop["quiet_polls"] = 0
    control, err = _read_checked(Path(saved) / "supervisor_control.json")
    if err is None and isinstance(control, dict) and not stop.get("pause_kept"):
        ours = control.get("cmd") == "pause" and control.get("stop_id") == sid
        if not ours and stop.get("pause_written", True):
            stop["released"] = dict(time=now, because="the supervisor pause this stop wrote was replaced",
                                    control_cmd=control.get("cmd"), control_reason=str(control.get("reason"))[:160])
            return "released"
        if not stop.get("pause_written", True):
            note = write_pause(saved, stop.get("reason"), sid)
            stop["pause_written"] = note is None
    act = active_run(saved)
    targets = []
    for r in runs:
        if before_stop(r) and (str(r) in live or (act is not None and _same(r, act))):
            targets.append(r)
    for r in targets:
        name = Path(r).name
        if name in consumed or (Path(r) / "control.json").exists():
            continue            # it read this stop and is ending, or a control.json waits to be read
        control_written, _ = write_stop_files(r, stop.get("reason"), sid)
        if control_written:
            stop["written_to"] = sorted(set(stop.get("written_to") or []) | {str(r)})
            stop.setdefault("rewrites", []).append(dict(run=str(r), time=time.time()))
    return "confirming"


# ------------------------------------------------------------------ ending the watch by hand
def end_request(deploy_run):
    """The deploy run's end request (<deploy run>/motor_fixes_watch_end.json, written by --end), or None."""
    d = _read_json(Path(deploy_run) / END_NAME)
    return d if isinstance(d, dict) else None


def write_end(deploy_run, reason):
    """--end: ask the watch of this deploy run to end (at a revert: the third review, point G). The
    watcher reads it at its next poll, records phase 'ended' and exits; --ensure starts no watcher after
    it. Nothing else is written; the stops the watch wrote stay as they are."""
    d = dict(time=time.time(), reason=str(reason or "no reason given"), writer=WRITER)
    tmp = Path(deploy_run) / (END_NAME + ".tmp")
    tmp.write_text(json.dumps(d), encoding="utf-8")
    tmp.replace(Path(deploy_run) / END_NAME)
    return d


# ------------------------------------------------------------------ the state file, the lock, the status
def load_state(deploy_run):
    state = _read_json(Path(deploy_run) / STATE_NAME, {}) or {}
    return state if isinstance(state, dict) else {}


def _stop_key(s):
    return s.get("stop_id") or "%s|%s|%s" % (s.get("run"), s.get("step"), s.get("reason"))


def merge_stops(old, new):
    """The union of two lists of stop records by stop id, a stop in both merged field by field: a value
    that is set wins over one that is not, and the runs it was written to are joined. So a second or a
    stale watcher can add to the record but never erase a written stop (the deploy branch's second
    review, finding 3)."""
    merged = {}
    for s in list(old or []) + list(new or []):
        k = _stop_key(s)
        m = dict(merged.get(k) or {})
        for key, value in s.items():
            if key == "written_to":
                m[key] = sorted(set(m.get(key) or []) | set(value or []))
            elif key == "written":
                m[key] = bool(m.get(key)) or bool(value)
            elif value is not None or key not in m:
                m[key] = value
        merged[k] = m
    return list(merged.values())


def _write_state(deploy_run, state):
    """Write the state file MERGED with what it holds (merge_stops; the union of judged blocks), and
    return what was written."""
    dest = Path(deploy_run) / STATE_NAME
    old = load_state(deploy_run)
    out = dict(state)
    out["stops"] = merge_stops(old.get("stops"), state.get("stops"))
    out["judged_blocks"] = sorted(set(str(x) for x in old.get("judged_blocks") or [])
                                  | set(str(x) for x in state.get("judged_blocks") or []))
    try:
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        tmp.replace(dest)
    except OSError as error:
        print("STATE WRITE FAILED: %s" % error, flush=True)
    return out


def write_status(saved, state):
    """The watcher's status for the dashboards: <saved>/motor_fixes_watch_status.json (the deploy
    branch's second review, finding 6: a waiting stop had no reader but the log and the state file).
    Its phase, a waiting stop and what it waits for, a written stop not yet confirmed, the NOT COVERED
    changes, and the watched runs."""
    open_stop = state.get("open_stop")
    status = dict(deploy_run=state.get("deploy_run"), runs=[Path(r).name for r in state.get("runs") or []],
                  time=state.get("time"), pid=state.get("pid"), poll_s=state.get("poll_s"),
                  phase=state.get("phase"), status=state.get("status"),
                  pending_stop=state.get("pending_stop"),
                  open_stop=None if not open_stop else {k: open_stop.get(k) for k in (
                      "reason", "time", "stop_id", "written_to", "consumed_in", "trainers_running",
                      "trainers_unlisted", "notes")},
                  same_relaunch_changes=state.get("same_relaunch_changes"), writer=WRITER,
                  watch_steps=state.get("watch_steps"), watch_end_step=state.get("watch_end_step"),
                  complete=state.get("complete"), ended=state.get("ended"),
                  not_followed=state.get("not_followed"),
                  anchored_eye=None if not state.get("anchor") else {
                      k: state["anchor"].get(k) for k in ("line", "before", "floor", "why_blind")})
    try:
        dest = Path(saved) / STATUS_NAME
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=1, default=str), encoding="utf-8")
        tmp.replace(dest)
    except OSError as error:
        print("STATUS WRITE FAILED: %s" % error, flush=True)
    return status


class Memory:
    """What a watcher carries between polls and across a restart: the judged eye blocks, the stops it or
    another watcher wrote (absorbed from the state file at every poll unless fresh: --fresh), and when a
    stop it found began to wait."""

    def __init__(self, previous=None, fresh=False):
        self.judged_blocks = set()
        self.stops = []
        self.pending = None
        self.fresh = bool(fresh)
        if not self.fresh:
            self.absorb(previous or {})

    def absorb(self, state):
        if self.fresh:
            return
        self.judged_blocks |= set(str(x) for x in (state or {}).get("judged_blocks") or [])
        self.stops[:] = merge_stops(self.stops, (state or {}).get("stops"))

    def judged_until(self):
        """{run name: step}: a written stop's crossing run, and the window that completed the crossing
        (the deploy branch's second review, point 7: only that run's windows are skipped)."""
        out = {}
        for s in self.stops:
            if s.get("written") and isinstance(s.get("step"), int):
                run = s.get("crossing_run") or Path(str(s.get("run") or "")).name
                out[run] = max(out.get(run, s["step"]), s["step"])
        return out

    def judged_after(self):
        steps = [s.get("step") for s in self.stops if s.get("written") and isinstance(s.get("step"), int)]
        return max(steps) if steps else None

    def closed_runs(self):
        """{run name: the step of the written stop that closed it}: a written stop's crossing run is not
        judged again (judge_windows' closed; the third review, finding A). An eye-stop crossing names no
        run: its block is kept as judged instead."""
        out = {}
        for s in self.stops:
            run = s.get("crossing_run")
            if s.get("written") and run and run not in (EYE_CROSSING, ANCHOR_CROSSING):
                out[run] = s.get("step")
        return out

    def written_stops(self):
        """The written stops, oldest first, as the completion record names them."""
        return [dict(stop_id=s.get("stop_id"), step=s.get("step"), reason=s.get("reason"),
                     confirmed_how=s.get("confirmed_how"), released=bool(s.get("released")))
                for s in self.stops if s.get("written")]

    def open_stop(self):
        """The newest written stop that is neither confirmed nor released, or None."""
        for s in reversed(self.stops):
            if s.get("written") and not s.get("confirmed") and not s.get("released"):
                return s
        return None


def watch_once(deploy_run, saved, after_step=None, baseline_run=None, reader=None, memory=None,
               write_stop=True, poll_s=None, watch_steps=WATCH_STEPS, anchor=None):
    """One poll. Returns (stop reason written by this poll or None, state dict). Writes the state file
    (merged) into the deploy run and the status file into `saved`. While a written stop is open the poll
    only confirms it (confirm_stop). watch_steps: the watch's length in updates after N (the deploy's one
    watch; 0: no end): windows after N + watch_steps and eye blocks starting at or after it are outside
    the watch, and once every one inside it is judged the phase is 'complete'. anchor: dict(before=<result
    .json>, floor=<result.json>, folders=[...]) for the anchored eye line (judge_anchor); None or without both
    readings, the line judges nothing and the state says so."""
    reader = reader or StatsReader()
    memory = memory if memory is not None else Memory()
    memory.absorb(load_state(deploy_run))
    not_followed = {}
    runs = watched_runs(deploy_run, not_followed)
    state = dict(time=time.time(), pid=os.getpid(), poll_s=poll_s, deploy_run=str(deploy_run),
                 runs=[str(r) for r in runs], not_followed=not_followed)
    ended = end_request(deploy_run)
    if ended is not None:
        # --end (a revert; the third review, point G): the watch ends here, whatever it was doing; the
        # stops it wrote stay in the state file
        state.update(phase="ended", ended=ended, stops=memory.stops, judged_blocks=sorted(memory.judged_blocks),
                     status="the watch was ended at %s: %s" % (_clock(ended.get("time")), ended.get("reason")))
        state = _write_state(deploy_run, state)
        write_status(saved, state)
        return None, state
    open_stop = memory.open_stop()
    if open_stop is not None:
        phase = confirm_stop(open_stop, runs, saved)
        state.update(phase=phase, open_stop=None if phase != "confirming" else open_stop,
                     status={"confirmed": "stop confirmed: a trainer read it and no trainer of a watched "
                                          "run remains: " + str(open_stop.get("reason")),
                             "released": "stop released: " + str((open_stop.get("released") or {})
                                                                 .get("because")),
                             "confirming": "stop written, not yet confirmed: " + str(open_stop.get("reason"))
                             }[phase],
                     stops=memory.stops, judged_blocks=sorted(memory.judged_blocks))
        state = _write_state(deploy_run, state)
        memory.absorb(state)
        write_status(saved, state)
        return None, state
    per_run = [(Path(r).name, reader.read(r)) for r in runs]
    N = after_step if after_step is not None else switch_step(per_run[0][1])
    state.update(after_step=N, stops=memory.stops, judged_blocks=sorted(memory.judged_blocks),
                 watch_steps=int(watch_steps or 0))
    watch_end = None if N is None or not watch_steps else int(N) + int(watch_steps)
    # the whole stats clock, before the watch's cut: training may run on while the observer chunks lag
    whole_clock = max((r["step"] for _, recs in per_run for r in recs
                       if r.get("type") == "mimo" and isinstance(r.get("step"), int)), default=None)
    if watch_end is not None:
        state["watch_end_step"] = watch_end
        # the windows inside the watch only: a record at step C covers the 100 steps before C
        per_run = [(name, [r for r in recs if not (isinstance(r.get("step"), int) and r["step"] > watch_end
                                                     and r.get("type") == "mimo")])
                   for name, recs in per_run]
    if N is None:
        state["phase"] = "no switch"
        state["status"] = ("no switch recorded: the deploy run's stats hold no motor_interface_change "
                           "record (the flags did not change at its launch) and no --after-step was given")
        state = _write_state(deploy_run, state)
        write_status(saved, state)
        return None, state
    report_every = report_every_of(deploy_run)
    state["same_relaunch_changes"] = same_relaunch_changes(deploy_run, baseline_run)
    spans = chain_spans(runs, deploy_run, N)
    state["spans"] = {Path(r).name: [lo, hi] for r, (lo, hi) in spans}
    state["baselines"] = {Path(r).name: ensure_baselines(r, baseline_run, lo, N) for r, (lo, _) in spans}
    judged_until = memory.judged_until()
    closed = memory.closed_runs()
    why, report = judge_windows(per_run, N, report_every=report_every,
                                spans={Path(r).name: sp for r, sp in spans}, judged_after=judged_until,
                                closed=closed)
    state["windows"] = report
    state["judged_until"] = judged_until
    state["closed_runs"] = closed
    stop_at = stop_step_of(report) if why else None
    crossing = crossing_run_of(report) if why else None
    stats_clock = max((r["step"] for _, recs in per_run for r in recs
                       if r.get("type") == "mimo" and isinstance(r.get("step"), int)), default=None)
    eye_why, rows, blind = judge_eye_blocks(chain_eye_series(spans, N), N, memory.judged_blocks,
                                            stats_clock=stats_clock, before=watch_end)
    state["eye_blocks"] = rows
    state["eye_blind"] = blind
    # THE ANCHORED EYE LINE (the fourth review's finding 4): every reading of a watched run's checkpoint at or
    # after N, judged once against the lower of the two references less ANCHOR_DROP
    anchor = anchor or {}
    a_line, a_refs = anchor_references(anchor.get("before"), anchor.get("floor"))
    a_why, a_rows = judge_anchor(anchor_readings(saved, anchor.get("folders") or ()), runs, N, a_line,
                                 memory.judged_blocks, references=a_refs)
    state["anchor"] = dict(a_refs, readings=a_rows)
    state["judged_blocks"] = sorted(memory.judged_blocks)
    eye_key = None
    if why is None and eye_why:
        why = eye_why
        stop_at = next((b.get("stop_step") for b in rows if b.get("stop_step") is not None), None)
        eye_key = next((b.get("stop_key") for b in rows if b.get("stop_key") is not None), None)
        crossing = EYE_CROSSING
    if why is None and a_why:
        why = a_why
        hit = next((r for r in a_rows if r.get("stop_key") is not None), {})
        stop_at, eye_key, crossing = hit.get("step"), hit.get("stop_key"), ANCHOR_CROSSING
    act = active_run(saved)
    live = next((r for r in runs if act and _same(r, act)), None)
    state["active_run_watched"] = live is not None
    written = None
    state["phase"] = "watching"
    state["status"] = "watching"
    if why:
        hold = (None if live is not None else
                "the lineage to be active: the active run (%s) is not one of the watched runs" % act)
        # a plain supervisor pause holds the stop only while no trainer of the active run runs (the third
        # review, finding E); processes that cannot be listed count as none
        running = bool(trainers_of(live)) if hold is None else False
        hold = hold or write_hold(saved, live, trainer_running=running)
        if hold or not write_stop:
            if memory.pending is None or memory.pending.get("reason") != why:
                memory.pending = dict(reason=why, since=time.time())
            state["phase"] = "waiting"
            state["status"] = "line crossed; the stop waits: " + (hold or "writing is off")
            state["pending_stop"] = dict(reason=why, step=stop_at, waiting_for=hold or "writing is off",
                                         since=memory.pending["since"])
        else:
            stop = stop_run(live, saved, why, keep_pause=plain_pause(saved))
            stop.update(step=stop_at, crossing_run=crossing, found=(memory.pending or {}).get("since"))
            memory.pending = None
            if stop["written"]:
                memory.stops.append(stop)
                if eye_key is not None:
                    memory.judged_blocks.add(str(eye_key))
                state["phase"] = "confirming"
                state["status"] = "stop written, not yet confirmed: " + why
                state["open_stop"] = stop
                written = why
            else:
                memory.pending = dict(reason=why, since=time.time())
                state["phase"] = "waiting"
                state["status"] = "line crossed; the stop waits: " + "; ".join(stop["notes"])
                state["pending_stop"] = dict(reason=why, step=stop_at, waiting_for="; ".join(stop["notes"]),
                                             since=memory.pending["since"])
            state["stops"] = memory.stops
            state["judged_blocks"] = sorted(memory.judged_blocks)
    else:
        memory.pending = None
        done, awaited, blind_blocks = watch_complete(N, watch_steps, whole_clock, rows, anchor_rows=a_rows,
                                                     anchor_line=a_line)
        state["watch_awaits"] = awaited
        if done:
            stops = memory.written_stops()
            state["phase"] = "complete"
            state["status"] = ("watch complete: %s in the %d updates after the switch (steps %d to %d)%s" % (
                "no kill line was crossed" if not stops else
                "%d stop%s written (%s), and no crossing after %s" % (
                    len(stops), "s were" if len(stops) > 1 else " was",
                    "; ".join("at step %s: %s" % (s["step"], s["reason"]) for s in stops),
                    "them" if len(stops) > 1 else "it"),
                int(watch_steps), int(N), int(N) + int(watch_steps),
                ("; the eye-stop line was blind for the blocks from step " + ", ".join(str(f) for f in blind_blocks)
                 if blind_blocks else "")
                + ("; " + a_refs["why_blind"] if a_line is None else "")))
            state["complete"] = dict(time=time.time(), steps=[int(N), int(N) + int(watch_steps)],
                                     eye_blind_blocks=blind_blocks, stops=stops,
                                     anchored_eye=dict(line=a_line, blind=a_refs.get("why_blind"),
                                                       readings=[dict(step=r["step"], run=r["run"], value=r["value"])
                                                                 for r in a_rows]))
    state = _write_state(deploy_run, state)
    memory.absorb(state)
    write_status(saved, state)
    return written, state


def _clock(t):
    return time.strftime("%I:%M %p", time.localtime(float(t))) if _num(t) else "an unrecorded time"


def waiting_lines(state, now=None):
    """The lines that name a waiting or unconfirmed stop, first in every poll's print and in --ensure."""
    now = time.time() if now is None else now
    out = []
    p = state.get("pending_stop")
    if p:
        since = p.get("since")
        out.append("STOP WAITING since %s (%s): %s; it waits for: %s" % (
            _clock(since), "%.0f minutes" % ((now - float(since)) / 60.0) if _num(since) else "unknown",
            p.get("reason"), p.get("waiting_for")))
    o = state.get("open_stop")
    if o:
        out.append("STOP WRITTEN at %s, NOT YET CONFIRMED: written into %s; read by a trainer in %s; "
                   "trainers still running in %s%s" % (
                       _clock(o.get("time")), ", ".join(Path(r).name for r in o.get("written_to") or []) or "none",
                       ", ".join(Path(c).parent.name for c in o.get("consumed_in") or []) or "no run yet",
                       ", ".join(sorted(o.get("trainers_running") or {})) or "no run",
                       "; trainers cannot be listed for %s" % ", ".join(o["trainers_unlisted"])
                       if o.get("trainers_unlisted") else ""))
    return out


def print_state(state):
    for line in waiting_lines(state):
        print(line, flush=True)
    print("motor fixes watch %s: N = %s, runs %s; %s" % (
        time.strftime("%I:%M %p"), state.get("after_step"),
        ", ".join(Path(r).name for r in state.get("runs", [])), state.get("status")), flush=True)
    if state.get("watch_end_step") is not None and state.get("phase") != "complete":
        print("  the watch: the %s updates after the switch, to step %s; it awaits %s" % (
            state.get("watch_steps"), state.get("watch_end_step"), state.get("watch_awaits") or "nothing"),
            flush=True)
    for run in (state.get("windows") or {}).get("runs", []):
        if "closed_by_stop_at" in run:
            print("  %s: not judged again, closed by the stop written for its crossing at step %s" % (
                run["run"], run["closed_by_stop_at"]), flush=True)
            continue
        vals = run.get("last") or {}
        print("  %s: %d full windows judged, %d short stop records not counted; %s" % (
            run["run"], run["judged"], run.get("short", 0), "; ".join(
                "%s %s (%d in a row past the line)" % (k, describe(k, vals.get(k)), run["counts"].get(k, 0))
                for k in LINES)), flush=True)
    rows = state.get("eye_blocks") or []
    done = [b for b in rows if b.get("judged")]
    if done:
        b = done[-1]
        print("  eye stop, newest judged block from step %d: %.1f of every 100 records within %g degrees "
              "of a vertical stop (the line: more than %g)"
              % (b["first_step"], b["vertical_near_stop_per_100"], MB.STOP_BAND_DEG,
                 EYE_STOP_LINE_PER_100), flush=True)
    for line in state.get("eye_blind") or []:
        print("  EYE-STOP LINE BLIND: " + line, flush=True)
    anchor = state.get("anchor") or {}
    if anchor.get("why_blind"):
        print("  ANCHORED EYE LINE BLIND: " + anchor["why_blind"], flush=True)
    elif _num(anchor.get("line")):
        newest = (anchor.get("readings") or [None])[-1]
        print("  anchored eye: the line is %.1f in 100 (before the switch %.1f, right after it with no training %.1f, "
              "less %.1f); %s" % (100.0 * anchor["line"], 100.0 * anchor["before"], 100.0 * anchor["floor"],
                                  100.0 * ANCHOR_DROP,
                                  "no reading of a checkpoint after the switch yet" if not newest else
                                  "newest reading %.1f in 100 at checkpoint %d of %s" % (
                                      100.0 * newest["value"], newest["step"], newest["run"])), flush=True)
    for name, reverted in sorted((state.get("not_followed") or {}).items()):
        print("  NOT FOLLOWED: %s reverts the deploy (%s); it is not judged (end this watch with --end at a "
              "revert)" % (name, ", ".join("--%s from %s to %s" % (k.replace("_", "-"), a, b)
                                           for k, (a, b) in reverted.items())), flush=True)
    changed = state.get("same_relaunch_changes")
    if isinstance(changed, dict) and changed:
        print("  NOT COVERED: the deploy relaunch also changed %s; the deploy plan switches these in a "
              "relaunch of their own: the switch-jump verdicts and the following line's pre-deploy band do "
              "not cover them" % ", ".join("--%s from %s to %s" % (k.replace("_", "-"), a, b)
                                           for k, (a, b) in changed.items()), flush=True)
    elif isinstance(changed, str):
        print("  THE DECODER'S POSITION ROWS AND THE LIMB FLAGS ARE UNCHECKED: " + changed, flush=True)


# ------------------------------------------------------------------ one watcher, staying alive
def acquire_lock(deploy_run):
    """The deploy run's watcher lock (<deploy run>/motor_fixes_watch.lock), held for the process's life:
    the OwnerLock the supervisor and the ability scheduler use. OSError when another process holds it."""
    from mimo_bridge.tools.training_continuation import OwnerLock
    return OwnerLock(Path(deploy_run) / LOCK_NAME)


def lock_held(deploy_run):
    """Whether another process holds the deploy run's watcher lock."""
    try:
        lock = acquire_lock(deploy_run)
    except OSError:
        return True
    lock.close()
    return False


def start_detached(argv, deploy_run):
    """Start this program as a process of its own that outlives the caller, its output appended to
    <deploy run>/motor_fixes_watch.log. Returns the process id."""
    flags = 0
    if os.name == "nt":
        flags = 0x00000008 | 0x00000200          # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    log = open(Path(deploy_run) / LOG_NAME, "ab")
    try:
        p = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve())] + list(argv),
                             cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             creationflags=flags, close_fds=True)
    finally:
        log.close()
    return p.pid


def ensure(argv, deploy_run, poll_s, held=lock_held, start=start_detached):
    """Start a detached watcher unless one holds the deploy run's lock. Says a waiting or unconfirmed
    stop first, and a watcher that holds the lock but has not polled within ALIVE_POLLS polls (it cannot
    be replaced while it lives: end it by its process id). Returns (started, message)."""
    st = load_state(deploy_run)
    pid, t = st.get("pid"), st.get("time")
    age = None if not _num(t) else time.time() - float(t)
    lines = waiting_lines(st)
    if st.get("phase") == "complete" and st.get("complete") and not held(deploy_run):
        return False, "\n".join(lines + ["the watch is complete, no watcher is started: %s" % st.get("status")])
    ended = end_request(deploy_run)
    if ended is not None and not held(deploy_run):
        return False, "\n".join(lines + ["the watch was ended at %s (%s), no watcher is started"
                                         % (_clock(ended.get("time")), ended.get("reason"))])
    if held(deploy_run):
        if age is None or age > ALIVE_POLLS * float(poll_s):
            msg = ("ALIVE BUT SILENT: process %s holds the watcher lock and last polled %s; it is not "
                   "replaced while it lives - end it by its process id, then run --ensure again"
                   % (pid, "never" if age is None else "%.0f s ago" % age))
        else:
            msg = "alive: process %s holds the watcher lock and polled %.0f s ago" % (pid, age)
        return False, "\n".join(lines + [msg])
    new = start(argv, deploy_run)
    return True, "\n".join(lines + ["started process %s (the one before: %s, last poll %s)" % (
        new, pid, "never" if age is None else "%.0f s ago" % age)])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--deploy-run", required=True, help="the run folder the deploy launched")
    ap.add_argument("--after-step", type=int, default=None,
                    help="N; default: the deploy run's motor_interface_change record")
    ap.add_argument("--baseline-run", default=None,
                    help="the pre-deploy run whose watcher_baselines.json the hourly watcher should judge by")
    ap.add_argument("--saved", default=str(SAVED), help="Saved/mimo_bridge (tests pass their own)")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--hours", type=float, default=0.0,
                    help="0 (the default): until the watch is complete (a confirmed or released stop does not "
                         "end it)")
    ap.add_argument("--watch-steps", type=int, default=WATCH_STEPS,
                    help="the watch's length in updates after the switch (the deploy's one watch, 6,000 by "
                         "default); 0: no end but a stop")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore the state file's written stops and judged blocks")
    ap.add_argument("--ensure", action="store_true",
                    help="start a detached watcher unless one for this deploy run holds its lock, then return")
    ap.add_argument("--end", default=None, metavar="REASON",
                    help="end this deploy run's watch (at a revert): its watcher ends at its next poll and "
                         "--ensure starts none after it")
    ap.add_argument("--anchor-before", default=None,
                    help="the anchored eye reading before the switch: result.json of motor_fixes_eye_anchor.py (or "
                         "probe_saved_gaze.py) on the deploy stop's checkpoint")
    ap.add_argument("--anchor-floor", default=None,
                    help="the anchored eye reading right after the switch with no training: result.json of "
                         "motor_fixes_eye_anchor.py --eye-row-convention physical on the same checkpoint")
    ap.add_argument("--anchor-readings", default="",
                    help="comma-separated folders holding the deploy's own anchored eye readings (result.json under "
                         "them), judged beside the ability scheduler's gaze tests")
    a = ap.parse_args(argv)
    anchor = dict(before=a.anchor_before, floor=a.anchor_floor,
                  folders=[f for f in str(a.anchor_readings or "").split(",") if f.strip()])
    if not Path(a.deploy_run).is_dir():
        print("motor fixes watch: no such deploy run folder: %s" % a.deploy_run, flush=True)
        return 2
    if a.end is not None:
        d = write_end(a.deploy_run, a.end)
        print("motor fixes watch --end: written at %s (%s); a watcher of %s ends at its next poll"
              % (_clock(d["time"]), d["reason"], Path(a.deploy_run).name), flush=True)
        return 0
    if a.ensure:
        child = [x for x in argv if x != "--ensure"]
        started, message = ensure(child, a.deploy_run, a.poll)
        print("motor fixes watch --ensure: " + message, flush=True)
        return 0
    try:
        lock = acquire_lock(a.deploy_run)
    except OSError:
        print("motor fixes watch: another watcher holds %s; this one ends (one watcher per deploy run)"
              % (Path(a.deploy_run) / LOCK_NAME), flush=True)
        return 4
    memory = Memory(None if a.fresh else load_state(a.deploy_run), fresh=a.fresh)
    # the new watcher is marked at once, before its first poll (which may read hours of records)
    _write_state(a.deploy_run, dict(load_state(a.deploy_run), time=time.time(), pid=os.getpid(),
                                    poll_s=a.poll, started=time.time()))
    reader = StatsReader()
    deadline = None if a.hours <= 0 else time.time() + 3600.0 * a.hours
    print("motor fixes watch started at %s, process %d%s" % (
        time.strftime("%I:%M %p"), os.getpid(),
        "" if not memory.judged_until() else "; windows at or before a written stop's step in its crossing "
        "run are not judged again: %s" % memory.judged_until()), flush=True)
    code = 0
    try:
        while True:
            try:
                why, state = watch_once(a.deploy_run, a.saved, a.after_step, a.baseline_run, reader, memory,
                                        poll_s=a.poll, watch_steps=a.watch_steps, anchor=anchor)
            except Exception as error:      # a failed poll (a file being written, a drive hiccup) must
                # never end the watch: it is retried at the next poll, and it is said
                print("POLL FAILED at %s: %s: %s" % (time.strftime("%I:%M %p"), type(error).__name__, error),
                      flush=True)
                if a.once:
                    return 3
                time.sleep(a.poll)
                continue
            try:
                print_state(state)
            except Exception as error:
                print("STATE LINES FAILED: %s: %s" % (type(error).__name__, error), flush=True)
            phase = state.get("phase")
            if why:
                code = 1
                if a.once:
                    break
            # a confirmed or released stop does not end the watch (the third review, finding A): its
            # crossing run is closed, and a continuation is judged from its first full windows
            if phase == "complete":
                code = 6
                break
            if phase == "ended":
                code = 7
                break
            if state.get("after_step") is None and phase == "no switch" and a.once:
                code = 2
                break
            if a.once:
                break
            if phase != "confirming" and deadline is not None and time.time() > deadline:
                break
            time.sleep(a.poll)
    finally:
        lock.close()
    print("motor fixes watch ended at %s (%s)" % (
        time.strftime("%I:%M %p"), {0: "its time or its one poll is up", 1: "a stop was written",
                                     2: "no switch recorded",
                                     6: "the watch is complete: every update of it judged",
                                     7: "the watch was ended by --end"
                                     }.get(code, str(code))),
          flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
