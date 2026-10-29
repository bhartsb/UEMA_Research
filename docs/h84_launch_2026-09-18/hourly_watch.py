"""H84 hourly watcher (2026-09-18): full gauge sweep + stop rules, OWN-BASELINE edition.

Forked from docs/consequence_repair_2026-09-09/hourly_watch.py (the H67-H83 watcher;
the supervisor recognises a watcher by this file name). That program judged erosion
against LINEAGE baselines - causal errors 0.016 / 0.043 / 0.0055, the medians of a
trained lineage's tails. H84 starts from fresh weights, so a trained lineage's levels
are the wrong reference: an untrained model's errors are higher at hour 1 and fall
afterwards, and the lineage lines would stop a fresh run for being new. Every erosion
rule here is judged against THIS RUN'S OWN first hourly window: check 1 measures the
baselines and writes them to <run>/watcher_baselines.json, check 2 onward judges.

Every volatile gauge is judged and printed as the mean over the last hour of records
(by timestamp), never a single-record snapshot (three false alarms and one near-miss
came from snapshot reads of oscillating gauges, 2026-09-08).

Stop rules (this program writes snapshot_and_stop itself; each hourly rule needs TWO
consecutive window means past its line):
- non-finite training update (nan_steps > 0): 5-minute check, immediate
- planner selection worsens any body's total score, omits comparison scores or
  encounters a non-finite search: 5-minute check, snapshot and repair stop;
  goal-error/strength tradeoffs are separately recorded in planner_monitor.json. The hand
  stage must report the hand calibration its inherited planner contract names, followed
  through each replacement the trainer admitted and recorded (expected_hand_calibration,
  added Oct 9, 2026)
- with liveness repair enabled, a layer below half responding in two complete report
  windows after its initial repair observation window: 5-minute check, repair stop
- free disk under 8 GB on the stats drive, the checkpoint drive or the caregiver-record
  drive (added 2026-09-22), or commit room under COMMIT_STOP_GB (4 GB since 2026-09-30, operator; free memory under 1.5 GB is an alert): 5-minute check, immediate (one
  checkpoint + episode companion need ~2.6 GB and a half-written save loses the newest
  evidence)
- vision causal error > 2x this run's hour-1 mean (the H55 signature: a mis-aimed
  loss eroding vision); tactile causal error > 3x this run's hour-1 mean
- replay loss > 3x this run's hour-1 mean (declared at launch: the re-study lesson
  diverging)
- model steps per second under one third of this run's hour-1 mean (declared at
  launch as "sustained step-time blow-up": memory spilling off the card or a stuck
  worker pool)
- token variance (vision or tactile) < 0.15 on two consecutive window means, judged
  from run-hour 3 only (a fresh encoder's first two hours are not a collapse gauge);
  the first breach takes a free snapshot. Flat tokens can be a flat encoder or flat
  input; a fresh-world probe on the snapshot attributes which
- follow_bout < -0.038 twice (inherited line: mean 0.006 - 3x std 0.015)
- any entry of <run>/declared_criteria.json with mode "stop": 5-minute check (since Oct 5, 2026, the relaunch
  review's findings F3 and F4) on the mean of the last hour's records, and at every hourly sweep; a stop entry whose
  field no record of that hour carries stops the run once the run's records span a whole hour after the entry's
  from_run_hour (the re-check's finding N1: production writes some fields only from a segment's third or later
  record); an unreadable declared_criteria.json, or an entry that cannot be judged, stops the run (its line cannot be
  read). A watcher started before the run's first full stats record looks for it every 30 s, so its first 5-minute
  check comes 300 s after that record
- the vestibular encoder repair's kill lines (--vestibular-encoder-repair, added 2026-09-25;
  docs/gain_of_function_v4/vestibular_repair_watch.py): 5-minute check, on every run the switch
  reaches, through 10,000 updates after it - the switch itself, the other senses' reports 2 and 3
  after it, a balance token that stays dead, the other senses in every 20-report window (against the
  window before, two windows running, and against their level before the switch at two levels: an
  alert, printed once and recorded, never a stop, and a stop), the balance encoder's liveness and
  non-finite updates, and the step rate (two windows running), across run folders. The supervisor ends
  this program at a relaunch and starts it for the new run; the new one reads the whole timeline
  from the switch, so a relaunch neither loses a judgement nor stops again on a crossing whose stop
  was taken. The hourly sweep prints these lines and the repair's acceptance lines.
- the asked-change path's stop lines (--motor-asked-change on, added 2026-10-03;
  docs/gain_of_function_v4/motor_asked_change_watch.py): 5-minute check, on every run that the path's
  switch record reaches, through the first full report at or past 10,000 model steps after it - the
  switch record itself, a non-finite update, the path's hidden layer under 0.9 live share in 2 full
  reports with a reading running, the other senses in
  reports 2 and 3 and in every 20-report window against their level before the switch (an alert,
  printed once and recorded, never a stop, and a stop; the vestibular watch's multiples), the step rate
  (two windows running) and the path's output above its growth line. A run that no
  motor_asked_change_switch record reaches is not judged and prints nothing. The hourly sweep prints
  these lines, the pass lines S1 and S2 and the free readings.
- the vestibular input path's stop lines (--vestibular-input-path on, added 2026-10-04;
  docs/gain_of_function_v4/vestibular_input_path_watch.py): 5-minute check, on every run that the path's
  switch record reaches, through the first full report at or past 10,000 model steps after it - the
  switch record itself, a non-finite update, the other senses in reports 2 and 3 and in every 20-report
  window against their level before the switch (an alert and a stop; the vestibular watch's multiples),
  the step rate (two windows running) and the path's output above its growth line. A run that no
  vestibular_input_path_switch record reaches is not judged and prints nothing. The hourly sweep prints
  these lines and the pass lines T1 (head tilt read from the vestibular token) and T2 (vestibular's
  prediction error below the error of predicting no change).
- the all-slot paths' stop lines (--heads-all-slot-path on, added 2026-10-04;
  docs/gain_of_function_v4/heads_all_slot_path_watch.py): 5-minute check, on every run that the paths'
  switch record reaches, through the first full report at or past 10,000 model steps after it - the
  switch record itself (every head's prediction unchanged on 64 probe rows), a non-finite update, a
  path's hidden layer under 0.9 live share in 2 full reports with a reading running, the senses in
  reports 2 and 3 and in every 20-report window against their level before the switch (the vestibular
  watch's multiples), the step rate (two windows running) and a path's output above its sense's real
  change per number between two model steps (re-derived Oct 4 after the first segment's runaway; the
  first design's growth line is printed no more). An alert, never a stop: a path whose fixed vector
  points against the old head's and is half that real change or more. A run that no
  heads_all_slot_path_switch record reaches is not judged and prints nothing. The hourly sweep prints
  these lines and the pass lines per head.
Reachability of the own-baseline rules: a fresh model's errors and losses FALL over
the first hours, so a line at 2x or 3x the hour-1 level sits above the whole normal
trajectory; only a rise back past twice the untrained level reaches it, which is
erosion by any standard. The steps-per-second line is one third of the hour-1 rate:
the hour-1 window already contains the slow start-up steps, so a normal run never
reads under it; a spill off the card (measured 2026-09-17: 32 bodies at 8,487 MiB on
an 8,188 MiB card) does.

The proprioception token repair's kill lines (--proprio-token-repair within_body, added 2026-09-25;
docs/gain_of_function_v4/proprio_repair_watch.py): 5-minute check, on every run its switch reaches, through its
horizon - the switch and the other senses' first report after it (vision, touch, hearing, and balance when its
token is alive), a body's token overshooting or jittering, the other senses' 20 reports after the first and the 20
after the lesson's start, non-finite updates, the step rate at both, and a body losing response per unit of its own
posture movement (version 6); bodies numbered 1-8 and body 3 (face down since Sept 19) left out of the lines on bodies.
The hourly sweep prints these lines, the fatigue hold and the repair's acceptance lines.

Alerts (printed, never a stop): a failed hourly save (checkpoint_failures > 0) or no
complete checkpoint in the last 2 h; body movement under the historical 0.0023 reference
(exposure to movement only; it cannot tell a motor fault from useful stillness); hand
foveation under 0.03 from run-hour 6; every declared_criteria.json entry with mode
"alert"; an isolated layer reading under DESIGN section 10's line; a body whose token of a
sense stays under that sense's line (added 2026-09-25: the per-body collapse watch
token_temporal_std_bodies, hour means; the lines and the 480 measured windows behind them are
in mimo_bridge/deaf_senses.py), a DEAF SENSE ALERT naming every such body and sense. That one
is not a stop because on 2026-09-25 five bodies were deaf in body position and all eight in
balance, both known and under repair: a stop would halt training for known defects, the ground
of the hand-foveation and body-movement alerts. Persistent low response
with repair enabled is a stop, not merely an alert. A stats file older
than 20 minutes prints a STALL warning only - the supervisor owns restarts, and two
restarters would race.

NUMBERS A PERSON READS (2026-09-23, DESIGN.md section 11, the operator's instruction of
2026-09-22 23:55). Every printed line carries its numbers as counts per 100 (or per 10,000
when under 1 in 100), real units (cm, degrees, GB, MiB, ms, seconds, hours, days) or a
percentage between two named ends - never a bare score decimal, a cosine, a correlation, a
mean squared error, an entropy or a loss value. A gauge the record carries only in one of
those forms prints its plain meaning and "not yet measured as a count". The stop rules still
judge the raw values; their lines print each value as a percentage of its line or of this
run's own hour-1 level. tests/test_h84_hourly_watch.py scans the printed lines of a real
segment 31 record and of a record built from the producers' own summary functions with
tools/check_no_bare_decimals.bare_decimals. A line builder that raises prints "LINE FAILED"
and the check goes on: an informational line never stops the stop rules.

GAIN-OF-FUNCTION BLOCKS (2026-09-22, docs/gain_of_function_v3/DESIGN.md section 3, and the
2026-09-23 field contract between the lanes). Each hourly check also prints, in plain words
with each scale's two ends, the blocks the new flags write: causal_chain, efference,
controllability, learning_progress, wellness_rank, contact_control, synergy, vestibular,
next_frame_norm and machine; and every liveness / pool_spread sub-block and the top-level
liveness block against DESIGN section 10's live-share line (50 of every 100 units; pool
spread has no declared line and prints "no line declared"). A block no record in the hour
carries prints "unmeasured" - absent means the feature is off or not yet written, never 0.
Where the contract adds a field (contact_control.onset_auc_self, success_paired,
gate_self_open; controllability.difference_mean; causal_chain.chosen_tied_share,
plan_exec_distance_cm; wellness_rank.goal_age, weight_share_by_channel) the line reads it when
present and otherwise falls back, naming the field it used. declared_criteria entries may name
any field by dot path, including a list position ("efference.gaze_deg_err.1" = the 100 ms gaze
error).

STORAGE AND MACHINE LINES (2026-09-22, repaired 2026-09-23 after review ops-storage finding
2). Free space on the drive the caregiver records and the episode companions are written to is
judged by the same 8 GB stop as the stats and checkpoint drives. The D: archive's free space is
printed with its runway in days at the archive intake measured from the move manifests (the
larger of the last 24 hours' and the last 7 days' daily rate); under 5 days is an ALERT,
because when D: is full the archiver stops and C: takes the whole 1.46-1.49 GB an hour
(storage survey). An archive drive that cannot be read is an ALERT naming the drive. The
checkpoint archiver's status file is its heartbeat: an error in it (the pass's or the
recordings pass's) is an ALERT quoting the error; its age is judged (ALERT past 1,800 s = two
missed passes) only when the status was written by a WATCHING archiver whose process still
runs, because a one-off or dry run refreshes the file too and would hide a dead watcher. The
machine lines give RAM available, the commit charge against its limit (ALERT inside the commit
guard's 6 GB reserve: on 2026-09-21 a charge of 77.3 of 78.1 GB killed the supervisor) and
graphics-card memory used by all programs, both from the trainer's own records and read live by
this program. UEMA_WATCH_SAVED and UEMA_WATCH_ARCHIVE override the Saved/mimo_bridge and D:
archive folders these lines read (tests).
"""
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

# set by configure() when run as a program; importing this file for its line builders runs nothing
RUN = P = BASE_PATH = None
HOURS = 8
RUN_ARGS = {}
REPO = Path(__file__).resolve().parents[2]
SAVED = Path(os.environ.get("UEMA_WATCH_SAVED") or (REPO / "Saved" / "mimo_bridge"))
ARCHIVE = Path(os.environ.get("UEMA_WATCH_ARCHIVE") or "D:/UEMA_Research/external/D/UEMA_recordings/H84_checkpoint_archive")
DISK_STOP_GB = 8.0
D_RUNWAY_ALERT_DAYS = 5.0          # the lead's line (2026-09-22 build task); D: had 15.1 days then
ARCHIVER_INTERVAL_S = 900.0        # the archiver's --interval as the supervisor starts it
ARCHIVER_STALE_S = 2 * ARCHIVER_INTERVAL_S
COMMIT_RESERVE_GB = 6.0            # tools/commit_guard.py --reserve-gb default
C_TAKES_GB_PER_H = "1.46-1.49"     # C: growth with no archiver (storage survey, 2026-09-22)
VARIANCE_LINE = 0.15               # token variance stop line (see the docstring)
ACTIVITY_REF = 0.0023              # historical body-movement reference (alert only)
HAND_FOV_LINE = 0.03               # hand foveation alert line from run-hour 6
FOLLOW_LINE = -0.038               # bout direction-match stop line (inherited)
# FREE-RAM STOP LINE. 2.5 GB until 2026-09-30; lowered to 1.5 GB by the operator (Sep 29, 10:5x PM PDT: "lower the
# stop line and I'll try to keep fewer tabs open") after segment 41 stopped at 1.4 GB free (10:52 PM) and at 2.4 GB
# (about 4:06 PM) on a 29.7 GB machine. The accepted risk: a save that runs out of memory crashes and loses the steps
# since the last complete checkpoint (the 10:49 AM MemoryError lost 14).
RAM_STOP_GB = 1.5
# COMMIT-ROOM STOP LINE (operator, Sep 30 2026, 11:0x PM PDT: "yes" to judging commit room instead of free memory).
# With the page file at 56,000 MB, low free memory means Windows is paging, not that an allocation will fail; a
# request fails only when the commit charge reaches its limit. Eight free-memory stops since Sep 25 (three on Sep 30)
# came during hourly saves with 7.1 to 11.6 GB of commit room left. The stop now fires under COMMIT_STOP_GB of
# commit room (GiB, mimo_bridge.tools.commit_guard.commit_reading); free memory under RAM_STOP_GB is printed as an
# alert. When commit cannot be read, the free-memory line stops as before.
COMMIT_STOP_GB = 4.0
LIVE_SHARE_LINE = 0.5              # DESIGN section 10: a layer with fewer live units is dead
# LAYERS WHOSE LOW ACTIVITY IS AN ALERT, NOT A STOP (2026-09-29, H84 segment 41). Each entry names the recorded
# functional measurement that showed its inactivity costs nothing measured, and the functional stop that still
# guards it. efference.lesson.vision_in: removing its 52 inactive units changed visual prediction error from
# 0.006759 to 0.006758 on the same 800 examples (coordinator/recovery_reasoning_2026-09-28.json, efference vision:
# "blocks_protected_training": false); it declined in production before any repair (73 of 100 at 411,200, 63 at
# 415,000; the no-repair trial arms reached 45) and stopped segment 41 at 416,900 at 48. The vision causal-error
# stop (2x hour 1) still guards the prediction it serves.
LIVENESS_ALERT_ONLY = {
    # gaze copy: 65 of 128 units idle on the measured moments; switching them off cost nothing measurable
    # (docs/gain_of_function_v4/idle_units/results/vision_in__H84_fresh_drivers_20260918_seg41m_2030_resume_20260930_225714__ckpt_step_447350.json)
    'efference.lesson.vision_in': 'functional cost measured none at step 447,350 (65 idle units of 128 switched off '
                                  'on 800 saved model steps: vision error 0.004877 to 0.004877, 0.007377 for the '
                                  'picture staying the same; '
                                  'docs/gain_of_function_v4/idle_units/results/vision_in__H84_fresh_drivers_20260918_seg41m_2030_resume_20260930_225714__ckpt_step_447350.json); '
                                  'vision causal-error stop still applies',
    # body head: 146 of 256 units idle on the measured moments; switching them off cost nothing measurable
    # (docs/gain_of_function_v4/idle_units/results/body_mlp.0__H84_fresh_drivers_20260918_seg41m_2030_resume_20260930_225714__ckpt_step_447350.json)
    'efference.lesson.body_mlp.0': 'functional cost measured none at step 447,350 (146 idle units of 256 switched '
                                   'off on 800 saved model steps: forecast miss 0.6890 cm to 0.6890 cm, 0.8231 cm for '
                                   'no movement; '
                                   'docs/gain_of_function_v4/idle_units/results/body_mlp.0__H84_fresh_drivers_20260918_seg41m_2030_resume_20260930_225714__ckpt_step_447350.json)',
}
LIVE_INPUT_LINE = -3               # DESIGN section 10: a unit is live when its input rises above
# b3-wellness, 2026-09-22: learning progress read -0.071 to -0.016 on FROZEN weights (no learning
# possible), the range a reading must leave before it is evidence of learning, until the model
# lane's same-moment control (learning_progress.control_mean) is in the record
LP_FROZEN_RANGE = (-0.071, -0.016)
NOT_A_COUNT = "not yet measured as a count"
# THE NAMES THE PRINTED LINES USE (the style guide "ONE WORD PER THING", applied to these lines 2026-10-03; until then they
# said "order", "moment", "touch", "sight and touch" and "vestibular sense"). A model step is one model update; a
# command is what the body is sent; the senses go by the model registry's names. Each name carries its plain meaning
# where the hourly check first uses it: the check's second line (names_line). The senses' meanings are the opening
# words of mimo_bridge/explainer/schema.py SENSE_DEFINITIONS, written again here because this file runs by path;
# tests/test_h84_hourly_watch.py holds the two equal.
SENSE_MEANS = (("vision", "what the two eyes see"), ("tactile", "the skin's sense of touch"),
               ("audio", "what the two ears hear"),
               ("proprioception", "where each tracked body part is, how it is turned and how fast it moves"),
               ("motor", "the command the body was last given, beside the pose each part reached"),
               ("vestibular", "the body's load and balance in 29 numbers"))


def names_line():
    """The hourly check's second line: every name the lines below use, with its plain meaning."""
    repeat, hold = RUN_ARGS.get("action_repeat"), RUN_ARGS.get("motor_hold")
    return ("  names used below: a model step = one model update%s; a command = 9 numbers per body handle sent to "
            "the body (target position, target orientation as 4 numbers, radius, strength)%s; the senses, by the "
            "model registry's names: %s"
            % ("" if repeat is None else " (%s physics steps of the world)" % repeat,
               "" if hold is None else ", held for %s model steps" % hold,
               "; ".join("%s = %s" % pair for pair in SENSE_MEANS)))


def configure(argv):
    global RUN, P, HOURS, BASE_PATH, RUN_ARGS
    RUN = os.path.abspath(argv[1])
    P = os.path.join(RUN, "replay_training_stats.jsonl")
    HOURS = int(argv[2]) if len(argv) > 2 else 8
    BASE_PATH = os.path.join(RUN, "watcher_baselines.json")
    # the supervisor attaches to a repair run at its trainer's start, so the watcher can start before the trainer has
    # written args.json (after its imports): it waits for the file, looking every 15 s for at most 600 s (the re-check
    # of Oct 5, 2026, finding N5)
    path = os.path.join(RUN, "args.json")
    for _ in range(40):
        if os.path.exists(path):
            break
        time.sleep(15)
    with open(path, encoding="utf-8") as f:
        RUN_ARGS = json.load(f)


def rows():
    out = []
    try:
        with open(P, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("type") == "mimo":
                    out.append(r)
    except OSError:
        pass
    return out


def stop_run(why):
    # Pause the supervisor attached to this run before asking its trainer to stop.
    # A stale watcher of a different run must never pause the current trainer.
    try:
        state = json.loads((SAVED / "training_continuation.json").read_text(encoding="utf-8"))
        if Path(state.get("active_run", "")).resolve() == Path(RUN).resolve():
            path = SAVED / "supervisor_control.json"
            control = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            control.update(cmd="pause", reason=why, timestamp=time.time())
            tmp = path.with_suffix(".watcher.tmp")
            tmp.write_text(json.dumps(control), encoding="utf-8")
            tmp.replace(path)
    except (OSError, ValueError, TypeError):
        pass  # stop_reason.json below still forbids automatic relaunch
    print(f"STOP RULE REACHED: {why} -> snapshot_and_stop written", flush=True)
    with open(os.path.join(RUN, "stop_reason.json"), "w", encoding="utf-8") as f:
        json.dump({"reason": why, "resume_policy": "repair_required", "timestamp": time.time()}, f)
    with open(os.path.join(RUN, "control.json"), "w") as f:
        json.dump({"cmd": "snapshot_and_stop"}, f)


def liveness_breach(records, report_every=100):
    """Require two distinct complete windows, after the repair's measured startup window.

    The response line is the existing DESIGN section 10 requirement. A single report
    cannot establish persistence; two complete windows also exceed the prior 100-step
    post-repair retention check. Incomplete or unsampled reports never count as zero.
    """
    windows = {}
    for r in records:
        l = r.get("liveness") or {}
        if not isinstance(l,dict):
            continue
        if (l.get("mode") == "on" and (l.get("samples") or 0) > 0
                and (l.get("window_steps") or 0) >= report_every and isinstance(r.get("step"), int)):
            windows[r["step"]] = l
    if len(windows) < 2:
        return None
    selected = sorted(windows)[-2:]
    before, latest = [windows[s] for s in selected]
    readings = {}
    for r in records:
        if r.get('step') not in selected:
            continue
        merged = dict((r.get('liveness') or {}).get('layers') or {})
        for owner,block in r.items():
            if owner == 'liveness' or not isinstance(block,dict):
                continue
            sub = block.get('liveness')
            if not isinstance(sub,dict):
                continue
            for name,value in sub.items():
                if isinstance(value,(int,float)):
                    # Efference's own readings contain the correctly paired live
                    # lesson only. Generic hooks also see shuffled commands and
                    # older tenure examples; their broader coverage must never
                    # hide inactivity on the orders the body actually received.
                    prefix = ('efference.lesson.' if owner == 'efference' else
                              'contact_head.' if owner == 'contact_control' else owner+'.')
                    key = prefix + name
                    merged.setdefault(key,value)
        readings[r['step']] = merged
    bad = []
    for name, value in readings[selected[-1]].items():
        old = readings[selected[0]].get(name)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v < LIVE_SHARE_LINE
                   for v in (old, value)):
            continue
        if name in LIVENESS_ALERT_ONLY:
            print("  LIVENESS ALERT (not a stop): %s: %.0f of 100 units responding; %s"
                  % (name, 100 * value, LIVENESS_ALERT_ONLY[name]), flush=True)
            continue
        repair_name = name.replace('efference.lesson.','efference.',1)
        # Skip a repair layer only while its repair window can still end. Under --liveness-rule none
        # no window starts (needed None), so a dying repair layer is judged like every other layer;
        # before 2026-09-26 it was skipped always and nothing stopped a layer that died (re-review
        # docs/gain_of_function_v4/agents/liveness_rule_none_review_2026-09-26.md section 4).
        if any(repair_name in (l.get("watching") or {}) and
               not (l["watching"][repair_name].get("ready") is True) and
               ("needed" not in l["watching"][repair_name] or l["watching"][repair_name]["needed"] is not None)
               for l in (before, latest)):
            continue
        bad.append(f"{name}: {100 * value:.0f} of 100 units responding")
    if bad:
        return ("Persistent layer inactivity after the initial repair window: " + "; ".join(bad)
                + ". Below 50 of 100 in two complete reports; snapshot and repair required.")
    return None


_PROPRIO_RULE = []


def _proprio_rule():
    """docs/gain_of_function_v4/proprio_repair_watch.py, loaded once by path: its reader keeps the lineage's stats
    between checks and reads only what the trainer appended since."""
    if not _PROPRIO_RULE:
        import importlib.util
        path = REPO / "docs" / "gain_of_function_v4" / "proprio_repair_watch.py"
        spec = importlib.util.spec_from_file_location("proprio_repair_watch", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _PROPRIO_RULE.append(mod)
    return _PROPRIO_RULE[0]


def proprio_repair_breach():
    """The proprioception token repair's kill lines for this run: a stop reason, or None (also for every run its switch
    does not reach). A rule that cannot be judged says so and never costs another stop rule."""
    try:
        return _proprio_rule().breach(RUN)
    except Exception as error:
        print(f"  PROPRIOCEPTION REPAIR RULE FAILED - its kill lines are not judged this check: "
              f"{type(error).__name__}: {error}", flush=True)
        return None


def proprio_repair_lines():
    """The repair's kill and acceptance lines for a person; nothing for a run its switch does not reach."""
    return _proprio_rule().lines(RUN)


def training_paused():
    """supervisor_control.json holds a pause (read-only; unreadable counts as not paused, so the lines still stop)."""
    try:
        return json.loads((SAVED / "supervisor_control.json").read_text(encoding="utf-8")).get("cmd") == "pause"
    except (OSError, ValueError, AttributeError):
        return False


def trainer_alive():
    """A trainer the supervisor records for this run is alive (read-only). Unknown counts as alive."""
    try:
        state = json.loads((SAVED / "training_continuation.json").read_text(encoding="utf-8"))
        if Path(state.get("active_run", "")).resolve() != Path(RUN).resolve():
            return True
        pids = state.get("trainer_pids")
        if not isinstance(pids, list):
            return True
        import psutil
    except (OSError, ValueError, TypeError, AttributeError, ImportError):
        return True
    for pid in pids:
        try:
            if "train_mimo" in " ".join(psutil.Process(int(pid)).cmdline()):
                return True
        except (psutil.Error, ValueError, TypeError):
            continue
    return False


def quick_breach():
    """The 5-minute machine-and-integrity check. Returns a stop reason or None."""
    mi = rows()
    if mi and (mi[-1].get("nan_steps") or 0) > 0:
        return "non-finite training update (5-minute quick check)"
    failure = liveness_breach(mi, int(RUN_ARGS.get("report_every") or 100))
    if failure:
        return failure
    planner = planner_monitor(RUN)
    if planner["stop_reason"]:
        return planner["stop_reason"]
    why = vestibular_repair_breach()
    if why:
        return why
    why = motor_asked_change_breach()
    if why:
        return why
    why = heads_all_slot_path_breach()
    if why:
        return why
    why = vestibular_input_path_breach()
    if why:
        return why
    drives = {os.path.splitdrive(RUN)[0] + os.sep}
    # the caregiver records and the ~600 MB episode companion of every save go here
    for key in ("checkpoint_storage", "caregiver_record_dir"):
        where = RUN_ARGS.get(key) or ""
        if where:
            drives.add(os.path.splitdrive(os.path.abspath(where))[0] + os.sep)
    machine = None
    for d in sorted(drives):
        try:
            free_gb = shutil.disk_usage(d).free / 1e9
        except OSError:
            continue
        if free_gb < DISK_STOP_GB:
            machine = (f"free disk on {d} is {free_gb:.1f} GB, under the 8 GB line - "
                       f"one checkpoint + episode companion need ~2.6 GB and a "
                       f"half-written save loses the newest evidence")
            break
    if machine is None:
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / 1e9
        except ImportError:
            avail_gb = None
        try:
            from mimo_bridge.tools.commit_guard import commit_reading
            reading = commit_reading()
        except (ImportError, OSError):
            reading = None
        if reading is not None:
            room = reading[1] - reading[0]
            if room < COMMIT_STOP_GB:
                machine = (f"commit room {room:.1f} GB (commit charge {reading[0]:.1f} of {reading[1]:.1f} GB), "
                           f"under the {COMMIT_STOP_GB} GB line - the next allocation can fail and kill the "
                           f"trainer mid-save")
            elif avail_gb is not None and avail_gb < RAM_STOP_GB:
                print(f"  MEMORY ALERT (not a stop): free memory {avail_gb:.1f} GB, under {RAM_STOP_GB} GB; "
                      f"commit room {room:.1f} GB against the {COMMIT_STOP_GB} GB stop line", flush=True)
        elif avail_gb is not None and avail_gb < RAM_STOP_GB:
            machine = (f"free memory {avail_gb:.1f} GB, under the {RAM_STOP_GB} GB line (commit room unreadable) - "
                       f"the next allocation can kill the trainer mid-save")
    if machine:
        # THE MACHINE LINES PROTECT A RUNNING TRAINER (operator-approved fix, 2026-09-26). At 9:13 PM on 2026-09-25
        # this check wrote its memory stop inside a coordinated training pause with no trainer running (the pause's
        # jobs and agents' test runs held the memory): the stop replaced the pause's owner, so training could not
        # resume by itself and stayed down until a person cleared it. Held by a pause with no trainer alive, a
        # machine line has nothing to stop: it is reported and read again at the next check. Without a pause (a
        # crashed trainer the supervisor would relaunch into the same shortage) it still stops.
        if trainer_alive() or not training_paused():
            return machine
        print(f"  MACHINE LINE CROSSED IN A TRAINING PAUSE WITH NO TRAINER RUNNING: {machine} - nothing to stop, "
              f"no stop written; the next 5-minute check reads it again", flush=True)
    why = proprio_repair_breach()
    if why:
        return why
    if mi:
        # every declared stop at every 5-minute check, on the hourly sweep's window (declared_stops), after the disk
        # and memory lines: a declared_criteria.json that cannot be judged never skips them (the re-check of Oct 5,
        # 2026, finding N4)
        newest = mi[-1].get("timestamp", 0)
        try:
            why = declared_stops([r for r in mi if r.get("timestamp", 0) >= newest - 3600],
                                 mi[-1].get("elapsed_s", 0) / 3600, newest - mi[0].get("timestamp", newest))
        except Exception as error:          # a stop line that cannot be judged stops the run, naming why
            why = "the declared stop lines could not be judged (%s: %s)" % (type(error).__name__, error)
        if why:
            return why
        age_min = (time.time() - mi[-1].get("timestamp", 0)) / 60
        if age_min > 20:
            print(f"  STALL WARNING (quick check): stats {age_min:.0f} min old - "
                  f"trainer alive but not reporting; the supervisor owns restarts",
                  flush=True)
    return None


_VESTIBULAR_RULE = []


def _vestibular_rule():
    """docs/gain_of_function_v4/vestibular_repair_watch.py, loaded once by path: its reader keeps the
    lineage's stats between checks and reads only what the trainer appended since."""
    if not _VESTIBULAR_RULE:
        import importlib.util
        path = REPO / "docs" / "gain_of_function_v4" / "vestibular_repair_watch.py"
        spec = importlib.util.spec_from_file_location("vestibular_repair_watch", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _VESTIBULAR_RULE.append(mod)
    return _VESTIBULAR_RULE[0]


def vestibular_repair_breach():
    """The vestibular encoder repair's kill lines for this run: a stop reason, or None (also for every run
    the repair's switch does not reach). A rule that cannot be judged says so and never costs another
    stop rule."""
    try:
        return _vestibular_rule().breach(RUN)
    except Exception as error:
        print(f"  VESTIBULAR REPAIR RULE FAILED - its kill lines are not judged this check: "
              f"{type(error).__name__}: {error}", flush=True)
        return None


def vestibular_repair_lines():
    """The repair's kill lines and acceptance lines for a person; nothing for a run its switch does not reach."""
    return _vestibular_rule().lines(RUN)


_MOTOR_ASKED_CHANGE_RULE = []


def _motor_asked_change_rule():
    """docs/gain_of_function_v4/motor_asked_change_watch.py, loaded once by path: its reader keeps the lineage's
    stats between checks and reads only what the trainer appended since. It runs a private instance of the
    vestibular watch's file; the vestibular rule above is another instance and is not changed by it."""
    if not _MOTOR_ASKED_CHANGE_RULE:
        import importlib.util
        path = REPO / "docs" / "gain_of_function_v4" / "motor_asked_change_watch.py"
        spec = importlib.util.spec_from_file_location("motor_asked_change_watch", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MOTOR_ASKED_CHANGE_RULE.append(mod)
    return _MOTOR_ASKED_CHANGE_RULE[0]


def motor_asked_change_breach():
    """The asked-change path's stop lines for this run: a stop reason, or None (also for every run that no
    motor_asked_change_switch record reaches). A rule that cannot be judged says so and never costs another
    stop rule."""
    try:
        return _motor_asked_change_rule().breach(RUN)
    except Exception as error:
        print(f"  ASKED-CHANGE PATH RULE FAILED - its stop lines are not judged this check: "
              f"{type(error).__name__}: {error}", flush=True)
        return None


def motor_asked_change_lines():
    """The path's stop lines, pass lines and readings for a person; nothing for a run its switch does not reach."""
    return _motor_asked_change_rule().lines(RUN)


_HEADS_ALL_SLOT_PATH_RULE = []


def _heads_all_slot_path_rule():
    """docs/gain_of_function_v4/heads_all_slot_path_watch.py, loaded once by path: its reader keeps the lineage's
    stats between checks and reads only what the trainer appended since. It runs a private instance of the
    asked-change watch's file (which runs its own private instance of the vestibular watch's); the two rules above
    are other instances and are not changed by it."""
    if not _HEADS_ALL_SLOT_PATH_RULE:
        import importlib.util
        path = REPO / "docs" / "gain_of_function_v4" / "heads_all_slot_path_watch.py"
        spec = importlib.util.spec_from_file_location("heads_all_slot_path_watch", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _HEADS_ALL_SLOT_PATH_RULE.append(mod)
    return _HEADS_ALL_SLOT_PATH_RULE[0]


# THE ATTENTION CAP EXPERIMENT'S LATER ARMS (the operator's decision of Oct 6, 2026, received at 1:32 PM PDT): its control
# run 1 was stopped at model step 484,600 by the all-slot paths' line M3 (a path's hidden layer under 0.9 live share in 2
# full reports), an internal statistic, and ended at 484,675. Control run 2 and the cap arm end at 484,675 (--max-steps)
# and read M3 as an alert, so the three arms train under the same conditions to the same step; every other line stops
# them as before. A run is one of them when its args.json resumes checkpoint 481,800 of run
# H84_fresh_drivers_20260918_seg41o_1431 and reads max_steps 484675; every other run is judged as before.
CAP_EXPERIMENT_LATER_ARMS = dict(
    resume_tail="H84_fresh_drivers_20260918_seg41o_1431/checkpoints/ckpt_step_481800", max_steps=484675,
    alerts={"M3": ("the operator's decision of Oct 6, 2026: control run 1 of the attention cap experiment was stopped "
                   "by this line at model step 484,600, so control run 2 and the cap arm read it as an alert and "
                   "end at model step 484,675, as control run 1 did")})



# THE REMEMBERED-HAND CONTINUATION (the operator's decision of Oct 11, 2026, 1:57 PM PDT, on the independent review
# D:/UEMA_Research/external/D/UEMA_scratch/remembered_hand_learning_2026-10-11/reviews/M3_DECISION_REVIEW.md): the one continuation from checkpoint
# 484,675 of run H84_fresh_drivers_20260918_seg41j_1526 to model step 489,350 reads M3 as an alert; every other line stops it
# as before. Over 164 full reports of the lineage and the five arms from 481,800 the whole-model monitor read a path's hidden
# layer under 0.9 in 0 reports; all 12 readings under the line came from the path's own 64-row gauge. A run is this
# continuation when its args.json resumes that checkpoint and reads max_steps 489350; every other run is judged as before.
REMEMBERED_HAND_CONTINUATION = dict(
    resume_tail="H84_fresh_drivers_20260918_seg41j_1526/checkpoints/ckpt_step_484675", max_steps=489350,
    alerts={"M3": ("the operator's decision of Oct 11, 2026, 1:57 PM PDT: the one remembered-hand continuation, checkpoint 484,675 "
                   "to model step 489,350, reads this line as an alert; every other line stops it as before")})
HEADS_ALL_SLOT_PATH_ALERT_RUNS = (CAP_EXPERIMENT_LATER_ARMS, REMEMBERED_HAND_CONTINUATION)

def heads_all_slot_path_alerts():
    """{line: why} of the all-slot paths' stop lines this run reads as alerts: the first entry of
    HEADS_ALL_SLOT_PATH_ALERT_RUNS whose resume tail and max_steps both match the run's args.json, else {}."""
    resume = str(RUN_ARGS.get("resume") or "").replace("\\", "/")
    try:
        max_steps = int(RUN_ARGS.get("max_steps") or 0)
    except (TypeError, ValueError):
        max_steps = 0
    for arm in HEADS_ALL_SLOT_PATH_ALERT_RUNS:
        if resume.endswith(arm["resume_tail"]) and max_steps == arm["max_steps"]:
            return dict(arm["alerts"])
    return {}


def heads_all_slot_path_breach():
    """The all-slot paths' stop lines for this run: a stop reason, or None (also for every run that no
    heads_all_slot_path_switch record reaches). A rule that cannot be judged says so and never costs another
    stop rule. The lines heads_all_slot_path_alerts names are read as alerts for this run."""
    try:
        return _heads_all_slot_path_rule().breach(RUN, alert_instead_of_stop=heads_all_slot_path_alerts())
    except Exception as error:
        print(f"  HEADS' ALL-SLOT PATHS RULE FAILED - its stop lines are not judged this check: "
              f"{type(error).__name__}: {error}", flush=True)
        return None


def heads_all_slot_path_lines():
    """The paths' stop lines and pass lines per head for a person; nothing for a run their switch does not reach."""
    return _heads_all_slot_path_rule().lines(RUN, alert_instead_of_stop=heads_all_slot_path_alerts())


_VESTIBULAR_INPUT_PATH_RULE = []


def _vestibular_input_path_rule():
    """docs/gain_of_function_v4/vestibular_input_path_watch.py, loaded once by path: its reader keeps the lineage's
    stats between checks. It runs private instances of the asked-change watch's and the vestibular watch's files; the
    rules above are other instances and are not changed by it."""
    if not _VESTIBULAR_INPUT_PATH_RULE:
        import importlib.util
        path = REPO / "docs" / "gain_of_function_v4" / "vestibular_input_path_watch.py"
        spec = importlib.util.spec_from_file_location("vestibular_input_path_watch", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _VESTIBULAR_INPUT_PATH_RULE.append(mod)
    return _VESTIBULAR_INPUT_PATH_RULE[0]


def vestibular_input_path_breach():
    """The vestibular input path's stop lines for this run: a stop reason, or None (also for every run that no
    vestibular_input_path_switch record reaches). A rule that cannot be judged says so and never costs another
    stop rule."""
    try:
        return _vestibular_input_path_rule().breach(RUN)
    except Exception as error:
        print(f"  VESTIBULAR INPUT PATH RULE FAILED - its stop lines are not judged this check: "
              f"{type(error).__name__}: {error}", flush=True)
        return None


def vestibular_input_path_lines():
    """The path's stop lines, pass lines and readings for a person; nothing for a run its switch does not reach."""
    return _vestibular_input_path_rule().lines(RUN)


def hand_search_integrity(records, expected_bodies=8, calibration_sha256=None, required=False, _seen=None):
    """Verify hand-only search traces without treating sensory scores as its objective."""
    result=dict(searches=0,body_comparisons=0,improvements=0,ties=0,skipped_events=0,
                malformed_results=0,objective_regressions=0,latest_step=None,stop_reason=None)
    seen=set() if _seen is None else _seen
    def fail(reason):
        result['malformed_results']+=1
        result['stop_reason']=result['stop_reason'] or reason+' Save and repair required.'
    for record in records:
        plan=record.get('plan') if record.get('type') in ('latch','plan') else record
        if not isinstance(plan,dict):continue
        hand=plan.get('hand_optimizer')
        if hand is None:
            if required:fail('Recorded plan omits its required hand search or explicit skip reason.')
            continue
        step=plan.get('step',record.get('step'))
        if type(step) is not int or not isinstance(hand,dict):
            fail('Hand search has no valid decision step or result.');continue
        if step in seen:continue
        seen.add(step);result['latest_step']=step if result['latest_step'] is None else max(step,result['latest_step'])
        if hand.get('stopped_by') in ('awaiting_native_joint_readings','no_valid_memory_goal'):
            if any(key in hand for key in ('first','selected','trace','selected_iteration')):
                fail(f'Hand search at step {step} mixes skipped and scored states.');continue
            result['skipped_events']+=1;continue
        n=plan.get('bodies',record.get('bodies',expected_bodies))
        first,selected,valid,indices,trace=(hand.get(k) for k in ('first','selected','valid','selected_iteration','trace'))
        arrays=(first,selected,valid,indices)
        shape=(type(n) is int and n>0 and (expected_bodies is None or n==expected_bodies)
               and all(isinstance(v,list) and len(v)==n for v in arrays)
               and isinstance(trace,list) and len(trace)==21
               and all(isinstance(row,list) and len(row)==n for row in trace))
        contract=(hand.get('version')=='hand_joint_response_v1' and hand.get('goal')=='wrist_relative_finger_shape'
                  and hand.get('steps')==20 and hand.get('horizon_ms')==250 and hand.get('raw_joint_bound_deg')==11.
                  and (calibration_sha256 is None or hand.get('calibration_sha256')==calibration_sha256))
        if not shape or not contract:
            fail(f'Hand search at step {step} lacks its qualified contract or complete body comparisons.');continue
        finite=lambda value:type(value) in (int,float) and math.isfinite(value)
        if (not all(finite(v) for row in (first,selected,*trace) for v in row)
                or not all(type(v) is bool for v in valid)
                or not all(type(v) is int and 0<=v<=20 for v in indices)):
            fail(f'Hand search at step {step} contains malformed or nonfinite scores.');continue
        result['searches']+=1
        for body in range(n):
            values=[row[body] for row in trace]
            if (first[body]!=values[0] or selected[body]!=values[indices[body]]
                    or selected[body]!=min(values)
                    or (not valid[body] and (indices[body]!=0 or selected[body]!=first[body]))):
                fail(f'Hand search at step {step}, body {body+1}, did not retain the best evaluated candidate.')
            if selected[body]>first[body]:
                result['objective_regressions']+=1
                result['stop_reason']=result['stop_reason'] or f'Hand search returned a worse command at step {step}, body {body+1}. Save and repair required.'
            if valid[body]:
                result['body_comparisons']+=1
                result['improvements']+=int(selected[body]<first[body])
                result['ties']+=int(selected[body]==first[body])
    return result


def planner_integrity(records, required_schema=False, expected_bodies=None,
                      required_hand=False, hand_calibration_sha256=None):
    """Judge each recorded action-search result, not a sampled latest-plan gauge.

    Selected objective values are copied from scored candidates without rounding.
    The repaired search retains the starting candidate, so any strict increase is
    an implementation failure. Goal error may rise when another charged cost falls;
    that tradeoff is counted separately and remains an investigation, not a stop.
    """
    result = dict(searches=0, decision_events=0, body_comparisons=0, objective_regressions=0,
                  goal_improvements=0, goal_regressions=0, goal_ties=0,
                  goal_regressions_with_lower_effort=0, malformed_results=0,
                  nonfinite_search_stops=0, latest_step=None, stop_reason=None)
    seen = set()
    hand_seen=set()
    hand_latched=set();hand_latched_bodies=set();hand_latched_searches=0
    hand_result=hand_search_integrity([],expected_bodies,hand_calibration_sha256,required_hand)
    for record in records:
        plan = record.get("plan") if record.get("type") in ("latch", "plan") else record
        if not isinstance(plan, dict):
            continue
        if required_hand or plan.get('hand_optimizer') is not None:
            checked=hand_search_integrity([record],expected_bodies,
                hand_calibration_sha256,required_hand,_seen=hand_seen)
            for name,value in checked.items():
                if name=='stop_reason':hand_result[name]=hand_result[name] or value
                elif name=='latest_step':
                    if value is not None:hand_result[name]=value if hand_result[name] is None else max(hand_result[name],value)
                else:hand_result[name]+=value
            step=plan.get('step',record.get('step'))
            hand=plan.get('hand_optimizer')
            valid=hand.get('valid') if isinstance(hand,dict) else None
            if (record.get('type')=='latch' and record.get('reached_body') is True
                    and type(step) is int and step not in hand_latched
                    and isinstance(valid,list) and all(type(v) is bool for v in valid)
                    and any(valid)):
                hand_latched.add(step);hand_latched_searches+=1
                hand_latched_bodies.update(i for i,value in enumerate(valid) if value)
        raw_descents=plan.get('optimizer')
        descents=[] if raw_descents is None else raw_descents
        if required_schema and (not isinstance(raw_descents, list) or not descents):
            result["malformed_results"] += 1
            result["stop_reason"] = result["stop_reason"] or "Recorded plan lacks optimizer comparisons; save and repair required."
            continue
        if not isinstance(descents,list):descents=[]
        eyes=plan.get('eye_optimizer') or []
        if not isinstance(eyes,list):
            result['malformed_results']+=1
            result['stop_reason']=result['stop_reason'] or 'Eye planning lacks a valid comparison list; save and repair required.'
            eyes=[]
        for stage,descent in [('body',d) for d in descents]+[('eye',d) for d in eyes]:
            if not isinstance(descent, dict) or descent.get("selection") != "best_evaluated_per_body_v1":
                if required_schema:
                    result["malformed_results"] += 1
                    result["stop_reason"] = result["stop_reason"] or "Planner comparison lacks the qualified selection contract; save and repair required."
                continue
            step, depth = record.get("step"), descent.get("depth")
            if not isinstance(step, int) or not isinstance(depth, int):
                result["malformed_results"] += 1
                result["stop_reason"] = result["stop_reason"] or "Planner search lacks its step or depth; save and repair required."
                continue
            key = (step, stage, depth)
            if key in seen:
                continue
            seen.add(key)
            result["searches"] += 1
            result["latest_step"] = step if result["latest_step"] is None else max(step, result["latest_step"])
            if str(descent.get("stopped_by", "")).startswith("nonfinite"):
                result["nonfinite_search_stops"] += 1
                result["stop_reason"] = result["stop_reason"] or (
                    f"Planner search encountered a non-finite candidate or gradient at step {step}; "
                    "a finite earlier command was retained, but save and repair are required.")
            values = descent.get("per_body") or {}
            arrays = [(values.get(name) or {}).get(which) for name in ("objective", "goal", "effort")
                      for which in ("start", "selected")]
            valid = (all(isinstance(v, list) and v for v in arrays)
                     and len({len(v) for v in arrays if isinstance(v, list)}) == 1)
            for body_count in (record.get("bodies"), expected_bodies):
                if body_count is not None:
                    valid = (valid and type(body_count) is int and body_count > 0
                             and len(arrays[0]) == body_count)
            if not valid:
                result["malformed_results"] += 1
                result["stop_reason"] = result["stop_reason"] or (
                    f"Planner search at step {step} has no complete per-body comparison; save and repair required.")
                continue
            for body, (start, selected, goal0, goal1, effort0, effort1) in enumerate(zip(*arrays), 1):
                result["body_comparisons"] += 1
                if not all(isinstance(v, (int, float)) and math.isfinite(v)
                           for v in (start, selected, goal0, goal1, effort0, effort1)):
                    result["malformed_results"] += 1
                    result["stop_reason"] = result["stop_reason"] or (
                        f"Planner search at step {step}, body {body}, has a non-finite score; save and repair required.")
                    continue
                if selected > start:
                    result["objective_regressions"] += 1
                    result["stop_reason"] = result["stop_reason"] or (
                        f"Planner returned a worse total objective than its starting command at step {step}, "
                        f"body {body}; save and repair required.")
                result["goal_improvements"] += int(goal1 < goal0)
                result["goal_regressions"] += int(goal1 > goal0)
                result["goal_ties"] += int(goal1 == goal0)
                result["goal_regressions_with_lower_effort"] += int(goal1 > goal0 and effort1 < effort0)
    result["decision_events"] = len({step for step, stage, depth in seen})
    result['hand']=hand_result
    result['hand'].update(latched_searches=hand_latched_searches,latched_bodies=sorted(hand_latched_bodies))
    result['stop_reason']=result['stop_reason'] or result['hand']['stop_reason']
    result["status"] = ("repair_required" if result["stop_reason"] else
                        "tradeoff_investigation" if result["goal_regressions"] else
                        "search_checked" if result["body_comparisons"] or result['hand']['body_comparisons'] else "awaiting_scored_decisions")
    return result


def planner_contract(run):
    """Inherit the first explicit contract from sibling continuation runs only."""
    current = Path(run).resolve()
    experiments_root = current.parent
    visited = set()
    while True:
        if current.parent != experiments_root:
            return {}, None, "Planner monitor ancestry escapes this experiments directory."
        if current in visited:
            return {}, None, "Planner monitor ancestry contains a cycle."
        visited.add(current)
        receipt_path = current / "continuation_receipt.json"
        if not receipt_path.exists():
            return {}, None, None
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}, None, "Planner monitor ancestry receipt is unreadable."
        if not isinstance(receipt, dict):
            return {}, None, "Planner monitor ancestry receipt is malformed."
        if "planner_selection_contract" in receipt:
            # An explicit declaration, including an older one, terminates lookup.
            if (receipt.get("planner_selection_contract") == "best_evaluated_per_body_v1"
                    and (type(receipt.get("planner_bodies")) is not int or receipt["planner_bodies"] <= 0)):
                return {}, None, "Planner monitor contract lacks its expected body count."
            if (receipt.get('hand_selection_contract')=='hand_joint_response_v1' and
                    (not isinstance(receipt.get('hand_calibration_sha256'),str)
                     or len(receipt['hand_calibration_sha256'])!=64)):
                return {}, None, 'Hand monitor contract lacks its qualified calibration identity.'
            return receipt, str(current), None
        parent = receipt.get("parent_run")
        if parent is None:
            return {}, None, None
        if not isinstance(parent, str) or not parent or not Path(parent).is_absolute():
            return {}, None, "Planner monitor ancestry has an invalid parent run."
        current = Path(parent).resolve()


# A HAND CALIBRATION REPLACED UNDER ITS ADMISSION RECORD (Oct 9, 2026; the matched comparison of the finger fix, the
# operator's decision received by 11:53 PM PDT, Oct 8). The planner contract a run inherits names the hand calibration
# its lineage was qualified with: seg37's continuation receipt names the admitted fit, payload fe953beb..., and no
# later receipt declares a contract. A launch with --replace-saved-hand-calibration replaces the checkpoint's saved
# calibration only after the trainer's admission check passed (LemLite._replace_saved_hand_calibration), and the trainer
# records the replacement in the run's stats stream (type hand_calibration_replacement, written before its first model
# step) and in the model receipt of every checkpoint it saves (hand_calibration_replacements, which a later resume
# keeps). The hand stage then reports the replacing calibration's payload hash (lem/model/hand_response.py,
# calibration_sha256). Until this rule the monitor held every run to the contract's hash, so the finger arm, whose hand
# stage reports the refit's f68754c8..., would have been stopped at its first 5-minute check. Now the contract's hash is
# followed through those records: a replacement whose replaced payload is the hash expected so far moves the
# expectation to its replacing payload. A run with no replacement record is held to the contract's hash as before; a
# replacement of another calibration, or a hand stage reporting any other hash, stops the run. Only records that match
# the run itself count (the independent review of Oct 9, 2026, finding F1): a stream record when the run's args.json
# names its admission record as --replace-saved-hand-calibration and resumes its checkpoint, a resumed receipt's list
# when the receipt's own saved calibration is the list's last replacing payload; any other record stops the run.
REPLACEMENT_RECORD = '"type": "hand_calibration_replacement"'


def _same_path(a, b):
    """Two path texts naming the same file, compared after normalizing separators, '.' parts and (on Windows) case."""
    return (isinstance(a, str) and isinstance(b, str) and bool(a) and bool(b)
            and os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b)))


def _replacement_hashes(entry):
    """(replaced payload hash, replacing payload hash, time) of one replacement report (the trainer's
    hand_calibration_replacement_report), or None when either hash is missing or not 64 characters."""
    if not isinstance(entry, dict):
        return None
    old, new = ((entry.get(side) or {}).get("payload_sha256") if isinstance(entry.get(side), dict) else None
                for side in ("replaced", "replaced_by"))
    if not all(isinstance(h, str) and len(h) == 64 for h in (old, new)):
        return None
    return old, new, entry.get("at")


def _located(path):
    """A checkpoint folder where it is now: the path itself when it exists, else where the archiver's move records say
    it went (mimo_bridge.storage_locations.locate), else the path."""
    path = Path(path)
    if path.exists():
        return path
    try:
        from mimo_bridge.storage_locations import locate
        return Path(locate(path, verify=False))
    except Exception:
        return path


def hand_calibration_replacements(run):
    """([replacement report, ...] oldest first, [notes], problem or None): the hand calibration replacements that bind
    `run`'s hand stage. First those the checkpoint the run resumed keeps in its model_receipt.json
    (hand_calibration_replacements: every replacement of the run's ancestors), then those the run's own trainer
    recorded at its launch (records of type hand_calibration_replacement in replay_training_stats.jsonl). A resumed
    checkpoint whose receipt cannot be read is a note, never a stop by itself: without it the contract's hash is not
    moved, and a hand stage reporting another hash is stopped by the contract check.
    ONLY THE RUN'S OWN REPLACEMENTS COUNT (the independent review of Oct 9, 2026, finding F1; before it a well-formed
    record was followed whoever wrote it). A stream record counts only when the run's args.json names its admission
    record (admission.path) as replace_saved_hand_calibration and resumes its checkpoint, as
    LemLite._replace_saved_hand_calibration writes them from those two launch settings; a resumed receipt's list counts
    only when the receipt's own hand_response.sha256, the calibration that checkpoint saved, is the list's last
    replacing payload. Any other record is the problem returned, and the run stops."""
    out, notes = [], []
    try:
        args = json.loads((Path(run) / "args.json").read_text(encoding="utf-8-sig"))
        args = args if isinstance(args, dict) else {}
    except FileNotFoundError:
        args = {}
    except (OSError, ValueError):
        args = {}
        notes.append("the run's args.json cannot be read, so the checkpoint it resumed is unknown")
    resume = args.get("resume")
    replacing = args.get("replace_saved_hand_calibration") or ""
    if resume:
        receipt_path = _located(str(resume)) / "model_receipt.json"
        try:
            saved = json.loads(receipt_path.read_text(encoding="utf-8"))
            kept = saved.get("hand_calibration_replacements") or []
            kept = list(kept) if isinstance(kept, list) else [kept]
            own = saved["hand_response"].get("sha256") if isinstance(saved.get("hand_response"), dict) else None
        except (OSError, ValueError, AttributeError) as error:
            kept, own = [], None
            notes.append("the resumed checkpoint's receipt %s cannot be read (%s: %s), so replacements recorded before "
                         "this run are not followed" % (receipt_path, type(error).__name__, error))
        if kept:
            last = _replacement_hashes(kept[-1])
            if last is None or own != last[1]:
                return out, notes, ("The resumed checkpoint's receipt %s lists hand calibration replacements whose last "
                                    "replacing payload is %s, but its own saved hand calibration is payload %s."
                                    % (receipt_path, last[1] if last else "unreadable", own))
            out += kept
    try:
        with open(Path(run) / "replay_training_stats.jsonl", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if REPLACEMENT_RECORD not in line[:64]:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, dict) or record.get("type") != "hand_calibration_replacement":
                    continue
                report = record.get("replacement")
                if not isinstance(report, dict):
                    return out, notes, ("A hand_calibration_replacement record of the run's stats stream holds no "
                                        "replacement report.")
                admission = report.get("admission") if isinstance(report.get("admission"), dict) else {}
                if not replacing:
                    return out, notes, ("The run's stats stream holds a hand_calibration_replacement record, but its "
                                        "args.json names no --replace-saved-hand-calibration.")
                if not _same_path(admission.get("path"), replacing):
                    return out, notes, ("A hand_calibration_replacement record names the admission record %s, but the "
                                        "run's args.json names --replace-saved-hand-calibration %s."
                                        % (admission.get("path"), replacing))
                if not _same_path(report.get("checkpoint"), resume):
                    return out, notes, ("A hand_calibration_replacement record replaced the calibration of checkpoint %s, "
                                        "but the run's args.json resumes %s." % (report.get("checkpoint"), resume))
                out.append(report)
    except OSError:
        pass
    return out, notes, None


def expected_hand_calibration(contract_sha256, replacements):
    """(the payload hash the hand stage must report, [{replaced, replaced_by, at, admission}] followed, problem or None):
    the contract's hash followed through each replacement in order. A replacement whose replaced payload is not the hash
    expected at that point, or whose hashes cannot be read, is a problem, and the hash stays where it was. A contract
    without a hand calibration (None) follows nothing."""
    expected, followed, seen = contract_sha256, [], set()
    if not expected:
        return expected, followed, None
    for entry in replacements:
        hashes = _replacement_hashes(entry)
        if hashes is None:
            return expected, followed, ("A hand calibration replacement record lacks its replaced or replacing payload "
                                        "hash.")
        old, new, at = hashes
        if (old, new, str(at)) in seen:          # the same replacement read twice (a receipt and the stats stream)
            continue
        seen.add((old, new, str(at)))
        if old != expected:
            return expected, followed, ("A hand calibration replacement replaces payload %s, but the hand contract "
                                        "expects payload %s at that point." % (old, expected))
        expected = new
        followed.append(dict(replaced=old, replaced_by=new, at=entry.get("at_text") or at,
                             admission=(entry.get("admission") or {}).get("path")
                             if isinstance(entry.get("admission"), dict) else None))
    return expected, followed, None


def planner_monitor(run):
    """Persist search integrity and goal/effort tradeoffs for the dashboard and investigator."""
    def records():
        for path in Path(run).glob("causal_chain*.jsonl"):
            try:
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            yield json.loads(line)
                        except ValueError:
                            continue  # the trainer can be appending its last line
            except OSError:
                continue
    receipt, contract_origin, contract_error = planner_contract(run)
    required_schema = receipt.get("planner_selection_contract") == "best_evaluated_per_body_v1"
    # the contract's hand calibration, followed through the replacements the trainer admitted and recorded (read only
    # when the contract names one, so a run without a hand contract reads nothing more)
    hand_expected = receipt.get('hand_calibration_sha256')
    replacements, replacement_notes, replacement_error = (hand_calibration_replacements(run) if hand_expected
                                                          else ([], [], None))
    if replacement_error:           # a record that does not match the run (finding F1): nothing is followed; it stops
        followed = []
    else:
        hand_expected, followed, replacement_error = expected_hand_calibration(hand_expected, replacements)
    result = planner_integrity(records(), required_schema=required_schema,
                               expected_bodies=receipt.get("planner_bodies") if required_schema else None,
                               required_hand=receipt.get('hand_selection_contract')=='hand_joint_response_v1',
                               hand_calibration_sha256=hand_expected)
    result["selection_contract_required"] = required_schema
    result["selection_contract_origin"] = contract_origin
    result['hand_selection_contract_required']=receipt.get('hand_selection_contract')=='hand_joint_response_v1'
    result['hand_calibration_expected'] = hand_expected
    result['hand_calibration_replacements_followed'] = followed
    if replacement_notes:
        result['hand_calibration_replacement_notes'] = replacement_notes
    if replacement_error:
        # the replacement record is the cause a person needs first; a search's own stop reason follows it
        result['hand_calibration_replacement_error'] = replacement_error
        result['stop_reason'] = (replacement_error + ' Save and repair required.'
                                 + (' ' + result['stop_reason'] if result['stop_reason'] else ''))
        result['status'] = 'repair_required'
    if contract_error:
        result["selection_contract_error"] = contract_error
        result["stop_reason"] = result["stop_reason"] or contract_error + " Save and repair required."
        result["status"] = "repair_required"
    result.update(timestamp=time.time(), scope="recorded decisions retained in this run's causal logs")
    destination = Path(run) / "planner_monitor.json"
    pending = destination.with_suffix(".tmp")
    try:
        pending.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        pending.replace(destination)
    except OSError as error:
        print(f"PLANNER MONITOR WRITE FAILED: {error}", flush=True)
    return result


def _of_line(v, line):
    """A value as a percentage of a declared line, for a person (the line itself can be any
    number the declaration chose)."""
    if v is None:
        return "unmeasured"
    if line == 0:
        return "%s its line of 0" % ("above" if v > 0 else "below" if v < 0 else "at")
    return "%s of its line" % pct_plain(v / line)


def _declared_list():
    """The entries of <run>/declared_criteria.json, None when the file is absent, or the text "unreadable" (it cannot be
    opened or read, it is no JSON, or it holds no list). A file written with a byte-order mark (PowerShell 5.1's
    Out-File) reads as the same list (the re-check of Oct 5, 2026, finding N4)."""
    path = os.path.join(RUN, "declared_criteria.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8-sig") as f:
            crits = json.load(f)
    except (OSError, ValueError):
        return "unreadable"
    return crits if isinstance(crits, list) else "unreadable"


# ONE FAILED READ IS READ AGAIN (the round-two review of Oct 5, 2026, finding R6): a declared_criteria.json that cannot
# be read at one 5-minute check is read again at the next; it stops the run when it cannot be read at two checks running
# (a file another program held for a moment would otherwise stop a run whose lines hold)
_UNREADABLE_CHECKS = [0]


# A STOP LINE'S FIELD HAS AN HOUR TO APPEAR (the re-check of Oct 5, 2026, finding N1): production writes some fields
# only from a segment's third or later full stats record (contact_control.gate_self_mean is absent from the first 2 of
# every seg41 segment; proprio_token_repair.lesson_pace from the first 4 of seg41o_1431), and a field written now and
# then is absent for at most 5 records in a row there, against the 13 to 16 records an hour holds. So a stop entry's
# absent field stops the run only once the run's records span a whole hour after the entry's from_run_hour.
ABSENT_FIELD_WINDOW_S = 3600.0


def _absent_is_a_stop(c, run_h, span_s):
    return span_s >= ABSENT_FIELD_WINDOW_S and run_h >= float(c.get("from_run_hour", 0)) + ABSENT_FIELD_WINDOW_S / 3600


def declared_stops(w, run_h, span_s=0.0):
    """THE 5-MINUTE CHECK OF EVERY DECLARED STOP (the relaunch review of Oct 5, 2026, findings F3 and F4): each entry of
    mode "stop" in <run>/declared_criteria.json is judged at every 5-minute check, on the same window as the hourly sweep
    (the records of the hour before the newest), from its from_run_hour on. Before Oct 5 the declared stops were read
    only at the hourly sweeps, the first 3,060 s after the watcher started, so a repair run shorter than that had none
    read. A stop entry whose field no record of the window carries stops the run once the run's records (span_s: from
    its first record to its newest) span a whole hour after the entry's from_run_hour (_absent_is_a_stop): a misspelled
    path is stopped then, and a field production writes from a later record is not (finding N1). An entry that cannot be
    judged (no path, a line that is not a number) stops the run, naming it, instead of ending this check (finding N4).
    Prints only a stop."""
    crits = _declared_list()
    if crits is None:
        _UNREADABLE_CHECKS[0] = 0
        return None
    if crits == "unreadable":
        _UNREADABLE_CHECKS[0] += 1
        if _UNREADABLE_CHECKS[0] >= 2:
            return ("declared_criteria.json is unreadable at %d 5-minute checks running, so its stop lines cannot be "
                    "judged" % _UNREADABLE_CHECKS[0])
        print("  DECLARED CRITERIA UNREADABLE at this check - read again at the next one", flush=True)
        return None
    _UNREADABLE_CHECKS[0] = 0
    for i, c in enumerate(crits, 1):
        if not isinstance(c, dict) or c.get("mode") != "stop":
            continue
        try:
            if c.get("op", "<") not in ("<", ">"):
                raise ValueError("op %r is neither < nor >" % (c.get("op"),))
            if run_h < float(c.get("from_run_hour", 0)):
                continue
            # a stop field read as Infinity or NaN in the newest record is a breach at once (the round-two review's
            # R3): the window mean leaves non-finite numbers out, so it would read as absent for the hour of grace
            newest = field(w[-1], *str(c["path"]).split(".")) if w else None
            if isinstance(newest, float) and not math.isfinite(newest):
                return (f"declared stop criterion '{c.get('name')}': its field {c['path']} reads {newest} in the newest "
                        f"record (5-minute check)")
            v = wmean(w, *str(c["path"]).split("."))
            if v is None:
                if _absent_is_a_stop(c, run_h, span_s):
                    return (f"declared stop criterion '{c.get('name')}': its field {c['path']} is absent from the last "
                            f"hour's records, so its line cannot be judged")
                continue
            line = float(c["line"])
        except (KeyError, TypeError, ValueError) as error:
            return f"declared stop criterion {i} cannot be judged ({type(error).__name__}: {error})"
        below = c.get("op", "<") == "<"
        if v < line if below else v > line:
            return (f"declared criterion '{c.get('name')}': {c['path']} read {_of_line(v, line)}, "
                    f"{'under' if below else 'over'} its declared line (5-minute check)")
    return None


def declared_criteria(w, run_h, span_s=0.0):
    """Judge <run>/declared_criteria.json each sweep: every experiment ships its
    pass/fail lines, and the watcher - not a waiting human - reads them.
    Entries: {name, path (dot field), op ("<" or ">"), line, from_run_hour,
    mode ("alert"|"stop"), note}. Window means, like every other gauge. Printed as the
    value's percentage of its line (DESIGN section 11). A stop entry is also judged at every
    5-minute check (declared_stops); a stop entry whose field is absent stops the run once the
    run's records span a whole hour after its from_run_hour, as there."""
    crits = _declared_list()
    if crits is None:
        return None
    if crits == "unreadable":
        # the 5-minute checks count the failed reads and stop at the second running (declared_stops)
        print("  DECLARED CRITERIA UNREADABLE - fix declared_criteria.json", flush=True)
        return None
    for i, c in enumerate(crits, 1):
        if not isinstance(c, dict):
            continue
        try:
            if c.get("op", "<") not in ("<", ">"):
                raise ValueError("op %r is neither < nor >" % (c.get("op"),))
            line = float(c["line"])
            if run_h < float(c.get("from_run_hour", 0)):
                print(f"  criterion '{c.get('name')}': waits until run-hour {c['from_run_hour']}", flush=True)
                continue
            v = wmean(w, *str(c["path"]).split("."))
        except (KeyError, TypeError, ValueError) as error:
            # an entry that cannot be judged is named, never the end of the sweep (the round-two review's R6)
            print(f"  criterion {i} CANNOT BE JUDGED ({type(error).__name__}: {error}) - fix declared_criteria.json",
                  flush=True)
            if c.get("mode") == "stop":
                return f"declared stop criterion {i} cannot be judged ({type(error).__name__}: {error})"
            continue
        if v is None:
            print(f"  criterion '{c.get('name')}': field {c['path']} absent from the window", flush=True)
            if c.get("mode") == "stop" and _absent_is_a_stop(c, run_h, span_s):
                return (f"declared stop criterion '{c.get('name')}': its field {c['path']} is absent from the last "
                        f"hour's records, so its line cannot be judged")
            continue
        below = c.get("op", "<") == "<"
        breach = v < line if below else v > line
        verdict = "BREACHED" if breach else "holding"
        side = "under" if below else "over"
        print(f"  criterion '{c.get('name')}': {c['path']} reads {_of_line(v, line)} "
              f"(a reading {side} the line is a breach) -> {verdict}. {c.get('note', '')}", flush=True)
        if breach and c.get("mode") == "stop":
            return (f"declared criterion '{c.get('name')}': {c['path']} read {_of_line(v, line)}, "
                    f"{side} its declared line")
    return None


def field(r, *path):
    """One record's value at a dot path. A digit step indexes a list ("gaze_deg_err", "1")."""
    cur = r
    for k in path:
        if isinstance(cur, dict):
            cur = cur.get(k)
        elif isinstance(cur, (list, tuple)) and str(k).lstrip("-").isdigit():
            i = int(k)
            cur = cur[i] if -len(cur) <= i < len(cur) else None
        else:
            return None
    return cur


def wmean(rr, *path):
    """Window mean of a number at a dot path; records without it (or with a text, list or block
    there) do not count. True/False count as 1/0."""
    vs = [v for v in (field(r, *path) for r in rr) if isinstance(v, (int, float))
          and not (isinstance(v, float) and not math.isfinite(v))]
    return (sum(vs) / len(vs)) if vs else None


def wsum(rr, *path):
    vs = [v for v in (field(r, *path) for r in rr) if isinstance(v, (int, float))
          and not (isinstance(v, float) and not math.isfinite(v))]
    return sum(vs) if vs else None


def wkeys(rr, *path):
    """Every key of the per-sense (or per-channel) block at a dot path, in first-seen order."""
    seen = []
    for r in rr:
        d = field(r, *path)
        if isinstance(d, dict):
            seen += [k for k in d if k not in seen]
    return seen


def wlast(rr, *path):
    """The newest record's value at a dot path (texts, flags, lists), or None."""
    for r in reversed(rr):
        v = field(r, *path)
        if v is not None:
            return v
    return None


def whas(rr, *path):
    return any(field(r, *path) is not None for r in rr)


def rnd(v, nd=4):
    return round(v, nd) if v is not None else None


# ------------------------------------------------------------------ numbers a person reads
def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def show(v, nd=3, signed=False, pct=False):
    """A number for a person, or the word unmeasured. pct: a share shown as a percentage (the
    caller names both ends). Without pct, use only for real units (cm, degrees, GB, ms)."""
    if not _finite(v):
        return "unmeasured"
    if pct:
        return ("%+.1f%%" if signed else "%.1f%%") % (100.0 * v)
    return ("%+.*f" if signed else "%.*f") % (nd, v)


def pct_plain(share, signed=False):
    """A share as a whole percentage (2 significant digits under 1%)."""
    if not _finite(share):
        return "unmeasured"
    p = 100.0 * share
    sign = "+" if signed and p > 0 else ""
    if p == 0 or abs(p) >= 10:
        return "%s%.0f%%" % (sign, p)
    if abs(p) >= 1:
        return "%s%.1f%%" % (sign, p)
    return "%s%.2g%%" % (sign, p)


def of100(share, what=""):
    """A share 0..1 as a count a person can picture: 'N of every 100 <what>', or per 10,000
    when it is under 1 in 100 (so a small real share never reads 0)."""
    if not _finite(share):
        return "unmeasured"
    tail = (" " + what) if what else ""
    n = 100.0 * share
    if n == 0 or abs(n) >= 1:
        return "%d of every 100%s" % (round(n), tail)
    m = round(10000.0 * share)
    if m == 0:
        return "under 1 of every 10,000%s" % tail
    return "%d of every 10,000%s" % (m, tail)


def pairs100(auc):
    """An AUC as the count it means: of 100 (onset, quiet) pairs, how many the onset outranks."""
    if not _finite(auc):
        return "unmeasured"
    return "%d of every 100 pairs" % round(100.0 * auc)


def count(v):
    return "unmeasured" if not _finite(v) else "{:,}".format(int(round(v)))


def _beats(pred, base):
    """Share of the gap from a baseline's error to zero error that a prediction closes:
    1 - pred/base (0 = no better than the baseline, 1 = exact, negative = worse)."""
    if pred is None or base is None or base <= 0:
        return None
    return 1.0 - pred / base


def _has(w, block):
    return any(isinstance(r.get(block), dict) for r in w)


def _unmeasured(block, flag=None, why=None):
    why = why or f"{flag} off, or the run predates it"
    return f"  {block}: unmeasured - no record in the hour carries the block ({why})"


def _per_sense_beats(w, block, pred_key, base_key):
    parts = []
    for s in wkeys(w, block, pred_key):
        parts.append(f"{s} {show(_beats(wmean(w, block, pred_key, s), wmean(w, block, base_key, s)), signed=True, pct=True)}")
    return ", ".join(parts) or "unmeasured"


# ------------------------------------------------------------------ the gain-of-function blocks
def _causal_chain_lines(w):
    out = []
    alts = wmean(w, "causal_chain", "alternatives_n")
    chance = (1.0 / (alts + 1.0)) if alts is not None else None
    best = wmean(w, "causal_chain", "chosen_best_share")
    tied = wmean(w, "causal_chain", "chosen_tied_share")
    out.append(
        f"  causal chain: {count(wsum(w, 'causal_chain', 'latch_predictions'))} commands predicted "
        f"before they were sent and {count(wsum(w, 'causal_chain', 'planning_events') or wsum(w, 'causal_chain', 'planner_events'))} "
        f"planning events in the hour (summed over the hour's records).")
    if whas(w, "causal_chain", "tie_band") or tied is not None:
        best_txt = (f"the chosen plan was predicted to feel better than all {count(alts)} alternatives by more "
                    f"than the tie band in {of100(best, 'planning events')} (field chosen_best_share) and tied "
                    f"with the best of them in {of100(tied)} (chosen_tied_share)")
    else:
        best_txt = (f"the chosen plan was predicted to feel at least as good as all {count(alts)} "
                    f"alternatives in {of100(best, 'planning events')} (field chosen_best_share; this record "
                    f"counts a tie as best, so a planner whose evaluator predicts the same for every "
                    f"candidate reads 100)")
    spread, band = wmean(w, "causal_chain", "candidate_spread_mean"), wmean(w, "causal_chain", "tie_band")
    spread_txt = ("; the candidates' predicted wellness spread %s the tie band (under 1 time = the "
                  "evaluator cannot tell them apart)" % ("%.2g times" % (spread / band))
                  if _finite(spread) and _finite(band) and band > 0 else
                  "; the candidates' spread against the tie band is not in this record")
    out.append(
        f"  causal chain: {best_txt}; a random pick scores {of100(chance)}; goal line unmeasured (the "
        f"planner aims at an entry of the goal ring, not at predicted wellness){spread_txt}.")
    out.append(
        "  causal chain: whether the predicted change of wellness over the command's 250 ms matches the "
        f"realised change: {NOT_A_COUNT} (the record carries a correlation, which DESIGN section 11 "
        "does not print).")
    out.append(
        f"  causal chain: the prediction made BEFORE the command was sent beats 'nothing changes' "
        f"over 250 ms by: {_per_sense_beats(w, 'causal_chain', 'pred_err_latch', 'persist_err_latch')}; "
        f"over the next 50 ms by: {_per_sense_beats(w, 'causal_chain', 'pred_err_next', 'persist_err_next')} "
        f"(0% = no better than assuming nothing changes, 100% = exact - the goal, negative = worse)")
    fails = wsum(w, "causal_chain", "failures")
    dist = wmean(w, "causal_chain", "plan_exec_distance_cm")
    if dist is not None:
        exec_txt = (f"the command the body executed sat {dist:.1f} cm from the plan's own first command "
                    f"(0 cm = the body was sent the plan itself; field plan_exec_distance_cm)")
    else:
        exec_txt = (f"the plan's part in the command the body executed after smoothing "
                    f"{pct_plain(wmean(w, 'causal_chain', 'planner_share_of_executed'))} (0% = none of the "
                    f"plan, 100% = all of it; field planner_share_of_executed, plan_exec_distance_cm not in "
                    f"this record)")
    out.append(
        f"  causal chain: plans whose first command reached the body at a latch "
        f"{of100(wmean(w, 'causal_chain', 'reached_body_share'), 'plans')}; {exec_txt}"
        + (f"; {count(fails)} predictions failed to resolve" if fails else ""))
    return out


def _efference_lines(w):
    def body(h, unit, label):
        e = wmean(w, "efference", f"body_{unit}_err_{h}")
        d = wmean(w, "efference", f"body_{unit}_do_nothing_{h}")
        s = wmean(w, "efference", f"body_{unit}_shuffled_{h}")
        u = "cm" if unit == "cm" else "degrees"
        return (f"{label}: off by {show(e, 2)} {u}; 'nothing moves' is off by {show(d, 2)} {u} and "
                f"another body's command by {show(s, 2)} {u}; it closes {show(_beats(e, d), pct=True)} of "
                f"the gap from 'nothing moves' (0%) to exact (100%, the goal)")
    out = ["  efference body head (predicts each handle's move from the command it is given; "
           "below the other-body number means it uses its own command): "
           + body("next", "cm", "next 50 ms") + "; " + body("latch", "cm", "the command's 250 ms")
           + "; " + body("latch", "deg", "rotation over 250 ms")]
    g = [[wmean(w, "efference", f"gaze_deg_{k}", str(i)) for i in range(3)]
         for k in ("err", "do_nothing", "shuffled")]
    out.append(
        "  efference gaze head (predicts the eye's turn from the eyes' command): off by "
        + "/".join(show(v, 2) for v in g[0]) + " degrees at 50/100/150 ms; 'the eye stays put' is "
        "off by " + "/".join(show(v, 2) for v in g[1]) + " degrees and another body's command by "
        + "/".join(show(v, 2) for v in g[2]) + " degrees (0 degrees = exact, the goal); the heads' "
        "loss values are not printed (DESIGN section 11): the errors above are their reading")
    return out


def _gate_line_words(w, block, state):
    """Which line the gate is judged against, from the record (verify-fix-ops 2026-09-23: the
    words "twice the readings' own standard error" were fixed text; the model lane's controllability
    gate opens above 3 and closes below 1 of those errors and names both in gate_lines_noise)."""
    lines = wlast(w, block, "gate_lines_noise")
    if isinstance(lines, dict):
        which = "close" if state == "OPEN" else "open"
        n = lines.get(which)
        if _finite(n):
            return ("its %s line, %g time%s the readings' own standard error (it opens above %s and, "
                    "once open, closes below %s; field gate_lines_noise)"
                    % ("closing" if which == "close" else "opening", n, "" if n == 1 else "s",
                       "%g" % lines["open"] if _finite(lines.get("open")) else "an unrecorded number",
                       "%g" % lines["close"] if _finite(lines.get("close")) else "an unrecorded number"))
    return "the line the producer drew from the readings' own standard error (field gate_line)"


def _gate_text(w, block, key_open="gate_open", key_mean="gate_mean", key_line="gate_line",
               key_why="gate_why", advantage="the result agreed with what the model expected from the command "
                                            "it sent by %s more than with the control's expectation (the "
                                            "control named in the line above; on the agreement scale 0%% = "
                                            "unrelated, 100%% = exactly the expected effect)"):
    """The evidence gate's state from its own fields, without the producer's decimals."""
    op = wlast(w, block, key_open)
    mean, line = wlast(w, block, key_mean), wlast(w, block, key_line)
    state = "OPEN" if op else "shut" if op is not None else "state unmeasured"
    if _finite(mean) and _finite(line):
        why = ("over its last readings " + advantage % pct_plain(mean, signed=True)
               + ", against a line of %s: %s" % (pct_plain(line), _gate_line_words(w, block, state)))
    else:
        text = wlast(w, block, key_why)
        why = text if isinstance(text, str) and not _has_bare_decimal(text) else "its reason is in the record"
    return state, why


def _has_bare_decimal(text):
    """Whether a producer's own sentence carries a bare score decimal (then it is not quoted)."""
    try:
        from mimo_bridge.tools.check_no_bare_decimals import bare_decimals
    except ImportError:                 # cannot check: quote nothing that holds a digit
        return any(ch.isdigit() for ch in text)
    return bool(bare_decimals(text))


def _agreement(v, signed=True):
    """Agreement runs -1 (the opposite effect) .. 0 (unrelated) .. +1 (exactly as expected):
    shown as the % of the way from 'unrelated' to 'exactly as expected'."""
    return pct_plain(v, signed=signed)


def _controllability_lines(w):
    ba = wlast(w, "controllability", "by_author") or {}
    authors = ", ".join("%s %s" % (k.replace("_", " "), _agreement(wmean(w, "controllability", "by_author", k)))
                        for k in (ba if isinstance(ba, dict) else {})) or "unmeasured"
    src = wlast(w, "controllability", "source") or "unmeasured"
    agree = wmean(w, "controllability", "agreement_mean")
    ctl = wmean(w, "controllability", "control_mean")
    diff = wmean(w, "controllability", "difference_mean")
    diff_src = "difference_mean"
    # WHICH CONTROL control_mean IS (verify-fix-ops 2026-09-23). The field contract makes it the
    # shuffled-ORDER control (this body's state with another body's order; control_kind "shuffled
    # order", difference_mean beside it). The first build's control_mean (wellness_channels_v1, the
    # committed producer until the model lane's fix merges) is another body's EXPECTATION against
    # this body's result, the control the b3 review showed an order-blind expectation passes; this
    # line called both "the shuffled-order control".
    shuffled = (wlast(w, "controllability", "control_kind") == "shuffled order"
                or whas(w, "controllability", "difference_mean"))
    ctl_words = ("with ANOTHER body's command from this body's state (the shuffled-command control, field "
                 "control_mean)" if shuffled else
                 "with ANOTHER body's expectation against this body's result (the first build's control, "
                 "field control_mean; the shuffled-command control is not in this record)")
    if diff is None and agree is not None and ctl is not None:
        diff, diff_src = agree - ctl, ("agreement_mean minus control_mean, computed here: not counted on "
                                       "exactly the same commands")
    state, why = _gate_text(w, "controllability")
    value = wmean(w, "controllability", "value_mean")
    into = ("0 while the gate is shut (the channel's weight is 0 until its evidence gate opens)"
            if state == "shut" else
            "its running agreement %s, weighted by the ranking" % _agreement(value) if state == "OPEN" else
            "its running agreement %s, if the gate is open (its state is not in this record)" % _agreement(value))
    return [
        "  controllability (did the body do what the model expected from its own command?): measured on "
        f"{of100(wmean(w, 'controllability', 'measured_share'), 'commands')} (the rest expected, or were "
        f"commanded, less than holding still drifts; stillness is never counted); on those the result "
        f"agreed {_agreement(agree)} of the way from 'unrelated to what it expected' (0%) to 'exactly the "
        f"expected effect' (100%, negative = the opposite); {ctl_words} {_agreement(ctl)}; own command minus "
        f"control {_agreement(diff)} ({diff_src}); goal line "
        f"unmeasured. By author: {authors}. Expected effect from: {src}.",
        f"  controllability: evidence gate {state} - {why}; value into the goal choice {into}"]


def _learning_progress_lines(w):
    parts = []
    for s in wkeys(w, "learning_progress", "per_sense"):
        parts.append(
            f"{s} {pct_plain(wmean(w, 'learning_progress', 'per_sense', s), signed=True)} (the model now "
            f"makes {pct_plain(wmean(w, 'learning_progress', 'ratio_fast', s))} of the error of 'nothing "
            f"changes', earlier {pct_plain(wmean(w, 'learning_progress', 'ratio_slow', s))})")
    ctl = wmean(w, "learning_progress", "control_mean")
    lo, hi = LP_FROZEN_RANGE
    ctl_txt = (f"same-model-step control {pct_plain(ctl, signed=True)} (field control_mean)" if ctl is not None else
               f"no same-model-step control in this record; frozen weights, which cannot learn, read "
               f"{pct_plain(lo, signed=True)} to {pct_plain(hi, signed=True)} (b3-wellness, 2026-09-22), so "
               f"a reading inside that range is not evidence of learning")
    return [
        "  learning progress (is prediction getting better, judged against 'nothing changes' so an "
        f"easier world does not count?): {'; '.join(parts) or 'unmeasured'} (progress 0% = no change, "
        f"+100% = the error relative to 'nothing changes' fell to zero, negative = it rose; the error "
        f"share reads 0% = exact, 100% = no better than assuming nothing changes); {ctl_txt}; goal line "
        f"unmeasured; value into the goal choice "
        f"{pct_plain(wmean(w, 'learning_progress', 'value_mean'), signed=True)} on the same progress scale"]


def _wellness_rank_lines(w):
    mode = wlast(w, "wellness_rank", "mode") or "unmeasured"

    def top(key, k=3):
        vals = {c: wmean(w, "wellness_rank", key, c) for c in wkeys(w, "wellness_rank", key)}
        return sorted(((c, v) for c, v in vals.items() if v is not None), key=lambda kv: -abs(kv[1]))[:k]

    def parts(vals):
        return ", ".join("%s %d parts in 100" % (c, round(100 * v)) for c, v in vals) or "unmeasured"
    var, mar = top("variance_share"), top("winner_margin_share")
    weights = wlast(w, "wellness_rank", "weight_share_by_channel")
    w_src = "weight_share_by_channel"
    if not isinstance(weights, dict):
        raw = wlast(w, "wellness_rank", "weights_in_force") or wlast(w, "wellness_rank", "weights")
        w_src = "computed from weights_in_force" if wlast(w, "wellness_rank", "weights_in_force") else "computed from weights"
        tot = sum(v for v in raw.values() if _finite(v)) if isinstance(raw, dict) else 0
        weights = {c: v / tot for c, v in raw.items() if _finite(v)} if tot else None
    shown = {c for c, _ in var + mar}
    wt = (", ".join("%s %d of 100" % (c, round(100 * weights[c])) for c in weights if c in shown)
          if isinstance(weights, dict) and weights else "unmeasured")
    out = [
        f"  goal ranking ({mode}): each channel's part of how the candidates' values spread: {parts(var)}; "
        f"its part of the winning goal's lead over the runner-up: {parts(mar)} (a part below 0 pushed "
        f"against the winner, above 100 pushed for it by more than the whole lead); the same channels' "
        f"share of the weights ({w_src}): {wt} - a channel whose part far exceeds its weight wins by the size "
        f"of its numbers; candidates the integrity gate kept out of the goal choice "
        f"{of100(wmean(w, 'wellness_rank', 'integrity_gated_share'), 'candidates')}"]
    if whas(w, "wellness_rank", "goal_age"):
        out.append(
            "  goal ranking: the chosen goals sat on average "
            f"{pct_plain(wmean(w, 'wellness_rank', 'goal_age', 'mean_share_back'))} of the way from the newest "
            f"entry of the goal ring (0%) to the oldest (100%); {of100(wmean(w, 'wellness_rank', 'goal_age', 'oldest_tenth_share'), 'goals')} "
            f"came from the oldest tenth and {of100(wmean(w, 'wellness_rank', 'goal_age', 'newest_tenth_share'))} from "
            f"the newest tenth (an even pick: 10 each), over {count(wsum(w, 'wellness_rank', 'goal_age', 'events'))} choices")
    else:
        out.append("  goal ranking: goal age not in this record (the model lane adds wellness_rank.goal_age)")
    return out


def _contact_lines(w):
    cc = lambda k: wmean(w, "contact_control", k)          # noqa: E731
    paired = cc("success_paired")
    if paired is not None:
        success = (f"foreseen before they began, on the SAME onsets: {of100(paired, 'self-touch onsets')} with "
                   f"this body's own command (success_paired) against {of100(cc('success_control'))} with "
                   f"another body's command (success_control)")
    else:
        success = (f"foreseen before they began: {of100(cc('success_self'), 'self-touch onsets')} (success_self, "
                   f"counted on every onset) beside {of100(cc('success_control'))} with another body's command "
                   f"(success_control, counted only on the model steps the control ran - not the same onsets; "
                   f"success_paired is not in this record)")
    # outside touches, the same pairing (verify-fix-ops 2026-09-23: this half still put success_world,
    # counted on every onset, beside success_control_world, counted only where the control ran -
    # b4 review M4's mismatch on the world side; ContactTracker.summary writes success_paired_world
    # for the same onsets)
    world_ctl, world_paired = cc("success_control_world"), cc("success_paired_world")
    if world_paired is not None:
        world = (f"outside touches on the SAME onsets: {of100(world_paired)} with this body's own command "
                 f"(success_paired_world) against {of100(world_ctl)} with another body's command "
                 f"(success_control_world)")
    elif world_ctl is not None:
        world = (f"outside touches {of100(cc('success_world'))} (success_world, every onset) beside "
                 f"{of100(world_ctl)} with another body's command (success_control_world, only the model steps the "
                 f"control ran - not the same onsets; success_paired_world is not in this record)")
    else:
        world = f"outside touches {of100(cc('success_world'))} (success_world; no control in this record)"
    out = [
        f"  contact control: onsets in the hour's records (summed) - self "
        f"{count(wsum(w, 'contact_control', 'onsets_self'))}, world {count(wsum(w, 'contact_control', 'onsets_world'))}; "
        f"{success}; {world} (0 = never, 100 = every onset); goal line unmeasured"]
    ons = cc("onset_auc_self")
    if ons is not None:
        head = (f"on patches untouched the model step before, it ranks a real own-body touch onset above a quiet "
                f"patch in {pairs100(ons)} (onset_auc_self) against {pairs100(cc('control_onset_auc_self'))} "
                f"for the base-rate control (control_onset_auc_self); outside touches "
                f"{pairs100(cc('onset_auc_world'))} against {pairs100(cc('control_onset_auc_world'))}")
    else:
        head = (f"it ranks a patch touched at a model step above an untouched one in {pairs100(cc('head_auc'))} (head_auc) "
                f"against {pairs100(wmean(w, 'contact_control', 'control_auc_bits', 'any'))} for the base-rate "
                f"control (control_auc_bits.any) - this counts touches that simply continue, so it measures "
                f"persistence more than foresight; onset_auc_self is not in this record")
    out.append(f"  contact head: {head} (a coin flip scores 50, every onset foreseen 100); goal line unmeasured")
    g_self, g_world = wlast(w, "contact_control", "gate_self_open"), wlast(w, "contact_control", "gate_world_open")
    if g_self is not None or g_world is not None:
        gates = (f"evidence gates: self {'OPEN' if g_self else 'shut'}, world {'OPEN' if g_world else 'shut'} "
                 f"(gate_self_open, gate_world_open)")
        shut = not g_self and not g_world
    else:
        g = wlast(w, "contact_control", "gate_open")
        gates = (f"evidence gate {'OPEN' if g else 'shut' if g is not None else 'state unmeasured'} (one "
                 f"gate pooling self and outside touches: gate_open)")
        # a gate whose state the record does not carry is not "shut": print what was paid
        shut = g is not None and not g
    paid = cc("paid_mean")
    paid_txt = ("0 while the gate is shut" if shut else
                "%s points per 1,000 model steps of one body (2 per foreseen self-touch, 1 per foreseen outside touch, "
                "times the habituation factor)" % count(None if paid is None else 1000.0 * paid))
    out.append(
        f"  contact control: {gates}; paid into wellness {paid_txt}; habituation: the average pay factor "
        f"{pct_plain(cc('habituation_mean'))} (100% = a new pair, lower = repeated)")
    return out


def _synergy_lines(w):
    deg = lambda v: None if not _finite(v) else math.degrees(v)          # noqa: E731
    exp = wmean(w, "synergy", "explained_share")
    counts = []
    for author in sorted(wkeys(w, "synergy", "following_by_author")):
        n = wsum(w, "synergy", "following_by_author", author, "part_orders")
        closer = wsum(w, "synergy", "following_by_author", author, "closer")
        chance = wsum(w, "synergy", "following_by_author", author, "chance_count")
        if n and closer is not None and chance is not None:
            counts.append(f"{author.replace('_', ' ')}: {count(closer)} of {count(n)} body-part commands "
                          f"ended closer, against {show(chance, 1)} expected by the random-axis control "
                          f"({show(100 * (closer - chance) / n, 1)} more per 100 commands)")
    if counts:
        return ["  motor synergy, executed commands: " + "; ".join(counts) + ".",
                "  the random-axis control is geometric, not a physical replay. These counts measure the "
                "executed authors; learning the map does not mean the map controlled them. Improvement from "
                "using it and faster learning need paired comparisons.",
                "  the map reproduces " + pct_plain(exp, signed=True) + " of the observed posture changes "
                "(0% = no better than no change, 100% = the whole change); this alone does not show control."]
    return [
        "  motor synergy (a compact intention turned into coordinated handle targets): how well the body "
        f"follows the commands the synergy wrote - item 5's measurement - is {NOT_A_COUNT} (the record "
        "carries a direction match, which DESIGN section 11 does not print; b5's physics check "
        "tools/synergy_following.py measures it); commands graded "
        f"{count(wsum(w, 'synergy', 'orders'))}; uses {show(wmean(w, 'synergy', 'effective_dim'), 1)} of its "
        f"{count(wmean(w, 'synergy', 'dim'))} directions; the turn each command set for the body handles was "
        f"{show(deg(wmean(w, 'synergy', 'cmd_mag_rad')), 1)} degrees at issue; babble turned each handle "
        f"{show(deg(wmean(w, 'synergy', 'babble_turn_rad')), 1)} degrees against today's wobble "
        f"{show(deg(wmean(w, 'synergy', 'babble_target_rad')), 1)} degrees, moved it "
        f"{show(wmean(w, 'synergy', 'babble_move_cm'), 2)} cm, and sat on a reach limit in "
        f"{of100(wmean(w, 'synergy', 'babble_at_rail'), 'babble model steps')}.",
        "  motor synergy, logged beside it (not item 5's measure): it reproduces "
        f"{pct_plain(exp, signed=True)} of the posture changes the body made (0% = no better than asking for no "
        "change, 100% = the whole change); a map can do this while controlling nothing"]


def _vestibular_lines(w):
    roll = wmean(w, "vestibular", "trunk_roll_deg")
    steps, real = wsum(w, "vestibular", "steps"), wsum(w, "vestibular", "real_steps")
    invalid = wsum(w, "vestibular", "frames_invalid")
    graded = (f"the sense was a real reading on {count(real)} of {count(steps)} model steps (real_steps against "
              f"steps; the rest were stale: the first frame after a resume, or a missing or non-finite reading "
              f"that makes the whole batch stale), with {count(invalid)} invalid body-frames"
              if steps is not None else "real readings against steps not in this record")
    return [
        f"  vestibular: {graded}; its prediction beats 'nothing changes' by "
        f"{show(wmean(w, 'vestibular', 'improvement_over_persistence'), signed=True, pct=True)} (0% = "
        f"no better, 100% = exact - the goal, negative = worse)",
        f"  vestibular: trunk turned {show(None if roll is None else abs(roll), 0)} degrees from lying "
        f"on its back (0 = flat on its back, 90 = on its side, 180 = belly down); trunk pitch "
        f"{show(wmean(w, 'vestibular', 'trunk_pitch_deg'), 0)} degrees (0 = horizontal, 90 = head straight "
        f"up); head tilt from lying on its back {show(wmean(w, 'vestibular', 'head_tilt_deg'), 0)} degrees "
        f"(read from gravity: 0 = the head as when lying flat on its back, 90 = a quarter turn from that, "
        f"180 = upside down from it); the trunk carries {of100(wmean(w, 'vestibular', 'support_trunk'), 'parts')} "
        f"of the external support; {of100(wmean(w, 'vestibular', 'limit_load'), 'joints')} sit within 2.0 "
        f"degrees of a joint limit"]


def _next_frame_lines(w):
    keys = wkeys(w, "next_frame_norm", "weight_effective")
    vals = {s: wmean(w, "next_frame_norm", "weight_effective", s) for s in keys}
    vals = {s: v for s, v in vals.items() if _finite(v)}
    mean = (sum(vals.values()) / len(vals)) if vals else None
    parts = ", ".join("%s %s" % (s, pct_plain(v / mean)) for s, v in vals.items()) if mean else ""
    return [
        "  next-frame balance: each sense's lesson weight after dividing by its own 'nothing changes' "
        "error, as a share of the senses' average weight: " + (parts or "unmeasured") + " (100% each = "
        "every sense pushed equally to beat 'nothing changes'; before this, audio carried 67 of every 100 "
        "parts of the next-frame loss)"]


def gof_lines(w):
    """The gain-of-function blocks (DESIGN.md section 3) as plain lines, window means."""
    blocks = (
        ("causal_chain", "causal chain", "--causal-chain", _causal_chain_lines),
        ("efference", "efference heads", "--efference-heads", _efference_lines),
        ("controllability", "controllability", "--controllability", _controllability_lines),
        ("learning_progress", "learning progress", "--learning-progress", _learning_progress_lines),
        ("wellness_rank", "goal ranking", "--wellness-rank normalized", _wellness_rank_lines),
        ("contact_control", "contact control", "--contact-head / --contact-control-wellness", _contact_lines),
        ("synergy", "motor synergy", "--motor-synergy", _synergy_lines),
        ("vestibular", "vestibular", "--vestibular-sense", _vestibular_lines),
        ("next_frame_norm", "next-frame balance", "--next-frame-norm persistence", _next_frame_lines),
    )
    out = []
    for key, name, flag, build in blocks:
        if not _has(w, key):
            out.append(_unmeasured(name, flag))
            continue
        out += _safe(build, w, name=name)
    return out


def _safe(build, *a, name="a line"):
    """A line builder that raises prints why and never stops the check (the stop rules follow)."""
    try:
        return build(*a)
    except Exception as error:          # an informational line must never cost a stop rule
        return ["  LINE FAILED (%s): %s: %s" % (name, type(error).__name__, error)]


# --------------------------------------------------------------------- liveness (DESIGN section 10)
def _gelu_slope(x):
    """The slope of GELU, x * Phi(x), at x: the share of a change in a unit's input its output passes on."""
    density = math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) + x * density


def live_line_words(line):
    """What the live-input line means, in words: the printed line never shows a hidden unit's raw
    input, a number a reader cannot place (the readout scanner's rule, 2026-09-23), and uses the
    explainer's and the probes' words (tools/gof_probe_harness.LIVE_RULE). Below -1.41 (the square
    root of 2) GELU's slope only shrinks toward its flat low end, so the line is described by how far
    a change in the input moves the output there: at DESIGN section 10's -3, 1.2% of the change (for
    a large input, 100%); at -2, 8.5%."""
    if not _finite(line) or line > -math.sqrt(2.0):
        return "input rises above the line the whole-model report names (field liveness.line)"
    p = 100.0 * abs(_gelu_slope(float(line)))
    if p <= 0:
        return "input rises above the line the whole-model report names (field liveness.line)"
    digits = max(0, 1 - int(math.floor(math.log10(p))))
    up = math.ceil(p * 10 ** digits) / 10 ** digits          # rounded up, so "less than" stays true
    return ("input leaves the flat low end of its activation function (GELU), where a change in the input moves "
            "the output by less than %s%% of the change (100%% = the output moves as much as the input)"
            % ("%g" % up))


def liveness_lines(w):
    """Every liveness and pool_spread sub-block of any block, and the top-level liveness block.

    liveness {layer: share of units whose input rose above -3 on the window's samples}: judged
    against DESIGN section 10's line (50 of every 100 units). pool_spread {pool: attention entropy
    as a share of the even maximum}: printed, with no line declared."""
    layers, pools = {}, {}
    for r in w:
        for k, v in r.items():
            if not isinstance(v, dict):
                continue
            if k == "liveness":
                sub_l, sub_p = v.get("layers"), v.get("pools")
                owner = "whole model"
            else:
                sub_l, sub_p = v.get("liveness"), v.get("pool_spread")
                owner = k
            for target, sub in ((layers, sub_l), (pools, sub_p)):
                if isinstance(sub, dict):
                    for name, share in sub.items():
                        if _finite(share):
                            target.setdefault((owner, name), []).append(float(share))
    if not layers and not pools:
        return ["  liveness: unmeasured - no record in the hour carries a liveness or pool_spread report "
                "(DESIGN section 10; the whole-model block comes with the repair agent after b6)"]
    out = []
    top = [r.get("liveness") for r in w if isinstance(r.get("liveness"), dict)]
    # the input line the whole-model block says it used (the field contract's liveness.line), else
    # DESIGN section 10's -3
    input_line = top[-1].get("line") if top and _finite(top[-1].get("line")) else LIVE_INPUT_LINE
    if layers:
        mean = {k: sum(v) / len(v) for k, v in layers.items()}
        low = sorted((k for k, v in mean.items() if v < LIVE_SHARE_LINE), key=lambda k: mean[k])
        by_owner = {}
        for (owner, name), v in mean.items():
            by_owner.setdefault(owner, []).append("%s %d of 100" % (name, round(100 * v)))
        out.append(
            "  liveness (a unit is live when its %s, on the window's samples; DESIGN section 10's line: 50 of "
            "every 100 units live): " % live_line_words(input_line)
            + "; ".join("%s: %s" % (o, ", ".join(p)) for o, p in by_owner.items()))
        if low:
            out.append(
                "  LIVENESS ALERT: %d layer%s under the line of 50 live units in 100 - %s. "
                "Fewer than half the units responded on the sampled inputs; this does not mean the whole "
                "layer stopped working. Two complete low-response reports require a checkpoint and stop "
                "at the next five-minute check. With revival on, a repair layer is judged only after its "
                "repair window ends; with revival off (--liveness-rule none) no window comes, so every "
                "layer is judged from its first two complete reports."
                % (len(low), "" if len(low) == 1 else "s",
                   ", ".join("%s %s %d of 100" % (o, n, round(100 * mean[(o, n)])) for o, n in low)))
    if pools:
        out.append(
            "  pool spread (attention spread over the elements a pool reads, as a share of an even spread: 0% "
            "= it reads one element, 100% = even; no line declared): "
            + ", ".join("%s %s %s" % (o, n, pct_plain(sum(v) / len(v))) for (o, n), v in pools.items()))
    if top:
        revived = top[-1].get("revived")
        if revived:
            out.append("  liveness: layers revived by the repair (the record's 'revived'): %s"
                       % ", ".join(map(str, revived)))
    return out


# --------------------------------------------------------------------- machine and storage
def live_machine():
    """This program's own reading now: RAM available, commit charge and limit, card memory used
    by all programs. A reading the platform cannot give stays None (printed unmeasured)."""
    out = dict(ram_available_gb=None, commit_gb=None, commit_limit_gb=None, gpu_used_mib=None,
               gpu_total_mib=None)
    try:
        import psutil
        out["ram_available_gb"] = psutil.virtual_memory().available / 1e9
    except (ImportError, OSError, AttributeError):
        pass
    try:
        from mimo_bridge.tools.commit_guard import commit_reading
        c = commit_reading()
        if c is not None:
            out["commit_gb"], out["commit_limit_gb"] = c[0], c[1]
    except (ImportError, OSError):
        pass
    try:
        from mimo_bridge.machine_stats import nvidia_smi_memory
        g = nvidia_smi_memory()
        if g is not None:
            out["gpu_used_mib"], out["gpu_total_mib"] = g
    except ImportError:
        pass
    return out


def machine_lines(w, live):
    out = []
    m = lambda k: wmean(w, "machine", k)          # noqa: E731
    if _has(w, "machine"):
        out.append(
            f"  machine (the trainer's records, hour means): RAM free for new work {show(m('ram_available_gb'), 1)} GB "
            f"(0 GB = none left); commit charge {show(m('commit_gb'), 1)} of {show(m('commit_limit_gb'), 1)} GB "
            f"(at the limit Windows refuses the next process that asks for memory); the trainer holds "
            f"{show(m('rss_gb'), 1)} GB; graphics card {show(m('gpu_used_mib'), 0)} of {show(m('gpu_total_mib'), 0)} "
            f"MiB used by all programs (past the total, memory spills into shared system memory: 32 bodies "
            f"at 8,487 MiB on the 8,188 MiB card, 2026-09-17); one model step {show(m('step_ms'), 0)} ms "
            f"({show(m('model_ms'), 0)} ms model, {show(m('env_ms'), 0)} ms bodies); page faults "
            f"{show(m('page_faults_per_s'), 0)} a second")
    else:
        out.append(_unmeasured("machine (the trainer's records)",
                               why="the trainer's code predates the machine block"))
    head = (None if live["commit_gb"] is None or live["commit_limit_gb"] is None
            else live["commit_limit_gb"] - live["commit_gb"])
    out.append(
        f"  machine (read now by this watcher): RAM available {show(live['ram_available_gb'], 1)} GB "
        f"(stop line {RAM_STOP_GB} GB); commit charge {show(live['commit_gb'], 1)} of {show(live['commit_limit_gb'], 1)} "
        f"GB, {show(head, 1)} GB of headroom (the commit guard keeps {COMMIT_RESERVE_GB:.0f} GB); graphics "
        f"card {show(live['gpu_used_mib'], 0)} of {show(live['gpu_total_mib'], 0)} MiB used by all programs")
    if head is not None and head < COMMIT_RESERVE_GB:
        out.append(
            f"  COMMIT ALERT: {head:.1f} GB between the commit charge and its limit, inside the commit "
            f"guard's {COMMIT_RESERVE_GB:.0f} GB reserve - the next process that asks for memory can be "
            f"refused (2026-09-21: 77.3 of 78.1 GB killed the supervisor). Training continues.")
    return out


def archive_intake(archive=None, now=None):
    """GB a day moved onto the D: archive, from the archiver's two manifests: over the last 24 h,
    and over the last 7 days (or the manifests' whole span when shorter) per day."""
    archive = ARCHIVE if archive is None else Path(archive)
    now = time.time() if now is None else now
    rows_ = []
    for mf in (archive / "move_manifest.jsonl", archive / "caregiver_tapes" / "tape_move_manifest.jsonl"):
        try:
            with open(mf, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("event") == "moved" and isinstance(r.get("bytes"), (int, float)) \
                            and isinstance(r.get("time"), (int, float)):
                        rows_.append((float(r["time"]), float(r["bytes"])))
        except OSError:
            continue
    if not rows_:
        return None
    first = min(t for t, _ in rows_)
    day = sum(b for t, b in rows_ if now - 86400.0 < t <= now) / 1e9
    # the window starts 7 days back, or at the first recorded move when the history is shorter;
    # rows strictly after its start count, so the first move marks where the history begins
    # instead of adding bytes that accumulated before it
    span_days = min(7.0, max(1.0, (now - first) / 86400.0))
    start = now - span_days * 86400.0
    week = sum(b for t, b in rows_ if start < t <= now) / 1e9 / span_days
    return dict(day_gb=day, week_gb_per_day=week, span_days=span_days, rows=len(rows_))


def archiver_process(pid):
    """(alive, watching) for the process a status names: alive = it exists and runs the archiver
    (tools/archive_checkpoints.is_archiver_command), watching = its command carries --watch.
    (None, None) when this platform cannot say."""
    try:
        import psutil
    except ImportError:
        return None, None
    try:
        cmd = psutil.Process(int(pid)).cmdline()
    except (psutil.Error, OSError, ValueError, TypeError):
        return False, False
    named = any("archive_checkpoints" in str(x) for x in cmd)
    return named and "--dry-run" not in cmd, named and "--watch" in cmd


def watching_archivers():
    """PIDs of running processes that are a watching archiver; None when this cannot be read."""
    try:
        import psutil
    except ImportError:
        return None
    found = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = p.info["cmdline"] or []
        except (psutil.Error, OSError):
            continue
        if any("archive_checkpoints" in str(x) for x in cmd) and "--watch" in cmd \
                and "--dry-run" not in cmd:
            found.append(p.info["pid"])
    return found


def _status_errors(st):
    errs = []
    if st.get("error"):
        errs.append(("the checkpoint pass", str(st["error"])))
    tapes = st.get("tapes")
    if isinstance(tapes, dict) and tapes.get("error"):
        errs.append(("the recordings pass", str(tapes["error"])))
    return errs


def storage_lines(now=None, run_args=None, saved=None, archive=None, process=None, watchers=None):
    """process(pid) -> (alive, watching) and watchers() -> [pid] default to the live process
    table (archiver_process, watching_archivers); tests pass their own."""
    now = time.time() if now is None else now
    run_args = RUN_ARGS if run_args is None else run_args
    saved = SAVED if saved is None else Path(saved)
    archive = ARCHIVE if archive is None else Path(archive)
    process = archiver_process if process is None else process
    watchers = watching_archivers if watchers is None else watchers
    out = []
    rec = run_args.get("caregiver_record_dir") or ""
    if rec:
        drive = os.path.splitdrive(os.path.abspath(rec))[0] + os.sep
        try:
            free = shutil.disk_usage(drive).free / 1e9
            free_txt = "has %.1f GB free" % free
        except OSError as error:
            free_txt = "cannot be read (%s)" % error
        out.append(f"  storage: the caregiver-record drive {drive} {free_txt} (stop line "
                   f"{DISK_STOP_GB:.0f} GB: the recorder and every save's ~600 MB episode companion write there)")
    else:
        out.append("  storage: caregiver-record drive unmeasured (this run records no caregiver)")
    drive = archive.anchor or str(archive)
    try:
        d_free = shutil.disk_usage(drive).free / 1e9
        d_error = None
    except OSError as error:
        d_free, d_error = None, error
    if d_error is not None:
        out.append(f"  storage: the archive drive {drive}: free space unreadable ({d_error})")
        out.append(
            f"  ARCHIVE DRIVE ALERT: the archive drive {drive} cannot be read ({d_error}). The archiver "
            f"cannot move anything, so C: takes the whole {C_TAKES_GB_PER_H} GB an hour until it reaches the "
            f"8 GB stop line; D: is a USB disk that has dropped off before. Reconnect it or name another "
            f"archive. Training continues.")
    else:
        intake = archive_intake(archive, now)
        rate = None if intake is None else max(intake["day_gb"], intake["week_gb_per_day"])
        runway = None if not rate else d_free / rate
        out.append(
            f"  storage: the archive drive {drive} has {d_free:.1f} GB free; intake "
            + ("unmeasured (no move manifest readable)" if intake is None else
               f"{intake['day_gb']:.2f} GB in the last 24 h and {intake['week_gb_per_day']:.2f} GB a day over the "
               f"last {intake['span_days']:.1f} days")
            + (f"; runway {runway:.1f} days at the larger rate" if runway is not None else "; runway unmeasured")
            + f" (alert under {D_RUNWAY_ALERT_DAYS:.0f} days)")
        if runway is not None and runway < D_RUNWAY_ALERT_DAYS:
            out.append(
                f"  STORAGE ALERT: the archive drive fills in {runway:.1f} days at {rate:.1f} GB a day. When it is "
                f"full the archiver stops and C: takes the whole {C_TAKES_GB_PER_H} GB an hour. Nothing thins "
                f"the archive; thinning deletes and is the operator's decision. Training continues.")
    status_path = saved / "checkpoint_archive_status.json"
    try:
        st = json.loads(status_path.read_text(encoding="utf-8"))
        age = now - float(st.get("time"))
    except (OSError, ValueError, TypeError, AttributeError):
        st, age = None, None
    if st is None:
        out.append(f"  ARCHIVER ALERT: no readable status file {status_path} - nothing is known to move "
                   f"checkpoints off C:, which then grows {C_TAKES_GB_PER_H} GB an hour. Training continues.")
        return out
    tapes = st.get("tapes") or {}
    pid = st.get("pid")
    out.append(
        f"  storage: the checkpoint archiver's status was written {age:.0f} s ago by pid "
        f"{pid if pid is not None else 'unrecorded'} ({'watching' if st.get('watch') else 'one-off pass' if 'watch' in st else 'watch unrecorded'}"
        f"{', dry run' if st.get('dry_run') else ''}): moved {count(st.get('moved'))} checkpoints and "
        f"{count(tapes.get('moved_files'))} recording files that pass; C: then had "
        f"{show(st.get('free_gb'), 1)} GB free")
    for what, err in _status_errors(st):
        out.append(
            f"  ARCHIVER ALERT: {what} failed: \"{err}\". The status file is fresh but nothing moved; C: takes "
            f"the {C_TAKES_GB_PER_H} GB an hour the archiver would move. Training continues.")
    if "watch" not in st or "pid" not in st:
        # a status written by code older than 2026-09-22 carries neither field: age alone
        judged = True
        out.append("  storage: this status predates the watch and pid fields; its age is judged alone")
    else:
        alive, watching = process(pid)
        judged = bool(st.get("watch")) and not st.get("dry_run") and alive is not False
        if st.get("watch") and alive is False:
            out.append(
                f"  ARCHIVER ALERT: the watching archiver that wrote the status (pid {pid}) is not running; "
                f"checkpoints stay on C: at {C_TAKES_GB_PER_H} GB an hour until one starts. The supervisor "
                f"starts one only when none runs. Training continues.")
        elif not judged:
            live = watchers()
            if live:
                out.append(
                    f"  storage: the newest status came from a one-off or dry pass, not the watching archiver "
                    f"(pid {', '.join(map(str, live))} runs with --watch); its heartbeat is judged from its "
                    f"next status")
            elif live is not None:
                out.append(
                    f"  ARCHIVER ALERT: the newest status came from a one-off or dry pass and no watching "
                    f"archiver is running - nothing moves checkpoints off C:, which then grows "
                    f"{C_TAKES_GB_PER_H} GB an hour. Training continues.")
            else:
                out.append("  storage: the newest status came from a one-off or dry pass; whether a "
                           "watching archiver runs cannot be read on this machine")
    if judged and age > ARCHIVER_STALE_S:
        out.append(
            f"  ARCHIVER ALERT: the archiver's status is {age:.0f} s old, past {ARCHIVER_STALE_S:.0f} s - it "
            f"has missed two passes or stopped; checkpoints then stay on C: at {C_TAKES_GB_PER_H} GB an "
            f"hour. The supervisor starts one only when none runs. Training continues.")
    return out


def load_baselines():
    try:
        with open(BASE_PATH, encoding="utf-8") as f:
            b = json.load(f)
        print(f"own baselines loaded from {BASE_PATH} (measured at step {b.get('measured_at_step')}, "
              f"run-hour {b.get('run_hour')})", flush=True)
        return b
    except (OSError, ValueError):
        return {}


def newest_checkpoint_age_h():
    """Hours since the newest COMPLETE checkpoint (a folder with meta.json) in the run's
    configured storage; None when there is none yet."""
    folder = RUN_ARGS.get("checkpoint_storage") or os.path.join(RUN, "checkpoints")
    newest = None
    try:
        for name in os.listdir(folder):
            meta = os.path.join(folder, name, "meta.json")
            if name.startswith("ckpt_step_") and os.path.exists(meta):
                t = os.path.getmtime(meta)
                newest = t if newest is None else max(newest, t)
    except OSError:
        return None
    return None if newest is None else (time.time() - newest) / 3600


def counter(name, cond):
    """Consecutive-breach debounce shared by every 2-check rule."""
    n = globals().setdefault(name, 0) + 1 if cond else 0
    globals()[name] = n
    return n


# --------------------------------------------------------------------- the hourly check's lines
def _vs_own(v, ref, line_mult=None):
    """A value as a percentage of this run's own hour-1 level, with its stop line."""
    if v is None:
        return "unmeasured"
    if not _finite(ref) or ref <= 0:
        return "its hour-1 level is measured from this window" if ref is None else "hour-1 level unusable"
    return "%s of its own hour-1 level%s" % (pct_plain(v / ref), "" if line_mult is None
                                              else " (stop line %d%%)" % round(100 * line_mult))


def check_lines(h, w, m, base, run_h, ck_age):
    """The hourly check's gauge lines for a person (DESIGN section 11): the stop rules in main()
    judge the same window means."""
    var = wmean(w, "token_temporal_std", "vision")
    tvar = wmean(w, "token_temporal_std", "tactile")
    tp, tg_ = wmean(w, "tenure_pe", "pose"), wmean(w, "tenure_gap", "pose")
    cv = {s: wmean(w, "causal_pe_per_modality", s) for s in ("vision", "tactile", "proprioception", "audio")}
    sps, vram = wmean(w, "model_steps_per_s"), wmean(w, "peak_vram_mb")
    replay, imag = wmean(w, "replay_loss"), wmean(w, "imagination_loss")
    out = []
    ref_sps = base.get("steps_per_s")
    out.append(
        f"H84 check-{h}  {run_h:.1f} hours into the run  step {m.get('step')}  non-finite updates "
        f"{m.get('nan_steps')}  {show(sps, 2)} model steps a second"
        + (f" ({pct_plain(sps / ref_sps)} of this run's hour-1 rate; stop line 33%)" if _finite(sps) and _finite(ref_sps) and ref_sps > 0 else "")
        + f"  peak allocation {show(vram, 0)} MiB of 8,188 on the card (own hour-1 {show(base.get('peak_vram_mb'), 0)} "
        f"MiB; alert past 500 MiB more)  ({len(w)} records in the window; every gauge is a window mean)")
    out.append(names_line())

    def of_var(v):
        return "unmeasured" if v is None else "%s of the stop line" % pct_plain(v / VARIANCE_LINE)
    out.append(
        f"  token variety: vision {of_var(var)}, tactile {of_var(tvar)} (under 100% on two checks running, "
        f"from run-hour 3, stops the run); causal error vision {_vs_own(cv['vision'], base.get('vision'), 2.0)}, "
        f"tactile {_vs_own(cv['tactile'], base.get('tactile'), 3.0)}, proprioception "
        f"{_vs_own(cv['proprioception'], base.get('proprioception'))}, audio {_vs_own(cv['audio'], base.get('audio'))} "
        f"(lower is better)")
    decoys = wmean(w, "tenure_tok", "n_decoys")
    chance = None if decoys is None else 1.0 / (decoys + 1.0)
    margin = (tg_ / tp) if tp else None
    out.append(
        f"  matching: a command's projection picks that command's own sensory change out of "
        f"{count(None if decoys is None else decoys + 1)} in "
        f"{of100(wmean(w, 'tenure_tok', 'rank1'), 'tries')} (chance {of100(chance)}); the vision gate open on "
        f"{of100(wmean(w, 'tenure_tok', 'vis_gate_open'), 'command endings')}; the 250 ms pose head's margin over "
        f"'nothing moves' {pct_plain(margin, signed=True)} of its error (0% = no better, 100% = exact)")
    prior0 = RUN_ARGS.get("agency_prior")
    prior = wmean(w, "agency", "prior_w")
    prior_txt = (f"the hand-watching bootstrap's weight {pct_plain(prior / float(prior0))} of its starting value"
                 if _finite(prior) and _finite(prior0) and float(prior0) > 0 else
                 f"the hand-watching bootstrap's weight {NOT_A_COUNT}")
    out.append(
        f"  agency: bodies with the hand in view {of100(wmean(w, 'agency', 'vis_adv_elig'), 'model steps')}; whether "
        f"the vision and tactile tokens match their own command's result better than another body's: {NOT_A_COUNT} (the record "
        f"carries cosine advantages); {prior_txt}")
    out.append(
        f"  mobile: hand strikes on {of100(wmean(w, 'task_mobile_hand'), 'model steps')}, any strike on "
        f"{of100(wmean(w, 'task_mobile_hit'))}, the mobile moving {show(wmean(w, 'task_mobile_speed'), 2)} cm a second")
    cq = {k: wmean(w, "consequence", k) for k in ("kinds_mean", "rate", "goal_rate")}
    out.append(
        f"  consequence access: model steps of a body carrying one {of100(cq['rate'], 'model steps')}; kinds per model step "
        f"{count(None if cq['kinds_mean'] is None else 100 * cq['kinds_mean'])} per 100 model steps (at most 300); "
        f"chosen goals carrying one {of100(cq['goal_rate'], 'plans')}")
    lic = m.get("plan_license") or {}
    wins = lic.get("win_emas") or []
    gap = wmean(w, "intention", "gap_closed_pct")
    out.append(
        f"  planner: lookahead depths 1/2/3 beat 'nothing changes' in "
        + ("/".join(str(round(100 * x)) for x in wins if _finite(x)) or "unmeasured")
        + f" of every 100 tries (each needs more than 50, depth 1 before 2 before 3); lookahead depths licensed "
        f"{lic.get('licensed', 'unmeasured')} (0 = the planner stays one command ahead); imagined gap closed "
        f"{'unmeasured at horizon 1' if gap is None else '%.1f%%' % gap} (0% = never moved toward its goal, 100% = "
        f"arrived); plans ending closer than they started {of100(wmean(w, 'intention', 'closer_than_start'), 'plans')}; "
        f"replay lesson {_vs_own(replay, base.get('replay_loss'), 3.0)}; imagination lesson "
        f"{'running' if imag is not None else 'ABSENT'} ({count(sum(1 for r in w if _finite(r.get('imagination_loss'))))} "
        f"of the hour's records carry it; its loss value is not printed)")
    fol = wmean(w, "follow_bout")
    fol = wmean(w, "cmd_response_cos_held") if fol is None else fol
    out.append(
        f"  execution: how closely the body's 250 ms turn follows its command is {NOT_A_COUNT} (the record "
        f"carries a direction match, a cosine); the babble-bout reading "
        + ("is BELOW its stop line" if _finite(fol) and fol < FOLLOW_LINE else
           "is above its stop line" if _finite(fol) else "is unmeasured")
        + " (two checks below it stop the run: a gaze force broke obedience)")
    bbs = wmean(w, "bout_body_speed")
    hf = wmean(w, "task_hand_fov")
    rob = wmean(w, "rest_over_bout_speed")
    sl = [(r.get("hself") or {}).get("slots_live") for r in w]
    sl = [v for v in sl if v is not None]
    out.append(
        f"  window: babble on {of100(wmean(w, 'babble_duty'), 'model steps')}; body speed at rest "
        f"{pct_plain(rob)} of its speed in babble bouts (100% = no rest); bout body speed "
        f"{'unmeasured' if bbs is None else pct_plain(bbs / ACTIVITY_REF)} of the historical activity reference "
        f"(alert under 100%, not a stop); eyes on the hand {of100(hf, 'model steps')} (alert under 3 of every 100 "
        f"from run-hour 6); slots live {'%d-%d' % (min(sl), max(sl)) if sl else 'unmeasured'} "
        f"(a gauge over about 50 s, it oscillates)")
    out.append(
        f"  gaze: ring snaps {count(m.get('task_ring_snaps'))}, anticipation reached the moving target on "
        f"{of100(wmean(w, 'task_antic_ok'), 'model steps')}, squeaker seen on {of100(wmean(w, 'task_squeaker_seen'))}, "
        f"dragonfly darts {count(m.get('task_dragonfly_darts'))}")
    ck_fail = m.get("checkpoint_failures")
    out.append(
        f"  checkpoints: newest complete {'none' if ck_age is None else '%.2f h old' % ck_age} (alert past 2 h "
        f"from run-hour 2)  saves that failed this segment {ck_fail} (0 = every save landed)"
        + (f"  last error: {m.get('checkpoint_last_error')}" if ck_fail else ""))
    return out


def deaf_sense_lines(w):
    """Every body's collapse watch per sense (token_temporal_std_bodies) as hour means, against each
    sense's line (mimo_bridge/deaf_senses.py: the lines and the measurements behind them). Every body
    under its sense's line is named in a DEAF SENSE ALERT, which is never a stop rule (module
    docstring). A digit step of the dot path reads one body's place in the list, so a body missing
    from some records is averaged over the records that carry it."""
    from mimo_bridge import deaf_senses as DS
    senses = wkeys(w, "token_temporal_std_bodies")
    if not senses:
        return ["  per-body collapse watch: unmeasured - no record in the hour carries token_temporal_std_bodies "
                "(the trainer's code predates it); a missing reading is not a deaf body"]
    readings = {}
    for s in senses:
        n = max((len(v) for v in (field(r, "token_temporal_std_bodies", s) for r in w) if isinstance(v, list)),
                default=0)
        readings[s] = [wmean(w, "token_temporal_std_bodies", s, str(i)) for i in range(n)]
    judged = DS.judge(readings)
    runs = {j["sense"]: counter("_deaf_" + j["sense"], bool(j["bodies"])) for j in judged}
    out = ["  per-body collapse watch (hour means of each body's token change over 50 s: %s; bodies numbered "
           "from 1 as on the dashboard): %s" % (DS.SCALE, DS.summary_text(readings) or "unmeasured")]
    alert = DS.alert_text(judged, runs)
    if alert:
        out.append("  " + alert)
    return out


def hself_lines(w, m):
    hs = m.get("hself") or {}
    out = []
    # -1 = unmeasured: no slot has lived the windows a perfect link would need to clear
    # the family-wise line; the record says how many model steps remain until one could
    if not hs:
        out.append("  hself(dark): unmeasured - the newest record carries no hself block (--hierarchical-self "
                   "off, or the run predates it)")
    else:
        out.append(
        f"  hself(dark): filled {hs.get('slots_filled')}/{RUN_ARGS.get('hself_slots', 64)}  "
        f"minutes level: {hs.get('l2_sig')} slots with a wellness link (-1 = unmeasured), windows lived "
        f"{hs.get('l2_ticks')} of the {hs.get('l2_ticks_needed')} needed, {hs.get('l2_steps_to_first_evidence')} steps to go  "
        f"hours level: {hs.get('l3_sig')}, {hs.get('l3_ticks')} of {hs.get('l3_ticks_needed')}, "
        f"{hs.get('l3_steps_to_first_evidence')} steps to go")
    # WHETHER THE TABLE IS ABOUT TODAY'S MOMENTS (audit P238, 2026-09-19). A slot stores one pattern:
    # what all the senses reported at one moment. A stored pattern is refined only by moments that
    # BELONG to its slot (every sense at cosine 0.9 or above); with that share at 0 the table is
    # about other moments and the evidence counts above describe those, not today. The two gauges
    # were first written as moments_with_a_row / best_row_cos (segments 6 and 7); both are read.
    if any(hs.get(k) is not None for k in ("moments_with_a_slot", "moments_with_a_row", "slots_twinned")):
        share = wmean(w, 'hself', 'moments_with_a_slot')
        share = wmean(w, 'hself', 'moments_with_a_row') if share is None else share
        out.append(
            f"  hself(dark): model steps that belong to a slot {of100(share, 'model steps')} (0 = none: no stored "
            f"pattern is being refined and the table is about other model steps; 100 = every model step belongs to a "
            f"slot); how alike a model step and its nearest slot are: {NOT_A_COUNT} (a cosine); slots that are "
            f"copies of another slot {hs.get('slots_twinned')} of {hs.get('slots_filled')}")
    # THE NAMED THINGS (entity self) AND THE READER. A slot of the whole-moment table above is
    # a pattern of tokens; an entity here is a named thing - the caregiver, a toy, the mobile -
    # with its own presence signal read from the world. The reader turns a PROVEN link into
    # value the planner acts on, and adds nothing until its gate opens.
    es = m.get("entity_self") or {}
    if es.get("names"):
        names, pres = es["names"], es.get("presence_mean") or []
        link, rank = es.get("l2_link") or [], es.get("l2_entity_windows") or []
        shown = sorted(range(len(names)), key=lambda i: -abs(link[i] if i < len(link) else 0.0))[:4]
        out.append("  entity self (dark): %d named things, learning from %s | %s windows lived of the %s "
                   "a link needs | counted: %s"
                   % (len(names), es.get("teacher"), es.get("l2_ticks"), es.get("windows_needed"),
                      ", ".join(es.get("l2_counted_names") or []) or "none yet"))
        out.append("    the named things with the strongest links (minutes level; the link itself is a "
                   "correlation, %s): " % NOT_A_COUNT + "; ".join(
                       "%s present on %s, %s windows of its own"
                       % (names[i], of100(pres[i] if i < len(pres) else None, "model steps"),
                          rank[i] if i < len(rank) else "?") for i in shown))
        rd = es.get("reader") or {}
        if rd:
            asked, applied = rd.get("weight") or 0.0, rd.get("applied_weight") or 0.0
            why = rd.get("why") or ""
            if _has_bare_decimal(why):
                why = "its reason is in the record"
            out.append("    the reader: gate %s, weight applied %s of the weight it asks for. %s"
                       % ("OPEN - proven links are changing which goal ring entry the planner aims at"
                          if rd.get("gate_open") else "shut, so nothing is added",
                          pct_plain(applied / asked) if asked else "0%", why))
    return out


def main(argv):
    configure(argv)
    print(f"Monitoring {RUN}; H84 own-baseline stop rules; {HOURS} hourly checks", flush=True)
    if rows():
        time.sleep(60)
    else:
        # A REPAIR RUN'S WATCHER STARTS WITH ITS TRAINER (the relaunch review of Oct 5, 2026, finding F1), before the
        # first full stats record: it looks for that record every 30 s (at most 600 s, as before), so its first
        # 5-minute check comes 300 s after the record instead of 900 s after its start (the re-check's finding N2)
        waited = 0
        while not rows() and waited < 600:
            time.sleep(30)
            waited += 30
    base = load_baselines()
    stopped_quick = False
    for h in range(1, HOURS + 1):
        target = 3600 - (600 if h == 1 else 0)
        slept = 0
        while slept < target:            # the 5-minute quick check between sweeps
            pause = min(300, target - slept)
            time.sleep(pause)
            slept += pause
            why = quick_breach()
            if why:
                stop_run(why)
                stopped_quick = True
                break
        if stopped_quick:
            break
        mi = rows()
        if not mi:
            print(f"H84 check-{h}: NO DATA - run never started?", flush=True)
            continue
        m = mi[-1]
        last_ts = m.get("timestamp", 0)
        w = [r for r in mi if r.get("timestamp", 0) >= last_ts - 3600]
        age_min = (time.time() - last_ts) / 60
        run_h = m.get("elapsed_s", 0) / 3600
        if age_min > 10 and m.get("elapsed_s", 0) / 60 < HOURS * 60 - 15:
            print(f"H84 check-{h}: STATS STALE ({age_min:.0f} min old) - run died or "
                  f"stalled at step {m.get('step')}", flush=True)
            continue
        var = wmean(w, "token_temporal_std", "vision")
        tvar = wmean(w, "token_temporal_std", "tactile")
        cv = {s: wmean(w, "causal_pe_per_modality", s) for s in ("vision", "tactile", "proprioception", "audio")}
        bbs = wmean(w, "bout_body_speed")
        fol_b = wmean(w, "follow_bout")
        fol_raw = wmean(w, "cmd_response_cos_held")
        hf = wmean(w, "task_hand_fov")
        replay = wmean(w, "replay_loss")
        sps = wmean(w, "model_steps_per_s")
        vram = wmean(w, "peak_vram_mb")
        ck_age = newest_checkpoint_age_h()
        ck_fail = m.get("checkpoint_failures")
        lines = _safe(check_lines, h, w, m, base, run_h, ck_age, name="the hourly gauges")
        for line in lines:
            print(line, flush=True)
        if (ck_fail or 0) > 0 or (run_h >= 2 and (ck_age is None or ck_age > 2.0)):
            print(f"  CHECKPOINT ALERT: {'saves are failing' if (ck_fail or 0) > 0 else 'no complete checkpoint in the last 2 h'} - "
                  f"the run continues but a crash now loses everything since the newest complete save; check the "
                  f"storage disk ({RUN_ARGS.get('checkpoint_storage') or 'run/checkpoints'})", flush=True)
        # the self-model, the gain-of-function blocks, liveness, then storage and the machine
        for build, args_, name in ((hself_lines, (w, m), "hself"), (gof_lines, (w,), "gain of function"),
                                   (liveness_lines, (w,), "liveness"), (storage_lines, (), "storage"),
                                   (vestibular_repair_lines, (), "vestibular repair"),
                                   (motor_asked_change_lines, (), "asked-change path"),
                                   (heads_all_slot_path_lines, (), "heads' all-slot paths"),
                                   (vestibular_input_path_lines, (), "vestibular input path"),
                                   (lambda: machine_lines(w, live_machine()), (), "machine")):
            try:
                got = build(*args_)
            except Exception as error:      # an informational line must never cost a stop rule
                got = ["  LINE FAILED (%s): %s: %s" % (name, type(error).__name__, error)]
            for line in got:
                print(line, flush=True)

        # every body's collapse watch (the deaf-sense alert, never a stop) and the proprioception repair's lines
        for build, args_, name in ((deaf_sense_lines, (w,), "per-body collapse watch"),
                                   (proprio_repair_lines, (), "proprioception repair")):
            for line in _safe(build, *args_, name=name):
                print(line, flush=True)
        if not base:
            base = {"vision": cv["vision"], "tactile": cv["tactile"], "proprioception": cv["proprioception"],
                    "audio": cv["audio"], "replay_loss": replay, "steps_per_s": sps,
                    "peak_vram_mb": vram,
                    "measured_at_step": m.get("step"), "window_records": len(w), "run_hour": round(run_h, 2)}
            with open(BASE_PATH, "w", encoding="utf-8") as f:
                json.dump(base, f, indent=2)
            print(f"  OWN BASELINES measured from this window ({len(w)} records, step {m.get('step')}) and written "
                  f"to watcher_baselines.json. Erosion rules are judged from the next check.", flush=True)
        crit_stop = declared_criteria(w, run_h, last_ts - mi[0].get("timestamp", last_ts))
        if crit_stop:
            stop_run(crit_stop)
            break
        if (m.get("nan_steps") or 0) > 0:
            stop_run("non-finite training update recorded")
            break
        # Reduced activity alone cannot identify broken actuation or a learned choice.
        # Keep the old reference visible; do not lower it to fit the new observations.
        n_low = counter("_cm_low", bbs is not None and bbs < ACTIVITY_REF)
        if n_low:
            print(f"  ACTIVITY ALERT: bout body speed {pct_plain(bbs / ACTIVITY_REF)} of the historical "
                  f"reference ({n_low} consecutive checks). Less movement exposure; cause untested.", flush=True)
            if n_low >= 2:
                print("  Training continues. Check command response and matched-world behavior; "
                      "low activity alone proves neither motor failure nor purposeful comfort seeking.", flush=True)
        fol = fol_b if fol_b is not None else fol_raw
        n_fol = counter("_fol_low", fol is not None and fol < FOLLOW_LINE)
        if n_fol:
            print(f"  WARNING: the babble-bout direction match is below its stop line ({n_fol}/2 consecutive)",
                  flush=True)
            if n_fol >= 2:
                stop_run("the babble-bout direction match stayed below its stop line two checks running "
                         "- a gaze force broke obedience")
                break
        n_hf = counter("_hf_miss", run_h >= 6 and hf is not None and hf < HAND_FOV_LINE)
        if n_hf:
            print(f"  DEVELOPMENT ALERT: eyes on the hand {of100(hf, 'model steps')}, under the alert line of 3 "
                  f"at run-hour {run_h:.1f} ({n_hf}/2)", flush=True)
            if n_hf >= 2:
                print("  Hand viewing remains unresolved. Training continues; this world-dependent "
                      "reading is not a valid restart criterion.", flush=True)
        # Variance: debounced (2 consecutive), free snapshot on the first breach, judged
        # from run-hour 3 (a fresh encoder's first two hours are not a collapse gauge).
        # 2026-09-14: a single-check version stopped H81 on tactile 0.053 while the
        # probed encoder read 0.21-0.32 - the collapse was constant INPUT (a settled
        # posture pressing the same contacts), not a broken encoder.
        vlow = run_h >= 3 and ((var is not None and var < VARIANCE_LINE) or (tvar is not None and tvar < VARIANCE_LINE))
        which = " and ".join(
            n for n, v in (("vision variance", var), ("tactile variance", tvar))
            if v is not None and v < VARIANCE_LINE) or "token variance"
        n_var = counter("_var_low", vlow)
        vtxt = (f"vision {'unmeasured' if var is None else pct_plain(var / VARIANCE_LINE)} / tactile "
                f"{'unmeasured' if tvar is None else pct_plain(tvar / VARIANCE_LINE)} of the stop line")
        if n_var:
            print(f"  CRITICAL: {which} below the stop line - {vtxt} ({n_var}/2 consecutive). Flat tokens can "
                  f"be a flat encoder OR flat input; attribution needs a fresh-world probe on the "
                  f"snapshot (explain_probe).", flush=True)
            if n_var == 1:
                with open(os.path.join(RUN, "control.json"), "w") as f:
                    json.dump({"cmd": "snapshot"}, f)
                print("  snapshot requested (free); stop only on a second consecutive breach", flush=True)
            if n_var >= 2:
                stop_run(f"{which} below the stop line two hourly checks running - {vtxt} - probe the "
                         f"snapshot to attribute encoder vs input before resuming")
                break
        judging = base.get("measured_at_step") is not None and m.get("step") != base.get("measured_at_step")
        rules = (
            ("_vis_high", "vision causal error", cv["vision"], base.get("vision"), 2.0, "H55 signature"),
            ("_tac_high", "tactile causal error", cv["tactile"], base.get("tactile"), 3.0, "erosion"),
            ("_rep_high", "replay loss", replay, base.get("replay_loss"), 3.0, "re-study lesson diverging"),
        )
        stopped = False
        for key, name, val, ref, mult, why in rules:
            hit = judging and ref is not None and val is not None and ref > 0 and val > mult * ref
            n = counter(key, hit)
            if n:
                print(f"  WARNING: {name} at {pct_plain(val / ref)} of its own hour-1 baseline, past the "
                      f"{round(100 * mult)}% line ({n}/2 consecutive)", flush=True)
                if n >= 2:
                    stop_run(f"{name} at {pct_plain(val / ref)} of its own hour-1 baseline, past the "
                             f"{round(100 * mult)}% line, two hourly checks running ({why})")
                    stopped = True
                    break
        if stopped:
            break
        ref_vram = base.get("peak_vram_mb")
        if judging and ref_vram and vram is not None and vram > ref_vram + 500:
            print(f"  MEMORY ALERT: peak allocation {vram:.0f} MiB is {vram - ref_vram:.0f} above own hour-1 {ref_vram:.0f} "
                  f"(the 150-step smoke climbed 1.7 MiB per step; the card holds 8,188 and the renderers ~100 per body) - "
                  f"a climb that continues spills off the card and the steps-per-second rule then stops the run", flush=True)
        ref_sps = base.get("steps_per_s")
        n_slow = counter("_sps_low", judging and ref_sps and sps is not None and sps < ref_sps / 3.0)
        if n_slow:
            print(f"  WARNING: model steps per second at {pct_plain(sps / ref_sps)} of own hour-1, under the 33% "
                  f"line ({n_slow}/2 consecutive) - memory spilling off the card or a stuck worker pool", flush=True)
            if n_slow >= 2:
                stop_run(f"model steps per second at {pct_plain(sps / ref_sps)} of own hour-1 baseline, under "
                         f"the 33% line, two hourly checks running (sustained step-time blow-up)")
                break
    print("H84 hourly watcher done", flush=True)


if __name__ == "__main__":
    main(sys.argv)
