"""The stop lines of the asked-change path's switch in production (--motor-asked-change on), judged by the H84 hourly
watcher. Plan and arithmetic: docs/gain_of_function_v4/restart_candidate/RESTART_CANDIDATE_MOTOR_2026-10-03.md;
registry row motor_asked_change_production_segment.

WHAT THE SWITCH IS. Production's command gets --motor-asked-change on at a relaunch. The load then builds the path
beside the motor encoder (lem/model/motor_asked_change.py: per handle the asked position change, the asked turn, the
strength and the radius, through a shared network 10 -> 64 -> 16 and an output layer of 676 -> 192 that starts at zero)
and the trainer writes one motor_asked_change_switch record into the stats stream. From the next model step on the
path's output is added to the motor encoder's output in units of that output's own standard deviation.

HOW IT IS JUDGED. As the vestibular encoder repair's switch is (vestibular_repair_watch.py, whose code this module
runs wherever the arithmetic is the same): the supervisor keeps one watcher per run and knows it only by the file name
hourly_watch.py, so these lines live in that watcher. Its 5-minute check calls breach(RUN) and writes the stop with
its own stop_run (the supervisor pause, <run>/stop_reason.json with resume_policy repair_required, <run>/control.json
snapshot_and_stop); its hourly sweep prints lines(RUN). A watcher started for a continuation reads the whole timeline
from the switch through the run folders before it. A run that no motor_asked_change_switch record reaches is not
judged: breach() returns None and lines() returns nothing for it.

By hand, reading only (nothing is written, not even the state file):
    python docs/gain_of_function_v4/motor_asked_change_watch.py <run folder>

THE BASE MODULE. `BASE` is a private instance of vestibular_repair_watch.py, loaded by path under another module name,
with three of its names replaced on that instance only: SWITCH_TYPE (this switch's record type), STATE_NAME (this
watch's state file) and compact (the fields this watch reads from a stats record). Its timeline (before_reports,
after_records), its 20-report windows, its level before the switch, its two K6 multiples per sense and its
bookkeeping of crossings whose stop was taken are then this module's, unchanged. The hourly watcher's own
vestibular rule is another instance of that file and reads its own names: nothing of the vestibular watch changes.
LINES is that instance's dict with this watch's own keys added, so the multiples below ARE the vestibular watch's
(tests/test_motor_asked_change_watch.py holds them equal to a fresh load of that file).

WHAT THIS MODULE DOES ITSELF, AND WHY (each found by planted cases on Oct 3, 2026; each has its test):
  the walk  find_switch is the vestibular watch's walk with one rule added. That walk watches a later run only when
            it resumed a checkpoint AFTER the switch's model step N. A switched run that consumes a stop before its
            first model step saves the switched state AT N in its own folder, and the run that resumes that
            checkpoint carries the path with no switch record of its own: the lineage holds this case once in 128
            runs (seg40_resume_20260925_030344 saved at 402,150, its launch step, and the next run resumed it). Here
            such a run is watched when the checkpoint it resumed lies in the folder of a run of the chain from the
            switch on. The revert resumes the checkpoint at N in the folder of the switched run's parent and is not
            watched.
  the reader  StatsReader commits a read's offset and its records together, after every line of the read was parsed.
            The vestibular watch's reader moves its offset first, so one exception inside a read (a MemoryError at
            the commit limit) loses every record of that read for good and the watch goes silent.
  the records  compact reads every block through _obj: a block that is not an object (a string, a number, a list)
            reads as empty and never raises.
  the watch's end  watch_records: the watch ends at the first FULL report at or past N + 10,000, the report that
            closes the last window, and every record up to it is judged. Cut at N + 10,100 as the vestibular watch
            cuts, a relaunch that leaves that record partial left the last up to 19 full reports outside every
            window. The 5-minute check that finds the watch over with nothing crossed writes `closed` into the state
            file, and breach() judges nothing after it. Until then a run that began up to CLOSE_SLACK (2,000) model
            steps past the vestibular watch's limit is still watched: when the run that wrote the closing report
            stops before its own watcher's next check, the next run's watcher judges the last window.
  crossings  a stop counts as taken when the run it was written into holds its reason in stop_reason.json (the
            vestibular watch's rule) or in the file a resume step renamed that to, resolved_stop_reason_<time>.json
            (resume_after_disk_stop_2026-09-24.py does). A taken crossing stands only under three conditions. (1) It
            is recorded with the run that wrote the record at its model step (owner), and that record is still that
            run's: after a rollback (a later run resumed a checkpoint before the crossing and ran those model steps
            again) the new records are judged afresh. (2) The stop was written before the next run of the chain
            began: a stop written into a run that had already been continued (a watcher left running on an ended
            run: twice on Oct 3, before the relaunch script ended such watchers) stopped no trainer. (3) For lines
            M2, M3 and M6: once a later run continues the stopped run, only the records after the stopped run's last
            one are judged, since the records it wrote between the crossing and the trainer's exit belong to the
            same stop; while the stopped run is still the newest run, every record after the crossing's model step
            is judged, as the vestibular watch does.

THE TIMELINE. A FULL REPORT is a mimo record whose next_frame_norm.steps is 100 (a run launched without
--next-frame-norm writes no such block: it has no full report, lines M3 to M6 judge nothing on it, and every check
says so once 3 records exist after the switch). BEFORE: the up to 100 full reports
with the largest model steps at or below the switch's model step N, back through the parent chain. AFTER: every
record after N along the runs from the switch to this run, through the first full report at or past N + 10,000.
Report k is the k-th full report after N.

OPTIMIZER UPDATES. One optimizer update is one Adam update of the weights. The trainer makes one per model step from
the live lessons and, with --replay on, one more every --replay-every model steps from replay (H84: every 2), so H84
makes 1.5 optimizer updates per model step. The growth bound below is per optimizer update.

THE STOP LINES (each writes snapshot_and_stop; the arithmetic behind each is in the record's section 5):
  M1  the switch record is as declared: it says exact, every precision it checked (float32, and the run's autocast
      when the run is on the graphics card) reads a largest change of the motor token of exactly 0.0 on bit-identical
      tokens, the output layer's largest absolute weight and bias are 0.0, the check ran on 64 probe rows, and the
      version is this build's. Judged at once.
  M2  a non-finite update after N, from two sources. The stats: nan_steps above 0 in any record (the trainer counts
      there the model steps whose live loss was not finite). The trainer's log: a line that reports a skipped
      optimizer update ("optimizer step skipped": lem_model.py writes it for a non-finite gradient norm of the live
      lesson and for a non-finite replay loss or gradient, skips that optimizer update, and no stats field counts
      it), in <run>/trainer.log of any run from the switch on. A non-finite gradient in the path's own clipping
      group skips the whole update, so this source is where a path that stops all learning is seen. MEASURED with
      nothing switched (motor_asked_change_false_stops.py lineage, read Oct 3, 2026, 1:18 PM PDT): 0 such lines in
      the lineage's 104 trainer logs and 0 records with nan_steps above 0 among its 4,439 stats records.
  M3  the path's hidden layer (motor_asked_change.net.0) under 0.9 live share in 2 full reports running, counted
      over the full reports that carry a reading (a report without one neither counts nor ends the count). The
      reading is the LOWER of the whole-model liveness monitor's (liveness.layers: the layer's calls on the model
      steps the monitor samples, every 7th on H84, about 14 of a report's 100) and the path's own gauge
      (motor_asked_change.liveness: the commanded handles of the newest lesson rows, at most 64 body rows). The
      monitor's sample holds every handle of every row of those calls, commanded or not, so it can read a unit live
      that responds on no commanded handle; the hourly watcher's own
      liveness rule reads the monitor first for this layer, so this line is where the commanded handles' reading
      can stop the run. MEASURED at the switch (motor_asked_change_false_stops.py growth, the 2,000 recorded rows
      of checkpoint 455,800 in 32 blocks of at most 64 rows, three seeds): the gauge's arithmetic reads all 64
      units live in every block; the lowest of the units' largest inputs is 0.74 against the line of -3.
  M4  the other senses' next-model-step error ratio (next_frame_norm.ratio: prediction error over the error of
      predicting no change; 0 = exact, 1 = no better than no change), for vision, tactile, audio and proprioception.
      Early: the same sense above its median over the 20 reports before the switch x W1 in report 2 and in report 3
      (report 1 is printed, never judged). Windows: in every window of 20 full reports from report 2, a sense's
      median above its LEVEL x RISE is an ALERT, printed once and recorded, never a stop; above LEVEL x RISE x 1.2
      it is a STOP at its first crossing. LEVEL is the larger of the sense's median over the 20 and over the up to
      100 full reports before the switch. W1, RISE and 1.2 are the vestibular watch's (W1: vision 4.01, tactile 2.09,
      audio 2.75, proprioception 2.17; RISE x 1.2: 3.95 for vision, 1.72 for the other three). No allowance is added:
      the path's output is exactly 0 at the switch, so no cost was measured or declared. The vestibular watch's line K3
      (each window against the window before) is not carried. Vestibular's own ratio is an ALERT only, at
      3.67 times its level. Its token was dead until model step 455,800 (its ratio before then read 0.000003 to
      0.26 and says nothing), so the lineage's record of it is the 145 full reports from 455,900 to 470,500. In
      them a 20-report median reached 3.06 times the level before it with nothing switched
      (motor_asked_change_false_stops.py, alert_only_sense); the alert sits at that rise x 1.2, the vestibular
      watch's margin. Its level reads only the reports after the revival (the 20 before the switch must all lie
      after it, else the sense is not judged). 145 reports are at most 1 watch of 10,000 model steps, so no stop
      line is put on this sense, and the alert's own false-alert chance is unmeasured.
  M5  the step rate: in the same windows, the median model steps per second under 0.9 of the reference's in two
      windows running (the reference: the 20 reports before the switch for the first window, the window before for
      every later one). This is the vestibular watch's K5.
  M6  the path's output against its growth bound: at a full report, motor_asked_change.output_rms (root mean square of
      the path's output on the newest lesson rows, in token units: 0 = the path adds nothing, 1 = as large as the
      motor encoder's own standardised output) above 0.0027 token units x the optimizer updates since the switch.
      0.0027 = the learning rate 3e-4 x a summed input size of 239 / 26: the most one optimizer update can add on
      one row when Adam moves every weight of the output layer by the learning rate in the direction of its input
      on that row. One set of weights serves every row, so the root mean square over rows is lower. MEASURED on the
      2,000 recorded motor encoder inputs of checkpoint 455,800, three seeds of the shared network
      (motor_asked_change_false_stops.py growth; result docs/gain_of_function_v4/restart_candidate/motor/
      growth_line_result.json, Oct 3, 2026): while every weight has moved at most the learning rate per optimizer
      update from zero, the output's root mean square reaches 59 to 62 of every 100 parts of the line for an
      explicit set of weight signs and cannot pass 91 (the largest eigenvalue's bound). So a crossing says the
      weights moved further than one learning rate per optimizer update on average (Adam's single step can reach
      3.16 learning rates after a jump of the gradient) or the inputs are at least 1.10 times the measured size.
      An output_rms that is not a finite number is a crossing too. The optimizer updates per model step come from
      the switched run's args.json; while that file cannot be read the growth line is not judged and every check
      of a watch that is not over says so (a guessed rate of 1 would stop a healthy run at 67 of every 100 parts
      of the true line).
      An ALERT, printed once and recorded, never a stop: output_rms at or above 1 token unit (half of the token's
      variance before its final normalisation is then the path's).
WHAT IS JUDGED WITHOUT BEFORE: M1, M2, M3, M6 and M5 from its second window need no report before the switch and are
judged whatever the parent chain holds; M4 and M5's first window need the 20 reports before it. breach() prints a loud
line about a missing part in every 5-minute check of a switched run whose watch is not over.

THE PASS LINES (printed by the hourly sweep, never a stop):
  S1  the main route uses the body's own command: for proprioception at the hold's first model step
      (prediction_audit_hold_first), the own-command reading on the size-matched swap, 1 - (summed prediction error
      with the body's own command) / (summed error with another body's command applied as a change from this body's
      pose), pooled over a window's full reports and the bodies without body 2 from the per-body sums, with its
      standard error from 2,000 seeded resamplings of the (report, body) cells (the trial driver's arithmetic,
      motor_asked_change_trial_2026-10-03.py). JUDGED over the full reports of model steps N + 2,001 to N + 3,000: the
      reading is above 3.14 x its standard error AND exceeds the REFERENCE by more than 3.14 x the standard error of
      the difference. The REFERENCE is production's own reading with the path absent: the full reports before the
      switch (up to 100) at model steps above 457,825, the first model step on the code that records the block.
      Read again, as a reading, over every 1,000 model steps through the watch and over N + 9,001 to N + 10,000.
  S2  the token carries the asked change: a CPU read-out on a checkpoint after the switch
      (motor_asked_change_segment.py readout), read here from <deploy run>/motor_asked_change_readout.json when the
      job has written it. It says only that the path's output is not zero.
  Readings beside them, never judged: the hold loss's ratio, the contact gates' open shares, both controllability
  gates on the same graded commands, the planner's tied share, and the first three full reports' step rate, output
  size, feature sizes and live share.
  THE LAUNCH SETTINGS AROUND THE SWITCH (printed by the hourly sweep, never judged): each run's args.json against the
  args.json of the run it continues, the settings that name folders left out. The line names what the switch's own
  relaunch changed besides --motor-asked-change (another training change at the same relaunch cannot be told from
  the path by lines M4 and M5) and every relaunch inside the up to 100 full reports before the switch that changed
  a setting, with the number of those reports that follow it (the early line and the step rate's first window read
  the last 20), and every relaunch after the switch, inside the watch, that changed one. The runs are walked along
  the parent chain, so a run that wrote no full report is read too. A setting the run before does not hold (a
  build that added a flag) is named apart when it stands at an off value (None, off, false, 0, an empty text).

A STOP IS NOT REPEATED ON ITS OWN CROSSING. breach() records each crossing it reports in <deploy run>/
motor_asked_change_watch.json with its own id (M1:switch, M4:reports2-3, one 20-report window of M4 or M5, the model
step at which M2, M3 or M6 crossed, or for the trainer's log the run and its count of lines), with the run the stop
goes into, the run that wrote the record at the crossing's model step, and the time. A recorded crossing whose stop
reached its run is taken, and while it stands (WHAT THIS MODULE DOES ITSELF, crossings) that crossing is not judged
again and M2, M3 and M6 judge only later records. An ALERT is recorded once by its id in the same file; reading by
hand records nothing.
Tests: Source/Python/mimo_bridge/tests/test_motor_asked_change_watch.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_base():
    spec = importlib.util.spec_from_file_location("vestibular_repair_watch_as_motor_asked_change_base",
                                                  HERE / "vestibular_repair_watch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE = _load_base()
REPORT_EVERY = BASE.REPORT_EVERY
SWITCH_TYPE = "motor_asked_change_switch"
STATE_NAME = "motor_asked_change_watch.json"
READOUT_NAME = "motor_asked_change_readout.json"
PATH_VERSION = "motor_asked_change_v1"            # lem/model/motor_asked_change.py ASKED_CHANGE_VERSION
PATH_LAYER = "motor_asked_change.net.0"
OTHER_SENSES = BASE.OTHER_SENSES                  # vision, tactile, audio, proprioception
ALERT_ONLY_SENSES = ("vestibular",)
# STOP LINES READ AS ALERTS FOR ONE RUN: {line: why}. Empty unless a caller sets it for the one run it judges (the
# heads' all-slot paths' watch does, from the hourly watcher, for the attention cap experiment's control run 2 and cap
# arm: the operator's decision of Oct 6, 2026). A line named here that crosses is recorded and printed as an alert
# and is never chosen as the stop, so every other line is judged and can stop the run as before.
ALERT_INSTEAD_OF_STOP = {}
EXCLUDED_BODIES = (2,)                            # the operator's decision of Sep 25 (prediction_audit.EXCLUDED_BODIES)
HOLD_FIRST = "prediction_audit_hold_first"
JUDGED_SENSE = "proprioception"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261003                         # the trial driver's

LINES = BASE.LINES                                # the private instance's dict: W1, RISE, the margins, the windows
LINES.update(
    early_reports=(2, 3),
    M3_liveness=LINES["K4_liveness"], M3_consecutive=LINES["K4_consecutive"],     # the vestibular watch's K4
    M5_share=LINES["K5_share"], M5_consecutive=LINES["K5_consecutive"],           # and its K5
    # M6: the most one optimizer update can add to the path's output, token units (lem/model/motor_asked_change.py
    # output_gain: learning rate 3e-4 x a summed input size of 239, measured on the 2,000 recorded rows of checkpoint
    # 455,800, / 26), and the alert level
    growth_per_optimizer_update=3e-4 * 239.0 / 26.0, learning_rate=3e-4, summed_input_size=239.0,
    output_alert=1.0,
    # vestibular's alert multiple (module note, M4): its largest rise on its own record since its revival x the
    # vestibular watch's stop margin; alert_only_since is the model step of that revival
    alert_only_rise=3.06, alert_only_multiple=round(3.06 * LINES["stop_margin"], 2), alert_only_since=455800,
    # S1: the judged window after the switch, the line's z (the own-command read-outs' and the trial's), the first
    # model step whose code records the hold-first block with the size-matched arm (run ..._seg41f_0322)
    S1_window=(2000, 3000), S1_late_window=(9000, 10000), S1_z=3.14, reference_after_step=457825,
    S1_goal_reference=0.08)                       # H71's pooled reading on the absolute swap (the trial's ladder)
FREE_WINDOW = 10                                  # the free readings read the newest up to 10 full reports


_num, _median, _pct, _read_json = BASE._num, BASE._median, BASE._pct, BASE._read_json
is_full = BASE.is_full


# ------------------------------------------------------------------ the records
def _obj(v):
    """A stats block as a dict: {} for a missing block and for one that is not an object."""
    return v if isinstance(v, dict) else {}


def swap_cells(record, block=HOLD_FIRST, sense=JUDGED_SENSE, arm="size_matched"):
    """The (report, body) cells of one swap-test block for one sense: per body not left out, [transitions, summed
    own-command error, summed error with the control command]. The stats block holds per-body means; a cell's sums
    are mean x transitions. (The trial driver's swap_cells; a test holds the two equal.)"""
    per = _obj(_obj(_obj(_obj(record.get(block)).get("senses")).get(sense)).get("per_body"))
    n, own, ctl = per.get(arm + "_transitions"), per.get(arm + "_own_prediction_mse"), per.get(arm + "_action_mse")
    if not (isinstance(n, list) and isinstance(own, list) and isinstance(ctl, list)):
        return []
    out = []
    for b in range(min(len(n), len(own), len(ctl))):
        if b in EXCLUDED_BODIES or not _num(n[b]) or not n[b] or not _num(own[b]) or not _num(ctl[b]):
            continue
        out.append([int(n[b]), float(own[b]) * int(n[b]), float(ctl[b]) * int(n[b])])
    return out


def own_command_reading(cells, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """1 - summed own-command error / summed control error over the cells, with its standard error from `draws`
    seeded resamplings of the cells; None without cells or with a control error of 0. (The trial driver's
    own_command_reading; a test holds the two equal.)"""
    if not cells:
        return None
    own, ctl = sum(c[1] for c in cells), sum(c[2] for c in cells)
    if not ctl > 0:
        return None
    rng = random.Random(seed)
    k = len(cells)
    vals = []
    for _ in range(int(draws)):
        o = c = 0.0
        for _i in range(k):
            x = cells[rng.randrange(k)]
            o += x[1]
            c += x[2]
        if c > 0:
            vals.append(1.0 - o / c)
    se = None
    if len(vals) > 1:
        m = sum(vals) / len(vals)
        se = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return dict(value=1.0 - own / ctl, se=se, transitions=sum(c[0] for c in cells), cells=k)


def hold_loss_ratio(record):
    """tenure_pe.pose / (tenure_pe.pose + tenure_gap.pose): 1 = the command adds nothing to the pose prediction
    over the hold (the hinge's line is 0.96). (The trial driver's hold_loss_ratio.)"""
    pe, gap = _obj(record.get("tenure_pe")).get("pose"), _obj(record.get("tenure_gap")).get("pose")
    if not (_num(pe) and _num(gap)) or not (pe + gap) > 0:
        return None
    return pe / (pe + gap)


def path_live_readings(record):
    """(the whole-model monitor's reading, the path's own gauge's reading) of the hidden layer's live share in one
    report, each None when the report carries none."""
    a = _obj(_obj(record.get("liveness")).get("layers")).get(PATH_LAYER)
    b = _obj(_obj(record.get("motor_asked_change")).get("liveness")).get("net.0")
    return (float(a) if _num(a) else None), (float(b) if _num(b) else None)


def path_live(record):
    """The whole-model monitor's reading, else the path's own. (The trial driver's path_live; a test holds the two
    equal. Line M3 reads path_live_lowest.)"""
    a, b = path_live_readings(record)
    return a if a is not None else b


def path_live_lowest(record):
    """The lower of the two readings a report carries (module note, M3); None when it carries neither."""
    got = [v for v in path_live_readings(record) if v is not None]
    return min(got) if got else None


def compact(r):
    """The fields this watch's lines read, from a stats record (None for a record no line reads)."""
    t = r.get("type")
    if t == SWITCH_TYPE:
        return {"type": t, "step": r.get("step"), "timestamp": r.get("timestamp"), "switch": r.get("switch")}
    if t != "mimo":
        return None
    nf = _obj(r.get("next_frame_norm"))
    mac = r.get("motor_asked_change") if isinstance(r.get("motor_asked_change"), dict) else None
    cc = _obj(r.get("contact_control"))
    ct = _obj(r.get("controllability"))
    ch = _obj(r.get("causal_chain"))
    mach = _obj(r.get("machine"))
    spread, band = ch.get("candidate_spread_mean"), ch.get("tie_band")
    path = None
    if mac is not None:
        path = {"rows": mac.get("rows"), "output_rms": mac.get("output_rms"),
                "output_weight_rms": mac.get("output_weight_rms"), "encoder_output_sd": mac.get("encoder_output_sd"),
                "reach_spread": _obj(mac.get("handle_reach")).get("spread"),
                "features_rms": _obj(mac.get("features_rms")) or None}
    return {"type": "mimo", "step": r.get("step"), "timestamp": r.get("timestamp"),
            "nf_steps": nf.get("steps"),
            "ratio": {s: _obj(nf.get("ratio")).get(s) for s in OTHER_SENSES + ALERT_ONLY_SENSES},
            "nan_steps": r.get("nan_steps"), "sps": r.get("model_steps_per_s"),
            "path": path, "path_live": path_live(r), "path_live_low": path_live_lowest(r), "s1": swap_cells(r),
            "hold_ratio": hold_loss_ratio(r),
            "contact": {k: cc.get(k) for k in ("gate_self_open_share", "gate_world_open_share")},
            "gates": {k: ct.get(k) for k in ("gate_commands", "gate_open_commands", "shadow_gate_commands",
                                             "shadow_gate_open_commands", "gates_same_commands",
                                             "gate_only_open_commands", "shadow_gate_only_open_commands")},
            "tied": ch.get("chosen_tied_share"),
            "spread_over_band": (spread / band) if (_num(spread) and _num(band) and band > 0) else None,
            "gpu_used_mib": mach.get("gpu_used_mib"), "gpu_total_mib": mach.get("gpu_total_mib")}


BASE.SWITCH_TYPE = SWITCH_TYPE
BASE.STATE_NAME = STATE_NAME
BASE.compact = compact


SKIPPED_MARK = b"optimizer step skipped"       # lem_model.py's log lines for an optimizer update it skipped


class StatsReader(BASE.StatsReader):
    """The vestibular watch's reader with one change (module note, the reader): a read's offset and its records are
    committed together, after every line of the read was parsed. An exception inside a read leaves the reader as it
    was, and the next check reads the same bytes again. It also counts, per run, the trainer.log lines that report
    a skipped optimizer update (module note, M2), read the same way."""

    def skipped_updates(self, run):
        """(how many lines of <run>/trainer.log report a skipped optimizer update, the first such line): read from
        where the last read ended; a partly written last line is left for the next read; no file reads as 0."""
        if not hasattr(self, "log_offsets"):
            self.log_offsets, self.log_counts, self.log_first = {}, {}, {}
        path = Path(run) / "trainer.log"
        key = str(path)
        try:
            with open(path, "rb") as fh:
                fh.seek(self.log_offsets.get(key, 0))
                data = fh.read()
        except OSError:
            return self.log_counts.get(key, 0), self.log_first.get(key)
        end = data.rfind(b"\n")
        if end >= 0:
            found = [ln for ln in data[:end].split(b"\n") if SKIPPED_MARK in ln]
            if found and key not in self.log_first:
                self.log_first[key] = found[0].decode("utf-8", "replace").strip()[-160:]
            self.log_counts[key] = self.log_counts.get(key, 0) + len(found)
            self.log_offsets[key] = self.log_offsets.get(key, 0) + end + 1
        return self.log_counts.get(key, 0), self.log_first.get(key)

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
        new = []
        for line in data[:end].decode("utf-8", "replace").splitlines():
            if '"mimo"' not in line and SWITCH_TYPE not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            c = compact(r) if isinstance(r, dict) else None
            if c is not None:
                new.append(c)
        recs.extend(new)
        self.offsets[key] = self.offsets.get(key, 0) + end + 1
        return recs


READER = StatsReader()          # the hourly watcher's process keeps it between checks
load_state = BASE.load_state
record_crossing, record_alerts = BASE.record_crossing, BASE.record_alerts
level_before, windows, k6_levels = BASE.level_before, BASE.windows, BASE.k6_levels
CLOSE_SLACK = 2000              # model steps past the vestibular watch's limit within which a run is still watched


def taken_records(deploy_run, write=True):
    """The recorded crossings whose stop reached their run (module note, crossings): the run's stop_reason.json, or
    a resolved_stop_reason_<time>.json beside it, holds the crossing's reason. With `write`, the ones found are
    marked taken in the state file for good. A crossing whose stop never arrived is left out (judged again)."""
    state = load_state(deploy_run)
    out, changed = [], False
    for c in state.get("crossings") or []:
        ok = bool(c.get("taken"))
        if not ok:
            folder = Path(str(c.get("run")))
            for f in [folder / "stop_reason.json"] + sorted(folder.glob("resolved_stop_reason_*.json")):
                got = _read_json(f, {}) or {}
                if isinstance(got, dict) and got.get("reason") == c.get("reason"):
                    ok = True
                    break
            if ok and write:
                c["taken"] = time.time()
                changed = True
        if ok and c.get("id"):
            out.append(c)
    if changed:
        BASE._write_state(deploy_run, state)
    return out


def taken_crossings(deploy_run, write=True):
    """{crossing id: model step} of the crossings whose stop reached their run (taken_records)."""
    return {str(c["id"]): c.get("step") for c in taken_records(deploy_run, write=write)}


def still_taken(records, after, chain=()):
    """{crossing id: the model step after which its line judges again} for the taken crossings (`records`:
    taken_records) that stand on the timeline as it is now (module note, crossings); `after`: the watch's records;
    `chain`: the runs from the deploy to the judged run."""
    names = [Path(r).name for r in chain]
    began = {}
    for r in chain:
        receipt = _read_json(Path(r) / "continuation_receipt.json")
        began[Path(r).name] = receipt.get("timestamp") if isinstance(receipt, dict) else None
    on_timeline = {r["step"]: r.get("run") for r in after}
    out = {}
    for c in records:
        cid, st = str(c["id"]), c.get("step")
        owner = c.get("owner")
        if owner is not None and on_timeline.get(st) != owner:
            continue                              # (1) a rollback ran those model steps again
        stopped = Path(str(c.get("run") or "")).name
        if stopped in names and stopped != names[-1]:
            later = began.get(names[names.index(stopped) + 1])
            if _num(c.get("time")) and _num(later) and c["time"] > later:
                continue                          # (2) written after the continuation had begun
            if cid[:3] in ("M2@", "M3@", "M6@") and _num(st):
                last = max((r["step"] for r in after if r.get("run") == stopped), default=None)
                if _num(last):
                    st = max(st, last)            # (3) the stopped run's later records belong to the same stop
        if cid not in out or (_num(st) and _num(out[cid]) and st > out[cid]):
            out[cid] = st
    return out


def _saved_by(switched_runs, run):
    """Whether the checkpoint `run` resumed lies in the folder of one of `switched_runs` (compared by folder name: a
    resume path can name the archive copy of a run folder)."""
    ck = BASE.parent_of(run)[1]
    if not ck:
        return False
    return Path(str(ck)).parent.parent.name in {Path(r).name for r in switched_runs}


def find_switch(run, reader=None):
    """(the deploy run, its switch record, the runs from the deploy to `run` in order) when the switch reaches `run`,
    else None: the vestibular watch's walk (BASE.find_switch) with one rule added (module note, the walk). A later
    run is on the switched side when it resumed a checkpoint after the switch's model step N, or a checkpoint AT N
    that a run of the chain from the deploy on saved. A run that resumed before N, or at N from a folder before the
    deploy (the revert), is not watched, and neither is any run after it."""
    reader = reader or READER
    run = Path(run)
    experiments = run.parent
    s0 = BASE.start_of(run)
    floor = (s0 - LINES["horizon_steps"] - REPORT_EVERY - CLOSE_SLACK) if _num(s0) else None
    back, cur, seen = [], run, set()
    while cur is not None and len(back) < BASE.WALK_LIMIT:
        key = str(cur.resolve())
        if key in seen:
            return None
        seen.add(key)
        start = BASE.start_of(cur)
        if back and floor is not None and _num(start) and start < floor:
            return None
        back.append(cur)
        sw = BASE.switch_record(reader.read(cur))
        if sw is not None:
            N = int(sw["step"])
            chain = list(reversed(back))                      # the deploy first
            for i in range(1, len(chain)):
                start = BASE.start_of(chain[i])
                if not _num(start) or start < N:
                    return None
                if start == N and not _saved_by(chain[:i], chain[i]):
                    return None                               # the revert
            return cur, sw, chain
        parent, _ = BASE.parent_of(cur)
        cur = BASE.run_folder(parent, experiments) if parent else None
    return None


def watch_records(after, N):
    """(the records the watch judges, whether the watch is over), from every mimo record after N in model step
    sequence (module note, the watch's end): the records through the first FULL report at or past N + 10,000; while
    no such report exists the watch is not over and every record so far is judged."""
    end = int(N) + LINES["horizon_steps"]
    close = next((r["step"] for r in after if is_full(r) and r["step"] >= end), None)
    if close is None:
        return list(after), False
    return [r for r in after if r["step"] <= close], True


PATH_SETTINGS = ("out", "resume", "checkpoint_storage")     # the launch settings that name folders
OFF_VALUES = (None, "off", False, 0, 0.0, "", "none")       # a new setting at one of these is named apart


def _is_off(value):
    return any(value is v or (type(value) is type(v) and value == v) for v in OFF_VALUES)


def launch_changes(run, experiments=None):
    """What `run`'s launch changed against the run it continues, from the two args.json files, the settings that
    name folders left out: {"changed": {setting: [before, now]}, "new": {setting: value}} (new: the run before holds
    no such setting, as after a build that added a flag); None when either file or the parent cannot be read."""
    parent, _ck = BASE.parent_of(run)
    folder = BASE.run_folder(parent, experiments or Path(run).parent) if parent else None
    now = _read_json(Path(run) / "args.json")
    was = _read_json(Path(folder) / "args.json") if folder else None
    if not (isinstance(now, dict) and now and isinstance(was, dict) and was):
        return None
    keys = [k for k in sorted(set(now) | set(was)) if k not in PATH_SETTINGS]
    return dict(changed={k: [was.get(k), now.get(k)] for k in keys if k in was and was.get(k) != now.get(k)},
                new={k: now.get(k) for k in keys if k not in was and not _is_off(now.get(k))},
                new_at_an_off_value=sorted(k for k in keys if k not in was and _is_off(now.get(k))))


def relaunches_back(run, rows):
    """The runs on the timeline back from `run` (itself first, then the runs it continues), newest first, as far as
    they began inside the span of `rows` (full reports in model step sequence, the first one's own 100 model steps
    included): per run the model step it began at, what its launch changed (launch_changes; None = a file could not
    be read) and how many of `rows` lie after that model step. Every run on the chain is listed, also one that
    wrote no full report. The walk ends at the first run that began before the span."""
    out, seen = [], set()
    if run is None or not rows:
        return out
    cur, experiments = Path(run), Path(run).parent
    lowest = rows[0]["step"] - REPORT_EVERY
    while cur is not None and len(seen) < BASE.WALK_LIMIT:
        key = str(cur.resolve())
        if key in seen:
            break
        seen.add(key)
        start = BASE.start_of(cur)
        if not _num(start) or start < lowest:
            break
        out.append(dict(run=cur.name, began_at=start, launch=launch_changes(cur, experiments),
                        full_reports_after=sum(1 for r in rows if r["step"] > start)))
        parent, _ = BASE.parent_of(cur)
        cur = BASE.run_folder(parent, experiments) if parent else None
    return out


def _changed_something(x):
    return bool(x["launch"] and (x["launch"]["changed"] or x["launch"]["new"]))


def launch_settings(deploy, before, chain=()):
    """THE LAUNCH SETTINGS AROUND THE SWITCH (module note): the switch's own relaunch, the relaunches inside the
    reports before it, and the relaunches after it along `chain` (the runs from the deploy to the judged run)."""
    own = launch_changes(deploy)
    parent, _ = BASE.parent_of(deploy)
    folder = BASE.run_folder(parent, Path(deploy).parent) if parent else None
    walked = relaunches_back(folder, before)
    others = [x for x in walked if _changed_something(x)]
    later = [dict(run=Path(r).name, began_at=BASE.start_of(r), launch=launch_changes(r)) for r in list(chain)[1:]]
    besides = None
    if own is not None:
        besides = sorted(k for k in list(own["changed"]) + list(own["new"]) if k != "motor_asked_change")
    return dict(switch=own, besides_the_flag=besides, before=others,
                unread=[x for x in walked if x["launch"] is None],
                within_the_20_before=[x for x in others if x["full_reports_after"] < LINES["before_reports"]],
                after=[x for x in later if _changed_something(x)],
                after_unread=[x for x in later if x["launch"] is None])


def optimizer_updates_per_model_step(args):
    """From a run's args.json: 1 from the live lessons, plus 1 / --replay-every when replay is on (module note)."""
    args = args if isinstance(args, dict) else {}
    every = args.get("replay_every")
    if str(args.get("replay") or "off") == "on" and _num(every) and every > 0:
        return 1.0 + 1.0 / float(every)
    return 1.0


def growth_line(model_steps, per_step):
    """M6's line at `model_steps` after the switch, token units; None while the optimizer updates per model step
    are unknown."""
    if not _num(per_step):
        return None
    return LINES["growth_per_optimizer_update"] * float(per_step) * float(model_steps)


def check_switch(sw, on_card=False):
    """None when the switch record is as declared (module note, M1), else why not (in words)."""
    sw = sw if isinstance(sw, dict) else {}
    checks = {k: v for k, v in sw.items() if isinstance(v, dict) and "change_max_abs" in v}
    if sw.get("exact") is not True:
        return "the switch record does not say that the motor token with the path equals the token without it"
    if not checks:
        return "the switch record holds no check of the motor token"
    bad = sorted(k for k, v in checks.items() if not (v.get("change_max_abs") == 0.0 and v.get("bit_identical") is True))
    if bad:
        return ("the motor token with the path differs from the token without it on the probe rows in %s"
                % ", ".join(bad))
    if on_card and "autocast" not in checks:
        return "the token check did not run under the run's autocast, the precision the graphics card computes in"
    if sw.get("output_weight_max_abs") != 0.0 or sw.get("output_bias_max_abs", 0.0) != 0.0:
        return "the path's output layer was not zero at the load"
    if sw.get("rows") != 64:
        return "the token check ran on %s probe rows, not 64" % sw.get("rows")
    if sw.get("version") != PATH_VERSION:
        return "the switch record names path version %s, not %s" % (sw.get("version"), PATH_VERSION)
    return None


# ------------------------------------------------------------------ the judgement
def judge(before, after, switch, N, taken=None, per_step=1.0, on_card=False, readings=True, skipped=(),
          newest_run=None):
    """(stop reason or None, the crossing's id or None, report). `before`: the full reports before the switch in
    model step sequence (up to 100); `after`: every mimo record after N; `taken`: {crossing id: model step} of
    crossings whose stop was taken; per_step: optimizer updates per model step (None: the run's args.json could not
    be read); readings: also the pass lines and the free readings (the hourly sweep; the 5-minute check judges the
    stop lines alone); skipped: per run from the switch on (its folder name, the lines of its trainer.log that
    report a skipped optimizer update, the first such line); newest_run: the judged run's folder name."""
    taken = dict(taken or {})
    N = int(N)
    full = [r for r in after if is_full(r)]
    rep = dict(N=N, before_reports=len(before), after_full_reports=len(full), lines={}, alerts=[], readings={},
               acceptance={}, newest_step=(after[-1]["step"] if after else None), warnings=[],
               optimizer_updates_per_model_step=per_step)
    have_before = len(before) >= LINES["before_reports"]
    base, level, level_n = level_before(before)
    recent = before[-LINES["before_reports"]:] if have_before else []
    sps0 = _median(r.get("sps") for r in recent) if have_before else None
    rep["before"] = dict(median_ratio=base, level=level, level_reports=level_n, median_steps_per_s=sps0,
                         steps=[recent[0]["step"], recent[-1]["step"]] if recent else None)
    if not have_before:
        rep["warnings"].append(
            "only %d of the 20 full reports before the switch were found along the parent chain: the other senses' "
            "lines and the first window's step rate cannot be judged; the lines that need no report before the "
            "switch (the switch record, non-finite updates, the path's hidden layer, the path's output, the later "
            "windows' step rate) are judged on the records after it" % len(before))
    if len(after) >= 3 and not full:
        rep["warnings"].append("%d stats records after the switch and none is a full report (next_frame_norm.steps "
                               "is 100 in none of them): the path's hidden layer, its output, the other senses and "
                               "the step rate are not judged; the switch record and non-finite updates are"
                               % len(after))
    if full and not any(isinstance(r.get("path"), dict) for r in full):
        rep["warnings"].append("no full report after the switch carries the motor_asked_change block: the path's "
                               "output cannot be judged")
    if not _num(per_step):
        rep["warnings"].append("the switched run's args.json could not be read: the optimizer updates per model step "
                               "are unknown, so the path's output is not judged against its growth line in this "
                               "check (an output that is not a finite number still stops the run), and the switch "
                               "record's check under autocast cannot be asked for")
    if have_before:
        flat = [s for s in OTHER_SENSES if not (_num(base.get(s)) and base[s] > 0 and _num(level.get(s))
                                                 and level[s] > 0)]
        if flat:
            rep["warnings"].append("%s: the reports before the switch hold no next-model-step error ratio above 0 "
                                   "for it, so its lines cannot be judged" % ", ".join(flat))
    crossings = []                            # (rank, id, line, why, model step)

    def cross(rank, cid, line, why, step, **kw):
        was = cid in taken
        rep["lines"].setdefault(line, dict(state="held"))
        if line in ALERT_INSTEAD_OF_STOP:
            # an alert for this run (ALERT_INSTEAD_OF_STOP): recorded once by its id, never a candidate for the stop.
            # The id names the judged run: the deploy run's state file holds the alerts of every run under it, so two
            # runs crossing at one model step each record and print their own (the check of the M3 alert, branch
            # review/m3-alert-end-step-2026-10-06, commit d795e90b7, its second later finding)
            rep["lines"][line].update(kw, state="crossed, an alert for this run", why=why, crossing=cid,
                                      alert_because=ALERT_INSTEAD_OF_STOP[line])
            rep["alerts"].append(dict(id="%s-alert-for-this-run:%s%s" % (line, "%s:" % newest_run if newest_run else "",
                                                                          cid), line=line, step=step,
                                      why="asked-change path ALERT for this run, not a stop (%s): %s"
                                          % (ALERT_INSTEAD_OF_STOP[line], why)))
            return
        rep["lines"][line].update(kw, state="crossed, its stop already taken" if was else "crossed", why=why,
                                  crossing=cid)
        if not was:
            crossings.append((rank, cid, line, "asked-change path: " + why, step))

    def after_taken(line):
        steps = [st for cid, st in taken.items() if cid.startswith(line + "@") and _num(st)]
        return max(steps) if steps else None

    # M1: the switch record, at once
    bad = check_switch((switch or {}).get("switch"), on_card=on_card)
    if bad:
        cross(0, "M1:switch", "M1", "the switch at model step %d is not as declared: %s" % (N, bad), N)
    else:
        rep["lines"]["M1"] = dict(state="held", switch="as declared")
    # M2: a non-finite update in any record after N (after a taken M2's model step)
    m2_after = after_taken("M2")
    nan = next((r for r in after if (m2_after is None or r["step"] > m2_after)
                and _num(r.get("nan_steps")) and r["nan_steps"] > 0), None)
    if nan is not None:
        cross(1, "M2@%d" % nan["step"], "M2", "a non-finite training update is in the record of model step %d"
              % nan["step"], nan["step"])
    else:
        rep["lines"]["M2"] = dict(state="held" if after else "waiting", records=len(after))
    # M2, the trainer's log: an optimizer update the learner skipped (module note). A run that a taken stop ended is
    # covered whole once a later run continues it (the lines it wrote before the trainer exited belong to that stop)
    rep["lines"]["M2"]["skipped_updates_in_the_logs"] = sum(int(c) for _n, c, _f in skipped or ())
    for name, count, first in skipped or ():
        if not count:
            continue
        if name != newest_run and any(cid.startswith("M2:log:%s#" % name) for cid in taken):
            continue
        cross(1.5, "M2:log:%s#%d" % (name, count), "M2",
              "the trainer's log of run %s holds %d line(s) that report an optimizer update skipped for a "
              "non-finite gradient or replay loss; the first one ends: %s" % (name, count, first), None)
    # M3: the path's hidden layer under its live share in consecutive full reports that carry a reading (the lower of
    # the monitor's and the path's own gauge's; a report without a reading neither counts nor ends the count)
    m3_after = after_taken("M3")
    cont = [r for r in full if m3_after is None or r["step"] > m3_after]
    run_low, m3, lowest, read = 0, None, None, 0
    for r in cont:
        v = r.get("path_live_low")
        if not _num(v):
            continue
        read += 1
        lowest = v if lowest is None else min(lowest, v)
        run_low = run_low + 1 if v < LINES["M3_liveness"] else 0
        if run_low >= LINES["M3_consecutive"] and m3 is None:
            m3 = (r["step"], v)
    if m3 is not None:
        cross(2, "M3@%d" % m3[0], "M3", "the path's hidden layer had %s of its units responding (the lower of the "
              "whole-model monitor's reading and the path's own reading on the commanded rows), under the line of "
              "%s, in %d full reports with a reading running, up to model step %d"
              % (_pct(m3[1]), _pct(LINES["M3_liveness"]), LINES["M3_consecutive"], m3[0]), m3[0])
    else:
        rep["lines"]["M3"] = dict(state="held" if read else "waiting", reports=read, lowest_live_share=lowest,
                                  reports_without_a_reading=len(cont) - read)
    if cont and not read:
        rep["warnings"].append("no full report after the switch carries the live share of the path's hidden layer "
                               "(liveness.layers or motor_asked_change.liveness): that line cannot be judged")
    elif len(cont) >= 3 and not any(_num(r.get("path_live_low")) for r in cont[-3:]):
        rep["warnings"].append("the newest 3 full reports carry no live share of the path's hidden layer: that line "
                               "is not judged on them")
    # M6: the path's output against its growth line, at every full report (after a taken M6's model step)
    m6_after = after_taken("M6")
    m6, newest_out, top_share, unread6 = None, None, None, 0
    for r in full:
        out = (r.get("path") or {}).get("output_rms")
        if isinstance(out, float) and not math.isfinite(out):
            if m6 is None and (m6_after is None or r["step"] > m6_after):
                m6 = (r["step"], None, None)
            continue
        if not _num(out):
            rows_read = (r.get("path") or {}).get("rows")
            unread6 += 1 if (_num(rows_read) and rows_read > 0) else 0
            continue
        line = growth_line(r["step"] - N, per_step)
        share = out / line if _num(line) and line > 0 else None
        newest_out = dict(step=r["step"], output_rms=out, line=line, share_of_line=share,
                          output_weight_rms=(r.get("path") or {}).get("output_weight_rms"))
        if _num(share):
            top_share = share if top_share is None else max(top_share, share)
        if m6 is None and _num(share) and share > 1.0 and (m6_after is None or r["step"] > m6_after):
            m6 = (r["step"], out, line)
        if out >= LINES["output_alert"] and not any(a["id"] == "M6-alert:unit" for a in rep["alerts"]):
            rep["alerts"].append(dict(
                id="M6-alert:unit", line="M6", step=r["step"],
                why=("asked-change path ALERT (printed and recorded, never a stop): at model step %d the path's "
                     "output reads %s of one token unit; at 100%% the path's output is as large as the motor "
                     "encoder's own standardised output, and half of the token's variance before its final "
                     "normalisation is the path's" % (r["step"], _pct(out)))))
    if m6 is not None and m6[1] is None:
        cross(2.5, "M6@%d" % m6[0], "M6", "at model step %d the path's output is not a finite number" % m6[0], m6[0],
              newest=newest_out)
    elif m6 is not None:
        cross(2.5, "M6@%d" % m6[0], "M6", "at model step %d the path's output reads %s of one token unit, above its "
              "growth line of %s (the most Adam can add in %s optimizer updates since the switch when it moves "
              "every weight of the output layer by the learning rate in the direction of its input, on inputs of "
              "the measured size)" % (m6[0], _pct(m6[1]), _pct(m6[2]), "{:,.0f}".format((m6[0] - N) * per_step)),
              m6[0], newest=newest_out)
    else:
        rep["lines"]["M6"] = dict(state="held" if (newest_out and _num(per_step)) else "waiting", newest=newest_out,
                                  largest_share_of_line=top_share, reports_with_rows_and_no_output=unread6)
        if not _num(per_step):
            rep["lines"]["M6"]["why"] = "the optimizer updates per model step are unknown (args.json unreadable)"
    if unread6:
        rep["warnings"].append("%d full report(s) after the switch carry the path's block with rows and no output "
                               "size: the growth line is not judged on them" % unread6)
    with_block = [r for r in full if isinstance(r.get("path"), dict)]
    if len(with_block) >= 3 and not any(isinstance(r["path"].get("output_rms"), (int, float))
                                        and not isinstance(r["path"].get("output_rms"), bool)
                                        for r in with_block[-3:]):
        rep["warnings"].append("the newest 3 full reports carry the path's block and no output size (rows 0: no "
                               "lesson call with a command in the report's 100 model steps): the growth line is not "
                               "judged on them")
    # M4, early: the same sense above its 20-report median x W1 in reports 2 and 3
    ka, kb = LINES["early_reports"]
    if len(full) >= kb and have_before:
        ra, rb = full[ka - 1], full[kb - 1]
        shares, over = {}, []
        for s in OTHER_SENSES:
            va, vb = ra["ratio"].get(s), rb["ratio"].get(s)
            if not (_num(base.get(s)) and base[s] > 0 and _num(va) and _num(vb)):
                continue
            line = base[s] * LINES["W1"][s]
            shares[s] = min(va / line, vb / line)
            if va > line and vb > line:
                over.append("%s's next-model-step error ratio reached %s and %s of its line"
                            % (s, _pct(va / line), _pct(vb / line)))
        if over:
            cross(3, "M4:reports2-3", "M4_early", "in reports %d and %d after the switch (model steps %d and %d), %s "
                  "(the line: the sense's median over the 20 reports before the switch, times the wander of a single "
                  "report from the 20 before it that 1 report in 1,000 exceeded on the lineage's record)"
                  % (ka, kb, ra["step"], rb["step"], "; ".join(over)), rb["step"], shares=shares)
        else:
            rep["lines"]["M4_early"] = dict(state="held", shares=shares, reports=[ra["step"], rb["step"]])
    else:
        rep["lines"]["M4_early"] = dict(state="waiting", why=(
            "%d of the 3 full reports after the switch" % len(full) if len(full) < kb else
            "the reports before the switch are missing"))
    if full:
        r1 = full[0]
        rep["readings"]["report1"] = dict(step=r1["step"], shares_of_median={
            s: (r1["ratio"].get(s) / base[s] if _num(base.get(s)) and base[s] > 0 and _num(r1["ratio"].get(s))
                else None) for s in OTHER_SENSES})
    # M4 and M5 in every 20-report window through the watch
    wins = windows(full, N)
    newest = {"M4": None, "M5": None}
    prev, prev_slow = None, False
    crossed_windows = {"M4": [], "M5": []}
    level_only = {}
    if have_before and recent[0]["step"] > LINES["alert_only_since"]:
        # the alert-only sense's level reads only the reports after its revival: before it the ratio says nothing
        longer = [r for r in before[-LINES["level_reports"]:] if r["step"] > LINES["alert_only_since"]]
        for s in ALERT_ONLY_SENSES:
            a, b = _median(r["ratio"].get(s) for r in recent), _median(r["ratio"].get(s) for r in longer)
            level_only[s] = max(a, b) if _num(a) and _num(b) else a
    for i, (wid, win) in enumerate(wins):
        first_k = full.index(win[0]) + 1
        span = "the 20 full reports %d-%d after the switch (model steps %d to %d)" % (
            first_k, first_k + len(win) - 1, win[0]["step"], win[-1]["step"])
        med = {s: _median(r["ratio"].get(s) for r in win) for s in OTHER_SENSES + ALERT_ONLY_SENSES}
        if have_before:
            shares, alert_shares, over, mild = {}, {}, [], []
            for s in OTHER_SENSES:
                m = med[s]
                if not (_num(level.get(s)) and level[s] > 0 and _num(m)):
                    continue
                a_mult, s_mult = k6_levels(s)
                shares[s], alert_shares[s] = m / (s_mult * level[s]), m / (a_mult * level[s])
                words = ("%s's median next-model-step error ratio reached %s of its level before the switch (%s of "
                         "its stop line, %s of its alert line)" % (s, _pct(m / level[s]), _pct(shares[s]),
                                                                  _pct(alert_shares[s])))
                if shares[s] > 1.0:
                    over.append(words)
                elif alert_shares[s] > 1.0:
                    mild.append((s, words))
            for s in ALERT_ONLY_SENSES:
                m, lev = med[s], level_only.get(s)
                if _num(m) and _num(lev) and lev > 0:
                    alert_shares[s] = m / (LINES["alert_only_multiple"] * lev)
                    if alert_shares[s] > 1.0:
                        mild.append((s, "%s's median next-model-step error ratio reached %s of its level before "
                                        "the switch (%s of its alert line; this sense has no stop line)"
                                     % (s, _pct(m / lev), _pct(alert_shares[s]))))
            line_words = line_words_m4(level_n)
            if over:
                cid = "M4:%s" % wid
                why = "over %s, %s (the stop line: %s)" % (span, "; ".join(over), line_words)
                if cid not in taken:
                    crossings.append((4 + 2 * i, cid, "M4", "asked-change path: " + why, win[-1]["step"]))
                crossed_windows["M4"].append(cid)
                newest["M4"] = dict(state="crossed, its stop already taken" if cid in taken else "crossed", why=why,
                                    crossing=cid, shares=shares, alert_shares=alert_shares, window=i)
            else:
                newest["M4"] = dict(state="held", shares=shares, alert_shares=alert_shares, window=i,
                                    steps=[win[0]["step"], win[-1]["step"]])
            if mild:
                rep["alerts"].append(dict(
                    id="M4-alert:%s" % wid, line="M4", window=i, step=win[-1]["step"], senses=[s for s, _w in mild],
                    why=("asked-change path ALERT (printed and recorded, never a stop): over %s, %s (the alert "
                         "line: %s)" % (span, "; ".join(w for _s, w in mild), line_words))))
        # M5: the step rate against the reference's, two consecutive windows
        ref_sps = sps0 if prev is None else _median(r.get("sps") for r in prev)
        sps = _median(r.get("sps") for r in win)
        slow_now = False
        if _num(ref_sps) and ref_sps > 0:
            share = sps / ref_sps if _num(sps) else None
            slow_now = bool(_num(share) and share < LINES["M5_share"])
            if slow_now and prev_slow:
                cid = "M5:%s" % wid
                why = ("over %s the trainer ran at %s of its model steps per second in %s, the second window "
                       "running under the stop line of %s"
                       % (span, _pct(share), "the 20 reports before the switch" if prev is None
                          else "the 20 reports before this window", _pct(LINES["M5_share"])))
                if cid not in taken:
                    crossings.append((5 + 2 * i, cid, "M5", "asked-change path: " + why, win[-1]["step"]))
                crossed_windows["M5"].append(cid)
                newest["M5"] = dict(state="crossed, its stop already taken" if cid in taken else "crossed", why=why,
                                    crossing=cid, share=share, window=i)
            else:
                newest["M5"] = dict(state="held", share=share, window=i, under_line_this_window=slow_now)
        prev, prev_slow = win, slow_now
    for line in ("M4", "M5"):
        if newest[line] is not None:
            rep["lines"][line] = dict(newest[line], windows=len(wins), crossed_windows=crossed_windows[line])
        else:
            rep["lines"][line] = dict(state="waiting", windows=len(wins), crossed_windows=[], why=(
                "%d of the 21 full reports after the switch" % len(full) if not wins else
                "the reports before the switch are missing"))
    if readings:
        rep["acceptance"] = acceptance(before, full, N)
        rep["readings"].update(free_readings(before, full), first_reports=first_reports(recent, full, N, per_step))
    if not crossings:
        return None, None, rep
    rank, cid, line, why, step = min(crossings, key=lambda c: c[0])
    rep["stop"] = dict(id=cid, line=line, step=step,
                       owner=next((r.get("run") for r in after if r["step"] == step), None))
    return why, cid, rep


def line_words_m4(level_n):
    """What M4's two window lines are, for a person (shares, never bare decimals)."""
    steady = sorted({LINES["RISE"][s] for s in LINES["steady_senses"]})
    return ("the sense's level before the switch, the larger of its median over the 20 and over the %d full reports "
            "before it, times the highest a 20-report median reached over that level on the lineage's record with "
            "nothing switched, %s of the level for tactile, audio and proprioception and %s for vision; times %s for "
            "the stop line and %s for the alert line"
            % (level_n, "/".join(_pct(x) for x in steady), _pct(LINES["RISE"]["vision"]),
               _pct(LINES["stop_margin"]), _pct(LINES["alert_margin"])))


# ------------------------------------------------------------------ the pass lines and the readings
def _cells(rows):
    return [c for r in rows for c in (r.get("s1") or [])]


def s1_judgement(reference, window):
    """Line S1 (module note) from two readings (own_command_reading dicts): met / missed with its arithmetic."""
    z = LINES["S1_z"]
    out = dict(reading=window, reference=reference, z=z)
    if not window or window.get("se") is None:
        out["state"] = "waiting"
        return out
    if not reference or reference.get("se") is None:
        out["state"] = "unmeasured"
        out["why"] = "no full report before the switch carries the hold-first block with the size-matched arm"
        return out
    diff = window["value"] - reference["value"]
    se_d = math.sqrt(window["se"] ** 2 + reference["se"] ** 2)
    out.update(difference=diff, se_difference=se_d, smallest_rise_the_line_can_declare=z * se_d,
               reading_above_its_line=window["value"] > z * window["se"], difference_above_its_line=diff > z * se_d,
               share_of_goal_reference=window["value"] / LINES["S1_goal_reference"])
    out["state"] = "met" if (out["reading_above_its_line"] and out["difference_above_its_line"]) else "missed"
    return out


def acceptance(before, full, N):
    """S1 with its reference, its 1,000-model-step windows and its late reading (module note)."""
    ref_rows = [r for r in before if r["step"] > LINES["reference_after_step"]]
    reference = own_command_reading(_cells(ref_rows))
    if reference:
        reference.update(reports=sum(1 for r in ref_rows if r.get("s1")),
                         steps=[ref_rows[0]["step"], ref_rows[-1]["step"]] if ref_rows else None)
    out = dict(reference=reference, windows=[])

    def window(lo, hi):
        """(the full reports of model steps N + lo + 1 to N + hi, whether the window is over): over once a full
        report at or past the window's last report step exists. A switch off the hundreds (a requested stop's
        checkpoint) leaves the first window one full report short: the record after a resume is partial."""
        rows = [r for r in full if N + lo < r["step"] <= N + hi]
        last_due = ((N + hi) // REPORT_EVERY) * REPORT_EVERY
        return rows, bool(rows) and any(r["step"] >= last_due for r in full)
    for lo in range(0, LINES["horizon_steps"], 1000):
        rows, complete = window(lo, lo + 1000)
        out["windows"].append(dict(model_steps=[lo + 1, lo + 1000], reports=len(rows), complete=complete,
                                   reading=own_command_reading(_cells(rows)) if complete else None))
    for key, (lo, hi) in (("S1", LINES["S1_window"]), ("S1_late", LINES["S1_late_window"])):
        rows, complete = window(lo, hi)
        j = s1_judgement(reference, own_command_reading(_cells(rows)) if complete else None)
        j.update(model_steps=[N + lo + 1, N + hi], reports=len(rows))
        if j["state"] == "waiting" and complete:
            j.update(state="unmeasured", why="the window's reports carry no hold-first block with the size-matched arm")
        out[key] = j
    return out


def free_readings(before, full):
    """Never judged (module note): over the newest up to FREE_WINDOW full reports after the switch, beside the 20
    before it."""
    def block(rows):
        gates = {}
        for r in rows:
            for k, v in (r.get("gates") or {}).items():
                if _num(v):
                    gates[k] = gates.get(k, 0) + v
        return dict(reports=len(rows), steps=[rows[0]["step"], rows[-1]["step"]] if rows else None,
                    hold_loss_ratio=_median(r.get("hold_ratio") for r in rows),
                    contact_gate_self_open_share=_median((r.get("contact") or {}).get("gate_self_open_share")
                                                         for r in rows),
                    contact_gate_world_open_share=_median((r.get("contact") or {}).get("gate_world_open_share")
                                                          for r in rows),
                    gates=gates, planner_tied_share=_median(r.get("tied") for r in rows),
                    candidate_spread_over_tie_band=_median(r.get("spread_over_band") for r in rows))
    return dict(free_before=block(before[-LINES["before_reports"]:]), free_after=block(full[-FREE_WINDOW:]))


def first_reports(recent, full, N, per_step):
    """The first three full reports after the switch, for the relaunch's checklist (the record's section 9)."""
    sps3 = [r.get("sps") for r in recent[-3:] if _num(r.get("sps"))]
    ref = sum(sps3) / len(sps3) if sps3 else None
    out = []
    for r in full[:3]:
        p = r.get("path") or {}
        out.append(dict(step=r["step"],
                        steps_per_s_share=(r["sps"] / ref if _num(r.get("sps")) and _num(ref) and ref > 0 else None),
                        gpu_used_mib=r.get("gpu_used_mib"), gpu_total_mib=r.get("gpu_total_mib"),
                        output_rms=p.get("output_rms"), growth_line=growth_line(r["step"] - N, per_step),
                        output_weight_rms=p.get("output_weight_rms"), features_rms=p.get("features_rms"),
                        reach_spread=p.get("reach_spread"), path_live=r.get("path_live"),
                        nan_steps=r.get("nan_steps"), hold_first_cells=len(r.get("s1") or [])))
    return out


# ------------------------------------------------------------------ what the hourly watcher calls
def evaluate(run, reader=None, write=True, readings=True):
    """The judgement for `run`: None when no switch reaches it, else a dict (deploy, N, chain, judgement, why, id,
    warnings, watch_over)."""
    reader = reader or READER
    found = find_switch(run, reader)
    if found is None:
        return None
    deploy, sw, chain = found
    N = int(sw["step"])
    args = _read_json(Path(deploy) / "args.json")
    known = isinstance(args, dict) and bool(args)
    per_step = optimizer_updates_per_model_step(args) if known else None
    on_card = known and str(args.get("device") or "").startswith("cuda")
    before = BASE.before_reports(reader, deploy, N, need=LINES["level_reports"])
    after, over = watch_records(BASE.after_records(reader, chain, N), N)
    taken = still_taken(taken_records(deploy, write=write), after, chain)
    skipped = [(Path(r).name,) + tuple(reader.skipped_updates(r)) for r in chain] if hasattr(
        reader, "skipped_updates") else ()
    why, cid, rep = judge(before, after, sw, N, taken, per_step=per_step, on_card=on_card, readings=readings,
                          skipped=skipped, newest_run=Path(chain[-1]).name)
    rep["readout"] = _read_json(Path(deploy) / READOUT_NAME)
    if readings:
        rep["readings"]["launch_settings"] = launch_settings(deploy, before, chain)
    return dict(deploy=deploy, N=N, chain=[str(r) for r in chain], judgement=rep, why=why, id=cid,
                warnings=list(rep.get("warnings") or []), watch_over=over)


def breach(run, reader=None):
    """The hourly watcher's 5-minute check: a stop reason for `run`, or None. It prints a loud line for every part of
    a switched run it cannot judge, prints each new alert once, and records a reported crossing in the deploy run's
    state file first (module note: A STOP IS NOT REPEATED ON ITS OWN CROSSING)."""
    found = find_switch(run, reader or READER)
    if found is None or load_state(found[0]).get("closed"):
        return None                               # no switch reaches the run, or its watch was judged to its end
    ev = evaluate(run, reader, write=True, readings=False)
    if ev is None:
        return None
    if not ev["watch_over"]:
        for w in ev["warnings"]:
            print("  ASKED-CHANGE PATH STOP LINES INCOMPLETE: " + w, flush=True)
    for alert in record_alerts(ev["deploy"], run, ev["judgement"].get("alerts") or []):
        print("  " + alert["why"], flush=True)
    if not ev["why"]:
        if ev["watch_over"]:
            # the watch was judged to its end with nothing left to stop: no later check judges it again
            BASE._write_state(ev["deploy"], dict(load_state(ev["deploy"]), closed=dict(
                time=time.time(), run=str(Path(run)), newest_step=ev["judgement"].get("newest_step"))))
        return None
    stop = ev["judgement"].get("stop") or {}
    record_crossing(ev["deploy"], dict(id=ev["id"], line=stop.get("line"), reason=ev["why"], run=str(Path(run)),
                                       time=time.time(), step=stop.get("step"), owner=stop.get("owner")))
    return ev["why"]


def _ppm(x):
    """An own-command reading as whole parts in a million (H84's readings are a few parts in 100,000, which a
    percentage would print with an exponent)."""
    return "unmeasured" if not _num(x) else "{:,.0f} parts in a million".format(1e6 * x)


def _hold(x):
    """The hold loss's ratio with two decimals of a percent (H84 reads within a tenth of a percent of 100%)."""
    return "unmeasured" if not _num(x) else "%.2f%%" % (100.0 * x)


def _reading_words(x):
    """An own-command reading for a person: 0 = the own command predicts no better than another body's command of
    the same size, 1,000,000 parts in a million = exact where the other is not."""
    if not x or not _num(x.get("value")):
        return "unmeasured"
    return "%s with a standard error of %s on %s transitions" % (
        _ppm(x["value"]), _ppm(x.get("se")), "{:,}".format(int(x.get("transitions") or 0)))


def lines(run, reader=None, write=True):
    """The hourly sweep's lines for a person (shares of lines, counts, model steps); [] when no switch reaches `run`."""
    ev = evaluate(run, reader, write=write)
    if ev is None:
        return []
    rep, N = ev["judgement"], ev["N"]
    newest = rep.get("newest_step")
    L = rep.get("lines") or {}

    def state(k):
        v = L.get(k) or {}
        return v.get("state", "waiting") + ((" - " + v["why"]) if v.get("why") and v.get("state") != "held" else "")

    def shares(k, key="shares"):
        s = (L.get(k) or {}).get(key) or {}
        return ", ".join("%s %s" % (x, _pct(v)) for x, v in s.items()) or "unmeasured"
    out = ["  asked-change path: switched on at model step {:,} (--motor-asked-change on); {} full reports since, "
           "newest model step {}; the stop lines judge through the first full report at or past model step {:,}{}"
           .format(N, rep.get("after_full_reports"), "none" if newest is None else "{:,}".format(newest),
                   N + LINES["horizon_steps"], " (reached: the watch is over)" if ev["watch_over"] else "")]
    for w in ev["warnings"]:
        out.append("  ASKED-CHANGE PATH STOP LINES INCOMPLETE: " + w)
    out.append("  asked-change path, the switch record (the motor token with the path against without it on 64 probe "
               "rows must be the same number): %s; non-finite updates (the stats' count of non-finite live losses, "
               "and the trainer's log lines for a skipped optimizer update: %d so far): %s"
               % (state("M1"), (L.get("M2") or {}).get("skipped_updates_in_the_logs") or 0, state("M2")))
    m3 = L.get("M3") or {}
    out.append("  asked-change path, its hidden layer: %s; %s (stop line %s in %d full reports with a reading running; "
               "the reading is the lower of the whole-model monitor's and the path's own on the commanded rows)"
               % (state("M3"), "no full report read yet" if not _num(m3.get("lowest_live_share")) else
                  "at its lowest %s of its units responded" % _pct(m3["lowest_live_share"]),
                  _pct(LINES["M3_liveness"]), LINES["M3_consecutive"]))
    m6 = (L.get("M6") or {}).get("newest") or {}
    out.append("  asked-change path, its output (0%% = the path adds nothing to the motor token, 100%% = as large as "
               "the motor encoder's own standardised output): %s; %s (above 100%% of the growth line stops the run; "
               "the line is the most Adam can add since the switch, %s of a token unit per optimizer update, %s "
               "optimizer updates per model step)"
               % (state("M6"), "no report carries it yet" if not _num(m6.get("output_rms")) else
                  "at model step {:,} it reads {} of one token unit, {} of its growth line".format(
                      m6["step"], _pct(m6["output_rms"]), _pct(m6.get("share_of_line"))),
                  _pct(LINES["growth_per_optimizer_update"]),
                  ("%g" % rep["optimizer_updates_per_model_step"]) if _num(rep.get("optimizer_updates_per_model_step"))
                  else "an unknown number of"))
    r1 = (rep.get("readings") or {}).get("report1")
    m4 = L.get("M4") or {}
    out.append("  asked-change path, the other senses in reports 2 and 3 after the switch (each sense's next-model-"
               "step error ratio as a share of its line, the lower of the two reports; over 100%% in both stops the "
               "run): %s; %s. Report 1, printed and never judged: %s"
               % (state("M4_early"), shares("M4_early"),
                  "unmeasured" if not r1 else ", ".join(
                      "%s at %s of its median before" % (s, _pct(v))
                      for s, v in (r1.get("shares_of_median") or {}).items())))
    out.append("  asked-change path, the other senses against their level before the switch, every 20 reports, two "
               "levels (over 100%% of the stop line stops the run; over 100%% of the alert line is an alert, printed "
               "and recorded, never a stop): %s; the newest window (number %s), of the stop line: %s; of the alert "
               "line: %s" % (state("M4"), "none" if m4.get("window") is None else m4["window"] + 1, shares("M4"),
                             shares("M4", "alert_shares")))
    alerts = rep.get("alerts") or []
    out.append("  asked-change path, alerts (never a stop): %s" % (
        "none so far" if not alerts else "%d, the newest: %s" % (len(alerts), alerts[-1]["why"])))
    m5 = L.get("M5") or {}
    out.append("  asked-change path, speed: %s; in the newest window the trainer ran at %s of its model steps per "
               "second in %s (under %s in two windows running stops the run)"
               % (state("M5"), _pct(m5.get("share")),
                  "the 20 reports before the switch" if m5.get("window") == 0 else "the window before it",
                  _pct(LINES["M5_share"])))
    A = rep.get("acceptance") or {}
    s1, late, ref = A.get("S1") or {}, A.get("S1_late") or {}, A.get("reference")
    trend = "; ".join("model steps {:,} to {:,} after the switch {}".format(
        w["model_steps"][0], w["model_steps"][1], _ppm((w.get("reading") or {}).get("value")))
        for w in (A.get("windows") or []) if w.get("reading")) or "no window of 1,000 model steps is complete"
    out.append("  asked-change path, pass line S1, never a stop (proprioception at the hold's first model step: how "
               "much better the main route predicts with the body's own command than with another body's command "
               "of the same size; 0 = no better, 1,000,000 parts in a million = exact where the other is not; the "
               "last lineage measured to depend on its command read {:,.0f} parts in a million on the absolute "
               "swap): {} over model steps {}; the reading {}; the reference before the switch {}; the smallest "
               "rise the line can declare {}; the later reading over the watch's last 1,000 model steps: {}. Per "
               "window: {}".format(
                   1e6 * LINES["S1_goal_reference"], s1.get("state", "waiting"),
                   "{:,} to {:,}".format(*s1["model_steps"]) if s1.get("model_steps") else "unmeasured",
                   _reading_words(s1.get("reading")), _reading_words(ref),
                   _ppm(s1.get("smallest_rise_the_line_can_declare")), late.get("state", "waiting"), trend))
    ro = rep.get("readout")
    if isinstance(ro, dict) and isinstance(ro.get("line"), dict):
        groups = ro["line"].get("groups") or {}
        s2 = "; ".join("%s: the read-out recovers %s of the held-out body's variation (0%% = nothing beyond the "
                       "mean, 100%% = exact), its permuted control %s" % (
                           g.replace("_", " "), _pct((v or {}).get("r2")), _pct((v or {}).get("control_q975")))
                       for g, v in groups.items())
        out.append("  asked-change path, pass line S2, never a stop (the motor token carries the asked change; it "
                   "says only that the path's output is not zero): %s on checkpoint %s; %s"
                   % ("met" if ro["line"].get("above") else "missed", Path(str(ro.get("checkpoint") or "")).name, s2))
    else:
        out.append("  asked-change path, pass line S2, never a stop (the motor token carries the asked change): "
                   "waiting for the read-out job on a checkpoint after the switch")
    fr = rep.get("readings") or {}
    fa, fb = fr.get("free_after") or {}, fr.get("free_before") or {}
    ga, gb = fa.get("gates") or {}, fb.get("gates") or {}

    def gate_words(g):
        if not g.get("gates_same_commands"):
            return "unmeasured"
        return ("of {:,.0f} graded commands judged by both gates the gate in force stood open after {:,.0f} and the "
                "shadow gate after {:,.0f}, and they differed on {:,.0f}".format(
                    g["gates_same_commands"], g.get("gate_open_commands") or 0,
                    g.get("shadow_gate_open_commands") or 0,
                    (g.get("gate_only_open_commands") or 0) + (g.get("shadow_gate_only_open_commands") or 0)))
    out.append("  asked-change path, readings (never judged; the newest %d full reports, beside the 20 before the "
               "switch): the hold loss's ratio %s, before %s (100%% = the command adds nothing to the pose prediction "
               "over the hold; the hinge's line is 96%%); the contact gates open on %s (self) and %s (world) of their "
               "readings, before %s and %s; controllability: %s; before: %s; the planner's chosen command tied with "
               "another candidate at %s of its planning events, before %s (100%% = the predictions cannot tell the "
               "candidates apart)"
               % (fa.get("reports") or 0, _hold(fa.get("hold_loss_ratio")), _hold(fb.get("hold_loss_ratio")),
                  _pct(fa.get("contact_gate_self_open_share")), _pct(fa.get("contact_gate_world_open_share")),
                  _pct(fb.get("contact_gate_self_open_share")), _pct(fb.get("contact_gate_world_open_share")),
                  gate_words(ga), gate_words(gb), _pct(fa.get("planner_tied_share")),
                  _pct(fb.get("planner_tied_share"))))
    ls = fr.get("launch_settings") or {}

    def names(launch):
        got = sorted(list((launch or {}).get("changed") or {}) + list((launch or {}).get("new") or {}))
        off = (launch or {}).get("new_at_an_off_value") or []
        return (", ".join(got) if got else "nothing") + (
            " (and %d setting(s) the run before does not hold, each at an off value: %s)" % (len(off), ", ".join(off))
            if off else "")
    if ls.get("switch") is None:
        own = "the switched run's args.json or its parent's could not be read"
    elif ls.get("besides_the_flag"):
        own = ("the switch's relaunch changed %s - MORE THAN THE PATH CHANGED AT THE SWITCH: a stop on the other senses "
               "or on the step rate cannot be put down to the path alone" % names(ls["switch"]))
    else:
        own = "the switch's relaunch changed %s and nothing else" % names(ls["switch"])
    earlier = "; ".join(
        "the relaunch at model step {:,} changed {}, with {} full reports between it and the switch{}".format(
            x["began_at"], names(x["launch"]), x["full_reports_after"],
            " (INSIDE THE 20 REPORTS the early line and the step rate's first window read)"
            if x["full_reports_after"] < LINES["before_reports"] else "")
        for x in ls.get("before") or []) or "no relaunch changed a setting"
    if ls.get("unread"):
        earlier += "; the launch settings of %d relaunch(es) inside those reports could not be read" % len(ls["unread"])
    after_words = "; ".join(
        "the relaunch at model step {:,} changed {} - A SETTING CHANGED INSIDE THE WATCH".format(
            x["began_at"], names(x["launch"])) for x in ls.get("after") or []) or "no relaunch changed a setting"
    if ls.get("after_unread"):
        after_words += "; the launch settings of %d relaunch(es) after the switch could not be read" % len(
            ls["after_unread"])
    out.append("  asked-change path, the launch settings around the switch (each run's args.json against the run it "
               "continues; never judged): %s; within the %s full reports before the switch: %s; after the switch: %s"
               % (own, rep.get("before_reports"), earlier, after_words))
    first = fr.get("first_reports") or []
    if first:
        out.append("  asked-change path, the first full reports after the switch: " + "; ".join(
            "model step {:,}: {} of the step rate of the 3 reports before the switch, graphics card {} of {} MiB, "
            "the path's output {} of a token unit against a growth line of {} of a token unit, its hidden layer {} "
            "live, {} non-finite updates".format(
                f["step"], _pct(f.get("steps_per_s_share")),
                "unmeasured" if not _num(f.get("gpu_used_mib")) else "{:,.0f}".format(f["gpu_used_mib"]),
                "unmeasured" if not _num(f.get("gpu_total_mib")) else "{:,.0f}".format(f["gpu_total_mib"]),
                _pct(f.get("output_rms")), _pct(f.get("growth_line")), _pct(f.get("path_live")),
                f.get("nan_steps")) for f in first))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Print the asked-change path's stop lines and pass lines for a run "
                                             "(reading only: nothing is written).")
    ap.add_argument("run")
    a = ap.parse_args(argv)
    ev = evaluate(a.run, StatsReader(), write=False)
    if ev is None:
        print("no motor_asked_change_switch record reaches %s" % a.run)
        return 0
    for line in lines(a.run, StatsReader(), write=False):
        print(line)
    if ev["why"]:
        print("A STOP LINE IS CROSSED (the hourly watcher writes the stop): " + ev["why"])
    return 1 if ev["why"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
