"""The stop lines and pass lines of the all-slot paths' switch in production (--heads-all-slot-path on), judged by the
H84 hourly watcher. Plan and arithmetic: docs/gain_of_function_v4/restart_candidate/RESTART_CANDIDATE_HEADS_2026-10-04.md;
registry row heads_all_slot_path_production_segment.

WHAT THE SWITCH IS. Production's command gets --heads-all-slot-path on at a relaunch. The load then builds, beside each
of the five prediction heads (vision, tactile, audio, proprioception, vestibular), a path that reads the 6 newest
tokens minus the persistent memory's constant, centred by a running mean and LayerNormed, through a 2-layer network
whose output layer starts at zero (lem/model/heads_all_slot_path.py), and the trainer writes one
heads_all_slot_path_switch record into the stats stream. From the next optimizer update on each path's output is
added to its head's prediction.

HOW IT IS JUDGED. As the asked-change path's switch is (motor_asked_change_watch.py, whose code this module runs
wherever the arithmetic is the same): the hourly watcher's 5-minute check calls breach(RUN) and writes the stop with
its own stop_run; its hourly sweep prints lines(RUN). A run that no heads_all_slot_path_switch record reaches is not
judged: breach() returns None and lines() returns nothing for it.

By hand, reading only (nothing is written, not even the state file):
    python docs/gain_of_function_v4/heads_all_slot_path_watch.py <run folder>

THE BASE MODULE. `MW` is a private instance of motor_asked_change_watch.py, loaded by path under another module name
(it loads its own private instance of vestibular_repair_watch.py), with these names replaced on that instance only:
SWITCH_TYPE and STATE_NAME (this switch's record type and state file, also on its vestibular instance), PATH_VERSION,
compact (the fields this watch reads from a stats record), check_switch (what "as declared" means for this switch),
acceptance (this watch's early and late lines), judge (the base's judgement of M1 to M6 with this watch's M7 and M8
added) and launch_settings (the flag's name), and four numbers of its LINES dict (the
growth line's bound, the output alert, and the two keys they are printed from). Its walk over the run folders, its
reader, its timeline, its 20-report windows, its level before the switch, its two multiples per sense, its bookkeeping
of crossings whose stop was taken, and its judgement of the stop lines M1 to M6 are then this module's, unchanged.
The hourly watcher's asked-change rule and its vestibular rule are other instances of those files and read their own
names: nothing of either watch changes (tests/test_heads_all_slot_path_watch.py holds the instances apart).

THE STOP LINES (each writes snapshot_and_stop; arithmetic in the record's section 7), judged from the switch at model
step N through the first full report at or past N + 10,000:
  M1  the switch record is as declared: it says exact; every precision it checked (float32, and the run's autocast
      when the run is on the graphics card) reads a largest change of any head's prediction of exactly 0.0 on
      bit-identical predictions; every output layer's largest absolute weight and bias are 0.0; the check ran on 64
      probe rows; the version is this build's. Judged at once.
  M2  a non-finite update after N: nan_steps above 0 in any stats record, or a line of the trainer's log that reports
      a skipped optimizer update.
  M3  a path's hidden layer under 0.9 live share in 2 full reports running that carry a reading. The reading of a
      report is the LOWEST over the five paths of the whole-model liveness monitor's reading
      (liveness.layers["heads_all_slot_path.paths.<head>.net.0"]) and the path's own gauge
      (heads_all_slot_path.heads.<head>.live, the newest lesson rows).
  M4  the senses' next-model-step error ratio (next_frame_norm.ratio: prediction error over the error of predicting
      no change; 0 = exact, 1 = no better than no change) for vision, tactile, audio and proprioception: in reports 2
      and 3 against the sense's median over the 20 reports before the switch x W1, and in every window of 20 full
      reports against its level before the switch (above level x RISE: an alert, printed once and recorded, never a
      stop; above level x RISE x 1.2: a stop). W1, RISE and 1.2 are the vestibular watch's. Vestibular's own ratio is
      an alert only (the asked-change watch's rule and multiple). THIS IS THE LINE THAT PROTECTS PROPRIOCEPTION AND
      AUDIO, whose heads follow their input today.
  M5  the step rate: the median model steps per second of a 20-report window under 0.9 of the window before it (the
      first window: of the 20 reports before the switch) in two windows running.
  M6  A PATH'S OUTPUT AGAINST THE SIZE OF WHAT IT PREDICTS (re-derived Oct 4, 2026 after the first segment's
      runaway; the record's sections 13 and 14). At a full report, per head: heads_all_slot_path.heads.<head>.
      output_rms (root mean square per number of the path's output on the newest lesson rows, token units) over the
      sense's real change per number, sqrt(next_frame_norm.persistence_ema[<head>]) (the running mean squared change
      of that sense's token from one model step to the next). 0 = the path adds nothing; 1 = its output is as large
      as the whole change it predicts. THE STOP: the largest of the five shares above M6_SHARE (1.0), or an output
      that is not a finite number. THE REAL CHANGE IS TAKEN AS AT LEAST M6_FLOOR (0.5) OF ITS VALUE AT THE LAST FULL
      REPORT BEFORE THE SWITCH: a sense whose token stops changing (the vestibular token read 0 before its encoder's
      repair at model step 455,800) would otherwise put the line at 0 and stop the run on any path output, for a
      fault that is the encoder's. Over the lineage's record since model step 415,200 each living sense's real
      change stayed at or above 0.64 of its value at 479,300 (m6_line_result.json, lineage). WHY THIS BOUND. A path's output is added to the old head's to predict the token's
      change. Both parts receive the same lessons (version 2), so the lessons' request is on their SUM, not on either
      part: the path alone can exceed the whole change while the old head's output cancels part of it, as a fixed
      vector or row by row (an old head of -d and a path of 2d predict a change d exactly). M6 is therefore a declared
      operational guard on the path's size, set from the readings below, not a bound that every useful prediction
      obeys; a crossing is read with the old head, the path and the target on the same rows before a learned
      component is called defective. Its two numbers also have different samples: output_rms the newest lesson rows
      (up to 64: the newest 8 lesson calls x 8 bodies), the real change a running mean with a 2,000-model-step horizon
      (next_frame_norm.HORIZON_STEPS). Corrected Oct 13, 2026 under proxy decision
      proxy-recovery-stop-492800-next-actions-20261013; the line, its numbers and its logic are unchanged. In the first
      segment the tactile path read 0.06, 0.52, 3.3 and 17 of the real change at model steps 479,400 to 479,700: the
      line is crossed at 479,600, the report the first design's growth line stopped at. On the bench (transformer held
      fixed, 1,500 updates) the five paths read at most 0.50, and in the repaired design's CPU run of 2,100 updates
      at most the value the record's section 14 states. The first design's line (0.00078 token units x the optimizer
      updates since the switch) bounded the output layer's own movement and held the layers before it fixed; they
      learn too, so that line was not a bound.
  M9  THE PAIR THAT RAN AWAY (a stop line by the coordinator's decision of Oct 4, 1:50 PM PDT). At a full report, a
      head whose path's fixed vector (heads_all_slot_path.fixed.<head>.path_rms: the root mean square of the path's
      output averaged over the lesson rows) is at or above FIXED_ALERT_SHARE (0.5) of the sense's real change while
      its cosine with the old head's fixed vector (fixed.<head>.cosine) is at or under FIXED_ALERT_COSINE (-0.9): the
      two parts add opposite fixed vectors. In checkpoint 479,700 the tactile pair read 2.68 and 2.78 token units
      with cosine -0.9998. COUNTED (m6_line_result.json, pair): the lineage's records carry no such block, so no
      count can be taken on them; the repaired design's CPU run crosses at 0 of its model steps; the first design's
      CPU run crosses before it crosses M6. A report without the block is not judged by this line.

  M7  PROPRIOCEPTION KEEPS ITS OWN-COMMAND READING. In every window of 1,000 model steps after the switch (the full
      reports of model steps N + 1,000(k-1) + 1 to N + 1,000k), proprioception's own-command reading at the hold's
      first model step (1 - summed error with the own command / summed error with the size-matched command, pooled
      over the window's (report, body) cells, bodies without body 2) at or under 3.14 x its standard error (2,000
      seeded resamplings of the cells). 3.14 standard errors is the asked-change watch's pass line S1, read here as a
      floor: the asked-change path brought this reading from 94 parts in a million to 0.32 to 0.49, and the four
      windows from model step 475,876 to 479,300 read 10 to 15 standard errors. Judged only when the 20 full reports
      before the switch themselves read above the floor.
  M8  NO HEAD'S AUDITED ERROR RISES. Per head, a window's audited error ratio (summed prediction error / summed
      no-change error over the window's (report, body) cells of prediction_audit_hold_first, bodies without body 2:
      0 = exact, 1 = no better than no change) above its value over the 20 full reports before the switch by more
      than SPREAD, in two windows of 1,000 model steps running. SPREAD per head is the LARGEST rise of any of the
      watch's 10 windows over the 20 reports before the switch, at every switch position of the lineage's record
      with nothing switched whose 20 reports before carry the audit (for proprioception, whose level changed at the
      asked-change switch, the largest rise from one separate window to a later one among the windows since model
      step 475,500): M8_SPREAD below; heads_all_slot_path_false_stops.py prints the measurement (key audit). A
      head is judged when at least 15 of the 20 reports before the switch carry its audit cells.
  ALERT, never a stop: the tactile or the vestibular lesson's gradient on the transformer's output
  (heads_all_slot_path.lesson_gradient.<head>.rms) under GRADIENT_ALIVE at a full report past model step N + 1,000.

THE EARLY LINES (printed by the hourly sweep, never a stop), over the full reports of model steps N + 2,001 to
N + 3,000:
  E1  the switch record says every prediction is unchanged (stop line M1's own reading).
  E2  THE LESSONS' GRADIENT REACHES THE TRANSFORMER, for tactile and vestibular: the median over the window's reports
      of heads_all_slot_path.lesson_gradient.<head>.rms (per lesson row, the derivative of that row's mean squared
      error with respect to each of the 6 x 192 numbers the heads read; root mean square) at or above GRADIENT_ALIVE
      (1e-6). Through the old heads alone it reads 3e-10 (tactile) and 6e-10 (vestibular) at checkpoint 479,350;
      through a path trained on the bench 3e-4 and 6e-5 to 1e-4. A LIVENESS reading: it says the transformer receives
      these two lessons, not that anything is predicted.
  E3  EVERY HEAD READS MORE THAN ONE SLOT: the median of head_read.<head>.spread (the entropy over the 6 slots of how
      much the head's whole prediction moves when one slot's row is replaced by another body's, over log 6: 0 = one
      slot, 1 = all six evenly) at or above P4_LINE. The old attention pools keep reading one slot
      (liveness.pools["prediction_heads.heads.<head>.attn"], printed beside it): this candidate does not unlock them.

THE LATE PASS LINES (printed, never a stop). Goal references: what a ridge fit from the encoder tokens of 3 model
steps and the command reaches on bodies it did not see, with no transformer (registry row
prediction_heads_reachability_probe, checkpoint 479,350): GOAL below.
  L1  TACTILE'S ERROR IS BELOW NO CHANGE: over the full reports of model steps N + 9,001 to N + 10,000, tactile's
      audited error ratio below 1.0 by more than 3.14 x its standard error. Goal 0.88 (the probe reads 0.877 and
      0.902); the head reads 1.02 to 1.04 today.
  L2  VESTIBULAR'S PREDICTION IS BETTER WITH THE OWN COMMAND: vestibular's own-command reading above 3.14 x its
      standard error, pooled over the full reports of model steps N + 2,001 to N + 10,000. Goal 0.04 (the probe reads
      0.052 and 0.041). Why pooled over 8,000 model steps and not over the last 1,000: one window's standard error
      for the senses that have one is 0.004 (audio) to 0.03 (vision, proprioception), so a reading of 0.04 would be
      1.3 to 10 standard errors in one window and 3.8 to 28 pooled; the late window's own reading is printed beside it.
  WATCHED, with the reference before the switch and the goal reference, never judged: per head the audited error
  ratio, the own-command reading and how far the prediction moves under the swapped command
  (sensitivity_size_matched); tactile's own-command reading (the probe: 0.003 to 0.007, under 3 of its standard
  errors, so no line is put on it); vestibular's error ratio (the probe: 0.96 in one fold, 1.03 in the other);
  audio, vision and proprioception against their values before the switch.
The reference of every reading is the up to 100 full reports before the switch at model steps above
REFERENCE_AFTER_STEP (474,875: the asked-change switch; before it no sense's prediction depended on the command).

A STOP IS NOT REPEATED ON ITS OWN CROSSING: the asked-change watch's bookkeeping, in <deploy run>/
heads_all_slot_path_watch.json.
Tests: Source/Python/mimo_bridge/tests/test_heads_all_slot_path_watch.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_base():
    spec = importlib.util.spec_from_file_location("motor_asked_change_watch_as_heads_all_slot_path_base",
                                                  HERE / "motor_asked_change_watch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MW = _load_base()
BASE = MW.BASE
SWITCH_TYPE = "heads_all_slot_path_switch"
STATE_NAME = "heads_all_slot_path_watch.json"
PATH_VERSION = "heads_all_slot_path_v2"           # lem/model/heads_all_slot_path.py ALL_SLOT_PATH_VERSION
BLOCK = "heads_all_slot_path"
FLAG = "heads_all_slot_path"
HEADS = ("vision", "tactile", "audio", "proprioception", "vestibular")
HOLD_FIRST = MW.HOLD_FIRST
LABEL = "heads' all-slot paths"
REPORT_EVERY = MW.REPORT_EVERY

# THE FIRST DESIGN'S GROWTH LINE, kept as a printed reference only (module text, M6): learning rate x the summed
# absolute size of what a fresh output layer reads on one row (36.0 at the largest row of checkpoint 479,350:
# restart_candidate/heads/growth_line_result.json) x the output's gain 1 / sqrt(192) = 0.00078 token units per
# optimizer update. It is no longer a stop line: it held the layers before the output layer fixed, and they learn.
SUMMED_HIDDEN = 36.0
GROWTH_PER_UPDATE = 3e-4 * SUMMED_HIDDEN / math.sqrt(192.0)
# M6 (module text): a path's output over its sense's real change per number; above this share the run stops.
# MEASURED (restart_candidate/heads/m6_line_result.json, written by prediction_heads_dead/runaway/m6_line.py): the
# bench's largest share over 5 heads, 2 folds, 3 seeds and 6 readings to 1,500 updates; the repaired design's CPU run;
# the first segment's four reports; the lineage's lowest real change per sense.
M6_SHARE = 1.0
M6_FLOOR = 0.5                                    # of the sense's real change at the last full report before the switch
FIXED_ALERT_SHARE = 0.5
FIXED_ALERT_COSINE = -0.9
# an output of 2 token units is twice a token's own size per number; the real change of a token from one model step
# to the next is 0.08 to 0.18 per number, and the vision head's own constant output is 0.95 (the gaze copy cancels
# it), so a path output near 1 for vision is within what the lesson may ask; 2 is above every such need
OUTPUT_ALERT = 2.0
REFERENCE_AFTER_STEP = 474875
P4_LINE = 0.25
Z = 3.14
WINDOW_STEPS = 1000                               # M7, M8: one window is 1,000 model steps (10 full reports)
WINDOW_MIN_REPORTS = 5                            # a window with fewer full reports is not judged
M8_REFERENCE_REPORTS = 20
M8_REFERENCE_MIN = 15
# M8's SPREAD per head (module text): docs/gain_of_function_v4/restart_candidate/heads/false_stops_lineage_result.json,
# key audit.largest_rise (a test holds these numbers to that file)
M8_SPREAD = {"vision": 0.156, "tactile": 0.0178, "audio": 0.156, "proprioception": 0.075, "vestibular": 0.0121}
GRADIENT_ALIVE = 1e-6
GRADIENT_HEADS = ("tactile", "vestibular")
EARLY_WINDOW = (2000, 3000)
LATE_WINDOW = (9000, 10000)
POOLED_WINDOW = (2000, 10000)
# the probe's held-out readings at 50 ms, the two folds (registry row prediction_heads_reachability_probe; the
# record's section 6.2): error ratio and own-command reading; and at the end of the 250 ms hold
GOAL = {"tactile": dict(error=(0.877, 0.902), own=(0.003, 0.007), error_250=(0.773, 0.813), own_250=(0.012, 0.037)),
        "vestibular": dict(error=(0.963, 1.030), own=(0.041, 0.052), error_250=(0.849, 1.060), own_250=(0.087, 0.142)),
        "proprioception": dict(error=(0.650, 0.966), own=(0.520, 0.575), error_250=(0.525, 0.761),
                               own_250=(0.561, 0.629)),
        "vision": dict(error=(0.610, 0.674), own=(0.066, 0.077), error_250=(0.832, 0.835), own_250=(0.010, 0.011)),
        "audio": dict(error=(0.879, 0.896), own=(0.000, 0.005), error_250=(0.673, 0.710), own_250=(0.005, 0.018))}

LINES = MW.LINES                                  # the private instance's dict
LINES.update(growth_per_optimizer_update=GROWTH_PER_UPDATE, summed_input_size=SUMMED_HIDDEN, output_alert=OUTPUT_ALERT,
             M6_share=M6_SHARE, M6_floor=M6_FLOOR, fixed_alert_share=FIXED_ALERT_SHARE,
             fixed_alert_cosine=FIXED_ALERT_COSINE,
             reference_after_step=REFERENCE_AFTER_STEP, P4_line=P4_LINE, P_z=Z, M7_z=Z, M8_spread=M8_SPREAD,
             M8_reference_reports=M8_REFERENCE_REPORTS, window_steps=WINDOW_STEPS, gradient_alive=GRADIENT_ALIVE,
             early_window=EARLY_WINDOW, late_window=LATE_WINDOW, pooled_window=POOLED_WINDOW)

_num, _median, _pct, _read_json = MW._num, MW._median, MW._pct, MW._read_json
is_full = MW.is_full
_obj = MW._obj
_motor_compact = MW.compact


def path_layer(head: str) -> str:
    return "%s.paths.%s.net.0" % (BLOCK, head)


def relabel(text):
    """The base module's printed words for this switch."""
    if not isinstance(text, str):
        return text
    return (text.replace("asked-change path", LABEL).replace("ASKED-CHANGE PATH", LABEL.upper())
            .replace("motor_asked_change", BLOCK).replace("the path's hidden layer", "a path's hidden layer")
            .replace("the path's output", "the largest path output").replace("motor token", "predictions"))


# ------------------------------------------------------------------ the records
def compact(r):
    """The fields this watch's lines read, from a stats record (None for a record no line reads): the base module's
    fields with the path's taken from the heads_all_slot_path block, and this watch's own."""
    c = _motor_compact(r)
    if c is None or c.get("type") != "mimo":
        return c
    blk = r.get(BLOCK) if isinstance(r.get(BLOCK), dict) else None
    heads = _obj((blk or {}).get("heads"))
    read = _obj((blk or {}).get("head_read"))
    layers = _obj(_obj(r.get("liveness")).get("layers"))
    pools = _obj(_obj(r.get("liveness")).get("pools"))
    path, live = None, []
    per_head = {}
    ema = _obj(_obj(r.get("next_frame_norm")).get("persistence_ema"))
    fixed = _obj((blk or {}).get("fixed"))
    if blk is not None:
        outs = [(_obj(heads.get(m)).get("output_rms")) for m in HEADS]
        bad = [o for o in outs if isinstance(o, float) and not math.isfinite(o)]
        good = [o for o in outs if _num(o)]
        wts = [_obj(heads.get(m)).get("output_weight_rms") for m in HEADS]
        path = {"rows": blk.get("rows"), "output_rms": (bad[0] if bad else (max(good) if good else None)),
                "output_weight_rms": max([w for w in wts if _num(w)], default=None)}
        for m in HEADS:
            h = _obj(heads.get(m))
            a, b = layers.get(path_layer(m)), h.get("live")
            live += [float(v) for v in (a, b) if _num(v)]
            e = ema.get(m)
            per_head[m] = {"output_rms": h.get("output_rms"), "output_weight_rms": h.get("output_weight_rms"),
                           "live": b, "slot_reach": _obj(h.get("slot_reach")).get("spread"),
                           "head_read": _obj(read.get(m)).get("spread"),
                           "real_change": (math.sqrt(e) if _num(e) and e > 0 else None),
                           "output_fixed_rms": h.get("output_fixed_rms"),
                           "fixed": {k: _obj(fixed.get(m)).get(k) for k in ("old_rms", "path_rms", "sum_rms", "cosine")}}
    hf = _obj(_obj(r.get(HOLD_FIRST)).get("senses"))
    grad = _obj((blk or {}).get("lesson_gradient"))
    c.update(path=path, path_live=(min(live) if live else None), path_live_low=(min(live) if live else None),
             heads=per_head,
             real_change={m: (math.sqrt(ema[m]) if _num(ema.get(m)) and ema[m] > 0 else None) for m in HEADS},
             own={m: MW.swap_cells(r, sense=m) for m in HEADS},
             err={m: error_cells(r, m) for m in HEADS},
             gradient={m: _obj(grad.get(m)).get("rms") for m in HEADS},
             move={m: _obj(_obj(hf.get(m)).get("without_body_2")).get("sensitivity_size_matched") for m in HEADS},
             hf_ratio={m: _obj(_obj(hf.get(m)).get("without_body_2")).get("ratio") for m in HEADS},
             pool={m: pools.get("prediction_heads.heads.%s.attn" % m) for m in HEADS},
             vision_alone=_obj(_obj(r.get("efference")).get("gaze_copy")).get("vision_err_without_copy"))
    return c


def error_cells(record, sense, block=HOLD_FIRST):
    """The (report, body) cells of one sense's audited error: per body not left out, [transitions, summed prediction
    error, summed no-change error] (the stats block holds per-body means; a cell's sums are mean x transitions)."""
    per = _obj(_obj(_obj(_obj(record.get(block)).get("senses")).get(sense)).get("per_body"))
    n, pred, pers = per.get("transitions"), per.get("prediction_mse"), per.get("persistence_mse")
    if not (isinstance(n, list) and isinstance(pred, list) and isinstance(pers, list)):
        return []
    out = []
    for b in range(min(len(n), len(pred), len(pers))):
        if b in MW.EXCLUDED_BODIES or not _num(n[b]) or not n[b] or not _num(pred[b]) or not _num(pers[b]):
            continue
        out.append([int(n[b]), float(pred[b]) * int(n[b]), float(pers[b]) * int(n[b])])
    return out


def check_switch(sw, on_card=False):
    """None when the switch record is as declared (module note, M1), else why not (in words)."""
    sw = sw if isinstance(sw, dict) else {}
    checks = {k: v for k, v in sw.items() if isinstance(v, dict) and "change_max_abs" in v}
    if sw.get("exact") is not True:
        return "the switch record does not say that every prediction with the paths equals the prediction without them"
    if not checks:
        return "the switch record holds no check of the predictions"
    bad = sorted(k for k, v in checks.items() if not (v.get("change_max_abs") == 0.0 and v.get("bit_identical") is True))
    if bad:
        return "a prediction with the paths differs from the prediction without them on the probe rows in %s" % ", ".join(bad)
    if on_card and "autocast" not in checks:
        return "the prediction check did not run under the run's autocast, the precision the graphics card computes in"
    if sw.get("output_weight_max_abs") != 0.0 or sw.get("output_bias_max_abs", 0.0) != 0.0:
        return "an output layer was not zero at the load"
    if sw.get("rows") != 64:
        return "the prediction check ran on %s probe rows, not 64" % sw.get("rows")
    if sw.get("version") != PATH_VERSION:
        return "the switch record names path version %s, not %s" % (sw.get("version"), PATH_VERSION)
    return None


def launch_settings(deploy, before, chain=()):
    """The base module's launch_settings with this switch's flag as the one setting the switch itself changes."""
    own = MW.launch_changes(deploy)
    parent, _ = BASE.parent_of(deploy)
    folder = BASE.run_folder(parent, Path(deploy).parent) if parent else None
    walked = MW.relaunches_back(folder, before)
    others = [x for x in walked if MW._changed_something(x)]
    later = [dict(run=Path(r).name, began_at=BASE.start_of(r), launch=MW.launch_changes(r)) for r in list(chain)[1:]]
    besides = None
    if own is not None:
        besides = sorted(k for k in list(own["changed"]) + list(own["new"]) if k != FLAG)
    return dict(switch=own, besides_the_flag=besides, before=others,
                unread=[x for x in walked if x["launch"] is None],
                within_the_20_before=[x for x in others if x["full_reports_after"] < LINES["before_reports"]],
                after=[x for x in later if MW._changed_something(x)],
                after_unread=[x for x in later if x["launch"] is None])


# ------------------------------------------------------------------ the windows and their readings
def _window(full, N, lo, hi):
    """(the full reports of model steps N + lo + 1 to N + hi, whether the window is over)."""
    rows = [r for r in full if N + lo < r["step"] <= N + hi]
    last_due = ((N + hi) // REPORT_EVERY) * REPORT_EVERY
    return rows, bool(rows) and any(r["step"] >= last_due for r in full)


def error_reading(cells):
    """A window's audited error ratio from its (report, body) cells [transitions, summed prediction error, summed
    no-change error]: value = summed prediction error / summed no-change error, with the standard error of the
    asked-change watch's resampling; below = 1 - value (how far under the no-change error). None without cells."""
    oc = MW.own_command_reading(cells)
    if not oc:
        return None
    return dict(value=1.0 - oc["value"], below=oc["value"], se=oc["se"], transitions=oc["transitions"],
                cells=oc["cells"])


def _cells(rows, key, head):
    return [c for r in rows for c in (_obj(r.get(key)).get(head) or [])]


def pooled_ratio(cells):
    """Summed second number / summed third number of the cells (for error cells: the audited error ratio); None
    without cells or with a denominator of 0. No resampling: the stop line M8 reads the value alone."""
    den = sum(c[2] for c in cells)
    return (sum(c[1] for c in cells) / den) if cells and den > 0 else None


_READINGS = {}                                    # resampled readings by their cells (the 5-minute check and the
                                                  # false-stop count judge the same windows again and again)


def cached_reading(cells):
    """MW.own_command_reading(cells), kept by the cells' content (at most 4,096 entries)."""
    if not cells:
        return None
    key = (len(cells), sum(c[0] for c in cells), sum(c[1] for c in cells), sum(c[2] for c in cells),
           cells[0][1], cells[-1][2])
    if key not in _READINGS:
        if len(_READINGS) >= 4096:
            _READINGS.clear()
        _READINGS[key] = MW.own_command_reading(cells)
    return _READINGS[key]


def head_readings(rows, head):
    """A window's readings for one head, from compacted full reports."""
    return dict(reports=len(rows),
                own_command=MW.own_command_reading(_cells(rows, "own", head)),
                error=error_reading(_cells(rows, "err", head)),
                move=_median(_obj(r.get("move")).get(head) for r in rows),
                head_read=_median(_obj(_obj(r.get("heads")).get(head)).get("head_read") for r in rows),
                slot_reach=_median(_obj(_obj(r.get("heads")).get(head)).get("slot_reach") for r in rows),
                pool_spread=_median(_obj(r.get("pool")).get(head) for r in rows),
                gradient=_median(_obj(r.get("gradient")).get(head) for r in rows))


def _above_noise(reading):
    """True / False for a reading dict with value and se; None when it cannot be judged."""
    if not reading or reading.get("se") is None:
        return None
    return bool(reading["value"] > Z * reading["se"])


# ------------------------------------------------------------------ stop lines M7 and M8, and the gradient alert
def audited_lines(before, full, N, taken=None):
    """Lines M7 and M8 and the gradient alert (module text) on the full reports after the switch. Returns
    dict(lines={"M7": ..., "M8": ...}, crossings=[(model step, id, line, why)], alerts=[...], warnings=[...])."""
    taken = dict(taken or {})
    out = dict(lines={}, crossings=[], alerts=[], warnings=[])
    ref_rows = before[-M8_REFERENCE_REPORTS:] if len(before) >= M8_REFERENCE_REPORTS else []
    # the reference's own-command reading: resampled only when its value is above 0 (a reading at or under 0 is not
    # above any floor)
    ref_cells = _cells(ref_rows, "own", MW.JUDGED_SENSE)
    ref_value = pooled_ratio(ref_cells)
    ref_own = cached_reading(ref_cells) if (ref_value is not None and 1.0 - ref_value > 0) else None
    m7_judged = bool(_above_noise(ref_own))
    # a head's reference needs the audit's cells in at least M8_REFERENCE_MIN of the 20 reports (a reference of a few
    # reports is itself as noisy as one window)
    ref_err = {m: (pooled_ratio(_cells(ref_rows, "err", m))
                   if sum(1 for r in ref_rows if _obj(r.get("err")).get(m)) >= M8_REFERENCE_MIN else None)
               for m in HEADS}
    m7 = dict(state="waiting", judged=m7_judged, reference=ref_own, windows=[])
    m8 = dict(state="waiting", reference=dict(ref_err), windows=[],
              not_judged=sorted(m for m in HEADS if ref_err[m] is None))
    if not m7_judged:
        m7["state"] = "not judged"
        m7["why"] = ("the 20 full reports before the switch are missing" if not ref_rows else
                     "proprioception's own-command reading over the 20 full reports before the switch is not above "
                     "3.14 standard errors itself, so there is no reading to keep")
    if full and (not ref_rows or m8["not_judged"]):
        out["warnings"].append(
            "the audited-error line cannot be judged for %s: %s" % (
                ", ".join(m8["not_judged"]) if ref_rows else "any head",
                "fewer than 15 of the 20 full reports before the switch hold prediction_audit_hold_first cells for "
                "it" if ref_rows
                else "the 20 full reports before the switch were not found along the parent chain"))
    over_before = {m: None for m in HEADS}
    for k in range(1, LINES["horizon_steps"] // WINDOW_STEPS + 1):
        rows, complete = _window(full, N, WINDOW_STEPS * (k - 1), WINDOW_STEPS * k)
        if not complete:
            break
        span = "model steps {:,} to {:,}".format(N + WINDOW_STEPS * (k - 1) + 1, N + WINDOW_STEPS * k)
        step = rows[-1]["step"]
        if len(rows) < WINDOW_MIN_REPORTS:
            over_before = {m: None for m in HEADS}
            continue
        # M7
        oc = cached_reading(_cells(rows, "own", MW.JUDGED_SENSE)) if m7_judged else None
        m7["windows"].append(dict(window=k, steps=[rows[0]["step"], step], reading=oc))
        if m7_judged and oc and oc.get("se") is not None:
            under = not oc["value"] > Z * oc["se"]
            cid = "M7:w%d" % k
            if under:
                why = ("%s: over %s proprioception's error at the hold's first model step was %s lower with the own "
                       "command than with the size-matched command, with a standard error of %s: at or under 3.14 "
                       "times its standard error, the floor of the motor-encoder repair's pass line (over the 20 reports before "
                       "the switch it was %s lower)"
                       % (LABEL, span, _share(oc["value"]), _share(oc["se"]), _share(ref_own["value"])))
                if cid not in taken:
                    out["crossings"].append((step, cid, "M7", why))
                m7.update(state="crossed, its stop already taken" if cid in taken else "crossed", why=why,
                          crossing=cid)
            elif not str(m7["state"]).startswith("crossed"):
                m7["state"] = "held"
        # M8
        entry = dict(window=k, steps=[rows[0]["step"], step], heads={})
        for m in HEADS:
            er, ref = pooled_ratio(_cells(rows, "err", m)), ref_err[m]
            if er is None or ref is None:
                over_before[m] = None
                continue
            rise = er - ref
            over = rise > M8_SPREAD[m]
            entry["heads"][m] = dict(value=er, rise=rise, share_of_spread=rise / M8_SPREAD[m], over=over)
            if over and over_before[m]:
                cid = "M8:w%d:%s" % (k, m)
                why = ("%s: over %s the %s head's audited error (its prediction error as a share of the no-change "
                       "error at the hold's first model step) read %s, and %s over the 1,000 model steps before; over "
                       "the 20 reports before the switch it read %s. Both windows are above that by more than %s of the no-change error, the "
                       "largest rise of one window in the lineage's record with nothing switched"
                       % (LABEL, span, m, _share(er), _share(over_before[m]), _share(ref), _share(M8_SPREAD[m])))
                if cid not in taken:
                    out["crossings"].append((step, cid, "M8", why))
                m8.update(state="crossed, its stop already taken" if cid in taken else "crossed", why=why,
                          crossing=cid)
            over_before[m] = er if over else None
        m8["windows"].append(entry)
        if entry["heads"] and not str(m8["state"]).startswith("crossed"):
            m8["state"] = "held"
    out["lines"]["M7"], out["lines"]["M8"] = m7, m8
    # the gradient alert: the first full report past N + 1,000 at which a judged lesson's gradient reads under the line
    for m in GRADIENT_HEADS:
        for r in full:
            g = _obj(r.get("gradient")).get(m)
            if r["step"] > N + WINDOW_STEPS and _num(g) and g < GRADIENT_ALIVE:
                out["alerts"].append(dict(
                    id="G-alert:%s" % m, line="gradient", step=r["step"], senses=[m],
                    why=("%s ALERT (printed and recorded, never a stop): at model step %s the %s lesson's gradient on "
                         "the transformer's output read %s of its liveness line (one millionth per lesson row and "
                         "number; through the old head alone it read under 0.1%% of that line)"
                         % (LABEL, "{:,}".format(r["step"]), m, _grad_pct(g)))))
                break
    return out


def reference_change(before):
    """{head: the sense's real change per number at the newest full report before the switch that carries it}."""
    ref = {}
    for r in before:
        if not is_full(r):
            continue
        for m, v in _obj(r.get("real_change")).items():
            if _num(v) and v > 0:
                ref[m] = v
    return ref


def output_lines(full, N, taken=None, reference=None):
    """Stop line M6 (a path's output over its sense's real change) and the fixed-vector alert (module text) on the
    full reports after the switch. reference: {head: the sense's real change before the switch}; the real change a
    report is judged against is at least M6_FLOOR of it. Returns dict(line=the M6 entry, crossings=[(model step, id,
    "M6", why)], alerts=[...], warnings=[...]). A crossing whose stop was taken (an id M6@<model step> in `taken`)
    is not judged again: only reports after that model step are."""
    reference = dict(reference or {})
    done = [int(c[3:]) for c in (taken or {}) if str(c).startswith("M6@") and str(c)[3:].isdigit()]
    after = max(done) if done else None
    done9 = [int(c[3:]) for c in (taken or {}) if str(c).startswith("M9@") and str(c)[3:].isdigit()]
    after9 = max(done9) if done9 else None
    pair = []
    out = dict(line=dict(state="waiting", newest=None, largest_share_of_line=None), crossings=[], alerts=[],
               warnings=[])
    top, unread, seen_alert, floored = None, 0, set(), set()
    for r in full:
        heads = _obj(r.get("heads"))
        shares = {}
        changes = {}
        for m in HEADS:
            h = _obj(heads.get(m))
            o, rc = h.get("output_rms"), h.get("real_change")
            floor = M6_FLOOR * reference[m] if _num(reference.get(m)) else None
            if _num(floor) and (not _num(rc) or rc < floor):
                if _num(rc) or not _num(reference.get(m)):
                    floored.add(m)                # the sense's token has all but stopped changing: the floor
                    rc = floor
                else:
                    rc = reference[m]             # the report carries no real change: its value before the switch
            if _num(o) and _num(rc) and rc > 0:
                shares[m] = o / rc
                changes[m] = rc
            elif _num(o):
                unread += 1
            f = _obj(h.get("fixed"))
            if (m not in seen_alert and _num(f.get("path_rms")) and _num(f.get("cosine")) and _num(rc) and rc > 0
                    and f["path_rms"] >= FIXED_ALERT_SHARE * rc and f["cosine"] <= FIXED_ALERT_COSINE):
                seen_alert.add(m)
                if (after9 is None or r["step"] > after9) and not pair:
                    pair.append((r["step"], "M9@%d" % r["step"], "M9", (
                        "%s: at model step %s the %s path adds one fixed vector to every row that is %s of the "
                        "sense's real change per number, and the old head adds a fixed vector pointing the other way "
                        "(cosine between the two %s, where -100%% is exactly opposite): the pair that grew against "
                        "each other in the first segment"
                        % (LABEL, "{:,}".format(r["step"]), m, _pct(f["path_rms"] / rc), _signed(f["cosine"])))))
        if not shares:
            continue
        m = max(shares, key=shares.get)
        h = _obj(heads.get(m))
        out["line"]["newest"] = dict(step=r["step"], head=m, output_rms=h.get("output_rms"),
                                     real_change=changes[m], share_of_line=shares[m] / M6_SHARE,
                                     shares={k: v / M6_SHARE for k, v in shares.items()})
        top = shares[m] if top is None else max(top, shares[m])
        if shares[m] > M6_SHARE and (after is None or r["step"] > after) and not any(
                c[2] == "M6" for c in out["crossings"]):
            out["crossings"].append((r["step"], "M6@%d" % r["step"], "M6", (
                "%s: at model step %s the %s path's output reads %s of one token unit, %s of that sense's real change "
                "per number between two model steps (%s of one token unit); above 100%% this path's output alone is "
                "larger than that change. M6 is a declared guard on the path's size: the old head's output is added to "
                "it and can cancel part of it, and the output is read on the newest lesson rows (up to 64) against a "
                "running mean of the change with a 2,000-model-step horizon"
                % (LABEL, "{:,}".format(r["step"]), m, _pct(h.get("output_rms")), _pct(shares[m]),
                   _pct(changes[m])))))
    out["pair_line"] = dict(state="waiting" if not full else "held")
    if pair:
        out["pair_line"].update(state="crossed", why=pair[0][3], crossing=pair[0][1])
        out["crossings"].append(pair[0])
    out["line"]["largest_share_of_line"] = None if top is None else top / M6_SHARE
    if out["line"]["newest"]:
        out["line"]["state"] = "held"
    first6 = [c for c in out["crossings"] if c[2] == "M6"]
    if first6:
        out["line"].update(state="crossed", why=first6[0][3], crossing=first6[0][1])
    if floored:
        out["warnings"].append("the real change of %s read under %s of its value before the switch in a report after "
                               "the switch: the output line judged that sense against that floor (the sense's token "
                               "has all but stopped changing, which is its encoder's matter)"
                               % (", ".join(sorted(floored)), _pct(M6_FLOOR)))
    if unread:
        out["warnings"].append("%d head reading(s) after the switch carry a path output and no real change of the "
                               "sense (next_frame_norm.persistence_ema): the output line is not judged on them"
                               % unread)
    return out


def _signed(x):
    """A cosine as a signed percentage of full alignment (-100% = exactly opposite)."""
    return "unmeasured" if not _num(x) else "%d%%" % int(round(100.0 * x))


_base_judge = MW.judge


def judge(before, after, switch, N, taken=None, **kw):
    """The base module's judge (M1 to M6) with this watch's M7 and M8: the stop reported is the crossing at the
    lowest model step (the base's own choice among its lines stands; a tie goes to the base's)."""
    why, cid, rep = _base_judge(before, after, switch, N, taken, **kw)
    full = [r for r in after if is_full(r)]
    extra = audited_lines(before, full, int(N), taken)
    rep["lines"].update(extra["lines"])
    rep["alerts"] = list(rep.get("alerts") or []) + extra["alerts"]
    rep["warnings"] = list(rep.get("warnings") or []) + extra["warnings"]
    cands = [(st, 1, c, line, w) for st, c, line, w in extra["crossings"]]
    # M6: the base judges only an output that is not a finite number (its growth line is switched off below); the
    # share of the sense's real change is judged here
    m6 = output_lines(full, int(N), taken, reference=reference_change(before))
    if not str((rep["lines"].get("M6") or {}).get("state", "")).startswith("crossed"):
        rep["lines"]["M6"] = m6["line"]
    rep["lines"]["M9"] = m6["pair_line"]
    rep["alerts"] += m6["alerts"]
    rep["warnings"] += m6["warnings"]
    cands += [(st, 1, c, line, w) for st, c, line, w in m6["crossings"]]
    if why:
        cands.append((rep["stop"]["step"] if _num(rep["stop"].get("step")) else -1, 0, cid, rep["stop"]["line"], why))
    if not cands:
        return None, None, rep
    step, _rank, cid, line, why = min(cands, key=lambda c: (c[0], c[1]))
    rep["stop"] = dict(id=cid, line=line, step=step,
                       owner=next((r.get("run") for r in after if r["step"] == step), None))
    return why, cid, rep


# ------------------------------------------------------------------ the early lines, the late lines, the readings
def acceptance(before, full, N):
    """The early lines E2 and E3, the late pass lines L1 and L2, and per head the watched readings of every 1,000
    model steps with the reference before the switch and the goal reference (module text)."""
    ref_rows = [r for r in before if r["step"] > REFERENCE_AFTER_STEP]
    ref = {m: head_readings(ref_rows, m) for m in HEADS}
    out = dict(reference=ref, reference_reports=len(ref_rows), goal=GOAL,
               reference_steps=[ref_rows[0]["step"], ref_rows[-1]["step"]] if ref_rows else None, windows=[])
    for lo in range(0, LINES["horizon_steps"], WINDOW_STEPS):
        rows, complete = _window(full, N, lo, lo + WINDOW_STEPS)
        out["windows"].append(dict(model_steps=[lo + 1, lo + WINDOW_STEPS], reports=len(rows), complete=complete,
                                   heads=({m: head_readings(rows, m) for m in HEADS} if complete else None)))
    # the early lines
    rows, complete = _window(full, N, *EARLY_WINDOW)
    early = dict(model_steps=[N + EARLY_WINDOW[0] + 1, N + EARLY_WINDOW[1]], reports=len(rows), complete=complete,
                 gradient={}, head_read={})
    rd = {m: head_readings(rows, m) for m in HEADS} if complete else {}
    for m in GRADIENT_HEADS:
        g = (rd.get(m) or {}).get("gradient")
        early["gradient"][m] = dict(value=g, line=GRADIENT_ALIVE, reference=ref[m].get("gradient"),
                                    state="waiting" if not _num(g) else "met" if g >= GRADIENT_ALIVE else "missed")
    for m in HEADS:
        h = (rd.get(m) or {}).get("head_read")
        early["head_read"][m] = dict(value=h, line=P4_LINE, old_pool_spread=(rd.get(m) or {}).get("pool_spread"),
                                     path_slot_reach=(rd.get(m) or {}).get("slot_reach"),
                                     state="waiting" if not _num(h) else "met" if h >= P4_LINE else "missed")
    states = [v["state"] for v in list(early["gradient"].values()) + list(early["head_read"].values())]
    early["state"] = "waiting" if "waiting" in states else "met" if all(x == "met" for x in states) else "missed"
    out["early"] = early
    # the late pass lines
    rows, complete = _window(full, N, *LATE_WINDOW)
    er = error_reading(_cells(rows, "err", "tactile")) if complete else None
    l1 = dict(model_steps=[N + LATE_WINDOW[0] + 1, N + LATE_WINDOW[1]], reading=er, goal=GOAL["tactile"]["error"],
              reference=(ref["tactile"].get("error") or {}).get("value"))
    if not er or er.get("se") is None:
        l1["state"] = "waiting"
    else:
        l1.update(state="met" if er["below"] > Z * er["se"] else "missed", line=1.0 - Z * er["se"])
    late_own = MW.own_command_reading(_cells(rows, "own", "vestibular")) if complete else None
    rows_p, complete_p = _window(full, N, *POOLED_WINDOW)
    oc = MW.own_command_reading(_cells(rows_p, "own", "vestibular")) if complete_p else None
    l2 = dict(model_steps=[N + POOLED_WINDOW[0] + 1, N + POOLED_WINDOW[1]], reading=oc, late_window=late_own,
              goal=GOAL["vestibular"]["own"], reference=(ref["vestibular"].get("own_command") or {}).get("value"))
    judged = _above_noise(oc)
    l2["state"] = "waiting" if (not complete_p or judged is None) else "met" if judged else "missed"
    if oc and oc.get("se") is not None:
        l2["smallest_reading_the_line_can_declare"] = Z * oc["se"]
    out["late"] = dict(L1=l1, L2=l2)
    return out


# ------------------------------------------------------------------ the names replaced on the private instance
MW.SWITCH_TYPE = SWITCH_TYPE
MW.STATE_NAME = STATE_NAME
MW.PATH_VERSION = PATH_VERSION
MW.compact = compact
MW.check_switch = check_switch
MW.acceptance = acceptance
MW.judge = judge
MW.launch_settings = launch_settings
# the base's growth line is switched off on this private instance (no output is above a line of 1e30 token units): the
# base still stops on an output that is not a finite number; the share of the real change is judged in judge() above
MW.growth_line = lambda model_steps, per_step: 1e30
BASE.SWITCH_TYPE = SWITCH_TYPE
BASE.STATE_NAME = STATE_NAME
BASE.compact = compact
StatsReader = MW.StatsReader
READER = StatsReader()
load_state = MW.load_state
find_switch = MW.find_switch
k6_levels = MW.k6_levels


def growth_line(model_steps, per_step):
    """The first design's growth line at `model_steps` after the switch, token units (printed as a reference; not a
    stop line since version 2): GROWTH_PER_UPDATE x the optimizer updates."""
    return None if not _num(per_step) else GROWTH_PER_UPDATE * float(per_step) * float(model_steps)


def _relabel_report(rep):
    for line in (rep.get("lines") or {}).values():
        if isinstance(line, dict) and line.get("why"):
            line["why"] = relabel(line["why"])
    for a in rep.get("alerts") or []:
        a["why"] = relabel(a.get("why"))
    rep["warnings"] = [relabel(w) for w in rep.get("warnings") or []]
    return rep


def evaluate(run, reader=None, write=True, readings=True):
    """The judgement for `run`: None when no switch reaches it, else the base module's dict with this switch's words."""
    ev = MW.evaluate(run, reader or READER, write=write, readings=readings)
    if ev is None:
        return None
    ev["why"] = relabel(ev["why"])
    ev["warnings"] = [relabel(w) for w in ev.get("warnings") or []]
    _relabel_report(ev["judgement"])
    return ev


def _with_alerts(alert_instead_of_stop, fn, *a, **kw):
    """`fn` with the base instance's ALERT_INSTEAD_OF_STOP ({line: why}, motor_asked_change_watch.py) set for this call
    only: a line named there that crosses is an alert for the judged run, never its stop; every other line is judged
    as before. The hourly watcher names M3 for the attention cap experiment's control run 2 and cap arm (the operator's
    decision of Oct 6, 2026)."""
    before = MW.ALERT_INSTEAD_OF_STOP
    MW.ALERT_INSTEAD_OF_STOP = dict(alert_instead_of_stop or {})
    try:
        return fn(*a, **kw)
    finally:
        MW.ALERT_INSTEAD_OF_STOP = before


def breach(run, reader=None, alert_instead_of_stop=None):
    """The hourly watcher's 5-minute check: a stop reason for `run`, or None (the base module's breach with this
    switch's words: the crossing is recorded with the reason the watcher writes). alert_instead_of_stop: {line: why}
    of the lines judged as alerts for this run (_with_alerts)."""
    return _with_alerts(alert_instead_of_stop, _breach, run, reader)


def _breach(run, reader=None):
    found = find_switch(run, reader or READER)
    if found is None or load_state(found[0]).get("closed"):
        return None
    ev = evaluate(run, reader, write=True, readings=False)
    if ev is None:
        return None
    if not ev["watch_over"]:
        for w in ev["warnings"]:
            print("  %s STOP LINES INCOMPLETE: %s" % (LABEL.upper(), w), flush=True)
    for alert in MW.record_alerts(ev["deploy"], run, ev["judgement"].get("alerts") or []):
        print("  " + alert["why"], flush=True)
    if not ev["why"]:
        if ev["watch_over"]:
            BASE._write_state(ev["deploy"], dict(load_state(ev["deploy"]), closed=dict(
                time=time.time(), run=str(Path(run)), newest_step=ev["judgement"].get("newest_step"))))
        return None
    stop = ev["judgement"].get("stop") or {}
    MW.record_crossing(ev["deploy"], dict(id=ev["id"], line=stop.get("line"), reason=ev["why"], run=str(Path(run)),
                                          time=time.time(), step=stop.get("step"), owner=stop.get("owner")))
    return ev["why"]


def _share(x):
    """A 0-to-1 reading as a percentage; a reading under 1 part in 1,000 in whole parts in a million (the dead
    heads read a few parts in a million, which a percentage would print with an exponent)."""
    if not _num(x):
        return "unmeasured"
    if abs(x) < 0.001:
        return "{:,.0f} parts in a million".format(1e6 * x)
    return _pct(x)


def _round_pct(x):
    """A share as a whole percentage (a rise of 1e-16 of a spread prints as 0%, never with an exponent)."""
    return "unmeasured" if not _num(x) else "%d%%" % int(round(100.0 * x))


def _grad_pct(g):
    """A lesson's gradient as a share of its liveness line; under one thousandth of the line it is named so."""
    if not _num(g):
        return "unmeasured"
    share = g / GRADIENT_ALIVE
    return "under 0.1%" if share < 0.001 else _pct(share) if share < 10 else "{:,.0f}%".format(100.0 * share)


def _range(pair):
    return "%s and %s" % (_share(pair[0]), _share(pair[1]))


def _watched_words(m, A):
    """One head's watched readings: the newest complete window of 1,000 model steps, the reference, the goal."""
    done = [w for w in (A.get("windows") or []) if w.get("complete") and w.get("heads")]
    ref, goal = (A.get("reference") or {}).get(m) or {}, GOAL[m]
    if not done:
        now, where = {}, "no window of 1,000 model steps is complete yet"
    else:
        now = done[-1]["heads"][m]
        where = "over model steps {:,} to {:,} after the switch".format(*done[-1]["model_steps"])
    er, oc = now.get("error") or {}, now.get("own_command") or {}
    rer, roc = ref.get("error") or {}, ref.get("own_command") or {}
    return ("  %s, %s head, watched (%s): its error is %s of the no-change error (before the switch %s; the probe's "
            "goal reference %s); its error is %s lower with the own command than with the size-matched command, "
            "standard error %s (before the switch %s; goal reference %s); the swapped command moves its prediction "
            "by %s of the predicted change (before the switch %s)"
            % (LABEL, m, where, _share(er.get("value")), _share(rer.get("value")), _range(goal["error"]),
               _share(oc.get("value")), _share(oc.get("se")), _share(roc.get("value")), _range(goal["own"]),
               _share(now.get("move")), _share(ref.get("move"))))


def lines(run, reader=None, write=True, alert_instead_of_stop=None):
    """The hourly sweep's lines for a person (shares of lines, counts, model steps); [] when no switch reaches `run`.
    alert_instead_of_stop: as breach's."""
    return _with_alerts(alert_instead_of_stop, _lines, run, reader, write)


def _lines(run, reader=None, write=True):
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
    out = ["  {}: switched on at model step {:,} (--heads-all-slot-path on); {} full reports since, newest model step "
           "{}; the stop lines judge through the first full report at or past model step {:,}{}"
           .format(LABEL, N, rep.get("after_full_reports"), "none" if newest is None else "{:,}".format(newest),
                   N + LINES["horizon_steps"], " (reached: the watch is over)" if ev["watch_over"] else "")]
    for w in ev["warnings"]:
        out.append("  %s STOP LINES INCOMPLETE: %s" % (LABEL.upper(), w))
    out.append("  %s, the switch record (every head's prediction with the paths against without them on 64 probe rows "
               "must be the same number): %s; non-finite updates (the stats' count, and the trainer's log lines for a "
               "skipped optimizer update: %d so far): %s"
               % (LABEL, state("M1"), (L.get("M2") or {}).get("skipped_updates_in_the_logs") or 0, state("M2")))
    m3 = L.get("M3") or {}
    out.append("  %s, the hidden layers: %s; %s (stop line %s in %d full reports with a reading running; the reading "
               "is the lowest over the five paths of the whole-model monitor's and the path's own)"
               % (LABEL, state("M3"), "no full report read yet" if not _num(m3.get("lowest_live_share")) else
                  "at its lowest %s of a path's hidden units responded" % _pct(m3["lowest_live_share"]),
                  _pct(LINES["M3_liveness"]), LINES["M3_consecutive"]))
    m6 = (L.get("M6") or {}).get("newest") or {}
    out.append("  %s, each path's output as a share of its sense's real change per number between two model steps "
               "(0%% = the path adds nothing, 100%% = it adds as much as the whole change it predicts; above 100%% "
               "stops the run): %s; %s"
               % (LABEL, state("M6"), "no report carries it yet" if not _num(m6.get("output_rms")) else
                  "at model step {:,} the largest is the {} path's, {} of one token unit, {} of that sense's real "
                  "change; all five: {}".format(
                      m6["step"], m6.get("head"), _pct(m6["output_rms"]), _pct(m6.get("share_of_line")),
                      ", ".join("%s %s" % (k, _pct(v)) for k, v in (m6.get("shares") or {}).items()))))
    out.append("  %s, a path's fixed vector against the old head's (a fixed vector of half the sense's real change or "
               "more that points against the old head's stops the run): %s" % (LABEL, state("M9")))
    out.append("  %s, the senses in reports 2 and 3 after the switch (each sense's next-model-step error ratio as a "
               "share of its line; over 100%% in both stops the run): %s; %s"
               % (LABEL, state("M4_early"), shares("M4_early")))
    m4 = L.get("M4") or {}
    out.append("  %s, the senses against their level before the switch, every 20 reports (over 100%% of the stop line "
               "stops the run; over 100%% of the alert line is an alert, never a stop): %s; the newest window (number "
               "%s), of the stop line: %s; of the alert line: %s"
               % (LABEL, state("M4"), "none" if m4.get("window") is None else m4["window"] + 1, shares("M4"),
                  shares("M4", "alert_shares")))
    alerts = rep.get("alerts") or []
    out.append("  %s, alerts (never a stop): %s" % (LABEL, "none so far" if not alerts else "%d, the newest: %s"
                                                    % (len(alerts), alerts[-1]["why"])))
    m5 = L.get("M5") or {}
    out.append("  %s, speed: %s; in the newest window the trainer ran at %s of its model steps per second in %s "
               "(under %s in two windows running stops the run)"
               % (LABEL, state("M5"), _pct(m5.get("share")),
                  "the 20 reports before the switch" if m5.get("window") == 0 else "the window before it",
                  _pct(LINES["M5_share"])))
    m7 = L.get("M7") or {}
    w7 = (m7.get("windows") or [{}])[-1].get("reading") or {}
    out.append("  %s, proprioception's own-command reading in every 1,000 model steps (at or under 3.14 times its standard "
               "error stops the run): %s; %s"
               % (LABEL, state("M7"), "no window is complete yet" if not w7 else
                  "in the newest window its error was %s lower with the own command, standard error %s"
                  % (_share(w7.get("value")), _share(w7.get("se")))))
    m8 = L.get("M8") or {}
    w8 = (m8.get("windows") or [{}])[-1].get("heads") or {}
    out.append("  %s, every head's audited error against the 20 reports before the switch (a rise past 100%% of the "
               "head's measured spread in two windows of 1,000 model steps running stops the run): %s; the newest "
               "window's rise as a share of the spread: %s"
               % (LABEL, state("M8"), ", ".join("%s %s" % (m, _round_pct(v.get("share_of_spread"))) for m, v in w8.items())
                  or "no window is complete yet"))
    A = rep.get("acceptance") or {}
    E, late = A.get("early") or {}, A.get("late") or {}
    if E:
        out.append("  {}, early lines, never a stop, over model steps {:,} to {:,}: {}. The switch record: {}. The "
                   "lessons' gradient on the transformer's output, a liveness reading (100% = one millionth per lesson "
                   "row and number, the liveness line; before the switch through the old heads under 0.1%): {}. Every "
                   "head's whole prediction answers to more than one slot (0% = one slot, 100% = all six evenly; line "
                   "{}): {}".format(
                       LABEL, E["model_steps"][0], E["model_steps"][1], E.get("state", "waiting"), state("M1"),
                       ", ".join("%s %s (%s)" % (m, _grad_pct(v.get("value")), v.get("state"))
                                 for m, v in (E.get("gradient") or {}).items()),
                       _pct(P4_LINE),
                       ", ".join("%s %s (%s; its old attention pool %s)"
                                 % (m, _share(v.get("value")), v.get("state"), _share(v.get("old_pool_spread")))
                                 for m, v in (E.get("head_read") or {}).items())))
    l1, l2 = late.get("L1") or {}, late.get("L2") or {}
    if l1:
        r1 = l1.get("reading") or {}
        out.append("  {}, late pass line for tactile, never a stop, over model steps {:,} to {:,}: {}. Its error must "
                   "read below 100% of the no-change error by more than 3.14 times its standard error; it reads {} with a "
                   "standard error of {} (before the switch {}; the probe's goal reference {})".format(
                       LABEL, l1["model_steps"][0], l1["model_steps"][1], l1.get("state", "waiting"),
                       _share(r1.get("value")), _share(r1.get("se")), _share(l1.get("reference")),
                       _range(GOAL["tactile"]["error"])))
    if l2:
        r2, rl = l2.get("reading") or {}, l2.get("late_window") or {}
        out.append("  {}, late pass line for vestibular, never a stop, pooled over model steps {:,} to {:,}: {}. Its "
                   "error must be lower with the own command than with the size-matched command by more than 3.14 times "
                   "its standard error; it reads {} lower with a standard error of {} (over the last 1,000 model "
                   "steps {}; before the switch {}; the probe's goal reference {})".format(
                       LABEL, l2["model_steps"][0], l2["model_steps"][1], l2.get("state", "waiting"),
                       _share(r2.get("value")), _share(r2.get("se")), _share(rl.get("value")),
                       _share(l2.get("reference")), _range(GOAL["vestibular"]["own"])))
    if A:
        out.append("  {}, the reference of every watched reading: the {} full reports before the switch above model "
                   "step {:,}".format(LABEL, A.get("reference_reports") or 0, REFERENCE_AFTER_STEP))
        for m in HEADS:
            out.append(_watched_words(m, A))
    fr = rep.get("readings") or {}
    first = fr.get("first_reports") or []
    if first:
        out.append("  %s, the first full reports after the switch: " % LABEL + "; ".join(
            "model step {:,}: {} of the step rate of the 3 reports before the switch, graphics card {} of {} MiB, the "
            "largest path output {} of a token unit, the hidden layers {} live at the lowest, {} non-finite "
            "updates".format(
                f["step"], _pct(f.get("steps_per_s_share")),
                "unmeasured" if not _num(f.get("gpu_used_mib")) else "{:,.0f}".format(f["gpu_used_mib"]),
                "unmeasured" if not _num(f.get("gpu_total_mib")) else "{:,.0f}".format(f["gpu_total_mib"]),
                _pct(f.get("output_rms")), _pct(f.get("path_live")),
                f.get("nan_steps")) for f in first))
    ls = fr.get("launch_settings") or {}
    if ls.get("switch") is None:
        own = "the switched run's args.json or its parent's could not be read"
    elif ls.get("besides_the_flag"):
        own = ("the switch's relaunch changed %s besides the flag - MORE THAN THE PATHS CHANGED AT THE SWITCH: a stop "
               "on the senses or on the step rate cannot be put down to the paths alone"
               % ", ".join(ls["besides_the_flag"]))
    else:
        own = "the switch's relaunch changed --heads-all-slot-path and nothing else"
    inside = len(ls.get("within_the_20_before") or [])
    out.append("  %s, the launch settings around the switch (never judged): %s; %d relaunch(es) that changed a setting "
               "lie inside the 20 full reports before the switch, %d after it inside the watch"
               % (LABEL, own, inside, len(ls.get("after") or [])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Print the all-slot paths' stop lines and pass lines for a run (reading "
                                             "only: nothing is written).")
    ap.add_argument("run")
    a = ap.parse_args(argv)
    ev = evaluate(a.run, StatsReader(), write=False)
    if ev is None:
        print("no %s record reaches %s" % (SWITCH_TYPE, a.run))
        return 0
    for line in lines(a.run, StatsReader(), write=False):
        print(line)
    if ev["why"]:
        print("A STOP LINE IS CROSSED (the hourly watcher writes the stop): " + ev["why"])
    return 1 if ev["why"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
