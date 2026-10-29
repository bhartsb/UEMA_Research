"""The stop lines and pass lines of the vestibular input path's switch in production (--vestibular-input-path on),
judged by the H84 hourly watcher. Plan and arithmetic: docs/gain_of_function_v4/agents/vestibular_token_2026-10-04.md;
registry row vestibular_input_path_production_segment.

WHAT THE SWITCH IS. Production's command gets --vestibular-input-path on at a relaunch. The load then builds the path
beside the vestibular encoder (lem/model/vestibular_input_path.py: one Linear(28 -> 192) without bias from the
encoder's own input row, its weight zero at the start, added to the encoder's last-layer output in units of that
output's standard deviation) and the trainer writes one vestibular_input_path_switch record into the stats stream.

HOW IT IS JUDGED. As the asked-change path's switch is (motor_asked_change_watch.py, whose code this module runs
wherever the arithmetic is the same): BASE is a private instance of that file, loaded by path under another module
name, with its switch record type, its state file's name, its record reader's field list, its check of the switch
record and its growth constant replaced on that instance only. Its timeline (the walk back to the switch through the
run folders, the up to 100 full reports before it, every record after it through the first full report at or past
10,000 model steps), its 20-report windows, its level before the switch, its stop and alert multiples per sense, its
reader and its bookkeeping of crossings whose stop was taken are then this module's, unchanged. The hourly watcher's
asked-change rule and vestibular repair rule are other instances of their files and read their own names.

By hand, reading only (nothing is written):
    python docs/gain_of_function_v4/vestibular_input_path_watch.py <run folder>

THE STOP LINES (each writes snapshot_and_stop through the hourly watcher's own stop):
  M1  the switch record is as declared: it says exact, every precision it checked (float32, and the run's autocast
      when the run is on the graphics card) reads a largest change of the vestibular token of exactly 0.0 on
      bit-identical tokens, the path's largest absolute weight is 0.0, the check ran on 64 probe rows, and the version
      is this build's. Judged at once.
  M2  a non-finite update after the switch: nan_steps above 0 in any stats record, or a line of a trainer.log that
      reports a skipped optimizer update (the asked-change watch's M2, unchanged).
  M4  the other senses' next-model-step error ratio (prediction error over the error of predicting no change; 0 =
      exact, 1 = no better than no change) for vision, tactile, audio and proprioception: the same sense above its
      median over the 20 reports before the switch x W1 in report 2 and in report 3; and in every window of 20 full
      reports a sense's median above its level before the switch x RISE x 1.2 (3.95 times the level for vision, 1.72
      for the other three). Above the level x RISE it is an ALERT, printed once and recorded, never a stop. The
      multiples are the vestibular repair watch's, measured on the lineage with nothing switched (its false stops out
      of sample: 0 of 206 and 0 of 234 watches). Vestibular's own ratio is an alert only (the path changes what its
      head has to predict).
  M5  the step rate: the median model steps per second of a 20-report window under 0.9 of the reference's in two
      windows running (the asked-change watch's M5, unchanged).
  M6  the path's output against its growth bound: at a full report, vestibular_input_path.output_rms (root mean
      square per number of the path's output over the report's encoder calls, in token units) above
      GROWTH_PER_MODEL_STEP x the model steps since the switch. GROWTH_PER_MODEL_STEP = the learning rate 3e-4 x the
      summed input size SUMMED_INPUT x the path's gain 1 / sqrt(28): the most one optimizer update can add on one row
      when Adam moves every weight by the learning rate in the direction of its input on that row. The path's weight
      gets one update per model step (the live lessons; replay encodes no vestibular row). An output that is not a
      finite number is a crossing too. An ALERT, never a stop: the output at or above 1 token unit.
      THE OUTPUT'S OWN BOUND is in the path, not in this watch: per row the path scales its output back to
      OUTPUT_CAP (0.3 token units, root mean square over the row's 192 numbers), so no row is ever larger
      (lem/model/vestibular_input_path.py; the report carries the cap as vestibular_input_path.output_cap). The share
      of a report's rows the cap scaled down (vestibular_input_path.capped_share) is printed beside the output, never
      judged: 0 = the path is the linear map, 1 = every row sits at the cap. Once every row sits at the cap the
      output reads the cap itself and stays there; the growth line is far above it by then (the line passes 0.3 at
      232 model steps after the switch).
      THE TWO PARTS: the token is LayerNorm(v / s + u), v / s the encoder's own part (its last-layer output over its
      standard deviation s) and u the path's part. Under the flag value `on`, the wiring this text describes, the
      previous row's vestibular token enters the delayed lesson without gradient, so that lesson trains neither the
      encoder's own layers nor the path's weight, and no lesson trains one part and not the other (the module's text,
      WHICH LESSONS TRAIN WHICH PART); the live row's VICReg and reconstruction decoder terms train both. The report
      carries vestibular_input_path.change_slope (-1 = the encoder's part takes the path's change out of the sum, 0 =
      unrelated, +1 = it doubles the path's change), printed as a reading, never judged: on the encoder bench of path
      version 2 (row vestibular_token_path_bench, bench.json arm path_none_cap03, 60,000 updates) `on` read -0.015 to
      +0.198 at every reading from 2,000 updates on, while head tilt stayed readable from the token (0.434 to 0.535 on
      the 5 reports no update read, from 4,000 updates on; line 0.25). The exception `on_delayed_encoder_only`, where
      the delayed lesson trains the encoder's own layers and not the path's weight, is read under S1 below. The line
      that judges the outcome is T1 (and T1 late).
      MEASURED GROWTH, for scale (row vestibular_token_path_live_bench, Oct 4, 2026: checkpoint 479,350's whole
      model updated on its own ring on the graphics card, the path's weight updated once per update as in
      production, the cap not reached): the output read 0.013 token units after 100 updates, 0.031 after 500, 0.054
      after 1,000, 0.13 after 2,000 and 0.22 after 3,000: 4 to 10 in 100 of this line at every reading.
  The asked-change watch's M3 (a hidden layer's live share) has no counterpart: the path has no hidden layer.
  WHAT M6 IS AND IS NOT (the independent review of Oct 4, 2026, change 5): the growth line rises by 0.0013 token units
  per model step and the path's cap is 0.3, so the line can be crossed in the first 232 model steps after the switch
  only; from there on a finite output cannot reach it, and what M6 still stops is an output that is not a finite
  number. It is a line on the first reports' growth. It is no bound on the path and the encoder's part together: the
  standing lines below read those.

THE STANDING LINES (the review's changes 2 and 4; ALERTS, printed once and recorded, never a stop). Judged on every
complete window of 1,000 model steps from N + 2,001 on, for as long as the run's flag is on: past the watch's end
too, and on a run so far past the switch that the watch's walk no longer reaches it (then on the model step count's
own thousands, from the run's reports and those of the runs it continues).
  T1 standing  the target, the same reading and the same line as T1: the window's median head-tilt reading of the
      trainer's probe under 0.25, or less than 0.25 above its body-shuffled control's median, in two windows running.
  S1  the change slope (vestibular_input_path.change_slope per report: -1 = the encoder's own part takes the path's
      change out of the sum, 0 = unrelated): the window's median under SLOPE_LINE in two windows running. SLOPE_LINE
      = -0.10: on the encoder bench of path version 2 the wiring in which the delayed lesson trains neither part (the
      flag value on) read -0.015 to +0.198 at every reading from 2,000 to 60,000 updates, and the wiring in which the
      delayed lesson trains the encoder alone (on_delayed_encoder_only) fell through -0.10 near 10,000 updates on
      its way to -0.34 at 40,000 and -0.46 at 60,000. Under -0.10 the encoder's part has begun to take the path's
      change out: under `on` that is outside what any bench read; under the exception it is its measured cost
      arriving (the independent review's long bench read its cancelling levelling at 70 to 83 of 100).
  W1  the path's weight size (vestibular_input_path.output_weight_rms): the window's median above WEIGHT_LINE in two
      windows running. WEIGHT_LINE = 0.2: with the weight held at the cap (path version 2) the encoder bench read
      0.105 at most through 60,000 updates in both wirings; without the hold the review measured 0.20 where rows
      first reached the cap and 3.39 at 60,000. Above 0.2 the hold is not doing what it was benched to do.
  An alert names the two windows, the reading, and the two other standing readings beside it.

THE PASS LINES (printed by the hourly sweep, never a stop):
  T1  THE TARGET: the token carries head tilt. The trainer's own probe (stats block vestibular.repair.balance_probe:
      a ridge regression from each body's token change to that body's change of head tilt within a report of 100
      model steps, fitted on the even-numbered bodies and scored on the odd ones and the other way round, over the
      last 10 reports; 1 = every scored change predicted, 0 = no better than predicting no change). MET when, over the
      full reports of model steps N + T1_WINDOW[0] + 1 to N + T1_WINDOW[1], the median head-tilt reading is at least
      0.25 AND at least 0.25 above the median of its body-shuffled control. Read again over the watch's last 1,000
      model steps (T1_late), and per 1,000 model steps as a reading. The trunk's roll is printed beside it.
  T2  vestibular's prediction error is below the error of predicting no change: over the 20 full reports ending at
      the watch's end, the 5th percentile of 2,000 bootstrap medians of vestibular.improvement_over_persistence, drawn
      in blocks of 5 consecutive reports, above 0 (the vestibular repair watch's reading A5, its arithmetic).
  Readings beside them, never judged: the path's output size and each input field group's reach, the token's spread
  between the bodies and its change between model steps, vestibular's own error ratio.
Tests: Source/Python/mimo_bridge/tests/test_vestibular_input_path_watch.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import random
import statistics
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_base():
    spec = importlib.util.spec_from_file_location("motor_asked_change_watch_as_vestibular_input_path_base",
                                                  HERE / "motor_asked_change_watch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE = _load_base()                               # a private instance of the asked-change watch
VEST = BASE.BASE                                  # and its own private instance of the vestibular repair watch
REPORT_EVERY = BASE.REPORT_EVERY
SWITCH_TYPE = "vestibular_input_path_switch"
STATE_NAME = "vestibular_input_path_watch.json"
PATH_VERSION = "vestibular_input_path_v2"         # lem/model/vestibular_input_path.py INPUT_PATH_VERSION
OTHER_SENSES = BASE.OTHER_SENSES
PROBE_LABELS = ("head_tilt", "trunk_roll")

# M6's constants (module note). SUMMED_INPUT: the sum over the 28 input fields of their absolute values, mean over the
# replayed rows of checkpoint 459,961 normalised with checkpoint 479,350's statistics (vt_bench.py, bench.json
# summed_input_size); the path's gain is 1 / sqrt(28).
LEARNING_RATE = 3e-4
SUMMED_INPUT = 22.8                               # bench.json summed_input_size.mean (22.84; 99th percentile 42.5)
PATH_GAIN = 1.0 / math.sqrt(28.0)
GROWTH_PER_MODEL_STEP = LEARNING_RATE * SUMMED_INPUT * PATH_GAIN
# T1's judged window after the switch (model steps), from the bench's reachability arithmetic (the record's section
# on the pass line), and the probe's line and margin (the vestibular repair watch's A4)
T1_WINDOW = (2000, 3000)
T1_LATE_WINDOW = (9000, 10000)
T1_R2 = 0.25
T1_MARGIN = 0.25
# the standing lines (module note): the window, the change slope's line, the weight size's line, the path's cap
STANDING_WINDOW = 1000
SLOPE_LINE = -0.10
WEIGHT_LINE = 0.2
OUTPUT_CAP = 0.3                                  # lem/model/vestibular_input_path.py OUTPUT_CAP
M6_REACH_STEPS = int(OUTPUT_CAP / GROWTH_PER_MODEL_STEP)     # 232: past it the growth line is above the cap
T2_DRAWS, T2_BLOCK = 2000, 5

LINES = BASE.LINES
LINES.update(growth_per_optimizer_update=GROWTH_PER_MODEL_STEP, learning_rate=LEARNING_RATE,
             summed_input_size=SUMMED_INPUT, T1_window=T1_WINDOW, T1_late_window=T1_LATE_WINDOW, T1_r2=T1_R2,
             T1_margin=T1_MARGIN)

_num, _median, _pct, _read_json = BASE._num, BASE._median, BASE._pct, BASE._read_json
_obj = BASE._obj
is_full = BASE.is_full


def compact(r):
    """The fields this watch's lines read, from a stats record (None for a record no line reads). The keys the
    asked-change watch's judgement reads keep their names: `path` holds this path's output."""
    t = r.get("type")
    if t == SWITCH_TYPE:
        return {"type": t, "step": r.get("step"), "timestamp": r.get("timestamp"), "switch": r.get("switch")}
    if t != "mimo":
        return None
    nf = _obj(r.get("next_frame_norm"))
    vip = r.get("vestibular_input_path") if isinstance(r.get("vestibular_input_path"), dict) else None
    ve = _obj(r.get("vestibular"))
    rep = _obj(ve.get("repair"))
    mach = _obj(r.get("machine"))
    path = None
    if vip is not None:
        path = {"rows": vip.get("encoder_calls"), "output_rms": vip.get("output_rms"),
                "capped_share": vip.get("capped_share"), "output_cap": vip.get("output_cap"),
                "change_slope": vip.get("change_slope"), "own_change": vip.get("own_change"),
                "path_change": vip.get("path_change"), "wiring": vip.get("wiring"),
                "weight_held_share": vip.get("weight_held_share"),
                "output_weight_rms": vip.get("output_weight_rms"), "encoder_output_sd": vip.get("encoder_output_sd"),
                "field_reach": _obj(vip.get("field_reach")) or None}
    return {"type": "mimo", "step": r.get("step"), "timestamp": r.get("timestamp"), "nf_steps": nf.get("steps"),
            "ratio": {s: _obj(nf.get("ratio")).get(s) for s in OTHER_SENSES + BASE.ALERT_ONLY_SENSES},
            "nan_steps": r.get("nan_steps"), "sps": r.get("model_steps_per_s"), "path": path,
            # the asked-change watch's hidden-layer line reads these two; this path has no hidden layer
            "path_live": None, "path_live_low": None, "s1": [],
            "probe": rep.get("balance_probe") if isinstance(rep.get("balance_probe"), dict) else None,
            "improvement": ve.get("improvement_over_persistence"),
            "between": rep.get("token_between_bodies"), "change": rep.get("token_change"),
            "gpu_used_mib": mach.get("gpu_used_mib"), "gpu_total_mib": mach.get("gpu_total_mib")}


def check_switch(sw, on_card=False):
    """None when the switch record is as declared (module note, M1), else why not (in words)."""
    sw = sw if isinstance(sw, dict) else {}
    checks = {k: v for k, v in sw.items() if isinstance(v, dict) and "change_max_abs" in v}
    if sw.get("exact") is not True:
        return "the switch record does not say that the vestibular token with the path equals the token without it"
    if not checks:
        return "the switch record holds no check of the vestibular token"
    bad = sorted(k for k, v in checks.items() if not (v.get("change_max_abs") == 0.0 and v.get("bit_identical") is True))
    if bad:
        return ("the vestibular token with the path differs from the token without it on the probe rows in %s"
                % ", ".join(bad))
    if on_card and "autocast" not in checks:
        return "the token check did not run under the run's autocast, the precision the graphics card computes in"
    if sw.get("output_weight_max_abs") != 0.0:
        return "the path's weight was not zero at the load"
    if sw.get("rows") != 64:
        return "the token check ran on %s probe rows, not 64" % sw.get("rows")
    if sw.get("version") != PATH_VERSION:
        return "the switch record names path version %s, not %s" % (sw.get("version"), PATH_VERSION)
    return None


# this switch's names on the private instances (module note)
for _m in (BASE, VEST):
    _m.SWITCH_TYPE = SWITCH_TYPE
    _m.STATE_NAME = STATE_NAME
    _m.compact = compact
BASE.check_switch = check_switch
BASE.PATH_VERSION = PATH_VERSION

READER = BASE.StatsReader()                       # the hourly watcher's process keeps it between checks
load_state = BASE.load_state


def _reword(text):
    """The asked-change watch's sentences with this path's names."""
    if not isinstance(text, str):
        return text
    for a, b in (("asked-change path", "vestibular input path"), ("ASKED-CHANGE PATH", "VESTIBULAR INPUT PATH"),
                 ("motor encoder's own standardised output", "vestibular encoder's own standardised output"),
                 ("the motor_asked_change block", "the vestibular_input_path block"),
                 ("every weight of the output layer", "every weight of the path"),
                 ("optimizer updates since the switch", "model steps since the switch (one update of the path's "
                                                        "weight each)")):
        text = text.replace(a, b)
    return text


def _own_warnings(warnings):
    """The asked-change watch's warnings without the two about its hidden layer's live share (this path has no hidden
    layer), and with that layer taken out of the lists the others name."""
    out = []
    for w in warnings:
        if "live share of the path's hidden layer" in w:
            continue
        out.append(_reword(w.replace("the path's hidden layer, ", "")))
    return out


# ------------------------------------------------------------------ the pass lines
def _probe_rows(rows):
    return [r["probe"] for r in rows if isinstance(r.get("probe"), dict) and isinstance(r["probe"].get("r2"), dict)
            and isinstance(r["probe"].get("r2_shuffled"), dict)]


def probe_reading(rows):
    """Per label over the reports' probe readings: the median R squared, the median of its body-shuffled control, the
    median margin between them and the median turn within a report (degrees); None without a reading."""
    probes = _probe_rows(rows)
    if not probes:
        return None
    out = {"reports": len(probes)}
    for lab in PROBE_LABELS:
        r2 = _median(p["r2"].get(lab) for p in probes)
        ctl = _median(p["r2_shuffled"].get(lab) for p in probes)
        margin = _median((p["r2"].get(lab) - p["r2_shuffled"].get(lab)) for p in probes
                         if _num(p["r2"].get(lab)) and _num(p["r2_shuffled"].get(lab)))
        turn = _median((p.get("within_body_turn_deg") or {}).get(lab) for p in probes)
        out[lab] = dict(r2=r2, control=ctl, margin=margin, within_body_turn_deg=turn)
    return out


def t1_judgement(reading):
    """Line T1 from a probe_reading: met when head tilt's median R squared is at least T1_R2 and its median margin
    over the control at least T1_MARGIN."""
    if not reading:
        return dict(state="waiting")
    h = reading["head_tilt"]
    ok = _num(h.get("r2")) and _num(h.get("margin")) and h["r2"] >= T1_R2 and h["margin"] >= T1_MARGIN
    return dict(state="met" if ok else "missed", reading=reading, line=T1_R2, margin_line=T1_MARGIN)


def improvement_p5(vals, draws=T2_DRAWS, block=T2_BLOCK, seed=0):
    """The 5th percentile of `draws` bootstrap medians of `vals`, drawn in blocks of `block` consecutive values (the
    vestibular repair watch's A5)."""
    rng = random.Random(seed)
    blocks = [vals[i:i + block] for i in range(0, len(vals) - block + 1)]
    meds = []
    for _ in range(int(draws)):
        draw = []
        while len(draw) < len(vals):
            draw.extend(rng.choice(blocks))
        meds.append(statistics.median(draw[:len(vals)]))
    meds.sort()
    return meds[int(0.05 * (len(meds) - 1))]


def acceptance(before, full, N):
    """T1 with its windows and its late reading, T2, and the readings beside them (module note)."""
    out = dict(before=probe_reading(before[-LINES["before_reports"]:]), windows=[])

    def window(lo, hi):
        rows = [r for r in full if N + lo < r["step"] <= N + hi]
        last_due = ((N + hi) // REPORT_EVERY) * REPORT_EVERY
        return rows, bool(rows) and any(r["step"] >= last_due for r in full)
    for lo in range(0, LINES["horizon_steps"], 1000):
        rows, complete = window(lo, lo + 1000)
        out["windows"].append(dict(model_steps=[lo + 1, lo + 1000], reports=len(rows), complete=complete,
                                   reading=probe_reading(rows) if complete else None))
    for key, (lo, hi) in (("T1", T1_WINDOW), ("T1_late", T1_LATE_WINDOW)):
        rows, complete = window(lo, hi)
        j = t1_judgement(probe_reading(rows) if complete else None)
        j.update(model_steps=[N + lo + 1, N + hi], reports=len(rows))
        if j["state"] == "waiting" and complete:
            j.update(state="unmeasured", why="the window's reports carry no reading of the trainer's probe (the probe is in the "
                                             "stats only with --vestibular-encoder-repair on)")
        out[key] = j
    end_step = N + LINES["horizon_steps"]
    end = [i for i, r in enumerate(full) if r["step"] >= end_step]
    n = LINES["window_reports"]
    win = full[end[0] + 1 - n:end[0] + 1] if end and end[0] + 1 >= n else None
    if win:
        vals = [r.get("improvement") for r in win if _num(r.get("improvement"))]
        if len(vals) >= n // 2:
            p5 = improvement_p5(vals)
            out["T2"] = dict(state="met" if p5 > 0 else "missed", p5_of_median=p5, median=statistics.median(vals),
                             reports=len(vals), block=T2_BLOCK, steps=[win[0]["step"], win[-1]["step"]])
        else:
            out["T2"] = dict(state="unmeasured", why="%d of %d reports carry vestibular's prediction reading"
                             % (len(vals), len(win)))
    else:
        out["T2"] = dict(state="waiting")
    newest = full[-10:]
    out["readings"] = dict(
        reports=len(newest),
        improvement=_median(r.get("improvement") for r in newest),
        improvement_before=_median(r.get("improvement") for r in before[-LINES["before_reports"]:]),
        token_between_bodies=_median(r.get("between") for r in newest),
        token_between_bodies_before=_median(r.get("between") for r in before[-LINES["before_reports"]:]),
        token_change=_median(r.get("change") for r in newest),
        token_change_before=_median(r.get("change") for r in before[-LINES["before_reports"]:]),
        vestibular_ratio=_median(r["ratio"].get("vestibular") for r in newest),
        vestibular_ratio_before=_median(r["ratio"].get("vestibular") for r in before[-LINES["before_reports"]:]),
        field_reach=next((r["path"].get("field_reach") for r in reversed(full)
                          if isinstance(r.get("path"), dict) and r["path"].get("field_reach")), None))
    return out


# ------------------------------------------------------------------ the standing lines
def _flag_on(run):
    """True when the run's args.json holds --vestibular-input-path at one of its on values."""
    args = _read_json(Path(run) / "args.json")
    value = args.get("vestibular_input_path") if isinstance(args, dict) else None
    return isinstance(value, str) and value not in ("", "off")


def standing_windows(full, base=None):
    """The complete windows of STANDING_WINDOW model steps over the full reports `full` (model step sequence), counted
    from model step `base` (N + 2,000 when the switch is known; None: on the model step count's own thousands). Per
    window: its model steps, its reports, T1's judgement on the probe's readings, and the medians of the change
    slope and of the path's weight size. A window is complete when a full report at or past its last model step
    exists and it holds at least half of its reports."""
    if not full:
        return []
    start = 0 if base is None else int(base)
    groups = {}
    for r in full:
        if r["step"] > start:
            groups.setdefault((r["step"] - start - 1) // STANDING_WINDOW, []).append(r)
    newest = full[-1]["step"]
    out = []
    for k in sorted(groups):
        lo = start + k * STANDING_WINDOW
        hi = lo + STANDING_WINDOW
        rows = groups[k]
        if newest < hi or len(rows) < STANDING_WINDOW // REPORT_EVERY // 2:
            continue
        reading = probe_reading(rows)
        j = t1_judgement(reading)
        head = (reading or {}).get("head_tilt") or {}
        paths = [r["path"] for r in rows if isinstance(r.get("path"), dict)]
        out.append(dict(index=k, model_steps=[lo + 1, hi], reports=len(rows),
                        t1="unmeasured" if j["state"] == "waiting" else j["state"],
                        head_tilt=head.get("r2"), control=head.get("control"),
                        slope=_median(p.get("change_slope") for p in paths),
                        weight=_median(p.get("output_weight_rms") for p in paths),
                        wiring=next((p.get("wiring") for p in reversed(paths) if p.get("wiring")), None)))
    return out


def _span(a, b):
    return "model steps {:,} to {:,} and {:,} to {:,}".format(a["model_steps"][0], a["model_steps"][1],
                                                            b["model_steps"][0], b["model_steps"][1])


def _beside(w):
    """The three standing readings of one window, in words."""
    return ("head tilt %s (its line %s), the change slope %s (its line %s), the path's weight size %s of its line"
            % (_r2(w.get("head_tilt")), _r2(T1_R2), _pct(w.get("slope")), _pct(SLOPE_LINE),
               _pct(w["weight"] / WEIGHT_LINE) if _num(w.get("weight")) else "unmeasured"))


def standing_alerts(windows):
    """One alert per run of windows across a standing line, raised at its second window (module note)."""
    tests = (
        ("T1-standing", "T1", lambda w: w["t1"] == "missed",
         "head tilt read from the vestibular token by the trainer's probe is under its line (the reading at least %s, "
         "and at least %s above its body-shuffled control; 0%% = no better than predicting no change, 100%% = exact): "
         "THE TARGET IS MISSED WHILE THE PATH IS ON" % (_r2(T1_R2), _r2(T1_MARGIN))),
        ("S1", "S1", lambda w: _num(w.get("slope")) and w["slope"] < SLOPE_LINE,
         "the encoder's own part works against the path's part: the change slope is under its line of %s (-100%% = the "
         "encoder's part takes the path's change out of the sum, 0%% = the two changes are unrelated)"
         % _pct(SLOPE_LINE)),
        ("W1", "W1", lambda w: _num(w.get("weight")) and w["weight"] > WEIGHT_LINE,
         "the path's weight size is above its line (the bench's largest size with the weight held at the cap was 53%% "
         "of the line): the hold is not doing what it was benched to do"),
    )
    alerts = []
    for tag, line, crossed, what in tests:
        run_length = 0
        for prev, w in zip([None] + windows[:-1], windows):
            adjacent = prev is not None and w["index"] == prev["index"] + 1
            run_length = (run_length + 1 if adjacent else 1) if crossed(w) else 0
            if run_length == 2:
                alerts.append(dict(
                    id="%s@%d" % (tag, w["model_steps"][1]), line=line, step=w["model_steps"][1],
                    why=("vestibular input path ALERT (printed and recorded, never a stop; a standing line, two "
                         "windows of 1,000 model steps running): %s. Over %s the windows read: %s; then %s"
                         % (what, _span(prev, w), _beside(prev), _beside(w)))))
    return alerts


def standing(run, reader=None):
    """The standing lines for `run`: None when no switch reaches it and its flag is off, else a dict (home: the run
    folder whose state file holds the alerts, N, windows, alerts)."""
    reader = reader or READER
    run = Path(run)
    found = BASE.find_switch(run, reader)
    if found is not None:
        deploy, sw, chain = found
        N = int(sw["step"])
        rows = VEST.after_records(reader, chain, N)
        home, base = deploy, N + T1_WINDOW[0]
    elif _flag_on(run):
        # the switch lies further back than the watch's walk: this run's reports and those of the runs it continues
        # (while their flag is on, at most 6 runs back), on the model step count's own thousands
        chain, cur = [run], run
        while len(chain) < 6:
            parent, _ck = VEST.parent_of(cur)
            parent = VEST.run_folder(parent, run.parent) if parent else None
            if parent is None or not _flag_on(parent):
                break
            chain.insert(0, Path(parent))
            cur = Path(parent)
        first = VEST.start_of(chain[0])
        rows = VEST.after_records(reader, chain, first if _num(first) else 0)
        N, home, base = None, run, None
    else:
        return None
    windows = standing_windows([r for r in rows if is_full(r)], base)
    return dict(home=home, N=N, windows=windows, alerts=standing_alerts(windows))


def standing_line(st):
    """The hourly sweep's line for the standing lines."""
    held = list(load_state(st["home"]).get("alerts") or [])
    mine = [a for a in held if str(a.get("id", "")).startswith(("T1-standing@", "S1@", "W1@"))]
    w = st["windows"][-1] if st["windows"] else None
    return ("  vestibular input path, the standing lines (judged on every window of 1,000 model steps while the path "
            "is on, past the watch's end too; two windows running across a line is an alert, never a stop): %s; "
            "standing alerts so far: %s"
            % ("no window is complete yet" if w is None else
               "the newest complete window, model steps {:,} to {:,}, reads {}; the target's line is {}".format(
                   w["model_steps"][0], w["model_steps"][1], _beside(w), w["t1"]),
               "none" if not mine else "%d, the newest: %s" % (len(mine), mine[-1].get("why"))))


# ------------------------------------------------------------------ what the hourly watcher calls
def evaluate(run, reader=None, write=True, readings=True):
    """The judgement for `run`: None when no switch reaches it, else a dict (deploy, N, chain, judgement, why, id,
    warnings, watch_over)."""
    reader = reader or READER
    found = BASE.find_switch(run, reader)
    if found is None:
        return None
    deploy, sw, chain = found
    N = int(sw["step"])
    args = _read_json(Path(deploy) / "args.json")
    known = isinstance(args, dict) and bool(args)
    on_card = known and str(args.get("device") or "").startswith("cuda")
    before = VEST.before_reports(reader, deploy, N, need=LINES["level_reports"])
    after, over = BASE.watch_records(VEST.after_records(reader, chain, N), N)
    taken = BASE.still_taken(BASE.taken_records(deploy, write=write), after, chain)
    skipped = [(Path(r).name,) + tuple(reader.skipped_updates(r)) for r in chain]
    # one update of the path's weight per model step (module note, M6)
    why, cid, rep = BASE.judge(before, after, sw, N, taken, per_step=1.0, on_card=on_card, readings=False,
                               skipped=skipped, newest_run=Path(chain[-1]).name)
    rep["lines"].pop("M3", None)
    rep["warnings"] = _own_warnings(rep.get("warnings") or [])
    for a in rep.get("alerts") or []:
        a["why"] = _reword(a.get("why"))
    for v in (rep.get("lines") or {}).values():
        if isinstance(v, dict) and v.get("why"):
            v["why"] = _reword(v["why"])
    if readings:
        full = [r for r in after if is_full(r)]
        rep["acceptance"] = acceptance(before, full, N)
        # the share of the newest full report's rows the path's cap scaled down (a reading, never judged)
        carried = [r for r in full if isinstance(r.get("path"), dict) and _num(r["path"].get("capped_share"))]
        rep["readings"]["capped_share"] = (None if not carried else
                                           {"step": carried[-1]["step"],
                                            "share": float(carried[-1]["path"]["capped_share"]),
                                            "cap": (float(carried[-1]["path"]["output_cap"])
                                                    if _num(carried[-1]["path"].get("output_cap")) else None)})
        # the encoder's own part against the path's (a reading, never judged): the newest full report's and the
        # lowest since the switch
        sloped = [r for r in full if isinstance(r.get("path"), dict) and _num(r["path"].get("change_slope"))]
        rep["readings"]["two_parts"] = (None if not sloped else
                                        {"step": sloped[-1]["step"],
                                         "slope": float(sloped[-1]["path"]["change_slope"]),
                                         "lowest": min(float(r["path"]["change_slope"]) for r in sloped),
                                         "reports": len(sloped)})
        rep["readings"]["launch_settings"] = BASE.launch_settings(deploy, before, chain)
        besides = rep["readings"]["launch_settings"].get("switch")
        if besides is not None:
            rep["readings"]["launch_settings"]["besides_the_flag"] = sorted(
                k for k in list(besides["changed"]) + list(besides["new"]) if k != "vestibular_input_path")
    return dict(deploy=deploy, N=N, chain=[str(r) for r in chain], judgement=rep, why=_reword(why), id=cid,
                warnings=list(rep.get("warnings") or []), watch_over=over)


def breach(run, reader=None):
    """The hourly watcher's 5-minute check: a stop reason for `run`, or None. It prints a loud line for every part of
    a switched run it cannot judge, prints each new alert once, and records a reported crossing in the deploy run's
    state file first (so a stop is not repeated on its own crossing)."""
    st = standing(run, reader)
    if st is not None:
        # the standing lines are judged on every check, past the watch's end too (module note)
        for alert in BASE.record_alerts(st["home"], run, st["alerts"]):
            print("  " + alert["why"], flush=True)
    found = BASE.find_switch(run, reader or READER)
    if found is None or load_state(found[0]).get("closed"):
        return None
    ev = evaluate(run, reader, write=True, readings=False)
    if ev is None:
        return None
    if not ev["watch_over"]:
        for w in ev["warnings"]:
            print("  VESTIBULAR INPUT PATH STOP LINES INCOMPLETE: " + w, flush=True)
    for alert in BASE.record_alerts(ev["deploy"], run, ev["judgement"].get("alerts") or []):
        print("  " + alert["why"], flush=True)
    if not ev["why"]:
        if ev["watch_over"]:
            VEST._write_state(ev["deploy"], dict(load_state(ev["deploy"]), closed=dict(
                time=time.time(), run=str(Path(run)), newest_step=ev["judgement"].get("newest_step"))))
        return None
    stop = ev["judgement"].get("stop") or {}
    BASE.record_crossing(ev["deploy"], dict(id=ev["id"], line=stop.get("line"), reason=ev["why"], run=str(Path(run)),
                                            time=time.time(), step=stop.get("step"), owner=stop.get("owner")))
    return ev["why"]


def _r2(x):
    """A read-out's R squared as a percentage between its two ends (0% = no better than predicting no change, 100% =
    exact; below 0% = worse than predicting no change)."""
    return _pct(x)


def _probe_words(reading, label="head_tilt"):
    if not reading or not isinstance(reading.get(label), dict):
        return "unmeasured"
    v = reading[label]
    return ("%s, its body-shuffled control %s, over %d reports in which the bodies turned %s degrees within a report"
            % (_r2(v.get("r2")), _r2(v.get("control")), reading.get("reports") or 0,
               "unmeasured" if not _num(v.get("within_body_turn_deg")) else "%.1f" % v["within_body_turn_deg"]))


def lines(run, reader=None, write=True):
    """The hourly sweep's lines for a person; [] when no switch reaches `run`."""
    ev = evaluate(run, reader, write=write)
    st = standing(run, reader)
    if ev is None:
        return [] if st is None else [standing_line(st)]
    rep, N = ev["judgement"], ev["N"]
    newest = rep.get("newest_step")
    L = rep.get("lines") or {}

    def state(k):
        v = L.get(k) or {}
        return v.get("state", "waiting") + ((" - " + v["why"]) if v.get("why") and v.get("state") != "held" else "")

    def shares(k, key="shares"):
        s = (L.get(k) or {}).get(key) or {}
        return ", ".join("%s %s" % (x, _pct(v)) for x, v in s.items()) or "unmeasured"
    out = ["  vestibular input path: switched on at model step {:,} (--vestibular-input-path on); {} full reports "
           "since, newest model step {}; the stop lines judge through the first full report at or past model step "
           "{:,}{}".format(N, rep.get("after_full_reports"), "none" if newest is None else "{:,}".format(newest),
                           N + LINES["horizon_steps"], " (reached: the watch is over)" if ev["watch_over"] else "")]
    for w in ev["warnings"]:
        out.append("  VESTIBULAR INPUT PATH STOP LINES INCOMPLETE: " + w)
    out.append("  vestibular input path, the switch record (the vestibular token with the path against without it on "
               "64 probe rows must be the same number): %s; non-finite updates (the stats' count of non-finite live "
               "losses, and the trainer's log lines for a skipped optimizer update: %d so far): %s"
               % (state("M1"), (L.get("M2") or {}).get("skipped_updates_in_the_logs") or 0, state("M2")))
    m6 = (L.get("M6") or {}).get("newest") or {}
    out.append("  vestibular input path, its output (0%% = the path adds nothing to the vestibular token, 100%% = as "
               "large as the vestibular encoder's own standardised output): %s; %s (above 100%% of the growth line "
               "stops the run; the line is the most Adam can add since the switch, %s of a token unit per model step. "
               "WHAT THIS LINE IS: a line on the first %d model steps after the switch; from there on it is above the "
               "path's cap of %s of one token unit, and what it still stops is an output that is not a finite number. "
               "It is no bound on the path and the encoder's part together: the standing lines read those)"
               % (state("M6"), "no report carries it yet" if not _num(m6.get("output_rms")) else
                  "at model step {:,} it reads {} of one token unit, {} of its growth line".format(
                      m6["step"], _pct(m6["output_rms"]), _pct(m6.get("share_of_line"))),
                  _pct(GROWTH_PER_MODEL_STEP), M6_REACH_STEPS, _pct(OUTPUT_CAP)))
    cs = (rep.get("readings") or {}).get("capped_share")
    out.append("  vestibular input path, its output's cap (the path scales a row's output back to the cap when it is "
               "larger, so no row is ever above it; 0%% of the rows scaled back = the path is its plain linear map, "
               "100%% = every row sits at the cap; a reading, never a stop): %s"
               % ("no report carries the share yet" if cs is None else
                  "at model step {:,} the cap{} scaled back {} of the report's rows".format(
                      cs["step"], "" if cs.get("cap") is None else " ({} of one token unit)".format(_pct(cs["cap"])),
                      _pct(cs["share"]))))
    tp = (rep.get("readings") or {}).get("two_parts")
    out.append("  vestibular input path, the encoder's own part against the path's part (the token is made of the sum "
               "of the two; under the flag value on the delayed lesson trains neither part, so no lesson trains one "
               "part and not the other, and under on_delayed_encoder_only it trains the encoder's part alone: -100%% "
               "= the encoder's part takes the path's change out of the sum, 0%% = the two changes are unrelated, "
               "100%% = the encoder's part's change follows the path's by the same amount, doubling the sum's change; "
               "the encoder bench read it between -1.5%% and 19.8%% at every reading from 2,000 updates on under on, "
               "and -46%% at 60,000 updates under on_delayed_encoder_only, while head tilt stayed readable from the "
               "token; a reading, never a stop: pass line T1 judges the outcome): %s"
               % ("no report carries it yet" if tp is None else
                  "at model step {:,} it reads {}, the lowest since the switch {}".format(
                      tp["step"], _pct(tp["slope"]), _pct(tp["lowest"]))))
    m4 = L.get("M4") or {}
    out.append("  vestibular input path, the other senses in reports 2 and 3 after the switch (each sense's "
               "next-model-step error ratio as a share of its line, the lower of the two reports; over 100%% in both "
               "stops the run): %s; %s" % (state("M4_early"), shares("M4_early")))
    out.append("  vestibular input path, the other senses against their level before the switch, every 20 reports, two "
               "levels (over 100%% of the stop line stops the run; over 100%% of the alert line is an alert, printed "
               "and recorded, never a stop): %s; the newest window (number %s), of the stop line: %s; of the alert "
               "line: %s" % (state("M4"), "none" if m4.get("window") is None else m4["window"] + 1, shares("M4"),
                             shares("M4", "alert_shares")))
    alerts = rep.get("alerts") or []
    out.append("  vestibular input path, alerts (never a stop): %s" % (
        "none so far" if not alerts else "%d, the newest: %s" % (len(alerts), alerts[-1]["why"])))
    m5 = L.get("M5") or {}
    out.append("  vestibular input path, speed: %s; in the newest window the trainer ran at %s of its model steps per "
               "second in %s (under %s in two windows running stops the run)"
               % (state("M5"), _pct(m5.get("share")),
                  "the 20 reports before the switch" if m5.get("window") == 0 else "the window before it",
                  _pct(LINES["M5_share"])))
    A = rep.get("acceptance") or {}
    t1, late = A.get("T1") or {}, A.get("T1_late") or {}
    trend = "; ".join("model steps {:,} to {:,} after the switch {}".format(
        w["model_steps"][0], w["model_steps"][1], _r2(((w.get("reading") or {}).get("head_tilt") or {}).get("r2")))
        for w in (A.get("windows") or []) if w.get("reading")) or "no window of 1,000 model steps is complete"
    out.append("  vestibular input path, pass line T1, never a stop (THE TARGET: head tilt read from the vestibular "
               "token by the trainer's own probe, each body's change within a report, scored on bodies the fit did not "
               "read; 0%% = no better than predicting no change, 100%% = exact; the line is %s with the "
               "control at least %s lower): %s over model steps %s; head tilt %s; the trunk's roll %s; before the "
               "switch: head tilt %s. The later reading over the watch's last 1,000 model steps: %s. Per window, head "
               "tilt: %s"
               % (_r2(T1_R2), _r2(T1_MARGIN), t1.get("state", "waiting"),
                  "{:,} to {:,}".format(*t1["model_steps"]) if t1.get("model_steps") else "unmeasured",
                  _probe_words(t1.get("reading")), _probe_words(t1.get("reading"), "trunk_roll"),
                  _probe_words(A.get("before")), late.get("state", "waiting"), trend))
    t2 = A.get("T2") or {}
    out.append("  vestibular input path, pass line T2, never a stop (vestibular's prediction error below the error of "
               "predicting no change; the reading is 1 - prediction error / no-change error: 0 = no better than no "
               "change, 1 = exact, below 0 = worse): %s%s"
               % (t2.get("state", "waiting"),
                  "" if not _num(t2.get("median")) else "; over the 20 full reports ending at the watch's end the "
                  "median reads %s and the 5th percentile of its resampled medians %s (the line: above 0)"
                  % (_pct(t2["median"]), _pct(t2.get("p5_of_median")))))
    rd = A.get("readings") or {}
    reach = rd.get("field_reach") or {}
    out.append("  vestibular input path, readings (never judged; the newest %d full reports, beside the 20 before the "
               "switch): vestibular's prediction reading %s, before %s; its token's spread between the 8 bodies %s of "
               "a token unit, before %s; its change between two model steps %s, before %s; its error ratio %s, before "
               "%s; the path's reach per input field group (the output's change per number when one field of the group "
               "changes by one running spread): %s"
               % (rd.get("reports") or 0, _pct(rd.get("improvement")), _pct(rd.get("improvement_before")),
                  _pct(rd.get("token_between_bodies")), _pct(rd.get("token_between_bodies_before")),
                  _pct(rd.get("token_change")), _pct(rd.get("token_change_before")),
                  _pct(rd.get("vestibular_ratio")), _pct(rd.get("vestibular_ratio_before")),
                  ", ".join("%s %s" % (k.replace("_", " "), _pct(v)) for k, v in reach.items()) or "unmeasured"))
    ls = (rep.get("readings") or {}).get("launch_settings") or {}
    if ls.get("switch") is None:
        own = "the switched run's args.json or its parent's could not be read"
    elif ls.get("besides_the_flag"):
        own = ("the switch's relaunch changed %s besides the path - MORE THAN THE PATH CHANGED AT THE SWITCH: a stop on "
               "the other senses or on the step rate cannot be put down to the path alone"
               % ", ".join(ls["besides_the_flag"]))
    else:
        own = "the switch's relaunch changed the path's flag and nothing else"
    out.append("  vestibular input path, the launch settings at the switch (the switched run's args.json against the "
               "run it continues; never judged): %s" % own)
    if st is not None:
        out.append(standing_line(st))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Print the vestibular input path's stop lines and pass lines for a run "
                                             "(reading only: nothing is written).")
    ap.add_argument("run")
    a = ap.parse_args(argv)
    ev = evaluate(a.run, BASE.StatsReader(), write=False)
    if ev is None:
        print("no vestibular_input_path_switch record reaches %s" % a.run)
        return 0
    for line in lines(a.run, BASE.StatsReader(), write=False):
        print(line)
    if ev["why"]:
        print("A STOP LINE IS CROSSED (the hourly watcher writes the stop): " + ev["why"])
    return 1 if ev["why"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
