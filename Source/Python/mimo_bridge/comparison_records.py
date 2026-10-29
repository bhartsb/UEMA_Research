# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""THE COMPARISON RECORDS (Oct 10, 2026): what a run keeps so that another vision rise like the one in the finger run's
record of model step 482,200 can be traced to its cause (D:/UEMA_Research/external/D/UEMA_scratch/plan_review/VISION_SPIKE_482200.md, section 6;
design note docs/gain_of_function_v4/comparison_records/COMPARISON_RECORDS.md).

A model step is one 50 ms model update (5 physics steps of 10 ms). A transition is one body's change from what was
sensed at model step k-1 to what was sensed at model step k; the learner predicts each sense's token of model step k from
model step k-1's tokens, the context of earlier tokens and the command in force between them.

THE FOUR RECORDS, all computed without gradient and without any random draw, from numbers the model step already
computes, and written into the run folder:
  (a) persist_pe_body, in the observer's model lane (<run>/observer/model/chunk_*.npz, under --observer record): per
      model step, body and sense, the persistence error of the transition ("nothing changes": mean over the token's
      numbers of (older token - newer token) squared), beside next_pe_body, the prediction error of the same transition.
  (b) vision_no_copy_body and gaze_copy_applied, in the same lane: per model step and body, vision's prediction error
      of the same transition with the gaze copy's term left out (the transformer and the prediction head alone); the
      error with it is next_pe_body's vision column.
  (c) <run>/comparison_records/steps_<pid>.jsonl, one line per model step: the step's loss, the gradient norm of every
      clipping group before the clip, the clip line and the count of skipped non-finite updates; and
      <run>/comparison_records/replay_passes_<pid>.jsonl, one line per replay pass: when it ran, its loss per sense,
      the motor term, the gradient norms before the clip, whether the optimizer stepped, and per window the body, the
      model steps of its first and target entries and the window's command contract codes.
  (d) <run>/comparison_records/vision_spike/: every model step on which some body's vision prediction error exceeds
      SPIKE_LINE is a line of crossings_<pid>.jsonl; when one body exceeds it on SPIKE_RUN consecutive model steps (the
      SPIKE RULE), the inputs of those transitions (inputs_step_<k>_<launch>.pt, written under a temporary name and
      renamed when complete) and the model's weights after that model step's update (checkpoints/<launch>/ckpt_step_<k>,
      without the episode companion, by a checkpoint manager that removes no snapshot: spike_checkpoint_manager) are
      kept, at most SPIKE_MAX_SNAPSHOTS times per launch and at least SPIKE_MIN_GAP_STEPS model steps apart; <launch> is
      the trainer process's id and its records' creation time in milliseconds. The count lives in the trainer process's memory: it
      starts at 0 at every launch, so a run folder reused across relaunches can hold more events than the cap, and a
      production lineage must clean up or archive these folders (nothing here does). An event is refused, and the
      refusal written as its events line, when the folder's drive holds less free space than the event's estimated
      size (its inputs' bytes plus the weights' bytes, SPIKE_WEIGHTS_BYTES_MEASURED or the last event's own) plus the
      reserve (DEFAULT_RESERVE_BYTES, 12 GiB); the weights are refused alone when the inputs' write left too little.

THE SPIKE RULE'S GROUND, measured on the learning-on measurement's model lane (3 runs from checkpoint 481,800, 1,000
model steps x 8 bodies each; D:/UEMA_Research/external/D/UEMA_scratch/finger_investigation/learning_on/measure/run/arm_*/run/observer/model):
SPIKE_GROUND below. A single transition above 0.2 occurs in ordinary records (control 0.2437 at 482,239, body 3;
control_again 0.2215 at 481,915, body 3), so a single-transition line cannot tell the finger run's event from them;
three consecutive transitions of one body can: outside the finger run's model steps 482,100 to 482,115 no body stayed
above 0.0745 for three consecutive model steps in any of the 3 runs, and body 0 stayed above 0.2797 for 482,106 to
482,108. The line 0.14 lies near the geometric middle of the two (0.144): about 1.9 times above the largest ordinary
three-step minimum and 2.0 times below the event's.

Nothing here writes outside <run>/comparison_records and the lane's own folder.

FAILURES (the bundle review of Oct 10, FIX-1 and FIX-3). Every record computation, in the learner (lane_moment,
pass_record, the gaze copy's reading, the slot map, the spike trigger) and in the trainer's writer, sits behind an error
guard: a failure is counted per record (record_failed: its count, first and last error), logged at counts 1, 2, 4, 8,
..., written into the status file at every stats record and at close, and recording continues; no failure stops the
learner or turns a writer off. A line holding a number that is not finite is written with that number as null and the
field's path named in the line's "nonfinite" list; a value JSON has no form for is written as its repr and named in
"not_json"; a line that still cannot be serialised is dropped and counted (unserialisable) with its error kept.

FLUSHING (FIX-2). The trainer calls ComparisonRecords.flush at every stats record (train_mimo's flush_stats, every
--report-every model steps, before the record is written) and at close: every held line reaches disk then, with the
status file. A stop that ends the trainer's process with no close (learning_on's stops kill the process tree) loses at
most the lines added after the last stats record.

RECORDING is a module setting for tests only (never a launch flag): False makes the learner compute none of these
records, so a test can show that a learner with them and one without train bit for bit alike.
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
import weakref
from collections import deque
from typing import Optional

import numpy as np

log = logging.getLogger("mimo_bridge.comparison_records")

RECORDS_VERSION = "comparison_records_v1"

# True in every run; tests set False to show that the records change no training number (not a launch flag)
RECORDING = True

# ------------------------------------------------------------------------------------------------ the spike rule
SPIKE_LINE = 0.14             # per-transition vision prediction error (mean squared difference per token number)
SPIKE_RUN = 3                 # consecutive model steps of ONE body above the line
SPIKE_MAX_SNAPSHOTS = 3       # inputs and weights kept at most this many times per launch (counted in the process)
SPIKE_MIN_GAP_STEPS = 300     # model steps between two kept events at least
# the weights' size an event's free-space check assumes before this launch has saved any: checkpoint 481,800's
# model.safetensors 73,119,988 + optimizer.pt 136,545,297 + runtime_state.pt 489,336,432 + model_receipt.json 234,749
# + meta.json 112 bytes (docs/gain_of_function_v4/comparison_records/evidence/checkpoint_files_and_disk.txt); after
# an event's save the larger of this and that save's own bytes is used
SPIKE_WEIGHTS_BYTES_MEASURED = 699_236_578
SPIKE_GROUND = {
    "source": ("the learning-on measurement's model lane (next_pe_body, vision column): D:/UEMA_Research/external/D/UEMA_scratch/"
               "finger_investigation/learning_on/measure/run/arm_{control,finger,control_again}/run/observer/model, "
               "1,000 model steps (481,800 to 482,799) x 8 bodies per run"),
    "single_transition_max_outside_event": {"control": [0.2437, 482239, 3], "finger": [0.1920, 482095, 5],
                                            "control_again": [0.2215, 481915, 3]},
    "three_step_minimum_max_outside_event": {"control": [0.0745, 482241, 3], "finger": [0.0713, 482097, 5],
                                             "control_again": [0.0562, 481982, 3]},
    "three_step_minimum_in_event": [0.2797, 482108, 0],
    "fires_with_this_rule": {"control": 0, "finger": 1, "control_again": 0},
    "median_per_transition": {"control": 0.0032, "finger": 0.0034, "control_again": 0.0029},
    "event": "finger run, model steps 482,100 to 482,115 (body 0: 0.2847, 0.2814, 0.2797, 0.2727, 0.2092 at 482,106 "
             "to 482,110)",
}

# the senses of the observer's model lane (observer_model_lane.SENSES)
SENSES = ("vision", "tactile", "audio", "proprioception", "vestibular")

FLUSH_LINES = 100                         # a writer appends its lines to the file every this many lines (and at
                                          # every stats record and at close: ComparisonRecords.flush)
DEFAULT_RESERVE_BYTES = 12 * 2 ** 30      # the records' reserve (the observer's default; not the run's
                                          # --observer-reserve-gb): nothing is written below this much free space
NAMED_MAX = 50                            # field paths a line names in "nonfinite" / "not_json" at most

DEFINITIONS = {
    "version": RECORDS_VERSION,
    "model_step": "one 50 ms model update (5 physics steps of 10 ms)",
    "lines": ("every .jsonl line is one JSON object; a number that was not finite is written as null and its field's "
              "path is named in the line's nonfinite list (nonfinite_count when more than %d); a value JSON has no "
              "form for is written as its repr and named in not_json; a null with no such name is a value that was "
              "missing" % NAMED_MAX),
    "steps": ("one line per model step k: loss (the step's total loss, the number the stats average), grad_norm "
              "(the gradient norm of the core group before the clip, as the step returns it: 0.0 when the update was "
              "skipped), group_norms (every clipping group's norm before the clip: core = every parameter outside "
              "the named groups, which are clipped on their own), clip (the clip line), nonfinite_steps (the "
              "learner's count of skipped non-finite updates so far)"),
    "replay_passes": ("one line per replay pass: step (the trainer's model-step counter when the pass ran; it runs "
                      "after model step step-1 and its burst, before model step step), loss (the pass's total), terms "
                      "(per sense the mean over the windows whose target model step is real of (prediction - next token) "
                      "squared, unweighted), term_weights (the --next-frame-norm weight of each sense, when on), motor "
                      "(the motor term added to the loss), grad_norm and group_norms (before the clip), stepped "
                      "(whether the optimizer stepped), contract (the command contract code the pass required; null = "
                      "none), windows: per window body, first (the model step of its first entry), target (of its "
                      "last entry, the one predicted), contract_min and contract_max (the command contract codes of "
                      "its step-from entries, limb_target_control's bits), restored (true when the window holds "
                      "entries this process did not add: their model steps are read as newest - entries since, which "
                      "holds when the ring took one entry per model step)"),
    "lane_persist_pe_body": ("[rows, bodies, senses] per model step k, body and sense: mean over the token's numbers "
                             "of (older token - newer token) squared, the older token as the transformer received it "
                             "(re-encoded with the command in force); NaN where the sense was not graded"),
    "lane_vision_no_copy_body": ("[rows, bodies] per model step k and body: vision's prediction error of the same "
                                 "transition with the gaze copy's term left out; equal to next_pe_body's vision column "
                                 "where the copy added nothing (gaze_copy_applied false)"),
    "spike_rule": ("one body's vision prediction error above %.2f on %d consecutive model steps; kept at most %d "
                   "times per launch (the count starts at 0 at every launch of the trainer), at least %d model steps "
                   "apart; refused when the drive's free space is less than the event's estimated size plus the "
                   "reserve" % (SPIKE_LINE, SPIKE_RUN, SPIKE_MAX_SNAPSHOTS, SPIKE_MIN_GAP_STEPS)),
    "spike_inputs": ("inputs_step_<k>_<launch>.pt (torch.save under a temporary name, then renamed): definitions, and per transition of the run (its model steps): "
                     "prev_tokens (each sense's token of the older model step as the transformer received it), "
                     "target_tokens (each sense's token of the newer model step, the lesson's target), context (the "
                     "context buffer the forward pass read), memory_bias (or None), command (the 26 x 9 command in "
                     "force), pose (the 26 x 10 achieved handle state of the older model step, which the gaze copy reads), "
                     "prediction (each sense's prediction, the gaze copy included), vision_without_copy, raw_prev and "
                     "raw_now (the older and the newer model step's raw sense inputs as the encoders received them), "
                     "masks, errors and persistence (per body, vision); every tensor in its own dtype, on the "
                     "processor"),
    "spike_weights": ("vision_spike/checkpoints/<launch>/ckpt_step_<k>: the learner's checkpoint after model step k's "
                      "update (model.safetensors, optimizer.pt, runtime_state.pt, meta.json, model_receipt.json), "
                      "written by LemLite.save_checkpoint into this launch's own folder, without the episode companion, "
                      "by a checkpoint manager whose cleanup removes no snapshot (no keep_last_n, no registered "
                      "retention); not listed in the run's checkpoints.json, so no resume reads it. Transition k was predicted with the weights "
                      "before that update; transitions k-1 and k-2 with weights 1 and 2 live updates earlier still, "
                      "with up to 1 replay pass between. The weights before the onset are not kept, so the model as it "
                      "stood before the rise cannot be run again; the inputs file holds every captured prediction, so "
                      "the differences can be measured offline."),
}


def _lp(path: str) -> str:
    from .observer_record import _lp as lp
    return lp(path)


def _atomic_json(path: str, value) -> None:
    from .observer_record import _atomic_json as aj
    aj(path, value)


def _num(x):
    """A JSON number, or None for a missing or non-finite one."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _float(x):
    """A plain float, non-finite kept (the line writer writes it as null and names its field), or None for a missing
    or non-numeric value."""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _jsonable(value, path: str, nonfinite: list, other: list, depth: int = 0):
    """A copy of `value` that json.dumps(allow_nan=False) accepts: a float that is not finite becomes None and its path
    joins `nonfinite`; numpy numbers and arrays and tensors become plain numbers and lists; an object JSON has no form
    for becomes its repr and its path joins `other`."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if math.isfinite(v):
            return v
        nonfinite.append(path or "<line>")
        return None
    if depth >= 32:
        other.append(path or "<line>")
        return repr(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v, ("%s.%s" % (path, k)) if path else str(k), nonfinite, other, depth + 1)
                for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [_jsonable(v, "%s.%d" % (path, i), nonfinite, other, depth + 1) for i, v in enumerate(value)]
    if hasattr(value, "tolist") and callable(value.tolist):         # numpy arrays, tensors
        return _jsonable(value.tolist(), path, nonfinite, other, depth + 1)
    other.append(path or "<line>")
    return repr(value)


class LineWriter:
    """Lines appended to one JSONL file, flush_lines at a time, at every flush() (the trainer's stats record:
    ComparisonRecords.flush) and at close.

    No error turns the writer off and none reaches the caller (the bundle review of Oct 10, FIX-1):
      - a line holding a number that is not finite is written with that number as null and the field's path named in
        the line's "nonfinite" list; a value JSON has no form for is written as its repr and named in "not_json"
        (counted: repaired);
      - a line that still cannot be serialised is dropped and counted (unserialisable), its error kept;
      - nothing is written while the disk has less than reserve_bytes free beside the lines: the lines held then are
        dropped and counted (dropped_disk);
      - a write that fails drops its lines and counts them (dropped_write; some of them may have reached the file).
    dropped is the sum of the three. The first error, and then every error whose count is a power of 2, is logged and
    written to <name>_error_<pid>.json with the counts; the status (status()) carries them all."""

    def __init__(self, folder: str, name: str, reserve_bytes: int = DEFAULT_RESERVE_BYTES,
                 flush_lines: int = FLUSH_LINES):
        self.folder = folder
        self.name = name
        self.pid = os.getpid()
        self.path = os.path.join(folder, "%s_%d.jsonl" % (name, self.pid))
        self.reserve_bytes = int(reserve_bytes)
        self.flush_lines = max(1, int(flush_lines))
        self.lines = []
        self.enabled = True          # read by the status file; no error sets it False
        self.error = None
        self.errors = 0
        self.written = 0
        self.dropped = 0
        self.dropped_disk = 0
        self.dropped_write = 0
        self.unserialisable = 0
        self.repaired = 0
        self.bytes = 0

    def add(self, row: dict) -> None:
        try:
            text = json.dumps(row, separators=(",", ":"), allow_nan=False)
        except Exception:
            try:
                nonfinite, other = [], []
                clean = _jsonable(row, "", nonfinite, other)
                if isinstance(clean, dict):
                    if nonfinite:
                        clean["nonfinite"] = nonfinite[:NAMED_MAX]
                        if len(nonfinite) > NAMED_MAX:
                            clean["nonfinite_count"] = len(nonfinite)
                    if other:
                        clean["not_json"] = other[:NAMED_MAX]
                text = json.dumps(clean, separators=(",", ":"), allow_nan=False)
                self.repaired += 1
            except Exception as ex:
                self.unserialisable += 1
                self.dropped += 1
                self._note(ex, row)
                return
        self.lines.append(text)
        if len(self.lines) >= self.flush_lines:
            self.flush()

    def flush(self) -> None:
        if not self.lines:
            return
        lines, self.lines = self.lines, []
        text = "\n".join(lines) + "\n"
        try:
            free = shutil.disk_usage(self.folder).free
            if free - len(text) < self.reserve_bytes:
                self.dropped += len(lines)
                self.dropped_disk += len(lines)
                return
            with open(_lp(self.path), "a", encoding="utf-8") as f:
                f.write(text)
            self.written += len(lines)
            self.bytes += len(text.encode("utf-8"))
        except Exception as ex:          # a record must never stop the learner
            self.dropped += len(lines)
            self.dropped_write += len(lines)
            self._note(ex)

    def _note(self, ex, row=None) -> None:
        """An error counted and kept; logged and written to the error file at counts 1, 2, 4, 8, ..."""
        self.errors += 1
        self.error = repr(ex)
        if self.errors & (self.errors - 1):
            return
        step = row.get("step") if isinstance(row, dict) else None
        log.warning("comparison records: %s line not written (%d errors so far; recording continues): %r",
                    self.name, self.errors, ex)
        try:
            _atomic_json(os.path.join(self.folder, "%s_error_%d.json" % (self.name, self.pid)),
                         dict(time=time.time(), error=repr(ex), step=_float(step), status=self.status()))
        except Exception:
            pass

    def status(self) -> dict:
        return dict(path=self.path, written=self.written, held=len(self.lines), dropped=self.dropped,
                    dropped_disk=self.dropped_disk, dropped_write=self.dropped_write,
                    unserialisable=self.unserialisable, repaired=self.repaired, bytes=self.bytes,
                    enabled=self.enabled, errors=self.errors, error=self.error)


# ------------------------------------------------------------------------------------------------ failures
# THE COUNT OF EACH RECORD'S FAILURES, per owner (a learner, a lane, the trainer's writer), kept here and not on the
# owner, as the spike trigger is (weak keys: an owner that is gone leaves no entry). record_failed is called by every
# record's error guard; failures_of is read into the status file.
_FAILURES = weakref.WeakKeyDictionary()
_FAILURES_UNKEYED = {}


def record_failed(owner, name: str, ex) -> int:
    """Count one failure of record `name` computed for `owner`; log it at counts 1, 2, 4, 8, ... Returns the count.
    Never raises."""
    try:
        try:
            table = _FAILURES.setdefault(owner, {})
        except TypeError:
            table = _FAILURES_UNKEYED.setdefault(id(owner), {})
        entry = table.setdefault(str(name), dict(count=0, first_error=None, last_error=None))
        entry["count"] += 1
        entry["last_error"] = repr(ex)
        if entry["first_error"] is None:
            entry["first_error"] = entry["last_error"]
        n = entry["count"]
        if not n & (n - 1):
            log.warning("comparison record %s failed (%d so far; training unaffected, recording continues): %r",
                        name, n, ex)
        return n
    except Exception:
        return -1


def failures_of(owner) -> dict:
    """{record name: {count, first_error, last_error}} of `owner`, a copy."""
    try:
        try:
            table = _FAILURES.get(owner)
        except TypeError:
            table = _FAILURES_UNKEYED.get(id(owner))
        return {k: dict(v) for k, v in (table or {}).items()}
    except Exception:
        return {}


# ------------------------------------------------------------------------------------------------ the spike trigger
class VisionSpike:
    """The spike rule over the per-transition vision prediction errors of one run (module text, record d).

    observe(step, errors, valid, persistence) is called once per model step that graded vision, with every body's
    numbers; it updates each body's count of consecutive model steps above `line` (a model step not observed, or a
    body not graded, ends the count), writes a crossings line when some body is above the line, and returns whether the
    caller should capture this model step's inputs (some body above the line and an event still allowed). keep(step,
    inputs) holds the newest `run` captures. judge(step) fires when some body's count reaches `run` and the rate limit
    allows it: `pending` then holds the event (its bodies, errors and the captures of its transitions) for the trainer
    to write; the trainer clears it."""

    def __init__(self, folder: Optional[str], line: float = SPIKE_LINE, run: int = SPIKE_RUN,
                 max_events: int = SPIKE_MAX_SNAPSHOTS, min_gap: int = SPIKE_MIN_GAP_STEPS,
                 reserve_bytes: int = DEFAULT_RESERVE_BYTES):
        self.folder = folder
        self.line = float(line)
        self.run = max(1, int(run))
        self.max_events = int(max_events)
        self.min_gap = int(min_gap)
        self.counts = None
        self.last_step = None
        self.captures = deque(maxlen=self.run)
        self.fired = []
        self.suppressed = 0
        self.pending = None
        self.enabled = True          # no failure sets it False (fail counts)
        self.error = None
        self.failures = 0
        self.crossings = (LineWriter(folder, "crossings", reserve_bytes=reserve_bytes, flush_lines=1)
                          if folder else None)

    def allowed(self, step: int) -> bool:
        """Whether an event at model step `step` would be kept under the rate limit."""
        if len(self.fired) >= self.max_events:
            return False
        return not self.fired or int(step) - self.fired[-1] >= self.min_gap

    def observe(self, step: int, errors, valid, persistence=None) -> bool:
        e = np.asarray(errors, dtype=np.float64).reshape(-1)
        ok = np.asarray(valid, dtype=np.float64).reshape(-1) > 0
        over = ok & np.isfinite(e) & (e > self.line)
        step = int(step)
        if self.counts is None or len(self.counts) != len(e) or self.last_step is None or step != self.last_step + 1:
            self.counts = np.zeros(len(e), dtype=np.int64)
            self.captures.clear()
        self.counts = np.where(over, self.counts + 1, 0)
        self.last_step = step
        if not over.any():
            self.captures.clear()
            return False
        if self.crossings is not None:
            p = None if persistence is None else np.asarray(persistence, dtype=np.float64).reshape(-1)
            self.crossings.add(dict(step=step, bodies=[int(b) for b in np.nonzero(over)[0]],
                                    errors=[_float(e[b]) for b in np.nonzero(over)[0]],
                                    persistence=(None if p is None else [_float(p[b]) for b in np.nonzero(over)[0]]),
                                    counts=[int(self.counts[b]) for b in np.nonzero(over)[0]]))
        # inputs are taken only while an event that includes this transition could still be kept: such an event
        # fires at this model step or at most run - 1 model steps later, and allowed() only opens with time
        return self.allowed(step + self.run - 1)

    def keep(self, step: int, inputs: dict) -> None:
        self.captures.append((int(step), inputs))

    def judge(self, step: int):
        """The event of model step `step`, or None (module text). An event the rate limit refuses is counted and
        written to the crossings file."""
        step = int(step)
        if self.counts is None or int(self.counts.max(initial=0)) < self.run:
            return None
        bodies = [int(b) for b in np.nonzero(self.counts >= self.run)[0]]
        if not self.allowed(step):
            self.suppressed += 1
            if self.crossings is not None:
                self.crossings.add(dict(step=step, rule_met=bodies, kept=False,
                                        why=("%d events kept already" % len(self.fired)
                                             if len(self.fired) >= self.max_events
                                             else "%d model steps after the event of %d, fewer than %d"
                                             % (step - self.fired[-1], self.fired[-1], self.min_gap))))
            return None
        lo = step - self.run + 1
        caps = [(s, c) for s, c in self.captures if lo <= s <= step]
        self.fired.append(step)
        self.pending = dict(step=step, bodies=bodies, first_step=lo, captures=caps)
        if self.crossings is not None:
            self.crossings.add(dict(step=step, rule_met=bodies, kept=True, captured_steps=[s for s, _ in caps]))
        return self.pending

    def fail(self, ex) -> None:
        """A failure of the trigger's reading or capture at one model step: counted and kept (record_failed under the
        name "spike"), written to spike_error_<pid>.json at counts 1, 2, 4, 8, ...; the trigger stays on and observes
        the next model step (an event then names the captures it holds)."""
        self.failures += 1
        self.error = repr(ex)
        record_failed(self, "spike", ex)
        if self.folder and not self.failures & (self.failures - 1):
            try:
                _atomic_json(os.path.join(self.folder, "spike_error_%d.json" % os.getpid()),
                             dict(time=time.time(), error=repr(ex), failures=self.failures))
            except Exception:
                pass


# THE TRIGGER OF EACH LEARNER, kept here and not on the learner: a copied execution of the learner
# (copied_execution.capture) copies every public attribute and refuses an object it cannot copy, and a copy must not
# fire the original's trigger. Weak keys: a learner that is gone leaves no entry.
_SPIKES = weakref.WeakKeyDictionary()


def attach_spike(model, spike) -> None:
    _SPIKES[model] = spike


def spike_of(model):
    """The spike trigger attached to this learner, or None (every learner a trainer did not attach one to)."""
    try:
        return _SPIKES.get(model)
    except TypeError:
        return None


def cpu_tree(value):
    """A copy of a nest of tensors, arrays and plain values on the processor, every tensor in its own dtype (no
    gradient, no arithmetic: the numbers are the step's own)."""
    import torch
    if torch.is_tensor(value):
        return value.detach().to("cpu").clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool, np.generic)):
        return value
    return repr(value)


# ------------------------------------------------------------------------------------------------ replay windows
def window_record(ring, slot_steps, env, start, ctx_len: int) -> dict:
    """Per window of one replay pass: its body, the model steps of its first and target entries, the command
    contract codes of its step-from entries and whether it holds entries this process did not add (module text).
    ring: the ReplayMemory the windows came from (its pointer and contract codes are read, nothing is written);
    slot_steps: [ring size] the model step each slot holds, -1 where this process wrote none; env, start:
    each window's body and first slot (ReplayMemory.sample_windows' source_env and source_slot)."""
    env = np.asarray(env, dtype=np.int64).reshape(-1)
    start = np.asarray(start, dtype=np.int64).reshape(-1)
    size, need = int(ring.size), int(ctx_len) + 1
    idx = (start[:, None] + np.arange(need)[None, :]) % size                    # [R, need]
    newest_slot = (int(ring.ptr) - 1) % size
    steps = np.asarray(slot_steps, dtype=np.int64)
    newest = int(steps[newest_slot]) if steps.size == size else -1
    since = (newest_slot - idx) % size                                           # entries added after each slot's
    own = steps[idx] if steps.size == size else np.full(idx.shape, -1, dtype=np.int64)
    derived = np.where(own >= 0, own, (newest - since) if newest >= 0 else -1)
    codes = ring.action_contract.numpy() if hasattr(ring.action_contract, "numpy") else np.asarray(ring.action_contract)
    from_codes = codes[env[:, None], idx[:, :-1]] if need > 1 else np.zeros((len(env), 1), dtype=np.int64)
    return dict(body=[int(x) for x in env], first=[int(x) for x in derived[:, 0]],
                target=[int(x) for x in derived[:, -1]],
                contract_min=[int(x) for x in from_codes.min(axis=1)],
                contract_max=[int(x) for x in from_codes.max(axis=1)],
                restored=[bool(x) for x in (own < 0).any(axis=1)])


# ------------------------------------------------------------------------------------------------ the learner's side
def tree_bytes(value) -> int:
    """The bytes of every tensor and array in a nest (the size an inputs file's numbers take, before torch.save's own
    few bytes of structure)."""
    try:
        import torch
        if torch.is_tensor(value):
            return int(value.numel()) * int(value.element_size())
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(tree_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple, deque)):
        return sum(tree_bytes(v) for v in value)
    return 0


def lane_moment(step: int, senses, per, nf_in: dict, tokens_raw: dict, gaze_copy, B: int) -> dict:
    """Records a and b of one model step, for the observer's model lane (LemLite.observer_moment): the per-body
    prediction error the step already computed (`per` [senses, B], on the device), the persistence error of the same
    transitions ("nothing changes": mean over the token's numbers of (older token as the transformer received it -
    newer token) squared), and vision's error without the gaze copy's term; read from the device in one transfer.
    Called by LemLite.step behind its error guard (module text, FAILURES): a failure leaves the step's own
    next_pe_per_body in place."""
    import torch
    senses = list(senses)
    with torch.no_grad():
        pers = torch.stack([((nf_in[m].detach().float().reshape(B, -1)
                              - tokens_raw[m].squeeze(1).detach().float().reshape(B, -1)) ** 2).mean(-1)
                            for m in senses])
        rows = [per, pers]
        if gaze_copy is not None and "vision" in senses:
            rows.append(((gaze_copy["without"].float().reshape(B, -1)
                          - tokens_raw["vision"].squeeze(1).detach().float().reshape(B, -1)) ** 2).mean(-1)
                        .reshape(1, B))
        read = torch.cat(rows, dim=0).cpu().numpy()
    S = len(senses)
    return {"step": int(step), "senses": senses, "next_pe_per_body": read[:S],
            "persistence_per_body": read[S:2 * S],
            "vision_without_copy_per_body": (read[2 * S] if read.shape[0] > 2 * S else None),
            "gaze_copy_applied": (None if gaze_copy is None else bool(gaze_copy["applied"]))}


def pass_record(*, loss_value, batch, graded, terms, term_values, weights, motor_term, grad_norm, loss_finite,
                clip_norms, contract, windows, ring, slot_steps, ctx_len) -> dict:
    """Record c's pass line (module text; DEFINITIONS replay_passes) from numbers one replay pass already computed:
    term_values is the per-sense list the pass read for --next-frame-norm, or None (then read here from `terms`,
    without gradient); weights the --next-frame-norm weights or None. Numbers that are not finite stay in the line as
    they are, and the line writer writes them as null and names them (FIX-1). Called by LemLite.replay_update behind
    its error guard."""
    if term_values is None and terms:
        import torch
        term_values = torch.stack(terms).detach().float().tolist()
    gn = _float(grad_norm)
    stepped = bool(loss_finite) and gn is not None and math.isfinite(gn)
    return dict(
        loss=_float(loss_value), batch=int(batch),
        terms=(None if term_values is None else {m: _float(v) for m, v in zip(graded, term_values)}),
        term_weights=(None if weights is None else {m: _float(v) for m, v in weights.items()}),
        motor=_float(motor_term.detach().float().item()),
        grad_norm=(gn if stepped else None),
        group_norms=({k: _float(v) for k, v in (clip_norms or {}).items()} if loss_finite else None),
        stepped=stepped, contract=contract,
        windows=(None if windows.get("source_env") is None else
                 window_record(ring, slot_steps, windows["source_env"], windows["source_slot"], ctx_len)))


# ------------------------------------------------------------------------------------------------ the trainer's side
def _folder_bytes(path) -> int:
    total = 0
    for root, _, files in os.walk(_lp(str(path))):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


class ComparisonRecords:
    """The trainer's writer of records (c) and (d) (module text): one per launch, in <run>/comparison_records.
    flush() is called at every stats record and by close(); every method counts its failures and never raises."""

    def __init__(self, folder: str, run: str, n_envs: int, reserve_bytes: int = DEFAULT_RESERVE_BYTES,
                 weights_bytes_estimate: int = SPIKE_WEIGHTS_BYTES_MEASURED):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.spike_folder = os.path.join(folder, "vision_spike")
        os.makedirs(self.spike_folder, exist_ok=True)
        self.pid = os.getpid()
        self.reserve_bytes = int(reserve_bytes)
        self.weights_bytes_estimate = int(weights_bytes_estimate)
        self.last_weights_bytes = None
        self.refused_events = 0
        self.refused_weights = 0
        self.flushes = 0
        self.steps = LineWriter(folder, "steps", reserve_bytes=reserve_bytes)
        self.passes = LineWriter(folder, "replay_passes", reserve_bytes=reserve_bytes)
        self.spike = VisionSpike(self.spike_folder, line=SPIKE_LINE, run=SPIKE_RUN, max_events=SPIKE_MAX_SNAPSHOTS,
                                 min_gap=SPIKE_MIN_GAP_STEPS, reserve_bytes=reserve_bytes)
        self.events = LineWriter(self.spike_folder, "events", reserve_bytes=reserve_bytes, flush_lines=1)
        self.cost_s = 0.0
        self._model = None
        created = time.time()
        # THIS LAUNCH'S NAME (process id and creation time in milliseconds, as the header file's name): a later launch
        # in the same run folder writes its spike files under another name, so no save of one launch replaces another's
        self.launch = "%d_%d" % (self.pid, int(created * 1000))
        # the weights of this launch's events: vision_spike/checkpoints/<launch>/ckpt_step_<k> (snapshot_weights)
        self.weights_folder = os.path.join(self.spike_folder, "checkpoints", self.launch)
        self.header = dict(version=RECORDS_VERSION, run=str(run), n_envs=int(n_envs), pid=self.pid,
                           created=created, launch=self.launch, weights_folder=self.weights_folder,
                           definitions=DEFINITIONS, senses=list(SENSES),
                           spike=dict(line=SPIKE_LINE, run=SPIKE_RUN, max_events=SPIKE_MAX_SNAPSHOTS,
                                      max_events_per="launch", min_gap_steps=SPIKE_MIN_GAP_STEPS, ground=SPIKE_GROUND,
                                      weights_bytes_estimate=self.weights_bytes_estimate),
                           reserve_bytes=self.reserve_bytes, flush_lines=FLUSH_LINES,
                           flushed="every stats record (train_mimo flush_stats) and at close")
        _atomic_json(os.path.join(folder, "header_%d_%d.json" % (self.pid, int(self.header["created"] * 1000))),
                     self.header)

    def attach(self, model) -> None:
        """Hand the learner the spike trigger (it reads the vision rows where the step computes them); the learner's
        record failures are read into the status file."""
        attach_spike(model, self.spike)
        try:
            self._model = weakref.ref(model)
        except TypeError:
            self._model = None

    def after_step(self, step: int, model, res, snapshot=None) -> None:
        """Record (c)'s step line, and record (d)'s event when the step fired one. snapshot(step) -> path: the
        weights' save (the trainer's); None leaves the weights unsaved."""
        t0 = time.perf_counter()
        try:
            norms = getattr(model, "clip_norms", None) or {}
            self.steps.add(dict(step=int(step), loss=_float(getattr(res, "loss", None)),
                                grad_norm=_float(getattr(res, "grad_norm", None)),
                                group_norms={k: _float(v) for k, v in norms.items()},
                                clip=_float(getattr(model, "grad_clip", None)),
                                nonfinite_steps=int(getattr(model, "_nonfinite_steps", 0) or 0)))
        except Exception as ex:
            record_failed(self, "step_line", ex)
        try:
            ev = self.spike.pending
            if ev is not None:
                self.spike.pending = None
                self.write_event(ev, model, snapshot)
        except Exception as ex:
            record_failed(self, "spike_event", ex)
        self.cost_s += time.perf_counter() - t0

    def write_event(self, ev: dict, model, snapshot=None) -> None:
        """One kept event (record d): the free-space check, the inputs file, then the weights (module text)."""
        row = dict(step=None)
        try:
            import hashlib
            import torch
            step = int(ev["step"])
            row = dict(step=step, bodies=list(ev["bodies"]), first_step=int(ev["first_step"]),
                       captured_steps=[int(s) for s, _ in ev["captures"]], time=time.time())
            inputs_est = tree_bytes([c for _, c in ev["captures"]])
            weights_est = (max(self.weights_bytes_estimate, int(self.last_weights_bytes or 0))
                           if snapshot is not None else 0)
            free = int(shutil.disk_usage(self.spike_folder).free)
            row.update(free_bytes=free, inputs_bytes_estimate=inputs_est, weights_bytes_estimate=weights_est,
                       reserve_bytes=self.reserve_bytes)
            if free < inputs_est + weights_est + self.reserve_bytes:
                self.refused_events += 1
                row["refused"] = ("the drive has %.2f GB free, less than the event's estimated %.2f GB (inputs %.2f GB "
                                  "+ weights %.2f GB) plus the %.2f GB reserve: nothing of the event is written"
                                  % (free / 1e9, (inputs_est + weights_est) / 1e9, inputs_est / 1e9,
                                     weights_est / 1e9, self.reserve_bytes / 1e9))
                self.events.add(row)
                return
            # written under a temporary name and renamed when complete (the review's NOTE-4): a process ended during
            # the write leaves inputs_step_<k>_<launch>.pt.part, never a partial file under the final name
            path = os.path.join(self.spike_folder, "inputs_step_%d_%s.pt" % (step, self.launch))
            part = path + ".part"
            torch.save(dict(version=RECORDS_VERSION, definitions=DEFINITIONS["spike_inputs"],
                            rule=DEFINITIONS["spike_rule"], step=step, bodies=ev["bodies"], launch=self.launch,
                            transitions=[dict(step=int(s), **c) for s, c in ev["captures"]]), _lp(part))
            os.replace(_lp(part), _lp(path))
            h = hashlib.sha256()
            with open(_lp(path), "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            row.update(inputs=path, inputs_bytes=os.path.getsize(_lp(path)), inputs_sha256=h.hexdigest())
            if snapshot is not None:
                free = int(shutil.disk_usage(self.spike_folder).free)
                if free < weights_est + self.reserve_bytes:
                    self.refused_weights += 1
                    row["weights_refused"] = ("after the inputs the drive has %.2f GB free, less than the weights' "
                                              "estimated %.2f GB plus the %.2f GB reserve: the weights are not saved"
                                              % (free / 1e9, weights_est / 1e9, self.reserve_bytes / 1e9))
                else:
                    t = time.perf_counter()
                    row["weights"] = snapshot(step)
                    row["weights_s"] = round(time.perf_counter() - t, 3)
                    if row["weights"]:
                        row["weights_bytes"] = _folder_bytes(row["weights"])
                        self.last_weights_bytes = row["weights_bytes"]
        except Exception as ex:          # a record must never stop the learner
            row["error"] = repr(ex)
            record_failed(self, "spike_event", ex)
        self.events.add(row)

    def replay_pass(self, step: int, rr) -> None:
        """Record (c)'s line of one replay pass. A pass whose record failed in the learner (rr["record"] None) is a
        line naming that, so every pass the trainer ran has its line."""
        t0 = time.perf_counter()
        try:
            if rr and "record" in rr:
                rec = rr["record"]
                if rec is None:
                    self.passes.add(dict(step=int(step), record_failed=True,
                                         why="the learner's pass record failed; counted in the status file's failures"))
                else:
                    self.passes.add(dict(step=int(step), **rec))
        except Exception as ex:
            record_failed(self, "replay_pass_line", ex)
        self.cost_s += time.perf_counter() - t0

    def status(self) -> dict:
        model = self._model() if self._model is not None else None
        return dict(time=time.time(), flushes=self.flushes, steps=self.steps.status(),
                    replay_passes=self.passes.status(),
                    spike=dict(fired=list(self.spike.fired), suppressed=self.spike.suppressed,
                               enabled=self.spike.enabled, failures=self.spike.failures, error=self.spike.error,
                               crossings=None if self.spike.crossings is None else self.spike.crossings.status()),
                    events=self.events.status(), refused_events=self.refused_events,
                    refused_weights=self.refused_weights, last_weights_bytes=self.last_weights_bytes,
                    failures=dict(learner=(failures_of(model) if model is not None else {}),
                                  writer=failures_of(self), spike=failures_of(self.spike)),
                    cost_s=round(self.cost_s, 3))

    def flush(self) -> None:
        """Every writer's held lines to disk, then the status file (status_<pid>.json): the trainer calls this at every
        stats record (train_mimo's flush_stats, before the record is written) and close() calls it, so a stop that
        ends the process with no close loses at most the lines added after the last stats record. Never raises."""
        t0 = time.perf_counter()
        self.flushes += 1
        for w in (self.steps, self.passes, self.events, self.spike.crossings):
            if w is not None:
                try:
                    w.flush()
                except Exception as ex:
                    record_failed(self, "flush", ex)
        try:
            _atomic_json(os.path.join(self.folder, "status_%d.json" % self.pid), self.status())
        except Exception as ex:
            record_failed(self, "status", ex)
        self.cost_s += time.perf_counter() - t0

    def close(self) -> None:
        self.flush()


_SPIKE_CHECKPOINTS = None


def spike_checkpoint_manager(config, folder: str):
    """The checkpoint manager of the spike weights (the review's NOTE-5): lem.training.checkpoint.CheckpointManager with
    its cleanup after a save (_cleanup_old) doing nothing. That cleanup is the one rule of a save that removes other
    snapshots: it keeps the configuration's keep_last_n newest ckpt_step_* of the folder (10, lem/master_config.yaml),
    or hands the folder to a registered retention policy (lem/training/retention.py) when one owns it. A spike folder
    keeps every snapshot it is given; the snapshots of one launch are at least SPIKE_MIN_GAP_STEPS model steps apart,
    so no save of a launch replaces another (the manager's same-step replacement), and each launch has its own folder
    (ComparisonRecords.weights_folder)."""
    global _SPIKE_CHECKPOINTS
    if _SPIKE_CHECKPOINTS is None:
        from lem.training.checkpoint import CheckpointManager

        class SpikeCheckpoints(CheckpointManager):
            def _cleanup_old(self, current_step: int) -> None:
                return None
        _SPIKE_CHECKPOINTS = SpikeCheckpoints
    return _SPIKE_CHECKPOINTS(config, folder)


def snapshot_weights(model, step: int, folder: str):
    """The learner's checkpoint after model step `step`'s update, in `folder` (record d; the trainer passes
    ComparisonRecords.weights_folder): LemLite.save_checkpoint with its checkpoint manager replaced, for this one save,
    by spike_checkpoint_manager(folder) and no episode companion, so the run's own checkpoint folder, its
    checkpoints.json and any resume are untouched and no retention rule removes a spike snapshot. Returns the
    checkpoint's path."""
    held = getattr(model, "checkpoints", None)
    model.checkpoints = spike_checkpoint_manager(model.config, folder)
    try:
        return model.save_checkpoint(int(step))
    finally:
        model.checkpoints = held
