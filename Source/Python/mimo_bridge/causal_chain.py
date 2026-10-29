# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The causal chain, recorded: intention -> prediction -> predicted wellness -> action -> result.

Addendum v2, section 19 item 1, and docs/gain_of_function_v3/DESIGN.md D13 (--causal-chain on).
The model (LemLite) keeps each planning event's record in `last_plan_record`: the goal it aimed
at, what its first action was predicted to do, the approximate evaluator's predicted wellness for
that action and for four alternatives scored from the same captured state, and the action's rank
among them. This module is the trainer's half:

  AT THE LATCH (the step the trainer hands a new order to the body), for EVERY latched order,
  whoever authored it: the model predicts, before the order is sent, the next moment and the
  order's end (LemLite.predict_before_execution, learning route: raw tokens). A planning event
  of the same step is stamped with whether its first action reached the body (reached_body),
  the order's serial number and how far the order actually sent was from the order the planner
  imagined sending (plan_exec_distance_cm; 0 = the body got what was imagined). When the worlds
  change the order before it acts (the motor interface's limb fixes, limb_target_control.py:
  calf and chest goals on the parent, arm rows projected into reach), the trainer hands over the
  order the worlds recorded after the burst (on_executed) and the distance is taken against that,
  the order the body received.

  ONE MOMENT LATER and AT THE ORDER'S END, each prediction is graded against the raw tokens the
  body then produced, per sense, beside persistence (the error of predicting that nothing
  changes), and the predicted change of the approximate evaluator's wellness sum is set beside
  the grounded change of the same five terms. A planning event's own depth-1 prediction (acting
  route: the top-down-gated tokens) is graded at the next moment too, so the acting route and
  the learning route can be told apart.

  A prediction is never paired with a moment from a different body history: the model's reset
  counter (_transition_epoch) must match, the body count must match, and clear() drops every
  pending event at a reset or teleport.

Each resolved event is one line of <run>/causal_chain.jsonl (about 1 KB). Past 50 MB the file is
renamed aside to causal_chain_<YYYYmmdd_HHMMSS>.jsonl and a new one begins, so A READER OF THE
RECORD MUST READ THE RENAMED FILES TOO (oldest first by the time in their names, then
causal_chain.jsonl). window_summary() gives the `causal_chain` block of the mimo record.

THE PLANNER'S CHOICE AMONG ITS ALTERNATIVES (review of b1, MAJOR 3, 2026-09-23). Each planning
event carries, per body, the chosen plan's predicted wellness sum minus each alternative's, the
spread of the five candidates, and the outcome against the evaluator's measured tie band (TIE_BAND):
best, tied or behind. A gap smaller than the band is never counted as the chosen plan being best.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

log = logging.getLogger("mimo_bridge.causal_chain")

MAX_BYTES = 50 * 2 ** 20
# the five terms the approximate evaluator predicts, in its output order
from lem.model.wellness import APPROXIMATE_TERMS  # noqa: E402

# A RENAME THAT A READER BLOCKS IS RETRIED, NOT DROPPED (review MINOR 7). On Windows os.replace
# fails while any other process holds the file open (WinError 5, measured by ops-storage); the
# first version then dropped the event line with one failure count. The waits are the tape
# writer's (mimo_bridge/event_tape.py _REPLACE_WAITS_S: 1.275 s in total, 7.6 times the 167 ms the
# archiver held a file on the measured occasion). If the rename still fails, the line is appended
# to the current file anyway (it grows past the limit until the next attempt) and the window counts
# rotation_failures.
from mimo_bridge.event_tape import _REPLACE_WAITS_S as ROTATE_WAITS_S  # noqa: E402

# THE TIE BAND of the approximate evaluator (Judge B) on a DIFFERENCE between two candidate orders
# from the same moment, on its five-term sum (0 to 5 scale): a predicted gap smaller than this is
# not resolved and counts as a tie (review MAJOR 3). None = unmeasured: the chosen-best and tied
# shares are then not reported (the gaps and the spread still are). TIE_BAND_MEASUREMENT says how
# and where it was measured.
#
# Measured 2026-09-23 on ckpt_step_344400 (H84 segment 31) from snapshot-restored twins: 24 moments
# of the checkpoint's own behaviour (8 bodies x 3), five candidate orders at each (the model's own,
# hold, three babble-sized perturbations; the eyes held), each one predicted by Judge B on the state
# transition() gives and then sent for one model step from the restored snapshot and scored by the
# grounded formulas. Over the 240 same-moment pairs, Judge B's error on the DIFFERENCE between two
# orders' sums was 0.023 at the median, at most 0.073 in 9 of 10 pairs (the band) and 0.172 at
# worst. The realised differences were 0.022 at the median; Judge B's predicted differences were
# 0.0014 at the median and at most 0.0039 in 9 of 10 pairs, every one inside the band, and its sign
# matched the realised sign in 130 of 240 pairs (a coin matches 120). At this checkpoint the
# evaluator does not resolve which of the planner's candidates is better, so planning events are
# expected to read 'tied'. Source: docs/gain_of_function_v3/measurements/
# tie_band_twins_ckpt_step_344400.json (every moment's predicted and realised sums).
TIE_BAND: Optional[float] = 0.0734
TIE_BAND_MEASUREMENT: dict = {
    "date": "2026-09-23",
    "checkpoint": "H84_fresh_drivers_20260918_seg31_toy_and_push/checkpoints/ckpt_step_344400",
    "method": "snapshot-restored twins: 24 moments x 5 candidate orders, one model step each; "
              "band = 90th percentile of |predicted difference - realised difference| over the "
              "240 same-moment pairs, Judge B's five-term sum (0 to 5 scale)",
    "abs_error_on_difference": {"median": 0.0226, "p90": 0.0734, "max": 0.1724},
    "abs_predicted_difference": {"median": 0.0014, "p90": 0.0039},
    "abs_realised_difference": {"median": 0.0225, "p90": 0.0743},
    "sign_agreement_pairs": [130, 240],
    "source": "docs/gain_of_function_v3/measurements/tie_band_twins_ckpt_step_344400.json",
}


def rank_of_chosen(chosen: torch.Tensor, alternatives: torch.Tensor,
                   tie_band: float = 0.0) -> torch.Tensor:
    """[B] the chosen action's rank by predicted wellness among itself and the alternatives.

    chosen: [B] predicted wellness sums; alternatives: [K, B]. 1 = no alternative was predicted
    higher by more than tie_band (ties count for the chosen action), K + 1 = every alternative
    was. tie_band 0 is the strict rank (any higher prediction counts, however small)."""
    return 1 + (alternatives > chosen.unsqueeze(0) + float(tie_band)).sum(dim=0)


def plan_gaps(pred_wellness: dict, names) -> Optional[dict]:
    """Per body: the chosen plan's predicted wellness sum minus each alternative's (positive = the
    chosen was predicted better), minus the best alternative's, and the spread of all candidates
    (largest minus smallest sum). pred_wellness: {"chosen": [B], name: [B], ...}. None when a
    body has no alternative."""
    ch = pred_wellness.get("chosen") or []
    alts = [n for n in (names or []) if pred_wellness.get(n) is not None]
    if not ch or not alts:
        return None
    gap = {n: [None if c is None or a is None else round(float(c) - float(a), 6)
               for c, a in zip(ch, pred_wellness[n])] for n in alts}
    best, spread = [], []
    for b, c in enumerate(ch):
        vals = [pred_wellness[n][b] for n in alts
                if b < len(pred_wellness[n]) and pred_wellness[n][b] is not None]
        if c is None or not vals:
            best.append(None)
            spread.append(None)
            continue
        best.append(round(float(c) - max(vals), 6))
        spread.append(round(max(vals + [c]) - min(vals + [c]), 6))
    return {"gap": gap, "gap_to_best_alt": best, "candidate_spread": spread}


def outcome(gap_to_best: Optional[float], tie_band: Optional[float]) -> Optional[str]:
    """'best' when the chosen plan beat every alternative by more than the tie band, 'behind'
    when an alternative beat it by more than the band, 'tied' otherwise; None when unmeasured."""
    if gap_to_best is None or tie_band is None:
        return None
    if gap_to_best > tie_band:
        return "best"
    if gap_to_best < -tie_band:
        return "behind"
    return "tied"


def _r(x, nd=5):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _sense_mse(pred: Dict[str, torch.Tensor], actual: Dict[str, torch.Tensor], senses) -> Dict[str, torch.Tensor]:
    """{sense: [B]} mean squared error per body, over the token's width."""
    out = {}
    for m in senses:
        if m in pred and m in actual and pred[m].shape == actual[m].shape:
            out[m] = (pred[m].float() - actual[m].float()).pow(2).mean(-1)
    return out


def grounded_terms(model) -> Optional[torch.Tensor]:
    """[B, 5] the grounded values of the approximate evaluator's five terms, this step, in its
    output order (APPROXIMATE_TERMS)."""
    g = getattr(model, "last_wellness_grounded", None)
    if not g:
        return None
    cols = [g[n].float().reshape(-1) for n in APPROXIMATE_TERMS if n in g]
    if len(cols) != len(APPROXIMATE_TERMS):
        return None
    return torch.stack(cols, dim=-1)


def grounded_five(model) -> Optional[torch.Tensor]:
    """[B] the grounded wellness sum over the approximate evaluator's five terms, this step."""
    t = grounded_terms(model)
    return None if t is None else t.sum(-1)


class _Window:
    def __init__(self):
        self.events = 0
        self.planner_events = 0
        self.latch_predictions = 0
        self.dropped = 0
        self.failures = 0
        self.rotation_failures = 0
        self.reached = []
        self.exec_cm = []            # per reached (event, body): imagined vs sent order, cm
        self.exec_deg = []           # the same, turn in degrees
        self.exec_against_ledger = 0     # reached planning events whose distance was taken against the
                                         # order the worlds recorded (on_executed), not the order as sent
        # per author of the latched order: the order's-end forecast error and persistence, per sense
        # ({source: {sense: [sum, n]}}), so the planner's own orders can be read apart
        self.err_by_source = {"pred_latch": {}, "persist_latch": {}}
        self.clamp = []              # per (event, body): share of the planned step the clamp removed
        self.rank = []
        self.rank_resolved = []
        self.outcome = []            # per (event, body): 'best' / 'tied' / 'behind'
        self.gap_best = []
        self.spread = []
        self.closest = []
        self.pw_chosen = []
        self.pw_best_alt = []
        self.alternatives_n = None
        self.plan_record_failures_total = None
        self.err = {"pred_next": {}, "persist_next": {}, "pred_latch": {}, "persist_latch": {},
                    "plan_next": {}}
        self.wchg = {"pred_next": [], "real_next": [], "pred_latch": [], "real_latch": []}
        # per term of the approximate evaluator, over the order's tenure: predicted and realised
        # change, one pair per (order, body)
        self.tchg = {n: ([], []) for n in APPROXIMATE_TERMS}

    def add_err(self, key, per_sense: Dict[str, torch.Tensor], source: Optional[str] = None):
        for m, v in per_sense.items():
            s = self.err[key].setdefault(m, [0.0, 0])
            s[0] += float(v.sum())
            s[1] += int(v.numel())
            if source is not None and key in self.err_by_source:
                t = self.err_by_source[key].setdefault(str(source), {}).setdefault(m, [0.0, 0])
                t[0] += float(v.sum())
                t[1] += int(v.numel())


class CausalChainRecorder:
    """The trainer's half of the causal chain. hold = --motor-hold (moments per order)."""

    def __init__(self, out_dir: Optional[str], hold: int, max_bytes: int = MAX_BYTES,
                 strict: bool = False, tie_band: Optional[float] = "measured",
                 rotate_waits=ROTATE_WAITS_S):
        """tie_band: the evaluator's tie band on the sum (default: the measured TIE_BAND);
        rotate_waits: the rename retry's waits in seconds (tests pass short ones)."""
        self.path = None if out_dir is None else os.path.join(out_dir, "causal_chain.jsonl")
        self.hold = max(1, int(hold))
        self.max_bytes = int(max_bytes)
        self.strict = bool(strict)          # tests: raise instead of counting a failure
        self.tie_band = TIE_BAND if tie_band == "measured" else (
            None if tie_band is None else float(tie_band))
        self.rotate_waits = tuple(rotate_waits)
        self.pending: List[dict] = []
        self.win = _Window()
        # the reached planning event of the latest latch, until the trainer hands over the order the
        # worlds recorded for it (on_executed) or the next latch replaces it
        self._exec_pending: Optional[dict] = None

    # ------------------------------------------------------------------ plumbing
    def _fail(self, where: str):
        if self.strict:
            raise
        self.win.failures += 1
        import traceback
        log.warning("causal chain: %s failed (%d this window): %s", where, self.win.failures,
                    traceback.format_exc().strip().splitlines()[-1])

    def _rotate(self) -> None:
        """Rename the full file aside (causal_chain_<time>.jsonl; a reader of the record must read
        those too), retrying while a reader holds it; on a final refusal the file stays and the
        window counts rotation_failures (the next line tries again)."""
        stem = self.path[:-len(".jsonl")] + time.strftime("_%Y%m%d_%H%M%S")
        aside, k = stem + ".jsonl", 1
        while os.path.exists(aside):         # two renames in one second never overwrite
            aside, k = "%s_%d.jsonl" % (stem, k), k + 1
        for wait in self.rotate_waits + (None,):
            try:
                os.replace(self.path, aside)
                log.info("causal chain: %s passed %d MB, renamed to %s", self.path,
                         self.max_bytes // 2 ** 20, aside)
                return
            except PermissionError:
                if wait is None:
                    break
                time.sleep(wait)
        self.win.rotation_failures += 1
        log.warning("causal chain: %s could not be renamed aside (a reader holds it); the line is "
                    "appended to it and the rename is tried again with the next line", self.path)

    def _write(self, rec: dict):
        if self.path is None:
            return
        line = json.dumps(rec, separators=(",", ":")) + "\n"
        try:
            if os.path.exists(self.path) and os.path.getsize(self.path) + len(line) > self.max_bytes:
                self._rotate()
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError:
            self._fail("writing causal_chain.jsonl")

    def clear(self, reason: str = "reset"):
        """A reset or teleport: nothing pending may pair across it."""
        self.win.dropped += len(self.pending)
        self.pending = []
        self._exec_pending = None

    # ------------------------------------------------------------------ the latch
    def on_latch(self, model, step: int, latched: bool, order_seq: int, send_motor,
                 gaze_out, source: Optional[str]):
        """After the trainer's latch decision, before the burst. Every step; acts on latch
        steps and on steps with a planning event."""
        try:
            self._on_latch(model, step, latched, order_seq, send_motor, gaze_out, source)
        except Exception:
            self._fail("the latch record at step %d" % step)

    def _on_latch(self, model, step, latched, order_seq, send_motor, gaze_out, source):
        # the model counts planning records that raised; carried into the block, never hidden
        self.win.plan_record_failures_total = int(getattr(model, "_causal_failures", 0) or 0)
        if latched:
            self._exec_pending = None        # a new order: an earlier event can no longer be revised
        plan = getattr(model, "last_plan_record", None)
        plan = plan if (plan and plan.get("step") == step and "error" not in plan) else None
        reached = bool(latched and plan is not None and source == "planner")
        exec_gap = None
        li = None
        if plan is not None:
            self.win.planner_events += 1
            self.win.reached.append(1.0 if reached else 0.0)
            if reached:
                exec_gap = self._plan_exec_distance(model, plan, send_motor)
                if exec_gap is not None:
                    self.win.exec_cm.extend(exec_gap["cm"])
                    self.win.exec_deg.extend(exec_gap["deg"])
            li = getattr(model, "last_intention", None)
            if isinstance(li, dict) and li.get("step") == step:
                li["reached_body"] = reached
                li["order_seq"] = int(order_seq) if reached else None
                li["plan_exec_distance_cm"] = (None if exec_gap is None
                                               else _r(np.mean(exec_gap["cm"]), 3))
            else:
                li = None
            self._plan_window(plan)
        if reached and exec_gap is not None:
            self._exec_pending = {"step": int(step), "plan": plan, "gap": exec_gap, "intention": li,
                                  "window": self.win, "event": None}
        if not latched:
            if plan is not None:
                self._write({"type": "plan", "step": int(step), "reached_body": False,
                             "plan": self._plan_json(plan)})
            return
        pred = model.predict_before_execution(send_motor, gaze_out, depth=self.hold)
        if pred is None:
            return
        self.win.latch_predictions += 1
        g_now = grounded_five(model)
        t_now = grounded_terms(model)
        event = {
            "step": int(step), "order_seq": int(order_seq), "source": source,
            "epoch": pred["epoch"], "B": pred["B"], "pred": pred,
            "wellness_now": None if g_now is None else g_now.detach(),
            "terms_now": None if t_now is None else t_now.detach(),
            "plan": plan, "reached_body": reached, "plan_exec": exec_gap,
            "next": None}
        self.pending.append(event)
        if self._exec_pending is not None and self._exec_pending["step"] == int(step):
            self._exec_pending["event"] = event

    def on_executed(self, model, step: int, executed) -> None:
        """After the burst, when the worlds changed the order before it acted (limb_target_control;
        the trainer calls this only then): the plan-to-executed distance of this step's reached
        planning event is taken again against the order the worlds recorded, the order the body
        received, in the window, the planning record and the event's line."""
        try:
            self._on_executed(model, step, executed)
        except Exception:
            self._fail("the executed order at step %d" % step)

    def _on_executed(self, model, step, executed):
        p = self._exec_pending
        if p is None or p["step"] != int(step):
            return
        self._exec_pending = None
        gap = self._plan_exec_distance(model, p["plan"], executed)
        if gap is None:
            return
        w = p["window"]
        n = len(p["gap"]["cm"])
        if w is self.win and len(w.exec_cm) >= n and len(w.exec_deg) >= n:
            # the sent order's distances were the last ones added (nothing is added between the latch
            # and the end of its burst); a window already summarised keeps what it reported
            del w.exec_cm[-n:]
            del w.exec_deg[-n:]
            w.exec_cm.extend(gap["cm"])
            w.exec_deg.extend(gap["deg"])
            w.exec_against_ledger += 1
        li = p["intention"]
        if isinstance(li, dict) and li.get("step") == int(step):
            li["plan_exec_distance_cm"] = _r(np.mean(gap["cm"]), 3)
        if p["event"] is not None:
            p["event"]["plan_exec"] = dict(gap, against="the order the worlds recorded (limb fixes)")

    @staticmethod
    def _plan_exec_distance(model, plan, send_motor) -> Optional[dict]:
        """How far the order the body was sent is from the order the planner imagined it would
        receive (review MINOR 2), per body over the body handles (the eyes left out): the mean
        distance between the two target positions in cm and the mean turn between the two target
        orientations in degrees. 0 = the body got exactly what the planner imagined (the smoother
        included, which the planner imagines too). A rest window that replaces the order, or a
        smoother the planner did not imagine, shows here. None without the imagined order."""
        exf = plan.get("exec_form")
        if not torch.is_tensor(exf):
            return None
        exf = exf.detach().float().cpu()
        sent = torch.as_tensor(np.asarray(send_motor, dtype=np.float32))
        if sent.shape != exf.shape:
            return None
        from lem.model.efference import rotation_change_deg
        gi = int(getattr(model, "GAZE_IDX", -1))
        body = [i for i in range(exf.shape[1]) if i != gi]
        cm = (sent[:, body, 0:3] - exf[:, body, 0:3]).norm(dim=-1).mean(-1)
        qa = exf[:, body, 3:7] / exf[:, body, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        qb = sent[:, body, 3:7] / sent[:, body, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        deg = rotation_change_deg(qa, qb).norm(dim=-1).mean(-1)
        return {"cm": [round(float(v), 4) for v in cm.tolist()],
                "deg": [round(float(v), 4) for v in deg.tolist()]}

    def _gaps(self, plan) -> Optional[dict]:
        """The per-body gaps, the spread, the resolved rank and the outcome of one planning event
        (plan_gaps + the tie band); None when the event has no alternatives."""
        pw = plan.get("pred_wellness") or {}
        names = plan.get("alternatives") or []
        g = plan_gaps(pw, names)
        if g is None:
            return None
        band = self.tie_band
        g["tie_band"] = band
        g["outcome"] = [outcome(v, band) for v in g["gap_to_best_alt"]]
        if band is not None:
            ch = pw.get("chosen") or []
            alts = [n for n in names if pw.get(n) is not None]
            g["chosen_rank_resolved"] = [
                None if c is None else 1 + sum(1 for n in alts if b < len(pw[n])
                                               and pw[n][b] is not None
                                               and float(pw[n][b]) > float(c) + band)
                for b, c in enumerate(ch)]
        return g

    def _plan_window(self, plan):
        pw = plan.get("pred_wellness") or {}
        names = plan.get("alternatives") or []
        self.win.alternatives_n = len(names)
        ch = pw.get("chosen") or []
        alts = [pw.get(n) for n in names if pw.get(n) is not None]
        for b, c in enumerate(ch):
            vals = [a[b] for a in alts if b < len(a)]
            if not vals or c is None:
                continue
            self.win.pw_chosen.append(c)
            self.win.pw_best_alt.append(max(vals))
        for r in plan.get("chosen_rank") or []:
            self.win.rank.append(float(r))
        g = self._gaps(plan)
        if g is not None:
            self.win.gap_best += [v for v in g["gap_to_best_alt"] if v is not None]
            self.win.spread += [v for v in g["candidate_spread"] if v is not None]
            self.win.outcome += [o for o in g["outcome"] if o is not None]
            self.win.rank_resolved += [float(r) for r in (g.get("chosen_rank_resolved") or [])
                                       if r is not None]
        for v in plan.get("reach_clamp_removed_share") or []:
            if v is not None:
                self.win.clamp.append(float(v))
        gd = plan.get("goal_distance") or {}
        gch = gd.get("chosen") or []
        galt = [gd.get(n) for n in names if gd.get(n) is not None]
        for b, c in enumerate(gch):
            vals = [a[b] for a in galt if b < len(a)]
            if vals and c is not None:
                self.win.closest.append(1.0 if c <= min(vals) else 0.0)

    def _plan_json(self, plan) -> dict:
        """The plan's JSON line: every field but the tensors, plus its gaps, spread and outcome."""
        out = {k: v for k, v in plan.items() if k not in ("pred_tokens", "exec_form", "step")}
        g = self._gaps(plan)
        if g is not None:
            out["pred_wellness_gaps"] = g
        return out

    # ------------------------------------------------------------------ resolution
    def after_step(self, model, step: int):
        """Right after model.step: grade the pending predictions whose moment has come."""
        try:
            self._after_step(model, step)
        except Exception:
            self.pending = []
            self._fail("resolving at step %d" % step)

    def _after_step(self, model, step):
        if not self.pending:
            return
        actual = getattr(model, "last_tokens", None)
        epoch = int(getattr(model, "_transition_epoch", -1))
        mask = getattr(model, "_last_source_mask", None) or {}
        keep = []
        for ev in self.pending:
            age = int(step) - ev["step"]
            if (actual is None or ev["epoch"] != epoch
                    or any(v.shape[0] != ev["B"] for v in actual.values())):
                self.win.dropped += 1
                continue
            senses = [m for m in ev["pred"]["tokens_now"]
                      if str(mask.get(m, "real")) == "real" and m in actual]
            if age == 1:
                ev["next"] = self._grade(model, ev, 1, "next", actual, senses)
                if ev["plan"] is not None and ev["reached_body"]:
                    pe = _sense_mse(ev["plan"]["pred_tokens"], actual, senses)
                    self.win.add_err("plan_next", pe)
                    ev["next"]["plan_pred_err"] = {m: _r(v.mean()) for m, v in pe.items()}
            if age >= self.hold:
                latch = self._grade(model, ev, self.hold, "latch", actual, senses)
                self._finish(ev, latch)
                continue
            keep.append(ev)
        self.pending = keep

    def _grade(self, model, ev, depth, tag, actual, senses) -> dict:
        """Grade the depth-`depth` prediction against the tokens observed now, per sense,
        beside persistence; and the predicted change of the approximate evaluator's five-term
        sum beside the grounded change of the same five terms. tag: "next" or "latch"."""
        d = ev["pred"]["depths"].get(depth)
        now = ev["pred"]["tokens_now"]
        pe = _sense_mse(d["tokens"], actual, senses)
        pp = _sense_mse(now, actual, senses)
        # at the order's end also per author (planner, reflex_noise, reflex): the planner's own orders
        source = (ev.get("source") or "unknown") if tag == "latch" else None
        self.win.add_err("pred_" + tag, pe, source=source)
        self.win.add_err("persist_" + tag, pp, source=source)
        pred_chg = (d["judge"].sum(-1) - ev["pred"]["judge_now"].sum(-1)).cpu()
        g = grounded_five(model)
        real_chg = (None if g is None or ev["wellness_now"] is None
                    else (g.detach() - ev["wellness_now"]).cpu())
        if real_chg is not None:
            self.win.wchg["pred_" + tag] += [float(v) for v in pred_chg.tolist()]
            self.win.wchg["real_" + tag] += [float(v) for v in real_chg.tolist()]
        # PER TERM (b3-wellness): which of the five the evaluator predicts moved as predicted.
        # These are the only wellness channels readable this way; the others (controllability,
        # learning progress, consequence, drives) have no prediction to compare against.
        # (at the order's end only: one moment in is inside the approximate evaluator's noise and
        # would double every line of causal_chain.jsonl for it)
        terms = None
        tn = grounded_terms(model) if tag == "latch" else None
        t0 = ev.get("terms_now")
        if tn is not None and t0 is not None and tn.shape == t0.shape:
            p_t = (d["judge"].float() - ev["pred"]["judge_now"].float()).cpu()     # [B, 5]
            r_t = (tn.detach().float() - t0.float()).cpu()
            terms = {n: [_r(p_t[:, i].mean()), _r(r_t[:, i].mean())]
                     for i, n in enumerate(APPROXIMATE_TERMS)}
            if tag == "latch":
                for i, n in enumerate(APPROXIMATE_TERMS):
                    self.win.tchg[n][0].extend(float(v) for v in p_t[:, i].tolist())
                    self.win.tchg[n][1].extend(float(v) for v in r_t[:, i].tolist())
        out = {"pred_err": {m: _r(v.mean()) for m, v in pe.items()},
               "persist_err": {m: _r(v.mean()) for m, v in pp.items()},
               "pred_wellness": [_r(v) for v in d["judge"].mean(0).tolist()],
               "predicted_wellness_change": _r(pred_chg.mean()),
               "realized_wellness_change": None if real_chg is None else _r(real_chg.mean())}
        if terms is not None:
            out["wellness_terms_pred_real"] = terms
        return out

    def _finish(self, ev, latch):
        self.win.events += 1
        rec = {"type": "latch", "step": ev["step"], "order_seq": ev["order_seq"],
               "source": ev["source"], "bodies": ev["B"], "hold": self.hold,
               "route": "learning: raw tokens, transition() with normalised head output",
               "judge_now": [_r(v) for v in ev["pred"]["judge_now"].mean(0).tolist()],
               "next": ev["next"], "latch": latch}
        if ev["plan"] is not None:
            rec["plan"] = self._plan_json(ev["plan"])
            rec["reached_body"] = ev["reached_body"]
            rec["plan_exec_distance"] = ev.get("plan_exec")
        self._write(rec)

    # ------------------------------------------------------------------ the window
    def window_summary(self) -> dict:
        """The `causal_chain` block (DESIGN.md section 3); a fresh window starts.

        events / planner_events / latch_predictions: counts in the window (orders whose end
          was graded, planning events, orders predicted before they were sent).
        THE PLANNER DOES NOT AIM AT PREDICTED WELLNESS TODAY (it descends on distance to a
          remembered moment), so the next four measure whether the two agree, not whether it
          tried; no goal line exists for them until it does.
        tie_band: the approximate evaluator's measured resolution on a DIFFERENCE between two
          candidate orders from the same moment (TIE_BAND_MEASUREMENT); a gap smaller than it is
          a tie. None = unmeasured, and then the three shares below are None.
        chosen_best_share / chosen_tied_share / chosen_behind_share: shares of (planning event,
          body) pairs where the chosen plan's predicted wellness beat the best of the four
          alternatives by more than tie_band / stayed within it / fell below it by more than it
          (the three add to 1). A rising best share read on gaps inside the band would be the
          evaluator's noise, which is why ties are reported and never counted as best.
        gap_to_best_alt_mean: the chosen plan's predicted wellness sum minus the best
          alternative's (0 to 5 scale; positive = the chosen was predicted better).
        candidate_spread_mean: the largest minus the smallest of the five candidates' sums: how
          far apart the evaluator placed the candidates at all.
        chosen_rank_mean: the strict rank (1 = no alternative predicted higher by any amount,
          5 = all four were), which counts gaps inside the tie band; chosen_rank_resolved_mean
          counts an alternative only when it was predicted higher by more than tie_band.
        chosen_closest_share: the share where the chosen plan's predicted next state was the
          closest of the five to the planner's goal (its own objective).
        pred_wellness_chosen / pred_wellness_best_alt: the approximate evaluator's five-term
          sum (0 to 5) for the chosen action and for the best alternative, window means.
        predicted_wellness_change / realized_wellness_change: over the order's tenure, the
          evaluator's predicted change of that sum and the grounded change of the same five
          terms; wellness_change_corr their correlation across (order, body) pairs (-1 to 1;
          None under 3 pairs). *_next: the same one moment after the latch.
        pred_err_* / persist_err_*: per sense, the mean squared distance between the predicted
          and the actual raw token, and between the present token and the actual one (the
          error of predicting that nothing changes). A prediction worth having is below its
          persistence line. next = one moment, latch = the order's end.
        plan_pred_err_next: the planner's own depth-1 prediction (top-down-gated tokens, the
          acting route) against the next moment's raw tokens, for plans that reached the body.
        reached_body_share: planning events whose first action was latched to the body.
        plan_exec_distance_cm / plan_exec_turn_deg: for plans that reached the body, how far the
          order the body received is from the order the planner imagined it would receive, mean
          over the body handles (0 = the body got exactly what was imagined, the smoother
          included); replaces planner_share_of_executed, which returned the smoother's setting by
          construction. With the motor interface's limb fixes on, the order the body received is
          the one the worlds recorded after projecting it (calf and chest goals on the parent, arm
          rows moved into reach): plan_exec_against_ledger counts the events measured that way (the
          rest were measured against the order as sent, which is then the order received).
          reach_clamp_removed_share: share of each plan's position step that the
          per-axis reach clamp removed (0 = none, 1 = all of it).
        pred_err_latch_by_source / persist_err_latch_by_source: pred_err_latch and persist_err_latch
          split by who wrote the order (planner; reflex_noise = the controller with babble; reflex =
          the controller alone): whether the model still foresees what the planner's own orders do.
        dropped_at_reset: predictions discarded because a reset or teleport came before their
          moment. failures: recorder errors this window; rotation_failures: renames of the full
          file that a reader blocked past the retries (the line was kept); plan_record_failures_total:
          the model's planning records that raised, since the process started."""
        w = self.win

        def mean(xs, nd=4):
            return None if not xs else round(float(np.mean(xs)), nd)

        def share(label):
            return (None if self.tie_band is None or not w.outcome
                    else round(sum(1 for o in w.outcome if o == label) / len(w.outcome), 4))

        def errs(key):
            return {m: round(s[0] / s[1], 6) for m, s in w.err[key].items() if s[1] > 0} or None

        def errs_by_source(key):
            return {src: {m: round(s[0] / s[1], 6) for m, s in per.items() if s[1] > 0}
                    for src, per in w.err_by_source[key].items()} or None

        def corr(a, b):
            if len(a) < 3:
                return None
            a, b = np.asarray(a, float), np.asarray(b, float)
            if a.std() < 1e-12 or b.std() < 1e-12:
                return None
            return round(float(np.corrcoef(a, b)[0, 1]), 4)

        out = {
            "events": w.events, "planner_events": w.planner_events,
            "latch_predictions": w.latch_predictions,
            "tie_band": self.tie_band,
            "chosen_best_share": share("best"), "chosen_tied_share": share("tied"),
            "chosen_behind_share": share("behind"),
            "gap_to_best_alt_mean": mean(w.gap_best, 5),
            "candidate_spread_mean": mean(w.spread, 5),
            "chosen_rank_mean": mean(w.rank, 3),
            "chosen_rank_resolved_mean": mean(w.rank_resolved, 3),
            "alternatives_n": w.alternatives_n,
            "chosen_closest_share": mean(w.closest),
            "pred_wellness_chosen": mean(w.pw_chosen), "pred_wellness_best_alt": mean(w.pw_best_alt),
            "predicted_wellness_change": mean(w.wchg["pred_latch"], 5),
            "realized_wellness_change": mean(w.wchg["real_latch"], 5),
            "wellness_change_corr": corr(w.wchg["pred_latch"], w.wchg["real_latch"]),
            "predicted_wellness_change_next": mean(w.wchg["pred_next"], 5),
            "realized_wellness_change_next": mean(w.wchg["real_next"], 5),
            "wellness_change_corr_next": corr(w.wchg["pred_next"], w.wchg["real_next"]),
            "pred_err_next": errs("pred_next"), "persist_err_next": errs("persist_next"),
            "pred_err_latch": errs("pred_latch"), "persist_err_latch": errs("persist_latch"),
            "pred_err_latch_by_source": errs_by_source("pred_latch"),
            "persist_err_latch_by_source": errs_by_source("persist_latch"),
            "plan_pred_err_next": errs("plan_next"),
            "reached_body_share": mean(w.reached),
            "plan_exec_distance_cm": mean(w.exec_cm, 3), "plan_exec_turn_deg": mean(w.exec_deg, 3),
            "plan_exec_against_ledger": w.exec_against_ledger,
            "reach_clamp_removed_share": mean(w.clamp),
            "dropped_at_reset": w.dropped, "failures": w.failures,
            "rotation_failures": w.rotation_failures,
            "plan_record_failures_total": w.plan_record_failures_total,
            "hold": self.hold,
            # per term of the evaluator over the order's tenure: its predicted change, the
            # grounded change, and their correlation across (order, body) pairs (-1 to 1;
            # None under 3 pairs). The wellness channels the evaluator cannot predict are absent
            # by construction (see wellness_rank.planner_blind).
            "wellness_terms_latch": ({n: {"pred": mean(p, 5), "real": mean(r, 5), "corr": corr(p, r)}
                                      for n, (p, r) in w.tchg.items()}
                                     if any(p for p, _ in w.tchg.values()) else None),
        }
        self.win = _Window()
        return out
