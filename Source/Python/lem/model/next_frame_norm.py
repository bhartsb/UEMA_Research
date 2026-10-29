# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Each sense's next-moment lesson measured against "nothing changes" (--next-frame-norm persistence).

WHY THIS EXISTS (docs/gain_of_function_v3/DESIGN.md decision D12, addendum section 6 "keep gaze
prediction visible in the loss so larger sensory channels do not drown it out"). The delayed
next-frame lesson - S_t-1 plus the order A_t-1 predicts S_t - adds up each real sense's squared
error in token units, unweighted (mimo_bridge/lem_model.py, the nf_terms sum). A sense whose token
simply moves more from one moment to the next owns the lesson. MEASURED on H84 segment 31's own
record (161 report windows of 100 model steps x 8 bodies, replay_training_stats.jsonl,
prediction_audit and next_pe fields; b6 scratch seg_persistence.py): the error of predicting that
nothing changes (persistence) was audio 0.0839, touch 0.0248, vision 0.0090, proprioception
0.00393 per token dimension, and the lesson's error split audio 67 of every 100 parts, touch 21,
vision 9, proprioception 4. The model was no better than "nothing changes" on vision and
proprioception (model error / persistence error 1.045 and 1.029, where 1 = no better than
"nothing changes") while audio sat at 0.866: two thirds of the lesson's gradient went to audio.

WHAT IT DOES. Each sense's error is divided by that sense's own running persistence error P_s,
and the sum is multiplied by the mean of those divisors over the senses graded at that moment:

    lesson = P_mean x sum_s  error_s / max(P_s, P_mean / WEIGHT_CAP)

so every sense's weight is w_s = P_mean / max(P_s, floor): the pressure on each sense is its error
RELATIVE to "nothing changes", the same for every sense. The common factor P_mean keeps the
lesson's size where it was: when every sense sits at persistence (error_s = P_s, which is where
H84 is), the normalised lesson equals the plain sum exactly, so the lesson's share of the whole
loss (8.7 of every 100 parts on segment 31) does not jump because of a change of units. Without
it the lesson would read sum_s error_s / P_s - 3.86 on segment 31's four senses against the plain
sum's 0.109, 35 times larger - and would take most of the loss from VICReg and the imagination
chain; that would be a second change riding on this one, and it is not what D12 asks for. Stated
as a deviation from the literal wording of D12 (which says "divides each sense's error by its own
running persistence error" and names no common factor).

P_s is an exponential running average, corrected for starting at zero, of the batch-mean
persistence error the prediction audit already measures every step (mimo_bridge/
prediction_audit.paired_errors: the previous moment's token re-encoded with today's encoder, against
this moment's token), taken only when BOTH moments were real readings of that sense. A sense's
divisor at a step is the average BEFORE that step's own reading joins it (its first reading when
it has none yet). Two kinds of graded moment are left out of the lesson (weight 0) and counted:
one whose input moment was not a real reading of that sense (the vestibular sense's first graded
moment after the resume, predicted from the padded legacy vector: its error is the jump from a
stand-in to a reading, 0.95 against 0.044 over the next 50 real moments in b2's proof runs -
review-b2 minor 7), and one of a sense
with no running average and no reading yet (unscaled).

The same weights apply to the replay re-study of the same next-moment lesson (LemLite.
replay_update): on H84 replay runs every 2 live steps and its loss (0.111 window mean over 782
windows of segments 30-31) is as large as the live lesson's (0.109), so leaving it unweighted would
leave half of the next-moment training where it was. The imagination chain (lem/live_test.py,
23.6 of every 100 parts of segment 31's positive loss mass) is NOT weighted by this: its per-sense
sum lives in a file outside the model build's ownership (DESIGN section 5) and is reported open.

THE ENCODER'S SHARE (found and fixed 2026-09-23). With residual prediction on (H84), a prediction
is the head's output plus the CURRENT token, and the live lesson re-encodes the previous moment
with gradient, so part of the lesson's gradient runs into the encoder through that `+ current`
and pulls the previous moment's token toward the next one: a push on the encoder to stop the
token changing. A plain weight multiplied that push. MEASURED on ckpt_step_344400 (CPU, the
segment 31 world, 2 bodies, training, 60 steps, same seeds; b6 scratch run_abl_*.pt and
enc_compare.py): with only this flag on, the proprioception token's change between moments fell
to 1.3e-5 to 7.5e-5 (median per 10 steps) against 9e-5 to 1.8e-4 with it off, at the same body
movement (0.23 to 0.66 cm per step against 0.24 to 0.55); read on the same 60 frames, that
encoder's token moved x0.26 of the start encoder's, against x1.63 for the flags-off run - the
encoder had learned to report the body 6 times less. So the weight is applied by weighted_term:
the head and the core behind it (and the value of the term) carry the weight; the path into the
encoder through the residual `+ current` keeps the plain weight 1, as with the flag off. With
that, the same run's proprioception change read 9.3e-5 to 1.5e-4, the flags-off level, and on the
same 60 frames its encoder's token moved x2.11 of the start encoder's (x1.63 flags off): the
weight now reaches that encoder only through the core, which trains the token to carry more of
the body's change, not less (every chain flag on: x4.96 with this fix, x0.28 without it, x1.26
with the normalisation off). As a sense's token comes to change more, its own running
persistence error rises and its weight falls, so that direction limits itself.

Every constant below carries its measurement.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch

NEXT_FRAME_NORM_VERSION = "next_frame_norm_v1"
MODES = ("off", "persistence")

# THE RUNNING AVERAGE'S HORIZON, in model steps (alpha = 1 / horizon per step). MEASURED 2026-09-23
# on the live record of H84 segments 30 and 31 (782 windows of 100 steps x 8 bodies; b6 scratch
# seg_horizon.py -> seg_horizon.json): the running average was replayed over the windows at
# horizons of 500, 1,000, 2,000 and 5,000 steps and the resulting weight of each sense read. At
# 2,000 the largest move of any sense's weight within one 200-step goal-ring span was 9.4% (touch;
# audio 1.9%, vision 4.7%, proprioception 6.8%) against 26.8% at 500 and 16.0% at 1,000, while it
# still follows the world's slow changes (touch's weight ranged 0.80 to 2.41 over the two segments;
# 0.89 to 2.14 at 5,000). 2,000 steps is 100 s of body time, longer than one full caregiver cycle
# (60 s present + 15 s absent = 1,500 steps), and the same horizon as the wellness normaliser and
# learning progress's slow average (lem/model/wellness_channels.py).
HORIZON_STEPS = 2000

# THE MEASURED FLOOR, as the largest weight a sense may get. A sense's divisor is never taken
# below P_mean / WEIGHT_CAP, so a sense that has stopped changing (a frozen or blank input) cannot
# turn its last digits into most of the lesson. MEASURED 2026-09-23 on the same 782 live windows
# (seg_horizon.py, w_max_window): the largest weight any sense needed in any single 100-step window,
# mean over the four real senses of their persistence error divided by that sense's, was 30.70
# (proprioception at step 269,400 of segment 30: 0.00108 against a lesson mean of 0.0333; the
# largest for vision 7.26, touch 6.97, audio 0.54). A running average over 2,000 steps cannot reach
# a single window's extreme unless that condition lasts, so on the lineage's own record the cap
# never binds; it binds only on a sense quieter for 100 s than its quietest 5 s ever were. That
# does happen OFF the lived record: in the CPU re-runs of ckpt_step_344400 in fresh worlds the
# proprioception token changed 2e-4 per moment against 3.9e-3 live, and the cap held its weight on
# every step (b6 scratch abl_off_norm_fix.log, chain_on.log). Relative, not absolute, so it applies
# to a sense with no lived history (the vestibular sense's fresh encoder) the same way. Every step
# it binds is counted per sense (capped_steps).
WEIGHT_CAP = 30.7

# A NUMERICAL GUARD, not a modelling constant: when the mean persistence error of the graded
# senses is below this, every graded token is unchanged to float precision (two unit-scale tokens
# that agree to float32 resolution, a relative 6e-8, differ by a squared error near 4e-15; the
# quietest lived window of any sense was 1.08e-3, nine orders above). There is then no change to
# weigh and the lesson falls back to the plain sum (every weight 1).
ZERO_GUARD = 1e-12


class _PlainResidualGradient(torch.autograd.Function):
    """Identity on the prediction; on the way back, the residual input `current` receives
    (k - 1) x the prediction's gradient ON TOP of the gradient it already receives through the
    prediction head's own `+ current` (prediction_heads.PredictionHead residual). Total into
    `current`: k x the prediction's gradient. With k = 1 / w and the lesson term weighted by w,
    the path into the encoder that produced `current` carries the PLAIN (unweighted) gradient,
    while the head and the core carry the weighted one."""

    @staticmethod
    def forward(ctx, pred, current, k):
        ctx.k = float(k)
        return pred.clone()

    @staticmethod
    def backward(ctx, g):
        return g, g * (ctx.k - 1.0), None


def weighted_term(pred, current, target, weight: float, residual: bool):
    """weight x mean squared error of `pred` against `target`, with the residual input's share of
    the gradient kept at weight 1 (see _PlainResidualGradient and the module docstring, "THE
    ENCODER'S SHARE"). `current` is the token the prediction head added (residual prediction);
    with residual prediction off, or a weight of 0 or 1, it is the plain weighted error."""
    import torch.nn.functional as F
    w = float(weight)
    if w == 0.0:
        return F.mse_loss(pred, target) * 0.0
    if residual and current is not None and w != 1.0 and current.requires_grad:
        pred = _PlainResidualGradient.apply(pred, current.reshape(pred.shape), 1.0 / w)
    return F.mse_loss(pred, target) * w


class NextFrameNorm:
    """Running persistence error per sense NAME, the weights it gives, and the window's record.

    PUBLIC on the model (LemLite.nf_norm), so continuation_state.capture_learner, which serialises
    every _* and last_* attribute into the episode companion, never copies it: its state rides
    runtime_state.pt under "next_frame_norm", like the wellness normaliser's. Plain Python floats
    (float64) throughout, so no device read is added: the readings are the prediction audit's
    floats and the lesson's floats, both already read on the step."""

    def __init__(self, horizon: int = HORIZON_STEPS, cap: float = WEIGHT_CAP):
        if cap is None or not (cap > 1.0):
            raise ValueError("next_frame_norm: WEIGHT_CAP must be a measured number above 1")
        self.horizon = int(max(1, int(horizon)))
        self.alpha = 1.0 / float(self.horizon)
        self.cap = float(cap)
        # sense -> [running sum, running weight, readings]; the average is sum / weight
        self.stats: Dict[str, List[float]] = {}
        self._win_reset()

    # ---------------------------------------------------------------- the running average
    def scale(self, sense: str) -> Optional[float]:
        """The running persistence error of `sense` (token units, a mean squared error per token
        dimension), or None when it has no reading yet."""
        s = self.stats.get(sense)
        if not s or s[1] <= 0.0:
            return None
        return s[0] / s[1]

    def update(self, readings: Dict[str, float]) -> None:
        """Fold this step's persistence readings in (one per sense whose two moments were real)."""
        a = self.alpha
        for sense, v in (readings or {}).items():
            if v is None:
                continue
            v = float(v)
            if not math.isfinite(v) or v < 0.0:
                continue
            s = self.stats.setdefault(str(sense), [0.0, 0.0, 0])
            s[0] = (1.0 - a) * s[0] + a * v
            s[1] = (1.0 - a) * s[1] + a
            s[2] = int(s[2]) + 1

    def weights(self, senses: Iterable[str], readings: Optional[Dict[str, float]] = None,
                input_not_real: Iterable[str] = ()) -> Tuple[Dict[str, float], dict]:
        """The weight of each graded sense this step, and what it was made of.

        The divisor of a sense is its running average from BEFORE this step's reading, or this
        step's reading when it has no average yet; a sense with neither gets weight 0 (listed in
        info["unscaled"]). A sense named in input_not_real - its previous moment, the one the
        prediction was made FROM, was not a real reading (the padded legacy vestibular vector of
        the first moment after a resume, a stale camera) - gets weight 0 too: its error measures
        the jump from a stand-in to a reading, not a prediction (review-b2 minor 7). Returns
        ({sense: weight}, {"p_mean", "floor", "capped": [...], "unscaled": [...],
        "input_not_real": [...], "divisor": {sense: value}, "guard": bool})."""
        senses = list(senses)
        readings = readings or {}
        skip = [s for s in senses if s in set(input_not_real or ())]
        div: Dict[str, float] = {}
        unscaled: List[str] = []
        for s in senses:
            if s in skip:
                continue
            v = self.scale(s)
            if v is None:
                r = readings.get(s)
                v = float(r) if r is not None and math.isfinite(float(r)) and float(r) >= 0.0 else None
            if v is None:
                unscaled.append(s)
            else:
                div[s] = v
        info = {"p_mean": None, "floor": None, "capped": [], "unscaled": unscaled,
                "input_not_real": skip, "divisor": dict(div), "guard": False}
        if not div:
            return {s: 0.0 for s in senses}, info
        p_mean = sum(div.values()) / len(div)
        info["p_mean"] = p_mean
        if p_mean < ZERO_GUARD:
            info["guard"] = True
            return {s: (1.0 if s in div else 0.0) for s in senses}, info
        floor = p_mean / self.cap
        info["floor"] = floor
        w = {}
        for s in senses:
            if s not in div:
                w[s] = 0.0
                continue
            d = div[s]
            if d < floor:
                info["capped"].append(s)
                d = floor
            w[s] = p_mean / d
        return w, info

    # ---------------------------------------------------------------- the window's record
    def _win_reset(self) -> None:
        self.win = {"steps": 0, "raw": {}, "weighted": {}, "weight": {}, "weight_n": {},
                    "ratio": {}, "ratio_n": {},
                    "capped": {}, "unscaled": {}, "input_not_real": {}, "p_mean": 0.0, "p_mean_n": 0,
                    "lesson_plain": 0.0, "lesson_norm": 0.0, "guard": 0,
                    "replay_steps": 0, "replay_raw": {}, "replay_weighted": {},
                    "replay_unscaled": {}}

    def add_live(self, raw: Dict[str, float], weights: Dict[str, float], info: dict) -> None:
        """One live step: each graded sense's plain error (token units), its weight, and its error
        over the divisor it was weighed by (the divisor after the cap)."""
        w = self.win
        w["steps"] += 1
        plain = norm = 0.0
        divisor = info.get("divisor") or {}
        floor = info.get("floor")
        for s, e in raw.items():
            e = float(e)
            if not math.isfinite(e):
                continue
            k = float(weights.get(s, 0.0))
            w["raw"][s] = w["raw"].get(s, 0.0) + e
            w["weighted"][s] = w["weighted"].get(s, 0.0) + k * e
            w["weight"][s] = w["weight"].get(s, 0.0) + k
            w["weight_n"][s] = w["weight_n"].get(s, 0) + 1
            d = divisor.get(s)
            if d is not None:
                d = max(float(d), float(floor)) if floor is not None else float(d)
                if d > 0.0:
                    w["ratio"][s] = w["ratio"].get(s, 0.0) + e / d
                    w["ratio_n"][s] = w["ratio_n"].get(s, 0) + 1
            plain += e
            norm += k * e
        w["lesson_plain"] += plain
        w["lesson_norm"] += norm
        for s in info.get("capped") or ():
            w["capped"][s] = w["capped"].get(s, 0) + 1
        for s in info.get("unscaled") or ():
            w["unscaled"][s] = w["unscaled"].get(s, 0) + 1
        for s in info.get("input_not_real") or ():
            w["input_not_real"][s] = w["input_not_real"].get(s, 0) + 1
        if info.get("p_mean") is not None:
            w["p_mean"] += float(info["p_mean"])
            w["p_mean_n"] += 1
        if info.get("guard"):
            w["guard"] += 1

    def add_replay(self, raw: Dict[str, float], weights: Dict[str, float]) -> None:
        """One replay re-study: each graded sense's plain error and its weighted error."""
        w = self.win
        w["replay_steps"] += 1
        for s, e in raw.items():
            e = float(e)
            if not math.isfinite(e):
                continue
            k = float(weights.get(s, 0.0))
            w["replay_raw"][s] = w["replay_raw"].get(s, 0.0) + e
            w["replay_weighted"][s] = w["replay_weighted"].get(s, 0.0) + k * e
            if k == 0.0:
                # graded with no divisor yet (a re-study before the live lesson's first reading of
                # that sense): left out of that pass, counted
                w["replay_unscaled"][s] = w["replay_unscaled"].get(s, 0) + 1

    def summary(self, sig: int = 5) -> dict:
        """The `next_frame_norm` block of the stats record (DESIGN section 3), and a fresh window.

        persistence_ema: each sense's running persistence error at the window's end (token units,
          the mean squared change of its token from one moment to the next; 0 = the token never
          changes). weight_effective: the window mean of each sense's weight in the lesson (1 = as
          in the plain sum; above 1 = the sense counts more than its raw error did).
        share_plain / share_normalized: of every 100 parts of the lesson over the window, how many
          each sense would carry in the plain sum and carries under the normalisation.
        ratio: each sense's error over its persistence divisor, window mean (0 = exact, 1 = no
          better than "nothing changes", above 1 = worse than it) - what the weights make equal.
        lesson_plain / lesson_normalized: the lesson's window mean both ways (equal when every
          sense sits at persistence). capped_steps / unscaled_steps / input_not_real_steps: per
          sense, steps its weight hit the measured cap, steps it was graded with no divisor, and
          steps its previous moment was not a real reading (both weight 0).
        replay_*: the same shares for the replay re-study (replay_steps re-study passes)."""
        W = self.win

        def r(x):
            # significant digits, not decimal places: a persistence error of 1.2e-4 keeps its digits
            return None if x is None or not math.isfinite(x) else float("%.*g" % (sig, float(x)))

        def shares(d):
            t = sum(v for v in d.values() if math.isfinite(v))
            return {s: r(v / t) for s, v in d.items()} if t > 0 else None
        ratio = {s: r(W["ratio"][s] / W["ratio_n"][s]) for s in W["ratio"] if W["ratio_n"].get(s)}
        n = max(1, W["steps"])
        out = {"mode": "persistence", "version": NEXT_FRAME_NORM_VERSION,
               "horizon_steps": self.horizon, "weight_cap": self.cap,
               "steps": int(W["steps"]),
               "persistence_ema": {s: r(self.scale(s)) for s in sorted(self.stats)},
               "readings": {s: int(v[2]) for s, v in sorted(self.stats.items())},
               "weight_effective": {s: r(W["weight"][s] / W["weight_n"][s])
                                    for s in W["weight"] if W["weight_n"].get(s)},
               "share_plain": shares(W["raw"]), "share_normalized": shares(W["weighted"]),
               "ratio": ratio,
               "lesson_plain": r(W["lesson_plain"] / n) if W["steps"] else None,
               "lesson_normalized": r(W["lesson_norm"] / n) if W["steps"] else None,
               "p_mean": r(W["p_mean"] / W["p_mean_n"]) if W["p_mean_n"] else None,
               "capped_steps": dict(W["capped"]), "unscaled_steps": dict(W["unscaled"]),
               "input_not_real_steps": dict(W["input_not_real"]),
               "guard_steps": int(W["guard"]),
               "replay_steps": int(W["replay_steps"]),
               "replay_share_plain": shares(W["replay_raw"]),
               "replay_share_normalized": shares(W["replay_weighted"]),
               "replay_unscaled_steps": dict(W["replay_unscaled"]),
               "scale": {
                   "persistence_ema": "mean squared change of the sense's token between moments; "
                                      "0 = the token never changes",
                   "weight_effective": "1 = as in the plain sum; 2 = the sense counts twice its raw error",
                   "share_plain/share_normalized": "of every 1.0 of the lesson, the part each sense "
                                                   "carries, plain sum vs normalised",
                   "ratio": "0 = predicted exactly, 1 = no better than 'nothing changes', above 1 = worse",
                   "weight_cap": "no sense weighs more than this; measured as the largest weight any "
                                 "sense needed in any 100-step window of H84 segments 30-31"}}
        self._win_reset()
        return out

    # ---------------------------------------------------------------- runtime_state.pt
    def state_dict(self) -> dict:
        return {"version": NEXT_FRAME_NORM_VERSION, "horizon_steps": self.horizon,
                "weight_cap": self.cap,
                "stats": {k: [float(v[0]), float(v[1]), int(v[2])] for k, v in self.stats.items()}}

    def load_state_dict(self, sd: Optional[dict]) -> List[str]:
        """Restore the running averages by sense name; returns the names restored. A state saved
        under another horizon keeps its averages (they are still averages of the same quantity)
        and continues at this build's horizon."""
        self.stats = {}
        for k, v in ((sd or {}).get("stats") or {}).items():
            try:
                m, w, n = float(v[0]), float(v[1]), int(v[2])
            except (TypeError, ValueError, IndexError):
                continue
            if math.isfinite(m) and math.isfinite(w) and w > 0.0 and m >= 0.0:
                self.stats[str(k)] = [m, w, n]
        return sorted(self.stats)
