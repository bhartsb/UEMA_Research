"""Why the planner's choices tie: a frozen, saved-state measurement of its candidates.

THE QUESTION. In production (H84) every planning event reads "tied": the chosen order's predicted
wellness never differs from the best of four alternatives by more than the tie band (the
approximate evaluator's measured error on a difference, 0.0734 on its 0-to-5 sum). Three causes can
produce that, and this probe separates them:

  (a) FORWARD MODEL: the order does not reach the prediction, so the candidates' predicted futures
      are the same to any evaluator;
  (b) EVALUATOR: the futures differ, but the approximate evaluator (Judge B, the network that scores
      imagined moments) gives them nearly the same wellness - it cannot rank even the TRUE outcomes;
  (c) PLANNER: predicted wellness differs beyond the band but the selection does not follow it.

WHAT IS MEASURED. At declared saved moments per body - planning events of the checkpoint's own
behaviour, in all eight saved worlds of its episode companion - the planner's chosen order and the
four alternatives the production causal chain scores (mimo_bridge/lem_model.py
_causal_plan_record: hold, the controller's own proposal, two babble-sized perturbations of the
chosen order, seeded exactly as production seeds them) are:

  PREDICTED before anything moves, on the planner's own route (the top-down-gated tokens it
  descends from, its captured context, pose and memory bias, the same execution transform and
  smoother, transition() with the order held 1, 2 and 3 moments) and on the learning route (the raw
  tokens): every sense's predicted token, Judge B's five wellness terms, the planner's goal distance;

  EXECUTED, each from the identical restored snapshot, for the order's whole tenure (motor_hold
  moments of 50 ms), the eyes held at the candidate's own gaze command, the saved world's reflex and
  saccades left as they run in production. Every branch is run a second time and its physical trace
  (joint positions and velocities, both eyes' angles and the executed hand targets at every 10 ms
  physics step) must be bit-identical; once per body the chosen branch is also observed twice and
  its observations must be bit-identical. After every moment: the raw tokens the body produced, the
  grounded wellness terms, Judge B on those ACTUAL tokens, the model's next-moment prediction error,
  the pose.

THE SEPARATION (every interval: 95%, whole bodies resampled with replacement, 8 bodies).
  Judge B's difference between two candidates' PREDICTED outcomes (dJP), its difference between
  their ACTUAL outcomes (dJA) and the grounded wellness difference (dW) are taken for every pair of
  candidates from the same moment. The chain's error on a difference (the tie band's quantity)
  splits exactly into dJP - dW = (dJP - dJA) + (dJA - dW): the forward model's part (what imperfect
  foresight adds, in Judge B's own units) and the evaluator's part (its error even on true outcomes).
  transfer_forward_model = slope of dJP on dJA and transfer_evaluator = slope of dJA on dW (1 = the
  stage passes the whole difference on, 0 = none of it); in token space the forward model's
  captured share of the between-candidate difference (0 = foresaw none, 1 = all) per sense.
  A stage is FLAT when its transfer's interval does not exclude zero (nothing measurable passes).
  The planner ignores predicted wellness when its objective's wellness weight is 0 and the chosen
  order's rank by predicted wellness is not better than chance (3 of 5).

THE DEPTH COMPARISON. The production chain (_plan_chain) forces depth 1 whenever
--hand-target-mode is not 'world' and picks its first order with the depth-1 search before it
imagines anything deeper; the probe records both facts from the running code (it asks the planner
for 3 steps in production mode and, with the hand path set aside, compares the first order of a
1-step and a 3-step chain). The behavioural test is then made the way the observer declared it:
from identical saved starts, the same five candidates are selected by predicted wellness at depth 1
(50 ms) and at depth 3 (150 ms, the order held), and by the planner's own objective (goal distance
plus effort) at both depths; every candidate was executed, so both selections' actual outcomes are
compared: actual wellness over the tenure (primary), wellness at 150 ms, physical controllability
(the share of each ordered joint turn the body closed; ordered hand and wrist positions), the eyes'
error from their command, the next-moment prediction error after the order, and the actual distance
to the goal. The claim under test is "looking further ahead causes a better physical decision":
supported only when the depth-3 selection's actual wellness beats the depth-1 selection's with the
paired body interval above zero.

THE RANKING (the outside review of 2026-09-24, in every result since planner_candidates_v2). Of all
pairs of the five candidates from one saved start, how many in 100 the forecast orders as their
actual outcomes are ordered (50 = chance, 80 to 90 = a ranking a planner can optimize): the planner's
predicted goal distance against the actual distance at 50 ms and at the order's end, and its
predicted wellness against the grounded wellness at 50 ms and over the tenure; each with its 95% body
interval, the spread ratio (how far apart the forecasts are over how far apart the outcomes are) and
the reviewer's case label by mimo_bridge/tools/command_ranking.py RULES (action-insensitive, wrong
order, or a usable ranking whose chosen order still does not beat the controller's own proposal in
the actual outcome). The verdict above does not read it.

VERSION 3 (2026-10-02, the review of docs/gain_of_function_v4/coordinator/own_command_analysis_2026-10-02.html).
  EXECUTED POSE AT DEPTHS 2 AND 3. transition() builds the motor token from a command and an achieved pose (the 26
  handles' positions and orientations). Up to version 2 every depth ran with the decision's pose, so depths 2 and 3
  predicted from a pose 1 and 2 model steps old. Every candidate is executed, so its real pose after 1 and 2 model
  steps exists: after the branches run and the snapshot is restored, the transitions are run again with depth d
  reading the candidate's own pose after d - 1 executed model steps (depth 1 reads the decision's pose and must
  reproduce the first pass bit for bit, else the probe stops), and the token-space shares
  (summary.forward_model.senses) are taken on those predictions. The decision-pose shares of depths 2 and 3 stay
  beside them (summary.forward_model.senses_decision_pose). The depth comparison's selections and the planner's
  ranks keep the decision-pose predictions: a planner at the decision has no later pose, so selecting with it would
  score a forecast no planner can make.
  NOT MEASURABLE BELOW THE NOISE FLOOR. The replays are bit-identical by construction (a branch whose replay differs
  stops the probe), so the replay-repeat difference is exactly 0 and cannot be a floor. The floor is the float32
  resolution of the token itself (float32_floor): per pair of candidates, the mean over the token's d numbers of
  (d x the float32 grid step at that number)^2 - d grid steps per number, the worst-case rounding of one float32 sum
  over d numbers (n x unit roundoff for an n-term sum). A sense whose summed real difference over all pairs does not
  exceed its summed floor reads "not measurable": no captured share, no ratio. Measured on the nine saved rechecks
  (checkpoints 399,100 to 450,525, their start_*.npz): vestibular's real differences were 2 to 19 grid steps per
  number (root mean square), under this floor by a factor of 10 or more; vision, tactile, audio and proprioception
  were 270,000 grid steps or more, above it by a factor of 1,400 or more.
  SCRIPT IDENTITY. Every per-start row, status.json and the summary carry the SHA-256 of this file (the scheduler's
  script_sha256 is the same digest); the Development dashboard labels each round by it.

Nothing trains: every model step runs with learning off, every parameter has requires_grad False,
and the learned weights are hashed before and after (a change stops the probe). Nothing here
touches the live run, its control files or the supervisor.

Usage (the scheduler's form): python probe_planner_candidates.py --checkpoint <ckpt> --out <dir>
Writes <out>/declaration.json first, then starts.jsonl, start_<body>_<k>.npz (tokens), status.json
and finally result.json.
"""
from __future__ import annotations

import os
if __name__ == "__main__":  # this probe runs on the CPU; importing the probe harness no longer hides the card
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # (2026-09-26, probe_gpu_hiding_2026-09-26.md)
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

# v2 (2026-09-25): v1 plus summary.ranking, the outside review's pairwise statistic
# v3 (2026-10-02): depths 2 and 3 read each candidate's executed pose; "not measurable" below a sense's float32
# floor; the script's SHA-256 in every row, status file and summary (VERSION 3 in the docstring)
PROBE_VERSION = "planner_candidates_v3"
# which achieved pose each depth's transition reads (declared in declaration.json; the scheduler checks it)
DEPTH_POSE = "executed"
DEPTH_POSE_RULE = ("depth 1 reads the decision's achieved pose; depth d > 1 reads the candidate's own achieved pose "
                   "after d - 1 executed model steps of its branch; the depth comparison's selections and the "
                   "planner's ranks keep the decision-pose predictions (a planner at the decision has no later pose)")
NOISE_FLOOR_RULE = ("a sense is measurable at a depth only when its real difference between candidates, summed over "
                    "all pairs, exceeds its float32 floor summed over the same pairs; a pair's floor is the mean over "
                    "the token's d numbers of (d x the float32 grid step at that number)^2, the grid step averaged over "
                    "the two candidates' real tokens (numpy.spacing); below it the captured share is 'not measurable'")
NOT_MEASURABLE = ("not measurable: the real difference between candidates is within the float32 floor of the "
                  "token (NOISE_FLOOR_RULE)")
# identical inputs give an identical depth-1 prediction: the second pass (executed poses) must reproduce the first.
# The replays' observations are already required bit-identical; 1e-6 on tokens whose numbers have a root mean
# square of 1 (the registry's output LayerNorm) allows a last-bit difference of a threaded sum and nothing a changed
# model state would produce
EXECUTED_POSE_DEPTH1_TOLERANCE = 1e-6
CANDIDATES = ("chosen", "hold", "reflex", "perturb_1", "perturb_2")
DEPTHS = (1, 2, 3)
# the senses Judge B reads besides the motor token (lem.model.wellness APPROXIMATE_EXCLUDED leaves
# out intermind and vestibular); vestibular is predicted and graded, so it is reported beside them
JUDGE_SENSES = ("vision", "tactile", "audio", "proprioception")
ALL_SENSES = JUDGE_SENSES + ("vestibular",)
# lem.model.wellness.APPROXIMATE_TERMS, in Judge B's output order (a test holds the two equal)
FIVE_TERMS = ("prediction_error", "tactile_comfort", "capability", "coherence", "motor_state")
BOOT_DRAWS = 5000
BOOT_SEED = 240924
NEED_GIB = 12.0
RESERVE_GIB = 6.0
PRODUCTION_TIE_BAND = 0.0734       # mimo_bridge.causal_chain.TIE_BAND when this probe was written
# How closely the probe must reproduce the production planning record at the same decision before it
# measures anything: the predicted wellness sums (0-to-5) and goal distances of the five candidates,
# and the alternatives' execution form against LemLite._causal_execution_candidates (cm). The same
# functions on the same state differ only by float32 rounding between a batch of 4 and of 5 (about
# 1e-6 on a sum near 3); 1e-3 is a thousand times that and 1/70 of the tie band.
ROUTE_TOLERANCE = {"record_pred_wellness_max_abs": 1e-3, "record_goal_distance_max_abs": 1e-3,
                   "alternatives_execution_max_abs_cm": 1e-3}
SOURCES = ("Source/Python/mimo_bridge/lem_model.py", "Source/Python/mimo_bridge/causal_chain.py",
           "Source/Python/mimo_bridge/tools/gof_probe_harness.py",
           "Source/Python/lem/model/predictive_state.py", "Source/Python/lem/model/wellness.py",
           "Source/Python/lem/model/intention_generator.py",
           "Source/Python/mimo_bridge/hand_target_control.py")
REPO = Path(__file__).resolve().parents[2]


# ============================================================================ pure arithmetic
def pair_indices(k):
    """Every pair (a, b) with a < b among k candidates."""
    return [(a, b) for a in range(int(k)) for b in range(a + 1, int(k))]


def pair_differences(values):
    """[values[a] - values[b] for every pair a < b]."""
    v = [float(x) for x in values]
    return [v[a] - v[b] for a, b in pair_indices(len(v))]


def float32_floor(token_a, token_b):
    """The noise floor of the real difference between two tokens ([d] each, as float32 numbers): the mean over the
    d numbers of (d x s)^2, s = the float32 grid step at that number averaged over the two tokens (numpy.spacing).
    d grid steps per number is the worst-case rounding of one float32 sum over the token's d numbers (n x unit
    roundoff for an n-term sum), so a mean squared difference at or below it cannot be told apart from the rounding
    of the token's own computation (NOISE_FLOOR_RULE)."""
    a = np.abs(np.asarray(token_a, dtype=np.float32).reshape(-1))
    b = np.abs(np.asarray(token_b, dtype=np.float32).reshape(-1))
    step = 0.5 * (np.spacing(a).astype(np.float64) + np.spacing(b).astype(np.float64))
    return float(np.mean((a.size * step) ** 2))


def sense_pair_stats(pred, actual, current):
    """Between-candidate differences of one sense at one horizon.

    pred, actual: [K, d] the K candidates' predicted and actual tokens; current: [d] the raw token
    at the decision. Per pair (a < b): pred_diff = mean over the token of (P_a - P_b)^2, real_diff
    the same for the actual tokens, dot = mean of (P_a - P_b)(A_a - A_b), floor = float32_floor(A_a, A_b)
    (the real difference's noise floor). Per candidate: err = mean (P_k - A_k)^2 (the prediction's own
    error), persist = mean (current - A_k)^2 (the error of predicting that nothing changes)."""
    p = np.asarray(pred, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    a32 = np.asarray(actual, dtype=np.float32)
    c = np.asarray(current, dtype=np.float64).reshape(-1)
    out = dict(pred_diff=[], real_diff=[], dot=[], floor=[])
    for i, j in pair_indices(p.shape[0]):
        dp, da = p[i] - p[j], a[i] - a[j]
        out["pred_diff"].append(float(np.mean(dp * dp)))
        out["real_diff"].append(float(np.mean(da * da)))
        out["dot"].append(float(np.mean(dp * da)))
        out["floor"].append(float32_floor(a32[i], a32[j]))
    out["err"] = [float(np.mean((p[k] - a[k]) ** 2)) for k in range(p.shape[0])]
    out["persist"] = [float(np.mean((c - a[k]) ** 2)) for k in range(p.shape[0])]
    return out


def ratio_of_sums(num, den):
    """sum(num) / sum(den), None when the denominator is not positive."""
    n, d = float(np.sum(np.asarray(num, float))), float(np.sum(np.asarray(den, float)))
    if not d > 0:
        return None
    r = n / d
    return r if math.isfinite(r) else None


def uncentered_corr(x, y):
    """sum(xy) / sqrt(sum(x^2) sum(y^2)): the cosine between two lists of pair differences.
    Reversing a pair (a - b instead of b - a) flips both signs, so the value does not depend on
    how pairs are oriented. 1 = move together in size and sign, 0 = unrelated, -1 = opposite."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    den = math.sqrt(float(np.sum(x * x)) * float(np.sum(y * y)))
    return None if not den > 0 else float(np.sum(x * y) / den)


def slope_through_origin(y, x):
    """How much of x's difference reaches y: sum(xy) / sum(x^2). 1 = all of it, 0 = none."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    den = float(np.sum(x * x))
    return None if not den > 0 else float(np.sum(x * y) / den)


def sign_agreement(x, y):
    """Share of pairs, among those where both are nonzero, whose signs agree (0.5 = a coin)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x != 0) & (y != 0)
    return None if not ok.any() else float(np.mean(np.sign(x[ok]) == np.sign(y[ok])))


def quantile(values, q):
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return None if not v else float(np.quantile(np.asarray(v), q))


def cluster_bootstrap(groups, statistic, draws=BOOT_DRAWS, seed=BOOT_SEED):
    """statistic over every item, with a 95% interval from resampling whole bodies.

    groups: {body: [item, ...]}; statistic(list of items) -> float or None. Returns
    dict(value, interval, bodies, draws_used); the interval is None when fewer than half the draws
    gave a value."""
    keys = sorted(k for k in groups if groups[k])
    value = statistic([it for k in keys for it in groups[k]])
    if not keys:
        return dict(value=value, interval=None, bodies=0, draws_used=0)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(draws)):
        pick = rng.integers(len(keys), size=len(keys))
        v = statistic([it for i in pick for it in groups[keys[i]]])
        if v is not None and math.isfinite(v):
            vals.append(v)
    interval = (None if len(vals) < 0.5 * draws
                else [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))])
    return dict(value=value, interval=interval, bodies=len(keys), draws_used=len(vals))


def body_mean_interval(values_by_body, draws=BOOT_DRAWS, seed=BOOT_SEED):
    """The mean of the body means, with a 95% interval from resampling bodies. Used for paired
    differences (depth-3 choice minus depth-1 choice from the same starts)."""
    keys = sorted(k for k, v in values_by_body.items() if v)
    means = np.asarray([float(np.mean(values_by_body[k])) for k in keys], float)
    if not len(means):
        return dict(mean=None, interval=None, bodies=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(means), size=(int(draws), len(means)))
    boot = means[idx].mean(-1)
    return dict(mean=float(means.mean()),
                interval=[float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                bodies=len(keys), bodies_above_zero=int((means > 0).sum()),
                bodies_below_zero=int((means < 0).sum()))


def excludes_zero_above(stat):
    iv = (stat or {}).get("interval")
    return bool(iv) and iv[0] > 0


def select(scores, higher_is_better=True):
    """The best candidate's index; an exact tie goes to the lower index (the chosen order first)."""
    s = np.asarray(scores, float)
    return int(np.argmax(s) if higher_is_better else np.argmin(s))


def strict_rank(scores, index, higher_is_better=True):
    """1 + how many candidates score strictly better than `index` (1 = best, K = worst)."""
    s = np.asarray(scores, float)
    better = (s > s[index]) if higher_is_better else (s < s[index])
    return 1 + int(better.sum())


def mid_rank(scores, index, higher_is_better=True):
    """strict_rank with every exact tie counted as half better: K equal scores give every candidate
    the chance rank (K + 1) / 2, so a tie is never read as the chosen order being best."""
    s = np.asarray(scores, float)
    better = (s > s[index]) if higher_is_better else (s < s[index])
    equal = (s == s[index])
    return 1.0 + float(better.sum()) + 0.5 * float(equal.sum() - 1)


def tie_outcome(gap_to_best_alt, band):
    """The production rule (mimo_bridge.causal_chain.outcome): best / tied / behind."""
    if gap_to_best_alt > band:
        return "best"
    if gap_to_best_alt < -band:
        return "behind"
    return "tied"


def perturb_seed(causal_seed, step):
    """The generator seed LemLite._causal_plan_record uses for this step's perturbations."""
    return (int(causal_seed) * 1_000_003 + int(step)) % (2 ** 62)


def hold_candidate(plan0, achieved, exec_prev, gaze_idx):
    """The production causal chain's 'hold' (LemLite._causal_plan_record): every handle's target
    where it is now, the plan's radius and strength, the eyes at the last realised eye command."""
    hold = plan0.detach().float().clone()
    if achieved is not None:
        hold[..., 0:3] = achieved[..., 0:3].float()
        q = achieved[..., 3:7].float()
        hold[..., 3:7] = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    B = plan0.shape[0]
    if exec_prev is not None and exec_prev.shape[0] == B:
        hold[:, gaze_idx, 3:7] = exec_prev[:, gaze_idx, 3:7].float()
    else:
        hold[:, gaze_idx] = plan0[:, gaze_idx]
    return hold


def clean(value):
    """JSON-ready: numpy scalars and arrays to Python, non-finite floats to None."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_SCRIPT_SHA256 = []


def script_sha256():
    """The SHA-256 of this file as it was when first asked (the scheduler's script_sha256 is the same digest)."""
    if not _SCRIPT_SHA256:
        _SCRIPT_SHA256.append(sha256_file(__file__))
    return _SCRIPT_SHA256[0]


# ============================================================================ the summary
def _pairs_by_body(rows, getter):
    out = {}
    for r in rows:
        v = getter(r)
        if v is not None:
            out.setdefault(r["body"], []).extend(v)
    return out


def _stat_pairs(groups, fn, draws=BOOT_DRAWS):
    """cluster_bootstrap over items that are (x, y) tuples with fn(xs, ys)."""
    return cluster_bootstrap(groups, lambda items: fn([a for a, _ in items], [b for _, b in items])
                             if items else None, draws=draws)


def _decided_share(groups, band, draws=BOOT_DRAWS):
    return cluster_bootstrap(groups, lambda items: (float(np.mean([abs(x) > band for x in items]))
                                                    if items else None), draws=draws)


def summarise(rows, production_band=PRODUCTION_TIE_BAND, draws=BOOT_DRAWS):
    """Every aggregate of the result from the per-start rows (pure: tests feed it synthetic rows).
    Each row carries, for K candidates (index 0 = the planner's chosen order) and T moments:
      judge_pred[route][depth] [K][5], judge_pred_fixed_motor[depth] [K][5], judge_pred_motor_only
      [K][5], goal_distance_pred[depth] [K], effort [K], sense_stats[route][depth][sense]
      (sense_pair_stats), route_disagreement[sense] [K], motor_token_change [K],
      actual: wellness_terms [K][T][5], judge_terms [K][T][5], next_pe_sum [K][T], goal_distance [K][T],
      control: turn_closed_share [K], closer_share [K], chance_share [K], position_closed_share [K],
      eye_error_deg [K]; production: the planning record's sums."""
    def cb(groups, statistic):
        return cluster_bootstrap(groups, statistic, draws=draws)

    def bmi(values):
        return body_mean_interval(values, draws=draws)

    def sp(groups, fn):
        return _stat_pairs(groups, fn, draws)

    def ds(groups, band):
        return _decided_share(groups, band, draws)

    K = len(rows[0]["candidates"]) if rows else len(CANDIDATES)
    T = len(rows[0]["actual"]["wellness_terms"][0]) if rows else 0
    out = dict(starts=len(rows), bodies=len({r["body"] for r in rows}), candidates=K, moments=T,
               # which code computed these numbers: this summary's arithmetic, and the measurement of each row
               # (a row written before planner_candidates_v3 carries no hash: null)
               probe=PROBE_VERSION, summarised_by_script_sha256=script_sha256(),
               rows_measured_by_script_sha256=sorted({str(r.get("script_sha256")) for r in rows
                                                      if r.get("script_sha256")}) or None)

    def five(terms):
        return float(np.sum(terms))

    # ------------------------------------------------ wellness differences, one moment after
    def w_actual(r, k):   # grounded five-term sum, candidate c, moment k (1-based)
        return [five(r["actual"]["wellness_terms"][c][k - 1]) for c in range(K)]

    def ja_actual(r, k):
        return [five(r["actual"]["judge_terms"][c][k - 1]) for c in range(K)]

    def jp(r, route, d):
        return [five(t) for t in r["judge_pred"][route][str(d)]]

    def tenure_w(r):
        return [float(np.mean([five(t) for t in r["actual"]["wellness_terms"][c]])) for c in range(K)]

    dW = _pairs_by_body(rows, lambda r: pair_differences(w_actual(r, 1)))
    dJA = _pairs_by_body(rows, lambda r: pair_differences(ja_actual(r, 1)))
    dJP = {route: _pairs_by_body(rows, lambda r, route=route: pair_differences(jp(r, route, 1)))
           for route in ("planner", "learning")}

    def zipped(a, b):
        return {k: list(zip(a[k], b[k])) for k in a if k in b}

    flat = lambda g: [x for k in sorted(g) for x in g[k]]
    w_all, ja_all, jp_all = flat(dW), flat(dJA), flat(dJP["planner"])
    band_planner = quantile([abs(p - w) for p, w in zip(jp_all, w_all)], 0.9)
    band_learning = quantile([abs(p - w) for p, w in zip(flat(dJP["learning"]), w_all)], 0.9)
    band_eval = quantile([abs(a - w) for a, w in zip(ja_all, w_all)], 0.9)
    e_fm = np.asarray(jp_all, float) - np.asarray(ja_all, float)
    e_ev = np.asarray(ja_all, float) - np.asarray(w_all, float)
    total = e_fm + e_ev
    ss_total = float(np.sum(total * total))
    decomposition = None
    if ss_total > 0:
        decomposition = dict(
            forward_model_share=float(np.sum(e_fm * e_fm)) / ss_total,
            evaluator_share=float(np.sum(e_ev * e_ev)) / ss_total,
            cross_share=float(2 * np.sum(e_fm * e_ev)) / ss_total,
            p90_abs_forward_model_part=quantile(np.abs(e_fm), 0.9),
            p90_abs_evaluator_part=quantile(np.abs(e_ev), 0.9))
    out["tie_band"] = dict(
        production=production_band,
        remeasured_planner_route=band_planner, remeasured_learning_route=band_learning,
        evaluator_on_actual_outcomes=band_eval, decomposition=decomposition,
        definition="90th percentile over candidate pairs of |predicted difference - grounded "
                   "difference| one moment after the order, on the 0-to-5 sum")
    band = band_planner if band_planner is not None else production_band
    out["wellness_differences"] = dict(
        grounded_abs=dict(median=quantile(np.abs(w_all), 0.5), p90=quantile(np.abs(w_all), 0.9)),
        judge_actual_abs=dict(median=quantile(np.abs(ja_all), 0.5), p90=quantile(np.abs(ja_all), 0.9)),
        judge_predicted_abs=dict(median=quantile(np.abs(jp_all), 0.5), p90=quantile(np.abs(jp_all), 0.9)),
        sign_agreement_predicted_vs_grounded=sign_agreement(jp_all, w_all),
        sign_agreement_actual_judge_vs_grounded=sign_agreement(ja_all, w_all),
        decided_pairs_production=ds(dJP["planner"], production_band),
        decided_pairs_remeasured=ds(dJP["planner"], band),
        decided_pairs_if_forecasts_were_perfect=ds(dJA, band_eval if band_eval else band))
    # ------------------------------------------------ the three transfers
    ev_pairs, fm_pairs, chain_pairs = zipped(dW, dJA), zipped(dJA, dJP["planner"]), zipped(dW, dJP["planner"])
    evaluator = dict(
        transfer=sp(ev_pairs, lambda x, y: slope_through_origin(y, x)),
        corr=sp(ev_pairs, uncentered_corr),
        definition="slope of Judge B's difference between two candidates' ACTUAL outcomes on their "
                   "grounded wellness difference, one moment after (1 = sees the whole true "
                   "difference, 0 = blind to it)")
    per_term = {}
    for t, name in enumerate(FIVE_TERMS):
        gw = _pairs_by_body(rows, lambda r, t=t: pair_differences(
            [r["actual"]["wellness_terms"][c][0][t] for c in range(K)]))
        ga = _pairs_by_body(rows, lambda r, t=t: pair_differences(
            [r["actual"]["judge_terms"][c][0][t] for c in range(K)]))
        gp = _pairs_by_body(rows, lambda r, t=t: pair_differences(
            [r["judge_pred"]["planner"]["1"][c][t] for c in range(K)]))
        fw, fa, fp = flat(gw), flat(ga), flat(gp)
        per_term[name] = dict(
            share_of_grounded_difference=(None if not np.sum(np.square(w_all)) > 0 else
                                          float(np.sum(np.square(fw)) / np.sum(np.square(w_all)))),
            grounded_abs_median=quantile(np.abs(fw), 0.5),
            judge_actual_abs_median=quantile(np.abs(fa), 0.5),
            judge_predicted_abs_median=quantile(np.abs(fp), 0.5),
            evaluator_transfer=slope_through_origin(fa, fw), evaluator_corr=uncentered_corr(fa, fw))
    evaluator["per_term"] = per_term
    # Judge B's candidate spread: full, predicted senses with the motor token held at the present
    # one, and the motor token alone (the present senses with each candidate's motor token)
    spread = lambda sums: float(np.max(sums) - np.min(sums))
    evaluator["candidate_spread_median"] = dict(
        predicted=quantile([spread(jp(r, "planner", 1)) for r in rows], 0.5),
        predicted_senses_present_motor=quantile(
            [spread([five(t) for t in r["judge_pred_fixed_motor"]["1"]]) for r in rows], 0.5),
        present_senses_candidate_motor=quantile(
            [spread([five(t) for t in r["judge_pred_motor_only"]]) for r in rows], 0.5),
        actual=quantile([spread(ja_actual(r, 1)) for r in rows], 0.5),
        grounded=quantile([spread(w_actual(r, 1)) for r in rows], 0.5))
    forward = dict(
        transfer_judge_units=sp(fm_pairs, lambda x, y: slope_through_origin(y, x)),
        corr_judge_units=sp(fm_pairs, uncentered_corr),
        definition="slope of Judge B's difference between two candidates' PREDICTED outcomes on "
                   "its difference between their ACTUAL outcomes (1 = the forecast hands Judge B "
                   "every difference reality does, 0 = none)")
    # token space, per sense and pooled over Judge B's senses: on the executed-pose predictions (sense_stats) and,
    # beside them, depths 2 and 3 on the decision's pose (sense_stats_decision_pose, the route of versions 1 and 2)
    def token_shares(key):
        senses = {}

        def stats_of(r, route, d):
            return (((r.get(key) or {}).get(route) or {}).get(str(d))) or {}

        for route in ("planner", "learning"):
            for d in DEPTHS:
                for m in ALL_SENSES + ("judge_senses",):
                    wanted = JUDGE_SENSES if m == "judge_senses" else (m,)

                    def items(r, wanted=wanted, route=route, d=d):
                        st = stats_of(r, route, d)
                        use = [s for s in wanted if s in st]
                        if not use:
                            return None
                        n = len(st[use[0]]["dot"])
                        floors = all("floor" in st[s] for s in use)
                        return [(sum(st[s]["dot"][i] for s in use), sum(st[s]["real_diff"][i] for s in use),
                                 sum(st[s]["pred_diff"][i] for s in use),
                                 sum(st[s]["floor"][i] for s in use) if floors else None) for i in range(n)]
                    g = _pairs_by_body(rows, items)
                    if not g:
                        continue
                    have = [r for r in rows if stats_of(r, route, d)]
                    errs = [sum(stats_of(r, route, d)[s]["err"][c] for s in wanted if s in stats_of(r, route, d))
                            for r in have for c in range(K)]
                    pers = [sum(stats_of(r, route, d)[s]["persist"][c] for s in wanted if s in stats_of(r, route, d))
                            for r in have for c in range(K)]
                    allp = [it for k in sorted(g) for it in g[k]]
                    real_sum = float(sum(b for _, b, _, _ in allp))
                    # rows written before planner_candidates_v3 carry no floor: their shares are computed as then
                    floor_sum = None if any(f is None for *_, f in allp) else float(sum(f for *_, f in allp))
                    measurable = None if floor_sum is None else bool(real_sum > floor_sum)
                    if measurable is False:
                        captured = dict(value=None, interval=None, bodies=len(g), draws_used=0, reason=NOT_MEASURABLE)
                    else:
                        captured = cb(g, lambda its: ratio_of_sums(
                            [a for a, _, _, _ in its], [b for _, b, _, _ in its]) if its else None)
                    captured["measurable"] = measurable
                    senses.setdefault(route, {}).setdefault(str(d), {})[m] = dict(
                        captured_share=captured, measurable=measurable,
                        real_difference_over_noise_floor=(None if not floor_sum else real_sum / floor_sum),
                        noise_floor_mean=(None if floor_sum is None else floor_sum / len(allp)),
                        predicted_over_realised_difference=(None if measurable is False else ratio_of_sums(
                            [p for _, _, p, _ in allp], [b for _, b, _, _ in allp])),
                        predicted_difference_over_prediction_error=(
                            None if not np.mean(errs) > 0 else float(np.mean([p for _, _, p, _ in allp]) / np.mean(errs))),
                        realised_difference_over_prediction_error=(
                            None if not np.mean(errs) > 0 else float(np.mean([b for _, b, _, _ in allp]) / np.mean(errs))),
                        prediction_error_mean=float(np.mean(errs)), persistence_error_mean=float(np.mean(pers)),
                        predicted_difference_mean=float(np.mean([p for _, _, p, _ in allp])),
                        realised_difference_mean=float(np.mean([b for _, b, _, _ in allp])))
        return senses
    forward["senses"] = token_shares("sense_stats")
    decision_pose = token_shares("sense_stats_decision_pose")
    if decision_pose:
        forward["senses_decision_pose"] = decision_pose
    forward["depth_pose"] = (DEPTH_POSE if any(r.get("sense_stats_decision_pose") for r in rows)
                             else "decision (rows written before planner_candidates_v3)")
    forward["noise_floor_rule"] = NOISE_FLOOR_RULE
    rd = {m: [x for r in rows for x in (r.get("route_disagreement") or {}).get(m, [])] for m in ALL_SENSES}
    pd1 = {m: [x for r in rows for x in (r["sense_stats"]["planner"].get("1", {}).get(m, {}) or {}).get("pred_diff", [])]
           for m in ALL_SENSES}
    forward["route_disagreement_over_candidate_difference"] = {
        m: (None if not (rd[m] and pd1[m] and np.mean(pd1[m]) > 0) else float(np.mean(rd[m]) / np.mean(pd1[m])))
        for m in ALL_SENSES}
    forward["motor_token_change_median"] = quantile(
        [x for r in rows for x in r["motor_token_change"][1:]], 0.5)
    chain = dict(transfer=sp(chain_pairs, lambda x, y: slope_through_origin(y, x)),
                 corr=sp(chain_pairs, uncentered_corr),
                 definition="slope of the planner's predicted wellness difference on the grounded "
                            "difference: the whole route the production tie is judged on")
    out["evaluator"], out["forward_model"], out["chain"] = evaluator, forward, chain
    # ------------------------------------------------ the planner
    def ranks(fn, higher=True):
        return _pairs_by_body(rows, lambda r: [mid_rank(fn(r), 0, higher)])

    rank_pred = ranks(lambda r: jp(r, "planner", 1))
    obj = lambda r, d: [g + e for g, e in zip(r["goal_distance_pred"][str(d)], r["effort"])]
    rank_obj = ranks(lambda r: obj(r, 1), higher=False)
    rank_actual_w = ranks(lambda r: tenure_w(r))
    rank_actual_goal = ranks(lambda r: [r["actual"]["goal_distance"][c][-1] for c in range(K)], higher=False)
    mean_stat = lambda items: float(np.mean(items)) if items else None
    passed_over = sum(1 for r in rows if max(jp(r, "planner", 1)[1:]) - jp(r, "planner", 1)[0] > band)
    prod = [r.get("production") or {} for r in rows]
    gaps = [p.get("gap_to_best_alt") for p in prod if p.get("gap_to_best_alt") is not None]
    out["planner"] = dict(
        objective_wellness_weight=(rows[0].get("planner_wellness_weight") if rows else None),
        chance_rank=(K + 1) / 2.0,
        chosen_rank_by_predicted_wellness=cb(rank_pred, mean_stat),
        chosen_rank_by_own_objective=cb(rank_obj, mean_stat),
        chosen_rank_by_actual_wellness_over_tenure=cb(rank_actual_w, mean_stat),
        chosen_rank_by_actual_goal_distance=cb(rank_actual_goal, mean_stat),
        chosen_actual_wellness_minus_alternatives=bmi(_pairs_by_body(
            rows, lambda r: [tenure_w(r)[0] - float(np.mean(tenure_w(r)[1:]))])),
        starts_where_an_alternative_was_predicted_better_beyond_the_band=passed_over,
        production_record_outcomes={o: sum(1 for p in prod for x in (p.get("outcome") or []) if x == o)
                                    for o in ("best", "tied", "behind")},
        production_record_gap_abs_p90=quantile(np.abs(gaps), 0.9) if gaps else None)
    # ------------------------------------------------ the depth comparison
    def outcomes(r, c):
        tw = tenure_w(r)[c]
        return dict(wellness_tenure=tw, wellness_150ms=five(r["actual"]["wellness_terms"][c][min(3, T) - 1]),
                    turn_closed_share=r["control"]["turn_closed_share"][c],
                    position_closed_share=r["control"]["position_closed_share"][c],
                    eye_error_deg=r["control"]["eye_error_deg"][c],
                    next_prediction_error=float(np.mean(r["actual"]["next_pe_sum"][c])),
                    goal_distance_end=r["actual"]["goal_distance"][c][-1])
    selectors = {"wellness": (lambda r, d: jp(r, "planner", d), True),
                 "objective": (lambda r, d: obj(r, d), False)}
    depth = {}
    for name, (score, higher) in selectors.items():
        picks = [(select(score(r, 1), higher), select(score(r, 3), higher)) for r in rows]
        block = dict(differs_share=(float(np.mean([a != b for a, b in picks])) if picks else None),
                     picks=[list(p) for p in picks])
        block["predicted_difference_at_150ms"] = quantile(
            [score(r, 3)[b] - score(r, 3)[a] for r, (a, b) in zip(rows, picks)], 0.5)
        for key in ("wellness_tenure", "wellness_150ms", "turn_closed_share", "position_closed_share",
                    "eye_error_deg", "next_prediction_error", "goal_distance_end"):
            diffs = {}
            for r, (a, b) in zip(rows, picks):
                va, vb = outcomes(r, a)[key], outcomes(r, b)[key]
                if va is None or vb is None:
                    continue
                diffs.setdefault(r["body"], []).append(float(vb) - float(va))
            block[key + "_depth3_minus_depth1"] = bmi(diffs)
        depth[name] = block
    prod_vs = {}
    for key in ("wellness_tenure", "turn_closed_share", "next_prediction_error", "goal_distance_end"):
        diffs = {}
        for r in rows:
            b = select(jp(r, "planner", 3), True)
            va, vb = outcomes(r, 0)[key], outcomes(r, b)[key]
            if va is None or vb is None:
                continue
            diffs.setdefault(r["body"], []).append(float(vb) - float(va))
        prod_vs[key + "_wellness_depth3_minus_chosen"] = bmi(diffs)
    depth["against_the_planners_own_choice"] = prod_vs
    structural = [r.get("structural") for r in rows if r.get("structural")]
    depth["structural"] = dict(
        checks=len(structural),
        production_asked_3_returned_length=[s.get("production_asked_3_returned_length") for s in structural],
        production_depth_forced_to_one=all(bool(s.get("production_depth_forced_to_one")) for s in structural)
        if structural else None,
        world_mode_first_order_identical=[s.get("world_mode_first_order_identical") for s in structural],
        world_mode_deeper_plan_emptied=[s.get("world_mode_deeper_plan_emptied") for s in structural])
    out["depth"] = depth
    # ------------------------------------------------ controllability of every candidate
    out["controllability"] = {
        key: {CANDIDATES[c] if c < len(CANDIDATES) else str(c): quantile([r["control"][key][c] for r in rows
                                                                          if r["control"][key][c] is not None], 0.5)
              for c in range(K)}
        for key in ("turn_closed_share", "closer_share", "chance_share", "position_closed_share", "eye_error_deg")}
    # ------------------------------------------------ does the forecast rank the commands as reality does
    out["ranking"] = ranking_summary(rows, draws)
    return out


RANKING_OUTCOMES = dict(
    goal_distance_50ms="the planner's predicted distance to its goal 50 ms ahead against the actual distance "
                       "50 ms after each command ran from the same saved state",
    goal_distance_end="the same forecast against the actual distance to the goal at the end of the order's "
                      "tenure",
    wellness_50ms="the planner route's predicted wellness 50 ms ahead (Judge B's five-term sum) against the "
                  "grounded wellness 50 ms after each command",
    wellness_tenure="the same forecast against the grounded wellness averaged over the order's tenure")
# case 3's comparison: the planner's chosen order against the command its search starts from, the
# controller's own proposal (the production chain's 'reflex' alternative)
RANKING_PLANNER, RANKING_REFERENCE = "chosen", "reflex"
RANKING_NOTES = dict(
    planner_comparison="case 3 compares the chosen order with the controller's own proposal for the goal "
                       "distances always, and for wellness only when the planner's objective weighs "
                       "predicted wellness (planner_wellness_weight above 0): a planner that does not search "
                       "on wellness cannot be faulted for not improving it",
    without_chosen="each outcome's without_chosen block is the same statistic over the four candidates the "
                   "planner did not choose (6 pairs a start): the chosen order was selected by searching this "
                   "forecast, so the pairs containing it favor the forecast's own errors")


def ranking_summary(rows, draws=BOOT_DRAWS):
    """The outside review's statistic (mimo_bridge.tools.command_ranking) for every outcome in
    RANKING_OUTCOMES over the per-start rows: the pairs of candidates the forecast orders as their
    actual outcomes are ordered, in 100, with its body interval, the spread ratio and the case label,
    and the same over the four candidates the planner did not choose (RANKING_NOTES). Wellness is
    negated so that lower is better for every outcome."""
    from mimo_bridge.tools.command_ranking import RULES, ranking

    def five(terms):
        return float(np.sum(terms))

    def starts(forecast, actual):
        return [dict(body=r["body"], names=list(r["candidates"]), predicted=forecast(r), actual=actual(r))
                for r in rows]

    def wellness_forecast(r):
        return [-five(t) for t in r["judge_pred"]["planner"]["1"]]

    def goal_forecast(r):
        return [float(x) for x in r["goal_distance_pred"]["1"]]
    outcomes = dict(
        goal_distance_50ms=starts(goal_forecast, lambda r: [c[0] for c in r["actual"]["goal_distance"]]),
        goal_distance_end=starts(goal_forecast, lambda r: [c[-1] for c in r["actual"]["goal_distance"]]),
        wellness_50ms=starts(wellness_forecast, lambda r: [-five(c[0]) for c in r["actual"]["wellness_terms"]]),
        wellness_tenure=starts(wellness_forecast, lambda r: [-float(np.mean([five(t) for t in c]))
                                                             for c in r["actual"]["wellness_terms"]]))
    wellness_searched = bool(rows) and float(rows[0].get("planner_wellness_weight") or 0.0) > 0.0

    def unchosen(s):
        out = []
        for st in s:
            keep = [i for i, n in enumerate(st["names"]) if n != RANKING_PLANNER]
            out.append(dict(st, names=[st["names"][i] for i in keep],
                            predicted=[st["predicted"][i] for i in keep], actual=[st["actual"][i] for i in keep]))
        return out
    out = {}
    for name, s in outcomes.items():
        judged = name.startswith("goal_distance") or wellness_searched
        block = ranking(s, RANKING_OUTCOMES[name], planners=(RANKING_PLANNER,) if judged else (),
                        reference=RANKING_REFERENCE if judged else None, draws=draws)
        block["without_chosen"] = ranking(unchosen(s), RANKING_OUTCOMES[name] + " (the four candidates the planner "
                                          "did not choose)", draws=draws)
        out[name] = block
    out.update(planner=RANKING_PLANNER, reference=RANKING_REFERENCE, wellness_searched=wellness_searched,
               rules=RULES, notes=RANKING_NOTES)
    return out


def ranking_rules():
    """The ranking's rules and outcomes as declaration.json records them, before anything is measured."""
    from mimo_bridge.tools.command_ranking import RULES
    return dict(RULES, outcomes=RANKING_OUTCOMES, planner=RANKING_PLANNER, reference=RANKING_REFERENCE,
                notes=RANKING_NOTES)


def decide(summary):
    """The three-way verdict from the summary, by the rules declared before measurement."""
    ev = summary["evaluator"]["transfer"]
    fm = summary["forward_model"]["transfer_judge_units"]
    cs = ((summary["forward_model"]["senses"].get("planner") or {}).get("1") or {}).get("judge_senses", {})
    cs = cs.get("captured_share") if cs else None
    evaluator_flat = not excludes_zero_above(ev)
    forward_flat = not (excludes_zero_above(fm) and excludes_zero_above(cs))
    pl = summary["planner"]
    rank = pl["chosen_rank_by_predicted_wellness"]
    rank_better_than_chance = bool(rank.get("interval")) and rank["interval"][1] < pl["chance_rank"]
    planner_ignores = (pl.get("objective_wellness_weight") in (0, 0.0)) and not rank_better_than_chance
    tv = lambda s: (s or {}).get("value")
    order = [("forward model", forward_flat), ("evaluator", evaluator_flat),
             ("planner", planner_ignores)]
    flat = [name for name, f in order if f]
    if not flat:
        a, b = tv(fm), tv(ev)
        smaller = None if a is None or b is None else ("forward model" if a < b else "evaluator")
        first = "none flat; the smaller transfer is the " + str(smaller)
    else:
        first = flat[0]
    primary = summary["depth"]["wellness"]["wellness_tenure_depth3_minus_depth1"]
    claim = bool(primary.get("interval")) and primary["interval"][0] > 0
    return dict(forward_model_flat=bool(forward_flat), evaluator_flat=bool(evaluator_flat),
                planner_ignores_predicted_wellness=bool(planner_ignores), flat_stages=flat,
                first_flat_stage_along_the_route=first,
                forward_model_transfer=tv(fm), evaluator_transfer=tv(ev),
                forward_model_captured_share=tv(cs),
                # False = the pooled real difference was within its float32 floor (no share; the rule reads flat)
                forward_model_captured_share_measurable=(cs or {}).get("measurable"),
                depth3_better_physical_decision=claim,
                rules=DECISION_RULES)


DECISION_RULES = dict(
    forward_model_flat="flat unless BOTH the slope of Judge B's predicted-outcome difference on its "
                       "actual-outcome difference AND the pooled captured share of the between-candidate "
                       "token difference over Judge B's four senses (planner route, 50 ms) have 95% body "
                       "intervals entirely above 0",
    evaluator_flat="flat unless the slope of Judge B's actual-outcome difference on the grounded wellness "
                   "difference (50 ms) has a 95% body interval entirely above 0",
    planner_ignores_predicted_wellness="the planner's objective gives predicted wellness weight 0 "
                                       "(intention_gen._wellness_w) and the chosen order's mean rank by "
                                       "predicted wellness is not better than chance (the interval's upper "
                                       "end is not below (K + 1) / 2)",
    depth_claim="looking further ahead causes a better physical decision only when the order selected by "
                "predicted wellness at 150 ms beats the order selected at 50 ms in actual grounded wellness "
                "averaged over the order's tenure, paired over the same starts, 95% body interval above 0")


# ============================================================================ the model side
def weights_digest(model):
    digest = hashlib.sha256()
    for parameter in model.params:
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def stale_scheduler(repo=REPO, process=None):
    """(True, pid) when this process was started by an ability scheduler that began before the last
    change to tools/ability_rechecks.py: its in-memory result validator may not know this probe, and
    a full run would be rejected after it finished. (False, None) when started by hand."""
    import psutil
    module = Path(repo) / "Source/Python/mimo_bridge/tools/ability_rechecks.py"
    try:
        changed = module.stat().st_mtime
        for ancestor in (process or psutil.Process()).parents():
            try:
                command = ancestor.cmdline()
            except (psutil.Error, OSError):
                continue
            if "mimo_bridge.tools.ability_rechecks" in command:
                return ancestor.create_time() < changed, ancestor.pid
    except (psutil.Error, OSError):
        return False, None
    return False, None


def candidate_raws(model, plan0, motor, achieved, exec_prev, step):
    """The five raw candidates exactly as LemLite._causal_plan_record forms them."""
    import torch
    gen = torch.Generator(device="cpu")
    gen.manual_seed(perturb_seed(model._causal_seed, step))
    return {"chosen": plan0.detach().float().clone(),
            "hold": hold_candidate(plan0, achieved, exec_prev, model.GAZE_IDX),
            "reflex": motor.detach().float().clone(),
            "perturb_1": model._causal_perturb(plan0, gen),
            "perturb_2": model._causal_perturb(plan0, gen)}


def executed_candidate(model, raw, decision, raw_joint_goal=None):
    """(execution-form order [1, H, 9] in cm, smoothed hand joint goal or None): the decision's
    own execution transform and smoother, and the hand path LemLite._causal_execution_candidates
    applies (the planned raw joint goal for the chosen order, the controller's projection else)."""
    import torch
    command = model.planned_execution(raw.float(), decision["dev"], handle_state=decision["achieved"],
                                      previous=decision.get("smooth_prev"),
                                      smoothing=decision.get("smoothing", 0.0),
                                      keep_gaze_idx=decision.get("smooth_keep_gaze"))
    joints = None
    if getattr(model, "hand_target_mode", "world") != "world":
        if raw_joint_goal is None:
            rawx = model.differentiable_execution_transform(raw.float(), decision["dev"],
                                                            handle_state=decision["achieved"])
            _, jarr, _ = model.hand_kinematics.project_world_orders(rawx.detach().cpu().numpy(),
                                                                    position_scale=1.)
            raw_joint_goal = torch.as_tensor(jarr, device=command.device, dtype=command.dtype)
        joints = model._smooth_hand_joints(raw_joint_goal.to(command), decision.get("hand_previous_joints"),
                                           decision.get("smoothing", 0.0))
        command = model.hand_kinematics.assemble_world(command, joints, position_scale=1.)
    return command.detach(), (None if joints is None else joints.detach())


def replicate_state(st, K):
    from lem.model.predictive_state import PredictiveState

    def rep(t):
        return None if t is None else t.repeat(K, *([1] * (t.dim() - 1)))
    return PredictiveState(tokens={m: rep(v) for m, v in st.tokens.items()}, context=rep(st.context),
                           achieved=rep(st.achieved), domain="real", memory_bias=rep(st.memory_bias),
                           head_motion=rep(getattr(st, "head_motion", None)))


def goal_senses(decision, goal, senses=ALL_SENSES):
    mask = decision.get("source_mask") or {}
    return [m for m in senses if m in goal and str(getattr(mask.get(m, "real"), "value",
                                                            mask.get(m, "real"))) == "real"]


def goal_distance(tokens, goal, senses):
    """[K] the planner's distance: sum over senses of the mean squared distance to the goal."""
    import torch
    total = None
    for m in senses:
        if m in tokens:
            d = ((tokens[m].float().reshape(tokens[m].shape[0], -1)
                  - goal[m].float().reshape(1, -1)) ** 2).mean(-1)
            total = d if total is None else total + d
    return total if total is not None else torch.zeros(next(iter(tokens.values())).shape[0])


def executed_poses(observations, depths=DEPTHS):
    """{d: [K, H, F]} the achieved pose each candidate's branch reached after d - 1 executed model steps, for every
    depth d > 1 (observations[c][k] is candidate c's observation after k + 1 model steps; depth 1 reads the
    decision's pose and is absent here)."""
    import torch
    return {d: torch.cat([obs[d - 2]["achieved"].float() for obs in observations]) for d in depths if d > 1}


def predictions(model, st0, enc, K, goal, gsenses, depths=DEPTHS, poses=None):
    """Roll the K held orders forward from st0: per depth the predicted tokens, Judge B's terms, the
    goal distance, Judge B with the present motor token, and (depth 1) Judge B with the present
    senses and each candidate's motor token. poses: None = every depth reads the pose st0 carries (the
    decision's); {d: [K, H, F]} = depth d's transition reads that achieved pose (executed_poses)."""
    import torch
    from lem.model.predictive_state import transition
    wl = model.wellness
    st = replicate_state(st0, K)
    z_now = st0.tokens["motor"].float().repeat(K, 1)
    now_senses = {m: v.float().repeat(K, 1) for m, v in st0.tokens.items() if m != "motor"}
    out = {}
    with torch.no_grad():
        for d in range(1, max(depths) + 1):
            pose = None if poses is None else poses.get(d)
            st = transition(model.forward_model, model.encoder_registry, model.prediction_heads, st, enc,
                            achieved_state=pose, advance_context=True)
            preds = {m: v.float() for m, v in st.tokens.items() if m != "motor"}
            zm = st.tokens["motor"].float()
            rec = dict(tokens=preds, motor=zm, judge=wl.score_predicted_wellness(preds, zm).float(),
                       judge_fixed_motor=wl.score_predicted_wellness(preds, z_now).float(),
                       goal_distance=goal_distance(preds, goal, gsenses))
            if d == 1:
                rec["judge_motor_only"] = wl.score_predicted_wellness(now_senses, zm).float()
            out[d] = rec
    return out


def mean_eyes(env):
    values = np.asarray(env.eye_yaw_pitch(), dtype=np.float64)
    return values.reshape(2, 2).mean(axis=0)


def observation(session):
    """What one moment produced, read right after observe()."""
    import torch
    m = session.model
    with torch.no_grad():
        raw = {k: v.detach().float().reshape(1, -1).clone() for k, v in (m.last_tokens or {}).items()}
        fed = {k: v.detach().float().reshape(1, -1).clone() for k, v in (m.last_tokens_fed or {}).items()}
        judge = m.wellness.score_predicted_wellness({k: v for k, v in raw.items() if k != "motor"},
                                                    raw["motor"]).float().reshape(-1)
    g = m.last_wellness_grounded or {}
    terms = [float(g[n].reshape(-1)[0]) for n in FIVE_TERMS]
    pos, quat = session.pose()
    return dict(raw=raw, fed=fed, terms=terms, judge=[float(x) for x in judge],
                next_pe={k: float(v) for k, v in (m.last_next_pe or {}).items()},
                achieved=m.last_achieved.detach().float().clone(), pos_cm=pos, quat=quat,
                eyes=mean_eyes(session.env))


def run_arm(session, snap, order, gaze, joints, steps, observe):
    """Send `order` (held) for `steps` model steps from the restored snapshot, the eyes at `gaze`.
    Returns the physical trace's SHA-256 and, when observing, each moment's observation."""
    session.restore(snap)
    physical = hashlib.sha256()

    def record(s, _rep):
        d = s.env.data
        for arr in (d.qpos, d.qvel, np.asarray(s.env.eye_yaw_pitch(), dtype=np.float64)):
            a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
            if not np.isfinite(a).all():
                raise RuntimeError("non-finite physical state in a candidate branch")
            physical.update(a.tobytes())
        meta = s.frames[0].meta
        if joints is not None:
            physical.update(np.ascontiguousarray(np.asarray(meta["executed_motor_targets"],
                                                            dtype=np.float64)).tobytes())
    obs = []
    for k in range(int(steps)):
        session.act(order, gaze_rad=gaze, new_order=(k == 0), hand_joint_goal=joints, on_env_step=record)
        if observe:
            session.observe()
            obs.append(observation(session))
    out = dict(physical_sha256=physical.hexdigest())
    if observe:
        h = hashlib.sha256()
        for o in obs:
            for key in sorted(o["raw"]):
                h.update(o["raw"][key].numpy().tobytes())
            h.update(np.asarray(o["terms"], np.float64).tobytes())
            h.update(json.dumps(sorted(o["next_pe"].items())).encode())
        out.update(observations=obs, observation_sha256=h.hexdigest())
    return out


def control_of(model, order_cm, ach0, ach1, gaze_cmd, eyes_end):
    """Did the body do what was ordered, over the order's tenure: the share of every ordered body
    handle turn the body closed (1 - remaining / ordered, pooled; 1 = arrived, 0 = no nearer, below 0
    = further away), how many handles ended nearer their target than they began and how many a
    random axis would give (motor_synergy.following_counts, its exact control), the share of the
    ordered position change of the position handles the body closed, and the eyes' distance from
    their command in degrees."""
    import torch
    from lem.model.motor_synergy import following_counts
    from mimo_bridge.action_decoder import POSITION_HANDLES
    gi = model.GAZE_IDX
    rot_idx = [i for i in range(order_cm.shape[1]) if i != gi]
    c = following_counts(order_cm.double(), ach0.double(), ach1.double(), rot_idx)
    n = int(c["closer"].numel())
    ordered = float(c["ordered_deg"].sum()) if n else 0.0
    turn = None if not ordered > 0 else 1.0 - float(c["remaining_deg"].sum()) / ordered
    pidx = [i for i, name in enumerate(model.handle_names) if name in POSITION_HANDLES]
    tgt = order_cm[0, pidx, 0:3].double()
    start = ach0[0, pidx, 0:3].double() * 200.0
    end = ach1[0, pidx, 0:3].double() * 200.0
    asked = float((tgt - start).norm(dim=-1).sum())
    left = float((tgt - end).norm(dim=-1).sum())
    eye = float(np.degrees(np.linalg.norm(np.asarray(eyes_end, float) - np.asarray(gaze_cmd, float))))
    return dict(turn_closed_share=turn, closer_share=(float(c["closer"].float().mean()) if n else None),
                chance_share=(float(c["chance"].mean()) if n else None), handles=n,
                position_closed_share=(None if not asked > 0 else 1.0 - left / asked),
                eye_error_deg=eye)


def structural_checks(model, pi):
    """What the running code does with depth: the production chain asked for 3 steps, and (the hand
    path set aside) the first order of a 1-step and a 3-step chain from the same captured decision."""
    import torch
    out = dict(production_depth_forced_to_one=getattr(model, "hand_target_mode", "world") != "world")
    cur = lambda: {m: v.clone() for m, v in pi["cur"].items()}
    goal = lambda: {m: v.clone() for m, v in pi["goal"].items()}
    with torch.random.fork_rng(devices=[]), torch.enable_grad():
        plan, meta = model._plan_chain(cur(), goal(), pi["motor"].clone(), max_steps=3,
                                       decision=pi["decision"])
    out["production_asked_3_returned_length"] = len(plan)
    out["production_stopped_by"] = meta.get("stopped_by")
    saved = (model.hand_target_mode, model.hand_goals)
    try:
        model.hand_target_mode, model.hand_goals = "world", "off"
        dec = {k: v for k, v in pi["decision"].items() if k not in ("hand_goal", "physical_goal")}
        plans = {}
        for depth in (1, 3):
            with torch.random.fork_rng(devices=[]), torch.enable_grad():
                plans[depth] = model._plan_chain(cur(), goal(), pi["motor"].clone(), max_steps=depth,
                                                 decision=dec)
        (p1, m1), (p3, m3) = plans[1], plans[3]
        out.update(world_mode_plan_length={"1": len(p1), "3": len(p3)},
                   world_mode_stopped_by={"1": m1.get("stopped_by"), "3": m3.get("stopped_by")},
                   world_mode_first_order_identical=(bool(torch.equal(p1[0], p3[0])) if p1 and p3 else None),
                   world_mode_deeper_plan_emptied=bool(p1) and not bool(p3),
                   world_mode_goal_distance_by_depth=m3.get("mse_by_depth"))
    finally:
        model.hand_target_mode, model.hand_goals = saved
    return out


def next_decision(session, minimum_steps, bound=80):
    """Live on under the model's own behaviour until a planning event with a recorded plan, at least
    `minimum_steps` model steps on; the snapshot is taken right after that observation."""
    from mimo_bridge.tools import gof_probe_harness as H
    model = session.model
    for step in range(int(minimum_steps) + int(bound)):
        prior = model._last_plan_inputs
        session.observe()
        pi = H.fresh_plan_inputs(model, prior)
        rec = model.last_plan_record
        if (step >= minimum_steps and pi is not None and pi.get("plan0") is not None
                and isinstance(rec, dict) and "error" not in rec
                and rec.get("step") == model.step_count - 1):
            return session.snapshot(), step
        session.policy_act()
    raise RuntimeError("no planning event with a recorded plan within the declared search bound")


def measure_start(session, snap, body, start, tenure, out_dir, checks):
    """One saved decision: predict, execute and score the five candidates."""
    import torch
    from lem.model.predictive_state import PredictiveState
    from mimo_bridge.lem_model import command_for_encoder
    model = session.model
    session.restore(snap)
    pi = model._last_plan_inputs
    rec = model.last_plan_record
    decision = pi["decision"]
    plan0, motor = pi["plan0"].float(), pi["motor"].float()
    hand_chosen = getattr(model, "_hand_chosen_plan", None)
    exec_prev = None if snap["executed"] is None else torch.as_tensor(np.asarray(snap["executed"], np.float32))
    step = int(rec["step"])
    raws = candidate_raws(model, plan0, motor, decision["achieved"], exec_prev, step)
    names = list(CANDIDATES)
    K = len(names)
    execs = {n: executed_candidate(model, raws[n], decision,
                                   hand_chosen if (n == "chosen" and hand_chosen is not None) else None)
             for n in names}
    route = {}
    with torch.no_grad():
        ref = model._causal_execution_candidates([raws[n] for n in names[1:]], decision)
        mine = torch.cat([execs[n][0] for n in names[1:]])
        route["alternatives_execution_max_abs_cm"] = float((ref - mine).abs().max())
        res = snap["res"]
        gi = model.GAZE_IDX
        body_rows = [i for i in range(len(model.handle_names)) if i != gi]
        prod = torch.as_tensor(np.asarray(res.motor_for_action, np.float32))
        route["chosen_vs_production_order_max_abs"] = float(
            (execs["chosen"][0][:, body_rows] - prod[:, body_rows]).abs().max())
        if execs["chosen"][1] is not None and getattr(res, "hand_joint_goal", None) is not None:
            route["chosen_vs_production_hand_goal_max_abs_rad"] = float(
                (execs["chosen"][1] - torch.as_tensor(np.asarray(res.hand_joint_goal, np.float32))).abs().max())
    orders = torch.cat([execs[n][0] for n in names]).float()
    enc = command_for_encoder(orders)
    goal = {m: v.float() for m, v in pi["goal"].items()}
    # the production record's rule: every predicted sense (all but motor) present in the goal and
    # real in the decision's source mask
    gsenses = goal_senses(decision, goal, senses=[m for m in pi["cur"] if m != "motor"])
    raw_now = {m: v.float().reshape(1, -1) for m, v in model.last_tokens.items()}
    st_plan = PredictiveState(tokens={m: v.float().reshape(1, -1) for m, v in pi["cur"].items()},
                              context=decision["context"], achieved=decision["achieved"], domain="real",
                              memory_bias=decision.get("memory_bias"))
    st_learn = PredictiveState(tokens=raw_now, context=decision["context"], achieved=decision["achieved"],
                               domain="real", memory_bias=decision.get("memory_bias"))
    pred = {"planner": predictions(model, st_plan, enc, K, goal, gsenses),
            "learning": predictions(model, st_learn, enc, K, goal, gsenses)}
    # THE ROUTE CHECK: the same five candidates scored here and by the production planning record
    # at this decision (its alternatives through _causal_score_candidates, its chosen order from the
    # planner's own depth-1 state); a mismatch means this probe is not measuring what production saw
    rec_pw = rec.get("pred_wellness") or {}
    mine_pw = pred["planner"][1]["judge"].sum(-1).tolist()
    route["record_pred_wellness_max_abs"] = max(
        [abs(mine_pw[i] - float(rec_pw[n][0])) for i, n in enumerate(names) if rec_pw.get(n)] or [0.0])
    rec_gd = rec.get("goal_distance") or {}
    mine_gd = pred["planner"][1]["goal_distance"].tolist()
    route["record_goal_distance_max_abs"] = max(
        [abs(mine_gd[i] - float(rec_gd[n][0])) for i, n in enumerate(names) if rec_gd.get(n)] or [0.0])
    failed = {k: v for k, v in route.items() if k in ROUTE_TOLERANCE and not v <= ROUTE_TOLERANCE[k]}
    if failed:
        # stop before any branch runs: a probe off the production route would spend its saved
        # training pause measuring something production never computed
        raise RuntimeError(f"body {body} start {start}: the probe's candidates do not reproduce the "
                           f"production planning record ({failed}, allowed {ROUTE_TOLERANCE})")
    structural = None
    if checks:
        structural = structural_checks(model, pi)
        session.restore(snap)
    gazes = {n: np.asarray(model._gaze_from_handle(raws[n]), np.float64).reshape(-1)[:2] for n in names}
    arms, exact = {}, {}
    for n in names:
        order_np = execs[n][0].detach().cpu().numpy()
        joints_np = None if execs[n][1] is None else execs[n][1].detach().cpu().numpy()
        order_x = session.exec_order(order_np, gazes[n], hand_joint_goal=joints_np)
        first = run_arm(session, snap, order_x, gazes[n], joints_np, tenure, observe=True)
        second = run_arm(session, snap, order_x, gazes[n], joints_np, tenure, observe=False)
        if first["physical_sha256"] != second["physical_sha256"]:
            raise RuntimeError(f"body {body} start {start} candidate {n}: the physical replay differed")
        exact[n] = dict(physical=True)
        if checks and n == "chosen":
            third = run_arm(session, snap, order_x, gazes[n], joints_np, tenure, observe=True)
            if third["observation_sha256"] != first["observation_sha256"]:
                raise RuntimeError(f"body {body} start {start}: the observations of a replay differed")
            exact[n]["observations"] = True
        arms[n] = first
    session.restore(snap)
    # ------------------------------------------------ per-start numbers
    T = int(tenure)
    obs = [arms[n]["observations"] for n in names]
    # EXECUTED POSE (v3): the same transitions again from the same states, depth d reading each candidate's own
    # achieved pose after d - 1 executed model steps (DEPTH_POSE_RULE). Depth 1 reads the decision's pose in both
    # passes, so it must come out the same: a difference means the model's state moved between the passes
    poses = executed_poses(obs)
    pred_exec = {"planner": predictions(model, st_plan, enc, K, goal, gsenses, poses=poses),
                 "learning": predictions(model, st_learn, enc, K, goal, gsenses, poses=poses)}
    route["executed_pose_depth1_max_abs"] = max(
        float((pred_exec[rn][1]["tokens"][m] - pred[rn][1]["tokens"][m]).abs().max())
        for rn in pred for m in pred[rn][1]["tokens"])
    if not route["executed_pose_depth1_max_abs"] <= EXECUTED_POSE_DEPTH1_TOLERANCE:
        raise RuntimeError(f"body {body} start {start}: the executed-pose pass changed the depth-1 prediction by "
                           f"{route['executed_pose_depth1_max_abs']} (allowed {EXECUTED_POSE_DEPTH1_TOLERANCE}): the "
                           "model's state moved between the two passes")
    ach0 = decision["achieved"].float()
    executed_pose_shift_cm = {str(d): [float(v) for v in ((p[..., 0:3] - ach0[..., 0:3]) * 200.0).norm(dim=-1)
                                       .mean(-1)] for d, p in poses.items()}
    sense_stats, sense_stats_decision_pose = {}, {}
    for key, source, kept in (("executed", pred_exec, DEPTHS), ("decision", pred, tuple(d for d in DEPTHS if d > 1))):
        target = sense_stats if key == "executed" else sense_stats_decision_pose
        for rname, rp in source.items():
            for d in kept:
                if d > T:
                    continue
                for m in ALL_SENSES:
                    if m not in rp[d]["tokens"] or m not in raw_now:
                        continue
                    actual = torch.cat([obs[c][d - 1]["raw"][m] for c in range(K)]).numpy()
                    target.setdefault(rname, {}).setdefault(str(d), {})[m] = sense_pair_stats(
                        rp[d]["tokens"][m].numpy(), actual, raw_now[m].numpy())
    route_disagreement = {m: [float(((pred["planner"][1]["tokens"][m][c] - pred["learning"][1]["tokens"][m][c]) ** 2)
                                    .mean()) for c in range(K)]
                          for m in ALL_SENSES if m in pred["planner"][1]["tokens"]}
    zm = pred["planner"][1]["motor"]
    z0 = zm[0]
    motor_change = [float((zm[c] - z0).norm() / z0.norm().clamp_min(1e-12)) for c in range(K)]
    effort = [float(model.intention_gen._strength_reg * (raws[n][..., 8].float() ** 2).sum()) for n in names]
    control = {key: [] for key in ("turn_closed_share", "closer_share", "chance_share",
                                   "position_closed_share", "eye_error_deg", "handles")}
    for c, n in enumerate(names):
        ct = control_of(model, execs[n][0].float(), decision["achieved"].float(), obs[c][-1]["achieved"],
                        gazes[n], obs[c][-1]["eyes"])
        for key in control:
            control[key].append(ct[key])
    actual = dict(
        wellness_terms=[[o["terms"] for o in obs[c]] for c in range(K)],
        judge_terms=[[o["judge"] for o in obs[c]] for c in range(K)],
        next_pe_sum=[[sum(o["next_pe"].get(s, 0.0) for s in JUDGE_SENSES) for o in obs[c]] for c in range(K)],
        next_pe=[[o["next_pe"] for o in obs[c]] for c in range(K)],
        goal_distance=[[float(goal_distance({m: o["fed"][m] for m in gsenses if m in o["fed"]}, goal, gsenses)[0])
                        for o in obs[c]] for c in range(K)])
    # the production outcome of this decision, from the record's own sums and the production rule
    # (the model's in-memory record carries the sums; the recorder adds its gaps only to the JSON line)
    rec_sums = [float(rec_pw[n][0]) for n in names if rec_pw.get(n)]
    rec_gap = (rec_sums[0] - max(rec_sums[1:])) if len(rec_sums) == K else None
    row = dict(
        body=int(body), start=int(start), step=step, candidates=names, tenure_steps=T,
        script_sha256=script_sha256(), depth_pose=DEPTH_POSE,
        # per depth d > 1, per candidate: the mean distance (cm, over the handles) between the pose its branch
        # reached after d - 1 model steps and the decision's pose (0 = the depth-0 pose was still right)
        executed_pose_shift_cm=executed_pose_shift_cm, sense_stats_decision_pose=sense_stats_decision_pose,
        planner_wellness_weight=float(getattr(model.intention_gen, "_wellness_w", 0.0)),
        judge_pred={rn: {str(d): rp[d]["judge"].tolist() for d in DEPTHS} for rn, rp in pred.items()},
        judge_pred_fixed_motor={str(d): pred["planner"][d]["judge_fixed_motor"].tolist() for d in DEPTHS},
        judge_pred_motor_only=pred["planner"][1]["judge_motor_only"].tolist(),
        goal_distance_pred={str(d): pred["planner"][d]["goal_distance"].tolist() for d in DEPTHS},
        goal_senses=gsenses, effort=effort, sense_stats=sense_stats,
        route_disagreement=route_disagreement, motor_token_change=motor_change,
        actual=actual, control=control, route_checks=route, exact_replay=exact, structural=structural,
        gaze_command_deg={n: np.degrees(gazes[n]).tolist() for n in names},
        production=dict(pred_wellness={k: v for k, v in rec_pw.items()},
                        goal_distance={k: v for k, v in rec_gd.items()},
                        outcome=(None if rec_gap is None else [tie_outcome(rec_gap, PRODUCTION_TIE_BAND)]),
                        gap_to_best_alt=rec_gap, chosen_rank=rec.get("chosen_rank"), horizon=rec.get("horizon")),
        plan=dict(horizon=pi.get("max_steps"), outcome=pi.get("outcome")))
    np.savez_compressed(
        out_dir / f"start_{body}_{start}.npz",
        **{f"pred_{rn}_d{d}_{m}": rp[d]["tokens"][m].numpy() for rn, rp in pred_exec.items() for d in DEPTHS
           for m in rp[d]["tokens"]},
        **{f"pred_decisionpose_{rn}_d{d}_{m}": rp[d]["tokens"][m].numpy() for rn, rp in pred.items() for d in DEPTHS
           if d > 1 for m in rp[d]["tokens"]},
        **{f"actual_t{k + 1}_{m}": torch.cat([obs[c][k]["raw"][m] for c in range(K)]).numpy()
           for k in range(T) for m in obs[0][k]["raw"]},
        **{f"now_raw_{m}": v.numpy() for m, v in raw_now.items()},
        **{f"now_planner_{m}": v.float().reshape(1, -1).numpy() for m, v in pi["cur"].items()},
        **{f"goal_{m}": v.reshape(1, -1).numpy() for m, v in goal.items()})
    return row


# ============================================================================ main
def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--out", type=Path, help="a folder that must not exist yet (the scheduler's form)")
    where.add_argument("--out-root", type=Path,
                       help="a folder under which <run>/<checkpoint>_<time> is created (a reusable job entry)")
    parser.add_argument("--bodies", type=int, default=8)
    parser.add_argument("--starts-per-body", type=int, default=2)
    parser.add_argument("--spacing", type=int, default=10, help="model steps between a body's starts")
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args(argv)
    if args.out is None:
        ckpt = Path(args.checkpoint)
        args.out = (args.out_root / ckpt.parent.parent.name
                    / (ckpt.name + time.strftime("_%Y%m%d_%H%M%S")))
    began = time.time()
    stale, pid = stale_scheduler()
    if stale:
        raise RuntimeError(f"the ability scheduler (process {pid}) started before the last change to "
                           "tools/ability_rechecks.py, so it may not hold this probe's result validator "
                           "and would reject a complete result; restart the scheduler, then it retries")
    from mimo_bridge.tools.commit_guard import commit_reading, verdict
    reading = commit_reading()
    if reading is not None and not verdict(reading[0], reading[1], NEED_GIB, RESERVE_GIB):
        raise RuntimeError(f"Memory guard refused: {reading}")
    args.out.mkdir(parents=True, exist_ok=False)
    script_sha = script_sha256()
    declaration = dict(
        probe=PROBE_VERSION, checkpoint=str(args.checkpoint), script_sha256=script_sha,
        depth_pose=DEPTH_POSE, depth_pose_rule=DEPTH_POSE_RULE, noise_floor_rule=NOISE_FLOOR_RULE,
        executed_pose_depth1_tolerance=EXECUTED_POSE_DEPTH1_TOLERANCE,
        source_sha256={s: sha256_file(REPO / s) for s in SOURCES if (REPO / s).exists()},
        bodies=int(args.bodies), starts_per_body=int(args.starts_per_body), spacing_steps=int(args.spacing),
        candidates=list(CANDIDATES), depths=list(DEPTHS), learning=False, production_adoption=False,
        candidate_definition="LemLite._causal_plan_record's candidates at the saved planning event: the "
                             "planner's chosen first order, hold (every handle where it is, the eyes at the "
                             "last realised eye command), the controller's own proposal, and two perturbations "
                             "of the chosen order at the babble's sizes from the production seed",
        execution="each candidate through the decision's own execution transform, smoother and hand path; "
                  "held for the order's tenure (motor_hold moments) with the eyes at its gaze command; the "
                  "saved world's reflex and saccades run as in production; every branch replayed and its "
                  "physical trace required bit-identical",
        rules=DECISION_RULES, production_tie_band=PRODUCTION_TIE_BAND,
        ranking_rules=ranking_rules(),
        guard=None if reading is None else dict(committed_gb=reading[0], limit_gb=reading[1],
                                                need_gb=NEED_GIB, reserve_gb=RESERVE_GIB),
        memory_budget_note="12 GiB declared from the sibling probes' budget (the harness measured the v4 "
                           "probe suite at 8.24 GB peak private commit); this run records its own peak")
    write_json(args.out / "declaration.json", declaration)
    import torch
    from mimo_bridge.tools import gof_probe_harness as H
    frozen = H.load_frozen(args.checkpoint, intention=True, threads=args.threads)
    episode, note = H.load_episode(frozen)
    if episode is None or len(episode["worlds"]) < args.bodies:
        raise RuntimeError(f"{args.bodies} saved worlds are required; no fresh-world fallback ({note})")
    model = frozen.model
    if model.intention_gen is None or not getattr(model, "causal_chain", False):
        raise RuntimeError("the checkpoint's build has no planner or no causal-chain record")
    before = weights_digest(model)
    for parameter in model.params:
        parameter.requires_grad_(False)
    rows = []
    total = args.bodies * args.starts_per_body
    tenure = None
    for body in range(args.bodies):
        session = H.Session(frozen, episode=episode, body=body, babble=float(frozen.args.babble_noise))
        try:
            if abs(session.clock.model_step_seconds - 0.05) > 1e-8:
                raise RuntimeError("the declared horizons need 50 ms model steps")
            tenure = int(session.hold)
            if tenure < max(DEPTHS):
                raise RuntimeError("the order's tenure is shorter than the deepest horizon compared")
            for k in range(args.starts_per_body):
                snap, advanced = next_decision(session, 0 if k == 0 else args.spacing)
                row = measure_start(session, snap, body, k, tenure, args.out, checks=(k == 0))
                row["advanced_steps"] = int(advanced)
                row["start_note"] = session.start
                rows.append(row)
                with (args.out / "starts.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(clean(row), allow_nan=False) + "\n")
                write_json(args.out / "status.json", dict(stage="measuring", completed_starts=len(rows),
                                                          total_starts=total, elapsed_s=time.time() - began,
                                                          probe=PROBE_VERSION, script_sha256=script_sha))
                print(json.dumps(dict(body=body, start=k, step=row["step"],
                                      production_outcome=row["production"]["outcome"])), flush=True)
                session.restore(snap)
                session.policy_act()
        finally:
            session.close()
    if weights_digest(model) != before:
        raise RuntimeError("the frozen probe changed learned weights")
    summary = summarise(rows)
    verdict_block = decide(summary)
    try:
        import psutil
        mem = psutil.Process().memory_info()
        peak = getattr(mem, "peak_pagefile", None)
        memory = dict(peak_private_commit_gib=None if peak is None else peak / 2 ** 30,
                      private_commit_gib=getattr(mem, "private", mem.vms) / 2 ** 30)
    except Exception:
        memory = None
    route_checks = {key: max(float(r["route_checks"].get(key, 0.0)) for r in rows)
                    for key in rows[0]["route_checks"]} if rows else {}
    result = dict(
        declaration, checkpoint_episode=note, evaluated_starts=len(rows), tenure_steps=tenure,
        model_step_ms=50, summary=summary, verdict=verdict_block,
        passed=bool(verdict_block["depth3_better_physical_decision"]),
        route_checks_max=route_checks,
        exact_replay=True, all_candidate_traces_replayed_exactly=True,
        observation_replay_checks=sum(1 for r in rows if r["exact_replay"]["chosen"].get("observations")),
        weights_unchanged=True, memory=memory, elapsed_s=time.time() - began,
        definitions=dict(
            tie_band="the approximate evaluator's error on the difference between two candidates, 90th "
                     "percentile; a predicted gap inside it counts as tied (0-to-5 wellness sum)",
            transfer="how much of one stage's difference between two candidates reaches the next stage "
                     "(regression slope through zero over candidate pairs): 1 = all of it, 0 = none",
            captured_share="how much of the ACTUAL token difference between two candidates the PREDICTED "
                           "difference foresaw along its direction: 1 = all of it, 0 = none, negative = the "
                           "wrong way; null with measurable false when the actual difference is within the "
                           "token's float32 floor (noise_floor_rule). Depths 2 and 3 read each candidate's "
                           "executed pose (depth_pose_rule); senses_decision_pose holds them on the decision's pose",
            turn_closed_share="of every body handle's ordered turn, the share the body closed over the "
                              "order's tenure: 1 = arrived, 0 = no nearer, below 0 = further away",
            eye_error_deg="the eyes' mean azimuth/elevation distance from their command at the order's end, "
                          "degrees; the decoder drives each eye joint to its signed command inside the joint "
                          "limits, so this is the travel still left (servo speed, joint limits, reflex and "
                          "saccade overrides); hold commands where the eyes already are",
            depth="1 = 50 ms, 3 = 150 ms (the order held); the order's tenure is motor_hold moments"))
    write_json(args.out / "result.json", result)
    write_json(args.out / "status.json", dict(stage="complete", completed_starts=len(rows), total_starts=total,
                                              elapsed_s=time.time() - began, probe=PROBE_VERSION,
                                              script_sha256=script_sha))
    print(json.dumps(dict(result=str(args.out / "result.json"), verdict=clean(verdict_block),
                          elapsed_s=result["elapsed_s"])), flush=True)


if __name__ == "__main__":
    main()
