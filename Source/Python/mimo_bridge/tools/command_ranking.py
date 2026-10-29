# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Does the forecast rank candidate commands the way reality does? (outside review, 2026-09-24)

THE STATISTIC. Of all pairs of candidate commands tried from the same saved state, how many in 100 the
model's forecast picks the pair's actual winner: 50 = chance (the ranking gives a planner nothing to
optimize), 80 to 90 = a ranking a planner can optimize, 100 = every pair ordered as reality orders it.
docs/gain_of_function_v4/candidate_ranking.py computed it once from saved results; the two scheduled
probes (docs/gain_of_function_v4/probe_planner_candidates.py and docs/gain_of_function_v3/
compare_hand_body_goals_2026-09-24.py) carry it in every result through this module, so every round
has it and both compute it one way.

  pairs          every pair of one start's candidates; lower = better for forecast and outcome alike.
                 A pair whose actual outcomes differ by no more than TIE_SHARE of that start's actual
                 spread (the standard deviation over its candidates) has no actual winner and is left
                 out; a pair the forecast scores exactly equal counts half right (the forecast offers
                 a coin flip, never a pick). A start with a missing or non-finite value is left out.
                 TIE_SHARE is the coordinator's first computation's (docs/gain_of_function_v4/
                 candidate_ranking.py TIE_EPS), kept so the two stay comparable; the branches replay
                 bit-identically, so it is not a noise floor. It left out 4 and 2 of the step-399,100
                 planner round's 160 wellness pairs and 1 of the 7 hand rounds' 336 pairs; every block
                 reports its count.
  interval       95%, whole bodies resampled with replacement (a body's starts share its world, so a
                 body is one sample), DRAWS draws.
  spread ratio   per start, the standard deviation of the candidates' forecasts over that of their
                 actual outcomes; the median over starts. 0 = the forecast gives every command the same
                 outcome, 1 = it separates the commands as much as reality does, above 1 = it
                 exaggerates their differences. Forecast and outcome are the same quantity in the same
                 units in every use here, so the ratio has no unit.

THE CASE LABEL (the reviewer's three failure cases), by RULES, declared before any result:
  better than chance (the interval's lower end above 50):
      case_3_planner_not_better  a planner's own command does not beat the reference command its
                                 search started from, in the actual outcome (the paired body interval
                                 of planner minus reference is not entirely below 0): a usable ranking
                                 the search does not turn into a better command
      ranking_used               every planner's command beats the reference
      better_than_chance         no planner comparison was given (the outcome is not what the planner
                                 searches on)
  below chance (the interval's upper end under 50):
      case_2_wrong_order         the forecast picks the loser more often than a coin: it responds to the
                                 command, the wrong way (below_chance is set)
  not measurably better than chance (the interval reaches 50):
      case_1_under_separated     spread ratio below 1: the forecast separates the commands less than
                                 reality does and its order is not measurably better than a coin. The
                                 reviewer's case 1 (a forecast that does not respond to the command) is
                                 its spread-near-0 end; the spread ratio says where on that scale a
                                 result sits, so the label claims no more than the number shows.
      case_2_wrong_order         spread ratio 1 or more: it separates them at least as much as reality
                                 does, in an order not measurably better than a coin
  undetermined                   fewer than two bodies, or no pair with an actual winner.
The spread line 1 is what a forecast that knew every outcome would read, not a tuned number.
"""
from __future__ import annotations

import math

import numpy as np

TIE_SHARE = 0.01
DRAWS = 5000
SEED = 240925
CHANCE = 50.0
CASES = ('ranking_used', 'better_than_chance', 'case_1_under_separated', 'case_2_wrong_order',
         'case_3_planner_not_better', 'undetermined',
         # the first name of case_1_under_separated (commit fd713b068): results written under it stay valid
         'case_1_action_insensitive')
RULES = dict(
    statistic="of all pairs of candidate commands tried from the same saved state, how many in 100 the "
              "forecast's order picks the pair's actual winner (lower = better for both); 50 = chance, "
              "80 to 90 = a ranking a planner can optimize",
    ties="a pair whose actual outcomes differ by no more than %d in 100 of that start's actual standard "
         "deviation has no winner and is left out; a pair the forecast scores exactly equal counts half right"
         % round(100 * TIE_SHARE),
    interval="95%%, whole bodies resampled with replacement, %d draws" % DRAWS,
    spread_ratio="median over starts of the standard deviation of the candidates' forecasts over that of "
                 "their actual outcomes: 0 = the forecast gives every command the same outcome, 1 = it "
                 "separates them as much as reality does",
    case="better than chance when the interval's lower end is above 50: then case_3_planner_not_better "
         "when a planner's command does not beat the reference command its search started from in the "
         "actual outcome (paired body interval of planner minus reference not entirely below 0), else "
         "ranking_used (better_than_chance without a planner comparison). Below chance (the interval's "
         "upper end under 50): case_2_wrong_order. Otherwise case_1_under_separated when the spread ratio "
         "is below 1 and case_2_wrong_order when it is 1 or more; undetermined with fewer than two bodies "
         "or no pair with an actual winner")


def _values(values):
    """A 1-D float array of at least two finite values, else None."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    return array if array.ndim == 1 and array.size >= 2 and np.isfinite(array).all() else None


def pair_counts(predicted, actual, tie_share=TIE_SHARE):
    """One start's pairs: dict(right, judged, actual_ties, forecast_ties), or None when a value is
    missing or non-finite. right counts a forecast tie as a half."""
    p, a = _values(predicted), _values(actual)
    if p is None or a is None or p.shape != a.shape:
        return None
    scale = float(np.std(a))
    right, judged, actual_ties, forecast_ties = 0.0, 0, 0, 0
    for i in range(a.size):
        for j in range(i + 1, a.size):
            da, dp = float(a[i] - a[j]), float(p[i] - p[j])
            if not abs(da) > tie_share * scale:
                actual_ties += 1
                continue
            judged += 1
            if dp == 0.0:
                forecast_ties += 1
                right += 0.5
            elif (dp > 0.0) == (da > 0.0):
                right += 1.0
    return dict(right=right, judged=judged, actual_ties=actual_ties, forecast_ties=forecast_ties)


def spread_ratio(predicted, actual):
    """std(forecasts) / std(actual outcomes) over one start's candidates; None without actual spread."""
    p, a = _values(predicted), _values(actual)
    if p is None or a is None or p.shape != a.shape:
        return None
    sa = float(np.std(a))
    return float(np.std(p)) / sa if sa > 0 else None


def _quantiles(values):
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def share_interval(per_body, draws=DRAWS, seed=SEED):
    """95% interval of the pooled right/judged share in 100, resampling bodies; per_body {body: (right,
    judged)}. None with fewer than two bodies."""
    keys = sorted(k for k, (_r, n) in per_body.items() if n > 0)
    if len(keys) < 2:
        return None
    rights = np.asarray([per_body[k][0] for k in keys], float)
    counts = np.asarray([per_body[k][1] for k in keys], float)
    pick = np.random.default_rng(seed).integers(len(keys), size=(int(draws), len(keys)))
    shares = 100.0 * rights[pick].sum(1) / np.maximum(counts[pick].sum(1), 1.0)
    return _quantiles(shares)


def median_interval(per_body, draws=DRAWS, seed=SEED):
    """95% interval of the median of the pooled per-start values, resampling bodies; per_body {body:
    [values]}. None with fewer than two bodies."""
    keys = sorted(k for k, v in per_body.items() if v)
    if len(keys) < 2:
        return None
    rng = np.random.default_rng(seed)
    medians = []
    for _ in range(int(draws)):
        pick = rng.integers(len(keys), size=len(keys))
        medians.append(float(np.median([x for i in pick for x in per_body[keys[i]]])))
    return _quantiles(medians)


def body_mean_interval(per_body, draws=DRAWS, seed=SEED):
    """The mean of the body means with a 95% interval from resampling bodies; per_body {body:
    [values]}. Lower = better: bodies_better counts body means below 0."""
    keys = sorted(k for k, v in per_body.items() if v)
    means = np.asarray([float(np.mean(per_body[k])) for k in keys], float)
    if not len(means):
        return dict(mean=None, interval=None, bodies=0, bodies_better=0, bodies_worse=0)
    interval = None
    if len(means) >= 2:
        pick = np.random.default_rng(seed).integers(len(means), size=(int(draws), len(means)))
        interval = _quantiles(means[pick].mean(-1))
    return dict(mean=float(means.mean()), interval=interval, bodies=len(keys),
                bodies_better=int((means < 0).sum()), bodies_worse=int((means > 0).sum()))


def case_of(interval, spread, planners=None):
    """The case label from RULES. interval: the statistic's 95% interval in 100 (or None); spread: the
    spread ratio (or None); planners: {name: body_mean_interval of planner minus reference} or None."""
    if not interval:
        return 'undetermined'
    if interval[0] > CHANCE:
        if not planners:
            return 'better_than_chance'
        better = all(bool(p.get('interval')) and p['interval'][1] < 0 for p in planners.values())
        return 'ranking_used' if better else 'case_3_planner_not_better'
    if interval[1] < CHANCE:
        return 'case_2_wrong_order'           # it picks the loser: it responds to the command, wrongly
    if spread is None:
        return 'undetermined'
    return 'case_1_under_separated' if spread < 1.0 else 'case_2_wrong_order'


def ranking(starts, what, *, planners=(), reference=None, draws=DRAWS, seed=SEED, tie_share=TIE_SHARE):
    """The statistic over saved starts, with its interval, spread ratio and case label.

    starts: [dict(body, predicted=[K], actual=[K], names=[K])], lower = better for both. planners and
    reference: candidate names; when given, each planner's actual outcome minus the reference's is
    compared per start (body means, resampled over bodies) and decides case 3."""
    per_body, spreads = {}, {}
    differences = {name: {} for name in planners}
    totals = dict(right=0.0, judged=0, actual_ties=0, forecast_ties=0)
    used = left_out = 0
    for start in starts:
        counts = pair_counts(start.get('predicted'), start.get('actual'), tie_share)
        if counts is None:
            left_out += 1
            continue
        used += 1
        body = start['body']
        for key in totals:
            totals[key] += counts[key]
        right, judged = per_body.get(body, (0.0, 0))
        per_body[body] = (right + counts['right'], judged + counts['judged'])
        ratio = spread_ratio(start['predicted'], start['actual'])
        if ratio is not None:
            spreads.setdefault(body, []).append(ratio)
        names = list(start.get('names') or [])
        if reference is not None and reference in names:
            actual = [float(x) for x in start['actual']]
            for name in planners:
                if name in names:
                    differences[name].setdefault(body, []).append(
                        actual[names.index(name)] - actual[names.index(reference)])
    judged = int(totals['judged'])
    value = 100.0 * totals['right'] / judged if judged else None
    interval = share_interval(per_body, draws, seed) if judged else None
    pooled = [x for body in sorted(spreads) for x in spreads[body]]
    spread = float(np.median(pooled)) if pooled else None
    compared = {name: body_mean_interval(d, draws, seed) for name, d in differences.items() if d}
    return dict(what=what, per_100=value, interval=interval, chance_per_100=CHANCE,
                pairs_judged=judged, pairs_right=float(totals['right']),
                actual_ties_left_out=int(totals['actual_ties']),
                forecast_ties_counted_half=int(totals['forecast_ties']),
                starts=used, starts_left_out=left_out, bodies=len([b for b, (_r, n) in per_body.items() if n]),
                spread_ratio=spread, spread_ratio_interval=median_interval(spreads, draws, seed),
                reference=reference, planners=compared,
                better_than_chance=bool(interval) and interval[0] > CHANCE,
                below_chance=bool(interval) and interval[1] < CHANCE,
                case=case_of(interval, spread, compared if planners else None))


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def valid_block(block, allow_undetermined=False, allow_empty=None):
    """Whether a result's ranking block (ranking() output, read back from JSON) is complete: the value
    and its interval inside 0 to 100, judged pairs, a finite spread ratio and a declared case.
    allow_undetermined: also accept the two measured outcomes that have no interval - no pair with an
    actual winner (every candidate ended exactly where the others did; no value), or judged pairs in
    fewer than two bodies (a value, no interval) - when the block says 'undetermined'. allow_empty is
    its first name, still passed by a scheduler started on the code of commit fd713b068."""
    if allow_empty is not None:
        allow_undetermined = bool(allow_empty)
    if not isinstance(block, dict):
        return False
    judged = block.get('pairs_judged')
    if allow_undetermined and block.get('case') == 'undetermined' and isinstance(judged, int) \
            and not isinstance(judged, bool) and block.get('interval') is None:
        bodies = block.get('bodies')
        return ((judged == 0 and block.get('per_100') is None)
                or (judged > 0 and isinstance(bodies, int) and not isinstance(bodies, bool) and bodies < 2
                    and finite_number(block.get('per_100')) and 0 <= block['per_100'] <= 100))
    interval = block.get('interval')
    return (finite_number(block.get('per_100')) and 0 <= block['per_100'] <= 100
            and isinstance(interval, list) and len(interval) == 2 and all(finite_number(v) for v in interval)
            and 0 <= interval[0] <= interval[1] <= 100
            and isinstance(block.get('pairs_judged'), int) and not isinstance(block['pairs_judged'], bool)
            and block['pairs_judged'] > 0
            and finite_number(block.get('spread_ratio')) and block['spread_ratio'] >= 0
            and block.get('case') in CASES and block['case'] != 'undetermined')
