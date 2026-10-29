# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Layer 5, the development tracker: the addendum's acceptance ladder for every capability.

  observed_once           above babble's threshold in SIZE or in FREQUENCY: the share of OCCASIONS
                          whose first measured act passes babble's 95th percentile of the same
                          (each babble occasion's first measured act) is above 5% (one-sided
                          binomial; only when babble measured QUALITY_MIN_BASE or more occasions),
                          or it comes more often per body-step than babble on BOTH counts,
                          separate occasions and single acts (the two-sample test below, the larger
                          of the two p-values: occasions alone reward pauses); size and frequency
                          are one row's two chances, so their smaller p-value is doubled
                          (Bonferroni) when both were tested. Dated where the running test's last
                          continuous stretch under the line begins, as repeated is (DATES below)
  repeated                3 or more separate OCCASIONS (an act starting at least OCCASION_GAP_STEPS
                          after the previous one on its body and part - a hand, a finger, an
                          object: the row's sub-row; a burst of acts is one occasion), and
                          more occasions AND more single acts per body-step than babble made in its
                          own body-steps
                          (two-sample conditional binomial: given both counts, would this many
                          fall on the lineage's side by chance?). The single-act count is judged
                          with the burst's spread (DISPERSION below), not as independent acts
  predicted_before_action 3 or more judged OCCASIONS whose verdict held against its control
                          (evidence.py). An occasion's verdict is its FIRST JUDGED ACT's: the later
                          acts of a burst come within a second of the first, so their predictions
                          are formed while the burst is already under way (a learner that knows
                          the hand is moving foresees its next touch for free), and one act's
                          verdict has one act's chance rate, so the control applies unchanged.
                          Movements and effects on objects: a share above one half (binomial). An
                          act of the hands alone (rules v6) gives one unit per command in force
                          during it, its commands one cluster (cluster_binom_above)
                          Touches: above the verdict's measured rate at matched no-touch moments of
                          the same bodies and skin patches (evidence.contact_nulls; one-sided
                          two-sample test, the no-touch rate being itself a sample); without those
                          moments the touch test is UNMEASURED. Act counts are descriptive
  selected_intentionally  3 or more OCCASIONS whose first act the planner chose decisively
                          (evidence.py: the order the act is credited to, the one in force at its
                          peak motion), and then occasions come more often per step while the
                          planner writes the orders than while babble does (conditional binomial;
                          an occasion belongs to the author of its first act's credited order);
                          with fewer decisive occasions it is not tested at all
  generalized             repeated, with acts on 3 or more bodies (each world has its own start and
                          object layout) from 2 or more starting situations (posture at the start,
                          objects involved)
  retained                after it first repeated, every later window of RET_WINDOW steps with
                          enough exposure keeps at least half of that window's OCCASION rate (a
                          window a test puts below half fails it); 2 or more later windows are
                          needed. The newest window a test puts below half marks REGRESSED.
                          Each window is tested against the first-repeat window's count as a
                          sample, not a known rate (rate_below_half), and each window's tests are
                          judged together across every row of the report with the
                          Benjamini-Hochberg line (tracker_rules_v5: RETENTION below)

THE LADDER IS IN ORDER: a stage whose earlier stage is not reached keeps its numbers but is not
reached (`waits_for`); observed once gates repeated, repeated gates the rest.

THE OCCASION IS THE INDEPENDENT UNIT (tracker_rules_v3, outside review of 2026-09-24): acts inside
one burst are correlated, so a test that counted them as independent trials was overconfident.
Occasions are the tested unit of predicted before action, selected intentionally and retained.
The single-act half of "more often than babble" stays a test (occasions alone reward pauses) with
the burst's spread: DISPERSION. A burst process's act count varies by the mean of the SQUARED acts
per occasion over the mean acts per occasion times more than independent acts would (a compound
Poisson count: variance = rate x E[n^2] against rate x E[n]), so both act counts are divided by
their dispersion before the conditional binomial (the quasi-Poisson test): the learner's from its
own occasions, babble's from its summary's squared occasion sizes when recorded (acts_sq), else
its lower bound, babble's mean acts per occasion. One act per occasion leaves every test as it was.
The size half of observed once counts occasions too (tracker_rules_v4): each occasion's FIRST
measured act, against the 95th percentile of babble's occasions' first measured acts
(measures_occasion in babble's summary). The first act, not the burst's largest: a longer burst
would otherwise get more draws at the line. A babble summary written before v4 carries no
occasion-level line, so the size half is not tested against it (frequency still is).

COUNTED means: a live record (not an abandoned future), not scripted (the toy's hand-off), not a
probe, baseline or scheduled-rest act - by the order at its first step OR the order it is
credited to (its peak motion's) - no caregiver touch during it, and no object the world moves
(the caregiver's toy, a critter) moving into it (detectors.OBJECT_CONTACT_CAPS).
Caregiver-involved acts are tallied separately, and every stage is also computed on
caregiver-absent acts alone, the "no-overseer" column the addendum requires.

A stage is reached, not reached, or UNMEASURED (reached = None) when its control is missing (no
babble baseline for this world yet, no learner records): an unmeasured stage is never shown as
failed or passed.

MANY TESTS AT ONCE. One report tests every capability and sub-row together (about 60 rows on
H84), so at 5% each, about three would pass by chance alone. track_all therefore judges the four
tested stages (observed once, repeated, predicted before action, selected intentionally) with the
Benjamini-Hochberg procedure across all rows of the report that have enough acts to test: of the
rows it calls reached, at most 5% are expected to be chance (the false discovery rate). Each stage
keeps its own p-value and the line it was judged against (`alpha`), and says how many tests were
made together.

RETENTION, MANY WINDOWS AT ONCE (tracker_rules_v5; OBSERVER_DESIGN.md section 9, review finding 32).
Until v5 each later window of each row was tested alone at 5% against the first-repeat window's rate
taken as a known constant. Both halves made "not kept" and "regressed" too easy:
  * the first-repeat window's count is a sample, and one picked for being high (it is where the
    running test first passed), so a later window read as "below half" whenever that count was
    lucky. Each window is now tested with the exact conditional test of the two counts
    (rate_below_half: given both counts, would this few fall on the later window's side if its rate
    were half the reference's?);
  * every row's every window was a separate chance at a false mark, and the chances add up as the
    windows accrue. Each WINDOW's tests are one family across every row, sub-row and no-overseer
    column of the report that tests it, judged with the Benjamini-Hochberg line (judge_retention):
    of the windows it marks below half, at most 5% are expected to be chance; where no row really
    fell below half in a window, the chance that any row is marked below in it is at most 5 in 100.
    Every window of a row's history carries its p-value, its line (`alpha`) and how many rows tested
    that window together (`tests_together`).
Measured by simulation (the measurement script's simulate-retention: 20 rows at the lineage's rates of
3 to 880 occasions a window, a reference and 8 later windows, 2,000 simulated lineages): with every
rate unchanged, the old rule marked 0.12 of the 20 rows "not kept" in each lineage (rows of 3 to 8
occasions a window: 1.4-3.1 in 100) and 0.021 "regressed"; this rule marked none. The cost, for rows
that really fell to a quarter of their rate (1,000 lineages): rows of 8 occasions a window are marked
"not kept" in 27-30 of 100 lineages against the old rule's 63-68, rows of 25 in 81-83 against 94,
rows of 115 or more in all of them under both.
evaluate() on one row alone judges each window at the plain 5%.

DATES (tracker_rules_v5, independent review of 2026-09-25, finding 2). Seen and Repeats are dated from
their RUNNING test: the same test over the occasions up to each one, with the body-steps observed up to
it. Until the review the date was the FIRST occasion at which the running test was under the line; the
v5 line (q k / m) is higher than the old one, so a brief early dip then qualified: "arm into own view"
(left) was dated 20,698 steps earlier, at a moment after which its running size test sat under the line
for only 31 of the next 100 occasions. A stage is now dated where the running test's LAST continuous
stretch under the line begins (last_stretch_start): after that occasion it never rose above the line
again, on the evidence so far. A dip that ends above the line dates nothing. Each stage keeps where its
running test first went under the line and the share of the later occasions it was under the line at
(running_record).
KEPT STILL COUNTS FROM THE FIRST PASS: the block where the Repeats test first went under the line (no
earlier than Seen's first pass), the anchor before the review. Anchored at the last stretch, a behaviour
that repeated, faded and came back would lose its faded blocks from that test: on the lineage's digests
of 7:09 PM PDT, a brief fixation and a return look at the left upper arm (G02, G03) made 23 and 11
occasions in their first-repeat block and 0 to 2 in each of the next four, then came back, and would read
"not yet" instead of "not kept". The cost of an early reference block picked for being high is bounded by
the independent review's simulation: at most 1.3 of 100 rows ever marked "not kept" over 200 blocks with
their rates unchanged.

The detections arrive as columns (store.DetTable), so a lineage of weeks is judged in one pass;
evaluate() also takes a plain list of detection dicts.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from scipy import special, stats

from . import store as ST

STAGES = ["observed_once", "repeated", "predicted_before_action", "selected_intentionally",
          "generalized", "retained"]
# The counting rules' version: the stage history starts afresh when it changes (a new rule is not a
# loss). v1 occasions per body; v2 (2026-09-24) occasions per body and part, and "more often than
# babble" needs both occasions and single acts above babble's.
# v3 (2026-09-24, outside review): the occasion is the tested unit of predicted before action (its
# first judged act; touches against the verdict's matched no-touch rate), selected intentionally
# (the credited order's author and decisiveness) and retained; the single-act count is judged with
# the burst's dispersion; an act credited to a probe, baseline or rest order is not counted.
# v4 (2026-09-24): the size half of observed once counts occasions (their first measured act)
# against babble's occasion-level 95th percentile.
# v5 (2026-09-25, observer statistics track, agents/observer_statistics_2026-09-25.md): a family's
# Benjamini-Hochberg line is q k / m, not its largest discovered p-value (fdr_line: the same passes,
# earlier dates and a visible margin); retention's windows are tested against the first-repeat
# window's count as a sample (rate_below_half) and judged per window across rows with the
# Benjamini-Hochberg line (module doc, RETENTION); Seen and Repeats are dated where their running test's
# last continuous stretch under the line begins (module doc, DATES).
# v6 (2026-10-08, observer_attribution_v1; the operator's instruction of 7:37 AM PDT, Oct 8, 2026): an act of the hands
# alone is credited to no single order. Its writer counts for the planner or for the reflex controller in "selected
# intentionally" only when every order in force during it shares that writer class (store ATTR_ONE_CLASS); an act of
# several writers (ATTR_MIXED) counts for neither; one written before observer_attribution_v1 (ATTR_HAND_UNMEASURED,
# credited then to one order) counts for neither and for no decisive choice, and its movement verdict, read on that one
# order's efference record, is NOT JUDGED: the stage says so with the count, keeps that earlier reading beside it
# (`earlier_rule`), and reads "not judged", never "not measurable" with another cause (independent review of 4342f9194,
# blocker 1). In "predicted before action" each order in force during an act of the hands alone is one unit, the act's
# orders one cluster (cluster_binom_above; review blocker 2: the act-level "every order passes" passed 85 in 100 acts of
# 1 order and 0 of 31 of 13 or more on control run 1, against a chance line of one half). The decisive test and its tie
# band are unchanged. Each row also counts its acts' separate stage readings (planner selected, achieved:
# evidence.Joiner.four_stages) and H05's other digits inside the act (`alone_inside`): reported, never a pass.
RULES_VERSION = "tracker_rules_v6"
RET_WINDOW = 3000              # steps; the ability-recheck cadence on H84 (every 3000 updates)
MIN_COUNT = 3
ALPHA = 0.05
OCCASION_GAP_STEPS = 20        # 1 s at 50 ms a step: a new act this long after the last is a new occasion
QUALITY_MIN_BASE = 20          # babble occasions measured before their 95th percentile is a usable line
EXCLUDED = {"probe-forced", "baseline", "external-rest"}


def poisson_above(count: int, expected: float) -> float:
    """P(X >= count) for X ~ Poisson(expected)."""
    if count <= 0:
        return 1.0
    return float(stats.poisson.sf(count - 1, max(expected, 1e-12)))


def poisson_below(count: int, expected: float) -> float:
    return float(stats.poisson.cdf(count, max(expected, 1e-12)))


def binom_above(k: int, n: int, p: float) -> float:
    if n <= 0:
        return 1.0
    return float(stats.binom.sf(k - 1, n, p))


def cluster_binom_above(k, m, p: float = 0.5) -> float:
    """THE PREDICTED STAGE'S TEST OVER UNITS IN CLUSTERS (tracker rules v6; independent review of 4342f9194, blocker 2).
    Each judged occasion gives m_i units, of which k_i passed: the commands in force during its first judged act of the
    hands alone (each command's own expectation against both controls), or 1 unit for any other act (its one verdict).
    Under no prediction skill a unit passes with chance at most p = one half (its expectation and the other body's
    command's are then interchangeable, and the persistence control can only lower the chance). The units of one
    cluster may depend on each other, so the counts are divided by the design effect at its upper bound,
    D = sum(m_i^2) / sum(m_i) (the units of a cluster all alike: the passed count's variance is then sum(m_i^2) p (1 - p)
    against sum(m_i) p (1 - p) for independent units), and the one-sided binomial tail is read at the effective counts,
    k / D passed of n / D, through the regularized incomplete beta function (for whole counts the binomial tail itself,
    P(X >= k) = I_p(k, n - k + 1)). With every m_i = 1, D = 1 and this is binom_above. 1 when nothing passed. The
    false-pass rate under no skill is measured by simulation (tests/test_observer_attribution.py and
    D:/UEMA_Research/external/D/UEMA_scratch/finger_investigation/observer/predicted_null_simulation.json)."""
    k = np.asarray(k, dtype=np.float64).ravel()
    m = np.asarray(m, dtype=np.float64).ravel()
    total = float(m.sum())
    if total <= 0:
        return 1.0
    design = float((m ** 2).sum() / total)
    n_eff, k_eff = total / design, float(k.sum()) / design
    if k_eff <= 0:
        return 1.0
    return float(special.betainc(k_eff, n_eff - k_eff + 1.0, p))


def _stage(reached, step=None, **kw) -> dict:
    return dict(reached=reached, first_step=step, **kw)


def _count(n: int, word: str) -> str:
    """'1 act', '3 acts'."""
    return "%d %s" % (int(n), word if int(n) == 1 else word + "s")


def counted_mask(t: ST.DetTable, live: np.ndarray) -> np.ndarray:
    """Which rows count (module doc): live, not scripted, no caregiver touch, not excluded by the
    order at the first step or by the order the act is credited to."""
    flags = t["flags"]
    ex_ids = [t.s.index[k] for k in EXCLUDED if k in t.s.index]
    excluded = np.isin(t["prov"], ex_ids) | np.isin(t["cprov"], ex_ids)
    return (np.asarray(live, dtype=bool) & ((flags & ST.F_SCRIPTED) == 0)
            & ((flags & ST.F_CARE_TOUCH) == 0) & ((flags & ST.F_WORLD_MOVED) == 0) & ~excluded)


def occasion_starts(steps: np.ndarray, bodies: np.ndarray, parts: Optional[np.ndarray] = None) -> np.ndarray:
    """The start step of every separate occasion, all bodies together, in time order: per body and
    part (the act's sub-row code: a hand, a finger, an object; one part when None), an act starting
    at least OCCASION_GAP_STEPS after the previous act's start on the same body and part opens a
    new one.

    PER PART, NOT PER BODY (adversarial review 2026-09-24, finding 1). Merged per body, two hands
    moving at once were one occasion, and babble moves both hands almost without a pause: finger
    spreading's babble made 81 merged occasions against 305 and 306 per hand, while the learner's
    pauses kept its merged count high (224 merged, 376 and 389 per hand, page of 09:07 PDT). The
    merge made the row read 2.5 times babble's rate (224 against 90 expected) where each hand read
    1.11 and 1.14 times (376 against 339, 389 against 340)."""
    steps, bodies = np.asarray(steps), np.asarray(bodies, dtype=np.int64)
    key = bodies if parts is None else bodies * 1_000_003 + np.asarray(parts, dtype=np.int64)
    out = []
    for k in np.unique(key):
        s = np.sort(steps[key == k])
        if s.size:
            out.append(s[np.r_[True, np.diff(s) >= OCCASION_GAP_STEPS]])
    return np.sort(np.concatenate(out)) if out else np.zeros(0, dtype=np.int64)


def occasions(steps: np.ndarray, bodies: np.ndarray, parts: Optional[np.ndarray] = None) -> int:
    return int(occasion_starts(steps, bodies, parts).size)


def occasion_ids(steps: np.ndarray, bodies: np.ndarray, parts: Optional[np.ndarray] = None) -> np.ndarray:
    """Each act's occasion (an index, the same grouping occasion_starts counts): per body and part,
    in start order, an act starting at least OCCASION_GAP_STEPS after the previous act's start
    opens a new occasion. Numbered in the order the occasions start (ties: body, then part)."""
    steps = np.asarray(steps, dtype=np.int64)
    if not steps.size:
        return np.zeros(0, dtype=np.int64)
    bodies = np.asarray(bodies, dtype=np.int64)
    key = bodies if parts is None else bodies * 1_000_003 + np.asarray(parts, dtype=np.int64)
    order = np.lexsort((steps, key))
    ks, ss = key[order], steps[order]
    new = np.r_[True, (ks[1:] != ks[:-1]) | (np.diff(ss) >= OCCASION_GAP_STEPS)]
    raw = np.cumsum(new) - 1                          # occasion index in (key, step) order
    first_step = ss[new]
    first_key = ks[new]
    rank = np.empty(raw.max() + 1, dtype=np.int64)
    rank[np.lexsort((first_key, first_step))] = np.arange(first_step.size)
    out = np.empty(steps.size, dtype=np.int64)
    out[order] = rank[raw]
    return out


def first_in_occasion(steps: np.ndarray, occ: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """The index of each occasion's first act (in start order) among those where `mask` holds;
    occasions without such an act are left out. Returned in the occasions' order."""
    steps = np.asarray(steps, dtype=np.int64)
    idx = np.arange(steps.size) if mask is None else np.nonzero(np.asarray(mask, dtype=bool))[0]
    if not idx.size:
        return np.zeros(0, dtype=np.int64)
    o = np.asarray(occ, dtype=np.int64)[idx]
    order = np.lexsort((idx, steps[idx], o))          # by occasion, then start, then position
    o_sorted = o[order]
    keep = np.r_[True, o_sorted[1:] != o_sorted[:-1]]
    return idx[order][keep]


def dispersion(occ: np.ndarray) -> float:
    """A burst process's spread (module doc, DISPERSION): the mean squared acts per occasion over
    the mean acts per occasion, E[n^2] / E[n]; 1 when every occasion is one act."""
    occ = np.asarray(occ, dtype=np.int64)
    if not occ.size:
        return 1.0
    n = np.bincount(occ).astype(np.float64)
    n = n[n > 0]
    return max(1.0, float(np.sum(n ** 2) / np.sum(n)))


def babble_dispersion(base: Optional[dict]) -> float:
    """Babble's E[n^2] / E[n] from its summary's squared occasion sizes (acts_sq) when recorded,
    else its lower bound, babble's mean acts per occasion (E[n^2] >= E[n]^2)."""
    b = base or {}
    n = float(b.get("count") or 0)
    if n <= 0:
        return 1.0
    sq = b.get("acts_sq")
    if sq is not None and float(sq) > 0:
        return max(1.0, float(sq) / n)
    occ = b.get("occasions")
    return max(1.0, n / float(occ)) if occ else 1.0


def two_sample_above_dispersed(k1: float, e1: float, phi1: float, k0: float, e0: float, phi0: float) -> float:
    """two_sample_above for counts whose events come in bursts: each count and its exposure are
    divided by its dispersion (the quasi-Poisson approximation), and the conditional binomial is
    read continuously (the regularized incomplete beta: P(X >= k) = I_p(k, n - k + 1)). With both
    dispersions 1 it is two_sample_above exactly."""
    if k1 <= 0 or e1 <= 0 or e0 <= 0:
        return 1.0
    a, b = float(k1) / phi1, max(0.0, float(k0)) / phi0
    ea, eb = float(e1) / phi1, float(e0) / phi0
    return float(special.betainc(a, b + 1.0, ea / (ea + eb)))


def two_sample_above_dispersed_many(k1s, e1s, phi1: float, k0: float, e0: float, phi0: float) -> np.ndarray:
    k1s = np.asarray(k1s, dtype=np.float64)
    e1s = np.maximum(np.asarray(e1s, dtype=np.float64), 1e-12)
    a = k1s / phi1
    ea, eb = e1s / phi1, float(e0) / phi0
    out = special.betainc(np.maximum(a, 1e-12), max(0.0, float(k0)) / phi0 + 1.0, ea / (ea + eb))
    return np.where(k1s > 0, out, 1.0)


def two_share_above(k1: int, n1: int, k0: int, n0: int) -> float:
    """P(this many or more of the k1 + k0 passes fall among the n1 tested ones) when both samples
    pass at the same rate: the one-sided exact conditional test of two shares (Fisher's). The
    touch verdict's chance rate at no-touch moments is itself a sample, so it carries its own
    uncertainty here, which a binomial against it as a known constant would not."""
    if n1 <= 0 or k1 <= 0:
        return 1.0
    return float(stats.hypergeom.sf(int(k1) - 1, int(n1) + int(n0), int(k1) + int(k0), int(n1)))


def two_sample_above(k1: int, e1: float, k0: int, e0: float) -> float:
    """P(this many or more of the k1 + k0 events fall on side 1) when both sides have the same
    rate per step: the exact conditional test of two Poisson counts with exposures e1 and e0. It
    carries babble's own count uncertainty, which a test against babble's rate as a known constant
    does not."""
    n = int(k1) + int(k0)
    if k1 <= 0 or n <= 0 or e1 <= 0 or e0 <= 0:
        return 1.0
    return float(stats.binom.sf(int(k1) - 1, n, e1 / (e1 + e0)))


def rate_below_half(k: int, e: float, k0: int, e0: float) -> float:
    """One-sided p-value that a later window's rate (k occasions in e body-steps) is BELOW HALF the
    reference window's (k0 in e0), with both counts samples: at the boundary of "at least half"
    (the later rate exactly half the reference's), given the k + k0 occasions the later window's
    share is binomial with probability 0.5 e / (0.5 e + e0), and the p-value is the chance of k or
    fewer. The exact conditional test of two Poisson counts (as two_sample_above), turned toward a
    fall to half. 1 when there is nothing to compare."""
    n = int(k) + int(k0)
    if n <= 0 or e <= 0 or e0 <= 0:
        return 1.0
    return float(stats.binom.cdf(int(k), n, 0.5 * e / (0.5 * e + e0)))


# ====================================================================== what passing takes
# THE COUNT AT WHICH A STAGE WOULD PASS (the Development page, 2026-09-30, operator approved). Each
# function below inverts one stage's own test at a given pass line: the smallest count at which that
# test's p-value reaches the line, every other input held (babble's counts and body-steps, the
# learner's body-steps, its acts per occasion and both burst spreads, the number judged). Nothing here
# changes a verdict: the tracker's tests above decide; these only say how far a count is from passing.
def pass_line_for(p_others, q: float = ALPHA) -> float:
    """The Benjamini-Hochberg line one test must reach to be a discovery, every other test of its family
    held at its p-value: the largest q j / m (m = the others + this one) at which at least j - 1 of the
    others are at or under q j / m. A p-value at or under it makes the family's line (fdr_line) at least
    this high, with this test among the discoveries; one above it does not."""
    ps = np.sort(np.asarray([p for p in p_others if p is not None and np.isfinite(p)], dtype=np.float64))
    m = ps.size + 1
    lines = q * np.arange(1, m + 1) / m
    ok = np.searchsorted(ps, lines, side="right") >= np.arange(m)
    return float(lines[np.nonzero(ok)[0][-1]])


def smallest_passing(test, lo: int, hi: int) -> Optional[int]:
    """The smallest integer n in [lo, hi] with test(n) true, for a test that stays true once true; None
    when even hi fails."""
    lo = max(0, int(lo))
    if lo > hi or not test(hi):
        return None
    if test(lo):
        return lo
    while hi - lo > 1:                     # test(lo) false, test(hi) true
        mid = (lo + hi) // 2
        if test(mid):
            hi = mid
        else:
            lo = mid
    return hi


def needed_occasions(line: float, ratio: float, b_occ: float, acts_per_occasion: float, b_acts: float,
                     phi: float = 1.0, phi_b: float = 1.0, chances: int = 1, least: int = 1,
                     most: int = 10 ** 9) -> Optional[int]:
    """Seen's frequency half and Repeats: the fewest occasions at which "more often than babble on both
    counts" (the larger of the occasions test and the dispersed single-act test, times `chances` for
    Seen's doubled p-value) reaches `line`, the learner's body-steps held (ratio = its body-steps over
    babble's), babble's counts held, and the single acts scaled with the occasions at the same acts per
    occasion. `least`: the stage's own minimum (Repeats needs 3)."""
    if ratio <= 0:
        return None
    a = max(1.0, float(acts_per_occasion or 1.0))

    def ok(n):
        p = max(two_sample_above(int(n), ratio, int(round(b_occ)), 1.0),
                two_sample_above_dispersed(n * a, ratio, phi, b_acts, 1.0, phi_b))
        return min(1.0, chances * p) <= line
    return smallest_passing(ok, max(1, least), most)


def needed_beyond(line: float, n: int, chances: int = 1) -> Optional[int]:
    """Seen's size half: the fewest of the n measured occasions that must start beyond babble's 95th
    percentile for the binomial test against 5 in 100 (times `chances`) to reach `line`; None when even
    all n would not."""
    return smallest_passing(lambda k: k > 0 and min(1.0, chances * binom_above(k, n, 0.05)) <= line, 1, int(n))


def needed_predicted(line: float, n: int, null_k: Optional[int] = None, null_n: Optional[int] = None,
                     design: float = 1.0) -> Optional[int]:
    """Predicted: the fewest of n judged units whose prediction must hold for the test to reach `line` (MIN_COUNT at
    least): against one half for movements and object effects, the units counted at the design effect `design` (the
    commands of acts of the hands alone in clusters: cluster_binom_above; 1 for one unit per occasion), against the
    touch verdict's own rate at matched no-touch moments (null_k of null_n, Fisher's test) for touches."""
    if null_n:
        def ok(k):
            return two_share_above(k, n, int(null_k or 0), int(null_n)) <= line
    elif design and float(design) != 1.0:
        def ok(k):
            ke, ne = k / float(design), n / float(design)
            return ke > 0 and float(special.betainc(ke, ne - ke + 1.0, 0.5)) <= line
    else:
        def ok(k):
            return binom_above(k, n, 0.5) <= line
    return smallest_passing(lambda k: k >= MIN_COUNT and ok(k), MIN_COUNT, int(n))


def needed_planner_occasions(line: float, reflex_occasions: int, planner_steps: float, reflex_steps: float,
                             most: int = 10 ** 9) -> Optional[int]:
    """Chosen, once tested: the fewest occasions during the planner's steps for "more often while the planner
    writes the orders than while the reflex controller does" to reach `line`, the reflex controller's
    occasions and both step counts held."""
    if planner_steps <= 0 or reflex_steps <= 0:
        return None
    share = planner_steps / (planner_steps + reflex_steps)
    return smallest_passing(lambda k: k > 0 and binom_above(k, k + int(reflex_occasions), share) <= line, 1, most)


def wellness_summary(t: ST.DetTable, c: np.ndarray, step0: np.ndarray, oid: np.ndarray) -> dict:
    """THE WELLNESS CHANGE UNDER A BEHAVIOUR AGAINST ITS REFERENCES (OBSERVER_DESIGN.md section 9, entry 41):
    the summed grounded wellness change over each occasion's first act that has one (the occasion is
    the unit, as for every stage), against the same change over the stretches of the act's length that
    end 1 to 3 s before it starts (evidence.wellness_references): every such stretch (all_*, whatever
    happened in it) and those without the act (reference_*). Internal evidence shown beside the
    behaviour, never a stage.

    EACH OCCASION IS ONE PAIR (independent review of 2026-09-25, finding 4). The difference is the mean
    over occasions of the act's change minus the mean change of that act's own stretches, with its 95%
    interval from the spread of those differences (Student's t): an act's stretches start one second
    apart and overlap when it lasts longer than a second, so they are averaged within the occasion, not
    counted as separate samples (until then a Welch interval over every stretch, too narrow for long
    acts). reference_n and all_n count the occasions that have stretches of that set; *_stretches the
    stretches; in_act_share is the share of the stretches (whatever happened) in which the row's
    behaviour went on."""
    w = t["wellness"][c].astype(np.float64) if c.size else np.zeros(0)
    fin = np.isfinite(w)
    fo = first_in_occasion(step0, oid, fin) if c.size else np.zeros(0, dtype=np.int64)
    a = w[fo]
    n_a = int(a.size)
    out = dict(unit="occasion (its first act with a wellness change)",
               scale="the sum of the learner's grounded wellness channels, each 0 to 1",
               reach_s="stretches of the act's length ending 1 to 3 s before it starts, one a second",
               acts_n=n_a, acts_mean=float(a.mean()) if n_a else None,
               acts_sd=float(a.std(ddof=1)) if n_a > 1 else None)
    rows = c[fo] if c.size else np.zeros(0, dtype=np.int64)
    counts = {}
    for name, (kn, ks) in (("reference", ("wref_n", "wref_sum")), ("all", ("wall_n", "wall_sum"))):
        rn = t[kn][rows].astype(np.float64) if rows.size else np.zeros(0)
        rs = t[ks][rows].astype(np.float64) if rows.size else np.zeros(0)
        has = rn > 0
        counts[name] = float(rn.sum())
        ref = rs[has] / rn[has]
        d = a[has] - ref
        k = int(has.sum())
        diff = ci = None
        if k > 1:
            diff = float(d.mean())
            ci = float(stats.t.ppf(0.975, k - 1) * d.std(ddof=1) / np.sqrt(k))
        tag = "" if name == "reference" else "_all"
        out.update({name + "_n": k, name + "_stretches": int(rn.sum()),
                    name + "_mean": float(ref.mean()) if k else None,
                    name + "_acts_mean": float(a[has].mean()) if k else None,
                    "difference" + tag: diff, "difference" + tag + "_ci95": ci})
    out["in_act_share"] = (1.0 - counts["reference"] / counts["all"]) if counts["all"] else None
    return out


def _judge_retained(st: dict, lines: Optional[dict]) -> bool:
    """Judge a retained stage's windows (module doc, RETENTION): a window is below half when its
    rate is under half the reference's and its p-value is at or under its window's line (lines:
    window -> Benjamini-Hochberg line) or, without lines, under the plain 5%. Sets each window's
    `below` and `alpha`, the stage's verdict and its lowest window's p-value and line; returns
    whether the newest window is below (REGRESSED)."""
    hist = st.get("history")
    if hist is None:
        return False
    for h in hist:
        line = None if lines is None else lines.get(int(h["window"]), lines.get(str(h["window"])))
        if lines is not None and line is None:
            line = 0.0                                      # a window no family judged: nothing below
        h["alpha"] = _alpha(line)
        h["below"] = bool(h.get("rate_low") and h.get("p_value") is not None and _below(h["p_value"], line))
    later = int(st.get("windows_after") or 0)
    st["reached"] = bool(later >= 2 and not any(h["below"] for h in hist))
    tested = [h for h in hist if h.get("p_value") is not None]
    if tested:
        worst = min(tested, key=lambda h: h["p_value"])
        st.update(worst_window=int(worst["window"]), worst_window_p=worst["p_value"],
                  worst_window_alpha=worst["alpha"])
    return bool(hist and hist[-1]["below"])


def two_sample_above_many(k1s: np.ndarray, e1s: np.ndarray, k0: int, e0: float) -> np.ndarray:
    k1s = np.asarray(k1s, dtype=np.int64)
    e1s = np.maximum(np.asarray(e1s, dtype=np.float64), 1e-12)
    return stats.binom.sf(k1s - 1, k1s + int(k0), e1s / (e1s + float(e0)))


def _cum_exposure(steps: np.ndarray, E: float, windows: dict, n_total: int) -> np.ndarray:
    """Counted body-steps observed up to each act's start, from the per-window exposure; the
    even spread E * (j + 1) / n when no window counts exist."""
    if not windows:
        return E * np.arange(1, n_total + 1) / max(1, n_total)
    ws = sorted((int(w), float(v)) for w, v in windows.items())
    keys = np.asarray([w for w, _ in ws], dtype=np.int64)
    vals = np.asarray([v for _, v in ws])
    before = np.concatenate([[0.0], np.cumsum(vals)])
    w = steps // RET_WINDOW
    pos = np.searchsorted(keys, w)                       # windows wholly before
    inside = np.where((pos < keys.size) & (keys[np.minimum(pos, keys.size - 1)] == w),
                      vals[np.minimum(pos, keys.size - 1)], 0.0)
    frac = (steps % RET_WINDOW) / float(RET_WINDOW)
    return before[pos] + inside * frac


def fdr_line(pvalues: List[float], q: float = ALPHA) -> float:
    """The Benjamini-Hochberg line for one family of tests: q k / m, where m tests were made and k
    is the largest number whose k-th smallest p-value is at or under q k / m. Every p at or under
    the line is a discovery, with at most a share q of the discoveries expected to be chance; 0 when
    there is none.

    THE LINE, NOT THE LARGEST DISCOVERED P-VALUE (tracker_rules_v5). Until v5 this returned the k-th
    smallest p-value itself. Both make the same discoveries (no p-value of the family lies between
    the two: the (k+1)-th is above q (k+1) / m), but the weakest discovery's p-value then always
    equalled its own line, so the page could not show a pass's margin (review finding 24: visually
    guided reach "passed with its p-value equal to the line"), and a stage dated by its running test
    against the line waited until that test reached the weakest discovery's FINAL p-value (all the
    way to 0 where every p-value of a family underflows)."""
    ps = np.sort(np.asarray([p for p in pvalues if p is not None and np.isfinite(p)], dtype=np.float64))
    m = ps.size
    if not m:
        return 0.0
    ok = np.nonzero(ps <= q * np.arange(1, m + 1) / m)[0]
    return float(q * (ok[-1] + 1) / m) if ok.size else 0.0


def last_stretch_start(running: np.ndarray, line: float, eligible: Optional[np.ndarray] = None) -> Optional[int]:
    """Where a running test's LAST continuous stretch at or under `line` begins (module doc, DATES): the
    index after the last one above the line or not eligible (too few occasions yet); None when the
    running test ends above the line."""
    below = np.asarray(running, dtype=np.float64) <= line
    if eligible is not None:
        below &= np.asarray(eligible, dtype=bool)
    if not below.size or not below[-1]:
        return None
    above = np.nonzero(~below)[0]
    return int(above[-1] + 1) if above.size else 0


def running_record(running: np.ndarray, line: float, eligible: Optional[np.ndarray], at: np.ndarray) -> dict:
    """What a stage's running test did, kept beside its date (module doc, DATES): the step at which it first
    went under the line (the date until the review of 2026-09-25) and the share of the later occasions,
    from there on, at which it was under the line (1 when it never rose above it again)."""
    below = np.asarray(running, dtype=np.float64) <= line
    if eligible is not None:
        below &= np.asarray(eligible, dtype=bool)
    hit = np.nonzero(below)[0]
    if not hit.size:
        return dict(running_first_below_step=None, running_below_share_after=None)
    return dict(running_first_below_step=int(at[hit[0]]), running_below_share_after=float(below[hit[0]:].mean()))


def _below(p: float, line: Optional[float]) -> bool:
    """p against a stage's line: the plain 5% (strict) when no family line is given."""
    return p < ALPHA if line is None else p <= line


def _alpha(line: Optional[float]) -> float:
    return ALPHA if line is None else line


def _evaluate_rows(t: ST.DetTable, rows: np.ndarray, counted: np.ndarray, exposure: dict,
                   baseline: Optional[dict], quality: Optional[tuple], label: str = "",
                   lines: Optional[dict] = None) -> dict:
    lines = lines or {}
    rows = np.asarray(rows, dtype=np.int64)
    cmask = counted[rows]
    c = rows[cmask]
    c = c[np.argsort(t["step0"][c], kind="stable")]
    others = rows[~cmask]
    flags = t["flags"]
    E = float(exposure.get("steps", 0))
    out = dict(label=label, counted=int(c.size), not_counted=int(others.size),
               scripted=int(np.count_nonzero(flags[others] & ST.F_SCRIPTED)),
               caregiver_involved=int(np.count_nonzero(flags[others] & ST.F_CARE_TOUCH)),
               world_moved=int(np.count_nonzero(flags[others] & ST.F_WORLD_MOVED)),
               exposure_steps=E, stages={})
    Eb = float((baseline or {}).get("steps") or 0)
    has_base = bool(baseline) and Eb > 0
    nb = int((baseline or {}).get("count", 0) or 0)
    out["rate_per_1000_steps"] = 1000.0 * c.size / E if E > 0 else None
    # internal evidence shown under the behaviour, never in its place: the learner's grounded
    # wellness change over each counted act (evidence.py)
    wv = t["wellness"][c].astype(np.float64) if c.size else np.zeros(0)
    wv = wv[np.isfinite(wv)]
    out["wellness_change_mean"] = float(wv.mean()) if wv.size else None
    out["wellness_change_n"] = int(wv.size)
    # babble's measured rate; a babble that never produced it is bounded, not measured at zero
    out["baseline_rate_per_1000_steps"] = (1000.0 * nb / Eb) if has_base and nb > 0 else None
    out["baseline_bound_per_1000_steps"] = (1000.0 * 3.0 / Eb) if has_base and nb == 0 else None
    step0 = t["step0"][c]
    bodies = t["body"][c]

    def ref(i):
        """The act a stage points at, for its replay: run, body, process and steps."""
        return dict(run=t.s.get(t["run"][i]), body=int(t["body"][i]), pid=int(t["pid"][i]),
                    step0=int(t["step0"][i]), step1=int(t["step1"][i]), episode=t.s.get(t["episode"][i]))

    def line_of(stage):
        return lines.get(stage)

    def gated(stage, before):
        """A later stage waits for the earlier one: its numbers stay, its verdict does not stand;
        once both stand, the later one is reached no earlier than the earlier one."""
        st, prev = out["stages"][stage], out["stages"][before]
        if st["reached"] and not prev["reached"]:
            st.update(reached=False, first_step=None, waits_for=before)
        elif st["reached"] and st.get("first_step") is not None and prev.get("first_step") is not None:
            st["first_step"] = max(int(st["first_step"]), int(prev["first_step"]))

    # occasions, used by every stage: each act's occasion, and the occasions' starts in time order
    oid = occasion_ids(step0, bodies, t["sub"][c]) if c.size else np.zeros(0, dtype=np.int64)
    starts = occasion_starts(step0, bodies, t["sub"][c]) if c.size else np.zeros(0, dtype=np.int64)
    out["wellness"] = wellness_summary(t, c, step0, oid)
    occ = int(starts.size)
    nb_occ = (baseline or {}).get("occasions")
    b_occ = int(nb_occ) if nb_occ is not None else nb
    n_acts = int(c.size)
    # the single-act counts' burst spread (module doc, DISPERSION): the learner's from its own
    # occasions, babble's from its summary
    phi = dispersion(oid)
    phi_b = babble_dispersion(baseline)
    out["acts_per_occasion"] = (n_acts / occ) if occ else None
    out["dispersion"] = phi
    out["babble_dispersion"] = phi_b if has_base else None

    # MORE OFTEN THAN BABBLE MEANS BOTH COUNTS (review of 2026-09-24): separate occasions and single
    # acts must each come more often than babble's; the test of both is the larger p-value (an
    # intersection-union test). Occasions alone reward pauses: babble's finger spreading runs on
    # without the 1-second pauses that separate occasions, so on H84 a learner making fewer acts
    # than babble (2,389 against 3,279) still had more occasions (1,025 against 894). The acts of
    # one burst are not independent events, so the single-act test carries the dispersion.
    def above_both(k_occ, k_acts, e):
        return max(two_sample_above(k_occ, e, b_occ, Eb),
                   two_sample_above_dispersed(k_acts, e, phi, nb, Eb, phi_b))

    def above_both_running(upto):
        acts_upto = np.searchsorted(step0, starts, side="right")      # acts begun by each occasion
        return np.maximum(two_sample_above_many(np.arange(1, occ + 1), upto, b_occ, Eb),
                          two_sample_above_dispersed_many(np.maximum(acts_upto, 1), upto, phi, nb, Eb, phi_b))

    # 1 observed once: above babble in size (the share of occasions beyond its occasions' 95th
    # percentile) or in frequency (occasions per body-step)
    q = None
    if quality and has_base:
        q = (baseline.get("measures_occasion") or {}).get(quality[0])
        if not q or int(baseline.get("measured_occasions") or 0) < QUALITY_MIN_BASE:
            q = None
    if not has_base:
        out["stages"]["observed_once"] = _stage(None if c.size else (False if E > 0 else None), count=int(c.size),
                                                note="babble's range for this world is not measured")
    else:
        p_share = k = n_fin = None
        first_beyond = None
        if q is not None:
            v = t["measure"][c].astype(np.float64)
            fin = np.isfinite(v)                    # an act without the measure is no evidence
            fm = first_in_occasion(step0, oid, fin)                 # each occasion's first measured act
            fm = fm[np.argsort(step0[fm], kind="stable")]
            beyond = (v[fm] >= q[2]) if quality[1] == "higher" else (v[fm] <= q[0])
            n_fin, k = int(fm.size), int(beyond.sum())
            p_share = binom_above(k, n_fin, 0.05) if (n_fin and k) else (1.0 if n_fin else None)
            first_beyond = int(fm[np.nonzero(beyond)[0][0]]) if k else None
        p_rate = above_both(occ, n_acts, E) if occ else None
        ps = [p for p in (p_share, p_rate) if p is not None]
        p = min(1.0, len(ps) * min(ps)) if ps else None
        ok = bool(p is not None and _below(p, line_of("observed_once")))
        by = None
        seen_step = None
        trace = {}
        if ok:
            by = "size" if (p_share is not None and (p_rate is None or p_share <= p_rate)) else "frequency"
            # dated like repeated: where the running test of the reading that passed (with the same
            # doubling) began its last continuous stretch under the line (module doc, DATES)
            a = _alpha(line_of("observed_once"))
            if by == "size":
                kc, nc = np.cumsum(beyond), np.arange(1, n_fin + 1)  # the occasions in step order
                pk = stats.binom.sf(kc - 1, nc, 0.05) * len(ps)
                hit = last_stretch_start(pk, a, kc >= 1)
                seen_step = int(step0[fm[hit]]) if hit is not None else int(step0[fm[-1]])
                trace = running_record(pk, a, kc >= 1, step0[fm])
            else:
                upto = _cum_exposure(starts, E, exposure.get("windows") or {}, occ)
                pk = above_both_running(upto) * len(ps)
                hit = last_stretch_start(pk, a)
                seen_step = int(starts[hit]) if hit is not None else int(starts[-1])
                trace = running_record(pk, a, None, starts)
        st = _stage(ok, seen_step, **trace,
                    beyond=k, measured=n_fin, share_unit="occasion (its first measured act)",
                    share_p=p_share, occasions=occ, babble_occasions=b_occ,
                    expected_by_chance=b_occ * E / Eb, acts=n_acts, expected_acts=nb * E / Eb,
                    rate_p=p_rate, p_value=p, alpha=_alpha(line_of("observed_once")), by=by,
                    exposure_steps=E, babble_steps=Eb, babble_acts=nb, dispersion=phi, babble_dispersion=phi_b,
                    measure=quality[0] if q is not None else None,
                    direction=quality[1] if q is not None else None,
                    beyond_line=(q[2] if quality[1] == "higher" else q[0]) if q is not None else None)
        ex = first_beyond if first_beyond is not None else (0 if c.size else None)
        if ex is not None:
            r = ref(int(c[ex]))
            st.update(example=r["episode"], act=r)
        out["stages"]["observed_once"] = st

    # 2 repeated: separate occasions, against babble's occasions over its own body-steps. Every row
    # also carries the opposite test, fewer occasions than babble (the page names rows below babble:
    # a floor passed is no goal, and a row far under the floor is a finding too)
    rep_step = None
    exp_b = (b_occ * E / Eb) if (has_base and E > 0) else None
    exp_acts = (nb * E / Eb) if (has_base and E > 0) else None
    below_p = (max(two_sample_above(b_occ, Eb, occ, E),
                   two_sample_above_dispersed(nb, Eb, phi_b, n_acts, E, phi))
               if (has_base and E > 0 and b_occ > 0 and nb > 0) else None)
    if c.size < MIN_COUNT or occ < MIN_COUNT:
        out["stages"]["repeated"] = _stage(False if E > 0 else None, count=int(c.size), occasions=occ,
                                           expected_by_chance=exp_b, acts=n_acts, expected_acts=exp_acts,
                                           below_babble_p=below_p)
    elif not has_base:
        out["stages"]["repeated"] = _stage(None, count=int(c.size), occasions=occ,
                                           note="recurs; babble's chance rate for this world is unmeasured")
    else:
        p = above_both(occ, n_acts, E)
        line = line_of("repeated")
        trace = {}
        if _below(p, line):
            # the occasion from which both running counts stayed above babble's for the body-steps
            # observed up to each (the last continuous stretch under the line: module doc, DATES)
            ks = np.arange(1, occ + 1)
            seen = _cum_exposure(starts, E, exposure.get("windows") or {}, occ)
            ps = above_both_running(seen)
            ok = last_stretch_start(ps, _alpha(line), ks >= MIN_COUNT)
            rep_step = int(starts[ok]) if ok is not None else int(starts[-1])
            trace = running_record(ps, _alpha(line), ks >= MIN_COUNT, starts)
        out["stages"]["repeated"] = _stage(_below(p, line), rep_step, **trace, count=int(c.size), occasions=occ,
                                           babble_occasions=b_occ, babble_steps=Eb,
                                           babble_occasions_measured=nb_occ is not None,
                                           expected_by_chance=b_occ * E / Eb, acts=n_acts, expected_acts=exp_acts,
                                           p_value=p, alpha=_alpha(line), below_babble_p=below_p,
                                           exposure_steps=E, babble_acts=nb, dispersion=phi, babble_dispersion=phi_b)
    gated("repeated", "observed_once")
    # the step the later stages build on is repeated's AFTER the ladder moved it (generalized and
    # retained used the running test's own step, which came before 'seen' on H84 rows)
    rep_step = out["stages"]["repeated"].get("first_step") if out["stages"]["repeated"]["reached"] else None
    # Kept counts from where the Repeats test FIRST passed (module doc, DATES), no earlier than Seen's first
    # pass: the anchor of the dates before the review of 2026-09-25
    first_rep = None
    if rep_step is not None:
        firsts = [(out["stages"][s].get("running_first_below_step") or out["stages"][s].get("first_step"))
                  for s in ("observed_once", "repeated")]
        firsts = [int(x) for x in firsts if x is not None]
        first_rep = max(firsts) if firsts else rep_step

    # 3 predicted before action: the OCCASION is the tested unit (module doc). An occasion's verdict
    # is its first judged act's; movements and object effects are tested against one half, touches
    # against the verdict's own rate at matched no-touch moments (evidence.contact_nulls).
    # RULES v6 (module doc, RULES_VERSION v6): an act of the hands alone recorded since observer_attribution_v1 gives one
    # unit per command in force during it (store pred_n judged, pred_k predicted), the act's commands one cluster
    # (cluster_binom_above); one recorded before had its movement verdict read on ONE credited command's efference
    # record: NOT JUDGED, counted, and its earlier reading kept beside the stage (`earlier_rule`), never the stage
    attr = t["attr"][c]
    kinds_all = np.asarray([t.s.get(int(x)) for x in t["pred_kind"][c]], dtype=object)
    eff = kinds_all == "efference"
    by_command = ((attr == ST.ATTR_ONE_CLASS) | (attr == ST.ATTR_MIXED)) & eff
    one_order = (attr == ST.ATTR_HAND_UNMEASURED) & eff
    flag_judged = (flags[c] & ST.F_PRED_JUDGED) > 0
    flag_true = (flags[c] & ST.F_PRED_TRUE) > 0
    pn = t["pred_n"][c].astype(np.int64)
    pk = t["pred_k"][c].astype(np.int64)
    judged = np.where(by_command, pn > 0, flag_judged) & ~one_order
    units_n = np.where(by_command, pn, 1)
    units_k = np.where(by_command, pk, flag_true.astype(np.int64))
    # an act counts as predicted (for the occasion counts and the date) when more than half its units passed: one unit,
    # its verdict
    true = judged & (2 * units_k > units_n)
    n_pv, n_pt = int(judged.sum()), int(true.sum())
    fj = first_in_occasion(step0, oid, judged)                      # each occasion's first judged act
    fj = fj[np.argsort(step0[fj], kind="stable")]
    kinds = kinds_all[fj]
    touch = kinds == "contact_head"
    n_oj, n_ot = int(fj.size), int(true[fj].sum())
    # the touch verdict's matched no-touch moments: every counted touch act's, each moment given
    # to one act only, so the sums never count a moment twice
    tmask = judged & (kinds_all == "contact_head")
    null_n, null_k = int(t["null_n"][c][tmask].sum()), int(t["null_k"][c][tmask].sum())
    counts = dict(predicted=n_ot, judged=n_oj, acts_predicted=n_pt, acts_judged=n_pv,
                  unit="occasion (its first judged act)")
    move = fj[~touch]
    if by_command[move].any():
        m_i, k_i = units_n[move], units_k[move]
        counts.update(commands_judged=int(m_i.sum()), commands_predicted=int(k_i.sum()),
                      design_effect=float((m_i ** 2).sum() / max(1, m_i.sum())),
                      unit="occasion (its first judged act); for an act of the hands alone each command in force during "
                           "it is a unit and the act's commands are one cluster")
    n_old = int(one_order.sum())
    n_none = int(np.count_nonzero(~judged & ~one_order))
    if n_old:
        counts["acts_read_on_one_credited_order_not_judged"] = n_old
        # the earlier reading of those acts, as the rules before v6 judged them: each occasion's first act judged then,
        # its verdict read on the one command it was credited to; reported beside the stage, never the stage
        old_j = one_order & flag_judged
        fo = first_in_occasion(step0, oid, old_j)
        k_o, n_o = int(flag_true[fo].sum()), int(fo.size)
        p_o = binom_above(k_o, n_o, 0.5) if n_o else None
        # judged at the line of the family the earlier rule would make (track_all: every cell's earlier p-value where it
        # has one, else its own), so a pass of the reports before v6 reads as one here
        earlier_line = line_of("predicted_before_action_earlier")
        counts["earlier_rule"] = dict(
            predicted=k_o, judged=n_o, p_value=p_o, alpha=_alpha(earlier_line),
            passes_at_line=bool(n_o and k_o >= MIN_COUNT and _below(p_o, earlier_line)),
            unit="occasion (its first act judged then)",
            rule="each act's movement verdict read on the one command it was credited to (analysis v5 to Oct 8, 2026), "
                 "withdrawn by the operator's attribution rule of 7:37 AM PDT, Oct 8, 2026")
    if touch.any():
        counts.update(null_moments=null_n, null_passed=null_k,
                      null_rate=(null_k / null_n) if null_n else None,
                      null="the same verdict at matched no-touch moments of the same bodies and skin patches")
    elif fj.size:
        counts.update(null_rate=0.5, null="one half: a prediction no better than its control wins at most half the time")
    if not c.size:
        out["stages"]["predicted_before_action"] = _stage(False if E > 0 else None, predicted=0, judged=0,
                                                          note="no acts to judge")
    elif not n_pv and n_old:
        # BLOCKER 1 of the independent review of 4342f9194: the true reason and the count, never the lane's
        note = ("not judged: %s of the hands alone %s recorded before Oct 8, 2026, when an act's movement verdict was "
                "read on the one command it was credited to; the current rule judges every command in force during an "
                "act, and those records do not list them" % (_count(n_old, "act"), "was" if n_old == 1 else "were"))
        if n_none:
            note += ("; %s had no verdict: the learner's lane had no record at %s, or no touch started on the touching "
                     "skin" % (_count(n_none, "other act"), "it" if n_none == 1 else "them"))
        out["stages"]["predicted_before_action"] = _stage(None, not_judged=True, note=note, **counts)
    elif not n_pv:
        out["stages"]["predicted_before_action"] = _stage(
            None, note="none of its acts could be judged: the learner's lane had no record at them, or no touch "
                       "started on the touching skin")
    elif n_oj < MIN_COUNT:
        out["stages"]["predicted_before_action"] = _stage(False, note="too few judged occasions to test", **counts)
    elif touch.any() and not null_n:
        out["stages"]["predicted_before_action"] = _stage(
            None, note="the touch verdict's chance rate at matched no-touch moments is not measured", **counts)
    else:
        parts = []
        if touch.any():
            parts.append(two_share_above(int(true[fj][touch].sum()), int(touch.sum()), null_k, null_n))
        if (~touch).any():
            parts.append(cluster_binom_above(units_k[move], units_n[move], 0.5))
        p = max(parts)                      # a row of two kinds must pass both (never in the catalog)
        line = line_of("predicted_before_action")
        ok = n_ot >= MIN_COUNT and _below(p, line)
        tpos = fj[true[fj]]
        kind = t.s.get(t["pred_kind"][c[fj[0]]])
        out["stages"]["predicted_before_action"] = _stage(ok, int(step0[tpos[MIN_COUNT - 1]]) if ok else None,
                                                          p_value=p, kind=kind, alpha=_alpha(line), **counts)
    gated("predicted_before_action", "repeated")
    er = out["stages"]["predicted_before_action"].get("earlier_rule")
    if er is not None:
        # what the stage would read under the earlier rule: its test passing, and the ladder (Repeats reached)
        er["stage_reached"] = bool(er["passes_at_line"] and out["stages"]["repeated"].get("reached") is True)

    # 4 selected intentionally: only decisive planner choices can make it, and only then is the
    # planner-versus-babble rate tested. The OCCASION is the unit: its first act, credited to the
    # order in force at that act's peak motion (evidence.py), decides whether the planner chose it
    # and whose steps it came in. An act of the hands alone has no credited order (rules v6): it counts by the
    # writer class EVERY order in force during it shares; one of several writers counts for neither side, and one
    # written before observer_attribution_v1 for neither and for no decisive choice. The tie band is unchanged
    measured_attr = attr != ST.ATTR_HAND_UNMEASURED
    intent = ((flags[c] & ST.F_INTENT_TRUE) > 0) & measured_attr
    fa = first_in_occasion(step0, oid)                              # each occasion's first act
    fa = fa[np.argsort(step0[fa], kind="stable")]
    iv = fa[intent[fa]]
    by_author = exposure.get("by_author") or {}
    Em, Es = float(by_author.get("planner-selected", 0)), float(by_author.get("spontaneous", 0))
    cprov = t["cprov"][c]
    pm, ps_ = t.s.index.get("planner-selected", -9), t.s.index.get("spontaneous", -9)
    Cm = int(np.count_nonzero((cprov[fa] == pm) & measured_attr[fa]))
    Cs = int(np.count_nonzero((cprov[fa] == ps_) & measured_attr[fa]))
    moved = t["credit_moved"][c]
    sel = dict(decisive=int(iv.size), decisive_acts=int(intent.sum()),
               credited_later=int(np.count_nonzero(moved == 1)), credited_earlier=int(np.count_nonzero(moved == 2)),
               credit_measured=int(np.count_nonzero(moved >= 0)),
               planner_occasions=Cm, babble_occasions=Cs, planner_steps=Em, reflex_steps=Es,
               writer_not_measured_occasions=int(np.count_nonzero(attr[fa] == ST.ATTR_HAND_UNMEASURED)),
               mixed_writer_occasions=int(np.count_nonzero(attr[fa] == ST.ATTR_MIXED)),
               unit="occasion (its first act's credited command; an act of the hands alone: the writer class of "
                    "every command in force during it)")
    if Em <= 0:
        out["stages"]["selected_intentionally"] = _stage(None if E <= 0 else False,
                                                         note="the planner wrote no command in the counted steps", **sel)
    elif iv.size < MIN_COUNT:
        out["stages"]["selected_intentionally"] = _stage(
            False,
            note="fewer than %d occasions the planner chose decisively (its evaluator ranked the chosen plan best "
                 "beyond its tie band); the planner-versus-babble rate is not tested until then" % MIN_COUNT, **sel)
    elif Es <= 0:
        out["stages"]["selected_intentionally"] = _stage(
            None, note="no babble-authored steps to compare the planner's with", **sel)
    else:
        p = binom_above(Cm, Cm + Cs, Em / (Em + Es)) if (Cm + Cs) > 0 else 1.0
        line = line_of("selected_intentionally")
        ok = _below(p, line)
        out["stages"]["selected_intentionally"] = _stage(
            ok, int(step0[iv[MIN_COUNT - 1]]) if ok else None,
            planner_rate_per_1000=1000.0 * Cm / Em, babble_rate_per_1000=(1000.0 * Cs / Es if Es else None),
            p_value=p, alpha=_alpha(line), **sel)
    gated("selected_intentionally", "repeated")

    # THE SEPARATE STAGE READINGS OF THE ACTS OF THE HANDS ALONE (rules v6; evidence.Joiner.four_stages), counted over
    # the counted acts: reported beside the stages, never a pass, and a planner's selection is never read as intention
    hand = (attr == ST.ATTR_ONE_CLASS) | (attr == ST.ATTR_MIXED)

    def _three(col, other="not_measured"):
        v = t[col][c][hand]
        return {"yes": int(np.count_nonzero(v == 1)), "no": int(np.count_nonzero(v == 0)),
                other: int(np.count_nonzero(v == -1))}
    out["act_stage_readings"] = dict(
        unit="acts of the hands alone recorded since observer_attribution_v1", acts=int(hand.sum()),
        planner_selected=_three("sel", "not_judged"), achieved=_three("ach"),
        note="planner_selected: no when a command in force during the act was not the planner's; not judged when "
             "every one was, since no noise band of the planner's goal distance has been measured. achieved: the act's "
             "joints moved the way every command's hand joint goal asked, over each command's whole hold. Not a stage "
             "pass; a selection is not intention")
    al = t["alone"][c]
    if np.any(al >= 0):
        from .detectors import ISOLATION_OTHERS_MAX_DEG
        out["alone_inside"] = dict(yes=int(np.count_nonzero(al == 1)), no=int(np.count_nonzero(al == 0)),
                                   not_measured=int(np.count_nonzero(al == -1)), unit="acts",
                                   line_deg=float(ISOLATION_OTHERS_MAX_DEG),
                                   note="the other four digits' largest movement inside the act, averaged, at most "
                                        "the line (yes) or beyond it (no); acts written before it was measured: "
                                        "not_measured")

    # 5 generalized: repeated, with acts on 3+ bodies (each world its own layout) from 2+ starting
    # situations (the posture at the start and the objects involved)
    ctx = t["context"][c]
    n_bodies, n_ctx = int(np.unique(bodies).size), int(np.unique(ctx).size)
    rep = out["stages"]["repeated"]["reached"]
    gen_step = None
    if rep is None:
        gen = None
    else:
        gen = bool(n_bodies >= 3 and n_ctx >= 2)
        if gen:
            _, fb = np.unique(bodies, return_index=True)
            _, fc = np.unique(ctx, return_index=True)
            at = max(int(np.sort(fb)[2]), int(np.sort(fc)[1]))
            gen_step = max(int(step0[at]), rep_step or 0)
    out["stages"]["generalized"] = _stage(gen, gen_step, bodies=n_bodies, contexts=n_ctx)
    gated("generalized", "repeated")

    # 6 retained / regressed: every later window keeps at least half the first-repeat rate (or is
    # not put below it by the test); the newest window the test puts below half marks REGRESSED.
    # Counted in OCCASIONS (each window's occasion starts): a burst's acts are not independent
    # events, and a Poisson test of acts called a window "below half" too readily. Each window is
    # tested against the reference window's COUNT (rate_below_half) and, in a report, judged with
    # its window's line across rows (judge_retention; module doc, RETENTION)
    windows = exposure.get("windows") or {}
    if not rep or rep_step is None:
        out["stages"]["retained"] = _stage(None if rep is None else False, note="not repeated yet",
                                           **({} if rep is None else dict(waits_for="repeated")))
        out["regressed"] = False
    else:
        # the block where the Repeats test FIRST passed (first_rep), not the date shown (module doc, DATES:
        # anchored at the last stretch, a behaviour that repeated, faded and came back would lose its
        # faded blocks from this test)
        w0 = int(first_rep // RET_WINDOW)
        wins, cnts = np.unique(starts // RET_WINDOW, return_counts=True)
        cnt = defaultdict(int, {int(w): int(n) for w, n in zip(wins, cnts)})
        e0 = float(windows.get(str(w0), windows.get(w0, 0)) or 0)
        ref_rate = cnt[w0] / e0 if e0 > 0 else None
        later = sorted(int(w) for w in windows if int(w) > w0
                       and float(windows[w]) >= 0.5 * RET_WINDOW * max(1, exposure.get("bodies", 1)))
        hist = []
        for w in later:
            ew = float(windows.get(str(w), windows.get(w)))
            cw = cnt[w]
            pw = rate_below_half(cw, ew, cnt[w0], e0) if ref_rate else None
            hist.append(dict(window=w, count=cw, steps=ew, p_value=pw,
                             rate_low=bool(ref_rate and cw / ew < 0.5 * ref_rate)))
        st = _stage(False, windows_after=len(later), reference_window=w0, reference_count=cnt[w0],
                    reference_steps=e0, reference_per_1000=(1000.0 * ref_rate if ref_rate else None),
                    reference_step=int(first_rep), history=hist, unit="occasions",
                    test="each later window against half the rate of the window where the Repeats test first "
                         "passed, both counts samples")
        out["regressed"] = _judge_retained(st, (lines or {}).get("retained"))
        out["stages"]["retained"] = st
    return out


def evaluate(dets, exposure: dict, baseline: Optional[dict], quality: Optional[tuple],
             label: str = "") -> dict:
    """The six stages for one capability (or one sub-row) from its detections: a list of dicts,
    or a (DetTable, rows, live) tuple.

    exposure: {'steps': counted body-steps, 'by_author': {class: steps}, 'windows': {w: steps},
               'absent_steps': caregiver-absent steps}
    baseline: {'count', 'steps', 'rate_per_step', 'measures': {name: [p05, p50, p95]}} or None
    """
    if isinstance(dets, tuple):
        t, rows, live = dets
    else:
        t = ST.DetTable.from_dicts(list(dets), quality[0] if quality else None)
        rows = np.arange(len(t))
        live = (t["flags"] & ST.F_LIVE_RECORDED) > 0
    return _evaluate_rows(t, rows, counted_mask(t, live), exposure, baseline, quality, label)


def exposure_of(exposure_all: Dict[str, dict], caregiver_absent: bool = False) -> dict:
    """Sum the daemon's per-body exposure into the tracker's form."""
    steps, by_author, windows = 0.0, defaultdict(float), defaultdict(float)
    bodies = 0
    for b, ex in exposure_all.items():
        bodies += 1
        if caregiver_absent:
            steps += ex.get("absent_steps", 0)
            for k, v in (ex.get("absent_by_author") or {}).items():
                by_author[k] += v
            for k, v in (ex.get("absent_windows") or {}).items():
                windows[str(k)] += v
        else:
            steps += ex.get("steps", 0)
            for k, v in (ex.get("by_author") or {}).items():
                by_author[k] += v
            for k, v in (ex.get("windows") or {}).items():
                windows[str(k)] += v
    return dict(steps=steps, by_author=dict(by_author), windows=dict(windows), bodies=max(1, bodies))


def track_all(dets, exposure_all: Dict[str, dict], baselines: Dict[str, dict], catalog: dict,
              live: Optional[np.ndarray] = None) -> dict:
    """Stages for every catalog capability and each sub-row, with and without the caregiver.
    `dets` is a DetTable (with `live`, one flag per row) or a list of detection dicts."""
    if isinstance(dets, ST.DetTable):
        t = dets.freeze()
        live = np.ones(len(t), dtype=bool) if live is None else np.asarray(live, dtype=bool)
    else:
        quality = {c["id"]: c["quality"][0] for c in catalog["capabilities"] if c.get("quality")}
        t = ST.DetTable.from_dicts(list(dets), quality)
        live = (t["flags"] & ST.F_LIVE_RECORDED) > 0
    counted = counted_mask(t, live)
    ex = exposure_of(exposure_all)
    ex_abs = exposure_of(exposure_all, caregiver_absent=True)
    caps = t["cap"]
    order = np.argsort(caps, kind="stable")
    bounds = np.r_[0, np.nonzero(np.diff(caps[order]))[0] + 1, order.size]
    by_cap = {t.s.get(int(caps[order[a]])): order[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a}
    present = (t["flags"] & ST.F_CARE_PRESENT) > 0
    first = _track_pass(t, catalog, by_cap, counted, present, ex, ex_abs, baselines, None)
    lines = {}
    for stage in FAMILY_STAGES:
        ps = [st.get("p_value") for st in _all_stages(first, stage)]
        lines[stage] = fdr_line(ps)
    # rules v6: the family the earlier rule would make for Predicted, every cell's earlier p-value (acts of the hands
    # alone recorded before observer_attribution_v1, single-command credit) where it has one, else its own; the earlier
    # readings are judged at its line, so a pass of the reports before v6 reads as one beside the stage
    lines["predicted_before_action_earlier"] = fdr_line([
        ((st.get("earlier_rule") or {}).get("p_value") if (st.get("earlier_rule") or {}).get("p_value") is not None
         else st.get("p_value")) for st in _all_stages(first, "predicted_before_action")])
    out = _track_pass(t, catalog, by_cap, counted, present, ex, ex_abs, baselines, lines)
    for stage in FAMILY_STAGES:
        tests = sum(1 for st in _all_stages(out, stage) if st.get("p_value") is not None)
        for st in _all_stages(out, stage):
            if st.get("p_value") is not None:
                st["tests_together"] = tests
    judge_retention(out)
    return out


def judge_retention(tracked: dict) -> Dict[int, float]:
    """Judge every retention window of a report as one family per window (module doc, RETENTION):
    each window's p-values from every row, sub-row and no-overseer column that tests it, the
    Benjamini-Hochberg line over them, and each row's verdict and REGRESSED flag again from those
    lines. Returns window -> line."""
    per_window: Dict[int, List[float]] = defaultdict(list)
    for st in _all_stages(tracked, "retained"):
        for h in st.get("history") or []:
            if h.get("p_value") is not None:
                per_window[int(h["window"])].append(float(h["p_value"]))
    lines = {w: fdr_line(ps) for w, ps in per_window.items()}
    for row in tracked.values():
        holders = [(row, (row.get("stages") or {}).get("retained"))]
        holders += [(None, (row.get("no_overseer") or {}).get("retained"))]
        holders += [(sr, (sr.get("stages") or {}).get("retained")) for sr in (row.get("subs") or {}).values()]
        for owner, st in holders:
            if not st or st.get("history") is None:
                continue
            regressed = _judge_retained(st, lines)
            for h in st["history"]:
                h["tests_together"] = len(per_window.get(int(h["window"]), []))
            if owner is not None:
                owner["regressed"] = regressed
    return lines


FAMILY_STAGES = ("observed_once", "repeated", "predicted_before_action", "selected_intentionally")


def _all_stages(tracked: dict, stage: str):
    """Every row's and sub-row's result for one stage (both overseer columns)."""
    for row in tracked.values():
        for src in (row.get("stages") or {}, row.get("no_overseer") or {}):
            st = src.get(stage)
            if st:
                yield st
        for sr in (row.get("subs") or {}).values():
            st = (sr.get("stages") or {}).get(stage)
            if st:
                yield st


def _track_pass(t, catalog, by_cap, counted, present, ex, ex_abs, baselines, lines) -> dict:
    out = {}
    for cap in catalog["capabilities"]:
        cid = cap["id"]
        rows = by_cap.get(cid, np.zeros(0, dtype=np.int64))
        base = (baselines or {}).get(cid)
        quality = tuple(cap["quality"]) if cap.get("quality") else None
        row = dict(id=cid, name=cap["name"], group=cap["group"], observable=cap.get("observable", True),
                   why_not=cap.get("why_not"))
        if not row["observable"]:
            row.update(stages={s: _stage(None) for s in STAGES}, counted=0, note=cap.get("why_not"))
            out[cid] = row
            continue
        row.update(_evaluate_rows(t, rows, counted, ex, base, quality, cap["name"], lines))
        absent = rows[~present[rows]]
        row["no_overseer"] = _evaluate_rows(t, absent, counted, ex_abs, base, quality,
                                            cap["name"] + " (caregiver absent)", lines)["stages"]
        subs = t["sub"][rows]
        if (subs >= 0).any():
            row["subs"] = {}
            for sid in sorted(set(int(x) for x in subs if x >= 0), key=lambda i: t.s.get(i)):
                sk = t.s.get(sid)
                sb = ((base or {}).get("subs") or {}).get(sk)
                if sb is None and base:
                    # babble was measured and never made this sub-row: zero, not unmeasured
                    sb = dict(count=0, occasions=0, steps=base.get("steps", 0))
                row["subs"][sk] = _evaluate_rows(t, rows[subs == sid], counted, ex, sb, quality,
                                                 "%s %s" % (cap["name"], sk), lines)
        out[cid] = row
    return out


def timeline(tracked: dict) -> List[dict]:
    """First observation, first repeat, first intentional episode, regression (section 32)."""
    ev = []
    for cid, row in tracked.items():
        for s in STAGES:
            st = (row.get("stages") or {}).get(s) or {}
            if st.get("reached") and st.get("first_step") is not None:
                ev.append(dict(step=int(st["first_step"]), cap=cid, name=row.get("name"), stage=s))
        if row.get("regressed"):
            hist = ((row.get("stages") or {}).get("retained") or {}).get("history") or []
            w = hist[-1]["window"] if hist else None
            ev.append(dict(step=int((w or 0) * RET_WINDOW), cap=cid, name=row.get("name"), stage="regressed"))
    return sorted(ev, key=lambda e: e["step"])
