# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Layer 6, the novelty miner: recurring event sequences that no named detector explains.

Addendum section 28. The miner's UNITS are the segmenter's units (segmenter.py, detector v6): each
effector's episodes (one hand, the gaze, the trunk, one leg: that effector's salient primitives
as tokens, segmenter.token) and the cross-effector PAIR units (two effectors' fresh acts starting
within 0.5 s: their two opening tokens). A unit is an occasion of its effector: its acts are
separated from the next unit's by 0.5 s of that effector's quiet. Before v6 the unit was a global
episode that under continuous activity ran to its 10 s cap, so a run's count was the number of
10 s windows holding it. The miner:
  1. takes every ordered run of 2 or 3 consecutive distinct tokens in each unit (episode_runs,
     computed once when the unit is digested and stored as the runs' ids);
  2. drops each run a named detection in the same unit already accounts for: ONE detection whose
     capability covers the kind of every token of the run (EXPLAINS below). Before v6 the kinds
     covered by all of the unit's detections together were pooled, so a touch explained by one
     detection and a head turn explained by another dropped "touch, then head turn", which no
     detector names;
  3. keeps patterns seen, unexplained, in MIN_EPISODES separate counted units on MIN_BODIES
     or more bodies, that pass two controls:
       * ORDER (since 2026-09-25, the within-unit permutation null; order_test): the run comes in
         more units than the same units would show with each unit's own tokens in a random order.
         A busy unit holds many adjacent pairs by chance, so a pair of common tokens recurs in
         many units without any sequence behind it. Each unit's chance of showing the run once its
         tokens are shuffled is exact (order_chance: it depends only on the unit's token count and
         how many of each of the run's tokens it holds), and the run's count is judged against the
         sum of those chances (the Poisson-binomial distribution's upper tail);
       * BABBLE, when the world's babble baseline exists: a rate the tracker's two-sample test
         puts above the rate random orders produce the same run (baseline_runs counts the babble
         units that contain each run, with the same segmenter). A run babble makes as often is
         chance co-occurrence, not a discovery (addendum section 36: "random babble labeled as
         skill").
     A run must pass both (the larger p-value of the two; the order control alone without a
     baseline, and the babble half is then marked unmeasured). Thousands of runs are tested
     together, so the line is the Benjamini-Hochberg one over every run tested (tracker.fdr_line):
     at most 5% of the kept candidates are expected to be chance;
  4. ranks them by recurrence, rising consistency (share of occurrences in the later half of the
     observed steps), causal world effect (an object moved, was lifted or dropped) and
     generalization (bodies);
  5. names each "novel candidate #<id>" with a stable id (a hash of its tokens), example
     episodes for replay, and never a behaviour name: a human may name it later, and that name
     never becomes a reward.
Candidates with a world effect that recur become O12 "novel manipulation candidate" detections.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

MIN_EPISODES = 5
MIN_BODIES = 2
TOP = 20
ALPHA = 0.05
FULL_ORDER_TESTS = False   # measurement only: run the order control on every tested run (mine_table)

# what each named capability accounts for, by token kind (the part before the first ':')
_TOUCH, _CURL, _DIGIT, _WRIST, _LOOK = "touch", "curl", "digit", "wrist", "look"
EXPLAINS = {
    "H01": {_TOUCH, _CURL}, "H02": {_TOUCH, _CURL}, "H03": {_TOUCH, _CURL}, "H04": {_CURL},
    "H05": {_DIGIT}, "H06": {_DIGIT, _TOUCH}, "H07": {_WRIST}, "H08": {_WRIST}, "H09": {_TOUCH},
    "B01": {_TOUCH, "approach"}, "B02": {_TOUCH, _DIGIT}, "B03": {_TOUCH}, "B04": {_TOUCH},
    "B05": {_TOUCH}, "B06": {"hand_view"}, "G01": {"eye_move"}, "G02": {_LOOK},
    "G03": {_LOOK, "eye_move"}, "G04": {_LOOK, "object_moves"}, "G05": {_LOOK, "head_turn"},
    "R01": {"approach"}, "R02": {_TOUCH, "approach"}, "R03": {"approach", _LOOK},
    "R04": {_TOUCH}, "R05": {"lift", _TOUCH}, "R06": {_TOUCH, "object_moves"}, "R07": {_TOUCH},
    "R08": {_TOUCH, "drop", "object_moves"}, "P01": {"head_turn"}, "P02": {"posture"},
    "P03": {"posture"}, "P04": {"posture"}, "P05": {"support_shift", _TOUCH},
    "P06": {"support_shift"}, "P07": {"support_shift"}, "P08": {"posture"},
    "P09": {"support_shift"}, "P10": {"support_shift", _TOUCH},
    "O01": {_TOUCH, "object_moves"}, "O02": {_TOUCH, "object_moves"}, "O03": {"drop"},
    "O04": {_TOUCH, "object_moves"}, "O05": {"object_moves"}, "O06": {"object_moves"},
    "O07": {_TOUCH}, "O08": {_TOUCH, "object_moves"}, "O11": {_TOUCH, "object_moves"},
    "HX1": {_DIGIT}, "HX2": {_DIGIT}, "HX3": {_TOUCH},
}
WORLD_EFFECT = ("object_moves", "lift", "drop")
# tokens that never carry a pattern: eye movements are not salient since 2026-09-24
# (segmenter.salient) and are dropped here too for episodes digested before
BACKGROUND = ("eye_move:",)


def _kind(tok: str) -> str:
    return tok.split(":", 1)[0]


def _runs(tokens: List[str]):
    toks = []
    for t in tokens:
        if t.startswith(BACKGROUND):
            continue
        if not toks or toks[-1] != t:
            toks.append(t)
    for n in (2, 3):
        for i in range(len(toks) - n + 1):
            run = tuple(toks[i:i + n])
            if len(set(run)) == n:
                yield run


def explained(run, caps) -> bool:
    """One detection of the unit accounts for the whole run: its capability covers the kind of
    every token (EXPLAINS)."""
    kinds = {_kind(t) for t in run}
    return any(kinds <= EXPLAINS.get(c, set()) for c in caps or [])


def episode_runs(tokens: List[str], caps: List[str]) -> List[tuple]:
    """The distinct unexplained runs of one unit, in first-seen order."""
    seen, out = set(), []
    for run in _runs(tokens or []):
        if run in seen:
            continue
        seen.add(run)
        if explained(run, caps):
            continue
        out.append(run)
    return out


def candidate_id(run) -> str:
    return hashlib.sha256("|".join(run).encode("utf-8")).hexdigest()[:8]


# ====================================================================== the order control
def token_bag(tokens: List[str]) -> Dict[str, int]:
    """A unit's tokens as _runs reads them (background tokens left out), counted: what a shuffle of
    the unit keeps."""
    bag: Dict[str, int] = defaultdict(int)
    for t in tokens or []:
        if not t.startswith(BACKGROUND):
            bag[t] += 1
    return dict(bag)


_FACT = [1]


def _fact(n: int) -> int:
    while len(_FACT) <= n:
        _FACT.append(_FACT[-1] * len(_FACT))
    return _FACT[n]


@lru_cache(maxsize=None)
def order_chance(n: int, counts: tuple) -> float:
    """The chance that a unit of n tokens, put in a uniformly random order, shows a run of 2 or 3
    distinct tokens held counts[0], counts[1] (and counts[2]) times among them, as _runs finds runs:
    equal neighbours merge, so the run (a, b) shows where some a is directly followed by b, and
    (a, b, c) where some a is followed by one or more b and then c. Exact, by inclusion-exclusion
    over the run's occurrences (two occurrences never overlap, since the run's tokens differ):
    gluing j occurrences into blocks leaves (n - j)! / (j! (A - j)! (B - j)!) orders of the rest for
    (a, b), and for (a, b, c) sum over the blocks' total b count s of C(s - 1, j - 1) (n - s - j)! /
    (j! (A - j)! (B - s)! (C - j)!); the other tokens cancel. Integers throughout (Python's), so the
    alternating sum loses nothing."""
    n = int(n)
    if len(counts) == 2:
        a, b = (int(x) for x in counts)
        if a <= 0 or b <= 0 or a + b > n:
            return 0.0
        total = _fact(n) // (_fact(a) * _fact(b))
        avoid = 0
        for j in range(min(a, b) + 1):
            term = _fact(n - j) // (_fact(j) * _fact(a - j) * _fact(b - j))
            avoid += -term if j % 2 else term
        return (total - avoid) / total
    a, b, c = (int(x) for x in counts)
    if a <= 0 or b <= 0 or c <= 0 or a + b + c > n:
        return 0.0
    total = _fact(n) // (_fact(a) * _fact(b) * _fact(c))
    shown = 0                                  # sum over j >= 1 of (-1)^(j+1) M_j
    for j in range(1, min(a, b, c) + 1):
        m = 0
        for s in range(j, b + 1):
            m += math.comb(s - 1, j - 1) * (_fact(n - s - j) // (_fact(j) * _fact(a - j) * _fact(b - s) * _fact(c - j)))
        shown += m if j % 2 else -m
    return shown / total


def order_test(count: int, chances: List[float]) -> tuple:
    """(p-value, the count expected) of a run seen in `count` units when each relevant unit shows it
    by chance with its own probability once its tokens are shuffled (order_chance): the upper tail of
    the Poisson-binomial distribution of those chances (scipy; its floor near 1e-15 is far below
    any line)."""
    from scipy import stats
    ps = np.asarray([p for p in chances if p > 0], dtype=np.float64)
    if count <= 0:
        return 1.0, float(ps.sum())
    if not ps.size:
        return 0.0, 0.0
    ps = np.minimum(ps, 1.0)
    return float(max(0.0, stats.poisson_binom(ps).sf(int(count) - 1))), float(ps.sum())


def _poisson_above(count: int, expected: float) -> float:
    from scipy import stats
    if count <= 0:
        return 1.0
    return float(stats.poisson.sf(count - 1, max(expected, 1e-12)))


def _order_control(eps, idx: List[int]):
    """For the order control: token string id -> {unit: how many times it holds the token}, over the
    counted units, and a unit -> capability names cache."""
    holders: Dict[int, Dict[int, int]] = defaultdict(dict)
    for i in idx:
        r = eps.rows[i]
        for t, k in zip(r.get("bag", ()), r.get("bag_n", ())):
            holders[int(t)][i] = int(k)
    caps: Dict[int, List[str]] = {}

    def caps_of(i):
        if i not in caps:
            caps[i] = [eps.s.get(int(c)) for c in eps.rows[i].get("caps", ())]
        return caps[i]
    return holders, caps_of


def _order_of_run(eps, toks, members, holders, caps_of) -> Optional[tuple]:
    """(p-value, expected count, units that could show it) of one run under the order control, or
    None when a counted unit carries no token counts (a store saved before v4 and not rebuilt)."""
    ids = [eps.s.index.get(t) for t in toks]
    if any(i is None or i not in holders for i in ids):
        return None
    if any(eps.rows[i].get("n_tok", 0) <= 0 for i in members):
        return None
    lists = sorted((holders[i] for i in ids), key=len)
    units = [u for u in lists[0] if all(u in h for h in lists[1:])]
    chances = []
    for u in units:
        if explained(toks, caps_of(u)):
            continue                         # a detection in it names the run: it never counts there
        chances.append(order_chance(int(eps.rows[u]["n_tok"]), tuple(holders[i][u] for i in ids)))
    p, expected = order_test(len(members), chances)
    return p, expected, len(chances)


def mine_table(eps, use: np.ndarray, run_tokens: Dict[str, List[str]], baseline: Optional[dict] = None,
               exposure_steps: Optional[float] = None, tests: Optional[dict] = None) -> List[dict]:
    """Candidates from an EpisodeTable (store.py). `use` marks the episodes that count (live,
    no caregiver touch); `run_tokens` maps a run id to its tokens; `baseline` is the babble
    baseline's {'runs': {run id: episodes containing it}, 'steps': body-steps} and
    `exposure_steps` the lineage's counted body-steps. `tests`, when given, receives how many runs
    were tested, the line, and how many passed each control."""
    rows = eps.rows
    idx = [i for i in range(len(rows)) if use[i]]
    if not idx:
        return []
    steps = np.asarray([rows[i]["step0"] for i in idx], dtype=np.float64)
    mid = (steps.min() + steps.max()) / 2.0
    pairs_run, pairs_ep = [], []
    for i in idx:
        r = rows[i]["runs"]
        if r.size:
            pairs_run.append(r)
            pairs_ep.append(np.full(r.size, i, dtype=np.int64))
    if not pairs_run:
        return []
    pr = np.concatenate(pairs_run)
    pe = np.concatenate(pairs_ep)
    order = np.argsort(pr, kind="stable")
    pr, pe = pr[order], pe[order]
    starts = np.r_[0, np.nonzero(np.diff(pr))[0] + 1]
    ends = np.r_[starts[1:], pr.size]
    base_runs = (baseline or {}).get("runs") or {}
    base_steps = float((baseline or {}).get("steps") or 0)
    holders, caps_of = _order_control(eps, idx)
    tested = []
    for a, b in zip(starts, ends):
        if b - a < MIN_EPISODES:
            continue
        members = pe[a:b]
        bodies = {rows[i]["body"] for i in members}
        if len(bodies) < MIN_BODIES:
            continue
        rid = eps.s.get(int(pr[a]))
        toks = run_tokens.get(rid)
        if toks is None:
            continue
        n = int(b - a)
        chance = dict(expected_by_chance=None, p_value=None, baseline=None)
        if base_steps > 0 and exposure_steps:
            nb = int(base_runs.get(rid, 0))
            # both counts are samples: the conditional two-sample test (tracker.two_sample_above)
            from .tracker import two_sample_above
            exp = nb * float(exposure_steps) / base_steps
            chance = dict(expected_by_chance=exp, baseline=nb,
                          p_value=two_sample_above(n, float(exposure_steps), nb, base_steps))
        # the order control (module doc): the same units with their own tokens shuffled. A run whose
        # babble test is already above ALPHA can pass no line (a Benjamini-Hochberg line never
        # exceeds ALPHA), so its order test would change nothing but the time a report takes (3,477
        # runs tested on H84 at 2:18 PM: 8.3 s against 1.1 s); FULL_ORDER_TESTS runs it anyway
        need = FULL_ORDER_TESTS or chance["p_value"] is None or chance["p_value"] <= ALPHA
        oc = _order_of_run(eps, toks, [int(i) for i in members], holders, caps_of) if need else None
        chance.update(order_p=oc[0] if oc else None, order_expected=oc[1] if oc else None,
                      order_units=oc[2] if oc else None)
        if not need:
            chance["order_note"] = "not tested: babble makes this run as often"
        ps = [p for p in (chance["p_value"], chance["order_p"]) if p is not None]
        chance["combined_p"] = max(ps) if ps else None      # both controls must pass
        tested.append((rid, toks, members, bodies, n, chance))
    from .tracker import fdr_line
    line = fdr_line([c[5]["combined_p"] for c in tested])
    cands = []
    b_line = fdr_line([c[5]["p_value"] for c in tested])
    o_line = fdr_line([c[5]["order_p"] for c in tested])
    stat = dict(tested=len(tested), line=line, passed_both=0,
                babble_measured=sum(1 for c in tested if c[5]["p_value"] is not None),
                order_measured=sum(1 for c in tested if c[5]["order_p"] is not None),
                babble_alone_line=b_line, order_alone_line=o_line,
                passed_babble_alone=sum(1 for c in tested if c[5]["p_value"] is not None and c[5]["p_value"] <= b_line),
                passed_order_alone=sum(1 for c in tested if c[5]["order_p"] is not None and c[5]["order_p"] <= o_line))
    if FULL_ORDER_TESTS:
        stat["runs"] = [dict(id=c[0], tokens=list(c[1]), units=c[4], bodies=len(c[3]),
                             **{k: c[5].get(k) for k in ("p_value", "expected_by_chance", "order_p", "order_expected",
                                                         "order_units", "combined_p")}) for c in tested]
    for rid, toks, members, bodies, n, chance in tested:
        if chance["combined_p"] is not None:
            chance["alpha"] = line
            chance["tests_together"] = len(tested)
            if not chance["combined_p"] <= line:
                continue                     # babble makes it as often, or its units' own tokens in any order do
            stat["passed_both"] += 1
        mem = sorted(members, key=lambda i: rows[i]["step0"])
        later = float(np.mean([rows[i]["step0"] > mid for i in mem]))
        effect = any(_kind(t) in WORLD_EFFECT for t in toks)
        score = math.log(n) + 1.0 * (later - 0.5) + (1.0 if effect else 0.0) + len(bodies) / 8.0
        cands.append(dict(id=rid, tokens=list(toks), episodes=n, bodies=len(bodies), later_share=later,
                          world_effect=effect, score=score, first_step=rows[mem[0]]["step0"],
                          last_step=rows[mem[-1]]["step0"], chance_measured=chance["p_value"] is not None,
                          examples=[rows[i]["id"] for i in mem[::-1][:3]],
                          occurrences=[rows[i]["id"] for i in mem], **chance))
    # a shorter run whose every occurrence is inside a longer kept run adds nothing
    cands.sort(key=lambda c: (-len(c["tokens"]), -c["score"]))
    kept = []
    for c in cands:
        dominated = any(len(k["tokens"]) > len(c["tokens"]) and k["episodes"] >= c["episodes"]
                        and all(t in k["tokens"] for t in c["tokens"]) for k in kept)
        if not dominated:
            kept.append(c)
    kept.sort(key=lambda c: -c["score"])
    for rank, c in enumerate(kept[:TOP], 1):
        c["label"] = "novel candidate #%s" % c["id"]
        c["rank"] = rank
    if tests is not None:
        tests.update(stat, kept=len(kept[:TOP]))
    return kept[:TOP]


def mine(episodes: List[dict], counted_ids=None, baseline: Optional[dict] = None,
         exposure_steps: Optional[float] = None, tests: Optional[dict] = None) -> List[dict]:
    """mine_table over plain segmenter episodes (with 'tokens', 'caps', 'body', 'step0', 'id',
    'live')."""
    from .store import EpisodeTable
    t = EpisodeTable()
    use = []
    for e in episodes:
        t.add(e)
        use.append(bool(e.get("live", True)) and not e.get("caregiver_touch")
                   and (counted_ids is None or e["id"] in counted_ids))
    return mine_table(t, np.asarray(use, dtype=bool), t.tokens, baseline, exposure_steps, tests)


def baseline_runs(episodes: List[dict]) -> Dict[str, int]:
    """For the babble baseline (baseline.py): how many units (effector episodes and pair units)
    contain each run. Units the report never counts in production are left out here too: the
    caregiver touched the body, or the scripted toy hand-off carried the toy in that hand."""
    out: Dict[str, int] = defaultdict(int)
    for e in episodes:
        if e.get("caregiver_touch") or e.get("toy_held"):
            continue
        for r in episode_runs(e.get("tokens") or [], e.get("caps") or []):
            out[candidate_id(r)] += 1
    return dict(out)


def o12_detections(cands: List[dict], episodes_by_id: Dict[str, dict]) -> List[dict]:
    """World-changing recurring candidates as O12 detections, one per occurrence."""
    from .detectors import AUTHOR_NAMES, provenance_class
    out = []
    for c in cands:
        if not c["world_effect"]:
            continue
        for eid in c.get("occurrences") or c["examples"]:
            e = episodes_by_id.get(eid)
            if e is None:
                continue
            author = e.get("author")
            if author is None:
                authors = e.get("authors") or {}
                top = max(authors.items(), key=lambda kv: kv[1])[0] if authors else 0
                author = AUTHOR_NAMES.get(int(top), "unknown")
            if e.get("toy_held"):
                continue                     # the scripted toy's hand-off moved the object
            out.append(dict(cap="O12", sub=c["id"], key=c["id"], body=e["body"], pid=e.get("pid", 0),
                            run=e.get("run"), step0=e["step0"], step1=e["step1"], t0=e.get("t0", 0.0),
                            duration_s=e.get("duration_s", 0.0), episode=eid,
                            measures=dict(recurrence_episodes=c["episodes"], bodies=c["bodies"]),
                            confidence=min(1.0, 0.5 + 0.05 * c["episodes"]), objects=[], parts=[],
                            author=author, provenance=provenance_class(author), live=e.get("live", True),
                            scripted=False, caregiver_touch=bool(e.get("caregiver_touch")),
                            caregiver_present=e.get("caregiver_present", False),
                            posture=None, candidate=c["label"]))
    return out
