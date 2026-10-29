# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The outside review's ranking statistic (mimo_bridge/tools/command_ranking.py).

Of all pairs of candidate commands tried from the same saved state, how many in 100 the forecast
orders as their actual outcomes are ordered; its body interval, the spread ratio and the reviewer's
case label. Pure arithmetic on fixtures whose answer is known by construction, plus - when the saved
results exist on this machine - a check that it reproduces the coordinator's first computation
(docs/gain_of_function_v4/measurements/candidate_ranking_2026-09-24.json) from the same files.
No model or world is loaded.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mimo_bridge.tools import command_ranking as R

REPO = Path(__file__).resolve().parents[4]
# the saved probe outputs live in the main checkout; a sparse worktree points here with UEMA_DATA_ROOT
DATA = Path(os.environ.get('UEMA_DATA_ROOT', REPO))


def starts(bodies=8, per_body=2, K=5, forecast=lambda a, rng: a, seed=0, reference_gap=None):
    """Synthetic starts: actual outcomes drawn per start; the forecast a function of them. With
    reference_gap, candidate 0 ('chosen') ends reference_gap below candidate 2 ('reflex')."""
    rng = np.random.default_rng(seed)
    out = []
    names = ['chosen', 'hold', 'reflex', 'perturb_1', 'perturb_2'][:K]
    for body in range(bodies):
        for _ in range(per_body):
            actual = rng.normal(0.0, 1.0, K)
            if reference_gap is not None:
                actual[0] = actual[2] - reference_gap
            out.append(dict(body=body, names=names, actual=actual.tolist(),
                            predicted=np.asarray(forecast(actual, rng), float).tolist()))
    return out


def test_pairs_count_the_forecasts_order_against_the_actual_order():
    got = R.pair_counts([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    assert got == dict(right=3.0, judged=3, actual_ties=0, forecast_ties=0)
    got = R.pair_counts([3.0, 2.0, 1.0], [10.0, 20.0, 30.0])
    assert got['right'] == 0.0 and got['judged'] == 3


def test_an_actual_tie_has_no_winner_and_a_forecast_tie_is_a_coin_flip():
    # 10 and 10.001 differ by far less than 1 in 100 of the spread: no actual winner
    got = R.pair_counts([1.0, 2.0, 3.0], [10.0, 10.001, 30.0])
    assert got['actual_ties'] == 1 and got['judged'] == 2 and got['right'] == 2.0
    # the forecast scores the first two exactly equal: half right
    got = R.pair_counts([1.0, 1.0, 3.0], [10.0, 20.0, 30.0])
    assert got['forecast_ties'] == 1 and got['right'] == 2.5
    # every actual outcome equal: nothing to judge
    assert R.pair_counts([1.0, 2.0], [5.0, 5.0])['judged'] == 0


def test_a_missing_or_nonfinite_value_leaves_the_start_out():
    assert R.pair_counts([1.0, None, 3.0], [1.0, 2.0, 3.0]) is None
    assert R.pair_counts([1.0, float('nan')], [1.0, 2.0]) is None
    assert R.pair_counts([1.0, 2.0], [1.0, 2.0, 3.0]) is None
    out = R.ranking([dict(body=0, predicted=[1.0, None], actual=[1.0, 2.0]),
                     dict(body=0, predicted=[1.0, 2.0], actual=[1.0, 2.0])], 'x')
    assert out['starts'] == 1 and out['starts_left_out'] == 1


def test_spread_ratio_is_forecast_spread_over_actual_spread():
    assert R.spread_ratio([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(0.5)
    assert R.spread_ratio([1.0, 2.0], [3.0, 3.0]) is None


def test_a_forecast_that_knows_every_outcome_scores_100_and_is_used_when_the_planner_wins():
    out = R.ranking(starts(reference_gap=0.5), 'exact', planners=('chosen',), reference='reflex', draws=400)
    assert out['per_100'] == 100.0 and out['interval'] == [100.0, 100.0]
    # 16 starts x 10 pairs; a pair whose two draws land within 1 in 100 of the spread is a tie
    assert out['spread_ratio'] == pytest.approx(1.0) and out['bodies'] == 8
    assert out['pairs_judged'] + out['actual_ties_left_out'] == 160 and out['pairs_judged'] >= 150
    assert out['planners']['chosen']['interval'][1] < 0 and out['planners']['chosen']['bodies_better'] == 8
    assert out['better_than_chance'] and not out['below_chance'] and out['case'] == 'ranking_used'


def test_a_usable_ranking_whose_planner_does_not_beat_its_start_is_case_3():
    out = R.ranking(starts(reference_gap=-0.5), 'exact', planners=('chosen',), reference='reflex', draws=400)
    assert out['per_100'] == 100.0 and out['planners']['chosen']['bodies_worse'] == 8
    assert out['case'] == 'case_3_planner_not_better'
    assert R.ranking(starts(), 'no planner', draws=400)['case'] == 'better_than_chance'


def half_right(scale):
    """8 bodies x 2 starts: bodies 0-3 forecast scale x the outcome (every pair right), bodies 4-7 minus
    that (every pair wrong): exactly 50 in 100 with no tie left out (tie_share 0), an interval that
    reaches 50, spread ratio |scale|."""
    rows = starts()
    for row in rows:
        sign = 1.0 if row['body'] < 4 else -1.0
        row['predicted'] = [sign * scale * a for a in row['actual']]
    return rows


def test_a_forecast_that_separates_the_commands_a_hundredth_as_much_as_reality_at_chance_is_case_1():
    out = R.ranking(half_right(0.01), 'under-separated', draws=400, tie_share=0.0)
    assert out['per_100'] == 50.0 and out['interval'][0] <= 50 <= out['interval'][1]
    assert out['spread_ratio'] == pytest.approx(0.01) and out['case'] == 'case_1_under_separated'
    wide = R.ranking(half_right(3.0), 'over-separated', draws=400, tie_share=0.0)
    assert wide['per_100'] == 50.0 and wide['case'] == 'case_2_wrong_order'
    old = dict(json.loads(json.dumps(out)), case='case_1_action_insensitive')
    assert R.valid_block(old), 'the label results were written under at commit fd713b068 stays valid'


def test_a_forecast_that_separates_the_commands_backwards_is_case_2_below_chance():
    out = R.ranking(starts(forecast=lambda a, rng: -8.0 * a), 'backwards', draws=400)
    assert out['per_100'] == 0.0 and out['spread_ratio'] == pytest.approx(8.0)
    assert out['below_chance'] and out['case'] == 'case_2_wrong_order'


def test_a_forecast_that_reliably_picks_the_loser_is_case_2_however_small_its_spread():
    """Review of 2026-09-25: a forecast below chance responds to the command, the wrong way; a spread
    under 1 must not relabel it as one that separates the commands less than reality."""
    out = R.ranking(starts(forecast=lambda a, rng: -0.01 * a), 'backwards, small', draws=400)
    assert out['per_100'] == 0.0 and out['spread_ratio'] == pytest.approx(0.01)
    assert out['below_chance'] and out['case'] == 'case_2_wrong_order'


def test_the_interval_resamples_whole_bodies_and_one_body_is_undetermined():
    # body 0 always right, body 1 always wrong: the interval spans both
    mixed = ([dict(body=0, predicted=[1.0, 2.0], actual=[1.0, 2.0])] * 4
             + [dict(body=1, predicted=[2.0, 1.0], actual=[1.0, 2.0])] * 4)
    out = R.ranking(mixed, 'two bodies', draws=400)
    assert out['per_100'] == 50.0 and out['interval'][0] == 0.0 and out['interval'][1] == 100.0
    one = R.ranking([dict(body=0, predicted=[1.0, 2.0], actual=[1.0, 2.0])] * 3, 'one body', draws=400)
    assert one['interval'] is None and one['case'] == 'undetermined'
    assert R.case_of([51.0, 60.0], None) == 'better_than_chance'
    assert R.case_of([50.0, 60.0], 0.5) == 'case_1_under_separated', 'exactly 50 is not better than chance'
    assert R.case_of([40.0, 60.0], 1.0) == 'case_2_wrong_order'
    assert R.case_of([40.0, 60.0], 0.999) == 'case_1_under_separated'
    assert R.case_of([10.0, 49.9], 0.1) == 'case_2_wrong_order'


def test_a_block_read_back_from_json_is_checked_for_every_field():
    block = json.loads(json.dumps(R.ranking(starts(reference_gap=0.5), 'exact', planners=('chosen',),
                                            reference='reflex', draws=200)))
    assert R.valid_block(block)
    for key, value in [('per_100', None), ('per_100', 101.0), ('interval', None), ('interval', [60.0, 50.0]),
                       ('interval', [1.0]), ('pairs_judged', 0), ('pairs_judged', True), ('spread_ratio', None),
                       ('spread_ratio', float('nan')), ('case', 'unknown'), ('case', 'undetermined')]:
        bad = dict(block, **{key: value})
        assert not R.valid_block(bad), key
    empty = R.ranking([dict(body=b, predicted=[1.0, 2.0], actual=[3.0, 3.0]) for b in range(8)], 'ties', draws=50)
    empty = json.loads(json.dumps(empty))
    assert empty['pairs_judged'] == 0 and empty['case'] == 'undetermined'
    assert not R.valid_block(empty) and R.valid_block(empty, allow_undetermined=True)
    assert R.valid_block(empty, allow_empty=True), 'the first name, still passed by a scheduler on commit fd713b068'
    # pairs with a winner in one body only: a value but no interval
    one = json.loads(json.dumps(R.ranking([dict(body=0, predicted=[1.0, 2.0], actual=[1.0, 2.0])]
                                          + [dict(body=b, predicted=[1.0, 2.0], actual=[3.0, 3.0]) for b in range(1, 8)],
                                          'one body', draws=50)))
    assert one['bodies'] == 1 and one['interval'] is None and one['per_100'] == 100.0
    assert not R.valid_block(one) and R.valid_block(one, allow_undetermined=True)
    assert not R.valid_block(dict(one, bodies=2), allow_undetermined=True), 'two bodies must carry an interval'
    assert not R.valid_block(None) and not R.valid_block([])


PLANNER_PROBE = sorted((DATA / 'Saved/mimo_bridge/ability_rechecks').glob('planner_candidates_399100_attempt1_*/starts.jsonl'))
HAND_ROUNDS = sorted((DATA / 'Saved/mimo_bridge/ability_rechecks').glob('body_and_synergy_*_attempt*/trials.jsonl'))
FIRST = DATA / 'docs/gain_of_function_v4/measurements/candidate_ranking_2026-09-24.json'


@pytest.mark.skipif(not PLANNER_PROBE or not FIRST.is_file(), reason='the saved planner-candidate round is not on this machine')
def test_it_reproduces_the_first_computation_of_the_planner_candidate_round():
    """The coordinator's docs/gain_of_function_v4/candidate_ranking.py read the same starts.jsonl: its
    counts and the ones here agree wherever no forecast gives two commands exactly the same value (it
    left such a pair out; here it counts half)."""
    rows = [json.loads(line) for line in PLANNER_PROBE[0].open(encoding='utf-8')]
    first = [r for r in json.loads(FIRST.read_text(encoding='utf-8'))['rows'] if r['source'] == 'planner probe']

    def block(pred, actual):
        return R.ranking([dict(body=r['body'], names=r['candidates'], predicted=pred(r), actual=actual(r)) for r in rows],
                         'x', planners=('chosen',), reference='reflex', draws=400)
    goal = lambda r: r['goal_distance_pred']['1']
    wellness = lambda r: [-r['production']['pred_wellness'][c][0] for c in r['candidates']]
    ours = [block(goal, lambda r: [c[0] for c in r['actual']['goal_distance']]),
            block(goal, lambda r: [c[-1] for c in r['actual']['goal_distance']]),
            block(wellness, lambda r: [-float(np.sum(c[0])) for c in r['actual']['wellness_terms']]),
            block(wellness, lambda r: [-float(np.mean([np.sum(t) for t in c])) for c in r['actual']['wellness_terms']])]
    for mine, theirs in zip(ours, first):
        if mine['forecast_ties_counted_half'] == 0:
            assert mine['pairs_judged'] == theirs['pairs_judged'] and mine['pairs_right'] == theirs['right']
            assert mine['per_100'] == pytest.approx(theirs['right_per_100'], abs=0.05)
        assert mine['spread_ratio'] == pytest.approx(theirs['spread_ratio_median'], rel=1e-9)
        assert abs(mine['interval'][0] - theirs['interval_per_100'][0]) <= 3.0


@pytest.mark.skipif(not HAND_ROUNDS or not FIRST.is_file(), reason='the saved hand-goal rounds are not on this machine')
def test_it_reproduces_the_first_computation_of_the_hand_goal_rounds():
    first = [r for r in json.loads(FIRST.read_text(encoding='utf-8'))['rows'] if r['source'] == 'hand-goal comparison'][0]
    listed = json.loads(FIRST.read_text(encoding='utf-8'))['notes']['hand_rounds']['rounds']
    rows = []
    for path in HAND_ROUNDS:
        if path.parent.name not in listed:
            continue
        for line in path.open(encoding='utf-8'):
            row = json.loads(line)
            arms = [m for m in ('controller', 'direct', 'synergy') if m in row['outcomes']]
            rows.append(dict(body=row['body'], names=arms,
                             predicted=[row['outcomes'][m]['predicted_goal_error'] for m in arms],
                             actual=[row['outcomes'][m]['physical_goal_error'] for m in arms]))
    out = R.ranking(rows, 'x', planners=('direct', 'synergy'), reference='controller', draws=400)
    assert out['forecast_ties_counted_half'] == 0
    assert out['pairs_judged'] == first['pairs_judged'] and out['pairs_right'] == first['right']
    assert out['actual_ties_left_out'] == first['ties_left_out']
    assert out['spread_ratio'] == pytest.approx(first['spread_ratio_median'], rel=1e-9)
    assert out['case'] == 'case_2_wrong_order' and out['below_chance']
