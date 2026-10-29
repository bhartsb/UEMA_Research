# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The three-layer self-touch comparison (docs/gain_of_function_v3/probe_saved_self_touch_choices.py):
reachability, prediction and selection on synthetic inputs, the scheduler's validator and the
dashboard's words for its result. No model is loaded and no world is built."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mimo_bridge.body_layout import names_for

ROOT = Path(__file__).resolve().parents[4]
PATH = ROOT / 'docs/gain_of_function_v3/probe_saved_self_touch_choices.py'
SPEC = importlib.util.spec_from_file_location('saved_self_touch_choices', PATH)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)
SCRIPT = 'probe_saved_self_touch_choices.py'
BAND = 0.0734
BODY = [name for name in names_for('trunk_v1') if name != 'gaze']


def asked(**sizes):
    """An asked move per body handle (asked_move's form): asked(wrist_L=[5, 0, .2]) asks 5 cm of the left
    wrist, nothing of any other handle."""
    return [list(sizes.get(name, [0.0, 0.0, 0.0])) for name in BODY]


def everywhere(size):
    return asked(**{name: size for name in BODY})


def make_row(body, touch, contact=None, *, p=None, wellness=None, objective=None, families=None, untouched=None,
             verified=True, start=0, touch_tenure=None, contact_tenure=None, hand=None, hand_tenure=None,
             regions=None, finger_only=(), moves=None, patches=3, hand_stage=None, shared=None, onsets=None,
             by_sense=None, senses=()):
    """One start: touch[c] = patches command c newly touched within 50 ms; contact[c] = patches with any
    own-part contact within 50 ms (defaults to the new touches); hand[c] = the new touches made by a hand
    on another body area (defaults to none); regions defaults to one region per patch; moves[c] = the move
    command c asked of each handle (defaults to none at all, so every pair is size-matched); shared = the
    (wellness, objective) scored with the controller's finger goal (defaults to the executed ones);
    onsets[c] = the region pairs of command c's hand-to-body touches."""
    K = len(touch)
    contact = [list(t) for t in touch] if contact is None else contact
    p = [[0.0] * patches for _ in range(K)] if p is None else p
    hand = [[] for _ in range(K)] if hand is None else hand
    wellness = [0.0] * K if wellness is None else wellness
    objective = [0.0] * K if objective is None else objective
    shared = (list(wellness), list(objective)) if shared is None else shared
    onsets = [[] for _ in range(K)] if onsets is None else onsets
    return dict(body=body, start=start, step=0, candidates=[f'c{i}' for i in range(K)],
                families=families or ['controller'] * K, intended=[None] * K, intended_target=[None] * K,
                untouched=[True] * patches if untouched is None else untouched,
                patch_regions=[f'region {i}' for i in range(patches)] if regions is None else regions,
                targets=[], target_gaps_cm={}, p_self=p, p_self_forward_model_only=p, p_self_order_input_only=p,
                wellness=wellness, objective=objective, wellness_shared_fingers=shared[0],
                objective_shared_fingers=shared[1], goal_senses=list(senses),
                goal_distance_by_sense=by_sense, goal_distance_by_sense_shared_fingers=by_sense,
                objective_wellness_weight=0.0, planner_route=dict(verified=verified, effort_per_body=True),
                finger_only_pairs=[list(x) for x in finger_only], hand_stage=hand_stage, request_clipped=[None] * K,
                touch_50ms=touch, contact_50ms=contact,
                asked_move=[asked() for _ in range(K)] if moves is None else moves,
                touch_tenure=touch if touch_tenure is None else touch_tenure,
                contact_tenure=contact if contact_tenure is None else contact_tenure,
                touch_hand_50ms=hand, touch_hand_tenure=hand if hand_tenure is None else hand_tenure,
                onset_regions_50ms=onsets, onset_regions_tenure=onsets,
                onset_hand_regions_50ms=onsets if any(hand) else [[] for _ in range(K)],
                onset_hand_regions_tenure=onsets if any(hand) else [[] for _ in range(K)])


FAMILIES = ['exploration', 'exploration', 'reach_push', 'reach_push', 'hold', 'hold']


def layered_rows(bodies=8, starts=3, *, hand=True, knows=True, second=True, toucher_agrees=True, matched_agrees=False,
                 lead=0.5, families=None, **kw):
    """Six commands per start over five patches. Command 0 (an exploration) newly touches patches 1 and 3,
    and, when `second`, command 2 (a push reach) newly touches patches 2 and 4 (a hand on another body area
    when `hand`). Commands 1 and 3, of the touchers' own kinds, touched nothing: their matched commands.
    Commands 4 and 5 (holding still) trail all four in predicted wellness by `lead` or more. A head that
    `knows` gives each toucher 90 in 100 on its own patches and 20 elsewhere, against 10 for every other
    command; otherwise every command gets 10 everywhere. The planner's objective prefers each toucher (and
    each matched command) to the two held commands when it agrees."""
    touch = [[1, 3], [], [2, 4] if second else [], [], [], []]
    t = [.5 if toucher_agrees else 5.0]
    m = [.5 if matched_agrees else 5.0]
    objective = t + m + t + m + [1.0, 1.1]
    wellness = [4.0, 4.0, 4.0, 4.0, 4.0 - lead, 3.99 - lead]
    if knows:
        p = [[.2, .9, .2, .9, .2], [.1] * 5, [.2, .2, .9, .2, .9], [.1] * 5, [.1] * 5, [.1] * 5]
    else:
        p = [[.1] * 5 for _ in range(6)]
    onsets = [['left hand|trunk'], [], ['right hand|trunk'] if second else [], [], [], []]
    return [make_row(b, [list(x) for x in touch], start=s, families=list(families or FAMILIES), patches=5,
                     p=[list(x) for x in p], hand=[list(x) for x in touch] if hand else None,
                     wellness=list(wellness), objective=list(objective), onsets=onsets, **kw)
            for b in range(bodies) for s in range(starts)]


def summarise(rows, bodies=8):
    return PROBE.summarise(rows, bodies, BAND)


def _head(bias, gain, size):
    return [float(v) for v in 1.0 / (1.0 + np.exp(-(np.asarray(bias) + np.asarray(gain) * size)))]


# ---------------------------------------------------------------------------------------- layer 1
def test_pairs_need_a_new_touch_and_a_command_that_left_the_patch_alone():
    row = make_row(0, touch=[[1, 0], [], [], []], contact=[[1, 0], [], [1], []], untouched=[False, True, True])
    # patch 0 was touched at the start; command 2 touched patch 1 without a new onset (neither side)
    assert PROBE.patch_pairs(row) == [(0, 1, 1), (0, 3, 1)]
    assert PROBE.reachable(row) and not PROBE.reachable(row, kind='hand_to_body')
    assert PROBE.patch_pairs(make_row(0, [[], []])) == [] and not PROBE.reachable(make_row(0, [[], []]))


def test_a_round_where_nothing_touches_says_not_reachable_and_measures_nothing_else():
    rows = [make_row(b, [[], [], []]) for b in range(8) for _ in range(3)]
    layers, passed = summarise(rows)
    reach = layers['reachability']
    assert reach['reachable_starts_50ms'] == 0 and reach['reachable_starts_tenure'] == 0
    assert reach['verdict'] == "not reachable from these starts within 50 ms or within the order's tenure"
    for kind in PROBE.KINDS:
        block = layers['prediction'][kind]
        assert block['all_pairs']['bodies'] == 0 and block['verdict'] == PROBE.NOT_REACHABLE
        for part in ('whether', 'where'):
            assert block[part]['value'] is None and block[part]['distance_to_goal_line'] is None
        stat = layers['selection'][kind]
        assert stat['bodies'] == 0 and stat['value'] is None and stat['verdict'] == PROBE.NOT_REACHABLE
    assert passed is False
    json.dumps(layers, allow_nan=False)


def test_a_touch_reached_only_within_the_whole_order_is_reported_as_such():
    rows = [make_row(b, [[], []], touch_tenure=[[1], []], contact_tenure=[[1], []], hand_tenure=[[1], []])
            for b in range(8)]
    layers, passed = summarise(rows)
    reach = layers['reachability']
    assert (reach['reachable_starts_50ms'], reach['reachable_starts_hand_to_body_tenure']) == (0, 8)
    assert reach['starts_offering_a_hand_to_body_pair_tenure'] == 8
    assert reach['verdict'] == "not reachable from these starts within 50 ms; reachable within the order's tenure"
    assert layers['prediction']['hand_to_body']['all_pairs']['bodies'] == 0 and not passed


def test_every_command_touching_is_reachable_but_offers_no_pair():
    rows = [make_row(b, [[1], [1]], hand=[[1], [1]]) for b in range(8)]
    layers, _ = summarise(rows)
    assert layers['reachability']['reachable_starts_hand_to_body_50ms'] == 8
    assert layers['reachability']['starts_offering_a_hand_to_body_pair_50ms'] == 0
    assert layers['prediction']['verdict'] == PROBE.NOT_REACHABLE


def test_touching_branches_cut_reach_requests_and_touch_regions_are_counted():
    families = ['controller', 'hold', 'reach_onto', 'reach_push']
    rows = [make_row(b, [[], [1], [], [2]], families=families, hand=[[], [], [], [2]],
                     onsets=[[], ['right hand|right hand'], [], ['left hand|trunk']]) for b in range(2)]
    for r in rows:
        r['intended'] = [None, None, [2], [2]]
        r['request_clipped'] = [None, None, False, True]
    reach = summarise(rows, 2)[0]['reachability']
    assert reach['touching_branches_50ms'] == {
        'controller': dict(touched=0, touched_hand_to_body=0, branches=2),
        'hold': dict(touched=2, touched_hand_to_body=0, branches=2),
        'reach_onto': dict(touched=0, touched_hand_to_body=0, branches=2),
        'reach_push': dict(touched=2, touched_hand_to_body=2, branches=2)}
    assert reach['reach_commands'] == dict(built=4, requests_cut_by_the_reach_clamp={'reach_push': 2},
                                           own_target_touched_50ms=2, own_target_touched_tenure=2)
    assert reach['new_touch_region_pairs_50ms'] == {'left hand|trunk': 2, 'right hand|right hand': 2}


def test_horizon_sets_split_the_first_50_ms_from_the_order_and_skip_structural_and_touched_patches():
    names = ['hand', 'arm', 'chest', 'leg', 'finger']
    frames = [dict(patch_state=[0, 0, 0, 0, 0], self_contact_onsets=[]) for _ in range(10)]
    frames[1]['self_contact_onsets'] = [dict(pair=['hand', 'thumb'], regions=['right hand', 'right hand'],
                                             structural=True)]
    frames[1]['patch_state'] = [2, 0, 0, 0, 0]
    frames[2]['self_contact_onsets'] = [dict(pair=['hand', 'chest'], regions=['right hand', 'trunk'], structural=False),
                                        dict(pair=['finger', 'hand'], regions=['right hand', 'right hand'],
                                             structural=False)]
    frames[2]['patch_state'] = [2, 0, 2, 0, 2]
    frames[7]['self_contact_onsets'] = [dict(pair=['arm', 'leg'], regions=['left lower arm', 'left upper leg'],
                                             structural=False)]
    frames[7]['patch_state'] = [0, 2, 0, 2, 0]
    untouched = [False, True, True, True, True]           # the hand was touched at the start
    assert PROBE.horizon_sets(frames, 5, untouched, names) == dict(
        contact=[0, 2, 4], touched=[2, 4], by_hand=[2], regions=['right hand|right hand', 'right hand|trunk'],
        hand_regions=['right hand|trunk'])
    assert PROBE.horizon_sets(frames, 10, untouched, names) == dict(
        contact=[0, 1, 2, 3, 4], touched=[1, 2, 3, 4], by_hand=[2],
        regions=['left lower arm|left upper leg', 'right hand|right hand', 'right hand|trunk'],
        hand_regions=['right hand|trunk'])


def test_the_self_touch_under_test_is_a_hand_on_another_body_area():
    """The second review: 'two different regions' also took an upper arm on the trunk and a hand on its own
    forearm. The rule is the reach targets' rule: a hand on a region that is neither that hand nor that
    hand's own arm."""
    for regions in (['left hand', 'trunk'], ['trunk', 'right hand'], ['right hand', 'left hand'],
                    ['left hand', 'right lower arm'], ['right hand', 'left upper leg'], ['left hand', 'head']):
        assert PROBE.hand_to_body(dict(regions=regions)), regions
    for regions in (['left hand', 'left lower arm'], ['left upper arm', 'left hand'], ['left hand', 'left hand'],
                    ['left upper arm', 'trunk'], ['right lower arm', 'right upper arm'], ['', 'trunk'], ['trunk']):
        assert not PROBE.hand_to_body(dict(regions=regions)), regions
    assert not PROBE.hand_to_body({})


def test_a_structural_or_already_touched_patch_is_never_a_new_touch():
    outcome = {'trace': [{'self_contact_onsets': [
        {'pair': ['palm', 'thumb_base'], 'structural': True},
        {'pair': ['index', 'thumb_tip'], 'structural': False},
        {'pair': ['wrist', 'arm'], 'structural': False}]}]}
    assert PROBE.nonstructural_onsets(outcome, ['palm', 'index', 'thumb_tip']) == ['index', 'thumb_tip']


# ---------------------------------------------------------------------------------------- layer 2
def test_a_head_that_foresees_where_each_touch_lands_passes_both_judged_comparisons():
    layers, _ = summarise(layered_rows())
    block = layers['prediction']['hand_to_body']
    assert block['all_pairs']['value'] == 1.0 and block['all_pairs']['judged'] is False
    assert block['all_pairs']['starts'] == 24 and block['all_pairs']['pairs'] == 24 * 20
    whether, where = block['whether'], block['where']
    assert whether['value'] == 1.0 and whether['pairs'] == 24 * 20
    assert whether['per_body_events'] == {str(b): 6 for b in range(8)}          # a start and a toucher
    assert where['value'] == 1.0 and where['pairs'] == 24 * 4 and where['per_body_events'] == {str(b): 3 for b in range(8)}
    for part in (whether, where):
        assert part['votes']['bodies_voting'] == 8 and part['votes']['bodies_above'] == 8
        assert part['votes']['p_value'] == pytest.approx(1 / 256) and part['verdict'] == PROBE.ABOVE
        assert part['goal_line'] == 0.8 and part['distance_to_goal_line'] == pytest.approx(-0.2)
    assert whether['touch_region_pairs'] == {'left hand|trunk': 24 * 10, 'right hand|trunk': 24 * 10}
    assert where['touch_region_pairs'] == {'left hand|trunk': 24 * 4, 'right hand|trunk': 24 * 4}
    assert block['verdict'] == PROBE.ABOVE and layers['prediction']['verdict'] == PROBE.ABOVE


def test_crossed_pairs_are_two_commands_that_touched_different_patches_and_left_each_others_alone():
    row = make_row(0, [[1], [2], [1, 3], []], contact=[[1], [2, 3], [1, 3], []], patches=4)
    # command 1 had own contact on patch 3, so command 2's patch 3 is not crossed with command 1
    assert PROBE.crossed_pairs(row) == [(0, 1, 1, 2), (1, 2, 2, 1)]
    row['untouched'] = [True, True, False, True]                     # patch 2 was touched at the start
    assert PROBE.crossed_pairs(row) == []


def test_the_third_reviews_wrist_driven_head_is_not_credited():
    """The third review's counterexample: p = sigmoid(-3 + g_patch x the move asked of the most-moved wrist),
    with the hand patches the most sensitive. The touching reaches move one wrist 3 and 2 cm; the
    explorations move every handle, the wrists 1.5 cm. Every pair favours a reach, but no pair is
    size-matched handle by handle, and on the crossed pairs the head ranks both reaches the same way on
    both patches: exactly 50 of 100."""
    gain = [1.0, 0.5, 0.1, 0.1, 1.0]          # left hand, trunk, left upper leg, right foot, right hand
    bias = [-3.0] * 5
    left, right = asked(wrist_L=[3.0, 0.0, .2]), asked(wrist_R=[2.0, 0.0, .2])
    explore = everywhere([1.5, 30.0, .25])
    rows = [make_row(b, [[0, 1], [4, 2]] + [[] for _ in range(4)], hand=[[0, 1], [4, 2]] + [[] for _ in range(4)],
                     start=s, families=['reach_onto', 'reach_onto'] + ['exploration'] * 4, patches=5,
                     regions=['left hand', 'trunk', 'left upper leg', 'right foot', 'right hand'],
                     p=[_head(bias, gain, 3.0), _head(bias, gain, 2.0)] + [_head(bias, gain, 1.5) for _ in range(4)],
                     moves=[left, right] + [explore] * 4)
            for b in range(8) for s in range(3)]
    block = summarise(rows)[0]['prediction']['hand_to_body']
    assert block['all_pairs']['value'] > 0.8
    assert block['whether']['pairs'] == 0 and block['whether']['verdict'].startswith('not measured')
    assert block['where']['value'] == 0.5 and block['where']['pairs'] == 24 * 4
    assert block['where']['votes']['bodies_above'] == 0 and block['where']['verdict'] == PROBE.NOT_ABOVE
    assert block['verdict'] == PROBE.NOT_ABOVE


@pytest.mark.parametrize('summary', ['size', 'signed'])
def test_a_head_that_ranks_every_patch_by_one_summary_of_the_command_scores_half_on_crossed_pairs(summary):
    """Whatever one number the head reads from a command (its size, or its signed displacement along one
    axis), with gains of one sign it favours the same command on both patches of a crossed pair."""
    rng = np.random.default_rng(3)
    rows = []
    for b in range(8):
        for s in range(3):
            u = rng.normal(0.0, 1.0, 4) if summary == 'signed' else rng.uniform(0.0, 3.0, 4)
            gain = rng.uniform(0.2, 1.5, 4)
            rows.append(make_row(b, [[0], [1], [2], [3]], hand=[[0], [1], [2], [3]], start=s, patches=4,
                                 families=['reach_push'] * 4, regions=['left hand', 'trunk', 'head', 'right hand'],
                                 p=[_head([-2.0] * 4, gain, u[c]) for c in range(4)]))
    where = summarise(rows)[0]['prediction']['hand_to_body']['where']
    assert where['pairs'] + where['unresolved_crossed_pairs'] == 24 * 6 and where['pairs'] > 24 * 5
    assert where['value'] == 0.5 and where['votes']['bodies_above'] == 0


def _reach_rows(p_toucher, p_reaches, p_small, starts=3):
    """Command 0 (a push reach) touches patch 1 with a hand; commands 1 and 2 (push reaches toward farther
    patches: a bigger move of the same wrist) and the controller, hold and production commands (smaller
    moves of every handle) leave it alone."""
    toucher, farther, small = asked(wrist_L=[5.0, 5.0, 5.0]), asked(wrist_L=[6.0, 6.0, 5.0]), everywhere([.1, .1, .1])
    return [make_row(b, [[1]] + [[] for _ in range(5)], hand=[[1]] + [[] for _ in range(5)], start=s,
                     families=['reach_push'] * 3 + ['controller', 'hold', 'production'],
                     p=[list(p_toucher)] + [list(p_reaches)] * 2 + [list(p_small)] * 3,
                     moves=[toucher, farther, farther, small, small, small])
            for b in range(8) for s in range(starts)]


def test_a_head_whose_chances_grow_with_any_handles_move_is_held_to_a_coin_flip_on_whether():
    """The first review's counterexample: the head raises every patch's chance with the size of the move.
    It favours the toucher over the three smaller moves (60 of 100 over every pair) and gives it the lower
    chance than the two bigger reaches, the only pairs in which the toucher asked no handle for more."""
    bias = [-2.0, -1.5, -1.0]
    stat = summarise(_reach_rows(_head(bias, .2, 5.0), _head(bias, .2, 6.0), _head(bias, .2, .1)))[0]
    block = stat['prediction']['hand_to_body']
    assert block['all_pairs']['value'] == pytest.approx(3 / 5)
    assert block['whether']['value'] == 0.0 and block['whether']['pairs'] == 48
    assert block['whether']['verdict'] == PROBE.NOT_ABOVE and block['verdict'] == PROBE.NOT_ABOVE


def test_a_head_that_knows_which_reach_touches_is_credited_on_whether():
    stat = summarise(_reach_rows((0.2, .9, 0.2), (0.3, .5, 0.3), (0.1, .1, 0.1)))[0]
    whether = stat['prediction']['hand_to_body']['whether']
    assert whether['value'] == 1.0 and whether['votes']['bodies_above'] == 8 and whether['verdict'] == PROBE.ABOVE


def test_no_bigger_compares_every_handle():
    assert PROBE.no_bigger(asked(wrist_L=[1.0, 2.0, 3.0]), asked(wrist_L=[1.0, 2.0, 3.0 + 1e-9]))
    assert PROBE.no_bigger(asked(wrist_L=[1.0, 2.0, 3.0 + 1e-9]), asked(wrist_L=[1.0, 2.0, 3.0]))   # within tolerance
    # the third review: an arm's sum no bigger is not enough when one handle of it asks for more
    assert not PROBE.no_bigger(asked(wrist_L=[3.0, 0.0, 0.2]), asked(wrist_L=[1.5, 30.0, .25], upperarm_L=[9.0, 9.0, 9.0]))
    assert [PROBE.move_relation(a, b) for a, b in (
        (asked(), asked()), (asked(head=[1, 0, 0]), asked(head=[2, 0, 0])), (asked(head=[2, 0, 0]), asked(head=[1, 0, 0])),
        (asked(head=[2, 0, 0]), asked(chest=[1, 0, 0])))] == ['equal', 'smaller', 'bigger', 'mixed']


def test_the_asked_move_is_read_per_handle_as_the_contact_head_reads_it():
    names = ['wrist_L', 'thumb_L', 'thigh_R', 'head', 'chest', 'gaze']
    achieved = np.zeros((1, 6, 10))
    achieved[0, :, 6] = 1.0                                     # every handle at the identity turn
    achieved[0, 0, 0:3] = np.array([10.0, 0.0, 0.0]) / 200
    order = np.zeros((1, 6, 9))
    order[0, :, 6] = 1.0
    order[0, 0, 0:3] = [13.0, 4.0, 0.0]                         # the wrist asked 5 cm
    order[0, 1, 0:3] = [0.0, 0.0, 2.0]                          # the thumb asked 2 cm
    order[0, 2, 3:7] = [np.sin(np.pi / 8), 0.0, 0.0, np.cos(np.pi / 8)]     # the right thigh asked 45 degrees
    order[0, 4, 0:3] = [50.0, 0.0, 0.0]                         # the head reads every handle's asked position
    order[0, 4, 3:7] = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]     # the chest asked 90 degrees
    order[0, 5, 0:3] = [99.0, 0.0, 0.0]                         # the eyes' row is left out
    order[0, :, 8] = [.2, .1, .3, .4, .5, .9]
    moves = PROBE.asked_move(order, achieved, names)
    assert len(moves) == 5
    np.testing.assert_allclose(moves, [[5.0, 0.0, .2], [2.0, 0.0, .1], [0.0, 45.0, .3], [0.0, 0.0, .4],
                                       [50.0, 90.0, .5]], rtol=1e-9, atol=1e-9)


def test_turns_stay_exact_near_zero():
    """The second review: 2 arccos(|dot|) resolves about 0.03 degrees from float32 quaternions."""
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    for degrees in (1e-4, 1.0, 90.0, 179.0):
        half = np.radians(degrees) / 2
        turned = np.array([0.0, np.sin(half), 0.0, np.cos(half)])
        assert PROBE.turn_degrees(identity, turned) == pytest.approx(degrees, rel=1e-9)
        assert PROBE.turn_degrees(turned, identity) == pytest.approx(degrees, rel=1e-9)
    assert PROBE.turn_degrees(identity, -identity) == 0.0              # the same orientation


def test_an_unresolved_crossed_pair_is_left_out_and_counted():
    p = [[.9, .3], [.3, .3 + 1e-9]]                   # the head does not tell the two apart on patch 1
    row = make_row(0, [[0], [1]], hand=[[0], [1]], p=p, patches=2)
    block = PROBE.prediction_block([row], 'p_self', 'hand_to_body', 1)
    assert block['where']['pairs'] == 0 and block['where']['unresolved_crossed_pairs'] == 1


def wrong_in(rows, bodies):
    """The head gives every touching command the lower chance in these bodies."""
    for r in rows:
        if r['body'] in bodies:
            r['p_self'] = r['p_self_forward_model_only'] = r['p_self_order_input_only'] = (
                [[.1] * 5 if r['touch_50ms'][c] else [.95] * 5 for c in range(len(r['families']))])
    return rows


def test_the_sign_rule_needs_five_voting_bodies_and_seven_of_eight():
    assert PROBE.MIN_VOTING_BODIES == 5
    assert PROBE.exact_tail(8, 7) == pytest.approx(9 / 256) and PROBE.exact_tail(8, 6) == pytest.approx(37 / 256)
    assert PROBE.exact_tail(4, 4) == 1 / 16 > PROBE.ALPHA

    def part(rows):
        return summarise(rows)[0]['prediction']['hand_to_body']['whether']
    seven = part(wrong_in(layered_rows(), {7}))
    assert (seven['votes']['bodies_above'], seven['votes']['p_value'], seven['verdict']) == (7, 9 / 256, PROBE.ABOVE)
    six = part(wrong_in(layered_rows(), {6, 7}))
    assert (six['votes']['bodies_above'], six['verdict']) == (6, PROBE.NOT_ABOVE)
    six_of_seven = [r for r in wrong_in(layered_rows(), {6}) if r['body'] != 7]
    assert part(six_of_seven)['votes']['p_value'] == 8 / 128 and part(six_of_seven)['verdict'] == PROBE.NOT_ABOVE
    four = part([r for r in layered_rows() if r['body'] < 4])
    assert four['verdict'].startswith('not decided: 4 of 8 bodies had at least 5 judged pairs from at least 2')


def test_both_sides_of_one_touch_are_one_touch_event():
    """The third review: the two patches of one hand-on-trunk onset are one touch. One start per body with
    one toucher gives each body 10 pairs from one touch event: no body votes."""
    part = summarise(layered_rows(starts=1, second=False))[0]['prediction']['hand_to_body']['whether']
    assert part['per_body_pairs'] == {str(b): 10 for b in range(8)} and part['per_body_events'] == {str(b): 1 for b in range(8)}
    assert part['votes']['bodies_voting'] == 0 and part['verdict'].startswith('not decided: 0 of 8 bodies')
    two = summarise(layered_rows(starts=2, second=False))[0]['prediction']['hand_to_body']['whether']
    assert two['votes']['bodies_voting'] == 8 and two['verdict'] == PROBE.ABOVE


def test_one_body_holding_most_pairs_at_chance_and_three_thin_bodies_decide_nothing():
    """The first review's counterexample to the pooled interval: one body with 100 pairs at chance and three
    bodies with one right pair each. Pooled it reads 51.5 of 100; the thin bodies cannot vote."""
    big = [make_row(0, [[1]] + [[] for _ in range(50)], hand=[[1]] + [[] for _ in range(50)], start=s,
                    families=['exploration'] * 51,
                    p=[[0.0, .5, 0.0]] + [[0.0, .5, 0.0]] * 25 + [[0.0, .5 + 1e-3, 0.0]] * 12 + [[0.0, .5 - 1e-3, 0.0]] * 13)
           for s in range(2)]
    thin = [make_row(b, [[1], []], hand=[[1], []], families=['exploration'] * 2, p=[[0.0, .9, 0.0], [0.0, .1, 0.0]])
            for b in (1, 2, 3)]
    part = summarise(big + thin)[0]['prediction']['hand_to_body']['whether']
    assert part['pairs'] == 103 and part['bodies'] == 4
    assert part['votes']['bodies_voting'] == 1
    assert part['verdict'].startswith('not decided: 1 of 8 bodies had at least 5 judged pairs')


def test_the_scores_pool_pairs_and_resample_bodies_as_the_scheduled_ranking_does():
    from mimo_bridge.tools import command_ranking
    per_body = {b: (float(b % 2) * 3 + (b == 6) * .5, 3 + b) for b in range(8)}
    stat = PROBE.pooled_share(per_body)
    assert stat['value'] == pytest.approx(sum(r for r, _ in per_body.values()) / sum(n for _, n in per_body.values()))
    shared = command_ranking.share_interval(per_body, draws=PROBE.BOOT_DRAWS, seed=PROBE.BOOT_SEED)
    np.testing.assert_allclose(100.0 * np.asarray(stat['interval']), shared, rtol=0, atol=1e-9)
    assert PROBE.pooled_share({3: (1.0, 2), 4: (0.0, 0)}) == dict(value=0.5, interval=None, bodies=1, per_body={'3': 0.5})


def test_other_touches_are_scored_beside_but_never_decide_the_pass():
    layers, passed = summarise(layered_rows(hand=False))
    reach = layers['reachability']
    assert (reach['reachable_starts_50ms'], reach['reachable_starts_hand_to_body_50ms']) == (24, 0)
    assert reach['verdict'].startswith('reachable within 50 ms only by touches other than a hand on another body area')
    assert layers['prediction']['any_touch']['verdict'] == PROBE.ABOVE
    assert layers['selection']['any_touch']['verdict'] == PROBE.ABOVE
    assert layers['prediction']['hand_to_body']['verdict'] == PROBE.NOT_REACHABLE and passed is False


def test_the_route_split_is_scored_over_every_pair():
    rows = layered_rows()
    for r in rows:
        r['p_self_forward_model_only'] = [[.1] * 5 if r['touch_50ms'][c] else [.9] * 5 for c in range(6)]
    routes = summarise(rows)[0]['prediction']['routes']
    assert routes['forward_model_only']['all_pairs']['value'] < 0.5
    assert routes['order_input_only']['all_pairs']['value'] == 1.0


# ---------------------------------------------------------------------------------------- layer 3
def test_selection_prefers_each_touching_command_to_its_matched_command():
    layers, passed = summarise(layered_rows())
    stat = layers['selection']['hand_to_body']
    assert stat['value'] == 1.0 and stat['pairs'] == 24 * 2 and stat['per_body_events'] == {str(b): 6 for b in range(8)}
    assert stat['votes']['voters'] == {str(b): 0.5 for b in range(8)} and stat['verdict'] == PROBE.ABOVE and passed
    assert stat['goal_line'] == 0.8 and stat['distance_to_goal_line'] == pytest.approx(-0.2)
    assert stat['move_relation'] == dict(smaller=0, bigger=0, equal=48, mixed=0)
    assert stat['unfiltered']['value'] == 1.0 and stat['unfiltered']['judged'] is False
    # each toucher leads the two held commands beyond the band, and the matched commands and the other
    # toucher within it
    assert stat['wellness_preference'] == dict(favours_toucher=24 * 4, favours_other=0, within_band=24 * 6)
    assert stat['touch_region_pairs'] == {'left hand|trunk': 24, 'right hand|trunk': 24}


def test_a_planner_indifferent_between_the_toucher_and_its_matched_command_is_not_above():
    stat = summarise(layered_rows(matched_agrees=True))[0]['selection']['hand_to_body']
    assert stat['value'] == 0.5 and stat['votes']['bodies_above'] == 0 and stat['verdict'] == PROBE.NOT_ABOVE


def test_the_selection_rule_keeps_its_declared_false_pass_rate_when_touching_does_not_matter_to_the_objective():
    """The null: the objective of a toucher and of its matched command are drawn alike, whether they are
    explorations or reaches. The round passes at most 5 times in 100 over 400 simulated rounds (the
    declared rule allows 9 in 256 with 8 voting bodies)."""
    rng = np.random.default_rng(20260925)
    passes = above = voters = 0
    for _ in range(400):
        rows = []
        for b in range(8):
            for s in range(3):
                objective = [float(v) for v in rng.normal(0.0, 1.0, 4)] + [5.0, 5.1]
                rows.append(make_row(b, [[1, 3], [], [2, 4], [], [], []], hand=[[1, 3], [], [2, 4], [], [], []],
                                     start=s, families=list(FAMILIES), patches=5, objective=objective,
                                     wellness=[4.0, 4.0, 4.0, 4.0, 3.5, 3.49]))
        stat = PROBE.selection_block(rows, 'hand_to_body', BAND, 8)
        passes += stat['verdict'] == PROBE.ABOVE
        above += stat['votes']['bodies_above']
        voters += stat['votes']['bodies_voting']
    assert voters == 400 * 8 and above <= voters / 2
    assert passes <= 400 * 5 / 100


def test_a_planner_that_only_prefers_smaller_moves_passes_and_the_record_shows_it():
    """The third review's confound, kept visible rather than hidden: an objective that only prefers smaller
    moves, where each touching command is the smaller move of its kind. Selection passes - the planner
    does prefer the commands that touched - and the record says the toucher asked the smaller move in every
    judged pair, the reading a size preference would give."""
    rows = layered_rows()
    for r in rows:
        r['asked_move'] = [asked(wrist_L=[3.0, 0.0, .2]), asked(wrist_L=[5.0, 0.0, .2]), asked(wrist_R=[3.0, 0.0, .2]),
                           asked(wrist_R=[5.0, 0.0, .2]), asked(wrist_L=[4.0, 0.0, .2]), asked(wrist_R=[4.0, 0.0, .2])]
        r['objective'] = [0.3, 0.5, 0.3, 0.5, 0.4, 0.4]
        r['objective_shared_fingers'] = list(r['objective'])
    stat = summarise(rows)[0]['selection']['hand_to_body']
    assert stat['verdict'] == PROBE.ABOVE and stat['move_relation'] == dict(smaller=48, bigger=0, equal=0, mixed=0)


def test_the_matched_command_is_of_the_toucher_s_own_kind():
    stat = summarise(layered_rows(families=['exploration', 'curl', 'reach_push', 'curl', 'hold', 'hold']))[0]
    stat = stat['selection']['hand_to_body']
    assert stat['pairs'] == 0
    assert stat['touchers_without_a_matched_command']['no_same_kind_command_that_touched_nothing'] == 48
    assert stat['verdict'].startswith('not measured: no touching command had a matched command (48 had no command')


def test_the_matched_command_must_look_the_same_as_the_toucher_to_the_evaluator():
    rows = layered_rows()
    for r in rows:
        r['wellness'][1] = r['wellness'][3] = 4.2
    stat = summarise(rows)[0]['selection']['hand_to_body']
    assert stat['pairs'] == 0 and stat['touchers_without_a_matched_command']['no_same_kind_command_within_the_band'] == 48


def test_each_matched_command_serves_one_toucher_per_start():
    rows = layered_rows(families=['exploration'] * 4 + ['hold', 'hold'])
    for r in rows:
        r['touch_50ms'][1] = r['touch_hand_50ms'][1] = r['contact_50ms'][1] = [0]    # only command 3 touched nothing
    stat = summarise(rows)[0]['selection']['hand_to_body']
    assert stat['pairs'] == 24 and stat['touchers_without_a_matched_command']['no_same_kind_command_that_touched_nothing'] == 48


def test_selection_counts_only_wellness_gaps_beyond_the_evaluators_tie_band():
    stat = summarise(layered_rows(lead=0.01))[0]['selection']['hand_to_body']
    assert stat['pairs'] == 0 and stat['wellness_preference']['favours_toucher'] == 0
    assert stat['verdict'] == ('not measured: predicted wellness never favoured a touching command over another command '
                               "by more than the evaluator's tie band (1.5% of the 0-to-5 wellness scale; the largest "
                               'gap in either direction was 0.40% of it)')
    assert stat['wellness_gap_abs']['max'] == pytest.approx(0.02)
    assert stat['unfiltered']['value'] == 1.0 and stat['unfiltered']['pairs'] == 48


def test_a_share_of_the_wellness_scale_never_prints_a_bare_or_scientific_decimal():
    assert [PROBE.scale_percent(v) for v in (0.0734, 0.0014, 0.0039, 1.0, 5.0, 1e-6, 0.0)] == [
        '1.5%', '0.028%', '0.078%', '20%', '100%', '0.000020%', '0%']


def test_selection_leaves_out_finger_curls_unknown_requests_and_finger_only_differences():
    rows = layered_rows()
    for r in rows:
        r['families'][2] = r['families'][3] = 'curl'                 # the second toucher and its match are curls
        r['objective'][0] = None                                     # the first toucher's request is not known
    stat = summarise(rows)[0]['selection']['hand_to_body']
    assert stat['eligible_starts'] == 0
    assert stat['verdict'] == 'not measured: every touching command was a finger curl or had a request that is not known'
    rows = layered_rows()
    for r in rows:
        r['finger_only_pairs'] = [[0, 1], [2, 3]]
    stat = summarise(rows)[0]['selection']['hand_to_body']
    assert stat['pairs'] == 0 and stat['touchers_without_a_matched_command']['no_same_kind_command_that_touched_nothing'] == 48


def test_selection_uses_one_shared_finger_goal_where_the_hand_stage_ran():
    """The second review: the hand stage re-chooses the fingers after the planner's search, so where it ran
    every command is scored with the controller's finger goal. Here only the fingers made the touching
    commands look worse to the planner."""
    executed = layered_rows(toucher_agrees=False, matched_agrees=True)
    shared = layered_rows()
    for r, s in zip(executed, shared):
        r['hand_stage'] = 'completed'
        r['wellness_shared_fingers'], r['objective_shared_fingers'] = s['wellness'], s['objective']
    stat = summarise(executed)[0]['selection']['hand_to_body']
    assert stat['shared_finger_goal_starts'] == 24 and stat['value'] == 1.0 and stat['verdict'] == PROBE.ABOVE
    for r in executed:
        r['hand_stage'] = 'no_valid_memory_goal'
    stat = summarise(executed)[0]['selection']['hand_to_body']
    assert stat['shared_finger_goal_starts'] == 0 and stat['value'] == 0.0 and stat['verdict'] == PROBE.NOT_ABOVE


def test_the_sense_parts_say_which_part_of_the_objective_preferred_the_toucher():
    rows = layered_rows()
    for r in rows:
        r['goal_senses'] = ['touch', 'proprioception']
        r['goal_distance_by_sense'] = [[.1, .4], [.4, .4], [.1, .4], [.4, .4], [.3, .3], [.3, .3]]
    stat = summarise(rows)[0]['selection']['hand_to_body']
    assert stat['sense_parts'] == dict(proprioception=dict(toucher_lower=0, matched_lower=0, equal=48),
                                       touch=dict(toucher_lower=48, matched_lower=0, equal=0))


def test_selection_leaves_out_starts_whose_planner_route_was_not_reproduced():
    layers, passed = summarise(layered_rows(verified=False))
    assert layers['prediction']['verdict'] == PROBE.ABOVE
    stat = layers['selection']['hand_to_body']
    assert stat['verdict'] == 'not measured: no start that offered a pair had its planner route reproduced'
    assert stat['unverified_starts_left_out'] == 24 and passed is False
    assert layers['selection']['any_touch']['verdict'] == stat['verdict']


def test_the_planner_effort_is_taken_on_the_request_as_its_action_space_holds_it():
    assert PROBE.planner_effort(None, .01) is None
    expected = .01 * (0.25 + 1e-8 + (1 - 1e-4) ** 2)
    assert PROBE.planner_effort([0.5, 0.0, 1.2], .01) == pytest.approx(expected, rel=1e-12)


def test_pairs_whose_orders_differ_only_in_the_fingers_are_found():
    base = np.zeros((1, 4, 9))
    fingers_moved = base.copy()
    fingers_moved[0, 2:, 0] = 3.0
    wrist_moved = base.copy()
    wrist_moved[0, 0, 0] = 1e-3
    assert PROBE.finger_only_pairs([base, fingers_moved, wrist_moved], [2, 3]) == [[0, 1]]


# ---------------------------------------------------------------------------------------- commands
def test_targets_are_each_hands_nearest_eligible_patches():
    gaps = {'L': [5.0, 1.0, np.inf, 2.0], 'R': [0.5, 3.0, 3.0, 9.0]}
    eligible = {'L': [True, False, True, True], 'R': [False, True, True, True]}
    assert PROBE.choose_targets(gaps, eligible, 2) == [('L', 3, 2.0), ('L', 0, 5.0), ('R', 1, 3.0), ('R', 2, 3.0)]


def test_aim_goes_past_the_nearest_point_and_has_a_fallback_direction():
    aim = PROBE.aim_vector_cm([0, 0, 0], [0.03, 0, 0])
    np.testing.assert_allclose(aim, [3.0 + PROBE.AIM_PAST_SURFACE_CM, 0, 0])
    np.testing.assert_allclose(PROBE.aim_vector_cm([0, 0, 0], [0, 0, 0], [0, 2, 0]), [0, PROBE.AIM_PAST_SURFACE_CM, 0])


def _reach_setup():
    H = 5
    raw = np.zeros((1, H, 9), np.float32)
    raw[..., 6] = 1
    raw[..., 8] = .2
    achieved = np.zeros((1, H, 10))
    achieved[0, 2, :3] = np.array([10.0, 20.0, 30.0]) / 200
    achieved[0, 2, 3:7] = [0, 0, 2, 0]
    previous = np.zeros((1, H, 9))
    previous[0, 2, :3] = [12.0, 18.0, 31.0]
    return raw, achieved, previous


def test_reach_request_puts_the_wrist_or_its_smoothed_target_onto_the_aim():
    raw, achieved, previous = _reach_setup()
    aim = np.array([1.0, -2.0, 0.5])
    onto = PROBE.reach_request(raw, 2, [2, 3, 4], achieved, aim)
    np.testing.assert_allclose(onto[0, 2, :3] * 200, [11.0, 18.0, 30.5], atol=1e-4)
    np.testing.assert_allclose(onto[0, 2, 3:7], [0, 0, 1, 0])
    np.testing.assert_array_equal(onto[0, :, 8], raw[0, :, 8])
    np.testing.assert_array_equal(np.delete(onto, 2, axis=1), np.delete(raw, 2, axis=1))
    a = .35
    push = PROBE.reach_request(raw, 2, [2, 3, 4], achieved, aim, smooth_prev=previous, smoothing=a, strength=.7)
    smoothed = a * push[0, 2, :3].astype(np.float64) * 200 + (1 - a) * previous[0, 2, :3]
    np.testing.assert_allclose(smoothed, [11.0, 18.0, 30.5], atol=1e-3)
    np.testing.assert_allclose(push[0, [2, 3, 4], 8], .7)
    np.testing.assert_allclose(push[0, [0, 1], 8], .2)


def test_a_compensated_request_beyond_the_reach_clamp_is_flagged():
    raw, achieved, previous = _reach_setup()
    near = PROBE.reach_request(raw, 2, [2, 3, 4], achieved, np.array([4.0, 0, 0]), smooth_prev=previous, smoothing=.35)
    edge = PROBE.reach_request(raw, 2, [2, 3, 4], achieved, np.array([15.0, 0, 0]), smooth_prev=previous, smoothing=.35)
    far = PROBE.reach_request(raw, 2, [2, 3, 4], achieved, np.array([20.0, 0, 0]), smooth_prev=previous, smoothing=.35)
    assert not PROBE.request_clipped(near, achieved, 2, 1.0)
    assert not PROBE.request_clipped(edge, achieved, 2, 1.0)  # (10 + 15 - 0.65 x 12) / 0.35 - 10 = 39.1 cm of 40
    assert PROBE.request_clipped(far, achieved, 2, 1.0)       # (10 + 20 - 0.65 x 12) / 0.35 - 10 = 53.4 cm
    assert PROBE.request_clipped(near, achieved, 2, 0.05)     # a 2 cm reach at a neonate's range of motion
    world = raw.copy()
    world[0, 2, 2] = 1.2
    assert PROBE.request_clipped(world, achieved, 2, 10.0)


def test_executed_wrist_targets_strengths_and_share_of_the_aim_are_read_from_the_order():
    names = ['wrist_L', 'lowerarm_L', 'upperarm_L', 'wrist_R', 'lowerarm_R', 'upperarm_R']
    achieved = np.zeros((1, 6, 10))
    achieved[0, 3, :3] = np.array([10.0, 0.0, 0.0]) / 200
    order = np.zeros((1, 6, 9))
    order[0, 3, :3] = [10.0 + 0.35 * 4.0, 0.0, 0.0]
    order[0, 3:6, 8] = [.5, .4, .3]
    summary = PROBE.executed_summary(order, achieved, names)
    np.testing.assert_allclose(summary['R']['wrist_shift_cm'], [1.4, 0, 0])
    assert summary['R']['arm_strength'] == pytest.approx([.5, .4, .3])
    assert PROBE.share_of_aim(order, achieved, 3, [4.0, 0, 0]) == pytest.approx(0.35)


def test_curl_follows_the_witness_recipe_on_one_hand_only():
    geometry = dict(bodies=[dict(side='L', joints=[0, 1, 2]), dict(side='R', joints=[3])],
                    joints=[dict(name='l_ff_knuckle', limits=[-1, 2]), dict(name='l_th_base', limits=[-1, 2]),
                            dict(name='l_ff_distal', limits=[0, 1]), dict(name='r_ff_knuckle', limits=[-1, 2])])
    out = PROBE.curl_joints(np.zeros((1, 4)), geometry, 'L')
    np.testing.assert_allclose(out, [[.95 * 2, 0, .95 * 1, 0]])


def _contract_model(calls):
    import torch

    class Kinematics:
        def project_world_orders(self, base, position_scale):
            calls.append('project')
            return base, np.array([[base[0, 0, 0] ** 2]], np.float32), {}

        def assemble_world(self, base, joints, position_scale):
            calls.append('assemble')
            out = base.clone()
            out[:, 1, 0] = base[:, 0, 0] + joints[:, 0]
            return out

    def smooth_joints(joints, previous, alpha):
        calls.append('joint_smoothing')
        return alpha * joints + (1 - alpha) * previous

    def smooth_body(base, previous, alpha, keep_row):
        calls.append('body_smoothing')
        return alpha * base + (1 - alpha) * previous
    model = SimpleNamespace(hand_kinematics=Kinematics(), execution_transform=lambda raw, *a, **kw: raw.clone(),
                            _smooth_hand_joints=smooth_joints, smooth_blend=smooth_body)
    raw = torch.zeros(1, 2, 9)
    raw[..., 6] = 1
    raw[..., 0] = 2
    previous = raw.clone()
    previous[..., 0] = 6
    decision = dict(achieved=raw, dev={}, hand_previous_joints=torch.tensor([[2.]]), smooth_prev=previous,
                    smoothing=.25, smooth_keep_gaze=None)
    return model, raw, decision


def test_new_hand_contract_projects_before_smoothing_and_retains_joint_goal():
    calls = []
    model, raw, decision = _contract_model(calls)
    order, joints = PROBE.new_contract_candidate(model, raw, decision)
    assert calls == ['project', 'joint_smoothing', 'body_smoothing', 'assemble']
    np.testing.assert_array_equal(joints, [[2.5]])
    np.testing.assert_array_equal(order[0, :, 0], [5., 7.5])
    meta = dict(canonical_motor_targets=order[0].copy(), hand_joint_goal=joints[0].copy())
    PROBE.verify_hand_readback(meta, order, joints)
    meta['hand_joint_goal'][0] += .01
    with pytest.raises(AssertionError):
        PROBE.verify_hand_readback(meta, order, joints)


def test_an_explicit_joint_goal_replaces_the_projection_and_the_request_before_the_smoother_is_returned():
    calls = []
    model, raw, decision = _contract_model(calls)
    order, joints, base = PROBE.contract_parts(model, raw, decision, raw_joint_goal=np.array([[10.]]))
    assert calls == ['joint_smoothing', 'body_smoothing', 'assemble']
    np.testing.assert_array_equal(joints, [[4.]])            # .25 x 10 + .75 x 2
    np.testing.assert_array_equal(order[0, :, 0], [5., 9.])
    np.testing.assert_array_equal(base[0, :, 0], [2., 2.])    # the request as it enters the smoother


# ---------------------------------------------------------------------------------------- physics
class HeldSession:
    """Five environment steps per model step; the patch state turns on at env step 7."""
    reps = 5

    def __init__(self, joints):
        self.joints, self.frames, self.calls, self.step = joints, [], [], 0
        self.env = SimpleNamespace(data=SimpleNamespace(time=0.0, qpos=np.zeros(2), qvel=np.zeros(2)))

    def restore(self, snapshot):
        assert snapshot == 'saved'
        self.step, self.env.data.time = 0, 0.0

    def act(self, command, **kwargs):
        self.calls.append(kwargs['new_order'])
        np.testing.assert_array_equal(kwargs['hand_joint_goal'], self.joints)
        for rep in range(self.reps):
            self.step += 1
            self.env.data.time = self.step * .01
            meta = dict(canonical_motor_targets=command[0].copy(), hand_joint_goal=self.joints[0].copy(),
                        task=dict(contact_patch_state=[0, 2 if self.step >= 7 else 0], self_contact_onsets=[]))
            self.frames = [SimpleNamespace(meta=meta)]
            kwargs['on_env_step'](self, rep)


def test_a_branch_holds_its_order_for_the_tenure_and_reads_the_target_gaps_at_50_and_250_ms(monkeypatch):
    order = np.zeros((1, 2, 9), np.float32)
    order[..., 6] = 1
    joints = np.array([[.2, .4]], np.float32)
    monkeypatch.setattr(PROBE, 'gap_cm', lambda env, pairs: round(10 - 10 * env.data.time, 6))
    session = HeldSession(joints)
    outcome = PROBE.physical_trace(session, 'saved', order, [0, 0], joints, model_steps=5, gap_pairs={'L0': [[1, 2]]})
    assert session.calls == [True, False, False, False, False]
    assert len(outcome['trace']) == 25 and outcome['any_self'] == [False, True]
    assert [f['env_step'] for f in outcome['trace'] if 'gaps_cm' in f] == [5, 25]
    compact = PROBE.compact_outcome('c', outcome, 5, [True, True], ['a', 'b'])
    assert compact['contact_50ms'] == [] and compact['contact_tenure'] == [1]
    assert compact['touch_tenure'] == [] and compact['touch_hand_tenure'] == []    # no onset reported
    assert compact['onset_regions_tenure'] == [] and compact['onset_hand_regions_tenure'] == []
    assert compact['gaps_cm'] == {'50ms': {'L0': 9.5}, 'tenure': {'L0': 7.5}}
    assert compact['own_contact_by_moment'] == [[], [1], [1], [1], [1]]
    assert compact['trace_sha256'] == PROBE.trace_digest(outcome['trace'])


# ---------------------------------------------------------------------------------------- the scheduler
def _layered(rows=None, sha='sha', **extra):
    declaration = dict(probe=PROBE.PROBE_VERSION, endpoint_version=PROBE.ENDPOINT_VERSION, checkpoint='ckpt',
                       script_sha256=sha, bodies=8, starts_per_body=3, tie_band=BAND, learning=False)
    rows = rows if rows is not None else [make_row(b, [[], []], start=s) for b in range(8) for s in range(3)]
    return json.loads(json.dumps(PROBE.result_from(declaration, rows, weights_unchanged=True,
                                                   all_candidate_traces_replayed_exactly=True, elapsed_s=1.0, **extra),
                                 allow_nan=False))


def _passing():
    return _layered(layered_rows())


def _record(sha='sha'):
    return dict(checkpoint='ckpt', script_sha256=sha, command=['python', '-u', SCRIPT])


def _mutations():
    def pred(p):
        return p['layers']['prediction']['hand_to_body']

    def sel(p):
        return p['layers']['selection']['hand_to_body']
    return [
        (lambda p: p.update(passed=False), 'declared rule'),
        (lambda p: p.update(evaluated_starts=23), 'start count'),
        (lambda p: p.update(weights_unchanged=False), 'state verification'),
        (lambda p: p.update(tie_band=None), 'tie band'),
        (lambda p: p.update(script_sha256='other'), 'script identity'),
        (lambda p: p.update(probe='self_touch_layers_v3'), 'not one this validator knows'),
        (lambda p: p['layers']['reachability'].update(starts_offering_a_pair_50ms=25), 'reachability'),
        (lambda p: p['layers']['reachability'].update(reachable_bodies_hand_to_body_50ms=9), 'reachability'),
        (lambda p: p['layers']['reachability'].update(reachable_starts_hand_to_body_tenure=25), 'reachability'),
        (lambda p: pred(p)['all_pairs'].update(interval=None), 'prediction'),
        (lambda p: pred(p)['all_pairs'].update(bodies=7), 'prediction'),
        (lambda p: pred(p)['all_pairs'].update(judged=True), 'prediction'),
        (lambda p: pred(p)['whether']['votes'].update(p_value=.5), 'prediction'),
        (lambda p: pred(p)['whether']['votes'].update(bodies_voting=9), 'prediction'),
        (lambda p: pred(p)['whether']['votes']['voters'].pop('3'), 'prediction'),
        (lambda p: pred(p)['whether']['votes']['voters'].update({'3': -.5}), 'prediction'),
        (lambda p: pred(p)['whether']['per_body_events'].update({'0': 1}), 'prediction'),
        (lambda p: pred(p)['whether']['per_body_pairs'].update({'0': 3}), 'prediction'),
        (lambda p: pred(p)['whether'].update(verdict='not above its control'), 'prediction'),
        (lambda p: pred(p)['whether'].update(distance_to_goal_line=.3), 'prediction'),
        (lambda p: pred(p)['whether'].update(pairs=10 ** 4), 'prediction'),
        (lambda p: pred(p)['whether'].update(value=.3, interval=[.2, .4], distance_to_goal_line=.5), 'prediction'),
        (lambda p: pred(p)['where'].update(value=1.5), 'prediction'),
        (lambda p: pred(p)['where'].update(unresolved_crossed_pairs=-1), 'prediction'),
        (lambda p: pred(p)['where'].update(touch_region_pairs={'left hand|trunk': -1}), 'prediction'),
        (lambda p: pred(p).update(verdict='not decided: anything'), 'prediction'),
        (lambda p: p['layers']['prediction'].update(verdict='not above its control'), 'prediction'),
        (lambda p: p['layers']['prediction']['routes']['order_input_only']['all_pairs'].update(starts=3), 'prediction'),
        (lambda p: sel(p).update(value=.3, interval=[.2, .4], distance_to_goal_line=.5), 'selection'),
        (lambda p: sel(p).update(chance=.4), 'selection'),
        (lambda p: sel(p).update(tie_band=.01), 'selection'),
        (lambda p: sel(p)['per_body'].update({'0': .5}), 'selection'),
        (lambda p: sel(p)['move_relation'].update(smaller=1), 'selection'),
        (lambda p: sel(p)['unfiltered'].update(judged=True), 'selection'),
        (lambda p: sel(p)['wellness_gap_abs'].update(median=10.0), 'selection'),
        (lambda p: sel(p)['votes'].update(min_events_per_body=1), 'selection'),
        (lambda p: sel(p).update(verdict='not measured: anything'), 'selection'),
        (lambda p: p['layers']['selection'].update(verdict='not measured: anything'), 'selection')]


def test_the_validator_accepts_layered_results_and_rejects_broken_ones():
    from mimo_bridge.tools.ability_rechecks import validate_result
    validate_result(_layered(), _record())
    reached = _passing()
    assert reached['passed'] is True
    validate_result(reached, _record())
    for mutate, reason in _mutations():
        bad = copy.deepcopy(reached)
        mutate(bad)
        with pytest.raises(ValueError, match=reason):
            validate_result(bad, _record())


def test_the_validator_accepts_every_undecided_and_unmeasured_form_the_probe_writes():
    from mimo_bridge.tools.ability_rechecks import validate_result
    quiet = [make_row(b, [[], []], start=s) for b in range(8) for s in (1, 2)]
    for rows in (layered_rows(hand=False), layered_rows(verified=False), layered_rows(lead=0.01),
                 layered_rows(matched_agrees=True), layered_rows(knows=False), layered_rows(second=False),
                 wrong_in(layered_rows(), {5, 6, 7}),
                 layered_rows(families=['exploration', 'curl', 'reach_push', 'curl', 'hold', 'hold']),
                 [r for r in layered_rows() if r['body'] < 4] + [make_row(b, [[], []], start=s) for b in (4, 5, 6, 7)
                                                                  for s in range(3)],
                 layered_rows(starts=1) + quiet):
        payload = _layered(rows)
        validate_result(payload, _record())
        assert payload['passed'] is False


def test_a_value_outside_its_interval_is_refused():
    from mimo_bridge.tools.ability_rechecks import validate_result
    bad = _passing()
    bad['layers']['prediction']['hand_to_body']['whether']['interval'] = [0.2, 0.3]
    with pytest.raises(ValueError, match='prediction'):
        validate_result(bad, _record())


def test_the_validator_keeps_judging_the_earlier_form():
    from mimo_bridge.tools.ability_rechecks import validate_result
    row = dict(bodies=8, trials=16, selected_success_fraction=0., comparison_success_fraction=0.,
               difference_percentage_points_ci95=[0., 0.], improvement_above_uncertainty=False)
    earlier = dict(checkpoint='ckpt', script_sha256='sha', learning=False, elapsed_s=1., bodies=8,
                   evaluated_held_out_trials=16, weights_unchanged=True, all_candidate_traces_replayed_exactly=True,
                   passed_private_comparison=False,
                   comparisons={n: dict(row) for n in ('controller', 'hold', 'mean_exploration')})
    validate_result(earlier, _record())
    with pytest.raises(ValueError, match='trial count'):
        validate_result(dict(earlier, evaluated_held_out_trials=15), _record())


def test_a_layered_result_completes_under_the_scheduler_and_an_earlier_layered_version_fails_safely(tmp_path):
    from mimo_bridge.tools import ability_rechecks as worker
    for payload, status in ((_layered(), 'complete'), (dict(_layered(), probe='self_touch_layers_v3'), 'failed')):
        payload['checkpoint'] = str(tmp_path / 'ckpt')
        output = tmp_path / ('attempt_' + status)
        output.mkdir()
        (output / 'result.json').write_text(json.dumps(payload), encoding='utf-8')
        record = dict(output=str(output), checkpoint=payload['checkpoint'], script_sha256='sha',
                      command=['python', '-u', SCRIPT])
        outcome = worker.finished_result(record)
        assert outcome['status'] == status
        if status == 'complete':
            assert outcome['outcome'] is False


class FakeProcess:
    def __init__(self, parents, created=0.0):
        self._parents, self._created, self.pid = parents, created, 7

    def parents(self):
        return self._parents

    def cmdline(self):
        return ['python', '-m', 'mimo_bridge.tools.ability_rechecks', '--once']

    def create_time(self):
        return self._created


def test_a_scheduled_run_refuses_at_once_when_its_judge_cannot_accept_the_layered_result(monkeypatch):
    from mimo_bridge.tools import ability_rechecks as worker
    assert PROBE.scheduler_validator_problem(FakeProcess([])) is None       # launched by hand: nothing judges it
    stale = FakeProcess([FakeProcess([], created=0.0)])
    assert worker.STALE_REFUSAL in PROBE.scheduler_validator_problem(stale)
    fresh = FakeProcess([FakeProcess([], created=4e9)])
    assert PROBE.scheduler_validator_problem(fresh, 'sha') is None           # the validator on disk accepts it

    def refuse(payload, record):
        raise ValueError('Self-touch start count is incomplete')
    monkeypatch.setattr(worker, 'validate_result', refuse)
    assert 'does not accept the layered result' in PROBE.scheduler_validator_problem(fresh, 'sha')


def test_the_example_result_is_accepted_and_carries_the_measured_tie_band():
    from mimo_bridge.causal_chain import TIE_BAND
    from mimo_bridge.tools.ability_rechecks import validate_result
    example = PROBE.example_result('sha', 'ckpt')
    assert example['tie_band'] == TIE_BAND and example['layers']['selection']['tie_band'] == TIE_BAND
    validate_result(json.loads(json.dumps(example, allow_nan=False)), _record())


# ---------------------------------------------------------------------------------------- the dashboard
def plain(text):
    from mimo_bridge.tools.check_no_bare_decimals import bare_decimals, doubled_sentences, long_numbers, unanchored_numbers
    assert not bare_decimals(text) and not unanchored_numbers(text) and not long_numbers(text), text
    assert not doubled_sentences(text), text


def test_the_dashboard_reads_a_round_where_no_hand_could_touch_another_body_area():
    from mimo_bridge.tools import development_dashboard as D
    said = D._touch_words(_layered(layered_rows(hand=False), tenure_steps=5))
    assert said['reachable'] == 0 and said['passed'] is False and said['form'] == 'layers'
    assert said['headline'].startswith('No command made a hand touch another body area')
    assert 'held for a whole 250 ms command, none did either.' in said['headline']
    assert ('Another new touch, such as a finger on its own hand or an arm on the trunk, happened within 50 ms in 24 '
            'of those model steps.') in said['headline']
    assert said['lead'].startswith('no command could make a hand touch another body area')
    assert 'held for a whole command' in D._touch_words(_layered(layered_rows(hand=False)))['headline']
    for text in (said['headline'], said['detail'], said['lead']):
        plain(text)


def test_the_dashboard_reads_a_passing_round_with_its_judged_numbers_and_goal_line():
    from mimo_bridge.tools import development_dashboard as D
    said = D._touch_words(_passing())
    assert said['passed'] is True and said['reachable'] == 24
    assert ('gave the touching command the higher chance of that touch in 100 of 100 pairs in which it asked no body '
            'part for a bigger move, turn or strength than the other command (8 of 8 bodies with enough pairs were '
            'above a coin flip)') in said['headline']
    assert ('when two commands touched different patches, gave each the higher chance on its own patch in 100 of 100 '
            '(8 of 8 bodies') in said['headline']
    assert 'Foresight verdict: above a coin flip on both, by the declared rule over bodies.' in said['headline']
    assert ('the planner\'s objective preferred it to a command of the same kind from the same model step that touched '
            'nothing and that the evaluator scored the same, in 100 of 100 such pairs (50 = a coin flip, goal line '
            '80; the touching command asked the smaller move in 0 of the 48 pairs and the bigger one in 0)') in said['headline']
    assert '100 of 100 pairs in which it asked no bigger move' in said['lead'] and 'goal line 80' in said['lead']
    assert 'picked' not in said['lead'] and 'foresaw' not in said['lead']
    for text in (said['headline'], said['detail'], said['lead']):
        plain(text)


def test_the_dashboard_says_what_was_not_scored_once_and_leads_with_the_judged_numbers():
    from mimo_bridge.tools import development_dashboard as D
    said = D._touch_words(_layered(layered_rows(lead=0.01)))
    assert said['headline'].count("The planner's choice was not scored: predicted wellness never favoured") == 1
    assert 'not measured (not measured' not in said['headline'] and 'not measured: not measured' not in said['headline']
    assert said['lead'] == ('self-touch foresight and choice did not both pass: the contact head gave the touching '
                            'command the higher chance in 100 of 100 pairs in which it asked no bigger move and gave '
                            'each of two touching commands the higher chance on its own patch in 100 of 100 (50 = a '
                            "coin flip, goal line 80); the planner's choice was not measured")
    for text in (said['headline'], said['lead']):
        plain(text)


def test_the_dashboard_leads_with_the_judged_score_not_the_score_over_every_pair():
    """The second review: in the move-size counterexample the lead read 60 of 100 over every pair while the
    judged score was 0."""
    from mimo_bridge.tools import development_dashboard as D
    bias = [-2.0, -1.5, -1.0]
    said = D._touch_words(_layered(_reach_rows(_head(bias, .2, 5.0), _head(bias, .2, 6.0), _head(bias, .2, .1))))
    assert 'in 0 of 100 pairs in which it asked no bigger move' in said['lead']
    assert '60 of 100' not in said['lead'] and '60 of 100' not in said['headline']
    assert 'had no two commands that touched different patches and that it told apart' in said['headline']
    plain(said['headline'])


def test_a_score_no_body_could_vote_on_says_so():
    from mimo_bridge.tools import development_dashboard as D
    said = D._touch_words(_layered(layered_rows(starts=1, second=False) +
                                   [make_row(b, [[], []], start=s) for b in range(8) for s in (1, 2)]))
    assert 'no body had enough pairs to vote' in said['headline'] and '0 of 0 bodies' not in said['headline']
    plain(said['headline'])


def test_the_page_shows_the_layered_rounds_own_lead_and_counts_each_form_apart(tmp_path):
    from mimo_bridge.tools import development_dashboard as D
    results = tmp_path / 'Saved/mimo_bridge/ability_rechecks'
    earlier = dict(evaluated_held_out_trials=16, selected_touches=0, selected_patch_physically_reachable_trials=0,
                   passed_private_comparison=False)
    history = []
    for i, payload in enumerate((earlier, _layered(layered_rows(hand=False)))):
        path = results / f'chosen_self_touch_{400000 + 3000 * i}/result.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload), encoding='utf-8')
        history.append(dict(status='complete', attempt_step=400000 + 3000 * i, started_at=1000 + i, completed_at=1060 + i,
                            result=str(path)))
    (tmp_path / D.ABILITY_STATUS).write_text(json.dumps(dict(tasks={'chosen_self_touch': dict(
        title='Self-touch', status='complete', due_step=406000, history=history)})), encoding='utf-8')
    out = D.ability_results(tmp_path)
    assert out['lead'] == ('In the newest rounds, no command could make a hand touch another body area within 50 ms from '
                           'the saved model steps, so self-touch foresight and choice were not tested.')
    assert out['rows'][0]['trend'] == (
        'The earlier single-choice comparison passed in 0 of 1 completed rounds since step 400,000. In 1 of them no '
        'command touched the skin being compared. The three-layer self-touch test passed in 0 of 1 completed rounds. '
        'In 1 of them no command made a hand touch another body area within 50 ms.')
    plain(out['rows'][0]['trend'])


def test_the_dashboard_keeps_the_earlier_forms_words_and_leaves_other_layered_versions_unread():
    from mimo_bridge.tools import development_dashboard as D
    earlier = dict(evaluated_held_out_trials=16, selected_touches=0, selected_patch_physically_reachable_trials=0,
                   passed_private_comparison=False)
    assert 'No command touched the chosen skin patches' in D._touch_words(earlier)['headline']
    assert D._trend('chosen_self_touch', [dict(D._touch_words(earlier))], '') == (
        'Passed its declared comparison in 0 of 1 completed rounds. In 1 of them no command touched the skin being '
        'compared.')
    with pytest.raises(ValueError):
        D._touch_words(dict(_layered(), probe='self_touch_layers_v3'))
