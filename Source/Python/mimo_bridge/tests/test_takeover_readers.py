# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The new executed-order contract must not be read as the old direction cosine."""
from mimo_bridge.tools import lineage_series as ls
from mimo_bridge.tools.build_explainer import _loss_share


def test_bootstrap_counts_are_visible_without_claiming_synergy_control():
    record = dict(synergy=dict(following_by_author=dict(planner=dict(
        part_orders=200, closer=110, chance_count=60, above_chance_per_100=25,
        moved_deg_median=12, ordered_deg_median=16))))
    args = dict(motor_synergy="on", synergy_planner="off", synergy_babble="off")
    row = next(r for r in ls.gain_of_function(record, args, {})['items'] if r['item'] == 5)
    assert not row['measured']
    text = ls.item_text(row)
    assert '110 of 200' in text and '25 more per 100' in text
    # the headline's words since a6b3f7f1c (the owner revision of 2026-09-23): learning, not directing
    assert 'not directing it' in row['big']
    assert ls.SERIES['syn.executed.planner']['get'](record) == 25
    assert ls.SERIES['syn.following']['get'](record) is None
    args['synergy_planner'] = 'on'
    row = next(r for r in ls.gain_of_function(record, args, {})['items'] if r['item'] == 5)
    assert 'comparison pending' in row['big']
    assert not row['measured'], 'one control-on window is not a paired comparison'


def test_an_args_file_without_synergy_babble_reads_as_the_trainers_default_off():
    """--synergy-babble defaults to off in the trainer (train_mimo.parse_args, build_learner), so an
    args file without the key must not make item 5 say the synergy's exploration orders were used."""
    record = dict(synergy=dict(following_by_author=dict(reflex_noise=dict(
        part_orders=200, closer=110, chance_count=60, above_chance_per_100=25))))
    args = dict(motor_synergy="on", synergy_planner="off")
    row = next(r for r in ls.gain_of_function(record, args, {})['items'] if r['item'] == 5)
    assert 'not directing it' in row['big']
    args['synergy_babble'] = 'on'
    row = next(r for r in ls.gain_of_function(record, args, {})['items'] if r['item'] == 5)
    assert 'comparison pending' in row['big']


def _item5(record, args):
    return next(r for r in ls.gain_of_function(record, args, {})['items'] if r['item'] == 5)


def _directing(row):
    """True when item 5's headline says orders from the synergy were used, False when it says they were
    not. A headline that makes both claims, or neither, fails here."""
    pending, not_directing = 'comparison pending' in row['big'], 'not directing' in row['big']
    assert pending != not_directing, row['big']
    return pending


def _counted(*authors):
    """following_by_author with executed orders counted for each named author."""
    return {a: dict(part_orders=200, closer=110, chance_count=60, above_chance_per_100=25) for a in authors}


def test_item5_headlines_exclude_each_other_and_a_missing_planner_key_reads_as_on():
    """prod_tests_review_2026-09-25.md finding 4: the substring checks above passed one headline making both
    claims whatever the flags, and a missing synergy_planner read as off where the trainer's default is on
    (train_mimo.parse_args, build_learner). This record carries no plans or babble_steps counts, so the
    item reads the launch flags."""
    record = dict(synergy=dict(following_by_author=_counted('planner')))
    assert not _directing(_item5(record, dict(motor_synergy='on', synergy_planner='off', synergy_babble='off')))
    assert _directing(_item5(record, dict(motor_synergy='on', synergy_planner='on', synergy_babble='off')))
    assert _directing(_item5(record, dict(motor_synergy='on', synergy_babble='off')))


def test_item5_reads_a_missing_flag_as_the_trainers_own_default():
    """Ties the item's reading of a missing synergy_planner or synergy_babble to train_mimo.parse_args
    itself, not to a copy of its defaults: a change of the trainer's default fails here
    (prod_tests_review_2026-09-25.md finding 4)."""
    from mimo_bridge import train_mimo
    default = train_mimo.parse_args(['--out', 'unused'])
    record = dict(synergy=dict(following_by_author=_counted('planner', 'reflex_noise')))
    for key in ('synergy_planner', 'synergy_babble'):
        full = dict(motor_synergy='on', synergy_planner='off', synergy_babble='off')
        full[key] = getattr(default, key)
        partial = dict(full)
        del partial[key]
        assert _item5(record, partial)['big'] == _item5(record, full)['big'], key


def test_item5_decides_from_the_records_own_counts_not_the_launch_flags():
    """Every synergy record counts the planner decisions made through the synergy (plans) and the moments of
    exploration drawn in it (babble_steps) (lem_model.synergy_window_summary). The item reads those counts
    beside the executed orders counted by author; the flags only say which outputs were allowed
    (prod_tests_review_2026-09-25.md finding 5; the style guide: a flag state is never the evidence that an
    effect exists)."""
    def record(plans, babble_steps, *authors):
        return dict(synergy=dict(plans=plans, babble_steps=babble_steps, following_by_author=_counted(*authors)))
    allowed = dict(motor_synergy='on', synergy_planner='on', synergy_babble='on')
    refused = dict(motor_synergy='on', synergy_planner='off', synergy_babble='off')
    # both outputs allowed, neither used in this window
    assert not _directing(_item5(record(0, 0, 'planner', 'reflex_noise'), allowed))
    # used and executed, whatever the flags say, and with no flag in the args at all
    for args in (allowed, refused, dict(motor_synergy='on')):
        assert _directing(_item5(record(12, 0, 'planner'), args))
        assert _directing(_item5(record(0, 40, 'reflex_noise'), args))
    # formed but never executed: plans with no planner order counted, exploration drawn in the synergy with
    # no exploration order counted
    assert not _directing(_item5(record(12, 0, 'reflex_noise'), allowed))
    assert not _directing(_item5(record(0, 40, 'planner'), allowed))
    # the card prints the counts that decided it, and says when a record has none
    text = ls.item_text(_item5(record(12, 40, 'planner'), refused))
    assert 'planned through the synergy 12 times' in text and 'at 40 model steps' in text
    assert 'launch settings decide' not in text
    assert 'launch settings decide' in ls.item_text(_item5(dict(synergy=dict(
        following_by_author=_counted('planner'))), refused))


def test_item5_falls_back_on_one_flag_only_for_the_count_a_record_lacks():
    """A record without one count reads that output's launch flag (at the trainer's default when the key is
    missing too) and still reads the other output from its count."""
    no_babble_count = dict(synergy=dict(plans=0, following_by_author=_counted('planner', 'reflex_noise')))
    assert _directing(_item5(no_babble_count, dict(motor_synergy='on', synergy_planner='on', synergy_babble='on')))
    assert not _directing(_item5(no_babble_count, dict(motor_synergy='on', synergy_planner='on')))
    no_plans_count = dict(synergy=dict(babble_steps=0, following_by_author=_counted('planner', 'reflex_noise')))
    assert _directing(_item5(no_plans_count, dict(motor_synergy='on', synergy_babble='on')))
    assert not _directing(_item5(no_plans_count, dict(motor_synergy='on', synergy_planner='off',
                                                      synergy_babble='on')))


def test_independent_synergy_lesson_does_not_dilute_shared_loss_shares():
    record = dict(loss_breakdown=dict(next_frame=2, imagination=1, motor=1, synergy=100, self=-8))
    assert _loss_share('next_frame')(record) == .5


def test_probe_commit_measurement_does_not_substitute_resident_memory(monkeypatch):
    from types import SimpleNamespace
    import psutil
    from mimo_bridge.tools import gof_probe_suite as suite
    memory = SimpleNamespace(peak_pagefile=8_240_000_000, peak_wset=4_880_000_000)
    monkeypatch.setattr(psutil, 'Process', lambda: SimpleNamespace(memory_info=lambda: memory))
    assert suite._peak_commit_gb() == 8.24
    del memory.peak_pagefile
    assert suite._peak_commit_gb() is None
