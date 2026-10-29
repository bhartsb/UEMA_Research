# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The defect register (tools/defect_register.py): ages, last meaningful work, the 12-hour and 24-hour rules, the
runner's gate and the checks of each typed state.

The operator's recovery directive of Oct 5, 2026, sections 4 and 5. Every register here is a temporary file with a
fixed clock; a branch's newest commit is given by a small function in place of git. The last test reads this
checkout's own register.
"""
import json
from pathlib import Path

import pytest

from mimo_bridge.tools import defect_register as dr
from mimo_bridge.tools import test_registry as tr
from mimo_bridge.tools import test_runner as runner
from test_run_in_training_pause import complete_row, write_registry

REPO = Path(__file__).resolve().parents[4]
NOW = tr.parse_clock('Oct 5, 2026 3:00 PM PDT')
HOUR = 3600.0


def defect(did, severity='GOF-BLOCKER', found='Oct 4, 2026 2:45 PM PDT', **more):
    entry = dict(id=did, title='A measured fault of %s.' % did, component='PLANNER_SEARCH', repairs=[], stage='PLANNER',
                 found_at=found, severity=severity, severity_basis='a test', owner='the coordinator',
                 backup_owner='an agent', cause_status='FOUND: not named', next_action='find the cause',
                 waits_on='nothing', branch=None, commit=None, state='FOUND',
                 closing_test=dict(row=None, text='the fault no longer reads'), tests=[])
    entry.update(more)
    return entry


def register(tmp, *defects):
    data = dict(about='a test register', schema=1, defects=list(defects))
    dr.save(data, tmp)
    return data


def no_branch(root, branch, defect_id=None):
    return None, None


def branch_at(times):
    """A branch whose newest commit that names the defect's id was made at times[branch] (the commit's message is
    searched by the real function, defect_register._branch_time)."""
    return lambda root, branch, defect_id=None: (times.get(branch), 'abc123def' if branch in times else None)


# ------------------------------------------------------------------------------------------ times and work
def test_a_time_or_a_day_is_read_and_anything_else_is_not():
    assert dr.when('Oct 4, 2026 2:45 PM PDT') == (tr.parse_clock('Oct 4, 2026 2:45 PM PDT'), 'minute')
    assert dr.when('Sep 25, 2026') == (tr.parse_clock('Sep 25, 2026 12:00 AM PDT'), 'day')
    assert dr.when('not on record') == (None, None) and dr.when(None) == (None, None)
    assert dr.span_text(26 * HOUR) == '26 hours 0 minutes' and dr.span_text(10 * 24 * HOUR + 2 * HOUR) == '10 days 2 hours'
    assert dr.span_text(125) == '2 minutes'


def test_last_meaningful_work_is_the_newest_of_a_naming_commit_a_naming_rows_run_and_stated_work(tmp_path):
    """The dashboard review of Oct 5, 2026, finding F2: only a commit whose message names the defect, a run of a test
    whose row names the defect (its "defect" field), and stated work count; a run of a row the entry merely links
    (its tests, its closing test) does not."""
    rows = [complete_row('probe', defect='X', last_run=dict(started='Oct 5, 2026 9:00 AM PDT',
                                                            ended='Oct 5, 2026 9:30 AM PDT')),
            complete_row('closing', defect=['X', 'Z'], last_run=dict(started='Oct 5, 2026 11:00 AM PDT', ended=None)),
            complete_row('linked', last_run=dict(started='Oct 5, 2026 2:00 PM PDT', ended='Oct 5, 2026 2:59 PM PDT'))]
    d = defect('X', branch='fix/x', tests=['probe', 'linked'], closing_test=dict(row='closing', text='t'))
    work = dr.last_work(d, tmp_path, rows, branch_at({'fix/x': tr.parse_clock('Oct 5, 2026 8:00 AM PDT')}))
    assert work['at_text'] == 'Oct 5, 2026 11:00 AM PDT' and work['what'] == 'test closing started (its row names X)'
    assert work['basis'] == 'test run' and work['basis_text'] == 'a run of a test whose registry row names the defect'
    work = dr.last_work(d, tmp_path, rows, branch_at({'fix/x': tr.parse_clock('Oct 5, 2026 1:00 PM PDT')}))
    assert work['basis'] == 'branch commit' and work['what'] == 'commit abc123def of branch fix/x, whose message names X'
    stated = dict(d, last_work_stated=dict(at='Oct 5, 2026 2:30 PM PDT', what='the cause read in the code', source='s',
                                           by='an agent'))
    assert dr.last_work(stated, tmp_path, rows, no_branch)['what'] == 'the cause read in the code (stated by an agent)'
    assert dr.last_work(defect('Y'), tmp_path, rows, no_branch)['at_s'] is None, 'no branch, no test, nothing stated'
    linked_only = dr.last_work(defect('Z2', tests=['linked']), tmp_path, rows, no_branch)
    assert linked_only['at_s'] is None and linked_only['basis_text'].startswith('no meaningful work is on record')


# ------------------------------------------------------------------------------------------ the 12-hour and 24-hour rules
@pytest.mark.parametrize('idle_hours,top,needs', [(11.9, False, False), (12.0, True, False), (23.9, True, False),
                                                  (24.0, True, True), (240.0, True, True)])
def test_a_gof_blocker_goes_to_the_top_at_12_hours_and_needs_an_explanation_at_24(idle_hours, top, needs):
    d = defect('X', found='Sep 20, 2026 9:00 AM PDT')
    work = dict(at_s=NOW - idle_hours * HOUR)
    rule = dr.rule_state(d, work, NOW)
    assert (rule['top'], rule['needs_explanation']) == (top, needs)
    # the rule is the GOF-BLOCKER's alone, and an open defect's alone
    assert dr.rule_state(dict(d, severity='TRAINING-RISK'), work, NOW)['top'] is False
    assert dr.rule_state(dict(d, state='VERIFIED'), work, NOW)['needs_explanation'] is False
    assert dr.rule_state(dict(d, state='FAILED/SUPERSEDED', closed=dict(at='x', by='y', reason='z')), work,
                         NOW)['top'] is False
    # a "closed" typed beside the state FOUND does not close the entry (finding L7)
    assert dr.rule_state(dict(d, closed=dict(at='x', by='y', reason='z')), work, NOW)['top'] is top


def test_a_defect_without_work_is_idle_since_it_was_found_and_one_without_any_time_is_held():
    found_25h_ago = defect('X', found=tr.clock(NOW - 25 * HOUR))
    rule = dr.rule_state(found_25h_ago, dict(at_s=None), NOW)
    assert rule['top'] and rule['needs_explanation'] and abs(rule['idle_s'] - 25 * HOUR) < 61
    undated = dr.rule_state(defect('Y', found='not on record'), dict(at_s=None), NOW)
    assert undated['idle_s'] is None and undated['top'] and undated['needs_explanation']


def test_an_explanation_holds_for_24_hours_and_work_after_it_starts_the_count_again():
    work = dict(at_s=NOW - 30 * HOUR)

    def with_explanation(at, **more):
        return defect('X', explanation=dict(dict(text='waits for the operator', next_action='run the owed check',
                                                 by='the coordinator', at=tr.clock(at)), **more))
    assert dr.rule_state(with_explanation(NOW - 1 * HOUR), work, NOW)['needs_explanation'] is False
    assert dr.rule_state(with_explanation(NOW - 23.9 * HOUR), work, NOW)['needs_explanation'] is False
    assert dr.rule_state(with_explanation(NOW - 24.1 * HOUR), work, NOW)['needs_explanation'] is True, 'it lapsed'
    assert dr.rule_state(with_explanation(NOW - 1 * HOUR, next_action=''), work, NOW)['needs_explanation'] is True
    assert dr.rule_state(with_explanation(NOW - 1 * HOUR, text=' '), work, NOW)['needs_explanation'] is True
    # an explanation written before the last work explains an earlier idle time, not this one
    assert dr.rule_state(with_explanation(NOW - 31 * HOUR), work, NOW)['needs_explanation'] is True


def test_the_view_lists_stale_blockers_first_and_computes_every_age(tmp_path):
    write_registry(tmp_path, [complete_row('probe', defect='FRESH_BLOCKER',
                                           last_run=dict(started='Oct 5, 2026 1:00 PM PDT',
                                                         ended='Oct 5, 2026 2:00 PM PDT'))])
    register(tmp_path,
             defect('FRESH_BLOCKER', tests=['probe']),
             defect('RISK', severity='TRAINING-RISK', found='Sep 25, 2026'),
             defect('STALE_BLOCKER', found='Oct 4, 2026 7:26 AM PDT'),
             defect('OLDER_STALE_BLOCKER', found='Sep 25, 2026'),
             defect('CLOSED', state='VERIFIED', closing_test=dict(row='probe', text='t')))
    view = dr.view(tmp_path, tmp_path, now=NOW, with_manifest=False, branch_time=no_branch)
    assert [d['id'] for d in view['defects']] == ['OLDER_STALE_BLOCKER', 'STALE_BLOCKER', 'FRESH_BLOCKER', 'RISK',
                                                 'CLOSED']
    assert view['top'] == ['OLDER_STALE_BLOCKER', 'STALE_BLOCKER'] == view['holding']
    by_id = {d['id']: d for d in view['defects']}
    assert by_id['OLDER_STALE_BLOCKER']['age_text'] == '10 days 15 hours since it was found (a day without its time of day)'
    assert by_id['STALE_BLOCKER']['idle_text'] == ('31 hours 34 minutes since it was found; no meaningful work is on '
                                                   'record: no commit of its branch names it, no test whose row names '
                                                   'it has run, and no work is stated')
    assert by_id['FRESH_BLOCKER']['idle_text'] == ('1 hour 0 minutes since its last meaningful work, a run of a test '
                                                   'whose registry row names the defect: test probe ended (its row '
                                                   'names FRESH_BLOCKER)')
    assert by_id['FRESH_BLOCKER']['top'] is False and by_id['CLOSED']['open'] is False
    assert view['status_lines'] == ['OLDER_STALE_BLOCKER: FOUND', 'STALE_BLOCKER: FOUND', 'FRESH_BLOCKER: FOUND',
                                    'RISK: FOUND']
    assert view['sentence'].startswith('4 open defects, 3 of them GOF-BLOCKER. 2 GOF-BLOCKER have had no meaningful '
                                       'work for 12 hours or more. The 24-hour rule holds back every test')
    assert 'CLOSED is typed VERIFIED and its closing test probe has no PASS written by a runner' in view['problems']


# ------------------------------------------------------------------------------------------ the runner's gate
def test_at_24_hours_the_runner_refuses_a_test_that_serves_no_open_gof_blocker(tmp_path):
    stale = defect('STALE_BLOCKER', found='Oct 4, 2026 7:26 AM PDT', component='PLANNER_SEARCH',
                   repairs=[dict(component='PLANNER_SEARCH_REPAIR', state='FOUND')], tests=['its_probe'])
    other = defect('OTHER_BLOCKER', found=tr.clock(NOW - 2 * HOUR), component='ENCODER_VESTIBULAR')
    risk = defect('RISK', severity='TRAINING-RISK', component='LIVENESS_REPAIR', tests=['risk_probe'])
    reg = dict(defects=[stale, other, risk])
    gate = lambda row: dr.runner_gate(row, register=reg, now=NOW, branch_time=no_branch)   # noqa: E731
    lower = gate(dict(id='risk_probe', component='LIVENESS_REPAIR'))
    assert lower.startswith('the defect register holds back every test that serves no open GOF-BLOCKER: STALE_BLOCKER')
    assert 'has had no meaningful work for 31 hours 34 minutes' in lower and 'defect_register explain STALE_BLOCKER' in lower
    assert gate(dict(id='dashboard_check', component='DASHBOARD')) is not None, 'a row that serves no defect at all'
    # a row that serves any open GOF-BLOCKER starts: by its id, by its component, by a repair's component, by name
    assert gate(dict(id='its_probe')) is None
    assert gate(dict(id='x', component='PLANNER_SEARCH')) is None
    assert gate(dict(id='x', component='PLANNER_SEARCH_REPAIR')) is None
    assert gate(dict(id='x', component='ENCODER_VESTIBULAR')) is None, 'another open GOF-BLOCKER'
    assert gate(dict(id='x', defect='OTHER_BLOCKER')) is None
    # the coordinator's explanation with a named next action lifts it, for 24 hours
    stale['explanation'] = dict(text='waits for a qualified checkpoint', next_action='rerun the physical test',
                                by='the coordinator', at=tr.clock(NOW - 60))
    assert gate(dict(id='risk_probe', component='LIVENESS_REPAIR')) is None
    assert dr.runner_gate(dict(id='risk_probe'), register=reg, now=NOW + 25 * HOUR, branch_time=no_branch) is not None
    # under 24 hours nothing is held
    young = dict(defects=[defect('YOUNG', found=tr.clock(NOW - 23 * HOUR))])
    assert dr.runner_gate(dict(id='anything'), register=young, now=NOW, branch_time=no_branch) is None


def test_the_test_runner_applies_the_gate_before_it_starts_a_job(tmp_path, monkeypatch):
    import sys
    path = write_registry(tmp_path, [complete_row('lower'), complete_row('serves', component='PLANNER_SEARCH')])
    monkeypatch.setenv('UEMA_TEST_RUNS_DIR', str(tmp_path/'runs'))
    marker = tmp_path/'ran.txt'
    cmd = [sys.executable, '-c', 'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("x")', str(marker)]
    reg = dict(defects=[defect('STALE_BLOCKER', found='Sep 25, 2026')])
    monkeypatch.setattr(dr, 'load', lambda root=None, path=None: reg)
    monkeypatch.setattr(dr, '_branch_time', no_branch)
    assert runner.run_command('lower', cmd, path=path, root=tmp_path)[0] == runner.EXIT_REFUSED
    assert not marker.exists() and tr.rows_with_id(tr.load(path), 'lower')[0]['status'] == 'queued'
    assert runner.run_command('serves', cmd, path=path, root=tmp_path)[:2] == (0, 'PASS') and marker.exists()
    # a register that cannot be read refuses every test: a missing file does not switch the rule off
    def broken(root=None, path=None):
        raise dr.RegisterError('the defect register cannot be read')
    monkeypatch.setattr(dr, 'load', broken)
    why = runner.refusal('lower', path=path, root=tmp_path)
    assert "24-hour rule could not be applied" in why and 'every test is refused' in why


def test_the_tests_page_shows_a_held_row_as_blocked(tmp_path, monkeypatch):
    write_registry(tmp_path, [complete_row('lower'), complete_row('serves', component='PLANNER_SEARCH')])
    reg = dict(defects=[defect('STALE_BLOCKER', found='Sep 25, 2026')])
    monkeypatch.setattr(dr, 'load', lambda root=None, path=None: reg)
    monkeypatch.setattr(dr, '_branch_time', no_branch)
    view = tr.dashboard_view(tmp_path, audit=False)
    by_id = {v['id']: v for g in view['groups'] for v in g['rows']}
    assert by_id['lower']['section'] == 'BLOCKED' and by_id['serves']['section'] == 'QUEUED'
    assert by_id['lower']['section_reason'].startswith('Blocked by the 24-hour rule: the defect register holds back')


# ------------------------------------------------------------------------------------------ the typed states
def test_a_typed_state_that_its_check_does_not_support_is_a_problem():
    def status(on=None, qa=None, stale=False):
        return dict(switch=dict(repair_on=on, sentence='the switch sentence'), last_qa=qa, qa_stale=stale,
                    qa_stale_text='QA STALE: x changed' if stale else None)
    review = dict(reviewed_commit='48f11f612')
    d = defect('TILT', repairs=[dict(component='HEAD_TILT_REPAIR', state='IN TRAINING')], state='IN TRAINING')
    off = dr.state_problems(d, dict(HEAD_TILT_REPAIR=status(on=False)))
    assert 'TILT: HEAD_TILT_REPAIR is typed IN TRAINING and its switch is off in the active run' in off[0]
    assert 'TILT is typed IN TRAINING and none of its repairs is switched on in the active run' in off[1]
    assert dr.state_problems(d, dict(HEAD_TILT_REPAIR=status(on=True))) == []
    built = defect('TILT', repairs=[dict(component='HEAD_TILT_REPAIR', state='BUILT/OFF')], state='BUILT/OFF')
    assert 'is typed BUILT/OFF and its switch is on' in dr.state_problems(built, dict(HEAD_TILT_REPAIR=status(on=True)))[0]
    assert dr.state_problems(built, dict(HEAD_TILT_REPAIR=status(on=False))) == []
    reviewed = defect('TILT', repairs=[dict(component='HEAD_TILT_REPAIR', state='REVIEWED')], state='REVIEWED')
    assert 'is typed REVIEWED and no independent review of it is on record' in dr.state_problems(
        reviewed, dict(HEAD_TILT_REPAIR=status(on=False)))[0]
    stale = dr.state_problems(reviewed, dict(HEAD_TILT_REPAIR=status(on=False, qa=review, stale=True)))
    assert 'is typed REVIEWED and its code changed after the reviewed commit 48f11f612' in stale[0]
    assert dr.state_problems(reviewed, dict(HEAD_TILT_REPAIR=status(on=False, qa=review))) == []
    assert 'its state is not one of' in dr.state_problems(defect('X', state='FIXED'))[0]
    unknown = dr.state_problems(defect('X', repairs=[dict(component='HEAD_REPAIR', state='BUILT/OFF')]), {})
    assert unknown == ['X: its repair HEAD_REPAIR is not an id of the component manifest']


def test_verified_needs_the_closing_tests_pass_written_by_a_runner(tmp_path):
    path = write_registry(tmp_path, [complete_row('closing')])
    d = defect('X', state='VERIFIED', closing_test=dict(row='closing', text='the fault no longer reads'))
    rows = tr.load(path)['rows']
    assert dr.state_problems(d, rows=rows) == ['X is typed VERIFIED and its closing test closing has no PASS written '
                                               'by a runner']
    typed = [dict(rows[0], status='done', outcome='PASS', last_run=dict(state='PASS'))]      # typed, not run
    assert dr.state_problems(d, rows=typed) != []
    assert tr.job_started('closing', path=path) is None and tr.finish_job('closing', exit_code=0, path=path)[1] == 'PASS'
    assert dr.state_problems(d, rows=tr.load(path)['rows']) == []


def test_a_malformed_register_is_named(tmp_path):
    data = register(tmp_path, defect('lower_case'), defect('X', severity='URGENT'), defect('X'),
                    dict(defect('Y', found='some day'), owner=''), defect('Z', component='HEAD_REPAIR'))
    found = dr.structure_problems(data, component_ids={'PLANNER_SEARCH'})
    assert 'lower_case: the id is not upper case words joined by underscores' in found
    assert ('X: severity is not one of GOF-BLOCKER, TRAINING-RISK, MEASUREMENT-RISK, NONBLOCKING, '
            'CURRENT EFFECT UNKNOWN') in found
    assert 'X: two defects have this id' in found and 'Y: owner is missing' in found
    assert 'Y: found_at is not a time or a day, and the entry does not list it as unverified' in found
    assert 'Z: its component HEAD_REPAIR is not an id of the component manifest' in found
    dr.register_path(tmp_path).write_text('{"defects": ', encoding='utf-8')
    view = dr.view(tmp_path, tmp_path, now=NOW, with_manifest=False)
    assert view['available'] is False and 'cannot be read' in view['problems'][0]


def test_the_two_writers_stamp_the_clock_and_refuse_an_empty_field(tmp_path):
    register(tmp_path, defect('X'))
    with pytest.raises(dr.RegisterError, match='needs its text, a named next action and who writes it'):
        dr.explain('X', 'waits', '', 'the coordinator', tmp_path)
    with pytest.raises(dr.RegisterError, match='no single defect has id'):
        dr.explain('NOBODY', 'waits', 'run it', 'the coordinator', tmp_path)
    written = dr.explain('X', 'waits for the operator', 'run the owed check', 'the coordinator', tmp_path)
    assert tr.parse_clock(written['explanation']['at']) is not None
    assert dr.state_work('X', 'the cause read in the code', 'an agent', 'a record', tmp_path)['last_work_stated']['by'] \
        == 'an agent'
    assert json.loads(dr.register_path(tmp_path).read_text(encoding='utf-8'))['defects'][0]['explanation']['by'] \
        == 'the coordinator'


# ------------------------------------------------------------------------------------------ the review's re-check, N1 and N2
def test_stated_work_carries_the_time_the_work_was_done_no_later_than_now(tmp_path):
    """Finding N2: work stated at 1:48 PM PDT counted from 1:48 PM, hours after the runs it named had ended. The
    statement now carries the time the work was done (no later than now; the clock's time without one) and the time it
    was written, and the idle text prints both."""
    register(tmp_path, defect('X'))
    now = tr.parse_clock('Oct 5, 2026 2:40 PM PDT') + 20
    entry = dr.state_work('X', 'the design-3 rule read', 'the coordinator', 'row r', tmp_path,
                          at='Oct 5, 2026 10:48 AM PDT', now=now)
    stated = entry['last_work_stated']
    assert (stated['at'], stated['written_at']) == ('Oct 5, 2026 10:48 AM PDT', 'Oct 5, 2026 2:40 PM PDT')
    work = dr.last_work(entry, tmp_path, [], no_branch)
    assert work['at_s'] == tr.parse_clock('Oct 5, 2026 10:48 AM PDT') and work['basis'] == 'stated'
    assert work['what'] == ('the design-3 rule read (done Oct 5, 2026 10:48 AM PDT; stated by the coordinator at '
                            'Oct 5, 2026 2:40 PM PDT)')
    assert dr.rule_state(entry, work, now)['idle_s'] == now - tr.parse_clock('Oct 5, 2026 10:48 AM PDT')
    with pytest.raises(dr.RegisterError, match='is later than now'):
        dr.state_work('X', 'w', 'the coordinator', None, tmp_path, at='Oct 5, 2026 2:41 PM PDT', now=now)
    with pytest.raises(dr.RegisterError, match='is not a time as the register writes one'):
        dr.state_work('X', 'w', 'the coordinator', None, tmp_path, at='this morning', now=now)
    plain = dr.state_work('X', 'w', 'the coordinator', None, tmp_path, now=now)['last_work_stated']
    assert plain['at'] == plain['written_at'] == 'Oct 5, 2026 2:40 PM PDT', 'without --at: the clock\'s time'
    # a time typed into the file by hand, later than now or than its writing, is a problem of the register
    future = dict(defect('Y'), last_work_stated=dict(at='Oct 5, 2026 3:00 PM PDT', written_at='Oct 5, 2026 2:00 PM PDT',
                                                     what='w', by='b'))
    found = dr.stated_problems(future, now)
    assert any('later than now' in p for p in found) and any('later than the time it was written' in p for p in found)


def test_stated_work_taken_back_is_kept_and_no_longer_counts(tmp_path):
    """Finding N2: a restated finding is not work; the coordinator takes the statement back, and the register keeps
    what was taken back, by whom and why."""
    register(tmp_path, defect('X'))
    dr.state_work('X', 'found and measured', 'the coordinator', None, tmp_path)
    entry = dr.withdraw_work('X', 'the coordinator', 'it restates the finding', tmp_path)
    assert 'last_work_stated' not in entry and entry['withdrawn_work'][0]['statement']['what'] == 'found and measured'
    assert entry['withdrawn_work'][0]['why'] == 'it restates the finding'
    assert dr.last_work(dr.load(tmp_path)['defects'][0], tmp_path, [], no_branch)['at_s'] is None
    with pytest.raises(dr.RegisterError, match='has no stated work to take back'):
        dr.withdraw_work('X', 'the coordinator', 'again', tmp_path)
    with pytest.raises(dr.RegisterError, match='needs who does it and why'):
        dr.withdraw_work('X', 'the coordinator', '', tmp_path)


def test_a_printed_text_with_a_bare_score_decimal_is_a_register_problem():
    """Finding N1: the Now page printed a title's cosines as 0.86, 0.45 and 0.23. A text the pages print (the title,
    the next action, what it waits on, the owner, the stated work) is held to DESIGN.md section 11; a text the pages
    do not print (the cause status) is not judged here."""
    shown = defect('X', title='Mean absolute cosine 0.86 predicted, 0.45 real.',
                   last_work_stated=dict(at='Oct 5, 2026 10:48 AM PDT', what='a share of 0.5 read', by='b'))
    found = dr.structure_problems(dict(defects=[shown]))
    assert [p.split(' (')[0] for p in found] == ['X: its title, which the pages print, holds 0.86',
                                                 'X: its title, which the pages print, holds 0.45',
                                                 'X: its stated work, which the pages print, holds 0.5'], found
    counts = defect('Y', title='Mean absolute cosine, in every 100: 86 predicted, 45 real.',
                    cause_status='FOUND: a standard deviation of 0.127')
    assert dr.structure_problems(dict(defects=[counts])) == []


# ------------------------------------------------------------------------------------------ this checkout's register
def test_this_checkouts_register_is_well_formed_and_names_the_known_defects():
    from mimo_bridge.tools import component_manifest
    data = dr.load(REPO)
    ids = set(component_manifest.components(component_manifest.load(REPO)))
    assert dr.structure_problems(data, ids) == []
    known = {d['id']: d for d in data['defects']}
    for name in ('VESTIBULAR_TOKEN_NO_HEAD_TILT', 'PLANNER_SEARCH_UNSTABLE', 'PLANNER_NO_REAL_GAIN',
                 'TACTILE_VESTIBULAR_PREDICTIONS', 'WELLNESS_PREDICTOR_LAYER_ZERO_GRADIENT',
                 'MOTOR_CONFIDENCE_HEAD_PINNED', 'PERSISTENT_MEMORY_CONSTANT', 'GOAL_VALUE_HAND_PRESSING_FLAG',
                 'EYES_HELD_UPWARD', 'MOTOR_TOKEN_ASKED_CHANGE', 'UNIT_TEST_TRANSITION_CONTRACT'):
        assert name in known, name
    assert all(d.get('sources') and isinstance(d.get('unverified'), dict) for d in data['defects'])
    assert all(r['component'] in ids and r['state'] in dr.STATES for d in data['defects'] for r in d['repairs'])
    tilt = known['VESTIBULAR_TOKEN_NO_HEAD_TILT']
    assert tilt['state'] == 'BUILT/OFF' and tilt['severity'] == 'GOF-BLOCKER' and tilt['found_at'] == 'Sep 25, 2026'
    # Oct 7, 2026: a defect may now stand at VERIFIED, but only by the register's own rule (its closing test's row at
    # PASS written by a runner); the blanket "nothing can reach VERIFIED" of Oct 5 held only while nothing had.
    rows = tr.load(tr.registry_path(REPO))['rows']
    unearned = [p for d in data['defects'] if d['state'] == 'VERIFIED' for p in dr.state_problems(d, rows=rows)]
    assert unearned == [], unearned
