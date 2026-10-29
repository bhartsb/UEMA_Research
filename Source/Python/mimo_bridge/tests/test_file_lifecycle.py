# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""File-management regressions; temporary files and tiny subprocesses, no model/world."""
import copy
import json
import os
import sys
from pathlib import Path

import pytest

from mimo_bridge.tools import file_lifecycle as fl
from mimo_bridge.tools import test_registry as tr
from mimo_bridge.tools import test_runner as runner
from test_test_runner import new_row
from test_run_in_training_pause import write_registry


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, 'component_ids', lambda root=None: {'TEST_REGISTRY'})
    monkeypatch.setattr(tr, 'code_state', lambda *a, **k: ('test-code', False, 'test-branch'))
    monkeypatch.setenv('UEMA_TEST_RUNS_DIR', str(tmp_path/'runner'))
    policy = dict(version=1, manifest_dir=str(tmp_path/'records'), managed_roots=[str(tmp_path/'scratch')])
    base = tmp_path/'scratch'/'work'/'attempts'/'one'
    result = base/'evidence'/'result.json'
    code = "import os,json,pathlib; p=pathlib.Path(os.environ['UEMA_EVIDENCE_DIR'])/'result.json'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps({'verdict':'pass','tmp':os.environ['TMP']}))"
    command = [sys.executable, '-c', code]
    row = new_row('files', result_source={'file': str(result), 'keys': ['verdict']})
    registry = write_registry(tmp_path, [row])
    data = tr.load(registry)
    data['file_lifecycle_policy'] = policy
    fl.write(registry, data)
    record = fl.prepare(registry, 'work', 'one', 'files', 'Verify output ownership', 'test owner', command, str(tmp_path/'scratch'))
    return dict(policy=policy, base=base, registry=registry, record=record, command=command, tmp=tmp_path)


def row(case):
    return tr.load(case['registry'])['rows'][0]


def test_registration_creates_no_run_output_and_validates_new_attempt(case):
    assert not case['base'].exists()
    assert fl.admission(row(case), case['policy']) is None


@pytest.mark.parametrize('damage', ['absent', 'outside_result', 'wrong_declaration', 'missing_owner'])
def test_incomplete_or_conflicting_ownership_refuses(case, damage):
    r = row(case)
    if damage == 'absent':
        del r['file_lifecycle']
    elif damage == 'outside_result':
        r['result_source']['file'] = str(case['tmp']/'elsewhere.json')
    elif damage == 'wrong_declaration':
        r['file_lifecycle']['retention'] = 'changed'
    else:
        p = fl.manifest_path(case['policy'], 'work')
        d = fl.read(p); d['owner'] = ''; fl.write(p, d)
    assert fl.admission(r, case['policy'])


@pytest.mark.parametrize('value', [[], {'managed_attempts':[1]}, {'managed_attempts':None}])
def test_malformed_manifest_is_plain_refusal(case, value):
    fl.write(fl.manifest_path(case['policy'], 'work'), value)
    assert fl.admission(row(case), case['policy']).startswith('file lifecycle:')


def test_path_escape_and_evidence_tmp_alias_are_refused(case):
    r = row(case)
    r['file_lifecycle']['tmp_dir'] = r['file_lifecycle']['evidence_dir']
    assert fl.admission(r, case['policy'])
    assert not fl.inside(case['tmp']/'scratch-other'/'x', case['tmp']/'scratch')
    assert not fl.inside(str(case['base'])+'/../escape', case['base'])


def test_link_in_an_ancestor_is_refused(case, monkeypatch):
    real = fl.link
    monkeypatch.setattr(fl, 'link', lambda p: Path(p)==case['base'].parent or real(p))
    assert fl.admission(row(case), case['policy'])


def test_actual_command_must_equal_recorded_generator_before_claim(case):
    with pytest.raises(ValueError, match='actual command'):
        fl.claim_start(row(case), case['policy'], 'run1', ['different'])
    assert fl.admission(row(case), case['policy']) is None


def test_claim_prevents_second_launcher_and_used_directory_is_refused(case):
    fl.claim_start(row(case), case['policy'], 'run1', case['command'])
    with pytest.raises(ValueError, match='already used'):
        fl.claim_start(row(case), case['policy'], 'run2', case['command'])


def test_unsupported_pause_scheduler_path_refuses_before_spawn(case):
    assert 'run_command' in tr.refusal('files', path=case['registry'], check_rule=False)


def test_real_runner_routes_tmp_and_records_closeout(case):
    code, state, log = runner.run_command('files', case['command'], path=case['registry'], check_rule=False,
                                        end_when_trainer_starts=False, poll_s=.05)
    assert code == 0 and state == 'PASS'
    assert Path(log) == case['base']/'evidence'/'runner.log'
    r = row(case)
    assert r['file_lifecycle_close']['state'] == 'recorded'
    assert tr.sealed(r)
    result = fl.read(case['base']/'evidence'/'result.json')
    assert Path(result['tmp']) == case['base']/'tmp'
    d = fl.read(fl.manifest_path(case['policy'], 'work'))
    assert d['managed_attempts'][0]['state'] == 'closed'
    assert 'no deletion' in d['managed_attempts'][0]['temporary_review']


def test_old_completed_run_does_not_close_next_planned_attempt(case):
    runner.run_command('files', case['command'], path=case['registry'], check_rule=False,
                       end_when_trainer_starts=False, poll_s=.05)
    new_result = case['base'].parent/'two'/'evidence'/'result.json'
    tr.update_row('files', {'result_source':{'file':str(new_result), 'keys':['verdict']}}, path=case['registry'])
    fl.prepare(case['registry'], 'work', 'two', 'files', 'same purpose', 'owner', case['command'], str(case['tmp']/'scratch'))
    fl.close_from_registry(row(case), case['policy'])
    attempts = fl.read(fl.manifest_path(case['policy'], 'work'))['managed_attempts']
    assert [a['state'] for a in attempts] == ['closed', 'planned']


def test_failed_close_preserves_outcome_and_records_reconciliation(case, monkeypatch):
    def broken(*a):
        raise OSError('disk problem')
    monkeypatch.setattr(fl, 'close_from_registry', broken)
    code, state, _ = runner.run_command('files', case['command'], path=case['registry'], check_rule=False,
                                       end_when_trainer_starts=False, poll_s=.05)
    assert state == 'PASS' and code == 0
    assert row(case)['file_lifecycle_close']['state'] == 'reconcile_required'


def test_failed_registry_start_marks_claim_not_started(case, monkeypatch):
    monkeypatch.setattr(tr, '_write', lambda *a, **k: 'simulated registry write failure')
    why = tr.job_started('files', path=case['registry'], command=case['command'], run_id='failed_start', commit='source')
    assert 'simulated registry' in why
    d=fl.read(fl.manifest_path(case['policy'], 'work'))
    assert d['managed_attempts'][0]['state']=='start_record_failed'
    assert not case['base'].exists()


def test_claim_is_not_redirected_by_concurrent_row_change(case, monkeypatch):
    real=fl.claim_start
    def change(r, p, run_id, command):
        real(r,p,run_id,command)
        tr.update_row('files', {'result_source':{'file':str(case['tmp']/'elsewhere.json'),'keys':['verdict']}},path=case['registry'])
    monkeypatch.setattr(fl,'claim_start',change)
    why=tr.job_started('files',path=case['registry'],command=case['command'],run_id='race',commit='source')
    assert 'changed during' in why
    assert not case['base'].exists()


def test_production_receipt_preserves_existing_checkpoint_paths(case, monkeypatch):
    # This records a fixture process; it never starts a trainer or touches learned state.
    monkeypatch.setattr(tr,'rule_refusal',lambda *a:None)
    tr.update_row('files',{'file_lifecycle':None},path=case['registry'])
    run=runner.segment_start('files',case['tmp']/'preserved_training',pid=1234,path=case['registry'],check_rule=False)
    assert row(case)['last_run']['production_run']==str(case['tmp']/'preserved_training')
    assert not case['base'].exists()


def test_old_unbound_result_is_not_counted_as_closed(case):
    r=row(case);r['last_run']={'run_id':'older_history','ended':'earlier'}
    assert fl.close_from_registry(r,case['policy']) is False


@pytest.mark.parametrize('policy', [[], {}, None, {'version':99}])
def test_malformed_configured_policy_refuses(case, policy):
    assert fl.runner_refusal(row(case), {'file_lifecycle_policy':policy})


def test_completion_uses_bound_result_when_row_changes(case):
    run=runner.Run.start('files',path=case['registry'],command=case['command'],check_rule=False,lifecycle_supported=True)
    bound=case['base']/'evidence'/'result.json';bound.parent.mkdir(parents=True)
    fl.write(bound,{'verdict':'pass'})
    other=case['tmp']/'wrong-result.json';fl.write(other,{'verdict':'fail'})
    tr.update_row('files',{'result_source':{'file':str(other),'keys':['verdict']}},path=case['registry'])
    why,state=run.finish(exit_code=0)
    assert why is None and state=='PASS'
    r=row(case)
    assert Path(r['last_run']['result_file'])==bound
    assert Path(r['last_run']['result_file_record']['path'])==bound
    first=copy.deepcopy(r['last_run'])
    run.finish(exit_code=1)
    assert row(case)['last_run']==first


def test_tmp_creation_failure_closes_started_row(case, monkeypatch):
    def broken(*a):
        raise OSError('cannot create temporary output')
    monkeypatch.setattr(runner.Run, 'env', broken)
    _, state, _ = runner.run_command('files', case['command'], path=case['registry'], check_rule=False,
                                    end_when_trainer_starts=False)
    assert state == 'INVALID'
    assert row(case)['status'] != 'running'


def test_inventory_uses_evidence_not_filenames_and_replaces_own_database(tmp_path, monkeypatch):
    root = tmp_path/'scratch'; root.mkdir()
    (root/'pytest_old').mkdir()
    unknown = root/'pytest_old'/'unknown.py'; unknown.write_text('not proved disposable')
    (root/'checkpoints').mkdir(); (root/'checkpoints'/'opaque.pt').write_text('preserve')
    monkeypatch.setattr(fl, 'documented_groups', lambda: ([], []))
    store=tmp_path/'index'
    a=fl.inventory(root, store); b=fl.inventory(root, store)
    assert a['files']==b['files']==2
    assert b['categories']=={'protected_state':1, 'unidentified':1}
    assert unknown.exists()
    assert fl.inventory(root, store, retry_errors=True)['files']==2
    with pytest.raises(ValueError, match='outside'):
        fl.inventory(root, root/'index')


@pytest.mark.skipif(os.name != 'nt', reason='Windows refuses a replace over a file open without delete sharing')
def test_write_waits_out_a_reader_holding_the_record(tmp_path):
    """Run u's start claim got WinError 5 on Oct 25, 2026; a reader holding the record reproduces that error (a
    possible cause; the original holder is unknown). A reader that lets go within the bound no longer refuses the
    write."""
    import threading
    path = tmp_path/'record.json'
    fl.write(path, {'v': 1})
    held = open(path, 'rb')
    threading.Timer(.3, held.close).start()
    fl.write(path, {'v': 2})
    assert held.closed and fl.read(path) == {'v': 2}
    assert [p.name for p in tmp_path.iterdir()] == ['record.json']


@pytest.mark.skipif(os.name != 'nt', reason='Windows refuses a replace over a file open without delete sharing')
def test_write_refusal_past_the_bound_raises_and_keeps_the_record(tmp_path, monkeypatch):
    """A handle held past REPLACE_WAIT_S: the PermissionError is raised, the previous JSON stays whole and the
    writer's own temporary file is removed."""
    monkeypatch.setattr(fl, 'REPLACE_WAIT_S', .2)
    path = tmp_path/'record.json'
    fl.write(path, {'v': 1})
    with open(path, 'rb'):
        with pytest.raises(PermissionError):
            fl.write(path, {'v': 2})
    assert fl.read(path) == {'v': 1}
    assert [p.name for p in tmp_path.iterdir()] == ['record.json']
