# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Malformed control records cannot authorize a training restart."""
import json
import pytest
from mimo_bridge.tools import training_continuation as tc
from mimo_bridge.tools import training_preflight as pf


@pytest.mark.parametrize('value', [{'cmd':'paused'}, {'cmd':None}, {'cmd':True}, {'cmd':''}, []])
def test_unknown_supervisor_command_stays_paused(tmp_path, value):
    (tmp_path / 'supervisor_control.json').write_text(json.dumps(value), encoding='utf8')
    assert tc.pause_reason(tmp_path, tmp_path)


def test_explicit_run_and_bom_pause(tmp_path):
    path = tmp_path / 'supervisor_control.json'
    path.write_text('{"cmd":"run"}', encoding='utf8')
    assert tc.pause_reason(tmp_path, tmp_path) is None
    path.write_text('{"cmd":"pause"}', encoding='utf-8-sig')
    assert tc.pause_reason(tmp_path, tmp_path) == 'explicit supervisor pause'


@pytest.mark.parametrize('name', ['stop_reason.json', 'control_done_10.json'])
def test_nonobject_stop_record_is_a_pause(tmp_path, name):
    (tmp_path / name).write_text('[]', encoding='utf8')
    assert tc.pause_reason(tmp_path, tmp_path)


@pytest.mark.parametrize('reason', [None, '', ' ', 0, False])
def test_repair_required_does_not_depend_on_explanation_truthiness(tmp_path, reason):
    (tmp_path / 'stop_reason.json').write_text(json.dumps(
        {'resume_policy':'repair_required', 'reason':reason}), encoding='utf8')
    assert tc.pause_reason(tmp_path, tmp_path) == 'watcher requires repair'


@pytest.mark.parametrize('resume', ['false', 'true', 1, None, False])
def test_manual_stop_requires_literal_true_to_allow_resume(tmp_path, resume):
    (tmp_path / 'control_done_10.json').write_text(json.dumps(
        {'cmd':'snapshot_and_stop', 'auto_resume':resume}), encoding='utf8')
    assert tc.pause_reason(tmp_path, tmp_path) == 'deliberate stop recorded; explicit continuation required'


def test_literal_true_keeps_scheduled_resume(tmp_path):
    (tmp_path / 'control_done_10.json').write_text(
        '{"cmd":"snapshot_and_stop","auto_resume":true}', encoding='utf8')
    assert tc.pause_reason(tmp_path, tmp_path) is None


def test_resume_uses_existing_admission_decision(monkeypatch):
    seen = []
    def refusal(command):
        seen.append(command)
        return 'source changed'
    monkeypatch.setattr(pf, 'launch_refusal', refusal)
    command = ['python', '-m', 'mimo_bridge.train_mimo', '--resume', 'saved']
    assert tc.resume_launch_refusal(command) == 'source changed'
    assert seen == [command]


def test_unreadable_admission_does_not_launch(monkeypatch):
    def broken(command):
        raise OSError('unreadable record')
    monkeypatch.setattr(pf, 'launch_refusal', broken)
    assert 'unreadable record' in tc.resume_launch_refusal(['python'])
