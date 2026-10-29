# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The remembered hand-goal comparison as a scheduled ability test.

docs/gain_of_function_v3/compare_hand_memory_route_2026-09-24.py compares the learner's remembered
hand-goal stage on against off at two motor-latch decisions of held-out bodies 4-7. It passed at
3:37 AM PDT on 2026-09-24 (step 380,200); the register promised it every 3,000 updates from step
380,550 and no task existed (dashboard staleness check, 2026-09-24). These tests pin its launch
command (the script requires --calibration), its result validator on the script's own recorded
output, and its manifest entry with the script's own declared memory need. No model is loaded.
"""
import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from mimo_bridge.tools import ability_rechecks as worker

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / 'docs/gain_of_function_v3/compare_hand_memory_route_2026-09-24.py'
CALIBRATION = REPO / 'docs/gain_of_function_v3/measurements/hand_response_calibration_380200.json'
# the script's result of 3:37 AM PDT, filed with every field its result.json writes
MEASURED = REPO / 'docs/gain_of_function_v3/measurements/hand_memory_route_2026-09-24.json'


def manifest_task():
    manifest = json.loads((REPO / 'docs/gain_of_function_v3/ability_rechecks.json').read_text(encoding='utf-8'))
    return {t['id']: t for t in manifest['tasks']}['hand_memory_route']


def payload_and_record(tmp_path):
    payload = json.loads(MEASURED.read_text(encoding='utf-8'))
    record = dict(checkpoint=payload['checkpoint'], script_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
                  command=worker.task_command(REPO, manifest_task(), payload['checkpoint'], tmp_path / 'out'))
    return payload, record


def test_the_launch_command_passes_every_required_option_with_the_calibration_as_a_repository_file(tmp_path):
    task = manifest_task()
    command = worker.task_command(REPO, task, tmp_path / 'ckpt_step_399050', tmp_path / 'out')
    assert Path(command[2]) == SCRIPT.resolve()
    calibration = Path(command[command.index('--calibration') + 1])
    assert calibration.is_absolute() and calibration == CALIBRATION.resolve()
    required = re.findall(r"add_argument\('(--[a-z-]+)'[^)]*required=True", SCRIPT.read_text(encoding='utf-8'))
    assert set(required) == {'--checkpoint', '--calibration', '--out'}
    assert all(flag in command for flag in required), (required, command)
    assert command[command.index('--checkpoint') + 1] == str(tmp_path / 'ckpt_step_399050')


@pytest.mark.parametrize('arguments,reason', [
    ({'--calibration': '../outside.json'}, 'workspace'),
    ({'--calibration': 'docs/no_such_calibration.json'}, 'missing file'),
    ({'-c': 'docs/gain_of_function_v3/measurements/hand_response_calibration_380200.json'}, 'plain long option'),
    ({'--out': 'docs/gain_of_function_v3/measurements/hand_response_calibration_380200.json'}, 'plain long option'),
])
def test_an_extra_argument_outside_the_repository_missing_or_malformed_is_refused(arguments, reason):
    task = dict(manifest_task(), path_arguments=arguments)
    with pytest.raises(ValueError, match=reason):
        worker.path_arguments(REPO, task)


def test_the_scheduler_accepts_the_scripts_own_result(tmp_path):
    payload, record = payload_and_record(tmp_path)
    assert payload['script_sha256'] == record['script_sha256'], 'the script changed since its recorded run'
    worker.validate_result(payload, record)


@pytest.mark.parametrize('damage', [
    'script', 'trials', 'body_starts', 'replay', 'weights', 'nonfinger', 'production', 'route', 'calibration',
    'no_calibration_argument', 'nan_mean', 'reversed_interval', 'improved_count', 'outcome', 'learning'])
def test_an_incomplete_or_inconsistent_result_is_refused(tmp_path, damage):
    payload, record = payload_and_record(tmp_path)
    bad, rec = copy.deepcopy(payload), copy.deepcopy(record)
    if damage == 'script':
        bad['script_sha256'] = '0' * 64
    elif damage == 'trials':
        bad['completed_trials'] = 7
    elif damage == 'body_starts':
        bad['collected']['7']['starts'] = 1
    elif damage == 'replay':
        bad['exact_replay'] = False
    elif damage == 'weights':
        bad['weights_unchanged'] = False
    elif damage == 'nonfinger':
        bad['nonfinger_commands_unchanged'] = False
    elif damage == 'production':
        bad['production_modified'] = True
    elif damage == 'route':
        bad['overrides']['hand_goals'] = 'off'
    elif damage == 'calibration':
        bad['calibration_file_sha256'] = '0' * 64
    elif damage == 'no_calibration_argument':
        rec['command'] = rec['command'][:rec['command'].index('--calibration')]
    elif damage == 'nan_mean':
        bad['summary']['hand_shape_error']['means']['hand_memory'] = float('nan')
    elif damage == 'reversed_interval':
        lo, hi = bad['summary']['finger_position_error_cm']['paired95_difference']
        bad['summary']['finger_position_error_cm']['paired95_difference'] = [hi, lo]
    elif damage == 'improved_count':
        bad['summary']['finger_orientation_error_deg']['bodies_improved'] = 5
    elif damage == 'outcome':
        bad['passed'] = not bad['passed']
    else:
        bad['learning'] = True
    with pytest.raises(ValueError):
        worker.validate_result(bad, rec)


def test_a_measured_failure_completes_and_its_outcome_is_the_scripts_own_rule(tmp_path):
    """A round where the hand stage does not improve every body is a completed test with outcome
    False, never a failed attempt; a result file with NaN in it is a failed attempt."""
    payload, record = payload_and_record(tmp_path)
    out = tmp_path / 'attempt'
    out.mkdir()
    record = dict(record, output=str(out))
    worse = copy.deepcopy(payload)
    worse['summary']['hand_shape_error']['bodies_improved'] = 3
    worse['passed'] = False
    (out / 'result.json').write_text(json.dumps(worse), encoding='utf-8')
    finished = worker.finished_result(record)
    assert finished['status'] == 'complete' and finished['outcome'] is False
    worse['summary']['hand_shape_error']['means']['baseline'] = float('nan')
    (out / 'result.json').write_text(json.dumps(worse), encoding='utf-8')
    assert worker.finished_result(record)['status'] == 'failed'
    (out / 'result.json').write_text(json.dumps(payload), encoding='utf-8')
    assert worker.finished_result(record)['outcome'] is True


def test_the_manifest_declares_the_comparison_with_the_scripts_own_memory_need():
    task = manifest_task()
    assert worker.confined(REPO, task['script']) == SCRIPT.resolve() and SCRIPT.is_file()
    assert task['path_arguments'] == {
        '--calibration': 'docs/gain_of_function_v3/measurements/hand_response_calibration_380200.json'}
    # the script's own guard: verdict(reading[0], reading[1], <need GB>, <reserve GB>)
    need = re.search(r"verdict\(reading\[0\],reading\[1\],(\d+(?:\.\d+)?),6\)", SCRIPT.read_text(encoding='utf-8'))
    assert need and float(need.group(1)) == task['memory_gib'] == 12
    # 3,000 -> 12,000 updates by the operator's yes of Oct 1, 2026 (commit f11afbffc; the manifest's cadence_note)
    assert task['every_steps'] == 12000 and worker.task_class(task) == 'frequent'
    # Released by the operator's decision of 2026-09-25 9:09 AM PDT, with a failed round holding training
    # for review: a scheduler with that hold starts it at first_step_under_hold (step 399,050, as planned);
    # first_step stays out of reach of a scheduler started before the hold existed, because every
    # scheduler reads the manifest each cycle (2026-09-24).
    assert worker.first_step(task) == 399050 and task['first_step'] >= 10 ** 8
    assert task['hold_training_on_failed_comparison'] is True and not task.get('held')
    assert re.fullmatch(r'[A-Za-z0-9_-]+', task['id'])
