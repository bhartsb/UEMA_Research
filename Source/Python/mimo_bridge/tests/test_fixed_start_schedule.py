# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The fixed-start body-goal probe under the ability scheduler (docs/gain_of_function_v3/fixed_probe_set_2026-10-18.py,
task fixed_start_body_and_synergy; October 18, 2026).

The scheduler accepts a result of the new script only through its own validator branch (ability_rechecks.validate_result)
and reads its line from the result's gain object (declared_line). These tests pin: a well-formed result of the anchored mode
and of the plain frozen-start mode validates; a result whose mode, frozen set, anchor, restored starts, scope, means,
intervals, ranking, witness checks, weights, script or probe version is wrong is refused; a malformed result fails the
attempt instead of stopping the scheduler's cycle; the production manifest entry passes the scheduler's own entry checks
and runs the anchored mode on the repository's frozen-set receipt and anchor. Two tests of the script itself (no model is
loaded): the companion substitution changes only the frozen fields, and the equality comparison catches a change of one
value while skipping the timing keys and, between modes, the keys that name the mode.

THE GAIN RULE (proxy decision movement-viewer-fixed-probe-decision-20261018, October 18, 2026) adds controls for:
a missing or unequal velocity or world identity, a missing or repeated trial, a nonfinite value (each an invalid comparison);
equal, better and worse intervals; the direct arm alone deciding the line; the schedule's first-step floor under triggers
and ride-along selection; and the reference bound by its receipt's SHA-256. The script's comparisons run on saved outputs
written here (fake_runs); no model or world is loaded.
"""
import copy
import functools
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimo_bridge.tools import ability_rechecks as worker

REPO = Path(__file__).resolve().parents[4]
SCRIPT_NAME = 'fixed_probe_set_2026-10-18.py'
SCRIPT = REPO / 'docs/gain_of_function_v3' / SCRIPT_NAME
METRICS = ('physical_goal_error', 'hand_joint_error_deg', 'wrist_goal_error_cm')


def _load_script():
    sys.path.insert(0, str(SCRIPT.parent))          # the normal probe's sibling helpers (compare_synergy_physical)
    spec = importlib.util.spec_from_file_location('fixed_probe_set_2026_10_18', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gain_arm(upper):
    """One arm of a gain object as gain_block writes it (the keys the validator reads); its line is upper < 0."""
    goals = worker.BODY_GOAL_SUITES['hand_and_body']
    interval = [upper - .05, upper]
    part = dict(difference_mean=upper - .025, paired_bootstrap95=list(interval))
    return dict(metric='physical_goal_error', bodies=8, goals=5, **part,
                per_body=[dict(body=b, difference_mean=upper - .025) for b in range(8)],
                per_goal={g: dict(part) for g in goals}, line_met=upper < 0,
                hand_joint_error_deg=dict(part), wrist_goal_error_cm=dict(part))


def payload_and_record(tmp_path, anchored=True, upper=.15, direct_upper=.03, synergy_upper=.04):
    """A result of the new script as its frozen_run writes it (the keys the validator reads), with the receipt, anchor and
    reference receipt files the scheduler's command names; `upper` is the old planner-against-controller interval's upper
    end, `direct_upper` and `synergy_upper` the gain arms'."""
    from mimo_bridge.tools.command_ranking import ranking
    goals = worker.BODY_GOAL_SUITES['hand_and_body']
    rng = np.random.default_rng(0)

    def block():
        rows = [dict(body=b, names=['controller', 'direct', 'synergy'], predicted=rng.normal(size=3).tolist(),
                     actual=rng.normal(size=3).tolist()) for b in range(8) for _ in range(2)]
        return ranking(rows, 'x', planners=('direct', 'synergy'), reference='controller', draws=200)
    receipt = tmp_path / 'fixed_probe_set_receipt.json'
    receipt.write_text(json.dumps(dict(version='fixed_probe_set_v1', set_sha256='5' * 64)), encoding='utf-8')
    anchor = tmp_path / 'fixed_probe_anchor.json'
    anchor.write_text(json.dumps(dict(version='fixed_probe_anchor_v1', bodies=[])), encoding='utf-8')
    anchor_sha = hashlib.sha256(anchor.read_bytes()).hexdigest()
    reference = tmp_path / 'fixed_probe_reference.json'
    reference.write_text(json.dumps(dict(
        version='fixed_probe_reference_v1', rule_version=worker.FIXED_START_GAIN_RULE, checkpoint_step=494025,
        mode='frozen_start_anchored' if anchored else 'frozen_start', frozen_set=dict(sha256='5' * 64),
        anchor=dict(sha256=anchor_sha) if anchored else None,
        files={'result.json': dict(sha256='r' * 64), 'trials.jsonl': dict(sha256='t' * 64)})), encoding='utf-8')
    reference_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
    gain = dict(rule_version=worker.FIXED_START_GAIN_RULE, status='valid', invalid_reasons=[], primary_arm='direct',
                reference=dict(result='r/result.json', sha256='r' * 64, trials_sha256='t' * 64, checkpoint_step=494025,
                               receipt=dict(path=str(reference), sha256=reference_sha)),
                this=dict(checkpoint=str(tmp_path / 'checkpoint')), line_met=direct_upper < 0,
                arms=dict(direct=gain_arm(direct_upper), synergy=gain_arm(synergy_upper)),
                changed_controls=dict(summary={}), label='whole-arm checkpoint performance')
    against = {k: dict(mean_difference=.1, paired_bootstrap95=[.05, upper], bodies_better=0) for k in ('direct', 'synergy')}
    metric = dict(means=dict(controller=1., direct=1.1, synergy=1.2), paired_bootstrap95=[-.1, .2],
                  against_controller=against)
    check = dict(bodies=8, expected_bodies=8, informative_bodies=5, unit='degrees', parts=['chest'], requested_raw=11.,
                 requested_executed=3.9, achieved_beyond_controller=1., achieved_share=.26,
                 separation_from_controller=1.2, moved_from_start=2.)
    payload = dict(checkpoint=str(tmp_path / 'checkpoint'), learning=False, elapsed_s=10., bodies=8, goals=goals,
                   goal_suite='hand_and_body', weights_unchanged=True, exact_replay=True, script_sha256='expected',
                   mode='frozen_start_anchored' if anchored else 'frozen_start', probe='fixed_start_body_goals_v2',
                   frozen_set=dict(path='D:/set.pt.gz', sha256='5' * 64), starts_equal_to_set=True,
                   anchor=(dict(path=str(anchor), sha256=anchor_sha) if anchored else None),
                   summary={m: copy.deepcopy(metric) for m in METRICS},
                   by_goal={g: dict(witness_check=dict(check)) for g in goals},
                   ranking=dict(pooled=block(), by_goal={g: block() for g in goals}), gain=gain)
    # the scheduler's task_command: script, checkpoint, output, the plain arguments, then the path arguments sorted by flag
    command = (['python', '-u', SCRIPT_NAME, '--checkpoint', 'c', '--out', 'o', '--reference-receipt-sha256', reference_sha]
               + (['--anchor', str(anchor)] if anchored else []) + ['--frozen-set-receipt', str(receipt)]
               + ['--reference-receipt', str(reference)])
    return json.loads(json.dumps(payload)), dict(checkpoint=payload['checkpoint'], script_sha256='expected', command=command)


@pytest.mark.parametrize('anchored', [True, False])
def test_a_frozen_start_result_validates(tmp_path, anchored):
    payload, record = payload_and_record(tmp_path, anchored)
    worker.validate_result(payload, record)


@pytest.mark.parametrize('damage', ['mode', 'probe', 'set', 'anchor', 'anchor_missing', 'starts', 'bodies', 'goals',
                                    'mean_nan', 'interval', 'against', 'ranking', 'witness', 'weights', 'replay',
                                    'script', 'learning'])
def test_a_malformed_frozen_start_result_is_refused(tmp_path, damage):
    payload, record = payload_and_record(tmp_path)
    bad = copy.deepcopy(payload)
    goal = 'elbow_flex_witness'
    if damage == 'mode':
        bad['mode'] = 'frozen_start'                         # the command asks for the anchored mode
    elif damage == 'probe':
        bad['probe'] = 'fixed_start_body_goals_v1'
    elif damage == 'set':
        bad['frozen_set']['sha256'] = '6' * 64
    elif damage == 'anchor':
        bad['anchor']['sha256'] = '7' * 64
    elif damage == 'anchor_missing':
        del bad['anchor']
    elif damage == 'starts':
        bad['starts_equal_to_set'] = False
    elif damage == 'bodies':
        bad['bodies'] = 7
    elif damage == 'goals':
        bad['goals'] = bad['goals'][:2]
    elif damage == 'mean_nan':
        bad['summary']['wrist_goal_error_cm']['means']['synergy'] = float('nan')
    elif damage == 'interval':
        bad['summary']['physical_goal_error']['paired_bootstrap95'] = [0.1]
    elif damage == 'against':
        bad['summary']['physical_goal_error']['against_controller']['direct']['paired_bootstrap95'] = None
    elif damage == 'ranking':
        del bad['ranking']
    elif damage == 'witness':
        del bad['by_goal'][goal]['witness_check']
    elif damage == 'weights':
        bad['weights_unchanged'] = False
    elif damage == 'replay':
        bad['exact_replay'] = False
    elif damage == 'script':
        bad['script_sha256'] = 'another script'
    else:
        bad['learning'] = True
    with pytest.raises(ValueError):
        worker.validate_result(bad, record)


def test_a_plain_frozen_start_command_refuses_an_anchored_result(tmp_path):
    payload, _ = payload_and_record(tmp_path, anchored=True)
    _, plain = payload_and_record(tmp_path, anchored=False)
    with pytest.raises(ValueError, match='mode'):
        worker.validate_result(payload, plain)


def test_a_malformed_frozen_start_result_fails_the_attempt_instead_of_stopping_the_cycle(tmp_path):
    payload, record = payload_and_record(tmp_path)
    out = tmp_path / 'attempt'
    out.mkdir()
    for damage in ('summary', 'by_goal', 'ranking', 'frozen_set', 'anchor', 'gain'):
        bad = dict(payload, **{damage: ['not', 'an', 'object']})
        (out / 'result.json').write_text(json.dumps(bad), encoding='utf-8')
        got = worker.finished_result(dict(record, output=str(out)))
        assert got['status'] == 'failed' and 'Unreadable or mismatched result' in got['reason'], damage
    (out / 'result.json').write_text(json.dumps(payload), encoding='utf-8')
    got = worker.finished_result(dict(record, output=str(out)))
    assert got['status'] == 'complete' and got['line_met'] is False


def test_the_declared_line_of_a_frozen_start_result_is_the_direct_arms_gain_line(tmp_path):
    """The gain rule: the line is the direct arm's against the same arm of the bound reference, never the planner-against-
    controller interval the body comparison reads (before October 18, 2026, 7:38 PM PDT the two were one line)."""
    payload, record = payload_and_record(tmp_path, upper=-.01, direct_upper=.03)
    assert worker.declared_line(SCRIPT_NAME, payload) is False
    assert worker.declared_line('compare_hand_body_goals_2026-09-24.py', payload) is True    # the normal mode's own line
    payload, record = payload_and_record(tmp_path, upper=.15, direct_upper=-.01)
    worker.validate_result(payload, record)
    assert worker.declared_line(SCRIPT_NAME, payload) is True
    invalid = copy.deepcopy(payload)
    invalid['gain'].update(status='invalid', invalid_reasons=['the reference: 1 of the 40 declared trials missing'],
                           line_met=None, arms=None)
    assert worker.declared_line(SCRIPT_NAME, invalid) is None
    assert worker.declared_line(SCRIPT_NAME, dict(payload, gain=None)) is None


def test_the_production_manifest_runs_the_anchored_frozen_start_task():
    manifest = json.loads((REPO / 'docs/gain_of_function_v3/ability_rechecks.json').read_text(encoding='utf-8'))
    tasks = {t['id']: t for t in manifest['tasks']}
    task, body = tasks['fixed_start_body_and_synergy'], tasks['body_and_synergy']
    assert task['script'] == 'docs/gain_of_function_v3/' + SCRIPT_NAME
    assert set(task['path_arguments']) == {'--frozen-set-receipt', '--anchor', '--reference-receipt'}
    assert worker.task_class(task) == worker.task_class(body) and task['every_steps'] == body['every_steps']
    assert task['triggers']['code'] == body['triggers']['code'] and task['triggers']['flags'] is True
    assert task['memory_gib'] >= body['memory_gib']
    assert set(task['path_arguments'].values()) <= set(task['depends'])
    assert 'docs/gain_of_function_v3/compare_hand_body_goals_2026-09-24.py' in task['depends']
    worker.plain_arguments(task)
    worker.path_arguments(REPO, task)
    worker.dependency_paths(REPO, task)
    worker.validate_schedule(REPO, task, list(tasks), manifest['ride_along_steps'])
    command = worker.task_command(REPO, task, Path('checkpoint'), Path('out'))
    assert command[2].endswith(SCRIPT_NAME) and '--anchor' in command and '--frozen-set-receipt' in command
    assert '--reference-receipt' in command and '--reference-receipt-sha256' in command
    # the anchor was made from the frozen set the receipt names
    receipt = json.loads((REPO / task['path_arguments']['--frozen-set-receipt']).read_text(encoding='utf-8'))
    anchor = json.loads((REPO / task['path_arguments']['--anchor']).read_text(encoding='utf-8'))
    assert anchor['frozen_set']['sha256'] == receipt['set_sha256'] and len(anchor['bodies']) == receipt['bodies']


def test_the_companion_substitution_changes_only_the_frozen_fields():
    F = _load_script()
    n = 8

    def world(b, tag):
        return dict(env_kwargs=dict(tag=tag), world=dict(qpos=np.full(146, b + (0.5 if tag == 'set' else 0.0))))
    episode = dict(version='x', worlds=[world(b, 'own') for b in range(n)], config={},
                   learner=dict(data={'_prev_mfa': torch.zeros(n, 26, 9), '_hand_previous_joints': torch.zeros(n, 44),
                                      '_last_smooth_prev': torch.zeros(n, 26, 9), '_smooth_exec_prev': None,
                                      'context_buffer': torch.full((n, 5, 3), 7.0)},
                                objects={'replay': {'r': 1}}),
                   trainer=dict(step=483411, prev_executed=np.zeros((n, 26, 9), np.float32), prev_order_id=11,
                                held_motor=np.zeros((n, 26, 9), np.float32),
                                held_hand_joint_goal=np.zeros((n, 44), np.float32),
                                frame_meta=[dict(who='own') for _ in range(n)]))
    frozen_set = dict(n_bodies=n, worlds=[world(b, 'set') for b in range(n)], frame_meta=[dict(who='set') for _ in range(n)],
                      trainer_carry=dict(prev_executed=np.ones((n, 26, 9), np.float32), prev_order_id=98805,
                                         held_motor=np.ones((n, 26, 9), np.float32),
                                         held_hand_joint_goal=np.ones((n, 44), np.float32)),
                      learner_carry={'_prev_mfa': torch.ones(n, 26, 9), '_hand_previous_joints': torch.ones(n, 44),
                                     '_last_smooth_prev': torch.ones(n, 26, 9), '_smooth_exec_prev': None},
                      learner_carry_present={k: True for k in F.LEARNER_CARRY})
    mixed = F.frozen_episode(episode, frozen_set)
    part = F.H.episode_body(mixed, 3)
    assert part['world']['env_kwargs']['tag'] == 'set' and part['frame_meta']['who'] == 'set'
    assert part['trainer']['prev_order_id'] == 98805 and part['trainer']['step'] == 483411
    assert torch.equal(part['learner']['data']['_prev_mfa'], torch.ones(1, 26, 9))
    assert torch.equal(part['learner']['data']['context_buffer'], torch.full((1, 5, 3), 7.0))
    assert episode['worlds'][3]['env_kwargs']['tag'] == 'own' and episode['trainer']['prev_order_id'] == 11


def test_the_equality_comparison_catches_one_value_and_skips_timing_and_mode_keys(tmp_path):
    F = _load_script()
    rows = [dict(body=b, goal=g, outcomes=dict(controller=dict(qpos=[0.1 * b, 0.2], physical_goal_error=0.3)),
                 elapsed_s=1.0 + b, inverse_projection=dict(calls=b), mode='frozen_start')
            for b in range(2) for g in ('curl_witness', 'extension_witness')]

    def write(folder, rows, mode):
        folder.mkdir()
        (folder / 'trials.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        (folder / 'result.json').write_text(json.dumps(dict(mode=mode, bodies=2, checkpoint='c', checkpoint_step=1)),
                                            encoding='utf-8')
    write(tmp_path / 'a', rows, 'frozen_start')
    changed = copy.deepcopy(rows)
    for r in changed:
        r.update(mode='frozen_start_anchored', elapsed_s=99.0, anchor=dict(sha256='x'))
    write(tmp_path / 'b', changed, 'frozen_start_anchored')
    same = F.compare_runs(tmp_path / 'b', tmp_path / 'a', 'equality')
    assert same['verdict'] == 'pass' and same['values_differing'] == 0 and 'mode' in same['excluded_keys']
    changed[2]['outcomes']['controller']['qpos'][1] += 1e-12
    (tmp_path / 'b' / 'trials.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in changed), encoding='utf-8')
    one = F.compare_runs(tmp_path / 'b', tmp_path / 'a', 'equality')
    assert one['verdict'] == 'fail' and one['values_differing'] == 1
    assert list(one['differing_keys']) == ['outcomes.controller.qpos']


def test_the_per_process_model_identity_is_not_compared_and_the_model_content_is(tmp_path):
    """Row fixed_start_anchored_equality_494025 (October 18, 2026) read FAIL on 120 values, all in world_model.file,
    world_model.exact_sha256 and world_model.gzip_sha256: the compiled crib model stores its texture paths, which lie in the
    probe harness's per-process scene folder gof_probes/_scenes/p<process id>. Those keys are never compared; the structure
    fingerprint is, and the saved model files are compared by their bytes with the scene folder replaced."""
    import gzip
    F = _load_script()
    model = b'header' + b'C:/x/Saved/mimo_bridge/gof_probes/_scenes/p%d/pen_art/play_mat.png\x00' + b'tail'

    def write(folder, pid, fingerprint, extra=b''):
        folder.mkdir()
        (folder / 'world_models').mkdir()
        raw = (model % pid) + extra
        name = 'world_models/%s_%s.mjb.gz' % (fingerprint[:8], hashlib.sha256(raw).hexdigest()[:12])
        (folder / name).write_bytes(gzip.compress(raw))
        wm = dict(file=name, structure_fingerprint=fingerprint, exact_sha256=hashlib.sha256(raw).hexdigest(),
                  bytes=len(raw), gzip_sha256='g%d' % pid)
        rows = [dict(body=0, goal='curl_witness', outcomes=dict(controller=dict(qpos=[0.5])), world_model=wm,
                     mode='frozen_start')]
        (folder / 'trials.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        (folder / 'result.json').write_text(json.dumps(dict(mode='frozen_start', bodies=1, checkpoint='c')),
                                            encoding='utf-8')
    write(tmp_path / 'a', 12776, 'f' * 64)
    write(tmp_path / 'b', 26548, 'f' * 64)
    same = F.compare_runs(tmp_path / 'b', tmp_path / 'a', 'equality')
    assert same['verdict'] == 'pass' and same['values_differing'] == 0, same['differing_keys']
    assert same['world_model_content'] == dict(same['world_model_content'], trials_compared=1, trials_differing=0)
    assert set(same['excluded_process_keys']) == {'world_model.file', 'world_model.exact_sha256', 'world_model.gzip_sha256'}
    write(tmp_path / 'c', 26548, 'e' * 64)                  # another structure: compared and caught
    other = F.compare_runs(tmp_path / 'c', tmp_path / 'a', 'equality')
    assert other['verdict'] == 'fail' and list(other['differing_keys']) == ['world_model.structure_fingerprint']
    write(tmp_path / 'd', 26548, 'f' * 64, extra=b'one more byte')   # other model content: caught from the files
    content = F.compare_runs(tmp_path / 'd', tmp_path / 'a', 'equality')
    assert content['verdict'] == 'fail' and content['world_model_content']['trials_differing'] == 1
    assert F.model_content_sha256(model % 1) == F.model_content_sha256(model % 99999)


# ------------------------------------------------------------------------------------------ the gain rule
GOALS = list(worker.BODY_GOAL_SUITES['hand_and_body'])
NAN = float('nan')


@functools.lru_cache(maxsize=1)
def _script():
    return _load_script()


def base(arm, body, goal):
    """The physical goal error of every arm in a run that equals the reference."""
    return 1.0 + 0.1 * body + 0.01 * GOALS.index(goal)


def fake_runs(tmp_path, anchor_sha='a' * 64, set_sha='5' * 64):
    """Completed anchored runs of the fixed-start script as their saved outputs (rows, result, status, declaration, world
    models, hand geometry), every mandatory value equal from run to run: run(name, step, errors, tweak, extra) writes one.
    errors(arm, body, goal) is the physical goal error (hand joint error 2x and wrist error 3x it); tweak(rows, result)
    damages it; extra changes its saved world model's bytes."""
    F = _script()
    receipt = tmp_path / 'set_receipt.json'
    receipt.write_text(json.dumps(dict(set_sha256=set_sha, bodies=8, goals=GOALS)), encoding='utf-8')
    names = ['handle_%d' % i for i in range(25)] + ['gaze']

    def command(strength):
        return [[[0.01 * i, 0.02, 0.03, 0., 0., 0., 1., 0.05, strength] for i in range(26)]]

    def run(name, step, errors=base, tweak=None, extra=b'', pid=100):
        folder = tmp_path / name
        (folder / 'world_models').mkdir(parents=True)
        raw = (b'model C:/x/gof_probes/_scenes/p%d/pen_art/mat.png\x00 tail' % pid) + extra
        file = 'world_models/%s_%s.mjb.gz' % ('f' * 32, hashlib.sha256(raw).hexdigest()[:12])
        (folder / file).write_bytes(gzip.compress(raw))
        (folder / 'hand_geometry.json').write_text(json.dumps(dict(
            handle_names=names, hands=[dict(finger_indices=list(range(15, 20))), dict(finger_indices=list(range(20, 25)))])),
            encoding='utf-8')
        strength, eye = (0.5, 0.1) if step == 494025 else (0.6, 0.2)       # each checkpoint's own strengths and eye command
        rows = []
        for b in range(8):
            for j, g in enumerate(GOALS):
                outcomes = {arm: dict(physical_goal_error=errors(arm, b, g), hand_joint_error_deg=2 * errors(arm, b, g),
                                      wrist_goal_error_cm=3 * errors(arm, b, g), canonical_command=command(strength))
                            for arm in ('controller', 'direct', 'synergy')}
                rows.append(dict(
                    body=b, goal=g, outcomes=outcomes, initial_qpos=[0.1 * b, 0.2, 0.3], initial_qvel=[0.01 * b, 0.],
                    start_check=dict(body=b, sim_time=24701.75, set_sim_time=24701.75, qpos_equal_to_set=True),
                    initial_pose=[[0.5, b]], initial_hand_joint_angles_rad=[0.1] * 4,
                    initial_hand_joint_velocity_rad_s=[0.] * 4, previous_hand_joint_goal=None,
                    frozen_set=dict(path='D:/set.pt.gz', sha256=set_sha),
                    world_model=dict(file=file, structure_fingerprint='f' * 64, exact_sha256=hashlib.sha256(raw).hexdigest(),
                                     bytes=len(raw), gzip_sha256='g%d' % pid),
                    witness=dict(pose=[[b, j]], joint_angles_rad=[0.2, j]),
                    anchor=dict(sha256=anchor_sha, command_sha256='c%d' % b, body=b),
                    anchor_offset=dict(largest_coordinate_cm=float(step % 7), largest_turn_deg=1., handles_outside_region=0,
                                       handles=15),
                    controller_raw_command=command(strength), gaze_rad=[eye, 0.], smoothing=0.35, mode=F.ANCHORED_MODE))
        result = dict(checkpoint='C:/run/checkpoints/ckpt_step_%d' % step, checkpoint_step=step, mode=F.ANCHORED_MODE,
                      probe=F.PROBE_VERSION, bodies=8, goals=GOALS, exact_replay=True, weights_unchanged=True,
                      starts_equal_to_set=True, frozen_set=dict(sha256=set_sha, receipt=str(receipt), reference_step=494025),
                      anchor=dict(sha256=anchor_sha), script_sha256='script', source_sha256={'x.py': '1'},
                      normal_script=dict(sha256='n'), contract='c', hold_seconds=.25, learning=False)
        if tweak:
            tweak(rows, result)
        (folder / 'trials.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        for name_, value in (('result.json', result), ('declaration.json', result),
                             ('status.json', dict(state='completed', completed=len(rows), total=len(rows)))):
            (folder / name_).write_text(json.dumps(value), encoding='utf-8')
        return folder
    return F, run


def gain_of(F, this, reference):
    return F.compare_runs(this, reference, 'checkpoints')['gain']


@pytest.mark.parametrize('damage', ['qvel_missing', 'qvel_unequal', 'qvel_none', 'sim_time_missing', 'world_content_unequal',
                                    'world_file_missing', 'world_structure_unequal', 'recorded_content_disagrees'])
def test_a_missing_or_unequal_velocity_or_world_identity_is_an_invalid_comparison(tmp_path, damage):
    F, run = fake_runs(tmp_path)
    reference = run('reference', 494025)
    tweak, extra = None, b''
    if damage == 'qvel_missing':
        tweak = lambda rows, result: rows[3].pop('initial_qvel')
    elif damage == 'qvel_unequal':
        tweak = lambda rows, result: rows[3]['initial_qvel'].__setitem__(0, rows[3]['initial_qvel'][0] + 1e-12)
    elif damage == 'qvel_none':
        tweak = lambda rows, result: rows[3].__setitem__('initial_qvel', None)
    elif damage == 'sim_time_missing':
        tweak = lambda rows, result: rows[3]['start_check'].pop('sim_time')
    elif damage == 'world_content_unequal':
        extra = b'one more byte'
    elif damage == 'world_structure_unequal':
        tweak = lambda rows, result: rows[0]['world_model'].__setitem__('structure_fingerprint', 'e' * 64)
    elif damage == 'recorded_content_disagrees':
        tweak = lambda rows, result: rows[0]['world_model'].__setitem__('content_sha256', '0' * 64)
    this = run('this', 506025, tweak=tweak, extra=extra, pid=200)
    if damage == 'world_file_missing':
        for path in (this / 'world_models').iterdir():
            path.unlink()
    gain = gain_of(F, this, reference)
    assert gain['status'] == 'invalid' and gain['line_met'] is None and gain['arms'] is None, gain['invalid_reasons']
    words = {'qvel_missing': 'initial_qvel in this run', 'qvel_unequal': 'initial_qvel differs', 'qvel_none': 'initial_qvel',
             'sim_time_missing': 'start_check.sim_time', 'world_content_unequal': 'world model content differs',
             'world_file_missing': 'no readable saved model file', 'world_structure_unequal': 'structure_fingerprint differs',
             'recorded_content_disagrees': 'recorded content_sha256 differs'}[damage]
    assert any(words in reason for reason in gain['invalid_reasons']), gain['invalid_reasons']
    comparison = F.compare_runs(this, reference, 'checkpoints')
    assert comparison['verdict'] == 'invalid' and comparison['compatibility']['verdict'] == 'fail'


def test_a_saved_model_file_gone_with_its_content_recorded_is_read_from_the_record(tmp_path):
    """Validate from existing saved records where available: the row's own content_sha256 stands in for a file that is gone;
    with neither the comparison is invalid (the test above)."""
    F, run = fake_runs(tmp_path)
    reference = run('reference', 494025)
    raw = b'model C:/x/gof_probes/_scenes/p<process>/pen_art/mat.png\x00 tail'
    recorded = hashlib.sha256(raw).hexdigest()
    this = run('this', 506025, tweak=lambda rows, result: [r['world_model'].__setitem__('content_sha256', recorded)
                                                            for r in rows], pid=200)
    for path in (this / 'world_models').iterdir():
        path.unlink()
    assert gain_of(F, this, reference)['status'] == 'valid'


@pytest.mark.parametrize('damage', ['twice_here', 'twice_in_reference', 'missing_here', 'missing_in_reference', 'outside',
                                    'subset_of_bodies'])
def test_a_missing_or_repeated_trial_is_an_invalid_comparison_before_any_averaging(tmp_path, damage):
    F, run = fake_runs(tmp_path)
    here = there = None
    if damage == 'twice_here':
        here = lambda rows, result: rows.append(copy.deepcopy(rows[0]))
    elif damage == 'twice_in_reference':
        there = lambda rows, result: rows.append(copy.deepcopy(rows[7]))
    elif damage == 'missing_here':
        here = lambda rows, result: rows.pop(7)
    elif damage == 'missing_in_reference':       # the earlier detection looked one way only: from the other run's trials
        there = lambda rows, result: rows.pop(12)
    elif damage == 'outside':
        here = lambda rows, result: rows.append(dict(copy.deepcopy(rows[0]), goal='unknown_witness'))
    else:                                        # a run of 2 bodies is not the declared full body count
        def here(rows, result):
            del rows[10:]
            result['bodies'] = 2
    reference = run('reference', 494025, tweak=there)
    this = run('this', 506025, tweak=here, pid=200)
    gain = gain_of(F, this, reference)
    assert gain['status'] == 'invalid' and gain['line_met'] is None and gain['arms'] is None, gain['invalid_reasons']
    words = {'twice_here': 'this run: 1 trials held more than once', 'twice_in_reference': 'the reference: 1 trials held',
             'missing_here': 'this run: 1 of the 40 declared trials missing',
             'missing_in_reference': 'the reference: 1 of the 40 declared trials missing',
             'outside': 'trials outside the declared', 'subset_of_bodies': 'this run ran 2 bodies'}[damage]
    assert any(words in reason for reason in gain['invalid_reasons']), gain['invalid_reasons']
    if damage in ('twice_here', 'missing_in_reference'):       # the equality kind refuses them too
        assert F.compare_runs(this, reference, 'equality')['verdict'] == 'fail'


@pytest.mark.parametrize('damage', ['metric_nan_here', 'metric_inf_in_reference', 'controller_metric_missing',
                                    'qpos_nan_on_both_sides'])
def test_a_nonfinite_value_is_an_invalid_comparison(tmp_path, damage):
    F, run = fake_runs(tmp_path)
    here = there = None
    if damage == 'metric_nan_here':
        here = lambda rows, result: rows[5]['outcomes']['direct'].__setitem__('physical_goal_error', NAN)
    elif damage == 'metric_inf_in_reference':
        there = lambda rows, result: rows[5]['outcomes']['synergy'].__setitem__('wrist_goal_error_cm', float('inf'))
    elif damage == 'controller_metric_missing':
        here = lambda rows, result: rows[5]['outcomes']['controller'].pop('hand_joint_error_deg')
    else:                     # JSON reads every NaN as one object, so lists holding it compare equal: refused all the same
        def here(rows, result):
            rows[2]['initial_qpos'][0] = NAN
        there = here
    reference = run('reference', 494025, tweak=there)
    this = run('this', 506025, tweak=here, pid=200)
    gain = gain_of(F, this, reference)
    assert gain['status'] == 'invalid' and gain['line_met'] is None, gain['invalid_reasons']
    assert any(('not finite' in reason) for reason in gain['invalid_reasons']), gain['invalid_reasons']


@pytest.mark.parametrize('change,line,reading', [(0., False, 'no clear change'), (-.05, True, 'gain'),
                                                 (.05, False, 'worsening')])
def test_equal_better_and_worse_intervals(tmp_path, change, line, reading):
    """A valid comparison: line_met only for a direct interval below 0; an interval that includes 0 (equal runs: [0, 0]) is
    not met and not invalid; an interval above 0 is a worsening, not a gain."""
    F, run = fake_runs(tmp_path)
    reference = run('reference', 494025)
    this = run('this', 506025, errors=lambda arm, b, g: base(arm, b, g) + (change * (1 + 0.1 * b) if arm != 'controller'
                                                                           else 0.), pid=200)
    comparison = F.compare_runs(this, reference, 'checkpoints')
    gain = comparison['gain']
    assert gain['status'] == 'valid' and gain['invalid_reasons'] == [] and gain['line_met'] is line
    for arm in ('direct', 'synergy'):
        block = gain['arms'][arm]
        assert block['interval_reading'] == reading and block['line_met'] is line and block['bodies'] == 8
        assert set(block['per_goal']) == set(GOALS) and len(block['per_body']) == 8
        assert block['hand_joint_error_deg']['interval_reading'] == reading
        assert block['wrist_goal_error_cm']['interval_reading'] == reading
    low, high = gain['arms']['direct']['paired_bootstrap95']
    assert (high < 0) is line and (change != 0 or low == high == 0.)
    assert comparison['verdict'] == ('pass' if line else 'fail') and comparison['compatibility']['verdict'] == 'pass'
    direct = next(r for r in comparison['readings'] if r['name'].startswith('direct arm (the primary arm'))
    assert direct['statistically_clear'] == ('yes' if line else 'no')
    # the changed controls: each checkpoint's own strengths (0.6 against 0.5) and eye command (0.2 against 0.1 radian)
    controls = gain['changed_controls']['summary']
    assert controls['bodies_with_search_start_strengths_differing'] == 8
    assert abs(controls['largest_executed_strength_difference'] - 0.1) < 1e-12
    assert abs(controls['largest_eye_command_difference_deg'] - np.degrees(0.1)) < 1e-9
    assert 'Whole-arm checkpoint performance' in gain['label'] and 'does not isolate' in gain['label']
    # the reference compared with itself: valid, zero change, not met
    itself = gain_of(F, reference, reference)
    assert itself['status'] == 'valid' and itself['line_met'] is False
    assert itself['arms']['direct']['paired_bootstrap95'] == [0., 0.]
    assert itself['changed_controls']['summary']['bodies_with_eye_command_differing'] == 0


def test_the_direct_arm_alone_decides_the_line(tmp_path):
    """Synergy is reported with the same arithmetic and cannot rescue a direct arm that does not meet the line."""
    F, run = fake_runs(tmp_path)
    reference = run('reference', 494025)
    synergy_better = run('synergy_better', 506025, errors=lambda arm, b, g: base(arm, b, g) - (.05 if arm == 'synergy'
                                                                                                else 0.), pid=200)
    gain = gain_of(F, synergy_better, reference)
    assert gain['primary_arm'] == 'direct' and gain['arms']['synergy']['line_met'] is True
    assert gain['arms']['direct']['line_met'] is False and gain['line_met'] is False
    direct_better = run('direct_better', 506025, errors=lambda arm, b, g: base(arm, b, g) + {
        'direct': -.05, 'synergy': .05}.get(arm, 0.), pid=300)
    gain = gain_of(F, direct_better, reference)
    assert gain['line_met'] is True and gain['arms']['synergy']['interval_reading'] == 'worsening'


def test_the_first_step_floor_holds_for_triggers_and_ride_along_selection():
    """first_step 506,025 (494,025 + 12,000; the gain rule's schedule): a fired trigger moves the task neither to a run step
    nor to a pause step before it, and no checkpoint below it is due, the window's own included (the scheduler's window
    lists a task only when its run step is at or below the window's checkpoint step: cycle, runnable)."""
    manifest = json.loads((REPO / 'docs/gain_of_function_v3/ability_rechecks.json').read_text(encoding='utf-8'))
    task = {t['id']: t for t in manifest['tasks']}['fixed_start_body_and_synergy']
    assert task['first_step'] == 506025 and task['every_steps'] == 12000 and worker.first_step(task) == 506025
    assert worker.HOLD_FEATURE not in task and 'first_step_under_hold' not in task
    context = dict(root=REPO, head=None, flags=None, gauges={}, live_step=495000,
                   ride_along_steps=manifest['ride_along_steps'])
    assert worker.schedule(task, {}, context, {}) == (506025, 506025, [])
    record = dict(completed_step=480000)         # a round below the floor, then a listed line crossed since
    tasks = dict(body_and_synergy=dict(title='Body', line_changed=dict(step=495000, before=True, after=False)),
                 fixed_start_body_and_synergy=record)
    run, opened, reasons = worker.schedule(task, record, context, tasks)
    assert reasons and run == 506025 and opened == 506025
    candidates = [(500100, 'a'), (505975, 'b'), (506050, 'c')]
    assert worker.due_checkpoint(task, record, candidates, threshold=run) == (506050, 'c')
    assert worker.due_checkpoint(task, record, [(505975, 'b')], threshold=run) is None
    assert worker.due_checkpoint(task, {}, candidates) == (506050, 'c')


def test_the_reference_is_bound_by_its_receipts_sha256(tmp_path):
    """The receipt names every file a comparison reads with its SHA-256 and is itself named by its SHA-256 (the manifest's
    plain argument): another SHA-256, a changed file, a comparison with another folder, or a result naming another
    reference is refused; a result the script computed against the bound receipt passes the scheduler's validator, and its
    line is the gain line."""
    (tmp_path / 'payload').mkdir()
    payload, record = payload_and_record(tmp_path / 'payload', direct_upper=.03)
    F, run = fake_runs(tmp_path, anchor_sha=payload['anchor']['sha256'])
    reference = run('reference', 494025)
    target = tmp_path / 'reference_receipt.json'
    receipt, digest = F.make_reference(SimpleNamespace(reference_run=reference, reference_out=target,
                                                       reference_evidence='a test reference'))
    assert receipt['self_comparison']['status'] == 'valid' and receipt['self_comparison']['line_met'] is False
    with pytest.raises(RuntimeError, match='never overwritten'):
        F.make_reference(SimpleNamespace(reference_run=reference, reference_out=target, reference_evidence='again'))
    bound = F.load_reference(target, digest)
    with pytest.raises(RuntimeError, match='declared SHA-256'):
        F.load_reference(target, '0' * 64)
    this = run('this', 506025, errors=lambda arm, b, g: base(arm, b, g) - (.05 if arm == 'direct' else 0.), pid=200)
    other = run('other_reference', 494025, pid=300)
    args = SimpleNamespace(out=this, compare_with=[other], compare_kind=['checkpoints'], compare_name=['gain.json'],
                           reference_receipt=target, reference_receipt_sha256=digest)
    with pytest.raises(RuntimeError, match='not with the bound reference'):
        F.write_comparisons(args, tmp_path / 'refused')
    comparison = F.compare_runs(this, reference, 'checkpoints', reference=bound)
    gain = comparison['gain']
    assert gain['status'] == 'valid' and gain['line_met'] is True and gain['reference']['receipt']['sha256'] == digest
    # the scheduler's validator: the command names the receipt and its SHA-256
    command = [c for c in record['command']]
    command[command.index('--reference-receipt') + 1] = str(target)
    command[command.index('--reference-receipt-sha256') + 1] = digest
    payload.update(checkpoint=str(Path(json.loads((this / 'result.json').read_text())['checkpoint'])), gain=gain)
    good = dict(record, command=command, checkpoint=payload['checkpoint'])
    worker.validate_result(json.loads(json.dumps(payload)), good)
    assert worker.declared_line(SCRIPT_NAME, payload) is True
    for damage in ('declared_sha', 'named_result', 'named_receipt', 'other_checkpoint'):
        bad, cmd = copy.deepcopy(payload), list(command)
        if damage == 'declared_sha':
            cmd[cmd.index('--reference-receipt-sha256') + 1] = '0' * 64
        elif damage == 'named_result':
            bad['gain']['reference']['sha256'] = '0' * 64
        elif damage == 'named_receipt':
            bad['gain']['reference']['receipt']['sha256'] = '0' * 64
        else:
            bad['gain']['this']['checkpoint'] = 'C:/run/checkpoints/ckpt_step_1'
        with pytest.raises(ValueError):
            worker.validate_result(bad, dict(good, command=cmd))
    # an invalid comparison is refused by the validator, so the attempt fails (INVALID in its row), with its reasons
    invalid = copy.deepcopy(payload)
    invalid['gain'] = F.invalid_gain(gain['reference'], json.loads((this / 'result.json').read_text()), {},
                                     ['the reference: 1 of the 40 declared trials missing'])
    invalid['gain']['this']['checkpoint'] = payload['checkpoint']
    with pytest.raises(ValueError, match='1 of the 40 declared trials missing'):
        worker.validate_result(invalid, good)
    # a file the receipt names that changed afterwards
    (reference / 'trials.jsonl').write_text((reference / 'trials.jsonl').read_text() + '\n', encoding='utf-8')
    with pytest.raises(RuntimeError, match='trials.jsonl'):
        F.load_reference(target, digest)


def test_the_production_manifest_binds_the_accepted_reference_by_sha256():
    manifest = json.loads((REPO / 'docs/gain_of_function_v3/ability_rechecks.json').read_text(encoding='utf-8'))
    task = {t['id']: t for t in manifest['tasks']}['fixed_start_body_and_synergy']
    path = REPO / task['path_arguments']['--reference-receipt']
    assert task['arguments'] == ['--reference-receipt-sha256', hashlib.sha256(path.read_bytes()).hexdigest()]
    assert task['path_arguments']['--reference-receipt'] in task['depends']
    receipt = json.loads(path.read_text(encoding='utf-8'))
    assert receipt['version'] == 'fixed_probe_reference_v1' and receipt['rule_version'] == worker.FIXED_START_GAIN_RULE
    assert receipt['checkpoint_step'] == 494025 and receipt['mode'] == 'frozen_start_anchored' and receipt['bodies'] == 8
    set_receipt = json.loads((REPO / task['path_arguments']['--frozen-set-receipt']).read_text(encoding='utf-8'))
    anchor = REPO / task['path_arguments']['--anchor']
    assert receipt['frozen_set']['sha256'] == set_receipt['set_sha256']
    assert receipt['anchor']['sha256'] == hashlib.sha256(anchor.read_bytes()).hexdigest()
    F = _script()
    assert (F.GAIN_RULE, F.PRIMARY_ARM, F.GAIN_ARMS) == (worker.FIXED_START_GAIN_RULE, worker.FIXED_START_PRIMARY_ARM,
                                                       worker.FIXED_START_GAIN_ARMS)
    folder = Path(receipt['run_folder'])
    if not folder.is_dir():
        pytest.skip('The bound reference run folder is not on this machine: %s' % folder)
    F.load_reference(path, hashlib.sha256(path.read_bytes()).hexdigest())     # every named file still hashes as recorded


def frozen_run_inputs(tmp_path):
    """What frozen_run reads before and around its model load, written here: a frozen set file (8 starts, no rows) and its
    receipt, an anchor of it, and two bound references (receipt path and SHA-256): 'same' ran under this run's search
    conditions, 'other' under another frozen set. args(name, out) is the run's command line."""
    F = _script()
    set_path = tmp_path / 'set.pt.gz'
    starts = [dict(qpos=[0.1 * b, 0.2, 0.3], qvel=[0.01 * b, 0.], sim_time=24701.75, frame=0) for b in range(8)]
    with gzip.open(set_path, 'wb') as stream:
        torch.save(dict(version=F.SET_VERSION, goal_suite=F.GOAL_SUITE, goals=GOALS, n_bodies=8, reference_rows=[],
                        starts=starts, reference_checkpoint='C:/run/checkpoints/ckpt_step_494025', reference_step=494025,
                        made_at='x'), stream)
    set_sha = hashlib.sha256(set_path.read_bytes()).hexdigest()
    set_receipt = tmp_path / 'set_receipt_of_the_run.json'
    set_receipt.write_text(json.dumps(dict(version=F.SET_VERSION, set_path=str(set_path), set_sha256=set_sha, bodies=8,
                                           goals=GOALS)), encoding='utf-8')
    proposal = np.zeros((26, 9), np.float32)            # 26 x 9, not the proposal's 1 x 26 x 9: the run stops on it
    anchor = tmp_path / 'anchor.json'
    anchor.write_text(json.dumps(dict(
        version=F.ANCHOR_VERSION, frozen_set=dict(sha256=set_sha), reference_checkpoint='C:/run/checkpoints/ckpt_step_494025',
        reference_step=494025, made_at='x',
        bodies=[dict(body=b, controller_command=proposal.tolist(),
                     command_sha256=hashlib.sha256(proposal.tobytes()).hexdigest()) for b in range(8)])), encoding='utf-8')
    anchor_sha = hashlib.sha256(anchor.read_bytes()).hexdigest()
    for name in ('same', 'other'):
        (tmp_path / name).mkdir()
    _, run = fake_runs(tmp_path / 'same', anchor_sha=anchor_sha, set_sha=set_sha)
    _, other_run = fake_runs(tmp_path / 'other', anchor_sha=anchor_sha)          # another frozen set
    receipts = {}
    for name, maker in (('same', run), ('other', other_run)):
        target = tmp_path / ('%s_reference.json' % name)
        receipts[name] = (target, F.make_reference(SimpleNamespace(reference_run=maker('reference', 494025),
                                                                   reference_out=target, reference_evidence='test'))[1])

    def args(name, out):
        return SimpleNamespace(checkpoint=Path('C:/run/checkpoints/ckpt_step_506025'), out=tmp_path / out,
                               frozen_set_receipt=set_receipt, anchor=anchor, bodies=None, compare_with=None,
                               compare_kind=None, compare_name=None, reference_receipt=receipts[name][0],
                               reference_receipt_sha256=receipts[name][1])
    return SimpleNamespace(F=F, args=args, receipts=receipts)


def test_a_frozen_start_run_checks_and_names_its_reference_before_any_model_is_loaded(tmp_path, monkeypatch):
    """frozen_run with --reference-receipt: a reference of other search conditions refuses the run before its output folder
    exists; a matching one is named in the declaration (gain_rule and the reference record) written before the model load,
    which is replaced here by a stop, so no model or world is loaded. The declaration records the SHA-256 of every depends
    and code-trigger file of the script's manifest entry, read from the manifest."""
    from unittest import mock
    inputs = frozen_run_inputs(tmp_path)
    F = inputs.F

    class Stop(Exception):
        pass

    def stop(*args, **kwargs):
        raise Stop
    monkeypatch.setattr(F, 'guard', lambda need: None)
    monkeypatch.setattr(F, 'load_model_and_frozen_episode', stop)
    try:
        with pytest.raises(RuntimeError, match='names other search conditions'):
            F.frozen_run(inputs.args('other', 'refused'))
        assert not (tmp_path / 'refused').exists()
        with pytest.raises(Stop):
            F.frozen_run(inputs.args('same', 'run'))
    finally:
        mock.patch.stopall()                     # the run's own timing patch, left started by the stop
    declaration = json.loads((tmp_path / 'run' / 'declaration.json').read_text(encoding='utf-8'))
    assert declaration['gain_rule'] == F.GAIN_RULE and declaration['mode'] == F.ANCHORED_MODE
    assert declaration['reference']['receipt']['sha256'] == inputs.receipts['same'][1]
    assert declaration['reference']['checkpoint_step'] == 494025
    manifest = json.loads((REPO / 'docs/gain_of_function_v3/ability_rechecks.json').read_text(encoding='utf-8'))
    task = {t['id']: t for t in manifest['tasks']}['fixed_start_body_and_synergy']
    named = set(task['depends']) | set(task['triggers']['code'])
    recorded = {key.replace('\\', '/'): value for key, value in declaration['source_sha256'].items()}
    assert declaration['source_files']['task'] == 'fixed_start_body_and_synergy'
    assert declaration['source_files']['declared'] == len(named) and declaration['source_files']['not_files'] == []
    assert named <= set(recorded) and len(recorded) == len(named)      # the six earlier source files are among them
    assert all(recorded[p] == hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in named)


def test_a_changed_world_is_refused_before_any_arm_of_the_body_runs(tmp_path, monkeypatch):
    """Right after a body's compiled world model is saved its content (the normalized content gain_block compares) is
    compared with the bound reference rows' of the same body: a different world refuses the round there, status.json reads
    refused with the reason, and no arm has run; an equal world (saved by another process, so another scene folder in its
    bytes) passes on to the anchored search start, where the run stops here on the anchor's shape, still before any arm.
    The model load and the body setup are stubs: no model or world is loaded."""
    from unittest import mock
    inputs = frozen_run_inputs(tmp_path)
    F = inputs.F
    arms = []

    class Model:
        synergy = object()
        GAZE_IDX = 25

        def _plan_chain(self, *args, **kwargs):
            arms.append('a planner arm')
            raise AssertionError('an arm ran')

    class Session:
        handle_names = ['handle_%d' % i for i in range(25)] + ['gaze']

        def close(self):
            pass

    def setup(frozen, episode, body, out=None):
        data = SimpleNamespace(qpos=np.array([0.1 * body, 0.2, 0.3]), qvel=np.array([0.01 * body, 0.]), time=24701.75)
        return SimpleNamespace(session=Session(), env=SimpleNamespace(data=data), raw=torch.zeros(1, 26, 9),
                               initial_joints=torch.zeros(1, 44), decision={}, current={}, snapshot=None,
                               base_command=np.zeros((1, 26, 9)), base_joints=np.zeros((1, 44)), gaze=np.zeros(2),
                               geometry=dict(hands=[]), initial_joint_angles=np.zeros(44),
                               initial_joint_velocity=np.zeros(44), fingers=[])

    def saver(raw):
        def save(env, out, saved):
            (Path(out) / 'world_models').mkdir(exist_ok=True)
            name = 'world_models/%s.mjb.gz' % hashlib.sha256(raw).hexdigest()[:12]
            (Path(out) / name).write_bytes(gzip.compress(raw))
            return dict(file=name, structure_fingerprint='f' * 64, exact_sha256=hashlib.sha256(raw).hexdigest(),
                        bytes=len(raw), gzip_sha256='g', content_sha256=F.model_content_sha256(raw))
        return save
    monkeypatch.setattr(F, 'guard', lambda need: None)
    monkeypatch.setattr(F, 'load_model_and_frozen_episode',
                        lambda checkpoint, frozen_set: (SimpleNamespace(model=Model(), info=dict(lineage_step=506025)), {}))
    monkeypatch.setattr(F, 'body_setup', setup)
    monkeypatch.setattr(F.P, 'checksum', lambda model: 'weights')
    same = b'model C:/x/gof_probes/_scenes/p999/pen_art/mat.png\x00 tail'     # the reference's world, saved by process 999
    try:
        monkeypatch.setattr(F, 'save_world_model', saver(same + b' changed'))
        with pytest.raises(RuntimeError, match="body 0: the compiled world model differs from the bound reference's"):
            F.frozen_run(inputs.args('same', 'changed'))
        status = json.loads((tmp_path / 'changed' / 'status.json').read_text(encoding='utf-8'))
        assert status['state'] == 'refused' and status['completed'] == 0 and 'differs' in status['reason']
        assert arms == [] and not (tmp_path / 'changed' / 'trials.jsonl').exists()
        monkeypatch.setattr(F, 'save_world_model', saver(same))
        with pytest.raises(RuntimeError, match='Anchor body 0 has shape'):
            F.frozen_run(inputs.args('same', 'equal'))
        assert arms == [] and json.loads((tmp_path / 'equal' / 'status.json').read_text())['state'] == 'running'
    finally:
        mock.patch.stopall()                     # the run's own timing patch, left started by the refusal


def test_the_code_disclosure_compares_every_recorded_file_and_names_the_ones_not_compared(tmp_path):
    """A code-trigger file of the task whose SHA-256 differs is named; one the reference did not record reads not compared,
    and the label then never reads as the same code."""
    F, run = fake_runs(tmp_path)
    manifest = json.loads((REPO / 'docs/gain_of_function_v3/ability_rechecks.json').read_text(encoding='utf-8'))
    triggers = {t['id']: t for t in manifest['tasks']}['fixed_start_body_and_synergy']['triggers']['code']
    differing, unrecorded = triggers[3], triggers[7]      # forward_model.py and action_decoder.py today

    def recording(files):
        return lambda rows, result: result.update(source_sha256=dict(result['source_sha256'], **files))
    reference = run('reference', 494025, tweak=recording({differing: 'b' * 64}))
    this = run('this', 506025, tweak=recording({differing: 'a' * 64, unrecorded: 'c' * 64}), pid=200)
    code = gain_of(F, this, reference)['changed_controls']['code']
    assert code['source_files_differing'] == [differing] and code['source_files_compared'] == 2
    assert code['not_compared'] == dict(by_the_reference=[unrecorded], by_this_run=[])
    assert differing in code['label'] and 'not compared' in code['label'] and unrecorded in code['label']
    only_unrecorded = run('only_unrecorded', 506025, tweak=recording({differing: 'b' * 64, unrecorded: 'c' * 64}), pid=300)
    code = gain_of(F, only_unrecorded, reference)['changed_controls']['code']
    assert code['source_files_differing'] == [] and code['not_compared']['by_the_reference'] == [unrecorded]
    assert 'the same code' not in code['label'] and 'not compared' in code['label']
    everything = run('everything', 506025, tweak=recording({differing: 'b' * 64}), pid=400)
    code = gain_of(F, everything, reference)['changed_controls']['code']
    assert code['label'].startswith('the same code as the reference') and 'not compared' not in code['label']


def test_the_label_carries_the_generalization_caveat(tmp_path):
    """The decision: "Repeated monitoring on this one fixed suite is not a confirmatory generalization claim", with the
    suite named, in the label of a valid and of an invalid comparison."""
    F, run = fake_runs(tmp_path)
    reference = run('reference', 494025)
    caveat = ('Repeated monitoring on this one fixed suite (the 8 frozen starts and 40 frozen goals of checkpoint 494,025) '
              'is not a confirmatory generalization claim.')
    assert gain_of(F, run('this', 506025, pid=200), reference)['label'].endswith(caveat)
    invalid = gain_of(F, run('missing', 506025, tweak=lambda rows, result: rows.pop(3), pid=300), reference)
    assert invalid['status'] == 'invalid' and invalid['label'].endswith(caveat)
