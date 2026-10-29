# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The development dashboard's choice of run (registry row dashboard_experiment_view_tests_20261013): production's run
while production trains; the live experiment run, read from tools/experiment_mirror.py's copies inside the project,
while production is paused or its run ended and the experiment's owner process is alive; production's paused run
otherwise. Stand-in folders in pytest's temporary folder only: a project root (two production segments, the
supervisor's records, an evidence registry) and an experiment run outside it."""
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import psutil
import pytest

from mimo_bridge.tools import development_dashboard as dashboard
from mimo_bridge.tools import experiment_mirror as em

ME = dict(pid=os.getpid(), created=psutil.Process().create_time())
DEAD = dict(pid=4_000_000, created=1.0)
NOW = time.time()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def clip(path, seconds_ago=1000):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b''.join(struct.pack('!I4s', 8, k) for k in (b'ftyp', b'mdat', b'moov')))
    os.utime(path, (time.time() - seconds_ago,) * 2)


def snapshot(folder):
    return {p.relative_to(folder).as_posix(): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def stage(tmp_path, production='paused', owner=ME, phase='training', mirror=ME):
    """production: 'training' (a fresh record), 'relaunched' (the supervisor reports training, the newest record is
    5,000 s old) or 'paused'. mirror: the process the mirror's service record names."""
    root = tmp_path / 'repo'
    saved = root / 'Saved/mimo_bridge'
    seg1, seg2 = saved / 'experiments/H_seg1', saved / 'experiments/H_seg2'
    write(seg1 / 'args.json', {})
    write(seg2 / 'args.json', dict(resume=str(seg1 / 'checkpoints/ckpt_step_100')))
    (seg2 / 'replay_training_stats.jsonl').write_text(
        json.dumps(dict(type='mimo', step=300, timestamp=NOW - (30 if production == 'training' else 5000))) + '\n',
        encoding='utf-8')
    if production in ('training', 'relaunched'):
        write(saved / 'training_continuation.json', dict(active_run=str(seg2), timestamp=NOW - 10, status='training'))
    else:
        write(saved / 'training_continuation.json', dict(active_run=str(seg2), timestamp=NOW - 10,
              status='paused for repair or explicit instruction', pause_reason='explicit supervisor pause'))
        write(saved / 'supervisor_control.json', dict(cmd='pause', reason='the production run ended at its declared step'))
    for step in (70, 90):
        clip(seg1 / f'videos/step_{step}_120000.mp4', 5000)
        clip(seg1 / f'videos/step_{step}_120000_eye.mp4', 5000)
    receipt = root / 'docs/result.json'
    write(receipt, {'gate': 'PASS'})
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    records = [dict(id=f'before_{s}', capability='predict', date='2026-10-01', kind='live_behavior', outcome='demonstrated',
                    subject=str(seg1 / f'checkpoints/ckpt_step_{s}'), title='Example', summary='Separate fit',
                    missing='Transfer', method='Frozen fit', receipts=[dict(path='docs/result.json', sha256=digest)],
                    checks=[dict(receipt=0, path='gate', equals='PASS')]) for s in (70, 95)]
    write(root / 'docs/development_dashboard/evidence.json', dict(schema=1, abilities=[dict(id='predict')], records=records))
    # the experiment: a run outside the project that resumed production segment 1 at step 80
    run = tmp_path / 'outside/run/X1_skip13_80'
    write(run / 'args.json', dict(resume=str(seg1 / 'checkpoints/ckpt_step_80'), report_every=100, live_view_every=1))
    (run / em.STATS).write_text(json.dumps(dict(type='launch', step=80, timestamp=NOW - 600, code_commit='abc1234'))
                                + '\n' + json.dumps(dict(type='mimo', step=120, timestamp=NOW - 30,
                                                         model_steps_per_s=0.45)) + '\n', encoding='utf-8')
    (run / 'live').mkdir(parents=True)
    (run / 'live/frame.jpg').write_bytes(b'\xff\xd8' + b'\x10' * 20 + b'\xff\xd9')
    clip(run / 'videos/step_110_150000.mp4')
    clip(run / 'videos/step_110_150000_eye.mp4')
    state = tmp_path / 'outside/receipts/owner_state.json'
    write(state, dict(owner_pid=owner['pid'], owner_created=owner['created'], phase=phase, run=str(run),
                      trainer=dict(pid=ME['pid'], created=ME['created'], run=str(run))))
    em.write_pointer(root, run, state, tmp_path / 'outside/launch.py', 'the test')
    pointer, st = em.read_pointer(root), {}
    em.main_pass(root, pointer, st, NOW - 2, settle_s=0)
    em.main_pass(root, pointer, st, NOW - 1, settle_s=0)
    em.copy_frame(run, em.mirror_dir(root, pointer), st['copied'], [])
    write(root / em.MIRRORS / em.SERVICE, dict(pid=mirror['pid'], pid_created=mirror['created']))
    return root, run, state


def test_paused_production_and_a_live_experiment_show_the_experiment_from_its_copies(tmp_path):
    root, run, _ = stage(tmp_path)
    before = snapshot(run.parent.parent)
    state = dashboard.build_state(root, now=NOW)
    t = state['training']
    assert t['view'] == 'experiment' and t['run_name'] == 'X1_skip13_80'
    assert t['run'] == 'Saved/mimo_bridge/experiment_mirror/X1_skip13_80'
    assert t['step'] == 120 and t['record_timestamp'] == NOW - 30 and t['supervisor_status'] == 'training'
    assert t['source'] == '/files/Saved/mimo_bridge/experiment_mirror/X1_skip13_80/mirror_status.json'
    assert t['experiment']['project_parent'] == dict(run='Saved/mimo_bridge/experiments/H_seg1', step=80)
    v = state['verdict']
    assert v['state'] == 'running'
    assert v['sentence'].startswith('The experiment run X1_skip13_80 is training outside the production supervisor: '
                                    'it is at model step 120; its newest record was written ')
    assert any(f.startswith('Production training is paused on purpose: the production run ended') for f in v['facts'])
    assert state['what_changed'].startswith('Experiment run X1_skip13_80 started')
    videos = state['videos']
    assert [(c['step'], c['experiment'], c['previous_run']) for c in videos] == [(110, True, False), (70, False, True)]
    assert videos[0]['outside'] == '/files/Saved/mimo_bridge/experiment_mirror/X1_skip13_80/videos/step_110_150000.mp4'
    lineage = {r['id']: r['lineage'] for r in state['records']}
    assert lineage['before_70'] == 'ancestor_checkpoint' and lineage['before_95'] == 'not_linked_to_current_model'
    live = state['live']
    assert live['view'] == 'experiment' and live['frame'] == '/files/Saved/mimo_bridge/experiment_mirror/X1_skip13_80/live/frame.jpg'
    # three live-view periods at 0.45 model steps per second, plus the mirror's copy period
    assert live['frame_fresh_seconds'] == pytest.approx(3 / 0.45 + em.TICK_S)
    assert live['frame_publishing'] is True and 'by the experiment mirror' in live['texts']['pub']
    assert snapshot(run.parent.parent) == before


def test_running_production_keeps_the_production_view_unchanged(tmp_path):
    root, _, _ = stage(tmp_path, production='training')
    with_pointer = dashboard.build_state(root, now=NOW)
    em.pointer_path(root).unlink()
    without = dashboard.build_state(root, now=NOW)
    assert 'view' not in with_pointer['training'] and with_pointer['training']['run_name'] == 'H_seg2'
    assert with_pointer['training'] == without['training']
    assert with_pointer['verdict'] == without['verdict']
    # production's clips are the same; the experiment's clips are listed either way, only their state differs
    assert [v for v in with_pointer['videos'] if not v['experiment']] == [v for v in without['videos'] if not v['experiment']]
    assert [v['id'] for v in with_pointer['videos']] == [v['id'] for v in without['videos']]
    assert with_pointer['verdict']['sentence'].startswith('Training is running: H_seg2 is at lineage step 300')


@pytest.mark.parametrize('owner,phase,mirror', [(DEAD, 'training', ME), (ME, 'done', ME), (ME, 'arm_end', ME),
                                                (ME, 'training', DEAD)])
def test_a_dead_or_finished_owner_or_a_dead_mirror_returns_to_the_production_view(tmp_path, owner, phase, mirror):
    root, _, _ = stage(tmp_path, owner=owner, phase=phase, mirror=mirror)
    state = dashboard.build_state(root, now=NOW)
    assert 'view' not in state['training'] and state['training']['run_name'] == 'H_seg2'
    assert state['verdict']['sentence'] == ('Training is paused on purpose: the production run ended at its declared '
                                            'step.')
    # the recordings list does not depend on the owner: the experiment's clip stays listed, newest first
    assert [(c['step'], c['experiment'], c['run_name']) for c in state['videos']][0] == (110, True, 'X1_skip13_80')
    assert state['videos'][0]['experiment_state'] in ('ended', 'not confirmed')


def test_a_production_relaunch_before_its_first_record_keeps_the_production_view(tmp_path):
    root, _, _ = stage(tmp_path, production='relaunched')
    state = dashboard.build_state(root, now=NOW)
    assert 'view' not in state['training'] and state['training']['run_name'] == 'H_seg2'
    assert state['verdict']['state'] == 'unconfirmed'


def test_a_production_stop_rule_stays_on_the_page_in_the_experiment_view(tmp_path):
    root, _, _ = stage(tmp_path)
    write(root / 'Saved/mimo_bridge/experiments/H_seg2/stop_reason.json',
          dict(reason='motor fixes kill line crossed', resume_policy='repair_required', timestamp=NOW - 4000,
               writer='docs/gain_of_function_v4/motor_fixes_watch.py'))
    v = dashboard.build_state(root, now=NOW)['verdict']
    assert v['state'] == 'running'
    assert any("stopped production's H_seg2" in f for f in v['facts']), v['facts']


def test_the_experiment_view_refuses_live_commands_and_serves_only_project_paths(tmp_path):
    root, run, _ = stage(tmp_path)
    before, mirror = snapshot(run.parent.parent), root / 'Saved/mimo_bridge/experiment_mirror/X1_skip13_80'
    training = dashboard.training_state(root, NOW)
    for action in ('record', 'stop', 'save', 'sensors', 'open_viewer'):
        with pytest.raises(ValueError, match='through copies'):
            dashboard.live_command(root, training, dict(action=action, name='x', file='y.mp4', show=True))
    assert not (mirror / 'live/record.json').exists() and not (mirror / 'live/viewer_flags.json').exists()
    assert snapshot(run.parent.parent) == before
    with pytest.raises(ValueError):
        dashboard.confined(root, str(run / 'videos/step_110_150000.mp4'))
    text = json.dumps(dashboard.build_state(root, now=NOW))
    served = {part.split('"')[0] for part in text.split('"/files/')[1:]}
    assert served and all(dashboard.confined(root, s) for s in served)


def test_a_stale_mirror_reads_unconfirmed_not_running(tmp_path):
    root, _, _ = stage(tmp_path)
    v = dashboard.build_state(root, now=NOW + 400)['verdict']
    assert v['state'] == 'unconfirmed'
    assert v['sentence'].startswith('The experiment run X1_skip13_80 is not confirmed as training')
    assert "the mirror's last pass is 6 minutes old" in v['sentence']        # its last pass was at NOW - 1


def test_the_recordings_list_merges_every_mirrored_run_newest_first_after_the_end(tmp_path):
    """Operator, Oct 13, 2026: "why is the dashboard showing old video evidence and not the most recent!" The list
    keeps an ended experiment's clips, adds a run copied once after its end, includes an experiment's event clips and
    orders everything by recording time; production's own event clips stay out, as before."""
    root, run, state = stage(tmp_path, phase='done')
    pointer = em.read_pointer(root)
    em.main_pass(root, pointer, {}, NOW, settle_s=0)              # the mirror's pass that reads the end
    clip(run / 'videos/moment_headlook_step_115.mp4', 500)
    clip(run / 'videos/moment_headlook_step_115_eye.mp4', 500)
    em.main_pass(root, pointer, {}, NOW + 1, settle_s=0)
    st = {}
    em.main_pass(root, pointer, st, NOW + 2, settle_s=0)
    em.main_pass(root, pointer, st, NOW + 3, settle_s=0)
    older = tmp_path / 'outside/run/X0_skip13_60'
    write(older / 'args.json', {})
    clip(older / 'videos/step_70_010000.mp4', 3000)
    clip(older / 'videos/step_70_010000_eye.mp4', 3000)
    rec = em.copy_once(root, older, 'the test', now=NOW)
    assert rec['copied_once'] and sorted(rec['copied_this_call']) == ['step_70_010000.mp4', 'step_70_010000_eye.mp4']
    clip(root / 'Saved/mimo_bridge/experiments/H_seg1/videos/moment_headlook_step_75.mp4', 100)
    clip(root / 'Saved/mimo_bridge/experiments/H_seg1/videos/moment_headlook_step_75_eye.mp4', 100)
    state_ = dashboard.build_state(root, now=NOW)
    listed = [(v['step'], v['run_name'], v['kind'], v['experiment_state']) for v in state_['videos']]
    assert listed == [(115, 'X1_skip13_80', 'event', 'ended'), (110, 'X1_skip13_80', 'observation', 'ended'),
                      (70, 'X0_skip13_60', 'observation', 'ended'), (90, 'H_seg1', 'observation', None),
                      (70, 'H_seg1', 'observation', None)], listed
    assert state_['videos'][0]['events'] == ['headlook']
    assert all(v['outside'].startswith('/files/Saved/mimo_bridge/') for v in state_['videos'])
    assert [r for r in state_['videos'] if r['run_name'] == 'H_seg1'][0]['previous_run'] is True
    # the training card names the ended experiment in production's view
    assert any(f.startswith('The experiment run X1_skip13_80 has ended: the owner state reads phase "done"')
               for f in state_['verdict']['facts']), state_['verdict']['facts']


def test_copy_once_refuses_the_run_the_pointer_names(tmp_path):
    root, run, _ = stage(tmp_path)
    with pytest.raises(SystemExit):
        em.copy_once(root, run, 'the test', now=NOW)
