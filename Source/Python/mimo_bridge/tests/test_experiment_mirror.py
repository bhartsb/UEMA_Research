# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""tools/experiment_mirror.py (registry row experiment_mirror_tests_20261013): the copies of a live experiment run kept
inside the project for the development dashboard. Every case runs on stand-in folders in pytest's temporary folder: a
"project" root and a run folder outside it; no production file is read or written."""
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import psutil
import pytest

from mimo_bridge.tools import experiment_mirror as em

ME = dict(pid=os.getpid(), created=psutil.Process().create_time())
DEAD = dict(pid=4_000_000, created=1.0)          # no process has this id (Windows ids stay far below)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def mp4_bytes(complete=True):
    kinds = [b'ftyp', b'mdat', b'moov'] if complete else [b'ftyp', b'mdat']
    return b''.join(struct.pack('!I4s', 16, k) + b'\0' * 8 for k in kinds)


def wav_bytes(n=40):
    body = b'WAVE' + b'\0' * n
    return b'RIFF' + struct.pack('<I', len(body)) + body


JPEG = b'\xff\xd8' + b'\x10' * 30 + b'\xff\xd9'


def age(path, seconds_ago):
    t = time.time() - seconds_ago
    os.utime(path, (t, t))


def snapshot(folder):
    """{relative path: (size, mtime_ns, sha256)} of every file under folder."""
    out = {}
    for p in sorted(Path(folder).rglob('*')):
        if p.is_file():
            out[p.relative_to(folder).as_posix()] = (p.stat().st_size, p.stat().st_mtime_ns,
                                                     hashlib.sha256(p.read_bytes()).hexdigest())
    return out


def stage(tmp_path, phase='training', owner=ME, trainer=ME, clip_age=1000):
    """A project root, a run folder outside it with a full set of display files, its owner state and the pointer."""
    root = tmp_path / 'repo'
    (root / 'Saved/mimo_bridge').mkdir(parents=True)
    run = tmp_path / 'outside/run/R1_skip13_100'
    write(run / 'args.json', dict(resume=str(tmp_path / 'outside/prev/checkpoints/ckpt_step_100'), max_steps=500))
    write(run / 'checkpoints.json', [dict(step=150, reason='hourly', time='2026-10-13 15:12:24', elapsed_min=60.0)])
    lines = [dict(type='launch', step=100, timestamp=1000.0), dict(type='mimo', step=150, timestamp=1010.0)]
    (run / em.STATS).write_text(''.join(json.dumps(r) + '\n' for r in lines) + '{"type": "mi', encoding='utf-8')
    (run / 'live').mkdir(parents=True)
    (run / 'live/frame.jpg').write_bytes(JPEG)
    v = run / 'videos'
    v.mkdir()
    for name, data in (('step_140_120000.mp4', mp4_bytes()), ('step_140_120000_eye.mp4', mp4_bytes()),
                       ('step_140_120000_control.json', json.dumps(dict(version='body_control_video_v1')).encode()),
                       ('step_140_120000_heard.wav', wav_bytes()),
                       ('moment_headlook_step_145.mp4', mp4_bytes()), ('moment_headlook_step_145_eye.mp4', mp4_bytes())):
        (v / name).write_bytes(data)
        age(v / name, clip_age)
    state = tmp_path / 'outside/receipts/owner_state.json'
    write(state, dict(owner_pid=owner['pid'], owner_created=owner['created'], phase=phase, run=str(run),
                      trainer=dict(pid=trainer['pid'], created=trainer['created'], run=str(run)),
                      started_at='Oct 13, 2026, 02:08:48 PM PDT'))
    em.write_pointer(root, run, state, tmp_path / 'outside/launch.py', 'the test')
    return root, run, state


def test_a_pass_copies_the_display_files_atomically_and_never_writes_into_the_run(tmp_path):
    root, run, _ = stage(tmp_path)
    before = snapshot(run.parent.parent)
    pointer = em.read_pointer(root)
    st, t = {}, time.time()
    em.main_pass(root, pointer, st, t, settle_s=90)
    rec = em.main_pass(root, pointer, st, t + 100, settle_s=90)
    em.copy_frame(run, em.mirror_dir(root, pointer), st['copied'], [])
    mirror = root / 'Saved/mimo_bridge/experiment_mirror/R1_skip13_100'
    assert sorted(rec['videos_copied_this_pass']) == sorted(os.listdir(run / 'videos'))
    for rel in ('args.json', 'checkpoints.json', 'live/frame.jpg', 'videos/step_140_120000.mp4',
                'videos/step_140_120000_eye.mp4', 'videos/step_140_120000_control.json',
                'videos/step_140_120000_heard.wav', 'videos/moment_headlook_step_145.mp4'):
        assert (mirror / rel).read_bytes() == (run / rel).read_bytes(), rel
        assert (mirror / rel).stat().st_mtime == pytest.approx((run / rel).stat().st_mtime, abs=1e-3), rel
    text = (run / em.STATS).read_bytes()
    assert (mirror / em.STATS).read_bytes() == text[:text.rfind(b'\n') + 1]
    assert json.loads((mirror / 'owner_state.json').read_text())['phase'] == 'training'
    assert not list(mirror.rglob('*.tmp'))
    assert snapshot(run.parent.parent) == before            # the run folder and its owner's files are untouched
    assert rec['ended'] is False and rec['owner_alive'] is True and rec['trainer_alive'] is True
    status = json.loads((mirror / em.STATUS).read_text())
    assert status['pass_at'] == t + 100 and status['mirror'] == 'Saved/mimo_bridge/experiment_mirror/R1_skip13_100'


def test_a_clip_is_read_only_after_its_files_settle_and_only_when_complete(tmp_path):
    root, run, _ = stage(tmp_path, clip_age=0)
    pointer, st, t = em.read_pointer(root), {}, time.time()
    mirror = em.mirror_dir(root, pointer)
    assert em.main_pass(root, pointer, st, t, settle_s=90)['videos_copied_this_pass'] == []
    # the trainer replaces the clip (its sound mux) 30 s later: the group's wait starts again
    (run / 'videos/step_140_120000.mp4').write_bytes(mp4_bytes() + struct.pack('!I4s', 8, b'free'))
    rec = em.main_pass(root, pointer, st, t + 30, settle_s=90)
    assert rec['videos_copied_this_pass'] == [] and 'step_140_120000' in rec['videos_waiting_s']
    rec = em.main_pass(root, pointer, st, t + 100, settle_s=90)
    assert not any(n.startswith('step_140') for n in rec['videos_copied_this_pass'])   # 70 s since the change: waiting
    assert 'moment_headlook_step_145.mp4' in rec['videos_copied_this_pass']           # 100 s unchanged: copied
    rec = em.main_pass(root, pointer, st, t + 125, settle_s=90)
    assert 'step_140_120000.mp4' in rec['videos_copied_this_pass']
    assert (mirror / 'videos/step_140_120000.mp4').read_bytes() == (run / 'videos/step_140_120000.mp4').read_bytes()
    # an unfinished container (no moov box) that has settled is never copied
    (run / 'videos/step_200_130000.mp4').write_bytes(mp4_bytes(complete=False))
    age(run / 'videos/step_200_130000.mp4', 1000)
    em.main_pass(root, pointer, st, t + 200, settle_s=90)
    rec = em.main_pass(root, pointer, st, t + 300, settle_s=90)
    assert not (mirror / 'videos/step_200_130000.mp4').exists()
    assert any('step_200_130000.mp4 is not complete' in n for n in rec['notes'])


def test_the_stats_copy_is_a_growing_prefix_ending_at_a_complete_line(tmp_path):
    root, run, _ = stage(tmp_path)
    src, dst = run / em.STATS, tmp_path / 'mirror_stats.jsonl'
    st = {}
    added = em.copy_stats(src, dst, st)
    first = src.read_bytes()
    assert dst.read_bytes() == first[:first.rfind(b'\n') + 1] and added == len(dst.read_bytes())
    assert em.copy_stats(src, dst, st) == 0                    # nothing new: nothing written
    with open(src, 'ab') as f:
        f.write(b'mo", "step": 160, "timestamp": 1020.0}\n{"type": "mi')
    em.copy_stats(src, dst, st)
    rows = [json.loads(x) for x in dst.read_text().splitlines()]
    assert [r['step'] for r in rows] == [100, 150, 160]
    # a run file that no longer starts with the copied bytes is copied again from its start
    src.write_text(json.dumps(dict(type='launch', step=900, timestamp=1.0)) + '\n', encoding='utf-8')
    em.copy_stats(src, dst, {})
    assert json.loads(dst.read_text().splitlines()[0])['step'] == 900 and len(dst.read_text().splitlines()) == 1


def test_the_live_picture_copy_skips_an_unfinished_picture_and_keeps_the_picture_time(tmp_path):
    root, run, _ = stage(tmp_path)
    mirror, copied, notes = em.mirror_dir(root, em.read_pointer(root)), {}, []
    (run / 'live/frame.jpg').write_bytes(JPEG[:-2])
    assert em.copy_frame(run, mirror, copied, notes) is False and not (mirror / 'live/frame.jpg').exists()
    (run / 'live/frame.jpg').write_bytes(JPEG)
    age(run / 'live/frame.jpg', 3)
    assert em.copy_frame(run, mirror, copied, notes) is True
    assert (mirror / 'live/frame.jpg').stat().st_mtime == pytest.approx((run / 'live/frame.jpg').stat().st_mtime, abs=1e-3)
    assert em.copy_frame(run, mirror, copied, notes) is False   # unchanged: not read again


END_PHASES_CASES = ('done', 'arm_end', 'completion')


@pytest.mark.parametrize('case,ended', [('training', False), ('done', True), ('arm_end', True), ('completion', True),
                                        ('other_run', True), ('all_dead', True), ('trainer_dead_owner_alive', False),
                                        ('gone', True), ('unreadable', False)])
def test_the_end_of_the_run_is_read_from_the_owner_state(tmp_path, case, ended):
    run = tmp_path / 'run'
    state = dict(owner_pid=ME['pid'], owner_created=ME['created'], phase='training', run=str(run),
                 trainer=dict(pid=ME['pid'], created=ME['created']))
    if case in END_PHASES_CASES:
        state['phase'] = case
    if case == 'other_run':
        state['run'] = str(tmp_path / 'another')
    if case == 'all_dead':
        state.update(owner_pid=DEAD['pid'], owner_created=DEAD['created'], trainer=dict(DEAD))
    if case == 'trainer_dead_owner_alive':
        state['trainer'] = dict(DEAD)
    got, why = em.run_ended(None if case == 'unreadable' else state, run, exists=case != 'gone')
    assert got is ended and (why is not None) is ended


class Clock:
    """A clock for the loop: each call moves it on by `step` seconds."""
    def __init__(self, start, step):
        self.t, self.step = start, step

    def __call__(self):
        self.t += self.step
        return self.t


def service_of(root):
    return json.loads((root / em.MIRRORS / em.SERVICE).read_text())


def test_the_loop_stops_by_itself_only_after_the_end_holds_for_the_confirmation_period(tmp_path, monkeypatch):
    monkeypatch.setattr(em, 'opt_out_of_power_throttling', lambda: 'not changed in a test')
    root, run, state = stage(tmp_path)
    loop = dict(tick_s=0, main_every_s=0, settle_s=0, sleep=lambda s: None, end_confirm_s=60)
    assert em.serve(root, now=Clock(float(int(time.time())), 10), max_ticks=3, **loop) == 0
    assert service_of(root)['state'] == 'stopped'            # still training: only this call's tick limit ended it
    write(state, dict(json.loads(state.read_text()), phase='done', trainer=None))
    assert em.serve(root, now=Clock(float(int(time.time())), 10), max_ticks=5, **loop) == 0
    assert service_of(root)['state'] == 'stopped'            # 40 s of "done" is under the 60 s confirmation
    assert em.serve(root, now=Clock(float(int(time.time())), 10), max_ticks=50, **loop) == 0
    service = service_of(root)
    assert service['state'] == 'ended' and 'phase "done"' in service['why'] and 'for 60 s' in service['why']
    assert (em.mirror_dir(root, em.read_pointer(root)) / 'videos/step_140_120000.mp4').is_file()
    # without a pointer the loop ends after the same confirmation
    em.pointer_path(root).unlink()
    assert em.serve(root, now=Clock(float(int(time.time())), 10), max_ticks=50, **loop) == 0
    assert 'no readable pointer' in service_of(root)['why'] and 'for 60 s' in service_of(root)['why']


def test_one_failed_look_at_the_owner_state_does_not_end_the_loop(tmp_path):
    root, run, state = stage(tmp_path)
    states, memo, t = {}, {}, time.time()
    text = state.read_text()
    state.unlink()                                           # the disk did not answer for one pass
    assert em.tick(root, states, memo, t, 0, 0, 60) is None and memo['ended_at'] == t
    state.write_text(text, encoding='utf-8')
    assert em.tick(root, states, memo, t + 15, 0, 0, 60) is None and memo['ended_at'] is None
    state.unlink()
    for k in range(4):
        assert em.tick(root, states, memo, t + 30 + 15 * k, 0, 0, 60) is None
    assert em.tick(root, states, memo, t + 90, 0, 0, 60)['state'] == 'ended'


def test_an_error_inside_a_tick_is_recorded_and_the_loop_goes_on(tmp_path, monkeypatch):
    monkeypatch.setattr(em, 'opt_out_of_power_throttling', lambda: 'not changed in a test')
    root, _, _ = stage(tmp_path)

    def full(*a, **k):
        raise OSError(28, 'No space left on device')
    monkeypatch.setattr(em, 'copy_stats', full)
    assert em.serve(root, tick_s=0, main_every_s=0, settle_s=0, sleep=lambda s: None, max_ticks=4,
                    now=Clock(time.time(), 1)) == 0
    service = service_of(root)
    assert service['state'] == 'stopped' and service['errors'] >= 2 and 'No space left' in service['last_error']['error']


def test_a_folder_of_another_run_with_the_same_name_is_moved_aside(tmp_path):
    root, run, _ = stage(tmp_path)
    mirror = em.mirror_dir(root, em.read_pointer(root))
    write(mirror / em.STATUS, dict(run=str(tmp_path / 'elsewhere/R1_skip13_100')))
    (mirror / 'videos').mkdir(parents=True)
    (mirror / 'videos/step_9_000000.mp4').write_bytes(mp4_bytes())
    rec = em.main_pass(root, em.read_pointer(root), {}, time.time(), settle_s=0)
    assert any('moved to' in n for n in rec['notes'])
    assert not (mirror / 'videos/step_9_000000.mp4').exists()
    assert list((root / em.MIRRORS / '_replaced').glob('R1_skip13_100_*/videos/step_9_000000.mp4'))
    assert json.loads((mirror / em.STATUS).read_text())['run'] == str(run)


def test_a_clip_whose_disk_drops_its_sub_second_time_is_not_copied_again(tmp_path):
    root, run, _ = stage(tmp_path)
    pointer, st, t = em.read_pointer(root), {}, time.time()
    em.main_pass(root, pointer, st, t, settle_s=90)
    assert 'step_140_120000.mp4' in em.main_pass(root, pointer, st, t + 100, settle_s=90)['videos_copied_this_pass']
    clip = run / 'videos/step_140_120000.mp4'
    whole = int(clip.stat().st_mtime)
    os.utime(clip, (whole, whole))                           # measured on D: (module text): .69 s later read .00
    for k in range(1, 4):
        assert em.main_pass(root, pointer, st, t + 100 + 100 * k, settle_s=90)['videos_copied_this_pass'] == []


def test_one_loop_at_a_time(tmp_path, monkeypatch):
    monkeypatch.setattr(em, 'opt_out_of_power_throttling', lambda: 'not changed in a test')
    root, _, _ = stage(tmp_path)
    from mimo_bridge.tools.training_continuation import OwnerLock
    (root / em.MIRRORS).mkdir(parents=True, exist_ok=True)
    held = OwnerLock(root / em.MIRRORS / 'mirror.lock')
    try:
        assert em.serve(root, tick_s=0, main_every_s=0, sleep=lambda s: None, max_ticks=2) == 3
    finally:
        held.close()


def test_the_nearest_ancestor_inside_the_project_is_found_through_runs_outside_it(tmp_path):
    root = tmp_path / 'repo'
    parent = root / 'Saved/mimo_bridge/experiments/H_seg1'
    write(parent / 'args.json', dict(out=str(parent), checkpoint_storage=str(parent / 'checkpoints')))
    middle = tmp_path / 'outside/middle'
    write(middle / 'args.json', dict(resume=str(parent / 'checkpoints/ckpt_step_100')))
    run = tmp_path / 'outside/run'
    write(run / 'args.json', dict(resume=str(middle / 'checkpoints/ckpt_step_200')))
    found, chain = em.project_parent(root, run)
    assert found == dict(run='Saved/mimo_bridge/experiments/H_seg1', step=100)
    assert [c['step'] for c in chain] == [200, 100]
    write(middle / 'args.json', {})
    assert em.project_parent(root, run)[0] is None


def test_the_pointer_refuses_an_unplain_name_and_an_owner_state_of_another_run(tmp_path):
    root, run, state = stage(tmp_path)
    write(em.pointer_path(root), dict(run=str(tmp_path / '..'), owner_state=str(state)))
    assert em.read_pointer(root) is None
    bad = tmp_path / 'outside/run/bad name'
    bad.mkdir(parents=True)
    with pytest.raises(SystemExit):
        em.write_pointer(root, bad, state, 'x', 'the test')
    other = tmp_path / 'outside/run/R2'
    other.mkdir(parents=True)
    with pytest.raises(SystemExit):
        em.write_pointer(root, other, state, 'x', 'the test')


def test_the_dashboard_view_needs_the_mirror_and_the_owner_alive_and_the_run_unfinished(tmp_path):
    root, run, state = stage(tmp_path)
    pointer, st, t = em.read_pointer(root), {}, time.time()
    assert em.live_view(root) is None                         # nothing copied yet
    em.main_pass(root, pointer, st, t, settle_s=0)
    write(root / em.MIRRORS / em.SERVICE, dict(pid=DEAD['pid'], pid_created=DEAD['created']))
    assert em.live_view(root) is None                         # the mirror process is gone: its copies stop following
    write(root / em.MIRRORS / em.SERVICE, dict(pid=ME['pid'], pid_created=ME['created']))
    view = em.live_view(root)
    assert view and view['run_name'] == 'R1_skip13_100' and view['owner_alive'] and view['trainer_alive']
    assert view['mirror'] == root / 'Saved/mimo_bridge/experiment_mirror/R1_skip13_100' and view['mirror_pass_at'] == t
    write(state, dict(json.loads(state.read_text()), owner_pid=DEAD['pid'], owner_created=DEAD['created']))
    em.main_pass(root, pointer, st, t + 1, settle_s=0)
    assert em.live_view(root) is None
    write(state, dict(json.loads(state.read_text()), owner_pid=ME['pid'], owner_created=ME['created'], phase='done'))
    em.main_pass(root, pointer, st, t + 2, settle_s=0)
    assert em.live_view(root) is None


def test_copy_once_copies_an_ended_runs_settled_clips_and_nothing_twice(tmp_path):
    root, run, _ = stage(tmp_path)
    ended = tmp_path / 'outside/run/R0_skip13_50'
    write(ended / 'args.json', dict(resume='x'))
    (ended / 'videos').mkdir(parents=True)
    for name in ('step_60_010000.mp4', 'step_60_010000_eye.mp4', 'step_70_020000.mp4', 'step_70_020000_eye.mp4'):
        (ended / 'videos' / name).write_bytes(mp4_bytes())
    for name in ('step_60_010000.mp4', 'step_60_010000_eye.mp4'):
        age(ended / 'videos' / name, 1000)
    before = snapshot(ended)
    rec = em.copy_once(root, ended, 'the test')
    mirror = root / em.MIRRORS / 'R0_skip13_50'
    assert sorted(rec['copied_this_call']) == ['step_60_010000.mp4', 'step_60_010000_eye.mp4']
    assert rec['waiting'] == ['step_70_020000'] and rec['copied_once'] is True and rec['ended'] is True
    assert (mirror / 'args.json').is_file() and not (mirror / 'videos/step_70_020000.mp4').exists()
    assert em.copy_once(root, ended, 'the test')['copied_this_call'] == []
    assert snapshot(ended) == before
    with pytest.raises(SystemExit):
        em.copy_once(root, run, 'the test')                  # the pointer's run belongs to the loop
    with pytest.raises(SystemExit):
        em.copy_once(root, tmp_path / 'outside/run/missing', 'the test')


def test_copy_once_refuses_a_run_that_may_still_train_or_shares_the_live_runs_folder_name(tmp_path):
    root, run, _ = stage(tmp_path)
    em.main_pass(root, em.read_pointer(root), {}, time.time(), settle_s=0)     # the live run's mirror folder exists
    twin = tmp_path / 'elsewhere/R1_skip13_100'                # the pointer's run's folder name, another place
    (twin / 'videos').mkdir(parents=True)
    with pytest.raises(SystemExit):
        em.copy_once(root, twin, 'the test')
    fresh = tmp_path / 'outside/run/R9_skip13_10'
    (fresh / 'videos').mkdir(parents=True)
    (fresh / em.STATS).write_text(json.dumps(dict(type='mimo', step=20, timestamp=time.time() - 60)) + '\n',
                                  encoding='utf-8')
    with pytest.raises(SystemExit):
        em.copy_once(root, fresh, 'the test')                  # a record 60 s old: it may still train
    (fresh / em.STATS).write_text(json.dumps(dict(type='mimo', step=20, timestamp=time.time() - 4000)) + '\n',
                                  encoding='utf-8')
    assert em.copy_once(root, fresh, 'the test')['copied_once'] is True
    assert (root / em.MIRRORS / 'R1_skip13_100').is_dir() and not list((root / em.MIRRORS).glob('_replaced/*'))
