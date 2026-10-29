# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import json
from pathlib import Path
import time

import pytest
from mimo_bridge.tools import ability_comparison_window as window


@pytest.fixture
def saved_window(tmp_path,monkeypatch):
    saved=tmp_path/'Saved/mimo_bridge';run=saved/'experiments/test';run.mkdir(parents=True)
    folder=saved/'ability_rechecks';folder.mkdir()
    supervisor=dict(active_run=str(run),timestamp=time.time())
    window.atomic_json(saved/'training_continuation.json',supervisor)
    window.atomic_json(saved/'supervisor_control.json',dict(cmd='run',archiver='off'))
    class Trainer:
        pid=71
        def create_time(self):return 123.
    running=[Trainer()]
    monkeypatch.setattr(window,'processes',lambda *args:list(running))
    state=dict(tasks={'touch':dict(title='Touch',due_step=105)})
    checkpoint=run/'checkpoints/ckpt_step_108';checkpoint.mkdir(parents=True)
    physical=checkpoint/'episode.pt';physical.write_bytes(b'preserved eight worlds')
    monkeypatch.setattr(window,'complete_candidates',lambda _: [checkpoint])
    def finish_save():
        running.clear()
        control=window.read(run/'control.json')
        window.atomic_json(run/'control_done_window.json',control)
        (run/'control.json').unlink()
        window.atomic_json(run/'stage_done.json',dict(ok=True,steps=108,finished_at=time.time(),final_checkpoint=str(checkpoint)))
        window.atomic_json(checkpoint/'episode_state.json',dict(step=108,times=[1.]*8,
                           storage_path=str(physical),bytes=physical.stat().st_size,sha256=window.sha256_file(physical)))
    return dict(root=tmp_path,saved=saved,run=run,folder=folder,state=state,
                supervisor=supervisor,running=running,finish_save=finish_save,checkpoint=checkpoint)


def start(x):
    return window.request(x['root'],x['folder'],x['state'],x['run'],x['supervisor'],105)


def advance(x):return window.advance(x['root'],x['folder'],x['state'],x['run'])


def test_window_saves_before_testing_and_releases_only_to_existing_supervisor(saved_window):
    x=saved_window;assert start(x)
    owned=x['state']['comparison_window'];token=owned['token']
    assert owned['state']=='saving'
    assert window.read(x['run']/'control.json')['auto_resume'] is True
    assert window.read(x['saved']/'supervisor_control.json')['ability_window_token']==token
    assert advance(x) is False
    x['finish_save']()
    # A reloaded state, not a parent-owned subprocess object, resumes the workflow.
    x['state']=window.read(x['folder']/'status.json')
    assert advance(x) is True
    x['state']['tasks']['touch'].update(completed_step=108,status='complete')
    assert advance(x) is False
    assert x['state']['comparison_window']['state']=='released'
    assert window.read(x['saved']/'supervisor_control.json')==dict(cmd='run',archiver='off')
    assert (x['folder']/('window_'+token+'.json')).is_file()


@pytest.mark.parametrize('interruption',['operator','repair','manual_stop','different_run'])
def test_window_never_overrides_new_stop_or_owner(saved_window,interruption):
    x=saved_window;assert start(x);x['finish_save']();assert advance(x)
    if interruption=='operator':window.atomic_json(x['saved']/'supervisor_control.json',dict(cmd='pause',reason='operator'))
    elif interruption=='repair':window.atomic_json(x['run']/'stop_reason.json',dict(resume_policy='repair_required',reason='bad control'))
    elif interruption=='manual_stop':window.atomic_json(x['run']/'control_done_manual.json',dict(cmd='stop'))
    else:window.atomic_json(x['saved']/'training_continuation.json',dict(active_run=str(x['run'].with_name('other'))))
    x['state']['tasks']['touch'].update(completed_step=108,status='complete')
    window.release(x['root'],x['folder'],x['state'],'done')
    assert window.read(x['saved']/'supervisor_control.json')['cmd']=='pause'
    assert x['state']['comparison_window']['state'] in ('ownership changed','repair required')


@pytest.mark.parametrize('fault',['missing_episode','missing_body','old_save','failed_save','wrong_path','wrong_hash','wrong_token'])
def test_bad_final_save_never_starts_comparison_or_resumes(saved_window,fault):
    x=saved_window;assert start(x);x['finish_save']()
    if fault=='missing_episode':(x['checkpoint']/'episode_state.json').unlink()
    elif fault=='missing_body':
        path=x['checkpoint']/'episode_state.json';d=window.read(path);d['times']=d['times'][:7];window.atomic_json(path,d)
    elif fault=='wrong_hash':
        path=x['checkpoint']/'episode_state.json';d=window.read(path);d['sha256']='0'*64;window.atomic_json(path,d)
    elif fault=='wrong_token':
        path=x['run']/'control_done_window.json';d=window.read(path);d['ability_window_token']='another';window.atomic_json(path,d)
    else:
        path=x['run']/'stage_done.json';d=window.read(path)
        if fault=='old_save':d['finished_at']=0
        elif fault=='wrong_path':d['final_checkpoint']=str(x['checkpoint'].with_name('other'))
        else:d['ok']=False
        window.atomic_json(path,d)
    assert not advance(x)
    assert x['state']['comparison_window']['state']=='repair required'
    assert window.read(x['saved']/'supervisor_control.json')['cmd']=='pause'


def test_a_companion_past_260_characters_is_verified_not_called_missing(saved_window):
    """2026-09-24: a resume folder's eight-world save sat at a 262-character path; the window
    called it missing and stopped training for repair."""
    import os
    from mimo_bridge.storage_locations import long_path
    x=saved_window;assert start(x);x['finish_save']()
    deep=x['saved']
    while len(str(deep))<262:
        deep=deep/('seg38_observer_resume_'+'z'*30)
    os.makedirs(long_path(deep),exist_ok=True)
    far=deep/'ckpt_step_108_companion.pt.gz'
    with open(long_path(far),'wb') as f:f.write(b'preserved eight worlds, far away')
    path=x['checkpoint']/'episode_state.json';d=window.read(path)
    d.update(storage_path=str(far),bytes=os.stat(long_path(far)).st_size,sha256=window.sha256_file(far))
    window.atomic_json(path,d)
    assert len(str(far))>260
    assert advance(x) is True
    assert x['state']['comparison_window']['state']=='ready'


def test_existing_trainer_control_prevents_claiming_window(saved_window):
    x=saved_window;window.atomic_json(x['run']/'control.json',dict(cmd='snapshot'))
    assert start(x) is False
    assert 'comparison_window' not in x['state']
    assert window.read(x['saved']/'supervisor_control.json')['cmd']=='run'


def test_recovery_of_prepared_window_does_not_duplicate_consumed_save(saved_window):
    x=saved_window;assert start(x);x['finish_save']()
    x['state']['comparison_window']['state']='preparing'
    assert advance(x)
    assert not (x['run']/'control.json').exists()


def test_exhausted_memory_window_releases_training_with_visible_failed_task(saved_window):
    x=saved_window;assert start(x);x['finish_save']();assert advance(x)
    x['state']['tasks']['touch'].update(status='memory capacity needs repair')
    assert advance(x) is False
    assert x['state']['comparison_window']['state']=='released'
    assert x['state']['tasks']['touch']['status']=='memory capacity needs repair'
    assert window.read(x['saved']/'supervisor_control.json')['cmd']=='run'
