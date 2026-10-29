# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mimo_bridge.video_control import annotate, body_control, VERSION


def test_labels_follow_command_override_not_observed_motion():
    assert body_control(True)=='scheduled_hold'
    assert body_control(True, True)=='scheduled_hold'
    assert body_control(False)=='model'
    assert body_control(False, True)=='random'


def test_observer_banner_never_modifies_camera_input():
    camera=np.full((288,288,3),180,np.uint8)
    before=camera.copy()
    rendered=annotate(camera,'scheduled_hold')
    np.testing.assert_array_equal(camera,before)
    assert not np.array_equal(rendered[:50],camera[:50])
    np.testing.assert_array_equal(rendered[60:],camera[60:])
    assert not np.array_equal(rendered,annotate(camera,'model'))


def test_caregiver_subtitle_is_only_on_observer_copy():
    camera=np.full((288,288,3),180,np.uint8);before=camera.copy()
    rendered=annotate(camera,'model',{'label':'Nearby','text':"Here's the ring."})
    np.testing.assert_array_equal(camera,before)
    assert not np.array_equal(rendered[-60:],camera[-60:])
    np.testing.assert_array_equal(rendered[70:170],camera[70:170])


def test_recording_labels_switch_on_the_frame_of_the_applied_command(tmp_path,monkeypatch):
    from mimo_bridge.env import BridgeEnv
    import imageio.v2 as imageio
    camera=np.full((288,288,3),180,np.uint8)
    renderer=SimpleNamespace(render=lambda *args:camera)
    env=BridgeEnv.__new__(BridgeEnv)
    env._video=None;env._audio=None;env._watch=renderer
    env._renderer_obj=lambda:renderer
    env.dt=.02;env.data=SimpleNamespace(time=0.)
    videos={}
    monkeypatch.setattr(imageio,'mimwrite',lambda path,frames,**kw:videos.update({str(path):(list(frames),kw)}))
    path=str(tmp_path/'sample.mp4')
    env.start_video(path,n_frames=4,every=2)
    for i,control in enumerate(['model','scheduled_hold','scheduled_hold','model']):
        env.data.time=(i+1)*env.dt
        env._video_tick(control)
    receipt=json.loads((tmp_path/'sample_control.json').read_text())
    assert receipt['version']==VERSION and receipt['sensory_input_modified'] is False
    assert [x['body_control'] for x in receipt['frames']]==['scheduled_hold','model']
    assert [x['simulation_seconds'] for x in receipt['frames']]==[.04,.08]
    for frames,kw in videos.values():
        assert kw['fps']==25
        np.testing.assert_array_equal(frames[0],annotate(camera,'scheduled_hold'))
        np.testing.assert_array_equal(frames[1],annotate(camera,'model'))


@pytest.mark.parametrize('pooled',[False,True])
def test_control_annotation_transport_cannot_change_the_physical_outcome(tmp_path,pooled):
    from mimo_bridge.datasource import SingleEnvSource,PoolSource
    kw=dict(scene='crib',vision=False,run_dir=str(tmp_path),age=24,caregiver_seconds=0)
    src=PoolSource(1,kw) if pooled else SingleEnvSource(**kw)
    try:
        frames=src.reset();frame=frames[0]
        command=np.zeros((1,len(frame.bones),9),np.float32)
        command[0,:,:3]=frame.bones[:,4:7];command[0,:,3:7]=frame.bones[:,:4];command[:,:,8]=.05
        saved=src.save_full_state()
        expected=src.step(command)[0]
        src.restore_full_state(saved)
        actual=src.step(command,video_control='scheduled_hold')[0]
        np.testing.assert_array_equal(expected.bones,actual.bones)
        np.testing.assert_array_equal(expected.patches,actual.patches)
        assert expected.sim_time==actual.sim_time
    finally:src.close()
