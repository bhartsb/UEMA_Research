# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import copy
import numpy as np
import pytest
import torch
from mimo_bridge.datasource import SingleEnvSource
from mimo_bridge.lem_model import LemLite
from mimo_bridge import continuation_state


def test_incomplete_required_episode_never_silently_resets_the_world(tmp_path):
    import json
    (tmp_path/'model_receipt.json').write_text(json.dumps({'episode_continuation':continuation_state.VERSION}))
    with pytest.raises(ValueError,match='requires a completed'):
        continuation_state.load(tmp_path)


def hold(frame):
    cmd=np.zeros((1,len(frame.bones),9),np.float32)
    cmd[0,:,:3]=frame.bones[:,4:7];cmd[0,:,3:7]=frame.bones[:,:4];cmd[:,:,8]=.05
    return cmd


def test_physical_source_continues_identically_in_a_new_environment(tmp_path):
    pytest.importorskip('mujoco')
    kw=dict(scene='crib',vision=False,run_dir=str(tmp_path),age=24,caregiver_seconds=0,
            body_function='modulate',contact_profile='mechanical_v2')
    src=SingleEnvSource(**kw);restored=None
    try:
        f=src.reset();cmd=hold(f[0])
        for _ in range(5):f=src.step(cmd)
        saved=src.save_full_state();expected=[src.step(cmd)[0] for _ in range(3)]
        restored=SingleEnvSource(**kw);again=restored.restore_full_state(saved)
        assert again[0].sim_time==f[0].sim_time
        for e in expected:
            a=restored.step(cmd)[0]
            np.testing.assert_array_equal(e.bones,a.bones)
            np.testing.assert_array_equal(e.patches,a.patches)
            assert e.meta['body_function']==a.meta['body_function']
    finally:
        src.close()
        if restored:restored.close()


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA continuation requires a GPU'))])
def test_live_self_learning_and_exact_learner_continuation(tmp_path, device):
    torch.set_num_threads(1)
    src=SingleEnvSource(scene='crib',vision=False,run_dir=str(tmp_path/'world'),age=24,
                        caregiver_seconds=0,body_function='observe',contact_profile='mechanical_v2')
    try:
        m=LemLite(device=device,imagination=False,no_vision=True,next_frame_loss=True,
                  persistent_memory='on',self_state='perceive',checkpoint_dir=str(tmp_path/'checkpoints'))
        frames=src.reset();cmd=None;dev=m.dev_limits(0)
        # Fill the actual collapse-watch writer's history far enough for the
        # reporting path to run. A short, CPU-only resume missed mixed devices.
        steps=40
        for i in range(steps):
            r=m.step(frames,dev,train=True,step_index=i,executed_action=cmd)
            assert np.isfinite(r.loss)
            cmd=r.motor_for_action;frames=src.step(cmd)
        assert 'consequence_error_5' in m.last_self_state
        assert m.self_state.residual.weight.abs().sum()>0
        path=m.save_checkpoint(steps)
        continuation_state.save(path,src,m,dict(step=steps,cmd=cmd),dict(example='test'))
        state=continuation_state.load(path)
        expected=m.step(frames,dev,train=False,step_index=steps,executed_action=cmd)
        expected_std=m.token_temporal_std()
        new=LemLite(device=device,imagination=False,no_vision=True,next_frame_loss=True,
                    persistent_memory='on',self_state='perceive')
        new.load_checkpoint(path);continuation_state.restore_learner(new,state['learner'])
        actual=new.step(frames,dev,train=False,step_index=steps,executed_action=cmd)
        np.testing.assert_array_equal(expected.motor_for_action,actual.motor_for_action)
        assert actual.loss==expected.loss
        assert new.token_temporal_std()==expected_std
        assert expected_std and all(np.isfinite(v) for v in expected_std.values())
        assert all(t.device.type=='cpu' for ring in new.token_ring.values() for t in ring)
        assert new.context_buffer.device==new.device
    finally:src.close()


def test_reported_motor_availability_really_limits_mujoco_force(tmp_path):
    src=SingleEnvSource(scene='crib',vision=False,run_dir=str(tmp_path),age=24,
                        caregiver_seconds=0,body_function='modulate')
    try:
        src.reset();e=src.env;known=e.proprio.act_set>=0
        e.data.ctrl[:]=.4;e.proprio.fatigue.cap[:]=1
        e._apply_motor_function();e._mj.mj_forward(e.model,e.data)
        normal=e.data.actuator_force.copy()
        e.proprio.fatigue.cap[:]=.25;e._apply_motor_function();e._mj.mj_forward(e.model,e.data)
        np.testing.assert_allclose(e.data.actuator_force[known],normal[known]*.25,atol=1e-12)
        np.testing.assert_array_equal(e.data.actuator_force[~known],normal[~known])
        src.reset();np.testing.assert_array_equal(e.model.actuator_gainprm,e._native_actuator_gain)
    finally:src.close()


def test_neutral_extension_keeps_saved_episode_and_old_learning(tmp_path):
    torch.set_num_threads(1)
    src=SingleEnvSource(scene='crib',vision=False,run_dir=str(tmp_path/'world'),age=24,
                        caregiver_seconds=0,body_function='off',contact_profile='mechanical_v2')
    restored=None
    try:
        old=LemLite(device='cpu',imagination=False,no_vision=True,next_frame_loss=True,
                    persistent_memory='on',checkpoint_dir=str(tmp_path/'checkpoints'))
        frames=src.reset();dev=old.dev_limits(0);cmd=None
        for i in range(3):
            cmd=old.step(frames,dev,train=True,step_index=i,executed_action=cmd).motor_for_action
            frames=src.step(cmd)
        path=old.save_checkpoint(3)
        continuation_state.save(path,src,old,dict(step=3,cmd=cmd,frame_meta=[f.meta for f in frames]),
                                dict(self_state='off',body_function='off'))
        original=continuation_state.load(path)
        upgraded=continuation_state.add_observed_self(original)
        assert original['config']['self_state']=='off'
        assert original['worlds'][0]['world']['attrs']['body_function'] is None
        assert upgraded['config']==dict(self_state='learn',body_function='observe')
        np.testing.assert_array_equal(upgraded['worlds'][0]['payload']['patches'],frames[0].patches)
        assert upgraded['worlds'][0]['payload']['frame_bytes']==original['worlds'][0]['payload']['frame_bytes']
        migrated=tmp_path/'migrated';migrated.mkdir()
        continuation_state.write_episode(migrated,upgraded)
        upgraded=continuation_state.load(migrated)
        restored=SingleEnvSource(**upgraded['worlds'][0]['env_kwargs'])
        new_frames=restored.restore_full_state(upgraded['worlds'])
        new=LemLite(device='cpu',imagination=False,no_vision=True,next_frame_loss=True,
                    persistent_memory='on',self_state='learn')
        new.load_checkpoint(path);continuation_state.restore_learner(new,upgraded['learner'])
        # Restore the SAME random stream before each branch, as a real resumed
        # learner does after constructors have consumed initialization randomness.
        continuation_state.restore_learner(old,original['learner'])
        expected=old.step(frames,dev,train=True,step_index=3,executed_action=cmd)
        continuation_state.restore_learner(new,upgraded['learner'])
        actual=new.step(new_frames,dev,train=True,step_index=3,executed_action=cmd)
        np.testing.assert_array_equal(actual.motor_for_action,expected.motor_for_action)
        from mimo_bridge.tools.migrate_trunk import modules
        new_modules=modules(new)
        for name,module in modules(old).items():
            for key,value in module.state_dict().items():
                assert torch.equal(value,new_modules[name].state_dict()[key]),(name,key)
        for _ in range(3):
            expected_frame=src.step(cmd)[0];actual_frame=restored.step(cmd)[0]
            np.testing.assert_array_equal(actual_frame.bones,expected_frame.bones)
            np.testing.assert_array_equal(actual_frame.patches,expected_frame.patches)
    finally:
        src.close()
        if restored:restored.close()
