# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import numpy as np
import pytest
import torch

from mimo_bridge.body_layout import names_for, extend_limits
from mimo_bridge.tools.migrate_trunk import expanded


def test_layout_appends_without_changing_eye_or_legacy_wire_contract():
    from tokenizer.config import MOTOR_MAX_HANDLES, PROPRIO_MAX_BONES
    legacy, trunk = names_for('legacy'), names_for('trunk_v1')
    assert trunk[:24] == legacy and trunk[23] == 'gaze'
    assert MOTOR_MAX_HANDLES == PROPRIO_MAX_BONES == 24
    assert extend_limits({'handle_mask':[False]*24}, 26)['handle_mask'] == [False]*24+[True]*2


def test_controller_input_migration_preserves_meaning_and_moments():
    old = torch.arange(1224.).reshape(1,-1)
    new = expanded('motor_controller.input_proj.weight',old,torch.zeros(1,1262))
    assert torch.equal(new[:,:1008],old[:,:1008])
    assert torch.equal(new[:,1028:1244],old[:,1008:])
    assert new[:,1008:1028].count_nonzero() == new[:,1244:].count_nonzero() == 0


def test_trunk_commands_reach_each_joint_and_return_actual_sensation(tmp_path):
    mujoco = pytest.importorskip('mujoco')
    from mimo_bridge.datasource import SingleEnvSource
    source = SingleEnvSource(scene='crib', run_dir=str(tmp_path), vision=False,
                             body_layout='trunk_v1', age=24, caregiver_seconds=0)
    try:
        env = source.env
        frames = source.reset()
        assert frames[0].tokenized.proprio.shape == (26,12)
        assert frames[0].tokenized.motor_handles.shape == (26,10)
        start = env.save_full_state()
        checked = set()
        for row in (24,25):
            info = env.decoder.handles[row]
            for joint, actuator in zip(info.joint_ids,info.act_ids):
                dq = []
                for sign in (-1,1):
                    env.restore_full_state(start)
                    bones = env.proprio.read(0).bones
                    cmd = np.zeros((26,9))
                    cmd[:,:3],cmd[:,3:7],cmd[:,8] = bones[:,4:7],bones[:,:4],1
                    axis = env.data.xaxis[joint].copy()
                    angle = sign*np.radians(10.)
                    delta = np.r_[np.cos(angle/2),axis*np.sin(angle/2)]
                    target = np.zeros(4)
                    mujoco.mju_mulQuat(target,delta,env.data.xquat[info.body_id])
                    cmd[row,3:7] = target[[1,2,3,0]]
                    ctrl = env.decoder.decode(env.data,cmd)
                    assert sign*ctrl[actuator] > 0
                    before = float(env.data.qpos[env.model.jnt_qposadr[joint]])
                    for _ in range(25):
                        frame = source.step(cmd)[0]
                    dq.append(float(env.data.qpos[env.model.jnt_qposadr[joint]])-before)
                    np.testing.assert_allclose(frame.tokenized.motor_handles[row,3:7],frame.bones[row,:4])
                assert dq[1] > dq[0], (env.model.joint(joint).name,dq)
                checked.add(env.model.actuator(actuator).name)
        assert checked == {'act:hip_bend','act:hip_twist','act:hip_lean','act:chest_twist','act:chest_lean'}
    finally:
        source.close()
