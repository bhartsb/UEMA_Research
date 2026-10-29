# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
from pathlib import Path
import numpy as np
import pytest

from mimo_bridge.body_layout import chest_axes,chest_orientation


def test_turning_in_the_pen_is_not_a_roll_and_an_actual_roll_is_visible():
    from scipy.spatial.transform import Rotation
    axes=chest_axes(np.eye(3),[0,1,0],[0,-1,0])
    for yaw in (0,70,150):
        turning=Rotation.from_euler('z',yaw,degrees=True).as_matrix()
        assert abs(chest_orientation(turning,axes)['roll_degrees'])<1e-8
        for roll in (0,45,90,150,180):
            r=turning@Rotation.from_euler('y',roll,degrees=True).as_matrix()
            assert abs(chest_orientation(r,axes)['roll_degrees'])==pytest.approx(roll)


def test_bent_trunk_hinge_commands_match_actual_rotation_derivatives(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.action_decoder import rotmat_log
    from scipy.spatial.transform import Rotation
    s=SingleEnvSource(scene='crib',vision=False,body_layout='trunk_v1',age=24,
                      run_dir=str(tmp_path),caregiver_seconds=0)
    try:
        e=s.env;m=e.model;d=e.data;mj=e._mj
        for h in e.decoder.handles:
            for j in h.joint_ids:
                if m.jnt_limited[j]:d.qpos[m.jnt_qposadr[j]]=np.mean(m.jnt_range[j])+.2*np.ptp(m.jnt_range[j])
        d.qvel[:]=0;mj.mj_forward(m,d);pose=d.qpos.copy()
        checked=[]
        for row in (24,25):
            h=e.decoder.handles[row]
            for joint in h.joint_ids:
                for sign in (-1,1):
                    d.qpos[:]=pose;mj.mj_forward(m,d)
                    initial=d.xmat[h.body_id].reshape(3,3).copy()
                    wanted=sign*1e-4*d.xaxis[joint].copy()
                    target=Rotation.from_rotvec(wanted).as_matrix()@initial
                    cmd=np.zeros((26,9));cmd[row,3:7]=Rotation.from_matrix(target).as_quat();cmd[row,8]=1
                    # This checks the ORIENTATION rows' derivative alone. Since 2026-09-18 the
                    # lower-trunk handle also carries position rows, and a hip rotation moves
                    # the lower body's origin (5 cm lever), so any finite position target -
                    # even the current position - is a second ask the solve balances against
                    # the rotation (measured: 15% of a 1e-4 rad ask with the target held at
                    # the current position). A non-finite target switches the rows off.
                    cmd[row,0:3]=np.nan
                    ctrl=e.decoder.decode(d,cmd)
                    velocity=np.zeros(m.nv)
                    for a,j,dof in zip(h.act_ids,h.joint_ids,h.dof_adrs):velocity[dof]=ctrl[a]/e.decoder.kp
                    mj.mj_integratePos(m,d.qpos,velocity,1);mj.mj_forward(m,d)
                    actual=rotmat_log(d.xmat[h.body_id].reshape(3,3)@initial.T)
                    assert np.linalg.norm(actual-wanted)<5e-7,(m.joint(joint).name,actual,wanted)
                checked.append(m.joint(joint).name)
        assert len(checked)==5
    finally:s.close()


def test_caregiver_follows_displacement_and_waits_after_arrival(tmp_path):
    from mimo_bridge.datasource import SingleEnvSource
    speech=Path(__file__).resolve().parents[4]/'Saved/mimo_bridge/caregiver_speech'
    s=SingleEnvSource(scene='crib',vision=False,audio_real=True,mobile=True,age=24,
        caregiver_mode='responsive',caregiver_speech_dir=str(speech),run_dir=str(tmp_path))
    try:
        e=s.env;c=e.caregiver;cid=e.model.camera('eye_left').id
        # Isolate adult control with a stationary physical body and elapsed audio.
        e.dt=.1;eye=e.data.cam_xpos[cid];eye[:2]+=[.25,-.1]
        previous=e.data.mocap_pos[c.ids['caregiver_head']].copy()
        changes=[];last=True;arrived=None;away=None;returned=None;hidden_at=None
        for _ in range(1300):
            e._audio.speech.cursor+=int(e.dt*e._audio.speech.sample_rate)
            c.tick(e)
            head=e.data.mocap_pos[c.ids['caregiver_head']].copy()
            assert np.linalg.norm(head-previous)<=.05*e.dt+1e-9
            previous=head
            if c.state['scene_phase']=='engaged' and arrived is None:arrived=c.state['time']
            if c.state['scene_phase']=='withdrawing' and away is None:away=c.state['time']
            if c.state['scene_phase']=='absent' and hidden_at is None:hidden_at=c.state['time']
            if hidden_at and c.state['scene_phase']=='presenting' and returned is None:returned=c.state['time']
            now=c.state['object_present']
            if now!=last:changes.append((c.state['time'],now));last=now
        assert away-arrived>=60.-1e-8
        assert returned-hidden_at>=15.-1e-8
        assert sum(not value for _,value in changes)==1
        # The adult tracks the displaced infant from beside the mobile.
        np.testing.assert_allclose(head,c.state['head_goal'],atol=1e-6)
        assert .15<=c.diagnostics(e)['head_horizontal_distance_m']<=.35
        assert all(e.model.geom_rgba[e.model.geom(n+'_geom').id,3]==1 for n in c.ids)
    finally:s.close()
