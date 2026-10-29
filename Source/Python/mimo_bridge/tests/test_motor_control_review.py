# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Motor targets must respect linked anatomy and directional torque limits."""
import numpy as np
import pytest
mujoco = pytest.importorskip('mujoco')
import mimo_bridge.action_decoder as AD


@pytest.mark.parametrize('side', ['L', 'R'])
@pytest.mark.parametrize('pose', [(0., 0., 0.), (.4, -.7, .2)])
def test_arm_equations_include_every_upstream_joint(monkeypatch, side, pose):
    model=mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
      <worldbody><body name="upper"><joint name="shoulder" axis="0 0 1" range="-2 2"/>
        <geom type="sphere" size=".02"/>
        <body name="lower" pos=".2 0 0"><joint name="elbow" axis="0 1 0" range="-2 2"/>
          <geom type="sphere" size=".02"/>
          <body name="hand" pos=".2 0 0"><joint name="wrist" axis="1 0 0" range="-2 2"/>
            <geom type="sphere" size=".02"/>
          </body></body></body></worldbody>
      <actuator><motor joint="shoulder"/><motor joint="elbow"/><motor joint="wrist"/></actuator>
      </mujoco>''')
    names=tuple(f'{n}_{side}' for n in ('wrist','lowerarm','upperarm'))
    monkeypatch.setattr(AD,'HANDLE_NAMES',names)
    monkeypatch.setattr(AD,'BONE_MAP',dict(zip(names,('hand','lower','upper'))))
    monkeypatch.setattr(AD,'HANDLE_CHAINS',dict(zip(names,(('hand','lower','upper'),('lower',),('upper',)))))
    monkeypatch.setattr(AD,'EYE_JOINTS',{})
    data=mujoco.MjData(model);data.qpos[:]=pose;mujoco.mj_forward(model,data)
    decoder=AD.ActionDecoder(model)
    motor=np.zeros((3,9));motor[:,8]=1
    for i,h in enumerate(decoder.handles):
        motor[i,:3]=data.xpos[h.body_id]*100
        motor[i,3:7]=data.xquat[h.body_id][[1,2,3,0]]
    captured=[]
    original=AD.damped_least_squares
    def capture(A,w,lam):
        captured.append(A.copy())
        return original(A,w,lam)
    monkeypatch.setattr(AD,'damped_least_squares',capture)
    decoder.decode(data,motor)
    observed=captured[-1]
    # Numerical motion, independent of the decoder's ancestry/axis construction.
    baseline=data.qpos.copy()
    columns=[]
    for joint in decoder.handles[0].joint_ids:
        minus=mujoco.MjData(model);plus=mujoco.MjData(model)
        minus.qpos[:]=plus.qpos[:]=baseline
        adr=model.jnt_qposadr[joint];eps=1e-6
        minus.qpos[adr]-=eps;plus.qpos[adr]+=eps
        mujoco.mj_forward(model,minus);mujoco.mj_forward(model,plus)
        values=[]
        for i,h in enumerate(decoder.handles):
            values.extend(AD.rotmat_log(plus.xmat[h.body_id].reshape(3,3) @
                minus.xmat[h.body_id].reshape(3,3).T)/(2*eps))
            if decoder.use_position[i]:
                values.extend(AD.POSITION_WEIGHT*(plus.xpos[h.body_id]-minus.xpos[h.body_id])/(2*eps))
        columns.append(values)
    np.testing.assert_allclose(observed,np.array(columns).T,atol=1e-8)


@pytest.mark.parametrize('angle, expected', [(1., .3),(-1., -.2)])
def test_saturation_preserves_joint_ratios_with_asymmetric_limits(monkeypatch,angle,expected):
    model=mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
      <worldbody><body name="base"><joint name="j1" axis="0 0 1" range="-2 2"/>
        <geom type="sphere" size=".02"/>
        <body name="tip" pos=".1 0 0"><joint name="j2" axis="0 0 1" range="-2 2"/>
          <geom type="sphere" size=".02"/></body></body></worldbody>
      <actuator><motor joint="j1" ctrlrange="-.2 1"/>
        <motor joint="j2" ctrlrange="-1 .3"/></actuator></mujoco>''')
    monkeypatch.setattr(AD,'HANDLE_NAMES',('tip',))
    monkeypatch.setattr(AD,'BONE_MAP',{'tip':'tip'})
    monkeypatch.setattr(AD,'HANDLE_CHAINS',{'tip':('tip','base')})
    monkeypatch.setattr(AD,'EYE_JOINTS',{})
    decoder=AD.ActionDecoder(model,kp=10.)
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    motor=np.zeros((1,9));motor[0,3:7]=[0,0,np.sin(angle/2),np.cos(angle/2)];motor[0,8]=1
    np.testing.assert_allclose(decoder.decode(data,motor),[expected,expected],atol=1e-9)
