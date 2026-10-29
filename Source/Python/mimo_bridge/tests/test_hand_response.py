# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import copy
import io

import pytest
import torch

from lem.model.hand_response import CalibratedHandResponse,FEATURES,VERSION,content_hash


def fixture():
    geometry=dict(contract='wrist_relative_joint_v1',handle_names=['wrist_L','index_L','gaze'],
        joints=[dict(name='index_bend',limits=[-2.,2.],reference=0.,axis=[0.,0.,1.],anchor=[0.,0.,0.])],
        constraints=[],bodies=[dict(position=[.01,0.,0.],quaternion=[1.,0.,0.,0.],parent=-1,joints=[0])],
        hands=[dict(wrist_index=0,finger_indices=[1],finger_bodies=[0])])
    receipt=dict(version=VERSION,features=FEATURES,horizon_ms=250,geometry_sha256=content_hash(geometry),
        train_bodies=[0,1,2,3],test_bodies=[4,5,6,7],joint_names=['index_bend'],coefficients=[[0.,.5,.4]],
        source_sha256={'private_physical_test':'0'*64},choice=dict(steps=20,lr=.1,raw_joint_bound_deg=11.,nonfinger_orders='fixed'))
    receipt['sha256']=content_hash(receipt)
    return geometry,receipt,CalibratedHandResponse(geometry,receipt)


def test_actual_derivatives_and_frozen_calibration_roundtrip():
    geometry,receipt,model=fixture()
    q=torch.tensor([[.1]],dtype=torch.float64,requires_grad=True)
    v=torch.tensor([[.3]],dtype=torch.float64,requires_grad=True)
    goal=torch.tensor([[.8]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(model.predict,(q,v,goal),eps=1e-6,atol=1e-7,rtol=1e-6)
    assert list(model.parameters())==[]
    assert all(torch.is_tensor(value) for value in model.state_dict().values())
    before=model.predict(q,v,goal)
    buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
    restored=CalibratedHandResponse(geometry,receipt)
    restored.load_state_dict(torch.load(buffer,weights_only=True))
    torch.testing.assert_close(restored.predict(q,v,goal),before,atol=0,rtol=0)
    assert restored.receipt==model.receipt
    damaged=copy.deepcopy(model.state_dict());damaged['coefficients'][0,1]+=.01
    with pytest.raises(ValueError,match='Restored'):
        restored.load_state_dict(damaged)


def test_copied_learner_snapshot_keeps_response_without_a_native_helper():
    from mimo_bridge import copied_execution
    from mimo_bridge.lem_model import LemLite
    torch.set_num_threads(1)
    _,_,response=fixture()
    model=LemLite(device='cpu',no_vision=True,imagination=False,next_frame_loss=True,
                  overrides={'model':{'context_length':2}})
    model.hand_response=response
    parameters=list(model.params)
    saved=copied_execution.capture(model)
    with torch.no_grad():response.coefficients.fill_(.9)
    response.receipt['source_sha256']['private_physical_test']='1'*64
    copied_execution.restore(model,saved)
    torch.testing.assert_close(response.coefficients,torch.tensor([[0.,.5,.4]],dtype=torch.float64),atol=0,rtol=0)
    assert response.receipt['source_sha256']['private_physical_test']=='0'*64
    assert all(after is before for after,before in zip(model.params,parameters))


def test_calibration_rejects_changed_source_geometry_and_overlapping_split():
    geometry,receipt,_=fixture()
    changed=copy.deepcopy(receipt);changed['coefficients'][0][0]=1.
    with pytest.raises(ValueError,match='hash'):CalibratedHandResponse(geometry,changed)
    changed=copy.deepcopy(geometry);changed['joints'][0]['limits'][1]=3.
    with pytest.raises(ValueError,match='geometry'):CalibratedHandResponse(changed,receipt)
    changed=copy.deepcopy(receipt);changed['test_bodies'][0]=0
    changed['sha256']=content_hash({k:v for k,v in changed.items() if k!='sha256'})
    with pytest.raises(ValueError,match='split'):CalibratedHandResponse(geometry,changed)


def selection_fixture(model):
    command=torch.zeros(3,3,9);command[...,6]=1.;command[...,8]=.4
    command[:,0,:3]=torch.tensor([5.,3.,8.])
    raw=torch.full((3,1),.3);previous=torch.zeros_like(raw);filtered=.35*raw
    command=model.kinematics.assemble_world(command,filtered,position_scale=1.)
    target=torch.zeros(3,3,10,dtype=torch.float64);target[...,6]=1.
    target=model.kinematics.assemble_world(target,torch.tensor([[.1],[.0525],[.15]],dtype=torch.float64))
    return dict(canonical_execution=command,raw_initial_goal=raw,previous_goal=previous,smoothing=.35,
        current_angles=torch.zeros_like(raw),current_velocity=torch.zeros_like(raw),target_pose=target,
        valid=torch.tensor([True,True,False]))


def test_partial_batch_keeps_each_baseline_and_filters_once():
    _,_,model=fixture();inputs=selection_fixture(model)
    selected=model.choose_hand_goal(**inputs)
    command=selected['command'];goal=selected['joint_goal']
    torch.testing.assert_close(command[:,[0,2]],inputs['canonical_execution'][:,[0,2]],atol=0,rtol=0)
    torch.testing.assert_close(command[...,7:],inputs['canonical_execution'][...,7:],atol=0,rtol=0)
    torch.testing.assert_close(command[2],inputs['canonical_execution'][2],atol=0,rtol=0)
    assert selected['metadata']['selected_iteration'][0]>0
    assert selected['metadata']['selected_iteration'][1:]==[0,0]
    assert float(goal[0,0])>.105
    assert float(goal[0,0])<=.35*(.3+torch.deg2rad(torch.tensor(11.)).item())+1e-7
    torch.testing.assert_close(goal,.35*selected['raw_joint_goal'],atol=1e-7,rtol=0)
    inputs['canonical_execution'][2,1,0]+=.1
    with pytest.raises(ValueError,match='baseline'):model.choose_hand_goal(**inputs)
