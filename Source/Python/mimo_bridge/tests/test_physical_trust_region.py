# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import math
from types import SimpleNamespace

import torch
from lem.model.efference import rotation_change_deg
from lem.model.motor_synergy import quat_mul, rotvec_to_quat
from lem.model.physical_goal import PhysicalTrustRegion, PhysicalGoalCost


def baseline():
    command=torch.zeros(2,3,9);command[...,6]=1;command[...,7:]=.4
    return command


def test_raw_trust_region_has_finite_zero_rotation_gradients_and_measured_bounds():
    command=baseline();space=PhysicalTrustRegion(None,command,['arm','trunk','gaze'])
    variables=space.init();out,_,_=space.assemble()
    torch.testing.assert_close(out,command)
    (out[...,:3].sum()+out[...,3].sum()).backward()
    assert all(v.grad is None or torch.isfinite(v.grad).all() for v in variables)
    assert space.rotation.grad[:,:2,:3].abs().sum()>0
    with torch.no_grad():
        space.position.add_(4)
        space.rotation.copy_(rotvec_to_quat(torch.ones(2,3,3)*2))
    out,_,_=space.assemble()
    assert float((out[:,:2,:3]-command[:,:2,:3]).abs().max())<=space.position_cm/200+1e-7
    turn=rotation_change_deg(command[:,:2,3:7],out[:,:2,3:7]).norm(dim=-1)
    torch.testing.assert_close(turn,torch.full_like(turn,space.rotation_deg),atol=1e-4,rtol=1e-5)
    torch.testing.assert_close(out[:,2], command[:,2])


def test_rotation_bound_is_equivariant_under_global_rotation_and_quaternion_sign():
    command=baseline();turn=rotvec_to_quat(torch.tensor([.3,.2,-.1]))
    global_q=rotvec_to_quat(torch.tensor([.2,-.4,.7]))
    outputs=[]
    for rotated in [False,True]:
        start=command.clone()
        candidate=turn.expand(2,3,4).clone()
        if rotated:
            start[...,3:7]=quat_mul(global_q,start[...,3:7]);candidate=-quat_mul(global_q,candidate)
        space=PhysicalTrustRegion(None,start,['arm','trunk','gaze']);space.init()
        with torch.no_grad():space.rotation.copy_(candidate)
        outputs.append(space.assemble()[0][...,:2,3:7])
    expected=quat_mul(global_q,outputs[0])
    torch.testing.assert_close((expected*outputs[1]).sum(-1).abs(),torch.ones(2,2),atol=1e-6,rtol=1e-6)


def test_joint_bound_preserves_native_coupling_without_escaping_starting_trust_box():
    geometry={'joints':[{'limits':[0,1]},{'limits':[0,1]}],
              'constraints':[{'terms':[[0,1],[1,-1]],'limits':[-.1,.1]}], 'hands':[]}
    initial=torch.tensor([[.9,.8],[.5,.5]])
    proposed=torch.tensor([[1.5,-1.],[2.,-2.]],requires_grad=True)
    hand=SimpleNamespace(kinematics=SimpleNamespace(geometry=geometry,groups=[[0,1]]),
                         initial=initial,joints=proposed)
    space=PhysicalTrustRegion(hand,baseline(),['arm','trunk','gaze'])
    selected=space.goal_joints()
    assert ((selected-initial).abs()<=math.radians(space.rotation_deg)+1e-7).all()
    assert ((selected[:,0]-selected[:,1]).abs()<=.100001).all()
    assert (selected>=0).all() and (selected<=1).all()
    selected.sum().backward();assert torch.isfinite(proposed.grad).all()


def test_physical_cost_preserves_actual_order_gradient_and_legacy_fallback():
    class Head(torch.nn.Module):
        def body(self,latent,order,achieved):
            out=order.new_zeros(order.shape[0],order.shape[1],12)
            out[...,6:9]=order[...,:3]
            return out
    achieved=torch.zeros(2,3,10);achieved[...,6]=1
    target=achieved.clone();target[:,0,0]=.01
    cost=PhysicalGoalCost(Head(),target,torch.tensor([True,False]),['arm','trunk','gaze'])
    order=baseline().requires_grad_(True)
    fallback=torch.tensor([2.,3.],requires_grad=True)
    error=cost(None,order,achieved,fallback)
    assert error[1]==3
    error.sum().backward()
    assert order.grad[0,0,0]<0
    assert order.grad[1].abs().sum()==0
    assert fallback.grad.tolist()==[0,1]
