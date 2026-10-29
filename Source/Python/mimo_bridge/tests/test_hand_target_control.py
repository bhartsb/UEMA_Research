# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Feasible hand coordinates must match native kinematics and survive replay."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip('mujoco')
from scipy.spatial.transform import Rotation
from mimo_bridge.hand_target_control import (
    DIGITS, HandActionSpace, HandKinematics, HandTargetControl, geometry_from_model)


@pytest.fixture
def body():
    # Nonzero anchors, reference angles and several hinges on one body expose
    # frame/order errors that a single zero-anchor chain would miss.
    hands = []
    tendons = []
    names, body_names = ['torso'], ['torso']
    for side, prefix, y in [('L', 'left', .1), ('R', 'right', -.1)]:
        names.append('wrist_'+side); body_names.append(prefix+'_wrist')
        fingers = []
        for index, digit in enumerate(DIGITS):
            name = prefix+'_'+digit
            names.append(digit+'_'+side); body_names.append(name+'_tip')
            fingers.append(f'''<body name="{name}_base" pos="{index*.015} 0 .02" quat=".9800666 .1986693 0 0">
              <joint name="{name}_bend" axis="1 0 0" pos=".001 .002 .003" range="-.3 1.5" ref=".1"/>
              <joint name="{name}_spread" axis="0 0 1" pos="-.001 .002 0" range="-.3 .3"/>
              <geom type="sphere" size=".005"/>
              <body name="{name}_tip" pos="0 .04 0">
                <joint name="{name}_end" axis="1 0 0" range="-.2 1.5"/>
                <geom type="sphere" size=".005"/>
              </body></body>''')
        for a, b in zip(DIGITS[1:-1], DIGITS[2:]):
            tendons.append(f'''<fixed name="finger_couple_{prefix}_{a}" limited="true" range="-.25 .25">
              <joint joint="{prefix}_{a}_bend" coef="1"/><joint joint="{prefix}_{b}_bend" coef="-1"/>
              </fixed>''')
        hands.append(f'''<body name="{prefix}_wrist" pos="0 {y} .2">
            <joint name="{prefix}_arm" axis="0 1 0" range="-1 1"/><geom type="sphere" size=".01"/>
            {''.join(fingers)}</body>''')
    xml = f'''<mujoco><compiler angle="radian"/><worldbody><body name="torso">
      <geom type="sphere" size=".03"/>{''.join(hands)}</body></worldbody>
      <tendon>{''.join(tendons)}</tendon></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    decoder = SimpleNamespace(handle_names=names,
        handles=[SimpleNamespace(body_id=model.body(name).id) for name in body_names])
    geometry = geometry_from_model(model, decoder)
    command = np.zeros((len(names), 9)); command[:, 8] = .47; command[:, 7] = 2
    for index, handle in enumerate(decoder.handles):
        command[index, :3] = data.xpos[handle.body_id]*100
        command[index, 3:7] = data.xquat[handle.body_id][[1,2,3,0]]
    return model, data, decoder, geometry, command


def test_torch_fk_matches_native_compiled_frames(body):
    model, data, decoder, geometry, command = body
    kin = HandKinematics(geometry)
    rng = torch.Generator().manual_seed(815)
    for _ in range(12):
        angles = kin.project_joints(torch.randn(kin.n_joints, generator=rng, dtype=torch.float64))
        kin.validate_joints(angles.numpy())
        for joint, angle in zip(geometry['joints'], angles):
            data.qpos[joint['qpos_address']] = angle
        mujoco.mj_kinematics(model, data)
        local_p, local_q = kin.local_poses(angles)
        for hand in geometry['hands']:
            wrist = hand['wrist_body']; rotation = data.xmat[wrist].reshape(3,3)
            for index, body_index in zip(hand['finger_indices'], hand['finger_bodies']):
                native = decoder.handles[index].body_id
                np.testing.assert_allclose(local_p[body_index], rotation.T@(data.xpos[native]-data.xpos[wrist]), atol=1e-12)
                np.testing.assert_allclose(Rotation.from_quat(local_q[body_index].numpy()[[1,2,3,0]]).as_matrix(),
                    rotation.T@data.xmat[native].reshape(3,3), atol=1e-12)


def test_projection_enforces_all_native_limits_and_preserves_feasible_goals(body):
    kin = HandKinematics(body[3]); rng = torch.Generator().manual_seed(871)
    raw = torch.randn(1000, kin.n_joints, generator=rng, dtype=torch.float64)*100
    projected = kin.project_joints(raw)
    kin.validate_joints(projected.numpy(), tolerance=1e-12)
    torch.testing.assert_close(kin.project_joints(projected), projected, atol=1e-12, rtol=0)
    # No dependency joins thumb to the other fingers.
    index = next(i for i,j in enumerate(body[3]['joints']) if j['name']=='left_thumb_end')
    torch.testing.assert_close(projected[:, index], raw[:, index].clamp(-.2,1.5))


def test_fk_and_projection_have_correct_gradients(body):
    kin = HandKinematics(body[3])
    # Interior, away from nonsmooth clamp boundaries and active-set ties.
    x = torch.full((kin.n_joints,), .18, dtype=torch.float64, requires_grad=True)
    x.data[1::3] = .03
    template = torch.from_numpy(body[4])
    assert torch.autograd.gradcheck(lambda q: kin.assemble_world(template, kin.project_joints(q), 1.),
                                   (x,), eps=1e-6, atol=1e-5, rtol=1e-4)
    # A uniquely active coupling inequality also uses the real derivative.
    active=x.detach().clone()
    index=next(i for i,j in enumerate(body[3]['joints']) if j['name']=='left_index_bend')
    active[index]=.9
    active.requires_grad_(True)
    assert torch.autograd.gradcheck(kin.project_joints,(active,),eps=1e-6,atol=1e-5,rtol=1e-4)


def test_world_projection_recovers_feasible_reference(body):
    kin = HandKinematics(body[3])
    angles = np.full(kin.n_joints, .15)
    reference = kin.canonical_from_joints(body[4], angles)
    canonical, solved, _ = kin.project_world_orders(reference, 1.)
    kin.validate_joints(solved)
    np.testing.assert_allclose(canonical[:,:3],reference[:,:3],atol=.002)
    np.testing.assert_allclose(np.abs((canonical[:,3:7]*reference[:,3:7]).sum(-1)),1,atol=1e-7)


def test_native_inverse_gradient_at_coupled_poses_and_near_pi(body):
    from mimo_bridge.hand_target_control import _native_inverse_objective
    kin=HandKinematics(body[3]);model,data=kin._mini_model();rng=np.random.default_rng(418)
    for hand in kin.geometry['hands']:
        indices=sorted({i for b in kin.geometry['bodies'] if b['side']==hand['side'] for i in b['joints']})
        joints=[model.joint(kin.geometry['joints'][i]['name']).id for i in indices]
        addresses=model.jnt_qposadr[joints]
        bodies=[model.body(kin.geometry['bodies'][i]['name']).id for i in hand['finger_bodies']]
        for case in range(6):
            q=kin.project_joints(torch.from_numpy(rng.uniform(-.1,.8,kin.n_joints))).numpy()[indices]
            data.qpos[addresses]=q;mujoco.mj_kinematics(model,data)
            position=data.xpos[bodies].copy()+.003
            axis=np.asarray([1.,2.,3.]);axis/=np.linalg.norm(axis)
            angle=(np.pi-1e-3)*(1 if case==4 else -1) if case>=4 else .2
            target=Rotation.from_rotvec(axis*angle).as_matrix()@data.xmat[bodies].reshape(-1,3,3)
            objective=_native_inverse_objective(model,data,joints,bodies,position,target,8.)
            _,analytic=objective(q);numeric=np.zeros_like(q)
            for i in range(len(q)):
                offset=np.zeros_like(q);offset[i]=1e-6
                numeric[i]=(objective(q+offset)[0]-objective(q-offset)[0])/2e-6
            np.testing.assert_allclose(analytic,numeric,atol=1e-6,rtol=1e-5)
    request=body[4].copy();flipped=request.copy();flipped[:,3:7]*=-1
    canonical,angles,_=kin.project_world_orders(request,1.)
    _,opposite_angles,_=kin.project_world_orders(flipped,1.)
    np.testing.assert_array_equal(angles,opposite_angles)


@pytest.mark.parametrize('magnitude',[0.,1e-14])
def test_zero_quaternion_uses_decoder_identity_without_changing_strength(body,magnitude):
    from mimo_bridge.action_decoder import quat_xyzw_to_mat
    from mimo_bridge.hand_target_control import _rotation_xyzw
    model,data,decoder,geometry,command=body
    zero=np.zeros_like(command);zero[:,3]=magnitude
    identity=zero.copy();identity[:,3:7]=[0.,0.,0.,1.]
    np.testing.assert_array_equal(_rotation_xyzw(zero[0,3:7]).as_matrix(),
                                  quat_xyzw_to_mat(zero[0,3:7]))
    kin=HandKinematics(geometry)
    canonical,angles,_=kin.project_world_orders(zero,1.)
    reference,reference_angles,_=kin.project_world_orders(identity,1.)
    np.testing.assert_array_equal(angles,reference_angles)
    fingers=[i for h in geometry['hands'] for i in h['finger_indices']]
    np.testing.assert_array_equal(canonical[fingers],reference[fingers])
    np.testing.assert_array_equal(kin.canonical_from_joints(zero,angles)[fingers],reference[fingers])
    controller=HandTargetControl(model,decoder)
    executed,cache=controller.execute(data,zero)
    np.testing.assert_array_equal(executed[:,7:],zero[:,7:])
    others=[i for i in range(len(zero)) if i not in fingers]
    np.testing.assert_array_equal(executed[others],zero[others])
    repeated,_=controller.execute(data,zero,cache)
    np.testing.assert_array_equal(repeated,executed)


def test_execution_tracks_moving_wrist_and_cache_replays_without_mutation(body, monkeypatch):
    model, data, decoder, geometry, command = body
    controller = HandTargetControl(model, decoder)
    angles = np.full(controller.kinematics.n_joints,.15)
    canonical = controller.kinematics.canonical_from_joints(command,angles)
    before = data.qpos.copy()
    first, cache = controller.execute(data,canonical,joint_goal=angles)
    np.testing.assert_array_equal(data.qpos,before)
    original = copy.deepcopy(cache)
    monkeypatch.setattr(controller.kinematics,'project_world_orders',lambda *a,**k: pytest.fail('Cache unnecessarily re-solved'))
    # Wrist motion cannot change the requested hand shape.
    data.qpos[model.jnt_qposadr[model.joint('left_arm').id]] += .4
    mujoco.mj_kinematics(model,data)
    second, cache2 = controller.execute(data,canonical,cache,angles)
    fingers = geometry['hands'][0]['finger_indices']
    assert not np.array_equal(first[fingers,:3],second[fingers,:3])
    repeated, _ = controller.execute(data,canonical,copy.deepcopy(original),angles)
    np.testing.assert_array_equal(second,repeated)
    nonfingers = [i for i in range(len(command)) if i not in controller._finger_bodies]
    np.testing.assert_array_equal(second[nonfingers],command[nonfingers])
    np.testing.assert_array_equal(second[:,7:],command[:,7:])
    changed=canonical.copy(); changed[0,:3]+=1; changed[:,8]=.23
    newer, cache3=controller.execute(data,changed,cache2,angles)
    np.testing.assert_array_equal(newer[0],changed[0])
    np.testing.assert_array_equal(newer[:,8],changed[:,8])
    np.testing.assert_array_equal(cache3['canonical'][0],changed[0])


def test_action_space_gradients_and_invariant_local_encoding(body):
    model,data,decoder,geometry,command=body
    kin=HandKinematics(geometry)
    template=torch.from_numpy(command[None]).float(); template[:,:,:3]/=200
    angles=torch.full((1,kin.n_joints),.15)
    space=HandActionSpace(geometry,template,angles); variables=space.init()
    world,_,_=space.assemble()
    world.square().sum().backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in variables)
    kin.validate_joints(kin.project_joints(space.joints).detach().numpy())
    local=kin.local_encoding(world)
    translated=template.clone()
    for hand in geometry['hands']:translated[:,hand['wrist_index'],:3]+=.12
    moved=kin.assemble_world(translated,angles)
    local_moved=kin.local_encoding(moved)
    fingers=[i for hand in geometry['hands'] for i in hand['finger_indices']]
    torch.testing.assert_close(local[:,fingers,:7],local_moved[:,fingers,:7],atol=1e-7,rtol=1e-5)


def test_composed_action_space_retains_base_gradients_and_penalty(body):
    geometry=body[3];kin=HandKinematics(geometry)
    template=torch.from_numpy(body[4][None]).float();template[:,:,:3]/=200
    class Base:
        def init(self):
            self.offset=torch.tensor(.1,requires_grad=True)
            return [self.offset]
        def assemble(self):
            command=template.clone();command[:,:,:3]+=self.offset
            return command,command[:,:,8:9],self.offset.square()
        def frozen_modules(self):return ['frozen_test_module']
        def penalty_per_body(self):return self.offset.square().expand(1)
        def summary(self):return {'base_summary_retained':True}
    base=Base();space=HandActionSpace(geometry,template,torch.full((1,kin.n_joints),.15),base_space=base)
    variables=space.init();world,strength,penalty=space.assemble()
    (world.square().sum()+penalty).backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in variables)
    assert space.frozen_modules()==['frozen_test_module']
    assert space.summary()['base_summary_retained']
    torch.testing.assert_close(space.penalty_per_body(),base.penalty_per_body())
    fingers={i for h in geometry['hands'] for i in h['finger_indices']}
    others=[i for i in range(template.shape[1]) if i not in fingers]
    torch.testing.assert_close(world[:,others],base.assemble()[0][:,others])
    torch.testing.assert_close(world[:,:,7:],template[:,:,7:])


def test_selected_joint_goal_and_live_filter_reproduce_planner_candidate(body,monkeypatch):
    from lem.model.intention_generator import IntentionGenerator
    from mimo_bridge.lem_model import LemLite
    geometry=body[3];kin=HandKinematics(geometry)
    template=torch.from_numpy(body[4][None]).float().repeat(2,1,1)
    template[:,:,:3]/=200;template[:,:,7:]=.47
    joints=torch.full((2,kin.n_joints),.15)
    template=kin.assemble_world(template,joints)
    previous_joints=torch.full_like(joints,.1)
    previous=kin.assemble_world(template,previous_joints).detach();previous[:,:,:3]*=200
    achieved=torch.cat((template[:,:,:7],torch.zeros(2,template.shape[1],3)),-1)
    dummy=SimpleNamespace(hand_kinematics=kin)
    space=HandActionSpace(geometry,template,joints)
    limits={'rom_limit':1.,'strength_limit':1.}
    def execution(raw,dev,handle_state=None):
        command=LemLite.planned_execution(raw,dev,handle_state,previous=previous,smoothing=.35)
        smoothed=LemLite._smooth_hand_joints(dummy,space.joints,previous_joints,.35)
        return kin.assemble_world(command,smoothed,1.)
    finger=geometry['hands'][0]['finger_indices'][1]
    def transition(_fm,_enc,_heads,_state,action,**kw):
        return SimpleNamespace(tokens={'touch':action[:,finger,:3]})
    monkeypatch.setattr('lem.model.predictive_state.transition',transition)
    modules=[torch.nn.Linear(3,3) for _ in range(3)]
    config={'body':{'n_handles':template.shape[1]},'model':{'d_model':3},
        '_active_modalities':['touch','motor'],
        'training':{'intention':{'n_steps':5,'lr':.1,'strength_reg':0.,'effort_per_body':True}}}
    generator=IntentionGenerator(config,*modules)
    space.init()
    first=execution(template,limits,achieved).detach();first[:,:,:3]/=200
    goal=first[:,finger,:3].clone();goal[1,0]+=.03
    selected,meta=generator.generate_action({'touch':torch.zeros(2,3)}, {'touch':goal},template,
        execution_transform=execution,dev_limits=limits,achieved_state=achieved,
        action_space=space,source_mask={'touch':'real'})
    # The optimizer restores each body's selected parameters, not its last try.
    torch.testing.assert_close(selected,space.assemble()[0].detach(),atol=0,rtol=0)
    assert meta['selected_iteration'][0]==0
    raw_joint_goal=kin.project_joints(space.joints).detach()
    live=LemLite.execution_transform(dummy,selected,limits,achieved)
    live=LemLite.smooth_blend(live,previous,.35)
    smoothed=LemLite._smooth_hand_joints(dummy,raw_joint_goal,previous_joints,.35)
    live=kin.assemble_world(live,smoothed,1.)
    torch.testing.assert_close(live,execution(selected,limits,achieved).detach(),atol=2e-5,rtol=1e-6)
    # The explicit radians supplied to the environment produce the same canonical
    # world command; native FK independently checks the planner's Torch transform.
    native=kin.canonical_from_joints(live[0].numpy(),smoothed[0].numpy())
    np.testing.assert_allclose(native[:,:3],live[0,:,:3],atol=2e-5)
    np.testing.assert_allclose(abs((native[:,3:7]*live[0,:,3:7].numpy()).sum(-1)),1,atol=1e-6)
