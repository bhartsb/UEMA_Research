# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
from types import SimpleNamespace
import numpy as np

from mimo_bridge.tools.gof_probe_harness import Session


class Kinematics:
    geometry={'hands':[{'wrist_index':0,'finger_indices':[1]}]}
    def __init__(self):self.solves=0
    def project_world_orders(self,order,position_scale):
        self.solves+=1
        canonical=order.copy();canonical[:,1,0]=canonical[:,0,0]+.5
        return canonical,np.array([[.5]]),[]
    def validate_joints(self,value,tolerance):assert value.shape==(1,1)
    def canonical_from_joints(self,order,joints,position_scale):
        out=order.copy();out[1,0]=out[0,0]+joints[0]
        return out


def session():
    s=Session.__new__(Session)
    s.model=SimpleNamespace(hand_target_mode='wrist_relative_v1',hand_kinematics=Kinematics())
    s.n_handles=26;s.reps=5;s.order_id=0;s._probe_hand_cache=None
    s.handle_names=['h%d'%i for i in range(26)];s.world_args=SimpleNamespace()  # limb flags off
    s.eyes_rad=lambda:np.zeros(2)
    s.calls=[]
    def step(order,gaze,mask,render,hand_joint_goal):
        s.calls.append((order.copy(),hand_joint_goal.copy()))
        actual=order[0].copy();actual[1,0]+=len(s.calls)*.01
        return [SimpleNamespace(meta=dict(gaze_cmd=[0.,0.],task={},
            hand_target_contract='wrist_relative_joint_v1',hand_joint_goal=hand_joint_goal[0].copy(),
            executed_motor_targets=actual))]
    s.src=SimpleNamespace(step=step)
    return s


def test_prediction_and_execution_share_canonical_order_and_record_every_substep():
    s=session();order=np.zeros((1,26,9),np.float32);order[:,:,6]=1;order[:,1,0]=999
    expected=s.exec_order(order)
    s.act(expected)
    assert s.model.hand_kinematics.solves==1
    # the model is told the command as issued (the trainer's ledger); the world receives the canonical form
    np.testing.assert_array_equal(s.executed,expected)
    canonical=expected.copy();canonical[:,1,0]=canonical[:,0,0]+.5
    assert len(s.calls)==5
    for motor,joint_goal in s.calls:
        np.testing.assert_array_equal(motor,canonical)
        np.testing.assert_array_equal(joint_goal,[[.5]])
    assert s.last_motor_substeps.shape==(1,5,26,9)
    assert len(np.unique(s.last_motor_substeps[0,:,1,0]))==5
    np.testing.assert_array_equal(s.frames[0].meta['motor_substeps'],s.last_motor_substeps[0])


def test_policy_holds_the_selected_joint_goal_with_its_five_step_motor_order():
    s=session();s.hold=5;s._policy_i=0;s._held=None;s._held_hand_goal=None
    held=[]
    s.act=lambda order,**kw:held.append((order.copy(),kw['hand_joint_goal'].copy(),kw['new_order']))
    for i in range(6):
        s.res=SimpleNamespace(motor_for_action=np.full((1,26,9),i,dtype=np.float32),
                             gaze_out=np.zeros((1,2)),hand_joint_goal=np.array([[i+.2]]))
        s.policy_act()
    for order,goal,new in held[:5]:
        assert np.all(order==0)
        np.testing.assert_array_equal(goal,[[.2]])
    assert [item[2] for item in held]==[True,False,False,False,False,True]
    np.testing.assert_array_equal(held[5][1],[[5.2]])


def test_explicit_prediction_goal_is_retained_when_its_canonical_order_is_sent():
    s=session();order=np.zeros((1,26,9),np.float32);order[:,:,6]=1
    expected=s.exec_order(order,hand_joint_goal=np.array([[.7]]))
    s.act(expected,hand_joint_goal=np.array([[.7]]))   # exec_order no longer carries the goal through the cache
    assert s.model.hand_kinematics.solves==0
    np.testing.assert_array_equal(s.last_hand_joint_goal,[[.7]])


def test_restored_arm_transports_joint_goal_without_resolving_canonical_poses():
    s=session();order=np.zeros((1,26,9),np.float32);order[:,:,6]=1
    goal=np.array([[.7]])
    canonical=s.exec_order(order,hand_joint_goal=goal)
    s.restore=lambda snap:setattr(s,'_probe_hand_cache',None)
    s.observe=lambda:None
    s.readout=lambda:s.last_hand_joint_goal.copy()
    frames=[]
    out=s.run_arm({},canonical,steps=3,hand_joint_goal=goal,
                  on_env_step=lambda session,index:frames.append(index))
    assert len(out)==3 and len(frames)==15
    assert s.model.hand_kinematics.solves==0
    for row in out:np.testing.assert_array_equal(row,goal)
