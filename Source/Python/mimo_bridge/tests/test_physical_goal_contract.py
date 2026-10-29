# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
from collections import deque
import numpy as np
import pytest
import torch

from lem.model.physical_goal import align_pose_memory, selected_pose, pose_error_per_body
from mimo_bridge.replay import ReplayMemory
from mimo_bridge.transition import Pending


def test_goal_pose_is_the_selected_bodys_actual_memory_not_an_embedding():
    achieved = torch.zeros(2, 3, 10)
    achieved[..., 6] = 1
    earlier, later = achieved.clone(), achieved.clone()
    earlier[0, :, 0] = .12
    later[1, :, 1] = .24
    ring = deque([earlier, later], maxlen=200)
    goal, valid = selected_pose(ring, [0, 1], achieved)
    assert valid.tolist() == [True, True]
    assert torch.equal(goal[0], earlier[0]) and torch.equal(goal[1], later[1])
    assert torch.equal(achieved[..., :3], torch.zeros(2, 3, 3))
    align_pose_memory(ring, 4)
    assert list(ring)[:2] == [None, None]
    goal, valid = selected_pose(ring, [0, 3], achieved)
    assert valid.tolist() == [False, True]
    assert torch.equal(goal[0], achieved[0])


def test_physical_goal_units_and_gradients_point_toward_target():
    target = torch.zeros(2, 2, 10)
    target[..., 6] = 1
    target[:, 0, 0] = .01  # Two centimetres.
    position = torch.zeros(2, 2, 3, requires_grad=True)
    quaternion = target[..., 3:7].clone()
    error = pose_error_per_body(position, quaternion, target, [0])
    error.sum().backward()
    assert (position.grad[:, 0, 0] < 0).all()
    assert torch.equal(position.grad[:, 1], torch.zeros(2,3))
    assert torch.equal(pose_error_per_body(target[..., :3]*200, quaternion, target, [0]), torch.zeros(2))


def test_new_hand_replay_keeps_old_observations_but_never_mixes_action_contracts():
    memory = ReplayMemory(['vision'], 2, n_handles=2, size_per_env=32)
    achieved = torch.zeros(1,2,10)
    for i in range(24):
        memory.add({'vision':torch.full((1,1,2), float(i))}, achieved, {},
                   executed=torch.full((1,2,9), float(i)), action_contract=0 if i<18 else 1)
    before = memory.tokens.clone()
    sample = memory.sample_windows(64, 3, 'cpu', action_contract=1)
    assert sample is not None
    assert sample['tokens'][:, 0, 0, 0].min() >= 17  # Action at17 is back-filled on18.
    assert torch.equal(memory.tokens, before)
    assert memory.count == 24
    old = memory.sample_windows(64, 3, 'cpu', action_contract=0)
    assert old is not None and old['tokens'][:, -2, 0, 0].max() <= 16
    memory.mark_reset()
    for i in range(24,40):
        memory.add({'vision':torch.full((1,1,2), float(i))}, achieved, {},
                   executed=torch.full((1,2,9), float(i)), action_contract=1)
    sample = memory.sample_windows(64, 3, 'cpu', action_contract=1)
    assert torch.equal(sample['tokens'][:,1:,0,0]-sample['tokens'][:,:-1,0,0], torch.ones(64,3))


def test_new_hand_transition_requires_exact_physics_substeps():
    p = Pending(0, 0, 1, {}, None, None)
    args = dict(executed_action=np.zeros((2,9)), state_t1={}, frame_t1=2, duration_s=.05,
                action_contract='wrist_relative_joint_v1')
    with pytest.raises(ValueError, match='joint goal and physical substeps'):
        p.complete(**args)
    record = p.complete(**args, hand_joint_goal=np.zeros(44), motor_substeps=np.zeros((5,2,9)))
    assert record.action_contract == 'wrist_relative_joint_v1'
    with pytest.raises(ValueError, match='every handle'):
        p.complete(**args, hand_joint_goal=np.zeros(44), motor_substeps=np.zeros((5,1,9)))


def test_command_readback_accepts_quaternion_sign_but_rejects_another_pose():
    from mimo_bridge.train_mimo import same_hand_goal
    goal=np.zeros((2,26,9));goal[...,6]=1;goal[...,8]=.4
    other=goal.copy();other[...,3:7]*=-1
    assert same_hand_goal(other,goal)
    other[0,0,0]=.1
    assert not same_hand_goal(other,goal)
    other=goal.copy();other[0,0,8]=.5
    assert not same_hand_goal(other,goal)
