# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import copy
import pytest
import torch

from mimo_bridge import copied_execution
from mimo_bridge.hand_target_control import CONTRACT, HandKinematics
from mimo_bridge.lem_model import LemLite


def test_hand_snapshot_rebuilds_native_cache_and_restores_independent_joint_state():
    torch.set_num_threads(1)
    model=LemLite(device='cpu',no_vision=True,imagination=False,next_frame_loss=True,
                  overrides={'model':{'context_length':2}})
    geometry=dict(contract=CONTRACT,handle_names=model.handle_names,constraints=[],
                  bodies=[],hands=[],joints=[{'limits':[-1.,1.]}])
    model.configure_hand_geometry(geometry,torch.tensor([[.25]]))
    model.hand_kinematics._projection_model=object()
    model.hand_kinematics._projection_data=object()
    saved=copied_execution.capture(model)
    assert 'hand_kinematics' not in saved['public']
    model._hand_previous_joints.fill_(.7)
    model.hand_kinematics.geometry['joints'][0]['limits'][0]=-2
    copied_execution.restore(model,saved)
    assert isinstance(model.hand_kinematics,HandKinematics)
    assert model.hand_kinematics._projection_model is None
    assert model.hand_kinematics._projection_data is None
    assert model.hand_kinematics.geometry==geometry
    assert model._hand_previous_joints.item()==.25
    model._hand_previous_joints.fill_(.9)
    assert saved['continuation']['data']['_hand_previous_joints'].item()==.25
    model.hand_kinematics.geometry['joints'][0]['limits'][0]=-2
    with pytest.raises(ValueError,match='pure geometry'):
        copied_execution.capture(model)
