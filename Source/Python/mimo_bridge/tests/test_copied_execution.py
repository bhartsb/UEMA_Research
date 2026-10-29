# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
from copy import deepcopy

import numpy as np
import pytest
import torch

from mimo_bridge import copied_execution
from mimo_bridge.datasource import SingleEnvSource
from mimo_bridge.lem_model import LemLite
from mimo_bridge.tools.self_recovery_experiment import hold


def test_complete_real_model_replay_reproduces_original_first_and_later_decisions(tmp_path):
    """Actual encoders/transformer/planner, mutable schema/self and physics inputs."""
    torch.set_num_threads(1)
    model=LemLite(device='cpu',no_vision=True,next_frame_loss=True,intention=True,intention_every=2,
        imagination=False,persistent_memory='on',self_state='perceive',hierarchical_self='dark',
        body_schema_updates=True,smooth_actions=.35,latent_reg=True,
        overrides={'model':{'context_length':4}})
    for module in copied_execution.model_modules(model).values():module.eval()
    source=SingleEnvSource(scene='crib',vision=False,age=24,caregiver_seconds=0,
        body_function='modulate',contact_profile='mechanical_v2',run_dir=str(tmp_path/'world'))
    try:
        frames=source.reset();dev=dict(model.dev_limits(0),exploration_noise=.01)
        executed=hold(frames[0],dev['strength_limit'])
        for _ in range(4):
            result=model.step(frames,dev,train=False,step_index=model.step_count,executed_action=executed,executed_order_id=1)
            executed=result.motor_for_action.copy();frames=source.step(executed,result.gaze_out)
        saved=copied_execution.capture(model)
        original_mean=saved['runtime']['latent']['mean'].clone()
        samples=[];expected=[]
        for tick in range(6):
            samples.append(dict(frames=deepcopy(frames),dev=deepcopy(dev),executed_action=executed.copy(),executed_order_id=2+tick//2))
            result=model.step(frames,dev,train=False,step_index=model.step_count,
                executed_action=executed,executed_order_id=2+tick//2)
            expected.append(copied_execution.output_record(model,result))
            executed=result.motor_for_action.copy();frames=source.step(executed,result.gaze_out)
        assert model.step_count==saved['public']['step_count']+6
        assert model._hself is not None and model.intention_gen is not None
        receipt=copied_execution.verify_recorded_frames(model,saved,samples,expected)
        assert receipt['complete_model_replay_exact'] is True
        assert receipt['frames']==6
        assert torch.equal(saved['runtime']['latent']['mean'],original_mean)
        assert model.step_count==saved['public']['step_count']
        # A future mutable helper must be explicitly serialized, never ignored.
        model.new_mutable_execution_helper=object()
        with pytest.raises(ValueError,match='new_mutable_execution_helper'):
            copied_execution.capture(model)
    finally:source.close()


def test_repeatability_rejects_nonfinite_scientific_outputs():
    assert not copied_execution.equal_tree(np.array([np.nan]),np.array([np.nan]))
