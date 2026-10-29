# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import numpy as np
import pytest
import torch

from mimo_bridge.tools.body_sequence_fit import final_positions, absolute_scores, prediction_gate


def test_last_frame_uses_every_body_point_not_last_flat_segment():
    path = torch.arange(24*5*3,dtype=torch.float64).reshape(1,1,24,5,3).repeat(1,4,1,1,1)
    actual = final_positions(path.flatten(2))
    expected = torch.cat([path[:,:,point,4,:] for point in range(24)],-1)
    assert torch.equal(actual,expected)
    assert not torch.equal(actual,path.flatten(2)[...,-72:])


def test_final_wrist_error_cannot_be_hidden_by_stationary_feet():
    true = torch.zeros(2,4,24,5,3,dtype=torch.float64)
    pred = true.clone()
    pred[:,:,0,4,0] = 12
    data = dict(y=true.flatten(2),v=torch.zeros(2,72),horizon=.25)
    assert absolute_scores(pred.flatten(2),data)['model'] == pytest.approx(2.)


def test_constant_velocity_uses_matching_bones_and_centimetres():
    velocity = torch.arange(72,dtype=torch.float64).reshape(1,24,3)
    path = velocity[:,:,None,:]*torch.arange(1,6,dtype=torch.float64)[None,None,:,None]*.05
    y = path[:,None].repeat(1,4,1,1,1).flatten(2)
    data = dict(y=y,v=velocity.flatten(1),horizon=.25)
    scores = absolute_scores(y,data)
    assert scores['model'] == 0
    assert scores['constant_velocity'] == pytest.approx(0.,abs=1e-26)
    assert scores['no_movement'] > 0


def valid_evidence():
    per={0:dict(sequence=.7,linear_comparator=.9,effect_mse_cm2=1.,replay_floor_max=0.)}
    absolute={0:dict(model=.5,no_movement=1.,constant_velocity=2.)}
    return per,absolute,{0:[1.]*8}


def test_gate_rejects_empty_nan_missing_and_unresolved_replay():
    assert prediction_gate(*valid_evidence())
    with pytest.raises(ValueError,match='Missing'):
        prediction_gate({}, {}, {})
    per,absolute,null=valid_evidence()
    per[0]['replay_floor_max']=.11
    with pytest.raises(ValueError,match='unresolved'):
        prediction_gate(per,absolute,null)
    per,absolute,null=valid_evidence()
    null[0][0]=float('nan')
    with pytest.raises(ValueError,match='Invalid'):
        prediction_gate(per,absolute,null)
    per,absolute,null=valid_evidence()
    null[0].pop()
    with pytest.raises(ValueError,match='Invalid'):
        prediction_gate(per,absolute,null)


def test_gate_cannot_pass_when_all_fitted_comparators_are_worse_than_no_effect():
    per,absolute,null=valid_evidence()
    per[0].update(sequence=1.1,linear_comparator=1.2)
    assert not prediction_gate(per,absolute,{0:[1.3]*8})


def test_final_output_does_not_gain_temporal_integration_from_extra_targets():
    # Kernel multi-output regression fits each output column independently.
    # Extra path frames can affect hyperparameter selection, not a fixed fit's
    # prediction at the last frame. This constrains claims about this design.
    from mimo_bridge.tools.body_state_prediction import fit,predict
    rng=torch.Generator().manual_seed(74)
    s=torch.randn(20,4,generator=rng,dtype=torch.float64)
    c=torch.randn(20,4,3,generator=rng,dtype=torch.float64)
    y=torch.randn(20,4,360,generator=rng,dtype=torch.float64)
    data=dict(s=s,c=c,y=y)
    path_model=fit(data,'state_action_kernel',.1,1.)
    final_model=fit(dict(data,y=final_positions(y)),'state_action_kernel',.1,1.)
    assert torch.allclose(final_positions(predict(path_model,data)),predict(final_model,data),atol=1e-12)


def test_confirmation_checks_command_history_timestamps_and_all_worlds():
    from mimo_bridge.tools.body_sequence_confirmation import validate_confirmation
    declaration=dict(evaluation_only=True,independent_streams=True,seed_offset=300,
                     train_worlds=[],worlds=6,events_per_world=16)
    rows=[]
    for world in range(6):
        for event in range(16):
            now=4.5+event*.5
            history=[dict(command_start_time=now-.25+i*.05,outcome_time=now-.2+i*.05) for i in range(5)]
            rows.append(dict(world=world,event=event,sim_time=now,history=history,
                             physical_paths={a:[{} for _ in range(5)] for a in range(5)}))
    validate_confirmation(rows,declaration)
    rows[0]['history'][-1]['outcome_time'] += .05
    with pytest.raises(ValueError,match='History'):
        validate_confirmation(rows,declaration)
