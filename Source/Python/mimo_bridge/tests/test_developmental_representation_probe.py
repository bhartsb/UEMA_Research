# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import numpy as np
import pytest

from mimo_bridge.tools.developmental_representation_probe import score_event, aggregate, sign_p


def fixture():
    target=np.array([[-1.,0.],[0.,1.],[1.,-1.]])[:,None,:]
    current=np.zeros_like(target)
    return target,current


def test_planted_action_effect_beats_swapped_persistence_and_constant():
    target,current=fixture()
    result=score_event(target,target,np.roll(target,1,0),current,target[0])
    assert result['resolved']
    assert result['action_effect_error_removed']==1.
    assert result['action_identification_accuracy']==1.
    assert result['correct_first_mse']==0.
    assert result['swapped_first_mse']>0.


def test_constant_predictor_cannot_claim_recognition_of_a_real_effect():
    target,current=fixture()
    result=score_event(target,current,current,current,target[0])
    assert result['action_effect_error_removed']==0.
    assert result['action_identification_accuracy']==pytest.approx(1/3)


def test_no_effect_and_effect_smaller_than_repeat_noise_are_unresolved():
    target,current=fixture()
    result=score_event(current,current,current,current,current[0])
    assert not result['resolved']
    assert result['action_identification_accuracy'] is None
    noisy=score_event(target,target,np.roll(target,1,0),current,target[0]+1.)
    assert not noisy['resolved']


@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_nonfinite_prediction_is_an_instrument_error(bad):
    target,current=fixture();prediction=target.copy();prediction.flat[0]=bad
    with pytest.raises(ValueError,match='nonfinite'):
        score_event(target,prediction,target,current,target[0])


def test_later_observed_history_does_not_pass_the_same_state_effect_gate():
    target,current=fixture()
    target=np.repeat(target,3,axis=1)
    prediction=target.copy();prediction[:,0]=0
    result=score_event(target,prediction,np.roll(target,1,0),np.zeros_like(target),target[0])
    assert result['action_effect_error_removed']==0.
    assert result['online_correct_mse']<result['persistence_first_mse']


def test_world_sign_test_keeps_ties_and_does_not_count_frames_as_worlds():
    assert sign_p([1,0,0,0])==15/16
    worlds=[{'modalities':{'tactile':dict(resolved=True,action_effect_error_removed=.5,
        swapped_advantage=.1,persistence_advantage=.1)}} for _ in range(8)]
    assert aggregate(worlds[:1],['tactile'])['modalities']['tactile']['verdict']=='COMPARISON_NOT_PASSED'
    result=aggregate(worlds,['tactile'])
    assert result['modalities']['tactile']['verdict']=='PREDICTION_COMPARISON_PASS'
    assert result['capability_pass'] is False
    assert aggregate(worlds,['tactile'],smoke=True)['modalities']['tactile']['verdict']=='COMPARISON_NOT_PASSED'


def test_missing_real_vision_is_not_replaced_by_a_body_metric():
    worlds=[{'modalities':{}} for _ in range(8)]
    assert aggregate(worlds,['vision'])['modalities']['vision']['verdict']=='UNMEASURED'
