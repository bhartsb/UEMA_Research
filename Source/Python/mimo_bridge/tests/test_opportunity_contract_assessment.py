# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import copy
import pytest
from mimo_bridge.tools.opportunity_contract_assessment import observed_predictor_contract,assess


def metadata():
    return (dict(motor_hold=5,action_repeat=5,learning_opportunity=dict(audit_every=5)),
        dict(learner_updates_independent=True,decision=dict(horizon_seconds=.25),
             learning_opportunity=dict(completed_comparisons=100,audit_health=dict(available_bodies=8),
                eligibility=[{'vision':dict(reason='available',available=1.)}]*8)))


def test_native_calls_use_current_predictor_but_actor_encoder_and_targets():
    traced=observed_predictor_contract()
    assert traced['current_prediction']['predictor']=='continuing_learner'
    assert traced['current_prediction']['encoder']=='frozen_actor'
    assert traced['target_value']==7 and traced['continuing_learner_target_value']==13


def test_favorable_scores_cannot_certify_untrained_mixed_coordinates():
    config,actor=metadata();result=assess(config,actor,.01,observed_predictor_contract())
    assert not result['complete_learner_retention_certified']
    assert not result['complete_learner_learning_certified']
    assert not result['production_qualification_supported']
    assert result['timing']['audit_outcome_seconds']==.05
    assert result['timing']['held_order_seconds']==.25
    assert result['timing']['order_phases_sampled']==[0]
    assert not result['timing']['forecast_duration_measured_by_audit']


def test_altered_audit_frequency_changes_phases_without_extending_target_duration():
    config,actor=metadata();config['learning_opportunity']['audit_every']=2
    result=assess(config,actor,.01,observed_predictor_contract())
    assert result['timing']['order_phases_sampled']==[0,1,2,3,4]
    assert result['timing']['audit_outcome_seconds']==.05
    assert not result['production_qualification_supported']


@pytest.mark.parametrize('damage',['timing','missing_actor','unknown_interface'])
def test_unknown_or_missing_provenance_cannot_return_a_pass(damage):
    config,actor=metadata();observed=observed_predictor_contract()
    if damage=='timing':config['motor_hold']=0
    elif damage=='missing_actor':actor={}
    else:observed['current_prediction']['predictor']='unverified_new_route'
    with pytest.raises(ValueError):assess(config,actor,.01,observed)
