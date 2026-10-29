# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Qualification contracts: constructed fields test validation, never evidence."""
import pytest
from mimo_bridge.self_intention import SelfIntention


def acquisition_receipt():
    return dict(schema='physical_self_choice_v2',qualification_route='acquisition',
        minimum_resolution=.9,minimum_capacity=.8,maximum_uncertainty=.1,
        candidate_position_fraction=.02,candidate_rotation_fraction=.03,
        body_function='modulate',held_out_worlds=['constructed-test-only'],
        replay_verified=True,retention_pass=True,acquisition_gain_lower=.1,
        action_effect_gain_lower=.2,viability_gain_lower=0.,
        acquisition_metric='controllable_distinctions',acquisition_comparator='matched former actor',
        prediction_error_upper=.02,decision_effect_separation_lower=.1,
        action_prediction_scale='constructed equal-unit physical outcome comparison',
        learning_opportunity={'contract':'constructed test only'})


def test_healthy_acquisition_does_not_require_deficit_or_restoration_reward():
    receipt=acquisition_receipt()
    assert 'choice_gain_lower' not in receipt
    assert SelfIntention('on',receipt).mode=='on'


@pytest.mark.parametrize('field,value',[
    ('viability_gain_lower',-.00001),('viability_gain_lower',float('nan')),
    ('acquisition_gain_lower',0.),('action_effect_gain_lower',0.),
    ('prediction_error_upper',.05),('prediction_error_upper',-.01),
    ('decision_effect_separation_lower',0.),('retention_pass',False),
    ('replay_verified',False),('held_out_worlds',[]),('acquisition_comparator',''),
    ('acquisition_metric','movement count'),('learning_opportunity',None),
    ('action_prediction_scale',''),('qualification_route','invented'),
])
def test_acquisition_never_bypasses_effect_viability_prediction_or_retention(field,value):
    receipt=acquisition_receipt();receipt[field]=value
    with pytest.raises(ValueError):SelfIntention('on',receipt)


def test_restoration_v2_also_requires_error_small_enough_to_resolve_choice():
    receipt=acquisition_receipt();receipt.update(qualification_route='restoration',choice_gain_lower=.1)
    assert SelfIntention('on',receipt).mode=='on'
    receipt['prediction_error_upper']=.05
    with pytest.raises(ValueError,match='cannot resolve'):SelfIntention('on',receipt)


def test_legacy_restoration_receipt_keeps_its_existing_contract():
    receipt=acquisition_receipt();receipt.update(schema='physical_self_choice_v1',choice_gain_lower=.1)
    for field in ('prediction_error_upper','action_prediction_scale','qualification_route'):
        receipt.pop(field)
    assert SelfIntention('on',receipt).mode=='on'
