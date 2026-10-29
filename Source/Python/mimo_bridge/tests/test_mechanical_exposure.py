# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import copy
import numpy as np
import pytest
import torch
from lem.skin_mechanics import MechanicalExposure, PROFILE
from lem.model.contact_wellness import contact_response


def fields(p, shear=0., slip=0.):
    a=np.zeros((len(p),8));a[:,0]=p;a[:,1]=shear;a[:,2]=slip;a[:,3]=100
    return a


def response(x, history):
    from tokenizer.modalities.touch import _FIELD_NORMS
    return contact_response(torch.tensor(x/_FIELD_NORMS[:8]).unsqueeze(0),
                            profile='mechanical_v2',history=torch.tensor(history).unsqueeze(0))


def test_support_and_localized_load_have_different_aversion():
    x=fields([14710., 438800.]);e=MechanicalExposure(2);h=e.update(x,.01)
    r=response(x,h)
    assert r['local_pain'][0,0] == 0
    assert r['local_pain'][0,1] > .63
    assert response(x[[0]],h[[0]])['comfort'] > r['comfort']


def test_regional_sensitivity_changes_sustained_and_rubbing_onsets():
    x=fields([40000.,40000.],shear=20000.,slip=100.)
    e=MechanicalExposure(2,pain_sensitivity=[1.,2.]);h=e.update(x,1.)
    assert h[0,1]==0 and h[0,2]==0
    assert h[1,1]>0 and h[1,2]>0


def test_unloading_relieves_pressure_and_rubbing_recovers():
    x=fields([300000.],shear=50000.,slip=100.)
    e=MechanicalExposure(1)
    for _ in range(100): e.update(x,.01)
    before=response(x,e.values)['pain'].item()
    initial_work=e.values[0,2]
    empty=fields([0.])
    for _ in range(1000): e.update(empty,.01)
    assert response(empty,e.values)['pain'].item() < before
    assert e.values[0,2] < initial_work
    assert before > response(fields([300000.]),MechanicalExposure(1).update(fields([300000.]),.01))['pain'].item()


def test_exposure_uses_seconds_and_is_invariant_to_subdivision():
    x=fields([200000.],shear=50000.,slip=50.)
    a,b=MechanicalExposure(1),MechanicalExposure(1)
    for _ in range(200): a.update(x,.01)
    for _ in range(20): b.update(x,.1)
    np.testing.assert_allclose(a.values,b.values,rtol=1e-12,atol=1e-9)
    divided=np.repeat(x,10,axis=0);divided[:,3]/=10
    c=MechanicalExposure(10)
    for _ in range(20): c.update(divided,.1)
    torch.testing.assert_close(response(x,b.values)['pain'],response(divided,c.values)['pain'])


def test_short_impact_survives_until_next_observation_without_cross_body_leak():
    e=MechanicalExposure(2)
    e.update(fields([1000000.,0.]),.01)
    for _ in range(4): e.update(fields([0.,0.]),.01)
    r=response(fields([0.,0.]),e.values)
    assert r['local_pain'][0,0] > .8
    assert r['local_pain'][0,1] == 0
    assert MechanicalExposure(2).values.sum() == 0


def test_snapshot_and_invalid_input():
    e=MechanicalExposure(1);e.update(fields([500000.]),.01)
    saved=copy.deepcopy(e);e.update(fields([0.]),1.)
    assert saved.values[0,0] > e.values[0,0]
    with pytest.raises(ValueError): e.update(fields([float('nan')]),.01)
    with pytest.raises(ValueError): e.update(fields([1.]),0.)
    with pytest.raises(ValueError): contact_response(torch.zeros(1,1,8),profile='mechanical_v2')
