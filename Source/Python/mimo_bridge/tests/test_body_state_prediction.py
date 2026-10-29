# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import numpy as np
import pytest
import torch

from mimo_bridge.tools.action_effect_learning import collection_steps
from mimo_bridge.tools.probe_clock import ProbeClock
from mimo_bridge.tools.body_state_prediction import extract, fit, predict, score, gate, measurements


def test_observation_horizon_does_not_slow_policy_or_extend_held_order():
    clock = ProbeClock(.01,5,5)
    for horizon, observation in ((.05,5),(.25,25)):
        branch, policy = collection_steps(clock,horizon)
        assert branch == observation
        assert policy * clock.physics_step_seconds == .05
        assert policy * clock.motor_hold * clock.physics_step_seconds == .25


def planted(n, seed):
    rng = torch.Generator().manual_seed(seed)
    s = torch.randn(n,2,generator=rng,dtype=torch.float64)
    c = torch.randn(n,4,2,generator=rng,dtype=torch.float64)
    # A command has opposite consequences on either side of the posture axis.
    y = (c[:,:,:1] * torch.tanh(2*s[:,:1,None]) + .1*s[:,:1,None])
    return dict(s=s,c=c,y=y,v=torch.zeros(n,1),floor=np.zeros(n),worlds=np.arange(n)%3,horizon=.05)


def test_kernel_learns_state_dependent_effect_that_additive_state_cannot():
    train, test = planted(100,1), planted(30,2)
    nonlinear = fit(train,'state_action_kernel',.001,1.)
    linear = fit(train,'linear',.001,1.)
    ns, ls = score(predict(nonlinear,test),test['y']), score(predict(linear,test),test['y'])
    assert ns < .25
    assert ls > .7
    null = fit(train,'state_action_kernel',.001,1.,shuffle_seed=4)
    assert score(predict(null,test),test['y']) > ns


def test_extract_uses_raw_bone_layout_and_keeps_future_only_in_targets():
    bone = torch.zeros(24,11)
    bone[:,3] = 1
    bone[:,4] = torch.arange(24.)
    bone[:,7] = 3.  # cm/s, not a normalized token velocity
    cmd = torch.zeros(4,24,9)
    cmd[...,:3] = bone[:,4:7]
    cmd[...,6] = 1
    paths = {}
    for a in range(5):
        b = bone.clone()
        b[:,5] += a if a < 4 else 0
        paths[a] = [dict(bones=b.clone()) for _ in range(5)]
    row = dict(world=0,commands=cmd,physical_start=dict(bones=bone),physical_paths=paths)
    first = extract([row],.05)
    assert torch.all(first['v'].reshape(24,3)[:,0] == 3)
    assert torch.all(first['y'][0,2].reshape(24,3)[:,1] == 2)
    assert first['floor'][0] == 0
    paths[2][0]['bones'] += 50
    second = extract([row],.05)
    assert torch.equal(first['s'],second['s']) and torch.equal(first['c'],second['c'])
    assert not torch.equal(first['y'],second['y'])


def test_empty_nonfinite_and_unresolved_evidence_cannot_pass():
    with pytest.raises(ValueError,match='Empty'):
        gate([],{}, {})
    with pytest.raises(ValueError,match='No physical'):
        score(torch.ones(2,4,3),torch.ones(2,4,3))
    with pytest.raises(ValueError,match='Non-finite'):
        score(torch.full((2,4,3),float('nan')),torch.randn(2,4,3))
    data = planted(20,4)
    data['y'] = data['y'].repeat(1,1,72)
    data['v'] = data['v'].repeat(1,72)
    model = fit(data,'linear',1.,1.)
    data['floor'][:] = 1000
    with pytest.raises(ValueError,match='replay'):
        measurements(model,data)


def test_scoring_units_are_per_point_not_sum_across_body():
    data = planted(20,4)
    data['y'] = data['y'].repeat(1,1,72)
    data['v'] = data['v'].repeat(1,72)
    model = fit(data,'linear',1.,1.)
    measured = measurements(model,data)
    # All body points share the same planted movement, so wrists and body agree.
    assert measured['effect_rms_cm_per_body_point'] == pytest.approx(measured['effect_rms_cm_per_wrist'])


def wrists_only(layout, states):
    """`states` states x 4 candidate orders on the `layout` body. Only the two wrists move, 10 cm
    more per candidate (wrist_L along x, wrist_R along y), so the 6 candidate pairs differ by 10, 20,
    30, 10, 20 and 10 cm: each wrist's root-mean-square action effect is sqrt(2000 / 6) = 18.26 cm."""
    from mimo_bridge.body_layout import names_for
    names = names_for(layout)
    y = torch.zeros(states, 4, len(names) * 3, dtype=torch.float64)
    for a in range(4):
        y[:, a, names.index('wrist_L') * 3] = 10. * a
        y[:, a, names.index('wrist_R') * 3 + 1] = 10. * a
    return dict(s=torch.zeros(states, 2, dtype=torch.float64), c=torch.zeros(states, 4, 2, dtype=torch.float64),
                y=y, v=torch.zeros(states, y.shape[-1], dtype=torch.float64), floor=np.zeros(states),
                worlds=np.zeros(states, dtype=int), horizon=.05)


@pytest.mark.parametrize('layout,states', [('legacy', 3), ('legacy', 4), ('trunk_v1', 3), ('trunk_v1', 4)])
def test_wrist_readings_find_the_wrists_by_name_in_either_body_layout(monkeypatch, layout, states):
    # trunk_v1 records (26 body rows, since commit 826f4e14a on 2026-09-11) were cut into 24-row
    # pieces until 2026-09-23: with 4 states other body points were read as the wrists, with 3 the
    # reshape stopped. Legacy records (24 rows) read correctly before and after.
    from mimo_bridge.body_layout import names_for
    from mimo_bridge.tools import body_state_prediction as BSP
    monkeypatch.setattr(BSP, 'predict', lambda model, data: torch.zeros_like(data['y']))
    measured = BSP.measurements(None, wrists_only(layout, states))
    wrist_cm = float(np.sqrt(2000 / 6))
    assert measured['effect_rms_cm_per_wrist'] == pytest.approx(wrist_cm)
    # predicting no movement misses the whole effect, so the error equals the effect
    assert measured['error_rms_cm_per_wrist'] == pytest.approx(wrist_cm)
    # the per-body-point reading averages both wrists over every body row
    rows = len(names_for(layout))
    assert measured['effect_rms_cm_per_body_point'] == pytest.approx(float(np.sqrt(2 * 2000 / 6 / rows)))


@pytest.mark.parametrize('numbers', [75, 73])
def test_a_record_that_is_no_known_body_stops_instead_of_naming_other_points_wrists(monkeypatch, numbers):
    # 75 numbers = 25 body points, neither the 24-row legacy nor the 26-row trunk_v1 body;
    # 73 numbers are not whole body points. 12 states make the old fixed 24-row reshape succeed for both.
    from mimo_bridge.tools import body_state_prediction as BSP
    monkeypatch.setattr(BSP, 'predict', lambda model, data: torch.zeros_like(data['y']))
    data = wrists_only('trunk_v1', 12)
    data['y'], data['v'] = data['y'][..., :numbers], data['v'][..., :numbers]
    with pytest.raises(ValueError, match='body'):
        BSP.measurements(None, data)


def test_historical_readout_cannot_report_empty_evidence_as_zero_error():
    from mimo_bridge.tools.body_predictor_fit import _effect_ratio
    with pytest.raises(ValueError,match='Missing'):
        _effect_ratio([],torch.zeros(289,72),200.)


def test_historical_feature_map_cancels_shared_state_in_action_comparison():
    from mimo_bridge.tools.body_predictor_fit import _features
    rng = torch.Generator().manual_seed(3)
    row = dict(physical_start=dict(achieved=torch.randn(24,10,generator=rng),
                                  proprioception=torch.randn(24,12,generator=rng)),
               commands=torch.randn(4,24,9,generator=rng))
    before = _features(row,0,200.) - _features(row,1,200.)
    row['physical_start']['achieved'] *= 3
    row['physical_start']['proprioception'] *= -2
    after = _features(row,0,200.) - _features(row,1,200.)
    assert torch.allclose(before,after,atol=1e-4)


def test_historical_solver_handles_redundant_mixed_scale_features():
    from mimo_bridge.tools.body_predictor_fit import _fit_ridge, _predict
    x = torch.randn(80,4,generator=torch.Generator().manual_seed(20))
    mixed = torch.cat([x*100, x, x*.001],-1)
    y = x[:,:2]*.25
    w = _fit_ridge(mixed,y,.0001)
    assert w.dtype == torch.float64
    assert torch.isfinite(w).all()
    assert (_predict(w,mixed)-y).abs().max() < 1e-5
