# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import math
import numpy as np
import pytest
from mimo_bridge.contact_mechanics import G, falling_impact, stopped_impact, pressure_pa, surface_point_velocity, SCENARIOS
from mimo_bridge.contact_mechanics import conservative_spread_linear
from types import SimpleNamespace


def test_independent_energy_momentum_and_units():
    r=falling_impact(12, .2, .04, .01)
    v=math.sqrt(2*G*.2)
    a=v*v/(2*.04)
    assert r['force_n'] == pytest.approx(12*(a+G))
    assert r['contact_impulse_ns'] == pytest.approx(12*v+12*G*r['duration_s'])
    assert r['work_residual_j'] == pytest.approx(0, abs=1e-12)
    assert r['impulse_residual_ns'] == pytest.approx(0, abs=1e-12)
    assert pressure_pa(1,1e-4) == 10000
    assert pressure_pa(21.94,1e-4) == pytest.approx(219400)


def test_scenario_order_and_unknown_peak():
    soft=stopped_impact(.2,2,.02,.001)
    firm=stopped_impact(.2,2,.002,.001)
    sharp=stopped_impact(.2,2,.002,.00001)
    assert firm['force_n'] == pytest.approx(10*soft['force_n'])
    assert sharp['force_n'] == firm['force_n']
    assert sharp['pressure_pa'] == pytest.approx(100*firm['pressure_pa'])
    assert sharp['peak_force_n'] is None
    for s in SCENARIOS:
        r=s.calculate()
        assert r['maximum_work_residual_j'] < 1e-10
        assert r['maximum_impulse_residual_ns'] < 1e-10


def test_rotating_surface_has_nonzero_contact_velocity():
    np.testing.assert_allclose(surface_point_velocity([0,0,0], [0,0,2], [.3,0,0]), [0,.6,0])


@pytest.mark.parametrize('force', [[0,0,10],[0,7,0],[5,0,0],[-3,4,5]])
def test_skin_spreading_conserves_force_in_every_direction(force):
    touch=SimpleNamespace(sensor_scales={1:.01},sensor_outputs={1:np.zeros((3,3))},
        sensor_positions={1:np.array([[0,0,0],[.01,0,0],[.015,0,0]])},
        get_contact_position_relative=lambda **k: np.zeros(3),
        get_sensors_within_distance=lambda *a: (np.array([0,1,2]),np.array([0.,.01,.015])))
    conservative_spread_linear(touch,0,1,np.array(force))
    np.testing.assert_allclose(touch.sensor_outputs[1].sum(0),force,atol=1e-12)


def test_denser_skin_sampling_does_not_shrink_the_physical_contact_footprint():
    areas=[]
    for spacing in (.005,.0025):
        grid=np.arange(-.04,.04001,spacing)
        xx,yy=np.meshgrid(grid,grid)
        points=np.stack([xx.ravel(),yy.ravel(),np.zeros(xx.size)],1)
        dist=np.linalg.norm(points,axis=1)
        touch=SimpleNamespace(sensor_scales={1:spacing},force_spread_scales={1:.015},
            sensor_outputs={1:np.zeros_like(points)},sensor_positions={1:points},
            get_contact_position_relative=lambda **k:np.zeros(3),
            get_sensors_within_distance=lambda *a:(np.arange(len(points)),dist))
        conservative_spread_linear(touch,0,1,np.array([0.,0.,10.]))
        np.testing.assert_allclose(touch.sensor_outputs[1].sum(0),[0,0,10],atol=1e-12)
        areas.append(float((np.linalg.norm(touch.sensor_outputs[1],axis=1)>1e-6).sum()*spacing**2))
    assert abs(areas[0]/areas[1]-1)<.05  # finite-grid area error, not a density multiplier


@pytest.mark.parametrize('bad', [0,-1,float('nan'),float('inf')])
def test_unknown_area_or_stop_cannot_become_zero_load(bad):
    with pytest.raises(ValueError): pressure_pa(1,bad)
    with pytest.raises(ValueError): stopped_impact(1,1,bad,1)
