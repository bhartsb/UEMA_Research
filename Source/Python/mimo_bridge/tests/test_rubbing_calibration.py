# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The ordinary-contact rubbing recalibration (operator decision 2026-09-25 9:09 AM PDT;
docs/gain_of_function_v4/agents/ordinary_pain_calibration_2026-09-25.md).

Today every friction power counts as rubbing once the shear stress exceeds 21.94 kPa, so the creep and
sliding of ordinary support fill the pain memory: rubbing is 87-95 of every 100 parts of every body's load.
'excess_shear_v1' stores only the work of the shear stress above the 219.4 kPa first-discomfort pressure.
Only the world's pain memory changes: the peak and exposure memories, the 10 s rubbing memory and its unit,
the model's load formula and the tissue-impairment memory stay as they are. No world here.
"""
import math
import pickle

import numpy as np
import pytest

from lem.skin_mechanics import (EXCESS_SHEAR_PROFILE, PROFILE, RUBBING_CALIBRATIONS, RUBBING_RULES,
                                WIDE_RANGE_PROFILE, MechanicalExposure, MechanicalProfile, load_parts_np, pain_np)
from mimo_bridge.touch_adapter import PatchSpec
from mimo_bridge.virtual_skin import VirtualTaxelSkin

REF = 219400.0


def fields(pressure, shear, slip_mm_s=0.0):
    x = np.zeros((len(pressure), 10))
    x[:, 0], x[:, 1], x[:, 2], x[:, 3] = pressure, shear, slip_mm_s, 100.0
    return x


def test_the_default_profile_and_every_named_profile_keep_todays_rule():
    assert PROFILE.rubbing_rule == WIDE_RANGE_PROFILE.rubbing_rule == 'power_above_onset'
    assert PROFILE.shear_reference_pa == 21940.0 and RUBBING_RULES == ('power_above_onset', 'excess_shear')
    assert RUBBING_CALIBRATIONS['v4'] is None
    assert RUBBING_CALIBRATIONS['excess_shear_v1'] is RUBBING_CALIBRATIONS['excess_shear_v1_shadow'] is EXCESS_SHEAR_PROFILE
    # the recalibration moves the onset and the rule only
    for name in ('pressure_reference_pa', 'pressure_exposure_start_fraction', 'pressure_exposure_seconds',
                 'recovery_seconds', 'impact_trace_seconds', 'rubbing_reference_j_m2', 'aversion_shape'):
        assert getattr(EXCESS_SHEAR_PROFILE, name) == getattr(PROFILE, name), name
    assert EXCESS_SHEAR_PROFILE.shear_reference_pa == REF and EXCESS_SHEAR_PROFILE.rubbing_rule == 'excess_shear'
    with pytest.raises(ValueError, match='rubbing rule'):
        MechanicalProfile(rubbing_rule='sometimes')


def test_an_old_pickled_profile_reads_as_todays_rule():
    """A checkpoint's MechanicalExposure carries a profile pickled before the field existed."""
    old = pickle.loads(pickle.dumps(PROFILE))
    del old.__dict__['rubbing_rule']
    assert old.rubbing_rule == 'power_above_onset'


@pytest.mark.parametrize('shear_kpa,kept', [(10, 0.0), (30, 0.0), (150, 0.0), (219.4, 0.0), (300, 1 - 219.4 / 300),
                                             (1000, 1 - 219.4 / 1000)])
def test_measured_friction_work_under_each_rule(shear_kpa, kept):
    """One 5 ms tick of 1,000 W/m2 at a given shear stress: today's rule stores all of it above 21.94 kPa,
    the recalibrated one only the share of the shear above 219.4 kPa."""
    dt, q, tau = .005, 1000.0, 10.0
    today, recal = MechanicalExposure(1), MechanicalExposure(1, EXCESS_SHEAR_PROFILE)
    x = fields([shear_kpa * 1e3 / 0.8], [shear_kpa * 1e3], 1.0)
    today.update(x, dt, np.array([q]))
    recal.update(x, dt, np.array([q]))
    step = tau * -math.expm1(-dt / tau)           # the exact leaky integral of one constant tick
    assert today.values[0, 2] == pytest.approx(q * step if shear_kpa * 1e3 > 21940 else 0.0)
    assert recal.values[0, 2] == pytest.approx(q * kept * step, abs=1e-12)
    # the peak and exposure memories are the same under both rules
    np.testing.assert_array_equal(today.values[:, :2], recal.values[:, :2])


def test_the_measured_and_the_unmeasured_path_agree_under_the_recalibrated_rule():
    """Without a measured friction power the memory stores max(shear - onset, 0) x slip; with it, the
    recalibrated rule stores power x (1 - onset/shear), the same number when power = shear x slip."""
    tau, v = 350e3, 0.02
    a, b = MechanicalExposure(1, EXCESS_SHEAR_PROFILE), MechanicalExposure(1, EXCESS_SHEAR_PROFILE)
    x = fields([400e3], [tau], v * 1000)
    a.update(x, .005)
    b.update(x, .005, np.array([tau * v]))
    assert a.values[0, 2] == pytest.approx(b.values[0, 2], rel=1e-12) and a.values[0, 2] > 0


def test_ordinary_sliding_stores_nothing_and_a_painful_drag_the_same_scale_as_today():
    """Ten seconds of the median ordinary rubbing contact of segments 36-40 (106 kPa pressure, 75 kPa shear,
    247 W/m2): today's rule stores a load of 0.71, the recalibrated one nothing. A drag at 600 kPa shear stores
    the work of its shear above 219.4 kPa, on today's 2,194 J/m2 scale."""
    ticks = 2000
    today, recal = MechanicalExposure(1), MechanicalExposure(1, EXCESS_SHEAR_PROFILE)
    for _ in range(ticks):
        x = fields([106e3], [75e3], 247 / 75e3 * 1000)
        today.update(x, .005, np.array([247.0]))
        recal.update(x, .005, np.array([247.0]))
    assert today.values[0, 2] / 2194 == pytest.approx(247 * 10 * -math.expm1(-1) / 2194, rel=1e-9)
    assert today.values[0, 2] / 2194 == pytest.approx(0.71, abs=0.005) and recal.values[0, 2] == 0.0
    drag = MechanicalExposure(1, EXCESS_SHEAR_PROFILE)
    q = 600e3 * 0.05
    for _ in range(100):
        drag.update(fields([700e3], [600e3], 50.0), .005, np.array([q]))
    assert drag.values[0, 2] == pytest.approx(q * (1 - REF / 600e3) * 10 * -math.expm1(-0.5 / 10), rel=1e-9)


def test_the_numpy_load_and_pain_equal_the_models_formula():
    import torch
    from lem.skin_mechanics import mechanical_load
    rng = np.random.default_rng(3)
    pressure = rng.uniform(0, 2e6, 64)
    history = np.stack([rng.uniform(0, 3e6, 64), rng.uniform(0, 5e7, 64), rng.uniform(0, 5e5, 64)], -1)
    gain = rng.uniform(0.5, 1.5, 64)
    for profile in (PROFILE, WIDE_RANGE_PROFILE):
        ours = pain_np(load_parts_np(pressure, history, gain, profile).sum(-1), profile.aversion_shape)
        model = mechanical_load(torch.tensor(pressure), torch.tensor(history), torch.tensor(gain), profile).numpy()
        np.testing.assert_allclose(ours, model, rtol=1e-12, atol=1e-15)


def make_skin(calibrated):
    axis = np.arange(-.02, .0201, .005)
    x, y = np.meshgrid(axis, axis)
    points = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    skin = VirtualTaxelSkin([PatchSpec(1, 'thigh', len(points), .005)], {1: points}, np.array([[1., 1.]]))
    if calibrated:
        skin.set_pain_profile(EXCESS_SHEAR_PROFILE)
    return skin


def record(force, tangential, power):
    from mimo_bridge.contact_footprint import Geometry, footprint
    sphere = Geometry(2, np.array([.03, 0, 0]), np.zeros(3), np.eye(3))
    plane = Geometry(0, np.ones(3), np.zeros(3), np.eye(3))
    r = footprint([sphere, plane], np.zeros(3), np.array([0., 0, 1.]), force)
    r.update(body_id=1, position_body=[0, 0, 0], tangent_basis_body=r['tangent_basis'],
             tangential_force_n=tangential, friction_power_w=power, slip_mm_s=5.0)
    return r


def test_the_second_memory_never_changes_the_skin_that_senses():
    """The same contacts through a skin with and without the recalibrated memory: identical sensing
    (physical fields, local function, today's memory); only the second memory differs."""
    plain, calibrated = make_skin(False), make_skin(True)
    r = record(20.0, 14.0, 0.07)          # an ordinary thigh contact: shear between the two onsets
    for tick in range(300):
        a = plain.advance([r], tick * .005, .005)
        b = calibrated.advance([r], tick * .005, .005)
    np.testing.assert_array_equal(plain.exposure.values, calibrated.exposure.values)
    np.testing.assert_array_equal(plain.function.quality, calibrated.function.quality)
    np.testing.assert_array_equal(a['experienced'], b['experienced'])
    loaded = calibrated.loaded_area > 0
    shear = calibrated.physical[loaded, 1]
    assert (shear > 21940).all() and (shear < REF).all(), "the contact must sit between the two onsets"
    assert (calibrated.exposure.values[loaded, 2] > 0).all()
    assert (calibrated.pain_exposure.values[:, 2] == 0).all()
    np.testing.assert_array_equal(calibrated.pain_exposure.values[:, :2], calibrated.exposure.values[:, :2])


def test_a_new_second_memory_starts_from_the_stored_loads_and_can_be_dropped():
    skin = make_skin(False)
    r = record(20.0, 14.0, 0.07)
    for tick in range(100):
        skin.advance([r], tick * .005, .005)
    stored = skin.exposure.values.copy()
    skin.set_pain_profile(EXCESS_SHEAR_PROFILE)
    np.testing.assert_array_equal(skin.pain_exposure.values, stored)      # nothing vanishes at the switch
    assert skin.pain_exposure.profile is EXCESS_SHEAR_PROFILE
    for tick in range(100, 300):
        skin.advance([], tick * .005, .005)
    # unloaded, both fade at the same 10 s rate from the same start
    np.testing.assert_allclose(skin.pain_exposure.values[:, 2], skin.exposure.values[:, 2], rtol=1e-12)
    skin.set_pain_profile(None)
    assert skin.pain_exposure is None


def test_the_report_judges_sudden_deposits_apart_from_ordinary_frames():
    """Review 2026-09-25, finding 3: a sudden deposit (load up by more than 10 within one planning interval,
    50 frames) is an episode until the load is back within 2; its frames stay out of the ordinary histogram
    and are counted as episode frames over the harm line. Impact frames (acute part at least 2.36, a 30 cm
    fall's peak) are counted on their own. Driven through the env's own report on a one-patch skin."""
    from types import SimpleNamespace
    from mimo_bridge.env import BridgeEnv
    skin = make_skin(True)
    constants = ('RUBBING_REPORT_BINS', 'RUBBING_REPORT_DETREND_FRAMES', 'RUBBING_REPORT_DEPOSIT_JUMP',
                 'RUBBING_REPORT_DEPOSIT_SETTLE', 'RUBBING_REPORT_IMPACT_ACUTE', 'HARM_LINE')
    env = SimpleNamespace(_virtual_skin=skin, pooler=SimpleNamespace(specs=skin.specs),
                          touch_sensitivity=np.array([[1.0, 1.0]]), contact_profile='mechanical_v4',
                          rubbing_calibration='excess_shear_v1_shadow', _rubbing_report=None,
                          _close_rubbing_episode=BridgeEnv._close_rubbing_episode,
                          **{k: getattr(BridgeEnv, k) for k in constants})
    # today's rule: load 1, a deposit to load 21 held 30 frames, back to 2.5 (within 2 of 1), then 5 impact
    # frames (peak 800 kPa = acute 2.65) on top; the recalibrated memory stays at load 1 throughout
    for frame in range(200):
        load = 1.0 if frame < 100 else 21.0 if frame < 130 else 2.5
        peak = 0.8e6 if 150 <= frame < 155 else 0.0
        skin.exposure.values[:, 0], skin.exposure.values[:, 2] = peak, load * 2194.0
        skin.pain_exposure.values[:, 0], skin.pain_exposure.values[:, 2] = peak, 2194.0
        report = BridgeEnv.rubbing_calibration_report(env, np.zeros((1, 10)))
    today = report['since_built']['rules']['v4']
    recal = report['since_built']['rules']['excess_shear_v1']
    assert today['episodes']['count'] == 1 and today['episodes']['frames'] == 30
    assert today['episodes']['largest_load'] == pytest.approx(21.0)
    assert today['over_line'] == dict(all=30, ordinary=0, impact=0, episode=30)   # 21/22 = 0.955
    assert sum(today['histogram']) == 200 and sum(today['ordinary_histogram']) == 170
    assert sum(today['impact_histogram']) == sum(recal['impact_histogram']) == 5
    assert recal['episodes']['count'] == 0 and recal['over_line']['all'] == 0
    assert sum(recal['ordinary_histogram']) == 200
    assert BridgeEnv.RUBBING_REPORT_IMPACT_ACUTE == pytest.approx(2.3614, abs=1e-4)


def test_a_skin_saved_before_the_second_memory_existed_takes_one():
    skin = make_skin(False)
    old = pickle.loads(pickle.dumps(skin))
    del old.__dict__['pain_exposure']
    old.advance([record(20.0, 14.0, 0.07)], 0.0, .005)                  # advances without it
    old.set_pain_profile(EXCESS_SHEAR_PROFILE)
    np.testing.assert_array_equal(old.pain_exposure.values, old.exposure.values)
