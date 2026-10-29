# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The footprint guard: a flat-face sliver is scored as the rounded edge that carries it, and no
footprint's mean pressure exceeds the ceiling; off, the v1 footprints are unchanged.

Defect (docs/gain_of_function_v4/agents/body2_pain_2026-09-25.md): two aligned flat box faces are
scored by the exact overlap of their rectangles, with no smallest area. H84 body 2's left palm side
face met the heavy box's top edge at its corner: the overlap read 0.0165 mm2 at the sampled tick of
model step 399,600, an ordinary 4.8 N read 3.17e11 Pa, and a stored rubbing load of 12,941 (harm line
11.5) held the body over the harm line for 71 s. Numpy only: no world, no torch.
"""
import math

import numpy as np
import pytest

from mimo_bridge.contact_footprint import (CONTRACT, FOOTPRINT_GUARDS, RESCORED_METHOD, FootprintGuard,
                                           Geometry, VirtualSkinProfile, aggregate_contact_footprints,
                                           bound_mean_pressure, footprint, holds_disk, overlap_strip,
                                           rounded_edge_contact, rounded_edge_rescore)
from mimo_bridge.touch_adapter import ContactInfo, PatchSpec
from mimo_bridge.virtual_skin import VirtualTaxelSkin

GUARD = FOOTPRINT_GUARDS['rounded_edge_v1']
E = 1e6                       # VirtualSkinProfile().effective_modulus_pa
REFERENCE_PA = 219400.0       # skin_mechanics: first-discomfort pressure
RUBBING_REFERENCE = 2194.0    # J/m2 for load 1
HARM_LOAD = 11.5              # the goal ranking's harm line 0.92 as a load: 0.92 / (1 - 0.92)
FALL_90CM_MEDIAN_PA = 2533091.6   # scale_anchors.json fall_response.mechanical_v4, 90 cm
FALL_LARGEST_PA = 4787537.5       # the largest of its 72 trials

# H84 body 2 at model step 399,600: left palm box (geom 41, geom:left_hand1) and box_heavy (geom 108)
# on the recorded pose, rebuilt from the observer's world recording by forward kinematics over the
# saved world model (docs/gain_of_function_v4/skin_footprint_measurements.py palm), with the stats
# record's sampled contact point, normal and force (replay_training_stats.jsonl of
# H84_fresh_drivers_20260918_seg40_resume_20260924_234404, type mimo, step 399600, body 2).
PALM = dict(kind=6, size=[0.01802821749743656, 0.006189881449883724, 0.018531956993467266],
            position=[0.6449505995277404, 0.29315558986417967, 0.07370357827843571],
            rotation=[[-0.8913800640933138, -0.44221774918729234, 0.09942355676963437],
                      [0.08605838251563505, 0.05024209251931945, 0.9950224554943837],
                      [-0.44501183819735907, 0.8954994106324166, -0.006728255435289917]])
HEAVY_BOX = dict(kind=6, size=[0.03, 0.03, 0.03],
                 position=[0.6484369999999999, 0.343425, 0.029984999999999998],
                 rotation=[[-0.9988807277823728, -0.04729995630035556, -7.615078649763784e-05],
                           [0.04729996549427631, -0.9988807225738123, -0.00012383319155085998],
                           [-7.020824809243535e-05, -0.0001272965180736372, 0.9999999894331991]])
POINT = [0.634757584883691, 0.31405321133496605, 0.059930398685326076]
NORMAL = [0.04728239283899627, 0.9984751689784943, -0.02849056441677087]
FORCE = 4.784016310180743
FRICTION_POWER_W = 0.010628348627753782
TANGENTIAL_N = 3.634730749061172
SAMPLED_TRIANGLE = [[0.0002620063577539883, -5.570107270614147e-05],
                    [2.8606405390841207e-05, 5.847666695337419e-05],
                    [-2.7254337646972936e-05, -5.571279592157546e-05]]
SAMPLED_AREA = 1.6514934285572623e-08


def shape(d):
    return Geometry(d['kind'], np.array(d['size']), np.array(d['position']), np.array(d['rotation']))


def palm(force=FORCE, guard=None):
    return footprint([shape(PALM), shape(HEAVY_BOX)], np.array(POINT), np.array(NORMAL), force, guard=guard)


def geom(kind, size, pos=(0, 0, 0)):
    return Geometry(kind, np.asarray(size, dtype=float), np.asarray(pos, dtype=float), np.eye(3))


def corner_regime_area(force, radius=1e-3, modulus=E):
    b = (2 * force * radius / (math.pi * modulus)) ** (1 / 3)
    return 4 * b * b


# ------------------------------------------------------------------ the 399,600 contact, replayed
def test_the_palm_on_box_contact_reproduces_the_sliver_with_the_guard_off():
    """The recorded pose (one physics tick after the scored one) gives an overlap 1,400 times thinner
    than the sampled 1.65e-8 m2: the record's 'at least 100 times thinner at unsampled ticks'."""
    r = palm()
    assert r['method'] == 'aligned_flat_face_overlap' and 'footprint_guard' not in r
    assert r['area_m2'] == pytest.approx(1.1651988742943993e-11, rel=1e-6)
    assert r['mean_pressure_pa'] > 1e11            # 410 GPa, 1.9 million times the reference


def test_the_palm_on_box_contact_is_scored_as_the_rounded_edge_with_the_guard_on():
    r = palm(guard=GUARD)
    assert r['method'] == RESCORED_METHOD and r['contract'] == CONTRACT
    # a corner (the overlap is 3 um across): the edge is never shorter than its contact is wide
    assert r['area_m2'] == pytest.approx(corner_regime_area(FORCE), rel=1e-9)
    assert r['mean_pressure_pa'] == pytest.approx(569225.2, rel=1e-6)
    assert r['mean_pressure_pa'] < GUARD.pressure_ceiling_pa
    assert r['mean_pressure_pa'] < FALL_90CM_MEDIAN_PA
    note = r['footprint_guard']
    assert note['setting'] == 'rounded_edge_v1' and note['flat_overlap_area_m2'] == pytest.approx(1.1651988742943993e-11)
    assert note['flat_overlap_width_m'] < 1e-5 and 'area_before_ceiling_m2' not in note
    assert note['flat_parts_touch'] is False
    # the moduli bounds of the coating still bracket the nominal pressure
    low, high = r['pressure_bounds_pa']
    assert low < r['mean_pressure_pa'] < high <= GUARD.pressure_ceiling_pa


@pytest.mark.parametrize('force', [2.5, 4.784, 10.0, 20.0, 53.0, 68.0])
def test_every_recorded_hand_to_box_force_reads_under_the_largest_fall(force):
    """The world recording's hand-to-box forces at 399,585-399,602 were 2.5-68 N."""
    r = palm(force, GUARD)
    assert r['method'] == RESCORED_METHOD
    assert r['mean_pressure_pa'] < FALL_90CM_MEDIAN_PA < FALL_LARGEST_PA < GUARD.pressure_ceiling_pa


def test_the_sampled_tick_triangle_is_rescored_the_same_way():
    """The stats record's own overlap polygon (0.29 mm x 0.11 mm triangle) through the guard's decision."""
    res = rounded_edge_rescore(SAMPLED_TRIANGLE, SAMPLED_AREA, FORCE, GUARD, E)
    assert res is not None
    area, length, halfwidth, (width, span, _) = res
    assert width == pytest.approx(1.1418719884578251e-4) and span == pytest.approx(2.892606956385216e-4)
    assert area == pytest.approx(corner_regime_area(FORCE), rel=1e-9)
    assert length == pytest.approx(2 * halfwidth)


def test_the_skin_stores_no_harm_line_load_from_the_palm_contact_with_the_guard_on():
    """The palm contact held for the 0.5 s the jump lay in (100 physics ticks of 5 ms), through the
    virtual skin, on a palm-like patch (7 mm taxel spacing, 49 mm2 per taxel). Off: the load is
    thousands of times the harm line (11.5). On: 1.89, pain 0.65, and the closed form of the exact
    leaky integrals: the peak's 1.59, 0.5 s of exposure 0.02 and of rubbing 0.28."""
    axis = np.arange(-.021, .0211, .007)
    x, y = np.meshgrid(axis, axis)
    points = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    loads = {}
    for name, guard in (('off', None), ('on', GUARD)):
        skin = VirtualTaxelSkin([PatchSpec(27, 'left_hand', len(points), .007)], {27: points}, np.array([[1., 1.]]))
        r = palm(guard=guard)
        # the patch lies in its own x-y plane; the contact sits on its centre taxel
        r.update(body_id=27, position_body=[0., 0., 0.], tangent_basis_body=[[1., 0.], [0., 1.], [0., 0.]],
                 friction_power_w=FRICTION_POWER_W, tangential_force_n=TANGENTIAL_N, slip_mm_s=2.99)
        for tick in range(100):
            skin.advance([r], tick * .005, .005)
        peak, exposure, rubbing = skin.exposure.values.max(axis=0)
        load = max(peak / REFERENCE_PA - 1, 0) + exposure / (REFERENCE_PA * 60) + rubbing / RUBBING_REFERENCE
        loads[name] = load
        if guard is not None:
            p = r['mean_pressure_pa']
            assert peak == pytest.approx(p, rel=1e-9)   # the centre taxel carries it all
            expected = (p / REFERENCE_PA - 1
                        + (p - .25 * REFERENCE_PA) * 60 * -math.expm1(-.5 / 60) / (REFERENCE_PA * 60)
                        + FRICTION_POWER_W / r['area_m2'] * 10 * -math.expm1(-.5 / 10) / RUBBING_REFERENCE)
            assert load == pytest.approx(expected, rel=1e-6)
    assert loads['off'] > 1000 * HARM_LOAD
    assert loads['on'] == pytest.approx(1.89, abs=0.01) and loads['on'] < HARM_LOAD / 5


# ------------------------------------------------------------------ the rounded-edge formula
def test_the_rounded_edge_line_regime_is_the_existing_hertz_line_formula():
    """The record's scale estimate: a 3.6 cm palm edge rounded to 1 mm under 4.8 N reads 161 kPa."""
    area, length, halfwidth = rounded_edge_contact(FORCE, .036, 1e-3, E)
    assert length == .036
    assert halfwidth == pytest.approx(math.sqrt(4 * FORCE * 1e-3 / (math.pi * .036 * E)))
    assert area == pytest.approx(2.962e-5, rel=1e-3) and FORCE / area == pytest.approx(161.5e3, rel=1e-3)
    capsule = footprint([geom(3, [1e-3, .018, 0]), geom(0, [1, 1, 1])], np.array([1e-3, 0, 0]),
                        np.array([1., 0, 0]), FORCE)
    assert capsule['method'] == 'finite_hertz_line_ignores_end_effects'
    assert capsule['area_m2'] == pytest.approx(area, rel=1e-12)


def test_a_short_edge_becomes_a_rounded_corner_continuously():
    b_corner = (2 * FORCE * 1e-3 / (math.pi * E)) ** (1 / 3)
    below = rounded_edge_contact(FORCE, 2 * b_corner * (1 - 1e-9), 1e-3, E)
    above = rounded_edge_contact(FORCE, 2 * b_corner * (1 + 1e-9), 1e-3, E)
    assert below[0] == pytest.approx(above[0], rel=1e-6)
    tiny = rounded_edge_contact(FORCE, 1e-9, 1e-3, E)
    assert tiny[0] == pytest.approx(4 * b_corner ** 2) and tiny[1] == pytest.approx(2 * b_corner)
    hertz_sphere = math.pi * (3 * FORCE * 1e-3 / (4 * E)) ** (2 / 3)
    assert tiny[0] / hertz_sphere == pytest.approx(1.1414, abs=1e-3)


def test_holds_disk_is_the_inset_test_in_either_winding():
    def rect(w, l):
        return [[0, 0], [l, 0], [l, w], [0, w]]
    assert holds_disk(rect(2.2e-3, .03), 1e-3)
    assert not holds_disk(rect(1.8e-3, .03), 1e-3)
    assert not holds_disk(rect(1.8e-3, .03)[::-1], 1e-3) and holds_disk(rect(2.2e-3, .03)[::-1], 1e-3)
    assert not holds_disk(SAMPLED_TRIANGLE, 1e-3)
    equilateral = [[0, 0], [.01, 0], [.005, .005 * math.sqrt(3)]]    # inradius 2.887 mm
    assert holds_disk(equilateral, 2.8e-3) and not holds_disk(equilateral, 2.95e-3)
    assert overlap_strip(rect(1.8e-3, .03))[:2] == pytest.approx((1.8e-3, .03))


def test_clipping_noise_cannot_change_the_guards_decision():
    """Clipping can leave a repeated vertex or a backwards spike of 1e-17 m. An inset or a strip measured
    along that edge's random direction read a 30 mm square as holding no 1 mm disk (measured before the
    hull); the guard's decision and note must be the clean polygon's."""
    square = [[0, 0], [.03, 0], [.03, .03], [0, .03]]
    for noisy in (square[:3] + [[.03 + 1e-17, .03 - 1e-17]] + square[3:],
                  square[:3] + [[.03 - 1e-17, .03 + 1e-17]] + square[3:],
                  square[:2] + [square[1]] + square[2:]):
        assert holds_disk(noisy, 1e-3)
        assert overlap_strip(noisy)[:2] == pytest.approx((.03, .03))
    clean = np.array([[0, 0], [3e-3, 0], [3e-3, 3e-3], [0, 3e-3]])
    noisy = np.array([[0, 0], [3e-3, 0], [3e-3, 3e-3], [3e-3 + 1e-17, 3e-3 - 1e-17], [0, 3e-3]])
    for force in (0.5, 20.0):
        a, b = rounded_edge_rescore(clean, 9e-6, force, GUARD, E), rounded_edge_rescore(noisy, 9e-6, force, GUARD, E)
        assert (a is None) == (b is None)
        if a is not None:
            assert a[:3] == pytest.approx(b[:3], rel=1e-12) and a[3][:2] == pytest.approx(b[3][:2], rel=1e-12)
    assert rounded_edge_rescore(clean, 9e-6, 0.5, GUARD, E) is None       # 0.5 N: the overlap is larger
    assert rounded_edge_rescore(clean, 9e-6, 20.0, GUARD, E) is not None  # 20 N: the corner carries more


def box_pair(offset, force, halfwidth=.015):
    """Two 30 x 30 x 20 mm boxes face to face (normal z), the upper shifted along x by `offset`: their
    faces overlap in a strip 2 * halfwidth - offset wide and 30 mm long."""
    lower = geom(6, [halfwidth, .015, .01], (0, 0, -.01))
    upper = geom(6, [halfwidth, .015, .01], (offset, 0, .01))
    point = np.array([halfwidth - (2 * halfwidth - offset) / 2, 0, 0])
    return footprint([lower, upper], point, np.array([0, 0, 1.]), force, guard=GUARD), \
        footprint([lower, upper], point, np.array([0, 0, 1.]), force)


def test_a_thin_strip_is_scored_as_the_rounded_edge_along_it():
    on, off = box_pair(.03 - .0001, 5.)           # 0.1 mm x 30 mm strip
    assert off['method'] == 'aligned_flat_face_overlap' and off['area_m2'] == pytest.approx(1e-4 * .03)
    assert on['method'] == RESCORED_METHOD
    assert on['area_m2'] == pytest.approx(rounded_edge_contact(5., .03, 1e-3, E)[0], rel=1e-9)
    assert on['tangent_axes_m'][0] == pytest.approx(.015)          # half the edge, along the strip
    along = np.asarray(on['tangent_basis'])[:, 0]
    assert abs(along[1]) == pytest.approx(1.0)                      # the strip runs along y


def test_an_overlap_larger_than_its_rounded_edge_contact_is_left_exactly_as_it_was():
    on, off = box_pair(.03 - .005, 5.)            # 5 mm x 30 mm under 5 N
    assert off['area_m2'] > rounded_edge_contact(5., .03, 1e-3, E)[0]
    assert on == off and 'footprint_guard' not in on
    on, off = box_pair(.03 - .0015, .2)           # 1.5 mm strip, 0.2 N: the edge contact is narrower
    assert off['area_m2'] > rounded_edge_contact(.2, .03, 1e-3, E)[0]
    assert on == off, 'the rounded edge is a floor on the area, never a shrink'


def corner_pair(width, force, guard=GUARD):
    """Two 30 x 30 x 20 mm boxes face to face, corner over corner: a width x width square overlap."""
    d = .03 - width
    lower = geom(6, [.015, .015, .01], (0, 0, -.01))
    upper = geom(6, [.015, .015, .01], (d, d, .01))
    return footprint([lower, upper], np.array([d / 2, d / 2, 0.]), np.array([0, 0, 1.]), force, guard=guard)


@pytest.mark.parametrize('force', [10.0, 34.0])
def test_the_floor_has_no_jump_where_the_overlap_can_first_hold_the_rounding(force):
    """Review 2026-09-25, finding 4: the first rule floored only overlaps too narrow to hold a 1 mm disk,
    so 1.99 x 1.99 mm under 34 N read 1.09 MPa and 2.01 x 2.01 mm 8.42 MPa (over the harm line). Now both
    are floored at the rounded corner's contact, and the pressure moves smoothly with the overlap."""
    narrow, wide = corner_pair(.00199, force), corner_pair(.00201, force)
    assert narrow['footprint_guard']['flat_parts_touch'] is False and wide['footprint_guard']['flat_parts_touch'] is True
    assert narrow['mean_pressure_pa'] == pytest.approx(force / corner_regime_area(force), rel=1e-9)
    assert wide['mean_pressure_pa'] == pytest.approx(narrow['mean_pressure_pa'], rel=1e-9)
    assert corner_pair(.00201, force, None)['mean_pressure_pa'] > 3 * wide['mean_pressure_pa']
    if force == 34.0:
        assert wide['mean_pressure_pa'] == pytest.approx(1.09e6, rel=0.01)
    # the floor lets go smoothly as the overlap grows past the corner's contact
    sizes = np.linspace(.002, .012, 41)
    p = np.array([corner_pair(s, force)['mean_pressure_pa'] for s in sizes])
    assert (np.diff(p) <= 1e-6 * p[:-1]).all(), 'pressure never rises as the overlap grows'
    assert np.abs(np.diff(np.log(p))).max() < 0.2, 'no step larger than one grid step of overlap allows'


def test_an_overlap_that_holds_the_rounding_is_floored_too_when_its_edge_carries_more():
    """The review's feet pressed together under 34 N: a 3.56 x 10.81 mm overlap, whose rounded edge's contact
    is larger, now reads the edge's contact (0.79 MPa) instead of the overlap's."""
    lower = geom(6, [.015, .015, .01], (0, 0, -.01))
    upper = geom(6, [.015, .015, .01], (.03 - .00356, .03 - .01081, .01))
    point = np.array([(.03 - .00356) / 2, (.03 - .01081) / 2, 0.])
    on = footprint([lower, upper], point, np.array([0, 0, 1.]), 34.0, guard=GUARD)
    off = footprint([lower, upper], point, np.array([0, 0, 1.]), 34.0)
    assert off['area_m2'] == pytest.approx(.00356 * .01081, rel=1e-9)
    assert on['method'] == RESCORED_METHOD and on['footprint_guard']['flat_parts_touch'] is True
    assert on['area_m2'] == pytest.approx(rounded_edge_contact(34.0, .01081, 1e-3, E)[0], rel=1e-9)
    assert on['mean_pressure_pa'] == pytest.approx(0.785e6, rel=0.01)
    assert on['mean_pressure_pa'] < off['mean_pressure_pa']


# ------------------------------------------------------------------ the ceiling and the off path
def test_the_ceiling_bounds_a_footprint_the_edge_rule_does_not_reach():
    """A 0.1 mm sphere under 1,000 N reaches its silhouette cap at 3.2e10 Pa; the ceiling outranks it."""
    sphere = geom(2, [1e-4, 0, 0], (0, 0, 1e-4))
    off = footprint([sphere, geom(0, [1, 1, 1])], np.zeros(3), np.array([0, 0, 1.]), 1000.)
    on = footprint([sphere, geom(0, [1, 1, 1])], np.zeros(3), np.array([0, 0, 1.]), 1000., guard=GUARD)
    assert off['geometry_cap_reached'] and off['mean_pressure_pa'] > 1e10
    assert on['mean_pressure_pa'] == pytest.approx(GUARD.pressure_ceiling_pa)
    assert on['footprint_guard']['area_before_ceiling_m2'] == pytest.approx(off['area_m2'])
    assert max(on['pressure_bounds_pa']) <= GUARD.pressure_ceiling_pa * (1 + 1e-12)
    assert on['method'] == off['method'] == 'hertz_circular'


def test_the_ceiling_bounds_the_sensor_support_fallback_record():
    record = dict(supported=True, geometry_supported=False, method='legacy_sensor_support_fallback',
                  normal_force_n=50., area_m2=1e-6, mean_pressure_pa=5e7, tangent_axes_m=[5.6e-4] * 2)
    bound_mean_pressure(record, GUARD)
    assert record['area_m2'] == pytest.approx(5e-6) and record['mean_pressure_pa'] == pytest.approx(1e7)
    assert record['tangent_axes_m'][0] == pytest.approx(math.sqrt(5e-6 / math.pi))
    assert record['footprint_guard']['area_before_ceiling_m2'] == 1e-6
    ordinary = dict(record, normal_force_n=1., area_m2=1e-5, mean_pressure_pa=1e5)
    ordinary.pop('footprint_guard')
    before = dict(ordinary)
    assert bound_mean_pressure(ordinary, GUARD) == before
    assert bound_mean_pressure(dict(record, area_m2=1e-9), None)['area_m2'] == 1e-9


@pytest.mark.parametrize('case', ['sphere', 'capsule_side', 'capsule_end', 'box_on_plane', 'wide_box_overlap'])
def test_ordinary_contacts_are_identical_with_the_guard_on(case):
    plane = geom(0, [1, 1, 1])
    args = dict(sphere=([geom(2, [.02, 0, 0]), plane], [0, 0, -.02], [0, 0, 1.], 10.),
                capsule_side=([geom(3, [.02, .05, 0]), plane], [.02, 0, 0], [1., 0, 0], 10.),
                capsule_end=([geom(3, [.02, .05, 0]), plane], [0, 0, .07], [0, 0, 1.], 10.),
                box_on_plane=([geom(6, [.02, .03, .01]), plane], [0, 0, -.01], [0, 0, 1.], 10.),
                wide_box_overlap=([geom(6, [.02, .03, .01]), geom(6, [.02, .03, .01], (.02, 0, -.02))],
                                  [.01, 0, -.01], [0, 0, 1.], 10.))[case]
    shapes, point, normal, force = args
    off = footprint(shapes, np.array(point), np.array(normal), force)
    on = footprint(shapes, np.array(point), np.array(normal), force, guard=GUARD)
    assert on == off and 'footprint_guard' not in off


def test_the_stop_rule_passes_every_change_the_guard_makes_and_catches_any_other():
    """The guard judged on its own record (review 2026-09-25, finding 1): a floored flat overlap, a ceiling
    case and a bounded fallback record keep the rule; a smaller area, a wrong method or an unknown note break it."""
    from mimo_bridge.contact_footprint import check_guard_note
    floored = palm(guard=GUARD)
    sphere = geom(2, [1e-4, 0, 0], (0, 0, 1e-4))
    ceiling = footprint([sphere, geom(0, [1, 1, 1])], np.zeros(3), np.array([0, 0, 1.]), 1000., guard=GUARD)
    fallback = bound_mean_pressure(dict(supported=True, method='legacy_sensor_support_fallback', normal_force_n=50.,
                                        area_m2=1e-6, mean_pressure_pa=5e7, tangent_axes_m=[5.6e-4] * 2), GUARD)
    corner = corner_pair(.00201, 34.0)
    for record in (floored, ceiling, fallback, corner):
        assert check_guard_note(record, GUARD) is None
    shrunk = dict(floored, area_m2=floored['footprint_guard']['flat_overlap_area_m2'] / 2)
    assert 'under its flat overlap' in check_guard_note(shrunk, GUARD)
    assert 'the rule gives' in check_guard_note(dict(corner, area_m2=corner['area_m2'] * 1.5), GUARD)
    assert 'method' in check_guard_note(dict(floored, method='aligned_flat_face_overlap'), GUARD)
    assert 'ceiling case' in check_guard_note(dict(ceiling, area_m2=ceiling['area_m2'] * 2), GUARD)
    assert 'no known kind' in check_guard_note(dict(floored, footprint_guard=dict(setting='rounded_edge_v1')), GUARD)


def test_the_env_checks_every_change_at_every_tick_not_only_the_sampled_ones():
    """The stop rule's live count: the env checks each guarded record of every physics tick (the stats
    records sample one tick in thousands) and counts a record outside the rule, or one the check cannot
    read, as a violation with its reason. Driven through the env's own method on a stand-in env."""
    from types import SimpleNamespace
    from mimo_bridge.env import BridgeEnv
    counts = dict(rescored_flat_overlaps=0, pressure_ceiling_contacts=0, largest_unguarded_pressure_pa=0.0,
                  rule_violations=0, last_violation=None)
    env = SimpleNamespace(_footprint_guard=GUARD, data=SimpleNamespace(time=1.5), _footprint_guard_counts=counts)
    good = dict(palm(guard=GUARD), body_id=3)
    BridgeEnv._count_footprint_guard(env, [good, dict(palm(), body_id=3)])
    assert counts['rescored_flat_overlaps'] == 1 and counts['rule_violations'] == 0
    assert counts['largest_unguarded_pressure_pa'] == pytest.approx(FORCE / 1.1651988742943993e-11, rel=1e-6)
    shrunk = dict(good, area_m2=good['footprint_guard']['flat_overlap_area_m2'] / 2)
    unreadable = dict(good, footprint_guard={k: v for k, v in good['footprint_guard'].items()
                                             if k != 'flat_overlap_length_m'})
    BridgeEnv._count_footprint_guard(env, [shrunk])
    assert counts['rule_violations'] == 1 and counts['last_violation'].startswith('body 3 at 1.500 s:')
    assert 'under its flat overlap' in counts['last_violation']
    BridgeEnv._count_footprint_guard(env, [unreadable])
    assert counts['rule_violations'] == 2 and 'the rule check failed' in counts['last_violation']


def test_the_guard_is_named_and_its_settings_are_checked():
    assert FOOTPRINT_GUARDS['off'] is None and GUARD.edge_rounding_m == 1e-3 and GUARD.pressure_ceiling_pa == 1e7
    with pytest.raises(ValueError):
        FootprintGuard(edge_rounding_m=0.)
    with pytest.raises(ValueError):
        FootprintGuard(pressure_ceiling_pa=float('nan'))


def test_aggregation_passes_the_guard_and_defaults_to_off():
    from types import SimpleNamespace
    model = SimpleNamespace(geom_type=np.array([6, 6]), geom_size=np.array([PALM['size'], HEAVY_BOX['size']]))
    data = SimpleNamespace(geom_xpos=np.array([PALM['position'], HEAVY_BOX['position']]),
                           geom_xmat=np.array([np.ravel(PALM['rotation']), np.ravel(HEAVY_BOX['rotation'])]),
                           xmat=np.tile(np.eye(3).ravel(), (2, 1)))
    contact = ContactInfo(0, (0, 1), FORCE, np.zeros(3), np.zeros(3), np.array(POINT), np.array(NORMAL), 0., 0.)
    off = aggregate_contact_footprints(model, data, [contact])[0]
    on = aggregate_contact_footprints(model, data, [contact], VirtualSkinProfile(), guard=GUARD)[0]
    assert off['method'] == 'aligned_flat_face_overlap' and off['mean_pressure_pa'] > 1e11
    assert on['method'] == RESCORED_METHOD and on['mean_pressure_pa'] == pytest.approx(569225.2, rel=1e-6)
