# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Pin what a fall can and cannot teach the model.

Measured 2026-09-16 on the live path (scene crib, --contact-profile
mechanical_v2_local, age 24 months, 7 drops per height, trials that start inside
geometry thrown away):

  peak per-patch pressure, the signal wellness reads today, FALLS with height at
  three of five rungs - 1,759,197 Pa from 0.30 m against 764,016 Pa from 0.90 m -
  and its spread at one height (18.1x) is larger than its whole rise across
  heights (13.8x)

  normal force on the hardest-loaded body RISES at every rung, 1,046 N from
  0.05 m to 4,535 N from 0.90 m, with a worst within-height spread of 1.6x
  against a 4.3x rise

These tests hold the two formula properties behind that result, so a later change
that reintroduces either one is caught here rather than in a run.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from lem.skin_mechanics import PROFILE, mechanical_load


def aversion(pressure_pa, held_seconds=0.0, profile=PROFILE):
    """What the live formula returns for one site at this pressure."""
    p = torch.tensor([float(pressure_pa)])
    history = torch.zeros(1, 3)
    history[0, 0] = float(pressure_pa)                      # decaying peak, Pa
    history[0, 1] = float(pressure_pa) * float(held_seconds)  # exposure, Pa.s
    return float(mechanical_load(p, history, torch.ones(1), profile))


def test_a_ninety_centimetre_fall_reads_less_painful_than_a_thirty():
    """The inversion itself, from measured inputs through the live formula.

    These two pressures are the measured medians of 7 drops each on the live
    path (mechanical_v2_local, age 24 months, 2026-09-16). The 0.90 m drop
    peaked LOWER than the 0.30 m one because peak patch pressure is divided by
    the sensor-support area, not a contact area, so the reading depends on which
    body part happened to land. Fed through the aversion curve, three times the
    fall height reads as less hurt.
    """
    from_thirty_cm = aversion(1_759_197)
    from_ninety_cm = aversion(764_016)
    assert from_ninety_cm < from_thirty_cm, (
        "the live path no longer inverts here; if that is a deliberate fix, "
        "retake the drop medians in this module's docstring and rewrite this test"
    )
    assert abs(from_thirty_cm - 0.9991) < 5e-4
    assert abs(from_ninety_cm - 0.9164) < 5e-4


def test_a_fall_below_the_reference_pressure_registers_no_pain_at_all():
    """The acute term is clamped at the reference, so low falls score zero PAIN.

    Measured medians: a 0.05 m drop peaks at 55,315 Pa and a 0.10 m drop at
    99,275 Pa, which are 0.25 and 0.45 times the 219,400 Pa reference. The clamp
    puts the pain of both at exactly zero.

    That does NOT make those falls free. Comfort is
    (0.5 + 0.35 * pleasant) * (1 - pain), and the pleasant half falls off its
    5 kPa hill long before the pain half engages: measured comfort runs 0.7081
    lying still, 0.5679 after a 5 cm drop and 0.5297 after a 10 cm one. The low
    rungs therefore DO grade, through pleasantness. It is the rungs above 30 cm
    that invert.
    """
    assert aversion(55_315) == 0.0
    assert aversion(99_275) == 0.0
    assert aversion(PROFILE.pressure_reference_pa) == 0.0


def test_the_usable_range_ends_at_six_and_a_half_times_the_reference():
    """Name the pressure past which the curve stops carrying information.

    'Usable' means one more quarter-multiple of load still moves the output by
    more than 0.001. Measured 2026-09-16: that runs out at 6.5x the reference,
    1,426,100 Pa. Every drop from 0.30 m up peaked between 3.48x and 8.02x, so
    the whole upper half of the ladder sits in or past the flat tail.
    """
    reference = PROFILE.pressure_reference_pa
    last_useful, multiple = None, 1.0
    while multiple <= 40.0:
        if aversion(reference * (multiple + 0.25)) - aversion(reference * multiple) > 0.001:
            last_useful = multiple + 0.25
        multiple += 0.25
    assert last_useful == pytest.approx(6.5), (
        f"the curve now carries information to {last_useful}x the reference "
        "instead of 6.5x; the drop measurements in this module's docstring were "
        "taken against 6.5x and need retaking"
    )
    assert reference * last_useful == pytest.approx(1_426_100)


def test_a_held_load_and_a_single_press_are_not_the_same_number():
    """The live profile is a time-under-load model, so holding must cost more."""
    press = aversion(100_000, held_seconds=0.0)
    held = aversion(100_000, held_seconds=600.0)
    assert held > press
    assert press < 0.01, "a single 100 kPa press should cost almost nothing here"


def test_pressure_decomposition_shares_sum_to_one():
    """The attribution in contact_concentration is exact, not an approximation.

    pressure = force / area, so log P = log F - log A, and the two covariance
    shares of var(log P) must sum to 1 for ANY data. A share outside 0 to 100
    percent is a real result - it means force and area move together - not a bug,
    so the test checks the sum rather than the range.
    """
    numpy = pytest.importorskip("numpy")
    rng = numpy.random.default_rng(0)
    for _ in range(20):
        force = numpy.exp(rng.normal(0, 1, 40))
        area = numpy.exp(rng.normal(0, 1, 40))
        pressure = force / area
        lf, la, lp = numpy.log(force), numpy.log(area), numpy.log(pressure)
        var = numpy.var(lp)
        share_force = numpy.cov(lp, lf, bias=True)[0, 1] / var
        share_area = numpy.cov(lp, -la, bias=True)[0, 1] / var
        assert math.isclose(share_force + share_area, 1.0, abs_tol=1e-9)


def test_anti_correlated_force_and_area_put_the_force_share_below_zero():
    """The case measured on real drops, held so the sign is not read as a bug.

    Across 30 measured drop peaks the force share came out at -20.5 percent:
    the hardest-pressed readings were fingertips carrying small forces onto tiny
    areas, so the reading ran OPPOSITE to impact force at -0.234. Constructing
    that case here shows a negative share is what the arithmetic does when a
    small contact area and a small force arrive together.
    """
    numpy = pytest.importorskip("numpy")
    # small area travels with small force, exactly like a fingertip landing
    force = numpy.array([250.0, 260.0, 240.0, 3200.0, 1860.0, 1040.0])
    area_m2 = numpy.array([7e-5, 7e-5, 7e-5, 2.4e-3, 1.9e-2, 1.9e-2])
    pressure = force / area_m2
    lf, la, lp = numpy.log(force), numpy.log(area_m2), numpy.log(pressure)
    var = numpy.var(lp)
    share_force = numpy.cov(lp, lf, bias=True)[0, 1] / var
    assert share_force < 0.0
    assert numpy.corrcoef(lf, lp)[0, 1] < 0.0
    assert math.isclose(
        share_force + numpy.cov(lp, -la, bias=True)[0, 1] / var, 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------- mechanical_v4

def test_v4_changes_only_the_squash_and_nothing_else():
    """Every onset, time constant and unit is shared with the default profile."""
    from lem.skin_mechanics import PROFILE, WIDE_RANGE_PROFILE
    from dataclasses import asdict
    a, b = asdict(PROFILE), asdict(WIDE_RANGE_PROFILE)
    differing = {k for k in a if a[k] != b[k]}
    assert differing == {"aversion_shape"}
    assert a["aversion_shape"] == "exponential"
    assert b["aversion_shape"] == "hyperbolic"


def test_v4_scores_zero_wherever_the_old_curve_did():
    """A load that cost nothing must still cost nothing, or this is not a squash."""
    from lem.skin_mechanics import WIDE_RANGE_PROFILE as WIDE
    for pa in (0.0, 1_000.0, 99_275.0, PROFILE.pressure_reference_pa):
        assert aversion(pa) == 0.0
        assert aversion(pa, profile=WIDE) == 0.0


def test_v4_never_ranks_two_loads_differently_from_the_old_curve():
    """Both shapes rise with load, so the ORDER of any two loads is the same.

    Only the spacing changes. Checked across four decades of pressure so a
    reordering anywhere in the usable range would be caught.
    """
    from lem.skin_mechanics import WIDE_RANGE_PROFILE as WIDE
    pressures = [PROFILE.pressure_reference_pa * m
                 for m in (1.0, 1.1, 1.5, 2, 3, 5, 8, 13, 21, 34, 55, 89)]
    old = [aversion(p) for p in pressures]
    new = [aversion(p, profile=WIDE) for p in pressures]
    assert old == sorted(old)
    assert new == sorted(new)
    for i in range(len(pressures)):
        for j in range(i + 1, len(pressures)):
            assert (old[i] < old[j]) == (new[i] < new[j]) or old[i] == old[j]


def test_v4_still_separates_the_loads_real_falls_produce():
    """The point of the change, measured against the drops that motivated it.

    Median peak pressures from 12 drops a height on the resolved-geometry path.
    Under the old squash the top two rungs land 0.000079 apart, which no learner
    can use. Under the new one they land far enough apart to rank.
    """
    from lem.skin_mechanics import WIDE_RANGE_PROFILE as WIDE
    sixty_six_cm, ninety_cm = 2_267_114.0, 2_777_385.0
    old_gap = aversion(ninety_cm) - aversion(sixty_six_cm)
    new_gap = aversion(ninety_cm, profile=WIDE) - aversion(sixty_six_cm, profile=WIDE)
    assert old_gap < 0.001
    assert new_gap > 0.01
    assert new_gap / old_gap > 100


def test_contact_response_picks_the_wide_curve_only_for_v4():
    """The profile NAME selects the squash; no other profile may move."""
    from lem.model.contact_wellness import contact_response

    def pain(profile, pa=2_777_385.0):
        features = torch.zeros(1, 4, 8)
        features[..., 0] = pa / 1e5
        features[..., 3] = 0.01
        history = torch.zeros(1, 4, 3)
        history[..., 0] = pa
        return float(contact_response(features, profile=profile,
                                      history=history)["pain"].max())

    same = [pain(p) for p in ("mechanical_v2", "mechanical_v2_local", "mechanical_v3")]
    assert max(same) - min(same) < 1e-9, "a non-v4 profile changed"
    assert same[0] > 0.999
    assert pain("mechanical_v4") < same[0]
    assert 0.9 < pain("mechanical_v4") < 0.95


def test_an_unknown_aversion_shape_is_refused_at_construction():
    from lem.skin_mechanics import MechanicalProfile
    with pytest.raises(ValueError, match="aversion shape"):
        MechanicalProfile(aversion_shape="whatever")


def test_the_comfort_numbers_the_explainer_quotes():
    """Hold the two worked examples in term.eval-tactile-comfort to the code.

    The entry previously said a 100 kPa press held ten minutes scores 0.42.
    Measured with the inputs named in the entry - 53 patches, area 0.01 of the
    area unit, mid-range texture - it scores 0.0056; 0.42 is what about 31
    seconds of that press gives. Both worked examples are pinned so the prose
    and the code cannot drift apart again.
    """
    from lem.model.contact_wellness import contact_response

    def comfort(pressure_pa, held_seconds, profile, sites=53):
        features = torch.zeros(1, sites, 8)
        features[..., 0] = pressure_pa / 1e5
        features[..., 3] = 0.01
        features[..., 4:8] = 0.5
        history = torch.zeros(1, sites, 3)
        history[..., 0] = pressure_pa
        history[..., 1] = pressure_pa * held_seconds
        return float(contact_response(features, profile=profile,
                                      history=history)["comfort"].mean())

    for profile in ("mechanical_v2_local", "mechanical_v3", "mechanical_v4"):
        assert abs(comfort(219_400, 0.05, profile) - 0.5069) < 5e-4

    for profile in ("mechanical_v2_local", "mechanical_v3"):
        assert abs(comfort(100_000, 600.0, profile) - 0.0056) < 5e-4
    assert abs(comfort(100_000, 600.0, "mechanical_v4") - 0.0952) < 5e-4


# ------------------------------------------------- skin field width, 8 vs 10

def test_the_contact_predictor_is_built_for_the_tokenizer_it_is_fed():
    """The width that broke every live call, pinned on both sides.

    tokenizer.modalities.touch._FIELD_NORMS defines how many numbers one skin
    site carries. It went from 8 to 10 when the skin-stretch channels landed;
    ContactConsequences kept a hard-coded 8 and every live call died inside
    nn.Linear with a bare shape error (built for 3+8=11 columns, fed 3+10=13).
    """
    from lem.model.contact_consequences import ContactConsequences
    from tokenizer.modalities.touch import _FIELD_NORMS

    width = len(_FIELD_NORMS)
    assert width == 10, (
        f"the tokenizer now carries {width} touch fields; the predictor is "
        "constructed from this number so nothing needs changing here, but the "
        "counts quoted in these tests do"
    )

    model = ContactConsequences(16, n_handles=2, n_features=width)
    assert model.site[0].in_features == 3 + width
    assert model.delta[-1].out_features == width

    out = model(torch.randn(1, 2, 16), torch.randn(1, 2, 16),
                torch.ones(1, 2, dtype=torch.bool), torch.randn(1, 2, 3),
                torch.zeros(1, 2, width), torch.ones(1, 2, dtype=torch.bool),
                torch.randn(1, 2, 2, 9), torch.full((1, 2), .05), torch.ones(1, 10))
    assert out["features"].shape == (1, 2, width)


def test_a_width_mismatch_says_what_is_wrong_instead_of_dying_in_nn_linear():
    from lem.model.contact_consequences import ContactConsequences
    model = ContactConsequences(16, n_handles=2, n_features=8)
    with pytest.raises(ValueError, match="10 wide but this predictor"):
        model(torch.randn(1, 2, 16), torch.randn(1, 2, 16),
              torch.ones(1, 2, dtype=torch.bool), torch.randn(1, 2, 3),
              torch.zeros(1, 2, 10), torch.ones(1, 2, dtype=torch.bool),
              torch.randn(1, 2, 2, 9), torch.full((1, 2), .05), torch.ones(1, 10))


def test_the_default_width_still_loads_an_older_eight_wide_predictor():
    """Default 8 is kept so a checkpoint saved before the change still loads."""
    from lem.model.contact_consequences import ContactConsequences
    assert ContactConsequences(16).n_features == 8
    assert ContactConsequences(16).site[0].in_features == 11
