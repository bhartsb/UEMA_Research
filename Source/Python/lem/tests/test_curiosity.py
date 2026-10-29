# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
test_curiosity.py — Tests for curiosity system (UEMA Spec §21).

Tests:
  - Drive formula returns correct values at key PE points, clamped to [0, 1]
  - Zone classification is correct
  - Validation rejects invalid pe_min/pe_max
  - Scalar multiplication with tensor preserves gradient flow
"""

import pytest
import torch

from lem.model.curiosity import CuriositySystem


@pytest.fixture
def cs():
    return CuriositySystem(pe_min=0.1, pe_max=0.4)


class TestDriveFormula:
    """Verify compute_drive at spec-defined PE values (clamped to [0, 1])."""

    def test_pe_zero_bored(self, cs):
        # Raw: 0/0.1 - 1 = -1.0, clamped to 0
        assert cs.compute_drive(0.0) == pytest.approx(0.0)

    def test_pe_min_boundary(self, cs):
        # Raw: 0.1/0.1 - 1 = 0.0
        assert cs.compute_drive(0.1) == pytest.approx(0.0)

    def test_pe_midpoint_peak(self, cs):
        assert cs.compute_drive(0.25) == pytest.approx(1.0)

    def test_pe_max_boundary(self, cs):
        # Raw: 1 - 0.4/0.4 = 0.0
        assert cs.compute_drive(0.4) == pytest.approx(0.0)

    def test_pe_high_overwhelmed(self, cs):
        # Raw: 1 - 1.0/0.4 = -1.5, clamped to 0
        assert cs.compute_drive(1.0) == pytest.approx(0.0)

    def test_pe_very_high_overwhelmed(self, cs):
        # Raw: 1 - 5.0/0.4 = -11.5, clamped to 0
        assert cs.compute_drive(5.0) == pytest.approx(0.0)

    def test_drive_always_in_unit_range(self, cs):
        for pe in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 1.0, 10.0]:
            d = cs.compute_drive(pe)
            assert 0.0 <= d <= 1.0, f"pe={pe} gave drive={d}"

    def test_goldilocks_symmetric(self, cs):
        d_low = cs.compute_drive(0.15)
        d_high = cs.compute_drive(0.35)
        assert d_low == pytest.approx(d_high)

    def test_goldilocks_interior_positive(self, cs):
        for pe in [0.15, 0.20, 0.25, 0.30, 0.35]:
            assert cs.compute_drive(pe) > 0.0

    def test_bored_zone_zero(self, cs):
        for pe in [0.0, 0.02, 0.05, 0.08]:
            assert cs.compute_drive(pe) == pytest.approx(0.0)

    def test_overwhelmed_zone_zero(self, cs):
        for pe in [0.5, 1.0, 2.0, 10.0]:
            assert cs.compute_drive(pe) == pytest.approx(0.0)


class TestBandEdgeContinuity:
    """The drive must not jump at a band edge, and must reach the low values.

    Every boundary test above approaches an edge from OUTSIDE, where the two
    outer branches hold. Dividing the tent by the full band width instead of the
    half width left it at 0.5 just INSIDE both edges while the outer branches
    were already 0, so the drive stepped by 0.5 at pe_min and again at pe_max and
    no value between 0 and 0.5 was reachable for any band.
    """

    BANDS = [(0.1, 0.4), (0.02, 0.4), (0.01, 0.05), (0.37, 0.93)]

    @pytest.mark.parametrize("pe_min,pe_max", BANDS)
    def test_no_step_at_either_edge(self, pe_min, pe_max):
        cs = CuriositySystem(pe_min=pe_min, pe_max=pe_max)
        eps = (pe_max - pe_min) * 1e-9
        for edge in (pe_min, pe_max):
            inside = cs.compute_drive(edge + eps if edge == pe_min else edge - eps)
            outside = cs.compute_drive(edge - eps if edge == pe_min else edge + eps)
            assert inside < 1e-6, f"band {pe_min}:{pe_max} edge {edge} steps to {inside} inside"
            assert abs(inside - outside) < 1e-6, (
                f"band {pe_min}:{pe_max} jumps {abs(inside - outside)} across edge {edge}")

    @pytest.mark.parametrize("pe_min,pe_max", BANDS)
    def test_midpoint_still_peaks_at_one(self, pe_min, pe_max):
        cs = CuriositySystem(pe_min=pe_min, pe_max=pe_max)
        assert cs.compute_drive((pe_min + pe_max) / 2) == pytest.approx(1.0)

    @pytest.mark.parametrize("pe_min,pe_max", BANDS)
    def test_every_tenth_of_the_range_is_reachable(self, pe_min, pe_max):
        """A dense sweep must land in all ten deciles, not only {0} and [0.5, 1]."""
        cs = CuriositySystem(pe_min=pe_min, pe_max=pe_max)
        step = (pe_max - pe_min) / 4000.0
        seen = set()
        pe = pe_min
        while pe <= pe_max:
            d = cs.compute_drive(pe)
            if d > 0.0:
                seen.add(min(int(d * 10), 9))
            pe += step
        missing = sorted(set(range(10)) - seen)
        assert not missing, f"band {pe_min}:{pe_max} never produces a drive in deciles {missing}"

    def test_drive_halves_midway_between_midpoint_and_edge(self):
        """Quarter of the way in from an edge is half the peak on a proper tent."""
        cs = CuriositySystem(pe_min=0.1, pe_max=0.4)
        assert cs.compute_drive(0.175) == pytest.approx(0.5)
        assert cs.compute_drive(0.325) == pytest.approx(0.5)


class TestZoneClassification:
    def test_bored(self, cs):
        assert cs.zone(0.0) == "bored"
        assert cs.zone(0.05) == "bored"
        assert cs.zone(0.1) == "bored"  # boundary → bored

    def test_goldilocks(self, cs):
        assert cs.zone(0.15) == "goldilocks"
        assert cs.zone(0.25) == "goldilocks"
        assert cs.zone(0.35) == "goldilocks"

    def test_overwhelmed(self, cs):
        assert cs.zone(0.4) == "overwhelmed"  # boundary → overwhelmed
        assert cs.zone(1.0) == "overwhelmed"
        assert cs.zone(5.0) == "overwhelmed"


class TestValidation:
    def test_pe_min_zero_raises(self):
        with pytest.raises(ValueError):
            CuriositySystem(pe_min=0.0, pe_max=0.4)

    def test_pe_min_negative_raises(self):
        with pytest.raises(ValueError):
            CuriositySystem(pe_min=-0.1, pe_max=0.4)

    def test_pe_min_gte_pe_max_raises(self):
        with pytest.raises(ValueError):
            CuriositySystem(pe_min=0.4, pe_max=0.4)

    def test_custom_range(self):
        cs = CuriositySystem(pe_min=0.01, pe_max=0.05)
        assert cs.compute_drive(0.03) == pytest.approx(1.0)  # midpoint


class TestGradientFlow:
    """Verify that drive * tensor preserves autograd."""

    def test_positive_drive_preserves_grad(self):
        cs = CuriositySystem()
        w = torch.tensor(3.0, requires_grad=True)
        loss = -w  # simple wellness-like loss
        drive = cs.compute_drive(0.25)  # 1.0 (goldilocks peak)
        scaled = drive * loss
        scaled.backward()
        assert w.grad is not None
        assert w.grad.item() == pytest.approx(-1.0)

    def test_zero_drive_no_grad(self):
        cs = CuriositySystem()
        w = torch.tensor(3.0, requires_grad=True)
        loss = -w
        drive = cs.compute_drive(0.0)  # 0.0 (bored)
        scaled = drive * loss
        scaled.backward()
        assert w.grad is not None
        assert w.grad.item() == pytest.approx(0.0)

    def test_overwhelmed_drive_no_grad(self):
        cs = CuriositySystem()
        w = torch.tensor(3.0, requires_grad=True)
        loss = -w
        drive = cs.compute_drive(1.0)  # 0.0 (overwhelmed)
        scaled = drive * loss
        scaled.backward()
        assert w.grad is not None
        assert w.grad.item() == pytest.approx(0.0)
