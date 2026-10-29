# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""test_body_sets.py -- Tests for handle-to-body-set mapping (UEMA Spec Section 2.13).

Validates that the Python-side body_sets module faithfully mirrors the C++
EBodySet enum and HandleToBodySet map in MotorCortexComponent.h/.cpp.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lem.model.body_sets import (
    BodySet,
    BODY_SET_HANDLES,
    HANDLE_BODY_SET,
    HANDLE_NAMES,
    NUM_BODY_SETS,
    NUM_HANDLES,
    body_set_for_handle,
    body_sets_for_handles,
    handle_mask_for_body_set,
    handles_for_body_set,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_num_body_sets_matches_enum(self):
        assert NUM_BODY_SETS == len(BodySet)

    def test_num_handles_matches_names(self):
        assert NUM_HANDLES == len(HANDLE_NAMES)

    def test_num_handles_matches_mapping(self):
        assert NUM_HANDLES == len(HANDLE_BODY_SET)

    def test_num_handles_is_23(self):
        assert NUM_HANDLES == 24    # 23 limbs + gaze (2026-08-24)

    def test_num_body_sets_is_10(self):
        assert NUM_BODY_SETS == 10


# ---------------------------------------------------------------------------
# Enum ordinals match C++ EBodySet (MotorCortexComponent.h:28-41)
# ---------------------------------------------------------------------------

class TestEnumOrdinals:
    def test_arm_l(self):
        assert BodySet.Arm_L == 0

    def test_arm_r(self):
        assert BodySet.Arm_R == 1

    def test_hand_l(self):
        assert BodySet.Hand_L == 2

    def test_hand_r(self):
        assert BodySet.Hand_R == 3

    def test_spine(self):
        assert BodySet.Spine == 4

    def test_head(self):
        assert BodySet.Head == 5

    def test_leg_l(self):
        assert BodySet.Leg_L == 6

    def test_leg_r(self):
        assert BodySet.Leg_R == 7

    def test_foot_l(self):
        assert BodySet.Foot_L == 8

    def test_foot_r(self):
        assert BodySet.Foot_R == 9


# ---------------------------------------------------------------------------
# Every handle assigned to exactly one body set
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_all_handles_assigned(self):
        for i in range(NUM_HANDLES):
            bs = body_set_for_handle(i)
            assert isinstance(bs, BodySet), f"Handle {i} not assigned to a BodySet"

    def test_nine_body_sets_have_handles(self):
        """9 of 10 body sets have at least one handle; Spine has 0 (by design)."""
        non_empty = [bs for bs in BodySet if len(handles_for_body_set(bs)) > 0]
        assert len(non_empty) == 9

    def test_spine_has_no_handles(self):
        assert handles_for_body_set(BodySet.Spine) == ()


# ---------------------------------------------------------------------------
# Known groupings (from C++ HandleToBodySet, MotorCortexComponent.cpp:1267-1302)
# ---------------------------------------------------------------------------

class TestKnownGroupings:
    def test_left_fingers_in_hand_l(self):
        # handles 1-5: thumb_L, index_L, middle_L, ring_L, pinky_L
        for i in range(1, 6):
            assert body_set_for_handle(i) == BodySet.Hand_L, \
                f"Handle {i} ({HANDLE_NAMES[i]}) should be Hand_L"

    def test_right_fingers_in_hand_r(self):
        # handles 7-11: thumb_R, index_R, middle_R, ring_R, pinky_R
        for i in range(7, 12):
            assert body_set_for_handle(i) == BodySet.Hand_R, \
                f"Handle {i} ({HANDLE_NAMES[i]}) should be Hand_R"

    def test_head(self):
        assert body_set_for_handle(12) == BodySet.Head

    def test_wrists_in_arm(self):
        assert body_set_for_handle(0) == BodySet.Arm_L   # wrist_L
        assert body_set_for_handle(6) == BodySet.Arm_R   # wrist_R

    def test_upper_lower_arms(self):
        assert body_set_for_handle(13) == BodySet.Arm_L  # upperarm_L
        assert body_set_for_handle(14) == BodySet.Arm_L  # lowerarm_L
        assert body_set_for_handle(15) == BodySet.Arm_R  # upperarm_R
        assert body_set_for_handle(16) == BodySet.Arm_R  # lowerarm_R

    def test_legs(self):
        assert body_set_for_handle(17) == BodySet.Leg_L  # thigh_L
        assert body_set_for_handle(18) == BodySet.Leg_L  # calf_L
        assert body_set_for_handle(19) == BodySet.Leg_R  # thigh_R
        assert body_set_for_handle(20) == BodySet.Leg_R  # calf_R

    def test_feet(self):
        assert body_set_for_handle(21) == BodySet.Foot_L
        assert body_set_for_handle(22) == BodySet.Foot_R

    def test_arm_l_has_3_handles(self):
        """Arm_L: wrist_L(0), upperarm_L(13), lowerarm_L(14)."""
        assert handles_for_body_set(BodySet.Arm_L) == (0, 13, 14)

    def test_hand_l_has_5_handles(self):
        """Hand_L: thumb(1), index(2), middle(3), ring(4), pinky(5)."""
        assert handles_for_body_set(BodySet.Hand_L) == (1, 2, 3, 4, 5)

    def test_head_has_1_handle(self):
        assert handles_for_body_set(BodySet.Head) == (12, 23)   # head + gaze


# ---------------------------------------------------------------------------
# Reverse mapping consistency
# ---------------------------------------------------------------------------

class TestReverseMapping:
    def test_round_trip(self):
        """handle -> body_set -> handles includes original handle."""
        for i in range(NUM_HANDLES):
            bs = body_set_for_handle(i)
            assert i in handles_for_body_set(bs)

    def test_no_duplicates_in_reverse(self):
        """No handle appears in more than one body set's handle list."""
        seen: set[int] = set()
        for bs in BodySet:
            handles = handles_for_body_set(bs)
            for h in handles:
                assert h not in seen, f"Handle {h} appears in multiple body sets"
                seen.add(h)

    def test_all_handles_covered_by_reverse(self):
        """Union of all body-set handle lists == {0..22}."""
        all_handles: set[int] = set()
        for bs in BodySet:
            all_handles.update(handles_for_body_set(bs))
        assert all_handles == set(range(NUM_HANDLES))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_body_sets_for_handles_early_infant(self):
        """Curriculum early_infant indices (0, 6, 12) -> {Arm_L, Arm_R, Head}."""
        result = body_sets_for_handles((0, 6, 12))
        assert result == {BodySet.Arm_L, BodySet.Arm_R, BodySet.Head}

    def test_body_sets_for_handles_empty(self):
        assert body_sets_for_handles(()) == set()

    def test_body_sets_for_handles_all(self):
        result = body_sets_for_handles(range(NUM_HANDLES))
        # All body sets except Spine (no handles assigned)
        assert BodySet.Spine not in result
        assert len(result) == 9

    def test_handle_mask_length(self):
        for bs in BodySet:
            mask = handle_mask_for_body_set(bs)
            assert len(mask) == NUM_HANDLES

    def test_handle_mask_sum_matches_handle_count(self):
        for bs in BodySet:
            mask = handle_mask_for_body_set(bs)
            assert sum(mask) == len(handles_for_body_set(bs))

    def test_handle_mask_spine_all_false(self):
        mask = handle_mask_for_body_set(BodySet.Spine)
        assert not any(mask)

    def test_handle_mask_hand_l(self):
        mask = handle_mask_for_body_set(BodySet.Hand_L)
        # Indices 1-5 should be True, all others False
        for i in range(NUM_HANDLES):
            assert mask[i] == (1 <= i <= 5), f"mask[{i}] wrong"

    def test_body_set_for_handle_out_of_range(self):
        assert body_set_for_handle(23) == BodySet.Head    # 23 is the gaze handle now
        with pytest.raises(IndexError):
            body_set_for_handle(24)


# ---------------------------------------------------------------------------
# Handle names
# ---------------------------------------------------------------------------

class TestHandleNames:
    def test_unique(self):
        assert len(set(HANDLE_NAMES)) == NUM_HANDLES

    def test_known_names(self):
        assert HANDLE_NAMES[0] == "wrist_L"
        assert HANDLE_NAMES[12] == "head"
        assert HANDLE_NAMES[22] == "foot_R"
