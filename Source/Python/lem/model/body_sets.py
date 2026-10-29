# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Handle-to-body-set mapping -- Python mirror of C++ EBodySet.

Authoritative source: MotorCortexComponent.h (EBodySet enum, lines 28-41)
and MotorCortexComponent.cpp (HandleToBodySet map, lines 1267-1302).

This module is pure data with zero model dependencies.  Any component that
needs body-set information imports from here.

SHM body-set order:  The tokenizer's motor capability vector
(``tokenizer/modalities/motor.py``) stores capacity at ``capability[i]``
and accumulated energy at ``capability[10 + i]`` where ``i`` is the
``BodySet`` ordinal.  This matches the C++ pipeline buffer write order
(``MotorCortexComponent.cpp`` BodySetOrder array) and the EBodySet enum.

C++ handle names carry a ``_PH_`` infix (e.g. ``Wrist_PH_L``).  Python
uses short names (``wrist_L``) matching ``live_test.py`` convention.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Iterable


# ---------------------------------------------------------------------------
# EBodySet enum  (must match C++ MotorCortexComponent.h:28-41 exactly)
# ---------------------------------------------------------------------------

class BodySet(IntEnum):
    Arm_L  = 0
    Arm_R  = 1
    Hand_L = 2
    Hand_R = 3
    Spine  = 4   # No PH handles -- posture managed by PCC, not physics handles
    Head   = 5
    Leg_L  = 6
    Leg_R  = 7
    Foot_L = 8
    Foot_R = 9


NUM_BODY_SETS: int = 10
NUM_HANDLES: int = 24


# ---------------------------------------------------------------------------
# Handle names  (index order matches C++ BuildHandleDefinitions / TrackedBoneNames)
# ---------------------------------------------------------------------------

HANDLE_NAMES: tuple[str, ...] = (
    # 0-5: left wrist + fingers
    "wrist_L", "thumb_L", "index_L", "middle_L", "ring_L", "pinky_L",
    # 6-11: right wrist + fingers
    "wrist_R", "thumb_R", "index_R", "middle_R", "ring_R", "pinky_R",
    # 12: head
    "head",
    # 13-14: left arm limbs
    "upperarm_L", "lowerarm_L",
    # 15-16: right arm limbs
    "upperarm_R", "lowerarm_R",
    # 17-18: left leg
    "thigh_L", "calf_L",
    # 19-20: right leg
    "thigh_R", "calf_R",
    # 21-22: feet
    "foot_L", "foot_R",
    # 23: the eyes. A real body part, not a side channel: felt in proprioception
    # (tracked bone = left_eye body), commanded like any handle (its orientation is
    # the gaze direction), graded by the same motor loss. The C++ side
    # (BuildHandleDefinitions / TrackedBoneNames) must add Gaze before the next UE run;
    # the wire format already has room (PROPRIO_MAX_BONES = MOTOR_MAX_HANDLES = 24).
    "gaze",
)


# ---------------------------------------------------------------------------
# Handle -> BodySet  (from C++ HandleToBodySet, MotorCortexComponent.cpp:1267-1302)
# ---------------------------------------------------------------------------

HANDLE_BODY_SET: tuple[BodySet, ...] = (
    # 0  wrist_L     1  thumb_L     2  index_L     3  middle_L    4  ring_L      5  pinky_L
    BodySet.Arm_L,  BodySet.Hand_L, BodySet.Hand_L, BodySet.Hand_L, BodySet.Hand_L, BodySet.Hand_L,
    # 6  wrist_R     7  thumb_R     8  index_R     9  middle_R   10  ring_R     11  pinky_R
    BodySet.Arm_R,  BodySet.Hand_R, BodySet.Hand_R, BodySet.Hand_R, BodySet.Hand_R, BodySet.Hand_R,
    # 12 head
    BodySet.Head,
    # 13 upperarm_L  14 lowerarm_L
    BodySet.Arm_L,  BodySet.Arm_L,
    # 15 upperarm_R  16 lowerarm_R
    BodySet.Arm_R,  BodySet.Arm_R,
    # 17 thigh_L     18 calf_L
    BodySet.Leg_L,  BodySet.Leg_L,
    # 19 thigh_R     20 calf_R
    BodySet.Leg_R,  BodySet.Leg_R,
    # 21 foot_L      22 foot_R
    BodySet.Foot_L, BodySet.Foot_R,
    # 23 gaze (the eyes live in the head)
    BodySet.Head,
)


# ---------------------------------------------------------------------------
# BodySet -> Handle indices  (reverse mapping, built from HANDLE_BODY_SET)
# ---------------------------------------------------------------------------

BODY_SET_HANDLES: dict[BodySet, tuple[int, ...]] = {
    bs: tuple(i for i, h in enumerate(HANDLE_BODY_SET) if h == bs)
    for bs in BodySet
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def body_set_for_handle(index: int) -> BodySet:
    """Return the body set for a handle index (0-22)."""
    return HANDLE_BODY_SET[index]


def handles_for_body_set(bs: BodySet) -> tuple[int, ...]:
    """Return handle indices belonging to a body set (may be empty for Spine)."""
    return BODY_SET_HANDLES[bs]


def body_sets_for_handles(indices: Iterable[int]) -> set[BodySet]:
    """Return the unique body sets covering the given handle indices."""
    return {HANDLE_BODY_SET[i] for i in indices}


def handle_mask_for_body_set(bs: BodySet) -> list[bool]:
    """Return a 23-element bool mask (True for handles in *bs*).

    Compatible with curriculum's ``handle_mask`` format.
    """
    members = BODY_SET_HANDLES[bs]
    return [i in members for i in range(NUM_HANDLES)]
