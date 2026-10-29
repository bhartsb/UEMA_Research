# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Rotation reconstruction at the branch cut; no physics model is needed."""
import numpy as np
import pytest

from mimo_bridge.action_decoder import rotmat_log


def rotation_matrix(axis, angle):
    axis = np.array(axis, dtype=float, copy=True)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


@pytest.mark.parametrize("axis", [
    (0, 1, -1), (1, 0, -1), (1, -1, 0),
    (-1, 2, -3), (1, -2, 3), (1, 0, 0),
])
@pytest.mark.parametrize("angle", [np.pi, np.pi - 2e-7, np.pi + 2e-7])
def test_near_pi_log_reconstructs_requested_rotation(axis, angle):
    requested = rotation_matrix(axis, angle)
    vector = rotmat_log(requested)
    magnitude = np.linalg.norm(vector)
    assert 0.0 < magnitude <= np.pi + 1e-14
    # acos(trace) loses precision near pi. This bound covers double precision
    # angle extraction, while a wrong axis causes order-one matrix error.
    np.testing.assert_allclose(rotation_matrix(vector, magnitude), requested, atol=4e-8, rtol=0)
    if angle != np.pi:
        expected_axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        if angle > np.pi:
            expected_axis = -expected_axis
        np.testing.assert_allclose(vector / magnitude, expected_axis, atol=1e-12, rtol=0)


def test_exact_half_turn_with_zero_x_and_opposed_y_z():
    requested = np.array([[-1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]])
    vector = rotmat_log(requested)
    np.testing.assert_allclose(rotation_matrix(vector, np.linalg.norm(vector)), requested, atol=1e-14, rtol=0)


@pytest.mark.parametrize("angle", [0.0, 1e-5, 0.7])
def test_ordinary_rotation_log_keeps_radian_magnitude_and_axis(angle):
    axis = np.array([1.0, -2.0, 3.0])
    axis /= np.linalg.norm(axis)
    np.testing.assert_allclose(rotmat_log(rotation_matrix(axis, angle)), axis * angle, atol=1e-12, rtol=0)
