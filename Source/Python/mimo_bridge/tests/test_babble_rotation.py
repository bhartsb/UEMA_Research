# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Exploration composes rotations from unchanged quaternion components."""
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from mimo_bridge.lem_model import LemLite


def test_exploration_quaternion_composition_matches_independent_rotation_library():
    seed, ns = 731, 0.05
    actor = SimpleNamespace(noise_rng=torch.Generator().manual_seed(seed))
    reference_rng = torch.Generator().manual_seed(seed)
    command = torch.zeros(2, 4, 9)
    rotations = Rotation.from_euler("xyz", np.arange(24).reshape(8, 3) * 7, degrees=True)
    command[..., 3:7] = torch.from_numpy(rotations.as_quat().reshape(2, 4, 4))
    command[..., 8] = 0.5
    before = command.clone()
    noise = torch.randn(command.shape, generator=reference_rng) * ns
    rotation_vectors = torch.randn(2, 4, 3, generator=reference_rng) * (10 * ns)
    expected = Rotation.from_rotvec(rotation_vectors.reshape(-1, 3).numpy()) * rotations
    actual = LemLite._independent_babble(actor, command, ns, torch.zeros(2, 4, 3), 40.0, 0.7)
    observed = Rotation.from_quat(actual[..., 3:7].reshape(-1, 4).numpy())
    residual = (expected.inv() * observed).magnitude()
    assert float(residual.max()) < 3e-7
    assert torch.allclose(actual[..., 3:7].norm(dim=-1), torch.ones(2, 4), atol=2e-7)
    assert torch.equal(command, before), "The helper must not overwrite the source order."
    assert torch.equal(actor.noise_rng.get_state(), reference_rng.get_state())
    assert torch.equal(actual[..., :3], (noise[..., :3] * 100).clamp(-40, 40))
