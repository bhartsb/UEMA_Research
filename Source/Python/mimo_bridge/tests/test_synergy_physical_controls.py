# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Private synergy controls match command magnitude without inventing missing directions."""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch


def probe(monkeypatch):
    directory = Path(__file__).resolve().parents[4] / 'docs/gain_of_function_v3'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('physical_controls', directory / 'compare_synergy_physical.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def order():
    result = torch.zeros(1, 3, 9)
    result[..., 6] = 1
    result[..., 8] = 0.2
    return result


def test_size_control_matches_mean_turn_reach_strength_and_gaze(monkeypatch):
    module = probe(monkeypatch)
    candidate, baseline = order(), order()
    achieved = torch.zeros(1, 3, 10)
    achieved[..., 6] = 1
    names = ['wrist_L', 'index_L', 'gaze']
    baseline[0, :2, 3:7] = torch.from_numpy(Rotation.from_euler('z', [[5], [15]], degrees=True).as_quat())
    candidate[0, :2, 3:7] = torch.from_numpy(Rotation.from_euler('y', [[1], [3]], degrees=True).as_quat())
    baseline[0, :2, 0] = torch.tensor([20.0, 10.0])
    candidate[0, :2, 1] = torch.tensor([10.0, 5.0])
    baseline[..., 8] = 0.8
    baseline[0, 2, 3:7] = torch.from_numpy(Rotation.from_euler('z', [20], degrees=True).as_quat())
    matched, receipts = module.size_controlled_order(candidate, baseline, achieved, names, 40)
    assert receipts[0]['matched']
    angles = Rotation.from_quat(matched[0, :2, 3:7].numpy()).magnitude()
    assert abs(np.degrees(angles.mean()) - 10) < 1e-5
    assert abs(matched[0, :2, :3].norm(dim=-1).mean().item() - 15) < 1e-6
    assert torch.equal(matched[..., 8], baseline[..., 8])
    assert torch.equal(matched[:, 2], baseline[:, 2])
    # The changed movement still uses the candidate's y direction, not the baseline's z/x directions.
    assert np.allclose(Rotation.from_quat(matched[0, :2, 3:7]).as_rotvec()[:, [0, 2]], 0, atol=1e-8)
    assert torch.equal(matched[0, :2, 0], torch.zeros(2))


def test_zero_candidate_directions_cannot_claim_size_matching(monkeypatch):
    module = probe(monkeypatch)
    candidate, baseline = order(), order()
    achieved = torch.zeros(1, 3, 10)
    achieved[..., 6] = 1
    baseline[0, :2, 3:7] = torch.from_numpy(Rotation.from_euler('z', [[5], [15]], degrees=True).as_quat())
    baseline[0, :2, 0] = 10
    _, receipts = module.size_controlled_order(candidate, baseline, achieved, ['wrist_L', 'index_L', 'gaze'], 40)
    assert not receipts[0]['matched']
    assert not receipts[0]['rotation_matched'] and not receipts[0]['position_matched']
