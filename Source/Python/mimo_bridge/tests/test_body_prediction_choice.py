# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Units and geometry of the private body-prediction inverse-control pilot."""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from types import SimpleNamespace


def probe(monkeypatch):
    folder = Path(__file__).resolve().parents[4] / 'docs/gain_of_function_v3'
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location('body_choice', folder / 'probe_body_prediction_choice.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trust_region_and_fixed_columns(monkeypatch):
    module = probe(monkeypatch)
    base = torch.zeros(1, 3, 9)
    base[..., 6] = 1
    base[..., 7:] = 0.4
    base[0, 0, 3:7] = torch.from_numpy(Rotation.from_euler('x', [30], degrees=True).as_quat())
    offsets = torch.full((1, 3, 6), 10.0)
    command = module.bounded_command(base, offsets, [0, 1], [0])
    assert torch.equal(command[:, 2], base[:, 2])
    assert torch.equal(command[..., 7:], base[..., 7:])
    assert torch.equal(command[:, 1, :3], base[:, 1, :3])
    np.testing.assert_allclose((command[0, 0, :3] - base[0, 0, :3]).numpy() * 200, 1.8)
    relative = Rotation.from_quat(command[0, :2, 3:7]) * Rotation.from_quat(base[0, :2, 3:7]).inv()
    np.testing.assert_allclose(np.degrees(relative.magnitude()), 11, atol=1e-5)


def test_world_pose_cost_uses_centimetres_and_rotation_composition(monkeypatch):
    module = probe(monkeypatch)
    position = torch.tensor([[[1.8, 0, 0], [0, 0, 0]]])
    quaternion = torch.from_numpy(Rotation.from_euler('xyz', [[30, 20, 10], [0, 0, 0]], degrees=True).as_quat()).float()[None]
    target_quat = Rotation.from_euler('z', [11], degrees=True) * Rotation.from_quat(quaternion[0, 0])
    target = dict(pos_cm=torch.zeros_like(position), quat=quaternion.clone())
    target['quat'][0, 0] = torch.from_numpy(target_quat.as_quat())
    cost, units = module.pose_score(position, quaternion, target, [0])
    assert abs(float(cost) - 2 / 3) < 1e-6
    assert abs(units['position_error_cm'] - 1.8) < 1e-6
    assert abs(units['rotation_error_deg'] - 11) < 1e-5


def test_oracle_diagnostic_removes_state_bias_without_replacing_action_effect(monkeypatch):
    module = probe(monkeypatch)
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    parameter.grad = torch.tensor(3.0)
    def body_prediction(latent, order, achieved):
        # True motion is the requested displacement. The learned predictor has
        # the correct action slope but adds a 10 cm state-forecast bias.
        position = (order[..., :3] * 200 + torch.tensor([10.0, 0.0, 0.0])) / 1.8
        zeros = torch.zeros_like(position)
        return torch.cat([zeros, zeros, position, zeros], -1)
    model = SimpleNamespace(handle_names=['wrist_L', 'gaze'], GAZE_IDX=1,
        params=[parameter], efference=SimpleNamespace(body=body_prediction),
        _latent_for_order=lambda state, encoded: encoded)
    base = torch.zeros(1, 2, 9)
    base[..., 6] = 1
    base[..., 8] = 0.4
    achieved = torch.zeros(1, 2, 10)
    achieved[..., 6] = 1
    state = SimpleNamespace(achieved=achieved)
    decision = dict(achieved=achieved, dev=dict(rom_limit=1, strength_limit=1),
                    smooth_prev=None, smoothing=0, smooth_keep_gaze=1)
    target = dict(pos_cm=torch.zeros(1, 2, 3), quat=base[..., 3:7].clone())
    target['pos_cm'][0, 0, 0] = 1.8
    measured = dict(pos_cm=torch.zeros(1, 2, 3), quat=base[..., 3:7].clone())
    raw, _ = module.choose_with_body_prediction(model, state, decision, base, target, 30)
    centred, metadata = module.choose_with_body_prediction(model, state, decision, base, target, 30,
                                                           measured_baseline=measured)
    assert float(raw[0, 0, 0]) < 0  # The biased absolute forecast steers away.
    assert abs(float(centred[0, 0, 0]) * 200 - 1.8) < 1e-5
    assert metadata['oracle_baseline_centred'] and metadata['selected_predicted_loss'] < 1e-10
    assert parameter.requires_grad and parameter.item() == 2 and parameter.grad.item() == 3
    past=dict(position_cm=torch.tensor([[[-10.0,0,0],[-10.0,0,0]]]),
              rotation_deg=torch.zeros(1,2,3))
    corrected, metadata = module.choose_with_body_prediction(model,state,decision,base,target,30,
                                                             past_residual=past)
    assert abs(float(corrected[0,0,0])*200-1.8)<1e-5
    assert metadata['completed_past_order_correction'] and not metadata['oracle_baseline_centred']
