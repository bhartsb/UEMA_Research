# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Movement-hunger drive: debt dynamics, asymmetry, gain floors - hand-checked."""
import torch

from mimo_bridge.movement_drive import MovementDrive


def _still(B=2):
    p = torch.zeros(B, 23, 12)
    return p


def _moving(B=2, cms=20.0):
    p = torch.zeros(B, 23, 12)
    v = torch.randn(B, 23, 3)
    v = v / v.norm(dim=-1, keepdim=True) * (cms / 500.0)   # tokenizer units (range 500 cm/s)
    p[:, :, 7:10] = v
    return p


def test_stillness_debt_grows_then_movement_clears_it():
    d = MovementDrive(2, "cpu", dt=1.0, debt_rise_s=10.0, debt_fall_s=2.0)
    for _ in range(10):
        out = d.update(_still())
    assert float(out["debt"].mean()) > 0.9, "10 s of stillness at rise=10 s should max the debt"
    w_starved = float(out["wellness"].mean())
    assert w_starved < 0.15, "full debt must punish hard (superlinear)"
    for _ in range(4):
        out = d.update(_moving())
    assert float(out["debt"].mean()) < 0.2, "movement clears debt on the fast time constant"


def test_asymmetry_mild_bonus_vs_severe_debt():
    d = MovementDrive(1, "cpu", dt=1.0)
    w_move = float(d.update(_moving(1))["wellness"].mean())
    assert 0.5 < w_move < 0.9, "movement bonus is mild and bounded"
    d2 = MovementDrive(1, "cpu", dt=1.0, debt_rise_s=5.0)
    for _ in range(2):
        w_early = float(d2.update(_still(1))["wellness"].mean())
    assert 0.35 < w_early <= 0.5, "brief stillness is only mildly negative"


def test_gain_floor_and_self_factor():
    d = MovementDrive(1, "cpu", dt=1.0, debt_rise_s=3.0)
    for _ in range(10):
        out = d.update(_still(1))
    assert float(out["gain"].min()) >= d.gain_floor, "numbness never reaches sensor death"
    d3 = MovementDrive(1, "cpu", dt=1.0)
    w_self = float(d3.update(_moving(1), self_factor=1.0)["wellness"].mean())
    d4 = MovementDrive(1, "cpu", dt=1.0)
    w_passive = float(d4.update(_moving(1), self_factor=0.0)["wellness"].mean())
    assert w_self > w_passive, "being moved must feel less good than moving yourself"


def test_lr_and_noise_bounds():
    d = MovementDrive(1, "cpu", dt=1.0, debt_rise_s=3.0)
    for _ in range(10):
        out = d.update(_still(1))
    assert 0.85 <= out["lr_mult"] <= 1.10
    assert out["noise_mult"] > 1.4, "high debt means real restlessness"
