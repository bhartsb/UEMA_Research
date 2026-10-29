# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""B10 curiosity-goal and B11 anticipation-gaze: the pieces that must be right.

The differentiable gaze extraction must carry gradient to the motor output; the
anticipation target fields must appear in a live crib env exactly when a real
self-caused change exists; the curiosity modulation is (1 + drive), ordering-
preserving for positive wellness and zeroing the extremes.
"""
import numpy as np
import pytest
import torch

from lem.model.curiosity import CuriositySystem


def test_gaze_from_handle_t_carries_gradient():
    from mimo_bridge.lem_model import LemLite
    motor = torch.zeros(2, 24, 9, requires_grad=True)
    with torch.no_grad():
        motor[:, :, 6] = 1.0     # unit quaternion w-component (xyzw layout: index 3..6)
    class _Stub:
        GAZE_IDX = LemLite.GAZE_IDX if hasattr(LemLite, "GAZE_IDX") else 23
        _gaze_from_handle_t = LemLite._gaze_from_handle_t
    out = _Stub()._gaze_from_handle_t(motor)
    assert out.shape == (2, 2)
    out.sum().backward()
    assert motor.grad is not None and float(motor.grad.abs().sum()) > 0.0


def test_curiosity_goal_modulation_boosts_goldilocks_and_leaves_extremes_alone():
    """The real drive clamps to [0, 1]: goldilocks moments get up to 2x felt value
    as goals; too-easy and too-hard moments get exactly no boost (never a penalty)."""
    cu = CuriositySystem(pe_min=0.02, pe_max=0.4)
    mid = (0.02 + 0.4) / 2
    w = 1.0
    assert w * (1 + cu.compute_drive(mid)) > w * (1 + cu.compute_drive(0.39))
    assert w * (1 + cu.compute_drive(0.005)) == w       # too easy: no boost
    assert w * (1 + cu.compute_drive(2.0)) == w         # too hard: no boost
    assert w * (1 + cu.compute_drive(mid)) <= 2 * w     # bounded at 2x


def test_curiosity_goal_multiply_lands_in_the_goal_ring_for_every_body(tmp_path):
    """The multiply just above goal_ring.append is the drive's only live effect
    when wellness_mode is baseline and contact learning is off, and nothing
    exercised it: every other curiosity test does its arithmetic on a bare float.
    Drive a real model and read the ring. agency_drive and movement_drive are off,
    so the stored value is the bodily value times (1 + drive) exactly, and both
    bodies must receive the SAME multiplier."""
    from mimo_bridge.lem_model import LemLite
    from mimo_bridge.datasource import SingleEnvSource

    def _model(pe_min, pe_max):
        return LemLite(device="cpu", seed=0, imagination=False, next_frame_loss=True,
                       intention=True, intention_execute=False, curiosity_goal=True,
                       agency_drive=False, movement_drive=False,
                       overrides={"training": {"steps_per_phase": 3000,
                                               "curiosity": {"pe_min": pe_min,
                                                             "pe_max": pe_max}}})

    # an untrained model's next-frame error sits near 0.95: 0.5:2.0 puts it inside
    # the band (a nonzero drive), 0.001:0.002 puts it outside it (drive exactly 0)
    for pe_min, pe_max, boosted in ((0.5, 2.0, True), (0.001, 0.002, False)):
        m = _model(pe_min, pe_max)
        src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / f"cur{pe_min}"),
                              tag="_curgoal", vision=False, seed=5)
        try:
            f1 = src.reset()
            dev = dict(m.dev_limits(10 ** 9))
            held = prev = None
            for i in range(6):
                res = m.step(list(f1) + list(f1), dev, train=True, step_index=i,
                             executed_action=prev)          # two bodies, one world
                if i % 5 == 0 or held is None:
                    held = res.motor_for_action
                f1 = src.step(held[:1],
                              res.gaze_out[:1] if res.gaze_out is not None else None,
                              None, render=False)
                prev = held.clone() if torch.is_tensor(held) else np.array(held)
        finally:
            src.close()
        nx = list((m.last_next_pe or {}).values())
        assert nx, "no real-sense next-frame errors: the drive had nothing to read"
        expect = CuriositySystem(pe_min=pe_min, pe_max=pe_max).compute_drive(sum(nx) / len(nx))
        assert (expect > 0.0) == boosted, f"band {pe_min}:{pe_max} gave drive {expect}"
        stored, bodily = m.goal_ring[-1][1], m._goal_ring_w0[-1]
        assert stored.shape[0] == 2 and bodily.shape[0] == 2
        ratios = [float(a) / float(b) for a, b in zip(stored, bodily)]
        for r in ratios:
            assert abs(r - (1.0 + expect)) < 1e-5, (
                f"the goal ring stored {r}x the bodily value, expected {1.0 + expect}x")
        assert abs(ratios[0] - ratios[1]) < 1e-5, (
            f"the two bodies got different multipliers: {ratios}")


def test_env_emits_anticipation_target_when_change_exists(tmp_path):
    from mimo_bridge.env import BridgeEnv
    e = BridgeEnv(scene="crib", run_dir=str(tmp_path), vision=True, eye_size=256,
                  fovea_size=288, fovea_fov=24.0, render_every=2, seed=11,
                  tag="_antic", mobile=True, critters=True)
    try:
        e.reset()
        saw_ok = saw_fields = False
        for i in range(200):
            raw = e.step(None, None, None, ctrl=np.zeros(e.model.nu), render=(i % 2 == 0))
            tk = raw.info.get("task", {})
            if tk.get("antic_ok"):
                saw_ok = True
                saw_fields = ("antic_az" in tk and "antic_el" in tk
                              and abs(tk["antic_az"]) <= 0.785 + 1e-6
                              and -0.820 - 1e-6 <= tk["antic_el"] <= 0.576 + 1e-6)
                break
        assert saw_ok, "200 steps with a swinging-capable mobile and moving hands must produce a consequence"
        assert saw_fields, "target must be inside the eye's command range"
    finally:
        e.close()
