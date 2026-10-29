# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The eyes' order row and the eyes' sense row share one convention (motor interface audit
2026-09-25, finding F1 option B; train_mimo --eye-row-convention physical).

The defect (docs/gain_of_function_v4/agents/motor_interface_audit.md, "Eyes"): the eye servo sets
the vertical eye joint to PLUS the ordered elevation, the ledger wrote the eyes' order row with
pitch = PLUS the elevation, and proprioception reports the eye's physical head-relative rotation,
whose pitch is MINUS the vertical joint angle because MIMo's vertical eye joint turns about
(0, -1, 0). The motor lesson grades the controller's eye row against the next achieved row, so a
correctly executed elevation was graded as its mirror image and only level gaze agreed with it.

Pinned here:
  * the 'legacy' convention (the default) writes and reads exactly what the code wrote before;
  * under 'physical' an order row written for the eye command (h, v) IS the row proprioception
    reads once the eye is at (h, v) - checked on the compiled crib model against
    ProprioAdapter.read, 100 random eye angles, within 1e-6 rad - and reading that row back gives
    (h, v); under 'legacy' the same comparison is off by twice the vertical angle (the defect);
  * every reader and writer of the eye row agrees within one convention: the controller's gaze
    (_gaze_from_handle and its differentiable twin), the planner's row as sent (gaze_row_as_sent),
    the efference heads' gaze features, the learner's settings and the trainer's ledger.
"""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mimo_bridge.gaze_reflex import AZ_RANGE, EL_RANGE, EYE_ROW_CONVENTIONS, eye_pitch_sign
from mimo_bridge.lem_model import LemLite

GI = LemLite.GAZE_IDX
PKG = Path(__file__).resolve().parents[1]


class _Eyes:
    """What _gaze_from_handle reads from a model: the eyes' row index and the convention."""
    GAZE_IDX = GI

    def __init__(self, convention):
        self.eye_row_convention = convention


def _qmul(a, b):
    """xyzw quaternion product, numpy, float64, broadcast over leading axes."""
    x1, y1, z1, w1 = np.moveaxis(np.asarray(a, np.float64), -1, 0)
    x2, y2, z2, w2 = np.moveaxis(np.asarray(b, np.float64), -1, 0)
    return np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                     w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], axis=-1)


def _angle(a, b):
    """Radians between two orientations given as xyzw quaternions (q and -q the same), computed
    from the vector part of a x conj(b) so it stays exact near zero."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    r = _qmul(a, b * np.array([-1.0, -1.0, -1.0, 1.0]))
    return 2.0 * np.arctan2(np.linalg.norm(r[..., :3], axis=-1), np.abs(r[..., 3]))


def _gazes(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(*AZ_RANGE, n), rng.uniform(*EL_RANGE, n)], 1).astype(np.float32)


def _orders(n, H=26, seed=1):
    m = np.random.default_rng(seed).normal(size=(n, H, 9)).astype(np.float32)
    return m


def test_the_two_conventions_are_named_and_nothing_else_is_accepted():
    assert EYE_ROW_CONVENTIONS == ("legacy", "physical")
    assert eye_pitch_sign("legacy") == 1.0 and eye_pitch_sign("physical") == -1.0
    for bad in ("Physical", "mirror", "", None):
        with pytest.raises(ValueError):
            eye_pitch_sign(bad)
    with pytest.raises(ValueError):
        LemLite.gaze_to_handle(_orders(1), _gazes(1), convention="servo")


def test_the_legacy_row_is_written_bit_for_bit_as_before():
    """The default keeps today's ledger exactly: the old formula, inlined, against the function."""
    m, g = _orders(64), _gazes(64)
    ref = np.array(m, copy=True)
    az, el = g[:, 0], g[:, 1]
    ca, sa, ce, se = np.cos(az / 2.0), np.sin(az / 2.0), np.cos(el / 2.0), np.sin(el / 2.0)
    ref[:, GI, 3], ref[:, GI, 4], ref[:, GI, 5], ref[:, GI, 6] = -sa * se, ca * se, sa * ce, ca * ce
    assert np.array_equal(LemLite.gaze_to_handle(m, g), ref)
    assert np.array_equal(LemLite.gaze_to_handle(m, g, convention="legacy"), ref)


@pytest.mark.parametrize("convention", EYE_ROW_CONVENTIONS)
def test_a_row_written_for_an_eye_command_reads_back_as_that_command(convention):
    m, g = _orders(200), _gazes(200)
    row = LemLite.gaze_to_handle(m, g, convention=convention)
    q = torch.as_tensor(row[:, GI, 3:7]).double()
    q = q / q.norm(dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)).numpy()
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0)).numpy()
    assert np.abs(yaw - g[:, 0]).max() < 1e-5
    assert np.abs(pitch - eye_pitch_sign(convention) * g[:, 1]).max() < 1e-5
    # the writer touches only the eyes' quaternion (test_transition_contract pins the legacy case)
    others = [i for i in range(m.shape[1]) if i != GI]
    assert np.array_equal(row[:, others], m[:, others])
    assert np.array_equal(row[:, GI, [0, 1, 2, 7, 8]], m[:, GI, [0, 1, 2, 7, 8]])
    # the controller's reading of the row returns the command, in the same convention
    back = LemLite._gaze_from_handle(_Eyes(convention), torch.as_tensor(row))
    assert np.abs(back - g).max() < 1e-5
    # and read in the OTHER convention it is the mirror in elevation: the defect's arithmetic
    other = "legacy" if convention == "physical" else "physical"
    mirror = LemLite._gaze_from_handle(_Eyes(other), torch.as_tensor(row))
    inside = (g[:, 1] > -EL_RANGE[1]) & (g[:, 1] < -EL_RANGE[0])      # the mirror not clamped
    assert np.abs(mirror[inside, 1] + g[inside, 1]).max() < 1e-5
    assert np.abs(mirror[:, 0] - g[:, 0]).max() < 1e-5


@pytest.mark.parametrize("convention", EYE_ROW_CONVENTIONS)
def test_the_differentiable_twin_the_row_as_sent_and_the_efference_features_agree(convention):
    """Every reader of an eye row reads the same (azimuth, elevation) in one convention, including
    rows whose angles lie beyond the eye's range (all clamp alike)."""
    from lem.model.efference import gaze_order_features
    rng = np.random.default_rng(3)
    cmd = torch.as_tensor(rng.normal(size=(300, 26, 9)).astype(np.float32))
    eyes = _Eyes(convention)
    sent = LemLite._gaze_from_handle(eyes, cmd)
    twin = LemLite._gaze_from_handle_t(eyes, cmd.clone().requires_grad_(True))
    assert np.abs(twin.detach().numpy() - sent).max() < 1e-6
    # the planner's row as sent: re-read, it is the command the servo gets; written for a command
    # inside the range, it is that command's ledger row
    as_sent = LemLite.gaze_row_as_sent(cmd, GI, convention=convention)
    assert np.abs(LemLite._gaze_from_handle(eyes, as_sent) - sent).max() < 1e-5
    ledger = LemLite.gaze_to_handle(cmd.numpy(), sent, convention=convention)
    assert _angle(as_sent[:, GI, 3:7].numpy(), ledger[:, GI, 3:7]).max() < 1e-5
    assert torch.equal(as_sent[:, :GI], cmd[:, :GI]) and torch.equal(as_sent[:, GI + 1:], cmd[:, GI + 1:])
    # the efference heads' copy of the eye command: the two angles are the ones the servo is sent
    feats = gaze_order_features(torch.as_tensor(ledger), GI, convention)
    assert np.abs(feats[:, 9:11].numpy() - sent).max() < 1e-5
    assert torch.equal(feats[:, :9], torch.as_tensor(ledger)[:, GI])


def test_the_legacy_twin_and_row_as_sent_are_unchanged():
    """The default path computes exactly what it computed before (the old code, inlined)."""
    rng = np.random.default_rng(4)
    cmd = torch.as_tensor(rng.normal(size=(50, 26, 9)).astype(np.float32))
    q = cmd[:, GI, 3:7] / cmd[:, GI, 3:7].norm(dim=-1, keepdim=True).clamp(min=1e-6)
    x, y, z, w = q.unbind(-1)
    az = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)).clamp(*AZ_RANGE)
    el = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0)).clamp(*EL_RANGE)
    ca, sa, ce, se = torch.cos(az / 2.0), torch.sin(az / 2.0), torch.cos(el / 2.0), torch.sin(el / 2.0)
    ref = cmd.clone()
    ref[:, GI, 3:7] = torch.stack([-sa * se, ca * se, sa * ce, ca * ce], dim=-1)
    assert torch.equal(LemLite.gaze_row_as_sent(cmd, GI), ref)
    assert torch.equal(LemLite._gaze_from_handle_t(_Eyes("legacy"), cmd), torch.stack([az, el], -1))

    class _Old:           # an object from before the setting existed reads as legacy
        GAZE_IDX = GI
    assert np.array_equal(LemLite._gaze_from_handle(_Old(), cmd), torch.stack([az, el], -1).numpy())


# ------------------------------------------------------------------ the compiled crib model
def test_on_the_crib_model_the_physical_order_row_is_the_row_proprioception_reads(tmp_path):
    """The audit's test: on the compiled crib model, 100 random eye joint angles (h, v) inside the
    joint ranges (and the servo's command ranges), set through qpos with mj_kinematics. With the
    eye at (h, v) - where the eye servo puts it for the command (h, v) (action_decoder.EYE_JOINTS:
    left horizontal +1, left vertical +1) - gaze_to_handle((h, v), 'physical') equals the
    ProprioAdapter.read eye row within 1e-6 rad, and _gaze_from_handle of that row returns
    (h, v) within 1e-6 rad. The legacy row is off by twice the vertical angle."""
    import mujoco
    from mimo_bridge.action_decoder import EYE_JOINTS
    from mimo_bridge.env import BridgeEnv
    from mimo_bridge.proprio_adapter import ProprioAdapter
    env = BridgeEnv(scene="crib", run_dir=str(tmp_path), vision=False, tag="_eye_row", age=24,
                    contact_profile="mechanical_v4", caregiver_seconds=0, kp=6.0,
                    body_layout="trunk_v1")
    try:
        m = env.model
        # the two facts the convention rests on, so a change to either says why this test fails
        vj = m.joint("robot:left_eye_vertical").id
        assert np.allclose(m.jnt_axis[vj], [0.0, -1.0, 0.0])
        assert np.allclose(m.jnt_axis[m.joint("robot:left_eye_horizontal").id], [0.0, 0.0, 1.0])
        assert EYE_JOINTS["robot:left_eye_vertical"] == (1, +1.0)
        assert EYE_JOINTS["robot:left_eye_horizontal"] == (0, +1.0)
        d = mujoco.MjData(m)
        adapter = ProprioAdapter(m, d, body_layout="trunk_v1")
        gi = adapter.handle_names.index("gaze")
        assert gi == GI
        adr = {n: int(m.jnt_qposadr[m.joint("robot:" + n).id]) for n in (
            "left_eye_horizontal", "left_eye_vertical", "right_eye_horizontal", "right_eye_vertical")}
        rng_h = [max(AZ_RANGE[0], float(m.jnt_range[m.joint("robot:left_eye_horizontal").id][0])),
                 min(AZ_RANGE[1], float(m.jnt_range[m.joint("robot:left_eye_horizontal").id][1]))]
        rng_v = [max(EL_RANGE[0], float(m.jnt_range[vj][0])), min(EL_RANGE[1], float(m.jnt_range[vj][1]))]
        rng = np.random.default_rng(2026)
        motor = np.zeros((1, len(adapter.handle_names), 9), np.float32)
        motor[..., 6] = 1.0
        worst_physical = worst_back = 0.0
        legacy_gap = []
        for _ in range(100):
            h, v = rng.uniform(*rng_h), rng.uniform(*rng_v)
            d.qpos[:] = m.qpos0
            # both eyes as the servo drives them for the command (h, v): the right horizontal hinge's
            # axis is (0, 0, -1), so the decoder sends it -h (EYE_JOINTS)
            d.qpos[adr["left_eye_horizontal"]], d.qpos[adr["left_eye_vertical"]] = h, v
            d.qpos[adr["right_eye_horizontal"]], d.qpos[adr["right_eye_vertical"]] = -h, v
            mujoco.mj_kinematics(m, d)
            sensed = adapter.read(0.01).bones[gi, 0:4]
            g = np.array([[h, v]], np.float32)
            physical = LemLite.gaze_to_handle(motor, g, convention="physical")[0, GI, 3:7]
            legacy = LemLite.gaze_to_handle(motor, g, convention="legacy")[0, GI, 3:7]
            worst_physical = max(worst_physical, float(_angle(physical, sensed)))
            row = motor.copy()
            row[0, GI, 3:7] = sensed
            back = LemLite._gaze_from_handle(_Eyes("physical"), torch.as_tensor(row))[0]
            worst_back = max(worst_back, float(np.abs(back - np.array([h, v])).max()))
            legacy_gap.append((float(_angle(legacy, sensed)), 2.0 * abs(v)))
        assert worst_physical < 1e-6, f"the physical order row misses the sensed eye row by {worst_physical} rad"
        assert worst_back < 1e-6, f"the sensed row reads back {worst_back} rad from the eye angles"
        gap = np.asarray(legacy_gap)
        assert np.abs(gap[:, 0] - gap[:, 1]).max() < 1e-5, "legacy: off by twice the vertical angle"
        assert np.median(gap[:, 0]) > np.radians(5.0), "the defect this test exists for is visible"
    finally:
        env.close()


# ------------------------------------------------------------------ the learner and the trainer
def _lite(**kw):
    torch.set_num_threads(1)
    return LemLite(device="cpu", seed=0, imagination=False, no_vision=True, body_layout="trunk_v1", **kw)


def test_the_learner_carries_its_convention_to_every_reader():
    legacy = _lite()
    assert legacy.eye_row_convention == "legacy" and "eye_row_convention" not in legacy.receipt
    physical = _lite(eye_row_convention="physical", next_frame_loss=True, efference_heads=True)
    assert physical.receipt["eye_row_convention"] == "physical"
    assert physical.efference.eye_row_convention == "physical", "the efference gaze features"
    # a public name: the episode companion (every _* and last_* attribute) cannot carry it back
    from mimo_bridge import continuation_state as CS
    assert "eye_row_convention" not in CS.capture_learner(physical, include_replay=False)["data"]
    # the controller's gaze, and the predicted order the causal chain writes, follow the model
    out = torch.zeros(1, 26, 9)
    out[..., 6] = 1.0
    out[0, GI, 3:7] = torch.as_tensor(LemLite.gaze_to_handle(out.numpy(), [[0.1, 0.3]], "physical")[0, GI, 3:7])
    assert np.allclose(physical._gaze_from_handle(out), [[0.1, 0.3]], atol=1e-5)
    assert np.allclose(legacy._gaze_from_handle(out), [[0.1, -0.3]], atol=1e-5)
    with pytest.raises(ValueError):
        _lite(eye_row_convention="servo")


def test_the_planner_decision_carries_the_convention_into_the_row_as_sent():
    from types import SimpleNamespace
    p = torch.nn.Parameter(torch.ones(1))
    stub = SimpleNamespace(last_context=torch.ones(1, 3, 4), context_buffer=torch.zeros(1, 3, 4),
                           _prev_mfa=None, last_achieved=torch.zeros(1, 26, 10),
                           _last_source_mask={}, _last_dev={'rom_limit': 1.}, smooth_actions=.35,
                           _hand_previous_joints=None, _hand_current_angles=None,
                           _hand_current_velocities=None,
                           params=[p], device=torch.device('cpu'), GAZE_IDX=GI,
                           eye_row_convention="physical", executed_fields_only_on=False)
    stub._plan_parameter_versions = lambda: LemLite._plan_parameter_versions(stub)
    stub._plan_settings = lambda: None
    d = LemLite._capture_plan_state(stub)
    assert d["eye_row_convention"] == "physical" and d["executed_fields_only"] is False
    # the decision's limits carry them too (a copy: the live limits are untouched), so a probe that
    # passes only decision['dev'] forms the order the live step forms
    assert d["dev"]["eye_row_convention"] == "physical" and d["dev"]["executed_fields_only"] is False
    assert "eye_row_convention" not in stub._last_dev
    raw = torch.zeros(1, 26, 9)
    raw[..., 6] = 1.0
    raw[..., 8] = .5
    raw[0, GI, 3:7] = torch.as_tensor(LemLite.gaze_to_handle(raw.numpy(), [[0.2, -0.4]], "physical")[0, GI, 3:7])
    for kw in (dict(eye_row_convention=d['eye_row_convention']), {}):
        dev = dict(d['dev'], strength_limit=1.) if not kw else {'rom_limit': 1., 'strength_limit': 1.}
        planned = LemLite.planned_execution(raw, dev, torch.zeros(1, 26, 10), previous=None,
                                            smoothing=.35, keep_gaze_idx=d['smooth_keep_gaze'], **kw)
        assert np.allclose(LemLite._gaze_from_handle(_Eyes("physical"), planned), [[0.2, -0.4]], atol=1e-5)
    # limits without the settings read as the old defaults: the same row read as legacy
    legacy = LemLite.planned_execution(raw, {'rom_limit': 1., 'strength_limit': 1.}, torch.zeros(1, 26, 10),
                                       keep_gaze_idx=GI)
    assert np.allclose(LemLite._gaze_from_handle(_Eyes("legacy"), legacy), [[0.2, 0.4]], atol=1e-5)
    # the old stub without the settings reads as the old defaults
    old = SimpleNamespace(**{k: v for k, v in vars(stub).items()
                             if k not in ("eye_row_convention", "executed_fields_only_on")})
    d0 = LemLite._capture_plan_state(old)
    assert d0["eye_row_convention"] == "legacy" and d0["executed_fields_only"] is False


def test_the_trainer_flag_defaults_to_legacy_and_the_ledger_uses_the_learners_convention():
    from mimo_bridge.train_mimo import acting_configuration, parse_args
    a = parse_args(["--out", "unused"])
    assert a.eye_row_convention == "legacy"
    assert parse_args(["--out", "unused", "--eye-row-convention", "physical"]).eye_row_convention == "physical"
    with pytest.raises(SystemExit):
        parse_args(["--out", "unused", "--eye-row-convention", "mirror"])
    # it changes what the body is sent and learns from: part of the acting configuration
    assert acting_configuration(a)["eye_row_convention"] == "legacy"
    import re
    src = (PKG / "train_mimo.py").read_text(encoding="utf-8")
    assert "eye_row_convention=getattr(args, 'eye_row_convention', 'legacy')" in src
    # the ledger: the eye command actually issued, written in the learner's own convention, on the order
    # as sent or, with the motor interface's limb fixes on, the order the worlds recorded
    assert re.search(r"prev_executed = LemLite\.gaze_to_handle\(send_motor if limb_ledger is None else limb_ledger,"
                     r"\s*gaze_seq\.mean\(axis=0\),\s*convention=model\.eye_row_convention\)", src)
    assert len(re.findall(r"prev_executed = LemLite\.gaze_to_handle\(", src)) == 1
    # the resume check only warns (a deploy switches it), and it is not a continuation flag
    cont = src[src.index("continuation_flags = ("):src.index("continuation_config = {")]
    assert "eye_row_convention" not in cont
