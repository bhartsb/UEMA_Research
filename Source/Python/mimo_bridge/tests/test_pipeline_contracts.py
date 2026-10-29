# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Pipeline contracts: units, layout and alignment at every hand-off from MuJoCo to the model.

Two defects found on 2026-08-27, both invisible to every test that existed:

  1. The motor row the encoder sees is [achieved (10), executed command (9)]. The achieved
     half comes off the tokenizer wire with positions / 200 cm. The executed command
     carried positions in raw centimetres. Concatenated unscaled, 99.6% of the row's
     input energy was the centimetre block; rotation 0.1%, strength 0.0%. The motor token
     encoded where the handles were, not what they were told to do: two completely
     different orders moved it by 1.6 while zeroing the block moved it by 14.6. Every
     causal test since the ledger existed (H28 on) was resolving pose, not action.

  2. LemLite.dev_limits() takes a STEP. The causal test passed a phase index 0-3, which
     is step 0-3 of the defaults table - the neonate row, rom 0.05, strength 0.02. Every
     causal test ever run drove the body at 2% strength regardless of what the checkpoint
     had trained under (mature: rom 1.0, strength 1.0).

Both slipped because every existing test checked that a value ARRIVED (shape, presence,
the ledger fills) and none checked what it MEANT next to its neighbour (scale, unit,
argument semantics). The tests here check meaning.
"""
import inspect
import json
import math
import os
import re

import numpy as np
import pytest
import torch

from lem.model.body_sets import HANDLE_NAMES
from lem.model.encoder_registry import (COMMAND_POSITION_RANGE_CM as LIB_RANGE,
                                        build_motor_input, normalize_executed_command)
from lem.live_test import apply_developmental_constraints
from mimo_bridge import lem_model as LM
from mimo_bridge.lem_model import (LemLite, MOTOR_ACHIEVED_F, MOTOR_COMMAND_F, MOTOR_INPUT_F,
                                   COMMAND_POSITION_RANGE_CM, command_for_encoder,
                                   TOUCH_TOKENS, TOUCH_TOKEN_F)
from mimo_bridge.datasource import motor_to_echo, N_HANDLES
from mimo_bridge.replay import ReplayMemory
from mimo_bridge.tools import causal_test as CT
from tokenizer.config import TokenizerConfig, MOTOR_MAX_HANDLES


@pytest.fixture(scope="module")
def model():
    return LemLite(device="cpu", seed=0, intention=False, imagination=False)


def _achieved_row(n=24, seed=0):
    """A handle-state row as the tokenizer delivers it: pos/200, unit quat, radius,
    strength, is_active."""
    g = torch.Generator().manual_seed(seed)
    a = torch.zeros(1, n, MOTOR_ACHIEVED_F)
    a[..., 0:3] = torch.randn(1, n, 3, generator=g) * 0.07          # +-14 cm / 200
    q = torch.randn(1, n, 4, generator=g); a[..., 3:7] = q / q.norm(dim=-1, keepdim=True)
    a[..., 7] = 0.5; a[..., 8] = 0.5; a[..., 9] = 1.0
    return a


def _executed_row(achieved, seed=1):
    """An executed order as motor_for_action produces it: position in CENTIMETRES near
    the achieved position, unit quaternion, radius, strength."""
    g = torch.Generator().manual_seed(seed)
    n = achieved.shape[1]
    c = torch.zeros(1, n, MOTOR_COMMAND_F)
    c[..., 0:3] = achieved[..., 0:3] * 200.0 + torch.randn(1, n, 3, generator=g) * 4.0
    q = torch.randn(1, n, 4, generator=g); c[..., 3:7] = q / q.norm(dim=-1, keepdim=True)
    c[..., 7] = 0.5; c[..., 8] = 0.54
    return c


# =========================================================== 1. the motor row's units
def test_the_command_position_range_is_the_tokenizers():
    assert COMMAND_POSITION_RANGE_CM == LIB_RANGE == TokenizerConfig().position_range_cm == 200.0


def test_command_for_encoder_divides_position_only_and_keeps_the_gradient():
    cmd = _executed_row(_achieved_row()).clone().requires_grad_(True)
    out = command_for_encoder(cmd)
    torch.testing.assert_close(out[..., 0:3], cmd[..., 0:3] / 200.0)
    torch.testing.assert_close(out[..., 3:], cmd[..., 3:])
    out.sum().backward()
    assert cmd.grad is not None and float(cmd.grad[..., 0:3].abs().mean()) == pytest.approx(1 / 200.0)
    assert float(cmd.grad[..., 3:].abs().mean()) == pytest.approx(1.0)


def test_the_two_halves_of_a_motor_row_are_on_the_same_scale():
    """The defect, stated as a number: with raw cm the position block owns the row."""
    ach = _achieved_row(); cmd = _executed_row(ach)
    def share(row):
        e = row.pow(2).mean(dim=(0, 1))
        return float(e[MOTOR_ACHIEVED_F:MOTOR_ACHIEVED_F + 3].sum() / e.sum())
    raw = torch.cat([ach, cmd], dim=-1)
    fixed = torch.cat([ach, command_for_encoder(cmd)], dim=-1)
    assert share(raw) > 0.95, "the raw-cm row should reproduce the defect (99.6% measured)"
    assert share(fixed) < 0.10, (
        f"command position still owns {share(fixed):.1%} of the row after normalisation")
    # and no block is more than ~10x any other in RMS once fixed
    blocks = {"ach pos": fixed[..., 0:3], "ach quat": fixed[..., 3:7], "cmd pos": fixed[..., 10:13],
              "cmd quat": fixed[..., 13:17], "cmd str": fixed[..., 17:19]}
    rms = {k: float(v.pow(2).mean().sqrt()) for k, v in blocks.items()}
    assert max(rms.values()) / min(rms.values()) < 15, rms


def test_every_command_entering_the_encoder_goes_through_the_helper():
    """Source-level: a concatenation onto the achieved half without the helper is the bug."""
    lines = inspect.getsource(LM).splitlines()
    # every slice of the achieved half that is about to meet a command: the three live
    # sites (back-fill, delayed lesson, replay rebuild). A concatenation can span lines,
    # so the helper is looked for in a short window around the slice, not on its line.
    sites = [i for i, ln in enumerate(lines)
             if ":MOTOR_ACHIEVED_F]" in ln and "def " not in ln]
    assert len(sites) >= 3, [lines[i] for i in sites]
    for i in sites:
        window = "\n".join(lines[max(0, i - 1):i + 3])
        assert "command_for_encoder(" in window, f"a command enters the encoder unscaled:\n{window}"
    assert "command_for_encoder(_exec_form)" in inspect.getsource(LM.LemLite._plan_chain)
    assert "command_for_encoder(" in inspect.getsource(CT._predict)


def test_bridge_and_library_build_the_same_motor_row(model):
    """The bridge's row and the library's canonical layout must be one layout."""
    ach = _achieved_row(); cmd = _executed_row(ach)
    reg = model.encoder_registry
    bridge_row = torch.cat([ach, command_for_encoder(cmd)], dim=-1)
    lib_row = build_motor_input(reg, normalize_executed_command(cmd), achieved=ach)
    torch.testing.assert_close(bridge_row, lib_row)
    assert bridge_row.shape[-1] == MOTOR_INPUT_F == 19


def test_the_ledger_keeps_execution_units_and_the_encoder_converts():
    """The ring stores what the body got (cm, float16); units change only at the encoder."""
    mem = ReplayMemory(["vision", "proprioception"], d_model=4, n_handles=24, size_per_env=8, n_envs=1)
    dev = {"rom_limit": 1.0, "strength_limit": 1.0}
    ach = _achieved_row(); cmd = _executed_row(ach)
    toks = {m: torch.zeros(1, 1, 4) for m in mem.mods}
    mem.add(toks, ach, dev)
    mem.add(toks, ach, dev, executed=cmd)
    stored = mem.executed[0, 0].float()
    torch.testing.assert_close(stored, cmd[0], rtol=0, atol=0.05)   # float16 at +-200 cm: 0.1 cm steps
    assert float(stored[:, 0:3].abs().mean()) > 1.0, "the ledger should hold centimetres"
    torch.testing.assert_close(command_for_encoder(stored)[:, 0:3], stored[:, 0:3] / 200.0)


def test_live_push_backfill_and_replay_agree_on_a_context_slot(model):
    """Three writers of the same context slot: the live push (ctx_update), the live
    back-fill and the replay rebuild. All must produce token + modality embedding."""
    fm = getattr(model.forward_model, "_orig_mod", model.forward_model)
    fm._refresh_embedding_cache()
    emb = fm._mod_emb_expanded.reshape(-1, model.d_model)
    mi = model.active.index("motor")
    ach = _achieved_row(); cmd = _executed_row(ach)
    row = torch.cat([ach, command_for_encoder(cmd)], dim=-1)
    with torch.no_grad():
        z = model.encoder_registry.encode({"motor": row}, {"motor": "real"})["motor"].reshape(1, -1)
        backfill = z + emb[mi]
        tokens = {m: torch.randn(1, 1, model.d_model) for m in model.active}
        tokens["motor"] = z.reshape(1, 1, -1)
        ctx = torch.zeros(1, (model.ctx_len - 1) * model.n_mod, model.d_model)
        _l, _last, ctx_update = model.forward_model(tokens, ctx)
    torch.testing.assert_close(ctx_update[0, mi], backfill[0], rtol=1e-5, atol=1e-5)


# =========================================================== 2. the causal test's regime
def test_dev_limits_takes_a_step_not_a_phase_index(model):
    """The trap: dev_limits(3) is STEP 3, the neonate row."""
    assert model.dev_limits(3)["phase_name"] == "neonate"
    assert model.dev_limits(3)["strength_limit"] == pytest.approx(0.02)
    assert model.dev_limits(10 ** 9)["strength_limit"] == pytest.approx(1.0)


def test_causal_test_collects_under_limits_not_a_phase_index():
    sig = inspect.signature(CT._collect)
    assert "limits" in sig.parameters and "dev_phase" not in sig.parameters
    assert "dev_limits(dev_phase" not in inspect.getsource(CT)


def test_limits_come_from_the_runs_own_log(tmp_path, model):
    run = tmp_path / "H99"; ck = run / "checkpoints" / "ckpt_step_123456"; ck.mkdir(parents=True)
    log = run / "replay_training_stats.jsonl"
    log.write_text(json.dumps({"type": "training", "x": 1}) + "\n" +
                   json.dumps({"type": "mimo", "step": 5, "rom_limit": 0.2, "strength_limit": 0.1, "phase": "early_infant"}) + "\n" +
                   json.dumps({"type": "mimo", "step": 9, "rom_limit": 1.0, "strength_limit": 1.0, "phase": "mature"}) + "\n",
                   encoding="utf-8")
    limits, src = CT._limits_for_run(str(ck), model)
    assert limits["rom_limit"] == 1.0 and limits["strength_limit"] == 1.0 and limits["phase_name"] == "mature"
    assert "run log" in src
    # no log: the defaults table AT THE CHECKPOINT'S STEP, never step 0-3
    log.unlink()
    limits2, src2 = CT._limits_for_run(str(ck), model)
    assert limits2 == model.dev_limits(123456) and "defaults" in src2
    assert limits2["strength_limit"] > 0.02


def test_null_action_is_a_valid_do_nothing_in_encoder_units():
    a = CT._null_action(torch.randn(1, 24, MOTOR_COMMAND_F))
    assert float(a[..., 6].min()) == 1.0 and float(a[..., :6].abs().sum()) == 0.0 and float(a[..., 7:].abs().sum()) == 0.0
    torch.testing.assert_close(command_for_encoder(a), a)


# =========================================================== 3. units along the wire
def test_handle_count_is_24_with_gaze_last_everywhere():
    assert len(HANDLE_NAMES) == N_HANDLES == 24 and HANDLE_NAMES[-1] == "gaze"
    assert MOTOR_MAX_HANDLES >= 24
    for fn in (LM.StepResult.__doc__ or "",):
        pass
    src = inspect.getsource(LM)
    # The StepResult rows must NOT hard-code a handle count. The default layout has 24
    # and the trunk layout this project runs has 26, so a literal "[B, 24, 9]" was wrong
    # for the live body (2026-09-15); the shape is written symbolically instead.
    assert "[B, n_handles, 9]" in src, "StepResult rows must state the shape symbolically"
    for literal in ("[B, 23, 9]", "[B, 24, 9]", "[B, 26, 9]"):
        assert literal not in src, f"StepResult must not hard-code a handle count: {literal}"


def test_achieved_row_is_physics_pose_in_cm_plus_commanded_radius_strength():
    rng = np.random.default_rng(0)
    bones = np.zeros((24, 11), np.float32)
    bones[:, 0:4] = rng.normal(size=(24, 4)); bones[:, 0:4] /= np.linalg.norm(bones[:, 0:4], axis=1, keepdims=True)
    bones[:, 4:7] = rng.uniform(-30, 30, size=(24, 3))                      # cm
    motor9 = rng.uniform(0, 1, size=(24, 9)).astype(np.float32)
    echo = motor_to_echo(motor9, None, bones=bones)
    assert echo.shape == (24, 10)
    np.testing.assert_allclose(echo[:, 0:3], bones[:, 4:7])                  # achieved pos, cm
    np.testing.assert_allclose(echo[:, 3:7], bones[:, 0:4])                  # achieved quat xyzw
    np.testing.assert_allclose(echo[:, 7], motor9[:, 7])                     # commanded radius
    np.testing.assert_allclose(echo[:, 8], motor9[:, 8])                     # commanded strength
    assert np.all(echo[:, 9] == 1.0)


def test_the_wire_divides_handle_positions_by_200_and_the_encoder_gets_that():
    """cm on the wire -> /200 in the tokenized achieved row. The command half must match."""
    from mimo_bridge import frame_encoder as FE
    from mimo_bridge.touch_adapter import bone_hash
    rng = np.random.default_rng(1)
    bones = np.zeros((24, 11), np.float32); bones[:, 3] = 1.0
    bones[:, 4:7] = rng.uniform(-40, 40, size=(24, 3))
    hashes = [bone_hash(n) for n in HANDLE_NAMES]
    handles = motor_to_echo(rng.uniform(0, 1, size=(24, 9)).astype(np.float32), None, bones=bones)
    energy = np.array([1.0, 2.0, 3.0], np.float32)
    body_sets = np.column_stack([rng.uniform(0, 1, 10), rng.uniform(0, 100, 10)]).astype(np.float32)
    patches = np.zeros((53, 10), np.float32); patches[:, 8:10] = 1.0
    buf = FE.encode_frame(frame_number=1, timestamp=0.01, dt=0.01, wall_clock=0.0,
                          gaze_az_deg=0.0, gaze_el_deg=0.0, bones=bones, bone_hashes=hashes,
                          handles=handles, energy=energy, body_sets=body_sets, patches=patches,
                          collisions=[], vision_valid=False)
    tf = FE.decode_with_model_tokenizer(bytes(buf))
    mh = tf.motor_handles.float().numpy() if torch.is_tensor(tf.motor_handles) else np.asarray(tf.motor_handles)
    np.testing.assert_allclose(mh[:24, 0:3], handles[:, 0:3] / 200.0, atol=1e-6)
    np.testing.assert_allclose(mh[:24, 7:9], handles[:, 7:9], atol=1e-6)
    # the command half, after the helper, is on that same scale
    cmd = torch.zeros(1, 24, 9); cmd[0, :, 0:3] = torch.as_tensor(handles[:, 0:3])
    torch.testing.assert_close(command_for_encoder(cmd)[0, :, 0:3], torch.as_tensor(mh[:24, 0:3]), atol=1e-6, rtol=0)


def test_execution_transform_emits_centimetres_and_clamps_strength(model):
    g = torch.Generator().manual_seed(0)
    cmd = torch.rand(1, 24, 9, generator=g) * 2 - 1
    cmd[..., 7:9] = torch.rand(1, 24, 2, generator=g)
    dev = {"rom_limit": 1.0, "strength_limit": 0.3}
    out = apply_developmental_constraints(cmd, dev)
    torch.testing.assert_close(out[..., 0:3], cmd[..., 0:3].clamp(-1, 1) * 200.0)
    assert float(out[..., 8].max()) <= 0.3 + 1e-6
    ach = _achieved_row()
    out2 = model.execution_transform(cmd, dev, handle_state=ach)
    reach = 40.0
    assert float((out2[..., 0:3] - ach[..., 0:3] * 200.0).abs().max()) <= reach + 1e-4, "reach clamp in cm"


def test_action_decoder_reads_strength_from_column_8_and_documents_24_rows():
    from mimo_bridge import action_decoder as AD
    src = inspect.getsource(AD)
    assert "strength = float(np.clip(motor[i, 8]" in src
    assert "(24, 9)" in src and "row 23 = gaze" in src


def test_touch_point_cloud_row_layout_and_units():
    from mimo_bridge.lem_model import touch_point_cloud, TOUCH_POS_SCALE
    # 10 wide, the live width since 2026-09-14: with an 8-wide fixture the deliberate
    # 10-to-8 slice inside touch_point_cloud is a no-op and this test pins nothing
    # about which two channels get dropped.
    imp = torch.zeros(256, 10); imp[7, 0] = 0.3
    imp[7, 8], imp[7, 9] = 0.77, -0.66                # the two principal stretches
    pos = np.zeros((53, 3), np.float32); pos[7] = (0.1, -0.2, 0.05)         # metres, root-relative
    self_cols = np.zeros((53, 2), np.float32); self_cols[7] = (0.75, 0.5)
    push = np.zeros((53, 5), np.float32); push[7] = (0.0, 0.0, -1.0, 0.03, 0.12)
    out = touch_point_cloud(imp, pos, self_cols, push)
    assert out.shape == (TOUCH_TOKENS, TOUCH_TOKEN_F)
    np.testing.assert_allclose(out[0, 0:3].numpy(), np.array([0.1, -0.2, 0.05]) / TOUCH_POS_SCALE, atol=1e-6)
    assert out[0, 3] == pytest.approx(0.3) and out[0, TOUCH_TOKEN_F - 1] == 1.0   # valid is last
    # fields 0-7 land in slots 3-10; fields 8-9 (the UE avatar's principal stretches, a
    # constant in MIMo) are dropped and appear nowhere
    assert float(out[0, 4:11].abs().sum()) == 0.0
    row = np.abs(out[0].numpy())
    assert not np.any(np.isclose(row, 0.77)) and not np.any(np.isclose(row, 0.66)), \
        "a principal-stretch channel leaked into the touch row"
    # SELF-TOUCH lands in columns 11 and 12, and only there
    assert out[0, 11] == pytest.approx(0.75) and out[0, 12] == pytest.approx(0.5)
    assert float(out[1].abs().sum()) == 0.0, 'an empty row must stay empty'
    # THE PUSH: which way it came from, how deep, how fast it arrived - the information
    # the collision entries carried while nothing read them
    assert out[0, 15] == pytest.approx(-1.0) and out[0, 16] == pytest.approx(0.03)
    assert out[0, 17] == pytest.approx(0.12)
    plain = touch_point_cloud(imp, pos)
    assert float(plain[0, 11]) == 0.0 and float(plain[0, 12]) == 0.0
    assert float(plain[0, 13:18].abs().sum()) == 0.0, "no contact geometry without a source"
    assert plain[0, TOUCH_TOKEN_F - 1] == 1.0


def _self_touch_fixture():
    """Three bodies, a touch token and one prediction vector per body, all hand-worked.

    Body 0 has two of its own parts on one patch (column 12 reads 2 of 4 = 0.5) carrying
    0.8 of that patch's force, and also rests on the floor; body 1 touches nothing of
    itself; body 2 has one own part on it (1 of 4 = 0.25) carrying 0.6 of the force. The
    target embedding is zero, so every squared error below is just the prediction's own
    value squared and can be read off by hand."""
    from mimo_bridge.lem_model import TOUCH_TOKEN_F, TOUCH_VALID_COL
    t = torch.zeros(3, 4, TOUCH_TOKEN_F)
    t[0, 1, 11], t[0, 1, 12], t[0, 1, TOUCH_VALID_COL] = 0.8, 0.5, 1.0   # a self contact
    t[0, 2, TOUCH_VALID_COL] = 1.0                        # the floor, on the same body
    t[1, 0, TOUCH_VALID_COL] = 1.0                        # the floor only
    t[2, 0, 11], t[2, 0, 12], t[2, 0, TOUCH_VALID_COL] = 0.6, 0.25, 1.0  # a self contact
    target = torch.zeros(3, 4)
    prediction = torch.zeros(3, 4)
    prediction[0], prediction[1], prediction[2] = 0.1, 0.4, 0.2
    shuffled = torch.zeros(3, 4)
    shuffled[0], shuffled[1], shuffled[2] = 0.3, 0.5, 0.5
    return t, prediction, target, shuffled


def test_the_self_touch_audit_computes_its_contrast_and_refuses_what_it_cannot_score():
    """The number the self-touch wellness gate reads, and the four shapes that must stop it.

    The audit was reached by no test: the one test that named it hand-set the value it
    produces, so neither the arithmetic nor the guards ran, and the function was wrong twice
    in two days. Worked by hand on the fixture above, with the target at zero:

      bodies touching themselves (0 and 2): the prediction's error is (0.01 + 0.04) / 2
        = 0.025 and the shuffled-action control's is (0.09 + 0.25) / 2 = 0.17, so knowing
        the real action recovered (0.17 - 0.025) / 0.17 = 0.853 of the control's error,
        where 0 = the real action helped no more than a shuffled one and 1 = it predicted
        the next moment exactly;
      the body that was not touching itself (1): 0.16 against 0.25, so 0.360 recovered;
      the contrast, which is what the gate judges: 0.853 - 0.360 = 0.493, where 0 = knowing
        the action helped a self-touching body exactly as much as any other body.
    """
    from mimo_bridge.lem_model import LemLite, TOUCH_TOKEN_F

    class Stub:
        pass
    audit = LemLite._self_touch_audit.__get__(Stub())
    t, prediction, target, shuffled = _self_touch_fixture()
    out = audit(prediction, target, shuffled, t)
    assert out["bodies"] == 2 and out["other_bodies"] == 1 and out["rows"] == 2
    assert out["prediction_mse"] == pytest.approx(0.025)
    assert out["shuffled_mse"] == pytest.approx(0.17)
    assert out["improvement_over_shuffled"] == pytest.approx((0.17 - 0.025) / 0.17)
    assert out["other_prediction_mse"] == pytest.approx(0.16)
    assert out["other_shuffled_mse"] == pytest.approx(0.25)
    assert out["other_improvement_over_shuffled"] == pytest.approx(0.36)
    assert out["improvement_over_other_bodies"] == pytest.approx((0.17 - 0.025) / 0.17 - 0.36)

    # guard one: nothing that is not a touch token of this build's width can be scored, and
    # a moment in which no contact was the body's own has no self-touch to score
    assert audit(prediction, target, shuffled, None) is None
    assert audit(prediction, target, shuffled, torch.zeros(3, 4, 12)) is None
    assert audit(prediction, target, shuffled, torch.zeros(3, 4, TOUCH_TOKEN_F)) is None
    assert audit(prediction, target, shuffled, torch.rand(3, 192)) is None, (
        "the encoder's output is d_model wide and its columns mean nothing here")
    # guard two: a prediction or a target that is not a tensor at all
    assert audit(None, target, shuffled, t) is None
    assert audit(prediction, None, shuffled, t) is None
    # guard three: a prediction and a target of different shapes, and a prediction whose
    # first dimension is not the bodies the token describes
    assert audit(prediction, torch.zeros(3, 5), shuffled, t) is None
    assert audit(torch.zeros(2, 4), torch.zeros(2, 4), shuffled, t) is None
    # the control is optional: without one there is a prediction error and no number the
    # gate could read
    bare = audit(prediction, target, None, t)
    assert bare["prediction_mse"] == pytest.approx(0.025)
    assert bare["shuffled_mse"] is None and bare["improvement_over_shuffled"] is None
    assert bare["improvement_over_other_bodies"] is None
    assert audit(prediction, target, torch.zeros(3, 5), t)["improvement_over_shuffled"] is None
    # a control that was already perfect leaves nothing to recover, so the share is not a
    # number and must not be reported as one
    assert audit(prediction, target, target, t)["improvement_over_shuffled"] is None
    # with every body touching itself there is nothing to contrast against
    t_all = t.clone(); t_all[1, 0, 11] = 0.5
    every = audit(prediction, target, shuffled, t_all)
    assert every["bodies"] == 3 and every["other_bodies"] == 0
    assert every["other_improvement_over_shuffled"] is None
    assert every["improvement_over_other_bodies"] is None


def test_self_touch_reaches_the_model_and_pays_nothing_until_it_is_earned():
    """Self-touch was computed and thrown away: env.py knew which body was on the other
    side of every contact and nothing read it, so a hand on its own thigh was
    indistinguishable from the floor. It now rides the touch token, and the wellness term
    it feeds stays at exactly zero until a RUN of controlled readings of the audit's
    contrast stands above the line those readings' own spread draws. What the term pays for
    when it does open is the body's own parts ARRIVING on it - how much the count of own
    parts rose since the last step - so holding a hand in place pays once and then nothing.

    This test used to hand-set the value the audit produces, so the function that decides
    whether that value is ever set never ran here. It now starts from the audit's own
    number and carries it through the ring of readings into the wellness term.
    """
    from collections import deque
    from mimo_bridge.lem_model import (self_touch_rows, self_touching_bodies, LemLite,
                                       TOUCH_TOKEN_F, TOUCH_VALID_COL,
                                       SELF_TOUCH_READINGS, SELF_TOUCH_STALE_STEPS)
    from mimo_bridge.touch_adapter import region_of, SKIN_REGIONS
    t, prediction, target, shuffled = _self_touch_fixture()
    # the rows the audit scores: a real contact carrying force from the body's own parts
    rows = self_touch_rows(t)
    assert rows is not None and int(rows.sum()) == 2 and bool(rows[0, 1]) and bool(rows[2, 0])
    assert not bool(rows[0, 2]), "the floor is a real contact and carries none of the body"
    assert self_touch_rows(torch.zeros(2, 4, 12)) is None, "a narrower token is not scored"
    # AND the encoder's OUTPUT must be refused: it is d_model wide and its columns mean
    # nothing here. Scoring it was the first version's defect - the audit read an
    # embedding as though it were a touch token and reported nothing for hours.
    assert self_touch_rows(torch.rand(2, 192)) is None, "an embedding is not a touch token"
    assert self_touching_bodies(t).tolist() == [True, False, True]

    class Stub:
        pass
    stub = Stub()
    for name in ("_self_touch_wellness", "_self_touch_gate", "_self_touch_audit"):
        setattr(stub, name, getattr(LemLite, name).__get__(stub))
    stub.step_count = 10_000
    stub._self_touch_ring = deque(maxlen=SELF_TOUCH_READINGS)
    stub._self_touch_ring_step = 10_000
    # the number the gate will judge is the audit's own, not one written in by hand
    contrast = stub._self_touch_audit(prediction, target, shuffled, t)["improvement_over_other_bodies"]
    assert contrast == pytest.approx((0.17 - 0.025) / 0.17 - 0.36)

    # no weight -> exactly zero, whatever the body is doing
    stub.self_touch_wellness_weight = 0.0
    assert float(stub._self_touch_wellness(t, 3).abs().sum()) == 0.0
    assert "weight of 0" in stub.last_self_touch_gate_why
    # weight, and one reading of the contrast -> still exactly zero: one reading is not a run
    stub.self_touch_wellness_weight = 0.5
    stub._self_touch_ring.append(float(contrast))
    assert float(stub._self_touch_wellness(t, 3).abs().sum()) == 0.0
    assert "1 of the %d controlled readings needed" % SELF_TOUCH_READINGS \
        in stub.last_self_touch_gate_why
    # a full run whose average is zero -> still exactly zero. The readings alternate by
    # 0.0169, the spread one reading was measured to have over 154 of them.
    stub._self_touch_ring.clear()
    for k in range(SELF_TOUCH_READINGS):
        stub._self_touch_ring.append(0.0169 if k % 2 else -0.0169)
    assert float(stub._self_touch_wellness(t, 3).abs().sum()) == 0.0
    assert stub.last_self_touch_gate_mean == pytest.approx(0.0, abs=1e-12)
    # the line is twice the readings' own standard error: 32 readings of spread 0.0169
    # about their mean give 2 x 0.0169 / sqrt(31) = 0.0061 of the contrast's scale
    assert stub.last_self_touch_gate_line == pytest.approx(
        2.0 * 0.0169 / ((SELF_TOUCH_READINGS - 1) ** 0.5), rel=1e-6)
    # a full run of the audit's own contrast, with that same spread on it -> the gate opens
    stub._self_touch_ring.clear()
    for k in range(SELF_TOUCH_READINGS):
        stub._self_touch_ring.append(float(contrast) + (0.0169 if k % 2 else -0.0169))
    # the step before: nothing of the body's own is on it yet, so the count can rise
    nothing_of_its_own = torch.zeros(3, 4, TOUCH_TOKEN_F)
    nothing_of_its_own[:, 0, TOUCH_VALID_COL] = 1.0                 # resting on the floor
    assert float(stub._self_touch_wellness(nothing_of_its_own, 3).abs().sum()) == 0.0
    assert stub.last_self_touch_gate == 1.0, stub.last_self_touch_gate_why
    assert stub.last_self_touch_gate_mean == pytest.approx(float(contrast))
    # now the own parts arrive, and the body is paid for how far that count ROSE: body 0
    # went from no own part on it to 2 of the 4 the column counts, so 0.5 of a full hand
    paid = stub._self_touch_wellness(t, 3)
    assert paid.shape == (3, 1)
    assert float(paid[0, 0]) == pytest.approx(0.5 * 0.5)
    assert float(paid[1, 0]) == 0.0, "a body touching nothing of itself is paid nothing"
    assert float(paid[2, 0]) == pytest.approx(0.5 * 0.25)
    assert stub.last_self_touch_parts == pytest.approx((0.5 + 0.0 + 0.25) / 3)
    # holding the same contact pays once and then nothing, however long it is held
    for _ in range(3):
        assert float(stub._self_touch_wellness(t, 3).abs().sum()) == 0.0
    # and touching the world as well does not reduce it: the count of own parts is not a
    # share of the force, which is what made a body curled onto itself the best-paid one
    also_the_floor = t.clone()
    also_the_floor[:, 3, TOUCH_VALID_COL] = 1.0                     # one more world contact
    stub._self_touch_prev_parts = torch.zeros(3, 1)
    torch.testing.assert_close(stub._self_touch_wellness(also_the_floor, 3), paid)
    # and a gate is never held open by a measurement that has stopped arriving
    stub._self_touch_prev_parts = torch.zeros(3, 1)
    stub.step_count = 10_000 + SELF_TOUCH_STALE_STEPS + 1
    assert float(stub._self_touch_wellness(t, 3).abs().sum()) == 0.0
    assert "steps old" in stub.last_self_touch_gate_why

    # every sensing body belongs to exactly one named region
    names = [n for n, _ in SKIN_REGIONS]
    assert len(names) == len(set(names)) == 14
    for body in ("left_hand", "right_ffdistal", "chest", "left_big_toe", "head"):
        assert region_of(body) in names, body


def test_gaze_and_wellness_and_sensor_gain_read_the_documented_columns():
    src = inspect.getsource(LM)
    assert "math.radians(t.metadata.gaze_azimuth_deg)" in src, "gaze target: degrees on the wire, radians in the model"
    assert 'frame["tactile"][..., 3:11]' in src, "wellness reads the touch FEATURE columns (pressure at 0 of the slice)"
    assert "t2[..., 3:11] = t2[..., 3:11] * g" in src, "the movement gain scales sense columns, never geometry"


def test_tokenizer_ranges_are_the_ones_the_bridge_assumes():
    c = TokenizerConfig()
    assert c.position_range_cm == 200.0
    assert c.velocity_range_cms == 500.0
    assert c.energy_max == 1000.0




def test_no_test_writes_into_the_source_tree():
    """A test wrote 19 scene files beside itself on 2026-08-28 and they were committed.
    Every run_dir a test passes must come from tmp_path.

    The pattern is assembled from pieces: written literally, this check matched its own
    source and failed on itself - the grep-versus-execute trap, for the fourth time this
    week. A detector that cannot survive reading itself is not a detector."""
    import pathlib
    here = pathlib.Path(__file__).parent
    stray = [q for q in here.rglob('*')
             if q.is_dir() and q.name in ('_scratch', 'scenes', 'views', 'videos')]
    assert not stray, f'tests are writing into the source tree: {stray}'
    bad = 'os.path.dirname(' + '__file__), "_scr' + 'atch"'
    for q in here.glob('test_*.py'):
        src = q.read_text(encoding='utf-8')
        assert bad not in src, f'{q.name} builds a run_dir beside itself instead of using tmp_path'


# =========================================================== the perturbed control
def test_the_causal_test_has_a_same_state_control():
    """shuffled and cross-body both substitute an action chosen for a DIFFERENT state, so
    both can be beaten by a model that only learned which action belongs to which state
    and never learned what an action causes.

    Measured on H35 with the planner on, 74 transitions: another transition's action
    (distance 14.7) cost 20-68%, while perturbing the model's own chosen action by as much
    as 11.0 cost about 1%. Distance was not what the model responded to - kind was. The
    perturbed control keeps the state fixed and the action plausible for it, so beating it
    requires predicting a consequence."""
    assert "perturbed" in CT.CONTROLS
    src = inspect.getsource(CT)
    assert "def _perturb(" in src
    assert "PERTURB_FRACTION" in src, "the perturbation must be scaled to the stream's own spread"
    assert "_sd = _acts" in src, "the scale must come from the collected actions, not a constant"
    # the quaternion must be renormalised, or the perturbed action is not a rotation
    i = src.index("def _perturb(")
    body = src[i:i + 600]
    assert "norm(dim=-1, keepdim=True)" in body


def test_the_perturbed_control_reaches_the_verdict():
    """A control that is scored but not counted would be decoration."""
    src = inspect.getsource(CT)
    assert '("perturbed", _perturb(a_true))' in src
    # every summary/verdict loop iterates CONTROLS, so membership is enough - assert that
    assert src.count("for c in CONTROLS") >= 2
    assert 'for c in CONTROLS if c != "correct"' in src, "the beats map must include it"


# ============================ the architecture page against the code it describes
def test_the_sense_and_encoder_doc_text_matches_the_resolved_config():
    """Every number in the sense and encoder popups, re-derived from the code here.

    Seventeen explainer tests passed - the page builds, its embedded JSON parses, its
    tour is drawable - while eight statements in these entries were wrong (audit,
    2026-09-15): an encoder described as attention that is an MLP, a layer count of two
    where one is built, a skin-point count belonging to a different body, an arrow that
    claimed to carry nothing. None of those tests ever compared a sentence to the thing
    it describes. This one does, for the world, the six senses and the six encoders.
    """
    from mimo_bridge.explainer.schema import DOC
    from mimo_bridge.lem_model import build_config, _deep_merge
    from mimo_bridge.body_layout import config_overrides, names_for
    from mimo_bridge.vision_adapter import BinocularPatcher
    from tokenizer.modalities.touch import _FIELD_NORMS
    from lem.model import encoder_registry as ER
    from mimo_bridge import train_mimo as TM

    cfg = build_config(_deep_merge(
        {"modalities": {"audio": {"data_source": "real",
                                  "encoder": {"freq_band_path": "on"}}}},
        config_overrides("trunk_v1")))
    enc = {m: cfg["modalities"][m]["encoder"] for m in cfg["_active_modalities"]}
    n_handles = len(names_for("trunk_v1"))

    # body sense: a flat MLP over one long list - no attention, no pooling query
    assert enc["proprioception"]["type"] == "mlp_project"
    p = DOC["node.enc.proprioception"]["w"]
    assert f"{n_handles} body-part rows" in p
    assert f'{enc["proprioception"]["_input_dim"]}-number list' in p
    assert "attends to the others" not in p, "an MLP cannot compare one row with another"
    # HOW MANY LAYERS that MLP has is a launch setting (--encoder-layers L builds 1 + L
    # hidden layers and an output layer), so the entry may not type it: "a three-layer MLP"
    # was printed on H84's page, whose encoder has four. The sentence is filled from the
    # run's receipt, and is held here to the MODEL THAT WAS BUILT - its linear layers are
    # counted on the module, the receipt's inventory is made by the checkpoint code from
    # that same module, and the filled sentence has to name that count.
    from lem.training.checkpoint import _save_receipt
    from mimo_bridge.explainer import schema as SCH
    assert "three-layer" not in p and "{proprio_mlp}" in p
    for hidden in (None, [96, 96, 96]):
        ov = ({"modalities": {"proprioception": {"encoder": {"hidden_dims": hidden}}}}
              if hidden else None)
        m = LemLite(device="cpu", seed=0, intention=False, imagination=False, overrides=ov)
        mlp = m.encoder_registry.encoders["proprioception"]
        linear = [x for x in mlp.modules() if isinstance(x, torch.nn.Linear)]
        rec = _save_receipt(m.receipt, {"encoder_registry": m.encoder_registry},
                            m.optimizer, 0)
        said = SCH.run_fill(rec, {}, "RUNZ")["{proprio_mlp}"]
        assert f"a {len(linear)}-layer [[mlp|MLP]]" in said, (len(linear), said)
        assert f"{len(linear) - 1} hidden layer" in said
        assert f"each {linear[0].out_features} numbers wide" in said
        assert "RUNZ" in said, "the sentence must name the run the count was read from"
        assert SCH.encoder_depths(rec)["proprio_linear_layers"] == len(linear)
    assert len(linear) == 4, "the control: a 3-hidden-layer encoder is a 4-layer MLP"
    # a receipt with no tensor inventory says so and types no count
    bare = SCH.run_fill({}, {}, "RUNZ")["{proprio_mlp}"]
    assert "does not carry" in bare and not re.search(r"\d-layer", bare)

    # intermind: as many self-attention layers as the config actually builds
    assert enc["intermind"]["n_self_attn_layers"] == 1
    assert "one self-[[attention]] layer" in DOC["node.enc.intermind"]["w"]

    # hearing: the spectrum AND the two binaural numbers travel this line
    ap = cfg["modalities"]["audio"]["pipeline"]
    n_audio = ap["fft_n_bins"] * ap["n_channels"] + enc["audio"]["spatial_branch_input_dim"]
    a = DOC["edge.sense_audio.enc"]["w"]
    assert str(n_audio) in a and "0.204 ms" in a
    assert enc["audio"]["combine"] == "gate" and "see-saw" in DOC["node.enc.audio"]["w"]

    # touch: the row cap, the row width and the divisors that undo them
    t = DOC["edge.sense_tactile.enc"]["w"]
    assert f"{TOUCH_TOKENS} rows x {TOUCH_TOKEN_F} numbers" in t
    assert f"{int(_FIELD_NORMS[0]):,} Pa" in t and f"{int(_FIELD_NORMS[3]):,} mm2" in t

    # vision: the patch count and both patch spans, from the adapter's own arithmetic
    bp = BinocularPatcher([("wide", 256, 60.0), ("fovea", 288, 24.0)])
    v = DOC["edge.sense_vision.enc"]["w"]
    assert f"{bp.n_total:,} rows" in v
    for src_ in bp.sources:
        assert f"{math.degrees(src_[4]):.2f} degrees" in v

    # vision is fresh every moment because the trainer names the frames it will read
    assert "render=(rep == reps - 1)" in inspect.getsource(TM), \
        "if the render cadence changes, node.sense.vision must change with it"
    assert "stale" not in DOC["node.sense.vision"]["when"]

    # the shared LayerNorm runs inside the encoder call, so no un-normalised token travels
    assert "self.output_norm(out)" in inspect.getsource(ER.EncoderRegistry._encode_one)
    assert "already held at [[rms|RMS]] 1.0" in DOC["edge.enc.ln_out"]["w"]

    # and the picture itself carries a contract on every sense-to-encoder line
    import mimo_bridge.explainer as EX
    vjs = open(os.path.join(os.path.dirname(EX.__file__), "viewer.js"),
               encoding="utf-8").read()
    assert "\u2192 [1\u00d7" not in vjs, "the encoder sub-label must carry the batch"
    assert "WIRE[m]" in vjs, "every sense-to-encoder line must carry its short contract"


# ------------------------------------------------------------ P021: the proprioception row index
def _proprio_rows_as_datasource_builds_them(n_rows):
    """The 12-column row exactly as datasource._payload_to_frame assembles it."""
    import numpy as np
    from tokenizer.config import TokenizerConfig
    cfg = TokenizerConfig()
    rng = np.random.default_rng(0)
    bones = np.zeros((n_rows, 11), np.float32)
    q = rng.normal(size=(n_rows, 4)); bones[:, :4] = q / np.linalg.norm(q, axis=1, keepdims=True)
    bones[:, 4:7] = rng.uniform(-30, 30, (n_rows, 3))          # cm, world
    bones[:, 7:10] = rng.uniform(-5, 5, (n_rows, 3))           # cm/s
    bones[:, 10] = rng.uniform(0, 1, n_rows)                   # fatigue
    normalized = np.zeros((n_rows, 12), np.float32)
    normalized[:, :11] = bones
    normalized[:, 4:7] = (bones[:, 4:7] - bones[0, 4:7]) / cfg.position_range_cm
    normalized[:, 7:10] /= cfg.velocity_range_cms
    normalized[:, 11] = np.arange(n_rows) / cfg.proprio_index_range
    return normalized


@pytest.mark.parametrize("n_rows", [24, 26])
def test_no_proprioception_channel_leaves_the_normalised_band(n_rows):
    """Every channel in a proprioception row is built to land inside 1.0 - quaternion
    components, cm/200, cm-s/500, fatigue 0..1. The row index used to arrive raw, 0..25.
    Measured before the fix: max |channel| 25.0; after: 1.0000 on both body sizes."""
    import numpy as np
    row = _proprio_rows_as_datasource_builds_them(n_rows)
    assert float(np.abs(row[:, 11]).max()) <= 1.0
    assert float(np.abs(row).max()) <= 1.0 + 1e-6


def test_the_row_index_does_not_own_the_proprioception_row():
    """The index block's size must sit within 15x of the quaternion block's - the same
    15x the motor row's contract test allows between its two halves. Measured: 29.15x
    before the fix, 1.17x after. NOT a blanket max/min over all five blocks: the
    velocity block is small for a data reason (this crib's body moves at ~1 cm/s
    against a 500 cm/s divisor), and quaternion-vs-velocity is already 20.6x for
    that unrelated reason."""
    import numpy as np
    row = _proprio_rows_as_datasource_builds_them(26)
    rms = lambda a: float(np.sqrt(np.mean(np.square(a))))
    index_rms = rms(row[:, 11])
    quat_rms = rms(row[:, :4])
    ratio = max(index_rms, quat_rms) / max(min(index_rms, quat_rms), 1e-9)
    assert ratio < 15.0, f"row index vs quaternion block RMS ratio {ratio:.2f}x (was 29.15x raw)"
    # and the index is no longer most of the row's energy
    share = float(np.sum(np.square(row[:, 11])) / np.sum(np.square(row)))
    assert share < 0.5, f"the row index still carries {100*share:.1f}% of the row's energy"


def test_both_proprioception_paths_use_the_same_index_scale():
    """The UE 24-row tokenizer path and the bridge 26-row path write the same channel;
    a fixed constant keeps row 12 meaning the same thing on both (0.48), where
    len(bones)-1 would give 0.5217 on 24 rows and 0.4800 on 26."""
    import numpy as np
    from tokenizer.config import TokenizerConfig
    from tokenizer.modalities import proprioception as P
    import inspect
    src = inspect.getsource(P)
    assert "float(i) / config.proprio_index_range" in src, "the 24-row path writes a different scale"
    r24 = _proprio_rows_as_datasource_builds_them(24)
    r26 = _proprio_rows_as_datasource_builds_them(26)
    assert r24[12, 11] == r26[12, 11] == 12 / TokenizerConfig().proprio_index_range



# ---------------------------------------------------------------- audit P024 (2026-09-18)
def test_proprioception_and_motor_carry_the_same_bodies_in_two_named_frames():
    """Mode A path: the proprioception row and the achieved motor row are built from the
    SAME bones array in the same step and use DIFFERENT origins - proprioception subtracts
    row 0 (wrist_L), the achieved motor half subtracts nothing (world centimetres). Both
    divide by 200 cm. Pinned as a pair so neither origin can move on its own."""
    import numpy as np
    from mimo_bridge import frame_encoder as FE
    from mimo_bridge.datasource import motor_to_echo, _payload_to_frame
    from mimo_bridge.touch_adapter import bone_hash
    n = 26
    rng = np.random.default_rng(7)
    bones = np.zeros((n, 11), np.float32); bones[:, 3] = 1.0
    bones[:, 4:7] = rng.uniform(-40, 40, size=(n, 3))
    echo = motor_to_echo(np.zeros((n, 9), np.float32), None, bones=bones)
    hashes = [bone_hash(f"b{i}") for i in range(24)]
    buf = FE.encode_frame(frame_number=1, timestamp=0.01, dt=0.01, wall_clock=0.0,
                          gaze_az_deg=0.0, gaze_el_deg=0.0, bones=bones[:24],
                          bone_hashes=hashes, handles=echo[:24],
                          energy=np.zeros(3, np.float32),
                          body_sets=np.zeros((10, 2), np.float32),
                          patches=np.zeros((53, 10), np.float32), collisions=[],
                          vision_valid=False)
    f = _payload_to_frame(dict(frame_bytes=bytes(buf), extended_handles=echo, bones=bones,
                               eye_left=None, sim_time=0.01, dt=0.01, frame=1, n_contacts=0,
                               patches=np.zeros((53, 10), np.float32),
                               vestibular=np.zeros(6, np.float32),
                               eye_angles=((0.0, 0.0), (0.0, 0.0)), gaze_deg=(0.0, 0.0),
                               ctrl=np.zeros(1, np.float32), info={}), 0)
    pro = f.tokenized.proprio.numpy()[:, 4:7]
    mot = f.tokenized.motor_handles.numpy()[:n, 0:3]
    np.testing.assert_allclose(mot, bones[:, 4:7] / 200.0, atol=1e-6)
    np.testing.assert_allclose(pro, (bones[:, 4:7] - bones[0, 4:7]) / 200.0, atol=1e-6)
    np.testing.assert_allclose(mot - pro, np.tile(bones[0, 4:7] / 200.0, (n, 1)), atol=1e-6)
    np.testing.assert_allclose(mot - mot.mean(0), pro - pro.mean(0), atol=1e-6)


def test_the_page_names_both_frames_of_the_motor_row():
    from mimo_bridge.explainer.schema import DOC
    w = DOC["edge.world.sense_motor"]["w"]
    assert "left wrist" in w.lower() and "world centimetres" in w
