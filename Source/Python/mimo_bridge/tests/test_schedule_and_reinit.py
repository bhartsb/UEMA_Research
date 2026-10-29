# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""One developmental schedule; a fresh encoder on resume; the four ledger rows nobody checked.

The schedule had two copies (lem/config/defaults.py, four phases by absolute step; the
CurriculumManager, six phases by steps_per_phase). The trainer used one, LemLite.dev_limits()
and the causal test the other. Now one producer: lem.training.curriculum.limits_for().
Operator decisions of 2026-08-27 recorded here as assertions: babble never below 0.05
(movement is a wellness term; at 0.0 the body froze in seven runs), the values otherwise
a first draft to be revisited after the next run.
"""
import inspect
import os

import numpy as np
import pytest
import torch

from lem.config.defaults import get_defaults
from lem.training.curriculum import (CurriculumManager, PHASE_ORDER, PHASE_DEFINITIONS,
                                     BABBLE_FLOOR, limits_for, limits_at_step, phases_table)
from lem.training.harness import _get_developmental_phase
from mimo_bridge.lem_model import LemLite

# a test must not write into the source tree: this one dropped 19 scene files beside
# itself and they were committed before anyone noticed. tmp_path, always.


# =========================================================== one schedule
@pytest.mark.parametrize("spp", [3000, 50_000])
def test_dev_limits_answers_what_the_curriculum_manager_answers(spp):
    """Same clock, same values: the manager stepped to `step` vs limits_at_step(step)."""
    cm = CurriculumManager(steps_per_phase=spp, mastery_advancement=False)
    checkpoints = [0, 1, spp - 1, spp, 2 * spp + 7, 5 * spp, 6 * spp + 1, 40 * spp]
    step = 0
    for target in checkpoints:
        while step < target:
            cm.step()
            step += 1
        a, b = cm.get_dev_limits(), limits_at_step(target, spp)
        for k in ("phase_name", "rom_limit", "strength_limit", "exploration_noise", "context_length"):
            assert a[k] == b[k], (target, k, a[k], b[k])


def test_model_dev_limits_uses_the_runs_clock():
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False,
                overrides={"training": {"steps_per_phase": 3000}})
    assert m.dev_limits(3)["phase_name"] == "neonate"
    assert m.dev_limits(15_000)["phase_name"] == "mature"
    assert m.dev_limits(897_695)["strength_limit"] == 1.0
    d = LemLite(device="cpu", seed=0, intention=False, imagination=False)   # config default 50k
    assert d.dev_limits(15_000)["phase_name"] == "neonate"


def test_the_config_table_is_generated_from_the_definitions():
    c = get_defaults()
    t = c["training"]["developmental_phases"]
    assert list(t) == PHASE_ORDER
    assert t == phases_table(c["training"]["steps_per_phase"])
    for name in PHASE_ORDER:
        lim = limits_for(PHASE_DEFINITIONS[name])
        assert t[name]["rom_limit"] == lim["rom_limit"]
        assert t[name]["strength_limit"] == lim["strength_limit"]
        assert t[name]["exploration_noise"] == lim["exploration_noise"]


def test_babble_never_reaches_zero_from_any_reader():
    c = get_defaults()
    for name in PHASE_ORDER:
        assert limits_for(PHASE_DEFINITIONS[name])["exploration_noise"] >= BABBLE_FLOOR
        assert c["training"]["developmental_phases"][name]["exploration_noise"] >= BABBLE_FLOOR
    for step in (0, 10 ** 4, 10 ** 6, 10 ** 9):
        assert _get_developmental_phase(c, step)["exploration_noise"] >= BABBLE_FLOOR
        assert limits_at_step(step, 3000)["exploration_noise"] >= BABBLE_FLOOR
    cm = CurriculumManager(steps_per_phase=10, mastery_advancement=False, start_phase="mature")
    assert cm.get_dev_limits()["exploration_noise"] >= BABBLE_FLOOR
    assert BABBLE_FLOOR == 0.05


def test_legacy_configs_without_a_clock_still_read_their_table():
    c = get_defaults()
    c["training"] = dict(c["training"]); c["training"].pop("steps_per_phase")
    lim = _get_developmental_phase(c, 10 ** 9)
    assert lim["phase_name"] == "mature" and lim["strength_limit"] == 1.0
    assert _get_developmental_phase(c, 0)["phase_name"] == "neonate"


# =========================================================== fresh encoder on resume
def test_reinit_starts_one_encoder_fresh_and_keeps_the_rest(tmp_path):
    from lem.training.checkpoint import CheckpointManager
    torch.manual_seed(0)
    a = LemLite(device="cpu", seed=0, intention=False, imagination=False)
    # perturb so the saved weights differ from any fresh init, and give the optimizer state
    with torch.no_grad():
        for p in a.params:
            p.add_(torch.randn_like(p) * 0.01)
    for p in a.encoder_registry.encoders["motor"].parameters():
        a.optimizer.state[p] = {"exp_avg": torch.zeros_like(p), "step": torch.tensor(5.0)}
    a.checkpoints = CheckpointManager(a.config, str(tmp_path))
    path = a.save_checkpoint(123)
    assert path

    b = LemLite(device="cpu", seed=1, intention=False, imagination=False)
    b.checkpoints = CheckpointManager(b.config, str(tmp_path))
    res = b.load_checkpoint(path, load_optimizer=True, reinit=("encoder:motor",))
    assert "encoder:motor" in res.get("reinitialized_components", []), res
    # motor differs from the saved weights, every other encoder equals them
    sa = dict(a.encoder_registry.encoders["motor"].named_parameters())
    sb = dict(b.encoder_registry.encoders["motor"].named_parameters())
    assert any(not torch.allclose(sa[k], sb[k]) for k in sa), "motor encoder was loaded, not reinitialised"
    for name in ("tactile", "proprioception", "vision"):
        ea = dict(a.encoder_registry.encoders[name].named_parameters())
        eb = dict(b.encoder_registry.encoders[name].named_parameters())
        assert all(torch.allclose(ea[k], eb[k]) for k in ea), name
    fa = dict(a.forward_model.named_parameters()); fb = dict(b.forward_model.named_parameters())
    assert all(torch.allclose(fa[k], fb[k]) for k in fa)
    # and the fresh encoder carries no optimizer moments
    for p in b.encoder_registry.encoders["motor"].parameters():
        assert p not in b.optimizer.state or not b.optimizer.state[p]


def test_reinit_flag_reaches_load_checkpoint():
    from mimo_bridge import train_mimo
    src = inspect.getsource(train_mimo)
    assert '"--reinit"' in src and "reinit=tuple(" in src
    # The assignment moved into bind_schedule_to_config (2026-09-15) so the launch flag
    # reaches the model at resume as well as at start, instead of only on the one path
    # this grep used to sit on. The property is the same: the trainer sets the run's
    # clock on the model, from args.steps_per_phase.
    assert "def bind_schedule_to_config(" in src, "the trainer must bind its schedule in one place"
    assert 'tr["steps_per_phase"] = int(steps_per_phase)' in src, "the trainer must set the run's clock on the model"
    assert "bind_schedule_to_config(model, cm, args.steps_per_phase" in src, "and it must be called with the launch flag"


# =========================================================== the four unguarded ledger rows
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from mimo_bridge.env import BridgeEnv
    e = BridgeEnv(scene="reach", run_dir=str(tmp_path_factory.mktemp("env")), vision=False, tag="_contracts")
    yield e
    e.close()


def test_ctrl_echo_is_what_the_actuators_received(env):
    rng = np.random.default_rng(0)
    motor = np.zeros((24, 9), np.float32); motor[:, 6] = 1.0; motor[:, 8] = 0.5
    ax = rng.normal(size=(24, 3)); ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    motor[:, 3:6] = ax * np.sin(0.2); motor[:, 6] = np.cos(0.2)
    raw = env.step(motor, None, None, render=False)
    # the decoder clips to the BODY's action space, which is narrower than
    # actuator_ctrlrange on 56 of 90 MIMo actuators; before 2026-08-27 it clipped to
    # +/-1 and MIMo clamped silently, so the echo and the saturation statistic lied
    np.testing.assert_allclose(env.decoder.ctrl_low, np.maximum(env.env.action_space.low, -1.0), atol=1e-7)
    np.testing.assert_allclose(env.decoder.ctrl_high, np.minimum(env.env.action_space.high, 1.0), atol=1e-7)
    assert float(env.decoder.ctrl_high.min()) < 0.8, "MIMo's asymmetric bounds should be visible here"
    np.testing.assert_allclose(raw.ctrl, env.data.ctrl, atol=1e-7)
    assert raw.ctrl.shape == (env.model.nu,) and float(np.abs(raw.ctrl).max()) <= 1.0


def test_energy_and_capability_cross_the_wire_normalised():
    from mimo_bridge import frame_encoder as FE
    from mimo_bridge.datasource import motor_to_echo
    from mimo_bridge.touch_adapter import bone_hash
    from lem.model.body_sets import HANDLE_NAMES
    from tokenizer.config import TokenizerConfig
    rng = np.random.default_rng(3)
    bones = np.zeros((24, 11), np.float32); bones[:, 3] = 1.0
    handles = motor_to_echo(np.zeros((24, 9), np.float32), None, bones=bones)
    energy = np.array([120.0, 340.0, 460.0], np.float32)
    body_sets = np.column_stack([rng.uniform(0, 1, 10), rng.uniform(0, 900, 10)]).astype(np.float32)
    patches = np.zeros((53, 10), np.float32); patches[:, 8:10] = 1.0
    buf = FE.encode_frame(frame_number=1, timestamp=0.01, dt=0.01, wall_clock=0.0, gaze_az_deg=0.0, gaze_el_deg=0.0,
                          bones=bones, bone_hashes=[bone_hash(n) for n in HANDLE_NAMES], handles=handles,
                          energy=energy, body_sets=body_sets, patches=patches, collisions=[], vision_valid=False)
    tf = FE.decode_with_model_tokenizer(bytes(buf))
    emax = TokenizerConfig().energy_max
    e = np.asarray(tf.motor_energy); cap = np.asarray(tf.motor_capability)
    np.testing.assert_allclose(e, energy / emax, atol=1e-6)
    assert cap.shape == (21,)
    np.testing.assert_allclose(cap[:10], body_sets[:, 0], atol=1e-6)          # capacity [0,1] as is
    np.testing.assert_allclose(cap[10:20], body_sets[:, 1] / emax, atol=1e-6)  # accumulated / energy_max
    assert cap[20] == pytest.approx(energy[2] / emax)


def test_the_trainers_next_state_error_is_the_causal_tests_formula(tmp_path):
    """The dashboard's next_pe and the test's 'correct' arm must be one measurement.
    Verified on one stream: 0.0854 vs 0.0850 (H33). Here on a short CPU stream."""
    import torch.nn.functional as F
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.tools.causal_test import _predict
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False, next_frame_loss=True,
                overrides={"training": {"steps_per_phase": 3000}})
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "parity"), tag="_parity", vision=False, seed=3)
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9)); dev["exploration_noise"] = 0.05
        held = prev = pending = None; pairs = []
        for i in range(14):
            with torch.no_grad():
                res = m.step(frames, dev, train=False, step_index=i, executed_action=prev)
                toks = {k: v.detach().float().cpu() for k, v in (m.last_tokens or {}).items()}
                ach = m.last_achieved.detach().float().cpu()
                cb = m.context_buffer.detach().float().cpu()
                ctx = torch.cat([torch.zeros_like(cb[:, :m.n_mod]), cb[:, :-m.n_mod]], dim=1).clone()
            if pending is not None and prev is not None and i >= 6:
                tr = {"t": pending[0], "ach": pending[1], "ctx": pending[2], "t1": toks,
                      "act": torch.as_tensor(prev, dtype=torch.float32)}
                pred = _predict(m, tr, tr["act"], "cpu")
                test_side = {k: float(F.mse_loss(pred[k].float(), toks[k].float())) for k in ("proprioception", "tactile") if k in pred}
                trainer_side = {k: float(res.next_pe_per_modality.get(k)) for k in test_side}
                pairs.append((trainer_side, test_side))
            pending = (toks, ach, ctx)
            if i % 5 == 0 or held is None:
                held = res.motor_for_action
            for _ in range(5):
                frames = src.step(held, res.gaze_out, None, render=False)
            prev = held
    finally:
        src.close()
    assert pairs
    for k in ("proprioception", "tactile"):
        a = np.mean([p[0][k] for p in pairs]); b = np.mean([p[1][k] for p in pairs])
        assert abs(a - b) <= 0.15 * max(a, b) + 1e-6, (k, a, b)


def test_babble_flag_reaches_collection_and_never_the_passive_arm():
    from mimo_bridge.tools import causal_test as CT
    src = inspect.getsource(CT._collect)
    assert 'dev["exploration_noise"] = float(babble)' in src
    assert "0.0 if limp else babble" in inspect.getsource(CT.run)
    assert "--babble-noise" in inspect.getsource(CT.main)
