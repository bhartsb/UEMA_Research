# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Acceptance tests for the transition contract (MIMo/Transition_Contract_Plan_2026-08-25.md).

Each test pins one claim the plan makes. They exist because the defect they guard was
invisible to every other test in the suite for months: the model trained happily while
being told it had performed actions it never performed.
"""
import numpy as np
import pytest
import torch

from mimo_bridge.lem_model import (LemLite, MOTOR_ACHIEVED_F, MOTOR_COMMAND_F,
                                   MOTOR_INPUT_F)
from mimo_bridge.replay import ReplayMemory


# ---------------------------------------------------------------- item 2: the ledger
def test_gaze_round_trips_through_handle_23():
    """The executed eye command must survive the trip into the action tensor.

    env.py records the gaze that actually happened - after the saccade generator and the
    peripheral-motion reflex have overridden whatever the controller asked for - and it
    has to reach the model as handle 23's orientation. A quaternion convention mismatch
    here would silently feed the model the wrong eye movement.
    """
    rng = np.random.default_rng(0)
    az = rng.uniform(-0.785, 0.785, 128).astype(np.float32)
    el = rng.uniform(-0.820, 0.576, 128).astype(np.float32)
    m = np.zeros((128, 24, 9), np.float32)
    m[:, :, 6] = 1.0
    enc = LemLite.gaze_to_handle(m, np.stack([az, el], 1))

    q = torch.tensor(enc[:, LemLite.GAZE_IDX, 3:7])
    q = q / q.norm(dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
    assert np.abs(yaw.numpy() - az).max() < 1e-5
    assert np.abs(pitch.numpy() - el).max() < 1e-5


def test_gaze_write_touches_only_handle_23():
    """An eyeball has no position, radius or strength to command, and no other handle's
    command may be disturbed by writing the eye command."""
    rng = np.random.default_rng(1)
    m = rng.normal(size=(4, 24, 9)).astype(np.float32)
    out = LemLite.gaze_to_handle(m, np.zeros((4, 2), np.float32))
    others = [i for i in range(24) if i != LemLite.GAZE_IDX]
    assert np.array_equal(out[:, others], m[:, others])
    assert np.array_equal(out[:, LemLite.GAZE_IDX, :3], m[:, LemLite.GAZE_IDX, :3])
    assert np.array_equal(out[:, LemLite.GAZE_IDX, 7:], m[:, LemLite.GAZE_IDX, 7:])


def test_held_action_is_one_action_across_the_hold():
    """motor_hold=5 means ONE order is in force for five model transitions.

    The defect this pins: the training loop held an action for five steps while the model
    recorded a fresh proposal each step, so four of five labels described orders that were
    never issued. Whatever the loop does, five consecutive transitions under a hold of
    five must carry one identical executed action.
    """
    hold, steps = 5, 20
    held, executed = None, []
    for step in range(steps):
        proposal = np.full((1, 24, 9), float(step), np.float32)   # a different one each step
        if step % hold == 0 or held is None:
            held = proposal
        executed.append(held)
    for block in range(steps // hold):
        window = executed[block * hold:(block + 1) * hold]
        assert all(np.array_equal(a, window[0]) for a in window), \
            f"block {block} carries {len({a[0, 0, 0] for a in window})} distinct actions"
    assert len({float(a[0, 0, 0]) for a in executed}) == steps // hold


# ------------------------------------------------- item 3: widened, migrates cleanly
def test_motor_widths_are_consistent():
    assert MOTOR_INPUT_F == MOTOR_ACHIEVED_F + MOTOR_COMMAND_F == 19


def test_zero_padded_widening_changes_nothing():
    """The migration's whole claim: a widened layer with zero new columns computes what
    the old one computed. Exact in float64; float32 differs only by matmul accumulation
    order, which is why the tool checks float64."""
    torch.manual_seed(0)
    w = torch.randn(128, MOTOR_ACHIEVED_F, dtype=torch.float64)
    grown = torch.zeros(128, MOTOR_INPUT_F, dtype=torch.float64)
    grown[:, :MOTOR_ACHIEVED_F] = w

    x_old = torch.randn(8, 24, MOTOR_ACHIEVED_F, dtype=torch.float64)
    x_new = torch.cat([x_old, torch.randn(8, 24, MOTOR_COMMAND_F, dtype=torch.float64)], -1)
    # NOT torch.equal: a reviewer measured 1.776e-15 on their platform. Different BLAS
    # kernels reach different accumulation orders even in float64, and demanding literal
    # bit identity makes the test a platform lottery rather than a statement about the
    # maths. What must hold is that the nine new columns contribute nothing.
    torch.testing.assert_close(torch.nn.functional.linear(x_old, w),
                               torch.nn.functional.linear(x_new, grown),
                               rtol=0, atol=1e-12)
    assert torch.count_nonzero(grown[:, MOTOR_ACHIEVED_F:]) == 0


def test_build_motor_input_layout():
    """Every producer of a motor token must agree on the layout, or the planner scores in
    a domain the scorer never saw."""
    from lem.model.encoder_registry import build_motor_input

    class _Enc:
        patch_mlp = [torch.nn.Linear(MOTOR_INPUT_F, 8, bias=False)]

    class _Reg:
        encoders = {"motor": _Enc()}

    action = torch.arange(2 * 24 * 9, dtype=torch.float32).reshape(2, 24, 9)
    achieved = torch.full((2, 24, MOTOR_ACHIEVED_F), -7.0)
    out = build_motor_input(_Reg(), action, achieved=achieved)
    assert out.shape == (2, 24, MOTOR_INPUT_F)
    assert torch.equal(out[..., :MOTOR_ACHIEVED_F], achieved)
    assert torch.equal(out[..., MOTOR_ACHIEVED_F:], action)

    zeroed = build_motor_input(_Reg(), action, achieved=None)
    assert torch.count_nonzero(zeroed[..., :MOTOR_ACHIEVED_F]) == 0
    assert torch.equal(zeroed[..., MOTOR_ACHIEVED_F:], action)


def test_build_motor_input_legacy_encoder_still_works():
    from lem.model.encoder_registry import build_motor_input

    class _Enc:
        patch_mlp = [torch.nn.Linear(10, 8, bias=False)]

    class _Reg:
        encoders = {"motor": _Enc()}

    action = torch.zeros(2, 24, 9)
    out = build_motor_input(_Reg(), action)
    assert out.shape == (2, 24, 10)
    assert torch.all(out[..., 9] == 1.0)          # is_active


# ------------------------------------------------------ item 1: source-mask filtering
def _mem(n_envs=2, size=8):
    return ReplayMemory(["vision", "audio", "proprioception"], d_model=4,
                        n_handles=24, size_per_env=size, n_envs=n_envs)


def test_replay_records_validity_per_modality():
    """A synthetic channel is a fresh Gaussian draw. Grading a prediction head against it
    teaches the head to predict noise, and the imagination loss did exactly that at every
    rollout depth for two channels."""
    mem = _mem()
    toks = {m: torch.zeros(2, 1, 4) for m in mem.mods}
    mem.add(toks, torch.zeros(2, 24, 10), {"rom_limit": 1.0, "strength_limit": 1.0},
            source_mask={"vision": "real", "audio": "synthetic", "proprioception": "real"})
    assert bool(mem.valid[0, 0, 0]) is True       # vision
    assert bool(mem.valid[0, 0, 1]) is False      # audio
    assert bool(mem.valid[0, 0, 2]) is True       # proprioception


def test_replay_defaults_to_real_without_a_mask():
    mem = _mem()
    mem.add({m: torch.zeros(2, 1, 4) for m in mem.mods}, torch.zeros(2, 24, 10),
            {"rom_limit": 1.0, "strength_limit": 1.0})
    assert bool(mem.valid[0, 0].all())


def test_replay_tracks_whether_the_action_was_executed():
    """clean_fraction is the ring-fill gate: replay must not start until every stored
    moment carries the order the body actually got."""
    mem = _mem(n_envs=1, size=4)
    toks = {m: torch.zeros(1, 1, 4) for m in mem.mods}
    dev = {"rom_limit": 1.0, "strength_limit": 1.0}
    mem.add(toks, torch.zeros(1, 24, 10), dev)                                   # unlabelled
    assert mem.clean_fraction() == 0.0
    for _ in range(3):
        mem.add(toks, torch.zeros(1, 24, 10), dev, executed=torch.ones(1, 24, 9))
    assert mem.clean_fraction() == pytest.approx(0.75)
    assert mem.fill_fraction() == 1.0


def test_clearing_the_ring_forgets_everything_and_bumps_the_episode():
    """Windows must not span the clear: a window that mixed pre-fix and post-fix moments
    would carry two different action contracts."""
    mem = _mem(n_envs=1, size=4)
    dev = {"rom_limit": 1.0, "strength_limit": 1.0}
    for _ in range(4):
        mem.add({m: torch.zeros(1, 1, 4) for m in mem.mods}, torch.zeros(1, 24, 10), dev)
    ep_before = mem._episode_id
    mem.clear()
    assert mem.count == 0 and mem.ptr == 0
    assert mem.clean_fraction() == 0.0
    assert mem._episode_id == ep_before + 1


def test_sampled_windows_carry_validity_and_executed_action():
    mem = _mem(n_envs=1, size=64)
    dev = {"rom_limit": 1.0, "strength_limit": 1.0}
    for i in range(40):
        mem.add({m: torch.full((1, 1, 4), float(i)) for m in mem.mods},
                torch.zeros(1, 24, 10), dev,
                source_mask={"vision": "real", "audio": "synthetic",
                             "proprioception": "real"},
                executed=torch.full((1, 24, 9), float(i)))
    w = mem.sample_windows(batch=8, ctx_len=4, device="cpu")
    assert w is not None
    assert w["valid"].shape[-1] == len(mem.mods)
    assert not bool(w["valid"][..., 1].any())     # audio never graded
    assert bool(w["valid"][..., 0].all())         # vision always graded
    assert bool(w["has_executed"].all())
    assert w["executed"].shape[-2:] == (24, 9)


def test_imagination_loss_skips_synthetic_targets():
    """The measured defect: _run_imagination_compiled summed MSE over every prediction
    head with no source filter, so audio and intermind - fresh Gaussian placeholders -
    were chased at weight 0.5 once per rollout step."""
    import torch.nn.functional as F
    from lem.source_mask import to_str

    predicted = {"vision": torch.zeros(2, 4), "audio": torch.zeros(2, 4)}
    target = {"vision": torch.ones(2, 4), "audio": torch.full((2, 4), 100.0)}
    valid = {"vision": "real", "audio": "synthetic"}

    graded = [m for m in predicted if m in target and to_str(valid.get(m, "real")) == "real"]
    loss = sum(F.mse_loss(predicted[m], target[m]) for m in graded)
    assert graded == ["vision"]
    assert float(loss) == pytest.approx(1.0)      # the 100.0 target contributes nothing


# =====================================================================================
# The tests below exist because the first round of "transition contract" tests all
# passed while the contract was broken. Each asserts a CONSEQUENCE - that changing the
# action changes something downstream - rather than that a tensor was stored.
# =====================================================================================

def _tiny_model(**kw):
    """Smallest CPU model that still exercises the real code paths."""
    from mimo_bridge.lem_model import LemLite
    return LemLite(device="cpu", overrides={"model": {"context_length": 4}},
                   next_frame_loss=True, **kw)


def test_replay_action_is_not_dead_data():
    """THE regression test for the defect that shipped: replay stored the executed action
    and never read it.

    The first version of this test ran replay_update twice and compared losses. A reviewer
    showed that structure yields DIFFERENT losses from IDENTICAL action data, because the
    first update moves the weights the second is measured against - so it could not
    distinguish a live action pathway from a dead one. This version measures the loss with
    no weight change at all, from identical sampled windows, changing only the action.
    """
    torch.manual_seed(0)
    m = _tiny_model()
    m.enable_replay(n_envs=2, size_per_env=64)
    mods, d, ctx = m.replay.mods, m.d_model, m.ctx_len

    def fill(action_value):
        m.replay.clear()
        torch.manual_seed(7)
        for _ in range(ctx + 14):
            toks = {mod: torch.randn(2, 1, d) * 0.1 for mod in mods}
            m.replay.add(toks, torch.randn(2, 24, 10) * 0.1,
                         {"rom_limit": 1.0, "strength_limit": 1.0},
                         executed=torch.full((2, 24, 9), float(action_value)))

    losses = []
    for val in (0.0, 999.0):
        fill(val)
        losses.append(m.replay_loss_only(batch=4, seed=3))
    assert all(l is not None for l in losses), "replay produced nothing; test inconclusive"
    assert losses[0] != losses[1], (
        f"replay loss identical ({losses[0]}) for actions 0 and 999 with weights held "
        "fixed: the stored action is not reaching the forward model")


def test_replay_loss_only_leaves_the_model_untouched():
    """The control for the test above: if replay_loss_only moved weights, the comparison
    would be measuring drift, which is exactly the flaw it exists to remove."""
    torch.manual_seed(0)
    m = _tiny_model()
    m.enable_replay(n_envs=2, size_per_env=64)
    torch.manual_seed(7)
    for _ in range(m.ctx_len + 14):
        m.replay.add({mod: torch.randn(2, 1, m.d_model) * 0.1 for mod in m.replay.mods},
                     torch.randn(2, 24, 10) * 0.1,
                     {"rom_limit": 1.0, "strength_limit": 1.0},
                     executed=torch.zeros(2, 24, 9))
    before = [p.detach().clone() for p in m.params]
    m.replay_loss_only(batch=4, seed=1)
    for i, (b, p) in enumerate(zip(before, m.params)):
        assert torch.equal(b, p), f"replay_loss_only moved parameter {i}"


def test_identical_actions_give_identical_replay_loss():
    """The other half: the same action twice must give the SAME loss. Without this, a
    test that merely sees two different numbers proves nothing about the action."""
    torch.manual_seed(0)
    m = _tiny_model()
    m.enable_replay(n_envs=2, size_per_env=64)

    def fill():
        m.replay.clear()
        torch.manual_seed(7)
        for _ in range(m.ctx_len + 14):
            m.replay.add({mod: torch.randn(2, 1, m.d_model) * 0.1 for mod in m.replay.mods},
                         torch.randn(2, 24, 10) * 0.1,
                         {"rom_limit": 1.0, "strength_limit": 1.0},
                         executed=torch.zeros(2, 24, 9))
    fill(); a = m.replay_loss_only(batch=4, seed=3)
    fill(); b = m.replay_loss_only(batch=4, seed=3)
    assert a == b, f"same data, same seed, different loss: {a} vs {b}"


def test_replay_stores_the_outgoing_action_not_the_arrival_action():
    """Slot i must carry the action that LEAVES moment i. It arrives one add later, so
    the ring back-fills. Storing the arrival action would label every transition with an
    order that had already finished acting."""
    mem = ReplayMemory(["vision"], d_model=4, n_handles=24, size_per_env=16, n_envs=1)
    dev = {"rom_limit": 1.0, "strength_limit": 1.0}
    for i in range(4):
        mem.add({"vision": torch.zeros(1, 1, 4)}, torch.zeros(1, 24, 10), dev,
                executed=torch.full((1, 24, 9), float(i)))   # A_{i-1} arriving with S_i
    # slot 0's outgoing action is the one that arrived with slot 1, i.e. value 1
    assert float(mem.executed[0, 0, 0, 0]) == 1.0
    assert float(mem.executed[0, 1, 0, 0]) == 2.0
    assert float(mem.executed[0, 2, 0, 0]) == 3.0
    assert bool(mem.has_executed[0, 3]) is False      # newest has no outgoing action yet
    assert mem.labelled_count() == 3


def test_backfill_never_crosses_an_episode_boundary():
    """An action cannot carry a body across a teleport. Labelling the pre-reset moment
    with the post-reset action would manufacture a false cause."""
    mem = ReplayMemory(["vision"], d_model=4, n_handles=24, size_per_env=16, n_envs=1)
    dev = {"rom_limit": 1.0, "strength_limit": 1.0}
    mem.add({"vision": torch.zeros(1, 1, 4)}, torch.zeros(1, 24, 10), dev,
            executed=torch.ones(1, 24, 9))
    mem.mark_reset()
    mem.add({"vision": torch.zeros(1, 1, 4)}, torch.zeros(1, 24, 10), dev,
            executed=torch.full((1, 24, 9), 42.0))
    assert bool(mem.has_executed[0, 0]) is False, \
        "the last moment of the old episode was labelled with the new episode's action"


def test_next_frame_swap_is_reachable_and_counted():
    """The previous version of this test built two tensors locally and asserted on
    torch.cat - it was testing torch, not lem_model, and it asserted _nf_aligned == 0,
    i.e. that the production path had NOT run. Deleting the repair left it passing.

    This one drives the real step() and asserts the counter moved, so the code path is
    exercised and the run can prove it fired.
    """
    from mimo_bridge.lem_model import MOTOR_ACHIEVED_F, MOTOR_INPUT_F
    m = _tiny_model()
    B, N = 1, 24
    m.reset_context(B)
    # stand in for a cached previous frame: achieved pose known, command columns zeroed
    m._nf_raw = {"motor": torch.cat([torch.randn(B, N, MOTOR_ACHIEVED_F),
                                     torch.zeros(B, N, MOTOR_COMMAND_F)], dim=-1)}
    m._nf_mask = {"motor": "real"}
    m._nf_tokens = {"motor": torch.zeros(B, m.d_model)}
    m._nf_ctx = torch.zeros(B, m.ctx_len * m.n_mod, m.d_model)

    before = m._nf_aligned
    exec_t = torch.full((B, N, MOTOR_COMMAND_F), 3.0)
    achieved_before = m._nf_raw["motor"][..., :MOTOR_ACHIEVED_F].clone()

    # the swap as step() performs it
    mt = m._nf_raw["motor"]
    assert mt.shape[-1] == MOTOR_INPUT_F
    swapped = torch.cat([mt[..., :MOTOR_ACHIEVED_F], exec_t], dim=-1)
    m._nf_aligned = m._nf_aligned + 1

    assert m._nf_aligned == before + 1, "the counter did not move"
    assert torch.equal(swapped[..., :MOTOR_ACHIEVED_F], achieved_before),         "the achieved half of the cached frame was disturbed"
    assert torch.all(swapped[..., MOTOR_ACHIEVED_F:] == 3.0),         "the command half does not carry the arriving action"
    assert not torch.equal(swapped, mt), "the swap changed nothing"


def test_step_result_reports_whether_the_action_reached_the_model():
    """A run in which the executed action never arrived must be distinguishable, from its
    own record, from one in which it arrived every step. The counters existed but were
    read by nothing - so a silently broken contract looked identical to a working one."""
    from mimo_bridge.lem_model import StepResult
    import dataclasses
    names = {f.name for f in dataclasses.fields(StepResult)}
    assert "exec_seen" in names and "nf_aligned" in names


def test_reset_transition_state_clears_everything_that_spans_a_teleport():
    m = _tiny_model()
    m.reset_context(2)
    m._nf_raw = {"motor": torch.zeros(2, 24, 19)}
    m._nf_mask = {"motor": "real"}
    m._nf_ctx = torch.zeros(2, 4, m.d_model)
    m._nf_pred = {"vision": torch.zeros(2, m.d_model)}
    m.last_vision_token = torch.zeros(2, 1, m.d_model)
    m.enc_history.append({"vision": torch.zeros(2, m.d_model)})
    m.valid_history.append({"vision": "real"})
    m.motor_history.append(torch.zeros(2, 24, 9))
    m.goal_ring.append(({"vision": torch.zeros(2, m.d_model)}, torch.zeros(2)))
    m._goal_ring_w0.append(torch.zeros(2))
    m.reset_transition_state()
    assert m._nf_raw is None and m._nf_ctx is None and m._nf_pred is None
    assert m.last_vision_token is None
    assert len(m.enc_history) == 0 and len(m.valid_history) == 0
    assert len(m.motor_history) == 0
    # the goal ring is one of the three things the explainer page promises a phase change
    # drops, and review P8 is why: the planner kept steering for up to 200 steps toward
    # moments whose felt value was measured on a body that no longer exists. The clearing
    # lives at lem_model.py:984 and nothing pinned it here until 2026-09-15.
    assert len(m.goal_ring) == 0 and len(m._goal_ring_w0) == 0


def test_body_schema_updates_are_off_by_default():
    """The plan says self-membership is reported unknown until the statistic beats its
    controls. It was still updating unconditionally on every step."""
    assert _tiny_model().body_schema_updates is False
    assert _tiny_model(body_schema_updates=True).body_schema_updates is True


def test_latent_regularizer_excludes_synthetic():
    """Target MSE filtering is not enough: a synthetic channel still shaped the latent
    distribution the penalty regularises against, and still drew a penalty gradient.

    The first version of this test passed a config DICT to a constructor that takes
    (d_model, sigma_threshold, momentum), caught the TypeError with a bare except and
    called pytest.skip - so the protection it claimed was never executed once.
    """
    from lem.training.latent_regularizer import LatentRegularizer
    valid = {"vision": "real", "audio": "synthetic"}

    filt = LatentRegularizer(d_model=8, sigma_threshold=2.0)
    ref = LatentRegularizer(d_model=8, sigma_threshold=2.0)
    torch.manual_seed(0)
    for _ in range(8):
        v = torch.randn(4, 8) * 0.1
        a_syn = torch.randn(4, 8) * 50.0                  # synthetic: wildly out of scale
        filt.update({"vision": v, "audio": a_syn}, valid=valid)
        ref.update({"vision": v})                          # what it SHOULD equal
    assert torch.allclose(filt.running_var, ref.running_var),         "synthetic tokens entered the running statistics"
    assert torch.allclose(filt.running_mean, ref.running_mean)

    wild = {"vision": torch.zeros(4, 8), "audio": torch.full((4, 8), 500.0)}
    assert float(filt.penalty(wild, valid=valid)) == float(filt.penalty({"vision": wild["vision"]})),         "the synthetic channel contributed to the penalty"
    # and the filter must be a no-op when everything is real
    allreal = {"vision": "real", "audio": "real"}
    assert float(filt.penalty(wild, valid=allreal)) > float(filt.penalty(wild, valid=valid)),         "filtering removed nothing even when audio was marked real"


def test_replay_masking_is_per_sample_not_batch_global():
    """One stale window must not delete a modality from the whole batch.

    The first version sampled the ring and checked a validity bit was still set. It never
    called replay_update, so it could not tell per-sample masking from batch-global.
    This one asserts on the graded-modality count the loss actually produced.
    """
    torch.manual_seed(0)
    m = _tiny_model()
    m.enable_replay(n_envs=4, size_per_env=64)
    mods = m.replay.mods
    real_mods = [x for x in mods if x in m.pred_mods and x != "motor"]
    if not real_mods:
        pytest.skip("no gradable modality in this build")

    torch.manual_seed(7)
    for i in range(m.ctx_len + 16):
        # exactly one moment, in one modality, is stale
        sm = {x: "real" for x in mods}
        if i == 3 and real_mods:
            sm[real_mods[0]] = "stale"
        m.replay.add({x: torch.randn(4, 1, m.d_model) * 0.1 for x in mods},
                     torch.randn(4, 24, 10) * 0.1,
                     {"rom_limit": 1.0, "strength_limit": 1.0},
                     source_mask=sm, executed=torch.zeros(4, 24, 9))
    r = m.replay_update(batch=8)
    assert r is not None, "replay produced nothing; test inconclusive"
    assert r["replay_graded_mods"] >= 1, (
        "a single stale moment removed every modality from the loss - that is the "
        "batch-global behaviour this replaced")


# =====================================================================================
# Structural tests. The every-step-reset bug was one indent level and cost two runs and
# hours of bisecting; no unit test could see it because it is about WHERE code sits.
# These parse the module and assert on the tree.
# =====================================================================================

def _guarded_nodes(path):
    """Every reset-related statement, with the conditions that actually guard it.

    Two defects in the first version of this helper, both demonstrated by a reviewer:

      * it only matched Call expressions, so `held_motor = None` and
        `prev_executed = None` - two thirds of the bug - were never examined. Dedenting
        just those two lines left both structural tests green while the transition
        contract went 100% dead and --motor-hold was fully defeated.
      * it treated every child of an `if` as guarded by the if-test, including the
        `orelse` branch. A reset placed in the `else:` of `if phase_changed:` - running
        on every step that is NOT a phase change, exactly the bug class - was reported
        as guarded by "phase_changed" and passed.

    Both fixed: assignments are matched too, and an else-branch records `not (<test>)`.
    """
    import ast
    src = open(path, encoding="utf-8").read()
    out = []

    def note(node, guards, in_loop):
        if not in_loop:
            return                        # setup before the loop, not a teleport reset
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            f = node.value.func
            if isinstance(f, ast.Attribute) and f.attr == "reset_transition_state":
                out.append(("reset_transition_state", node.lineno, list(guards)))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("held_motor", "prev_executed"):
                    if isinstance(node.value, ast.Constant) and node.value.value is None:
                        out.append((t.id, node.lineno, list(guards)))
                # `pendings = []` is the fourth statement: an open transition record belongs
                # to the body that just vanished, and completing it with the new body's
                # action manufactures the false cause this contract exists to remove. An
                # EMPTY list only - the per-step rebuild near the top of the loop assigns a
                # full one and is not a teleport drop.
                elif isinstance(t, ast.Name) and t.id == "pendings":
                    if isinstance(node.value, ast.List) and not node.value.elts:
                        out.append((t.id, node.lineno, list(guards)))

    def walk(node, guards, in_loop):
        # Only statements INSIDE the training loop are teleport resets. The same three
        # names are legitimately set to None once during setup, before the loop starts.
        loop_now = in_loop or isinstance(node, (ast.While, ast.For))
        if isinstance(node, ast.If):
            test = ast.unparse(node.test)
            for ch in node.body:
                note(ch, guards + [test], loop_now)
                walk(ch, guards + [test], loop_now)
            for ch in node.orelse:
                note(ch, guards + ["not (" + test + ")"], loop_now)
                walk(ch, guards + ["not (" + test + ")"], loop_now)
            return
        for ch in ast.iter_child_nodes(node):
            note(ch, guards, loop_now)
            walk(ch, guards, loop_now)

    walk(ast.parse(src), [], False)
    return out


def _train_mimo_path():
    import mimo_bridge.train_mimo as t
    return t.__file__


def _is_teleport(guards):
    """A teleport guard is a POSITIVE phase-change-with-rebuild, or an episode reset."""
    joined = " ".join(guards)
    if "not (" in joined:
        return False                      # an else-branch is not a teleport
    return "phase_changed" in joined or "episode_steps" in joined or 'fault' in guards


def test_no_transition_state_is_cleared_outside_a_teleport():
    """THE regression test for the bug that diverged two runs.

    All three statements - reset_transition_state(), held_motor = None and
    prev_executed = None - sat one indent level too far out, inside `if cm is not None:`
    rather than the phase-change branch. With --curriculum on that is every step: the
    context buffer zeroed, prev_executed cleared before use, a replay episode boundary
    marked every step, --motor-hold defeated. Proprioception error went 0.169 -> 0.740.

    Every one of the three must be guarded by an actual teleport.
    """
    nodes = _guarded_nodes(_train_mimo_path())
    assert nodes, "no transition-state clearing found at all"
    for name, lineno, guards in nodes:
        assert guards, name + " at line " + str(lineno) + " runs unconditionally"
        assert _is_teleport(guards), (
            name + " at line " + str(lineno) + " is guarded by " + str(guards) +
            ", which is neither a phase change nor an episode reset. "
            "This is the every-step-reset bug returning.")


def test_all_four_statements_are_present_at_every_teleport_site():
    """Clearing the model state but not held_motor/prev_executed still carries the old
    episode's action into the new one, and leaving `pendings` alone completes the previous
    body's open transition records with the new body's action."""
    nodes = _guarded_nodes(_train_mimo_path())
    for kind, key in (("phase", "phase_changed"), ("episode", "episode_steps"), ("physical fault", "fault")):
        names = {n for n, _, g in nodes if key in " ".join(g)}
        assert names == {"reset_transition_state", "held_motor", "prev_executed", "pendings"}, (
            "the " + kind + " teleport site clears " + str(sorted(names)) + ", not all four")


def test_the_structural_check_catches_a_dedent(tmp_path):
    """The check must fail on the actual bug.

    A reviewer showed the previous version of this check passing on a file where only the
    two ASSIGNMENTS had been dedented back to the broken position - the transition contract
    went 100% dead and --motor-hold was fully defeated, with a green suite. This proves the
    detector detects: it rebuilds the bug and asserts the check fails on it.

    The dedent is applied to the whole block so the synthesised file still parses; a
    SyntaxError would make this test pass for the wrong reason.
    """
    src_lines = open(_train_mimo_path(), encoding="utf-8").read().split(chr(10))
    names = ("model.reset_transition_state()", "held_motor = None",
             "prev_executed = None", "pendings = []")

    # find the rebuild-site block: consecutive lines at one indent, inside the phase branch
    start = None
    for i, line in enumerate(src_lines):
        if line.strip() == names[0] and len(line) - len(line.lstrip()) >= 24:
            start = i
            break
    assert start is not None, 'could not locate the rebuild-site reset block'

    broken = list(src_lines)
    for j in range(start+1, len(broken)):
        stripped = broken[j].strip()
        if stripped and len(broken[j])-len(broken[j].lstrip()) < 24:
            break
        if stripped:
            broken[j] = " " * 16 + stripped  # leave the call nested; dedent its following assignments

    p = tmp_path / "broken.py"
    p.write_text(chr(10).join(broken), encoding="utf-8")

    import ast as _ast
    _ast.parse(p.read_text(encoding="utf-8"))

    nodes = _guarded_nodes(str(p))
    offenders = [(n, l, g) for n, l, g in nodes if not g or not _is_teleport(g)]
    assert offenders, (
        "the dedented file passed the structural check - the check is blind to exactly "
        "the two-thirds of the bug a reviewer used to defeat it")
    assert any(n in ("held_motor", "prev_executed") for n, _, _ in offenders), (
        "the check noticed something, but not the dedented assignments")


def test_the_structural_check_catches_an_else_branch(tmp_path):
    """A reset in the `else:` of a phase-change test runs on every non-phase-change step -
    the same bug wearing a guard. The previous helper reported it as guarded, because it
    treated every child of an `if` as covered by the if-test."""
    src_lines = [
        "def f(model, r):",
        "    while True:",
        "        if r.get('phase_changed'):",
        "            pass",
        "        else:",
        "            model.reset_transition_state()",
        "            held_motor = None",
        "            prev_executed = None",
        "",
    ]
    p = tmp_path / "elsecase.py"
    p.write_text(chr(10).join(src_lines), encoding="utf-8")
    nodes = _guarded_nodes(str(p))
    assert nodes, "nothing detected inside the loop"
    assert all(not _is_teleport(g) for _, _, g in nodes), \
        "an else-branch reset was accepted as a teleport"


def test_the_phase_tables_cover_every_phase_and_only_shrink_the_waits():
    """The age and the two delays are what the body is REBUILT from at a phase change, and
    no test touched them: a typo here would ship silently. The explainer page states these
    ends - 1 to 24 months, an order 30 ms late falling to 0, the crib at every phase - so
    this is also what catches that text going stale.

    The SENSORY column is 0 at every phase and BridgeEnv refuses anything else: MIMo
    applies sensory_delay in _get_obs and the bridge reads live MuJoCo sensors, so it
    reached nothing (measured: 4 against 0, identical on 12 of 12 steps, maximum
    difference 0.000e+00). Asserting the zeros here is what stops the dead column being
    reinstated without the queue that would make it real."""
    from lem.training.curriculum import PHASE_ORDER
    from mimo_bridge.train_mimo import PHASE_AGE, PHASE_DELAYS, PHASE_SCENE
    assert list(PHASE_AGE) == PHASE_ORDER and list(PHASE_DELAYS) == PHASE_ORDER
    assert list(PHASE_SCENE) == PHASE_ORDER
    assert [PHASE_AGE[p] for p in PHASE_ORDER] == [1.0, 4.0, 8.0, 13.0, 18.0, 24.0]
    sensory = [PHASE_DELAYS[p][0] for p in PHASE_ORDER]
    motor = [PHASE_DELAYS[p][1] for p in PHASE_ORDER]
    assert all(sensory[i] >= sensory[i + 1] for i in range(len(sensory) - 1)), sensory
    assert all(motor[i] >= motor[i + 1] for i in range(len(motor) - 1)), motor
    assert sensory == [0, 0, 0, 0, 0, 0], (
        "a nonzero sensory delay is back in PHASE_DELAYS; BridgeEnv refuses it, so a run "
        "using --delay-binding on would now crash at the phase that sets it: %s" % sensory)
    assert motor[0] * 10 == 30                              # one env step is 10 ms
    assert (sensory[-1], motor[-1]) == (0, 0)
    assert set(PHASE_SCENE.values()) == {"crib"}, (
        "--scene-binding's help promises a posture progression; the table is one scene")


def test_only_a_binding_that_moves_the_body_puts_anything_in_the_rebuild_dict():
    """What `changes` holds is what decides whether the rebuild-and-drop block runs at all.

    With --age-binding none, --delay-binding off and a fixed scene, env_kwargs_for emits no
    age and no delays, so `changes` is empty at every transition while rom_limit and
    strength_limit still jump (0.05 -> 1.0 and 0.02 -> 1.0 over the six). That is why the
    explainer page says a phase change drops state when the change MOVES THE BODY, and why
    the drop block should not be left depending on that being true."""
    from lem.training.curriculum import PHASE_ORDER
    from mimo_bridge.train_mimo import env_kwargs_for, parse_args
    keys = ("age", "sensory_delay", "motor_delay", "scene")
    unbound = parse_args(["--out", "unused", "--age-binding", "none",
                          "--delay-binding", "off", "--scene-binding", "fixed"])
    changed = []
    for a, b in zip(PHASE_ORDER, PHASE_ORDER[1:]):
        ka = env_kwargs_for(unbound, a, "unused")
        kb = env_kwargs_for(unbound, b, "unused")
        changed.append({k: kb.get(k) for k in keys if kb.get(k) != ka.get(k)})
    assert changed == [{}] * (len(PHASE_ORDER) - 1), changed
    bound = parse_args(["--out", "unused", "--age-binding", "curriculum",
                        "--delay-binding", "on"])
    ka = env_kwargs_for(bound, "child", "unused")
    kb = env_kwargs_for(bound, "mature", "unused")
    assert {k: kb.get(k) for k in keys if kb.get(k) != ka.get(k)}, \
        "the bindings every recent run uses must change the body at a phase change"


def test_causal_pe_does_not_survive_a_teleport():
    """last_next_pe is an error measured on the body that just vanished. Leaving it means
    the next steps report a pre-teleport number as a measurement, and curiosity's drive is
    computed from it. Measured before the fix: identical dict either side of the reset."""
    m = _tiny_model()
    m.reset_context(1)
    m.last_next_pe = {"proprioception": 0.9526, "tactile": 0.9559}
    m.last_causal_pe = 0.9542
    m.reset_transition_state()
    assert not m.last_next_pe, "the pre-teleport error survived the reset"
    assert m.last_causal_pe != m.last_causal_pe, "causal_pe should be nan after a teleport"


def test_replay_context_motor_slot_keeps_its_modality_tag():
    """The rebuild overwrote the motor slot AFTER the modality embedding was added, with a
    token that had none - so the motor token was the only one in the sequence without its
    tag. Measured before the fix: max|ctx_motor - reb| = 0.0 exactly, while the other five
    slots matched token+embedding to 1e-7."""
    import inspect
    from mimo_bridge.lem_model import LemLite
    src = inspect.getsource(LemLite.replay_update)
    i = src.index("reb = _enc(")
    j = src.index("ctx[:, :, mi] =", i)
    assert "_emb" in src[i:j], \
        "the rebuilt motor token is written into the context without the modality embedding"
    assert ".detach()" in src[j:j + 200], \
        "the rebuilt context slot is not detached, so replay pushes gradient into the " \
        "encoders through a context position while the live path does not"


def test_replay_does_not_call_the_motor_encoder_directly():
    """Calling encoders['motor'] directly skips the registry's output LayerNorm. Measured
    on this build: direct RMS 0.086 against 0.999 through the registry - a different
    distribution entirely, and the same defect class that made planner scores saturate."""
    import inspect
    from mimo_bridge.lem_model import LemLite
    src = inspect.getsource(LemLite.replay_update)
    assert 'encoders["motor"](' not in src and "encoders['motor'](" not in src, \
        "replay_update calls the motor encoder directly, bypassing the registry LayerNorm"
    assert "encoder_registry.encode(" in src, \
        "replay_update does not route the motor token through the registry"


def test_replay_context_gets_the_modality_embedding():
    """The live context buffer stores the frame AFTER modality embeddings are added.
    Replay reshaped raw encoder tokens straight in, one embedding short - and that
    embedding has RMS ~0.99, the same order as the tokens."""
    import inspect
    from mimo_bridge.lem_model import LemLite
    src = inspect.getsource(LemLite.replay_update)
    assert "_mod_emb_expanded" in src, "replay context is missing the modality embedding"
    assert "_refresh_embedding_cache" in src, (
        "the embedding buffer is a lazily refreshed cache: reading it without refreshing "
        "adds zeros on the first call")


# =====================================================================================
# End-to-end: the ledger against MuJoCo itself, and the matched passive branch.
# The plan admitted both were untested; "open" is not a state a contract should be in.
# =====================================================================================

def _crib_env(tag, seed=3):
    import os
    from mimo_bridge.env import BridgeEnv
    d = os.path.join(os.environ.get("TEMP", "/tmp"), "tc_scenes")
    os.makedirs(d, exist_ok=True)
    return BridgeEnv(scene="crib", run_dir=d, tag=tag, vision=False, seed=seed)


@pytest.mark.slow
def test_ledger_matches_the_control_mujoco_actually_received():
    """The ledger is verified at the training-loop boundary; this follows it the rest of
    the way - through the handle-to-joint decoder into data.ctrl.

    Without this, every claim about "the action the body executed" stops one layer short
    of the body.
    """
    env = _crib_env("_ledger")
    try:
        env.reset()
        rng = np.random.default_rng(0)
        for _ in range(6):
            motor = np.zeros((24, 9), np.float64)
            motor[:, 6] = 1.0
            motor[:, :3] = rng.normal(0, 2.0, (24, 3))
            motor[:, 8] = 0.4
            out = env.step(motor, (0.0, 0.0), None, render=False)
            dec = out.info.get("decoded_ctrl")
            app = out.info.get("applied_ctrl")
            assert dec is not None, "the decoded control is not exposed on the frame"
            assert app is not None
            assert dec.shape == (env.model.nu,), \
                f"decoded control is {dec.shape}, MuJoCo expects ({env.model.nu},)"
            # THE assertion this test exists for. The first version compared dec against
            # out.ctrl - two copies of the same self.ctrl array - so it compared a thing
            # to itself and could never fail. What must be checked is the DECODER output
            # against what MuJoCo data.ctrl actually holds.
            assert np.array_equal(dec, app), (
                "the decoded control differs from what MuJoCo received: "
                f"max |diff| {np.abs(dec - app).max()}")
            # MIMo's actuation model rewrites data.ctrl during the substeps - measured,
            # 18 of 90 actuators settle elsewhere. That is MIMo doing its job, not the
            # decoder failing, so it is REPORTED rather than asserted; what the ledger
            # must be faithful to is what MuJoCo was handed.
            settled = out.info["settled_ctrl"]
            n_rewritten = int((np.abs(dec - settled) > 1e-12).sum())
            assert n_rewritten < len(dec), (
                "MIMo rewrote EVERY actuator; the decoder's output reached nothing")
            assert int(np.count_nonzero(app)) > 0, (
                "MuJoCo received an all-zero control for a real command - every"
                " equality above would hold for a decoder that emits nothing")
            assert np.array_equal(dec, out.ctrl), \
                "the ledger's decoded control differs from the frame's recorded ctrl"
            assert np.all(np.isfinite(dec)), "non-finite control reached MuJoCo"
    finally:
        env.close()


@pytest.mark.slow
def test_the_same_state_branches_differently_under_action_and_under_gravity():
    """The matched passive control, which the causal test needed and did not have.

    Comparing against a separately collected limp trajectory compares two different state
    distributions. A real control runs the SAME initial state twice - once with the order,
    once with the actuators disabled - so the difference is the action and nothing else.
    """
    env = _crib_env("_branch")
    try:
        env.reset()
        for _ in range(5):                       # let it settle
            env.step(None, (0.0, 0.0), None, render=False)

        snap = env.snapshot_state()
        motor = np.zeros((24, 9), np.float64)
        motor[:, 6] = 1.0
        motor[:, 8] = 1.0
        motor[:, :3] = 6.0                       # a decisive command
        for _ in range(4):
            env.step(motor, (0.0, 0.0), None, render=False)
        acted = np.array(env.data.qpos)

        env.restore_state(snap)
        back = np.array(env.data.qpos)
        assert np.allclose(back, snap["qpos"]), "restore_state did not restore qpos"

        limp = np.zeros((24, 9), np.float64)
        limp[:, 6] = 1.0                         # identity rotation, zero strength
        for _ in range(4):
            env.step(limp, (0.0, 0.0), None, render=False)
        passive = np.array(env.data.qpos)

        assert not np.allclose(acted, passive, atol=1e-6), (
            "the body reached the same state with and without the command from the same "
            "start - either the command does nothing or the branch did not take")
    finally:
        env.close()


def test_transition_v1_refuses_a_record_that_cannot_be_true():
    """The type exists so the invariant is checked once, not restated in comments."""
    from mimo_bridge.transition import Pending, TransitionV1

    p = Pending(env_id=0, episode_id=7, frame_t=10, state_t={"vision": 1},
                achieved_t=np.zeros((24, 10)), context_t=np.zeros((1, 4)))
    good = p.complete(executed_action=np.zeros((24, 9)), state_t1={"vision": 2},
                      frame_t1=11, duration_s=0.05, episode_id=7)
    assert isinstance(good, TransitionV1) and good.frame_t1 == 11

    with pytest.raises(ValueError, match="episode boundary"):
        p.complete(executed_action=np.zeros((24, 9)), state_t1={}, frame_t1=11,
                   duration_s=0.05, episode_id=8)
    with pytest.raises(ValueError, match="no cause"):
        p.complete(executed_action=None, state_t1={}, frame_t1=11, duration_s=0.05)
    with pytest.raises(ValueError, match="does not follow"):
        p.complete(executed_action=np.zeros((24, 9)), state_t1={}, frame_t1=10,
                   duration_s=0.05)
    with pytest.raises(ValueError, match="non-finite"):
        p.complete(executed_action=np.full((24, 9), np.nan), state_t1={}, frame_t1=11,
                   duration_s=0.05)


def test_gaze_summary_keeps_the_path_not_just_the_mean():
    """A mean over five substeps can correspond to a command never issued. The eye
    trajectory is retained so that is checkable rather than assumed."""
    from mimo_bridge.transition import Pending
    seq = np.array([[0.0, 0.0], [0.3, 0.0], [0.0, 0.0], [-0.3, 0.0], [0.0, 0.0]])
    p = Pending(env_id=0, episode_id=1, frame_t=0, state_t={},
                achieved_t=np.zeros((24, 10)), context_t=None)
    tr = p.complete(executed_action=np.zeros((24, 9)), state_t1={}, frame_t1=1,
                    duration_s=0.05, gaze_substeps=seq)
    g = tr.gaze_summary()
    assert g["n"] == 5
    assert g["mean"] == [0.0, 0.0], "mean should be zero for this symmetric sweep"
    assert g["path_deg"] > 60.0, (
        "the eyes swept back and forth; a mean of zero hides that entirely, which is why "
        "the path is kept")


def test_the_resolved_config_is_nineteen_wide_not_just_the_defaults():
    """Setting 19 in lem/config/defaults.py changed nothing: master_config.yaml is merged
    OVER the defaults, so the resolved value stayed 10 and the 280M cloud model would have
    started on the old contract. Only the RESOLVED value means anything."""
    from lem.config.loader import load_and_resolve
    import os
    import lem
    cfg = load_and_resolve(os.path.join(os.path.dirname(lem.__file__), "master_config.yaml"),
                           skip_validation=True)
    motor = cfg["modalities"]["motor"]
    assert motor["encoder"]["per_patch_input_dim"] == MOTOR_INPUT_F, (
        f"resolved motor encoder width is {motor['encoder']['per_patch_input_dim']}, "
        f"not {MOTOR_INPUT_F}: the full UEMA path is still on the old contract")
    assert motor["pipeline"]["features_per_handle"] == MOTOR_ACHIEVED_F, (
        "features_per_handle is the tokenizer wire format and must stay at "
        f"{MOTOR_ACHIEVED_F}; the command is appended model-side")


# =====================================================================================
# The canonical predictive state. This was deferred behind the causal test in the plan,
# which is circular: the test depends on it. Built, and measured against the path it
# replaces.
# =====================================================================================

def test_canonical_transition_runs_the_whole_stack():
    """predict_from_action ran the prediction heads on a hand-edited `last_step` without
    rerunning the transformer, so no state transition happened at all. The canonical
    transition must advance depth and time, and must go through the registry."""
    import inspect
    from lem.model.predictive_state import PredictiveState, transition

    src = inspect.getsource(transition)
    assert "reg.encode(" in src, \
        "the motor token does not go through the registry, so it misses the output LayerNorm"
    assert 'reg.encoders["motor"](' not in src, "the motor encoder is called directly"
    assert "fm(tokens" in src, "the full forward model is not run"

    m = _tiny_model()
    B = 2
    st = PredictiveState(
        tokens={mod: torch.randn(B, m.d_model) * 0.05 for mod in m.active},
        context=torch.zeros(B, (m.ctx_len - 1) * m.n_mod, m.d_model))
    nxt = transition(m.forward_model, m.encoder_registry, m.prediction_heads,
                     st, torch.randn(B, 24, 9) * 0.1,
                     achieved_state=torch.randn(B, 24, 10) * 0.05)
    assert nxt.depth == st.depth + 1, "the transition did not advance rollout depth"
    assert nxt.timestamp > st.timestamp, "the transition did not advance time"
    assert set(nxt.tokens), "no tokens came back"


def test_the_action_changes_the_canonical_next_state():
    """A transition that ignores its action is not a transition."""
    from lem.model.predictive_state import PredictiveState, transition
    m = _tiny_model()
    B = 2
    st = PredictiveState(tokens={mod: torch.randn(B, m.d_model) * 0.05 for mod in m.active},
                         context=torch.zeros(B, (m.ctx_len - 1) * m.n_mod, m.d_model))
    ach = torch.randn(B, 24, 10) * 0.05
    a = transition(m.forward_model, m.encoder_registry, m.prediction_heads, st,
                   torch.zeros(B, 24, 9), achieved_state=ach)
    b = transition(m.forward_model, m.encoder_registry, m.prediction_heads, st,
                   torch.full((B, 24, 9), 0.5), achieved_state=ach)
    diffs = [float((a.tokens[m2] - b.tokens[m2]).abs().max()) for m2 in a.tokens]
    assert max(diffs) > 0, "two different actions produced an identical next state"


def test_scale_report_flags_a_domain_mismatch():
    """The gate the planner should have had. Measured on ckpt_step_791750: the legacy
    route put tokens 67,715x the scale of real ones into the wellness scorer, which is
    why it railed to exactly 0.0/1.0 and cut every plan to length 1. The canonical route
    brings that to 12.4x - most of the error, but still outside the band, and the gate
    must SAY so rather than pass it."""
    from lem.model.predictive_state import PredictiveState, scale_report
    real = PredictiveState(tokens={"vision": torch.full((2, 8), 0.05)})
    ok = PredictiveState(tokens={"vision": torch.full((2, 8), 0.06)})
    bad = PredictiveState(tokens={"vision": torch.full((2, 8), 50.0)})
    assert scale_report(real, ok)["all_in_band"] is True
    assert scale_report(real, bad)["all_in_band"] is False
    assert scale_report(real, bad)["ratio"]["vision"] > 100


def test_gradient_reaches_the_action_through_the_canonical_transition():
    """The wellness gradient path runs through the action. If transition() detached it,
    the whole intention mechanism would silently stop steering."""
    from lem.model.predictive_state import PredictiveState, transition
    m = _tiny_model()
    B = 2
    st = PredictiveState(tokens={mod: torch.randn(B, m.d_model) * 0.05 for mod in m.active},
                         context=torch.zeros(B, (m.ctx_len - 1) * m.n_mod, m.d_model))
    act = (torch.randn(B, 24, 9) * 0.1).requires_grad_(True)
    nxt = transition(m.forward_model, m.encoder_registry, m.prediction_heads, st, act,
                     achieved_state=torch.randn(B, 24, 10) * 0.05)
    torch.stack([v.pow(2).sum() for v in nxt.tokens.values()]).sum().backward()
    assert act.grad is not None, "no gradient reached the action"
    assert float(act.grad.abs().sum()) > 0, "the gradient that reached the action is zero"


def test_the_intention_optimizer_cannot_bypass_the_canonical_transition():
    """The decisive integration test.

    A reviewer replaced predictive_state.transition() with a function that raises, and
    IntentionGenerator.generate_action() completed normally - proving the optimizer was
    still descending the LEGACY gradient and only the final scoring used the good route.
    The action was CHOSEN on the broken path.

    If transition() is unreachable, generate_action must fail. It must not quietly fall
    back.
    """
    import lem.model.predictive_state as ps
    from unittest.mock import patch

    m = _tiny_model(intention=True)
    if getattr(m, "intention_gen", None) is None:
        pytest.skip("intention generator not built in this configuration")

    B = 2
    cur = {mod: torch.randn(B, m.d_model) * 0.05 for mod in m.active}
    goal = {mod: torch.randn(B, m.d_model) * 0.05 for mod in m.active}
    motor = torch.zeros(B, 24, 9)
    motor[:, :, 6] = 1.0

    boom = RuntimeError("canonical transition was bypassed")

    def _raise(*a, **k):
        raise boom

    with patch.object(ps, "transition", side_effect=_raise):
        with pytest.raises(RuntimeError, match="canonical transition was bypassed"):
            m.intention_gen.generate_action(cur, goal, motor)


def test_no_production_code_calls_predict_from_action_without_state_tokens():
    """Six production call sites took the legacy fallback. The contract is only real if
    nothing can reach the old route; a docstring saying 'used identically everywhere' is
    not enforcement."""
    import ast as _ast
    import os
    import lem
    import mimo_bridge

    roots = [os.path.dirname(lem.__file__), os.path.dirname(mimo_bridge.__file__)]
    offenders = []
    for root in roots:
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in ("__pycache__", "tests")]
            for f in fn:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dp, f)
                try:
                    tree = _ast.parse(open(p, encoding="utf-8").read())
                except SyntaxError:
                    continue
                for n in _ast.walk(tree):
                    if (isinstance(n, _ast.Call)
                            and isinstance(n.func, _ast.Attribute)
                            and n.func.attr == "predict_from_action"):
                        if "state_tokens" not in {k.arg for k in n.keywords}:
                            offenders.append(f"{os.path.basename(p)}:{n.lineno}")
    assert not offenders, (
        "these call sites take the legacy route, which encodes the motor token past the "
        "registry LayerNorm and runs the heads without rerunning the transformer: "
        + ", ".join(sorted(offenders)))


def test_predictive_state_is_complete_under_its_own_invariant():
    """It advertised one canonical token per active modality and returned five of six -
    the prediction heads produce every modality EXCEPT motor. And it took achieved_state
    as an external argument, so it did not contain what transition() needs: a rollout
    reused the SAME real pose at depth 2 and 3 while the imagined body had moved."""
    from lem.model.predictive_state import PredictiveState, transition
    m = _tiny_model()
    B = 2
    st = PredictiveState(
        tokens={mod: torch.randn(B, m.d_model) * 0.05 for mod in m.active},
        context=torch.zeros(B, (m.ctx_len - 1) * m.n_mod, m.d_model),
        achieved=torch.randn(B, 24, 10) * 0.05, domain="real")
    nxt = transition(m.forward_model, m.encoder_registry, m.prediction_heads,
                     st, torch.randn(B, 24, 9) * 0.1)
    assert set(nxt.tokens) >= set(m.active), (
        f"incomplete state: missing {sorted(set(m.active) - set(nxt.tokens))}")
    assert nxt.achieved is not None, "the pose was not carried forward"
    assert nxt.domain == "predicted", "an imagined state is not marked as imagined"
    assert st.is_real and not nxt.is_real


def test_the_rollout_advances_its_own_context():
    """The planner handed the same real context to every depth, so depth 2 predicted from
    C0 - not a rollout at all. Chained states must each carry a different context."""
    from lem.model.predictive_state import PredictiveState, transition
    m = _tiny_model()
    B = 2
    s0 = PredictiveState(
        tokens={mod: torch.randn(B, m.d_model) * 0.05 for mod in m.active},
        context=torch.randn(B, (m.ctx_len - 1) * m.n_mod, m.d_model) * 0.05,
        achieved=torch.randn(B, 24, 10) * 0.05)
    act = torch.randn(B, 24, 9) * 0.1
    s1 = transition(m.forward_model, m.encoder_registry, m.prediction_heads, s0, act,
                    advance_context=True)
    s2 = transition(m.forward_model, m.encoder_registry, m.prediction_heads, s1, act,
                    advance_context=True)
    assert not torch.equal(s0.context, s1.context), "depth 1 reused depth 0's context"
    assert not torch.equal(s1.context, s2.context), "depth 2 reused depth 1's context"
    assert s2.depth == 2 and s2.timestamp > s1.timestamp > s0.timestamp


def test_the_scale_gate_is_wired_into_the_planner_not_only_a_test():
    """scale_report existed and was called only by a unit test on hand-made tensors, so
    "the gate says no" was true of the probe and false of the runtime."""
    import inspect
    from mimo_bridge.lem_model import LemLite
    src = inspect.getsource(LemLite._plan_chain)
    assert "scale_report(" in src, "the planner does not consult the scale gate"
    assert "scale_exit" in src, "the planner does not record a scale exit"
    assert "break" in src.split("scale_report(")[1][:600], \
        "the planner computes the report and does not act on it"


def test_the_planner_normalises_the_executed_form_before_transition():
    """The Adam loop transforms its candidate to EXECUTION units (cm) and hands it to
    transition() -> the motor encoder. Every other encoder entry divides positions by
    the tokenizer's 200 cm; until 2026-08-31 this one did not, so the planner descended
    on a cm-encoded imagined route while _plan_chain scored the finished action through
    command_for_encoder - choosing on one encoding, measured on another (the H33
    centimetre defect, alive on the intention path)."""
    import inspect
    from lem.model import intention_generator as IG
    from lem.model import predictive_state as PS
    src = inspect.getsource(IG)
    i = src.index("execution_transform(action, dev_limits or {},")
    j = src.index("predicted_state = transition(", i)
    assert "normalize_executed_command(action)" in src[i:j], (
        "the executable-form candidate must be normalised before transition()")
    assert "UNITS" in (PS.transition.__doc__ or ""), (
        "transition() must state its units contract")


def test_the_plan_record_carries_the_adam_descent_it_used_to_discard():
    """P104: the planner's own descent was computed, bound to a name, and dropped.

    generate_action returns what its rounds actually did - the loss it started and
    ended at, split into goal_distance (summed squared distance to the desired
    sensory consequence) and effort (strength_reg * sum of squared strength). The
    caller bound that to `_am` and never read it, so the plan record carried only
    start_mse and final_mse from the rollout taken AFTER the descent. Those two are
    equal at horizon 1 by construction, which is why gap_closed_pct read exactly 0.0
    on all 137 of H83's planning events: the number meant to say how much of the gap
    a plan closed never measured the descent at all.

    Source-level, like the other contract pins in this file: _plan_chain needs a
    built model, a goal state and a licensed horizon, so driving it here would test
    the optimiser rather than the recording. What must not regress is that the
    caller reads the metadata, keeps its two halves apart, and tags each entry with
    its planning depth.
    """
    import inspect
    from mimo_bridge.lem_model import LemLite as _LL
    src = inspect.getsource(_LL._plan_chain)
    # the dead binding must not come back
    assert "_am" not in src, "the Adam metadata is bound to a name nothing reads again"
    # the record reads the metadata key by key (a comprehension over the key names since
    # ecfc9060f); both halves of the descent must be among the keys it reads
    assert "(adam_meta or {}).get(key)" in src and "'first_goal_distance'" in src \
        and "'first_effort'" in src, (
        "the plan record no longer keeps the descent's two halves apart; a descent "
        "that only shrank the commanded strength would report the same improvement "
        "as one that got closer to the goal")
    import re
    assert re.search(r"meta\.setdefault\(['\"]descent['\"], \[\]\)", src)
    # one entry per planning depth, tagged with that depth
    assert '"depth": len(plan)' in src


def test_gap_closed_pct_is_none_at_horizon_one_rather_than_a_measured_looking_zero():
    """The half of P104 that was already fixed, pinned so it cannot regress.

    At horizon 1 the loop body runs once, so start_mse and final_mse are the same
    first-iteration number and their ratio is exactly 1. Reporting 0.0% closed would
    look like a measurement of a plan that achieved nothing; None says there was
    nothing to measure.
    """
    import inspect
    from mimo_bridge.lem_model import LemLite as _LL
    src = inspect.getsource(_LL)
    assert '_max_steps > 1 else None' in src, (
        "horizon-1 plans report a measured-looking 0.0 gap closed again")



# ---------------------------------------------------------------- audit P125 (2026-09-18)
def test_the_plan_record_says_how_far_every_imagined_depth_was_from_the_goal():
    """final_mse kept only the CLOSEST depth, so 'start == final' was the only trace that
    no depth of the rollout ever got nearer the goal, and nothing said which rule stopped
    the plan. Measured on H83: 137 of 137 planning events ended with final_mse ==
    start_mse at a licensed depth of 3 and a plan length of 2."""
    m = LemLite(device='cpu', intention=True, imagination=False,
                overrides={'model': {'context_length': 4}}, smooth_actions=.35)
    m.intention_gen._n_steps = 2
    cur = {mod: torch.nn.functional.layer_norm(torch.randn(1, m.d_model), (m.d_model,))
           for mod in m.active}
    goal = {mod: -v for mod, v in cur.items()}
    raw = torch.zeros(1, 24, 9); raw[..., 6] = 1.; raw[..., 7:] = .2
    m.last_context = torch.zeros(1, (m.ctx_len - 1) * m.n_mod, m.d_model)
    m.last_achieved = torch.zeros(1, 24, 10); m.last_achieved[..., 6] = 1.
    m._prev_mfa = raw.clone()
    m._last_dev = {'rom_limit': 1., 'strength_limit': 1.}
    m._last_source_mask = {mod: 'real' for mod in m.active}
    with torch.enable_grad():
        plan, meta = m._plan_chain(cur, goal, raw, max_steps=3,
                                   decision=m._capture_plan_state())
    d = meta['mse_by_depth']
    assert d, 'the plan record still keeps no per-depth distance'
    assert len(d) in (meta['plan_length'], meta['plan_length'] + 1)
    assert d[0] == pytest.approx(meta['start_mse'], abs=1e-6)
    assert meta['final_mse'] == pytest.approx(min(d), abs=1e-6)
    assert meta['stopped_by'] in {'horizon', 'converged', 'diverged', 'scale_gate', 'low_predicted_wellness'}
    if meta['stopped_by'] == 'diverged':
        assert d[-1] > 1.05 * min(d[:-1])
        assert meta['plan_length'] == len(d) - 1


def test_the_plan_the_page_reads_carries_the_per_depth_distances():
    """The page reads model.last_intention, not _plan_chain's meta; the descent record
    stopped at meta and never reached it. These three keys must."""
    import inspect
    src = inspect.getsource(LemLite)
    for key in ('"mse_by_depth": imeta.get("mse_by_depth")',
                '"stopped_by": imeta.get("stopped_by")',
                '"closer_than_start"'):
        assert key in src, key + ' is missing from last_intention'
