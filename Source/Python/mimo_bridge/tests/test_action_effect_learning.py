# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import numpy as np
import pytest
import torch

from mimo_bridge.tools.action_effect_learning import (
    MODS, learning_loss, paired_differences, measured_scales, raw_candidates, predict)


@pytest.mark.parametrize("layout", ["legacy", "trunk_v1"])
def test_controls_share_raw_units_before_clamping(layout):
    from mimo_bridge.lem_model import LemLite
    from mimo_bridge.body_layout import names_for
    names = names_for(layout)
    n = len(names)
    # every body row, the eyes' row left out by name: in trunk_v1 the lower trunk and the
    # chest come after it, so "rows 0-22" would leave them out
    body = [i for i, nm in enumerate(names) if nm != "gaze"]
    raw = torch.zeros(1, n, 9)
    raw[..., :3] = .2
    raw[..., 6] = 1
    raw[..., 8] = .5
    ach = torch.zeros(1, n, 10)
    ach[..., :3] = .1
    ach[..., 6] = 1
    bones = np.zeros((n, 10))
    bones[:, 4:7] = 20
    bones[:, 3] = 1
    c = raw_candidates(raw, ach, bones, np.random.default_rng(1))
    assert [tuple(x.shape) for x in c] == [(1, n, 9)] * len(c)
    assert torch.allclose(c[1][:, body, :3], ach[:, body, :3])
    for alt in c[2:]:
        assert torch.allclose((alt[:, body, :3] - ach[:, body, :3]).norm(dim=-1),
                              (raw[:, body, :3] - ach[:, body, :3]).norm(dim=-1))
    executed = LemLite.planned_execution(c[1], {"rom_limit": 1., "strength_limit": 1.}, ach)
    assert torch.allclose(executed[:, body, :3], torch.full((1, len(body), 3), 20.))


def test_effect_lesson_ignores_common_offset_but_rejects_wrong_actions():
    y = torch.randn(3, 4, 8, generator=torch.Generator().manual_seed(5))
    assert torch.allclose(paired_differences(y + 10), paired_differences(y), atol=2e-6)
    scales = {m: {"point": 1., "effect": 1.} for m in MODS}
    target = {m: y for m in MODS}
    assert learning_loss(target, target, scales, effect=True) == 0
    wrong = {m: y.roll(1, dims=1) for m in MODS}
    assert learning_loss(wrong, target, scales, effect=True) > 0


def test_no_physical_effect_cannot_pass_resolution():
    rows = [{"target": {m: torch.ones(4, 8) for m in MODS},
             "tokens": {m: torch.ones(8) for m in MODS},
             "replay_mse": {m: torch.tensor(0.) for m in MODS}}]
    with pytest.raises(ValueError, match="unresolved"):
        measured_scales(rows)


def test_batched_prediction_matches_canonical_transition():
    from mimo_bridge.lem_model import LemLite
    from lem.model.predictive_state import PredictiveState, transition
    from lem.model.encoder_registry import normalize_executed_command
    m = LemLite(device="cpu", intention=False, imagination=False,
                overrides={"model": {"context_length": 4, "residual_prediction": True}})
    for part in (m.encoder_registry, m.forward_model, m.prediction_heads):
        part.eval()
    row = {"tokens": {k: torch.randn(m.d_model) for k in m.active},
           "context": torch.randn(3 * m.n_mod, m.d_model),
           "achieved": torch.zeros(24, 10), "commands": torch.randn(4, 24, 9)}
    row["achieved"][:, 6] = 1
    with torch.no_grad():
        p = predict(m, [row], canonical=True)
        for a in range(4):
            state = PredictiveState(tokens={k: v[None] for k, v in row["tokens"].items()},
                context=row["context"][None], achieved=row["achieved"][None])
            expected = transition(m.forward_model, m.encoder_registry, m.prediction_heads,
                state, normalize_executed_command(row["commands"][a:a+1]), advance_context=False)
            for sense in MODS:
                assert torch.allclose(p[sense][0, a], expected.tokens[sense][0], atol=2e-6)


@pytest.mark.parametrize("layout", ["legacy", "trunk_v1"])
def test_command_residual_preserves_checkpoint_at_birth_and_receives_only_command(layout):
    from mimo_bridge.lem_model import LemLite
    from mimo_bridge.tools.action_effect_learning import add_command_residual
    m = LemLite(device="cpu", intention=False, imagination=False,
                overrides={"model": {"context_length": 4}}, body_layout=layout)
    n = len(m.handle_names)
    enc = m.encoder_registry.encoders["motor"]
    enc.eval()
    x = torch.randn(2, n, 19)
    before = enc(x).detach()
    add_command_residual(m)
    enc = m.encoder_registry.encoders["motor"]
    assert enc.command_projection.in_features == n * 9
    with pytest.raises(ValueError, match=f"for {n} handles"):
        enc(torch.randn(2, 50 - n, 19))       # the other layout's row count
    from lem.model.encoder_registry import build_motor_input
    actual = build_motor_input(m.encoder_registry, x[..., 10:], achieved=x[..., :10])
    assert torch.equal(actual, x), "wrapper must preserve the canonical 19-field motor layout"
    assert torch.equal(before, enc(x).detach())
    weight = torch.randn_like(before)
    (enc(x) * weight).sum().backward()
    proj = enc.command_projection
    assert proj.weight.grad is not None and proj.weight.grad.abs().sum() > 0
    with torch.no_grad():
        proj.weight.add_(-.1 * proj.weight.grad)
    observed = []
    hook = proj.register_forward_pre_hook(lambda module, args: observed.append(args[0].detach()))
    enc(x)
    hook.remove()
    assert torch.equal(observed[0], x[..., 10:19].flatten(1))


def test_readout_learns_planted_action_effect_in_new_states():
    from mimo_bridge.tools.action_effect_readout import fit_readout, scores
    rng = torch.Generator().manual_seed(34)
    x = torch.randn(40, 4, 8, generator=rng).double()
    w = torch.randn(8, 5, generator=rng).double()
    y = x @ w + torch.randn(40, 1, 5, generator=rng).double() * 20
    fitted = fit_readout(x[:30], y[:30], .01)
    assert scores(x[30:] @ fitted, y[30:]) < .01
