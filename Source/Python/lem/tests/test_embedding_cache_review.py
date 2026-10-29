# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Fresh weights must be used by the very next no_grad whole-model prediction."""
import copy
import pickle

import pytest
import torch

from lem.model.forward_model import ForwardModel


def _config():
    return {"_active_modalities": ["vision", "motor"], "model": {
        "d_model": 8, "context_length": 2,
        "forward_model": {"_n_modalities": 2, "_seq_length": 4,
                          "n_heads": 2, "n_layers": 1, "ffn_expansion": 2,
                          "stages": {"cross_modal": True, "causal_temporal": True,
                                     "cross_modal_bridge": True}}}}


def _forward(model, tokens):
    with torch.no_grad():
        return tuple(value.clone() for value in model(tokens))


@pytest.mark.parametrize("route", ["direct", "nested", "deepcopy", "pickle"])
def test_load_refreshes_no_grad_whole_forward(route):
    torch.manual_seed(340)
    model = ForwardModel(_config())
    tokens = torch.randn(2, 2, 8)
    old = _forward(model, tokens)
    assert model._emb_cache_valid
    replacement = copy.deepcopy(model.state_dict())
    for key in ("modality_embedding.weight", "temporal_embedding.weight"):
        replacement[key] = replacement[key] + torch.linspace(-2.0, 3.0, 8)
    fresh = ForwardModel(_config())
    fresh.load_state_dict(replacement)
    expected = _forward(fresh, tokens)
    assert not torch.allclose(old[0], expected[0]), "fixture must expose stale embedding use"
    if route == "deepcopy":
        model = copy.deepcopy(model)
    elif route == "pickle":
        model = pickle.loads(pickle.dumps(model))
    if route == "nested":
        parent = torch.nn.ModuleDict({"core": model})
        parent.load_state_dict({"core." + key: value for key, value in replacement.items()})
    else:
        model.load_state_dict(replacement)
    actual = _forward(model, tokens)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_replay_update_next_no_grad_forward_matches_fresh_weights():
    from mimo_bridge.lem_model import LemLite
    torch.set_num_threads(1)
    model = LemLite(device="cpu", seed=91, imagination=False, no_vision=True,
                    overrides={"model": {"context_length": 2,
                                         "forward_model": {"n_layers": 1}}})
    model.enable_replay(n_envs=2, size_per_env=16)
    for _ in range(8):
        tokens = {m: torch.randn(2, 1, model.d_model) for m in model.active}
        achieved = torch.randn(2, len(model.handle_names), 10) * 0.05
        command = torch.zeros(2, len(model.handle_names), 9)
        command[..., 6] = 1.0
        model.replay.add(tokens, achieved, {"rom_limit": 1.0, "strength_limit": 1.0}, executed=command)
    probe = torch.randn(2, model.n_mod, model.d_model)
    _forward(model.forward_model, probe)
    before = model.forward_model.modality_embedding.weight.detach().clone()
    result = model.replay_update(batch=2)
    assert result is not None and torch.isfinite(torch.tensor(result["replay_loss"]))
    assert not torch.equal(before, model.forward_model.modality_embedding.weight)
    fresh = ForwardModel(model.config)
    fresh.load_state_dict(model.forward_model.state_dict())
    actual = _forward(model.forward_model, probe)
    expected = _forward(fresh, probe)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
