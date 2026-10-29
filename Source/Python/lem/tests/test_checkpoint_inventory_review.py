# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Modern snapshot integrity must be checked before any model tensor is copied.

Tiny CPU modules only: no LemLite, simulator or production checkpoint is loaded.
"""
import json
from pathlib import Path

import pytest
import torch

from lem.config.defaults import get_defaults
from lem.training.checkpoint import CheckpointManager


class _Registry(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = torch.nn.ModuleDict({"motor": torch.nn.Linear(2, 2)})
        self.output_norm = torch.nn.LayerNorm(2)


def _modules():
    return [_Registry()] + [torch.nn.Linear(2, 2) for _ in range(4)]


def _optimizer(modules):
    return torch.optim.AdamW([p for module in modules for p in module.parameters()], lr=0.001)


def _snapshot(modules):
    return [{k: v.detach().clone() for k, v in module.state_dict().items()} for module in modules]


def _assert_same(modules, expected):
    for module, state in zip(modules, expected):
        for key, value in module.state_dict().items():
            assert torch.equal(value, state[key]), key


@pytest.fixture(params=["pytorch", "safetensors"])
def saved(request, tmp_path):
    if request.param == "safetensors":
        pytest.importorskip("safetensors")
    config = get_defaults()
    config["model"]["d_model"] = 2
    config["checkpoint"]["format"] = request.param
    manager = CheckpointManager(config, str(tmp_path / "checkpoints"))
    modules = _modules()
    optimizer = _optimizer(modules)
    sum(p.square().sum() for module in modules for p in module.parameters()).backward()
    optimizer.step()
    path = Path(manager.save(7, *modules, optimizer, {}))
    return manager, path, modules, optimizer, request.param


def _damage(path, fmt, damage):
    if fmt == "safetensors":
        from safetensors.torch import load_file, save_file
        tensor_path = path / "model.safetensors"
        flat = load_file(str(tensor_path))
    else:
        tensor_path = path / "checkpoint.pt"
        state = torch.load(tensor_path, weights_only=False, map_location="cpu")
        flat = {f"{component}.{key}": value
                for component, values in state.items()
                if component not in ("optimizer", "scaler") and isinstance(values, dict)
                for key, value in values.items() if torch.is_tensor(value)}
    if damage == "missing_tensor":
        del flat["forward_model.weight"]
    elif damage == "missing_component":
        flat = {key: value for key, value in flat.items() if not key.startswith("forward_model.")}
    elif damage == "wrong_shape":
        flat["forward_model.weight"] = torch.zeros(3, 2)
    elif damage == "unexpected_tensor":
        flat["forward_model.unlisted"] = torch.ones(1)
    elif damage == "wrong_dtype":
        flat["forward_model.weight"] = flat["forward_model.weight"].double()
    elif damage == "wrong_step":
        if fmt == "safetensors":
            meta_path = path / "meta.json"
            meta = json.loads(meta_path.read_text())
            meta["step"] += 1
            meta_path.write_text(json.dumps(meta))
        else:
            state["step"] += 1
    elif damage == "missing_optimizer":
        if fmt == "safetensors":
            (path / "optimizer.pt").unlink()
        else:
            del state["optimizer"]
    elif damage == "empty_optimizer_payload":
        if fmt == "safetensors":
            torch.save({}, path / "optimizer.pt")
        else:
            state["optimizer"] = {}
    elif damage in ("empty_optimizer_groups", "wrong_optimizer_id", "wrong_optimizer_count",
                    "duplicate_optimizer_id", "orphan_optimizer_state"):
        if fmt == "safetensors":
            opt_payload = torch.load(path / "optimizer.pt", weights_only=False, map_location="cpu")
            opt = opt_payload["optimizer"]
        else:
            opt = state["optimizer"]
        if damage == "empty_optimizer_groups":
            opt["param_groups"] = []
            opt["state"] = {}
        elif damage == "wrong_optimizer_id":
            opt["param_groups"][0]["params"][0] = "not an optimizer index"
        elif damage == "duplicate_optimizer_id":
            opt["param_groups"][0]["params"].append(opt["param_groups"][0]["params"][0])
        elif damage == "orphan_optimizer_state":
            opt["state"][max(opt["param_groups"][0]["params"]) + 1] = {}
        else:
            opt["param_groups"][0]["params"].pop()
        if fmt == "safetensors":
            torch.save(opt_payload, path / "optimizer.pt")
    if fmt == "safetensors":
        save_file(flat, str(tensor_path))
    else:
        for component in ("encoder_registry", "forward_model", "prediction_heads", "motor_controller", "wellness"):
            state[component] = {key.split(".", 1)[1]: value for key, value in flat.items()
                                if key.startswith(component + ".")}
        torch.save(state, tensor_path)


@pytest.mark.parametrize("damage", ["missing_tensor", "missing_component", "wrong_shape",
                                    "unexpected_tensor", "missing_optimizer", "empty_optimizer_payload",
                                    "empty_optimizer_groups", "wrong_optimizer_id", "wrong_optimizer_count",
                                    "wrong_step", "wrong_dtype", "duplicate_optimizer_id",
                                    "orphan_optimizer_state"])
def test_incomplete_modern_snapshot_refuses_before_copy(saved, damage):
    manager, path, _, _, fmt = saved
    _damage(path, fmt, damage)
    fresh = _modules()
    before = _snapshot(fresh)
    optimizer = _optimizer(fresh)
    with pytest.raises(RuntimeError, match="checkpoint .*inventory"):
        manager.load(str(path), *fresh, optimizer=optimizer)
    _assert_same(fresh, before)
    assert not optimizer.state


def test_valid_snapshot_restores_tensors_and_adam(saved):
    manager, path, modules, original_optimizer, _ = saved
    fresh = _modules()
    optimizer = _optimizer(fresh)
    result = manager.load(str(path), *fresh, optimizer=optimizer)
    assert result["step"] == 7
    _assert_same(fresh, _snapshot(modules))
    expected = original_optimizer.state_dict()["state"]
    actual = optimizer.state_dict()["state"]
    assert actual.keys() == expected.keys()
    for index, values in expected.items():
        for key, value in values.items():
            assert torch.equal(actual[index][key], value), (index, key)


def test_inference_can_omit_optimizer(saved):
    manager, path, modules, _, fmt = saved
    _damage(path, fmt, "missing_optimizer")
    fresh = _modules()
    manager.load(str(path), *fresh, optimizer=None)
    _assert_same(fresh, _snapshot(modules))


def test_parameters_without_prior_gradients_need_no_adam_moments(saved):
    manager, _, _, _, _ = saved
    modules = _modules()
    never_updated = _optimizer(modules)
    assert not never_updated.state
    path = manager.save(0, *modules, never_updated, {})
    fresh = _modules()
    optimizer = _optimizer(fresh)
    manager.load(path, *fresh, optimizer=optimizer)
    _assert_same(fresh, _snapshot(modules))
    assert not optimizer.state


def test_legacy_receipt_keeps_existing_addition_and_optimizer_rules(saved):
    manager, path, modules, _, fmt = saved
    receipt_path = path / "model_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    del receipt["checkpoint_inventory"]
    receipt_path.write_text(json.dumps(receipt))
    _damage(path, fmt, "missing_optimizer")
    fresh = _modules()
    result = manager.load(str(path), *fresh, optimizer=_optimizer(fresh),
                          side_modules={"new_zero_init_path": torch.nn.Linear(2, 2)})
    _assert_same(fresh, _snapshot(modules))
    assert "new_zero_init_path" in result["reinitialized_components"]


def test_declared_current_architecture_addition_and_encoder_skip_still_work(saved):
    manager, path, modules, _, _ = saved
    fresh = _modules()
    original_motor = {key: value.clone() for key, value in fresh[0].encoders["motor"].state_dict().items()}
    result = manager.load(str(path), *fresh, optimizer=None, skip_encoders=["motor"],
                          side_modules={"new_zero_init_path": torch.nn.Linear(2, 2)})
    for key, value in fresh[0].encoders["motor"].state_dict().items():
        assert torch.equal(value, original_motor[key])
    _assert_same(fresh[1:], _snapshot(modules[1:]))
    assert "encoder:motor" in result["reinitialized_components"]
    assert "new_zero_init_path" in result["reinitialized_components"]
