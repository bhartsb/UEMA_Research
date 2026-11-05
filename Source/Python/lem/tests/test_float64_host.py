# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Tests for lem/model/float64_host.py and the float64 sites it places (REQ-A-4 of
docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html, work item WI-5).

  - the three mappings of float64_device: "mps" -> cpu, "cuda:0" -> cuda:0, "cpu" -> cpu
  - KeepsFloat64OnHost keeps float64 buffers on the host through .to(device) and leaves cpu moves alone
  - the released command's sites on cpu: dtype float64 and device float64_device(cpu) at the persistent
    memory's six buffers, the hand response's buffers and search tensors, the hand joint readings' site
    (by its source), the self-state clock and the executed-command spread
  - the same sites with a model device whose float64 host differs (cuda stands in for mps through the
    injected policy; skipped without a card): float64 work on cpu, float32 results on the model's device
  - the checkpoint loader places float64 tensors on the host and everything else on map_location
  - an AST scan: no float64 allocation on a model device without float64_device in the released path's files

No mps device exists on this machine, so the mps mapping is tested through torch.device("mps") (a device
object needs no backend) and the host-differs behaviour through the injected policy.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lem.model import float64_host
from lem.model.float64_host import KeepsFloat64OnHost, float64_device, same_device, to_model_device

CPU = torch.device("cpu")
HAS_CUDA = torch.cuda.is_available()


def cuda_as_mps(device):
    """A stand-in policy for a machine with a card: cuda plays the device without float64, so every
    float64 tensor goes to cpu, as float64_device sends it for mps."""
    device = torch.device(device)
    return CPU if device.type in ("cuda", "mps") else device


# ------------------------------------------------------------------------------------------ the mappings
@pytest.mark.parametrize("given, expected", [
    ("mps", "cpu"), ("mps:0", "cpu"), (torch.device("mps"), "cpu"),
    ("cuda:0", "cuda:0"), (torch.device("cuda:0"), "cuda:0"), ("cuda", "cuda"),
    ("cpu", "cpu"), (torch.device("cpu"), "cpu"),
])
def test_the_three_mappings(given, expected):
    out = float64_device(given)
    assert isinstance(out, torch.device)
    assert out == torch.device(expected)


def test_same_device_reads_a_missing_index_as_zero():
    assert same_device("cuda", "cuda:0") and same_device(torch.device("cpu"), "cpu")
    assert not same_device("cuda:1", "cuda:0") and not same_device("cpu", "mps")


def test_to_model_device_leaves_a_tensor_on_its_device_and_casts_only_when_it_moves():
    x = torch.zeros(3, dtype=torch.float64)
    assert to_model_device(x, "cpu") is x                       # cpu and cuda: unchanged, dtype kept
    moved = to_model_device(x, "meta")                          # another device: a float32 copy there
    assert moved.device.type == "meta" and moved.dtype == torch.float32 and moved.shape == x.shape


# --------------------------------------------------------------------------------- the module base
class _Carrier(KeepsFloat64OnHost):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.register_buffer("flag", torch.zeros((), dtype=torch.bool))
        self.register_buffer("weight64", torch.ones(3, dtype=torch.float64))


class _CarrierOnMetaHost(_Carrier):
    float64_host = staticmethod(lambda device: CPU if torch.device(device).type == "meta" else torch.device(device))


def test_float64_buffers_stay_on_the_host_when_the_module_moves_to_a_device_without_float64():
    m = _CarrierOnMetaHost().to("meta")
    assert m.linear.weight.device.type == "meta" and m.flag.device.type == "meta"
    assert m.weight64.device == CPU and m.weight64.dtype == torch.float64
    assert torch.equal(m.weight64, torch.ones(3, dtype=torch.float64))


def test_cpu_moves_and_dtype_conversions_are_unchanged():
    m = _Carrier()
    before = m.weight64
    m.to("cpu")
    assert m.weight64 is before                                 # nothing changed
    m.double()
    assert m.linear.weight.dtype == torch.float64 and m.weight64.dtype == torch.float64
    m.float()                                                   # the usual conversion still reaches float64 buffers
    assert m.weight64.dtype == torch.float32


@pytest.mark.skipif(not HAS_CUDA, reason="needs a card to stand in for mps")
def test_the_stand_in_policy_keeps_float64_on_cpu_through_cuda(monkeypatch):
    monkeypatch.setattr(_Carrier, "float64_host", staticmethod(cuda_as_mps))
    m = _Carrier().cuda()
    assert m.linear.weight.device.type == "cuda" and m.flag.device.type == "cuda"
    assert m.weight64.device == CPU and m.weight64.dtype == torch.float64
    plain = _Carrier()
    monkeypatch.setattr(_Carrier, "float64_host", staticmethod(float64_device))
    plain.cuda()
    assert plain.weight64.device.type == "cuda"                 # the real policy: cuda holds float64 as before


# ------------------------------------------------------------------------- persistent memory's six buffers
SIX = ("event_centre_weight", "state_centre_weight", "moment_weight", "request_centre_weight",
       "needs_request_centre_weight", "updates_since_switch")


def test_persistent_memory_six_float64_buffers_on_cpu_are_float64_on_the_host():
    from lem.model.persistent_memory import PersistentMemory
    pm = PersistentMemory(8, 2, 4, 3, hidden_size=12, repair="centred")
    for name in SIX:
        buf = getattr(pm, name)
        assert buf.dtype == torch.float64 and buf.device == float64_device(CPU) == CPU, name
    others = [n for n, b in pm.named_buffers() if n not in SIX]
    assert all(getattr(pm, n).dtype != torch.float64 for n in others)


def test_persistent_memory_six_buffers_stay_on_the_host_when_the_module_moves(monkeypatch):
    from lem.model.persistent_memory import PersistentMemory
    monkeypatch.setattr(PersistentMemory, "float64_host",
                        staticmethod(lambda d: CPU if torch.device(d).type == "meta" else torch.device(d)))
    pm = PersistentMemory(8, 2, 4, 3, hidden_size=12, repair="centred").to("meta")
    assert pm.readout.weight.device.type == "meta" and pm.state_ref.device.type == "meta"
    for name in SIX:
        assert getattr(pm, name).device == CPU and getattr(pm, name).dtype == torch.float64, name


@pytest.mark.skipif(not HAS_CUDA, reason="needs a card to stand in for mps")
def test_persistent_memory_constant_factor_reads_a_cpu_clock_beside_card_buffers(monkeypatch):
    from lem.model.persistent_memory import PersistentMemory
    monkeypatch.setattr(PersistentMemory, "float64_host", staticmethod(cuda_as_mps))
    pm = PersistentMemory(8, 2, 4, 3, hidden_size=12, repair="centred").cuda()
    assert pm.updates_since_switch.device == CPU and pm.readout.weight.device.type == "cuda"
    factor = pm.constant_factor()                               # the 0-dim cpu clock beside cuda buffers
    assert factor.device.type == "cuda" and factor.shape == (2,) and bool((factor == 1).all())


# ----------------------------------------------------------------------------- the hand response's sites
def hand_fixture():
    """mimo_bridge/tests/test_hand_response.py's one-joint calibration, copied so this file needs no import of
    another test module."""
    from lem.model.hand_response import FEATURES, VERSION, CalibratedHandResponse, content_hash
    geometry = dict(contract='wrist_relative_joint_v1', handle_names=['wrist_L', 'index_L', 'gaze'],
                    joints=[dict(name='index_bend', limits=[-2., 2.], reference=0., axis=[0., 0., 1.],
                                 anchor=[0., 0., 0.])],
                    constraints=[], bodies=[dict(position=[.01, 0., 0.], quaternion=[1., 0., 0., 0.], parent=-1,
                                                 joints=[0])],
                    hands=[dict(wrist_index=0, finger_indices=[1], finger_bodies=[0])])
    receipt = dict(version=VERSION, features=FEATURES, horizon_ms=250, geometry_sha256=content_hash(geometry),
                   train_bodies=[0, 1, 2, 3], test_bodies=[4, 5, 6, 7], joint_names=['index_bend'],
                   coefficients=[[0., .5, .4]], source_sha256={'private_physical_test': '0' * 64},
                   choice=dict(steps=20, lr=.1, raw_joint_bound_deg=11., nonfinger_orders='fixed'))
    receipt['sha256'] = content_hash(receipt)
    return CalibratedHandResponse(geometry, receipt)


def selection_inputs(model, device=CPU):
    command = torch.zeros(3, 3, 9); command[..., 6] = 1.; command[..., 8] = .4
    command[:, 0, :3] = torch.tensor([5., 3., 8.])
    raw = torch.full((3, 1), .3); previous = torch.zeros_like(raw); filtered = .35 * raw
    command = model.kinematics.assemble_world(command, filtered, position_scale=1.)
    target = torch.zeros(3, 3, 10, dtype=torch.float64); target[..., 6] = 1.
    target = model.kinematics.assemble_world(target, torch.tensor([[.1], [.0525], [.15]], dtype=torch.float64))
    return dict(canonical_execution=command.to(device), raw_initial_goal=raw.to(device),
                previous_goal=previous.to(device), smoothing=.35,
                current_angles=torch.zeros(3, 1, dtype=torch.float64, device=float64_device(device)),
                current_velocity=torch.zeros(3, 1, dtype=torch.float64, device=float64_device(device)),
                target_pose=target, valid=torch.tensor([True, True, False], device=device))


def test_hand_response_sites_on_cpu_are_float64_on_the_host():
    model = hand_fixture()
    assert model.coefficients.dtype == torch.float64 and model.coefficients.device == float64_device(CPU)
    assert model.ranges.dtype == torch.float64 and model.ranges.device == float64_device(CPU)
    q = torch.tensor([[.1]], dtype=torch.float32); v = torch.tensor([[.3]]); goal = torch.tensor([[.8]])
    predicted = model.predict(q, v, goal)                       # float32 in: float64 arithmetic on the host
    assert predicted.dtype == torch.float64 and predicted.device == float64_device(CPU)
    chosen = model.choose_hand_goal(**selection_inputs(model))
    costs = torch.tensor(chosen['metadata']['first'])
    assert torch.isfinite(costs).all() and chosen['command'].device == CPU
    assert chosen['metadata']['selected_iteration'][0] > 0     # the search still improves body 0


@pytest.mark.skipif(not HAS_CUDA, reason="needs a card to stand in for mps")
def test_hand_response_on_a_device_without_float64_works_on_cpu_and_answers_on_the_device(monkeypatch):
    from lem.model import hand_response
    from lem.model.hand_response import CalibratedHandResponse
    monkeypatch.setattr(hand_response, "float64_device", cuda_as_mps)
    monkeypatch.setattr(CalibratedHandResponse, "float64_host", staticmethod(cuda_as_mps))
    reference = hand_fixture().choose_hand_goal(**selection_inputs(hand_fixture()))
    model = hand_fixture().cuda()
    assert model.coefficients.device == CPU and model.ranges.device == CPU   # kept on the host by .cuda()
    inputs = selection_inputs(model, torch.device("cuda:0"))
    inputs['current_angles'] = inputs['current_angles'].cpu(); inputs['current_velocity'] = inputs['current_velocity'].cpu()
    chosen = model.choose_hand_goal(**inputs)
    assert chosen['command'].device.type == "cuda" and chosen['joint_goal'].device.type == "cuda"
    torch.testing.assert_close(chosen['command'].cpu(), reference['command'], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(chosen['joint_goal'].cpu(), reference['joint_goal'], atol=1e-5, rtol=1e-5)
    assert chosen['metadata']['selected_iteration'] == reference['metadata']['selected_iteration']


# ------------------------------------------------------------------------------------- the self-state clock
def frames_for_clock(n_sites=2, bodies=2):
    from lem.body_function import CONTRACT, FUNCTION_FIELDS
    frames = []
    for b in range(bodies):
        record = dict(contract=CONTRACT, mode='observe', fields=FUNCTION_FIELDS, time=0.05 * (b + 1),
                      values=[[0.5, 0.25, 0.125]] * n_sites, site_ids=list(range(n_sites)))
        frames.append(SimpleNamespace(meta=dict(body_function=record), sim_time=0.05 * (b + 1),
                                      skin_features=np.zeros((n_sites, 8), dtype=np.float32),
                                      touch_pos=np.zeros((n_sites, 3), dtype=np.float32)))
    return frames


def test_the_self_state_clock_is_float64_on_the_host_and_the_sites_are_float32_on_the_device():
    from mimo_bridge.self_state_integration import batch_observation
    sites, valid, now = batch_observation(frames_for_clock(), CPU)
    assert now.dtype == torch.float64 and now.device == float64_device(CPU)
    assert sites.dtype == torch.float32 and sites.device == CPU and valid.dtype == torch.bool
    assert now.tolist() == [0.05, 0.1]


@pytest.mark.skipif(not HAS_CUDA, reason="needs a card to stand in for mps")
def test_the_self_state_clock_goes_to_cpu_when_the_device_has_no_float64(monkeypatch):
    from mimo_bridge import self_state_integration
    monkeypatch.setattr(self_state_integration, "float64_device", cuda_as_mps)
    sites, valid, now = self_state_integration.batch_observation(frames_for_clock(), torch.device("cuda:0"))
    assert now.dtype == torch.float64 and now.device == CPU
    assert sites.device.type == "cuda"
    delta = to_model_device((now - (now - 0.05)).float(), sites.device)       # the step length handed to the model
    assert delta.device.type == "cuda" and delta.dtype == torch.float32


# ------------------------------------------------------------------------------ the executed-command spread
def test_the_command_spread_is_float64_on_the_host_on_cpu():
    from mimo_bridge.prediction_audit import command_spread_update
    state = command_spread_update(None, torch.randn(4, 26, 9))
    assert state["mean"].dtype == torch.float64 and state["mean"].device == float64_device(CPU)
    assert state["sq"].dtype == torch.float64 and state["steps"] == 1


@pytest.mark.skipif(not HAS_CUDA, reason="needs a card to stand in for mps")
def test_the_command_spread_lives_on_cpu_for_a_device_without_float64_and_its_reader_returns(monkeypatch):
    from mimo_bridge import prediction_audit
    from mimo_bridge.prediction_audit import command_spread_std, command_spread_update, shifted_own_order
    monkeypatch.setattr(prediction_audit, "float64_device", cuda_as_mps)
    exec_t = torch.randn(4, 26, 9, device="cuda:0")
    state = None
    for _ in range(prediction_audit.COMMAND_SPREAD_MIN_STEPS):
        state = command_spread_update(state, exec_t)
    assert state["mean"].device == CPU and state["mean"].dtype == torch.float64
    std = command_spread_std(state)
    assert std.device == CPU and std.dtype == torch.float64
    shifted = shifted_own_order(exec_t, std, 7, 1)
    assert shifted.device.type == "cuda" and shifted.dtype == exec_t.dtype and shifted.shape == exec_t.shape


# -------------------------------------------------------------------------------------- attention's scores
def test_cosine_scores_on_cpu_are_unchanged():
    from lem.model.attention_cap import EPS, cosine_scores
    q, k = torch.randn(2, 3, 5), torch.randn(2, 4, 5)
    out = cosine_scores(q, k, 2.5)
    qn = q / q.norm(dim=-1, keepdim=True).clamp_min(EPS); kn = k / k.norm(dim=-1, keepdim=True).clamp_min(EPS)
    torch.testing.assert_close(out, 2.5 * qn @ kn.transpose(-1, -2))
    assert out.dtype == torch.float32 and out.device == CPU


@pytest.mark.skipif(not HAS_CUDA, reason="needs a card to stand in for mps")
def test_cosine_scores_for_a_device_without_float64_compute_on_cpu_and_return_to_the_device(monkeypatch):
    from lem.model import attention_cap
    from lem.model.attention_cap import cosine_scores
    q, k = torch.randn(2, 3, 5), torch.randn(2, 4, 5)
    reference = cosine_scores(q, k, 2.5)
    monkeypatch.setattr(attention_cap, "float64_device", cuda_as_mps)
    qc, kc = q.cuda().requires_grad_(True), k.cuda().requires_grad_(True)
    out = cosine_scores(qc, kc, 2.5)
    assert out.device.type == "cuda" and out.dtype == torch.float32
    torch.testing.assert_close(out.cpu(), reference, atol=1e-6, rtol=1e-6)
    out.sum().backward()                                        # the gradient crosses the copies
    assert qc.grad is not None and qc.grad.device.type == "cuda" and torch.isfinite(qc.grad).all()


# ------------------------------------------------------------------------------------- the checkpoint loader
def write_checkpoint(folder: Path):
    from safetensors.torch import save_file
    folder.mkdir(parents=True, exist_ok=True)
    save_file({"forward_model.w": torch.ones(2, 2), "persistent_memory.moment_weight": torch.ones((), dtype=torch.float64)},
              str(folder / "model.safetensors"))
    (folder / "meta.json").write_text(json.dumps({"step": 7}), encoding="utf-8")
    torch.save({"optimizer": {"state": {0: {"exp_avg": torch.zeros(2)}}}}, folder / "optimizer.pt")


def loader():
    from lem.training.checkpoint import CheckpointManager
    return CheckpointManager({"checkpoint": {"format": "safetensors", "keep_last_n": 1, "progressive_resize": False}})


def test_the_loader_on_cpu_places_everything_on_cpu(tmp_path):
    write_checkpoint(tmp_path / "ckpt")
    state = loader()._load_state(tmp_path / "ckpt", map_location=None, want_optimizer=True)
    assert state["step"] == 7
    assert state["forward_model"]["w"].device == CPU
    assert state["persistent_memory"]["moment_weight"].dtype == torch.float64
    assert state["persistent_memory"]["moment_weight"].device == CPU
    assert state["optimizer"]["state"][0]["exp_avg"].device == CPU


def test_the_loader_keeps_float64_on_the_host_for_a_device_without_float64(tmp_path, monkeypatch):
    from lem.training import checkpoint
    monkeypatch.setattr(checkpoint, "float64_device",
                        lambda d: CPU if torch.device(d).type == "meta" else torch.device(d))
    write_checkpoint(tmp_path / "ckpt")
    state = loader()._load_state(tmp_path / "ckpt", map_location="meta", want_optimizer=True)
    assert state["forward_model"]["w"].device.type == "meta"
    assert state["persistent_memory"]["moment_weight"].device == CPU
    assert state["persistent_memory"]["moment_weight"].dtype == torch.float64
    assert state["optimizer"]["state"][0]["exp_avg"].device.type == "meta"


@pytest.mark.skipif(not HAS_CUDA, reason="needs a card")
def test_the_loader_on_cuda_is_one_load_to_the_card(tmp_path):
    write_checkpoint(tmp_path / "ckpt")
    state = loader()._load_state(tmp_path / "ckpt", map_location=torch.device("cuda:0"), want_optimizer=False)
    assert state["forward_model"]["w"].device.type == "cuda"
    assert state["persistent_memory"]["moment_weight"].device.type == "cuda"     # cuda holds float64: as before


# ------------------------------------------------------------------------------------------- the AST scan
RELEASED_PATH_FILES = (
    "lem/model/float64_host.py", "lem/model/hand_response.py", "lem/model/persistent_memory.py",
    "lem/model/attention_cap.py", "lem/training/checkpoint.py", "mimo_bridge/lem_model.py",
    "mimo_bridge/prediction_audit.py", "mimo_bridge/self_state_integration.py",
)


def _is_float64(node) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "float64"
            and isinstance(node.value, ast.Name) and node.value.id == "torch")


def _is_float64_device_call(node) -> bool:
    f = node.func if isinstance(node, ast.Call) else None
    return ((isinstance(f, ast.Name) and f.id == "float64_device")
            or (isinstance(f, ast.Attribute) and f.attr == "float64_device"))


def _host_names(scope) -> set:
    """The names a function (or the module) binds to float64_device(...) with a plain assignment, such as
    `host = float64_device(dev)`; a device=host keyword then counts as float64_device."""
    names = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign) and _is_float64_device_call(node.value):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def float64_allocations_on_a_model_device(source: str, filename: str):
    """Every call in `source` that makes a float64 tensor on a device named by the caller without
    float64_device: a call with dtype=torch.float64 whose device= keyword is neither float64_device(...) nor a name
    the enclosing function bound to float64_device(...), a .double() call, or a .to(torch.float64) /
    .to(dtype=torch.float64) call without such a device= keyword. A call with dtype=torch.float64 and no device
    keyword allocates on cpu and is not listed. Output: sorted [(line, text)]."""
    out = set()
    lines = source.splitlines()
    module = ast.parse(source, filename)
    module_hosts = _host_names(module)
    functions = [n for n in ast.walk(module) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        # the names bound to float64_device(...) that this call can read: the module's and its functions'
        hosts = set(module_hosts)
        for fn in functions:
            if fn.lineno <= node.lineno <= fn.end_lineno:
                hosts |= _host_names(fn)
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        f = node.func
        is_method = isinstance(f, ast.Attribute)
        placed = "device" in kw and (_is_float64_device_call(kw["device"]) or (
            isinstance(kw["device"], ast.Name) and kw["device"].id in hosts))
        float64_dtype = ("dtype" in kw and _is_float64(kw["dtype"])) or (
            is_method and f.attr == "to" and any(_is_float64(a) for a in node.args))
        if is_method and f.attr == "double":
            out.add((node.lineno, lines[node.lineno - 1].strip()))
        elif float64_dtype and is_method and f.attr == "to" and not placed:
            out.add((node.lineno, lines[node.lineno - 1].strip()))
        elif float64_dtype and "device" in kw and not placed:
            out.add((node.lineno, lines[node.lineno - 1].strip()))
    return sorted(out)


def test_no_float64_allocation_on_a_model_device_without_float64_device_in_the_released_path():
    found = {}
    for rel in RELEASED_PATH_FILES:
        path = ROOT / rel
        hits = float64_allocations_on_a_model_device(path.read_text(encoding="utf-8"), str(path))
        if hits:
            found[rel] = hits
    assert not found, found


def test_the_scan_catches_each_shape_it_is_for():
    bad = ("x = torch.zeros(3, dtype=torch.float64, device=dev)\n"
           "y = v.double()\n"
           "z = v.to(torch.float64)\n"
           "w = v.to(device=dev, dtype=torch.float64)\n")
    assert [line for line, _ in float64_allocations_on_a_model_device(bad, "bad")] == [1, 2, 3, 4]
    good = ("x = torch.zeros(3, dtype=torch.float64, device=float64_device(dev))\n"
            "y = v.to(device=float64_device(v.device), dtype=torch.float64)\n"
            "c = torch.as_tensor(values, dtype=torch.float64)\n"
            "def f(dev):\n"
            "    host = float64_device(dev)\n"
            "    return torch.zeros((), dtype=torch.float64, device=host)\n")
    assert float64_allocations_on_a_model_device(good, "good") == []
    unbound = "def g(dev, host):\n    return torch.zeros((), dtype=torch.float64, device=host)\n"
    assert [line for line, _ in float64_allocations_on_a_model_device(unbound, "unbound")] == [2]
