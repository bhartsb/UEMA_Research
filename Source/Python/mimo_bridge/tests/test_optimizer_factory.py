# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""optimizer_factory: the fused rule holds in one place and the cpu step is unchanged (macOS and iOS port, WI-4).

Pass lines (docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html, row WI-4): fused False for cpu and mps
devices, True for cuda; one cpu AdamW step equals the stored reference (optimizer_factory_reference.json, made on
cpu with torch 2.6.0+cu124 on 2026-11-04), 0 difference. Added at review: the flag torch.optim.AdamW receives for
each device type (torch recorded in place, no cuda or mps tensor made), and the rule put back after an optimizer
state load that carries a saved ``fused: True`` (the released checkpoint saves one). Every device here is a
``torch.device`` descriptor; no tensor is placed on cuda or mps, so the file runs on any build.
"""
import json
import os

import pytest
import torch

from mimo_bridge.optimizer_factory import FUSED_DEVICE_TYPES, apply_fused_rule, fused_for, make_adamw

REFERENCE = os.path.join(os.path.dirname(__file__), "optimizer_factory_reference.json")
LEM_MODEL = os.path.join(os.path.dirname(__file__), "..", "lem_model.py")


@pytest.mark.parametrize("device, expected", [
    (torch.device("cpu"), False),
    (torch.device("mps"), False),
    (torch.device("cuda"), True),
    (torch.device("cuda", 1), True),
    ("cpu", False), ("mps", False), ("cuda:0", True),
])
def test_fused_rule_by_device_type(device, expected):
    assert fused_for(device) is expected


def test_fused_rule_names_cuda_only():
    assert FUSED_DEVICE_TYPES == ("cuda",)


def _one_param():
    p = torch.nn.Parameter(torch.tensor([[0.5, -1.25, 2.0], [0.125, 3.5, -0.75]], dtype=torch.float32))
    p.grad = torch.tensor([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]], dtype=torch.float32)
    return p


def test_make_adamw_passes_groups_and_settings_through_and_sets_fused_false_on_cpu():
    decay, no_decay = _one_param(), torch.nn.Parameter(torch.zeros(3))
    opt = make_adamw([{"params": [decay], "weight_decay": 0.01}, {"params": [no_decay], "weight_decay": 0.0}],
                     torch.device("cpu"), lr=0.1, betas=(0.9, 0.999))
    assert isinstance(opt, torch.optim.AdamW)
    assert [g["weight_decay"] for g in opt.param_groups] == [0.01, 0.0]
    assert [g["lr"] for g in opt.param_groups] == [0.1, 0.1]
    assert [g["betas"] for g in opt.param_groups] == [(0.9, 0.999), (0.9, 0.999)]
    assert [g["fused"] for g in opt.param_groups] == [False, False]
    assert [g["eps"] for g in opt.param_groups] == [1e-8, 1e-8]
    assert [g["amsgrad"] for g in opt.param_groups] == [False, False]


@pytest.mark.parametrize("device, want_fused", [
    ("cuda", True), (torch.device("cuda", 1), True), ("cpu", False), (torch.device("mps"), False),
])
def test_make_adamw_hands_the_rule_to_torch_for_every_device_type(monkeypatch, device, want_fused):
    """What torch.optim.AdamW receives for each device type, with torch's constructor recorded in its place."""
    seen = {}

    def record(params, **kwargs):
        seen.update(kwargs)
        seen["params"] = list(params)
        return "built"

    monkeypatch.setattr(torch.optim, "AdamW", record)
    p = _one_param()
    assert make_adamw([p], device, lr=0.1, betas=[0.9, 0.999]) == "built"
    assert seen["fused"] is want_fused
    assert seen["params"] == [p]
    assert seen["lr"] == 0.1
    assert seen["betas"] == (0.9, 0.999)
    assert seen["weight_decay"] == 1e-2
    assert set(seen) == {"fused", "params", "lr", "betas", "weight_decay"}


def test_one_cpu_step_equals_the_stored_reference_exactly():
    with open(REFERENCE, encoding="utf-8") as f:
        ref = json.load(f)
    inp, want = ref["inputs"], ref["after_one_step"]
    p = torch.nn.Parameter(torch.tensor(inp["param"], dtype=torch.float32, device="cpu"))
    p.grad = torch.tensor(inp["grad"], dtype=torch.float32)
    opt = make_adamw([p], torch.device("cpu"), lr=inp["lr"], betas=tuple(inp["betas"]),
                     weight_decay=inp["weight_decay"])
    opt.step()
    got = p.detach()
    assert torch.equal(got, torch.tensor(want["param"], dtype=torch.float32)), \
        "max abs difference %r" % (got - torch.tensor(want["param"])).abs().max().item()
    st = opt.state[p]
    assert torch.equal(st["exp_avg"], torch.tensor(want["exp_avg"], dtype=torch.float32))
    assert torch.equal(st["exp_avg_sq"], torch.tensor(want["exp_avg_sq"], dtype=torch.float32))
    assert float(st["step"]) == want["step"] == 1.0


def test_one_cpu_step_equals_torch_adamw_with_fused_false():
    """The factory against torch.optim.AdamW called directly with the loop, same inputs: 0 difference."""
    a, b = _one_param(), _one_param()
    make_adamw([a], "cpu", lr=0.1, betas=(0.9, 0.999), weight_decay=0.01).step()
    torch.optim.AdamW([b], lr=0.1, betas=(0.9, 0.999), weight_decay=0.01, fused=False).step()
    assert torch.equal(a.detach(), b.detach())


def test_apply_fused_rule_undoes_the_saved_flag_an_optimizer_state_load_brings():
    """torch.optim.Optimizer.load_state_dict (torch 2.6.0, torch/optim/optimizer.py, update_group) keeps the saved
    groups' settings. The released checkpoint ckpt_step_494025's optimizer.pt saves fused True on both groups (read
    2026-11-04), so a cpu resume would step on torch's fused cpu kernel; apply_fused_rule puts the rule back, and the
    step then equals the loop's, 0 difference."""
    p = _one_param()
    opt = make_adamw([p], "cpu", lr=0.1, betas=(0.9, 0.999), weight_decay=0.01)
    saved = opt.state_dict()
    for g in saved["param_groups"]:
        g["fused"] = True
    opt.load_state_dict(saved)
    assert [g["fused"] for g in opt.param_groups] == [True], "torch no longer carries the saved flag over"
    assert apply_fused_rule(opt, torch.device("cpu")) == 1
    assert [g["fused"] for g in opt.param_groups] == [False]
    assert apply_fused_rule(opt, "cpu") == 0
    opt.step()
    b = _one_param()
    torch.optim.AdamW([b], lr=0.1, betas=(0.9, 0.999), weight_decay=0.01, fused=False).step()
    assert torch.equal(p.detach(), b.detach())


def test_apply_fused_rule_keeps_true_on_cuda_and_counts_only_changed_groups():
    """Descriptor only: the groups are plain dicts here, so no cuda tensor is made."""
    class _Opt:
        param_groups = [{"fused": True}, {"fused": None}, {"fused": False}]

    assert apply_fused_rule(_Opt(), torch.device("cuda", 0)) == 2
    assert [g["fused"] for g in _Opt.param_groups] == [True, True, True]
    assert apply_fused_rule(_Opt(), "mps") == 3
    assert [g["fused"] for g in _Opt.param_groups] == [False, False, False]


def test_lem_model_builds_its_optimizer_through_the_factory_and_reapplies_the_rule_after_a_load():
    """The rule lives here only: lem_model.py no longer spells fused=(d.type == "cuda") itself, builds through
    make_adamw and, in load_checkpoint, calls apply_fused_rule after the optimizer loads and migrations."""
    with open(LEM_MODEL, encoding="utf-8") as f:
        src = f.read()
    assert 'fused=' not in src, "lem_model.py sets fused itself again"
    assert "make_adamw(" in src
    load_src = src[src.index("def load_checkpoint("):]
    assert "apply_fused_rule(self.optimizer, self.device)" in load_src
    # after the last optimizer-touching migration of the load and before the step counter is taken over
    assert load_src.index("_migrate_synergy_scales(path, load_optimizer)") \
        < load_src.index("apply_fused_rule(self.optimizer, self.device)") \
        < load_src.index("self.step_count = int(res.get(\"step\", 0))")
