# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Everything trained must survive a save/load round trip.

Two things did not, and both were found by reading checkpoints off disk rather than by
any test:

  * encoder_registry.output_norm - the shared LayerNorm every token passes through - was
    WRITTEN to every checkpoint and never read back. CheckpointManager._load_encoder_registry
    only ever matched keys under "encoders.", so the two tensors outside that prefix were
    dropped. Each resume restarted the shared normaliser from a fresh LayerNorm while the
    encoders around it stayed adapted to the old one.

    The proof it was a reset and not decay: H26_tc resumed from a parent whose
    ||output_norm.bias|| was 1.5042, and its own first checkpoint 983 steps later read
    0.0997 - 6.6% of the parent. AdamW decoupled decay at lr 3e-4 and wd 0.01 is a factor
    0.999997 per step, so 983 steps of pure decay with no gradient at all could only reach
    1.4998. The observed value sits 15x below that floor. A control tensor in the same
    files, encoders.vision.patch_embed.weight, round-tripped at cosine 0.959-0.999.

  * recon_decoders and td_gates were never passed to save() at all - 0 of the 274 tensors
    a checkpoint holds - so both were rebuilt from a fresh random init on every resume.
    Measured cost: the reconstruction loss restarts at 0.87-0.98 and took 16,600 steps on
    H31 and 64,300 on H28 to come back within 10% of its floor. td_gates is worse than a
    cost, because the top-down gate feeds the very token next_pe is graded against.

These tests assert the general property rather than those two names, so the next module
added to the model is covered without anyone remembering to extend them.
"""
import os

import pytest
import torch

pytest.importorskip("safetensors")

from lem.config.defaults import get_defaults
from lem.training.checkpoint import CheckpointManager


class _Tiny(torch.nn.Module):
    """A registry-shaped module: named sub-encoders plus SHARED state beside them."""

    def __init__(self, d=8):
        super().__init__()
        self.encoders = torch.nn.ModuleDict({
            "vision": torch.nn.Linear(d, d),
            "motor": torch.nn.Linear(d, d),
        })
        self.output_norm = torch.nn.LayerNorm(d)


def _mgr(tmp_path):
    cfg = get_defaults()
    cfg.setdefault("model", {})["d_model"] = 8
    return CheckpointManager(cfg, str(tmp_path))


def _plain(d=8):
    return torch.nn.Linear(d, d)


def test_shared_registry_state_survives_the_round_trip(tmp_path):
    """output_norm sits outside the encoders. prefix and was silently dropped on load."""
    reg = _Tiny()
    with torch.no_grad():                      # move it off the fresh-init values
        reg.output_norm.weight.mul_(0.77)
        reg.output_norm.bias.add_(1.5)
    want_w = reg.output_norm.weight.detach().clone()
    want_b = reg.output_norm.bias.detach().clone()

    fwd, heads, mc, wl = _plain(), _plain(), _plain(), _plain()
    opt = torch.optim.AdamW(list(reg.parameters()) + list(fwd.parameters()), lr=1e-4)
    mgr = _mgr(tmp_path)
    path = mgr.save(1, reg, fwd, heads, mc, wl, opt, {})

    fresh = _Tiny()
    assert not torch.allclose(fresh.output_norm.bias, want_b), "test cannot detect a reset"
    mgr.load(path, fresh, _plain(), _plain(), _plain(), _plain(), optimizer=None)

    assert torch.allclose(fresh.output_norm.weight, want_w, atol=1e-6), (
        "output_norm.weight was not restored - every resume restarts the shared LayerNorm")
    assert torch.allclose(fresh.output_norm.bias, want_b, atol=1e-6), (
        "output_norm.bias was not restored - this is the bug that reset ||beta|| 1.5042 "
        "to 0.0997 across a real resume")


def test_no_registry_tensor_is_silently_dropped(tmp_path):
    """The general property: whatever the registry holds must come back."""
    reg = _Tiny()
    with torch.no_grad():
        for p in reg.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    before = {k: v.detach().clone() for k, v in reg.state_dict().items()}

    mgr = _mgr(tmp_path)
    path = mgr.save(2, reg, _plain(), _plain(), _plain(), _plain(),
                    torch.optim.AdamW(reg.parameters(), lr=1e-4), {})
    fresh = _Tiny()
    mgr.load(path, fresh, _plain(), _plain(), _plain(), _plain(), optimizer=None)
    after = fresh.state_dict()

    missed = [k for k, v in before.items()
              if k not in after or not torch.allclose(after[k], v, atol=1e-6)]
    assert not missed, f"registry tensors lost across a round trip: {missed}"


def test_side_modules_are_saved_and_restored(tmp_path):
    """recon_decoders and td_gates were never handed to save() in the first place."""
    side = {
        "recon_decoders": torch.nn.ModuleDict({"vision": _plain()}),
        "td_gates": torch.nn.ModuleDict({"vision": torch.nn.Linear(2, 1)}),
    }
    with torch.no_grad():
        for m in side.values():
            for p in m.parameters():
                p.add_(2.0)                    # unmistakably not a fresh init
    want = {k: {n: t.detach().clone() for n, t in m.state_dict().items()}
            for k, m in side.items()}

    reg = _Tiny()
    mgr = _mgr(tmp_path)
    path = mgr.save(3, reg, _plain(), _plain(), _plain(), _plain(),
                    torch.optim.AdamW(reg.parameters(), lr=1e-4), {}, side_modules=side)

    fresh_side = {
        "recon_decoders": torch.nn.ModuleDict({"vision": _plain()}),
        "td_gates": torch.nn.ModuleDict({"vision": torch.nn.Linear(2, 1)}),
    }
    res = mgr.load(path, _Tiny(), _plain(), _plain(), _plain(), _plain(),
                   optimizer=None, side_modules=fresh_side)

    for k, m in fresh_side.items():
        got = m.state_dict()
        for n, t in want[k].items():
            assert torch.allclose(got[n], t, atol=1e-6), f"{k}.{n} was not restored"
    assert any("recon_decoders" in c for c in res.get("loaded_components", [])), res


def test_a_checkpoint_without_side_modules_still_loads(tmp_path):
    """Every checkpoint written before today lacks them; that must not be an error."""
    mgr = _mgr(tmp_path)
    path = mgr.save(4, _Tiny(), _plain(), _plain(), _plain(), _plain(),
                    torch.optim.AdamW(_Tiny().parameters(), lr=1e-4), {})
    side = {"recon_decoders": torch.nn.ModuleDict({"vision": _plain()})}
    res = mgr.load(path, _Tiny(), _plain(), _plain(), _plain(), _plain(),
                   optimizer=None, side_modules=side)
    assert "recon_decoders" in res.get("reinitialized_components", []), (
        "an absent side module should be reported as reinitialized, not pass silently")


def test_the_model_declares_every_side_module_it_trains():
    """A trained parameter that is not in _side_modules() will not be saved.

    This is the check that generalises: it asserts the declaration covers the modules
    that exist, so adding a new one and forgetting to declare it fails here.
    """
    import inspect
    from mimo_bridge import lem_model
    src = inspect.getsource(lem_model.LemLite)
    declared = inspect.getsource(lem_model.LemLite._side_modules)
    for name in ("recon_decoders", "td_gates"):
        assert f"self.{name} = " in src, f"{name} is no longer built - update this test"
        assert name in declared, (
            f"self.{name} is trained but not declared in _side_modules(), so it will be "
            f"re-randomised at every resume")
