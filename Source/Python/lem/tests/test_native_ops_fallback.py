# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The native-extension loaders stop at a platform gate before any subprocess.

lem/native_ops.py (the lem_native C++ extension: GIL-free scalar reads, gradient
clipping, the C++ inference loop) and tokenizer/modalities/vision_kernels.py (the
fused BGRA-to-RGB patch kernel) are compiled at first use with
torch.utils.cpp_extension.load_inline. Both need cl.exe (MSVC) and CUDA. Until
2026-11-04 each loader's first act was subprocess.run(['where', 'cl']), a Windows
command, so off Windows the first call raised inside the probe, was caught, and the
loader returned None only after a failed process start.

These tests patch sys.platform to "darwin" (macOS) and record every subprocess.run
call: each loader returns None and starts 0 subprocesses; the same holds on win32
with torch.cuda.is_available() patched to False. The Python fallback of
clip_grad_norm_gil_free is compared with torch.nn.utils.clip_grad_norm_ on 3 fixed
cpu tensors: the clipped gradients and the returned total norm are equal, 0
difference. The loaders' once-only caches are reset for each test and restored
afterwards by monkeypatch.

Two checks added at review (2026-11-04): on win32 with CUDA both loaders pass the
gate and reach the MSVC probe (the gate does not block the Windows build path); and
the clip_grad_norm_gil_free fallback returns the reference result when torch's
foreach support check answers False, as it does for mps tensors (the fallback passed
foreach=True, which raises RuntimeError on such a device in torch 2.6.0).
"""
from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from lem import native_ops
from tokenizer.modalities import vision_kernels


@pytest.fixture
def subprocess_calls(monkeypatch):
    """Record every subprocess.run call. A recorded call also raises, so a loader
    that reached its probe returns None through its except clause and the test
    reports the call list instead of compiling anything on this machine."""
    calls = []

    def _record(*args, **kwargs):
        calls.append(args[0] if args else kwargs.get("args"))
        raise RuntimeError("subprocess.run reached: the platform gate did not stop the loader")

    monkeypatch.setattr(subprocess, "run", _record)
    return calls


@pytest.fixture
def fresh_loaders(monkeypatch):
    """Clear both loaders' once-only caches so the gate runs in this test."""
    monkeypatch.setattr(native_ops, "_LOAD_ATTEMPTED", False)
    monkeypatch.setattr(native_ops, "_MODULE", None)
    monkeypatch.setattr(native_ops, "_DISABLED", False)
    monkeypatch.setattr(vision_kernels, "_LOAD_ATTEMPTED", False)
    monkeypatch.setattr(vision_kernels, "_KERNEL_MODULE", None)


# ------------------------------------------------------------------ the gate
def test_off_win32_both_loaders_return_none_with_no_subprocess(
        monkeypatch, fresh_loaders, subprocess_calls):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert native_ops.load_native_ops() is None
    assert vision_kernels.load_vision_kernels() is None
    assert subprocess_calls == [], f"subprocess.run was called: {subprocess_calls}"


def test_without_cuda_both_loaders_return_none_with_no_subprocess(
        monkeypatch, fresh_loaders, subprocess_calls):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert native_ops.load_native_ops() is None
    assert vision_kernels.load_vision_kernels() is None
    assert subprocess_calls == [], f"subprocess.run was called: {subprocess_calls}"


def test_the_msvc_probe_itself_runs_no_subprocess_off_win32(monkeypatch, subprocess_calls):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert native_ops._setup_msvc_env() is False
    assert vision_kernels._setup_msvc_env() is False
    assert subprocess_calls == []


@pytest.mark.parametrize("platform, cuda, expect_open", [
    ("darwin", True, False),
    ("linux", True, False),
    ("win32", False, False),
    ("win32", True, True),
])
def test_the_gate_names_the_missing_part_and_opens_only_on_win32_with_cuda(
        monkeypatch, platform, cuda, expect_open):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    for gate in (native_ops.native_build_platform_block,
                 vision_kernels.native_build_platform_block):
        block = gate()
        assert (block == "") is expect_open, block
        if not expect_open:
            assert ("win32" in block) if platform != "win32" else ("cuda" in block)


def test_the_gate_result_is_cached_like_a_failed_build(
        monkeypatch, fresh_loaders, subprocess_calls):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert native_ops.load_native_ops() is None
    assert native_ops._LOAD_ATTEMPTED is True
    monkeypatch.setattr(sys, "platform", "win32")      # a later call does not retry
    assert native_ops.load_native_ops() is None
    assert subprocess_calls == []


def test_on_win32_with_cuda_both_loaders_pass_the_gate_to_the_msvc_probe(
        monkeypatch, fresh_loaders):
    """The gate must not block the Windows build path. The MSVC probe is replaced
    by a recorder that answers False, so each loader returns None one step after
    the gate and nothing is compiled on this machine."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    probes = []
    monkeypatch.setattr(native_ops, "_setup_msvc_env",
                        lambda: probes.append("native_ops") or False)
    monkeypatch.setattr(vision_kernels, "_setup_msvc_env",
                        lambda: probes.append("vision_kernels") or False)
    assert native_ops.load_native_ops() is None
    assert vision_kernels.load_vision_kernels() is None
    assert probes == ["native_ops", "vision_kernels"]


# ------------------------------------------------------------ the fallbacks
def _three_fixed_tensors(device):
    """3 parameters with fixed gradients. Their total 2-norm is
    sqrt(1+4+9+16+25+36+49+64+81) = sqrt(285) = 16.88, above max_norm 1.0, so
    clipping rescales every gradient."""
    params = []
    values = [[1.0, -2.0, 3.0], [-4.0, 5.0], [6.0, -7.0, 8.0, -9.0]]
    for v in values:
        p = torch.nn.Parameter(torch.zeros(len(v), device=device))
        p.grad = torch.tensor(v, device=device)
        params.append(p)
    return params


def test_clip_grad_norm_fallback_equals_torch_on_three_fixed_tensors(monkeypatch):
    monkeypatch.setattr(native_ops, "load_native_ops", lambda: None)
    device = torch.device("cpu")
    ours = _three_fixed_tensors(device)
    reference = _three_fixed_tensors(device)
    norm_ours = native_ops.clip_grad_norm_gil_free(ours, 1.0)
    norm_ref = torch.nn.utils.clip_grad_norm_(reference, 1.0)
    assert torch.equal(norm_ours, norm_ref), (norm_ours, norm_ref)
    for a, b in zip(ours, reference):
        assert torch.equal(a.grad, b.grad), (a.grad - b.grad).abs().max()
    assert norm_ref.item() == pytest.approx(285 ** 0.5)
    total_after = torch.sqrt(sum((p.grad ** 2).sum() for p in ours))
    assert total_after.item() == pytest.approx(1.0, abs=1e-6)


def test_clip_grad_norm_fallback_works_on_a_device_without_foreach_kernels(monkeypatch):
    """torch 2.6.0's clip_grad_norm_ raises RuntimeError("foreach=True was passed,
    but can't use the foreach API on <device> tensors") on a device outside
    _get_foreach_kernels_supported_devices() + ["cpu"]; mps is one. The fallback
    passed foreach=True until 2026-11-04. Here the support check is patched to
    answer False, as it does for mps tensors, on cpu tensors: the fallback must
    still return the reference result, 0 difference."""
    import torch.nn.utils.clip_grad as clip_grad
    monkeypatch.setattr(native_ops, "load_native_ops", lambda: None)
    monkeypatch.setattr(clip_grad, "_device_has_foreach_support", lambda device: False)
    device = torch.device("cpu")
    ours = _three_fixed_tensors(device)
    reference = _three_fixed_tensors(device)
    norm_ours = native_ops.clip_grad_norm_gil_free(ours, 1.0)
    norm_ref = torch.nn.utils.clip_grad_norm_(reference, 1.0, foreach=False)
    assert torch.equal(norm_ours, norm_ref), (norm_ours, norm_ref)
    for a, b in zip(ours, reference):
        assert torch.equal(a.grad, b.grad), (a.grad - b.grad).abs().max()


def test_the_other_fallbacks_run_without_the_extension(monkeypatch):
    monkeypatch.setattr(native_ops, "load_native_ops", lambda: None)
    device = torch.device("cpu")
    assert native_ops.scalar_item_gil_free(torch.tensor(2.5, device=device)) == 2.5
    assert native_ops.bool_item_gil_free(torch.tensor(True, device=device)) is True
    assert native_ops.tensor_to_list_gil_free(
        torch.tensor([1.0, 2.0], device=device)) == [1.0, 2.0]
    x = torch.tensor([3.0], device=device, requires_grad=True)
    native_ops.backward_gil_free((x * x).sum())
    assert x.grad.tolist() == [6.0]
    src, dst = torch.tensor([1.0, 2.0], device=device), torch.zeros(2, device=device)
    native_ops.copy_pairs_gil_free([(src, dst)], non_blocking=False)
    assert torch.equal(dst, src)
