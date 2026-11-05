# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Work item WI-2 (docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; REQ-A-1 of the specification):
resolve_device turns a device request into a torch.device, refuses an unavailable one by name, and LemLite takes
the resolved device without reading torch.cuda.is_available itself. Eight cases, each on cpu; the machine facts are
patched on torch so the cases read the same on a CUDA machine and on a Mac."""
import argparse
import os
import re
import traceback

import pytest
import torch

from mimo_bridge.device_resolve import TorchAvailability, resolve_device, resolved_device_name


def _no_gpu(monkeypatch):
    """Patch torch so neither CUDA nor mps is available (the facts of a machine with no GPU)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


def _names(name):
    """The regex a refusal message must match: the request itself, quoted as %r prints it ('cuda:0')."""
    return re.escape(repr(name))


def test_auto_without_gpu_gives_cpu(monkeypatch):
    _no_gpu(monkeypatch)
    device = resolve_device("auto")
    assert isinstance(device, torch.device) and device.type == "cpu"
    assert resolved_device_name("auto") == "cpu"
    assert resolve_device(" AUTO ").type == "cpu"


def test_each_unavailable_name_raises_naming_it(monkeypatch):
    _no_gpu(monkeypatch)
    for name in ("cuda", "cuda:0", "mps"):
        with pytest.raises(ValueError, match=_names(name)):
            resolve_device(name)
        with pytest.raises(argparse.ArgumentTypeError, match=_names(name)):
            resolved_device_name(name)
    with pytest.raises(ValueError, match=_names("cuda:0")):
        resolve_device(torch.device("cuda:0"))
    # a CUDA index beyond the machine's count is refused even with CUDA present
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match=_names("cuda:7")):
        resolve_device("cuda:7")
    assert resolve_device("cuda:0") == torch.device("cuda:0")
    assert resolve_device("cuda") == torch.device("cuda")
    # a type this trainer has no table for, and a word torch rejects, are refused by name
    with pytest.raises(ValueError, match=_names("xpu")):
        resolve_device("xpu")
    with pytest.raises(ValueError, match=_names("not-a-device")):
        resolve_device("not-a-device")
    with pytest.raises(ValueError, match="None"):
        resolve_device(None)


def test_cpu_works_with_or_without_a_gpu(monkeypatch):
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")
    assert resolved_device_name("cpu") == "cpu"
    assert resolve_device(" CPU ") == torch.device("cpu")     # case and surrounding spaces are not part of a request
    _no_gpu(monkeypatch)
    assert resolve_device("cpu").type == "cpu"


def test_patched_mps_returns_mps_type(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("mps").type == "mps"
    assert resolve_device("auto").type == "mps"        # auto: cuda first, then mps, then cpu
    assert resolved_device_name("mps") == "mps"
    assert TorchAvailability().mps_available() is True


def test_injected_availability_is_read_instead_of_torch():
    class Facts:
        def __init__(self, cuda, count, mps):
            self.cuda, self.count, self.mps = cuda, count, mps

        def cuda_available(self):
            return self.cuda

        def cuda_device_count(self):
            return self.count

        def mps_available(self):
            return self.mps

    assert resolve_device("auto", Facts(True, 2, True)) == torch.device("cuda", 0)
    assert resolve_device("cuda:1", Facts(True, 2, False)) == torch.device("cuda", 1)
    assert resolve_device("auto", Facts(False, 0, True)).type == "mps"
    assert resolve_device("auto", Facts(False, 0, False)).type == "cpu"
    with pytest.raises(ValueError, match=_names("cuda:1")):
        resolve_device("cuda:1", Facts(True, 1, False))


def test_train_mimo_device_option_resolves_or_exits_2(monkeypatch, tmp_path, capsys):
    """train_mimo.parse_args: --device goes through resolved_device_name, so args.device (what args.json records) is
    the resolved name; the default "auto" resolves too; an unavailable name is an argparse error (exit code 2) whose
    message names the request."""
    from mimo_bridge.train_mimo import parse_args

    out = ["--out", str(tmp_path / "run")]
    assert parse_args(out + ["--device", "cpu"]).device == "cpu"
    _no_gpu(monkeypatch)
    assert parse_args(out).device == "cpu"                       # the default "auto" on a machine with no GPU
    with pytest.raises(SystemExit) as exit_info:
        parse_args(out + ["--device", "mps"])
    assert exit_info.value.code == 2
    assert "'mps'" in capsys.readouterr().err


def test_lemlite_on_cpu_takes_the_resolved_device_without_asking_cuda(monkeypatch):
    from mimo_bridge import lem_model
    from mimo_bridge.lem_model import LemLite

    callers = []
    real = torch.cuda.is_available

    def spy():
        frame = traceback.extract_stack(limit=3)[-2]
        callers.append((os.path.basename(frame.filename), frame.lineno))
        return real()

    monkeypatch.setattr(torch.cuda, "is_available", spy)
    model = LemLite(device="cpu", intention=False, imagination=False)
    assert model.device == torch.device("cpu")
    assert model.use_amp is False and model.dtype == torch.float32
    assert [c for c in callers if c[0] == "lem_model.py"] == [], callers
    assert "cuda.is_available" not in open(lem_model.__file__, encoding="utf-8").read()
    # a torch.device is accepted as the resolved device
    assert LemLite(device=torch.device("cpu"), intention=False, imagination=False).device.type == "cpu"


def test_lemlite_default_device_is_auto(monkeypatch):
    """A caller that names no device gets "auto" (the trainer's default), never a refused "cuda:0": with no GPU the
    model lands on cpu, as the line LemLite held before 2026-11-04 put it there."""
    from mimo_bridge.lem_model import LemLite

    _no_gpu(monkeypatch)
    assert LemLite(intention=False, imagination=False).device == torch.device("cpu")
