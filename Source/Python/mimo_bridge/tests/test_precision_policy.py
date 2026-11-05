# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""mimo_bridge/precision_policy.py: the per-device dtype / autocast table (Apple port REQ-A-2, REQ-A-3)."""
import contextlib

import pytest
import torch

from mimo_bridge.precision_policy import PrecisionPolicy, dtype_name, full_precision, policy_of


@pytest.mark.parametrize("device, requested, mps_autocast, dtype, use_amp", [
    ("cuda", "bf16", None, torch.bfloat16, True),        # the released command: receipt precision bf16
    ("cuda:0", "fp16", None, torch.float16, True),
    ("cuda", "fp32", None, torch.float32, False),
    ("cpu", "bf16", None, torch.float32, False),         # cpu: float32, off, whatever the config asks
    ("mps", "bf16", None, torch.float32, False),         # mps: float32, off by default
    ("mps", "bf16", "bf16", torch.bfloat16, True),       # --mps-autocast bf16
    ("mps", "fp32", "fp16", torch.float16, True),        # --mps-autocast fp16
    ("meta", "bf16", None, torch.float32, False),        # any other device type: float32, off
])
def test_the_table(device, requested, mps_autocast, dtype, use_amp):
    p = PrecisionPolicy(torch.device(device), requested=requested, mps_autocast=mps_autocast)
    assert p.dtype == dtype and p.use_amp is use_amp
    assert p.device.type == torch.device(device).type
    assert p.scaler() is None
    assert p.name == (dtype_name(dtype) if use_amp else "fp32")
    r = p.receipt()
    assert r["device_type"] == p.device.type and r["autocast"] is use_amp and r["precision"] == p.name
    assert r["requested"] == requested and r["mps_autocast"] == mps_autocast
    # every row's context builds and enters on this machine (an enabled mps or cuda one needs its backend)
    if not use_amp or torch.device(device).type == "cpu":
        with p.autocast():
            pass


def test_a_device_type_without_autocast_gets_a_no_op_context():
    # torch 2.6.0 raises RuntimeError "unsupported scalarType" building even a disabled "meta" context
    assert torch.amp.is_autocast_available("meta") is False
    p = PrecisionPolicy(torch.device("meta"), requested="bf16")
    assert p.dtype == torch.float32 and p.use_amp is False
    for ctx in (p.autocast(), p.autocast(enabled=False), full_precision(torch.device("meta"))):
        assert isinstance(ctx, contextlib.nullcontext)
        with ctx:
            pass


def test_a_device_string_or_a_tensor_is_accepted():
    assert PrecisionPolicy("cpu").dtype == torch.float32
    assert PrecisionPolicy("cuda", requested="bf16").use_amp is True
    assert PrecisionPolicy(torch.zeros(1), requested="bf16").device.type == "cpu"


def test_mps_autocast_is_refused_off_mps_and_with_an_unknown_name():
    with pytest.raises(ValueError, match="acts on an mps device only"):
        PrecisionPolicy(torch.device("cuda"), requested="bf16", mps_autocast="bf16")
    with pytest.raises(ValueError, match="acts on an mps device only"):
        PrecisionPolicy(torch.device("cpu"), mps_autocast="bf16")
    with pytest.raises(ValueError, match="must be one of"):
        PrecisionPolicy(torch.device("mps"), mps_autocast="fp32")


def test_the_cpu_context_is_off_and_names_the_cpu():
    p = PrecisionPolicy(torch.device("cpu"), requested="bf16")
    ctx = p.autocast()
    assert ctx.device == "cpu" and ctx._enabled is False
    with ctx:
        assert torch.is_autocast_enabled("cpu") is False
    # the full-precision form the synergy lesson and the slot watcher use
    ctx = p.autocast(enabled=False)
    assert ctx.device == "cpu" and ctx._enabled is False
    # asked on, a cpu policy opens a bf16 cpu context (torch's cpu autocast accepts bf16)
    ctx = PrecisionPolicy.from_attributes("cpu", dtype=torch.bfloat16, use_amp=True).autocast()
    assert ctx.device == "cpu" and ctx._enabled is True and ctx.fast_dtype == torch.bfloat16
    with ctx:
        assert torch.is_autocast_enabled("cpu") is True
        y = torch.nn.functional.linear(torch.ones(2, 3), torch.ones(4, 3))
    assert y.dtype == torch.bfloat16


def test_the_mps_context_names_mps_and_is_off_by_default():
    # torch 2.6.0 builds the disabled mps context on any machine (autocast_mode.py checks the dtype and the
    # macOS version only when the context is enabled)
    ctx = PrecisionPolicy(torch.device("mps"), requested="bf16").autocast()
    assert ctx.device == "mps" and ctx._enabled is False
    p = PrecisionPolicy(torch.device("mps"), mps_autocast="bf16")
    assert p.dtype == torch.bfloat16 and p.use_amp is True
    if not torch.backends.mps.is_available():
        pytest.skip("an enabled mps autocast context needs the MPS backend")
    ctx = p.autocast()
    assert ctx.device == "mps" and ctx._enabled is True and ctx.fast_dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="an enabled cuda context needs a CUDA device")
def test_the_cuda_context_is_bf16_and_on():
    ctx = PrecisionPolicy(torch.device("cuda"), requested="bf16").autocast()
    assert ctx.device == "cuda" and ctx._enabled is True and ctx.fast_dtype == torch.bfloat16
    ctx = PrecisionPolicy(torch.device("cuda"), requested="bf16").autocast(enabled=False)
    assert ctx.device == "cuda" and ctx._enabled is False


def test_policy_of_reads_the_model_or_builds_one_from_its_attributes():
    class WithPolicy:
        precision = PrecisionPolicy(torch.device("cuda"), requested="bf16")
    assert policy_of(WithPolicy(), torch.device("cpu")) is WithPolicy.precision

    class Bare:                     # the switch checks' test doubles
        use_amp = False
    p = policy_of(Bare(), torch.device("cpu"))
    assert p.device.type == "cpu" and p.dtype == torch.float32 and p.use_amp is False

    class Attributes:
        dtype = torch.bfloat16
        use_amp = True
    p = policy_of(Attributes(), torch.device("cuda"))
    assert p.device.type == "cuda" and p.dtype == torch.bfloat16 and p.use_amp is True

    assert policy_of(object(), "cpu").use_amp is False


def test_full_precision_takes_a_tensor_a_device_or_a_string():
    x = torch.zeros(2)
    for arg in (x, x.device, "cpu"):
        ctx = full_precision(arg)
        assert ctx.device == "cpu" and ctx._enabled is False
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu") is True
        with full_precision(x):
            assert torch.is_autocast_enabled("cpu") is False
            y = torch.nn.functional.linear(torch.ones(2, 3), torch.ones(4, 3))
        assert y.dtype == torch.float32
    assert full_precision(torch.device("mps")).device == "mps"


def test_the_policy_survives_the_copied_execution_snapshot_copy():
    # copied_execution.capture copies every public model attribute through continuation_state.tree
    from mimo_bridge.continuation_state import tree
    p = PrecisionPolicy(torch.device("cuda"), requested="bf16")
    q = tree(p)
    assert q is not p and q == p
    assert q.device == torch.device("cuda") and q.dtype == torch.bfloat16 and q.use_amp is True
    assert q.autocast(enabled=False).device == "cuda"
    q = tree(PrecisionPolicy(torch.device("mps"), mps_autocast="fp16"))
    assert q.dtype == torch.float16 and q.use_amp is True and q.mps_autocast == "fp16"


def test_lemlite_on_cpu_carries_a_float32_policy_with_autocast_off():
    from mimo_bridge.lem_model import LemLite
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False)
    assert isinstance(m.precision, PrecisionPolicy)
    assert m.precision.device.type == "cpu"
    assert m.dtype == m.precision.dtype == torch.float32
    assert m.use_amp is m.precision.use_amp is False
    assert m.precision.requested == m.config["model"].get("precision", "fp32") == "bf16"
    ctx = m.precision.autocast()
    assert ctx.device == "cpu" and ctx._enabled is False
    with pytest.raises(ValueError, match="acts on an mps device only"):
        LemLite(device="cpu", seed=0, intention=False, imagination=False, mps_autocast="bf16")


def test_lemlite_takes_an_injected_policy_for_its_own_device_type():
    # the composition root may build the policy and hand it in (specification section 5: injected into LemLite)
    from mimo_bridge.lem_model import LemLite
    given = PrecisionPolicy.from_attributes("cpu", dtype=torch.bfloat16, use_amp=True)
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False, precision=given)
    assert m.precision is given and m.dtype == torch.bfloat16 and m.use_amp is True
    # a policy for another device type is refused before any module is built
    with pytest.raises(ValueError, match="device type"):
        LemLite(device="cpu", seed=0, intention=False, imagination=False,
                precision=PrecisionPolicy(torch.device("cuda"), requested="bf16"))
    # a policy and --mps-autocast together: one of them would be ignored, so both are refused
    with pytest.raises(ValueError, match="not both"):
        LemLite(device="cpu", seed=0, intention=False, imagination=False,
                precision=PrecisionPolicy("cpu"), mps_autocast="bf16")
