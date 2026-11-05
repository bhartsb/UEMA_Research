# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Where a float64 tensor lives when the model's device cannot hold one (REQ-A-4 of
docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html).

The MPS backend (torch.backends.mps) has no float64 ("The MPS backend does not support
torch.float64": a float64 tensor cannot be created on, or copied to, an mps device). The released
model keeps float64 working tensors beside its float32 parameters at five sites (the
specification's REQ-A-4 list): the hand response calibration's two buffers (coefficients, ranges)
and its three search tensors (initial angles, velocity, target pose; lem/model/hand_response.py),
the persistent memory repair's six running weights (lem/model/persistent_memory.py), the two hand
joint readings (angles, velocities; mimo_bridge/lem_model.py), the self-state clock
(mimo_bridge/self_state_integration.py) and the executed-command spread's two running tensors
(mean, mean square; mimo_bridge/prediction_audit.py). Each of those sites allocates its float64
tensor on float64_device(model device) and hands the result to the model as float32
(to_model_device). On cpu and cuda float64_device is the model's own device, so nothing changes
there.

Four parts, one job each:
  float64_device(device)        the device that holds float64 tensors for a model on `device`
  same_device(a, b)             whether two devices are one device ("cuda" and "cuda:0" are)
  to_model_device(value, device) a float64 result handed back to the model's device as float32
  KeepsFloat64OnHost            an nn.Module base whose .to(device) / .cuda() / .cpu() keep every
                                float64 buffer on float64_device(target) instead of on target
"""
from __future__ import annotations

from typing import Callable, Union

import torch
from torch import nn

DeviceLike = Union[str, torch.device]

# The one device type without float64 (torch's MPS backend notes, read 2026-11-04 for research1 torch-apple).
_NO_FLOAT64 = ("mps",)


def float64_device(device: DeviceLike) -> torch.device:
    """The device that holds float64 tensors for a model running on `device`.

    Input: the model's device as a string ("mps", "cuda:0", "cpu") or a torch.device.
    Operation: the device's type is compared with the types that have no float64 (mps only).
    Output: torch.device("cpu") for an mps device; otherwise the same device as a torch.device
    (so "cuda:0" stays "cuda:0" and "cpu" stays "cpu").
    """
    dev = torch.device(device)
    return torch.device("cpu") if dev.type in _NO_FLOAT64 else dev


def same_device(a: DeviceLike, b: DeviceLike) -> bool:
    """Whether two devices are one device: equal types, and equal indexes once a missing index
    reads 0 (torch.device("cuda") == torch.device("cuda:0") is False in torch; a tensor on the
    default card reports cuda:0). Input: two devices. Output: True or False."""
    a, b = torch.device(a), torch.device(b)
    if a.type != b.type:
        return False
    return (a.index if a.index is not None else 0) == (b.index if b.index is not None else 0)


def to_model_device(value: torch.Tensor, device: DeviceLike, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """A result computed on float64_device handed to the model's device.

    Input: `value`, a tensor (float64 on the float64 host at the call sites); `device`, the model's
    device; `dtype`, what it becomes when it has to move (float32).
    Operation: when `value` is already on `device` (cpu and cuda: always, since the host is the
    model's device) it is returned as it is, dtype untouched; otherwise it is copied to `device`
    as `dtype`, through torch.Tensor.to, so a gradient still flows back across the copy.
    Output: the tensor on the model's device.
    """
    return value if same_device(value.device, device) else value.to(device=device, dtype=dtype)


class KeepsFloat64OnHost(nn.Module):
    """An nn.Module base for modules that register float64 buffers.

    torch moves every parameter and buffer of a module through Module._apply(fn) for .to(device),
    .cuda(), .cpu(), .double(), .half() and .share_memory(). This base replaces fn for float64
    tensors only: it first runs fn on a 0-dimensional float32 probe (one number, on the float64
    tensor's own device) to learn the device fn would put a tensor on, then keeps a float64 tensor
    on float64_host(that device) when the two differ (an mps target: the buffer stays on cpu) and
    runs fn unchanged when they are the same (cpu, cuda: nothing changes, including dtype
    conversions such as .double()).

    float64_host is the policy; a subclass or a test replaces it with its own callable
    (staticmethod) from device to device, so the behaviour is testable without an mps device.
    """

    float64_host: Callable[[DeviceLike], torch.device] = staticmethod(float64_device)

    def _apply(self, fn, recurse: bool = True):
        policy = type(self).float64_host

        def guarded(t: torch.Tensor) -> torch.Tensor:
            if t.dtype != torch.float64:
                return fn(t)
            probe = fn(torch.zeros((), dtype=torch.float32, device=t.device))
            host = policy(probe.device)
            if same_device(host, probe.device):
                return fn(t)
            return t.to(device=host)

        return super()._apply(guarded, recurse)
