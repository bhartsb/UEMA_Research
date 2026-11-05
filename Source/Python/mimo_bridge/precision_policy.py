# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Mixed precision per device for LemLite (Apple port specification, REQ-A-2 and REQ-A-3).

One table decides, from the model's torch.device, which number format the forward model and the
prediction heads run in and whether torch autocast is on. Before this module the rule stood on three
lines of lem_model.py (dtype from the config's "precision", float32 and autocast off unless the device
type was "cuda") and nine lem_model.py autocast contexts, three switch checks and six proprio_token.py
contexts named the device type "cuda" as a literal. Every one of those now takes the device type from
the policy, so a model on "cpu" or "mps" opens an autocast context for its own device.

The table (inputs: device.type, the config's "precision" string, the --mps-autocast setting):
  cuda: the config's precision ("bf16" on the released command: the checkpoint receipt records
        "precision": "bf16"), autocast ON when that precision is not fp32. Unchanged from the three
        old lines, so every CUDA number is the number the previous code produced.
  cpu:  float32, autocast OFF (unchanged).
  mps:  float32, autocast OFF unless --mps-autocast names "bf16" or "fp16": then that dtype, autocast
        ON. Off by default because torch's MPS autocast accepts bfloat16 and float16 only
        (torch/amp/autocast_mode.py, the "mps" branch, torch 2.6.0) and how its bf16 rounding compares
        with the CUDA bf16 of the receipts is unmeasured (specification, open question 9).
  any other device type: float32, autocast OFF; autocast() and full_precision() return a no-op context
        (contextlib.nullcontext) when torch.amp.is_autocast_available(device type) is False, since torch 2.6.0
        raises RuntimeError "unsupported scalarType" building even a disabled context for such a type
        ("meta", measured 2026-11-04 on torch 2.6.0+cu124).

scaler() returns None on every device: no torch.amp.GradScaler exists on the MIMo path (lem_model.py's
checkpoint load and optimizer carry none; lem/live_test.py's scaler is the Unreal path), and bf16 needs
none (lem/master_config.yaml:281).

Outputs: dtype (torch.dtype), use_amp (bool), autocast(enabled=None) (a torch.autocast context for
device.type), scaler() (None), name ("bf16" | "fp16" | "fp32" of the dtype in use), receipt() (a dict
for args.json).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import ContextManager, Optional, Union

import torch

# the config's "precision" names (lem/master_config.yaml: fp16 | bf16 | fp32) to torch dtypes
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
# the --mps-autocast settings torch's MPS autocast accepts (autocast_mode.py, "mps" branch)
MPS_AUTOCAST_CHOICES = ("bf16", "fp16")

DeviceLike = Union[torch.device, torch.Tensor, str]


def _device_type(device: DeviceLike) -> str:
    """The device type string ("cuda", "cpu", "mps") of a torch.device, a tensor or a device string."""
    if isinstance(device, torch.Tensor):
        return device.device.type
    return torch.device(device).type


def dtype_name(dtype: torch.dtype) -> str:
    """The config's name of a dtype: "bf16" for torch.bfloat16, "fp16" for torch.float16, "fp32" otherwise."""
    for name, value in DTYPES.items():
        if value == dtype:
            return name
    return "fp32"


@dataclass
class PrecisionPolicy:
    """The dtype and autocast rule of one model on one device (the module docstring's table).

    Inputs: device (a torch.device, a tensor or a device string; its type picks the table row);
    requested (the config's "precision": "fp16", "bf16" or "fp32", read on cuda only); mps_autocast
    (None, "bf16" or "fp16"; read on mps only, and refused with ValueError on any other device so a
    launch line never carries a setting that does nothing).
    Operation: the table lookup, once, in __post_init__.
    Outputs: self.dtype, self.use_amp; autocast(), scaler(), name, receipt().

    A dataclass whose fields are a torch.device, strings, a bool and None on purpose: LemLite carries the
    policy as a public attribute, and the copied-execution snapshot (mimo_bridge/copied_execution.capture)
    copies every public attribute through continuation_state.tree, which accepts exactly those field types
    (a torch.dtype it refuses, so the dtype in use is held as its config name, dtype_in_use). The snapshot's
    restore (copied_execution.restore) puts the SAVED run's policy back on the model, device and all, as it
    does for every public attribute outside copied_execution._STATIC; a companion captured on cuda and
    restored on another device type carries its cuda policy until _STATIC lists "precision".
    """

    device: torch.device
    requested: str = "fp32"
    mps_autocast: Optional[str] = None
    # set by the table: the config name of the dtype the autocast regions run in, and whether they open
    dtype_in_use: str = field(init=False, default="fp32")
    use_amp: bool = field(init=False, default=False)

    def __post_init__(self):
        device = self.device
        self.device = device.device if isinstance(device, torch.Tensor) else torch.device(device)
        self.requested = str(self.requested)
        kind = self.device.type
        if self.mps_autocast is not None and kind != "mps":
            raise ValueError("--mps-autocast %r acts on an mps device only; the model is on %r"
                             % (self.mps_autocast, kind))
        if kind == "cuda":
            self.dtype_in_use = self.requested if self.requested in DTYPES else "fp32"
            self.use_amp = self.dtype_in_use != "fp32"
        elif kind == "mps" and self.mps_autocast is not None:
            if self.mps_autocast not in MPS_AUTOCAST_CHOICES:
                raise ValueError("--mps-autocast must be one of %s, not %r"
                                 % (MPS_AUTOCAST_CHOICES, self.mps_autocast))
            self.dtype_in_use = self.mps_autocast
            self.use_amp = True
        else:
            self.dtype_in_use = "fp32"
            self.use_amp = False

    @classmethod
    def from_attributes(cls, device: DeviceLike, dtype: torch.dtype = torch.float32,
                        use_amp: bool = False) -> "PrecisionPolicy":
        """A policy with a given dtype and use_amp instead of the table's (for a model object that carries
        only the two attributes, as the switch checks' test doubles do). Inputs: device, dtype, use_amp.
        Output: a PrecisionPolicy whose autocast() opens for device.type with exactly those settings."""
        policy = cls(device, requested=dtype_name(dtype))
        policy.dtype_in_use = dtype_name(dtype)
        policy.use_amp = bool(use_amp)
        return policy

    @property
    def dtype(self) -> torch.dtype:
        """The torch dtype of dtype_in_use (torch.bfloat16 on the released cuda command, torch.float32 on
        cpu and on mps without --mps-autocast)."""
        return DTYPES[self.dtype_in_use]

    @property
    def name(self) -> str:
        """"bf16", "fp16" or "fp32": the dtype the autocast regions run in (fp32 whenever autocast is off)."""
        return self.dtype_in_use if self.use_amp else "fp32"

    def autocast(self, enabled: Optional[bool] = None) -> ContextManager:
        """A torch.autocast context for this device's type. Input: enabled (None = this policy's use_amp;
        False = full precision, the way the synergy lesson and the slot watcher run; True = the policy's
        dtype). Output: the context manager (torch.autocast(device_type=self.device.type, ...)); when off,
        no dtype is passed, so torch does not warn about float32 on cpu or mps. For a device type torch's
        autocast does not serve (torch.amp.is_autocast_available False, e.g. "meta") the output is
        contextlib.nullcontext(): the code inside runs in the tensors' own dtype, which is the table's row."""
        on = self.use_amp if enabled is None else bool(enabled)
        if not torch.amp.is_autocast_available(self.device.type):
            return contextlib.nullcontext()
        if on:
            return torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=True)
        return torch.autocast(device_type=self.device.type, enabled=False)

    def scaler(self) -> None:
        """None on every device: the MIMo path has no GradScaler (module docstring)."""
        return None

    def receipt(self) -> dict:
        """What args.json and the checkpoint receipt may record: {"device_type", "precision" (the name
        in use), "requested" (the config's string), "autocast" (bool), "mps_autocast"}."""
        return {"device_type": self.device.type, "precision": self.name, "requested": self.requested,
                "autocast": bool(self.use_amp), "mps_autocast": self.mps_autocast}


def policy_of(model, device: DeviceLike) -> PrecisionPolicy:
    """The policy a model object runs under. Input: model (LemLite, which carries .precision, or any object
    with .dtype / .use_amp attributes, as the switch checks' test doubles do); device (the parameters'
    device, used when the model carries no policy). Output: model.precision when it is a PrecisionPolicy,
    else PrecisionPolicy.from_attributes(device, model.dtype or float32, model.use_amp or False)."""
    policy = getattr(model, "precision", None)
    if isinstance(policy, PrecisionPolicy):
        return policy
    return PrecisionPolicy.from_attributes(device, dtype=getattr(model, "dtype", torch.float32),
                                           use_amp=bool(getattr(model, "use_amp", False)))


def full_precision(device: DeviceLike) -> ContextManager:
    """An autocast-OFF context for the device of a tensor, a torch.device or a device string: the code
    inside runs in the tensors' own dtype (float32 on this path). Replaces the pair of "cuda" and "cpu"
    contexts proprio_token.py opened, which left an mps model's autocast untouched. A device type torch's
    autocast does not serve (torch.amp.is_autocast_available False) gets contextlib.nullcontext()."""
    kind = _device_type(device)
    if not torch.amp.is_autocast_available(kind):
        return contextlib.nullcontext()
    return torch.autocast(device_type=kind, enabled=False)
