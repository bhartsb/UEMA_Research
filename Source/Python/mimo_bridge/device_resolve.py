# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Device selection for the trainer: one place that turns a device request into a ``torch.device``.

WHY. lem_model.py line 785 read ``torch.device(device if torch.cuda.is_available() else "cpu")`` on 2026-11-04: any
request, "mps" included, became cpu in silence on a machine without CUDA, and a request for a CUDA card that is not
there ran on cpu without a word. The Apple port (specification docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html,
REQ-A-1; work item WI-2 of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html) adds the Metal device
("mps") and needs a request that cannot be served to stop with a message naming it, so a run never trains on a
device other than the one its command asked for.

WHAT IT DOES. resolve_device(name) reads one request and returns the ``torch.device`` the trainer runs on:
  "auto"           - "cuda:0" when torch.cuda.is_available(), else "mps" when torch.backends.mps.is_available(),
                     else "cpu";
  "cpu"            - always;
  "cuda", "cuda:N" - when torch.cuda.is_available() and N is below torch.cuda.device_count();
  "mps"            - when torch.backends.mps.is_available();
  anything else    - ValueError naming the request (a word torch rejects as a device, a type this trainer has no
                     table for such as "xpu", or a type that is not available on this machine).
A ``torch.device`` is accepted in place of the string and goes through the same availability check.

Inputs:  name         - the request: a string from the command line (train_mimo.py --device, resume.py --device, both
                        default "auto") or from a caller building LemLite, or a ``torch.device``;
         availability - the three machine facts the table reads, as an Availability object; None (the default) reads
                        torch at call time, so a test that patches torch.cuda.is_available or
                        torch.backends.mps.is_available is read as patched.
Operation: the table above, in that order; the availability facts are read once per call.
Output:  a ``torch.device`` with type "cuda", "mps" or "cpu"; the CUDA index is the one requested (None when the
         request named no index, 0 under "auto"). resolved_device_name(name) is the same as a string, for argparse.

SOLID. One responsibility (choose the device); callers depend on resolve_device and on the Availability protocol, not
on torch.cuda or torch.backends.mps; the torch reader is injected, never a module-level singleton.
"""
from __future__ import annotations

import argparse
from typing import Optional, Protocol, Tuple, Union

import torch

AUTO = "auto"
# Device types this trainer runs on. Any other torch device type ("xpu", "meta", ...) is refused by name.
DEVICE_TYPES: Tuple[str, ...] = ("cuda", "mps", "cpu")

DeviceLike = Union[str, torch.device]


class Availability(Protocol):
    """The three machine facts resolve_device reads. Implemented by TorchAvailability; a test may inject its own."""

    def cuda_available(self) -> bool:
        """True when a CUDA device can be used (torch.cuda.is_available())."""
        ...

    def cuda_device_count(self) -> int:
        """The number of CUDA devices (torch.cuda.device_count()); 0 when none can be used."""
        ...

    def mps_available(self) -> bool:
        """True when the Metal device can be used (torch.backends.mps.is_available())."""
        ...


class TorchAvailability:
    """Availability read from torch at call time: nothing is cached, so patches on torch are seen."""

    def cuda_available(self) -> bool:
        """Input: none. Operation: torch.cuda.is_available(). Output: that bool."""
        return bool(torch.cuda.is_available())

    def cuda_device_count(self) -> int:
        """Input: none. Operation: torch.cuda.device_count() when CUDA is available, else 0. Output: that count."""
        return int(torch.cuda.device_count()) if self.cuda_available() else 0

    def mps_available(self) -> bool:
        """Input: none. Operation: torch.backends.mps.is_available(); False when torch has no mps backend attribute
        (torch 2.6.0 has it on every platform; the guard covers a build without it). Output: that bool."""
        mps = getattr(torch.backends, "mps", None)
        return bool(mps is not None and mps.is_available())


def resolve_device(name: DeviceLike, availability: Optional[Availability] = None) -> torch.device:
    """Turn a device request into the ``torch.device`` the trainer runs on, or raise ValueError naming the request.

    Input: ``name``, a request string ("auto", "cpu", "cuda", "cuda:N", "mps") or a ``torch.device``; ``availability``,
    the machine facts (None reads torch at call time).
    Operation: the table in the module docstring: "auto" takes the first of cuda:0, mps, cpu that is available; a named
    device is checked against the same facts and returned unchanged when available.
    Output: a ``torch.device`` of type "cuda", "mps" or "cpu". Raises ValueError, with the request in the message,
    for a word torch rejects, a type outside DEVICE_TYPES, a CUDA or mps device that is not available, or a CUDA index
    at or beyond torch.cuda.device_count().
    """
    facts = availability if availability is not None else TorchAvailability()
    if isinstance(name, str):
        name = name.strip().lower()    # " CPU " and "cpu" are the same request; torch's device names are lower case
    if name == AUTO:
        if facts.cuda_available():
            return torch.device("cuda", 0)
        if facts.mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        device = torch.device(name)
    except (RuntimeError, TypeError, ValueError) as err:
        raise ValueError("device %r is not a torch device: %s" % (name, err)) from err
    if device.type not in DEVICE_TYPES:
        raise ValueError("device %r has type %r; this trainer runs on %s"
                         % (str(name), device.type, ", ".join(DEVICE_TYPES)))
    if device.type == "cuda":
        if not facts.cuda_available():
            raise ValueError("device %r requested but no CUDA device is available on this machine "
                             "(torch.cuda.is_available() is False); use --device auto, mps or cpu" % (str(name),))
        count = facts.cuda_device_count()
        if device.index is not None and device.index >= count:
            raise ValueError("device %r requested but this machine has %d CUDA device(s) (indices 0..%d)"
                             % (str(name), count, count - 1))
    elif device.type == "mps":
        if not facts.mps_available():
            raise ValueError("device %r requested but the Metal device is not available on this machine "
                             "(torch.backends.mps.is_available() is False); use --device auto, cuda or cpu"
                             % (str(name),))
    return device


def resolved_device_name(name: str) -> str:
    """argparse ``type`` for a --device option: the resolved device as a string, so args.json records the device the
    run took ("auto" on a CUDA machine records "cuda:0").

    Input: the option's text. Operation: str(resolve_device(name)). Output: "cuda:0", "cuda", "mps" or "cpu".
    Raises argparse.ArgumentTypeError with resolve_device's message, which argparse prints verbatim and exits 2 on.
    """
    try:
        return str(resolve_device(name))
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from err
