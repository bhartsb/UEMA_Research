# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The trainer's AdamW optimizer, built in one place, with the fused-kernel rule beside it.

WHY. lem_model.py built its optimizer with ``torch.optim.AdamW(..., fused=(d.type == "cuda"))``: the rule that
chooses torch's fused CUDA kernel sat inside the model's constructor, untested, and the macOS and iOS port (specification
docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html, section "Target A"; work item WI-4) adds two more device types
(cpu and mps) that must take the per-tensor loop. This module holds that rule once, so a device the rule does not
cover is a change here with a test, not a search through lem_model.py's 9,984 lines (wc -l, 2026-11-04).

WHAT IT DOES (no numerical change). make_adamw(params, device, lr, betas) returns ``torch.optim.AdamW`` with the
same parameter groups, learning rate, betas and weight decay the caller passes; the only thing decided here is the
``fused`` flag: True when the device type is "cuda", False for every other device type ("cpu", "mps"). torch's fused
AdamW (``torch.optim.AdamW``, argument ``fused``) is one CUDA kernel over all the parameters; with ``fused=False``
torch runs its per-tensor loop, the implementation the released cpu tests exercise. Whether the fused kernel and the
loop agree to the last bit on CUDA is not measured in this release; the rule only reproduces what the trainer did
on 2026-11-04 (lem_model.py, the optimizer built after the two weight-decay groups).

THE RULE MUST BE REAPPLIED AFTER A CHECKPOINT LOAD. ``torch.optim.Optimizer.load_state_dict`` (torch 2.6.0,
torch/optim/optimizer.py, its inner ``update_group``) keeps every setting of the SAVED parameter groups and puts only
the live parameters into them. The released checkpoint ckpt_step_494025's optimizer.pt saves ``fused: True`` on both
of its groups (297 and 379 parameters; read 2026-11-04), so a cpu or mps resume would step on torch's fused cpu or mps
kernel (torch 2.6.0 lists "mps", "cuda", "xpu", "cpu" as fused-capable in ``_get_fused_kernels_supported_devices``),
not on the loop. apply_fused_rule(optimizer, device) puts the rule back on every group; lem_model.load_checkpoint
calls it after its optimizer loads and migrations.

Inputs:  params  - what ``torch.optim.AdamW`` accepts: an iterable of parameter tensors, or a list of parameter-group
                   dicts (each with "params" and any per-group setting such as "weight_decay");
         device  - the device the parameters live on, a ``torch.device`` or its string ("cuda", "cuda:0", "cpu",
                   "mps"); only its ``.type`` is read;
         lr      - the learning rate, one float, applied to every group that does not set its own;
         betas   - the two Adam moment decay factors, (beta1, beta2);
         weight_decay - the decay applied to a group that does not set its own; default 1e-2, ``torch.optim.AdamW``'s
                   own default.
Output:  a ``torch.optim.AdamW``; ``fused_for(device)`` is the flag it was built with; ``apply_fused_rule`` returns
         how many groups it changed (0 when the rule already held).

SOLID. One responsibility (build the optimizer and keep its fused flag to the rule); the trainer depends on these
functions, not on the torch keyword.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Union

import torch

# Device types on which the optimizer takes torch's fused kernel. Every other device type takes the per-tensor loop.
# "cuda" alone, the rule lem_model.py held on 2026-11-04. torch 2.6.0 also offers fused kernels for "cpu" and "mps"
# (torch/optim/optimizer.py, _get_fused_kernels_supported_devices returns ["mps", "cuda", "xpu", "cpu", ...]); the
# trainer never used them and their agreement with the loop is unmeasured here, so they stay off.
FUSED_DEVICE_TYPES: Tuple[str, ...] = ("cuda",)

ParamsLike = Union[Iterable[torch.Tensor], Sequence[dict]]


def fused_for(device: Union[torch.device, str]) -> bool:
    """The fused rule: True when ``device``'s type is in FUSED_DEVICE_TYPES ("cuda"), else False.

    Input: a ``torch.device`` or a device string ("cuda", "cuda:1", "cpu", "mps"); the index is ignored.
    Operation: ``torch.device(device).type in FUSED_DEVICE_TYPES``.
    Output: the bool handed to ``torch.optim.AdamW`` as ``fused``.
    """
    return torch.device(device).type in FUSED_DEVICE_TYPES


def make_adamw(params: ParamsLike,
               device: Union[torch.device, str],
               lr: float,
               betas: Tuple[float, float],
               weight_decay: float = 1e-2) -> torch.optim.AdamW:
    """Build ``torch.optim.AdamW`` for ``params`` on ``device`` with ``fused=fused_for(device)``.

    Inputs: see the module docstring. Operation: one call to ``torch.optim.AdamW(params, lr=lr, betas=tuple(betas),
    weight_decay=weight_decay, fused=fused_for(device))``; nothing else is set, so eps stays torch's 1e-8 and
    amsgrad stays off, as in the trainer. Output: the optimizer, with its parameter groups in the order given.
    """
    return torch.optim.AdamW(params, lr=lr, betas=tuple(betas), weight_decay=weight_decay,
                             fused=fused_for(device))


def apply_fused_rule(optimizer: torch.optim.Optimizer, device: Union[torch.device, str]) -> int:
    """Set every parameter group's ``fused`` to ``fused_for(device)``; call it after each optimizer state load.

    Input: the live optimizer (its ``param_groups`` list of dicts) and the device its parameters live on.
    Operation: for each group, ``group["fused"] = fused_for(device)`` where the stored value differs (a saved
    ``True`` from a cuda checkpoint loaded on cpu or mps, or torch's ``None`` default).
    Output: the number of groups whose flag changed; 0 means the rule already held. The groups are changed in place.
    """
    want = fused_for(device)
    changed = 0
    for group in optimizer.param_groups:
        if group.get("fused") is not want:
            group["fused"] = want
            changed += 1
    return changed
