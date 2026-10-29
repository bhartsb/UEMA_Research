# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Widen a checkpoint's angular position encoder so a trained brain keeps its eyes.

    python -m mimo_bridge.tools.grow_vision <ckpt_dir> [--inputs 4] [--out DIR]

What changes and why. Every patch reaching the vision encoder carries positional numbers
saying where it is looking. Two of them - azimuth and elevation - were enough for one
uniform camera per eye. A binocular, foveated feed needs two more:

    eye     -1 left, +1 right
    span    how much of the world one patch covers, in radians

The eye number is the one that matters most. Everything before the encoder's pool is
computed per patch, and the pool is a single learned query over the whole set, so with only
(azimuth, elevation) the pooled token comes out identical whether an object sits in the left
eye or the right one. Disparity - the only depth cue two eyes give you - was not merely
unused, it was not representable. The span number lets one encoder serve a coarse wide
camera and a sharp narrow one at the same time.

Why the checkpoint needs surgery. That is one weight matrix, angular_pos_enc.mlp.0.weight,
shaped (hidden, 2). Widening it to (hidden, 4) is a SHAPE change, and CheckpointManager
loads each component with strict=False - which tolerates missing and unexpected keys but
not a shape clash: the clash raises, the exception is caught, and the whole component is
logged "reinitialized". Resuming without this tool would silently throw away the entire
trained vision encoder, not just the two columns.

How the new columns are filled: with ZERO. The encoder then starts out computing exactly
what it computed before - the two new numbers contribute nothing until training gives them
a use - so the brain carries over intact and stereo is something it learns rather than a
shock it has to recover from.

Nothing else in the model is touched. Patch COUNT needs no surgery at all: the encoder is
per-patch and window-based, so it already accepts any number of patches.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

KEY = "encoder_registry.encoders.vision.angular_pos_enc.mlp.0.weight"


def widen(w: torch.Tensor, n_new: int) -> torch.Tensor:
    hidden, n_old = w.shape
    if n_new <= n_old:
        return w[:, :n_new].clone()
    out = torch.zeros(hidden, n_new, dtype=w.dtype)
    out[:, :n_old] = w
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--inputs", type=int, default=4, help="positional numbers per patch (default 4)")
    ap.add_argument("--blocks", type=int, default=0,
                    help="grow the encoder to this many attention blocks (0 = leave alone). "
                         "New blocks are written as exact pass-throughs, so the brain carries "
                         "over unchanged and the block learns from nothing rather than from noise.")
    ap.add_argument("--out", default=None, help="output folder (default: <ckpt>_pos<N>)")
    a = ap.parse_args(argv)

    src = os.path.abspath(a.ckpt_dir)
    dst = os.path.abspath(a.out or f"{src.rstrip(os.sep)}_pos{a.inputs}")
    weights = os.path.join(src, "model.safetensors")
    if not os.path.exists(weights):
        print(f"no model.safetensors in {src}")
        return 2

    tensors = load_file(weights)

    added_blocks = []
    if a.blocks > 0:
        # A fresh attention block dropped on top of a trained one perturbs everything
        # downstream on the first step. Written as a pass-through instead, it changes
        # nothing: a block computes x + attn(norm(x)) then x + ffn(norm(x)), so zeroing the
        # two output projections makes both terms zero and the block returns x exactly. The
        # rest of its weights are copied from the trained block, so when the zeros start
        # moving they move from sensible statistics rather than from random ones.
        pre = "encoder_registry.encoders.vision.blocks."
        have = sorted({int(k[len(pre):].split(".")[0]) for k in tensors if k.startswith(pre)})
        if not have:
            print(f"no vision blocks found under {pre}")
            return 4
        src_i = have[-1]
        for new_i in range(max(have) + 1, a.blocks):
            for k in [k for k in list(tensors) if k.startswith(f"{pre}{src_i}.")]:
                tail = k[len(f"{pre}{src_i}."):]
                v = tensors[k].clone()
                if tail in ("attn.out_proj.weight", "attn.out_proj.bias", "ffn.3.weight", "ffn.3.bias"):
                    v = torch.zeros_like(v)          # the two residual outputs -> identity
                tensors[f"{pre}{new_i}.{tail}"] = v
            added_blocks.append(new_i)

    if KEY not in tensors:
        print(f"{KEY} not found; keys like it: "
              f"{[k for k in tensors if 'angular' in k][:4]}")
        return 3
    old = tensors[KEY]
    nothing_to_do = old.shape[1] == a.inputs and not added_blocks
    if nothing_to_do and not a.out:
        print(f"already {a.inputs} inputs and {a.blocks or 'enough'} blocks; nothing to do")
        return 0
    # With --out given, "nothing to do" must still produce that folder. It used to print
    # and return 0 without writing anything, so a caller that grew a checkpoint and then
    # resumed from --out got FileNotFoundError from a step that reported success.
    tensors[KEY] = widen(old, a.inputs)

    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name != "model.safetensors":
            p = os.path.join(src, name)
            (shutil.copytree if os.path.isdir(p) else shutil.copy2)(p, os.path.join(dst, name))
    save_file(tensors, os.path.join(dst, "model.safetensors"))

    # the optimizer's moment estimates for that matrix no longer match its shape; setting
    # them aside re-warms in a few hundred steps, where a shape clash crashes at step one
    opt = os.path.join(dst, "optimizer.pt")
    if os.path.exists(opt):
        os.replace(opt, opt + ".pre_vision_growth")

    note = {"grown_from": src, "old_inputs": int(old.shape[1]), "new_inputs": int(a.inputs),
            "hidden": int(old.shape[0]), "parameters_added": int(old.shape[0] * (a.inputs - old.shape[1])),
            "blocks_added": added_blocks,
            "note": "angular_pos_enc.mlp.0.weight widened with ZERO columns; any new "
                    "attention block written as an exact pass-through (both residual output "
                    "projections zeroed); every other tensor copied unchanged. optimizer "
                    "state set aside (shape clash)."}
    json.dump(note, open(os.path.join(dst, "vision_growth.json"), "w"), indent=2)
    print(f"widened angular position inputs {old.shape[1]} -> {a.inputs} "
          f"({note['parameters_added']} new weights, all zero, of "
          f"{sum(t.numel() for t in tensors.values()):,})")
    if added_blocks:
        print(f"added attention block(s) {added_blocks} as pass-throughs "
              f"(output projections zeroed, so the encoder computes exactly what it did before)")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
