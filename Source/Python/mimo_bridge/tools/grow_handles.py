# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Grow a 23-handle checkpoint into the 24-handle body (the eyes become handle 23).

    python -m mimo_bridge.tools.grow_handles <ckpt_dir> --out DIR

Where the new numbers go and why the brain is untouched. The eyes joined the body plan
(lem/model/body_sets.py): one more tracked bone in proprioception, one more handle in the
motor command and echo. Five tensors change shape; every new weight is ZERO, so on the
same inputs the grown network computes bit-for-bit what the old one computed, plus an
inert 24th row. The one exception is a bias: the gaze handle's quaternion-w output bias
is set to 1.0, so the commanded eye orientation starts as the identity (looking straight
ahead) instead of the zero quaternion.

  proprio encoder mlp.0.weight   (128,276) -> (128,288)  12 zero cols appended (bone-major,
                                                          the eye bone is last)
  motor_controller.input_proj    (128,1205) -> (128,1224) input is [latents 768 | handle
                                                          state 230->240 | wellness grad
                                                          207->216]: 10 zero cols spliced
                                                          at 998, 9 appended at the end
  motor_controller.output_proj   (207,128) -> (216,128)   9 zero rows appended; bias row
                                                          213 (gaze qw) = 1.0
  wellness motor_state net.0     (64,207)  -> (64,216)    9 zero cols appended

Optimizer moments no longer match and are set aside (re-warm beats a shape crash).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

GROW = {
    "encoder_registry.encoders.proprioception.mlp.0.weight": ("append_cols", 12),
    "motor_controller.input_proj.weight": ("splice_input", None),
    "motor_controller.output_proj.weight": ("append_rows", 9),
    "motor_controller.output_proj.bias": ("append_bias", 9),
    "wellness.grounded.motor_state.net.0.weight": ("append_cols", 9),
}


def grow(tensors: dict) -> list:
    changed = []
    for key, (how, n) in GROW.items():
        if key not in tensors:
            raise KeyError(f"expected {key} in the checkpoint")
        w = tensors[key]
        if how == "append_cols":
            new = torch.zeros(w.shape[0], w.shape[1] + n, dtype=w.dtype)
            new[:, : w.shape[1]] = w
        elif how == "append_rows":
            new = torch.zeros(w.shape[0] + n, w.shape[1], dtype=w.dtype)
            new[: w.shape[0]] = w
        elif how == "append_bias":
            new = torch.zeros(w.shape[0] + n, dtype=w.dtype)
            new[: w.shape[0]] = w
            new[w.shape[0] + 6] = 1.0        # gaze handle's qw -> identity orientation
        elif how == "splice_input":
            assert w.shape[1] == 1205, f"input_proj is {tuple(w.shape)}, expected (*,1205)"
            new = torch.zeros(w.shape[0], 1224, dtype=w.dtype)
            new[:, :998] = w[:, :998]                      # latents + 23 handles' state
            new[:, 1008:1215] = w[:, 998:1205]             # the 207 wellness-grad cols
        tensors[key] = new
        changed.append(f"{key}: {tuple(w.shape)} -> {tuple(new.shape)}")
    return changed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src, dst = os.path.abspath(a.ckpt_dir), os.path.abspath(a.out)
    tensors = load_file(os.path.join(src, "model.safetensors"))
    changed = grow(tensors)
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name not in ("model.safetensors", "optimizer.pt"):
            p = os.path.join(src, name)
            (shutil.copytree if os.path.isdir(p) else shutil.copy2)(p, os.path.join(dst, name))
    save_file(tensors, os.path.join(dst, "model.safetensors"))
    json.dump({"grown_from": src, "changes": changed,
               "note": "all new weights zero except gaze qw bias = 1 (identity gaze); "
                       "optimizer state set aside"},
              open(os.path.join(dst, "handle_growth.json"), "w"), indent=2)
    print("\n".join("  " + c for c in changed))
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
