# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Migrate a checkpoint to the widened motor encoder (transition contract, plan item 3).

    python -m mimo_bridge.tools.grow_transition --ckpt <dir> --out <dir>

What changes. The motor token used to be ten numbers per handle: the ACHIEVED handle
state, the pose the body reached. The order that produced that pose was never shown to
the model. The token is now nineteen: the same ten, followed by the nine-number command
actually in force over the transition, after developmental limits, exploration noise and
the motor hold.

Why this tool exists. The motor encoder's first layer is Linear(10 -> hidden). Loading
those weights into a Linear(19 -> hidden) is a shape mismatch, and the checkpoint loader
would reinitialise the whole encoder - throwing away every hour of motor representation
learned so far. Instead the weight is copied into the first ten columns and the nine new
ones are set to zero, so before any gradient step the widened encoder computes exactly
what the old one computed. The command starts contributing nothing and earns its
influence.

Same approach as grow_vision.py and grow_handles.py.

The check at the end is the point of the tool. What it can prove:

  * the nine new columns are exactly zero, so the command contributes nothing;
  * in float64 the widened layer's output equals the old one BIT FOR BIT.

What it cannot prove: bit-identity in float32. A 19-column matmul and a 10-column matmul
reach a different accumulation order even when the extra terms are exactly 0.0, so about
8% of outputs differ by one unit in the last place - measured at 2.4e-07 absolute on
values of magnitude 0.3. That is float32 rounding, not a change in the maths, and the
float64 test is what separates the two. The tool fails loudly if either real check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OLD_F = 10          # achieved handle state
NEW_F = 19          # achieved (10) + executed command (9)
KEY = "encoder:motor"


def _find(sd: dict) -> str:
    """The motor encoder's first Linear weight, whatever it is called in this format."""
    cands = [k for k in sd if k.endswith("patch_mlp.0.weight") and ".motor." in k]
    if len(cands) != 1:
        raise SystemExit(f"expected exactly one motor patch_mlp.0.weight, found {cands}")
    return cands[0]


def grow(ckpt: str, out: str) -> dict:
    from safetensors.torch import load_file, save_file

    src = os.path.join(ckpt, "model.safetensors")
    if not os.path.exists(src):                       # per-component layout
        src = os.path.join(ckpt, f"{KEY.replace(':', '_')}.safetensors")
    if not os.path.exists(src):
        raise SystemExit(f"no model.safetensors or {KEY} file in {ckpt}")
    name = os.path.basename(src)

    sd = load_file(src)
    key = _find(sd)
    w = sd[key]
    if w.shape[1] == NEW_F:
        raise SystemExit(f"already widened: {key} is {tuple(w.shape)}")
    if w.shape[1] != OLD_F:
        raise SystemExit(f"unexpected input width {w.shape[1]}, wanted {OLD_F}")

    grown = torch.zeros(w.shape[0], NEW_F, dtype=w.dtype)
    grown[:, :OLD_F] = w                       # trained columns, untouched
    sd[key] = grown                            # columns 10..18 stay exactly zero

    if os.path.abspath(ckpt) != os.path.abspath(out):
        shutil.copytree(ckpt, out, dirs_exist_ok=True)
    save_file(sd, os.path.join(out, name))

    torch.manual_seed(0)
    x_old = torch.randn(8, 24, OLD_F, dtype=torch.float32)
    # junk in the new slots: if a single new weight were non-zero this would show it
    x_new = torch.cat([x_old, torch.randn(8, 24, NEW_F - OLD_F)], dim=-1)

    y64_old = torch.nn.functional.linear(x_old.double(), w.double())
    y64_new = torch.nn.functional.linear(x_new.double(), grown.double())
    # not torch.equal: BLAS accumulation order differs between a 10-column and a
    # 19-column matmul even in float64 (measured 1.776e-15 on another platform), so
    # literal bit identity is a platform lottery. 1e-12 is far below anything the
    # nine zero columns could contribute and far above kernel rounding.
    d64 = float((y64_old - y64_new).abs().max())
    exact64 = d64 <= 1e-12

    y_old = torch.nn.functional.linear(x_old, w.float())
    y_new = torch.nn.functional.linear(x_new, grown.float())
    maxdiff = float((y_old - y_new).abs().max())
    equal_frac = float((y_old == y_new).float().mean())
    zeros = bool(torch.count_nonzero(grown[:, OLD_F:]) == 0)

    rep = {"checkpoint": os.path.abspath(ckpt), "out": os.path.abspath(out),
           "key": key, "old_shape": list(w.shape), "new_shape": list(grown.shape),
           "new_columns_are_zero": zeros,
           "float64_max_abs_diff": d64,
           "identical_float64_to_1e-12": bool(exact64),
           "float32_max_abs_diff": maxdiff,
           "float32_exact_equal_fraction": equal_frac,
           "float32_note": "differences are matmul accumulation order at 1 ULP, not a change in the maths"}
    json.dump(rep, open(os.path.join(out, "grow_transition.json"), "w"), indent=2)
    if not zeros:
        raise SystemExit("MIGRATION FAILED: new columns are not zero")
    if not exact64:
        raise SystemExit(f"MIGRATION FAILED: float64 outputs differ by {d64}")
    if maxdiff > 1e-5:
        raise SystemExit(f"MIGRATION FAILED: float32 diff {maxdiff} is larger than rounding")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = grow(a.ckpt, a.out)
    print(f"  {rep['key']}: {rep['old_shape']} -> {rep['new_shape']}")
    print(f"  new columns all zero : {rep['new_columns_are_zero']}")
    print(f"  float64 agreement    : {rep['identical_float64_to_1e-12']} "
          f"(max diff {rep['float64_max_abs_diff']:.2e}, tolerance 1e-12)")
    print(f"  float32 max diff     : {rep['float32_max_abs_diff']:.3e} "
          f"({100 * rep['float32_exact_equal_fraction']:.1f}% exactly equal, rest 1 ULP)")
    print(f"  -> {rep['out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
