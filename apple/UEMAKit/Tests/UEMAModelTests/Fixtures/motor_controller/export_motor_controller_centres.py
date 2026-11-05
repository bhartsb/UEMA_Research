# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Export the two running centres the released motor controller reads at inference (work item WI-36).

WHY. The released motor controller does not read its 25 motor_controller.* tensors alone. Two runtime hooks change
what its layers compute at every call, inference included (measured on the three model-pass fixtures of WI-34; see
apple/UEMAKit/Sources/UEMAModel/MotorController.swift, header section "WHAT THE RELEASED CONTROLLER COMPUTES"):
  1. motor_input_fold (lem/model/motor_input_fold.py:183-198): the input projection's 1,152 token columns and the
     confidence head read the newest tokens minus persistent_memory.readout.bias (1,152 numbers, in model.safetensors).
  2. liveness_repair (lem/model/liveness.py:355-361, hook at :718-728): in motor_controller.confidence_head.0 (84 of
     128 units revived) and motor_controller.residual_blocks.0.0 (127 of 128), each revived unit's pre-activation is
     lin(LayerNorm(x - centre_mean))[unit]. mask, norm and lin are in model.safetensors (liveness_repair.rows.*);
     centre_mean is a running statistic that rides runtime_state.pt only (liveness.py:1132-1134), a torch pickle the
     device cannot read. This script copies those two vectors into JSON.

INPUT. runtime_state.pt of weights/b1_continue_B1K_skip13_492800/run/checkpoints/ckpt_step_494025 (path given by
--checkpoint; default is that folder under the repository root).
OPERATION. torch.load the file; read runtime_state['liveness_repair'][key]['centre_mean'] and ['centre_weight'] for
key motor_controller__confidence_head__0 (1,152 numbers) and motor_controller__residual_blocks__0__0 (128 numbers);
record the SHA-256 of runtime_state.pt and model.safetensors.
OUTPUT. apple/UEMAKit/Sources/UEMAModel/Resources/motor_controller_centres.json (--out), float32 values written with
repr(float(numpy.float32)), so the Swift Float parse returns the same 32-bit number.

Run with an interpreter that has torch (the training environment): python export_motor_controller_centres.py
"""
import argparse
import hashlib
import json
import os

KEYS = {"confidence_head_0": ("motor_controller__confidence_head__0", 1152),
        "residual_blocks_0_0": ("motor_controller__residual_blocks__0__0", 128)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, *([os.pardir] * 6)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(
        root, "weights", "b1_continue_B1K_skip13_492800", "run", "checkpoints", "ckpt_step_494025"))
    ap.add_argument("--out", default=os.path.join(
        root, "apple", "UEMAKit", "Sources", "UEMAModel", "Resources", "motor_controller_centres.json"))
    a = ap.parse_args()
    import numpy as np
    import torch
    rs_path = os.path.join(a.checkpoint, "runtime_state.pt")
    rs = torch.load(rs_path, map_location="cpu", weights_only=False)
    out = {"schema": "uema_motor_controller_centres_v1",
           "source": {"runtime_state": os.path.relpath(rs_path, root).replace(os.sep, "/"),
                      "runtime_state_sha256": sha256(rs_path),
                      "model_safetensors_sha256": sha256(os.path.join(a.checkpoint, "model.safetensors")),
                      "model_step": 494025,
                      "reader": "lem/model/liveness.py:1132-1134 runtime_state()"}}
    for name, (key, n) in KEYS.items():
        entry = rs["liveness_repair"][key]
        cm = entry["centre_mean"].detach().cpu().float().numpy().astype(np.float32)
        if cm.shape != (n,) or not np.isfinite(cm).all():
            raise SystemExit("%s: centre_mean shape %s, expected (%d,), or not finite" % (key, cm.shape, n))
        out[name] = {"key": key, "centre_weight": float(entry["centre_weight"]),
                     "centre_mean": [float(x) for x in cm]}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("wrote %s: %s" % (a.out, ", ".join("%s %d numbers" % (k, len(out[k]["centre_mean"])) for k in KEYS)))


if __name__ == "__main__":
    main()
