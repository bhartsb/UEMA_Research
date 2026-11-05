# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Export the running centres of the revived units the prediction path reads (work item WI-35).

WHY. The released run has --liveness-repair on (weights/b1_continue_B1K_skip13_492800/run/args.json). Its
RevivedRows (Source/Python/lem/model/liveness.py:290-361) replace a revived unit's pre-activation by
lin(norm(x - centre_mean))[j]; the mask, norm and lin are in model.safetensors, but centre_mean is registered with
persistent=False (liveness.py:309) and rides runtime_state.pt, which Swift cannot read. The prediction path reads
eight such layers: prediction_heads.heads.{vision,tactile,vestibular}.mlp.0 (192 of 192 units revived each) and
heads_all_slot_path.paths.{vision,tactile,audio,proprioception,vestibular}.net.0 (139, 17, 87, 71 and 172 of 192
revived; the masks in model.safetensors).

INPUT. runtime_state.pt of checkpoint 494,025 (489,337,136 bytes), a zip of torch's pickle format.
OPERATION. data.pkl is read by a restricted unpickler that resolves only torch._utils._rebuild_tensor_v2,
_rebuild_parameter, the torch storage type names, collections.OrderedDict and builtins; every tensor becomes a numpy
array read from the zip's data/<key> record (a contiguous tensor; anything else is refused). The eight layers'
centre_mean [192] float32 and centre_weight are taken from the "liveness_repair" entry (the dict that
liveness.py runtime_state() returns).
OUTPUT. apple/UEMAKit/Sources/UEMAModel/Resources/prediction_head_centres.json, schema
uema_prediction_head_centres_v1: the source file, its sha256, the model.safetensors sha256, and per layer key the
centre_weight and the 192 centre_mean numbers.

Run from D:/UEMA_Research (numpy only; no torch):
    python -I apple/UEMAKit/Tests/UEMAModelTests/Fixtures/prediction_heads/export_prediction_head_centres.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import pickle
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", ".."))
CKPT = os.path.join(REPO, "weights", "b1_continue_B1K_skip13_492800", "run", "checkpoints", "ckpt_step_494025")
OUT = os.path.join(REPO, "apple", "UEMAKit", "Sources", "UEMAModel", "Resources", "prediction_head_centres.json")
SCHEMA = "uema_prediction_head_centres_v1"
LAYERS = (
    "prediction_heads__heads__vision__mlp__0",
    "prediction_heads__heads__tactile__mlp__0",
    "prediction_heads__heads__vestibular__mlp__0",
    "heads_all_slot_path__paths__vision__net__0",
    "heads_all_slot_path__paths__tactile__net__0",
    "heads_all_slot_path__paths__audio__net__0",
    "heads_all_slot_path__paths__proprioception__net__0",
    "heads_all_slot_path__paths__vestibular__net__0",
)
STORAGE = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64,
           "BoolStorage": np.bool_, "IntStorage": np.int32, "HalfStorage": np.float16, "ByteStorage": np.uint8,
           "BFloat16Storage": np.uint16, "ShortStorage": np.int16, "CharStorage": np.int8}


class _Storage:
    def __init__(self, kind: str) -> None:
        self.kind = kind


class _Inert:
    """Stands for any class the exporter does not need (dtypes, devices, numpy scalars, user classes): it takes
    whatever arguments and state the pickle hands it and keeps nothing."""

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __setstate__(self, state) -> None:
        pass


def read_torch_zip(path: str):
    """INPUT: a torch zip checkpoint. OPERATION: the restricted unpickling described in the module text.
    OUTPUT: the unpickled object with numpy arrays in place of tensors."""
    zf = zipfile.ZipFile(path)
    root = zf.namelist()[0].split("/")[0]

    def rebuild(storage, offset, size, stride, *_rest):
        size = tuple(size)
        count = int(np.prod(size)) if size else 1
        expected, acc = [], 1
        for extent in reversed(size):
            expected.append(acc)
            acc *= extent
        if size and tuple(reversed(expected)) != tuple(stride):
            raise ValueError("a non-contiguous tensor %s %s is not read by this exporter" % (size, tuple(stride)))
        return storage[offset: offset + count].reshape(size)

    class Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return rebuild
            if module == "torch._utils" and name == "_rebuild_parameter":
                return lambda data, requires_grad, hooks: data
            if module == "torch" and name in STORAGE:
                return _Storage(name)
            if module == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            if module == "builtins" and name in ("set", "frozenset", "slice", "complex"):
                return super().find_class(module, name)
            # anything else (dtypes, devices, user classes) becomes an inert placeholder
            return _Inert

        def persistent_load(self, pid):
            _tag, storage, key, _location, _numel = pid
            return np.frombuffer(zf.read("%s/data/%s" % (root, key)), dtype=STORAGE[storage.kind])

    with zf.open("%s/data.pkl" % root) as handle:
        return Restricted(handle).load()


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    runtime = os.path.join(CKPT, "runtime_state.pt")
    state = read_torch_zip(runtime)
    repair = state.get("liveness_repair") if isinstance(state, dict) else None
    if not isinstance(repair, dict):
        print("FAIL: runtime_state.pt holds no liveness_repair entry")
        return 1
    layers = {}
    for key in LAYERS:
        entry = repair.get(key)
        if not isinstance(entry, dict) or "centre_mean" not in entry:
            print("FAIL: no centre for %s" % key)
            return 1
        centre = np.asarray(entry["centre_mean"], dtype=np.float32).reshape(-1)
        if centre.shape != (192,) or not np.isfinite(centre).all():
            print("FAIL: centre of %s has shape %s or a non-finite number" % (key, centre.shape))
            return 1
        layers[key] = {"centre_weight": float(entry.get("centre_weight", 0.0)),
                       "centre_mean": [float(v) for v in centre]}
    out = {"schema": SCHEMA,
           "source": {"runtime_state": os.path.relpath(runtime, REPO).replace(os.sep, "/"),
                      "runtime_state_sha256": sha256(runtime),
                      "model_safetensors_sha256": sha256(os.path.join(CKPT, "model.safetensors")),
                      "model_step": 494025,
                      "reader": "lem/model/liveness.py:290-361 RevivedRows (centre_mean persistent=False)"},
           "layers": layers}
    with open(OUT, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(out, handle, indent=1)
        handle.write("\n")
    print("PASS: %d layers written to %s" % (len(layers), OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
