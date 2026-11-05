# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Write the weight manifest of the released model.safetensors (model step 494,025) for the Swift loader.

    python -I apple/scripts/dump_weight_manifest.py            (from the repository root)
    python -I apple/scripts/dump_weight_manifest.py --check    (read the manifest already written and compare)

Inputs: weights/b1_continue_B1K_skip13_492800/run/checkpoints/ckpt_step_494025/model.safetensors (73,119,988 bytes,
fetched by fetch_weights.py) and model_receipt.json beside it (its `component_params` block, written by
Source/Python/mimo_bridge/lem_model.py at the line `self.receipt['component_params'] = ...`, and its
`checkpoint_inventory.components` block, which names every saved tensor with its kind: parameter or buffer).
Operation: the file is opened with safetensors' safe_open(framework="np"), the same reader the trainer uses on this
file (lem_model.py, `with safe_open(weights_path,framework='pt',device='cpu') as sf:`), and for every key its shape,
its safetensors dtype name ("F32", "F64", "BOOL", "I64") and its element count (the product of the shape; 1 for a
scalar) are read through get_slice without materialising the array. Each key is assigned to the receipt's component by
its first dotted segment, except that `encoder_registry.encoders.<sense>.` keys go to `enc_<sense>` and any other
`encoder_registry.` key to `enc_shared`, the same split lem_model.py applies when it writes component_params. The
tensor's kind comes from the receipt's checkpoint_inventory entry of the same name. Six checks run before anything is
written: every file key is in the receipt inventory and every inventory tensor is in the file; the parameter elements
sum to the receipt's total_params (18,131,745); each component's parameter elements equal the receipt's
component_params value; each inventory module's element total and buffer element total equal the inventory's
saved_tensor_elements and saved_buffer_elements; the count of parameter tensors equals the inventory's
unique_parameter_tensors (676); the file's byte count equals the sum of the tensors' bytes plus the header (safetensors
itself refuses a file its header does not cover, so this last check guards the BYTES_PER_ELEMENT table below). A failed
check prints every mismatch by name and exits 1; a receipt missing a block or an unreadable safetensors file also
exits 1 with the block or file named.
Output: apple/UEMAKit/Sources/UEMAModel/Resources/weight_manifest.json, sorted by key, with one record per tensor
(key, shape, dtype, elements, bytes, kind, component, inference_set) and a components block (parameter elements,
receipt value, tensor count, inference_set). `inference_set` is true for the components the specification's REQ-B-11
names as the acting path the Swift model loads (the six encoders, forward_model, prediction_heads, motor_controller,
wellness; 9,926,358 parameters) and false for the rest, which the Swift loader lists and skips. Exit code 0 on success.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO / "weights" / "b1_continue_B1K_skip13_492800" / "run" / "checkpoints" / "ckpt_step_494025"
WEIGHTS = CHECKPOINT / "model.safetensors"
RECEIPT = CHECKPOINT / "model_receipt.json"
MANIFEST = REPO / "apple" / "UEMAKit" / "Sources" / "UEMAModel" / "Resources" / "weight_manifest.json"
SCHEMA = "uema_weight_manifest_v1"

# Components the Swift model loads (specification REQ-B-11: six encoders, forward model, prediction heads, motor
# controller, wellness). Every other component is listed in the manifest and skipped by the loader.
INFERENCE_COMPONENTS = frozenset({
    "enc_vision", "enc_tactile", "enc_audio", "enc_proprioception", "enc_motor", "enc_vestibular",
    "forward_model", "prediction_heads", "motor_controller", "wellness",
})

# Bytes per element for the safetensors dtype names (safetensors format: F64/F32/F16/BF16, I64/I32/I16/I8, U8, BOOL).
BYTES_PER_ELEMENT = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}


def component_of(key: str) -> str:
    """Return the receipt component a safetensors key belongs to.

    Input: a dotted key such as `encoder_registry.encoders.vision.patch_embed.weight`.
    Operation: the first segment names the component; `encoder_registry.encoders.<sense>.` keys map to
    `enc_<sense>` and other `encoder_registry.` keys to `enc_shared` (lem_model.py's component_params split).
    Output: the component name, for example `enc_vision` or `forward_model`.
    """
    parts = key.split(".")
    if parts[0] == "encoder_registry":
        if len(parts) >= 4 and parts[1] == "encoders":
            return "enc_" + parts[2]
        return "enc_shared"
    return parts[0]


def read_tensor_records(weights_path: Path) -> List[Dict[str, Any]]:
    """Read every tensor's key, shape, dtype and element count from a safetensors file.

    Input: the path of a safetensors file.
    Operation: safe_open(framework="np"); for each key get_slice gives the shape and dtype name without loading
    the data; elements = product of the shape (math.prod of an empty shape is 1, the scalar case);
    bytes = elements x BYTES_PER_ELEMENT[dtype].
    Output: a list of dicts with key, shape (list of int), dtype (safetensors name), elements, bytes, component,
    sorted by key; `kind` and `inference_set` are filled in later from the receipt. A file safetensors cannot
    parse raises ValueError naming the path.
    """
    from safetensors import SafetensorError, safe_open

    records: List[Dict[str, Any]] = []
    try:
        handle_context = safe_open(str(weights_path), framework="np")
    except SafetensorError as exc:
        raise ValueError("safetensors cannot read %s: %s" % (weights_path, exc)) from exc
    with handle_context as handle:
        for key in sorted(handle.keys()):
            tensor_slice = handle.get_slice(key)
            shape = [int(n) for n in tensor_slice.get_shape()]
            dtype = str(tensor_slice.get_dtype())
            if dtype not in BYTES_PER_ELEMENT:
                raise ValueError("unknown safetensors dtype %r on key %s" % (dtype, key))
            elements = int(math.prod(shape))
            records.append({
                "key": key,
                "shape": shape,
                "dtype": dtype,
                "elements": elements,
                "bytes": elements * BYTES_PER_ELEMENT[dtype],
                "component": component_of(key),
            })
    return records


def safetensors_header_bytes(weights_path: Path) -> int:
    """Return 8 plus the header length of a safetensors file (the bytes before the tensor data).

    Input: the file path. Operation: the first 8 bytes hold the header length as a little-endian unsigned 64-bit
    integer (safetensors format). Output: 8 + that length, in bytes.
    """
    with open(weights_path, "rb") as stream:
        (header_len,) = struct.unpack("<Q", stream.read(8))
    return 8 + int(header_len)


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file read in 4 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_kinds(receipt: Dict[str, Any]) -> Dict[str, str]:
    """Map every saved tensor's safetensors key to its kind from the receipt's checkpoint inventory.

    Input: the parsed model_receipt.json. Operation: for each component in checkpoint_inventory.components, each
    entry of its `tensors` block is keyed `<component>.<name>` and its `kind` ("parameter" or "buffer") is kept.
    Output: a dict from safetensors key to kind.
    """
    kinds: Dict[str, str] = {}
    components = receipt["checkpoint_inventory"]["components"]
    for component, block in components.items():
        for name, entry in block["tensors"].items():
            kinds[component + "." + name] = str(entry["kind"])
    return kinds


def check_against_receipt(records: List[Dict[str, Any]], receipt: Dict[str, Any]) -> List[str]:
    """Compare the tensor records with the receipt and return the list of mismatches (empty when all agree).

    Input: the records from read_tensor_records with `kind` filled in, and the parsed receipt.
    Operation: (1) key sets of the file and the inventory are equal; (2) parameter elements sum to
    receipt total_params; (3) per-component parameter elements equal receipt component_params, and every
    component_params entry has a component in the file; (4) per inventory module (the key's first segment,
    `encoder_registry` undivided) the elements of all its tensors equal saved_tensor_elements and the elements of
    its buffers equal saved_buffer_elements; (5) the count of parameter tensors equals the inventory's
    unique_parameter_tensors.
    Output: a list of mismatch sentences, each naming the key or component and the two numbers compared.
    """
    problems: List[str] = []
    inventory = receipt["checkpoint_inventory"]
    kinds = inventory_kinds(receipt)
    file_keys = {r["key"] for r in records}
    for key in sorted(kinds.keys() - file_keys):
        problems.append("receipt inventory tensor %s is not in model.safetensors" % key)
    for key in sorted(file_keys - kinds.keys()):
        problems.append("model.safetensors key %s is not in the receipt inventory" % key)
    param_total = sum(r["elements"] for r in records if r["kind"] == "parameter")
    if param_total != int(receipt["total_params"]):
        problems.append("parameter elements sum to %d, receipt total_params is %d"
                        % (param_total, int(receipt["total_params"])))
    per_component: Dict[str, int] = {}
    for r in records:
        if r["kind"] == "parameter":
            per_component[r["component"]] = per_component.get(r["component"], 0) + r["elements"]
    expected = {k: int(v) for k, v in receipt["component_params"].items()}
    for name in sorted(set(expected) | set(per_component)):
        got, want = per_component.get(name, 0), expected.get(name, 0)
        if got != want:
            problems.append("component %s: parameter elements %d in the file, receipt component_params %d"
                            % (name, got, want))
    per_module_all: Dict[str, int] = {}
    per_module_buffers: Dict[str, int] = {}
    for r in records:
        module = r["key"].split(".")[0]
        per_module_all[module] = per_module_all.get(module, 0) + r["elements"]
        if r["kind"] == "buffer":
            per_module_buffers[module] = per_module_buffers.get(module, 0) + r["elements"]
    for module, block in sorted(inventory["components"].items()):
        got_all, want_all = per_module_all.get(module, 0), int(block["saved_tensor_elements"])
        if got_all != want_all:
            problems.append("module %s: %d elements in the file, inventory saved_tensor_elements %d"
                            % (module, got_all, want_all))
        got_buf, want_buf = per_module_buffers.get(module, 0), int(block["saved_buffer_elements"])
        if got_buf != want_buf:
            problems.append("module %s: %d buffer elements in the file, inventory saved_buffer_elements %d"
                            % (module, got_buf, want_buf))
    param_tensors = sum(1 for r in records if r["kind"] == "parameter")
    if param_tensors != int(inventory["unique_parameter_tensors"]):
        problems.append("%d parameter tensors in the file, inventory unique_parameter_tensors %d"
                        % (param_tensors, int(inventory["unique_parameter_tensors"])))
    return problems


def build_manifest(weights_path: Path, receipt_path: Path) -> Dict[str, Any]:
    """Build the manifest dict from the weights file and the receipt, raising ValueError on any mismatch.

    Input: the two paths. Operation: read_tensor_records, fill `kind` from the receipt inventory (a key the
    inventory lacks keeps kind "unknown" and fails the check), set inference_set from INFERENCE_COMPONENTS, run
    check_against_receipt and the byte check (header + tensor bytes == file size), then assemble the dict.
    Output: the manifest dict (schema, generated_at, source, counts, components, tensors).
    """
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    records = read_tensor_records(weights_path)
    kinds = inventory_kinds(receipt)
    for r in records:
        r["kind"] = kinds.get(r["key"], "unknown")
        r["inference_set"] = r["component"] in INFERENCE_COMPONENTS
    problems = check_against_receipt(records, receipt)
    file_bytes = weights_path.stat().st_size
    data_bytes = sum(r["bytes"] for r in records)
    header_bytes = safetensors_header_bytes(weights_path)
    if header_bytes + data_bytes != file_bytes:
        problems.append("header %d + tensor data %d bytes != file size %d bytes"
                        % (header_bytes, data_bytes, file_bytes))
    if problems:
        raise ValueError("\n".join(problems))

    components: Dict[str, Dict[str, Any]] = {}
    for r in records:
        block = components.setdefault(r["component"], {
            "parameter_elements": 0, "buffer_elements": 0, "tensors": 0,
            "receipt_component_params": int(receipt["component_params"].get(r["component"], 0)),
            "inference_set": r["inference_set"],
        })
        block["tensors"] += 1
        block["parameter_elements" if r["kind"] == "parameter" else "buffer_elements"] += r["elements"]
    inference_params = sum(b["parameter_elements"] for b in components.values() if b["inference_set"])
    try:
        source_path = weights_path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        source_path = str(weights_path)
    return {
        "schema": SCHEMA,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "path": source_path,
            "bytes": file_bytes,
            "sha256": sha256_of(weights_path),
            "header_bytes": header_bytes,
            "model_step": int(receipt["checkpoint_inventory"]["step"]),
            "receipt_generated_at": receipt["generated_at"],
            "receipt_total_params": int(receipt["total_params"]),
        },
        "counts": {
            "tensors": len(records),
            "parameter_tensors": sum(1 for r in records if r["kind"] == "parameter"),
            "buffer_tensors": sum(1 for r in records if r["kind"] == "buffer"),
            "parameter_elements": sum(r["elements"] for r in records if r["kind"] == "parameter"),
            "buffer_elements": sum(r["elements"] for r in records if r["kind"] == "buffer"),
            "inference_set_parameter_elements": inference_params,
            "dtypes": {d: sum(1 for r in records if r["dtype"] == d) for d in sorted({r["dtype"] for r in records})},
        },
        "components": dict(sorted(components.items())),
        "tensors": records,
    }


def compare_manifest(written: Dict[str, Any], fresh: Dict[str, Any]) -> List[str]:
    """Return the differences between a manifest on disk and one freshly built, ignoring generated_at."""
    problems: List[str] = []
    for field in ("schema", "source", "counts", "components"):
        if written.get(field) != fresh.get(field):
            problems.append("field %s differs" % field)
    if written.get("tensors") != fresh.get("tensors"):
        problems.append("tensor records differ")
    return problems


def main(argv: List[str] | None = None) -> int:
    """Command entry: build (or with --check compare) the manifest; print the totals; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--weights", type=Path, default=WEIGHTS, help="model.safetensors path")
    parser.add_argument("--receipt", type=Path, default=RECEIPT, help="model_receipt.json path")
    parser.add_argument("--out", type=Path, default=MANIFEST, help="manifest path to write")
    parser.add_argument("--check", action="store_true",
                        help="do not write; compare the manifest at --out with a fresh build")
    args = parser.parse_args(argv)
    for path in (args.weights, args.receipt):
        if not path.is_file():
            print("missing input: %s (run fetch_weights.py --accept-license first)" % path)
            return 1
    try:
        manifest = build_manifest(args.weights, args.receipt)
    except ValueError as exc:
        print("manifest check failed:\n%s" % exc)
        return 1
    except KeyError as exc:
        print("receipt %s lacks the block %s" % (args.receipt, exc))
        return 1
    if args.check:
        if not args.out.is_file():
            print("no manifest at %s" % args.out)
            return 1
        written = json.loads(args.out.read_text(encoding="utf-8"))
        problems = compare_manifest(written, manifest)
        if problems:
            print("manifest at %s differs from the file:\n%s" % (args.out, "\n".join(problems)))
            return 1
        print("manifest at %s matches the file" % args.out)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        print("wrote %s" % args.out)
    counts = manifest["counts"]
    print("tensors %d (parameters %d, buffers %d); parameter elements %d = receipt total_params %d; "
          "inference-set parameter elements %d"
          % (counts["tensors"], counts["parameter_tensors"], counts["buffer_tensors"],
             counts["parameter_elements"], manifest["source"]["receipt_total_params"],
             counts["inference_set_parameter_elements"]))
    for name, block in manifest["components"].items():
        print("  %-22s parameter elements %9d = receipt %9d; tensors %3d; %s"
              % (name, block["parameter_elements"], block["receipt_component_params"], block["tensors"],
                 "loaded by the Swift model" if block["inference_set"] else "listed and skipped"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
