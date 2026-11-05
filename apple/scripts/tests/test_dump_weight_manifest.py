# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Tests for apple/scripts/dump_weight_manifest.py.

Run from the repository root:
    python -m pytest apple/scripts/tests/test_dump_weight_manifest.py -p no:cacheprovider -q
or from Source/Python (the acceptance working directory of the implementation plan):
    python -m pytest ../../apple/scripts/tests/test_dump_weight_manifest.py -p no:cacheprovider -q
The small-file tests build a three-tensor safetensors file and a matching receipt in a temporary folder; the
released-file test runs only when weights/.../model.safetensors is present (fetch_weights.py) and checks the
acceptance numbers of work item WI-12: parameter elements 18,131,745 and every component equal to the receipt.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "dump_weight_manifest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("dump_weight_manifest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dump_weight_manifest"] = module
    spec.loader.exec_module(module)
    return module


dwm = _load_module()


def _small_files(tmp_path: Path, drop_inventory_key: str | None = None, wrong_total: bool = False,
                 drop_file_key: str | None = None, component_params: dict | None = None,
                 inventory_totals: dict | None = None, unique_parameter_tensors: int = 2):
    """Write a three-tensor safetensors file and its receipt; return (weights path, receipt path).

    The file holds 12 + 5 parameter elements (F32) and one BOOL buffer. `drop_inventory_key` removes a tensor from
    the receipt inventory; `drop_file_key` removes it from the file instead; `wrong_total` writes total_params 18;
    `component_params` replaces the receipt's block; `inventory_totals` replaces one module's saved_tensor_elements
    and saved_buffer_elements; `unique_parameter_tensors` sets the inventory's count (2 is correct).
    """
    from safetensors.numpy import save_file

    arrays = {
        "encoder_registry.encoders.vision.proj.weight": np.ones((4, 3), dtype=np.float32),   # 12 parameters
        "forward_model.bias": np.zeros((5,), dtype=np.float32),                               # 5 parameters
        "forward_model.flag": np.array(True),                                                 # 1 buffer (BOOL)
    }
    if drop_file_key:
        del arrays[drop_file_key]
    weights = tmp_path / "model.safetensors"
    save_file(arrays, str(weights))
    inventory = {
        "encoder_registry": {
            "saved_tensor_elements": 12, "saved_buffer_elements": 0,
            "tensors": {
                "encoders.vision.proj.weight": {"shape": [4, 3], "dtype": "torch.float32", "kind": "parameter",
                                                "elements": 12}}},
        "forward_model": {
            "saved_tensor_elements": 6, "saved_buffer_elements": 1,
            "tensors": {
                "bias": {"shape": [5], "dtype": "torch.float32", "kind": "parameter", "elements": 5},
                "flag": {"shape": [], "dtype": "torch.bool", "kind": "buffer", "elements": 1}}},
    }
    if drop_inventory_key:
        comp, name = drop_inventory_key.split(".", 1)
        del inventory[comp]["tensors"][name]
    if inventory_totals:
        for module, totals in inventory_totals.items():
            inventory[module].update(totals)
    receipt = {
        "generated_at": "2026-10-14T05:29:51+00:00",
        "total_params": 18 if wrong_total else 17,
        "component_params": {"enc_vision": 12, "forward_model": 5} if component_params is None else component_params,
        "checkpoint_inventory": {"step": 494025, "unique_parameter_tensors": unique_parameter_tensors,
                                 "components": inventory},
    }
    receipt_path = tmp_path / "model_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return weights, receipt_path


def test_component_of_splits_encoders_and_keeps_first_segment():
    assert dwm.component_of("encoder_registry.encoders.vision.patch_embed.weight") == "enc_vision"
    assert dwm.component_of("encoder_registry.shared.weight") == "enc_shared"
    assert dwm.component_of("forward_model.blocks.0.attn.in_proj_weight") == "forward_model"
    assert dwm.component_of("td_gates.vision.weight") == "td_gates"


def test_small_file_manifest_counts_and_kinds(tmp_path):
    weights, receipt = _small_files(tmp_path)
    manifest = dwm.build_manifest(weights, receipt)
    assert manifest["counts"] == {
        "tensors": 3, "parameter_tensors": 2, "buffer_tensors": 1, "parameter_elements": 17,
        "buffer_elements": 1, "inference_set_parameter_elements": 17, "dtypes": {"BOOL": 1, "F32": 2}}
    by_key = {r["key"]: r for r in manifest["tensors"]}
    flag = by_key["forward_model.flag"]
    assert (flag["shape"], flag["dtype"], flag["elements"], flag["bytes"], flag["kind"]) == ([], "BOOL", 1, 1, "buffer")
    proj = by_key["encoder_registry.encoders.vision.proj.weight"]
    assert (proj["component"], proj["bytes"], proj["inference_set"]) == ("enc_vision", 48, True)
    assert manifest["components"]["enc_vision"]["receipt_component_params"] == 12
    assert manifest["source"]["bytes"] == weights.stat().st_size
    assert manifest["source"]["header_bytes"] + sum(r["bytes"] for r in manifest["tensors"]) == weights.stat().st_size
    assert [r["key"] for r in manifest["tensors"]] == sorted(r["key"] for r in manifest["tensors"])


def test_small_file_mismatches_are_named(tmp_path):
    weights, receipt = _small_files(tmp_path, wrong_total=True)
    with pytest.raises(ValueError, match="parameter elements sum to 17, receipt total_params is 18"):
        dwm.build_manifest(weights, receipt)
    weights, receipt = _small_files(tmp_path, drop_inventory_key="forward_model.bias")
    with pytest.raises(ValueError, match="forward_model.bias is not in the receipt inventory"):
        dwm.build_manifest(weights, receipt)
    weights, receipt = _small_files(tmp_path, drop_file_key="forward_model.bias")
    with pytest.raises(ValueError, match="inventory tensor forward_model.bias is not in model.safetensors"):
        dwm.build_manifest(weights, receipt)


def test_component_and_inventory_total_mismatches_are_named(tmp_path):
    weights, receipt = _small_files(tmp_path, component_params={"enc_vision": 12, "forward_model": 4})
    with pytest.raises(ValueError, match="component forward_model: parameter elements 5 in the file, "
                                         "receipt component_params 4"):
        dwm.build_manifest(weights, receipt)
    weights, receipt = _small_files(tmp_path, component_params={"enc_vision": 12, "forward_model": 5, "wellness": 3})
    with pytest.raises(ValueError, match="component wellness: parameter elements 0 in the file, "
                                         "receipt component_params 3"):
        dwm.build_manifest(weights, receipt)
    weights, receipt = _small_files(
        tmp_path, inventory_totals={"forward_model": {"saved_tensor_elements": 7, "saved_buffer_elements": 2}})
    with pytest.raises(ValueError) as caught:
        dwm.build_manifest(weights, receipt)
    assert "module forward_model: 6 elements in the file, inventory saved_tensor_elements 7" in str(caught.value)
    assert "module forward_model: 1 buffer elements in the file, inventory saved_buffer_elements 2" in str(caught.value)
    weights, receipt = _small_files(tmp_path, unique_parameter_tensors=3)
    with pytest.raises(ValueError, match="2 parameter tensors in the file, inventory unique_parameter_tensors 3"):
        dwm.build_manifest(weights, receipt)


def test_byte_count_mismatch_is_named(tmp_path, monkeypatch):
    """A file that the header does not cover is refused by safetensors itself (one appended byte gives
    "incomplete metadata, file not fully covered"); the script's byte check guards its own bytes-per-element table,
    so the test shrinks the F32 entry from 4 to 2 and expects the named mismatch: 17 F32 elements x 2 bytes short."""
    weights, receipt = _small_files(tmp_path)
    size = weights.stat().st_size
    header = dwm.safetensors_header_bytes(weights)
    with open(weights, "ab") as stream:
        stream.write(b"\x00")
    with pytest.raises(ValueError, match="safetensors cannot read"):
        dwm.build_manifest(weights, receipt)
    with open(weights, "r+b") as stream:
        stream.truncate(size)
    monkeypatch.setitem(dwm.BYTES_PER_ELEMENT, "F32", 2)
    with pytest.raises(ValueError, match=r"header %d \+ tensor data %d bytes != file size %d bytes"
                                         % (header, 17 * 2 + 1, size)):
        dwm.build_manifest(weights, receipt)


def test_main_writes_then_checks(tmp_path, capsys):
    weights, receipt = _small_files(tmp_path)
    out = tmp_path / "out" / "weight_manifest.json"
    argv = ["--weights", str(weights), "--receipt", str(receipt), "--out", str(out)]
    assert dwm.main(argv + ["--check"]) == 1
    assert "no manifest at" in capsys.readouterr().out
    assert dwm.main(argv) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == dwm.SCHEMA
    assert dwm.main(argv + ["--check"]) == 0
    assert "matches the file" in capsys.readouterr().out
    written = json.loads(out.read_text(encoding="utf-8"))
    written["tensors"][0]["shape"] = [9, 9]
    out.write_text(json.dumps(written), encoding="utf-8")
    assert dwm.main(argv + ["--check"]) == 1
    assert "tensor records differ" in capsys.readouterr().out
    assert dwm.main(["--weights", str(tmp_path / "absent.safetensors"), "--receipt", str(receipt)]) == 1


def test_main_names_a_receipt_block_or_file_it_cannot_read(tmp_path, capsys):
    weights, receipt = _small_files(tmp_path)
    bare = tmp_path / "bare_receipt.json"
    bare.write_text(json.dumps({"total_params": 17}), encoding="utf-8")
    out = tmp_path / "out" / "weight_manifest.json"
    assert dwm.main(["--weights", str(weights), "--receipt", str(bare), "--out", str(out)]) == 1
    assert "lacks the block 'checkpoint_inventory'" in capsys.readouterr().out
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"\x08" + b"\x00" * 7 + b"garbage!")
    assert dwm.main(["--weights", str(broken), "--receipt", str(receipt), "--out", str(out)]) == 1
    assert "safetensors cannot read" in capsys.readouterr().out
    assert not out.exists()


@pytest.mark.skipif(not dwm.WEIGHTS.is_file(), reason="released model.safetensors not fetched")
def test_released_file_matches_receipt_and_written_manifest():
    manifest = dwm.build_manifest(dwm.WEIGHTS, dwm.RECEIPT)
    assert manifest["counts"]["parameter_elements"] == 18131745
    assert manifest["counts"]["inference_set_parameter_elements"] == 9926358
    receipt = json.loads(dwm.RECEIPT.read_text(encoding="utf-8"))
    for name, value in receipt["component_params"].items():
        assert manifest["components"][name]["parameter_elements"] == int(value), name
    assert dwm.MANIFEST.is_file(), "run apple/scripts/dump_weight_manifest.py first"
    written = json.loads(dwm.MANIFEST.read_text(encoding="utf-8"))
    assert dwm.compare_manifest(written, manifest) == []
