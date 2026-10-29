# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Coherence validator — 12 checks at initialization.

Reports ALL failures before exit. Never short-circuits on first failure.
If any check fails, prints a structured error report and calls sys.exit(1).

validate_coherence(config, pipeline_receipt, tokenizer_receipt, model_receipt) -> None
"""

from __future__ import annotations

import sys
from typing import Any, Optional


class CoherenceFailure:
    """Single coherence check failure."""

    __slots__ = ("check_id", "check_name", "expected", "actual", "cause")

    def __init__(self, check_id: str, check_name: str,
                 expected: Any, actual: Any, cause: str):
        self.check_id = check_id
        self.check_name = check_name
        self.expected = expected
        self.actual = actual
        self.cause = cause

    def __str__(self) -> str:
        return (
            f"  [{self.check_id}] {self.check_name}\n"
            f"    Expected: {self.expected}\n"
            f"    Actual:   {self.actual}\n"
            f"    Cause:    {self.cause}"
        )


def _check_c1(config: dict, pipeline_receipt: dict,
               tokenizer_receipt: dict) -> list[CoherenceFailure]:
    """C1: Modality set match — master enabled == pipeline == tokenizer."""
    failures = []
    master_set = set(config["_active_modalities"])

    pipe_set = set(pipeline_receipt.get("enabled_modalities", []))
    if master_set != pipe_set:
        failures.append(CoherenceFailure(
            "C1", "Modality set match (pipeline)",
            sorted(master_set), sorted(pipe_set),
            "master_config enabled modalities differ from pipeline_receipt"
        ))

    tok_set = set(tokenizer_receipt.get("enabled_modalities", []))
    if master_set != tok_set:
        failures.append(CoherenceFailure(
            "C1", "Modality set match (tokenizer)",
            sorted(master_set), sorted(tok_set),
            "master_config enabled modalities differ from tokenizer_receipt"
        ))

    return failures


def _check_c2(config: dict, pipeline_receipt: dict,
               tokenizer_receipt: dict,
               model_receipt: Optional[dict]) -> list[CoherenceFailure]:
    """C2: Version match — master.version == all receipt versions."""
    failures = []
    master_ver = config.get("version", "unknown")

    for name, receipt in [("pipeline", pipeline_receipt),
                          ("tokenizer", tokenizer_receipt),
                          ("model", model_receipt)]:
        if receipt is None:
            continue
        recv_ver = receipt.get("version", "missing")
        if master_ver != recv_ver:
            failures.append(CoherenceFailure(
                "C2", f"Version match ({name})",
                master_ver, recv_ver,
                f"master_config version != {name}_receipt version"
            ))

    return failures


def _check_c3(config: dict, pipeline_receipt: dict) -> list[CoherenceFailure]:
    """C3: Body params match — master.body.* == pipeline_receipt.body.*."""
    failures = []
    master_body = config.get("body", {})
    pipe_body = pipeline_receipt.get("body", {})

    for key in master_body:
        if key in pipe_body and master_body[key] != pipe_body[key]:
            failures.append(CoherenceFailure(
                "C3", f"Body param match ({key})",
                master_body[key], pipe_body[key],
                f"master_config body.{key} != pipeline_receipt body.{key}"
            ))

    return failures


def _check_c4(config: dict, tokenizer_receipt: dict) -> list[CoherenceFailure]:
    """C4: Tensor shapes match — resolved shapes == tokenizer output shapes."""
    failures = []
    tok_tensors = tokenizer_receipt.get("output_tensors", {})

    for mod_name in config["_active_modalities"]:
        if mod_name not in tok_tensors:
            continue
        tok_shape = tok_tensors[mod_name].get("shape")
        if tok_shape is None:
            continue

        # Compare against resolved encoder expectations
        enc = config["modalities"][mod_name]["encoder"]
        expected_shape = None

        if mod_name == "vision":
            n_patches = enc.get("_n_patches")
            patch_dim = enc.get("_patch_dim")
            if n_patches and patch_dim:
                expected_shape = [n_patches, patch_dim]
        elif mod_name == "proprioception":
            dim = enc.get("_input_dim")
            if dim:
                expected_shape = [dim]
        elif mod_name == "tactile":
            pipe = config["modalities"][mod_name]["pipeline"]
            expected_shape = pipe.get("shape")
        elif mod_name == "motor":
            pipe = config["modalities"][mod_name]["pipeline"]
            n_h = pipe.get("n_handles", config["body"]["n_handles"])
            fpf = pipe.get("features_per_handle", 8)
            expected_shape = [n_h, fpf]
        elif mod_name == "intermind":
            pipe = config["modalities"][mod_name]["pipeline"]
            n_a = pipe.get("n_agents", config["body"]["n_agents"])
            embed = enc.get("_embed_input_dim")
            if embed:
                expected_shape = [n_a, embed]
        elif mod_name == "voice":
            dim = enc.get("_input_dim")
            if dim:
                expected_shape = [dim]

        if expected_shape is not None and list(tok_shape) != list(expected_shape):
            failures.append(CoherenceFailure(
                "C4", f"Tensor shape match ({mod_name})",
                expected_shape, tok_shape,
                f"Resolved shape for {mod_name} != tokenizer_receipt shape"
            ))

    return failures


def _check_c5(config: dict, pipeline_receipt: dict) -> list[CoherenceFailure]:
    """C5: SHM layout match — resolved offsets == pipeline_receipt buffer_layout."""
    failures = []
    resolved_shm = config.get("_shm_layout", {})
    pipe_layout = pipeline_receipt.get("buffer_layout", {})

    if not pipe_layout:
        return failures

    pipe_offsets = pipe_layout.get("offsets", {})
    pipe_sizes = pipe_layout.get("sizes", {})

    for mod_name, resolved in resolved_shm.items():
        if mod_name in pipe_offsets:
            if resolved["offset"] != pipe_offsets[mod_name]:
                failures.append(CoherenceFailure(
                    "C5", f"SHM offset match ({mod_name})",
                    resolved["offset"], pipe_offsets[mod_name],
                    f"Resolved SHM offset for {mod_name} != pipeline_receipt"
                ))
        if mod_name in pipe_sizes:
            if resolved["size"] != pipe_sizes[mod_name]:
                failures.append(CoherenceFailure(
                    "C5", f"SHM size match ({mod_name})",
                    resolved["size"], pipe_sizes[mod_name],
                    f"Resolved SHM size for {mod_name} != pipeline_receipt"
                ))

    return failures


def _check_c6(config: dict, pipeline_receipt: dict,
               tokenizer_receipt: dict) -> list[CoherenceFailure]:
    """C6: Rate consistency — master rate_hz consistent across pipeline, tokenizer."""
    failures = []
    for mod_name in config["_active_modalities"]:
        master_rate = config["modalities"][mod_name].get("rate_hz")
        if master_rate is None:
            continue

        pipe_rates = pipeline_receipt.get("rates", {})
        if mod_name in pipe_rates and pipe_rates[mod_name] != master_rate:
            failures.append(CoherenceFailure(
                "C6", f"Rate consistency ({mod_name}, pipeline)",
                master_rate, pipe_rates[mod_name],
                f"master rate_hz for {mod_name} != pipeline_receipt rate"
            ))

        tok_rates = tokenizer_receipt.get("rates", {})
        if mod_name in tok_rates and tok_rates[mod_name] != master_rate:
            failures.append(CoherenceFailure(
                "C6", f"Rate consistency ({mod_name}, tokenizer)",
                master_rate, tok_rates[mod_name],
                f"master rate_hz for {mod_name} != tokenizer_receipt rate"
            ))

    return failures


def _check_c7(config: dict, pipeline_receipt: dict) -> list[CoherenceFailure]:
    """C7: Data source coherent — if real, pipeline must report active."""
    failures = []
    pipe_active = set(pipeline_receipt.get("enabled_modalities", []))

    for mod_name in config["_active_modalities"]:
        src = config["modalities"][mod_name].get("data_source")
        if src == "real" and mod_name not in pipe_active:
            failures.append(CoherenceFailure(
                "C7", f"Data source coherent ({mod_name})",
                f"{mod_name} in pipeline active set", f"{mod_name} not found",
                f"data_source='real' but pipeline_receipt does not list {mod_name}"
            ))

    return failures


def _check_c8(config: dict, model_receipt: dict) -> list[CoherenceFailure]:
    """C8: Encoder type match — master encoder.type == model_receipt encoders_built."""
    failures = []
    encoders_built = model_receipt.get("encoders_built", {})

    for mod_name in config["_active_modalities"]:
        master_type = config["modalities"][mod_name]["encoder"].get("type")
        if mod_name in encoders_built:
            built_type = encoders_built[mod_name].get("type")
            if master_type != built_type:
                failures.append(CoherenceFailure(
                    "C8", f"Encoder type match ({mod_name})",
                    master_type, built_type,
                    f"master encoder type for {mod_name} != model_receipt"
                ))

    return failures


def _check_c9(config: dict, model_receipt: dict) -> list[CoherenceFailure]:
    """C9: d_model propagation — all encoder d_output == master.model.d_model."""
    failures = []
    d_model = config["model"]["d_model"]
    encoders_built = model_receipt.get("encoders_built", {})

    for mod_name, enc_info in encoders_built.items():
        d_out = enc_info.get("d_output")
        if d_out is not None and d_out != d_model:
            failures.append(CoherenceFailure(
                "C9", f"d_model propagation ({mod_name})",
                d_model, d_out,
                f"encoder d_output for {mod_name} != master d_model"
            ))

    return failures


def _check_c10(config: dict, model_receipt: dict) -> list[CoherenceFailure]:
    """C10: Prediction heads correct — resolved list == model_receipt list."""
    failures = []
    expected = sorted(config["_prediction_head_modalities"])
    actual = sorted(model_receipt.get("prediction_head_modalities", []))

    if expected != actual:
        failures.append(CoherenceFailure(
            "C10", "Prediction heads correct",
            expected, actual,
            "Resolved _prediction_head_modalities != model_receipt list"
        ))

    return failures


def _check_c11(config: dict, model_receipt: dict) -> list[CoherenceFailure]:
    """C11: Motor controller input dim — resolved == model_receipt."""
    failures = []
    expected = config["model"]["motor_controller"]["_input_dim"]
    actual = model_receipt.get("motor_controller", {}).get("_input_dim")

    if actual is not None and expected != actual:
        failures.append(CoherenceFailure(
            "C11", "Motor controller input dim",
            expected, actual,
            "Resolved motor_controller._input_dim != model_receipt"
        ))

    return failures


def _check_c12(config: dict, model_receipt: dict) -> list[CoherenceFailure]:
    """C12: Seq length match — n_active * context == model_receipt._seq_length."""
    failures = []
    expected = config["model"]["forward_model"]["_seq_length"]
    actual = model_receipt.get("forward_model", {}).get("_seq_length")

    if actual is not None and expected != actual:
        failures.append(CoherenceFailure(
            "C12", "Seq length match",
            expected, actual,
            "Resolved _seq_length != model_receipt _seq_length"
        ))

    return failures


def _check_c13(config: dict) -> list[CoherenceFailure]:
    """C13: Action buffer size — buffer_size must match field layout."""
    failures = []
    ab = config.get("action_buffer", {})
    if not ab.get("enabled", False):
        return failures

    buf_size = ab.get("buffer_size", 0)
    if buf_size < 1176:
        failures.append(CoherenceFailure(
            "C13", "Action buffer minimum size",
            ">= 1176", buf_size,
            "action_buffer.buffer_size too small for header(20)+motor(828)+gaze(8)+voice(320)"
        ))

    # Verify size matches field layout: header(20) + motor + gaze(8) + voice(320)
    mc = config.get("model", {}).get("motor_controller", {})
    body = config.get("body", {})
    output_per_handle = mc.get("output_per_handle", 9)
    n_handles = body.get("n_handles", 23)
    expected = 20 + (output_per_handle * n_handles * 4) + 8 + 320
    if buf_size != expected:
        failures.append(CoherenceFailure(
            "C13", "Action buffer size matches layout",
            expected, buf_size,
            f"header(20) + motor({output_per_handle}*{n_handles}*4) + gaze(8) + voice(320) != buffer_size"
        ))

    return failures


def validate_coherence(config: dict,
                       pipeline_receipt: Optional[dict] = None,
                       tokenizer_receipt: Optional[dict] = None,
                       model_receipt: Optional[dict] = None) -> None:
    """
    Run all 13 coherence checks. Report ALL failures, then sys.exit(1) if any.

    Args:
        config: Fully-resolved config dict (output of resolve()).
        pipeline_receipt: Parsed pipeline_receipt.json (or None to skip pipeline checks).
        tokenizer_receipt: Parsed tokenizer_receipt.json (or None to skip tokenizer checks).
        model_receipt: Parsed model_receipt.json (or None to skip model checks).
    """
    if pipeline_receipt is None:
        pipeline_receipt = {}
    if tokenizer_receipt is None:
        tokenizer_receipt = {}
    if model_receipt is None:
        model_receipt = {}

    all_failures: list[CoherenceFailure] = []

    # C1-C7: Config vs pipeline/tokenizer
    all_failures.extend(_check_c1(config, pipeline_receipt, tokenizer_receipt))
    all_failures.extend(_check_c2(config, pipeline_receipt, tokenizer_receipt, model_receipt))
    all_failures.extend(_check_c3(config, pipeline_receipt))
    all_failures.extend(_check_c4(config, tokenizer_receipt))
    all_failures.extend(_check_c5(config, pipeline_receipt))
    all_failures.extend(_check_c6(config, pipeline_receipt, tokenizer_receipt))
    all_failures.extend(_check_c7(config, pipeline_receipt))

    # C8-C12: Config vs model
    all_failures.extend(_check_c8(config, model_receipt))
    all_failures.extend(_check_c9(config, model_receipt))
    all_failures.extend(_check_c10(config, model_receipt))
    all_failures.extend(_check_c11(config, model_receipt))
    all_failures.extend(_check_c12(config, model_receipt))

    # C13: Config-only (no receipt needed)
    all_failures.extend(_check_c13(config))

    if all_failures:
        print("\n" + "=" * 60)
        print("COHERENCE VALIDATION FAILED")
        print("=" * 60)
        for f in all_failures:
            print(f)
            print()
        print(f"Total failures: {len(all_failures)}")
        print("=" * 60)
        sys.exit(1)
