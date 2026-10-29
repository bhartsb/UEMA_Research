# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
load_and_resolve(path) -> dict

End-to-end: YAML in, validated resolved config out.

1. Load master_config.yaml
2. Merge with tier-3 defaults
3. Call resolve() to compute all derived fields
4. Optionally run coherence validation against receipts
5. Return fully-resolved config dict or exit on failure
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import yaml

from .resolver import resolve
from .validator import validate_coherence


def load_and_resolve(path: str,
                     pipeline_receipt_path: Optional[str] = None,
                     tokenizer_receipt_path: Optional[str] = None,
                     model_receipt_path: Optional[str] = None,
                     skip_validation: bool = False) -> dict:
    """
    Load master_config.yaml, resolve all derived fields, optionally validate.

    Args:
        path: Path to master_config.yaml.
        pipeline_receipt_path: Path to pipeline_receipt.json (or None).
        tokenizer_receipt_path: Path to tokenizer_receipt.json (or None).
        model_receipt_path: Path to model_receipt.json (or None).
        skip_validation: If True, skip coherence validation (useful during
                         initial setup before receipts exist).

    Returns:
        Fully-resolved config dict with all derived _ fields.
        Calls sys.exit(1) on validation failure.
    """
    # Load YAML
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"master_config.yaml not found: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raise ValueError(f"Empty or invalid YAML: {path}")

    # Resolve all derived fields (merge with defaults happens inside resolve)
    resolved = resolve(raw_config)

    # Optionally validate against receipts
    if not skip_validation:
        pipeline_receipt = _load_json(pipeline_receipt_path)
        tokenizer_receipt = _load_json(tokenizer_receipt_path)
        model_receipt = _load_json(model_receipt_path)

        # Only validate if at least one receipt is available
        if any(r is not None for r in [pipeline_receipt, tokenizer_receipt, model_receipt]):
            validate_coherence(
                resolved,
                pipeline_receipt=pipeline_receipt,
                tokenizer_receipt=tokenizer_receipt,
                model_receipt=model_receipt,
            )

    return resolved


def _load_json(path: Optional[str]) -> Optional[dict]:
    """Load a JSON receipt file, returning None if path is None or file missing."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
