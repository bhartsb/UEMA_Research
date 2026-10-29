# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The capability catalog the observer measures against (docs/gain_of_function_v4/
capability_catalog.json): the addendum's embedded catalog verbatim, with each row's detector, main
measure, timescale, sub-rows, observability in this world and the dashboard hand actions it
informs. A catalog, not a curriculum: nothing in it reaches the learner."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
CATALOG = ROOT / "docs" / "gain_of_function_v4" / "capability_catalog.json"


def load(path: str = None) -> dict:
    with open(path or CATALOG, encoding="utf-8") as f:
        cat = json.load(f)
    cat["by_id"] = {c["id"]: c for c in cat["capabilities"]}
    return cat
