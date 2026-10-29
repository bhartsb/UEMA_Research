# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Atomic, separate hierarchy state with strict morphology/configuration binding."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

VERSION = "predictive_latent_body_v1"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, learner):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = learner.state_dict()
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load(path, learner):
    # These files contain only tensors and primitive containers; never executable objects.
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    learner.load_state_dict(state)


def validate_header(state, learner):
    if (state.get("version") != VERSION or state.get("tree") != learner.tree.fingerprint
            or state.get("configuration") != fingerprint(learner.config)):
        raise ValueError("Hierarchy checkpoint version, body tree or measured configuration changed")
