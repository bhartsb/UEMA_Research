# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""THE TRAINER CODE'S HASH: one definition, read by two checks so that they cannot drift apart.

- The training preflight (mimo_bridge/tools/training_preflight.py, configuration): the per-file hashes are a preflight
  record's `code`, the code hash its `code_hash`, one of the three parts of every launch command's fingerprint.
- The admission check of a replacement hand calibration (lem/model/hand_response.py admission_check, called by
  LemLite._replace_saved_hand_calibration only when a launch gives --replace-saved-hand-calibration and the requested
  calibration differs from the saved one): an admission record is refused unless its `code_hash` equals
  trainer_code_hash() of the code that runs the launch.

SCOPE, as paths relative to Source/Python with forward slashes (in_scope):
- every .py file under lem/, mimo_bridge/ and tokenizer/ (ROOTS; the trainer imports tokenizer, for example
  mimo_bridge/lem_model.py and mimo_bridge/frame_encoder.py);
- the data files the trainer reads inside those folders (DATA): lem/master_config.yaml (the model's configuration,
  mimo_bridge/lem_model.py CONFIG_PATH) and every file under mimo_bridge/assets/ (the crib's sound loops, read by
  mimo_bridge/audio_adapter.py);
- skipped: mimo_bridge/tools/, mimo_bridge/explainer/ and mimo_bridge/experiments/ (SKIP), every folder named tests,
  every folder whose name starts with '.' (an ignored cache such as mimo_bridge/.pytest_cache, which a pytest run
  rewrites) and Python's caches (__pycache__).
A file inside the scope counts whether or not git tracks it: the hash describes the files a launch from that folder runs.

OPERATION: file hash = SHA-256 of the file's bytes with every CRLF read as LF (a checkout with core.autocrlf true and a
git archive of the same commit give the same file hashes); code hash = SHA-256 of the UTF-8 text made of one line
'<path> <file hash>' per file in scope, sorted by path, joined by newlines.

This module imports only the standard library: the trainer's code reads it without importing the tools package or torch.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

ROOTS = ("lem", "mimo_bridge", "tokenizer")
DATA = ("lem/master_config.yaml", "mimo_bridge/assets/")
SKIP = ("mimo_bridge/tools/", "mimo_bridge/explainer/", "mimo_bridge/experiments/")
SKIP_FOLDER_NAMES = ("tests", "__pycache__")
SOURCE_PYTHON = Path(__file__).resolve().parents[1]


def skipped_folder(name: str) -> bool:
    """True for a folder name the scope never enters at any depth: tests, __pycache__, or a name starting with '.'."""
    return name.startswith(".") or name in SKIP_FOLDER_NAMES


def in_scope(rel: str) -> bool:
    """True when the file at `rel` (relative to Source/Python, forward slashes) belongs to the trainer code (the module
    text's SCOPE)."""
    parts = str(rel).split("/")
    if len(parts) < 2 or parts[0] not in ROOTS or any(skipped_folder(p) for p in parts[:-1]):
        return False
    if str(rel).startswith(SKIP):
        return False
    return (str(rel).endswith(".py") or str(rel) in DATA
            or any(d.endswith("/") and str(rel).startswith(d) for d in DATA))


def file_hash(path) -> str:
    """SHA-256 of a file's bytes with every CRLF read as LF."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def file_hashes(source_python=None) -> dict:
    """{path relative to Source/Python: file_hash} of every file in scope under `source_python` (default: the
    Source/Python folder this file lies in), sorted by path. A root folder that does not exist adds nothing."""
    root = SOURCE_PYTHON if source_python is None else Path(source_python)
    out = {}
    for top in ROOTS:
        if not (root / top).is_dir():
            continue
        for folder, dirs, files in os.walk(root / top):
            here = Path(folder).relative_to(root).as_posix()
            dirs[:] = sorted(d for d in dirs if not skipped_folder(d) and not (here + "/" + d + "/").startswith(SKIP))
            for name in files:
                rel = here + "/" + name
                if in_scope(rel):
                    out[rel] = file_hash(Path(folder) / name)
    return dict(sorted(out.items()))


def combined_hash(hashes: dict) -> str:
    """The code hash of per-file hashes: SHA-256 of the lines '<path> <file hash>' sorted by path, joined by newlines."""
    return hashlib.sha256("\n".join("%s %s" % kv for kv in sorted(hashes.items())).encode()).hexdigest()


def trainer_code(source_python=None):
    """(file_hashes, code hash) of the trainer code under `source_python` (default: this file's Source/Python)."""
    hashes = file_hashes(source_python)
    return hashes, combined_hash(hashes)


def trainer_code_hash(source_python=None) -> str:
    """The code hash of the trainer code under `source_python` (default: this file's Source/Python)."""
    return trainer_code(source_python)[1]
