# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""mimo_bridge: lets the LEM model drive the MIMo MuJoCo infant body.

The model never sees MIMo. It sees one sensory frame (bytes laid out exactly as the
UE pipeline writes them, decoded by the model's own tokenizer) and writes one action
frame (23 handles x 9 floats + gaze). This package is the translator.

Layout (all paths under Source/Python/mimo_bridge):
  env.py             make and step a MIMo env (v2 body, own scene XML, age, delays)
  touch_adapter.py   5332 skin points -> at most 256 patches with derived fields
  proprio_adapter.py 23 tracked bones from MIMo bodies (quat, pos, vel, fatigue)
  vision_adapter.py  two eye cameras -> patches + angular positions
  frame_encoder.py   packs the 31040-byte frame using tokenizer/config.py constants
  action_decoder.py  23 handle targets -> torque-motor ctrl (position servo)
  datasource.py      Mode A: in-process frame provider for the model
  shm_writer.py      Mode B: writes LEM_Pipeline / LEM_Vision, reads LEM_Action_Buffer
  train_mimo.py      training loop (many envs, curriculum, stats, checkpoints)
  experiments/       E1..E5
  tests/             unit tests with hand-worked answers

MIMo is an external editable install (C:\\Users\\bhart\\Documents\\MIMo_private, branch v3,
commit f2be26b). Nothing from MIMo is copied here.
"""
from __future__ import annotations

import os
import sys

# Make `tokenizer` and `lem` importable when this package is imported from anywhere.
_SRC_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_PY not in sys.path:
    sys.path.insert(0, _SRC_PY)

# Paths the records name inside D:\UEMA_Research are read from this checkout, wherever it is (repo_paths.py).
from . import repo_paths as _repo_paths

_repo_paths.install()

__version__ = "0.1.0"
