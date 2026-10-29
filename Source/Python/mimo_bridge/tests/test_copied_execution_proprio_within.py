# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The probes' snapshot copies the body-position repair's tracker (2026-09-29, H84 segment 41).

Every scheduled ability probe stopped at its first snapshot with "Unserialized copied model state: proprio_within
(WithinBody)" once --proprio-token-repair within_body was on.
"""
import torch

from mimo_bridge import copied_execution as CE
from mimo_bridge.proprio_token import WithinBody


def test_the_tracker_is_copied_exactly_without_its_step_graph():
    w = WithinBody(line=3.77, deaf=2.87, skip="3")
    w.ring = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    w.engaged, w.attributed, w.count, w.ptr = [True, False], [0.1, 0.2], 3, 1
    leaf = torch.ones(2, requires_grad=True)
    w._read_graph = (leaf * 2).log()                   # a non-leaf tensor: deepcopy of it raises
    w.rows_transform = lambda rows: rows
    found, saved = CE._capture_special(w, "proprio_within")
    assert found
    back = CE._restore_special(None, saved)
    assert back is not w and back._read_graph is None and back.rows_transform is w.rows_transform
    assert torch.equal(back.ring, w.ring) and back.ring is not w.ring
    assert (back.engaged, back.attributed, back.count, back.ptr) == (w.engaged, w.attributed, w.count, w.ptr)
    assert w._read_graph is not None                   # the live tracker keeps its graph
    back.ring[0, 0, 0] = 99.0
    assert w.ring[0, 0, 0] == 0.0                      # restored copies never share the snapshot's tensors
