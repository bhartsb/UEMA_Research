# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The overseer's "can the model read its own action" check runs on the run's own body and size.

Found 2026-09-23 (fix-rows-r2): the check built a default LemLite - the 24-row legacy body at the
lite width - and loaded the checkpoint's encoders into it with strict=False. strict=False does not
forgive a size mismatch, and every H84 checkpoint's motor encoder carries a position table for 26
rows (the trunk_v1 body) at width 192 (patch_pos [1, 26, 192]), so the check printed "could not
run" for the whole H84 lineage. It now builds only the motor encoder from the checkpoint's receipt
(body layout) and the run's args.json (size flags), with dashboard.load_motor_encoder, and feeds it
one synthetic row per body part of that layout. The size here is H84's (d_model 192, encoder
width 192, 2 encoder layers) for both layouts, so the legacy case fails on the old code too.
"""
import json

import pytest

from mimo_bridge.lem_model import LemLite
from mimo_bridge.tools import overseer
from mimo_bridge.tools.check_no_bare_decimals import readout_problems

# H84's size flags (every H84 args.json: d_model 192, encoder_width 192, encoder_layers 2), as
# train_mimo.size_overrides applies them to the core width and the motor encoder
H84_ARGS = {"d_model": 192, "encoder_width": 192, "encoder_layers": 2}
H84_MOTOR = {"model": {"d_model": 192},
             "modalities": {"motor": {"encoder": {"n_cross_attn_layers": 2, "d_internal": 192,
                                                  "per_patch_hidden": 192}}}}


def _run(root, layout, size_args, overrides):
    from lem.training.checkpoint import CheckpointManager
    run = root / layout
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False, body_layout=layout,
                overrides=overrides)
    m.checkpoints = CheckpointManager(m.config, str(run / "checkpoints"))
    assert m.save_checkpoint(7)
    (run / "args.json").write_text(json.dumps(dict(size_args, body_layout=layout)))
    (run / "replay_training_stats.jsonl").write_text(json.dumps(
        {"type": "mimo", "step": 7, "timestamp": 0, "elapsed_s": 60,
         "next_pe": {"vision": 0.1, "tactile": 0.1, "proprioception": 0.1}}) + "\n")
    return str(run)


@pytest.mark.parametrize("layout,n_parts,size_args,overrides", [
    ("trunk_v1", 26, H84_ARGS, H84_MOTOR),      # H84 itself: 26 rows at width 192
    ("legacy", 24, H84_ARGS, H84_MOTOR),        # the old body at H84's size
    ("legacy", 24, {}, None),                   # the default lite model: worked before, still does
])
def test_the_readability_check_runs_on_the_runs_own_body_and_size(tmp_path, monkeypatch, layout,
                                                                  n_parts, size_args, overrides):
    monkeypatch.setattr(overseer, "free_gb", lambda: (99.0, 999.0))   # no RAM re-sample sleep
    run = _run(tmp_path, layout, size_args, overrides)
    found = overseer.inspect(run)["findings"]["researcher"]
    failed = [f for f in found if "could not run" in f]
    assert not failed, failed
    lines = [f for f in found if "motor token" in f]
    assert len(lines) == 1, found
    line = lines[0]
    assert f"all {n_parts} body parts" in line, line
    if layout == "trunk_v1":
        # the recorded lived rows are H84's 26-part body: the one-part-at-a-time reading runs
        assert "20-degree turn" in line, line
        assert ("UNREAD for " in line and f"of {n_parts} body parts" in line) or (
            f"action readable for all {n_parts} body parts" in line), line
    else:
        assert "per-body-part readability unmeasured" in line, line
    assert not readout_problems(line), readout_problems(line)


def test_a_defect_is_still_reported_as_could_not_run(tmp_path, monkeypatch):
    """A checkpoint that cannot be read must stay visible, never silently skipped."""
    monkeypatch.setattr(overseer, "free_gb", lambda: (99.0, 999.0))
    run = _run(tmp_path, "trunk_v1", H84_ARGS, H84_MOTOR)
    (tmp_path / "trunk_v1" / "args.json").write_text(json.dumps({"body_layout": "trunk_v1"}))
    found = overseer.inspect(run)["findings"]["researcher"]
    assert any("action-readability check could not run" in f and "size mismatch" in f
               for f in found), found
