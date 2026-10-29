# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The physics-only following anchor scores the chest, and the H84 launcher quotes the stored anchor.

What was wrong (found 2026-09-23): the command_following anchor in Saved/mimo_bridge/scale_anchors.json
(measured 2026-09-18) was taken on the env gauge that took the LAST handle row for the eyes' row. In
trunk_v1 (every H84 segment) the last row is the chest, so every chest trial scored no part and went
into its cell as a direction match of 0: chest 0.0 in all 20 cells, 2 of every 30 trials. The gain-6
cell read 0.716, and tools/launch_h84.py quoted 0.716 as the line "the physics allows" in its docstring
and in the note the hourly watcher prints for the execution_direction_match criterion.

The tool builds trunk_v1 worlds only (it orders the chest and the lower trunk, rows the 24-row legacy
layout does not have), so there is no legacy case. These tests read the stored anchor and the launcher;
they build no world.
"""
import json
from pathlib import Path

import pytest

from mimo_bridge.tools import launch_h84
from mimo_bridge.tools.check_no_bare_decimals import bare_decimals

REPO = Path(__file__).resolve().parents[4]
ANCHORS = REPO / "Saved" / "mimo_bridge" / "scale_anchors.json"
DOCS_CRITERIA = REPO / "docs" / "h84_launch_2026-09-18" / "declared_criteria.json"
H84_GAIN, H84_STRENGTH = 6.0, 0.5          # launch_h84 --decoder-kp 6.0; the anchor's strength 0.5 cell


@pytest.fixture(scope="module")
def anchor():
    return json.loads(ANCHORS.read_text(encoding="utf-8"))["command_following"]


@pytest.fixture(scope="module")
def h84_cell(anchor):
    return next(r for r in anchor["table"] if r["kp"] == H84_GAIN and r["strength"] == H84_STRENGTH)


def _note():
    return next(c for c in launch_h84.CRITERIA if c["name"] == "execution_direction_match")["note"]


def test_every_cell_of_the_anchor_scored_the_chest(anchor):
    for r in anchor["table"]:
        assert r.get("trials_unscored") == 0, (r["kp"], r["strength"], "a trial the gauge scored no part for")
        assert r["per_group"]["chest"] is not None, (r["kp"], r["strength"])
        assert r["per_group"]["chest"] != 0.0, (r["kp"], r["strength"], "the chest reads exactly 0: unscored")
        assert r["trials"] == anchor["trials_per_cell"]


def test_the_anchor_carries_counts_a_person_can_picture(h84_cell):
    assert 0.0 <= h84_cell["closer_out_of_100"] <= 100.0
    assert h84_cell["closer_parts"] == h84_cell["trials"]
    lo, hi = h84_cell["direction_match_ci95"]
    assert lo <= h84_cell["direction_match_mean"] <= hi


def test_the_watcher_note_quotes_the_stored_anchor_without_bare_decimals(h84_cell):
    note = _note()
    pct = round(100 * h84_cell["direction_match_mean"])
    lo, hi = (round(100 * v) for v in h84_cell["direction_match_ci95"])
    assert f"{pct}% of the way" in note
    assert f"{lo}% to {hi}%" in note
    assert f"{round(h84_cell['closer_out_of_100'])} of every 100" in note
    assert "0.716" not in note, "the 2026-09-18 figure counted the chest trials as 0"
    assert bare_decimals(note) == [], "DESIGN.md section 11: the watcher prints this note"


def test_the_launcher_docstring_quotes_the_re_measured_gain_6_cell(h84_cell):
    doc = launch_h84.__doc__
    assert "direction match 0.716" not in doc
    assert f"{round(100 * h84_cell['direction_match_mean'])}% of the way" in doc
    assert "re-measured 2026-09-23" in doc


def test_the_tracked_copy_of_the_launch_criteria_carries_the_correction(h84_cell):
    crit = json.loads(DOCS_CRITERIA.read_text(encoding="utf-8"))
    note = next(c for c in crit if c["name"] == "execution_direction_match")["note"]
    assert f"{round(100 * h84_cell['direction_match_mean'])}% of the way" in note
    assert "CORRECTION 2026-09-23" in note
