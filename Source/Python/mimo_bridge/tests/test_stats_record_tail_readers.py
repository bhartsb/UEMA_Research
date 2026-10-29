# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The readers that take the newest record from the END of replay_training_stats.jsonl still find it when one
record is larger than their first tail size.

Why (2026-10-03): H84's last 59 'mimo' records were 182,761 to 234,927 bytes. The swap test's added readings and
controllability's shadow gate add 36,215 bytes to each record, which puts 7 of those 59 past the 256,000 bytes
development_dashboard.last_training_record read from the file's end, and 3 past the 262,144 bytes
sensor_action_contract_check.latest_phase read. A tail that begins inside the newest record's own line holds no
complete record: the Development page then had no newest record for that report window, and latest_phase read the
phase as absent. Both now grow the tail until a complete record is found. Temporary files only."""
import json

import pytest

from mimo_bridge.tools import development_dashboard as dd
from mimo_bridge.tools import sensor_action_contract_check as sacc


def _stats(tmp_path, record_bytes, trailing=True, steps=(100, 200, 300)):
    """A stats file whose 'mimo' records are each `record_bytes` long (padded with a text field), with the small
    records a trainer writes between them."""
    path = tmp_path / "replay_training_stats.jsonl"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"type": "launch", "step": 0, "timestamp": 0}) + "\n")
        for step in steps:
            f.write(json.dumps({"type": "training", "step": step, "loss": 1.0}) + "\n")
            row = {"type": "mimo", "step": step, "timestamp": 1_790_000_000 + step, "phase": "mature", "pad": ""}
            row["pad"] = "x" * (record_bytes - len(json.dumps(row)))
            line = json.dumps(row)
            assert len(line) == record_bytes
            f.write(line + "\n")
            if trailing:
                f.write(json.dumps({"type": "inference", "step": step, "ms": 12.5}) + "\n")
    return path


@pytest.mark.parametrize("record_bytes", [200_000, 256_500, 271_142, 900_000])
def test_the_newest_record_is_found_whatever_its_size(tmp_path, record_bytes):
    """271,142 bytes = H84's largest record of Oct 3 (234,927) plus the 36,215 the candidate adds."""
    path = _stats(tmp_path, record_bytes)
    row = dd.last_training_record(path)
    assert row.get("step") == 300 and row.get("type") == "mimo" and len(row["pad"]) > record_bytes - 200
    assert sacc.latest_phase(str(tmp_path)) == "mature"


def test_a_record_being_written_does_not_replace_the_last_complete_one(tmp_path):
    path = _stats(tmp_path, 271_142)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write('{"type": "mimo", "step": 400, "timestamp": 1790000400, "phase": "mature", "pad": "' + "y" * 150_000)
    assert dd.last_training_record(path).get("step") == 300
    assert sacc.latest_phase(str(tmp_path)) == "mature"


def test_a_missing_or_empty_file_reads_as_no_record(tmp_path):
    assert dd.last_training_record(tmp_path / "replay_training_stats.jsonl") == {}
    assert sacc.latest_phase(str(tmp_path)) is None
    (tmp_path / "replay_training_stats.jsonl").write_text("", encoding="utf-8")
    assert dd.last_training_record(tmp_path / "replay_training_stats.jsonl") == {}
    assert sacc.latest_phase(str(tmp_path)) is None
    (tmp_path / "replay_training_stats.jsonl").write_text(json.dumps({"type": "mimo", "step": 5, "timestamp": 9,
                                                                    "phase": "fixed"}) + "\n", encoding="utf-8")
    assert dd.last_training_record(tmp_path / "replay_training_stats.jsonl")["step"] == 5
    assert sacc.latest_phase(str(tmp_path)) is None, "'fixed' means no curriculum"


def test_the_check_sees_a_reader_held_to_its_first_tail_size(tmp_path, monkeypatch):
    """The planted defect: each reader with its first size alone, as before 2026-10-03. A record of 271,142 bytes is
    then not found, which is what the tests above would catch."""
    path = _stats(tmp_path, 271_142)
    monkeypatch.setattr(dd, "TRAINING_RECORD_TAILS", (256_000,))
    monkeypatch.setattr(sacc, "PHASE_TAILS", (262144,))
    assert dd.last_training_record(path) == {}
    assert sacc.latest_phase(str(tmp_path)) is None
    # a record under the first size is found by the first read alone
    small = tmp_path / "small"
    small.mkdir()
    assert dd.last_training_record(_stats(small, 200_000)).get("step") == 300
