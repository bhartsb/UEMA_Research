# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""An empty failed launch must not replace the selected run in current readings."""
import pytest
from mimo_bridge.tools.observer import daemon as DM
from mimo_bridge.tools.observer import catalog as CAT
from mimo_bridge.tools.observer import store as ST


@pytest.mark.parametrize("recorded", [True, False])
def test_report_uses_selected_run_before_trailing_empty_launch(tmp_path, monkeypatch, recorded):
    selected = tmp_path / "H99_seg1_active"
    failed = tmp_path / "H99_seg1_empty_failed"
    for path in (selected, failed):
        (path / "observer/world").mkdir(parents=True)
    if recorded:
        (selected / "observer/model").mkdir()
    monkeypatch.setattr(DM, "SAVED", str(tmp_path / "saved"))
    monkeypatch.setattr(DM, "ModelLaneIndex", lambda files: files.model)
    monkeypatch.setattr(DM.HZ, "report", lambda lane, chain: {"selected_lane": lane})
    monkeypatch.setattr(DM, "recorder_health", lambda path: {"selected": str(path)})
    monkeypatch.setattr(DM, "_run_vision", lambda path: str(path))
    store = ST.LineageStore(None, DM.quality_map(CAT.load()))
    result = DM.build_report([str(selected), str(failed)], str(tmp_path / "out/development.json"),
                             render_replays=False, store=store, current_run=str(selected))
    assert result["current_run"] == selected.name
    assert result["health"] == {selected.name: {"selected": str(selected)}}
    assert result["baseline"]["training_vision"] == str(selected)
    assert result["horizon"]["selected_lane"] == (str(selected / "observer/model") if recorded else None)
    assert result["runs"] == [selected.name, failed.name]
