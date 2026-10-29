# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""lineage_series' following goal lines, and the mark where the following gauge changed
(fix-rows-r2-lineage_series, 2026-09-23).

What was wrong. The dashboards' goal line for "the body turns the way it was ordered" came from the
case of measurements/following_out_of_100.json measured with tools/whole_body_following.py as written
before commit b55ee9a4d: the gauge took the LAST handle row for the eyes, which in the trunk_v1 layout
is the chest, so it scored the eyes' row and never the chest. When that case was absent it fell back
to scale_anchors.json whole_body_following (2026-09-19), the same defect. Its reference was labelled
"25 independent 20-degree turns" whatever the stored part count. follow_bout, a one-frame (10 ms)
reading, drew that 250 ms goal line. And the lineage charts joined readings from before and after
b55ee9a4d with no mark.

Every test here runs without a MuJoCo world; the physics numbers are fixtures. The part list is
checked in both body layouts: trunk_v1 (26 rows; the gauge scores 25, the chest and lower trunk
included) and legacy (24 rows; it scores 23).
"""
import json
from pathlib import Path

import pytest

from mimo_bridge.body_layout import names_for
from mimo_bridge.tools import lineage_series as ls
from mimo_bridge.tools.check_no_bare_decimals import readout_problems

AS_WRITTEN = "physics_only_as_whole_body_following_is_written"
REPAIRED = "physics_only_on_the_repaired_gauge"
HEADLINE = {"live_model_orders": {"closer_out_of_100": 54.2},
            "physics_only_reachable_pose": {"closer_out_of_100": 87.4},
            "physics_only_independent_random_turns": {"closer_out_of_100": 69.1},
            "chance_order_ignored": {"closer_out_of_100": 31.3}}


def _every_part_but_the_eyes(layout):
    return [n for n in names_for(layout) if n != "gaze"]


def _repaired(layout, parts=None):
    return {REPAIRED: {"measured": "2026-09-23", "layout": layout, "angle_deg": 20.0,
                       "parts_scored": list(parts if parts is not None else _every_part_but_the_eyes(layout)),
                       "reachable_pose": {"held_250ms_mean": 0.83, "one_frame_10ms_mean": 0.46},
                       "independent_turns": {"held_250ms_mean": 0.55, "one_frame_10ms_mean": 0.3}}}


def _as_written():
    return {AS_WRITTEN: {"note": "row n-1 = chest left unordered, the gaze row ordered",
                         "reachable_pose": {"tool_gauge_mean": 0.826},
                         "independent_turns": {"tool_gauge_mean": 0.518}}}


def _disk(tmp_path, monkeypatch, cases, superseded_anchor=True):
    """A measurement file and a scale_anchors.json under tmp_path, read through the module's own
    MEASUREMENTS and REPO (the names the reader used before and after the fix)."""
    meas = tmp_path / "docs/gain_of_function_v3/measurements"
    meas.mkdir(parents=True)
    (meas / "following_out_of_100.json").write_text(json.dumps(
        {"measured": "2026-09-23", "headline": HEADLINE, "cases": cases}), encoding="utf-8")
    if superseded_anchor:
        anchors = tmp_path / "Saved/mimo_bridge/scale_anchors.json"
        anchors.parent.mkdir(parents=True)
        anchors.write_text(json.dumps({"whole_body_following": {
            "consistent": {"direction_match_mean": 0.63}, "independent": {"direction_match_mean": 0.37},
            "superseded": "measured on the gauge that took the chest for the eyes"}}), encoding="utf-8")
    monkeypatch.setattr(ls, "MEASUREMENTS", meas)
    monkeypatch.setattr(ls, "REPO", tmp_path)


def test_the_as_written_case_and_the_superseded_anchor_are_never_a_goal_line(tmp_path, monkeypatch):
    _disk(tmp_path, monkeypatch, _as_written())
    ref = ls.following_reference()
    assert ref["gauge_reachable"] is None and ref["gauge_independent"] is None, \
        "the case measured on the gauge that skipped the chest is not a goal line"
    assert ref["live"] == 54.2 and ref["reachable"] == 87.4, "the headline counts are still read"
    assert "repaired gauge" in ref["gauge_why"]


def test_no_fallback_to_the_superseded_anchor_when_no_case_is_stored(tmp_path, monkeypatch):
    _disk(tmp_path, monkeypatch, {})
    ref = ls.following_reference()
    assert ref["gauge_reachable"] is None and ref["gauge_independent"] is None, \
        "scale_anchors.json whole_body_following (2026-09-19) carries the same defect"
    lines = ls.follow_lines(ref)
    assert lines["goal"] is None and lines["bout_goal"] is None and lines["refs"] == []
    assert "repaired gauge" in lines["goal_why"]


@pytest.mark.parametrize("layout,count", [("trunk_v1", 25), ("legacy", 23)])
def test_the_repaired_case_is_read_and_labelled_by_its_own_part_list(tmp_path, monkeypatch, layout, count):
    _disk(tmp_path, monkeypatch, dict(_as_written(), **_repaired(layout)))
    ref = ls.following_reference()
    assert (ref["gauge_reachable"], ref["gauge_independent"]) == (0.83, 0.55)
    assert (ref["frame_reachable"], ref["frame_independent"]) == (0.46, 0.3)
    assert len(ref["gauge_parts"]) == count and "gaze" not in ref["gauge_parts"]
    if layout == "trunk_v1":
        assert "chest" in ref["gauge_parts"] and "lower_trunk" in ref["gauge_parts"]
    lines = ls.follow_lines(ref)
    label = lines["refs"][0][1]
    assert label.startswith(f"{count} independent 20-degree turns") and "every body part but the eyes" in label
    assert f"the {count} body parts scored are every one but the eyes" in lines["goal"][1]
    assert not readout_problems(label + ". " + lines["goal"][1])


def test_a_reference_over_another_part_set_is_refused_by_name(tmp_path, monkeypatch):
    # the as-written part set in trunk_v1: the eyes' row ordered and scored, the chest left out
    as_written_parts = [n for n in names_for("trunk_v1") if n != "chest"]
    _disk(tmp_path, monkeypatch, _repaired("trunk_v1", parts=as_written_parts))
    ref = ls.following_reference()
    assert ref["gauge_reachable"] is None and ref["frame_reachable"] is None
    assert "left out chest" in ref["gauge_why"] and "also scored the eyes" in ref["gauge_why"]
    # a legacy reference that lists the trunk rows is not the legacy gauge's part set either
    legacy_with_trunk = _every_part_but_the_eyes("legacy") + ["lower_trunk", "chest"]
    parts, why = ls.check_parts({"layout": "legacy", "parts_scored": legacy_with_trunk})
    assert parts is None and "also scored lower_trunk, chest" in why
    assert ls.check_parts({"layout": "trunk_v1", "parts_scored": _every_part_but_the_eyes("trunk_v1")})[0]


def test_follow_bout_never_draws_the_250_ms_goal_line():
    # the registry itself: follow_bout's goal is the stored one-frame value or none, never the 250 ms one
    fb, held = ls.SERIES["follow_bout"], ls.SERIES["follow"]
    if fb["goal"] is None:
        assert fb["goal_why"]
    else:
        assert fb["goal"][0] == ls.FOLLOW.get("frame_reachable") != ls.FOLLOW.get("gauge_reachable"), fb["goal"]
    assert held["goal"] is None or held["goal"][0] == ls.FOLLOW.get("gauge_reachable")
    ref = dict(gauge_reachable=0.83, gauge_independent=0.55, frame_reachable=None, frame_independent=None,
               gauge_parts=_every_part_but_the_eyes("trunk_v1"), gauge_angle=20.0, gauge_source="x",
               gauge_why=None)
    lines = ls.follow_lines(ref)
    assert lines["goal"][0] == 0.83 and lines["bout_goal"] is None and lines["bout_refs"] == []
    assert "250 ms held-command gauge" in lines["bout_goal_why"] and "one-frame (10 ms)" in lines["bout_goal_why"]
    lines = ls.follow_lines(dict(ref, frame_reachable=0.46, frame_independent=0.3))
    assert lines["bout_goal"][0] == 0.46 and lines["bout_refs"][0][0] == 0.3
    assert "one-frame (10 ms) gauge this series averages" in lines["bout_goal"][1]


def test_the_stored_reference_scored_every_body_part_but_the_eyes():
    """The measurement committed with this fix: the goal line the pages draw is read, and it scored the
    live gauge's own part set, found by name."""
    ref = ls.FOLLOW
    assert ref.get("gauge_parts") and ref.get("frame_reachable") is not None, ref.get("gauge_why")
    assert ref["gauge_reachable"] is not None
    assert ref["gauge_layout"] == "trunk_v1" and ref["gauge_parts"] == ls.gauge_parts("trunk_v1")
    assert "chest" in ref["gauge_parts"] and "gaze" not in ref["gauge_parts"]
    assert ls.SERIES["follow"]["goal"][0] == ref["gauge_reachable"]
    assert ls.SERIES["follow_bout"]["goal"][0] == ref["frame_reachable"]


@pytest.mark.parametrize("layout,count", [("trunk_v1", 25), ("legacy", 23)])
def test_item_5_names_the_stored_part_count_not_a_fixed_25(monkeypatch, layout, count):
    ref = dict(ls.FOLLOW, gauge_reachable=0.83, gauge_independent=0.55, gauge_parts=_every_part_but_the_eyes(layout),
               gauge_angle=20.0, gauge_source="measurements/following_out_of_100.json")
    monkeypatch.setattr(ls, "FOLLOW", ref)
    turns = ls.independent_turns_text(ref) if hasattr(ls, "independent_turns_text") else None
    monkeypatch.setattr(ls, "FOLLOW_TURNS", turns, raising=False)
    record = {"type": "mimo", "step": 1, "synergy": {"following_by_author": {"synergy": 0.4}, "following_mean": 0.3}}
    panel = ls.gain_of_function(record, {"motor_synergy": "on"}, {"items": {}, "index_found": False})
    text = ls.item_text(panel["items"][4])
    assert f"{count} independent 20-degree turns, one for every body part but the eyes, reach 55%" in text, text
    assert (count == 25) or "25 independent" not in text
    assert not readout_problems(text), readout_problems(text)


def _stats(run, rows, commit):
    run.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"type": "launch", "step": rows[0]["step"] - 100, "code_commit": commit,
                         "timestamp": 1_790_000_000})]
    lines += [json.dumps(r) for r in rows]
    (run / "replay_training_stats.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _lineage(root):
    saved = root / "Saved/mimo_bridge"
    parent, child = saved / "experiments/H1", saved / "experiments/H1_seg2"
    (parent).mkdir(parents=True)
    (parent / "args.json").write_text("{}", encoding="utf-8")
    child.mkdir(parents=True)
    (child / "args.json").write_text(json.dumps({"resume": str(parent / "checkpoints/ckpt_step_300")}),
                                     encoding="utf-8")

    def rec(step, held, bout):
        return {"type": "mimo", "step": step, "timestamp": 1_790_000_000 + step, "cmd_response_cos_held": held,
                "follow_bout": bout, "cmd_mag_rad": 0.4}
    _stats(parent, [rec(s, 0.14, 0.067) for s in (100, 200, 300)], "oldc0de1")
    _stats(child, [rec(s, 0.30, 0.070) for s in (400, 500)], "newc0de2")
    (saved / "training_continuation.json").write_text(json.dumps(
        {"active_run": str(child), "timestamp": 1_790_000_700, "status": "training"}), encoding="utf-8")


def _commits(runs_fix):
    def fake(commit, fix=ls.ENV_FIX_COMMIT, root=None):
        if fix != "b55ee9a4d":
            return None
        return runs_fix.get(commit)
    return fake


def test_the_gauge_change_is_marked_where_the_first_segment_running_it_begins(tmp_path, monkeypatch):
    _lineage(tmp_path)
    monkeypatch.setattr(ls, "commit_has", _commits({"oldc0de1": False, "newc0de2": True}))
    out = ls.build(tmp_path, cache_dir=tmp_path / "cache", keys="all", now=1_790_001_000)
    for key in ("follow", "follow_bout", "cmd_deg"):
        assert out["defs"][key]["seam_at"] == {"step": 400, "label": "seg2"}, key
    follow = next(p for c in out["cards"] for p in c["panels"] if p["keys"] == ["follow"])
    bout = next(p for p in out["more"] if p["keys"] == ["follow_bout"])
    for panel, before, after in ((follow, "14%", "30%"), (bout, "6.7%", "7.0%")):
        seam = panel["texts"]["seam"]
        assert "commit b55ee9a4d" in seam and "segment seg2 (lineage step 400)" in seam, seam
        assert f"read {before} of the way" in seam or f"was {before} of the way" in seam, seam
        assert after in seam and "a step there is the change of gauge, not of the body" in seam
        assert not readout_problems(seam), readout_problems(seam)
        # the latest reading (segment seg2) is on the gauge the goal line was measured on
        assert "percentage points short of it" in panel["texts"]["goal"], panel["texts"]["goal"]
    assert "defs" in out and out["defs"]["act.vision"].get("seam") is None, "an unchanged gauge carries no seam"


def test_before_any_segment_runs_the_change_every_reading_is_said_to_be_on_the_old_gauge(tmp_path, monkeypatch):
    _lineage(tmp_path)
    monkeypatch.setattr(ls, "commit_has", _commits({"oldc0de1": False, "newc0de2": False}))
    out = ls.build(tmp_path, cache_dir=tmp_path / "cache", keys="all", now=1_790_001_000)
    assert out["defs"]["follow"]["seam_at"] is None
    follow = next(p for c in out["cards"] for p in c["panels"] if p["keys"] == ["follow"])
    assert "No segment of this lineage runs that commit yet, so every reading here is on the old gauge" \
        in follow["texts"]["seam"]
    # a reading on the old gauge is not set against a goal line measured on the new one
    bout = next(p for p in out["more"] if p["keys"] == ["follow_bout"])
    for panel in (follow, bout):
        goal = panel["texts"]["goal"]
        assert "Distance not given: this reading is on the gauge before commit b55ee9a4d" in goal, goal
        assert "percentage points" not in goal
    # a segment whose commit git cannot check is said to be counted on the old gauge without proof
    monkeypatch.setattr(ls, "commit_has", _commits({"oldc0de1": None, "newc0de2": True}))
    text, seg = ls.seam_text(ls.SERIES["follow"], [dict(code_commit="oldc0de1", label="H1", first_step=100),
                                                   dict(code_commit="newc0de2", label="seg2", first_step=400)])
    assert seg["label"] == "seg2" and "1 segment of this lineage before it recorded no code commit" in text


def test_the_page_prints_the_seam_sentence_and_draws_the_mark():
    js = (Path(ls.__file__).with_name("development_dashboard.js")).read_text(encoding="utf-8")
    assert "T.seam" in js and "d0.seam_at" in js and "class:'seam'" in js
    css = (Path(ls.__file__).with_name("development_dashboard.css")).read_text(encoding="utf-8")
    assert ".gauge-svg .seam{" in css


def test_the_whole_life_page_says_where_the_following_gauge_changed(tmp_path, monkeypatch):
    from mimo_bridge.tools import lifetime
    from mimo_bridge.tools.check_no_bare_decimals import page_text
    saved = tmp_path / "Saved/mimo_bridge"
    parent, child = saved / "experiments/P", saved / "experiments/P_seg2"
    parent.mkdir(parents=True)
    (parent / "args.json").write_text("{}", encoding="utf-8")
    child.mkdir(parents=True)
    (child / "args.json").write_text(json.dumps({"resume": str(parent / "checkpoints/ckpt_step_109")}),
                                     encoding="utf-8")

    def rec(step, held, bout):
        return {"type": "mimo", "step": step, "timestamp": 1_790_000_000 + step, "cmd_response_cos_held": held,
                "follow_bout": bout}
    _stats(parent, [rec(100 + i, 0.14, 0.067) for i in range(10)], "oldc0de1")
    _stats(child, [rec(200 + i, 0.30, 0.070) for i in range(10)], "newc0de2")
    (saved / "training_continuation.json").write_text(json.dumps({"active_run": str(child), "timestamp": 1}),
                                                      encoding="utf-8")
    for name, value in (("SAVED", str(saved)), ("EXP", str(saved / "experiments")),
                        ("OUT_HTML", str(saved / "lifetime.html")), ("OUT_JSON", str(saved / "lifetime.json"))):
        monkeypatch.setattr(lifetime, name, value)
    monkeypatch.setattr(ls, "commit_has", _commits({"oldc0de1": False, "newc0de2": True}))
    lifetime.build(force=True)
    runs = json.loads((saved / "lifetime.json").read_text(encoding="utf-8"))
    assert [(r["run"], r.get("code_commit"), r.get("first_step")) for r in runs] == \
        [("P", "oldc0de1", 100), ("P_seg2", "newc0de2", 200)]
    text = page_text((saved / "lifetime.html").read_text(encoding="utf-8"))
    item2 = text[text.index("2. The body turns the way it was commanded"):]
    item2 = item2[:item2.index("3. ")]
    assert "commit b55ee9a4d" in item2 and "segment P_seg2 (lineage step 200)" in item2, item2
    assert "14% of the way" in item2 and "30%" in item2
    assert "percentage points short of it" in item2, "the newest run is on the gauge the goal was measured on"
    card = text[text.index("Following a command"):]
    assert "segment P_seg2 (lineage step 200)" in card and "6.7% of the way" in card, card[:800]
    assert not readout_problems(item2) and not readout_problems(card[:1500])
    # no run runs the change yet: every reading is on the old gauge and no distance is given
    monkeypatch.setattr(ls, "commit_has", _commits({"oldc0de1": False, "newc0de2": False}))
    lifetime.build(force=True)
    text = page_text((saved / "lifetime.html").read_text(encoding="utf-8"))
    item2 = text[text.index("2. The body turns the way it was commanded"):]
    item2 = item2[:item2.index("3. ")]
    assert "Distance not given: this reading is on the gauge before commit b55ee9a4d" in item2, item2
    assert "No segment of this lineage runs that commit yet" in item2
