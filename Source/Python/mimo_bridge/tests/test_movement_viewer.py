# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The movement viewer (tools/movement_viewer; the operator's request of Oct 18, 2026) and the specification's acceptance
checks (docs/H84_DASHBOARD_MOVEMENT_VIEWER_SPEC_2026-10-17.html, "Small build plan and acceptance checks"):

  * the body-4 command at model step 482110 of seg41j reproduces the October 17 joint values exactly, and its left-hand
    pictures match the October 17 pictures except at edge pixels (the graphics driver's edge smoothing differs from one
    drawing to the next: measured Oct 18, 2026, 4 of 35 pictures differed between two drawings in one process, and at most
    13 of 518,400 pixels of the 28 October 17 pictures differed by more than 16 of 255 levels);
  * a wrong model is refused; a model step where no command starts is refused; a command never runs across an episode
    change or a missing model step;
  * drawing leaves the saved records unchanged, and a repeated request reuses the pictures;
  * the copied decoder definitions equal mimo_bridge/action_decoder.py's;
  * the chart's sums (gap closed, gain kept) and the request checks of the dashboard routes.
Tests that need the saved recordings skip when the disk holding them is absent.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mimo_bridge.tools.movement_viewer import commands as C
from mimo_bridge.tools.movement_viewer import index as I

REPO = Path(__file__).resolve().parents[4]
SEG41J = REPO / "Saved/mimo_bridge/experiments/H84_fresh_drivers_20260918_seg41j_1526"
OCT17 = REPO / "docs/h84_finger_shapes_2026-10-17"
EDGE_PIXELS = 100            # pixels per picture allowed to differ by more than EDGE_LEVELS (measured worst: 13)
EDGE_LEVELS = 16


def _seg41j():
    from mimo_bridge.tools.movement_viewer import sources as S
    if not (SEG41J / "observer/world/b4").is_dir():
        pytest.skip("the seg41j recording is not on this disk")
    recs = S.scan(roots=(("production", str(SEG41J.parent / SEG41J.name / "observer" / "world")),), registry_rows=[])
    assert len(recs) == 1
    return recs[0]


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_the_copied_decoder_definitions_equal_the_decoders():
    from mimo_bridge import action_decoder as AD
    assert C.POSITION_HANDLES == frozenset(AD.POSITION_HANDLES)
    assert C.HEAD_PELVIS_POSITION_HANDLES == frozenset(AD.HEAD_PELVIS_POSITION_HANDLES)
    rng = np.random.default_rng(0)
    for q in rng.normal(size=(20, 4)):
        assert np.allclose(C.quat_xyzw_to_mat(q), AD.quat_xyzw_to_mat(q))
    for rows in ("all", "no_head_pelvis"):
        names = ["wrist_L", "head", "lower_trunk", "thigh_L"]
        assert list(C.position_handle_mask(names, rows)) == [n in AD.position_handles(rows) for n in names]


def test_a_command_never_crosses_an_episode_change_or_a_missing_model_step():
    n = 12
    rec = dict(order_on=np.ones(n, bool), row=np.zeros(n, np.int64),
               order_seq=np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]),
               episode=np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
               step=np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22]))
    assert C.segment(rec) == [(0, 2), (3, 4), (5, 6), (7, 8), (9, 11)]
    rec["order_on"][1] = False                       # a scheduled rest in the middle splits the command
    assert C.segment(rec)[:2] == [(0, 0), (2, 2)]


def test_the_sums_of_the_summary():
    sel = dict(s=np.array([10.0, 20.0, np.nan]), e=np.array([5.0, 10.0, 3.0]), n=np.array([6.0, 15.0, 4.0]))
    out = I.summary(sel)
    assert out["gap_closed"] == round((30 - 15) / 30, 4)                 # (sum start - sum end) / sum start
    assert out["gain_kept"] == round((30 - 21) / (30 - 15), 4)           # (sum start - sum next) / (sum start - sum end)
    assert out["commands"] == 3 and out["with_start"] == 2


def test_the_october_17_command_scores_as_its_evidence_file():
    rec = _seg41j()
    ev = json.loads((OCT17 / "evidence.json").read_text(encoding="utf-8"))
    b = rec["bodies"]["4"]
    chunks = [c for c in b["chunks"] if 482000 <= c[3] <= 482200]
    data = C.load_records(chunks)
    model = C.BodyModel(b["model"], data["headers"][0])
    cols = C.measure(data, model, "all")
    i = int(np.flatnonzero(cols["step"] == ev["command_step"])[0])
    assert int(cols["last_step"][i]) == ev["end_step"] and int(cols["n"][i]) == 5
    assert abs(cols["fing_s"][i][0] - ev["hands"]["left"]["start_mean_error_deg"]) < 1e-4
    assert abs(cols["fing_e"][i][0] - ev["hands"]["left"]["end_mean_error_deg"]) < 1e-4
    assert abs(cols["fing_s"][i][1] - ev["hands"]["right"]["start_mean_error_deg"]) < 1e-4
    assert abs(cols["fing_e"][i][1] - ev["hands"]["right"]["end_mean_error_deg"]) < 1e-4


def test_a_model_with_another_structure_is_refused():
    rec = _seg41j()
    b = rec["bodies"]["4"]
    from mimo_bridge.observer_record import read_header
    h = dict(read_header(b["headers"][0]))
    h["model_fingerprint"] = "0" * 64
    with pytest.raises(C.Refused, match="fingerprint"):
        C.BodyModel(b["model"], h)


def test_the_october_17_command_reproduces_and_its_records_stay_unchanged(tmp_path):
    """The joint values reproduce exactly; build.py's camera (0.20 m, no pale overlay) still draws the October 17 hand
    pictures; the drawing (since the operator's review of Oct 19, 2026: cameras fitted to the hand, the asked-for shape
    pale over each sample, the whole body from three sides) leaves the records unchanged."""
    from PIL import Image
    from mimo_bridge.tools.movement_viewer import render as R
    rec = _seg41j()
    ev = json.loads((OCT17 / "evidence.json").read_text(encoding="utf-8"))
    files = [c[1] for c in rec["bodies"]["4"]["chunks"] if 481900 <= c[3] <= 482300]
    before = {p: _sha(p) for p in files}
    man = R.render_command(rec["id"], 4, 482110, out_root=str(tmp_path), recording=rec)
    assert man["start_step"] == ev["start_step"] and man["last_step"] == ev["end_step"]
    data, (a, z), _, _ = R.find_command(rec, 4, 482110)
    model = C.BodyModel(rec["bodies"]["4"]["model"], data["headers"][0])
    q = np.asarray(data["qpos"])
    adr = np.asarray(ev["qpos_addresses"])
    assert list(model.qadr) == ev["qpos_addresses"]
    assert np.array_equal(q[a - 1][adr], np.asarray(ev["start_joint_rad"]))
    assert np.array_equal(q[z][adr], np.asarray(ev["actual_joint_rad"]))
    goal = np.asarray(data["goals"][int(data["chunk"][a])][int(data["row"][a])], dtype=np.float64)
    assert np.array_equal(goal, np.asarray(ev["target_joint_rad"]))
    q_target = q[a - 1].copy()
    q_target[adr] = goal
    classic = tmp_path / "classic"
    classic.mkdir()
    hands = R._Hands(rec["bodies"]["4"]["model"])
    try:
        hands.draw("left", [("start", q[a - 1]), ("target", q_target)] +
                   [("t%03d" % ((r - a + 1) * 50), q[r]) for r in range(a, z + 1)], str(classic), distance=0.20)
    finally:
        hands.close()
    names = {"start": "start", "target": "target", "actual": "t250", "t50": "t050", "t100": "t100", "t150": "t150",
             "t200": "t200"}
    for view in ("front", "side"):
        for old, new in names.items():
            x = np.asarray(Image.open(OCT17 / ("left_%s_%s.png" % (view, old))).convert("RGB")).astype(int)
            y = np.asarray(Image.open(classic / ("hand_left_%s_%s.png" % (view, new))).convert("RGB")).astype(int)
            assert int((np.abs(x - y).max(-1) > EDGE_LEVELS).sum()) <= EDGE_PIXELS, (view, old)
    # 2 hands x 2 views x 7 states, and the whole body from 3 sides x 7 states
    assert len(man["images"]) == 28 + 21 and man["has_start"] and man["has_hand_goal"]
    assert sorted({v["view"] for k, v in man["images"].items() if k.startswith("body_")}) == ["front", "left", "right"]
    assert all(v["ghost"] == v["state"][1:].isdigit() for k, v in man["images"].items() if k.startswith("hand_"))
    assert {p: _sha(p) for p in files} == before, "drawing changed a saved record"
    again = R.render_command(rec["id"], 4, 482110, out_root=str(tmp_path), recording=rec)
    assert again["reused"] is True and again["key"] == man["key"]
    with pytest.raises(C.Refused, match="no command starts"):
        R.render_command(rec["id"], 4, 482111, out_root=str(tmp_path), recording=rec)


def test_the_routes_refuse_what_is_not_theirs():
    from mimo_bridge.tools.movement_viewer import api as A
    for name in ("../../secret.png", "hand_left_front_start.jpg", "body_t250.png/../x"):
        with pytest.raises(ValueError):
            A.image_path({"key": ["0" * 20], "name": [name]})
    for key in ("../x", "abc", "0" * 19):
        with pytest.raises(ValueError):
            A.image_path({"key": [key], "name": ["body_front_start.png"]})
    for name in ("body_start.png", "body_top_t250.png"):           # the drawing of Oct 18 and an unknown view
        with pytest.raises(ValueError):
            A.image_path({"key": ["0" * 20], "name": [name]})
    assert A.image_path({"key": ["0" * 20], "name": ["body_right_t250.png"]}).name == "body_right_t250.png"
    with pytest.raises(ValueError):
        A.render_request({"recording": "not_an_id", "body": 0, "step": 1})
    with pytest.raises(ValueError):
        A.render_request({"probe": "not_an_id", "body": 0, "goal": "curl_witness", "arm": "direct"})
    assert A.image_path({"key": ["0" * 20], "name": ["hand_right_side_t050.png"]}).name == "hand_right_side_t050.png"
    for key in ("../x", "f" * 21):
        with pytest.raises(ValueError):
            A.export_path({"key": [key]})
    with pytest.raises(ValueError, match="no drawing"):
        A.export_path({"key": ["0" * 20]})


def test_only_the_15_handles_the_controller_chases_are_scored():
    """The independent review of Oct 18, 2026 (finding 2): the fingertip rows are replaced by the hand control before
    the decoder reads them, and the eyes' row moves no joint; neither may enter the whole-body score."""
    from mimo_bridge.body_layout import names_for
    names = list(names_for("trunk_v1"))
    keep = C.scored(names)
    assert int(keep.sum()) == 15
    assert not any(keep[names.index(n)] for n in ("gaze", "thumb_L", "pinky_R", "index_L"))
    assert all(keep[names.index(n)] for n in ("wrist_L", "wrist_R", "head", "lower_trunk", "chest", "foot_R"))
    rec = _seg41j()
    b = rec["bodies"]["4"]
    data = C.load_records([c for c in b["chunks"] if 482000 <= c[3] <= 482200])
    model = C.BodyModel(b["model"], data["headers"][0])
    cols = C.measure(data, model, "all")
    for h, n in enumerate(model.names):
        if not keep[h]:
            assert np.isnan(cols["rot_e"][:, h]).all() and np.isnan(cols["pos_e"][:, h]).all(), n
    assert "m_tips_L_e" not in cols and np.isfinite(cols["m_body_turn_e"]).all()


def test_settings_name_the_finger_servo_and_the_arm_stage():
    from mimo_bridge.tools.movement_viewer import sources as S
    assert S.settings_of({})["key"] == "servo_off|arm_off"
    assert S.settings_of({"finger_joint_servo": "on", "arm_stage": "memory"})["label"] == \
        "finger joint servo on, arm stage memory"
    # the second review of Oct 18, 2026: the two flags that replace command rows before the decoder make their own
    # group; the learner settings are carried beside it without splitting the group
    st = S.settings_of({"finger_joint_servo": "on", "parent_relative_targets": "on", "reach_projection": "off",
                        "proprio_token_skip": "1,3", "hand_response_calibration": "C:/x/y/calibration_v1.json"})
    assert st["key"] == "servo_on|arm_off|parent_relative_targets"
    assert st["label"] == "finger joint servo on, parent-relative targets on"
    assert st["other"] == {"proprio_token_skip": "1,3", "proprio_pose": "off", "attention_cap": "off",
                           "hand_response_calibration": "calibration_v1.json"}
    assert S.settings_of({"proprio_pose": "off", "attention_cap": "off", "hand_response_calibration": ""})["other"] == \
        S.settings_of({})["other"]                         # an unnamed setting reads as the trainer's default
    assert S.settings_of({"finger_joint_servo": "on", "parent_relative_targets": "off"})["key"] == "servo_on|arm_off"


def _recordings_with_index():
    from mimo_bridge.tools.movement_viewer import sources as S
    recs = [r for r in S.load().get("recordings") or [] if I.read_meta(r["id"], min(int(b) for b in r["bodies"]))]
    if not recs:
        pytest.skip("no indexed recording on this disk")
    return recs


def test_the_summary_is_given_per_settings_group():
    """The second review of Oct 18, 2026 (finding 1): a summary pooled over runs with different control settings put the
    retracted 26% against 38% comparison back on the panel. Each settings group is summed on its own; the groups add up
    to the pooled count, and a group's numbers equal those of its own recordings alone."""
    recs = _recordings_with_index()
    out = I.series(recs, "all", "fingers_L", "planner")
    groups = out["summary_by_settings"]
    keys = [g["settings_key"] for g in groups]
    assert len(keys) == len(set(keys)) == len({(r.get("settings") or {}).get("key") for r in recs
                                                if any(s["recording"] == r["id"] for s in out["series"])})
    assert sum(g["summary"]["commands"] for g in groups) == out["summary"]["commands"]
    g = max(groups, key=lambda g: g["summary"]["commands"])
    alone = [r for r in recs if (r.get("settings") or {}).get("key") == g["settings_key"]]
    assert I.summary(I.selection(alone, "all", "fingers_L", "planner")) == g["summary"]


def test_observer_events_are_dated_only_from_the_one_digested_run_that_covers_them():
    """The second review of Oct 18, 2026 (finding 2): events were dated on test runs the observer never read, and on the
    first of several runs resumed from one checkpoint. An event gets a date only from the one digested run that covers
    its model step, and only when that run is shown; otherwise it says why it has none."""
    from mimo_bridge.tools.movement_viewer import api as A
    recs = _recordings_with_index()
    dev = A._read(Path(A.S.REPO) / "Saved/mimo_bridge/observer/H84_fresh_drivers_20260918/development.json")
    if not dev:
        pytest.skip("no observer record on this disk")
    digested = set(dev.get("runs") or [])
    tests_only = [r for r in recs if r["kind"] == "test"]
    for mk in A.markers(recs):
        covering = {r["run"] for r in recs if r["run"] in digested and r["first_step"] <= mk["step"] <= r["last_step"]}
        if len(covering) == 1:
            assert mk["time"] is not None and mk["run"] in covering, mk
        else:
            assert mk["time"] is None and mk["why"], mk
    for mk in A.markers(tests_only):
        assert mk["time"] is None and mk["why"], mk


def test_a_malformed_date_bound_is_refused_and_the_export_counts_the_held_steps():
    from mimo_bridge.tools.movement_viewer import api as A
    for bad in ("abc", "nan", "inf"):
        with pytest.raises(ValueError, match="t0 must be a time"):
            A._filters({"t0": [bad]})
    assert A._filters({"t0": [""], "t1": ["1792370000"]})[3:5] == (None, 1792370000.0)
    # the second review (finding 7): an automatic drawing draws one sample; the command was held from step to last_step
    assert A.held_steps({"step": 494250, "last_step": 494254, "samples_ms": [250]}) == 5
    assert A.held_steps({"step": 10}) == 1


def test_the_elbow_share_divides_by_the_signed_request():
    """The probe's rule divides the elbow's flexion by the SIGNED executed request; a request toward extension keeps its
    sign (the second review of Oct 18, 2026, latent finding). The trial saves the request's size and the reference
    command's share, from which the sign follows."""
    from mimo_bridge.tools.movement_viewer import probes as P
    deg = np.radians(1.0)
    base = dict(goal="elbow_flex_witness", outcomes=dict(controller=dict(qpos=[0.0]), direct=dict(qpos=[-0.5 * deg])))
    # a flexion request of 2 degrees: the reference command flexed 1 degree (the angle fell), share +0.5
    flex = dict(base, witness=dict(qpos=[-deg]), witness_check=dict(requested_executed=2.0, achieved_share=0.5))
    out = P.request_shares(flex, {}, 0)
    assert abs(out["witness"] - 0.5) < 1e-9 and abs(out["direct"] - 0.25) < 1e-9 and out["controller"] == 0.0
    # an extension request of 2 degrees: the reference command extended 1 degree, which is half the request, share +0.5
    ext = dict(base, witness=dict(qpos=[deg]), witness_check=dict(requested_executed=2.0, achieved_share=0.5))
    out = P.request_shares(ext, {}, 0)
    assert abs(out["witness"] - 0.5) < 1e-9 and abs(out["direct"] + 0.25) < 1e-9


def test_probe_runs_off_the_chart_say_why_and_the_unanchored_frozen_start_is_replaced(tmp_path, monkeypatch):
    """The second review of Oct 18, 2026 (findings 3 and 8): the earliest scheduled runs were left out without a word,
    and the frozen start at 483,411 whose planners searched around that checkpoint's own command was drawn as planner
    results. Every run with no point names its reason; a replaced run names the anchored run drawn instead."""
    from mimo_bridge.tools.movement_viewer import probes as P
    monkeypatch.setattr(P, "INDEX_DIR", tmp_path)
    out = P.series("curl_witness")
    if not out["runs"]:
        pytest.skip("no saved probe run on this disk")
    shown = {x["id"]: x for x in out["runs"]}
    assert all(x["why"] for x in out["not_drawn"])
    for x in out["runs"]:
        if x["replaced_by"]:
            y = shown[x["replaced_by"]]
            assert x["mode"] == "frozen start" and not x["own_baseline"]
            assert y["mode"] == "frozen start, search anchored" and y["checkpoint_step"] == x["checkpoint_step"]
    ids = set(shown) | {x["id"] for x in out["not_drawn"]}
    assert ids == {r["id"] for r in P.scan()}                 # every probe run is drawn, replaced or named


def test_the_live_route_keeps_its_own_size_rule(monkeypatch):
    """The second review of Oct 18, 2026: the size cap added for drawing requests also capped /api/live. It applies to
    the drawing route only; /api/live reads its body as before."""
    import io
    import types
    from mimo_bridge.tools import development_dashboard as D
    calls = []
    monkeypatch.setattr(D, "training_state", lambda root, now: {})
    monkeypatch.setattr(D, "live_command", lambda root, training, body: calls.append(body) or {"ok": True})

    def post(path, payload):
        raw = json.dumps(payload).encode("utf-8")
        h = D.Handler.__new__(D.Handler)
        h.headers = {"Host": "127.0.0.1", "Content-Length": str(len(raw))}
        h.path, h.rfile, h.wfile = path, io.BytesIO(raw), io.BytesIO()
        h.server = types.SimpleNamespace(root=str(REPO))
        sent = []
        h.send_response = sent.append
        h.send_error = sent.append
        h.send_header = lambda *a: None
        h.end_headers = lambda: None
        h.do_POST()
        return sent, h.wfile.getvalue()
    sent, _ = post("/api/live", {"pad": "x" * 20000})
    assert sent == [200] and len(calls) == 1
    sent, body = post("/api/movement/render", {"pad": "x" * 20000})
    assert sent == [400] and b"Request too large" in body


def test_a_drawing_refuses_records_from_a_chunk_of_another_model(tmp_path, monkeypatch):
    """The second review of Oct 18, 2026 (latent): the model was checked against the first loaded chunk only. Every
    record a drawing uses (the start, the command, the next command) must come from a chunk of the same model."""
    from mimo_bridge.tools.movement_viewer import render as R
    rec = _seg41j()
    cols = I.body_cols(rec["id"], 4)
    if cols is None:
        pytest.skip("seg41j is not indexed on this disk")
    chunks = rec["bodies"]["4"]["chunks"]
    firsts = [int(c[3]) for c in chunks]
    # a command that starts at the first record of a chunk (not the first chunk): its start record is in the chunk
    # before, which the drawing loads first
    step = next(int(s) for s in cols["step"] if int(s) in firsts[1:])
    real = C.load_records

    def mixed(use):
        data = real(use)
        data["headers"] = [dict(h) for h in data["headers"]]
        data["headers"][0]["model_fingerprint"] = "another model"
        return data
    monkeypatch.setattr(C, "load_records", mixed)
    with pytest.raises(C.Refused, match="another model"):
        R.render_command(rec["id"], 4, step, out_root=str(tmp_path), recording=rec)


def test_the_probe_share_rule_reproduces_the_probes_own_witness_check():
    """probes.request_shares on each witness gives the achieved_share the probe stored (its own WITNESS_RULE)."""
    from mimo_bridge.tools.movement_viewer import probes as P
    runs = [r for r in P.scan() if r["mode"] in ("scheduled", "normal mode") and r["complete"]]
    if not runs:
        pytest.skip("no saved probe run on this disk")
    run = runs[-1]
    geo = json.loads(Path(run["folder"], "hand_geometry.json").read_text(encoding="utf-8"))
    eq = P._elbow_qpos()
    compared = 0
    for tr in P.trials(run):
        stored = (tr.get("witness_check") or {}).get("achieved_share")
        mine = P.request_shares(tr, geo, eq)
        if stored is None or mine["witness"] is None:
            continue
        assert abs(mine["witness"] - stored) < 1e-6, (tr["goal"], tr["body"])
        assert mine["controller"] == 0.0
        compared += 1
    assert compared >= 30


def test_a_trial_that_does_not_count_carries_the_probes_own_two_shares(tmp_path, monkeypatch):
    """The independent review of Oct 19, 2026 (round 2, finding 3): a trial is left out when the request did not move
    the part past the starting line OR the same change reversed moved it as far; the page names the condition that
    failed from the probe's own two shares, which every trial row now carries."""
    from mimo_bridge.tools.movement_viewer import probes as P
    monkeypatch.setattr(P, "INDEX_DIR", tmp_path)
    runs = [r for r in P.scan() if r["mode"] == "scheduled" and r["complete"]]
    if not runs:
        pytest.skip("no saved probe run on this disk")
    rows = [r for run in runs[-3:] for r in P.shares_of_run(run) if r["checked"]]
    assert rows
    for r in rows:
        if r["own_share"] is not None and r["mirror_share"] is not None:
            assert r["informative"] == (r["own_share"] > 0 and r["own_share"] > r["mirror_share"]), r


def test_a_wrist_reach_try_is_drawn_with_a_close_up_and_a_mark(tmp_path, monkeypatch):
    """The independent review of Oct 19, 2026 (round 3, finding 2): the 1 cm reach could not be seen on the whole body
    and its mark was drawn in the arm's own color. A wrist-reach try carries a close-up of the left wrist."""
    from mimo_bridge.tools.movement_viewer import api as A
    from mimo_bridge.tools.movement_viewer import probes as P
    monkeypatch.setattr(P, "INDEX_DIR", tmp_path)
    runs = [r for r in P.scan() if r["mode"] == "scheduled" and r["complete"]]
    if not runs:
        pytest.skip("no saved probe run on this disk")
    run = runs[-1]
    body = next(r["body"] for r in P.shares_of_run(run) if r["goal"] == "wrist_reach_witness")
    man = P.render_trial(run["id"], body, "wrist_reach_witness", "direct", out_root=str(tmp_path / "pics"))
    assert man["body_views"][-1] == "wrist"
    assert {"body_wrist_target.png", "body_wrist_t250.png"} <= set(man["images"])
    # round 4: the starting line's place is marked too, and the move splits into the controller's part and the test's
    wm = man["wrist_marks"]
    assert wm["line_to_test_cm"] is not None and wm["got_to_test_cm"] is not None
    if wm["start_to_test_cm"] is not None:
        assert wm["start_to_test_cm"] <= wm["start_to_line_cm"] + wm["line_to_test_cm"] + 1e-6   # the triangle rule
    assert A.image_path({"key": ["0" * 20], "name": ["body_wrist_t250.png"]}).name == "body_wrist_t250.png"
    curl = P.render_trial(run["id"], body, "curl_witness", "direct", out_root=str(tmp_path / "pics"))
    assert "wrist" not in curl["body_views"]


def test_the_controller_is_shown_as_the_zero_point_where_its_own_endpoint_is_the_baseline(tmp_path, monkeypatch):
    """A probe run measured from its own controller endpoint carries no controller result (its share is 0 by
    definition); a frozen start at another checkpoint keeps its measured controller share, scored from the endpoint of
    the reference checkpoint's controller command (zero_point_step = the checkpoint that made the requested poses)."""
    from mimo_bridge.tools.movement_viewer import probes as P
    monkeypatch.setattr(P, "INDEX_DIR", tmp_path)          # the cached shares go to a scratch folder
    out = P.series("curl_witness")
    if not out["runs"]:
        pytest.skip("no saved probe run on this disk")
    for x in out["runs"]:
        assert x["own_baseline"] == (x["zero_point_step"] == x["checkpoint_step"]), x["id"]
        if x["own_baseline"]:
            assert "controller" not in x["tries"], x["id"]
        else:
            assert x["mode"].startswith("frozen") and x["zero_point_step"] == x["goals_from"], x["id"]
    assert any(x["own_baseline"] for x in out["runs"])


def test_an_automatic_drawing_is_the_smaller_set_under_its_own_key(tmp_path):
    from mimo_bridge.tools.movement_viewer import render as R
    rec = _seg41j()
    lite = R.render_command(rec["id"], 4, 482110, out_root=str(tmp_path), recording=rec, lite=True)
    assert lite["key"] == R.key_for(rec["id"], 4, 482110, True) != R.key_for(rec["id"], 4, 482110)
    assert len(lite["images"]) == 9 and lite["samples_ms"] == [250] and lite["sample_steps"] == [482114]
    assert sorted(n for n in lite["images"] if n.startswith("body_")) == [
        "body_front_start.png", "body_front_t250.png", "body_front_target.png"]
    # the per-sample numbers cover every model step of the command even when only the last is drawn
    assert lite["all_samples_ms"] == [50, 100, 150, 200, 250]


def test_the_asked_for_body_turns_each_targeted_part_and_carries_the_rest():
    """render._Body.requested_frames: a part with a target points exactly as the target says, turned about its first
    joint's anchor (the anchor stays where the parent chain puts it); a part without one keeps its position and turn
    relative to its parent; parts outside the body are not moved."""
    from mimo_bridge.tools.movement_viewer import render as R
    rec = _seg41j()
    data, (a, z), _, _ = R.find_command(rec, 4, 482110)
    header = data["headers"][int(data["chunk"][a])]
    body = R._Body(rec["bodies"]["4"]["model"], header)
    try:
        m = body.m
        q0 = np.asarray(data["qpos"])[a - 1]
        upper, lower, hand = m.body("left_upper_arm").id, m.body("left_lower_arm").id, m.body("left_hand").id
        turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])      # 90 degrees about the vertical
        xp, xm, gp, gm = body.requested_frames(q0, {upper: turn})
        assert np.allclose(gm[upper], turn)
        js = [j for j in range(m.njnt) if int(m.jnt_bodyid[j]) == upper]
        body.place(q0)
        anchor = np.array(body.d.xanchor[js[0]])
        loc = xm[upper].T @ (anchor - xp[upper])
        assert np.allclose(gp[upper] + gm[upper] @ loc, anchor)                    # the shoulder stays put
        for child, parent in ((lower, upper), (hand, lower)):
            assert np.allclose(gm[parent].T @ gm[child], xm[parent].T @ xm[child])  # the turn relative to the parent
            assert np.allclose(gm[parent].T @ (gp[child] - gp[parent]), xm[parent].T @ (xp[child] - xp[parent]))
        right = m.body("right_upper_arm").id
        assert np.allclose(gm[right], xm[right]) and np.allclose(gp[right], xp[right])   # the other arm is untouched
        not_own = [b for b in range(m.nbody) if b not in body.own_set]
        assert np.allclose(gp[not_own], xp[not_own]) and np.allclose(gm[not_own], xm[not_own])
        body.set_frame(q0)
        assert body.posture() in ("sitting or standing up", "head down", "lying on its back", "lying on its front",
                                  "lying on its right side", "lying on its left side",
                                  "leaning, between lying and sitting up")
    finally:
        body.close()


def test_each_drawing_saves_every_part_and_finger_joint_at_every_sample(tmp_path):
    """The page's per-part and per-finger bars read these lists; the last sample is the command's end, so each list's
    last value equals the end value of the scores (commands.measure), and the fingers' joint lists give the hand's
    finger measure."""
    from mimo_bridge.tools.movement_viewer import render as R
    rec = _seg41j()
    man = R.render_command(rec["id"], 4, 482110, out_root=str(tmp_path), recording=rec, lite=True)
    keep = [h for h in man["handles"] if h["powered"] and h["scored"]]
    assert keep and all(len(h["turn_deg"]["samples"]) == 5 for h in keep)
    for h in keep:
        assert abs(h["turn_deg"]["samples"][-1] - h["turn_deg"]["end"]) < 1e-3, h["name"]
        if h["position_row"]:
            assert abs(h["position_cm"]["samples"][-1] - h["position_cm"]["end"]) < 1e-3, h["name"]
    for side, mk in (("left", "fingers_L"), ("right", "fingers_R")):
        joints = [j for f in man["fingers"][side] for j in f["joints"]]
        assert len(joints) == 22 and [f["finger"] for f in man["fingers"][side]] == [
            "thumb", "index finger", "middle finger", "ring finger", "little finger"]
        start = np.mean([abs(j["start"] - j["target"]) for j in joints])
        end = np.mean([abs(j["samples"][-1] - j["target"]) for j in joints])
        assert abs(start - man["measures"][mk]["start"]) < 1e-2 and abs(end - man["measures"][mk]["end"]) < 1e-2


def test_the_asked_for_places_are_measured_on_the_drawing_they_belong_to(tmp_path):
    """The independent review of Oct 19, 2026 (blocking 1): the asked-for body follows the asked-for directions only, so
    each drawing states how far its parts are from their asked-for places; the start distance is the score's own."""
    from mimo_bridge.tools.movement_viewer import render as R
    rec = _seg41j()
    man = R.render_command(rec["id"], 4, 482110, out_root=str(tmp_path), recording=rec, lite=True)
    rows = {h["name"]: h for h in man["handles"] if h["powered"] and h["scored"] and h["position_row"]}
    assert rows and {x["name"] for x in man["asked_for_places"]} == set(rows)
    for x in man["asked_for_places"]:
        assert abs(x["start_cm"] - rows[x["name"]]["position_cm"]["start"]) < 1e-3, x
        assert x["asked_for_body_cm"] is not None and x["asked_for_body_cm"] >= 0
    assert man["body_weight_n"] > 0 and man["posture_from"] == "start"


def test_a_pale_part_where_the_drawn_part_lies_is_left_out():
    """The independent review of Oct 19, 2026 (blocking 2): two surfaces in one place draw as interleaved patches."""
    from mimo_bridge.tools.movement_viewer import render as R
    eye = np.eye(3)
    turn = np.array([[np.cos(0.05), -np.sin(0.05), 0], [np.sin(0.05), np.cos(0.05), 0], [0, 0, 1.0]])   # 2.9 degrees
    assert R._same_place([0, 0, 0], eye, [0.001, 0, 0], eye)
    assert not R._same_place([0, 0, 0], eye, [0.002, 0, 0], eye)
    assert not R._same_place([0, 0, 0], eye, [0, 0, 0], turn)
    # the whole body's wider tolerance (4 mm, 3 degrees): 2 mm and 2.9 degrees count as one place there
    assert R._same_place([0, 0, 0], eye, [0.002, 0, 0], turn, within_m=R.BODY_SAME_M, within_deg=R.BODY_SAME_DEG)


def test_every_run_named_in_words_has_its_registry_row():
    """sources.RUN_WORDS names the test-registry row each description comes from; every named row exists and its title
    describes the same run (its id names the run's label or its own row is the run's)."""
    from mimo_bridge.tools.movement_viewer import sources as S
    reg = json.loads((REPO / "docs/gain_of_function_v4/test_registry.json").read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in (reg.get("rows") if isinstance(reg, dict) else reg)}
    for label, (words, row) in S.RUN_WORDS.items():
        assert words
        if row is not None:
            assert row in rows, (label, row)
    assert S.about_of({"label": "training 20260918_seg41c_0841", "kind": "training"})["name"] == "training segment 41c"
    # three runs of Oct 6 share segment 41c: each name carries its first saved record's clock time (round 2, finding 1)
    names = {S.about_of(r)["name"] for r in (S.load().get("recordings") or []) if r.get("kind") == "training"}
    assert len(names) == len([r for r in (S.load().get("recordings") or []) if r.get("kind") == "training"])
    assert S.about_of({"label": "POSE_A_off_494025", "kind": "test"})["row"] == "pose_readout_comparison_A_off_494025_run"


def test_each_bin_of_a_run_carries_the_share_of_the_way_its_commands_got():
    """index.series: a bin's `closed` is (sum of start values - sum of end values) / sum of start values over its
    commands with both, the same sums as a run's summary."""
    recs = _recordings_with_index()
    out = I.series(recs[:1], "all", "fingers_L", "all")
    s = out["series"][0]
    sel = I.selection(recs[:1], "all", "fingers_L", "all")
    b = s["bins"][0]
    keep = (sel["time"] >= b["t0"]) & (sel["time"] <= b["t1"])
    both = keep & np.isfinite(sel["s"]) & np.isfinite(sel["e"])
    want = (sel["s"][both].sum() - sel["e"][both].sum()) / sel["s"][both].sum()
    assert abs(b["closed"] - want) < 1e-3 and b["with_start"] == int(both.sum())


def test_a_replace_waits_for_a_reader_to_close_the_file(tmp_path):
    """The viewer service ended on Oct 18, 2026, 6:35 PM PDT: replacing service_status.json failed with PermissionError
    (WinError 5) while the dashboard had the file open. replace_file retries until the reader closes it."""
    import threading
    from mimo_bridge.tools.movement_viewer import part_path, replace_file
    dst = tmp_path / "service_status.json"
    dst.write_text("old", encoding="utf-8")
    if os.name == "nt":                                   # the failure itself: a plain replace while a reader holds dst
        plain = tmp_path / "plain.part"
        plain.write_text("x", encoding="utf-8")
        with open(dst, encoding="utf-8"):
            with pytest.raises(PermissionError):
                os.replace(plain, dst)
    tmp = part_path(dst)
    assert tmp.endswith(".%d.part" % os.getpid())
    Path(tmp).write_text("new", encoding="utf-8")
    reader = open(dst, encoding="utf-8")
    threading.Timer(0.3, reader.close).start()          # the reader closes the file 0.3 s later
    replace_file(tmp, dst)
    assert dst.read_text(encoding="utf-8") == "new"


def test_the_service_keeps_running_when_its_status_cannot_be_written(monkeypatch):
    from mimo_bridge.tools.movement_viewer import service as SV
    calls = []

    def refused(path, value):
        calls.append(str(path))
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(SV, "_atomic", refused)
    monkeypatch.setattr(SV, "_own_process_settings", lambda: {})       # leaves the test process's priority alone
    monkeypatch.setattr(SV.Service, "paused", lambda self: True)        # no drawing, no background pass
    s = SV.Service()
    s.run(once=True)                                      # before the repair the PermissionError ended the service here
    assert calls and "PermissionError" in s.status["errors"][-1]["text"]


def test_the_improvement_line_is_shown_beside_the_run_it_names(tmp_path, monkeypatch):
    """The proxy decision of Oct 18, 2026: the fixed-start probe's improvement line (each planner arm against the same
    arm at 494,025, DIRECT deciding) is read from the gain object of a comparison file or of a scheduled round's result,
    and shown beside the run that file names; an invalid comparison keeps its reasons and no line."""
    from mimo_bridge.tools.movement_viewer import probes as P
    run = tmp_path / "anchored_483411"
    run.mkdir()
    (tmp_path / "anchored_483411_gain").mkdir()
    gain = dict(rule_version="movement-viewer-fixed-probe-decision-20261018", status="valid", invalid_reasons=[],
                primary_arm="direct", line_met=False, reference=dict(checkpoint_step=494025),
                this=dict(checkpoint_step=483411), label="Whole-arm checkpoint performance",
                arms=dict(direct=dict(difference_mean=0.0199, paired_bootstrap95=[-0.0214, 0.0671], line_met=False,
                                      wrist_goal_error_cm=dict(difference_mean=0.115, paired_bootstrap95=[0.01, 0.24],
                                                               interval_reading="worsening")),
                          synergy=dict(difference_mean=0.0116, paired_bootstrap95=[-0.0195, 0.0419], line_met=False)))
    (tmp_path / "anchored_483411_gain" / "gain_to_494025.json").write_text(
        json.dumps(dict(this_run=str(run), made_at="Oct 18", gain=gain)), encoding="utf-8")
    sched = tmp_path / "fixed_start_body_and_synergy_506025"       # a scheduled round's result: its own folder
    sched.mkdir()
    (sched / "result.json").write_text(json.dumps(dict(gain=dict(gain, status="invalid", line_met=None,
                                                                  invalid_reasons=["a start differs"]))), encoding="utf-8")
    monkeypatch.setattr(P, "GAIN_GLOBS", (str(tmp_path / "*" / "gain_*.json"),
                                          str(tmp_path / "fixed_start_body_and_synergy_*" / "result.json")))
    monkeypatch.setattr(P, "_GAINS", {})
    found = P.gains()
    assert len(found) == 2
    g = found[P._folder_key(run)]
    assert g["status"] == "valid" and g["line_met"] is False and g["reference_step"] == 494025
    assert g["arms"]["direct"]["interval"] == [-0.0214, 0.0671] and g["arms"]["direct"]["wrist_cm"]["reading"] == "worsening"
    s = found[P._folder_key(sched)]
    assert s["status"] == "invalid" and s["invalid_reasons"] == ["a start differs"] and s["line_met"] is None
