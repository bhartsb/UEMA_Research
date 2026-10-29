# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The whole-body capacity check (docs/gain_of_function_v4/wholebody_capacity_check_2026-10-11/capacity_check.py), the
tool of registry row wholebody_capacity_check_484675.

Every test but the last builds no world: the readers of the record (the hold, rise and roll lines, the failed transition,
the audit's attribution rules and their confirmation, the readings from counts, the verdict, the demand reader's own
check, the support polygon), the companion reader on a small archive written by the trainer's own writer
(episode_archive), and the run command's refusals. The last test builds body 0's saved world of 484,675 and runs half a
second of the hold through the live route; it runs only with WHOLEBODY_CAPACITY_WORLD=1, under the memory guard and with
no trainer running (conftest).
"""
import gzip
import hashlib
import importlib.util
import io
import math
import os
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[4]
MODULE = REPO / "docs" / "gain_of_function_v4" / "wholebody_capacity_check_2026-10-11" / "capacity_check.py"


def _load():
    spec = importlib.util.spec_from_file_location("wholebody_capacity_check", str(MODULE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cc = _load()


def test_import_leaves_the_visible_cards_alone():
    before = os.environ.get("CUDA_VISIBLE_DEVICES")
    _load()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == before


def test_every_planted_fault_is_named_by_its_reader():
    result = cc.run_validation()
    assert result["failed"] == 0, [f["name"] for f in result["failures"]]
    assert result["cases"] >= 50


# ------------------------------------------------------------------------------------------ small readers
def test_longest_run_and_wrap():
    assert cc.longest_run([0, 1, 1, 0, 1, 1, 1, 0]) == (3, 4)
    assert cc.longest_run([]) == (0, -1)
    assert cc.wrap_deg(190.0) == pytest.approx(-170.0)
    assert cc.wrap_deg(-181.0) == pytest.approx(179.0)


def test_polygon_margin_inside_outside_and_degenerate():
    hull = cc.convex_hull([[0, 0], [2, 0], [2, 1], [0, 1], [1, 0.5], [1, 0]])
    assert len(hull) == 4
    assert cc.polygon_margin([1.0, 0.5], hull) == pytest.approx(0.5)
    assert cc.polygon_margin([3.0, 0.5], hull) == pytest.approx(-1.0)
    assert cc.polygon_margin([0.0, 0.0], []) == -math.inf
    assert cc.polygon_margin([0.0, 1.0], [[0.0, 0.0]]) == pytest.approx(-1.0)


def test_body_groups_name_every_support_part():
    assert cc.body_group("left_hand") == "hand_L"
    assert cc.body_group("right_ffdistal") == "hand_R"
    assert cc.body_group("left_lower_leg") == "knee_L"
    assert cc.body_group("right_upper_leg") == "thigh_R"
    assert cc.body_group("left_big_toe") == "foot_L"
    assert cc.body_group("chest") == "trunk"
    assert cc.body_group("right_eye") == "head"
    assert cc.body_group("mobile_el_c") is None


def test_share_is_zero_when_nothing_touches():
    fn = np.zeros((3, len(cc.GROUPS)))
    fn[1, cc.GROUPS.index("hand_L")] = 1.0
    fn[1, cc.GROUPS.index("trunk")] = 3.0
    np.testing.assert_allclose(cc.share(fn, ("hand_L", "hand_R")), [0.0, 0.25, 0.0])


# ------------------------------------------------------------------------------------------ the lines
PLACED = dict(chest_z=0.20, pelvis_z=0.18, hand_xyz=[[0.1, 0.05, 0.0], [0.1, -0.05, 0.0]])


def test_hold_line_passes_a_quiet_hold_and_fails_each_condition():
    dt = 0.01
    assert cc.hold_line(cc.synthetic_record(1000), PLACED, dt)["passed"]
    r = cc.synthetic_record(1000)
    r["pelvis_z"][700:] = 0.14                 # 78% of 0.18
    line = cc.hold_line(r, PLACED, dt)
    assert not line["passed"] and line["held_share"] == pytest.approx(0.7)
    r = cc.synthetic_record(1000)
    r["support_fn"][:, cc.GROUPS.index("head")] = 2.0
    assert not cc.hold_line(r, PLACED, dt)["passed"]
    r = cc.synthetic_record(1000)
    r["support_fn"][:60, cc.GROUPS.index("hand_R")] = 0.0      # 94% contact
    line = cc.hold_line(r, PLACED, dt)
    assert not line["passed"] and line["contact_share"]["hand_R"] == pytest.approx(0.94)
    r = cc.synthetic_record(1000)
    r["support_fn"][:50, cc.GROUPS.index("hand_R")] = 0.0      # 95% contact holds the line
    assert cc.hold_line(r, PLACED, dt)["passed"]


def test_hold_failed_transition_skips_a_recovered_dip():
    dt = 0.01
    r = cc.synthetic_record(1000)
    r["chest_z"][100:120] = 0.10               # recovers within 0.5 s
    r["chest_z"][600:] = 0.10                  # does not
    line = cc.hold_line(r, PLACED, dt)
    t, how = cc.failed_transition("hold", line, dt, cc.contact_loss_steps(r))
    assert t == 600 and "did not recover" in how


def test_roll_line_dwell_and_model_step_reading():
    dt = 0.01
    r = cc.synthetic_record(300)
    r["start_roll_deg"] = 10.0
    r["roll_deg"][100:160] = 165.0             # 155 degrees from the start, 0.6 s
    r["front_up"][100:160] = -0.9
    line = cc.roll_line(r, dt)
    assert line["passed"] and line["dwell_s"] == pytest.approx(0.6)
    assert line["dwell_model_observations_s"] == pytest.approx(0.6)
    assert line["peak_signed_change_deg"] == pytest.approx(155.0)
    r["support_fn"][130, :] = 0.0              # one step off the floor breaks the dwell into 0.3 s runs
    assert not cc.roll_line(r, dt)["passed"]


def test_rise_progress_and_failed_transition():
    dt = 0.01
    r = cc.synthetic_record(400)
    r["chest_z"][:] = 0.05
    r["chest_z"][100:140] = 0.17
    r["chest_z"][140:] = 0.05
    line = cc.rise_line(r, 0.20, dt)
    assert not line["passed"] and line["peak_progress"] == pytest.approx(1.0)
    t, how = cc.failed_transition("rise", line, dt)
    assert t == 140
    r["chest_z"][140:] = 0.17                  # never falls: the peak is the stall point
    r["support_fn"][:, cc.GROUPS.index("trunk")] = 1.0
    line = cc.rise_line(r, 0.20, dt)
    t, how = cc.failed_transition("rise", line, dt)
    assert "stalled" in how


def test_search_score_orders_attainment_first():
    dt = 0.01
    good = cc.synthetic_record(300)
    good["chest_z"][:] = 0.17
    bad = cc.synthetic_record(300)
    bad["chest_z"][:] = 0.10
    assert cc.search_score("rise", cc.rise_line(good, 0.20, dt)) > 2.0
    assert cc.search_score("rise", cc.rise_line(bad, 0.20, dt)) < 2.0


# ------------------------------------------------------------------------------------------ attribution
def test_needed_joint_moving_its_way_is_not_read():
    meta, dt = cc.synthetic_meta(), 0.01
    r = cc.synthetic_record()
    r["q"][:, 0] = np.linspace(0.0, 0.2, 100)          # the elbow extends 2.9 degrees in the window
    att = cc.attribute(cc.finish_synthetic(r), meta, 80, {"robot:left_elbow": +1}, dt)
    assert att["needed_joints"] == []


def _saturated_elbow():
    r = cc.synthetic_record()
    r["q"][55:, 0] = np.linspace(0.0, -0.10, 45)
    r["demand"][:, 0] = 1.2
    r["handed"][:, 0] = 0.83
    r["act_force"][:, 0] = 0.83 * 0.6
    return cc.finish_synthetic(r)


def test_force_limited_needs_both_confirmations():
    meta, dt = cc.synthetic_meta(), 0.01
    att = cc.attribute(_saturated_elbow(), meta, 80, {"robot:left_elbow": +1}, dt)
    assert cc.reading_from_counts(att, cc.confirm_force_limited(att, att, att))["reading"] == "force_limited"
    assert (cc.reading_from_counts(att, cc.confirm_force_limited(att, None, att))["reading"]
            == "force_saturated_unconfirmed")
    assert (cc.reading_from_counts(att, cc.confirm_force_limited(att, att, None))["reading"]
            == "force_saturated_unconfirmed")


def test_saturation_at_a_range_end_is_joint_range_not_force():
    meta, dt = cc.synthetic_meta(), 0.01
    r = _saturated_elbow()
    r["to_hi"][:, 0] = math.radians(0.2)
    att = cc.attribute(r, meta, 80, {"robot:left_elbow": +1}, dt)
    assert att["needed_joints"][0]["categories"] == ["joint_range"]


def test_one_category_per_needed_joint_and_counts():
    meta, dt = cc.synthetic_meta(), 0.01
    r = cc.synthetic_record()
    r["demand"][:, 0] = 0.3
    r["demand"][:, 1] = 2.0                    # the wrist past its rail scales the elbow
    r["handed"][:, 0] = 0.3 * 0.57 / 2.0
    r["handed"][:, 1] = 0.57
    r["demand"][:, 3] = -0.4                   # the knee driven against its needed direction
    r["handed"][:, 3] = -0.4
    att = cc.attribute(cc.finish_synthetic(r), meta, 80, {"robot:left_elbow": +1, "robot:left_knee": +1}, dt)
    assert all(len(j["categories"]) == 1 for j in att["needed_joints"])
    assert att["counts"] == {"rescaled_by_other": 1, "wrong_command": 1}
    assert cc.reading_from_counts(att)["reading"] == "mixed"


def test_the_binding_joint_is_named_at_the_minimum_factor_and_never_the_needed_joint():
    meta, dt = cc.synthetic_meta(), 0.01
    r = cc.synthetic_record()
    r["demand"][:, 0] = 0.3
    r["handed"][:, 0] = 0.3
    r["demand"][70, 1] = 1.14                  # the wrist twice its rail on one step only
    r["handed"][70, 1] = 0.57
    r["handed"][70, 0] = 0.15
    att = cc.attribute(cc.finish_synthetic(r), meta, 80, {"robot:left_elbow": +1}, dt)
    j = att["needed_joints"][0]
    assert j["categories"] == ["rescaled_by_other"] and j["rescale_set_by"] == "act:left_wrist_flexion"
    assert j["rescale_set_by_ratio"] == pytest.approx(2.0)


def test_stage_reading_texts_follow_the_row():
    assert cc.stage_reading(True, False)["reading"] == "possible"
    assert "731,356 body model steps" in cc.stage_reading(True, False)["text"]
    assert cc.stage_reading(False, True)["reading"] == "fatigue"
    force = cc.stage_reading(False, False, [dict(reading="force_limited", counts={"force_limited": 1})])
    assert "no force increase" in force["text"]
    assert "not more force" in cc.stage_reading(False, False, [])["text"]
    defect = [k for k, v in cc.READINGS.items() if "a decoder defect" in v or "command-interface defect" in v]
    assert defect == ["world_error_increasing"]
    for k in ("command_interface", "route_reversed", "reversed_body_moved", "reversed_still_world_error_decreasing"):
        assert "not a defect and not a repair trigger" in cc.READINGS[k]


def test_verdict_is_pass_only_when_every_stage_meets_its_line_under_arm_a():
    ok, bad = dict(pass_arm_A=True), dict(pass_arm_A=False)
    stages = ("hold", "rise", "roll")
    assert cc.run_verdict(dict(hold=ok, rise=ok, roll=ok), stages, [])["verdict"] == "pass"
    assert cc.run_verdict(dict(hold=ok, rise=bad, roll=ok), stages, [])["verdict"] == "fail"
    assert cc.run_verdict(dict(hold=ok, rise=ok), stages, [])["verdict"] == "invalid"
    assert cc.run_verdict(dict(hold=ok, rise=ok, roll=ok), stages, ["x"])["verdict"] == "invalid"


def _fresh(err_change, root_turn_deg=0.0, elbow_gap=0.05, chest_turn_deg=0.0, chest_shift_m=0.0):
    r = cc.synthetic_record()
    r["keyframe_target"] = np.full((100, 5), np.nan)
    r["keyframe_origin"] = np.full((100, 5), np.nan)
    r["keyframe_target"][:, 0] = elbow_gap
    r["keyframe_origin"][:, 0] = 0.0
    r["demand"][:, 0] = -0.3
    r["handed"][:, 0] = -0.3
    r["dq"][:, 0] = -0.05
    r["world_err_before"] = np.ones((100, 2))
    r["world_err_after"] = np.ones((100, 2)) + err_change
    r["root_pose"] = np.tile([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0], (100, 1))
    r["formation_root_pose"] = r["root_pose"].copy()
    half = math.radians(root_turn_deg) / 2
    r["root_pose"][:, 3:7] = [math.cos(half), 0.0, 0.0, math.sin(half)]
    r["formation_chest_quat"] = np.tile([1.0, 0.0, 0.0, 0.0], (100, 1))
    half_c = math.radians(chest_turn_deg) / 2
    r["chest_quat"] = np.tile([math.cos(half_c), 0.0, math.sin(half_c), 0.0], (100, 1))
    r["formation_chest_pos"] = np.tile([0.1, 0.0, 0.25], (100, 1))
    r["chest_pos"] = r["formation_chest_pos"] + [chest_shift_m, 0.0, 0.0]
    return cc.finish_synthetic(r)


FMETA = dict(cc.synthetic_meta(), world_error_groups=[dict(name="arm_L", act_ids=[0, 1, 2]),
                                                       dict(name="calf_L", act_ids=[3])])


def test_fresh_decode_categories_replace_reading_c():
    att = cc.attribute(_fresh(+0.2), FMETA, 80, {"robot:left_elbow": +1}, 0.01)
    assert att["needed_joints"][0]["categories"] == ["world_error_increasing"]
    assert cc.reading_from_counts(att)["reading"] == "world_error_increasing"
    att = cc.attribute(_fresh(+0.2, elbow_gap=0.2), FMETA, 80, {"robot:left_elbow": +1}, 0.01)
    assert att["needed_joints"][0]["categories"] == ["reversed_still_other"]
    att = cc.attribute(_fresh(-0.2), FMETA, 80, {"robot:left_elbow": +1}, 0.01)
    assert att["needed_joints"][0]["categories"] == ["reversed_still_world_error_decreasing"]
    att = cc.attribute(_fresh(-0.2, chest_turn_deg=3.0), FMETA, 80, {"robot:left_elbow": +1}, 0.01)
    assert att["needed_joints"][0]["categories"] == ["reversed_body_moved"]
    att = cc.attribute(_fresh(-0.2, root_turn_deg=3.0), FMETA, 80, {"robot:left_elbow": +1}, 0.01)
    assert att["needed_joints"][0]["categories"] == ["reversed_still_world_error_decreasing"]   # the arms hang from the chest
    att = cc.attribute(_fresh(-0.2, chest_shift_m=0.006), FMETA, 80, {"robot:left_elbow": +1}, 0.01)
    assert att["needed_joints"][0]["categories"] == ["reversed_body_moved"]
    calf = dict(FMETA, world_error_groups=[dict(name="calf_L", act_ids=[0])])
    att = cc.attribute(_fresh(-0.2, root_turn_deg=3.0), calf, 80, {"robot:left_elbow": +1}, 0.01)
    assert att["needed_joints"][0]["categories"] == ["reversed_body_moved"]                     # a leg hangs from the root
    assert cc.reading_of_counts({"reversed_body_moved": 2, "reversed_still_world_error_decreasing": 1}) == \
        "command_interface"


def test_command_interface_report_fills_x_y_z_w():
    assert cc.STILL_TRANSLATION_M == pytest.approx(math.radians(2.0) / 8.0)
    rep = cc.command_interface_report(_fresh(-0.2), FMETA)
    assert (rep["X"], rep["Y"], rep["Z"], rep["W"]) == (1.0, 1.0, 1.0, 0.0)
    assert "100.0%" in rep["text"] and "0.0%" in rep["text"]
    assert cc.command_interface_report(cc.synthetic_record(), FMETA) == dict(available=False)


def test_rail_reading_gives_no_reading_of_its_own_on_a_failure():
    att = cc.attribute(_fresh(-0.2), FMETA, 80, {"robot:left_elbow": +1}, 0.01)
    assert cc.rail_reading(dict(passed=False), att, None, {})["reading"] is None
    assert cc.rail_reading(dict(passed=True), None, None, {})["reading"] == "search_coverage"


def test_rail_flag_splits_each_generation_in_half():
    flags = [cc.rail_flag(c) for c in range(8)]
    assert sum(flags) == 4 and flags[:4] == [False, True, False, True] and flags[4:] == [True, False, True, False]


def test_roll_needed_picks_the_side_the_body_turned():
    class Syn:
        side = None
        left_roll_sign = -1.0
        signs = {"L": {p: [("robot:hip_rot1", 1.0)] if p == "twist" else [] for p in cc.ROLL_PARTS},
                 "R": {p: [("robot:hip_rot1", -1.0)] if p == "twist" else [] for p in cc.ROLL_PARTS}}
    r = cc.synthetic_record(10)
    r["roll_deg"][:] = -40.0
    assert cc.roll_needed(r, 5, 0.0, Syn) == {"robot:hip_rot1": 1.0}
    r["roll_deg"][:] = 40.0
    assert cc.roll_needed(r, 5, 0.0, Syn) == {"robot:hip_rot1": -1.0}


def test_demand_check_skips_rescaled_groups_clips_the_eyes_and_finds_a_difference():
    meta = cc.synthetic_meta()
    r = cc.synthetic_record()
    r["demand"][:, 1] = 2.0                    # the wrist past its rail: the arm group is not checked
    r["handed"][:, 1] = 0.57
    r["demand"][:, 3] = 0.4
    r["handed"][:, 3] = 0.4
    r["demand"][:, 4] = -123.7                 # an eye servo, compared after its clip
    r["handed"][:, 4] = -1.0
    dc = cc.demand_check(r, meta)
    assert dc["checked"] == 200 and dc["max_abs_difference"] == 0.0
    r["handed"][50, 3] = 0.41
    assert cc.demand_check(r, meta)["max_abs_difference"] == pytest.approx(0.01)


def test_hold_summary_rules():
    def body(b, a, bb, control):
        return dict(body=b, motors_off_control=dict(passed=control),
                    trials=[dict(arm=arm, strength=s, passed=ok,
                                 reading=None if ok else dict(reading="small_change", counts={"small_change": 3}))
                            for arm, ok in (("A", a), ("B", bb)) for s in (1.0, 0.46)])
    s = cc.hold_summary([body(0, False, False, False), body(1, True, True, False)], (1.0, 0.46))
    assert not s["pass_arm_B"] and s["reading"]["reading"] == "small_change"
    assert s["reading"]["counts"] == {"small_change": 3}
    assert s["per_arm_strength"]["armA_strength1.00"]["bodies_passing"] == [1]
    s = cc.hold_summary([body(0, True, True, True), body(1, True, True, False)], (1.0, 0.46))
    assert s["reading"]["reading"] == "control_also_held" and "[0]" in s["reading"]["text"]


GOOD_IDENTITY = dict(named_commit="abc", tool_dirty=False, named_commit_exists=True, tool_equals_named_commit=True,
                     modules_dirty=[], modules_differing_from_named_commit=[], head="abc", tool_sha256="0" * 64,
                     named_commit_in_head_history=True, source_python_equals_named_commit=True,
                     source_python_uncommitted_or_untracked=[], mimo_head="c" * 40, mimo_dirty=False)


def test_identity_refusals():
    good = dict(GOOD_IDENTITY)
    assert cc.identity_refusal(good) is None
    assert "--commit" in cc.identity_refusal(dict(good, named_commit=None))
    assert "uncommitted" in cc.identity_refusal(dict(good, tool_dirty=True))
    assert "does not exist" in cc.identity_refusal(dict(good, named_commit_exists=False))
    assert "differs" in cc.identity_refusal(dict(good, tool_equals_named_commit=False))
    assert "uncommitted" in cc.identity_refusal(dict(good, modules_dirty=["Source/Python/mimo_bridge/env.py"]))
    assert "not in the history" in cc.identity_refusal(dict(good, named_commit_in_head_history=False))
    assert "untracked" in cc.identity_refusal(dict(good, source_python_uncommitted_or_untracked=["x.py"]))
    assert "Source/Python differs" in cc.identity_refusal(dict(good, source_python_equals_named_commit=False))
    assert "MIMo_private" in cc.identity_refusal(dict(good, mimo_dirty=True))
    assert "MIMo_private" in cc.identity_refusal(dict(good, mimo_head=None))
    why = cc.identity_refusal(dict(good, modules_differing_from_named_commit=["Source/Python/mimo_bridge/env.py"]))
    assert "differ from commit abc" in why and "env.py" in why


def test_decisive_findings_are_named_beside_other_categories():
    assert cc.reading_of_counts({"force_limited": 1, "small_change": 1}) == "force_limited"
    assert cc.reading_of_counts({"world_error_increasing": 1, "rescaled_by_other": 1}) == "world_error_increasing"
    assert cc.reading_of_counts({"force_limited": 1, "world_error_increasing": 1}) == \
        "force_limited+world_error_increasing"
    st = cc.stage_reading(False, False, [dict(reading="force_limited", counts={"force_limited": 1, "small_change": 1},
                                              decisive_joints=[dict(joint="robot:left_elbow")]),
                                         dict(reading="mixed", counts={"small_change": 1, "wrong_command": 2})])
    assert st["reading"] == "force_limited" and st["decisive_joints"] == [dict(joint="robot:left_elbow")]
    assert "no force increase" in st["text"]


def test_live_route_modules_are_hashed():
    ident = cc.tool_identity()
    assert ident["mimo_repository"].endswith("MIMo_private") and "mimo_dirty" in ident
    assert set(ident["module_sha256"]) == set(cc.LIVE_ROUTE_MODULES)
    assert all(v and len(v) == 64 for v in ident["module_sha256"].values())


# ------------------------------------------------------------------------------------------ companion reader
def _fake_companion(tmp_path, step=484675, worlds=8):
    """A companion written by the trainer's own writer (episode_archive), small: 8 worlds and a learner tensor."""
    torch = pytest.importorskip("torch")
    from mimo_bridge.episode_archive import EpisodeArchive, world_blob
    raw = io.BytesIO()
    archive = EpisodeArchive(raw)
    records = [archive.put_world(world_blob(dict(index=i, env_kwargs=dict(tag="_w%d" % i))))
               for i in range(worlds)]
    archive.finish(dict(version="episode_continuation_v1", worlds=records,
                        learner=dict(weights=torch.ones(3)), trainer=dict(step=step),
                        config=dict(n_envs=worlds, body_layout="trunk_v1")))
    src = tmp_path / "preserved" / "companion.pt.gz"
    src.parent.mkdir()
    with gzip.open(src, "wb") as f:
        f.write(raw.getvalue())
    return src, hashlib.sha256(src.read_bytes()).hexdigest()


def test_companion_reader_reads_worlds_without_the_learner(tmp_path):
    src, sha = _fake_companion(tmp_path)
    comp = cc.Companion(tmp_path / "inputs", source=src, sha256=sha)
    try:
        assert comp.step == 484675 and len(comp.world_keys) == 8
        assert comp.world(5) == dict(index=5, env_kwargs=dict(tag="_w5"))
        assert comp.config["body_layout"] == "trunk_v1"
        assert (tmp_path / "inputs" / "companion.pt.gz").exists() and (tmp_path / "inputs" / "companion.pt").exists()
    finally:
        comp.close()


def test_companion_reader_refuses_a_wrong_hash_or_step(tmp_path):
    src, sha = _fake_companion(tmp_path)
    with pytest.raises(cc.InvalidRun, match="sha256"):
        cc.Companion(tmp_path / "inputs", source=src, sha256="0" * 64)
    (tmp_path / "b").mkdir()
    src2, sha2 = _fake_companion(tmp_path / "b", step=1)
    with pytest.raises(cc.InvalidRun, match="step"):
        cc.Companion(tmp_path / "inputs2", source=src2, sha256=sha2)


def test_companion_reader_never_writes_beside_its_source(tmp_path):
    src, sha = _fake_companion(tmp_path)
    with pytest.raises(cc.InvalidRun, match="only ever copied"):
        cc.Companion(src.parent, source=src, sha256=sha)


# ------------------------------------------------------------------------------------------ refusals of the run command
def test_dev_run_never_writes_the_registered_folder(capsys):
    assert cc.main(["run", "--out", str(cc.DEFAULT_OUT), "--dev", "--stages", "hold"]) == 2
    assert "never the registered result folder" in capsys.readouterr().out


def test_reduced_run_needs_dev(tmp_path, capsys):
    assert cc.main(["run", "--out", str(tmp_path / "x"), "--stages", "hold"]) == 2
    assert "must carry --dev" in capsys.readouterr().out
    assert not (tmp_path / "x").exists()


def test_registered_declaration_is_never_overwritten(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cc, "tool_identity", lambda named_commit=None: dict(GOOD_IDENTITY))
    (tmp_path / "declaration.json").write_text("{}", encoding="utf-8")
    assert cc.main(["run", "--out", str(tmp_path), "--commit", "abc"]) == 2
    assert "never overwritten" in capsys.readouterr().out
    assert (tmp_path / "declaration.json").read_text(encoding="utf-8") == "{}"


def test_registered_run_names_its_commit(tmp_path, capsys):
    assert cc.main(["run", "--out", str(tmp_path / "x")]) == 2
    assert "--commit" in capsys.readouterr().out
    assert cc.main(["run", "--out", str(tmp_path / "x"), "--commit", "0" * 40]) == 2
    out = capsys.readouterr().out
    assert "does not exist" in out or "uncommitted" in out
    assert not (tmp_path / "x").exists()


def test_full_specification_lists_each_deviation():
    import argparse
    args = argparse.Namespace(stages="hold,rise,roll", bodies="0,1,2,3,4,5,6,7", hold_seconds=10.0, population=16,
                              generations=6, phases=4, phase_seconds=1.5, rise_starts="2,0t", strengths="1.0,0.46",
                              hold_replay=True)
    assert cc.full_specification(args) == []
    args.population = 4
    assert cc.full_specification(args) == ["population 4 instead of 16"]


# ------------------------------------------------------------------------------------------ the search arm (--search-arm)
def _entry(passed, counts=None, exact=None, decisive=()):
    e = dict(passed=passed, reading=dict(reading=cc.reading_of_counts(counts or {}), counts=counts or {},
                                         decisive_joints=list(decisive)))
    if exact is not None:
        e["replay_exact"] = exact
    return e


def test_search_arms():
    assert cc.search_arms("B") == ("B", "A")
    assert cc.search_arms("A") == ("A",)
    with pytest.raises(ValueError):
        cc.search_arms("C")


def test_start_outcome_under_search_arm_a_runs_no_arm_b():
    fams = {"world_targets": dict(armA=_entry(False, {"small_change": 2}, exact=True)),
            "joint_keyframes": dict(armA=_entry(False, {"fatigue_limited_confirmed": 1}, exact=True,
                                                decisive=[dict(joint="robot:left_elbow",
                                                               category="fatigue_limited_confirmed")]),
                                    rail=dict(same_as_search_best=False, replay_exact=True,
                                              armA=_entry(False, {"rescaled_by_other": 1})))}
    out = cc.start_outcome(fams, "A")
    assert out["pass_b"] is None and out["pass_a"] is False
    assert all(d["B"] is None for d in out["demos"]) and len(out["demos"]) == 3
    assert out["failure"]["reading"] == "fatigue_limited_confirmed"
    assert out["failure"]["counts"] == {"small_change": 2, "fatigue_limited_confirmed": 1, "rescaled_by_other": 1}
    st = cc.stage_reading(out["pass_a"], out["pass_b"], [out["failure"]])
    assert "fatigue question" in st["text"] and "actuators cannot make" not in st["text"]
    assert out["failure"]["decisive_joints"][0]["replay"] == "joint_keyframes"
    assert "arm-A replay" in out["failure"]["source"]
    fams["joint_keyframes"]["rail"]["armA"] = _entry(True)
    out = cc.start_outcome(fams, "A")
    assert out["pass_a"] is True and out["failure"] is None


def test_a_confirmed_joint_reads_by_the_search_arm():
    meta, dt = cc.synthetic_meta(), 0.01
    att = cc.attribute(_saturated_elbow(), meta, 80, {"robot:left_elbow": +1}, dt)
    conf_b = cc.confirm_force_limited(att, att, att)
    conf_a = cc.confirm_force_limited(att, att, att, arm="A")
    assert cc.reading_from_counts(att, conf_b)["reading"] == "force_limited"
    assert cc.reading_from_counts(att, conf_a, search_arm="A")["reading"] == "fatigue_limited_confirmed"
    assert "under arm B" in conf_b["joints"][0]["why"] and "under arm A" in conf_a["joints"][0]["why"]
    unconf = cc.confirm_force_limited(att, None, att, arm="A")
    assert "not saturated the same way under arm A" in unconf["joints"][0]["why"]
    assert "arm B" not in cc.READINGS["force_saturated_unconfirmed"]
    assert "fatigue_limited_confirmed" in cc.DECISIVE_CATEGORIES


def test_row_id_by_search_arm():
    assert cc.row_id("B") == "wholebody_capacity_check_484675"
    assert cc.row_id("A") == "wholebody_capacity_search_fatigued_484675"


def test_start_outcome_under_search_arm_b_is_todays_rule():
    fams = {"world_targets": dict(armB=_entry(True, exact=True), armA=_entry(False, {"small_change": 1})),
            "joint_keyframes": dict(armB=_entry(False, {"wrong_command": 1}, exact=True),
                                    armA=_entry(False, {"wrong_command": 1}))}
    out = cc.start_outcome(fams, "B")
    assert out["pass_b"] is True and out["pass_a"] is False and out["failure"] is None
    assert [d["source"] for d in out["demos"]] == ["world_targets search best", "joint_keyframes search best"]
    fams["world_targets"]["armB"] = _entry(True, exact=False)          # not exact: no demonstration
    out = cc.start_outcome(fams, "B")
    assert out["pass_b"] is False and out["failure"]["counts"] == {"wrong_command": 1}
    assert "arm-B replay" in out["failure"]["source"]


def test_full_specification_follows_the_search_arm():
    import argparse
    args = argparse.Namespace(stages="rise,roll", bodies="0,1,2,3,4,5,6,7", hold_seconds=10.0, population=16,
                              generations=6, phases=4, phase_seconds=1.5, rise_starts="2,0t", strengths="1.0,0.46",
                              hold_replay=True, search_arm="A")
    assert cc.full_specification(args) == []
    args.stages = "hold,rise,roll"
    assert cc.full_specification(args) == ["stages 'hold,rise,roll' instead of 'rise,roll'"]
    args.search_arm = "B"
    assert cc.full_specification(args) == []


def test_search_arm_folders_and_choices(tmp_path, capsys, monkeypatch):
    assert cc.main(["run", "--out", str(cc.FATIGUED_OUT), "--dev", "--search-arm", "A", "--stages", "rise"]) == 2
    assert "never the registered result folder" in capsys.readouterr().out
    monkeypatch.setattr(cc, "tool_identity", lambda named_commit=None: dict(GOOD_IDENTITY))
    assert cc.main(["run", "--out", str(cc.DEFAULT_OUT), "--search-arm", "A", "--stages", "rise,roll",
                    "--commit", "abc"]) == 2
    assert "writes its registered result to" in capsys.readouterr().out
    (tmp_path / "declaration.json").write_text("{}", encoding="utf-8")
    assert cc.main(["run", "--out", str(tmp_path), "--search-arm", "A", "--stages", "rise,roll", "--commit", "abc"]) == 2
    assert "never overwritten" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cc.main(["run", "--out", str(tmp_path / "x"), "--search-arm", "C"])


# ------------------------------------------------------------------------------------------ rest then replay (--rest-replay)
def _source_run(root, rise_via="joint_keyframes search best", drop=None):
    """A source run folder as the registered check writes it: rise from body0_turned (keyframes, with a rail-respecting
    replay), roll from body0 (world targets), each start's report, and each arm-A record."""
    import json
    rise_params = dict(offsets=[[0.1] * 8] * 4, rail=False)
    roll_params = [[[0.01 * k] * 6] * 6 for k in range(4)]
    stages = {
        "rise": ("body0_turned", rise_via, dict(
            joint_keyframes=dict(search_best=dict(params=rise_params),
                                 armA=dict(passed=False, record="records/joint_keyframes_armA.npz",
                                           attribution=dict(counts={"force_saturated": 2}),
                                           reading=dict(reading="force_saturated_unconfirmed",
                                                        counts={"force_saturated_unconfirmed": 2})),
                                 rail=dict(params=dict(rise_params, rail=True), same_as_search_best=False,
                                           armA=dict(passed=False, record="records/joint_keyframes_rail_armA.npz"))))),
        "roll": ("body0", "world_targets search best", dict(
            world_targets=dict(search_best=dict(params=roll_params),
                               armA=dict(passed=False, record="records/world_targets_armA.npz",
                                         attribution=dict(counts={"small_change": 2}),
                                         reading=dict(reading="small_change", counts={"small_change": 2}))))),
    }
    for stage, (tag, source, families) in stages.items():
        sub = root / stage / tag
        (sub / "records").mkdir(parents=True)
        for fam in families.values():
            for holder in (fam, fam.get("rail") or {}):
                if holder.get("armA"):
                    np.savez_compressed(sub / holder["armA"]["record"], q=np.zeros((3, 2)))
        if drop == "params":
            families[next(iter(families))]["search_best"].pop("params")
        if drop == "record":
            (sub / families[next(iter(families))]["armA"]["record"]).unlink()
        (sub / "report.json").write_text(json.dumps(dict(
            start=tag, body=0, families=families,
            families_described=[dict(family="world_targets"), dict(family="joint_keyframes", side_chosen="R")])),
            encoding="utf-8")
        via = [] if drop == "via" else [dict(start=tag, source=source, B=True, A=False)]
        summary = dict(reading=dict(reading="fatigue", via=via))
        if drop == "arm":
            summary["search_arm"] = "A"
        (root / stage / "report.json").write_text(json.dumps(dict(stage=stage, summary=summary)), encoding="utf-8")
    return root


def test_rest_plan_reads_each_rested_demonstration(tmp_path):
    run = _source_run(tmp_path)
    plan = cc.rest_plan(run)
    assert [(p["stage"], p["start"], p["family"], p["variant"]) for p in plan] == [
        ("rise", "body0_turned", "joint_keyframes", "search best"), ("roll", "body0", "world_targets", "search best")]
    rise, roll = plan
    assert rise["turned"] and not roll["turned"] and rise["body"] == 0
    assert rise["params"]["rail"] is False and np.asarray(roll["params"]).shape == (4, 6, 6)
    report = tmp_path / "rise" / "body0_turned" / "report.json"
    assert rise["report_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    rec = tmp_path / "roll" / "body0" / "records" / "world_targets_armA.npz"
    assert roll["arm_a_record"] == str(rec) and roll["arm_a_record_sha256"] == cc.sha256_file(rec)
    assert rise["registered_arm_a"]["attribution_counts"] == {"force_saturated": 2}
    assert roll["roll_side_chosen"] == "R"
    assert [p["stage"] for p in cc.rest_plan(run, ["roll"])] == ["roll"]


def test_rest_plan_takes_the_rail_respecting_replay_when_it_was_the_demonstration(tmp_path):
    plan = cc.rest_plan(_source_run(tmp_path, rise_via="joint_keyframes rail-respecting replay"), ["rise"])
    assert plan[0]["variant"] == "rail-respecting replay" and plan[0]["params"]["rail"] is True
    assert plan[0]["arm_a_record"].endswith("joint_keyframes_rail_armA.npz")


@pytest.mark.parametrize("drop,words", [("params", "parameters"), ("record", "arm-A record"),
                                        ("via", "from no start"), ("arm", "not B")])
def test_rest_plan_refuses_a_missing_best(tmp_path, drop, words):
    with pytest.raises(cc.InvalidRun, match=words):
        cc.rest_plan(_source_run(tmp_path, drop=drop))


def test_rest_plan_refuses_a_missing_stage(tmp_path):
    with pytest.raises(cc.InvalidRun, match="no rise report"):
        cc.rest_plan(tmp_path)


def test_same_record_is_bit_for_bit_with_nan_equal():
    reg = dict(q=np.array([[1.0, np.nan]]), contacts=np.zeros((0, 10)), roll_valid=np.array([True]))
    mine = dict(q=np.array([[1.0, np.nan]]), contacts=np.zeros((0, 10)), roll_valid=np.array([True]),
                effort_per_set=np.zeros((1, 10)))
    out = cc.same_record(mine, reg)
    assert out["identical"] and out["fields_compared"] == 3
    mine["q"] = np.array([[1.0 + 1e-15, np.nan]])
    assert cc.same_record(mine, reg)["differing"] == ["q"]
    mine["q"] = np.array([[1.0, np.nan, 0.0]])
    assert cc.same_record(mine, reg)["differing"] == ["q"]
    del mine["q"]
    assert cc.same_record(mine, reg)["differing"] == ["q (absent)"]


def test_rest_verdict_needs_every_planned_case():
    expected = [("roll", "body0", t, v) for t in (0.0, 10.0) for v in cc.REST_VARIANTS]
    done = [dict(stage="roll", start="body0", rest_s=t, variant=v, measured=True)
            for t in (0.0, 10.0) for v in cc.REST_VARIANTS]
    assert cc.rest_verdict(expected, done, []) == "pass"
    assert cc.rest_verdict(expected, done[:3], []) == "invalid"          # T = 10 posture_kept unmeasured
    assert cc.rest_verdict(expected, done, ["outside contact"]) == "invalid"
    assert cc.rest_verdict([], [], ["the plan could not be read"]) == "invalid"


def test_capacity_rest_summaries_and_start_difference():
    cap = np.array([0.5, 0.57, 0.43, 0.40, 0.85, 0.43, 0.66, 0.66, 0.77, 0.79])
    s = cc.capacity_summary(cap)
    assert s["lowest_set"] == "Hand_R" and s["lowest"] == pytest.approx(0.40) and list(s["sets"]) == list(cc.CAPACITY_SETS)
    arrays = dict(effort_per_set=np.tile(np.arange(10.0), (4, 1)),
                  force_hi=np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.5, 3.0]]))
    q1 = np.r_[0.03, 0.04, 0.0, np.zeros(4), math.radians(3.0), math.radians(-5.0), 0]
    moved = cc.start_difference(np.zeros(10), q1, [7, 8], 0, ["robot:left_hip1", "robot:left_toes"])
    assert moved["max_joint_change_deg"] == pytest.approx(5.0) and moved["joint"] == "robot:left_toes"
    assert [j for j, _ in moved["largest_changes_deg"]] == ["robot:left_toes", "robot:left_hip1"]
    assert moved["root_moved_m"] == pytest.approx(0.05)
    r = cc.rest_summary(arrays, 10.0, cap, np.minimum(1.0, cap + 0.2), moved)
    assert r["physics_steps"] == 4 and r["effort_per_set_median"]["Hand_R"] == 3.0
    assert r["capacity_at_end"]["lowest"] == pytest.approx(0.60) and r["force_range_end_last_over_first_median"] == 1.5


def test_rest_outcome_names_each_case():
    assert cc.rest_outcome(True, True) == "met_in_both"
    assert cc.rest_outcome(False, True) == "met_only_posture_kept"
    assert cc.rest_outcome(False, False) == "met_in_neither"
    assert cc.rest_outcome(True, False) == "met_only_in_place"
    assert cc.rest_outcome(None, True) is None and cc.rest_outcome(True, None) is None
    assert "the whole state the rest ended in" in cc.REST_OUTCOMES["met_only_posture_kept"]
    assert "does not suffice for this command" in cc.REST_OUTCOMES["met_in_neither"]
    text = cc.REST_VARIANT_TEXT["posture_kept"]
    assert "not an artificial reset to 1.0" in text and "not the same as a real rest then movement" in text
    assert "not from the formula" in text


def test_rest_readings_pair_the_two_variants():
    cap = cc.capacity_summary(np.full(10, 0.9))
    drift = dict(max_joint_change_deg=38.7, joint="robot:right_hip1", root_moved_m=0.017)
    fail = dict(reading="force_saturated_unconfirmed", counts={"force_saturated_unconfirmed": 1}, decisive_joints=[])
    rows = [dict(stage="roll", start="body0", rest_s=0.0, variant=v, passed=False, capacity_before_movement=cap,
                 reading=fail, control=dict(identical=True), rest_movement=None) for v in cc.REST_VARIANTS]
    rows += [dict(stage="roll", start="body0", rest_s=10.0, variant="in_place", passed=False,
                  capacity_before_movement=cap, reading=fail, rest_movement=drift,
                  start_posture="the posture the body drifted to during the rest"),
             dict(stage="roll", start="body0", rest_s=10.0, variant="posture_kept", passed=True,
                  capacity_before_movement=cap, dwell_s=0.6, rest_movement=drift)]
    t0, t10 = cc.rest_readings(rows)
    assert t0["control"] and t0["outcome"] == "met_in_neither" and t0["in_place"]["control_identical"]
    assert t0["posture_kept"]["reading"] == "force_saturated_unconfirmed" and "unconfirmed" in t0["in_place"]["text"]
    assert t10["outcome"] == "met_only_posture_kept" and t10["drift_during_rest"]["joint"] == "robot:right_hip1"
    assert t10["posture_kept"]["reading"] == "met the line" and t10["posture_kept"]["counts"] is None
    assert t10["in_place"]["start_posture"].startswith("the posture the body drifted")
    assert t10["lowest_capacity_before"] == 0.9 and t10["text"] == cc.REST_OUTCOMES["met_only_posture_kept"]
    assert cc.rest_readings(rows[:3])[1]["outcome"] is None          # posture_kept at T = 10 not yet measured


def test_rollout_writes_a_capacity_vector_under_arm_a_only():
    from types import SimpleNamespace
    saved = np.full(10, 0.4)

    class Env:
        def __init__(self):
            self.proprio = SimpleNamespace(fatigue=SimpleNamespace(cap=np.zeros(10)))
            self.data = SimpleNamespace(qpos=np.zeros(3), qvel=np.zeros(3))
            self.model = SimpleNamespace(nu=2)

        def restore_full_state(self, snap):
            self.proprio.fatigue.cap[:] = snap["cap"]

    rec = SimpleNamespace(_install=lambda env: None, reset=lambda: None)
    ctx = SimpleNamespace(torch=None, a=SimpleNamespace(action_repeat=5, motor_hold=5, smooth_actions=0.35), dev={})
    env = Env()
    cc.rollout(ctx, env, dict(cap=saved), "A", None, 0, rec, motors_off=True)
    assert np.array_equal(env.proprio.fatigue.cap, saved)
    rested = np.linspace(0.6, 0.99, 10)
    cc.rollout(ctx, env, dict(cap=saved), "A", None, 0, rec, motors_off=True, capacity=rested)
    assert np.array_equal(env.proprio.fatigue.cap, rested)
    with pytest.raises(ValueError):
        cc.rollout(ctx, env, dict(cap=saved), "B", None, 0, rec, motors_off=True, capacity=rested)
    for bad in (np.full(9, 0.5), np.full(10, 1.5), np.r_[np.full(9, 0.5), np.nan]):
        with pytest.raises(ValueError):
            cc.rollout(ctx, env, dict(cap=saved), "A", None, 0, rec, motors_off=True, capacity=bad)


def test_rest_specification_lists_each_deviation(tmp_path):
    import argparse
    args = argparse.Namespace(rest_replay="10,30,60", stages="rise,roll", phases=4, phase_seconds=1.5,
                              rest_strength=1.0, from_run=cc.DEFAULT_OUT)
    assert cc.rest_specification(args) == []
    args.rest_replay, args.stages, args.from_run = "10", "roll", tmp_path
    assert cc.rest_specification(args) == ["rest_replay '10' instead of '10,30,60'",
                                           "stages 'roll' instead of 'rise,roll'",
                                           "from_run %s instead of %s" % (tmp_path, cc.DEFAULT_OUT)]


def test_declared_durations_are_the_formulas():
    for t, v in ((10, 0.636), (30, 0.866), (60, 0.970)):
        assert round(1 - 0.6 * math.exp(-0.05 * t), 3) == v
    assert "0.636, 0.866 and 0.970" in cc.MEASURED_RECOVERY["expected"]
    assert "0.0500" in cc.MEASURED_RECOVERY["fit"] and "0.0200" in cc.MEASURED_RECOVERY["fit"]
    assert "effort_per_set" in cc.RECORD_FIELDS


def test_rest_replay_refusals(tmp_path, capsys, monkeypatch):
    rest = ["--rest-replay", "10,30,60"]
    assert cc.main(["run", "--out", str(cc.REST_OUT), "--dev"] + rest) == 2
    assert "never the registered result folder" in capsys.readouterr().out
    assert cc.main(["run", "--out", str(cc.DEFAULT_OUT / "rest"), "--dev"] + rest) == 2
    assert "never writes into its source run" in capsys.readouterr().out
    assert cc.main(["run", "--out", str(tmp_path / "x"), "--rest-replay", "10"]) == 2
    assert "must carry --dev" in capsys.readouterr().out
    assert cc.main(["run", "--out", str(tmp_path / "x"), "--search-arm", "A", "--dev"] + rest) == 2
    assert "runs no search" in capsys.readouterr().out
    monkeypatch.setattr(cc, "tool_identity", lambda named_commit=None: dict(GOOD_IDENTITY))
    assert cc.main(["run", "--out", str(tmp_path / "x"), "--commit", "abc"] + rest) == 2
    assert "writes its registered result to" in capsys.readouterr().out
    assert not (tmp_path / "x").exists()
    assert cc.main(["run", "--out", str(cc.REST_OUT), "--commit", "abc"]) == 2
    assert "rest replay's registered result folder" in capsys.readouterr().out
    monkeypatch.setattr(cc, "REST_OUT", tmp_path)
    (tmp_path / "declaration.json").write_text("{}", encoding="utf-8")
    assert cc.main(["run", "--out", str(tmp_path), "--commit", "abc"] + rest) == 2
    assert "never overwritten" in capsys.readouterr().out
    assert (tmp_path / "declaration.json").read_text(encoding="utf-8") == "{}"
    for bad in (["--rest-replay", "0"], ["--rest-replay", "10,10"], ["--rest-replay", "x"],
                ["--rest-replay", "10", "--stages", "hold"]):
        with pytest.raises(SystemExit):
            cc.main(["run", "--out", str(tmp_path / "y"), "--dev"] + bad)


# ------------------------------------------------------------------------------------------ one saved world (heavy)
@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("WHOLEBODY_CAPACITY_WORLD") != "1",
                    reason="builds body 0's saved MuJoCo world: set WHOLEBODY_CAPACITY_WORLD=1 under the memory guard")
def test_one_saved_world_half_second_hold(tmp_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    comp = cc.Companion(cc.DEFAULT_INPUTS)
    try:
        ctx = cc.Context(tmp_path)
        env, report = cc.build_world(ctx, comp.world(0), tmp_path / "w0")
        try:
            assert report["outside"]["outside_contacts_now"] == 0
            assert report["differs_from_launch"] == {}
            place = cc.Placement(ctx, env)
            solved = place.solve()
            assert solved["deepest_penetration_mm"] <= 2.0 + 1e-9
            placed = place.apply()
            snap = env.save_full_state()
            n_model = 10
            finals = []
            for full in (True, False):
                rec = cc.Recorder(ctx, env, full=full)
                finals.append(cc.rollout(ctx, env, snap, "A", cc.hold_command(ctx, 1.0), n_model, rec))
                arrays = rec.arrays()
                assert len(arrays["t"]) == n_model * ctx.a.action_repeat
                assert int(arrays["outside_contacts"].max()) == 0
                if full:
                    dc = cc.demand_check(arrays, rec.meta())
                    assert dc["checked"] > 0 and dc["max_abs_difference"] <= 1e-9
                    assert arrays["demand"].shape == (n_model * ctx.a.action_repeat, env.model.nu)
                    assert np.isfinite(arrays["act_force"]).all()
                    eff, cap = arrays["effort_per_set"], arrays["capacity"]
                    assert eff.shape == (n_model * ctx.a.action_repeat, 10)
                    np.testing.assert_allclose(np.diff(cap, axis=0),
                                               env.dt * (0.05 * (1 - cap[:-1]) - 0.02 * eff[1:]), atol=1e-12)
            np.testing.assert_array_equal(finals[0]["final_qpos"], finals[1]["final_qpos"])
            np.testing.assert_array_equal(finals[0]["final_qvel"], finals[1]["final_qvel"])
            assert placed["chest_z"] > 0.15
        finally:
            env.close()
    finally:
        comp.close()
