# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""THE TARGET OUTCOME of the training preflight (proxy decision pose-v2-implementation-cleared-20261015, answer 4): an experiment whose purpose is a registered outcome, scoped by
its route, with all seven gates kept and every FAIL excepted. Stand-in trainer source, records, registry, defect register
and decision message in pytest's temporary folder; nothing of production is read or written. NOT RUN by the build: the
grouped check pose_readout_comparison_admission_checks_20261015 runs it once the coordinator relays the overseer's OK."""
import hashlib
import json
from pathlib import Path

import pytest

from mimo_bridge.tests.test_training_preflight import (CAP_FLAGS, STOP_CRITERIA, _at, _bounded, _cmd, _experiment_setup,
                                                       _run, _src)
from mimo_bridge.tools import training_preflight as TP

D = "POSTURE_READOUT_TEST_DEFECT"
AUTHOR = "the overseer"
TEXT = "H84 proxy decision: the test pair\nAuthor: %s.\nImplement the target outcome.\n" % AUTHOR


def _sha_text(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def _sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _setup(tmp_path, monkeypatch):
    src, root = _src(tmp_path), tmp_path / "pre"
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps(dict(rows=[dict(id="stop_row", status="written"),
                                         dict(id="read_row", status="written", defect=D)])), encoding="utf-8")
    register = tmp_path / "defect_register.json"
    register.write_text(json.dumps(dict(defects=[dict(id=D, state="BUILT/OFF", severity="GOF-BLOCKER")])),
                        encoding="utf-8")
    (tmp_path / "mailbox" / "dec").mkdir(parents=True)
    (tmp_path / "mailbox" / "dec" / "message.json").write_text(json.dumps(dict(id="dec", text=TEXT)), encoding="utf-8")
    reader = tmp_path / "reader.py"
    reader.write_text("def main(argv=None):\n    return 0\n", encoding="utf-8")
    a = _run(tmp_path, "run_a", (1000,))
    start = a / "checkpoints/ckpt_step_1000"
    (a / "replay_training_stats.jsonl").write_text("".join(json.dumps(dict(
        type="mimo", step=step, elapsed_s=elapsed, causal_pe_per_modality=dict(vision=0.01),
        cmd_response_cos_held=0.5)) + "\n" for step, elapsed in ((900, 167.0), (1000, 440.0))), encoding="utf-8")
    # each arm's package folder (its --out lies inside it) and the start's MANIFEST (corrections of Oct 15, 2026)
    pkg_r, pkg_c = tmp_path / "pkg_r", tmp_path / "pkg_c"
    control = TP.apply_sets(_at(_cmd(), start), [("--out", str(pkg_c / "run" / "control"))])
    pose = TP.apply_sets(_at(_cmd(), start), [("--attention-cap", "on"), ("--attention-cap-bound", "3"),
                                              ("--out", str(pkg_r / "run" / "pose"))])
    fp_r = Path(TP.new_record(pose, "the pose arm", "tester", root=root, src=src)).stem
    fp_c = Path(TP.new_record(control, "the matched control", "tester", root=root, src=src)).stem
    code_hash = TP.configuration(pose, src)["code_hash"]
    manifest = tmp_path / "preserved" / "MANIFEST.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(dict(verdict="pass", files=[])), encoding="utf-8")
    for pkg, fp in ((pkg_r, fp_r), (pkg_c, fp_c)):
        pkg.mkdir()
        (pkg / "arm_continuation.json").write_text(json.dumps(dict(
            fingerprint=fp, code=dict(commit="abc123"), preserved_prefix="pose_comparison_",
            route=dict(manifest_sha256=_sha_file(manifest)))), encoding="utf-8")
    route = dict(experiment="pose_pair", arms=dict(repair=fp_r, control=fp_c), defect=D, reading_row="read_row",
                 decision="dec", decision_text_sha256=_sha_text(TEXT), checkpoint=str(start), end_step=3000,
                 wall_cap_s=6339.0, code_commit="abc123", start_manifest=str(manifest),
                 start_manifest_sha256=_sha_file(manifest), preserved_prefix="pose_comparison_",
                 packages=dict(repair=str(pkg_r), control=str(pkg_c)), execution=None)
    monkeypatch.setattr(TP, "TARGET_OUTCOME_ROUTES", (route,))
    monkeypatch.setattr(TP, "DEFECT_REGISTER", register)
    monkeypatch.setattr(TP, "MAILBOX", tmp_path / "mailbox")
    to = dict(text="the pose read-out keeps the hand's goal progress", defect=D, reading_row="read_row",
              rule="CONSISTENT ADVANTAGE B when the median first-hold difference is above 0 in each owner state",
              inputs=[dict(name="the reader", source=str(reader), sha256=_sha_file(reader))],
              reader=dict(module=str(reader), sha256=_sha_file(reader)),
              pins=dict(authorization=dict(id="dec", text_sha256=_sha_text(TEXT), author=AUTHOR), code_commit="abc123",
                        code_hash=code_hash, start_checkpoint=str(start), start_manifest_sha256=_sha_file(manifest),
                        end_step=3000, wall_cap_s=6339.0, preserved_prefix="pose_comparison_"))
    exp = dict(id="pose_pair", authorization="the delegation's words", authorized_by="the overseer",
               authorized_at="Oct 15, 2026 10:47 AM PDT", arms=dict(repair=fp_r, control=fp_c),
               differs_only_in=list(CAP_FLAGS),
               exceptions={str(n): dict(failure="gate %d fails" % n, why_measurable="shared by both arms",
                                        evidence=["row_%d" % n]) for n in range(1, 8)},
               learning_signal=dict(text="the head receives its lesson's gradient", evidence=["row_ls"]),
               outcome_validity=dict(text="the reading reads both arms on common inputs", evidence=["row_ov"]),
               target_outcome=to)
    common = dict(start_checkpoint=str(start), rollback_checkpoint=str(start), stop_lines="S1 S2",
                  stop_lines_row="stop_row", stop_criteria=STOP_CRITERIA, max_model_steps=2000,
                  affected_components=["PROPRIOCEPTION_ENCODER"], no_auto_continue="the owner leaves no continuation",
                  reachability="the first check reads the stop lines about 330 model steps in")
    TP.set_repair_run(fp_r, dict(common, role="repair", purpose="the pose read-out's outcome", target_gates=[],
                                 repairs=["POSE_READOUT"], remeasure={"outcome": "read_row"}, experiment=exp),
                      "tester", root=root, registry=reg)
    TP.set_repair_run(fp_c, dict(common, role="control", purpose="the matched control", target_gates=[], repairs=[],
                                 experiment=exp), "tester", root=root, registry=reg)
    for fp in (fp_r, fp_c):
        for n in range(1, 8):
            TP.set_gate(fp, n, "FAIL", "fails on the lineage", ["row_%d" % n], "tester", root=root)
    return dict(src=src, root=root, reg=reg, pose=_bounded(pose, 3000), control=_bounded(control, 3000), fp_r=fp_r,
                fp_c=fp_c, start=start, reader=reader, register=register, exp=exp, common=common, pose_raw=pose,
                route=route, manifest=manifest, pkg_r=pkg_r, pkg_c=pkg_c)


def _why(s, which="pose"):
    return TP.launch_refusal(s[which], root=s["root"], src=s["src"], registry=s["reg"])


def _edit(s, fp, fn):
    rec = TP._load(TP.record_path(fp, s["root"]))
    fn(rec["repair_run"])
    TP._save(TP.record_path(fp, s["root"]), rec)


def test_both_arms_launch_with_every_gate_failed_and_excepted_and_the_plan_carries_the_cap(tmp_path, monkeypatch):
    s = _setup(tmp_path, monkeypatch)
    assert _why(s, "pose") is None and _why(s, "control") is None
    for which in ("pose", "control"):
        plan = TP.launch_plan(s[which], root=s["root"], src=s["src"])
        assert plan["kind"] == "repair_run" and plan["block"] == "repair_run" and plan["max_steps"] == 3000
        assert plan["wall_cap_s"] == 6339.0


def test_a_fail_without_its_complete_exception_blocks(tmp_path, monkeypatch):
    s = _setup(tmp_path, monkeypatch)
    _edit(s, s["fp_r"], lambda rr: rr["experiment"]["exceptions"].pop("3"))
    why = _why(s)
    assert why and "gate 3 Scale/conditioning: FAIL" in why
    _edit(s, s["fp_r"], lambda rr: rr["experiment"]["exceptions"].__setitem__("3", dict(failure="x", evidence=["r"])))
    assert "gate 3 Scale/conditioning: FAIL" in _why(s)


def test_a_gate_never_judged_or_read_as_a_repair_target_blocks(tmp_path, monkeypatch):
    s = _setup(tmp_path, monkeypatch)
    with pytest.raises(TP.PreflightError, match="only when the record's repair run names it"):
        TP.set_gate(s["fp_r"], 4, TP.REPAIR_TARGET, "would be the target", ["r"], "tester", root=s["root"])
    rec = TP._load(TP.record_path(s["fp_r"], s["root"]))
    rec["gates"] = [dict(g, status="OPEN") if g["n"] == 5 else g for g in rec["gates"]]
    TP._save(TP.record_path(s["fp_r"], s["root"]), rec)
    assert "gate 5 Causal controllability: OPEN" in _why(s)


def test_mixed_targets_are_refused(tmp_path, monkeypatch):
    s = _setup(tmp_path, monkeypatch)
    block = TP._load(TP.record_path(s["fp_r"], s["root"]))["repair_run"]
    fresh = {k: v for k, v in block.items() if k not in ("written_at", "written_by", "launched_at", "launched_at_s",
                                                          "run")}
    with pytest.raises(TP.PreflightError, match="mixed targets"):
        TP.set_repair_run(s["fp_r"], dict(fresh, target_gates=[4], remeasure={"4": "r", "outcome": "read_row"}),
                          "tester", root=s["root"], registry=s["reg"])
    with pytest.raises(TP.PreflightError, match="remeasure is exactly"):
        TP.set_repair_run(s["fp_r"], dict(fresh, remeasure={"outcome": "another_row"}), "tester", root=s["root"],
                          registry=s["reg"])


def test_an_absent_or_non_executable_reading_is_refused(tmp_path, monkeypatch):
    s = _setup(tmp_path, monkeypatch)
    s["reg"].write_text(json.dumps(dict(rows=[dict(id="stop_row", status="written")])), encoding="utf-8")
    assert "the reading row read_row is not in the test registry" in _why(s)
    s["reg"].write_text(json.dumps(dict(rows=[dict(id="stop_row", status="written"),
                                              dict(id="read_row", status="done", defect=D)])), encoding="utf-8")
    assert "the reading row read_row reads done" in _why(s)
    s["reg"].write_text(json.dumps(dict(rows=[dict(id="stop_row", status="written"),
                                              dict(id="read_row", status="written", defect="ANOTHER")])),
                        encoding="utf-8")
    assert "does not name the defect" in _why(s)
    s["reg"].write_text(json.dumps(dict(rows=[dict(id="stop_row", status="written"),
                                              dict(id="read_row", status="written", defect=D)])), encoding="utf-8")
    s["reader"].write_text("def main(argv=None):\n    return 1\n", encoding="utf-8")
    why = _why(s)
    assert "the target outcome's reader" in why and "not the pinned" in why
    assert "does not hash to" in why                                      # the input that pins the same file
    _edit(s, s["fp_r"], lambda rr: rr["experiment"]["target_outcome"].__setitem__("rule", ""))
    assert "the target outcome's rule is missing" in _why(s)


def test_a_reader_without_main_is_refused(tmp_path, monkeypatch):
    s = _setup(tmp_path, monkeypatch)
    s["reader"].write_text("x = 1\n", encoding="utf-8")
    sha = _sha_file(s["reader"])
    for fp in (s["fp_r"], s["fp_c"]):
        _edit(s, fp, lambda rr: (rr["experiment"]["target_outcome"]["reader"].__setitem__("sha256", sha),
                                 rr["experiment"]["target_outcome"]["inputs"][0].__setitem__("sha256", sha)))
    assert "defines no main" in _why(s)


def test_mismatched_arms_are_refused(tmp_path, monkeypatch):
    s = _setup(tmp_path, monkeypatch)
    _edit(s, s["fp_c"], lambda rr: rr["experiment"]["target_outcome"].__setitem__("text", "another purpose"))
    assert "the arms' experiment blocks differ in target_outcome" in _why(s)
    assert "the arms' experiment blocks differ in target_outcome" in _why(s, "control")


def test_an_unbounded_or_shorter_launch_is_refused(tmp_path, monkeypatch):
    s = _setup(tmp_path, monkeypatch)
    why = TP.launch_refusal(_bounded(s["pose_raw"], 2500), root=s["root"], src=s["src"], registry=s["reg"])
    assert "runs exactly to the pinned end step 3000" in why
    why = TP.launch_refusal(s["pose_raw"], root=s["root"], src=s["src"], registry=s["reg"])
    assert "not bounded as declared" in why and "runs exactly to the pinned end step" in why
    _edit(s, s["fp_r"], lambda rr: rr.__setitem__("max_model_steps", 1500))
    assert "max_model_steps 1500 is not the pinned end step 3000" in _why(s)


def test_consumed_authority_is_refused(tmp_path, monkeypatch):
    s = _setup(tmp_path, monkeypatch)
    other_cmd = TP.apply_sets(_at(_cmd(), s["start"]), [("--planner-search", "trust_v1")])
    fp_o = Path(TP.new_record(other_cmd, "another experiment", "tester", root=s["root"], src=s["src"])).stem
    rec = TP._load(TP.record_path(fp_o, s["root"]))
    rec["repair_run"] = dict(experiment=dict(id="another_pair", target_outcome=s["exp"]["target_outcome"]))
    TP._save(TP.record_path(fp_o, s["root"]), rec)
    why = _why(s)
    assert "the decision dec already authorized experiment another_pair of record %s" % fp_o in why


def test_a_pair_outside_its_route_is_refused(tmp_path, monkeypatch):
    s = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(TP, "TARGET_OUTCOME_ROUTES", ())
    assert "not an entry of TARGET_OUTCOME_ROUTES" in _why(s)
    route = dict(experiment="pose_pair", arms=dict(repair=s["fp_r"], control="0" * 16), defect=D,
                 reading_row="read_row", decision="dec", decision_text_sha256=_sha_text(TEXT),
                 checkpoint=str(s["start"]), end_step=3000, wall_cap_s=6339.0)
    monkeypatch.setattr(TP, "TARGET_OUTCOME_ROUTES", (route,))
    assert "the target outcome's arms reads" in _why(s)


def test_an_unregistered_or_closed_defect_is_refused(tmp_path, monkeypatch):
    s = _setup(tmp_path, monkeypatch)
    s["register"].write_text(json.dumps(dict(defects=[dict(id=D, state="VERIFIED")])), encoding="utf-8")
    assert "is closed (state VERIFIED)" in _why(s)
    s["register"].write_text(json.dumps(dict(defects=[])), encoding="utf-8")
    assert "is not in the defect register" in _why(s)


def test_a_changed_decision_text_or_code_is_refused(tmp_path, monkeypatch):
    s = _setup(tmp_path, monkeypatch)
    (tmp_path / "mailbox" / "dec" / "message.json").write_text(json.dumps(dict(id="dec", text=TEXT + " ")),
                                                               encoding="utf-8")
    assert "the authorizing decision's text hashes" in _why(s)
    (tmp_path / "mailbox" / "dec" / "message.json").write_text(json.dumps(dict(id="dec", text=TEXT)), encoding="utf-8")
    for fp in (s["fp_r"], s["fp_c"]):
        _edit(s, fp, lambda rr: rr["experiment"]["target_outcome"]["pins"].__setitem__("code_hash", "f" * 64))
    assert "is not the pinned " + "f" * 64 in _why(s)


def test_each_arm_is_used_once_and_never_reaches_normal_training(tmp_path, monkeypatch):
    s = _setup(tmp_path, monkeypatch)
    run = tmp_path / "experiments" / "pose_run"
    run.mkdir(parents=True)
    TP.note_attempt(s["fp_r"], str(run), "tester", root=s["root"])
    TP.mark_launched(s["fp_r"], str(run), "tester", root=s["root"])
    why = _why(s)
    assert why and "gate 1 Information flow: FAIL" in why                  # judged as normal training: refused
    assert _why(s, "control") is None                                       # the other arm keeps its one use


# ------------------------------------------------------------------ the corrections of Oct 15, 2026 (ADMISSION)
def test_the_control_of_a_target_outcome_is_used_once_too(tmp_path, monkeypatch):
    """Decision pose-build-b-corrections-and-cpu-authority-20261015: "Enforce one use for BOTH named target-outcome
    arms; reject a control repeat above one"."""
    s = _setup(tmp_path, monkeypatch)
    _edit(s, s["fp_c"], lambda rr: rr.__setitem__("repeats", 2))
    assert "a target-outcome arm is used once, whatever its role (repeats reads 2)" in _why(s, "control")
    block = TP._load(TP.record_path(s["fp_c"], s["root"]))["repair_run"]
    fresh = {k: v for k, v in block.items() if k not in ("written_at", "written_by", "launched_at", "launched_at_s",
                                                          "run")}
    with pytest.raises(TP.PreflightError, match="used once, whatever its role"):
        TP.set_repair_run(s["fp_c"], dict(fresh, repeats=2), "tester", root=s["root"], registry=s["reg"])
    _edit(s, s["fp_c"], lambda rr: rr.__setitem__("repeats", 1))
    assert _why(s, "control") is None


def test_a_consumed_control_is_refused(tmp_path, monkeypatch):
    s = _setup(tmp_path, monkeypatch)
    run = tmp_path / "experiments" / "control_run"
    run.mkdir(parents=True)
    TP.note_attempt(s["fp_c"], str(run), "tester", root=s["root"])
    TP.mark_launched(s["fp_c"], str(run), "tester", root=s["root"])
    why = _why(s, "control")
    assert why and "gate 1 Information flow: FAIL" in why                  # its one use is spent
    assert _why(s) is None                                                  # the repair arm keeps its own


def test_a_changed_start_manifest_is_refused(tmp_path, monkeypatch):
    """The corrections: "Verify the actual start MANIFEST against its digest ... Add changed-start ... refusals"."""
    s = _setup(tmp_path, monkeypatch)
    s["manifest"].write_text(json.dumps(dict(verdict="pass", files=["changed"])), encoding="utf-8")
    why = _why(s)
    assert why and "the start MANIFEST %s hashes" % s["manifest"] in why
    assert "the start MANIFEST" in _why(s, "control")
    for fp in (s["fp_r"], s["fp_c"]):
        _edit(s, fp, lambda rr: rr["experiment"]["target_outcome"]["pins"].__setitem__("start_manifest_sha256",
                                                                                       "0" * 64))
    assert "the target outcome's start manifest sha256 reads '%s'" % ("0" * 64) in _why(s)


def test_the_code_commit_and_preservation_destination_are_bound_to_the_route_and_package(tmp_path, monkeypatch):
    s = _setup(tmp_path, monkeypatch)
    for fp in (s["fp_r"], s["fp_c"]):
        _edit(s, fp, lambda rr: rr["experiment"]["target_outcome"]["pins"].update(code_commit="def456",
                                                                                  preserved_prefix="elsewhere_"))
    why = _why(s)
    assert "the target outcome's code commit reads 'def456', the route's 'abc123'" in why
    assert "the target outcome's preserved prefix reads 'elsewhere_'" in why
    assert "the repair arm's package names code commit 'abc123', the launch 'def456'" in why


def test_a_launch_outside_its_package_or_from_a_changed_package_is_refused(tmp_path, monkeypatch):
    s = _setup(tmp_path, monkeypatch)
    away = TP.apply_sets(s["pose"], [("--out", str(tmp_path / "elsewhere" / "pose"))])
    why = TP.launch_refusal(away, root=s["root"], src=s["src"], registry=s["reg"])
    assert why and "does not lie inside the repair arm's package" in why
    ac = json.loads((s["pkg_c"] / "arm_continuation.json").read_text(encoding="utf-8"))
    (s["pkg_c"] / "arm_continuation.json").write_text(json.dumps(dict(ac, preserved_prefix="other_")), encoding="utf-8")
    assert "the control arm's package names preserved prefix 'other_'" in _why(s, "control")


def test_a_production_launch_needs_the_exact_execution_decision_and_a_rehearsal_is_no_authority(tmp_path, monkeypatch):
    """The corrections: "Keep implementation and execution authority distinct ... not a truthy execution field or the
    implementation-only decision"."""
    s = _setup(tmp_path, monkeypatch)
    assert _why(s) is None                                                  # a rehearsal folder: no execution read
    monkeypatch.setattr(TP, "PREFLIGHT_DIR", s["root"])                    # judged as the production folder
    assert "the route names no execution decision" in _why(s)
    s["route"]["execution"] = dict(id="dec", text_sha256=_sha_text(TEXT), author=AUTHOR)
    assert "is its implementation decision" in _why(s)
    names = "%s %s %s %s" % (s["fp_r"], s["fp_c"], s["pkg_r"], s["pkg_c"])
    ex_text = "H84 proxy decision: execute the pair\nAuthor: %s.\nRun %s.\n" % (AUTHOR, names)
    (tmp_path / "mailbox" / "exec").mkdir(parents=True)
    (tmp_path / "mailbox" / "exec" / "message.json").write_text(json.dumps(dict(id="exec", text=ex_text)),
                                                                encoding="utf-8")
    s["route"]["execution"] = dict(id="exec", text_sha256=_sha_text(ex_text), author=AUTHOR)
    assert _why(s) is None and _why(s, "control") is None
    s["route"]["execution"] = dict(id="exec", text_sha256="f" * 64, author=AUTHOR)
    assert "the execution decision: the authorizing decision's text hashes" in _why(s)
    short = "H84 proxy decision: execute\nAuthor: %s.\nRun %s.\n" % (AUTHOR, s["fp_r"])
    (tmp_path / "mailbox" / "exec" / "message.json").write_text(json.dumps(dict(id="exec", text=short)),
                                                                encoding="utf-8")
    s["route"]["execution"] = dict(id="exec", text_sha256=_sha_text(short), author=AUTHOR)
    assert "does not name" in _why(s)


def test_an_experiment_without_a_target_outcome_is_judged_as_before(tmp_path):
    src, root, reg, start, cap, control, fp_r, fp_c = _experiment_setup(tmp_path)
    assert TP.launch_refusal(_bounded(cap, 3000), root=root, src=src, registry=reg) is None
    assert TP.launch_refusal(_bounded(control, 3000), root=root, src=src, registry=reg) is None
    assert "wall_cap_s" not in TP.launch_plan(_bounded(cap, 3000), root=root, src=src)
