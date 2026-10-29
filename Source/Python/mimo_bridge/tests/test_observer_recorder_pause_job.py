# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The recorder track's pause job (docs/gain_of_function_v4/observer_recorder_pause_job.py), without a
world: the ONE validate run's outcome read against the declared failures (expected failures matched,
unexpected failures, unexpected passes; only the last two block - the coordinator's rule of the evening
of 2026-09-25), and what keeps a fresh babble recording from being clean (the qualification gate).
"""
import importlib.util
import json
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize('fault', [None, 'missing_body', 'zero_records', 'dropped', 'missing_field',
                                   'no_written_records', 'invalid_cost', 'status_error', 'digested_chunks',
                                   'missing_pid', 'wrong_header_pid', 'wrong_body', 'wrong_run',
                                   'invalid_created', 'historical_error', 'current_error'])
def test_after_resume_requires_eight_complete_recordings(tmp_path, monkeypatch, fault):
    import numpy as np
    from mimo_bridge.observer_record import FIELDS
    job = _job()
    monkeypatch.setattr(job, 'MEASUREMENTS', str(tmp_path/'receipts'))
    for body in range(8):
        if body == 7 and fault == 'missing_body':
            continue
        folder = tmp_path/'run/observer/world'/('b%d' % body)
        folder.mkdir(parents=True)
        header = 'header_101_123.json'
        fields = list(FIELDS)
        status = dict(pid=101, body=body, run='run', time=1001, header=header, records=2,
                      written_bytes=128, dropped=0, ms_per_record=3.0)
        head = dict(pid=101, body=body, run='run', created=1000, version='observer_world_v2', fields=fields)
        chunk_pid = 101
        if body == 7:
            if fault == 'zero_records': status['records'] = 0
            if fault == 'dropped': status['dropped'] = 1
            if fault == 'missing_field': fields.remove('orders')
            if fault == 'no_written_records': status['written_bytes'] = 0
            if fault == 'invalid_cost': status['ms_per_record'] = float('nan')
            if fault == 'status_error': status['error'] = 'failed to record'
            if fault == 'missing_pid': status.pop('pid')
            if fault == 'wrong_header_pid': head['pid'] = 102
            if fault == 'wrong_body': head['body'] = 0
            if fault == 'wrong_run': head['run'] = 'another_run'
            if fault == 'invalid_created': head['created'] = float('nan')
            if fault in ('historical_error', 'current_error'):
                err = dict(pid=100 if fault == 'historical_error' else 101,
                           time=999 if fault == 'historical_error' else 1001, reason='recording failed')
                (folder/'recorder_error.json').write_text(json.dumps(err), encoding='utf8')
        (folder/header).write_text(json.dumps(head), encoding='utf8')
        (folder/'status_101.json').write_text(json.dumps(status), encoding='utf8')
        if fault != 'digested_chunks':
            np.savez(folder/('chunk_000000001_%d_00000.npz' % chunk_pid), step=np.array([1, 2]), header=header)
    passed = fault in (None, 'digested_chunks', 'historical_error')
    assert job.after_resume(str(tmp_path/'run')) == (0 if passed else 1)
    receipt = json.loads(next((tmp_path/'receipts').glob('*.json')).read_text())
    assert receipt['integrity_passed'] is passed
    assert len(receipt['bodies']) == 8
    if fault == 'historical_error':
        assert receipt['bodies'][-1]['historical_errors']


def _job():
    spec = importlib.util.spec_from_file_location(
        "observer_recorder_pause_job_under_test", ROOT / "docs" / "gain_of_function_v4" / "observer_recorder_pause_job.py")
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    return job


def _case(name, approved=True, present=None, absent=None, changed=False):
    c = dict(name=name, approved=approved, present_found=present or {}, absent_clear=absent or {})
    if changed:
        c["scenario_changed"] = dict(approved="aaaa", recorded="bbbb")
    return c


def _report(cases, **kw):
    rep = dict(cases=cases, checks=dict(finger_sign=dict(ok=True), panel_order=dict(ok=True), fast_lane=dict(ok=True)),
               missing_locked_cases=[], lock=dict(intact=True), not_approved=[], proposal="proposals/x")
    rep.update(kw)
    return rep


EXPECTED = dict(locked_line_failures=[dict(case="a", line="H01:R present")],
                scenario_changed=[dict(case="b"), dict(case="c")],
                code_line_failures=[dict(case="d", line="B03:R present")])


def test_declared_failures_that_happen_do_not_block():
    J = _job()
    rep = _report([_case("a", present={"H01:R": False, "H05:R": True}), _case("b", changed=True),
                   _case("c", changed=True), _case("d", present={"B03:R": True})])
    code = dict(a={"H01:R present": False}, d={"B03:R present": False})
    v = J.classify(rep, dict(EXPECTED, code_line_failures=EXPECTED["code_line_failures"]
                             + [dict(case="a", line="H01:R present")]), code)
    assert set(v["expected_failures_matched"]) == {"locked line: a: H01:R present", "scenario changed: b",
                                                    "scenario changed: c", "code line: a: H01:R present",
                                                    "code line: d: B03:R present"}
    assert v["unexpected_failures"] == [] and v["unexpected_passes"] == [] and v["blocks"] is False


def test_an_undeclared_failure_and_a_declared_failure_that_passes_both_block():
    J = _job()
    rep = _report([_case("a", present={"H01:R": False}), _case("b", changed=True),
                   _case("c"),                                        # declared to change; it did not
                   _case("e", absent={"R04": False}),                 # a failure nobody declared
                   _case("n", approved=False, present={"B03:R": False})])  # new: never judged by the lock
    code = dict(a={"H01:R present": False}, d={"B03:R present": True}, n={"B03:R present": False})
    v = J.classify(rep, EXPECTED, code)
    assert set(v["unexpected_failures"]) == {"locked line: e: R04 absent", "code line: a: H01:R present",
                                             "code line: n: B03:R present"}
    assert set(v["unexpected_passes"]) == {"scenario unchanged: c", "code line: d: B03:R present"}
    assert v["blocks"] is True


def test_a_missing_case_a_failed_world_check_and_a_broken_lock_are_unexpected_failures():
    J = _job()
    rep = _report([_case("a", present={"H01:R": False}), _case("b", changed=True), _case("c", changed=True)],
                  missing_locked_cases=["z"], checks=dict(finger_sign=dict(ok=False), fast_lane=dict(ok=True)),
                  lock=dict(intact=False, error="the locked folder differs from its last commit"))
    v = J.classify(rep, dict(EXPECTED, code_line_failures=[]), dict(q=None))
    assert set(v["unexpected_failures"]) == {
        "locked case not run: z", "world check failed: finger_sign",
        "the locked lines: the locked folder differs from its last commit",
        "code lines not judged (no recorded detections): q"}
    assert v["blocks"] is True


def test_a_declared_line_of_a_case_whose_scenario_changed_is_superseded_not_passed():
    J = _job()
    rep = _report([_case("a", changed=True), _case("b", changed=True), _case("c", changed=True)])
    v = J.classify(rep, dict(EXPECTED, code_line_failures=[]), {})
    assert v["declared_but_superseded"] == ["locked line in a changed scenario: a: H01:R present"]
    assert v["unexpected_failures"] == ["scenario changed: a"] and v["unexpected_passes"] == []


def test_a_fresh_babble_recording_is_clean_only_with_v2_headers_no_error_and_nothing_unmeasured(tmp_path):
    J = _job()
    from mimo_bridge import observer_record as OR
    summary = tmp_path / "random_orders.json"
    world = tmp_path / "random_orders" / "observer" / "world"
    for b in (0, 1):
        (world / ("b%d" % b)).mkdir(parents=True)
        (world / ("b%d" % b) / "header_x.json").write_text(json.dumps(dict(version="observer_world_v2",
                                                                            fields=list(OR.FIELDS))), encoding="utf-8")
    summary.write_text(json.dumps(dict(recordings=[dict(body=0, records=2500), dict(body=1, records=2500)],
                                       unmeasured_caps={})), encoding="utf-8")
    assert J.baseline_clean(str(summary)) == []
    (world / "b1" / "header_x.json").write_text(json.dumps(dict(version="observer_world_v1")), encoding="utf-8")
    (world / "b0" / "recorder_error.json").write_text("{}", encoding="utf-8")
    summary.write_text(json.dumps(dict(recordings=[dict(body=0, records=2500), dict(body=1, records=0, error="boom")],
                                       unmeasured_caps={"R04": "no normals"})), encoding="utf-8")
    problems = J.baseline_clean(str(summary))
    assert len(problems) == 5, problems
    assert any("observer_world_v1" in p for p in problems) and any("recorder error" in p for p in problems)
    assert any("boom" in p for p in problems) and any("recorded nothing" in p for p in problems)
    assert any("unmeasured: R04" in p for p in problems)


def test_a_declared_failure_of_an_r04_or_h02_present_line_is_refused():
    """A detector that cannot fire on its own positive case must not be locked (the second review's finding
    1; the coordinator: R04 on grasp_lift_and_hold is never an expected failure): declaring such a line a
    failure excuses nothing and blocks by itself. An absent line may still be declared."""
    J = _job()
    declared = dict(locked_line_failures=[dict(case="grasp_lift_and_hold", line="R04:R present"),
                                          dict(case="q", line="R04 absent")],
                    scenario_changed=[],
                    code_line_failures=[dict(case="grasp_lift_and_hold", line="H02:R present"),
                                        dict(case="t", line="R04:R:box@own present")])
    rep = _report([_case("grasp_lift_and_hold", present={"R04:R": False, "H02:R": False}),
                   _case("q", absent={"R04": False})])
    code = dict(grasp_lift_and_hold={"H02:R present": False}, t={"R04:R:box@own present": False})
    v = J.classify(rep, declared, code)
    refused = "declared failure refused (a detector that cannot fire must not be locked): "
    assert set(v["unexpected_failures"]) == {
        "locked line: grasp_lift_and_hold: R04:R present", "locked line: grasp_lift_and_hold: H02:R present",
        "code line: grasp_lift_and_hold: H02:R present", "code line: t: R04:R:box@own present",
        refused + "grasp_lift_and_hold: H02:R present", refused + "grasp_lift_and_hold: R04:R present",
        refused + "t: R04:R:box@own present"}
    assert v["expected_failures_matched"] == ["locked line: q: R04 absent"] and v["blocks"] is True


def _summary_tree(root, kinds, version, fields):
    """A staged Saved/mimo_bridge: each kind's summary and its recording's two body headers."""
    folder = root / "observer" / "baselines" / "74302686aba93fca"
    for kind in kinds:
        for b in (0, 1):
            world = folder / kind / "observer" / "world" / ("b%d" % b)
            world.mkdir(parents=True)
            (world / "header_x.json").write_text(json.dumps(dict(version="observer_world_v2", fields=fields)),
                                                 encoding="utf-8")
        (folder / (kind + ".json")).write_text(json.dumps(dict(
            kind=kind, detector_version=version, start="saved_worlds", vision=True, checkpoint_step=415000,
            recordings=[dict(body=0, records=2500), dict(body=1, records=2500)], unmeasured_caps={},
            recording_dir=str(folder / kind))), encoding="utf-8")
    return folder


def test_a_candidate_tree_writes_a_staging_saved_and_promote_places_it_in_production(tmp_path, monkeypatch):
    """The qualification runs in a candidate worktree (the coordinator's decision of 2026-09-26): --saved
    names production's Saved there, the writing steps go to a staging Saved on D:, and after the verdict
    `promote` - run from production - places each clean staged recording where the tracker reads it."""
    J = _job()
    from mimo_bridge import observer_record as OR
    from mimo_bridge.tools.observer import daemon as DM
    prod, stage = tmp_path / "prod" / "mimo_bridge", tmp_path / "stage" / "mimo_bridge"
    monkeypatch.setattr(DM, "SAVED", str(prod))
    assert J.in_candidate(None) is False and J.in_candidate(str(prod)) is False
    assert J.in_candidate(str(tmp_path / "other" / "mimo_bridge")) is True
    monkeypatch.setattr(J, "CANDIDATE_SAVED", str(stage))
    assert J.stage_saved() == str(stage) and DM.SAVED == str(stage)
    monkeypatch.setattr(DM, "SAVED", str(prod))
    assert J.promote(str(stage)) == 3, "nothing staged yet"
    _summary_tree(stage, J.BASELINE_KINDS, DM.DETECTOR_VERSION, list(OR.FIELDS))
    assert J.promote(str(stage)) == 0
    for kind in J.BASELINE_KINDS:
        placed = json.loads((prod / "observer" / "baselines" / J.SIG / (kind + ".json")).read_text(encoding="utf-8"))
        assert placed["promoted_from"].endswith(kind + ".json") and placed["kind"] == kind
        assert (prod / "observer" / "baselines" / J.SIG / kind / "observer" / "world" / "b1" / "header_x.json").exists()
        assert (stage / "observer" / "baselines" / J.SIG / kind).is_dir(), "the staged copy stays"


def test_promote_refuses_a_staged_recording_of_other_detectors(tmp_path, monkeypatch):
    J = _job()
    from mimo_bridge import observer_record as OR
    from mimo_bridge.tools.observer import daemon as DM
    prod, stage = tmp_path / "prod" / "mimo_bridge", tmp_path / "stage" / "mimo_bridge"
    monkeypatch.setattr(DM, "SAVED", str(prod))
    _summary_tree(stage, J.BASELINE_KINDS, "observer_detectors_v8", list(OR.FIELDS))
    assert J.promote(str(stage)) == 1
    assert not (prod / "observer" / "baselines" / J.SIG / "random_orders.json").exists()


def test_from_a_candidate_tree_the_after_verdict_steps_refuse(tmp_path):
    """adopt, rates and promote run from production after the verdict, never from a candidate tree (--saved
    naming another tree's Saved); check() reads production's kept baselines where they lie."""
    J = _job()
    saved = tmp_path / "prod" / "mimo_bridge"
    (saved / "observer" / "baselines" / J.SIG).mkdir(parents=True)
    for step in ("adopt", "rates", "promote"):
        assert J.main([step, "--saved", str(saved)]) == 3, step


def test_a_digested_case_that_found_nothing_is_judged_not_skipped(tmp_path):
    """RunDigest writes no detections file when it finds nothing: such a case has zero detections, so its
    absent lines pass and its present lines fail; only a case never recorded and digested is not judged
    (20 of the merged tree's 91 cases find nothing in its dry run)."""
    J = _job()

    class _V:
        CASES = [("quiet", None, [], ["H04"], {}), ("wanted", None, ["R04:R"], [], {}), ("never", None, [], ["H04"], {}),
                 ("found", None, ["H04:R"], ["R04"], {})]

        @staticmethod
        def _match(d, want):
            return d["cap"] == want.partition(":")[0]
    for name in ("quiet", "wanted", "found"):
        (tmp_path / name / "observer" / "analysis").mkdir(parents=True)
        (tmp_path / name / "observer" / "analysis" / "progress.json").write_text("{}", encoding="utf-8")
    (tmp_path / "found" / "observer" / "analysis" / "detections.jsonl").write_text(
        json.dumps(dict(cap="H04", sub="R")) + "\n", encoding="utf-8")
    code = J.code_lines(_V, str(tmp_path))
    assert code == {"quiet": {"H04 absent": True}, "wanted": {"R04:R present": False}, "never": None,
                    "found": {"H04:R present": True, "R04 absent": True}}


def test_the_gate_judges_each_scripted_truth_value_of_the_code(tmp_path):
    """The code's truth values are gate lines (the observer candidate's record, 2026-09-26): a first detection that
    reads the scripted value within validate.TRUTH_TOLERANCE meets its line, one that reads it 10% off or reads
    nothing fails it, and classify counts the failure as an undeclared code line. Before, code_lines read the
    present and absent lines only, so a wrong height passed the gate."""
    from mimo_bridge.tools.observer import validate as V
    J = _job()

    class _V:
        CASES = [("good", None, ["R05"], [], {"R05.height_cm": 3.0}), ("off", None, ["R05"], [], {"R05.height_cm": 3.0}),
                 ("blind", None, ["R05"], [], {"R05.height_cm": 3.0})]
        _truth_errors = staticmethod(V._truth_errors)

        @staticmethod
        def _match(d, want):
            return d["cap"] == want.partition(":")[0]
    for name, height in (("good", 2.99), ("off", 3.3), ("blind", None)):
        a = tmp_path / name / "observer" / "analysis"
        a.mkdir(parents=True)
        measures = {} if height is None else dict(height_cm=height)
        (a / "detections.jsonl").write_text(json.dumps(dict(cap="R05", sub="box", measures=measures)) + "\n",
                                            encoding="utf-8")
    code = J.code_lines(_V, str(tmp_path))
    assert code["good"] == {"R05 present": True, "R05.height_cm 3.0 truth": True}
    assert code["off"]["R05.height_cm 3.0 truth"] is False
    assert code["blind"]["R05.height_cm 3.0 truth"] is False
    v = J.classify(_report([_case(n, approved=False) for n in ("good", "off", "blind")]),
                   dict(locked_line_failures=[], scenario_changed=[], code_line_failures=[]), code)
    assert v["unexpected_failures"] == ["code line: blind: R05.height_cm 3.0 truth",
                                        "code line: off: R05.height_cm 3.0 truth"] and v["blocks"] is True


def test_a_case_left_out_of_the_lock_is_reported_and_never_judged(tmp_path):
    """validate.NOT_LOCKED (P02 since 2026-09-26: the body cannot roll itself yet): the gate's code_lines leaves
    such a case out, so its failing line blocks nothing, and not_locked_lines reports it with its reason."""
    J = _job()

    class _V:
        CASES = [("roll", None, ["P02"], [], {}), ("reach", None, ["R01"], [], {})]
        NOT_LOCKED = {"roll": "not locked: the body cannot roll itself yet (trunk authority)"}

        @staticmethod
        def _match(d, want):
            return d["cap"] == want.partition(":")[0]
    for name in ("roll", "reach"):
        a = tmp_path / name / "observer" / "analysis"
        a.mkdir(parents=True)
        (a / "progress.json").write_text("{}", encoding="utf-8")
    (tmp_path / "reach" / "observer" / "analysis" / "detections.jsonl").write_text(
        json.dumps(dict(cap="R01", sub="R")) + "\n", encoding="utf-8")
    code = J.code_lines(_V, str(tmp_path))
    assert code == {"reach": {"R01 present": True}}
    assert J.not_locked_lines(_V, str(tmp_path)) == {
        "roll": dict(reason=_V.NOT_LOCKED["roll"], lines={"P02 present": False})}
    v = J.classify(_report([_case("reach", approved=False)]),
                   dict(locked_line_failures=[], scenario_changed=[], code_line_failures=[]), code)
    assert v["unexpected_failures"] == [] and v["blocks"] is False
