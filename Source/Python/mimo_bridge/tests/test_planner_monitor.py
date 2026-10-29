# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A goal/effort tradeoff must remain visible; a search invariant failure stops."""
import copy
import json
from pathlib import Path
import runpy

WATCH = runpy.run_path(str(Path(__file__).resolve().parents[4] / "docs/h84_launch_2026-09-18/hourly_watch.py"))


def event(step=10):
    return {"type": "latch", "step": step, "plan": {"optimizer": [{
        "depth": 1, "selection": "best_evaluated_per_body_v1", "stopped_by": "step_budget",
        "per_body": {
            "objective": {"start": [2.0, 2.0], "selected": [1.9, 1.8]},
            "goal": {"start": [1.0, 1.0], "selected": [1.1, 0.9]},
            "effort": {"start": [1.0, 1.0], "selected": [0.8, 0.9]},
        },
    }]}}


def test_total_objective_regression_stops_even_if_batch_average_improves():
    row = event()
    row["plan"]["optimizer"][0]["per_body"]["objective"]["selected"] = [2.1, 0.0]
    result = WATCH["planner_integrity"]([row])
    assert result["objective_regressions"] == 1
    assert result["status"] == "repair_required"
    assert "body 1" in result["stop_reason"]


def test_goal_regression_for_effort_is_counted_for_investigation():
    result = WATCH["planner_integrity"]([event()])
    assert result["stop_reason"] is None
    assert result["status"] == "tradeoff_investigation"
    assert result["goal_regressions_with_lower_effort"] == 1
    assert result["goal_improvements"] == 1
    assert result["body_comparisons"] == 2


def test_eye_search_cannot_hide_behind_same_step_body_search():
    row=event()
    row['plan']['eye_optimizer']=copy.deepcopy(row['plan']['optimizer'])
    row['plan']['eye_optimizer'][0]['per_body']['objective']['selected'][0]=2.01
    result=WATCH['planner_integrity']([row,copy.deepcopy(row)],required_schema=True,expected_bodies=2)
    assert result['searches']==2 and result['decision_events']==1
    assert result['body_comparisons']==4 and result['objective_regressions']==1
    assert result['stop_reason']


def test_nonfinite_and_missing_scores_are_integrity_failures():
    for bad in (None, float("nan")):
        row = event()
        row["plan"]["optimizer"][0]["per_body"]["objective"]["selected"][0] = bad
        assert WATCH["planner_integrity"]([row])["status"] == "repair_required"
    row = event()
    del row["plan"]["optimizer"][0]["per_body"]["goal"]
    assert WATCH["planner_integrity"]([row])["status"] == "repair_required"


def test_finite_fallback_does_not_hide_nonfinite_search_failure():
    row = event()
    row["plan"]["optimizer"][0]["stopped_by"] = "nonfinite_gradient"
    result = WATCH["planner_integrity"]([row])
    assert result["nonfinite_search_stops"] == 1
    assert result["status"] == "repair_required"


def test_duplicates_and_old_logs_do_not_create_false_counts():
    row = event()
    before = copy.deepcopy(row)
    before["plan"]["optimizer"][0].pop("selection")
    result = WATCH["planner_integrity"]([row, copy.deepcopy(row), before])
    assert result["searches"] == result["decision_events"] == 1
    assert result["body_comparisons"] == 2
    assert WATCH["planner_integrity"]([before])["status"] == "awaiting_scored_decisions"


def test_monitor_persists_exact_counts_and_reads_unlatched_plan_records(tmp_path):
    row = event()
    unlatched = {"type": "plan", "step": 20, "plan": copy.deepcopy(row["plan"])}
    (tmp_path / "causal_chain.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(unlatched) + "\n{", encoding="utf-8")
    result = WATCH["planner_monitor"](tmp_path)
    assert result["decision_events"] == 2
    assert result["body_comparisons"] == 4
    assert json.loads((tmp_path / "planner_monitor.json").read_text())["status"] == "tradeoff_investigation"


def test_reported_body_count_cannot_hide_missing_comparisons():
    row = event()
    row['bodies'] = 8
    result = WATCH['planner_integrity']([row])
    assert result['status'] == 'repair_required' and result['malformed_results'] == 1
    assert result['body_comparisons'] == 0
    row['bodies'] = 2
    assert WATCH['planner_integrity']([row])['stop_reason'] is None


def test_qualified_run_requires_contract_and_all_bodies_including_unlatched_plans(tmp_path):
    receipt = dict(planner_selection_contract='best_evaluated_per_body_v1',planner_bodies=8)
    (tmp_path/'continuation_receipt.json').write_text(json.dumps(receipt))
    row = dict(type='plan',step=10,plan=event()['plan'])
    (tmp_path/'causal_chain.jsonl').write_text(json.dumps(row)+'\n')
    assert WATCH['planner_monitor'](tmp_path)['status'] == 'repair_required'
    row['plan']['optimizer'][0].pop('selection')
    (tmp_path/'causal_chain.jsonl').write_text(json.dumps(row)+'\n')
    result = WATCH['planner_monitor'](tmp_path)
    assert result['status'] == 'repair_required'
    assert result['selection_contract_required'] and result['malformed_results'] == 1
    row['plan'].pop('optimizer')
    (tmp_path/'causal_chain.jsonl').write_text(json.dumps(row)+'\n')
    assert WATCH['planner_monitor'](tmp_path)['status'] == 'repair_required'
    # An exploratory latch with no selected plan is legitimate.
    (tmp_path/'causal_chain.jsonl').write_text(json.dumps(dict(type='latch',step=20,bodies=8))+'\n')
    assert WATCH['planner_monitor'](tmp_path)['stop_reason'] is None


def test_automatic_continuation_inherits_parent_contract_and_stops_at_explicit_override(tmp_path):
    parent=tmp_path/'parent';child=tmp_path/'child'
    parent.mkdir();child.mkdir()
    (parent/'continuation_receipt.json').write_text(json.dumps(dict(
        planner_selection_contract='best_evaluated_per_body_v1',planner_bodies=8)))
    (child/'continuation_receipt.json').write_text(json.dumps(dict(parent_run=str(parent))))
    (child/'causal_chain.jsonl').write_text(json.dumps(event())+'\n')
    result=WATCH['planner_monitor'](child)
    assert result['selection_contract_required'] and result['selection_contract_origin']==str(parent.resolve())
    assert result['status']=='repair_required'  # two scores cannot stand for eight bodies
    (child/'continuation_receipt.json').write_text(json.dumps(dict(parent_run=str(parent),
        planner_selection_contract='legacy')))
    result=WATCH['planner_monitor'](child)
    assert not result['selection_contract_required']
    assert result['selection_contract_origin']==str(child.resolve()) and not result['stop_reason']


def test_malformed_escaping_or_cyclic_ancestry_cannot_inherit_a_contract(tmp_path):
    experiments=tmp_path/'experiments';experiments.mkdir()
    child=experiments/'child';child.mkdir()
    outside=tmp_path/'outside';outside.mkdir()
    (outside/'continuation_receipt.json').write_text(json.dumps(dict(
        planner_selection_contract='best_evaluated_per_body_v1',planner_bodies=8)))
    for receipt in (dict(parent_run=str(outside)),dict(parent_run=str(child)),
                    dict(parent_run=123),dict(parent_run='../outside'),[]):
        (child/'continuation_receipt.json').write_text(json.dumps(receipt))
        result=WATCH['planner_monitor'](child)
        assert not result['selection_contract_required'] and result['selection_contract_origin'] is None
        assert result['selection_contract_error'] and result['status']=='repair_required'
    (child/'continuation_receipt.json').write_text('{')
    assert WATCH['planner_monitor'](child)['status']=='repair_required'


def hand_event():
    row=event();row['plan']['bodies']=2
    row['plan']['hand_optimizer']=dict(version='hand_joint_response_v1',goal='wrist_relative_finger_shape',
        calibration_sha256='e'*64,steps=20,horizon_ms=250,raw_joint_bound_deg=11.,
        first=[2.,0.],selected=[1.,0.],valid=[True,False],selected_iteration=[1,0],
        trace=[[2.,0.]]+[[1.,0.]]*20,stopped_by='completed')
    return row


def test_hand_stage_is_scored_separately_and_duplicate_plan_latch_is_counted_once():
    row=hand_event();row['reached_body']=True;other=copy.deepcopy(row);other['type']='plan'
    result=WATCH['planner_integrity'](iter([other,row,row]),required_schema=True,expected_bodies=2,
                                    required_hand=True,hand_calibration_sha256='e'*64)
    assert result['stop_reason'] is None
    assert result['hand']['searches']==1 and result['hand']['body_comparisons']==1
    assert result['hand']['improvements']==1
    assert result['hand']['latched_searches']==1 and result['hand']['latched_bodies']==[0]


def test_hand_metadata_cannot_replace_missing_required_sensory_optimizer():
    row=hand_event();row['plan']['sensory_converged']=True;row['plan']['optimizer']=[]
    result=WATCH['planner_integrity']([row],required_schema=True,expected_bodies=2,required_hand=True)
    assert result['stop_reason'] and result['body_comparisons']==0
    assert result['hand']['body_comparisons']==1 and result['status']=='repair_required'


def test_hand_stage_cannot_hide_worse_or_missing_per_body_selection():
    for mutation in ('regression','not_best','bad_index','nonfinite','missing_body','wrong_calibration','invalid_changed'):
        row=hand_event();hand=row['plan']['hand_optimizer']
        if mutation=='regression':hand['selected'][0]=3.
        if mutation=='not_best':hand['trace'][12]=[.5,0.]
        if mutation=='bad_index':hand['selected_iteration'][0]=0
        if mutation=='nonfinite':hand['trace'][5]=[float('nan'),0.]
        if mutation=='missing_body':hand['valid']=[True]
        if mutation=='wrong_calibration':hand['calibration_sha256']='f'*64
        if mutation=='invalid_changed':hand['selected_iteration'][1]=1
        result=WATCH['planner_integrity']([row],expected_bodies=2,required_hand=True,hand_calibration_sha256='e'*64)
        assert result['status']=='repair_required',mutation
        assert result['hand']['stop_reason'],mutation


def test_hand_stage_skips_are_explicit_and_do_not_claim_a_scored_comparison():
    for reason in ('no_valid_memory_goal','awaiting_native_joint_readings'):
        row=hand_event();row['plan']['hand_optimizer']={'stopped_by':reason}
        result=WATCH['planner_integrity']([row],expected_bodies=2,required_hand=True)
        assert result['stop_reason'] is None and result['hand']['skipped_events']==1
        assert result['hand']['body_comparisons']==0
    row=event()
    assert WATCH['planner_integrity']([row],expected_bodies=2,required_hand=True)['stop_reason']
    assert WATCH['planner_integrity']([row],expected_bodies=2)['stop_reason'] is None


def test_resumed_run_inherits_hand_contract_and_calibration_identity(tmp_path):
    parent=tmp_path/'parent';child=tmp_path/'child';parent.mkdir();child.mkdir()
    receipt=dict(planner_selection_contract='best_evaluated_per_body_v1',planner_bodies=2,
                 hand_selection_contract='hand_joint_response_v1',hand_calibration_sha256='e'*64)
    (parent/'continuation_receipt.json').write_text(json.dumps(receipt))
    (child/'continuation_receipt.json').write_text(json.dumps(dict(parent_run=str(parent))))
    (child/'causal_chain.jsonl').write_text(json.dumps(hand_event())+'\n')
    assert WATCH['planner_monitor'](child)['hand_selection_contract_required']
    assert WATCH['planner_monitor'](child)['hand']['body_comparisons']==1
    del receipt['hand_calibration_sha256']
    (parent/'continuation_receipt.json').write_text(json.dumps(receipt))
    assert WATCH['planner_monitor'](child)['stop_reason']
