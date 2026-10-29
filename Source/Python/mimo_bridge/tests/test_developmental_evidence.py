# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Physical evidence cannot pass because a sensor exists or a frame looks good."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mimo_bridge.developmental_evidence import ABILITY_MEASUREMENTS, assess_comparison, assess_trial


def declaration():
    return {
        "ability_id": "gaze_control",
        "criteria": [
            {"metric": "target_angle_deg", "op": "le", "value": 5.0},
            {"metric": "target_visible", "op": "eq", "value": True},
        ],
        "minimum_dwell_s": 0.2,
        "maximum_sample_gap_s": 0.1,
        "calibration": {
            "repeatability_verified": True, "reachability_verified": True,
            "receipt": "fixture:independently-executed-geometry-and-replay",
            "measurement_noise": {"target_angle_deg": 0.01},
            "successful_reference": {"target_angle_deg": 0.0, "target_visible": True},
        },
        "meaningful_success_gain": 0.2,
        "minimum_independent_worlds": 6,
        "required_phases": ["acquisition"],
        "required_conditions": ["present"],
    }


def trace(success=True, seed=0, pair="a", arm="candidate", phase="acquisition", condition="present"):
    return {
        "ability_id": "gaze_control", "world_seed": seed, "pair_id": pair,
        "arm": arm, "phase": phase, "condition": condition,
        "time_s": [i / 10 for i in range(9)],
        "model_authored": [True] * 9, "external_assistance": [False] * 9,
        "task_valid": [True] * 9,
        "measurements": {"target_angle_deg": [0.0 if success else 45.0] * 9,
                         "target_visible": [True] * 9},
    }


def pairs(n=10, phases=("acquisition",), conditions=("present",), baseline=False):
    trials = [trace(success, seed, arm=arm, phase=phase, condition=condition)
            for phase in phases for condition in conditions for seed in range(n)
            for arm, success in (("candidate", True), ("baseline", baseline))]
    for t in trials:
        if t["condition"] == "interrupted":
            t["interruption_time_s"] = 0.2
        if t["condition"] == "removed":
            t["measurements"]["target_visible"] = [False] * 9
            t["measurements"]["new_toy_directed_movement"] = [False] * 9
    return trials


def test_continuous_dwell_and_time_are_scored_not_frame_count():
    t = trace()
    t["measurements"]["target_angle_deg"] = [45, 0, 45, 0, 45, 0, 45, 0, 45]
    result = assess_trial(t, declaration())
    assert result["state"] == "observed_failure"
    assert result["successful_duration_s"] == 0
    t["measurements"]["target_angle_deg"][3:6] = [0, 0, 0]
    result = assess_trial(t, declaration())
    assert result["state"] == "observed_success"
    assert result["longest_success_s"] == pytest.approx(0.2)


@pytest.mark.parametrize("field", ["repeatability_verified", "reachability_verified", "receipt"])
def test_missing_physics_evidence_never_becomes_behavior_failure(field):
    d = declaration()
    d["calibration"][field] = False if field != "receipt" else ""
    assert assess_trial(trace(), d)["state"] == "unassessable"


def test_unreachable_declared_reference_is_rejected():
    d = declaration()
    d["calibration"]["successful_reference"]["target_angle_deg"] = 10
    assert "cannot attain" in assess_trial(trace(), d)["reason"]


@pytest.mark.parametrize("corruption", ["nan", "missing", "mismatched", "duplicate_time", "sparse", "invalid_task", "assisted"])
def test_invalid_traces_fail_closed(corruption):
    t = trace()
    if corruption == "nan":
        t["measurements"]["target_angle_deg"][4] = float("nan")
        t["measurements"]["target_visible"][4] = False
    elif corruption == "missing":
        del t["measurements"]["target_angle_deg"]
    elif corruption == "mismatched":
        t["task_valid"].pop()
    elif corruption == "duplicate_time":
        t["time_s"][3] = t["time_s"][2]
    elif corruption == "sparse":
        t["time_s"] = [i * 10 for i in range(9)]
    elif corruption == "invalid_task":
        t["task_valid"][2] = False
    else:
        t["external_assistance"][2] = True
    result = assess_trial(t, declaration())
    assert result["state"] == "unassessable"
    assert result["success"] is None


def test_candidate_cannot_borrow_motion_from_forced_commands():
    t = trace()
    t["model_authored"][0] = False
    assert assess_trial(t, declaration())["state"] == "unassessable"
    t["model_authored"] = [False] * 9
    t["arm"] = "baseline"
    assert "provenance" in assess_trial(t, declaration())["reason"]
    t["reference_action_source"] = "pose_hold"
    assert assess_trial(t, declaration())["state"] == "observed_success"


def test_interruption_does_not_reuse_prechange_success():
    t = trace()
    t["interruption_time_s"] = 0.5
    t["measurements"]["target_angle_deg"][5:] = [45] * 4
    result = assess_trial(t, declaration())
    assert result["longest_success_s"] >= 0.2
    assert result["state"] == "observed_failure"
    assert result["correction_latency_s"] is None
    t["measurements"]["target_angle_deg"][6:] = [0] * 3
    result = assess_trial(t, declaration())
    assert result["state"] == "observed_success"
    assert result["correction_latency_s"] == pytest.approx(0.3)


@pytest.mark.parametrize("corruption", ["missing_baseline", "duplicate", "wrong_world", "wrong_condition", "invalid_trace"])
def test_unmatched_or_corrupt_comparisons_fail_closed(corruption):
    ts = pairs()
    if corruption == "missing_baseline":
        ts.pop()
    elif corruption == "duplicate":
        ts.append(deepcopy(ts[0]))
    elif corruption == "wrong_world":
        ts[1]["world_seed"] = 999
    elif corruption == "wrong_condition":
        ts[1]["condition"] = "made_up"
    else:
        ts[1]["measurements"]["target_angle_deg"][0] = float("nan")
    result = assess_comparison(ts, declaration())
    assert result["state"] == "unassessable"
    assert result["capability_pass"] is False


def test_many_repeats_of_one_world_are_not_independent_evidence():
    ts = pairs()
    for t in ts:
        t["pair_id"] = str(t["world_seed"])
        t["world_seed"] = 0
    assert "independent worlds" in assess_comparison(ts, declaration())["reason"]


def test_real_negative_is_reported_without_promoting_low_error_to_gain():
    result = assess_comparison(pairs(baseline=True), declaration())
    assert result["state"] == "comparison_failed"
    assert result["cells"][0]["mean_success_gain"] == 0


def test_acquisition_only_positive_cannot_pass_whole_capability():
    result = assess_comparison(pairs(), declaration())
    assert result["state"] == "comparison_pass"
    assert result["capability_pass"] is False
    assert set(result["missing_causal_conditions"]) == {"relocated", "removed", "interrupted"}


def full_declaration():
    d = declaration()
    d["required_phases"] = ["acquisition", "retention", "transfer"]
    d["required_conditions"] = ["present", "relocated", "removed", "interrupted"]
    d["condition_criteria"] = {"removed": [
        {"metric": "target_visible", "op": "eq", "value": False},
        {"metric": "new_toy_directed_movement", "op": "eq", "value": False},
    ]}
    d["condition_goal_fraction"] = {"removed": 0.9}
    d["condition_modes"] = {"removed": "absolute_outcome"}
    d["absolute_success_targets"] = {"removed": 0.8}
    d["evidence_scope"] = {key: {"verified": True, "receipt": f"fixture:{key}"}
                           for key in ("learning_comparison", "retention_without_training", "held_out_transfer")}
    return d


def test_complete_comparison_requires_independent_scope_receipts():
    d = full_declaration()
    ts = pairs(phases=d["required_phases"], conditions=d["required_conditions"])
    result = assess_comparison(ts, d)
    assert result["capability_pass"] is True
    del d["evidence_scope"]["held_out_transfer"]
    result = assess_comparison(ts, d)
    assert result["state"] == "comparison_pass"
    assert result["capability_pass"] is False
    assert result["missing_evidence_receipts"] == ["held_out_transfer"]


def test_phase_labels_do_not_replace_causal_interventions():
    d = full_declaration()
    d["required_conditions"] = ["present"]
    result = assess_comparison(pairs(phases=d["required_phases"]), d)
    assert result["state"] == "comparison_pass"
    assert result["capability_pass"] is False


def test_removal_can_require_correct_abstention_when_both_arms_succeed():
    d = declaration()
    d["required_conditions"] = ["removed"]
    d["condition_modes"] = {"removed": "absolute_outcome"}
    d["absolute_success_targets"] = {"removed": 0.8}
    d["condition_criteria"] = {"removed": [
        {"metric": "target_visible", "op": "eq", "value": False},
        {"metric": "new_toy_directed_movement", "op": "eq", "value": False},
    ]}
    ts = pairs(conditions=("removed",), baseline=True)
    for t in ts:
        t["measurements"]["target_visible"] = [False] * 9
        t["measurements"]["new_toy_directed_movement"] = [False] * 9
    result = assess_comparison(ts, d)
    assert result["state"] == "comparison_pass"
    assert result["cells"][0]["mean_success_gain"] == 0
    assert result["cells"][0]["comparison_mode"] == "absolute_outcome"
    assert result["capability_pass"] is False


def test_ties_on_meaningful_line_cannot_be_discarded_to_manufacture_significance():
    d = declaration()
    d["meaningful_success_gain"] = 0.5
    ts = []
    # 10 positive worlds and 90 worlds exactly on the null line have median
    # gain 0.5, not greater. Removing the 90 ties produces a false pass.
    for seed in range(100):
        for rep in range(2):
            ts += [trace(True, seed, pair=str(rep)),
                   trace(False if seed < 10 or rep == 0 else True, seed, pair=str(rep), arm="baseline")]
    result = assess_comparison(ts, d)
    assert result["state"] == "comparison_failed"
    assert result["cells"][0]["median_test_p"] > 0.9


def test_all_twenty_five_abilities_have_measured_contracts_matching_registry():
    repository = Path(__file__).resolve().parents[4]
    registry = json.loads((repository / "docs/gain_function_2026-09-13/requirements.json").read_text(encoding="utf-8"))
    assert len(ABILITY_MEASUREMENTS) == 25
    assert set(ABILITY_MEASUREMENTS) == {g["id"] for g in registry["abilities"]}


def test_rolling_requires_rotation_not_only_an_already_prone_pose():
    d = declaration()
    d["ability_id"] = "roll_over"
    d["criteria"] = [
        {"metric": "body_ventral_up_dot", "op": "le", "value": -0.8},
        {"metric": "body_supported", "op": "eq", "value": True},
        {"metric": "body_rotation_from_start_deg", "op": "ge", "value": 150},
    ]
    d["calibration"].pop("successful_reference")
    t = trace()
    t["ability_id"] = "roll_over"
    t["measurements"] = {"body_ventral_up_dot": [-1] * 9, "body_supported": [True] * 9,
                         "body_rotation_from_start_deg": [0] * 9}
    assert assess_trial(t, d)["state"] == "observed_failure"
    t["measurements"]["body_rotation_from_start_deg"] = [160] * 9
    assert assess_trial(t, d)["state"] == "observed_success"


def test_removed_label_does_not_prove_target_was_removed():
    t = trace(condition="removed")
    assert "still contains a visible target" in assess_trial(t, declaration())["reason"]


def test_interrupted_label_requires_actual_intervention_time():
    assert "no measured intervention time" in assess_trial(trace(condition="interrupted"), declaration())["reason"]


def test_short_abstention_does_not_satisfy_mostly_stopping():
    d = full_declaration()
    t = pairs(conditions=("removed",))[0]
    t["measurements"]["new_toy_directed_movement"][4:] = [True] * 5
    result = assess_trial(t, d)
    assert result["longest_success_s"] >= d["minimum_dwell_s"]
    assert result["state"] == "observed_failure"
