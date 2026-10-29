# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Assess declared physical outcomes; do not infer intention from activity telemetry.

Inputs are measured world traces, not learner rewards. A trial result describes one
observed outcome. Comparison results require independent matched worlds, controls,
and explicitly declared acquisition/retention/transfer scope. Missing measurements
produce ``unassessable`` rather than a fabricated failure or pass.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence


# These are recorder contracts, not default target values or a motor curriculum.
# All numeric physical tolerances must come from the experiment declaration.
ABILITY_MEASUREMENTS: dict[str, tuple[str, ...]] = {
    "self_world_partition": ("self_cause_accuracy",),
    "proprioceptive_body_map": ("body_location_error_m",),
    "action_self_recognition": ("action_identification_accuracy",),
    "gaze_control": ("target_angle_deg", "target_visible"),
    "head_control": ("head_angle_error_deg", "head_supported_without_help"),
    "visual_self_recognition": ("visual_self_cause_accuracy",),
    "weight_shift": ("support_shift_error_m", "support_stable"),
    "anticipation": ("anticipation_lead_s", "predicted_effect_prevented"),
    "reach_to_target": ("target_distance_m", "target_contact"),
    "deliberate_strike": ("hand_target_contact", "target_effect"),
    "orient_to_sound": ("target_angle_deg", "auditory_only_target"),
    "head_lift_and_look": ("head_lift_m", "target_angle_deg", "target_visible"),
    "roll_over": ("body_ventral_up_dot", "body_supported", "body_rotation_from_start_deg"),
    "prone_prop": ("torso_height_m", "arm_support_contact", "body_ventral_up_dot"),
    "grasp": ("object_lift_m", "hand_object_contact", "object_attached"),
    "hold_under_slip": ("slip_challenge", "object_retained", "excessive_grip_force"),
    "place_release": ("object_support_contact", "hand_object_contact", "placement_error_m"),
    "bimanual_hold": ("left_object_contact", "right_object_contact", "object_lift_m"),
    "sit_balance": ("sitting_posture", "support_stable", "body_supported"),
    "crawl": ("travel_distance_m", "target_distance_m", "floor_locomotion"),
    "kneel": ("knee_support_contact", "support_stable", "kneeling_posture"),
    "proto_tool_use": ("tool_target_contact", "retrieval_distance_m", "direct_hand_target_contact"),
    "pull_to_stand": ("height_gain_m", "support_grasp", "upright_posture"),
    "stand": ("upright_posture", "feet_support_only", "support_stable"),
    "cruise_first_step": ("upright_posture", "travel_distance_m", "step_count"),
}

REPRESENTATIONAL_ABILITIES = {
    "self_world_partition", "proprioceptive_body_map", "action_self_recognition",
    "visual_self_recognition",
}

CAUSAL_CONDITIONS = {
    "self_world_partition": {"contingent", "external_replay"},
    "proprioceptive_body_map": {"vision_masked", "joint_perturbed"},
    "action_self_recognition": {"contingent", "action_swapped"},
    "visual_self_recognition": {"contingent", "external_replay", "viewpoint_changed"},
}
DEFAULT_CAUSAL_CONDITIONS = {"present", "relocated", "removed", "interrupted"}


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("measurement must be a numeric scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite number")
    return value


def _criterion(value: Any, criterion: Mapping[str, Any]) -> bool:
    operator, target = criterion["op"], criterion["value"]
    if operator == "eq":
        if isinstance(target, bool):
            if not isinstance(value, bool):
                raise ValueError("boolean criterion requires boolean observations")
            return value is target
        return _number(value) == _number(target)
    value = _number(value)
    if operator == "ge":
        return value >= _number(target)
    if operator == "le":
        return value <= _number(target)
    if operator == "range":
        lower, upper = (_number(x) for x in target)
        if lower > upper:
            raise ValueError("inverted criterion range")
        return lower <= value <= upper
    raise ValueError(f"unsupported criterion operator {operator!r}")


def _calibration(declaration: Mapping[str, Any]) -> tuple[str, list[Mapping[str, Any]], float]:
    ability = declaration["ability_id"]
    if ability not in ABILITY_MEASUREMENTS:
        raise ValueError("unknown ability")
    calibration = declaration["calibration"]
    if calibration.get("repeatability_verified") is not True:
        raise ValueError("physical repeatability has not been verified")
    if calibration.get("reachability_verified") is not True:
        raise ValueError("physical reachability has not been verified")
    if not isinstance(calibration.get("receipt"), str) or not calibration["receipt"].strip():
        raise ValueError("calibration receipt is missing")
    noise = calibration["measurement_noise"]
    if isinstance(noise, Mapping):
        if not noise or any(_number(v) < 0 for v in noise.values()):
            raise ValueError("invalid measurement noise")
    elif _number(noise) < 0:
        raise ValueError("invalid measurement noise")
    criteria = declaration["criteria"]
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("no declared outcome criteria")
    names = [c["metric"] for c in criteria]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("invalid criterion metric")
    if not set(ABILITY_MEASUREMENTS[ability]).issubset(names):
        raise ValueError("criteria omit required physical outcomes")
    reference = calibration.get("successful_reference")
    if reference is not None and not all(_criterion(reference[c["metric"]], c) for c in criteria):
        raise ValueError("calibrated reference cannot attain the declared outcome")
    dwell = _number(declaration["minimum_dwell_s"])
    if dwell <= 0:
        raise ValueError("minimum dwell must be positive")
    return ability, criteria, dwell


def assess_trial(trace: Mapping[str, Any], declaration: Mapping[str, Any]) -> dict[str, Any]:
    """Judge sustained, observed outcomes using strictly increasing simulation time.

    Each measurement array has the same length as ``time_s``. An interval counts
    only when both of its recorded endpoints meet every criterion and have valid
    unassisted authorship. This avoids awarding an entire interval for a one-frame
    contact. All assisted trials remain unassessable for autonomous behavior.
    """
    result: dict[str, Any] = {
        "ability_id": trace.get("ability_id"), "world_seed": trace.get("world_seed"),
        "pair_id": trace.get("pair_id"), "phase": trace.get("phase"),
        "condition": trace.get("condition"), "arm": trace.get("arm"),
        "state": "unassessable", "success": None,
    }
    try:
        ability, criteria, dwell = _calibration(declaration)
        criteria = declaration.get("condition_criteria", {}).get(trace.get("condition"), criteria)
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("condition has no declared physical criteria")
        if trace["ability_id"] != ability:
            raise ValueError("trace and declaration refer to different abilities")
        if trace.get("world_seed") is None or not trace.get("pair_id"):
            raise ValueError("world identity or matched-pair identity is missing")
        if trace.get("phase") not in {"acquisition", "retention", "transfer"}:
            raise ValueError("unknown evidence phase")
        times = [_number(t) for t in trace["time_s"]]
        n = len(times)
        if n < 2 or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("need at least two strictly increasing simulation times")
        arrays = {name: list(trace[name]) for name in ("model_authored", "external_assistance", "task_valid")}
        measurements = trace["measurements"]
        for name, values in arrays.items():
            if len(values) != n or any(not isinstance(x, bool) for x in values):
                raise ValueError(f"invalid {name} mask")
        if any(arrays["external_assistance"]):
            raise ValueError("external assistance occurred in the scored trial")
        if not all(arrays["task_valid"]):
            raise ValueError("the physical task was not valid throughout the scored trial")
        required_metrics = set(ABILITY_MEASUREMENTS[ability]) | {c["metric"] for c in criteria}
        for metric in required_metrics:
            values = measurements[metric]
            if len(values) != n:
                raise ValueError(f"measurement length mismatch for {metric}")
            if any(not isinstance(v, bool) and not math.isfinite(_number(v)) for v in values):
                raise ValueError(f"invalid measurement in {metric}")
        if trace.get("condition") == "removed" and "target_visible" in required_metrics:
            if any(measurements["target_visible"]):
                raise ValueError("removed-target trace still contains a visible target; score after removal")
        if trace.get("condition") == "interrupted" and trace.get("interruption_time_s") is None:
            raise ValueError("interrupted condition has no measured intervention time")
        for c in criteria:
            values = measurements[c["metric"]]
            if len(values) != n:
                raise ValueError(f"measurement length mismatch for {c['metric']}")
            # Check every value, including frames which do not meet other goals.
            for value in values:
                _criterion(value, c)
        # A decoded self-state test has no action-authorship requirement. It still
        # requires unassisted, validated physical examples and counterfactuals.
        reference = trace.get("arm") == "baseline"
        if reference and not all(arrays["model_authored"]) and trace.get("reference_action_source") not in {"pose_hold","replayed_commands","scripted_reference"}:
            raise ValueError("reference controller provenance is missing")
        authorship = [True] * n if ability in REPRESENTATIONAL_ABILITIES or reference else arrays["model_authored"]
        if not any(authorship):
            raise ValueError("no model-authored opportunity was measured")
        if ability not in REPRESENTATIONAL_ABILITIES and not reference and not all(authorship):
            raise ValueError("candidate action source changed within the scored trial")
        max_gap = _number(declaration.get("maximum_sample_gap_s", dwell))
        if not 0 < max_gap <= dwell:
            raise ValueError("maximum sample gap must be positive and no larger than the dwell")
        if any(b-a > max_gap + 1e-12 for a,b in zip(times,times[1:])):
            raise ValueError("trace is too sparse for the declared outcome duration")
        passing = [bool(authorship[i]) and all(_criterion(measurements[c["metric"]][i], c) for c in criteria) for i in range(n)]
        elapsed = times[-1]-times[0]
        total = longest = streak = 0.0
        first = None
        for i, dt in enumerate(b-a for a,b in zip(times,times[1:])):
            if passing[i] and passing[i+1]:
                streak += dt
                total += dt
                if first is None and streak >= dwell:
                    first = times[i+1]
                longest = max(longest, streak)
            else:
                streak = 0.0
        minimum_fraction = _number(declaration.get("condition_goal_fraction", {}).get(
            trace.get("condition"), declaration.get("minimum_goal_fraction", 0.0)))
        if not 0 <= minimum_fraction <= 1:
            raise ValueError("minimum goal fraction must lie between zero and one")
        success = longest + 1e-12 >= dwell and total/elapsed + 1e-12 >= minimum_fraction
        result.update(state="observed_success" if success else "observed_failure", success=success,
                      elapsed_s=elapsed, successful_duration_s=total, longest_success_s=longest,
                      goal_fraction=total/elapsed, first_confirmed_success_s=first,
                      criterion_meaning="All declared physical outcomes persisted together for the required duration.",
                      model_authored_frame_fraction=sum(arrays["model_authored"])/n,
                      calibration_receipt=declaration["calibration"]["receipt"],
                      scope="representation" if ability in REPRESENTATIONAL_ABILITIES else "physical_behavior")
        interruption = trace.get("interruption_time_s")
        if interruption is not None:
            interruption = _number(interruption)
            if not times[0] <= interruption < times[-1]:
                raise ValueError("interruption lies outside the scored trace")
            # Recompute success after the changed target, never reuse pre-change dwell.
            start = next(i for i,t in enumerate(times) if t >= interruption)
            streak = 0.0
            corrected = None
            for i in range(start,n-1):
                streak = streak + times[i+1]-times[i] if passing[i] and passing[i+1] else 0.0
                if streak + 1e-12 >= dwell:
                    corrected = times[i+1]-interruption
                    break
            result["correction_latency_s"] = corrected
            post_elapsed = times[-1]-times[start]
            post_total = sum(times[i+1]-times[i] for i in range(start,n-1) if passing[i] and passing[i+1])
            post_fraction = post_total/post_elapsed if post_elapsed > 0 else 0.0
            result["post_interruption_goal_fraction"] = post_fraction
            result["success"] = corrected is not None and post_fraction + 1e-12 >= minimum_fraction
            result["state"] = "observed_success" if result["success"] else "observed_failure"
        return result
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        result.update(state="unassessable",success=None,reason=str(exc))
        return result


def _sign_probability(positive: int, non_tied: int) -> float:
    if non_tied == 0:
        return 1.0
    return sum(math.comb(non_tied,k) for k in range(positive,non_tied+1)) / (2**non_tied)


def assess_comparison(trials: Sequence[Mapping[str, Any]], declaration: Mapping[str, Any]) -> dict[str, Any]:
    """Require matched independent worlds and declared practical outcome advantage.

    Conditions are assessed separately. For each phase/condition, compare candidate
    trials with their baseline using ``arm``. Repeated trials are averaged within
    world before the exact one-sided sign test. The test concerns the median
    world-level gain above the declared useful difference, not generic intelligence.
    """
    result: dict[str, Any] = {"ability_id": declaration.get("ability_id"), "state": "unassessable", "capability_pass": False}
    try:
        _calibration(declaration)
        phases=list(declaration["required_phases"])
        conditions=list(declaration["required_conditions"])
        if not phases or not conditions or len(set(phases))!=len(phases) or len(set(conditions))!=len(conditions):
            raise ValueError("phases and conditions must be nonempty and unique")
        if not set(phases).issubset({"acquisition","retention","transfer"}):
            raise ValueError("unknown evidence phase")
        gain=_number(declaration["meaningful_success_gain"])
        if not 0 < gain < 1:
            raise ValueError("useful success gain must lie strictly between zero and one")
        minimum=declaration["minimum_independent_worlds"]
        if isinstance(minimum,bool) or not isinstance(minimum,int) or minimum<2:
            raise ValueError("independent-world minimum must be an integer of at least two")
        alpha=_number(declaration.get("alpha",0.05))
        if not 0 < alpha < 1:
            raise ValueError("invalid alpha")
        rows=[]
        paired: dict[tuple[Any,...], dict[str,dict[str,Any]]] = defaultdict(dict)
        for trace in trials:
            assessed=assess_trial(trace,declaration)
            rows.append(assessed)
            if assessed["state"]=="unassessable":
                raise ValueError(f"invalid trial: {assessed['reason']}")
            if trace["phase"] not in phases or trace["condition"] not in conditions:
                raise ValueError("trial falls outside the declared comparison scope")
            arm=trace["arm"]
            if arm not in {"candidate","baseline"}:
                raise ValueError("comparison arm must be candidate or baseline")
            key=(trace["phase"],trace["condition"],trace["world_seed"],trace["pair_id"])
            if arm in paired[key]:
                raise ValueError("duplicate trial in a matched pair")
            paired[key][arm]=assessed
        if not paired:
            raise ValueError("no trials")
        by_world: dict[tuple[Any,...], list[float]]=defaultdict(list)
        successes_by_world: dict[tuple[Any,...], list[float]]=defaultdict(list)
        for key, arms in paired.items():
            if set(arms)!={"candidate","baseline"}:
                raise ValueError("unmatched candidate or baseline")
            by_world[key[:3]].append(float(arms['candidate']['success'])-float(arms['baseline']['success']))
            successes_by_world[key[:3]].append(float(arms['candidate']['success']))
        cells=[]
        # Family adjustment covers all conditions/phases judged in this declaration.
        cell_alpha=alpha/(len(phases)*len(conditions))
        for phase in phases:
            for condition in conditions:
                values=[sum(v)/len(v) for (p,c,w),v in by_world.items() if p==phase and c==condition]
                if len(values)<minimum:
                    raise ValueError(f"insufficient independent worlds for {phase}/{condition}")
                positive=sum(v>gain for v in values)
                # Ties remain in the denominator. Dropping them would let a
                # majority of worlds exactly on the null line become a pass.
                p_value=_sign_probability(positive,len(values))
                mean=sum(values)/len(values)
                passed=mean>=gain and p_value<=cell_alpha
                mode=declaration.get('condition_modes',{}).get(condition,'superiority')
                extra={}
                if mode=='absolute_outcome':
                    target=_number(declaration['absolute_success_targets'][condition])
                    if not 0<target<1:
                        raise ValueError('absolute success target must lie between zero and one')
                    rates=[sum(v)/len(v) for (p,c,w),v in successes_by_world.items() if p==phase and c==condition]
                    pos=sum(v>target for v in rates)
                    p_value=_sign_probability(pos,len(rates))
                    passed=sum(rates)/len(rates)>=target and p_value<=cell_alpha
                    extra=dict(absolute_success_target=target,world_candidate_success_rates=rates)
                elif mode!='superiority':
                    raise ValueError('unknown condition comparison mode')
                cells.append(dict(phase=phase,condition=condition,independent_worlds=len(values),world_success_differences=values,mean_success_gain=mean,meaningful_success_gain=gain,median_test_p=p_value,adjusted_alpha=cell_alpha,comparison_mode=mode,passed=passed,**extra))
        causal=CAUSAL_CONDITIONS.get(declaration['ability_id'],DEFAULT_CAUSAL_CONDITIONS)
        evidence_scope=declaration.get('evidence_scope',{})
        missing_receipts=[]
        for item in ('learning_comparison','retention_without_training','held_out_transfer'):
            evidence=evidence_scope.get(item,{})
            if evidence.get('verified') is not True or not isinstance(evidence.get('receipt'),str) or not evidence['receipt'].strip():
                missing_receipts.append(item)
        has_superiority=any(c['comparison_mode']=='superiority' for c in cells)
        complete_scope=(set(phases)=={'acquisition','retention','transfer'} and causal.issubset(conditions)
                        and not missing_receipts and has_superiority)
        all_pass=all(c['passed'] for c in cells)
        result.update(state='capability_pass' if all_pass and complete_scope else 'comparison_pass' if all_pass else 'comparison_failed',capability_pass=all_pass and complete_scope,cells=cells,trial_results=rows,scope=phases,missing_causal_conditions=sorted(causal-set(conditions)),missing_evidence_receipts=missing_receipts,has_superiority_control=has_superiority,meaning='Passed cells show the declared physical outcomes under matched controls. Full gain status additionally requires verified acquisition, retention, transfer and the causal intervention conditions.')
    except (KeyError,TypeError,ValueError,OverflowError) as exc:
        result.update(reason=str(exc),state='unassessable',capability_pass=False)
    return result
