# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Identical saved-world comparisons through the existing execution harness."""
from __future__ import annotations

import copy
import math

import numpy as np

from lem.latent_body.metrics import prediction_ranking


def compare(session, candidates, score, *, steps, predicted_scores, repeat_noise,
            candidate_counts, group, quality, measure=None):
    """Run each command twice from one snapshot; restore caller state even on failure.

    `session` is gof_probe_harness.Session on a detached checkpoint world. `score`
    reads the final physical readout and returns a smaller-is-better physical error.
    All three planners must supply the same issued-command interface and horizon.
    This function never changes a launch flag or grants control authority.
    """
    from mimo_bridge import copied_execution as CE
    required = {"direct", "synergy", "latent"}
    if not required.issubset(candidates) or set(predicted_scores) != set(candidates):
        raise ValueError("Compare direct, synergy and latent with predictions for every candidate")
    if type(steps) is not int or steps < 1 or not math.isfinite(repeat_noise) or repeat_noise < 0:
        raise ValueError("Declare the horizon and measured repeat noise")
    if set(candidate_counts) != set(candidates) or any(type(v) is not int or v < 1 for v in candidate_counts.values()):
        raise ValueError("Record each planner's actual candidate count")
    if quality not in ("plumbing", "post_repair") or not group:
        raise ValueError("Name the data quality and held-out pose/body group")
    initial = session.snapshot()
    rows = {}
    try:
        for name, command in candidates.items():
            command = np.asarray(command, dtype=np.float32).copy()
            if command.shape != (1, session.n_handles, 9) or not np.isfinite(command).all():
                raise ValueError(f"Invalid issued command: {name}")
            outcomes = []
            for _ in range(2):
                trajectory = session.run_arm(initial, command.copy(), steps=steps)
                # Include actual expanded substeps, not only an endpoint comparison.
                outcomes.append(dict(trajectory=copy.deepcopy(trajectory),
                    executed_substeps=copy.deepcopy(session.last_motor_substeps),
                    hand_goal=copy.deepcopy(session.last_hand_joint_goal)))
            if not CE.equal_tree(outcomes[0], outcomes[1]):
                raise ValueError(f"Physical replay differs for {name}: {CE.first_difference(outcomes[0], outcomes[1])}")
            error = float(score(outcomes[0]["trajectory"][-1]))
            predicted = float(predicted_scores[name])
            if not math.isfinite(error) or not math.isfinite(predicted):
                raise ValueError("Non-finite physical or predicted score")
            rows[name] = dict(physical_error=error, predicted_error=predicted,
                candidates=candidate_counts[name], repeated_exactly=True,
                outcome=outcomes[0]["trajectory"][-1], replay=outcomes[0])
            if measure is not None:
                rows[name]["measurements"] = measure(command.copy(), copy.deepcopy(outcomes[0]["trajectory"][-1]))
        order = list(rows)
        ranking = prediction_ranking([rows[n]["predicted_error"] for n in order],
                                     [rows[n]["physical_error"] for n in order], noise=repeat_noise)
        return dict(authority="test_world_only", production_control=False, group=group,
                    data_quality=quality, horizon_steps=steps, candidates=rows, ranking=ranking,
                    ranking_order=order, measured_repeat_noise=repeat_noise)
    finally:
        session.restore(initial)
