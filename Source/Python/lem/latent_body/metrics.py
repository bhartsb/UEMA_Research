# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Separate prediction measurements from repeated physical consequences."""
import math

import torch


def vector(values, name):
    value = torch.as_tensor(values, dtype=torch.float64).detach().cpu()
    if value.ndim != 1 or not len(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return value


def paired_gain(reference, candidate, *, noise):
    """Positive gain = candidate has less physical goal error on identical starts.

    The sign-test tail is exact under equally likely wins/losses. Ties inside the
    predeclared, measured repeat noise are excluded, never counted as wins.
    """
    a, b = vector(reference, "reference"), vector(candidate, "candidate")
    if a.shape != b.shape or not math.isfinite(noise) or noise < 0:
        raise ValueError("Paired scores require matching starts and measured nonnegative noise")
    gain = a - b
    wins, losses = int((gain > noise).sum()), int((gain < -noise).sum())
    n = wins + losses
    p = sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n if n else 1.0
    return dict(starts=len(gain), wins=wins, losses=losses, ties=len(gain) - n,
                mean_gain=float(gain.mean()), worst_gain=float(gain.min()),
                one_sided_sign_p=p, measured_noise=noise)


def prediction_ranking(predicted, physical, *, noise):
    p, y = vector(predicted, "predicted"), vector(physical, "physical")
    if p.shape != y.shape or not math.isfinite(noise) or noise < 0:
        raise ValueError("Ranking requires matching candidates and measured nonnegative noise")
    correct = wrong = tied = 0
    for i in range(len(p)):
        for j in range(i):
            dp, dy = float(p[i] - p[j]), float(y[i] - y[j])
            if abs(dy) <= noise or dp == 0:
                tied += 1
            elif dp * dy > 0:
                correct += 1
            else:
                wrong += 1
    return dict(correct_pairs=correct, wrong_pairs=wrong, unresolved_pairs=tied,
                predicted_best=int(p.argmin()), physical_best=int(y.argmin()))


def intervention_spread(predictions, outcomes):
    """Both arrays are candidates x physical features on the same declared scales."""
    p, y = torch.as_tensor(predictions), torch.as_tensor(outcomes)
    if p.shape != y.shape or p.ndim != 2 or len(p) < 2:
        raise ValueError("Interventions require matched predictions and outcomes for >=2 candidates")
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():
        raise ValueError("Non-finite intervention")
    return dict(predicted_variance=float(p.float().var(0, unbiased=False).mean()),
                physical_variance=float(y.float().var(0, unbiased=False).mean()),
                prediction_mse=float((p.float() - y.float()).square().mean()))


def acceptance(*, quality, off_identity, comparisons, alpha, held_out_groups,
               minimum_groups, substrate_valid, physical_interventions, evidence=None):
    """An evidence summary only. It cannot enable a production feature.

    Caller supplies declared test size and significance before running a trial;
    missing substrate, independent identity or held-out evidence fails closed.
    """
    if not 0 < alpha < 1 or minimum_groups < 1:
        raise ValueError("Declare the statistical line and required independent groups")
    checks = dict(post_repair=quality == "post_repair", off_identity=off_identity is True,
                  comparison=bool(comparisons) and all(c["one_sided_sign_p"] < alpha
                             and c["mean_gain"] > c["measured_noise"] for c in comparisons),
                  generalization=len(set(held_out_groups)) >= minimum_groups,
                  substrate=substrate_valid is True, physical_interventions=physical_interventions is True)
    # ALL PROPOSAL GATES (2026-09-26). A good paired goal score alone cannot qualify
    # a hierarchy. Each remaining gate needs its completed, reviewable receipt.
    evidence = evidence or {}
    for gate in ("representation", "distinctness", "ranking", "reachability",
                 "cross_level", "efficiency", "wellness"):
        item = evidence.get(gate, {})
        checks[gate] = item.get("passed") is True and bool(item.get("receipt"))
    return dict(checks=checks, eligible_for_review=all(checks.values()), control_authority=False)
