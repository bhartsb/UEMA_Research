# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Held-out representation controls; every result remains separate from physical proof."""
import torch

from .losses import cross_level_consistency


@torch.no_grad()
def representation_probe(learner, examples):
    """Compare the learned consequence code with zero and mismatched intentions.

    All decoders see identical current state/history. Posterior codes do contain
    observed futures: this is a representation test, never a planner prediction
    score. The cloned-world ranking job measures predictions made without futures.
    """
    if len(examples) < 2:
        raise ValueError("A mismatched-intention control needs at least two examples")
    batch = learner.batch(examples)
    codes = learner.model.encode(batch)
    roots = {r: codes[r] for r in learner.tree.roots}
    conditions = dict(posterior=roots, zero={r: torch.zeros_like(v) for r, v in roots.items()},
                      mismatched={r: v.roll(1, 0) for r, v in roots.items()})
    rows, predictions = {}, {}
    weights = batch.change_weight or {}

    def per_example(values, name):
        # the numbers the future lesson grades (whole-body rows: trajectory.body_lesson_weights); all, before
        w = weights.get(name)
        return values.mean(-1) if w is None else (values * w).sum(-1) / w.sum()
    graded = [name for name in batch.change if name not in weights or weights[name].sum() > 0]
    for label, intentions in conditions.items():
        decoded, _ = learner.model.decode(batch.state, batch.history, intentions)
        predictions[label] = decoded
        rows[label] = dict(future_mse_by_node={name: per_example((decoded[name].change - batch.change[name]).square(),
                                                                 name).tolist() for name in graded},
                           cross_level_mse=float(cross_level_consistency(learner.tree, decoded, batch)))
    rows["persistence"] = {name: per_example(batch.change[name].square(), name).tolist() for name in graded}
    sensitivity = {name: float(per_example((predictions["posterior"][name].change
                                            - predictions["zero"][name].change).square(), name).mean())
                   for name in graded}
    return dict(authority="none", physical_qualification=False, examples=len(examples),
        body_episodes=[[ex.start.env_id, ex.start.episode_id] for ex in examples],
        controls=rows, predicted_action_sensitivity=sensitivity,
        units="mean squared normalized physical change; use the recorded scales to interpret channels",
        interpretation="hindsight representation control; no physical controllability claim")
