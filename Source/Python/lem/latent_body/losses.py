# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Dynamics lessons only; wellness never enters representation learning."""
import math

import torch
import torch.nn.functional as F

from .hierarchy import CHANGE_DIM

LOSS_NAMES = ("future", "contact", "action", "child", "consistency", "latent")


def validate_weights(weights):
    if set(weights) != set(LOSS_NAMES):
        raise ValueError("Every dynamics loss needs an explicitly calibrated weight")
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in weights.values()):
        raise ValueError("Dynamics loss weights must be finite and positive")


def weighted_mse(prediction, target, weight=None):
    """Mean squared error; with `weight` (one weight per number, shared by every example) the weighted mean over
    the weighted numbers (BodyRows lesson weights). None = F.mse_loss, the earlier meaning."""
    if weight is None:
        return F.mse_loss(prediction, target)
    w = weight.to(prediction.dtype)
    return ((prediction - target).square() * w).sum() / (w.sum() * len(prediction))


def weighted_bce(logits, target, weight=None):
    if weight is None:
        return F.binary_cross_entropy_with_logits(logits, target)
    w = weight.to(logits.dtype)
    return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * w).sum() / (w.sum() * len(logits))


def _weights(batch, field):
    return None if batch is None else getattr(batch, field, None)


def cross_level_consistency(tree, predictions, batch=None):
    terms = []
    change_w, contact_w = _weights(batch, "consistency_weight"), _weights(batch, "contact_weight")
    for node in tree.nodes:
        for child in tree.children[node.name]:
            parent = predictions[node.name]
            indices = [tree.subtree[node.name].index(i) for i in tree.subtree[child]]
            parent_change = parent.change.reshape(len(parent.change), -1, CHANGE_DIM)[:, indices]
            w = None if change_w is None else change_w[child]
            if w is None or w.sum() > 0:
                terms.append(weighted_mse(parent_change.flatten(1), predictions[child].change, w))
            w = None if contact_w is None else contact_w[child]
            if w is None or w.sum() > 0:
                terms.append(weighted_mse(parent.contact_logits[:, indices].sigmoid(),
                                          predictions[child].contact_logits.sigmoid(), w))
    if not terms:
        return next(iter(predictions.values())).change.sum() * 0
    return torch.stack(terms).mean()


def dynamics_loss(model, batch, weights):
    validate_weights(weights)
    z = model.encode(batch)
    local = model.reconstruct(batch, z)
    decoded, _ = model.decode(batch.state, batch.history, {r: z[r] for r in model.tree.roots})
    terms = {name: [] for name in LOSS_NAMES}
    change_w, action_w, contact_w = (_weights(batch, f) for f in ("change_weight", "action_weight", "contact_weight"))
    # BOTH ROUTES MUST PREDICT (2026-09-26). A good posterior reconstruction does
    # not show that a parent's intention can make a child produce that outcome.
    for predictions in (local, decoded):
        for node in model.tree.nodes:
            name, pred = node.name, predictions[node.name]
            w = None if change_w is None else change_w[name]
            if w is None or w.sum() > 0:
                terms["future"].append(weighted_mse(pred.change, batch.change[name], w))
            w = None if contact_w is None else contact_w[name]
            if w is None or w.sum() > 0:
                terms["contact"].append(weighted_bce(pred.contact_logits, batch.contact[name], w))
            w = None if action_w is None else action_w[name]
            if pred.action.shape[-1] and (w is None or w.sum() > 0):
                terms["action"].append(weighted_mse(pred.action, batch.action[name], w))
            for child, intention in pred.children.items():
                terms["child"].append(F.mse_loss(intention, z[child].detach()))
        terms["consistency"].append(cross_level_consistency(model.tree, predictions, batch))
    terms["latent"] = [v.square().mean() for v in z.values()]
    zero = next(iter(z.values())).sum() * 0
    means = {name: torch.stack(values).mean() if values else zero for name, values in terms.items()}
    total = sum(means[name] * weights[name] for name in LOSS_NAMES)
    return total, means, decoded
