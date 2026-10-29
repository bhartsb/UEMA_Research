# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Search root intentions using predicted consequences, without future-state access."""
from dataclasses import dataclass, field
import math

import torch


@dataclass
class LatentChoice:
    roots: dict
    predictions: dict
    scores: torch.Tensor
    index: int
    candidates: int
    sampling: str = "box"
    code_range: dict = field(default_factory=dict)


@torch.no_grad()
def search(model, state, history, score, *, candidates, radius=None, generator, sampling="box", code_range=None):
    """One observed body, one candidate budget. Smaller supplied scores are better.

    sampling "box" (the earlier search, the default): candidates drawn uniformly in [-radius, radius] per dimension,
    the zero intention included. Radius is a measured latent range, supplied by the caller; there is no default search
    scale that could confer control authority.
    sampling "encoded" / "gaussian" (Oct 1, 2026; lem.latent_body.code_range): candidates inside the encoder's own code
    range, measured by the caller (code_range, a CodeRange of the roots' real codes): the root codes of one real row
    each (encoded), or drawn from the fitted mean and covariance inside the range (gaussian); the encoder's mean code
    is candidate 0. The box reaches corners the encoder never visits (whole-body fit of Oct 1: box codes a median
    4.9-26.6 Mahalanobis units out, the encoder's own 2.2-2.5); there the decoder has learned nothing.
    With a code_range the choice records each root's candidates' distances against the range (code_range.report).
    `score` sees predictions, never outcomes of the future physical trial.
    """
    if type(candidates) is not int or candidates < 2:
        raise ValueError("Supply a candidate budget >= 2 and a measured positive latent radius")
    if sampling == "box":
        if radius is None or not math.isfinite(radius) or radius <= 0:
            raise ValueError("Supply a candidate budget >= 2 and a measured positive latent radius")
    elif sampling in ("encoded", "gaussian"):
        if code_range is None or not set(model.tree.roots) <= set(code_range.mean):
            raise ValueError(f"Sampling {sampling!r} needs the measured code range of every root")
    else:
        raise ValueError(f"Unknown sampling {sampling!r}")
    if any(value.shape[0] != 1 for value in (*state.values(), *history.values())):
        raise ValueError("Search each body's starting state separately")
    roots = {}
    if sampling == "box":
        for name in model.tree.roots:
            z = (2 * torch.rand(candidates, model.latent_dim, generator=generator) - 1) * radius
            z[0].zero_()
            roots[name] = z
    else:
        drawn = code_range.sample_joint(model.tree.roots, candidates - 1, sampling, generator)
        roots = {name: torch.cat((code_range.centre(name), drawn[name])) for name in model.tree.roots}
    ranges = {} if code_range is None else {name: code_range.report(name, roots[name][1:]) for name in roots}
    expanded = lambda values: {k: v.expand(candidates, -1) for k, v in values.items()}
    predictions, _ = model.decode(expanded(state), expanded(history), roots)
    scores = torch.as_tensor(score(predictions)).detach().cpu()
    if scores.shape != (candidates,) or not torch.isfinite(scores).all():
        raise ValueError("Every candidate requires one finite predicted score")
    index = int(scores.argmin())
    selected = {k: v[index:index + 1].clone() for k, v in roots.items()}
    chosen, _ = model.decode(state, history, selected)
    for name in ranges:
        ranges[name]["selected"] = float(code_range.distance(name, selected[name])[0])
    return LatentChoice(selected, chosen, scores.clone(), index, candidates, sampling, ranges)
