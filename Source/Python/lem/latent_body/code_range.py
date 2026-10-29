# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The encoder's own code range, and trial codes drawn inside it (Oct 1, 2026).

Why: the readings tool, the causal-control test and the planner search drew trial codes in a BOX (each code dimension
independently within 2 standard deviations of the mean, or within a radius of zero). The code dimensions are
correlated, so the box reaches corners the encoder never visits: on the whole-body fit of Oct 1 the box codes lay a
median 4.9-26.6 Mahalanobis units from the encoder's own codes, which lie 2.2-2.5 units from their centre, and there a
digit's predicted movement and its decoded command agreed 0.21-0.47 against 0.43-0.71 on the encoder's own codes
(agents/latent_neighbor_commands_2026-10-01.md). The decoder has learned nothing off that cloud.

CodeRange measures, per part, the encoder's own codes on real rows: their mean, their covariance and each code's
Mahalanobis distance from the mean (sqrt((z - mean)^T cov^-1 (z - mean)): 0 = the mean code; a code of a Gaussian
cloud in 8 dimensions lies a median 2.7 units out). The RANGE is the QUANTILE (95th percentile) of the encoder's own
codes' distances. Trial code modes (SAMPLING):
  box       the earlier mode, kept selectable: drawn by the tools themselves exactly as before (not by this class)
  encoded   the default for whole-body hierarchies: real codes the encoder produced for other rows, only those inside
            the range (the pool); a row's own code can be excluded
  gaussian  drawn from the fitted mean and covariance, rejected beyond the range
Every tool records the mode and `report` of its trial codes: their distances against the range.
"""
from __future__ import annotations

import torch

SAMPLING = ("box", "encoded", "gaussian")
QUANTILE = 0.95
MIN_ROWS = 50


def resolve_sampling(requested, config):
    """'auto' -> 'encoded' for a whole-body hierarchy (configuration key body_rows), 'box' otherwise (the pilot keeps
    the earlier mode); any other value is returned after checking it."""
    if requested == "auto":
        return "encoded" if config.get("body_rows") else "box"
    if requested not in SAMPLING:
        raise ValueError(f"Unknown code sampling {requested!r}; choose auto or one of {SAMPLING}")
    return requested


class CodeRange:
    """codes: {part: [N, latent] real codes from model.encode}. Covariances get a ridge of `ridge` times their mean
    variance, so a direction the encoder never varies is measured as far, not as infinite."""

    def __init__(self, codes, *, quantile=QUANTILE, ridge=1e-8):
        if not 0 < quantile <= 1:
            raise ValueError("The range quantile lies in (0, 1]")
        self.quantile = float(quantile)
        self.codes, self.mean, self.chol, self.inverse, self.own, self.range, self.pool = {}, {}, {}, {}, {}, {}, {}
        for name, z in codes.items():
            z = z.detach().double().reshape(len(z), -1)
            if len(z) < max(MIN_ROWS, 2 * z.shape[1]):
                raise ValueError(f"Measure the code range of {name} on at least {max(MIN_ROWS, 2 * z.shape[1])} real "
                                 f"codes (got {len(z)})")
            mean = z.mean(0)
            cov = torch.cov(z.T).reshape(z.shape[1], z.shape[1])
            cov = cov + ridge * float(cov.diagonal().mean().clamp_min(1e-30)) * torch.eye(z.shape[1], dtype=z.dtype)
            self.codes[name], self.mean[name] = z, mean
            self.chol[name] = torch.linalg.cholesky(cov)
            self.inverse[name] = torch.cholesky_inverse(self.chol[name])
            d = self.distance(name, z)
            self.own[name] = d
            self.range[name] = float(torch.quantile(d, self.quantile))
            self.pool[name] = torch.nonzero(d <= self.range[name]).flatten()

    def distance(self, name, codes):
        """[...] Mahalanobis distance of codes [..., latent] from the encoder's mean code of part `name`."""
        x = torch.as_tensor(codes).detach().double() - self.mean[name]
        return (x @ self.inverse[name] * x).sum(-1).clamp_min(0).sqrt()

    def centre(self, name):
        """[1, latent] float: the encoder's mean code."""
        return self.mean[name].float()[None].clone()

    def sample(self, name, count, mode, generator, *, exclude=None):
        """[count, latent] float trial codes of part `name`, every one within the range. exclude: a row index of the
        measured codes whose own code is not drawn (encoded)."""
        if count < 0:
            raise ValueError("Ask for a non-negative number of codes")
        if mode == "encoded":
            pool = self.pool[name]
            if exclude is not None:
                pool = pool[pool != int(exclude)]
            if len(pool) == 0:
                raise ValueError(f"No real code of {name} inside its range")
            pick = pool[torch.randint(len(pool), (count,), generator=generator)]
            return self.codes[name][pick].float()
        if mode == "gaussian":
            return self._gaussian(name, count, generator).float()
        raise ValueError(f"CodeRange draws encoded or gaussian codes, not {mode!r} (the box is drawn by each tool)")

    def _gaussian(self, name, count, generator, rounds=1000):
        dim = len(self.mean[name])
        out, have = [], 0
        for _ in range(rounds):
            if have >= count:
                break
            eps = torch.randn(max(2 * (count - have), 16), dim, generator=generator, dtype=torch.float64)
            z = self.mean[name] + eps @ self.chol[name].T
            z = z[self.distance(name, z) <= self.range[name]]
            out.append(z)
            have += len(z)
        if have < count:
            raise ValueError(f"Could not draw {count} Gaussian codes of {name} inside its range")
        return torch.cat(out)[:count]

    def sample_joint(self, names, count, mode, generator):
        """{name: [count, latent]} codes drawn together for several parts (the planner's roots). encoded: every part's
        code from the SAME real row, chosen among the rows inside every part's range, so the roots keep the relation
        the encoder gives them; gaussian: each part from its own fitted cloud."""
        names = list(names)
        if mode == "gaussian":
            return {n: self.sample(n, count, mode, generator) for n in names}
        if mode != "encoded":
            raise ValueError(f"CodeRange draws encoded or gaussian codes, not {mode!r}")
        rows = {n: len(self.codes[n]) for n in names}
        if len(set(rows.values())) != 1:
            raise ValueError("Joint codes need every part measured on the same rows")
        inside = torch.ones(next(iter(rows.values())), dtype=torch.bool)
        for n in names:
            inside &= self.own[n] <= self.range[n]
        pool = torch.nonzero(inside).flatten()
        if len(pool) == 0:
            raise ValueError("No real row lies inside every part's range")
        pick = pool[torch.randint(len(pool), (count,), generator=generator)]
        return {n: self.codes[n][pick].float() for n in names}

    def report(self, name, codes):
        """The trial codes' Mahalanobis distances against the encoder's own: the range (QUANTILE of the encoder's own
        codes' distances), the encoder's own median, and the trial codes' median, 95th percentile, largest and share
        beyond the range."""
        d = self.distance(name, torch.as_tensor(codes).reshape(-1, len(self.mean[name])))
        return dict(range=self.range[name], range_quantile=self.quantile, own_median=float(self.own[name].median()),
                    trial_median=float(d.median()), trial_p95=float(torch.quantile(d, .95)), trial_max=float(d.max()),
                    beyond_range=float((d > self.range[name] * (1 + 1e-9)).double().mean()), trial_codes=int(len(d)))

    def receipt(self):
        """Per part: rows measured, the range, the encoder's own median distance, the pool size."""
        return {n: dict(rows=int(len(self.codes[n])), range=self.range[n], range_quantile=self.quantile,
                        own_median=float(self.own[n].median()), pool=int(len(self.pool[n])),
                        latent_dim=int(len(self.mean[n])))
                for n in self.codes}


def summarize_reports(reports):
    """One line over several report() dicts: the largest trial median and maximum, the largest share beyond range."""
    reports = [r for r in reports if r]
    if not reports:
        return None
    return dict(trial_median_max=max(r["trial_median"] for r in reports),
                trial_max=max(r["trial_max"] for r in reports),
                beyond_range_max=max(r["beyond_range"] for r in reports),
                range_min=min(r["range"] for r in reports), range_max=max(r["range"] for r in reports))

