# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""probe_reslice: the activity-conditioned re-read of saved probe events.

The raw slice advantage averages over the probe episode's own weather (a quiet
episode reads low), so checkpoints could not be compared. The re-slice must be
a TRUE PARTITION of the same per-row advantages - gate-open plus gate-closed
rows recombine exactly to the raw mean - and must inherit _delta's vision-gate
mechanics: an event where both bodies have the gate shut contributes exactly
zero vision advantage (its own vision delta is zeroed AND its same-instant
decoy is the other body's zeroed vision delta).
"""
import os
import tempfile

import pytest
import torch

from mimo_bridge.tools.probe_reslice import reslice
from mimo_bridge.tools.tenure_probe_v2 import SENSES

D = 8
B = 2
N_OPEN, N_CLOSED = 6, 8


def _blob():
    torch.manual_seed(0)
    proj = torch.nn.Sequential(
        torch.nn.Linear(24 * 9, 256), torch.nn.GELU(),
        torch.nn.Linear(256, 3 * D))
    events = []
    for k in range(N_OPEN + N_CLOSED):
        open_ev = k < N_OPEN
        ev = {
            "order": torch.randn(B, 24, 9),
            "hv_b": torch.tensor([open_ev] * B),
            "hv_e": torch.tensor([open_ev] * B),
            "tb": {m: torch.randn(B, D) for m in SENSES},
            "te": {m: torch.randn(B, D) for m in SENSES},
        }
        events.append(ev)
    return {"events": events, "meta": {"d_model": D},
            "proj_in": 24 * 9, "order_proj": proj.state_dict()}


@pytest.fixture(scope="module")
def report():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "events.pt")
        torch.save(_blob(), p)
        yield reslice(p)


def test_partition_counts_and_gate_frac(report):
    total = (N_OPEN + N_CLOSED) * B
    assert report["n_rows"] == total
    assert report["gate_open_frac"] == pytest.approx(N_OPEN / (N_OPEN + N_CLOSED))
    for sense in ("vision", "tactile", "proprioception"):
        r = report[sense]
        assert r["gate_open"]["n"] + r["gate_closed"]["n"] == total


def test_conditioned_means_recombine_to_raw(report):
    # the re-slice is a partition of the SAME advantage vector, not a re-measure
    for sense in ("vision", "tactile", "proprioception"):
        r = report[sense]
        no, nc = r["gate_open"]["n"], r["gate_closed"]["n"]
        recombined = (r["gate_open"]["mean"] * no + r["gate_closed"]["mean"] * nc) / (no + nc)
        assert recombined == pytest.approx(r["raw_mean"], abs=1e-6)


def test_both_bodies_gated_event_gives_zero_vision_advantage(report):
    # _delta zeroes vision when the gate is shut; with the whole instant shut,
    # the same-instant decoy is zero too, so the advantage must be exactly 0
    assert report["vision"]["gate_closed"]["mean"] == pytest.approx(0.0, abs=1e-6)
    # the non-gated senses must NOT collapse to zero on those same rows
    assert abs(report["tactile"]["gate_closed"]["mean"]) > 1e-4
    assert abs(report["proprioception"]["gate_closed"]["mean"]) > 1e-4
