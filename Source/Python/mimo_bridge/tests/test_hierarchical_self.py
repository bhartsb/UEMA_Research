# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Level 2/3 hierarchical self, dark mode: entities must form from experience alone,
membership must find the wellness-linked state and score nothing under shuffled
wellness, and the module must be OUTSIDE both the gradient path and the behavior
path (dark means dark - the wire into wellness is a later, separately certified
switch).
"""
import logging
import os
import re

import numpy as np
import pytest
import torch

from mimo_bridge.hierarchical_self import HierarchicalSelfDark


def _stream(rng, n, d_model=16, sep=4.0):
    """Two world-regimes A/B as fused-token streams: distinct prototypes + noise."""
    pa = rng.standard_normal(2 * d_model) * sep
    pb = rng.standard_normal(2 * d_model) * sep
    frames, labels = [], []
    for i in range(n):
        proto = pa if (i // 50) % 2 == 0 else pb
        v = proto + rng.standard_normal(2 * d_model)
        frames.append(v.astype(np.float32))
        labels.append(0 if (i // 50) % 2 == 0 else 1)
    return np.stack(frames), np.array(labels)


def _observe(hs, vec, w, d_model=16):
    toks = {"a": torch.tensor(vec[:d_model]).unsqueeze(0),
            "b": torch.tensor(vec[d_model:]).unsqueeze(0)}
    hs.observe(toks, {"g": torch.tensor([[w]], dtype=torch.float32)})


def test_entities_form_and_separate_regimes():
    rng = np.random.default_rng(0)
    frames, labels = _stream(rng, 600)
    hs = HierarchicalSelfDark(["a", "b"], 16, n_slots=8)
    winners = []
    for v, lb in zip(frames, labels):
        _observe(hs, v, 0.0)
        k = hs._filled_cpu
        sim = torch.nn.functional.normalize(
            torch.tensor(v).unsqueeze(0), dim=-1) @ hs.centers[:k].T
        winners.append((int(sim.argmax()), lb))
    late = winners[300:]
    a_slots = {s for s, lb in late if lb == 0}
    b_slots = {s for s, lb in late if lb == 1}
    assert a_slots and b_slots and a_slots.isdisjoint(b_slots), \
        f"regimes must claim distinct slots (A {a_slots}, B {b_slots})"
    st = hs.stats()
    # novelty gating means slots open only for genuinely new states - two regimes
    # need at least two entities, and filling all 8 from noise would be the defect
    assert 2 <= st["slots_filled"] <= 8 and st["frames"] == 600


def test_membership_finds_the_wellness_linked_state_and_shuffle_scores_nothing():
    rng = np.random.default_rng(1)
    # the correlation estimator's noise scales with effective samples (stream length /
    # band window): 1200 frames left the shuffled control at ~0.2 by chance alone,
    # failing a 3x criterion the mechanism actually meets - so give it the samples
    frames, labels = _stream(rng, 3600)
    wellness = np.where(labels == 0, 1.0, -1.0) + rng.standard_normal(len(labels)) * 0.1

    def run(w_seq):
        hs = HierarchicalSelfDark(["a", "b"], 16, n_slots=8, m_l2=0.98, m_l3=0.995)
        for v, w in zip(frames, w_seq):
            _observe(hs, v, float(w))
        return hs

    hs = run(wellness)
    memb = hs.membership("l2")
    # which slots the A-regime occupies, by presence
    occA = torch.zeros(8)
    occB = torch.zeros(8)
    for v, lb in zip(frames[600:], labels[600:]):
        x = torch.nn.functional.normalize(torch.tensor(v).unsqueeze(0), dim=-1)
        p = torch.softmax(hs.temperature * (x @ hs.centers.T), dim=-1)[0]
        (occA if lb == 0 else occB).add_(p)
    a_mask = occA > occB
    gap = float(memb[a_mask].mean() - memb[~a_mask].mean())
    assert gap > 0.2, f"slots of the wellness-linked regime must carry higher " \
                      f"membership (gap {gap:.3f})"
    # the control that ships with it: shuffled wellness must score ~nothing
    hs_sh = run(rng.permutation(wellness))
    sh = float(hs_sh.membership("l2").abs().mean())
    st = float(memb.abs().mean())
    assert st > 3 * max(sh, 1e-6), \
        f"structured membership {st:.4f} must beat shuffled {sh:.4f} by 3x"


def test_outside_the_gradient_path():
    hs = HierarchicalSelfDark(["a"], 8, n_slots=4)
    assert list(hs.parameters()) == [], "no trainable weights, never in the optimizer"
    x = torch.randn(2, 8, requires_grad=True)
    w = torch.randn(2, 1, requires_grad=True)
    hs.observe({"a": x}, {"g": w})
    assert all(not b.requires_grad for b in hs.buffers())
    assert x.grad_fn is None and w.grad_fn is None


def test_dark_means_dark_in_the_model_wiring():
    """The model may construct, observe, save, load and report the module - and
    nothing else. Any other attribute use is a wire into behavior and fails here."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "lem_model.py"),
               encoding="utf-8").read()
    uses = set(re.findall(r"_hself\.(\w+)", src))
    allowed = {"observe", "stats", "state_dict", "load_state_dict"}
    assert uses <= allowed, f"dark mode must not touch behavior: {uses - allowed}"
    assert "observe" in uses, "the observation hook must exist (dead wire otherwise)"


def test_slot_count_migration_preserves_earned_evidence():
    """Changing n_slots on resume must carry the overlapping slots, not restart -
    Level-3 evidence takes 2,700,000 steps to earn on a 6,400-slot table
    (first_evidence_steps('l3', 6400)) and a warning does not give it back."""
    rng = np.random.default_rng(3)
    frames, labels = _stream(rng, 400)
    hs = HierarchicalSelfDark(["a", "b"], 16, n_slots=8, m_l2=0.98, m_l3=0.995)
    for v, lb in zip(frames, labels):
        _observe(hs, v, 1.0 if lb == 0 else -1.0)
    grown = HierarchicalSelfDark(["a", "b"], 16, n_slots=16, m_l2=0.98, m_l3=0.995)
    grown.load_state_dict(hs.state_dict(), strict=False)
    assert grown._filled_cpu == hs._filled_cpu
    assert torch.allclose(grown.centers[:8], hs.centers)
    assert torch.allclose(grown.membership("l2")[:8], hs.membership("l2"))
    shrunk = HierarchicalSelfDark(["a", "b"], 16, n_slots=4, m_l2=0.98, m_l3=0.995)
    shrunk.load_state_dict(hs.state_dict(), strict=False)
    assert shrunk._filled_cpu <= 4
    assert torch.allclose(shrunk.centers, hs.centers[:4])
    # a changed sense set cannot be migrated: fresh start, not a crash
    other = HierarchicalSelfDark(["a", "b", "c"], 16, n_slots=8)
    other.load_state_dict(hs.state_dict(), strict=False)
    assert other._filled_cpu == 0


def test_state_roundtrip():
    rng = np.random.default_rng(2)
    frames, _ = _stream(rng, 100)
    hs = HierarchicalSelfDark(["a", "b"], 16, n_slots=8)
    for v in frames:
        _observe(hs, v, float(rng.standard_normal()))
    hs2 = HierarchicalSelfDark(["a", "b"], 16, n_slots=8)
    hs2.load_state_dict(hs.state_dict())
    assert hs2._filled_cpu == hs._filled_cpu
    assert torch.allclose(hs2.centers, hs.centers)
    assert torch.allclose(hs2.membership("l2"), hs.membership("l2"))



# ---------------------------------------------------------------- audit P222 / P224 (2026-09-18)
def test_no_evidence_count_before_the_line_is_reachable():
    """The reachability arithmetic for the module's own pass line, and the rule that
    the count reads unmeasured (-1) until a perfect slot could have cleared it."""
    import math
    import torch
    from mimo_bridge.hierarchical_self import HierarchicalSelfDark
    h = HierarchicalSelfDark(["a", "b"], 16, n_slots=8)
    # legacy per-slot line: 9 ticks at momentum 0.9
    assert h.first_evidence_tick() == 9
    for n in range(1, 9):
        assert h._rank_ceiling(n) <= h._rank_line(n)
    assert h._rank_ceiling(9) > h._rank_line(9)
    assert h.first_evidence_steps("l2") == 110_000 and h.first_evidence_steps("l3") == 1_100_000
    # family-wise line: 14 ticks on a 64-slot table, 25 on a 6,400-slot table
    assert h.first_evidence_tick(64) == 14 and h.first_evidence_tick(6400) == 25
    assert h.first_evidence_steps("l3", 6400) == 2_700_000
    assert abs(h._rank_ceiling(8) - round(0.5 + 0.5 * (1 - 0.9 ** 8), 4)) < 1e-3
    # live class, small windows: -1 with steps remaining until the line is reachable,
    # then a count, on a wellness locked to the regime
    torch.manual_seed(0)
    live = HierarchicalSelfDark(["a", "b"], d_model=4, n_slots=8, m_l2=0.99, m_l3=0.999)
    proto = torch.randn(2, 4)
    need = None
    seen = {}
    for fr in range(100 * 40):
        phase = (fr // 25) % 2
        x = proto[phase].unsqueeze(0) + 0.01 * torch.randn(1, 4)
        live.observe({"a": x, "b": x.flip(-1)}, {"w": torch.tensor([[1.0 if phase == 0 else 0.0]])})
        if (fr + 1) % 100 == 0:
            s = live.stats()
            need = s["l2_ticks_needed"]
            seen[s["l2_ticks"]] = s
    assert need == live.first_evidence_tick(live.stats()["slots_filled"])
    for t_, s in seen.items():
        if 3 <= t_ < need:
            assert s["l2_sig"] == -1 and s["l2_line"] is None, (t_, s["l2_sig"])
            assert s["l2_steps_to_first_evidence"] == (need - t_) * 100
        if t_ >= need:
            assert s["l2_steps_to_first_evidence"] == 0 and s["l2_line"] is not None
    late = seen[max(seen)]
    assert late["l2_sig"] >= 0
    assert late["l2_rank_top"] > 0.5


# ---------------------------------------------------------------- audit P235 (2026-09-19)
def _proto_stream(rng, n, d_model=16, n_proto=12, sep=4.0):
    protos = [rng.standard_normal(2 * d_model) * sep for _ in range(n_proto)]
    for i in range(n):
        lb = (i // 20) % n_proto
        yield (protos[lb] + rng.standard_normal(2 * d_model)).astype(np.float32), lb


def test_shrink_migration_names_what_it_discards(caplog):
    """A smaller table drops every slot above the new count. The one migration line must say how many
    FILLED slots that was and how many of them the pass line had counted as evidence - 'N slots
    dropped' alone cannot tell the operator whether earned evidence was discarded. Measured on this
    fixture: 31 of 64 slots filled, slots 0, 1 and 2 counted at the minutes level; a shrink to 2 discards
    29 filled slots, 1 of them counted."""
    rng = np.random.default_rng(3)
    src = HierarchicalSelfDark(["a", "b"], 16, n_slots=64, m_l2=0.98, m_l3=0.995)
    for v, lb in _proto_stream(rng, 1200):
        src.observe({"a": torch.tensor(v[:16]).unsqueeze(0), "b": torch.tensor(v[16:]).unsqueeze(0)},
                    {"g": torch.tensor([[1.0 if lb == 0 else -1.0]])})
    k, st = src._filled_cpu, src.stats()
    flags = src._evidence_flags(src.l2_rank_ema, src.l2_slot_ticks, k, int(src.l2_ticks))
    assert flags is not None and int(flags.sum()) == st["l2_sig"], "the helper and stats() must count alike"
    lost_evidence = int(flags[2:].sum())
    assert k > 2 and lost_evidence >= 1, "the fixture must discard counted evidence or it tests nothing"
    small = HierarchicalSelfDark(["a", "b"], 16, n_slots=2, m_l2=0.98, m_l3=0.995)
    with caplog.at_level(logging.WARNING, logger="mimo_bridge.hself"):
        small.load_state_dict({kk: vv.clone() for kk, vv in src.state_dict().items()}, strict=False)
    lines = [r.getMessage() for r in caplog.records if "hself: migrated" in r.getMessage()]
    assert len(lines) == 1, lines
    assert f"{k - 2} FILLED slots discarded" in lines[0], lines[0]
    assert f"minutes level {lost_evidence}" in lines[0], lines[0]
    # growing discards nothing and says so by saying nothing about discards
    caplog.clear()
    big = HierarchicalSelfDark(["a", "b"], 16, n_slots=128, m_l2=0.98, m_l3=0.995)
    with caplog.at_level(logging.WARNING, logger="mimo_bridge.hself"):
        big.load_state_dict({kk: vv.clone() for kk, vv in src.state_dict().items()}, strict=False)
    lines = [r.getMessage() for r in caplog.records if "hself: migrated" in r.getMessage()]
    assert len(lines) == 1 and "discarded" not in lines[0], lines
