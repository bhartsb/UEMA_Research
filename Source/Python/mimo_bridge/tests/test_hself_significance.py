# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Validation of the hierarchical-self significance statistic on SYNTHETIC data
(follow-up review 2026-09-08: the l2_sig/l3_sig counts were reported live without
ever being shown to fire on a known effect and stay silent on a known null).

Ground truth by construction:
  null    presence varies (two regimes) but wellness is independent noise -
          no slot may cross the significance floor
  effect  wellness co-moves with the regime at the l2 timescale - at least one
          slot must cross the floor
  floor   the 3/sqrt(n_eff) floor must shrink as frames accumulate
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mimo_bridge.hierarchical_self import HierarchicalSelfDark  # noqa: E402

B, DM = 4, 16


def _tracker():
    # shortened windows so the synthetic run is fast: l2 window 100 frames,
    # l3 window 1000 (same code path, smaller constants)
    return HierarchicalSelfDark(["vision"], d_model=DM, n_slots=8,
                                m_l2=0.99, m_l3=0.999)


def _regime_vectors(gen):
    a = torch.zeros(DM); a[0] = 1.0
    b = torch.zeros(DM); b[1] = 1.0
    return a, b


def _run(tracker, frames, dwell, wellness_fn, seed=0):
    gen = torch.Generator().manual_seed(seed)
    a, b = _regime_vectors(gen)
    for i in range(frames):
        in_a = (i // dwell) % 2 == 0
        base = a if in_a else b
        toks = {"vision": base.unsqueeze(0).repeat(B, 1)
                + 0.05 * torch.randn(B, DM, generator=gen)}
        w = wellness_fn(in_a, gen)
        tracker.observe(toks, {"well": w})
    return tracker.stats()


def test_null_no_significance():
    """Independent wellness noise: zero slots may read significant."""
    # 13,500 frames: the hours level (window 1,000) needs 2 warm-up windows plus
    # first_evidence_tick(k) ticks before its 0 is a MEASURED zero rather than -1
    s = _run(_tracker(), frames=13500, dwell=200,
             wellness_fn=lambda in_a, g: torch.randn(B, 1, generator=g))
    assert s["l2_sig"] == 0, s
    assert s["l3_sig"] == 0, s


def test_effect_detected_at_l2():
    """Wellness locked to the regime at the l2 timescale: the statistic must fire."""
    s = _run(_tracker(), frames=6000, dwell=200,
             wellness_fn=lambda in_a, g: (1.0 if in_a else -1.0)
             + 0.1 * torch.randn(B, 1, generator=g))
    assert s["slots_filled"] >= 2, "both regimes must have opened slots"
    assert s["l2_sig"] >= 1, s
    # and the winning slot's persistence must clear the line with margin
    assert s["l2_rank_top"] > s["l2_line"], s


def test_effect_absent_reads_smaller_than_effect_present():
    """The statistic must separate the two worlds, not merely differ from zero."""
    null_s = _run(_tracker(), frames=6000, dwell=200,
                  wellness_fn=lambda in_a, g: torch.randn(B, 1, generator=g), seed=1)
    eff_s = _run(_tracker(), frames=6000, dwell=200,
                 wellness_fn=lambda in_a, g: (1.0 if in_a else -1.0)
                 + 0.1 * torch.randn(B, 1, generator=g), seed=1)
    assert eff_s["l2_abs_mean"] > 2.0 * null_s["l2_abs_mean"], (null_s, eff_s)


def test_significance_contract_rank_persistence():
    """2026-09-10 capacity review replaced the analytic 3/sqrt(frames/window)
    floor (it FELL with lifetime frames while the EMA estimator forgets -
    changing only the frame counter moved the significant count 20 -> 61).
    Significance is now RANK PERSISTENCE against per-slot sign-flip nulls
    (third design; the analytic floor and two null constructions each failed
    an adversarial fixture): the line must exist once ticks accumulate, and
    before three window ticks the count must read UNMEASURED (-1)."""
    warm = _run(_tracker(), frames=6000, dwell=200,
                wellness_fn=lambda in_a, g: torch.randn(B, 1, generator=g), seed=2)
    assert warm["l2_line"] is not None and warm["l2_line"] > 0.5
    assert warm["l2_ticks"] >= 3
    cold = _run(_tracker(), frames=60, dwell=20,
                wellness_fn=lambda in_a, g: torch.randn(B, 1, generator=g), seed=2)
    assert cold["l2_sig"] == -1 and cold["l2_line"] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))



# ---------------------------------------------------------------- audit P224 (2026-09-18)
def test_line_depends_on_filled_slots_not_capacity():
    small = _run(_tracker(), frames=4000, dwell=200,
                 wellness_fn=lambda in_a, g: torch.tensor([[1.0 if in_a else 0.0]] * B))
    tr = _tracker()
    _ = _run(tr, frames=4000, dwell=200,
             wellness_fn=lambda in_a, g: torch.tensor([[1.0 if in_a else 0.0]] * B))
    big = HierarchicalSelfDark(["vision"], d_model=DM, n_slots=800, m_l2=0.99, m_l3=0.999)
    big.load_state_dict({k: v.clone() for k, v in tr.state_dict().items()}, strict=False)
    a, b = tr.stats(), big.stats()
    for key in ("l2_line", "l2_sig", "l2_ticks_needed"):
        assert a[key] == b[key], (key, a[key], b[key])


def test_slot_ticks_count_only_mature_slots():
    tr = _tracker()
    gen = torch.Generator().manual_seed(1)
    a, b = _regime_vectors(gen)
    for i in range(700):                      # regime a only: one slot
        tr.observe({"vision": a.unsqueeze(0).repeat(B, 1) + 0.05 * torch.randn(B, DM, generator=gen)},
                   {"well": torch.randn(B, 1, generator=gen)})
    t_first = int(tr.l2_ticks.item())
    for i in range(700):                      # regime b appears: a second, younger slot
        tr.observe({"vision": b.unsqueeze(0).repeat(B, 1) + 0.05 * torch.randn(B, DM, generator=gen)},
                   {"well": torch.randn(B, 1, generator=gen)})
    k = tr._filled_cpu
    assert k >= 2
    st = tr.l2_slot_ticks[:k]
    assert int(st[0]) == int(tr.l2_ticks.item()) or int(st[0]) >= int(st[1])
    assert int(st[1]) < int(tr.l2_ticks.item()), "a slot born later must have fewer ticks than the table"


# ---------------------------------------------------------------- audit P224, measured 2026-09-18
def _reviewer_fixture(seed, planted=False, n_frames=200 * 16):
    """The reviewer's no-association fixture through the production class: 6 senses x
    128, 64 slots, batch 2, window 200 (vf_p224_fixture.py). Measured over 6 seeds x
    64 slots: the old per-slot 3-sigma line miscounted 14 of 384 no-link slots, the
    family-wise line 0 of 384; on the planted-effect fixture the family-wise line
    counted slots in 6 of 6 seeds while a gate at the table's own middle counted
    nothing in 6 of 6."""
    import torch.nn.functional as F
    mods = ["a", "b", "c", "d", "e", "f"]; dm, b, win = 128, 2, 200; m = 1.0 - 1.0 / win
    torch.manual_seed(seed)
    hs = HierarchicalSelfDark(mods, dm, n_slots=64, m_l2=m, m_l3=m)
    bank = F.normalize(torch.randn(64, len(mods) * dm), dim=-1)
    wp = torch.zeros(b)
    for t in range(n_frames):
        which = torch.randint(0, 64, (b,))
        v = F.normalize(bank[which] + 0.15 * torch.randn(b, len(mods) * dm), dim=-1)
        if planted:
            on = (t // (win * 2)) % 2 == 0
            if on:
                v = F.normalize(bank[0].unsqueeze(0).repeat(b, 1) + 0.15 * torch.randn(b, len(mods) * dm), dim=-1)
            w = torch.full((b,), 4.5 if on else 3.0)
        else:
            wp = 0.999 * wp + 0.002 * torch.randn(b); w = 3.75 + wp
        toks = {mm: v[:, i * dm:(i + 1) * dm].contiguous() for i, mm in enumerate(mods)}
        hs.observe(toks, {f"e{j}": (w / 5.0).reshape(b, 1) for j in range(5)})
    return hs.stats()


def test_false_alarms_stay_at_the_declared_rate_and_a_planted_link_is_counted():
    null_sig = null_old = 0
    for seed in (0, 1):                     # measured old-line counts 8 and 1 on these seeds
        s = _reviewer_fixture(seed)
        assert s["l2_sig"] >= 0, s["l2_ticks"]
        null_sig += s["l2_sig"]; null_old += s["l2_sig_1slot"]
        for key in ("l2_rank_mean", "l2_rank_sd", "l2_expected_false", "l2_sig_1slot",
                    "l2_expected_false_1slot", "l2_line_empirical"):
            assert key in s
    assert null_sig == 0, "the family-wise line counted a no-link slot"
    assert null_old >= 1, "the old per-slot count must still expose the multiplicity the page prints"
    for seed in (0, 1):                     # measured 8 and 11 slots counted on these seeds
        s = _reviewer_fixture(seed, planted=True)
        assert s["l2_sig"] >= 1, "a planted link was not counted"


# ---------------------------------------------------------------- audit P233, measured 2026-09-19
def _no_link_table(seed, K=6400, win=50, n_frames=1500, dm=16, b=2):
    """A table with NO link between any slot and wellness, at the run's slot count: 6,400 filled slots
    (6 senses x 16 wide keeps it to about 10 s on a CPU), frozen centres, each frame two bodies near two
    random centres, wellness an independent slow AR(1) drift around 3.75 of 5 (the reviewer's P224 recipe)."""
    import torch.nn.functional as F
    mods = ["a", "b", "c", "d", "e", "f"]; m = 1.0 - 1.0 / win
    torch.manual_seed(seed)
    hs = HierarchicalSelfDark(mods, dm, n_slots=K, m_center=1.0, m_l2=m, m_l3=m)
    bank = F.normalize(torch.randn(K, len(mods) * dm), dim=-1)
    hs.centers.copy_(bank); hs._filled_cpu = K; hs.filled.fill_(K)
    wp = torch.zeros(b)
    for _ in range(n_frames):
        which = torch.randint(0, K, (b,))
        v = F.normalize(bank[which] + 0.15 * torch.randn(b, len(mods) * dm), dim=-1)
        wp = 0.999 * wp + 0.002 * torch.randn(b); w = 3.75 + wp
        toks = {mm: v[:, i * dm:(i + 1) * dm].contiguous() for i, mm in enumerate(mods)}
        hs.observe(toks, {f"e{j}": (w / 5.0).reshape(b, 1) for j in range(5)})
    return hs.stats()


def test_false_alarm_rate_at_the_runs_slot_count(record_property):
    """The count's false-alarm rate measured where the run lives: 6,400 slots judged together, no link
    anywhere. Measured 2026-09-19 on seeds 0 and 1: the family-wise line (0.9618 at 6,400 slots) counted
    0 of 12,800 no-link slots; the old one-slot line (0.7739) counted 14 and 48 of 6,400 - 0.22 and 0.75
    in 100 against the 0.135 in 100 it claims. The rates are recorded as numbers, not only as a pass."""
    fam = old = slots = 0
    for seed in (0, 1):
        s = _no_link_table(seed)
        assert s["l2_sig"] >= 0, ("the line must be reachable in this fixture", s["l2_ticks"], s["l2_ticks_needed"])
        fam += s["l2_sig"]; old += s["l2_sig_1slot"]; slots += s["slots_filled"]
    record_property("hself_no_link_slots_judged", slots)
    record_property("hself_no_link_counted_family_wise", fam)
    record_property("hself_no_link_counted_one_slot_line", old)
    print(f"no-link slots counted: {fam} of {slots} (family-wise line), {old} of {slots} "
          f"= {100.0 * old / slots:.2f} in 100 (old one-slot line, which claims 0.135 in 100)")
    assert fam == 0, f"the family-wise line counted {fam} of {slots} no-link slots"
    assert old >= 1, "the old one-slot count must still expose the multiplicity this fixture exists to show"


# ---------------------------------------------------------------- audit P239 (2026-09-19)
def test_stats_say_how_much_of_the_table_is_old_enough_to_be_counted():
    """The pass line is judged on each slot's OWN ticks. After the table's first-evidence tick has passed,
    a slot minted late still cannot be counted until its own ticks reach the same number - its 0 is forced
    by arithmetic. stats() must say how many slots are eligible so that 0 is not read as measured."""
    torch.manual_seed(0)
    dm, b = 16, 4
    h = HierarchicalSelfDark(["vision"], d_model=dm, n_slots=8, m_l2=0.9, m_l3=0.99)      # window 10 frames
    a_ = torch.zeros(dm); a_[0] = 1.0
    b_ = torch.zeros(dm); b_[1] = 1.0
    g = torch.Generator().manual_seed(1)
    def feed(base, n, w):
        for _ in range(n):
            h.observe({"vision": base.unsqueeze(0).repeat(b, 1) + 0.05 * torch.randn(b, dm, generator=g)},
                      {"well": torch.full((b, 1), w) if w is not None else torch.randn(b, 1, generator=g)})
    feed(a_, 400, None)                                  # one situation; the table passes its needed ticks
    st = h.stats()
    assert st["slots_filled"] == 1 and st["l2_ticks"] >= st["l2_ticks_needed"]
    assert st["l2_slots_eligible"] == 1 and st["l2_slot_ticks_min"] == st["l2_ticks"]
    feed(b_, 10, 1.0); feed(a_, 10, 0.0); feed(b_, 10, 1.0)  # a second situation appears: a slot is minted late
    st = h.stats()
    assert st["slots_filled"] == 2
    assert st["l2_steps_to_first_evidence"] == 0          # the TABLE is past its line ...
    assert st["l2_slots_eligible"] == 1                   # ... but only one of its two slots can be counted
    assert st["l2_slot_ticks_min"] < st["l2_ticks_needed"] <= st["l2_ticks"]
    assert st["l2_slot_ticks_min"] == int(h.l2_slot_ticks[1])
