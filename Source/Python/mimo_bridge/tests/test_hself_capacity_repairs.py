# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Regression tests for the 2026-09-10 slot-capacity review repairs.

Each pins one measured defect in its FIXED state:
  batch duplicates   8 identical novel inputs opened 8 identical slots
  combinations       a0+b1 after a0+b0 and a1+b1 was invisible to admission
  capacity gauges    growing 64 -> 6,400 with identical evidence moved
                     slots_live 22 -> 64 (allocated capacity in denominators)
  significance       the analytic floor fell with lifetime frames while the
                     estimator forgets; the floor is now the time-shifted
                     null's own spread, and it is UNMEASURED until the null
                     has a full window behind it
  demand             novelty went unmeasured once full; distinct turned-away
                     patterns and their sightings are now counted
"""
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mimo_bridge.hierarchical_self import HierarchicalSelfDark  # noqa: E402

MODS = ["a", "b"]
DM = 8


def _mk(n_slots=8, **kw):
    torch.manual_seed(0)
    return HierarchicalSelfDark(MODS, DM, n_slots=n_slots, **kw)


def _state(seed, b=1):
    g = torch.Generator().manual_seed(seed)
    va = torch.randn(1, DM, generator=g).repeat(b, 1)
    vb = torch.randn(1, DM, generator=g).repeat(b, 1)
    return {"a": va, "b": vb}


def _blocks(sa, sb, b=1):
    ga = torch.Generator().manual_seed(sa)
    gb = torch.Generator().manual_seed(1000 + sb)
    return {"a": torch.randn(1, DM, generator=ga).repeat(b, 1),
            "b": torch.randn(1, DM, generator=gb).repeat(b, 1)}


def _well(b=1):
    return {"w": torch.zeros(b, 1)}


def _obs(h, tokens, well=None):
    # admission runs on every 4th call; pad with repeats so it fires once
    for _ in range(4):
        h.observe(tokens, well or _well(tokens["a"].shape[0]))


def test_identical_batch_opens_exactly_one_slot():
    h = _mk()
    _obs(h, _state(1, b=8))          # eight IDENTICAL novel inputs at once
    assert int(h.filled.item()) == 1, (
        "sequential admission must let the first acceptance block the other "
        "seven copies (the review found 8 identical slots)")
    _obs(h, _state(2, b=8))
    assert int(h.filled.item()) == 2


def test_new_combination_is_admitted_and_counted():
    h = _mk()
    _obs(h, _blocks(1, 1))           # a0+b0
    _obs(h, _blocks(2, 2))           # a1+b1
    n0 = int(h.filled.item())
    _obs(h, _blocks(1, 1))           # exact repeat: refused
    assert int(h.filled.item()) == n0
    _obs(h, _blocks(1, 2))           # a0+b1: every sense known, whole new
    assert int(h.filled.item()) == n0 + 1, (
        "a new combination of known senses must open a slot")
    assert int(h.admit_combo.item()) >= 1


def test_gauges_are_capacity_invariant():
    """Same evidence, bigger table: slots_live, entropy and the significance
    count must not move (the review measured 22 -> 64 from empty slots)."""
    small = _mk(n_slots=8)
    for s in range(4):
        _obs(small, _state(s))
    big = HierarchicalSelfDark(MODS, DM, n_slots=800)
    sd = {k: v.clone() for k, v in small.state_dict().items()}
    big.load_state_dict(sd, strict=False)   # migration pads with fresh init
    a, b = small.stats(), big.stats()
    for key in ("slots_filled", "slots_live", "occ_entropy",
                "l2_sig", "l3_sig", "presence_top1", "presence_eff", "slots_live_margin"):
        assert a[key] == b[key], f"{key}: {a[key]} != {b[key]} across capacity"


def test_significance_unmeasured_until_null_warm_then_floor_is_empirical():
    # short windows so the test can outlive the null warm-up: l2 window = 10
    h = _mk(n_slots=4, m_l2=0.9, m_l3=0.99)
    s0, s1 = _state(1), _state(2)
    g = torch.Generator().manual_seed(3)
    for i in range(9):
        h.observe(s0 if i % 2 else s1,
                  {"w": torch.randn(1, 1, generator=g)})
    st = h.stats()
    assert st["l2_sig"] == -1 and st["l2_line"] is None, (
        "before persistence ticks exist, significance must read "
        "UNMEASURED, not a number")
    for i in range(300):
        h.observe(s0 if i % 2 else s1,
                  {"w": torch.randn(1, 1, generator=g)})
    st = h.stats()
    assert st["l2_sig"] >= 0 and st["l2_line"] is not None
    # white-noise wellness: persistence must hold the count low
    assert st["l2_sig"] <= st["slots_filled"]
    assert st["l2_ticks"] >= 3


def test_demand_is_counted_when_full():
    h = _mk(n_slots=2)
    _obs(h, _state(1))
    _obs(h, _state(2))
    assert int(h.filled.item()) == 2      # full
    _obs(h, _state(7))                    # novel, refused for capacity
    _obs(h, _state(7))                    # the SAME pattern again
    _obs(h, _state(8))                    # a second distinct pattern
    st = h.stats()
    assert st["shadow"]["distinct"] == 2, st["shadow"]
    assert st["shadow"]["sightings"] >= 3
    assert int(h.filled.item()) == 2      # the table itself untouched


def test_slot_count_flag_reaches_the_constructor(tmp_path, monkeypatch):
    import json as _json
    from mimo_bridge.tools import causal_test as ct
    run = tmp_path / "R5"
    ckpt = run / "checkpoints" / "c1"
    ckpt.mkdir(parents=True)
    (run / "args.json").write_text(_json.dumps({
        "context_length": 4, "residual_prediction": "on", "intention": "on",
        "hierarchical_self": "dark", "hself_slots": 6400}), encoding="utf-8")
    seen = {}

    class Spy:
        def __init__(self, **kw):
            seen.update(kw)
            raise RuntimeError("spy stop")

    import mimo_bridge.lem_model as LM
    import mimo_bridge.train_mimo as TM
    monkeypatch.setattr(LM, "LemLite", Spy)
    # since 2026-09-22 _build_model builds through train_mimo.build_learner, which
    # calls the LemLite name train_mimo imported - the spy must sit there too
    monkeypatch.setattr(TM, "LemLite", Spy)
    with pytest.raises(RuntimeError):
        ct._build_model(str(ckpt), "cpu", 0, intention=True)
    assert seen.get("hself_slots") == 6400


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_persistent_noise_control_and_effect_detection(record_property):
    """The reviewer's 2026-09-10 acceptance protocol, compact: independent
    slowly-drifting AR(1) wellness must NOT read as knowledge (their fixture
    measured 37/256 false streams under the sign-flip floor), while wellness
    genuinely following the regime must. Run through the PRODUCTION tracker,
    16 streams per condition, window 50."""
    import math as _m
    mom, fmom = 0.98, 0.8            # window 50, fast 5
    T = 1500
    flagged_noise = flagged_effect = 0
    for cond in ("noise", "effect"):
        for stream in range(16):
            h = HierarchicalSelfDark(["a"], 2, n_slots=2,
                                     m_center=1.0, m_l2=mom, m_l3=0.999)
            h.centers.copy_(torch.eye(2))
            h._filled_cpu = 2
            h.filled.fill_(2)
            rng = torch.Generator().manual_seed(500 + stream)
            a = b = 0.0
            rho = 0.995
            for t in range(T):
                a = rho * a + _m.sqrt(1 - rho * rho) * float(
                    torch.randn(1, generator=rng))
                b = rho * b + _m.sqrt(1 - rho * rho) * float(
                    torch.randn(1, generator=rng))
                ang = _m.pi / 4 + 0.7 * _m.tanh(a)
                x = torch.tensor([[_m.cos(ang), _m.sin(ang)]])
                w = (_m.tanh(a) + 0.1 * float(torch.randn(1, generator=rng))
                     if cond == "effect" else b)
                h.observe({"a": x}, {"w": torch.tensor([[w]])})
            st = h.stats()
            if st["l2_sig"] > 0:
                if cond == "noise":
                    flagged_noise += 1
                else:
                    flagged_effect += 1
    # MEASURED 2026-09-19 under the family-wise line (0.7926 at 2 slots, 28 ticks): 1 of these
    # 16 no-link streams is counted (2 under the old one-slot line 0.7739), and 2 of 64 on
    # seeds 500-563 - 3.1 linkless tables in 100 against the 0.135 in 100 the line is set
    # for (95% interval 0.4 to 10.8 in 100). The allowance below is that measured count,
    # not a tolerance: a second counted stream is a regression. It was 2 of 16 - 12.5 in
    # 100 - beside a line that claims 0.135. All 16 genuine-effect streams are counted
    # (lowest best standing 0.9521 against the line 0.7926).
    record_property("hself_ar1_no_link_streams_counted_of_16", flagged_noise)
    record_property("hself_ar1_effect_streams_counted_of_16", flagged_effect)
    print(f"AR(1) protocol: {flagged_noise} of 16 no-link streams counted, "
          f"{flagged_effect} of 16 genuine-effect streams counted")
    assert flagged_noise <= 1, (
        f"{flagged_noise}/16 independent drifting-noise streams read as "
        "knowledge (measured 1 on 2026-09-19) - the persistence control has regressed")
    assert flagged_effect >= 12, (
        f"only {flagged_effect}/16 genuine-effect streams detected - the "
        "statistic has gone blind")



# ---------------------------------------------------------------- audit P221 (2026-09-18)
def test_assignment_concentration_is_reported_and_capacity_invariant():
    """The share the top slot holds and the slots a moment spreads over are COMPUTED
    every frame and were reported nowhere, so the page quoted a frozen probe number.
    They must reach stats(), be a real distribution's numbers, and not move when empty
    capacity is added."""
    h = _mk(n_slots=16)
    for s in range(6):
        _obs(h, _state(s))
    for _ in range(1200):                      # clear the 0.001-rate EMA warm-up
        h.observe(_state(1), _well())
    st = h.stats()
    kf = st["slots_filled"]
    assert st["presence_eff"] is not None and st["presence_top1"] is not None
    assert 1.0 <= st["presence_eff"] <= kf     # an entropy count cannot be < 1
    assert 1.0 / kf <= st["presence_top1"] <= 1.0
    assert st["slots_live_margin"] >= 1.0 or st["slots_live"] < kf
    big = HierarchicalSelfDark(MODS, DM, n_slots=1600)
    big.load_state_dict({k: v.clone() for k, v in h.state_dict().items()}, strict=False)
    b = big.stats()
    for key in ("presence_top1", "presence_eff", "slots_live_margin"):
        assert st[key] == b[key], f"{key}: {st[key]} != {b[key]} across capacity"


def test_concentration_reads_unmeasured_while_the_ema_is_cold():
    """A resumed lineage's checkpoint carries no presence_* key, so the EMA starts at 0.
    Zero effective slots is not a measurement."""
    h = _mk(n_slots=16)
    _obs(h, _state(1))
    assert h.stats()["presence_eff"] is None


# ---------------------------------------------------------------- audit P225 (2026-09-18)
def test_admission_rate_has_its_denominator():
    h = _mk(n_slots=2)
    _obs(h, _state(1)); _obs(h, _state(2))                 # fills the table
    st = h.stats()
    assert st["admission"]["checked"] > 0
    before = st["admission"]["checked"]
    _obs(h, _state(3, b=8))                                 # one admission-cadence frame, 8 bodies
    st = h.stats()
    assert st["admission"]["checked"] - before >= 8         # every body in the batch is counted
    # the SAME stored state again: checked rises, judged_new does not
    c0, n0 = st["admission"]["checked"], st["admission"]["judged_new"]
    _obs(h, _state(1))
    st = h.stats()
    assert st["admission"]["checked"] > c0 and st["admission"]["judged_new"] == n0
    # distinct states with the table full: judged_new rises on every candidate
    n1 = st["admission"]["judged_new"]
    for s in range(20, 26):
        _obs(h, _state(s))
    st = h.stats()
    assert st["admission"]["judged_new"] >= n1 + 6
    assert st["admission"]["share_new"] == round(st["admission"]["judged_new"] / st["admission"]["checked"], 4)
    assert st["slot_age_newest_frames"] <= st["slot_age_median_frames"] <= st["slot_age_oldest_frames"]
    # resume guard: the two counters start together at 0 when a checkpoint lacks them
    sd = {k: v.clone() for k, v in h.state_dict().items()}
    del sd["checked"], sd["checked_novel"]
    fresh = _mk(n_slots=2)
    fresh.load_state_dict(sd, strict=False)
    st2 = fresh.stats()
    assert st2["admission"]["checked"] == 0 and st2["admission"]["judged_new"] == 0
    assert st2["shadow"]["overflow"] == st["shadow"]["overflow"]



def test_the_page_states_the_classes_own_buffer_count():
    """schema.py names the hierarchical self's buffer count; it must be the class's."""
    from mimo_bridge.explainer.schema import DOC
    n = len(list(_mk(n_slots=8).buffers()))
    text = " ".join(v.get("w", "") for v in DOC.values())
    assert f"0 weights and {n} buffers" in text, (n, "not stated on the page")


# ---------------------------------------------------------------- audit P234: the run's table size
RUN_MODS = ["audio", "intermind", "motor", "proprioception", "tactile", "vision"]


def _run_table(K=6400, seed=0):
    torch.manual_seed(seed)
    dm = 128
    h = HierarchicalSelfDark(RUN_MODS, dm, n_slots=K)            # the constructor defaults the trainer uses
    cen = F.normalize(torch.randn(K, len(RUN_MODS) * dm), dim=-1)
    h.centers.copy_(cen); h._filled_cpu = K; h.filled.fill_(K); h.occupancy.fill_(1.0 / K)
    return h, cen, dm


def _run_batch(cen, j=123, a=0.75, b=8):
    return a * cen[j].unsqueeze(0) + math.sqrt(1 - a * a) * F.normalize(torch.randn(b, cen.shape[1]), dim=-1)


def _run_toks(x, dm):
    return {m: x[:, i * dm:(i + 1) * dm].contiguous() for i, m in enumerate(RUN_MODS)}


def test_gauges_at_the_runs_table_size():
    """Every other hself test fills at most 64 slots. This one builds the run's table - 6 senses x 128 =
    768 wide, 6,400 FILLED slots, temperature 10, m_center 0.999, a batch of 8 bodies - and holds the
    per-frame gauges that only misbehave at that size against float64 references computed here."""
    K = 6400
    h, cen, dm = _run_table(K)
    assert h.temperature == 10.0 and h.m_center == 0.999
    # --- the reachability arithmetic at this slot count and the run's window lengths
    assert h.first_evidence_tick(K) == 25
    assert h.first_evidence_steps("l2", K) == 270_000 and h.first_evidence_steps("l3", K) == 2_700_000
    # --- one frame from zeroed gauges: the rotation gauges against an independent float64 route.
    # Since audit P238 a frame moves only the row it belongs to (every live sense at cosine 0.9 or
    # above), so the 8 bodies sit at cosine 0.97 from row 123 and that one row moves, at 1 - m_center.
    j = 123
    x = _run_batch(cen, j=j, a=0.97); xn = F.normalize(x, dim=-1)
    c0 = h.centers.clone()
    own, hit = h._own_slots(xn, xn @ c0.T, K)
    assert bool(hit.all()) and own.tolist() == [j] * 8
    e = 1.0 - h.m_center
    c64 = c0[j].double(); c64 = c64 / c64.norm()
    m = (1.0 - e) * c64 + e * xn.double().mean(0)
    mh = m / m.norm()
    ref = float(0.5 * (mh - c64).pow(2).sum())                   # |a - b|^2 = 2 (1 - cos) for unit vectors
    h.observe(_run_toks(x, dm), {"w": torch.rand(8, 1)})
    got_mean, got_max = float(h.center_drift) / 0.001, float(h.center_drift_max) / 0.001
    assert got_mean > 0.0 and got_max >= got_mean
    assert abs(got_mean - ref) <= 1e-4 * ref, (got_mean, ref)
    assert abs(got_max - ref) <= 1e-4 * ref, (got_max, ref)
    assert abs(float(h.center_step) / 0.001 - e) <= 1e-6 * e
    # every other row is bit-identical, and row 123 landed where the float64 route says
    keep = torch.ones(K, dtype=torch.bool); keep[j] = False
    assert torch.equal(h.centers[keep], c0[keep])
    assert float((h.centers[j].double() - mh).abs().max()) < 1e-6
    # the float32 subtraction this route replaced cannot resolve a rotation this small: 1 - cos of two
    # float32 unit vectors moves in steps of 6.0e-08, and this rotation is smaller than one step
    naive = float(1.0 - (F.normalize(((1.0 - e) * c0[j] + e * xn.mean(0)), dim=-1) * c0[j]).sum())
    assert ref < 6.0e-08 and abs(naive - ref) > 0.1 * ref, (naive, ref)
    # --- 103 more frames: the assignment gauges against a float64 softmax kept beside them
    top, eff, n = 0.0, 0.0, 0
    for t in range(104):
        if t:                                                     # frame 0 was observed above
            x = _run_batch(cen); h_c = h.centers.clone()
            h.observe(_run_toks(x, dm), {"w": torch.rand(8, 1)})
        else:
            h_c = c0
        p64 = torch.softmax(h.temperature * (F.normalize(x, dim=-1).double() @ h_c.double().T), dim=-1)
        top = 0.999 * top + 0.001 * float(p64.max(-1).values.mean())
        eff = 0.999 * eff + 0.001 * float((-(p64 * p64.clamp_min(1e-30).log()).sum(-1)).exp().mean())
        n += 1
    st = h.stats()
    bc = 1.0 - 0.999 ** n
    assert st["presence_top1"] is not None and st["presence_eff"] is not None
    assert abs(st["presence_top1"] - top / bc) <= 1e-3 * (top / bc), (st["presence_top1"], top / bc)
    assert abs(st["presence_eff"] - eff / bc) <= 1e-3 * (eff / bc), (st["presence_eff"], eff / bc)
    assert 1.0 / K < st["presence_top1"] < 1.0 and 1.0 < st["presence_eff"] < K


def test_observe_is_float32_under_a_callers_autocast():
    """The trainer runs under bfloat16 autocast and switches it off around this tracker. That guard must
    hold wherever observe() is called from: the same frames under an enclosing autocast must leave
    bit-identical buffers. CPU autocast stands in for the cuda one."""
    mods, dm, K = ["a", "b", "c"], 32, 256
    def run(autocast):
        torch.manual_seed(0)
        h = HierarchicalSelfDark(mods, dm, n_slots=K)
        cen = F.normalize(torch.randn(K, len(mods) * dm), dim=-1)
        h.centers.copy_(cen); h._filled_cpu = K; h.filled.fill_(K)
        for _ in range(12):
            x = 0.75 * cen[7:8] + 0.66 * F.normalize(torch.randn(8, cen.shape[1]), dim=-1)
            toks = {m: x[:, i * dm:(i + 1) * dm].contiguous() for i, m in enumerate(mods)}
            w = {"w": torch.rand(8, 1)}
            if autocast:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    h.observe(toks, w)
            else:
                h.observe(toks, w)
        return h
    a, b = run(False), run(True)
    for name in ("centers", "occupancy", "presence_top1", "presence_eff", "l2_p_fast", "l2_cov"):
        assert torch.equal(getattr(a, name), getattr(b, name)), f"{name} changed under the caller's autocast"


# ---------------------------------------------------------------- audit P237
def _p237_tracker():
    torch.manual_seed(0)
    return HierarchicalSelfDark(MODS, DM, n_slots=8, m_l2=0.9, m_l3=0.99)     # windows 10 and 100 frames


def _p237_frame(gen, b=4):
    return ({"a": torch.randn(b, DM, generator=gen), "b": torch.randn(b, DM, generator=gen)},
            {"w": torch.randn(b, 1, generator=gen)})


def test_an_ordinary_frame_asks_the_device_for_nothing(monkeypatch):
    """observe() runs once per model step on the GPU; every .item() there stalls the host until the
    device catches up. Measured before this fix: 3 per ordinary frame (the frame counter once, the
    null warm-up mark once per level). Admission frames (every 4th) still read decisions back."""
    h = _p237_tracker()
    gen = torch.Generator().manual_seed(1)
    for _ in range(8):
        h.observe(*_p237_frame(gen))
    calls = []
    real = torch.Tensor.item
    monkeypatch.setattr(torch.Tensor, "item", lambda self: (calls.append(1), real(self))[1])
    seen = 0
    for _ in range(8):
        fr = h._frames_cpu
        calls.clear()
        toks, well = _p237_frame(gen)
        h.observe(toks, well)
        if fr % 4 != 0:
            assert len(calls) == 0, f"frame {fr}: {len(calls)} .item() calls on an ordinary frame"
            seen += 1
    assert seen == 6


def test_host_shadows_match_the_buffers_and_the_flip_cache_changes_nothing():
    gen_a, gen_b = torch.Generator().manual_seed(2), torch.Generator().manual_seed(2)
    a, b = _p237_tracker(), _p237_tracker()
    for i in range(260):                       # 26 minutes-level blocks, 2 hours-level blocks
        a.observe(*_p237_frame(gen_a))
        b._flip_cache = {}                     # b rebuilds the signs every frame, as the old code did
        b.observe(*_p237_frame(gen_b))
    for name in ("l2_nw_ema", "l2_ncov", "l3_nw_fast", "l3_ncov", "l2_rank_ema", "l3_rank_ema"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert a._frames_cpu == int(a.frames.item()) == 260
    assert a._null_valid_from_cpu == int(a.null_valid_from.item())
    # a resume restores both shadows; one from before the sign-flip null existed marks the warm-up from NOW
    c = _p237_tracker()
    c.load_state_dict(a.state_dict())
    assert c._frames_cpu == 260 and c._null_valid_from_cpu == a._null_valid_from_cpu and c._flip_cache == {}
    null_keys = {"null_valid_from"} | {f"{t}_{n}" for t in ("l2", "l3") for n in ("nw_ema", "nw_fast", "ncov", "nwvar")}
    old = {k: v.clone() for k, v in a.state_dict().items() if k not in null_keys}
    d = _p237_tracker()
    d.load_state_dict(old, strict=False)
    assert d._null_valid_from_cpu == int(d.null_valid_from.item()) == 260
    # and the resumed tracker keeps step with the original, frame for frame
    gen_c = torch.Generator().manual_seed(3); gen_d = torch.Generator().manual_seed(3)
    for _ in range(40):
        a.observe(*_p237_frame(gen_c)); c.observe(*_p237_frame(gen_d))
    assert torch.equal(a.l2_ncov, c.l2_ncov) and torch.equal(a.l2_rank_ema, c.l2_rank_ema)


# ---------------------------------------------------------------- audit P238
def _near(v, cosv):
    o = torch.randn_like(v); o = F.normalize(o - (o @ v) * v, dim=-1)
    return cosv * v + math.sqrt(1 - cosv ** 2) * o


def test_twin_census_counts_one_situation_stored_twice():
    """slots_filled counts slots. Two slots whose centres coincide are one pattern stored twice,
    and stats() must say so: how many rows have a twin, how many groups they form, how many distinct
    situations are left, and the share of all occupancy the twinned rows hold."""
    torch.manual_seed(0)
    h = HierarchicalSelfDark(["a", "b", "c"], 32, n_slots=64)
    c = F.normalize(torch.randn(40, 96), dim=-1)
    c[[3, 9, 17, 22]] = c[1].clone()                  # one situation stored 5 times
    c[30] = c[29].clone()                             # another stored twice
    c[36] = _near(c[35], 0.9990)                      # close, but under the census line 0.9999
    c[12] = _near(c[11], 0.99995); c[13] = c[12].clone(); c[10] = c[11].clone()   # two exact pairs, one group
    h.centers[:40] = c; h._filled_cpu = 40; h.filled.fill_(40)
    occ = torch.full((40,), 1.0); occ[[1, 3, 9, 17, 22]] = 4.0
    h.occupancy[:40] = occ
    st = h.stats()
    assert st["slots_filled"] == 40
    assert st["slots_twinned"] == 5 + 2 + 4
    assert st["twin_groups"] == 3
    assert st["slots_distinct"] == 40 - 11 + 3
    assert st["twin_occupancy"] == round((5 * 4.0 + 2 + 4) / float(occ.sum()), 4)
    assert 0.0 < st["nearest_other_cos_median"] < 0.9999
    assert 1.0 <= st["occupancy_effective_slots"] <= 40.0
    # capacity must not move it
    big = HierarchicalSelfDark(["a", "b", "c"], 32, n_slots=640)
    big.load_state_dict({k: v.clone() for k, v in h.state_dict().items()}, strict=False)
    sb = big.stats()
    for key in ("slots_twinned", "twin_groups", "slots_distinct", "twin_occupancy", "occupancy_effective_slots"):
        assert st[key] == sb[key], key


def _two_situations(dm):
    def pair():
        s = F.normalize(torch.randn(dm), dim=-1)
        return s, _near(s, 0.70)
    (sa, ra), (sb_, rb) = pair(), pair()
    return torch.cat([sa, sb_]), torch.cat([ra, rb])


def test_a_rare_kind_of_moment_keeps_its_own_slot():
    """The cure's acceptance test (audit P238). A situation on 97 of 100 frames, a second on 3 of 100 at
    cosine 0.70 from it per sense, a batch of 8 bodies, the class at its defaults except a 1,000-frame
    minutes window so the evidence ticks fit in the test. MEASURED UNDER THE SOFT UPDATE THIS REPLACED
    (2026-09-19): the rare situation's row was pulled onto the common one, a fresh row opened for it and was
    pulled in too - several rows on one point after 12,000 frames, 9 of 11 after 40,000, and the rare row
    never kept a window of its own. Now that a moment changes only its own slot, the table holds exactly the 2 kinds of moment, no slot
    is a copy, the two centres stay where the situations are, and the rare row keeps every window it lived."""
    torch.manual_seed(0)
    dm, b = 16, 8
    h = HierarchicalSelfDark(["a", "b"], dm, n_slots=16, m_l2=0.999)
    S, R = _two_situations(dm)
    for t in range(12000):
        rare = (t % 1000) >= 970
        x = (R if rare else S).unsqueeze(0).repeat(b, 1) + 0.02 * torch.randn(b, 2 * dm)
        h.observe({"a": x[:, :dm], "b": x[:, dm:]}, {"w": torch.full((b, 1), 1.0 if rare else 0.0)})
    k = h._filled_cpu
    c = F.normalize(h.centers[:k], dim=-1); sim = c @ c.T; sim.fill_diagonal_(-2.0)
    independent = int((sim.max(-1).values > 0.9999).sum())
    st = h.stats()
    assert st["slots_twinned"] == independent == 0
    assert k == 2 and st["slots_distinct"] == 2, (k, st["slots_distinct"])
    to_s, to_r = c @ F.normalize(S, dim=-1), c @ F.normalize(R, dim=-1)
    common, rare_row = int(to_s.argmax()), int(to_r.argmax())
    assert common != rare_row
    assert float(to_s[common]) > 0.99 and float(to_r[rare_row]) > 0.99, "each row sits on its situation"
    assert float(c[common] @ c[rare_row]) < 0.75, "the two rows were pulled together"
    ticks = int(h.l2_ticks.item())
    assert ticks >= 8
    assert int(h.l2_slot_ticks[rare_row].item()) >= ticks - 1, (
        "the rare situation's row lost windows: it was re-opened instead of kept")
    assert st["moments_with_a_slot"] > 0.99 and st["best_slot_cos"] > 0.99


def test_a_far_frame_moves_no_slot():
    """The failure the soft update was brought in to cure on 2026-09-08 must not come back: under plain
    winner-only learning the winning centre moved toward every frame it won, however far away, and one slot
    ended with 96.5% of occupancy. Now that a moment changes only its own slot, a frame that sits outside the admission rule's
    line of every row moves NOTHING - here 2,000 frames at cosine 0.5 per sense from the nearest row leave
    all three centres bit-identical and read moments_with_a_slot 0 - while frames inside the line of row 0
    move row 0 and only row 0."""
    torch.manual_seed(0)
    dm, b = 16, 8
    h = HierarchicalSelfDark(["a", "b"], dm, n_slots=3)
    rows = [torch.cat([F.normalize(torch.randn(dm), dim=-1), F.normalize(torch.randn(dm), dim=-1)]) for _ in range(3)]
    h.centers[:3] = F.normalize(torch.stack(rows), dim=-1); h._filled_cpu = 3; h.filled.fill_(3)
    before = h.centers.clone()
    far = torch.cat([_near(rows[0][:dm], 0.5), _near(rows[0][dm:], 0.5)])
    for _ in range(2000):
        x = far.unsqueeze(0).repeat(b, 1) + 0.01 * torch.randn(b, 2 * dm)
        h.observe({"a": x[:, :dm], "b": x[:, dm:]}, {"w": torch.zeros(b, 1)})
    assert torch.equal(h.centers, before), "a frame outside every row's line moved a centre"
    st = h.stats()
    assert st["moments_with_a_slot"] == 0.0 and 0.3 < st["best_slot_cos"] < 0.7
    assert st["shadow"]["distinct"] >= 1, "the far frames are what the admission rule calls new"
    near = torch.cat([_near(rows[0][:dm], 0.97), _near(rows[0][dm:], 0.97)])
    for _ in range(500):
        x = near.unsqueeze(0).repeat(b, 1) + 0.01 * torch.randn(b, 2 * dm)
        h.observe({"a": x[:, :dm], "b": x[:, dm:]}, {"w": torch.zeros(b, 1)})
    assert torch.equal(h.centers[1:3], before[1:3]), "a row was moved by frames that belong to another row"
    moved = float(F.normalize(h.centers[0], dim=-1) @ F.normalize(before[0], dim=-1))
    assert moved < 1.0 - 1e-4, "row 0 did not follow the frames that belong to it"
    assert float(F.normalize(h.centers[0], dim=-1) @ F.normalize(near, dim=-1)) > float(F.normalize(before[0], dim=-1) @ F.normalize(near, dim=-1))


# ---------------------------------------------------------------- audit P240 (2026-09-19)
def test_drift_gauges_keep_their_digits_at_the_runs_table_size():
    """At m_center 0.999 one update turns a centre by at most 1 - cos = 5.0e-07; on a 6,400-slot table the
    mean measured 4.6e-12 and the fastest slot 2.7e-08. stats() rounded to 7 decimals and printed 0.0 for
    both. The report must carry the buffer's value to 4 significant digits, and the lived-terms rows
    (moments for a quarter turn) must exist whenever the rotation is above 0."""
    h, cen, dm = _run_table(6400)
    for _ in range(24):                          # at cosine 0.97 the frames belong to row 123, so it moves
        h.observe(_run_toks(_run_batch(cen, a=0.97), dm), {"w": torch.rand(8, 1)})
    st = h.stats()
    for key in ("center_drift", "center_drift_max", "center_step"):
        buf = float(getattr(h, key))
        assert buf > 0.0
        assert st[key] > 0.0, f"{key}: the buffer holds {buf:.3e} and stats() printed {st[key]!r}"
        assert abs(st[key] - buf) <= 1e-3 * buf, (key, st[key], buf)
    assert st["center_drift_max"] >= st["center_drift"]
    fast, mean = st["center_quarter_turn_moments_fastest"], st["center_quarter_turn_moments_mean"]
    assert fast is not None and mean is not None and 0 < fast <= mean
    theta = 2.0 * math.asin(math.sqrt(float(h.center_drift_max) / 2.0))
    assert abs(fast - (math.pi / 2.0) / theta) <= 1.0
    # a tracker that has not moved reports no quarter-turn time rather than a division by zero
    cold = HierarchicalSelfDark(["a"], 8, n_slots=4, m_center=1.0)
    cold.observe({"a": torch.randn(2, 8)}, {"w": torch.zeros(2, 1)})
    assert cold.stats()["center_quarter_turn_moments_fastest"] is None


# ---------------------------------------------------------------- the frame tap (audit P238)
def test_the_frame_tap_writes_what_the_table_saw_and_changes_nothing(tmp_path, monkeypatch):
    """With HSELF_TAP_DIR and HSELF_TAP_STEPS set, the module writes the fused unit vectors, the
    summed wellness and the frame counter of the first N frame batches, in chunks, and then stops.
    The tapped module ends with every buffer bit-identical to an untapped one fed the same frames,
    and its state dict keeps the same entries - the tap is a recorder, not a participant."""
    torch.manual_seed(0)
    frames = [({"a": torch.randn(4, 8), "b": torch.randn(4, 8)}, {"w": torch.rand(4, 1)}) for _ in range(9)]
    plain = HierarchicalSelfDark(["a", "b"], 8, n_slots=16)
    for tok, g in frames:
        plain.observe(tok, g)
    monkeypatch.setenv("HSELF_TAP_DIR", str(tmp_path / "tap"))
    monkeypatch.setenv("HSELF_TAP_STEPS", "7")
    monkeypatch.setenv("HSELF_TAP_CHUNK", "3")
    tapped = HierarchicalSelfDark(["a", "b"], 8, n_slots=16)
    assert tapped._tap is not None
    for tok, g in frames:
        tapped.observe(tok, g)
    assert tapped._tap is None, "the tap must switch itself off after the frames it was asked for"
    files = sorted((tmp_path / "tap").glob("frames_*.pt"))
    assert [f.name for f in files] == ["frames_000000000_000000002.pt", "frames_000000003_000000005.pt",
                                       "frames_000000006_000000006.pt"]
    got = [torch.load(f) for f in files]
    x = torch.cat([g["x"] for g in got]); w = torch.cat([g["w"] for g in got])
    assert x.shape == (7, 4, 16) and w.shape == (7, 4)
    assert torch.cat([g["frames"] for g in got]).tolist() == list(range(7))
    for i in range(7):
        tok, g = frames[i]
        want = torch.nn.functional.normalize(torch.cat([tok["a"], tok["b"]], dim=-1), dim=-1)
        assert torch.equal(x[i], want) and torch.allclose(w[i], g["w"].reshape(-1))
    sp, st = plain.state_dict(), tapped.state_dict()
    assert list(sp) == list(st)
    for k in sp:
        assert torch.equal(sp[k], st[k]), k
