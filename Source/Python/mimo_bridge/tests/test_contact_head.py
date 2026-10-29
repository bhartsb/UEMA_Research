# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The contact head and the contact_control channel (docs/gain_of_function_v3/DESIGN.md D10).

What is pinned here: the head's shapes and targets (53 patches, three bits from the world's
0-3 patch state), the masking (a body without a report, and the "given contact" outputs on
untouched patches, grade nothing), the order columns it reads (no position request of a handle
the body only turns), the centring that removes a shared offset exactly and survives a
non-finite latent, the onset success signal judged against EACH PATCH'S OWN line (both touching
parts foreseen, structural pairs never, a side already touched or without skin exempt), the two
evidence gates (one per kind of touch, one reading per control moment, two lines, shut when their
evidence stops arriving and at every load), the habituation that stops a touch-release loop
paying, an outside touch paid once per body-moment, contact without a foreseen onset paying
nothing, the size-matched shuffled-order control on the same onsets, the onset AUCs beside their
base-rate control, the pseudo-count on the base rates, the running value the ranking reads, and
the registration rules every new module follows (appended last, side module, own clip group,
receipt stamp, refusal, exact migration, runtime state, companion).

Rewritten by lane-b4-contact (2026-09-23) after the review docs/gain_of_function_v3/reviews/
b4-self-contact.md: the first file swapped the two measured margins for one number, 0.1, so a swap
of the two lines would have passed (minor 11); every test here gives the own-part and the outside
line different values, and one test gives two patches different lines."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lem.model import contact_head as CH
from lem.model.contact_head import (ContactHead, ContactTracker, PATCH_INDEX, PATCH_NAMES,
                                    N_PATCHES, order_features, state_bits, habituation_factor,
                                    record_onset)
from mimo_bridge.lem_model import LemLite

SELF_LINE, WORLD_LINE = 0.3, 0.1        # different on purpose: a swap of the two must fail


@pytest.fixture(autouse=True)
def _gates_never_stale(monkeypatch):
    """The tests step far apart in time; only the staleness test uses the real rule."""
    monkeypatch.setattr(CH, "CONTACT_GATE_STALE_STEPS", 10 ** 9)


def _quat_z(deg):
    a = math.radians(deg)
    return torch.tensor([0.0, 0.0, math.sin(a / 2), math.cos(a / 2)])


# ------------------------------------------------------------------------------ the module
def test_the_patch_list_is_53_unique_skin_bodies_with_an_index_and_the_worlds_regions():
    from mimo_bridge.touch_adapter import region_of
    assert N_PATCHES == 53 and len(set(PATCH_NAMES)) == 53
    assert all(PATCH_INDEX[n] == i for i, n in enumerate(PATCH_NAMES))
    assert "left_eye" not in PATCH_INDEX and "right_thhub" in PATCH_INDEX
    assert list(CH.PATCH_REGIONS) == [region_of(n) for n in PATCH_NAMES], \
        "the false-alarm regions are the world's own skin regions"
    assert len(CH.REGIONS) == 14 and "" not in CH.REGIONS


def test_the_world_state_becomes_three_bits():
    s = torch.tensor([0, 1, 2, 3])
    b = state_bits(s)
    assert b.tolist() == [[False, False, False], [True, False, True],
                          [True, True, False], [True, True, True]]


def test_head_shapes_and_an_own_part_is_never_likelier_than_any_contact():
    torch.manual_seed(0)
    h = ContactHead(d_model=16, n_mod=3, n_body_handles=4)
    lg, pre = h(torch.randn(2, 3, 16), torch.randn(2, 4, CH.ORDER_F), torch.zeros(2, N_PATCHES, 2),
                return_pre=True)
    assert lg.shape == (2, N_PATCHES, 3)
    assert set(pre) == set(ContactHead.LIVE_LAYERS)
    p = ContactHead.probabilities(lg)
    assert p.shape == (2, N_PATCHES, 3)
    assert bool((p[..., 1] <= p[..., 0] + 1e-7).all()) and bool((p[..., 2] <= p[..., 0] + 1e-7).all())


def test_the_head_reads_no_position_request_of_a_handle_the_body_only_turns():
    """Review of b4, minor 3: 11 of the 25 body handles carried position requests the body never
    executes; the head reads position columns only for action_decoder.POSITION_HANDLES."""
    from mimo_bridge.action_decoder import POSITION_HANDLES
    from mimo_bridge.body_layout import names_for
    names = [n for n in names_for("trunk_v1") if n != "gaze"]
    mask = CH.order_input_mask(names, POSITION_HANDLES)
    assert mask.shape == (25, CH.ORDER_F)
    turned_only = [n for n in names if n not in POSITION_HANDLES]
    assert len(turned_only) == 11 and int(mask.sum()) == 25 * 7 - 11 * 3 == 142
    torch.manual_seed(9)
    h = ContactHead(16, 3, len(names), order_mask=mask)
    assert h.ctx_ord.in_features == 142
    lat, prev = torch.randn(2, 3, 16), torch.zeros(2, N_PATCHES, 2)
    o = torch.randn(2, 25, CH.ORDER_F)
    o2 = o.clone()
    i = names.index("thigh_L")
    o2[:, i, 0:3] += 50.0                           # a position request the body ignores
    j = names.index("wrist_L")
    o3 = o.clone()
    o3[:, j, 0:3] += 50.0                           # one it executes
    with torch.no_grad():
        assert torch.equal(h(lat, o, prev), h(lat, o2, prev))
        assert not torch.equal(h(lat, o, prev), h(lat, o3, prev))
    with pytest.raises(ValueError, match="order_mask"):
        ContactHead(16, 3, 4, order_mask=torch.ones(3, CH.ORDER_F, dtype=torch.bool))


def test_the_loss_grades_only_reported_bodies_and_the_conditionals_only_on_touched_patches():
    torch.manual_seed(1)
    lg = torch.randn(2, N_PATCHES, 3, requires_grad=True)
    tgt = torch.zeros(2, N_PATCHES, dtype=torch.long)
    tgt[0, 3] = 2          # own part on patch 3
    tgt[0, 5] = 1          # outside thing on patch 5
    tgt[1, 7] = 3
    bits = state_bits(tgt)
    valid = torch.tensor([True, False])
    terms = ContactHead.loss(lg, bits, valid)
    assert set(terms) == {"contact", "self", "world"}
    sum(terms.values()).backward()
    g = lg.grad
    assert float(g[1].abs().sum()) == 0.0, "a body with no report must not be graded"
    untouched = torch.ones(N_PATCHES, dtype=torch.bool)
    untouched[[3, 5]] = False
    assert float(g[0, untouched][:, 1:].abs().sum()) == 0.0, \
        "'given contact' is undefined where nothing touched"
    assert float(g[0, :, 0].abs().sum()) > 0.0
    # the contact term is the plain mean cross-entropy over the reported body's 53 patches
    want = torch.nn.functional.binary_cross_entropy_with_logits(
        lg[0, :, 0].detach(), bits[0, :, 0].float())
    assert torch.allclose(terms["contact"].detach(), want)
    assert ContactHead.loss(lg, bits, torch.tensor([False, False])) == {}


def test_order_features_are_zero_for_holding_still_and_read_centimetres_and_degrees():
    B, H, gi = 1, 4, 2
    ach = torch.zeros(B, H, 10)
    ach[..., 6] = 1.0
    ach[0, 0, 0] = 0.1                          # 20 cm along x (positions / 200 cm)
    order = torch.zeros(B, H, 9)
    order[..., 0:3] = ach[..., 0:3] * 200.0
    order[..., 3:7] = ach[..., 3:7]
    order[..., 8] = 0.5
    f = order_features(order, ach, gi)
    assert f.shape == (B, H - 1, CH.ORDER_F), "the eyes' handle is left out"
    assert torch.allclose(f[..., :6], torch.zeros(B, H - 1, 6), atol=1e-5)
    assert torch.allclose(f[..., 6], torch.full((B, H - 1), 0.5))
    order[0, 0, 0] += 10.0                      # asked 10 cm further along x
    order[0, 1, 3:7] = _quat_z(90.0)            # asked a 90 degree turn about z
    f = order_features(order, ach, gi)
    assert abs(float(f[0, 0, 0]) - 10.0 / CH.ORDER_POS_CM) < 1e-5
    assert abs(float(f[0, 1, 5]) - 90.0 / CH.ORDER_ROT_DEG) < 1e-3


def test_the_shuffled_order_asks_exactly_the_other_bodys_change_from_this_bodys_pose():
    """Review of b4, minor 2: the rolled absolute order asked larger changes than the bodies' own
    orders (26% larger turns). The control now carries the other body's change, row for row."""
    torch.manual_seed(10)
    B, H, gi = 3, 5, 4
    ach = torch.zeros(B, H, 10)
    ach[..., 0:3] = torch.randn(B, H, 3) * 0.1
    q = torch.randn(B, H, 4)
    ach[..., 3:7] = q / q.norm(dim=-1, keepdim=True)
    order = torch.zeros(B, H, 9)
    order[..., 0:3] = ach[..., 0:3] * 200.0 + torch.randn(B, H, 3) * 8.0
    turn = torch.randn(B, H, 4)
    order[..., 3:7] = turn / turn.norm(dim=-1, keepdim=True)
    order[..., 7] = torch.rand(B, H)
    order[..., 8] = torch.rand(B, H)
    sh = CH.shuffled_order(order, ach, gi)
    want = order_features(order, ach, gi).roll(1, dims=0)
    got = order_features(sh, ach, gi)
    assert torch.allclose(got, want, atol=1e-3), "every asked move and turn is the other body's"
    assert torch.equal(sh[:, gi], order.roll(1, dims=0)[:, gi]), "the eyes' command is the other's"
    assert torch.equal(sh[..., 7:9], order.roll(1, dims=0)[..., 7:9])
    # the absolute roll the first control used asks for other sizes
    old = order_features(order.roll(1, dims=0), ach, gi)
    assert not torch.allclose(old, want, atol=1e-2)


def test_the_centre_removes_a_shared_offset_exactly():
    """A part shared by every moment (DESIGN section 10) must not reach the layers: with the centre
    learned from the moments themselves, adding the same offset to every latent changes nothing."""
    torch.manual_seed(2)
    a = ContactHead(d_model=16, n_mod=3, n_body_handles=4)
    b = ContactHead(d_model=16, n_mod=3, n_body_handles=4)
    b.load_state_dict(a.state_dict())
    lat = torch.randn(4, 3, 16)
    shift = torch.randn(1, 3, 16) * 60.0
    a.update_centre(lat)
    b.update_centre(lat + shift)
    assert abs(a.centre_weight - 1.0 / CH.CENTRE_HORIZON_STEPS) < 1e-12
    assert torch.allclose(a.centre_mean, lat.mean(0), atol=1e-5), \
        "the first moment sets the mean to itself (corrected for starting at zero)"
    o, s = torch.randn(4, 4, CH.ORDER_F), torch.zeros(4, N_PATCHES, 2)
    with torch.no_grad():
        ya = a(lat, o, s)
        yb = b(lat + shift, o, s)
    assert torch.allclose(ya, yb, atol=1e-4)
    assert "centre_mean" not in a.state_dict(), "a running statistic, saved in runtime_state.pt"


def test_a_non_finite_latent_leaves_the_centre_where_it_was():
    """Review of b4, minor 4: one NaN latent used to poison the centre for good."""
    torch.manual_seed(11)
    h = ContactHead(d_model=16, n_mod=3, n_body_handles=4)
    lat = torch.randn(2, 3, 16)
    h.update_centre(lat)
    h.update_centre(lat * 2.0)
    before = h.centre_mean.clone()
    bad = lat.clone()
    bad[0, 1, 5] = float("nan")
    bad[1, 2, 0] = float("inf")
    h.update_centre(bad)
    assert bool(torch.isfinite(h.centre_mean).all())
    assert float(h.centre_mean[1, 5]) == float(before[1, 5]) and float(h.centre_mean[2, 0]) == float(before[2, 0])
    assert not torch.equal(h.centre_mean, before), "the finite elements still learned the moment"
    with torch.no_grad():
        y = h(lat, torch.randn(2, 4, CH.ORDER_F), torch.zeros(2, N_PATCHES, 2))
    assert bool(torch.isfinite(y).all()), "a later finite moment reads a finite prediction"


# ------------------------------------------------------------------------------ the tracker
def _set_line(tr, j, p, value):
    """Give patch p's line for bit row j (0 own part, 1 outside) the value `value` (on the bin
    grid): the whole histogram in the bin whose upper edge is the value, resting on many moments."""
    width = (CH.LINE_HI - CH.LINE_LO) / CH.LINE_BINS
    b = int(round((value - CH.LINE_LO) / width)) - 1
    tr.lh[j, p] = 0.0
    tr.lh[j, p, b] = 1.0
    tr.lw[j, p] = 1.0
    tr.ln[j, p] = 10 * CH.LINE_MIN_OBSERVATIONS


def _open(tr, kind):
    for _ in range(CH.CONTACT_GATE_READINGS):
        CH.gate_add(tr.gates[kind], 1.0, 0)


def _warm(tr, off_rate=0.05, on_rate=0.9, gate=True, self_line=SELF_LINE, world_line=WORLD_LINE):
    """Base rates that rest on enough observations: starting a touch 5 in 100, keeping one 90 in
    100; every patch's own-part line self_line and outside line world_line; and (gate=True) both
    evidence gates already opened by readings where only the body's own order foresaw onsets."""
    tr.m[:, 0, :] = off_rate
    tr.m[:, 1, :] = on_rate
    tr.w[:] = 1.0
    tr.n[:] = 10 * CH.BASE_RATE_MIN_OBSERVATIONS
    for p in range(N_PATCHES):
        _set_line(tr, 0, p, self_line)
        _set_line(tr, 1, p, world_line)
    if gate:
        for k in CH.GATE_KINDS:
            _open(tr, k)


def _bits(B, touches=()):
    """[B, P, 3] bits from (body, patch, state) triples."""
    s = torch.zeros(B, N_PATCHES, dtype=torch.long)
    for b, p, v in touches:
        s[b, p] = v
    return state_bits(s).numpy()


def _q(B, highs=(), value=0.9):
    """[B, P, 3] probabilities at 0.01 except (body, patch, bit) -> value."""
    q = np.full((B, N_PATCHES, 3), 0.01, dtype=np.float32)
    for b, p, k in highs:
        q[b, p, k] = value
    return q


def _onset(a, b, structural=False):
    return {"pair": sorted([a, b]), "regions": ["x", "y"], "structural": structural,
            "in_view": False, "view_deg": 90.0}


def test_the_lines_are_each_patchs_own_95th_percentile_of_its_untouched_moments():
    tr = ContactTracker()
    tr.m[:, 0, :] = 0.05
    tr.w[:] = 1.0
    tr.n[:] = 10 * CH.BASE_RATE_MIN_OBSERVATIONS
    rate = float(tr.rates()[1, 0, 0])
    rng = np.random.default_rng(3)
    pa, pb_ = PATCH_INDEX["right_upper_arm"], PATCH_INDEX["left_foot"]
    none = _bits(4)
    n_moments = CH.LINE_MIN_OBSERVATIONS // 4 + 400
    for i in range(n_moments):
        q = np.full((4, N_PATCHES, 3), rate, dtype=np.float32)
        q[:, pa, 1] = rate + rng.uniform(0.0, 0.6, 4)       # a patch the head is loud on
        q[:, pb_, 1] = rate + rng.uniform(0.0, 0.1, 4)      # a quiet one
        if i == 10:
            assert np.isnan(tr.lines()[0, pa]), "no line before it rests on enough observations"
        tr.update_lines(q, none, none, np.ones(4, dtype=bool), tr.rates()[:, 0, :])
    L = tr.lines()
    assert abs(L[0, pa] - 0.57) < 0.03, L[0, pa]
    assert abs(L[0, pb_] - 0.095) < 0.01, L[0, pb_]
    assert L[0, pa] > 4 * L[0, pb_], "each patch keeps its own line"


def test_each_patch_judges_an_onset_against_its_own_line():
    """Review of b4, M1: one fixed margin for all 53 patches gave 53 false alarms in 100 on the
    right upper arm and 0 on 29 patches. The same excess counts on a quiet patch, not a loud one."""
    tr = ContactTracker()
    _warm(tr)
    rate = float(tr.rates()[1, 0, 0])
    loud, quiet = PATCH_INDEX["right_upper_arm"], PATCH_INDEX["left_foot"]
    _set_line(tr, 1, loud, 0.5)
    _set_line(tr, 1, quiet, 0.05)
    q = _q(1, [(0, loud, 2), (0, quiet, 2)], value=rate + 0.2)
    pay, det = tr.score(5, q, _bits(1), _bits(1, [(0, loud, 1), (0, quiet, 1)]), np.array([True]),
                        [[]], [{}])
    assert det["bodies"][0]["world_predicted"] == [["left_foot", 1.0]]
    s = tr.summary()
    assert s["onsets_world"] == 2 and s["predicted_world"] == 1 and s["success_world"] == 0.5


def test_each_bit_is_judged_against_its_own_line():
    """The own-part line (0.3 here) and the outside line (0.1) are different; the same probability
    0.2 above the base rate counts for an outside touch and not for an own part."""
    tr = ContactTracker()
    _warm(tr)
    rate = float(tr.rates()[1, 0, 0])
    pa, pf = PATCH_INDEX["left_hand"], PATCH_INDEX["left_ffdistal"]
    po = PATCH_INDEX["left_foot"]
    q = _q(1, [(0, pa, 1), (0, pf, 1), (0, po, 2)], value=rate + 0.2)
    pay, det = tr.score(5, q, _bits(1), _bits(1, [(0, pa, 2), (0, pf, 2), (0, po, 1)]),
                        np.array([True]), [[_onset("left_hand", "left_ffdistal")]], [{}])
    assert det["bodies"][0]["self_predicted"] == [] and det["bodies"][0]["world_predicted"] != []
    q = _q(1, [(0, pa, 1), (0, pf, 1), (0, po, 2)], value=rate + 0.35)
    pay, det = tr.score(6, q, _bits(1), _bits(1, [(0, pa, 2), (0, pf, 2), (0, po, 1)]),
                        np.array([True]), [[_onset("left_hand", "left_ffdistal")]], [{}])
    assert det["bodies"][0]["self_predicted"] != []


def test_a_self_onset_counts_only_when_both_touching_parts_were_foreseen():
    tr = ContactTracker()
    _warm(tr)
    pa, pb_ = PATCH_INDEX["left_hand"], PATCH_INDEX["right_ffdistal"]
    cur = _bits(1, [(0, pa, 2), (0, pb_, 2)])
    prev = _bits(1)
    habit = [{}]
    # only one side foreseen: not predicted, pays nothing
    pay, _ = tr.score(10, _q(1, [(0, pa, 1)]), prev, cur, np.array([True]),
                      [[_onset("left_hand", "right_ffdistal")]], habit)
    assert pay.tolist() == [0.0]
    # both sides foreseen (0.9 > 0.05 + 0.3): predicted, pays 2 (self-contact weighs 2 to 1)
    pay, det = tr.score(300, _q(1, [(0, pa, 1), (0, pb_, 1)]), prev, cur, np.array([True]),
                        [[_onset("left_hand", "right_ffdistal")]], [{}])
    assert pay.tolist() == [CH.SELF_ONSET_PAY]
    assert det["bodies"][0]["self_predicted"][0][:2] == ["left_hand", "right_ffdistal"]
    s = tr.summary()
    assert s["onsets_self"] == 2 and s["scorable_self"] == 2 and s["both_sides_self"] == 2
    assert s["predicted_self"] == 1 and s["success_self"] == 0.5 and s["success_self_both"] == 0.5


def test_a_side_already_touched_or_without_skin_is_not_required_and_an_unknown_name_is_counted():
    tr = ContactTracker()
    _warm(tr)
    ph = PATCH_INDEX["left_hand"]
    # the palm already had an own part on it the moment before: only the finger's side is judged
    pf = PATCH_INDEX["left_ffdistal"]
    prev = _bits(1, [(0, ph, 2)])
    cur = _bits(1, [(0, ph, 2), (0, pf, 2)])
    pay, _ = tr.score(5, _q(1, [(0, pf, 1)]), prev, cur, np.array([True]),
                      [[_onset("left_hand", "left_ffdistal")]], [{}])
    assert pay.tolist() == [CH.SELF_ONSET_PAY]
    # a hand on an eye: the eye has no skin, the hand's side alone decides
    pr = PATCH_INDEX["right_hand"]
    pay, _ = tr.score(6, _q(1, [(0, pr, 1)]), _bits(1), _bits(1, [(0, pr, 2)]), np.array([True]),
                      [[_onset("right_hand", "right_eye")]], [{}])
    assert pay.tolist() == [CH.SELF_ONSET_PAY]
    # a name the patch list does not know is counted, and with no side left it is not scored
    tr.score(7, _q(1), _bits(1), _bits(1), np.array([True]), [[_onset("tail", "wing")]], [{}])
    s = tr.summary()
    assert s["unmapped_sides"] == 2 and s["unscorable_self"] == 1 and s["both_sides_self"] == 0


def test_a_side_whose_patch_has_no_line_yet_waits():
    tr = ContactTracker()
    _warm(tr)
    pa, pb_ = PATCH_INDEX["left_hand"], PATCH_INDEX["right_hand"]
    tr.ln[0, pb_] = 0                              # the right hand has no own-part line yet
    pay, _ = tr.score(5, _q(1, [(0, pa, 1), (0, pb_, 1)]), _bits(1),
                      _bits(1, [(0, pa, 2), (0, pb_, 2)]), np.array([True]),
                      [[_onset("left_hand", "right_hand")]], [{}])
    s = tr.summary()
    assert pay.tolist() == [0.0] and s["unlined_self"] == 1 and s["scorable_self"] == 0


def test_a_pair_touching_at_rest_never_counts_and_never_pays():
    tr = ContactTracker()
    _warm(tr)
    p1, p2 = PATCH_INDEX["left_hand"], PATCH_INDEX["left_thhub"]
    pay, _ = tr.score(9, _q(1, [(0, p1, 1), (0, p2, 1)]), _bits(1),
                      _bits(1, [(0, p1, 2), (0, p2, 2)]), np.array([True]),
                      [[_onset("left_hand", "left_thhub", structural=True)]], [{}])
    assert pay.tolist() == [0.0]
    s = tr.summary()
    assert s["onsets_self"] == 0 and s["onsets_self_structural"] == 1 and s["scorable_self"] == 0


def test_habituation_pays_a_repeated_pair_less_and_recovers_after_200_moments():
    tr = ContactTracker()
    _warm(tr)
    p1, p2 = PATCH_INDEX["right_hand"], PATCH_INDEX["left_hand"]
    q = _q(1, [(0, p1, 1), (0, p2, 1)])
    prev, cur = _bits(1), _bits(1, [(0, p1, 2), (0, p2, 2)])
    habit = [{}]
    pays = []
    for step in (1000, 1010, 1020):          # a touch-release loop inside one goal ring
        pay, _ = tr.score(step, q, prev, cur, np.array([True]), [[_onset("left_hand", "right_hand")]], habit)
        pays.append(float(pay[0]))
    assert pays == pytest.approx([2.0, 1.0, 2.0 / 3.0])
    pay, _ = tr.score(1020 + CH.HABITUATION_STEPS + 1, q, prev, cur, np.array([True]),
                      [[_onset("left_hand", "right_hand")]], habit)
    assert float(pay[0]) == pytest.approx(2.0), "200 moments later the pair pays in full again"
    s = tr.summary()
    assert s["habituation_mean"] == pytest.approx((1.0 + 0.5 + 1.0 / 3.0 + 1.0) / 4, abs=1e-4), \
        "1 = a new pair, lower = repeated"
    # the factor counts every onset of the pair, paid or not
    h = {}
    record_onset(h, "s:a|b", 50)
    record_onset(h, "s:a|b", 60)
    assert habituation_factor(h, "s:a|b", 61) == pytest.approx(1.0 / 3.0)
    assert habituation_factor(h, "s:a|b", 50 + CH.HABITUATION_STEPS + 1) == pytest.approx(0.5)


def test_contact_that_was_not_a_foreseen_onset_pays_nothing():
    tr = ContactTracker()
    _warm(tr)
    p = PATCH_INDEX["chest"]
    # the chest lying on the floor, predicted with certainty, moment after moment: no onset
    for step in range(20):
        pay, _ = tr.score(step, _q(1, [(0, p, 0), (0, p, 2)], value=0.99), _bits(1, [(0, p, 1)]),
                          _bits(1, [(0, p, 1)]), np.array([True]), [[]], [{}])
        assert pay.tolist() == [0.0]
    # an onset the head did not foresee (probability at the base rate): nothing
    pay, _ = tr.score(30, _q(1, [(0, PATCH_INDEX["left_foot"], 2)], value=0.05), _bits(1),
                      _bits(1, [(0, PATCH_INDEX["left_foot"], 1)]), np.array([True]), [[]], [{}])
    assert pay.tolist() == [0.0]
    s = tr.summary()
    assert s["onsets_world"] == 1 and s["predicted_world"] == 0 and s["paid_mean"] == 0.0


def test_an_outside_touch_onset_is_a_patch_that_starts_being_touched():
    tr = ContactTracker()
    _warm(tr)
    p, q_ = PATCH_INDEX["left_foot"], PATCH_INDEX["right_foot"]
    prev = _bits(1, [(0, q_, 1)])               # the right foot was already on the floor
    cur = _bits(1, [(0, p, 1), (0, q_, 1)])
    pay, det = tr.score(40, _q(1, [(0, p, 2), (0, q_, 2)]), prev, cur, np.array([True]), [[]],
                        [{}], world_report=[3])
    assert pay.tolist() == [CH.WORLD_ONSET_PAY]
    assert det["bodies"][0]["world_predicted"] == [["left_foot", 1.0]]
    s = tr.summary()
    assert s["onsets_world"] == 1 and s["success_world"] == 1.0 and s["onsets_world_report"] == 3


def test_outside_touches_that_begin_together_pay_once():
    """Review of b4, minor 5: a roll onto the floor started up to 5 patches at once and paid 5,
    against 2 for a foreseen self-touch."""
    tr = ContactTracker()
    _warm(tr)
    ps = [PATCH_INDEX[n] for n in ("chest", "upper_body", "hip", "left_upper_arm", "head")]
    q = _q(1, [(0, p, 2) for p in ps])
    habit = [{}]
    record_onset(habit[0], CH._key_world(ps[0]), 90)       # the chest touched the floor lately
    pay, det = tr.score(100, q, _bits(1), _bits(1, [(0, p, 1) for p in ps]), np.array([True]),
                        [[]], habit)
    assert pay.tolist() == [CH.WORLD_ONSET_PAY], "one outside onset per body-moment, at the least " \
                                                 "repeated patch's factor"
    assert len(det["bodies"][0]["world_predicted"]) == 5
    s = tr.summary()
    assert s["predicted_world"] == 5 and s["world_pay_events"] == 1


def test_the_shuffled_order_control_is_scored_on_the_same_onsets():
    tr = ContactTracker()
    _warm(tr)
    pa, pb_ = PATCH_INDEX["left_hand"], PATCH_INDEX["right_hand"]
    q = _q(2, [(0, pa, 1), (0, pb_, 1)])
    qc = _q(2)                                  # with another body's order nothing is foreseen
    cur = _bits(2, [(0, pa, 2), (0, pb_, 2)])
    tr.score(3, q, _bits(2), cur, np.array([True, True]),
             [[_onset("left_hand", "right_hand")], []], [{}, {}], q_control=qc)
    s = tr.summary()
    assert s["success_paired"] == 1.0 and s["success_control"] == 0.0 and s["control_steps"] == 1
    assert s["control_onsets_self"] == 1


def test_each_kind_pays_only_while_its_own_gate_is_open():
    """Review of b4, M1: one pooled gate let outside-touch evidence open the pay for
    self-touches (70 of its 157 readings on m1 came from outside touches)."""
    tr = ContactTracker()
    _warm(tr, gate=False)
    pa, pb_, pf = PATCH_INDEX["left_hand"], PATCH_INDEX["right_hand"], PATCH_INDEX["left_foot"]
    q = _q(2, [(0, pa, 1), (0, pb_, 1), (0, pf, 2), (1, pf, 2)])
    cur = _bits(2, [(0, pa, 2), (0, pb_, 2), (0, pf, 1), (1, pf, 1)])
    ons = [[_onset("left_hand", "right_hand")], []]
    # outside touches foreseen only with the body's own order: their gate opens, the self one not
    for i in range(CH.CONTACT_GATE_READINGS):
        tr.score(10 + i, _q(2, [(0, pf, 2), (1, pf, 2)]), _bits(2), _bits(2, [(0, pf, 1), (1, pf, 1)]),
                 np.array([True, True]), [[], []], [{}, {}], q_control=_q(2))
    assert tr.gates["world"]["open"] and not tr.gates["self"]["open"]
    pay, det = tr.score(100, q, _bits(2), cur, np.array([True, True]), ons, [{}, {}])
    assert pay.tolist() == [CH.WORLD_ONSET_PAY, CH.WORLD_ONSET_PAY], "the self-touch is not paid"
    assert det["bodies"][0]["would_pay"] == CH.SELF_ONSET_PAY + CH.WORLD_ONSET_PAY
    assert det["gate_world_open"] and not det["gate_self_open"]
    s = tr.summary()
    assert s["gate_world_open"] is True and s["gate_self_open"] is False and s["gate_open"] is True
    assert s["gate_world_readings"] == CH.CONTACT_GATE_READINGS and s["gate_self_readings"] == 0


def test_a_gate_takes_one_reading_per_control_moment_and_opens_and_closes_on_two_lines():
    tr = ContactTracker()
    _warm(tr, gate=False)
    pf, pg = PATCH_INDEX["left_foot"], PATCH_INDEX["right_foot"]
    # three bodies, two outside onsets each, one control moment: ONE reading, their mean
    q = _q(3, [(b, p, 2) for b in range(3) for p in (pf, pg)])
    qc = _q(3, [(0, pf, 2)])                   # the control foresaw one of the six
    tr.score(1, q, _bits(3), _bits(3, [(b, p, 1) for b in range(3) for p in (pf, pg)]),
             np.array([True] * 3), [[], [], []], [{}, {}, {}], q_control=qc)
    g = tr.gates["world"]
    assert list(g["readings"]) == [pytest.approx(5.0 / 6.0)]
    # the two lines, on the gate itself: above 3 errors it is open, under 1 shut, and in between
    # it keeps the state it had (so it is seen in that band both open and shut)
    g = CH.new_gate("self")
    for v in [0.5, -0.1, 0.3, 0.2] * 8:
        CH.gate_add(g, v, 0)
    n, mean, se, above, below = CH.gate_stats(g)
    assert g["open"] and mean > CH.CONTACT_GATE_OPEN_NOISE * se and (above, below) == (24, 8)
    prev, band = g["open"], set()
    for v in [0.0, 0.1, -0.1, 0.05] * 2 + [-0.2, 0.0, -0.1, 0.0] * 6 + [0.2, 0.0, 0.1, 0.0] * 8:
        CH.gate_add(g, v, 0)
        n, mean, se, _, _ = CH.gate_stats(g)
        r = mean / se
        if r > CH.CONTACT_GATE_OPEN_NOISE:
            assert g["open"]
        elif r < CH.CONTACT_GATE_CLOSE_NOISE:
            assert not g["open"]
        else:
            assert g["open"] == prev, "between the two lines the gate keeps its state"
            band.add(g["open"])
        prev = g["open"]
    assert band == {True, False}, "the band was crossed both ways"
    g2 = CH.new_gate("world")
    for v in [0.0, 0.0, -0.5, 0.25] * 8:
        CH.gate_add(g2, v, 0)
    assert not g2["open"] and "It is shut" in CH.gate_why(g2)
    assert "0." not in CH.gate_why(g2).split("average advantage")[0], "counts, not score decimals"


def test_a_gate_shuts_when_its_readings_stop_arriving(monkeypatch):
    """Review of b4, minor 1: nothing shut a gate whose evidence stopped (at one body the control
    never runs, so a gate loaded open paid for ever)."""
    monkeypatch.setattr(CH, "CONTACT_GATE_STALE_STEPS", 200)
    tr = ContactTracker()
    _warm(tr, gate=False)
    for _ in range(CH.CONTACT_GATE_READINGS):
        CH.gate_add(tr.gates["self"], 1.0, 1000)
    pa, pb_ = PATCH_INDEX["left_hand"], PATCH_INDEX["right_hand"]
    q = _q(1, [(0, pa, 1), (0, pb_, 1)])
    cur = _bits(1, [(0, pa, 2), (0, pb_, 2)])
    ons = [[_onset("left_hand", "right_hand")]]
    pay, _ = tr.score(1200, q, _bits(1), cur, np.array([True]), ons, [{}])
    assert pay.tolist() == [CH.SELF_ONSET_PAY], "200 moments after the last reading: still open"
    pay, _ = tr.score(1201, q, _bits(1), cur, np.array([True]), ons, [{}])
    assert pay.tolist() == [0.0] and not tr.gates["self"]["open"]
    assert len(tr.gates["self"]["readings"]) == 0, "a stale gate drops its readings"
    s = tr.summary()
    assert s["gate_self_stale_shuts"] == 1 and "moments old" in s["gate_self_why"]


def test_the_channel_pays_nothing_until_foreseeing_beats_the_shuffled_order_control():
    tr = ContactTracker()
    _warm(tr, gate=False)
    pa, pb_ = PATCH_INDEX["left_hand"], PATCH_INDEX["right_hand"]
    q = _q(2, [(0, pa, 1), (0, pb_, 1)])
    cur = _bits(2, [(0, pa, 2), (0, pb_, 2)])
    ons = [[_onset("left_hand", "right_hand")], []]
    # foreseen, but no controlled evidence yet: scored and reported, not paid
    pay, det = tr.score(10, q, _bits(2), cur, np.array([True, True]), ons, [{}, {}])
    assert pay.tolist() == [0.0, 0.0] and det["bodies"][0]["would_pay"] == CH.SELF_ONSET_PAY
    assert not det["gate_open"]
    # the control foresees the same onsets with another body's order: readings of 0, gate stays shut
    for i in range(CH.CONTACT_GATE_READINGS):
        tr.score(20 + 300 * i, q, _bits(2), cur, np.array([True, True]), ons, [{}, {}], q_control=q)
    assert tr.gate("self")[0] is False
    s = tr.summary()
    assert s["paid_mean"] == 0.0 and s["would_pay_mean"] > 0 and s["gate_self_open"] is False
    # only this body's order foresees them: readings of +1 open the gate, and the next moment pays
    for i in range(CH.CONTACT_GATE_READINGS):
        tr.score(20000 + 300 * i, q, _bits(2), cur, np.array([True, True]), ons, [{}, {}],
                 q_control=_q(2))
    assert tr.gate("self")[0] is True
    pay, det = tr.score(90000, q, _bits(2), cur, np.array([True, True]), ons, [{}, {}])
    assert pay.tolist() == [CH.SELF_ONSET_PAY, 0.0] and det["gate_open"]
    # nothing reads the payment: scored and reported, never paid
    pay, det = tr.score(90010, q, _bits(2), cur, np.array([True, True]), ons, [{}, {}], pay_on=False)
    assert pay.tolist() == [0.0, 0.0] and det["bodies"][0]["would_pay"] == CH.SELF_ONSET_PAY
    # a load never inherits an open gate or its readings (review of b4, minor 1)
    sd = tr.state_dict()
    assert "gate_readings" not in sd
    other = ContactTracker()
    assert other.load_state_dict(sd)
    assert other.gate("self")[0] is False and len(other.gates["self"]["readings"]) == 0
    assert np.array_equal(other.rates(), tr.rates(), equal_nan=True)
    assert np.array_equal(other.lines(), tr.lines(), equal_nan=True)


def test_false_alarms_are_untouched_patch_moments_over_their_line_by_region():
    tr = ContactTracker()
    _warm(tr)
    q = _q(1, [(0, PATCH_INDEX["hip"], 1), (0, PATCH_INDEX["lower_body"], 1)])
    tr.score(1, q, _bits(1), _bits(1), np.array([True]), [[]], [{}])
    s = tr.summary()
    assert s["false_alarm_self"] == pytest.approx(2.0 / N_PATCHES, abs=1e-4)
    assert s["false_alarm_world"] == 0.0
    assert s["false_alarm_by_region"]["trunk"] == {"self": 0.5, "world": 0.0}
    assert s["false_alarm_by_region"]["left hand"] == {"self": 0.0, "world": 0.0}


def test_the_onset_aucs_read_only_patches_untouched_the_moment_before():
    """Review of b4, M3: the any-contact AUC mostly measured whether a touch continues."""
    tr = ContactTracker()
    _warm(tr)
    pa, pc, ph = PATCH_INDEX["left_hand"], PATCH_INDEX["chest"], PATCH_INDEX["hip"]
    # continuing touches on the chest and hip, predicted well: in the any-contact AUC, not in the
    # onset AUC
    prev = _bits(1, [(0, pc, 2), (0, ph, 2)])
    cur = _bits(1, [(0, pc, 2), (0, ph, 2), (0, pa, 2)])
    q = _q(1, [(0, pc, 1), (0, pc, 0), (0, ph, 1), (0, ph, 0)], value=0.99)
    q[0, pa, 1] = 0.005                        # the new own-part touch was NOT foreseen
    tr.score(1, q, prev, cur, np.array([True]), [[]], [{}])
    s = tr.summary()
    assert s["onset_auc_self"] < 0.1, "the only onset scored below every quiet patch"
    assert s["head_auc_bits"]["self"] > 0.5
    # the base-rate control on the same patch-moments: one rate for every patch -> a coin flip
    assert s["control_onset_auc_self"] == pytest.approx(0.5)
    assert s["onset_auc_world"] is None, "no outside onset in the window"
    q[0, pa, 1] = 0.9
    tr.score(2, q, prev, cur, np.array([True]), [[]], [{}])
    assert tr.summary()["onset_auc_self"] == pytest.approx(1.0)


def test_the_auc_reads_how_often_a_touched_patch_scored_above_an_untouched_one():
    h = CH._Hist()
    h.add(np.array([0.9, 0.8, 0.1, 0.2]), np.array([True, True, False, False]))
    assert h.auc() == pytest.approx(1.0)
    h = CH._Hist()
    h.add(np.array([0.1, 0.2, 0.9, 0.8]), np.array([True, True, False, False]))
    assert h.auc() == pytest.approx(0.0)
    h = CH._Hist()
    h.add(np.array([0.5, 0.5]), np.array([True, False]))
    assert h.auc() == pytest.approx(0.5), "a tie is a coin flip"
    assert CH._Hist().auc() is None


def test_base_rates_depend_on_the_moment_before_count_observations_and_never_read_zero():
    tr = ContactTracker()
    P = N_PATCHES
    rng = np.random.default_rng(0)
    # two bodies: each (bit, previous state, patch) cell gets one observation per moment, so a
    # rate is used after BASE_RATE_MIN_OBSERVATIONS moments here (at 8 bodies after 13)
    for i in range(CH.BASE_RATE_MIN_OBSERVATIONS + 50):
        prev = np.zeros((2, P, 3), dtype=bool)
        prev[1, 0] = True                        # body 1: patch 0 touched before, stays touched
        cur = prev.copy()
        cur[0, 0] = rng.random(3) < 0.2          # body 0: patch 0 starts a touch 20 in 100 times
        if i == CH.BASE_RATE_MIN_OBSERVATIONS - 2:
            assert np.isnan(tr.rates()[:, :, 0]).all(), "no base rate before 100 observations"
        tr.update_rates(prev, cur, np.array([True, True]))
    r = tr.rates()
    assert r[0, 1, 0] == pytest.approx(1.0, abs=0.01), "a kept touch stays: rate 1 given touched before"
    assert 0.1 < r[0, 0, 0] < 0.3, "starting a touch from untouched: about 20 in 100"
    # a patch never seen to start a touch reads half an observation's rate, never 0 (minor 9)
    assert 0.0 < r[0, 0, 5] < 0.01
    at_full = (0 + CH.BASE_RATE_PSEUDO * tr.alpha) / (1.0 + tr.alpha)
    assert at_full == pytest.approx(0.5 / (CH.BASE_RATE_HORIZON_STEPS + 1))


def test_the_tracker_state_round_trips_and_refuses_another_patch_list():
    tr = ContactTracker()
    _warm(tr, 0.07, 0.8)
    sd = tr.state_dict()
    other = ContactTracker()
    assert other.load_state_dict(sd) and np.array_equal(other.rates(), tr.rates())
    assert np.array_equal(other.lines(), tr.lines())
    bad = dict(sd, patch_names=list(PATCH_NAMES[:-1]))
    assert not ContactTracker().load_state_dict(bad)
    assert not ContactTracker().load_state_dict(None)
    # a version-1 state (no lines) restores its base rates and starts the lines cold
    v1 = {k: v for k, v in sd.items() if k not in ("lh", "lw", "ln")}
    cold = ContactTracker()
    assert cold.load_state_dict(v1) and np.isnan(cold.lines()).all()


def test_the_tracker_is_plain_data_the_probe_snapshot_can_copy_exactly():
    """The probe harness snapshots a model through mimo_bridge/copied_execution, which refuses any
    object inside a tracker that is not a listed class or a plain value (the first form of the gates,
    a class of their own, was refused there on the checkpoint test)."""
    from mimo_bridge.copied_execution import _deepcopy_exact
    tr = ContactTracker()
    _warm(tr)
    _deepcopy_exact(tr, "contact_tracker")


def test_the_value_the_ranking_reads_is_each_bodys_running_average_of_its_payments():
    """Review of b4, M2 (CONTACT_VALUE): a lone payment of 2 no longer arrives as one moment 5.5
    of the ranking's units above the rest."""
    hold = 5
    a = 1.0 / (CH.CONTACT_VALUE_ORDERS * hold)
    st, v = CH.running_value(None, np.array([2.0, 0.0]), np.array([True, False]), hold)
    assert v[0] == pytest.approx(2.0) and np.isnan(v[1]), "the first scored moment is its own average"
    st, v = CH.running_value(st, np.array([0.0, 0.0]), np.array([True, True]), hold)
    assert v[0] == pytest.approx((1 - a) * a * 2.0 / ((1 - a) * a + a))
    assert v[1] == pytest.approx(0.0)
    for _ in range(200):
        st, v = CH.running_value(st, np.array([0.0, 0.0]), np.array([True, True]), hold)
    assert v[0] < 0.01
    # another body count starts empty
    st2, v2 = CH.running_value(st, np.zeros(3), np.ones(3, dtype=bool), hold)
    assert v2.tolist() == [0.0, 0.0, 0.0]


def test_liveness_reports_the_share_of_units_that_rose_above_the_line():
    tr = ContactTracker()
    x = torch.full((2, 4), -10.0)
    x[0, 1] = -2.0                              # one of four units rose above -3
    tr.add_liveness({"ctx1": x})
    tr.add_liveness({"ctx1": torch.full((2, 4), -10.0)})
    assert tr.liveness_summary() == {"ctx1": 0.25}
    assert tr.liveness_summary() is None, "reset for the next window"


def test_the_control_period_never_shares_a_factor_with_the_order_hold():
    for hold in (1, 2, 3, 5, 7, 10, 14, 77):
        assert math.gcd(CH.control_period(hold), hold) == 1
    assert CH.control_period(5) == 7


# ------------------------------------------------------------------------------ the model
def _lite(**kw):
    torch.set_num_threads(1)
    return LemLite(device="cpu", seed=0, imagination=False, no_vision=True,
                   next_frame_loss=True, body_layout="trunk_v1", **kw)


def _latent(m, B):
    toks = {mm: torch.randn(B, 1, m.d_model) for mm in m.active}
    ctx = torch.zeros(B, (m.ctx_len - 1) * m.n_mod, m.d_model)
    _, last, _ = m.forward_model(toks, ctx)
    return last


def _pose(B, H):
    p = torch.zeros(B, H, 10)
    p[..., 6] = 1.0
    return p


def _frames(B, states, onsets=None):
    out = []
    for b in range(B):
        t = {"contact_patch_state_burst": [int(v) for v in states[b]],
             "self_contact_onsets": (onsets or [[]] * B)[b], "contact_onsets_world": 0}
        out.append(SimpleNamespace(meta={"task": t}))
    return out


def test_flag_off_builds_nothing_and_flag_on_appends_last_registers_and_stamps():
    off = _lite()
    on = _lite(contact_head=True)
    assert off.contact_head is None and "contact_head" not in off._side_modules()
    assert "contact_head" not in off.receipt and not off.contact_control_enabled
    n = sum(p.numel() for p in on.contact_head.parameters())
    assert on.n_params == off.n_params + n
    assert [p.shape for p in on.params[:len(off.params)]] == [p.shape for p in off.params]
    ids = {id(p) for p in on.contact_head.parameters()}
    assert all(id(p) in ids for p in on.params[len(off.params):]), "appended LAST"
    assert on._side_modules()["contact_head"] is on.contact_head
    st = on.receipt["contact_head"]
    assert st["patch_names"] == list(PATCH_NAMES) and st["version"] == CH.CONTACT_HEAD_VERSION
    assert st["order_scales"] == {"pos_cm": CH.ORDER_POS_CM, "rot_deg": CH.ORDER_ROT_DEG}
    assert st["order_inputs"] == 142 and st["line_quantile"] == CH.LINE_QUANTILE
    assert st["gate"]["open_noise"] == CH.CONTACT_GATE_OPEN_NOISE
    assert on.contact_head.ctx_ord.in_features == 142
    assert on.contact_head_weight == CH.CONTACT_HEAD_WEIGHT
    # built in a forked random stream: the run's own stream is where a build without it leaves it
    torch.manual_seed(5)
    _lite()
    a = torch.rand(3)
    torch.manual_seed(5)
    _lite(contact_head=True)
    assert torch.equal(a, torch.rand(3))


def test_what_is_refused():
    with pytest.raises(ValueError, match="next_frame_loss"):
        LemLite(device="cpu", seed=0, imagination=False, no_vision=True, contact_head=True)
    with pytest.raises(ValueError, match="--contact-head on"):
        _lite(contact_control_wellness=0.5)
    with pytest.raises(ValueError, match="0 or more"):
        _lite(contact_head=True, contact_control_wellness=-1.0)


def test_contact_control_joins_the_normalised_ranking_at_its_weight_and_never_the_raw_sum():
    from lem.model import wellness_channels as WC
    assert WC.FLOORS.get("contact_control") is not None, "the channel's floor is measured"
    kw = dict(intention=True, wellness_rank="normalized", curiosity_goal=False, contact_head=True)
    m = _lite(contact_control_wellness=0.5, **kw)
    assert m.contact_control_enabled and "contact_control" in m._rank_channel_names
    assert m.rank_weights["contact_control"] == 0.5
    assert m.receipt["wellness_channels"]["weights"]["contact_control"] == 0.5
    m2 = _lite(contact_control_wellness=0.8, wellness_weights="contact_control=0.8", **kw)
    assert m2.rank_weights["contact_control"] == 0.8
    with pytest.raises(ValueError, match="given twice"):
        _lite(contact_control_wellness=0.5, wellness_weights="contact_control=0.3", **kw)
    # with the channel off it is not a ranking channel at all, its weight in force is 0, and
    # naming one is refused (review of b4, minor 7)
    off = _lite(**kw)
    assert "contact_control" not in off._rank_channel_names
    assert off.rank_weights["contact_control"] == 0.0
    assert off.receipt["wellness_channels"]["weights"]["contact_control"] == 0.0
    with pytest.raises(ValueError, match="channel is off"):
        _lite(wellness_weights="contact_control=0.8", **kw)
    # the raw grounded sum never carries it: the six grounded terms are unchanged
    B = 2
    names = m._rank_channel_names
    m._contact_control_value = torch.tensor([2.0, 0.0])
    wo = {"grounded": {k: torch.full((B, 1), 0.5) for k in
                       ("prediction_error", "tactile_comfort", "capability", "coherence",
                        "motor_state", "self_touch")}}
    frames = [SimpleNamespace(vestibular=None) for _ in range(B)]
    v = m._wellness_channel_vector(B, wo, torch.zeros(B), None, None, torch.zeros(B), 0.0, frames)
    assert v[names.index("contact_control")].tolist() == [2.0, 0.0]
    assert "contact_control" not in wo["grounded"]


def test_the_lesson_grades_the_next_moment_into_the_core_and_caches_the_skin(monkeypatch):
    torch.manual_seed(3)
    m = _lite(contact_head=True)
    B, H = 2, len(m.handle_names)
    exec_t = torch.zeros(B, H, 9)
    exec_t[..., 6] = 1.0
    m._nf_ach = _pose(B, H)
    s0 = np.zeros((B, N_PATCHES), dtype=np.int64)
    s1 = s0.copy()
    s1[0, PATCH_INDEX["left_hand"]] = 2
    s1[0, PATCH_INDEX["right_hand"]] = 2
    # the first moment has no past: nothing graded, the value unmeasured, the skin cached
    out = m._contact_lesson(_latent(m, B), None, None, exec_t, _frames(B, s0), 10, B)
    assert out == {} and bool(torch.isnan(m._contact_control_value).all())
    assert m._contact_prev_state.tolist() == s0.tolist()
    # the next: graded (the core gets gradient), scored, the skin of this moment cached
    frames = _frames(B, s1, [[_onset("left_hand", "right_hand")], []])
    out = m._contact_lesson(_latent(m, B), None, None, exec_t, frames, 11, B)
    assert set(out) == {"contact_head"}
    out["contact_head"].backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in m.forward_model.parameters()), "the next-moment grading trains the core"
    v = m._contact_control_value
    assert v.shape == (B,) and bool(torch.isfinite(v).all()), "reported bodies read a measured value"
    assert v.tolist() == [0.0, 0.0], "no base rate yet: nothing can have been foreseen"
    assert m._contact_prev_state.tolist() == s1.tolist()
    assert m.contact_head.centre_weight > 0
    # a body without a report is not graded and reads unmeasured (the ranking's centre)
    fr = _frames(B, s1)
    fr[1].meta["task"] = {}
    m._contact_lesson(_latent(m, B), None, None, exec_t, fr, 12, B)
    assert math.isnan(float(m._contact_control_value[1])) and not math.isnan(float(m._contact_control_value[0]))
    w = m.contact_control_window_summary()
    assert w["onsets_self"] == 1 and w["scorable_self"] == 0
    assert w["head_bce"] is None, "no base-rate control yet, so no paired cross-entropy"
    assert set(w["liveness"]) == set(ContactHead.LIVE_LAYERS) and w["pool_spread"] == {}
    assert w["channel"] == {"weight": 0.0, "in_ranking": False}
    for k in ("onset_auc_self", "onset_auc_world", "control_onset_auc_self", "control_onset_auc_world",
              "gate_self_open", "gate_world_open", "success_paired", "success_control",
              "habituation_mean"):
        assert k in w, k


def test_the_payment_is_not_made_when_no_ranking_reads_it(monkeypatch):
    """Review of b4, minor 6: pay_on was never passed, so paid_mean reported payments that went
    nowhere (the channel off, or the ranking raw)."""
    seen = []
    m = _lite(contact_head=True)
    real = m.contact_tracker.score

    def spy(*a, **k):
        seen.append(k.get("pay_on"))
        return real(*a, **k)
    m.contact_tracker.score = spy
    B, H = 1, len(m.handle_names)
    exec_t = torch.zeros(B, H, 9)
    exec_t[..., 6] = 1.0
    m._nf_ach = _pose(B, H)
    s = np.zeros((B, N_PATCHES), dtype=np.int64)
    with torch.no_grad():
        for step in (1, 2):
            m._contact_lesson(_latent(m, B), None, None, exec_t, _frames(B, s), step, B)
    assert seen == [False]
    on = _lite(contact_head=True, contact_control_wellness=0.5, intention=True,
               wellness_rank="normalized", curiosity_goal=False)
    assert on.contact_control_enabled and on.wellness_rank_mode == "normalized"


def test_a_reset_forgets_the_bodys_history_and_keeps_the_statistics():
    m = _lite(contact_head=True)
    m._contact_prev_state = torch.zeros(2, N_PATCHES, dtype=torch.long)
    m._contact_prev_valid = torch.ones(2, dtype=torch.bool)
    m._contact_habit = [{"s:a|b": [1]}, {}]
    m._contact_value_state = [np.ones(2), np.ones(2)]
    m.contact_head.update_centre(torch.randn(2, m.n_mod, m.d_model))
    _warm(m.contact_tracker)
    m.reset_transition_state()
    assert m._contact_prev_state is None and m._contact_prev_valid is None and m._contact_habit is None
    assert m._contact_value_state is None
    assert m.contact_head.centre_weight > 0 and np.isfinite(m.contact_tracker.rates()).all()


def test_every_new_attribute_serialises_into_the_episode_companion():
    from mimo_bridge import continuation_state as CS
    torch.manual_seed(4)
    m = _lite(contact_head=True)
    B, H = 2, len(m.handle_names)
    exec_t = torch.zeros(B, H, 9)
    exec_t[..., 6] = 1.0
    m._nf_ach = _pose(B, H)
    s = np.zeros((B, N_PATCHES), dtype=np.int64)
    for step in (1, 2):
        with torch.no_grad():
            m._contact_lesson(_latent(m, B), None, None, exec_t, _frames(B, s), step, B)
    m._contact_habit = [{"s:left_hand|right_hand": [1, 2]}, {}]
    state = CS.capture_learner(m, include_replay=False)
    for k in ("_contact_prev_state", "_contact_prev_valid", "_contact_habit",
              "_contact_control_value", "_contact_value_state", "last_contact_control"):
        assert k in state["data"], k
    assert "contact_tracker" not in state["data"] and "contact_head" not in state["data"]
    other = _lite(contact_head=True)
    CS.restore_learner(other, state)
    assert other._contact_habit == m._contact_habit
    assert torch.equal(other._contact_prev_state.cpu(), m._contact_prev_state.cpu())
    assert np.array_equal(other._contact_value_state[0], m._contact_value_state[0])


def test_the_head_is_clipped_as_its_own_group():
    m = _lite(contact_head=True)
    for p in m.params:
        p.grad = torch.zeros_like(p)
    core = next(m.forward_model.parameters())
    core.grad.fill_(1e-4)
    for p in m.contact_head.parameters():
        p.grad.fill_(10.0)
    before = core.grad.clone()
    norm = m._clip_training_gradients()
    assert torch.equal(core.grad, before), "the core's step was shrunk by the head's gradient"
    assert abs(norm - float(before.norm())) < 1e-6


def test_the_running_statistics_ride_the_checkpoint_and_a_build_without_the_head_refuses(tmp_path):
    torch.manual_seed(6)
    m = LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                body_layout="trunk_v1", contact_head=True, checkpoint_dir=str(tmp_path))
    m.contact_head.update_centre(torch.randn(3, m.n_mod, m.d_model))
    _warm(m.contact_tracker, 0.03, 0.7)
    assert m.contact_tracker.gate("self")[0], "open before the save"
    path = m.save_checkpoint(7)
    other = LemLite(device="cpu", seed=1, imagination=False, no_vision=True, next_frame_loss=True,
                    body_layout="trunk_v1", contact_head=True)
    res = other.load_checkpoint(path)
    assert "contact_head" not in (res.get("reinitialized_components") or [])
    assert torch.equal(other.contact_head.centre_mean, m.contact_head.centre_mean)
    assert other.contact_head.centre_weight == m.contact_head.centre_weight
    assert np.array_equal(other.contact_tracker.rates(), m.contact_tracker.rates())
    assert np.array_equal(other.contact_tracker.lines(), m.contact_tracker.lines())
    assert not other.contact_tracker.gate("self")[0] and not other.contact_tracker.gate("world")[0], \
        "a load never inherits an open gate"
    for a, b in zip(m.contact_head.parameters(), other.contact_head.parameters()):
        assert torch.equal(a, b)
    bare = LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                   body_layout="trunk_v1")
    with pytest.raises(ValueError, match="contact head"):
        bare.load_checkpoint(path)
    # a head of another version cannot load its order weights into this one
    rp = tmp_path / "ckpt_step_7" / "model_receipt.json"
    if not rp.is_file():
        rp = next(tmp_path.rglob("model_receipt.json"))
    r = json.loads(rp.read_text())
    r["contact_head"]["version"] = "contact_head_v1"
    rp.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="contact_head_v1"):
        LemLite(device="cpu", seed=1, imagination=False, no_vision=True, next_frame_loss=True,
                body_layout="trunk_v1", contact_head=True).load_checkpoint(path)


def test_an_order_scale_change_migrates_the_head_exactly(tmp_path):
    torch.manual_seed(7)
    m = _lite(contact_head=True)
    B, H = 2, len(m.handle_names)
    lat = torch.randn(B, m.n_mod, m.d_model)
    order, ach = torch.randn(B, H, 9) * 10.0, _pose(B, H)
    order[..., 3:7] = _quat_z(30.0)
    prev = torch.zeros(B, N_PATCHES, 2)
    old = {"pos_cm": CH.ORDER_POS_CM * 1.5, "rot_deg": CH.ORDER_ROT_DEG * 0.5}
    # what the head computed under the old divisors
    with torch.no_grad():
        f_old = order_features(order, ach, m.GAZE_IDX)
        f_old[..., 0:3] *= CH.ORDER_POS_CM / old["pos_cm"]
        f_old[..., 3:6] *= CH.ORDER_ROT_DEG / old["rot_deg"]
        before = m.contact_head(lat, f_old, prev)
    (tmp_path / "model_receipt.json").write_text(json.dumps({"contact_head": {"order_scales": old}}))
    m._migrate_contact_order_scales(str(tmp_path), load_optimizer=False)
    with torch.no_grad():
        after = m.contact_head(lat, order_features(order, ach, m.GAZE_IDX), prev)
    assert torch.allclose(before, after, atol=1e-4, rtol=1e-4)


def test_predict_contacts_gives_three_probabilities_per_patch():
    from lem.model.predictive_state import PredictiveState
    torch.manual_seed(8)
    m = _lite(contact_head=True)
    B, H = 2, len(m.handle_names)
    st = PredictiveState(tokens={mm: torch.randn(B, m.d_model) for mm in m.active},
                         context=torch.zeros(B, (m.ctx_len - 1) * m.n_mod, m.d_model),
                         achieved=_pose(B, H), domain="real")
    order = torch.zeros(B, H, 9)
    order[..., 6] = 1.0
    out = m.predict_contacts(st, order, patch_state=np.zeros((B, N_PATCHES), dtype=np.int64))
    assert out["p_contact"].shape == (B, N_PATCHES) and out["patch_names"] == list(PATCH_NAMES)
    assert bool((out["p_self"] <= out["p_contact"] + 1e-6).all())
    assert m.predict_contacts(st, order) is None, "no skin state lived yet and none given"
    with pytest.raises(ValueError):
        m.predict_contacts(st, order, patch_state=np.zeros((B, 5), dtype=np.int64))


@pytest.mark.slow
def test_the_control_forward_runs_on_two_bodies_and_equals_the_real_path_on_one_state(tmp_path,
                                                                                  monkeypatch):
    """Review of b4, minor 11: no test ran _contact_control_forward. Two worlds built from the
    same seed and sent the same orders are the same state: the shuffled-order control then feeds
    the core and the head exactly what the real path fed them, and its probabilities equal the
    real ones. With babble, each body sending its own order, the two orders differ and so do the
    two predictions."""
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.env import merge_burst_contact_report
    torch.set_num_threads(1)
    got = {}

    def run(seeds, name, same_order=True, noise=0.0):
        m = LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                    contact_head=True, overrides={"training": {"steps_per_phase": 3000}})
        m.contact_control_period = 3
        calls = []
        real = m._contact_control_forward

        def spy(nf_live, nf_mb, exec_t, prev_in, B):
            qc = real(nf_live, nf_mb, exec_t, prev_in, B)
            calls.append(qc)
            return qc
        m._contact_control_forward = spy
        seen = []
        score = m.contact_tracker.score

        def keep(step, q, pb, cb, valid, onsets, habit, wrep=(), q_control=None, pay_on=True):
            seen.append((step, q.copy(), None if q_control is None else q_control.copy()))
            return score(step, q, pb, cb, valid, onsets, habit, wrep, q_control, pay_on)
        m.contact_tracker.score = keep
        srcs = [SingleEnvSource(scene="reach", run_dir=str(tmp_path / name), tag=f"_cc{name}{i}",
                                vision=False, seed=s) for i, s in enumerate(seeds)]
        try:
            frames = [s.reset()[0] for s in srcs]
            dev = dict(m.dev_limits(10 ** 9))
            dev["exploration_noise"] = float(noise)
            prev = held = None
            for i in range(7):
                r = m.step(frames, dev, train=False, step_index=i, executed_action=prev)
                if held is None or i % 5 == 0:
                    held = r.motor_for_action.copy()
                    if same_order:
                        held[1] = held[0]              # both bodies get body 0's order
                new = []
                for bi, s in enumerate(srcs):
                    tasks, f = [], None
                    for rep in range(5):
                        f = s.step(held[bi:bi + 1], np.asarray(r.gaze_out)[0:1], None,
                                   render=False)[0]
                        tasks.append(dict(f.meta.get("task") or {}))
                    f.meta["task"] = {**(f.meta.get("task") or {}), **merge_burst_contact_report(tasks)}
                    new.append(f)
                frames = new
                g = np.asarray(r.gaze_out)[0:1].repeat(2, axis=0)
                prev = LemLite.gaze_to_handle(held, g)
        finally:
            for s in srcs:
                s.close()
        got[name] = [(st, q, qc) for st, q, qc in seen if qc is not None]
        return m, calls
    m, calls = run((5, 5), "same")
    assert calls and all(c is not None for c in calls), "the control forward ran"
    assert len(got["same"]) == len(calls) >= 1, "every control moment was scored beside the control"
    for step, q, qc in got["same"]:
        assert np.abs(q - qc).max() <= 1e-5, f"step {step}: one state, one order -> one prediction"
    run((5, 5), "other", same_order=False, noise=0.05)
    assert got["other"] and any(np.abs(q - qc).max() > 1e-4 for _, q, qc in got["other"])
    s = m.contact_control_window_summary()
    assert s["control_steps"] >= 1


# ------------------------------------------------------------------------------ the trainer
def test_skin_regions_are_window_means_per_body():
    from mimo_bridge.train_mimo import skin_regions_window
    skr = {"0": {"trunk": [4.0, 1.6, 0.0], "left hand": [2.0, 0.4, 2.0]},
           "1": {"trunk": [6.0, 2.0, 0.0]}}
    mean, per = skin_regions_window(skr, {"0": 2, "1": 2}, 3)
    assert per[0]["trunk"] == {"patches": 2.0, "force_share": 0.8, "self_patches": 0.0}
    assert per[1]["trunk"]["patches"] == 3.0 and per[2] == {}, "a body that never reported"
    assert mean["trunk"]["patches"] == 2.5 and mean["left hand"]["self_patches"] == 1.0
    assert skin_regions_window({}, {}, 2) == (None, None)


def test_self_contact_events_count_onsets_pairs_posture_and_view():
    from mimo_bridge.train_mimo import self_contact_events_window
    acc = {"sce_body_steps": 4, "sce_onsets": 4, "sce_structural": 1, "sce_in_view": 2,
           "sce_regions": {"left hand|left hand": 3, "left hand|trunk": 1},
           "sce_bodies": {"left_hand|left_thhub": 1, "left_hand|left_thdistal": 3}}
    s = self_contact_events_window(acc)
    assert s["onsets"] == 4 and s["onsets_per_body_step"] == 1.0
    assert s["pairs_top"][0] == ["left hand", "left hand", 3]
    assert s["pairs_top_bodies"][0] == ["left_hand", "left_thdistal", 3]
    assert s["structural_share"] == 0.25 and s["in_view_share"] == 0.5
    assert self_contact_events_window({}) is None, "no report: absent, never zero"


def test_a_region_pair_is_one_entry_whichever_body_name_comes_first():
    """Review of b4, minor 8: the pair is sorted by body name, so "left hand|trunk" and
    "trunk|left hand" were counted apart. The trainer's key sorts the two regions."""
    from mimo_bridge.train_mimo import self_contact_region_key
    a = self_contact_region_key({"pair": ["chest", "left_hand"], "regions": ["trunk", "left hand"]})
    b = self_contact_region_key({"pair": ["left_ffdistal", "upper_body"], "regions": ["left hand", "trunk"]})
    assert a == b == "left hand|trunk"
    assert self_contact_region_key({}) == "?|?"


def test_the_flags_default_to_the_model_that_ran():
    from mimo_bridge.train_mimo import parse_args
    a = parse_args(["--out", "x"])
    assert a.contact_head == "off" and a.contact_head_weight is None and a.contact_control_wellness == 0.0
