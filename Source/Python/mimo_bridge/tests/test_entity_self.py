# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Levels 2 and 3 over named entities, and the reader that acts on what they prove.

The measured defects these tests pin, all found in the 2026-09-19 adversarial review of the
hierarchical self (docs/Hierarchical_Self_Adversarial_Review_2026-09-19.md):
  B9 the slow averages started at zero, so every early link read positive - here corrected;
  B3 a slot gained a window by the clock even when nothing belonged to it - here a window
     counts for an entity only when the entity was there;
  the 8 bodies were averaged into one number before anything was correlated - here each
     body's presence is correlated with its own wellness and the result pooled.
"""
import math

import pytest
import torch

from mimo_bridge import self_evidence as EV
from mimo_bridge.entity_self import EntitySelfDark
from mimo_bridge.hierarchical_self import HierarchicalSelfDark
from mimo_bridge.self_value_reader import SelfValueReader, derangement, natural_takeaway

NAMES = ["caregiver", "toy", "mobile", "squeaker", "own_hand"]


# ---------------------------------------------------------------- the shared evidence test
def test_the_shared_evidence_test_is_the_one_the_whole_moment_table_uses():
    """Both levels of the self-model must pass the same bar, or a number from one cannot be
    read beside a number from the other. The shared functions are held against the older
    table's own methods at every size it has run at."""
    h = HierarchicalSelfDark(["a"], 8, n_slots=4)
    assert EV.RANK_ALPHA == h.RANK_ALPHA and EV.RANK_MOMENTUM == h.rank_momentum
    for k in (1, 2, 5, 10, 64, 640, 6400):
        assert EV.first_evidence_tick(k) == h.first_evidence_tick(k), k
        for ticks in (1, 3, 9, 25, 100):
            assert EV.rank_line(ticks, k) == pytest.approx(h._rank_line(ticks, k)), (k, ticks)
            assert EV.rank_ceiling(ticks) == pytest.approx(h._rank_ceiling(ticks)), ticks
    assert EV.first_evidence_tick(None) == h.first_evidence_tick(None)
    # the pass mark rises with the number judged together, which is the whole point of it
    assert EV.rank_line(10, 1) < EV.rank_line(10, 10) < EV.rank_line(10, 6400)
    # fewer candidates means evidence arrives sooner: 10 entities against 6,400 whole moments
    assert EV.first_evidence_tick(10) == 11 and EV.first_evidence_tick(6400) == 25
    assert EV.first_evidence_steps(10_000, 10) == 130_000
    # a table so large that the bar sits above any reachable score can never count
    assert EV.first_evidence_tick(10 ** 9) is None
    assert EV.first_evidence_steps(10_000, 10 ** 9) is None


def test_the_start_up_correction_removes_the_bias_that_made_every_early_link_positive():
    """Audit finding B9. A running average that starts at zero sits below the signal for its
    whole first window. The fast and the slow average are short by different amounts, so
    their difference carries a start-up bias with nothing to do with the world: on run H84 at
    moment 138,812 the hours-level slow average of wellness read 2.52 against a wellness of
    3.34, and 6,400 of 6,400 links read positive. The correction divides each average by
    1 minus its momentum to the power of the moments lived."""
    assert EV.bias_correction(0.9, 0) == 1.0
    assert EV.bias_correction(0.9, 1) == pytest.approx(0.1)
    assert EV.bias_correction(0.99, 1000) == pytest.approx(1.0, abs=1e-4)
    m, x = 0.999, 3.34
    ema = 0.0
    for n in range(1, 501):
        ema = m * ema + (1 - m) * x
    assert ema / x < 0.41, "the uncorrected average is far below the signal, as on H84"
    assert ema / EV.bias_correction(m, 500) == pytest.approx(x, rel=1e-9)
    # and in the tracker, on a world where nothing is linked to anything: with the correction
    # off, the start-up bias makes every entity's link read the SAME SIGN, which is what H84
    # showed; with it on, the links fall on both sides of zero
    def early_links(correct):
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(4)
        h = EntitySelfDark(NAMES, n_bodies=4, m_l2=0.99, m_l3=0.999, bias_correct=correct)
        for _ in range(120):
            h.observe(torch.rand(4, len(NAMES), generator=g),
                      3.0 + 0.3 * torch.rand(4, generator=g))
        return h.links("l2")
    off, on = early_links(False), early_links(True)
    assert int((off > 0).sum()) in (0, len(NAMES)), (
        "the fixture must show the start-up bias: %s" % off.tolist())
    assert 0 < int((on > 0).sum()) < len(NAMES), (
        "the corrected links still all share one sign: %s" % on.tolist())


# ---------------------------------------------------------------- what the tracker finds
def _cycle(n_steps, n_bodies, period, wellness_gain, seed=0, phase_per_body=False):
    # period must be shorter than the level window wherever a within-window comparison is
    # judged (the reader's gate): a thing that never comes or goes inside the window it is
    # judged on gives that window nothing to weigh.
    """A world in which one entity (index 0) comes and goes, and the body's wellness rises
    while it is there by `wellness_gain`. Every other entity is noise."""
    g = torch.Generator().manual_seed(seed)
    P, W = [], []
    for t in range(n_steps):
        p = torch.rand(n_bodies, len(NAMES), generator=g) * 0.3
        here = torch.zeros(n_bodies)
        for b in range(n_bodies):
            off = (b * period // n_bodies) if phase_per_body else 0
            here[b] = 1.0 if ((t + off) % period) < period // 2 else 0.0
        p[:, 0] = here
        w = 3.0 + wellness_gain * here + 0.05 * torch.randn(n_bodies, generator=g)
        P.append(p)
        W.append(w)
    return P, W


def test_a_real_link_is_found_and_the_others_are_not():
    """The caregiver comes and goes; wellness is higher while it is there. Its link must be
    clearly positive, every other entity's near zero, and the evidence must pass."""
    torch.manual_seed(0)
    h = EntitySelfDark(NAMES, n_bodies=4, m_l2=0.995, m_l3=0.9995)
    P, W = _cycle(6000, 4, period=400, wellness_gain=0.5)
    for p, w in zip(P, W):
        h.observe(p, w)
    link = h.links("l2")
    assert float(link[0]) > 0.3, "the entity that goes with wellness shows no link: %.3f" % link[0]
    assert float(link[1:].abs().max()) < float(link[0]) / 2, (
        "an unrelated entity's link is as large as the real one: %s" % link.tolist())
    st = h.stats()
    assert st["l2_counted"] >= 1 and NAMES[0] in st["l2_counted_names"], st
    assert st["teacher"] == "wellness" and st["windows_needed"] == EV.first_evidence_tick(len(NAMES))


def test_a_world_with_no_link_counts_nothing():
    """The same shape of world with wellness unrelated to any entity. Nothing may pass - and
    the count must have been REACHABLE, or a zero says nothing.

    l2_counted reads -1 for "not measured yet": before an entity has lived the windows a
    perfect link would need, its count of 0 is forced by arithmetic. Accepting -1 here let
    the test pass on a run that measured nothing at all. Measured 2026-09-21 on this fixture:
    the caregiver lived 29 windows of its own against the 10 a perfect link needs at 5
    entities, so the count is measured, and it reads 0 of 5."""
    torch.manual_seed(1)
    h = EntitySelfDark(NAMES, n_bodies=4, m_l2=0.995, m_l3=0.9995)
    P, W = _cycle(6000, 4, period=400, wellness_gain=0.0, seed=3)
    for p, w in zip(P, W):
        h.observe(p, w)
    st = h.stats()
    # reachability first: an entity has lived enough windows of its own for any score to pass
    assert st["windows_needed"] == EV.first_evidence_tick(len(NAMES)), st
    assert max(st["l2_entity_windows"]) >= st["windows_needed"], (
        "no entity lived the %s windows a perfect link needs, so a count of 0 is forced by "
        "arithmetic and measures nothing: %s" % (st["windows_needed"], st["l2_entity_windows"]))
    assert st["l2_steps_to_first_evidence"] == 0 and st["l2_entities_old_enough"] >= 1, st
    assert st["l2_counted"] == 0, ("a world with no link counted %s: %s"
                                   % (st["l2_counted"], st["l2_counted_names"]))


def test_each_bodys_presence_is_judged_against_its_own_wellness():
    """The hierarchical self averaged all bodies into one presence and one wellness before
    correlating, so a link that every body has at a different time cancelled. Here the four
    bodies meet the entity a quarter-cycle apart: the batch average of presence is then flat,
    and only per-body correlation can find the link."""
    torch.manual_seed(0)
    P, W = _cycle(6000, 4, period=400, wellness_gain=0.5, seed=5, phase_per_body=True)
    batch_presence = torch.stack([p[:, 0].mean() for p in P])
    assert float(batch_presence.std()) < 0.3, "the fixture must hide the link from a batch mean"
    h = EntitySelfDark(NAMES, n_bodies=4, m_l2=0.995, m_l3=0.9995)
    for p, w in zip(P, W):
        h.observe(p, w)
    assert float(h.links("l2")[0]) > 0.3, "the per-body link was lost"


def test_a_window_counts_for_an_entity_only_when_its_presence_moved():
    """Audit finding B3. The hierarchical self gave every slot a window every window, even
    when no moment belonged to it, so a count could describe something seen for seconds. An
    entity whose presence never moves - never there, or never away - must gather no windows
    of its own, and so can never pass, however long the run."""
    torch.manual_seed(0)
    h = EntitySelfDark(NAMES, n_bodies=2, m_l2=0.99, m_l3=0.999)
    for t in range(2000):
        p = torch.zeros(2, len(NAMES))
        p[:, 0] = 1.0 if (t % 200) < 100 else 0.0      # comes and goes
        p[:, 1] = 1.0                                   # always here, so it never moves
        h.observe(p, torch.full((2,), 3.0) + 0.4 * p[:, 0])
    st = h.stats()
    assert st["l2_ticks"] >= 10, st["l2_ticks"]
    assert st["l2_entity_windows"][0] >= 10, "the entity that came and went gathered no windows"
    assert st["l2_entity_windows"][1] <= 1, (
        "an entity that never left gathered windows anyway: %s" % st["l2_entity_windows"])
    assert st["l2_entity_windows"][2:] == [0] * (len(NAMES) - 2), (
        "an entity that was never present gathered windows anyway: %s" % st["l2_entity_windows"])
    assert not bool(h.counted("l2")[1:].any()), "an entity with no movement passed the test"


def test_the_tracker_has_no_trained_weights_and_survives_a_resume():
    h = EntitySelfDark(NAMES, n_bodies=2)
    assert sum(p.numel() for p in h.parameters()) == 0
    torch.manual_seed(0)
    for _ in range(50):
        h.observe(torch.rand(2, len(NAMES)), torch.rand(2) + 3.0)
    again = EntitySelfDark(NAMES, n_bodies=2)
    again.load_state_dict({k: v.clone() for k, v in h.state_dict().items()}, strict=True)
    for k, v in h.state_dict().items():
        assert torch.equal(v, again.state_dict()[k]), k
    assert again._frames_cpu == h._frames_cpu == 50
    # a different entity list cannot be carried across: those rows would mean other things
    other = EntitySelfDark(NAMES + ["extra"], n_bodies=2)
    other.load_state_dict({k: v.clone() for k, v in h.state_dict().items()}, strict=False)
    assert int(other.frames.item()) == 50 and float(other.l2_cov.abs().max()) == 0.0


def test_a_resumes_episode_companion_cannot_clear_the_watching_the_checkpoint_carried(tmp_path):
    """Measured 2026-09-20 on the segment-10 resume of H84. The checkpoint's runtime state
    held 34,950 moments of watching; load_checkpoint put them on the model as
    _entity_self_saved, for the tracker to pick up at the first step, when the body count is
    known. The episode companion, restored AFTER the checkpoint, carried its own spent value
    of None for that same attribute and set it back - so the tracker was built empty, the
    run's 2 lived windows read 0, and no warning said so. The companion carries the physical
    episode, never the checkpoint's hand-off."""
    from mimo_bridge.lem_model import LemLite
    from mimo_bridge import continuation_state
    torch.set_num_threads(1)
    options = dict(device='cpu', imagination=False, no_vision=True, next_frame_loss=True)
    old = LemLite(**options, checkpoint_dir=str(tmp_path / 'old'))
    new = LemLite(**options, checkpoint_dir=str(tmp_path / 'new'))
    old._entity_self_saved = None                      # spent when the companion was written
    old._entity_self_saved_names = None
    carried = {'frames': torch.tensor([34950])}        # what this resume's checkpoint holds
    new._entity_self_saved = carried
    new._entity_self_saved_names = list(NAMES)
    state = continuation_state.capture_learner(old, include_replay=False)
    continuation_state.restore_learner(new, state)
    assert new._entity_self_saved is carried, 'the companion cleared the checkpoint hand-off'
    assert new._entity_self_saved_names == list(NAMES)
    assert not any(k in state['data'] for k in ('_entity_self_saved', '_entity_self_saved_names'))


def test_the_teacher_can_be_an_ability_signal_instead_of_wellness(record_property):
    """The project's goal is abilities, not comfort. The tracker learns from whatever scalar
    the caller hands it, and says in its own stats which one that was."""
    torch.manual_seed(0)
    h = EntitySelfDark(NAMES, n_bodies=2, m_l2=0.995, m_l3=0.9995, teacher="ability")
    for t in range(4000):
        p = torch.zeros(2, len(NAMES))
        p[:, 2] = 1.0 if (t % 400) < 200 else 0.0          # the mobile is within reach
        strikes = 0.4 * p[:, 2] + 0.02 * torch.randn(2)     # an ability signal: hand strikes
        h.observe(p, strikes)
    st = h.stats()
    record_property("ability_link_mobile", st["l2_link"][2])
    assert st["teacher"] == "ability"
    assert st["l2_link"][2] > 0.3, st["l2_link"]


def test_the_control_signals_are_measured_not_assumed(record_property):
    """A control is a fake teacher signal that cannot carry a true link. Two ways to make
    one: flip the sign of the whole value, or flip only its movement. Flipping the movement
    was measured worse on 2026-09-10 (counts by luck 2.34 in 100 against 7.55), so the level
    flip ships. This test records what each does HERE rather than repeating that claim: eight
    no-link worlds per mode, 5 entities in each, so 40 judgements per mode.

    WHAT IS ASSERTED, AND WHY IT IS AN ABSOLUTE COUNT. The old line was
    "level <= deviation + 4", which both modes pass while counting all 40 of 40 by luck: it
    compares the two modes and says nothing about either. The line is now a rate: at most 5
    counts in 100 judgements, so at most 2 of the 40. Reachability arithmetic for that line,
    at declaration time: the module sets its own pass mark so a whole 5-entity table with no
    link would be miscounted 0.135 times in 100 tables, which over 8 tables predicts 0.011
    counts - the 2 allowed sit 185 times above what the arithmetic predicts and 20 times
    below the 40 the old line allowed. Measured here on 2026-09-21, after the control was
    centred on its own slow average: 0 of 40 for both modes."""
    allowed = int(0.05 * 8 * len(NAMES))            # 5 counts in 100 judgements, so 2 of 40
    rates, reachable = {}, {}
    for mode in ("level", "deviation"):
        counted, old_enough = 0, 0
        for seed in range(8):
            h = EntitySelfDark(NAMES, n_bodies=2, m_l2=0.99, m_l3=0.999, null_mode=mode)
            P, W = _cycle(2600, 2, period=200, wellness_gain=0.0, seed=100 + seed)
            for p, w in zip(P, W):
                h.observe(p, w)
            c = h.counted("l2")
            counted += int(c.sum().item())
            old_enough += int((h.stats() or {})["l2_entities_old_enough"])
        rates[mode] = counted
        reachable[mode] = old_enough
        record_property("counted_by_luck_" + mode, counted)
        record_property("entities_old_enough_" + mode, old_enough)
    # a count of 0 means nothing unless something could have been counted
    for mode in ("level", "deviation"):
        assert reachable[mode] >= 8, (
            "%s: only %d of the 40 entities lived the windows a perfect link needs, so these "
            "zeros are forced by arithmetic" % (mode, reachable[mode]))
        assert rates[mode] <= allowed, (
            "%s counted %d of 40 no-link judgements by luck, over the %d allowed (5 in 100): %s"
            % (mode, rates[mode], allowed, rates))


# ---------------------------------------------------------------- the reader
def test_the_shuffle_that_makes_the_control_keeps_nothing_in_place():
    for n in range(2, 12):
        perm = derangement(n, seed=n)
        assert sorted(perm) == list(range(n))
        assert all(perm[i] != i for i in range(n)), (n, perm)
    assert derangement(1) == [0] and derangement(0) == []


def test_the_reader_adds_nothing_until_its_control_is_beaten():
    """The gate. With no entity counted the reader adds exactly zero, whatever its weight;
    with a real link it opens only after the real reader has tracked wellness better than the
    shuffled one for the run of windows asked for."""
    torch.manual_seed(0)
    h = EntitySelfDark(NAMES, n_bodies=4, m_l2=0.995, m_l3=0.9995)
    r = SelfValueReader(NAMES, weight=0.5, gate_windows=2)
    P, W = _cycle(6000, 4, period=100, wellness_gain=0.5, seed=7)
    assert float(r.value(P[0]).abs().max()) == 0.0 and r.applied_weight == 0.0
    window = h.window("l2")
    opened_at = None
    for t, (p, w) in enumerate(zip(P, W)):
        h.observe(p, w)
        r.compare(p, w)
        if (t + 1) % window == 0:
            rep = r.window_tick(h)
            if rep["gate_open"] and opened_at is None:
                opened_at = t + 1
    assert opened_at is not None, ("the gate never opened on a world with a real link: %s"
                                   % r.last_report)
    assert r.applied_weight == 0.5
    rep = r.last_report          # the report from the last window, before its sums were reset
    assert NAMES[0] in rep["counted_entities"]
    assert rep["real_tracks_wellness"] > rep["control_tracks_wellness"]
    # the value it adds is larger when the entity with the link is present
    on = torch.zeros(1, len(NAMES)); on[0, 0] = 1.0
    off = torch.zeros(1, len(NAMES))
    assert float(r.value(on)) > float(r.value(off))


# The gate can stay shut for four different reasons, and each one is a different finding
# about the world. Until 2026-09-21 one test accepted any of three of them through an OR over
# substrings, so it could not say which had fired - and the reason it was written for was not
# the one the fixture produced. There is one test per reason below. Each names the reason by
# the report KEY that selects it, which does not move when the wording is rewritten, and then
# by a short fragment of the sentence the operator reads.
def _hand_set_reader(**kw):
    """A reader holding one proven link, as a window boundary would leave it: the caregiver's
    link at 0.8 on a scale where 0 = its presence says nothing about wellness and 1 = wellness
    rises and falls with it exactly. Set here rather than trained so each fixture below can
    put the gate in exactly one of its four states."""
    r = SelfValueReader(NAMES, weight=0.5, gate_windows=2, **kw)
    r.link.zero_(); r.link[0] = 0.8
    r.counted.zero_(); r.counted[0] = 1.0
    return r


def _feed(reader, n_moments, presence_of, wellness_gain=0.5, noise=0.02, seed=21, n_bodies=4):
    """n_moments of a world where the caregiver comes and goes every 40 moments and wellness
    is `wellness_gain` higher while it is there. presence_of(p, here) may change any other
    entity's column before the moment is handed over."""
    g = torch.Generator().manual_seed(seed)
    for t in range(n_moments):
        p = torch.rand(n_bodies, len(NAMES), generator=g) * 0.3
        here = torch.full((n_bodies,), 1.0 if (t % 40) < 20 else 0.0)
        p[:, 0] = here
        presence_of(p, here)
        reader.compare(p, 3.0 + wellness_gain * here
                       + noise * torch.randn(n_bodies, generator=g))


def test_the_gate_stays_shut_because_no_entity_has_passed_the_evidence_test():
    """Reason one of four, on a world where wellness is unrelated to every entity: the
    tracker counts nothing, so the reader has nothing to read at all.

    Re-measured 2026-09-21, after the control was centred on its own slow average: this
    fixture's tracker counts 0 of 5 entities and the reader compares 800 body-moments
    without a single link to put into its value. The old assertion here accepted this
    answer, "a shuffle tracks wellness as well as the real reader" and "the real reader is
    under its own line" interchangeably, and on the code of the morning it was the last of
    those three that fired, not the one the docstring claimed. Which one fires is the whole
    finding, so it is asserted by name."""
    torch.manual_seed(2)
    h = EntitySelfDark(NAMES, n_bodies=4, m_l2=0.995, m_l3=0.9995)
    r = SelfValueReader(NAMES, weight=0.5, gate_windows=2)
    P, W = _cycle(6000, 4, period=100, wellness_gain=0.0, seed=11)
    window = h.window("l2")
    for t, (p, w) in enumerate(zip(P, W)):
        h.observe(p, w)
        r.compare(p, w)
        if (t + 1) % window == 0:
            r.window_tick(h)
    rep = r.last_report
    assert not rep["gate_open"] and r.applied_weight == 0.0
    assert float(r.value(torch.ones(2, len(NAMES))).abs().max()) == 0.0
    # the reason, by the key that selects it and then by its own words
    assert rep["n_counted"] == 0 and rep["counted_entities"] == [], rep
    assert rep["moments_compared"] >= 30, ("the window must be long enough to judge, or this "
                                           "is the waiting answer and not this one: %s" % rep)
    assert "nothing to read" in rep["why"], rep["why"]


def test_the_gate_stays_shut_because_the_real_reader_never_reached_its_own_line():
    """Reason two of four: an entity IS counted and the shuffles do have something to read,
    but the reader's own value tracks wellness at 0.036 on a scale where 0 = the value said
    nothing about wellness and 1 = it rose and fell with it exactly - under the 0.100 it must
    reach before any shuffle is consulted. The world: wellness rises by 0.02 while the
    caregiver is there against its own moment-to-moment spread of 0.5, so the link is real
    and far too small to read."""
    r = _hand_set_reader()
    _feed(r, 200, lambda p, here: None, wellness_gain=0.02, noise=0.5, seed=31)
    rep = r.window_tick()               # no tracker: the links under test are not overwritten
    assert rep["n_counted"] == 1, rep
    assert rep["shuffles_that_moved"] >= r.n_controls // 2, (
        "enough shuffles must have moved, or this is reason four and not this one: %s" % rep)
    assert rep["real_tracks_wellness"] < r.min_correlation, rep
    assert not rep["gate_open"] and r.applied_weight == 0.0
    assert "must reach on its own" in rep["why"], rep["why"]


def test_the_gate_stays_shut_when_a_shuffle_tracks_wellness_as_well_as_the_real_reader():
    """Reason three of four, which no test in the repository reached before 2026-09-21: the
    real reader DOES clear its own 0.100 line, and one of the eight shuffles tracks wellness
    just as well, so which entity a link belongs to carries nothing and no value is added.

    How the fixture makes that happen with no luck involved: a shuffle holds the same link
    values attached to the wrong entities, so the whole of the caregiver's link lands on one
    other entity - the one that shuffle sends the caregiver's place to. Give that entity
    exactly the caregiver's presence and the two readers compute the same number on every
    moment, so their two correlations with wellness are equal to the fourth decimal the
    report rounds to and the 0.050 the real reader must lead by is unreachable."""
    r = _hand_set_reader()
    mirror = r.perms[0].index(0)        # the entity shuffle 0 hands the caregiver's link to
    assert mirror != 0
    _feed(r, 200, lambda p, here: p[:, mirror].copy_(here), seed=21)
    rep = r.window_tick()
    assert rep["n_counted"] == 1 and rep["shuffles_that_moved"] == r.n_controls, rep
    assert rep["real_tracks_wellness"] >= r.min_correlation, rep
    assert rep["control_tracks_wellness"] == rep["real_tracks_wellness"], rep
    assert not rep["gate_open"] and r.applied_weight == 0.0
    assert float(r.value(torch.ones(2, len(NAMES))).abs().max()) == 0.0
    assert "carries nothing yet" in rep["why"], rep["why"]


def test_the_gate_stays_shut_when_too_few_shuffles_had_a_value_that_moved():
    """Reason four of four: a shuffle whose value never moved was never a test, so it cannot
    be beaten, and at least half the eight must have moved before the comparison counts.

    The world that produces it is an ordinary one: each shuffle hands the caregiver's link to
    one other entity, and a toy that is never brought out has presence exactly 0 at every
    moment, so every shuffle pointed at it computes the same 0 all window. Here the entities
    the shuffles point at are held at 0 until fewer than half of them can move, while the
    real reader's value keeps tracking wellness at 0.997 of a possible 1.000 - so the gate
    shuts on the control's emptiness and not on the reader's score."""
    from collections import Counter
    r = _hand_set_reader()
    mirrors = [p.index(0) for p in r.perms]
    assert 0 not in mirrors, "no shuffle may hand the caregiver's link back to the caregiver"
    silent = []
    for entity, _count in Counter(mirrors).most_common():
        silent.append(entity)
        if sum(1 for m in mirrors if m not in silent) < r.n_controls // 2:
            break
    still_moving = sum(1 for m in mirrors if m not in silent)
    assert still_moving < r.n_controls // 2, (mirrors, silent)
    quiet = torch.tensor(silent)
    _feed(r, 200, lambda p, here: p.index_fill_(1, quiet, 0.0), seed=41)
    rep = r.window_tick()
    assert rep["n_counted"] == 1, rep
    assert rep["shuffles_that_moved"] == still_moving, rep
    assert rep["real_tracks_wellness"] >= r.min_correlation, (
        "the real reader must be above its own line, or this is reason two: %s" % rep)
    assert not rep["gate_open"] and r.applied_weight == 0.0
    assert "shuffles had a value that moved" in rep["why"], rep["why"]


def test_the_reader_survives_a_resume_and_keeps_its_gate_shut_until_re_earned():
    """What a resume must do with a gate that was OPEN when the checkpoint was written.

    Before 2026-09-21 this test opened no gate, so "the resumed reader's gate is shut" was
    already true before anything was copied. Measured with the gate opened first: a plain
    load_state_dict carried the open gate through with the running comparison empty, and
    applied_weight read 0.25 with 0 body-moments compared. The reader now restarts its own
    gate on any load, keeps the links the tracker spent windows earning, and re-opens only
    by beating its shuffled control again."""
    h = EntitySelfDark(NAMES, n_bodies=2)
    r = SelfValueReader(NAMES, weight=0.25, seed=3, gate_windows=1)
    r.refresh(h)
    r.link.zero_(); r.link[0] = 0.8
    r.counted.zero_(); r.counted[0] = 1.0
    r.gate_open.fill_(True); r.gate_streak.fill_(1)
    on = torch.zeros(1, len(NAMES)); on[0, 0] = 1.0
    assert r.applied_weight == 0.25 and float(r.value(on)) == pytest.approx(0.25 * 0.8)

    again = SelfValueReader(NAMES, weight=0.25, seed=3, gate_windows=1)
    again.load_state_dict({k: v.clone() for k, v in r.state_dict().items()}, strict=True)
    assert again.perm == r.perm
    # the earned links are carried: they cost the tracker its windows and a resume that threw
    # them away would make the evidence unreachable across a run boundary
    assert torch.equal(again.link, r.link) and torch.equal(again.counted, r.counted)
    # the gate is not carried, and neither is the run of windows behind it
    assert not bool(again.gate_open.item()) and int(again.gate_streak.item()) == 0
    assert again.applied_weight == 0.0 and float(again.value(on)) == 0.0
    assert int(again.cmp_n.item()) == 0 and float(again.cmp_sums.abs().sum()) == 0.0
    assert "waiting for" in again.report()["why"], again.report()["why"]

    # and it re-earns the gate from the moments after the resume, on a world where the
    # caregiver's comings and goings really do go with wellness
    g = torch.Generator().manual_seed(5)
    for t in range(200):
        p = torch.rand(1, len(NAMES), generator=g) * 0.3
        here = torch.full((1,), 1.0 if (t % 40) < 20 else 0.0)
        p[:, 0] = here
        again.compare(p, 3.0 + 0.5 * here + 0.02 * torch.randn(1, generator=g))
    rep = again.window_tick()
    assert rep["gate_open"] and again.applied_weight == 0.25, rep


def test_the_take_away_test_reads_the_worlds_own_comings_and_goings():
    """The crib takes things away by itself: the adult leaves every 60 s of body time. A
    proven link must show the same sign in that natural comparison."""
    torch.manual_seed(0)
    P, W = _cycle(2000, 2, period=200, wellness_gain=0.5, seed=13)
    out = natural_takeaway(torch.stack(P), torch.stack(W), NAMES)
    assert out[NAMES[0]]["difference"] > 0.3, out[NAMES[0]]
    assert out[NAMES[0]]["moments_present"] > 100 and out[NAMES[0]]["moments_absent"] > 100
    for nm in NAMES[1:]:
        d = out[nm]["difference"]
        assert math.isnan(d) or abs(d) < 0.2, (nm, out[nm])


def test_a_value_that_is_constant_but_not_zero_did_not_move():
    """The guard that refuses a reader whose value never moved compared its variance against a
    flat tenth of a million-millionth. The running sums are kept in single precision, and a
    variance taken as the mean of the squares minus the square of the mean loses almost every
    digit when the value is constant: measured 2026-09-21, a value held at exactly 0.16 over
    800 moments left a residue of 8.1e-09, which is 8,100 times the line it was tested
    against. So all eight shuffles read as having moved and the strongest was handed a
    correlation of 0.0118 built from nothing but rounding.

    The line is now relative to the value's own size. Held constant at a value far from zero,
    neither the real reader nor any shuffle may read as a test."""
    import torch
    from mimo_bridge.self_value_reader import SelfValueReader
    names = ["a", "b", "c", "d", "e"]
    r = SelfValueReader(names, weight=0.25, gate_windows=3)
    r.link.copy_(torch.tensor([0.4, 0.0, 0.0, 0.0, 0.0]))
    r.counted.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
    g = torch.Generator().manual_seed(0)
    for _ in range(800):
        present = torch.zeros(4, len(names))
        present[:, 0] = 0.4                    # there, unchangingly, all window long
        r.compare(present, 3.0 + 0.02 * torch.randn(4, generator=g))
    real, strongest, n_moved = r._tracking()
    assert real is None, "a value that never moved was given a correlation of %s" % (real,)
    assert strongest is None and n_moved == 0, (strongest, n_moved)
    rep = r.window_tick()
    assert rep["shuffles_that_moved"] == 0 and not rep["gate_open"]
    assert "never moved" in rep["why"], rep["why"]


def test_a_value_that_wiggles_by_one_part_in_a_hundred_of_its_level_did_move():
    """The line must not be so high that a real but small movement is thrown away. One part in
    a hundred of the value's own level clears it."""
    import torch
    from mimo_bridge.self_value_reader import SelfValueReader
    names = ["a", "b", "c", "d", "e"]
    r = SelfValueReader(names, weight=0.25, gate_windows=3)
    r.link.copy_(torch.tensor([0.4, 0.0, 0.0, 0.0, 0.0]))
    r.counted.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
    g = torch.Generator().manual_seed(1)
    for t in range(800):
        present = torch.zeros(4, len(names))
        # the reader's value is 0.4 times this, so it sits at 0.16 and swings by 0.0016,
        # which is one part in a hundred of its own level
        present[:, 0] = 0.4 + 0.004 * (1.0 if (t // 40) % 2 else -1.0)
        r.compare(present, 3.0 + 0.02 * torch.randn(4, generator=g))
    real, strongest, n_moved = r._tracking()
    assert real is not None, "a movement of one part in a hundred of the level was thrown away"
    assert -1.0 <= real <= 1.0
    # and the shuffles here are all held at exactly zero, so none of them is a test: that is
    # the guard working in the other direction on the same window
    assert n_moved == 0 and strongest is None
