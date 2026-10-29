# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Controllability's shadow gate (2026-10-03, lem/model/wellness_channels.py SHADOW_CONTROL_KIND).

The evidence gate in force reads the ABSOLUTE control (the body head's expectation from another body's command as
it stands). The shadow gate is the same gate fed by the SIZE-MATCHED control (the expectation from another body's
asked moves and turns applied to this body's pose), recorded only: the operator's decision of Oct 3 is that both
gates are read on the same commands for one segment before the gate is switched.

CPU only, hand-made graded commands and the small fixture learner; no MuJoCo world, no checkpoint. Checked here:
the shadow reading's arithmetic; fed the same control, the shadow gate equals the gate in force after every graded
command; the per-window counts; the state's save and load (an older state without the shadow keys included); the
controllability weight and the ranking under every pairing of the two gates' states; a fixture run with the shadow
gate held open against one with it held shut; the sentences, rows, series and chart the pages build from the new
keys. Each guard of the tracker also runs against its planted defect in
docs/gain_of_function_v4/swap_test_phases/planted_defects.py."""
import math

import pytest
import torch

from lem.model import wellness_channels as WC

H = 4            # handles 0-2 are body handles, 3 = the eyes' row
N = WC.CONTROL_GATE_READINGS


def _tracker():
    return WC.ControllabilityTracker(H, gaze_idx=3, hold=2, floors={"command": 0.5, "efference": 0.5},
                                     ema_orders=1, expect_efference=True)


def _pose(B):
    a = torch.zeros(B, H, 10)
    a[..., 6] = 1.0                       # identity quaternion (x y z w)
    return a


def _qz(deg):
    h = math.radians(deg) / 2.0
    return torch.tensor([0.0, 0.0, math.sin(h), math.cos(h)])


def _command(B, deg=40.0):
    """Every body is sent a turn of `deg` degrees about z on handle 0 (3.6 measured hold spreads at 40)."""
    o = torch.zeros(B, H, 9)
    o[..., 6] = 1.0
    o[..., 8] = 0.5
    o[:, 0, 3:7] = _qz(deg)
    return o


def _turns(values):
    """[B, H, 3] degrees: handle 0 turned about z by each body's value."""
    t = torch.zeros(len(values), H, 3)
    t[:, 0, 2] = torch.tensor([float(v) for v in values])
    return t


def _grade(tr, k, own, absolute=None, sized=None, achieved=40.0, commanded=None):
    """One graded command number k. Every body achieves a turn of `achieved` degrees about z on handle 0; the
    body head's expectation per body is `own` degrees, the expectation from another body's absolute command
    `absolute` and from its size-matched command `sized` (None = that key is absent from the record). The
    agreement between an expectation of e degrees and the achieved a degrees about the same axis is
    sign(e a) x min(|e|, |a|) / max(|e|, |a|)."""
    B = len(own)
    begin, end = _pose(B), _pose(B)
    end[:, 0, 3:7] = _qz(achieved)
    cmd = _command(B) if commanded is None else commanded
    s0 = 100 + 10 * k
    eff = {"step": s0 + 3, "order_began": s0, "expected_cm": torch.zeros(B, H, 3), "expected_deg": _turns(own)}
    if absolute is not None:
        eff["expected_shuffled_cm"] = torch.zeros(B, H, 3)
        eff["expected_shuffled_deg"] = _turns(absolute)
    if sized is not None:
        eff["expected_size_matched_cm"] = torch.zeros(B, H, 3)
        eff["expected_size_matched_deg"] = _turns(sized)
    tr.step(s0, _command(B, 0.0), 10 * k, begin, "reflex", None)
    tr.step(s0 + 1, cmd, 10 * k + 1, begin, "planner", None)       # the command appears
    tr.step(s0 + 2, cmd, 10 * k + 1, end, "planner", None)
    return tr.step(s0 + 3, cmd, 10 * k + 2, end, "planner", eff)   # held 2 model steps: graded


# ------------------------------------------------------------------------------ the reading's arithmetic
def shadow_reading_violations(tr_factory):
    """What is wrong with a tracker's shadow reading, on one graded command of two bodies that both achieve a
    40-degree turn: own expectations 40 and 20 degrees (agreements 1 and 0.5), absolute-control expectations -40
    and 10 (agreements -1 and 0.25), size-matched expectations 20 and -20 (agreements 0.5 and -0.5). The gate in
    force must read mean(1 - -1, 0.5 - 0.25) = 1.125 and the shadow gate mean(1 - 0.5, 0.5 - -0.5) = 0.75."""
    bad = []
    tr = tr_factory()
    rec = _grade(tr, 0, own=[40.0, 20.0], absolute=[-40.0, 10.0], sized=[20.0, -20.0])
    if rec is None or rec.get("source") != "efference":
        return ["the command was not graded against the body head's expectation"]
    if len(tr.gate_readings) != 1 or abs(tr.gate_readings[-1] - 1.125) > 1e-5:
        bad.append("the gate in force's reading is not agreement minus the absolute control: %s"
                   % list(tr.gate_readings))
    if len(tr.shadow_gate_readings) != 1 or abs(tr.shadow_gate_readings[-1] - 0.75) > 1e-5:
        bad.append("the shadow reading is not agreement minus the size-matched control: %s"
                   % list(tr.shadow_gate_readings))
    s = tr.summary(2)
    want = {"agreement_mean": 0.75, "control_mean": -0.375, "difference_mean": 1.125,
            "size_matched_control_mean": 0.0, "size_matched_difference_mean": 0.75}
    for k, v in want.items():
        if s.get(k) is None or abs(s[k] - v) > 1e-4:
            bad.append("%s reads %s, the arithmetic gives %s" % (k, s.get(k), v))
    return bad


def test_the_shadow_reading_is_agreement_minus_the_size_matched_control():
    assert shadow_reading_violations(_tracker) == []


def test_the_shadow_reading_uses_the_measured_bodies_of_the_gate_in_force():
    """A body whose expectation is under the stillness floor is measured by neither gate: both readings are the
    mean over the same measured bodies."""
    tr = _tracker()
    # body 1 expects a turn of 2 degrees (0.18 measured hold spreads, under the floor of 0.5): not measured
    _grade(tr, 0, own=[40.0, 2.0], absolute=[10.0, 40.0], sized=[20.0, 40.0])
    assert tr.gate_readings[-1] == pytest.approx(1.0 - 0.25, abs=1e-5)
    assert tr.shadow_gate_readings[-1] == pytest.approx(1.0 - 0.5, abs=1e-5)
    s = tr.summary(2)
    assert s["measured_share"] == 0.5
    assert s["size_matched_control_mean"] == pytest.approx(0.5, abs=1e-4)
    # no body measured: neither gate takes a reading, and neither count moves
    tr = _tracker()
    _grade(tr, 0, own=[2.0, 2.0], absolute=[10.0, 40.0], sized=[20.0, 40.0])
    s = tr.summary(2)
    assert len(tr.gate_readings) == 0 == len(tr.shadow_gate_readings)
    assert s["gate_commands"] == 0 == s["shadow_gate_commands"] and s["size_matched_control_mean"] is None


def test_without_the_size_matched_expectation_only_the_gate_in_force_takes_a_reading():
    """The first graded command after a resume from a state saved before the size-matched control existed carries
    no size-matched expectation: the gate in force reads as before and the shadow gate takes no reading."""
    tr = _tracker()
    _grade(tr, 0, own=[40.0, 20.0], absolute=[-40.0, 10.0], sized=None)
    assert len(tr.gate_readings) == 1 and len(tr.shadow_gate_readings) == 0
    s = tr.summary(2)
    assert s["difference_mean"] == pytest.approx(1.125, abs=1e-4)
    assert s["size_matched_control_mean"] is None and s["size_matched_difference_mean"] is None
    assert (s["gate_commands"], s["shadow_gate_commands"], s["gates_same_commands"]) == (1, 0, 0)
    # the reverse: a record with the size-matched expectation alone feeds the shadow gate alone
    tr = _tracker()
    _grade(tr, 0, own=[40.0, 20.0], absolute=None, sized=[20.0, -20.0])
    assert len(tr.gate_readings) == 0 and list(tr.shadow_gate_readings) == pytest.approx([0.75], abs=1e-5)
    # a size-matched expectation of another body count is not read
    tr = _tracker()
    B = 2
    begin, end = _pose(B), _pose(B)
    end[:, 0, 3:7] = _qz(40.0)
    eff = {"step": 103, "order_began": 100, "expected_cm": torch.zeros(B, H, 3), "expected_deg": _turns([40.0, 20.0]),
           "expected_size_matched_cm": torch.zeros(3, H, 3), "expected_size_matched_deg": _turns([20.0, 20.0, 20.0])}
    tr.step(100, _command(B, 0.0), 0, begin, "reflex", None)
    tr.step(101, _command(B), 1, begin, "planner", None)
    tr.step(102, _command(B), 1, end, "planner", None)
    assert tr.step(103, _command(B), 2, end, "planner", eff) is not None
    assert len(tr.shadow_gate_readings) == 0


def test_the_stats_block_names_and_defines_every_new_key():
    tr = _tracker()
    _grade(tr, 0, own=[40.0, 20.0], absolute=[-40.0, 10.0], sized=[20.0, -20.0])
    s = tr.summary(2)
    new = ["size_matched_control_mean", "size_matched_difference_mean", "shadow_control_kind", "shadow_gate_open",
           "shadow_gate_mean", "shadow_gate_standard_error", "shadow_gate_readings_above_zero",
           "shadow_gate_readings_held", "gate_standard_error", "gate_readings_above_zero", "gate_readings_held",
           "gate_commands", "gate_open_commands", "shadow_gate_commands", "shadow_gate_open_commands",
           "gates_same_commands", "gates_both_open_commands", "gate_only_open_commands",
           "shadow_gate_only_open_commands", "gate_definitions"]
    for k in new:
        assert k in s, k
    # the keys the block carried before keep their names and values
    for k in ("value_mean", "per_body", "measured_share", "agreement_mean", "control_mean", "difference_mean",
              "control_kind", "command_agreement_mean", "by_author", "by_author_orders", "orders", "source",
              "tenure_mismatch", "efference_missing", "floors", "gate_open", "gate_mean", "gate_line",
              "gate_lines_noise", "gate_why"):
        assert k in s, k
    assert s["control_kind"] == "shuffled order" and s["shadow_control_kind"] == WC.SHADOW_CONTROL_KIND
    assert s["shadow_gate_readings_held"] == 1 == s["gate_readings_held"]
    assert s["shadow_gate_mean"] is None and s["shadow_gate_open"] is False     # fewer than 32 readings
    # every new key is named in the definitions the block carries; the texts use the canonical names
    text = " ".join(s["gate_definitions"].values())
    for k in new:
        if k not in ("gate_definitions", "shadow_control_kind"):
            assert k in text, "gate_definitions does not define %s" % k
    low = text.lower()
    for word in (" order", "moment", " tick", "reward"):
        assert word not in low, "a definition uses %r" % word
    assert "command" in low and "model step" in low and "recorded only" in low
    import json
    assert len(json.dumps(s)) < 6000


# ------------------------------------------------------------------------------ the same rule as the gate in force
def _phases():
    """(own, control) expectations per graded command for two bodies achieving 40 degrees: 40 commands with an
    advantage of 0.5 +- 0.05 (the gate opens at the 32nd reading), 60 with +-0.5 around 0 (it closes), 50 with
    the advantage again (it reopens)."""
    out = []
    for k in range(150):
        j = 1.0 if k % 2 else -1.0
        if k < 40 or k >= 100:
            out.append((40.0, 40.0 * (0.5 + 0.05 * j)))          # reading 1 - (0.5 +- 0.05)
        else:
            out.append((40.0 * (0.75 + 0.25 * j), 40.0 * (0.75 - 0.25 * j)))   # reading +-0.5
    return out


def same_control_violations(tr_factory, n=150):
    """What is wrong with a shadow gate fed the SAME control as the gate in force: after every graded command its
    readings and its open or shut state must equal the gate's, through an opening, a closing and a reopening, and
    the window's counts must say so."""
    bad = []
    tr = tr_factory()
    states = []
    for k, (own, ctl) in enumerate(_phases()[:n]):
        _grade(tr, k, own=[own, own], absolute=[ctl, ctl], sized=[ctl, ctl])
        if list(tr.shadow_gate_readings) != list(tr.gate_readings):
            bad.append("command %d: the readings differ" % k)
            break
        if tr.shadow_gate_is_open != tr.gate_is_open or tr.shadow_gate()[0] != tr.gate()[0]:
            bad.append("command %d: the gate in force is %s, the shadow gate %s"
                       % (k, tr.gate_is_open, tr.shadow_gate_is_open))
            break
        states.append(bool(tr.gate_is_open))
    if not bad:
        changes = [k for k in range(1, len(states)) if states[k] != states[k - 1]]
        if not (len(changes) >= 3 and states[N - 1] and not states[N - 2]):
            bad.append("the run did not open at the %dth reading, close and reopen: changes at %s" % (N, changes))
        s = tr.summary(2)
        opened = sum(states)
        want = {"gate_commands": n, "shadow_gate_commands": n, "gates_same_commands": n,
                "gate_open_commands": opened, "shadow_gate_open_commands": opened,
                "gates_both_open_commands": opened, "gate_only_open_commands": 0,
                "shadow_gate_only_open_commands": 0}
        for key, v in want.items():
            if s.get(key) != v:
                bad.append("%s reads %s, counted %s" % (key, s.get(key), v))
        g = tr._gate_stats()
        if (s["shadow_gate_mean"], s["shadow_gate_standard_error"], s["shadow_gate_readings_above_zero"]) != (
                round(g[1], 5), round(g[2], 5), g[3]):
            bad.append("the shadow gate's mean, standard error or readings above zero differ from the gate's")
        if (s["gate_mean"], s["gate_standard_error"], s["gate_readings_above_zero"]) != (
                s["shadow_gate_mean"], s["shadow_gate_standard_error"], s["shadow_gate_readings_above_zero"]):
            bad.append("the block's two gates read differently on the same readings")
    return bad


def test_fed_the_same_control_the_shadow_gate_equals_the_gate_in_force_at_every_command():
    assert same_control_violations(_tracker) == []


# ------------------------------------------------------------------------------ the counts
def count_violations(tr_factory):
    """What is wrong with the per-window counts. Both bodies achieve a 40-degree turn and the body head expected
    30 degrees (agreement 0.75). Window 1, 40 graded commands: the absolute control's expectation alternates 10
    and 12 degrees (agreements 0.25 and 0.3: the own expectation beats it by 0.5 and 0.45), and the size-matched
    control's alternates 36 and 24 degrees (agreements 0.9 and 0.6: readings -0.15 and +0.15, no advantage) - the
    case the size-matched control exists for. The gate in force opens at its 32nd reading and is open at
    commands 32 to 40 (9 commands); the shadow gate never opens. Window 2, 40 more with the two controls
    exchanged."""
    bad = []
    tr = tr_factory()
    beaten = lambda k: [10.0 if k % 2 else 12.0] * 2        # noqa: E731
    level = lambda k: [36.0 if k % 2 else 24.0] * 2         # noqa: E731
    for k in range(40):
        _grade(tr, k, own=[30.0, 30.0], absolute=beaten(k), sized=level(k))
    s = tr.summary(2)
    want = {"gate_commands": 40, "gate_open_commands": 40 - (N - 1), "shadow_gate_commands": 40,
            "shadow_gate_open_commands": 0, "gates_same_commands": 40, "gates_both_open_commands": 0,
            "gate_only_open_commands": 40 - (N - 1), "shadow_gate_only_open_commands": 0}
    for key, v in want.items():
        if s.get(key) != v:
            bad.append("window 1: %s reads %s, counted %s" % (key, s.get(key), v))
    if not (s["gate_open"] is True and s["shadow_gate_open"] is False):
        bad.append("window 1: gate_open %s, shadow_gate_open %s" % (s["gate_open"], s["shadow_gate_open"]))
    if abs(s["shadow_gate_mean"]) > 1e-6 or s["shadow_gate_readings_above_zero"] != N // 2:
        bad.append("window 1: readings of -0.15 and +0.15 in turn must read a mean of 0 with half above zero: %s, %s"
                   % (s["shadow_gate_mean"], s["shadow_gate_readings_above_zero"]))
    if abs(s["gate_mean"] - 0.475) > 1e-4 or s["gate_readings_above_zero"] != N:
        bad.append("window 1: readings of 0.5 and 0.45 in turn must read a mean of 0.475, all above zero: %s, %s"
                   % (s["gate_mean"], s["gate_readings_above_zero"]))
    # the window's counts start again; the gates' states and readings carry on. Now the absolute control is level
    # with the own expectation and the size-matched one is beaten: the gate in force closes at the first reading
    # that brings its mean under 1 standard error, the shadow gate opens when its mean clears 3
    states = []
    for k in range(40, 80):
        _grade(tr, k, own=[30.0, 30.0], absolute=level(k), sized=beaten(k))
        states.append((bool(tr.gate_is_open), bool(tr.shadow_gate_is_open)))
    s2 = tr.summary(2)
    want2 = {"gate_commands": 40, "shadow_gate_commands": 40, "gates_same_commands": 40,
             "gate_open_commands": sum(a for a, _ in states),
             "shadow_gate_open_commands": sum(b for _, b in states),
             "gates_both_open_commands": sum(a and b for a, b in states),
             "gate_only_open_commands": sum(a and not b for a, b in states),
             "shadow_gate_only_open_commands": sum(b and not a for a, b in states)}
    for key, v in want2.items():
        if s2.get(key) != v:
            bad.append("window 2: %s reads %s, counted %s" % (key, s2.get(key), v))
    if not (want2["shadow_gate_open_commands"] > 0 and want2["gate_open_commands"] > 0
            and want2["gate_open_commands"] < 40 and states[-1] == (False, True)):
        bad.append("window 2 did not close the gate in force and open the shadow gate: %s" % states[-3:])
    if tr.summary(2)["gate_commands"] != 0:
        bad.append("a window with no graded command still counts commands")
    return bad


def test_the_window_counts_say_how_often_each_gate_was_open():
    assert count_violations(_tracker) == []


# ------------------------------------------------------------------------------ nothing on the training path
def _readings(mean_in_se, n=N):
    """n readings whose mean stands `mean_in_se` of their own standard errors above 0."""
    base = [(-1.0) ** i for i in range(n)]
    m0 = sum(base) / n
    sd = (sum((v - m0) ** 2 for v in base) / (n - 1)) ** 0.5
    return [v + mean_in_se * sd / n ** 0.5 for v in base]


def _with_states(real_open, shadow_open):
    """A tracker whose two gates stand as asked, each on readings that keep it there."""
    tr = _tracker()
    tr.gate_readings.extend(_readings(3.5 if real_open else 0.5))
    tr._update_gate()
    tr.shadow_gate_readings.extend(_readings(3.5 if shadow_open else 0.5))
    tr._update_shadow_gate()
    assert tr.gate_is_open is real_open and tr.shadow_gate_is_open is shadow_open
    return tr


def leak_violations(tr_factory=None):
    """What is wrong when the shadow gate's state reaches the weight or the ranking: under every pairing of the two
    gates' states, ControllabilityTracker.gate() and LemLite._rank_weights_now must answer by the gate in force
    alone, and the normalised ranking must be equal for both shadow states."""
    from mimo_bridge.lem_model import LemLite
    bad = []
    g = torch.Generator().manual_seed(5)
    names = ["integrity", "controllability", "coherence", WC.PAIN_ROW, WC.HARM_ROW]
    ch = torch.randn(30, len(names), 2, generator=g)
    ch[:, 3:] = 0.1
    norm = WC.ChannelNormalizer(horizon=100000)
    for t in range(ch.shape[0]):
        for ci, n in enumerate(names[:3]):
            norm.update(n, ch[t, ci])
    floors = {n: 1e-6 for n in names}
    ranked = {}
    for real_open in (False, True):
        for shadow_open in (False, True):
            tr = _with_states(real_open, shadow_open)

            class Stub:
                rank_weights = dict(WC.DEFAULT_WEIGHTS)
                wellness_channels = WC.WellnessChannelState(True, tr, None)
            if tr.gate()[0] is not real_open:
                bad.append("gate() answers %s with the gate in force %s and the shadow gate %s"
                           % (tr.gate()[0], real_open, shadow_open))
            w = LemLite._rank_weights_now(Stub())
            want = WC.DEFAULT_WEIGHTS["controllability"] if real_open else 0.0
            if w["controllability"] != want:
                bad.append("the controllability weight is %s with the gate in force %s and the shadow gate %s"
                           % (w["controllability"], real_open, shadow_open))
            if {k: v for k, v in w.items() if k != "controllability"} != {
                    k: v for k, v in WC.DEFAULT_WEIGHTS.items() if k != "controllability"}:
                bad.append("another channel's weight moved")
            ranked[(real_open, shadow_open)] = WC.rank_values(ch, names, w, norm, floors, 0.9)[0]
            # judging the shadow gate again leaves the gate in force as it stood
            before = (bool(tr.gate_is_open), list(tr.gate_readings))
            tr.shadow_gate_readings.extend(_readings(0.5 if shadow_open else 3.5))
            tr._update_shadow_gate()
            if (bool(tr.gate_is_open), list(tr.gate_readings)) != before:
                bad.append("an update of the shadow gate changed the gate in force")
    for real_open in (False, True):
        if not torch.equal(ranked[(real_open, False)], ranked[(real_open, True)]):
            bad.append("the ranking differs between the shadow gate's two states")
    if torch.equal(ranked[(False, False)], ranked[(True, False)]):
        bad.append("the ranking does not depend on the gate in force (the check cannot see a leak)")
    return bad


def test_the_weight_and_the_ranking_read_the_gate_in_force_under_every_shadow_state():
    assert leak_violations() == []


def test_a_graded_command_changes_the_gate_in_force_as_it_did_before_the_shadow_gate():
    """The readings, state, running average and window sums the gate in force had before the shadow gate existed
    are the same with the size-matched expectation present or absent, on a run that opens and closes the gate."""
    a, b = _tracker(), _tracker()
    for k, (own, ctl) in enumerate(_phases()):
        j = 1.0 if k % 3 else -1.0
        ra = _grade(a, k, own=[own, own], absolute=[ctl, ctl], sized=[own * (0.75 + 0.2 * j)] * 2)
        rb = _grade(b, k, own=[own, own], absolute=[ctl, ctl], sized=None)
        assert ra == rb
        assert list(a.gate_readings) == list(b.gate_readings) and a.gate_is_open == b.gate_is_open
        assert a.gate() == b.gate()
        assert torch.equal(a.value(2, "cpu"), b.value(2, "cpu"))
    sa, sb = a.summary(2), b.summary(2)
    old_keys = ("value_mean", "per_body", "measured_share", "agreement_mean", "control_mean", "difference_mean",
                "command_agreement_mean", "by_author", "by_author_orders", "orders", "source", "tenure_mismatch",
                "efference_missing", "gate_open", "gate_mean", "gate_line", "gate_why", "gate_commands",
                "gate_open_commands")
    assert {k: sa[k] for k in old_keys} == {k: sb[k] for k in old_keys}
    assert sa["shadow_gate_commands"] == 150 and sb["shadow_gate_commands"] == 0


# ------------------------------------------------------------------------------ save and load
def test_the_shadow_gate_is_saved_and_follows_the_gates_load_rule(tmp_path):
    tr = _tracker()
    tr.ema_m, tr.ema_w = torch.tensor([0.3, 0.1]), torch.tensor([0.5, 0.2])
    tr.gate_readings.extend([0.1, -0.2, 0.3])
    tr.shadow_gate_readings.extend([0.4, -0.5])
    sd = tr.state_dict()
    assert sd["shadow_gate_readings"] == [0.4, -0.5] and sd["shadow_control_kind"] == WC.SHADOW_CONTROL_KIND
    assert sd["gate_readings"] == [0.1, -0.2, 0.3] and sd["control_kind"] == "shuffled order"
    # through a file, as runtime_state.pt carries it
    path = tmp_path / "runtime_state.pt"
    state = WC.WellnessChannelState(True, tr, None)
    torch.save({"wellness_channels": state.state_dict()}, str(path))
    rt = torch.load(str(path), weights_only=False, map_location="cpu")
    tr2 = _tracker()
    st2 = WC.WellnessChannelState(True, tr2, None)
    assert "controllability" in st2.load_state_dict(rt["wellness_channels"])
    assert list(tr2.shadow_gate_readings) == [0.4, -0.5] and list(tr2.gate_readings) == [0.1, -0.2, 0.3]
    assert torch.equal(tr2.value(2, "cpu"), tr.value(2, "cpu"))
    # THE LOAD RULE, the same for both gates: the readings come back, the gate starts shut and is judged again by
    # the opening line. With the same readings in both, the two gates stand alike after every load, whatever
    # state each was saved in
    for mean_in_se, opens in ((0.5, False), (2.0, False), (3.5, True)):
        for was_open in (False, True):
            src = _tracker()
            src.gate_readings.extend(_readings(mean_in_se))
            src.shadow_gate_readings.extend(_readings(mean_in_se))
            src.gate_is_open = src.shadow_gate_is_open = was_open
            dst = _tracker()
            dst.gate_is_open = dst.shadow_gate_is_open = True          # a state the load must not keep
            dst.load_state_dict(src.state_dict())
            assert list(dst.shadow_gate_readings) == list(src.shadow_gate_readings)
            assert dst.gate_is_open is opens and dst.shadow_gate_is_open is opens, (mean_in_se, was_open)
    # only the newest 32 readings are kept, as for the gate
    src = _tracker()
    sd = src.state_dict()
    sd["shadow_gate_readings"] = [float(i) for i in range(N + 5)]
    dst = _tracker()
    dst.load_state_dict(sd)
    assert list(dst.shadow_gate_readings) == [float(i) for i in range(5, N + 5)]


def test_a_state_saved_before_the_shadow_gate_loads_with_it_empty_and_shut():
    """The keys production's ControllabilityTracker.state_dict writes (commit e022cae26: ema_m, ema_w, alpha,
    gate_readings, control_kind), with readings that open the gate in force: the gate reopens as before, the shadow
    gate is empty and shut; a state whose shadow readings were taken against another control is not carried."""
    old = {"ema_m": torch.tensor([0.3, 0.1]), "ema_w": torch.tensor([0.5, 0.2]), "alpha": 1.0,
           "gate_readings": _readings(3.5), "control_kind": "shuffled order"}
    tr = _tracker()
    tr.shadow_gate_readings.extend(_readings(3.5))       # what this tracker held before the load is dropped
    tr.shadow_gate_is_open = True
    tr.load_state_dict(old)
    assert tr.gate()[0] is True and list(tr.gate_readings) == pytest.approx(_readings(3.5))
    assert len(tr.shadow_gate_readings) == 0 and tr.shadow_gate_is_open is False
    s = tr.summary(2)
    assert s["shadow_gate_open"] is False and s["shadow_gate_mean"] is None and s["shadow_gate_readings_held"] == 0
    assert s["gate_open"] is True and s["gate_readings_held"] == N
    other = dict(tr.state_dict(), shadow_gate_readings=_readings(3.5), shadow_control_kind="shuffled order")
    tr2 = _tracker()
    tr2.load_state_dict(other)
    assert len(tr2.shadow_gate_readings) == 0 and tr2.shadow_gate_is_open is False
    # the whole wellness-channel state of an older build (no controllability entry at all) loads too
    st = WC.WellnessChannelState(True, _tracker(), None)
    assert st.load_state_dict({"version": WC.WELLNESS_CHANNELS_VERSION, "normalizer": None,
                               "controllability": None, "learning_progress": None}) == []
    assert st.control.shadow_gate_is_open is False
    # and this build's state read by the older build's loader: it reads only the keys it knows
    new = _tracker()
    new.gate_readings.extend(_readings(3.5))
    new.shadow_gate_readings.extend(_readings(3.5))
    sd = new.state_dict()
    assert {"ema_m", "ema_w", "alpha", "gate_readings", "control_kind"} <= set(sd)
    assert all(isinstance(v, float) for v in sd["shadow_gate_readings"])


# ------------------------------------------------------------------------------ the pages
def _window_block():
    """The tracker's own block for count_violations' first window: the gate in force open after 9 of 40 graded
    commands, the shadow gate after none."""
    tr = _tracker()
    for k in range(40):
        _grade(tr, k, own=[30.0, 30.0], absolute=[10.0 if k % 2 else 12.0] * 2, sized=[36.0 if k % 2 else 24.0] * 2)
    return tr.summary(2)


def test_both_pages_print_the_two_gates_as_counts_of_graded_commands():
    from mimo_bridge.tools import dashboard
    from mimo_bridge.tools import lineage_series as ls
    from mimo_bridge.tools.check_no_bare_decimals import bare_decimals, long_numbers, unanchored_numbers
    block = _window_block()
    text = ls.controllability_gates_text(block)
    assert ("the gate in force (its control is another body's absolute command) stood open after 9 of the 40 "
            "graded commands it judged") in text
    assert ("the shadow gate (its control is another body's size-matched command; recorded only, no weight reads "
            "it) stood open after 0 of the 40 graded commands it judged") in text
    assert ("On the 40 graded commands both gates judged, both stood open after 0, only the gate in force after 9 "
            "and only the shadow gate after 0") in text
    assert text.endswith("At the window's end the gate in force is open and the shadow gate is shut.")
    # counts only, in the canonical words
    assert not bare_decimals(text) and not long_numbers(text) and not unanchored_numbers(text)
    low = text.lower()
    for word in (" order", "moment", " tick", "reward"):
        assert word not in low, word
    # a record written before the shadow gate has no such sentence and no row; never a 0
    old = {k: v for k, v in block.items()
           if "shadow" not in k and not k.startswith(("gates_", "size_matched_", "gate_definitions"))}
    assert ls.controllability_gates_text(old) is None and ls.controllability_gates_text(None) is None
    # a window in which only the gate in force took a reading (no size-matched expectation in the records)
    tr = _tracker()
    _grade(tr, 0, own=[40.0, 20.0], absolute=[-40.0, 10.0], sized=None)
    one = ls.controllability_gates_text(tr.summary(2))
    assert "stood open after 0 of the 1 graded command it judged" in one and "took no reading" in one
    assert "both gates judged" not in one
    # a window in which neither did
    assert ls.controllability_gates_text(_tracker().summary(2)).startswith("Neither controllability gate took")
    # the run page: the row, with its explanation, only where the record carries the shadow gate
    label = "Controllability gates on the same commands"
    assert dict(dashboard.gof_rows({"controllability": block}))[label] == text
    assert label not in dict(dashboard.gof_rows({"controllability": old}))
    key = dashboard.doc_for(label)
    assert key == label and dashboard.ROW_DOC[key]["use"] >= 8 and dashboard.ROW_DOC[key]["clear"] >= 8
    entry = " ".join(dashboard.ROW_DOC[key][f] for f in ("t", "w", "h", "f")).lower()
    for field in ("gate_commands", "gate_open_commands", "shadow_gate_commands", "shadow_gate_open_commands",
                  "gates_same_commands", "size_matched_control_mean", "gate_definitions"):
        assert field in entry, field
    for word in (" order", "moment", " tick", "reward"):
        assert word not in entry, word
    assert dashboard.doc_for("Controllability channel") == "Controllability channel"
    # the Development page: the size-matched control's series read the block's fields, and the controllability
    # card carries the sentence
    rec = {"controllability": block}
    assert ls.SERIES["ctl.control_sized"]["get"](rec) == block["size_matched_control_mean"]
    assert ls.SERIES["ctl.difference_sized"]["get"](rec) == block["size_matched_difference_mean"]
    assert ls.SERIES["ctl.control_sized"]["get"]({"controllability": old}) is None
    for key in ("ctl.control_sized", "ctl.difference_sized"):
        words = (ls.SERIES[key]["title"] + " " + ls.SERIES[key]["note"]).lower()
        assert "command" in words and " order" not in words
    panel = ls.gain_of_function({"type": "mimo", "step": 1, "timestamp": 1, "controllability": block},
                                {"controllability": "on"}, {})
    card = {r["item"]: r for r in panel["items"]}[2]
    assert text in ls.item_text(card)


def test_the_explainer_charts_each_gates_open_commands_out_of_every_100_it_judged():
    from mimo_bridge.tools import build_explainer as BX
    block = _window_block()
    recs = [{"step": 100 * (i + 1), "type": "mimo", "timestamp": 1_790_000_000 + i,
             "controllability": dict(block, gate_open_commands=9 + i, shadow_gate_open_commands=i)}
            for i in range(4)]
    out = BX.build_series({"mimo": recs, "training": [], "launch": []}, [])
    charts = [c for c in out["charts"] if c["doc"] == "chart.controllability_gates"]
    assert len(charts) == 1 and charts[0]["unit"] == "/100" and not charts[0].get("ends")
    series = {s["key"]: [p[1] for p in s["points"]] for s in charts[0]["series"]}
    assert series["controllability.gate_open_commands_per100"] == pytest.approx([22.5, 25.0, 27.5, 30.0])
    assert series["controllability.shadow_gate_open_commands_per100"] == pytest.approx([0.0, 2.5, 5.0, 7.5])
    doc = BX.DOC_CHARTS["chart.controllability_gates"]
    words = " ".join(doc.values()).lower()
    for field in ("gate_open_commands", "gate_commands", "shadow_gate_open_commands", "shadow_gate_commands",
                  "gate_definitions"):
        assert field in words, field
    for word in (" order", "moment", " tick", "reward"):
        assert word not in words, word
    # a log written before the shadow gate names the chart as a feature the run lacks, never draws a 0
    before = [{"step": r["step"], "type": "mimo", "timestamp": r["timestamp"],
               "controllability": {"agreement_mean": 0.02, "control_mean": 0.0, "measured_share": 0.9,
                                   "gate_open": False}} for r in recs]
    out = BX.build_series({"mimo": before, "training": [], "launch": []}, [])
    assert not [c for c in out["charts"] if c["doc"] == "chart.controllability_gates"]
    assert any("a gate judged" in label for label in out["off"])
    # a window in which a gate took no reading draws no point for it
    quiet = [dict(r, controllability=dict(r["controllability"], shadow_gate_commands=0,
                                          shadow_gate_open_commands=0)) for r in recs]
    out = BX.build_series({"mimo": quiet, "training": [], "launch": []}, [])
    keys = [s["key"] for c in out["charts"] if c["doc"] == "chart.controllability_gates" for s in c["series"]]
    assert keys == ["controllability.gate_open_commands_per100"]


def test_the_explainers_control_term_names_the_shadow_gates_fields():
    from mimo_bridge.explainer import schema
    term = schema.DOC["term.shuffled-command-control"]["w"]
    for field in ("size_matched_control_mean", "size_matched_difference_mean", "shadow_gate_open",
                  "shadow_gate_commands", "shadow_gate_open_commands", "gate_commands", "gate_open_commands"):
        assert field in term, field
    assert "recorded only" in term and "evidence gate (control_mean)" in term


# ------------------------------------------------------------------------------ the learner
@pytest.fixture(scope="module")
def fx():
    import importlib
    return importlib.import_module("mimo_bridge.tests.swap_test_fixture")


def _learner(fx):
    import random

    import numpy as np
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    return fx.build(fx.argv())


def _run(fx, n, hold_shadow=None, monkeypatch=None, leak=False):
    """Drive the fixture n model steps. hold_shadow True / False: the shadow gate is held open / shut from the
    first graded command on (its update is replaced). leak=True plants the defect the comparison must see: the
    gate in force answers open whenever the shadow gate is.
    THE STILLNESS FLOORS ARE SET TO 0 FOR THESE RUNS. Measured 2026-10-03 on this fixture (60 model steps, 11
    graded commands): the fresh body head's expected effect is 0.9 to 3.0 measured hold spreads per body, under
    the floor of 6.67 (wellness_channels.CONTROL_FLOOR, measured on H84's bodies), so with the floors as they
    stand no body is measured and neither gate takes a reading; at 0 every body is measured and both gates take
    a reading at every graded command."""
    monkeypatch.setattr(WC, "CONTROL_FLOOR", {"efference": 0.0, "command": 0.0})
    if hold_shadow is not None:
        monkeypatch.setattr(WC.ControllabilityTracker, "_update_shadow_gate",
                            lambda self: setattr(self, "shadow_gate_is_open", bool(hold_shadow)))
    if leak:
        real_gate = WC.ControllabilityTracker.gate

        def leaking(self):
            ok, mean, line, why = real_gate(self)
            return bool(ok or self.shadow_gate_is_open), mean, line, why
        monkeypatch.setattr(WC.ControllabilityTracker, "gate", leaking)
    try:
        m, a = _learner(fx)
        weights = []
        real_weights = m._rank_weights_now

        def spy():
            w = real_weights()
            weights.append(w["controllability"])
            return w
        m._rank_weights_now = spy
        rows = fx.drive(m, fx.SyntheticWorld(a.n_envs, seed=1), n)
        tr = m.wellness_channels.control
        return dict(rows=rows, digest=fx.state_digest(m), weights=weights,
                    shadow_readings=len(tr.shadow_gate_readings), readings=len(tr.gate_readings),
                    shadow_open=bool(tr.shadow_gate_is_open), summary=m.controllability_window_summary())
    finally:
        if monkeypatch is not None:
            monkeypatch.undo()


def test_the_learner_trains_alike_with_the_shadow_gate_held_open_or_shut(fx, monkeypatch):
    """45 model steps of the fixture learner (H84's learner flags at width 32, 3 bodies): the shadow gate takes
    a reading at every graded command, the stats block carries its keys, and a run with the shadow gate held
    open is equal to one with it held shut in every row, in the state digest and in the controllability weight
    of every planning event; with the leak planted the weights differ."""
    shut = _run(fx, 45, hold_shadow=False, monkeypatch=monkeypatch)
    held = _run(fx, 45, hold_shadow=True, monkeypatch=monkeypatch)
    assert shut["shadow_readings"] > 0 and shut["shadow_readings"] == shut["readings"]
    assert held["shadow_open"] is True and shut["shadow_open"] is False
    assert held["rows"] == shut["rows"] and held["digest"] == shut["digest"]
    assert held["weights"] and held["weights"] == shut["weights"] and set(held["weights"]) == {0.0}
    s = held["summary"]
    assert s["source"] == "efference" and s["shadow_gate_commands"] == s["gate_commands"] > 0
    assert s["shadow_gate_open_commands"] == s["shadow_gate_commands"] and s["gate_open_commands"] == 0
    assert s["shadow_gate_only_open_commands"] == s["gates_same_commands"] == s["gate_commands"]
    assert s["size_matched_control_mean"] is not None and s["control_mean"] is not None
    leaking = _run(fx, 45, hold_shadow=True, monkeypatch=monkeypatch, leak=True)
    assert leaking["weights"] != held["weights"], "the comparison cannot see the shadow gate reaching the weight"
