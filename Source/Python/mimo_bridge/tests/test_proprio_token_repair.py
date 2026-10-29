# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""--proprio-token-repair within_body, sixth version (mimo_bridge/proprio_token.py; docs/gain_of_function_v4/agents/
proprio_token_2026-09-25.md sections 42-45; the reviews docs/gain_of_function_v4/agents/proprio_token_review_2026-09-25.md,
proprio_token_review2_2026-09-26.md and proprio_token_review3_2026-09-26.md).

What is pinned: the default build is today's model (nothing imported, stamped, saved or reported; no parameter); the
line is unmeasured until the calibration gives it and a line on another scale (version 4's R2) is refused as a
mistyping; the window is the replay ring's 300 s at every 5th real reading per body, read in time order; the reading is
each body's response per unit of its own posture movement, and the posture-only rows hold the velocity and capacity
numbers at their window mean; the lesson pushes every body under the line and no other, whatever it moves (no movement
floor: the coordinator's decision), never a skipped body and never a body whose posture did not change (no ratio); its
push goes through the posture-only rows, so the velocity weights receive none of it (the control: a push on the plain
response moves them); its gradient is added after the clip at a set share of each encoder tensor's running gradient
size, whatever the clip norm, and the step each weight takes is reported in learning rates per input field; at the
gauge cadence it reads the posture-only ratio and the content control, and each body's attributed change of ln ratio
(the lesson's own first-order share, which matches a step taken); the start is read once, before the first update, and
rides runtime_state.pt with the attributed lift, and a start of another version is not restored; it lifts a deaf
encoder's ratio to the line against a clipped stasis pressure and its posture content rises with it; its gradient
reaches the proprioception encoder and nothing else; the heading-free rows reach the movement; the switch trains
bit-identically until the window is full; the trainer flags. Version 6: DEAF is measured and refused when missing or
not under the line; the hinge engages a body under DEAF and releases it at the line, leaving a body between them alone;
the hold keeps a pushed body's fatigue answer at the live bodies' largest at the start (its line recorded in the start,
unmeasured with no live body; it acts only on bodies the lesson has pushed, also after their release; its gradient is
the hand-computed one and lowers the fatigue change); the gauges read the content with the token's gain divided out (a
planted token that moves with the posture and carries no posture information reads less), the fatigue answer, the
jitter shares of the token and of the posture by one formula, and a held-out reading of the window through the encoder
saved one window earlier; the engaged and pushed flags, the saved encoder and the held-out reading ride
runtime_state.pt. Version 7 (the record's section 47.5; the review's finding 4): the hold's line is the median fatigue
answer on the posture-only rows of the judged bodies the lesson never pushed, read at the start and again each window,
kept when no such body moves; the line in force and its step ride runtime_state.pt, and a version-6 start is not
restored."""
import inspect
import math
import os

import pytest
import torch

from mimo_bridge import proprio_token as PT
from mimo_bridge.proprio_token import WithinBody, posture_columns, posture_content

LINE = 3.0          # a stand-in line for the tests: the real one is the calibration job's measurement (3.75 on H84)
DEAF = 2.0          # a stand-in DEAF (2.86 on H84's committed readings)
DEAF_T = 2.99       # a stand-in DEAF just under LINE: every body the version-5 tests push reads under it (engaged)
DEAF_MAX = PT.LINE_MAX - 0.01
FLAG = dict(proprio_token_repair="within_body", proprio_token_line=LINE, proprio_token_deaf=DEAF)


# ---------------------------------------------------------------- the constants
def test_the_window_is_the_replay_rings_300_s_at_the_collapse_watchs_sampling_and_the_lesson_reads_it_full():
    src = inspect.getsource(__import__("mimo_bridge.lem_model", fromlist=["LemLite"]))
    assert "if self.step_count % 5 == 0:" in src
    assert "def enable_replay(self, n_envs: int, size_per_env: int = 6000):" in src
    assert PT.SAMPLE_EVERY == 5 and PT.WINDOW == 1200 and PT.WINDOW * PT.SAMPLE_EVERY == 6000, \
        "300 s of body time at 0.05 s per model step: the replay ring's span"
    assert PT.MIN_SAMPLES == PT.WINDOW, "review finding 5: no lesson on a part-filled window"
    assert PT.VERSION == "proprio_token_repair_v7" and PT.READS_VERSIONS[-1] == PT.VERSION
    assert "proprio_token_repair_v6" in PT.READS_VERSIONS, "version 6's window rows are read (its start is not)"
    assert PT.GAUGE_EVERY == 100, "the stats record's cadence (the watcher's REPORT_EVERY)"
    assert PT.CONTROL_LAMBDA == 200.0, "the content control's penalty: version 4's calibrated RULE LAMBDA"
    assert PT.FOLDS == 10 and PT.GAP * PT.SAMPLE_EVERY == 50 and PT.SHIFT * PT.SAMPLE_EVERY == 3000


def test_the_line_is_unmeasured_and_a_line_on_another_scale_is_refused():
    """The line on the ratio is the calibration's RULE LINE (version 5), never assumed; version 4's line was an R2
    (0.32), which typed into version 5 would push no body at all: a value outside [1, 20] is refused as a mistyping."""
    assert PT.LINE is None
    with pytest.raises(ValueError, match="line on the response per unit movement is unmeasured"):
        WithinBody()
    for bad in (0.32, 0.9, 25.0):
        with pytest.raises(ValueError, match="mistyping"):
            WithinBody(line=bad)
    with pytest.raises(ValueError, match="finite"):
        WithinBody(line=float("nan"))
    for good in (3.75, PT.LINE_MAX):
        assert WithinBody(line=good, deaf=PT.LINE_MIN).line == good
    assert not hasattr(PT, "STILL") and not hasattr(PT, "LAMBDA"), "version 4's movement floor and penalty are gone"


def test_deaf_is_measured_and_lies_under_the_line():
    """Version 6: the hinge engages a body under DEAF, the calibration's RULE DEAF in the lesson's units: unmeasured
    until then (refused without it), on the same scale as the line (a value outside [1, 20] is a mistyping) and under
    the line (a body engaged at or over the line would be released at once)."""
    assert PT.DEAF is None
    with pytest.raises(ValueError, match="deaf on the response per unit movement is unmeasured"):
        WithinBody(line=LINE)
    with pytest.raises(ValueError, match="mistyping"):
        WithinBody(line=LINE, deaf=0.5)
    for bad in (LINE, LINE + 0.5):
        with pytest.raises(ValueError, match="must lie under"):
            WithinBody(line=LINE, deaf=bad)
    assert WithinBody(line=LINE, deaf=DEAF).deaf == DEAF


def test_the_strength_carries_its_arithmetic():
    assert PT.STRENGTH == 0.02 and PT.STRENGTH_MAX == 0.1
    for bad in (0.0, -0.1, PT.STRENGTH_MAX * 1.01, float("nan")):
        with pytest.raises(ValueError, match="strength"):
            WithinBody(line=LINE, deaf=DEAF_T, strength=bad)
    assert WithinBody(line=LINE, deaf=DEAF_T, strength=0.05).strength == 0.05


def test_the_skipped_bodies_are_numbered_from_one():
    assert PT.parse_skip(None) == () and PT.parse_skip("3") == (3,) and PT.parse_skip("5,3") == (3, 5)
    assert PT.parse_skip([3]) == (3,) and PT.parse_skip("3 5") == (3, 5)
    for bad in ("0", "-1", "a", "2.5"):
        with pytest.raises(ValueError, match="number from 1"):
            PT.parse_skip(bad)
    assert WithinBody(line=LINE, deaf=DEAF_T, skip="3").skip == (3,)
    assert WithinBody(line=LINE, deaf=DEAF_T, skip="3").pushable(4).tolist() == [True, True, False, True]


def test_the_posture_part_is_the_quaternion_and_position_and_the_held_part_is_velocity_and_capacity():
    cols = posture_columns(312)
    assert len(cols) == 26 * 7 and {int(c) % 12 for c in cols} == set(range(7))
    held = PT.held_columns(312)
    assert len(held) == 26 * 4 and {int(c) % 12 for c in held} == {7, 8, 9, 10}
    with pytest.raises(ValueError):
        posture_columns(313)
    g = torch.Generator().manual_seed(0)
    X = torch.randn(2, 50, 312, generator=g)
    Xp = PT.posture_only(X)
    assert torch.equal(Xp[..., cols], X[..., cols]), "the posture moves as it did"
    assert torch.allclose(Xp[..., held], X[..., held].mean(1, keepdim=True).expand_as(Xp[..., held])), \
        "velocity and capacity sit at their window mean"
    row = [j for j in range(312) if j % 12 == 11]
    assert torch.equal(Xp[..., row], X[..., row]), "the row index is left"


# ---------------------------------------------------------------- the window
def test_the_window_keeps_every_fifth_real_reading_per_body_in_time_order():
    w = WithinBody(line=LINE, deaf=DEAF_T)
    x = torch.randn(3, 312)
    kept = [w.observe(x + i, i, real=True) for i in range(26)]
    assert kept == [i % 5 == 0 for i in range(26)]
    assert w.count == 6 and w.ring.shape == (3, PT.WINDOW, 312)
    assert not w.observe(x, 30, real=False), "a reading that is not real is never kept"
    assert w.count == 6
    w = WithinBody(line=LINE, deaf=DEAF_T)
    for i in range(PT.WINDOW + 7):
        w.observe(torch.full((2, 312), float(i)), 5 * i, real=True)
    X = w.window()
    assert X.shape == (2, PT.WINDOW, 312)
    assert float(X[0, 0, 0]) == 7.0 and float(X[0, -1, 0]) == float(PT.WINDOW + 6)
    assert bool((X[0, 1:, 0] - X[0, :-1, 0] == 1).all()), "time order across the wrap"
    w.observe(torch.randn(3, 312), 5000, real=True)
    assert w.count == 1 and w.ring.shape[0] == 3 and "ring_restarted" in w.last


def test_the_lesson_waits_for_a_full_window():
    w = WithinBody(line=LINE, deaf=DEAF_T)
    for i in range(PT.WINDOW - 1):
        w.observe(torch.randn(2, 312), 5 * i, real=True)
    assert w.window() is None and w.lesson(lambda xx: xx[:, :192]) is None
    w.observe(torch.randn(2, 312), 5 * PT.WINDOW, real=True)
    assert w.window().shape == (2, PT.WINDOW, 312)


# ---------------------------------------------------------------- the reading
def _latent(B, n, k=4, seed=0, step=0.05):
    """A slow walk in k directions per body: a body's joints move together, in few directions at once."""
    g = torch.Generator().manual_seed(seed)
    return torch.cumsum(step * torch.randn(B, n, k, generator=g), dim=1)


def _fed(w, X):
    for i in range(X.shape[1]):
        w.observe(X[:, i], 5 * i, real=True)
    return w


def _gain_bodies(gains, seed=0, moves=None):
    """Bodies whose posture walks (4 directions) beside a large constant, and a linear encoder with a LayerNorm whose
    posture weights are scaled per body by `gains` (the body's region of a deaf or live encoder): the token's response
    per unit movement follows the gain. moves: each body's posture movement scale. Every body shares one constant and
    one walk, so they differ only by their gain and the size of their movement."""
    g = torch.Generator().manual_seed(seed)
    B = len(gains)
    cols = posture_columns(312)
    X = 0.5 * torch.randn(1, 1, 312, generator=g).expand(B, PT.WINDOW, 312).clone()
    walk = (_latent(1, PT.WINDOW, seed=seed + 1) @ torch.randn(4, len(cols), generator=g)).expand(B, -1, -1)
    walk = walk / walk.std(1).mean(-1).view(B, 1, 1)
    X[..., cols] += walk * torch.tensor(moves or [0.06] * B).view(B, 1, 1)
    M = torch.randn(312, 192, generator=g) / 312 ** 0.5
    scale = torch.tensor(gains, dtype=torch.float32)

    def enc(xx):
        x = xx.reshape(B, -1, 312)
        c = x.mean(1, keepdim=True)
        return torch.nn.functional.layer_norm(((x - c) * scale.view(B, 1, 1) + 4.0 * c) @ M, (192,)).reshape(-1, 192)
    return X, enc


def test_the_reading_is_the_response_per_unit_of_the_bodys_own_posture_movement():
    """The ratio: the token's standard deviation over the window (mean over numbers) over the posture's. A body whose
    encoder region passes three times the movement reads about three times the ratio at the same movement, and a body
    that moves twice as much at the same gain reads about the same ratio (the reading is per unit movement)."""
    X, enc = _gain_bodies([1.0, 3.0, 1.0], moves=[0.06, 0.06, 0.12])
    w = _fed(WithinBody(line=LINE, deaf=DEAF_T), X)
    w.lesson(enc)
    r = w.last["ratio"]
    assert r[1] / r[0] == pytest.approx(3.0, rel=0.1), r
    assert r[2] / r[0] == pytest.approx(1.0, rel=0.15), "twice the movement, the same gain: the same ratio"
    s, u = w.last["spread"], w.last["posture_spread"]
    assert all(rb == pytest.approx(sb / ub, rel=1e-5) for rb, sb, ub in zip(r, s, u))
    assert u[2] == pytest.approx(2 * u[0], rel=0.05)


def test_the_lesson_pushes_every_body_under_the_line_and_no_other_whatever_it_moves():
    """The coordinator's decision: no movement floor. A quiet body (a posture moving 0.002 per number) whose ratio is
    under the line is pushed like any other (version 4 left a body under its STILL floor alone, and version 3 pushed it
    with a reading that could not rise); a body over the line, a skipped body and a body whose posture did not change at
    all (no ratio: the division guard) are not."""
    X, enc = _gain_bodies([0.5, 20.0, 0.5, 0.5, 0.5], moves=[0.06, 0.06, 0.002, 0.06, 0.06])
    X[4] = X[4, :1]                                                       # body 5: no movement at all
    w = _fed(WithinBody(line=LINE, deaf=DEAF_T, skip="4"), X)
    loss = w.lesson(enc)
    last = w.last
    assert last["ratio"][1] > LINE > last["ratio"][0] and last["ratio"][2] < LINE, last["ratio"]
    assert last["pushed"] == [True, False, True, False, False], "body 3, quiet, is pushed; 4 is skipped; 5 has no ratio"
    assert last["below_line"] == 2 and last["no_movement"] == 1 and last["ratio"][4] == 0.0
    short = [math.log(LINE / last["ratio"][b]) if last["pushed"][b] else 0.0 for b in range(5)]
    assert last["shortfall"] == pytest.approx(short, rel=1e-5)
    assert last["lesson"] == pytest.approx(0.5 * sum(x * x for x in short) / 5, rel=1e-5)
    assert loss is not None, "bodies under the line: a lesson (this toy encoder has no weights to push)"
    X_live, enc_live = _gain_bodies([20.0, 20.0])
    over = _fed(WithinBody(line=LINE, deaf=DEAF_T), X_live)
    assert over.lesson(enc_live) is None and not any(over.last["pushed"]), "every body over the line: no lesson"
    assert min(over.last["ratio"]) > LINE
    assert set(w.gauges()) >= {"ratio", "shortfall", "spread", "posture_spread", "line", "strength", "skip", "lesson"}


def test_the_hinge_engages_a_body_under_deaf_and_releases_it_at_the_line():
    """Version 6 (the coordinator's option (b) of the third review): a body is pushed from the moment it reads under DEAF
    until it reaches LINE (one flag per body); a body between DEAF and LINE is left alone (at 344,400 bodies 1, 2, 6, 7
    and 8 read 3.21-3.63 there, and 17 of the 35 healthy pre-deploy readings); a released body is engaged again only
    under DEAF; a body with no reading is released. Version 5 pushed every body under LINE."""
    w = WithinBody(line=3.0, deaf=2.0)
    has = torch.tensor([True, True, True, True])

    def step(r, h=has):
        return w.engage(torch.tensor(r), h).tolist()
    assert step([1.0, 2.5, 3.5, 1.5]) == [True, False, False, True], "under DEAF only"
    assert step([2.5, 2.5, 3.5, 3.1]) == [True, False, False, False], "body 1 rises and stays engaged; body 4 released"
    assert step([3.1, 2.5, 2.9, 2.5]) == [False, False, False, False], "released at the line; between: left alone"
    assert step([2.5, 1.9, 2.9, 1.8]) == [False, True, False, True], "under DEAF again: engaged again"
    assert step([2.5, 1.9, 2.9, 1.8], torch.tensor([True, False, True, True])) == [False, False, False, True], \
        "a body with no reading is released"


def test_the_lesson_pushes_only_the_engaged_bodies(monkeypatch):
    """The hinge inside the lesson: of three bodies under LINE, the one under DEAF is pushed and the two between DEAF
    and LINE are not (version 5 pushed all three); the start records the engaged bodies."""
    monkeypatch.setattr(PT, "LINE_MIN", 1e-3)                  # the typing guard, set under this toy's scale
    X, enc = _gain_bodies([0.5, 1.0, 1.5, 20.0])
    probe = _fed(WithinBody(line=PT.LINE_MAX, deaf=DEAF_MAX), X)
    probe.lesson(enc)
    r = probe.last["ratio"]
    assert r[0] < r[1] < r[2] < r[3], r
    deaf, line = (r[0] + r[1]) / 2, (r[2] + r[3]) / 2
    w = _fed(WithinBody(line=line, deaf=deaf), X)
    loss = w.lesson(enc)
    assert w.last["engaged"] == [True, False, False, False] and w.last["pushed"] == [True, False, False, False]
    assert loss is not None and w.start["engaged"] == [True, False, False, False]
    assert w.pushed_ever == [True, False, False, False], "the hold's bodies: the pushed ones only"
    v5 = _fed(WithinBody(line=line, deaf=(r[2] + r[3]) / 2 - 1e-6), X)
    v5.lesson(enc)
    assert v5.last["pushed"] == [True, True, True, False], "the control: a DEAF at the line pushes every body under it"


def test_the_content_is_read_with_the_tokens_gain_divided_out():
    """The third review's MAJOR: the content at the penalty 200 pays for amplitude (a token scaled by G reads what the
    unscaled token reads at 200 / G^2), so a lift that adds movement without posture information could keep A1's content
    control. The gain-matched content reads at the penalty 200 x G^2: a pure gain reads the unscaled token's content; a
    planted token that moves with the posture and carries no linear posture information (the absolute value of the
    posture's slowest direction, its least-squares part on the posture removed, sized to triple the response) reads
    less, by more than A1's margin (2 x 0.069 / sqrt 3 = 0.080); the plain content of both rises or holds."""
    g = torch.Generator().manual_seed(11)
    n = PT.WINDOW
    P = _latent(1, n, k=4, seed=11) @ torch.randn(4, 182, generator=g) * 0.02
    P = P + 0.01 * torch.randn(1, n, 182, generator=g)
    Z0 = 3.0 + P @ (torch.randn(182, 192, generator=g) * 0.3) + 0.02 * torch.randn(1, n, 192, generator=g)
    base = float(PT.posture_content(Z0, P, lam=PT.CONTROL_LAMBDA)[0])
    s0 = float(PT.response(Z0)[0])
    amp = Z0.mean(1, keepdim=True) + 3.0 * (Z0 - Z0.mean(1, keepdim=True))
    Pc = (P - P.mean(1, keepdim=True))[0]
    y = Pc @ torch.linalg.svd(Pc, full_matrices=False)[2][-1]          # a direction with little variance
    q = y.abs()
    A = torch.cat([Pc, torch.ones(n, 1)], dim=1).double()
    q = (q.double() - A @ torch.linalg.lstsq(A, q.double()[:, None]).solution[:, 0]).float()
    q = (q - q.mean()) / q.std()
    v = torch.randn(192, generator=g)

    def planted(a):
        return Z0 + a * q[None, :, None] * v

    lo, hi = 0.0, 1.0
    while float(PT.response(planted(hi))[0]) < 3.0 * s0:
        lo, hi = hi, 2.0 * hi
    for _ in range(40):                                                 # the plant's size: the response triples
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if float(PT.response(planted(mid))[0]) < 3.0 * s0 else (lo, mid)
    plant = planted(hi)
    for Z, gain_expect in ((amp, 3.0), (plant, None)):
        G = float(PT.response(Z)[0]) / s0
        if gain_expect:
            assert G == pytest.approx(gain_expect, rel=1e-4)
        gm = float(PT.gain_matched_content(Z, P, [G])[0])
        plain = float(PT.posture_content(Z, P, lam=PT.CONTROL_LAMBDA)[0])
        if gain_expect:
            assert gm == pytest.approx(base, abs=2e-3) and plain > base, (gm, plain, base)
        else:
            assert G > 2.5 and gm < base - 0.08, ("the plant reads its gain-matched content under A1's margin", G, gm,
                                                  base)
    per_body = PT.posture_content(torch.cat([Z0, amp]), torch.cat([P, P]), lam=torch.tensor([20.0, 1800.0]))
    one = PT.posture_content(Z0, P, lam=20.0)
    two = PT.posture_content(amp, P, lam=1800.0)
    assert torch.allclose(per_body, torch.cat([one, two]), atol=1e-5), "one penalty per body"


def _capacity_toy(seed=61):
    g = torch.Generator().manual_seed(seed)
    cols = posture_columns(312)
    X = 0.3 * torch.randn(2, 1, 312, generator=g).expand(2, PT.WINDOW, 312).clone()
    X[..., cols] += _latent(2, PT.WINDOW, seed=seed) @ (0.2 * torch.randn(4, len(cols), generator=g))
    X[..., PT.capacity_columns(312)] = 0.8
    X[..., [j for j in range(312) if j % 12 == 11]] = torch.arange(26, dtype=torch.float32).repeat(1) / 25.0
    lin = torch.nn.Linear(312, 64, bias=False)
    return X, lin


def _fatigue_toy(seed=61, B=3):
    """B bodies whose postures move (4 directions) with every capacity (fatigue) input constant at 0.8 over the window
    (the ring's rebuilt rows hold it so) and the row index as the input carries it, through a small encoder with a
    LayerNorm and a GELU between two layers (the fatigue step reaches the token through the shared layers, as on H84)."""
    g = torch.Generator().manual_seed(seed)
    cols = posture_columns(312)
    X = 0.3 * torch.randn(B, 1, 312, generator=g).expand(B, PT.WINDOW, 312).clone()
    X[..., cols] += _latent(B, PT.WINDOW, seed=seed) @ (0.2 * torch.randn(4, len(cols), generator=g))
    X[..., PT.capacity_columns(312)] = 0.8
    X[..., [j for j in range(312) if j % 12 == 11]] = torch.arange(26, dtype=torch.float32) / 25.0
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(312, 64, bias=False), torch.nn.LayerNorm(64), torch.nn.GELU(),
                              torch.nn.Linear(64, 64, bias=False))
    with torch.no_grad():
        net[0].weight[:, PT.capacity_columns(312)] *= 20.0
    return X, net


def _toy_encode(net):
    return lambda xx: torch.nn.functional.layer_norm(net(xx), (64,))


def test_the_fatigue_answer_and_the_holds_line_are_read_at_the_start(monkeypatch):
    """The third review's MAJOR and the coordinator's decision of 2026-09-26 (record section 45): the gauges read the
    FATIGUE ANSWER (the token's mean absolute change per number for a capacity step of FATIGUE_STEP, over its response),
    and the start records THE HOLD'S LINE from the live bodies (the judged bodies that moved and were not engaged at the
    start; version 7: their median answer on the posture-only rows; version 6 read their largest). With no live body
    the line is unmeasured (None) and the hold does not act."""
    monkeypatch.setattr(PT, "LINE_MIN", 1e-6)                  # the typing guard, set under this toy's scale
    assert PT.FATIGUE_STEP == 0.042 and not hasattr(PT, "MASK_CAPACITY"), "version 6 chose the hold, not the mask"
    X, net = _fatigue_toy()
    enc = _toy_encode(net)
    probe = _fed(WithinBody(line=PT.LINE_MAX, deaf=DEAF_MAX), X)
    probe.lesson(enc)
    fa = probe.last["fatigue_answer"]
    with torch.no_grad():
        z = enc(X.reshape(-1, 312)).reshape(3, PT.WINDOW, -1)
        z2 = enc(PT.fatigue_stepped(X).reshape(-1, 312)).reshape(3, PT.WINDOW, -1)
        want = (z2 - z).abs().mean((1, 2)) / PT.response(z)
    assert fa == pytest.approx(want.tolist(), rel=1e-4)
    assert probe.start["live"] == [False] * 3 and probe.start["fatigue_line"] is None, \
        "every body engaged at the start: no live body, the hold's line is unmeasured"
    r = sorted(probe.last["ratio"])
    w = _fed(WithinBody(line=2.0 * r[-1], deaf=(r[0] + r[1]) / 2, skip="3"), X)
    w.lesson(enc)
    low = probe.last["ratio"].index(r[0])
    live = [b != low and b != 2 for b in range(3)]
    assert w.start["live"] == live, "judged, moved and not engaged; body 3 skipped"
    live_p = sorted(a for a, ok in zip(w.start["fatigue_answer_posture_only"], live) if ok)
    k = len(live_p) // 2
    assert w.start["fatigue_line"] == pytest.approx(live_p[k] if len(live_p) % 2 else 0.5 * (live_p[k - 1] + live_p[k]))
    assert {"fatigue_answer", "fatigue_line", "held", "pushed_ever", "hold"} <= set(w.gauges())


def test_the_holds_line_is_the_never_pushed_bodies_median_on_the_posture_only_rows_read_again_each_window(monkeypatch):
    """Version 7 (the review's finding 4; the coordinator's decision): THE HOLD'S LINE F is the MEDIAN fatigue answer on
    the POSTURE-ONLY rows (the rows the hold judges) of the judged bodies the lesson never pushed, read at the start and
    AGAIN at the first gauge step a window (6,000 model steps) after the last reading, so a lifted body is held at the
    answer the live bodies give now; with no such body moving, the line in force is kept and said so. The gauges carry
    the line in force, its step and the hold's share of the steps since the last gauge step. Here the bodies' limb speeds
    change over the window (so the posture-only rows read another answer than the real rows), body 1 is engaged, bodies
    2 and 3 are live and body 4 is skipped: F is the mean of bodies 2 and 3's posture-only answers, not version 6's
    largest answer on the real rows. Fails on 67335783f (F the largest real-row answer, read once)."""
    monkeypatch.setattr(PT, "LINE_MIN", 1e-6)                  # the typing guard, set under this toy's scale
    X, net = _fatigue_toy(seed=65, B=4)
    vel = [j for j in range(312) if j % 12 in PT.VELOCITY_FIELDS]
    X[..., vel] += 0.05 * torch.randn(4, PT.WINDOW, len(vel), generator=torch.Generator().manual_seed(66))
    enc = _toy_encode(net)
    probe = _fed(WithinBody(line=PT.LINE_MAX, deaf=DEAF_MAX), X)
    probe.lesson(enc)
    r = probe.last["ratio"][:3]
    low = r.index(min(r))
    deaf = (sorted(r)[0] + sorted(r)[1]) / 2
    w = _fed(WithinBody(line=2.0 * max(r), deaf=deaf, skip="4"), X)
    w.lesson(enc)
    live = [b != low and b != 3 for b in range(4)]
    assert w.start["live"] == live and w.start["engaged"][low]

    def posture_only_answers(e):
        Xp = PT.posture_only(X)
        with torch.no_grad():
            zp = e(Xp.reshape(-1, 312)).reshape(4, PT.WINDOW, -1)
            zpf = e(PT.fatigue_stepped(Xp).reshape(-1, 312)).reshape(4, PT.WINDOW, -1)
            return (PT.fatigue_change(zp, zpf) / PT.response(zp)).tolist()
    ans = posture_only_answers(enc)
    f0 = sum(a for a, ok in zip(ans, live) if ok) / 2.0
    assert w.start["fatigue_line"] == pytest.approx(f0, rel=1e-4) and w.fatigue_line == w.start["fatigue_line"]
    v6_line = max(a for a, ok in zip(w.start["fatigue_answer"], live) if ok)
    assert w.start["fatigue_line"] != pytest.approx(v6_line, rel=1e-3), "not version 6's line on the real rows"
    assert w.start["fatigue_answer_posture_only"][3] is None, "the skipped body is not read"
    s0 = w.fatigue_line_step
    g = w.gauges()
    assert g["fatigue_line"] == pytest.approx(f0, rel=1e-3) and g["fatigue_line_step"] == s0
    assert "hold_any_share_since_gauge" in g and g["hold_steps_since_gauge"] == 1
    # a window later the live bodies answer fatigue more (the posture weights halved: the token moves half as much with
    # the posture and as much with fatigue): F is read again from them (DEAF set under every ratio from here, so no body
    # is newly engaged and the live bodies stay never pushed)
    with torch.no_grad():
        net[0].weight[:, posture_columns(312)] *= 0.5
    w.deaf = 1e-6
    w._step = s0 + PT.WINDOW * PT.SAMPLE_EVERY - 5                  # not a gauge step a window on: F stays
    w.lesson(enc)
    assert w.fatigue_line_step == s0
    w._step = (s0 // PT.GAUGE_EVERY + 61) * PT.GAUGE_EVERY          # the first gauge step a window on
    w.lesson(enc)
    never = [ok and not pe for ok, pe in zip([b != 3 for b in range(4)], w.pushed_ever)]
    ans2 = posture_only_answers(enc)
    vals = sorted(a for a, ok in zip(ans2, never) if ok)
    f1 = vals[len(vals) // 2] if len(vals) % 2 else 0.5 * (vals[len(vals) // 2 - 1] + vals[len(vals) // 2])
    assert w.fatigue_line == pytest.approx(f1, rel=1e-4) and w.fatigue_line_step == w._step
    assert w.fatigue_line > f0, "the live bodies' answer rose: the hold's line follows it"
    assert w.start["fatigue_line"] == pytest.approx(f0, rel=1e-4), "the start keeps its own line"
    # with no judged body left unpushed, the line in force is kept and the gauges say so
    kept = _fed(WithinBody(line=PT.LINE_MAX, deaf=DEAF_MAX, skip="4"), X)
    kept.lesson(enc)
    assert kept.fatigue_line is None and all(kept.pushed_ever[:3])
    kept._step = (kept.fatigue_line_step // PT.GAUGE_EVERY + 61) * PT.GAUGE_EVERY
    kept.lesson(enc)
    assert kept.fatigue_line is None and "kept" in kept.last["fatigue_line_note"]


def test_the_holds_line_read_again_leaves_out_a_body_released_at_the_line(monkeypatch):
    """The re-review's finding 6 (a planted defect no test caught: F read again from the bodies "not engaged now"
    instead of the bodies "never pushed"; in the test above the pushed body is never released, so the two sets
    coincide). A body the lesson pushed and then released at LINE is not engaged but was pushed: it must not enter F,
    or a lifted body would set its own hold's line. Here body 1 is pushed at the start, released at LINE before the
    reading a window later, and F read again is the median of bodies 2 and 3 (never pushed) alone."""
    monkeypatch.setattr(PT, "LINE_MIN", 1e-6)                  # the typing guard, set under this toy's scale
    X, net = _fatigue_toy(seed=65, B=4)
    vel = [j for j in range(312) if j % 12 in PT.VELOCITY_FIELDS]
    X[..., vel] += 0.05 * torch.randn(4, PT.WINDOW, len(vel), generator=torch.Generator().manual_seed(66))
    enc = _toy_encode(net)
    probe = _fed(WithinBody(line=PT.LINE_MAX, deaf=DEAF_MAX), X)
    probe.lesson(enc)
    r = probe.last["ratio"][:3]
    low = r.index(min(r))
    w = _fed(WithinBody(line=2.0 * max(r), deaf=(sorted(r)[0] + sorted(r)[1]) / 2, skip="4"), X)
    w.lesson(enc)
    assert w.start["engaged"][low] and w.pushed_ever[low], "the low body is pushed at the start"
    # a window later the low body reads at LINE (the line set just under its ratio) and is released; DEAF under every
    # ratio so no body is engaged anew
    w.deaf, w.line = 1e-6, 0.99 * r[low]
    w._step = (w.fatigue_line_step // PT.GAUGE_EVERY + 61) * PT.GAUGE_EVERY
    w.lesson(enc)
    assert w.engaged[low] is False and w.pushed_ever[low] is True, "released at the line, pushed before"
    never = [b for b in range(3) if b != low]
    ans = w.last["fatigue_answer_posture_only"]
    want = sum(ans[b] for b in never) / 2.0
    wrong = sorted(ans[b] for b in range(3))[1]                    # the median with the released body in it
    assert abs(want - wrong) > 1e-3, ("the fixture must tell the two sets apart", ans)
    assert w.fatigue_line == pytest.approx(want, rel=1e-6) and w.fatigue_line_step == w._step


def test_the_hold_acts_only_on_pushed_bodies_and_its_gradient_lowers_their_fatigue_change(monkeypatch):
    """THE HOLD (proprio_token.py; the coordinator's decision): from a body's first push on, also after it is released,
    the lesson adds relu(ln answer_b - ln F) x grad ln f_b / B for a body whose answer on the posture-only rows is over
    the hold's line F (f_b: the token's mean absolute change for the fatigue step, the answer's numerator); never for a
    body the lesson never pushed, and not at all while F is unmeasured. Here no body is engaged (every ratio over LINE),
    F sits under every answer and body 1 has been pushed before: the lesson is the hold on body 1 alone, equal to the
    gradient computed by hand, and one small step along it lowers body 1's fatigue change."""
    monkeypatch.setattr(PT, "LINE_MIN", 1e-6)
    X, net = _fatigue_toy(seed=63)
    enc = _toy_encode(net)
    params = list(net.parameters())

    def fresh(pushed, f_line):
        w = _fed(WithinBody(line=2e-6, deaf=1e-6), X)
        w.start = dict(ratio=[1.0] * 3, fatigue_line=f_line)
        w.pushed_ever = list(pushed)
        return w
    probe = fresh([False] * 3, None)
    assert probe.lesson(enc) is None and probe.last["held"] == [False] * 3, "no push, no line: nothing acts"
    Xp = PT.posture_only(X)
    with torch.no_grad():
        zp = enc(Xp.reshape(-1, 312)).reshape(3, PT.WINDOW, -1)
        zpf = enc(PT.fatigue_stepped(Xp).reshape(-1, 312)).reshape(3, PT.WINDOW, -1)
        answers = (PT.fatigue_change(zp, zpf) / PT.response(zp)).tolist()
    F = 0.5 * min(answers)
    w = fresh([True, False, False], F)
    loss = w.lesson(enc)
    assert loss is not None and w.last["pushed"] == [False] * 3 and w.last["engaged"] == [False] * 3
    assert w.last["held"] == [True, False, False], "bodies 2 and 3 are over the line too but were never pushed"
    assert w.last["fatigue_answer_posture_only"][0] == pytest.approx(answers[0], rel=1e-4)
    assert w.last["fatigue_answer_posture_only"][1:] == [None, None] and w.last["fatigue_line"] == F
    over = math.log(answers[0] / F)
    assert w.last["hold"] == pytest.approx(0.5 * over ** 2 / 3, rel=1e-4)
    got = torch.autograd.grad(loss, params, allow_unused=True)
    z1 = enc(Xp[0])
    f1 = (enc(PT.fatigue_stepped(Xp[0])) - z1).abs().mean()
    want = torch.autograd.grad(over * torch.log(f1) / 3, params, allow_unused=True)
    for a, b in zip(got, want):
        assert (a is None) == (b is None)
        if a is not None:
            assert float((a - b).norm()) <= 1e-4 * float(b.norm()), (float((a - b).norm()), float(b.norm()))
    with torch.no_grad():
        f_before = float(f1)
        rms = math.sqrt(sum(float(g.square().sum()) for g in got if g is not None))
        for p, g in zip(params, got):
            if g is not None:
                p.sub_(1e-3 / rms * g)
        f_after = float((enc(PT.fatigue_stepped(Xp[0])) - enc(Xp[0])).abs().mean())
    assert f_after < f_before, (f_before, f_after)
    released = fresh([True, True, False], F)
    assert released.lesson(enc) is not None and released.last["held"] == [True, True, False], \
        "a pushed body stays held after it is released (no body is engaged here)"
    waiting = fresh([True, True, True], None)
    assert waiting.lesson(enc) is None and waiting.last["held"] == [False] * 3, "no live body at the start: no hold"


def test_the_pushed_flags_follow_the_pushes():
    """pushed_ever records every body the lesson has pushed since the start and a release never clears it (the
    coordinator's limit: the hold acts only on bodies the lesson has pushed, from the first push on). Body 1 is pushed,
    then reads over the line (the same rows through a livelier encoder) and is released: its flag stays; body 2 was
    never pushed and body 3 is skipped."""
    X, enc = _gain_bodies([0.5, 20.0, 0.5])
    w = _fed(WithinBody(line=LINE, deaf=DEAF_T, skip="3"), X)
    w.lesson(enc)
    assert w.last["pushed"] == [True, False, False] and w.pushed_ever == [True, False, False]
    X2, lively = _gain_bodies([20.0, 20.0, 0.5])
    assert torch.equal(X2, X)
    w.lesson(lively)
    assert w.last["engaged"] == [False, False, False] and w.last["pushed"] == [False, False, False]
    assert w.pushed_ever == [True, False, False], "released, still one of the hold's bodies"


def test_the_gauges_read_the_jitter_shares_of_the_token_and_the_posture_by_one_formula():
    """K2 (version 6) judges the token's jitter share minus the posture's own: the lesson reads both on its window by
    proprio_token.jitter_share, the formula of the trace (proprio_mechanism/jitter_share.share: per number, the mean
    squared change between samples over twice the variance); a gain leaves the token's share as it was."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "jitter_share_trace", Path(__file__).resolve().parents[4] / "docs" / "gain_of_function_v4" /
        "proprio_mechanism" / "jitter_share.py")
    X, enc = _gain_bodies([1.0, 3.0])
    w = _fed(WithinBody(line=PT.LINE_MAX, deaf=DEAF_MAX), X)
    w.lesson(enc)
    z = enc(X.reshape(-1, 312)).reshape(2, PT.WINDOW, -1)
    P = X[..., posture_columns(312)]
    assert w.last["jitter_share"] == pytest.approx(PT.jitter_share(z).tolist(), rel=1e-5)
    assert w.last["posture_jitter_share"] == pytest.approx(PT.jitter_share(P).tolist(), rel=1e-5)
    assert w.last["jitter_share"][0] == pytest.approx(w.last["jitter_share"][1], rel=0.05), "a gain leaves the share"
    if spec is not None and spec.loader is not None:
        src = Path(spec.origin).read_text(encoding="utf-8")
        assert "d = (x[:, 1:] - x[:, :-1]).square().mean(1)" in src and "(d / (2 * v)).mean(-1)" in src, \
            "the trace's formula is the lesson's"


def test_the_held_out_reading_reads_the_window_through_the_encoder_saved_one_window_earlier():
    """The third review's SEMANTIC: the success metric read the lesson's own objective on its training window. The
    lesson saves the encoder at its start and, at the first gauge step a whole window later, reads the window (all of it
    newer than that encoder) through the saved encoder, then saves the encoder again. Checked: the held-out ratio is the
    ratio through the saved weights, not the moved ones; the reading and the saved encoder ride runtime_state.pt."""
    X, lin = _capacity_toy(seed=71)

    def enc(xx):
        return torch.nn.functional.layer_norm(lin(xx), (64,))

    def enc_with(state):
        return lambda xx: torch.nn.functional.layer_norm(torch.func.functional_call(lin, state, (xx,)), (64,))

    def now():
        return {k: v.detach().clone() for k, v in lin.state_dict().items()}
    w = _fed(WithinBody(line=PT.LINE_MAX, deaf=DEAF_MAX), X)
    w._step = 0
    w.lesson(enc, enc_with, now)
    assert w.snapshot is not None and w.snapshot_step == 0 and w.held_out is None
    saved = now()
    with torch.no_grad():
        lin.weight[:, posture_columns(312)] *= 3.0
    w._step = 100
    w.lesson(enc, enc_with, now)
    assert w.held_out is None, "less than a window since the encoder was saved"
    w._step = PT.WINDOW * PT.SAMPLE_EVERY
    w.lesson(enc, enc_with, now)
    ho = w.held_out
    with torch.no_grad():
        z_old = enc_with(saved)(X.reshape(-1, 312)).reshape(2, PT.WINDOW, -1)
        u = PT.movement(X[..., posture_columns(312)])
        want = PT.ratio_of(PT.response(z_old), u).tolist()
    assert ho["encoder_step"] == 0 and ho["step"] == PT.WINDOW * PT.SAMPLE_EVERY
    assert ho["ratio"] == pytest.approx(want, rel=1e-4)
    assert ho["ratio"][0] != pytest.approx(w.last["ratio"][0], rel=1e-3), "not the moved encoder's reading"
    assert w.snapshot_step == PT.WINDOW * PT.SAMPLE_EVERY and w.last["held_out"] == ho
    v = WithinBody(line=PT.LINE_MAX, deaf=DEAF_MAX)
    v.load_state_dict(w.state_dict(), "cpu")
    assert v.held_out == ho and v.snapshot_step == w.snapshot_step
    assert all(torch.equal(v.snapshot[k], w.snapshot[k]) for k in w.snapshot)


def test_the_push_goes_through_the_posture_only_rows_and_leaves_the_velocity_weights_alone():
    """The velocity (limb speed) weights of H84's first layer move 6.8-13.5 times further per update than their tensor's
    average under Adam (record section 22), so a push on the plain response could raise it through the velocity inputs.
    The lesson's gradient reaches the encoder through the posture-only rows (velocity and capacity at their window
    mean): a velocity whose window mean is 0 (a body that ends where it began) gives the velocity weights of a linear
    encoder no gradient at all; the control, the same shortfall pushed through the plain rows, does move them. On H84
    (the pace job, 415,075, where a body's velocity averages near 0 over 300 s but not exactly): the lesson's step on the
    velocity columns had a median of 0.0001 learning rates against 0.0105 on the quaternion columns."""
    g = torch.Generator().manual_seed(7)
    cols = posture_columns(312)
    vel = [j for j in range(312) if j % 12 in PT.VELOCITY_FIELDS]
    X = 0.3 * torch.randn(2, 1, 312, generator=g).expand(2, PT.WINDOW, 312).clone()
    X[..., cols] += _latent(2, PT.WINDOW, seed=7) @ (0.05 * torch.randn(4, len(cols), generator=g))
    jitter = 0.02 * torch.randn(2, PT.WINDOW, len(vel), generator=g)                # the limbs' speeds jitter
    X[..., vel] = jitter - jitter.mean(1, keepdim=True)
    lin = torch.nn.Linear(312, 192, bias=False)

    def enc(xx):
        return torch.nn.functional.layer_norm(lin(xx), (192,))

    def velocity_rms(grad):
        grad = grad / grad.square().mean().sqrt()                                   # each push at one size
        return float(grad[:, vel].square().mean().sqrt())
    w = _fed(WithinBody(line=PT.LINE_MAX, deaf=DEAF_MAX), X)
    loss = w.lesson(enc)
    lesson = velocity_rms(torch.autograd.grad(loss, [lin.weight])[0])
    s_plain = PT.response(enc(X.reshape(-1, 312)).reshape(2, PT.WINDOW, -1))
    control = velocity_rms(torch.autograd.grad(-torch.log(s_plain).sum(), [lin.weight])[0])
    assert control > 0.01, "the control: a push on the plain response reaches the velocity weights"
    assert lesson < 1e-5 * control, (lesson, control)


def _adam_with_state(params, v=1e-4):
    opt = torch.optim.Adam(params, lr=1e-3)
    for p in params:
        opt.state[p] = dict(step=torch.tensor(10.0), exp_avg=torch.zeros_like(p),
                            exp_avg_sq=torch.full_like(p, v))
    return opt


def test_the_gradient_is_added_after_the_clip_at_its_share_of_the_running_size():
    # float64: in float32 a 0.002 added to 1,000 keeps about three of its digits (measured 2026-09-26)
    lin = torch.nn.Linear(4, 3).double()
    params = list(lin.parameters())
    v = 1e-4
    opt = _adam_with_state(params, v)
    bc = 1 - 0.999 ** 10
    size = (v / bc) ** 0.5
    for other in (1e-6, 1e3):                  # whatever the other (clipped) gradient's size
        for p in params:
            p.grad = torch.full_like(p, other)
        w = WithinBody(line=LINE, deaf=DEAF_T, strength=0.02)
        lesson = lin(torch.randn(5, 4, dtype=torch.float64)).square().mean()
        before = [p.grad.clone() for p in params]
        out = w.apply(lesson, params, opt)
        assert out["tensors"] == 2 and out["skipped"] == 0
        for p, b in zip(params, before):
            added = p.grad - b
            assert float(added.square().mean().sqrt()) == pytest.approx(0.02 * size, rel=1e-5)
    fresh = torch.nn.Linear(4, 3)
    opt2 = torch.optim.Adam(fresh.parameters(), lr=1e-3)
    out = WithinBody(line=LINE, deaf=DEAF_T).apply(fresh(torch.randn(5, 4)).square().mean(), list(fresh.parameters()), opt2)
    assert out["tensors"] == 0 and out["skipped"] == 2, "a tensor with no optimizer state yet is left out"


def _deaf_toy(seed=41):
    """Three bodies whose postures move (4 directions each) beside a large constant through one shared linear encoder
    whose posture weights are small: every body is deaf, the lesson has something to push on each."""
    g = torch.Generator().manual_seed(seed)
    cols = posture_columns(312)
    X = 0.3 * torch.randn(3, 1, 312, generator=g).expand(3, PT.WINDOW, 312).clone()
    X[..., cols] += _latent(3, PT.WINDOW, seed=seed) @ (0.2 * torch.randn(4, len(cols), generator=g))
    X += 0.02 * torch.randn(3, PT.WINDOW, 312, generator=g)
    enc = torch.nn.Linear(312, 192, bias=False).double()
    with torch.no_grad():
        enc.weight.copy_(torch.randn(192, 312, generator=g).double() * 0.05)
        enc.weight[:, cols] *= 0.02
    return X.double(), enc


def test_the_attribution_is_each_bodys_first_order_change_of_ln_ratio_from_the_lessons_step():
    """The gauges' lesson_pace (the watcher's K6 and A1 read its running sum, 'attributed'): each body's first-order
    change of ln ratio from the lesson's own step through Adam's second moment. Checked against that step taken by hand
    at a small size, for every body; the control: doubling the learning rate doubles it."""
    X, enc = _deaf_toy()
    params = list(enc.parameters())

    def encode(xx):
        return torch.nn.functional.layer_norm(enc(xx.double()), (192,))
    paces = {}
    for lr in (1e-3, 2e-3):
        opt = torch.optim.Adam(params, lr=lr)
        g = torch.Generator().manual_seed(5)
        v = (1e-4 * (0.2 + torch.rand(params[0].shape, generator=g))).double()
        opt.state[params[0]] = dict(step=torch.tensor(50.0), exp_avg=torch.zeros_like(params[0]), exp_avg_sq=v)
        w = _fed(WithinBody(line=PT.LINE_MAX, deaf=DEAF_MAX), X)
        loss = w.lesson(encode)
        for p in params:
            p.grad = torch.zeros_like(p)
        w.apply(loss, params, opt)
        paces[lr] = list(w.last["lesson_pace"])
        assert w.attributed == pytest.approx(paces[lr]), "one update since the start: the sum is the pace"
        if lr == 1e-3:
            size = (v / (1 - 0.999 ** 50)).sqrt()
            step = -lr * params[0].grad.detach() / (size + 1e-8)
            with torch.no_grad():
                def ln_ratio():
                    z = encode(X.reshape(-1, 312)).reshape(3, PT.WINDOW, -1)
                    return torch.log(PT.response(z))
                base = ln_ratio()
                k = 2e-3 / max(abs(x) for x in paces[lr])
                params[0].add_(k * step)
                moved = ln_ratio()
                params[0].sub_(k * step)
            for b in range(3):
                assert float(moved[b] - base[b]) / k == pytest.approx(paces[lr][b], rel=0.1), b
    assert all(p > 0 for p in paces[1e-3]), "the lesson lifts every pushed body"
    assert [b / a for a, b in zip(paces[1e-3], paces[2e-3])] == pytest.approx([2.0] * 3, rel=1e-6)


def test_the_content_control_solves_eight_bodies_at_the_jobs_two_threads():
    """The content control (version 4's reading, on gauge steps and in the calibration) solves eight 192 x 192 systems.
    The measurement jobs run torch on 2 threads, where the batched float32 LU solve failed on this machine's MKL
    (measured 2026-09-26); the penalised Gram matrix is symmetric positive definite, so the reading solves it by
    Cholesky. Run in its own process so the thread count is the jobs' own (a second process: the unit-test pause job
    holds it, light runs leave it out)."""
    import subprocess
    import sys
    code = ("import torch; torch.set_num_threads(2)\n"
            "from mimo_bridge import proprio_token as PT\n"
            "g = torch.Generator().manual_seed(0)\n"
            "Z = (0.3 * torch.randn(8, 1200, 192, generator=g) + 1.0).requires_grad_(True)\n"
            "P = torch.cumsum(0.01 * torch.randn(8, 1200, 182, generator=g), 1)\n"
            "r = PT.posture_content(Z, P, lam=PT.CONTROL_LAMBDA)\n"
            "r.sum().backward()\n"
            "print('ok', bool(torch.isfinite(r).all()), bool(torch.isfinite(Z.grad).all()))\n")
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=here, timeout=600)
    assert out.returncode == 0 and "ok True True" in out.stdout, (out.stdout[-300:], out.stderr[-600:])


def test_the_gauges_read_the_posture_only_ratio_and_the_content_control_on_the_gauge_step():
    X, enc = _gain_bodies([0.5, 5.0])
    w = _fed(WithinBody(line=LINE, deaf=DEAF_T), X)
    w._step = 7
    w.lesson(enc)                                           # the start forces the gauges
    assert len(w.last["ratio_posture_only"]) == 2 and len(w.last["content"]) == 2 and w.last["gauge_step"] == 7
    got = w.last["content"]
    P = X[..., posture_columns(312)]
    want = posture_content(enc(X.reshape(-1, 312)).reshape(2, PT.WINDOW, -1), P, lam=PT.CONTROL_LAMBDA)
    assert got == pytest.approx(want.tolist(), abs=1e-5), "version 4's reading at the content control's penalty"
    w._step = 8
    w.lesson(enc)
    assert w.last["gauge_step"] == 7, "not a gauge step: the last gauges are carried, not read again"


def test_the_start_is_read_once_before_the_first_update_and_rides_the_checkpoint():
    X, enc = _deaf_toy(seed=43)
    params = list(enc.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    for p in params:
        opt.state[p] = dict(step=torch.tensor(10.0), exp_avg=torch.zeros_like(p), exp_avg_sq=torch.full_like(p, 1e-4))

    def encode(xx):
        return torch.nn.functional.layer_norm(enc(xx.double()), (192,))
    w = _fed(WithinBody(line=PT.LINE_MAX, deaf=DEAF_MAX), X)
    w._step = 7
    loss = w.lesson(encode)
    start = dict(w.start)
    assert start["step"] == 7 and start["ratio"] == w.last["ratio"] and len(start["ratio_posture_only"]) == 3
    assert start["pushed"] == [True, True, True] and len(start["content"]) == 3 and w.attributed == [0.0] * 3
    for p in params:
        p.grad = torch.zeros_like(p)
    w.apply(loss, params, opt)
    with torch.no_grad():
        opt.step()
    w._step = 100
    w.lesson(encode)
    assert w.last["ratio"] != start["ratio"] and w.start == start, "read once"
    sd = w.state_dict()
    w.reset()
    assert w.start == start and w.ring is None, "a ring reset keeps the deploy's start"
    v = WithinBody(line=PT.LINE_MAX, deaf=DEAF_MAX)
    assert "the start, the attributed lift, the engaged and pushed flags" in v.load_state_dict(sd, "cpu")
    assert v.start == start and v.attributed == w.attributed
    assert v.pushed_ever == w.pushed_ever == [True] * 3, "the pushed flags ride the checkpoint (the hold's bodies)"
    # a version-4 window keeps its rows (the same raw inputs) but its start read another quantity (an R2)
    old = dict(sd, version="proprio_token_repair_v4", start=dict(step=3, r2=[0.1] * 3), attributed=[0.01] * 3)
    u = WithinBody(line=PT.LINE_MAX, deaf=DEAF_MAX)
    msg = u.load_state_dict(old, "cpu")
    assert "restored" in msg and "not this version's reading" in msg
    assert u.start is None and u.attributed is None and u.count == sd["count"] == PT.WINDOW
    v5 = dict(sd, version="proprio_token_repair_v5", engaged=[True] * 3)
    t = WithinBody(line=PT.LINE_MAX, deaf=DEAF_MAX)
    assert "not this version's reading" in t.load_state_dict(v5, "cpu") and t.engaged is None, \
        "version 5 had no hinge: its start and flags are not version 6's"
    assert v.engaged == w.engaged == [True] * 3, "the engaged flags ride the checkpoint"
    # version 7: the hold's line in force and its step ride the checkpoint; a version-6 start (its line read once, the
    # largest answer on the plain rows) is not restored
    assert sd["version"] == "proprio_token_repair_v7" and "fatigue_line" in sd and "fatigue_line_step" in sd
    assert v.fatigue_line == w.fatigue_line and v.fatigue_line_step == w.fatigue_line_step
    v6 = dict(sd, version="proprio_token_repair_v6", fatigue_line=0.2, fatigue_line_step=7)
    s6 = WithinBody(line=PT.LINE_MAX, deaf=DEAF_MAX)
    assert "not this version's reading" in s6.load_state_dict(v6, "cpu") and s6.start is None
    assert s6.fatigue_line is None and s6.count == PT.WINDOW, "the window's rows are kept"


def test_the_lessons_step_is_reported_per_weight_in_learning_rates_per_input_field():
    """Review 2 SEMANTIC: apply rescales per TENSOR and Adam divides per WEIGHT, so a weight whose production gradient
    runs k times under its tensor's typical size moves k times the tensor-average step. With the position columns' running
    size set 8 times under the quaternion's, they move 8 times as far; the control, a uniform second moment."""
    g = torch.Generator().manual_seed(51)
    X = torch.randn(2, PT.WINDOW, 312, generator=g) * 0.3
    X[..., posture_columns(312)] += _latent(2, PT.WINDOW, seed=51) @ (0.3 * torch.randn(4, 182, generator=g))
    pos = [j for j in range(312) if j % 12 in PT.INPUT_FIELDS["position"]]
    ratios = {}
    for name, slow in (("position 8 times under", 8.0), ("uniform", 1.0)):
        torch.manual_seed(0)
        enc = torch.nn.Linear(312, 192, bias=False)
        params = list(enc.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        v = torch.full_like(params[0], 1e-6)
        v[:, pos] /= slow ** 2
        opt.state[params[0]] = dict(step=torch.tensor(1e6), exp_avg=torch.zeros_like(params[0]), exp_avg_sq=v)
        w = _fed(WithinBody(line=PT.LINE_MAX, deaf=DEAF_MAX, strength=0.02), X)
        loss = w.lesson(lambda xx: torch.nn.functional.layer_norm(enc(xx), (192,)))
        params[0].grad = torch.zeros_like(params[0])
        got = w.apply(loss, params, opt)
        f = got["first_layer_step_lr"]
        assert set(f) == set(PT.INPUT_FIELDS) and got["step_lr_rms"] > 0 and got["step_lr_max"] >= got["step_lr_rms"]
        ratios[name] = f["position"]["rms"] / f["quaternion"]["rms"]
        expect = (params[0].grad / (v.sqrt() + 1e-8)).abs()
        assert got["step_lr_max"] == pytest.approx(float(expect.max()), rel=1e-4)
    assert ratios["position 8 times under"] / ratios["uniform"] == pytest.approx(8.0, rel=1e-3)


def _toy_run(with_lesson, updates=150, seed=0):
    """A toy of a deaf encoder: two bodies whose inputs are a large constant per body, a posture that moves in 4
    directions and jitter in the velocity numbers; the encoder starts nearly blind to the posture numbers, so its
    response per unit movement is small. A stasis pressure (consecutive tokens pulled together, as the next-moment
    lesson's residual path pulls) is clipped as production clips; the lesson, when on, is added after the clip.
    Returns each body's (ratio, posture content at CONTROL_LAMBDA) after `updates`, and their values before."""
    torch.manual_seed(seed)
    enc = torch.nn.Linear(312, 192, bias=False)
    cols = posture_columns(312)
    with torch.no_grad():
        enc.weight[:, cols] *= 0.01
    g = torch.Generator().manual_seed(seed + 7)
    X = 3.0 * torch.randn(2, 1, 312, generator=g).expand(2, PT.WINDOW, 312).clone()
    X[..., cols] += _latent(2, PT.WINDOW, seed=seed + 8) @ (0.3 * torch.randn(4, len(cols), generator=g))
    vel = [j for j in range(312) if j % 12 in PT.VELOCITY_FIELDS]
    X[..., vel] += 0.05 * torch.randn(2, PT.WINDOW, len(vel), generator=g)
    opt = torch.optim.Adam(enc.parameters(), lr=3e-3)
    w = _fed(WithinBody(line=PT.LINE_MAX, deaf=DEAF_MAX, strength=PT.STRENGTH_MAX), X)

    def encode(xx):
        return torch.nn.functional.layer_norm(enc(xx), (192,))

    def reading():
        with torch.no_grad():
            z = encode(X.reshape(-1, 312)).reshape(2, PT.WINDOW, 192)
            u = PT.movement(X[..., cols])
            return (PT.ratio_of(PT.response(z), u).tolist(),
                    posture_content(z, X[..., cols], lam=PT.CONTROL_LAMBDA).tolist())
    before = reading()
    for t in range(updates):
        opt.zero_grad()
        z = encode(X.reshape(-1, 312)).reshape(2, PT.WINDOW, 192)
        stasis = (z[:, 1:] - z[:, :-1]).pow(2).mean() * 50.0
        w._step = 5 * t
        lesson = w.lesson(encode) if with_lesson else None
        stasis.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        if lesson is not None:
            w.apply(lesson, list(enc.parameters()), opt)
        opt.step()
    return before, reading()


def test_the_lesson_lifts_a_deaf_encoders_response_per_unit_movement_and_its_posture_content_rises_with_it():
    (r0, c0), (off_r, off_c) = _toy_run(with_lesson=False)
    (_r0, _c0), (on_r, on_c) = _toy_run(with_lesson=True)
    assert all(b > 3 * a for a, b in zip(off_r, on_r)), (off_r, on_r)
    assert all(b >= a for a, b in zip(r0, on_r)) and max(off_r) < max(r0) * 1.5, "the stasis pressure alone lifts none"
    assert all(b > a for a, b in zip(off_c, on_c)), ("the lift follows the posture: the content rises", off_c, on_c)


def test_the_gradient_reaches_only_the_encoder_it_reencodes():
    torch.manual_seed(0)
    from lem.model.encoders.mlp_project import MLPProjectEncoder
    enc = MLPProjectEncoder(312, [64, 64], 192, "gelu", 0.0, False, "layernorm")
    other = torch.nn.Linear(4, 4)
    X, _enc = _gain_bodies([0.3, 0.3, 0.3])
    w = _fed(WithinBody(line=PT.LINE_MAX, deaf=DEAF_MAX), X)
    loss = w.lesson(lambda xx: torch.nn.functional.layer_norm(enc(xx).squeeze(1), (192,)))
    loss.backward()
    assert all(p.grad is not None for p in enc.parameters())
    assert all(p.grad is None for p in other.parameters())
    assert w.ring.grad is None and not w.ring.requires_grad, "the inputs are data"


def test_the_heading_free_rows_reach_the_movement():
    w = WithinBody(line=LINE, deaf=DEAF_T)
    X, enc = _gain_bodies([1.0, 1.0])
    _fed(w, X)
    seen = []
    w.rows_transform = lambda rows: (seen.append(rows.shape), rows * 0.5)[1]
    w.lesson(enc)
    assert seen and seen[0] == (2, PT.WINDOW, 312), "the posture is read from the rows the encoder reads"
    plain = _fed(WithinBody(line=LINE, deaf=DEAF_T), X)
    plain.lesson(enc)
    assert w.last["posture_spread"] == pytest.approx([0.5 * u for u in plain.last["posture_spread"]], rel=1e-5)
    assert w.gauges()["heading_free_rows"] is True


# ---------------------------------------------------------------- the saved window
def test_the_window_round_trips_and_a_foreign_or_empty_state_starts_empty():
    w = WithinBody(line=LINE, deaf=DEAF_T)
    for i in range(12):
        w.observe(torch.randn(8, 312), 5 * i, real=True)
    sd = w.state_dict()
    v = WithinBody(line=LINE, deaf=DEAF_T)
    msg = v.load_state_dict(sd, torch.device("cpu"))
    assert "restored" in msg and v.count == 12 and v.ptr == 12 and torch.equal(v.ring, w.ring)
    assert v.gauges()["restored_from"] == "runtime_state.pt"
    assert "restored" in v.load_state_dict(dict(sd, version="proprio_token_repair_v3"), "cpu")
    assert "another shape" in v.load_state_dict(dict(sd, version="proprio_token_repair_v2",
                                                      ring=torch.zeros(8, 200, 312)), "cpu")
    assert "empty" in v.load_state_dict(None, "cpu") and v.ring is None and v.count == 0
    assert "empty" in v.load_state_dict(dict(sd, version="other"), "cpu")
    assert "empty" in v.load_state_dict(dict(sd, ring=torch.zeros(8, 7, 312)), "cpu")


def test_reset_forgets_the_window():
    w = WithinBody(line=LINE, deaf=DEAF_T)
    for i in range(50):
        w.observe(torch.randn(2, 312), 5 * i, real=True)
    w.reset()
    assert w.ring is None and w.count == 0 and w.window() is None


# ---------------------------------------------------------------- the model
def _lite(**kw):
    torch.set_num_threads(1)
    from mimo_bridge.lem_model import LemLite
    return LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                   body_layout="trunk_v1", **kw)


def test_the_default_build_is_todays_model():
    from mimo_bridge import continuation_state as CS
    off = _lite()
    on = _lite(**FLAG, proprio_token_skip="3")
    assert off.proprio_within is None and off.proprio_token_repair == "off"
    assert "proprio_token_repair" not in off.receipt
    assert on.n_params == off.n_params and len(on.params) == len(off.params), "no parameter"
    st = on.receipt["proprio_token_repair"]
    assert st["mode"] == "within_body" and st["line"] == LINE and st["strength"] == PT.STRENGTH
    assert st["deaf"] == DEAF and "fatigue answer held" in st["hold"] and "mask_capacity" not in st
    assert st["window"] == PT.WINDOW and st["version"] == PT.VERSION and st["skip"] == [3]
    assert set(CS.capture_learner(on, include_replay=False)["data"]) == \
        set(CS.capture_learner(off, include_replay=False)["data"]), "the window is public"
    with pytest.raises(ValueError, match="proprio-token-repair"):
        _lite(proprio_token_repair="rescale")
    for kw in (dict(proprio_token_strength=0.05), dict(proprio_token_line=LINE), dict(proprio_token_skip="3"),
               dict(proprio_token_deaf=DEAF)):
        with pytest.raises(ValueError, match="need --proprio-token-repair within_body"):
            _lite(**kw)
    with pytest.raises(ValueError, match="unmeasured"):
        _lite(proprio_token_repair="within_body")
    with pytest.raises(ValueError, match="deaf on the response per unit movement is unmeasured"):
        _lite(proprio_token_repair="within_body", proprio_token_line=LINE)
    with pytest.raises(ValueError, match="mistyping"):
        _lite(proprio_token_repair="within_body", proprio_token_line=0.32)
    with pytest.raises(TypeError):
        _lite(**FLAG, proprio_token_lambda=200.0)
    assert type(off).lesson_probe is None and off.lesson_probe is None, "no probe in a run"


def test_the_window_rides_the_checkpoint(tmp_path):
    m = _lite(**FLAG, checkpoint_dir=str(tmp_path))
    for i in range(9):
        m.proprio_within.observe(torch.randn(8, 312), 5 * i, real=True)
    path = m.save_checkpoint(3)
    rt = torch.load(os.path.join(path, "runtime_state.pt"), weights_only=False)
    assert rt["proprio_token_repair"]["count"] == 9 and rt["proprio_token_repair"]["version"] == PT.VERSION
    other = _lite(**FLAG)
    other.load_checkpoint(path)
    assert other.proprio_within.count == 9 and torch.equal(other.proprio_within.ring, m.proprio_within.ring)
    bare = _lite()
    bare.load_checkpoint(path)
    assert bare.proprio_within is None
    plain = _lite(checkpoint_dir=str(tmp_path / "plain"))
    p2 = plain.save_checkpoint(4)
    assert "proprio_token_repair" not in torch.load(os.path.join(p2, "runtime_state.pt"), weights_only=False)
    fresh = _lite(**FLAG)
    fresh.load_checkpoint(p2)
    assert fresh.proprio_within.count == 0


def test_a_reset_of_the_transition_state_forgets_the_window():
    m = _lite(**FLAG)
    for i in range(45):
        m.proprio_within.observe(torch.randn(8, 312), 5 * i, real=True)
    m.reset_transition_state()
    assert m.proprio_within.count == 0


def test_the_lesson_reaches_only_the_proprioception_encoder_through_the_registrys_path():
    m = _lite(**dict(FLAG, proprio_token_line=PT.LINE_MAX, proprio_token_deaf=DEAF_MAX))
    g = torch.Generator().manual_seed(3)
    base = 5.0 * torch.randn(1, 312, generator=g)      # a large constant, a small posture change
    walk = torch.zeros(8, 312)
    for i in range(PT.WINDOW):
        walk = walk + 0.01 * torch.randn(8, 312, generator=g)
        m.proprio_within.observe(base + walk, 5 * i, real=True)
    m.step_count = 1                                   # not a sampling step: the window is unchanged
    frame = {"proprioception": torch.randn(8, 312)}
    loss = m._proprio_within_lesson(frame, {m_: "real" for m_ in m.active}, train=True)
    assert loss is not None and math.isfinite(float(loss))
    grads = torch.autograd.grad(loss, m.params, allow_unused=True)
    prop = {id(p) for p in m.encoder_registry.encoders["proprioception"].parameters()}
    got = [p for p, gr in zip(m.params, grads) if gr is not None and float(gr.abs().sum()) > 0]
    assert got and all(id(p) in prop for p in got)
    assert m.proprio_within.snapshot is not None, "the registry's path saves the encoder for the held-out reading"
    m.step_count = 5
    assert m._proprio_within_lesson(frame, {m_: "real" for m_ in m.active}, train=False) is None
    assert m.proprio_within.count == PT.WINDOW


def test_the_hold_and_the_held_out_reading_run_through_the_registrys_path():
    """On the learner itself (lem_model._proprio_within_lesson): the hold's fatigue-stepped rows go through the same
    registry path as the push, and its gradient reaches the proprioception encoder and nothing else; the held-out
    reading reads the window through the encoder saved at the start (torch.func.functional_call on the registry's own
    module), not through the moved encoder. Here every body is engaged at the start (the line under every ratio: no
    live body), so the hold's line is set by hand under every answer; a window later (6,000 steps) the gauge step reads
    the held-out ratio: equal to the start's (the window unchanged, the encoder saved then), unlike the moved
    encoder's."""
    m = _lite(**dict(FLAG, proprio_token_line=PT.LINE_MAX, proprio_token_deaf=DEAF_MAX))
    pw = m.proprio_within
    g = torch.Generator().manual_seed(4)
    base = 5.0 * torch.randn(1, 312, generator=g)
    walk = torch.zeros(8, 312)
    for i in range(PT.WINDOW):
        walk = walk + 0.01 * torch.randn(8, 312, generator=g)
        pw.observe(base + walk, 5 * i, real=True)
    frame = {"proprioception": torch.randn(8, 312)}
    stale = {m_: "real" for m_ in m.active}
    stale["proprioception"] = "stale"                  # the step is recorded, the row is not kept
    m.step_count = 1
    m._proprio_within_lesson(frame, stale, train=True)
    assert pw.start["fatigue_line"] is None and pw.start["live"] == [False] * 8, "no live body: the hold waits"
    assert pw.pushed_ever == [True] * 8 and pw.snapshot_step == 1
    pw.start["fatigue_line"] = 1e-9
    m.step_count = 2
    loss = m._proprio_within_lesson(frame, stale, train=True)
    assert pw.last["held"] == [True] * 8 and all(a is not None for a in pw.last["fatigue_answer_posture_only"])
    grads = torch.autograd.grad(loss, m.params, allow_unused=True)
    prop = {id(p) for p in m.encoder_registry.encoders["proprioception"].parameters()}
    got = [p for p, gr in zip(m.params, grads) if gr is not None and float(gr.abs().sum()) > 0]
    assert got and all(id(p) in prop for p in got), "the hold reaches the proprioception encoder only"
    enc = m.encoder_registry.encoders["proprioception"]
    first = next(p for p in enc.parameters() if p.dim() == 2 and p.shape[1] == 312)
    with torch.no_grad():
        first[:, posture_columns(312)] *= 3.0          # the live encoder moves; the saved one does not
    m.step_count = 1 + PT.WINDOW * PT.SAMPLE_EVERY + 99  # the first gauge step a window after the save
    assert m.step_count % PT.GAUGE_EVERY == 0
    m._proprio_within_lesson(frame, stale, train=True)
    ho = pw.held_out
    assert ho is not None and ho["encoder_step"] == 1 and ho["step"] == m.step_count
    assert ho["ratio"] == pytest.approx(pw.start["ratio"], rel=1e-4), "the saved encoder on the unchanged window"
    assert pw.last["ratio"][0] != pytest.approx(ho["ratio"][0], rel=1e-2), "not the moved encoder's reading"
    assert pw.snapshot_step == m.step_count, "the encoder is saved again for the next window"


def _drive(m, tmp_path, n, tag):
    from mimo_bridge.datasource import SingleEnvSource
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / tag), tag="_" + tag, vision=False, seed=5)
    try:
        frames = src.reset()
        dev = dict(m.dev_limits(10 ** 9))
        dev["exploration_noise"] = 0.05
        held = prev = None
        out = []
        for i in range(n):
            res = m.step(frames, dev, train=True, step_index=i, executed_action=prev)
            out.append(res)
            if i % 5 == 0 or held is None:
                held = res.motor_for_action
            for _ in range(5):
                frames = src.step(held, res.gaze_out, None, render=False)
            prev = held
        return out
    finally:
        src.close()


@pytest.mark.slow
def test_flag_on_trains_bit_identically_until_the_window_is_full_then_adds_the_lesson_after_the_clip(
        tmp_path, monkeypatch):
    """The switch changes nothing: the same seeds, the same body, the same steps - every parameter equal while
    the window is short. Then the lesson's gradient joins the encoder's after the clip (the window shortened to
    reach it in a short run)."""
    monkeypatch.setattr(PT, "WINDOW", 12)
    monkeypatch.setattr(PT, "MIN_SAMPLES", 12)
    monkeypatch.setattr(PT, "FOLDS", 3)
    monkeypatch.setattr(PT, "GAP", 1)
    monkeypatch.setattr(PT, "SHIFT", 6)
    kw = dict(device="cpu", seed=0, intention=False, imagination=False, next_frame_loss=True,
              overrides={"training": {"steps_per_phase": 3000}})
    from mimo_bridge.lem_model import LemLite
    torch.manual_seed(0)
    off = LemLite(**kw)
    torch.manual_seed(0)
    on = LemLite(**dict(FLAG, proprio_token_line=PT.LINE_MAX, proprio_token_deaf=DEAF_MAX), **kw)
    torch.manual_seed(1)
    _drive(off, tmp_path, 40, "off")
    torch.manual_seed(1)
    _drive(on, tmp_path, 40, "on")
    # steps 0, 5, ..., 35 sampled -> 8 samples through step 39: below the window, no lesson anywhere
    assert on.proprio_within.count == 8 and not on.proprio_within.applied
    for a, b in zip(off.params, on.params):
        assert torch.equal(a, b), "flag on trained exactly as flag off while the window was short"
    _drive(on, tmp_path, 30, "more")
    assert on.proprio_within.applied.get("tensors", 0) > 0, "the lesson's gradient joined after the clip"
    g = on.proprio_within.gauges()
    assert g["samples"] == 12 and len(g["ratio"]) == 1 and "ratio_posture_only" in g and "applied" in g
    assert g["start"]["ratio"] and "content" in g and len(g["attributed"]) == 1 and "lesson_pace" in g


# ---------------------------------------------------------------- the trainer
def test_the_trainer_flags_default_off_and_reach_the_learner_and_the_record():
    from mimo_bridge import train_mimo as TM
    a = TM.parse_args(["--out", "x"])
    assert a.proprio_token_repair == "off" and a.proprio_token_strength is None and a.proprio_token_line is None
    assert a.proprio_token_skip is None
    assert a.proprio_token_deaf is None
    b = TM.parse_args(["--out", "x", "--proprio-token-repair", "within_body", "--proprio-token-strength", "0.03",
                       "--proprio-token-line", "3.75", "--proprio-token-deaf", "2.86", "--proprio-token-skip", "3"])
    assert b.proprio_token_repair == "within_body" and b.proprio_token_strength == 0.03
    assert b.proprio_token_line == 3.75 and b.proprio_token_skip == "3" and b.proprio_token_deaf == 2.86
    for gone in ("--proprio-token-lambda", "--proprio-token-still"):
        with pytest.raises(SystemExit):
            TM.parse_args(["--out", "x", gone, "1"])
    src = inspect.getsource(TM)
    assert 'proprio_token_repair=getattr(args, "proprio_token_repair", "off") or "off"' in src
    assert 'proprio_token_strength=getattr(args, "proprio_token_strength", None)' in src
    assert 'proprio_token_line=getattr(args, "proprio_token_line", None)' in src
    assert 'proprio_token_skip=getattr(args, "proprio_token_skip", None)' in src
    assert 'proprio_token_deaf=getattr(args, "proprio_token_deaf", None)' in src
    assert '"proprio_token_repair": model.proprio_within.gauges()' in src
    cont = src.split("continuation_flags = (")[1].split(")")[0]
    assert "proprio_token" not in cont, "a resume of the old command must not be refused"
    assert "proprio_token_repair" in TM.acting_configuration(b), "it changes what the model learns"
