# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The eye-row split job (docs/gain_of_function_v4/motor_fixes_eye_row_split.py): the ledger's eye row
split into the model's gaze command (res.gaze_out, after the planner; the output's 'controller' part)
and the world's parts (the saccade generator, the gaze reflex).

Pinned here, with no checkpoint:
  * the three parts add up to the ledger row the trainer and the probe harness write
    (LemLite.gaze_to_handle of the burst's mean sent gaze), in both conventions;
  * with nothing added by the world the ledger is the controller's row, and the controller's part
    keeps the controller row's numbers under both conventions (in the world the ledger keeps only the
    command's share of the sent gaze, all of it with no reflex snap and 0.3 of it during one, and the
    switch writes the rest mirrored: the row after a switch, below);
  * THE ROW AFTER A SWITCH (the fourth review's finding 2, fixed in its fix round): world_sent is the world's
    own arithmetic, equal to SpontaneousSaccades.apply and GazeReflex.apply themselves on random states
    (clipped offsets, snaps and no snaps); on the review's five cases the row after a switch is the row the
    world's classes make from the model's own mirrored command, where the formula before it was off by up to
    56 degrees of pitch; and run_body, on a world of the real classes with snaps steering, records for every
    step the row the world's classes make from the command the model's own row gives under the other
    convention, and reproduces the gaze the world sent from the recorded command (the reconstruction check);
  * the world tap reads the command at every physics step through the classes' own apply methods, with the
    generator's offset, the reflex's blend and target, and leaves the world's objects as it found them;
  * run_body, through the probe harness's own Session.policy_step / policy_act / act on a stub world:
    every burst is read, the parts equal the harness's own ledger, and the latch is the harness's;
  * the statistics and the centre offsets.
"""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mimo_bridge.body_layout import names_for
from mimo_bridge.gaze_reflex import EL_RANGE
from mimo_bridge.lem_model import LemLite

ROOT = Path(__file__).resolve().parents[4]
_spec = importlib.util.spec_from_file_location(
    "motor_fixes_eye_row_split", ROOT / "docs" / "gain_of_function_v4" / "motor_fixes_eye_row_split.py")
ES = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ES)

GI = LemLite.GAZE_IDX
H = len(names_for("trunk_v1"))            # H84's body: 26 rows, the eyes' row 23


def _order(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(1, H, 9)).astype(np.float32)


def _row_of_pitch(yaw, pitch):
    """A controller eye row with this yaw and pitch (x y z w as gaze_to_handle writes them)."""
    o = np.zeros((1, H, 9), np.float32)
    o[0, GI, 3:7] = [-math.sin(yaw / 2) * math.sin(pitch / 2), math.cos(yaw / 2) * math.sin(pitch / 2),
                     math.sin(yaw / 2) * math.cos(pitch / 2), math.cos(yaw / 2) * math.cos(pitch / 2)]
    return o


@pytest.mark.parametrize("convention", ["legacy", "physical"])
def test_the_parts_add_up_to_the_ledger_the_trainer_writes(convention):
    rng = np.random.default_rng(1)
    for t in range(50):
        order = _order(t)
        ctrl = rng.uniform([-0.5, -0.5], [0.5, 0.4])
        sacc = ctrl + rng.normal(0, 0.08, size=(5, 2))
        sent = sacc + rng.normal(0, 0.1, size=(5, 2)) * (rng.random((5, 1)) < 0.5)
        p = ES.step_parts(order, ctrl, sacc, sent, convention, GI)
        np.testing.assert_allclose(p["controller"] + p["saccade"] + p["reflex"], p["ledger"], atol=1e-12)
        ledger = LemLite.gaze_to_handle(order, sent.mean(0)[None], convention=convention)
        np.testing.assert_allclose(p["ledger"], ledger[0, GI, 3:5], atol=1e-7)


@pytest.mark.parametrize("convention", ["legacy", "physical"])
def test_with_nothing_added_by_the_world_the_ledger_is_the_controllers_row(convention):
    order = _order(3)
    ctrl = np.array([0.2, -0.3])
    p = ES.step_parts(order, ctrl, np.repeat(ctrl[None], 5, 0), np.repeat(ctrl[None], 5, 0), convention, GI)
    assert np.all(p["saccade"] == 0) and np.all(p["reflex"] == 0)
    np.testing.assert_allclose(p["ledger"], p["controller"], atol=0)


def test_the_controllers_part_keeps_the_controller_rows_numbers_in_both_conventions():
    """The same controller row asks for mirrored elevations in the two conventions, and the ledger writes
    each back as the controller's row: the controller's part never flips at a switch."""
    for yaw, pitch in [(0.3, -0.4), (-0.2, 0.25), (0.0, 0.5)]:
        row = _row_of_pitch(yaw, pitch)
        for convention, el in (("legacy", pitch), ("physical", -pitch)):
            p = ES.step_parts(row, [yaw, el], [[yaw, el]] * 5, [[yaw, el]] * 5, convention, GI)
            np.testing.assert_allclose(p["controller"], row[0, GI, 3:5], atol=1e-6)


# ------------------------------------------------------------------ the row after a switch, exact
def _world_classes_sent(command, offset, target=None, steer=False, weight=0.7):
    """The gaze the world's OWN classes send for one physics step: a SpontaneousSaccades holding `offset` (never
    resampling: its next switch far ahead), then, when `steer`, a GazeReflex with a snap pending toward `target`
    (only apply() is used: its weight, target and steps; no arrival check without the eyes' angles)."""
    from mimo_bridge.gaze_reflex import GazeReflex, SpontaneousSaccades
    s = SpontaneousSaccades(dt=0.01, seed=0)
    s._offset = np.asarray(offset, dtype=np.float64)
    s._next_switch = 10 ** 9
    g = s.apply(np.asarray(command, dtype=np.float64), frame=0, reflex_active=steer)
    r = GazeReflex.__new__(GazeReflex)
    r.weight, r._target = weight, (None if target is None else np.asarray(target, dtype=np.float64))
    r._steps_left, r.arrive_rad = (5 if steer else 0), 0.0
    return np.asarray(r.apply(g, None), dtype=np.float64).reshape(-1)[:2]


def test_world_sent_is_the_worlds_own_arithmetic():
    """The job's world_sent against SpontaneousSaccades.apply and GazeReflex.apply themselves: commands across the
    eye's range (its stops included), offsets up to 40 degrees (so the generator's clip binds), snaps on and off."""
    from mimo_bridge.gaze_reflex import AZ_RANGE, EL_RANGE
    rng = np.random.default_rng(12)
    clipped = 0
    for _ in range(300):
        c = np.array([rng.uniform(*AZ_RANGE), rng.choice([EL_RANGE[0], EL_RANGE[1], rng.uniform(*EL_RANGE)])])
        offs = rng.normal(0, 0.35, size=(5, 2))
        blends = rng.random(5) < 0.5
        tgts = np.stack([rng.uniform(*AZ_RANGE, 5), rng.uniform(*EL_RANGE, 5)], -1)
        got = ES.world_sent(c, offs, blends, tgts)
        want = np.stack([_world_classes_sent(c, offs[i], tgts[i], bool(blends[i])) for i in range(5)])
        assert np.array_equal(got, want), (c, offs, blends)
        raw = c[None] + offs
        clipped += int(((raw[:, 1] < EL_RANGE[0]) | (raw[:, 1] > EL_RANGE[1])).sum())
    assert clipped > 100, "the cases reach the generator's clip"


def _row_pitch_deg(order, gi=GI):
    x, y, z, w = [float(v) for v in np.asarray(order)[0, gi, 3:7]]
    return math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))


def test_the_row_after_a_switch_on_the_fourth_reviews_cases():
    """The fourth review's probe (probe_snap_row.py): a legacy row asking pitch p sends +p, the same row under
    'physical' sends -p (the model's clamp); the world adds its offset and, under a snap, blends 0.7 of the way to
    the target. The row after the switch must be the row the world's classes make from the mirrored command. The
    formula before the fix read 24.1 and 56.0 degrees off on the second and fourth cases."""
    cases = [("no snap", 0.30, None, (0.0, 0.05)),
             ("snap", 0.30, (0.10, 0.20), (0.0, 0.05)),
             ("snap", -0.30, (0.10, 0.20), (0.0, 0.05)),
             ("planner at the lower stop, snap", EL_RANGE[0], (0.05, 0.30), (0.0, 0.0)),
             ("planner at the lower stop, no snap", EL_RANGE[0], None, (0.0, 0.08))]
    order = _order(9)
    for name, p, target, off in cases:
        az = 0.1
        cmd_legacy = np.array([az, float(np.clip(p, *EL_RANGE))])
        cmd_physical = np.array([az, float(np.clip(-p, *EL_RANGE))])
        steer = target is not None
        tgt = np.full(2, np.nan) if target is None else np.asarray(target)
        true_sent = np.stack([_world_classes_sent(cmd_physical, off, target, steer)] * 5).mean(0)
        true_row = LemLite.gaze_to_handle(order, true_sent[None], convention="physical")
        xy, mean = ES.switched_row(order, cmd_physical, [off] * 5, [steer] * 5, [tgt] * 5, 0.7, "legacy", GI)
        np.testing.assert_allclose(mean, true_sent, atol=1e-12, err_msg=name)
        np.testing.assert_allclose(xy, ES.eye_xy(true_row, GI)[0], atol=1e-7, err_msg=name)
        stored = LemLite.gaze_to_handle(
            order, np.stack([_world_classes_sent(cmd_legacy, off, target, steer)] * 5).mean(0)[None],
            convention="legacy")
        # the row the world makes after the switch is not the stored row with the command's part mirrored in
        # full during a snap: only 0.3 of the command reaches the gaze sent
        if steer:
            assert abs(_row_pitch_deg(true_row) - _row_pitch_deg(stored)) > 1.0


def test_the_command_after_a_switch_is_the_models_own_for_the_same_row():
    """A command at the eye's UPPER stop (+33 degrees) came from a row that may ask for more; mirrored, the row asks
    for a point inside the range only the row names. The model's own _gaze_from_handle under the other convention
    decides (ModelTap); without the row the recorded command's mirror stands in and says so."""
    class _Eyes:
        GAZE_IDX = GI

        def __init__(self, convention):
            self.eye_row_convention = convention
    import torch
    for pitch in (0.9, 0.7, 0.576, 0.3, -0.5, -0.9):
        row = torch.as_tensor(_row_of_pitch(0.2, pitch))
        legacy = LemLite._gaze_from_handle(_Eyes("legacy"), row)[0]
        physical = LemLite._gaze_from_handle(_Eyes("physical"), row)[0]
        tap = ES.ModelTap("physical")
        tap.calls = [(np.asarray(legacy, np.float64), np.asarray(physical, np.float64))]
        after, from_row = ES.command_after_switch(legacy, tap.source_of(np.asarray(legacy, np.float64)))
        assert from_row and np.array_equal(after, np.asarray(physical, np.float64))
        mirrored, from_row = ES.command_after_switch(legacy, None)
        assert not from_row
        if pitch > EL_RANGE[1] + 1e-6:
            # the command sits at the upper stop; its mirror reads -33 degrees while the row asks for more
            assert mirrored[1] == pytest.approx(-EL_RANGE[1]) and physical[1] < mirrored[1] - 0.05
        else:
            # inside the range, and at the lower stop (whose mirror is past the upper stop either way), exact
            np.testing.assert_allclose(mirrored, physical, atol=1e-6)


class _Saccades:
    """The generator's interface as the world tap reads it: apply adds _offset (as SpontaneousSaccades does)."""
    def __init__(self, offset):
        self._offset = np.asarray(offset, float)
        self.n_saccades = 0

    def apply(self, gaze, frame, reflex_active=False):
        if frame % 3 == 0:
            self.n_saccades += 1
        return np.asarray(gaze, float).reshape(-1)[:2] + self._offset


class _Reflex:
    """The reflex's interface as the world tap reads it: its input returned unchanged unless a snap steers, then
    (1 - weight) of it plus weight of _target (as GazeReflex does)."""
    def __init__(self, target, steer_frames):
        self._target = np.asarray(target, float)
        self.weight = 0.7
        self.steer_frames = set(steer_frames)
        self.frame = 0

    def apply(self, gaze, eye_yaw_pitch=None):
        f = self.frame
        self.frame += 1
        if f not in self.steer_frames:
            return gaze
        return 0.3 * np.asarray(gaze, float).reshape(-1)[:2] + 0.7 * self._target


class _Env:
    """The order of BridgeEnv.step's eye path: the controller's gaze, the saccade generator, the reflex."""

    def __init__(self, offset=(0.01, -0.05), target=(0.2, 0.3), steer=(1, 2, 7)):
        self.saccades = _Saccades(offset)
        self.reflex = _Reflex(target, steer)
        self.frame = 0

    def step_gaze(self, gaze):
        g = self.saccades.apply(gaze, self.frame, reflex_active=False)
        g = self.reflex.apply(g, None)
        self.frame += 1
        g = np.asarray(g, float).reshape(-1)
        return (float(g[0]), float(g[1]))


class _Source:
    def __init__(self, env):
        self.env = env

    def step(self, order, gaze, mask, render=False, **kw):
        cmd = self.env.step_gaze(np.asarray(gaze, float).reshape(-1)[:2])
        return [SimpleNamespace(meta={"gaze_cmd": cmd})]

    def close(self):
        pass


def test_the_world_tap_reads_every_physics_step_and_leaves_the_world_as_it_was():
    env = _Env()
    tap = ES.WorldTap()
    tap.attach(env)
    assert "apply" in env.saccades.__dict__ and "apply" in env.reflex.__dict__
    ctrl = np.array([0.05, -0.1])
    tap.begin(_order(0), ctrl)
    for _ in range(5):
        cmd = env.step_gaze(ctrl)
        tap.substep(cmd)
    b = tap.burst
    np.testing.assert_allclose(b["sacc"], np.repeat([ctrl + env.saccades._offset], 5, 0))
    np.testing.assert_allclose(b["gaze_cmd"], b["sent"])
    assert b["steered"] == 2 and b["started"] == 2          # frames 1 and 2 steered; frames 0 and 3 started
    # the world's state each physics step: the offset added, whether a snap blended, toward which target
    np.testing.assert_allclose(b["offsets"], np.repeat([env.saccades._offset], 5, 0))
    assert b["blends"] == [False, True, True, False, False] and b["weight"] == 0.7
    np.testing.assert_allclose(b["targets"][1], env.reflex._target)
    assert np.isnan(b["targets"][0]).all()
    np.testing.assert_allclose(ES.world_sent(ctrl, b["offsets"], b["blends"], b["targets"], b["weight"]), b["sent"],
                               atol=1e-15)
    tap.detach()
    assert "apply" not in env.saccades.__dict__ and "apply" not in env.reflex.__dict__
    assert env.step_gaze(ctrl) is not None


def _stub_session(convention="legacy", hold=5, reps=5, steps_gaze=None):
    """The probe harness's own Session, its world and model replaced by stubs, so policy_step,
    policy_act and act run as they do on a checkpoint."""
    from mimo_bridge.tools import gof_probe_harness as Hh
    s = object.__new__(Hh.Session)
    rng = np.random.default_rng(11)
    calls = {"n": 0}

    def model_step(frames, dev, train=False, step_index=0, executed_action=None, executed_order_id=None):
        calls["n"] += 1
        # who authored the step's order, as LemLite.step stamps it (reset to "reflex" every step)
        s.model.last_action_source = ("planner" if calls["n"] % 4 == 0 else
                                      "self_choice" if calls["n"] % 7 == 0 else "reflex")
        g = rng.uniform([-0.3, -0.4], [0.3, 0.3]) if steps_gaze is None else steps_gaze[calls["n"] - 1]
        return SimpleNamespace(motor_for_action=_order(calls["n"]), gaze_out=np.asarray(g)[None],
                               hand_joint_goal=None)
    s.model = SimpleNamespace(eye_row_convention=convention, context_buffer=None, step_count=0, step=model_step,
                              hand_target_mode="world")
    s.src = _Source(_Env())
    s.env = s.src.env
    s.n_handles = H
    s.reps, s.hold = reps, hold
    s.dev = {}
    s.executed, s.order_id, s.res, s.pre_ctx, s.template = None, 0, None, None, None
    s._policy_i, s._held, s._held_hand_goal, s._probe_hand_cache = 0, None, None, None
    s.last_motor_substeps, s.last_hand_joint_goal = None, None
    s.frames = [SimpleNamespace(meta={})]
    return s, calls


@pytest.mark.parametrize("convention", ["legacy", "physical"])
def test_run_body_reads_every_burst_through_the_harness_own_policy(convention):
    s, calls = _stub_session(convention)
    out = ES.run_body(s, 12, convention, GI)
    assert calls["n"] == 12 and len(out["ledger_y"]) == 12
    assert float(out["ledger_check"].max()) < 1e-6, "the parts' ledger is the harness's own ledger row"
    assert float(out["gaze_cmd_check"].max()) == 0.0
    np.testing.assert_allclose(out["controller_y"] + out["saccade_y"] + out["reflex_y"], out["ledger_y"], atol=1e-12)
    assert out["new_order"].tolist() == [1.0 if i % 5 == 0 else 0.0 for i in range(12)], "the harness's latch"
    assert "act" not in s.__dict__, "the session is left with its own act"
    assert float(out["steered_substeps"].sum()) > 0 and float(out["saccade_starts"].sum()) > 0
    # the third review, finding B: the command's part is the model's gaze command (res.gaze_out, after the
    # planner), which entered the saccade generator unchanged; each step's author is kept
    assert float(np.nanmax(out["command_check"])) == 0.0
    want = [1.0 if n % 4 == 0 else 2.0 if n % 7 == 0 else 0.0 for n in range(1, 13)]
    assert out["author"].tolist() == want


class _RealWorld:
    """BridgeEnv.step's eye path with the world's REAL classes: a SpontaneousSaccades (seeded, resampling as it
    does) and a GazeReflex whose snaps are set by hand at chosen physics steps (no camera), each steering until its
    steps run out. The test keeps its own record of the world's state at every physics step (self.log)."""

    def __init__(self, seed=3, snaps=((2, (0.10, 0.25), 6), (14, (-0.20, -0.30), 5), (27, (0.30, 0.10), 7),
                                      (41, (0.0, 0.5), 9))):
        from mimo_bridge.gaze_reflex import GazeReflex, SpontaneousSaccades
        self.saccades = SpontaneousSaccades(dt=0.01, seed=seed, hold_s=(0.03, 0.08))
        r = GazeReflex.__new__(GazeReflex)
        r.weight, r._target, r._steps_left, r.arrive_rad = 0.7, None, 0, 0.0
        self.reflex = r
        self.snaps = {f: (np.asarray(t, float), n) for f, t, n in snaps}
        self.frame = 0
        self.log = []

    def step_gaze(self, gaze):
        if self.frame in self.snaps:
            t, n = self.snaps[self.frame]
            self.reflex._target, self.reflex._steps_left = t.copy(), n
        steer = self.reflex._steps_left > 0 and self.reflex._target is not None
        g1 = self.saccades.apply(gaze, self.frame, reflex_active=bool(self.reflex.active))
        g2 = self.reflex.apply(g1, None)
        self.log.append(dict(offset=np.array(self.saccades._offset, dtype=np.float64), steer=bool(steer),
                             target=None if not steer else np.array(self.reflex._target, dtype=np.float64)))
        self.frame += 1
        g = np.asarray(g2, float).reshape(-1)
        return (float(g[0]), float(g[1]))


class _EyeModel:
    """A stub learner whose eye command is LemLite's own _gaze_from_handle of the row it steps with: rows at the
    eye's upper stop (asking for more than it), at its lower stop, and near level."""
    GAZE_IDX = GI
    _gaze_from_handle = LemLite._gaze_from_handle

    def __init__(self, convention, pitches):
        self.eye_row_convention = convention
        self.pitches = list(pitches)
        self.context_buffer, self.step_count, self.hand_target_mode = None, 0, "world"
        self.last_action_source, self.rows = "reflex", []

    def step(self, frames, dev, train=False, step_index=0, executed_action=None, executed_order_id=None):
        import torch
        n = len(self.rows)
        row = _row_of_pitch(0.15 * math.sin(n), self.pitches[n % len(self.pitches)])
        row[0, :GI] = _order(n)[0, :GI]
        self.rows.append(row)
        self.last_action_source = "planner" if n % 4 == 0 else "reflex"
        gaze = self._gaze_from_handle(torch.as_tensor(row))
        return SimpleNamespace(motor_for_action=row, gaze_out=gaze, hand_joint_goal=None)


def test_run_body_writes_the_row_after_a_switch_as_the_worlds_classes_make_it():
    """The fourth review's finding 2, as the deploy stop's job meets it: on a world of the real classes with snaps
    steering, every recorded step's row after a switch equals the row the world's classes make, physics step by
    physics step, from the command the model's own row gives under 'physical'; the recorded command's arithmetic
    reproduces the gaze the world sent. (Before the fix the job wrote the command's whole mirror plus the world's
    angles, which is wrong wherever a snap steered or the command sat at the upper stop.)"""
    class _Eyes:
        GAZE_IDX = GI
        eye_row_convention = "physical"
    pitches = (0.8, 0.02, -0.9, -0.05, 0.3, 0.0, -0.4)          # 0.8: the upper stop, asking 46 degrees
    s, _ = _stub_session("legacy")
    s.model = _EyeModel("legacy", pitches)
    world = _RealWorld()
    s.src = _Source(world)
    s.env = world
    out = ES.run_body(s, 12, "legacy", GI)
    reps = s.reps
    assert len(world.log) == 12 * reps
    import torch
    steered = 0
    for k in range(12):
        row = s.model.rows[k]
        after = LemLite._gaze_from_handle(_Eyes(), torch.as_tensor(row))[0].astype(np.float64)
        sent = []
        for st in world.log[k * reps:(k + 1) * reps]:
            sent.append(_world_classes_sent(after, st["offset"], st["target"], st["steer"]))
            steered += st["steer"]
        # (the row's eye x and y depend only on the gaze written into it, not on the order's other rows)
        true_row = LemLite.gaze_to_handle(row, np.mean(sent, 0)[None], convention="physical")
        want = ES.eye_xy(true_row, GI)[0]
        np.testing.assert_allclose([out["switched_x"][k], out["switched_y"][k]], want, atol=1e-7,
                                   err_msg="step %d" % k)
    assert steered >= 20, "the snaps steered through the recorded steps"
    assert float(np.max(out["reconstruction_check"])) == 0.0, "the recorded command's arithmetic is the world's"
    assert out["command_after_from_row"].tolist() == [1.0] * 12
    for k in range(12):
        after = LemLite._gaze_from_handle(_Eyes(), torch.as_tensor(s.model.rows[k]))[0].astype(np.float64)
        assert out["el_command_after"][k] == pytest.approx(float(after[1]), abs=0)


def test_the_command_by_author_adds_up_to_the_pooled_mean():
    """Each author's contribution (its steps' sum over all steps) adds up to the command part's pooled mean;
    a rare author at an eye stop can carry most of it (the third review, question 4)."""
    rng = np.random.default_rng(5)
    per_body = {}
    for b in range(3):
        n = 60
        d = {k: rng.normal(0, 0.01, n) for k in ES.SERIES if k not in ("author",)}
        author = np.zeros(n)
        author[::10] = 1.0                                  # the planner every 10th step
        d["controller_y"] = np.where(author == 1.0, -0.35, rng.normal(0.0, 0.01, n))
        d["el_controller"] = np.where(author == 1.0, -0.8, 0.0)
        d["author"] = author
        d["ledger_check"] = np.zeros(n)
        d["gaze_cmd_check"] = np.zeros(n)
        d["command_check"] = np.zeros(n)
        per_body[b] = d
    got = ES.summarise(per_body)
    by = got["pooled"]["command_by_author"]
    assert by["planner"]["steps"] == 18 and by["planner"]["per_100_steps"] == 10.0
    total = sum(v["y_contribution_to_pooled_mean"] for v in by.values())
    assert total == pytest.approx(got["pooled"]["controller"]["y"]["mean"], abs=1e-12)
    assert abs(by["planner"]["y_contribution_to_pooled_mean"]) > abs(by["controller"]["y_contribution_to_pooled_mean"])
    assert by["planner"]["elevation_deg_mean"] == pytest.approx(-0.8 * 180 / math.pi)
    assert got["checks"]["command_largest_difference"] == 0.0


def test_run_body_records_the_memorys_path_at_every_step_when_asked():
    """The four F1 follow-ups, item 4: the job records, after every step, what the memory callable reads
    (motor_fixes_switch_jump.memory_bias_change on the body's history as it is then)."""
    s, calls = _stub_session("legacy")
    seen = []

    def memory(sess):
        seen.append(sess._policy_i)
        return np.full((3, 4), float(sess._policy_i), dtype=np.float32)
    out = ES.run_body(s, 7, "legacy", GI, memory=memory)
    assert seen == list(range(1, 8)), "read after each step"
    assert out["memory_change"].shape == (7, 3, 4) and out["memory_change"][2, 0, 0] == 3.0
    s2, _ = _stub_session("legacy")
    assert "memory_change" not in ES.run_body(s2, 3, "legacy", GI)
    # the job's reader: nothing without a memory or a history
    no_memory = SimpleNamespace(model=SimpleNamespace(persistent_memory=None, memory_history=None))
    assert ES.memory_change_of(no_memory) is None
    empty = SimpleNamespace(model=SimpleNamespace(persistent_memory=object(),
                                                  memory_history=SimpleNamespace(snapshot=lambda: None)))
    assert ES.memory_change_of(empty) is None
    # with a history it is the switch-jump job's memory_bias_change (the repair's running mean of events left as
    # saved, as the live switch leaves it since the fourth review's fix round: no ring's change is passed)
    seen = {}

    import torch

    class _SJ:
        @staticmethod
        def memory_bias_change(m, snap):
            seen.update(snap=snap)
            return torch.full((1, 3, 4), 2.0)

    real = ES._switch_jump
    ES._switch_jump = lambda: _SJ
    try:
        body = SimpleNamespace(model=SimpleNamespace(persistent_memory=object(),
                                                     memory_history=SimpleNamespace(snapshot=lambda: {"events": 1})))
        got = ES.memory_change_of(body)
    finally:
        ES._switch_jump = real
    assert seen == {"snap": {"events": 1}} and got.shape == (3, 4)


def test_the_eye_rows_file_for_the_switch_jump_job():
    """--eye-rows-out: per body the change of the elevation sent, the author and the elevation sent, per step."""
    per_body = {b: {"delta_el": np.arange(4.0) + b, "author": np.array([1.0, 0.0, 0.0, 2.0]),
                    "el_sent": np.zeros(4)} for b in (0, 3)}
    got = ES.eye_rows_record(per_body)
    assert sorted(got) == ["b0__author", "b0__delta_el", "b0__el_sent", "b3__author", "b3__delta_el", "b3__el_sent"]
    np.testing.assert_array_equal(got["b3__delta_el"], [3.0, 4.0, 5.0, 6.0])


def test_series_stats_and_the_centre_offsets():
    xs = [np.arange(40, dtype=float), np.arange(40, dtype=float) + 10.0]
    st = ES.series_stats(xs, block=20)
    assert st["n"] == 80 and st["blocks"] == 4 and st["bodies"] == 2
    assert st["mean"] == pytest.approx(24.5)
    assert st["se_bodies"] == pytest.approx(np.std([19.5, 29.5], ddof=1) / math.sqrt(2))
    offs = ES.centre_offsets([0.01, -0.02], [0.012, -0.01], [0.01, -0.02], [0.002, 0.01])
    np.testing.assert_allclose(offs["left_as_saved"]["new_orders"], [0.002, 0.01])
    np.testing.assert_allclose(offs["negated"]["new_orders"], [0.022, -0.03])
    np.testing.assert_allclose(offs["shifted_by_the_measured_change"]["new_orders"], [0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(offs["negated"]["rewritten_stored_rows"], [0.0, 0.0], atol=1e-15)
