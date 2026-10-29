# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The world's snapshot carries the per-frame trackers _collect and step read (2026-09-26,
docs/gain_of_function_v4/agents/env_full_state_trackers_2026-09-26.md).

Until then BridgeEnv.save_full_state left out eight of them: the mobile's chime gate
(_mobile_touch_prev, _chime_cool), the mobile's previous position (_mobile_prev), the jitter
report (_jit_prev, _jit_ring), the roll report (_chest_R0, _roll_max) and the command-following
window (_cmd_hist). A restored world read whatever it held before the restore: a trainer's resume
the values of the world it had just built, a paired branch the values the previous branch left.
The eye ability test measured the chime case: a run that chimed blocked the chime in the next run
of the same arm (gaze_probe_replay.md).

The snapshot carries seven of them. The roll maximum (_roll_max) is left out on purpose
(coordinator's decision of 2026-09-26 on the review's semantic finding,
env_full_state_trackers_review_2026-09-26.md): the trainer's "rolled over" share counts it per
training segment, and carried it could only rise across resumes.

The light tests put a real BridgeEnv's save_full_state and restore_full_state on a one-box MuJoCo
world (no MIMo body, no rendering): they build nothing heavy. The slow tests build the crib world.
"""
from __future__ import annotations

import copy
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from mimo_bridge import env as E

# the seven the snapshot carries, named here rather than read from the env, so these tests run on
# code without them
TRACKERS = ("_mobile_touch_prev", "_chime_cool", "_mobile_prev", "_jit_prev", "_jit_ring",
            "_chest_R0", "_cmd_hist")
# a snapshot saved before 2026-09-26 lacks all eight: the seven and the roll maximum
OLD_SNAPSHOT_LACKS = TRACKERS + ("_roll_max",)

XML = """<mujoco><worldbody><body name="box" pos="0 0 0.2"><freejoint/>
<geom type="box" size=".05 .05 .05"/></body></worldbody></mujoco>"""


class Chimes:
    """The audio synthesizer's chime, recorded. The tests pass the frame number as the struck
    element's position, so `frames` lists the frames it sounded on. Plain data only: the world's
    snapshot saves and restores the synthesizer's whole __dict__, as it does the real one's."""

    def __init__(self):
        self.frames = []

    def chime(self, where):
        self.frames.append(int(where))


def bare_world(**trackers):
    """A BridgeEnv holding only what save_full_state and restore_full_state read, on a one-box world,
    with the eight trackers as a freshly built world holds them (or as given), and the seven carried
    ones kept as the world's as-built values, as BridgeEnv.__init__ keeps them."""
    import mujoco
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    e = object.__new__(E.BridgeEnv)
    e._mj, e.model, e.data = mujoco, m, d
    e.frame, e.dt = 0, 0.01
    e._target_vel, e._last_gaze_cmd = np.zeros(3), (0.0, 0.0)
    e.rng = np.random.default_rng(0)
    e.env = SimpleNamespace(_obs_history=[], _action_history=[],
                            actuation_model=SimpleNamespace(control_input=np.zeros(0)))
    e.pooler = SimpleNamespace(**{n: None for n in E.BridgeEnv._FULL_STATE_POOLER})
    e.reflex = e.saccades = e.caregiver = None
    e._audio = Chimes()
    e.proprio = SimpleNamespace(fatigue=SimpleNamespace(cap=np.ones(2), accum=np.zeros(2)), _vel6=np.zeros(6))
    e.hand_target_mode, e.limb_target_control = "world", None
    e.contact_profile, e._virtual_skin = "legacy", None
    for n in E.BridgeEnv._FULL_STATE_ATTRS:
        if not hasattr(e, n):
            setattr(e, n, None)
    e._contact_last_seen, e._press_below = {}, {}
    fresh = dict(_mobile_touch_prev=False, _chime_cool=-1, _mobile_prev=np.array([0.06, 0.0, 0.2]),
                 _jit_prev=np.zeros(40), _jit_ring=[], _chest_R0=np.eye(3), _roll_max=0.0, _cmd_hist=[])
    fresh.update(trackers)
    for n, v in fresh.items():
        setattr(e, n, copy.deepcopy(v))
    e._built_trackers = {n: copy.deepcopy(v) for n, v in fresh.items() if n in TRACKERS}
    for _ in range(5):
        mujoco.mj_step(m, d)
    return e


def lived(e):
    """Tracker values a world reaches after living a while: every one differs from the as-built."""
    e.frame = 1200
    e._mobile_touch_prev, e._chime_cool = True, 1215
    e._mobile_prev = np.array([0.07, 0.01, 0.21])
    e._jit_prev = np.linspace(-1.0, 1.0, 40)
    e._jit_ring = [0.1, 0.2, 0.3]
    e._chest_R0 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    e._roll_max = 37.5
    e._cmd_hist = [(np.full((3, 3), 0.1), np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)))]


def trample(e):
    """What the next branch does to the trackers: changes every value, in place and by rebinding."""
    e.frame += 40
    e._mobile_touch_prev, e._chime_cool = False, e.frame + 30
    e._mobile_prev = np.array([9.0, 9.0, 9.0])
    e._jit_prev[:] = 7.0
    e._jit_ring.append(0.9)
    e._chest_R0[0, 0] = 5.0
    e._roll_max = 170.0
    e._cmd_hist.append((np.ones((3, 3)), np.ones((3, 4))))


def values(e):
    return {n: copy.deepcopy(getattr(e, n)) for n in TRACKERS}


def assert_same(got, want):
    assert set(got) == set(want)
    for n in want:
        a, b = got[n], want[n]
        if n == "_cmd_hist":
            assert len(a) == len(b), n
            for (wa, qa), (wb, qb) in zip(a, b):
                np.testing.assert_array_equal(wa, wb)
                np.testing.assert_array_equal(qa, qb)
        elif isinstance(b, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=n)
        else:
            assert a == b, (n, a, b)


def test_a_round_trip_carries_every_per_frame_tracker():
    e = bare_world()
    lived(e)
    saved = values(e)
    snap = e.save_full_state()
    trample(e)
    e.restore_full_state(snap)
    assert_same(values(e), saved)
    # the restored values are copies: a second branch from the same snapshot starts the same way
    trample(e)
    e.restore_full_state(snap)
    assert_same(values(e), saved)
    # and a world built afresh (the trainer's resume) continues from them too
    other = bare_world()
    other.restore_full_state(snap)
    assert_same(values(other), saved)


def test_a_snapshot_saved_without_them_restores_the_world_as_built():
    e = bare_world()
    built = values(e)
    lived(e)
    snap = e.save_full_state()
    old = dict(snap, attrs={k: v for k, v in snap["attrs"].items() if k not in OLD_SNAPSHOT_LACKS})
    # a freshly built world restores it unchanged: every resume before 2026-09-26 did this
    fresh = bare_world()
    fresh.restore_full_state(old)
    assert_same(values(fresh), built)
    # a world that lived on gets the same values back, as often as the snapshot is restored
    e.restore_full_state(old)
    assert_same(values(e), built)
    trample(e)
    e.restore_full_state(old)
    assert_same(values(e), built)
    assert e.frame == snap["frame"]                 # the rest of the state is restored as before
    assert e._contact_last_seen == {} and e._press_below == {}


def test_the_roll_maximum_restarts_every_segment_while_the_other_trackers_carry():
    """_roll_max stays out of the snapshot. The trainer's resume (a newly built world) starts it at
    0, so the "rolled over" share it feeds counts each segment afresh, as it always has; the seven
    carried trackers continue. A restore never touches it: a world restored in place keeps its own,
    as every restore did before 2026-09-26 (a probe session carries it itself)."""
    e = bare_world()
    lived(e)                                        # the roll maximum reached 37.5 degrees
    saved = values(e)
    snap = e.save_full_state()
    assert "_roll_max" not in snap["attrs"]
    resumed = bare_world()
    resumed.restore_full_state(snap)
    assert_same(values(resumed), saved)
    assert resumed._roll_max == 0.0
    trample(e)                                      # the next branch reaches 170 degrees
    e.restore_full_state(snap)
    assert_same(values(e), saved)
    assert e._roll_max == 170.0


def _touch(e, frames, touching):
    """The chime gate on each frame, as _collect runs it after env.step advanced the frame."""
    for f in frames:
        e.frame = f
        e._mobile_chime(bool(touching(f)), f)


def _strikes(f):
    """A hand strikes the mobile at 10 (held to 11), at 20 (inside the 0.3 s = 30 frame cooldown),
    at 45 holding on to 90, and at 100: it chimes at 10, 45 and 100."""
    return f in (10, 11, 20, 100) or 45 <= f <= 90


END = 120


@pytest.mark.parametrize("save_at,after", [(15, [45, 100]), (80, [100])])
def test_a_chime_before_the_save_is_gated_after_restore_as_in_the_unbroken_run(save_at, after):
    """save_at 15: inside the cooldown of the chime at 10 (the gate's _chime_cool); 80: inside the
    touch held since 45, its cooldown over (the gate's _mobile_touch_prev alone)."""
    e = bare_world()
    _touch(e, range(1, save_at + 1), _strikes)
    snap = e.save_full_state()
    _touch(e, range(save_at + 1, END), _strikes)
    assert e._audio.frames == [10, 45, 100]
    unbroken = [f for f in e._audio.frames if f > save_at]
    assert unbroken == after
    # another branch of the same arm, from the same snapshot: one strike, 25 frames on
    e.restore_full_state(snap)
    e._audio.frames = []
    _touch(e, range(save_at + 1, END), lambda f: f == save_at + 25)
    assert e._audio.frames == [save_at + 25]
    # and the unbroken commands again: the same chimes, not the gate the branches left behind
    e.restore_full_state(snap)
    e._audio.frames = []
    _touch(e, range(save_at + 1, END), _strikes)
    assert e._audio.frames == unbroken
    # a world built afresh and restored (the trainer's resume) gates the same way
    fresh = bare_world()
    fresh.restore_full_state(snap)
    fresh._audio.frames = []
    _touch(fresh, range(save_at + 1, END), _strikes)
    assert fresh._audio.frames == unbroken


# ---------------------------------------------------------------------------- the crib world
def _orders(env, rng, n_frames, hold=5):
    """Held orders of the body's own handles: each handle's place and orientation, turned by up to
    10 degrees about a random axis, strength 0.3, a new order every `hold` frames."""
    ids = env.proprio.body_ids
    out = []
    for k in range(n_frames):
        if k % hold == 0:
            q = np.array(env.data.xquat[ids])[:, [1, 2, 3, 0]]
            order = np.zeros((len(ids), 9))
            order[:, 0:3] = np.array(env.data.xpos[ids]) * 100.0
            axis = rng.normal(size=(len(ids), 3))
            axis /= np.linalg.norm(axis, axis=1, keepdims=True)
            half = np.radians(rng.uniform(0.0, 10.0, size=len(ids))) / 2.0
            turn = np.column_stack([axis * np.sin(half)[:, None], np.cos(half)])
            v1, w1, v2, w2 = turn[:, :3], turn[:, 3:], q[:, :3], q[:, 3:]
            order[:, 3:6] = w1 * v2 + w2 * v1 + np.cross(v1, v2)
            order[:, 6:7] = w1 * w2 - np.sum(v1 * v2, axis=1, keepdims=True)
            order[:, 8] = 0.3
        out.append(order.copy())
    return out


def _live(env, seed, n_frames=25):
    """Step n_frames under seeded orders; every frame's task, command response, hearing and body."""
    rng = np.random.default_rng(seed)
    record = []
    orders = None
    for k in range(n_frames):
        if k % 5 == 0:
            orders = _orders(env, rng, 5)
        raw = env.step(motor=orders[k % 5])
        record.append(dict(task=copy.deepcopy(raw.info["task"]), cmd=copy.deepcopy(raw.info.get("cmd_response")),
                           slow=copy.deepcopy(raw.info.get("cmd_response_slow")),
                           audio=np.array(raw.info.get("audio_spec")), qpos=np.array(env.data.qpos)))
    return record


def _differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}.{k}: only one side" for k in set(a) ^ set(b)]
        for k in set(a) & set(b):
            out += _differences(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b) \
            and not (a and isinstance(a[0], (int, float))):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _differences(x, y, f"{path}[{i}]")]
    try:
        same = np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    except (TypeError, ValueError):
        same = a == b
    return [] if same else [f"{path}: {a!r} != {b!r}"[:300]]


def _run_differences(a, b, skip=()):
    """Every difference between two runs' frames, leaving out the named task entries."""
    assert len(a) == len(b)
    cut = lambda r: dict(r, task={k: v for k, v in r["task"].items() if k not in skip})
    return [d for i, (x, y) in enumerate(zip(a, b)) for d in _differences(cut(x), cut(y), f"frame{i}")]


def _crib(tmp_path):
    return dict(scene="crib", run_dir=str(tmp_path), vision=False, tag="_trackertest", age=24,
                contact_profile="mechanical_v4", body_function="modulate", caregiver_seconds=0, kp=6.0,
                body_layout="trunk_v1", cmd_horizon=25, mobile=True, critters=True, audio_real=True, seed=3)


@pytest.mark.slow
def test_a_restored_crib_world_continues_exactly_as_the_unbroken_one(tmp_path):
    """Everything but the roll maximum continues bit for bit; the roll maximum restarts with the
    trainer's resume (test_the_roll_maximum_restarts_every_segment_while_the_other_trackers_carry)."""
    kw = _crib(tmp_path)
    env = E.BridgeEnv(**kw)
    try:
        env.reset()
        _live(env, seed=1, n_frames=40)                 # fills the jitter ring and the command window
        assert len(env._cmd_hist) == 10 and len(env._jit_ring) == 40
        env._roll_max = 150.0                           # as if the body had rolled over earlier in the segment
        snap = env.save_full_state()
        unbroken = _live(env, seed=2)
        _live(env, seed=3)                              # another branch from the same state
        env.restore_full_state(snap)
        again = _live(env, seed=2)
        diffs = _run_differences(unbroken, again, skip=("roll_max_deg",))
        assert diffs == [], diffs[:10]
        assert all(r["task"]["roll_max_deg"] == 150.0 for r in unbroken + again)
    finally:
        env.close()
    fresh = E.BridgeEnv(**kw)                           # the trainer's resume: a new world, then restore
    try:
        start = fresh._roll_max
        fresh.restore_full_state(snap)
        resumed = _live(fresh, seed=2)
        diffs = _run_differences(unbroken, resumed, skip=("roll_max_deg",))
        assert diffs == [], diffs[:10]
        # the new segment's maximum is the largest roll since the resume: this body, past 120 degrees
        # in the segment before, is not counted "rolled over" in this one until it rolls again
        rolls = [r["task"]["roll_deg"] for r in resumed]
        np.testing.assert_array_equal([r["task"]["roll_max_deg"] for r in resumed],
                                      np.maximum.accumulate([start] + rolls)[1:])
        assert start < 1e-6 and max(rolls) < 120.0
        # an episode reset clears the chime gate's frame-keyed cooldown with the zeroed frame counter
        fresh._chime_cool = fresh.frame + 1000
        fresh.reset()
        assert fresh._chime_cool == -1
    finally:
        fresh.close()


class _MigrationLines(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.lines = []

    def emit(self, record):
        if "per-frame trackers" in record.getMessage():
            self.lines.append(record.getMessage())


@pytest.mark.slow
def test_an_old_snapshot_restored_into_a_newly_built_world_runs_as_todays_resume(tmp_path):
    """The next resume's path (review of 2026-09-26, its first MINOR finding). Every companion on
    disk was saved without the trackers, and the trainer restores it into a world BridgeEnv.__init__
    has just built; today's restore_full_state leaves the trackers as that world was built. The
    migration must put back exactly those values, so the 25 frames after the restore are today's,
    bit for bit. The as-built values are read here right after __init__ returns, never from the
    world's own _built_trackers, so a capture taken at the wrong moment in __init__ fails."""
    kw = _crib(tmp_path)
    env = E.BridgeEnv(**kw)
    try:
        env.reset()
        _live(env, seed=1, n_frames=40)                 # a world that lived: every tracker moved
        snap = env.save_full_state()
    finally:
        env.close()
    old = dict(snap, attrs={k: v for k, v in snap["attrs"].items() if k not in OLD_SNAPSHOT_LACKS})

    def as_built(world):
        return {n: copy.deepcopy(getattr(world, n)) for n in OLD_SNAPSHOT_LACKS if hasattr(world, n)}

    today = E.BridgeEnv(**kw)                           # today's resume: build, restore, trackers untouched
    try:
        built = as_built(today)
        today.restore_full_state(old)
        for n in OLD_SNAPSHOT_LACKS:
            if n in built:
                setattr(today, n, copy.deepcopy(built[n]))
            elif hasattr(today, n):
                delattr(today, n)
        want = _live(today, seed=2)
    finally:
        today.close()
    lines = _MigrationLines()
    log = logging.getLogger("mimo_bridge.env")
    new = E.BridgeEnv(**kw)                             # the same resume through the migration
    try:
        built_new = as_built(new)
        log.addHandler(lines)
        try:
            new.restore_full_state(old)
        finally:
            log.removeHandler(lines)
        restored = as_built(new)
        got = _live(new, seed=2)
    finally:
        new.close()
    assert _differences(restored, built_new) == [], _differences(restored, built_new)[:10]
    diffs = _run_differences(want, got)
    assert diffs == [], (len(diffs), diffs[:10])
    assert len(lines.lines) == 1 and "saved before 2026-09-26" in lines.lines[0]
