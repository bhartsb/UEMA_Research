# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Self-contact and outside-contact onsets, and which patch is touched by what (2026-09-22).

DESIGN.md section 8. A touch the model can be scored on predicting is an ONSET: a pair of
bodies apart for at least five env frames (50 ms) and touching now. Self-touch is felt on
both sides, so it arrives as two records and must be one onset. A pair that touches while the
body merely lies at rest under zero command (the thumb in its own palm) is posture, not action,
and is marked structural so a success measure can leave it out.
"""
import numpy as np
import pytest

from mimo_bridge import env as E
from mimo_bridge.touch_adapter import ContactInfo


@pytest.fixture(scope="module")
def crib(tmp_path_factory):
    env = E.BridgeEnv(scene="crib", run_dir=str(tmp_path_factory.mktemp("onsets")), vision=False,
                      tag="_onsettest", age=24, contact_profile="mechanical_v4",
                      body_function="modulate", caregiver_seconds=0, kp=6.0,
                      body_layout="trunk_v1", cmd_horizon=25)
    env.reset()
    yield env
    env.close()


def _rec(env, body, other, force=1.0, point=(0.0, 0.0, 0.0), is_self=True, pair=(1, 2)):
    b = env.model.body(body).id
    return ContactInfo(body_id=int(b), pair_key=pair, normal_force=force,
                       tangential_body=np.zeros(3), pos_body=np.zeros(3),
                       pos_world=np.asarray(point, dtype=np.float64),
                       normal_world=np.array([0.0, 0.0, 1.0]), penetration_m=0.0,
                       rel_normal_speed=0.0, other_body_name=other, other_is_self=is_self)


def _report(env, contacts):
    task = {}
    env._contact_report(contacts, task)
    env.frame += 1
    return task


def _fresh(env):
    env._contact_last_seen = None
    env.frame = 1000


def _eye_point(env, deg, dist=0.25):
    """A world point `deg` degrees off the left eye's axis."""
    cid = env.model.camera("eye_left").id
    R = np.asarray(env.data.cam_xmat[cid]).reshape(3, 3)
    fwd, side = -R[:, 2], R[:, 0]
    th = np.radians(deg)
    return np.asarray(env.data.cam_xpos[cid]) + dist * (np.cos(th) * fwd + np.sin(th) * side)


def test_the_first_report_after_a_reset_starts_tracking_and_reports_nothing(crib):
    env = crib
    _fresh(env)
    both = [_rec(env, "right_ffdistal", "left_hand"), _rec(env, "left_hand", "right_ffdistal")]
    t = _report(env, both)
    assert t["self_contact_onsets"] == [] and t["contact_onsets_world"] == 0, \
        "how long a pair touching at the first frame had been touching is unknown"


def test_a_self_touch_felt_on_both_sides_is_one_onset_with_both_bodies_and_regions(crib):
    env = crib
    _fresh(env)
    _report(env, [])
    for _ in range(E.ONSET_ABSENT_FRAMES):
        _report(env, [])
    point = _eye_point(env, 5.0)
    both = [_rec(env, "right_ffdistal", "left_hand", point=point, force=2.0),
            _rec(env, "left_hand", "right_ffdistal", point=point, force=2.0)]
    t = _report(env, both)
    assert len(t["self_contact_onsets"]) == 1, t["self_contact_onsets"]
    o = t["self_contact_onsets"][0]
    assert o["pair"] == ["left_hand", "right_ffdistal"]
    assert o["regions"] == ["left hand", "right hand"]
    assert o["structural"] is False
    assert o["in_view"] is True and o["view_deg"] < 6.0
    # still touching next frame: not a new onset
    assert _report(env, both)["self_contact_onsets"] == []


def test_a_reset_clears_the_tracker_so_a_pair_seen_late_in_the_old_episode_can_start_again(crib):
    """review env-senses MINOR 1: reset() must forget the old episode's last-seen frames. The
    frame counter returns to 0 at a reset, so a pair last seen at frame 5,001 of the old episode
    would read as seen "in the future" and could not report an onset until the new episode
    passed frame 5,006. A reset that keeps the tracker (the review's mutant) fails here."""
    env = crib
    _fresh(env)
    env.frame = 5000
    pair = [_rec(env, "right_ffdistal", "left_hand"), _rec(env, "left_hand", "right_ffdistal")]
    _report(env, pair)                                  # tracking starts at frame 5,000, touching
    _report(env, pair)                                  # still touching at frame 5,001
    env.reset()                                         # frame 0; the reset's own frame is tracked
    assert env.frame == 0 and env._contact_last_seen is not None
    assert ("s", "left_hand", "right_ffdistal") not in env._contact_last_seen, \
        "the reset kept the old episode's last-seen frame for the pair"
    for _ in range(E.ONSET_ABSENT_FRAMES):
        _report(env, [])                                # apart for 5 frames
    t = _report(env, pair)                              # touching again
    assert [o["pair"] for o in t["self_contact_onsets"]] == [["left_hand", "right_ffdistal"]], \
        t["self_contact_onsets"]


def test_contact_chattering_off_for_fewer_than_five_frames_is_not_a_new_onset(crib):
    env = crib
    _fresh(env)
    pair = [_rec(env, "right_ffdistal", "left_hand"), _rec(env, "left_hand", "right_ffdistal")]
    _report(env, pair)                                  # tracking starts, touching
    for _ in range(E.ONSET_ABSENT_FRAMES - 1):
        _report(env, [])                                # apart 4 frames
    assert _report(env, pair)["self_contact_onsets"] == []
    for _ in range(E.ONSET_ABSENT_FRAMES):
        _report(env, [])                                # apart 5 frames
    assert len(_report(env, pair)["self_contact_onsets"]) == 1


def test_a_touch_out_of_sight_is_not_in_view(crib):
    env = crib
    _fresh(env)
    _report(env, [])
    for _ in range(E.ONSET_ABSENT_FRAMES):
        _report(env, [])
    point = _eye_point(env, 120.0)
    t = _report(env, [_rec(env, "right_foot", "left_foot", point=point)])
    (o,) = t["self_contact_onsets"]
    assert o["in_view"] is False and o["view_deg"] > E.IN_VIEW_DEG
    assert o["regions"] == ["left foot", "right foot"]


def test_outside_onsets_are_counted_and_every_patch_says_what_touches_it(crib):
    env = crib
    _fresh(env)
    _report(env, [])
    for _ in range(E.ONSET_ABSENT_FRAMES):
        _report(env, [])
    recs = [_rec(env, "right_hand", "world", is_self=False),          # floor on the palm
            _rec(env, "right_hand", "right_thhub"),                   # and the thumb in it
            _rec(env, "chest", "world", is_self=False),               # floor on the chest
            _rec(env, "left_ffdistal", "left_mfdistal")]              # one finger on another
    t = _report(env, recs)
    names = env.patch_names()
    state = dict(zip(names, t["contact_patch_state"]))
    assert state["right_hand"] == 3 and state["chest"] == 1 and state["left_ffdistal"] == 2
    assert state["left_upper_leg"] == 0
    assert len(t["contact_patch_state"]) == env.n_patches
    assert t["contact_onsets_world"] == 2
    assert all(isinstance(x, int) for x in t["contact_patch_state"]), "must stay JSON-serialisable"


def test_the_resting_thumb_in_the_palm_is_measured_structural(crib):
    """The pair that touches at rest under zero command, found on the real body: the thumb
    hub in the palm of the same hand (measured 2026-09-22, STRUCTURAL_MEASUREMENT)."""
    env = crib
    assert ("left_hand", "left_thhub") in E.STRUCTURAL_SELF_PAIRS
    assert ("right_hand", "right_thhub") in E.STRUCTURAL_SELF_PAIRS
    assert E.STRUCTURAL_MEASUREMENT.get("date"), "the structural set has no recorded measurement"
    env.reset()
    found = None
    for _ in range(300):
        env.step(ctrl=np.zeros(env.model.nu))
        cs = env._contacts()
        thumb = [c for c in cs if c.other_is_self
                 and {env._body_names[int(c.body_id)], c.other_body_name}
                 in ({"left_hand", "left_thhub"}, {"right_hand", "right_thhub"})]
        if thumb:
            found = cs
            break
    assert found is not None, "at rest the thumb never touched its palm (the premise moved)"
    # the same real contacts, as if the pair had been apart long enough to be an onset
    env._contact_last_seen = {}
    env._contact_track_start = env.frame - 10 * E.ONSET_ABSENT_FRAMES
    task = {}
    env._contact_report(found, task)
    thumbs = [o for o in task["self_contact_onsets"] if "thhub" in "".join(o["pair"])]
    assert thumbs and all(o["structural"] for o in thumbs), task["self_contact_onsets"]


def test_structural_means_touching_in_every_measured_rest_and_both_hands_are_treated_alike():
    """review env-senses MINOR 5: the list used to hold every pair seen in 10 s after one reset,
    so right-hand finger-to-finger touches were never scored and the left hand's always were."""
    m = E.STRUCTURAL_MEASUREMENT
    assert m.get("rule") and m.get("date") == "2026-09-23"
    counts = m["rests_touching_of_7"]
    assert E.STRUCTURAL_SELF_PAIRS == frozenset(p for p, n in counts.items() if n == 7)
    assert E.STRUCTURAL_SELF_PAIRS == {("left_hand", "left_thhub"), ("right_hand", "right_thhub")}
    for a, b in E.STRUCTURAL_SELF_PAIRS:
        mirror = tuple(sorted(x.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
                              for x in (a, b)))
        assert mirror in E.STRUCTURAL_SELF_PAIRS, (a, b)
    for pair in (("right_ffdistal", "right_mfmiddle"), ("right_ffdistal", "right_mfdistal"),
                 ("left_ffdistal", "left_mfdistal")):
        assert pair not in E.STRUCTURAL_SELF_PAIRS, "a finger-to-finger touch must be scored"


def test_a_scripted_self_touch_on_the_real_body_is_an_onset_with_both_bodies(crib):
    """Put the right arm where its hand touches another part of the body (set by joint
    angles, then MuJoCo finds the contact), after frames with the arm away."""
    import mujoco
    env = crib
    env.reset()
    m, d = env.model, env.data
    joints = [j for j in range(m.njnt) if (m.joint(j).name or "").startswith("robot:right_")
              and any(k in m.joint(j).name for k in ("shoulder", "elbow", "hand"))]
    adr = [int(m.jnt_qposadr[j]) for j in joints]
    rng_ = np.random.default_rng(7)
    base = d.qpos.copy()
    touching = None
    for _ in range(3000):
        d.qpos[:] = base
        for j, a in zip(joints, adr):
            lo, hi = m.jnt_range[j]
            d.qpos[a] = rng_.uniform(lo, hi)
        mujoco.mj_forward(m, d)
        hits = [c for c in env._contacts() if c.other_is_self
                and E.contact_region(env._body_names[int(c.body_id)]) == "right hand"
                and E.contact_region(c.other_body_name) not in ("right hand", "right lower arm", "")]
        if hits:
            touching = (d.qpos.copy(), hits[0])
            break
    assert touching is not None, "no arm pose put the right hand on another body part"
    d.qpos[:] = base
    mujoco.mj_forward(m, d)
    _fresh(env)
    for _ in range(E.ONSET_ABSENT_FRAMES + 1):
        task = {}
        env._contact_report(env._contacts(), task)
        env.frame += 1
    d.qpos[:] = touching[0]
    mujoco.mj_forward(m, d)
    task = {}
    env._contact_report(env._contacts(), task)
    hit = touching[1]
    want = sorted([env._body_names[int(hit.body_id)], hit.other_body_name])
    got = [o for o in task["self_contact_onsets"] if o["pair"] == want]
    assert len(got) == 1, (want, task["self_contact_onsets"])
    assert got[0]["regions"] == [E.contact_region(want[0]), E.contact_region(want[1])]
    assert "right hand" in got[0]["regions"]
    d.qpos[:] = base
    mujoco.mj_forward(m, d)


def test_the_tracker_survives_a_full_state_round_trip_and_an_old_snapshot_restarts_it(crib):
    env = crib
    env.reset()
    for _ in range(3):
        env.step(ctrl=np.zeros(env.model.nu))
    snap = env.save_full_state()
    assert "_contact_last_seen" in snap["attrs"] and snap["attrs"]["_contact_last_seen"] is not None
    kept = dict(snap["attrs"]["_contact_last_seen"])
    env.step(ctrl=np.zeros(env.model.nu))
    env.restore_full_state(snap)
    assert env._contact_last_seen == kept
    # a state saved before the report existed has no tracker: the next frame starts it
    old = dict(snap)
    old["attrs"] = {k: v for k, v in snap["attrs"].items()
                    if k not in ("_contact_last_seen", "_contact_track_start")}
    env.restore_full_state(old)
    assert env._contact_last_seen is None
    t = env.step(ctrl=np.zeros(env.model.nu)).info["task"]
    assert t["self_contact_onsets"] == [] and t["contact_onsets_world"] == 0


def test_a_model_steps_frames_merge_without_losing_an_onset():
    one = {"pair": ["left_hand", "right_ffdistal"], "regions": ["left hand", "right hand"],
           "structural": False, "in_view": True, "view_deg": 4.0}
    frames = [{"self_contact_onsets": [], "contact_onsets_world": 1, "contact_patch_state": [0, 1, 0]},
              {"self_contact_onsets": [one], "contact_onsets_world": 0, "contact_patch_state": [2, 0, 0]},
              {"self_contact_onsets": [], "contact_onsets_world": 2, "contact_patch_state": [0, 0, 0]}]
    out = E.merge_burst_contact_report(frames)
    assert out["self_contact_onsets"] == [one]
    assert out["contact_onsets_world"] == 3
    assert out["contact_patch_state_burst"] == [2, 1, 0]
    assert E.merge_burst_contact_report([{}, {"hand_pressing": True}]) == {}, \
        "frames that never measured onsets must not report zero onsets"
