# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Each side's own footprint pressure on the hands, and the onset of a world press (2026-09-23).

Review env-senses MINOR 3: split_hand_press gives each side the whole patch's pressure times its
share of the force, so a side's force is spread over the other side's footprint too - the thumb
pressing into its own palm read a median 19% of its own pressure whenever that palm also lay on
something. The world now also reports each side's force over that side's own footprint area.

MINOR 4: hand_pressing_world is a STATE: it fired on 17 of every 100 frames for a body doing
nothing (a limp hand lying against a box) against 9 of 100 under random control. The world now
also reports the ONSET of a world press: a hand's outside-side pressure rising above the line
after at least 5 frames at or below it. The old keys do not move.
"""
import types

import numpy as np
import pytest

from mimo_bridge import env as E


# ------------------------------------------------------------------ the per-side pressure
GEOM_BODY = np.array([10, 11, 50, 12, 10, 13])      # geom -> body; 10 12 hand patches, 11 13 own


def _record(body, pair, force, area):
    return {"body_id": body, "pair": list(pair), "normal_force_n": force, "area_m2": area}


def test_each_side_reads_its_own_force_over_its_own_footprint():
    side = {10: "left", 12: "right"}
    own = {10, 11, 12, 13}
    recs = [_record(10, (0, 1), 1.0, 1e-4),        # the thumb on the palm: 1 N on 1 cm2 = 10 kPa
            _record(10, (4, 2), 2.0, 1e-3)]        # the floor under the palm: 2 N on 10 cm2 = 2 kPa
    out = E.footprint_hand_press(recs, side, own, GEOM_BODY)
    assert out["left"] == pytest.approx((2.0, 10.0))
    assert out["right"] == (0.0, 0.0)
    # the split on the same patch: 3 N over 11 cm2 = 2.73 kPa, a third of it own
    split = E.split_hand_press([3.0 / 1.1e-3], [1.0 / 3.0])
    assert split["hand_press_self_kpa"] == pytest.approx(0.909, abs=1e-3), "the split under-reads the thumb"
    assert split["hand_press_world_kpa"] == pytest.approx(1.818, abs=1e-3)


def test_several_records_on_one_side_of_one_patch_add_force_and_area():
    side = {10: "left", 12: "right"}
    own = {10, 11, 12, 13}
    recs = [_record(12, (3, 2), 1.0, 1e-4), _record(12, (3, 2), 3.0, 3e-4),      # 4 N / 4 cm2
            _record(12, (5, 3), 0.5, 1e-4),                                       # own: 5 kPa
            _record(11, (1, 2), 100.0, 1e-5),                                     # not a hand patch
            _record(10, (0, 2), 1.0, 0.0)]                                        # no area: unmeasured
    out = E.footprint_hand_press(recs, side, own, GEOM_BODY)
    assert out["right"] == pytest.approx((10.0, 5.0))
    assert out["left"] == (0.0, 0.0)


def test_the_keys_are_absent_without_a_footprint_skin():
    fake = types.SimpleNamespace(contact_profile="mechanical_v2", _virtual_skin=None)
    task = {}
    E.BridgeEnv._hand_press_footprint(fake, task)
    assert task == {}, "no per-contact areas under this profile: unmeasured, never a stand-in"


# ------------------------------------------------------------------ the onset rule
def _step(below, left, right, L=2.0, quiet=(E.ONSET_ABSENT_FRAMES,)):
    below, counts = E.press_onsets(below, {"left": left, "right": right}, L, quiet)
    return below, counts[0]


def test_an_onset_needs_five_frames_at_or_below_the_line_first():
    L = 2.0
    below, n = _step(None, 0.0, 9.0)
    assert n == 0, "the first frame cannot report an onset"
    for _ in range(E.ONSET_ABSENT_FRAMES - 1):
        below, n = _step(below, 0.0, 9.0)
        assert n == 0
    below, n = _step(below, 3.0, 9.0)
    assert n == 1, "left was below for 5 frames (tracking began below) and rose; right never fell"
    below, n = _step(below, 3.0, 9.0)
    assert n == 0, "a press that continues is not a new onset"
    for _ in range(E.ONSET_ABSENT_FRAMES - 1):
        below, n = _step(below, L, 0.0)                    # exactly on the line counts as below
    below, n = _step(below, 5.0, 5.0)
    assert n == 0, "4 frames below is chatter, not a new press"
    for _ in range(E.ONSET_ABSENT_FRAMES):
        below, n = _step(below, 0.0, 0.0)
    below, n = _step(below, 5.0, 5.0)
    assert n == 2, "both hands pressed anew"


def test_the_tenure_rule_needs_a_whole_order_below_the_line():
    """The second count: quiet for one order's tenure (the env's cmd_horizon, 25 frames on H84)."""
    quiet = (E.ONSET_ABSENT_FRAMES, 25)
    below, c = E.press_onsets(None, {"left": 0.0}, 2.0, quiet)
    for _ in range(9):                                     # 10 frames below
        below, c = E.press_onsets(below, {"left": 0.0}, 2.0, quiet)
    below, c = E.press_onsets(below, {"left": 9.0}, 2.0, quiet)
    assert c == (1, 0), "10 frames below: an onset by the 5-frame rule, not by the tenure rule"
    for _ in range(25):
        below, c = E.press_onsets(below, {"left": 0.0}, 2.0, quiet)
    below, c = E.press_onsets(below, {"left": 9.0}, 2.0, quiet)
    assert c == (1, 1)


def test_the_world_counts_the_tenure_rule_on_its_own_order_length(crib):
    env = crib
    assert env._cmd_horizon == 25, "the fixture holds one order for 25 frames, as H84 does"
    env.reset()
    env._press_below = {"left": 24, "right": 25}          # frames each hand has been below
    side = env._hand_press_side()
    left = next(b for b, s in side.items() if s == "left")
    right = next(b for b, s in side.items() if s == "right")
    floor = int(env.model.geom("floor").id)
    # both hands pressed by the floor at 10 kPa (1 N on 1 cm2) on the next report
    env._virtual_skin.contact_records = [
        {"body_id": b, "pair": [int(env.model.body_geomadr[b]), floor], "normal_force_n": 1.0,
         "area_m2": 1e-4} for b in (left, right)]
    task = {}
    env._hand_press_footprint(task)
    assert task["hand_press_world_footprint_kpa"] == pytest.approx(10.0)
    assert task["hand_press_world_onsets"] == 2 and task["hand_press_world_onsets_tenure"] == 1
    env.reset()


def test_the_burst_merge_adds_up_press_onsets_and_leaves_them_out_when_unmeasured():
    frames = [{"hand_press_world_onsets": 1, "hand_press_world_onsets_tenure": 0},
              {"hand_press_world_onsets": 0, "hand_press_world_onsets_tenure": 0},
              {"hand_press_world_onsets": 2, "hand_press_world_onsets_tenure": 1}]
    out = E.merge_burst_contact_report(frames)
    assert out["hand_press_world_onsets"] == 3 and out["hand_press_world_onsets_tenure"] == 1
    none = E.merge_burst_contact_report([{"hand_pressing": True}])
    assert "hand_press_world_onsets" not in none and "hand_press_world_onsets_tenure" not in none


# ------------------------------------------------------------------ the live body
@pytest.fixture(scope="module")
def crib(tmp_path_factory):
    env = E.BridgeEnv(scene="crib", run_dir=str(tmp_path_factory.mktemp("footprint")), vision=False,
                      tag="_footprinttest", age=24, contact_profile="mechanical_v4",
                      body_function="modulate", caregiver_seconds=0, kp=6.0,
                      body_layout="trunk_v1", cmd_horizon=25)
    env.reset()
    yield env
    env.close()


NEW_KEYS = ("hand_press_world_footprint_kpa", "hand_press_self_footprint_kpa",
            "hand_pressing_world_footprint", "hand_pressing_self_footprint", "hand_press_world_onsets")


def test_on_a_patch_touched_from_one_side_the_footprint_pressure_is_the_patchs_pressure(crib):
    """Where only one side touches a hand patch, that side's own pressure IS the patch's
    pressure (all the force over all the area), so the new key and the skin agree exactly."""
    env = crib
    env.reset()
    names = env.patch_names()
    hand = [i for i, nm in enumerate(names) if any(k in nm for k in E.HAND_PATCH_KEYS)]
    rng = np.random.default_rng(17)
    ctrl = np.zeros(env.model.nu)
    checked = 0
    for i in range(400):
        if i % 5 == 0:
            ctrl = rng.uniform(-0.5, 0.5, env.model.nu) if i >= 100 else np.zeros(env.model.nu)
        raw = env.step(ctrl=ctrl)
        t = raw.info["task"]
        for k in NEW_KEYS:
            assert k in t, k
        assert t["hand_pressing"] == (t["hand_press_kpa"] > 2.0), "the old key's rule moved"
        pressure = np.asarray(raw.info["physical_touch_audit"])[hand, 0] / 1000.0
        state = np.asarray(t["contact_patch_state"])[hand]
        if np.any(state == 3):
            continue                                  # a patch pressed by both: checked above
        world = pressure[state == 1].max() if np.any(state == 1) else 0.0
        own = pressure[state == 2].max() if np.any(state == 2) else 0.0
        assert t["hand_press_world_footprint_kpa"] == pytest.approx(world, rel=1e-5, abs=1e-9)
        assert t["hand_press_self_footprint_kpa"] == pytest.approx(own, rel=1e-5, abs=1e-9)
        assert t["hand_pressing_world_footprint"] == (t["hand_press_world_footprint_kpa"] > E.HAND_PRESS_LINE_KPA)
        checked += 1
    assert checked > 300


def test_a_reset_and_an_old_snapshot_restart_the_press_onset_tracker(crib):
    env = crib
    env.reset()
    assert env._press_below is not None, "the reset's own frame starts tracking"
    for _ in range(3):
        env.step(ctrl=np.zeros(env.model.nu))
    snap = env.save_full_state()
    assert "_press_below" in snap["attrs"]
    kept = dict(snap["attrs"]["_press_below"])
    env.step(ctrl=np.zeros(env.model.nu))
    env.restore_full_state(snap)
    assert env._press_below == kept
    old = dict(snap)
    old["attrs"] = {k: v for k, v in snap["attrs"].items() if k != "_press_below"}
    env.restore_full_state(old)
    assert env._press_below is None
    t = env.step(ctrl=np.zeros(env.model.nu)).info["task"]
    assert t["hand_press_world_onsets"] == 0
    env._press_below = {"left": 99, "right": 99}         # an old episode's long quiet stretch
    env.reset()
    assert set(env._press_below.values()) <= {0, 1}, \
        "the reset kept the old episode's count; its first frame must start a new one"
