# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The world's following gauge picks the eyes' row by NAME and scores each order over one tenure.

What was wrong (found 2026-09-23, docs/gain_of_function_v3/measurements/following_out_of_100.json):
env._command_response took the LAST row of the body layout as the eyes' row. In the 24-row legacy
layout the last row is the eyes; in trunk_v1 (26 rows, lower trunk and chest after the eyes, every
run from H71 on and all 50 H84 segments) the last row is the CHEST. So the gauge (cmd_response and
cmd_response_slow, the source of cmd_response_cos, cmd_response_cos_held and follow_bout) never
scored the chest and scored the eyes' head-relative order against the eye body's world rotation.
tools/whole_body_following.py repeated it. Separately, the slow gauge waited for cmd_horizon + 1
queued orders, so each order was scored over 26 frames (260 ms) where its tenure is 25.

These tests build the real world (crib, vision off, no caregiver) under BOTH layouts.
"""
import math

import numpy as np
import pytest

from mimo_bridge.body_layout import names_for
from mimo_bridge.env import BridgeEnv
from mimo_bridge.tools.command_following import _turn, rest_order

TURN = math.radians(20.0)


def _env(tmp_path, layout, horizon=25):
    return BridgeEnv(scene="crib", run_dir=str(tmp_path), vision=False, tag=f"_rows_{layout}", age=24,
                     contact_profile="mechanical_v4", caregiver_seconds=0, kp=6.0,
                     body_layout=layout, cmd_horizon=horizon)


def _order_one(env, names, row, rng, strength=0.5):
    """Every row asked for its current pose at strength 0, except `row`: turned 20 degrees about a
    random axis at the given strength (the eyes' row is turned from the eye body's world pose, so
    the old code, which scored that row in trunk_v1, had a turn above its 1e-4 rad line to score)."""
    order = rest_order(env, len(names))
    order[:, 8] = 0.0
    order[row, 3:7] = _turn(order[row, 3:7], rng.normal(size=3), TURN)
    order[row, 8] = strength
    return order


@pytest.mark.parametrize("layout", ["legacy", "trunk_v1"])
def test_the_gauge_skips_exactly_the_eyes_row_by_name_and_scores_every_other_row(tmp_path, layout):
    names = list(names_for(layout))
    env = _env(tmp_path, layout)
    try:
        env.reset()
        assert list(env.proprio.handle_names) == names
        gaze = names.index("gaze")
        assert gaze == 23, "the eyes are row 23 in both layouts"
        if layout == "trunk_v1":
            assert names[-1] == "chest" and len(names) == 26, "the premise: the last row is the chest"
        else:
            assert names[-1] == "gaze" and len(names) == 24
        rng = np.random.default_rng(0)
        scored = {}
        for row, name in enumerate(names):
            out = env.step(_order_one(env, names, row, rng), None, None, render=False)
            scored[name] = int(out.info["cmd_response"]["n"])
        assert scored["gaze"] == 0, "the eyes' order is scored by the eye servo's own path, never here"
        assert all(scored[n] == 1 for n in names if n != "gaze"), scored
        if layout == "trunk_v1":
            assert scored["chest"] == 1 and scored["lower_trunk"] == 1

        # the window loop: the commanded rotation it stores for the eyes' row is zero, the chest's
        # (trunk_v1) or the last body row's (legacy) is not, with every row ordered at once
        order = rest_order(env, len(names))
        for i in range(len(names)):
            order[i, 3:7] = _turn(order[i, 3:7], rng.normal(size=3), TURN)
        order[:, 8] = 0.5
        out = env.step(order, None, None, render=False)
        asked = env._cmd_hist[-1][0]
        assert np.all(asked[gaze] == 0.0)
        assert all(np.linalg.norm(asked[i]) > 0.3 for i in range(len(names)) if i != gaze)
        assert int(out.info["cmd_response"]["n"]) == len(names) - 1
    finally:
        env.close()


def test_whole_body_following_weights_skip_the_eyes_and_score_the_chest(tmp_path):
    from mimo_bridge.tools.whole_body_following import _weighted_cos
    names = list(names_for("trunk_v1"))
    env = _env(tmp_path, "trunk_v1")
    try:
        env.reset()
        rng = np.random.default_rng(1)
        ids = env.proprio.body_ids
        for name, scored in (("gaze", False), ("chest", True)):
            order = _order_one(env, names, names.index(name), rng)
            xq0 = np.array([env.data.xquat[b] for b in ids])
            for _ in range(25):
                env.step(order, None, None, render=False)
            wcos, mag = _weighted_cos(env, order, xq0, names)
            assert (wcos is not None) == scored, (name, wcos)
            assert (mag > 0.0) == scored
    finally:
        env.close()


@pytest.mark.parametrize("horizon", [25, 5, 1])
def test_the_slow_gauge_scores_each_order_over_exactly_one_tenure(tmp_path, horizon):
    """A new order on EVERY frame, so each frame's slow reading can only belong to one order. The
    reading on frame k must be the gauge of the order entered on frame k - horizon + 1, from the pose
    before that frame to the pose after frame k: exactly `horizon` frames of physics."""
    names = list(names_for("trunk_v1"))
    env = _env(tmp_path, "trunk_v1", horizon)
    try:
        env.reset()
        rng = np.random.default_rng(2)
        ids = env.proprio.body_ids
        rec = []
        checked = 0
        for k in range(horizon + 12):
            order = rest_order(env, len(names))
            for i in range(len(names)):
                order[i, 3:7] = _turn(order[i, 3:7], rng.normal(size=3), TURN)
            order[:, 8] = 0.5
            rec.append((order, np.array(env.data.xpos[ids]), np.array(env.data.xquat[ids]),
                        float(env.data.time)))
            out = env.step(order, None, None, render=False)
            slow = out.info.get("cmd_response_slow")
            if k < horizon - 1:
                assert slow is None, f"frame {k}: no order has been in force for {horizon} frames yet"
                continue
            assert slow is not None, f"frame {k}: the order entered on frame {k - horizon + 1} is of age"
            j = k - horizon + 1
            frames = round((float(env.data.time) - rec[j][3]) / env.dt)
            assert frames == horizon, f"the scored span is {frames} frames, one tenure is {horizon}"
            ref = env._command_response(rec[j][0], rec[j][1], rec[j][2], np.zeros((len(ids), 3)),
                                        record_window=False)
            for key in ("cos", "ratio", "n", "pos_cos", "cmd_mag"):
                assert slow[key] == pytest.approx(ref[key], abs=1e-12), (k, key)
            if j >= 1:       # and not the order one frame older (the 26-frame pairing)
                older = env._command_response(rec[j - 1][0], rec[j - 1][1], rec[j - 1][2],
                                              np.zeros((len(ids), 3)), record_window=False)
                assert slow["cos"] != pytest.approx(older["cos"], abs=1e-9)
            checked += 1
        assert checked == 13
    finally:
        env.close()


def test_a_queue_saved_before_the_fix_is_trimmed_to_one_tenure(tmp_path):
    """A world saved before 2026-09-23 holds cmd_horizon queued orders (the old code popped at
    cmd_horizon + 1). The first step after restore must drop the surplus, not keep a 26-frame span."""
    names = list(names_for("trunk_v1"))
    H = 5
    env = _env(tmp_path, "trunk_v1", H)
    try:
        env.reset()
        rng = np.random.default_rng(3)
        ids = env.proprio.body_ids
        rec = []
        for k in range(H + 3):
            order = rest_order(env, len(names))
            for i in range(len(names)):
                order[i, 3:7] = _turn(order[i, 3:7], rng.normal(size=3), TURN)
            order[:, 8] = 0.5
            rec.append((order, np.array(env.data.xpos[ids]), np.array(env.data.xquat[ids])))
            env.step(order, None, None, render=False)
        assert len(env._cmd_delay) == H - 1
        # what the old code left behind: one entry more, the oldest one in front
        env._cmd_delay.insert(0, tuple(np.array(x, copy=True) for x in rec[1]))
        order = rest_order(env, len(names))
        order[:, 8] = 0.5
        out = env.step(order, None, None, render=False)
        assert len(env._cmd_delay) == H - 1
        j = len(rec) - H + 1          # the order entered H frames ago, counting this frame
        ref = env._command_response(rec[j][0], rec[j][1], rec[j][2], np.zeros((len(ids), 3)),
                                    record_window=False)
        assert out.info["cmd_response_slow"]["cos"] == pytest.approx(ref["cos"], abs=1e-12)
    finally:
        env.close()
