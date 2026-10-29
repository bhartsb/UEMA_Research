# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""tools/command_following.py reads position only where a position was ordered, and adds readings a
person can picture: parts per 100 that ended nearer the ordered orientation, and the share of the
ordered turn's degrees closed.

What was wrong (found 2026-09-23 while re-measuring the anchor for fix-rows-r1-launch_h84): env.py's
gauge reports pos_cos 0.0 when no part was ordered to MOVE, and only the wrist trials order a 5 cm
step. The tool kept every pos_cos, so 26 of every 30 trials went into position_match_mean as a match
of 0. The 2026-09-18 anchor reads 0.02-0.12 in every cell for that reason, and the gain-of-function
tree (X6) read it as "position following is weak at every gain".

The tool builds trunk_v1 worlds only (it orders the chest and the lower trunk, rows the 24-row legacy
layout does not have), so there is no legacy case. A stand-in world replaces MuJoCo: each step turns
every ordered row a fixed fraction of the remaining way toward (or away from) its ordered orientation,
and its gauge reports pos_cos only when a row's ordered position differs from where the row is, as
env._command_response does.
"""
import math
import sys
import types

import numpy as np
import pytest

from mimo_bridge.body_layout import names_for
from mimo_bridge.tools import command_following as cf

POS_COS = 0.3          # the stand-in gauge's position match on a trial that ordered a position step


def _quat_wxyz_from_xyzw(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _stand_in_world(monkeypatch, step_fraction):
    """A stand-in mimo_bridge.env.BridgeEnv. Each step moves every row with strength above 0 by
    `step_fraction` of the remaining rotation toward its ordered orientation (negative: away from it).
    Its gauge scores every ordered row but the eyes' row; each row's direction match is
    0.5 + 0.02 x row index, so cells have a spread to put an interval on."""
    built = []

    class FakeEnv:
        def __init__(self, **kw):
            assert kw["body_layout"] == "trunk_v1"
            self.names = list(names_for("trunk_v1"))
            rows = len(self.names)
            self.proprio = types.SimpleNamespace(body_ids=np.arange(rows), handle_names=tuple(self.names))
            self.data = types.SimpleNamespace(xpos=np.zeros((rows, 3)),
                                              xquat=np.tile([1.0, 0.0, 0.0, 0.0], (rows, 1)))
            built.append(self)

        def reset(self):
            pass

        def close(self):
            pass

        def step(self, order, *_a, render=False):
            gaze = self.names.index("gaze")
            ordered = [i for i in range(len(self.names)) if order[i, 8] > 0 and i != gaze]
            moved = []
            for i in ordered:
                q0 = self.data.xquat[i].copy()
                t = _quat_wxyz_from_xyzw(order[i, 3:7])
                if np.dot(q0, t) < 0:
                    t = -t
                # rotation from q0 to t, scaled: q = q0 * (q0^-1 t)^f (both unit quaternions)
                w, x, y, z = q0
                inv = np.array([w, -x, -y, -z])
                d = _mul(inv, t)
                ang = 2.0 * math.acos(min(1.0, abs(d[0])))
                if ang > 1e-9:
                    axis = d[1:] / max(np.linalg.norm(d[1:]), 1e-12)
                    a = ang * step_fraction
                    r = np.array([math.cos(a / 2), *(axis * math.sin(a / 2))])
                    self.data.xquat[i] = _mul(q0, r)
                e = order[i, 0:3] / 100.0 - self.data.xpos[i]
                moved.append(np.linalg.norm(e) > 1e-3)
            n = len(ordered)
            cr = {"cos": float(np.mean([0.5 + 0.02 * i for i in ordered])) if n else 0.0,
                  "ratio": 0.5 if n else 0.0, "n": n,
                  "pos_cos": POS_COS if any(moved) else 0.0}          # env: 0.0 when nothing moved
            return types.SimpleNamespace(info={"cmd_response_slow": cr, "decoded_ctrl": np.zeros(8)})

    mod = types.ModuleType("mimo_bridge.env")
    mod.BridgeEnv = FakeEnv
    monkeypatch.setitem(sys.modules, "mimo_bridge.env", mod)
    return built


def _mul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([aw*bw - ax*bx - ay*by - az*bz,
                     aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw])


def _run(trials=30, hold=25):
    return cf.run(6.0, 0.5, trials, 20.0, np.random.default_rng(0), "crib", 24.0, hold, quiet=True)


def test_the_position_match_is_read_only_on_trials_that_ordered_a_position_step(monkeypatch):
    _stand_in_world(monkeypatch, step_fraction=0.1)
    out = _run()
    groups = list(cf.HANDLE_GROUPS)
    wrist_trials = sum(1 for t in range(30) if groups[t % len(groups)].startswith("wrist"))
    assert wrist_trials == 4, "the premise: 4 of 30 trials order a 5 cm step"
    # before: the 26 trials without a step went in as 0, so the cell read POS_COS x 4 / 30
    assert out["position_match_mean"] == pytest.approx(POS_COS)


def test_every_part_that_turns_toward_its_order_counts_as_closer(monkeypatch):
    _stand_in_world(monkeypatch, step_fraction=0.1)
    hold = 25
    out = _run(hold=hold)
    assert out["closer_parts"] == 30
    assert out["closer_out_of_100"] == pytest.approx(100.0)
    # each step closes 10% of what is left: after 25 steps 1 - 0.9^25 of the 20 degrees
    assert out["gap_closed_pct"] == pytest.approx(100.0 * (1.0 - 0.9 ** hold), abs=0.05)


def test_a_part_that_turns_away_counts_as_farther_and_the_gap_grows(monkeypatch):
    _stand_in_world(monkeypatch, step_fraction=-0.02)
    out = _run()
    assert out["closer_out_of_100"] == pytest.approx(0.0)
    assert out["gap_closed_pct"] < 0.0, "ending farther from the order is a negative share closed"


def test_the_direction_match_carries_a_95_percent_interval(monkeypatch):
    _stand_in_world(monkeypatch, step_fraction=0.1)
    out = _run()
    names = list(names_for("trunk_v1"))
    groups = list(cf.HANDLE_GROUPS)
    per_trial = [0.5 + 0.02 * names.index(cf.HANDLE_GROUPS[groups[t % len(groups)]][0]) for t in range(30)]
    m, s = np.mean(per_trial), np.std(per_trial, ddof=1)
    assert out["direction_match_mean"] == pytest.approx(m)
    lo, hi = out["direction_match_ci95"]
    assert lo == pytest.approx(m - 1.96 * s / math.sqrt(30))
    assert hi == pytest.approx(m + 1.96 * s / math.sqrt(30))
