# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Read-only counterexamples for the October 12 observer audit; no simulated worlds."""
from types import SimpleNamespace

import numpy as np
import pytest

from mimo_bridge.tools.observer import detectors as D


def test_contact_cannot_borrow_hand_motion_from_after_its_onset(monkeypatch):
    palm = np.zeros((7, 2, 3))
    palm[:, 0, 0] = [0.0, 0.0, 0.0, 0.01, 0.04, 0.08, 0.10]
    obj = np.zeros((7, 1, 3))
    obj[:, 0, 0] = [0.12, 0.10, 0.02, 0.03, 0.06, 0.10, 0.12]
    tr = SimpleNamespace(palm=palm, obj_pos=obj,
                         ent=SimpleNamespace(objects=[dict(kind="box", name="box")], object_of_entity={2: 0}),
                         seconds=lambda i, j: (j - i) * 0.05)
    approach = dict(k=0, side="L", i0=0, i1=6, start_m=0.12, closed_m=0.10)
    contact = dict(other=2, i0=2, i1=6, peak_force_n=1.0)
    monkeypatch.setattr(D, "_hand_onsets", lambda ctx, side: [contact] if side == "L" else [])
    ctx = SimpleNamespace(tr=tr, by_type={"approach": [approach]}, det=lambda *args, **kw: kw)
    assert D.R02(ctx) == []
    # Same eventual movement, but completed before contact: retain the positive case.
    palm[2, 0, 0], obj[2, 0, 0] = 0.10, 0.12
    assert len(D.R02(ctx)) == 1


def test_contact_requires_the_existing_approach_distance_by_contact(monkeypatch):
    palm = np.zeros((7, 2, 3))
    palm[:, 0, 0] = np.linspace(0.0, 0.06, 7)
    obj = np.zeros((7, 1, 3))
    obj[:, 0, 0] = 0.12
    tr = SimpleNamespace(palm=palm, obj_pos=obj,
                         ent=SimpleNamespace(objects=[dict(kind="box", name="box")], object_of_entity={2: 0}),
                         seconds=lambda i, j: (j - i) * 0.05)
    a = dict(k=0, side="L", i0=0, i1=6, start_m=0.12, closed_m=0.06)
    c = dict(other=2, i0=1, i1=6, peak_force_n=1.0)
    monkeypatch.setattr(D, "_hand_onsets", lambda ctx, side: [c] if side == "L" else [])
    ctx = SimpleNamespace(tr=tr, by_type={"approach": [a]}, det=lambda *args, **kw: kw)
    assert D.R02(ctx) == []
    c["i0"] = 5
    assert len(D.R02(ctx)) == 1


def _two_hands(states, disjoint=False):
    states = np.asarray(states)
    n = len(states)
    left, right = (states & 1) != 0, (states & 2) != 0
    rows, entities, frames = [], [], []
    for i in range(n):
        for side, on in enumerate((left, right)):
            if on[i]:
                rows.append(i)
                entities.append([side, 2])
                frames.append(([0, 1] if side == 0 else [3, 4]) if disjoint else [0, 4])
    tr = SimpleNamespace(n=n, continuous=np.r_[False, np.ones(n - 1, dtype=bool)],
                         ent=SimpleNamespace(objects=[dict(kind="box", name="box", free=True)],
                                             hand={"L": np.array([True, False, False]),
                                                   "R": np.array([False, True, False])}),
                         object_mask=lambda k: np.array([False, False, True]),
                         c_row=np.asarray(rows), c_e=np.asarray(entities), c_frame=np.asarray(frames),
                         c_fmean=np.ones(len(rows)), c_normal=np.tile([1.0, 0.0, 0.0], (len(rows), 1)),
                         palm_speed=np.tile([0.0, 0.1], (n, 1)),
                         obj_quat=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1, 1)),
                         seconds=lambda i, j: (j - i) * 0.05)
    return SimpleNamespace(tr=tr, hand_obj=lambda side, k: {"on": left if side == "L" else right},
                           obj_floor=lambda k: {"on": np.zeros(n, dtype=bool)}, det=lambda *args, **kw: kw)


def test_two_hand_use_requires_overlapping_contact_frame_ranges():
    assert D.O08(_two_hands([3] * 8, disjoint=True)) == []
    assert len(D.O08(_two_hands([3] * 8))) == 1
    assert D.O07(_two_hands([1] * 4 + [3] * 4 + [2] * 6, disjoint=True)) == []
    assert len(D.O07(_two_hands([1] * 4 + [3] * 4 + [2] * 6))) == 1


def test_transfer_cannot_bridge_a_record_gap_in_either_later_phase():
    ctx = _two_hands([1] * 4 + [3] * 4 + [2] * 6)
    assert len(D.O07(ctx)) == 1
    ctx.tr.continuous[6] = False
    assert D.O07(ctx) == []
    ctx.tr.continuous[6] = True
    ctx.tr.continuous[10] = False
    assert D.O07(ctx) == []


def test_object_and_palm_geometry_share_the_restored_snapshot():
    """Only forward kinematics of four tiny bodies; no integration or training."""
    mj = pytest.importorskip("mujoco")
    from mimo_bridge.tools.observer.features import Track
    from mimo_bridge.tools.observer.kinematics import Kinematics
    from mimo_bridge.tests.test_observer_record_v2 import _TinyKin
    model = mj.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="palm"><freejoint/><geom size=".01" mass="1"/></body>
      <body name="free"><freejoint/><geom size=".01" mass="1"/></body>
      <body name="mocap" mocap="true"><geom size=".01"/></body>
      <body name="mounted"><geom size=".01"/></body>
    </worldbody></mujoco>''')
    data = mj.MjData(model)
    roots = [model.body(name).id for name in ("free", "mocap", "mounted")]
    native = Kinematics.__new__(Kinematics)
    native.m, native.d, native.mj, native.mounts = model, data, mj, [(2, roots[2])]

    class Geometry(_TinyKin):
        def place(self, qpos, mocap=None, object_pose=None):
            native.place(qpos, mocap, object_pose)

        def measure(self):
            out = super().measure()
            out.update(xpos=data.xpos.copy(), xquat=data.xquat.copy(),
                       palm=np.tile(data.xpos[model.body("palm").id], (2, 1)))
            return out

    tr = Track.__new__(Track)
    tr.n, tr.kin = 1, Geometry()
    tr.qpos = model.qpos0[None].copy()
    tr.qpos[0, 0], tr.qpos[0, 7] = 0.1, 0.2
    tr.qpos[0, 10:14] = [np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)]
    tr.mocap = np.array([[[0.3, 0, 0, 1, 0, 0, 0]]])
    tr.object_pose = np.array([[[0.5, 0, 0, 1, 0, 0, 0], [0.6, 0, 0, 1, 0, 0, 0],
                                [0.4, 0, 0, 1, 0, 0, 0]]])
    tr.header = {"joints": []}
    tr.ent = SimpleNamespace(objects=[dict(root=int(r)) for r in roots])
    tr.sim_time, tr.episode, tr.step = np.array([0.05]), np.array([0]), np.array([1])
    tr._measure()
    np.testing.assert_allclose(tr.obj_pos[0, :, 0], [0.2, 0.3, 0.4])
    np.testing.assert_allclose(tr.obj_pos[0, :, 0] - tr.palm[0, 0, 0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(tr.obj_quat[0, 0], tr.qpos[0, 10:14])
    # Preserve archived source poses; this correction only changes analysis geometry.
    np.testing.assert_allclose(tr.object_pose[0, :, 0], [0.5, 0.6, 0.4])
