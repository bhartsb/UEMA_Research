# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""R04 and H02 under detectors v9 (2026-09-25) on synthetic tracks: the grasp read as physics.

  * R04, the grip (detectors._grip): two surfaces of one hand - thumb and fingers, thumb and palm, or
    fingers and palm (a palmar grasp) - touching a free object in the SAME physics frames (c_frame
    ranges overlapping), their pushes at least OPPOSITION_MIN_DEG apart (the one angle exclusion) and
    balanced (what the stronger exceeds the weaker by, the object's weight, acceleration and the pair's
    friction must take), the object retained against gravity by the hand (everything else pushes it up
    with less than half its weight: not supported mainly by the floor, the chest, a forearm or the other
    hand) and moving with the palm; a track recorded without normals measures no R04, and the world's
    mocap toys are held by the world;
  * H02's support is contact (the floor, another object, the body but the hands and forearms) and, where
    the physics can be read, that upward force as well: a box squeezed between two palms rests on
    neither and is held by each hand when lifted; a box lying on the belly or on a forearm is supported
    by it; a box the floor still touches is not carried by a limp hand under it; a grip holds without
    carrying when the hand closed on the object.
"""
import numpy as np
import pytest

from mimo_bridge.tools.observer import detectors as D
from mimo_bridge.tools.observer.features import Entities, Track

_KINDS = ["self", "object", "caregiver", "floor", "wall", "art", "ceiling", "static"]
_ENTS = [dict(name="right_hand", kind="self", group="hand_R_palm"),
         dict(name="right_thdistal", kind="self", group="finger_R_thumb"),
         dict(name="right_ffdistal", kind="self", group="finger_R_index"),
         dict(name="left_hand", kind="self", group="hand_L_palm"),
         dict(name="left_thdistal", kind="self", group="finger_L_thumb"),
         dict(name="right_lower_arm", kind="self", group="arm_R_lower"),
         dict(name="upper_body", kind="self", group="trunk"),
         dict(name="head", kind="self", group="head"),
         dict(name="box_light", kind="object", group="box"),
         dict(name="floor", kind="floor", group="floor")]
PALM_R, THUMB_R, INDEX_R, PALM_L, THUMB_L, FOREARM_R, TRUNK, HEAD, BOX, FLOOR = range(10)
MASS = 0.05                      # kg: the box weighs 0.49 N; half of it is 0.245 N
UP, DOWN = (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)


class _Synth:
    """What Context, _holds, H02 and R04 read from a Track, for one box and two hands."""
    pair_series = Track.pair_series
    entity_mask = Track.entity_mask
    object_mask = Track.object_mask
    seconds = Track.seconds

    def __init__(self, n, normals=True, free=True):
        self.header = dict(entities=_ENTS, entity_kind=[_KINDS.index(e["kind"]) for e in _ENTS],
                           objects=[dict(entity=BOX, name="box_light", kind="box", root=1,
                                         mocap=-1 if free else 0, free=free, mass_kg=MASS)],
                           joints=[], weight_n=100.0)
        self.ent = Entities(self.header)
        self.n = n
        self.step = np.arange(n)
        self.sim_time = 0.05 * np.arange(n)
        self.dt = np.full(n, 0.05)
        self.dt[0] = np.nan
        self.continuous = np.isfinite(self.dt)
        z = np.zeros(n, dtype=bool)
        self.author = np.full(n, 1, dtype=np.int8)
        self.latched, self.rest, self.caregiver_contact, self.caregiver_present = z.copy(), z.copy(), z.copy(), z.copy()
        self.live = ~z
        self.order_seq = np.arange(n)
        self.toy_held = np.zeros(n, dtype=np.int8)
        self.roll_deg = np.zeros(n)
        self.sit_deg = np.full(n, 90.0)
        self.closure = np.full((n, 2), 25.0)
        self.palm = np.zeros((n, 2, 3))
        self.palm[:, 0] = (0.17, 0.20, 0.065)
        self.palm[:, 1] = (0.17, 0.32, 0.065)
        self.obj_pos = np.zeros((n, 1, 3))
        self.obj_pos[:, 0] = (0.17, 0.26, 0.031)
        self.has_normals = normals
        self.touch = []

    def contact(self, a, b, records, normal=(0.0, 0.0, 0.0), force=2.0, frames=(0, 4)):
        """a touches b over the records: normal, the direction a pushes b; force, the mean normal
        force (N); frames, the first and last physics frame of each record's burst it touched in."""
        self.touch += [(int(i), a, b, tuple(float(x) for x in normal), float(force), tuple(frames))
                       for i in records]

    def lift(self, start, cm=3.0, rise=4):
        """Box and both palms rise together by cm over `rise` records from `start`, then stay."""
        up = np.clip((np.arange(self.n) - start) / float(rise), 0.0, 1.0) * cm / 100.0
        self.obj_pos[:, 0, 2] += up
        self.palm[:, :, 2] += up[:, None]

    def done(self):
        vel = lambda x: np.concatenate([np.full((1,) + x.shape[1:], np.nan),
                                        np.diff(x, axis=0) / self.dt[1:].reshape((-1,) + (1,) * (x.ndim - 1))])
        self.palm_vel = vel(self.palm)
        self.palm_speed = np.linalg.norm(self.palm_vel, axis=-1)
        self.obj_vel = vel(self.obj_pos)
        self.obj_speed = np.linalg.norm(self.obj_vel, axis=-1)
        self.obj_ang_speed = np.zeros((self.n, 1))
        self.obj_quat = np.tile([1.0, 0.0, 0.0, 0.0], (self.n, 1, 1))
        t = sorted(self.touch)
        self.c_row = np.asarray([x[0] for x in t], dtype=np.int64)
        # rows keep the recorder's order: the lower entity code first, the normal from it to the other
        rows, normals = [], []
        for _, a, b, nv, _f, _fr in t:
            if a <= b:
                rows.append((a, b))
                normals.append(nv)
            else:
                rows.append((b, a))
                normals.append(tuple(-x for x in nv))
        self.c_e = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
        self.c_fmean = np.asarray([x[4] for x in t], dtype=np.float64)
        self.c_fmax = self.c_fmean.copy()
        self.c_frame = np.asarray([x[5] for x in t], dtype=np.int64).reshape(-1, 2)
        self.c_pos = np.asarray([self.obj_pos[x[0], 0] for x in t]).reshape(-1, 3)
        nm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        ln = np.linalg.norm(nm, axis=1, keepdims=True)
        self.c_normal = np.where(ln > 0, nm / np.maximum(ln, 1e-12), np.nan) if self.has_normals \
            else np.full((len(t), 3), np.nan)
        self._c_sorted = True
        return self


def _run(tr, *fns):
    ctx = D.Context(tr, [])
    return [d for fn in fns for d in fn(ctx)]


def _pinch(angle_deg, on_floor=False, normals=True, thumb_n=2.0, index_n=2.0, frames=((0, 4), (0, 4)),
           free=True):
    """Thumb and index on the box together for 20 records, their pushes angle_deg apart; the box in the
    air (nothing else touching it), or resting on the floor throughout."""
    tr = _Synth(30, normals, free)
    a = np.radians(angle_deg)
    tr.contact(THUMB_R, BOX, range(5, 25), (0.0, 1.0, 0.0), thumb_n, frames[0])
    tr.contact(INDEX_R, BOX, range(5, 25), (0.0, np.cos(a), np.sin(a)), index_n, frames[1])
    if on_floor:
        tr.contact(BOX, FLOOR, range(0, 30), DOWN)
    return tr.done()


def test_r04_is_an_opposed_grip_that_carries_the_object():
    assert D.OPPOSITION_MIN_DEG == 120.0 and D.GRIP_OTHERS_MAX_WEIGHT == 0.5
    for angle, found in ((180.0, True), (130.0, True), (119.0, False), (90.0, False), (0.0, False)):
        dets = _run(_pinch(angle), D.R04)
        assert bool(dets) is found, angle
        if found:
            m = dets[0]["measures"]
            assert m["opposition_deg"] == pytest.approx(angle, abs=1e-6)
            assert m["thumb_fingers_deg"] == pytest.approx(angle, abs=1e-6) and np.isnan(m["thumb_palm_deg"])
            assert m["opposed_share"] == 1.0 and m["grip_share"] == 1.0 and m["others_weight_share"] == 0.0
            assert dets[0]["grip_pair"] == "thumb-fingers"
    # the same squeeze of a box the floor carries: the floor pushes it up with 2 N, over half its 0.49 N
    # weight, so the hand does not retain it against gravity
    assert _run(_pinch(180.0, on_floor=True), D.R04) == []
    assert _run(_pinch(180.0, normals=False), D.R04) == [], "no normals recorded: R04 not measured"
    assert _run(_pinch(180.0, free=False), D.R04) == [], "a mocap toy is held by the world"


def test_r04_pairs_only_contacts_that_touched_in_the_same_frames():
    """The recorder gathers a record's contacts over its five physics frames: a thumb in frames 0-1 and
    a finger in frames 3-4 never touched together (the independent review's finding 3)."""
    assert _run(_pinch(180.0, frames=((0, 1), (3, 4))), D.R04) == []
    assert len(_run(_pinch(180.0, frames=((0, 2), (2, 4))), D.R04)) == 1


def test_r04_counts_fingers_closing_an_object_against_the_palm():
    """A palmar grasp: the fingers curl under the box and push it up into the palm, no thumb."""
    tr = _Synth(30)
    tr.contact(PALM_R, BOX, range(5, 25), DOWN)
    tr.contact(INDEX_R, BOX, range(5, 25), UP)
    dets = _run(tr.done(), D.R04)
    assert len(dets) == 1 and dets[0]["grip_pair"] == "fingers-palm"
    m = dets[0]["measures"]
    assert m["fingers_palm_deg"] == pytest.approx(180.0) and np.isnan(m["thumb_fingers_deg"])
    assert "finger_R_thumb" not in dets[0]["parts"]


def test_r04_judges_each_pair_of_surfaces_apart():
    """A palm pressing a face at right angles to the thumb does not hide a finger opposing it."""
    tr = _Synth(30)
    tr.contact(THUMB_R, BOX, range(5, 25), (0.0, 1.0, 0.0))
    tr.contact(INDEX_R, BOX, range(5, 25), (0.0, -1.0, 0.0))
    tr.contact(PALM_R, BOX, range(5, 25), DOWN, force=0.3)
    dets = _run(tr.done(), D.R04)
    assert len(dets) == 1 and dets[0]["measures"]["opposition_deg"] == pytest.approx(180.0)
    assert dets[0]["measures"]["thumb_palm_deg"] == pytest.approx(90.0)


def test_r04_needs_the_two_pushes_to_balance():
    """Straight against each other, what the stronger push exceeds the weaker by must be taken by the
    object's weight and acceleration (0.49 N here): a thumb pressing 3 N against a finger resting at
    0.5 N pushes the box, it does not squeeze it. At 130 degrees the pair's friction takes more."""
    assert _run(_pinch(180.0, thumb_n=3.0, index_n=0.5), D.R04) == []
    dets = _run(_pinch(180.0, thumb_n=2.0, index_n=1.8), D.R04)
    assert len(dets) == 1
    m = dets[0]["measures"]
    assert m["balance_used"] == pytest.approx(0.2 / (MASS * D.GRAVITY_MPS2), rel=1e-6)
    assert m["force_balance"] == pytest.approx(0.9)
    assert _run(_pinch(130.0, thumb_n=3.0, index_n=0.5), D.R04) == []       # 2.27 N against 1.97
    assert len(_run(_pinch(130.0, thumb_n=3.0, index_n=1.5), D.R04)) == 1  # 1.36 N against 2.39


def test_r04_needs_the_object_to_move_with_the_palm():
    """A box sliding out of the pinch 5 mm a record from the pinch's first record (0.049 m/s away from the
    palm) slips past GRIP_SLIP_MPS in every record it is pinched: each record's own burst moves it."""
    tr = _pinch(180.0)
    tr.obj_pos[5:25, 0, 2] -= 0.005 * (np.arange(20) + 1)
    tr.obj_pos[25:, 0, 2] = tr.obj_pos[24, 0, 2]
    assert _run(tr.done(), D.R04) == []


def test_a_box_held_between_two_palms_is_held_by_each_hand():
    """Each palm pushes sideways: once lifted off the floor nothing pushes the box up, so it rests on
    neither hand and each carries it."""
    for normals in (True, False):                   # by force, and by contact without normals
        tr = _Synth(40, normals)
        tr.contact(PALM_R, BOX, range(5, 35), (0.0, 1.0, 0.0))
        tr.contact(PALM_L, BOX, range(5, 35), (0.0, -1.0, 0.0))
        tr.contact(BOX, FLOOR, range(0, 11), DOWN)
        tr.lift(10)
        dets = _run(tr.done(), D.H02)
        assert sorted(d["sub"] for d in dets) == ["L", "R"], normals
        assert all(d["measures"]["carried_cm"] == pytest.approx(3.0, abs=1e-6) for d in dets)
        # a detection's numeric measures are stored as floats (Context.det), a flag as 1.0 or 0.0
        assert all(d["measures"]["physics_read"] == float(normals) for d in dets)


def test_a_box_resting_on_the_belly_moved_by_the_body_is_not_held():
    tr = _Synth(40)
    tr.contact(PALM_R, BOX, range(5, 35), DOWN)
    tr.contact(TRUNK, BOX, range(0, 40), UP)
    tr.lift(10)                                   # the body rises under it, the palm with it
    assert _run(tr.done(), D.H02) == []


def test_a_forearm_is_support_as_far_as_it_carries_the_box():
    """Lying ON the forearm (it pushes the box up with 2 N) the box is supported by it and the palm
    touching it carries nothing; pressed against the side of a forearm by the palm (the forearm pushes
    sideways), the forearm carries none of its weight and the lift is the hand's."""
    for forearm_push, held in ((UP, False), ((0.0, -1.0, 0.0), True)):
        tr = _Synth(40)
        tr.contact(PALM_R, BOX, range(5, 35), (0.0, 1.0, 0.0))
        tr.contact(FOREARM_R, BOX, range(5, 35), forearm_push)
        tr.contact(BOX, FLOOR, range(0, 11), DOWN)
        tr.lift(10)
        dets = _run(tr.done(), D.H02)
        assert [d["sub"] for d in dets] == (["R"] if held else []), forearm_push


def test_a_box_the_floor_still_touches_is_not_carried_by_a_hand_under_it():
    """By force alone a limp hand under a box that still touches the floor carries it (reset-world limp
    H02 0 -> 8, 2026-09-25 night); by contact the floor still holds it."""
    tr = _Synth(40)
    tr.contact(PALM_R, BOX, range(5, 35), UP, force=0.4)           # the palm under the box carries it
    tr.contact(BOX, FLOOR, range(0, 40), DOWN, force=0.05)         # the floor under it, barely
    slide = np.clip((np.arange(40) - 10) / 10.0, 0.0, 1.0) * 0.03
    tr.obj_pos[:, 0, 0] += slide
    tr.palm[:, :, 0] += slide[:, None]
    assert _run(tr.done(), D.H02) == []


def test_a_support_row_without_a_normal_counts_as_support():
    """Normals rebuilt for measuring leave a pair no longer touching at a record's last frame without one;
    read as no support, a limp hand's box sliding along the floor was 'carried' (2026-09-25 night)."""
    tr = _Synth(40)
    tr.contact(PALM_R, BOX, range(5, 35), DOWN)
    tr.contact(BOX, FLOOR, range(0, 40), (0.0, 0.0, 0.0))          # touching, its normal not known
    slide = np.clip((np.arange(40) - 10) / 10.0, 0.0, 1.0) * 0.03
    tr.obj_pos[:, 0, 0] += slide
    tr.palm[:, :, 0] += slide[:, None]
    assert _run(tr.done(), D.H02) == []


def test_the_dashboard_marks_that_the_grip_rows_exposure_begins_at_recorder_v2():
    """No lineage surgery (the coordinator, the evening of 2026-09-25): the exposure the page counts holds
    body-steps recorded before the recorder kept contact normals, and the rows that need them say so."""
    from mimo_bridge.tools.observer import view as VW
    note = "Observable exposure begins at recorder v2"
    for cid, marked in (("R04", True), ("H02", True), ("H03", True), ("H09", True), ("R06", True), ("R07", True),
                        ("H01", False), ("O05", False)):
        v = VW.row_view(dict(id=cid, name=cid, counted=0, exposure_steps=1000, stages={}), {})
        assert any(n.startswith(note) for n in v["notes"]) is marked, cid


def test_a_grip_that_closes_holds_without_carrying():
    """H02's other branch: a gripping record (a box pinched in the air) and the hand closing 10 degrees
    on it; the same closing on a box whose pushes meet at right angles is no hold."""
    for angle, found in ((180.0, True), (90.0, False)):
        tr = _pinch(angle)
        tr.closure[:, 1] = np.clip(15.0 + 1.0 * (np.arange(30) - 5), 15.0, 30.0)
        dets = _run(tr, D.H02)
        assert bool(dets) is found, angle
        if found:
            m = dets[0]["measures"]
            assert m["opposed"] and m["carried_cm"] == 0.0 and m["grip_records"] == 20


def test_a_grip_holds_only_when_the_hand_closes_on_it():
    """The same gripping pinch in the air with the hand STILL is no hold (H02's grip branch needs the hand to
    close HOLD_CLOSE_ON_DEG on the object: the limp baseline's 48 s false hold, 2026-09-24), while R04 still
    finds the grip; closing just under the line is no hold, just over it is. Without this test a grip that
    held without the hand closing survived every world-free test (the second review's planted fault F9)."""
    still = _pinch(180.0)
    assert np.ptp(still.closure[:, 1]) == 0.0
    assert len(_run(still, D.R04)) == 1, "the grip is there"
    assert _run(still, D.H02) == [], "a still hand around a gripped box does not hold it"
    for rise, held in ((D.HOLD_CLOSE_ON_DEG - 0.5, False), (D.HOLD_CLOSE_ON_DEG + 0.5, True)):
        tr = _pinch(180.0)
        tr.closure[:, 1] = 25.0 + np.clip((np.arange(30) - 8) / 8.0, 0.0, 1.0) * rise
        assert bool(_run(tr, D.H02)) is held, rise


def _one_record_lift(drop_in_own_record=False):
    """grasp_lift_no_retention's shape: the box on the floor under a thumb-index pinch (records 8-10), lifted 3 cm
    with the palm in record 9 only, back on the floor in record 10 while the palm stays up. With
    drop_in_own_record the box is already dropping (1.5 cm) at record 9's last frame."""
    tr = _Synth(20)
    tr.contact(BOX, FLOOR, [i for i in range(20) if i != 9], DOWN)
    tr.contact(THUMB_R, BOX, range(8, 11), (0.0, 1.0, 0.0))
    tr.contact(INDEX_R, BOX, range(8, 11), (0.0, -1.0, 0.0))
    tr.palm[9:, :, 2] += 0.03
    tr.obj_pos[9, 0, 2] += 0.015 if drop_in_own_record else 0.03
    return tr.done()


def test_slip_is_the_record_s_own_and_never_the_drop_after_it():
    """A record's slip is the object-to-palm change over its own burst. The one lifted record of a lift the
    hand could not keep grips (the box rose with the palm through that record; its drop is the NEXT record's):
    R04 present, no hold (3 records of contact, under HOLD_MIN_S). Before 2026-09-26 the next record's drop
    (0.38 m/s here) was counted too, and such a record could never grip. A box already falling away within
    the record itself still slips."""
    tr = _one_record_lift()
    ctx = D.Context(tr, [])
    g = D._grip(ctx, "R", 0)
    assert g["slip_mps"][9] == pytest.approx(0.0, abs=1e-9) and g["slip_mps"][10] > 10 * D.GRIP_SLIP_MPS
    assert np.isnan(g["slip_mps"][0]), "the first record has no burst start to compare with"
    assert list(np.nonzero(g["grip"])[0]) == [9]
    dets = _run(tr, D.R04)
    assert len(dets) == 1 and dets[0]["measures"]["grip_share"] == pytest.approx(1.0 / 3.0)
    assert _run(tr, D.H02) == []
    assert _run(_one_record_lift(drop_in_own_record=True), D.R04) == []
