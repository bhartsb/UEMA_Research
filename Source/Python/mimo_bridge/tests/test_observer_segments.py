# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Detectors v6 and v7 of the developmental observer (docs/gain_of_function_v4/agents/observer_segments.md,
OBSERVER_DESIGN.md decision 44): per-effector episodes, fixations from the foveal ray bundle, and
the fast lane's support fall below the usual level. Each test pins a behaviour detector v5 did not have, on synthetic tracks
whose right answer is known exactly.

  * each effector (a hand, the gaze, the trunk, a leg) has its own episodes, which overlap in
    time and never mix another effector's tokens; the left foot touching the right foot, then the
    head starting to turn, is a PAIR unit the novelty miner can count;
  * turns that continue a movement and a hand touching itself ride inside episodes; a turn from
    rest opens one; an object moving belongs to the part that touched it, else to no effector;
  * no episode crosses an absolute 200-step block, and one pass of the digester's cut over the
    returned list (latest start first) always ends inside no episode;
  * a fixation is the foveal bundle on one target for two records - the centre ray's thing, or
    where the centre ray is on a surface or nothing, the thing most of the nine rays are on - with
    its share of rays, aim error, dwell and target change; a return fixation needs the target to
    leave the fovea;
  * a support fall is a fall below the running median, so the falling edge of a one-frame impact
    is not one;
  * a novelty run is explained only by ONE detection covering all its token kinds, and babble's
    units are filtered as production's are.
"""
import numpy as np
import pytest

from mimo_bridge.tools.observer import detectors as D
from mimo_bridge.tools.observer import fastlane as FL
from mimo_bridge.tools.observer import features as F
from mimo_bridge.tools.observer import novelty as NV
from mimo_bridge.tools.observer import primitives as P
from mimo_bridge.tools.observer import segmenter as S

KINDS = ["self", "object", "caregiver", "floor", "wall", "art", "ceiling", "static"]
# name, group, kind
ENTITIES = [
    ("left_hand", "hand_L_palm", "self"),            # 0
    ("left_ff_distal", "finger_L_index", "self"),    # 1
    ("left_foot", "foot_L", "self"),                 # 2
    ("right_foot", "foot_R", "self"),                # 3
    ("head", "head", "self"),                        # 4
    ("chest", "trunk", "self"),                      # 5
    ("box_light", "box", "object"),                  # 6
    ("caregiver_torso", "caregiver", "caregiver"),   # 7
    ("floor", "floor", "floor"),                     # 8
    ("pen_art_n", "art", "art"),                     # 9
    ("dragonfly", "critter", "object"),              # 10
    ("right_hand", "hand_R_palm", "self"),           # 11
]
BOX, CARE, FLOOR, ART, CRITTER = 6, 7, 8, 9, 10


def _header():
    ents = []
    for code, (name, group, kind) in enumerate(ENTITIES):
        e = dict(code=code, name=name, kind=kind, group=group, bodies=[code + 1])
        if kind == "art":
            e.update(geoms=[0], motifs=list("ABCDEFGH"), axis=0, half_width_m=0.36, half_height_m=0.06)
        ents.append(e)
    objects = [dict(entity=BOX, name="box_light", kind="box", free=True, mocap=-1, root=BOX + 1),
               dict(entity=CARE, name="caregiver_torso", kind="caregiver", free=False, mocap=0, root=CARE + 1),
               dict(entity=CRITTER, name="dragonfly", kind="critter", free=False, mocap=1, root=CRITTER + 1)]
    return dict(entities=ents, entity_kind=[KINDS.index(k) for _, _, k in ENTITIES], objects=objects,
                weight_n=100.0, fovea_fovy_deg=24.0, joints=[])


def fake_track(n=400, first_step=1000):
    """A Track with every field the segmenter, the fixations and the gaze detectors read, built
    without a world model (no kinematics)."""
    tr = F.Track.__new__(F.Track)
    tr.header = _header()
    tr.ent = F.Entities(tr.header)
    tr.kin = None
    tr.pid = 7
    tr.n = n
    tr.step = np.arange(first_step, first_step + n)
    tr.sim_time = 0.05 * np.arange(n)
    tr.dt = np.r_[np.nan, np.full(n - 1, 0.05)]
    tr.continuous = np.isfinite(tr.dt)
    tr.episode = np.zeros(n, dtype=np.int64)
    for k in ("author", "order_seq", "toy_held"):
        setattr(tr, k, np.zeros(n, dtype=np.int64))
    for k in ("latched", "caregiver", "caregiver_present", "caregiver_contact", "rest", "gaze_saccade",
              "gaze_reflex"):
        setattr(tr, k, np.zeros(n, dtype=bool))
    tr.live = np.ones(n, dtype=bool)
    tr.curl = np.zeros((n, 2, 5))
    tr.closure = np.zeros((n, 2))
    tr.wrist = np.zeros((n, 2, 3))
    tr.head_ang_speed = np.zeros(n)
    tr.c_row = np.zeros(0, dtype=np.int64)
    tr.c_e = np.zeros((0, 2), dtype=np.int64)
    tr.c_fmax = np.zeros(0)
    tr.c_fmean = np.zeros(0)
    tr.c_pos = np.zeros((0, 3))
    tr.roll_deg = np.zeros(n)
    tr.sit_deg = np.full(n, 90.0)
    K = len(tr.header["objects"])
    tr.obj_pos = np.zeros((n, K, 3))
    tr.obj_speed = np.zeros((n, K))
    tr.xpos = np.zeros((n, len(ENTITIES) + 2, 3))
    tr.eye_pos = np.zeros((n, 2, 3))
    tr.eye_fwd = np.tile(np.array([1.0, 0.0, 0.0]), (n, 2, 1))
    tr.eye_angles = np.zeros((n, 2, 2))
    tr.ray_ent = np.full((n, 2, 9), FLOOR, dtype=np.int16)          # every ray on the floor
    tr.ray_cell = np.full((n, 2, 9), -1, dtype=np.int8)
    tr.ray_dist = np.full((n, 2, 9), 0.3, dtype=np.float32)
    return tr


def contact(tr, i0, i1, a, b):
    return dict(type="contact", i0=i0, i1=i1, a=a, b=b, a_class=P.entity_class(tr, a),
                b_class=P.entity_class(tr, b), open_start=False, peak_force_n=1.0,
                self_self=bool(tr.ent.is_self[a] and tr.ent.is_self[b]))


def look(i0, i1, cls="box", entity=BOX):
    return dict(type="fixation", i0=i0, i1=i1, eye=0, target=entity, entity=entity, target_class=cls,
                duration_s=0.05 * (i1 - i0))


def by_effector(eps):
    out = {}
    for e in eps:
        out.setdefault(e["effector"], []).append(e)
    for v in out.values():
        v.sort(key=lambda e: e["i0"])
    return out


def daemon_pass(eps, cut):
    """daemon._process's cut: one pass over the list in its order."""
    for e in eps:
        if e["i0"] <= cut < e["i1"]:
            cut = min(cut, e["i0"] - 1)
    return cut


# ------------------------------------------------------------------ per-effector episodes
def test_each_effector_has_its_own_episodes_which_overlap_and_never_mix():
    tr = fake_track()
    prims = [contact(tr, 20, 25, 0, BOX),          # the left hand touches the box
             look(21, 28),                          # the eyes rest on the box
             contact(tr, 22, 24, 2, 3)]             # the left foot touches the right foot
    eps = by_effector(S.segment(tr, prims, "run", 0))
    assert eps["hand_L"][0]["tokens"] == ["touch:box~hand_L_palm"]
    assert eps["gaze"][0]["tokens"] == ["look:box"]
    assert eps["leg_L"][0]["tokens"] == ["touch:foot_L~foot_R"] == eps["leg_R"][0]["tokens"], \
        "a foot touching a foot is an act of both legs"
    hand, gaze = eps["hand_L"][0], eps["gaze"][0]
    assert (hand["i0"], hand["i1"]) == (20, 25) and (gaze["i0"], gaze["i1"]) == (21, 28), "they overlap"
    assert "trunk" not in eps and "hand_R" not in eps
    ids = [e["id"] for v in eps.values() for e in v]
    assert len(ids) == len(set(ids)), "episodes starting on one step still have their own ids"


def test_the_left_foot_touching_the_right_then_the_head_turning_is_a_pair_unit():
    tr = fake_track()
    prims = [contact(tr, 50, 52, 2, 3)]
    tr.head_ang_speed[66:70] = 60.0                         # 3 degrees a step: a turn from rest
    prims += P.head_turns(tr)
    assert [p["i0"] for p in prims if p["type"] == "head_turn"] == [59]
    eps = S.segment(tr, prims, "run", 0)
    pairs = [e for e in eps if e["effector"] == S.PAIR]
    want = ["touch:foot_L~foot_R", "head_turn"]
    assert [p["tokens"] for p in pairs] == [want], "one unit: the same touch opened both legs"
    p = pairs[0]
    assert p["lead"] == "leg_L" and p["follow"] == "gaze" and p["lag_s"] == pytest.approx(0.45)
    assert NV.episode_runs(p["tokens"], p.get("caps")) == [tuple(want)]
    assert NV.baseline_runs([p]) == {NV.candidate_id(tuple(want)): 1}


def test_finger_turns_and_a_hand_touching_itself_ride_inside_the_hands_episodes():
    tr = fake_track()
    tri = np.r_[np.linspace(0, 20, 5)[:-1], np.linspace(20, 0, 5)[:-1]]     # 20 degrees up and down
    tr.curl[100:180, 0, 0] = np.tile(tri, 10)                               # the left index finger
    tr.closure[:] = np.mean(tr.curl[:, :, :4], axis=-1)
    prims = P.curls(tr)
    assert len([p for p in prims if p["type"] == "digit"]) >= 15
    prims += [contact(tr, 140, 142, 0, BOX), contact(tr, 160, 161, 1, 0)]  # a touch; finger on palm
    hand = by_effector(S.segment(tr, prims, "run", 0))["hand_L"]
    # detector v9: the index's first bend, after 100 records still, starts from rest and opens an
    # episode held only to its start (as a head turn from rest); the bends that continue it ride
    assert [(e["i0"], e["i1"]) for e in hand] == [(100, 100), (140, 142)], \
        "turns in progress and the hand touching itself open nothing; the touch of the box does"
    assert hand[0]["opening"] == "digit:L:index:bend" and hand[0]["tokens"] == ["digit:L:index:bend"]
    assert hand[1]["tokens"] == ["touch:box~hand_L_palm", "digit:L:index:bend"], "a turn inside rides along"
    assert hand[1]["opening"] == "touch:box~hand_L_palm"


def test_a_head_turn_opens_only_when_it_starts_from_rest():
    tr = fake_track()
    tr.head_ang_speed[100:200] = 60.0                        # the head keeps turning for 5 s
    turns = P.head_turns(tr)
    assert len(turns) >= 20 and all(b["i0"] == a["i1"] for a, b in zip(turns, turns[1:])), "they tile"
    gaze = by_effector(S.segment(tr, turns, "run", 0))["gaze"]
    assert [(e["i0"], e["i1"], e["tokens"]) for e in gaze] == [(turns[0]["i0"], turns[0]["i0"], ["head_turn"])]


def test_an_object_moving_belongs_to_the_part_that_touched_it_or_to_no_effector():
    tr = fake_track()
    tr.c_row = np.array([79])
    tr.c_e = np.array([[0, BOX]])                               # the left palm on the box
    moved = dict(type="object_motion", i0=80, i1=85, object="box_light", k=0)
    flew = dict(type="object_motion", i0=90, i1=95, object="dragonfly", k=2)
    assert S.effectors_of(tr, moved) == {"hand_L"}
    assert S.effectors_of(tr, flew) == set(), "the world moved it"
    eps = S.segment(tr, [moved, flew], "run", 0)
    assert [(e["effector"], e["tokens"]) for e in eps] == [("hand_L", ["object_moves:box_light"])]
    care = dict(type="caregiver", i0=30, i1=30, kind="arrive")
    assert S.effectors_of(tr, care) == set() and S.segment(tr, [care], "run", 0) == []


def test_no_episode_crosses_a_block_and_one_cut_pass_ends_inside_none():
    tr = fake_track()
    # the right hand touches the box every 6 steps across step 1,200 (a block boundary at index 200)
    prims = [contact(tr, i, i + 1, 11, BOX) for i in range(170, 242, 6)]
    prims += [contact(tr, 150, 170, 0, BOX), look(165, 185), contact(tr, 195, 215, 2, BOX)]
    eps = S.segment(tr, prims, "run", 0)
    right = by_effector(eps)["hand_R"]
    assert [(e["i0"], e["i1"]) for e in right] == [(170, 195), (200, 237)], "cut at step 1,200"
    assert all(int(tr.step[e["i0"]]) // 200 == int(tr.step[e["i1"]]) // 200 for e in eps)
    assert right[1]["after_cap"] and not right[0]["after_cap"], "a block's first 0.5 s continues the last"
    assert [e["i0"] for e in eps] == sorted((e["i0"] for e in eps), reverse=True), "latest start first"
    for cut in range(0, tr.n):
        end = daemon_pass(eps, cut)
        assert not any(e["i0"] <= end < e["i1"] for e in eps), (cut, end)
    # the same episodes earliest first: one pass can end inside an episode (the list order matters)
    asc = sorted(eps, key=lambda e: e["i0"])
    assert any(any(e["i0"] <= daemon_pass(asc, c) < e["i1"] for e in eps) for c in range(tr.n))


def test_a_pair_needs_two_fresh_starts_within_half_a_second():
    tr = fake_track()
    prims = [contact(tr, 200, 201, 11, BOX), contact(tr, 203, 204, 2, BOX),   # block start: no pair
             contact(tr, 300, 301, 11, BOX), contact(tr, 311, 312, 2, BOX)]   # 11 records apart
    assert [e for e in S.segment(tr, prims, "run", 0) if e["effector"] == S.PAIR] == []
    prims[3] = contact(tr, 309, 310, 2, BOX)                                  # 9 records apart
    pairs = [e for e in S.segment(tr, prims, "run", 0) if e["effector"] == S.PAIR]
    assert [(p["lead"], p["follow"], p["tokens"]) for p in pairs] == \
        [("hand_R", "leg_L", ["touch:box~hand_R_palm", "touch:box~foot_L"])]


def test_a_detection_joins_its_parts_effector_and_a_pair_lists_both_members_caps():
    tr = fake_track()
    prims = [contact(tr, 20, 25, 0, BOX), look(21, 28), contact(tr, 22, 24, 2, 3)]
    eps = S.segment(tr, prims, "run", 0)
    grasp = dict(cap="H01", i0=20, i1=24, parts=["hand_L"])
    stare = dict(cap="G02", i0=21, i1=28, parts=[])
    S.attach(eps, [grasp, stare])
    e = by_effector(eps)
    assert e["hand_L"][0]["caps"] == ["H01", "G02"] and e["gaze"][0]["caps"] == ["G02"]
    assert grasp["episode"] == e["hand_L"][0]["id"] and stare["episode"] == e["hand_L"][0]["id"]
    pair = next(p for p in e[S.PAIR] if p["lead"] == "hand_L" and p["follow"] == "gaze")
    assert pair["caps"] == ["H01", "G02"]


# ------------------------------------------------------------------ novelty
def test_a_run_is_explained_only_by_one_detection_covering_every_token():
    run = ["touch:foot_L~foot_R", "head_turn"]
    # B05 covers the touch and P01 the head turn, but no detector names the sequence
    assert NV.episode_runs(run, ["B05", "P01"]) == [tuple(run)]
    assert NV.episode_runs(["curl:R:close", "curl:R:open"], ["H04"]) == []
    assert NV.episode_runs(["touch:hand_R_palm~box", "curl:R:close"], ["H01"]) == []


def test_babble_units_are_filtered_as_the_report_filters_productions():
    tok = ["touch:hand_R_palm~mobile", "object_moves:mobile"]
    units = [dict(tokens=tok, caps=[]), dict(tokens=tok, caps=[], caregiver_touch=True),
             dict(tokens=tok, caps=[], toy_held=True)]
    assert NV.baseline_runs(units) == {NV.candidate_id(tuple(tok)): 1}


# ------------------------------------------------------------------ fixations from the foveal bundle
def test_a_fixation_is_the_centre_rays_thing_held_two_records():
    tr = fake_track(n=70, first_step=0)
    tr.obj_pos[:, 1] = [1.0, np.tan(np.radians(10.0)), 0.0]     # the caregiver 10 degrees off the axis
    tr.ray_ent[0:10, 0, 0] = BOX                                 # the centre ray alone on the box
    tr.ray_ent[10:20, 0, :6] = CARE                              # 6 of 9 rays on the caregiver
    tr.ray_ent[20:30, 0, :] = -1                                 # nothing hit
    tr.ray_ent[25, 0, :] = BOX                                   # one record only
    tr.ray_ent[30:36, 0, :] = BOX                                # all nine on the box
    tr.ray_ent[40:45, 0, :6] = ART                               # a panel: 3 rays in cell B, 3 in cell C
    tr.ray_cell[40:45, 0, :3] = 1
    tr.ray_cell[40:45, 0, 3:6] = 2
    tr.ray_ent[50:55, 0, 3:9] = CARE                             # the centre ray on the floor, 6 rays on the adult
    tr.ray_ent[60:65, 0, 0] = BOX                                # the centre ray on the box, 6 rays on the adult
    tr.ray_ent[60:65, 0, 3:9] = CARE
    fx = [f for f in P.fixations(tr) if f["eye"] == 0 and f["target_class"] != "floor"]
    assert [(f["i0"], f["i1"], f["target_label"]) for f in fx] == \
        [(0, 9, "box_light"), (10, 19, "caregiver_torso"), (30, 35, "box_light"), (40, 44, "pen_art_n (B)"),
         (50, 54, "caregiver_torso"), (60, 64, "box_light")], \
        "the centre ray's thing decides; the majority only where the centre ray is on a surface; one record is no fixation"
    box0, care, box, art, gap, front = fx
    assert box0["ray_share"] == pytest.approx(1 / 9), "a toy at reaching distance holds only the centre ray"
    assert care["ray_share"] == pytest.approx(6 / 9) and care["dwell_ms"] == pytest.approx(500.0)
    assert care["error_deg"] == pytest.approx(10.0, abs=1e-6)
    assert box["ray_share"] == 1.0 and box["distance_m"] == pytest.approx(0.3, rel=1e-3)
    assert box["switch"] and box["previous_target"] == "caregiver_torso"
    assert box["since_previous_s"] == pytest.approx(0.55)
    assert art["target"] == 10 ** 6 + ART * 16 + 1 + 1 and art["ray_share"] == pytest.approx(3 / 9), \
        "the motif cell under the centre ray, with the rays in that cell"
    assert gap["ray_share"] == pytest.approx(6 / 9) and gap["distance_m"] is None, \
        "the adult seen with the centre ray through a gap: no centre-ray distance to it"
    assert front["ray_share"] == pytest.approx(1 / 9), "the toy in front of the adult, not the adult behind it"


def test_a_return_fixation_needs_the_target_to_leave_the_fovea():
    def run(rays_left_on_target):
        tr = fake_track(n=60, first_step=0)
        tr.ray_ent[10:16, 0, :] = CARE                           # a fixation on the adult (0.3 s)
        tr.ray_ent[16:24, 0, 5:5 + rays_left_on_target] = CARE   # 0.4 s: the floor holds the majority
        tr.ray_ent[24:30, 0, :] = CARE                           # the same target again
        prims = P.fixations(tr)
        return D.G03(D.Context(tr, prims))
    assert run(4) == [], "four rays stayed on the adult: the gaze never left it"
    g = run(0)
    assert len(g) == 1 and g[0]["measures"]["away_s"] == pytest.approx(0.4)
    assert g[0]["measures"]["return_latency_s"] == pytest.approx(0.45)


def test_brief_fixation_reports_the_bundle():
    tr = fake_track(n=40, first_step=0)
    tr.ray_ent[5:15, 0, :7] = CARE
    g = D.G02(D.Context(tr, P.fixations(tr)))
    assert len(g) == 1
    m = g[0]["measures"]
    assert m["ray_share"] == pytest.approx(7 / 9) and m["duration_s"] == pytest.approx(0.5)
    assert m["distance_cm"] == pytest.approx(30.0, rel=1e-3)
    assert m["target_switch"] == 1.0 and g[0]["previous_target"] == "floor", "from the floor to the adult"


# ------------------------------------------------------------------ the fast lane's support fall
def test_a_support_fall_is_a_fall_below_the_usual_level_not_the_end_of_an_impact():
    n_steps = 200                                                # 1,000 frames of 10 ms
    fast = np.full((n_steps, 5, 8), np.nan)
    fast[:, 0, :5] = 9.81
    fast[:, 1, :5] = 0.1
    fast[:, 2, :5] = 100.0                                       # outside force = the body weight
    fast[:, 3, :5] = 40.0
    fast[:, 4, :5] = 0.0
    frames = np.full(n_steps, 5)
    fast[60, 2, 2] = 180.0                                       # a one-frame impact, back to usual
    fast[120, 2, 1:4] = 40.0                                     # 30 ms at 40% of the usual support
    drop = FL.world_series(fast, frames)["support_drop"]
    assert drop[60 * 5 + 3] == 0.0, "the impact's falling edge returns to the usual level"
    assert drop[120 * 5 + 1] == pytest.approx(60.0)
    sp = FL.summarize_world(fast, frames, 100.0)["support_drop"]
    assert sp["spikes"] == 1 and sp["largest"] == pytest.approx(60.0)
    usual = FL.running_median(np.r_[np.full(10, 1.0), 9.0, np.full(5, 1.0)])
    assert usual[10] == 1.0 and np.isnan(FL.running_median(np.ones(3))).all()


def test_the_detector_version_names_these_definitions():
    from mimo_bridge.tools.observer import daemon as DM
    assert DM.DETECTOR_VERSION == "observer_detectors_v11"
