# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""UEMA 11.2: an outside thing crossing into Level 1 of the self, and back out.

The spec has described this since v3.0 and no code did it: the body map had one row per body
handle and no columns for anything outside the body, so a grasped toy could never join the
self and letting it go could never be felt as the self contracting. These tests pin the
behaviour that now exists.

The rule, in one line: a handle is scored by how much the SENSORS move when that handle is
commanded; an outside thing is scored by how much IT moves when anything is commanded.

EVERY DISTANCE IN THIS FILE IS WORLD METRES, which is the point of half of it. The first cut
took a handle's movement out of proprioception, where a position is measured from the left
wrist and divided by 200 cm, and compared it with an object position in world metres. These
fixtures use the sizes the live crib scene measures (a hand moves a median 0.00156 m in one
model step of 0.01 s; a box nobody touches moves a median 0.0 m), so a fixture that agreed
with a broken unit would stop agreeing.
"""
import pytest
import torch

from lem.model.body_schema import DifferentiableBodySchema
from mimo_bridge.lem_model import (COMMAND_POSITION_RANGE_CM, HANDLE_POS_TO_METRES,
                                   SELF_OBJECTS, object_state_for)

HANDLES, PROPRIO, MOTOR = 4, 11, 9
HAND_STEP_M = 0.0016        # a hand's median world movement in one model step, measured
STILL_JITTER_M = 2e-5       # a thing nobody is touching, well under the map's MOVED_FLOOR


def _map(objects=("toy", "mobile"), momentum=0.9):
    return DifferentiableBodySchema(n_handles=HANDLES, n_proprio_per_handle=PROPRIO,
                                    motor_dims=MOTOR, ema_momentum=momentum,
                                    object_names=objects)


def _drive(m, steps, held=(), seed=0, present=(1.0, 1.0), hand_step_m=HAND_STEP_M,
           handle_scale=1.0):
    """Command the body, move the handles in the world, and move the world's things.

    A thing named in `held` travels exactly with handle 0, which is what being carried means.
    Anything else sits where it is, jittering by STILL_JITTER_M.

    handle_scale rescales ONLY the handle reading that is handed to the map, leaving the real
    world untouched, which is how a wrong unit is reproduced."""
    g = torch.Generator().manual_seed(seed)
    motor = torch.zeros(HANDLES, MOTOR)
    # where each handle is, in world metres: near the chest, a hand's width apart
    hand_pos = torch.tensor([[0.10, 0.02, 0.44], [0.10, -0.02, 0.44],
                             [0.0, 0.06, 0.30], [0.0, -0.06, 0.30]])
    pos = torch.tensor([[0.30, 0.0, 0.50], [0.0, 0.30, 0.90]])       # the two things, metres
    for _ in range(steps):
        motor = motor + 0.02 * torch.randn(HANDLES, MOTOR, generator=g)
        pro = torch.zeros(HANDLES, PROPRIO)
        pro[:, 4:7] = motor[:, :3]          # proprioception, on its own scale, as it arrives
        walk = hand_step_m * torch.nn.functional.normalize(
            torch.randn(HANDLES, 3, generator=g), dim=-1)
        hand_pos = hand_pos + walk
        moved = walk[0]                                 # what handle 0 did this moment
        nxt = pos.clone()
        for i in range(pos.shape[0]):
            nxt[i] = pos[i] + (moved if i in held
                               else STILL_JITTER_M * torch.randn(3, generator=g))
        pos = nxt
        m.update(motor, pro, object_state=pos, object_present=torch.tensor(present),
                 handle_world_pos=hand_pos * handle_scale)
    return m


def test_a_thing_that_travels_with_the_body_joins_the_self():
    """The first thing is carried by the hand: every command moves it. The second sits in the
    crib. Only the carried one may cross into the self."""
    m = _drive(_map(), 400, held=(0,))
    toy, mobile = [float(v) for v in m.object_membership()]
    assert toy > 0.5, "a thing carried by the body did not join the self: %.3f" % toy
    assert mobile < 0.2, "a thing that sat still joined the self anyway: %.3f" % mobile
    assert toy > 3 * mobile


def test_letting_go_contracts_the_self():
    """The slip of UEMA 11.3, measured: hold the toy until it is clearly self, then let go. Its
    score must fall back, and the extent of the perceived self must fall with it."""
    m = _drive(_map(), 400, held=(0,))
    held_score = float(m.object_membership()[0])
    held_extent = float(m.self_extent())
    _drive(m, 400, held=(), seed=1)
    let_go_score = float(m.object_membership()[0])
    assert held_score > 0.5 and let_go_score < 0.2, (held_score, let_go_score)
    assert float(m.self_extent()) < held_extent, "the perceived self did not contract"


def test_a_thing_that_is_away_is_never_taken_in():
    """A toy that is not out must gain nothing, however the body moves - including the moment
    it is put back somewhere else, which would otherwise read as one huge movement caused by
    whatever the body was doing."""
    m = _map()
    _drive(m, 300, held=(0,), present=(0.0, 0.0))
    assert float(m.object_membership()[0]) < 0.1, "a thing that was away joined the self"
    # now it is brought back in a different place: that jump must not count either
    g = torch.Generator().manual_seed(3)
    motor = 0.02 * torch.randn(HANDLES, MOTOR, generator=g)
    pro = torch.zeros(HANDLES, PROPRIO); pro[:, 4:7] = motor[:, :3]
    far = torch.tensor([[9.0, 9.0, 9.0], [0.0, 0.30, 0.90]])
    before = m.object_match.abs().sum()
    m.update(motor, pro, object_state=far, object_present=torch.tensor([1.0, 1.0]),
             handle_world_pos=m.prev_handle_pos + HAND_STEP_M)
    after = m.object_match.abs().sum()
    assert float(after) <= float(before) + 1e-6, (
        "bringing a thing back in a new place counted as the body moving it")


def test_the_body_is_scored_exactly_as_before():
    """Adding outside things must not move a single handle's score: a handle is judged on the
    body's own sensor columns only."""
    torch.manual_seed(0)
    with_objects = _drive(_map(), 200, held=(0,))
    torch.manual_seed(0)
    plain = _drive(_map(objects=()), 200, held=(0,))
    assert torch.allclose(with_objects.self_membership(), plain.self_membership(), atol=1e-6)
    assert plain.n_objects == 0 and plain.object_membership().numel() == 0
    # with no objects the extent is the body's own mean, as it was
    assert float(plain.self_extent()) == pytest.approx(float(plain.self_membership().mean()))


def test_the_extent_of_the_self_counts_what_was_taken_in():
    """The coherence term of wellness carries this number. A body that has taken a toy in must
    read a larger extent than the same body with the toy left in the crib, so wellness rises
    with the self and falls when it contracts - with nothing told to want either."""
    took_it = _drive(_map(), 400, held=(0,))
    left_it = _drive(_map(), 400, held=(), seed=2)
    assert float(took_it.self_extent()) > float(left_it.self_extent())
    # and the body's own share is the same in both: it is the toy that made the difference
    assert float(took_it.self_membership().mean()) == pytest.approx(
        float(left_it.self_membership().mean()), abs=0.05)


def test_a_resume_carries_what_was_taken_in():
    m = _drive(_map(), 300, held=(0,))
    again = _map()
    again.load_state_dict({k: v.clone() for k, v in m.state_dict().items()}, strict=True)
    assert torch.equal(again.object_membership(), m.object_membership())
    assert float(again.self_extent()) == float(m.self_extent())


def test_where_the_things_are_is_read_from_the_worlds_own_report():
    """The model reads each thing's place out of the per-body report the world already writes.
    A report that does not carry a thing reads as away, never as at the origin."""
    assert SELF_OBJECTS == ("toy", "mobile")
    pos, here = object_state_for({"caregiver": {
        "object_position": [0.1, 0.2, 0.3], "object_present": True,
        "mobile_position": [0.0, 0.0, 1.2], "mobile_present": True}})
    assert pos == [[0.1, 0.2, 0.3], [0.0, 0.0, 1.2]] and here == [1.0, 1.0]
    pos, here = object_state_for({"caregiver": {"object_present": False, "mobile_present": True,
                                                "mobile_position": [0.0, 0.0, 1.2]}})
    assert here == [0.0, 1.0] and pos[0] == [0.0, 0.0, 0.0]
    for meta in ({}, {"caregiver": {}}, {"caregiver": {"object_position": "nonsense",
                                                       "object_present": True}}):
        pos, here = object_state_for(meta)
        assert here == [0.0, 0.0] and pos == [[0.0, 0.0, 0.0]] * 2, meta


# ------------------------------------------------------------------ the unit and the frame

def test_a_handles_place_is_read_in_world_metres_not_out_of_proprioception():
    """The defect this pins: proprioception carries a handle's position as
    (world cm - left wrist world cm) / 200, so one of its units is two metres and every
    reading has the left wrist's own travel subtracted from it. Feeding that to the object
    half compared half-size numbers in a wrist-centred frame against world metres.

    Here one map is driven with the world-metre reading and a second with the same motion
    expressed the way proprioception carries it. Only the first may take the toy in."""
    metres = _drive(_map(), 400, held=(0,))
    as_proprio = _drive(_map(), 400, held=(0,),
                        handle_scale=1.0 / (COMMAND_POSITION_RANGE_CM / 100.0))
    assert float(metres.object_membership()[0]) > 0.5
    assert float(as_proprio.object_membership()[0]) < 0.5, (
        "a handle movement on proprioception's scale still read as carrying the toy, so the "
        "unit is not being checked: %.3f" % float(as_proprio.object_membership()[0]))


def test_the_conversion_from_the_achieved_pose_row_to_metres():
    """The model turns the achieved-pose half of the motor row into world metres with one
    constant. The contract assay (tools/sensor_action_contract_check.echo_learner_check)
    checks that the same row times 200 equals the body's world centimetres, so the metres
    conversion is that 200 over 100."""
    assert COMMAND_POSITION_RANGE_CM == 200.0
    assert HANDLE_POS_TO_METRES == pytest.approx(2.0)
    row_units = torch.tensor([[0.05, 0.0, 0.22]])            # what the row carries
    assert torch.allclose(row_units * HANDLE_POS_TO_METRES,
                          row_units * COMMAND_POSITION_RANGE_CM / 100.0)
    assert float((row_units * HANDLE_POS_TO_METRES)[0, 2]) == pytest.approx(0.44)


def test_the_left_wrist_can_carry_a_toy():
    """Handle 0 is the row proprioception is measured FROM, so its proprioceptive position is
    identically zero on every frame. Reading a handle's movement there made the left wrist -
    one of the two handles most likely to be holding something - unable to carry anything at
    all. It must score like any other handle now."""
    m = _map()
    g = torch.Generator().manual_seed(7)
    hand = torch.zeros(HANDLES, 3)
    toy = torch.tensor([[0.30, 0.0, 0.50], [0.0, 0.30, 0.90]])
    for _ in range(400):
        motor = 0.02 * torch.randn(HANDLES, MOTOR, generator=g)
        pro = torch.zeros(HANDLES, PROPRIO)      # every handle reads zero, as the wrist does
        walk = HAND_STEP_M * torch.nn.functional.normalize(
            torch.randn(HANDLES, 3, generator=g), dim=-1)
        hand = hand + walk
        toy = toy + torch.stack([walk[0], STILL_JITTER_M * torch.randn(3, generator=g)])
        m.update(motor, pro, object_state=toy, object_present=torch.tensor([1.0, 1.0]),
                 handle_world_pos=hand)
    assert float(m.object_membership()[0]) > 0.5, (
        "handle 0 could not carry a toy: %.3f" % float(m.object_membership()[0]))
    assert float(m.object_match[0, 0]) > float(m.object_match[1:, 0].max()), (
        "the carrying handle was not the best match")


def test_no_reading_of_where_the_handles_are_means_no_evidence():
    """A frame that carries no achieved pose must leave every thing's score exactly where it
    was, and must not turn the gap into one huge movement on the next frame."""
    m = _drive(_map(), 200, held=(0,))
    before = m.object_match.clone()
    motor = torch.zeros(HANDLES, MOTOR)
    pro = torch.zeros(HANDLES, PROPRIO)
    m.update(motor, pro, object_state=torch.tensor([[0.30, 0.0, 0.50], [0.0, 0.30, 0.90]]),
             object_present=torch.tensor([1.0, 1.0]), handle_world_pos=None)
    assert torch.equal(m.object_match, before), "a frame with no pose changed the scores"
    teleported = m.prev_handle_pos + 5.0            # the handles are read again, far away
    m.update(motor, pro, object_state=torch.tensor([[5.30, 5.0, 5.50], [0.0, 0.30, 0.90]]),
             object_present=torch.tensor([1.0, 1.0]), handle_world_pos=teleported)
    assert torch.equal(m.object_match, before), (
        "the first reading after a gap was scored as one huge movement")


def test_a_struck_thing_is_not_a_carried_thing():
    """Batting the mobile is not carrying it. The hand meets it for one moment and it swings
    away on its own, keeping going after the hand has gone elsewhere. Carrying means going the
    same way AND the same distance EVERY moment, so a thing that is struck must stay below the
    line a carried thing crosses.

    Measured on the crib scene, 400 babble steps: the mobile's centre element moves a median
    0.00107 m in a model step against a hand's 0.00156 m, and it keeps moving while the hand
    does something else."""
    m = _map()
    g = torch.Generator().manual_seed(11)
    hand = torch.zeros(HANDLES, 3)
    thing = torch.tensor([[0.30, 0.0, 0.50], [0.0, 0.30, 0.90]])
    swing = torch.zeros(3)
    for t in range(400):
        motor = 0.02 * torch.randn(HANDLES, MOTOR, generator=g)
        walk = HAND_STEP_M * torch.nn.functional.normalize(
            torch.randn(HANDLES, 3, generator=g), dim=-1)
        hand = hand + walk
        if t % 20 == 0:
            swing = 0.00107 * torch.nn.functional.normalize(walk[0], dim=-1)   # struck
        swing = swing * 0.9                            # and it coasts, slowing, on its own
        thing = thing + torch.stack([swing, STILL_JITTER_M * torch.randn(3, generator=g)])
        m.update(motor, torch.zeros(HANDLES, PROPRIO), object_state=thing,
                 object_present=torch.tensor([1.0, 1.0]), handle_world_pos=hand)
    struck = float(m.object_membership()[0])
    assert struck < 0.5, "a thing struck away read as carried: %.3f" % struck


def test_how_far_agrees_is_the_ratio_of_the_two_distances():
    """The size term is the smaller movement over the larger, so a thing dragged the same way
    but only part of the distance scores exactly that fraction. This pins the number the
    carried/not-carried line is drawn on: the map's temperature of 5 puts the line at a score
    of 0.5, so a thing has to keep more than half the hand's distance to cross it."""
    m = _map()
    g = torch.Generator().manual_seed(13)
    hand = torch.zeros(HANDLES, 3)
    thing = torch.tensor([[0.30, 0.0, 0.50], [0.0, 0.30, 0.90]])
    for _ in range(400):
        walk = HAND_STEP_M * torch.nn.functional.normalize(
            torch.randn(HANDLES, 3, generator=g), dim=-1)
        hand = hand + walk
        thing = thing + torch.stack([0.30 * walk[0],
                                     STILL_JITTER_M * torch.randn(3, generator=g)])
        m.update(0.02 * torch.randn(HANDLES, MOTOR, generator=g),
                 torch.zeros(HANDLES, PROPRIO), object_state=thing,
                 object_present=torch.tensor([1.0, 1.0]), handle_world_pos=hand)
    assert float(m.object_match[0, 0]) == pytest.approx(0.30, abs=0.02)
    assert float(m.object_membership()[0]) < 0.5


def test_the_two_ends_of_the_membership_scale():
    """What object_membership can actually read, so no report claims 0 or 1. The map turns a
    match into a number with sigmoid(5 x (match - 0.5)): a thing with no evidence at all reads
    0.0759, not 0, and a thing carried perfectly for long enough reads 0.9241, not 1."""
    none_at_all = _map()
    assert float(none_at_all.object_membership()[0]) == pytest.approx(0.0759, abs=0.0005)
    perfect = _map()
    perfect.object_match.fill_(1.0)
    assert float(perfect.object_membership()[0]) == pytest.approx(0.9241, abs=0.0005)


# ------------------------------------------------------------------ the control

def test_unrelated_movement_scores_nothing_once_its_control_is_taken_off():
    """Agreement is clamped at zero, so two unrelated movements still average above zero, and
    taking the best of many handles raises it again. Measured on the live run before this
    control existed, the toy read 0.140 while its own unrelated-motion null read 0.173.

    Here a thing wanders on its own at its own speed while the handles wander at theirs. The
    raw match must be clearly above zero and the control must sit with it, so what is left
    after the subtraction is the no-evidence floor."""
    m = _map()
    g = torch.Generator().manual_seed(5)
    hand = torch.zeros(HANDLES, 3)
    thing = torch.tensor([[0.30, 0.0, 0.50], [0.0, 0.30, 0.90]])
    for _ in range(600):
        walk = HAND_STEP_M * torch.nn.functional.normalize(
            torch.randn(HANDLES, 3, generator=g), dim=-1)
        hand = hand + walk
        drift = 0.0006 * torch.nn.functional.normalize(
            torch.randn(2, 3, generator=g), dim=-1)          # its own way, its own speed
        thing = thing + drift
        m.update(0.02 * torch.randn(HANDLES, MOTOR, generator=g),
                 torch.zeros(HANDLES, PROPRIO), object_state=thing,
                 object_present=torch.tensor([1.0, 1.0]), handle_world_pos=hand)
    ev = m.object_evidence()["toy"]
    assert ev["match"] > 0.1, (
        "the raw match on unrelated motion was already near zero, so this fixture does not "
        "exercise the control: %.3f" % ev["match"])
    assert ev["earned"] < 0.1, (
        "unrelated motion survived its own control: match %.3f, control %.3f"
        % (ev["match"], ev["control"]))
    assert float(m.object_membership()[0]) < 0.2


def test_the_control_does_not_touch_a_thing_that_is_really_carried():
    """Reversing the handle's direction destroys a real agreement, so a carried thing keeps
    everything it earned."""
    m = _drive(_map(), 400, held=(0,))
    ev = m.object_evidence()["toy"]
    assert ev["control"] < 0.05, "the control scored on a carried thing: %.3f" % ev["control"]
    assert ev["earned"] > 0.8 and float(m.object_membership()[0]) > 0.8


def test_the_reading_says_whether_the_world_let_the_body_move_the_thing():
    """The toy is a MuJoCo mocap body: MuJoCo writes its place from outside every step and
    never integrates it, so no reaching can displace it and a match can only ever be the adult
    carrying it the way a hand happened to go. The report must carry that fact, because a
    number quoted as incorporation is meaningless for a thing the body cannot move."""
    from mimo_bridge.lem_model import object_physics_driven_for
    assert object_physics_driven_for({"caregiver": {"object_physics_driven": False,
                                                    "mobile_physics_driven": True}}) == {
        "toy": False, "mobile": True}
    # a report that does not say reads as "the body cannot move it", the careful reading
    assert object_physics_driven_for({}) == {"toy": False, "mobile": False}
    assert object_physics_driven_for({"caregiver": {}}) == {"toy": False, "mobile": False}


def test_a_resume_carries_the_control_too():
    m = _drive(_map(), 300, held=(0,))
    again = _map()
    again.load_state_dict({k: v.clone() for k, v in m.state_dict().items()}, strict=True)
    assert torch.equal(again.object_match_null, m.object_match_null)
    assert again.object_evidence() == m.object_evidence()


# ------------------------------------------------- the whole way through, on a real body
@pytest.mark.slow
def test_a_real_step_hands_the_map_world_metres(tmp_path):
    """The unit tests above feed the map directly. This drives the model on the real crib
    body and checks what actually reaches the map: where each handle is, in world metres.

    The reading the map gets must sit where the body sits. A 24-month body lies with its parts
    within about half a metre of the floor, so every handle's place must be inside a metre of
    the origin. Proprioception's own numbers, which is what the map used to be handed, are that
    reading divided by 200 and measured from the left wrist: one of them would be under a
    hundredth of a metre and the left wrist's would be exactly zero."""
    import numpy as np
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.lem_model import LemLite

    src = SingleEnvSource(scene="crib", run_dir=str(tmp_path / "oi"), tag="_oi", vision=False,
                          seed=3, body_layout="trunk_v1")
    m = LemLite(device="cpu", seed=0, imagination=False, intention=False,
                body_layout="trunk_v1", body_schema_updates=True)
    frames = src.reset()
    dev = dict(m.dev_limits(10 ** 9))
    held = None
    for i in range(6):
        res = m.step(frames, dev, train=False, step_index=i)
        held = res.motor_for_action if (i % 5 == 0 or held is None) else held
        for _ in range(5):
            frames = src.step(held, res.gaze_out, None, render=False)

    sc = m._schemas[0]
    assert sc.n_objects == 2, "the map is not watching the outside things"
    seen = float(sc.handle_pos_seen)
    assert seen == 1.0, "no reading of where the handles are ever reached the map"
    pos = sc.prev_handle_pos.numpy()
    live = pos[np.abs(pos).sum(axis=1) > 0]
    assert live.shape[0] >= 20, "only %d handles had a place at all" % live.shape[0]
    far = float(np.abs(live).max())
    assert 0.02 < far < 1.0, (
        "the handles' places read %.4f at the furthest, which is not where a body is in "
        "world metres" % far)
    # and the left wrist, which proprioception measures everything else FROM, has a place
    assert float(np.abs(pos[0]).sum()) > 0.0, "the left wrist read no place at all"

    # the scores are in range and each one carries its control
    ev = sc.object_evidence()
    assert set(ev) == {"toy", "mobile"}
    for name, e in ev.items():
        assert 0.0 <= e["match"] <= 1.0 and 0.0 <= e["control"] <= 1.0, (name, e)
        assert 0.0759 - 1e-4 <= e["membership"] <= 0.9241 + 1e-4, (name, e)

    st = m.incorporation_stats()
    assert st is not None and set(st["objects"]) == {"toy", "mobile"}
    assert set(st["objects_evidence"]["toy"]) == {"match", "control", "earned"}
    assert set(st["objects_physics_driven"]) == {"toy", "mobile"}
