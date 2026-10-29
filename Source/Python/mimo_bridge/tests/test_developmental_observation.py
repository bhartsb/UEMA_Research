# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
from copy import deepcopy
import math

import numpy as np
import pytest

from mimo_bridge.developmental_observation import derive_observation, derive_trace, support_margin


def fixture():
    current = dict(body_positions={
        "chest": [0, 0, .1], "hip": [-.2, 0, .1], "head": [.2, 0, .1],
        "left_hand": [0, .1, .2], "right_hand": [0, -.1, .2],
        "left_foot": [-.4, .08, .02], "right_foot": [-.4, -.08, .02],
        "left_knee": [-.3, .08, .04], "right_knee": [-.3, -.08, .04],
        "box_light": [0, .1, .2]},
        body_quaternions_wxyz={"chest": [1, 0, 0, 0]},
        center_of_mass_xyz_m=[0, 0, .1], center_of_mass_velocity_m_s=[0, 0, 0],
        torso_angular_velocity_rad_s=[0, 0, 0], contacts=[],
        object_attachment_state={"box_light": False})
    context = dict(chest_local_axes=[[1, 0, 0], [0, 0, 1], [0, -1, 0]],
        avatar_body_names=list(current["body_positions"]),
        geom_body_names={"floor": "world", **{b+"_geom": b for b in current["body_positions"]}},
        support_geom_names=["floor"], contact_min_force_n=.01,
        object_body="box_light", target_body="box_light", free_object_body_names=["box_light"],
        arm_body_names=["left_hand", "right_hand"], knee_body_names=["left_knee", "right_knee"],
        support_stability=dict(minimum_support_margin_m=.001, maximum_com_speed_m_s=.1,
                               maximum_torso_angular_speed_rad_s=.2),
        posture_limits=dict(upright_min_longitudinal_up_dot=.9, maximum_floor_longitudinal_up_dot=.4),
        retention_max_relative_drift_m=.02, maximum_grip_force_n=5., floor_height_m=0)
    context["avatar_body_names"].remove("box_light")
    return current, context


def contact(body1, body2="world", point=(0, 0, 0), force=1.):
    return dict(geoms=[body1+"_geom", "floor" if body2 == "world" else body2+"_geom"],
                position_m=list(point), force_contact_frame=[force, 0, 0, 0, 0, 0])


def test_yaw_is_not_a_roll_and_real_roll_reverses_body_front():
    start, context = fixture()
    current = deepcopy(start)
    current["body_quaternions_wxyz"]["chest"] = [0, 0, 0, 1]
    m = derive_observation(current, start, context)["measurements"]
    assert m["body_rotation_from_start_deg"] == pytest.approx(0.)
    assert m["body_ventral_up_dot"] == pytest.approx(1.)
    current["body_quaternions_wxyz"]["chest"] = [0, 1, 0, 0]
    m = derive_observation(current, start, context)["measurements"]
    assert m["body_rotation_from_start_deg"] == pytest.approx(180.)
    assert m["body_ventral_up_dot"] == pytest.approx(-1.)


def test_standing_has_no_defined_gravity_roll_but_has_longitudinal_alignment():
    start, context = fixture()
    current = deepcopy(start)
    current["body_quaternions_wxyz"]["chest"] = [math.sqrt(.5), 0, -math.sqrt(.5), 0]
    m = derive_observation(current, start, context)["measurements"]
    assert m["upright_posture"] is True
    assert "body_rotation_from_start_deg" not in m


def test_fixed_or_carried_object_does_not_hide_its_attachment():
    start, context = fixture()
    current = deepcopy(start)
    current["body_positions"]["box_light"][2] += .1
    current["contacts"] = [contact("left_hand", "box_light")]
    current["object_attachment_state"]["box_light"] = True
    m = derive_observation(current, start, context)["measurements"]
    assert m["object_attached"] is True
    assert m["object_lift_m"] == pytest.approx(.1)
    del current["object_attachment_state"]
    result = derive_observation(current, start, context)
    assert "object_attached" in result["unavailable"]["grasp"]


def test_contact_presence_and_force_are_actual_local_pairs():
    start, context = fixture()
    current = deepcopy(start)
    current["contacts"] = [contact("left_hand", "box_light", force=3.),
                           contact("right_hand", "world", force=7.)]
    m = derive_observation(current, start, context)["measurements"]
    assert m["left_object_contact"] is True
    assert m["right_object_contact"] is False
    assert m["hand_object_normal_force_n"] == 3
    assert m["excessive_grip_force"] is False
    assert m["object_support_contact"] is False
    current["contacts"][0]["force_contact_frame"][0] = 6
    assert derive_observation(current, start, context)["measurements"]["excessive_grip_force"] is True


def test_one_hand_retains_object_while_other_hand_moves_away():
    start, context = fixture()
    current = deepcopy(start)
    current["contacts"] = [contact("left_hand", "box_light")]
    current["body_positions"]["right_hand"][0] += .5
    m = derive_observation(current, start, context)["measurements"]
    assert m["object_retained"] is True
    assert m["object_relative_hand_drift_m"] == 0


def test_static_polygon_alone_does_not_certify_dynamic_balance():
    start, context = fixture()
    current = deepcopy(start)
    current["contacts"] = [contact("left_foot", point=p) for p in ((-.1,-.1,0),(.1,-.1,0),(.1,.1,0),(-.1,.1,0))]
    m = derive_observation(current, start, context)["measurements"]
    assert m["support_stable"] is True
    assert m["feet_support_only"] is True
    current["center_of_mass_velocity_m_s"] = [2, 0, 0]
    assert derive_observation(current, start, context)["measurements"]["support_stable"] is False
    del current["center_of_mass_velocity_m_s"]
    assert "support_stable" not in derive_observation(current, start, context)["measurements"]


def test_degenerate_support_polygon_stays_unknown():
    assert support_margin([0, 0], [[0, 0], [1, 0]]) is None
    assert support_margin([0, 0], [[0, 0], [1, 0], [2, 0]]) is None
    assert support_margin([2, 0], [[-1,-1],[1,-1],[1,1],[-1,1]]) < 0


@pytest.mark.parametrize("bad", ["unknown_geometry", "nan_force", "bad_axes", "bad_quaternion"])
def test_bad_physics_is_rejected(bad):
    start, context = fixture()
    current = deepcopy(start)
    current["contacts"] = [contact("left_hand")]
    if bad == "unknown_geometry":
        current["contacts"][0]["geoms"][0] = "unmapped"
    elif bad == "nan_force":
        current["contacts"][0]["force_contact_frame"][0] = float("nan")
    elif bad == "bad_axes":
        context["chest_local_axes"] = np.zeros((3, 3)).tolist()
    else:
        current["body_quaternions_wxyz"]["chest"] = [0, 0, 0, 0]
    with pytest.raises(ValueError):
        derive_observation(current, start, context)


def test_empty_telemetry_never_fabricates_self_recognition_or_anticipation():
    start, context = fixture()
    result = derive_observation(start, start, context)
    for ability in ("self_world_partition", "proprioceptive_body_map", "action_self_recognition", "visual_self_recognition", "anticipation"):
        assert result["unavailable"][ability]
    result = derive_observation({}, {}, {})
    assert result["measurements"] == {}
    assert all(result["unavailable"].values())


def test_step_requires_real_foot_swing_then_support_not_whole_body_translation():
    start, context = fixture()
    context["step_measurement"] = dict(minimum_clearance_m=.02, minimum_advance_m=.03)
    start["contacts"] = [contact("left_foot"), contact("right_foot")]
    start["foot_lowest_surface_height_m"] = {"left_foot": 0, "right_foot": 0}
    moved = deepcopy(start)
    moved["body_positions"]["left_foot"][0] += .1
    assert derive_trace([start, moved], context)["measurements"]["step_count"][-1] == 0
    swing = deepcopy(start)
    swing["contacts"] = [contact("right_foot")]
    swing["body_positions"]["left_foot"][0] += .05
    swing["foot_lowest_surface_height_m"]["left_foot"] = .03
    assert derive_trace([start, swing, moved], context)["measurements"]["step_count"][-1] == 1
    del swing["foot_lowest_surface_height_m"]
    assert "step_count" in derive_trace([start, swing, moved], context)["unavailable"]["cruise_first_step"]


def test_fields_missing_in_one_frame_are_not_filled_with_zero():
    start, context = fixture()
    end = deepcopy(start)
    end["object_attachment_state"] = {}
    result = derive_trace([start, end], context)
    assert "object_attached" not in result["measurements"]
    assert "object_attached" in result["unavailable"]["grasp"]


def test_finger_contact_counts_for_the_hand_and_toe_contact_for_foot_support():
    start,context=fixture()
    context['geom_body_names'].update(left_finger_geom='left_finger',left_toe_geom='left_toe')
    context['avatar_body_names']+=['left_finger','left_toe']
    context['hand_body_groups']={'left':['left_hand','left_finger']}
    context['foot_body_groups']={'left':['left_foot','left_toe']}
    current=deepcopy(start)
    current['contacts']=[contact('left_finger','box_light'),contact('left_toe')]
    m=derive_observation(current,start,context)['measurements']
    assert m['left_object_contact'] is True
    assert m['left_foot_support'] is True
    assert m['feet_support_only'] is True
