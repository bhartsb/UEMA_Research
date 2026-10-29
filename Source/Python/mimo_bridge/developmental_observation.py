# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Derive audit measurements from world geometry without feeding answers to a policy.

Missing anatomical axes, contacts or declared physical tolerances remain missing.
This adapter measures the body; it cannot discover what the network represents.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .body_layout import chest_orientation
from .developmental_evidence import ABILITY_MEASUREMENTS


def _scalar(value):
    if isinstance(value, bool):
        raise ValueError("expected a physical scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite physical measurement")
    return result


def _xyz(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("expected a finite three-dimensional position")
    return result


def quaternion_rotation(value):
    """MuJoCo uses w,x,y,z; reject invalid rotations instead of assuming identity."""
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("invalid wxyz quaternion")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    w, x, y, z = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _positions(row):
    result = {name: _xyz(pos) for name, pos in row.get("body_positions", {}).items()}
    result.update({name: _xyz(pos) for name, pos in row.get("object_positions", {}).items()})
    if "hip_position_xyz_m" in row:
        result.setdefault("hip", _xyz(row["hip_position_xyz_m"]))
    if "hand_positions_xyz_m" in row:
        hands = np.asarray(row["hand_positions_xyz_m"], dtype=float)
        if hands.shape != (2, 3) or not np.isfinite(hands).all():
            raise ValueError("hand positions must be left/right xyz")
        result.setdefault("left_hand", hands[0])
        result.setdefault("right_hand", hands[1])
    return result


def _rotation(row, body):
    value = row.get("body_quaternions_wxyz", {}).get(body)
    if value is None and body == "chest":
        value = row.get("torso_quaternion_wxyz")
    return None if value is None else quaternion_rotation(value)


def _contacts(row, context):
    if "contacts" not in row or "contact_min_force_n" not in context or "geom_body_names" not in context:
        return None
    minimum = _scalar(context["contact_min_force_n"])
    if minimum < 0:
        raise ValueError("contact noise threshold cannot be negative")
    mapping = context["geom_body_names"]
    result = []
    for contact in row["contacts"]:
        geoms = contact["geoms"]
        if len(geoms) != 2 or any(g not in mapping for g in geoms):
            raise ValueError("contact geometry has no physical-body mapping")
        force = _scalar(contact["force_contact_frame"][0])
        if force < -minimum:
            raise ValueError("negative normal force; check contact-frame convention")
        point = _xyz(contact["position_m"])
        if force > minimum:
            result.append(dict(geoms=tuple(geoms), bodies=tuple(mapping[g] for g in geoms),
                               normal_force_n=force, position=point))
    return result


def _hull(points):
    points = sorted(set(tuple(map(float, p)) for p in points))
    if len(points) < 3:
        return points
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    halves = []
    for ordered in (points, list(reversed(points))):
        half = []
        for point in ordered:
            while len(half) >= 2 and cross(half[-2], half[-1], point) <= 0:
                half.pop()
            half.append(point)
        halves.append(half[:-1])
    return halves[0] + halves[1]


def support_margin(point, contacts):
    """Signed distance to the convex support polygon; degenerate support is unknown.

    This is a static geometric measurement, not a complete dynamic balance test.
    """
    polygon = _hull(contacts)
    if len(polygon) < 3:
        return None
    p = np.asarray(point, dtype=float)
    margins = []
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        a, b = np.asarray(a), np.asarray(b)
        edge = b-a
        margins.append(float((edge[0]*(p[1]-a[1])-edge[1]*(p[0]-a[0]))/np.linalg.norm(edge)))
    return min(margins)


def derive_observation(current: Mapping[str, Any], initial: Mapping[str, Any],
                       context: Mapping[str, Any], previous: Mapping[str, Any] | None = None):
    """Return ``measurements`` and per-ability ``unavailable`` field lists.

    ``context`` contains actual wiring and declared instrument tolerances, never
    learner goals. Local axes are calibrated in a supine reference by body_layout.
    Anatomical body groups may be extended for UE without changing this contract.
    """
    now, start = _positions(current), _positions(initial)
    anatomy = dict(chest="chest", hip="hip", head="head", left_hand="left_hand", right_hand="right_hand",
                   left_foot="left_foot", right_foot="right_foot")
    anatomy.update(context.get("anatomy", {}))
    hand_groups = {side: set(context.get("hand_body_groups", {}).get(side, [anatomy[side+"_hand"]])) for side in ("left", "right")}
    foot_groups = {side: set(context.get("foot_body_groups", {}).get(side, [anatomy[side+"_foot"]])) for side in ("left", "right")}
    hands = set.union(*hand_groups.values())
    feet = set.union(*foot_groups.values())
    knees = set(context.get("knee_body_names", ()))
    body_names = set(context.get("avatar_body_names", ()))
    m: dict[str, float | bool] = {}
    for name in ("target_angle_deg", "target_distance_m"):
        if name in current:
            m[name] = _scalar(current[name])
    if "target_visible" in current:
        if not isinstance(current["target_visible"], bool):
            raise ValueError("target visibility must be measured as boolean")
        m["target_visible"] = current["target_visible"]
    chest, hip, head = (anatomy[key] for key in ("chest", "hip", "head"))
    for body, name in ((chest, "torso_height_m"),):
        if body in now:
            m[name] = float(now[body][2])
    if "torso_height_m" not in m and "body_height_m" in current:
        m["torso_height_m"] = _scalar(current["body_height_m"])
    if chest in now and chest in start:
        m["height_gain_m"] = float(now[chest][2]-start[chest][2])
    if head in now and head in start:
        m["head_lift_m"] = float(now[head][2]-start[head][2])
    head_rotation = _rotation(current, head)
    if head_rotation is not None and head in now and "head_local_forward" in context and "head_target_xyz_m" in context:
        forward = head_rotation @ _xyz(context["head_local_forward"])
        direction = _xyz(context["head_target_xyz_m"])-now[head]
        denominator = np.linalg.norm(forward)*np.linalg.norm(direction)
        if denominator > 1e-12:
            m["head_angle_error_deg"] = float(np.degrees(np.arccos(np.clip(np.dot(forward, direction)/denominator, -1, 1))))
    if "target_audio_rms" in current and "audio_noise_floor_rms" in context and "target_visible" in m:
        m["auditory_only_target"] = _scalar(current["target_audio_rms"]) > _scalar(context["audio_noise_floor_rms"]) and not m["target_visible"]
    if hip in now and hip in start:
        m["travel_distance_m"] = float(np.linalg.norm((now[hip]-start[hip])[:2]))

    rotation, initial_rotation = _rotation(current, chest), _rotation(initial, chest)
    axes = context.get("chest_local_axes")
    if rotation is not None and axes is not None:
        axes = np.asarray(axes, dtype=float)
        if axes.shape != (3, 3) or not np.isfinite(axes).all() or not np.allclose(axes @ axes.T, np.eye(3), atol=1e-5):
            raise ValueError("chest anatomical axes must be an orthonormal calibrated basis")
        orient = chest_orientation(rotation, axes)
        m["body_ventral_up_dot"] = orient["front_up_component"]
        longitudinal_up = float(np.dot(rotation @ axes[0], [0, 0, 1]))
        m["body_longitudinal_up_dot"] = longitudinal_up
        if initial_rotation is not None:
            original = chest_orientation(initial_rotation, axes)
            if orient["valid"] and original["valid"]:
                difference = (orient["roll_degrees"]-original["roll_degrees"]+180) % 360-180
                m["body_rotation_from_start_deg"] = abs(float(difference))
        limits = context.get("posture_limits", {})
        if "upright_min_longitudinal_up_dot" in limits:
            m["upright_posture"] = longitudinal_up >= _scalar(limits["upright_min_longitudinal_up_dot"])
        if "maximum_floor_longitudinal_up_dot" in limits:
            m["floor_oriented"] = abs(longitudinal_up) <= _scalar(limits["maximum_floor_longitudinal_up_dot"])

    contacts = _contacts(current, context)
    obj = context.get("object_body")
    target = context.get("target_body", obj)
    tool = context.get("tool_body")
    support_geoms = set(context.get("support_geom_names", ()))
    if contacts is not None:
        def between(group, other):
            return [c for c in contacts if (c["bodies"][0] in group and c["bodies"][1] == other)
                    or (c["bodies"][1] in group and c["bodies"][0] == other)]
        if target:
            target_contacts = between(hands, target)
            m["target_contact"] = bool(target_contacts)
            m["hand_target_contact"] = bool(target_contacts)
            m["direct_hand_target_contact"] = bool(target_contacts)
        if head in now and body_names and "external_assistance" in current:
            head_external_contact = any(head in c["bodies"] and not set(c["bodies"]).issubset(body_names) for c in contacts)
            m["head_supported_without_help"] = not head_external_contact and current["external_assistance"] is False
        if tool and target:
            m["tool_target_contact"] = bool(between({tool}, target))
        if obj:
            object_contacts = between(hands, obj)
            m["hand_object_contact"] = bool(object_contacts)
            m["left_object_contact"] = bool(between(hand_groups["left"], obj))
            m["right_object_contact"] = bool(between(hand_groups["right"], obj))
            m["hand_object_normal_force_n"] = sum(c["normal_force_n"] for c in object_contacts)
            if "maximum_grip_force_n" in context:
                m["excessive_grip_force"] = m["hand_object_normal_force_n"] > _scalar(context["maximum_grip_force_n"])
        if support_geoms and body_names:
            supports = []
            supported_body = set()
            for c in contacts:
                for i in (0, 1):
                    if c["geoms"][i] in support_geoms and c["bodies"][1-i] in body_names:
                        supports.append(c)
                        supported_body.add(c["bodies"][1-i])
            m["body_supported"] = bool(supports)
            m["arm_support_contact"] = bool(supported_body & set(context.get("arm_body_names", hands)))
            m["feet_support_only"] = bool(supported_body) and supported_body.issubset(feet)
            m["left_foot_support"] = bool(foot_groups["left"] & supported_body)
            m["right_foot_support"] = bool(foot_groups["right"] & supported_body)
            if knees:
                m["knee_support_contact"] = bool(supported_body & knees)
            if "floor_oriented" in m:
                limbs = set(context.get("locomotion_support_body_names", ()))
                if limbs:
                    m["floor_locomotion"] = m["floor_oriented"] and bool(supported_body) and supported_body.issubset(limbs)
            if "center_of_mass_xyz_m" in current and supports:
                com = _xyz(current["center_of_mass_xyz_m"])
                margin = support_margin(com[:2], [c["position"][:2] for c in supports])
                if margin is not None:
                    m["static_support_margin_m"] = margin
                    stable = context.get("support_stability", {})
                    if set(stable) >= {"minimum_support_margin_m", "maximum_com_speed_m_s", "maximum_torso_angular_speed_rad_s"} and "center_of_mass_velocity_m_s" in current and "torso_angular_velocity_rad_s" in current:
                        m["support_stable"] = (margin >= _scalar(stable["minimum_support_margin_m"])
                            and np.linalg.norm(_xyz(current["center_of_mass_velocity_m_s"])) <= _scalar(stable["maximum_com_speed_m_s"])
                            and np.linalg.norm(_xyz(current["torso_angular_velocity_rad_s"])) <= _scalar(stable["maximum_torso_angular_speed_rad_s"]))
                        m["support_stable"] = bool(m["support_stable"])
            support_hands = set(context.get("graspable_support_body_names", ()))
            if support_hands:
                # Contact alone is not a load-bearing grasp: additionally require
                # a measured mechanical support force in the declared direction.
                gripped = [c for name in support_hands for c in between(hands, name)]
                if "support_grasp_verified" in current:
                    m["support_grasp"] = bool(gripped) and current["support_grasp_verified"] is True
        if obj and support_geoms:
            m["object_support_contact"] = any(obj in c["bodies"] and bool(set(c["geoms"]) & support_geoms) for c in contacts)

    if obj in now and obj in start:
        m["object_lift_m"] = float(now[obj][2]-start[obj][2])
        m["object_displacement_m"] = float(np.linalg.norm(now[obj]-start[obj]))
        if "placement_target_xyz_m" in context:
            m["placement_error_m"] = float(np.linalg.norm(now[obj]-_xyz(context["placement_target_xyz_m"])))
        if hip in start:
            m["retrieval_distance_m"] = float(np.linalg.norm(start[obj]-start[hip])-np.linalg.norm(now[obj]-start[hip]))
        attached = current.get("object_attachment_state", {}).get(obj)
        if isinstance(attached, bool) and obj in context.get("free_object_body_names", ()):
            m["object_attached"] = attached
        anchors_present = {anatomy['left_hand'],anatomy['right_hand']}
        if all(hand in now and hand in start for hand in anchors_present) and "retention_max_relative_drift_m" in context and "hand_object_contact" in m:
            active_hands = [anatomy[side+"_hand"] for side in ("left", "right") if m.get(side+"_object_contact")]
            anchors = active_hands or list(anchors_present)
            drift = min(float(np.linalg.norm((now[obj]-now[h])-(start[obj]-start[h]))) for h in anchors)
            m["object_relative_hand_drift_m"] = drift
            m["object_retained"] = m["hand_object_contact"] and drift <= _scalar(context["retention_max_relative_drift_m"])
    if target in now and target in start:
        m["target_displacement_m"] = float(np.linalg.norm(now[target]-start[target]))
        if "target_effect_min_displacement_m" in context:
            m["target_effect"] = m["target_displacement_m"] >= _scalar(context["target_effect_min_displacement_m"])
    if "slip_challenge_verified" in current and "object_relative_hand_drift_m" in m and "slip_min_relative_drift_m" in context:
        m["slip_challenge"] = (current["slip_challenge_verified"] is True
            and m["object_relative_hand_drift_m"] >= _scalar(context["slip_min_relative_drift_m"]))
    if "desired_support_center_xyz_m" in context and contacts is not None and support_geoms:
        support_contacts = [c for c in contacts if bool(set(c["geoms"]) & support_geoms) and bool(set(c["bodies"]) & body_names)]
        if support_contacts:
            pressure_center = np.average([c["position"] for c in support_contacts], axis=0,
                                         weights=[c["normal_force_n"] for c in support_contacts])
            m["support_shift_error_m"] = float(np.linalg.norm((pressure_center-_xyz(context["desired_support_center_xyz_m"]))[:2]))

    # Sitting/kneeling are available only with declared articulated geometry.
    limits = context.get("posture_limits", {})
    if m.get("upright_posture") is not None and hip in now and "floor_height_m" in context:
        hip_height = float(now[hip][2]-_scalar(context["floor_height_m"]))
        if "sitting_max_hip_height_m" in limits and "sitting_hip_support_body_names" in context and contacts is not None and support_geoms:
            seated = any(bool(set(c["bodies"]) & set(context["sitting_hip_support_body_names"])) and bool(set(c["geoms"]) & support_geoms) for c in contacts)
            m["sitting_posture"] = m["upright_posture"] and seated and hip_height <= _scalar(limits["sitting_max_hip_height_m"])
        if knees and knees.issubset(now) and "kneeling_min_hip_above_knee_m" in limits and "knee_support_contact" in m:
            m["kneeling_posture"] = bool(m["upright_posture"] and m["knee_support_contact"] and
                now[hip][2]-np.mean([now[k][2] for k in knees]) >= _scalar(limits["kneeling_min_hip_above_knee_m"]))

    unavailable = {ability: [name for name in fields if name not in m] for ability, fields in ABILITY_MEASUREMENTS.items()}
    return dict(measurements=m, unavailable=unavailable)


def derive_trace(rows: Sequence[Mapping[str, Any]], context: Mapping[str, Any]):
    """Convert a physical trace, retaining only fields measured for every frame.

    A walking step requires measured foot clearance followed by support contact
    and displacement. The thresholds are declared instrument tolerances. A body
    translated across the floor while its feet stay planted earns no steps.
    """
    if not rows:
        raise ValueError("empty physical trace")
    derived = []
    lifted = {}
    count = 0
    step = context.get("step_measurement", {})
    anatomy = dict(left_foot="left_foot", right_foot="right_foot")
    anatomy.update(context.get("anatomy", {}))
    can_count = set(step) >= {"minimum_clearance_m", "minimum_advance_m"} and "floor_height_m" in context
    previous_contacts = {}
    for index, row in enumerate(rows):
        m = derive_observation(row, rows[0], context, rows[index-1] if index else None)["measurements"]
        positions = _positions(row)
        surface_heights = row.get("foot_lowest_surface_height_m", {})
        if can_count and all(anatomy[s+"_foot"] in positions and s+"_foot_support" in m and anatomy[s+"_foot"] in surface_heights for s in ("left", "right")):
            for side in ("left", "right"):
                position = positions[anatomy[side+"_foot"]]
                supported = m[side+"_foot_support"]
                if not supported:
                    if previous_contacts.get(side) is True:
                        prior_positions = _positions(rows[index-1])
                        lifted[side] = dict(origin=prior_positions[anatomy[side+"_foot"]].copy(), clear=False)
                    if side in lifted and _scalar(surface_heights[anatomy[side+"_foot"]])-_scalar(context["floor_height_m"]) >= _scalar(step["minimum_clearance_m"]):
                        lifted[side]["clear"] = True
                elif side in lifted:
                    swing = lifted.pop(side)
                    if swing["clear"] and np.linalg.norm((position-swing["origin"])[:2]) >= _scalar(step["minimum_advance_m"]):
                        count += 1
                previous_contacts[side] = supported
            m["step_count"] = count
        derived.append(m)
    fields = set.intersection(*(set(row) for row in derived))
    measurements = {name: [row[name] for row in derived] for name in sorted(fields)}
    unavailable = {ability: [name for name in names if name not in measurements] for ability, names in ABILITY_MEASUREMENTS.items()}
    return dict(measurements=measurements, unavailable=unavailable)
