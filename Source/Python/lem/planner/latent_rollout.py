# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Convert hierarchy actions through existing motor helpers in test/shadow paths."""
import copy

import torch

from lem.latent_body.trajectory import quaternion, tensor


def handle_targets(tree, predictions, start, template, scales, hand_frame="recorded", digit_frame="world",
                   digit_scales=None, body=None):
    """Issued target form: world centimetres, xyzw, radius, strength.

    Only owned handles change. Production must still apply its normal
    development limits, hand projection, servo mapping and command ledger.
    hand_frame "actual_wrist": the hierarchy's digit actions are relative to the wrist the body had at the
    start, and are placed back on the issued wrist, which is what wrist_relative_joint_v1 records.
    digit_frame "wrist": digit actions are in that wrist's coordinates (trajectory.action_rows); they are turned
    into world displacements and turns first (trajectory.digit_vectors_to_world).
    body (trajectory.BodyRows, the whole-body rows of Oct 1, 2026): calf actions under parent_relative_calf_v1 are
    relative to the actual thigh and are placed back on the issued thigh row; a handle whose position no executor
    reads keeps the controller's position (its position is not learned); the eyes' turn is head-relative and the
    issued eye row is written in the recorded eye row convention.
    """
    from lem.model.motor_synergy import quat_mul, rotvec_to_quat
    from lem.latent_body.trajectory import (check_body_tree, check_digit_frame, digit_vectors_to_world,
                                            eye_row_physical, move_children, row_scales)
    check_digit_frame(digit_frame, hand_frame, digit_scales)
    check_body_tree(tree, body)
    names = tree.handle_names
    out = tensor(template)
    if out.shape != (len(names), 9):
        raise ValueError("Expected one complete handle command")
    for node in tree.nodes:
        own = list(tree.owned[node.name])
        if not own:
            continue
        values = predictions[node.name].action
        if values.shape != (1, len(own) * 8) or not torch.isfinite(values).all():
            raise ValueError("Invalid hierarchy action prediction")
        action = values.detach().reshape(len(own), 8).cpu() * row_scales(
            names, own, scales, "action", digit_scales, body)
        action[:, :6] = digit_vectors_to_world(start, own, action[:, :6], names, digit_frame, body)
        out[own, :3] = start.bones[own, 4:7] + action[:, :3]
        out[own, 3:7] = quaternion(quat_mul(rotvec_to_quat(action[:, 3:6]),
                                            quaternion(start.bones[own, :4])))
        # No new strength or radius authority: retain the controller template.
        # Reconstruction of those executed fields is still measured in the lesson.
    actual = torch.cat((start.bones[:, 4:7], start.bones[:, :4]), -1)
    if hand_frame == "actual_wrist":
        from lem.latent_body.trajectory import move_digits
        out = move_digits(out, names, actual, out[:, :7].clone())
    elif hand_frame != "recorded":
        raise ValueError("Unknown hand action frame")
    if body is not None:
        if body.reanchor:
            out = move_children(out, names, body.reanchor, actual, out[:, :7].clone())
        owned = sorted(i for ids in tree.owned.values() for i in ids)
        unread = [i for i in owned if names[i] not in body.position_read]
        out[unread, :3] = tensor(template)[unread, :3]
        if "gaze" in tree.owners:
            g = names.index("gaze")
            out[g, 3:7] = eye_row_physical(out[g, 3:7], body.eye_row_convention)   # the map is its own inverse
    return out


def through_synergy(synergy, tree, start, requested, template):
    """Use a detached copy of the learned synergy; preserve its live centres and weights.

    Projection can change the hierarchy's predicted consequence. Clone comparison
    measures that discrepancy; reconstruction alone cannot pass the physical gate.
    `requested` is an issued command: its finger rows sit on its own wrist row, because the hand controller solves
    the finger goal against the ORDERED wrist (hand_target_control.project_world_orders). realize() builds it with
    the hierarchy's hand frame first, so the synergy is asked for the same rows the handle path would issue.
    """
    from lem.latent_body.trajectory import rotation_delta
    module = copy.deepcopy(synergy).cpu().eval().requires_grad_(False)
    if tuple(module.handle_names) != tree.handle_names:
        raise ValueError("Synergy and hierarchy handle layouts differ")
    with torch.no_grad():
        achieved = torch.zeros(1, len(tree.handle_names), 10)
        achieved[0, :, :3] = start.bones[:, 4:7] / 200.0
        achieved[0, :, 3:7] = quaternion(start.bones[:, :4])
        achieved[0, :, 7:9] = tensor(template)[:, 7:9]
        achieved[0, :, 9] = 1
        raw_template = tensor(template).unsqueeze(0)
        raw_template[..., :3] /= 200.0
        target = tensor(requested)
        rot = rotation_delta(start.bones[module.rot_idx, :4], target[module.rot_idx, 3:7]).unsqueeze(0)
        pos = (target[module.pos_idx, :3] - start.bones[module.pos_idx, 4:7]).unsqueeze(0)
        z = module.encode(module.to_units(rot, pos), achieved)
        turns, moves = module.delta(z, achieved)
        mapped = module.order_from_delta(turns, moves, achieved, raw_template)[0]
        mapped[:, :3] *= 200.0
        out = tensor(template)
        owned = sorted(i for indices in tree.owned.values() for i in indices)
        out[owned, :7] = mapped[owned, :7]
        return out


def realize(choice, tree, start, template, scales, *, synergy=None, hand_frame="recorded", digit_frame="world",
            digit_scales=None, body=None):
    requested = handle_targets(tree, choice.predictions, start, template, scales, hand_frame, digit_frame,
                               digit_scales, body)
    return requested if synergy is None else through_synergy(synergy, tree, start, requested, template)
