# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Explicit MIMo body extensions; the UE shared-memory layout remains unchanged."""
from lem.model.body_sets import HANDLE_NAMES

TRUNK_NAMES = ('lower_trunk', 'chest')
TRUNK_BODIES = dict(lower_trunk='lower_body', chest='chest')


def names_for(layout):
    if layout == 'legacy':
        return HANDLE_NAMES
    if layout == 'trunk_v1':
        return HANDLE_NAMES + TRUNK_NAMES
    raise ValueError(f'unknown body layout: {layout}')


def names_for_rows(count):
    """The handle names of the layout whose arrays have `count` rows (24 legacy, 26 trunk_v1),
    for code that sees only an array. A body part is picked by its name here, never by a fixed row
    number: the trunk rows come after the eyes' row, so 'the last row' and 'the first 24 rows'
    mean different parts in the two layouts (fix-following-gauge, 2026-09-23)."""
    for layout in ('legacy', 'trunk_v1'):
        if len(names_for(layout)) == count:
            return names_for(layout)
    raise ValueError(f'no body layout has {count} handle rows')


def extend_limits(limits, count):
    result = dict(limits)
    if result.get('handle_mask') is not None:
        mask = list(result['handle_mask'])
        if len(mask) > count:
            raise ValueError('developmental mask exceeds body layout')
        # The trunk uses the same ROM, strength and exploration limits as the body.
        result['handle_mask'] = mask + [True] * (count-len(mask))
    return result


def config_overrides(layout):
    n = len(names_for(layout))
    if layout == 'legacy':
        return {}
    # The action buffer does NOT grow with the body. Its size is the UE consumer's
    # compile-time frame: ModelBridgeComponent.h asserts sizeof(FActionFrame) == 1176
    # with NUM_HANDLES 23, GAZE_TARGET at 848 and VOICE_PARAMS at 856, and
    # lem/action/writer.py packs to exactly those offsets. Deriving 20+n*9*4+8+320 from
    # n=26 wrote 1284 into the config - 108 bytes, 3 handles, past the frame - which put
    # the configured gaze field at 956 where the consumer reads 848.
    from lem.action.writer import ACTION_FRAME_SIZE
    return dict(body=dict(n_handles=n, tracked_bones=n, layout=layout),
                modalities=dict(proprioception=dict(pipeline=dict(input_dim=n*12)),
                                motor=dict(pipeline=dict(n_handles=n))),
                model=dict(wellness=dict(grounded=dict(motor_state=dict(input_dim=n*9)))),
                action_buffer=dict(buffer_size=ACTION_FRAME_SIZE))


def chest_axes(rotation,head_position,hip_position):
    """Calibrate anatomical directions while the reset body is lying on its back."""
    import numpy as np
    longitudinal=rotation.T@(np.asarray(head_position)-np.asarray(hip_position))
    longitudinal=longitudinal/np.linalg.norm(longitudinal)
    front=rotation.T@np.array([0.,0.,1.])
    front=front-longitudinal*np.dot(front,longitudinal)
    if np.linalg.norm(front)<1e-6:return None
    front/=np.linalg.norm(front)
    return np.stack([longitudinal,front,np.cross(longitudinal,front)])


def chest_orientation(rotation,axes):
    """Roll about the body's length, independent of turning around the pen.

    Zero: lying on back; 90 degrees: side; 180: belly facing down. Near
    vertical the roll is undefined. This measures posture, not its cause.
    """
    import numpy as np
    if axes is None:
        return dict(roll_degrees=None,pitch_degrees=None,front_up_component=None,valid=False)
    along,front,side=axes@(rotation.T@np.array([0.,0.,1.]))
    valid=float(np.hypot(front,side))>1e-6
    return dict(roll_degrees=float(np.degrees(np.arctan2(side,front))) if valid else None,
                pitch_degrees=float(np.degrees(np.arcsin(np.clip(along,-1,1)))),
                front_up_component=float(front),valid=valid)
