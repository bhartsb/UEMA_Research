# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Explicit checkpoint-compatible boundary for the virtual local skin operator."""
from copy import deepcopy


# Which profile an episode may be carried FROM. Both write the same history
# contract (lem.skin_mechanics.MECHANICAL_CONTRACT), so the one frame sitting at
# the boundary is readable by the profile being switched to; a v3 or v4 episode is
# NOT here because it already writes contact_footprint.CONTRACT and needs no
# boundary at all to reach the other of the two.
TRANSFERABLE_SOURCES = ('mechanical_v2', 'mechanical_v2_local')
LOCAL_SKIN_TARGETS = ('mechanical_v2_local', 'mechanical_v3', 'mechanical_v4')


def activate_local_skin(episode,target='mechanical_v2_local'):
    if target not in LOCAL_SKIN_TARGETS:raise ValueError('unknown local skin profile')
    source=episode['config'].get('contact_profile')
    if source not in TRANSFERABLE_SOURCES:
        raise ValueError('virtual skin transition requires an episode on '
                         f'{" or ".join(TRANSFERABLE_SOURCES)}, not {source!r}')
    if source==target:
        raise ValueError(f'episode is already on {target}')
    if episode['config'].get('body_function') not in ('observe','modulate'):
        raise ValueError('virtual skin transition requires preserved physical function history')
    updated=deepcopy(episode);updated['config']['contact_profile']=target
    for i,saved in enumerate(updated['worlds']):
        if saved['env_kwargs'].get('contact_profile')!=source:raise ValueError('world profile disagrees with episode')
        function=saved['world']['attrs'].get('body_function')
        if function is None or function.last_time!=saved['payload']['sim_time']:
            raise ValueError('virtual skin cannot reconstruct missing physical-function history')
        transition=dict(version='virtual_skin_boundary_v1',source=source,target=target,
                        after_time=saved['payload']['sim_time'])
        saved['env_kwargs']['contact_profile']=target
        saved['world']['attrs']['_virtual_skin']=None
        saved['world']['attrs']['_virtual_skin_transition']=transition
        # The actual old pressure, q, sensory bytes and prediction history are
        # preserved. This annotation permits exactly that old frame at the
        # boundary; future frames must use the new transducer's contract.
        saved['payload']['info']['contact_profile_transition']=transition
        if 'frame_meta' in updated['trainer']:
            updated['trainer']['frame_meta'][i]['contact_profile_transition']=deepcopy(transition)
    updated['virtual_skin_transition']=dict(version='virtual_skin_boundary_v1',
        previous_observations_unchanged=True,old_body_quality_inherited_by_existing_taxels=True,
        old_pressure_history_inherited=True,new_operator_begins='next physical tick',production_choice_license=False)
    return updated


def activate_virtual_skin(episode):
    return activate_local_skin(episode,target='mechanical_v3')


def activate_graded_impact_skin(episode):
    """Carry an episode onto mechanical_v4: resolved contact geometry AND the
    aversion shape that keeps separating loads at the pressures falls produce.

    Measured 2026-09-16, 12 drops a height at age 24 months, pain rising at every
    rung with no two tied: 0.1466, 0.1624, 0.7025, 0.8711, 0.8808, 0.9134 across
    5, 10, 30, 45, 66 and 90 cm. The profile this replaces read 0.0000, 0.0000,
    0.9991, 0.9975, 0.9945, 0.9164 - free at the bottom and backwards at the top.

    THE BODY WILL FEEL DIFFERENT FOR DOING NOTHING. Lying still scores comfort
    0.5461 under v4 against 0.7081 under mechanical_v2_local, because the resolved
    footprint is smaller than the sensor support area and the same rest reads
    73,844 Pa instead of 17,024 Pa. Anything the run learned to predict about how
    an action will feel was fitted against the old numbers.
    """
    return activate_local_skin(episode,target='mechanical_v4')
