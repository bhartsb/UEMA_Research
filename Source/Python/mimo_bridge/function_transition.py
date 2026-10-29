# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Explicit observation-only -> physical-function transition of a saved episode.

The last observed frame remains old experience. Modulation begins with the next
physical tick; no past sensor value or learned target is rewritten as impairment.
"""
from copy import deepcopy
import numpy as np


def activate_function(episode):
    if episode['config'].get('body_function') != 'observe':
        raise ValueError('function activation requires an observation-only episode')
    if episode['config'].get('self_state') != 'learn':
        raise ValueError('function activation requires the self-perception learner')
    updated=deepcopy(episode)
    updated['config']['body_function']='modulate'
    records=[]
    for saved in updated['worlds']:
        if saved['env_kwargs'].get('body_function') != 'observe':
            raise ValueError('world and episode function modes disagree')
        attrs=saved['world']['attrs'];function=attrs.get('body_function')
        info=saved['payload']['info'];observed=info.get('body_function',{})
        if function is None or observed.get('mode') != 'observe':
            raise ValueError('missing last observation and physical-function history')
        q=np.asarray(observed['values'])[:,0]
        if not np.array_equal(q,np.ones_like(q)):
            raise ValueError('observation-only frame claims actual impaired sensing')
        if function.last_time != saved['payload']['sim_time']:
            raise ValueError('function clock does not match the continuing body')
        records.append(dict(time=function.last_time,site_ids=function.site_ids,
            shadow_state=function.state_dict(),
            fatigue_capacity=deepcopy(saved['world']['fatigue']['cap'])))
        # Shadow calculations were never experienced. Preserve them in the
        # receipt, then start the newly active operator from actual last quality.
        function.quality[:]=q
        saved['env_kwargs']['body_function']='modulate'
        # Some later snapshot versions may store this previously constructor-only
        # field. Set it when present; never change the actual old frame's regime.
        if 'body_function_mode' in attrs:attrs['body_function_mode']='modulate'
    updated['function_transition']=dict(version='actual_function_activation_v1',worlds=records,
        old_observations_preserved=True,first_modulated_observation='next physical tick',
        motor_capacity='Existing fatigue state begins limiting fixed actuator gain on the next tick.')
    return updated
