# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Native actor selection; legacy shadows require explicit migration, never a pass."""
from mimo_bridge.acting_snapshot import FrozenActingSnapshot,CONTRACT as LEGACY,calibration_for_candidate
from mimo_bridge.decorated_acting_snapshot import DecoratedActingSnapshot,CONTRACT,record_manifest


def inner_state_for_report(state):
    """Read counters only. This never restores or executes a legacy inner state."""
    if state is None:return None
    if state.get('contract')==CONTRACT:return state['actor']
    if state.get('contract')==LEGACY:return state
    raise ValueError('unknown actor state contract')


def require_outer_actor(actor):
    if not isinstance(actor,DecoratedActingSnapshot):
        raise ValueError('legacy shadow needs explicit decorated-code migration before qualification')


def actor_for_run(learner,factory,configuration,mode,receipt,*,saved=None):
    if mode not in ('shadow','on'):raise ValueError('actor selection needs shadow or qualified mode')
    if saved is not None:
        details=inner_state_for_report(saved)
        if details['configuration']!=configuration:raise ValueError('acting configuration changed without qualification')
        if bool(details['qualification'])!=(mode=='shadow'):
            raise ValueError('saved qualification cannot change through a run-mode flag')
        if saved['contract']==CONTRACT:
            actor=DecoratedActingSnapshot.restore(saved,factory,learner.device,allow_qualification=mode=='shadow')
        else:
            if mode!='shadow':raise ValueError('legacy qualified actors require new decorated-code qualification')
            actor=FrozenActingSnapshot.restore(saved,factory,learner.device,allow_qualification=True)
    else:
        current=receipt if mode=='on' else calibration_for_candidate(learner,receipt,configuration)
        inner=FrozenActingSnapshot.from_learner(learner,factory,current,configuration,qualification=mode=='shadow')
        actor=DecoratedActingSnapshot(inner,record_manifest(inner.model))
    if mode=='on':
        require_outer_actor(actor)
        if actor.receipt!=receipt:raise ValueError('acting receipt changed without explicit snapshot replacement')
    return actor
