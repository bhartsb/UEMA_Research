# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import copy
import numpy as np
import pytest

from mimo_bridge.tools.native_local_audit_probe import completed_audit_evidence


def evidence():
    modalities=('vision','tactile','audio','proprioception')
    row=dict(prediction_mse=[.2]*8,action_control_mse=[.3]*8,
             persistence_mse=[.4]*8,eligible=[True]*8)
    return dict(completed=1,anchors=[dict(id='earned')],
        last=dict(completed_comparisons=1,latest_comparisons={m:copy.deepcopy(row) for m in modalities}),
        pending=dict(step=10,**{key:{m:np.ones((8,4)) for m in modalities}
                              for key in ('prediction','shuffled','reference')}))


def test_completed_audit_with_next_pending_is_valid():
    payload,step=completed_audit_evidence(evidence(),11)
    assert payload['completed_comparisons']==1
    assert step==10


def test_completed_audit_without_pending_is_valid():
    audit=evidence();audit['pending']=None
    assert completed_audit_evidence(audit,11)[1] is None


def test_uncounted_motor_predictions_may_coexist_with_sensory_audit():
    audit=evidence()
    for key in ('prediction','shuffled','reference'):
        audit['pending'][key]['motor']=np.ones((8,18))
    assert completed_audit_evidence(audit,11)[1]==10


@pytest.mark.parametrize('damage',('missing_completed','counter_mismatch','missing_modality',
                                  'missing_body','nan_outcome','stale_pending','nan_pending'))
def test_invalid_audit_evidence_is_rejected(damage):
    audit=evidence()
    if damage=='missing_completed':audit['completed']=0
    elif damage=='counter_mismatch':audit['last']['completed_comparisons']=2
    elif damage=='missing_modality':del audit['last']['latest_comparisons']['vision']
    elif damage=='missing_body':audit['last']['latest_comparisons']['vision']['eligible']=[True]*7
    elif damage=='nan_outcome':audit['last']['latest_comparisons']['tactile']['prediction_mse'][3]=float('nan')
    elif damage=='stale_pending':audit['pending']['step']=5
    elif damage=='nan_pending':audit['pending']['prediction']['audio'][1,2]=float('nan')
    with pytest.raises(ValueError):completed_audit_evidence(audit,11)
