# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import copy
import numpy as np
import pytest
from lem.model.learning_opportunity import LearningOpportunity


def meter(groups=None):
    return LearningOpportunity(4,{'sense':1e-8,'same_sense':1e-8},'constructed calibration only',groups)


def improving(m,modality='sense'):
    for i in range(4):
        m.observe_fresh(0,modality,prediction=.1,shuffled=1.,persistence=1.,time=float(i),healthy=True)
        m.observe_anchor(0,modality,str(i),reference=.5,prediction=.1,healthy=True)


def test_controlled_learning_changes_healthy_candidate_preference():
    m=meter();improving(m)
    p=np.array([[0.,0.],[.2,.2],[.7,.7]])
    values=m.score(0,{'sense':p},{'sense':p[0]},eligible=[True]*3,healthy=True)
    assert values.argmax()==2 and 0<values[2]<=1 and values[0]==0


def test_mastery_and_relearning_after_regression_do_not_keep_paying():
    m=meter();improving(m);assert m.available(0,'sense')[0]>0
    for i in range(4):m.observe_anchor(0,'sense',str(i),reference=.5,prediction=.1,healthy=True)
    assert m.available(0,'sense')[0]==0
    for i in range(4):m.observe_anchor(0,'sense',str(i),reference=.5,prediction=.8,healthy=True)
    for i in range(4):m.observe_anchor(0,'sense',str(i),reference=.8,prediction=.1,healthy=True)
    assert m.available(0,'sense')[0]==0
    m.reset_episode();improving(m)
    assert m.available(0,'sense')[0]==0


@pytest.mark.parametrize('kind',['noise','collapse','stillness','missing_action_gain'])
def test_unlearnable_or_motionless_effects_do_not_become_opportunities(kind):
    m=meter()
    for i in range(4):
        prediction,shuffled,persistence=(.1,.1,1.) if kind in ('noise','missing_action_gain') else (0.,0.,0.)
        m.observe_fresh(0,'sense',prediction=prediction,shuffled=shuffled,persistence=persistence,time=i,healthy=True)
        m.observe_anchor(0,'sense',str(i),reference=1.,prediction=.1,healthy=True)
    assert m.available(0,'sense')[0]==0


def test_impaired_body_and_nonviable_future_cannot_buy_exploration():
    m=meter();improving(m)
    p={'sense':np.ones((3,2))};h={'sense':np.zeros(2)}
    assert not m.score(0,p,h,eligible=[True]*3,healthy=False).any()
    assert not m.score(0,p,h,eligible=[False]*3,healthy=True).any()
    m.observe_fresh(0,'sense',prediction=.01,shuffled=1.,persistence=1.,time=4,healthy=False)
    assert m.available(0,'sense')[0]==0


def test_body_and_sensor_duplication_do_not_multiply_drive():
    m=meter({'same_sense':'sense'});improving(m);improving(m,'same_sense')
    p=np.array([[0.,0.],[.5,.5]]);h=p[0]
    a=m.score(0,{'sense':p},{'sense':h},eligible=[True]*2,healthy=True)
    b=m.score(0,{'sense':np.repeat(p,50,1),'same_sense':p},
              {'sense':np.repeat(h,50),'same_sense':h},eligible=[True]*2,healthy=True)
    np.testing.assert_array_equal(a,b)
    assert not m.score(1,{'sense':p},{'sense':h},eligible=[True]*2,healthy=True).any()


def test_fixed_anchor_history_roundtrip_and_repeated_evidence_refusal():
    m=meter();improving(m);state=m.state_dict();restored=meter();restored.load_state_dict(state)
    assert restored.available(0,'sense')==m.available(0,'sense')
    m.best.clear();assert restored.best
    with pytest.raises(ValueError,match='replayed'):
        restored.observe_fresh(0,'sense',prediction=.1,shuffled=1.,persistence=1.,time=3,healthy=True)
    with pytest.raises(ValueError,match='nonfinite'):
        restored.observe_fresh(0,'sense',prediction=float('nan'),shuffled=1.,persistence=1.,time=4,healthy=True)


def test_new_anchor_timestamps_cannot_hide_regression_on_immutable_retention_panel():
    m=LearningOpportunity(4,{'sense':1e-8},'constructed calibration only',require_retention=True)
    improving(m)
    assert m.available(0,'sense')[0]==0
    m.observe_retention(0,'sense','fixed original panel',prediction=.1,reference=.5)
    assert m.available(0,'sense')[0]>0
    m.observe_retention(0,'sense','fixed original panel',prediction=.3,reference=.5)
    for i in range(4):m.observe_anchor(0,'sense','new timestamp '+str(i),reference=.9,prediction=.1,healthy=True)
    assert m.available(0,'sense')[0]==0
    assert 'worse than earned best' in m.explain(0,'sense')['reason']
    # A reference refresh cannot lower the standard or swap the old panel out.
    m.observe_retention(0,'sense','fixed original panel',prediction=.3,reference=.9)
    assert m.available(0,'sense')[0]==0
    with pytest.raises(ValueError,match='panel changed'):
        m.observe_retention(0,'sense','new easier panel',prediction=0.,reference=1.)
    restored=LearningOpportunity(4,{'sense':1e-8},'constructed calibration only',require_retention=True)
    restored.load_state_dict(m.state_dict())
    assert restored.retention==m.retention and restored.available(0,'sense')[0]==0


def test_repeated_lesson_after_time_and_other_body_changes_cannot_reearn_progress():
    import torch
    from mimo_bridge.opportunity_integration import anchor_identities
    inputs=dict(tokens={'sense':torch.zeros(2,3)},context=torch.zeros(2,2,3),
        achieved=torch.zeros(2,1,9),memory_bias=torch.zeros(2,3),source_mask={'sense':'real'},time=torch.zeros(2))
    command=torch.zeros(2,1,9);target={'sense':torch.ones(2,3)}
    first=anchor_identities(inputs,command,target)
    m=LearningOpportunity(4,{'sense':1e-8},'constructed calibration only',require_retention=True)
    m.observe_retention(0,'sense','unchanged other skill',prediction=.1,reference=.5)
    for i in range(4):
        m.observe_fresh(0,'sense',prediction=.1,shuffled=1.,persistence=1.,time=i,healthy=True)
        m.observe_anchor(0,'sense',first[0],reference=.5,prediction=.1,healthy=True)
    # Only body 1 changes. Body 0 forgets and then relearns the exact old lesson.
    inputs['time']+=100
    inputs['tokens']['sense'][1]+=8
    repeated=anchor_identities(inputs,command,target)
    assert repeated[0]==first[0] and repeated[1]!=first[1]
    for i in range(4):m.observe_anchor(0,'sense',repeated[0],reference=.8,prediction=.8,healthy=True)
    for i in range(4):m.observe_anchor(0,'sense',repeated[0],reference=.8,prediction=.1,healthy=True)
    assert m.available(0,'sense')[0]==0
    assert m.retention[('0','sense')]['preserved']


@pytest.mark.parametrize('changed',['context','achieved','memory_bias','target'])
def test_different_effective_prediction_problem_keeps_distinct_evidence(changed):
    import torch
    from mimo_bridge.opportunity_integration import anchor_identities
    inputs=dict(tokens={'sense':torch.zeros(2,3)},context=torch.zeros(2,2,3),
        achieved=torch.zeros(2,1,9),memory_bias=torch.zeros(2,3),source_mask={'sense':'real'},time=torch.zeros(2))
    command=torch.zeros(2,1,9);target={'sense':torch.ones(2,3)}
    first=anchor_identities(inputs,command,target)
    if changed=='target':target['sense'][0]+=1
    else:inputs[changed][0]+=1
    after=anchor_identities(inputs,command,target)
    assert first[0]!=after[0] and first[1]==after[1]
