# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import numpy as np
import pytest
from mimo_bridge.tools.caregiver_learning_trial import donor_map,alignment,paired_result
from mimo_bridge.train_mimo import build_learner,parse_args


def test_yoked_control_never_uses_a_worlds_own_recording():
    for n in (2,6,12):
        donors=donor_map(n)
        assert sorted(donors)==list(range(n))
        assert all(i!=donor for i,donor in enumerate(donors))
    with pytest.raises(ValueError):donor_map(1)


def test_alignment_endpoints_and_constant_gaze_cannot_create_a_difference():
    eye=np.zeros(3);target=np.array([0,0,1.])
    assert alignment(eye,target,target)==1
    assert alignment(eye,-target,target)==0
    assert alignment(eye,np.array([1.,0,0]),target)==.5
    assert paired_result([0.]*6)['paired_permutation_p']==1
    with pytest.raises(ValueError):alignment(eye,target,eye)


def test_planted_benefit_is_detectable_and_reverses_with_arm_labels():
    p=paired_result([.1]*6);q=paired_result([-.1]*6)
    assert p['paired_permutation_p']==.03125 and p['mean']>0
    assert q['paired_permutation_p']==p['paired_permutation_p'] and q['mean']<0
    assert not paired_result([0,float('nan')])['valid']


def test_copied_trials_use_the_production_constructor_including_eye_teacher(monkeypatch,tmp_path):
    # The stand-in is an object: build_learner sets settings on the learner it built (since
    # 4ff86f130, which replay windows it studies), and a copied trial must get them too.
    seen={};made=[]
    class Learner:
        def __init__(self,**kwargs):seen.update(kwargs);made.append(self)
    monkeypatch.setattr('mimo_bridge.train_mimo.LemLite',Learner)
    a=parse_args(['--out',str(tmp_path),'--anticipation-gaze','on','--anticipation-gaze-w','.13',
                  '--audio','real','--audio-freq-bands','on','--intention-every','7',
                  '--parent-relative-targets','calf_chest','--position-rows','no_head_pelvis'])
    model=build_learner(a,str(tmp_path))
    assert made==[model]
    assert seen['anticipation_gaze'] and seen['anticipation_gaze_w']==.13
    assert seen['intention_every']==7
    assert seen['overrides']['modalities']['audio']['encoder']['freq_band_path']=='on'
    assert (model.parent_relative_targets,model.position_rows)==('calf_chest','no_head_pelvis')


def test_trial_rejects_missing_receipts_reinitialized_components_and_lost_optimizer(tmp_path):
    from mimo_bridge.tools.caregiver_learning_trial import verify_loaded
    good=dict(loaded_components=['encoder:vision'],reinitialized_components=[])
    for receipt in ({},dict(good,reinitialized_components=['encoder:vision'])):
        with pytest.raises(ValueError):verify_loaded(None,receipt,tmp_path,False)
    # Actual by-name migration and corruption checks use saved tensors and real
    # optimizer parameter groups in test_copied_optimizer_verification.py.


def test_the_trial_builds_its_learner_through_build_learner_from_the_parent_runs_saved_settings(monkeypatch,tmp_path):
    """The test above covers the builder; this one runs the trial's own settings() and learner(). The trial
    must read the parent run's args.json and build through train_mimo.build_learner, so its copy gets
    production's constructor keywords and the two replay settings build_learner sets after construction.
    Review of 2026-09-25 (prod_tests_review_2026-09-25.md, finding 2): a trial that built LemLite itself,
    or dropped the parent run's parent_relative_targets and position_rows, passed every test. Builds no
    model and no world and reads nothing under Saved/."""
    import json
    from argparse import Namespace
    from mimo_bridge import train_mimo
    from mimo_bridge.tools import caregiver_learning_trial as trial
    class Learner:
        def __init__(self,**kwargs):self.kwargs=kwargs
        def load_checkpoint(self,path,load_optimizer=True):
            self.loaded=(path,load_optimizer)
            return dict(step=7,loaded_components=['encoder:vision'],reinitialized_components=[])
        def reset_transition_state(self):self.reset=True
    class Direct(Learner):
        """What the trial gets if it constructs its own LemLite instead of calling build_learner."""
    monkeypatch.setattr(train_mimo,'LemLite',Learner)
    monkeypatch.setattr(trial,'LemLite',Direct)
    monkeypatch.setattr(trial,'validate_checkpoint',lambda path:None)   # the stand-in checkpoint has no files
    run=tmp_path/'parent';checkpoint=run/'checkpoints'/'ckpt_step_7';checkpoint.mkdir(parents=True)
    # the parent run's settings, each away from the parser's default so a dropped one reads differently
    saved=vars(parse_args(['--out',str(run),'--hand-target-mode','wrist_relative_v1','--intention-every','7',
                           '--parent-relative-targets','calf_chest','--position-rows','no_head_pelvis']))
    (run/'args.json').write_text(json.dumps(saved),encoding='utf-8')
    speech=tmp_path/'speech';speech.mkdir()
    a=trial.settings(checkpoint,tmp_path/'out',speech,'cpu',3)
    copy=tmp_path/'copy'
    model,step=trial.learner(a,checkpoint,copy,False)
    assert type(model) is Learner, 'the trial must build through train_mimo.build_learner'
    assert step==7 and model.loaded==(str(checkpoint),False) and model.reset
    assert (model.parent_relative_targets,model.position_rows)==('calf_chest','no_head_pelvis')
    assert model.kwargs['intention_every']==7 and model.kwargs['hand_target_mode']=='wrist_relative_v1'
    # every constructor keyword and every setting made after construction is what production's builder
    # makes from the parent's own settings, with the trial's device, seed and checkpoint folder
    reference=build_learner(Namespace(**dict(saved,device='cpu',seed=3)),str(copy/'checkpoints'))
    assert model.kwargs==reference.kwargs
    made_after=lambda learner:{k:v for k,v in vars(learner).items() if k not in ('kwargs','loaded','reset')}
    assert made_after(model)==made_after(reference)=={'parent_relative_targets':'calf_chest',
                                                      'position_rows':'no_head_pelvis'}
