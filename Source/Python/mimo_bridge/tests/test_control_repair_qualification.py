# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Repair admission must catch lost physical history and omitted execution records."""
import copy
import importlib
from pathlib import Path

import pytest


def qualification(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4]/'docs/gain_of_function_v3'))
    return importlib.import_module('qualify_control_repairs_2026_09_24')


def rows():
    return [dict(type='mimo',step=step,physical_goal_mode='off',
        hand_control=dict(contract='wrist_relative_joint_v1',canonical_readback_failures=0,
            physical_substeps_recorded=200,body_observations=40,joint_error_deg=4.),
        efference=dict(liveness={name:.75 for name in ('body_mlp.0','body_mlp.2','gaze_in','vision_in')}))
        for step in range(380205,380301,5)]


def test_all_execution_steps_and_dedicated_layers_required(monkeypatch):
    q=qualification(monkeypatch)
    records=rows()
    assert all(q.control_checks(records,380200,100,5,'off').values())
    lost=copy.deepcopy(records)
    lost[4]['hand_control']['physical_substeps_recorded']-=1
    assert not q.control_checks(lost,380200,100,5,'off')['complete_hand_execution']
    lost=copy.deepcopy(records)
    del lost[4]['efference']['liveness']['vision_in']
    assert not q.control_checks(lost,380200,100,5,'off')['dedicated_efference_responses']
    records.append(dict(type='episode_reset',step=380201))
    assert not q.control_checks(records,380200,100,5,'off')['no_world_resets']


@pytest.mark.parametrize('value',[float('nan'),float('inf'),.499])
def test_nonfinite_or_failed_eye_layer_cannot_pass(monkeypatch,value):
    q=qualification(monkeypatch)
    records=rows()
    records[-1]['efference']['liveness']['vision_in']=value
    assert not q.control_checks(records,380200,100,5,'off')['dedicated_efference_responses']


def test_restart_failure_preserves_new_updates(monkeypatch,tmp_path):
    qualification(monkeypatch)
    module=importlib.import_module('resume_control_repairs_2026_09_24')
    run=tmp_path/'run';run.mkdir()
    pause=tmp_path/'supervisor_control.json';pause.write_text('{"cmd":"run"}')
    monkeypatch.setattr(module,'OUT',run)
    monkeypatch.setattr(module.D,'PAUSE',pause)
    module.pause_and_save('injected error')
    import json
    assert json.loads(pause.read_text())['cmd']=='pause'
    assert json.loads((run/'control.json').read_text())['cmd']=='snapshot_and_stop'
    assert json.loads((run/'stop_reason.json').read_text())['resume_policy']=='repair_required'


def test_restart_observes_the_full_qualified_interval(monkeypatch):
    qualification(monkeypatch)
    module=importlib.import_module('resume_control_repairs_2026_09_24')
    assert module.startup_verification_steps({'steps':300})==300
    assert module.startup_verification_steps({'steps':600})==600
    for invalid in (100,299,300.5,None):
        with pytest.raises(ValueError):
            module.startup_verification_steps({'steps':invalid})


def test_production_length_response_windows_preserve_all_checks(monkeypatch):
    q=qualification(monkeypatch)
    assert q.reporting_period(300,100,{'report_every':100})==100
    for steps,period in ((300,5),(305,100),(300,0)):
        with pytest.raises(ValueError):
            q.reporting_period(steps,period,{'report_every':100})
    records=[]
    for step in (380300,380400,380500):
        sampled=sum(s%7==0 for s in range(step-100,step))
        records.append(dict(type='mimo',step=step,physical_goal_mode='off',
            hand_control=dict(contract='wrist_relative_joint_v1',canonical_readback_failures=0,
                physical_substeps_recorded=4000,body_observations=800,joint_error_deg=4.),
            efference=dict(liveness={name:.75 for name in ('body_mlp.0','body_mlp.2','gaze_in','vision_in')}),
            liveness=dict(sample_every=7,window_steps=100,sampled_steps=sampled,samples=sampled*8,
                layers={'observed':.75},layer_line=.5,below_line=[],
                watching={'observed':{'ready':step==380500,'samples':800,'needed':800}})))
    assert all(q.control_checks(records,380200,300,5,'off').values())
    assert q.assess_liveness(records,380200,380500,bodies=8)['passed']
    records[-1]['efference']['liveness']['body_mlp.0']=.499
    assert not q.control_checks(records,380200,300,5,'off')['dedicated_efference_responses']
    records[-1]['liveness']['samples']-=8
    assert not q.assess_liveness(records,380200,380500,bodies=8)['passed']


def test_hand_goal_mode_must_be_reported_in_every_qualified_window(monkeypatch):
    q=qualification(monkeypatch);records=rows()
    assert not q.control_checks(records,380200,100,5,'off','memory')['hand_goal_mode']
    for row in records:row['hand_goal_mode']='memory'
    assert q.control_checks(records,380200,100,5,'off','memory')['hand_goal_mode']
    records[-1]['hand_goal_mode']='off'
    assert not q.control_checks(records,380200,100,5,'off','memory')['hand_goal_mode']


@pytest.mark.parametrize('changed',['calibration_file','saved_receipt','preserved_source'])
def test_first_production_save_rechecks_qualified_identity(monkeypatch,tmp_path,changed):
    import hashlib
    import json
    qualification(monkeypatch)
    module=importlib.import_module('resume_control_repairs_2026_09_24')
    binding=importlib.import_module('calibration_binding')
    calibration=tmp_path/'calibration.json'
    spec=dict(version='visual_linear_input_conditioning_v1',scale=2.,source_model_sha256='a'*64)
    calibration.write_text(json.dumps(spec))
    changes=dict(vision_input_calibration=str(calibration),hand_goals='off')
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    receipt=dict(visual_input_conditioning=dict(spec,
        calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest()))
    (checkpoint/'model_receipt.json').write_text(json.dumps(receipt))
    source={'model.safetensors':'a'*64,'optimizer.pt':'b'*64,'runtime_state.pt':'c'*64}
    monkeypatch.setattr(module,'source_hashes',lambda _:dict(source))
    proof=dict(changes=changes,calibration_binding=binding.capture(changes),source_sha256=dict(source))
    assert module.verify_startup_identity(proof,checkpoint)
    if changed=='calibration_file':
        spec['scale']=4.;calibration.write_text(json.dumps(spec))
    elif changed=='saved_receipt':
        receipt['visual_input_conditioning']['scale']=4.
        (checkpoint/'model_receipt.json').write_text(json.dumps(receipt))
    else:
        source['optimizer.pt']='d'*64
    with pytest.raises((ValueError,RuntimeError)):
        module.verify_startup_identity(proof,checkpoint)
