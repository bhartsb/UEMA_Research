# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import importlib
import hashlib
import json
from pathlib import Path
import pytest


def test_changed_calibration_refuses_resume_even_if_runtime_and_source_stay_same(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4]/'docs/gain_of_function_v3'))
    binding=importlib.import_module('calibration_binding')
    path=tmp_path/'measured.json'
    spec=dict(version='visual_linear_input_conditioning_v1',scale=256.,source_model_sha256='a'*64)
    path.write_text(json.dumps(spec))
    changes={'vision_input_calibration':str(path)}
    identity=binding.capture(changes)
    receipt={'visual_input_conditioning':dict(spec,calibration_sha256=hashlib.sha256(path.read_bytes()).hexdigest())}
    assert binding.assert_bound(changes,identity,receipt)
    spec['scale']=128.;path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='changed after qualification'):
        binding.assert_bound(changes,identity,receipt)
    # Merely updating the saved identity cannot forge what the learner loaded.
    with pytest.raises(ValueError,match='Loaded calibration receipt'):
        binding.assert_bound(changes,binding.capture(changes),receipt)


def test_calibration_payload_and_every_enabled_calibration_are_required(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4]/'docs/gain_of_function_v3'))
    binding=importlib.import_module('calibration_binding')
    changes={};receipt={}
    for flag,key in binding.FIELDS.items():
        path=tmp_path/(flag+'.json')
        hand=flag=='hand_response_calibration'
        spec=(dict(version='hand_joint_response_v1',sha256='c'*64,source_sha256={'fit':'d'*64},
                   coefficients=[[.19357185554100118,0.,0.]]) if hand else
              dict(version=key,source_model_sha256='b'*64,**({'scale':2.} if 'input' in flag else {'shift':[3.],'gain':[.5]})))
        path.write_text(json.dumps(spec));changes[flag]=str(path)
        receipt[key]=dict(spec,**{('calibration_file_sha256' if hand else 'calibration_sha256'):
                                  hashlib.sha256(path.read_bytes()).hexdigest()})
    identity=binding.capture(changes)
    assert binding.assert_bound(changes,identity,receipt)
    receipt['visual_activation_compensation']['gain']=[.75]
    with pytest.raises(ValueError,match='payload differs'):
        binding.assert_bound(changes,identity,receipt)


def test_hand_coefficients_cannot_change_between_qualification_and_resume(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4]/'docs/gain_of_function_v3'))
    binding=importlib.import_module('calibration_binding')
    visual=tmp_path/'visual.json';visual.write_text(json.dumps(dict(scale=2.,source_model_sha256='a'*64)))
    hand=tmp_path/'hand.json'
    spec=dict(version='hand_joint_response_v1',sha256='b'*64,source_sha256={'fit':'c'*64},coefficients=[[.25,0.,0.]])
    hand.write_text(json.dumps(spec))
    changes=dict(vision_input_calibration=str(visual),hand_goals='memory',hand_response_calibration=str(hand))
    receipt=dict(visual_input_conditioning=dict(scale=2.,source_model_sha256='a'*64,
        calibration_sha256=hashlib.sha256(visual.read_bytes()).hexdigest()),
        hand_response=dict(spec,calibration_file_sha256=hashlib.sha256(hand.read_bytes()).hexdigest()))
    identity=binding.capture(changes)
    assert binding.assert_bound(changes,identity,receipt)
    spec['coefficients'][0][0]=.5;hand.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='changed after qualification'):binding.assert_bound(changes,identity,receipt)
    with pytest.raises(ValueError,match='Loaded calibration receipt'):binding.assert_bound(changes,binding.capture(changes),receipt)
    del changes['hand_response_calibration']
    with pytest.raises(ValueError,match='require a bound response'):binding.capture(changes)
