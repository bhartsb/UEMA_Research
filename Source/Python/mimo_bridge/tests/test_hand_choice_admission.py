# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import copy
import importlib
import json
from pathlib import Path

import pytest


def fixture(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4]/'docs/gain_of_function_v3'))
    admission=importlib.import_module('hand_choice_admission')
    calibration=tmp_path/'calibration.json';calibration.write_text(json.dumps({'sha256':'a'*64}))
    script=tmp_path/'compare.py';script.write_text('print("measured")')
    checkpoint=tmp_path/'checkpoint'
    proof=dict(runtime_sha256='b'*64,source_model_sha256='c'*64,
        calibration_file_sha256=admission.sha256(calibration),calibration_payload_sha256='a'*64,
        checkpoint=str(checkpoint),held_out_bodies=[4,5,6,7],completed_trials=8,exact_replay=True,
        weights_unchanged=True,nonfinger_commands_unchanged=True,sensory_goals_identical=True,
        learning=False,passed=True,tested_route=admission.ROUTE,script_path=str(script),
        script_sha256=admission.sha256(script),paired95_difference=[-.3,-.1])
    path=tmp_path/'result.json';path.write_text(json.dumps(proof))
    args=(checkpoint,'b'*64,'c'*64,calibration)
    return admission,path,proof,args,script


def test_actual_hand_comparison_is_bound_to_source_code_calibration_and_receipt(monkeypatch,tmp_path):
    admission,path,proof,args,script=fixture(monkeypatch,tmp_path)
    binding=admission.bind(path,*args)
    assert admission.verify_bound(binding,*args)
    changed=copy.deepcopy(proof);changed['paired95_difference']=[-.4,-.2]
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='changed after qualification'):admission.verify_bound(binding,*args)
    path.write_text(json.dumps(proof));script.write_text('print("changed")')
    with pytest.raises(ValueError,match='script changed'):admission.bind(path,*args)


@pytest.mark.parametrize('key,value',[
    ('completed_trials',7),('held_out_bodies',[0,1,2,3]),('exact_replay',False),
    ('weights_unchanged',False),('nonfinger_commands_unchanged',False),('sensory_goals_identical',False),
    ('runtime_sha256','d'*64),('source_model_sha256','d'*64),('calibration_file_sha256','d'*64),
    ('passed',False),('tested_route','private surrogate only'),('paired95_difference',[-.2,.01]),
    ('paired95_difference',[-.2,float('nan')]),
])
def test_partial_mismatched_or_unsuccessful_hand_comparison_cannot_qualify(monkeypatch,tmp_path,key,value):
    admission,path,proof,args,_=fixture(monkeypatch,tmp_path)
    proof[key]=value;path.write_text(json.dumps(proof))
    with pytest.raises(ValueError):admission.bind(path,*args)
