# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('qualification_liveness',
    Path(__file__).resolve().parents[4]/'docs/gain_of_function_v3/qualification_liveness.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def readings():
    rows=[]
    for end in range(5, 106, 5):
        count = sum(s % 7 == 0 for s in range(end-5, end))
        rows.append(dict(type='mimo',step=end,liveness=dict(sample_every=7,window_steps=5,
            sampled_steps=count,samples=count*8,layer_line=.5,below_line=[],
            layers={'common':.6,'occasional':.7} if count else {},
            watching={'common':{'ready':True},'occasional':{'ready':True}},revival={})))
    return rows


def test_unsampled_short_reports_pass_only_with_exact_cadence_and_full_coverage():
    result=module.assess(readings(),0,105)
    assert result['passed'] and result['explicitly_unsampled_reports']
    assert result['final_interval_layers']==['common','occasional']


def test_missing_scheduled_readings_or_body_samples_fail():
    for field,value in [('layers',{}),('samples',7),('sampled_steps',0)]:
        rows=readings()
        next(row for row in reversed(rows) if row['liveness']['sampled_steps'])['liveness'][field]=value
        assert not module.assess(rows,0,105)['passed']


def test_a_layer_missing_from_final_interval_is_not_silently_omitted():
    rows=readings()
    for row in rows[1:]:row['liveness']['layers'].pop('occasional',None)
    result=module.assess(rows,0,105)
    assert not result['passed'] and result['missing_layers']==['occasional']
    assert not module.assess(readings(),0,105,expected_layers=['never_observed'])['passed']


def test_actual_low_response_nonfinite_or_changed_line_fail_even_without_below_list():
    for value in (.4999,float('nan'),None):
        rows=readings()
        next(row for row in reversed(rows) if row['liveness']['sampled_steps'])['liveness']['layers']['common']=value
        assert not module.assess(rows,0,105)['passed']
    rows=readings();rows[-1]['liveness']['layer_line']=.4
    assert not module.assess(rows,0,105)['passed']


def test_omitted_report_or_unfinished_repair_window_fails():
    rows=readings();rows.pop(4)
    assert not module.assess(rows,0,105)['passed']
    rows=readings();rows[-1]['liveness']['watching']['common']['ready']=False
    assert not module.assess(rows,0,105)['passed']
    assert not module.assess([],0,105)['passed']
