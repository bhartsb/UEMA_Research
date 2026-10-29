# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Proof-only hashes locate differences without changing learned execution."""
from copy import deepcopy
import json
import random
from types import SimpleNamespace

import numpy as np
import torch

from mimo_bridge.tools import prove_caregiver_fit_continuation as proof
from tokenizer import config


def frame():
    return SimpleNamespace(frame_bytes=bytes(65536),tokenized=None,
        vision_patches=np.arange(12,dtype=np.float32).reshape(1,12),
        vision_angles=np.zeros((1,4),np.float32),meta={'task':{'visible':True}},sim_time=.5)


def test_input_hash_excludes_only_host_wall_clock_and_detects_actual_pixel_metadata_changes():
    a=frame();b=deepcopy(a);raw=bytearray(b.frame_bytes)
    offset=config.HEADER_OFFSET+config.HDR_WALL_CLOCK;raw[offset:offset+8]=b'12345678';b.frame_bytes=bytes(raw)
    assert proof.frame_evidence([a])==proof.frame_evidence([b])
    b.vision_patches[0,3]+=1
    assert proof.frame_evidence([a])[0]['vision_patches']!=proof.frame_evidence([b])[0]['vision_patches']
    b=deepcopy(a);b.meta['task']['visible']=False
    assert proof.frame_evidence([a])[0]['meta']!=proof.frame_evidence([b])[0]['meta']


def test_update_hash_does_not_consume_rng_and_detects_weight_and_adam_changes():
    torch.manual_seed(1);np.random.seed(2);random.seed(3)
    module=torch.nn.Linear(1,1);optimizer=torch.optim.Adam(module.parameters(),lr=.1)
    model=SimpleNamespace(encoder_registry=module,forward_model=None,prediction_heads=None,
        motor_controller=None,wellness=None,gaze_head=None,voice_head=None,_side_modules=lambda:{},
        optimizer=optimizer,device=torch.device('cpu'),noise_rng=torch.Generator().manual_seed(4),step_count=5)
    a=proof.update_evidence(model,[frame()]);b=proof.update_evidence(model,[frame()])
    assert a==b
    module(torch.ones(1,1)).sum().backward();optimizer.step()
    c=proof.update_evidence(model,[frame()])
    assert a['rng']==c['rng']
    assert a['inputs']==c['inputs']
    assert a['learned']!=c['learned'] and a['adam']!=c['adam']


def test_prefix_requires_matching_inputs_and_first_update_adam_before_resume_claim(tmp_path):
    for arm in ('uninterrupted','interrupted'):
        folder=tmp_path/arm/'update_trace';folder.mkdir(parents=True)
        for step in range(2):
            (folder/f'update_{step}.json').write_text(json.dumps({'step':step,'before':{'inputs':'same'},'after':{'adam':'same'}}))
    assert proof.prefix_verdict(tmp_path,2)['exact']
    path=tmp_path/'interrupted/update_trace/update_0.json';record=json.loads(path.read_text())
    record['after']['adam']='different';path.write_text(json.dumps(record))
    result=proof.prefix_verdict(tmp_path,2)
    assert result['valid'] and not result['exact']
    assert result['updates'][0]['difference']=='update.after.adam'
