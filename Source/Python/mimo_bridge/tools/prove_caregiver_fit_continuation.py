# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Bounded native copied-learning interruption proof; never a skill verdict."""
import argparse
from copy import deepcopy
import gc
import gzip
import json
from pathlib import Path
import random
import time
from contextlib import contextmanager
from types import SimpleNamespace

import psutil
import torch
from safetensors import safe_open

from mimo_bridge import continuation_state
from mimo_bridge.event_tape import EventTape
from .caregiver_learning_trial import fit,settings,write
from .caregiver_fit_workers import CopyResourcesUnavailable,resources,admission,guard_for
from .causal_test import _limits_for_run
from .prepare_local_shadow import state_digest


def frame_evidence(frames):
    """Hash actual learner inputs, excluding only the unsensed wall-clock header."""
    from tokenizer import config as contract
    result=[]
    for frame in frames:
        raw=bytearray(frame.frame_bytes);offset=contract.HEADER_OFFSET+contract.HDR_WALL_CLOCK
        raw[offset:offset+8]=b'\0'*8
        values={k:v for k,v in vars(frame).items() if k not in ('frame_bytes','tokenized')}
        values['frame_bytes']=bytes(raw)
        result.append({k:state_digest(v) for k,v in values.items()})
    return result


def update_evidence(model,frames):
    import numpy as np
    from .migrate_trunk import modules
    return dict(inputs=frame_evidence(frames),
        learned={name:state_digest(module.state_dict()) for name,module in modules(model).items()},
        adam=state_digest(model.optimizer.state_dict()),
        rng={name:state_digest(value) for name,value in dict(torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if model.device.type=='cuda' else None,
            numpy=np.random.get_state(),python=random.getstate(),noise=model.noise_rng.get_state()).items()},
        step_count=model.step_count,
        previous_prediction={name:state_digest(getattr(model,name,None)) for name in
            ('_nf_raw','_nf_ctx','_nf_pred','_nf_tokens','_nf_mask','last_tokens','last_gate','_prev_mfa')})


@contextmanager
def trace_updates(out,arm):
    """Proof-local instrumentation; no production or study behavior is patched."""
    from mimo_bridge.lem_model import LemLite
    original=LemLite.step;folder=out/arm/'update_trace';folder.mkdir(parents=True,exist_ok=True);seen=0
    def traced(model,frames,*args,**kwargs):
        nonlocal seen
        step=int(kwargs['step_index']);path=folder/f'update_{step}.json'
        if path.exists():raise ValueError('Refusing to overwrite a native update trace')
        before=update_evidence(model,frames)
        # The first two updates retain actual input pixels and the learned vision
        # parameters/Adam, so hash differences can be quantified without another
        # physical run. Later steps retain hashes, commands and counter evidence.
        retain=seen<2;seen+=1
        record=dict(step=step,before=before,phase='entered')
        if retain:
            with gzip.open(folder/f'input_{step}.pt.gz','wb',compresslevel=1) as stream:
                torch.save(frames,stream)
        write(path,record)
        result=original(model,frames,*args,**kwargs)
        record.update(after=update_evidence(model,frames),phase='completed',
            observed_tokens={k:state_digest(v) for k,v in model.last_tokens.items()},
            loss=float(result.loss),gaze=state_digest(result.gaze_out),
            command=state_digest(result.motor_for_action),
            actual_counter=int(model.step_count))
        if retain:
            vision=model.encoder_registry.encoders['vision']
            preserved=dict(weights={k:v.detach().cpu() for k,v in vision.state_dict().items()},
                adam={k:{n:(v.detach().cpu() if torch.is_tensor(v) else v) for n,v in model.optimizer.state.get(p,{}).items()}
                    for k,p in vision.named_parameters()},raw_tokens={k:v.detach().cpu() for k,v in model.last_tokens.items()})
            with gzip.open(folder/f'vision_after_{step}.pt.gz','wb',compresslevel=1) as stream:
                torch.save(preserved,stream)
        write(path,record)
        return result
    LemLite.step=traced
    try:yield
    finally:LemLite.step=original


def prefix_verdict(out,cut):
    left=sorted((out/'uninterrupted'/'update_trace').glob('update_*.json'))[:cut]
    right=sorted((out/'interrupted'/'update_trace').glob('update_*.json'))[:cut]
    if len(left)!=cut or len(right)!=cut:return dict(valid=False,reason='The older proof did not capture a complete prefix trace')
    differences=[]
    for a,b in zip(left,right):
        x=json.loads(a.read_text());y=json.loads(b.read_text())
        differences.append(dict(step=x['step'],difference=first_difference(x,y,'update')))
    return dict(valid=True,exact=all(row['difference'] is None for row in differences),updates=differences)


def experiential(value):
    # Writers intentionally fork when a saved episode resumes. Compare every
    # recorded event in order, rather than its operational UUID/chunk filenames.
    if isinstance(value,EventTape):return list(value)
    if isinstance(value,dict):return {key:experiential(item) for key,item in value.items()}
    if isinstance(value,list):return [experiential(item) for item in value]
    if isinstance(value,tuple):return tuple(experiential(item) for item in value)
    return value


def first_difference(a,b,path='state'):
    if state_digest(a)==state_digest(b):return None
    if isinstance(a,dict) and isinstance(b,dict) and a.keys()==b.keys():
        for key in a:
            result=first_difference(a[key],b[key],path+'.'+str(key))
            if result:return result
    if isinstance(a,(tuple,list)) and type(a)==type(b) and len(a)==len(b):
        for index,(x,y) in enumerate(zip(a,b)):
            result=first_difference(x,y,path+f'[{index}]')
            if result:return result
    return path


def scientific_state(checkpoint):
    episode=continuation_state.load(checkpoint);trainer=episode['trainer']
    world=episode['worlds'][0]
    payload=dict(world['payload'])
    from tokenizer import config as frame_contract
    frame=bytearray(payload['frame_bytes']);offset=frame_contract.HEADER_OFFSET+frame_contract.HDR_WALL_CLOCK
    frame[offset:offset+8]=b'\0'*8;payload['frame_bytes']=bytes(frame)
    return dict(learner=episode['learner'],execution=trainer['fit_execution'],
        physical_world=experiential(world['world']),payload=experiential(payload),echo=world['echo'],
        orders={key:trainer[key] for key in ('step','trial_step','held','executed','order','bout','until','bout_rng','frame_meta')},
        speech={key:trainer['fit_resume'][key] for key in ('utterances','last_utterance','initial_encoder_sha256')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--speech',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--verify-existing',action='store_true',help='Compare completed native endpoints on CPU; never allocate a learner or renderer')
    parser.add_argument('--continue-existing',action='store_true',help='Keep a completed diagnostic arm and execute only the missing arm after resource admission')
    args=parser.parse_args()
    if args.verify_existing and args.continue_existing:parser.error('choose read-only verification or missing-arm continuation')
    args.out.mkdir(parents=True,exist_ok=args.verify_existing or args.continue_existing)
    torch.set_num_threads(1)
    if args.device.startswith('cuda') and not args.verify_existing:torch.cuda.set_per_process_memory_fraction(.25,device=args.device)
    # Cross the source's five-update held-order cadence after the interruption,
    # so both the retained command and the first newly authored order are tested.
    seed=24031;steps=6;cut=2
    limits,origin=_limits_for_run(str(args.checkpoint),SimpleNamespace(dev_limits=lambda _:{}))
    parent_hash=continuation_state.file_sha256(args.checkpoint/'model.safetensors')
    declaration=dict(instrument_smoke=True,parent=str(args.checkpoint),parent_sha256=parent_hash,
        device=args.device,steps=steps,interrupt_after=cut,seed=seed,limits=limits,limits_source=origin,
        pass_rule='All native learned tensors, Adam histories, runtime, causal execution, body state, orders and actual speech match exactly after uninterrupted versus paused/resumed learning.',
        exclusions='Wall elapsed time, recording writer UUID/chunk paths and fit output directory identify operations, not sensed experience.',
        scope='Native continuation proof only. No caregiver benefit or acquired skill claim.')
    if args.verify_existing or args.continue_existing:
        if json.loads((args.out/'declaration.json').read_text())!=declaration:
            raise ValueError('Existing native proof declaration differs; do not reinterpret another trial')
    else:write(args.out/'declaration.json',declaration)
    if not args.verify_existing:
        from .caregiver_fit_continuation import execution_source_digest
        versions=args.out/'process_versions';versions.mkdir(exist_ok=True)
        write(versions/(str(time.time_ns())+'.json'),dict(timestamp=time.time(),
            continue_existing=args.continue_existing,proof_sha256=continuation_state.file_sha256(Path(__file__)),
            execution_source_sha256=execution_source_digest(),
            checkpoint_writer_sha256=continuation_state.file_sha256(Path(__file__).parents[2]/'lem/training/checkpoint.py'),
            torch_version=torch.__version__,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            scope='Operational source/precision receipt; input/model/Adam/RNG equality still decides reproducibility.'))
    paths={};started=time.time()
    if args.verify_existing:
        for name in ('uninterrupted','interrupted'):
            paths[name]=Path(json.loads((args.out/name/'fit.json').read_text())['checkpoint'])
        paths['partial']=Path(json.loads((args.out/'interrupted'/'paused_fit.json').read_text())['checkpoint'])
    elif args.continue_existing:
        for name in ('uninterrupted','interrupted'):
            receipt=args.out/name/'fit.json'
            if receipt.exists():
                completed=json.loads(receipt.read_text())
                if completed['online_updates']!=steps:raise ValueError('Existing native diagnostic has a different budget')
                paths[name]=Path(completed['checkpoint'])
        partial=args.out/'interrupted'/'paused_fit.json'
        if partial.exists():paths['partial']=Path(json.loads(partial.read_text())['checkpoint'])
    owner=dict(pid=psutil.Process().pid,process_created=psutil.Process().create_time())
    try:
        for name in (() if args.verify_existing else ('uninterrupted','interrupted')):
            if name in paths:continue
            folder=args.out/name;a=settings(args.checkpoint,folder,args.speech,args.device,seed);random.seed(seed)
            allowed,reason=admission(resources(args.device),[])
            if not allowed:raise RuntimeError('Native proof resource admission: '+reason)
            physical_guard=guard_for(dict(out=str(folder),coordinator=owner,device=args.device,
                gpu_reserve_mib=1024,ram_reserve_gib=4.,check_every_updates=1))
            if name=='uninterrupted':
                with trace_updates(args.out,name):
                    paths[name]=fit(a,args.checkpoint,folder,seed,steps,limits,step_guard=physical_guard)
            else:
                def stop(model,index):
                    if index==cut:raise CopyResourcesUnavailable('Declared native continuity interruption')
                    physical_guard(model,index)
                stop.before_save=physical_guard.before_save
                if 'partial' not in paths:
                    try:
                        with trace_updates(args.out,name):fit(a,args.checkpoint,folder,seed,steps,limits,step_guard=stop)
                    except CopyResourcesUnavailable:pass
                partial=Path(json.loads((folder/'paused_fit.json').read_text())['checkpoint'])
                if json.loads((folder/'paused_fit.json').read_text())['completed_updates']!=cut:
                    raise ValueError('Native proof paused before its declared interruption')
                paths['partial']=partial
                before={p.name:continuation_state.file_sha256(p) for p in partial.iterdir() if p.is_file()}
                gc.collect()
                if args.device.startswith('cuda'):torch.cuda.empty_cache()
                with trace_updates(args.out,name):
                    paths[name]=fit(a,args.checkpoint,folder,seed,steps,limits,resume_checkpoint=partial,step_guard=physical_guard)
                assert before=={p.name:continuation_state.file_sha256(p) for p in partial.iterdir() if p.is_file()}
            gc.collect()
            if args.device.startswith('cuda'):torch.cuda.empty_cache()
        left,right=paths['uninterrupted'],paths['interrupted'];checks={}
        with safe_open(str(left/'model.safetensors'),framework='pt',device='cpu') as x,safe_open(str(right/'model.safetensors'),framework='pt',device='cpu') as y:
            checks['all_learned_tensors']=set(x.keys())==set(y.keys()) and all(torch.equal(x.get_tensor(k),y.get_tensor(k)) for k in x.keys())
        for filename in ('optimizer.pt','runtime_state.pt'):
            x=torch.load(left/filename,map_location='cpu',weights_only=False);y=torch.load(right/filename,map_location='cpu',weights_only=False)
            checks[filename]=state_digest(x)==state_digest(y);del x,y;gc.collect()
        a=scientific_state(left);b=scientific_state(right)
        differences={key:first_difference(a[key],b[key],key) for key in a}
        checks.update({key:value is None for key,value in differences.items()})
        prefix=prefix_verdict(args.out,cut)
        checks['independent_initial_prefix_exact']=prefix.get('valid') and prefix.get('exact')
        checks['parent_unchanged']=continuation_state.file_sha256(args.checkpoint/'model.safetensors')==parent_hash
        for name in ('uninterrupted','interrupted'):
            receipt=json.loads((args.out/name/'fit.json').read_text());checks[name+'_budget_exact']=receipt['online_updates']==steps
        passed=all(checks.values())
        write(args.out/'report.json',dict(instrument_smoke=True,continuation_pass=passed,capability_pass=False,
            checks=checks,first_differences=differences,checkpoints={k:str(v) for k,v in paths.items()},wall_seconds=time.time()-started,
            verification_only=args.verify_existing,verifier_sha256=continuation_state.file_sha256(Path(__file__)),
            state_digest_source_sha256=continuation_state.file_sha256(Path(__file__).with_name('prepare_local_shadow.py'))))
        write(args.out/'prefix_report.json',prefix)
        if not passed:raise ValueError('Native caregiver fit continuation differed: '+str({k:v for k,v in differences.items() if v}))
    except Exception as error:
        write(args.out/'failure.json',dict(error=str(error),timestamp=time.time()));raise


if __name__=='__main__':main()
