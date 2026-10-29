# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Development-only factorial on the frozen v1 packets. No native capture.

All eighty worlds/decisions have now informed design. Development comparisons
cannot qualify an actor or become a fresh confirmation by relabelling a split.
"""
import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import torch

from lem.model.causal_readout_v2 import (LocalCausalReadout, GROUPS, rotation_columns,
    layout_for, group_scales, grouped_objective)
from mimo_bridge.causal_packets import (MODALITIES, frame_from_payload, fixed_targets,
    packed_target, load_packet, tensor_digest)
from mimo_bridge.tools.causal_readout_study import (restore_encoder, validate_index,
    file_hash, metrics, admission)
from mimo_bridge.tools.training_continuation import atomic_json


SCHEMA = 'local_achieved_causal_readout_development_v2'
V1_REPORT_SHA = 'a4f9af8304274310683d5bb95219dbb4ee1aaeddfaae9dd11e3fa5b43756b73b'
FIT_WORLDS = list(range(93010,93016))
EVALUATION_WORLDS = list(range(93020,93024))


def sensed_achieved(frame):
    """Only the decoded achieved motor observation; no simulator body query."""
    from tokenizer.config import TokenizerConfig
    motor = frame.tokenized.motor_handles.detach().cpu()
    if motor.ndim != 2 or motor.shape[1] != 10 or not torch.isfinite(motor).all():
        raise ValueError('native achieved motor sense is required')
    position = motor[:,:3]*TokenizerConfig().position_range_cm/200.
    return torch.cat((position,rotation_columns(motor[:,3:7])), -1)


@torch.no_grad()
def encode_window(model, context):
    frames = [frame_from_payload(p) for p in context]
    raw, mask, _, _ = model._batch_frames(frames)
    if any(mask.get(k) != 'real' for k in MODALITIES):
        raise ValueError('each snapshot must encode its own real sensory window')
    registry = model.encoder_registry
    modes = {id(m):m.training for m in registry.modules()}
    try:
        registry.eval()
        from lem.model.spatial import encode_spatial
        global_tokens, regions, valid = encode_spatial(registry,{k:raw[k] for k in MODALITIES},mask)
        return dict(encoded=torch.cat([global_tokens[k].reshape(len(context),-1) for k in MODALITIES],-1).cpu(),
            regions=regions.detach().cpu(),region_valid=valid.detach().cpu().bool(),
            sensed=torch.from_numpy(np.stack([packed_target(fixed_targets(f)) for f in frames])),
            achieved=torch.stack([sensed_achieved(f) for f in frames]),
            site_position=torch.from_numpy(np.stack([np.asarray(f.touch_pos,np.float32)/2. for f in frames])))
    finally:
        for module in registry.modules(): module.training=modes[id(module)]


def immutable_study(study):
    study = Path(study)
    if file_hash(study/'report.json') != V1_REPORT_SHA:
        raise ValueError('this development declaration requires the exact retained negative v1 report')
    declaration = json.loads((study/'declaration.json').read_text())
    if (declaration['train_worlds'] != FIT_WORLDS or declaration['test_worlds'] != EVALUATION_WORLDS
        or declaration['states_per_world'] != 8):
        raise ValueError('historical packet matrix differs from the declared development source')
    index = json.loads((study/'packet_index.json').read_text())
    report = json.loads((study/'report.json').read_text())
    args = SimpleNamespace(out=study,train_worlds=FIT_WORLDS,test_worlds=EVALUATION_WORLDS,states=8)
    validate_index(args,index,report['source_checkpoint_sha256'])
    names = ('report.json','declaration.json','packet_index.json','frozen_encoder.pt',
             'action_conditioned.pt','action_blind.pt')
    hashes = {name:file_hash(study/name) for name in names}
    return index, report, hashes


def materialize(study, model, index, progress=None):
    rows = []
    for receipt in index:
        packet = load_packet(Path(study)/'packets'/receipt['path'])
        row = encode_window(model,packet['context'])
        times = np.array([p['sim_time'] for p in packet['context']])
        row.update(world=packet['world_id'],event=packet['decision'],phase=packet['phase'],
            split='development_fit' if packet['world_id'] in FIT_WORLDS else 'development_evaluation',
            history_dt=torch.tensor(np.r_[0,np.diff(times)],dtype=torch.float32),
            actions=torch.tensor(packet['commands'],dtype=torch.float32),
            duration=torch.tensor(packet['durations'],dtype=torch.float32),
            start=torch.tensor(packet['start']),target=torch.tensor(packet['targets']),
            layout=packet['target_layout'],packet_sha256=receipt['sha256'])
        handles = row['achieved'].shape[1]; sites=row['site_position'].shape[1]
        if row['layout'] != layout_for(handles,sites):
            raise ValueError('fixed target ordering differs from the local model contract')
        if row['achieved'].shape[0] != 6 or row['site_position'].shape != (6,sites,3):
            raise ValueError('current achieved and site positions must be present throughout history')
        rows.append(row)
        status=dict(stage='encoding_existing_development_packets',completed=len(rows),total=len(index))
        print(json.dumps(status),flush=True)
        if progress is not None:progress(status)
    return rows


def normalization(rows):
    layout=rows[0]['layout']; scale,contrast,evidence=group_scales(rows,layout)
    # Input normalization, also shared across sites/handles within each group.
    past=torch.cat([r['sensed'] for r in rows]); sensed_scale=torch.ones_like(scale)
    for lo,hi in layout.values():
        value=past[:,lo:hi].square().mean().sqrt()
        sensed_scale[lo:hi]=value if value>1e-6 else 1.
    encoded_scale=torch.cat([r['encoded'] for r in rows]).square().mean(0).sqrt().clamp_min(1e-6)
    valid_regions=torch.cat([r['regions'][r['region_valid']] for r in rows])
    region_scale=valid_regions.square().mean(0).sqrt().clamp_min(1e-6)
    return dict(scale=scale,contrast_scale=contrast,sensed_scale=sensed_scale,
                encoded_scale=encoded_scale,region_scale=region_scale,evidence=evidence)


def batch(rows,norm):
    inputs={}
    for key in ('encoded','regions','region_valid','sensed','achieved','site_position','history_dt'):
        inputs[key]=torch.cat([r[key][None].expand(4,*r[key].shape) for r in rows])
    inputs['encoded']=inputs['encoded']/norm['encoded_scale']
    inputs['regions']=inputs['regions']/norm['region_scale']
    inputs['sensed']=inputs['sensed']/norm['sensed_scale']
    inputs['actions']=torch.cat([r['actions'] for r in rows])
    inputs['durations']=torch.cat([r['duration'] for r in rows])
    target=torch.stack([(r['target']-r['start'])/norm['scale'] for r in rows])
    return inputs,target


@torch.no_grad()
def predictions(model,rows,norm,blind=False,batch_size=4):
    output=[]; model.eval()
    for start in range(0,len(rows),batch_size):
        selected=rows[start:start+batch_size];inputs,_=batch(selected,norm)
        pred=model(**inputs,blind=blind)*norm['scale']
        origin=torch.cat([r['start'][None].expand(4,-1) for r in selected])[:,None]
        output.append(pred+origin)
    return torch.cat(output)


def source_record():
    from lem.model import causal_readout_v2
    from mimo_bridge import causal_packets
    from mimo_bridge.tools import causal_readout_study
    paths=[Path(p).resolve() for p in (__file__,causal_readout_v2.__file__,
           causal_packets.__file__,causal_readout_study.__file__)]
    return {str(p):dict(sha256=file_hash(p),source=p.read_bytes()) for p in paths}


def save_stage(path,model,optimizer,architecture,norm,rows,source,stage,update,blind):
    if path.exists():raise FileExistsError('existing auxiliary stage evidence is retained')
    state=dict(schema=SCHEMA,architecture=architecture,model=model.state_dict(),
        optimizer=optimizer.state_dict(),normalization=norm,source=source,
        packet_ids=[r['packet_sha256'] for r in rows],stage=stage,update=update,
        blind=blind,production_qualification=False,intention_demonstrated=False)
    buffer=io.BytesIO();torch.save(state,buffer);payload=buffer.getvalue()
    current=sum(p.stat().st_size for p in path.parent.glob('*.pt'))
    if current+len(payload)>512*1024**2 or shutil.disk_usage(path.parent).free-len(payload)<2*1024**3:
        raise OSError('auxiliary checkpoint budget/free-space reserve reached; previous evidence retained')
    path.write_bytes(payload)
    loaded=torch.load(path,map_location='cpu',weights_only=False)
    if tensor_digest(state)!=tensor_digest(loaded):raise ValueError('saved model/optimizer state changed')
    copy=LocalCausalReadout(**loaded['architecture']);copy.load_state_dict(loaded['model']);copy.eval()
    inputs,_=batch(rows[:1],norm);model.eval()
    with torch.no_grad():
        if not torch.equal(model(**inputs,blind=blind),copy(**inputs,blind=blind)):
            raise ValueError('saved local readout does not reproduce the exact forecast')
    return file_hash(path)


def restore_readout(path):
    """Checked reconstruction: code, preprocessing, encoder and weights remain bound."""
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('schema')!=SCHEMA:raise ValueError('not a v2 development readout')
    if set(state['source']['code'])!=set(source_record()):
        raise ValueError('readout/preprocessing source manifest is incomplete or differs')
    for name,record in state['source']['code'].items():
        if hashlib.sha256(record['source']).hexdigest()!=record['sha256'] or file_hash(name)!=record['sha256']:
            raise ValueError('readout/preprocessing source changed; use the retained version explicitly')
    encoder_path=Path(state['source']['encoder_path'])
    if file_hash(encoder_path)!=state['source']['encoder_file_sha256']:
        raise ValueError('frozen encoder artifact changed')
    encoder=restore_encoder(encoder_path)
    head=LocalCausalReadout(**state['architecture']);head.load_state_dict(state['model']);head.eval()
    return head,encoder,state


def run_factorial(rows,args,source):
    fit=[r for r in rows if r['world'] in args.fit_worlds]
    evaluation=[r for r in rows if r['world'] in args.evaluation_worlds]
    expected={(w,e) for w in args.fit_worlds+args.evaluation_worlds for e in range(args.states)}
    actual=[(r['world'],r['event']) for r in rows]
    if not expected or set(actual)!=expected or len(actual)!=len(expected):
        raise ValueError('incomplete/duplicated development decision matrix')
    if set(args.fit_worlds)&set(args.evaluation_worlds):raise ValueError('development fit/evaluation overlap')
    if any(r['layout']!=rows[0]['layout'] for r in rows):raise ValueError('inconsistent target layout')
    early=[r for r in fit if r['world'] in args.fit_worlds[:len(args.fit_worlds)//2]]
    if not early or not evaluation:raise ValueError('development stages need disjoint nonempty worlds')
    # Stage one may not inspect even the later FIT outcomes for normalization.
    norm=normalization(early);layout=rows[0]['layout'];out=Path(args.out)
    architecture=dict(encoded_width=rows[0]['encoded'].shape[-1],region_width=rows[0]['regions'].shape[-1],
        handles=rows[0]['achieved'].shape[1],sites=rows[0]['site_position'].shape[1],hidden=args.hidden)
    all_results=[];phase_results=[];paired=[];artifacts={};loss_records=[]
    for achieved in (False,True):
        for contrast in (False,True):
            condition=f'achieved_{int(achieved)}_contrast_{int(contrast)}';comparison=[]
            for blind in (False,True):
                torch.manual_seed(args.seed)
                model=LocalCausalReadout(**architecture,achieved_input=achieved)
                optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                selection=torch.Generator().manual_seed(args.seed+100)
                for stage,data in enumerate((early,fit),1):
                    # Initial units remain fixed across stages: retained targets
                    # cannot change coordinates when the new fit worlds arrive.
                    for update in range(1,max(args.budgets)+1):
                        indices=torch.randperm(len(data),generator=selection)[:args.batch_decisions].tolist()
                        inputs,target=batch([data[i] for i in indices],norm)
                        optimizer.zero_grad(set_to_none=True);model.train()
                        pred=model(**inputs,blind=blind).reshape(len(indices),4,5,-1)
                        loss,parts=grouped_objective(pred,target,layout,norm['contrast_scale'],contrast)
                        if not torch.isfinite(loss):raise ValueError('nonfinite auxiliary fit; no further updates')
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                        optimizer.step()
                        if update not in args.budgets:continue
                        label=('action_blind' if blind else 'action_conditioned')+f'_stage_{stage}'
                        row=dict(condition=condition,blind=blind,stage=stage,update=update,
                                 outcome_loss=float(parts['outcome']),contrast_loss=float(parts['contrast']))
                        loss_records.append(row);print(json.dumps(dict(activity='development_fit',**row)),flush=True)
                        atomic_json(out/'status.json',dict(activity='development_fit',**row))
                        for name,panel in (('fit',fit),('evaluation',evaluation)):
                            p=predictions(model,panel,norm,blind,args.batch_decisions)
                            measurements=metrics(panel,p,label)
                            all_results.extend([dict(r,condition=condition,panel=name,budget=update) for r in measurements])
                            if update==max(args.budgets):
                                phase_results.extend([dict(r,condition=condition,panel=name,budget=update)
                                    for r in metrics(panel,p,label,by_phase=True)])
                            if name=='evaluation' and update==max(args.budgets):comparison.extend(measurements)
                        name=f'{condition}_blind_{int(blind)}_stage_{stage}_update_{update}.pt'
                        artifacts[name]=save_stage(out/name,model,optimizer,dict(architecture,achieved_input=achieved),
                            norm,rows,source,stage,update,blind)
            descriptive=admission(comparison,args.evaluation_worlds)
            descriptive['scope']='Already-seen development worlds; not confirmation or an acting license.'
            paired.append(dict(condition=condition,development_resolution=descriptive))
    return dict(schema=SCHEMA,verdict='development_only',conditions=paired,results=all_results,phase_results=phase_results,
        loss_records=loss_records,normalization_evidence=norm['evidence'],artifacts=artifacts,
        auxiliary_parameters=sum(p.numel() for p in model.parameters()),packet_count=len(rows),
        production_qualification=False,intention_demonstrated=False,old_brain_trained=False,
        live_actions_changed=False,
        limits='Only achieved-input and contrast-loss factors are isolated within v2. Shared architecture differs from v1; development scores do not identify its individual causal contribution. One initialization seed; no new native worlds. Meaningful choice and general skill remain untested.')


def declaration(args):
    return dict(schema=SCHEMA,study=str(args.study.resolve()),v1_report_sha256=V1_REPORT_SHA,
        method_source_sha256={name:record['sha256'] for name,record in source_record().items()},
        fit_worlds=FIT_WORLDS,development_evaluation_worlds=EVALUATION_WORLDS,states_per_world=8,
        all_packets_previously_seen=True,factorial=dict(achieved_input=[False,True],contrast_objective=[False,True]),
        separately_fitted_blind_per_condition=True,budgets=args.budgets,stages=2,seed=args.seed,
        hidden=args.hidden,batch_decisions=args.batch_decisions,cpu_threads=1,
        maximum_updates=8*2*max(args.budgets),maximum_packets=80,maximum_new_artifact_mib=512,
        inputs='Past native achieved motor world poses, fixed proprioception, experienced contact moments/root site positions, own frozen global tokens and spatial regions; future precommitted body orders only.',
        loss='Equal five-group outcome loss plus, when enabled, candidate-centered loss divided by each group measured action-contrast RMS. Units fixed from stage-one fit worlds; no per-site output mask.',
        controls='Full world/group matrices, trained action-blind, wrong-body/same-gaze, persistence, constant motion and retained-panel errors. Original v1 targets/negative evidence remain exact.',
        resolution='Descriptive development screen uses v1 half-contrast RMS, all absolute-error comparators and 5% contrast retention allowance. No numerical threshold grants a license.',
        stage_protocol='First three fit worlds, then all six with rehearsal. Save model AND optimizer at every budget in both stages. No score-based stopping or seed choice.',
        fresh_confirmation='Requires a new signed-off declaration before new world capture; disjoint world seeds and raw packet identities, fixed chosen architecture/budget and no tuning from confirmation scores.',
        production_qualification=False,intention_demonstrated=False,native_capture=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,default=Path('Saved/mimo_bridge/causal_readout_study_20260913b'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--budgets',type=int,nargs='+',default=[40,120,240])
    p.add_argument('--seed',type=int,default=619);p.add_argument('--hidden',type=int,default=48)
    p.add_argument('--batch-decisions',type=int,default=4);p.add_argument('--declare-only',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    if (args.budgets!=sorted(set(args.budgets)) or not args.budgets or min(args.budgets)<1
        or max(args.budgets)>240 or args.hidden not in (24,48) or not 1<=args.batch_decisions<=4):
        raise ValueError('development bounds require unique ascending budgets <=240, hidden24/48, batch1..4')
    args.fit_worlds=FIT_WORLDS;args.evaluation_worlds=EVALUATION_WORLDS;args.states=8
    if args.out.resolve()==args.study.resolve() or args.out.resolve() in args.study.resolve().parents:
        raise ValueError('v1 study and its parents cannot be used as v2 output')
    args.out.mkdir(parents=True,exist_ok=True);decl=declaration(args)
    path=args.out/'declaration.json'
    if path.exists():
        if json.loads(path.read_text())!=decl:raise ValueError('existing development declaration differs')
    else:atomic_json(path,decl)
    if args.declare_only:return
    if any(args.out.glob('*.pt')) or any((args.out/f).exists() for f in ('report.json','failure.json','started.json')):
        raise FileExistsError('completed or partial v2 evidence retained; choose a fresh output directory')
    atomic_json(args.out/'started.json',dict(declaration_sha256=file_hash(path)))
    try:
        index,original,hashes=immutable_study(args.study)
        # Standalone encoder contains exact old weights but no old optimizer;
        # nothing from the old learner is trainable or imported into this head.
        frozen=args.study/'frozen_encoder.pt';model=restore_encoder(frozen)
        before=tensor_digest(model.encoder_registry.state_dict())
        rows=materialize(args.study,model,index,lambda row:atomic_json(args.out/'status.json',row))
        source=dict(code=source_record(),encoder_path=str(frozen.resolve()),encoder_file_sha256=file_hash(frozen),
                    packet_root=str((args.study/'packets').resolve()),v1_artifact_hashes=hashes,
                    original_model_sha256=original['source_checkpoint_sha256'])
        report=run_factorial(rows,args,source)
        if tensor_digest(model.encoder_registry.state_dict())!=before:
            raise ValueError('frozen encoder changed during auxiliary learning')
        for name,digest in hashes.items():
            if file_hash(args.study/name)!=digest:raise ValueError('frozen v1 evidence changed')
        for name,record in source['code'].items():
            if file_hash(name)!=record['sha256']:raise ValueError('auxiliary source changed during the declared fit')
        if sum(x.stat().st_size for x in args.out.glob('*.pt'))>512*1024**2:
            raise ValueError('declared auxiliary artifact budget exceeded; artifacts retained')
        report.update(frozen_encoder_exact=True,v1_artifacts_exact=True,v1_artifact_hashes=hashes,
                      old_optimizer_loaded=False,source_checkpoint_sha256=original['source_checkpoint_sha256'])
        atomic_json(args.out/'report.json',report)
        atomic_json(args.out/'status.json',dict(activity='complete_development_only',report='report.json'))
    except Exception as exc:
        atomic_json(args.out/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        atomic_json(args.out/'status.json',dict(activity='failed',error=str(exc)))
        raise


if __name__=='__main__':main()
