# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Actual self-network learning from recoverable physical contact on model copies.

The operator's live trainer is never controlled here. Contact comes from a visible
sphere and MuJoCo, never injected function targets. Complete physical restores
separate actions. Held-out arrangements are constructed only after fitting.
"""
import argparse
from copy import deepcopy
from collections import deque
import json
import random
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from mimo_bridge.train_mimo import parse_args,build_learner,env_kwargs_for
from mimo_bridge.datasource import SingleEnvSource
from mimo_bridge.lem_model import LemLite,command_for_encoder
from mimo_bridge.tools.causal_test import _limits_for_run
from mimo_bridge.tools.migrate_trunk import modules
from mimo_bridge.tools.caregiver_learning_trial import paired_result
from mimo_bridge.tools.training_continuation import atomic_json


def status(out,stage,**kw):
    row=dict(stage=stage,timestamp=time.time(),**kw)
    atomic_json(out/'status.json',row);print(json.dumps(row),flush=True)
    from html import escape
    title={'measuring_recovery':'Checking whether movement can restore functioning',
           'collecting_function_changes':'Recording movements and changes in functioning',
           'fitting_self_network':'Learning movement consequences in a copy',
           'held_out_recovery':'Testing choices in unfamiliar arrangements',
           'testing_function_predictions':'Testing forecasts in unfamiliar arrangements',
           'complete':'Recovery comparison finished','failed':'Recovery comparison stopped'}[stage]
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
    if stage not in ('complete','failed'):page+='<meta http-equiv="refresh" content="30">'
    page+='<title>Can the baby learn to restore its functioning?</title><style>body{font:18px/1.6 system-ui;max-width:850px;margin:5vh auto;padding:24px;background:#f5f3ed;color:#18333b}h1{line-height:1.15}pre{white-space:pre-wrap}</style>'
    page+='<h1>'+title+'</h1><p>Live learning continues separately. This study first checks whether movement helps the simulated body recover its sensing or force capacity. It fits a copy only when that opportunity is measured.</p>'
    page+='<p>Recovery that happens without movement cannot establish intention. The study compares identical starting bodies, repeated commands and removal of the pressure source.</p>'
    page+='<pre>'+escape(json.dumps(kw,indent=2))+'</pre><p><a href="declaration.json">Declared comparison</a> · <a href="status.json">Current record</a>'
    for name,label in [('reachability.json','Physical measurements'),('report.json','Result'),('failure.json','Recorded error')]:
        if (out/name).exists():page+=' · <a href="'+name+'">'+label+'</a>'
    page+='</p><p>This study cannot authorize control of the live body. Closed-loop correction, completion and retention remain untested.</p></html>'
    pending=out/'index.tmp';pending.write_text(page,encoding='utf-8');pending.replace(out/'index.html')


def comparison_result(values):
    result=paired_result(values)
    result['scale']='Reduction in worst-site function deficit: -1=full loss added throughout, 0=no advantage, +1=full loss prevented throughout.'
    return result


def check_storage(out):
    if shutil.disk_usage(out).free < 4*1024**3:
        raise OSError('Recovery study stopped with less than 4 GiB free; existing records retained')


NUMERICAL_FLOOR=float(8*np.finfo(np.float32).eps)


def recovery_opportunity(row):
    """Require ongoing source effect and improvement through a different action."""
    return (row['source_skin_sensitivity']>NUMERICAL_FLOOR
            and row['available_skin_gain_over_hold']>NUMERICAL_FLOOR
            and row['available_gain_over_hold']>NUMERICAL_FLOOR)


def action_prediction_result(actual,predicted,valid):
    """Measure action differences; a correct constant forecast earns no gain."""
    x=np.asarray(actual,dtype=np.float64)[...,np.asarray(valid),0]
    p=np.asarray(predicted,dtype=np.float64)[...,np.asarray(valid),0]
    if x.shape!=p.shape or not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ValueError('invalid prediction comparison')
    target=x-x.mean(0,keepdims=True);prediction=p-p.mean(0,keepdims=True)
    baseline=float(np.mean(target**2));error=float(np.mean((prediction-target)**2))
    gain=None if baseline<=NUMERICAL_FLOOR**2 else 1-error/baseline
    return dict(valid=gain is not None,action_blind_error=baseline,prediction_error=error,
                action_error_removed=gain,scale='0=no action-effect error removed; 1=perfect action differences; negative=worse than ignoring the action.')


def configuration(checkpoint,out):
    # the run's args.json wherever the checkpoint now lives: beside its run folder while it is on C:,
    # else through lineage.json's "out" (2026-09-26: the archive on D: holds checkpoints only, so the
    # beside-path read failed for every archived checkpoint, DESIGN D3's baseline 344,400 among them)
    from mimo_bridge.storage_locations import resolve_run_args
    a=parse_args(['--out',str(out)])
    vars(a).update(json.loads(resolve_run_args(Path(checkpoint)).read_text()))
    a.device='cpu';a.out=str(out);a.n_envs=1;a.self_choice='off'
    return a


def hold(frame,strength):
    cmd=np.zeros((1,len(frame.bones),9),np.float32)
    cmd[0,:,:3]=frame.bones[:,4:7];cmd[0,:,3:7]=frame.bones[:,:4];cmd[...,8]=strength
    return cmd


def targets(frame):
    record=frame.meta['body_function'];q=np.asarray(record['values'],np.float32)
    cap=frame.meta['motor_function'];tail=np.zeros((10,3),np.float32);tail[:,0]=cap['values']
    return torch.from_numpy(np.concatenate([q,tail])),torch.from_numpy(np.r_[np.ones(len(q),bool),cap['valid']])


def recovery_cost(trace,valid):
    """Seconds-integrated loss of actual available function, averaged over time.

    0: every measured site fully functions; 1: at least one site fully unavailable
    throughout. This is external physical measurement, not a supplied reward.
    """
    x=np.asarray(trace)
    mask=np.asarray(valid)
    if (x.ndim!=3 or x.shape[0]==0 or x.shape[2]!=3 or mask.dtype!=np.bool_
        or mask.shape!=(x.shape[1],) or not mask.any() or not np.isfinite(x).all()
        or (x<0).any() or (x>1).any()):raise ValueError('invalid physical trajectory')
    return float((1-x[:,mask,0].min(-1)).mean())


def sequences(frame,proposal,dev,a,count,steps,seed):
    rng=np.random.default_rng(seed);ach=hold(frame,min(.3,float(dev['strength_limit'])))
    raw=np.repeat(ach[:,None],count,axis=1);raw[:,:, :, :3]/=200.
    raw[:,0]=np.asarray(proposal);raw[:,0,:,:3]/=200.
    # Same unlabelled exploration menu whatever body site the object contacts.
    for k in range(2,count):
        for indices in (range(0,6),range(6,12)):
            shift=rng.normal(size=3);shift=shift/max(np.linalg.norm(shift),1e-8)*.04
            raw[0,k,list(indices),:3]+=shift
        q=raw[0,k,:,3:7]+rng.normal(0,.08,(raw.shape[2],4))
        raw[0,k,:,3:7]=q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-8)
    hs=torch.from_numpy(ach.copy());hs[...,:3]/=200.
    previous=torch.from_numpy(np.repeat(ach,count,axis=0));raw=torch.from_numpy(raw[0])
    rows=[]
    for i in range(steps):
        if i%a.motor_hold==0:
            previous=LemLite.planned_execution(raw,dev,hs.expand(count,-1,-1),previous=previous,smoothing=a.smooth_actions)
            # The isolated eye servo receives zero azimuth/elevation. Forecast
            # that executed order, not a discarded candidate eye orientation.
            previous=torch.from_numpy(LemLite.gaze_to_handle(previous.numpy(),np.zeros((count,2))))
        rows.append(previous.detach().clone())
    return torch.stack(rows,1)


def episode(model,a,dev,out,seed,count,steps,warm,checkpoint,baseline=None):
    check_storage(out)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    # No learned runtime tracker or random state may leak from another test world.
    loaded=model.load_checkpoint(str(checkpoint),load_optimizer=False)
    if loaded.get('reinitialized_components')!=[]:raise ValueError('incomplete trial checkpoint')
    kw=env_kwargs_for(a,dev['phase_name'],str(out))
    kw.update(seed=seed,body_function='modulate',caregiver_mode='off',caregiver_seconds=0.,
              caregiver_record_dir='',critters=False,mobile=False,gaze_reflex=False,spontaneous_saccades=False)
    src=SingleEnvSource(**kw);frames=src.reset();e=src.env;mid=e._target_id
    # This copied fixture fixes the existing sphere in space. Its geometry stays
    # visible and colliding; disabling its random-walk owner introduces no sense.
    e._target_id=None;e.data.mocap_pos[mid]=[0,0,2]
    model.reset_transition_state();dt=e.dt*a.action_repeat
    command=hold(frames[0],min(.3,float(dev['strength_limit'])));executed=None
    tick=0;result=None
    baseline_percept=None;baseline_relations=deque(maxlen=model.self_history.capacity)
    def observe_baseline():
        nonlocal baseline_percept
        if baseline is None:return
        raw=dict(model.self_history.inputs)
        raw['previous']=None if baseline_percept is None else baseline_percept['state']
        raw['relations']=None if not baseline_relations else torch.stack(list(baseline_relations),1)
        with torch.no_grad():baseline_percept=baseline.perceive(**raw)
        baseline_relations.append(baseline_percept['belief'])
    def observe_and_hold(point=None):
        nonlocal frames,executed,tick,result
        result=model.step(frames,dict(dev,exploration_noise=0.),train=False,step_index=model.step_count,
                          executed_action=executed,executed_order_id=tick+1)
        observe_baseline()
        gaze=[]
        for sub in range(a.action_repeat):
            if point is not None:e.data.mocap_pos[mid]=point
            frames=src.step(command,np.zeros((1,2)),dev.get('handle_mask'),render=sub==a.action_repeat-1)
            gaze.append(frames[0].meta['gaze_cmd'])
        executed=LemLite.gaze_to_handle(command,np.asarray(gaze).mean(0)[None]);tick+=1
    try:
        for _ in range(warm):observe_and_hold()
        command=hold(frames[0],min(.3,float(dev['strength_limit'])))
        rng=np.random.default_rng(seed);side='left' if seed%2==0 else 'right'
        body=e.model.body(side+'_hand').id;center=e.data.xpos[body].copy()
        normal=np.r_[rng.uniform(-.25,.25,2),1.];normal/=np.linalg.norm(normal)
        origin=center+normal*.2;hits=[]
        for gid in range(e.model.ngeom):
            if e.model.geom_bodyid[gid]==body:
                distance=e._mj.mju_rayGeom(e.data.geom_xpos[gid],e.data.geom_xmat[gid],e.model.geom_size[gid],origin,-normal,int(e.model.geom_type[gid]))
                if distance>=0:hits.append(distance)
        if not hits:raise ValueError('pressure fixture missed the real hand geometry')
        surface=origin-normal*min(hits);depth=float(rng.uniform(.001,.004))
        point=surface+normal*(.025-depth)
        # Approach over one second, then retain the source for one second. No
        # body, sensory quality or force value is teleported to a target value.
        for i in range(round(1/dt)):
            fraction=(i+1)/round(1/dt);observe_and_hold(point+normal*.02*(1-fraction))
        for _ in range(round(1/dt)):observe_and_hold(point)
        result=model.step(frames,dict(dev,exploration_noise=0.),train=False,step_index=model.step_count,
                          executed_action=executed,executed_order_id=tick+1)
        observe_baseline()
        inputs=deepcopy(model.self_history.inputs)
        start,valid=targets(frames[0]);cmds=sequences(frames[0],result.motor_for_action,dev,a,count,steps,seed+1893)
        normalized=command_for_encoder(cmds).reshape(count,steps,-1)
        with torch.no_grad():
            percept=model.self_state.perceive(**inputs)
            branch={k:v.expand(count,*v.shape[1:]) for k,v in percept.items()}
            pred,var=model.self_state.forecast(branch,normalized,torch.full((count,steps),dt))
            if baseline is not None:
                before,_=baseline.forecast({k:v.expand(count,*v.shape[1:]) for k,v in baseline_percept.items()},normalized,torch.full((count,steps),dt))
        snap=src.save_full_state();traces=[];poses=[];pressures=[]
        for index in list(range(count))+[1,-1]:
            src.restore_full_state(snap);q=[];pose=[];pressure=[]
            if index==-1:e.data.mocap_pos[mid]=[0,0,2]
            for i in range(steps):
                cmd=cmds[1 if index==-1 else index,i].numpy()[None]
                for sub in range(a.action_repeat):frames=src.step(cmd,np.zeros((1,2)),dev.get('handle_mask'),render=False)
                target,mask=targets(frames[0]);q.append(target.numpy());pose.append(frames[0].bones.copy());pressure.append(float(frames[0].patches[:,0].max()))
            traces.append(np.asarray(q));poses.append(np.asarray(pose));pressures.append(pressure)
        np.testing.assert_array_equal(traces[1],traces[-2]);np.testing.assert_array_equal(poses[1],poses[-2])
        actual=np.asarray(traces[:count]);costs=[recovery_cost(q,valid) for q in actual]
        skin_mask=valid.numpy().copy();skin_mask[-10:]=False
        skin_costs=[recovery_cost(q,skin_mask) for q in actual]
        predicted=[recovery_cost(q,valid) for q in pred.numpy()]
        selected=int(np.argmin(predicted));oracle=int(np.argmin(costs))
        row=dict(seed=seed,side=side,depth_m=depth,source_position_m=point.tolist(),normal=normal.tolist(),
            initial_function_min=float(start[valid,0].min()),costs=costs,predicted_costs=predicted,
            initial_skin_min=float(start[:-10,0].min()),initial_motor_min=float(start[-10:][valid[-10:],0].min()),
            selected=selected,best_measured_candidate=oracle,choice_gain=costs[0]-costs[selected],
            available_gain_over_planner=costs[0]-costs[oracle],available_gain_over_hold=costs[1]-costs[oracle],
            removed_source_hold_cost=recovery_cost(traces[-1],valid),replay_max_error=0.,
            source_sensitivity=costs[1]-recovery_cost(traces[-1],valid),
            skin_costs=skin_costs,source_skin_sensitivity=skin_costs[1]-recovery_cost(traces[-1],skin_mask),
            available_skin_gain_over_hold=skin_costs[1]-min(skin_costs),
            peak_pressure_pa=max(map(max,pressures)),horizon_seconds=steps*dt,
            goal='Restore actual sensing/force availability; 0 cost means full function throughout, 1 means a fully unavailable site throughout.')
        row['recovery_opportunity_measured']=recovery_opportunity(row)
        row['action_prediction']=action_prediction_result(actual,pred.numpy(),valid)
        if baseline is not None:row['unfitted_action_prediction']=action_prediction_result(actual,before.numpy(),valid)
        training=dict(inputs=inputs,actions=normalized,durations=torch.full((count,steps),dt),
                      targets=torch.from_numpy(actual),valid=valid,start=start)
        atomic_json(out/'result.json',row);torch.save(training,out/'branches.pt')
        torch.save(dict(hold=torch.from_numpy(traces[-2]),removed_source=torch.from_numpy(traces[-1]),
                        pressures_pa=pressures,valid=valid),out/'controls.pt')
        return row,training
    finally:src.close()


def train(model,data,epochs,out):
    before={f'{name}.{k}':v.detach().clone() for name,m in modules(model).items() if name!='self_state' for k,v in m.state_dict().items()}
    updates=0
    for epoch in range(epochs):
        check_storage(out)
        losses=[]
        for sample in data:
            model.optimizer.zero_grad(set_to_none=True)
            count=sample['actions'].shape[0];percept=model.self_state.perceive(**sample['inputs'])
            branch={k:v.expand(count,*v.shape[1:]) for k,v in percept.items()}
            mean,var=model.self_state.forecast(branch,sample['actions'],sample['durations'])
            valid=sample['valid'][None,None].expand(count,mean.shape[1],-1)
            loss=model.self_state.consequence_loss(mean,var,sample['targets'],valid)
            mask=sample['valid'][None,:,None]
            loss=loss+((percept['present']-sample['start'][None]).square()*mask).sum()/(3*mask.sum())
            if not torch.isfinite(loss):raise ValueError('nonfinite copied self learning')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.self_state.parameters(),model.grad_clip)
            model.optimizer.step();updates+=1;losses.append(float(loss.detach()))
        status(out,'fitting_self_network',epoch=epoch+1,epochs=epochs,updates=updates,loss=float(np.mean(losses)))
    for name,m in modules(model).items():
        if name=='self_state':continue
        for k,v in m.state_dict().items():
            if not torch.equal(v,before[f'{name}.{k}']):raise ValueError('old brain weights changed during isolated self learning')
    return updates


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--reachability-only',action='store_true')
    p.add_argument('--prediction-only',action='store_true',help='Learn actual local function consequences; does not require or claim recovery-choice gain')
    p.add_argument('--worlds',type=int,default=8);p.add_argument('--candidates',type=int,default=8)
    p.add_argument('--steps',type=int,default=100);p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if args.smoke:args.worlds=2;args.candidates=3;args.steps=5;args.epochs=1
    if min(args.worlds,args.steps,args.epochs)<1 or args.worlds>16 or args.candidates<3:p.error('invalid experiment dimensions')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    a=configuration(args.checkpoint,out);model=build_learner(a,str(out/'checkpoints'));loaded=model.load_checkpoint(str(args.checkpoint))
    if loaded.get('reinitialized_components')!=[] or model.self_state is None:raise ValueError('requires a complete self-enabled checkpoint')
    baseline=deepcopy(model.self_state).eval().requires_grad_(False)
    dev,origin=_limits_for_run(str(args.checkpoint),model)
    declaration=dict(parent=str(args.checkpoint.resolve()),development_seeds=list(range(3100,3100+args.worlds)),
        held_out_seeds=list(range(7200,7200+args.worlds)),steps=args.steps,candidates=args.candidates,epochs=args.epochs,
        dt=.01*a.action_repeat,motor_hold=a.motor_hold,smoothing=a.smooth_actions,limits=dev,limits_source=origin,
        primary='Held-out reduction in physical function deficit compared with a smoothed held parent proposal over the same horizon; not the full online parent policy.',
        acceptance='Positive lower endpoint of the paired 95% mean interval, paired two-sided sign permutation p<=.05 with at least six worlds, exact branch replay and retained prior weights. No production license is issued by this experiment.',
        control='Identical physics before each action, repeated hold, source removed, left/right and unfamiliar source geometry.',
        physical_admission='Each development world must show source-dependent skin loss and recovery that improves on holding. Otherwise no fitting starts.',
        numerical_floor=float(NUMERICAL_FLOOR),numerical_floor_basis='Eight float32 unit-range epsilons; repeated physical branch must agree exactly.',
        scope='Single held-sequence recovery choices. Interruption, closed-loop completion, skill retention and UE transfer remain separate requirements.',
        smoke=args.smoke,reachability_only=args.reachability_only,
        prediction_only=args.prediction_only,
        prediction_primary='When prediction-only: every untouched world must remove positive action-effect error against an action-blind forecast AND beat the unfitted same network replaying its own history. Admission requires measured action differences, not recovery opportunity. No recovery choice is certified.',
        scene='Copied crib; existing sphere held fixed. Caregiver, critters and mobile absent. No infant movement reward or injected function loss.')
    if args.prediction_only:
        declaration['primary']=declaration['prediction_primary']
        declaration['physical_admission']='Each development world must contain measured action-dependent changes in actual functioning. No recovery opportunity is assumed.'
        declaration['acceptance']='Every held-out world beats both action-blind and unfitted forecasts; at least six worlds; exact branch replay; prior weights retained. No production license.'
    atomic_json(out/'declaration.json',declaration);data=[];development=[]
    for seed in declaration['development_seeds']:
        folder=out/str(seed);folder.mkdir();status(out,'collecting_function_changes' if args.prediction_only else 'measuring_recovery',seed=seed)
        row,sample=episode(model,a,dev,folder,seed,args.candidates,args.steps,4 if args.smoke else a.context_length,args.checkpoint)
        development.append(row);data.append(sample)
    atomic_json(out/'reachability.json',development)
    if args.reachability_only:
        status(out,'complete',verdict='PHYSICAL_BRANCHES_MEASURED',behavior_demonstrated=False);return
    admitted=(all(row['action_prediction']['valid'] for row in development) if args.prediction_only
              else all(recovery_opportunity(row) for row in development))
    if not admitted and not args.smoke:
        status(out,'complete',verdict='NO_MEASURED_ACTION_DIFFERENCE' if args.prediction_only else 'NO_MEASURED_RECOVERY_OPPORTUNITY',behavior_demonstrated=False);return
    updates=train(model,data,args.epochs,out);model.step_count=int(loaded['step'])+updates
    checkpoint=model.save_checkpoint(model.step_count);atomic_json(out/'args.json',dict(vars(a),_copied_self_updates=updates))
    held_out=[]
    for seed in declaration['held_out_seeds']:
        folder=out/str(seed);folder.mkdir();status(out,'testing_function_predictions' if args.prediction_only else 'held_out_recovery',seed=seed)
        row,_=episode(model,a,dev,folder,seed,args.candidates,args.steps,4 if args.smoke else a.context_length,checkpoint,baseline);held_out.append(row)
    comparison=comparison_result([r['choice_gain'] for r in held_out])
    passed=(not args.smoke and args.worlds>=6 and all(recovery_opportunity(row) for row in held_out)
            and comparison['valid'] and comparison['lower']>0 and comparison['paired_permutation_p']<=.05)
    report=dict(verdict='INSTRUMENT_EXECUTED' if args.smoke else ('RECOVERY_CHOICE_GAIN' if passed else 'RECOVERY_GAIN_NOT_ESTABLISHED'),
        comparison=comparison,worlds=held_out,checkpoint=checkpoint,old_weights_exact=True,production_license=False,
        scope=declaration['scope'])
    if args.prediction_only and not args.smoke:
        passed=args.worlds>=6 and all(r['action_prediction']['valid'] and r['action_prediction']['action_error_removed']>0
                   and r['action_prediction']['prediction_error']<r['unfitted_action_prediction']['prediction_error'] for r in held_out)
        report['verdict']='FUNCTION_PREDICTION_GAIN' if passed else 'FUNCTION_PREDICTION_GAIN_NOT_ESTABLISHED'
        report['scope']='Only predicts local physical functioning consequences. Does not establish purposeful recovery, completion or retention.'
    atomic_json(out/'report.json',report);status(out,'complete',verdict=report['verdict'],production_license=False)


if __name__=='__main__':
    try:main()
    except Exception as error:
        import sys,traceback
        if '--out' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--out')+1])
            if out.exists():
                atomic_json(out/'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
                status(out,'failed',error=str(error))
        raise
