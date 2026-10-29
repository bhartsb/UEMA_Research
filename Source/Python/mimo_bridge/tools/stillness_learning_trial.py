# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Matched learning with and without externally commanded body stillness.

Both arms retain identical exploration-noise timing, physical exposure and old
optimizer evidence. Frozen probes measure actual gaze and body outcomes. They
never supply rolling, reaching or toy-looking rewards to either learner.
"""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace
import numpy as np
import torch
from mimo_bridge import continuation_state,copied_execution
from mimo_bridge.acting_snapshot import perception_fingerprint
from mimo_bridge.datasource import SingleEnvSource
from mimo_bridge.lem_model import LemLite
from mimo_bridge.train_mimo import build_learner,env_kwargs_for
from .causal_test import _limits_for_run
from .caregiver_learning_trial import Orders,paired_result,verify_loaded
from .developmental_probe import run_world
from .self_recovery_experiment import configuration,hold,targets
from .training_continuation import atomic_json
from .caregiver_fit_continuation import execution_source_digest

SCHEMA='matched_stillness_learning_v1'
ARMS=('scheduled','model')


class MatchedOrders(Orders):
    """Production noise-bout logic, with only the body hold switched by arm."""
    def limits(self,dev,i,dt,train):
        dev=dict(dev);a=self.a
        if not train or a.no_babble:dev['exploration_noise']=0.
        elif a.babble_noise is not None:dev['exploration_noise']=float(a.babble_noise)
        elif a.babble_scale!=1.:dev['exploration_noise']=dev.get('exploration_noise',0.)*a.babble_scale
        if train and a.babble_bouts in ('on','real') and not a.no_babble:
            if i>=self.until:
                self.bout=not self.bout
                seconds=self.rng.uniform(2.,5.) if self.bout else self.rng.uniform(1.,3.)
                self.until=i+max(1,int(seconds/dt))
            if not self.bout:dev['exploration_noise']=0.
        return dev

    def state_dict(self):
        return dict(held=self.held,executed=self.executed,order=self.order,
                    bout=self.bout,until=self.until,rng=self.rng.bit_generator.state)

    def load_state_dict(self,state):
        for key in ('held','executed','order','bout','until'):setattr(self,key,deepcopy(state[key]))
        self.rng.bit_generator.state=deepcopy(state['rng'])


def source_integrity(checkpoint):
    """A corrupt runtime cannot quietly become a cold comparison baseline."""
    state=torch.load(checkpoint/'runtime_state.pt',map_location='cpu',weights_only=False)
    if not isinstance(state,dict):raise ValueError('invalid source working statistics')
    del state
    return dict(runtime_readable=True,source_episode='New independent world; old learned runtime retained')


def status(out,stage,**data):
    row=dict(stage=stage,pid=os.getpid(),timestamp=time.time(),**data)
    atomic_json(out/'status.json',row)
    print(json.dumps(row),flush=True)


def reserve(out):
    if shutil.disk_usage(out).free<4*1024**3:raise OSError('Less than 4 GiB free; preserve existing comparisons and checkpoints')


def file_digest(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024*1024):digest.update(block)
    return digest.hexdigest()


def settings(parent,out,arm=None):
    a=configuration(parent,out);a.device='cpu';a.self_choice='off';a.caregiver_record_dir=''
    # The question concerns the same noise schedule with vs without body hold.
    if a.babble_bouts!='real' or a.no_babble:
        raise ValueError('source run has no real stillness/noise schedule to compare')
    if arm is not None:a.rest_control='scheduled' if arm=='scheduled' else 'choice'
    return a


def save_copy(model,source,orders,path,step,trial_step,phase,config,rows):
    reserve(path)
    model.receipt['episode_continuation']=continuation_state.VERSION
    def companion(staging,final):
        copied_execution.save_episode(staging,final,source,model,
            dict(step=step,trial_step=trial_step,phase=phase,orders=orders.state_dict(),rows=rows,
                execution_source_sha256=execution_source_digest(extra=[Path(__file__)])),
            dict(schema=SCHEMA,args=vars(config)))
    saved=model.save_checkpoint(step,companion_writer=companion)
    atomic_json(path/'resume.json',dict(checkpoint=saved,trial_step=trial_step,phase=phase,
        resume_command='Repeat this trial command with --resume; saved world, learner and order state continue.'))
    inventory_path=path/'checkpoint_inventory.json'
    inventory=json.loads(inventory_path.read_text()) if inventory_path.exists() else []
    inventory.append(dict(path=str(Path(saved).resolve()),step=step,trial_step=trial_step,phase=phase,
        files={p.name:dict(bytes=p.stat().st_size,sha256=file_digest(p))
               for p in Path(saved).iterdir() if p.is_file()},pruned=False))
    atomic_json(inventory_path,inventory)
    prune_superseded_intermediates(path,inventory,Path(saved))
    atomic_json(inventory_path,inventory)
    return Path(saved)


def prune_superseded_intermediates(folder,inventory,newest):
    """Remove only superseded fitting snapshots; retain phase endpoints and hashes."""
    root=(folder/'checkpoints').resolve();newest=newest.resolve()
    if newest.parent!=root:raise ValueError('new checkpoint is outside this copied arm')
    for row in inventory:
        target=Path(row['path']).resolve()
        if row['phase']!='fitting' or row.get('pruned') or target==newest:continue
        if target.parent!=root or not target.name.startswith('ckpt_step_'):
            raise ValueError('refusing to prune outside the copied checkpoint directory')
        if target.exists():shutil.rmtree(target)
        row.update(pruned=True,pruned_after=str(newest),reason='Superseded copied fitting state; full file hashes and step record retained')


def latest_copy(folder,fallback):
    # The atomically published checkpoint can outlive a crash before the small
    # resume pointer was written. Recover that newest complete copy too.
    candidates=list((folder/'checkpoints').glob('ckpt_step_*'))
    for path in sorted(candidates,key=lambda p:int(p.name.rsplit('_',1)[1]),reverse=True):
        state=continuation_state.load(path)
        if state is None or state['config'].get('schema')!=SCHEMA:
            raise ValueError('unexpected checkpoint in the copied arm')
        return path
    return fallback


def warm_pair(parent,out,a,dev,seed,warm):
    """Save one fully warmed physical/learner start, used identically by both arms."""
    cached=out/'initial.json'
    if cached.exists():return Path(json.loads(cached.read_text())['checkpoint'])
    reserve(out);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    folder=out/'initial';folder.mkdir(exist_ok=True);a=deepcopy(a);a.out=str(folder)
    atomic_json(folder/'args.json',vars(a))
    model=build_learner(a,str(folder/'checkpoints'));loaded=model.load_checkpoint(str(parent))
    verify_loaded(model,loaded,parent,True);model.reset_transition_state();initial_hash=perception_fingerprint(model)
    kw=env_kwargs_for(a,dev['phase_name'],str(out/'physical_world'));kw['seed']=seed
    source=SingleEnvSource(**kw)
    try:
        frames=source.reset();order=MatchedOrders(a,seed);command=hold(frames[0],float(dev['strength_limit']))
        for i in range(warm):
            result=model.step(frames,dict(dev,exploration_noise=0.),train=False,step_index=model.step_count,
                              executed_action=order.executed,executed_order_id=order.order or None)
            result.motor_for_action=command.copy();result.gaze_out=np.zeros((1,2))
            frames=order.advance(model,result,source,i,False)
        if perception_fingerprint(model)!=initial_hash:raise ValueError('warm-up changed trained weights')
        # Reset only the declared schedule clock. Physical and sensory histories
        # and the last actually executed order all remain continuous.
        order.bout=True;order.until=0
        checkpoint=save_copy(model,source,order,folder,model.step_count,0,'warm',a,[])
        atomic_json(cached,dict(checkpoint=str(checkpoint),warm_steps=warm,world_seed=seed,
                               optimizer_preserved=True,weights_unchanged=True))
        return checkpoint
    finally:source.close()


def fit_arm(initial,out,dev,seed,train_steps,retention_steps,arm,progress_out):
    out.mkdir(exist_ok=True);a=settings(initial,out,arm);atomic_json(out/'args.json',vars(a))
    previous=out/'resume.json';parent=latest_copy(out,initial)
    episode=continuation_state.load(parent)
    if episode is None:raise ValueError('matched fit requires its complete warmed episode')
    if episode['trainer'].get('execution_source_sha256')!=execution_source_digest(extra=[Path(__file__)]):
        raise ValueError('Saved stillness fit lacks the current exact execution contract; earlier attempts remain preserved')
    model=build_learner(a,str(out/'checkpoints'));loaded=model.load_checkpoint(str(parent))
    verify_loaded(model,loaded,parent,True)
    source=SingleEnvSource(**episode['worlds'][0]['env_kwargs'])
    rows=deepcopy(episode['trainer']['rows']);orders=MatchedOrders(a,seed)
    orders.load_state_dict(episode['trainer']['orders']);i0=episode['trainer']['trial_step']
    checkpoints=json.loads((out/'phases.json').read_text()) if (out/'phases.json').exists() else {}
    if episode['trainer']['phase'] in ('end','retention'):
        checkpoints[episode['trainer']['phase']]=str(parent)
    try:
        frames=copied_execution.restore_episode(model,source,episode)
        del episode
        for i in range(i0,train_steps+retention_steps):
            limits=orders.limits(dev,i,source.env.dt*a.action_repeat,True)
            result=model.step(frames,limits,train=True,step_index=model.step_count,
                executed_action=orders.executed,executed_order_id=orders.order or None)
            if not np.isfinite(result.loss):raise ValueError('nonfinite copied learning loss')
            frames=orders.advance(model,result,source,i,True)
            target,valid=targets(frames[0]);q=float(target[valid,0].min())
            rows.append(dict(trial_step=i,sim_time=float(frames[0].sim_time),function_min=q,
                noise=float(limits.get('exploration_noise',0.)),bout=bool(orders.bout),
                scheduled_hold=bool(arm=='scheduled' and not orders.bout),loss=float(result.loss)))
            if i%25==0:status(progress_out,'fitting',arm=arm,world_seed=seed,step=i,steps=train_steps+retention_steps)
            end=(i+1==train_steps);retained=(i+1==train_steps+retention_steps)
            if (i+1)%300==0 or end or retained:
                phase='end' if end else 'retention' if retained else 'fitting'
                saved=save_copy(model,source,orders,out,model.step_count,i+1,phase,a,rows)
                if end or retained:
                    checkpoints[phase]=str(saved);atomic_json(out/'phases.json',checkpoints)
        atomic_json(out/'fit.json',dict(arm=arm,world_seed=seed,online_updates=train_steps+retention_steps,
            learning_steps=train_steps,intervening_retention_learning_steps=retention_steps,
            exposure_seconds=(train_steps+retention_steps)*source.env.dt*a.action_repeat,
            minimum_function=min(row['function_min'] for row in rows),
            mean_function=float(np.mean([row['function_min'] for row in rows])),
            scheduled_steps=sum(row['scheduled_hold'] for row in rows),trace=rows,checkpoints=checkpoints,
            offline_replay_updates=0,scope='Same physical time and online updates in both arms; continuation is saved'))
        return checkpoints
    except Exception as error:
        atomic_json(out/'failure.json',dict(error=str(error),last_completed_trial_step=rows[-1]['trial_step'] if rows else None,
            saved_resume=str(previous),partial_copy_only=True))
        raise
    finally:source.close()


def behavior(checkpoint,out,seed,dev,warm,steps,progress_out):
    """Frozen tests remove scheduled holds from both fitted arms equally."""
    result_path=out/'result.json'
    if result_path.exists():return json.loads(result_path.read_text())
    out.mkdir(exist_ok=True);a=configuration(checkpoint,out);a.device='cpu';a.self_choice='off'
    summaries={}
    for profile in ('toy','body_floor'):
        path=out/profile;path.mkdir(exist_ok=True)
        status(progress_out,'behavior_probe',checkpoint=str(checkpoint),world_seed=seed,profile=profile)
        summaries[profile]=run_world(a,checkpoint,path,seed,dev,warm,steps,profile)
    conditions={r.get('source_condition',r['condition']):r for r in summaries['toy']['conditions']
                if r.get('arm','candidate')=='candidate'}
    body=json.loads((out/'body_floor'/'body_only.json').read_text())
    absent=json.loads((out/'toy'/'absent.json').read_text())
    threshold=a.fovea_fov/2 if a.fovea_size else a.eye_fov/2
    candidate_rows=[(profile,r) for profile,summary in summaries.items() for r in summary['conditions']
                    if r.get('arm','candidate')=='candidate']
    result=dict(gaze_relocated_fraction=conditions['relocated']['target_center_in_fovea_fraction'],
        gaze_absent_direction_fraction=float(np.mean(np.asarray(absent['measurements']['target_angle_deg'])<=threshold)),
        gaze_interrupted_dwell_s=conditions['interrupted']['longest_foveal_dwell_s'],
        prone_orientation_fraction=float(np.mean(np.asarray(body['measurements']['body_ventral_up_dot'])<0)),
        body_displacement_m=summaries['body_floor']['conditions'][0]['maximum_horizontal_displacement_m'],
        function_min=min(min(json.loads((out/profile/(r.get('source_condition',r['condition'])+'.json')).read_text())['measurements']['function_min'])
                         for profile,r in candidate_rows),
        raw=summaries,scope='Actual orientation and posture; rolling intention and generalized acquired function require further causal skill tests')
    atomic_json(result_path,result);return result


def summarize(worlds,smoke=False):
    if not worlds:raise ValueError('no paired behavior measurements')
    endpoint='gaze_relocated_fraction'
    effects={phase:[w['model'][phase][endpoint]-w['scheduled'][phase][endpoint] for w in worlds]
             for phase in ('end','retention')}
    viability=[w['model']['retention']['function_min']-w['scheduled']['retention']['function_min'] for w in worlds]
    tests={phase:paired_result(values) for phase,values in effects.items()}
    tests['viability']=paired_result(viability)
    for arm in ARMS:
        for phase in ('end','retention'):
            tests[arm+'_'+phase+'_gain']=paired_result([w[arm][phase][endpoint]-w['baseline'][endpoint] for w in worlds])
        tests[arm+'_viability']=paired_result([w[arm]['retention']['function_min']-w['baseline']['function_min'] for w in worlds])
    for key,value in tests.items():
        value['scale']=('Arm minus unchanged baseline ' if '_' in key else 'Free-body minus scheduled-body ')+(
            'minimum skin function' if 'viability' in key else 'relocated-target foveation fraction')
    def positive(row):return row.get('valid') and row['lower']>0 and row['paired_permutation_p']<=.05
    def negative(row):return row.get('valid') and row['upper']<0 and row['paired_permutation_p']<=.05
    safe=(tests['viability'].get('valid') and tests['viability']['lower']>=0 and
          tests['model_viability'].get('valid') and tests['model_viability']['lower']>=0)
    reverse_safe=(tests['viability'].get('valid') and tests['viability']['upper']<=0 and
                  tests['scheduled_viability'].get('valid') and tests['scheduled_viability']['lower']>=0)
    verdict='INCONCLUSIVE'
    if not smoke and all(positive(tests[p]) for p in ('end','retention','model_end_gain','model_retention_gain')) and safe:
        verdict='FREE_BODY_IMPROVES_RETAINED_OBJECT_LOOKING'
    if (not smoke and all(negative(tests[p]) for p in ('end','retention')) and reverse_safe
            and all(positive(tests[p]) for p in ('scheduled_end_gain','scheduled_retention_gain'))):
        verdict='SCHEDULED_HOLD_IMPROVES_RETAINED_OBJECT_LOOKING'
    return dict(verdict='INSTRUMENT_EXECUTED' if smoke else verdict,paired_worlds=len(worlds),comparisons=tests,
        capability_pass=False,production_license=False,
        interpretation='A schedule comparison can favor a learning condition. It cannot certify purposeful looking or rolling by itself.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--worlds',type=int,default=8);parser.add_argument('--train-steps',type=int,default=900)
    parser.add_argument('--probe-steps',type=int,default=100);parser.add_argument('--retention-steps',type=int,default=0)
    parser.add_argument('--seed',type=int,default=72163);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args(argv)
    if args.smoke:args.worlds=2;args.train_steps=4;args.probe_steps=4;args.retention_steps=2
    if not 2<=args.worlds<=16 or args.train_steps<2 or args.probe_steps<2:parser.error('invalid bounded comparison size')
    if not args.retention_steps:args.retention_steps=max(2,args.train_steps//5)
    if args.retention_steps<2:parser.error('retention needs intervening learning and physical exposure')
    args.out.mkdir(parents=True,exist_ok=args.resume);torch.set_num_threads(1)
    try:
        integrity=source_integrity(args.checkpoint)
        a=settings(args.checkpoint,args.out)
        saved=json.loads((args.checkpoint.parent.parent/'args.json').read_text())
        if saved.get('self_choice')=='on':raise ValueError('this comparison must use the saved acting snapshot when a qualified actor drives the source run')
        dev,origin=_limits_for_run(str(args.checkpoint),SimpleNamespace(dev_limits=lambda _:{}))
        if not dev.get('phase_name'):raise ValueError('actual body-development limits unavailable')
        warm=3 if args.smoke else int(a.context_length)
        declaration=dict(schema=SCHEMA,checkpoint=str(args.checkpoint),seed=args.seed,worlds=args.worlds,
            train_steps=args.train_steps,retention_steps=args.retention_steps,probe_steps=args.probe_steps,
            warm_steps=warm,integrity=integrity,limits=dev,limits_source=origin,
            arms=list(ARMS),same='Checkpoint weights, all Adam history, warmed world and causal history, noise timing, online updates, physical exposure',
            changed='During a noise pause, scheduled arm holds achieved body pose; model arm continues its chosen body command',
            retention='After additional equal-budget unsupervised learning in the same continuing physical world',
            primary='Relocated-target actual foveation fraction at end AND after intervening learning; free minus scheduled',
            gate='Both end and retained gain over other arm AND unchanged baseline have positive paired 95% intervals and permutation p<=.05; minimum skin function noninferior to other arm AND baseline; reverse rule for scheduled advantage',
            secondary='Absent/interrupted object looking, prone orientation, horizontal displacement and raw contact/body traces; no selective pass from these',
            scope='No skill reward, policy target labels, production adoption, or actor license. Body posture alone does not establish intention.',
            no_offline_replay='Online optimizer retained and updated equally; no unequal offline replay budget',instrument_smoke=args.smoke)
        declaration_path=args.out/'declaration.json'
        if declaration_path.exists() and json.loads(declaration_path.read_text())!=declaration:raise ValueError('resumed trial differs from its preregistered declaration')
        atomic_json(declaration_path,declaration);results=[]
        for index in range(args.worlds):
            seed=args.seed+index;pair=args.out/str(seed);pair.mkdir(exist_ok=True)
            if (pair/'result.json').exists():results.append(json.loads((pair/'result.json').read_text()));continue
            initial=warm_pair(args.checkpoint,pair,a,dev,seed,warm);gc.collect()
            # Probe worlds are held out from fitting and identical for both arms.
            probe_seed=seed+1000003
            baseline=behavior(initial,pair/'baseline',probe_seed,dev,warm,args.probe_steps,args.out);gc.collect()
            row=dict(world_seed=seed,probe_seed=probe_seed,baseline=baseline)
            arm_order=ARMS if index%2==0 else tuple(reversed(ARMS))
            for arm in arm_order:
                folder=pair/arm;checkpoints=fit_arm(initial,folder,dev,seed,args.train_steps,args.retention_steps,arm,args.out);gc.collect()
                row[arm]={phase:behavior(Path(path),folder/('probe_'+phase),probe_seed,dev,warm,args.probe_steps,args.out)
                          for phase,path in checkpoints.items()}
                gc.collect()
            first=json.loads((pair/'scheduled'/'fit.json').read_text());second=json.loads((pair/'model'/'fit.json').read_text())
            if [(r['noise'],r['bout']) for r in first['trace']]!=[(r['noise'],r['bout']) for r in second['trace']]:
                raise ValueError('noise opportunity schedules differed between arms')
            if first['exposure_seconds']!=second['exposure_seconds']:raise ValueError('physical exposure budgets differ')
            atomic_json(pair/'result.json',row);results.append(row)
        report=summarize(results,args.smoke);report.update(worlds=results,declaration=declaration,completed_at=time.time())
        atomic_json(args.out/'report.json',report);status(args.out,'complete',verdict=report['verdict'])
        return 0
    except Exception as error:
        status(args.out,'failed',error=str(error));raise


if __name__=='__main__':raise SystemExit(main())
