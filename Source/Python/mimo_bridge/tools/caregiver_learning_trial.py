# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Copied learners: responsive adult versus another world's recorded adult.

Production is never controlled by this tool. The complete design is written
before fitting. Frozen probes have the same scene and audio across fitted arms,
and each condition reconstructs its learner so trials cannot share adaptation.
"""
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch
from mimo_bridge.train_mimo import parse_args, build_learner, env_kwargs_for
from mimo_bridge.tools.causal_test import _limits_for_run
from mimo_bridge.datasource import SingleEnvSource
from mimo_bridge.caregiver_visual import pose_adult, pose_record, clear_approach
from mimo_bridge.caregiver_language import GROUP_KEYS
from mimo_bridge.lem_model import LemLite, COMMAND_POSITION_RANGE_CM


def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    if path.name in ('failure.json','paused_fit.json') and path.exists():
        archive=path.parent/(path.stem+'_history');archive.mkdir(exist_ok=True)
        previous=path.read_bytes();digest=hashlib.sha256(previous).hexdigest()
        preserved=archive/(digest+'.json')
        if not preserved.exists():preserved.write_bytes(previous)
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
    if path.name=='status.json':
        from html import escape
        stage=value['stage'];done=stage in ('complete','failed')
        title={'fit':'Learning in the copied worlds','fit_parallel':'Learning in independent copied worlds',
            'waiting_for_resources':'Waiting for recording and production resources','probe':'Testing the copied learners',
            'waiting_to_save':'Copied learning is paused while its exact state awaits recording space',
            'building_fixed_probes':'Preparing identical tests','complete':'Comparison finished',
            'failed':'Comparison stopped; inspect the recorded error'}.get(stage,stage)
        detail='; '.join(f'{k}: {v}' for k,v in value.items() if k not in ('stage','timestamp','pid'))
        page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        if not done:page+='<meta http-equiv="refresh" content="30">'
        page+='<title>Does a responding caregiver help learning?</title><style>body{font:18px/1.6 system-ui;background:#f5f3ed;color:#18333b;max-width:850px;margin:6vh auto;padding:24px}h1{line-height:1.15}a{color:#125e72}pre{white-space:pre-wrap}</style>'
        page+='<p>CAREGIVER LEARNING COMPARISON</p><h1>'+escape(title)+'</h1><p>'+escape(detail)+'</p>'
        page+='<p>Live training continues separately. These copies compare a responding adult with another world’s recorded adult. The primary test asks whether learned object following improves. Intention and language understanding are not certified by this test.</p>'
        page+='<p><a href="declaration.json">Design fixed before learning</a> · <a href="status.json">Current measurement</a>'
        if stage=='complete':page+=' · <a href="report.json">Full result</a>'
        if stage=='failed':page+=' · <a href="failure.json">Recorded error</a>'
        page+='</p><p>Updated '+escape(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(value['timestamp'])))+'</p></html>'
        target=path.parent/'index.html';pending=target.with_suffix('.tmp');pending.write_text(page,encoding='utf-8');pending.replace(target)


def donor_map(worlds):
    if worlds<2 or worlds%2:raise ValueError('replay controls require disjoint pairs of responsive worlds')
    return [i^1 for i in range(worlds)]


def alignment(eye,forward,target):
    delta=np.asarray(target)-eye;norm=np.linalg.norm(delta)
    if norm<1e-9 or not np.isfinite(delta).all():raise ValueError('invalid gaze target')
    return float(np.clip((1+np.dot(forward,delta/norm))/2,0,1))


def paired_result(values):
    from scipy.stats import t
    x=np.asarray(values,float)
    if len(x)<2 or not np.isfinite(x).all():return dict(valid=False)
    mean=float(x.mean());se=x.std(ddof=1)/np.sqrt(len(x));margin=float(t.ppf(.975,len(x)-1)*se)
    perm=[abs(float(np.mean(x*np.asarray(sign)))) for sign in itertools.product((-1,1),repeat=len(x))]
    p=sum(v>=abs(mean)-1e-12 for v in perm)/len(perm)
    return dict(valid=True,mean=mean,lower=mean-margin,upper=mean+margin,paired_permutation_p=p,
                scale='Difference in gaze alignment: 0=no advantage, positive=favors responsive training')


def settings(checkpoint,out,speech,device,seed):
    saved=json.loads((checkpoint.parent.parent/'args.json').read_text(encoding='utf-8'))
    a=parse_args(['--out',str(out)]);vars(a).update(saved)
    a.out=str(out);a.resume=str(checkpoint);a.device=device;a.seed=seed;a.n_envs=1
    a.caregiver_speech_dir=str(speech.resolve());a.caregiver_record_dir=''
    return a


def validate_checkpoint(checkpoint):
    """Do not reuse a completed fit whose weights or learned runtime are missing."""
    checkpoint=Path(checkpoint)
    for name in ('model.safetensors','model_receipt.json','optimizer.pt','runtime_state.pt'):
        if not (checkpoint/name).is_file() or not (checkpoint/name).stat().st_size:
            raise ValueError(f'copied checkpoint is incomplete: {checkpoint/name}')
    runtime=torch.load(checkpoint/'runtime_state.pt',map_location='cpu',weights_only=False)
    if not isinstance(runtime,dict):raise ValueError('copied checkpoint working statistics are invalid')


def completed_fit(base,steps):
    """Resolve the real completed receipt, including replacement attempts."""
    paths=[base]+sorted(base.glob('attempt_*')) if base.exists() else []
    completed=[]
    for folder in paths:
        receipt=folder/'fit.json'
        if not receipt.exists():continue
        data=json.loads(receipt.read_text())
        if data.get('online_updates')!=steps or data.get('offline_replay_updates')!=0:
            raise ValueError('completed fit differs from the original learning budget')
        validate_checkpoint(Path(data['checkpoint']))
        if not Path(data['adult_tape']).is_file() or not isinstance(data['source_utterances'],list):
            raise ValueError('completed fit lost its actual adult tape or speech dose')
        completed.append((folder,data))
    if len(completed)>1:raise ValueError('multiple completed fits for one arm; do not silently select a winner')
    return completed[0] if completed else None


def next_attempt(base):
    if not base.exists():return base
    index=2
    while (base/f'attempt_{index:03d}').exists():index+=1
    return base/f'attempt_{index:03d}'


def completed_probe(base,kind,steps):
    paths=[base]+sorted(base.glob('attempt_*')) if base.exists() else []
    found=[]
    for folder in paths:
        path=folder/'result.json'
        if not path.exists():continue
        result=json.loads(path.read_text())
        if (result.get('kind')!=kind or result.get('samples')!=steps or
                not np.isfinite(result.get('mean_alignment',np.nan))):
            raise ValueError('completed probe differs from its declared condition or exposure')
        found.append((folder,result))
    if len(found)>1:raise ValueError('multiple completed probes for one condition')
    return found[0] if found else None


def record_resume_version(out,hashes,declaration,operations=None):
    """Preserve the original protocol/source receipt and append operational repairs."""
    import uuid
    version=out/'resume_versions';version.mkdir(exist_ok=True)
    original=out/'declaration.json'
    if not original.exists() or json.loads(original.read_text())!=declaration:
        raise ValueError('resume changed the original declared experiment')
    write(version/(str(time.time_ns())+'_'+uuid.uuid4().hex+'.json'),dict(timestamp=time.time(),
        declaration_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),source_hashes=hashes,
        changes='Resume orchestration, intact-runtime validation, atomic checkpoint publication and optional isolated fitting workers. Original budgets, worlds, device, controls and pass rule retained.',
        operations=operations,
        incomplete_fits='New-format partials continue their exact saved episode, speech and remaining update budget. Older incompatible partials remain preserved; a new attempt starts from the original parent.',
        probe_isolation='Every probe condition reloads the completed fit and its learned runtime; no schema or tracker carries from a previous probe condition.'))


def learner(a,checkpoint,out,train):
    out.mkdir(parents=True,exist_ok=True)
    write(out/'args.json',vars(a))
    validate_checkpoint(checkpoint)
    model=build_learner(a,str(out/'checkpoints'))
    loaded=model.load_checkpoint(str(checkpoint),load_optimizer=train)
    verify_loaded(model,loaded,checkpoint,train)
    model.reset_transition_state()
    return model,int(loaded['step'])


def verify_loaded(model,loaded,checkpoint,train):
    if loaded.get('reinitialized_components')!=[] or not loaded.get('loaded_components'):
        raise ValueError('copied learner reinitialized existing modules or lacks a complete load receipt')
    if train:
        expected=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False)['optimizer']
        from .migrate_trunk import modules
        from safetensors import safe_open
        with safe_open(str(checkpoint/'model.safetensors'),framework='pt',device='cpu') as saved:
            saved_names=set(saved.keys())
        names={id(parameter):f'{module_name}.{name}' for module_name,module in modules(model).items()
               for name,parameter in module.named_parameters()}
        if not expected['state'] or len(expected['param_groups'])!=len(model.optimizer.param_groups):
            raise ValueError('copied learner lost optimizer histories')
        checked=0
        for old_group,new_group in zip(expected['param_groups'],model.optimizer.param_groups):
            # New neutral parameters shift integer IDs. Existing qualified names
            # retain their order within each decay group; compare actual histories.
            old_parameters=[parameter for parameter in new_group['params'] if names[id(parameter)] in saved_names]
            if len(old_parameters)!=len(old_group['params']):raise ValueError('saved optimizer layout cannot be reconstructed')
            for key,parameter in zip(old_group['params'],old_parameters):
                state=expected['state'].get(key,{})
                restored=model.optimizer.state.get(parameter,{})
                if set(restored)!=set(state):raise ValueError('copied optimizer fields changed')
                for name,value in state.items():
                    same=torch.equal(value.cpu(),restored[name].cpu()) if torch.is_tensor(value) else value==restored[name]
                    if not same:raise ValueError(f'copied optimizer history changed: {names[id(parameter)]}/{name}')
                checked+=bool(state)
        if checked!=len(expected['state']):raise ValueError('not every saved optimizer history was checked')


def source(a,out,seed,phase,tape=None,isolate=False,record=False):
    kw=env_kwargs_for(a,phase,str(out));kw.update(seed=seed,scene='crib',run_dir=str(out),
        caregiver_mode='replay' if tape else 'responsive',caregiver_replay=str(tape) if tape else '',
        caregiver_record_dir=str(out/'recordings') if record and not tape else '')
    if isolate:kw.update(gaze_reflex=False,spontaneous_saccades=False)
    return SingleEnvSource(**kw)


class Orders:
    def __init__(self,a,seed):
        self.a=a;self.held=None;self.executed=None;self.order=0
        self.rng=np.random.default_rng(seed+4242);self.bout=True;self.until=0

    def limits(self,dev,i,dt,train):
        a=self.a;dev=dict(dev)
        dev['exploration_noise']=0. if not train or a.no_babble else float(a.babble_noise or 0.)
        if train and a.babble_bouts in ('on','real'):
            if i>=self.until:
                self.bout=not self.bout
                seconds=self.rng.uniform(2.,5.) if self.bout else self.rng.uniform(1.,3.)
                self.until=i+max(1,int(seconds/dt))
            if not self.bout:dev['exploration_noise']=0.
        return dev

    def advance(self,model,result,src,i,train):
        if self.held is None or i%max(1,self.a.motor_hold)==0:
            self.held=np.array(result.motor_for_action,copy=True);self.order+=1
        cmd=self.held.copy()
        if train and self.a.babble_bouts=='real' and not self.bout and self.a.rest_control=='scheduled':
            achieved=model.last_achieved.detach().float().cpu().numpy()
            cmd[...,:3]=achieved[...,:3]*COMMAND_POSITION_RANGE_CM
            quat=achieved[...,3:7];cmd[...,3:7]=quat/np.maximum(np.linalg.norm(quat,axis=-1,keepdims=True),1e-6)
        gaze=[]
        for rep in range(max(1,self.a.action_repeat)):
            frames=src.step(cmd,result.gaze_out,render=rep==self.a.action_repeat-1)
            gaze.append(frames[0].meta['gaze_cmd'])
        self.executed=LemLite.gaze_to_handle(cmd,np.asarray(gaze).mean(0)[None])
        return frames


def fit(a,parent,out,seed,steps,dev,tape=None,progress_out=None,step_guard=None,resume_checkpoint=None,segment_out=None):
    from . import caregiver_fit_continuation as continuation
    from mimo_bridge.self_intention import fingerprint
    copy_started=time.time()
    contract=continuation.binding(a,parent,out,seed,steps,dev,tape)
    saved=None if resume_checkpoint is None else continuation.load(resume_checkpoint,contract)
    model,loaded_step=learner(a,parent if saved is None else Path(resume_checkpoint),out,True)
    src=source(a,out/'scene',seed,dev['phase_name'],tape,record=True)
    try:
        # Reserve one utterance per group for the later word-cue probe.
        src.env.caregiver.bank.groups={k:list(v[:-1]) for k,v in src.env.caregiver.bank.groups.items()}
        orders=Orders(a,seed);checkpoint=None
        if saved is None:
            frames=src.reset();step0=loaded_step;completed_before=0
            initial=fingerprint(model.encoder_registry);utterances=[];last_utterance=None;prior_wall=0.
        else:
            trainer=saved['trainer'];resume=trainer['fit_resume']
            if loaded_step!=trainer['step']:raise ValueError('Partial fit weights and episode step disagree')
            frames=continuation.restore(model,src,orders,saved)
            step0=resume['parent_step'];completed_before=trainer['trial_step'];prior_wall=resume['active_wall_seconds']
            initial=resume['initial_encoder_sha256'];utterances=deepcopy(resume['utterances'])
            last_utterance=deepcopy(resume['last_utterance']);checkpoint=Path(resume_checkpoint)
            write((segment_out or out)/'resumed_fit.json',dict(checkpoint=str(checkpoint),
                completed_updates=completed_before,remaining_updates=steps-completed_before,
                contract=continuation.CONTRACT,original_parent=str(parent),timestamp=time.time()))
            del saved
        last_saved=completed_before if checkpoint is not None else -1
        def save_at(completed):
            nonlocal checkpoint,last_saved
            if completed==last_saved:return checkpoint
            if step_guard is not None and hasattr(step_guard,'before_save'):step_guard.before_save(completed)
            def companion(staging,final):
                continuation.save(staging,final,src,model,dict(step=step0+completed,trial_step=completed,
                    held=orders.held,executed=orders.executed,order=orders.order,
                    bout=orders.bout,until=orders.until,bout_rng=orders.rng.bit_generator.state,
                    frame_meta=[deepcopy(f.meta) for f in frames],
                    fit_resume=dict(contract=continuation.CONTRACT,binding=contract,parent_step=step0,
                        utterances=deepcopy(utterances),last_utterance=deepcopy(last_utterance),
                        initial_encoder_sha256=initial,active_wall_seconds=prior_wall+time.time()-copy_started)),vars(a))
            checkpoint=Path(model.save_checkpoint(step0+completed,companion_writer=companion));last_saved=completed
            return checkpoint
        def pause_at(completed,error):
            paused=save_at(completed)
            write(out/'paused_fit.json',dict(checkpoint=str(paused),completed_updates=completed,
                declared_updates=steps,reason=str(error),timestamp=time.time(),
                reusable_as_completed_fit=False,resumable_contract=continuation.CONTRACT,
                scope='Exact partial learning, speech and physical episode preserved; no comparison outcome'))
        for i in range(completed_before,steps):
            if step_guard:
                from .caregiver_fit_workers import CopyResourcesUnavailable
                try:step_guard(model,i)
                except CopyResourcesUnavailable as error:
                    pause_at(i,error)
                    raise
            limits=orders.limits(dev,i,src.env.dt*a.action_repeat,True)
            result=model.step(frames,limits,train=True,step_index=step0+i,
                              executed_action=orders.executed,executed_order_id=orders.order or None)
            if not np.isfinite(result.loss):raise ValueError('nonfinite copied training loss')
            frames=orders.advance(model,result,src,i,True)
            utterance=src.env.caregiver.state.get('last_utterance')
            if utterance and utterance!=last_utterance:
                utterances.append((utterance['time'],utterance['key']));last_utterance=deepcopy(utterance)
            if i%50==0:
                now=time.time()
                progress=dict(stage='fit',seed=seed,step=i+1,steps=steps,loss=float(result.loss),
                    sim_time=frames[0].sim_time,timestamp=now,arm=out.name,pid=__import__('os').getpid(),
                    copy_started_at=copy_started,copy_wall_seconds=prior_wall+now-copy_started,
                    resumed_after_updates=completed_before,segment_updates=i+1-completed_before,
                    updates_per_wall_second=(i+1-completed_before)/(now-copy_started))
                if str(a.device).startswith('cuda'):
                    progress.update(cuda_allocated_bytes=torch.cuda.memory_allocated(a.device),
                        cuda_reserved_bytes=torch.cuda.memory_reserved(a.device),
                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device))
                write(out/'status.json',progress);print(json.dumps(progress),flush=True)
                if segment_out is not None:write(segment_out/'status.json',progress)
                if (progress_out or out.parent)!=out:write((progress_out or out.parent)/'status.json',progress)
            if (i+1)%3000==0 or i+1==steps:
                if step_guard:
                    from .caregiver_fit_workers import CopyResourcesUnavailable
                    try:step_guard(model,i+1)
                    except CopyResourcesUnavailable as error:
                        pause_at(i+1,error)
                        raise
                checkpoint=save_at(i+1)
        recorded=src.env.caregiver.state['events']
        if hasattr(recorded,'manifest'):
            src.env.caregiver.close_recording();adult_tape=str(recorded.manifest.resolve())
        else:
            src.env.caregiver.export_events(out/'adult.json');adult_tape=str((out/'adult.json').resolve())
        changed=fingerprint(model.encoder_registry)!=initial
        if not changed:raise ValueError('copied sensory encoder weights did not learn')
        write(out/'fit.json',dict(checkpoint=str(checkpoint),simulated_seconds=frames[0].sim_time,
            online_updates=steps,experience_seconds=steps*src.env.dt*a.action_repeat,
            adult_tape=adult_tape,offline_replay_updates=0,source_utterances=utterances,
            encoder_weights_changed=changed,copy_started_at=copy_started,copy_finished_at=time.time(),
            copy_wall_seconds=prior_wall+time.time()-copy_started,
            completed_before_segment=completed_before,continuation_contract=continuation.CONTRACT))
        return checkpoint
    finally:src.close()


def fixture(a,out,seed,dev,warm,steps,kind):
    """Build a fixed physical adult/object trajectory before any copied fitting."""
    src=source(a,out/'scene',seed,dev['phase_name'],isolate=True);e=src.env;c=e.caregiver
    try:
        src.reset();c.mode='replay';c.replay=[];c.state['events']=[]
        eye=e.data.cam_xpos[e.model.camera('eye_left').id].copy()
        forward=-e.data.cam_xmat[e.model.camera('eye_left').id].reshape(3,3)[:,2].copy()
        right=e.data.cam_xmat[e.model.camera('eye_left').id].reshape(3,3)[:,0].copy()
        head=eye+clear_approach(e,c,eye)
        object_center=eye+forward*.25
        c.state['style']=2;c.state['adult_phase']='shared';c.state['adult_visibility']=1.
        tape=[];t0=warm*a.action_repeat*e.dt
        for n in range((warm+steps)*a.action_repeat):
            now=(n+1)*e.dt;t=max(0.,now-t0)
            point=object_center+right*(.045*np.sin(t*2.) if kind.startswith('follow') else .04)
            e.data.mocap_pos[c.ids['caregiver_head']]=head;e.data.mocap_pos[e._target_id]=point
            c.state['anchor']=eye
            c.state['adult_visibility']=float(kind=='return_visible' and now>=t0)
            pose_adult(e,c)
            if n==warm*a.action_repeat and kind in ('return_visible','cue_correct','cue_wrong'):
                group='return' if kind=='return_visible' else ('present_ring' if kind=='cue_correct' else 'present_ball')
                key=GROUP_KEYS[group][-1]
                e._audio.speech.start(c.bank.clips[key],e._audio.speech.position,key)
                tape.append(dict(time=now,phrase=key,speech_group=group))
                pose_adult(e,c)
            record=dict(time=now,target_actual=point.tolist(),head_actual=head.tolist(),
                infant_head_actual=eye.tolist(),style=2,visible=True,object_visibility=float(kind!='follow_hidden'),
                mobile_visibility=0.,scene_phase='engaged',**pose_record(e,c))
            tape.append(record)
            e._audio.speech.cursor=min(len(e._audio.speech.clip),e._audio.speech.cursor+int(e.dt*16000))
        write(out/(kind+'.json'),tape)
        return out/(kind+'.json')
    finally:src.close()


def probe(a,checkpoint,out,seed,dev,tape,warm,steps,kind):
    model,step0=learner(a,checkpoint,out,False)
    src=source(a,out/'scene',seed,dev['phase_name'],tape,isolate=True)
    try:
        frames=src.reset();orders=Orders(a,seed);scores=[];hits=[]
        for i in range(warm+steps):
            result=model.step(frames,dict(dev,exploration_noise=0.),train=False,step_index=step0+i,
                              executed_action=orders.executed,executed_order_id=orders.order or None)
            frames=orders.advance(model,result,src,i,False)
            if i>=warm:
                e=src.env;cid=e.model.camera('eye_left').id
                target=e.data.mocap_pos[e.caregiver.ids['caregiver_head']] if kind.startswith('return') else e.data.mocap_pos[e._target_id]
                score=alignment(e.data.cam_xpos[cid],-e.data.cam_xmat[cid].reshape(3,3)[:,2],target)
                scores.append(score);hits.append(score>=(1+np.cos(np.radians(15)))/2)
        result=dict(kind=kind,mean_alignment=float(np.mean(scores)),aligned_fraction=float(np.mean(hits)),samples=len(scores),
            scale='alignment: 0=looking opposite the target, 1=directly toward it; fraction=within 15 degrees',
            gaze_reflex=False,spontaneous_saccades=False,optimizer_updates=0,
            interpretation='Orientation response; this alone does not establish intention or word understanding.')
        write(out/'result.json',result);return result
    finally:src.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--speech',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--worlds',type=int,default=12)
    p.add_argument('--train-steps',type=int,default=9000);p.add_argument('--probe-steps',type=int,default=100)
    p.add_argument('--seed',type=int,default=24031);p.add_argument('--device',default='cpu')
    p.add_argument('--fit-workers',type=int,choices=(1,2),default=1,
        help='Independent fitting processes; original scientific declaration and device remain unchanged')
    p.add_argument('--fit-gpu-reserve-mib',type=int,default=1024)
    p.add_argument('--fit-worker-budget-mib',type=int,default=1024)
    p.add_argument('--fit-ram-reserve-gib',type=float,default=4.)
    p.add_argument('--smoke',action='store_true',help='Check instrument execution only; cannot establish learning benefit')
    p.add_argument('--resume',action='store_true',help='Reuse completed fits/fixtures and exact new-format partials; preserve older incompatible partials in their original directories')
    p.add_argument('--resume-check',action='store_true',help='Validate the existing experiment and report reusable work without launching copied fitting')
    args=p.parse_args(argv)
    if args.resume_check and not args.resume:p.error('--resume-check requires --resume')
    if args.smoke:args.worlds=2;args.train_steps=8;args.probe_steps=4
    if args.worlds<2 or args.worlds>24 or args.worlds%2 or min(args.train_steps,args.probe_steps)<1:p.error('invalid bounded trial size')
    if args.fit_gpu_reserve_mib<512 or args.fit_worker_budget_mib<512 or not np.isfinite(args.fit_ram_reserve_gib) or args.fit_ram_reserve_gib<2:
        p.error('copied worker resource reserve is too small')
    args.out.mkdir(parents=True,exist_ok=args.resume)
    if args.resume_check:return run_study(args)
    from .caregiver_fit_workers import study_ownership
    with study_ownership(args.out):return run_study(args)


def run_study(args):
    torch.set_num_threads(1)
    a=settings(args.checkpoint,args.out,args.speech,args.device,args.seed)
    # Read limits without constructing an extra trained copy.
    from types import SimpleNamespace
    dev,origin=_limits_for_run(str(args.checkpoint),SimpleNamespace(dev_limits=lambda _: {}))
    if not dev.get('phase_name'):raise ValueError('trial needs located parent developmental limits')
    warm=4 if args.smoke else int(a.context_length);donors=donor_map(args.worlds)
    conditions=['follow_visible','follow_hidden','return_visible','return_hidden','cue_correct','cue_wrong']
    declaration=dict(parent=str(args.checkpoint.resolve()),instrument_smoke=args.smoke,worlds=args.worlds,seeds=list(range(args.seed,args.seed+args.worlds)),
        train_steps=args.train_steps,experience_seconds_per_learner=args.train_steps*.01*a.action_repeat,
        probe_steps=args.probe_steps,warm_steps=warm,donors=donors,device=args.device,limits=dev,limits_source=origin,
        primary='Responsive-minus-yoked improvement in visible-minus-absent object following.',
        primary_gate='Positive primary difference and two-sided sign permutation p <= .05 across at least six independent TWO-WORLD BLOCKS. No live policy promotion.',
        secondary=['return response','congruent-versus-wrong word cue effect'],
        retention='Frozen probes after a new episode and a full observation context; not long-duration retention.',
        controls='Within each disjoint two-world block, donor tapes swap. Block-total source words/times are identical. Statistics average each block before testing; coupled worlds are not treated as independent. Each probe condition starts a fresh checkpoint load.',
        fitting='Canonical production learner and optimizer; native motor cadence and scheduled holds; equal online updates. Offline minibatch replay omitted in both copied arms.',
        scope='A bounded exposure comparison, not a test of the full eventual learning capacity.',
        stop='Stop this experiment on nonfinite values, missing learned state, replay-dose mismatch or failed controls. Production keeps running.')
    root=Path(__file__).resolve().parents[4]
    paths=list((root/'Source/Python/mimo_bridge').glob('*.py'))+[Path(__file__),
        Path(__file__).with_name('caregiver_fit_workers.py'),Path(__file__).with_name('caregiver_fit_continuation.py')]
    hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    operations=dict(fit_workers=args.fit_workers,gpu_reserve_mib=args.fit_gpu_reserve_mib,
        worker_budget_mib=args.fit_worker_budget_mib,ram_reserve_gib=args.fit_ram_reserve_gib)
    if args.resume:record_resume_version(args.out,hashes,declaration,operations)
    else:
        write(args.out/'declaration.json',declaration);write(args.out/'source_hashes.json',hashes)
    if args.resume_check:
        validate_checkpoint(args.checkpoint)
        fits={};incomplete=[]
        for arm in ('responsive','yoked'):
            for i in range(args.worlds):
                base=args.out/f'{arm}_{i}';done=completed_fit(base,args.train_steps)
                if done:fits[f'{arm}_{i}']=str(done[0]/'fit.json')
                else:incomplete.append(dict(arm=arm,world=i,next_attempt=str(next_attempt(base))))
        fixtures=[args.out/f'fixtures_{i}'/(kind+'.json') for i in range(args.worlds) for kind in conditions]
        for path in fixtures:
            if path.exists() and not isinstance(json.loads(path.read_text()),list):raise ValueError('invalid preserved fixture')
        report=dict(complete_fits=fits,incomplete_fits=incomplete,fixtures_present=sum(path.exists() for path in fixtures),
                    original_declaration_unchanged=True,production_untouched=True,checked_at=time.time())
        write(args.out/'resume_readiness.json',report);print(json.dumps(report),flush=True);return 0
    write(args.out/'status.json',dict(stage='building_fixed_probes',timestamp=time.time(),pid=__import__('os').getpid()))
    # Instrument limits and null differences must be correct before fitting.
    assert alignment(np.zeros(3),np.array([0,0,1]),np.array([0,0,1]))==1
    assert alignment(np.zeros(3),np.array([0,0,-1]),np.array([0,0,1]))==0
    assert paired_result([0.]*args.worlds)['paired_permutation_p']==1
    fixtures={}
    for i in range(args.worlds):
        folder=args.out/f'fixtures_{i}';folder.mkdir(exist_ok=args.resume)
        fixtures[i]={}
        for kind in conditions:
            path=folder/(kind+'.json')
            if path.exists():
                data=json.loads(path.read_text())
                if not isinstance(data,list) or not data:raise ValueError('existing fixed probe is incomplete')
                fixtures[i][kind]=path
            else:fixtures[i][kind]=fixture(a,folder,args.seed+i,dev,warm,args.probe_steps,kind)
    # Both one- and two-worker runs use the same resource guards and durable
    # continuation. Changing concurrency never changes a learner's experience.
    from .caregiver_fit_workers import run_fits
    checkpoints,fit_paths=run_fits(args,dev,donors)
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.25,device=args.device)
    results={}
    for arm in ('responsive','yoked'):
        results[arm]=[]
        for i in range(args.worlds):
            values={}
            for kind in conditions:
                out=fit_paths[arm,i]/('probe_'+kind)
                a=settings(args.checkpoint,out,args.speech,args.device,args.seed+i)
                reused=completed_probe(out,kind,args.probe_steps)
                if reused:
                    out,values[kind]=reused
                else:
                    if out.exists():out=next_attempt(out)
                    values[kind]=probe(a,checkpoints[arm,i],out,args.seed+i,dev,fixtures[i][kind],warm,args.probe_steps,kind)
                write(args.out/'status.json',dict(stage='probe',arm=arm,world=i,kind=kind,timestamp=time.time(),pid=__import__('os').getpid()))
                print(json.dumps(dict(stage='probe',arm=arm,world=i,**values[kind])),flush=True)
            results[arm].append(values)
    comparisons={}
    for metric,present,absent in [('following','follow_visible','follow_hidden'),('return','return_visible','return_hidden'),('cue','cue_correct','cue_wrong')]:
        values=[]
        for i in range(args.worlds):
            contrasts=[results[arm][i][present]['mean_alignment']-results[arm][i][absent]['mean_alignment'] for arm in ('responsive','yoked')]
            values.append(contrasts[0]-contrasts[1])
        blocks=[float(np.mean(values[i:i+2])) for i in range(0,len(values),2)]
        comparisons[metric]=dict(world_differences=values,independent_blocks=blocks,**paired_result(blocks))
    primary=comparisons['following']
    passed=not args.smoke and args.worlds>=12 and primary['valid'] and primary['mean']>0 and primary['paired_permutation_p']<=.05
    report=dict(verdict='INSTRUMENT_EXECUTED' if args.smoke else ('FOLLOWING_BENEFIT_DETECTED' if passed else 'BENEFIT_NOT_ESTABLISHED'),
        comparisons=comparisons,source_speech_dose_matched=True,results=results,
        interpretation='Applies to this exposure and these copied learners; no intention or language milestone is certified.')
    write(args.out/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
    write(args.out/'status.json',dict(stage='complete',verdict=report['verdict'],timestamp=time.time()))


if __name__=='__main__':
    try:main()
    except Exception as error:
        import sys,traceback
        if '--out' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--out')+1])
            if out.exists():
                audit='--resume-check' in sys.argv
                write(out/('resume_check_failure.json' if audit else 'failure.json'),
                      dict(error=str(error),traceback=traceback.format_exc(),timestamp=time.time()))
                if not audit:write(out/'status.json',dict(stage='failed',error=str(error),timestamp=time.time()))
        raise
