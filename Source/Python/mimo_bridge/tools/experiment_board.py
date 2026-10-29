# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Current experiments and decisions, read from registered run artifacts."""
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.parse import quote


def read(path):
    def invalid(x):raise ValueError('Nonfinite experiment record')
    def finite_float(x):
        value=float(x)
        if not math.isfinite(value):raise ValueError('Nonfinite experiment record')
        return value
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=invalid,parse_float=finite_float)


def registry(root):
    path=Path(root)/'docs/development_dashboard/work.json'
    config=read(path) if path.exists() else {'schema':1,'jobs':[]}
    manifest=Path(root)/'docs/development_dashboard/queue.json'
    state_path=Path(root)/'Saved/mimo_bridge/developmental_queue.json'
    if manifest.is_file():
        specification=read(manifest)
        state=read(state_path) if state_path.is_file() else {'jobs':{}}
        config['queue_pause_reason']=state.get('pause_reason')
        config['resources']=state.get('resources')
        queue_written=time.strftime('%Y-%m-%d %H:%M',time.localtime(state_path.stat().st_mtime)) if state_path.is_file() else None
        for job in [*specification['jobs'],*state.get('generated_jobs',[])]:
            row=state['jobs'].get(job['id'],{})
            item={k:job[k] for k in ('id','title','question','decision','scope')}
            item['abilities']=job.get('abilities',[])
            if job.get('checkpoint'):item['checkpoint']=job['checkpoint']
            if job['id'] in config.get('reviews',{}):item['review']=config['reviews'][job['id']]
            if job['id'] in ('toy-interventions-20260913','body-only-movement-20260913'):
                item['limitation']='Earlier instrument version carried body-schema evidence between conditions. Individual physical observations remain recorded; the comparison is invalid and a corrected version is scheduled.'
            if not item['abilities'] and job['module']=='mimo_bridge.tools.developmental_probe':
                arguments=job['args']
                profile=arguments[arguments.index('--profile')+1] if '--profile' in arguments else None
                item['abilities']=(['gaze_control','head_lift_and_look'] if profile=='toy' else
                    ['weight_shift','head_control','roll_over','prone_prop'] if profile=='body_floor' else [])
            item.update(owner='Automatic behavior tests',kind='developmental_probe',
                planned_state=row.get('state','queued'),
                when='Runs after '+', '.join(row['waiting_for']) if row.get('waiting_for') else
                     'Waits for the developmental queue runner'+(f' (its state file was last written {queue_written})' if queue_written else '')+'.',
                progress_label='Independent worlds measured')
            if row.get('state')=='cancelled':item['cancel_reason']=row.get('cancel_reason')
            if row.get('attempts'):item['run_dir']=row['attempts'][-1]['out']
            if row.get('state')=='repair_required':item['queue_issue']='Failed on every allowed attempt; the queue marked it for repair.'
            if job['id'] in config.get('superseded_jobs',{}):item['superseded_by']=config['superseded_jobs'][job['id']]
            config['jobs']=[existing for existing in config['jobs'] if existing['id']!=item['id']]
            config['jobs'].append(item)
    return config


def directory(root,job):
    path=Path(job['run_dir'])
    return path.resolve() if path.is_absolute() else (Path(root)/path).resolve()


def artifact(root,job_id,name):
    # Only explicit public records of registered jobs can leave the repo root.
    if name not in ('status.json','report.json','declaration.json','index.html'):
        raise ValueError('Not an experiment public record')
    jobs=registry(root)['jobs'];job=next((j for j in jobs if j['id']==job_id),None)
    if job is None or not job.get('run_dir'):raise ValueError('Unknown experiment')
    folder=directory(root,job);path=(folder/name).resolve()
    if path.parent!=folder:raise ValueError('Experiment record leaves its directory')
    return path


def process_alive(pid,started):
    if not pid:return None
    try:
        import psutil
        p=psutil.Process(int(pid))
        return p.is_running() and (not started or p.create_time()<=started+120)
    except (psutil.Error,ValueError,TypeError):return False


def caregiver_progress(folder,declaration,status,started):
    """Counts of finished copied learners and behavior tests. No finish time is forecast:
    the style guide bans time estimates (operator 2026-09-08)."""
    count=int(declaration['worlds']);steps=int(declaration['train_steps']);done=0;complete=0;probes=0
    folder=Path(folder).resolve()
    index=read(folder/'completed_fits.json') if (folder/'completed_fits.json').is_file() else {}
    for arm in ('responsive','yoked'):
        for i in range(count):
            base=folder/f'{arm}_{i}';target=base;fit=target/'fit.json'
            if base.name in index:
                fit=Path(index[base.name]).resolve()
                if not fit.is_relative_to(base) or fit.name!='fit.json':
                    raise ValueError('Completed learner record leaves its registered arm')
                target=fit.parent
            if fit.exists():
                record=read(fit);done+=min(steps,int(record['online_updates']));complete+=1
                duration=record.get('copy_wall_seconds')
                if duration is not None and (not math.isfinite(duration) or duration<=0):
                    raise ValueError('Invalid copied training duration')
            elif status.get('stage') in ('fit','fit_parallel'):
                active_rows=status.get('workers',[]) if status['stage']=='fit_parallel' else [status]
                candidates=[]
                for current in active_rows:
                    active=current.get('arm')==base.name
                    attempt=(base/str(current.get('arm',''))/'attempt.json').resolve()
                    if attempt.is_relative_to(base) and attempt.is_file():
                        receipt=read(attempt)
                        seeds=declaration.get('seeds',[])
                        active=(receipt.get('arm')==arm and i<len(seeds) and
                                receipt.get('world_seed')==seeds[i]==current.get('seed'))
                    if active:candidates.append(max(0,min(steps,int(current.get('step',0)))))
                if len(candidates)>1:raise ValueError('Multiple active workers for one copied learner')
                if candidates:done+=candidates[0]
            probes+=sum((target/('probe_'+kind)/'result.json').exists() for kind in
                        ('follow_visible','follow_hidden','return_visible','return_hidden','cue_correct','cue_wrong'))
    total=2*count*steps;probe_total=2*count*6
    progress=dict(label='Copied learning updates',done=done,total=total,
        display=f'{complete} / {2*count} copied learners finished',
        detail=f'{complete} of {2*count} copied learners finished; {probes} of {probe_total} behavior tests finished.',
        phases=[dict(name='Prepare matched tests',state='complete'),dict(name='Learn in copied worlds',state='complete' if done==total else 'waiting' if status.get('stage')=='waiting_for_resources' else 'running'),
                dict(name='Test object following',state='complete' if probes==probe_total else ('running' if probes else 'waiting'))])
    return progress


def causal_readout_progress(declaration,status):
    """Display recorded work units; never turn a fit count into an ability pass."""
    stage=status['stage']
    total=(len(declaration['train_worlds'])+len(declaration['test_worlds']))*declaration['states_per_world']
    if not isinstance(total,int) or total<=0:raise ValueError('Invalid causal capture budget')
    if stage=='recording_raw_causal_windows':
        done=status['decisions_saved'];label='Matched movement situations recorded'
        detail='Each situation includes four commands and repeated-hold verification. Fitting and untouched-world tests follow.'
    elif stage=='encoding_own_raw_windows':
        done=status['complete'];label='Recorded sensory windows encoded'
        if status['total']!=total:raise ValueError('Causal encoding budget changed')
        detail='Using the saved brain\'s own sensory encoding. The separate predictor has not finished fitting.'
    elif stage=='fitting_isolated_auxiliary':
        labels=['action_conditioned_stage_1','action_conditioned_stage_2','action_blind_stage_1','action_blind_stage_2']
        done=labels.index(status['condition'])+1;total=len(labels);label='Separate predictor fits completed'
        detail='Two learning stages with commands and two without commands. Physical-outcome comparisons determine the result.'
    else:
        return None
    if isinstance(done,bool) or not isinstance(done,int) or not 0<=done<=total:
        raise ValueError('Invalid causal study progress')
    return dict(label=label,done=done,total=total,display=f'{done} / {total}',detail=detail)


def job_lineage(root,job,bounds):
    """Whether the job's tested checkpoint is on the running model's ancestry.

    'ancestor_checkpoint' or 'not_linked_to_current_model' when a checkpoint is recorded
    (in the job or its declaration.json); 'unknown' when neither records one."""
    checkpoint=job.get('checkpoint')
    if not checkpoint and job.get('run_dir'):
        try:checkpoint=read(directory(root,job)/'declaration.json').get('checkpoint')
        except (OSError,ValueError,AttributeError):checkpoint=None
    if not checkpoint:
        return 'unknown'
    if bounds is None:
        return 'not_linked_to_current_model'
    from .development_dashboard import lineage_relation
    return lineage_relation(Path(root),checkpoint,bounds)


def job_state(root,job,now):
    result={k:v for k,v in job.items() if k not in ('run_dir','launch','review')}
    result.update(state='not_started',updated_at=None,started_at=None,links=[])
    if job.get('planned_state')=='cancelled':
        result.update(state='cancelled',result=job.get('cancel_reason','Superseded measurement; earlier artifacts retained.'))
        return result
    if not job.get('run_dir'):
        planned=job.get('planned_state','not_started')
        # an explicit null in work.json is an unrecorded state, not "not started"
        result['state']=planned if planned else 'no_state_recorded';return result
    folder=directory(root,job);status_path=folder/'status.json'
    try:
        launch=read(Path(root)/job['launch']) if job.get('launch') else {}
        result['started_at']=launch.get('started_at',launch.get('started'))
    except (OSError,ValueError):launch={}
    if not folder.exists():
        # A registered folder that is gone (the D: folders of 2026-09-12/13) is not a stopped
        # or unstarted job: its results existed and are missing.
        if launch and process_alive(launch.get('pid'),result['started_at']):
            result['state']='starting'
        else:
            result.update(state='files_missing',failure=f'result files missing ({folder})',missing_path=str(folder))
        return result
    if not status_path.exists():
        if launch:
            result['state']='starting' if process_alive(launch.get('pid'),result['started_at']) else 'stopped'
        return result
    try:
        status=read(status_path);result['updated_at']=float(status['timestamp'])
        if not math.isfinite(result['updated_at']):raise ValueError('Invalid status time')
        stage=status['stage'];result['stage']=stage
        result['state']='completed_unreviewed' if stage=='complete' else 'failed' if stage=='failed' else 'running'
        if stage=='waiting_for_resources':result.update(state='waiting_for_resources',result=status.get('waiting_reason','Waiting for production resource headroom'))
        result['reported_verdict']=status.get('verdict');result['failure']=status.get('error')
        for name in ('declaration.json','report.json','status.json'):
            if (folder/name).is_file():result['links'].append(dict(label={'declaration.json':'Test design','report.json':'Full result','status.json':'Live record'}[name],url=f'/experiments/{quote(job["id"])}/{name}'))
        if stage=='complete' and not (folder/'report.json').exists():result['state']='result_missing'
        if stage not in ('complete','failed'):
            alive=process_alive(status.get('pid') or launch.get('pid'),result['started_at'])
            if alive is False:result['state']='stopped'
            elif now-result['updated_at']>300:result['state']='status_stale'
        if stage=='failed' and not (folder/'declaration.json').is_file():
            result['failure']=status.get('error','Experiment failed before initialization.')
            return result
        declaration=read(folder/'declaration.json')
        if job.get('kind')=='caregiver':result['progress']=caregiver_progress(folder,declaration,status,result['started_at'])
        elif job.get('kind')=='causal_readout':
            progress=causal_readout_progress(declaration,status)
            if progress is not None:result['progress']=progress
        elif job.get('kind')=='developmental_probe' and 'world' in status:
            seeds=declaration.get('seeds',[]);conditions=(['stationary','relocated','absent','interrupted','hold_control']
                if not declaration.get('probe_version') else declaration.get('conditions',[]))
            if declaration.get('profile')=='body_floor':conditions=['body_only','hold_control']
            result['progress']=dict(label='Worlds and frames',done=seeds.index(status['world']) if status['world'] in seeds else 0,
                total=len(seeds),display=f"World {seeds.index(status['world'])+1 if status['world'] in seeds else '?'} / {len(seeds)}",
                detail=f"{status.get('condition','Preparing world').replace('_',' ')}: frame {status.get('frame',0)} of {status.get('frames',declaration.get('steps',0))}. Each world includes separate controls.")
        elif 'step' in status and 'steps' in status:
            result['progress']=dict(label=job.get('progress_label','Measurements finished'),done=status['step'],total=status['steps'],detail=job.get('work_units',''))
        if stage=='complete' and (folder/'report.json').exists():
            report=read(folder/'report.json')  # malformed/NaN records cannot be completed evidence
            result['completed_at']=result['updated_at']
            if job.get('kind')=='early_probe':
                labels=dict(follow_visible='Moving toy',follow_hidden='Toy absent',return_visible='Adult returns',
                            return_hidden='Adult absent',cue_correct='Matching word',cue_wrong='Wrong word')
                observations=[]
                for key,label in labels.items():
                    row=report['results'][key];value=row['aligned_fraction']
                    if row['samples']<=0 or not 0<=value<=1:raise ValueError('Invalid behavior measurement')
                    observations.append(dict(label=label,value=value,samples=row['samples']))
                result['observations']=observations
                result['observation_scale']='Share of measured model steps facing within 15 degrees of the target. One frozen copied learner; no learning-benefit verdict.'
            elif job.get('kind')=='developmental_probe' and 'modalities' in report:
                observations=[]
                for name,label in (('proprioception','Proprioception'),('tactile','Tactile'),('vision','Vision')):
                    row=report['modalities'][name];values=row['world_values']['action_effect_error_removed']
                    if not values or len(values)!=row['worlds']:raise ValueError('Incomplete sensory-forecast observations')
                    value=sum(values)/len(values)
                    observations.append(dict(label=label,value=value,
                        display=f'{100*abs(value):.2f}% '+('less' if value>=0 else 'more')+' error than predicting no action effect'))
                result['observations']=observations
                result['observation_scale']='Error in predicted differences between moves, averaged across independent worlds. These are next-model-step forecasts; an error reduction alone does not pass the comparison or establish an ability.'
            elif job.get('kind')=='developmental_probe' and 'results' in report:
                observations=[]
                for world in report['results']:
                    for row in world.get('conditions',[]):
                        if declaration.get('profile') in ('body_floor','free_objects','support'):
                            arm='Model control' if row.get('arm')=='candidate' else 'Pose-hold control' if row.get('arm')=='baseline' else row['condition'].replace('_',' ')
                            tilt=math.degrees(math.acos(max(-1,min(1,row['minimum_front_up']))))
                            observations.extend([
                                dict(label=f"World {world['seed']} · {arm} · torso tilt from back",value=tilt,display=f'{tilt:.1f}°'),
                                dict(label=f"World {world['seed']} · {arm} · greatest horizontal displacement",value=row['maximum_horizontal_displacement_m'],
                                     display=f"{100*row['maximum_horizontal_displacement_m']:.1f} cm")])
                            continue
                        value=row['target_center_in_fovea_fraction']
                        if not 0<=value<=1:raise ValueError('Invalid foveal measurement')
                        arm='model' if row.get('arm')=='candidate' else 'pose hold' if row.get('arm')=='baseline' else ''
                        observations.append(dict(label=f"World {world['seed']} · {row['condition'].replace('_',' ')} · {arm}",
                            value=value,samples=None))
                result['observations']=observations
                result['observation_scale']='Share of scored model steps with the target center inside the actual fovea. Model and pose-hold controls are separate. An absent target always scores zero here; that zero does not show purposeful stopping.'
                if declaration.get('profile') in ('body_floor','free_objects','support'):
                    result['observation_scale']='Physical changes during the recorded window. Tilt can include pitch; displacement alone does not prove purposeful travel or crawling.'
            if job.get('review'):
                reviewed=read(Path(root)/job['review'])
                if hashlib.sha256((folder/'report.json').read_bytes()).hexdigest()!=reviewed['report_sha256']:
                    result['review_issue']='The result changed after review.'
                else:
                    result.update(state=reviewed['state'],result=reviewed['summary'],decision=reviewed['decision'],measurement=reviewed.get('measurement'))
                    result['links'].append(dict(label='Assessment',url='/files/'+quote(job['review'],safe='/')))
    except (OSError,ValueError,TypeError,KeyError) as error:
        result['state']='record_error';result['failure']=str(error)
    return result


def build(root,now=None,bounds=None):
    now=time.time() if now is None else now
    try:
        config=registry(root)
        if config.get('schema')!=1:raise ValueError('Unknown work board schema')
    except (OSError,ValueError) as error:
        return dict(schema=1,jobs=[],error=str(error))
    jobs=[]
    for spec in config['jobs']:
        try:
            state=job_state(root,spec,now)
            state['lineage']=job_lineage(root,spec,bounds)
            jobs.append(state)
        except (OSError,ValueError,TypeError,KeyError) as error:
            jobs.append(dict(id=spec.get('id','unknown'),title=spec.get('title','Experiment record'),
                             state='record_error',failure=str(error),links=[],lineage='unknown'))
    return {**{k:v for k,v in config.items() if k!='jobs'},'jobs':jobs}
