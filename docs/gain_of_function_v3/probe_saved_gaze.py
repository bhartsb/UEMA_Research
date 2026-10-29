"""Recheck learned eye consequences on all eight saved worlds, with exact replay.

This retains the qualified gaze probe's declared commands and prediction routes.
The auditor chooses left/right/up/down; this is not a test of useful looking goals.
Every command and the eye-hold comparison start from the same complete state.
How each body's start was chosen is declared (START_RULE) and reported per body (starts).
"""
from __future__ import annotations

import os
if __name__ == "__main__":  # this probe runs on the CPU; importing the probe harness no longer hides the card
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # (2026-09-26, probe_gpu_hiding_2026-09-26.md)
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def write(path,value):
    from mimo_bridge.tools.gof_probe_harness import _json_default
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=_json_default),encoding='utf-8')


def digest(value):
    """Hash typed arrays and named fields, without rounding physical trajectories."""
    import torch
    result=hashlib.sha256()
    def add(item):
        if torch.is_tensor(item):item=item.detach().cpu().numpy()
        if isinstance(item,np.ndarray):
            if not np.isfinite(item).all():raise ValueError('Nonfinite replay observation')
            result.update(str((item.dtype.str,item.shape)).encode());result.update(item.tobytes())
        elif isinstance(item,dict):
            for key in sorted(item):result.update(str(key).encode());add(item[key])
        elif isinstance(item,(list,tuple)):
            result.update(str(len(item)).encode())
            for entry in item:add(entry)
        else:result.update(json.dumps(item,allow_nan=False).encode())
    add(value)
    return result.hexdigest()


def first_difference(a,b,path=''):
    """The first leaf where two nested run records differ, as (path, what differs), else None."""
    import torch
    if torch.is_tensor(a):a=a.detach().cpu().numpy()
    if torch.is_tensor(b):b=b.detach().cpu().numpy()
    if isinstance(a,dict):
        if not isinstance(b,dict) or set(a)!=set(b):return path,dict(kind='different fields')
        for key in sorted(a,key=str):
            found=first_difference(a[key],b[key],f'{path}.{key}')
            if found:return found
        return None
    if isinstance(a,(list,tuple)):
        if not isinstance(b,(list,tuple)) or len(a)!=len(b):return path,dict(kind='different length')
        for i,(x,y) in enumerate(zip(a,b)):
            found=first_difference(x,y,f'{path}[{i}]')
            if found:return found
        return None
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        x,y=np.asarray(a),np.asarray(b)
        if x.shape!=y.shape:return path,dict(kind='different shape',first=list(x.shape),second=list(y.shape))
        if np.array_equal(x,y):return None
        found=dict(kind='different values',values=int(x.size),differing=int(np.sum(x!=y)))
        if x.dtype.kind in 'biuf' and y.dtype.kind in 'biuf':
            gap=np.abs(x.astype(np.float64)-y.astype(np.float64))
            found.update(largest_difference=float(gap.max()),largest_value=float(np.abs(x.astype(np.float64)).max()))
        return path,found
    return None if a==b else (path,dict(kind='different values',first=repr(a)[:200],second=repr(b)[:200]))


def sensed(session):
    """Fingerprints of every sense's input as the world produced it in one physics step, so a
    replay mismatch can be traced to the input that moved (pictures, touch, hearing, balance,
    body position, the task report)."""
    frame=session.frames[0]
    def mark(value):
        if value is None:return None
        array=np.ascontiguousarray(np.asarray(value))
        return hashlib.sha256(str((array.dtype.str,array.shape)).encode()+array.tobytes()).hexdigest()[:16]
    task=json.dumps(frame.meta.get('task') or {},sort_keys=True,default=lambda o:np.asarray(o).tolist())
    fields=dict(pictures='vision_patches',touch='patches',touch_self='touch_self',touch_contact='touch_contact',
                balance='vestibular',body='bones')
    found={sense:mark(getattr(frame,name,None)) for sense,name in fields.items()}
    return dict(found,hearing=mark(getattr(getattr(session.env,'_audio',None),'last_spec01',None)),
                task=hashlib.sha256(task.encode()).hexdigest()[:16])


def paired_body_summary(candidate,comparison):
    a,b=np.asarray(candidate,dtype=float),np.asarray(comparison,dtype=float)
    if a.shape!=(8,) or b.shape!=(8,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Eight finite paired body means are required')
    difference=a-b
    draws=np.random.default_rng(94121).integers(8,size=(10000,8))
    interval=np.quantile(difference[draws].mean(-1),[.025,.975])
    return dict(bodies=8,candidate_mean=float(a.mean()),comparison_mean=float(b.mean()),
                candidate_minus_comparison=float(difference.mean()),paired95_difference=interval.tolist(),
                candidate_error_lower_beyond_sample_uncertainty=bool(interval[1]<0),
                error_reduction_fraction=None if b.mean()<=0 else float(1-a.mean()/b.mean()))


def mean_eye_angles(env):
    """Match the gaze lesson's two-eye mean, in the simulator's signed convention."""
    values=np.asarray(env.eye_yaw_pitch(),dtype=np.float64)
    if values.shape!=(2,2) or not np.isfinite(values).all():raise ValueError('Both physical eye angles are required')
    return values.mean(axis=0)


def summarize(counterfactual):
    groups={'saved_starts':counterfactual['states'],
            'after_head_command':[entry.get('scored') for entry in counterfactual['generalization']]}
    summary={}
    for name,states in groups.items():
        if len(states)!=8 or any(state is None for state in states):
            raise ValueError('All eight bodies must complete both gaze conditions')
        for state in states:
            if set(state['arms'])!={'hold','left','right','up','down'}:
                raise ValueError('A gaze command or hold comparison is missing')
        result={}
        for k,milliseconds in enumerate((50,100,150)):
            eye_pred,eye_hold,visual_pred,visual_ablation=[],[],[],[]
            movement=[]
            for state in states:
                p,h,v,a,m=[],[],[],[],[]
                hold=np.asarray(state['arms']['hold']['eye_change_deg'][k])
                for direction in ('left','right','up','down'):
                    arm=state['arms'][direction]
                    actual=np.asarray(arm['eye_change_deg'][k],dtype=float)
                    predicted=np.asarray(arm['efference_gaze_deg'][k],dtype=float)
                    p.append(np.linalg.norm(predicted-actual));h.append(np.linalg.norm(actual))
                    m.append(np.linalg.norm(actual-hold))
                    route=f'{milliseconds}ms'
                    v.append(arm['imagined'][route]['vision']['pred_err'])
                    a.append(arm['imagined'][route+'_ablated']['vision']['pred_err'])
                eye_pred.append(np.mean(p));eye_hold.append(np.mean(h));movement.append(np.mean(m))
                visual_pred.append(np.mean(v));visual_ablation.append(np.mean(a))
            result[f'{milliseconds}ms']=dict(
                eye_prediction_error_degrees=paired_body_summary(eye_pred,eye_hold),
                visual_prediction_mse=paired_body_summary(visual_pred,visual_ablation),
                actual_eye_change_vs_eye_hold_degrees=float(np.mean(movement)))
        summary[name]=result
    summary['definitions']=dict(
        eye_prediction='Euclidean error in the two eyes\' mean azimuth/elevation, in degrees, matching the training target; comparison predicts zero eye movement.',
        visual_prediction='Mean squared feature error for the actual command versus the same prediction with the gaze command replaced by eye hold. The actual visual outcome is identical in both scores.',
        uncertainty='Paired bootstrap over eight body means; four directions are averaged within each body.',
        choice='The auditor selects four eye commands, not the model. Useful goal-directed looking requires a separate choice test.')
    return summary


def audited_session(base,records,out,began):
    from probe_saved_self_touch_choices import verify_hand_readback
    class VerifiedSession(base):
        def eyes_rad(self):
            return mean_eye_angles(self.env)

        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if self.env.reflex is not None or self.env.saccades is not None:
                raise RuntimeError('Reflex or spontaneous saccades could override the gaze command')
            if self.model.hand_target_mode!='wrist_relative_v1':
                raise RuntimeError('This scheduled recheck requires the repaired hand contract')
            if abs(self.clock.model_step_seconds-.05)>1e-8:
                raise RuntimeError('The eye probe requires 50 ms model steps')

        def run_arm(self,snap,order,*,gaze_rad=None,steps=1,before_step=None,
                    hand_joint_goal=None,on_env_step=None):
            if hand_joint_goal is None:raise RuntimeError('A restored gaze branch must retain its explicit hand goal')
            runs=[];kept=[];drawn=self.pictures.counts()
            for repeat in range(2):
                trace=[];inputs=[];self.pictures.drain()   # scenes drawn before this run are not part of it
                def record(session,index):
                    meta=session.frames[0].meta
                    verify_hand_readback(meta,order,hand_joint_goal)
                    np.testing.assert_allclose(meta['gaze_cmd'],gaze_rad,rtol=0,atol=2e-7)
                    # scenes: the fingerprint of every eye picture drawn in this physics step, so
                    # both runs must hand OpenGL identical scenes (gof_probe_harness.ScenePictures)
                    trace.append(dict(qpos=session.env.data.qpos.copy(),qvel=session.env.data.qvel.copy(),
                        time=float(session.env.data.time),gaze=np.asarray(meta['gaze_cmd']),
                        both_eye_angles_rad=np.asarray(session.env.eye_yaw_pitch()),
                        joint_goal=np.asarray(meta['hand_joint_goal']).copy(),
                        actual_motor_targets=np.asarray(meta['executed_motor_targets']).copy(),
                        scenes=session.pictures.drain()))
                    inputs.append(sensed(session))
                    if on_env_step is not None:on_env_step(session,index)
                observed=super().run_arm(snap,order,gaze_rad=gaze_rad,steps=steps,before_step=before_step,
                                        hand_joint_goal=hand_joint_goal,on_env_step=record)
                if len(trace)!=steps*self.reps:raise RuntimeError('Incomplete physical trace')
                values=[dict(tokens=row['tokens'],eyes_deg=row['eyes_deg'],pos_cm=row['pos_cm'],
                             quat=row['quat'],sim_time=row['sim_time']) for row in observed]
                runs.append(digest(dict(trace=trace,observations=values)))
                kept.append(dict(trace=trace,observations=values,inputs=inputs))
            if runs[0]!=runs[1]:
                # what differed, written before stopping, so the failure is attributable from files
                one,two=kept
                write(out/'replay_mismatch.json',dict(
                    body=self.episode_body['body'],command_branch=len(records)+1,
                    gaze_rad=np.asarray(gaze_rad).tolist(),model_steps=steps,
                    first_trace_difference=first_difference(one['trace'],two['trace'],'trace'),
                    first_observation_difference=first_difference(one['observations'],two['observations'],'observations'),
                    physics_steps_whose_inputs_differ=[dict(step=i,senses=sorted(k for k in x if x[k]!=y[k]))
                        for i,(x,y) in enumerate(zip(one['inputs'],two['inputs'])) if x!=y],
                    scene_fingerprints=[[row['scenes'] for row in run['trace']] for run in kept],
                    eye_picture_draws=self.pictures.counts()))
                raise RuntimeError('Eye trial did not replay exactly')
            pictures={k:v-drawn[k] for k,v in self.pictures.counts().items()}
            records.append(dict(body=self.episode_body['body'],gaze_rad=np.asarray(gaze_rad).tolist(),
                model_steps=steps,physics_steps=steps*self.reps,sha256=runs[0],exact=True,pictures=pictures))
            write(out/'replays.json',records)
            write(out/'status.json',dict(stage='measuring',completed_arms=len(records),maximum_arms=80,
                                        elapsed_s=time.time()-began))
            print(json.dumps(dict(body=self.episode_body['body'],completed_arms=len(records),exact_replay=True)),flush=True)
            return observed
    return VerifiedSession


START_RULE=('Each body starts at the first moment, within 30 model steps after its saved moment, where the '
            'planner made a plan. A body whose planner is consulted in that window but makes no plan starts at '
            'its first planning event instead, and its row in starts says so (measured 2026-09-25 at checkpoint '
            '400,350: body 2\'s harm reading was over the integrity gate line at every remembered moment, so its '
            'goal was its present moment and its planner stopped as converged at every planning event; the body '
            'used to be dropped without a word). Only the planner-direction reading (findings: '
            'planner_toward_better_predicted_wellness) uses the plan; the eye and view comparisons do not.')


def start_records(counterfactual):
    """How each body's saved start was chosen (gof_probe_harness START_WITHOUT_A_PLAN): the body, the
    model steps it lived past its saved moment, and what the planner did at the start."""
    rows=[]
    for state in counterfactual['states']:
        planner=state.get('start_planner')
        if not isinstance(planner,dict) or not planner.get('rule'):
            raise ValueError('Every saved start must record how it was chosen')
        rows.append(dict(body=state.get('body'),model_steps_after_saved_moment=state.get('step_offset'),
                         rule=planner['rule'],planned=bool(planner.get('planned')),
                         stopped_by=planner.get('stopped_by'),goal_distance_now=planner.get('goal_distance_now'),
                         planning_events_searched=len(planner.get('search') or [])))
    return rows


def validate_complete(counterfactual,replays):
    summarize(counterfactual)
    if len(replays)!=80 or any(not row['exact'] or row['model_steps']!=3 or row['physics_steps']!=15 for row in replays):
        raise ValueError('All 80 three-step command branches must replay exactly')
    if {row['body'] for row in replays}!=set(range(8)):
        raise ValueError('The replay receipt must include all eight saved bodies')
    if any(sum(row['body']==body for row in replays)!=10 for body in range(8)):
        raise ValueError('Each body needs five commands at each of two starts')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    from mimo_bridge.tools.commit_guard import commit_reading,verdict
    reading=commit_reading()
    if reading is None or not verdict(reading[0],reading[1],12,6):raise RuntimeError(f'Memory guard refused: {reading}')
    from mimo_bridge.tools import gof_probe_gaze as G,gof_probe_harness as H
    from prove_planner_repair import source_hashes,episode_integrity
    from prove_takeover_restart import runtime_fingerprint
    from probe_saved_self_touch_choices import weights_digest
    args.out.mkdir(parents=True,exist_ok=False)
    began=time.time();runtime=runtime_fingerprint()
    scripts={str(Path(path).resolve()):sha256(path) for path in (__file__,H.__file__,G.__file__)}
    declaration=dict(checkpoint=str(args.checkpoint.resolve()),script_sha256=sha256(__file__),
        helper_sha256=scripts,runtime_sha256=runtime,source_sha256=source_hashes(args.checkpoint),
        episode=episode_integrity(args.checkpoint),bodies=8,commands=['hold','left','right','up','down'],
        starts_per_body=2,start_rule=START_RULE,command_branches=80,replays_per_branch=2,horizons_ms=[50,100,150],
        learning=False,production_adoption=False,gaze_reflex=False,spontaneous_saccades=False,
        eye_angle_target='Mean azimuth/elevation of both eyes, matching LemLite._eye_angles_deg; old probe used the left eye alone.',
        eye_pictures=('Each distinct scene is drawn once per body: a later draw of the same scene (the same '
                      'fingerprint of everything MuJoCo is asked to draw, taken so it does not depend on the '
                      'previous draw) receives the first picture, because this machine\'s graphics driver returns '
                      'different pictures for one unchanged scene when semi-transparent shapes overlap in view '
                      '(measured 2026-09-24). The fingerprints are part of the replay digest, so both runs of a '
                      'command must also ask for identical scenes. A mismatch writes replay_mismatch.json.'),
        guard=dict(committed_gb=reading[0],limit_gb=reading[1],need_gb=12,reserve_gb=6))
    write(args.out/'declaration.json',declaration)
    frozen=H.load_frozen(args.checkpoint,threads=3)
    episode,note=H.load_episode(frozen)
    if episode is None or len(episode['worlds'])!=8:raise RuntimeError('Eight saved worlds required; no fresh-world fallback')
    del episode
    before=weights_digest(frozen.model)
    for parameter in frozen.model.params:parameter.requires_grad_(False)
    replays=[];original=H.Session
    H.Session=audited_session(original,replays,args.out,began)
    try:
        cf=G.counterfactual(frozen,states=8,warm=0,spacing=10,depth=3,seed=11,repeats=1)
    finally:H.Session=original
    validate_complete(cf,replays)
    starts=start_records(cf)
    if weights_digest(frozen.model)!=before:raise RuntimeError('Frozen probe changed learned weights')
    if runtime_fingerprint()!=runtime or any(sha256(path)!=value for path,value in scripts.items()):
        raise RuntimeError('Probe source changed during measurement')
    if source_hashes(args.checkpoint)!=declaration['source_sha256']:raise RuntimeError('Source checkpoint changed')
    write(args.out/'counterfactual.json',cf)
    summary=summarize(cf)
    findings=G.summarize(dict(summary=dict(measured=False,reason='Mechanical fresh-world servo test not repeated')),cf,frozen)
    pictures={k:int(sum(row['pictures'][k] for row in replays)) for k in ('draws','reused','redrawn_differently')}
    pictures['meaning']=('Eye pictures drawn in the 80 command branches, both runs: reused = draws answered with '
                         'the picture their identical scene got first; redrawn_differently = of those, draws whose '
                         'fresh picture differed from it (a graphics driver slip, or the floor placed around the '
                         'previous draw\'s camera).')
    result=dict(declaration,summary=summary,findings=findings,checkpoint_episode=note,eye_picture_draws=pictures,
        starts=starts,bodies_started_without_a_plan=sum(row['rule']=='no_plan_in_window' for row in starts),
        evaluated_bodies=8,completed_command_branches=len(replays),exact_replay=True,
        all_candidate_traces_replayed_exactly=True,weights_unchanged=True,elapsed_s=time.time()-began)
    write(args.out/'result.json',result)
    write(args.out/'status.json',dict(stage='complete',completed_arms=80,maximum_arms=80,elapsed_s=time.time()-began))
    print(json.dumps(dict(result=str(args.out/'result.json'),elapsed_s=result['elapsed_s'])),flush=True)


if __name__=='__main__':main()
