# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Measure whether a passive narrow support leaves a recoverable contact.

Only copied physics is changed. No function value or correct action is supplied
to a learner. All branches begin at the same physical snapshot; removal is a
separate external control. Recorded pressure units distinguish sensing precision
from an arbitrary percentage of wellness.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from mimo_bridge.train_mimo import build_learner,env_kwargs_for
from mimo_bridge.datasource import SingleEnvSource
from mimo_bridge.lem_model import LemLite
from .self_recovery_experiment import configuration,hold,targets
from .recorded_limits import recorded_limits
from .training_continuation import atomic_json


def pressure_resolution(profile,q):
    q=np.asarray(q)
    bits=np.rint(profile['minimum_bits']+(profile['maximum_bits']-profile['minimum_bits'])*q).astype(int)
    # The actual operator bypasses quantization at full function. A theoretical
    # 23-bit grid must not invent lost distinctions in unchanged float32 input.
    width=np.where(q>=1.,0.,profile['pressure_range_pa']/(2.**bits-1))
    return bits,width


def pressure_distinctions(raw_pressure,experienced,width):
    physical=int(len(np.unique(raw_pressure)))
    worst=float(np.max(width))
    return dict(distinct_physical_pressures=physical,
                distinct_experienced_pressures=int(len(np.unique(experienced))),
                distinctions_at_lowest_precision=(physical if worst==0 else
                    int(len(np.unique(np.rint(raw_pressure/worst))))))


def source_force(env,geometry):
    return sum(source_force_by_body(env,geometry).values())


def source_force_by_body(env,geometry):
    forces={}
    for index in range(env.data.ncon):
        contact=env.data.contact[index]
        if geometry in (int(contact.geom1),int(contact.geom2)):
            resolved=np.zeros(6);env._mj.mj_contactForce(env.model,env.data,index,resolved)
            other=int(contact.geom2) if int(contact.geom1)==geometry else int(contact.geom1)
            body=int(env.model.geom_bodyid[other]);name=env.model.body(body).name
            forces[name]=forces.get(name,0.)+abs(float(resolved[0]))
    return forces


def commands(frame,limits,side):
    base=hold(frame,min(.3,float(limits['strength_limit'])))[0]
    names=list(__import__('mimo_bridge.body_layout',fromlist=['names_for']).names_for(frame.meta['body_layout']))
    selected=[index for index,name in enumerate(names) if name.endswith('_L' if side=='left' else '_R')
              and any(part in name for part in ('wrist','thumb','index','middle','ring','pinky','upperarm','lowerarm'))]
    raw=[base.copy()];labels=['hold']
    for axis in range(3):
        for sign in (-1,1):
            cmd=base.copy();cmd[selected,axis]+=sign*8.  # Measured commands, 8 cm requested.
            raw.append(cmd);labels.append(f'axis_{axis}_{sign:+d}')
    normalized=torch.from_numpy(np.stack(raw));normalized[...,:3]/=200.
    achieved=torch.from_numpy(base[None]);achieved[...,:3]/=200.
    return normalized,achieved,labels


def one_world(config,limits,folder,seed,side,depth,seconds):
    kwargs=env_kwargs_for(config,limits['phase_name'],str(folder))
    kwargs.update(seed=seed,vision=False,body_function='modulate',caregiver_mode='off',caregiver_seconds=0.,
                  caregiver_record_dir='',critters=False,mobile=False,gaze_reflex=False,spontaneous_saccades=False)
    source=SingleEnvSource(**kwargs);frames=source.reset();env=source.env;dt=env.dt*config.action_repeat
    mocap=env._target_id;env._target_id=None;env.data.mocap_pos[mocap]=[0,0,2]
    geometry=int(env.model.geom('target_geom').id)
    command=hold(frames[0],min(.3,float(limits['strength_limit'])))
    def step(cmd):
        nonlocal frames
        for _ in range(config.action_repeat):frames=source.step(cmd,np.zeros((1,2)),limits.get('handle_mask'),render=False)
        return frames[0]
    try:
        for _ in range(round(6.4/dt)):step(command)
        command=hold(frames[0],min(.3,float(limits['strength_limit'])))
        body=int(env.model.body(side+'_hand').id);center=env.data.xpos[body].copy()
        normal=np.array([0.,0.,-1.]);origin=center+normal*.2;hits=[]
        for gid in range(env.model.ngeom):
            if env.model.geom_bodyid[gid]==body:
                ray=env._mj.mju_rayGeom(env.data.geom_xpos[gid],env.data.geom_xmat[gid],env.model.geom_size[gid],origin,-normal,int(env.model.geom_type[gid]))
                if ray>=0:hits.append(ray)
        if not hits:raise ValueError('support ray missed actual hand geometry')
        surface=origin-normal*min(hits);point=surface+normal*(.025-depth)
        for index in range(round(.5/dt)):
            env.data.mocap_pos[mocap]=point+normal*.03*(1-(index+1)/round(.5/dt));step(command)
        for _ in range(round(.25/dt)):step(command)
        start=frames[0];snapshot=source.save_full_state()
        geometry_audit=[]
        for contact_index in range(env.data.ncon):
            contact=env.data.contact[contact_index]
            if geometry not in (int(contact.geom1),int(contact.geom2)):continue
            geometry_audit.append(dict(distance_m=float(contact.dist),point_m=contact.pos.tolist(),
                normal=contact.frame[:3].tolist(),dimension=int(contact.dim),
                geoms=[dict(name=env.model.geom(gid).name,type=int(env.model.geom_type[gid]),
                            size=env.model.geom_size[gid].tolist(),position=env.data.geom_xpos[gid].tolist(),
                            orientation=env.data.geom_xmat[gid].tolist(),solref=env.model.geom_solref[gid].tolist())
                       for gid in (int(contact.geom1),int(contact.geom2))]))
        raw,achieved,labels=commands(start,limits,side)
        source_contacts=source_force_by_body(env,geometry)
        site_ids=start.meta['body_function']['site_ids']
        sensed_contacts={name:force for name,force in source_contacts.items() if name in site_ids}
        # Select the actually contacted skin at the decision instant. A sphere
        # aimed at a hand can meet an adjacent finger as the body settles.
        target_name=max(sensed_contacts,key=sensed_contacts.get) if sensed_contacts else side+'_hand'
        branches=[];traces=[];target_site=site_ids.index(target_name);unsupported_geometry={}
        for index in list(range(len(raw)))+[0,-1]:
            source.restore_full_state(snapshot)
            if index<0:env.data.mocap_pos[mocap]=[0,0,2]
            prior=torch.from_numpy(hold(start,min(.3,float(limits['strength_limit']))))
            values=[];raw_pressure=[];experienced=[];forces=[];poses=[];areas=[];local_changed=[];local_errors=[];affected=[];unsupported=[];peak_footprints=[];peak_pressure=-1.
            for tick in range(round(seconds/dt)):
                if tick%config.motor_hold==0:
                    prior=LemLite.planned_execution(raw[max(index,0):max(index,0)+1],limits,achieved,previous=prior,smoothing=config.smooth_actions)
                frame=step(prior.numpy());value,valid=targets(frame)
                values.append(value.numpy());poses.append(frame.bones.copy());forces.append(source_force(env,geometry))
                physical=np.asarray(frame.meta['physical_touch_audit'])
                raw_pressure.append(float(physical[target_site,0]));experienced.append(float(frame.patches[target_site,0]))
                areas.append(float(physical[target_site,3])/1e6)  # mm² -> m²
                if config.contact_profile=='mechanical_v3':
                    skin=env._virtual_skin;segment=skin.slices[env.pooler.specs[target_site].body_id]
                    difference=abs(skin.physical[segment,:2]-skin.experienced[segment,:2])
                    local_changed.append(int(np.count_nonzero(difference)))
                    local_errors.append(float(difference.max()))
                    affected.append(frame.meta['skin_function_granularity']['affected_sensor_area_fraction'][target_site])
                    unsupported.append(frame.meta['mechanical_contact']['unresolved_geometry_contacts'])
                    if raw_pressure[-1]>peak_pressure:
                        from copy import deepcopy
                        peak_pressure=raw_pressure[-1]
                        peak_footprints=deepcopy([r for r in skin.contact_records if r['body_id']==env.pooler.specs[target_site].body_id])
                    for patch in skin.unsupported:
                        key=(patch['body_id'],tuple(patch['pair']),patch.get('reason'))
                        previous=unsupported_geometry.get(key,{})
                        unsupported_geometry[key]=dict(body=env.model.body(patch['body_id']).name,
                            geoms=[env.model.geom(gid).name for gid in patch['pair']],reason=patch.get('reason'),
                            method=patch['method'],maximum_force_N=max(patch['normal_force_n'],previous.get('maximum_force_N',0)))
            values=np.asarray(values);poses=np.asarray(poses)
            bits,width=pressure_resolution(start.meta['body_function']['profile'],values[:,target_site,0])
            raw_pressure=np.asarray(raw_pressure);experienced=np.asarray(experienced)
            distinctions=pressure_distinctions(raw_pressure,experienced,width)
            if config.contact_profile=='mechanical_v3':distinctions['distinctions_at_lowest_precision']=None
            branches.append(dict(action='source_removed' if index<0 else labels[index],
                target_quality_mean=float(values[:,target_site,0].mean()),target_quality_min=float(values[:,target_site,0].min()),
                target_load_mean=float(values[:,target_site,1].mean()),contact_force_mean_N=float(np.mean(forces)),
                pressure_mean_Pa=float(raw_pressure.mean()),pressure_max_Pa=float(raw_pressure.max()),
                loaded_area_mean_cm2=float(np.mean(areas)*1e4),contact_profile=config.contact_profile,
                maximum_patch_force_from_pressure_area_N=float(np.max(raw_pressure*np.asarray(areas))),
                changed_pressure_observations=int(np.count_nonzero(raw_pressure!=experienced)),
                actual_pressure_quantization_error_max_Pa=float(np.max(np.abs(raw_pressure-experienced))),
                minimum_precision_bits=int(bits.min()),maximum_pressure_bin_width_Pa=float(width.max()),
                precision_scope='worst local taxel; pooled pressure has no uniform grid' if local_changed else 'uniform body-row operator',
                changed_local_pressure_shear_observations=sum(local_changed),
                maximum_local_quantization_error_Pa=max(local_errors,default=0.),
                maximum_affected_sensor_area_fraction=max(affected,default=0.),
                maximum_unresolved_geometry_contacts=max(unsupported,default=0),**distinctions))
            branches[-1]['target_footprints_at_peak_pressure']=peak_footprints
            traces.append(dict(function=values,poses=poses,source_force_N=np.asarray(forces),raw_pressure_Pa=raw_pressure,
                               experienced_pressure_Pa=experienced,active_sensor_support_area_m2=np.asarray(areas)))
        np.testing.assert_array_equal(traces[0]['function'],traces[-2]['function'])
        np.testing.assert_array_equal(traces[0]['poses'],traces[-2]['poses'])
        remove_gain=branches[-1]['target_quality_mean']-branches[0]['target_quality_mean']
        best=max(range(len(raw)),key=lambda index:branches[index]['target_quality_mean'])
        movement_gain=branches[best]['target_quality_mean']-branches[0]['target_quality_mean']
        threshold=8*np.finfo(np.float32).eps
        result=dict(seed=seed,side=side,depth_m=depth,source_position_m=point.tolist(),target_site=target_site,
                    target_site_name=target_name,initial_source_contact_forces_N=source_contacts,
                    initial_contact_geometry=geometry_audit,
                    unsupported_geometry=list(unsupported_geometry.values()),
                    initial_source_contact_force_N=float(sum(source_contacts.values())),
                    initial_quality=float(start.meta['body_function']['values'][target_site][0]),
                    branches=branches,source_removal_gain=remove_gain,best_command=labels[best],movement_gain=movement_gain,
                    recoverable=bool(remove_gain>threshold and movement_gain>threshold),replay_max_error=0.,
                    scope='Physical opportunity only. Quality is a precision index, not wellness. No learner behavior or gain of function is certified.')
        atomic_json(folder/'report.json',result);torch.save(traces,folder/'traces.pt');return result
    finally:source.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--depths-mm',default='2,6,12');parser.add_argument('--seconds',type=float,default=5.)
    parser.add_argument('--sides',nargs='+',choices=['left','right'],default=['left','right'])
    parser.add_argument('--contact-profile',choices=['mechanical_v2','mechanical_v3'],default='mechanical_v2')
    parser.add_argument('--limits-from',type=Path,help='Explicit prior declaration containing measured physical limits')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    from mimo_bridge import contact_footprint,virtual_skin,env
    paths=[Path(__file__),Path(contact_footprint.__file__),Path(virtual_skin.__file__),Path(env.__file__)]
    source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    config=configuration(args.checkpoint,out);config.contact_profile=args.contact_profile
    model=build_learner(config,str(out/'unused_checkpoints'))
    model.load_checkpoint(str(args.checkpoint),load_optimizer=False)
    limits,limits_source=recorded_limits(args.checkpoint,model,args.limits_from)
    del model
    import gc
    gc.collect()
    depths=[float(value)/1000 for value in args.depths_mm.split(',')]
    if any(not 0<depth<.025 for depth in depths) or args.seconds<=0:raise ValueError('invalid support probe dimensions')
    atomic_json(out/'declaration.json',dict(parent=str(args.checkpoint),depths_m=depths,seconds=args.seconds,sides=args.sides,
        contact_profile=args.contact_profile,limits=limits,limits_source=limits_source,source_hashes=source_hashes,
        question='Can an ordinary feasible arm movement relieve a continuing contact supplied from below?',
        admission='Same-state replay exact; source-removal and at least one movement each restore more target-site sensing precision than hold, exceeding eight float32 epsilons. Also report actual pressure distinctions, bin widths, forces and motor effects before making any functional claim.',
        control='Passive fixed existing sphere supports the hand. No invented sensory impairment. Source-removal is an external measuring intervention. Probe actions are only an external reachability instrument.'))
    rows=[]
    for index,(side,depth) in enumerate((side,depth) for side in args.sides for depth in depths):
        atomic_json(out/'status.json',dict(stage='measuring_passive_support',timestamp=time.time(),pid=os.getpid(),step=index,steps=len(args.sides)*len(depths)))
        folder=out/f'{side}_{depth*1000:g}mm';folder.mkdir()
        rows.append(one_world(config,limits,folder,8400+index,side,depth,args.seconds))
        atomic_json(out/'partial_results.json',rows)
    atomic_json(out/'report.json',dict(worlds=rows,recoverable_worlds=sum(row['recoverable'] for row in rows),production_license=False,
        source_hashes=source_hashes,source_changed_during_run=any(hashlib.sha256(path.read_bytes()).hexdigest()!=source_hashes[str(path)] for path in paths)))
    atomic_json(out/'status.json',dict(stage='complete',timestamp=time.time(),pid=os.getpid(),verdict='PHYSICAL_RECOVERY_OPPORTUNITIES_MEASURED'))


if __name__=='__main__':
    try:main()
    except Exception as error:
        import sys,traceback
        out=Path(sys.argv[sys.argv.index('--out')+1])
        if out.exists():
            atomic_json(out/'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
            atomic_json(out/'status.json',dict(stage='failed',timestamp=time.time(),error=str(error)))
        raise
