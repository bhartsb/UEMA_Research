# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Run two bounded native eight-body continuations of an isolated CPU copy.

Never controls the production trainer. The copied learner still learns and
chooses; its shadow actor only observes. Each pass saves a native full checkpoint.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from mimo_bridge import continuation_state
from .training_continuation import atomic_json,replace_argument,complete_candidates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--supervisor',type=Path,default=Path('Saved/mimo_bridge/training_continuation.json'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--steps',type=int,default=2)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    if not 1<=args.steps<=5:raise ValueError('native smoke budget must be one to five updates per pass')
    receipt=json.loads((args.checkpoint/'continuity_migration.json').read_text())
    if not receipt.get('cpu_probe_only') or receipt['world_count']!=8:raise ValueError('requires an explicit eight-body CPU migration copy')
    original=json.loads(args.supervisor.read_text())
    command=original['command'];source=args.checkpoint.resolve();initial_step=receipt['step'];rows=[]
    from .native_proof_role import CONTRACT,FLAG
    atomic_json(out/'declaration.json',dict(schema=CONTRACT,out=str(out),source=str(source),updates_per_pass=args.steps,passes=2,body_count=8,
        source_preservation=receipt,production_controls_used=False,
        device_scope='CPU functional startup/save/resume. Original production CUDA RNG remains in the separate production migration.',
        criteria='Both native runs complete all updates, preserve8 continuing times, keep observer unqualified and unapplied, and resume its observation count.'))
    for index in range(2):
        folder=out/f'pass_{index+1}';folder.mkdir();target_step=initial_step+(index+1)*args.steps
        argv=list(command)
        overrides={FLAG:out/'declaration.json','--resume':source,'--device':'cpu','--out':folder,'--checkpoint-storage':folder/'checkpoints',
            '--contact-profile':'mechanical_v2_local','--self-choice':'shadow',
            '--self-choice-receipt':receipt['changes']['self_choice_receipt'],
            '--max-steps':target_step,'--minutes':0,'--report-every':1,'--video-every-min':0,
            # Native zero means every step for this timer. End-of-run still saves;
            # avoid extra complete copied checkpoints between two test updates.
            '--checkpoint-every-min':1000000000,'--explain-every-min':0,
            '--caregiver-record-dir':folder/'recordings'}
        for key,value in overrides.items():argv=replace_argument(argv,key,value)
        atomic_json(folder/'command.json',argv)
        atomic_json(out/'status.json',dict(stage='native_startup_resume',pass_number=index+1,pid=os.getpid(),timestamp=time.time()))
        with (folder/'console.log').open('w',encoding='utf-8') as stream:
            done=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
        if done.returncode:
            atomic_json(out/'status.json',dict(stage='failed',pass_number=index+1,
                exit_code=done.returncode,log=str(folder/'console.log'),timestamp=time.time()))
            raise RuntimeError(f'native copied pass{index+1} failed; see {folder / "console.log"}')
        candidates=complete_candidates(folder)
        if not candidates or int(candidates[0].name.rsplit('_',1)[1])!=target_step:raise ValueError('native pass did not publish the exact completed step')
        source=candidates[0];episode=continuation_state.load(source)
        from .actor_runtime import inner_state_for_report
        saved_actor=episode['trainer'].get('acting_snapshot')
        actor=inner_state_for_report(saved_actor);times=[w['payload']['sim_time'] for w in episode['worlds']]
        initial_observations=receipt.get('actor_observations',0)
        if len(times)!=8 or not actor or not actor['qualification'] or actor['steps']!=initial_observations+(index+1)*args.steps:
            raise ValueError('native observer or eight-body continuation did not survive')
        if receipt.get('outer_contract'):
            if (saved_actor['contract']!=receipt['outer_contract']
                    or saved_actor['manifest']['binding']['sha256']!=receipt['outer_binding_sha256']
                    or actor['born_step']!=receipt['actor_born_step']
                    or actor['receipt']['model_execution_sha256']!=receipt['inner_execution_sha256']
                    or actor['receipt']['self_weights_sha256']!=receipt['actor_self_weights_sha256']
                    or actor['receipt']['perception_weights_sha256']!=receipt['actor_perception_weights_sha256']):
                raise ValueError('outer code binding, frozen weights or birth changed across native continuation')
        if actor.get('last',{}).get('applied_to_body') is not False:raise ValueError('unqualified observer authored a physical command')
        local_rows=[]
        for world,now in zip(episode['worlds'],times):
            skin=world['world']['attrs'].get('_virtual_skin')
            info=world['payload']['info']
            if (world['env_kwargs'].get('contact_profile')!='mechanical_v2_local' or skin is None
                    or abs(skin.last_time-now)>1e-9 or not np.isfinite(skin.function.quality).all()):
                raise ValueError('local skin did not run on every continuing physical world')
            if 'surface_moments_root' not in info.get('skin_function_granularity',{}):
                raise ValueError('actual within-body contact locations did not reach the learner frame')
            local_rows.append(dict(taxels=len(skin.function.quality),last_time=skin.last_time,
                minimum_quality=float(skin.function.quality.min()),maximum_quality=float(skin.function.quality.max()),
                surface_moment_rows=len(info['skin_function_granularity']['surface_moments_root'])))
        before=receipt['sim_times'] if not rows else rows[-1]['sim_times']
        expected_delta=args.steps*.05
        if any(abs(now-prior-expected_delta)>1e-7 for prior,now in zip(before,times)):
            raise ValueError('physical episode clock reset or advanced incorrectly')
        row=dict(pass_number=index+1,checkpoint=str(source),step=target_step,sim_times=times,
            completed_updates=args.steps,
            actor_observations=actor['steps'],actor_born_step=actor['born_step'],qualification_only=True,
            applied_to_body=False,world_count=len(times),local_skin=local_rows,
            runtime_sha256=continuation_state.file_sha256(source/'runtime_state.pt'))
        row.update(outer_contract=saved_actor['contract'],inner_execution_sha256=actor['receipt']['model_execution_sha256'],
            outer_binding_sha256=saved_actor.get('manifest',{}).get('binding',{}).get('sha256'),
            completed_audits=None if actor.get('opportunity') is None else actor['opportunity']['completed'],
            pending_audit=actor.get('opportunity') is not None and actor['opportunity']['pending'] is not None)
        rows.append(row);del episode,actor,saved_actor;gc.collect();atomic_json(out/'partial_report.json',rows)
    summary=dict(complete=True,passes=rows,production_choice_license=False,
        original_production_trainer_controlled=False,scope='Actual8-body CPU native training and observing actor startup/save/resume; no behavioral qualification.')
    if receipt.get('outer_contract'):
        manifest=Path(receipt['production_source_manifest'])
        if continuation_state.file_sha256(manifest)!=receipt['production_source_manifest_sha256']:
            raise ValueError('production source manifest changed during the copied proof')
        # This copy has device accommodations; the original production source
        # must still retain every native file and its complete physical blob.
        migration=json.loads((args.checkpoint/'decorated_execution_migration.json').read_text())
        parent=Path(receipt['parent']);names=migration.get('auxiliary_original_names',{})
        for name,digest in migration['native_file_sha256'].items():
            if continuation_state.file_sha256(parent/names.get(name,name))!=digest:
                raise ValueError('original production checkpoint bytes changed during the copied proof')
        physical=json.loads((parent/'episode_state.json').read_text())
        blob=Path(physical.get('storage_path') or parent/'episode_state.pt.gz')
        if continuation_state.file_sha256(blob)!=physical['sha256']:
            raise ValueError('original production physical/actor state bytes changed during the copied proof')
        summary.update(production_source_manifest_sha256=receipt['production_source_manifest_sha256'],
            production_binding_sha256=receipt['outer_binding_sha256'],source_checkpoint=receipt['parent'],n_envs=8,
            inner_actor_continuation_preserved=True,learner_and_physical_continuation_preserved=True,
            original_production_native_and_episode_bytes_unchanged=True,
            cpu_accommodations=dict(description=receipt['accommodation'],precision=receipt['precision_accommodation'],
                original_inner_execution_sha256=receipt['original_inner_execution_sha256'],
                copied_inner_execution_sha256=receipt['inner_execution_sha256'],
                original_values_archive=str(args.checkpoint/'cpu_probe_original_values.pt')))
    atomic_json(out/'report.json',summary)
    atomic_json(out/'status.json',dict(stage='complete',timestamp=time.time(),passes=2))


if __name__=='__main__':main()
