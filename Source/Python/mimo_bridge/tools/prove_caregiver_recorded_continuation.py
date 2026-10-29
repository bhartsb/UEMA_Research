# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Exact native learning continuation under identical recorded sensory inputs.

Physical rendering is measured separately. This instrument never replaces a
caregiver-benefit trial's observations or qualifies a new acting policy.
"""
import argparse
from copy import deepcopy
import gc
import gzip
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch
from safetensors import safe_open

from mimo_bridge import continuation_state
from mimo_bridge.datasource import SingleEnvSource
from . import caregiver_fit_continuation as continuation
from .caregiver_learning_trial import learner,settings,Orders,write
from .caregiver_fit_workers import resources,admission,guard_for
from .prove_caregiver_fit_continuation import update_evidence,scientific_state,first_difference
from .prove_caregiver_render_repeatability import physical_render_state
from .prepare_local_shadow import state_digest


def close(source):
    fovea=source.env._fovea_renderer;source.close()
    if fovea is not None:fovea.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--speech',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    if args.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.25,device=args.device)
    origin=continuation_state.load(args.checkpoint);trainer=origin['trainer'];resume=trainer['fit_resume']
    original_binding=resume['binding'];start=int(trainer['trial_step']);total=int(original_binding['steps']);cut=start+2
    if total-start!=4:raise ValueError('This bounded proof requires exactly four remaining native updates')
    parent=Path(original_binding['parent']);parent_step=resume['parent_step'];seed=original_binding['seed'];dev=original_binding['limits']
    owner=dict(pid=psutil.Process().pid,process_created=psutil.Process().create_time())
    write(args.out/'declaration.json',dict(source=str(args.checkpoint),source_episode_sha256=continuation_state.file_sha256(args.checkpoint/'episode_state.pt.gz'),
        starting_trial_step=start,total_updates=total,save_after_trial_step=cut,device=args.device,
        primary='All learned tensors, Adam, causal execution, random states, commands, order timing and speech accounting remain exact under the same recorded observations.',
        physical='Generated physical states and pixels are retained separately. A renderer difference is not replaced in a benefit trial.',
        scope='Native preservation instrument only; recorded inputs isolate learning continuation from the measured renderer floor. No acquired skill or end-to-end pixel pass.'))
    samples={};reference_commands={};reference_physics={};paths={};checks={};boundaries={};started=time.time()
    for arm in ('reference','restored'):
        out=args.out/arm;a=settings(parent,out,args.speech,args.device,seed)
        admitted,reason=admission(resources(args.device),[])
        if not admitted:raise RuntimeError('Preserving production resources: '+reason)
        saved=origin if arm=='reference' else continuation_state.load(paths['cut'])
        source_checkpoint=args.checkpoint if arm=='reference' else paths['cut']
        model,loaded=learner(a,source_checkpoint,out,True)
        source=SingleEnvSource(**saved['worlds'][0]['env_kwargs']);orders=Orders(a,seed)
        frames=continuation.restore(model,source,orders,saved)
        old=saved['trainer']['fit_resume'];utterances=deepcopy(old['utterances']);last=deepcopy(old['last_utterance'])
        initial=old['initial_encoder_sha256'];first=int(saved['trainer']['trial_step'])
        guard=guard_for(dict(out=str(out),coordinator=owner,device=args.device,gpu_reserve_mib=1024,ram_reserve_gib=4.,check_every_updates=1))
        binding=continuation.binding(a,parent,out,seed,total,dev,None)
        def save(completed):
            guard.before_save(completed)
            def companion(staging,final):
                continuation.save(staging,final,source,model,dict(step=parent_step+completed,trial_step=completed,
                    held=orders.held,executed=orders.executed,order=orders.order,bout=orders.bout,until=orders.until,
                    bout_rng=orders.rng.bit_generator.state,frame_meta=[deepcopy(f.meta) for f in frames],
                    fit_resume=dict(contract=continuation.CONTRACT,binding=binding,parent_step=parent_step,
                        utterances=deepcopy(utterances),last_utterance=deepcopy(last),initial_encoder_sha256=initial,
                        active_wall_seconds=old['active_wall_seconds']+time.time()-started)),vars(a))
            return Path(model.save_checkpoint(parent_step+completed,companion_writer=companion))
        try:
            for index in range(first,total):
                guard(model,index)
                limits=orders.limits(dev,index,source.env.dt*a.action_repeat,True)
                if arm=='reference':
                    samples[index]=dict(frames=deepcopy(frames),limits=deepcopy(limits),executed=deepcopy(orders.executed),order=orders.order)
                    with gzip.open(args.out/f'input_{index}.pt.gz','wb',compresslevel=1) as stream:torch.save(samples[index],stream)
                else:
                    sample=samples[index]
                    checks[f'actual_prior_order_{index}']=state_digest((orders.executed,orders.order,limits))==state_digest((sample['executed'],sample['order'],sample['limits']))
                    # Explicit fixed-input instrument; raw generated pixels remain
                    # in source payload and the physical comparison below.
                    frames=deepcopy(sample['frames'])
                if index==cut:
                    boundaries[arm]=update_evidence(model,frames)
                    write(out/'restoration_boundary.json',boundaries[arm])
                result=model.step(frames,limits,train=True,step_index=parent_step+index,
                    executed_action=orders.executed,executed_order_id=orders.order or None)
                if not np.isfinite(result.loss):raise ValueError('Nonfinite native replay learning')
                command=state_digest((result.motor_for_action,result.gaze_out))
                if arm=='reference':reference_commands[index]=command
                else:checks[f'command_{index}']=command==reference_commands[index]
                frames=orders.advance(model,result,source,index,True)
                utterance=source.env.caregiver.state.get('last_utterance')
                if utterance and utterance!=last:utterances.append((utterance['time'],utterance['key']));last=deepcopy(utterance)
                physical=physical_render_state(source.env)
                np.savez_compressed(out/f'physical_after_{index}.npz',**physical,pixels=frames[0].vision_patches)
                if arm=='reference':reference_physics[index]={k:state_digest(v) for k,v in physical.items()}
                else:checks[f'physical_after_{index}']=reference_physics[index]=={k:state_digest(v) for k,v in physical.items()}
                write(out/'status.json',dict(stage='fit',step=index+1,steps=total,timestamp=time.time(),recorded_input=(arm=='restored')))
                if arm=='reference' and index+1==cut:paths['cut']=save(cut)
            paths[arm]=save(total)
            write(out/'speech.json',dict(utterances=utterances,last=last))
        finally:
            close(source);del source,model,saved;gc.collect()
            if args.device.startswith('cuda'):torch.cuda.empty_cache()
    checks['restoration_boundary_exact']=state_digest(boundaries['reference'])==state_digest(boundaries['restored'])
    with safe_open(str(paths['reference']/'model.safetensors'),framework='pt',device='cpu') as a,safe_open(str(paths['restored']/'model.safetensors'),framework='pt',device='cpu') as b:
        checks['all_learned_tensors_exact']=set(a.keys())==set(b.keys()) and all(torch.equal(a.get_tensor(k),b.get_tensor(k)) for k in a.keys())
    for filename in ('optimizer.pt','runtime_state.pt'):
        checks[filename]=state_digest(torch.load(paths['reference']/filename,map_location='cpu',weights_only=False))==state_digest(torch.load(paths['restored']/filename,map_location='cpu',weights_only=False))
    left=scientific_state(paths['reference']);right=scientific_state(paths['restored'])
    for field in ('learner','execution','speech'):checks[field]=state_digest(left[field])==state_digest(right[field])
    report=dict(recorded_input_continuation_pass=all(checks.values()),checks=checks,
        boundary_first_difference=first_difference(boundaries['reference'],boundaries['restored']),
        final_first_difference={k:first_difference(left[k],right[k],k) for k in ('learner','execution','speech')},
        checkpoints={k:str(v) for k,v in paths.items()},wall_seconds=time.time()-started,
        end_to_end_pixel_pass=False,capability_pass=False,scope='Exact learning preservation with identical recorded inputs. Fresh-renderer variation remains separately measured.')
    write(args.out/'report.json',report);print(json.dumps(report,indent=2))
    if not report['recorded_input_continuation_pass']:raise ValueError('Recorded-input native preservation check failed')


if __name__=='__main__':main()
