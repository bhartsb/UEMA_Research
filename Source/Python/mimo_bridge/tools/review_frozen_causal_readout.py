# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Read-only evaluation of a completed readout; never fit or rewrite its result."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from lem.model.causal_readout import CausalReadout
from mimo_bridge.causal_packets import load_packet,encode_own_window
from mimo_bridge.lem_model import command_for_encoder
from mimo_bridge.tools.causal_readout_study import (restore_encoder,batch,metrics,
    file_hash,validate_index)
from mimo_bridge.tools.training_continuation import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    report=json.loads((args.study/'report.json').read_text());decl=json.loads((args.study/'declaration.json').read_text())
    report_hash=file_hash(args.study/'report.json')
    index=json.loads((args.study/'packet_index.json').read_text())
    from types import SimpleNamespace
    validate_index(SimpleNamespace(out=args.study,train_worlds=decl['train_worlds'],test_worlds=decl['test_worlds'],
                                   states=decl['states_per_world']),index,report['source_checkpoint_sha256'])
    encoder_path=args.study/report['standalone_encoder']['path']
    if file_hash(encoder_path)!=report['standalone_encoder']['sha256']:raise ValueError('saved encoder artifact changed')
    receiver=restore_encoder(encoder_path);rows=[]
    for receipt in index:
        p=load_packet(args.study/'packets'/receipt['path']);enc,past=encode_own_window(receiver,p['context'])
        times=np.array([f['sim_time'] for f in p['context']])
        rows.append(dict(world=p['world_id'],phase=p['phase'],split=p['split'],encoded=enc,
            sensed=torch.from_numpy(past),history_dt=torch.from_numpy(np.r_[0,np.diff(times)].astype(np.float32)),
            action=command_for_encoder(torch.from_numpy(p['commands'])).reshape(4,5,-1),
            duration=torch.tensor(p['durations'],dtype=torch.float32),target=torch.from_numpy(p['targets']),
            start=torch.from_numpy(p['start']),layout=p['target_layout']))
    results=[];exact=True
    for condition in ('action_conditioned','action_blind'):
        path=args.study/(condition+'.pt')
        if file_hash(path)!=report['new_auxiliary_checkpoints'][condition+'_stage_2']:
            raise ValueError('saved auxiliary artifact changed')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=CausalReadout(**saved['architecture']);model.load_state_dict(saved['model']);model.eval()
        for split in ('train','test'):
            selected=[r for r in rows if r['split']==split]
            inputs,_=batch(selected,saved['scale'],saved['encoded_scale'],condition=='action_blind',
                           saved['sensed_mean'],saved['sensed_scale'])
            with torch.no_grad():
                pred=model(*inputs)*saved['scale']+torch.cat([r['start'][None].expand(4,-1) for r in selected])[:,None]
            measured=metrics(selected,pred,condition+'_stage_2')
            if split=='test':
                original=[r for r in report['results'] if r['forecast']==condition+'_stage_2']
                if measured!=original:raise ValueError('saved readout cannot reproduce its held-out report exactly')
            results.extend(dict(r,split=split) for r in measured)
        del saved,model
    aggregates=[]
    for group in rows[0]['layout']:
        for split in ('train','test'):
            for condition in ('action_conditioned','action_blind'):
                selected=[r for r in results if r['group']==group and r['split']==split
                          and r['forecast']==condition+'_stage_2' and r['horizon_seconds']==.25]
                stats={key:float(np.sqrt(np.mean([r[key]**2 for r in selected]))) for key in
                       ('prediction_rmse','action_effect_rms','action_contrast_error_rms','persistence_rmse','constant_motion_rmse')}
                aggregates.append(dict(group=group,split=split,condition=condition,**stats))
    contrasts=[]
    for group,(lo,hi) in rows[0]['layout'].items():
        for split in ('train','test'):
            for endpoint in (None,4):
                changes=[];effects=[]
                for row in rows:
                    if row['split']!=split:continue
                    target=row['target'][:,:,lo:hi]
                    change=target-row['start'][lo:hi];effect=target-target.mean(0,keepdim=True)
                    if endpoint is not None:change=change[:,endpoint];effect=effect[:,endpoint]
                    changes.append(float(change.square().mean()));effects.append(float(effect.square().mean()))
                contrasts.append(dict(group=group,split=split,horizon='all five steps' if endpoint is None else '250ms',
                    mean_change_squared=float(np.mean(changes)),mean_action_contrast_squared=float(np.mean(effects)),
                    action_share_of_squared_change=float(np.mean(effects)/np.mean(changes)) if np.mean(changes)>0 else None))
    result=dict(schema='frozen_causal_readout_review_v1',source_report_sha256=report_hash,
        source_report=str((args.study/'report.json').resolve()),held_out_report_reproduced_exactly=exact,
        fitted_updates=0,production_modified=False,aggregates=aggregates,contrast_budget=contrasts,
        complete_measurements=results,original_admission=report['admission'])
    if file_hash(args.study/'report.json')!=report_hash:raise ValueError('original report changed during read-only review')
    if args.out.exists():raise FileExistsError('review evidence already exists')
    args.out.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.out,result)
    print(json.dumps(dict(held_out_report_reproduced_exactly=exact,review=str(args.out),aggregates=aggregates),indent=2))


if __name__=='__main__':main()
