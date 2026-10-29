# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Measure remaining study storage without changing any checkpoint or recording."""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import time


def files_below(root,exclude_attempts=False):
    root=Path(root)
    return [p for p in root.rglob('*') if p.is_file() and not (
        exclude_attempts and any(part.startswith('attempt_') for part in p.relative_to(root).parts[:-1]))]


def size(files):return sum(path.stat().st_size for path in files)


def estimate(out,probe_reference):
    out=Path(out);probe_reference=Path(probe_reference)
    declaration=json.loads((out/'declaration.json').read_text())
    index=json.loads((out/'completed_fits.json').read_text());measurements=[]
    for arm,record in index.items():
        fit_path=Path(record);fit=json.loads(fit_path.read_text());folder=fit_path.parent
        if fit['online_updates']!=declaration['train_steps']:raise ValueError('Incomplete copied fitting budget')
        checkpoints=[]
        for checkpoint in sorted((folder/'checkpoints').glob('ckpt_step_*')):
            if not checkpoint.is_dir():continue
            stored=files_below(checkpoint);total=size(stored)
            manifest=checkpoint/'episode_state.json';external=0
            if manifest.exists():
                episode=json.loads(manifest.read_text());blob=Path(episode.get('storage_path',checkpoint/'episode_state.pt.gz')).resolve()
                if not blob.is_relative_to(checkpoint.resolve()):external=blob.stat().st_size
            checkpoints.append(dict(path=str(checkpoint),bytes=total,external_episode_bytes=external,
                total_bytes=total+external,files={p.name:p.stat().st_size for p in stored}))
        if len(checkpoints)!=3:raise ValueError('Completed fit no longer has the original three checkpoints')
        all_files=files_below(folder,exclude_attempts=True)
        tapes=[p for p in all_files if 'recordings' in p.relative_to(folder).parts]
        checkpoint_bytes=sum(row['total_bytes'] for row in checkpoints)
        other=[p for p in all_files if 'checkpoints' not in p.relative_to(folder).parts and p not in tapes]
        measurements.append(dict(arm=arm,folder=str(folder),checkpoints=checkpoints,
            checkpoint_bytes=checkpoint_bytes,tape_bytes=size(tapes),other_fit_bytes=size(other),
            full_fit_bytes=checkpoint_bytes+size(tapes)+size(other)))
    conditions=('follow_visible','follow_hidden','return_visible','return_hidden','cue_correct','cue_wrong')
    probes=[]
    for condition in conditions:
        folder=probe_reference/condition;result=json.loads((folder/'result.json').read_text())
        if result['samples']!=declaration['probe_steps']:raise ValueError('Probe reference uses a different test length')
        probes.append(dict(condition=condition,bytes=size(files_below(folder))))
    fixture_paths=[p for folder in out.glob('fixtures_*') for p in files_below(folder)]
    expected_fixtures=declaration['worlds']*len(conditions)
    fixture_count=sum((folder/(condition+'.json')).is_file() for folder in out.glob('fixtures_*') for condition in conditions)
    if fixture_count!=expected_fixtures:raise ValueError('Not all fixed probe fixtures are present')
    remaining=2*declaration['worlds']-len(measurements)
    target_probes=2*declaration['worlds']*len(conditions)
    # All frozen test outputs are still pending in the main comparison. The
    # separate early probe supplies measured bytes only, never test outcomes.
    finished_probes=sum(1 for path in out.rglob('result.json') if path.parent.name.startswith('probe_'))
    if finished_probes:raise ValueError('Update the estimate for already completed main-study probes')
    fits=[row['full_fit_bytes'] for row in measurements]
    per_checkpoint=max(item['total_bytes'] for row in measurements for item in row['checkpoints'])
    additional=dict(lower_bytes=remaining*min(fits)+target_probes*min(p['bytes'] for p in probes),
        central_bytes=round(remaining*statistics.mean(fits)+target_probes*statistics.mean(p['bytes'] for p in probes)),
        upper_bytes=remaining*max(fits)+target_probes*max(p['bytes'] for p in probes))
    return dict(measured_at=time.time(),study=str(out),completed_fits=len(measurements),remaining_fits=remaining,
        measurements=measurements,measured_probe_outputs=probes,pending_probe_outputs=target_probes,
        fixed_fixture_count=fixture_count,fixed_fixture_bytes=size(fixture_paths),additional_fixture_bytes=0,
        additional=additional,current_free_bytes=shutil.disk_usage(out).free,
        production_free_reserve_bytes=4*1024**3,
        two_worker_transient_save_allowance_bytes=2*per_checkpoint,
        required_free_upper_with_reserve_bytes=additional['upper_bytes']+4*1024**3+2*per_checkpoint,
        model='Sixteen remaining fits, each retaining all three original checkpoints plus its measured tape/scene overhead; 144 pending probe outputs. Fixed fixtures already exist.',
        limitations=['Empirical range across completed responsive fits, not a guaranteed maximum. Yoked fits may record fewer new events; using responsive tapes for all 16 is conservative.',
            'Extra failed attempts, longer recordings, future periodic production snapshots and unrelated studies are outside the remaining-work estimate.',
            'Two largest measured checkpoint sizes are added as temporary save headroom for two workers; this does not authorize concurrent GPU work.',
            'No existing artifact was removed, compressed or changed.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--probe-reference',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    result=estimate(args.study,args.probe_reference);args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('completed_fits','remaining_fits','additional','current_free_bytes',
        'fixed_fixture_bytes','two_worker_transient_save_allowance_bytes','required_free_upper_with_reserve_bytes')}))


if __name__=='__main__':main()
