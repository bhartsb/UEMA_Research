# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Publish plain-language, hash-bound reviews of completed copied measurements.

This never rewrites experimental records or changes a declared pass criterion.
Physical observations are described separately from acquired abilities.
"""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import time
from mimo_bridge.tools.experiment_board import read


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def review(folder,identifier,prior_review=None):
    report=read(folder/'report.json');declaration=read(folder/'declaration.json')
    from .evidence_identity import checkpoint_identity
    checkpoint=Path(declaration['checkpoint'])
    identity=checkpoint_identity(checkpoint,declaration.get('checkpoint_sha256') or
        (prior_review or {}).get('model_identity',{}).get('model_sha256'))
    worlds=report['results'];count=len(worlds)
    if declaration.get('instrument_smoke') is not False or count!=declaration['worlds'] or count<1:
        raise ValueError('An incomplete or instrument-only test cannot become a behavior observation')
    if sorted(w.get('seed',w.get('world_seed')) for w in worlds)!=sorted(declaration['seeds']) or len(set(declaration['seeds']))!=count:
        raise ValueError('Completed world identities differ from the declared independent worlds')
    exact=all(w.get('same_action_replay_exact') and
        (w.get('complete_model_replay_exact') or w.get('complete_model_replay',{}).get('complete_model_replay_exact')) for w in worlds)
    if not exact:raise ValueError('This review only describes fully replay-verified copied measurements')
    measurements={}
    if 'modalities' in report:
        for name,row in report['modalities'].items():
            measurements[name]=dict(verdict=row['verdict'],
                mean_action_effect_error_removed=statistics.mean(row['world_values']['action_effect_error_removed']),
                worlds_beating_unchanged_state=sum(x>0 for x in row['world_values']['persistence_advantage']),
                worlds=row['worlds'])
        failed=[name for name,row in measurements.items() if row['verdict']=='COMPARISON_NOT_PASSED']
        if len(failed)==3:
            summary=(f'All three next-model-step prediction tests failed across {count} worlds. '
                'Vision and proprioception forecasts lost to assuming no change in every world.'
                if all(measurements[n]['worlds_beating_unchanged_state']==0 for n in ('vision','proprioception')) else
                f'All three next-model-step prediction tests failed across {count} worlds.')
        else:
            summary=f'Native prediction measured in {count} worlds. Results are listed separately for each sense; this is not a learned-skill verdict.'
        decision='Repair the action-to-sensory-consequence lesson and local contact information; qualify any replacement chooser on retained physical outcomes.'
        limitation='This is the actual 50 ms sensory forecast. The held command lasts 250 ms; this result must not be described as a 250 ms or one-second open-loop forecast.'
    else:
        profile=declaration['profile']
        if profile not in ('toy','body_floor','free_objects','support'):raise ValueError('Unsupported physical observation profile')
        expected={(condition,arm) for condition in (('present','relocated','removed','interrupted') if profile=='toy' else ('present',))
                  for arm in ('candidate','baseline')}
        for world in worlds:
            actual=[(row['condition'],row['arm']) for row in world['conditions']]
            if set(actual)!=expected or len(actual)!=len(expected):raise ValueError('Physical comparison conditions are incomplete or duplicated')
            if any(row['scored_seconds']<=0 for row in world['conditions']):raise ValueError('Physical observation has no scored experience')
        rows=[r for w in worlds for r in w['conditions'] if r.get('arm')=='candidate']
        for condition in sorted({r['condition'] for r in rows}):
            selected=[r for r in rows if r['condition']==condition]
            measurements[condition]=dict(target_center_fraction=statistics.mean(r['target_center_in_fovea_fraction'] for r in selected),
                longest_target_hold_s=max(r['longest_foveal_dwell_s'] for r in selected),
                maximum_torso_tilt_deg=max(math.degrees(math.acos(max(-1,min(1,r['minimum_front_up'])))) for r in selected),
                minimum_world_max_travel_m=min(r['maximum_horizontal_displacement_m'] for r in selected),
                maximum_world_max_travel_m=max(r['maximum_horizontal_displacement_m'] for r in selected))
        if profile=='toy':
            stationary=measurements['present'];relocated=measurements['relocated']
            summary=(f'The stationary toy stayed centered for {100*stationary["target_center_fraction"]:.2f}% of measured model steps; '
                f'the moved toy for {100*relocated["target_center_fraction"]:.2f}%. '
                f'The longest stationary-toy hold was {stationary["longest_target_hold_s"]:.2f} seconds.')
            decision='Prioritize target selection and holding an achieved look; use the same moved/removed/interrupted controls to check the repair.'
            held=statistics.mean(row['target_center_in_fovea_fraction'] for world in worlds for row in world['conditions']
                if row['condition']=='present' and row['arm']=='baseline')
            limitation=(f'{count} short worlds, frozen copied brain, reflexes and forced stillness off. '
                f'The initial pose-hold control keeps the stationary toy centered in {100*held:.2f}% of model steps. '
                'Zero visible-target score when the toy is absent is automatic, not proof that purposeful looks stopped. No acquisition or retention pass.')
        elif profile=='body_floor':
            m=measurements['present']
            summary=(f'Across {count} body-and-floor worlds, the torso tilted at most {m["maximum_torso_tilt_deg"]:.1f} degrees '
                f'and the body traveled {100*m["minimum_world_max_travel_m"]:.2f}–{100*m["maximum_world_max_travel_m"]:.2f} cm. '
                'No whole-body roll occurred in these measured windows.')
            decision='Finish a physically valid full-roll reference and improve retained action-effect learning; do not treat limb motion or pen travel as rolling.'
            duration=min(row['scored_seconds'] for row in rows)
            limitation=(f'At least {duration:.2f} scored seconds per world without adult direction. Tilt includes pitch; '
                'a short unsuccessful sample does not prove the body can never roll. Physical reachability needs its own executed reference.')
        else:
            name='free_objects' if profile=='free_objects' else 'support'
            traces=[read(folder/str(w['seed'])/(name+'.json')) for w in worlds]
            contact=sum(any(t['measurements']['hand_object_contact']) for t in traces)
            lift=max(max(t['measurements']['object_lift_m']) for t in traces)
            feet=sum(any(t['measurements']['feet_support_only']) for t in traces)
            measurements['actual_object_contact']=dict(worlds_with_hand_contact=contact,maximum_lift_m=lift,
                worlds_with_feet_only_support=feet)
            summary=(f'The hands contacted the selected free object in {contact} of {count} worlds; '
                f'its greatest lift was {100*lift:.2f} cm.' if profile=='free_objects' else
                f'The hands contacted the support in {contact} of {count} worlds. Feet-only body support occurred in {feet} worlds.')
            decision=('Verify a reachable grasping situation, then measure whether the body learns contact, grasp and release.' if profile=='free_objects' else
                'Calibrate an actually reachable handhold and body lift before using this scene to judge pulling to stand.')
            limitation='The traces establish these contact and posture observations. The scene still lacks a verified ability-specific reachability reference; it cannot certify acquisition, retention or transfer.'
    profile='action_forecast' if 'modalities' in report else declaration['profile']
    return dict(report_sha256=hashlib.sha256((folder/'report.json').read_bytes()).hexdigest(),
        declaration_sha256=hashlib.sha256((folder/'declaration.json').read_bytes()).hexdigest(),
        profile=profile,model_identity=dict(identity,
            captured='Declared before the test' if declaration.get('checkpoint_sha256') else
                'SHA-256 of the preserved checkpoint at review; the older physical declaration recorded its path only'),
        state='insufficient',summary=summary,decision=decision,measurement=measurements,
        limitation=limitation,reviewed_at=time.time(),report_path=str(folder/'report.json'),
        checkpoint=declaration.get('checkpoint'),worlds=count,complete_replay_verified=exact,
        capability_pass=False,job_id=identifier)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());args=p.parse_args()
    root=args.root.resolve();queue=read(root/'Saved/mimo_bridge/developmental_queue.json')
    config=read(root/'docs/development_dashboard/work.json');reviews=[]
    for identifier,state in queue['jobs'].items():
        if not identifier.startswith('development-v3-') or state['state']!='complete':continue
        relative=Path('docs/experiment_review_2026-09-13')/(identifier+'.json')
        prior=read(root/relative) if (root/relative).exists() else None
        folder=Path(state['attempts'][-1]['out']);result=review(folder,identifier,prior)
        write(root/relative,result);config.setdefault('reviews',{})[identifier]=relative.as_posix();reviews.append(result)
    latest=next((r for r in reversed(reviews) if 'action-forecast' in r['job_id']),None)
    if latest:config['focus_job']=latest['job_id']
    config['headline']='The measured movements still fall short of purposeful skills.'
    config['change']='Fresh controlled tests now show the problem directly: brief toy looks, small posture changes, and no hand contact with the tested free object or support. Copied learning comparisons are separate from these observations.'
    write(root/'docs/development_dashboard/work.json',config)
    write(root/'docs/experiment_review_2026-09-13/completed_developmental_reviews.json',dict(reviewed_at=time.time(),reviews=reviews))
    cards=''.join('<article><h2>'+html.escape(r['job_id'])+'</h2><p><strong>'+html.escape(r['summary'])+'</strong></p><p>'+html.escape(r['decision'])+'</p><p class="limit">'+html.escape(r['limitation'])+'</p></article>' for r in reversed(reviews))
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>What the completed behavior tests show</title><style>body{font:18px/1.55 system-ui;max-width:1000px;margin:40px auto;padding:24px;color:#182b36;background:#f5f3ee}h1,h2{line-height:1.2}h2{font-size:19px;overflow-wrap:anywhere}article{background:white;padding:22px;margin:18px 0;border-left:5px solid #ad6b28}.limit{font-size:15px;color:#435967}</style><h1>What the baby actually did in the completed tests</h1><p>These are observations from frozen copied brains in controlled worlds. Each test restores the complete brain and physics between conditions. No new skill is awarded for a repaired instrument or for a low prediction error.</p>'+cards+'</html>'
    (root/'docs/experiment_review_2026-09-13/completed_behavior_review.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(reviews=len(reviews),focus=config['focus_job'])))


if __name__=='__main__':main()
