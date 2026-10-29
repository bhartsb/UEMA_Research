"""Rebuilds plan_pictures.js (the numbers the page script reads) from plan_pictures.json in the pictures folder.
Run from anywhere: python docs/h84_reach_plan_visual_20261023/page_build/rebuild_js.py"""
import json, os
root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # the pictures folder
P = json.load(open(os.path.join(root, 'plan_pictures.json')))
r1 = lambda x: round(x, 1)
N = P['numbers']; base = N['baseline_8start']['bodies']
g = lambda d: round(d['gravity']['angle_from_line_of_sight_deg']) if d.get('gravity') else None
def grange(pics, names):
    a = [pics[n]['gravity']['angle_from_line_of_sight_deg'] for n in names if pics.get(n) and pics[n].get('gravity')]
    return dict(min=round(min(a)), max=round(max(a)), n=len(a))
seqg = N['sequence_body1']['gravity']
out = {'pairs': {b: dict(start_step=v['start_step'], target_step=v['target_step'], L=r1(v['D_start_cm']['L']), R=r1(v['D_start_cm']['R']), grav=g(v)) for b, v in N['pairs'].items() if 'start_step' in v},
       'done_b1_grav': g(N['pairs'].get('done_b1', {})), 'target_crop_grav': round(N['sequence_body1']['target_crop_gravity']['angle_from_line_of_sight_deg']),
       'sequence': [dict(step=f['step'], what=f['what'].replace(' (drawn with the bout off)', ''), L=r1(f['D_cm']['L']), R=r1(f['D_cm']['R']), grav=g(f)) for f in N['sequence_body1']['frames']],
       'sequence_grav': dict(min=round(seqg['angle_from_line_of_sight_deg_min']), max=round(seqg['angle_from_line_of_sight_deg_max'])),
       'limb': {f['picture']: dict(ms=f['ms'], L=r1(f['D_cm']['L']), R=r1(f['D_cm']['R']), grav=g(f)) for f in N['limb_control']['frames']},
       'b3': {k: dict(step=v['model_step'], L=r1(v['D_cm']['L']), R=r1(v['D_cm']['R']), grav=g(v)) for k, v in N['body3_POSE_A_off_494025'].items() if k.endswith('.png')},
       'cmp': {k: dict(L=r1(v['D_cm']['L']), R=r1(v['D_cm']['R']), grav=g(v)) for k, v in N['comparison_483120']['pictures'].items() if 'D_cm' in v},
       'baseline': {b: dict(target_record=v['target_record'], grav=grange(v['pictures'], ('start', 'T', 'D60', 'P60')), at={c: {k: dict(L=x['L'], R=x['R'], E=x['E']) for k, x in v['at'][c].items()} for c in ('P', 'D')}, flags=v['flags']) for b, v in base.items()},
       'b5strip_grav': grange(base['5']['pictures'], ('T', 'start', 'D5', 'D20', 'D60', 'P60')),
       'region_grav': g(base['5']['pictures']['region']),
       'target_step_body1': N['sequence_body1']['target_step'],
       'gravity_min_max': P['gravity'].get('angle_from_line_of_sight_deg_min_max')}
open(os.path.join(root, 'plan_pictures.js'), 'w').write('window.H84_PLAN=' + json.dumps(out) + ';\n')
print('js ok', out['gravity_min_max'], {b: v['grav'] for b, v in out['baseline'].items()}, 'seq', out['sequence_grav'], 'b5strip', out['b5strip_grav'], 'region', out['region_grav'])
