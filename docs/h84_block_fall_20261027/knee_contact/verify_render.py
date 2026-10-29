"""Inspect saved illustration poses and images; no stepping or learning."""
from pathlib import Path
import json
import numpy as np
import mujoco
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
EV=ROOT/'Saved/h84/block_fall_design_20261027/evidence/knee_contact'
j=json.loads((EV/'motion.json').read_text());r=j['records']
m=mujoco.MjModel.from_binary_path(str(ROOT/'Saved/h84/whole_body_reach_document_20261026/tmp/model.mjb'))
d=mujoco.MjData(m)
checks=[]
for x in r:
    if not j['landing']<=x['time']<5.5:continue
    d.qpos[:]=x['qpos'];mujoco.mj_forward(m,d)
    for side in ['left','right']:
        gs=[g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in [side+'_upper_leg',side+'_lower_leg']]
        g=min(gs,key=lambda g:mujoco.mj_geomDistance(m,d,g,0,2,None))
        line=np.zeros(6);gap=mujoco.mj_geomDistance(m,d,g,0,2,line)
        knee=d.xpos[m.body(side+'_lower_leg').id]
        checks.append({'time':x['time'],'side':side,'surface_gap_m':gap,'surface_point':line[:3].tolist(),'knee_center':knee.tolist(),'surface_horizontal_offset_m':float(np.linalg.norm(line[:2]-knee[:2]))})
decoded=0
for k in range(361):
    for view in ['side','front']:
        with Image.open(Path(__file__).parent/f'block_{view}_{k:03d}.webp') as im:im.load()
        decoded+=1
summary={'images_decoded':decoded,'supported_knee_samples':len(checks),'max_knee_surface_gap_m':max(x['surface_gap_m'] for x in checks),'min_knee_surface_gap_m':min(x['surface_gap_m'] for x in checks),'max_surface_horizontal_offset_from_knee_m':max(x['surface_horizontal_offset_m'] for x in checks),'max_supported_knee_horizontal_delta_m':max(float(np.linalg.norm(np.array(b['left_lower_leg'])[:2]-np.array(a['left_lower_leg'])[:2])) for a,b in zip(r,r[1:]) if j['landing']<=a['time'] and b['time']<5.5),'min_floor':j['min_floor'],'min_block':j['min_block'],'records':checks}
assert summary['min_floor']>=0 and summary['min_block']>=0
assert summary['max_knee_surface_gap_m']<.001
assert summary['max_surface_horizontal_offset_from_knee_m']<.005
assert summary['max_supported_knee_horizontal_delta_m']<.001
summary['verdict']='PASS'
(EV/'knee_contact_verification.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='records'}))
