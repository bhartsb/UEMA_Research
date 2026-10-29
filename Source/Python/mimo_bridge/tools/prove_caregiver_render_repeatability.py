# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Separate exact copied physics from fresh-renderer pixel repeatability."""
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import time

import numpy as np

from mimo_bridge import continuation_state
from mimo_bridge.datasource import SingleEnvSource
from .caregiver_learning_trial import write
from .prepare_local_shadow import state_digest


def physical_render_state(env):
    values={'time':np.array(env.data.time)}
    for name in ('qpos','qvel','qacc_warmstart','ctrl','act','mocap_pos','mocap_quat',
                 'xpos','xquat','xmat','cam_xpos','cam_xmat','geom_xpos','geom_xmat',
                 'light_xpos','light_xdir'):
        if hasattr(env.data,name):values['data_'+name]=np.array(getattr(env.data,name))
    for name in ('geom_rgba','geom_contype','geom_conaffinity','geom_size','geom_pos','geom_quat',
                 'geom_friction','geom_solref','body_pos','actuator_gainprm','actuator_forcerange',
                 'cam_fovy','cam_pos','cam_quat','light_pos','light_dir','light_diffuse','light_ambient',
                 'light_specular','tex_data','tex_rgb','mat_rgba','mat_emission','mat_specular','mat_shininess'):
        if hasattr(env.model,name):values['model_'+name]=np.array(getattr(env.model,name))
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=10)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    saved=continuation_state.load(args.checkpoint);worlds=saved['worlds'];order=saved['trainer']['held']
    if len(worlds)!=1:raise ValueError('This diagnostic requires one saved body')
    write(args.out/'declaration.json',dict(checkpoint=str(args.checkpoint),steps=args.steps,
        episode_sha256=continuation_state.file_sha256(args.checkpoint/'episode_state.pt.gz'),
        action='The same saved held body order and fixed zero gaze in each branch; auditor command only.',
        physics_rule='All saved physical, camera, light, texture and geometry fields match exactly immediately before each rendering.',
        pixels_rule='All actual pixel patches and angles match exactly; any difference is reported without tolerance substitution.',
        scope='Two fresh sources start from one exact physical cut-point. No learner, optimizer update, reward or skill claim.'))
    results=[];started=time.time()
    for arm in ('a','b'):
        folder=args.out/arm;folder.mkdir();src=SingleEnvSource(**worlds[0]['env_kwargs'])
        records=[]
        try:
            src.restore_full_state(worlds);original=src.env.render_eyes
            def capture_before_render():
                state=physical_render_state(src.env);records.append(state)
                return original()
            src.env.render_eyes=capture_before_render
            values=[]
            for i in range(args.steps):
                frames=src.step(order,np.zeros((1,2)),render=True);frame=frames[0]
                state=records[-1]
                np.savez_compressed(folder/f'frame_{i}.npz',**state,pixels=frame.vision_patches,angles=frame.vision_angles)
                values.append(dict(physical={k:state_digest(v) for k,v in state.items()},
                    pixels=state_digest(frame.vision_patches),angles=state_digest(frame.vision_angles)))
            results.append(values)
        finally:
            fovea=src.env._fovea_renderer
            src.close()
            if fovea is not None:fovea.close()
            del src;gc.collect()
    rows=[]
    for i,(a,b) in enumerate(zip(*results)):
        x=np.load(args.out/'a'/f'frame_{i}.npz');y=np.load(args.out/'b'/f'frame_{i}.npz')
        changed=[k for k in a['physical'] if a['physical'][k]!=b['physical'][k]]
        delta=np.abs(x['pixels'].astype(float)-y['pixels'].astype(float))
        rows.append(dict(step=i,physical_fields_different=changed,
            physical_max_abs={k:float(np.abs(x[k].astype(float)-y[k].astype(float)).max()) for k in changed if x[k].size},
            pixel_values_different=int((delta!=0).sum()),pixel_max_abs=float(delta.max()),
            angles_exact=a['angles']==b['angles']))
    write(args.out/'report.json',dict(physical_exact=all(not row['physical_fields_different'] for row in rows),
        pixels_exact=all(row['pixel_values_different']==0 and row['angles_exact'] for row in rows),
        frames=rows,elapsed_seconds=time.time()-started,learning_updates=0,capability_pass=False))
    print(json.dumps(json.loads((args.out/'report.json').read_text()),indent=2))


if __name__=='__main__':main()
