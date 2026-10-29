# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Independent MuJoCo checks of force, momentum, contact velocity and resolution."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco as mj
from mimo_bridge.contact_mechanics import G


def trial(dt, drop):
    model=mj.MjModel.from_xml_string(f'''<mujoco><option timestep="{dt}" gravity="0 0 -{G}" integrator="Euler"/>
      <worldbody><geom type="plane" size="2 2 .1" condim="3"/>
      <body pos="0 0 {'.4' if drop else '.1'}"><freejoint/><geom type="sphere" size=".1" mass="2"
        solref=".01 1" solimp=".99 .99 .001"/></body></worldbody></mujoco>''')
    data=mj.MjData(model);mj.mj_forward(model,data)
    impulse=np.zeros(3);forces=[];mass=float(model.body_mass.sum())
    v0=data.qvel[:3].copy();duration=2.
    force6=np.zeros(6)
    for i in range(round(duration/dt)):
        mj.mj_step(model,data)
        total=np.zeros(3)
        for j in range(data.ncon):
            c=data.contact[j];mj.mj_contactForce(model,data,j,force6)
            world=np.asarray(c.frame).reshape(3,3).T@force6[:3]
            total += world if int(model.geom_bodyid[c.geom2]) == 1 else -world
        impulse+=total*dt;forces.append(total[2])
    predicted_impulse=mass*(data.qvel[:3]-v0)-mass*np.array([0,0,-G])*duration
    static=float(np.mean(forces[-round(.25/dt):]))
    err=float(np.linalg.norm(impulse-predicted_impulse))
    return dict(dt_s=dt,mass_kg=mass,weight_n=mass*G,settled_support_n=static,
                impulse_ns=impulse.tolist(),momentum_balance_ns=predicted_impulse.tolist(),
                impulse_error_ns=err,peak_contact_force_n=float(max(forces)))


def run(out):
    results={f'{kind}_{dt}':trial(dt,kind=='drop') for kind in ('rest','drop') for dt in (.0005,.00025)}
    for r in results.values():
        if abs(r['settled_support_n']-r['weight_n']) > 1e-5 or r['impulse_error_ns'] > .05:
            raise RuntimeError(f'force conservation failed: {r}')
    # Relative peak sensitivity is reported: force conservation alone does not
    # prove that a contact peak is resolved or that compliance is biological.
    peak_change=abs(results['drop_0.0005']['peak_contact_force_n']/results['drop_0.00025']['peak_contact_force_n']-1)
    if peak_change > .05: raise RuntimeError(f'peak not converged under timestep halving: {peak_change}')
    result=dict(status='passed_mechanical_checks',trials=results,peak_relative_change_halved_dt=peak_change,
                scope='Simple known-mass body. Does not validate MIMo tissue contact area or child injury.')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8');return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    print(json.dumps(run(a.out),indent=2))
