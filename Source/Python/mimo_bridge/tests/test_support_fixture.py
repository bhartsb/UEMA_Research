# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import pytest
from mimo_bridge.support_fixture import add_to_scene,NAME,RAIL


def test_anchored_rail_is_visible_colliding_and_carries_a_measured_load(tmp_path):
    path=tmp_path/'world.xml'
    path.write_text('''<mujoco><option timestep=".001"/><worldbody>
      <body name="load" pos="0 0 .24"><freejoint/>
        <geom type="sphere" size=".02" mass=".5"/></body>
      </worldbody></mujoco>''')
    add_to_scene(path)
    tree=ET.parse(path);body=tree.getroot().find(f"worldbody/body[@name='{NAME}']")
    body.set('pos','0 0 .2');tree.write(path)
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    rail=model.geom(RAIL).id;mid=int(model.body_mocapid[model.body(NAME).id])
    start=data.mocap_pos[mid].copy()
    load=model.body('load').id;data.xfrc_applied[load,2]=-5.
    for _ in range(1200):mujoco.mj_step(model,data)
    forces=[]
    for i in range(data.ncon):
        c=data.contact[i]
        if rail in (c.geom1,c.geom2):
            f=np.zeros(6);mujoco.mj_contactForce(model,data,i,f);forces.append(f[0])
    assert forces
    assert sum(forces)==pytest.approx(5+.5*9.81,rel=.02)
    np.testing.assert_array_equal(data.mocap_pos[mid],start)
    assert model.geom_rgba[rail,3]==1.
    assert model.geom_contype[rail]>0
    with pytest.raises(ValueError,match='Duplicate'):
        add_to_scene(path)
