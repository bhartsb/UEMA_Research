# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A small purpose-built MuJoCo world the observer's recorder can read, for tests that must not build
the crib (a few megabytes, no torch): a head with two eye cameras looking along +x at a 5 cm toy pad
37 cm away, a 6 cm box of 50 g, and a thumb and an index finger on slides either side of the box,
driven by position actuators. The box stands on the floor with the digits 7.5 cm from its centre, or
(lifted=True) hangs 20 cm above the floor between the digits, which start touching its faces: nothing
but their squeeze holds it up. `world()` returns (mujoco, model, data, env) where env carries the
attributes WorldRecorder reads from BridgeEnv (proprio handles, patches, the eye movers)."""
from types import SimpleNamespace

import numpy as np

XML = """<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="head" pos="0 0 0.3">
      <geom type="sphere" size="0.05" contype="0" conaffinity="0"/>
      <body name="left_eye" pos="0.03 0 0">
        <geom type="sphere" size="0.008" contype="0" conaffinity="0"/>
        <camera name="eye_left" pos="0.01 0 0" xyaxes="0 -1 0 0 0 1" fovy="60"/>
      </body>
      <body name="right_eye" pos="0.03 0 0">
        <geom type="sphere" size="0.008" contype="0" conaffinity="0"/>
        <camera name="eye_right" pos="0.01 0 0" xyaxes="0 -1 0 0 0 1" fovy="60"/>
      </body>
    </body>
    <body name="box_light" pos="0.30 0 %(z)s">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.03 0.03 0.03" mass="0.05"/>
    </body>
    <body name="target" mocap="true" pos="0.42 0.0 0.3">
      <geom name="toy_pad" type="box" size="0.01 0.025 0.025" contype="0" conaffinity="0"/>
    </body>
    <body name="right_thdistal" pos="0.30 -%(y)s %(z)s">
      <joint name="thumb_slide" type="slide" axis="0 1 0" damping="2"/>
      <geom type="capsule" fromto="0 0 -0.01 0 0 0.01" size="0.01" mass="0.02"/>
    </body>
    <body name="right_ffdistal" pos="0.30 %(y)s %(z)s">
      <joint name="index_slide" type="slide" axis="0 1 0" damping="2"/>
      <geom type="capsule" fromto="0 0 -0.01 0 0 0.01" size="0.01" mass="0.02"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="thumb_slide" kp="40"/>
    <position joint="index_slide" kp="40"/>
  </actuator>
</mujoco>"""


def world(lifted: bool = False):
    """(mujoco, model, data, env) of the tiny world; env stands in for BridgeEnv."""
    import mujoco
    m = mujoco.MjModel.from_xml_string(XML % (dict(z="0.20", y="0.04") if lifted else dict(z="0.031", y="0.075")))
    d = mujoco.MjData(m)
    own = [m.body(n).id for n in ("head", "left_eye", "right_eye", "right_thdistal", "right_ffdistal")]
    env = SimpleNamespace(model=m, data=d, dt=0.01, frame=0, scene_kind="tiny", fovea_fov=24.0,
                          _own_body_ids=own, _root_body_id=m.body("head").id,
                          proprio=SimpleNamespace(handle_names=["thumb", "index"],
                                                  body_ids=np.asarray(own[3:], dtype=np.int64)),
                          pooler=SimpleNamespace(specs=[]), caregiver=None, saccades=None, reflex=None)
    env.patch_names = lambda: []
    return mujoco, m, d, env
