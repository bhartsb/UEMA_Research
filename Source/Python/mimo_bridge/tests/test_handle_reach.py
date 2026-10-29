# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""How far an order may pull a handle: its own reach, not one number for every handle alike.

The decoder turns "put this handle here" into joint movement, and the wanted move used to be
clipped to 50 cm whatever the handle was. A wrist's chain includes the whole arm and reaches
42.85 cm, so 50 cm is about right for a wrist. A finger has three or four small hinges and
reaches 5.48 to 6.80 cm, so 50 cm was nine times what the finger can do, and an unreachable
finger target went into the least squares at its full size, swamping the orientation rows it
shares the solve with.

Measured on the live run's own snapshots, 47 of them spanning steps 88,725 to 262,057: the
model orders a fingertip to a place 10.0 to 12.5 cm away while the fingertip sits 2.1 to 4.0 cm
from its own wrist, and the fingers sit at the end of their range - the middle angle of a
finger against its wrist reads 150 to 165 degrees.
"""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mimo_bridge.action_decoder import ActionDecoder, POSITION_HANDLES, _measure_reach  # noqa: E402

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

TINY = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="root" pos="0 0 0">
      <joint name="j0" type="hinge" axis="0 1 0" range="-1 1"/>
      <geom type="capsule" fromto="0 0 0  0.30 0 0" size="0.02"/>
      <body name="mid" pos="0.30 0 0">
        <joint name="j1" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0  0.02 0 0" size="0.005"/>
        <body name="tip" pos="0.02 0 0">
          <joint name="j2" type="hinge" axis="0 1 0" range="-1 1"/>
          <geom type="sphere" size="0.004"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j0"/><motor joint="j1"/><motor joint="j2"/>
  </actuator>
</mujoco>
"""


def test_a_long_chain_reaches_further_than_a_short_one():
    """The measure is the model's own geometry, so a part hanging off a 30 cm arm can be
    carried much further than one hanging off a 2 cm stub. The joint ranges here are radians,
    which the scene says: read as degrees the same model gives a 30 cm arm a reach of 1.05 cm,
    and reading the model's own units is half of what this measure is for."""
    m = mujoco.MjModel.from_xml_string(TINY)
    far = _measure_reach(m, int(m.body("mid").id), [int(m.joint("j0").id)])
    near = _measure_reach(m, int(m.body("tip").id), [int(m.joint("j2").id)])
    assert far > 0.3, "a part on a 30 cm arm read a reach of only %.3f m" % far
    assert near <= 0.02, "a part on its own hinge read a reach of %.3f m" % near
    assert far > 10 * near


def test_a_handle_with_no_joints_keeps_the_old_fifty_centimetres():
    """Nothing is measured, so nothing is claimed: the behaviour is what it was."""
    m = mujoco.MjModel.from_xml_string(TINY)
    assert _measure_reach(m, int(m.body("tip").id), []) == 0.5


def test_a_chain_with_more_joints_than_can_be_walked_keeps_the_old_fifty_centimetres():
    m = mujoco.MjModel.from_xml_string(TINY)
    js = [int(m.joint("j0").id)] * 11
    assert _measure_reach(m, int(m.body("tip").id), js, max_joints=10) == 0.5


def test_the_measure_never_widens_the_old_clip_and_never_reads_zero():
    """It can only tighten: a handle the body can swing further than half a metre is still
    held to half a metre, and a handle that cannot move at all is still given a centimetre so
    a direction is never divided away."""
    m = mujoco.MjModel.from_xml_string(TINY)
    for body, joints in (("mid", [int(m.joint("j0").id)]),
                         ("tip", [int(m.joint("j2").id)]),
                         ("root", [int(m.joint("j0").id)])):
        r = _measure_reach(m, int(m.body(body).id), joints)
        assert 0.01 <= r <= 0.5, "%s read %.4f m" % (body, r)


def test_measuring_does_not_disturb_the_state_it_is_handed():
    """It walks the joints to the ends of their ranges, so it must do that on a copy."""
    m = mujoco.MjModel.from_xml_string(TINY)
    d = mujoco.MjData(m)
    d.qpos[:] = [0.11, 0.22, 0.33]
    mujoco.mj_forward(m, d)
    before = np.array(d.qpos), np.array(d.xpos)
    _measure_reach(m, int(m.body("tip").id), [int(m.joint(n).id) for n in ("j0", "j1", "j2")])
    assert np.array_equal(d.qpos, before[0]) and np.array_equal(d.xpos, before[1])


@pytest.mark.slow
def test_on_the_real_body_a_finger_is_held_to_a_fingers_reach():
    """The numbers this was built for, taken off the crib body at 24 months."""
    import os
    import tempfile
    from mimo_bridge.env import BridgeEnv
    out = os.path.join(tempfile.gettempdir(), "mimo_handle_reach_test")
    os.makedirs(out, exist_ok=True)
    env = BridgeEnv(scene="crib", run_dir=out, age=24.0, vision=False, mobile=False,
                    critters=False, caregiver_mode="off", body_layout="trunk_v1",
                    contact_profile="mechanical_v4", seed=0, tag="_reachtest")
    env.reset()
    dec = env.decoder
    reach = {n: h.reach_m for n, h in zip(dec.handle_names, dec.handles)}
    fingers = [v for n, v in reach.items()
               if n.split("_")[0] in FINGERS and n in POSITION_HANDLES]
    wrists = [v for n, v in reach.items() if n.startswith("wrist")]
    assert fingers and wrists
    assert max(fingers) < 0.10, (
        "a finger was allowed to be pulled %.3f m, more than a finger can travel" % max(fingers))
    assert min(wrists) > 0.30, (
        "a wrist was held to %.3f m, far less than an arm can travel" % min(wrists))
    assert min(wrists) > 4 * max(fingers), (
        "a wrist and a finger were given comparable reach: %.3f against %.3f"
        % (min(wrists), max(fingers)))
    # and the number the old code used for every handle alike sits above every one of them
    assert all(v <= 0.5 for v in reach.values())
