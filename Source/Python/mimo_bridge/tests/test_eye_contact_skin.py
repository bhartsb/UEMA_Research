# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A contact on an eye must not end the run (2026-09-18).

MIMo gives the two eyes touch sensors, so the environment's contact collector admits
them, but the bridge's touch map (touch_adapter.py: 53 skin bodies; the eyes are
placeholders above SKIN_SCALE_MAX) has no patch for them. Under the mechanical_v4
profile every contact record goes to the virtual skin, which indexes by body: a hand or
object on an eye raised KeyError inside a physics worker and ended the run. Measured
in the H84 launch smokes of 2026-09-18: two runs died at steps 8 and 10 on body 46
(right_eye). The skin now drops such records and counts them; a skin body in the same
batch is still advanced.
"""
import numpy as np

from mimo_bridge.env import BridgeEnv
from mimo_bridge.touch_adapter import ContactInfo


def _contact(env, body_id, other_geom):
    m, d = env.model, env.data
    geom = next(g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == body_id)
    pos_world = np.asarray(d.geom_xpos[geom], dtype=float).copy()
    rot = np.asarray(d.xmat[body_id]).reshape(3, 3)
    pos_body = rot.T @ (pos_world - np.asarray(d.xpos[body_id]))
    return ContactInfo(body_id=body_id, pair_key=(int(geom), int(other_geom)), normal_force=2.0,
                       tangential_body=np.zeros(3), pos_body=pos_body, pos_world=pos_world,
                       normal_world=np.array([0.0, 0.0, -1.0]), penetration_m=1e-4, rel_normal_speed=0.0,
                       other_body_name="floor", rel_tangent_speed=0.0, friction_power_w=0.0)


def test_an_eye_contact_is_counted_and_dropped_and_a_skin_body_still_advances(tmp_path):
    env = BridgeEnv(scene="crib", run_dir=str(tmp_path), vision=False, tag="_eyeskin", age=24,
                    contact_profile="mechanical_v4", caregiver_seconds=0, kp=6.0,
                    body_layout="trunk_v1", cmd_horizon=25)
    try:
        env.reset()
        m = env.model
        eye = m.body("right_eye").id
        skin = env._virtual_skin
        assert env.touch.has_sensors(eye), "the premise: MIMo senses touch on the eye"
        assert eye not in skin.slices, "the premise: the touch map has no patch for the eye"
        hand = next(s.body_id for s in env.pooler.specs if "hand" in s.body_name)
        floor = next(g for g in range(m.ngeom) if m.geom(g).name in ("floor", "ground")) if any(
            m.geom(g).name in ("floor", "ground") for g in range(m.ngeom)) else 0
        env.touch.get_touch_obs()
        env.data.time += float(m.opt.timestep)      # past the skin's last advance (reset already ran one)
        contacts = [_contact(env, eye, floor), _contact(env, hand, floor)]
        env._advance_virtual_skin(contacts)          # raised KeyError: 46 before the fix
        assert env._skin_unmapped_contacts == 1
        assert skin.last_time == float(env.data.time), "the skin body's record was not advanced"
        # a second eye contact at a later time is counted again, never raised
        env.data.time += float(m.opt.timestep)
        env._advance_virtual_skin([_contact(env, eye, floor)])
        assert env._skin_unmapped_contacts == 2
    finally:
        env.close() if hasattr(env, "close") else None
