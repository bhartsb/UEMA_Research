# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Visible adult geometry must preserve physics, speech timing and yoked replay."""
from pathlib import Path
import copy
import numpy as np
import pytest

from mimo_bridge.caregiver_visual import ADULT_BODIES, VERSION, facing, speech_opening
from mimo_bridge.speech import SpeechStream


def test_lips_follow_only_emitted_speech_without_consuming_it():
    s=SpeechStream();s.start(np.r_[np.zeros(160),np.full(160,.10),np.zeros(160)], [0,0,0])
    assert speech_opening(s,.01)==0
    s.cursor=160
    assert .7<speech_opening(s,.01)<.9
    assert s.cursor==160 and s.fraction==0
    s.cursor=320
    assert speech_opening(s,.01)==0
    s.cursor=len(s.clip)
    assert speech_opening(s,.01)==0


@pytest.fixture
def env(tmp_path):
    from mimo_bridge.env import BridgeEnv
    speech=Path(__file__).resolve().parents[4]/'Saved/mimo_bridge/caregiver_speech_exchange_v1_final'
    e=BridgeEnv(scene='crib',vision=False,audio_real=True,mobile=True,age=24,
        caregiver_mode='responsive',caregiver_speech_dir=str(speech),run_dir=str(tmp_path))
    e.reset()
    try:yield e
    finally:e.close()


def test_adult_geometry_is_connected_faces_child_and_never_collides(env):
    e=env;m=e.model;d=e.data;c=e.caregiver;mj=e._mj
    e.step();mj.mj_forward(m,d)
    body_ids={m.body(n).id for n in ADULT_BODIES}
    gids=np.flatnonzero(np.isin(m.geom_bodyid,list(body_ids)))
    assert np.all(m.geom_contype[gids]==0) and np.all(m.geom_conaffinity[gids]==0)
    head=d.mocap_pos[c.ids['caregiver_head']];eye=c.state['anchor']
    rotation=d.xmat[m.body('caregiver_head').id].reshape(3,3)
    np.testing.assert_allclose(-rotation[:,1],(eye-head)/np.linalg.norm(eye-head),atol=1e-8)
    for side in ('l','r'):
        gid=m.geom('caregiver_forearm_'+side).id
        axis=d.geom_xmat[gid].reshape(3,3)[:,2]
        endpoint=d.geom_xpos[gid]+axis*m.geom_size[gid,1]
        np.testing.assert_allclose(endpoint,d.mocap_pos[c.ids['caregiver_hand_'+side]],atol=1e-8)
    # Audio originates at the same mouth visible in the scene.
    np.testing.assert_allclose(e._audio.speech.position,head+rotation@[0.,-.074,-.044])


def test_mouth_and_arm_transforms_survive_episode_restore(env):
    e=env;e._audio.speech.start(np.full(1000,.1),[0,0,0])
    e.step();snap=e.save_full_state()
    e.step();expected=e.save_full_state()
    e._audio.speech.clip[:]=0;e.step()
    e.restore_full_state(snap);e.step();actual=e.save_full_state()
    for name in ('geom_pos','geom_quat','geom_size','geom_rbound'):
        np.testing.assert_array_equal(actual['mutable_geometry'][name],expected['mutable_geometry'][name])
    np.testing.assert_array_equal(actual['integration_state'],expected['integration_state'])
    assert actual['caregiver']['mouth_opening']==expected['caregiver']['mouth_opening']


def test_adult_changes_side_when_mobile_obscures_face_without_moving_infant(env):
    from mimo_bridge.caregiver_visual import approach, clear_approach, mobile_obstructions
    e=env;c=e.caregiver;m=e.model;d=e.data
    eye=c.state['anchor'].copy();preferred=approach(eye,c.mobile_mount)
    qpos=d.qpos.copy();ctrl=d.ctrl.copy()
    # Put one visible mobile sphere directly between infant and adult, so the
    # chosen side has to change. This is a geometric intervention, not a score.
    sphere=next(g for g in c.mobile_geoms if m.geom_type[g]==e._mj.mjtGeom.mjGEOM_SPHERE)
    m.geom_rgba[c.mobile_geoms,3]=0;m.geom_rgba[sphere,3]=1
    d.geom_xpos[sphere]=eye+preferred*.55;m.geom_size[sphere,0]=.045
    assert mobile_obstructions(e,c,eye,preferred)>0
    c.state.pop('approach_eye',None);c.state['head_offset']=preferred
    chosen=clear_approach(e,c,eye)
    assert mobile_obstructions(e,c,eye,chosen)==0
    np.testing.assert_array_equal(d.qpos,qpos);np.testing.assert_array_equal(d.ctrl,ctrl)


def test_legacy_visual_migration_preserves_every_nonadult_row_and_rejects_other_changes(env):
    from mimo_bridge.caregiver_visual import upgrade_legacy_snapshot
    e=env;m=e.model;snap=e.save_full_state()
    start=int(m.body_geomadr[m.body('caregiver_head').id])
    legacy=copy.deepcopy(snap)
    for name,value in list(legacy['mutable_geometry'].items()):
        if name in ('geom_pos','geom_quat','geom_rbound'):
            del legacy['mutable_geometry'][name]
        elif name.startswith('geom_'):legacy['mutable_geometry'][name]=value[:start+9].copy()
    legacy['mutable_geometry']['geom_size'][start:]=[
        [.075,.065,.095],[.012,0,0],[.006,0,0],[.012,0,0],[.006,0,0],
        [.018,.007,.005],[.15,.08,.22],[.04,.025,.012],[.04,.025,.012]]
    for name in ('geom_xpos','geom_xmat'):legacy['observed_kinematics'][name]=legacy['observed_kinematics'][name][:start+9].copy()
    migrated,changed=upgrade_legacy_snapshot(e,legacy)
    assert changed and migrated['caregiver']['visual_contract']==VERSION
    for name,value in legacy['mutable_geometry'].items():
        np.testing.assert_array_equal(migrated['mutable_geometry'][name][:start] if name.startswith('geom_') else migrated['mutable_geometry'][name],
                                      value[:start] if name.startswith('geom_') else value)
    np.testing.assert_array_equal(migrated['integration_state'],legacy['integration_state'])
    legacy['mutable_geometry']['geom_contype'][-1]=1
    with pytest.raises(ValueError,match='colliding'):upgrade_legacy_snapshot(e,legacy)


# --------------------------------------------------- whose motion "a toy moves" reports
def test_the_infant_cannot_move_the_toy_and_the_adults_own_carry_writes_all_of_it(env, monkeypatch):
    """The adult says "the ball is moving" when the toy's mocap position has travelled
    more than 0.015 m in 0.5 s. The toy is a mocap body - dofnum 0, jntnum 0 - so it has
    no equation of motion and no contact can displace it: the infant physically cannot
    push it, and every centimetre of that displacement is written by the adult's own
    carry. This pins both halves so a later scene change cannot quietly invalidate the
    contingency claim the caregiver box rests on."""
    from mimo_bridge.caregiver import Caregiver
    e = env; m = e.model; d = e.data; c = e.caregiver
    target = m.body('target')
    assert int(m.body_dofnum[target.id]) == 0 and int(m.body_jntnum[target.id]) == 0
    assert int(target.mocapid[0]) >= 0

    # 1. nothing writes the toy's position: can the infant's own body move it at all?
    monkeypatch.setattr(Caregiver, '_move_and_record', lambda self, env_: None)
    before = np.array(d.mocap_pos[e._target_id]).copy()
    worst = 0.
    for k in range(100):
        e.step(ctrl=np.ones(m.nu) * (1 if (k // 5) % 2 == 0 else -1))
        worst = max(worst, float(np.linalg.norm(np.array(d.mocap_pos[e._target_id]) - before)))
    assert worst == 0.0, (
        "every infant actuator driven to full for 1.0 s moved the toy %.6g m; a mocap "
        "body has no equation of motion and contact cannot displace it" % worst)
    monkeypatch.undo()

    # 2. with the adult carrying and the infant still, one check window fills the line
    s = c.state
    s['target_goal'] = np.array(d.mocap_pos[e._target_id]) + np.array([0., .5, 0.])
    start = np.array(d.mocap_pos[e._target_id]).copy()
    previous = start.copy(); path = 0.
    for _ in range(50):                                  # 50 x 0.01 s = one 0.5 s window
        e.step(ctrl=np.zeros(m.nu))
        now = np.array(d.mocap_pos[e._target_id])
        path += float(np.linalg.norm(now - previous)); previous = now.copy()
    assert path == pytest.approx(0.030, abs=1e-4), (
        "the adult's carry is capped at 0.06 m/s, so one 0.5 s window writes 0.030 m; "
        "measured %.6f m" % path)
    assert path / .015 == pytest.approx(2.0, abs=0.02), (
        "the adult's own hand covers %.2fx the 0.015 m line that opens the "
        "'the toy is moving' window" % (path / .015))


# --------------------------------------------------- what the caregiver's look ray sees
def test_mj_ray_skips_a_transparent_geom_only_when_it_carries_no_material():
    """The engine rule the caregiver's look test depends on, measured rather than
    assumed. mujoco 3.3.0 steps over a fully transparent geom only when that geom has no
    material, and only at alpha exactly 0 - alpha 0.04 is still hit. A MuJoCo upgrade
    that changes this must fail here rather than silently change who the adult answers."""
    import mujoco
    xml = """<mujoco>
     <asset><material name="matte" specular=".1"/></asset>
     <worldbody>
      <body name="near" pos="0.12 0 0"><geom name="g_near" type="sphere" size="0.02" %MAT%
            rgba="1 0 0 %A%"/></body>
      <body name="far" pos="0.42 0 0"><geom name="g_far" type="sphere" size="0.02" rgba="0 0 1 1"/></body>
     </worldbody></mujoco>"""

    def first_hit(has_material, alpha):
        src = xml.replace('%MAT%', 'material="matte"' if has_material else '').replace('%A%', str(alpha))
        mo = mujoco.MjModel.from_xml_string(src); da = mujoco.MjData(mo)
        mujoco.mj_forward(mo, da)
        gid = np.array([-1], dtype=np.int32)
        mujoco.mj_ray(mo, da, np.array([0., 0., 0.]), np.array([1., 0., 0.]), None, 1, -1, gid)
        return mo.geom(int(gid[0])).name

    assert first_hit(True, 1.0) == 'g_near'
    assert first_hit(True, 0.0) == 'g_near', "a geom with a material still blocks at alpha 0"
    assert first_hit(False, 1.0) == 'g_near'
    assert first_hit(False, 0.0) == 'g_far', "a material-free geom at alpha 0 is skipped"
    assert first_hit(False, 0.04) == 'g_near', "the skip is alpha == 0, not alpha < 0.05"


def test_a_shape_the_infant_cannot_see_does_not_block_the_caregivers_look(env):
    """looks_at re-casts its ray from just past every hidden shape (alpha < .05). A withdrawn
    toy and the two switched-off toy shapes no longer refuse a look at the adult's face; a
    VISIBLE toy on the line still does; and a look at the visible ring's centre still counts,
    because the looked-at body's own hidden shapes (the ball that fills the ring's hole) keep
    counting as that body. Measured 2026-09-19 on the single-ray rule this replaced: counted,
    refused, refused, refused, counted (audit P277)."""
    from mimo_bridge.caregiver_interaction import looks_at
    e = env; m = e.model; d = e.data; c = e.caregiver
    for name in ('target_geom', 'caregiver_pad', 'caregiver_ring_0'):
        assert int(m.geom_matid[m.geom(name).id]) >= 0, (
            "%s lost its material; the engine's own alpha skip would now apply to it and "
            "this test no longer measures the re-cast" % name)
    cid = m.camera('eye_left').id
    eye = np.array(d.cam_xpos[cid]).copy()
    fwd = -np.asarray(d.cam_xmat[cid]).reshape(3, 3)[:, 2]
    axis = np.eye(3)[int(np.argmin(np.abs(fwd)))]
    side = np.cross(fwd, axis); side /= np.linalg.norm(side)
    head_mid = c.ids['caregiver_head']
    face = {m.body('caregiver_head').id}; toy = {m.body('target').id}

    def place(head, toy_at, visibility, style):
        c.state['style'] = style; c._appearance(e, style)
        d.mocap_pos[head_mid] = head; d.mocap_pos[e._target_id] = toy_at
        c._availability(e, visibility, 1.)
        m.geom_rgba[c.adult_geoms, 3] = 1.
        e._mj.mj_forward(m, d)

    place(eye + fwd * .50, eye - fwd * 1.0, 1., 0)
    assert looks_at(e, d.mocap_pos[head_mid], face), "the adult's face is not reachable at all"
    place(eye + fwd * .50, eye + fwd * .15, 0., 0)
    assert looks_at(e, d.mocap_pos[head_mid], face), (
        "a withdrawn toy - alpha 0, collisions off - on the line of sight refused the look")
    place(eye + fwd * .50, eye + fwd * .15, 1., 0)
    assert not looks_at(e, d.mocap_pos[head_mid], face), "a visible toy on the line must still block"
    place(eye + fwd * .50, eye + fwd * .15 + side * .030, 1., 0)
    assert looks_at(e, d.mocap_pos[head_mid], face), (
        "the switched-off pad's corner refused a look that passes 0.030 m beside the 0.025 m ball")
    place(eye + side * 1.0, eye + fwd * .30, 1., 2)
    assert looks_at(e, d.mocap_pos[e._target_id], toy), (
        "a look at the visible ring's centre lands on the hidden ball in its hole and must count")
