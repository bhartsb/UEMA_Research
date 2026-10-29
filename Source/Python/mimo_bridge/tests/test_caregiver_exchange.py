# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from mimo_bridge.caregiver_interaction import initialize, say


@pytest.fixture
def env(tmp_path):
    from mimo_bridge.env import BridgeEnv
    speech=Path(__file__).resolve().parents[4]/'Saved/mimo_bridge/caregiver_speech_exchange_v1_final'
    if not (speech/'manifest.json').exists():pytest.skip('build the versioned adult speech bank')
    e=BridgeEnv(scene='crib',vision=False,audio_real=True,mobile=True,age=24,
        caregiver_mode='responsive',caregiver_speech_dir=str(speech),run_dir=str(tmp_path))
    e.reset()
    try:yield e
    finally:e.close()


def adult_step(e):
    e.caregiver.tick(e);e._mj.mj_forward(e.model,e.data)
    e._audio.speech.render(e.dt,e.caregiver.state['anchor'],np.array([1,0,0]),np.array([0,0,1]),np.arange(10)*200+200)


def test_variant_choice_is_reproducible_never_repeats_and_waits_for_audio(env):
    e=env;c=e.caregiver;initialize(c,e);s=c.state
    previous=None
    for _ in range(12):
        e._audio.speech.cursor=len(e._audio.speech.clip)
        saved=copy.deepcopy(s);audio=copy.deepcopy(e._audio.speech.__dict__)
        assert say(c,e,'present_ball','object_presented')
        key=s['last_utterance']['key'];expected=copy.deepcopy(s)
        assert key!=previous
        assert s['next_reply']>=s['time']+len(e._audio.speech.clip)/16000+2
        assert not say(c,e,'present_ball','object_presented')
        c.state=copy.deepcopy(saved);e._audio.speech.__dict__.update(copy.deepcopy(audio))
        assert say(c,e,'present_ball','object_presented')
        s=c.state;assert s['last_utterance']==expected['last_utterance'];previous=key


def test_departure_has_no_adult_shapes_or_speech_and_returns(env):
    e=env;c=e.caregiver;e.dt=.1;initialize(c,e);c.state['departure_at']=0.
    phases=set();away_time=0.;last_head=e.data.mocap_pos[c.ids['caregiver_head']].copy()
    for _ in range(1800):
        adult_step(e);s=c.state;phases.add(s['adult_phase'])
        head=e.data.mocap_pos[c.ids['caregiver_head']].copy()
        assert np.linalg.norm(head-last_head)<=.05*e.dt+1e-8;last_head=head
        if s['adult_phase']=='away':
            away_time+=e.dt
            assert not e._audio.speech.busy
            assert np.all(e.model.geom_rgba[c.adult_geoms,3]==0)
        if s['adult_returns']:break
    assert {'farewell','leaving','away','returning','shared'}<=phases
    assert 30<=away_time<=60.2
    assert c.state['last_utterance']['group']=='return'


def test_new_dialogue_state_and_body_continue_exactly(env):
    e=env
    for _ in range(8):e.step()
    snap=e.save_full_state();e.step();expected=e.save_full_state()
    e.restore_full_state(snap);e.step();actual=e.save_full_state()
    np.testing.assert_array_equal(actual['integration_state'],expected['integration_state'])
    np.testing.assert_array_equal(actual['mutable_geometry']['geom_rgba'],expected['mutable_geometry']['geom_rgba'])
    assert actual['caregiver']['exchange_rng']==expected['caregiver']['exchange_rng']
    assert actual['caregiver']['last_utterance']==expected['caregiver']['last_utterance']
    np.testing.assert_array_equal(actual['obj__audio']['speech'].last_pcm,expected['obj__audio']['speech'].last_pcm)


def test_yoked_replay_preserves_speech_visibility_and_every_adult_pose(env,tmp_path):
    from mimo_bridge.env import BridgeEnv
    e=env;c=e.caregiver;e.dt=.1;initialize(c,e);c.state['departure_at']=0.
    expected=[]
    for _ in range(400):
        adult_step(e)
        expected.append((e.data.mocap_pos.copy(),e.data.mocap_quat.copy(),e.model.geom_rgba.copy(),
                         e._audio.speech.clip.copy(),e._audio.speech.cursor,c.state['mouth_opening']))
    c.export_events(tmp_path/'events.json')
    other=BridgeEnv(scene='crib',vision=False,audio_real=True,mobile=True,age=24,seed=7,
        caregiver_mode='replay',caregiver_replay=str(tmp_path/'events.json'),
        caregiver_speech_dir=e.caregiver.bank.directory if hasattr(e.caregiver.bank,'directory') else
          str(Path(__file__).resolve().parents[4]/'Saved/mimo_bridge/caregiver_speech_exchange_v1_final'),
        run_dir=str(tmp_path/'replay'))
    try:
        other.reset();other.dt=e.dt
        for pos,quat,rgba,clip,cursor,mouth in expected:
            other.data.cam_xpos[other.model.camera('eye_left').id,:2]+=[.12,-.08]
            adult_step(other)
            np.testing.assert_allclose(other.data.mocap_pos,pos,atol=1e-8)
            np.testing.assert_allclose(other.data.mocap_quat,quat,atol=1e-8)
            np.testing.assert_allclose(other.model.geom_rgba,rgba,atol=1e-8)
            np.testing.assert_array_equal(other._audio.speech.clip,clip)
            assert other._audio.speech.cursor==cursor
            assert other.caregiver.state['mouth_opening']==mouth
    finally:other.close()


def test_language_contract_contains_only_new_grounded_touch_statements(env):
    c=env.caregiver
    for group,keys in c.bank.groups.items():
        assert len(keys)>=3
        if group.startswith('touch_'):
            assert all('heard' not in c.bank.texts[k].lower() for k in keys)


def test_departure_respects_listening_interval_and_return_hand_stays_close(env):
    from mimo_bridge.caregiver_visual import pose_adult
    e=env;c=e.caregiver;initialize(c,e);s=c.state
    assert say(c,e,'present_ball','object_presented')
    e._audio.speech.cursor=len(e._audio.speech.clip)
    s['departure_at']=0.
    adult_step(e)
    assert s['adult_phase']=='shared'
    assert s['last_utterance']['group']=='present_ball'
    s['adult_phase']='returning';s['visible']=False
    e.data.mocap_pos[e._target_id]=s['anchor']+[0.,.55,.35]
    pose_adult(e,c)
    hand=e.data.mocap_pos[c.ids['caregiver_hand_l']]
    torso=e.data.mocap_pos[c.ids['caregiver_torso']]
    assert np.linalg.norm(hand-torso)<.25


def test_legacy_speech_extension_preserves_episode_and_utterance_in_progress(tmp_path):
    import shutil
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.continuation_state import extend_caregiver_speech
    from mimo_bridge.speech import PHRASES
    rich=Path(__file__).resolve().parents[4]/'Saved/mimo_bridge/caregiver_speech_exchange_v1_final'
    legacy=tmp_path/'legacy';legacy.mkdir()
    manifest=json.loads((rich/'manifest.json').read_text())
    manifest.pop('interaction_contract');manifest['phrases']={k:v for k,v in manifest['phrases'].items() if k in PHRASES}
    for entry in manifest['phrases'].values():shutil.copy2(rich/entry['file'],legacy/entry['file'])
    (legacy/'manifest.json').write_text(json.dumps(manifest))
    kw=dict(scene='crib',vision=False,audio_real=True,age=24,caregiver_mode='responsive',
            caregiver_speech_dir=str(legacy),run_dir=str(tmp_path/'scene'))
    old=SingleEnvSource(**kw);new=None
    try:
        old.reset();c=old.env.caregiver
        c._event(old.env,dict(phrase='hello'))
        old.env._audio.speech.cursor=100
        state=dict(config=dict(caregiver_mode='responsive',caregiver_speech_dir=str(legacy)),
                   worlds=old.save_full_state(),learner={'earned':np.arange(3)})
        migrated=extend_caregiver_speech(state,str(rich))
        assert migrated['learner'] is state['learner']
        assert state['worlds'][0]['env_kwargs']['caregiver_speech_dir']==str(legacy)
        new=SingleEnvSource(**dict(kw,caregiver_speech_dir=str(rich)))
        frames=new.restore_full_state(migrated['worlds'])
        assert frames[0].frame_bytes==state['worlds'][0]['payload']['frame_bytes']
        np.testing.assert_array_equal(new.env.data.qpos,old.env.data.qpos)
        assert new.env._audio.speech.cursor==100
        np.testing.assert_array_equal(new.env._audio.speech.clip,old.env._audio.speech.clip)
        assert new.env.caregiver.bank.groups
    finally:
        old.close()
        if new:new.close()


# ---------------------------------------------------------------- the four answered events
def ready(e):
    """An adult that is present, free to speak, with a toy already in reach."""
    c=e.caregiver;e.dt=.1;initialize(c,e);s=c.state
    s['greeting_pending']=False;s['next_reply']=0.
    s['departure_at']=s['time']+1e6;s['quiet_at']=s['time']+1e6
    s['scene_phase']='engaged';s['phase_until']=s['time']+1e6
    return c,s


def eye_axis(e):
    m,d=e.model,e.data;cid=m.camera('eye_left').id
    e._mj.mj_forward(m,d)
    return d.cam_xpos[cid].copy(),-d.cam_xmat[cid].reshape(3,3)[:,2]


def park_toy(e,point):
    e.data.mocap_pos[e._target_id]=point;e.caregiver.state['target_goal']=np.asarray(point,float).copy()


def test_a_held_look_at_the_face_is_answered(env):
    """Within 15 deg of the LEFT EYE camera axis, clear line of sight, held 0.4 s."""
    e=env;c,s=ready(e);m,d=e.model,e.data
    eye,fwd=eye_axis(e)
    park_toy(e,eye-fwd*1.)                       # the toy out of the line of sight
    for _ in range(30):
        d.mocap_pos[c.ids['caregiver_head']]=eye+fwd*.45
        s['head_goal']=d.mocap_pos[c.ids['caregiver_head']].copy()
        e._mj.mj_forward(m,d);adult_step(e)
        if s.get('last_utterance'):break
    assert s['last_utterance']['group']=='face_look'
    assert s['last_utterance']['reason']=='visible_face_attention'
    assert s['face_look_time']>=.4


def test_a_held_look_at_the_toy_is_answered(env):
    e=env;c,s=ready(e);m,d=e.model,e.data
    eye,fwd=eye_axis(e)
    side=np.cross(fwd,[0,0,1.]);side/=np.linalg.norm(side)
    for _ in range(30):
        d.mocap_pos[c.ids['caregiver_head']]=eye+side*1.   # the adult outside the cone
        s['head_goal']=d.mocap_pos[c.ids['caregiver_head']].copy()
        park_toy(e,eye+fwd*.30)
        e._mj.mj_forward(m,d);adult_step(e)
        if s.get('last_utterance'):break
    assert s['last_utterance']['group'].startswith('look_')
    assert s['last_utterance']['reason']=='visible_object_attention'
    assert s['object_look_time']>=.4


def test_any_touch_of_the_toy_is_answered_with_no_force_threshold(env):
    """Any contact pair between the toy and one of the infant's 53 touch bodies."""
    e=env;c,s=ready(e);m,d=e.model,e.data
    from mimo_bridge.caregiver_interaction import contact_now
    touching=None
    for b in map(int,e._touch_body_ids):
        park_toy(e,d.xpos[b].copy());e._mj.mj_forward(m,d)
        if contact_now(c,e):touching=b;break
    assert touching is not None
    adult_step(e)
    assert s['last_utterance']['group'].startswith('touch_')
    assert s['last_utterance']['reason']=='observed_contact'
    assert s['pending_touch_until']==0.           # edge triggered, then spent


@pytest.mark.parametrize('style',[0,1,2])
def test_every_toy_style_can_be_touched_and_the_touch_is_answered(env,style):
    """The ball, the pad and the ring are shapes of one body and _appearance switches one on.
    A geom compiled with contype 0 and conaffinity 0 is left out of the body's bounding-volume
    hierarchy (the tree MuJoCo's midphase collision search walks), and switching it on at run
    time made no contact: measured 2026-09-19 on mujoco 3.3.0 the pad and the ring met an
    infant shape at 0 of 64 placements, the ball at 64 of 64. Compiled at 1/1: 64, 64, 61."""
    from mimo_bridge.caregiver_interaction import contact_now,OBJECT_NAMES
    e=env;c,s=ready(e);m,d=e.model,e.data
    s['style']=style;c._appearance(e,style);c._availability(e,1.,1.)
    infant=set(map(int,e._touch_body_ids));first=None
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) not in infant or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
        park_toy(e,d.geom_xpos[g].copy());e._mj.mj_forward(m,d)
        if contact_now(c,e):first=g;break
    assert first is not None,'no infant shape can touch the '+OBJECT_NAMES[style]
    adult_step(e)
    assert s['last_utterance']['group']=='touch_'+OBJECT_NAMES[style]


def test_a_toy_that_moves_while_looked_at_is_answered(env):
    e=env;c,s=ready(e);m,d=e.model,e.data
    eye,fwd=eye_axis(e)
    side=np.cross(fwd,[0,0,1.]);side/=np.linalg.norm(side)
    s['last_object_attention']=True               # the look itself was already answered
    park_toy(e,eye+fwd*.30);e._mj.mj_forward(m,d)
    s['motion_anchor']=d.mocap_pos[e._target_id].copy();s['motion_checked_at']=s['time']
    step=0
    for _ in range(40):
        d.mocap_pos[c.ids['caregiver_head']]=eye+side*1.
        s['head_goal']=d.mocap_pos[c.ids['caregiver_head']].copy()
        step+=1
        park_toy(e,eye+fwd*.30+side*.004*step)    # 4 mm a tick = 2 cm per 0.5 s window
        e._mj.mj_forward(m,d);adult_step(e)
        if s.get('last_utterance'):break
    assert s['last_utterance']['group'].startswith('move_')
    assert s['last_utterance']['reason']=='observed_object_motion'
    assert s['motion_displacement']>.015


# ------------------------------------------------- which schedule each launch flag times
def test_present_and_absent_seconds_time_the_toy_not_the_adult(env):
    e=env;c=e.caregiver;start=c.state['time'];initialize(c,e);s=c.state
    # the toy's in-reach spell is exactly present_seconds
    s['scene_phase']='presenting'
    s['target_goal']=np.array(e.data.mocap_pos[e._target_id],float)
    c._move_and_record(e)
    assert s['scene_phase']=='engaged'
    assert s['phase_until']==pytest.approx(s['time']+c.present_seconds)
    # the gap before the next toy is exactly absent_seconds
    s['scene_phase']='withdrawing';s['visible']=False
    e.model.body_pos[c.mobile_id]=c.mobile_mount+[0.,0.,.7]
    far=s['anchor']+[0.,0.,1.2];e.data.mocap_pos[e._target_id]=far;s['target_goal']=far.copy()
    c._move_and_record(e)
    assert s['scene_phase']=='absent'
    assert s['phase_until']==pytest.approx(s['time']+c.absent_seconds)
    # the adult's own presence is a draw in [180, 300] s and owes nothing to either flag
    assert 180.<=s['departure_at']-start<=300.


def test_the_adult_away_hold_is_drawn_between_thirty_and_sixty_seconds(env):
    e=env;c=e.caregiver;e.dt=.1;initialize(c,e);s=c.state;s['departure_at']=0.
    for _ in range(1200):
        adult_step(e)
        if s['adult_phase']=='away':break
    assert s['adult_phase']=='away'
    assert 30.<=s['adult_away_until']-s['time']<=60.


# ------------------------------------------------------------------ the direction cues
def test_arrival_time_difference_never_exceeds_the_ear_spacing():
    from mimo_bridge.audio_adapter import EAR_HALF_M, SPEED_MS
    from mimo_bridge.speech import SpeechStream
    ceiling=2*EAR_HALF_M/SPEED_MS*1000.
    assert ceiling==pytest.approx(.2041,abs=1e-4)
    st=SpeechStream();st.start(np.full(4000,.1),[0,0,0])
    freqs=np.geomspace(40.,8000.,256)
    head=np.zeros(3);right=np.array([1.,0.,0.]);fwd=np.array([0.,1.,0.])
    seen=[]
    for x in np.linspace(-2.,2.,21):
        for y in (-1.,0.,.05,1.):
            st.position=np.array([x,y,0.])
            _,meta=st.render(.01,head,right,fwd,freqs)
            seen.append(meta['itd_ms'])
            assert abs(meta['itd_ms'])<=ceiling+1e-9
            assert abs(meta['ild_db'])<=20.
    assert max(seen)>.15 and min(seen)<-.15


def test_a_bank_without_an_interaction_contract_runs_the_older_inline_branch(tmp_path):
    import shutil
    from mimo_bridge.env import BridgeEnv
    from mimo_bridge.speech import PHRASES
    rich=Path(__file__).resolve().parents[4]/'Saved/mimo_bridge/caregiver_speech_exchange_v1_final'
    legacy=tmp_path/'legacy';legacy.mkdir()
    manifest=json.loads((rich/'manifest.json').read_text())
    manifest.pop('interaction_contract')
    manifest['phrases']={k:v for k,v in manifest['phrases'].items() if k in PHRASES}
    for entry in manifest['phrases'].values():shutil.copy2(rich/entry['file'],legacy/entry['file'])
    (legacy/'manifest.json').write_text(json.dumps(manifest))
    e=BridgeEnv(scene='crib',vision=False,audio_real=True,mobile=True,age=24,
        caregiver_mode='responsive',caregiver_speech_dir=str(legacy),run_dir=str(tmp_path/'scene'))
    try:
        e.reset();c=e.caregiver
        assert not c.bank.groups
        for _ in range(5):c.tick(e)
        assert c.state.get('interaction_contract') is None
        assert c.state.get('adult_phase') is None
    finally:e.close()


def test_the_adults_own_carry_opens_the_motion_window_but_no_line_is_spoken_without_a_look(env):
    """Audit P267. The adult's carry alone fills the 0.015 m line (0.06 m/s x 0.5 s = 0.030 m),
    so the window opens with a still infant; the line is spoken only if the infant looks at
    the toy. Measured with a still infant over 600 s of body time: 128 windows opened, 0 spoken."""
    e=env;c,s=ready(e);m,d=e.model,e.data
    eye,fwd=eye_axis(e)
    side=np.cross(fwd,[0,0,1.]);side/=np.linalg.norm(side)
    start=eye+fwd*.20+side*.30                       # 56 degrees off the eye axis: outside the 15 degree cone
    d.mocap_pos[e._target_id]=start;s['target_goal']=start+side*.50
    s['motion_anchor']=start.copy();s['motion_checked_at']=s['time']
    opened=False
    for _ in range(30):
        d.mocap_pos[c.ids['caregiver_head']]=eye-fwd*1.
        s['head_goal']=d.mocap_pos[c.ids['caregiver_head']].copy()
        e._mj.mj_forward(m,d);adult_step(e)
        opened=opened or s['motion_until']>s['time']
    assert opened,"the adult's carry did not open the 'toy moved' window"
    assert not (s.get('last_utterance') or {}).get('group','').startswith('move_')
