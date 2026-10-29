# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A visible, responding adult and changing reachable objects; no infant rewards.

The adult observes gaze direction and contact. A replay control uses a recorded
event sequence regardless of the infant's behavior. All timing is simulation time.
MIMo's own voice is outside this module.
"""
from __future__ import annotations

from collections import deque
import copy
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .speech import SpeechBank
from .caregiver_visual import VERSION, approach, clear_approach, pose_adult, pose_record


# THE EVENT LIST IS BOUNDED (2026-10-02). Without a recording folder the caregiver keeps its
# events in state['events'], a plain list that gains one adult pose per model step (5 per
# training update at action-repeat 5) and was never trimmed: H84's eight worlds held 203,757 to
# 203,873 events each at step 455,800. Every world snapshot and restore deep-copies that list at
# 26.9 us per event (5.5 s per copy at that length), the episode companion stored 690 pickled
# bytes per event (111.45 MB of a world's 118.24 MB at step 447,350), and each world worker held
# it as objects. While a world runs nothing reads the events back: the caregiver appends,
# BridgeEnv._collect reports len(), and a recording's fields are read only when the list is a
# tape (hasattr 'manifest'). The full sequence is read only by tools that export or iterate the
# events of a short fresh run (tools/caregiver_preview.py 50 s = 5,000 events,
# tools/caregiver_exchange_preview.py 125 s = 12,500, tools/developmental_trial.py at most
# 2,000 updates = 10,000 per episode) or of a replay-mode run (caregiver_learning_trial.py).
# So a responsive caregiver's list stays a plain list, exactly as before, up to
# EVENT_LIST_LIMIT events (1.6 times the longest of those runs); past it the list becomes an
# EventLog holding the newest EVENT_LOG_WINDOW events and the total count. A replay caregiver is
# never trimmed: its events are the replayed sequence, which its tools export.
EVENT_LIST_LIMIT = 20000
EVENT_LOG_WINDOW = 100


class EventLog:
    """A responsive caregiver's event list past EVENT_LIST_LIMIT: the newest `window` events and
    the count of every event ever appended.

    len() is that count, so the world report (BridgeEnv._collect) reads exactly what the full list
    gave. Reading the full sequence (iterating, exporting) is refused once events were dropped:
    a partial history must never pass for the adult's whole sequence. A tool that needs the whole
    sequence of a long run records it to a tape (caregiver_record_dir, event_tape.EventTape)."""

    def __init__(self, events=(), total=None, window=EVENT_LOG_WINDOW):
        self.window = int(window)
        if self.window < 0:
            raise ValueError('event window must be non-negative')
        self.recent = deque(events, maxlen=self.window)
        self.total = len(self.recent) if total is None else int(total)
        if self.total < len(self.recent):
            raise ValueError('event count is smaller than the events kept')

    @classmethod
    def from_events(cls, events, window=EVENT_LOG_WINDOW):
        """The newest `window` events of a list (or of another EventLog) and its full count. Only
        those events are referenced; the list itself is neither copied nor changed."""
        if isinstance(events, EventLog):
            kept = list(events.recent)
            return cls(kept[max(0, len(kept) - window):], len(events), window)
        if not isinstance(events, list):
            events = list(events)
        return cls(events[max(0, len(events) - window):], len(events), window)

    @property
    def dropped(self):
        return self.total - len(self.recent)

    def append(self, event):
        self.recent.append(event)
        self.total += 1

    def __len__(self):
        return self.total

    def _refuse(self, what):
        raise ValueError(f'caregiver kept only its newest {len(self.recent)} of {self.total} events; '
                         f'{what} needs the whole sequence: record it with caregiver_record_dir')

    def __iter__(self):
        if self.dropped:
            self._refuse('iterating the events')
        return iter(list(self.recent))

    def export_json(self, path):
        if self.dropped:
            self._refuse('exporting the events')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(list(self.recent), indent=2, allow_nan=False), encoding='utf-8')

    def __eq__(self, other):
        return (isinstance(other, EventLog) and self.window == other.window
                and self.total == other.total and list(self.recent) == list(other.recent))

    # SAVED AS THE COUNTER EVERY OLDER CODE LOADS (review of 2026-10-02). A pickle of an EventLog
    # names this class, and caregiver.py before this change has none: rolling back, a branch forked
    # before the merge, or production's code pinned for a comparison could not load any save made
    # after it (AttributeError: Can't get attribute 'EventLog'). So a pickle (the episode companion,
    # a world sent between a world worker and the trainer) stores babble's counter,
    # tools/observer/baseline.DiscardedEvents(count), in the code since 2026-09-24: append and len()
    # are all a running world uses, and the count carries on. The kept events, which nothing reads,
    # are dropped from the save. The counter's module is imported here by name when a save is
    # written; that is a class definition, not a reader of anything the observer writes. An
    # in-process copy (a world snapshot or restore, copy.deepcopy) stays an EventLog.
    def __reduce__(self):
        from importlib import import_module
        counter = import_module(__package__ + '.tools.observer.baseline').DiscardedEvents
        return (counter, (self.total,))

    def __deepcopy__(self, memo):
        out = EventLog(copy.deepcopy(list(self.recent), memo), self.total, self.window)
        memo[id(self)] = out
        return out

    def __copy__(self):
        return EventLog(list(self.recent), self.total, self.window)

    __hash__ = None

    def __repr__(self):
        return f'EventLog(total={self.total}, kept={len(self.recent)}, window={self.window})'


def add_caregiver_to_scene(path):
    tree = ET.parse(path)
    world = tree.getroot().find('worldbody')
    from .caregiver_visual import add_adult
    add_adult(tree.getroot(),world)
    target = world.find("body[@name='target']")
    if target is None:
        raise ValueError('caregiver needs a crib target body')
    # COMPILED COLLIDABLE, SWITCHED AT RUN TIME. MuJoCo builds each body's bounding-volume
    # hierarchy (the tree its midphase collision search walks) at compile time and leaves out
    # every geom whose contype and conaffinity are both 0; setting them to 1 later, as
    # _appearance does, cannot put the geom back. Measured on mujoco 3.3.0 on this scene:
    # compiled at 0/0 the pad and the ring met an infant shape at 0 of 64 placements; compiled
    # at 1/1 they meet one at 64 and 61 of 64, and a switched-off shape at 0 of 64.
    # _appearance still decides which one style collides; alpha 0 keeps the other two unseen.
    ET.SubElement(target, 'geom', name='caregiver_pad', type='box', size='.035 .035 .008',
                  mass='.02', rgba='.3 .65 .3 0', contype='1', conaffinity='1',
                  solref='.04 1', friction='1 .005 .0001')
    # A rigid ring made from six capsule segments, with actual empty space inside.
    for i in range(6):
        a, b = 2 * np.pi * i / 6, 2 * np.pi * (i + 1) / 6
        pts = (.028 * np.cos(a), .028 * np.sin(a), 0,
               .028 * np.cos(b), .028 * np.sin(b), 0)
        ET.SubElement(target, 'geom', name=f'caregiver_ring_{i}', type='capsule',
                      fromto=' '.join(map(str, pts)), size='.006', mass='.003',
                      rgba='.9 .7 .1 0', contype='1', conaffinity='1')
    tree.write(path, encoding='unicode')


class Caregiver:
    # THE TOY HANDED OVER (2026-09-21). The toy is a MuJoCo mocap body: its place is written
    # from outside every step and contact can never move it (measured 2026-09-15: every infant
    # actuator at full force for 1.0 s moved it exactly 0 m). So nothing the infant did could
    # carry it, and UEMA 11.2 - a grasped toy joining the self - could never be earned in this
    # scene. With toy_hand_off='follow' the adult lets go when the infant grips: once GRIP_PARTS
    # different parts of one hand have touched the toy for GRIP_STEPS model steps in a row, the
    # toy keeps its pose relative to that hand, turning with it, until no part of that hand has
    # touched it for RELEASE_STEPS steps; then the adult's own carry takes over again from
    # wherever the infant left it. The toy stays a mocap body, so nothing in the physical state
    # changes shape and an episode continues across the change. 'off' is the world as it was.
    GRIP_PARTS = 2
    GRIP_STEPS = 10           # 0.1 s at the crib's 0.01 s model step
    RELEASE_STEPS = 10
    HAND_BLEND_STEPS = 30     # the adult's own hand goes back to the toy over 0.3 s
    HAND_PART_ENDINGS = ("distal", "middle", "knuckle", "metacarpal", "thbase", "thhub")

    def __init__(self, env, mode, speech_dir, replay_path=None, record_dir=None, status_dir=None,
                 present_seconds=60., absent_seconds=15., toy_hand_off='off'):
        if mode not in ('responsive', 'replay'):
            raise ValueError('caregiver mode must be responsive or replay')
        if toy_hand_off not in ('off', 'follow'):
            raise ValueError("toy_hand_off must be 'off' or 'follow'")
        # a replayed adult is replayed exactly, so a second infant can never take the toy away
        # from the recorded trajectory the yoked control exists to reproduce
        self.toy_hand_off = toy_hand_off if mode == 'responsive' else 'off'
        if env._audio is None:
            raise ValueError('caregiver speech requires real audio')
        self.mode = mode
        self.present_seconds=float(present_seconds);self.absent_seconds=float(absent_seconds)
        if not all(np.isfinite(v) and v>0 for v in (self.present_seconds,self.absent_seconds)):
            raise ValueError('caregiver interaction durations must be finite positive seconds')
        self.record_dir = record_dir
        self.status_dir = status_dir or Path.cwd() / 'Saved/mimo_bridge'
        self.bank = SpeechBank(speech_dir)
        self.ids = {n: int(env.model.body(n).mocapid[0]) for n in
                    ('caregiver_head', 'caregiver_torso', 'caregiver_hand_l', 'caregiver_hand_r')}
        infant = set(int(x) for x in getattr(env, '_touch_body_ids', ()))
        self.hand_parts = {}          # body id -> 'left' or 'right', for the parts of each hand
        self.hand_anchor = {}         # 'left' or 'right' -> the hand body the toy moves with
        for b in range(env.model.nbody):
            name = env.model.body(b).name or ''
            side = ('left' if name.startswith('left_') else
                    ('right' if name.startswith('right_') else None))
            if side is None or b not in infant:
                continue
            base = name.split('_', 1)[1]
            if base == 'hand' or base.endswith(self.HAND_PART_ENDINGS):
                self.hand_parts[b] = side
            if base == 'hand':
                self.hand_anchor[side] = b
        self.object_geoms = [[env.model.geom('target_geom').id],
                            [env.model.geom('caregiver_pad').id],
                            [env.model.geom(f'caregiver_ring_{i}').id for i in range(6)]]
        self.replay = []
        if mode == 'replay':
            if not replay_path:
                raise ValueError('replay caregiver requires a recorded event file')
            if str(replay_path).endswith('.tape.json'):
                from .event_tape import TapeReplay
                self.replay = TapeReplay(replay_path)
            else:
                self.replay = json.loads(Path(replay_path).read_text(encoding='utf-8'))
            if not self.replay:
                raise ValueError('empty caregiver replay cannot be a control')
            if isinstance(self.replay, list):
                times = [float(e['time']) for e in self.replay]
                if not np.isfinite(times).all() or times != sorted(times):
                    raise ValueError('caregiver replay times must be finite and ordered')
        self.state = {}
        try:
            self.mobile_id = int(env.model.body('mobile').id)
            self.mobile_mount = np.array(env.model.body_pos[self.mobile_id])
        except KeyError:
            self.mobile_id, self.mobile_mount = None, None
        # WHICH PART OF THE MOBILE MOVES. body('mobile') is the ceiling anchor: its own
        # model.body_pos is the mount the adult raises and lowers, and its world position
        # never changes at all (measured on the crib scene, 400 babble steps: spread
        # 0.000000000 m). What swings when the infant bats it hangs 60 cm below on a ball
        # joint. MimoEnv already measures the mobile's speed on the centre element's geom
        # (mobile_el_c_g); this reports the place of that same geom, so the two agree.
        self.mobile_track_geom = None
        try:
            self.mobile_track_geom = int(env.model.geom('mobile_el_c_g').id)
        except KeyError:
            self.mobile_track_geom = None
        self.mobile_geoms = []
        if self.mobile_id is not None:
            for gid, body in enumerate(env.model.geom_bodyid):
                body = int(body)
                while body > 0 and body != self.mobile_id:
                    body = int(env.model.body_parentid[body])
                if body == self.mobile_id:
                    self.mobile_geoms.append(gid)
        self.mobile_appearance = {g: (float(env.model.geom_rgba[g, 3]),
                                      int(env.model.geom_contype[g]), int(env.model.geom_conaffinity[g]))
                                  for g in self.mobile_geoms}
        adult_bodies={env.model.body(n).id for n in self.ids}
        self.adult_geoms=np.flatnonzero(np.isin(env.model.geom_bodyid,list(adult_bodies)))

    def reset(self, env):
        self.close_recording()
        head = np.asarray(env.data.cam_xpos[env.model.camera('eye_left').id]).copy()
        target = np.asarray(env.data.mocap_pos[env._target_id]).copy()
        offset=approach(head,self.mobile_mount if self.mobile_mount is not None else head+[0.,-.2,0.])
        self.state = dict(time=0., next_scene=0., next_reply=0., look_time=0.,
                          last_contact=False, visible=True, style=env.seed % 3, replay_index=0,
                          anchor=head, target_home=target, target_goal=target,
                          head_goal=head+offset, head_offset=offset, visual_contract=VERSION,
                          scene_phase='starting', phase_until=None, scene_index=0,
                          object_disappearances=0, mobile_disappearances=0,
                          object_appearances=0, mobile_appearances=0,
                          events=[], replies=0)
        if self.record_dir and self.mode == 'responsive':
            from .event_tape import EventTape
            self.state['events'] = EventTape(
                self.record_dir, Path(self.status_dir) / f'caregiver_storage_{os.getpid()}.json')
        self.state['mobile_id'] = self.mobile_id
        if self.mobile_id is not None:
            self.state['mobile_mount'] = self.mobile_mount.copy()
            env.model.body_pos[self.mobile_id] = self.mobile_mount
        self.state['head_offset']=clear_approach(env,self,head)
        self.state['head_goal']=head+self.state['head_offset']
        env.data.mocap_pos[self.ids['caregiver_head']] = self.state['head_goal']
        self._appearance(env, self.state['style'])
        self._availability(env, True, True)
        pose_adult(env,self)

    def _availability(self, env, object_present, mobile_present):
        # Values may be fractional during withdrawal/return. Collisions remain
        # physical until the object has been carried fully out of this space.
        object_visibility=float(np.clip(object_present,0,1))
        mobile_visibility=float(np.clip(mobile_present,0,1))
        for label,value in (('object',object_visibility),('mobile',mobile_visibility)):
            before=self.state.get(label+'_present')
            after=value>0
            if before is not None and before!=after:
                key=label+('_appearances' if after else '_disappearances')
                self.state[key]=self.state.get(key,0)+1
            self.state[label+'_present']=after
            self.state[label+'_visibility']=value
        self._appearance(env, self.state['style'])
        for k,group in enumerate(self.object_geoms):
            for gid in group:
                env.model.geom_rgba[gid, 3] = object_visibility if k==self.state['style'] else 0
                if not object_visibility:
                    env.model.geom_contype[gid] = env.model.geom_conaffinity[gid] = 0
        for gid, (alpha, contype, affinity) in self.mobile_appearance.items():
            env.model.geom_rgba[gid, 3] = alpha * mobile_visibility
            env.model.geom_contype[gid] = contype if mobile_visibility else 0
            env.model.geom_conaffinity[gid] = affinity if mobile_visibility else 0

    def _appearance(self, env, style):
        for k, group in enumerate(self.object_geoms):
            for gid in group:
                env.model.geom_rgba[gid, 3] = float(k == style)
                env.model.geom_contype[gid] = int(k == style)
                env.model.geom_conaffinity[gid] = int(k == style)

    def _event(self, env, event):
        s = self.state
        event = copy.deepcopy(event)
        event.setdefault('time', s['time'])
        for key in ('target', 'head', 'target_actual', 'head_actual', 'mobile_actual'):
            if key in event:
                point = np.asarray(event[key], float)
                if point.shape != (3,) or not np.isfinite(point).all():
                    raise ValueError('caregiver positions must be finite 3D points')
        if 'style' in event and event['style'] not in (0, 1, 2):
            raise ValueError('unknown caregiver object style')
        self._record(event)
        if 'target_actual' in event:
            s['replay_pose'] = event
        if 'target' in event:
            s['target_goal'] = np.asarray(event['target'], float)
        if 'style' in event:
            s['style'] = int(event['style'])
            self._appearance(env, s['style'])
        if 'visible' in event:
            s['visible'] = bool(event['visible'])
        if 'head' in event:
            s['head_goal'] = np.asarray(event['head'], float)
            if self.mode=='responsive':s['head_offset']=s['head_goal']-s['anchor']
        if 'phrase' in event:
            position = env._audio.speech.position
            if env._audio.speech.start(self.bank.clips[event['phrase']], position, event['phrase']):
                s['replies'] += 1
                s['last_utterance']=dict(key=event['phrase'],text=self.bank.texts[event['phrase']],
                    time=s['time'],group=event.get('speech_group'),reason=event.get('reason'))
            else:
                raise RuntimeError('overlapping caregiver utterance in recorded event sequence')

    def tick(self, env):
        s, dt = self.state, env.dt
        s['time'] += dt
        if self.mode=='responsive' and self.bank.groups:
            from .caregiver_interaction import tick
            tick(self,env)
            self._move_and_record(env)
            return
        if self.mode == 'replay':
            while s['replay_index'] < len(self.replay) and self.replay[s['replay_index']]['time'] <= s['time']:
                self._event(env, self.replay[s['replay_index']])
                s['replay_index'] += 1
        else:
            # Follow where the infant actually moved. Only the adult's goals
            # change; its bounded movement below never relocates the infant.
            eye=np.asarray(env.data.cam_xpos[env.model.camera('eye_left').id]).copy()
            shift=eye-s['anchor']
            s['anchor']=eye
            s['target_home']=s['target_home']+shift
            s['target_goal']=s['target_goal']+shift
            s['head_offset']=clear_approach(env,self,eye)
            s['head_goal']=eye+s['head_offset']
            busy = env._audio.speech.busy
            phase=s.get('scene_phase','starting')
            if not busy:
                if phase=='starting' or (phase=='absent' and s['time']>=s['phase_until']):
                    index=s.get('scene_index',0)
                    goal = s['target_home'] + np.array([(-1 if index % 2 else 1) * .07, 0, .015])
                    style = (index + env.seed) % 3
                    self._event(env, dict(target=goal.tolist(), style=style, visible=True,
                                          phrase=('ball', 'cloth', 'ring')[style]))
                    s['scene_index']=index+1;s['scene_phase']='presenting';s['phase_until']=None
                    s['next_reply']=s['time']+6.
                elif phase=='engaged' and s['time']>=s['phase_until']:
                    self._event(env, dict(target=(s['anchor'] + [0., .55, .35]).tolist(),
                                          visible=False, phrase='away'))
                    s['scene_phase']='withdrawing';s['phase_until']=None
                    s['next_reply']=s['time']+6.
            cid = env.model.camera('eye_left').id
            eye = np.asarray(env.data.cam_xpos[cid])
            fwd = -np.asarray(env.data.cam_xmat[cid]).reshape(3, 3)[:, 2]
            toward = env.data.mocap_pos[self.ids['caregiver_head']] - eye
            looking = np.dot(toward / max(np.linalg.norm(toward), 1e-9), fwd) > np.cos(np.radians(15))
            s['look_time'] = s['look_time'] + dt if looking else 0.
            infant_bodies = set(int(x) for x in env._touch_body_ids)
            target_body = env.model.body('target').id
            touched = any((int(env.model.geom_bodyid[c.geom1]) == target_body
                           and int(env.model.geom_bodyid[c.geom2]) in infant_bodies)
                          or (int(env.model.geom_bodyid[c.geom2]) == target_body
                              and int(env.model.geom_bodyid[c.geom1]) in infant_bodies)
                          for c in env.data.contact)
            if not busy and s['time'] >= s['next_reply']:
                if touched and not s['last_contact']:
                    self._event(env, dict(phrase='touch'))
                    s['next_reply'] = s['time'] + 6.
                elif s['look_time'] >= .4:
                    self._event(env, dict(phrase='look'))
                    s['next_reply'] = s['time'] + 6.
                    s['look_time'] = 0.
            s['last_contact'] = touched
        self._move_and_record(env)

    def _hand_parts_on_toy(self, env):
        """Which parts of each hand are touching the toy this step."""
        target_body = env.model.body('target').id
        out = {'left': set(), 'right': set()}
        for i in range(env.data.ncon):
            c = env.data.contact[i]
            b1 = int(env.model.geom_bodyid[c.geom1])
            b2 = int(env.model.geom_bodyid[c.geom2])
            other = b2 if b1 == target_body else (b1 if b2 == target_body else None)
            if other is not None and other in self.hand_parts:
                out[self.hand_parts[other]].add(other)
        return out

    def _grip_step(self, env):
        """Advance the hand-off one model step. True while the infant holds the toy, and then
        the toy has already been placed where the gripping hand carries it."""
        import mujoco as _mj
        s = self.state
        g = s.setdefault('grip', dict(held=False, hold_steps=0, free_steps=0, side=None,
                                      local_pos=None, local_rot=None, adult_hand=None,
                                      blend=0, grips=0, releases=0))
        if self.toy_hand_off != 'follow' or not self.hand_anchor:
            return False
        parts = self._hand_parts_on_toy(env)
        d = env.data
        tid = env._target_id
        if not g['held']:
            side = max(parts, key=lambda k: len(parts[k]))
            g['hold_steps'] = g['hold_steps'] + 1 if len(parts[side]) >= self.GRIP_PARTS else 0
            if g['hold_steps'] >= self.GRIP_STEPS and side in self.hand_anchor:
                hb = self.hand_anchor[side]
                R = np.asarray(d.xmat[hb]).reshape(3, 3)
                R_toy = np.zeros(9)
                _mj.mju_quat2Mat(R_toy, np.asarray(d.mocap_quat[tid], dtype=np.float64))
                g.update(held=True, side=side, free_steps=0, grips=g['grips'] + 1,
                         local_pos=(R.T @ (np.asarray(d.mocap_pos[tid])
                                           - np.asarray(d.xpos[hb]))).tolist(),
                         local_rot=(R.T @ R_toy.reshape(3, 3)).tolist(),
                         adult_hand=np.asarray(d.mocap_pos[self.ids['caregiver_hand_l']]).tolist())
        else:
            g['free_steps'] = g['free_steps'] + 1 if not parts[g['side']] else 0
            if g['free_steps'] >= self.RELEASE_STEPS:
                g.update(held=False, hold_steps=0, local_pos=None, local_rot=None,
                         blend=self.HAND_BLEND_STEPS, releases=g['releases'] + 1)
        if not g['held']:
            return False
        hb = self.hand_anchor[g['side']]
        R = np.asarray(d.xmat[hb]).reshape(3, 3)
        d.mocap_pos[tid] = np.asarray(d.xpos[hb]) + R @ np.asarray(g['local_pos'])
        q = np.zeros(4)
        _mj.mju_mat2Quat(q, (R @ np.asarray(g['local_rot'])).reshape(-1))
        d.mocap_quat[tid] = q
        # wherever the infant takes it, that is where the adult will carry it from
        s['target_goal'] = np.asarray(d.mocap_pos[tid]).copy()
        return True

    def _move_and_record(self,env):
        s,dt=self.state,env.dt
        held = self._grip_step(env)
        # Adult movement is bounded and continuous. Stop bringing an object closer
        # during contact; this is the caregiver's behavior, never an infant motor override.
        target = env.data.mocap_pos[env._target_id]
        delta = s['target_goal'] - target
        target_body = env.model.body('target').id
        contact = any(int(env.model.geom_bodyid[c.geom1]) == target_body
                      or int(env.model.geom_bodyid[c.geom2]) == target_body for c in env.data.contact)
        if contact and np.dot(delta, target - s['anchor']) < 0:
            delta = np.zeros(3)
        if held:
            delta = np.zeros(3)          # the infant has it; the adult's carry waits
        target += delta * min(1., .06 * dt / max(np.linalg.norm(delta), 1e-9))
        if s['mobile_id'] is not None:
            mount = env.model.body_pos[s['mobile_id']]
            desired = s['mobile_mount'] + [0., 0., 0. if s['visible'] else .7]
            shift = desired - mount
            mount += shift * min(1., .08 * dt / max(np.linalg.norm(shift), 1e-9))
        head = env.data.mocap_pos[self.ids['caregiver_head']]
        delta = s['head_goal'] - head
        head += delta * min(1., .05 * dt / max(np.linalg.norm(delta), 1e-9))
        # Yoked control replays the ADULT'S actual trajectory. Merely replaying
        # goals would still let this second infant alter the adult's contact stop.
        if self.mode == 'replay' and 'replay_pose' in s:
            pose = s['replay_pose']
            target[:] = pose['target_actual']
            head[:] = pose['head_actual']
            if self.mobile_id is not None and 'mobile_actual' in pose:
                env.model.body_pos[self.mobile_id] = pose['mobile_actual']
        pose_adult(env,self,s.get('replay_pose') if self.mode=='replay' else None)
        # Carry objects out of reach before removing them from the scene. Raising
        # the mobile alone leaves it visible directly overhead and is not absence.
        # The same spatial fade applies in both directions: returning objects do
        # not pop back into view merely because a timer said "present".
        object_present = float(np.clip((.45-np.linalg.norm(target-s['anchor']))/.1,0,1))
        # NO MOBILE IN THE SCENE MEANS THE MOBILE IS NOT THERE. This read 1.0, so a scene
        # built without a mobile reported one permanently present: the withdrawing phase
        # could never reach absent (it waits for both things to be gone) and the entity
        # presence vector carried a column of ones for a thing that does not exist.
        mobile_present = (0. if self.mobile_id is None else float(np.clip(
            (.45-(env.model.body_pos[self.mobile_id,2]-self.mobile_mount[2]))/.15,0,1)))
        if self.mode == 'replay' and 'replay_pose' in s:
            object_present = s['replay_pose'].get('object_visibility',s['replay_pose'].get('object_present', object_present))
            mobile_present = s['replay_pose'].get('mobile_visibility',s['replay_pose'].get('mobile_present', mobile_present))
        self._availability(env, object_present, mobile_present)
        if self.mode == 'responsive':
            phase=s['scene_phase']
            if phase=='presenting' and (np.linalg.norm(target-s['target_goal'])<1e-5 or contact):
                s['scene_phase']='engaged';s['phase_until']=s['time']+self.present_seconds
            elif phase=='withdrawing' and not s['object_present'] and not s['mobile_present']:
                s['scene_phase']='absent';s['phase_until']=s['time']+self.absent_seconds
            pose = dict(time=s['time'], target_actual=target.tolist(), head_actual=head.tolist())
            pose.update(style=s['style'],visible=s['visible'])
            pose.update(object_present=bool(object_present), mobile_present=bool(mobile_present))
            pose.update(object_visibility=float(object_present),mobile_visibility=float(mobile_present),
                        infant_head_actual=s['anchor'].tolist(),scene_phase=s['scene_phase'])
            pose.update(pose_record(env,self))
            if self.mobile_id is not None:
                pose['mobile_actual'] = env.model.body_pos[self.mobile_id].tolist()
            self._record(pose)

    def _record(self, event):
        """Every event the caregiver keeps goes through here: appended, then the list bounded."""
        self.state['events'].append(event)
        self.state['events'] = self._bounded_events(self.state['events'])

    def _bounded_events(self, events):
        """A responsive caregiver's plain list longer than EVENT_LIST_LIMIT as an EventLog of its
        newest events and its count; anything else (a shorter list, a tape, an EventLog, a replay
        caregiver's list, a probe's counter) as it is. Never copies or changes the list given."""
        if self.mode == 'responsive' and type(events) is list and len(events) > EVENT_LIST_LIMIT:
            return EventLog.from_events(events)
        return events

    def diagnostics(self,env):
        s=self.state;head=env.data.mocap_pos[self.ids['caregiver_head']]
        eye=np.asarray(env.data.cam_xpos[env.model.camera('eye_left').id])
        return dict(schedule='arrival_timed_v2',simulation_seconds=s['time'],phase=s.get('scene_phase'),
            visual_contract=VERSION,mouth_opening=s.get('mouth_opening',0.),
            desired_face_points_obstructed=s.get('face_points_obstructed'),
            interaction_contract=s.get('interaction_contract'),adult_phase=s.get('adult_phase','shared'),
            adult_visibility=s.get('adult_visibility',1.),adult_departures=s.get('adult_departures',0),
            adult_returns=s.get('adult_returns',0),last_utterance=s.get('last_utterance'),
            present_seconds=self.present_seconds,absent_seconds=self.absent_seconds,
            head_horizontal_distance_m=float(np.linalg.norm((head-eye)[:2])),
            head_height_m=float(head[2]-eye[2]),adult_present=s.get('adult_visibility',1.)>0,
            # WHERE THE OUTSIDE THINGS ARE, in world metres. The body map needs this to see a
            # held toy move WITH the commands and so take it into the self (UEMA 11.2). World
            # coordinates, not coordinates relative to the eye: relative to the eye a toy
            # standing still appears to move whenever the head turns, which would read as the
            # head's command moving the toy. A thing that is not out reports its last place and
            # is marked absent by object_present / mobile_present beside it.
            # THE TOY IS A MOCAP BODY (<body name="target" mocap="true">). MuJoCo never
            # integrates a mocap body: its place is written each step by this caregiver and
            # the infant's hand cannot displace it. So this is where the toy is, and the
            # body map can read a match between a hand and the toy only when the adult
            # happens to carry it the way the hand is moving - never the infant carrying it.
            object_position=([float(v) for v in env.data.mocap_pos[env._target_id]]
                             if getattr(env,'_target_id',None) is not None else [0.,0.,0.]),
            object_physics_driven=False,
            # handed over: while the infant grips it, the toy goes where the infant's hand goes
            object_follows_grip=self.toy_hand_off == 'follow',
            object_held_by_infant=bool((s.get('grip') or {}).get('held', False)),
            object_grips=int((s.get('grip') or {}).get('grips', 0)),
            mobile_position=([float(v) for v in env.data.geom_xpos[self.mobile_track_geom]]
                             if self.mobile_track_geom is not None else [0.,0.,0.]),
            mobile_physics_driven=self.mobile_track_geom is not None,
            **{k:s.get(k) for k in ('object_present','mobile_present','object_disappearances',
                                    'object_appearances','mobile_disappearances','mobile_appearances')})

    def snapshot(self):
        return copy.deepcopy(self.state)

    def restore(self, state):
        """A saved state, copied. A saved list past EVENT_LIST_LIMIT (every H84 world saved from
        about step 419,100, when its list passed 20,000, to this change) is bounded BEFORE the copy,
        so only its newest events are copied; its count carries on. The state given is not changed,
        so it can be restored again."""
        self.close_recording()
        events = state.get('events') if isinstance(state, dict) else None
        bounded = self._bounded_events(events) if events is not None else None
        if bounded is not events:
            state = dict(state, events=bounded)
        self.state = copy.deepcopy(state)

    def export_events(self, path):
        """Replay the actual sequence independently of a second infant's behavior."""
        if hasattr(self.state['events'], 'export_json'):
            self.state['events'].export_json(path)
            return
        if not isinstance(self.state['events'], list):
            # a counter (an EventLog loaded from a save, babble's DiscardedEvents) keeps no events
            raise ValueError(f"caregiver kept only the count of its {len(self.state['events'])} events; "
                             'exporting the events needs the whole sequence: record it with caregiver_record_dir')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state['events'], indent=2, allow_nan=False), encoding='utf-8')

    def close_recording(self):
        events = self.state.get('events')
        if hasattr(events, 'close'):
            events.close()
