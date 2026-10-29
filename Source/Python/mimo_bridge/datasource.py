# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Mode A data source: MIMo frames straight into the model, in-process.

The model's input path is: bytes -> tokenizer -> TokenizedFrame -> build_frame_from_tokenized
-> EncoderRegistry. Mode A keeps that path but skips shared memory: the bridge packs the
31040-byte frame (frame_encoder) and decodes it with the model's own tokenizer functions
(decode_with_model_tokenizer). So Mode A and Mode B see bit-identical CPU frames; only the
transport differs. Vision is attached as tensors (the model's vision path is GPU-texture
only and has no CPU byte route, see plan check item 16).

Two sources with the same interface:
  SingleEnvSource   one BridgeEnv in this process (Phase 1 driver, tests)
  PoolSource        N BridgeEnvs in N worker processes (Phase 2 training, E5)

Each step returns a list of BridgeFrame (one per env).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from lem.model.body_sets import HANDLE_NAMES
from mimo_bridge import frame_encoder as FE

N_HANDLES = len(HANDLE_NAMES)


@dataclass
class BridgeFrame:
    tokenized: object                     # TokenizedFrame (CPU tensors) from the model's tokenizer
    frame_bytes: bytes                    # the exact 31232-byte slot we decoded (Mode B writes these)
    vision_patches: Optional[np.ndarray]  # (2N, 768) float32 or None when not rendered this step
    vision_angles: Optional[np.ndarray]   # (2N, 2) float32
    sim_time: float
    dt: float
    frame: int
    n_contacts: int
    patches: np.ndarray                   # (n_patches, 10) raw model-unit fields (for logging)
    bones: np.ndarray                     # (24, 11) - 23 body bones + gaze
    # (VESTIBULAR_DIM,) float32, vestibular_adapter.VESTIBULAR_FIELDS order; ALWAYS that
    # width here. A payload saved before 2026-09-22 carries MIMo's raw six numbers (the
    # first frame after a resume is that saved payload); _payload_to_frame pads it with
    # pad_legacy_vestibular, whose `valid` field is 0 so a reader can skip that frame.
    vestibular: np.ndarray
    eye_angles: Tuple[Tuple[float, float], Tuple[float, float]]
    gaze_deg: Tuple[float, float]
    ctrl: np.ndarray
    env_index: int = 0
    meta: dict = field(default_factory=dict)
    # root-relative skin-body positions, one row per tactile patch (see RawStep.touch_pos);
    # None on paths that predate the touch point-cloud (old pickles, the UE transport)
    touch_pos: Optional[np.ndarray] = None
    # per patch: [share of force from the body's own parts, own parts on it over 4]
    touch_self: Optional[np.ndarray] = None
    # the strongest contact's direction, depth and arrival speed, per patch
    touch_contact: Optional[np.ndarray] = None
    touch_sensitivity: Optional[np.ndarray] = None
    skin_features: Optional[np.ndarray] = None  # optional full-resolution normalized skin packet


def _identity_handles(n_handles=N_HANDLES) -> np.ndarray:
    h = np.zeros((n_handles, 10), dtype=np.float32)
    h[:, 6] = 1.0      # quat w
    h[:, 9] = 1.0      # is_active
    return h


def _pack(env, raw, handle_echo: np.ndarray, wall_clock: float) -> Tuple[bytes, Optional[np.ndarray], Optional[np.ndarray]]:
    """Encode one RawStep to frame bytes. Vision travels as the raw uint8 eye images
    (393 KB) instead of float32 patches (1.6 MB): 20 workers at 6 steps/s through Windows
    pipes exhausted the system pipe pool (WinError 1450, 8-hour run take 4). The parent
    patchifies (see _payload_to_frame)."""
    vision_valid = raw.eye_left is not None
    buf = FE.encode_frame(frame_number=raw.frame, timestamp=raw.sim_time, dt=raw.dt, wall_clock=wall_clock,
                          gaze_az_deg=raw.gaze_az_el_deg[0], gaze_el_deg=raw.gaze_az_el_deg[1],
                          bones=raw.proprio.bones[:N_HANDLES], bone_hashes=raw.proprio.hashes[:N_HANDLES],
                          handles=handle_echo[:N_HANDLES], energy=raw.proprio.energy, body_sets=raw.proprio.body_sets,
                          patches=raw.patches, collisions=raw.collisions, vision_valid=vision_valid,
                          audio_spectral=raw.info.get("audio_spec"),
                          audio_meta=raw.info.get("audio_meta"))
    return bytes(buf), None, None


_PARENT_PATCHER = {}


def _parent_patcher(layout):
    """layout: ((name, size_px, fovy_deg), ...) - the env's camera pairs, coarsest first."""
    key = tuple((str(a), int(b), round(float(c), 3)) for a, b, c in layout)
    if key not in _PARENT_PATCHER:
        from mimo_bridge.vision_adapter import BinocularPatcher
        _PARENT_PATCHER[key] = BinocularPatcher(key)
    return _PARENT_PATCHER[key]


def _raw_to_payload(env, raw, handle_echo, wall_clock) -> dict:
    b, vp, va = _pack(env, raw, handle_echo, wall_clock)
    return dict(frame_bytes=b, eye_left=raw.eye_left, eye_right=raw.eye_right,
                extended_handles=handle_echo if len(handle_echo) > N_HANDLES else None,
                fovea_left=raw.fovea_left, fovea_right=raw.fovea_right,
                vision_layout=tuple(env.vision_layout),
                sim_time=raw.sim_time, dt=raw.dt,
                frame=raw.frame, n_contacts=raw.n_contacts, patches=raw.patches, bones=raw.proprio.bones,
                vestibular=raw.vestibular, eye_angles=raw.eye_angles, gaze_deg=raw.gaze_az_el_deg, ctrl=raw.ctrl,
                info=dict(raw.info), touch_pos=getattr(raw, "touch_pos", None),
                touch_self=getattr(raw, "touch_self", None),
                touch_contact=getattr(raw, "touch_contact", None),
                touch_sensitivity=getattr(raw, "touch_sensitivity", None))


def _payload_to_frame(p: dict, idx: int) -> BridgeFrame:
    tf = FE.decode_with_model_tokenizer(p["frame_bytes"])
    if p.get('extended_handles') is not None:
        # Explicit Mode A extension. The UE shared-memory byte contract remains 24 rows.
        import torch
        from tokenizer.config import TokenizerConfig
        cfg = TokenizerConfig()
        bones = np.asarray(p['bones'], dtype=np.float32)
        normalized = np.zeros((len(bones), 12), dtype=np.float32)
        normalized[:, :11] = bones
        normalized[:, 4:7] = (bones[:, 4:7] - bones[0, 4:7]) / cfg.position_range_cm
        normalized[:, 7:10] /= cfg.velocity_range_cms
        # The row index on the row's own 0..1 scale (P021, 2026-09-17). Raw, the 0..25
        # ramp carried 99.16% of this 312-number row's squared energy - block RMS 14.58
        # against 0.50 for the unit quaternions, 29.2x, where the motor row's own contract
        # test allows 15x - and it never varied between frames (std exactly 0), so the
        # encoder's bias-free first layer was reading a body sense that was 99 parts fixed
        # ramp. Divided by 25 the ramp is 15.83% and 1.17x the quaternion block. The same
        # constant is used by tokenizer/modalities/proprioception.py for the 24-row path,
        # and the receipt stamps it, so a checkpoint trained under the old raw ramp is
        # refused rather than silently read on the new scale (measured: loading old
        # weights under the new input moves the token by 0.4892 mean abs).
        normalized[:, 11] = np.arange(len(bones)) / cfg.proprio_index_range
        handles = np.array(p['extended_handles'], dtype=np.float32, copy=True)
        # TWO FRAMES FOR ONE GEOMETRY, deliberately, and nothing downstream says so. The
        # proprioception rows above are made relative to row 0 (wrist_L) before the /200;
        # this achieved-pose half carries the SAME 26 bodies in world centimetres / 200
        # (motor_to_echo's docstring), so the two blocks differ by the wrist_L offset / 200
        # on every handle. The command half is world-framed too (command_for_encoder divides
        # by 200 with no origin shift), so shifting only this line would trade a
        # proprioception/motor frame mismatch for a commanded/achieved one inside one row.
        handles[:, :3] /= cfg.position_range_cm
        tf.proprio = torch.from_numpy(normalized)
        tf.motor_handles = torch.from_numpy(handles)
    vp = va = None
    if p.get("eye_left") is not None:
        layout = p.get("vision_layout") or (("wide", p["eye_left"].shape[0], 60.0),)
        patcher = _parent_patcher(layout)
        images = {"wide": (p["eye_left"], p["eye_right"])}
        if p.get("fovea_left") is not None:
            images["fovea"] = (p["fovea_left"], p["fovea_right"])
        vp, va = patcher(images, p["eye_angles"][0], p["eye_angles"][1])
    from mimo_bridge.vestibular_adapter import pad_legacy_vestibular
    return BridgeFrame(tokenized=tf, frame_bytes=p["frame_bytes"], vision_patches=vp,
                       vision_angles=va, sim_time=p["sim_time"], dt=p["dt"], frame=p["frame"],
                       n_contacts=p["n_contacts"], patches=p["patches"], bones=p["bones"],
                       vestibular=pad_legacy_vestibular(p["vestibular"]),
                       eye_angles=p["eye_angles"], gaze_deg=p["gaze_deg"], ctrl=p["ctrl"], env_index=idx,
                       meta=p.get("info", {}),
                       touch_pos=p.get("touch_pos"), touch_self=p.get("touch_self"),
                       touch_contact=p.get("touch_contact"),
                       touch_sensitivity=p.get("touch_sensitivity"))


def motor_to_echo(motor9: np.ndarray, handle_mask: Optional[Sequence[bool]],
                  bones: Optional[np.ndarray] = None) -> np.ndarray:
    """The motor slice used to echo the model's own last command, which made the motor loss
    grade the model on copying itself (self-imitation wart, fixed 2026-08-20). Now it carries
    the ACHIEVED handle state read from physics: pose from the bones (world cm + world quat,
    the same 23 bones in handle order), plus the commanded radius/strength (the body has no
    achieved counterpart for those). The motor loss therefore grades commands against what
    the body really is, and the controller is conditioned on the real pose."""
    count = len(bones) if bones is not None else (len(motor9) if motor9 is not None else N_HANDLES)
    echo = np.zeros((count, 10), dtype=np.float32)
    if bones is not None:
        echo[:, 0:3] = bones[:, 4:7]        # achieved position, world cm
        echo[:, 3:7] = bones[:, 0:4]        # achieved orientation, world quat xyzw
    elif motor9 is not None:
        echo[:, :9] = motor9
    if motor9 is not None:
        echo[:, 7] = motor9[:, 7]           # commanded radius
        echo[:, 8] = motor9[:, 8]           # commanded strength
    echo[:, 9] = 1.0 if handle_mask is None else np.asarray(handle_mask, dtype=np.float32)
    return echo


def _hand_state(env):
    control = getattr(env, 'hand_target_control', None)
    if control is None:
        return None
    cache = getattr(env, '_hand_target_cache', None)
    return dict(geometry=copy.deepcopy(control.geometry),
                current=np.array(env.data.qpos[control._joint_addresses], copy=True),
                velocity=np.array(env.data.qvel[control._joint_dof_addresses], copy=True),
                held=None if cache is None else np.array(cache['joint_goal'], copy=True))


def _observe_frame(recorder, env, raw, observe, order=None) -> None:
    """The developmental observer's world recorder (observer_record.py) reads this frame.

    WATCHING THE BODY MUST NEVER STOP IT, the live view's rule: any failure turns the recorder
    off with its reason on disk and the world keeps stepping. It runs after the reply to the
    learner is built (and, in a worker, sent), reads the world only, and returns nothing. order:
    the (motor, gaze, mask, ctrl, hand joint goal) this frame was handed, which the recorder
    copies (never writes) at a burst's first frame: the yoked baseline replays them."""
    if recorder is None or not observe or not recorder.enabled:
        return
    try:
        recorder.on_frame(env, raw, observe, order)
    except Exception:
        recorder.fail(traceback.format_exc())


def _observer_open(arg: dict, body: int):
    from mimo_bridge.observer_record import WorldRecorder
    opts = {k: arg[k] for k in ("chunk_steps", "images_every_s", "ring_rays", "extra",
                                "budget_bytes", "reserve_bytes") if k in arg}
    return WorldRecorder(arg["dir"], body=body, run=arg.get("run", ""), **opts)


# ============================================================================ single env
class SingleEnvSource:
    def __init__(self, **env_kwargs):
        from mimo_bridge.env import BridgeEnv
        self._BridgeEnv = BridgeEnv
        self.env_kwargs = dict(env_kwargs)
        self.env = BridgeEnv(**self.env_kwargs)
        self.n_handles = len(self.env.proprio.body_ids)
        self.n_envs = 1
        self._echo = _identity_handles(self.n_handles)
        self._t0 = time.time()
        self._last_payload = None
        self.observer = None

    def observer_open(self, out_dir: str, run: str = "", **opts):
        """Start the developmental observer's world recorder for this body (see datasource
        _observe_frame). Never part of env_kwargs: a recorder must not change the configuration
        a world snapshot is checked against."""
        if self.observer is not None:
            self.observer.close()
        self.observer = _observer_open(dict(opts, dir=out_dir, run=run), 0)
        return [self.observer.dir]

    def observer_close(self):
        if self.observer is not None:
            self.observer.close()
            self.observer = None

    @property
    def n_patches(self):
        return self.env.n_patches

    def hand_state(self):
        return [_hand_state(self.env)]

    def reset(self) -> List[BridgeFrame]:
        raw = self.env.reset()
        self._echo = _identity_handles(self.n_handles)
        self._last_payload = _raw_to_payload(self.env, raw, self._echo, time.time() - self._t0)
        if self.observer is not None:
            self.observer.mark("reset", self.env)
        return [_payload_to_frame(self._last_payload, 0)]

    def step(self, motor: np.ndarray, gaze: Optional[np.ndarray] = None,
             handle_mask: Optional[Sequence[bool]] = None, ctrl: Optional[np.ndarray] = None,
             render: Optional[bool] = None, video_control: Optional[str] = None,
             live: Optional[dict] = None, live_body: int = 0,
             hand_joint_goal: Optional[np.ndarray] = None,
             observe: Optional[dict] = None) -> List[BridgeFrame]:
        m = np.asarray(motor).reshape(-1, self.n_handles, 9)[0]
        g = None if gaze is None else np.asarray(gaze).reshape(-1, 2)[0]
        c = None if ctrl is None else np.asarray(ctrl).reshape(-1)[:self.env.model.nu]
        hj = None if hand_joint_goal is None else np.asarray(hand_joint_goal).reshape(self.n_envs, -1)[0]
        raw = self.env.step(m, g, handle_mask, ctrl=c,
                            render=render, video_control=video_control,
                            **({} if hj is None else {'hand_joint_goal':hj}))
        self._echo = motor_to_echo(m, handle_mask, bones=raw.proprio.bones)
        if live:
            try:
                if live.get("frame"):
                    self.env.live_frame(live["frame"], size=live.get("size", 480),
                                        control=video_control, keep=live.get("keep", ""))
                if live.get("state"):
                    self.env.live_state(live["state"])
            except Exception:
                pass
        self._last_payload = _raw_to_payload(self.env, raw, self._echo, time.time() - self._t0)
        frames = [_payload_to_frame(self._last_payload, 0)]
        _observe_frame(self.observer, self.env, raw, observe, (m, g, handle_mask, c, hj))
        return frames

    def save_full_state(self):
        if self._last_payload is None: raise ValueError('world has no observed state to save')
        return [copy.deepcopy(dict(env_kwargs=self.env_kwargs,world=self.env.save_full_state(),
            echo=self._echo,payload=self._last_payload,elapsed=time.time()-self._t0))]

    def iter_world_blobs(self):
        """The world as the episode companion stores it (continuation_state.save): one pickle, the same dict
        save_full_state returns. Pickling needs no deep copy first; env.save_full_state makes its own."""
        if self._last_payload is None: raise ValueError('world has no observed state to save')
        from .episode_archive import world_blob
        blob = world_blob(dict(env_kwargs=self.env_kwargs, world=self.env.save_full_state(),
                               echo=self._echo, payload=self._last_payload, elapsed=time.time() - self._t0))
        yield dict(sim_time=self._last_payload['sim_time'], nbytes=len(blob)), blob

    def restore_full_state(self,states):
        if len(states)!=1 or states[0]['env_kwargs']!=self.env_kwargs:
            raise ValueError('world snapshot requires identical embodiment and environment configuration')
        s=states[0];self.env.restore_full_state(s['world'])
        self._echo=copy.deepcopy(s['echo']);self._last_payload=copy.deepcopy(s['payload'])
        self._t0=time.time()-s['elapsed']
        if self.observer is not None:
            self.observer.mark("restore", self.env)
        # Collecting again would advance sensory histories at the same physical time.
        return [_payload_to_frame(self._last_payload,0)]

    def start_video(self, path: str, n_frames: int, env_idx: int = 0):
        self.env.start_video(path, n_frames)      # one body: env_idx is moot

    def save_views(self, path: str):
        self.env.save_views(path)

    def live_frame(self, path: str, env_idx: int = 0, size: int = 480, control=None) -> float:
        return self.env.live_frame(path, size=size, control=control)

    def live_state(self, path: str, env_idx: int = 0) -> float:
        return self.env.live_state(path)

    def rebuild(self, **changes):
        self.env.close()
        self.env_kwargs.update(changes)
        self.env = self._BridgeEnv(**self.env_kwargs)
        self.n_handles = len(self.env.proprio.body_ids)
        if self.observer is not None:
            self.observer.mark("rebuild", self.env)

    def close(self):
        self.observer_close()
        self.env.close()


# ============================================================================ worker pool
def _send_world_blob(conn, env, env_kwargs, echo, last_payload, t0):
    """The worker's answer to 'save_full_state_blob': a header, then its world pickled by the worker itself as
    one message (the dict 'save_full_state' sends, as episode_archive.world_blob pickles it). The trainer writes
    the bytes into the companion without unpickling them, and asks the next worker only after this one is read,
    so one world's copy and pickle exist at a time (2026-10-01)."""
    # a failure before anything is sent is answered and the worker keeps serving: the trainer's hourly save
    # fails and training continues with the pool in step (the old 'save_full_state' ended the worker instead)
    try:
        if last_payload is None: raise ValueError('world has no observed state to save')
        from mimo_bridge.episode_archive import world_blob
        blob = world_blob(dict(env_kwargs=env_kwargs, world=env.save_full_state(), echo=echo,
                               payload=last_payload, elapsed=time.time() - t0))
    except Exception:
        conn.send(('err', traceback.format_exc()))
        return
    conn.send(('ok', dict(sim_time=last_payload['sim_time'], nbytes=len(blob))))
    conn.send_bytes(blob)


def _worker_main(conn, env_kwargs: dict, idx: int):
    """Runs in a child process: owns one BridgeEnv, answers commands from the pipe."""
    try:
        # keep BLAS threads from fighting across workers
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        from mimo_bridge.env import BridgeEnv
        env = BridgeEnv(**env_kwargs)
        echo = _identity_handles(len(env.proprio.body_ids))
        t0 = time.time()
        last_payload = None
        recorder = None            # the developmental observer's world recorder, when asked for
        conn.send(("ready", dict(n_patches=env.n_patches, nu=int(env.model.nu), dt=env.dt, scene=env.scene_path)))
        while True:
            cmd, arg = conn.recv()
            if cmd == "reset":
                raw = env.reset(); echo = _identity_handles(len(env.proprio.body_ids))
                last_payload = _raw_to_payload(env, raw, echo, time.time() - t0)
                conn.send(("ok", last_payload))
                if recorder is not None:
                    recorder.mark("reset", env)
            elif cmd == "observer":
                # start (a dict) or stop (None) the world recorder; never touches env_kwargs
                if recorder is not None:
                    recorder.close()
                    recorder = None
                if arg:
                    recorder = _observer_open(arg, idx)
                conn.send(("ok", None if recorder is None else recorder.dir))
            elif cmd == 'hand_state':
                conn.send(('ok', _hand_state(env)))
            elif cmd == "step":
                motor, gaze, mask, ctrl, render, video_control, live, hand_joint_goal, observe = (arg + (None,)*9)[:9]
                raw = env.step(motor, gaze, mask, ctrl=ctrl, render=render, video_control=video_control,
                               **({} if hand_joint_goal is None else {'hand_joint_goal':hand_joint_goal}))
                if live:
                    # RIDES THE STEP THE WORKER IS ALREADY ANSWERING. A separate command
                    # cost a whole extra round trip through the pipe while the other seven
                    # workers waited: measured 2026-09-21, one frame every model step took
                    # the run from 1.206 to 0.972 steps per second, 19% slower, for a
                    # render that itself takes 4.3 ms of an 829 ms step.
                    try:
                        if live.get("frame"):
                            env.live_frame(live["frame"], size=live.get("size", 480),
                                           control=video_control, keep=live.get("keep", ""))
                        if live.get("state"):
                            env.live_state(live["state"])
                    except Exception:
                        pass
                if motor is not None:
                    echo = motor_to_echo(motor, mask, bones=raw.proprio.bones)
                last_payload = _raw_to_payload(env, raw, echo, time.time() - t0)
                conn.send(("ok", last_payload))
                # after the reply: the learner never waits on the recorder for this frame
                _observe_frame(recorder, env, raw, observe, (motor, gaze, mask, ctrl, hand_joint_goal))
            elif cmd == 'save_full_state':
                if last_payload is None: raise ValueError('world has no observed state to save')
                conn.send(('ok',dict(env_kwargs=env_kwargs,world=env.save_full_state(),
                    echo=echo,payload=last_payload,elapsed=time.time()-t0)))
            elif cmd == 'save_full_state_blob':
                _send_world_blob(conn, env, env_kwargs, echo, last_payload, t0)
            elif cmd == 'restore_full_state':
                if arg['env_kwargs']!=env_kwargs: raise ValueError('world snapshot configuration mismatch')
                env.restore_full_state(arg['world']);echo=copy.deepcopy(arg['echo'])
                last_payload=copy.deepcopy(arg['payload']);t0=time.time()-arg['elapsed']
                conn.send(('ok',last_payload))
                if recorder is not None:
                    recorder.mark("restore", env)
            elif cmd == "video":
                env.start_video(arg["path"], arg["n_frames"])
                conn.send(("ok", None))
            elif cmd == "views":
                env.save_views(arg["path"])
                conn.send(("ok", None))
            elif cmd in ("live", "livestate"):
                # LOOKING AT THE BODY MUST NEVER STOP IT. A picture or a state file is a
                # convenience; anything that goes wrong inside one is reported as a cost of
                # 0 ms and the world keeps stepping. Measured 2026-09-21: an imageio writer
                # that refused a ".part" suffix raised here, the worker exited, and the
                # trainer died on EOFError from the pipe 2.4 minutes into the run.
                try:
                    ms = (env.live_frame(arg["path"], size=arg.get("size", 480),
                                         control=arg.get("control")) if cmd == "live"
                          else env.live_state(arg["path"]))
                    conn.send(("ok", ms))
                except Exception:
                    conn.send(("ok", 0.0))
            elif cmd == "rebuild":
                env.close()
                del env                       # free the old MuJoCo model/data (400 MB arena each) BEFORE
                import gc; gc.collect()       # building the new one: 20 workers x 2 copies blew the RAM
                env_kwargs.update(arg)
                last_err = None
                for attempt in range(4):           # GL / file races are transient: retry with a pause
                    try:
                        time.sleep(0.2 * idx % 3.0 + attempt * 1.5)
                        env = BridgeEnv(**env_kwargs)
                        last_err = None
                        break
                    except Exception:
                        last_err = traceback.format_exc()
                if last_err is not None:
                    raise RuntimeError(last_err)
                echo = _identity_handles(len(env.proprio.body_ids))
                conn.send(("ok", dict(n_patches=env.n_patches, scene=env.scene_path)))
                if recorder is not None:
                    recorder.mark("rebuild", env)
            elif cmd == "close":
                if recorder is not None:
                    recorder.close()
                env.close(); conn.send(("ok", None)); break
            else:
                conn.send(("err", f"unknown command {cmd}"))
    except Exception:
        try:
            conn.send(("err", traceback.format_exc()))
        except Exception:
            pass


class PoolSource:
    """N BridgeEnvs in N processes. step() sends all actions, then waits for all frames."""

    def __init__(self, n_envs: int, env_kwargs: dict, per_env_kwargs: Optional[List[dict]] = None,
                 start_timeout_s: float = 600.0):
        ctx = mp.get_context("spawn")
        self.n_envs = n_envs
        self.conns = []
        self.procs = []
        self.env_kwargs = [dict(env_kwargs) for _ in range(n_envs)]
        for i in range(n_envs):
            kw = self.env_kwargs[i]
            # DISTINCT streams per body, always. setdefault only fired when the caller
            # passed no seed, and the trainer always passes one - so all 16 workers shared
            # a single RNG stream: identical saccade schedules, identical ball-placement
            # draws, sixteen bodies wandering in lockstep. Offset keeps determinism.
            kw["seed"] = int(kw.get("seed", 0)) + i
            kw["tag"] = kw.get("tag", "") + f"_w{i}"
            if per_env_kwargs and i < len(per_env_kwargs):
                kw.update(per_env_kwargs[i])
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker_main, args=(child, kw, i), daemon=True)
            p.start()
            self.conns.append(parent)
            self.procs.append(p)
        self.info = []
        for c in self.conns:
            if not c.poll(start_timeout_s):
                raise TimeoutError("worker did not start")
            tag, info = c.recv()
            if tag != "ready":
                raise RuntimeError(f"worker failed to start:\n{info}")
            self.info.append(info)
        self.n_patches = self.info[0]["n_patches"]

    RECV_TIMEOUT_S = 180.0   # a worker that says nothing for 3 minutes is hung, not slow

    def _recv(self, i):
        """recv with a deadline so a hung worker turns into a clean error (which the trainer
        answers with a crash snapshot and exit) instead of an invisible forever-hang."""
        if getattr(self, '_out_of_step', None):
            raise RuntimeError(self._out_of_step)
        if not self.conns[i].poll(self.RECV_TIMEOUT_S):
            raise RuntimeError(f"worker {i} did not answer for {self.RECV_TIMEOUT_S:.0f} s (hung)")
        return self.conns[i].recv()

    # A THREADED RECEIVE WAS TRIED HERE AND REMOVED (2026-09-17). The pool costs about
    # 8.5 ms per added body per model step (110 ms for 8 bodies, 198 for 16, 260 for 24,
    # against 59 for one body in-process), and the guess was that this loop - wait for
    # worker 0, read, decode, then worker 1 - was the reason. Measured with a thread per
    # worker at 24 bodies: 52.8 ms per tick against 55.7 serial, within noise. Timing the
    # two halves of each serial receive showed why: the pipe read is 0.19 ms per body and
    # the decode 0.88 ms - 16 ms of a 56 ms tick between them; the other 38 ms is WAITING
    # for the workers to finish, which no receive order can shorten. The workers finish
    # later as their number grows even though the physics parallelises across 32 cores,
    # and the one resource they all share is the GPU each renders its two eyes on
    # (tools/pool_step_cost.py --no-vision measures that directly). Do not re-thread this
    # loop; move the cost off the workers' shared resource instead.
    def _collect(self) -> List[BridgeFrame]:
        out = []
        for i in range(len(self.conns)):
            tag, payload = self._recv(i)
            if tag != "ok":
                raise RuntimeError(f"worker {i} error:\n{payload}")
            out.append(_payload_to_frame(payload, i))
        return out

    def reset(self) -> List[BridgeFrame]:
        for c in self.conns:
            c.send(("reset", None))
        return self._collect()

    def save_full_state(self):
        # one worker at a time (2026-10-01): asked together, every worker held its world's copy and that copy's
        # pickle until this loop reached it, all eight at once; the states returned are the same
        states=[]
        for i,c in enumerate(self.conns):
            c.send(('save_full_state',None))
            tag,state=self._recv(i)
            if tag!='ok':raise RuntimeError(f'worker {i} snapshot failed: {state}')
            states.append(state)
        return states

    def iter_world_blobs(self):
        """Each world's pickle bytes in worker order, one worker asked at a time (continuation_state.save writes
        each into the companion before the next is asked for). Yields (dict(sim_time, nbytes), bytes)."""
        for i,c in enumerate(self.conns):
            c.send(('save_full_state_blob',None))
            tag,header=self._recv(i)
            if tag!='ok':raise RuntimeError(f'worker {i} snapshot failed: {header}')
            if not c.poll(self.RECV_TIMEOUT_S):
                # its late message would answer the next command: every later receive refuses instead (the
                # hourly save's failure handler would otherwise step this pool again)
                self._out_of_step = (f"worker {i} did not send its world for {self.RECV_TIMEOUT_S:.0f} s (hung); "
                                     "the pool is out of step")
                raise RuntimeError(self._out_of_step)
            blob=c.recv_bytes()
            if len(blob)!=header['nbytes']:
                raise RuntimeError(f'worker {i} sent {len(blob)} bytes of world, announced {header["nbytes"]}')
            yield header,blob
            del blob

    def restore_full_state(self,states):
        if len(states)!=self.n_envs:raise ValueError('snapshot body count changed')
        for c,state in zip(self.conns,states):c.send(('restore_full_state',state))
        return self._collect()

    def observer_open(self, out_dir: str, run: str = "", **opts):
        """Start the developmental observer's world recorder in every worker (body i records
        under <out_dir>/b<i>). Sent as its own command, never through env_kwargs."""
        for c in self.conns:
            c.send(("observer", dict(opts, dir=out_dir, run=run)))
        dirs = []
        for i in range(len(self.conns)):
            tag, value = self._recv(i)
            if tag != "ok":
                raise RuntimeError("worker %d observer start failed:\n%s" % (i, value))
            dirs.append(value)
        return dirs

    def observer_close(self):
        for c in self.conns:
            c.send(("observer", None))
        for i in range(len(self.conns)):
            self._recv(i)

    def step(self, motor: np.ndarray, gaze: Optional[np.ndarray] = None,
             handle_mask: Optional[Sequence[bool]] = None, ctrl: Optional[np.ndarray] = None,
             render: Optional[bool] = None, video_control: Optional[str] = None,
             live: Optional[dict] = None, live_body: int = 0,
             hand_joint_goal: Optional[np.ndarray] = None,
             observe: Optional[dict] = None) -> List[BridgeFrame]:
        m = np.asarray(motor, dtype=np.float64).reshape(self.n_envs, -1, 9)
        g = None if gaze is None else np.asarray(gaze, dtype=np.float64).reshape(self.n_envs, 2)
        cc = None if ctrl is None else np.asarray(ctrl, dtype=np.float64).reshape(self.n_envs, -1)
        hj = None if hand_joint_goal is None else np.asarray(hand_joint_goal).reshape(self.n_envs, -1)
        watched = int(live_body) % len(self.conns)
        for i, c in enumerate(self.conns):
            c.send(("step", (m[i], None if g is None else g[i], handle_mask,
                             None if cc is None else cc[i], render, video_control,
                             live if (live and i == watched) else None,
                             None if hj is None else hj[i], observe)))
        return self._collect()

    def hand_state(self):
        for c in self.conns:
            c.send(('hand_state', None))
        result = []
        for i in range(self.n_envs):
            tag, value = self._recv(i)
            if tag != 'ok':
                raise RuntimeError('Hand geometry query failed: '+str(value))
            result.append(value)
        return result

    def start_video(self, path: str, n_frames: int, env_idx: int = 0):
        """Record on ONE worker's camera. env_idx picks which body - a moment video
        must show the body that had the moment (review 2026-09-06: always-worker-0
        filmed a bystander 7/8 of the time)."""
        i = int(env_idx) % len(self.conns)
        self.conns[i].send(("video", {"path": path, "n_frames": n_frames}))
        tag, _ = self._recv(i)
        if tag != "ok":
            raise RuntimeError("video command failed")

    def save_views(self, path: str):
        self.conns[0].send(("views", {"path": path}))
        tag, _ = self._recv(0)
        if tag != "ok":
            raise RuntimeError("views command failed")

    def live_state(self, path: str, env_idx: int = 0) -> float:
        """One body's physical state for an outside viewer. Costs a small file write."""
        i = int(env_idx) % len(self.conns)
        self.conns[i].send(("livestate", {"path": path}))
        tag, ms = self._recv(i)
        if tag != "ok":
            raise RuntimeError("live state command failed")
        return float(ms or 0.0)

    def live_frame(self, path: str, env_idx: int = 0, size: int = 480, control=None) -> float:
        """One frame from ONE body's watch camera, overwriting `path`. Returns the
        milliseconds it cost inside that worker, so the trainer can report what a live
        view is charging the run."""
        i = int(env_idx) % len(self.conns)
        self.conns[i].send(("live", {"path": path, "size": size, "control": control}))
        tag, ms = self._recv(i)
        if tag != "ok":
            raise RuntimeError("live frame command failed")
        return float(ms or 0.0)

    def rebuild(self, **changes):
        for kw in self.env_kwargs:kw.update(changes)
        for c in self.conns:
            c.send(("rebuild", dict(changes)))
        for i in range(len(self.conns)):
            tag, payload = self._recv(i)
            if tag != "ok":
                raise RuntimeError(f"worker {i} rebuild error:\n{payload}")

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
