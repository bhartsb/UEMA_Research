# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Section 27: test the observer before trusting it.

Validation worlds sit beside the playpen and never become the learner's world: a real crib
world (the training run's own settings) moved KINEMATICALLY - each frame sets the body's pose and
MuJoCo's forward dynamics computes contacts and forces exactly as in training - through scripted
positive, negative and boundary cases, recorded by the same world recorder and read by the same
analysis. Plus four checks that are not detector cases:

  finger sign      raising each digit's flexion hinges moves its fingertip toward the palm
  panel order      each painted panel, rendered from in front at both ends, shows the motif the
                   recorder's cell mapping names there (observer_record.ART_FIRST_AT_NEGATIVE_END)
  fast lane        the same impulse on the head, delivered in 10 ms or spread over 500 ms (real
                   physics): the fast lane's peak must separate them while a 2 s average does not
  replay           the recorded cases' detections, compared with the stored golden file of this
                   detector version: the same recording must give the same detections after any
                   code change (a changed recording is reported apart from a changed detector)

Reported per case (expected present / absent capabilities, what was found, the scripted truth
against the measured value) and per detector (true and false positives and negatives across the
cases), in docs/gain_of_function_v4/observer_validation/validation_report.json.

Added 2026-09-24 evening (docs/gain_of_function_v4/agents/observer_validation.md):
  the grasp family      cases breaking one link at a time (contact, opposition, lift, retention,
                        carry, release, and an object the world moves into a still hand), on a
                        grasp found by physics (_grasp, _settle)
  eight more detectors  G01, G03, G05, R03, B06, H05, HX2, HX1: positive, negative and boundary
  qualifiers            an expectation may name the detectors' provenance flags: "CAP@own",
                        "CAP@world_moved", "CAP@scripted", "CAP@caregiver_touch" (_match)
  the scene hook        a case may move what the world moves (the toy, the world's eye movers)
  KNOWN_DEFECTS         cases whose contract expectation a detector does not meet yet, each with the
                        defect; reports split failures into these and unexpected ones
  BOUNDARY, coverage()  which cases probe which detector's line, and every detector's cases
  ModelWorld, dry_run   the cases in the saved world model without BridgeEnv (about 1 GB), for
                        checking scripts beside training; never the golden recordings
  babble_rates()        the false-positive table per detector from the kept babble summaries
                        (observer_validation/babble_rates.json)
  `python -m mimo_bridge.tools.observer.validate babble-rates | dry-run` runs the last two.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from .kinematics import DIGITS, FINGERS, SIDES

HERE = os.path.dirname(os.path.abspath(__file__))
STEP_REPS = 5
# THE SCRIPTED MOTOR COMMANDS (detector v9). The hand and wrist rows count a movement only when the
# moving joints' own motors carried a command (detectors.MOTOR_ON), so a scripted case states the
# commands with the trajectory: motors="drive" (the default) gives every motor whose joint the
# script moves in a frame the command SCRIPT_COMMAND in the direction of that movement (the strength
# random-order babble's orders reach at most, baseline.ORDER_STRENGTH), and 0 to the rest;
# motors="against" gives the same command OPPOSITE to each movement (the hand moved by the world while its
# motors resist: detectors._motor_work is negative, detector v9); motors="off" gives every motor 0, as a
# limp body has (a case's pose function carries
# pose.motors = "off"). Before v9 every case recorded zero commands: the world is posed, not driven.
SCRIPT_COMMAND = 0.3
SCRIPT_MOVING_RADPS = 1e-3         # a joint moving at least this fast in a frame is being driven


class ScriptWorld:
    def __init__(self, run_dir: str, scratch: str):
        import mujoco
        from mimo_bridge.env import BridgeEnv
        from .baseline import world_kwargs_like
        self.mj = mujoco
        self.scratch = scratch
        kw, self.reps, self.hold = world_kwargs_like(run_dir, os.path.join(scratch, "world_scratch"), 7)
        self.env = BridgeEnv(**kw)
        self.env.reset()
        self.m, self.d = self.env.model, self.env.data
        # the adult, the critters and the toy's carrier leave the pen: the cases script the body
        # alone. Mocap bodies are moved, never deleted, so the world model is the training one.
        self.env.caregiver = None
        self._init_common()

    def _init_common(self):
        m, d = self.m, self.d
        self.far = {}
        for b in range(m.nbody):
            name = m.body(b).name
            if int(m.body_mocapid[b]) >= 0 and name != "target":
                self.far[int(m.body_mocapid[b])] = np.array([3.0 + 0.5 * len(self.far), 3.0, 3.0])
        self.q_base = np.array(d.qpos)
        self.mocap_base = np.array(d.mocap_pos)
        self.mocap_quat_base = np.array(d.mocap_quat)
        self.jq = {}
        self.jd = {}
        for j in range(m.njnt):
            nm = m.joint(j).name.split(":")[-1]
            self.jq[nm] = int(m.jnt_qposadr[j])
            self.jd[nm] = int(m.jnt_dofadr[j])
        self.jrange = {m.joint(j).name.split(":")[-1]: (float(m.jnt_range[j][0]), float(m.jnt_range[j][1]))
                       for j in range(m.njnt)}
        # every motor with a joint transmission, and the velocity index of its joint
        self.motor_dofs = [(u, int(m.jnt_dofadr[int(m.actuator_trnid[u][0])])) for u in range(int(m.nu))
                           if int(m.actuator_trntype[u]) == 0]

    def command(self, qv: np.ndarray, motors: str) -> None:
        """The scripted motor commands of one frame (SCRIPT_COMMAND above): toward each driven
        joint's movement, 0 elsewhere; all 0 with motors 'off'."""
        d = self.d
        d.ctrl[:] = 0.0
        if motors == "off":
            return
        if motors not in ("drive", "against"):
            raise ValueError("motors must be 'drive', 'against' or 'off', not %r" % (motors,))
        sign = 1.0 if motors == "drive" else -1.0
        for u, dof in self.motor_dofs:
            v = float(qv[dof])
            if abs(v) > SCRIPT_MOVING_RADPS:
                d.ctrl[u] = sign * (SCRIPT_COMMAND if v > 0 else -SCRIPT_COMMAND)

    # ------------------------------------------------------------------ helpers
    def base(self) -> np.ndarray:
        return np.array(self.q_base)

    def place_far(self):
        d = self.d
        d.mocap_pos[:] = self.mocap_base
        d.mocap_quat[:] = self.mocap_quat_base          # a case that turned the toy leaves no trace
        for k, p in self.far.items():
            d.mocap_pos[k] = p

    def fk(self, q):
        self.d.qpos[:] = q
        self.mj.mj_kinematics(self.m, self.d)
        self.mj.mj_comPos(self.m, self.d)

    def set_curl(self, q, side: str, digit: str, deg: float):
        pre = "left_" if side == "L" else "right_"
        key = {"index": "ff", "middle": "mf", "ring": "rf", "little": "lf", "thumb": "th"}[digit]
        roles = ("middle", "distal") if digit == "thumb" else ("knuckle", "middle", "distal")
        for r in roles:
            nm = "%s%s_%s" % (pre, key, r)
            lo, hi = self.jrange[nm]
            q[self.jq[nm]] = float(np.clip(math.radians(deg), lo, hi))
        return q

    def tip_local(self, side: str, digit: str):
        m = self.m
        pre = "left_" if side == "L" else "right_"
        key = {"index": "ff", "middle": "mf", "ring": "rf", "little": "lf", "thumb": "th"}[digit]
        b = m.body(pre + key + "distal").id
        g = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b][0]
        R = np.zeros(9)
        self.mj.mju_quat2Mat(R, m.geom_quat[g])
        axis = R.reshape(3, 3)[:, 2]
        return b, np.asarray(m.geom_pos[g]) + axis * float(m.geom_size[g][1] + m.geom_size[g][0])

    def point(self, q, body: int, local: np.ndarray) -> np.ndarray:
        self.fk(q)
        return np.asarray(self.d.xpos[body]) + self.d.xmat[body].reshape(3, 3) @ np.asarray(local)

    def ik(self, q, body: int, local, target, joints: List[str], iters: int = 200, lam: float = 0.02,
           tol: float = 5e-4):
        m, d = self.m, self.d
        q = np.array(q)
        cols = [self.jd[j] for j in joints]
        adrs = [self.jq[j] for j in joints]
        for _ in range(iters):
            p = self.point(q, body, local)
            e = np.asarray(target) - p
            if np.linalg.norm(e) < tol:
                break
            jacp = np.zeros((3, m.nv))
            self.mj.mj_jac(m, d, jacp, None, p, body)
            J = jacp[:, cols]
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(3), e)
            for a, j, v in zip(adrs, joints, dq):
                lo, hi = self.jrange[j]
                q[a] = float(np.clip(q[a] + 0.5 * v, lo, hi))
        return q

    def ik_points(self, q, targets, joints: List[str], iters: int = 400, lam: float = 0.01,
                  tol: float = 3e-4, weights=None):
        """Damped least squares on several points at once: targets is a list of (body, local
        point, world target). Returns (q, the largest remaining distance in metres)."""
        m, d = self.m, self.d
        q = np.array(q)
        cols = [self.jd[j] for j in joints]
        adrs = [self.jq[j] for j in joints]
        wts = np.ones(len(targets)) if weights is None else np.asarray(weights, dtype=np.float64)
        worst = float("inf")
        for _ in range(iters):
            self.fk(q)
            rows, errs, dist = [], [], []
            for (body, local, target), wt in zip(targets, wts):
                p = np.asarray(self.d.xpos[body]) + self.d.xmat[body].reshape(3, 3) @ np.asarray(local)
                e = np.asarray(target) - p
                dist.append(float(np.linalg.norm(e)))
                jacp = np.zeros((3, m.nv))
                self.mj.mj_jac(m, d, jacp, None, p, body)
                rows.append(wt * jacp[:, cols])
                errs.append(wt * e)
            worst = max(dist)
            if worst < tol:
                break
            J = np.concatenate(rows)
            e = np.concatenate(errs)
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(len(e)), e)
            for a, j, v in zip(adrs, joints, dq):
                lo, hi = self.jrange[j]
                q[a] = float(np.clip(q[a] + 0.5 * v, lo, hi))
        return q, worst

    def ik_pose(self, q, body: int, pos, mat, joints: List[str], iters: int = 400, lam: float = 0.01,
                tol: float = 2e-4):
        """One body's frame to a world position and orientation (3x3). Returns (q, position error
        in metres, orientation error in degrees)."""
        m, d = self.m, self.d
        q = np.array(q)
        cols = [self.jd[j] for j in joints]
        adrs = [self.jq[j] for j in joints]
        mat = np.asarray(mat).reshape(3, 3)
        ep, er = float("inf"), float("inf")
        for _ in range(iters):
            self.fk(q)
            R = self.d.xmat[body].reshape(3, 3)
            e_p = np.asarray(pos) - np.asarray(self.d.xpos[body])
            e_r = 0.5 * sum(np.cross(R[:, k], mat[:, k]) for k in range(3))
            ep = float(np.linalg.norm(e_p))
            er = math.degrees(math.asin(min(1.0, float(np.linalg.norm(e_r)))))   # |e_r| = sin(angle)
            if ep < tol and er < 0.05:
                break
            jacp = np.zeros((3, m.nv))
            jacr = np.zeros((3, m.nv))
            self.mj.mj_jacBody(m, d, jacp, jacr, body)
            J = np.concatenate([jacp[:, cols], 0.1 * jacr[:, cols]])
            e = np.r_[e_p, 0.1 * e_r]
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
            for a, j, v in zip(adrs, joints, dq):
                lo, hi = self.jrange[j]
                q[a] = float(np.clip(q[a] + 0.5 * v, lo, hi))
        return q, ep, er

    def frame(self, q, body: int):
        """(position, 3x3 rotation) of a body at pose q."""
        self.fk(q)
        return np.array(self.d.xpos[body]), np.array(self.d.xmat[body]).reshape(3, 3)

    def contacts_now(self, q, mocap: Optional[Dict[int, np.ndarray]] = None) -> List[tuple]:
        """(body name, body name, penetration in metres) of every contact at pose q, for checking
        a scripted grasp before it is recorded."""
        m, d = self.m, self.d
        self.place_far()
        if mocap:
            for k, p in mocap.items():                   # a position, or a position and quaternion
                p = np.asarray(p, dtype=np.float64)
                d.mocap_pos[k] = p[:3]
                if p.size == 7:
                    d.mocap_quat[k] = p[3:7]
        d.qpos[:] = q
        self.mj.mj_forward(m, d)
        out = []
        for i in range(int(d.ncon)):
            c = d.contact[i]
            out.append((m.body(int(m.geom_bodyid[c.geom1])).name, m.body(int(m.geom_bodyid[c.geom2])).name,
                        float(-c.dist)))
        return out

    def mocap_id(self, body_name: str) -> int:
        return int(self.m.body_mocapid[self.m.body(body_name).id])

    def free_adr(self, body_name: str) -> int:
        """The qpos address of the free joint that carries this body: its own or its nearest
        ancestor's (on H84 the infant's free joint 'mimo_orientation' sits on 'mimo_location',
        the hip's parent)."""
        m = self.m
        b = m.body(body_name).id
        while b > 0:
            for j in range(m.njnt):
                if int(m.jnt_bodyid[j]) == b and int(m.jnt_type[j]) == int(self.mj.mjtJoint.mjJNT_FREE):
                    return int(m.jnt_qposadr[j])
            b = int(m.body_parentid[b])
        raise KeyError(body_name)

    # ------------------------------------------------------------------ recording
    def record(self, name: str, n_steps: int, pose: Callable[[float], np.ndarray],
               eyes: Optional[Callable[[float], None]] = None,
               scene: Optional[Callable[[float, "ScriptWorld"], None]] = None,
               motors: Optional[str] = None) -> str:
        """Record one case. pose(t) gives qpos at time t (in model steps: step s's burst covers
        t = s + 0.2 ... s + 1.0 and the record keeps the last frame); eyes(t, q) may re-aim the
        eyes; scene(t, world) may move what the WORLD moves (mocap bodies such as the toy, the
        world's eye movers) before each frame is computed. The world's eye movers are restored
        after the case. motors: 'drive' or 'off' (command(); by default the pose function's
        `motors` attribute, else 'drive'). A pose function's `hold` attribute, hold(t) -> {motor:
        command}, sets those motors' commands on top (a grip the digits' motors hold: THE GRIP IS
        HELD BY THE HAND'S MOTORS)."""
        from mimo_bridge.observer_record import WorldRecorder
        env, m, d = self.env, self.m, self.d
        motors = motors or getattr(pose, "motors", "drive")
        hold = getattr(pose, "hold", None)
        out = os.path.join(self.scratch, name)
        if os.path.isdir(out):
            shutil.rmtree(out)
        rec = WorldRecorder(os.path.join(out, "observer", "world"), body=0, run="validate_" + name,
                            chunk_steps=1000, budget_bytes=2 ** 30, reserve_bytes=2 ** 30,
                            extra=dict(validation_case=name, motors=motors))
        movers = {k: getattr(env, k, None) for k in ("saccades", "reflex")}
        try:
            env.restart_clock(0.0)           # the clock and what keeps its own beside it (the virtual skin)
            env.frame = 0
            self.place_far()
            if scene is not None:
                scene(0.0, self)
            prev = pose(0.0)
            if eyes is not None:
                prev = eyes(0.0, prev)
            d.qpos[:] = prev
            d.qvel[:] = 0.0
            d.ctrl[:] = 0.0
            self.mj.mj_forward(m, d)
            rec.mark("reset")
            dt = float(env.dt)
            for s in range(n_steps):
                for rep in range(self.reps):
                    t = s + (rep + 1) / float(self.reps)
                    q = pose(t)
                    if eyes is not None:
                        q = eyes(t, q)
                    qv = np.zeros(m.nv)
                    self.mj.mj_differentiatePos(m, qv, dt, prev, q)
                    if scene is not None:
                        scene(t, self)
                    d.qpos[:] = q
                    d.qvel[:] = qv
                    self.command(qv, motors)
                    for u, c in (hold(t) if hold is not None else {}).items():
                        d.ctrl[u] = c                  # the motors holding a grip (THE GRIP IS HELD BY THE HAND'S MOTORS)
                    self.mj.mj_forward(m, d)
                    d.time += dt
                    env.frame += 1
                    rec.on_frame(env, None, dict(step=s, rep=rep, reps=self.reps, author="scripted",
                                                 latched=(rep == 0), order_seq=s, bout=False, rest=False))
                    prev = q
            rec.close()
        finally:
            for k, v in movers.items():
                setattr(env, k, v)
            d.ctrl[:] = 0.0                  # the next case's construction starts without commands
            if getattr(scene, "after", None) is not None:
                scene.after()                # what the case's scene changed in the model (_small_box_scene)
        if rec.error:
            raise RuntimeError(rec.error)
        return out

    # ------------------------------------------------------------------ driven by the body's motors
    def frame_step(self, ctrl: np.ndarray):
        """One frame of real dynamics under these motor commands: the world's own step (BridgeEnv,
        as the random-command baselines step it). Returns the world's reply."""
        return self.env.step(None, None, None, ctrl=np.asarray(ctrl, dtype=np.float64), render=False)

    def record_driven(self, name: str, case: "Driven") -> str:
        """Record one case the body's own motors drive, with real dynamics (detector v9; P05 needs
        real support forces, which a posed world cannot give): the case's start pose, then each frame
        the case's controller reads the body's state and sets every motor's command, and the world
        steps. The recorder stores the commands with the rest. Gravity, contacts and the whole body
        are free: what the recorder sees is physics.
        The world's own movers stay out of the case, as they do in a posed case (which never steps the
        world): its saccade generator and gaze reflex are off, its critters do not tick (they would put
        the squeaker back at its home by the feet and fly a visitor through view), and the toy's leash
        is re-anchored where the case's scene puts it (BridgeEnv.step pulls the toy back within 12 cm of
        its spawn point, beside the face). Every driven case starts with the body's fatigue at full
        capacity, so its motors' strength does not depend on which driven case ran before it
        (BridgeEnv under body_function 'modulate' scales every motor's gain by its body set's capacity).
        All of these are restored after the case."""
        from mimo_bridge.observer_record import WorldRecorder
        out = os.path.join(self.scratch, name)
        if os.path.isdir(out):
            shutil.rmtree(out)
        rec = WorldRecorder(os.path.join(out, "observer", "world"), body=0, run="validate_" + name,
                            chunk_steps=1000, budget_bytes=2 ** 30, reserve_bytes=2 ** 30,
                            extra=dict(validation_case=name, motors="driven with real dynamics"))
        self.drive(case, rec)
        if rec.error:
            raise RuntimeError(rec.error)
        return out

    def drive(self, case: "Driven", rec=None, watch: Optional[Callable] = None) -> None:
        """Run one driven case with real dynamics (record_driven's physics, above): the world's movers held
        out, full strength, the case's start pose, then every frame its controller's commands and a world
        step; everything restored after. rec: a WorldRecorder (marked at the reset, fed every frame, closed
        at the end), or None for a trial run; watch(step, rep, world): called after every frame (a builder
        searching for a controller that does what its case needs, case_roll_driven)."""
        env, m, d = self.env, self.m, self.d
        movers = {k: getattr(env, k, None) for k in ("saccades", "reflex", "_cr", "_target_home", "_target_vel")}
        fatigue = getattr(getattr(env, "proprio", None), "fatigue", None)
        capacity = None if fatigue is None else np.array(fatigue.cap)
        gains = (None if getattr(env, "_native_actuator_gain", None) is None else
                 (np.array(m.actuator_gainprm), np.array(m.actuator_forcerange)))
        try:
            env.saccades = None
            env.reflex = None
            env._cr = None
            if fatigue is not None:
                fatigue.cap[:] = 1.0
            # the clock starts over, and with it the world's virtual skin (BridgeEnv.restart_clock): setting
            # data.time alone left the skin's clock at reset()'s collect, and the first physics step stopped on
            # "virtual skin time reversed" in every world built like H84 (mechanical_v4; 2026-09-26)
            env.restart_clock(0.0)
            env.frame = 0
            self.place_far()
            if case.scene is not None:
                case.scene(0.0, self)
            toy = getattr(env, "_target_id", None)
            if toy is not None:
                env._target_home = np.array(d.mocap_pos[toy])
                env._target_vel = np.zeros(3)
            d.qpos[:] = case.start(self)
            d.qvel[:] = 0.0
            d.qacc_warmstart[:] = 0.0
            d.ctrl[:] = 0.0
            self.mj.mj_forward(m, d)
            if rec is not None:
                rec.mark("reset")
            for s in range(case.steps):
                for rep in range(self.reps):
                    t = s + rep / float(self.reps)                 # the frame begins at t
                    if case.scene is not None:
                        case.scene(t, self)
                    raw = self.frame_step(case.command(t, self))
                    if rec is not None:
                        rec.on_frame(env, raw, dict(step=s, rep=rep, reps=self.reps, author="scripted",
                                                    latched=(rep == 0), order_seq=s, bout=False, rest=False))
                    if watch is not None:
                        watch(s, rep, self)
            if rec is not None:
                rec.close()
        finally:
            for k, v in movers.items():
                setattr(env, k, v)
            d.xfrc_applied[:] = 0.0
            if capacity is not None:
                fatigue.cap[:] = capacity
            if gains is not None:
                m.actuator_gainprm[:], m.actuator_forcerange[:] = gains
            d.ctrl[:] = 0.0
            if getattr(case.scene, "after", None) is not None:
                case.scene.after()          # the scene's forces and model changes (_roll_scene, _small_box_scene)


class Driven:
    """A case the body's own motors drive with real dynamics (ScriptWorld.record_driven):
    start(world) gives the first pose, command(t, world) every motor's command for the frame that
    begins at t (in model steps), read from the body's state as a controller would; steps model
    steps; scene(t, world) may move what the world moves."""

    def __init__(self, start, command, steps: int, scene=None):
        self.start, self.command, self.steps, self.scene = start, command, int(steps), scene


class _ModelEnv:
    """What WorldRecorder and ScriptWorld read from a world, for a compiled model alone: the
    model and data, the step length, the infant's bodies, handle and skin-patch names (copied
    from the kept recording's header), no caregiver and no world eye movers."""

    def __init__(self, m, d, header: dict):
        from types import SimpleNamespace
        self.model, self.data = m, d
        self.dt = float(header.get("env_dt_s", 0.01))
        self.frame = 0
        self._own_body_ids = sorted({int(b) for e in header["entities"] if e["kind"] == "self"
                                     for b in (e.get("bodies") or [])})
        self.proprio = SimpleNamespace(handle_names=list(header.get("handle_names", [])),
                                       body_ids=np.asarray(header.get("handle_bodies", []), dtype=np.int64))
        self._patches = list(header.get("patch_names", []))
        self.pooler = SimpleNamespace(specs=[SimpleNamespace(body_id=int(b)) for b in header.get("patch_bodies", [])])
        self.scene_kind = header.get("scene", "")
        self.fovea_fov = float(header.get("fovea_fovy_deg", 16.0))
        self._root_body_id = int(header.get("root_body", 1))
        self.caregiver = None
        self.saccades = None
        self.reflex = None

    def patch_names(self):
        return list(self._patches)

    def restart_clock(self, time: float = 0.0) -> None:
        """BridgeEnv.restart_clock's physics side: this dry-run world keeps no touch state of its own."""
        self.data.time = float(time)

    def step(self, motor=None, gaze=None, handle_mask=None, ctrl=None, render=None):
        """One frame of MuJoCo under these motor commands (BridgeEnv.step's physics, without its
        decoder, fatigue, toy walk and critters: a dry run)."""
        import mujoco
        self.data.ctrl[:] = 0.0 if ctrl is None else np.asarray(ctrl, dtype=np.float64)
        for _ in range(max(1, int(round(self.dt / float(self.model.opt.timestep))))):
            mujoco.mj_step(self.model, self.data)
        self.frame += 1
        return None


class ModelWorld(ScriptWorld):
    """The validation world without BridgeEnv, for a DRY RUN of the cases beside training.

    The compiled world model the recorder saved with a kept case recording (the production
    world's own model: same bodies, geoms, contact settings and cameras) and that recording's
    first record as the reset pose (a case whose first second moves nothing, so the record is the
    pose ScriptWorld starts every case from, stored to 1e-6 m and 1e-6 rad). MuJoCo's forward
    dynamics then computes contacts and forces exactly as in ScriptWorld. It needs the model's
    physics arena (about 0.4 GB) instead of a whole BridgeEnv (2.4-3 GB), so it can run while
    training runs, under tools/commit_guard. The authoritative validation stays `validate --like`
    (run()), which builds the world from the run's own settings; a dry run's recordings are never
    the golden ones."""

    def __init__(self, base_recording: str, scratch: str):
        import glob
        import mujoco
        from mimo_bridge.observer_record import load_world_model, read_chunk
        self.mj = mujoco
        self.scratch = scratch
        world = os.path.join(base_recording, "observer", "world")
        chunks = sorted(glob.glob(os.path.join(world, "b0", "chunk_*.npz")))
        if not chunks:
            raise FileNotFoundError("no recorded chunk under %s" % world)
        z = read_chunk(chunks[0])
        h = z["header"]
        self.m = load_world_model(os.path.join(world, h["model_file"]))
        self.d = mujoco.MjData(self.m)
        self.env = _ModelEnv(self.m, self.d, h)
        self.reps = int(z["frames"][0]) if len(z["frames"]) else STEP_REPS
        self.hold = self.reps
        self.d.qpos[:] = z["qpos"][0]
        if self.m.nmocap:
            self.d.mocap_pos[:] = z["mocap"][0][:, :3]
            self.d.mocap_quat[:] = z["mocap"][0][:, 3:7]
        mujoco.mj_forward(self.m, self.d)
        self._init_common()


DEFAULT_BASE_RECORDING = "h07_wrist_flexion"     # nothing moves before step 4: its first record is the reset pose


# ====================================================================== the cases
def _ramp(t, t0, t1, a, b):
    if t <= t0:
        return a
    if t >= t1:
        return b
    return a + (b - a) * (t - t0) / (t1 - t0)


def case_h05_index(w, others_deg=0.0):
    def pose(t):
        q = w.base()
        w.set_curl(q, "R", "index", _ramp(t, 4, 8, 0, 40))
        for dg in ("middle", "ring", "little", "thumb"):
            w.set_curl(q, "R", dg, _ramp(t, 4, 8, 0, others_deg))
        return q
    return pose, 16


def case_h05_whole(w):
    def pose(t):
        q = w.base()
        for dg in FINGERS:
            w.set_curl(q, "R", dg, _ramp(t, 4, 8, 0, 40))
        return q
    return pose, 16


def case_h04_cycles(w, cycles=2):
    def pose(t):
        q = w.base()
        ph = max(0.0, t - 4) / 6.0            # 6 steps (300 ms) per half cycle
        k = int(ph)
        frac = ph - k
        closed = (k % 2 == 0)
        v = 45 * (frac if closed else 1 - frac) if k < 2 * cycles else 0.0
        for dg in FINGERS:
            w.set_curl(q, "R", dg, v)
        return q
    return pose, 4 + 12 * cycles + 8


def _box_on_palm(w, q, side, gap=-0.001):
    """qpos placing box_light (a 6 cm cube) face-on against the inside of the palm (gap < 0:
    pressed in). The palm normal is where the index fingertip goes when it curls from straight
    (kinematics.py); the cube turns so one face lies flat on the palm. (Before 2026-09-24 the
    normal came from UN-curling a curled finger and the cube kept the world's orientation, so a
    held cube ran through the hand: every finger segment touched it at hundreds of newtons, and
    no release could end the contact.)"""
    m = w.m
    hb = m.body(("left_" if side == "L" else "right_") + "hand").id
    b_i, loc_i = w.tip_local(side, "index")
    qa, qb = np.array(q), np.array(q)
    w.set_curl(qa, side, "index", 0)
    w.set_curl(qb, side, "index", 30)
    t0 = w.point(qa, b_i, loc_i)
    t1 = w.point(qb, b_i, loc_i)
    w.fk(q)
    n = (t1 - t0) / max(1e-9, np.linalg.norm(t1 - t0))
    palm = np.asarray(w.d.xpos[hb])
    g = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == m.body("box_light").id][0]
    half = float(m.geom_size[g][2])
    palm_half = 0.0062
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, n)
    sn, cs = float(np.linalg.norm(v)), float(np.dot(z, n))
    if sn < 1e-9:
        quat = np.array([1.0, 0.0, 0.0, 0.0]) if cs > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    else:
        ang = math.atan2(sn, cs)
        quat = np.r_[math.cos(ang / 2), (v / sn) * math.sin(ang / 2)]
    adr = w.free_adr("box_light")
    q = np.array(q)
    q[adr:adr + 3] = palm + n * (palm_half + half + gap)
    q[adr + 3:adr + 7] = quat
    return q


def case_h01(w, close_at=6.0, touch=True):
    def pose(t):
        q = w.base()
        away = _box_on_palm(w, q, "R", gap=0.05)
        on = _box_on_palm(w, q, "R", gap=-0.001)
        q = on if (touch and t >= 4) else away
        q = np.array(q)
        for dg in FINGERS:
            w.set_curl(q, "R", dg, _ramp(t, close_at, close_at + 3, 0, 30))
        return q
    return pose, int(close_at) + 12


def case_wrist(w, kind="flexion", elbow=0.0):
    role = {"rotation": "right_hand1", "flexion": "right_hand2", "deviation": "right_hand3"}[kind]

    def pose(t):
        q = w.base()
        q[w.jq[role]] += math.radians(_ramp(t, 4, 10, 0, -35 if kind == "flexion" else 35))
        if elbow:
            q[w.jq["right_elbow"]] += math.radians(_ramp(t, 4, 10, 0, -elbow))
        return q
    return pose, 16


def case_pinch(w, pair=("thumb", "index")):
    b_a, loc_a = w.tip_local("R", pair[0])
    b_b, loc_b = w.tip_local("R", pair[1])
    key = {"index": "ff", "middle": "mf", "ring": "rf", "little": "lf", "thumb": "th"}
    joints_a = (["right_th_swivel", "right_th_adduction", "right_th_pivot", "right_th_middle", "right_th_distal"]
                if pair[0] == "thumb" else ["right_%s_%s" % (key[pair[0]], r) for r in ("side", "knuckle", "middle", "distal")])
    joints_b = ["right_%s_%s" % (key[pair[1]], r) for r in ("side", "knuckle", "middle", "distal")]
    q0 = w.base()
    w.set_curl(q0, "R", pair[1], 35)
    target = w.point(q0, b_b, loc_b)
    # physical (detector v9, fix 3): each digit's tip pulled toward the other's with contacts on, so the
    # tips meet and stop; posed by inverse kinematics the middle fingertip went into the index finger
    # (b02_neighbours_touch: 135.9 N recorded, over the body's weight)
    q_end = _pulled_path(w, q0, [(b_a, loc_a, target)], joints_a, steps=300, force=0.5)[-1]
    # the first digit's joints stop short of the target (1.3 cm on H84's thumb, measured
    # 2026-09-24): the second digit then bends to meet the first one's tip
    q_end = _pulled_path(w, q_end, [(b_b, loc_b, w.point(q_end, b_a, loc_a))], joints_b, steps=300, force=0.5)[-1]
    moved = joints_a + joints_b

    def pose(t):
        a = _ramp(t, 4, 12, 0.0, 1.0)
        q = w.base()
        w.set_curl(q, "R", pair[1], 35)
        for j in moved:
            q[w.jq[j]] = q0[w.jq[j]] + a * (q_end[w.jq[j]] - q0[w.jq[j]])
        return q
    return pose, 18


ARM_R = ["right_shoulder_horizontal", "right_shoulder_ad_ab", "right_shoulder_rotation", "right_elbow",
         "right_hand1", "right_hand2", "right_hand3"]
ARM_L = ["left_shoulder_horizontal", "left_shoulder_ad_ab", "left_shoulder_rotation", "left_elbow",
         "left_hand1", "left_hand2", "left_hand3"]


def case_to_head(w, zone="mouth"):
    b, loc = w.tip_local("R", "index")
    q0 = w.base()
    w.fk(q0)
    m = w.m
    hb = m.body("head").id
    R = w.d.xmat[hb].reshape(3, 3)
    p = np.asarray(w.d.xpos[hb])
    geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == hb]
    c_local = np.asarray(m.geom_pos[geoms[0]])
    r = float(m.geom_size[geoms[0]][0])
    le, re = m.body("left_eye").id, m.body("right_eye").id
    f = R.T @ (0.5 * (w.d.xpos[le] + w.d.xpos[re]) - p) - c_local
    f /= np.linalg.norm(f)
    lat = R.T @ (w.d.xpos[le] - w.d.xpos[re])
    lat /= np.linalg.norm(lat)
    up = np.cross(f, lat)
    if np.dot(up, R.T @ (w.d.xpos[m.body("upper_body").id] - p)) > 0:
        up = -up
    if zone == "mouth":
        a = math.radians(35)
        local = c_local + (r - 0.002) * (math.cos(a) * f - math.sin(a) * up)
    else:                                        # the side of the head, 90 degrees from the face
        local = c_local + (r - 0.002) * (-lat)
    target = p + R @ local
    if zone == "mouth":
        return _to_mouth_physically(w, q0, b, loc, target)
    q_end = w.ik(q0, b, loc, target, ARM_R, iters=400)

    def pose(t):
        a = _ramp(t, 4, 20, 0.0, 1.0)
        q = w.base()
        for j in ARM_R:
            q[w.jq[j]] = q0[w.jq[j]] + a * (q_end[w.jq[j]] - q0[w.jq[j]])
        return q
    return pose, 28, float(np.linalg.norm(w.point(q_end, b, loc) - target))


MOUTH_PULL_N = 4.0          # the pull that draws the fingertip to the mouth (gravity off while it does)
MOUTH_SETTLE_STEPS = 600    # 3 s of 5 ms steps: the arm is at rest by step 529 (measured 2026-09-25)


def _to_mouth_physically(w, q0, b, loc, target):
    """b04_fingertip_to_mouth's approach, physical (detector v9; OBSERVER_DESIGN.md section 9: the
    kept case recorded a contact force above the recorder's 16-bit limit). Until v9 the arm's pose came
    from inverse kinematics that ignored the body: the approach drove the right upper arm up to 5.9
    cm into the chest, a recorded 74,639 N (the fast lane's float16 stores it as infinity). Now the
    index fingertip is drawn to the mouth point with MOUTH_PULL_N while MuJoCo's contacts keep the
    arm out of the trunk and the face (gravity off during this construction only), and the case
    replays that physical path at the time the physics took (from t 4), then holds the touch 8 steps:
    the fingertip ends 3.3 mm from a point 2 mm inside the head's surface, touching the face. (Replayed
    3.3 times faster, the arm brushing the chest read 169.6 N: a posed limb moving into a surface faster
    than contact physics allows gets a large constraint force.) Returns (pose, steps, the fingertip's
    end distance to the target in metres)."""
    path = []
    _settle(w, q0, [(b, loc, target)], ARM_R, steps=MOUTH_SETTLE_STEPS, force=MOUTH_PULL_N, path=path,
            gravity=False)
    path = [np.array(q0)] + path
    tips = np.array([w.point(q, b, loc) for q in path])
    moving = np.nonzero(np.linalg.norm(np.diff(tips, axis=0), axis=1) > 1e-5)[0]
    last = int(moving[-1]) + 1 if moving.size else 0
    per_step = float(w.env.dt) * w.reps / float(w.m.opt.timestep)     # physics steps per model step
    adrs = [w.jq[j] for j in ARM_R]
    t_touch = 4.0 + last / per_step

    def pose(t):
        x = min(float(last), max(0.0, (t - 4.0) * per_step))
        k = min(int(x), len(path) - 2)
        fr = x - k
        q = w.base()
        q[adrs] = (1.0 - fr) * path[k][adrs] + fr * path[k + 1][adrs]
        return q
    return pose, int(math.ceil(t_touch)) + 8, float(np.linalg.norm(tips[last] - target))


def _aim_eyes(w, q, target):
    """Eye joint angles (both eyes) that point each eye camera's axis at a world point."""
    m, d = w.m, w.d
    q = np.array(q)
    for side, cam in (("left", "eye_left"), ("right", "eye_right")):
        cid = m.camera(cam).id
        hj, vj = w.jq[side + "_eye_horizontal"], w.jq[side + "_eye_vertical"]
        for _ in range(40):
            d.qpos[:] = q
            w.mj.mj_kinematics(m, d)
            w.mj.mj_camlight(m, d)
            fwd = -d.cam_xmat[cid].reshape(3, 3)[:, 2]
            want = np.asarray(target) - d.cam_xpos[cid]
            want /= np.linalg.norm(want)
            err = want - fwd
            if np.linalg.norm(err) < 1e-4:
                break
            jac = np.zeros((3, 2))
            for k, j in enumerate((hj, vj)):
                q2 = np.array(q)
                q2[j] += 1e-4
                d.qpos[:] = q2
                w.mj.mj_kinematics(m, d)
                w.mj.mj_camlight(m, d)
                jac[:, k] = (-d.cam_xmat[cid].reshape(3, 3)[:, 2] - fwd) / 1e-4
            dq = np.linalg.lstsq(jac, err, rcond=None)[0]
            q[hj] += float(dq[0])
            q[vj] += float(dq[1])
    return q


def case_look(w, sweep=False):
    """The eyes turn to a box held 30 cm in front of the face and stay on it (a fixation), or
    sweep around it without settling. (Before 2026-09-24 the eyes aimed at the toy's resting
    place, beyond their range: every ray met the ceiling.)"""
    m = w.m
    adr = w.free_adr("box_light")
    q0 = w.base()
    w.fk(q0)
    cams = [m.camera(c).id for c in ("eye_left", "eye_right")]
    w.mj.mj_camlight(m, w.d)
    mid = np.mean([w.d.cam_xpos[c] for c in cams], axis=0)
    fwd = np.mean([-w.d.cam_xmat[c].reshape(3, 3)[:, 2] for c in cams], axis=0)
    fwd /= np.linalg.norm(fwd)
    # off the straight-ahead line by 10 cm, so reaching it takes an eye movement
    side = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-6 else np.array([1.0, 0.0, 0.0])
    spot = mid + 0.30 * fwd + 0.10 * side

    def eyes(t, q):
        tgt = spot
        if sweep:
            ang = t * 0.6
            tgt = spot + 0.15 * (math.cos(ang) * side + math.sin(ang) * np.cross(fwd, side))
        return _aim_eyes(w, q, tgt) if t >= 3 else q

    def pose(t):
        q = w.base()
        q[adr:adr + 3] = spot
        q[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
        return q
    return pose, 14, eyes


def _pulled_path(w, q0, pulls, joints: List[str], steps: int = 400, force: float = 2.0) -> List[np.ndarray]:
    """The path (qpos per 5 ms physics step, the start first) of these joints while each (body, local point,
    world target) is pulled toward its target with `force` newtons and MuJoCo's contacts stop the limb on the
    body, the floor or the other digits (validate._settle, gravity off during the construction only): a
    posed movement that cannot go through anything (detector v9, fix 3)."""
    path = []
    _settle(w, q0, pulls, joints, steps=steps, force=force, path=path, gravity=False)
    return [np.array(q0)] + path


def _replay(w, q_base, path: List[np.ndarray], joints: List[str], t0: float, t1: float):
    """pose(t): the joints along a pulled path from t0 to t1 (linear in the path's steps), q_base else."""
    adrs = [w.jq[j] for j in joints]

    def at(t) -> np.ndarray:
        x = _ramp(t, t0, t1, 0.0, 1.0) * (len(path) - 1)
        k = min(int(x), len(path) - 2)
        fr = x - k
        q = np.array(q_base)
        q[adrs] = (1.0 - fr) * path[k][adrs] + fr * path[k + 1][adrs]
        return q
    return at


def case_reach(w, away=False):
    """R01 (reach) and its twin: the palm pulled 8 cm toward a box 11 cm away (or away from it) on a
    physical path (_pulled_path, detector v9: the posed reach by inverse kinematics drove the upper arm
    into the chest, 1,435 N recorded), replayed over t 4-20."""
    m = w.m
    adr = w.free_adr("box_light")
    q0 = w.base()
    w.fk(q0)
    hb = m.body("right_hand").id
    palm0 = np.asarray(w.d.xpos[hb]).copy()
    box = palm0 + np.array([0.0, -0.10, 0.05]) if not away else palm0
    direction = (box - palm0) / max(1e-9, np.linalg.norm(box - palm0)) if not away else np.array([0.0, 0.1, 0.05])
    target = palm0 + (0.08 * direction if not away else np.array([0.0, 0.08, 0.04]))
    bq = box if not away else palm0 + np.array([0.0, -0.12, 0.05])
    q_box = np.array(q0)
    q_box[adr:adr + 3] = bq
    q_box[adr + 3:adr + 7] = (1, 0, 0, 0)
    # the box stands where the case puts it while the arm is pulled: the hand stops on it
    reach = _replay(w, q0, _pulled_path(w, q_box, [(hb, np.zeros(3), target)], ARM_R[:4]), ARM_R[:4], 4.0, 20.0)

    def pose(t):
        q = reach(t)
        q[adr:adr + 3] = bq
        q[adr + 3:adr + 7] = (1, 0, 0, 0)
        return q
    return pose, 26


FLOOR_REST_M = 0.0005       # a posed body rests on the floor this deep (about 10 N per mm of contact)


def _rest_on_floor(w, q, root_adr: int, tol: float = FLOOR_REST_M) -> np.ndarray:
    """q with the infant's root raised or lowered until its deepest contact with the floor is between 0 and
    2 x tol: the posed body lies on the floor without sinking into it or floating above it."""
    own = {w.m.body(int(b)).name for b in w.env._own_body_ids}
    q = np.array(q)
    for _ in range(60):
        pens = [pen for a, b, pen in w.contacts_now(q) if "world" in (a, b) and (a in own or b in own)]
        deep = max(pens) if pens else None
        if deep is None:
            q[root_adr + 2] -= 0.002                        # floating: come down until it touches
        elif deep > 2.0 * tol:
            q[root_adr + 2] -= tol - deep                   # sunk: rise until it rests
        else:
            return q
    raise RuntimeError("the posed body does not come to rest on the floor")


ROLL_TORQUE_GAINS = (60.0, 6.0, 30.0)     # N m per radian, N m per radian per second, the cap in N m


def _roll_scene(w, torque=None):
    """The roll cases' scene: the toy far away and the mobile's collisions off (its hanging elements reach
    down to a rolling body's up side: a touch by a mounted object that then moves, which P02 reads as an
    outside push), restored after; torque(t, world), when given, applies the outside torque each frame."""
    m = w.m
    toy = _toy_far(w)
    mobile = [g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name.startswith("mobile")]
    mobile_col = (np.array(m.geom_contype[mobile]), np.array(m.geom_conaffinity[mobile]))

    def scene(t, world):
        toy(t, world)
        world.m.geom_contype[mobile] = 0
        world.m.geom_conaffinity[mobile] = 0
        world.d.xfrc_applied[:] = 0.0
        if torque is not None:
            torque(t, world)

    def after():
        w.d.xfrc_applied[:] = 0.0
        w.m.geom_contype[mobile], w.m.geom_conaffinity[mobile] = mobile_col
    scene.after = after
    return scene


def case_roll_turned(w, degrees=80.0):
    """P02's motor twin (detector v9, the independent review of 2026-09-25): a body that does not roll itself,
    turned onto its side from outside. The whole body free under gravity and contacts, EVERY motor command zero
    (the limbs limp), turned about the trunk's long axis by a torque on the trunk (split between the upper body
    and the hip) that follows a target roll rising from 0 to `degrees` over t 4-24 (ROLL_TORQUE_GAINS: a
    proportional-derivative law on the upper body's measured roll, capped). The arms are first folded onto
    the chest (each hand pulled toward the chest's centre, contacts on, so it stops on the chest). This was
    p02_roll_to_side's positive under fix 3 (its arms and head then held by their motors): measured in the
    saved world model (2026-09-25 9:20 PM, explore_roll2.py), the target 95 degrees gave 91 degrees, the
    target 45 gave 40, the largest body contact 229 N (the arm pinned under the turning trunk). The roll
    crosses the side band, but no motor of the body drove it: P02 absent (its motor rule, detectors._body_drove)."""
    m = w.m
    ub, hb = m.body("upper_body").id, m.body("hip").id
    kp, kd, cap = ROLL_TORQUE_GAINS
    frame = {}

    def roll_now(world):
        lat = world.d.xmat[ub].reshape(3, 3)[:, 0]
        return math.atan2(float(np.dot(np.cross(frame["lat0"], lat), frame["axis"])), float(np.dot(frame["lat0"], lat)))

    def start(world):
        q = _all_boxes_far(world, world.base())
        world.fk(q)
        above = np.array(world.d.xpos[m.body("chest").id])            # the pull stops on the chest
        q = _pulled_path(world, q, [(m.body("left_hand").id, np.zeros(3), above),
                                    (m.body("right_hand").id, np.zeros(3), above)],
                         ARM_L[:4] + ARM_R[:4], steps=500, force=2.0)[-1]
        world.fk(q)
        frame["axis"] = np.array(world.d.xmat[ub].reshape(3, 3)[:, 2])
        frame["lat0"] = np.array(world.d.xmat[ub].reshape(3, 3)[:, 0])
        return q

    def torque(t, world):
        d = world.d
        if "axis" not in frame or t <= 0.0:
            return
        target = math.radians(_ramp(t, 4, 24, 0.0, degrees))
        om = float(np.dot(d.cvel[ub][:3], frame["axis"]))
        tau = float(np.clip(kp * (target - roll_now(world)) - kd * om, -cap, cap))
        d.xfrc_applied[ub, 3:6] = 0.5 * tau * frame["axis"]
        d.xfrc_applied[hb, 3:6] = 0.5 * tau * frame["axis"]

    def command(t, world):
        return np.zeros(world.m.nu)                          # limp: no motor drives anything
    return Driven(start, command, ROLL_STEPS, _roll_scene(w, torque))


# THE ROLL THE BODY DRIVES (detector v9; the independent review of 2026-09-25: "P02's positive is a limp body
# rolled by an applied torque", so the case froze into the lock that a roll the body did not drive is P02).
# p02_roll_to_side and p02_half_roll are driven by the body's own motors, with real dynamics and no force from
# outside (ScriptWorld.record_driven): every body motor but the fingers' and the eyes' holds its joint where the
# case starts (ROLL_HOLD_GAINS), and from ROLL_GO the motors of a ROLL SYNERGY take full command, ramped over
# 0.2 s, each in the direction that forward kinematics at the start pose says turns the body toward the chosen
# side: the trunk's twist joints (hip_rot1, chest_rot: the sign that raises the up-side shoulder), the head's
# swivel (the face toward that side), both knees dropped toward it (each hip2: the sign that moves that knee
# toward the side; the reset pose lies with the hips flexed 50 degrees and the knees 73), the up-side arm
# reaching across (its shoulder_horizontal: the hand toward the side), and in some trials the down-side arm
# cleared out to the side (its shoulder_ad_ab) and the up-side leg pushing on its planted foot (its hip1: the
# knee down). Which parts on which side roll this body onto its side is physics nobody has measured yet (this
# case was written in a detector pass without world runs; its first world run is a pause job), so the builder
# SEARCHES in the world's own physics, in ROLL_SEARCH order: trial runs of the same driven case, the first
# whose upper body turns |roll| >= ROLL_SIDE_DEG at 3 record ends in a row (the side band starts at 60) is
# p02_roll_to_side; for p02_half_roll the same synergy at a reduced command scale or cut short
# (ROLL_HALF_TRIALS), the first whose record ends turn it 35-52 degrees and never past 55 (short of the side
# band). Every trial is printed. When no trial meets its line the closest is recorded and the case's
# expectation fails in the report: a failure that says what the synergy reached, never a crash.
ROLL_STEPS = 32
ROLL_HOLD_GAINS = (4.0, 0.4)            # command per radian of error, per radian per second
ROLL_GO = (4.0, 8.0)                    # the synergy's command ramps 0 -> full over t 4-8 (0.2 s)
ROLL_SIDE_DEG = 70.0
ROLL_HALF_DEG = (35.0, 52.0)
ROLL_HALF_CEILING_DEG = 55.0
ROLL_PARTS = (("twist", "head", "knees", "reach"),
              ("twist", "head", "knees", "reach", "clear"),
              ("twist", "head", "knees", "reach", "push"),
              ("twist", "head", "knees", "reach", "clear", "push"))
ROLL_SEARCH = tuple((side, parts) for side in ("L", "R") for parts in ROLL_PARTS)
ROLL_HALF_TRIALS = tuple([(scale, None) for scale in (0.5, 0.35, 0.25, 0.65, 0.8)]
                         + [(1.0, stop) for stop in (8.0, 10.0, 12.0, 16.0)])
_ROLL_SKIP = ("_ff_", "_mf_", "_rf_", "_lf_", "_th_", "eye")


def _body_motors(m) -> Dict[str, int]:
    """joint name -> motor, for every motor with a joint transmission (the fingers' and eyes' included)."""
    out = {}
    for u in range(int(m.nu)):
        if int(m.actuator_trntype[u]) == 0:
            out[m.joint(int(m.actuator_trnid[u][0])).name.split(":")[-1]] = u
    return out


def roll_command(t: float, qpos, qvel, plan: dict) -> np.ndarray:
    """One frame's motor commands of a driven roll (THE ROLL THE BODY DRIVES): every held joint pulled toward
    its start angle by ROLL_HOLD_GAINS, then each synergy motor at scale x its sign, ramped over ROLL_GO,
    until `stop` (after it the synergy's joints are held again). plan: nu, hold [(qpos address, dof address,
    motor, start angle)], synergy {motor: +1 or -1}, scale, stop (t or None). A function of the body's state
    alone: the recorded commands are the controller's."""
    kp, kd = ROLL_HOLD_GAINS
    ctrl = np.zeros(int(plan["nu"]))
    for a, v, u, q0 in plan["hold"]:
        ctrl[u] = float(np.clip(kp * (q0 - float(qpos[a])) - kd * float(qvel[v]), -1.0, 1.0))
    stop = plan.get("stop")
    level = float(plan.get("scale", 1.0)) * _ramp(t, ROLL_GO[0], ROLL_GO[1], 0.0, 1.0)
    if level and (stop is None or t < stop):
        for u, sign in plan["synergy"].items():
            ctrl[u] = float(np.clip(level * sign, -1.0, 1.0))
    return ctrl


def _roll_synergy(w, q, side: str, parts) -> Dict[str, float]:
    """joint -> the sign of its synergy command (THE ROLL THE BODY DRIVES), found by forward kinematics at q:
    each joint turned 0.05 rad, and what it moved read."""
    m = w.m
    w.fk(q)
    lat = np.array(w.d.xmat[m.body("upper_body").id].reshape(3, 3)[:, 1])   # toward the body's left
    toward = lat if side == "L" else -lat
    up, down = ("right", "left") if side == "L" else ("left", "right")

    def pos(body):
        return lambda: np.array(w.d.xpos[m.body(body).id])

    def sign(joint, read, along):
        qa = np.array(q)
        qa[w.jq[joint]] += 0.05
        w.fk(q)
        a = float(np.dot(read(), along))
        w.fk(qa)
        b = float(np.dot(read(), along))
        return 1.0 if b > a else -1.0
    z = np.array([0.0, 0.0, 1.0])
    out = {}
    if "twist" in parts:
        for j in ("hip_rot1", "chest_rot"):
            if j in w.jq:
                out[j] = sign(j, pos(up + "_upper_arm"), z)                  # the up-side shoulder rises
    if "head" in parts and "head_swivel" in w.jq:
        qa = np.array(q)
        qa[w.jq["head_swivel"]] += 0.05
        f0, f1 = _eye_frame(w, q)[1], _eye_frame(w, qa)[1]
        out["head_swivel"] = 1.0 if float(np.dot(f1 - f0, toward)) > 0 else -1.0
    if "knees" in parts:
        for s_ in (up, down):
            out[s_ + "_hip2"] = sign(s_ + "_hip2", pos(s_ + "_lower_leg"), toward)
    if "reach" in parts:
        out[up + "_shoulder_horizontal"] = sign(up + "_shoulder_horizontal", pos(up + "_hand"), toward)
    if "clear" in parts:
        out[down + "_shoulder_ad_ab"] = sign(down + "_shoulder_ad_ab", pos(down + "_hand"), toward)
    if "push" in parts:
        out[up + "_hip1"] = -sign(up + "_hip1", pos(up + "_lower_leg"), z)     # the knee down: the foot pushes
    w.fk(q)
    return out


def _roll_driven_case(w, side: str, parts, scale: float = 1.0, stop: Optional[float] = None) -> "Driven":
    """A roll driven by the body's own motors (THE ROLL THE BODY DRIVES) toward `side`, with these synergy parts."""
    act = _body_motors(w.m)
    plan = dict(nu=int(w.m.nu), scale=float(scale), stop=stop, synergy={}, hold=[])

    def start(world):
        q = _all_boxes_far(world, world.base())
        syn = _roll_synergy(world, q, side, parts)
        plan["synergy"] = {act[j]: sgn for j, sgn in syn.items() if j in act}
        plan["hold"] = [(world.jq[j], world.jd[j], u, float(q[world.jq[j]])) for j, u in sorted(act.items())
                        if not any(k in j for k in _ROLL_SKIP)]
        plan["joints"] = syn
        return q

    def command(t, world):
        return roll_command(t, world.d.qpos, world.d.qvel, plan)
    case = Driven(start, command, ROLL_STEPS, _roll_scene(w))
    case.plan = plan
    return case


def _roll_deg_now(world) -> float:
    """The upper body's roll as kinematics.py reads it (0 on the back, +-90 on a side)."""
    R = world.d.xmat[world.m.body("upper_body").id].reshape(3, 3)
    return math.degrees(math.atan2(float(R[2, 1]), float(R[2, 0])))


def roll_reaches_side(rolls) -> bool:
    """A trial's record-end rolls turn the body onto its side: starting on the back (|roll| <= 30), then
    |roll| of at least ROLL_SIDE_DEG and at most 120 at 3 record ends in a row."""
    r = np.abs(np.asarray(rolls, dtype=np.float64))
    if not r.size or r[0] > 30.0:
        return False
    ok = (r >= ROLL_SIDE_DEG) & (r <= 120.0)
    return bool(any(ok[i:i + 3].all() for i in range(max(0, len(ok) - 2))))


def roll_stays_half(rolls) -> bool:
    """A trial's record-end rolls turn the body ROLL_HALF_DEG and never past ROLL_HALF_CEILING_DEG."""
    r = np.abs(np.asarray(rolls, dtype=np.float64))
    return bool(r.size and ROLL_HALF_DEG[0] <= r.max() <= ROLL_HALF_DEG[1] and r.max() < ROLL_HALF_CEILING_DEG)


def _roll_trial(w, case) -> List[float]:
    """The upper body's roll at every record end of one trial run of a driven case (no recording)."""
    rolls = []

    def watch(s, rep, world):
        if rep == world.reps - 1:
            rolls.append(_roll_deg_now(world))
    w.drive(case, None, watch)
    return rolls


def _roll_search(w) -> dict:
    """The driven rolls this world's physics makes (THE ROLL THE BODY DRIVES), searched once per world."""
    found = getattr(w, "_roll_search_cache", None)
    if found is not None:
        return found
    trials, side_pick = [], None
    for side, parts in ROLL_SEARCH:
        rolls = _roll_trial(w, _roll_driven_case(w, side, parts))
        trials.append(dict(side=side, parts=list(parts), scale=1.0, stop=None,
                           max_roll_deg=round(float(np.max(np.abs(rolls))), 1), reaches_side=roll_reaches_side(rolls)))
        print("roll search: %s %s -> largest roll %.1f degrees%s" % (side, "+".join(parts), trials[-1]["max_roll_deg"],
              " (onto the side)" if trials[-1]["reaches_side"] else ""), flush=True)
        if trials[-1]["reaches_side"]:
            side_pick = trials[-1]
            break
    if side_pick is None:
        side_pick = max(trials, key=lambda tr_: tr_["max_roll_deg"])
    half_pick, half_trials = None, []
    for scale, stop in ROLL_HALF_TRIALS:
        rolls = _roll_trial(w, _roll_driven_case(w, side_pick["side"], side_pick["parts"], scale, stop))
        half_trials.append(dict(side=side_pick["side"], parts=side_pick["parts"], scale=scale, stop=stop,
                                max_roll_deg=round(float(np.max(np.abs(rolls))), 1), stays_half=roll_stays_half(rolls)))
        print("roll search, half: scale %.2f stop %s -> largest roll %.1f degrees%s" % (
            scale, stop, half_trials[-1]["max_roll_deg"], " (a half roll)" if half_trials[-1]["stays_half"] else ""),
            flush=True)
        if half_trials[-1]["stays_half"]:
            half_pick = half_trials[-1]
            break
    if half_pick is None:
        under = [t_ for t_ in half_trials if t_["max_roll_deg"] < ROLL_HALF_CEILING_DEG]
        half_pick = (max(under, key=lambda t_: t_["max_roll_deg"]) if under
                     else min(half_trials, key=lambda t_: t_["max_roll_deg"]))
    found = dict(side=side_pick, half=half_pick, trials=trials, half_trials=half_trials)
    w._roll_search_cache = found
    return found


def case_roll_driven(w, kind: str) -> "Driven":
    """P02, rolling to the side, driven by the body's own motors (THE ROLL THE BODY DRIVES). kind 'side': the
    searched synergy that turns the body onto its side (p02_roll_to_side, P02 present); 'half': the same
    synergy weaker or cut short, turning it 35-52 degrees (p02_half_roll, P02 absent: short of the side band
    that starts at 60 degrees)."""
    if kind not in ("side", "half"):
        raise ValueError(kind)
    pick = _roll_search(w)[kind]
    return _roll_driven_case(w, pick["side"], pick["parts"], pick["scale"], pick["stop"])


# ====================================================================== the grasp family
# Addendum section 27: "for composite behaviors, break one link at a time: a grasp detector must
# fail on contact without opposition, opposition without lift, lift without retention and
# accidental object motion". The links: CONTACT (the hand touches the box) -> OPPOSITION (thumb
# against fingers, R04) -> LIFT off the floor in the hand's contact (R05) -> RETENTION (the box
# stays with the palm, a brief hold, H02) -> CARRY (R06) -> RELEASE (the hand opens, H03; onto
# support, R08). Each case below breaks exactly one link and names what must and must not be found.
#
# THE v8 GRASP (measured 2026-09-24; its construction replaced in v9 by THE GRASP REBUILT ON A SMALLER CUBE,
# below). The world's only free objects are
# three 6 cm cubes, as long as this hand's palm and fingers together. A box set on the palm face with
# the thumb on a side face sits at 90 degrees (no opposition), and the thumb cannot reach the face
# nearest the wrist. The grasp is found by physics, not by hand: the box is held still in the hand's
# frame (turned 25 degrees about the palm normal, 1.8 cm radial and 2.7 cm distal of the palm's
# centre, its top face 1.0 mm below the palm's face), a 1 N pull on each of the thumb, index,
# middle and ring fingertips closes them until MuJoCo's contacts stop them (_settle), and each
# pulled tip then moves 2.5 mm further into its face (the squeeze), so the contacts have a clear
# depth. No segment then sits deeper than 2.8 mm, the thumb presses 1.9 mm or more, and R04's
# opposition angle at the box's centre reads 119-128 degrees under 0.2 mm and 1-degree changes of
# the layout (median 128), against its 90-degree line; in the recorded dry runs, with the forces the
# recorder stores, R04 measured 107-132 degrees. That margin had to be searched for: 144 layouts were tried, and most
# natural ones read 83-126 degrees under the same small changes, a median near 90 - on these boxes
# R04's line cuts through the grasps this hand can make, which fits the four babble baselines making
# no opposing grasp in 80,000 body-steps (babble_rates.json). On the floor the palm faces down and
# the box stands upright.
GRASP_BOX = "box_light"
BOX_HALF = 0.03
FAR_BOXES = (("box_medium", (2.0, 2.0)), ("box_heavy", (2.2, 2.0)))    # the scene cleared of the other boxes
TOY = "target"
TOY_FAR = np.array([2.5, 3.0, 3.0])
THUMB_R = ["right_th_swivel", "right_th_adduction", "right_th_pivot", "right_th_middle", "right_th_distal"]
FINGER_R = {dg: ["right_%s_%s" % (k, r) for r in ("side", "knuckle", "middle", "distal")]
            for dg, k in (("index", "ff"), ("middle", "mf"), ("ring", "rf"), ("little", "lf"))}
DIGITS_R = THUMB_R + FINGER_R["index"] + FINGER_R["middle"] + FINGER_R["ring"] + FINGER_R["little"]


def _mat2quat(w, R) -> np.ndarray:
    qq = np.zeros(4)
    w.mj.mju_mat2Quat(qq, np.asarray(R, dtype=np.float64).reshape(9))
    return qq


def _quat2mat(w, quat) -> np.ndarray:
    R = np.zeros(9)
    w.mj.mju_quat2Mat(R, np.asarray(quat, dtype=np.float64))
    return R.reshape(3, 3)


def _settle(w, q, pulls, joints: List[str], steps: int = 150, force: float = 1.5,
            path: Optional[list] = None, gravity: bool = True) -> np.ndarray:
    """Close digits on held-still objects with physics: every joint not in `joints` (the body, the
    boxes) is put back each 5 ms step, and a Cartesian pull of `force` newtons on each (body, local
    point, world target) draws it toward its target until MuJoCo's contacts stop it. Deterministic:
    velocities, warm start, applied forces and controls start at zero and are restored after.
    path: a list that receives qpos after every step (a physical path for a case to replay);
    gravity False: gravity is off during the settle and restored after (a light pull then moves a
    whole arm; the recording itself always has gravity)."""
    m, d, mj = w.m, w.d, w.mj
    qa = [w.jq[j] for j in joints]
    va = [w.jd[j] for j in joints]
    keep = np.ones(m.nq, dtype=bool)
    keep[qa] = False
    vkeep = np.ones(m.nv, dtype=bool)
    vkeep[va] = False
    frozen = np.array(q)
    saved = (np.array(d.qfrc_applied), np.array(d.xfrc_applied), np.array(d.ctrl),
             np.array(d.act) if m.na else None, float(d.time), np.array(m.opt.gravity))
    d.qpos[:] = q
    d.qvel[:] = 0.0
    d.qacc_warmstart[:] = 0.0
    d.qfrc_applied[:] = 0.0
    d.xfrc_applied[:] = 0.0
    d.ctrl[:] = 0.0
    if m.na:
        d.act[:] = 0.0
    if not gravity:
        m.opt.gravity[:] = 0.0
    try:
        for _ in range(steps):
            mj.mj_forward(m, d)
            tau = np.zeros(m.nv)
            for body, local, target in pulls:
                p = np.asarray(d.xpos[body]) + d.xmat[body].reshape(3, 3) @ np.asarray(local)
                e = np.asarray(target) - p
                n = float(np.linalg.norm(e))
                if n < 1e-6:
                    continue
                jacp = np.zeros((3, m.nv))
                mj.mj_jac(m, d, jacp, None, p, body)
                tau += jacp.T @ (force * e / max(n, 0.005))
            d.qfrc_applied[:] = 0.0
            d.qfrc_applied[va] = tau[va]
            mj.mj_step(m, d)
            d.qpos[keep] = frozen[keep]
            d.qvel[vkeep] = 0.0
            if path is not None:
                path.append(np.array(d.qpos))
        return np.array(d.qpos)
    finally:
        d.qfrc_applied[:], d.xfrc_applied[:], d.ctrl[:] = saved[0], saved[1], saved[2]
        m.opt.gravity[:] = saved[5]
        if m.na:
            d.act[:] = saved[3]
        d.time = saved[4]


def _box_contacts(w, q, body_name: str = GRASP_BOX, mocap: Optional[Dict[int, np.ndarray]] = None) -> Dict[str, float]:
    """{infant body: deepest penetration in metres} of every contact with one object at pose q."""
    out: Dict[str, float] = {}
    for a, b, pen in w.contacts_now(q, mocap):
        if body_name in (a, b) and "world" not in (a, b):
            other = a if b == body_name else b
            out[other] = max(out.get(other, 0.0), pen)
    return out


def _opposition_deg(w, q, body_name: str = GRASP_BOX, mocap=None) -> Optional[float]:
    """R04's measure at pose q: the angle at the object's centre between the force-weighted mean
    contact point of the thumb and that of the fingers and palm (None when either is absent)."""
    m, d, mj = w.m, w.d, w.mj
    w.contacts_now(q, mocap)
    ob = m.body(body_name).id
    acc = {"thumb": [np.zeros(3), 0.0], "rest": [np.zeros(3), 0.0]}
    f6 = np.zeros(6)
    for i in range(int(d.ncon)):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if ob not in (b1, b2):
            continue
        nm = m.body(b2 if b1 == ob else b1).name
        if not nm.startswith("right_") or "arm" in nm:
            continue
        key = "thumb" if "_th" in nm else "rest"
        mj.mj_contactForce(m, d, i, f6)
        f = max(abs(float(f6[0])), 1e-6)
        acc[key][0] += f * np.asarray(c.pos)
        acc[key][1] += f
    if not acc["thumb"][1] or not acc["rest"][1]:
        return None
    ctr = np.asarray(d.xpos[ob])
    a = acc["thumb"][0] / acc["thumb"][1] - ctr
    b = acc["rest"][0] / acc["rest"][1] - ctr
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def _clear_scene(w, q) -> np.ndarray:
    """The other two boxes far away on the floor (the grasp needs the room they stand in)."""
    q = np.array(q)
    for nm, (x, y) in FAR_BOXES:
        a = w.free_adr(nm)
        q[a:a + 3] = (x, y, BOX_HALF - 0.001)
        q[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
    return q


def _toy_far(w):
    """A scene hook placing the toy (mocap) far away."""
    k = w.mocap_id(TOY)

    def scene(t, world):
        world.d.mocap_pos[k] = TOY_FAR
    return scene


def _set(q, w, values: Dict[str, float]) -> np.ndarray:
    q = np.array(q)
    for j, v in values.items():
        q[w.jq[j]] = v
    return q


# THE GRASP REBUILT ON A SMALLER CUBE (detector v9, item 9 of agents/observer_detectors_2026-09-25.md;
# the coordinator's decision of 8:20 PM: fix the test object, not the grasp definition). Read by its
# contact normals (the recorder track), the v8 grasp above put the thumb and the ring finger on
# NEIGHBOURING faces of the 6 cm cube (90-106 degrees), the index and middle tips 0.8-1.6 cm off it: no
# opposition under R04's normals (120-degree line). This hand cannot hold the 6 cm cube on opposite faces
# at all with the cube between its fingertips (of 2,762 random digit poses putting the thumb tip 6.0-6.8
# cm from a fingertip, no placement of 33,144 kept the rest of the hand out of the cube; another part at
# least 11.2 mm inside), and with the tips 1.0 cm below the palm's surface it opens at most 4.6 cm (1.5
# cm below: 2.5 cm; explore_grasp_depth.py). So the grasp cases hold box_light made a GRASP_SIDE_M cube
# (the 2.5 cm cube of infant block tests) for as long as each case lasts (_SmallBox; its density kept);
# every other case keeps the 6 cm box. The layout was searched like v8's (explore_small_layout.py:
# 7 turns x 4 x 4 centres x 2 heights, each settled by physics): the cube under the palm, its top face
# 1 mm below it, turned 35 degrees about the palm's normal, its centre 2.0 cm radial and 1.0 cm distal
# of the hand's origin; the thumb, index and middle tips pulled 1 N into their faces until MuJoCo's
# contacts stop them. The thumb and the index then press OPPOSITE faces, their contact normals 166
# degrees apart, and the grip HOLDS THE CUBE AGAINST GRAVITY: freed, with gravity on and the digits
# still pulled, it moves 2.4 mm in the first 0.25 s as the contacts settle and 3.4 mm by 3 s (at 0.5 N
# pulls 1.9 mm by 3 s); the same search's grips that did not hold crept 10-16 mm in 3 s
# (explore_small_hold.py, 2026-09-25 8:26 PM). _grasp measures both again on every build (g["normals_deg"],
# g["hold_travel_mm"]) and refuses a grip that fails them.
GRASP_SIDE_M = 0.025
GRASP_LAYOUT_V9 = dict(turn_deg=35.0, centre_cm=(2.0, 1.0), pull_n=1.0)     # no squeeze: THE GRIP IS PHYSICAL
GRASP_MIN_NORMALS_DEG = 150.0     # the recorder track's suggestion for a clear opposition (R04: 120)
GRASP_MAX_HOLD_MM = 5.0           # the freed cube's travel over 3 s in the grip
PALM_HALF_M = 0.0062              # the palm's half thickness (as _box_on_palm)


class _SmallBox:
    """box_light made a GRASP_SIDE_M cube (its density kept: mass and inertia scaled) while the context
    lasts, restored after; apply() may be called again and again (a recording's scene hook)."""

    def __init__(self, w):
        m = w.m
        self.w = w
        self.b = m.body(GRASP_BOX).id
        self.g = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == self.b][0]
        self.saved = None

    def apply(self):
        m = self.w.m
        if self.saved is None:
            self.saved = (np.array(m.geom_size[self.g]), float(m.body_mass[self.b]), np.array(m.body_inertia[self.b]))
        h0, h = float(self.saved[0][0]), GRASP_SIDE_M / 2.0
        m.geom_size[self.g][:] = h
        m.body_mass[self.b] = self.saved[1] * (h / h0) ** 3
        m.body_inertia[self.b][:] = float(m.body_mass[self.b]) * (2.0 * h) ** 2 / 6.0

    def restore(self):
        if self.saved is not None:
            m = self.w.m
            m.geom_size[self.g][:] = self.saved[0]
            m.body_mass[self.b] = self.saved[1]
            m.body_inertia[self.b][:] = self.saved[2]
            self.saved = None

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, *exc):
        self.restore()


def _small_box_scene(w, inner):
    """A scene hook that holds box_light at GRASP_SIDE_M through a recording (then calls the case's own
    scene); its `after` restores the box (ScriptWorld.record calls it)."""
    box = _SmallBox(w)

    def scene(t, world):
        box.apply()
        if inner is not None:
            inner(t, world)
    scene.after = box.restore
    return scene


def _on_small_box(builder):
    """A grasp case built and recorded with box_light at GRASP_SIDE_M."""
    def wrapped(w, *a, **k):
        with _SmallBox(w):
            pose, steps, eyes, scene = _built(builder(w, *a, **k))
        return pose, steps, eyes, _small_box_scene(w, scene)
    wrapped.__name__, wrapped.__doc__ = builder.__name__, builder.__doc__
    return wrapped


def _normals_deg(w, q, body_name: str = GRASP_BOX) -> Optional[float]:
    """The angle between the thumb's and the fingers' force-weighted mean contact normals on an object
    at pose q (the recorder track's R04 reads it; None when either touches nothing)."""
    m, d, mj = w.m, w.d, w.mj
    w.contacts_now(q)
    ob = m.body(body_name).id
    acc = {"thumb": np.zeros(3), "fingers": np.zeros(3)}
    f6 = np.zeros(6)
    for i in range(int(d.ncon)):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if ob not in (b1, b2):
            continue
        nm = m.body(b2 if b1 == ob else b1).name
        if not nm.startswith(HAND_DIGIT_BODIES):
            continue
        mj.mj_contactForce(m, d, i, f6)
        n = np.asarray(c.frame[:3]) * (1.0 if b1 != ob else -1.0)          # from the digit into the object
        acc["thumb" if nm.startswith("right_th") else "fingers"] += abs(float(f6[0])) * n
    a, b = acc["thumb"], acc["fingers"]
    if not np.linalg.norm(a) or not np.linalg.norm(b):
        return None
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


HAND_DIGIT_BODIES = ("right_th", "right_ff", "right_mf", "right_rf", "right_lf")


def _pull_torques(w, pulls, force: float) -> np.ndarray:
    """The generalized forces of pulling each (body, local point, world target) `force` newtons toward its target,
    at the world's current state (the Jacobian's transpose)."""
    m, d, mj = w.m, w.d, w.mj
    tau = np.zeros(m.nv)
    for body, local, target in pulls:
        pt = np.asarray(d.xpos[body]) + d.xmat[body].reshape(3, 3) @ np.asarray(local)
        e = np.asarray(target) - pt
        nrm = float(np.linalg.norm(e))
        if nrm < 1e-6:
            continue
        jacp = np.zeros((3, m.nv))
        mj.mj_jac(m, d, jacp, None, pt, body)
        tau += jacp.T @ (force * e / max(nrm, 0.005))
    return tau


def _free_grip(w, q, pulls, force: float, seconds: float, digits: bool = True, free_joints=None, torques=None):
    """Physics from pose q with box_light free: gravity on, the digits free (digits=True: DIGITS_R, or the joints
    named in free_joints) and each (body, local point, world target) pulled `force` newtons toward its target,
    every other joint held at q. Returns (the final qpos, the box's travel in metres); with `torques` a dict, it
    also receives {free joint: the pulls' torque on it at the final state}. Deterministic; the world's state is
    restored after."""
    m, d, mj = w.m, w.d, w.mj
    adr = w.free_adr(GRASP_BOX)
    box = m.body(GRASP_BOX).id
    free = list(free_joints) if free_joints is not None else (list(DIGITS_R) if digits else [])
    va = [w.jd[j] for j in free]
    keep = np.ones(m.nq, dtype=bool)
    keep[adr:adr + 7] = False
    if free:
        keep[[w.jq[j] for j in free]] = False
    vkeep = np.ones(m.nv, dtype=bool)
    bdof = int(m.jnt_dofadr[int(m.body_jntadr[box])])
    vkeep[bdof:bdof + 6] = False
    vkeep[va] = False
    saved = (np.array(d.qfrc_applied), np.array(d.ctrl), float(d.time))
    d.qpos[:] = q
    d.qvel[:] = 0.0
    d.qacc_warmstart[:] = 0.0
    d.ctrl[:] = 0.0
    d.qfrc_applied[:] = 0.0
    mj.mj_forward(m, d)
    start = np.array(d.xpos[box])
    try:
        for _ in range(int(round(seconds / float(m.opt.timestep)))):
            tau = _pull_torques(w, pulls, force)
            d.qfrc_applied[:] = 0.0
            if va:
                d.qfrc_applied[va] = tau[va]
            mj.mj_step(m, d)
            d.qpos[keep] = q[keep]
            d.qvel[vkeep] = 0.0
        d.qvel[:] = 0.0
        mj.mj_forward(m, d)
        if torques is not None:
            tau = _pull_torques(w, pulls, force)
            torques.update({j: float(tau[w.jd[j]]) for j in free})
        return np.array(d.qpos), float(np.linalg.norm(np.array(d.xpos[box]) - start))
    finally:
        d.qfrc_applied[:], d.ctrl[:] = saved[0], saved[1]
        d.time = saved[2]


def _holds_against_gravity(w, q, pulls, force: float, seconds: float = 3.0) -> float:
    """The grip at pose q with the object free: gravity on, the arm and body held at q, the digits free
    and still pulled `force` newtons toward the settle's targets. Returns the object's travel in metres
    after `seconds` of physics. Deterministic; the world's state is restored after."""
    return _free_grip(w, q, pulls, force, seconds)[1]


def _cube_rest_height(w, q) -> float:
    """box_light's centre height standing alone on the floor under its own weight: 0.5 s of physics with the cube
    alone free, where _all_boxes_far stands it, 2.4 m from the body (THE GRIP IS PHYSICAL, below)."""
    adr = w.free_adr(GRASP_BOX)
    q = np.array(q)
    q[adr:adr + 3] = (2.4, 2.0, GRASP_SIDE_M / 2.0 + 0.0005)       # where _all_boxes_far stands it
    q[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
    q_end, _ = _free_grip(w, q, [], 0.0, 0.5, digits=False)       # the cube alone
    return float(q_end[adr + 2])


def _cube_forces(w, q) -> Dict[str, float]:
    """{the body touching box_light: the summed normal force in newtons} at pose q, at rest (no velocity, no
    command): the force a posed recording of this pose records."""
    m, d, mj = w.m, w.d, w.mj
    w.contacts_now(q)
    ob = m.body(GRASP_BOX).id
    f6 = np.zeros(6)
    out: Dict[str, float] = {}
    for i in range(int(d.ncon)):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if ob not in (b1, b2):
            continue
        nm = m.body(b2 if b1 == ob else b1).name
        mj.mj_contactForce(m, d, i, f6)
        out[nm] = round(out.get(nm, 0.0) + abs(float(f6[0])), 3)
    return out


# THE GRIP IS PHYSICAL (detector v9; the independent review of 2026-09-25, "the grip itself"). Until then the posed
# recordings squeezed each touching tip 1 mm further into the 3.6 g cube after the physics settled the grip (the
# v8 construction's squeeze, kept so that contacts had a depth a 1e-6 difference in the start pose could not
# remove), and posed the cube 1 mm into the floor: the recordings read 25.8 N on the index fingertip, 18.4 N on
# the index knuckle and 6.3 N on the thumb - not in balance, a free cube would be pushed away - and 10.8 N
# between the cube and the floor, against the cube's weight of 0.035 N; the settled grip pressed with 1 N pulls.
# Now the grip is the state physics reaches: the cube stands on the floor at the height its own weight gives
# (_cube_rest_height), the arm at the floor grasp, the digits closed on it, and then the cube and the digits are
# free under gravity for GRIP_SETTLE_S with the digits still pulled GRASP_LAYOUT_V9's pull_n toward their
# targets (_free_grip): the floor keeps the cube upright, and it slides until the digits' pushes balance. The
# digits' angles and the cube's pose in the hand at the end are the grip every posed case records (the cube
# lifted in the hand keeps that pose; on the floor it stands where the physics left it). The squeeze is gone:
# the contacts keep the depth their own forces give. The state must still show the opposition the grasp needs
# (GRASP_MIN_NORMALS_DEG between the thumb's and the fingers' normals), the cube within GRASP_MAX_HOLD_MM of where
# it stood, upright within GRIP_MAX_TILT_DEG, and no key pose may press the arm or the cube into anything; a
# state that fails keeps the grip settled on the held cube (the same pulls, no squeeze) and says why in
# g["grip"], which also holds each part's force on the cube at rest (_cube_forces), printed at every build.
GRIP_SETTLE_S = 1.0
GRIP_MAX_TILT_DEG = 2.0


# THE GRIP IS HELD BY THE HAND'S MOTORS (the recorder track's review of 2026-09-26: posed, the lifted grasp cases
# recorded the index side pushing the 3.6 g cube with 3.3-3.7 N against the thumb's 1.47 N, 2.7-2.9 times what the
# cube's weight and the pair's friction can take, so R04's balance line refused every lifted record of
# grasp_lift_and_hold, grasp_lift_no_retention and grasp_carry_and_place). A posed frame's contact forces are what
# MuJoCo's forward dynamics computes for that state and those commands: a state physics never settled, with no
# command holding the digits, gives forces no hand would make. So the grip each lifted record shows is the state
# physics reaches with the cube free in the air in the hand at the lift pose (_grip_in_air: the arm at the 3 cm
# lift key, the thumb, index and middle free and pulled toward their targets, the ring and little fingers curled,
# gravity on, GRIP_SETTLE_S), and while the hand grips, the digits' motors carry the commands that make exactly
# those pulls (_hold_ctrl: each joint's pull torque over its motor's gear x gain; the posed case's pose.hold, read
# by ScriptWorld.record). Forward dynamics of that state under those commands reproduces the settled state's forces:
# the thumb against the fingers in balance, the cube's weight carried by the grip. On the floor the grip holds the
# floor state's commands (_grip_state); the cube's place in the hand moves from the floor state's to the air
# state's while the hand lifts it, and back while it sets it down.
def _hold_ctrl(w, torques: Dict[str, float]):
    """({motor: command} that applies these joint torques through each joint's own motor, the largest share of a
    motor's range they need - over 1 cannot be held exactly)."""
    m = w.m
    act = _body_motors(m)
    out, share = {}, 0.0
    for j, tau in torques.items():
        u = act.get(j)
        if u is None:
            share = max(share, float("inf"))
            continue
        tmax = float(m.actuator_gainprm[u][0]) * float(m.actuator_gear[u][0])
        c = tau / tmax if tmax else float("inf")
        lo = max(float(m.actuator_ctrlrange[u][0]), float(m.actuator_forcerange[u][0]) / max(1e-12, float(m.actuator_gainprm[u][0])))
        hi = min(float(m.actuator_ctrlrange[u][1]), float(m.actuator_forcerange[u][1]) / max(1e-12, float(m.actuator_gainprm[u][0])))
        if c > 0:
            share = max(share, c / hi if hi > 0 else float("inf"))
        elif c < 0:
            share = max(share, c / lo if lo < 0 else float("inf"))
        out[u] = float(np.clip(c, lo, hi))
    return out, share


GRIP_FREE_R = THUMB_R + FINGER_R["index"] + FINGER_R["middle"]     # the digits the grip's pulls close


def _grip_in_air(w, g0: dict, pull_local, force: float) -> dict:
    """The grip in the air (THE GRIP IS HELD BY THE HAND'S MOTORS): from the floor grip g0 (q0, hand, adr, lift arm
    pose, closed digits, ctr, R_hb), the state physics reaches with the cube free in the hand raised to the lift key:
    the thumb, index and middle free and pulled `force` newtons toward their targets, the ring and little fingers
    curled 35 degrees past the grip (as _grasp_frames' extra), everything else held. Returns dict(ok, why, closed,
    ctr, R_hb, hold, hold_share, travel_mm, normals_deg, forces_n)."""
    adr = g0["adr"]
    digits = dict(g0["closed"])
    for dg in ("ring", "little"):
        for r in ("knuckle", "middle", "distal"):
            j = "right_%s_%s" % ({"ring": "rf", "little": "lf"}[dg], r)
            digits[j] = float(min(w.jrange[j][1], digits[j] + math.radians(35.0)))
    q = _set(_set(g0["q0"], w, digits), w, g0["lift"])
    p_h, R_h = w.frame(q, g0["hand"])
    q[adr:adr + 3] = p_h + R_h @ g0["ctr"]
    q[adr + 3:adr + 7] = _mat2quat(w, R_h @ g0["R_hb"])
    pulls = [(b, loc, p_h + R_h @ t) for b, loc, t in pull_local]
    torques: Dict[str, float] = {}
    q_eq, travel = _free_grip(w, q, pulls, force, GRIP_SETTLE_S, free_joints=GRIP_FREE_R, torques=torques)
    hold, share = _hold_ctrl(w, torques)
    x, R_c = np.array(q_eq[adr:adr + 3]), _quat2mat(w, q_eq[adr + 3:adr + 7])
    normals = _normals_deg(w, q_eq)
    touching = _box_contacts(w, q_eq)
    why = []
    if normals is None or normals < GRASP_MIN_NORMALS_DEG:
        why.append("the normals %s degrees" % (None if normals is None else round(normals, 1)))
    if not (any(k.startswith("right_th") for k in touching) and any(k.startswith(("right_ff", "right_mf")) for k in touching)):
        why.append("the thumb or the fingers do not touch the cube")
    if travel > GRASP_MAX_HOLD_MM / 1000.0:
        why.append("the cube moved %.1f mm" % (1000.0 * travel))
    if share > 1.0:
        why.append("the digits' motors would need %.2f of their range" % share)
    return dict(ok=not why, why="; ".join(why), closed={j: float(q_eq[w.jq[j]]) for j in GRIP_FREE_R},
                ctr=R_h.T @ (x - p_h), R_hb=R_h.T @ R_c, hold=hold, hold_share=round(share, 3),
                travel_mm=round(1000.0 * travel, 2), normals_deg=None if normals is None else round(normals, 1),
                forces_n=_cube_forces_held(w, q_eq, hold))


def _cube_forces_held(w, q, hold: Dict[int, float]) -> Dict[str, float]:
    """_cube_forces with the digits' motors carrying `hold` ({motor: command}): the forces a posed record of this
    state records while the hand holds the grip."""
    d = w.d
    saved = np.array(d.ctrl)
    try:
        d.ctrl[:] = 0.0
        for u, c in hold.items():
            d.ctrl[u] = c
        m, mj = w.m, w.mj
        w.place_far()
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mj.mj_forward(m, d)
        ob = m.body(GRASP_BOX).id
        f6 = np.zeros(6)
        out: Dict[str, float] = {}
        for i in range(int(d.ncon)):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if ob not in (b1, b2):
                continue
            nm = m.body(b2 if b1 == ob else b1).name
            mj.mj_contactForce(m, d, i, f6)
            out[nm] = round(out.get(nm, 0.0) + abs(float(f6[0])), 3)
        return out
    finally:
        d.ctrl[:] = saved


def _grip_hold(g, t: float, s: float, on: float, off: float) -> Dict[int, float]:
    """pose.hold for a grasp case (THE GRIP IS HELD BY THE HAND'S MOTORS): while the hand grips (on <= t < off),
    the floor state's holding commands blended into the air state's by s (0 on the floor, 1 lifted)."""
    if not (on <= t < off):
        return {}
    floor = g.get("floor_hold") or {}
    air = (g.get("air") or {}).get("hold") or {}
    out = {}
    for u in set(floor) | set(air):
        out[u] = (1.0 - s) * floor.get(u, 0.0) + s * air.get(u, 0.0)
    return out


def _box_in_hand_blend(w, g, q, s: float) -> np.ndarray:
    """q with the grasped box in the hand, its place moved from the floor grip's (s 0) to the air grip's (s 1)
    (THE GRIP IS HELD BY THE HAND'S MOTORS); the floor grip's alone when there is no air grip."""
    air = g.get("air")
    if not air or s <= 0.0:
        return _box_in_hand(w, g, q)
    p, R = w.frame(q, g["hand"])
    ctr = (1.0 - s) * np.asarray(g["ctr"]) + s * np.asarray(air["ctr"])
    qa, qb = _mat2quat(w, g["R_hb"]), _mat2quat(w, air["R_hb"])
    if float(np.dot(qa, qb)) < 0.0:
        qb = -qb
    qr = (1.0 - s) * qa + s * qb
    qr = qr / np.linalg.norm(qr)
    q = np.array(q)
    q[g["adr"]:g["adr"] + 3] = p + R @ ctr
    q[g["adr"] + 3:g["adr"] + 7] = _mat2quat(w, R @ _quat2mat(w, qr))
    return q


def _air_digits(g, digits: Dict[str, float]) -> Dict[str, float]:
    """digits with the thumb, index and middle at the air grip's angles (none changed without an air grip)."""
    air = g.get("air")
    return dict(digits, **air["closed"]) if air else dict(digits)


def _grip_state(w, g0: dict, pull_local, force: float) -> dict:
    """The physical grip (THE GRIP IS PHYSICAL): from the designed floor grasp g0 (q0, hand, adr, arm, closed,
    box_floor, R_h, R_hb), the grip physics settles into. Returns dict(ok, why, closed, box_floor, floor_quat, ctr,
    R_hb, travel_mm, normals_deg)."""
    m = w.m
    adr = g0["adr"]
    q = _set(_set(g0["q0"], w, g0["closed"]), w, g0["arm"])
    q[adr:adr + 3] = g0["box_floor"]
    q[adr + 3:adr + 7] = _mat2quat(w, g0["R_h"] @ g0["R_hb"])
    p_hf, R_hf = w.frame(q, g0["hand"])
    pulls = [(b, loc, p_hf + R_hf @ t) for b, loc, t in pull_local]
    torques: Dict[str, float] = {}
    q_eq, travel = _free_grip(w, q, pulls, force, GRIP_SETTLE_S, free_joints=DIGITS_R, torques=torques)
    hold, share = _hold_ctrl(w, {j: v for j, v in torques.items() if j in GRIP_FREE_R})
    x = np.array(q_eq[adr:adr + 3])
    R_c = _quat2mat(w, q_eq[adr + 3:adr + 7])
    up = R_c @ np.array([0.0, 1.0, 0.0])                   # the cube's axis that stood vertical (R_h's y)
    tilt = math.degrees(math.acos(float(np.clip(abs(up[2]), -1.0, 1.0))))
    normals = _normals_deg(w, q_eq)
    touching = _box_contacts(w, q_eq)
    thumb = any(k.startswith("right_th") for k in touching)
    finger = any(k.startswith(("right_ff", "right_mf")) for k in touching)
    why = []
    if normals is None or normals < GRASP_MIN_NORMALS_DEG:
        why.append("the normals %s degrees" % (None if normals is None else round(normals, 1)))
    if not (thumb and finger):
        why.append("the thumb or the fingers do not touch the cube")
    if travel > GRASP_MAX_HOLD_MM / 1000.0:
        why.append("the cube moved %.1f mm" % (1000.0 * travel))
    if tilt > GRIP_MAX_TILT_DEG:
        why.append("the cube tilted %.1f degrees" % tilt)
    return dict(ok=not why, why="; ".join(why), closed={j: float(q_eq[w.jq[j]]) for j in DIGITS_R}, box_floor=x,
                floor_quat=np.array(q_eq[adr + 3:adr + 7]), ctr=R_hf.T @ (x - p_hf), R_hb=R_hf.T @ R_c,
                travel_mm=round(1000.0 * travel, 2), normals_deg=None if normals is None else round(normals, 1),
                tilt_deg=round(tilt, 2), q=q_eq, hold=hold if share <= 1.0 else {}, hold_share=round(share, 3))


def _grasp(w) -> dict:
    """The grasp construction for one world (cached; THE GRASP REBUILT ON A SMALLER CUBE above), made with
    box_light at GRASP_SIDE_M: the cube in the hand's frame, the digits open and closed on it, and arm
    poses that put the grasped cube upright on the floor, 2.5 cm above it (before closing), lifted 3 cm,
    carried 8 cm and set down again. The arm pose on the floor turns the palm down, the fingers along a
    heading (every 15 degrees), and is the first the arm reaches (2 mm, 2 degrees) with the arm, the trunk
    and the cube no more than 3 mm into one another or the floor (v8's put the upper arm 26 mm into it)."""
    g = getattr(w, "_grasp_cache", None)
    if g is not None:
        return g
    with _SmallBox(w):
        g = _build_small_grasp(w)
    w._grasp_cache = g
    return g


def _build_small_grasp(w) -> dict:
    m = w.m
    hb = m.body("right_hand").id
    adr = w.free_adr(GRASP_BOX)
    L = GRASP_LAYOUT_V9
    h = GRASP_SIDE_M / 2.0
    q0 = _clear_scene(w, w.base())
    w.place_far()
    w.d.mocap_pos[w.mocap_id(TOY)] = TOY_FAR
    start = {j: float(q0[w.jq[j]]) for j in THUMB_R}
    for dg in FINGER_R:
        for j in FINGER_R[dg]:
            start[j] = 0.0 if not j.endswith("_side") else float(q0[w.jq[j]])
    for dg in ("ring", "little"):
        for j in FINGER_R[dg][1:]:
            start[j] = float(min(w.jrange[j][1], 1.4))    # curled out of the way
    q = _set(q0, w, start)
    p0, R0 = w.frame(q, hb)
    th = math.radians(L["turn_deg"])
    a = np.array([-math.sin(th), 0.0, math.cos(th)])       # thumb -> fingers, in the palm's plane
    b = np.array([0.0, 1.0, 0.0])                          # toward the palm
    c = np.cross(a, b)
    ctr = np.array([L["centre_cm"][0] / 100.0, -(PALM_HALF_M + h + 0.001), L["centre_cm"][1] / 100.0])
    R_hb = np.stack([a, b, c], axis=1)
    qb = np.array(q)
    qb[adr:adr + 3] = p0 + R0 @ ctr
    qb[adr + 3:adr + 7] = _mat2quat(w, R0 @ R_hb)
    tips = {dg: w.tip_local("R", dg) for dg in ("thumb", "index", "middle")}
    targets = [("thumb", ctr - h * a), ("index", ctr + h * a + 0.4 * h * c), ("middle", ctr + h * a - 0.4 * h * c)]
    pull_local = []                                         # (tip body, tip point, target in the hand's frame)
    for dg, tg in targets:
        inward = (ctr - tg) / max(1e-9, float(np.linalg.norm(ctr - tg)))
        pull_local.append((tips[dg][0], tips[dg][1], tg + 0.01 * inward))
    pulls = [(b_, loc, p0 + R0 @ t_) for b_, loc, t_ in pull_local]
    qs = _settle(w, qb, pulls, DIGITS_R, force=float(L["pull_n"]))
    normals = _normals_deg(w, qs)
    hold = _holds_against_gravity(w, qs, pulls, float(L["pull_n"]))
    if normals is None or normals < GRASP_MIN_NORMALS_DEG or hold > GRASP_MAX_HOLD_MM / 1000.0:
        raise RuntimeError("the grasp: the settled grip fails (normals %s degrees, the freed cube moved %.1f mm)"
                           % (None if normals is None else round(normals, 1), 1000.0 * hold))
    # no squeeze (THE GRIP IS PHYSICAL): the settled grip here, and below the state physics reaches on the floor
    closed_digits = {j: float(qs[w.jq[j]]) for j in DIGITS_R}
    # the open hand: the four fingers straight and lifted 10 degrees at the knuckle, the thumb pulled 4 cm
    # toward the back of the hand (at rest its base lies under the palm)
    lifted = {j: float(q0[w.jq[j]]) for j in THUMB_R}
    for dg in FINGER_R:
        for j in FINGER_R[dg]:
            lifted[j] = 0.0 if not j.endswith("_side") else float(q0[w.jq[j]])
        j = FINGER_R[dg][1]
        lifted[j] = float(max(w.jrange[j][0], math.radians(-10.0)))
    bt, lt = tips["thumb"]
    q_open = _set(qb, w, lifted)
    q_open = _settle(w, q_open, [(bt, lt, w.point(q_open, bt, lt) + 0.04 * (R0 @ b))], THUMB_R, steps=100, force=1.0)
    for j in THUMB_R:
        lifted[j] = float(q_open[w.jq[j]])
    open_touching = _box_contacts(w, _set(qb, w, lifted))
    # arm poses: palm down, the fingers along GRASP_HEADING_DEG, the cube upright on the floor at GRASP_FLOOR_XY;
    # the carry toward GRASP_CARRY_DEG. Posed, nothing stops an arm going into the floor (the shoulder lies at the
    # floor's height): the place, heading and carry were searched so that no key pose presses the arm into the
    # floor or the trunk (explore_grasp_place.py, 2026-09-25 9:00 PM: 10 cm x 2 cm grid of places, headings
    # every 15 degrees, carries every 15 degrees; this one 0.0 mm at the grasp, above, lift, carry and set-down,
    # where the first build pressed 4.1 mm at the grasp, 81 N, and 13.5 mm at the set-down, 297.5 N)
    y_h = np.array([0.0, 0.0, 1.0])
    psi = math.radians(GRASP_HEADING_DEG)
    z_h = np.array([math.cos(psi), math.sin(psi), 0.0])
    R_h = np.stack([np.cross(y_h, z_h), y_h, z_h], axis=1)
    box_floor = np.array([GRASP_FLOOR_XY[0], GRASP_FLOOR_XY[1], _cube_rest_height(w, q0)])    # v9 before: h - 1 mm
    p_h = box_floor - R_h @ ctr
    arm_q, ep, er = w.ik_pose(_set(q0, w, closed_digits), hb, p_h, R_h, ARM_R, iters=400)
    if ep >= 0.002 or er >= 2.0:
        raise RuntimeError("the grasp: the arm cannot put the gripped cube at its place (%.4f m, %.2f deg)" % (ep, er))
    arm = {j: float(arm_q[w.jq[j]]) for j in ARM_R}

    def arm_at(offset) -> Dict[str, float]:
        qa, ep_, er_ = w.ik_pose(_set(q0, w, arm), hb, p_h + np.asarray(offset), R_h, ARM_R, iters=400)
        if ep_ > 0.002 or er_ > 2.0:
            raise RuntimeError("the grasp: the hand cannot reach %s (%.4f m, %.2f deg)" % (offset, ep_, er_))
        return {j: float(qa[w.jq[j]]) for j in ARM_R}

    ang = math.radians(GRASP_CARRY_DEG)
    carry_dir = np.array([math.cos(ang), math.sin(ang), 0.0])
    keys = dict(above=arm_at((0, 0, 0.025)), lift=arm_at((0, 0, 0.03)),
                carry=arm_at(tuple(0.08 * carry_dir + np.array([0, 0, 0.03]))),
                down=arm_at(tuple(0.08 * carry_dir)), away=arm_at(tuple(0.08 * carry_dir + np.array([0, 0, 0.03]))))
    def key_depths(digits, cube_floor, quat) -> Dict[str, float]:
        out = {}
        for name, pose_, cube in (("grasp", arm, cube_floor), ("above", keys["above"], cube_floor),
                                  ("lift", keys["lift"], cube_floor + np.array([0, 0, 0.03])),
                                  ("carry", keys["carry"], cube_floor + 0.08 * carry_dir + np.array([0, 0, 0.03])),
                                  ("down", keys["down"], cube_floor + 0.08 * carry_dir)):
            qd = _set(_set(q0, w, digits), w, pose_)
            qd[adr:adr + 3] = cube
            qd[adr + 3:adr + 7] = quat
            out[name] = 1000.0 * _grasp_depth(w, qd)
        return out
    floor_quat = _mat2quat(w, R_h @ R_hb)
    depths = key_depths(closed_digits, box_floor, floor_quat)
    if max(depths.values()) > 1000.0 * GRASP_MAX_FLOOR_M:
        raise RuntimeError("the grasp: a key pose presses the arm or the cube %.1f mm into something" % max(depths.values()))
    # THE GRIP IS PHYSICAL: the state physics reaches with the cube and the digits free on the floor
    state = _grip_state(w, dict(q0=q0, hand=hb, adr=adr, arm=arm, closed=closed_digits, box_floor=box_floor, R_h=R_h,
                                R_hb=R_hb), pull_local, float(L["pull_n"]))
    if state["ok"]:
        d_eq = key_depths(state["closed"], state["box_floor"], state["floor_quat"])
        if max(d_eq.values()) > 1000.0 * GRASP_MAX_FLOOR_M:
            state.update(ok=False, why="a key pose presses the arm or the cube %.1f mm into something" % max(d_eq.values()))
        else:
            depths = d_eq
    if state["ok"]:
        closed_digits, box_floor, floor_quat = state["closed"], state["box_floor"], state["floor_quat"]
        ctr, R_hb = state["ctr"], state["R_hb"]
        q_grip = state["q"]
        source = "physics: the cube and the digits free on the floor for %.1f s" % GRIP_SETTLE_S
    else:
        q_grip = _set(_set(q0, w, closed_digits), w, arm)
        q_grip[adr:adr + 3] = box_floor
        q_grip[adr + 3:adr + 7] = floor_quat
        source = "settled on the held cube (the floor state failed: %s)" % state["why"]
    floor_hold = state.get("hold") if state["ok"] else {}
    grip = dict(source=source, forces_n=_cube_forces_held(w, q_grip, floor_hold or {}), travel_mm=state["travel_mm"],
                normals_deg=state["normals_deg"], tilt_deg=state["tilt_deg"], rest_height_m=round(float(box_floor[2]), 6),
                hold_share=state.get("hold_share"))
    print("the grasp's grip: %s; forces on the cube, held %s" % (source, grip["forces_n"]), flush=True)
    # THE GRIP IS HELD BY THE HAND'S MOTORS: the grip the lifted records show
    air = _grip_in_air(w, dict(q0=q0, hand=hb, adr=adr, lift=keys["lift"], closed=closed_digits, ctr=ctr, R_hb=R_hb),
                       pull_local, float(L["pull_n"]))
    print("the grasp's grip in the air: %s; forces on the cube, held %s; travel %.2f mm; the motors' largest share %.2f"
          % ("physics" if air["ok"] else "not used (%s)" % air["why"], air["forces_n"], air["travel_mm"],
             air["hold_share"]), flush=True)
    grip["air"] = {k: air[k] for k in ("ok", "why", "forces_n", "travel_mm", "normals_deg", "hold_share")}
    touching = _box_contacts(w, q_grip)
    opposition = _opposition_deg(w, q_grip)
    err = (ep, er, depths["grasp"] / 1000.0)
    carry_depth = depths["down"] / 1000.0
    return dict(q0=q0, hand=hb, adr=adr, ctr=ctr, R_hb=R_hb, R_h=R_h, p_h=p_h, open=lifted, straight=lifted,
                closed=closed_digits, arm=arm, keys=keys, carry_dir=carry_dir, box_floor=box_floor, floor_quat=floor_quat,
                touching=touching, open_touching=open_touching, opposition_deg=opposition, arm_error=err,
                normals_deg=normals, hold_travel_mm=1000.0 * hold, carry_floor_mm=1000.0 * carry_depth,
                key_pose_depth_mm=depths, grip=grip, floor_hold=floor_hold or {},
                air=dict(closed=air["closed"], ctr=air["ctr"], R_hb=air["R_hb"], hold=air["hold"]) if air["ok"] else None)


GRASP_FLOOR_XY = (0.08, 0.26)
GRASP_HEADING_DEG = 150.0
GRASP_CARRY_DEG = 30.0
GRASP_MAX_FLOOR_M = 0.001
TRUNK_BODIES = ("upper_body", "chest", "lower_body", "hip")


def _grasp_depth(w, q) -> float:
    """The deepest contact at pose q between the grasped box and anything but the digits and the
    floor, or between the right arm and the floor or the trunk (metres)."""
    deep = 0.0
    for n1, n2, pen in w.contacts_now(q):
        pair = (n1, n2)
        if GRASP_BOX in pair:
            other = n2 if n1 == GRASP_BOX else n1
            if other != "world" and not other.startswith(HAND_DIGIT_BODIES):
                deep = max(deep, pen)
        elif any(n.startswith("right_") for n in pair) and ("world" in pair or any(n.startswith(TRUNK_BODIES) for n in pair)):
            deep = max(deep, pen)
    return deep


def _timeline(w, frames):
    """pose(t) from keyframes [(t, {joint: angle})]: each joint moves linearly between the
    keyframes that name it and holds before the first and after the last."""
    names = sorted({j for _, v in frames for j in v})
    tracks = {j: [(t, v[j]) for t, v in frames if j in v] for j in names}

    def at(t) -> Dict[str, float]:
        out = {}
        for j, tr in tracks.items():
            if t <= tr[0][0]:
                out[j] = tr[0][1]
                continue
            for (t0, v0), (t1, v1) in zip(tr[:-1], tr[1:]):
                if t0 <= t <= t1:
                    out[j] = v0 + (v1 - v0) * ((t - t0) / (t1 - t0) if t1 > t0 else 1.0)
                    break
            else:
                out[j] = tr[-1][1]
        return out
    return at


def _box_in_hand(w, g, q) -> np.ndarray:
    """q with the grasped box where the hand holds it (the grasp's box-in-hand pose)."""
    p, R = w.frame(q, g["hand"])
    q = np.array(q)
    q[g["adr"]:g["adr"] + 3] = p + R @ g["ctr"]
    q[g["adr"] + 3:g["adr"] + 7] = _mat2quat(w, R @ g["R_hb"])
    return q


def _box_on_floor(g, q, pos=None, quat=None) -> np.ndarray:
    q = np.array(q)
    q[g["adr"]:g["adr"] + 3] = g["box_floor"] if pos is None else pos
    if quat is not None:
        q[g["adr"] + 3:g["adr"] + 7] = quat
    return q


def _grasp_frames(w, g, t_close: float = 15.0, close_steps: float = 2.0):
    """The shared approach: rest (t < 4), the arm to 2.5 cm above the box (4-12), down to the
    grasp (12-15), the digits closing on the box (t_close .. t_close + close_steps), then the ring
    and little fingers curling 35 degrees further (closing on the object: H02's closed_on)."""
    rest = {j: float(g["q0"][w.jq[j]]) for j in ARM_R + DIGITS_R}
    above = dict(g["keys"]["above"], **g["open"])
    grasp = dict(g["arm"], **g["open"])
    closed = dict(g["arm"], **g["closed"])
    extra = dict(closed)
    for dg in ("ring", "little"):
        for r in ("knuckle", "middle", "distal"):
            j = "right_%s_%s" % ({"ring": "rf", "little": "lf"}[dg], r)
            extra[j] = float(min(w.jrange[j][1], closed[j] + math.radians(35.0)))
    return [(4.0, rest), (12.0, above), (t_close, grasp), (t_close + close_steps, closed), (t_close + close_steps + 2.0, extra)], extra


def _floor_yaw_quat(w, g) -> np.ndarray:
    """The box's world orientation while it stands in the grasp on the floor (THE GRIP IS PHYSICAL: where the
    physics left it)."""
    if g.get("floor_quat") is not None:
        return np.array(g["floor_quat"])
    return _mat2quat(w, g["R_h"] @ g["R_hb"])


# THE PINCH HELD BY THE BODY'S OWN MOTORS (the recorder track, 2026-09-26; the second review of
# fix/observer-recorder-2026-09-25, finding 1: R04 had never fired on the crib hand). A posed recording cannot
# give a grip's forces: every frame is computed from a pose the physics never settled. The posed grasp cases
# record the index side pushing the 3.6 g cube 2.2-2.5 times harder than the thumb in the air (3.3-3.7 N
# against 1.47 N in grasp_lift_and_hold's lifted records), 2.7-2.9 times what the cube's weight and the pair's
# friction can take, so R04's balance line refuses every posed grip (a ModelWorld dry run of the merged tree,
# 2026-09-26). This case has real dynamics end to end: the cube is free under gravity and contacts; the arm
# holds the construction's grasp pose raised PINCH_HOLD_M with its own motors (a proportional-derivative law
# plus the gravity and joint-spring torques the controller reads from the body's state); the thumb, index and
# middle push PINCH_PUSH_N newtons toward the construction's own pull targets (1 cm inside the cube's two
# opposite faces) through their own motors (the Jacobian's transpose, clipped to each motor's range); the ring
# and little fingers hold the construction's curl. It starts from the construction's settled grip and lets
# grip and body settle for PINCH_SETTLE_S under the same controller before the recording begins (unsettled,
# the raised arm rolls the lying body and the palm dropped 18 mm in the first 250 ms). Measured in the saved
# world model (ModelWorld, 2026-09-26): the case meets its lines in 21 of 21 runs - as written, ten starts
# with 1e-5 noise on every coordinate, the push at 1.2, 1.3, 1.7 and 2.0 N, the settle at 0.6, 0.8, 1.2 and
# 1.5 s, the arm's gain 3 and 8 - and in every run the thumb against the fingers grips in 36-40 of the 41
# records (their pushes 170-180 degrees apart), the cube's centre stays at least 17.7 mm above the floor (half
# its side is 12.5 mm), nothing but the hand touches it (in 20 of the 21; with the settle at 1.5 s a corner of
# the tilting cube meets the floor in some records), and the largest body contact is the head resting on the
# floor (43 N). H02 and R05 are found too (the hand carries the cube a little as the lying body settles):
# not asserted.
PINCH_PUSH_N = 1.5
PINCH_SETTLE_S = 1.0
PINCH_HOLD_M = 0.06                   # the hand's hold pose above the construction's grasp pose on the floor
PINCH_ARM_GAINS = (5.0, 0.5)          # command per radian of error and per radian per second, of the motor's range
PINCH_DIGIT_DAMPING = 0.05


def case_pinch_held_driven(w):
    """R04's positive on the crib hand with real dynamics (THE PINCH HELD BY THE BODY'S OWN MOTORS above):
    the right thumb against the index and middle fingers holds the GRASP_SIDE_M cube in the air, the hand
    PINCH_HOLD_M above its grasp pose on the floor, for 2 s (41 records), nothing else touching the cube."""
    m = w.m
    g = _grasp(w)
    q0, hb = g["q0"], g["hand"]
    h = GRASP_SIDE_M / 2.0
    a, c = g["R_hb"][:, 0], g["R_hb"][:, 2]                   # thumb -> fingers; across the face
    ctr = g["ctr"]
    tips = {dg: w.tip_local("R", dg) for dg in ("thumb", "index", "middle")}
    goal = {}                                                 # the construction's pull targets, in the hand's frame
    for dg, tg in (("thumb", ctr - h * a), ("index", ctr + h * a + 0.4 * h * c), ("middle", ctr + h * a - 0.4 * h * c)):
        goal[dg] = tg + 0.01 * (ctr - tg) / max(1e-9, float(np.linalg.norm(ctr - tg)))
    joints = dict(thumb=THUMB_R, index=FINGER_R["index"], middle=FINGER_R["middle"])
    act = {}
    for u in range(int(m.nu)):
        if int(m.actuator_trntype[u]) == 0:
            act[m.joint(int(m.actuator_trnid[u][0])).name.split(":")[-1]] = u
    gain = getattr(w.env, "_native_actuator_gain", None)      # BridgeEnv may hold gains scaled by fatigue
    gain = np.array(m.actuator_gainprm if gain is None else gain)
    tmax = {j: float(gain[act[j]][0] * m.actuator_gear[act[j]][0]) for j in ARM_R + DIGITS_R}
    # the hold pose: the construction's grasp pose raised PINCH_HOLD_M (at the 3 cm lift key the lying body's
    # settle lowers the hand until a corner of the tilting cube meets the floor)
    qa, ep, er = w.ik_pose(_set(q0, w, g["arm"]), hb, g["p_h"] + np.array([0.0, 0.0, PINCH_HOLD_M]), g["R_h"], ARM_R,
                           iters=400)
    if ep > 0.002 or er > 2.0:
        raise RuntimeError("the pinch: the hand cannot reach its hold pose (%.4f m, %.2f deg)" % (ep, er))
    lift = {j: float(qa[w.jq[j]]) for j in ARM_R}
    with _SmallBox(w):
        deep = _grasp_depth(w, _box_in_hand(w, g, _set(q0, w, dict(lift, **g["closed"]))))
    if deep > GRASP_MAX_FLOOR_M:
        raise RuntimeError("the pinch: the hold pose presses the arm or the cube %.1f mm into something" % (1000 * deep))
    # the construction's settled grip (_build_small_grasp before its 1 mm squeeze), made again: the digits
    # pulled GRASP_LAYOUT_V9's pull toward points 1 cm inside the cube's faces until the contacts stop them
    with _SmallBox(w):
        sd = {j: float(q0[w.jq[j]]) for j in THUMB_R}
        for dg in FINGER_R:
            for j in FINGER_R[dg]:
                sd[j] = 0.0 if not j.endswith("_side") else float(q0[w.jq[j]])
        for dg in ("ring", "little"):
            for j in FINGER_R[dg][1:]:
                sd[j] = float(min(w.jrange[j][1], 1.4))
        q = _set(q0, w, sd)
        p0, R0 = w.frame(q, hb)
        q[g["adr"]:g["adr"] + 3] = p0 + R0 @ ctr
        q[g["adr"] + 3:g["adr"] + 7] = _mat2quat(w, R0 @ g["R_hb"])
        qs = _settle(w, q, [(tips[dg][0], tips[dg][1], p0 + R0 @ goal[dg]) for dg in ("thumb", "index", "middle")],
                     DIGITS_R, force=float(GRASP_LAYOUT_V9["pull_n"]))
    settled = {j: float(qs[w.jq[j]]) for j in DIGITS_R}
    ka, kd = PINCH_ARM_GAINS

    def command(t, world):
        d = world.d
        ctrl = np.zeros(world.m.nu)
        pH, RH = np.array(d.xpos[hb]), np.array(d.xmat[hb]).reshape(3, 3)
        for dg, (body, local) in tips.items():
            pt = np.array(d.xpos[body]) + np.array(d.xmat[body]).reshape(3, 3) @ np.asarray(local)
            jacp = np.zeros((3, world.m.nv))
            world.mj.mj_jac(world.m, d, jacp, None, pt, body)
            e = pH + RH @ goal[dg] - pt                       # toward the construction's pull target
            tau = jacp.T @ (PINCH_PUSH_N * e / max(float(np.linalg.norm(e)), 0.005))
            for j in joints[dg]:
                dof = world.jd[j]
                ctrl[act[j]] = float(np.clip(tau[dof] / tmax[j] - PINCH_DIGIT_DAMPING * float(d.qvel[dof]), -1.0, 1.0))
        for j in ARM_R + FINGER_R["ring"] + FINGER_R["little"]:
            dof, qa = world.jd[j], world.jq[j]
            target = lift[j] if j in lift else settled[j]
            if j in lift:
                tau = (ka * (target - float(d.qpos[qa])) - kd * float(d.qvel[dof])) * tmax[j] \
                    + float(d.qfrc_bias[dof]) - float(d.qfrc_passive[dof])      # gravity and the joint's spring
            else:
                tau = (2.0 * (target - float(d.qpos[qa])) - PINCH_DIGIT_DAMPING * float(d.qvel[dof])) * tmax[j]
            ctrl[act[j]] = float(np.clip(tau / tmax[j], -1.0, 1.0))
        return ctrl

    def start(world):
        """The settled grip at the lift pose, then PINCH_SETTLE_S of the same controller (MuJoCo's own step)."""
        q = _box_in_hand(world, g, _set(q0, world, dict(lift, **settled)))
        d, mj = world.d, world.mj
        d.qpos[:] = q
        d.qvel[:] = 0.0
        d.qacc_warmstart[:] = 0.0
        d.ctrl[:] = 0.0
        mj.mj_forward(world.m, d)
        per = max(1, int(round(float(world.env.dt) / float(world.m.opt.timestep))))
        for _ in range(int(round(PINCH_SETTLE_S / float(world.env.dt)))):
            d.ctrl[:] = command(0.0, world)
            for _ in range(per):
                mj.mj_step(world.m, d)
        d.ctrl[:] = 0.0
        return np.array(d.qpos)

    return Driven(start, command, 41, _small_box_scene(w, _toy_far(w)))


# THE LIFT HELD BY THE BODY'S OWN MOTORS (the observer candidate, 2026-09-26; the coordinator's decision: the
# adopted rule is physical, motor-driven cases, never a settled pose). The posed lifted grasp records could not
# grip: the air settle lost the cube, so they carried the floor grip into the air, the index side pushing the
# 3.6 g cube 0.71-0.89 N against the thumb's 0.36 N, 1.81-2.56 times what its weight and the pair's friction can
# take (the candidate's dry run; measurements/observer_candidate_grip_per_record_2026-09-26.json). These two
# cases have real dynamics end to end, as grasp_pinch_held_driven: the cube is free under gravity and contacts;
# the thumb, index and middle push PINCH_PUSH_N toward the construction's pull targets through their own motors;
# the ring and little fingers hold the construction's curl; the arm holds the hand at the construction's floor
# grasp pose and orientation with its own motors (a force and a torque on the hand through the arm's Jacobian,
# plus the gravity and joint-spring torques the controller reads from the body's state), and lifts it. Each
# starts from the construction's physical floor grip (_grip_state) and lets grip and body settle for
# GRASP_DRIVEN_SETTLE_S before anything else. The lift is judged by the cube's height above its rest, so the
# hand's height carries a slow correction toward the cube's scripted height once the ramp ends (the grip gives
# a little as the floor stops carrying the cube).
#   hold          on the floor for 8 records, the cube lifted GRASP_DRIVEN_LIFT_M over GRASP_DRIVEN_RAMP_S, then
#                 held still to the end (38 records)
#   no_retention  the lift and GRASP_DRIVEN_HOLD_S of holding happen in the start (physics, not recorded): the
#                 recording opens on the cube held GRASP_DRIVEN_LIFT_M above its rest, and from its third record
#                 the thumb, index and middle open at full command, so the hand touches the cube in 3 records
#                 (under H02's 200 ms) and it falls back to the floor (26 records)
GRASP_DRIVEN_LIFT_M = 0.03
GRASP_DRIVEN_SETTLE_S = 1.0
GRASP_DRIVEN_RAMP_S = 0.4
GRASP_DRIVEN_HOLD_S = 0.6
GRASP_DRIVEN_HAND_GAINS = (1500.0, 60.0)     # newtons per metre of the hand's error, per metre per second
GRASP_DRIVEN_TURN_GAINS = (4.0, 0.2)          # newton-metres per radian of the hand's turn, per radian per second
GRASP_DRIVEN_LIFT_I = 4.0                     # per second: the hand's height correction toward the cube's height
GRASP_DRIVEN_LIFT_I_MAX = 0.01                # metres the correction may reach
GRASP_DRIVEN_RELEASE_AT = 3.0                 # no_retention: the record whose burst the digits start opening in
GRASP_DRIVEN_RELEASE_N = 3.0                  # no_retention: newtons pushing each tip straight away from the cube
GRASP_DRIVEN_DROP_STEPS = 100                 # no_retention: records; the dropped cube rocks on an edge for up to 2 s, and R05
#                                               reads the lift against the median of the floor records


def case_grasp_driven(w, kind: str):
    """The lifted grasp cases with real dynamics (THE LIFT HELD BY THE BODY'S OWN MOTORS above): kind "hold"
    (grasp_lift_and_hold) or "no_retention" (grasp_lift_no_retention)."""
    m = w.m
    g = _grasp(w)
    q0, hb, adr = g["q0"], g["hand"], g["adr"]
    h = GRASP_SIDE_M / 2.0
    a, c = g["R_hb"][:, 0], g["R_hb"][:, 2]
    ctr = g["ctr"]
    tips = {dg: w.tip_local("R", dg) for dg in ("thumb", "index", "middle")}
    goal = {}
    for dg, tg in (("thumb", ctr - h * a), ("index", ctr + h * a + 0.4 * h * c), ("middle", ctr + h * a - 0.4 * h * c)):
        goal[dg] = tg + 0.01 * (ctr - tg) / max(1e-9, float(np.linalg.norm(ctr - tg)))
    joints = dict(thumb=THUMB_R, index=FINGER_R["index"], middle=FINGER_R["middle"])
    act = {}
    for u in range(int(m.nu)):
        if int(m.actuator_trntype[u]) == 0:
            act[m.joint(int(m.actuator_trnid[u][0])).name.split(":")[-1]] = u
    gain = getattr(w.env, "_native_actuator_gain", None)
    gain = np.array(m.actuator_gainprm if gain is None else gain)
    tmax = {j: float(gain[act[j]][0] * m.actuator_gear[act[j]][0]) for j in ARM_R + DIGITS_R}
    closed = dict(g["closed"])
    p_floor, R_des = np.array(g["p_h"], dtype=np.float64), np.array(g["R_h"], dtype=np.float64)
    kp, kd = GRASP_DRIVEN_HAND_GAINS
    kr, dr = GRASP_DRIVEN_TURN_GAINS
    ramp_rec = GRASP_DRIVEN_RAMP_S / STEP_SECONDS
    with _SmallBox(w):
        flat = _cube_rest_height(w, q0)                      # the free cube's rest on a face, where the drop leaves it
    st = dict(rest=None, corr=0.0)                            # the cube's rest height; the hand's height correction

    def cube_z(world) -> float:
        return float(world.d.qpos[adr + 2])

    def control(world, lift: float, release: bool, correct: bool) -> np.ndarray:
        d = world.d
        ctrl = np.zeros(world.m.nu)
        pH, RH = np.array(d.xpos[hb]), np.array(d.xmat[hb]).reshape(3, 3)
        if correct and st["rest"] is not None:
            err = st["rest"] + lift - cube_z(world)
            st["corr"] = float(np.clip(st["corr"] + GRASP_DRIVEN_LIFT_I * err * float(world.env.dt),
                                       -GRASP_DRIVEN_LIFT_I_MAX, GRASP_DRIVEN_LIFT_I_MAX))
        p_des = p_floor + np.array([0.0, 0.0, lift + st["corr"]])
        jacp, jacr = np.zeros((3, world.m.nv)), np.zeros((3, world.m.nv))
        world.mj.mj_jac(world.m, d, jacp, jacr, pH, hb)
        force = kp * (p_des - pH) - kd * (jacp @ d.qvel)
        turn = 0.5 * sum(np.cross(RH[:, i], R_des[:, i]) for i in range(3))
        torque = kr * turn - dr * (jacr @ d.qvel)
        tau_arm = jacp.T @ force + jacr.T @ torque
        for j in ARM_R:
            dof = world.jd[j]
            tau = float(tau_arm[dof]) + float(d.qfrc_bias[dof]) - float(d.qfrc_passive[dof])
            ctrl[act[j]] = float(np.clip(tau / tmax[j], -1.0, 1.0))
        for dg, (body, local) in tips.items():
            pt = np.array(d.xpos[body]) + np.array(d.xmat[body]).reshape(3, 3) @ np.asarray(local)
            jac = np.zeros((3, world.m.nv))
            world.mj.mj_jac(world.m, d, jac, None, pt, body)
            if release:
                # each tip pushed straight away from the cube's place in the hand: opened joint by joint toward
                # the open hand, the index slid down the falling cube's face and touched it for 7 records (H02's
                # 1 cm carried) in 2 of 10 trials
                e = pt - (pH + RH @ ctr)
                tau = jac.T @ (GRASP_DRIVEN_RELEASE_N * e / max(float(np.linalg.norm(e)), 1e-6))
            else:
                e = pH + RH @ goal[dg] - pt
                tau = jac.T @ (PINCH_PUSH_N * e / max(float(np.linalg.norm(e)), 0.005))
            for j in joints[dg]:
                dof = world.jd[j]
                ctrl[act[j]] = float(np.clip(tau[dof] / tmax[j] - PINCH_DIGIT_DAMPING * float(d.qvel[dof]), -1.0, 1.0))
        for j in FINGER_R["ring"] + FINGER_R["little"]:
            dof, qa = world.jd[j], world.jq[j]
            tau = (2.0 * (closed[j] - float(d.qpos[qa])) - PINCH_DIGIT_DAMPING * float(d.qvel[dof])) * tmax[j]
            ctrl[act[j]] = float(np.clip(tau / tmax[j], -1.0, 1.0))
        return ctrl

    def run(world, seconds: float, lift_at, correct: bool) -> None:
        d, mj = world.d, world.mj
        per = max(1, int(round(float(world.env.dt) / float(world.m.opt.timestep))))
        n = int(round(seconds / float(world.env.dt)))
        for i in range(n):
            d.ctrl[:] = control(world, lift_at(i / max(1, n)), False, correct)
            for _ in range(per):
                mj.mj_step(world.m, d)

    def smooth(x: float) -> float:
        x = min(1.0, max(0.0, x))
        return 0.5 - 0.5 * math.cos(math.pi * x)

    def start(world):
        """The physical floor grip (the construction's _grip_state), GRASP_DRIVEN_SETTLE_S of the controller holding
        it on the floor; for no_retention also the lift and GRASP_DRIVEN_HOLD_S of holding it up."""
        q = _set(_set(q0, world, closed), world, g["arm"])
        q[adr:adr + 3] = g["box_floor"]
        q[adr + 3:adr + 7] = _floor_yaw_quat(world, g)
        d, mj = world.d, world.mj
        d.qpos[:] = q
        d.qvel[:] = 0.0
        d.qacc_warmstart[:] = 0.0
        d.ctrl[:] = 0.0
        mj.mj_forward(world.m, d)
        st.update(rest=None, corr=0.0)
        run(world, GRASP_DRIVEN_SETTLE_S, lambda x: 0.0, False)
        # hold: the cube's height in the floor grip, which R05 reads as its rest; no_retention: the free cube's rest on
        # a face, which the floor records after the drop give R05
        st["rest"] = cube_z(world) if kind == "hold" else flat
        if kind == "no_retention":
            run(world, GRASP_DRIVEN_RAMP_S, lambda x: GRASP_DRIVEN_LIFT_M * smooth(x), False)
            run(world, GRASP_DRIVEN_HOLD_S, lambda x: GRASP_DRIVEN_LIFT_M, True)
        d.ctrl[:] = 0.0
        return np.array(d.qpos)

    def command(t, world):
        if kind == "no_retention":
            return control(world, GRASP_DRIVEN_LIFT_M, t >= GRASP_DRIVEN_RELEASE_AT, t < GRASP_DRIVEN_RELEASE_AT)
        lift = GRASP_DRIVEN_LIFT_M * smooth((t - 8.0) / ramp_rec)
        return control(world, lift, False, t >= 8.0 + ramp_rec)

    if kind not in ("hold", "no_retention"):
        raise ValueError(kind)
    return Driven(start, command, 38 if kind == "hold" else int(GRASP_DRIVEN_DROP_STEPS), _small_box_scene(w, _toy_far(w)))


def case_grasp(w, kind: str):
    """The grasp family (_case_grasp_small's kinds); no_opposition is the palm pressed on the 6 cm box's
    top (case_top_press), the rest grip the GRASP_SIDE_M cube (detector v9)."""
    if kind == "no_opposition":
        return case_top_press(w)
    return _case_grasp_small(w, kind)


@_on_small_box
def _case_grasp_small(w, kind: str):
    """The grasp family on box_light, made the GRASP_SIDE_M cube, standing on the floor beside the right
    hand (THE GRASP REBUILT ON A SMALLER CUBE). Since 2026-09-26 grasp_lift_and_hold and grasp_lift_no_retention
    are driven by the body's motors (case_grasp_driven, THE LIFT HELD BY THE BODY'S OWN MOTORS): the posed "hold"
    and "no_retention" kinds below are no case's any more (their lifted records could not grip). kind:
      no_lift         the grasp closes (the thumb against the index) and holds still on the floor
      no_retention    the grasp closes and lifts the box 2.5 cm, and the box slips out and falls back
                      within the next 50 ms step: contact for 3 steps, never a hold
      hold            the grasp lifts the box 3 cm and holds it still for 0.65 s (no carry: 3 cm of
                      palm path)
      carry_place     lift 3 cm, carry 8 cm, set down, open the hand, withdraw; the box stays"""
    g = _grasp(w)
    q0 = g["q0"]
    frames, extra = _grasp_frames(w, g, t_close=15.0, close_steps=(1.0 if kind == "no_retention" else 2.0))
    quat_floor = _floor_yaw_quat(w, g)
    k = g["keys"]
    n_steps = 34
    lifted = (20.0, 23.0)                                     # the hand lifts the cube (THE GRIP IS HELD...)
    lowered = None                                            # ...and sets it down
    on, off = 17.0, float("inf")                              # the digits' motors hold the grip
    # no_lift belongs to this one chain: 7163c88f7 put the defaults above between its `if` and the `elif`
    # of no_retention, so no_lift fell through to the ValueError (the candidate's dry run, 2026-09-26)
    if kind == "no_lift":
        pass                                                  # the grip closes and holds on the floor
    elif kind == "no_retention":
        frames = frames[:4]                                   # no extra closing: lift at once
        frames.append((17.0, dict(k["lift"], **_air_digits(g, g["closed"]))))
        n_steps = 26
        lifted, on, off = (16.0, 17.0), 16.0, 17.0 + 1e-9
    elif kind == "hold":
        frames += [(20.0, dict(extra)), (23.0, dict(k["lift"], **_air_digits(g, {j: extra[j] for j in DIGITS_R})))]
        n_steps = 38
    elif kind in ("carry_place", "release_withdraw"):
        # carry_place: the hand opens, stays still 150 ms, then withdraws; release_withdraw: the
        # same, but the hand starts withdrawing (20 cm/s) in the step its digits finish opening
        dig = {j: extra[j] for j in DIGITS_R}
        dig_air = _air_digits(g, dig)
        frames += [(20.0, dict(extra)), (23.0, dict(k["lift"], **dig_air)), (31.0, dict(k["carry"], **dig_air)),
                   (34.0, dict(k["down"], **dig)), (36.0, dict(k["down"], **dig)),
                   (39.0, dict(k["down"], **g["open"]))]
        lowered, off = (31.0, 34.0), 36.0
        if kind == "carry_place":
            frames += [(42.0, dict(k["down"], **g["open"])), (45.0, dict(k["away"], **g["open"]))]
        else:
            frames += [(42.0, dict(k["away"], **g["open"]))]
        n_steps = 58
    else:
        raise ValueError(kind)
    at = _timeline(w, frames)
    down_pos = g["box_floor"] + 0.08 * g["carry_dir"]
    held = {}

    def air_share(t) -> float:
        """0 with the cube in the floor grip's place in the hand, 1 in the air grip's."""
        s = _ramp(t, lifted[0], lifted[1], 0.0, 1.0)
        return s - _ramp(t, lowered[0], lowered[1], 0.0, 1.0) if lowered else s

    def pose(t):
        q = _set(q0, w, at(t))
        if kind == "no_retention":
            if t <= 16.0:
                return _box_on_floor(g, q, quat=quat_floor)
            if t <= 17.0:
                q = _box_in_hand_blend(w, g, q, air_share(t))
                held["last"] = np.array(q[g["adr"]:g["adr"] + 7])
                return q
            # the box slips out and drops back to the floor within the next step (2.5 cm in 50 ms)
            last = held["last"]
            frac = min(1.0, t - 17.0)
            q[g["adr"]:g["adr"] + 3] = last[:3] + frac * (g["box_floor"] - last[:3])
            q[g["adr"] + 3:g["adr"] + 7] = quat_floor
            return q
        if kind == "no_lift" or t <= 20.0:
            return _box_on_floor(g, q, quat=quat_floor)
        if kind in ("carry_place", "release_withdraw") and t >= 34.0:
            return _box_on_floor(g, q, pos=down_pos, quat=quat_floor)
        return _box_in_hand_blend(w, g, q, air_share(t))
    pose.hold = lambda t: _grip_hold(g, t, air_share(t), on, off) if kind != "no_lift" else _grip_hold(g, t, 0.0, on, off)
    return pose, n_steps, None, _toy_far(w)


@_on_small_box
def case_top_press(w):
    """Contact without opposition: the palm and straight fingers come down flat onto the box's top
    face and the thumb settles on that face too; nothing closes and the box does not move."""
    g = _grasp(w)
    q0 = g["q0"]
    hb = g["hand"]
    adr = g["adr"]
    cache = getattr(w, "_top_press_cache", None)
    if cache is None:
        # the box under the palm, faces aligned with the hand; its top face 0.5 mm into the palm
        half = GRASP_SIDE_M / 2.0                                # the small cube (detector v9, fix 3)
        ctr = np.array([g["ctr"][0], -(0.0062 + half - 0.0005), g["ctr"][2]])   # the grip's place, the palm 0.5 mm into the top
        R_h = g["R_h"]
        box_floor = g["box_floor"]
        qa, ep, er = w.ik_pose(_set(q0, w, g["arm"]), hb, box_floor - R_h @ ctr, R_h, ARM_R, iters=500)
        if ep > 0.002 or er > 2.0:
            raise RuntimeError("top press: no arm pose (%.4f m, %.2f deg)" % (ep, er))
        q = _set(qa, w, g["open"])
        for j in ("right_ff_knuckle", "right_mf_knuckle", "right_rf_knuckle", "right_lf_knuckle"):
            q[w.jq[j]] = math.radians(-6.0)                  # the straight fingers rest on the face
        q[adr:adr + 3] = box_floor
        q[adr + 3:adr + 7] = _mat2quat(w, R_h)
        _, R = w.frame(q, hb)
        bt, lt = w.tip_local("R", "thumb")
        top = box_floor + np.array([0.0, 0.0, half - 0.004])
        thumb_goal = top + R @ np.array([0.025, 0.0, 0.0])    # radial of the palm: beside the 2.5 cm cube, off it
        qs = _settle(w, q, [(bt, lt, thumb_goal)], THUMB_R, force=0.8)
        arm = {j: float(qa[w.jq[j]]) for j in ARM_R}
        pressed = dict(arm, **{j: float(qs[w.jq[j]]) for j in DIGITS_R})
        above_q = w.ik_pose(qa, hb, box_floor - R_h @ ctr + np.array([0, 0, 0.025]), R_h, ARM_R, iters=400)[0]
        above = dict({j: float(above_q[w.jq[j]]) for j in ARM_R}, **{j: float(qs[w.jq[j]]) for j in DIGITS_R})
        cache = w._top_press_cache = dict(pressed=pressed, above=above, touching=_box_contacts(w, qs),
                                          opposition_deg=_opposition_deg(w, qs), quat=_mat2quat(w, R_h))
    rest = {j: float(q0[w.jq[j]]) for j in ARM_R + DIGITS_R}
    at = _timeline(w, [(4.0, rest), (12.0, cache["above"]), (15.0, cache["pressed"])])

    def pose(t):
        return _box_on_floor(g, _set(q0, w, at(t)), quat=cache["quat"])
    return pose, 30, None, _toy_far(w)


def case_toy_touch(w, world_moves: bool):
    """Accidental object motion (addendum section 27) and its twin. The toy's pad (a 7 x 7 x
    1.6 cm mocap box the caregiver's script moves; the world moves it, never the infant) faces the
    right hand's fingertips head-on. world_moves: the hand stays still, fingers at rest, and the
    toy comes in along the finger direction at 10 cm/s until it presses the middle fingertip (the
    longest) 1.5 mm - the index tip, 4.7 mm shorter, stays clear. Otherwise the toy stays at that
    final place and the arm carries the hand 5 cm toward it at 10 cm/s. The same touch, made by the
    world or by the hand: HX3 (one fingertip touching) must be marked world_moved in the first and
    count as the infant's own in the second."""
    m = w.m
    hb = m.body("right_hand").id
    k = w.mocap_id(TOY)
    q0 = _clear_scene(w, w.base())
    pad = [gg for gg in range(m.ngeom) if m.geom(gg).name == "caregiver_pad"][0]
    p0, R0 = w.frame(q0, hb)
    f = R0[:, 2]                                           # the finger direction
    travel = 0.05
    # the touch happens 5 cm along the fingers from the reset pose (detector v9): until v9 the hand's twin
    # started 5 cm BACK, a pose that put the upper arm into the chest (1,670 N in its first record)
    q_near, ep, er = w.ik_pose(q0, hb, p0 + travel * f, R0, ARM_R, iters=500)
    if ep > 0.002 or er > 2.0:
        raise RuntimeError("toy touch: the hand cannot move 5 cm along its fingers (%.4f m, %.2f deg)" % (ep, er))
    bm, lm = w.tip_local("R", "middle")
    tip = w.point(q_near, bm, lm)
    half_thick = float(m.geom_size[pad][2])
    R_pad = np.stack([R0[:, 0], np.cross(f, R0[:, 0]), f], axis=1)       # the pad's thin axis along f
    R_body = R_pad @ _quat2mat(w, m.geom_quat[pad]).T
    pad_final = tip + (half_thick - 0.0015) * f          # its face 1.5 mm past the middle tip
    body_final = pad_final - R_body @ np.asarray(m.geom_pos[pad])
    quat = _mat2quat(w, R_body)
    speed = 0.10 * 0.05                                  # 10 cm/s in metres per 50 ms step
    t0, t1 = 6.0, 6.0 + travel / speed                   # 10 steps
    if world_moves:
        def scene(t, world):
            s = _ramp(t, t0, t1, travel, 0.0)
            world.d.mocap_pos[k] = body_final + s * f
            world.d.mocap_quat[k] = quat

        def pose(t):
            return np.array(q_near)
        return pose, int(t1) + 12, None, scene
    # the hand's twin: the arm carries the hand 5 cm along its fingers from the reset pose to the touch
    far = {j: float(q0[w.jq[j]]) for j in ARM_R}
    near = {j: float(q_near[w.jq[j]]) for j in ARM_R}
    at = _timeline(w, [(t0, far), (t1, near)])

    def scene(t, world):
        world.d.mocap_pos[k] = body_final
        world.d.mocap_quat[k] = quat

    def pose(t):
        return _set(q0, w, at(t))
    return pose, int(t1) + 12, None, scene


@_on_small_box
def case_push_resting_hand(w):
    """The limp baseline's 'hold' replayed (babble_rates.json: all 8 limp bodies made one H02 and
    one R06 of about 48.7 s, a resting hand lying on the light box while the arm slid 8 cm and the
    box 2.8 cm with it). Here the hand comes down flat on the box's top with the fingers relaxed
    (index, middle and ring hanging over the far edge, 80 degrees at the last joint: closure 20
    degrees, never changing) and the arm then drifts 6 cm along the floor over 2 s, the box pushed
    along under the palm. No closure act, no opposition, no lift: a push (O01), not a hold (H02) or a
    move while holding (R06)."""
    g = _grasp(w)
    case_top_press(w)                                    # builds the pressed pose (cached)
    cache = w._top_press_cache
    q0 = g["q0"]
    hb = g["hand"]
    relaxed = dict(cache["pressed"])
    for dg in ("ff", "mf", "rf"):
        relaxed["right_%s_distal" % dg] = 0.0      # straight (80 degrees on the 6 cm box: on the 2.5 cm cube bent
    relaxed.update({j: g["open"][j] for j in THUMB_R})  # fingers and a thumb beside it touched opposite faces, R04)
    qa = _set(q0, w, relaxed)
    p, R = w.frame(qa, hb)
    drift = 0.06 * g["carry_dir"]
    q_end, ep, er = w.ik_pose(qa, hb, p + drift, R, ARM_R, iters=400)
    if ep > 0.002 or er > 2.0:
        raise RuntimeError("push: the hand cannot drift 6 cm (%.4f m, %.2f deg)" % (ep, er))
    end = {j: float(q_end[w.jq[j]]) for j in ARM_R}
    above = dict(cache["above"], **{j: relaxed[j] for j in DIGITS_R})
    rest = {j: float(q0[w.jq[j]]) for j in ARM_R + DIGITS_R}
    at = _timeline(w, [(4.0, rest), (12.0, above), (15.0, relaxed), (18.0, relaxed), (58.0, dict(relaxed, **end))])
    start_box = g["box_floor"]

    def pose(t):
        q = _set(q0, w, at(t))
        frac = _ramp(t, 18.0, 58.0, 0.0, 1.0)
        return _box_on_floor(g, q, pos=start_box + frac * drift, quat=cache["quat"])
    return pose, 64, None, _toy_far(w)


# ---- the palm-box cases rebuilt on the settled grasp (detector v9; OBSERVER_DESIGN.md section 9, "the
# kept palm-box cases hold the box inside the hand"). Until v9 h01_touch_then_close,
# h01_close_400ms_late, h02_h03_hold_then_release and h03_object_escapes pressed the box into the
# palm (_box_on_palm): palm 10.8 mm, forearm 23.5 mm and thumb base 13.3 mm inside it, no finger
# touching it, 633-681 N recorded. Now the box is held by the digits the grasp family settles by
# physics (_grasp: MuJoCo's contacts stop them on the box's faces).
def _first_touch(w, g, at, box_q, t0: float, t1: float, parts=("right_th",)) -> float:
    """The first frame time in [t0, t1] at which one of these parts (body name prefixes) touches the
    box posed by box_q(q, t)."""
    for n in range(int(round(t0 * w.reps)), int(round(t1 * w.reps)) + 1):
        t = n / float(w.reps)
        q = box_q(_set(g["q0"], w, at(t)), t)
        if any(b.startswith(parts) for b in _box_contacts(w, q)):
            return t
    raise RuntimeError("the grasp: no touch of %s between t %.1f and %.1f" % (parts, t0, t1))


@_on_small_box
def case_h01_on_grasp(w, late: bool):
    """Whole-hand close after touch (H01), on the settled grasp: the hand comes down around the box
    standing on the floor with its digits open (t 4-15), the thumb closes onto its face (t 15-16),
    and the four fingers close onto theirs over 100 ms starting in the record after the thumb's
    first touch (late: 400 ms after it). H01 needs the fingers' mean flexion to rise 10 degrees
    within 50-300 ms of a touch: the prompt close is one, the late close is not (the catalog's
    window; BOUNDARY). The fingers' own touches come as they stop on the box, with no closing left
    after them."""
    g = _grasp(w)
    q0 = g["q0"]
    quat_floor = _floor_yaw_quat(w, g)
    rest = {j: float(q0[w.jq[j]]) for j in ARM_R + DIGITS_R}
    above = dict(g["keys"]["above"], **g["open"])
    grasp = dict(g["arm"], **g["open"])
    thumb = dict(grasp, **{j: g["closed"][j] for j in THUMB_R})
    box_q = lambda q, t: _box_on_floor(g, q, quat=quat_floor)
    t_touch = _first_touch(w, g, _timeline(w, [(4.0, rest), (12.0, above), (15.0, grasp), (16.0, thumb)]),
                           box_q, 15.0, 16.0)
    t_close = math.ceil(t_touch) + (8.0 if late else 0.0)          # the record after the touch's record
    closed = dict(thumb, **{j: g["closed"][j] for dg in FINGER_R for j in FINGER_R[dg]})
    at = _timeline(w, [(4.0, rest), (12.0, above), (15.0, grasp), (16.0, thumb), (t_close, thumb),
                       (t_close + 2.0, closed)])

    def pose(t):
        return box_q(_set(q0, w, at(t)), t)
    return pose, int(t_close) + 12, None, _toy_far(w)


@_on_small_box
def case_hold_release_on_grasp(w, kind: str):
    """The hold, release and escape cases on the settled grasp. The grasp closes on the box on the
    floor (t 15-19), lifts it 3 cm (t 20-23) and holds it still in the air (t 23-26). kind:
      release        the digits open (t 26-29); once no thumb or finger touches the box it falls
                     freely, 3 cm in 78 ms, and stands on the floor where it was picked up: a
                     controlled release into the air (H03, detector v9) and a drop, not a placement
                     (R08 absent)
      yank_closed    the digits stay closed and the box is pulled out of the grip at 1.2 m/s: no
                     release (H03 absent). No hold forms either: a box cannot leave a closed grip
                     without sliding past the pads while touching them, and it is 6 cm from the palm
                     within the contact's last record, over H02's 2 cm line (the kept palm-box case
                     made no hold for the same reason)
      yank_opening   the digits open as in release, and from the end of the record in which they
                     let go the box is pulled away at 1.2 m/s, in the first of the grip's open
                     directions that takes it 6 cm without touching the hand: the hold is whole and
                     the hand opened, but the box leaves faster than a release lets it (H03 absent:
                     an escape). The pull waits for the record's end because the recorder keeps each
                     record's last frame while it gathers contacts over the whole burst: pulled from
                     mid-record the box was about 5 cm from the palm in its last contact record, which
                     H02 reads as sliding in the grip (its 2 cm line) - no hold"""
    g = _grasp(w)
    q0 = g["q0"]
    quat_floor = _floor_yaw_quat(w, g)
    frames, extra = _grasp_frames(w, g, t_close=15.0, close_steps=2.0)
    dig = {j: extra[j] for j in DIGITS_R}
    lift = dict(g["keys"]["lift"], **_air_digits(g, dig))     # the air grip (THE GRIP IS HELD BY THE HAND'S MOTORS)
    frames += [(20.0, dict(extra)), (23.0, lift), (26.0, lift)]
    if kind in ("release", "yank_opening"):
        frames.append((29.0, dict(g["keys"]["lift"], **g["open"])))
    elif kind != "yank_closed":
        raise ValueError(kind)
    at = _timeline(w, frames)
    step_s = float(w.env.dt) * w.reps
    if kind == "yank_closed":
        t_free = 26.0                     # the pull starts; the box slides past the closed digits
    else:
        # the grip lets go at the first frame no thumb or finger touches the box (the forearm resting
        # against the box's side does not hold it)
        t_free = _first_free(w, g, lambda q, t: _box_in_hand_blend(w, g, q, 1.0), at, 26.0, 29.0)
    q_free = _set(q0, w, at(t_free))
    p_free = np.array(_box_in_hand_blend(w, g, q_free, 1.0)[g["adr"]:g["adr"] + 7])
    away = _open_direction(w, g, q_free, p_free, strict=(kind == "yank_opening")) if kind != "release" else None
    t_pull = float(math.ceil(t_free)) if kind == "yank_opening" else t_free

    def pose(t):
        q = _set(q0, w, at(t))
        if t <= 20.0:
            return _box_on_floor(g, q, quat=quat_floor)
        if t <= t_free:
            return _box_in_hand_blend(w, g, q, _ramp(t, 20.0, 23.0, 0.0, 1.0))
        p = np.array(p_free)
        if away is not None:
            p[:3] += away * 1.2 * step_s * min(max(0.0, t - t_pull), 5.0)   # 30 cm in 0.25 s, then held away
            return _put(q, g, p)
        fall = 0.5 * 9.81 * ((t - t_free) * step_s) ** 2
        p[2] = max(float(g["box_floor"][2]), p[2] - fall)
        return _put(q, g, p)
    pose.hold = lambda t: _grip_hold(g, t, _ramp(t, 20.0, 23.0, 0.0, 1.0), 17.0, 26.0)
    return pose, 40, None, _toy_far(w)


def _open_direction(w, g, q, pose7, strict: bool) -> np.ndarray:
    """A horizontal world direction out of the grip for a pulled box: the first of the box's axes in
    the hand (along the pressed faces either way, then away from and toward the thumb) along which
    it moves 6 cm without touching the hand's palm or digits (strict), or the first at all (not
    strict: a closed grip has none)."""
    _, R = w.frame(q, g["hand"])
    cands = [R @ g["R_hb"][:, 2], -(R @ g["R_hb"][:, 2]), R @ g["R_hb"][:, 0], -(R @ g["R_hb"][:, 0])]
    hand_parts = ("right_hand", "right_th", "right_ff", "right_mf", "right_rf", "right_lf")
    for v in cands:
        v = np.array(v, dtype=np.float64)
        v[2] = 0.0
        if np.linalg.norm(v) < 1e-6:
            continue
        v /= np.linalg.norm(v)
        clear = True
        for s in np.linspace(0.005, 0.06, 12):
            p = np.array(pose7)
            p[:3] += v * s
            if any(b.startswith(hand_parts) for b in _box_contacts(w, _put(q, g, p))):
                clear = False
                break
        if clear or not strict:
            return v
    raise RuntimeError("the grasp: no direction takes the released box 6 cm without touching the hand")


def _put(q, g, pose7) -> np.ndarray:
    """q with the grasped box at a 7-number pose (position and quaternion)."""
    q = np.array(q)
    q[g["adr"]:g["adr"] + 7] = pose7
    return q


def _first_free(w, g, box_q, at, t0: float, t1: float) -> float:
    """The first frame time in [t0, t1] at which, with the box posed by box_q(q, t), no thumb or
    finger touches it."""
    for n in range(int(round(t0 * w.reps)), int(round(t1 * w.reps)) + 1):
        t = n / float(w.reps)
        q = box_q(_set(g["q0"], w, at(t)), t)
        if not any(b.startswith(("right_th", "right_ff", "right_mf", "right_rf", "right_lf")) for b in _box_contacts(w, q)):
            return t
    raise RuntimeError("the grasp: the digits still touch the box at t %.1f" % t1)


# ====================================================================== gaze, head, arm into view
# The gaze targets are 6 cm boxes held 12 cm from the eyes: the ring of 8 recorded rays (12 degrees
# out, the fovea's half-angle) then lands 1.9 cm from the face's centre, inside its 3 cm half-width,
# so the box fills the whole foveal bundle (the v6 fixations are read from the bundle). The kept
# positive G02 case's box at 29 cm catches the centre ray only: its ring lands on the ceiling (5)
# and the toy (3) in every fixated record (2026-09-24).
GAZE_DIST = 0.12
EYE_JOINTS = (("left_eye_horizontal", "left_eye_vertical"), ("right_eye_horizontal", "right_eye_vertical"))


def _eye_frame(w, q):
    """(midpoint of the eyes, their mean viewing axis, the infant's right across the eyes, up
    toward the crown) at pose q."""
    m, d = w.m, w.d
    w.fk(q)
    w.mj.mj_camlight(m, d)
    cams = [m.camera(c).id for c in ("eye_left", "eye_right")]
    mid = np.mean([d.cam_xpos[c] for c in cams], axis=0)
    fwd = np.mean([-d.cam_xmat[c].reshape(3, 3)[:, 2] for c in cams], axis=0)
    fwd /= np.linalg.norm(fwd)
    right = np.asarray(d.cam_xpos[cams[1]]) - np.asarray(d.cam_xpos[cams[0]])
    right -= np.dot(right, fwd) * fwd
    right /= np.linalg.norm(right)
    return np.array(mid), fwd, right, np.cross(right, fwd)


def _gaze_point(frame, deg_right: float, deg_up: float = 0.0, dist: float = GAZE_DIST) -> np.ndarray:
    mid, fwd, right, up = frame
    a, e = math.radians(deg_right), math.radians(deg_up)
    v = math.cos(e) * (math.cos(a) * fwd + math.sin(a) * right) + math.sin(e) * up
    return mid + dist * v


def _place_box(w, q, name: str, pos, face_toward=None) -> np.ndarray:
    """q with a box at pos, one face turned toward a point (the eyes) when given."""
    adr = w.free_adr(name)
    q = np.array(q)
    q[adr:adr + 3] = pos
    if face_toward is None:
        q[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
    else:
        x = np.asarray(face_toward) - np.asarray(pos)
        x /= np.linalg.norm(x)
        y = np.cross(np.array([0.0, 0.0, 1.0]), x)
        y = y / np.linalg.norm(y) if np.linalg.norm(y) > 1e-6 else np.array([0.0, 1.0, 0.0])
        q[adr + 3:adr + 7] = _mat2quat(w, np.stack([x, y, np.cross(x, y)], axis=1))
    return q


def _gaze_scene_q(w) -> np.ndarray:
    """The base pose with the three boxes far away on the floor (a case places what it needs)."""
    q = _clear_scene(w, w.base())
    a = w.free_adr(GRASP_BOX)
    q[a:a + 3] = (1.8, 2.0, BOX_HALF - 0.001)
    q[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
    return q


def _add_eyes(w, q, dh: float, dv: float = 0.0) -> np.ndarray:
    """Both eyes turned by dh (horizontal) and dv (vertical) degrees from where q has them."""
    q = np.array(q)
    for hj, vj in EYE_JOINTS:
        q[w.jq[hj]] += math.radians(dh)
        q[w.jq[vj]] += math.radians(dv)
    return q


def _head_sign(w, q0) -> float:
    """+1 when a positive head_swivel turns the face toward the infant's right, else -1."""
    _, fwd0, right0, _ = _eye_frame(w, q0)
    q = np.array(q0)
    q[w.jq["head_swivel"]] += math.radians(10.0)
    _, fwd1, _, _ = _eye_frame(w, q)
    return 1.0 if float(np.dot(fwd1 - fwd0, right0)) > 0 else -1.0


def _eye_h_sign(w, q0) -> float:
    """+1 when a positive horizontal eye angle turns the gaze toward the infant's right."""
    _, fwd0, right0, _ = _eye_frame(w, q0)
    _, fwd1, _, _ = _eye_frame(w, _add_eyes(w, q0, 5.0))
    return 1.0 if float(np.dot(fwd1 - fwd0, right0)) > 0 else -1.0


class _WorldSaccades:
    """Stands in for the world's saccade generator (gaze_reflex.py): the recorder marks a record
    gaze_saccade when n_saccades changes during its burst."""
    n_saccades = 0


class _WorldReflex:
    """Stands in for the world's peripheral-motion reflex (gaze_reflex.py): the recorder marks a
    record gaze_reflex when `active` is true in any frame of its burst."""
    active = False


def case_g01(w, kind: str):
    """G01, independent eye movement (catalog: eye orientation changes while the head remains
    within tolerance). Both eyes turn 12 degrees to the right within 20 ms (one saccade) at t = 6.2,
    and hold. kind:
      saccade        the head still: the infant's own eye movement
      head_1deg      the head turns 1 degree in that same step (within the 2-degree tolerance)
      head_4deg      the head turns 4 degrees in that step (past it)
      with_head      the head turns 24 degrees over 4 steps while the eyes counter-rotate to keep a
                     far point in view (6 degrees a step each): the eyes move, never independently
      world_saccade  the world's saccade generator made the jump: found, marked, never counted
      world_reflex   the world's motion reflex steered the eyes during the jump: the same"""
    q0 = _gaze_scene_q(w)
    s_eye = _eye_h_sign(w, q0)
    s_head = _head_sign(w, q0)
    jq_head = w.jq["head_swivel"]
    stub = _WorldSaccades()
    reflex = _WorldReflex()
    if kind == "with_head":
        far = _gaze_point(_eye_frame(w, q0), 0.0, 0.0, dist=1.0)

        root_adr = w.free_adr(w.m.body(int(w.env._root_body_id)).name)

        def pose(t):
            q = np.array(q0)
            q[jq_head] += s_head * math.radians(_ramp(t, 6.0, 10.0, 0.0, 24.0))
            return _rest_on_floor(w, q, root_adr)       # the turning head neither sinks into the floor (136 N) nor floats

        def eyes(t, q):
            return _aim_eyes(w, q, far)
        return pose, 16, eyes, _toy_far(w)
    head_deg = {"head_1deg": 1.0, "head_4deg": 4.0}.get(kind, 0.0)

    def pose(t):
        q = _add_eyes(w, q0, s_eye * _ramp(t, 6.2, 6.4, 0.0, 12.0))
        q[jq_head] += s_head * math.radians(_ramp(t, 6.0, 7.0, 0.0, head_deg))
        return q
    toy = _toy_far(w)

    def scene(t, world):
        toy(t, world)
        if kind == "world_saccade":
            world.env.saccades = stub
            if abs(t - 6.4) < 1e-9:
                stub.n_saccades += 1                 # during step 6's burst, after its first frame
        elif kind == "world_reflex":
            world.env.reflex = reflex
            reflex.active = 6.1 < t < 6.5            # steering in the frames the eyes move
    return pose, 14, None, scene


def case_g03(w, kind: str, dist: float = GAZE_DIST):
    """G03, return fixation (catalog: after looking away, gaze returns to the same object or
    region without an external cue). Box A (box_light) 20 degrees right and box B (box_medium) 20
    degrees left, both `dist` from the eyes (12 cm; 30 cm is reaching distance). Neither box is in
    the foveal bundle (12 degrees) while the eyes are on the other, 40 degrees away. kind:
      return      A (t 3-12), B (12-28), A again (28-40)
      cued        the same, but A moves 4 cm at 10 cm/s while the eyes are on B (an external cue)
      no_return   A (3-12), then B to the end
      short_away  A (3-12), B for 0.5 s (12-22), A again (22-34): a half-second look away"""
    q0 = _gaze_scene_q(w)
    fr = _eye_frame(w, q0)
    a_pos, b_pos = _gaze_point(fr, 20.0, dist=dist), _gaze_point(fr, -20.0, dist=dist)
    a_moved = a_pos + 0.04 * fr[3]                      # 4 cm toward the crown
    back = {"return": 28.0, "cued": 28.0, "short_away": 22.0}.get(kind)
    n = 40 if kind != "short_away" else 34

    def box_a(t):
        return a_pos + _ramp(t, 16.0, 24.0, 0.0, 1.0) * (a_moved - a_pos) if kind == "cued" else a_pos

    def pose(t):
        q = _place_box(w, q0, GRASP_BOX, box_a(t), fr[0])
        return _place_box(w, q, "box_medium", b_pos, fr[0])

    def eyes(t, q):
        if t < 3.0:
            return q
        if t < 12.0 or (back is not None and t >= back):
            return _aim_eyes(w, q, box_a(t))
        return _aim_eyes(w, q, b_pos)
    return pose, n, eyes, _toy_far(w)


def case_g05(w, kind: str):
    """G05, eye-head coordination (catalog: head motion reduces sustained eye eccentricity while
    gaze remains on target). A box 12 cm away, 22 degrees to the right of the face (16 degrees for
    'away'); the eyes turn to it at t = 3. kind:
      toward     the head turns 16 degrees toward it (t 10-22) while the eyes stay on it
      away       the head turns 12 degrees away from it while the eyes stay on it
      no_follow  the head turns 16 degrees toward it but the eyes keep their angle in the head, so
                 the gaze sweeps past the box"""
    q0 = _gaze_scene_q(w)
    fr = _eye_frame(w, q0)
    s_head = _head_sign(w, q0)
    box = _gaze_point(fr, 16.0 if kind == "away" else 22.0)
    turn = {"toward": 16.0, "away": -12.0, "no_follow": 16.0}[kind]
    held = {}

    def pose(t):
        q = _place_box(w, q0, GRASP_BOX, box, fr[0])
        q[w.jq["head_swivel"]] += s_head * math.radians(_ramp(t, 10.0, 22.0, 0.0, turn))
        return q

    def eyes(t, q):
        if t < 3.0:
            return q
        if kind == "no_follow" and t > 10.0:
            q = np.array(q)
            for hj, vj in EYE_JOINTS:
                q[w.jq[hj]], q[w.jq[vj]] = held[hj], held[vj]
            return q
        q = _aim_eyes(w, q, box)
        for hj, vj in EYE_JOINTS:
            held[hj], held[vj] = float(q[w.jq[hj]]), float(q[w.jq[vj]])
        return q
    return pose, 44, eyes, _toy_far(w)


def case_r03(w, kind: str, dist: float = 0.28):
    """R03, visually guided reach (catalog: object is fixated before or during reach and
    hand-object distance then decreases). box_light at reaching distance, 28 cm from the eyes, 30
    degrees toward the right hand and 10 below the eyes' axis; the right palm moves from rest to
    9 cm short of it (t 8-20, 0.6 s). kind:
      look_then_reach  the eyes are on the box from t = 3
      look_elsewhere   the eyes are on box_medium, as far away 25 degrees to the left
      look_after       the eyes reach the box 4 steps after the hand stops (t = 24)"""
    m = w.m
    q0 = _gaze_scene_q(w)
    fr = _eye_frame(w, q0)
    box = _gaze_point(fr, 30.0, -10.0, dist=dist)
    other = _gaze_point(fr, -25.0, dist=dist)
    hb = m.body("right_hand").id
    palm0, _ = w.frame(q0, hb)
    goal = box + 0.09 * (palm0 - box) / np.linalg.norm(palm0 - box)
    q_end = w.ik(q0, hb, np.zeros(3), goal, ARM_R[:4], iters=400)
    at = _timeline(w, [(8.0, {j: float(q0[w.jq[j]]) for j in ARM_R[:4]}),
                       (20.0, {j: float(q_end[w.jq[j]]) for j in ARM_R[:4]})])

    def pose(t):
        q = _set(q0, w, at(t))
        q = _place_box(w, q, GRASP_BOX, box, fr[0])
        return _place_box(w, q, "box_medium", other, fr[0]) if kind == "look_elsewhere" else q

    def eyes(t, q):
        start = 24.0 if kind == "look_after" else 3.0
        if t < start:
            return q
        return _aim_eyes(w, q, other if kind == "look_elsewhere" else box)
    return pose, 32, eyes, _toy_far(w)


def case_b06(w, kind: str):
    """B06, arm into own view (catalog: arm or hand enters the foveal/peripheral frustum
    following self-generated movement). At rest the hands lie 90 degrees outside the eyes' axis.
    kind:
      arm_raises    the right arm lifts the palm to 15 degrees right of the axis, 22 cm away, in
                    0.45 s (t 6-15): the hand enters the view by its own movement
      head_4deg     the same while the head turns 4 degrees over those steps (within the 10-degree
                    tolerance of the 250 ms before the entry)
      head_turns    the hand held still 45 degrees right of the axis; the head turns 30 degrees
                    toward it (t 6-14): the view comes to the hand
      eyes_turn     the same still hand; the eyes turn 25 degrees toward it (t 6-8)
    and for the LEFT hand (detector v9, defect 8: the page reports B06:L beyond babble without a
    scripted case), mirrored: left_arm_raises (the left palm to 15 degrees left of the axis) and
    left_head_turns (the still left hand 45 degrees left; the head turns 30 degrees toward it)."""
    m = w.m
    q0 = _gaze_scene_q(w)
    fr = _eye_frame(w, q0)
    left = kind.startswith("left_")
    kind = kind[len("left_"):] if left else kind
    hb = m.body("left_hand" if left else "right_hand").id
    arm = ARM_L if left else ARM_R
    toward = -1.0 if left else 1.0                        # degrees to the infant's right, per side
    s_head = _head_sign(w, q0) * toward
    s_eye = _eye_h_sign(w, q0) * toward
    if kind in ("arm_raises", "head_4deg"):
        goal = _gaze_point(fr, 15.0 * toward, 0.0, dist=0.22)
        q_end = w.ik(q0, hb, np.zeros(3), goal, arm[:4], iters=500)
        at = _timeline(w, [(6.0, {j: float(q0[w.jq[j]]) for j in arm[:4]}),
                           (15.0, {j: float(q_end[w.jq[j]]) for j in arm[:4]})])
        head = 4.0 if kind == "head_4deg" else 0.0

        def pose(t):
            q = _set(q0, w, at(t))
            q[w.jq["head_swivel"]] += s_head * math.radians(_ramp(t, 6.0, 15.0, 0.0, head))
            return q
        return pose, 22, None, _toy_far(w)
    goal = _gaze_point(fr, 45.0 * toward, 0.0, dist=0.22)
    q_hand = w.ik(q0, hb, np.zeros(3), goal, arm[:4], iters=500)

    root_adr = w.free_adr(m.body(int(w.env._root_body_id)).name)

    def pose(t):
        q = np.array(q_hand)
        if kind == "head_turns":
            q[w.jq["head_swivel"]] += s_head * math.radians(_ramp(t, 6.0, 14.0, 0.0, 30.0))
            return _rest_on_floor(w, q, root_adr)        # the turning head neither sinks into the floor (190 N) nor floats
        elif kind == "eyes_turn":
            q = _add_eyes(w, q, s_eye * _ramp(t, 6.0, 8.0, 0.0, 25.0))
        return q
    return pose, 22, None, _toy_far(w)


# ====================================================================== the hand's own rows
def case_h05_more(w, kind: str):
    """H05, individual finger isolation (catalog: one digit moves substantially while non-target
    digits remain near baseline), beyond the right index. kind:
      left_index     the LEFT index bends 40 degrees, every other digit still
      thumb          the right thumb bends 40 degrees (its two flexion hinges), the fingers still
      straighten     all five right digits start bent 40 degrees; the index straightens to 0
      two_fingers    index and middle bend 40 degrees together
    and the sub-rows the page reports beyond babble without a scripted case (detector v9, defect 8):
      left_little    the LEFT little finger bends 40 degrees alone
      left_thumb     the LEFT thumb bends 40 degrees alone
      right_little   the right little finger bends 40 degrees alone
      little_ring    the right little and ring fingers bend 40 degrees together: each moves with a
                     neighbour, so neither is isolated (the others' mean change 10 degrees, over the
                     5-degree line)"""
    def pose(t):
        q = w.base()
        a = _ramp(t, 4, 8, 0.0, 1.0)
        one = {"left_little": ("L", "little"), "left_thumb": ("L", "thumb"), "right_little": ("R", "little")}
        if kind in one:
            w.set_curl(q, one[kind][0], one[kind][1], 40.0 * a)
        elif kind == "little_ring":
            w.set_curl(q, "R", "little", 40.0 * a)
            w.set_curl(q, "R", "ring", 40.0 * a)
        elif kind == "left_index":
            w.set_curl(q, "L", "index", 40.0 * a)
        elif kind == "thumb":
            w.set_curl(q, "R", "thumb", 40.0 * a)
        elif kind == "straighten":
            for dg in DIGITS:
                w.set_curl(q, "R", dg, 40.0)
            w.set_curl(q, "R", "index", 40.0 * (1.0 - a))
        elif kind == "two_fingers":
            w.set_curl(q, "R", "index", 40.0 * a)
            w.set_curl(q, "R", "middle", 40.0 * a)
        return q
    return pose, 16


def case_hx2(w, cup_deg: float = 30.0, whole_hand: bool = False):
    """HX2, shape the palm (catalog: the little-finger metacarpal's cupping joint changes by at
    least 15 degrees): the right cupping hinge goes to cup_deg (t 4-8) and back (t 12-16); or,
    whole_hand, all four fingers curl 40 degrees with the cupping hinge still."""
    j = w.jq["right_lf_meta"]
    hi = w.jrange["right_lf_meta"][1]

    def pose(t):
        q = w.base()
        if whole_hand:
            for dg in FINGERS:
                w.set_curl(q, "R", dg, _ramp(t, 4, 8, 0, 40))
            return q
        a = _ramp(t, 4, 8, 0.0, 1.0) - _ramp(t, 12, 16, 0.0, 1.0)
        # capped at the hinge's upper limit only: at rest the hinge sits 0.003 degrees below its
        # lower limit (the reset pose), and clipping that would move the rest pose itself
        q[j] = float(min(hi, q[j] + math.radians(cup_deg) * a))
        return q
    return pose, 20


def case_hx1(w, kind: str):
    """HX1, spread and gather the fingers (catalog since v8: the index finger's sideways base angle
    minus the little finger's changes by at least 15 degrees, rising within 2 s; hand action 'Spread
    and gather'). Positive side angles move each fingertip toward the thumb (measured on this hand,
    all four fingers). kind:
      spread      index +8, ring -8, little -12 degrees (fingertips apart) in 0.2 s, back in 0.2 s
      sweep       all four fingers +8 degrees: they tilt toward the thumb together, the gaps
                  between them unchanged - no spreading
      spread_10   index +4, ring -4, little -4: a fan of 8 degrees
      slow_drift  the 'spread' pose reached over 8 s instead of 0.2 s, as the limp baseline's
                  hands drifted (catalog timescale of an act: 250 ms - 2 s)
    HX1's measure since v8 is the fan, the index finger's side angle minus the little finger's: 'spread'
    changes it by 8 + 12 = 20 degrees, 'spread_10' by 4 + 4 = 8 (under the 15-degree line), the sweep by
    0. (Until v7 it was the sum of the four |side angles|, which at rest are +0.50, +0.69, +0.49 and
    +0.41 degrees: 26.2 for 'spread', 10.2 for 'spread_10' and 32 for the sweep - the sweep read as a
    spread.)"""
    names = {"index": "right_ff_side", "middle": "right_mf_side", "ring": "right_rf_side", "little": "right_lf_side"}
    target = {"spread": dict(index=8, ring=-8, little=-12), "sweep": dict(index=8, middle=8, ring=8, little=8),
              "spread_10": dict(index=4, ring=-4, little=-4), "slow_drift": dict(index=8, ring=-8, little=-12)}[kind]
    slow = kind == "slow_drift"

    def pose(t):
        q = w.base()
        a = _ramp(t, 4, 164 if slow else 8, 0.0, 1.0)
        if not slow:
            a -= _ramp(t, 12, 16, 0.0, 1.0)
        for dg, deg in target.items():
            jn = names[dg]
            lo, hi = w.jrange[jn]
            q[w.jq[jn]] = float(np.clip(q[w.jq[jn]] + math.radians(deg) * a, lo, hi))
        return q
    return pose, (170 if slow else 20)


def case_settle_fan(w, motors: str):
    """A limp hand's passive fan, replayed (detector v9; OBSERVER_DESIGN.md section 9, "a limp body
    still makes one passive finger fan per body as the arm settles after a reset"). Measured on the
    reset-world limp baseline, body 0, steps 82-124: 4-6 s after the reset the limp right hand's side
    joints swung one after another - the little finger up 8 degrees and back, then the ring, middle
    and index fingers to their 25-degree limits - while the four fingers curled 13 degrees, with no
    contact anywhere on the hand and every motor command zero; v8 counted it as a 30.9-degree spread.
    Here: little +8 (t 4-9) and back -10 (t 9-24), ring +27, middle +29 (both stop at the 25-degree
    limit), index +22 (t 16-24), the curl 0 -> 13 degrees. motors 'off': the settling, no spread;
    'drive': the same movement driven by the joints' own motors, a spread of 31 degrees."""
    names = {"index": "right_ff_side", "middle": "right_mf_side", "ring": "right_rf_side", "little": "right_lf_side"}

    def pose(t):
        q = w.base()
        moves = dict(little=_ramp(t, 4, 9, 0.0, 8.0) + _ramp(t, 9, 24, 0.0, -10.0), ring=_ramp(t, 8, 14, 0.0, 27.0),
                     middle=_ramp(t, 12, 20, 0.0, 29.0), index=_ramp(t, 16, 24, 0.0, 22.0))
        for dg, deg in moves.items():
            jn = names[dg]
            lo, hi = w.jrange[jn]
            q[w.jq[jn]] = float(np.clip(q[w.jq[jn]] + math.radians(deg), lo, hi))
        for dg in FINGERS:
            w.set_curl(q, "R", dg, _ramp(t, 4, 24, 0.0, 13.0))
        return q
    pose.motors = motors
    return pose, 30


def case_support_push(w, kind: str) -> "Driven":
    """P05, push against support (catalog: hand or foot support force rises with limb extension and
    produces body displacement), with real dynamics driven by the body's own motors (detector v9; a
    posed world gives no real support force). A proportional-derivative controller computes the legs'
    motor commands from the body's state each frame (gains PUSH_GAIN per radian, damping a tenth of the
    gain per radian per second; the right knee's PUSH_KNEE_GAIN during the push; commands clipped to the
    motors' -1..1); every other motor gets 0. Gravity, contacts and the whole body are free; the three
    boxes stand far away (in the reset pose the medium box stands by the right hand, and the limp arm
    settling onto it read as a reach, R01, and a touch, R02, in the first dry run). The left leg is held
    lifted (hip flexed 90 degrees, knee 60) throughout, its weight on the pelvis.
      push        the right leg is brought to the hip flexed 40 degrees, the knee 110 and the ankle +30 -
                  the sole planted by the buttock - and held (t 0-40, 2 s); then the hip's motor gives its
                  full extending command, the knee's target straightens from 110 to 10 degrees over 0.35 s
                  and the ankle's motor points the foot down at full command: the foot
                  presses the floor while the knee extends and the pelvis lifts (P05:R:leg)
      in_the_air  the same knee extension and ankle command with the right hip held flexed 90 degrees:
                  the foot never meets the floor, so no support force rises (P05 absent)
    Why this push (2026-09-25, explore_push11-12 in the detector track's record; the recorder's reading
    emulated: per record, each contact pair's largest normal force over the burst's five frames): P05
    now asks the rise to HOLD over 3 records at each end (detectors.PUSH_HOLD_RECORDS). The v9 push of
    4:30 PM (hip 50, knee 110 -> 60 in 0.5 s, the other leg resting) crossed the line with two records
    (53 and 77 N) and failed that. A single supine leg carries at most 35 N of steady push (the pelvis's
    share of the weight), so the force must come from driving the pelvis up fast, with the other leg's
    weight on it: of 432 plants and drives 70 held the rise, and of 243 around the best of them 81; this
    one held in 80 of 100 of its neighbouring settings and in 20 of 20 starts with 1e-5 noise on the start
    pose (the pause's world starts from its own reset pose), its held rise 1.65-2.02 times the line; the
    twin in the air found none."""
    act = {}
    for u in range(int(w.m.nu)):
        if int(w.m.actuator_trntype[u]) == 0:
            act[w.m.joint(int(w.m.actuator_trnid[u][0])).name.split(":")[-1]] = u
    step_s = float(w.env.dt) * w.reps
    if kind not in ("push", "in_the_air"):
        raise ValueError(kind)
    hip0 = PUSH_PLANT_DEG["right_hip1"] if kind == "push" else -90.0

    def targets(sec):
        """(joint -> target degrees, joint -> fixed command, joint -> gain) at a time in seconds."""
        if sec < PUSH_AT_S:
            return dict(PUSH_PLANT_DEG, right_hip1=hip0, **PUSH_OTHER_LEG_DEG), {}, {}
        a = min(1.0, (sec - PUSH_AT_S) / PUSH_RAMP_S)
        knee = PUSH_PLANT_DEG["right_knee"] + a * PUSH_KNEE_SPAN_DEG
        gains = {"right_knee": PUSH_KNEE_GAIN}
        if kind == "push":
            return dict(PUSH_OTHER_LEG_DEG, right_knee=knee), {"right_hip1": 1.0, "right_foot1": -1.0}, gains
        return dict(PUSH_OTHER_LEG_DEG, right_hip1=hip0, right_knee=knee), {"right_foot1": -1.0}, gains

    def command(t, world):
        d = world.d
        ctrl = np.zeros(world.m.nu)
        tgt, fixed, gains = targets(t * step_s)
        for jn, deg in tgt.items():
            g = gains.get(jn, PUSH_GAIN)
            err = math.radians(deg) - float(d.qpos[world.jq[jn]])
            ctrl[act[jn]] = float(np.clip(g * err - 0.1 * g * float(d.qvel[world.jd[jn]]), -1.0, 1.0))
        for jn, c in fixed.items():
            ctrl[act[jn]] = c
        return ctrl
    return Driven(lambda world: _all_boxes_far(world, world.base()), command, 70, _toy_far(w))


PUSH_PLANT_DEG = {"right_hip1": -40.0, "right_knee": -110.0, "right_foot1": 30.0}
PUSH_OTHER_LEG_DEG = {"left_hip1": -90.0, "left_knee": -60.0}
PUSH_AT_S = 2.0
PUSH_RAMP_S = 0.35
PUSH_KNEE_SPAN_DEG = 100.0
PUSH_GAIN = 4.0
PUSH_KNEE_GAIN = 12.0


def _all_boxes_far(w, q) -> np.ndarray:
    """All three boxes far away on the floor (_clear_scene's two and the grasp box beside them)."""
    q = _clear_scene(w, q)
    a = w.free_adr(GRASP_BOX)
    q[a:a + 3] = (2.4, 2.0, BOX_HALF - 0.001)
    q[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
    return q


def _with_motors(built, motors: str):
    """A posed case's builder result with its pose function's motor commands set (ScriptWorld.command):
    'against' records the command opposite to every movement - the hand moved while its own motors resist
    (the motor-work rule, detector v9: such a movement is not the act)."""
    built[0].motors = motors
    return built


def case_h05_index_motors_off(w):
    """h05_index_alone's movement (the right index bends 40 degrees alone) with every motor command
    zero: the finger moved but nothing drove it, so H05 does not count it (detector v9)."""
    pose, steps = case_h05_index(w, 0.0)
    pose.motors = "off"
    return pose, steps


# ====================================================================== the page's rows without a case (defect 8)
# Detector v9, OBSERVER_DESIGN.md section 9: rows the development page reports beyond babble had no
# scripted positive case. From the production report of 2026-09-25 7:27 PM (detectors v8): B03 (both
# hands), B06:L, G02 and G03 on the left upper arm, G05:caregiver, H05 L:little, L:thumb and R:little,
# H06:R:index, HX3:R:little, R02:box (agents/observer_detectors_2026-09-25.md, defect 8). Each gets a
# positive case and a twin that breaks the one link its detector reads.
def _head_point(w, q, side: str, deg_side: float, deg_up: float = 0.0, depth: float = 0.002):
    """A point on the head's sphere, `depth` inside its surface, deg_side degrees from the face's
    forward axis toward the infant's `side` ('R' or 'L') and deg_up toward the crown, at pose q, with
    the outward direction there (world frame)."""
    m = w.m
    w.fk(q)
    hb = m.body("head").id
    R = w.d.xmat[hb].reshape(3, 3)
    p = np.asarray(w.d.xpos[hb])
    g = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == hb][0]
    c_local = np.asarray(m.geom_pos[g])
    r = float(m.geom_size[g][0])
    le, re = m.body("left_eye").id, m.body("right_eye").id
    f = R.T @ (0.5 * (w.d.xpos[le] + w.d.xpos[re]) - p) - c_local
    f /= np.linalg.norm(f)
    lat = R.T @ (w.d.xpos[le] - w.d.xpos[re])            # toward the infant's left
    lat -= np.dot(lat, f) * f
    lat /= np.linalg.norm(lat)
    up = np.cross(f, lat)
    if np.dot(up, R.T @ (w.d.xpos[m.body("upper_body").id] - p)) > 0:
        up = -up
    toward = -lat if side == "R" else lat
    a, e = math.radians(deg_side), math.radians(deg_up)
    v = math.cos(e) * (math.cos(a) * f + math.sin(a) * toward) + math.sin(e) * up
    return p + R @ (c_local + (r - depth) * v), R @ v


B03_CHEEK_DEG = 35.0      # from the face's forward axis toward the hand's side: the face zone (60-degree
                          # cone), 49 degrees from the mouth point (the mouth zone is 2.5 cm around it)


B03_PULL_N = 4.0           # the pull drawing the fingertip to the cheek (gravity off while it does)
B03_SETTLE_STEPS = 600     # 3 s of 5 ms physics steps
B03_ARRIVE_MPS = 0.15      # the replayed palm's speed over the 250 ms before the touch: 3 times B03's line
B03_CREEP_MPS = 0.01       # the creeping twin's speed over its last 1.5 cm


def case_b03(w, kind: str):
    """B03, hand to face (catalog: a hand reaches the face or mouth region; the palm approaches at
    FACE_APPROACH_MPS or more over the 250 ms before the touch - a head turning into a resting hand, or
    a hand drifting onto the face, is not this act). The index fingertip of one hand goes to its own
    side's cheek, a point B03_CHEEK_DEG from the face's forward axis 2 mm inside the head's surface, on
    a PHYSICAL path (as b04_fingertip_to_mouth since v9): a B03_PULL_N pull on the fingertip while
    MuJoCo's contacts keep the arm out of the trunk and the face (gravity off during this construction
    only). The first build posed the arm by inverse kinematics and drove the upper arm into the chest:
    111,813 N recorded (the fast lane's float32 read infinity in 51 frames). The path is replayed
    faster than the physics took, so that the palm arrives at B03_ARRIVE_MPS over the 250 ms before the
    hand's first contact with the head. kind:
      right, left   that path, from t 4
      right_creep   the same path to 1.5 cm (of the palm's path) short of the touch, a 1 s rest, then
                    the rest at B03_CREEP_MPS: under the 5 cm/s line"""
    side = "L" if kind == "left" else "R"
    if kind not in ("right", "left", "right_creep"):
        raise ValueError(kind)
    m = w.m
    arm = ARM_L if side == "L" else ARM_R
    pre = "left_" if side == "L" else "right_"
    hand_bodies = {m.body(b).name for b in range(m.nbody) if m.body(b).name.startswith(pre)
                   and not any(x in m.body(b).name for x in ("arm", "leg", "foot", "toe", "eye"))}
    hb = m.body(pre + "hand").id
    q0 = _gaze_scene_q(w)
    b, loc = w.tip_local(side, "index")
    target, _ = _head_point(w, q0, side, B03_CHEEK_DEG)
    path = []
    _settle(w, q0, [(b, loc, target)], arm, steps=B03_SETTLE_STEPS, force=B03_PULL_N, path=path, gravity=False)
    path = [np.array(q0)] + path
    touch = None
    for i in range(len(path)):
        if any("head" in (a, c) and (a in hand_bodies or c in hand_bodies) for a, c, _ in w.contacts_now(path[i])):
            touch = i
            break
    if touch is None:
        raise RuntimeError("hand to face: the %s hand never reached the head" % side)
    palm = np.array([w.frame(q, hb)[0] for q in path])
    step_len = np.r_[0.0, np.linalg.norm(np.diff(palm, axis=0), axis=1)]
    arc = np.cumsum(step_len)                                 # the palm's path length at each state
    dt = float(w.m.opt.timestep)
    k0 = max(0, touch - int(round(0.25 / dt)))
    v_phys = (arc[touch] - arc[k0]) / (0.25)
    scale = max(1.0, B03_ARRIVE_MPS / max(v_phys, 1e-6))     # replay speed-up
    per_step = float(w.env.dt) * w.reps / dt                  # physics states per model step
    adrs = [w.jq[j] for j in arm]
    last = len(path) - 1

    def at_index(x):
        x = min(float(last), max(0.0, x))
        k = min(int(x), len(path) - 2)
        fr = x - k
        q = np.array(q0)
        q[adrs] = (1.0 - fr) * path[k][adrs] + fr * path[k + 1][adrs]
        return q
    if kind in ("right", "left"):
        def pose(t):
            return at_index((t - 4.0) * per_step * scale)
        return pose, int(math.ceil(4.0 + last / (per_step * scale))) + 8, None, _toy_far(w)
    k_short = min(int(np.searchsorted(arc, arc[touch] - 0.015)), touch - 1)
    t_short = 4.0 + k_short / (per_step * scale)
    t_rest = t_short + 20.0
    v_seg = (arc[touch] - arc[k_short]) / ((touch - k_short) * dt)          # the physics' own speed there
    slow = B03_CREEP_MPS / max(v_seg, 1e-6)                                # replay factor (under 1: slower)
    end = min(last, touch + int(round(0.25 / dt)))

    def pose(t):
        if t <= t_short:
            return at_index((t - 4.0) * per_step * scale)
        if t <= t_rest:
            return at_index(k_short)
        return at_index(min(end, k_short + (t - t_rest) * per_step * slow))
    t_end = t_rest + (end - k_short) / (per_step * slow)
    return pose, int(math.ceil(t_end)) + 4, None, _toy_far(w)


def _geom_centre(w, q, body_name: str) -> np.ndarray:
    """The world position of a body's first geom at pose q."""
    m = w.m
    b = m.body(body_name).id
    g = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b][0]
    w.fk(q)
    w.mj.mj_kinematics(m, w.d)
    return np.array(w.d.geom_xpos[g])


def case_look_at_arm(w, kind: str):
    """G02 and G03 on a body part (the production report's arm_L_upper sub-rows). The left arm is
    raised from the start (the palm 15 degrees left of the eyes' axis, 22 cm away: its upper arm in
    view) and stays still; box_medium stands 25 degrees to the right, 12 cm away. kind:
      look_back  the eyes rest on the left upper arm's middle (t 3-12), turn to the box (12-24) - no
                 foveal ray on the arm - and back to the arm (24-36): a fixation, a look away of
                 0.6 s, and a return (G02 and G03 on arm_L_upper)
      glance     the eyes rest on the box, glance at the arm for three records (0.10 s: the records of
                 steps 12-14) and return to the box: a fixation of the arm shorter than G02's 0.15 s
    (A sweep across the arm cannot be the twin: 7 cm from the eyes the raised upper arm fills more of
    the view than the eyes can cross - a 40-degree sweep stayed on it 1.65 s in the dry run.)"""
    m = w.m
    q_base = _gaze_scene_q(w)
    fr = _eye_frame(w, q_base)
    hb = m.body("left_hand").id
    goal = _gaze_point(fr, -15.0, 0.0, dist=0.22)
    q0 = w.ik(q_base, hb, np.zeros(3), goal, ARM_L[:4], iters=500)
    box = _gaze_point(fr, 25.0)
    q0 = _place_box(w, q0, "box_medium", box, fr[0])
    arm_pt = _geom_centre(w, q0, "left_upper_arm")

    def eyes(t, q):
        if t < 3.0:
            return q
        if kind == "glance":
            return _aim_eyes(w, q, arm_pt if 12.0 < t <= 15.0 else box)
        if t < 12.0 or t >= 24.0:
            return _aim_eyes(w, q, arm_pt)
        return _aim_eyes(w, q, box)

    def pose(t):
        return np.array(q0)
    if kind not in ("look_back", "glance"):
        raise ValueError(kind)
    return pose, 36, eyes, _toy_far(w)


CAREGIVER_HEAD = "caregiver_head"


def case_g05_caregiver(w, kind: str):
    """G05 on the caregiver (the production report's caregiver sub-row): case_g05 with the adult's
    head as the target - the caregiver's head mocap body placed so its head sphere's centre is 35 cm
    from the eyes, 22 degrees to the right (16 for 'away'); the rest of the adult stays out of the pen.
    The eyes turn to it at t = 3. kind:
      toward  the head turns 16 degrees toward it (t 10-22) while the eyes stay on it
      away    the head turns 12 degrees away from it while the eyes stay on it"""
    m = w.m
    q0 = _gaze_scene_q(w)
    fr = _eye_frame(w, q0)
    s_head = _head_sign(w, q0)
    centre = _gaze_point(fr, 16.0 if kind == "away" else 22.0, dist=0.35)
    k = w.mocap_id(CAREGIVER_HEAD)
    b = m.body(CAREGIVER_HEAD).id
    g = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b][0]
    offset = np.asarray(m.geom_pos[g])                    # the head sphere in the mocap body's frame
    turn = {"toward": 16.0, "away": -12.0}[kind]
    toy = _toy_far(w)

    def scene(t, world):
        toy(t, world)
        R = _quat2mat(world, world.d.mocap_quat[k])
        world.d.mocap_pos[k] = centre - R @ offset

    def pose(t):
        q = np.array(q0)
        q[w.jq["head_swivel"]] += s_head * math.radians(_ramp(t, 10.0, 22.0, 0.0, turn))
        return q

    def eyes(t, q):
        return q if t < 3.0 else _aim_eyes(w, q, centre)
    return pose, 44, eyes, scene


def case_little_touch(w, world_moves: bool):
    """HX3 with the little finger (the production report's R:little sub-row): case_toy_touch with the
    index, middle and ring fingers curled 70 degrees and the thumb 40 degrees, so the little finger
    is the one reaching forward; the toy's pad faces its tip. world_moves: the hand stays still and the
    pad comes in at 10 cm/s until it presses the little fingertip 1.5 mm (HX3 marked world_moved);
    otherwise the pad waits there and the arm carries the hand 5 cm to it at 10 cm/s (HX3 the
    infant's own). The hand starts where the reset pose has it and moves 5 cm along its fingers (the
    world twin holds it at that end pose): case_toy_touch's hand starts 5 cm BACK, a pose that puts the
    upper arm into the chest (1,670 N in its first record, the defect-6 class; this case's first build
    read 1,668 N the same way)."""
    m = w.m
    hb = m.body("right_hand").id
    k = w.mocap_id(TOY)
    q0 = _clear_scene(w, w.base())
    for dg in ("index", "middle", "ring"):
        w.set_curl(q0, "R", dg, 70.0)
    w.set_curl(q0, "R", "thumb", 40.0)
    pad = [gg for gg in range(m.ngeom) if m.geom(gg).name == "caregiver_pad"][0]
    p0, R0 = w.frame(q0, hb)
    f = R0[:, 2]                                           # the finger direction
    travel = 0.05
    q_near, ep, er = w.ik_pose(q0, hb, p0 + travel * f, R0, ARM_R, iters=500)
    if ep > 0.002 or er > 2.0:
        raise RuntimeError("little touch: the hand cannot move 5 cm along its fingers (%.4f m, %.2f deg)" % (ep, er))
    bl, ll = w.tip_local("R", "little")
    tip = w.point(q_near, bl, ll)
    half_thick = float(m.geom_size[pad][2])
    R_pad = np.stack([R0[:, 0], np.cross(f, R0[:, 0]), f], axis=1)
    R_body = R_pad @ _quat2mat(w, m.geom_quat[pad]).T
    pad_final = tip + (half_thick - 0.0015) * f
    body_final = pad_final - R_body @ np.asarray(m.geom_pos[pad])
    quat = _mat2quat(w, R_body)
    speed = 0.10 * 0.05
    t0, t1 = 6.0, 6.0 + travel / speed
    if world_moves:
        def scene(t, world):
            s = _ramp(t, t0, t1, travel, 0.0)
            world.d.mocap_pos[k] = body_final + s * f
            world.d.mocap_quat[k] = quat

        def pose(t):
            return np.array(q_near)
        return pose, int(t1) + 12, None, scene
    at = _timeline(w, [(t0, {j: float(q0[w.jq[j]]) for j in ARM_R}), (t1, {j: float(q_near[w.jq[j]]) for j in ARM_R})])

    def scene(t, world):
        world.d.mocap_pos[k] = body_final
        world.d.mocap_quat[k] = quat

    def pose(t):
        return _set(q0, w, at(t))
    return pose, int(t1) + 12, None, scene


def case_index_to_still_thumb(w):
    """H06's twin (the production report's R:index sub-row; b02_thumb_meets_index is its positive):
    the thumb already turned to face the index (b02's pinch pose, taken by inverse kinematics toward
    the index bent 35 degrees) and still from the start; the index, straight at first, bends until its
    tip meets the thumb's (t 4-12). The fingertips meet (B02) but the thumb's swivel never turns: no
    opposition movement (H06)."""
    b_t, loc_t = w.tip_local("R", "thumb")
    b_i, loc_i = w.tip_local("R", "index")
    q0 = w.base()
    qb = np.array(q0)
    w.set_curl(qb, "R", "index", 35)
    q_thumb = w.ik(qb, b_t, loc_t, w.point(qb, b_i, loc_i), THUMB_R)
    start = np.array(q0)
    for j in THUMB_R:
        start[w.jq[j]] = q_thumb[w.jq[j]]
    q_end = w.ik(start, b_i, loc_i, w.point(start, b_t, loc_t), FINGER_R["index"])
    at = _timeline(w, [(4.0, {j: float(start[w.jq[j]]) for j in FINGER_R["index"]}),
                       (12.0, {j: float(q_end[w.jq[j]]) for j in FINGER_R["index"]})])

    def pose(t):
        return _set(start, w, at(t))
    return pose, 18


CASES = [
    # name, builder, expected present [cap or cap:sub], expected absent, truth {measure: value}
    ("h05_index_alone", lambda w: case_h05_index(w, 0.0), ["H05:R:index"], ["H04"], {"H05.change_deg": 40.0}),
    ("h05_whole_hand", case_h05_whole, [], ["H05"], {}),
    ("h05_others_7deg", lambda w: case_h05_index(w, 7.0), [], ["H05:R:index"], {}),
    ("h05_others_3deg", lambda w: case_h05_index(w, 3.0), ["H05:R:index"], [], {}),
    ("h04_two_cycles", case_h04_cycles, ["H04:R"], [], {"H04.amplitude_deg": 45.0}),
    ("h04_one_close", case_h05_whole, [], ["H04"], {}),
    ("h01_touch_then_close", lambda w: case_h01_on_grasp(w, False), ["H01:R", "R02:box"], [], {}),
    ("h01_close_without_touch", lambda w: case_h01(w, 6.0, False), [], ["H01"], {}),
    ("h01_close_400ms_late", lambda w: case_h01_on_grasp(w, True), [], ["H01"], {}),
    ("h02_h03_hold_then_release", lambda w: case_hold_release_on_grasp(w, "release"), ["H02:R", "H03:R"], ["R08"], {}),
    ("h03_object_escapes", lambda w: case_hold_release_on_grasp(w, "yank_closed"), [], ["H03"], {}),
    ("h03_hand_opens_as_box_is_pulled_out", lambda w: case_hold_release_on_grasp(w, "yank_opening"), ["H02:R"],
     ["H03"], {}),
    ("h07_wrist_flexion", lambda w: case_wrist(w, "flexion"), ["H07:R"], [], {"H07.range_deg": 35.0}),
    ("h07_with_elbow_turn", lambda w: case_wrist(w, "flexion", elbow=30.0), [], ["H07"], {}),
    ("h08_wrist_rotation", lambda w: case_wrist(w, "rotation"), ["H08:R"], [], {"H08.angle_deg": 35.0}),
    ("b02_thumb_meets_index", lambda w: case_pinch(w, ("thumb", "index")), ["B02", "H06:R:index"], [], {}),
    ("b02_neighbours_touch", lambda w: case_pinch(w, ("middle", "index")), [], ["B02", "H06"], {}),
    ("b04_fingertip_to_mouth", lambda w: case_to_head(w, "mouth")[:2], ["B04:R"], [], {}),
    ("b04_fingertip_to_side_of_head", lambda w: case_to_head(w, "side")[:2], [], ["B04", "B03"], {}),
    ("g02_look_at_toy", lambda w: case_look(w, False), ["G02"], [], {}),
    ("g02_sweeping_eyes", lambda w: case_look(w, True), [], ["G02"], {}),
    ("r01_reach_to_box", lambda w: case_reach(w, False), ["R01:R"], [], {}),
    ("r01_move_away", lambda w: case_reach(w, True), [], ["R01", "R02"], {}),
    ("p02_roll_to_side", lambda w: case_roll_driven(w, "side"), ["P02"], [], {}),
    ("p02_half_roll", lambda w: case_roll_driven(w, "half"), [], ["P02"], {}),
    ("p02_body_rolled_by_outside_torque", lambda w: case_roll_turned(w, 80.0), [], ["P02"], {}),
    # ---- the grasp family, one link broken at a time (added 2026-09-24 evening; note above)
    ("grasp_contact_no_opposition", lambda w: case_grasp(w, "no_opposition"),
     [], ["R04", "H02", "R05", "R06", "H03"], {}),
    # The grasp is read as physics (detectors._grip; the coordinator's decision of the evening of
    # 2026-09-25): a grip whose object the floor carries is neither an opposing grasp nor a hold, so this
    # case, the grip without the lift, expects both ABSENT; its lifted twin grasp_lift_and_hold grips the
    # same object off the floor and must fire both. The operator approves the change with the lock.
    ("grasp_opposition_no_lift", lambda w: case_grasp(w, "no_lift"),
     [], ["R04", "H02", "R05", "R06", "H03"], {}),
    ("grasp_lift_no_retention", lambda w: case_grasp_driven(w, "no_retention"),
     ["R04:R", "R05:box_light"], ["H02", "R06", "H03", "R08"], {"R05.height_cm": 3.0}),
    ("grasp_lift_and_hold", lambda w: case_grasp_driven(w, "hold"),
     ["R04:R", "R05:box_light", "H02:R"], ["R06", "H03"], {"R05.height_cm": 3.0}),
    ("grasp_carry_and_place", lambda w: case_grasp(w, "carry_place"),
     ["R04:R", "R05:box_light", "H02:R", "R06:R", "H03:R", "R08:R"], [], {"R06.path_cm": 14.0}),
    ("grasp_release_then_withdraw", lambda w: case_grasp(w, "release_withdraw"),
     ["H03:R", "R08:R"], [], {}),
    # R04 on the crib hand with real dynamics: the thumb against the fingers holds the cube in the air by the
    # hand's own motors (THE PINCH HELD BY THE BODY'S OWN MOTORS; the recorder track, 2026-09-26)
    ("grasp_pinch_held_driven", case_pinch_held_driven, ["R04:R"], ["H03", "R06", "R08"], {}),
    ("grasp_toy_moved_into_still_hand", lambda w: case_toy_touch(w, True),
     ["HX3@world_moved"], ["HX3@own", "H02@own", "R04@own", "R01", "R02"], {}),
    ("grasp_hand_touches_still_toy", lambda w: case_toy_touch(w, False),
     ["HX3:R:middle@own"], ["HX3@world_moved"], {}),
    ("grasp_push_resting_hand", case_push_resting_hand,
     ["O01"], ["H02", "R06", "R04", "R05"], {}),
    # ---- G01 independent eye movement
    ("g01_saccade_head_still", lambda w: case_g01(w, "saccade"), ["G01:L@own", "G01:R@own"], [], {"G01.change_deg": 12.0}),
    ("g01_head_turn_1deg", lambda w: case_g01(w, "head_1deg"), ["G01@own"], [], {}),
    ("g01_head_turn_4deg", lambda w: case_g01(w, "head_4deg"), [], ["G01"], {}),
    ("g01_eyes_counter_rotate_with_head", lambda w: case_g01(w, "with_head"), [], ["G01"], {}),
    ("g01_world_saccade", lambda w: case_g01(w, "world_saccade"), ["G01@scripted"], ["G01@own"], {}),
    ("g01_world_reflex", lambda w: case_g01(w, "world_reflex"), ["G01@scripted"], ["G01@own"], {}),
    # ---- G03 return fixation
    ("g03_look_away_and_back", lambda w: case_g03(w, "return"), ["G03:box", "G02:box"], [], {}),
    ("g03_look_away_and_back_30cm", lambda w: case_g03(w, "return", dist=0.30), ["G03:box", "G02:box"], [], {}),
    ("g03_target_moved_while_away", lambda w: case_g03(w, "cued"), [], ["G03"], {}),
    ("g03_no_return", lambda w: case_g03(w, "no_return"), [], ["G03"], {}),
    ("g03_half_second_away", lambda w: case_g03(w, "short_away"), ["G03:box"], [], {}),
    # ---- G05 eye-head coordination
    ("g05_head_turns_toward_target", lambda w: case_g05(w, "toward"), ["G05:box"], [], {"G05.head_turn_deg": 16.0}),
    ("g05_head_turns_away", lambda w: case_g05(w, "away"), [], ["G05"], {}),
    ("g05_eyes_do_not_follow", lambda w: case_g05(w, "no_follow"), [], ["G05"], {}),
    # ---- R03 visually guided reach
    ("r03_look_then_reach", lambda w: case_r03(w, "look_then_reach"), ["R03:R", "R01:R"], [], {}),
    ("r03_reach_looking_elsewhere", lambda w: case_r03(w, "look_elsewhere"), ["R01:R"], ["R03"], {}),
    ("r03_look_after_reach", lambda w: case_r03(w, "look_after"), ["R01:R"], ["R03"], {}),
    # ---- B06 arm into own view
    ("b06_arm_raises_into_view", lambda w: case_b06(w, "arm_raises"), ["B06:R"], [], {}),
    ("b06_arm_raises_head_4deg", lambda w: case_b06(w, "head_4deg"), ["B06:R"], [], {}),
    ("b06_head_turns_to_still_hand", lambda w: case_b06(w, "head_turns"), [], ["B06"], {}),
    ("b06_eyes_turn_to_still_hand", lambda w: case_b06(w, "eyes_turn"), [], ["B06"], {}),
    # ---- H05 individual finger isolation (beyond the right index)
    ("h05_left_index_alone", lambda w: case_h05_more(w, "left_index"), ["H05:L:index"], ["H05:R"], {"H05.change_deg": 40.0}),
    ("h05_thumb_alone", lambda w: case_h05_more(w, "thumb"), ["H05:R:thumb"], ["H05:R:index", "H05:R:middle"],
     {"H05.change_deg": 40.0}),
    ("h05_index_straightens_from_fist", lambda w: case_h05_more(w, "straighten"), ["H05:R:index"], [],
     {"H05.change_deg": 40.0}),
    ("h05_two_fingers_together", lambda w: case_h05_more(w, "two_fingers"), [], ["H05:R:index", "H05:R:middle"], {}),
    # ---- HX2 shape the palm
    ("hx2_cup_palm", lambda w: case_hx2(w, 30.0), ["HX2:R"], [], {"HX2.amount_deg": 30.0}),
    ("hx2_cup_20deg", lambda w: case_hx2(w, 20.0), ["HX2:R"], [], {}),
    ("hx2_cup_10deg", lambda w: case_hx2(w, 10.0), [], ["HX2"], {}),
    ("hx2_whole_hand_curl", lambda w: case_hx2(w, whole_hand=True), [], ["HX2"], {}),
    # ---- HX1 spread and gather
    ("hx1_spread_and_gather", lambda w: case_hx1(w, "spread"), ["HX1:R"], [], {"HX1.fan_deg": 20.0}),
    ("hx1_sideways_sweep", lambda w: case_hx1(w, "sweep"), [], ["HX1"], {}),
    ("hx1_spread_10deg", lambda w: case_hx1(w, "spread_10"), [], ["HX1"], {}),
    ("hx1_slow_drift", lambda w: case_hx1(w, "slow_drift"), [], ["HX1"], {}),
    # ---- detector v9: a hand act is driven by its own motors (the limp body's passive fan)
    ("hx1_settle_fan_motors_off", lambda w: case_settle_fan(w, "off"), [], ["HX1", "H05"], {}),
    ("hx1_settle_fan_driven", lambda w: case_settle_fan(w, "drive"), ["HX1:R"], ["H05"], {}),
    ("h05_index_alone_motors_off", case_h05_index_motors_off, [], ["H05"], {}),
    ("h05_index_moved_against_its_command", lambda w: _with_motors(case_h05_index(w, 0.0), "against"), [], ["H05"], {}),
    ("hx2_cup_palm_moved_against_its_command", lambda w: _with_motors(case_hx2(w, 30.0), "against"), [], ["HX2"], {}),
    # ---- detector v9: P05 push against support, driven by the body's motors with real dynamics
    ("p05_leg_pushes_the_floor", lambda w: case_support_push(w, "push"), ["P05:R:leg"], [], {}),
    ("p05_leg_extends_in_the_air", lambda w: case_support_push(w, "in_the_air"), [], ["P05"], {}),
    # ---- detector v9, defect 8: the rows the page reports beyond babble without a scripted case
    ("h05_left_little_alone", lambda w: case_h05_more(w, "left_little"), ["H05:L:little"], ["H05:R"],
     {"H05.change_deg": 40.0}),
    ("h05_left_thumb_alone", lambda w: case_h05_more(w, "left_thumb"), ["H05:L:thumb"], ["H05:R", "H05:L:index"],
     {"H05.change_deg": 40.0}),
    ("h05_right_little_alone", lambda w: case_h05_more(w, "right_little"), ["H05:R:little"], ["H05:R:ring"],
     {"H05.change_deg": 40.0}),
    ("h05_little_with_ring", lambda w: case_h05_more(w, "little_ring"), [], ["H05:R:little", "H05:R:ring"], {}),
    ("b06_left_arm_raises_into_view", lambda w: case_b06(w, "left_arm_raises"), ["B06:L"], ["B06:R"], {}),
    ("b06_head_turns_to_still_left_hand", lambda w: case_b06(w, "left_head_turns"), [], ["B06"], {}),
    ("b03_hand_to_right_cheek", lambda w: case_b03(w, "right"), ["B03:R"], [], {}),
    ("b03_hand_to_left_cheek", lambda w: case_b03(w, "left"), ["B03:L"], [], {}),
    ("b03_hand_creeps_onto_cheek", lambda w: case_b03(w, "right_creep"), [], ["B03"], {}),
    ("g02_g03_look_at_left_upper_arm", lambda w: case_look_at_arm(w, "look_back"),
     ["G02:arm_L_upper", "G03:arm_L_upper"], [], {}),
    ("g02_glance_at_left_upper_arm", lambda w: case_look_at_arm(w, "glance"), [], ["G02:arm_L_upper", "G03:arm_L_upper"],
     {}),
    ("g05_head_turns_toward_caregiver", lambda w: case_g05_caregiver(w, "toward"), ["G05:caregiver"], [], {}),
    ("g05_head_turns_away_from_caregiver", lambda w: case_g05_caregiver(w, "away"), [], ["G05"], {}),
    ("hx3_little_fingertip_touches_toy", lambda w: case_little_touch(w, False), ["HX3:R:little@own"],
     ["HX3:R:index", "HX3:R:middle", "HX3:R:ring", "HX3:R:thumb"], {}),
    ("hx3_toy_moved_into_little_fingertip", lambda w: case_little_touch(w, True), ["HX3:R:little@world_moved"],
     ["HX3@own", "R01", "R02"], {}),
    ("h06_index_bends_to_still_thumb", case_index_to_still_thumb, ["B02"], ["H06"], {}),
]

# The boundary cases: {case: ([detectors whose line it probes], what is varied against which line)}.
BOUNDARY: Dict[str, tuple] = {
    "h05_others_7deg": (["H05"], "the other digits move 7 degrees on average: over H05's 5-degree 'near baseline' line"),
    "h05_others_3deg": (["H05"], "the other digits move 3 degrees on average: under H05's 5-degree line"),
    "h01_close_400ms_late": (["H01"], "the hand closes 400 ms after the touch: past the catalog's 50-300 ms"),
    "p02_half_roll": (["P02"], "a roll the body's own motors drive to 35-52 degrees: short of the side band that starts "
                               "at 60 degrees"),
    "grasp_release_then_withdraw": (["H03"], "the hand withdraws at 20 cm/s in the step after its digits finish opening: "
                                             "H03's escape line is 10 cm/s of the box relative to the palm"),
    "g01_head_turn_1deg": (["G01"], "the head turns 1 degree in the saccade's step: within G01's 2-degree head tolerance"),
    "g01_head_turn_4deg": (["G01"], "the head turns 4 degrees in the saccade's step: past the 2-degree tolerance"),
    "g03_half_second_away": (["G03"], "a 0.5 s look away: past G03's 0.3 s minimum"),
    "r03_look_after_reach": (["R03"], "the eyes reach the box 0.2 s after the hand stops: the catalog's 'before or during reach'"),
    "b06_arm_raises_head_4deg": (["B06"], "the head turns 4 degrees while the arm brings the hand into view: within B06's "
                                          "10 degrees over the 250 ms before the entry"),
    "hx2_cup_20deg": (["HX2"], "a 20-degree cupping change: over the catalog's 15-degree line"),
    "hx2_cup_10deg": (["HX2"], "a 10-degree cupping change: under the catalog's 15-degree line"),
    "hx1_spread_10deg": (["HX1"], "a fan of 8 degrees (index +4, little -4; the sum of the four side angles 10.2): "
                                  "under the catalog's 15-degree line"),
    # detector v9
    "p05_leg_extends_in_the_air": (["P05"], "the same knee extension with the foot never on the floor: no support "
                                            "force rises (P05's line: 0.3 body weights of rise)"),
    "h05_little_with_ring": (["H05"], "the little and ring fingers bend 40 degrees together: the others' mean change "
                                      "10 degrees, over H05's 5-degree line, for each of them"),
    "b03_hand_creeps_onto_cheek": (["B03"], "the palm reaches the cheek at about 1 cm/s: under B03's 5 cm/s over "
                                            "the 250 ms before the touch"),
    "g02_glance_at_left_upper_arm": (["G02"], "the eyes stay on the arm for 0.10 s (three records): under G02's "
                                              "0.15 s"),
    "h06_index_bends_to_still_thumb": (["H06"], "the fingertips meet with the thumb's swivel still: under H06's "
                                                "10-degree swivel increase"),
    "p02_body_rolled_by_outside_torque": (["P02"], "the body turned onto its side by a torque from outside with every "
                                                   "motor command zero: P02's motor rule (no positive work of the "
                                                   "body's own motors)"),
}

# Cases whose contract expectation the current detectors do not meet, each with the defect it
# exposes (docs/gain_of_function_v4/agents/observer_validation.md, "Detector defects"). They stay in
# CASES with the contract's expectation; a report lists their failures apart from unexpected ones,
# and a fixed detector moves a case out of this register. Empty since observer_detectors_v8, which
# fixed the five it held under v5-v7: grasp_lift_no_retention (R08 read a dropped box's speed after
# it landed), grasp_release_then_withdraw (H03 read the palm's withdrawal as the box escaping),
# grasp_push_resting_hand (H02 and R06 counted a resting hand sliding a box as a hold and a carry;
# O01 missed a push from rest contact), hx1_sideways_sweep and hx1_slow_drift (HX1 summed the
# |side angles| with no act duration).
KNOWN_DEFECTS: Dict[str, str] = {}

# CASES LEFT OUT OF THE LOCK (the coordinator's decision of 2026-09-26, by the rule that refuses to excuse an R04
# or H02 "present" line: a detector that cannot fire on its own positive case must not be locked). They stay in
# CASES and are run, recorded and reported (each case carries `not_locked` with its reason), but a proposal
# (validate --like --write-golden) leaves them out of its expectations, its boundary registry and its golden
# detections, and lists them under `not_locked`; the qualification gate reports and does not judge them. P02
# (roll to the side): no roll synergy the detector branch's search tries (a423111f2: 8 part sets, both sides, the
# body's own motors, real dynamics) turns the upper body past 29.4 degrees, where p02_roll_to_side needs 70-120
# and p02_half_roll 35-52 (the observer candidate's dry run, 2026-09-26 3:03 AM PDT); the twin rolled by an
# outside torque reaches 74.7 degrees without P02. Revisited after the servo-strength deploy. The operator can
# overrule this at the lock's approval: the proposal says so in its `why`.
NOT_LOCKED_P02 = ("not locked: the body cannot roll itself yet (trunk authority). Its own motors turn the upper "
                  "body at most 29.4 degrees in the roll search, where p02_roll_to_side needs 70-120; revisited "
                  "after the servo-strength deploy (the coordinator's decision of 2026-09-26)")
NOT_LOCKED: Dict[str, str] = {"p02_roll_to_side": NOT_LOCKED_P02, "p02_half_roll": NOT_LOCKED_P02,
                              "p02_body_rolled_by_outside_torque": NOT_LOCKED_P02}


def proposal_parts(qpos_hashes: Dict[str, str], golden: Dict[str, dict]) -> tuple:
    """(expectations, golden, why) of a proposal from the code: CASES and its registers without the NOT_LOCKED
    cases, which are listed apart with their reasons."""
    from . import validation_lock as VL
    from .daemon import DETECTOR_VERSION
    keep = [c for c in CASES if c[0] not in NOT_LOCKED]
    code = VL.expectations_from_code(keep, KNOWN_DEFECTS, {k: v for k, v in BOUNDARY.items() if k not in NOT_LOCKED},
                                     qpos_hashes={n: h for n, h in qpos_hashes.items() if n not in NOT_LOCKED})
    code["not_locked"] = dict(NOT_LOCKED)
    gold = dict(detector_version=DETECTOR_VERSION, cases={n: g for n, g in golden.items() if n not in NOT_LOCKED})
    why = "validate --like --write-golden: this run's recordings and detections, with the code's expectations"
    if NOT_LOCKED:
        why += ("; NOT LOCKED, run and recorded but left out of these expectations and golden detections: %s - %s "
                "The operator can overrule this at this approval: refuse it and ask for a proposal that keeps them."
                % (", ".join(sorted(NOT_LOCKED)), "; ".join(sorted(set(NOT_LOCKED.values())))))
    return code, gold, why

# THE PASS/FAIL LINES ARE LOCKED (operator, 2026-09-25; tools/observer/validation_lock.py). CASES,
# KNOWN_DEFECTS and BOUNDARY above are the builder's working copy: a verdict uses the LOCKED
# expectations, known-defect register and approved scenario fingerprints, never these. A case the
# locked set lacks is run and reported as not approved, and never counts; a change here reaches the
# verdicts only through a proposal the operator approves.


def split_failures(failed: List[str], known: Optional[Dict[str, str]] = None) -> dict:
    """Failed case names split into those the known-defect register explains and the unexpected
    rest. The verdicts pass the LOCKED register; without one, the code's working copy."""
    known = KNOWN_DEFECTS if known is None else known
    return dict(known_defects={n: known[n] for n in failed if n in known},
                unexpected=sorted(n for n in failed if n not in known))


def _lock_state() -> dict:
    """The locked pass/fail lines, verified, or {'error': why} when they are missing or were changed
    without approval (then nothing passes: every verdict fails closed)."""
    from . import validation_lock as VL
    try:
        state = VL.verify()
        changed = VL.uncommitted()
        if changed:
            raise VL.LockError("the locked folder differs from its last commit (%s): the pass/fail lines were "
                               "written outside the operator's approval" % "; ".join(changed[:4]))
    except VL.LockError as error:
        return dict(error=str(error), expectations=dict(cases={}, known_defects={}), golden=dict(cases={}), lock={})
    return state


def _lock_summary(state: dict) -> dict:
    lock = state.get("lock") or {}
    if state.get("error"):
        return dict(intact=False, error=state["error"])
    return dict(intact=True, approved=bool(lock.get("approved")), approved_by=lock.get("approved_by"),
                approved_clock=lock.get("approved_clock"), proposal=lock.get("proposal"),
                note=None if lock.get("approved") else
                "the locked lines await the operator's first approval (validation_lock approve --current)")


def _expected(state: dict, name: str, present, absent, truth):
    """(present, absent, truth, approved, approved scenario fingerprint) for one case."""
    row = (state.get("expectations") or {}).get("cases", {}).get(name)
    if row is None:
        return present, absent, truth, False, None
    return row["present"], row["absent"], row.get("truth") or {}, True, row.get("qpos_hash")


def _check_scenario(case: dict, approved_hash: Optional[str]) -> dict:
    """A changed scenario (a box moved closer, a new pose, another world) is not the approved case: an
    approved case whose recorded trajectory differs from its approved fingerprint fails."""
    if case.get("approved") and approved_hash and case["qpos_hash"] != approved_hash:
        case.update(ok=False, scenario_changed=dict(approved=approved_hash, recorded=case["qpos_hash"]))
    return case


def _locked_verdicts(state: dict, cases: List[dict], ran: List[str]) -> dict:
    """The verdict over approved cases: every locked case run and met; unapproved cases never count."""
    locked = sorted((state.get("expectations") or {}).get("cases", {}))
    missing = sorted(set(locked) - set(ran))
    approved = [c for c in cases if c.get("approved")]
    failed = [c["name"] for c in approved if not c["ok"]]
    return dict(ok=bool(not state.get("error") and not missing and approved and not failed),
                failed=failed, missing_locked_cases=missing,
                not_approved=sorted(c["name"] for c in cases if not c.get("approved")))


def _analyze(folder: str) -> List[dict]:
    from .daemon import RunDigest, read_jsonl
    a = os.path.join(folder, "observer", "analysis")
    if os.path.isdir(a):
        shutil.rmtree(a)
    dg = RunDigest(folder)
    dg.digest(final=True)
    return read_jsonl(os.path.join(dg.out, "detections.jsonl"))


QUALIFIERS = ("own", "world_moved", "scripted", "caregiver_touch")


def _match(d: dict, want: str) -> bool:
    """An expectation: "CAP" any detection of the capability; "CAP:sub" one whose sub-row starts
    with sub; either followed by a qualifier on the provenance flags the detectors set:
      @own              the tracker could count it as the infant's own act: not world_moved, not
                        scripted (the toy's hand-off, the world's eye movers), no caregiver touch
                        (a validation act is always probe-forced by author, so "own" is judged on
                        the flags alone)
      @world_moved      marked world_moved (an object the world moves was moving into the act)
      @scripted         marked scripted
      @caregiver_touch  the caregiver touched the infant during it
    Without a qualifier the flags are not looked at (the cases before 2026-09-24 evening)."""
    spec, _, flag = want.partition("@")
    cap, _, sub = spec.partition(":")
    if d["cap"] != cap:
        return False
    if sub and not str(d.get("sub", "")).startswith(sub):
        return False
    if not flag:
        return True
    if flag == "own":
        return not (d.get("world_moved") or d.get("scripted") or d.get("caregiver_touch"))
    if flag in QUALIFIERS:
        return bool(d.get(flag))
    raise ValueError("unknown expectation qualifier %r in %r" % (flag, want))


def _cap_of(want: str) -> str:
    """The detector an expectation is about ("R04:R:box@own" -> "R04")."""
    return want.partition("@")[0].partition(":")[0]


def _built(built) -> tuple:
    """A builder's result as (pose, steps, eyes, scene): (pose, steps), (pose, steps, eyes) or
    (pose, steps, eyes, scene), where eyes and scene may be None."""
    pose, steps = built[0], built[1]
    eyes = built[2] if len(built) > 2 and callable(built[2]) else None
    scene = built[3] if len(built) > 3 and callable(built[3]) else None
    return pose, int(steps), eyes, scene


def _record_case(w, name: str, built) -> tuple:
    """Record one case's builder result: a Driven case with real dynamics (record_driven), else a
    posed one (record). Returns (folder, model steps)."""
    if isinstance(built, Driven):
        return w.record_driven(name, built), built.steps
    pose, steps, eyes, scene = _built(built)
    return w.record(name, steps, pose, eyes, scene), steps


# THE PHYSICAL LINE (detector v9, 2026-09-25; agents/observer_detectors_2026-09-25.md, "Fix 3"). A scripted
# case moves the body slowly and on purpose: no part of it should press on anything harder than the whole
# body weighs (123.2 N for this body). Measured in the dry runs: the head resting on the floor carries 43-45
# N in every case, a fingertip touch 4-16 N, the driven push's foot 54-85 N; the posed cases that drove a
# limb into the chest or the floor read 297 N to 111,813 N. A case over the line is listed in the report
# (`over_contact_line`) with its pair and record: its scenario, not a detector, is at fault - unless the case is
# `driven` (real dynamics: a body landing after a push can press harder than it weighs).
CONTACT_LINE_BODYWEIGHTS = 1.0


def _largest_body_contact(folder: str) -> dict:
    """The largest contact force a recording stored between a part of the infant and anything (the
    recorder's per-record largest frame of each pair), with its pair, its record and the body's weight."""
    import glob
    from mimo_bridge.observer_record import read_chunk
    best = dict(force_n=0.0, pair=None, record=None, weight_n=None)
    row0 = 0
    for c in sorted(glob.glob(os.path.join(folder, "observer", "world", "b0", "chunk_*.npz"))):
        z = read_chunk(c)
        ents = z["header"]["entities"]
        best["weight_n"] = float(z["header"].get("weight_n", 0.0)) or best["weight_n"]
        fm, ce, rows = np.asarray(z["c_fmax"], dtype=np.float64), np.asarray(z["c_e"]), np.asarray(z["c_row"])
        for j in range(len(fm)):
            a, b = int(ce[j, 0]), int(ce[j, 1])
            if ents[a]["kind"] != "self" and ents[b]["kind"] != "self":
                continue
            if fm[j] > best["force_n"]:
                best.update(force_n=float(fm[j]), pair=[ents[a]["name"], ents[b]["name"]], record=int(rows[j]) + row0)
        row0 += len(z["step"])
    return best


def _over_contact_line(cases: List[dict]) -> List[dict]:
    """The cases whose largest body contact exceeds CONTACT_LINE_BODYWEIGHTS of the body's weight."""
    out = []
    for c in cases:
        lb = c.get("largest_body_contact") or {}
        wgt = lb.get("weight_n") or 0.0
        if wgt and lb.get("force_n", 0.0) > CONTACT_LINE_BODYWEIGHTS * wgt:
            out.append(dict(case=c["name"], force_n=round(lb["force_n"], 1), bodyweights=round(lb["force_n"] / wgt, 2),
                            pair=lb.get("pair"), record=lb.get("record"), driven=bool(c.get("driven"))))
    return out


# THE SCRIPTED TRUTH IS JUDGED (detector v9; the independent review of 2026-09-25: "the truth values are reported,
# never judged", while the lock's text calls them part of the locked lines). A truth value is a measure the script
# sets exactly (H05.change_deg 40: the finger bent 40 degrees); the case's first detection of that capability must
# read it within TRUTH_TOLERANCE of the scripted value, or the case fails, and a truth with nothing to read fails
# too. Measured on the dry run of all 90 cases (2026-09-25 9:28 PM, 17 truth values of the kept cases): 15 read
# exactly, R05.height_cm 2.99 against 3.0 (0.3%) and R06.path_cm 14.04 against 14.0 (0.3%); the two the proposal
# changes fail, as they should: h01_touch_then_close's closure 68.7 against the locked 30 (the grip's closing is
# set by physics now) and hx1_spread_and_gather's HX1.amount_deg, a measure v8 renamed (nothing to read).
TRUTH_TOLERANCE = 0.05


def _truth_errors(dets: List[dict], truth) -> Dict[str, dict]:
    """Each scripted truth value against the first detection of its capability that measures it, and whether it is
    met (THE SCRIPTED TRUTH IS JUDGED)."""
    errors = {}
    for key, value in (truth or {}).items():
        cap, meas = key.split(".")
        vals = [d["measures"].get(meas) for d in dets if d["cap"] == cap and d["measures"].get(meas) is not None]
        if vals:
            err = float(abs(abs(vals[0]) - value))
            errors[key] = dict(scripted=value, measured=float(vals[0]), error=err,
                               met=bool(err <= TRUTH_TOLERANCE * abs(float(value))))
        else:
            errors[key] = dict(scripted=value, measured=None, error=None, met=False)
    return errors


def _judge(name: str, steps: int, dets: List[dict], present, absent, truth, qpos_hash: str) -> dict:
    """One case's verdict: every expected-present entry found, no expected-absent entry found, and every scripted
    truth value read within TRUTH_TOLERANCE (THE SCRIPTED TRUTH IS JUDGED)."""
    found = sorted({"%s:%s" % (d["cap"], d.get("sub", "")) for d in dets})
    ok_p = {p: any(_match(d, p) for d in dets) for p in present}
    ok_a = {a: not any(_match(d, a) for d in dets) for a in absent}
    errors = _truth_errors(dets, truth)
    flagged = sorted({"%s:%s@%s" % (d["cap"], d.get("sub", ""), f) for d in dets for f in QUALIFIERS[1:]
                      if d.get(f)})
    return dict(name=name, steps=steps, expected_present=list(present), expected_absent=list(absent),
                present_found=ok_p, absent_clear=ok_a, found=found, flagged=flagged, measure_errors=errors,
                truth_met={k: e["met"] for k, e in errors.items()},
                ok=all(ok_p.values()) and all(ok_a.values()) and all(e["met"] for e in errors.values()),
                qpos_hash=qpos_hash)


def _confusion(cases: List[dict]) -> Dict[str, dict]:
    """Per detector: true and false positives and negatives across the cases' expectations."""
    per_det: Dict[str, dict] = {}
    for c in cases:
        for p, ok in c["present_found"].items():
            row = per_det.setdefault(_cap_of(p), dict(tp=0, fn=0, fp=0, tn=0))
            row["tp" if ok else "fn"] += 1
        for a, ok in c["absent_clear"].items():
            row = per_det.setdefault(_cap_of(a), dict(tp=0, fn=0, fp=0, tn=0))
            row["tn" if ok else "fp"] += 1
    return per_det


# THE SCENARIO FINGERPRINT (the lock's qpos_hash; detector v9, the independent review of 2026-09-25, "the lock
# cannot see the motor commands that now decide six rows"). Until then it hashed the recorded joint positions
# alone, while since v9 the hand rows count a movement only when its motors carried a command that did positive
# work: h05_index_alone (H05 present), h05_index_alone_motors_off and h05_index_moved_against_its_command (both
# absent) shared one fingerprint, as did hx2_cup_palm and its moved-against twin, and the settle fan driven and
# with its motors off; the world's eye movers' flags decided g01_world_saccade and g01_world_reflex against
# g01_saccade_head_still under one fingerprint; and the grasp cases' 2.5 cm cube is a change of the model, not of
# a joint position. A case's commands, eye-mover flags or cube could be changed with no `scenario_changed`. The
# fingerprint now hashes, record by record, the joint positions (to 1e-6), the motor commands as the recorder
# stores them (float16), the world's saccade and reflex flags, and once per world model the recorder's
# structure fingerprint of it (observer_record.structure_fingerprint: the placement, shape and mass of the
# infant's and every free object's bodies and geoms - the boxes' sizes among them). Every approved fingerprint
# changes once with this, in the same proposal.
SCENARIO_FINGERPRINT = "observer_scenario_v2"


def _qpos_hash(folder: str) -> str:
    """A recording's scenario fingerprint (THE SCENARIO FINGERPRINT, above)."""
    import glob
    from mimo_bridge.observer_record import read_chunk
    h = hashlib.sha256(SCENARIO_FINGERPRINT.encode("utf-8"))
    models = []
    for p in sorted(glob.glob(os.path.join(folder, "observer", "world", "b0", "chunk_*.npz"))):
        z = read_chunk(p)
        h.update(np.round(z["qpos"], 6).tobytes())
        if "ctrl" in z:
            h.update(np.asarray(z["ctrl"], dtype=np.float16).tobytes())
        for k in ("gaze_saccade", "gaze_reflex"):
            if k in z:
                h.update(np.asarray(z[k], dtype=bool).tobytes())
        model = str((z.get("header") or {}).get("model_fingerprint"))
        if model not in models:
            models.append(model)
            h.update(("model:" + model).encode("utf-8"))
    return h.hexdigest()[:16]


def finger_sign(w) -> dict:
    out = {}
    q0 = w.base()
    ok = True
    for side in SIDES:
        hb = w.m.body(("left_" if side == "L" else "right_") + "hand").id
        for dg in DIGITS:
            b, loc = w.tip_local(side, dg)
            p0 = w.point(q0, b, loc)
            palm0 = np.asarray(w.d.xpos[hb]).copy()
            q1 = w.set_curl(np.array(q0), side, dg, 60)
            p1 = w.point(q1, b, loc)
            palm1 = np.asarray(w.d.xpos[hb]).copy()
            d0, d1 = float(np.linalg.norm(p0 - palm0)), float(np.linalg.norm(p1 - palm1))
            out["%s:%s" % (side, dg)] = dict(rest_cm=100 * d0, curled_cm=100 * d1, toward_palm=d1 < d0)
            ok = ok and d1 < d0
    return dict(ok=ok, digits=out)


def panel_order(w) -> dict:
    """Render each painted panel from in front at both ends; the motif found there must be the
    one art_cell names (by the ink colour of the source image's cells)."""
    import mujoco
    from PIL import Image
    from mimo_bridge import pen_art as PA
    from mimo_bridge.observer_record import ART_FIRST_AT_NEGATIVE_END, ART_MOTIFS, art_cell
    m, d = w.m, w.d
    q = w.base()
    w.fk(q)
    mujoco.mj_forward(m, d)
    src = {"n": PA._glyph_wall("ABCDEFGH", [PA.RED, PA.BLUE, PA.GREEN, PA.PURPLE, PA.ORANGE, PA.RED, PA.BLUE, PA.GREEN]),
           "e": PA._glyph_wall("12345678", [PA.BLUE, PA.GREEN, PA.RED, PA.ORANGE, PA.PURPLE, PA.BLUE, PA.GREEN, PA.RED]),
           "s": PA._shapes_wall(), "w": PA._living_wall()}
    alpha = np.array(m.geom_rgba[:, 3])
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) != 0:
            m.geom_rgba[g, 3] = 0.0
    r = mujoco.Renderer(m, 160, 160)

    def ink(img):
        a = np.asarray(img, dtype=np.float64).reshape(-1, 3)
        bg = np.asarray(PA.BG, dtype=np.float64)
        sel = np.linalg.norm(a - bg, axis=1) > 80
        sel &= np.linalg.norm(a - np.asarray(PA.INK), axis=1) > 60
        return a[sel].mean(0) if sel.sum() > 20 else None

    out = {}
    ok = True
    try:
        for key in ("n", "e", "s", "w"):
            g = m.geom("pen_art_" + key).id
            size = np.asarray(m.geom_size[g])
            axis = 0 if size[0] >= size[1] else 1
            half = float(max(size[0], size[1]))
            R = d.geom_xmat[g].reshape(3, 3)
            ctr = np.asarray(d.geom_xpos[g])
            normal = R[:, 2] if np.dot(R[:, 2], -ctr) > 0 else -R[:, 2]
            n = len(ART_MOTIFS[key])
            W = src[key].width
            inks = [ink(src[key].crop((int(j * W / n) + 30, 30, int((j + 1) * W / n) - 30, src[key].height - 30)))
                    for j in range(n)]
            ent = dict(axis=axis, half_width_m=half, half_height_m=float(min(size[0], size[1])),
                       motifs=ART_MOTIFS[key], first_at_negative_end=ART_FIRST_AT_NEGATIVE_END[key])
            res = []
            for end in (0, n - 1):
                local = np.zeros(3)
                local[axis] = -half + 2 * half * (end + 0.5) / n
                pt = ctr + R @ local
                named = art_cell(ent, ctr, R, pt)
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = pt
                cam.distance = 0.35
                cam.azimuth = float(np.degrees(np.arctan2(-normal[1], -normal[0])))
                cam.elevation = 0.0
                r.update_scene(d, camera=cam)
                seen = ink(r.render()[50:110, 50:110])
                cands = [j for j in (0, n - 1) if inks[j] is not None]
                best = min(cands, key=lambda j: np.linalg.norm(seen - inks[j])) if seen is not None and cands else None
                res.append(dict(end=end, named=named, rendered_motif=best, agree=(best == named)))
                ok = ok and best == named
            out[key] = res
    finally:
        r.close()
        m.geom_rgba[:, 3] = alpha
    return dict(ok=ok, panels=out)


def fast_lane_harness(run_dir: str, scratch: str) -> dict:
    """The same 0.6 N s upward impulse on the head in 10 ms or over 500 ms, with real physics.

    Upward, because a sideways push on a head resting on the floor is held by friction: the first
    version (0.2 N s along x) moved the head's accelerometer 0.08 m/s2 either way (2026-09-24).
    The brief push (60 N) exceeds the head's weight (1.96 kg, 19 N) and lifts it; the gradual one
    (1.2 N) never does. PASS: the 10 ms peak separates the two at least three times more than the
    2 s average does, and the brief peak stands at least ten times above the quiet record's."""
    from mimo_bridge.env import BridgeEnv
    from mimo_bridge.observer_record import WorldRecorder, read_chunk
    from .baseline import world_kwargs_like
    from . import fastlane as FL
    import glob
    res = {}
    for kind, frames, force in (("brief", 1, 60.0), ("gradual", 50, 1.2)):
        kw, reps, hold = world_kwargs_like(run_dir, os.path.join(scratch, "fl_" + kind), 11)
        env = BridgeEnv(**kw)
        env.reset()
        if getattr(env, "caregiver", None) is not None:
            env.caregiver = None
        out = os.path.join(scratch, "fastlane_" + kind)
        if os.path.isdir(out):
            shutil.rmtree(out)
        rec = WorldRecorder(os.path.join(out, "observer", "world"), body=0, run="fastlane_" + kind,
                            chunk_steps=1000, budget_bytes=2 ** 30, reserve_bytes=2 ** 30)
        rec.mark("reset")
        hb = env.model.body("head").id
        start = 40                      # frames of quiet first
        total = 200 + start
        zero = np.zeros(env.model.nu)
        for f in range(total):
            env.data.xfrc_applied[hb, :] = 0.0
            if start <= f < start + frames:
                env.data.xfrc_applied[hb, 2] = force          # a push up, along world z
            raw = env.step(None, None, None, ctrl=zero, render=False)
            rec.on_frame(env, raw, dict(step=f // reps, rep=f % reps, reps=reps, author="scripted",
                                        latched=f % reps == 0, order_seq=f // reps, bout=False, rest=False))
        rec.close()
        env.close()
        chunks = [read_chunk(p) for p in sorted(glob.glob(os.path.join(out, "observer", "world", "b0", "chunk_*.npz")))]
        fast = np.concatenate([c["fast"] for c in chunks])
        frames_n = np.concatenate([c["frames"] for c in chunks])
        series = FL.world_series(fast, frames_n)["head_jolt"]
        res[kind] = series[start - 5:]
        res[kind + "_quiet"] = series[start - 25:start - 5]
    cmp = FL.brief_vs_gradual(res["brief"], res["gradual"], slow_window=200)
    cmp["impulse_ns"] = 0.6
    quiet = float(np.nanmax(np.concatenate([res["brief_quiet"], res["gradual_quiet"]]))) if res["brief_quiet"].size else float("nan")
    cmp["quiet_peak"] = quiet
    cmp["separation"] = cmp["peak_ratio"] / max(cmp["slow_ratio"], 1e-12)
    cmp["ok"] = bool(cmp["separation"] >= 3.0 and cmp["peak_brief"] >= 10.0 * max(quiet, 1e-3))
    return cmp


def replay_check(scratch: str, golden_path: Optional[str], out_dir: str, write: bool = False) -> dict:
    """Section 27's replay test without simulating again: every kept case recording is digested
    by the CURRENT detectors and compared with the LOCKED golden detections and judged by the
    LOCKED expectations (validation_lock). Needs no world, so it runs beside training. write=True
    no longer writes anything locked: it PROPOSES the current detections as the new golden ones
    (validation_lock.propose), for the operator to approve. golden_path is ignored (kept for callers)."""
    from .daemon import DETECTOR_VERSION
    from . import validation_lock as VL
    state = _lock_state()
    g = state.get("golden") or dict(cases={})
    diffs, same = [], 0
    current = {}
    code = {c[0]: (c[2], c[3]) for c in CASES}
    expect = {name: (row["present"], row["absent"]) for name, row in
              ((state.get("expectations") or {}).get("cases") or {}).items()}
    truths = {name: row.get("truth") or {} for name, row in ((state.get("expectations") or {}).get("cases") or {}).items()}
    verdicts = {}
    for name, rec in sorted(g["cases"].items()):
        folder = os.path.join(scratch, name)
        if not os.path.isdir(os.path.join(folder, "observer", "world")):
            diffs.append(dict(case=name, change="recording missing"))
            continue
        if _qpos_hash(folder) != rec["qpos_hash"]:
            diffs.append(dict(case=name, change="the recording on disk is not the golden one"))
            continue
        dets = _analyze(folder)
        now = sorted("%s|%s|%d|%d" % (d["cap"], d.get("sub", ""), d["step0"], d["step1"]) for d in dets)
        current[name] = dict(qpos_hash=rec["qpos_hash"], detections=now)
        if name in expect:
            present, absent = expect[name]
            verdicts[name] = bool(all(any(_match(d, p) for d in dets) for p in present)
                                  and not any(_match(d, a) for d in dets for a in absent)
                                  and all(e["met"] for e in _truth_errors(dets, truths.get(name)).values()))
        if now != rec["detections"]:
            diffs.append(dict(case=name, change="same recording, different detections",
                              only_before=sorted(set(rec["detections"]) - set(now)),
                              only_now=sorted(set(now) - set(rec["detections"]))))
        else:
            same += 1
    failed = sorted(k for k, v in verdicts.items() if not v)
    unjudged = sorted(set(expect) - set(verdicts))      # a locked case without a golden recording
    known = (state.get("expectations") or {}).get("known_defects") or {}
    out = dict(detector_version=DETECTOR_VERSION, golden_version=g.get("detector_version"), generated=time.time(),
               lock=_lock_summary(state), cases=len(g["cases"]), same=same, differences=diffs,
               ok=bool(not state.get("error") and not diffs and not failed and not unjudged and g["cases"]),
               expectations=dict(cases=len(verdicts), met=sum(verdicts.values()), failed=failed,
                                 unjudged_locked_cases=unjudged, **split_failures(failed, known)),
               not_in_golden=sorted(set(code) - set(g["cases"])),
               code_expectations_not_approved=sorted(n for n, pa in code.items()
                                                     if n in expect and tuple(map(list, pa)) != tuple(map(list, expect[n]))))
    if write and len(current) == len(g["cases"]):
        folder = VL.propose(None, dict(detector_version=DETECTOR_VERSION, cases=current,
                                       proposed_from=g.get("detector_version")),
                            why="validate --replay --write-golden: the current detectors' detections on the kept "
                                "recordings", by="validate.replay_check")
        out["golden_proposed"] = str(folder)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "replay_report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out


# ====================================================================== babble false-positive rates
PROCESS_MEANING = {
    "limp": "no command at all: what physics and the world's own movers (the saccade generator, the critters, "
            "the mobile, the caregiver's toy) produce. The body chose nothing, so every act here is a false "
            "positive in the strict sense.",
    "random_orders": "the body's own controller given random orders (random orientation, a position within 3 cm, "
                     "strength up to 0.3, held 250 ms), eyes rendered as in training: the chance process every "
                     "stage is judged against.",
    "random_orders_vision_off": "the same random orders with the eyes not rendered, so the eyes' motion reflex never "
                                "runs: kept to show what rendering adds.",
    "random_ctrl": "uniformly random actuator commands, held 250 ms: flailing beyond anything a learned order "
                   "produces (reported, not judged against).",
}
STEPS_PER_1000 = 1000.0
STEP_SECONDS = 0.05


def _pdt(t: float) -> str:
    """A clock time as the operator reads it: AM/PM Pacific."""
    lt = time.localtime(t)
    zone = "PDT" if lt.tm_isdst and time.altzone == 7 * 3600 else "PST" if time.timezone == 8 * 3600 else time.strftime("%Z", lt)
    return time.strftime("%Y-%m-%d %I:%M %p ", lt).replace(" 0", " ") + zone


def _rate_row(c: dict, steps: int) -> dict:
    n, occ = int(c.get("count", 0)), int(c.get("occasions", 0))
    s = int(c.get("steps") or steps or 0)
    return dict(count=n, occasions=occ, body_steps=s,
                per_1000_steps=round(STEPS_PER_1000 * n / s, 3) if s else None,
                occasions_per_1000_steps=round(STEPS_PER_1000 * occ / s, 3) if s else None)


def babble_rates(folder: Optional[str] = None, out_path: Optional[str] = None) -> dict:
    """Addendum section 27's false-positive rates, per detector and sub-row, from the kept babble
    baselines' summaries (<Saved>/mimo_bridge/observer/baselines/<world signature>/<kind>.json):
    the count, the occasions and the rate per 1,000 body-steps under each chance process. Reads
    the JSON summaries only (no world, no digest); re-run it after `baseline --resummarize` or a new
    recording. Written to docs/gain_of_function_v4/observer_validation/babble_rates.json."""
    import glob
    from . import catalog as CAT
    root = str(CAT.ROOT)
    if folder is None:
        base = os.path.join(root, "Saved", "mimo_bridge", "observer", "baselines")
        cands = [p for p in glob.glob(os.path.join(base, "*")) if os.path.isdir(p) and not os.path.basename(p).startswith("_")]
        folder = max(cands, key=os.path.getmtime)
    out_path = out_path or os.path.join(root, "docs", "gain_of_function_v4", "observer_validation", "babble_rates.json")
    cat = CAT.load()
    summaries = {}
    for p in sorted(glob.glob(os.path.join(folder, "*.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                s = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(s, dict) and "caps" in s:
            summaries[os.path.splitext(os.path.basename(p))[0]] = s
    kinds = [k for k in ("limp", "random_orders", "random_orders_vision_off", "random_ctrl") if k in summaries]
    kinds += sorted(k for k in summaries if k not in kinds)
    processes = {}
    for k in kinds:
        s = summaries[k]
        recs = s.get("recordings") or []
        processes[k] = dict(meaning=PROCESS_MEANING.get(k, s.get("kind", k)), body_steps=int(s.get("steps", 0)),
                            bodies=len(recs) if recs else None, vision=s.get("vision"),
                            detector_version=s.get("detector_version"),
                            summarized=_pdt(float(s["generated"])) if s.get("generated") else None,
                            episodes=s.get("episodes"))
    ids = [c["id"] for c in cat["capabilities"]] + [x for x in sorted({c for s in summaries.values() for c in s["caps"]})
                                                     if x not in {c["id"] for c in cat["capabilities"]}]
    rows = {}
    for cid in ids:
        entry = cat["by_id"].get(cid, {})
        row = dict(name=entry.get("name", cid), group=entry.get("group"), observable=entry.get("observable", True),
                   processes={}, subs={})
        for k in kinds:
            c = summaries[k]["caps"].get(cid)
            if c is None:
                continue
            row["processes"][k] = _rate_row(c, summaries[k].get("steps", 0))
            for sub, cs in (c.get("subs") or {}).items():
                row["subs"].setdefault(sub, {})[k] = _rate_row(cs, summaries[k].get("steps", 0))
        if not row["subs"]:
            row.pop("subs")
        rows[cid] = row

    def per(cid, k):
        r = rows.get(cid, {}).get("processes", {}).get(k)
        return r["per_1000_steps"] if r else None

    limp_fp = sorted(((cid, per(cid, "limp")) for cid in rows if (per(cid, "limp") or 0) > 0), key=lambda x: -x[1])
    ref = "random_orders" if "random_orders" in kinds else (kinds[0] if kinds else None)
    top = sorted(((cid, per(cid, ref)) for cid in rows if per(cid, ref)), key=lambda x: -x[1])[:8] if ref else []
    # O12 is the novelty miner's (novelty.py), not a detector's: babble never "fires" it
    never = sorted(cid for cid in rows if rows[cid]["observable"] and rows[cid]["processes"] and cid != "O12"
                   and all(r["count"] == 0 for r in rows[cid]["processes"].values()))
    steps_ref = processes[ref]["body_steps"] if ref else 0
    headlines = []
    if "limp" in kinds:
        headlines.append("A limp body (no command) made %s of the %d detectors' acts; per 1,000 body-steps (50 s of one "
                         "body's life): %s. Each is an act nobody chose." % (
                             len(limp_fp), len(rows), ", ".join("%s %.2f" % (c, v) for c, v in limp_fp) or "none"))
    if ref:
        headlines.append("Random-order babble (the control), per 1,000 body-steps over %d body-steps: %s." % (
            steps_ref, ", ".join("%s %.1f" % (c, v) for c, v in top)))
    if "random_orders" in kinds and "random_orders_vision_off" in kinds:
        headlines.append("Rendering the eyes, as training does, raises babble's independent eye movements (G01) from "
                         "%.1f to %.1f per 1,000 body-steps: the eyes' motion reflex runs only on rendered images." % (
                             per("G01", "random_orders_vision_off") or 0.0, per("G01", "random_orders") or 0.0))
    headlines.append("%d observable detectors never fired under any chance process (%s). Against babble's zero a few "
                     "occasions pass 'more often than babble': with as many body-steps as babble, 5 occasions give "
                     "p = 0.5^5 = 0.03 in the tracker's two-sample test, before the many-tests line. Their size test, "
                     "which needs 20 babble occasions, cannot run." % (len(never), ", ".join(never)))
    explanation = [
        "What this is: how often each detector fires when no learner chooses anything, the false-positive rates the "
        "addendum's section 27 asks for. The tracker judges every stage against the random-order babble below.",
        "A rate of 1 per 1,000 body-steps means one act in every 50 s of one body's simulated life (a model step is "
        "50 ms). Counts are acts; occasions merge acts that start less than 1 s after the previous act of the same "
        "body and part, as the tracker counts them.",
        "Acts the tracker never counts are left out of these counts as well: the toy's scripted hand-off, the world's "
        "eye movers (the G01 rows), caregiver touches and objects the world moved into the act.",
        "Each process ran 8 bodies x 2,500 steps from a world reset (not from production's saved worlds). Each "
        "process's detector_version says which definitions counted it; a summary of another version than the "
        "tracker's is not used as a control.",
    ]
    try:
        source = os.path.relpath(folder, root).replace("\\", "/")
    except ValueError:                   # another drive than the repository's (a scratch folder on D:)
        source = os.path.abspath(folder).replace("\\", "/")
    out = dict(schema="observer_babble_rates_v1", generated=time.time(), generated_pdt=_pdt(time.time()),
               world_signature=os.path.basename(folder), source_folder=source,
               explanation=explanation, headlines=headlines, processes=processes,
               false_positives_in_a_limp_body={c: v for c, v in limp_fp}, never_fired=never, rows=rows)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path + ".part", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        os.replace(out_path + ".part", out_path)
    return out


def coverage() -> Dict[str, dict]:
    """Per detector: the cases that expect it present, absent, and the boundary cases that name
    it (BOUNDARY), from CASES alone (no world)."""
    out: Dict[str, dict] = {}
    for name, _, present, absent, _ in CASES:
        for p in present:
            out.setdefault(_cap_of(p), dict(present=[], absent=[], boundary=[]))["present"].append(name)
        for a in absent:
            out.setdefault(_cap_of(a), dict(present=[], absent=[], boundary=[]))["absent"].append(name)
    for name, (caps, _) in BOUNDARY.items():
        for cap in caps:
            out.setdefault(cap, dict(present=[], absent=[], boundary=[]))["boundary"].append(name)
    for row in out.values():
        for k in row:
            row[k] = sorted(set(row[k]))
    return dict(sorted(out.items()))


def dry_run(names: Optional[List[str]] = None, base_recording: Optional[str] = None,
            scratch: Optional[str] = None) -> dict:
    """Record and digest cases in a ModelWorld (module doc of ModelWorld): the same recorder,
    analysis and verdicts as run(), without BridgeEnv, the panel rendering or the fast-lane
    harness. For checking a case's script beside training; never writes the golden file.
    names: the cases to run (all by default). Needs about 1 GB: start it under
    tools/commit_guard."""
    from .catalog import ROOT
    from .daemon import DETECTOR_VERSION
    base = base_recording or os.path.join(str(ROOT), "Saved", "mimo_bridge", "observer", "validation_scratch",
                                          DEFAULT_BASE_RECORDING)
    scratch = scratch or os.path.join(str(ROOT), "Saved", "mimo_bridge", "observer", "validation_dryrun")
    os.makedirs(scratch, exist_ok=True)
    w = ModelWorld(base, scratch)
    report = dict(detector_version=DETECTOR_VERSION, generated=time.time(), world="ModelWorld (dry run)",
                  base_recording=base, cases=[])
    state = _lock_state()
    ran = []
    for name, build, present, absent, truth in CASES:
        if names and name not in names:
            continue
        built = build(w)
        folder, steps = _record_case(w, name, built)
        dets = _analyze(folder)
        present, absent, truth, approved, _approved_hash = _expected(state, name, present, absent, truth)
        case = _judge(name, steps, dets, present, absent, truth, _qpos_hash(folder))
        case["approved"] = approved                  # a dry run's world is not the approved one: no fingerprint check
        case["largest_body_contact"] = _largest_body_contact(folder)
        case["driven"] = isinstance(built, Driven)
        if name in NOT_LOCKED:
            case["not_locked"] = NOT_LOCKED[name]
        ran.append(name)
        case["detections"] = [dict(cap=d["cap"], sub=d.get("sub", ""), key=d.get("key", ""), step0=d["step0"],
                                   step1=d["step1"], world_moved=bool(d.get("world_moved")),
                                   scripted=bool(d.get("scripted")),
                                   measures={k: v for k, v in d["measures"].items() if isinstance(v, (int, float))})
                              for d in dets]
        report["cases"].append(case)
    report["confusion"] = _confusion(report["cases"])
    report["over_contact_line"] = _over_contact_line(report["cases"])
    verdict = _locked_verdicts(state, report["cases"], ran if not names else
                               sorted((state.get("expectations") or {}).get("cases", {})))
    report["lock"] = _lock_summary(state)
    report["ok"] = verdict["ok"] if not names else not verdict["failed"]
    report["not_approved"] = verdict["not_approved"]
    report["not_locked"] = {n: NOT_LOCKED[n] for n in ran if n in NOT_LOCKED}
    report["failures"] = split_failures([c["name"] for c in report["cases"] if not c["ok"] and c.get("approved")],
                                        (state.get("expectations") or {}).get("known_defects") or {})
    return report


def run(run_dir: str, scratch: str, out_dir: str, golden_path: Optional[str] = None,
        write_golden: bool = False) -> dict:
    from .daemon import DETECTOR_VERSION
    os.makedirs(scratch, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    w = ScriptWorld(run_dir, scratch)       # golden_path is ignored: the golden detections are locked
    report = dict(detector_version=DETECTOR_VERSION, generated=time.time(), cases=[], checks={})
    report["checks"]["finger_sign"] = finger_sign(w)
    report["checks"]["panel_order"] = panel_order(w)
    golden = {}
    state = _lock_state()
    ran = []
    for name, build, present, absent, truth in CASES:
        built = build(w)
        folder, steps = _record_case(w, name, built)
        dets = _analyze(folder)
        present, absent, truth, approved, approved_hash = _expected(state, name, present, absent, truth)
        case = _judge(name, steps, dets, present, absent, truth, _qpos_hash(folder))
        case["approved"] = approved
        case["largest_body_contact"] = _largest_body_contact(folder)
        case["driven"] = isinstance(built, Driven)
        if name in NOT_LOCKED:
            case["not_locked"] = NOT_LOCKED[name]
        _check_scenario(case, approved_hash)
        ran.append(name)
        report["cases"].append(case)
        golden[name] = dict(qpos_hash=case["qpos_hash"],
                            detections=sorted("%s|%s|%d|%d" % (d["cap"], d.get("sub", ""), d["step0"], d["step1"])
                                              for d in dets))
    report["confusion"] = _confusion(report["cases"])
    report["over_contact_line"] = _over_contact_line(report["cases"])
    report["coverage"] = coverage()
    report["checks"]["fast_lane"] = fast_lane_harness(run_dir, scratch)
    # the recordings' detections against the LOCKED golden; write_golden proposes, never writes
    g = state.get("golden") or dict(cases={})
    if write_golden:
        from . import validation_lock as VL
        code, gold, why = proposal_parts({n: rec["qpos_hash"] for n, rec in golden.items()}, golden)
        folder = VL.propose(code, gold, why=why, by="validate.run")
        report["proposal"] = str(folder)
    report["not_locked"] = {n: NOT_LOCKED[n] for n in ran if n in NOT_LOCKED}
    diffs = []
    for name, rec in golden.items():
        old = g["cases"].get(name)
        if old is None:
            diffs.append(dict(case=name, change="new case"))
        elif old["qpos_hash"] != rec["qpos_hash"]:
            diffs.append(dict(case=name, change="the recording changed (simulator or script), detections not comparable"))
        elif old["detections"] != rec["detections"]:
            diffs.append(dict(case=name, change="same recording, different detections",
                              before=old["detections"], after=rec["detections"]))
    report["replay"] = dict(compared=True, golden_version=g.get("detector_version"),
                            same=not diffs, differences=diffs)
    verdict = _locked_verdicts(state, report["cases"], ran)
    report["lock"] = _lock_summary(state)
    report["not_approved"] = verdict["not_approved"]
    report["missing_locked_cases"] = verdict["missing_locked_cases"]
    report["ok"] = (verdict["ok"] and report["checks"]["finger_sign"]["ok"]
                    and report["checks"]["panel_order"]["ok"] and report["checks"]["fast_lane"]["ok"])
    # a case failing on a documented detector defect is still a failure (ok stays strict); the
    # split says which failures the LOCKED register names and which are new
    report["failures"] = split_failures(verdict["failed"],
                                        (state.get("expectations") or {}).get("known_defects") or {})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    return report


def main(argv=None) -> int:
    """python -m mimo_bridge.tools.observer.validate babble-rates [--folder DIR] [--out FILE]
         the per-detector false-positive table from the kept babble summaries (no world)
    python -m mimo_bridge.tools.observer.validate dry-run [--case NAME ...] [--base DIR] [--scratch DIR] [--out FILE]
         the cases in a ModelWorld (about 1 GB: start it after tools/commit_guard --need-gb 1.5);
         exit 1 when a case fails for a reason KNOWN_DEFECTS does not name"""
    import argparse
    ap = argparse.ArgumentParser(prog="python -m mimo_bridge.tools.observer.validate", description=main.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("babble-rates")
    b.add_argument("--folder")
    b.add_argument("--out")
    d = sub.add_parser("dry-run")
    d.add_argument("--case", action="append")
    d.add_argument("--base")
    d.add_argument("--scratch")
    d.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "babble-rates":
        out = babble_rates(a.folder, a.out)
        print(json.dumps(dict(world_signature=out["world_signature"], headlines=out["headlines"]), indent=1))
        return 0
    rep = dry_run(a.case, a.base, a.scratch)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1, default=str)
    print(json.dumps(dict(detector_version=rep["detector_version"], cases=len(rep["cases"]),
                          met=sum(c["ok"] for c in rep["cases"]), failures=rep["failures"], report=a.out or None),
                     indent=1))
    return 1 if rep["failures"]["unexpected"] else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
