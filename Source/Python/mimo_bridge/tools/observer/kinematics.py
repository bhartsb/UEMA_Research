# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The body's geometry, recovered from a recorded step exactly as the simulator placed it.

The world recorder stores qpos, mocap and each object's root pose per step, and saves the
world's compiled model once per kinematic structure (<run>/observer/world/models/
<fingerprint>.mjb.gz, observer_record.structure_fingerprint). Forward kinematics of
that model at the stored state reproduces every infant body's pose (tests/
test_observer_record.py: within 1e-5 m). On top of it this module names the parts the detectors
reason about, each from the model's own geometry rather than a guessed constant:

  finger curl      per digit, the mean of its flexion hinges in degrees: 0 = straight (every
                   finger joint of MIMo rests at 0), 90 = the joints at the curl their ranges
                   allow on average (knuckle 90, middle 100, distal 80 degrees for the four
                   fingers; thumb middle and distal 90). Positive is flexion: the ranges are
                   asymmetric toward it, and validate.py checks that raising them moves each
                   fingertip toward the palm.
  hand closure     the mean curl of the four fingers (not the thumb)
  wrist            the three wrist hinges classified by geometry at the rest pose: the one whose
                   axis runs along the forearm is ROTATION, the one whose axis is along the palm's
                   normal is DEVIATION (side to side), the remaining one FLEXION; flexion is
                   signed so that positive bends the palm toward its own face
  palm normal      the direction the fingertips travel when the fingers curl
  fingertips       the far end of each distal phalanx's capsule
  face and mouth   the head is one sphere; its FACE direction runs from the sphere's centre to the
                   midpoint of the eyes; the face is the cap within 60 degrees of it; the MOUTH is
                   the point on the sphere 35 degrees below the face direction (toward the chin),
                   and the mouth zone is 2.5 cm around it
  trunk            the upper body's frame: belly normal, left-right axis, head-ward axis; the
                   trunk's ROLL is the angle of the belly normal from straight up about the
                   head-ward axis (0 lying on the back, +-90 on a side, 180 on the belly); SIT
                   angle is the head-ward axis from vertical (90 lying, 0 upright)
  eyes             each eye camera's position and viewing axis
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np

SIDES = ("L", "R")
DIGITS = ("index", "middle", "ring", "little", "thumb")
FINGERS = DIGITS[:4]
_DIGIT_KEY = (("_ff_", "index"), ("_mf_", "middle"), ("_rf_", "ring"), ("_lf_", "little"),
              ("_th_", "thumb"))
FINGER_FLEX_ROLES = ("knuckle", "middle", "distal")
THUMB_FLEX_ROLES = ("middle", "distal")
FACE_CONE_DEG = 60.0
MOUTH_DOWN_DEG = 35.0
MOUTH_RADIUS_M = 0.025

def joint_role(name: str):
    """(side, part, role) of one MIMo hinge from its name: ('R', 'index', 'knuckle'),
    ('L', 'wrist', '2'), ('R', 'eye', 'horizontal'), or ('', name, '')."""
    n = name.split(":")[-1]
    side = "R" if n.startswith("right_") else "L" if n.startswith("left_") else ""
    for key, digit in _DIGIT_KEY:
        if key in n:
            return side, digit, n.split(key, 1)[1]
    if side and "_hand" in n:
        return side, "wrist", n.rsplit("_hand", 1)[1]
    if side and "_eye_" in n:
        return side, "eye", n.split("_eye_", 1)[1]
    return side, n, ""


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v * 0.0


# THE HAND JOINT GOAL'S ORDER (observer_attribution_v1, 2026-10-08). A command's hand joint goal (observer_world_v2
# `order_hand_goal`, radians) lists the hand joints in the order mimo_bridge.hand_target_control.geometry_from_model
# builds them: for each hand, left then right, every body on the paths from the finger handles' bodies up to (not
# including) the wrist handle's body, in ascending body id, each body's joints in model order. The finger handles are
# the controller's DIGITS (thumb, index, middle, ring, pinky). The decoder and the proprioception adapter take their
# handle bodies from one map (action_decoder.py and proprio_adapter.py: {**BONE_MAP, **TRUNK_BODIES}), so the world
# header's handle_bodies are the controller's.
GOAL_DIGITS = ("thumb", "index", "middle", "ring", "pinky")


def hand_goal_qpos(m, handle_names, handle_bodies) -> Optional[List[int]]:
    """The qpos address of each entry of a command's hand joint goal, in the controller's order (above); None when the
    handles or the model cannot give it (a reader then treats every goal as unknown)."""
    try:
        hb = {str(n): int(b) for n, b in zip(handle_names, handle_bodies)}
        out: List[int] = []
        for side in SIDES:
            wrist = hb["wrist_" + side]
            needed = set()
            for dg in GOAL_DIGITS:
                b = hb[dg + "_" + side]
                while b != wrist:
                    if b == 0:
                        return None                 # a finger whose ancestry misses its wrist: not this contract
                    needed.add(b)
                    b = int(m.body_parentid[b])
            for b in sorted(needed):
                start = int(m.body_jntadr[b])
                for j in range(start, start + int(m.body_jntnum[b])):
                    out.append(int(m.jnt_qposadr[j]))
        return out
    except (KeyError, ValueError, TypeError, AttributeError, IndexError):
        return None


class Kinematics:
    """Forward kinematics for one world header (one compiled model)."""

    def __init__(self, header: dict, world_dir: str):
        import mujoco
        self.mj = mujoco
        self.header = header
        from mimo_bridge.observer_record import load_world_model
        path = os.path.join(world_dir, header["model_file"])
        # a private copy of the model: mount poses are written into it per step
        self.m = load_world_model(path)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.bid = {m.body(b).name: b for b in range(m.nbody)}
        self.cam = {nm: int(m.camera(nm).id) for nm in ("eye_left", "eye_right")
                    if self._has_camera(nm)}
        # objects whose root is neither free nor mocap: their mount can be moved by editing the
        # model (the caregiver raises and lowers the mobile), so the stored root pose is applied
        self.objects = header["objects"]
        self.mounts = [(k, int(o["root"])) for k, o in enumerate(self.objects)
                       if not o["free"] and int(o["mocap"]) < 0]
        # joints
        self.flex_q: Dict[tuple, List[int]] = {}
        self.flex_hi: Dict[tuple, List[float]] = {}
        self.wrist_q: Dict[str, Dict[str, int]] = {s: {} for s in SIDES}
        self.eye_q: Dict[str, Dict[str, int]] = {s: {} for s in SIDES}
        self.thumb_swivel_q: Dict[str, int] = {}
        for j in header["joints"]:
            side, part, role = joint_role(j["name"])
            if side not in SIDES:
                continue
            if part in FINGERS and role in FINGER_FLEX_ROLES or part == "thumb" and role in THUMB_FLEX_ROLES:
                self.flex_q.setdefault((side, part), []).append(int(j["qpos"]))
                self.flex_hi.setdefault((side, part), []).append(float(j["range"][1]))
            elif part == "thumb" and role == "swivel":
                self.thumb_swivel_q[side] = int(j["qpos"])
            elif part == "wrist":
                self.wrist_q[side][role] = int(j["qpos"])
            elif part == "eye":
                self.eye_q[side][role] = int(j["qpos"])
        self.tip_geom: Dict[tuple, tuple] = {}
        for side in SIDES:
            pre = "left_" if side == "L" else "right_"
            for digit, key in zip(DIGITS, ("ff", "mf", "rf", "lf", "th")):
                b = self.bid.get(pre + key + "distal")
                if b is None:
                    continue
                geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b]
                caps = [g for g in geoms if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE)]
                if caps:
                    g = caps[0]
                    self.tip_geom[(side, digit)] = (g, float(m.geom_size[g][1] + m.geom_size[g][0]))
        self.hand = {s: self.bid.get(("left_" if s == "L" else "right_") + "hand") for s in SIDES}
        self.lower_arm = {s: self.bid.get(("left_" if s == "L" else "right_") + "lower_arm") for s in SIDES}
        self.upper_arm = {s: self.bid.get(("left_" if s == "L" else "right_") + "upper_arm") for s in SIDES}
        self.head = self.bid.get("head")
        self.eyes = [self.bid.get("left_eye"), self.bid.get("right_eye")]
        self.trunk = self.bid.get("upper_body") or self.bid.get("chest")
        self.root = int(header.get("root_body", 1))
        self._head_geometry()
        self._hand_geometry()

    def hand_goal_qpos(self) -> Optional[List[int]]:
        """hand_goal_qpos for this header's world model (cached)."""
        if not hasattr(self, "_goal_qpos"):
            self._goal_qpos = hand_goal_qpos(self.m, self.header.get("handle_names") or [],
                                             self.header.get("handle_bodies") or [])
        return self._goal_qpos

    def _has_camera(self, name: str) -> bool:
        try:
            self.m.camera(name)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ calibration at rest
    def _rest(self):
        self.d.qpos[:] = self.m.qpos0
        self.mj.mj_kinematics(self.m, self.d)

    def _head_geometry(self):
        m, d = self.m, self.d
        self.head_ok = self.head is not None and all(e is not None for e in self.eyes)
        if not self.head_ok:
            return
        self._rest()
        geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == self.head
                 and int(m.geom_type[g]) == int(self.mj.mjtGeom.mjGEOM_SPHERE)]
        g = geoms[0] if geoms else None
        R = d.xmat[self.head].reshape(3, 3)
        p = d.xpos[self.head]
        self.head_c_local = (np.asarray(m.geom_pos[g], dtype=np.float64) if g is not None
                             else np.zeros(3))
        self.head_r = float(m.geom_size[g][0]) if g is not None else 0.075
        eyes_mid = 0.5 * (d.xpos[self.eyes[0]] + d.xpos[self.eyes[1]])
        f = _unit(R.T @ (eyes_mid - p) - self.head_c_local)
        lat = _unit(R.T @ (d.xpos[self.eyes[0]] - d.xpos[self.eyes[1]]))   # toward the left eye
        up = _unit(np.cross(f, lat) * -1.0)
        # "up" must point from the chin to the crown: the side of the head away from the neck
        neck = R.T @ (d.xpos[self.trunk] - p) if self.trunk is not None else -up
        if float(np.dot(up, neck)) > 0:
            up = -up
        self.face_local = f
        self.up_local = up
        a = math.radians(MOUTH_DOWN_DEG)
        self.mouth_local = self.head_c_local + self.head_r * (math.cos(a) * f - math.sin(a) * up)

    def _hand_geometry(self):
        """Palm normal (where curling fingertips go) and the wrist hinges' kinds, at rest."""
        m, d = self.m, self.d
        self.palm_normal_local: Dict[str, np.ndarray] = {}
        self.wrist_kind: Dict[str, Dict[str, str]] = {s: {} for s in SIDES}
        self.wrist_sign: Dict[str, float] = {}
        for side in SIDES:
            hb = self.hand[side]
            if hb is None or not all((side, f) in self.tip_geom for f in ("index", "middle")):
                continue
            self._rest()
            tips0 = np.stack([self._tip(side, f) for f in ("index", "middle")])
            q = np.array(m.qpos0)
            for f in ("index", "middle"):
                for adr in self.flex_q.get((side, f), [])[:1]:
                    q[adr] += 0.5
            self.d.qpos[:] = q
            self.mj.mj_kinematics(m, d)
            tips1 = np.stack([self._tip(side, f) for f in ("index", "middle")])
            R = d.xmat[hb].reshape(3, 3)
            n_world = _unit((tips1 - tips0).mean(0))
            self.palm_normal_local[side] = R.T @ n_world
            self._rest()
            R = d.xmat[hb].reshape(3, 3)
            normal = R @ self.palm_normal_local[side]
            fore = _unit(d.xpos[hb] - d.xpos[self.lower_arm[side]]) if self.lower_arm[side] is not None else None
            for role, adr in self.wrist_q[side].items():
                j = [jj for jj in range(m.njnt) if int(m.jnt_qposadr[jj]) == adr][0]
                axis = np.asarray(d.xaxis[j])
                if fore is not None and abs(float(np.dot(axis, fore))) > 0.7:
                    kind = "rotation"
                elif abs(float(np.dot(axis, normal))) > 0.7:
                    kind = "deviation"
                else:
                    kind = "flexion"
                self.wrist_kind[side][role] = kind
                if kind == "flexion":
                    # positive flexion bends the palm toward its own face: the fingertips move
                    # along the palm normal
                    q = np.array(m.qpos0)
                    q[adr] += 0.3
                    self.d.qpos[:] = q
                    self.mj.mj_kinematics(m, d)
                    moved = self._tip(side, "middle")
                    self._rest()
                    self.wrist_sign[side] = 1.0 if float(np.dot(moved - self._tip(side, "middle"), normal)) > 0 else -1.0
        self._rest()

    def _tip(self, side: str, digit: str) -> np.ndarray:
        g, reach = self.tip_geom[(side, digit)]
        R = self.d.geom_xmat[g].reshape(3, 3)
        return np.asarray(self.d.geom_xpos[g]) + R[:, 2] * reach

    # ------------------------------------------------------------------ joints (no kinematics)
    def curl_deg(self, qpos: np.ndarray) -> np.ndarray:
        """[..., 2 sides, 5 digits] mean flexion of each digit's flexion hinges, degrees."""
        qpos = np.asarray(qpos)
        out = np.full(qpos.shape[:-1] + (2, 5), np.nan)
        for si, side in enumerate(SIDES):
            for di, digit in enumerate(DIGITS):
                adr = self.flex_q.get((side, digit))
                if adr:
                    out[..., si, di] = np.degrees(qpos[..., adr].mean(-1))
        return out

    def curl_full_deg(self, side: str, digit: str) -> float:
        hi = self.flex_hi.get((side, digit))
        return float(np.degrees(np.mean(hi))) if hi else float("nan")

    def wrist_deg(self, qpos: np.ndarray) -> np.ndarray:
        """[..., 2 sides, 3] wrist rotation, flexion (signed), deviation in degrees."""
        qpos = np.asarray(qpos)
        out = np.full(qpos.shape[:-1] + (2, 3), np.nan)
        for si, side in enumerate(SIDES):
            for role, adr in self.wrist_q[side].items():
                kind = self.wrist_kind[side].get(role)
                if kind is None:
                    continue
                k = ("rotation", "flexion", "deviation").index(kind)
                v = np.degrees(qpos[..., adr])
                if kind == "flexion":
                    v = v * self.wrist_sign.get(side, 1.0)
                out[..., si, k] = v
        return out

    def thumb_swivel_deg(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos)
        out = np.full(qpos.shape[:-1] + (2,), np.nan)
        for si, side in enumerate(SIDES):
            if side in self.thumb_swivel_q:
                out[..., si] = np.degrees(qpos[..., self.thumb_swivel_q[side]])
        return out

    def eye_deg(self, qpos: np.ndarray) -> np.ndarray:
        """[..., 2 eyes, 2] eye-in-head horizontal and vertical angles, degrees."""
        qpos = np.asarray(qpos)
        out = np.full(qpos.shape[:-1] + (2, 2), np.nan)
        for si, side in enumerate(SIDES):
            for k, role in enumerate(("horizontal", "vertical")):
                if role in self.eye_q[side]:
                    out[..., si, k] = np.degrees(qpos[..., self.eye_q[side][role]])
        return out

    # ------------------------------------------------------------------ one step's geometry
    def place(self, qpos, mocap=None, object_pose=None) -> None:
        m, d = self.m, self.d
        d.qpos[:] = qpos
        if mocap is not None and m.nmocap:
            d.mocap_pos[:] = np.asarray(mocap)[:, :3]
            d.mocap_quat[:] = np.asarray(mocap)[:, 3:7]
        if object_pose is not None:
            op = np.asarray(object_pose)
            for k, root in self.mounts:
                m.body_pos[root] = op[k, :3]
                m.body_quat[root] = op[k, 3:7]
        self.mj.mj_kinematics(m, d)
        self.mj.mj_comPos(m, d)
        self.mj.mj_camlight(m, d)

    def measure(self) -> dict:
        """The named geometry at the placed state (call place() first)."""
        d = self.d
        out = {}
        tips = np.full((2, 5, 3), np.nan)
        palm = np.full((2, 3), np.nan)
        normal = np.full((2, 3), np.nan)
        fore = np.full((2, 3), np.nan)
        for si, side in enumerate(SIDES):
            for di, digit in enumerate(DIGITS):
                if (side, digit) in self.tip_geom:
                    tips[si, di] = self._tip(side, digit)
            hb = self.hand[side]
            if hb is not None:
                palm[si] = d.xpos[hb]
                R = d.xmat[hb].reshape(3, 3)
                if side in self.palm_normal_local:
                    normal[si] = R @ self.palm_normal_local[side]
                if self.lower_arm[side] is not None:
                    fore[si] = _unit(d.xpos[hb] - d.xpos[self.lower_arm[side]])
        out.update(tips=tips, palm=palm, palm_normal=normal, forearm=fore)
        if self.head_ok:
            R = d.xmat[self.head].reshape(3, 3)
            p = np.asarray(d.xpos[self.head])
            out["head_c"] = p + R @ self.head_c_local
            out["head_face"] = R @ self.face_local
            out["head_up"] = R @ self.up_local
            out["mouth"] = p + R @ self.mouth_local
            out["head_R"] = R.copy()
        else:
            for k in ("head_c", "head_face", "head_up", "mouth"):
                out[k] = np.full(3, np.nan)
            out["head_R"] = np.full((3, 3), np.nan)
        eye_pos = np.full((2, 3), np.nan)
        eye_fwd = np.full((2, 3), np.nan)
        for si, nm in enumerate(("eye_left", "eye_right")):
            c = self.cam.get(nm)
            if c is not None:
                eye_pos[si] = d.cam_xpos[c]
                eye_fwd[si] = -d.cam_xmat[c].reshape(3, 3)[:, 2]
        out.update(eye_pos=eye_pos, eye_fwd=eye_fwd)
        if self.trunk is not None:
            R = d.xmat[self.trunk].reshape(3, 3)
            belly, lat, head_ward = R[:, 0], R[:, 1], R[:, 2]
            out["roll_deg"] = math.degrees(math.atan2(float(lat[2]), float(belly[2])))
            out["sit_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, float(head_ward[2])))))
            out["trunk_pos"] = np.asarray(d.xpos[self.trunk]).copy()
            out["belly"] = belly.copy()
            out["trunk_lat"] = lat.copy()
            out["trunk_R"] = R.copy()
        else:
            out.update(roll_deg=float("nan"), sit_deg=float("nan"), trunk_pos=np.full(3, np.nan),
                       belly=np.full(3, np.nan), trunk_lat=np.full(3, np.nan),
                       trunk_R=np.full((3, 3), np.nan))
        sh = [self.upper_arm[s] for s in SIDES]
        out["shoulder_z"] = np.asarray([d.xpos[b][2] if b is not None else np.nan for b in sh])
        out["com"] = np.asarray(d.subtree_com[self.root]).copy()
        out["root_pos"] = np.asarray(d.xpos[self.root]).copy()
        out["xpos"] = np.asarray(d.xpos).copy()
        out["xquat"] = np.asarray(d.xquat).copy()
        return out

    def head_local(self, point: np.ndarray) -> Optional[np.ndarray]:
        """A world point in the head's frame (at the placed state)."""
        if self.head is None:
            return None
        R = self.d.xmat[self.head].reshape(3, 3)
        return R.T @ (np.asarray(point) - self.d.xpos[self.head])

    def face_zone(self, point_head_local: np.ndarray) -> str:
        """'mouth', 'face' or 'head' for a point on (or near) the head, in the head's frame."""
        if not self.head_ok:
            return "head"
        if float(np.linalg.norm(point_head_local - self.mouth_local)) <= MOUTH_RADIUS_M:
            return "mouth"
        v = _unit(point_head_local - self.head_c_local)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v, self.face_local))))))
        return "face" if ang <= FACE_CONE_DEG else "head"
