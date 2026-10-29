# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Action decoder: the model's 23 handle targets -> MIMo torque-motor controls.

The model outputs, per handle: target position (cm, world), target rotation (quat xyzw,
world), radius, strength. MIMo v2 only has torque motors (ctrl in [-1, 1], force =
gear x ctrl). MIMo's PositionalModel cannot be used (Phase 0/plan check: it zeroes ctrl
every substep and the v2 XML has no position actuators), so the bridge does its own
position servo on top of the default SpringDamperModel:

  1. For handle h the "chain" is a short list of MIMo bodies (e.g. index finger tip,
     middle, knuckle). The rotation still needed, in the world frame, is
        w = log( R_target * R_current(handle body)^T )          (rotation vector, rad)
  2. Each hinge joint j in the chain has a world-frame axis a_j. We want joint changes
     dq that add up to w:   sum_j a_j dq_j ~= w.  Solved by damped least squares
        dq = (A^T A + lam I)^-1 A^T w,   A = [a_1 .. a_n]  (3 x n)
     so a rotation about one joint's axis goes to that joint and a rotation shared by
     two parallel joints is split between them.
  3. ctrl_j = clip( strength_h * (kp * dq_j - kd * qdot_j), ctrl_low, ctrl_high )
     Gains kp 2, kd 0.3 (tuned on the real body: a 28.6 degree shoulder command is met within
     1.4 degrees in 0.6 s with the rotation direction matching the command, cos 0.98; kd 0.05
     reached the target too but oscillated, cos 0.2).
     Joints not in any chain get ctrl 0 (MIMo's spring-damper holds them).
  4. Gaze (azimuth, elevation in radians) drives the four eye joints with the same
     PD rule (right eye horizontal axis is flipped in the XML, handled here).

Position targets (small IK, added after E4 take 2): for the 12 hand handles (wrists and
finger tips) the wanted body-origin move e = p_target - p_current (m) is added as three more
rows: a hinge joint j moves the body origin by  a_j x (p_body - anchor_j) per radian, so
   [ A      ] dq = [ w        ]      (orientation rows, rad)
   [ wp * J ]      [ wp * e   ]      (position rows, m, weight wp = POSITION_WEIGHT)
The weight is 8, so 0.125 m of wanted move weighs the same as 1 radian of wanted turn. It was
5 and this line still said 5 after it was raised. The wanted move itself is clipped to the
handle's OWN reach (see _measure_reach), which the model's joint ranges give as 5.48 to
6.80 cm for a finger and 42.85 cm for a wrist, so a finger's position rows can now weigh at
most 0.54 against a turn of up to pi. Before that clip was per-handle a finger could bring a
50 cm move into the same solve, weighing 4.
solved together by the same damped least squares. For wrists the chain then includes the
arm (hand + lower arm + upper arm joints). The three arm handles (wrist, lowerarm,
upperarm) are solved TOGETHER in one stacked least squares per arm, so their targets are
reconciled by the solver instead of fighting on the shared shoulder/elbow actuators
(review fix 7); each joint's ctrl is gated by the strength of the handle that owns it.
Radius is ignored. Everything here is pure numpy + MuJoCo model reads, no MIMo imports,
so tests can build a tiny MuJoCo model by hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from lem.model.body_sets import HANDLE_NAMES
from mimo_bridge.proprio_adapter import BONE_MAP

MOTOR_CONTRACT = 'ancestor_arm_axes_signed_limits_v3'

# THE FINGER COMMAND'S MEANING (--finger-joint-servo, default off; the first-finger investigation of Oct 8, 2026,
# finding C, D:/UEMA_Research/external/D/UEMA_scratch/finger_investigation/body_execution/REVIEW.md). Under hand_target_mode
# wrist_relative_v1 the planner search and the hand stage choose one goal angle per finger hinge (44 per body,
# hand_target_control.HandActionSpace, lem/model/hand_response.py), the learner records the command's finger rows as
# the forward kinematics of those goals, and the world receives the goals themselves. With the setting off (today's
# rule) the decoder reads none of the goal angles: it servos each finger handle's POSE (the fingertip bone's position
# and orientation that the goals give on the actual arm) by the damped least squares of decode() below, whose hinge
# split for a large change can push a hinge against its own goal (episode v2 of the investigation: the left middle
# hinge commanded +0.525 to +0.585 while its goal lay 21.5 to 27.1 degrees lower). 'on': each finger hinge's step is
# its own goal minus its angle, dq_j = goal_j - q_j, in place of that least squares; the hinge range clip,
# u = strength x (kp x dq - kd x hinge speed), the chain's one rescale and the final clip stay as they are. The
# coupling tendons stay in the body (MuJoCo's tendon limits) and in the goals (HandKinematics.project_joints).
FINGER_JOINT_SERVO_CHOICES = ('off', 'on')
FINGER_SERVO_CONTRACT = 'finger_joint_servo_v1'
FINGER_DIGITS = ('thumb', 'index', 'middle', 'ring', 'pinky')

# handle -> MIMo bodies whose hinge joints the handle may move (distal first)
HANDLE_CHAINS: Dict[str, Tuple[str, ...]] = {
    "wrist_L": ("left_hand", "left_lower_arm", "left_upper_arm"),
    "wrist_R": ("right_hand", "right_lower_arm", "right_upper_arm"),
    "thumb_L": ("left_thdistal", "left_thhub", "left_thbase"),
    "thumb_R": ("right_thdistal", "right_thhub", "right_thbase"),
    "index_L": ("left_ffdistal", "left_ffmiddle", "left_ffknuckle"),
    "index_R": ("right_ffdistal", "right_ffmiddle", "right_ffknuckle"),
    "middle_L": ("left_mfdistal", "left_mfmiddle", "left_mfknuckle"),
    "middle_R": ("right_mfdistal", "right_mfmiddle", "right_mfknuckle"),
    "ring_L": ("left_rfdistal", "left_rfmiddle", "left_rfknuckle"),
    "ring_R": ("right_rfdistal", "right_rfmiddle", "right_rfknuckle"),
    "pinky_L": ("left_lfdistal", "left_lfmiddle", "left_lfknuckle", "left_lfmetacarpal"),
    "pinky_R": ("right_lfdistal", "right_lfmiddle", "right_lfknuckle", "right_lfmetacarpal"),
    "head": ("head",),
    "upperarm_L": ("left_upper_arm",), "lowerarm_L": ("left_lower_arm",),
    "upperarm_R": ("right_upper_arm",), "lowerarm_R": ("right_lower_arm",),
    "thigh_L": ("left_upper_leg",), "calf_L": ("left_lower_leg",),
    "thigh_R": ("right_upper_leg",), "calf_R": ("right_lower_leg",),
    "foot_L": ("left_foot",), "foot_R": ("right_foot",),
    # EMPTY chain, on purpose: the eye hinges are driven by the dedicated eye servo
    # (decode()'s gaze path), sourced from this handle's orientation model-side. Giving
    # the chain the eye bodies would put the same joints under the limb servo too and
    # they would fight.
    "gaze": (),
}
assert set(HANDLE_CHAINS) == set(HANDLE_NAMES)

# Handles whose POSITION target reaches the solve (the rest are orientation-only). Until
# 2026-09-18 only the 12 hand handles were here, so lifting the head or bridging the hips
# could be asked for only as a target ROTATION of one body - the model's head and trunk
# position numbers reached no actuator (gain-of-function tree X7). A position row is
# useful only where the handle's own joints can move the handle body's origin, which is
# the lever arm axis x (p_body - anchor) in the solve below. Measured on the crib body:
#   head        - head_tilt and head_tilt_side are anchored 1.2 and 2.4 cm behind the head
#                 origin: a lever, so "head up 5 cm" becomes neck extension (test passes,
#                 predicted displacement along the ask, cos > 0.7);
#   lower_trunk - the three hip hinges sit 5 cm from the lower-body origin: a lever;
#   chest       - chest_lean and chest_rot are anchored EXACTLY at the chest origin
#                 (0.249, 0, 0.056 both), so its lever arm is zero and a chest position
#                 target has no joint that can close it. Left orientation-only.
# Arm and leg segments stay orientation-only: the arm is solved as one group whose wrist
# already carries position rows, and a leg segment's own chain (hip, knee or ankle
# alone) cannot move that segment's origin.
POSITION_HANDLES = {"wrist_L", "wrist_R", "thumb_L", "index_L", "middle_L", "ring_L", "pinky_L",
                    "thumb_R", "index_R", "middle_R", "ring_R", "pinky_R",
                    "head", "lower_trunk"}
# THE HEAD AND PELVIS POSITION ROWS, REMOVED BY THE OPERATOR ON 2026-09-25 AT 9:09 AM PDT (motor
# interface audit F3 option A, docs/gain_of_function_v4/agents/motor_interface_audit.md; recorded in
# docs/gain_of_function_v3/WORKLOG.md "Operator decisions"): on real orders they performed 1.7% (head)
# and 10.5% (pelvis) of their ask and turned their part 3.9 and 11.5 degrees off its orientation-only
# solution; their orientation rows stay. This reverses the tree's X7 decision above. It reaches the
# body only through the decoder's position_rows argument (train_mimo --position-rows no_head_pelvis),
# whose default 'all' keeps the rows exactly as they were until a deploy turns it on.
POSITION_ROW_CHOICES = ("all", "no_head_pelvis")
HEAD_PELVIS_POSITION_HANDLES = frozenset({"head", "lower_trunk"})


def position_handles(position_rows: str = "all") -> set:
    """The handles whose position row the decoder solves for under this setting."""
    if position_rows not in POSITION_ROW_CHOICES:
        raise ValueError(f"position_rows must be one of {POSITION_ROW_CHOICES}, not {position_rows!r}")
    return set(POSITION_HANDLES) - (set(HEAD_PELVIS_POSITION_HANDLES) if position_rows == "no_head_pelvis" else set())


POSITION_WEIGHT = 8.0      # rad per metre of position error in the combined solve (5 -> 8: position rows pull harder, review follow-up)

EYE_JOINTS = {   # joint name -> (which gaze component, sign)
    "robot:left_eye_horizontal": (0, +1.0),
    "robot:left_eye_vertical": (1, +1.0),
    "robot:right_eye_horizontal": (0, -1.0),   # axis 0 0 -1 in MIMo_modelv2.xml
    "robot:right_eye_vertical": (1, +1.0),
}


# ------------------------------------------------------------------ small rotation helpers
def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = math_sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def math_sqrt(v: float) -> float:
    return float(np.sqrt(v))


def rotmat_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle), angle in [0, pi]."""
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    ang = float(np.arccos(c))
    if ang < 1e-9:
        return np.zeros(3)
    if abs(np.pi - ang) < 1e-6:
        # At pi the symmetric part is axis @ axis.T. Use its strongest
        # column so relative signs survive even when axis[0] is zero.
        A = (R + R.T + 2.0 * np.eye(3)) / 4.0
        axis = A[:, int(np.argmax(np.diag(A)))].copy()
        # Just below pi the skew part still determines the principal sign.
        # At exactly pi either sign describes the same rotation.
        skew = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if np.dot(axis, skew) < 0.0:
            axis = -axis
        n = np.linalg.norm(axis)
        return axis / n * ang if n > 0 else np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2.0 * np.sin(ang)) * ang


def damped_least_squares(A: np.ndarray, w: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Solve min |A dq - w|^2 + lam |dq|^2.  A is (3, n)."""
    n = A.shape[1]
    if n == 0:
        return np.zeros(0)
    AtA = A.T @ A + lam * np.eye(n)
    return np.linalg.solve(AtA, A.T @ w)


# ------------------------------------------------------------------ decoder
def _measure_reach(model, body_id: int, joint_ids: Sequence[int], max_joints: int = 10) -> float:
    """How far this handle's own joints can move it, in metres, from the model's own ranges.

    Every joint of the chain is taken to one end of its range or the other and the handle's
    body origin is placed; the answer is the widest separation between any two of those
    placements. Measured on the crib body at 24 months: a finger reaches 5.48 to 6.80 cm and a
    wrist, whose chain includes the whole arm, reaches 42.85 cm.

    WHY THIS EXISTS. The decoder clipped a wanted move to 50 cm for every handle alike. For a
    wrist that is about right; for a finger it is 9 times what the finger can do, so an
    unreachable finger target went into the least squares at its full size and swamped the
    orientation rows it shares the solve with. Measured on the live run's own snapshots, the
    model orders a fingertip to a place 10.0 to 12.5 cm away while the fingertip sits 2.1 to
    4.0 cm from its wrist, and the fingers' joints sit at the far end of their range: the
    middle angle of a finger against its wrist reads 150 to 165 degrees.

    Runs on a scratch copy of the state, so nothing the caller holds is disturbed. A handle
    with no joints, or more than max_joints of them, keeps the old 0.5 m.
    """
    import itertools
    import mujoco
    js = [int(j) for j in joint_ids]
    if not js or len(js) > max_joints:
        return 0.5
    scratch = mujoco.MjData(model)
    mujoco.mj_forward(model, scratch)
    base = np.array(scratch.qpos)
    pts = []
    for corner in itertools.product((0, 1), repeat=len(js)):
        scratch.qpos[:] = base
        for end, j in zip(corner, js):
            adr = int(model.jnt_qposadr[j])
            if model.jnt_limited[j]:
                lo, hi = float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1])
            else:
                lo, hi = -1.5, 1.5
            scratch.qpos[adr] = hi if end else lo
        mujoco.mj_forward(model, scratch)
        pts.append(np.array(scratch.xpos[body_id]))
    pts = np.stack(pts)
    spread = float(np.max(np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)))
    return float(min(0.5, max(spread, 0.01)))     # never wider than before, never under 1 cm


@dataclass
class _HandleInfo:
    body_id: int
    joint_ids: List[int]
    dof_adrs: List[int]
    act_ids: List[int]
    axis_body_ids: List[int]   # body the joint axis is expressed in
    # HOW FAR THIS HANDLE'S OWN JOINTS CAN MOVE IT, metres. Measured from the model at build
    # time by taking every joint of its chain to each end of its range. A wanted move is
    # clipped to this rather than to one number for every handle alike.
    reach_m: float = 0.5


class ActionDecoder:
    def __init__(self, model, kp: float = 2.0, kd: float = 0.3, gaze_kp: float = 2.0,
                 gaze_kd: float = 0.3, lam: float = 1e-3, ctrl_bounds=None, body_layout='legacy',
                 position_rows: str = 'all'):
        import mujoco
        self._mj = mujoco
        self.model = model
        from .body_layout import names_for, TRUNK_BODIES
        self.handle_names = HANDLE_NAMES if body_layout == 'legacy' else names_for(body_layout)
        body_map = {**BONE_MAP, **TRUNK_BODIES}
        chains = {**HANDLE_CHAINS, **{name: (body,) for name, body in TRUNK_BODIES.items()}}
        self.kp, self.kd, self.gaze_kp, self.gaze_kd, self.lam = kp, kd, gaze_kp, gaze_kd, lam
        self.nu = int(model.nu)
        # actuator lookup: joint id -> actuator id (first actuator on that joint)
        self.joint_to_act: Dict[int, int] = {}
        for a in range(model.nu):
            if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                j = int(model.actuator_trnid[a, 0])
                self.joint_to_act.setdefault(j, a)
        self.ctrl_low = np.array([model.actuator_ctrlrange[a, 0] if model.actuator_ctrllimited[a] else -1.0
                                  for a in range(model.nu)], dtype=np.float64)
        self.ctrl_high = np.array([model.actuator_ctrlrange[a, 1] if model.actuator_ctrllimited[a] else 1.0
                                   for a in range(model.nu)], dtype=np.float64)
        # THE RAILS ARE THE BODY'S, NOT THE MODEL FILE'S (2026-08-27). MIMo's action space
        # is narrower than actuator_ctrlrange on 56 of 90 actuators (wrist flexion tops
        # out at +0.57, hip bend at +0.765, left shoulder abduction bottoms at -0.6875)
        # and env.step clamps to it silently. Clipping here to +/-1 meant the recorded
        # order, the saturation statistic and the strength the PD asked for were all wrong
        # on those channels. ctrl_bounds = (low, high) from the env's action space.
        if ctrl_bounds is not None:
            lo, hi = (np.asarray(ctrl_bounds[0], dtype=np.float64),
                      np.asarray(ctrl_bounds[1], dtype=np.float64))
            self.ctrl_low = np.maximum(self.ctrl_low, lo)
            self.ctrl_high = np.minimum(self.ctrl_high, hi)
        self.handles: List[_HandleInfo] = []
        # 'all' (the default) is every row of POSITION_HANDLES, as before the setting existed
        self.position_rows = position_rows
        live_positions = position_handles(position_rows)
        self.use_position = [h in live_positions for h in self.handle_names]
        for h in self.handle_names:
            body_id = int(model.body(body_map[h]).id)
            joint_ids, dof_adrs, act_ids, axis_bodies = [], [], [], []
            for bname in chains[h]:
                b = int(model.body(bname).id)
                for j in range(model.njnt):
                    if int(model.jnt_bodyid[j]) != b or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                        continue
                    if j not in self.joint_to_act:
                        continue
                    joint_ids.append(j)
                    dof_adrs.append(int(model.jnt_dofadr[j]))
                    act_ids.append(self.joint_to_act[j])
                    axis_bodies.append(b)
            self.handles.append(_HandleInfo(body_id, joint_ids, dof_adrs, act_ids, axis_bodies,
                                            reach_m=_measure_reach(model, body_id, joint_ids)))
        # arm groups (review fix 7): indices of (wrist, lowerarm, upperarm) per side, plus
        # per-joint owner = the member whose own body carries the joint (that member's
        # strength gates the joint's ctrl)
        name_to_idx = {n: i for i, n in enumerate(self.handle_names)}
        self.arm_groups = []
        for side in ("L", "R"):
            if any(f"{n}_{side}" not in name_to_idx for n in ("wrist", "lowerarm", "upperarm")):
                continue          # shrunken handle tables (tests) have no arm groups
            members = [name_to_idx[f"wrist_{side}"], name_to_idx[f"lowerarm_{side}"], name_to_idx[f"upperarm_{side}"]]
            wrist = self.handles[members[0]]
            owners = []
            for j in wrist.joint_ids:
                jb = int(model.jnt_bodyid[j])
                own = members[0]
                for mi in members:
                    if self.handles[mi].body_id == jb or jb in self.handles[mi].axis_body_ids:
                        own = mi
                for mi in members[1:]:
                    if jb == self.handles[mi].body_id:
                        own = mi
                owners.append(own)
            # World-frame forearm rotation depends on the shoulder as well as
            # the elbow. The handle's actuator ownership is not its Jacobian.
            columns = {}
            for mi in members:
                ancestors = set()
                body = self.handles[mi].body_id
                while body > 0:
                    ancestors.add(body)
                    body = int(model.body_parentid[body])
                columns[mi] = [k for k, j in enumerate(wrist.joint_ids)
                               if int(model.jnt_bodyid[j]) in ancestors]
            self.arm_groups.append({"members": members, "owners": owners, "columns": columns})
        self._grouped = {i for g in self.arm_groups for i in g["members"]}
        # eye joints
        self.eye: List[Tuple[int, int, int, float]] = []   # (act, dof, gaze component, sign)
        for jname, (comp, sign) in EYE_JOINTS.items():
            try:
                j = int(model.joint(jname).id)
            except Exception:
                continue
            if j in self.joint_to_act:
                self.eye.append((self.joint_to_act[j], int(model.jnt_dofadr[j]), comp, sign))
        # the finger handles (FINGER_DIGITS): the chains a joint goal may drive under --finger-joint-servo on
        self.finger_handles = frozenset(i for i, h in enumerate(self.handle_names)
                                        if h.split("_")[0] in FINGER_DIGITS)
        self.last_ctrl = np.zeros(self.nu)

    # -------------------------------------------------------------- one chain's step
    def _pose_step(self, data, motor: np.ndarray, i: int, h: _HandleInfo) -> np.ndarray:
        """The chain's step dq (rad per hinge) toward handle i's ordered pose: the damped least squares of the
        rotation still needed and, for a position row, the wanted move weighted POSITION_WEIGHT (the rule every
        handle outside the arm groups has always used)."""
        R_t = quat_xyzw_to_mat(motor[i, 3:7])
        R_c = np.asarray(data.xmat[h.body_id]).reshape(3, 3)
        w = rotmat_log(R_t @ R_c.T)          # world-frame rotation still needed
        # world axes of the chain joints (orientation rows)
        n_j = len(h.joint_ids)
        A = np.zeros((3, n_j))
        for k, (j, b) in enumerate(zip(h.joint_ids, h.axis_body_ids)):
            # Several hinges can belong to one body. Its FINAL rotation
            # includes later hinges; applying it to every local axis rotates
            # earlier axes twice. MuJoCo supplies each actual world hinge axis.
            A[:, k] = data.xaxis[j]
        if self.use_position[i] and np.all(np.isfinite(motor[i, 0:3])):
            # position rows: hinge j moves the body origin by a_j x (p_body - anchor_j)
            p_body = np.asarray(data.xpos[h.body_id], dtype=np.float64)
            e = np.asarray(motor[i, 0:3], dtype=np.float64) / 100.0 - p_body   # cm -> m
            # never pull a handle further than its own joints can carry it (see
            # _measure_reach): 50 cm for every handle alike was 9 times a finger's reach
            e = np.clip(e, -h.reach_m, h.reach_m)
            J = np.zeros((3, n_j))
            for k, j in enumerate(h.joint_ids):
                J[:, k] = np.cross(A[:, k], p_body - np.asarray(data.xanchor[j], dtype=np.float64))
            A6 = np.vstack([A, POSITION_WEIGHT * J])
            w6 = np.concatenate([w, POSITION_WEIGHT * e])
            return damped_least_squares(A6, w6, self.lam)
        return damped_least_squares(A, w, self.lam)

    def _finger_joint_step(self, data, h: _HandleInfo, finger_goals) -> np.ndarray:
        """--finger-joint-servo on: the chain's step dq (rad per hinge) is each hinge's own goal minus its angle,
        read from data.qpos. finger_goals: {MuJoCo joint id: goal angle in rad} for every hinge of the chain (the
        world's latched hand joint goal, HandTargetControl.goals_by_joint)."""
        missing = [int(j) for j in h.joint_ids if int(j) not in finger_goals]
        if missing:
            raise ValueError("--finger-joint-servo on: no joint goal for finger hinges %s" % missing)
        return np.array([float(finger_goals[int(j)]) - float(data.qpos[int(self.model.jnt_qposadr[j])])
                         for j in h.joint_ids], dtype=np.float64)

    # -------------------------------------------------------------- main entry
    def decode(self, data, motor: np.ndarray, gaze: Optional[Sequence[float]] = None,
               handle_mask: Optional[Sequence[bool]] = None,
               finger_goals: Optional[Dict[int, float]] = None) -> np.ndarray:
        """motor: (24, 9) = pos cm(3), quat xyzw(4), radius, strength (strength in [0,1]); row 23 = gaze.
        gaze: (azimuth, elevation) radians or None. Returns ctrl (nu,).
        finger_goals: None (the default; --finger-joint-servo off) = every chain steps toward its handle's ordered
        pose (_pose_step). A dict {MuJoCo joint id: goal angle, rad} (--finger-joint-servo on) = the finger handles'
        chains step toward those hinge goals instead (_finger_joint_step); everything after the step is shared."""
        m = self.model
        ctrl = np.zeros(self.nu, dtype=np.float64)
        qd = np.asarray(data.qvel, dtype=np.float64)
        for i, h in enumerate(self.handles):
            if i in self._grouped:
                continue          # arms are solved as one stacked system below
            if handle_mask is not None and not handle_mask[i]:
                continue
            if not h.joint_ids:
                continue
            strength = float(np.clip(motor[i, 8], 0.0, 1.0))
            if strength <= 0.0:
                continue
            if finger_goals is not None and i in self.finger_handles:
                dq = self._finger_joint_step(data, h, finger_goals)
            else:
                dq = self._pose_step(data, motor, i, h)
            for k, (a, dof, j) in enumerate(zip(h.act_ids, h.dof_adrs, h.joint_ids)):
                # aim at the reachable part: a target outside the joint range becomes the range edge,
                # so the servo does not keep pressing into the stop (E4 take 1: cos ~0.03 because most
                # raw model targets were unreachable and the arm sat on its limits)
                if m.jnt_limited[j]:
                    qj = float(data.qpos[int(m.jnt_qposadr[j])])
                    lo, hi = float(m.jnt_range[j, 0]), float(m.jnt_range[j, 1])
                    dq[k] = float(np.clip(qj + dq[k], lo, hi)) - qj
                u = strength * (self.kp * dq[k] - self.kd * qd[dof])
                ctrl[a] += u          # handles that share a joint add their pulls; clipped below
        for grp in self.arm_groups:
            wrist_i = grp["members"][0]
            wrist = self.handles[wrist_i]
            if not wrist.joint_ids:
                continue
            active = [mi for mi in grp["members"]
                      if (handle_mask is None or handle_mask[mi]) and float(motor[mi, 8]) > 0.0]
            if not active:
                continue
            joints = wrist.joint_ids            # union: hand + lower arm + upper arm
            n_j = len(joints)
            rows, rhs = [], []
            # world axes once per joint
            axes = np.zeros((3, n_j))
            for k, (j, b) in enumerate(zip(joints, wrist.axis_body_ids)):
                axes[:, k] = data.xaxis[j]
            for mi in active:
                h = self.handles[mi]
                sm = float(np.clip(motor[mi, 8], 0.0, 1.0))
                R_t = quat_xyzw_to_mat(motor[mi, 3:7])
                R_c = np.asarray(data.xmat[h.body_id]).reshape(3, 3)
                w = rotmat_log(R_t @ R_c.T)
                A = np.zeros((3, n_j))
                for k in grp["columns"][mi]:
                    A[:, k] = axes[:, k]
                rows.append(sm * A)
                rhs.append(sm * w)
                if self.use_position[mi] and np.all(np.isfinite(motor[mi, 0:3])):
                    p_body = np.asarray(data.xpos[h.body_id], dtype=np.float64)
                    e = np.clip(np.asarray(motor[mi, 0:3], dtype=np.float64) / 100.0 - p_body,
                                -h.reach_m, h.reach_m)
                    J = np.zeros((3, n_j))
                    for k in grp["columns"][mi]:
                        j = joints[k]
                        J[:, k] = np.cross(axes[:, k], p_body - np.asarray(data.xanchor[j], dtype=np.float64))
                    rows.append(sm * POSITION_WEIGHT * J)
                    rhs.append(sm * POSITION_WEIGHT * e)
            dq = damped_least_squares(np.vstack(rows), np.concatenate(rhs), self.lam)
            for k, (a, dof, j) in enumerate(zip(wrist.act_ids, wrist.dof_adrs, joints)):
                if m.jnt_limited[j]:
                    qj = float(data.qpos[int(m.jnt_qposadr[j])])
                    lo, hi = float(m.jnt_range[j, 0]), float(m.jnt_range[j, 1])
                    dq[k] = float(np.clip(qj + dq[k], lo, hi)) - qj
                owner = grp["owners"][k]
                s_own = float(np.clip(motor[owner, 8], 0.0, 1.0)) if owner in active                     else max(float(np.clip(motor[mi, 8], 0.0, 1.0)) for mi in active)
                ctrl[a] += s_own * (self.kp * dq[k] - self.kd * qd[dof])
        # DIRECTION-PRESERVING RESCALE before the safety clip (B6 fix, 2026-08-26).
        # The per-element clip alone threw away most of the action: measured at the
        # run kp of 8.0, 55.8% of nonzero demands exceeded the +/-1 rail, 68.7% of
        # total demanded magnitude was discarded, and of a commanded change only
        # 43.7% survived to the body - with ~20% of channels delivering exactly
        # nothing because both sides were pinned to the same rail. Clipping each
        # element independently also destroys the RATIOS between joints, which is
        # the part of the order that says what movement it is. Each handle chain and
        # each arm group is scaled down as a unit instead, so a saturating order
        # keeps its shape; the clip below remains as a final safety.
        groups = [h.act_ids for h in self.handles if h.act_ids]
        groups += [grp["members"] and self.handles[grp["members"][0]].act_ids
                   for grp in self.arm_groups]
        for act_ids in groups:
            if not act_ids:
                continue
            ids = np.asarray(act_ids, dtype=int)
            values = ctrl[ids]
            # Limits can differ by direction. Using the larger absolute limit
            # let the final clip distort commands on the weaker side.
            bound = np.where(values >= 0, self.ctrl_high[ids], -self.ctrl_low[ids])
            ratios = np.ones_like(values)
            np.divide(bound, np.abs(values), out=ratios, where=values != 0)
            ctrl[ids] *= max(0.0, min(1.0, float(np.min(ratios))))
        np.clip(ctrl, self.ctrl_low, self.ctrl_high, out=ctrl)
        if gaze is not None:
            q = np.asarray(data.qpos, dtype=np.float64)
            for a, dof, comp, sign in self.eye:
                j = int(m.actuator_trnid[a, 0])
                qpos_adr = int(m.jnt_qposadr[j])
                target = sign * float(gaze[comp])
                lo, hi = m.jnt_range[j]
                if m.jnt_limited[j]:
                    target = float(np.clip(target, lo, hi))
                u = self.gaze_kp * (target - q[qpos_adr]) - self.gaze_kd * qd[dof]
                ctrl[a] = np.clip(u, self.ctrl_low[a], self.ctrl_high[a])
        self.last_ctrl = ctrl
        return ctrl

    def handle_joint_names(self) -> Dict[str, List[str]]:
        """For reports/tests: which MIMo joints each handle drives."""
        out = {}
        for name, h in zip(self.handle_names, self.handles):
            out[name] = [self.model.joint(j).name for j in h.joint_ids]
        return out
