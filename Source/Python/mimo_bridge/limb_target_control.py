# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Orders the calves, the chest and the arms can perform (motor interface audit F2a and F2b, 2026-09-25).

The motor interface audit (docs/gain_of_function_v4/agents/motor_interface_audit.md, revision
d7a348f03) found three body regions whose world-pose orders the body cannot perform, beside the
fingers that hand_target_control.py (wrist_relative_v1) repaired on 2026-09-24. Every fix below is
off unless a launch flag turns it on (train_mimo --parent-relative-targets, --reach-projection), so a
resume of an old command on this code changes nothing.

F2a, JOINT GOALS HELD ON THE PARENT (--parent-relative-targets calf | calf_chest).
  The calf's chain is one hinge, the knee, carried by the thigh. On the live replay ring (checkpoint
  399,225) the calf's target, taken relative to the thigh's target, lay a median 16.6 degrees from any
  knee pose, 13.5 to 15.0 degrees of its ask lay beyond the knee's reach, and under physics a thigh
  turning 13 to 16 degrees in 250 ms left a fixed world calf target 8.6 to 10.7 degrees out of reach.
  The chest's chain is two hinges (lean, twist) on the upper body, which the pelvis's hip joints carry
  through MIMo's tendon couplings; 13.2 degrees of its median 22.2-degree ask lay beyond its reach.
  The fix generalises the finger repair: at every new order the calf's target is projected onto the
  knee relative to the thigh's ORDERED row, and the chest's onto its two joints relative to the upper
  body the pelvis's ordered row implies (the pelvis's hip joints closest to that row inside their
  ranges, the upper body's hip joints placed as MIMo's tendons couple them). The joint goal is kept.
  Before every 10 ms frame the world row the decoder reads is regenerated around the parent where the
  decoder sees it (data.xquat), so the goal turns with the parent. The ledger records the canonical
  row: the joint goal placed on the parent's ordered pose.

F2b, THE ARM'S FOUR TARGETS FROM ONE JOINT SOLUTION (--reach-projection arms).
  The arm's upper-arm, forearm and hand rows and the hand's position contradicted each other in real
  orders (the forearm's target 21.4 degrees from any elbow pose relative to the upper arm's, the hand's
  position 23.8 cm from where the segment targets put it) and half the hand targets lay beyond the
  straight arm. At each new order, with the chest where it is, the decoder's own stacked arm solve is
  iterated to convergence inside the joint ranges from the arm's current pose (forward kinematics, no
  physics), and the four targets become the world poses of the joint solution it settles on. They stay
  world targets for the order's tenure: no regeneration around the chest. With finger goals on, the
  ledger's finger rows are placed again on the projected wrist, so they keep meaning the finger goal
  relative to the recorded wrist (the wrist_relative_joint_v1 contract).

THE PLANNER'S TWIN OF THE CALF PROJECTION (calf_rows_torch). With the calves projected, the learner's
planner imagines the calf rows the worlds record (LemLite.planned_execution, from the geometry the
worlds report). The chest's and the arms' projections need state the planner does not hold; the causal
chain's plan-to-executed distance against the recorded order and this module's stats block per author
read how far the planner's own orders are moved there.

F3, THE HEAD'S AND PELVIS'S POSITION ROWS (the operator's decision of 2026-09-25 9:09 AM PDT) is a
decoder setting (action_decoder.position_handles, train_mimo --position-rows); this module carries its
contract helpers (POSITION_ROWS_CONTRACT, check_position_rows). Not implemented here: the head's own
projection (its orientation orders were reachable at every measured latch) and the trunk and head servo
under load (F7), ordered separately by the same decision.
"""
from __future__ import annotations

import copy
import math
import time
from typing import Optional, Sequence, Tuple

import numpy as np

CALF_CONTRACT = 'parent_relative_calf_v1'
CHEST_CONTRACT = 'parent_relative_chest_v1'
ARM_CONTRACT = 'arm_reach_projection_v1'
# the head and pelvis position rows removed from the decoder's solve (action_decoder.position_handles,
# train_mimo --position-rows no_head_pelvis; the operator's decision of 2026-09-25 9:09 AM PDT)
POSITION_ROWS_CONTRACT = 'no_head_pelvis_position_v1'
PARENT_RELATIVE_CHOICES = ('off', 'calf', 'calf_chest')
REACH_PROJECTION_CHOICES = ('off', 'arms')
CALF_HANDLES = (('thigh_L', 'calf_L'), ('thigh_R', 'calf_R'))
# replay's action_contract bits for moments recorded under parent-relative goals (bit 0, value 1, is
# the wrist-relative finger contract, lem_model.py). A moment's calf or chest row means a different
# thing under the two interfaces, so replay never mixes them in one window; so do the head's and the
# pelvis's position numbers once the decoder no longer reads them.
CALF_CONTRACT_BIT = 2
CHEST_CONTRACT_BIT = 4
POSITION_ROWS_CONTRACT_BIT = 8
# The eye row's convention (train_mimo --eye-row-convention physical; motor fixes F1, the fourth review's
# fix round of 2026-09-26, the coordinator's decision): a moment whose outgoing order was recorded under the
# 'physical' eye row convention, with the model's gaze command recorded beside it (ReplayMemory.ctrl_gaze),
# carries this bit (LemLite.step sets it; the world does not know the learner's convention). Under 'physical'
# replay studies only windows of such moments, so after the switch no window reads a moment stored before it:
# its memory and self states were encoded from old-convention rows, which no rewrite reaches. Under 'legacy'
# no moment carries it and replay's contract is the one of before, bit for bit.
EYE_ROW_CONTRACT_BIT = 16
# The finger command's meaning (train_mimo --finger-joint-servo on; action_decoder.py, THE FINGER COMMAND'S MEANING):
# a moment executed while the decoder servoed each finger hinge toward its own joint goal carries this bit (the
# world reports 'finger_servo' only then), and so does the transition's action contract (action_contract_parts).
# The same command rows then lead to other finger movements than under the fingertip-pose servo, so replay never
# mixes the two in one window. With the setting off no moment carries it: replay's contract is the one of before.
FINGER_SERVO_CONTRACT_BIT = 32
# the arm projection: the decoder's own arm solve iterated at 0.3 of each solved change (the servo
# moves the arm a fraction of its solved change per 10 ms, never the whole of a large one), from the
# arm's current pose, then a bounded polish of the objective that iteration settles on. Measured on
# the saved model at 120 recorded poses (motor_fixes_limbs_2026-09-25.md): full steps left the
# decoder's own loop short of the projected order in 14 of 120 large random orders, this in 3
ARM_FLOW_STEP, ARM_FLOW_ITERATIONS, ARM_STEP_TOLERANCE = 0.3, 400, 1e-9


# ------------------------------------------------------------------ quaternions (numpy, wxyz)
def unit_wxyz_from_xyzw(q) -> np.ndarray:
    """A motor row's xyzw quaternion as a unit wxyz one, with the decoder's rule that a zero
    quaternion (norm under 1e-12) means the identity (action_decoder.quat_xyzw_to_mat)."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    x, y, z, w = q / n
    return np.array([w, x, y, z])


def qmul(a, b) -> np.ndarray:
    """Quaternion product (wxyz); leading dimensions broadcast."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def qconj(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a * np.array([1.0, -1.0, -1.0, -1.0])


def q_axis_angle(axis, angle) -> np.ndarray:
    """Turn about a unit axis; `angle` may be an array (the result then has its shape plus 4)."""
    axis = np.asarray(axis, dtype=np.float64)
    half = np.asarray(angle, dtype=np.float64) / 2.0
    return np.concatenate([np.cos(half)[..., None], np.sin(half)[..., None] * axis], axis=-1)


def qrotate(q, v) -> np.ndarray:
    return qmul(qmul(q, np.concatenate(([0.0], np.asarray(v, dtype=np.float64)))), qconj(q))[1:]


def xyzw_aligned(q_wxyz, reference_xyzw) -> np.ndarray:
    """q as an xyzw row, its sign chosen to match the reference row's hemisphere (q and -q are
    one rotation; keeping the recorded numbers continuous costs nothing)."""
    out = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
    out /= max(float(np.linalg.norm(out)), 1e-300)
    if float(out @ np.asarray(reference_xyzw, dtype=np.float64)) < 0.0:
        out = -out
    return out


def angle_between_deg(a_xyzw, b_xyzw) -> float:
    a, b = unit_wxyz_from_xyzw(a_xyzw), unit_wxyz_from_xyzw(b_xyzw)
    return math.degrees(2.0 * math.acos(min(1.0, abs(float(a @ b)))))


def _unit(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


# ------------------------------------------------------------------ one hinge on a parent (the knee)
def one_axis_goal(parent_xyzw, child_xyzw, body_quat_wxyz, axis, ref, lo, hi) -> Tuple[float, float]:
    """(joint angle, residual in radians) for a child body carried by its parent through ONE hinge.

    The child's pose relative to the parent is body_quat x rot(axis, q - ref) (MuJoCo's own
    placement, mj_kinematics). Returns the q inside [lo, hi] whose pose is closest to the child
    orientation asked for relative to the parent orientation asked for, and the angle that pose
    still leaves (0 = the ask was a pose of that hinge already). Closed form: for a unit relative
    rotation (w, v) the closeness |w cos(t/2) + (v.axis) sin(t/2)| peaks at t = 2 atan2(v.axis, w) and
    falls monotonically away from it within half a period, so the best angle inside the range is that
    peak (shifted by whole turns) or the nearer range end."""
    qp = unit_wxyz_from_xyzw(parent_xyzw)
    qc = unit_wxyz_from_xyzw(child_xyzw)
    a = _unit(axis)
    j = qmul(qconj(qmul(qp, _unit(body_quat_wxyz))), qc)     # the hinge's own turn asked for
    w, s = float(j[0]), float(j[1:] @ a)
    peak = 2.0 * math.atan2(s, w)
    best_score, best_t = -1.0, 0.0
    for k in (0, -1, 1):                       # k = 0 first: it wins exact ties
        t = float(np.clip(peak + 2.0 * math.pi * k, lo - ref, hi - ref))
        score = abs(w * math.cos(t / 2.0) + s * math.sin(t / 2.0))
        if score > best_score + 1e-15:
            best_score, best_t = score, t
    return best_t + ref, 2.0 * math.acos(min(1.0, best_score))


def one_axis_child_wxyz(parent_wxyz, body_quat_wxyz, axis, angle, ref) -> np.ndarray:
    """The child's world orientation (wxyz) with its hinge at `angle`, on a parent at parent_wxyz."""
    return qmul(qmul(np.asarray(parent_wxyz, dtype=np.float64), _unit(body_quat_wxyz)),
                q_axis_angle(_unit(axis), angle - ref))


# ------------------------------------------------------------------ a short chain of hinges on one body
class HingeChain:
    """The hinges of ONE body, applied in MuJoCo's order after the body's own frame (body_quat):
    orientation = parent x body_quat x rot(a_1, q_1 - ref_1) x ... Positions do not enter."""

    def __init__(self, body_quat, axes, refs, lo, hi):
        self.body_quat = _unit(body_quat)
        self.axes = [_unit(a) for a in axes]
        self.refs = np.asarray(refs, dtype=np.float64)
        self.lo, self.hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)

    @classmethod
    def of_body(cls, model, body, joints):
        joints = [int(j) for j in joints]
        adr = [int(model.jnt_qposadr[j]) for j in joints]
        if any(int(model.jnt_bodyid[j]) != int(body) for j in joints):
            raise ValueError('a hinge chain must belong to one body')
        lo = [float(model.jnt_range[j, 0]) if model.jnt_limited[j] else -math.pi for j in joints]
        hi = [float(model.jnt_range[j, 1]) if model.jnt_limited[j] else math.pi for j in joints]
        return cls(model.body_quat[body], [model.jnt_axis[j] for j in joints],
                   [float(model.qpos0[a]) for a in adr], lo, hi)

    def place(self, parent_wxyz, q) -> np.ndarray:
        """World orientation(s) of the body; q may be [n] or [G, n]."""
        q = np.asarray(q, dtype=np.float64)
        out = qmul(np.asarray(parent_wxyz, dtype=np.float64), self.body_quat)
        if q.ndim == 2:
            out = np.broadcast_to(out, (len(q), 4))
        for k, axis in enumerate(self.axes):
            out = qmul(out, q_axis_angle(axis, q[..., k] - self.refs[k]))
        return out

    def closest(self, parent_wxyz, target_wxyz, points: Sequence[int]) -> Tuple[np.ndarray, float]:
        """(joint angles inside the ranges, residual radians) of the pose closest to the target: a
        grid over the ranges for the start, then a bounded quasi-Newton polish of the squared angle
        (its gradient on each hinge is -2 x world axis . world turn still needed, as the decoder's)."""
        from scipy.optimize import minimize
        target = _unit(target_wxyz)
        grids = np.meshgrid(*[np.linspace(a, b, n) for a, b, n in zip(self.lo, self.hi, points)], indexing='ij')
        Q = np.stack([g.reshape(-1) for g in grids], axis=-1)
        close = np.abs(self.place(parent_wxyz, Q) @ target)
        q0 = Q[int(np.argmax(close))]
        base = qmul(np.asarray(parent_wxyz, dtype=np.float64), self.body_quat)

        def f(q):
            cur, world_axes = base, []
            for k, axis in enumerate(self.axes):
                world_axes.append(qrotate(cur, axis))
                cur = qmul(cur, q_axis_angle(axis, q[k] - self.refs[k]))
            err = qmul(target, qconj(cur))
            if err[0] < 0.0:
                err = -err
            s = float(np.linalg.norm(err[1:]))
            theta = 2.0 * math.atan2(s, float(err[0]))
            w = err[1:] * (theta / s) if s > 1e-300 else np.zeros(3)
            return theta * theta, np.array([-2.0 * float(a @ w) for a in world_axes])
        res = minimize(f, q0, jac=True, method='L-BFGS-B', bounds=list(zip(self.lo, self.hi)),
                       options=dict(maxiter=200, ftol=1e-16, gtol=1e-14))
        q = np.clip(np.asarray(res.x, dtype=np.float64), self.lo, self.hi)
        if f(q)[0] > f(q0)[0]:
            q = q0
        return q, math.sqrt(f(q)[0])


# ------------------------------------------------------------------ F2a: the calf
class _Calf:
    """One calf: its handle rows, bodies and knee, read from the compiled model."""

    def __init__(self, model, decoder, thigh_name, calf_name):
        names = list(decoder.handle_names)
        self.thigh_row, self.calf_row = names.index(thigh_name), names.index(calf_name)
        self.thigh_body = int(decoder.handles[self.thigh_row].body_id)
        self.calf_body = int(decoder.handles[self.calf_row].body_id)
        joints = list(decoder.handles[self.calf_row].joint_ids)
        if len(joints) != 1:
            raise ValueError('parent-relative calf goals need exactly one hinge on the calf, found %d' % len(joints))
        if int(model.body_parentid[self.calf_body]) != self.thigh_body:
            raise ValueError('the calf body is not carried by the thigh body')
        if int(model.body_jntnum[self.calf_body]) != 1:
            raise ValueError('the calf body carries more than the knee')
        self.joint = int(joints[0])
        self.qpos_address = int(model.jnt_qposadr[self.joint])
        self.body_quat = np.array(model.body_quat[self.calf_body], dtype=np.float64)
        self.axis = _unit(model.jnt_axis[self.joint])
        self.ref = float(model.qpos0[self.qpos_address])
        if model.jnt_limited[self.joint]:
            self.lo, self.hi = (float(v) for v in model.jnt_range[self.joint])
        else:
            self.lo, self.hi = -math.pi, math.pi
        self.name = calf_name

    def goal(self, order) -> Tuple[float, float]:
        return one_axis_goal(order[self.thigh_row, 3:7], order[self.calf_row, 3:7],
                             self.body_quat, self.axis, self.ref, self.lo, self.hi)

    def canonical_xyzw(self, order, angle) -> np.ndarray:
        """The ledger's calf row: the knee goal placed on the thigh's ORDERED row."""
        q = one_axis_child_wxyz(unit_wxyz_from_xyzw(order[self.thigh_row, 3:7]), self.body_quat,
                                self.axis, angle, self.ref)
        return xyzw_aligned(q, order[self.calf_row, 3:7])

    def regenerated_xyzw(self, data, angle, reference_xyzw) -> np.ndarray:
        """The row sent this 10 ms frame: the knee goal placed on the thigh where the decoder sees
        it (data.xquat is the kinematic snapshot the decoder reads its poses and axes from)."""
        q = one_axis_child_wxyz(np.array(data.xquat[self.thigh_body], dtype=np.float64),
                                self.body_quat, self.axis, angle, self.ref)
        return xyzw_aligned(q, reference_xyzw)


# ------------------------------------------------------------------ F2a: the chest
class _Chest:
    """The chest's two hinges on the upper body, and the upper body the pelvis's order implies.

    The upper body (MIMo's upper_body) is carried by the pelvis (lower_body, the lower_trunk handle)
    through three hinges no motor drives; limited fixed tendons hold each at a fixed ratio of one of
    the pelvis's hinges (bend and twist equal, lean twice). Those ratios are read from the model."""

    GRID_PELVIS = (15, 19, 25)          # about 2 degrees apart over the pelvis's ranges
    GRID_CHEST = (19, 37)               # 1 degree apart over the chest's ranges

    def __init__(self, model, decoder):
        names = list(decoder.handle_names)
        self.pelvis_row, self.chest_row = names.index('lower_trunk'), names.index('chest')
        self.pelvis_body = int(decoder.handles[self.pelvis_row].body_id)
        self.chest_body = int(decoder.handles[self.chest_row].body_id)
        self.upper_body = int(model.body_parentid[self.chest_body])
        self.root_body = int(model.body_parentid[self.pelvis_body])
        if int(model.body_parentid[self.upper_body]) != self.pelvis_body:
            raise ValueError('the chest\'s parent is not carried by the pelvis body')
        pelvis_joints = list(decoder.handles[self.pelvis_row].joint_ids)
        chest_joints = list(decoder.handles[self.chest_row].joint_ids)
        for body, joints in ((self.pelvis_body, pelvis_joints), (self.chest_body, chest_joints)):
            if int(model.body_jntnum[body]) != len(joints):
                raise ValueError('a trunk body carries hinges its handle does not drive')
        self.pelvis = HingeChain.of_body(model, self.pelvis_body, pelvis_joints)
        self.chest = HingeChain.of_body(model, self.chest_body, chest_joints)
        upper_joints = list(range(int(model.body_jntadr[self.upper_body]),
                                  int(model.body_jntadr[self.upper_body] + model.body_jntnum[self.upper_body])))
        self.upper = HingeChain.of_body(model, self.upper_body, upper_joints)
        # each upper-body hinge follows one pelvis hinge: c_p q_p + c_u q_u held at the tendon range's middle
        import mujoco
        self.coupling = []
        for u in upper_joints:
            found = None
            for t in range(model.ntendon):
                start, n = int(model.tendon_adr[t]), int(model.tendon_num[t])
                if n != 2 or not model.tendon_limited[t]:
                    continue
                terms = [(int(model.wrap_objid[w]), float(model.wrap_prm[w])) for w in range(start, start + n)
                         if int(model.wrap_type[w]) == int(mujoco.mjtWrap.mjWRAP_JOINT)]
                if len(terms) != 2:
                    continue
                by = dict(terms)
                if u in by:
                    other = [j for j, _ in terms if j != u]
                    if other and other[0] in pelvis_joints and by[u] != 0.0:
                        found = (pelvis_joints.index(other[0]), by[other[0]], by[u],
                                 float(np.mean(model.tendon_range[t])))
            if found is None:
                raise ValueError('upper-body hinge %s has no tendon coupling to the pelvis' % model.joint(u).name)
            self.coupling.append(found)

    def upper_from_pelvis(self, q_pelvis) -> np.ndarray:
        """The upper body's hinge angles MIMo's tendons hold for these pelvis hinge angles."""
        q = np.array([(mid - c_p * q_pelvis[k]) / c_u for k, c_p, c_u, mid in self.coupling])
        return np.clip(q, self.upper.lo, self.upper.hi)

    def goal(self, root_wxyz, order) -> dict:
        """The chest's joint goal on the upper body the pelvis's ordered row implies."""
        pelvis_ordered = unit_wxyz_from_xyzw(order[self.pelvis_row, 3:7])
        q_pelvis, pelvis_off = self.pelvis.closest(root_wxyz, pelvis_ordered, self.GRID_PELVIS)
        q_upper = self.upper_from_pelvis(q_pelvis)
        # the upper body sits on the pelvis's ORDERED row, as the calf sits on the thigh's
        upper = self.upper.place(pelvis_ordered, q_upper)
        q_chest, residual = self.chest.closest(upper, unit_wxyz_from_xyzw(order[self.chest_row, 3:7]),
                                               self.GRID_CHEST)
        return dict(pelvis=q_pelvis, upper=q_upper, chest=q_chest, upper_wxyz=upper,
                    pelvis_off_reach=pelvis_off, residual=residual)

    def canonical_xyzw(self, goal, order) -> np.ndarray:
        return xyzw_aligned(self.chest.place(goal['upper_wxyz'], goal['chest']), order[self.chest_row, 3:7])

    def regenerated_xyzw(self, data, goal, reference_xyzw) -> np.ndarray:
        """The chest goal placed on the upper body where the decoder sees it."""
        return xyzw_aligned(self.chest.place(np.array(data.xquat[self.upper_body], dtype=np.float64),
                                             goal['chest']), reference_xyzw)


# ------------------------------------------------------------------ F2b: the arms
class _Arm:
    """One arm group of the decoder: its seven joints and the three member rows solved together."""

    def __init__(self, model, decoder, group):
        self.members = list(group['members'])            # [wrist, lowerarm, upperarm] row indices
        wrist = decoder.handles[self.members[0]]
        self.joint_ids = [int(j) for j in wrist.joint_ids]
        self.act_ids = [int(a) for a in wrist.act_ids]
        if not self.joint_ids or len(self.act_ids) != len(self.joint_ids):
            raise ValueError('arm group without one actuator per joint')
        self.addresses = np.array([int(model.jnt_qposadr[j]) for j in self.joint_ids])
        lo, hi = [], []
        for j in self.joint_ids:
            limited = bool(model.jnt_limited[j])
            lo.append(float(model.jnt_range[j, 0]) if limited else -np.inf)
            hi.append(float(model.jnt_range[j, 1]) if limited else np.inf)
        self.lo, self.hi = np.array(lo), np.array(hi)
        self.owners = list(group['owners'])
        self.columns = {mi: list(group['columns'][mi]) for mi in self.members}
        self.body = {mi: int(decoder.handles[mi].body_id) for mi in self.members}
        self.position = {mi: bool(decoder.use_position[mi]) for mi in self.members}
        self.reach = {mi: float(decoder.handles[mi].reach_m) for mi in self.members}
        self.mask = np.zeros(len(decoder.handle_names), dtype=bool)
        self.mask[self.members] = True

    def active(self, order, handle_mask) -> list:
        """The members the decoder solves for: unmasked and with a strength above zero."""
        return [mi for mi in self.members
                if (handle_mask is None or handle_mask[mi]) and float(order[mi, 8]) > 0.0]

    def objective(self, model, scratch, base_qpos, order, active):
        """What the decoder's stacked arm solve settles on, with its exact gradient over the seven
        joints: each active member's strength squared x (its squared remaining turn + (8 rad per metre
        x its remaining move, cut per axis at the reach clip as the decoder cuts it)^2 for the hand).
        The decoder's damped least squares stops moving where this gradient is zero."""
        import mujoco
        from .action_decoder import POSITION_WEIGHT, quat_xyzw_to_mat, rotmat_log
        targets = []
        for mi in active:
            s2 = float(np.clip(order[mi, 8], 0.0, 1.0)) ** 2
            p_t = (np.asarray(order[mi, 0:3], dtype=np.float64) / 100.0
                   if self.position[mi] and np.all(np.isfinite(order[mi, 0:3])) else None)
            targets.append((mi, s2, quat_xyzw_to_mat(order[mi, 3:7]), p_t))
        addresses, joints, weight = self.addresses, self.joint_ids, float(POSITION_WEIGHT)

        def f(q):
            scratch.qpos[:] = base_qpos
            scratch.qpos[addresses] = q
            mujoco.mj_kinematics(model, scratch)
            value, grad = 0.0, np.zeros(len(joints))
            for mi, s2, R_t, p_t in targets:
                b = self.body[mi]
                w = rotmat_log(R_t @ np.asarray(scratch.xmat[b]).reshape(3, 3).T)   # the decoder's w
                value += s2 * float(w @ w)
                for k in self.columns[mi]:
                    grad[k] += -2.0 * s2 * float(np.asarray(scratch.xaxis[joints[k]]) @ w)
                if p_t is not None:
                    p = np.asarray(scratch.xpos[b], dtype=np.float64)
                    e, r = p_t - p, self.reach[mi]
                    over = np.abs(e) > r
                    # this potential's gradient is 2 x the cut move, as the decoder's position rows
                    value += s2 * weight ** 2 * float(np.sum(np.where(over, 2.0 * r * np.abs(e) - r * r, e * e)))
                    cut = np.clip(e, -r, r)
                    for k in self.columns[mi]:
                        j = joints[k]
                        lever = np.cross(np.asarray(scratch.xaxis[j]), p - np.asarray(scratch.xanchor[j]))
                        grad[k] += -2.0 * s2 * weight ** 2 * float(lever @ cut)
            return value, grad
        return f

    def project(self, model, scratch, servo, data, order, handle_mask):
        """(joint solution, report) with the chest where it is: the decoder's own stacked arm solve,
        iterated with forward kinematics from the arm's current pose at 0.3 of each solved change
        (everything but the arm's seven joints held), then a bounded quasi-Newton polish of the
        objective it settles on (`objective`), kept only where it lowers that objective.

        `servo` is a copy of the decoder with a tiny gain and no damping, so the control it hands
        out is owner strength x gain x the joint change it solved (no rail or rescale can bind) and
        the change is read back exactly. The decoder clips each change at the joint's range."""
        import mujoco
        from scipy.optimize import minimize
        active = self.active(order, handle_mask)
        if not active:
            return None, dict(active=0)
        t0 = time.perf_counter()
        base = np.array(data.qpos, dtype=np.float64)
        q = np.clip(base[self.addresses], self.lo, self.hi)
        s_active = [float(np.clip(order[mi, 8], 0.0, 1.0)) for mi in active]
        own = np.array([float(np.clip(order[o, 8], 0.0, 1.0)) if o in active else max(s_active)
                        for o in self.owners])
        mask = self.mask if handle_mask is None else (self.mask & np.asarray(handle_mask, dtype=bool))
        scratch.qvel[:] = 0.0
        settled, steps = False, 0
        for steps in range(1, ARM_FLOW_ITERATIONS + 1):
            scratch.qpos[:] = base
            scratch.qpos[self.addresses] = q
            mujoco.mj_kinematics(model, scratch)
            ctrl = servo.decode(scratch, order, None, handle_mask=mask)
            dq = np.asarray(ctrl, dtype=np.float64)[self.act_ids] / (servo.kp * own)
            largest = float(np.max(np.abs(dq)))
            if not np.isfinite(largest):
                break
            if largest < ARM_STEP_TOLERANCE:
                settled = True
                break
            q = np.clip(q + ARM_FLOW_STEP * dq, self.lo, self.hi)
        f = self.objective(model, scratch, base, order, active)
        f_flow = f(q)[0]
        polished = False
        try:
            res = minimize(f, q, jac=True, method='L-BFGS-B',
                           bounds=[(None if not np.isfinite(a) else a, None if not np.isfinite(b) else b)
                                   for a, b in zip(self.lo, self.hi)],
                           options=dict(maxiter=300, ftol=1e-15, gtol=1e-10, maxcor=20))
            if np.all(np.isfinite(res.x)) and float(res.fun) <= f_flow:
                q, polished = np.clip(np.asarray(res.x, dtype=np.float64), self.lo, self.hi), True
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            pass
        return q, dict(active=len(active), converged=settled, flow_steps=int(steps), polished=polished,
                       objective_flow=float(f_flow), objective_end=float(f(q)[0]),
                       ms=1000.0 * (time.perf_counter() - t0))


class LimbTargetControl:
    """Environment-side projections for the calves and the chest (F2a) and the arms (F2b), and the
    ledger rows the learner records.

    execute() is called once per 10 ms frame with the order the trainer sent. A new order (a change
    in the rows the fixes read) is projected once; its joint goals and projected rows are kept in the
    returned cache until the next order, and the calf and chest rows are regenerated around their
    parents on every frame."""

    def __init__(self, model, decoder, parent_relative: str = 'off', reach_projection: str = 'off'):
        import mujoco
        if parent_relative not in PARENT_RELATIVE_CHOICES:
            raise ValueError('parent_relative_targets must be one of %s' % (PARENT_RELATIVE_CHOICES,))
        if reach_projection not in REACH_PROJECTION_CHOICES:
            raise ValueError('reach_projection must be one of %s' % (REACH_PROJECTION_CHOICES,))
        self.model = model
        self.parent_relative = parent_relative
        self.reach_projection = reach_projection
        self.handle_names = list(decoder.handle_names)
        self.calves = ([_Calf(model, decoder, t, c) for t, c in CALF_HANDLES]
                       if parent_relative in ('calf', 'calf_chest') else [])
        self.chest = _Chest(model, decoder) if parent_relative == 'calf_chest' else None
        self.arms = ([_Arm(model, decoder, g) for g in decoder.arm_groups]
                     if reach_projection == 'arms' else [])
        if reach_projection == 'arms' and len(self.arms) != 2:
            raise ValueError('arm reach projection needs both arm groups, found %d' % len(self.arms))
        self.servo = None
        if self.arms:
            self.servo = copy.copy(decoder)
            self.servo.kp, self.servo.kd = 1e-3, 0.0
        self.contract = contract_of(parent_relative, reach_projection)
        self.scratch = mujoco.MjData(model)
        rows = {r for c in self.calves for r in (c.thigh_row, c.calf_row)}
        rows |= {mi for a in self.arms for mi in a.members}
        if self.chest is not None:
            rows |= {self.chest.pelvis_row, self.chest.chest_row}
        self._key_rows = sorted(rows)

    @property
    def enabled(self) -> bool:
        return self.contract is not None

    def calf_geometry(self) -> list:
        """The knees as the planner's twin of the calf projection needs them (calf_rows_torch): each
        calf's handle rows and its knee's frame, axis, reference angle and range, read from the model.
        Empty when the calves are not projected."""
        return [dict(name=c.name, thigh_row=int(c.thigh_row), calf_row=int(c.calf_row),
                     body_quat_wxyz=[float(v) for v in _unit(c.body_quat)],
                     axis=[float(v) for v in c.axis], ref=float(c.ref), lo=float(c.lo), hi=float(c.hi))
                for c in self.calves]

    def owned(self) -> dict:
        """{row: (first, last) pose columns} this control may change in the ledger. Radius and
        strength never; a position only on the rows whose position the decoder reads."""
        out = {c.calf_row: (3, 7) for c in self.calves}
        if self.chest is not None:
            out[self.chest.chest_row] = (3, 7)
        for a in self.arms:
            for mi in a.members:
                out[mi] = (0, 7) if a.position[mi] else (3, 7)
        return out

    def _new_order(self, data, requested, handle_mask):
        import mujoco
        ledger = np.array(requested, dtype=np.float64, copy=True)
        report = dict(contract=self.contract, calf=[], chest=None, arm=[])
        knee_goals = []
        for c in self.calves:
            angle, residual = c.goal(requested)
            knee_goals.append(angle)
            ledger[c.calf_row, 3:7] = c.canonical_xyzw(requested, angle)
            report['calf'].append(dict(handle=c.name, knee_goal_deg=math.degrees(angle),
                                       off_reach_before_deg=math.degrees(residual)))
        chest_goal = None
        if self.chest is not None:
            chest_goal = self.chest.goal(np.array(data.xquat[self.chest.root_body], dtype=np.float64), requested)
            ledger[self.chest.chest_row, 3:7] = self.chest.canonical_xyzw(chest_goal, requested)
            report['chest'] = dict(chest_goal_deg=np.degrees(chest_goal['chest']).tolist(),
                                   upper_body_deg=np.degrees(chest_goal['upper']).tolist(),
                                   off_reach_before_deg=math.degrees(chest_goal['residual']),
                                   pelvis_off_reach_deg=math.degrees(chest_goal['pelvis_off_reach']))
        arm_rows = {}
        for a in self.arms:
            q, info = a.project(self.model, self.scratch, self.servo, data, requested, handle_mask)
            info['side'] = self.handle_names[a.members[0]].split('_')[-1]
            if q is not None:
                self.scratch.qpos[:] = data.qpos
                self.scratch.qpos[a.addresses] = q
                mujoco.mj_kinematics(self.model, self.scratch)
                for mi in a.members:
                    b = a.body[mi]
                    row = np.array(requested[mi], dtype=np.float64, copy=True)
                    row[3:7] = xyzw_aligned(np.array(self.scratch.xquat[b]), requested[mi, 3:7])
                    if a.position[mi]:
                        row[0:3] = np.array(self.scratch.xpos[b]) * 100.0
                    arm_rows[mi] = row
                    ledger[mi, :7] = row[:7]
                    part = self.handle_names[mi].split('_')[0]
                    info[part + '_changed_deg'] = angle_between_deg(requested[mi, 3:7], row[3:7])
                    if a.position[mi]:
                        info[part + '_moved_cm'] = float(np.linalg.norm(row[0:3] - requested[mi, 0:3]))
            report['arm'].append(info)
        return dict(contract=self.contract,
                    key=np.array(requested[self._key_rows], dtype=np.float64, copy=True),
                    mask=None if handle_mask is None else tuple(bool(x) for x in handle_mask),
                    knee_goals=knee_goals, chest_goal=chest_goal, arm_rows=arm_rows,
                    ledger_base=ledger, report=report)

    def execute(self, data, motor, requested, cache=None, handle_mask=None, hand=None):
        """(rows for the decoder this frame, cache, ledger order).

        motor: the order after the finger execution (hand_target_control) for this frame;
        requested: the order the trainer sent; hand: (HandKinematics, joint goal) when finger goals
        are on. Only the arm projection moves a wrist row, so only then are the ledger's finger rows
        placed again, on the projected wrist; otherwise they stay as sent."""
        motor = np.array(motor, dtype=np.float64, copy=True)
        requested = np.asarray(requested, dtype=np.float64)
        if requested.shape != motor.shape or requested.shape[0] != len(self.handle_names):
            raise ValueError('limb targets need one full order per frame')
        mask = None if handle_mask is None else tuple(bool(x) for x in handle_mask)
        reuse = (cache is not None and cache.get('contract') == self.contract and cache.get('mask') == mask
                 and np.array_equal(cache['key'], requested[self._key_rows]))
        if not reuse:
            cache = self._new_order(data, requested, handle_mask)
        ledger = cache['ledger_base'].copy()
        # every column the projections do not own is the trainer's order as sent
        owned = self.owned()
        for i in range(len(ledger)):
            lo, hi = owned.get(i, (0, 0))
            keep = [k for k in range(9) if not lo <= k < hi]
            ledger[i, keep] = requested[i, keep]
        if hand is not None and self.arms:
            kinematics, goal = hand
            fingers = [i for h in kinematics.geometry['hands'] for i in h['finger_indices']]
            placed = kinematics.canonical_from_joints(ledger, goal)
            ledger[fingers, :7] = placed[fingers, :7]
        for mi, row in cache['arm_rows'].items():
            motor[mi, :7] = row[:7]
        for c, angle in zip(self.calves, cache['knee_goals']):
            motor[c.calf_row, 3:7] = c.regenerated_xyzw(data, angle, requested[c.calf_row, 3:7])
        if self.chest is not None:
            motor[self.chest.chest_row, 3:7] = self.chest.regenerated_xyzw(
                data, cache['chest_goal'], requested[self.chest.chest_row, 3:7])
        return motor, cache, ledger


# ------------------------------------------------------------------ the planner's twin of the calf projection
# The planner forms the order it imagines the body receiving with LemLite.planned_execution. With the
# calf projected by the worlds (--parent-relative-targets calf or calf_chest), the body receives the
# knee goal on the thigh's ordered row, so the planner must imagine that row too (review of the limb
# fixes, 2026-09-25, finding 5; the planner trust region record: "F2b should apply the same projection
# where the planner forms its executed order"). The knee's projection is closed form and needs nothing
# but the order, so it can run inside the planner's gradient path; the chest's needs the free root's
# pose and the arms' the arm's joint angles, which the planner does not hold (read-only readings watch
# those instead: causal_chain plan_exec_* against the order the worlds recorded).
CALF_GEOMETRY_KEYS = ('name', 'thigh_row', 'calf_row', 'body_quat_wxyz', 'axis', 'ref', 'lo', 'hi')


def check_calf_geometry(geometry, handle_names) -> list:
    """The calf geometry a world reports (LimbTargetControl.calf_geometry), checked against the
    learner's handle table: both calves, their rows by name, finite numbers, a unit axis."""
    names = list(handle_names)
    if not isinstance(geometry, (list, tuple)) or len(geometry) != len(CALF_HANDLES):
        raise ValueError('calf geometry must describe both calves')
    for g, (thigh, calf) in zip(geometry, CALF_HANDLES):
        if sorted(g) != sorted(CALF_GEOMETRY_KEYS) or g['name'] != calf:
            raise ValueError('calf geometry for %s has the wrong fields' % calf)
        if thigh not in names or calf not in names or (g['thigh_row'], g['calf_row']) != (names.index(thigh),
                                                                                           names.index(calf)):
            raise ValueError('calf geometry rows do not match the learner body layout')
        numbers = np.asarray(list(g['body_quat_wxyz']) + list(g['axis']) + [g['ref'], g['lo'], g['hi']], float)
        if not np.isfinite(numbers).all() or abs(np.linalg.norm(g['axis']) - 1.0) > 1e-6 or not g['lo'] <= g['hi']:
            raise ValueError('calf geometry for %s is not a hinge' % calf)
    return [dict(g) for g in geometry]


def planner_calf_geometry(metas, parent_relative: str):
    """The calf geometry for the learner's planner, from the worlds' frames (env info
    'limb_target_geometry'), or None when the calves are not projected. Every world must report the
    same one: the planner is shared."""
    if parent_relative not in ('calf', 'calf_chest'):
        return None
    geos = [(m or {}).get('limb_target_geometry') for m in metas]
    if not geos or any(g is None for g in geos):
        raise RuntimeError('the worlds report no calf geometry for the planner although the launch projects '
                           'the calves (--parent-relative-targets %s)' % parent_relative)
    if any(g != geos[0] for g in geos):
        raise RuntimeError('the worlds report different calf geometry: the shared planner cannot follow them')
    return geos[0]['calf']


def _wxyz_unit_torch(q_xyzw):
    """A motor row's xyzw quaternion as a unit wxyz one; a zero quaternion is the identity, as the
    decoder reads it (unit_wxyz_from_xyzw)."""
    import torch
    n = q_xyzw.norm(dim=-1, keepdim=True)
    ident = torch.zeros_like(q_xyzw)
    ident[..., 3] = 1.0
    q = torch.where(n < 1e-12, ident, q_xyzw / n.clamp_min(1e-12))
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


def _qmul_torch(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    import torch
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def calf_rows_torch(cmd, geometry):
    """The planner's twin of the calf projection: every calf row of an order [..., H, 9] replaced by
    the knee goal placed on the thigh's row of the same order, as the world's ledger records it
    (_Calf.goal, then canonical_xyzw: the closed form, the range, the sign of the ordered row). Out of
    place and differentiable almost everywhere, so the planner's gradient reaches the thigh and calf
    rows through it. geometry: LimbTargetControl.calf_geometry() (empty or None: cmd unchanged)."""
    import torch
    if not geometry:
        return cmd
    out = cmd
    for g in geometry:
        t, c = int(g['thigh_row']), int(g['calf_row'])
        bq = torch.as_tensor(g['body_quat_wxyz'], dtype=cmd.dtype, device=cmd.device)
        a = torch.as_tensor(g['axis'], dtype=cmd.dtype, device=cmd.device)
        ref, lo, hi = float(g['ref']), float(g['lo']), float(g['hi'])
        base = _qmul_torch(_wxyz_unit_torch(cmd[..., t, 3:7]), bq.expand(cmd.shape[:-2] + (4,)))
        conj = base * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=cmd.dtype, device=cmd.device)
        j = _qmul_torch(conj, _wxyz_unit_torch(cmd[..., c, 3:7]))       # the knee's own turn asked for
        w, s = j[..., 0], (j[..., 1:] * a).sum(-1)
        peak = 2.0 * torch.atan2(s, w)
        cands = torch.stack([torch.clamp(peak + 2.0 * math.pi * k, lo - ref, hi - ref) for k in (0.0, -1.0, 1.0)],
                            dim=-1)
        score = (w.unsqueeze(-1) * torch.cos(cands / 2.0) + s.unsqueeze(-1) * torch.sin(cands / 2.0)).abs()
        best = torch.gather(cands, -1, score.argmax(dim=-1, keepdim=True)).squeeze(-1)   # k = 0 wins ties
        turn = torch.cat([torch.cos(best / 2.0).unsqueeze(-1), torch.sin(best / 2.0).unsqueeze(-1) * a], dim=-1)
        child = _qmul_torch(base, turn)
        row = torch.cat([child[..., 1:4], child[..., 0:1]], dim=-1)
        row = row / row.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        flip = (row * cmd[..., c, 3:7]).sum(-1, keepdim=True) < 0.0
        row = torch.where(flip, -row, row)
        new_row = torch.cat([out[..., c, 0:3], row, out[..., c, 7:]], dim=-1).unsqueeze(-2)
        out = torch.cat([out[..., :c, :], new_row, out[..., c + 1:, :]], dim=-2)
    return out


# ------------------------------------------------------------------ the trainer's side
def contract_of(parent_relative: str, reach_projection: str) -> Optional[str]:
    """The limb contract a world reports under these launch settings (None = all off)."""
    parts = []
    if parent_relative in ('calf', 'calf_chest'):
        parts.append(CALF_CONTRACT)
    if parent_relative == 'calf_chest':
        parts.append(CHEST_CONTRACT)
    if reach_projection == 'arms':
        parts.append(ARM_CONTRACT)
    return '+'.join(parts) if parts else None


def action_contract_parts(meta: dict) -> list:
    """The parts of a frame's reports that change what the ledger's rows mean (the transition's
    action_contract, after the fingers'): the parent-relative calf and chest, then the head and pelvis
    position rows no longer read. The arm projection keeps world targets and adds none."""
    meta = meta or {}
    limb = str(meta.get('limb_target_contract')).split('+')
    parts = [p for p in (CALF_CONTRACT, CHEST_CONTRACT) if p in limb]
    if meta.get('position_rows') == 'no_head_pelvis':
        parts.append(POSITION_ROWS_CONTRACT)
    from .action_decoder import FINGER_SERVO_CONTRACT
    if meta.get('finger_servo') == FINGER_SERVO_CONTRACT:
        # --finger-joint-servo on: the finger rows' goal was servoed hinge by hinge (last, as in
        # transition.ACTION_CONTRACT_PARTS)
        parts.append(FINGER_SERVO_CONTRACT)
    return parts


def moment_contract(metas) -> int:
    """replay's action_contract for the order the last burst executed, from its frames' reports:
    bit 0 (1) wrist-relative finger goals, bit 1 (2) parent-relative calf goals, bit 2 (4) the
    chest's, bit 3 (8) no head or pelvis position rows, bit 5 (32) finger hinges servoed toward their
    own goals (--finger-joint-servo on). With every fix off this is the finger bit alone, exactly what
    lem_model stored before they existed."""
    from .action_decoder import FINGER_SERVO_CONTRACT
    metas = list(metas)
    hand = all(m.get('hand_target_contract') == 'wrist_relative_joint_v1' for m in metas)
    parts = [str(m.get('limb_target_contract')).split('+') for m in metas]
    calf = all(CALF_CONTRACT in p for p in parts)
    chest = all(CHEST_CONTRACT in p for p in parts)
    rows = all(m.get('position_rows') == 'no_head_pelvis' for m in metas)
    servo = bool(metas) and all(m.get('finger_servo') == FINGER_SERVO_CONTRACT for m in metas)
    return (int(hand) | (CALF_CONTRACT_BIT if calf else 0) | (CHEST_CONTRACT_BIT if chest else 0)
            | (POSITION_ROWS_CONTRACT_BIT if rows else 0) | (FINGER_SERVO_CONTRACT_BIT if servo else 0))


def replay_contract(hand_target_mode: str, parent_relative: str, position_rows: str = 'all',
                    eye_row_convention: str = 'legacy', finger_joint_servo: str = 'off') -> Optional[int]:
    """The action_contract replay windows must carry for this learner (None = any), so a window
    never mixes orders recorded under two meanings of the finger, calf, chest, head or pelvis rows,
    or of the eye row (EYE_ROW_CONTRACT_BIT under 'physical'), or two servo rules for the fingers
    (FINGER_SERVO_CONTRACT_BIT under --finger-joint-servo on)."""
    bits = 0 if hand_target_mode == 'world' else 1
    if parent_relative in ('calf', 'calf_chest'):
        bits |= CALF_CONTRACT_BIT
    if parent_relative == 'calf_chest':
        bits |= CHEST_CONTRACT_BIT
    if position_rows == 'no_head_pelvis':
        bits |= POSITION_ROWS_CONTRACT_BIT
    if eye_row_convention == 'physical':
        bits |= EYE_ROW_CONTRACT_BIT
    if finger_joint_servo == 'on':
        bits |= FINGER_SERVO_CONTRACT_BIT
    return bits or None


def check_finger_servo(metas, finger_joint_servo: str = 'off') -> None:
    """Refuse worlds that servo the fingers by another rule than this launch asked for (a world reports
    'finger_servo' only under --finger-joint-servo on), in either direction, as check_position_rows does."""
    from .action_decoder import FINGER_SERVO_CONTRACT
    reported = sorted({'on' if m.get('finger_servo') == FINGER_SERVO_CONTRACT else 'off' for m in metas})
    if reported != [finger_joint_servo]:
        raise RuntimeError('the worlds servo the fingers with --finger-joint-servo %s where this launch asked for %s: '
                           'a resumed world keeps its saved settings unless --resume-env-settings launch is passed'
                           % (reported, finger_joint_servo))


def check_position_rows(metas, position_rows: str = 'all') -> None:
    """Refuse worlds whose decoder reads other position rows than this launch asked for (a world
    reports position_rows only when it is not 'all'), in either direction, as ledger_from_frames
    refuses a mismatched limb contract."""
    reported = sorted({str(m.get('position_rows', 'all')) for m in metas})
    if reported != [position_rows]:
        raise RuntimeError('the worlds solve position rows %s where this launch asked for %s: a resumed '
                           'world keeps its saved settings unless --resume-env-settings launch is passed'
                           % (reported, position_rows))


def ledger_from_frames(metas: Sequence[dict], sent, handle_names, parent_relative: str = 'off',
                       reach_projection: str = 'off', finger_rows: Sequence[int] = ()):
    """The body rows the learner's ledger records for one burst, or None when every fix is off.

    metas: the last frame's meta of every body (the ledger is fixed per order, so every frame of
    the burst carries the same one); sent: [B, H, 9] the order the trainer sent; finger_rows: the
    finger rows under wrist-relative finger goals (placed again on a projected wrist). Refuses a
    world that runs a different limb contract than this launch asked for, in either direction: a
    world projecting orders the ledger does not record (a resume that kept projecting worlds under a
    command without the flags) would teach the learner the wrong cause, and so would the reverse.
    Checks that the world changed nothing but the rows the projections own."""
    expected_contract = contract_of(parent_relative, reach_projection)
    reported = [m.get('limb_target_contract') for m in metas]
    if expected_contract is None:
        if any(r is not None for r in reported):
            raise RuntimeError('the worlds run limb projections (%s) this launch did not ask for; '
                               'pass the same --parent-relative-targets/--reach-projection as the run '
                               'that turned them on' % sorted({str(r) for r in reported}))
        return None
    if any(r != expected_contract for r in reported):
        raise RuntimeError('the worlds report limb contract %s where this launch asked for %s: a '
                           'resumed world keeps its saved settings unless --resume-env-settings '
                           'launch is passed' % (sorted({str(r) for r in reported}), expected_contract))
    sent = np.asarray(sent, dtype=np.float64)
    ledger = np.stack([np.asarray(m['ledger_motor_targets'], dtype=np.float64) for m in metas])
    if ledger.shape != sent.shape or not np.isfinite(ledger).all():
        raise RuntimeError('the worlds returned a ledger order of the wrong shape or with non-finite values')
    names = list(handle_names)
    owned = {}
    if parent_relative in ('calf', 'calf_chest'):
        for _, calf in CALF_HANDLES:
            owned[names.index(calf)] = (3, 7)
    if parent_relative == 'calf_chest':
        owned[names.index('chest')] = (3, 7)
    if reach_projection == 'arms':
        from .action_decoder import POSITION_HANDLES
        for side in ('L', 'R'):
            for part in ('wrist', 'lowerarm', 'upperarm'):
                name = part + '_' + side
                owned[names.index(name)] = (0, 7) if name in POSITION_HANDLES else (3, 7)
        for i in finger_rows:
            owned[int(i)] = (0, 7)
    for i in range(sent.shape[1]):
        lo, hi = owned.get(i, (0, 0))
        keep = [k for k in range(9) if not lo <= k < hi]
        if not np.array_equal(ledger[:, i, keep], sent[:, i, keep]):
            raise RuntimeError('the world changed row %s of the order outside the limb projections'
                               % names[i])
    return ledger


def window_add(window: dict, reports, author: Optional[str] = None) -> None:
    """Add one latch's reports (one per body; None where a body reported none) to a stats window.

    `window` is a dict the trainer empties at every stats record. author: who wrote the order
    (planner, reflex_noise, reflex, ...; the trainer's held_author): the same readings are kept per
    author too, so how far the PLANNER's own orders are moved can be read apart (review of 2026-09-25,
    findings 4 and 5: the planner still imagines the chest's and the arms' rows as it ordered them)."""
    for r in reports:
        if not r:
            continue
        own = window.setdefault('by_author', {}).setdefault(str(author or 'unknown'), {}) if author is not None else None
        for w in (window, own):
            if w is None:
                continue
            w['orders'] = w.get('orders', 0) + 1
            for c in r.get('calf') or ():
                w.setdefault('calf_off_reach_before_deg', []).append(float(c['off_reach_before_deg']))
            chest = r.get('chest')
            if chest:
                w.setdefault('chest_off_reach_before_deg', []).append(float(chest['off_reach_before_deg']))
            for a in r.get('arm') or ():
                if not a.get('active'):
                    continue
                w['arm_orders'] = w.get('arm_orders', 0) + 1
                w['arm_flow_not_settled'] = w.get('arm_flow_not_settled', 0) + int(not a.get('converged'))
                for key in ('wrist_changed_deg', 'wrist_moved_cm', 'lowerarm_changed_deg', 'upperarm_changed_deg'):
                    if key in a:
                        w.setdefault(key, []).append(float(a[key]))
                w.setdefault('arm_solve_ms', []).append(float(a.get('ms', 0.0)))


def window_summary(window: dict) -> Optional[dict]:
    """The stats record's limb_targets block for one window (medians and 90th percentiles)."""
    if not window:
        return None

    def q(key, src=None):
        v = np.asarray((window if src is None else src).get(key) or [], dtype=np.float64)
        v = v[np.isfinite(v)]
        if not len(v):
            return None
        return dict(p50=round(float(np.median(v)), 3), p90=round(float(np.quantile(v, 0.9)), 3), n=int(len(v)))
    by_author = {au: dict(body_orders=int(w.get('orders', 0)),
                          calf_off_reach_before_deg=q('calf_off_reach_before_deg', w),
                          chest_off_reach_before_deg=q('chest_off_reach_before_deg', w),
                          hand_turned_deg=q('wrist_changed_deg', w), hand_moved_cm=q('wrist_moved_cm', w),
                          forearm_turned_deg=q('lowerarm_changed_deg', w),
                          upper_arm_turned_deg=q('upperarm_changed_deg', w))
                 for au, w in sorted((window.get('by_author') or {}).items())}
    return dict(
        by_author=by_author or None,
        body_orders=int(window.get('orders', 0)),
        calf_off_reach_before_deg=q('calf_off_reach_before_deg'),
        chest_off_reach_before_deg=q('chest_off_reach_before_deg'),
        arm_orders=int(window.get('arm_orders', 0)),
        arm_flow_not_settled=int(window.get('arm_flow_not_settled', 0)),
        hand_turned_deg=q('wrist_changed_deg'), hand_moved_cm=q('wrist_moved_cm'),
        forearm_turned_deg=q('lowerarm_changed_deg'), upper_arm_turned_deg=q('upperarm_changed_deg'),
        arm_solve_ms=q('arm_solve_ms'),
        meaning=("latched orders only, one per body. calf_ and chest_off_reach_before_deg: how far the "
                 "ordered row lay from every pose the part's own joints can make on its parent's ordered "
                 "pose, before it was replaced by the closest one (0 = the order was performable already). "
                 "hand_*, forearm_*, upper_arm_*: how far the arm projection moved each row to the pose the "
                 "decoder's arm solve settles on (0 = the order was one pose of the arm already); "
                 "arm_flow_not_settled counts arm solves whose iteration was still moving after 400 steps "
                 "before the polish (their rows are still one pose of the arm)"))
