# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Proprioception adapter: 23 tracked bones from MIMo bodies.

The model wants 23 bones in the SAME order as its 23 handles (lem/model/body_sets.py
HANDLE_NAMES; the C++ TrackedBoneNames use that order too). Per bone the frame holds
quat xyzw (world), position cm (world; the tokenizer makes it relative to bone 0 =
wrist_L), velocity cm/s (world), fatigue. UE writes the body set "capacity" (1.0 =
fresh) into the fatigue field, and we do the same.

MIMo facts used (verified against mimo_infant and MuJoCo):
  data.xquat is w,x,y,z   -> we reorder to x,y,z,w
  data.xpos is metres     -> x100 for cm
  mj_objectVelocity gives [rot(3), lin(3)] in the world frame at the body origin (m/s)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from lem.model.body_sets import HANDLE_NAMES, HANDLE_BODY_SET, BodySet
from mimo_bridge.touch_adapter import bone_hash

# handle / tracked-bone name -> MIMo v2 body name  (order = HANDLE_NAMES order)
BONE_MAP: Dict[str, str] = {
    "wrist_L": "left_hand", "thumb_L": "left_thdistal", "index_L": "left_ffdistal",
    "middle_L": "left_mfdistal", "ring_L": "left_rfdistal", "pinky_L": "left_lfdistal",
    "wrist_R": "right_hand", "thumb_R": "right_thdistal", "index_R": "right_ffdistal",
    "middle_R": "right_mfdistal", "ring_R": "right_rfdistal", "pinky_R": "right_lfdistal",
    "head": "head",
    "upperarm_L": "left_upper_arm", "lowerarm_L": "left_lower_arm",
    "upperarm_R": "right_upper_arm", "lowerarm_R": "right_lower_arm",
    "thigh_L": "left_upper_leg", "calf_L": "left_lower_leg",
    "thigh_R": "right_upper_leg", "calf_R": "right_lower_leg",
    "foot_L": "left_foot", "foot_R": "right_foot",
    # the eyes are a tracked bone like any other: felt, not bolted on. One eye body
    # stands for both (they are yoked); its orientation IS the gaze direction.
    "gaze": "left_eye",
}
assert tuple(BONE_MAP.keys()) == tuple(HANDLE_NAMES), "BONE_MAP must follow HANDLE_NAMES order"
N_BONES = len(HANDLE_NAMES)  # 24 (23 body bones + the gaze handle)

BODY_SET_ORDER: List[BodySet] = list(BodySet)   # Arm_L, Arm_R, Hand_L, Hand_R, Spine, Head, Leg_L, Leg_R, Foot_L, Foot_R


def body_set_for_mimo_body(name: str) -> Optional[BodySet]:
    """Which of the model's 10 body sets a MIMo body belongs to (None = not tracked)."""
    side = "L" if name.startswith("left_") else ("R" if name.startswith("right_") else None)
    base = name.split("_", 1)[1] if side else name
    if side:
        if base in ("hand",) or base.endswith(("distal", "middle", "knuckle", "metacarpal", "thbase", "thhub")):
            return BodySet.Hand_L if side == "L" else BodySet.Hand_R
        if base in ("upper_arm", "lower_arm"):
            return BodySet.Arm_L if side == "L" else BodySet.Arm_R
        if base in ("upper_leg", "lower_leg"):
            return BodySet.Leg_L if side == "L" else BodySet.Leg_R
        if base in ("foot", "toes", "big_toe"):
            return BodySet.Foot_L if side == "L" else BodySet.Foot_R
        if base in ("eye",):
            return BodySet.Head
        return None
    if name in ("head",):
        return BodySet.Head
    if name in ("hip", "lower_body", "upper_body", "chest"):
        return BodySet.Spine
    return None


class FatigueModel:
    """Per body set capacity in [0, 1] (1 = fresh), driven by mechanical effort.

    dcap/dt = recover * (1 - cap) - drain * effort
    effort  = ProprioAdapter.effort_per_set, divided by p_ref. Under effort_power no_gear (the
              default and every run so far) that is |actuator force x joint speed| summed over
              the set's actuators: MIMo's actuator force is taken BEFORE its gear, so it is not
              torque x speed and not watts, and every "W" below (p_ref, the 0.82 to 1.01 W of
              a babbling hand) is in that formula's unit. Under with_gear it is |actuator force
              x gear x joint speed|, watts (review env-senses MINOR 6; corrected 2026-09-23 by
              verify-fix-env: this line used to call the default "|torque x joint speed| (W)").
    Steady state with constant effort E: cap = 1 - (drain / recover) * E.
    Defaults: recover 0.05 /s, drain 0.02 /s, p_ref 1 W -> E = 1 W gives cap 0.6 at rest state.
    UE's own capacity logic is not copied.

    WHAT THE RATES DO ON THIS BODY, measured 2026-09-21 on the crib body at 24 months, so the
    constants carry their anchors rather than standing as a guess:
      - the effort a hand set spends under motor babble is a middle 0.82 to 1.01 W, the same
        size as the 1 W reference, so the reference is on this body's own scale;
      - 30 s of babble leaves the hands at 0.829 of full strength, and the steady state that
        babble implies is 0.60 to 0.67;
      - RECOVERY IS REACHABLE AND QUICK: resting, with no command at all, the hands are back
        to 0.90 of full strength after 10.7 s, 0.95 after 24.6 s and 0.99 after 56.8 s, and a
        resting hand spends 0.0000 W, so nothing holds the capacity down while the body rests.
    That closes the homeostatic self receipt's open item, "establish reachable recovery under
    the same limits" (docs/Homeostatic_Self_Implementation_Receipt_2026-09-12.html). A depleted
    body is the design - with body_function set to modulate, capacity scales actuator gain and
    force limits so the model has a body whose condition it can manage - and a body that could
    not recover would have been a trap. On the H84 run the hands read 0.46 to 0.58: lower than
    babble alone produces, which says the policy works the hands harder than random babble and
    does not rest them, a behaviour and not a fault of these rates. Whether the rates match an
    infant's is not measured; they are anchored to this body's own effort and recovery.
    """

    def __init__(self, n_sets: int = 10, recover: float = 0.05, drain: float = 0.02, p_ref: float = 1.0):
        self.cap = np.ones(n_sets, dtype=np.float64)
        self.accum = np.zeros(n_sets, dtype=np.float64)   # accumulated energy (J), never reset
        self.recover = recover
        self.drain = drain
        self.p_ref = p_ref

    def step(self, effort_w: np.ndarray, dt: float) -> None:
        e = np.asarray(effort_w, dtype=np.float64) / self.p_ref
        self.cap += dt * (self.recover * (1.0 - self.cap) - self.drain * e)
        np.clip(self.cap, 0.0, 1.0, out=self.cap)
        self.accum += np.asarray(effort_w, dtype=np.float64) * dt

    def reset(self):
        self.cap[:] = 1.0
        self.accum[:] = 0.0


@dataclass
class ProprioReadout:
    bones: np.ndarray          # (23, 11) quat xyzw, pos cm, vel cm/s, fatigue(capacity)
    hashes: List[int]
    energy: np.ndarray         # (3,) ph_work J, pcc_holding J, total J  (this step)
    body_sets: np.ndarray      # (10, 2) capacity, accumulated energy J


# HOW THE EFFORT THAT DRAINS THE FATIGUE MODEL IS COMPUTED (BridgeEnv keyword effort_power).
#   no_gear    |actuator force x joint speed| summed per body set: the formula every run up to
#              and including H84 segment 31 used. MIMo's actuator force is the motor's own
#              output before its gear (its control range is -1 to 1), so this leaves the gear out
#              and is not in watts.
#   with_gear  |actuator force x gear x joint speed| per actuator: the torque the joint really
#              receives times its speed, mechanical power in watts. On the segment 31 body (age
#              24 months) the gears run from 0.0026 N m (a right little-finger joint) to 12.62 N m
#              (a knee) of joint torque per unit of motor force, 4,890 times apart (measured
#              2026-09-23 by verify-fix-env; the review's 0.0037 to 10.58 is MIMo's XML before the
#              age scaling), so the two formulas weigh the body sets differently, not by one
#              common factor.
# The default is no_gear: switching a lineage to with_gear changes how fast each body set tires
# and, under body_function modulate, the force its motors can give - a change to the world the
# body lives in, which is the operator's decision (fix-env log, "FOR THE OPERATOR").
EFFORT_POWER_MODES = ("no_gear", "with_gear")


class ProprioAdapter:
    """Reads the 23 tracked bones out of a MuJoCo model/data pair."""

    def __init__(self, model, data, body_layout='legacy', effort_power: str = 'no_gear'):
        import mujoco
        if effort_power not in EFFORT_POWER_MODES:
            raise ValueError(f"effort_power must be one of {EFFORT_POWER_MODES}, not {effort_power!r}")
        # a constructor setting, deliberately NOT on self.fatigue: restore_full_state replaces
        # the fatigue model's whole __dict__ from the snapshot, which would drop it
        self.effort_power = effort_power
        self._mj = mujoco
        self.model = model
        self.data = data
        from .body_layout import names_for, TRUNK_BODIES
        self.handle_names = names_for(body_layout)
        body_map = {**BONE_MAP, **TRUNK_BODIES}
        self.body_ids = np.array([model.body(body_map[h]).id for h in self.handle_names], dtype=np.int32)
        self.hashes = [bone_hash(body_map[h]) for h in self.handle_names]
        self.handle_sets = [BODY_SET_ORDER.index(bs) for bs in HANDLE_BODY_SET]
        self.handle_sets += [BODY_SET_ORDER.index(BodySet.Spine)] * (len(self.handle_names)-len(HANDLE_NAMES))
        # actuator -> body set (by the actuated joint's body)
        self.act_set = np.full(model.nu, -1, dtype=np.int32)
        for a in range(model.nu):
            j = int(model.actuator_trnid[a, 0])
            b = int(model.jnt_bodyid[j])
            bs = body_set_for_mimo_body(model.body(b).name)
            if bs is not None:
                self.act_set[a] = BODY_SET_ORDER.index(bs)
        self.act_dof = np.array([int(model.jnt_dofadr[int(model.actuator_trnid[a, 0])]) for a in range(model.nu)])
        # each actuator's gear (joint torque per unit of actuator force), for effort_power
        # with_gear; read once, MIMo never changes a gear during a run
        self.act_gear = np.asarray(model.actuator_gear[:, 0], dtype=np.float64).copy()
        self.fatigue = FatigueModel(len(BODY_SET_ORDER))
        self._vel6 = np.zeros(6, dtype=np.float64)

    def effort_per_set(self) -> np.ndarray:
        """Effort summed per body set: |actuator force x joint speed| (effort_power no_gear, the
        default and every run so far) or |actuator force x gear x joint speed| in watts
        (with_gear). See EFFORT_POWER_MODES."""
        tau = np.asarray(self.data.actuator_force, dtype=np.float64)
        qd = np.asarray(self.data.qvel, dtype=np.float64)[self.act_dof]
        if self.effort_power == "with_gear":
            p = np.abs(tau * self.act_gear * qd)
        else:
            p = np.abs(tau * qd)
        out = np.zeros(len(BODY_SET_ORDER), dtype=np.float64)
        for a in range(self.model.nu):
            s = self.act_set[a]
            if s >= 0:
                out[s] += p[a]
        return out

    def read(self, dt: float) -> ProprioReadout:
        m, d = self.model, self.data
        effort = self.effort_per_set()
        self.fatigue.step(effort, dt)
        bones = np.zeros((len(self.handle_names), 11), dtype=np.float32)
        for i, b in enumerate(self.body_ids):
            q = d.xquat[b]                              # w x y z
            bones[i, 0:4] = (q[1], q[2], q[3], q[0])     # x y z w
            bones[i, 4:7] = d.xpos[b] * 100.0            # cm
            self._mj.mj_objectVelocity(m, d, self._mj.mjtObj.mjOBJ_BODY, int(b), self._vel6, 0)
            bones[i, 7:10] = self._vel6[3:6] * 100.0     # cm/s, world frame
            bones[i, 10] = self.fatigue.cap[self.handle_sets[i]]
        # THE GAZE ROW IS HEAD-RELATIVE (B8 fix, 2026-08-26). Handle 23 drives the
        # eye SERVO, whose consumer (_gaze_from_handle) reads the quaternion's yaw
        # and pitch as eye-joint angles and clamps them to the joint range
        # (+/-0.785 az, -0.820..0.576 el). The row used to carry the eye body's
        # WORLD orientation - mostly head pose - whose |yaw| measured 1.09-2.43 rad,
        # outside the entire joint range on 99-100% of frames. So the motor loss
        # taught the controller a target it could never reach, the clamp pinned the
        # commanded azimuth at +0.785 on 98.3% of steps, and the eye actuator sat
        # on a rail. Head-relative, the same numbers land inside the range and the
        # clamp stops binding. xyzw convention: rel = conj(head) * eye.
        gi, hi_ = self.handle_names.index('gaze'), self.handle_names.index("head")
        hx, hy, hz, hw = bones[hi_, 0:4]
        ex, ey, ez, ew = bones[gi, 0:4]
        cx, cy, cz, cw = -hx, -hy, -hz, hw          # conjugate of the head quat
        bones[gi, 0] = cw * ex + cx * ew + cy * ez - cz * ey
        bones[gi, 1] = cw * ey - cx * ez + cy * ew + cz * ex
        bones[gi, 2] = cw * ez + cx * ey - cy * ex + cz * ew
        bones[gi, 3] = cw * ew - cx * ex - cy * ey - cz * ez
        # energy for the motor slice: ph_work = mechanical work this step, pcc_holding = holding
        # cost (|torque| x dt x 0.01 W per Nm, a guess written down), total = sum
        # UNITS (verify-fix-env 2026-09-23): ph_work is effort_per_set x dt, so it is joules only
        # under effort_power with_gear; under the default no_gear it carries the gear-less
        # formula's unit. pcc_holding sums |actuator force| - before the gear, not joint torque -
        # under EITHER setting (on the review's random-control frames the geared sum was 1.9
        # times the gear-less one, median of 200 frames); left as it is, surfaced in the log.
        tau = np.asarray(d.actuator_force, dtype=np.float64)
        ph_work = float(np.sum(effort) * dt)
        pcc_hold = float(np.sum(np.abs(tau)) * dt * 0.01)
        energy = np.array([ph_work, pcc_hold, ph_work + pcc_hold], dtype=np.float32)
        body_sets = np.column_stack([self.fatigue.cap, self.fatigue.accum]).astype(np.float32)
        return ProprioReadout(bones=bones, hashes=self.hashes, energy=energy, body_sets=body_sets)

    def reset(self):
        self.fatigue.reset()
