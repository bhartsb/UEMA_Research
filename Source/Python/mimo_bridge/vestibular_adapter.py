# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The vestibular / internal-state vector: balance, neck, support, effort and limits.

What it is, in plain words. MIMo's head carries an accelerometer and a gyro (the inner
ear's two sensors), and until 2026-09-22 the bridge read those six numbers every frame and
nothing ever used them. This module turns them, and a few other body facts the model had no
way to feel, into one short vector per frame:

  * which way gravity pulls in the head (accelerometer / 9.81, so 1.0 = one g) and how fast
    the head turns (gyro);
  * the neck: its three joint angles on each joint's own range (-1 one end, +1 the other),
    how fast they move, and how hard the neck motors push (-1 to +1 of full strength);
  * which way is up for the trunk (world-up seen from the chest: graviception);
  * what holds the body up: the share of outside contact force carried by the head, the
    trunk, the arms and the legs (sums to 1 whenever anything outside touches the body),
    and the total outside force in body weights (1.0 = the floor carries the whole body);
  * effort: mechanical power of all motors, and the isometric load (the summed motor torque
    that holding a pose costs even when nothing moves);
  * the share of motor-driven joints within MIMo's own 0.035 rad band of a joint limit;
  * the lowest and the mean motor capacity of the fatigue model (1 = fresh, 0 = exhausted);
  * a validity flag: 1 on a real read, 0 on a vector padded from the old six numbers.

Contract (docs/gain_of_function_v3/DESIGN.md section 7): VESTIBULAR_FIELDS order is binding,
read_vestibular(env) is computed UNCONDITIONALLY on every collect frame (a resumed world is
rebuilt from its saved constructor arguments, so a new constructor switch would never reach
it), the vector rides RawStep.vestibular / BridgeFrame.vestibular, and a reader handed an
old (6,) vector pads it with pad_legacy_vestibular.

Every scale below was MEASURED on the H84 crib body (see the block above each constant and
Saved/mimo_bridge/scale_anchors.json key "vestibular_scales"); each is set so that the 99th
percentile of what this body produces reads about 1.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

VESTIBULAR_FIELDS = (
    "head_acc_x", "head_acc_y", "head_acc_z",
    "head_gyro_x", "head_gyro_y", "head_gyro_z",
    "neck_swivel", "neck_tilt", "neck_tilt_side",
    "neck_swivel_vel", "neck_tilt_vel", "neck_tilt_side_vel",
    "neck_swivel_effort", "neck_tilt_effort", "neck_tilt_side_effort",
    "trunk_up_x", "trunk_up_y", "trunk_up_z",
    "support_head", "support_trunk", "support_arms", "support_legs",
    "support_total",
    "effort_power", "effort_holding",
    "limit_load",
    "capacity_min", "capacity_mean",
    "valid",
)
VESTIBULAR_DIM = len(VESTIBULAR_FIELDS)
FIELD_INDEX: Dict[str, int] = {name: i for i, name in enumerate(VESTIBULAR_FIELDS)}
LEGACY_VESTIBULAR_DIM = 6          # MIMo's accelerometer (3) then gyro (3), raw m/s2 and rad/s

NECK_JOINTS = ("robot:head_swivel", "robot:head_tilt", "robot:head_tilt_side")
NECK_ACTUATORS = ("act:head_swivel", "act:head_tilt", "act:head_tilt_side")
SUPPORT_GROUPS = ("head", "trunk", "arms", "legs")

# One g. Fixed by the contract rather than measured: 1.0 then means "gravity alone", which
# is the lived reference a tilt is read against.
GRAVITY_MS2 = 9.81

# MIMo's own limit band (mimo_infant/proprioception/proprio.py: limit_thresh .035 rad, about
# 2 degrees; DEFAULT_PROPRIOCEPTION_PARAMS threshold .035). A joint closer than this to either
# end of its range is "at its limit" by MIMo's definition, not ours.
LIMIT_BAND_RAD = 0.035

# ----------------------------------------------------------------------------------------
# MEASURED SCALES (2026-09-22, env-senses agent, CPU). The H84 segment 31 crib world, built
# the way the trainer builds it (train_mimo.env_kwargs_for on that run's args.json: age 24
# months, mechanical_v4 skin, body_function modulate, trunk_v1, decoder kp 6, finger armature
# 0.0004, caregiver responsive with recording off, mobile and critters on; vision off, which
# moves nothing but the eyes), one body. 18,450 env frames (10 ms each) pooled from:
#   babble    15,000 frames: the run's motor babble (5 cm position and 0.5 rad turn noise per
#             handle, strength 0.47 + |N(0, 0.05)| - 0.47 is segment 31's executed mean -
#             smoothed 0.35 and held 5 model steps, as the trainer does)
#   side/prone 3 x (300 settling + 500 babbling) frames placed on each side and face down
#   rolls     2 x 300 frames rolled by a 12 N m torque about the long axis
#   drops     3 x 150 frames dropped from 10 cm (on the back, face down, on the side)
# Each scale is the pooled 99th percentile, rounded; so the 99th percentile reads about 1.
# Babble alone reads lower (its own 99th percentile in brackets, on the scale below); rolls and
# drops read higher, which is what they are.
GYRO_SCALE_RADS = 2.0             # head turning speed, per axis: p99 1.97 rad/s (babble 0.67 of it)
NECK_VEL_SCALE_RADS = 1.5         # neck joint speed: p99 1.47 rad/s (babble 0.90)
EFFORT_POWER_SCALE_W = 66.0       # all motors' mechanical power: p99 66.1 W (babble 0.90)
EFFORT_HOLDING_SCALE_NM = 63.0    # summed motor torque: p99 62.6 N m (babble 0.95)
MEASUREMENT: Dict[str, object] = {
    "date": "2026-09-22",
    "source": ("scratchpad measure_vestibular.py (env-senses agent) -> "
               "vestibular_measurement_v3.json; world = train_mimo.env_kwargs_for(H84 seg31 "
               "args.json), 1 body, CPU, vision off, caregiver recording off"),
    "frames": {"babble": 15000, "side_prone_settle_babble": 2400, "rolls": 600, "drops": 450,
               "pooled": 18450},
    # absolute value per axis / per joint, physical units
    "pooled": {
        "gyro_rads": {"p50": 0.273, "p90": 0.833, "p99": 1.970, "p99.9": 6.213, "max": 10.75},
        "neck_vel_rads": {"p50": 0.296, "p90": 0.863, "p99": 1.466, "p99.9": 2.379, "max": 4.964},
        "power_w": {"p50": 17.65, "p90": 37.90, "p99": 66.12, "p99.9": 116.1, "max": 135.3},
        "holding_nm": {"p50": 36.35, "p90": 48.32, "p99": 62.61, "p99.9": 79.02, "max": 97.08},
        "acc_g_per_axis": {"p50": 0.462, "p90": 0.943, "p99": 1.742, "p99.9": 2.815, "max": 41.20},
    },
    "babble_only": {
        "gyro_rads": {"p99": 1.330, "max": 2.662},
        "neck_vel_rads": {"p99": 1.357, "max": 2.175},
        "power_w": {"p99": 59.09, "max": 105.9},
        "holding_nm": {"p99": 59.97, "max": 79.48},
        "acc_g_per_axis": {"p99": 1.762, "max": 3.987},
    },
    # the head accelerometer is NOT scaled by measurement (the contract fixes 1.0 = one g);
    # its tail is recorded so the model side can decide how to bound it: lying still reads
    # 1.00 g, babbling on the back reaches 2.4 g (99th percentile of the vector's size), and a
    # 10 cm drop onto the back peaks at 41.6 g for one frame (MuJoCo's stiff contact).
    "acc_norm_g": {"rest_p99": 1.00, "babble_p99": 2.40, "babble_max": 4.18,
                   "roll_max": 16.2, "drop_supine_max": 41.6, "drop_side_max": 12.4},
    # what else the vector reads on the same frames
    "support_total_body_weights": {"rest_p50": 1.00, "babble_p50": 1.01, "babble_p99": 3.57,
                                   "drop_supine_max": 18.3},
    "limit_load_share": {"rest_p99": 0.048, "babble_p99": 0.119, "roll_p99": 0.238},
    "cost_ms_per_collect": {"read_vestibular_mean": 0.086, "read_vestibular_p99": 0.206,
                            "collect_mean": 7.35},
}


def _group_of_body(name: str) -> int:
    """Which support group (index into SUPPORT_GROUPS) a body belongs to; -1 for none.

    Uses the skin's own region names (touch_adapter.SKIN_REGIONS) so "arms" and "legs" mean
    the same parts the skin report already names. The eyes are not skin patches but they
    are part of the head, so a contact on an eye is carried by the head."""
    from mimo_bridge.touch_adapter import region_of
    region = region_of(name)
    if region == "head" or name.endswith("_eye"):
        return 0
    if region == "trunk":
        return 1
    if "hand" in region or "arm" in region:
        return 2
    if "leg" in region or "foot" in region:
        return 3
    return -1


class _Layout:
    """Model indices, looked up once per world (names never change inside one model)."""

    def __init__(self, env):
        import mujoco
        m = env.model
        self.acc_adr = self._sensor_adr(m, "vestibular_acc")
        self.gyro_adr = self._sensor_adr(m, "vestibular_gyro")
        self.neck_qpos = np.array([int(m.jnt_qposadr[m.joint(n).id]) for n in NECK_JOINTS])
        self.neck_dof = np.array([int(m.jnt_dofadr[m.joint(n).id]) for n in NECK_JOINTS])
        self.neck_range = np.array([m.jnt_range[m.joint(n).id] for n in NECK_JOINTS], dtype=np.float64)
        self.neck_act = np.array([int(m.actuator(n).id) for n in NECK_ACTUATORS])
        self.chest = int(m.body("chest").id)
        self.gear = np.asarray(m.actuator_gear[:, 0], dtype=np.float64).copy()
        # joints driven by a motor, eyes excluded: an eye resting at the edge of its range
        # is where the gaze servo was told to look, not a strained body
        hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
        joints = []
        for a in range(m.nu):
            j = int(m.actuator_trnid[a, 0])
            if (int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
                    and int(m.jnt_type[j]) == hinge and bool(m.jnt_limited[j])
                    and "_eye_" not in (m.joint(j).name or "") and j not in joints):
                joints.append(j)
        self.limit_qpos = np.array([int(m.jnt_qposadr[j]) for j in joints], dtype=np.int64)
        self.limit_range = np.array([m.jnt_range[j] for j in joints], dtype=np.float64).reshape(-1, 2)
        own = sorted(getattr(env, "_own_body_ids", set()) or [])
        self.own = np.zeros(m.nbody, dtype=bool)
        self.own[own] = True
        self.weight_n = float(np.sum(m.body_mass[own])) * GRAVITY_MS2 if own else 1.0
        self.group = np.array([_group_of_body(m.body(b).name) for b in range(m.nbody)], dtype=np.int64)

    @staticmethod
    def _sensor_adr(m, name):
        sid = int(m.sensor(name).id)
        adr = int(m.sensor_adr[sid])
        return np.arange(adr, adr + int(m.sensor_dim[sid]))


def _layout(env) -> _Layout:
    lay = getattr(env, "_vestibular_layout", None)
    if lay is None or getattr(lay, "_model_id", None) != id(env.model):
        lay = _Layout(env)
        lay._model_id = id(env.model)
        env._vestibular_layout = lay
    return lay


def _support(env, lay: _Layout, contacts) -> np.ndarray:
    """[force on head, trunk, arms, legs] in newtons from OUTSIDE contacts only.

    `contacts` is the ContactInfo list _collect already built (one record per sensing side;
    an outside object has no skin, so an outside contact appears once). With None, the
    forces are read straight from MuJoCo's contact list, which costs one mj_contactForce per
    contact and is for callers outside _collect."""
    out = np.zeros(4, dtype=np.float64)
    if contacts is not None:
        for c in contacts:
            if getattr(c, "other_is_self", False):
                continue
            g = int(lay.group[int(c.body_id)])
            if g >= 0:
                out[g] += max(0.0, float(c.normal_force))
        return out
    import mujoco
    m, d = env.model, env.data
    f6 = np.zeros(6)
    for i in range(int(d.ncon)):
        con = d.contact[i]
        b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
        if lay.own[b1] == lay.own[b2]:
            continue                       # self-contact, or two outside things
        mine = b1 if lay.own[b1] else b2
        g = int(lay.group[mine])
        if g < 0:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        out[g] += abs(float(f6[0]))
    return out


def read_vestibular_raw(env, contacts: Optional[Sequence] = None) -> Dict[str, np.ndarray]:
    """The same quantities in physical units, before any scale. Used by read_vestibular and
    by the scale measurement, so the measured distribution is of exactly what is scaled."""
    m, d = env.model, env.data
    lay = _layout(env)
    sd = np.asarray(d.sensordata, dtype=np.float64)
    q = np.asarray(d.qpos, dtype=np.float64)
    qd = np.asarray(d.qvel, dtype=np.float64)
    force = np.asarray(d.actuator_force, dtype=np.float64)
    native = getattr(env, "_native_actuator_force_range", None)
    frange = np.asarray(native if native is not None else m.actuator_forcerange, dtype=np.float64)
    neck_f = force[lay.neck_act]
    lo, hi = frange[lay.neck_act, 0], frange[lay.neck_act, 1]
    # signed share of the motor's FULL-strength range (the native range, before fatigue
    # scales it): +1 = pushing as hard as a fresh muscle can one way, -1 the other way
    neck_effort = np.where(neck_f >= 0.0, neck_f / np.where(hi > 0, hi, 1.0),
                           neck_f / np.where(lo < 0, -lo, 1.0))
    R_chest = np.asarray(d.xmat[lay.chest]).reshape(3, 3)
    support = _support(env, lay, contacts)
    lq = q[lay.limit_qpos]
    near = np.minimum(lq - lay.limit_range[:, 0], lay.limit_range[:, 1] - lq) < LIMIT_BAND_RAD
    cap = np.asarray(env.proprio.fatigue.cap, dtype=np.float64)
    return dict(
        acc=sd[lay.acc_adr].copy(),                              # m/s2, head frame
        gyro=sd[lay.gyro_adr].copy(),                            # rad/s, head frame
        neck_q=q[lay.neck_qpos].copy(),                          # rad
        neck_range=lay.neck_range,                               # rad, (3, 2)
        neck_qd=qd[lay.neck_dof].copy(),                         # rad/s
        neck_effort=neck_effort,                                 # -1..1 of full strength
        trunk_up=R_chest[2, :].copy(),                           # unit, chest frame
        support_n=support,                                       # N per group
        weight_n=np.float64(lay.weight_n),                       # N
        # mechanical power: actuator force x actuator velocity (= gear x joint speed), W
        power_w=np.float64(np.sum(np.abs(force * np.asarray(d.actuator_velocity, dtype=np.float64)))),
        # isometric load: summed joint torque the motors apply, N m
        holding_nm=np.float64(np.sum(np.abs(force * lay.gear))),
        limit_share=np.float64(near.mean()) if near.size else np.float64(0.0),
        cap=cap,
    )


def vestibular_from_raw(raw: Dict[str, np.ndarray]) -> np.ndarray:
    """Scale a raw read into the VESTIBULAR_FIELDS vector (float32)."""
    v = np.zeros(VESTIBULAR_DIM, dtype=np.float32)
    v[0:3] = raw["acc"] / GRAVITY_MS2
    v[3:6] = raw["gyro"] / GYRO_SCALE_RADS
    lo, hi = raw["neck_range"][:, 0], raw["neck_range"][:, 1]
    # each neck joint on its own range, -1 at one end and +1 at the other. MuJoCo's joint
    # limits are soft, so a joint can pass its end by a few thousandths of a radian; that
    # overshoot is clipped here and is what limit_load reports.
    v[6:9] = np.clip(2.0 * (raw["neck_q"] - lo) / np.maximum(hi - lo, 1e-9) - 1.0, -1.0, 1.0)
    v[9:12] = raw["neck_qd"] / NECK_VEL_SCALE_RADS
    v[12:15] = raw["neck_effort"]
    v[15:18] = raw["trunk_up"]
    total = float(np.sum(raw["support_n"]))
    if total > 0.0:
        v[18:22] = raw["support_n"] / total
    v[22] = total / max(float(raw["weight_n"]), 1e-9)
    v[23] = float(raw["power_w"]) / EFFORT_POWER_SCALE_W
    v[24] = float(raw["holding_nm"]) / EFFORT_HOLDING_SCALE_NM
    v[25] = float(raw["limit_share"])
    cap = raw["cap"]
    v[26] = float(np.min(cap)) if cap.size else 1.0
    v[27] = float(np.mean(cap)) if cap.size else 1.0
    v[28] = 1.0
    return v


def read_vestibular(env, contacts: Optional[Sequence] = None) -> np.ndarray:
    """The vector for this frame, from live MuJoCo (float32, VESTIBULAR_DIM).

    `env` is a BridgeEnv. `contacts` is the ContactInfo list the collect frame already
    computed; pass it to avoid reading contact forces twice."""
    return vestibular_from_raw(read_vestibular_raw(env, contacts))


def pad_legacy_vestibular(v) -> np.ndarray:
    """Widen an old (6,) vector (raw accelerometer m/s2, raw gyro rad/s) to VESTIBULAR_DIM.

    The six MIMo numbers go into their own fields on the same scales a live read uses; every
    other field is 0 and `valid` is 0, so a reader can tell a padded frame from a real one.
    A vector already VESTIBULAR_DIM wide is returned unchanged (as float32). This is the
    first frame after a resume from a checkpoint saved before 2026-09-22: the saved payload
    carries the old six numbers."""
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    if a.shape[0] == VESTIBULAR_DIM:
        return a
    if a.shape[0] != LEGACY_VESTIBULAR_DIM:
        raise ValueError(f"vestibular vector of width {a.shape[0]}: expected "
                         f"{LEGACY_VESTIBULAR_DIM} (legacy) or {VESTIBULAR_DIM}")
    out = np.zeros(VESTIBULAR_DIM, dtype=np.float32)
    out[0:3] = a[0:3] / GRAVITY_MS2
    out[3:6] = a[3:6] / GYRO_SCALE_RADS
    out[FIELD_INDEX["valid"]] = 0.0
    return out
