# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The vestibular / internal-state vector (DESIGN.md section 7, built 2026-09-22).

MIMo's head accelerometer and gyro were read every frame and thrown away. The vector that
now carries them, with the neck, the trunk's up-direction, what holds the body up, effort,
joint-limit load and fatigue, is the contract the model side builds on: its field order, its
width, the padding of an old six-number vector, and the physical sense of a few fields at
rest are checked here.
"""
import numpy as np
import pytest

from mimo_bridge import vestibular_adapter as VA


# ------------------------------------------------------------------ the contract itself
def test_fields_width_and_order_are_the_contract():
    assert VA.VESTIBULAR_DIM == len(VA.VESTIBULAR_FIELDS) == 29
    assert len(set(VA.VESTIBULAR_FIELDS)) == VA.VESTIBULAR_DIM, "a field name appears twice"
    f = VA.VESTIBULAR_FIELDS
    assert f[0:6] == ("head_acc_x", "head_acc_y", "head_acc_z",
                      "head_gyro_x", "head_gyro_y", "head_gyro_z"), \
        "the six MIMo numbers must lead, in MIMo's order, so padding places them by position"
    assert f[6:9] == ("neck_swivel", "neck_tilt", "neck_tilt_side")
    assert f[15:18] == ("trunk_up_x", "trunk_up_y", "trunk_up_z")
    assert f[18:23] == ("support_head", "support_trunk", "support_arms", "support_legs",
                        "support_total")
    assert f[-1] == "valid"
    assert all(VA.FIELD_INDEX[n] == i for i, n in enumerate(f))


def test_every_scale_carries_a_measurement():
    """A scale with no measurement is a number nobody can check (measure before demand)."""
    for name in ("GYRO_SCALE_RADS", "NECK_VEL_SCALE_RADS", "EFFORT_POWER_SCALE_W",
                 "EFFORT_HOLDING_SCALE_NM"):
        assert getattr(VA, name) > 0.0
    m = VA.MEASUREMENT
    assert m.get("date") and m.get("source"), "the scales were set without a recorded measurement"
    for key in ("gyro_rads", "neck_vel_rads", "power_w", "holding_nm"):
        assert "p99" in m["pooled"][key], key


# ------------------------------------------------------------------ the old six numbers
def test_a_legacy_six_number_vector_is_padded_into_its_own_fields_and_marked_invalid():
    old = np.array([9.81, 0.0, -4.905, 0.5, -1.0, 0.25], dtype=np.float32)
    v = VA.pad_legacy_vestibular(old)
    assert v.shape == (VA.VESTIBULAR_DIM,) and v.dtype == np.float32
    assert np.allclose(v[0:3], [1.0, 0.0, -0.5], atol=1e-6), "accelerometer not in g"
    assert np.allclose(v[3:6], old[3:6] / VA.GYRO_SCALE_RADS, atol=1e-6)
    assert v[VA.FIELD_INDEX["valid"]] == 0.0, "a padded frame must say it is not a real read"
    assert not np.any(v[6:VA.FIELD_INDEX["valid"]]), "fields the old vector never had must be 0"


def test_a_full_width_vector_passes_through_and_any_other_width_is_refused():
    full = np.arange(VA.VESTIBULAR_DIM, dtype=np.float32)
    assert np.array_equal(VA.pad_legacy_vestibular(full), full)
    with pytest.raises(ValueError):
        VA.pad_legacy_vestibular(np.zeros(7))


# ------------------------------------------------------------------ scaling a raw read
def _raw(**over):
    raw = dict(acc=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3),
               neck_q=np.zeros(3), neck_range=np.array([[-1.0, 1.0], [-1.0, 2.0], [-0.5, 0.5]]),
               neck_qd=np.zeros(3), neck_effort=np.zeros(3), trunk_up=np.array([0.0, 0.0, 1.0]),
               support_n=np.zeros(4), weight_n=np.float64(100.0), power_w=np.float64(0.0),
               holding_nm=np.float64(0.0), limit_share=np.float64(0.0), cap=np.ones(10))
    raw.update(over)
    return raw


def test_support_shares_sum_to_one_with_contact_and_are_zero_without():
    v = VA.vestibular_from_raw(_raw(support_n=np.array([10.0, 50.0, 15.0, 25.0])))
    i = VA.FIELD_INDEX
    shares = v[i["support_head"]:i["support_legs"] + 1]
    assert np.isclose(shares.sum(), 1.0) and np.allclose(shares, [0.1, 0.5, 0.15, 0.25])
    assert np.isclose(v[i["support_total"]], 1.0), "100 N on a 100 N body is one body weight"
    v0 = VA.vestibular_from_raw(_raw())
    assert not np.any(v0[i["support_head"]:i["support_total"] + 1]), "no contact must read 0"


def test_neck_angles_map_each_joints_own_range_onto_minus_one_to_one():
    i = VA.FIELD_INDEX["neck_swivel"]
    lo = VA.vestibular_from_raw(_raw(neck_q=np.array([-1.0, -1.0, -0.5])))[i:i + 3]
    hi = VA.vestibular_from_raw(_raw(neck_q=np.array([1.0, 2.0, 0.5])))[i:i + 3]
    mid = VA.vestibular_from_raw(_raw(neck_q=np.array([0.0, 0.5, 0.0])))[i:i + 3]
    past = VA.vestibular_from_raw(_raw(neck_q=np.array([1.01, -1.02, 0.6])))[i:i + 3]
    assert np.allclose(lo, -1.0) and np.allclose(hi, 1.0) and np.allclose(mid, 0.0)
    assert np.all(np.abs(past) <= 1.0), "a soft-limit overshoot must not leave [-1, 1]"


# ------------------------------------------------------------------ the live body
@pytest.fixture(scope="module")
def crib(tmp_path_factory):
    from mimo_bridge.env import BridgeEnv
    env = BridgeEnv(scene="crib", run_dir=str(tmp_path_factory.mktemp("vest")), vision=False,
                    tag="_vesttest", age=24, contact_profile="mechanical_v4",
                    body_function="modulate", caregiver_seconds=0, kp=6.0,
                    body_layout="trunk_v1", cmd_horizon=25)
    raw = env.reset()
    yield env, raw
    env.close()


def test_the_vector_is_computed_on_every_collect_and_is_finite(crib):
    env, raw = crib
    assert raw.vestibular.shape == (VA.VESTIBULAR_DIM,) and raw.vestibular.dtype == np.float32
    assert np.all(np.isfinite(raw.vestibular))
    assert raw.vestibular[VA.FIELD_INDEX["valid"]] == 1.0
    step = env.step(ctrl=np.zeros(env.model.nu))
    assert step.vestibular.shape == (VA.VESTIBULAR_DIM,) and np.all(np.isfinite(step.vestibular))


def test_the_head_sensors_are_mimos_own_numbers_on_the_contract_scales(crib):
    env, _ = crib
    raw = env.step(ctrl=np.zeros(env.model.nu))
    mimo = np.asarray(env.env.get_vestibular_obs(), dtype=np.float64)   # acc (3), gyro (3)
    assert np.allclose(raw.vestibular[0:3] * VA.GRAVITY_MS2, mimo[0:3], atol=1e-4)
    assert np.allclose(raw.vestibular[3:6] * VA.GYRO_SCALE_RADS, mimo[3:6], atol=1e-5)


def test_at_supine_rest_the_accelerometer_and_trunk_up_point_the_way_gravity_does(crib):
    """Lying still, the head accelerometer reads one g straight up (MuJoCo's accelerometer
    includes gravity), and world-up seen from the chest is world-up."""
    env, _ = crib
    for _ in range(50):
        raw = env.step(ctrl=np.zeros(env.model.nu))
    v = raw.vestibular.astype(np.float64)
    m, d = env.model, env.data
    R_head = np.asarray(d.xmat[m.body("head").id]).reshape(3, 3)
    R_chest = np.asarray(d.xmat[m.body("chest").id]).reshape(3, 3)
    acc_world = R_head @ v[0:3]
    assert 0.9 < np.linalg.norm(v[0:3]) < 1.1, "at rest the accelerometer should read one g"
    assert acc_world[2] / np.linalg.norm(acc_world) > 0.95, acc_world
    up_world = R_chest @ v[15:18]
    assert np.allclose(up_world, [0.0, 0.0, 1.0], atol=1e-5)
    assert np.all(np.abs(v[6:9]) <= 1.0), "neck angles must lie in [-1, 1]"
    i = VA.FIELD_INDEX
    shares = v[i["support_head"]:i["support_legs"] + 1]
    assert np.isclose(shares.sum(), 1.0, atol=1e-5), "lying on the floor, something carries it"
    assert 0.8 < v[i["support_total"]] < 1.5, "lying still, the floor carries about one body weight"
    assert 0.0 <= v[i["limit_load"]] <= 1.0
    assert 0.0 <= v[i["capacity_min"]] <= v[i["capacity_mean"]] <= 1.0


def test_a_body_in_the_air_has_no_support(crib):
    import mujoco
    env, _ = crib
    snap = env.save_full_state()
    try:
        m, d = env.model, env.data
        qa = int(m.jnt_qposadr[m.joint("mimo_orientation").id])
        d.qpos[qa + 2] += 0.5
        mujoco.mj_forward(m, d)
        i = VA.FIELD_INDEX
        for contacts in (None, []):
            v = VA.read_vestibular(env, contacts)
            assert not np.any(v[i["support_head"]:i["support_total"] + 1]), contacts
        # the contacts _collect builds and MuJoCo's own list agree when there is contact
    finally:
        env.restore_full_state(snap)
    contacts = env._contacts()
    a = VA.read_vestibular_raw(env, contacts)["support_n"]
    b = VA.read_vestibular_raw(env, None)["support_n"]
    assert np.allclose(a, b, atol=1e-6) and a.sum() > 0


# ------------------------------------------------------------------ effort, limits, fatigue
# review env-senses MINOR 2: these fields had no test of their value, only of their range.
def _moving(env, frames=12, seed=5):
    """A few frames of random control, then mj_forward so forces, joint torques and joint
    speeds all refer to the same instant (after mj_step, velocities are the step's end and
    actuator forces its last substep's start)."""
    import mujoco
    rng_ = np.random.default_rng(seed)
    ctrl = rng_.uniform(-0.8, 0.8, env.model.nu)
    for _ in range(frames):
        env.step(ctrl=ctrl)
    mujoco.mj_forward(env.model, env.data)


def _actuated_dofs(env):
    m = env.model
    dofs = [int(m.jnt_dofadr[int(m.actuator_trnid[a, 0])]) for a in range(m.nu)]
    assert len(set(dofs)) == len(dofs), "a joint driven by two motors: the per-joint check below would not hold"
    return np.asarray(dofs)


def test_effort_holding_is_the_summed_joint_torque_the_motors_apply(crib):
    env, _ = crib
    snap = env.save_full_state()
    try:
        _moving(env)
        d = env.data
        dofs = _actuated_dofs(env)
        joint_torque = np.abs(np.asarray(d.qfrc_actuator, dtype=np.float64)[dofs])
        raw = VA.read_vestibular_raw(env, env._contacts())
        assert joint_torque.sum() > 1.0, "the motors must be working for this check to mean anything"
        assert raw["holding_nm"] == pytest.approx(joint_torque.sum(), rel=1e-9)
        v = VA.vestibular_from_raw(raw)
        assert v[VA.FIELD_INDEX["effort_holding"]] == pytest.approx(
            joint_torque.sum() / VA.EFFORT_HOLDING_SCALE_NM, rel=1e-6)
    finally:
        env.restore_full_state(snap)


def test_effort_power_is_joint_torque_times_joint_speed_so_the_gear_is_in_it(crib):
    """Mechanical power: the torque each joint receives (actuator force x gear) times its
    speed. On the 24-month body the gears run from 0.0026 to 12.62, so a formula that left the
    gear out (as the fatigue model's default still does) would give a different number here."""
    env, _ = crib
    snap = env.save_full_state()
    try:
        _moving(env)
        m, d = env.model, env.data
        dofs = _actuated_dofs(env)
        qfrc = np.asarray(d.qfrc_actuator, dtype=np.float64)[dofs]
        qd = np.asarray(d.qvel, dtype=np.float64)[dofs]
        joint_power = float(np.sum(np.abs(qfrc * qd)))
        raw = VA.read_vestibular_raw(env, env._contacts())
        assert joint_power > 0.1, "the body must be moving for this check to mean anything"
        assert raw["power_w"] == pytest.approx(joint_power, rel=1e-9)
        no_gear = float(np.sum(np.abs(np.asarray(d.actuator_force) * qd)))
        assert abs(no_gear - joint_power) > 1e-3 * joint_power, "the check cannot tell the gear apart"
        v = VA.vestibular_from_raw(raw)
        assert v[VA.FIELD_INDEX["effort_power"]] == pytest.approx(joint_power / VA.EFFORT_POWER_SCALE_W, rel=1e-6)
    finally:
        env.restore_full_state(snap)


def test_a_neck_joint_inside_mimos_limit_band_raises_limit_load_by_one_joints_share(crib):
    """MIMo's band is 0.035 rad (2.0 degrees) from either end of a joint's range.

    The joint is placed at FIXED angles (1 degree and 3 degrees in from each end), not at
    multiples of the module's own constant, and the constant is pinned to MIMo's own setting:
    placed at multiples of LIMIT_BAND_RAD, the test passed with the band changed to 0.070 rad
    (4 degrees) in the module (verify-fix-env mutant, 2026-09-23), so it could not catch the
    band itself."""
    import mujoco
    from mimo_infant.simulation.envs.mimo_env import DEFAULT_PROPRIOCEPTION_PARAMS
    assert VA.LIMIT_BAND_RAD == DEFAULT_PROPRIOCEPTION_PARAMS["threshold"], \
        "limit_load must use MIMo's own limit band, not a band of our choosing"
    env, _ = crib
    snap = env.save_full_state()
    try:
        m, d = env.model, env.data
        lay = VA._layout(env)
        j = m.joint("robot:head_swivel").id
        adr = int(m.jnt_qposadr[j])
        assert adr in set(lay.limit_qpos.tolist()), "the neck swivel is a motor-driven limited joint"
        lo, hi = m.jnt_range[j]
        assert hi - lo > np.radians(10.0), "the range must be wide enough for both ends to be tested"
        d.qpos[adr] = 0.5 * (lo + hi)                  # mid-range: not at a limit
        mujoco.mj_forward(m, d)
        base = float(VA.read_vestibular_raw(env, [])["limit_share"])
        one_joint = 1.0 / len(lay.limit_qpos)
        for at in (lo + np.radians(1.0), hi - np.radians(1.0)):      # 1 degree from an end
            d.qpos[adr] = at
            mujoco.mj_forward(m, d)
            near = float(VA.read_vestibular_raw(env, [])["limit_share"])
            assert near - base == pytest.approx(one_joint, abs=1e-12)
        for at in (lo + np.radians(3.0), hi - np.radians(3.0)):      # 3 degrees: outside the band
            d.qpos[adr] = at
            mujoco.mj_forward(m, d)
            assert float(VA.read_vestibular_raw(env, [])["limit_share"]) == pytest.approx(base, abs=1e-12)
    finally:
        env.restore_full_state(snap)


def test_a_tired_neck_reads_its_effort_on_the_fresh_muscles_range(crib):
    """With body_function modulate a tired body set's motors lose force (gain and force range
    x capacity). The neck effort is divided by the FRESH range, so a neck at half capacity
    pushing as hard as it can reads 0.5 of full strength, not 1.0 (dividing by the tired range
    would say a tired neck pushes as hard as a fresh one)."""
    env, _ = crib
    assert env.body_function_mode == "modulate"
    snap = env.save_full_state()
    try:
        m = env.model
        lay = VA._layout(env)
        for sign in (1.0, -1.0):
            ctrl = np.zeros(m.nu)
            ctrl[lay.neck_act] = sign                       # full push on the three neck motors
            env.proprio.fatigue.cap[:] = 0.5                # every body set at half capacity
            raw = env.step(ctrl=ctrl)
            effort = raw.vestibular[VA.FIELD_INDEX["neck_swivel_effort"]:
                                    VA.FIELD_INDEX["neck_tilt_side_effort"] + 1].astype(np.float64)
            assert np.all(np.abs(effort) <= 0.5 + 1e-6), effort
            assert np.allclose(effort, 0.5 * sign, atol=1e-6), effort
    finally:
        env.restore_full_state(snap)


def test_the_first_frame_after_a_resume_from_an_old_payload_is_padded(crib):
    """The first frame after a resume is the SAVED payload; every checkpoint up to
    ckpt_step_344400 saved MIMo's six raw numbers."""
    from mimo_bridge.datasource import _raw_to_payload, _payload_to_frame, _identity_handles
    env, _ = crib
    raw = env.step(ctrl=np.zeros(env.model.nu))
    payload = _raw_to_payload(env, raw, _identity_handles(len(env.proprio.body_ids)), 0.0)
    from mimo_bridge import frame_encoder as FE
    assert len(payload["frame_bytes"]) == FE.SLOT_SIZE, \
        "the UE wire frame must keep its size; the vector rides the side channel only"
    live = _payload_to_frame(payload, 0)
    assert live.vestibular.shape == (VA.VESTIBULAR_DIM,) and live.vestibular[-1] == 1.0
    payload["vestibular"] = np.asarray(env.env.get_vestibular_obs(), dtype=np.float32)
    old = _payload_to_frame(payload, 0)
    assert old.vestibular.shape == (VA.VESTIBULAR_DIM,)
    assert old.vestibular[VA.FIELD_INDEX["valid"]] == 0.0
    assert np.allclose(old.vestibular[0:6], live.vestibular[0:6], atol=1e-4), \
        "the padded head sensors must read on the same scale as a live read"
