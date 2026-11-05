# Command decoder: the command (26 handles x 9 numbers) to the 92 motor controls

Work item WI-16, target A. Written 2026-11-04 05:21 PM PST on branch feature/apple-port-2026-11-04 from a complete read of `Source/Python/mimo_bridge/action_decoder.py` (527 lines) and the lines of `env.py`, `hand_target_control.py`, `lem_model.py`, `body_layout.py`, `body_sets.py`, `gaze_reflex.py` and the released run's `args.json` cited below. Every path is relative to `D:/UEMA_Research/Source/Python` unless it starts with `weights/`. This is the contract the Swift type `HandleCommandDecoder` (specification section 5, REQ-B-13) implements against.

## 1. Inputs, per env step (10 ms)

The decoder runs once per call of `BridgeEnv.step` (env.py:2847-2891), which the trainer calls 5 times per model step (`action_repeat` 5, weights/b1_continue_B1K_skip13_492800/run/args.json:154; train_mimo.py:2352, :3037). Each call reads the body's state of that env step; the command is the same for the 5 env steps of one model step and, under `motor_hold` 5 (args.json:31), for 5 model steps (train_mimo.py:3042-3044: one command held for `motor_hold` model steps).

| Input | Shape, type, unit | Source |
| --- | --- | --- |
| `motor` | [26, 9] float64. Columns 0-2 target position, cm, world frame; 3-6 target orientation as a quaternion x y z w, world frame; 7 radius (never read, action_decoder.py:43 "Radius is ignored"); 8 strength, clipped to [0, 1] at line 420. Rows follow `HANDLE_NAMES` (24, lem/model/body_sets.py:52-75) then `lower_trunk`, `chest` (mimo_bridge/body_layout.py:6-14, layout `trunk_v1`, args.json:140). | The executed command after the trainer's smoothing and exploration noise (outside this file; see planner_loss.md section 3 step 2). Under `hand_target_mode wrist_relative_v1` (args.json:187) the 10 finger rows (handles 1-5 and 7-11) are first REPLACED by the pose the latched finger-joint goal gives: `HandTargetControl.execute` (hand_target_control.py:465-503) writes the goal angles into a scratch `MjData`, runs `mujoco.mj_kinematics`, and sets `executed[index, :3] = scratch.xpos[body] * 100` and `executed[index, 3:7] = scratch.xquat[body][[1, 2, 3, 0]]` (lines 496-502). |
| `gaze` | 2 float64, radians: azimuth, elevation. | From the command's row 23 (`gaze`): `LemLite._gaze_from_handle` (mimo_bridge/lem_model.py:5122-5140): the row's quaternion normalised, yaw = atan2(2(w z + x y), 1 - 2(y y + z z)), pitch = asin(clamp(2(w y - z x), -1, 1)); azimuth = clamp(yaw, -0.785, 0.785) rad, elevation = clamp(pitch, -0.820, 0.576) rad (`AZ_RANGE`, `EL_RANGE`, mimo_bridge/gaze_reflex.py:41-42; `eye_row_convention legacy`, args.json:215, keeps elevation = pitch, lem_model.py:5137-5138). Before the decoder, env.py:2870-2874 lets the spontaneous-saccade generator (`spontaneous_saccades on`, args.json:113) and the gaze reflex (`gaze_reflex on`, args.json:19) rewrite this pair; those two programs are not transcribed here (open point A below). |
| `handle_mask` | [26] bool. | Every curriculum phase's mask is all True (built by lem/training/curriculum.py:236-246; lem_model.py:5720-5721 records that all six phases' masks are all True); `body_layout.extend_limits` pads it with True for the two trunk rows (body_layout.py:29-37). |
| `finger_goals` | {MuJoCo joint id: goal angle, rad} for the 44 hand hinges. | `finger_joint_servo on` (args.json:191): env.py:2886-2888 passes `HandTargetControl.goals_by_joint(cache['joint_goal'])` (hand_target_control.py:457-463), the 44 angles in the geometry sequence (the receipt's `hand_response.joint_names`, weights/.../ckpt_step_494025/model_receipt.json, 44 names). |
| MuJoCo state | `data.qpos`, `data.qvel`, `data.xmat` [nbody, 9], `data.xpos` [nbody, 3], `data.xaxis` [njnt, 3], `data.xanchor` [njnt, 3]; model `jnt_limited`, `jnt_range`, `jnt_qposadr`, `jnt_dofadr`, `actuator_trnid`, `actuator_trntype`, `actuator_ctrlrange`, `actuator_ctrllimited`, `body_parentid`. | Read directly in `decode` and `__init__` (action_decoder.py:271-279, 296-311, 365-389, 412-518). |

## 2. Constants built once (`ActionDecoder.__init__`, action_decoder.py:257-358)

| Constant | Value | Where |
| --- | --- | --- |
| `MOTOR_CONTRACT` | `'ancestor_arm_axes_signed_limits_v3'` | action_decoder.py:56 |
| `FINGER_SERVO_CONTRACT` | `'finger_joint_servo_v1'`; `FINGER_DIGITS` = thumb, index, middle, ring, pinky | :71-72 |
| `HANDLE_CHAINS` | handle -> body names whose hinge joints it may move, distal first; `gaze` -> empty | :75-99; trunk rows: `lower_trunk` -> `lower_body`, `chest` -> `chest` (:267; body_layout.py:7) |
| `POSITION_HANDLES` | the 12 hand handles (wrists and finger tips) plus `head`, `lower_trunk`: 14 handles whose position row enters the solve | :118-120; `position_rows 'all'` (args.json:190) keeps all 14 (action_decoder.py:132-136) |
| `POSITION_WEIGHT` | 8.0 rad per metre of wanted move | :139 |
| `EYE_JOINTS` | `robot:left_eye_horizontal` (gaze component 0, sign +1), `robot:left_eye_vertical` (1, +1), `robot:right_eye_horizontal` (0, -1), `robot:right_eye_vertical` (1, +1) | :141-146 |
| `kp` | 6.0 (per rad of joint change) | args.json:80 `decoder_kp` -> train_mimo.py:1406 -> env.py:1294 -> action_decoder.py:258 |
| `kd` | 0.3 (per rad/s) | env.py:991 default; train_mimo.py:1403-1430 (`env_kwargs_for`) passes no kd |
| `gaze_kp`, `gaze_kd` | 80.0, 3.0 | args.json:29-30; env.py:996, :1297 |
| `lam` | 1e-3 (damping of the least squares) | action_decoder.py:259, :190 |
| `ctrl_low`, `ctrl_high` [nu] | `actuator_ctrlrange` where `actuator_ctrllimited`, else -1, 1 (:276-279); then narrowed to the env action space: `max(low, space.low)`, `min(high, space.high)` (:286-290; env.py:1296 passes `self.env.action_space.low/high`) | |
| `joint_to_act` | joint id -> FIRST actuator whose `actuator_trntype == mjTRN_JOINT` and `actuator_trnid[a, 0] == joint` | :271-275 |
| per handle `_HandleInfo` | `body_id` of the handle's bone (`BONE_MAP` proprio_adapter.py:27-41 plus `TRUNK_BODIES`); `joint_ids` = every HINGE joint (`jnt_type == mjJNT_HINGE`) on a chain body that has an actuator, in chain sequence then joint-id sequence; `dof_adrs`, `act_ids`, `axis_body_ids`; `reach_m` | action_decoder.py:296-311 |
| `reach_m` | `_measure_reach` (:200-241): every joint of the chain set to each end of its range (unlimited joints: -1.5 and +1.5 rad), `mj_forward` on a scratch `MjData`, the handle body's `xpos` collected over all 2^n corners; reach = the largest distance between any two of those positions, then `min(0.5, max(reach, 0.01))` m; a chain with 0 or more than 10 joints keeps 0.5 m. The receipt records the results in cm (`motor_synergy.pos_reach_cm`): wrists 42.85, fingers 5.48 to 6.80, head 4.81, lower_trunk 5.58 (model_receipt.json). | |
| arm groups | per side: members (wrist, lowerarm, upperarm) in that sequence; `owners[k]` = the member whose body carries joint k of the wrist's chain; `columns[member]` = the wrist-chain joints whose body is an ancestor (inclusive) of that member's body | :312-345 |
| `finger_handles` | handle indices whose name starts with a `FINGER_DIGITS` word (10 of them) | :356-357 |
| `eye` | per `EYE_JOINTS` entry present and actuated: (actuator, dof, component, sign) | :347-354 |

## 3. Operation per call (`decode`, action_decoder.py:402-520)

1. `ctrl = zeros(nu)`; `qd = data.qvel` (:411-412).
2. For each handle i NOT in an arm group, with `handle_mask[i]` true, with at least one joint, and with strength = clip(motor[i, 8], 0, 1) > 0 (:413-422):
   - finger handle with `finger_goals` given: `dq[k] = goal[j_k] - qpos[jnt_qposadr[j_k]]` for each joint of the chain (`_finger_joint_step`, :391-399; a missing goal raises);
   - otherwise `_pose_step` (:361-389): `R_t` = rotation matrix of the row's quaternion (`quat_xyzw_to_mat`, :150-160, normalised; a quaternion of norm under 1e-12 gives identity); `R_c = xmat[body]`; `w = rotmat_log(R_t R_c^T)` (:167-187: angle = arccos(clamp((trace - 1)/2)), 0 below 1e-9; the pi case takes the strongest column of (R + R^T + 2I)/4 with the sign of the skew part); `A[:, k] = xaxis[j_k]`; if the handle is a position handle and the position is finite: `e = clip(motor[i, 0:3] / 100 - xpos[body], -reach_m, +reach_m)` m, `J[:, k] = A[:, k] x (xpos[body] - xanchor[j_k])`, and the solve is on the stacked `[A; 8 J] dq = [w; 8 e]`; otherwise on `A dq = w`. The solve is `dq = (M^T M + 1e-3 I)^-1 M^T r` (`damped_least_squares`, :190-196).
   - per joint k (:427-436): if `jnt_limited[j]`: `dq[k] = clip(q_j + dq[k], range_lo, range_hi) - q_j`; `u = strength * (6.0 * dq[k] - 0.3 * qvel[dof])`; `ctrl[act] += u`.
3. For each arm group (:437-482) whose wrist chain has joints and whose active members (`mask` true and `motor[m, 8] > 0`) are not empty: `axes[:, k] = xaxis[j_k]` over the wrist chain's joints (hand + lower arm + upper arm); for each active member m with `sm = clip(motor[m, 8], 0, 1)`: rows `sm * A_m` (only its `columns`), right side `sm * w_m`; and, for a position member with finite position, rows `sm * 8 * J_m` and `sm * 8 * e_m` (e clipped to that member's `reach_m`); one damped solve over the stacked rows; limit clip per joint as in step 2; `ctrl[act] += s_own * (6.0 dq[k] - 0.3 qvel[dof])` where `s_own` is the owner member's strength when the owner is active, else the largest active member's strength (:480-482).
4. Direction-preserving rescale per chain and per arm group (:483-506): for the group's actuator ids, `bound = ctrl_high` where `ctrl >= 0` else `-ctrl_low`; `ratio = bound / |ctrl|` on nonzero entries, 1 elsewhere; the whole group is multiplied by `max(0, min(1, min(ratio)))`.
5. `ctrl = clip(ctrl, ctrl_low, ctrl_high)` (:507).
6. Gaze (:508-518), when `gaze` is given: for each eye entry: joint j = `actuator_trnid[a, 0]`; `target = sign * gaze[component]`, clipped to `jnt_range[j]` when limited; `u = 80.0 * (target - qpos[jnt_qposadr[j]]) - 3.0 * qvel[dof]`; `ctrl[a] = clip(u, ctrl_low[a], ctrl_high[a])` (the eye actuators are SET, not added to).
7. `last_ctrl = ctrl`; return.

## 4. Output and what the world does with it

- `ctrl`: [nu] float64, nu = `model.nu` (92 motors on the released body, specification section "The setting"); stored on `env.ctrl` (env.py:2891).
- `body_function modulate` (args.json:51): after the decode, `_apply_motor_function` (env.py:2832-2845) sets `actuator_gainprm[:, 0] = native_gain * factor` and `actuator_forcerange = native_range * factor` where `factor[a] = fatigue.cap[body set of actuator a]` (proprio_rows.md section 3) and 1 for an actuator of no body set. It requires `actuator_gaintype == 0` and `actuator_biastype == 0` for every actuator (:2835-2836).
- The MIMo actuation model clips `ctrl` to the env action space and writes `data.ctrl[actuators]` (third_party/MIMo/mimo/mimo_infant/actuation/actuation.py:153-154, a path from the repository root `D:/UEMA_Research`), then the physics runs 2 x (`mj_step`, `mj_rnePostConstraint`) per env step (specification REQ-B-3, [I2 third_party/MIMo/mimo/mimo_infant/simulation/envs/mimo_env.py:573-619]; not re-read here).

## 5. Open points for the Swift implementation

A. The saccade generator and the gaze reflex (env.py:2870-2874; mimo_bridge/saccades and gaze_reflex programs) change `gaze` before the decoder; neither was transcribed in this item. The fixture of REQ-B-13 must record the gaze pair AS PASSED to `decode` (after both), or the Swift decoder reads a different azimuth.
B. The finger rows reach the decoder already replaced by `HandTargetControl.execute` (section 1); under `finger_joint_servo on` the decoder reads the goal angles, not those rows, for the 10 finger chains, but the arm groups still read the WRIST rows and every other row as given.
C. The receipt's `motor_contract` reported by the world is `MOTOR_CONTRACT + '+' + FINGER_SERVO_CONTRACT` (env.py:3576-3583); the Swift type should expose the same string.
D. `limb_target_control` is built only when `parent_relative_targets` or `reach_projection` is not `off` (env.py:1301-1305); both are `off` (args.json:188-189), so env.py:2860-2869 does nothing on the released configuration.

## 6. Reader's check (acceptance: 3 constants per file, checked by someone who did not write it)

| Constant | Claimed value | File:line | Checked by | Date |
| --- | --- | --- | --- | --- |
| `POSITION_WEIGHT` | 8.0 | action_decoder.py:139 | WI-16 adversarial reviewer (not the writer) | 2026-11-04 05:56 PM PST |
| `kp` in force | 6.0 | args.json:80, env.py:1294 | WI-16 adversarial reviewer (not the writer) | 2026-11-04 05:56 PM PST |
| `EL_RANGE` | (-0.820, 0.576) rad | gaze_reflex.py:42 | WI-16 adversarial reviewer (not the writer) | 2026-11-04 05:56 PM PST |

Signature of the reader: WI-16 adversarial reviewer (not the writer), 2026-11-04 05:56 PM PST. The three constants above were re-read at the cited lines; every other file:line cite in this file was re-read against the branch's working tree, and the cites into lem_model.py, train_mimo.py and hand_response.py were moved to the lines commit 23dae32 (5:31 PM PST) and the uncommitted edits after it gave them.
