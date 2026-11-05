# Proprioception rows [26, 12] and motor rows [26, 19]

Work item WI-16, target A. Written 2026-11-04 05:21 PM PST on branch feature/apple-port-2026-11-04 from a complete read of `Source/Python/mimo_bridge/proprio_adapter.py` (246 lines), `body_layout.py` (83 lines), `lem/model/body_sets.py` (139 lines), `tokenizer/modalities/proprioception.py` (85 lines), `tokenizer/modalities/motor.py` (114 lines) and the cited lines of `datasource.py`, `lem_model.py`, `encoder_registry.py`, `tokenizer/config.py`. Paths are relative to `D:/UEMA_Research/Source/Python`. Contract for the Swift types `ProprioceptionReader` and `MotorReader` (WI-27, REQ-B-4).

## 1. The 26 rows

`names_for('trunk_v1')` = `HANDLE_NAMES` (24: wrist_L, thumb_L, index_L, middle_L, ring_L, pinky_L, wrist_R, thumb_R, index_R, middle_R, ring_R, pinky_R, head, upperarm_L, lowerarm_L, upperarm_R, lowerarm_R, thigh_L, calf_L, thigh_R, calf_R, foot_L, foot_R, gaze; body_sets.py:52-75) + (lower_trunk, chest) (body_layout.py:6-14; `body_layout trunk_v1`, args.json:140). Bodies: `BONE_MAP` (proprio_adapter.py:27-41; gaze -> `left_eye`) and `TRUNK_BODIES` lower_trunk -> `lower_body`, chest -> `chest` (body_layout.py:7). Body sets per row: `HANDLE_BODY_SET` (body_sets.py:82-101) and Spine (4) for the two trunk rows (proprio_adapter.py:167-168). `config_overrides('trunk_v1')` sets `n_handles` 26, proprioception `input_dim` 26 x 12 = 312, motor `n_handles` 26, wellness motor_state `input_dim` 26 x 9 = 234 (body_layout.py:40-55).

## 2. Proprioception, per env step (`ProprioAdapter.read(dt)`, proprio_adapter.py:201-243), dt = 0.01 s

`bones` float32 [26, 11] per row i with body b:
- 0-3: `xquat[b]` reordered from w x y z to x y z w (:207-208), world frame;
- 4-6: `xpos[b] * 100` cm, world (:209);
- 7-9: `mj_objectVelocity(model, data, mjOBJ_BODY, b, vel6, 0)[3:6] * 100` cm/s, world frame, at the body origin (:210-211; flg_local 0);
- 10: `fatigue.cap[body set of row i]` (:212).
Gaze row (:223-230): its quaternion is replaced by `conj(head quaternion) * eye quaternion` (xyzw product written out at :227-230), so the row carries the eye relative to the head.
Energy (:238-241): `ph_work = sum(effort) * dt`, `pcc_hold = sum|actuator_force| * dt * 0.01`, total = their sum (float32 [3]); `body_sets` [10, 2] = (cap, accumulated effort) (:242).

## 3. The fatigue model (`FatigueModel`, proprio_adapter.py:71-120), advanced once per env step before the rows are read (:203-204)

- `effort_per_set` (:184-199), `effort_power no_gear` (the default; `EFFORT_POWER_MODES` :146): `p[a] = |actuator_force[a] * qvel[dof of actuator a's joint]|`, summed per body set (actuator -> set by `body_set_for_mimo_body` of the actuated joint's body, :48-68, :170-176); an actuator of no set is dropped.
- `step(effort, dt)` (:112-116): `cap += dt (0.05 (1 - cap) - 0.02 effort / 1.0)`; `cap = clip(cap, 0, 1)`; `accum += effort dt`. Constants: `recover` 0.05 /s, `drain` 0.02 /s, `p_ref` 1.0 (:105). Starts at `cap = 1`, `accum = 0`; `reset()` restores them (:118-120).
- These 10 capacities are also: the actuator gain and force-range factor under `body_function modulate` (command_decoder.md section 4), the vestibular fields 26-27, and the wellness capability term (wellness_sum.md). The fatigue state is saved with the world's snapshot (`restore_full_state` replaces `fatigue.__dict__`, :156-157); the fixture starts from the checkpoint's `cap` and `accum`.

## 4. From the rows to the proprioception encoder input [312] (datasource.py:119-141, the "extended handles" path taken because 26 > `N_HANDLES` 24, :107, :121)

`normalized` float32 [26, 12]: columns 0-10 = `bones`; then `[:, 4:7] = (bones[:, 4:7] - bones[0, 4:7]) / 200` (row 0 = wrist_L is the origin; `position_range_cm` 200.0, tokenizer/config.py:263); `[:, 7:10] /= 500` (`velocity_range_cms` 500.0, config.py:264); `[:, 11] = row index / 25` (`proprio_index_range` 25.0, tokenizer/config.py:271; datasource.py:131-141). The frame entry is `normalized.reshape(312)` (lem_model.py:2413). The 24-row tokenizer path (`tokenize_proprioception`, tokenizer/modalities/proprioception.py:23-85) applies the same rules and is not used by this body.

## 5. Motor rows: the achieved half [26, 10] and the command half [26, 9]

1. `motor_to_echo(motor9, handle_mask, bones)` (datasource.py:173-192), called with the EXECUTED command of this env step and this env step's `bones` (datasource.py:285, :420): `echo[:, 0:3] = bones[:, 4:7]` (achieved position, cm, world); `echo[:, 3:7] = bones[:, 0:4]` (achieved quaternion xyzw; the gaze row's is head-relative per section 2); `echo[:, 7] = motor9[:, 7]` (commanded radius); `echo[:, 8] = motor9[:, 8]` (commanded strength); `echo[:, 9] = handle_mask` as 1.0/0.0 (1.0 when no mask).
2. `_payload_to_frame` (datasource.py:142-152): `handles[:, :3] /= 200`; `tf.motor_handles = handles` float32 [26, 10]. Note (:143-150): the proprioception rows are made relative to wrist_L before the division; this achieved half stays in world cm / 200, so the two blocks differ by wrist_L's position / 200 on every row.
3. In the model (lem_model.py:2416-2418): `frame["motor"] = motor_handles` [B, 26, 10] = `handle_state` (:2608-2610); the encoder's input is widened to [26, 19] = [achieved 10, command 9] (`MOTOR_ACHIEVED_F` 10, `MOTOR_COMMAND_F` 9, `MOTOR_INPUT_F` 19, lem_model.py:69-71): the LIVE frame's command columns are zeros (:2657-2661; the outgoing command has not been chosen yet); the PREVIOUS model step's motor slot in the context buffer is rewritten with `[last_achieved[..., :10], command_for_encoder(exec_t)]` re-encoded through the registry plus the modality embedding (:2640-2656), where `command_for_encoder` divides the executed command's positions by 200 (`COMMAND_POSITION_RANGE_CM`, :81, :139-149; encoder_registry.py:245-268). Inside the planner and the imagination chain `build_motor_input` (encoder_registry.py:271-299) forms the same [achieved 10, command 9] row.
4. Capability [21] (tokenize_motor, tokenizer/modalities/motor.py:97-108): `[cap_0..cap_9, accum_0..accum_9 / 1000, total_energy / 1000]` (`energy_max` 1000.0, config.py:273) -> `extras["capability"]` (lem_model.py:2418) -> the wellness capability term reads `[:, :10]` (wellness_sum.md). Energy [3] = `[ph_work, pcc_hold, total] / 1000` (motor.py:85-95).

## 6. Open points

A. The mimo_bridge frame packs the 24-row wire contract and carries the 26-row arrays beside it (datasource.py:107, :142); the Swift reader builds the 26-row arrays directly and skips the byte layout.
B. The receipt key `proprio_index_scale` (model_receipt.json top-level key list) stamps the /25 rule; its value in the released receipt is 25.0.
C. `mj_objectVelocity` with `flg_local` 0 gives the world-frame velocity of the body origin (`rot(3), lin(3)` sequence, proprio_adapter.py:14); the MuJoCo C function documentation is the Swift builder's reference for the argument sequence.

## 7. Reader's check (acceptance: 3 constants per file)

| Constant | Claimed value | File:line | Checked by | Date |
| --- | --- | --- | --- | --- |
| `recover`, `drain`, `p_ref` | 0.05 /s, 0.02 /s, 1.0 | proprio_adapter.py:105 | WI-16 adversarial reviewer (not the writer) | 2026-11-04 05:56 PM PST |
| `proprio_index_range` | 25.0 | tokenizer/config.py:271 | WI-16 adversarial reviewer (not the writer) | 2026-11-04 05:56 PM PST |
| `MOTOR_INPUT_F` | 19 | lem_model.py:71 | WI-16 adversarial reviewer (not the writer) | 2026-11-04 05:56 PM PST |

Signature of the reader: WI-16 adversarial reviewer (not the writer), 2026-11-04 05:56 PM PST. The three constants above were re-read at the cited lines; every other file:line cite in this file was re-read against the branch's working tree, and the cites into lem_model.py, train_mimo.py and hand_response.py were moved to the lines commit 23dae32 (5:31 PM PST) and the uncommitted edits after it gave them.
