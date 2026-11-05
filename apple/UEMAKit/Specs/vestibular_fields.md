# Vestibular fields: 29 numbers per env step, 28 centred and scaled into the encoder

Work item WI-16, target A. Written 2026-11-04 05:21 PM PST on branch feature/apple-port-2026-11-04 from a complete read of `Source/Python/mimo_bridge/vestibular_adapter.py` (321 lines) and `vestibular_sense.py` (1,888 lines) and the cited lines of `lem_model.py`, `env.py`, `proprio_adapter.py`. Paths are relative to `D:/UEMA_Research/Source/Python`. Contract for the Swift type `VestibularSense` and its encoder input (REQ-B-4). `vestibular_sense on` (args.json:174) puts this sense in the sixth slot in place of intermind (vestibular_sense.py:120-121, 337-342).

## 1. The 29 fields (`VESTIBULAR_FIELDS`, vestibular_adapter.py:40-53), computed every env step by `read_vestibular(env, contacts)` (env.py:3139; vestibular_adapter.py:227-300)

Raw reads (`read_vestibular_raw`, :227-265), all float64: `sd = data.sensordata`, `q = qpos`, `qd = qvel`, `force = data.actuator_force`, `frange` = the NATIVE `actuator_forcerange` saved before fatigue scaling (`env._native_actuator_force_range`, env.py:2839; else the model's), `R_chest = xmat[chest]`.

| Index | Field | Formula (`vestibular_from_raw`, :268-292) | Constants |
| --- | --- | --- | --- |
| 0-2 | head_acc_x,y,z | `sensordata[acc_adr] / 9.81` (accelerometer sensor `vestibular_acc`, head frame) | `GRAVITY_MS2` 9.81 (:64) |
| 3-5 | head_gyro_x,y,z | `sensordata[gyro_adr] / 2.0` (gyro `vestibular_gyro`) | `GYRO_SCALE_RADS` 2.0 (:86) |
| 6-8 | neck_swivel, neck_tilt, neck_tilt_side | `clip(2 (q_j - lo) / max(hi - lo, 1e-9) - 1, -1, 1)` for joints `robot:head_swivel`, `robot:head_tilt`, `robot:head_tilt_side` | `NECK_JOINTS` (:58) |
| 9-11 | the three neck joint speeds | `qvel[dof_j] / 1.5` | `NECK_VEL_SCALE_RADS` 1.5 (:87) |
| 12-14 | the three neck efforts | `f / hi` when f >= 0 (hi > 0 else 1), `f / (-lo)` when f < 0 (lo < 0 else 1), f = `actuator_force` of `act:head_swivel`, `act:head_tilt`, `act:head_tilt_side` | `NECK_ACTUATORS` (:59); :238-243 |
| 15-17 | trunk_up_x,y,z | `R_chest[2, :]` (world up seen from the chest) | :244, :256 |
| 18-21 | support_head, support_trunk, support_arms, support_legs | `support_n / total` where `support_n[g]` sums `max(0, normal_force)` of every OUTSIDE contact (`other_is_self` false) on a body of group g (`_support`, :194-209; group from the skin region names, :127-143: head and the eyes 0, trunk 1, hands and arms 2, legs and feet 3); 0 when total = 0 | :281-283 |
| 22 | support_total | `total / weight_n`, `weight_n = sum(body_mass[own bodies]) * 9.81` N (:175) | :284 |
| 23 | effort_power | `sum |actuator_force * actuator_velocity| / 66.0` | `EFFORT_POWER_SCALE_W` 66.0 (:88); :260 |
| 24 | effort_holding | `sum |actuator_force * gear[:, 0]| / 63.0` | `EFFORT_HOLDING_SCALE_NM` 63.0 (:89); :262 |
| 25 | limit_load | share of motor-driven, limited, non-eye hinge joints with `min(q - lo, hi - q) < 0.035` rad | `LIMIT_BAND_RAD` 0.035 (:69); joints :162-171; :246-247 |
| 26 | capacity_min | `min(env.proprio.fatigue.cap)` (10 body-set capacities, proprio_rows.md section 3) | :248, :289 |
| 27 | capacity_mean | `mean(cap)` | :290 |
| 28 | valid | 1.0 on a live read | :291 |

Output: float32 [29] on `RawStep.vestibular` (env.py:3681). The model reads ONE vector per model step, the last env step's (`rate_hz` 20, vestibular_sense.py:289-290).

## 2. The encoder input (28 numbers): `VestibularNormaliser` (vestibular_sense.py:410-513)

- `ENCODER_FIELDS` = the 29 fields without `valid`; `ENCODER_DIM` = 28 (:129-131).
- `read_frames` (:376-407): a frame's vector of width 29 with `valid > 0.5` and all finite is a real read; a width-6 vector is padded (`pad_legacy_vestibular`, vestibular_adapter.py:303-321) and marked stale; a missing or non-finite vector is stale.
- `batch(frames)` (:450-465): `x = clip(raw[:, ENCODER_FIELDS], -4, 4)` (`INPUT_BOUND` 4.0, :172); `z = transform(x)` (:438-448): if `weight <= 0`, zeros; else `z = (x - mean) / sqrt(max(var, 1e-6))` (`SPREAD_FLOOR` 1e-3, :150, squared at :436), NaN -> 0, then `clip(z, -4, 4)` (`NORMALISED_BOUND` 4.0, :155). `z` float32 [B, 28] is the encoder input; `valid` [B] marks the batch stale when any body's read is not real (an input still goes through the encoder; it trains nothing).
- `commit()` (:467-478) folds the held real rows into the statistics only when the model trains: `lem_model.py:2588-2591` calls it under `if train`. Update: `a = 1 / 2000` (`NORM_HORIZON_STEPS`, vestibular_sense.py:144); `w_new = (1 - a) w + a`; `keep = (1 - a) w / w_new`; `mean = mean keep + rows.mean(0) a / w_new`; `var = var keep + mean((rows - mean_new)^2, 0) a / w_new`; `weight = w_new`; `n_real += len(rows)` (:467-478).

## 3. Where the running mean and variance are checkpointed (answers open question 13)

- `VestibularNormaliser.state_dict` (:481-486): `{version 'vestibular_sense_v2', fields [28 names], horizon_steps 2000, mean float64 [28], var float64 [28], weight float, n_real int}`.
- Saved in `runtime_state.pt` under the key `vestibular_norm` (lem_model.py:9991); restored at load from `rt.get("vestibular_norm")` (lem_model.py:9721-9722; `load_state_dict`, vestibular_sense.py:488-500, refuses another field list or version and starts cold).
- In an inference-only session nothing calls `commit`, so `mean`, `var`, `weight` stay at the checkpoint's values: the Swift normaliser is three frozen arrays plus the two bounds. The fixture tool exports `mean[28]`, `var[28]` and `weight` from `weights/.../ckpt_step_494025/runtime_state.pt['vestibular_norm']`.

## 4. Stamps the loader checks (vestibular_sense.py:222-267)

`stamp()` records version `vestibular_sense_v2`, the field list, dim 29, the 28 encoder fields, `scales` {head_acc_g_per_ms2 1/9.81, head_gyro_rads 2.0, neck_vel_rads 1.5, effort_power_w 66.0, effort_holding_nm 63.0}, `input_bound` 4.0, `normaliser` {horizon 2000, spread_floor 0.001, bound 4.0}, `adapter_sha` and the adapter definition; the released receipt carries exactly these (`weights/.../model_receipt.json` key `vestibular_sense`: adapter_sha `6a606a947cf9cdba`, lesson_weight 1.0, wellness_pe exclude). A Swift build that changes any scale reads a different quantity than the encoder was trained on.

## 5. Encoder shape (for WI-31; from the config, not the weights)

`modality_overrides` (:271-302): type `mlp_project`, input 28, `hidden_dims` and `d_internal` as proprioception's (`encoder_width` 192, `encoder_layers` 2, args.json:147-148), bias on every layer, output 192; the registry's output LayerNorm (no gain, no bias; vestibular_sense.py:1187-1190) follows. `vestibular_encoder_repair rescale_restart` (args.json:177) divided the last layer's weight and bias by a power of two at the switch (`repair_encoder`, vestibular_sense.py:1171-1265); the divided values are what `model.safetensors` holds, so the Swift loader applies nothing further; `encoder_precision` is "the run's" (float32 on the Swift side).

## 6. Open points

A. `env.proprio.fatigue.cap` is advanced by `ProprioAdapter.read(dt)` (proprio_rows.md section 3) BEFORE `read_vestibular` in `_collect` (env.py:3133-3139), so fields 26-27 are this env step's capacities.
B. `_support` with `contacts` given counts each outside contact once per sensing side (one record, since an outside object has no skin); with `contacts` None it reads `d.contact` directly (:210-224). The Swift reader takes the same contact list as the touch pooler.

## 7. Reader's check (acceptance: 3 constants per file)

| Constant | Claimed value | File:line | Checked by | Date |
| --- | --- | --- | --- | --- |
| `EFFORT_HOLDING_SCALE_NM` | 63.0 | vestibular_adapter.py:89 | WI-16 adversarial reviewer (not the writer) | 2026-11-04 05:56 PM PST |
| `NORM_HORIZON_STEPS` | 2000 | vestibular_sense.py:144 | WI-16 adversarial reviewer (not the writer) | 2026-11-04 05:56 PM PST |
| runtime_state key | `vestibular_norm` | lem_model.py:9991 | WI-16 adversarial reviewer (not the writer) | 2026-11-04 05:56 PM PST |

Signature of the reader: WI-16 adversarial reviewer (not the writer), 2026-11-04 05:56 PM PST. The three constants above were re-read at the cited lines; every other file:line cite in this file was re-read against the branch's working tree, and the cites into lem_model.py, train_mimo.py and hand_response.py were moved to the lines commit 23dae32 (5:31 PM PST) and the uncommitted edits after it gave them.
