# Wellness sum: the grounded terms, the goal-ring value and the ranking channels

Work item WI-16, target A. Written 2026-11-04 05:21 PM PST on branch feature/apple-port-2026-11-04 from a complete read of `Source/Python/lem/model/wellness.py` (630 lines), `lem/model/contact_wellness.py` (111 lines) and the cited lines of `lem_model.py`, `lem/model/wellness_channels.py`, `lem/master_config.yaml`, `mimo_bridge/vestibular_sense.py`. Paths are relative to `D:/UEMA_Research/Source/Python`. Contract for the Swift types `WellnessScorer` and `WellnessGoalRing` (WI-36, REQ-B-10, REQ-B-12).

## 1. The sum (lem_model.py:3099)

`total_wellness = sum over bodies of sum over the grounded terms` where the grounded terms are, in dict sequence, `prediction_error`, `tactile_comfort`, `capability`, `coherence`, `motor_state` (`WellnessEvaluators.EVALUATOR_NAMES`, wellness.py:140-143; `wellness_mode baseline`, args.json:27, so the BASELINE branch wellness.py:455-539 runs and every score is the formula, the MLPs' outputs being discarded, wellness.py:513-539) and `self_touch` (lem_model.py:3097). Each term is [B, 1] in [0, 1]. The same per-body sum is `wball` (:4148), the raw value stored in the goal ring (section 3).

## 2. The grounded terms

| Term | Formula | Inputs (lem_model.py) | Constants |
| --- | --- | --- | --- |
| prediction_error | `1 - clamp(mean over matched senses of mean over 192 of (pred - target)^2, 0, 1)` (wellness.py:199-226); matched = senses in both dicts, graded "real", and not in `pe_excluded` = ("vestibular",) (`vestibular_wellness_pe exclude`, args.json:176; vestibular_sense.py:330) | `_wp` = the delayed prediction of THIS model step's tokens made on the previous model step (`_nf_pred`), `_wt` = this model step's raw tokens (lem_model.py:3028-3038); on the first model step after a reset the same-frame pair `predictions` vs `targets` (lem_model.py:3008) | |
| tactile_comfort | `comfort = (0.5 + 0.35 pleasant) (1 - pain)` per body (contact_wellness.py:36-94): `pressure = features[..., 0]` (units of 100 kPa), `active = pressure > 1e-6 & valid`; `gentle = exp(-(log10 max(pressure, 1e-5) - log10 0.05)^2 / (2 0.6^2))`; `texture = tanh(mean|features[..., 4:8]| / 0.5)`; `local_pleasant = clamp(gentle (0.5 + 0.5 texture) sens_1, 0, 1) active`; `pleasant` = area-weighted mean over active sites (area = `features[..., 3]`, weight 1 where area <= 0); `pain = max over sites of mechanical_load(pressure x 1e5 Pa, history, sens_0, WIDE_RANGE_PROFILE)` (profile `mechanical_v4`, :64-76; `lem/skin_mechanics.py:177`, `:87`, not read in this item) | `raw_tactile = {features: all measured patches [B, n_patches, 10] (normalised fields, touch_pooler.md section 6), sensitivity [B, n, 2] (ones unless a skin profile is given), history [B, n, 3] = the frame's `mechanical_contact.history` (the exposure memory), profile}` (lem_model.py:2419-2454) | `TACTILE_PLEASANT_PEAK` 0.05, `TACTILE_PEAK_WIDTH` 0.6, `TACTILE_SUSTAINED_LOAD_START` 0.1 (legacy branch only), `TACTILE_TEXTURE_REF` 0.5 (contact_wellness.py:17-33) |
| capability | `clamp(mean of capability[:, :10], 0, 1)` (wellness.py:277-290) | `extras["capability"]` [B, 21] (proprio_rows.md section 5) | |
| coherence | `clamp(body_schema_coherence, 0, 1)` per body (wellness.py:252-265); `body_schema_updates on` (args.json:49) and `wellness_coherence extent` (args.json:98): "the share of everything the body map watches that it calls its own, its 26 handles and any outside thing taken in" (wellness.py:237-239), computed by `_schema_coherence(B)` from `DifferentiableBodySchema` (lem/model/body_schema.py, 401 lines, not read: open point A); the fallback without a body map is the mean cosine likeness of prediction to target (:266-275) | lem_model.py:3077-3078 | |
| motor_state | the boundary formula (motor_controller_input.md section 2; wellness.py:334-395): `1 - mean over 26 x 3 of penalty` | `motor_commands` = `handle_state[:, :, :9]` with positions centred on the handle centroid (lem_model.py:2999-3001) | `BODY_REACH` 0.21 (wellness.py:365), threshold 0.1 rom, `rom_limit` 1.0 |
| self_touch | `arrived * weight * gate` with `weight` = `self_touch_wellness` 0.0 (args.json:155) -> zeros (`_self_touch_wellness`, lem_model.py:8365; added at :3097) | the tactile token's column 12 | |

The wellness gradient the motor controller reads is `d total_wellness / d motor_commands` (lem_model.py:3101-3106); only motor_state reaches it.

## 3. The goal-ring value `wball` (lem_model.py:4148-4281), per model step and body

`wball = sum of the grounded terms` (:4119; `_w0` keeps this value) + the movement drive's wellness `mv["wellness"]` (`movement_drive on`, args.json:78; `MovementDrive`, mimo_bridge/movement_drive.py, not read: open point B; its launch numbers `movement_bonus` 0.2, `movement_penalty` 0.25, `movement_mref` 24.0, args.json:32-33, :152) + the agency payment `ag = 0.1 * _agency_pay_ema + w_prior * motion * fov` (`agency_weight` 0.1, args.json:70; `agency_prior` 0.0, :71, so `w_prior = 0`; `_agency_pay_ema` not transcribed: open point B; lem_model.py:4152-4191) + `0.5 * consequence kinds` (`consequence_w` 0.5, args.json:72; `consequence_kinds` of the world's task record, world-only rule `consequence_self_contact exclude`, args.json:183; lem_model.py:4218) + the entity self's value (`entity_self dark`, args.json:93; `_self_value.value(...)`, exactly 0 while its gate is shut, lem_model.py:4241) + `0.25 * drive` (`curiosity_goal on`, `curiosity_mode additive`, args.json:89, :182; the drive from the band `0.02:0.4`, args.json:84; lem_model.py:4258). The ring stores `(tokens, wball)` (lem_model.py:4271-4272), the pose (:4270), `_w0`, `cq`, the drive multiplier 0.0 (:4273-4279) and the channel record (:4280; section 4).

Under `wellness_rank normalized` the stored `wball` is a record only: the goal choice replaces `wl_stack` by the normalised ranking (:4265-4270; planner_loss.md section 1).

## 4. The ranking channels (`_wellness_channel_vector`, lem_model.py:7991-8051; names `_rank_channels_active`, :7895)

The released run's 16 rows (receipt `wellness_channels.channels`): integrity, controllability, learning_progress, coherence, contact_control, consequence, self_value, agency, prediction_error, tactile_comfort, capability, movement, curiosity, efficiency, pain, harm.

| Channel | Value this model step | Source |
| --- | --- | --- |
| coherence, prediction_error, tactile_comfort, capability | the grounded terms above | :8003-8004 |
| pain | `contact_response(...)["pain"]` on the same skin features (`_channel_pain`, :7931); NaN when no tactile sense was measured | :8006 |
| harm | `max(pain, 1 - motor_state)` (NaN pain -> NaN) | :8005, :8011 |
| integrity | `1 - harm` | :8012 |
| controllability | `wc.control.value(B)`: the running agreement of the efference prediction with the achieved change over 5 commands, counted only while its evidence gate is open (wellness_channels.py:208-263: `CONTROL_FLOOR` efference 6.67, command 4.91, `CONTROL_EMA_ORDERS` 5, `CONTROL_GATE_READINGS` 32); the gate and the averages live in `runtime_state.pt['wellness_channels']` (lem_model.py:9668, :9962) | :8014-8015 |
| learning_progress | `wc.lp.channel(B)`: a learning-time quantity (prediction-error improvement over two horizons; `lp_holdout_steps` 50 per the receipt) | :8016-8017 |
| contact_control | `_contact_control_value` from the contact head (`contact_head on`, args.json:184; weight 2.0 from `contact_control_wellness`, args.json:186) | lem_model.py:8018-8021 |
| consequence | `cq` (section 3) | :8022-8023 |
| self_value | the entity reader's raw value when its gate is open, else 0 | :8024-8030 |
| agency | `ag` (section 3) | :8031-8033 |
| movement | `mv["wellness"]` | :8034-8036 |
| curiosity | the drive | :8037-8039 |
| efficiency | `-(v[23] + v[24])` of the raw vestibular vector (effort_power + effort_holding); NaN on a padded read; weight 0.0 | :8044-8051 |

Storage: `detrend_moment` subtracts each ranked channel's running average over 10 model steps (`RING_DETREND_STEPS`, wellness_channels.py:382) and the normaliser learns the stored values (lem_model.py:8092-8095); pain and harm stay raw.

## 5. WELLNESS_PREDICTOR (`ApproximateEvaluator`) and the parameter count

`ApproximateEvaluator` (wellness.py:73-91): Linear(960, 256), GELU, Linear(256, 128), GELU, Linear(128, 5), Sigmoid (`hidden_dims` [256, 128], `n_outputs` 5, master_config.yaml:346-347; input 5 x 192 = 960, receipt `wellness.approximate_input_dim` 960: the active senses without vestibular, `APPROXIMATE_EXCLUDED` wellness.py:112, :174-177). Its input at the live call is the raw tokens of vision, tactile, audio, proprioception, motor in that sequence (`canonical_tokens`, wellness.py:568-576; lem_model.py:3051-3052); its five outputs mirror `APPROXIMATE_TERMS` (wellness.py:94-95). It reaches no goal choice and no command in the released configuration (`prospective_wellness off`), so the Swift scorer may skip it; the loader still lists its tensors.

Count check against the receipt's `component_params.wellness` 295,914: approximate 960 x 256 + 256 + 256 x 128 + 128 + 128 x 5 + 5 = 279,557; the five unused grounded MLPs (`GroundedEvaluator`, :57-70, input -> hidden -> 1 with sigmoid; dims master_config.yaml:329-344 with motor_state input 234 on this body, body_layout.py:54): prediction_error 1 x 32 + 32 + 32 + 1 = 97; tactile_comfort 8 x 32 + 32 + 33 = 321; capability 21 x 32 + 32 + 33 = 737; coherence 97; motor_state 234 x 64 + 64 + 65 = 15,105; sum 16,357. 279,557 + 16,357 = 295,914: equal.

## 6. What an inference-only session can and cannot compute (open, not decided here)

- Computable on the device from the frame: prediction_error (needs the previous model step's prediction of this model step), tactile_comfort and pain (need the skin exposure history, touch_pooler.md section 5), capability, motor_state, harm, integrity, efficiency, consequence (needs the world's task record), movement (needs the movement drive's state), curiosity (needs the next-model-step error band).
- Dependent on learning-time state: coherence (the body map's membership, which is learned in place), controllability, learning_progress, contact_control, self_value (gates and running averages in `runtime_state.pt`). With frozen weights these either stay at their saved values or are dropped from the ranking; the specification's REQ-B-12 fixture records the 21 scores and the selected command, which do not depend on them, but the GOAL choice does. Open point C.

## 7. Open points

A. `DifferentiableBodySchema` coherence (lem/model/body_schema.py) and `_schema_coherence`: read before the Swift coherence value is fixed.
B. `MovementDrive` (mimo_bridge/movement_drive.py), the agency payment EMA and the curiosity drive: not transcribed.
C. Which ranking channels the device session keeps is an open decision; this file lists what each needs.
D. The skin exposure memory (`lem/skin_mechanics.py`) feeds pain; its update rule was not read in this item.

## 8. Reader's check (acceptance: 3 constants per file)

| Constant | Claimed value | File:line | Checked by | Date |
| --- | --- | --- | --- | --- |
| `TACTILE_PLEASANT_PEAK`, `TACTILE_PEAK_WIDTH` | 0.05, 0.6 | contact_wellness.py:17-18 | WI-16 adversarial reviewer (not the writer) | 2026-11-04 05:56 PM PST |
| approximate input | 960 = 5 x 192 | model_receipt.json `wellness.approximate_input_dim`; wellness.py:112 | WI-16 adversarial reviewer (not the writer) | 2026-11-04 05:56 PM PST |
| `RING_DETREND_STEPS` | 10 | wellness_channels.py:382 | WI-16 adversarial reviewer (not the writer) | 2026-11-04 05:56 PM PST |

Signature of the reader: WI-16 adversarial reviewer (not the writer), 2026-11-04 05:56 PM PST. The three constants above were re-read at the cited lines; every other file:line cite in this file was re-read against the branch's working tree, and the cites into lem_model.py, train_mimo.py and hand_response.py were moved to the lines commit 23dae32 (5:31 PM PST) and the uncommitted edits after it gave them.
