# Motor controller input: the 1,646 numbers and the network they enter

Work item WI-16, target A. Written 2026-11-04 05:21 PM PST on branch feature/apple-port-2026-11-04 from a complete read of `Source/Python/lem/model/motor_controller.py` (251 lines), the cited lines of `lem_model.py`, `lem/model/wellness.py`, `lem/master_config.yaml`, and the released receipt `weights/b1_continue_B1K_skip13_492800/run/checkpoints/ckpt_step_494025/model_receipt.json`. Paths are relative to `D:/UEMA_Research/Source/Python`. Contract for the Swift type `MotorController` (WI-36, REQ-B-10).

## 1. The composition of 1,646 (receipt `motor_controller._input_dim` 1646)

`motor_input = cat([pooled, flat_handles, flat_wgrad], -1)` (motor_controller.py:158-160):

| Block | Width | Content | Source |
| --- | --- | --- | --- |
| `pooled` | 6 x 192 = 1,152 | `latents[:, -6:, :]` reshaped: the forward model's output for the NEWEST model step, one 192-number vector per sense in the sequence vision, tactile, audio, proprioception, motor, vestibular (receipt `active_modalities`), laid end to end | motor_controller.py:142-147; `n_active` 6 (:44) |
| `flat_handles` | 26 x 10 = 260 | `handle_state` = the achieved motor half (proprio_rows.md section 5: position cm / 200, quaternion xyzw, commanded radius, commanded strength, mask 1.0) | lem_model.py:2608-2610, :3107; motor_controller.py:159 |
| `flat_wgrad` | 26 x 9 = 234 | the wellness gradient (section 2) | motor_controller.py:152-156, `_wellness_grad_dim` :62 |

1,152 + 260 + 234 = 1,646.

## 2. The wellness gradient [26, 9] (lem_model.py:2999-3001, :3099-3106)

- `mc = handle_state[:, :, :9]` (float, detached, cloned); `mc[..., 0:3] -= mean over the 26 handles of mc[..., 0:3]` (the position centred on the handle centroid, :3000); `motor_commands = mc.requires_grad_(True)`.
- `total_wellness = sum over all bodies and all grounded wellness terms` (:3099); `wellness_grad = d total_wellness / d motor_commands` with `create_graph=True`, zeros when the gradient is unused or the call fails (:3101-3106).
- Only the motor_state term has a path to `motor_commands` (wellness.py:312-318 "Only the motor-state term has a path to the command"); with `motor_state_formula boundary` (args.json:28): `rom = rom_limit * 0.21` (`BODY_REACH` 0.21, wellness.py:365; `rom_limit` None at the live call, so `_rom_scale` (wellness.py:302-308) returns the module's `_rom_limit`, set to the phase's `rom_limit` by `set_developmental_limits` (wellness.py:191; called at lem_model.py:2994), 1.0 in the mature phase), `threshold = 0.1 rom`, `margin = rom - |pos|`, `violation = relu(threshold - margin)`, `q = violation / threshold`, `penalty = pen_soft + (pen_hard - pen_soft).detach()` with `pen_hard = min(q, 1)^2`, `pen_soft = q^2` for q <= 1 else `2q - 1`, score = `1 - mean over 26 x 3 of penalty` (wellness.py:334-395). So `wellness_grad[b, h, 0:3] = -(1 / 78) d penalty / d pos` per coordinate, nonzero only where `|pos| > 0.9 rom` (the last 10 % of 0.21 in position_range units: |pos| > 0.189, 37.8 cm from the handle centroid); columns 3-8 are 0.
- This gradient is computed from the same `handle_state`; no learning is involved, so the Swift session computes it in closed form (the derivative above) or by MLX `grad` on the motor_state formula.

## 3. The network (`MotorController`, motor_controller.py:38-119) with the released settings

| Setting | Value | Source |
| --- | --- | --- |
| `_input_dim` | 1646 | receipt |
| `output_per_handle` | 9 | receipt; master_config.yaml:309 |
| `n_layers`, `hidden_dim` | 2, 128 | receipt; lem_model.py `LITE_OVERRIDES` :282 |
| `adaptive`, `max_depth` | true, 4 (`think_harder on`, args.json:60 -> lem_model.py:795-797) | receipt |
| `recursive` | false | receipt |
| `confidence_hidden` | 128 | master_config.yaml:302 |
| `confidence_threshold` | 0.8 (unused in adaptive mode except as metadata) | master_config.yaml:303; motor_controller.py:239-241 |
| `initial_depth_threshold` | 0.7 -> `depth_threshold_raw = log(0.7 / 0.3)` as a learned parameter | master_config.yaml:308; motor_controller.py:80-82 |
| `thinking_energy_rate` | 0.001 (read, never used in `forward`) | master_config.yaml:307; motor_controller.py:56 |

Layers (adaptive branch, :76-98): `confidence_head`: Linear(1152, 128), GELU, Linear(128, 1), Sigmoid (:66-71); `input_proj`: Linear(1646, 128) (:89); `residual_blocks`: 4 x [Linear(128, 128), GELU, Linear(128, 128)] (:90-97); `output_proj`: Linear(128, 234) (:98); `depth_threshold_raw` scalar (:82).

Parameter count check against the receipt's `component_params.motor_controller` 520,812: input_proj 1646 x 128 + 128 = 210,816; residual blocks 4 x 2 x (128 x 128 + 128) = 132,096; output_proj 128 x 234 + 234 = 30,186; confidence head 1152 x 128 + 128 + 128 x 1 + 1 = 147,713; depth threshold 1. Sum 520,812: equal.

## 4. Forward pass per model step (`forward`, motor_controller.py:121-244; called lem_model.py:3107 with `depth=None`)

1. `pooled = latents[:, -6:, :].reshape(B, 1152)`; `confidence = confidence_head(pooled)` [B, 1] (:146-150).
2. `flat_wgrad = wellness_grad.reshape(B, 234)` (zeros when None); `flat_handles = handle_state.reshape(B, 260)`; `motor_input` [B, 1646] (:152-160).
3. `depth_threshold = sigmoid(depth_threshold_raw)`; `raw_depth = 1 + 3 sigmoid(5 (depth_threshold - confidence))` [B, 1] (:163-168).
4. `x = input_proj(motor_input)`; for block i = 0..3: `weight_i = sigmoid(5 (raw_depth - (i + 1)))`; `x = weight_i * block_i(x) + x` (:183-186: all four blocks always run; the soft gate is the released behaviour at inference too, "used for both training and inference", :178-182).
5. `output = output_proj(x).reshape(B, 26, 9)` (:188, :204).
6. Activations (:224-229): `pos = raw_pos + (clamp(raw_pos, -1, 1) - raw_pos).detach()` (the value is the clamp); `rot = rot / max(|rot|, 1e-8)`; `[radius, strength] = sigmoid(output[..., 7:9])`; `output = cat([pos, rot, rad_str], -1)`.
7. Returns `(output [B, 26, 9], confidence [B, 1], metadata {pos_excursion, depth 4, confidence mean, depth_threshold})`.

Output units: position in position_range units (x 200 = cm after the execution transform), quaternion xyzw, radius 0-1, strength 0-1. The command the body receives is this output after smoothing, exploration noise (off in the app) and the execution transform (planner_loss.md section 3 step 2).

## 5. Weight-level facts the loader must know

- `motor_input_fold on` (args.json:123; receipt `motor_input_fold.folded true, folded_at_step 415075`): the first layer (`input_proj`, the token columns) and `confidence_head.0` were rewritten in place at that model step; the folded weights are what `model.safetensors` holds. Whether `lem/model/motor_input_fold.py` (408 lines, not read in this item) also applies a runtime centre at inference is open point A.
- `liveness_repair on` (args.json:224; receipt `liveness_repair.layers` includes `motor_controller.residual_blocks.0.0` with 127 revived units and `motor_controller.confidence_head.0` with 84): the revivals were applied to the weights during training; whether `lem/model/liveness.py` (1,250 lines, not read) keeps a runtime hook on these layers at inference is open point B.
- Precision: the released run computed this module under bf16 autocast on cuda (receipt `precision bf16`); the stored weights are float32; the Swift module runs float32 (REQ-A-2's table).

## 6. Open points

A. Runtime effect of `motor_input_fold` at inference: read `lem/model/motor_input_fold.py` and the receipt's `motor_input_fold.readers` before the Swift forward is fixed.
B. Runtime effect of `liveness_repair` at inference: read `lem/model/liveness.py`.
C. The eye row (handle 23) of `output` drives the gaze (command_decoder.md section 1); its position, radius and strength are never read by the body.

## 7. Reader's check (acceptance: 3 constants per file)

| Constant | Claimed value | File:line | Checked by | Date |
| --- | --- | --- | --- | --- |
| `_input_dim` | 1646 | model_receipt.json `motor_controller._input_dim` | WI-16 adversarial reviewer (not the writer) | 2026-11-04 05:56 PM PST |
| depth gate | `1 + 3 sigmoid(5 (thr - conf))`, `sigmoid(5 (raw_depth - (i + 1)))` | motor_controller.py:167-168, :185 | WI-16 adversarial reviewer (not the writer) | 2026-11-04 05:56 PM PST |
| `BODY_REACH` | 0.21 | wellness.py:365 | WI-16 adversarial reviewer (not the writer) | 2026-11-04 05:56 PM PST |

Signature of the reader: WI-16 adversarial reviewer (not the writer), 2026-11-04 05:56 PM PST. The three constants above were re-read at the cited lines; every other file:line cite in this file was re-read against the branch's working tree, and the cites into lem_model.py, train_mimo.py and hand_response.py were moved to the lines commit 23dae32 (5:31 PM PST) and the uncommitted edits after it gave them.
