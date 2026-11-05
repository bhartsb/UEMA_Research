# Audio render: the crib's sound sources to the audio input [256, 2] + [5]

Work item WI-16, target A. Written 2026-11-04 05:21 PM PST on branch feature/apple-port-2026-11-04 from a complete read of `Source/Python/mimo_bridge/audio_adapter.py` (475 lines), `speech.py` (142 lines), `tokenizer/modalities/audio.py` (79 lines) and the cited lines of `env.py`, `caregiver.py`, `frame_encoder.py`. Paths are relative to `D:/UEMA_Research/Source/Python`. Contract for the Swift type `AudioSense` (REQ-B-7). `audio real` (args.json:86) builds `CribAudio(env.dt)` (env.py:1349-1351).

## 1. Constants

| Constant | Value | Line (audio_adapter.py unless stated) |
| --- | --- | --- |
| `N_BINS` | 256 | 34 |
| `F_LO`, `F_HI` | 40 Hz, 8000 Hz; `freqs = geomspace(40, 8000, 256)` float64 | 35, 78 |
| `EAR_HALF_M` | 0.035 m (ears 7 cm apart) | 36 |
| `SPEED_MS` | 343 m/s | 37 |
| `DB_FLOOR`, `DB_SPAN` | -80, 80 | 38 |
| `BODY_CONTACT_GAIN` | 0.15 | 44 |
| impact shape | `exp(-freqs / 1500)` | 79 |
| buzz wobble generator | `numpy.random.default_rng(7)` | 80 |
| bee sample | never used: `_buzz_shape_sample = None` by decision | 94, 107 |
| flutter sample | `dragonfly_loop.wav` in `mimo_bridge/assets/audio` when present: 16 kHz mono int16 / 32767, scaled to RMS 0.15 (`_load_wav_mono`, 52-62); shape = |rfft| interpolated onto `freqs`, divided by its maximum (`_shape_from_wave`, 65-72) | 87-96 |
| flutter synthesis fallback | `0.8 exp(-0.5 ((f - 180) / 120)^2) + 0.25 exp(-0.5 ((f - 700) / 350)^2)` | 98-99 |
| transient drop | a transient is removed once `exp(-age / tau) < 0.02` | 170-174 |
| distance law | `att = 0.02 / max(r_ear, 0.2)^1.5`, `r_ear` floored at 0.05 m | 227-234 |
| head shadow | far ear x `(1 - 0.35 max(0, +-lateral))` | 235-236 |
| speech sample rate | 16000 Hz; history 1024 samples; delay history 128 samples; gains `0.2 / max(d, 0.2)` | speech.py:16, 86-87, 122 |

## 2. Sources and when they sound

| Source | Trigger (file:line) | Transient record | Spectrum shape at render |
| --- | --- | --- | --- |
| impact | env.py:3523-3527: for every collision entry of this env step (`build_collisions`, the 16 strongest contacts, touch_adapter.py:365-388) with `frame_count <= 1` and `|impulse_ns| > 0.15` N s: `impact(point_cm / 100, |impulse|)`. In `impact` (111-125): `g = clip(impulse / 0.5, 0, 1)`; nothing below 0.05; nothing when 3 or more non-squeak transients are younger than 0.2 s | `{shape: impact shape, pos, t0, tau 0.12 s, gain g * 0.15}` | `shape` x `gain * a`, `a = exp(-age / tau)` (192-194) |
| chime (mobile struck) | env.py:3460 `_mobile_chime(touching, position)` from `_collect`'s mobile-touch flag (computed earlier in `_collect`, not transcribed); in `_mobile_chime` (3097-3111): rising edge of `touching` and `frame >= _chime_cool`; then `_chime_cool = frame + int(0.3 / dt)` | `{kind chime, pos, t0, tau 0.22 s, gain 1.6}` (127-135) | `sum over (880 Hz, 1.0), (880 x 2.4 Hz, 0.45 a)` of `g exp(-0.5 ((f - f_k) / 55)^2)` (175-181) |
| squeak (kick panel) | env.py:1614-1639 `_critters_tick`: a NEW contact between the squeaker geom and any `geom:`-named body geom, with `frame >= sq_cool_until` and no touch on the previous env step; cooldown `1.0 / dt` frames; position `sq_home` = the feet's mean position with x - 0.10 m and z = 0.10 m (1553-1561) | `{kind squeak, pos, t0, tau 0.28 s, gain 3.0}` (137-143) | `c = 500 + 700 sin(pi clip(age / 0.25, 0, 1))` Hz; `sum_{k=1..4} k^-0.7 exp(-0.5 ((f - c k) / (60 + 0.06 c k))^2)` (182-191) |
| buzz (bee) | env.py:3530-3535: `buzz_pos = mocap_pos[bee]` when `visitor == "bee"` and `bee_state != "landed"`, else None (critter state machine env.py:1548-1593 and later lines, not transcribed) | continuous | `wob = clip(0.995 wob + 0.006 N(0, 1), -0.05, 0.05)` (198-199); `f0 = 70 (1 + wob)` Hz (200); comb `sum_{k=1..10} (1 / roll_k) exp(-0.5 ((f - f0 k) / (18 + 0.02 f0 k))^2)`, `roll_k = k^0.9` for k <= 4 else `k^2` (204-208); times `1 + 1.5 exp(-0.5 ((f - 350) / 220)^2)` (209); `am = 1 + 0.25 sin(2 pi 5 t)` (210); gain `0.8 am` (211) |
| flutter (dragonfly) | env.py:3536-3537: `flutter_pos = mocap_pos[dragonfly]` when `visitor == "dragonfly"` | continuous | the flutter shape; `pulse = 0.8 + 0.2 sin(2 pi 18 t)`; gain `0.3 pulse` (212-220) |
| speech (caregiver) | caregiver.py:345-352: an event with a `phrase` key calls `env._audio.speech.start(clip, position, phrase)`; clips from the `caregiver_speech_dir` manifest (speech.py:46-71, phrases 17-26; args.json:137); `start` refuses while a clip is playing (speech.py:95-104). `caregiver_mode responsive` (args.json:132) drives events through `caregiver_interaction.tick` (caregiver.py:357-360, not read) | the clip's samples advance by `dt * 16000` per env step (speech.py:108-114) | `SpeechStream.render` (106-142): per ear, the mono samples delayed by `(d_ear - d_min) / 343 * 16000` samples (linear interpolation over a 128-sample history) and scaled by `0.2 / max(d_ear, 0.2)` times the head shadow; the last 1024 stereo samples under a Hann window; `power = |rfft|^2 / window_sum^2` interpolated onto `freqs`; the adapter adds `sqrt(power)` as MAGNITUDE (266-269) |

`t` is the adapter's own clock: `_t += dt` per render (159); `dt` is passed live from the env (env.py:3544, `self.dt` = 0.01 s).

## 3. Render, per env step (`CribAudio.render`, 146-291; called env.py:3542-3544)

Inputs: `head = cam_xpos[eye_left]`, `right = R[:, 0]`, `fwd = -R[:, 2]` with `R = cam_xmat[eye_left]` (env.py:3528-3529, 3542-3543); `buzz_pos`, `flutter_pos`.

1. `ear_l = head - 0.035 right`, `ear_r = head + 0.035 right` (163-164); `p_l = p_r = zeros(256)` (165-166).
2. Source list: every live transient (169-194), the buzz (195-211), the flutter (212-220).
3. Per source (222-260): `v = pos - head`, `u = v / max(|v|, 1e-6)`, `lateral = u . right`; `r_l = max(|pos - ear_l|, 0.05)`, `r_r` likewise; `att_l = 0.02 / max(r_l, 0.2)^1.5`; `gl = gain att_l (1 - 0.35 max(0, lateral))`, `gr = gain att_r (1 - 0.35 max(0, -lateral))`; `p_l += gl shape`, `p_r += gr shape`. The loudest source by `mean((gl + gr) / 2 shape)` sets the metadata: `itd_ms = (r_r - r_l) / 343 * 1000`, `ild_db = 20 log10(gl / gr)` (each floored at 1e-9), `az = atan2(lateral, max(u . fwd, -1))` folded to the front: `|az| > pi/2 -> sign(az) (pi - |az|)`; `front_back = |u . fwd|` (246-260).
4. Speech: `p_l += sqrt(power)[:, 0]`, `p_r += sqrt(power)[:, 1]` (261-269); the speech replaces the metadata when its mean magnitude exceeds the loudest source's (270-272).
5. `spec = stack([p_l, p_r], -1)` [256, 2]; `db = 10 log10(spec + 1e-9)`; `spec01 = clip((db + 80) / 80, 0, 1)` float32 (275-277).
6. `meta = {left_rms = sqrt(mean(p_l^2)), right_rms, ild_db, itd_ms, azimuth (rad), front_back}` (285-290).

## 4. Output into the frame and the encoder input

- `info["audio_spec"] = spec01`, `info["audio_meta"] = meta` (env.py:3545-3546) -> `write_audio` (frame_encoder.py:145-166): the spectrum stored planar (left 256 then right 256), the azimuth stored in DEGREES (:162), `left_rms`, `right_rms`, `ild_db`, `itd_ms`, `front_back`.
- `tokenize_audio` (tokenizer/modalities/audio.py:30-79) returns `spectral` [256, 2] float32 (the stored values) and `metadata` [5] = `[ild_db / 20, itd_ms / 1.0, azimuth_deg / 90, front_back, left_rms]`.
- The specification's "spatial [2]" (REQ-B-7, section 4.3 table "[256,2] + [2]") does not match the tokenizer's 5-number metadata read here; which numbers the audio encoder consumes is settled by the encoder's input layer in WI-31 (`lem/model/encoders.py`, not read in this item). Open point A.
- The model reads one audio input per model step: the frame of the last env step of the 5 (the same rule as the vestibular read, vestibular_sense.py:289).

## 5. Open points for the Swift implementation

A. Encoder input width: 5 metadata numbers come off the tokenizer; the specification table says 2. Read `lem/model/encoders.py` (`conv1d_stack`, receipt `encoders_built.audio`) before fixing the Swift shape.
B. The bee and dragonfly positions, visitor schedule and landing state come from the critter state machine (env.py:1548-1593 and `_critters_tick` beyond line 1639, not transcribed); the speech position (`speech.position`, read at caregiver.py:346) is written by the caregiver programs not read here. The fixture records `buzz_pos`, `flutter_pos`, the transient list and the speech cursor per env step, so the Swift render can be checked without re-implementing the critters first.
C. The buzz wobble is a seeded random walk (seed 7); exact equality within 1e-5 needs the same generator sequence (`default_rng(7).standard_normal()` draws), one draw per render with a bee present.

## 6. Reader's check (acceptance: 3 constants per file)

| Constant | Claimed value | File:line | Checked by | Date |
| --- | --- | --- | --- | --- |
| `BODY_CONTACT_GAIN` | 0.15 | audio_adapter.py:44 | WI-16 adversarial reviewer (not the writer) | 2026-11-04 05:56 PM PST |
| distance law | `0.02 / max(r, 0.2)^1.5` | audio_adapter.py:233-234 | WI-16 adversarial reviewer (not the writer) | 2026-11-04 05:56 PM PST |
| impact gate | `frame_count <= 1` and impulse > 0.15 N s | env.py:3524-3526 | WI-16 adversarial reviewer (not the writer) | 2026-11-04 05:56 PM PST |

Signature of the reader: WI-16 adversarial reviewer (not the writer), 2026-11-04 05:56 PM PST. The three constants above were re-read at the cited lines; every other file:line cite in this file was re-read against the branch's working tree, and the cites into lem_model.py, train_mimo.py and hand_response.py were moved to the lines commit 23dae32 (5:31 PM PST) and the uncommitted edits after it gave them.
