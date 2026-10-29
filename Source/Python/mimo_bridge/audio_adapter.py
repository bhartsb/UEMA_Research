# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The crib's sound: spectrum-space synthesis into the avatar's own hearing contract.

MuJoCo is silent. The UE avatar hears through HearingCaptureComponent: a 256-bin
stereo log-dB spectrum plus binaural metadata (loudness difference between ears,
arrival-time difference, azimuth), written into the frame's audio slot. This module
produces exactly that from crib events - no waveforms, straight to spectrum - so the
model's audio encoder (already trained-in-place, already loading from checkpoints)
receives the same contract on both bodies.

Sources:
  * impact  - a body contact (from the collision list the touch pipeline already
              packs): broadband thump, level from impulse, ~120 ms decay
  * squeak  - the kick panel's response: narrow band near 2.5 kHz + a half-strength
              harmonic, ~250 ms decay
  * buzz    - the bee, continuous while it exists: a 200 Hz wing-tone comb with
              rolled-off harmonics, level from distance

Ear model, deliberately simple physics: two ears 7 cm apart on the head; per-source
inverse-square level with a head-shadow factor on the far ear; arrival-time
difference from the actual ear-distance difference at 343 m/s. Levels sum as MAGNITUDES
(each source's gain times its spectral shape; speech joins as the square root of its
power - see the comment in render), then 10*log10 of that sum is mapped from -80..0 onto
[0, 1] like the UE component. One 0.125 step is therefore a tenfold rise in the summed
level, which for an amplitude is 20 conventional decibels, not 10.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

N_BINS = 256
F_LO, F_HI = 40.0, 8000.0
EAR_HALF_M = 0.035
SPEED_MS = 343.0
DB_FLOOR, DB_SPAN = -80.0, 80.0
# body-contact ("tapping") loudness relative to a raw strike. The bee source is 0.8;
# 0.15 puts a typical contact well under the bee's effective loudness so self-caused
# taps stay audible as a self-signal without masking the ambient critters (operator
# 2026-09-06: "make the body's sounds less loud than the bee"). Applies to the model's
# spectral input via the stored gain.
BODY_CONTACT_GAIN = 0.15
# after BODY_CONTACT_GAIN cut the source and the old x1.5 boost was removed, the video
# thump sits ~0.11x the bee at scale 1.0; 1.8 lifts it to ~0.20x to MATCH what the model
# hears (peak contact 0.21x the bee) - "what is in the video should be what is heard"
# (operator 2026-09-06). Measured against the bee in verify_body_vs_bee.
KNOCK_TRACK_SCALE = 1.8


def _load_wav_mono(path: str, sr: int = 16000) -> Optional[np.ndarray]:
    import wave
    try:
        with wave.open(path, "rb") as w:
            if w.getframerate() != sr or w.getnchannels() != 1:
                return None
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32767.0
        rms = float(np.sqrt((x ** 2).mean()))
        return (x * (0.15 / rms)) if rms > 1e-6 else None
    except Exception:
        return None


def _shape_from_wave(wave_x: np.ndarray, freqs: np.ndarray, sr: int = 16000) -> np.ndarray:
    """A recording's average spectral shape mapped onto the model's 256 log bins,
    so the model's coarse ear and the video's ear share one timbre."""
    sp = np.abs(np.fft.rfft(wave_x))
    fr = np.fft.rfftfreq(len(wave_x), 1.0 / sr)
    shape = np.interp(freqs, fr, sp)
    m = float(shape.max())
    return (shape / m) if m > 0 else shape


class CribAudio:
    def __init__(self, dt: float, sample_dir: Optional[str] = None):
        self.dt = float(dt)
        self.freqs = np.geomspace(F_LO, F_HI, N_BINS).astype(np.float64)
        self._impact_shape = np.exp(-self.freqs / 1500.0)
        self._rng = np.random.default_rng(7)
        # REAL recordings when present (operator order: sounds that sound right).
        # bee_loop.wav: public-domain bumblebee field recording (Wikimedia Commons,
        # "Hummel bee.ogg", author soerena). dragonfly_loop.wav: the same recording
        # sped 1.6x and high-passed - a fast-winged papery whirr. Synthesis remains
        # the fallback when the files are absent.
        import os as _os
        sd = sample_dir or _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                         "assets", "audio")
        self.bee_wave = _load_wav_mono(_os.path.join(sd, "bee_loop.wav"))
        self.fly_wave = _load_wav_mono(_os.path.join(sd, "dragonfly_loop.wav"))
        # the bee's voice is the operator-ACCEPTED deep 95 Hz synthesis (v6-v9 stood
        # without complaint); the sample experiment replaced it and regressed
        # (adversarial finding F2, 2026-09-06). Accepted voices are immovable.
        self._buzz_shape_sample = None
        self._flutter_shape_sample = (_shape_from_wave(self.fly_wave, self.freqs)
                                      if self.fly_wave is not None else None)
        # synthesis fallback shapes (used only when the samples are absent)
        self._flutter_shape = (0.8 * np.exp(-0.5 * ((self.freqs - 180.0) / 120.0) ** 2)
                               + 0.25 * np.exp(-0.5 * ((self.freqs - 700.0) / 350.0) ** 2))
        self._buzz_wob = 0.0
        self._transients: List[dict] = []    # {"shape", "pos", "t0", "tau", "gain"}
        self._t = 0.0
        from .speech import SpeechStream
        self.speech = SpeechStream()

    def sample_waves(self) -> Dict[str, Optional[np.ndarray]]:
        # buzz: None on purpose - the accepted synth bee plays, never the sample
        return {"buzz": None, "flutter": self.fly_wave}

    # ------------------------------------------------------------------ sources
    def impact(self, pos_m, impulse_ns: float) -> None:
        # gain scaled to REAL strike impulses (measured 0.2-0.45 Ns; the old /5
        # scale left true hits nearly silent), and a rate cap so even a wild
        # thrash can never machine-gun: at most 3 thumps alive per 0.2 s
        g = float(np.clip(impulse_ns / 0.5, 0.0, 1.0))
        if g < 0.05:
            return
        young = sum(1 for tr in self._transients
                    if tr.get("kind") != "squeak" and (self._t - tr["t0"]) < 0.2)
        if young >= 3:
            return
        # the 0.05 audibility gate and the 3-per-0.2s rate cap stay on the RAW impulse;
        # only the stored loudness is scaled down so body contacts sit under the bee
        self._transients.append({"shape": self._impact_shape, "pos": np.asarray(pos_m, float),
                                 "t0": self._t, "tau": 0.12, "gain": g * BODY_CONTACT_GAIN})

    def chime(self, pos_m) -> None:
        """The mobile's strike voice (operator decision 2026-09-06): a 12 g plastic
        element physically cannot reach the impact gate (impulse ceiling ~0.004 Ns vs
        the 0.15 body-scale gate), so batting - the certified deliberate act - was
        SILENT. A struck light bar: fundamental 880 Hz with a 2.4x partial, fast
        decay. FIRST PASS - the operator's ear owns this voice; gain sits below the
        squeak's 3.0 on purpose (a tap, not a toy honk)."""
        self._transients.append({"kind": "chime", "pos": np.asarray(pos_m, float),
                                 "t0": self._t, "tau": 0.22, "gain": 1.6})

    def squeak(self, pos_m) -> None:
        # a rubber-toy squeak is a pitch BEND, not a beep: the band glides
        # 1.8 -> 3.2 -> 1.8 kHz over its quarter second (rendered per-frame)
        self._transients.append({"kind": "squeak", "pos": np.asarray(pos_m, float),
                                 "t0": self._t, "tau": 0.28, "gain": 3.0})
        # 3.0, was 1.2: measured in the mix, the squeak peaked at HALF the bee's
        # band energy - present at every press, audible at none (operator confirmed)

    # ------------------------------------------------------------------ render
    def render(self, head_pos, right_axis, fwd_axis,
               buzz_pos=None, flutter_pos=None, dt: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """One frame of hearing. Returns (spectral [256, 2] float32 in [0, 1], meta).
        The bee's buzz throbs at 5 Hz - DELIBERATELY slower than its 20 Hz wings: at
        the model's ~20 Hz hearing read rate a 20 Hz tremolo would alias to nothing,
        so the audible throb is a slow envelope, not the wingbeat itself (recorded
        2026-09-06; the old docstring claimed a 20 Hz match that was never there).
        The dragonfly's flutter pulses at 18 Hz, matching its wings."""
        # The CALLER owns the clock. self.dt is only the construction-time default: a
        # caller whose step length changed after construction would otherwise advance the
        # speech cursor and the transient decay at the OLD rate while its own simulated
        # time moved at the new one. Omitting dt keeps every existing caller bit-identical.
        step_dt = self.dt if dt is None else float(dt)
        self._t += step_dt
        head = np.asarray(head_pos, float)
        right = np.asarray(right_axis, float)
        fwd = np.asarray(fwd_axis, float)
        ear_l = head - EAR_HALF_M * right
        ear_r = head + EAR_HALF_M * right
        p_l = np.zeros(N_BINS)
        p_r = np.zeros(N_BINS)
        best = None   # loudest source for the binaural metadata
        srcs = []
        for tr in list(self._transients):
            age = self._t - tr["t0"]
            a = float(np.exp(-age / tr["tau"]))
            if a < 0.02:
                self._transients.remove(tr)
                continue
            if tr.get("kind") == "chime":
                # struck light bar: two narrow partials, 880 Hz + 2.4x, brightness
                # fading with age so the spectrum dulls as the strike rings down
                shape = np.zeros(N_BINS)
                for k_f, k_g in ((880.0, 1.0), (880.0 * 2.4, 0.45 * a)):
                    shape += k_g * np.exp(-0.5 * ((self.freqs - k_f) / 55.0) ** 2)
                srcs.append((shape, tr["pos"], tr["gain"] * a, ("chime", id(tr), a)))
            elif tr.get("kind") == "squeak":
                # rubber-toy reed: fundamental glides 500 -> 1200 -> 500 Hz with a
                # bright harmonic stack (toy squeaks are LOW and reedy, not a whistle)
                c = 500.0 + 700.0 * float(np.sin(np.pi * np.clip(age / 0.25, 0.0, 1.0)))
                shape = np.zeros(N_BINS)
                for k in range(1, 5):
                    shape += (1.0 / k ** 0.7) * np.exp(
                        -0.5 * ((self.freqs - c * k) / (60.0 + 0.06 * c * k)) ** 2)
                srcs.append((shape, tr["pos"], tr["gain"] * a,
                             ("squeak", id(tr), c)))
            else:
                srcs.append((tr["shape"], tr["pos"], tr["gain"] * a,
                             ("impact", id(tr), age)))
        if buzz_pos is not None:
            # a REAL bumblebee recording's spectral shape when the sample is present
            # (public domain, Wikimedia Commons); the synthesized deep comb otherwise.
            self._buzz_wob = float(np.clip(self._buzz_wob * 0.995
                                           + 0.006 * self._rng.standard_normal(), -0.05, 0.05))
            f0 = 70.0 * (1.0 + self._buzz_wob)   # operator 2026-09-06: lower and deeper still
            if self._buzz_shape_sample is not None:
                comb = self._buzz_shape_sample
            else:
                comb = np.zeros(N_BINS)
                for k in range(1, 11):
                    roll = k ** 0.9 if k <= 4 else k ** 2.0
                    comb += (1.0 / roll) * np.exp(-0.5 * ((self.freqs - f0 * k)
                                                          / (18.0 + 0.02 * f0 * k)) ** 2)
                comb = comb * (1.0 + 1.5 * np.exp(-0.5 * ((self.freqs - 350.0) / 220.0) ** 2))
            am = 1.0 + 0.25 * float(np.sin(2 * np.pi * 5.0 * self._t))
            srcs.append((comb, np.asarray(buzz_pos, float), 0.8 * am, ("buzz", 0, f0)))
        if flutter_pos is not None:
            # fast wings (operator): mild breathing at 18 Hz over the recording's own
            # fast-whirr texture; the sample (bee sped 1.6x, high-passed) carries the
            # beat character. Quiet second voice at 0.3 of the bee's gain.
            pulse = 0.8 + 0.2 * float(np.sin(2 * np.pi * 18.0 * self._t))
            fsh = (self._flutter_shape_sample if self._flutter_shape_sample is not None
                   else self._flutter_shape)
            srcs.append((fsh, np.asarray(flutter_pos, float),
                         0.3 * pulse, ("flutter", 0, pulse)))
        rec = {"buzz": None, "flutter": None, "transients": []}
        for shape, pos, gain, tag in srcs:
            v = pos - head
            r = float(np.linalg.norm(v))
            u = v / max(r, 1e-6)
            lateral = float(np.dot(u, right))            # + = right side
            r_l = max(float(np.linalg.norm(pos - ear_l)), 0.05)
            r_r = max(float(np.linalg.norm(pos - ear_r)), 0.05)
            # floored 1/r^1.5, not inverse-square: at crib distances (0.1-0.6 m) the
            # square law spread sources 100x apart and peak normalization then buried
            # everything but the nearest thump (operator: no bee, no squeak, only
            # pops). This law keeps the whole scene within ~5x.
            att_l = 0.02 / max(r_l, 0.2) ** 1.5
            att_r = 0.02 / max(r_r, 0.2) ** 1.5
            gl = gain * att_l * (1.0 - 0.35 * max(0.0, lateral))
            gr = gain * att_r * (1.0 - 0.35 * max(0.0, -lateral))
            p_l += gl * shape
            p_r += gr * shape
            kind, sid, extra = tag
            if kind == "buzz":
                rec["buzz"] = (gl, gr, extra)            # extra = wandering f0
            elif kind == "flutter":
                rec["flutter"] = (gl, gr, extra)         # extra = pulse 0..1
            else:
                rec["transients"].append((kind, sid, gl, gr, extra))
            loud = float(((gl + gr) * .5 * shape).mean())
            if best is None or loud > best[0]:
                itd_ms = (r_r - r_l) / SPEED_MS * 1000.0
                ild_db = 20.0 * np.log10(max(gl, 1e-9) / max(gr, 1e-9))
                az = float(np.arctan2(lateral, max(float(np.dot(u, fwd)), -1.0)))
                # fold rear azimuths to their front mirror (150 deg -> 30 deg): two
                # ears without a pinna measure IDENTICAL timing/loudness differences
                # for a source and its front mirror, so the fold is what the ear
                # physically knows - and it keeps the value inside the byte
                # contract's +-90 deg (a supine baby's under-body thumps were
                # emitting az up to +-180, normalized to +-2 where the tokenizer
                # expects +-1). Front-vs-back still travels as its own field below.
                if abs(az) > np.pi / 2:
                    az = float(np.sign(az) * (np.pi - abs(az)))
                best = (loud, itd_ms, ild_db, az, abs(float(np.dot(u, fwd))))
        speech_power, speech_meta = self.speech.render(step_dt, head, right, fwd, self.freqs)
        # The existing synthesized sources use gain * spectral shape (magnitude),
        # despite the historic p_l/p_r names. Mixing squared PCM magnitudes into
        # those units buried speech: measured speech/bee was 1/1514 at crib distances.
        # Match the adapter's existing magnitude convention without changing its voices.
        speech_magnitude = np.sqrt(speech_power)
        speech_level = float(speech_magnitude.mean())
        p_l += speech_magnitude[:, 0]
        p_r += speech_magnitude[:, 1]
        if speech_level > 0 and (best is None or speech_level > best[0]):
            best = (speech_level, speech_meta['itd_ms'], speech_meta['ild_db'],
                    speech_meta['azimuth'], speech_meta['front_back'])
        rec['speech_pcm'] = self.speech.last_pcm.copy()
        rec['speech_sr'] = self.speech.sample_rate
        spec = np.stack([p_l, p_r], axis=-1)
        db = 10.0 * np.log10(spec + 1e-9)
        spec01 = np.clip((db - DB_FLOOR) / DB_SPAN, 0.0, 1.0).astype(np.float32)
        self.last_spec01 = spec01
        # the video's ear records SOURCES, not the model's coarse bins: rebuilding
        # audio from 256 log bins turns every tone into a cluster of detuned sines
        # (measured by ear, operator: "white noise with a machine gun"). The model
        # keeps the coarse spectrum; the soundtrack renders the same sources clean.
        rec["t"] = self._t
        self.last_sources = rec
        meta = {"left_rms": float(np.sqrt((p_l ** 2).mean())),
                "right_rms": float(np.sqrt((p_r ** 2).mean())),
                "ild_db": float(best[2]) if best else 0.0,
                "itd_ms": float(best[1]) if best else 0.0,
                "azimuth": float(best[3]) if best else 0.0,
                "front_back": float(best[4]) if best else 0.0}
        return spec01, meta


def render_track(frames_src, dt: float, sr: int = 16000,
                 samples: Optional[Dict[str, Optional[np.ndarray]]] = None) -> np.ndarray:
    """The soundtrack, rendered CLEAN from the recorded sources.

    Why not from the model's spectrum: 256 log bins smear every tone across ~15
    bins, and playing bins as sines makes detuned clusters - heard as white noise
    (operator, 2026-09-05). The model keeps its coarse spectrum; this renders the
    SAME sources (same gains, same sidedness, same moments) the way a microphone at
    the head would catch them: phase-continuous harmonics for buzz and squeak, a
    soft-enveloped noise band for wing flutter, one-shot thumps for impacts."""
    T = len(frames_src)
    if T == 0:
        return np.zeros((0, 2), dtype=np.int16)
    hop = max(1, int(round(sr * dt)))
    n = T * hop
    fi = np.arange(T)
    si = np.arange(n) / hop                       # sample time in frame units
    out = np.zeros((n, 2), dtype=np.float64)

    def _interp(vals):
        return np.interp(si, fi, np.asarray(vals, dtype=np.float64))

    def _tile(wave_x):
        reps = int(np.ceil(n / len(wave_x)))
        return np.tile(wave_x, reps)[:n]

    # ---- bumblebee: the REAL recording, looped, under the recorded per-frame gains;
    # synthesized deep comb only when no sample is available
    bz = [fs.get("buzz") for fs in frames_src]
    if any(b is not None for b in bz):
        gl = _interp([b[0] if b else 0.0 for b in bz])
        gr = _interp([b[1] if b else 0.0 for b in bz])
        sw = (samples or {}).get("buzz")
        if sw is not None:
            tone = _tile(sw) * 12.0          # sample normalized to 0.15 RMS
        else:
            f0 = _interp([b[2] if b else 70.0 for b in bz])
            phase = np.cumsum(2 * np.pi * f0 / sr)
            tone = np.zeros(n)
            for k in range(1, 9):
                roll = k ** 1.1 if k <= 3 else k ** 2.2
                tone += np.sin(k * phase) / roll
        out[:, 0] += gl * tone
        out[:, 1] += gr * tone
    # ---- dragonfly wingbeat: a low tonal WUB per visible flap (sine swept 160->90 Hz
    # under the smooth envelope) with the dry rustle reduced to quiet texture -
    # "white noisy rapid flapping" was the noise band carrying the whole sound
    fl = [fs.get("flutter") for fs in frames_src]
    if any(f is not None for f in fl):
        gl = _interp([f[0] if f else 0.0 for f in fl])
        gr = _interp([f[1] if f else 0.0 for f in fl])
        env = _interp([f[2] if f else 0.0 for f in fl])
        # a TONAL flap thrum in the speaker-audible midrange. The 90-160 Hz wub
        # vanished on small speakers, leaving only the pulsed 700 Hz noise - which is
        # the exact recipe for bubble wrap (operator, three times). Fundamental now
        # sweeps 170-240 Hz with three harmonics; the noise is cut to a whisper.
        sw = (samples or {}).get("flutter")
        if sw is not None:
            # the LOW soft flutter of a big slow-winged insect (recording pitched
            # DOWN 0.55x with its ~25 Hz wingbeat chop baked in - pitching UP had
            # built a gnat, adversarial finding F1). Scale 5, not 12: ears weigh
            # midrange ~10 dB louder than bass, so flat-RMS mixing let the flutter
            # shout over the deep bee (finding F3).
            fly = _tile(sw) * 2.5
            out[:, 0] += gl * fly
            out[:, 1] += gr * fly
        else:
            wub_f = 240.0 + 100.0 * env
            ph = np.cumsum(2 * np.pi * wub_f / sr)
            thrum = 0.25 * (np.sin(ph) + 0.5 * np.sin(2 * ph))
            rng = np.random.default_rng(3)
            wn = rng.standard_normal(n)
            spec = np.fft.rfft(wn)
            fr = np.fft.rfftfreq(n, 1.0 / sr)
            spec *= np.exp(-0.5 * ((fr - 600.0) / 280.0) ** 2)
            band = np.fft.irfft(spec, n)
            band /= max(float(np.abs(band).std()), 1e-9)
            rustle = band * (0.55 + 0.45 * env)
            out[:, 0] += gl * (thrum + rustle)
            out[:, 1] += gr * (thrum + rustle)
    # ---- transients: squeaks glide, impacts thump once
    seen = {}
    for i, fs in enumerate(frames_src):
        for kind, sid, gl, gr, extra in fs.get("transients", ()):
            seen.setdefault((kind, sid), []).append((i, gl, gr, extra))
    for (kind, sid), rows in seen.items():
        idx = np.array([r[0] for r in rows], dtype=float)
        if kind == "chime":
            start = int(rows[0][0]) * hop
            length = min(int(0.25 * sr), n - start)
            if length <= 0:
                continue
            tau = np.arange(length) / sr
            ring = (np.sin(2 * np.pi * 880.0 * tau) * np.exp(-tau / 0.09)
                    + 0.45 * np.sin(2 * np.pi * 880.0 * 2.4 * tau) * np.exp(-tau / 0.05))
            out[start:start + length, 0] += rows[0][1] * ring
            out[start:start + length, 1] += rows[0][2] * ring
        elif kind == "squeak":
            span = (si >= idx[0]) & (si <= idx[-1])
            ssi = si[span]
            gl = np.interp(ssi, idx, [r[1] for r in rows])
            gr = np.interp(ssi, idx, [r[2] for r in rows])
            f0 = np.interp(ssi, idx, [r[3] for r in rows])
            phase = np.cumsum(2 * np.pi * f0 / sr)
            tone = np.zeros(ssi.shape[0])
            for k in range(1, 5):
                tone += np.sin(k * phase) / k ** 0.9
            out[span, 0] += gl * tone
            out[span, 1] += gr * tone
        else:                                     # impact: one-shot thump
            start = int(rows[0][0]) * hop
            length = min(int(0.15 * sr), n - start)
            if length <= 0:
                continue
            tau = np.arange(length) / sr
            rng = np.random.default_rng(sid & 0xFFFF)
            # a dull soft thump, not a bubble-wrap pop: slower sine decay, and the
            # noise burst low-passed (moving average ~1 kHz) at a modest share
            nz = rng.standard_normal(length)
            nz = np.convolve(nz, np.ones(8) / 8.0, mode="same")
            knock = (np.sin(2 * np.pi * 85.0 * tau) * np.exp(-tau / 0.08)
                     + 0.35 * nz * np.exp(-tau / 0.06))
            # KNOCK_TRACK_SCALE re-levels the thump synth to the bee's loudness-per-gain
            # so the video's balance equals the model's (the old x1.5 pushed it the wrong
            # way). "What is in the video should be what is heard" (operator 2026-09-06).
            out[start:start + length, 0] += rows[0][1] * knock * KNOCK_TRACK_SCALE
            out[start:start + length, 1] += rows[0][2] * knock * KNOCK_TRACK_SCALE
    # soft limiter, not peak normalization: normalizing to the single loudest thump
    # buried every quieter source; anchoring to the 95th percentile and rounding the
    # rare peaks through tanh keeps the bee and the squeak audible beside a knock
    p95 = float(np.percentile(np.abs(out), 95)) if np.any(out) else 0.0
    if p95 > 0:
        out = np.tanh(out * (0.30 / p95))
    out *= .9
    # Add the exact ear waveform after the legacy synthesized-effect limiter.
    # Speech receives no per-clip normalization or spectrum reconstruction.
    for i, fs in enumerate(frames_src):
        pcm = fs.get('speech_pcm')
        if pcm is None or len(pcm) == 0:
            continue
        if fs.get('speech_sr', 16000) != sr:
            from scipy.signal import resample_poly
            from math import gcd
            source_sr = fs.get('speech_sr', 16000)
            g = gcd(sr, source_sr)
            pcm = resample_poly(pcm, sr // g, source_sr // g, axis=0)
        k = min(hop, len(pcm))
        out[i * hop:i * hop + k] += pcm[:k]
    return (out.clip(-1, 1) * 32767.0).astype(np.int16)


def write_wav(path: str, samples: np.ndarray, sr: int = 16000) -> None:
    import struct as _st
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.ascontiguousarray(samples).tobytes())
    _ = _st  # (wave writes the header itself)


def mux_wav_into_mp4(mp4_path: str, wav_path: str) -> bool:
    """Attach the wav as the mp4's audio track using imageio's own ffmpeg. Returns
    True when the mp4 was replaced with a sounding one."""
    import os
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        tmp = mp4_path + ".mux.mp4"
        r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", mp4_path,
                            "-i", wav_path, "-c:v", "copy", "-c:a", "aac",
                            "-shortest", tmp], capture_output=True, timeout=120)
        if r.returncode == 0 and os.path.getsize(tmp) > 0:
            os.replace(tmp, mp4_path)
            return True
        if os.path.exists(tmp):
            os.remove(tmp)
    except Exception:
        pass
    return False
