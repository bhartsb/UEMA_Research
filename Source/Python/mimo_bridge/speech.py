# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Caregiver PCM audio on the simulation clock, shared by hearing and video.

TTS runs outside the trainer. This module never imports its model or sends text
to the infant. A cached waveform can also be a recording or another TTS output.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import wave

import numpy as np

SAMPLE_RATE = 16000
PHRASES = {
    "hello": "Hello. I'm here.",
    "look": "I see you looking at me. Hello!",
    "touch": "You touched it. I heard that.",
    "ball": "Here is the ball. A little ball.",
    "cloth": "Here is the soft pad.",
    "ring": "Here is the ring.",
    "away": "I'll put this away for a little while.",
    "pause": "We can have a quiet moment.",
}


def read_pcm(path, sample_rate=SAMPLE_RATE):
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("speech cache must contain PCM16 WAV")
        rate, channels = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64) / 32768
    x = x.reshape(-1, channels).mean(1)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("speech clip is empty or nonfinite")
    if rate != sample_rate:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(rate, sample_rate)
        x = resample_poly(x, sample_rate // g, rate // g)
    return x.astype(np.float32)


class SpeechBank:
    def __init__(self, directory):
        import json
        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        self.clips = {}
        self.texts = {}
        for key, phrase in PHRASES.items():
            entry = manifest['phrases'][key]
            path = directory / entry['file']
            if entry['text'] != phrase or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f"speech cache mismatch: {key}")
            self.clips[key] = read_pcm(path)
            self.texts[key] = phrase
        self.interaction_contract=manifest.get('interaction_contract')
        self.groups={}
        if self.interaction_contract is not None:
            from .caregiver_language import CONTRACT, VARIANTS, GROUP_KEYS
            if self.interaction_contract!=CONTRACT:
                raise ValueError('unsupported caregiver language contract')
            for key,text in VARIANTS.items():
                entry=manifest['phrases'][key];path=directory/entry['file']
                if entry['text']!=text or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
                    raise ValueError(f'speech cache mismatch: {key}')
                self.clips[key]=read_pcm(path);self.texts[key]=text
            self.groups=GROUP_KEYS


class SpeechStream:
    """One adult speaker, exact waveform cursor and fractional ear delays.

    Samples advance by simulated time, not wall-clock synthesis speed. The latest
    causal 64 ms window is transformed for the model; PCM is retained for video.
    """
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.clip = np.zeros(0, dtype=np.float32)
        self.cursor = 0
        self.fraction = 0.
        self.position = np.zeros(3)
        self.history = np.zeros((1024, 2), dtype=np.float64)
        self.delay_history = np.zeros(128, dtype=np.float64)
        self.last_pcm = np.zeros((0, 2), dtype=np.float32)
        self.key = None

    @property
    def busy(self):
        return self.cursor < len(self.clip)

    def start(self, pcm, position, key=None):
        if self.busy:
            return False
        self.clip = np.asarray(pcm, np.float32).copy()
        if self.clip.ndim != 1 or not np.isfinite(self.clip).all():
            raise ValueError("speech needs a finite mono waveform")
        self.cursor = 0
        self.position = np.asarray(position, float).copy()
        self.key = key
        return True

    def render(self, dt, head, right, fwd, freqs):
        from .audio_adapter import EAR_HALF_M, SPEED_MS
        count_float = dt * self.sample_rate + self.fraction
        count = int(count_float + 1e-9)
        self.fraction = count_float - count
        mono = np.zeros(count)
        n = min(count, max(0, len(self.clip) - self.cursor))
        mono[:n] = self.clip[self.cursor:self.cursor + n]
        self.cursor += n
        vec = self.position - np.asarray(head)
        dist = max(float(np.linalg.norm(vec)), .05)
        lateral = float(np.dot(vec / dist, right))
        ears = [np.asarray(head) - EAR_HALF_M * np.asarray(right),
                np.asarray(head) + EAR_HALF_M * np.asarray(right)]
        distances = np.array([np.linalg.norm(self.position - ear) for ear in ears])
        delays = (distances - distances.min()) / SPEED_MS * self.sample_rate
        gains = .2 / np.maximum(distances, .2)
        gains *= [1 - .35 * max(lateral, 0), 1 - .35 * max(-lateral, 0)]
        combined = np.concatenate([self.delay_history, mono])
        indexes = np.arange(count) + len(self.delay_history)
        pcm = np.stack([np.interp(indexes - delays[i], np.arange(len(combined)), combined)
                        * gains[i] for i in range(2)], axis=1)
        self.delay_history = combined[-128:].copy()
        self.history = np.concatenate([self.history, pcm])[-1024:]
        window = np.hanning(len(self.history))
        spectrum = abs(np.fft.rfft(self.history * window[:, None], axis=0)) ** 2
        spectrum /= max(float(window.sum()) ** 2, 1)
        bins = np.fft.rfftfreq(len(window), 1 / self.sample_rate)
        power = np.stack([np.interp(freqs, bins, spectrum[:, i]) for i in range(2)], 1)
        self.last_pcm = pcm.astype(np.float32)
        az = float(np.arctan2(np.dot(vec, right), np.dot(vec, fwd)))
        if abs(az) > np.pi / 2:
            az = float(np.sign(az) * (np.pi - abs(az)))
        meta = dict(ild_db=float(20 * np.log10(max(gains[0], 1e-9) / max(gains[1], 1e-9))),
                    itd_ms=float((distances[1] - distances[0]) / SPEED_MS * 1000),
                    azimuth=az, front_back=abs(float(np.dot(vec / dist, fwd))))
        return power, meta
