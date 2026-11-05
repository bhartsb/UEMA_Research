# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Write audio_render.json: the Python CribAudio over a scripted 14-env-step scene.

The Swift port (apple/UEMAKit/Sources/UEMASenses/AudioReader.swift) replays the same events,
listener poses, visitor positions and Gaussian draws and compares its spectrum and metadata
(Tests/UEMASensesTests/AudioReaderTests.swift). The dragonfly's recorded timbre is left out
(sample_dir points at an empty folder, so the adapter's synthesis shape is used) because the
Swift port carries only that shape. Run from the repository root:

    python apple/UEMAKit/Tests/UEMASensesTests/Fixtures/audio/export_audio_fixture.py
"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 6))
sys.path.insert(0, os.path.join(ROOT, "Source", "Python"))
from mimo_bridge.audio_adapter import CribAudio  # noqa: E402


class RecordingRng:
    """numpy's default_rng(7), keeping each standard_normal draw."""

    def __init__(self, rng):
        self.rng = rng
        self.draws = []

    def standard_normal(self):
        x = float(self.rng.standard_normal())
        self.draws.append(x)
        return x


def rotation(yaw):
    """A camera matrix turned `yaw` radians about world z (MuJoCo cam_xmat, row-major)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.array(
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_render.json")
    with tempfile.TemporaryDirectory() as empty:
        audio = CribAudio(0.01, sample_dir=empty)
    rec = RecordingRng(audio._rng)
    audio._rng = rec
    events = {
        0: [("impact", [0.30, 0.05, 0.10], 0.40)],
        1: [("impact", [0.10, -0.20, 0.05], 0.02)],                     # below the 0.05 gate
        2: [("chime", [0.25, 0.00, 0.45])],
        3: [("squeak", [-0.10, 0.15, 0.10]),
            ("impact", [0.20, 0.10, 0.00], 0.30), ("impact", [0.20, 0.10, 0.00], 0.30)],  # 3 young: cap
        4: [("impact", [0.20, -0.10, 0.00], 0.50)],
    }
    steps = []
    for k in range(14):
        frame_events = events.get(k, [])
        for ev in frame_events:
            if ev[0] == "impact":
                audio.impact(np.asarray(ev[1]), ev[2])
            elif ev[0] == "chime":
                audio.chime(np.asarray(ev[1]))
            else:
                audio.squeak(np.asarray(ev[1]))
        head = np.array([0.02 * k, 0.0, 0.25])
        R = rotation(0.1 * k)
        buzz = [0.35, -0.10 + 0.02 * k, 0.40] if 5 <= k <= 10 else None
        flutter = [-0.20, 0.30, 0.35 - 0.01 * k] if k >= 8 else None
        spec, meta = audio.render(head, R[:, 0], -R[:, 2], buzz_pos=None if buzz is None else np.asarray(buzz),
                                  flutter_pos=None if flutter is None else np.asarray(flutter), dt=0.01)
        steps.append({
            "events": [{"kind": e[0], "position": e[1], **({"impulse": e[2]} if e[0] == "impact" else {})}
                       for e in frame_events],
            "head": head.tolist(), "xmat_row_major": R.reshape(-1).tolist(),
            "buzz": buzz, "flutter": flutter,
            "spectrum": spec.astype(float).reshape(-1).tolist(),
            "metadata": [meta[n] for n in ("left_rms", "right_rms", "ild_db", "itd_ms", "azimuth", "front_back")],
        })
    record = {
        "generator": "apple/UEMAKit/Tests/UEMASensesTests/Fixtures/audio/export_audio_fixture.py",
        "source": "Source/Python/mimo_bridge/audio_adapter.py CribAudio.render, dt 0.01 s, synthesis flutter shape",
        "frequencies_hz": audio.freqs.tolist(),
        "gaussian_draws": rec.draws,
        "steps": steps,
    }
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    print(out, len(steps), "steps,", len(rec.draws), "draws")


if __name__ == "__main__":
    main()
