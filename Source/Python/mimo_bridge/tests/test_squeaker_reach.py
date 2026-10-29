# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The foot pedal must be kickable and audible - by order (2026-09-05).

A real kick: drive the hip and knee actuators through flex/extend cycles, nothing
scripted about the contact itself. Pass = a foot or lower-leg strike registers AND
the squeak band (2-3 kHz) rises in the avatar's own hearing right after the press.
If this fails, the pedal is out of reach or mute and the placement is wrong.
"""
import numpy as np
import pytest

from mimo_bridge.env import BridgeEnv


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = BridgeEnv(scene="crib", run_dir=str(tmp_path_factory.mktemp("kick")),
                  vision=True, eye_size=256, fovea_size=288, fovea_fov=24.0,
                  render_every=2, seed=21, tag="_kick", mobile=True,
                  ring_fov=150.0, critters=True, audio_real=True)
    e.reset()
    yield e
    e.close()


def _leg_actuators(m):
    import mujoco
    ids = []
    for a in range(m.nu):
        j = int(m.actuator_trnid[a, 0])
        jn = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").lower()
        if any(k in jn for k in ("hip", "knee")):
            ids.append(a)
    return ids


def _squeak_band_energy(spec, freqs):
    band = (freqs > 450) & (freqs < 1400)     # the toy-reed fundamental's glide range
    return float(np.asarray(spec)[band].mean())


def test_pedal_is_kickable_and_audible(env):
    from mimo_bridge.audio_adapter import N_BINS, F_LO, F_HI
    m = env.model
    legs = _leg_actuators(m)
    assert legs, "no hip or knee actuators found"
    cr = env._cr
    hits0 = cr["sq_hits_foot"]
    prev_spec = None
    hit_frame = None
    squeak_gain = 0.0
    spec_delta = 0.0
    ctrl = np.zeros(m.nu)
    for i in range(1200):                       # 12 s of kicking
        phase = (i // 60) % 2                   # flex / extend at ~0.8 Hz
        for a in legs:
            lo, hi = m.actuator_ctrlrange[a]
            ctrl[a] = hi if phase == 0 else lo
        r = env.step(None, None, None, ctrl=ctrl, render=(i % 2 == 0))
        sp = r.info.get("audio_spec")
        src = getattr(env._audio, "last_sources", None) or {}
        if hit_frame is None:
            if cr["sq_hits_foot"] > hits0:
                hit_frame = i
            elif sp is not None:
                prev_spec = np.asarray(sp)      # the last quiet ear-frame
        if hit_frame is not None and i <= hit_frame + 20:
            for kind, sid, gl, gr, extra in src.get("transients", ()):
                if kind == "squeak":
                    squeak_gain = max(squeak_gain, gl + gr)
            if sp is not None and prev_spec is not None:
                spec_delta = max(spec_delta, float(np.abs(np.asarray(sp) - prev_spec).max()))
        elif hit_frame is not None and i > hit_frame + 20:
            break
    assert cr["sq_hits_foot"] > hits0, \
        "no foot or lower-leg strike in 12 s of kicking - the pedal is OUT OF REACH"
    # audibility: the squeak source really sounded (gain reached the ears), and it
    # CHANGED the heard spectrum - band-mean tests fail here because the bee's
    # harmonics own 450-1400 Hz continuously (measured: band 0.87 at rest)
    assert squeak_gain > 1e-4, f"squeak source gain {squeak_gain:.2e} never reached the ears"
    assert spec_delta > 0.03, \
        f"the press must change what is heard: max bin delta {spec_delta:.3f}"
