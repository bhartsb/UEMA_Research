# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Crib hearing (lane C): synthesis physics, the byte contract, and the live env path.

The synthesizer must be louder near than far, put the squeak in its band, flip the
arrival-time sign with the source's side, and decay to silence. The frame bytes must
round-trip through the avatar's own tokenizer. A live env with audio on must emit
frames whose audio slot carries the buzz.
"""
import numpy as np
import torch

from mimo_bridge.audio_adapter import CribAudio, N_BINS


HEAD = np.array([0.0, 0.0, 0.3])
RIGHT = np.array([0.0, 1.0, 0.0])
FWD = np.array([1.0, 0.0, 0.0])


def test_impact_louder_near_than_far():
    a, b = CribAudio(0.01), CribAudio(0.01)
    a.impact(HEAD + np.array([0.15, 0, 0]), 2.0)
    b.impact(HEAD + np.array([0.60, 0, 0]), 2.0)
    sa, _ = a.render(HEAD, RIGHT, FWD)
    sb, _ = b.render(HEAD, RIGHT, FWD)
    assert sa.mean() > sb.mean()


def test_squeak_is_a_reedy_glide_and_decays():
    a = CribAudio(0.01)
    a.squeak(HEAD + np.array([0.2, 0, 0]))
    freqs = np.geomspace(40.0, 8000.0, N_BINS)
    fund = (freqs > 450) & (freqs < 1400)     # the toy-reed fundamental's glide range
    low = freqs < 250
    centers = []
    for _ in range(24):                       # the first 240 ms of the squeak
        s, _ = a.render(HEAD, RIGHT, FWD)
        assert s[:, 0][fund].mean() > s[:, 0][low].mean()
        centers.append(float(freqs[np.argmax(s[:, 0])]))
    assert max(centers) > min(centers) * 1.5, "the pitch must BEND, not sit still"
    for _ in range(120):                      # 1.2 s >> the 280 ms decay
        s, _ = a.render(HEAD, RIGHT, FWD)
    assert s.max() < 0.05, "transients must die out"


def test_itd_sign_flips_with_side():
    a = CribAudio(0.01)
    _, m_right = a.render(HEAD, RIGHT, FWD, buzz_pos=HEAD + np.array([0.0, 0.4, 0.0]))
    b = CribAudio(0.01)
    _, m_left = b.render(HEAD, RIGHT, FWD, buzz_pos=HEAD + np.array([0.0, -0.4, 0.0]))
    assert m_right["itd_ms"] * m_left["itd_ms"] < 0, "opposite sides, opposite arrival lead"
    assert m_right["ild_db"] * m_left["ild_db"] < 0, "and opposite loudness difference"


def test_bytes_round_trip_through_the_avatar_tokenizer():
    from mimo_bridge import frame_encoder as FE
    a = CribAudio(0.01)
    a.squeak(HEAD + np.array([0.2, 0.1, 0.0]))
    spec, meta = a.render(HEAD, RIGHT, FWD, buzz_pos=HEAD + np.array([0.0, 0.3, 0.0]))
    buf = FE.new_frame_buffer()
    FE.write_audio(buf, spec, meta)
    from tokenizer.config import TokenizerConfig
    from tokenizer.modalities.audio import tokenize_audio
    s, md = tokenize_audio(bytes(buf), TokenizerConfig(device="cpu"))
    assert s.shape == (N_BINS, 2)
    assert torch.allclose(s, torch.from_numpy(spec), atol=1e-6)


def _track(a, n, **render_kw):
    from mimo_bridge.audio_adapter import render_track
    frames = []
    for _ in range(n):
        a.render(HEAD, RIGHT, FWD, **render_kw)
        frames.append(a.last_sources)
    return render_track(frames, 0.01)


def test_track_buzz_is_a_tone_not_noise():
    a = CribAudio(0.01)
    wav = _track(a, 100, buzz_pos=HEAD + np.array([0.3, 0.0, 0.05]))
    assert wav.shape == (100 * 160, 2) and int(np.abs(wav).max()) > 1000
    spec = np.abs(np.fft.rfft(wav[:, 0].astype(np.float64)))
    fr = np.fft.rfftfreq(wav.shape[0], 1 / 16000)
    peak_f = float(fr[np.argmax(spec)])
    assert 55 < peak_f < 95, \
        f"a DEEP bumblebee beats near 70 Hz (operator-tuned 2026-09-06), got {peak_f:.0f}"
    assert float(spec.max()) > 25 * float(np.median(spec[fr > 100])), \
        "a TONE towers over its floor; noise does not"


def test_track_is_lateralized():
    a = CribAudio(0.01)
    wav = _track(a, 60, buzz_pos=HEAD + np.array([0.0, 0.35, 0.05]))   # +y = right
    assert float(np.abs(wav[:, 1]).mean()) > float(np.abs(wav[:, 0]).mean())


def test_track_squeak_glides():
    a = CribAudio(0.01)
    a.squeak(HEAD + np.array([0.2, 0.0, 0.0]))
    wav = _track(a, 30).astype(np.float64)
    def dom(seg):
        sp = np.abs(np.fft.rfft(seg[:, 0]))
        return float(np.fft.rfftfreq(seg.shape[0], 1 / 16000)[np.argmax(sp)])
    mid = dom(wav[1600:3200])             # 100-200 ms: the glide's crest (~1200 Hz)
    late = dom(wav[3200:4800])            # 200-300 ms: the falling tail (~500 Hz)
    assert mid > late * 1.5, f"the squeak must bend ({mid:.0f} Hz crest -> {late:.0f} Hz tail)"


def test_every_source_is_audible_in_the_mix():
    """The burying test: one second with the bee at 35 cm, a squeak at 55 cm and a
    close thump at 15 cm together. Each must remain audible in the final track -
    peak-normalizing to the nearest thump once crushed the bee to 2% of full scale."""
    from mimo_bridge.audio_adapter import render_track
    a = CribAudio(0.01)
    frames = []
    for i in range(100):
        if i == 40:
            a.squeak(HEAD + np.array([-0.5, 0.0, -0.15]))
        if i == 70:
            a.impact(HEAD + np.array([0.10, 0.08, 0.0]), 0.4)
        a.render(HEAD, RIGHT, FWD, buzz_pos=HEAD + np.array([0.2, 0.25, 0.1]))
        frames.append(a.last_sources)
    wav = render_track(frames, 0.01).astype(np.float64) / 32767.0
    buzz_rms = float(np.sqrt((wav[0:6000] ** 2).mean()))          # bee alone
    squeak_rms = float(np.sqrt((wav[6600:8000] ** 2).mean()))     # squeak life
    thump_rms = float(np.sqrt((wav[11200:12800] ** 2).mean()))    # thump life
    assert buzz_rms > 0.02, f"the bee must be audible alone ({buzz_rms:.3f})"
    assert squeak_rms > buzz_rms * 1.02, "the squeak must add audibly over the buzz"
    # operator 2026-09-06: body contacts sit BELOW the bee, not just "not burying" it.
    # A close hard thump's window must stay within 1.5x the bee-only window (old full-gain
    # x1.5 thump was ~3x the bee and would fail this).
    assert thump_rms < buzz_rms * 1.5, \
        f"a close thump must stay under the bee ({thump_rms:.3f} vs buzz {buzz_rms:.3f})"


def test_real_recordings_load_and_sound():
    """The sample engine: the public-domain bumblebee recording and its fast-winged
    derivative must load, shape the model's spectrum, and drive the track."""
    from mimo_bridge.audio_adapter import render_track
    a = CribAudio(0.01)
    assert a.fly_wave is not None, "dragonfly sample loop missing from assets/audio"
    assert a._flutter_shape_sample is not None
    # the bee stays on its operator-ACCEPTED deep synthesis - the sample is loaded
    # but never voices the bee (adversarial finding F2: accepted voices are immovable)
    assert a._buzz_shape_sample is None
    assert a.sample_waves()["buzz"] is None
    frames = []
    for _ in range(60):
        a.render(HEAD, RIGHT, FWD, buzz_pos=HEAD + np.array([0.3, 0.0, 0.05]),
                 flutter_pos=HEAD + np.array([0.0, -0.3, 0.1]))
        frames.append(a.last_sources)
    wav = render_track(frames, 0.01, samples=a.sample_waves())
    assert int(np.abs(wav).max()) > 1000, "the recorded voices must be audible"


def test_resting_body_does_not_machine_gun(tmp_path):
    """The conviction test for the 355-thumps-per-second defect: a resting body's
    persistent mattress contacts must make essentially NO impact sounds."""
    from mimo_bridge.env import BridgeEnv
    e = BridgeEnv(scene="crib", run_dir=str(tmp_path), vision=True, eye_size=256,
                  fovea_size=288, fovea_fov=24.0, render_every=2, seed=33,
                  tag="_gun", mobile=True, critters=True, audio_real=True)
    try:
        e.reset()
        thumps = 0
        for i in range(300):                     # 3 seconds at rest
            e.step(None, None, None, ctrl=np.zeros(e.model.nu), render=(i % 2 == 0))
            src = getattr(e._audio, "last_sources", None) or {}
            thumps += sum(1 for tr in src.get("transients", ())
                          if tr[0] == "impact" and tr[4] < 0.011)   # newborn this step
        assert thumps <= 6, f"{thumps} impact onsets in 3 s of rest - still a machine gun"
    finally:
        e.close()


def test_live_env_emits_hearing(tmp_path):
    from mimo_bridge.env import BridgeEnv
    from mimo_bridge import frame_encoder as FE
    from tokenizer.config import MOD_AUDIO
    e = BridgeEnv(scene="crib", run_dir=str(tmp_path), vision=True, eye_size=256,
                  fovea_size=288, fovea_fov=24.0, render_every=2, seed=5,
                  tag="_aud", mobile=True, critters=True, audio_real=True)
    try:
        e.reset()
        heard = False
        for i in range(30):
            r = e.step(None, None, None, ctrl=np.zeros(e.model.nu), render=(i % 2 == 0))
            if r.info.get("audio_spec") is not None:
                tf = FE.decode_with_model_tokenizer(bytes(FE.encode_frame(
                    frame_number=r.frame, timestamp=r.sim_time, dt=r.dt, wall_clock=0.0,
                    gaze_az_deg=0.0, gaze_el_deg=0.0, bones=r.proprio.bones,
                    bone_hashes=r.proprio.hashes,
                    handles=np.zeros((24, 10), dtype=np.float32),
                    energy=r.proprio.energy, body_sets=r.proprio.body_sets,
                    patches=r.patches, collisions=r.collisions,
                    audio_spectral=r.info["audio_spec"], audio_meta=r.info["audio_meta"])))
                assert tf.metadata.modality_valid_mask & MOD_AUDIO
                assert tf.audio_spectral is not None and tf.audio_spectral.shape == (N_BINS, 2)
                assert float(tf.audio_spectral.max()) > 0.0, "the bee's buzz must be audible"
                heard = True
                break
        assert heard, "audio_real env must emit hearing every frame"
        # the dead-wire guard, visitor-aware: the PRESENT critter's voice must reach
        # the ear (the flutter was once built, tuned three times, and never passed
        # into render() - silent in every demo while unit tests passed, 2026-09-06)
        src = getattr(e._audio, "last_sources", None) or {}
        assert src.get("buzz") is not None, "the visiting bee's buzz must be wired into the env's ear"
        assert src.get("flutter") is None, "the absent dragonfly must be silent"
        cr = e._cr
        e._critter_show("bee", False)
        e._critter_show("dragonfly", True)
        cr["visitor"] = "dragonfly"
        cr["visitor_until"] = e.frame + 10**9
        cr["df_prev"] = None
        for i in range(6):
            e.step(None, None, None, ctrl=np.zeros(e.model.nu), render=(i % 2 == 0))
        src = getattr(e._audio, "last_sources", None) or {}
        assert src.get("flutter") is not None, "the visiting dragonfly's flutter must be wired in"
        assert src.get("buzz") is None, "the absent bee must be silent"
    finally:
        e.close()


def test_mobile_chime_reaches_the_ear_and_the_track():
    """The mobile's strike voice (2026-09-06): a batted element must SOUND - dead-wire
    guard like every other voice. Physics caps its impulse 30x under the impact gate,
    so it gets its own transient, not the thump path."""
    from mimo_bridge.audio_adapter import render_track
    a = CribAudio(0.01)
    quiet, _ = a.render(HEAD, RIGHT, FWD)
    a.chime(HEAD + np.array([0.0, 0.25, 0.20]))
    spec, _ = a.render(HEAD, RIGHT, FWD)
    kinds = [t[0] for t in (a.last_sources or {}).get("transients", ())]
    assert "chime" in kinds, "the chime transient must reach the source record"
    band = (a.freqs > 700) & (a.freqs < 2400)
    assert float(spec[band].mean()) > float(quiet[band].mean()) + 1e-4, \
        "a struck mobile must add energy in its own band"
    frames = [a.last_sources]
    for _ in range(30):
        a.render(HEAD, RIGHT, FWD)
        frames.append(a.last_sources)
    wav = render_track(frames, 0.01).astype(np.float64) / 32767.0
    assert float(np.abs(wav).max()) > 0.005, "the chime must ring in the soundtrack"
