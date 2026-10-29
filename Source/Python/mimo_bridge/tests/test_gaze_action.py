# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Gaze as part of the action, and the saccades that feed it data.

The load-bearing pair: the projection must start as EXACTLY nothing (old brains carry
over untouched), and the executed gaze must reach a trained pathway (otherwise the whole
mechanism is decoration). Plus the lesson from the reflex review: never test the trigger
without testing that the eyes actually moved.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mimo_bridge.gaze_reflex import SpontaneousSaccades, AZ_RANGE, EL_RANGE


# ------------------------------------------------------------------ the saccade generator
def test_jump_and_hold_not_noise():
    """A saccade lands and HOLDS - per-step noise would smear every frame."""
    s = SpontaneousSaccades(dt=0.01, seed=3)
    outs = [s.apply(np.zeros(2), f) for f in range(200)]
    diffs = [float(np.linalg.norm(a - b)) for a, b in zip(outs[1:], outs[:-1])]
    moves = sum(1 for d in diffs if d > 1e-9)
    assert 2 <= s.n_saccades <= 8, f"{s.n_saccades} saccades in 2 s; infants make ~2/s"
    assert moves == s.n_saccades - 1, "offset changed between saccades - that is noise, not fixation"


def test_saccades_pause_while_the_reflex_steers():
    s = SpontaneousSaccades(dt=0.01, seed=3)
    s.apply(np.zeros(2), 0)
    n = s.n_saccades
    for f in range(1, 400):
        s.apply(np.zeros(2), f, reflex_active=True)
    assert s.n_saccades == n, "resampled mid-snap; the reflex must win while it steers"


def test_saccade_output_respects_the_real_eye_ranges():
    s = SpontaneousSaccades(dt=0.01, seed=1, median_amp_deg=40.0, max_amp_deg=90.0)
    for f in range(400):
        g = s.apply(np.array([0.7, -0.7]), f)
        assert AZ_RANGE[0] <= g[0] <= AZ_RANGE[1]
        assert EL_RANGE[0] <= g[1] <= EL_RANGE[1]


def test_saccades_recentre_on_the_commanded_gaze():
    """Additive: purposeful gaze shifts where the eyes wander, it is not fought."""
    s = SpontaneousSaccades(dt=0.01, seed=5)
    a = np.mean([s.apply(np.array([0.4, 0.0]), f)[0] for f in range(600)])
    s2 = SpontaneousSaccades(dt=0.01, seed=5)
    b = np.mean([s2.apply(np.array([-0.4, 0.0]), f)[0] for f in range(600)])
    assert a - b > 0.5, "the wander must be centred on the command"


# ------------------------------------------------------------------ gaze in the action
def _tiny_model():
    import mimo_bridge.lem_model as LM
    from lem.model.forward_model import ForwardModel
    from lem.model.encoder_registry import EncoderRegistry
    from lem.model.prediction_heads import PredictionHeads
    cfg = LM.build_config()
    return (ForwardModel(cfg), EncoderRegistry(cfg), PredictionHeads(cfg))






def test_reset_revives_the_saccade_generator():
    """env.reset() rewinds the frame counter to 0. A next-switch left in the old clock's
    future held ONE offset for the whole next episode - measured, 11 saccades in the 600
    frames before a reset and 1 after. reset() must bring the rate back."""
    s = SpontaneousSaccades(dt=0.01, seed=3)
    for f in range(600):
        s.apply(np.zeros(2), f)
    first = s.n_saccades
    s.reset()
    for f in range(600):
        s.apply(np.zeros(2), f)
    assert s.n_saccades - first >= first // 2, "generator did not survive the reset"


def test_reset_revives_the_reflex():
    from mimo_bridge.gaze_reflex import GazeReflex
    rf = GazeReflex(size=256, fovy_deg=60.0, dt=0.01)
    a = np.zeros((256, 256, 3), np.uint8)
    b = a.copy(); b[40:56, 40:56] = 200
    eyes = ((0.0, 0.0), (0.0, 0.0))
    rf.observe(a, eyes, 0.0, frame=5000)
    rf.observe(b, eyes, 0.0, frame=5005)          # snap late in an episode
    assert rf.n_snaps == 1
    rf.reset()                                     # episode reset: frame goes back to 0
    assert not rf._hab, "habituation must clear at reset - its decay is frame-delta based"
    rf.observe(a, eyes, 0.0, frame=0)
    rf.observe(b, eyes, 0.0, frame=5)
    assert rf.n_snaps == 2, "refractory/stale-pair state survived the reset and killed it"


def test_different_seeds_wander_differently():
    """All 16 bodies shared one RNG stream (the trainer always passed a seed, so the
    per-worker setdefault never fired) - sixteen bodies saccading in lockstep."""
    a = SpontaneousSaccades(dt=0.01, seed=101)
    b = SpontaneousSaccades(dt=0.01, seed=102)
    ta = [a.apply(np.zeros(2), f)[0] for f in range(300)]
    tb = [b.apply(np.zeros(2), f)[0] for f in range(300)]
    assert not np.allclose(ta, tb), "two bodies produced the identical saccade schedule"


def test_saccade_amplitudes_are_skewed_small_like_a_human():
    """Gaussian sigma 12 gave a median 2D jump of 22 deg - five times a human's 4-5 - and
    oversized jumps cost fixation time on a slow eye. Lognormal gives the human skew."""
    s = SpontaneousSaccades(dt=0.01, seed=7)
    amps = []
    prev = s.apply(np.zeros(2), 0)
    for f in range(1, 6000):
        cur = s.apply(np.zeros(2), f)
        if not np.allclose(cur, prev):
            amps.append(np.degrees(np.linalg.norm(cur - prev)))
        prev = cur
    amps = np.array(amps)
    med, p90 = np.median(amps), np.percentile(amps, 90)
    assert 3.0 <= med <= 9.0, f"median jump {med:.1f} deg; human is 4-5"
    assert p90 > 1.5 * med, f"no long tail: median {med:.1f}, p90 {p90:.1f}"


# ---------------------------------------------------- the eyes as a real body part
def test_gaze_is_handle_23_everywhere():
    from lem.model.body_sets import HANDLE_NAMES, HANDLE_BODY_SET, BodySet
    from mimo_bridge.proprio_adapter import BONE_MAP
    from mimo_bridge.action_decoder import HANDLE_CHAINS
    assert HANDLE_NAMES[23] == "gaze"
    assert HANDLE_BODY_SET[23] == BodySet.Head
    assert BONE_MAP["gaze"] == "left_eye"          # felt in proprioception
    assert HANDLE_CHAINS["gaze"] == ()             # eye servo drives the joints, not the limb PD


def test_gaze_extraction_from_the_handle_quaternion():
    """Handle 23's orientation IS the gaze command: yaw/pitch out, clamped to the eyes."""
    import torch
    from mimo_bridge.lem_model import LemLite
    ex = LemLite._gaze_from_handle
    class Stub: GAZE_IDX = 23
    m = torch.zeros(2, 24, 9); m[..., 6] = 1.0                     # identity everywhere
    g = ex(Stub(), m)
    assert np.allclose(g, 0.0, atol=1e-6), "identity orientation must mean looking straight"
    yaw = np.radians(20.0)
    m[0, 23, 3:7] = torch.tensor([0.0, 0.0, np.sin(yaw/2), np.cos(yaw/2)])
    g = ex(Stub(), m)
    assert abs(g[0, 0] - yaw) < 1e-4 and abs(g[1, 0]) < 1e-6
    m[0, 23, 3:7] = torch.tensor([0.0, np.sin(np.radians(80)/2), 0.0, np.cos(np.radians(80)/2)])
    g = ex(Stub(), m)
    assert abs(g[0, 1] - 0.576) < 1e-3, "pitch must clamp to the real +33 deg eye limit"


def test_grown_checkpoint_is_bit_identical_for_the_old_body():
    """tools/grow_handles pads with zeros, so the 23 limb outputs cannot change."""
    import torch
    from mimo_bridge.tools.grow_handles import grow
    w = {
        "encoder_registry.encoders.proprioception.mlp.0.weight": torch.randn(128, 276),
        "motor_controller.input_proj.weight": torch.randn(128, 1205),
        "motor_controller.output_proj.weight": torch.randn(207, 128),
        "motor_controller.output_proj.bias": torch.randn(207),
        "wellness.grounded.motor_state.net.0.weight": torch.randn(64, 207),
    }
    keep = {k: v.clone() for k, v in w.items()}
    grow(w)
    x = torch.randn(3, 1205)
    xp = torch.zeros(3, 1224); xp[:, :998] = x[:, :998]; xp[:, 1008:1215] = x[:, 998:]
    a = x @ keep["motor_controller.input_proj.weight"].T
    b = xp @ w["motor_controller.input_proj.weight"].T
    assert torch.allclose(a, b, atol=1e-5)
    h = torch.randn(3, 128)
    oa = h @ keep["motor_controller.output_proj.weight"].T + keep["motor_controller.output_proj.bias"]
    ob = h @ w["motor_controller.output_proj.weight"].T + w["motor_controller.output_proj.bias"]
    assert torch.equal(oa, ob[:, :207])
    assert torch.equal(ob[:, 207:].reshape(3, 9)[:, 3:7],
                       torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(3, 4)),         "the gaze handle must be born looking straight (identity quat bias)"
