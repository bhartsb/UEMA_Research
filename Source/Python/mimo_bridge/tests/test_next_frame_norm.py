# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The next-moment lesson measured against "nothing changes" (DESIGN.md D12, --next-frame-norm).

What is pinned here: the weights (each sense's weight is the mean running persistence error over
its own, so the pressure on every sense is its error relative to "nothing changes"), the size
(when every sense sits at persistence the normalised lesson equals the plain sum), the measured
cap and the numerical guard, the running average (corrected for starting at zero, the divisor
taken from before the step's own reading, non-finite readings refused), the window record, the
round trip through runtime_state.pt, the live lesson and replay's re-study wired to the same
weights, the vestibular sense covered by name, and nothing at all with the flag off."""
import inspect
import math

import pytest
import torch

from lem.model import next_frame_norm as NFN
from lem.model.next_frame_norm import NextFrameNorm
from mimo_bridge.lem_model import LemLite


# ---------------------------------------------------------------- the weights
def test_each_weight_is_the_mean_divisor_over_the_senses_own():
    n = NextFrameNorm()
    P = {"audio": 0.0839, "tactile": 0.0248, "vision": 0.0090, "proprioception": 0.00393}
    w, info = n.weights(list(P), P)
    p_mean = sum(P.values()) / 4
    for s, p in P.items():
        assert w[s] == pytest.approx(p_mean / p)
    assert info["p_mean"] == pytest.approx(p_mean) and not info["capped"] and not info["unscaled"]
    # segment 31's numbers: audio loses weight, proprioception gains it
    assert w["audio"] < 1.0 < w["tactile"] < w["vision"] < w["proprioception"]


def test_at_persistence_the_normalised_lesson_equals_the_plain_sum():
    """The lesson keeps its size: every sense's error equal to its divisor gives the plain sum."""
    n = NextFrameNorm()
    P = {"audio": 0.08, "tactile": 0.02, "vision": 0.009, "proprioception": 0.004, "vestibular": 0.04}
    w, _ = n.weights(list(P), P)
    assert sum(w[s] * P[s] for s in P) == pytest.approx(sum(P.values()))


def test_the_gradient_on_each_sense_is_its_error_relative_to_persistence():
    """d lesson / d error_s = p_mean / P_s: a sense twice as far from 'nothing changes' in its own
    units pulls exactly twice as hard, whatever the size of its token's change."""
    n = NextFrameNorm()
    P = {"a": 0.1, "b": 0.01}          # a 10-fold difference, inside the measured cap
    w, info = n.weights(list(P), P)
    assert not info["capped"]
    e = {s: torch.tensor(P[s] * 0.5, requires_grad=True) for s in P}   # each at half its persistence
    lesson = sum(e[s] * w[s] for s in P)
    lesson.backward()
    # pressure per unit of RELATIVE error is equal: grad_s x P_s is the same for both senses
    assert float(e["a"].grad) * P["a"] == pytest.approx(float(e["b"].grad) * P["b"])
    # and each sense's weighted error is the same when their ratios are the same
    assert float(e["a"] * w["a"]) == pytest.approx(float(e["b"] * w["b"]))


def test_a_quiet_sense_is_capped_at_the_measured_weight():
    n = NextFrameNorm()
    P = {"loud": 0.1, "quiet": 1e-9}
    w, info = n.weights(list(P), P)
    p_mean = (0.1 + 1e-9) / 2
    assert w["quiet"] == pytest.approx(NFN.WEIGHT_CAP)
    assert info["capped"] == ["quiet"] and info["floor"] == pytest.approx(p_mean / NFN.WEIGHT_CAP)
    assert w["loud"] == pytest.approx(p_mean / 0.1)


def test_the_cap_is_the_measured_number_and_the_guard_is_far_below_any_lived_value():
    assert NFN.WEIGHT_CAP == 30.7          # seg_horizon.py: the largest single-window weight lived
    assert NFN.HORIZON_STEPS == 2000
    # the quietest window any sense lived (proprioception, 1.08e-3) is nine orders above the guard
    assert NFN.ZERO_GUARD * 1e8 < 1.08e-3
    with pytest.raises(ValueError):
        NextFrameNorm(cap=None)
    with pytest.raises(ValueError):
        NextFrameNorm(cap=1.0)


def test_when_nothing_changes_anywhere_the_lesson_is_the_plain_sum():
    n = NextFrameNorm()
    w, info = n.weights(["a", "b"], {"a": 0.0, "b": 0.0})
    assert w == {"a": 1.0, "b": 1.0} and info["guard"]


def test_a_sense_with_no_divisor_is_left_out_and_counted():
    """The vestibular sense's first graded moment after the resume: its input was the padded
    legacy vector, so there is no reading and no running average - weight 0, counted."""
    n = NextFrameNorm()
    n.update({"tactile": 0.02})
    w, info = n.weights(["tactile", "vestibular"], {})
    assert w["vestibular"] == 0.0 and info["unscaled"] == ["vestibular"]
    assert w["tactile"] == pytest.approx(1.0), "the mean of one divisor over itself"
    w, info = n.weights(["x"], {})
    assert w == {"x": 0.0} and info["p_mean"] is None


def test_a_sense_predicted_from_a_stand_in_carries_no_weight_and_leaves_the_mean():
    n = NextFrameNorm()
    n.update({"tactile": 0.02, "vestibular": 0.9})
    w, info = n.weights(["tactile", "vestibular"], {}, input_not_real=["vestibular"])
    assert w == {"tactile": pytest.approx(1.0), "vestibular": 0.0}
    assert info["input_not_real"] == ["vestibular"] and info["p_mean"] == pytest.approx(0.02)
    n.add_live({"tactile": 0.02, "vestibular": 0.95}, w, info)
    s = n.summary()
    assert s["input_not_real_steps"] == {"vestibular": 1} and "vestibular" not in s["ratio"]


def test_the_divisor_is_the_average_from_before_the_steps_own_reading():
    n = NextFrameNorm(horizon=10)
    n.update({"a": 1.0})
    w, info = n.weights(["a", "b"], {"a": 100.0, "b": 2.0})
    # "a" has an average (1.0) so its reading this step is not used; "b" has none, so its reading is
    assert info["divisor"] == {"a": 1.0, "b": 2.0}


@pytest.mark.parametrize("w", [0.3, 1.0, 12.0, NFN.WEIGHT_CAP])
def test_the_weight_reaches_the_head_and_the_encoder_keeps_the_plain_gradient(w):
    """THE ENCODER'S SHARE: with residual prediction the prediction is head + current token. The
    weight must reach the head (and the core behind it) but not the path into the encoder that
    made the current token, or a quiet sense's encoder is pushed w times harder to stop its token
    changing (measured on ckpt_step_344400: 5 to 25 times quieter proprioception token in 60
    steps at the same body motion). The value is the weighted error, unchanged."""
    torch.manual_seed(0)
    h = torch.randn(4, 8, requires_grad=True)          # the head's output
    c = torch.randn(4, 8, requires_grad=True)          # the current token (encoder output)
    tgt = torch.randn(4, 8)
    pred = h + c                                       # what PredictionHeads returns
    val = NFN.weighted_term(pred, c, tgt, w, residual=True)
    val.backward()
    plain = 2.0 * ((h + c).detach() - tgt) / tgt.numel()          # d mse / d pred
    assert float(val) == pytest.approx(w * float(((h + c) - tgt).pow(2).mean()), rel=1e-6)
    assert torch.allclose(h.grad, w * plain, atol=1e-7), "the head carries the weight"
    assert torch.allclose(c.grad, plain, atol=1e-7), "the encoder's path keeps the plain gradient"
    # without residual prediction there is no such path: everything carries the weight
    h2 = h.detach().clone().requires_grad_(True)
    NFN.weighted_term(h2, None, tgt, w, residual=False).backward()
    assert torch.allclose(h2.grad, w * 2.0 * (h2.detach() - tgt) / tgt.numel(), atol=1e-7)
    # a weight of 0 contributes nothing and no gradient at all
    h3 = h.detach().clone().requires_grad_(True)
    c3 = c.detach().clone().requires_grad_(True)
    z = NFN.weighted_term(h3 + c3, c3, tgt, 0.0, residual=True)
    z.backward()
    assert float(z) == 0.0 and float(h3.grad.abs().sum()) == 0.0 and float(c3.grad.abs().sum()) == 0.0


# ---------------------------------------------------------------- the running average
def test_the_running_average_is_corrected_for_starting_at_zero():
    n = NextFrameNorm(horizon=4)
    a = 0.25
    n.update({"s": 2.0})
    assert n.scale("s") == pytest.approx(2.0), "one reading: the average is that reading"
    n.update({"s": 6.0})
    expected = ((1 - a) * a * 2.0 + a * 6.0) / ((1 - a) * a + a)
    assert n.scale("s") == pytest.approx(expected)
    assert n.stats["s"][2] == 2


def test_non_finite_or_negative_readings_never_enter():
    n = NextFrameNorm()
    n.update({"s": float("nan"), "t": float("inf"), "u": -1.0, "v": None})
    assert n.scale("s") is None and n.scale("t") is None and n.scale("u") is None
    assert not n.stats


def test_runtime_state_round_trip_and_junk_is_dropped():
    n = NextFrameNorm()
    n.update({"audio": 0.08, "vestibular": 0.04})
    n.update({"audio": 0.09})
    sd = n.state_dict()
    import io
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    m = NextFrameNorm()
    assert m.load_state_dict(torch.load(buf, weights_only=False)) == ["audio", "vestibular"]
    for s in ("audio", "vestibular"):
        assert m.scale(s) == n.scale(s)
    assert m.load_state_dict({"stats": {"a": [float("nan"), 1.0, 1], "b": [1.0, 0.0, 1],
                                        "c": "junk", "d": [0.5, 0.5, 3]}}) == ["d"]
    assert m.load_state_dict(None) == [] and not m.stats


# ---------------------------------------------------------------- the window record
def test_the_window_record_reports_both_shares_and_resets():
    n = NextFrameNorm()
    n.update({"audio": 0.08, "proprioception": 0.004})
    raw = {"audio": 0.07, "proprioception": 0.004}
    w, info = n.weights(list(raw), {})
    n.add_live(raw, w, info)
    n.add_replay({"audio": 0.06, "proprioception": 0.005}, w)
    s = n.summary()
    assert s["steps"] == 1 and s["replay_steps"] == 1
    assert s["share_plain"]["audio"] == pytest.approx(0.07 / 0.074, abs=1e-5)
    norm = {k: w[k] * raw[k] for k in raw}
    assert s["share_normalized"]["proprioception"] == pytest.approx(
        norm["proprioception"] / sum(norm.values()), abs=1e-5)
    assert s["weight_effective"]["proprioception"] == pytest.approx(w["proprioception"], abs=1e-5)
    assert s["ratio"]["audio"] == pytest.approx(0.07 / 0.08, abs=1e-5)
    assert s["lesson_plain"] == pytest.approx(0.074, abs=1e-5)
    assert s["persistence_ema"] == {"audio": 0.08, "proprioception": 0.004}
    assert s["weight_cap"] == NFN.WEIGHT_CAP and s["horizon_steps"] == NFN.HORIZON_STEPS
    json_safe = __import__("json").dumps(s, allow_nan=False)
    assert "next_frame_norm_v1" in json_safe
    assert n.summary()["steps"] == 0, "the window starts fresh"


def test_the_ratio_uses_the_capped_divisor_and_an_unscaled_replay_pass_is_counted():
    n = NextFrameNorm()
    n.update({"loud": 0.1, "quiet": 1e-9})
    w, info = n.weights(["loud", "quiet"], {})
    n.add_live({"loud": 0.05, "quiet": 0.001}, w, info)
    n.add_replay({"loud": 0.04, "new": 0.2}, {"loud": w["loud"], "new": 0.0})
    s = n.summary()
    assert s["ratio"]["quiet"] == pytest.approx(0.001 / info["floor"], rel=1e-4)
    assert s["capped_steps"] == {"quiet": 1}
    assert s["replay_unscaled_steps"] == {"new": 1}
    assert s["persistence_ema"]["quiet"] == pytest.approx(1e-9), "significant digits, not decimals"


# ---------------------------------------------------------------- the model
def _lite(**kw):
    torch.set_num_threads(1)
    return LemLite(device="cpu", seed=0, imagination=False, no_vision=True,
                   next_frame_loss=True, body_layout="trunk_v1", **kw)


def test_flag_off_builds_nothing_and_flag_on_adds_no_parameter():
    off = _lite()
    on = _lite(next_frame_norm="persistence")
    assert off.nf_norm is None and "next_frame_norm" not in off.receipt
    assert off.next_frame_norm_window_summary() is None
    assert on.n_params == off.n_params and len(on.params) == len(off.params)
    st = on.receipt["next_frame_norm"]
    assert st["mode"] == "persistence" and st["weight_cap"] == NFN.WEIGHT_CAP and st["replay"] is True
    with pytest.raises(ValueError, match="next-frame-loss"):
        LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_norm="persistence")
    with pytest.raises(ValueError, match="next_frame_norm"):
        _lite(next_frame_norm="ratio")


def test_the_normaliser_is_public_so_the_episode_companion_never_copies_it():
    from mimo_bridge import continuation_state as CS
    m = _lite(next_frame_norm="persistence")
    m.nf_norm.update({"tactile": 0.02})
    data = CS.capture_learner(m, include_replay=False)["data"]
    assert "nf_norm" not in data and not any("nf_norm" in k for k in data)


def test_the_running_averages_ride_the_checkpoint(tmp_path):
    m = _lite(next_frame_norm="persistence", checkpoint_dir=str(tmp_path))
    m.nf_norm.update({"tactile": 0.02, "proprioception": 0.003})
    path = m.save_checkpoint(3)
    import os
    rt = torch.load(os.path.join(path, "runtime_state.pt"), weights_only=False)
    assert set(rt["next_frame_norm"]["stats"]) == {"tactile", "proprioception"}
    other = _lite(next_frame_norm="persistence")
    other.load_checkpoint(path)
    assert other.nf_norm.scale("tactile") == pytest.approx(0.02)
    assert other.nf_norm.scale("proprioception") == pytest.approx(0.003)
    # a build without the flag loads the same checkpoint and ignores the key
    bare = _lite()
    bare.load_checkpoint(path)
    assert bare.nf_norm is None


def test_a_checkpoint_saved_without_it_starts_cold(tmp_path):
    m = _lite(checkpoint_dir=str(tmp_path))
    path = m.save_checkpoint(4)
    on = _lite(next_frame_norm="persistence")
    on.load_checkpoint(path)
    assert on.nf_norm.stats == {}


def test_the_vestibular_sense_is_weighted_by_name():
    """review-b2 major 2: the per-sense scaling must cover the vestibular sense. The weights are
    keyed by the sense's name and the lesson iterates pred_mods, so a sixth real sense named
    'vestibular' is weighted like every other."""
    m = _lite(next_frame_norm="persistence", vestibular_sense=True)
    assert "vestibular" in m.pred_mods
    src = inspect.getsource(LemLite.step)
    assert "list(nf_terms), _nfr," in src
    # review-b2 minor 7: the moment predicted from the padded legacy vector carries no weight
    assert 'input_not_real=[m for m in nf_terms if self._nf_mask.get(m) != "real"]' in src
    assert "for m in self.pred_mods" in src.split("nf_terms = {")[1][:200]


def _drive(m, tmp_path, n, first=None):
    """n model steps of one body in the reach scene with babble; `first(frames)` may rewrite the
    first step's frames (the padded legacy vestibular vector of a resume)."""
    from mimo_bridge.datasource import SingleEnvSource
    src = SingleEnvSource(scene="reach", run_dir=str(tmp_path / "nf"), tag="_nfn",
                          vision=False, seed=5)
    try:
        frames = src.reset()
        if first is not None:
            first(frames)
        dev = dict(m.dev_limits(10 ** 9))
        dev["exploration_noise"] = 0.05
        held = prev = None
        out = []
        for i in range(n):
            before = None if m.nf_norm is None else {k: list(v) for k, v in m.nf_norm.stats.items()}
            res = m.step(frames, dev, train=True, step_index=i, executed_action=prev)
            out.append((res, before, dict(m.last_prediction_audit)))
            if i % 5 == 0 or held is None:
                held = res.motor_for_action
            for _ in range(5):
                frames = src.step(held, res.gaze_out, None, render=False)
            prev = held
        return out
    finally:
        src.close()


def test_the_live_lesson_is_the_weighted_sum_from_the_averages_before_the_step(tmp_path):
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False, next_frame_loss=True,
                next_frame_norm="persistence", overrides={"training": {"steps_per_phase": 3000}})
    runs = _drive(m, tmp_path, 8)
    graded = 0
    for res, before, audit in runs:
        if "next_frame" not in res.breakdown:
            continue
        graded += 1
        e = res.next_pe_per_modality
        ref = NextFrameNorm()
        ref.load_state_dict({"stats": before})
        readings = {s: a["persistence_mse"] for s, a in audit.items() if a and s in e}
        w, _ = ref.weights(list(e), readings)
        assert res.breakdown["next_frame"] == pytest.approx(sum(w[s] * e[s] for s in e), rel=1e-4)
        assert math.isfinite(res.loss)
    assert graded >= 6, "the delayed lesson ran on almost every step"
    blk = m.next_frame_norm_window_summary()
    assert blk["steps"] == graded and set(blk["weight_effective"]) == set(runs[-1][0].next_pe_per_modality)
    assert all(v > 0 for v in blk["persistence_ema"].values())


@pytest.mark.parametrize("lesson_weight", [1.0, 0.4, 2.0])
def test_the_vestibular_sense_is_weighted_in_the_live_lesson(tmp_path, lesson_weight):
    """review-b2 major 2, as behaviour: with --vestibular-sense on, the sixth real sense
    'vestibular' has its own running persistence error and its own weight in the delayed lesson,
    and the lesson is the weighted sum over all five real senses including it. The first frame
    carries the padded legacy vector (what the first frame after resuming ckpt_step_344400
    carries), so the transition predicted FROM it is excluded before grading (review-b2 minor 7).
    An explicit lesson weight must survive the persistence normalization too."""
    import numpy as np
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False, next_frame_loss=True,
                next_frame_norm="persistence", vestibular_sense=True,
                vestibular_lesson_weight=lesson_weight,
                overrides={"training": {"steps_per_phase": 3000}})
    assert "vestibular" in m.pred_mods

    def legacy(frames):
        # the old six numbers (accelerometer m/s2, gyro rad/s), padded by the reader with valid 0
        frames[0].vestibular = np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32)

    runs = _drive(m, tmp_path, 8, first=legacy)
    with_vest = 0
    for i, (res, before, audit) in enumerate(runs):
        if "next_frame" not in res.breakdown:
            continue
        e = res.next_pe_per_modality
        if i == 1:
            assert "vestibular" not in e, "a padded starting frame is not a real sensory transition"
        ref = NextFrameNorm()
        ref.load_state_dict({"stats": before})
        readings = {s: a["persistence_mse"] for s, a in audit.items() if a and s in e}
        # the moment predicted from the padded frame (step 1) is the one whose input was not real
        stand_in = ["vestibular"] if (i == 1 and "vestibular" in e) else []
        w, _ = ref.weights(list(e), readings, input_not_real=stand_in)
        expected = sum(w[s] * e[s] * (lesson_weight if s == "vestibular" else 1.0) for s in e)
        assert res.breakdown["next_frame"] == pytest.approx(expected, rel=1e-4)
        if "vestibular" in e and not stand_in:
            with_vest += 1
            assert w["vestibular"] > 0.0 and "vestibular" in readings
    assert with_vest >= 5, "the vestibular sense was graded and weighted on the later steps"
    blk = m.next_frame_norm_window_summary()
    assert blk["input_not_real_steps"] == {}, "invalid vestibular transitions never enter this lesson"
    assert "vestibular" in blk["weight_effective"] and blk["persistence_ema"]["vestibular"] > 0
    assert "vestibular" in blk["share_normalized"]


def test_with_the_flag_off_the_lesson_is_the_plain_sum(tmp_path):
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False, next_frame_loss=True,
                overrides={"training": {"steps_per_phase": 3000}})
    for res, _, _ in _drive(m, tmp_path, 4):
        if "next_frame" in res.breakdown:
            assert res.breakdown["next_frame"] == pytest.approx(
                sum(res.next_pe_per_modality.values()), rel=1e-5)
    assert m.nf_norm is None and m.next_frame_norm_window_summary() is None


def test_replay_carries_the_same_weights(tmp_path):
    """replay_loss_only restores the weights and the optimizer and seeds its draw, so the same
    windows are graded both ways: with equal divisors every weight is exactly 1 and the loss is
    the plain one; with unequal divisors it moves by sum_s (w_s - 1) x that sense's error."""
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False, next_frame_loss=True,
                next_frame_norm="persistence", overrides={"training": {"steps_per_phase": 3000}})
    m.enable_replay(1, size_per_env=64)
    _drive(m, tmp_path, 12)
    norm = m.nf_norm
    saved = {k: list(v) for k, v in norm.stats.items()}
    m.nf_norm = None
    plain = m.replay_loss_only(batch=4, seed=3)
    assert plain is not None
    m.nf_norm = norm
    norm.stats = {s: [0.01, 1.0, 1] for s in saved}           # every divisor equal: weights 1
    norm.summary()
    same = m.replay_loss_only(batch=4, seed=3)
    assert same == pytest.approx(plain, rel=1e-6)
    raw = dict(norm.win["replay_raw"])
    assert raw, "the re-study's per-sense errors were recorded"
    norm.stats = {s: [0.01 * (i + 1), 1.0, 1] for i, s in enumerate(sorted(saved))}
    w, _ = norm.weights(sorted(raw), None)
    moved = m.replay_loss_only(batch=4, seed=3)
    assert moved == pytest.approx(plain + sum((w[s] - 1.0) * raw[s] for s in raw), rel=1e-4)


# ---------------------------------------------------------------- the trainer
def test_the_trainer_flag_defaults_off_is_passed_through_and_writes_the_block():
    from mimo_bridge import train_mimo as TM
    a = TM.parse_args(["--out", "x"])
    assert a.next_frame_norm == "off"
    assert TM.parse_args(["--out", "x", "--next-frame-norm", "persistence"]).next_frame_norm == "persistence"
    src = inspect.getsource(TM)
    assert 'next_frame_norm=getattr(args, "next_frame_norm", "off") or "off"' in src
    assert '"next_frame_norm": model.next_frame_norm_window_summary()' in src
    cont = src.split("continuation_flags = (")[1].split(")")[0]
    assert "next_frame_norm" not in cont, "a resume of the old command must not be refused"
    assert "next_frame_norm" in TM.acting_configuration(a), "it changes what the model learns"
