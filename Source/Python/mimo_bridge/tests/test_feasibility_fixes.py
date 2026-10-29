# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Regression tests for the 2026-09-09 feasibility-review fixes.

Each test pins one of the reviewer's executable counterexamples in its FIXED
state, so a regression re-opens a named, measured defect:
  touch units      raw Pa scored every contact 0; normalized fields score it
  worst moment     endpoint gain with transient pain must NOT read as steering
  preservation     'goal pursuit unchanged' needs a declared margin + CI
  freeze dedupe    a module listed twice froze a trainable param forever
  revalue need     present capacity must be ABLE to change which goal wins
  audio frequency  conv+global-average erased WHERE a sound sits; band path
                   restores the information, zero-init keeps old ckpts exact
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lem.model.intention_generator import _freeze_params, _restore_params  # noqa: E402
from lem.model.wellness import (  # noqa: E402
    WellnessEvaluators, WellnessMode, MotorStateFormula)
from mimo_bridge.tools.first_action_benefit import (  # noqa: E402
    state_wellness, _wellness_verdict, _measurement_verdict)


def _evaluators():
    cfg = {"_active_modalities": ["vision", "tactile", "motor"],
           "model": {"d_model": 8,
                     "wellness": {"grounded": {"motor_state": {"input_dim": 216}},
                                  "approximate": {"_input_dim": 24,
                                                  "hidden_dims": [8],
                                                  "n_outputs": 5}}}}
    return WellnessEvaluators(cfg, WellnessMode.DIRECT, MotorStateFormula.BOUNDARY)


def _frame(pressure_pa: float):
    patches = np.zeros((4, 10), np.float32)
    patches[0, 0] = pressure_pa
    bones = np.zeros((24, 11), np.float32)
    bones[:, 4] = np.linspace(-15, 15, 24)
    return SimpleNamespace(bones=bones, patches=patches,
                           tokenized=SimpleNamespace(
                               motor_capability=np.ones(21, np.float32)))


def test_probe_touch_uses_normalized_fields():
    """09-09 review: 5,000 Pa raw into the comfort formula scored 0 while the
    training route (fields / _FIELD_NORMS) scored 0.675 - every contact in the
    24-event artifact was invisible to the touch channel."""
    w = SimpleNamespace(wellness=_evaluators())
    gentle = state_wellness(w, _frame(5000.0))
    assert gentle["tactile"] > 0.5, (
        "a gentle 5,000 Pa contact must score as comfortable, not zero - "
        "raw-Pa units are back in the probe")
    crush = state_wellness(w, _frame(100000.0))
    assert crush["tactile"] < gentle["tactile"], (
        "full-scale pressure must score worse than gentle contact")


def test_worst_moment_gates_the_wellness_verdict():
    """09-09 review: better endpoint + much worse minimum still passed. The
    verdict must now refuse the steering label when the worst moment during
    execution is measurably worse."""
    rows = [{"w_chosen": .1, "w_matched": 0., "w_replay": .1,
             "wmin_chosen": 0., "wmin_matched": 2., "wmin_replay": 0.,
             "contact_bearing": True}
            for _ in range(24)]
    out = _wellness_verdict(rows)
    assert out["verdict"] == "endpoint_gain_with_transient_harm"
    # and a clean gain (no transient harm) still passes
    for r in rows:
        r.update(wmin_chosen=1.0, wmin_matched=1.0, wmin_replay=1.0)
    assert _wellness_verdict(rows)["verdict"] == "wellness_steering_detected"


def test_worst_moment_line_uses_the_minimums_own_replay_floor():
    """09-09 follow-up counterexample: an exactly repeatable 0.1 minimum loss
    passed as steering because the ENDPOINT replay error (0.5) was used as the
    minimum's floor. The minimum's line must come from the minimum's own
    replay disagreement (here 0)."""
    rows = [{"w_chosen": 1.0, "w_matched": 0.0, "w_replay": 0.5,
             "wmin_chosen": 0.9, "wmin_matched": 1.0, "wmin_replay": 0.9,
             "contact_bearing": True} for _ in range(24)]
    out = _wellness_verdict(rows)
    assert out["verdict"] == "endpoint_gain_with_transient_harm", (
        "a repeatable worse minimum must not hide behind the endpoint's "
        "replay noise")
    assert out["worst_moment_replay_floor_p90"] == 0.0


def test_preservation_requires_declared_margin():
    """09-09 review: 'goal pursuit unchanged' was claimed from a nonsignificant
    difference. Without a declared margin the instrument must say unmeasured;
    with one, the CI must actually exclude the margin."""
    rng = np.random.default_rng(0)
    ch = rng.normal(0.004, 0.01, 24)   # small positive mean = slightly WORSE
    mt = np.zeros(24)
    out = _measurement_verdict(ch, mt)
    assert out["preservation"] == "unmeasured (no declared margin supplied)"
    wide = _measurement_verdict(ch, mt, preserve_margin=0.5)
    assert wide["preservation"]["verdict"] == "preserved_within_declared_margin"
    tight = _measurement_verdict(ch, mt, preserve_margin=1e-6)
    assert tight["preservation"]["verdict"] == "preservation_NOT_established"
    # 09-09 follow-up counterexample: zero measured harm but a replay floor of
    # 0.2 must NOT certify preservation within 0.01 - an instrument that
    # cannot repeat to the margin cannot certify it
    zeros = np.zeros(24)
    noisy = _measurement_verdict(zeros, zeros, preserve_margin=0.01,
                                 replay_floor=np.full(24, 0.2))
    assert noisy["preservation"]["verdict"] == "preservation_NOT_established"


def test_freeze_dedupes_shared_modules():
    """09-09 review: [m, m] recorded the already-frozen False as m's 'prior'
    flag; restore left the trainable weight frozen forever."""
    m = torch.nn.Linear(4, 4)
    m.bias.requires_grad_(False)
    prior = _freeze_params([m, m])
    _restore_params(prior)
    assert m.weight.requires_grad and not m.bias.requires_grad


def test_present_capacity_can_flip_goal_winner():
    """09-09 review counterexample: capacity was ADDED equally to every
    candidate, so present need could never change the winner. Now the deficit
    multiplies each candidate's judged capability: a fresh body picks the
    comfortable memory; a depleted body picks the fresh-body memory."""
    import mimo_bridge.lem_model as LM

    class Judge:
        def score_predicted_wellness(self, sens, zm):
            v = sens["vision"][:, 0]
            out = torch.zeros(v.shape[0], 5)
            out[:, 1] = v                  # judged comfort follows the marker
            out[:, 4] = v
            # the LESS comfy memory shows a much FRESHER body (capability)
            out[:, 2] = torch.where(v > 0.5, torch.tensor(0.05),
                                    torch.tensor(0.9))
            return out

        def _formula_capability(self, cap, B, device):
            return cap[:, :1]

    stub = SimpleNamespace(
        goal_ring=[({"vision": torch.tensor([[0.9]]),
                     "motor": torch.zeros(1, 1)}, torch.tensor([0.0])),
                   ({"vision": torch.tensor([[0.4]]),
                     "motor": torch.zeros(1, 1)}, torch.tensor([0.0]))],
        active=["vision", "motor"], d_model=1, wellness=Judge())
    wl = torch.zeros(2, 1)
    stub._last_capability = torch.full((1, 21), 1.0)     # fresh body
    fresh = LM.LemLite._revalue_ring(stub, wl)
    assert int(fresh.argmax(0)) == 0, "a fresh body ranks by judged comfort"
    stub._last_capability = torch.full((1, 21), 0.0)     # spent body
    spent = LM.LemLite._revalue_ring(stub, wl)
    assert int(spent.argmax(0)) == 1, (
        "a depleted body must be able to prefer the remembered fresh-body "
        "state - present need no longer shifts all candidates equally")


def _audio_encoder(freq_band_path: str):
    from lem.model.encoders.conv1d_stack import Conv1DStackEncoder
    torch.manual_seed(0)
    return Conv1DStackEncoder(conv_channels=[8, 8], kernel_size=3,
                              in_channels=2, spatial_branch_input_dim=2,
                              d_internal=16, d_output=16, combine="add",
                              dropout=0.0, bias=True, norm_type="layernorm",
                              pool_n_heads=1, freq_band_path=freq_band_path)


def _tone(center: int) -> torch.Tensor:
    s = torch.zeros(1, 256, 2)
    s[0, center - 2:center + 3, :] = 1.0
    return s


def test_audio_conv_average_erases_frequency_position():
    """The reviewer's measured counterexample, pinned: with the band path off,
    the same 5-bin pattern at bins 40/120/200 gives IDENTICAL tokens - the
    architecture, not undertraining, erases absolute frequency."""
    enc = _audio_encoder("off").eval()
    sp = torch.zeros(1, 2)
    outs = [enc(_tone(c), sp) for c in (40, 120, 200)]
    assert torch.allclose(outs[0], outs[1], atol=1e-6)
    assert torch.allclose(outs[1], outs[2], atol=1e-6)


def test_audio_band_path_restores_frequency_and_loads_old_ckpts_exact():
    """The fix: banded means through a zero-init projection. Zero init means
    behavior is bit-identical at load (old checkpoints stay valid); once the
    projection is nonzero, the three tones separate."""
    off = _audio_encoder("off").eval()
    on = _audio_encoder("on").eval()
    on.load_state_dict(off.state_dict(), strict=False)   # old-ckpt load path
    sp = torch.zeros(1, 2)
    for c in (40, 120, 200):
        assert torch.allclose(on(_tone(c), sp), off(_tone(c), sp), atol=1e-6), (
            "zero-init band path must not change a loaded old checkpoint")
    with torch.no_grad():
        on.band_proj.weight.copy_(torch.randn_like(on.band_proj.weight))
    outs = [on(_tone(c), sp) for c in (40, 120, 200)]
    assert not torch.allclose(outs[0], outs[1], atol=1e-4)
    assert not torch.allclose(outs[1], outs[2], atol=1e-4)
    # and the information really is absolute position: gradient can reach it
    assert on.band_proj.weight.requires_grad


def test_build_model_forwards_audio_band_flag(tmp_path, monkeypatch):
    """Probe/run parity (the four-strikes plumbing-hole class): a run trained
    with --audio-freq-bands on must be rebuilt with the band path on."""
    import json as _json
    from mimo_bridge.tools import causal_test as ct
    run = tmp_path / "R3"
    ckpt = run / "checkpoints" / "c1"
    ckpt.mkdir(parents=True)
    (run / "args.json").write_text(_json.dumps({
        "context_length": 4, "residual_prediction": "on", "intention": "on",
        "audio_freq_bands": "on"}), encoding="utf-8")
    seen = {}

    class Spy:
        def __init__(self, **kw):
            seen.update(kw)
            raise RuntimeError("spy stop")

    import mimo_bridge.lem_model as LM
    import mimo_bridge.train_mimo as TM
    monkeypatch.setattr(LM, "LemLite", Spy)
    # since 2026-09-22 _build_model builds through train_mimo.build_learner, which
    # calls the LemLite name train_mimo imported - the spy must sit there too
    monkeypatch.setattr(TM, "LemLite", Spy)
    with pytest.raises(RuntimeError):
        ct._build_model(str(ckpt), "cpu", 0, intention=True)
    enc = (seen.get("overrides") or {}).get("modalities", {}) \
        .get("audio", {}).get("encoder", {})
    assert enc.get("freq_band_path") == "on"


def test_probe_touch_matches_training_on_dense_contact():
    """09-09 follow-up counterexample: with 53 contacts (21 gentle), the probe
    averaged ALL of them (0.242) while training's strongest-32 point-cloud
    preparation reads the strong ones (0.017). The probe must select exactly
    as training does."""
    from tokenizer.modalities.touch import _FIELD_NORMS
    w = SimpleNamespace(wellness=_evaluators())
    patches = np.zeros((53, 10), np.float32)
    patches[:32, 0] = 50000.0          # strong contacts
    patches[32:, 0] = 500.0            # gentle contacts
    frame = SimpleNamespace(bones=np.zeros((24, 11), np.float32),
                            patches=patches,
                            tokenized=SimpleNamespace(
                                motor_capability=np.ones(21, np.float32)))
    got = state_wellness(w, frame)["tactile"]
    # training's route, built independently: strongest 32 by pressure
    normed = torch.as_tensor(patches[:, 0:8] / _FIELD_NORMS[:8])
    top = normed[torch.argsort(normed[:, 0], descending=True)[:32]]
    expect = float(w.wellness._formula_tactile_comfort(top[None], 1,
                                                       torch.device("cpu")))
    assert abs(got - expect) < 1e-6, (
        f"probe {got} != training-preparation {expect}: the strongest-32 "
        "selection is not being mirrored")


def test_optimizer_state_migrates_by_name_when_params_grow():
    """09-09 follow-up review: enabling the audio band path dropped ALL 256
    saved Adam states (torch refuses a group-size change). Migration by
    qualified name must preserve every saved moment and step counter, leaving
    only the genuinely new parameter fresh."""
    from lem.training.checkpoint import _migrate_optimizer_state
    torch.manual_seed(0)

    class Old(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(4, 4)
            self.c = torch.nn.Linear(4, 2)

    class New(Old):
        def __init__(self):
            super().__init__()
            self.b = torch.nn.Linear(3, 4, bias=False)   # inserted MID-order

    old = Old()
    opt_old = torch.optim.Adam(old.parameters(), lr=1e-3)
    (old.c(old.a(torch.randn(5, 4))).sum()).backward()
    opt_old.step()
    saved = opt_old.state_dict()
    assert len(saved["state"]) == 4          # a.w, a.b, c.w, c.b

    new = New()
    opt_new = torch.optim.Adam(new.parameters(), lr=1e-3)
    id2name = {id(p): f"m.{n}" for n, p in new.named_parameters()}
    saved_names = {f"m.{n}" for n, _ in old.named_parameters()}
    sd, fresh = _migrate_optimizer_state(opt_new, saved, id2name, saved_names)
    assert fresh == ["m.b.weight"]
    assert len(sd["state"]) == 4
    opt_new.load_state_dict(sd)
    # every migrated state sits under the parameter it belonged to: the
    # exp_avg shapes must match their new owners, and step counters survive
    for group in opt_new.param_groups:
        for p in group["params"]:
            st = opt_new.state.get(p)
            if id2name[id(p)] in saved_names:
                assert st and st["exp_avg"].shape == p.shape
                assert int(st["step"]) == 1
            else:
                assert not st                 # fresh param: no history yet
    # and a real update after migration works
    (new.c(new.a(torch.randn(5, 4)) + new.b(torch.randn(5, 3))).sum()).backward()
    opt_new.step()


def test_revalue_preserves_drive_contributions():
    """09-09 follow-up review: revaluation REPLACED the stored goal values,
    silently deleting curiosity/agency/movement from the ranking. Their
    contribution (stored value minus stored pre-adjustment bodily value) must
    ride on top of the re-felt bodily value."""
    import mimo_bridge.lem_model as LM
    from collections import deque

    class Judge:
        def score_predicted_wellness(self, sens, zm):
            v = sens["vision"][:, 0]
            out = torch.zeros(v.shape[0], 5)
            out[:, 1] = v
            out[:, 4] = v
            return out

        def _formula_capability(self, cap, B, device):
            return cap[:, :1]

    stub = SimpleNamespace(
        goal_ring=[({"vision": torch.tensor([[0.5]]),
                     "motor": torch.zeros(1, 1)}, torch.tensor([2.0])),
                   ({"vision": torch.tensor([[0.5]]),
                     "motor": torch.zeros(1, 1)}, torch.tensor([1.0]))],
        active=["vision", "motor"], d_model=1, wellness=Judge(),
        _last_capability=torch.full((1, 21), 1.0),
        # both moments had bodily value 1.0 at storage; the first carried a
        # +1.0 drive bonus (agency/curiosity) the ranking must keep
        _goal_ring_w0=deque([torch.tensor([1.0]), torch.tensor([1.0])]))
    wl = torch.stack([w for _, w in stub.goal_ring])
    out = LM.LemLite._revalue_ring(stub, wl)
    assert float(out[0, 0]) > float(out[1, 0]), (
        "equal bodies, unequal stored drives: the drive bonus must still "
        "separate the candidates after revaluation")
    assert abs(float(out[0, 0] - out[1, 0]) - 1.0) < 1e-6


def test_smoother_is_shared_and_hemisphere_aligned():
    """The executed-order smoother is ONE implementation used by the live step
    (source pin) and by offline probes; blending q against -q of the same
    rotation must not pass through zero."""
    import inspect
    import mimo_bridge.lem_model as LM
    step_src = inspect.getsource(LM.LemLite.step)
    assert "self.smooth_blend(mfa, self._last_smooth_prev" in step_src
    probe_src = open(Path(__file__).resolve().parents[1]
                     / "tools" / "first_action_benefit.py", encoding="utf-8").read()
    assert "LemLite.smooth_blend(" in probe_src
    cmd = torch.zeros(1, 2, 9)
    cmd[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    prev = cmd.clone()
    prev[..., 3:7] = -prev[..., 3:7]          # same rotation, opposite sign
    out = LM.LemLite.smooth_blend(cmd, prev, 0.35)
    assert torch.allclose(out[..., 3:7].abs(),
                          torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
