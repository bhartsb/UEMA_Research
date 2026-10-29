# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The vestibular sense in the intermind placeholder's slot (DESIGN.md D4, --vestibular-sense).

What is held here: the flag off adds nothing to the config; the flag on swaps the sixth input slot
from intermind to vestibular with every shape the rest of the model reads unchanged (Judge B's
width included); the placeholder system never fills it; a legacy vector is padded and marked not
real; an intermind checkpoint moves onto the vestibular build by name, weights and Adam moments
both, with the encoder fresh and started so the switch changes nothing the model computes (the
old slot held a learned constant); a vestibular checkpoint refuses the flag-off build; the replay
ring's stored slot-5 moments can never grade the new head; the input is centred and scaled by a
running mean and spread that only real frames move; the sense is graded in the next-moment lesson
at its declared weight and only from a real moment; the stats block's angles read the conventions
they claim and it reports the encoder's liveness.
"""
import logging
import math
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lem.config import resolver as R
from lem.model import wellness as W
from mimo_bridge import continuation_state as CS
from mimo_bridge import vestibular_adapter as VA
from mimo_bridge import vestibular_sense as VS
from mimo_bridge.lem_model import LITE_OVERRIDES, LemLite, build_config
from mimo_bridge.vestibular_adapter import FIELD_INDEX, VESTIBULAR_DIM, pad_legacy_vestibular


def _lite(**kw):
    torch.set_num_threads(1)
    return LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                   top_down_gate=True, body_layout="trunk_v1", **kw)


def _vec(valid=1.0, **fields):
    v = np.zeros(VESTIBULAR_DIM, dtype=np.float32)
    v[0] = 1.0                       # lying still on the back: one g along the head's x
    for k, x in fields.items():
        v[FIELD_INDEX[k]] = x
    v[FIELD_INDEX["valid"]] = valid
    return v


# ------------------------------------------------------------------------------ config
def test_flag_off_adds_nothing_to_the_config():
    m = _lite()
    assert "vestibular" not in m.config["modalities"]
    assert m.active[-1] == "intermind" and not m.vestibular_sense
    assert "vestibular_sense" not in m.receipt and m.vestibular_switch is None


def test_the_flag_swaps_the_sixth_slot_and_every_shape_the_model_reads_stays():
    off, on = _lite(), _lite(vestibular_sense=True)
    assert on.active == ["vestibular" if x == "intermind" else x for x in off.active]
    assert on.active.index("vestibular") == off.active.index("intermind")
    assert on.n_mod == off.n_mod
    assert on.forward_model.modality_embedding.weight.shape == off.forward_model.modality_embedding.weight.shape
    assert on.config["model"]["motor_controller"]["_input_dim"] == off.config["model"]["motor_controller"]["_input_dim"]
    # Judge B keeps its five-token input (four with no vision in this lite build)
    assert (on.config["model"]["wellness"]["approximate"]["_input_dim"]
            == off.config["model"]["wellness"]["approximate"]["_input_dim"])
    assert on.wellness._sensory_indices == off.wellness._sensory_indices
    for mods in ("recon_decoders", "td_gates"):
        a, b = getattr(off, mods), getattr(on, mods)
        assert list(b.keys()) == ["vestibular" if k == "intermind" else k for k in a.keys()]
        assert ([p.shape for p in b["vestibular"].parameters()]
                == [p.shape for p in a["intermind"].parameters()])
    assert ([p.shape for p in on.prediction_heads.heads["vestibular"].parameters()]
            == [p.shape for p in off.prediction_heads.heads["intermind"].parameters()])
    enc = on.encoder_registry.encoders["vestibular"]
    # the encoder reads the 28 fields without the validity flag, which is a mask only
    assert enc.mlp[0].in_features == VS.ENCODER_DIM == VESTIBULAR_DIM - 1
    assert "valid" not in VS.ENCODER_FIELDS
    # every layer has a bias: the switch starts the last one at the old constant
    assert all(m.bias is not None for m in enc.mlp if isinstance(m, torch.nn.Linear))
    st = on.receipt["vestibular_sense"]
    assert st["fields"][FIELD_INDEX["valid"]] == "valid" and st["dim"] == VESTIBULAR_DIM
    assert st["encoder_fields"] == list(VS.ENCODER_FIELDS)
    assert st["input_bound"] == VS.INPUT_BOUND
    assert st["lesson_weight"] == VS.LESSON_WEIGHT and st["wellness_pe"] == "exclude"
    # the same-moment route is guarded to 0.0 like every other sense under --next-frame-loss
    assert on.loss_manager._loss_weights["vestibular"] == 0.0


def test_the_encoder_is_built_like_proprioception_at_the_runs_width():
    from mimo_bridge.train_mimo import size_overrides
    from lem.config.resolver import _deep_merge
    ov = size_overrides(SimpleNamespace(encoder_width=192, encoder_layers=2, core_blocks=0,
                                        d_model=0, vision_swin_layers=0, vision_width=0))
    cfg = build_config(_deep_merge(ov, VS.modality_overrides(ov, LITE_OVERRIDES)), no_vision=True)
    enc = cfg["modalities"]["vestibular"]["encoder"]
    assert enc["hidden_dims"] == cfg["modalities"]["proprioception"]["encoder"]["hidden_dims"] == [192, 192, 192]
    assert enc["_input_dim"] == VS.ENCODER_DIM and enc["type"] == "mlp_project"
    assert enc["bias"] is True
    # every other encoder keeps the shared bias-free default
    assert cfg["modalities"]["proprioception"]["encoder"]["bias"] is False
    lite = build_config(VS.modality_overrides({}, LITE_OVERRIDES), no_vision=True)
    assert lite["modalities"]["vestibular"]["encoder"]["hidden_dims"] == [128, 128]


def test_the_slot_follows_intermind_even_when_voice_is_switched_on():
    ov = VS.modality_overrides({}, LITE_OVERRIDES)
    ov["modalities"]["voice"] = {"enabled": True}
    cfg = build_config(ov)
    base = build_config({"modalities": {"voice": {"enabled": True}}})
    i = base["_active_modalities"].index("intermind")
    assert cfg["_active_modalities"][i] == "vestibular"
    assert cfg["_active_modalities"][-1] == "voice"


def test_the_two_exclusion_lists_agree():
    assert R.APPROXIMATE_EXCLUDED == W.APPROXIMATE_EXCLUDED == ("intermind", "vestibular")


def test_judge_b_ignores_a_vestibular_token():
    m = _lite(vestibular_sense=True)
    B = 3
    toks = {k: torch.randn(B, m.d_model) for k in m.active if k != "motor"}
    z = torch.randn(B, m.d_model)
    a = m.wellness.score_predicted_wellness(toks, z)
    toks["vestibular"] = torch.randn(B, m.d_model) * 100.0
    b = m.wellness.score_predicted_wellness(toks, z)
    assert torch.equal(a, b)


def test_the_placeholder_system_never_fills_the_sense():
    m = _lite(vestibular_sense=True)
    gens = m.placeholders._generators
    assert "vestibular" not in gens and "intermind" not in gens
    frame = {}
    mask = m.placeholders.fill_frame(frame, batch_size=2)
    assert "vestibular" not in frame and "vestibular" not in mask


def test_the_trainer_flags_default_off_and_reach_the_learner(monkeypatch):
    import inspect
    from mimo_bridge import train_mimo as TM
    d = TM.parse_args(["--out", "x"])
    assert d.vestibular_sense == "off" and d.vestibular_lesson_weight is None
    assert d.vestibular_wellness_pe == "exclude"
    a = TM.parse_args(["--out", "x", "--vestibular-sense", "on", "--vestibular-lesson-weight",
                       "0.5", "--vestibular-wellness-pe", "include"])
    assert a.vestibular_sense == "on"
    # it changes what drives the body, so a frozen actor's qualification binds it
    assert TM.acting_configuration(a)["vestibular_sense"] == "on"
    # the continuation flags are a literal tuple inside main(); the sense is not one of them, so
    # a resume may switch it on
    src = inspect.getsource(TM.main)
    flags = src[src.index("continuation_flags = ("):src.index("continuation_config = ")]
    assert "vestibular" not in flags
    # the flags reach the learner (build_learner's own call, with the learner replaced)
    seen = {}

    class Stub:
        def __init__(self, **kw):
            seen.update(kw)
    monkeypatch.setattr(TM, "LemLite", Stub)
    TM.build_learner(a, None)
    assert seen["vestibular_sense"] is True and seen["vestibular_lesson_weight"] == 0.5
    assert seen["vestibular_wellness_pe"] == "include"
    seen.clear()
    TM.build_learner(d, None)
    assert seen["vestibular_sense"] is False and seen["vestibular_lesson_weight"] is None
    assert seen["vestibular_wellness_pe"] == "exclude"


def test_the_two_settings_need_the_sense_and_are_checked():
    with pytest.raises(ValueError, match="--vestibular-sense on"):
        _lite(vestibular_lesson_weight=0.5)
    with pytest.raises(ValueError, match="--vestibular-sense on"):
        _lite(vestibular_wellness_pe="include")
    with pytest.raises(ValueError, match="0 or more"):
        _lite(vestibular_sense=True, vestibular_lesson_weight=-1.0)
    with pytest.raises(ValueError, match="exclude"):
        _lite(vestibular_sense=True, vestibular_wellness_pe="maybe")
    m = _lite(vestibular_sense=True, vestibular_lesson_weight=0.25, vestibular_wellness_pe="include")
    assert m.vestibular_lesson_weight == 0.25 and m.wellness.pe_excluded == ()
    assert _lite(vestibular_sense=True).wellness.pe_excluded == ("vestibular",)
    off = _lite()
    assert off.vestibular_norm is None and off.wellness.pe_excluded == ()


def test_vestibular_is_a_predicted_sense_so_a_scale_keyed_by_pred_mods_covers_it():
    # review b2-vestibular finding 2: the per-sense scaling of the next-moment lesson (DESIGN D12,
    # --next-frame-norm persistence) is keyed by pred_mods; the sense must be one of them
    m = _lite(vestibular_sense=True)
    assert "vestibular" in m.pred_mods
    assert m.pred_mods == [x for x in m.active if x != "motor"]
    assert "vestibular" in m.prediction_heads.heads


def test_wellness_prediction_error_leaves_out_an_excluded_sense():
    m = _lite(vestibular_sense=True)
    B = 2
    d = m.d_model
    tg = {"tactile": torch.zeros(B, d), "vestibular": torch.zeros(B, d)}
    pr = {"tactile": torch.full((B, d), 0.1), "vestibular": torch.full((B, d), 0.5)}
    valid = {"tactile": "real", "vestibular": "real"}
    w = m.wellness
    only_touch = 1.0 - 0.01
    got = w._formula_prediction_error(pr, tg, B, torch.device("cpu"), valid=valid)
    assert torch.allclose(got, torch.full((B, 1), only_touch))
    w.pe_excluded = ()
    got = w._formula_prediction_error(pr, tg, B, torch.device("cpu"), valid=valid)
    assert torch.allclose(got, torch.full((B, 1), 1.0 - (0.01 + 0.25) / 2))


# ------------------------------------------------------------------------------ frames
def test_padding_clipping_and_validity():
    frames = [SimpleNamespace(vestibular=_vec()),
              SimpleNamespace(vestibular=np.array([9.81, 0, 0, 0.5, 0, 0], np.float32)),
              SimpleNamespace(vestibular=None),
              SimpleNamespace(vestibular=_vec(head_acc_z=41.6, support_total=18.2))]
    v, ok, clipped = VS.frame_vectors(frames)
    assert v.shape == (4, VESTIBULAR_DIM) and v.dtype == np.float32
    assert ok.tolist() == [True, False, False, True]
    assert np.allclose(v[1], pad_legacy_vestibular(np.array([9.81, 0, 0, 0.5, 0, 0], np.float32)))
    assert v[1, FIELD_INDEX["head_acc_x"]] == pytest.approx(1.0, abs=1e-6)
    assert np.all(v[2] == 0.0)
    assert v[3, FIELD_INDEX["head_acc_z"]] == VS.INPUT_BOUND
    assert v[3, FIELD_INDEX["support_total"]] == VS.INPUT_BOUND
    assert clipped.tolist() == [0, 0, 0, 2]


def _bare_frames(vectors):
    md = SimpleNamespace(modality_valid_mask=0, gaze_azimuth_deg=0.0, gaze_elevation_deg=0.0)
    return [SimpleNamespace(tokenized=SimpleNamespace(metadata=md), vision_patches=None,
                            vestibular=v, meta={}) for v in vectors]


def test_a_batch_is_real_only_when_every_body_read_is_real():
    m = _lite(vestibular_sense=True)
    frame, mask, _, _ = m._batch_frames(_bare_frames([_vec(), _vec(support_trunk=0.5)]))
    assert mask["vestibular"] == "real"
    assert frame["vestibular"].shape == (2, VS.ENCODER_DIM)
    legacy = np.array([9.81, 0.1, 0.2, 0.0, 0.0, 0.0], np.float32)
    frame, mask, _, _ = m._batch_frames(_bare_frames([_vec(), legacy]))
    assert mask["vestibular"] == "stale"
    off = _lite()
    frame, mask, _, _ = off._batch_frames(_bare_frames([_vec(), _vec()]))
    assert "vestibular" not in frame and "vestibular" not in mask


# ------------------------------------------------------------------------------ the normaliser
def _rows(n, seed=0):
    g = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        v = _vec(support_trunk=0.8, capacity_min=0.6, capacity_mean=0.85, trunk_up_x=1.0)
        v[:28] += g.normal(0.0, 0.05, 28).astype(np.float32)
        out.append(v)
    return out


def test_the_normaliser_centres_and_scales_on_real_frames_only():
    n = VS.VestibularNormaliser(horizon_steps=10)
    frames = _bare_frames(_rows(4, 1))
    z, ok, why = n.batch(frames)
    # before any real frame there is nothing to centre against: every input reads 0
    assert z.shape == (4, VS.ENCODER_DIM) and np.all(z == 0.0) and ok.all()
    n.commit()
    assert n.weight == pytest.approx(0.1) and n.n_real == 4
    x = np.stack([f.vestibular for f in frames]).astype(np.float64)[:, VS._ENC_IDX]
    # the first real batch sets the mean exactly (the correction for starting at zero) and the
    # spread from its own rows
    assert np.allclose(n.mean, x.mean(0)) and np.allclose(n.var, x.var(0))
    # a stale batch moves nothing: legacy, missing and non-finite reads
    legacy = np.array([9.81, 0, 0, 0, 0, 0], np.float32)
    bad = _vec()
    bad[3] = np.nan
    before = (n.mean.copy(), n.var.copy(), n.weight, n.n_real)
    z, ok, why = n.batch([SimpleNamespace(vestibular=legacy), SimpleNamespace(vestibular=None),
                          SimpleNamespace(vestibular=bad)])
    n.commit()
    assert not ok.any() and (n.mean == before[0]).all() and (n.var == before[1]).all()
    assert n.weight == before[2] and n.n_real == before[3]
    assert why == [VS.STALE_LEGACY, VS.STALE_NONE, VS.STALE_NONFINITE]
    # a field the frame does not carry reads as the running mean: 0 after centring
    legacy_known = [VS.ENCODER_FIELDS.index(f) for f in VS.LEGACY_FIELDS]
    others = [i for i in range(VS.ENCODER_DIM) if i not in legacy_known]
    assert np.all(z[0, others] == 0.0) and np.all(z[1] == 0.0)
    assert z[2, VS.ENCODER_FIELDS.index("head_gyro_x")] == 0.0
    # the legacy frame's six MIMo numbers are read on the live scale
    exp = (1.0 - n.mean[0]) / n.spread()[0]
    assert z[0, 0] == pytest.approx(float(np.clip(exp, -4, 4)), rel=1e-5)
    # a real read is centred and scaled, and bounded at 4 running spreads
    far = _vec()
    far[FIELD_INDEX["support_trunk"]] = 3.0
    z, ok, _ = n.batch([SimpleNamespace(vestibular=far)])
    assert z[0, VS.ENCODER_FIELDS.index("support_trunk")] == VS.NORMALISED_BOUND
    s = n.window_summary()
    assert s["input_clipped_share"] > 0 and s["input_real_body_moments"] == 4
    assert s["input_spread_min"]["floor"] == VS.SPREAD_FLOOR


def test_the_running_statistics_follow_a_new_level_over_their_horizon():
    n = VS.VestibularNormaliser(horizon_steps=20)
    a = _vec(support_trunk=0.2)
    for _ in range(5):
        n.batch([SimpleNamespace(vestibular=a)])
        n.commit()
    k = VS.ENCODER_FIELDS.index("support_trunk")
    assert n.mean[k] == pytest.approx(0.2, abs=1e-6)
    b = _vec(support_trunk=0.8)
    for _ in range(200):
        n.batch([SimpleNamespace(vestibular=b)])
        n.commit()
    assert n.mean[k] == pytest.approx(0.8, abs=0.01)
    assert n.spread()[k] >= VS.SPREAD_FLOOR


def test_frozen_steps_do_not_move_the_statistics_and_they_ride_the_runtime_state(tmp_path):
    m = _lite(vestibular_sense=True, checkpoint_dir=str(tmp_path))
    # _batch_frames holds the real rows; only a training step's commit folds them in
    m._batch_frames(_bare_frames(_rows(3, 2)))
    assert m.vestibular_norm.n_real == 0
    m.vestibular_norm.commit()
    assert m.vestibular_norm.n_real == 3
    path = m.save_checkpoint(step=4)
    again = _lite(vestibular_sense=True)
    again.load_checkpoint(path, load_optimizer=False)
    assert again.vestibular_norm.n_real == 3
    assert np.array_equal(again.vestibular_norm.mean, m.vestibular_norm.mean)
    assert np.array_equal(again.vestibular_norm.var, m.vestibular_norm.var)


def test_a_stale_batch_is_logged_once_per_reason_except_the_resume_frame(caplog):
    n = VS.VestibularNormaliser()
    caplog.set_level(logging.WARNING, logger="mimo_bridge.vestibular_sense")
    legacy = np.array([9.81, 0, 0, 0, 0, 0], np.float32)
    n.batch([SimpleNamespace(vestibular=legacy)])
    assert not caplog.records
    for _ in range(3):
        n.batch([SimpleNamespace(vestibular=None), SimpleNamespace(vestibular=_vec())])
    assert len(caplog.records) == 1 and VS.STALE_NONE in caplog.records[0].getMessage()
    n.batch([SimpleNamespace(vestibular=_vec(valid=0.0))])
    assert len(caplog.records) == 2 and VS.STALE_INVALID in caplog.records[1].getMessage()


# ------------------------------------------------------------------------------ checkpoints
def _fill_optimizer(m, seed):
    g = torch.Generator().manual_seed(seed)
    for p in m.params:
        m.optimizer.state[p] = {"step": torch.tensor(float(7 + seed)),
                                "exp_avg": torch.randn(p.shape, generator=g),
                                "exp_avg_sq": torch.rand(p.shape, generator=g)}


def _named(m):
    out = {}
    for comp, mod in VS._components(m).items():
        for k, t in mod.state_dict().items():
            out[f"{comp}.{k}"] = t
    return out


def _param_names(m):
    out = {}
    for comp, mod in VS._components(m).items():
        for k, p in mod.named_parameters():
            out[f"{comp}.{k}"] = p
    return out


def test_an_intermind_checkpoint_moves_onto_the_vestibular_build_by_name(tmp_path):
    torch.manual_seed(11)
    off = _lite(checkpoint_dir=str(tmp_path))
    with torch.no_grad():                    # make the slot's modules unmistakable
        for p in off.prediction_heads.heads["intermind"].parameters():
            p.add_(torch.randn_like(p))
        for p in off.td_gates["intermind"].parameters():
            p.add_(3.0)
    _fill_optimizer(off, 1)
    path = off.save_checkpoint(step=5)
    torch.manual_seed(12)
    on = _lite(vestibular_sense=True)
    fresh_enc = {k: v.clone() for k, v in on.encoder_registry.encoders["vestibular"].state_dict().items()}
    res = on.load_checkpoint(path, load_optimizer=True)
    rep = on.vestibular_switch
    assert rep is not None and rep["slot"] == on.active.index("vestibular")
    assert "encoder:vestibular" in res["reinitialized_components"]
    declared = set(rep["function_preserving"]["tensors_set"])
    # every saved tensor outside the old encoder is in the new build, under its new name, except
    # the declared last layer of the slot's head, which starts at 0
    saved, now = _named(off), _named(on)
    n_checked = 0
    for k, t in saved.items():
        if k.startswith(VS.ENC_OLD):
            continue
        k2 = VS._rename(k, "intermind", "vestibular")
        if k2 in declared:
            # the head's last layer: weight 0; its bias is the old token here (no residual
            # prediction in this build; test_without_residual_prediction_... holds its value)
            assert not torch.equal(now[k2], t), k2
            if k2.endswith(".weight"):
                assert float(now[k2].abs().max()) == 0.0, k2
            continue
        assert torch.equal(now[k2], t), k
        n_checked += 1
    assert n_checked > 100
    # the encoder is the new build's own fresh one, except its declared last layer
    last = len(on.encoder_registry.encoders["vestibular"].mlp) - 1
    for k, v in on.encoder_registry.encoders["vestibular"].state_dict().items():
        if k.startswith(f"mlp.{last}."):
            assert VS.ENC_NEW + k in declared
            continue
        assert torch.equal(v, fresh_enc[k])
    # every saved Adam moment of a kept parameter is on that parameter now; the new encoder has
    # none, and the slot's head, gate and decoder start fresh (their carried moments were 0 on
    # the real checkpoint, with a step count that switched Adam's start-up correction off)
    dropped = set(rep["adam_states_dropped_fresh_names"])
    slot = [n for n in _param_names(on) if ".vestibular." in f".{n}." and not n.startswith(VS.ENC_NEW)]
    assert dropped == set(slot) and len(dropped) > 0
    old_p, new_p = _param_names(off), _param_names(on)
    carried = 0
    for n, p in new_p.items():
        if n.startswith(VS.ENC_NEW) or n in dropped:
            assert p not in on.optimizer.state or not on.optimizer.state[p]
            continue
        src = off.optimizer.state[old_p[VS.saved_name(n)]]
        dst = on.optimizer.state[p]
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(dst[key].float(), src[key].float()), (n, key)
        carried += 1
    n_old_enc = len(list(off.encoder_registry.encoders["intermind"].parameters()))
    assert carried == len(off.params) - n_old_enc - len(dropped)
    assert rep["adam_states_carried"] == carried + len(dropped)
    assert rep["adam_states_dropped_with_old_encoder"] == n_old_enc
    assert rep["adam_states_saved"] == len(off.params)
    assert rep["adam_states_dropped_fresh"] == len(dropped)
    # and the optimizer steps with the carried moments without a shape clash
    for p in on.params:
        p.grad = torch.zeros_like(p)
    on.optimizer.step()


def test_a_misaligned_saved_optimizer_is_refused_before_anything_changes(tmp_path):
    off = _lite(checkpoint_dir=str(tmp_path))
    _fill_optimizer(off, 2)
    path = off.save_checkpoint(step=5)
    on = _lite(vestibular_sense=True)
    import safetensors.torch as st
    saved = st.load_file(f"{path}/model.safetensors")
    opt = torch.load(f"{path}/optimizer.pt", weights_only=False)["optimizer"]
    g0 = opt["param_groups"][0]["params"]
    g0[3], g0[4] = g0[4], g0[3]          # two positions swapped: the layout no longer matches
    before = on.optimizer.state_dict()
    with pytest.raises(ValueError):
        VS.migrate_optimizer(on, opt, saved)
    assert on.optimizer.state_dict()["state"] == before["state"] == {}
    opt["param_groups"][0]["params"] = g0[:-1]
    with pytest.raises(ValueError):
        VS.migrate_optimizer(on, opt, saved)


def test_a_vestibular_checkpoint_refuses_the_flag_off_build(tmp_path):
    on = _lite(vestibular_sense=True, checkpoint_dir=str(tmp_path))
    path = on.save_checkpoint(step=3)
    with pytest.raises(ValueError, match="vestibular"):
        _lite().load_checkpoint(path)
    # and the same build reloads it with no switch
    again = _lite(vestibular_sense=True)
    again.load_checkpoint(path)
    assert again.vestibular_switch is None


def test_a_changed_field_list_or_scale_is_refused():
    st = VS.stamp()
    assert VS.stamp_mismatch(st) is None and VS.stamp_mismatch(None) is None
    assert "dim" in VS.stamp_mismatch(dict(st, dim=VESTIBULAR_DIM - 1))
    assert VS.stamp_mismatch(dict(st, scales=dict(st["scales"], head_gyro_rads=3.0))) is not None
    assert VS.stamp_mismatch(dict(st, input_bound=8.0)) is not None
    # a v1 checkpoint read the 29 raw numbers: refused by version
    assert "v1" in VS.stamp_mismatch(dict(st, version="vestibular_sense_v1"))
    assert VS.stamp_mismatch(dict(st, normaliser=dict(st["normaliser"], horizon_steps=10)))


@pytest.mark.parametrize("name,value", [("LIMIT_BAND_RAD", 0.05), ("SUPPORT_GROUPS", ("a", "b")),
                                        ("NECK_JOINTS", ("x", "y", "z")),
                                        ("DEFINITION_VERSION", "2026-10-01 fatigue gear")])
def test_the_stamp_covers_what_each_field_means(monkeypatch, name, value):
    # review b2-vestibular finding 10: the limit band, the support groups, the neck lists and a
    # definition version the adapter's owner may add are in the stamp, read from the adapter
    saved = VS.stamp()
    monkeypatch.setattr(VA, name, value, raising=False)
    why = VS.stamp_mismatch(saved)
    assert why is not None and "adapter" in why
    assert VS.adapter_sha() != saved["adapter_sha"]


# ------------------------------------------------------------------------------ the switch
RESID = {"model": {"residual_prediction": True}}


def _constant_intermind(m):
    """Make the placeholder encoder output one nonzero constant whatever its noise input: the
    embedding reads nothing, and the first block's feed-forward norm carries a bias."""
    enc = m.encoder_registry.encoders["intermind"]
    g = torch.Generator().manual_seed(9)
    with torch.no_grad():
        enc.embed.weight.zero_()
        b = enc.layers[0]["ffn_norm"].bias
        b.copy_(torch.randn(b.shape, generator=g))


def _switched(tmp_path, load_optimizer=False, fill=False, **kw):
    """An intermind lite checkpoint whose placeholder encoder outputs one constant (as the trained
    H84 one does: its token moved 0.0000049 against a size of 13.86), loaded into the vestibular
    build."""
    torch.manual_seed(21)
    off = _lite(checkpoint_dir=str(tmp_path), overrides=RESID, **kw)
    _constant_intermind(off)
    with torch.no_grad():
        for p in off.prediction_heads.heads["intermind"].parameters():
            p.add_(torch.randn_like(p))
        if off.persistent_memory is not None:
            k = off.active.index("intermind")
            d = off.d_model
            off.persistent_memory.input_projection[0].weight[:, k * d:(k + 1) * d] += 0.3
    if fill:
        _fill_optimizer(off, 3)
    path = off.save_checkpoint(step=5)
    on = _lite(vestibular_sense=True, overrides=RESID, **kw)
    on.load_checkpoint(path, load_optimizer=load_optimizer)
    return off, on, path


def test_the_switch_starts_the_slot_where_the_old_model_was(tmp_path):
    off, on, _ = _switched(tmp_path, persistent_memory="on")
    B = 3
    ph = off.placeholders._generators["intermind"].generate(B).clone()
    old = off.encoder_registry.encode({"intermind": ph}, {"intermind": "synthetic"})["intermind"]
    new = on.encoder_registry.encode({"vestibular": torch.randn(B, VS.ENCODER_DIM) * 3.0},
                                     {"vestibular": "real"})["vestibular"]
    # whatever the vestibular input, the token is the old constant
    assert torch.allclose(new, old, atol=1e-5), float((new - old).abs().max())
    rep = on.vestibular_switch["function_preserving"]
    assert rep["old_token_spread"] < 1e-4 and rep["residual_prediction"]
    assert rep["old_token_size"] > 1.0 and float(old.norm(dim=-1).min()) > 1.0
    # the slot's head predicts no change: prediction = the current token
    last = torch.randn(B, on.n_mod, on.d_model)
    cur = {m: torch.randn(B, on.d_model) for m in on.pred_mods}
    pred = on.prediction_heads(last, current=cur, who="test")
    assert torch.equal(pred["vestibular"], cur["vestibular"])
    # the memory never read this slot (it was fed zeros for a placeholder): those weights are 0
    k = on.active.index("vestibular")
    d = on.d_model
    for a, b in ((on.persistent_memory.input_projection[0].weight,
                  off.persistent_memory.input_projection[0].weight),
                 (on.persistent_memory.query.weight, off.persistent_memory.query.weight)):
        assert float(a[:, k * d:(k + 1) * d].abs().max()) == 0.0
        keep = torch.ones(a.shape[1], dtype=torch.bool)
        keep[k * d:(k + 1) * d] = False
        assert torch.equal(a[:, keep], b[:, keep])
    assert rep["memory_slot_columns_zeroed"] == [k * d, (k + 1) * d]


def test_without_residual_prediction_the_head_starts_at_the_old_token(tmp_path):
    torch.manual_seed(22)
    off = _lite(checkpoint_dir=str(tmp_path))
    _constant_intermind(off)
    path = off.save_checkpoint(step=5)
    on = _lite(vestibular_sense=True)
    on.load_checkpoint(path, load_optimizer=False)
    ph = off.placeholders._generators["intermind"].generate(2).clone()
    old = off.encoder_registry.encode({"intermind": ph}, {"intermind": "synthetic"})["intermind"]
    pred = on.prediction_heads(torch.randn(2, on.n_mod, on.d_model), who="test")
    assert torch.allclose(pred["vestibular"], old.squeeze(1), atol=1e-5)


def test_the_load_line_says_when_no_optimizer_was_read(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    _switched(tmp_path / "a", load_optimizer=False)
    assert any("NO OPTIMIZER STATE WAS READ (the load asked for none)" in r.getMessage()
               for r in caplog.records)
    caplog.clear()
    off, on, path = _switched(tmp_path / "b", load_optimizer=True, fill=True)
    msgs = [r.getMessage() for r in caplog.records if "vestibular sense:" in r.getMessage()]
    assert msgs and "saved Adam states carried by name" in msgs[-1] and "None of None" not in msgs[-1]
    import os
    os.remove(os.path.join(path, "optimizer.pt"))
    caplog.clear()
    again = _lite(vestibular_sense=True, overrides=RESID)
    again.load_checkpoint(path, load_optimizer=True)
    assert again.vestibular_switch["optimizer_read"] is False
    assert any("optimizer.pt is ABSENT" in r.getMessage() for r in caplog.records)


def test_a_small_intermind_checkpoint_loads_into_the_deploy_flag_set(tmp_path):
    # review b2-vestibular finding 8: the load-fix of e5fa2d070 (modules appended since the
    # checkpoint was saved start fresh instead of making the optimizer layout refuse the load),
    # held without the real checkpoint. Before that fix this load was refused: "the checkpoint's
    # optimizer groups hold [113, 123] parameters; the layout rebuilt here holds [121, 132]".
    off = _lite(checkpoint_dir=str(tmp_path))
    _fill_optimizer(off, 4)
    path = off.save_checkpoint(step=5)
    on = _lite(vestibular_sense=True, efference_heads=True)
    on.load_checkpoint(path, load_optimizer=True)
    rep = on.vestibular_switch
    n_old_enc = len(list(off.encoder_registry.encoders["intermind"].parameters()))
    n_eff = len(list(on.efference.parameters()))
    assert rep["adam_states_saved"] == len(off.params)
    assert rep["adam_states_carried"] == len(off.params) - n_old_enc
    assert rep["fresh_parameters"] == len(list(
        on.encoder_registry.encoders["vestibular"].parameters())) + n_eff
    for p in on.efference.parameters():
        assert not on.optimizer.state.get(p)
    for p in on.params:
        p.grad = torch.zeros_like(p)
    on.optimizer.step()


# ------------------------------------------------------------------------------ the episode
def test_replay_moments_from_before_the_switch_never_grade_the_new_head():
    torch.manual_seed(5)
    off = _lite()
    B = 2
    off.reset_context(B)
    off.enable_replay(B, size_per_env=12)
    H = len(off.handle_names)
    for i in range(12):
        toks = {m: torch.randn(B, 1, off.d_model) for m in off.active}
        mask = {m: "real" for m in off.active}
        # a slot-5 moment marked real, which a placeholder never was: the migration must mask it
        mask["intermind"] = "real" if i % 3 == 0 else "synthetic"
        ex = torch.zeros(B, H, 9)
        ex[..., 6] = 1.0
        off.replay.add(toks, torch.zeros(B, H, 10), {"rom_limit": 1.0, "strength_limit": 1.0},
                       source_mask=mask, executed=ex)
        off.goal_ring.append(({m: t.squeeze(1) for m, t in toks.items()}, torch.zeros(B)))
        off._goal_ring_w0.append(torch.zeros(B))
        off._goal_ring_cq.append(torch.zeros(B))
        off._goal_ring_drive.append(0.0)
        off.token_ring["intermind"].append(torch.zeros(off.d_model))
        off.token_ring["tactile"].append(torch.zeros(off.d_model))
    k = off.replay.mods.index("intermind")
    other_valid = off.replay.valid.clone()
    state = CS.capture_learner(off)
    on = _lite(vestibular_sense=True)
    on.reset_context(B)
    on.enable_replay(B, size_per_env=12)
    rep = VS.migrate_learner_state(state, on)
    assert rep["replay_slot_valid_before"] == 8   # 4 of the 12 moments x 2 bodies
    CS.restore_learner(on, state)
    assert on.replay.mods == on.active and on.replay.mods[k] == "vestibular"
    assert not bool(on.replay.valid[..., k].any())
    keep = [j for j in range(len(on.active)) if j != k]
    assert torch.equal(on.replay.valid[..., keep], other_valid[..., keep])
    assert len(on.goal_ring) == 0 and len(on._goal_ring_w0) == 0
    assert list(on.token_ring) == on.active and len(on.token_ring["tactile"]) == 12
    assert len(on.token_ring["vestibular"]) == 0
    # the replay pass trains the other heads and leaves the vestibular head exactly as it was
    head = [p.detach().clone() for p in on.prediction_heads.heads["vestibular"].parameters()]
    tact = [p.detach().clone() for p in on.prediction_heads.heads["tactile"].parameters()]
    r = on.replay_update(batch=4)
    assert r is not None and r["replay_graded_mods"] >= 1
    assert all(torch.equal(a, b) for a, b in
               zip(head, on.prediction_heads.heads["vestibular"].parameters()))
    assert not all(torch.equal(a, b) for a, b in
                   zip(tact, on.prediction_heads.heads["tactile"].parameters()))
    # a second migration of the now-vestibular companion does nothing
    assert VS.migrate_learner_state(CS.capture_learner(on), on) is None


def test_the_migrated_companion_clears_every_state_keyed_by_the_old_name():
    off = _lite()
    B = 1
    off.reset_context(B)
    off._nf_raw = {"intermind": torch.zeros(B, 5, off.d_model + 3)}
    off._nf_tokens = {m: torch.zeros(B, 1, off.d_model) for m in off.active}
    off._nf_mask = {m: "real" for m in off.active}
    off._tenure_pending = {"tokens": {"intermind": torch.zeros(B, off.d_model)}}
    off.enc_history.append({m: torch.zeros(B, off.d_model) for m in off.active})
    off.motor_history.append(torch.zeros(B, 26, 9))
    off._td_err_ema = torch.ones(B, len(off.pred_mods))
    state = CS.capture_learner(off, include_replay=False)
    on = _lite(vestibular_sense=True)
    on.reset_context(B)
    rep = VS.migrate_learner_state(state, on)
    assert "_nf_raw" in rep["cleared"] and "_tenure_pending" in rep["cleared"]
    CS.restore_learner(on, state)
    assert on._nf_raw is None and on._tenure_pending is None
    assert len(on.enc_history) == 0 and len(on.motor_history) == 0
    kp = on.pred_mods.index("vestibular")
    assert float(on._td_err_ema[0, kp]) == 0.0 and float(on._td_err_ema[0, 0]) == 1.0
    assert "intermind" not in state["placeholders"]


class _ChainModel:
    """The causal-chain recorder's view of a model with the vestibular slot: tokens are constants
    per sense, the prediction for depth k is now + k."""

    def __init__(self, vest_mask):
        from lem.model.wellness import APPROXIMATE_TERMS
        self._transition_epoch = 0
        self._last_source_mask = {"vision": "real", "vestibular": vest_mask, "motor": "real"}
        self.last_wellness_grounded = {n: torch.full((2, 1), 0.5) for n in APPROXIMATE_TERMS}
        self.last_plan_record = self.last_intention = None
        self._causal_unsmoothed = self._last_smooth_prev = None
        self.set(0.0)

    def set(self, v):
        self.last_tokens = {m: torch.full((2, 4), float(v)) for m in ("vision", "vestibular", "motor")}

    def predict_before_execution(self, send_motor, gaze_out, depth=1):
        now = {m: v.clone() for m, v in self.last_tokens.items() if m != "motor"}
        return {"depths": {k: {"tokens": {m: v + k for m, v in now.items()},
                               "judge": torch.full((2, 5), 0.5)} for k in sorted({1, depth})},
                "tokens_now": now, "judge_now": torch.full((2, 5), 0.5),
                "epoch": 0, "B": 2}


@pytest.mark.parametrize("mask", ["real", "stale"])
def test_the_causal_chain_grades_the_sense_exactly_when_it_is_real(tmp_path, mask):
    from mimo_bridge.causal_chain import CausalChainRecorder
    m = _ChainModel(mask)
    r = CausalChainRecorder(str(tmp_path), hold=2, strict=True)
    r.on_latch(m, 10, latched=True, order_seq=1, send_motor=np.zeros((2, 4, 9), np.float32),
               gaze_out=None, source="reflex")
    m.set(1.0)
    r.after_step(m, 11)
    r.after_step(m, 12)
    w = r.window_summary()
    assert ("vestibular" in w["pred_err_next"]) == (mask == "real")
    assert "vision" in w["pred_err_next"]


# ------------------------------------------------------------------------------ the stats block
def test_the_angles_read_the_conventions_they_claim():
    g = np.asarray(VS.HEAD_GRAVITY_REST)
    assert VS.head_tilt_deg(g[None] * 1.0)[0] == pytest.approx(0.0, abs=1e-6)
    assert VS.head_tilt_deg(-g[None])[0] == pytest.approx(180.0, abs=1e-4)
    along, front, side = (np.asarray(a) for a in VS.CHEST_AXES)
    roll, pitch = VS.trunk_roll_pitch_deg(np.stack([front, side, -front, along]))
    assert roll[0] == pytest.approx(0.0, abs=1e-6) and abs(roll[2]) == pytest.approx(180.0, abs=1e-6)
    assert roll[1] == pytest.approx(90.0, abs=1e-6)
    assert pitch[3] == pytest.approx(90.0, abs=1e-4) and pitch[0] == pytest.approx(0.0, abs=1e-4)


def test_the_window_reports_only_real_reads_and_resets():
    w = VS.VestibularWindow()
    up_back = np.asarray(VS.CHEST_AXES[1])
    good = _vec(support_trunk=0.8, limit_load=0.05)
    good[0:3] = np.asarray(VS.HEAD_GRAVITY_REST)
    good[15:18] = up_back
    legacy = np.array([9.81, 0, 0, 0, 0, 0], np.float32)
    model = SimpleNamespace(_last_source_mask={"vestibular": "real"},
                            last_prediction_audit={"vestibular": {"transitions": 2,
                                                                  "prediction_mse": 0.02,
                                                                  "persistence_mse": 0.04}})
    w.add([SimpleNamespace(vestibular=good), SimpleNamespace(vestibular=good)], model)
    model._last_source_mask = {"vestibular": "stale"}
    w.add([SimpleNamespace(vestibular=good), SimpleNamespace(vestibular=legacy)], model)
    s = w.summary()
    assert s["steps"] == 2 and s["real_steps"] == 1 and s["frames_invalid"] == 1
    assert s["pred_err"] == pytest.approx(0.02) and s["improvement_over_persistence"] == pytest.approx(0.5)
    assert s["head_tilt_deg"] == 0.0 and s["trunk_roll_deg"] == 0.0 and s["trunk_pitch_deg"] == 0.0
    assert s["per_body"]["head_tilt_deg"] == [0.0, 0.0]
    assert s["per_body"]["trunk_roll_deg"] == [0.0, 0.0]
    assert s["support_trunk"] == pytest.approx(0.8) and s["clipped_share"] == 0.0
    # two bodies rolled to opposite sides: the scalar says "on a side", the signs stay per body
    w2 = VS.VestibularWindow()
    left, right = good.copy(), good.copy()
    left[15:18] = np.asarray(VS.CHEST_AXES[2])
    right[15:18] = -np.asarray(VS.CHEST_AXES[2])
    model._last_source_mask = {"vestibular": "real"}
    w2.add([SimpleNamespace(vestibular=left), SimpleNamespace(vestibular=right)], model)
    s2 = w2.summary()
    assert s2["trunk_roll_deg"] == pytest.approx(90.0, abs=0.1)
    assert s2["per_body"]["trunk_roll_deg"] == [pytest.approx(90.0, abs=0.1), pytest.approx(-90.0, abs=0.1)]
    assert set(s["fields_mean"]) == set(VS.VESTIBULAR_FIELDS)
    assert set(VS.VestibularWindow.SCALE) <= set(s)
    empty = w.summary()
    assert empty["steps"] == 0 and empty["pred_err"] is None and empty["fields_mean"] is None


def test_the_window_reports_the_encoders_liveness_and_the_normaliser():
    m = _lite(vestibular_sense=True)
    w = VS.VestibularWindow()
    rows = _rows(6, 3)
    for i in range(3):
        fr = _bare_frames(rows[2 * i:2 * i + 2])
        m._batch_frames(fr)
        m.vestibular_norm.commit()
        w.add(fr, SimpleNamespace(_last_source_mask={"vestibular": "real"}, active=m.active,
                                  encoder_registry=m.encoder_registry,
                                  vestibular_norm=m.vestibular_norm))
    s = w.summary()
    names = [f"{VS.ENC_NEW}mlp.{i}" for i, mod in
             enumerate(m.encoder_registry.encoders["vestibular"].mlp)
             if isinstance(mod, torch.nn.Linear)][:-1]
    assert list(s["liveness"]) == names
    assert all(0.0 <= v <= 1.0 for v in s["liveness"].values())
    assert s["pool_spread"] == {}
    assert s["input_real_body_moments"] == 6 and 0.0 < s["input_centre_filled_share"] < 1.0
    assert w.summary()["liveness"] is None


# ------------------------------------------------------------------------------ live runs
LIVE_OV = {"model": {"residual_prediction": True}, "training": {"steps_per_phase": 3000}}


def _live_lite(**kw):
    """The small model on the reach scene's own body (the legacy 24-handle layout)."""
    torch.set_num_threads(1)
    return LemLite(device="cpu", seed=0, intention=False, imagination=False, no_vision=True,
                   next_frame_loss=True, top_down_gate=True, overrides=LIVE_OV, **kw)


def _reach(tmp_path, tag):
    from mimo_bridge.datasource import SingleEnvSource
    return SingleEnvSource(scene="reach", run_dir=str(tmp_path / tag), tag="_" + tag, vision=False,
                           seed=5)


def _live_steps(m, src, n, train=True, first=None, record=None):
    frames = src.reset()
    if first is not None:
        frames[0].vestibular = first
    dev = dict(m.dev_limits(10 ** 9))
    dev["exploration_noise"] = 0.05
    held = prev = None
    out = []
    for i in range(n):
        r = m.step(frames, dev, train=train, step_index=i, executed_action=prev)
        out.append(r)
        if record is not None:
            record(i, m, frames)
        if i % 5 == 0 or held is None:
            held = r.motor_for_action
        for _ in range(5):
            frames = src.step(held, r.gaze_out, None, render=False)
        prev = held
    return out


@pytest.mark.slow
def test_the_lesson_weighs_the_sense_as_declared_and_only_from_a_real_moment(tmp_path):
    # review b2-vestibular findings 2 and 7. The first frame is the padded legacy vector (stale),
    # so the moment after it is NOT graded on this sense; from then on the next-moment lesson is
    # the other senses' errors plus the declared weight times this sense's error.
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False, next_frame_loss=True,
                vestibular_sense=True, vestibular_lesson_weight=0.25,
                overrides={"training": {"steps_per_phase": 3000}})
    src = _reach(tmp_path, "lw")
    try:
        res = _live_steps(m, src, 5, first=np.array([9.81, 0, 0, 0, 0, 0], np.float32))
    finally:
        src.close()
    assert res[0].source_mask["vestibular"] == "stale"
    assert res[1].source_mask["vestibular"] == "real"
    assert "vestibular" not in res[1].next_pe_per_modality, "graded from the stale moment"
    assert "tactile" in res[1].next_pe_per_modality
    for r in res[2:]:
        pe = r.next_pe_per_modality
        assert "vestibular" in pe
        want = sum(v for k, v in pe.items() if k != "vestibular") + 0.25 * pe["vestibular"]
        assert r.breakdown["next_frame"] == pytest.approx(want, rel=1e-5, abs=1e-7)


@pytest.mark.slow
def test_the_switch_changes_no_order_the_body_is_given(tmp_path):
    # review b2-vestibular finding 1: flag off and flag on from the same intermind checkpoint, the
    # same frames and random streams, learning off: every order, the gaze and the wellness terms
    # the planner ranks by are the old model's (before the fix: 11.7 in every 100 parts of the
    # order's size at the first moment)
    torch.manual_seed(31)
    off = _live_lite(checkpoint_dir=str(tmp_path / "ck"), persistent_memory="on")
    _constant_intermind(off)
    path = off.save_checkpoint(step=5)
    a = _live_lite(persistent_memory="on")
    a.load_checkpoint(path, load_optimizer=False)
    b = _live_lite(vestibular_sense=True, persistent_memory="on")
    b.load_checkpoint(path, load_optimizer=False)
    src = _reach(tmp_path, "fp")
    try:
        frames = src.reset()
        dev = dict(a.dev_limits(10 ** 9))
        dev["exploration_noise"] = 0.05
        held = prev = None
        for i in range(12):
            s0 = torch.get_rng_state().clone()
            ra = a.step(frames, dev, train=False, step_index=i, executed_action=prev)
            torch.set_rng_state(s0)
            rb = b.step(frames, dev, train=False, step_index=i, executed_action=prev)
            assert np.allclose(ra.motor_for_action, rb.motor_for_action, rtol=1e-4, atol=1e-5), i
            assert np.allclose(ra.gaze_out, rb.gaze_out, atol=1e-5), i
            assert ra.wellness["prediction_error"] == pytest.approx(
                rb.wellness["prediction_error"], abs=1e-5), i
            if i % 5 == 0 or held is None:
                held = ra.motor_for_action
            for _ in range(5):
                frames = src.step(held, ra.gaze_out, None, render=False)
            prev = held
    finally:
        src.close()


@pytest.mark.slow
def test_the_encoder_is_live_and_the_switched_token_starts_to_vary_after_100_steps(tmp_path):
    # DESIGN section 10 rules 2 and 4 (review finding 9): the encoder's hidden layers are live on
    # the body's own vectors at the start and after 100 training steps, and the token that the
    # switch started at the old constant has begun to vary between moments.
    torch.manual_seed(41)
    off = _live_lite(checkpoint_dir=str(tmp_path / "ck"))
    _constant_intermind(off)
    path = off.save_checkpoint(step=5)
    m = _live_lite(vestibular_sense=True)
    m.load_checkpoint(path, load_optimizer=False)
    w = VS.VestibularWindow()
    toks = []

    def rec(i, model, frames):
        w.add(frames, model)
        toks.append(model.last_tokens["vestibular"].detach().clone())
    src = _reach(tmp_path, "lv")
    try:
        _live_steps(m, src, 10, record=rec)
        first = w.summary()
        _live_steps(m, src, 100, record=rec)
        last = w.summary()
    finally:
        src.close()
    assert first["liveness"] and all(v > 0.5 for v in first["liveness"].values()), first["liveness"]
    assert all(v > 0.5 for v in last["liveness"].values()), last["liveness"]
    ph = off.placeholders._generators["intermind"].generate(1).clone()
    old = off.encoder_registry.encode({"intermind": ph}, {"intermind": "synthetic"})["intermind"]
    assert torch.allclose(toks[0], old.squeeze(1), atol=1e-5), "the switch starts at the old token"
    late = torch.stack(toks[-20:]).squeeze(1)
    assert float((late - late.mean(0)).norm(dim=-1).mean()) > 1e-3, "the token never began to vary"
    assert last["input_real_body_moments"] >= 100


def test_the_window_reports_the_senses_share_of_the_lesson_at_its_weight():
    w = VS.VestibularWindow()
    aud = {"vestibular": {"transitions": 2, "prediction_mse": 0.3, "persistence_mse": 0.4},
           "tactile": {"transitions": 2, "prediction_mse": 0.1, "persistence_mse": 0.1},
           "vision": {"transitions": 2, "prediction_mse": 0.2, "persistence_mse": 0.1}}
    model = SimpleNamespace(_last_source_mask={"vestibular": "real"}, last_prediction_audit=aud,
                            vestibular_lesson_weight=0.5)
    w.add([SimpleNamespace(vestibular=_vec())], model)
    w.add([SimpleNamespace(vestibular=_vec())], model)
    assert w.summary()["lesson_share"] == pytest.approx(0.15 / (0.15 + 0.3), abs=1e-4)
