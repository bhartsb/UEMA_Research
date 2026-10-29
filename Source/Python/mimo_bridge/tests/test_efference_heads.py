# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The efference heads, version 2 (docs/gain_of_function_v3/DESIGN.md D5, --efference-heads; the
review of the first version is docs/gain_of_function_v3/reviews/b1-causal-efference.md).

What is pinned here: the units (centimetres and degrees, by construction and by measured divisors),
the asked change (target minus achieved pose), the centring of the latent and its running mean, the
zero-initialised all-token path, the do-nothing baseline, the gradient path (the next-moment gradings
reach the core, the longer horizons only the heads, the vestigial GazeHead nothing), the loss as a
SUM of horizons with the part that reaches the core reported, the tenure expectation formed where the
order began, the shuffled controls changing only the order, the registration rules (appended last,
own random stream, saved as a side module, stamped), the checkpoint rules (divisors migrated exactly,
a different order length resets the tenure rows, another version starts fresh and keeps every other
Adam moment) and the window sums staying out of the episode companion."""
import json
import math
from types import SimpleNamespace

import pytest
import torch

from lem.model import efference as EF
from lem.model.efference import (EFFERENCE_SCALES, EfferenceHeads, asked_change, pose_change,
                                 rotation_change_deg, scaled_error)
from mimo_bridge.lem_model import LemLite


def _quat_z(deg):
    a = math.radians(deg)
    return torch.tensor([0.0, 0.0, math.sin(a / 2), math.cos(a / 2)])


def test_rotation_change_is_the_turn_in_degrees_and_ignores_the_quaternion_sign():
    q0 = torch.tensor([0.0, 0.0, 0.0, 1.0])
    for deg in (0.0, 1.0, 30.0, 90.0, 179.0):
        v = rotation_change_deg(q0, _quat_z(deg))
        assert torch.allclose(v, torch.tensor([0.0, 0.0, deg]), atol=1e-3), (deg, v)
    # q and -q are one orientation: the turn is the same short one
    assert torch.allclose(rotation_change_deg(q0, -_quat_z(40.0)), torch.tensor([0.0, 0.0, 40.0]),
                          atol=1e-3)
    # a turn from 10 to 50 degrees about z is 40 degrees
    assert torch.allclose(rotation_change_deg(_quat_z(10.0), _quat_z(50.0)),
                          torch.tensor([0.0, 0.0, 40.0]), atol=1e-3)


def test_pose_change_reads_positions_in_the_tokenizers_200_cm_unit():
    a0 = torch.zeros(1, 2, 10)
    a0[..., 6] = 1.0
    a1 = a0.clone()
    a1[0, 0, 0] = 0.01           # 0.01 x 200 cm = 2 cm along x
    a1[0, 1, 3:7] = _quat_z(90.0)
    cm, deg = pose_change(a0, a1)
    assert torch.allclose(cm[0, 0], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.allclose(deg[0, 1], torch.tensor([0.0, 0.0, 90.0]), atol=1e-3)


def test_every_divisor_is_measured_and_positive():
    assert set(EFFERENCE_SCALES) == {"pos_next_cm", "rot_next_deg", "pos_tenure_cm",
                                     "rot_tenure_deg", "gaze_deg_1", "gaze_deg_2", "gaze_deg_3"}
    assert all(v > 0 for v in EFFERENCE_SCALES.values())
    # a 250 ms order moves further than a 50 ms moment, and the eyes further at 150 ms than 50
    assert EFFERENCE_SCALES["pos_tenure_cm"] > EFFERENCE_SCALES["pos_next_cm"]
    assert EFFERENCE_SCALES["gaze_deg_3"] > EFFERENCE_SCALES["gaze_deg_2"] > EFFERENCE_SCALES["gaze_deg_1"]
    # the asked change is read in the tenure divisors (the lane's rule)
    assert EF.ASK_POS_CM == EFFERENCE_SCALES["pos_tenure_cm"]
    assert EF.ASK_ROT_DEG == EFFERENCE_SCALES["rot_tenure_deg"]


def test_the_asked_change_is_target_minus_achieved_in_the_tenure_divisors():
    """Review MAJOR 2: the body head is told what each handle is ASKED to do relative to where it
    is. A handle at x = 10 cm told to go to x = 13.6 cm reads 3.6 / 1.80 = 2 units; a handle
    turned 11 degrees reads 1 unit; strength passes; the eyes' row reads zero."""
    H = 26
    ach = torch.zeros(1, H, 10)
    ach[..., 6] = 1.0
    ach[0, 0, 0] = 10.0 / 200.0
    order = torch.zeros(1, H, 9)
    order[..., 6] = 1.0
    order[0, 0, 0] = 13.6 / 200.0                 # encoder units: cm / 200
    order[0, 1, 3:7] = _quat_z(11.0)
    order[..., 8] = 0.4
    a = asked_change(order, ach, gaze_idx=23)
    assert a.shape == (1, H, EF.ASK_F)
    assert torch.allclose(a[0, 0, 0:3], torch.tensor([2.0, 0.0, 0.0]), atol=1e-4)
    assert torch.allclose(a[0, 1, 3:6], torch.tensor([0.0, 0.0, 1.0]), atol=1e-4)
    assert torch.allclose(a[0, 5, 6], torch.tensor(0.4))
    assert float(a[0, 23].abs().sum()) == 0.0


def _heads(d=32, n_mod=6, vision_idx=0, seed=0):
    return EF.make_heads(d, 26, 23, 12, n_mod, vision_idx, seed)


def _pose(B, H=26, dx_cm=0.0):
    p = torch.zeros(B, H, 10)
    p[..., 6] = 1.0
    p[..., 0] = dx_cm / 200.0
    return p


def test_heads_shapes_lived_units_and_the_zero_initialised_paths():
    h = _heads()
    lat = torch.randn(3, 6, 32)
    order = torch.randn(3, 26, 9)
    ach = _pose(3)
    h.ensure_centres(lat, order)
    b, g = h.body(lat, order, ach), h.gaze(lat, order, ach)
    assert b.shape == (3, 26, 12) and g.shape == (3, 3, 2)
    lv = EfferenceHeads.to_lived(b, g)
    assert torch.allclose(lv["body_next_cm"], b[..., 0:3] * EFFERENCE_SCALES["pos_next_cm"])
    assert torch.allclose(lv["body_tenure_deg"], b[..., 9:12] * EFFERENCE_SCALES["rot_tenure_deg"])
    assert torch.allclose(lv["gaze_deg"][:, 2], g[:, 2] * EFFERENCE_SCALES["gaze_deg_3"])
    # the all-token paths, the head-motion columns and the gaze copy start at exactly zero
    for lin in (h.body_all, h.gaze_all, h.vision_out):
        assert float(lin.weight.abs().sum()) == 0.0 and float(lin.bias.abs().sum()) == 0.0
    assert float(h.gaze_motion.weight.abs().sum()) == 0.0
    assert float(h.vision_delta(lat, order, ach).abs().sum()) == 0.0
    # head motion through zero columns changes nothing at the start
    assert torch.equal(h.gaze(lat, order, ach, torch.randn(3, 6)), g)


def test_the_latent_is_centred_by_a_running_mean_over_lived_training_moments():
    """Review MAJOR 1 / DESIGN section 10 rule 1: a part every moment shares is subtracted before
    the LayerNorm. The first moment sets the centre; later ones move it at 1/2000 per moment
    (corrected for starting at zero)."""
    h = _heads()
    shared = torch.randn(1, 6, 32) * 50.0
    x1 = shared + torch.randn(64, 6, 32)
    h.ensure_centres(x1)
    assert torch.allclose(h.centre_mean, x1.mean(0), atol=1e-5)
    w1 = h.centre_weight
    x2 = shared + torch.randn(64, 6, 32)
    h.update_centres(x2)
    a = 1.0 / EF.CENTRE_HORIZON_STEPS
    w2 = (1 - a) * w1 + a
    expect = (x1.mean(0) * (1 - a) * w1 + x2.mean(0) * a) / w2
    assert torch.allclose(h.centre_mean, expect, atol=1e-4)
    assert abs(h.centre_weight - w2) < 1e-12

    # the normalised tokens no longer carry the shared part (per token: the length of the
    # batch's mean against the mean length; 1 = every sample the same, near 0 = nothing shared)
    def shared_share(n):
        return float(n.mean(0).norm(dim=-1).mean()) / float(n.norm(dim=-1).mean())
    x3 = shared + torch.randn(64, 6, 32)
    assert shared_share(h.normalise(x3)) < 0.5
    # the control: a LayerNorm alone keeps the shared direction
    assert shared_share(h.lat_norm(x3)) > 0.9
    # the running means round-trip through runtime_state.pt's dict
    other = _heads(seed=1)
    assert other.load_runtime_state(h.runtime_state())
    assert torch.equal(other.centre_mean, h.centre_mean) and other.centre_weight == h.centre_weight
    assert not other.load_runtime_state({"centre_mean": torch.zeros(3)})


def test_the_heads_report_liveness_pool_spread_and_the_shared_share():
    h = _heads()
    lat, order, ach = torch.randn(5, 6, 32), torch.randn(5, 26, 9), _pose(5)
    h.ensure_centres(lat, order)
    h.body(lat, order, ach, record=True)
    h.gaze(lat, order, ach, record=True)
    h.vision_delta(lat, order, ach, record=True)
    rep = h.window_liveness()
    assert set(rep["liveness"]) == set(EfferenceHeads.LIVE_LAYERS)
    assert set(rep["pool_spread"]) == set(EfferenceHeads.POOLS)
    assert all(0.0 <= v <= 1.0 for v in rep["pool_spread"].values())
    assert set(rep["input_shared_share"]) == {"latent", "order"}
    assert h.window_liveness() == {"liveness": {}, "liveness_varying": {}, "pool_spread": {}, "input_shared_share": {}}


def test_a_hidden_unit_held_at_one_active_value_reads_live_but_not_varying():
    """Fails on the heads' gauge before 2026-09-26, whose only rule was the -3 line (the bug class of production's
    liveness gauge, which read H84's vision and touch prediction heads 100 of 100 live while 2 and 0.5 of 100 varied
    at 415,025): a unit whose input sits above -3 on every moment reads live whatever it passes forward."""
    h = _heads()
    with torch.no_grad():
        h.vision_in.weight.zero_()
        h.vision_in.bias.fill_(5.0)                  # every gaze-copy hidden unit held at GELU(5) on every moment
    lat, order, ach = torch.randn(16, 6, 32), torch.randn(16, 26, 9), _pose(16)
    h.ensure_centres(lat, order)
    h.body(lat, order, ach, record=True)
    h.vision_delta(lat, order, ach, record=True)
    rep = h.window_liveness()
    assert rep["liveness"]["vision_in"] == 1.0, "the -3 line reads every held unit live"
    assert rep["liveness_varying"]["vision_in"] == 0.0, "none of them varies"
    assert rep["liveness_varying"]["body_mlp.0"] > 0.9, "the moving layer's units vary"


def test_pool_spread_is_per_head_so_four_heads_on_four_keys_read_as_locked():
    h = _heads()
    w = torch.zeros(1, 4, 1, 6)
    for k in range(4):
        w[0, k, 0, k] = 1.0                      # each head reads ONE different key
    h._note_spread("x", w)
    assert h.window_liveness()["pool_spread"]["x"] < 1e-6
    h._note_spread("y", torch.full((1, 4, 1, 6), 1.0 / 6))
    assert abs(h.window_liveness()["pool_spread"]["y"] - 1.0) < 1e-6


def _lite(**kw):
    torch.set_num_threads(1)
    return LemLite(device="cpu", seed=0, imagination=False, no_vision=True,
                   next_frame_loss=True, body_layout="trunk_v1", **kw)


def test_flag_off_builds_nothing_and_flag_on_appends_last_in_its_own_stream():
    s0 = torch.get_rng_state().clone()
    off = _lite()
    after_off = torch.get_rng_state().clone()
    torch.set_rng_state(s0)
    on = _lite(efference_heads=True)
    assert torch.equal(after_off, torch.get_rng_state()), "the heads drew from the run's stream"
    assert off.efference is None and "efference_heads" not in off._side_modules()
    assert "efference_heads" not in off.receipt
    n_eff = sum(p.numel() for p in on.efference.parameters())
    assert on.n_params == off.n_params + n_eff
    # appended LAST: the old parameters keep their order, so the optimizer migrates by name
    assert [p.shape for p in on.params[:len(off.params)]] == [p.shape for p in off.params]
    for pa, pb in zip(off.params, on.params):
        assert torch.equal(pa, pb), "a build with the heads changed an existing initial weight"
    eff_ids = {id(p) for p in on.efference.parameters()}
    assert all(id(p) in eff_ids for p in on.params[len(off.params):])
    assert on._side_modules()["efference_heads"] is on.efference
    st = on.receipt["efference_heads"]
    assert st["version"] == "efference_v2" and st["scales"] == EFFERENCE_SCALES
    assert st["weight"] == 0.1 and st["tenure_steps"] == 5 and st["gaze_head_repurposed"] is False
    # the gaze copy is a plain attribute of the prediction heads, never a registered submodule
    assert set(on.prediction_heads.state_dict()) == set(off.prediction_heads.state_dict())
    with pytest.raises(ValueError):
        LemLite(device="cpu", seed=0, imagination=False, no_vision=True, efference_heads=True)


def _frames(B, az_deg, el_deg=0.0):
    a, e = math.radians(az_deg), math.radians(el_deg)
    return [SimpleNamespace(eye_angles=((a, e), (a, e))) for _ in range(B)]


def _latent(m, B):
    toks = {mm: torch.randn(B, 1, m.d_model) for mm in m.active}
    ctx = torch.zeros(B, (m.ctx_len - 1) * m.n_mod, m.d_model)
    _, last, _ = m.forward_model(toks, ctx)
    return last


def _order(B, H):
    exec_t = torch.zeros(B, H, 9)
    exec_t[..., 6] = 1.0
    exec_t[..., 8] = 0.5
    return exec_t


def test_next_moment_gradings_reach_the_core_the_longer_horizons_do_not_and_the_GazeHead_nothing():
    torch.manual_seed(1)
    m = _lite(efference_heads=True)
    B, H = 2, len(m.handle_names)
    exec_t = _order(B, H)
    m._nf_ach = _pose(B, H, 0.0)
    m._eff_eye_prev = torch.zeros(B, 2)
    out = m._efference_lesson(_latent(m, B), exec_t, 1, _pose(B, H, 1.0), _frames(B, 2.0), 10, B)
    assert set(out) == {"efference_body", "efference_gaze"}
    sum(out.values()).backward()
    core = [p for p in m.forward_model.parameters() if p.grad is not None
            and float(p.grad.abs().sum()) > 0]
    assert core, "the next-moment gradings must reach the forward model"
    assert m.efference.gaze_out.weight.grad is not None
    assert all(p.grad is None for p in m.gaze_head.parameters()), \
        "efference_v2 never trains the vestigial GazeHead"
    # the window's do-nothing gauges are the size of the change: 1 cm for every body handle,
    # 2 degrees of azimuth for the eyes; one step graded only the core-reaching terms
    w = m.efference_window_summary()
    assert abs(w["body_cm_do_nothing_next"] - 1.0) < 1e-4
    assert abs(w["gaze_deg_do_nothing"][0] - 2.0) < 1e-3
    assert w["body_cm_shuffled_next"] is not None and w["events_next"] == 1
    assert w["gaze_deg_err"][1] is None      # nothing stored yet two steps back
    assert abs(w["loss_body_core"] - w["loss_body"]) < 1e-6
    assert abs(w["loss_gaze_core"] - w["loss_gaze"]) < 1e-6
    assert w["version"] == "efference_v2" and set(w["pool_spread"]) == {"body_pool", "gaze_pool"}

    # the stored latents (100/150 ms and the tenure) are graded with the core cut off
    m.forward_model.zero_grad(set_to_none=True)
    lat = _latent(m, B).detach()             # steps where no latent carries the core's graph
    for s in range(11, 18):
        oid = 1 if s < 15 else 2             # order 1 began at state 9; order 2 appears at 15
        m._nf_ach = _pose(B, H, float(s))
        out = m._efference_lesson(lat if s < 17 else None, exec_t, oid, _pose(B, H, 1.0 + s),
                                  _frames(B, 2.0 * s), s, B)
        if out:
            sum(out.values()).backward()
    w = m.efference_window_summary()
    assert w["gaze_deg_err"][1] is not None and w["gaze_deg_err"][2] is not None
    assert all(p.grad is None for p in m.forward_model.parameters())


def test_the_loss_sums_the_horizons_and_reports_the_part_that_reaches_the_core():
    """Review MINOR 4: the next-moment term's weight does not shrink on steps where the longer
    horizons are graded too."""
    torch.manual_seed(5)
    m = _lite(efference_heads=True)
    B, H = 2, len(m.handle_names)
    exec_t = _order(B, H)
    lat = _latent(m, B).detach()
    rows = []
    for s in range(20, 24):
        m._nf_ach = _pose(B, H, float(s))
        with torch.no_grad():
            m._efference_lesson(lat, exec_t, 1, _pose(B, H, s + 1.0), _frames(B, 3.0 * s), s, B)
        rows.append(m.efference_window_summary())
    last = rows[-1]                             # 50, 100 and 150 ms all graded
    assert last["gaze_deg_err"][2] is not None
    assert last["loss_gaze"] > last["loss_gaze_core"] > 0
    w = m.efference_weight
    g1 = last["loss_gaze_core"] / w
    assert g1 > 0 and last["loss_gaze"] / w >= g1


def test_the_tenure_expectation_is_formed_where_the_order_began():
    """Review MINOR 1: the expectation the controllability pair reads is the tenure output computed
    where the order began, with the weights of that moment, not a re-forward at the order's end."""
    torch.manual_seed(2)
    m = _lite(efference_heads=True)
    B, H = 2, len(m.handle_names)
    exec_t = _order(B, H)
    ids = [1, 1, 1, 2, 2, 2, 2, 2, 3]     # order 2 is in force for 5 steps; order 1 for 3
    lat = _latent(m, B).detach()
    for i, oid in enumerate(ids):
        m._nf_ach = _pose(B, H, float(i))
        with torch.no_grad():
            m._efference_lesson(lat, exec_t, oid, _pose(B, H, float(i + 1)),
                                _frames(B, 0.0), 100 + i, B)
            if i == 3:                       # order 2 has just begun: its expectation is stored
                began = m._eff_tenure["expected"].clone()
                with torch.no_grad():        # the heads change after it began
                    for p in m.efference.body_mlp[-1].parameters():
                        p.add_(0.5)
    w = m.efference_window_summary()
    assert w["events_latch"] == 1, "only the 5-step order counts as a tenure"
    # order 2 began at pose 3 and ended at pose 8: 5 cm on every body handle
    assert abs(w["body_cm_do_nothing_latch"] - 5.0) < 1e-3
    t = m.last_efference_tenure
    assert t["order_began"] == 102 and t["step"] == 108
    assert torch.allclose(t["expected_cm"], began[..., 0:3] * EFFERENCE_SCALES["pos_tenure_cm"])
    assert torch.allclose(t["expected_deg"], began[..., 3:6] * EFFERENCE_SCALES["rot_tenure_deg"])
    assert t["expected_shuffled_cm"].shape == (B, H, 3)
    assert torch.allclose(t["achieved_cm"][0, 0], torch.tensor([5.0, 0.0, 0.0]), atol=1e-4)
    m.reset_transition_state()
    assert m.last_efference_tenure is None


def test_the_shuffled_controls_change_only_the_order():
    """Review MINOR 9: the shuffled gauges hand the heads ANOTHER body's order with THIS body's
    latent and pose. Since 2026-10-02 there are two such controls on every graded model step: the
    absolute roll (another body's order as it stands) and the size-matched order (another body's
    order as a change from its own pose applied to this body's pose, the eyes' row as its asked eye
    turn applied to this body's eyes: size_matched_other_order)."""
    from mimo_bridge.lem_model import command_for_encoder
    torch.manual_seed(3)
    m = _lite(efference_heads=True)
    B, H = 3, len(m.handle_names)
    exec_t = _order(B, H)
    exec_t[:, :, 0] = torch.arange(B).view(B, 1).float()      # each body's order differs
    calls = []
    real_body, real_gaze = m.efference.body, m.efference.gaze

    def body(latent, order, achieved, record=False):
        calls.append(("body", latent.detach().clone(), order.clone(), achieved.clone(), record))
        return real_body(latent, order, achieved, record=record)

    def gaze(latent, order, achieved, head_motion=None, record=False):
        calls.append(("gaze", latent.detach().clone(), order.clone(), achieved.clone(), record))
        return real_gaze(latent, order, achieved, head_motion, record=record)
    m.efference.body, m.efference.gaze = body, gaze
    ach = _pose(B, H, 0.0)
    ach[..., 0] = torch.arange(B).view(B, 1).float() * 0.01     # the bodies stand 2 cm apart
    m._nf_ach = ach
    m._eff_eye_prev = torch.tensor([[0.0, 0.0], [5.0, -2.0], [10.0, 3.0]])
    sized = command_for_encoder(EF.size_matched_other_order(exec_t, ach, m.GAZE_IDX,
                                                            eye_deg=m._eff_eye_prev,
                                                            convention=m.eye_row_convention))
    lat = _latent(m, B)
    m._efference_lesson(lat, exec_t, 1, _pose(B, H, 1.0), _frames(B, 1.0), 10, B)
    for kind in ("body", "gaze"):
        real = [c for c in calls if c[0] == kind and c[4]]
        ctrl = [c for c in calls if c[0] == kind and not c[4]]
        assert len(real) == 1 and len(ctrl) == 2, kind
        (_, l0, o0, a0, _) = real[0]
        for (_, l1, o1, a1, _) in ctrl:
            assert torch.equal(l0, l1) and torch.equal(a0, a1), kind + ": latent or pose changed"
        assert torch.equal(ctrl[0][2], o0.roll(1, dims=0)) and not torch.equal(o0, ctrl[0][2]), kind
        assert torch.equal(ctrl[1][2], sized) and not torch.equal(sized, o0.roll(1, dims=0)), kind


def test_a_reset_forgets_everything_that_pairs_across_it_but_not_the_window_or_the_centre():
    m = _lite(efference_heads=True)
    B = 1
    m._eff_hist.append({"latent": torch.zeros(1), "order": None, "eye": None, "step": 1})
    m._eff_tenure = {"x": 1}
    m._eff_eye_prev = torch.zeros(B, 2)
    m.efference_window = {"body_cm_err_next": [1.0, 1]}
    m.efference.centre_weight = 0.5
    e0 = m._transition_epoch
    m.reset_transition_state()
    assert not m._eff_hist and m._eff_tenure is None and m._eff_eye_prev is None
    assert m._transition_epoch == e0 + 1
    assert m.efference_window, "window sums are a count of what was measured, kept across a reset"
    assert m.efference.centre_weight == 0.5, "the running mean is over many histories, kept"


def test_a_divisor_change_migrates_the_heads_exactly(tmp_path):
    torch.manual_seed(3)
    m = _lite(efference_heads=True)
    B, H = 2, len(m.handle_names)
    lat, order, ach = torch.randn(B, m.n_mod, m.d_model), torch.randn(B, H, 9), _pose(B, H)
    m.efference.ensure_centres(lat, order)
    old = {k: v * (1.5 if "pos" in k else 0.5) for k, v in EFFERENCE_SCALES.items()}
    with torch.no_grad():
        before = EfferenceHeads.to_lived(m.efference.body(lat, order, ach),
                                         m.efference.gaze(lat, order, ach), scales=old)
    (tmp_path / "model_receipt.json").write_text(json.dumps(
        {"efference_heads": {"version": "efference_v2", "scales": old, "tenure_steps": 5,
                             "weight": 0.1}}))
    m._migrate_efference_scales(str(tmp_path), load_optimizer=False)
    with torch.no_grad():
        after = EfferenceHeads.to_lived(m.efference.body(lat, order, ach),
                                        m.efference.gaze(lat, order, ach))
    for k in before:
        assert torch.allclose(before[k], after[k], atol=1e-5, rtol=1e-5), k
    assert m.efference_load["scales_migrated"] == old


def test_an_order_length_change_resets_only_the_tenure_rows(tmp_path):
    """Review MINOR 3: the stamped tenure length is checked at load."""
    torch.manual_seed(4)
    m = _lite(efference_heads=True)
    last = m.efference.body_mlp[-1]
    with torch.no_grad():
        last.weight.add_(1.0)
        last.bias.add_(1.0)
    for p in (last.weight, last.bias):
        m.optimizer.state[p] = {"step": torch.tensor(3.0), "exp_avg": torch.ones_like(p),
                                "exp_avg_sq": torch.ones_like(p)}
    trained = last.weight.detach().clone()
    (tmp_path / "model_receipt.json").write_text(json.dumps(
        {"efference_heads": {"version": "efference_v2", "scales": EFFERENCE_SCALES,
                             "tenure_steps": 7, "weight": 0.2}}))
    m._migrate_efference_scales(str(tmp_path), load_optimizer=True)
    fresh = EF.make_heads(m.d_model, len(m.handle_names), m.GAZE_IDX, 12, m.n_mod,
                          m.efference.vision_idx, m.efference_seed).body_mlp[-1]
    rows = torch.tensor([h * 12 + c for h in range(len(m.handle_names)) for c in range(6, 12)])
    keep = torch.ones(last.weight.shape[0], dtype=torch.bool)
    keep[rows] = False
    assert torch.equal(last.weight[rows], fresh.weight[rows])
    assert torch.equal(last.weight[keep], trained[keep])
    st = m.optimizer.state[last.weight]
    assert float(st["exp_avg"][rows].abs().sum()) == 0.0 and float(st["exp_avg"][keep].min()) == 1.0
    assert m.efference_load["tenure_rows_reset"] == {"saved": 7, "now": 5, "rows": len(rows)}
    assert m.efference_load["weight_changed"] == {"saved": 0.2, "now": 0.1}


def test_another_version_keeps_every_other_adam_moment_around_the_replaced_block():
    """A checkpoint with version-1 heads: the heads' block of the saved optimizer is dropped and
    every moment before and after it is carried by position (lem.model.efference.fresh_start)."""
    torch.manual_seed(6)
    m = _lite(efference_heads=True)
    names = {}
    for comp, mod in EF._named_components(m).items():
        for n, p in mod.named_parameters():
            names[id(p)] = f"{comp}.{n}"
    # a saved model whose heads' block had OTHER tensors (v1): 3 matrices and 2 vectors
    saved = {names[id(p)]: p.detach() for p in m.params if not names[id(p)].startswith("efference_heads.")}
    for k, shp in (("body_mlp.0.weight", (4, 4)), ("gaze_order.weight", (2, 3)),
                   ("gaze_out_23.weight", (4, 2)), ("body_mlp.0.bias", (4,)), ("gaze_order.bias", (2,))):
        saved["efference_heads." + k] = torch.zeros(shp)
    groups, state, idx = [], {}, 0
    for gi, g in enumerate(m.optimizer.param_groups):
        ids = []
        is_eff = [names[id(p)].startswith("efference_heads.") for p in g["params"]]
        first_eff = is_eff.index(True)
        n_eff = sum(is_eff)
        assert all(is_eff[first_eff:first_eff + n_eff]), "the heads are one block"
        before, after = g["params"][:first_eff], g["params"][first_eff + n_eff:]
        n_block = 3 if gi == 0 else 2
        for p in before:
            state[idx] = {"step": torch.tensor(1.0), "exp_avg": torch.full_like(p, 7.0),
                          "exp_avg_sq": torch.ones_like(p)}
            ids.append(idx)
            idx += 1
        for _ in range(n_block):
            state[idx] = {"step": torch.tensor(1.0), "exp_avg": torch.zeros(1), "exp_avg_sq": torch.zeros(1)}
            ids.append(idx)
            idx += 1
        for p in after:
            state[idx] = {"step": torch.tensor(1.0), "exp_avg": torch.full_like(p, 7.0),
                          "exp_avg_sq": torch.ones_like(p)}
            ids.append(idx)
            idx += 1
        groups.append({k: v for k, v in g.items() if k != "params"} | {"params": ids})
    rep = EF._carry_around_block(m, {"state": state, "param_groups": groups}, saved, "efference_heads.")
    n_other = sum(1 for p in m.params if not names[id(p)].startswith("efference_heads."))
    assert rep == {"adam_states_saved": idx, "adam_states_carried": n_other,
                   "adam_states_dropped_with_old_heads": 5}
    for p in m.params:
        st = m.optimizer.state.get(p)
        if names[id(p)].startswith("efference_heads."):
            assert not st, names[id(p)]
        else:
            assert st and float(st["exp_avg"].flatten()[0]) == 7.0, names[id(p)]


def test_a_checkpoint_with_learned_heads_refuses_a_build_without_them(tmp_path):
    m = _lite()
    (tmp_path / "model_receipt.json").write_text(json.dumps(
        {"body_layout": m.body_layout, "efference_heads": {"version": "efference_v2"}}))
    with pytest.raises(ValueError, match="efference heads"):
        m.load_checkpoint(str(tmp_path))


def test_a_GazeHead_the_first_version_retrained_is_refused_as_a_gaze_direction(tmp_path):
    """Review MINOR 5: lem/* callers read gaze_head(last_step) as the direction the eyes should
    point; a GazeHead the first version retrained as a 50 ms turn head is refused for them."""
    from lem.training.checkpoint import CheckpointManager
    m = _lite()
    (tmp_path / "model_receipt.json").write_text(json.dumps(
        {"efference_heads": {"version": "efference_v1", "gaze_head_repurposed": True}}))
    mgr = CheckpointManager(m.config, str(tmp_path / "ck"))
    with pytest.raises(ValueError, match="GazeHead was retrained"):
        mgr.load(str(tmp_path), m.encoder_registry, m.forward_model, m.prediction_heads,
                 m.motor_controller, m.wellness, gaze_head=m.gaze_head)


def test_the_heads_are_clipped_as_their_own_group():
    """A huge gradient on the heads must not shrink the step of any other module: the shared
    clip norm is computed without them."""
    m = _lite(efference_heads=True)
    for p in m.params:
        p.grad = torch.zeros_like(p)
    core = next(m.forward_model.parameters())
    core.grad.fill_(1e-4)                       # tiny, far under the clip line
    for p in m.efference.parameters():
        p.grad.fill_(10.0)                      # enormous
    before = core.grad.clone()
    norm = m._clip_training_gradients()
    assert torch.equal(core.grad, before), "the core's step was shrunk by the heads' gradient"
    assert abs(norm - float(before.norm())) < 1e-6
    eff_norm = torch.norm(torch.stack([p.grad.norm() for p in m.efference.parameters()]))
    assert float(eff_norm) <= m.grad_clip * (1 + 1e-4), "the heads' own group is still clipped"


def test_window_sums_stay_out_of_the_episode_companion_and_the_rest_round_trips():
    """continuation_state.capture_learner serialises every _* and last_* attribute and raises on
    anything it has no serializer for. Review MINOR 8: the window sums (and the lesson's timing)
    are public, so the first block after a resume holds only moments after it."""
    from mimo_bridge import continuation_state as CS
    torch.manual_seed(4)
    m = _lite(efference_heads=True, causal_chain=True)
    B, H = 2, len(m.handle_names)
    exec_t = _order(B, H)
    m._nf_ach = _pose(B, H, 0.0)
    for s in range(10, 13):
        with torch.no_grad():
            m._efference_lesson(_latent(m, B), exec_t, s, _pose(B, H, float(s)), _frames(B, s), s, B)
    state = CS.capture_learner(m, include_replay=False)
    for k in ("_eff_hist", "_eff_eye_prev", "_eff_tenure", "_transition_epoch", "_causal_seed"):
        assert k in state["data"], k
    for k in ("efference_window", "efference_timing", "_eff_window"):
        assert k not in state["data"], k
    assert m.efference_window, "the window holds this process's gauges"
    other = _lite(efference_heads=True, causal_chain=True)
    CS.restore_learner(other, state)
    assert len(other._eff_hist) == 3 and other._eff_hist.maxlen == 3
    assert torch.equal(other._eff_hist[-1]["latent"], m._eff_hist[-1]["latent"])
    assert torch.equal(other._eff_tenure["expected"], m._eff_tenure["expected"])
    assert other._eff_tenure["step"] == m._eff_tenure["step"]
    assert other.efference_window == {}
    assert other._transition_epoch == m._transition_epoch


def test_the_gaze_copy_is_zero_until_trained_masked_where_no_order_is_known():
    """Review MAJOR 4 (DESIGN D5 as amended): vision's next-moment prediction gets the explicit
    eye command through a zero-initialised path."""
    torch.manual_seed(7)
    heads = _heads(d=16, n_mod=3, vision_idx=0)

    class Heads(torch.nn.Module):
        def forward(self, last, current=None, who=""):
            return {"vision": last[:, 0] * 0.5}
    ph = Heads()
    lat = torch.randn(2, 3, 16)
    order = torch.randn(2, 26, 9)
    ach = _pose(2)
    heads.ensure_centres(lat, order)
    EF.attach_gaze_copy(ph, heads)
    assert list(ph.state_dict()) == [] and EF.gaze_copy_of(ph) is not None
    base = {"vision": lat[:, 0] * 0.5}
    out = EF.apply_gaze_copy(ph, base, lat, order, ach)
    assert torch.equal(out["vision"], base["vision"]), "zero-initialised: nothing moves on switch"
    with torch.no_grad():
        heads.vision_out.bias.fill_(0.25)
    out = EF.apply_gaze_copy(ph, base, lat, order, ach)
    assert torch.allclose(out["vision"], base["vision"] + 0.25)
    # masked windows (replay rows with no known order) get nothing
    out = EF.apply_gaze_copy(ph, base, lat, order, ach, mask=torch.tensor([True, False]))
    assert torch.allclose(out["vision"][0], base["vision"][0] + 0.25)
    assert torch.equal(out["vision"][1], base["vision"][1])
    # an unset centre (no lived moment yet) adds nothing
    heads.centre_weight = 0.0
    assert EF.apply_gaze_copy(ph, base, lat, order, ach) is base


def test_transition_adds_the_gaze_copy_exactly_as_the_lesson_does():
    """transition() (the planner, its Adam loop, the imagination chain, the causal-chain
    predictions and the probes' rollouts) calls the copy the prediction heads carry; a build without
    it is unchanged. Wired here on a small build's touch token standing in for vision."""
    from lem.model.predictive_state import PredictiveState, transition
    from mimo_bridge.lem_model import command_for_encoder
    torch.manual_seed(8)
    m = _lite()
    B, H = 2, len(m.handle_names)
    state = PredictiveState(tokens={mm: torch.randn(B, m.d_model) for mm in m.active},
                            context=torch.zeros(B, (m.ctx_len - 1) * m.n_mod, m.d_model),
                            achieved=_pose(B, H))
    act = command_for_encoder(_order(B, H))
    with torch.no_grad():
        plain = transition(m.forward_model, m.encoder_registry, m.prediction_heads, state, act)
        heads = EF.make_heads(m.d_model, H, m.GAZE_IDX, 12, m.n_mod, m.active.index("tactile"), 0)
        heads.ensure_centres(torch.randn(B, m.n_mod, m.d_model), act)
        object.__setattr__(m.prediction_heads, "_efference_gaze_copy", EF.GazeCopy(heads, "tactile"))
        zero = transition(m.forward_model, m.encoder_registry, m.prediction_heads, state, act)
        for k in plain.tokens:
            assert torch.equal(plain.tokens[k], zero.tokens[k]), k
        heads.vision_out.bias.copy_(torch.linspace(-1.0, 1.0, m.d_model))
        moved = transition(m.forward_model, m.encoder_registry, m.prediction_heads, state, act)
    assert not torch.allclose(moved.tokens["tactile"], plain.tokens["tactile"])
    for k in plain.tokens:
        if k != "tactile":
            assert torch.equal(plain.tokens[k], moved.tokens[k]), k


def test_head_motion_comes_from_the_vestibular_vectors_head_fields():
    from mimo_bridge.vestibular_adapter import FIELD_INDEX, VESTIBULAR_DIM
    v = torch.zeros(2, VESTIBULAR_DIM)
    for i, n in enumerate(EF.HEAD_MOTION_FIELDS):
        v[:, FIELD_INDEX[n]] = float(i + 1)
    hm = EF.head_motion_from_vestibular(v)
    assert torch.equal(hm, torch.tensor([[1.0, 2, 3, 4, 5, 6]] * 2))
    assert EF.head_motion_from_vestibular(None) is None
    assert EF.head_motion_of(SimpleNamespace(vestibular_sense=False), {"vestibular": v}) is None


def test_the_lesson_timer_counts_milliseconds_on_the_cpu():
    t = EF.LessonTimer()
    acc = [0.0, 0]
    t.start(torch.device("cpu"), 3)
    t.stop(torch.device("cpu"), acc)
    assert acc[1] == 1 and acc[0] >= 0.0
    model = SimpleNamespace(efference_timing=[30.0, 3])
    assert EF.lesson_ms(model) == 10.0 and model.efference_timing == [0.0, 0]
    assert EF.lesson_ms(model) is None
