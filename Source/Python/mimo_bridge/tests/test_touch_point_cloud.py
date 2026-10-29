# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Touch is a point cloud: contacts are (position, features) tokens, not slot indices.

The disease this replaces, measured on H32's ckpt_step_855903: shifting one contact by
ONE patch moved the tactile token 455 while teleporting it to a random body site moved
it 413 (ratio 1.10) and changing the pattern's CONTENT on the same sites moved it only
232. The encoder had 256 anonymous slots and no notion of neighbourhood, so next-state
touch prediction was rote memorisation of patch adjacencies: it degraded +70% over the
run while touch reconstruction improved 54% and contact statistics stayed flat (+/-2%).

The same probe on the new encoder AT INITIALISATION, before any training:

    shift/random ratio   0.180   (was 1.10 - a one-patch slide is now a small change)
    content/shift        9.46    (what is felt dominates where the slot was)
    slots reshuffled     0.000   (slot order is not identity at all)

Vision and proprioception - the two senses whose prediction improves - already work
this way (angular coordinates; continuous bone values). Perception_Space.pdf section 1:
"touch dimensions are structured by the body surface geometry."
"""
import numpy as np
import pytest
import torch

from mimo_bridge.lem_model import (LemLite, touch_point_cloud, TOUCH_POS_SCALE,
                                   TOUCH_TOKENS, TOUCH_TOKEN_F, TOUCH_VALID_COL)


@pytest.fixture(scope="module")
def enc():
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False)
    return m.encoder_registry.encoders["tactile"]


def _line_body(n=53, spacing=0.01):
    pos = np.zeros((n, 3), dtype=np.float32)
    pos[:, 0] = np.arange(n) * spacing
    return pos


def _contact(sites, feat, n_slots=256):
    imp = torch.zeros(n_slots, 8)
    imp[sites] = feat
    return imp


# ------------------------------------------------------------------ the builder
def test_builder_keeps_active_rows_sorted_by_pressure_and_pads():
    pos = _line_body()
    imp = torch.zeros(256, 8)
    imp[10, 0] = 0.2
    imp[40, 0] = 0.9
    imp[25, 0] = 0.5
    out = touch_point_cloud(imp, pos)
    assert out.shape == (TOUCH_TOKENS, TOUCH_TOKEN_F)
    assert out[0, 3] == pytest.approx(0.9)      # strongest contact first
    assert out[1, 3] == pytest.approx(0.5)
    assert out[2, 3] == pytest.approx(0.2)
    assert out[:3, TOUCH_VALID_COL].tolist() == [1.0, 1.0, 1.0]
    assert out[3:].abs().sum() == 0.0           # padding is exact zeros
    # coordinates are root-relative metres over the scale
    assert out[0, 0] == pytest.approx(40 * 0.01 / TOUCH_POS_SCALE)


def test_builder_with_no_contacts_is_all_zero():
    out = touch_point_cloud(torch.zeros(256, 8), _line_body())
    assert out.abs().sum() == 0.0


def test_builder_survives_missing_positions():
    """Old pickles and the UE transport carry no touch_pos: features still flow."""
    imp = torch.zeros(256, 8)
    imp[5, 0] = 0.3
    out = touch_point_cloud(imp, None)
    assert out[0, 3] == pytest.approx(0.3)
    assert out[0, TOUCH_VALID_COL] == 1.0
    assert out[0, 0:3].abs().sum() == 0.0


def test_more_contacts_than_slots_keeps_the_strongest():
    pos = _line_body()
    imp = torch.zeros(256, 8)
    imp[: TOUCH_TOKENS + 10, 0] = torch.linspace(0.1, 1.0, TOUCH_TOKENS + 10)
    out = touch_point_cloud(imp, pos)
    assert float(out[:, TOUCH_VALID_COL].sum()) == TOUCH_TOKENS
    assert out[0, 3] == pytest.approx(1.0)      # the strongest survived the cut


# ------------------------------------------------------- the property that matters
def test_a_one_patch_shift_is_a_small_change_not_a_teleport(enc):
    """The probe that measured 1.10 on the slot encoder must stay well below it."""
    pos = _line_body()
    g = torch.Generator().manual_seed(3)
    feat = torch.rand(4, 8, generator=g) * 0.5 + 0.1
    t0 = enc(touch_point_cloud(_contact([20, 21, 22, 23], feat), pos).unsqueeze(0))
    d_shift = float((enc(touch_point_cloud(_contact([21, 22, 23, 24], feat), pos)
                         .unsqueeze(0)) - t0).norm())
    ds = []
    for s in range(12):
        gg = torch.Generator().manual_seed(100 + s)
        where = torch.randperm(53, generator=gg)[:4].tolist()
        ds.append(float((enc(touch_point_cloud(_contact(where, feat), pos)
                             .unsqueeze(0)) - t0).norm()))
    d_rand = sum(ds) / len(ds)
    ratio = d_shift / d_rand
    assert ratio < 0.5, (
        f"shift/random ratio {ratio:.2f} - the boundary-aliasing disease is back "
        f"(slot encoder measured 1.10; this encoder measured 0.18 at init)")


def test_a_shift_stays_small_relative_to_a_teleport_on_trained_weights_too():
    """The invariant is shift << teleport. It is NOT "content beats position".

    This test used to demand that changing what a contact FEELS like moves the token
    at least 2x more than sliding it 1 cm. That held on fresh weights (15x) and failed
    on H33's trained weights (0.78x): the model learned that WHERE a touch lands on the
    body is highly informative, which is legitimate. The disease was never "position
    matters" - it was that a 1 cm slide looked like a random teleport (ratio 1.10 on
    the slot encoder). On H33's trained weights that ratio was 0.086 and had improved
    with training. So the check here is the shift/random ratio, on the newest trained
    checkpoint if one exists, and on fresh weights otherwise.
    """
    import glob, os, re
    from safetensors.torch import load_file
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False)
    root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "Saved",
                        "mimo_bridge", "experiments")
    cks = glob.glob(os.path.join(root, "*", "checkpoints", "ckpt_step_*", "model.safetensors"))
    cks = [c for c in cks if os.path.getmtime(c)]
    loaded = "fresh weights"
    lin = next(mm for mm in m.encoder_registry.encoders["tactile"].modules()
               if isinstance(mm, torch.nn.Linear))
    # newest first; the first checkpoint whose tactile encoder has THIS fixture's shape is
    # used. Since the size flags of 2026-09-17 a run can leave a 192- or 256-wide encoder
    # behind (the H84 launch smokes did), and the input width alone (12 numbers per touch
    # point, the same at every size) does not tell them apart: the hidden width must match.
    for ck in sorted(cks, key=os.path.getmtime, reverse=True):
        sd = load_file(ck)
        reg = {k[len("encoder_registry."):]: v.float() for k, v in sd.items()
               if k.startswith("encoder_registry.")}
        tact = {k: v for k, v in reg.items() if k.startswith("encoders.tactile")}
        w = tact.get("encoders.tactile.patch_mlp.0.weight")
        if w is not None and tuple(w.shape) == (lin.out_features, lin.in_features):
            # This assay only uses tactile weights. Trunk-expanded motor and
            # proprioception tensors must not be loaded into a 24-handle fixture.
            m.encoder_registry.load_state_dict(tact, strict=False)
            loaded = os.path.relpath(ck, root)
            break
    enc = m.encoder_registry.encoders["tactile"]
    norm = m.encoder_registry.output_norm
    pos = _line_body()
    f = lambda imp: norm(enc(touch_point_cloud(imp, pos).unsqueeze(0)))
    S = R = 0.0
    for t in range(6):
        g = torch.Generator().manual_seed(3 + t)
        feat = torch.rand(4, 8, generator=g) * 0.5 + 0.1
        base = 15 + t * 4
        sites = [base, base + 1, base + 2, base + 3]
        with torch.no_grad():
            t0 = f(_contact(sites, feat))
            S += float((f(_contact([x + 1 for x in sites], feat)) - t0).norm())
            ds = []
            for s_ in range(12):
                gg = torch.Generator().manual_seed(100 + s_ + 20 * t)
                where = torch.randperm(53, generator=gg)[:4].tolist()
                ds.append(float((f(_contact(where, feat)) - t0).norm()))
            R += sum(ds) / len(ds)
    ratio = S / R
    assert ratio < 0.5, (
        f"shift/random ratio {ratio:.3f} on {loaded} - a 1 cm slide is starting to look "
        f"like a teleport again (slot encoder: 1.10; point cloud fresh 0.118, H33 trained 0.086)")


def test_slot_order_is_not_identity(enc):
    """Shuffle which SLOT each patch occupies, keep the physical contacts identical."""
    n = 53
    pos = _line_body(n)
    g = torch.Generator().manual_seed(3)
    imp = torch.zeros(256, 8)
    imp[[20, 21, 22, 23]] = torch.rand(4, 8, generator=g) * 0.5 + 0.1
    t0 = enc(touch_point_cloud(imp, pos).unsqueeze(0))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(9))
    imp_p = torch.zeros(256, 8)
    imp_p[:n] = imp[:n][perm]
    d = float((enc(touch_point_cloud(imp_p, pos[perm.numpy()]).unsqueeze(0)) - t0).norm())
    assert d < 1e-4, f"reshuffling slots moved the token by {d} - slots are identity again"


def test_the_encoder_has_no_slot_embedding(enc):
    assert getattr(enc, "patch_pos", None) is None, (
        "the tactile encoder grew a per-slot embedding back - geometry is the identity now")
    lin = next(m for m in enc.modules() if isinstance(m, torch.nn.Linear))
    assert lin.in_features == TOUCH_TOKEN_F


# ------------------------------------------------ the row got wider twice in two days
# 12 -> 14 (the self-touch share and the count of own parts) -> 19 (the push's direction,
# depth and arrival speed). Every widening rewrites two things that must agree: the STORED
# touch tokens a resume brings back, and the one encoder projection whose input columns ARE
# those columns. They disagreed, and the disagreement is what these tests hold.
OLD_WIDTHS = (12, 14)


def _stored_token(width, seed=0, rows=3, bodies=2):
    """A touch token as an older build stored it: features, then the validity flag in that
    build's own last column."""
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(bodies, rows, width, generator=g)
    t[..., width - 1] = 1.0                 # every row a real contact, in the old flag column
    return t


@pytest.mark.parametrize("width", OLD_WIDTHS)
def test_a_stored_touch_token_is_widened_with_its_validity_flag_carried(width):
    """fit_tactile, which every replayed moment, delayed-lesson cache and imagination history
    passes through on a resume. Only its early return ran in any test before 2026-09-21, so
    the thirteen statements that do the widening were never executed - on a row that has been
    widened twice in two days."""
    from mimo_bridge.lem_model import fit_tactile
    t = _stored_token(width)
    frame = {"tactile": t, "vision": torch.zeros(1)}
    out = fit_tactile(frame)
    w = out["tactile"]
    assert w.shape == (t.shape[0], t.shape[1], TOUCH_TOKEN_F)
    assert w.dtype == t.dtype and w.device == t.device
    # the columns that still mean what they meant
    torch.testing.assert_close(w[..., :width - 1], t[..., :width - 1])
    # the validity flag is read from TOUCH_VALID_COL now, so that is where it has to be
    torch.testing.assert_close(w[..., TOUCH_VALID_COL], t[..., width - 1])
    # the column the flag used to sit in now means something else and starts at nothing
    assert float(w[..., width - 1].abs().sum()) == 0.0
    assert float(w[..., width:TOUCH_VALID_COL].abs().sum()) == 0.0
    # the frame is a new dict and the stored tensor is not written into
    assert out is not frame and out["vision"] is frame["vision"]
    assert float(t[..., width - 1].min()) == 1.0
    # and the widened row is read as a real contact, which is the whole point of moving the
    # flag: a row whose self-share is above zero counts as the body touching itself. Only
    # from 14 wide on, because the self-share column did not exist before that.
    from mimo_bridge.lem_model import self_touch_rows
    if width - 1 > 11:
        t2 = _stored_token(width); t2[0, 0, 11] = 0.7
        rows = self_touch_rows(fit_tactile({"tactile": t2})["tactile"])
        assert rows is not None and bool(rows[0, 0])
    else:
        # at 12 wide there was no self-share column, so a widened 12-wide row carries a
        # self-share of exactly zero and is not self-touch, which is the truth about it
        rows = self_touch_rows(w)
        assert rows is not None and not bool(rows.any())


def test_a_token_that_already_fits_is_handed_back_untouched_and_a_wider_one_is_cut():
    from mimo_bridge.lem_model import fit_tactile
    fits = {"tactile": torch.zeros(2, 3, TOUCH_TOKEN_F)}
    assert fit_tactile(fits) is fits, "the live path must pay nothing"
    assert fit_tactile({"vision": torch.zeros(1)}).get("tactile") is None
    assert fit_tactile({"tactile": [1, 2, 3]})["tactile"] == [1, 2, 3]
    wider = torch.rand(2, 3, TOUCH_TOKEN_F + 4)
    cut = fit_tactile({"tactile": wider})["tactile"]
    assert cut.shape[-1] == TOUCH_TOKEN_F
    torch.testing.assert_close(cut, wider[..., :TOUCH_TOKEN_F])


def test_a_resume_across_a_touch_width_change_keeps_the_encoder_on_the_same_columns(tmp_path):
    """The other half of the same widening: the tactile encoder's first projection, whose
    input columns ARE the token's columns.

    THE DEFECT THIS PINS, measured read-only on the live checkpoints of run H84
    (encoder_registry.encoders.tactile.patch_mlp.0.weight, mean |weight| per column): at 12
    wide, segment 22 carried 0.145 on column 11, the validity flag. After the 14-wide resume,
    segment 23 carried that same 0.145 on column 11 - which at 14 wide is the self-touch
    share, a column that had never existed and is meant to open at exactly zero - while
    column 13, the flag the widened tokens actually use, opened at 0.000. The tokens were
    permuted and the weights were not. The 19-wide resume of segment 29 repeated it.

    What is asserted is the contract in its strongest form: the same physical moment, stored
    at the old width, produces the same numbers out of the widened projection as it did out
    of the saved one."""
    import safetensors.torch as st
    from mimo_bridge.lem_model import fit_tactile, widen_touch_columns
    OLD = 14
    model = LemLite(device="cpu", seed=0, intention=False, imagination=False)
    enc = model.encoder_registry.encoders["tactile"]
    live = dict(enc.named_parameters()); live.update(dict(enc.named_buffers()))
    token_wide = [n for n, t in live.items() if t.ndim == 2 and t.shape[1] == TOUCH_TOKEN_F]
    assert token_wide == ["patch_mlp.0.weight"], (
        "the set of tensors indexed by touch column changed: %s" % token_wide)
    name = token_wide[0]
    g = torch.Generator().manual_seed(11)
    old_w = torch.randn(live[name].shape[0], OLD, generator=g)
    saved = {"encoder_registry.encoders.tactile." + n:
             (old_w.clone() if n == name else
              (torch.randn(t.shape, generator=g) if t.is_floating_point() else t.clone()))
             for n, t in live.items()}
    st.save_file(saved, str(tmp_path / "model.safetensors"))
    param = dict(enc.named_parameters())[name]
    old_moment = torch.randn(param.shape[0], OLD, generator=g)
    model.optimizer.state[param] = {"step": torch.tensor(7.0),
                                    "exp_avg": old_moment.clone(),
                                    "exp_avg_sq": old_moment.abs().clone()}

    # a result that does not name the tactile encoder leaves everything alone
    untouched = param.detach().clone()
    model._restore_widened_touch_encoder(str(tmp_path), {"reinitialized_components": ["encoder:vision"]})
    torch.testing.assert_close(param.detach(), untouched)

    model._restore_widened_touch_encoder(str(tmp_path), {"reinitialized_components": ["encoder:tactile"]})
    new_w = param.detach()
    assert new_w.shape[1] == TOUCH_TOKEN_F
    torch.testing.assert_close(new_w[:, :OLD - 1], old_w[:, :OLD - 1])
    torch.testing.assert_close(new_w[:, TOUCH_VALID_COL], old_w[:, OLD - 1])
    assert float(new_w[:, OLD - 1].abs().sum()) == 0.0, (
        "the trained validity weight is sitting on the column that now means something else "
        "- this is the defect measured on H84 segment 23")
    assert float(new_w[:, OLD:TOUCH_VALID_COL].abs().sum()) == 0.0
    # every other saved tensor was copied straight in
    for n, t in live.items():
        if n != name:
            torch.testing.assert_close(t.detach(),
                                       saved["encoder_registry.encoders.tactile." + n].to(t.dtype))
    # Adam's moments belong to their own columns and move with them
    state = model.optimizer.state[param]
    for key, source in (("exp_avg", old_moment), ("exp_avg_sq", old_moment.abs())):
        torch.testing.assert_close(state[key], widen_touch_columns(source))
        assert state[key].shape == param.shape
    assert float(state["step"]) == 7.0, "the optimizer's step count is not a per-column thing"

    # THE CONTRACT: the same stored moment produces the same numbers before and after
    tok = _stored_token(OLD, seed=5, rows=TOUCH_TOKENS, bodies=2)
    before = tok @ old_w.T
    after = fit_tactile({"tactile": tok})["tactile"] @ new_w.T
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
