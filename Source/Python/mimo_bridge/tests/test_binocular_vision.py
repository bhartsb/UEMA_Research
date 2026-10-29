# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Tests for the sharper, binocular eyes: the fovea pair, stereo, and no repeated frames.

The load-bearing one is test_two_positional_numbers_are_eye_blind. It does not assert that
the old feed was a bit worse at stereo - it shows the pooled vision token was BIT-IDENTICAL
when the two eyes' images were swapped, so disparity could not be read at all, and that
adding the eye number breaks that tie.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lem.model.encoders.swin_patch import build_swin_patch
from mimo_bridge.vision_adapter import BinocularPatcher, acuity, image_to_patches


def _encoder(n_pos: int, seed: int = 0):
    torch.manual_seed(seed)
    return build_swin_patch(dict(
        _n_patches=64, _patch_dim=768, _d_internal=32, _d_output=32,
        patch_size=16, n_swin_layers=1, n_heads=4, window_size=7,
        angular_pos_hidden=32, ffn_expansion=2, dropout=0.0, bias=False,
        norm_type="layernorm", pool_n_heads=4, patch_subsample=None,
        angular_pos_inputs=n_pos)).eval()


# ------------------------------------------------------------------ acuity arithmetic
def test_acuity_matches_the_snellen_convention():
    # 20/20 is resolving one arcmin, which is 30 cycles per degree, which needs 60 px/deg
    a = acuity(600, 10.0)
    assert a["px_per_deg"] == pytest.approx(60.0)
    assert a["cycles_per_deg"] == pytest.approx(30.0)
    assert a["snellen"] == pytest.approx(20.0, abs=0.01)


def test_the_two_cameras_land_where_we_claim():
    wide = acuity(256, 60.0)          # what MIMo's eye gives on its own
    fovea = acuity(288, 24.0)         # the narrow camera we add inside it
    assert wide["snellen"] == pytest.approx(281, abs=1)      # newborn-to-three-months
    assert fovea["snellen"] == pytest.approx(100, abs=1)     # six months
    assert fovea["px_per_deg"] / wide["px_per_deg"] == pytest.approx(2.81, abs=0.01)


# ------------------------------------------------------------------ the patcher
def _imgs(sizes, seed=0):
    r = np.random.default_rng(seed)
    return {n: (r.integers(0, 255, (s, s, 3), dtype=np.uint8),
                r.integers(0, 255, (s, s, 3), dtype=np.uint8)) for n, s in sizes.items()}


def test_layout_shapes_and_counts():
    p = BinocularPatcher([("wide", 256, 60.0), ("fovea", 288, 24.0)])
    assert p.n_per_eye == 256 + 324 and p.n_total == 1160
    patches, ang = p(_imgs({"wide": 256, "fovea": 288}))
    assert patches.shape == (1160, 768)
    assert ang.shape == (1160, 4)
    assert patches.dtype == np.float32 and ang.dtype == np.float32


def test_left_and_right_are_interleaved_not_concatenated():
    """A 7-patch attention window has to straddle both eyes to compare them."""
    p = BinocularPatcher([("wide", 64, 60.0)])
    im = _imgs({"wide": 64})
    patches, ang = p(im)
    left_only = image_to_patches(im["wide"][0])
    right_only = image_to_patches(im["wide"][1])
    assert np.allclose(patches[0::2], left_only)
    assert np.allclose(patches[1::2], right_only)
    assert np.all(ang[0::2, 2] == -1.0) and np.all(ang[1::2, 2] == +1.0)


def test_span_column_separates_the_two_cameras():
    p = BinocularPatcher([("wide", 256, 60.0), ("fovea", 288, 24.0)])
    _, ang = p(_imgs({"wide": 256, "fovea": 288}))
    wide_span = math.radians(60.0) * 16 / 256
    fovea_span = math.radians(24.0) * 16 / 288
    assert np.allclose(ang[:512, 3], wide_span)
    assert np.allclose(ang[512:, 3], fovea_span)
    assert wide_span / fovea_span == pytest.approx(2.81, abs=0.01)


def test_gaze_offsets_both_eyes_angles():
    """Audit P026: the eye's joints are COMPOSED onto each patch's ray, not added. With a
    level eye a pure yaw is an exact azimuth shift for every patch and leaves elevation
    alone; a pure pitch shifts the centre column's elevation by exactly the pitch and
    leaves the centre patch's azimuth at zero, while an off-centre patch's azimuth moves
    (the thing plain addition got wrong)."""
    from mimo_bridge.vision_adapter import compose_gaze, patch_grid_rays
    p = BinocularPatcher([("wide", 64, 60.0)])
    _, a0 = p(_imgs({"wide": 64}), (0.0, 0.0), (0.0, 0.0))
    _, a1 = p(_imgs({"wide": 64}), (0.3, 0.0), (0.25, 0.0))
    assert np.allclose(a1[0::2, 0] - a0[0::2, 0], 0.3, atol=1e-5)    # left eye yaw
    assert np.allclose(a1[1::2, 0] - a0[1::2, 0], 0.25, atol=1e-5)   # right eye yaw
    assert np.allclose(a1[:, 1], a0[:, 1], atol=1e-6)                 # elevation untouched by yaw
    _, a2 = p(_imgs({"wide": 64}), (0.0, -0.1), (0.0, -0.1))
    rays = patch_grid_rays(64, 64, 60.0)
    ref = compose_gaze(rays, 0.0, -0.1)
    assert np.allclose(a2[0::2, :2], ref, atol=1e-6)                  # the patcher IS the composition
    rows = cols = 64 // 16
    centre_col = [r * cols + c for r in range(rows) for c in range(cols) if c in (cols // 2 - 1, cols // 2)]
    d_el = (a2[0::2, 1] - a0[0::2, 1])[centre_col]
    assert np.allclose(d_el, -0.1, atol=2e-3)                         # centre columns: elevation shifts by the pitch
    corner = 0                                                         # top-left patch
    assert abs(float(a2[0::2, 0][corner] - a0[0::2, 0][corner])) > 1e-3   # its azimuth moves under pitch


def test_wrong_image_size_is_refused_not_silently_reshaped():
    p = BinocularPatcher([("wide", 64, 60.0)])
    with pytest.raises(ValueError, match="patches per eye"):
        p(_imgs({"wide": 128}))


# ------------------------------------------------------------------ the stereo claim
def _swap_eyes(patches, ang):
    """Put the left eye's pixels in the right eye's slots and vice versa."""
    p2, a2 = patches.copy(), ang.copy()
    p2[0::2], p2[1::2] = patches[1::2], patches[0::2]
    return p2, a2


def test_without_the_eye_number_both_eyes_get_the_same_tag():
    """Exact statement of the problem: the two eyes' positional tags were the same numbers.

    Both eyes are commanded to one gaze pair, so with only (azimuth, elevation) a patch from
    the left eye and the matching patch from the right carry byte-for-byte identical
    positional input. Nothing downstream can recover which eye it came from except by
    accident of where the attention windows happen to fall.
    """
    p = BinocularPatcher([("wide", 64, 60.0)])
    _, ang = p(_imgs({"wide": 64}), (0.2, -0.1), (0.2, -0.1))
    assert np.array_equal(ang[0::2, :2], ang[1::2, :2])      # tags collide with 2 numbers
    assert not np.array_equal(ang[0::2, :3], ang[1::2, :3])   # the eye number breaks the tie


def test_the_pool_cannot_recover_order_on_its_own():
    """The pool is one learned query cross-attending the set, so it is order-blind, exactly.

    This is why the eye has to be named in the tag: the last stage of the encoder throws
    away position entirely, so any eye identity carried only by sequence order is gone.
    """
    enc = _encoder(n_pos=4)
    feats = torch.randn(1, 60, 32)
    perm = torch.randperm(60)
    with torch.no_grad():
        a = enc.pool(feats)
        b = enc.pool(feats[:, perm])
    assert torch.allclose(a, b, atol=1e-6)


def test_naming_the_eye_makes_a_swap_visible():
    """Swap the two eyes' images and see how much the pooled token moves.

    Three configurations, one random encoder, one pair of images. With the old
    concatenated order and two positional numbers the token barely moves at all - what
    little movement there is comes from attention windows straddling the join, an artefact
    of patch count rather than a depth cue. Naming the eye moves it several times further,
    which is the difference between disparity being readable and not.
    """
    from mimo_bridge.vision_adapter import EyePatcher
    r = np.random.default_rng(1)
    L = r.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    R = r.integers(0, 255, (256, 256, 3), dtype=np.uint8)

    def moved(encoder, pa, aa, pb, ab):
        with torch.no_grad():
            x = encoder(torch.from_numpy(pa)[None], torch.from_numpy(aa)[None])
            y = encoder(torch.from_numpy(pb)[None], torch.from_numpy(ab)[None])
        return float((x - y).norm() / x.norm())

    e2 = _encoder(n_pos=2)
    old = EyePatcher(256, 256, 60.0)
    concat = moved(e2, *old(L, R), *old(R, L))

    bp = BinocularPatcher([("wide", 256, 60.0)])
    p1, a1 = bp({"wide": (L, R)})
    p2, a2 = bp({"wide": (R, L)})
    inter = moved(e2, p1, a1[:, :2], p2, a2[:, :2])

    e4 = _encoder(n_pos=4)
    with torch.no_grad():                      # zero columns until training weights them
        e4.angular_pos_enc.mlp[0].weight[:, 2:] = 0.5
    named = moved(e4, p1, a1, p2, a2)

    assert concat < 5e-3 and inter < 5e-3, (concat, inter)
    assert named > 4 * max(concat, inter), (concat, inter, named)


def test_zero_columns_reproduce_the_old_encoder_exactly():
    """What grow_vision relies on: widening with zeros must not move the output."""
    old = _encoder(n_pos=2, seed=7)
    new = _encoder(n_pos=4, seed=7)
    with torch.no_grad():
        for (ko, vo), (kn, vn) in zip(old.state_dict().items(), new.state_dict().items()):
            assert ko == kn
            if vo.shape == vn.shape:
                vn.copy_(vo)
            else:                                   # the one widened matrix
                vn.zero_()
                vn[:, :vo.shape[1]] = vo
    p = BinocularPatcher([("wide", 64, 60.0)])
    patches, ang = p(_imgs({"wide": 64}, seed=3))
    with torch.no_grad():
        a = old(torch.from_numpy(patches)[None], torch.from_numpy(ang[:, :2])[None])
        b = new(torch.from_numpy(patches)[None], torch.from_numpy(ang)[None])
    assert torch.allclose(a, b, atol=1e-6)


def test_encoder_takes_any_patch_count():
    """Patch COUNT needs no checkpoint surgery - only the positional width does."""
    enc = _encoder(n_pos=4)
    for n in (512, 800, 1160):
        with torch.no_grad():
            out = enc(torch.randn(1, n, 768), torch.randn(1, n, 4))
        assert out.shape == (1, 1, 32)


# ------------------------------------------------------------------ growing the encoder
def test_a_pass_through_block_changes_nothing():
    """grow_vision --blocks relies on this: a second block must start as the identity.

    A block computes x + attn(norm(x)) then x + ffn(norm(x)). Zero the two output
    projections and both added terms are zero, so the block returns x exactly and the
    trained encoder carries over unchanged instead of being perturbed on step one.
    """
    one = _encoder(n_pos=4, seed=11)
    torch.manual_seed(11)
    two = build_swin_patch(dict(
        _n_patches=64, _patch_dim=768, _d_internal=32, _d_output=32,
        patch_size=16, n_swin_layers=2, n_heads=4, window_size=7,
        angular_pos_hidden=32, ffn_expansion=2, dropout=0.0, bias=False,
        norm_type="layernorm", pool_n_heads=4, patch_subsample=None,
        angular_pos_inputs=4)).eval()

    sd = one.state_dict()
    with torch.no_grad():
        for k, v in two.state_dict().items():
            if k.startswith("blocks.1."):
                tail = k[len("blocks.1."):]
                src = sd["blocks.0." + tail]
                v.copy_(torch.zeros_like(src) if tail in
                        ("attn.out_proj.weight", "attn.out_proj.bias", "ffn.3.weight", "ffn.3.bias")
                        else src)
            else:
                v.copy_(sd[k])

    p = BinocularPatcher([("wide", 64, 60.0)])
    patches, ang = p(_imgs({"wide": 64}, seed=4))
    with torch.no_grad():
        a = one(torch.from_numpy(patches)[None], torch.from_numpy(ang)[None])
        b = two(torch.from_numpy(patches)[None], torch.from_numpy(ang)[None])
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()
