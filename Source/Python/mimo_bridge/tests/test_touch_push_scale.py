# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The push's depth and arrival speed, on this body's own scale, and the exact migration.

Measured 2026-09-21 on the crib body over 3,000 model steps of motor babble: the 99th percentile
contact pressed 0.108 cm deep and the 99th percentile arrival came in at 2.21 cm/s, so on the old
divisors of 10 cm and 500 cm/s they read 0.0108 and 0.0044 of their scales. The divisors are now
0.1 cm and 2 cm/s. A checkpoint trained under the old ones must read every token exactly as it
did, which is what these tests hold.
"""
import json

import numpy as np
import pytest
import torch

from mimo_bridge.lem_model import (LemLite, TOUCH_ARRIVAL_COL, TOUCH_DEPTH_COL, TOUCH_TOKEN_F,
                                   TOUCH_TOKENS, TOUCH_VALID_COL)
from mimo_bridge.touch_adapter import (ContactInfo, LEGACY_PUSH_SCALE, PatchSpec,
                                       PUSH_ARRIVAL_SCALE_CMS, PUSH_DEPTH_SCALE_CM, TouchPooler)


def test_the_divisors_put_this_bodys_99th_percentile_contact_near_one():
    assert PUSH_DEPTH_SCALE_CM == pytest.approx(0.1)
    assert PUSH_ARRIVAL_SCALE_CMS == pytest.approx(2.0)
    assert 0.108 / PUSH_DEPTH_SCALE_CM == pytest.approx(1.08, abs=0.01)
    assert 2.21 / PUSH_ARRIVAL_SCALE_CMS == pytest.approx(1.1, abs=0.01)
    assert LEGACY_PUSH_SCALE == (10.0, 500.0)
    # and the named columns are the two just before the validity flag
    assert (TOUCH_DEPTH_COL, TOUCH_ARRIVAL_COL, TOUCH_VALID_COL) == (16, 17, 18)


def test_the_pooler_writes_depth_and_arrival_on_the_new_scale_from_raw_units():
    """One millimetre deep and 2 cm/s: exactly 1.0 on each scale."""
    p = TouchPooler([PatchSpec(body_id=1, body_name="p", n_points=1, scale_m=0.01)], 0.01)
    c = ContactInfo(body_id=1, pair_key=(1, 2), normal_force=1.0, tangential_body=np.zeros(3),
                    pos_body=np.zeros(3), pos_world=np.zeros(3),
                    normal_world=np.array([0.0, 0.0, 1.0]), penetration_m=0.001,
                    rel_normal_speed=0.02)
    p.update({1: np.array([[0.0, 0.0, 1.0]])}, [c])
    assert float(p.last_contact[0, 3]) == pytest.approx(1.0)
    assert float(p.last_contact[0, 4]) == pytest.approx(1.0)


def _token(depth_value, arrival_value, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.zeros(1, TOUCH_TOKENS, TOUCH_TOKEN_F)
    t[0, :5, :13] = torch.rand(5, 13, generator=g)
    t[0, :5, TOUCH_DEPTH_COL] = depth_value
    t[0, :5, TOUCH_ARRIVAL_COL] = arrival_value
    t[0, :5, TOUCH_VALID_COL] = 1.0
    return t


def _receipt_dir(tmp_path, scale=None):
    d = tmp_path / "ckpt"
    d.mkdir()
    rec = {} if scale is None else {"touch_push_scale": list(scale)}
    (d / "model_receipt.json").write_text(json.dumps(rec), encoding="utf-8")
    return d


def test_a_checkpoint_trained_on_the_old_divisors_reads_every_token_exactly_as_before(tmp_path):
    """A contact 0.05 cm deep arriving at 1 cm/s. Under the old divisors it reached the encoder
    as 0.005 and 0.002; under the new ones as 0.5 and 0.5. After the migration the encoder must
    turn the new token into the same embedding it turned the old token into."""
    torch.manual_seed(0)
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False)
    enc = m.encoder_registry.encoders["tactile"]
    raw_depth_cm, raw_arrival_cms = 0.05, 1.0
    old_tok = _token(raw_depth_cm / LEGACY_PUSH_SCALE[0], raw_arrival_cms / LEGACY_PUSH_SCALE[1])
    new_tok = _token(raw_depth_cm / PUSH_DEPTH_SCALE_CM, raw_arrival_cms / PUSH_ARRIVAL_SCALE_CMS)
    with torch.no_grad():
        before = enc(old_tok).clone()
    m._migrate_touch_push_scale(str(_receipt_dir(tmp_path)))       # no stamp: the old scale
    with torch.no_grad():
        after = enc(new_tok)
    assert torch.allclose(before, after, atol=1e-5), float((before - after).abs().max())


def test_a_checkpoint_that_records_the_scale_in_force_is_left_alone(tmp_path):
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False)
    w = m.encoder_registry.encoders["tactile"].patch_mlp[0].weight
    before = w.detach().clone()
    m._migrate_touch_push_scale(str(_receipt_dir(
        tmp_path, (PUSH_DEPTH_SCALE_CM, PUSH_ARRIVAL_SCALE_CMS))))
    assert torch.equal(w.detach(), before)


def test_the_optimisers_averages_move_with_the_weights(tmp_path):
    """The gradient of a weight scales with its input, so the first moment is multiplied by k
    and the second by k squared, and every other column is untouched."""
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False)
    w = m.encoder_registry.encoders["tactile"].patch_mlp[0].weight
    st = m.optimizer.state[w]
    st["exp_avg"] = torch.ones_like(w)
    st["exp_avg_sq"] = torch.ones_like(w)
    st["step"] = torch.tensor(10.0)
    m._migrate_touch_push_scale(str(_receipt_dir(tmp_path)))
    k_depth = LEGACY_PUSH_SCALE[0] / PUSH_DEPTH_SCALE_CM
    k_arr = LEGACY_PUSH_SCALE[1] / PUSH_ARRIVAL_SCALE_CMS
    assert torch.allclose(st["exp_avg"][:, TOUCH_DEPTH_COL], torch.full_like(w[:, 0], k_depth))
    assert torch.allclose(st["exp_avg_sq"][:, TOUCH_ARRIVAL_COL],
                          torch.full_like(w[:, 0], k_arr ** 2))
    others = [c for c in range(TOUCH_TOKEN_F) if c not in (TOUCH_DEPTH_COL, TOUCH_ARRIVAL_COL)]
    assert torch.equal(st["exp_avg"][:, others], torch.ones_like(st["exp_avg"][:, others]))


def test_a_new_checkpoint_records_the_scale_it_was_trained_on():
    m = LemLite(device="cpu", seed=0, intention=False, imagination=False)
    assert m.receipt["touch_push_scale"] == [PUSH_DEPTH_SCALE_CM, PUSH_ARRIVAL_SCALE_CMS]
