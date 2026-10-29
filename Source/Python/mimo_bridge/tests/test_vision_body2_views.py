# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Body 2's face-down views through the vision encoders (docs/gain_of_function_v4/vision_body2_views.py).

Synthetic tokens, quaternions, weight files and tables only: no checkpoint, no world. They pin the eye-command grid,
the spread (0 for one token), the face-down reading of a head's orientation, the vision encoder's weights read by name
from a weight file, and the declared dating rule with its two controls."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def B():
    spec = importlib.util.spec_from_file_location("vision_body2_views", REPO / "docs/gain_of_function_v4/vision_body2_views.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_eye_grid_and_the_spread(B):
    g = B.gaze_grid()
    assert g.shape == (25, 2)
    assert np.degrees(g[:, 0]).min() == pytest.approx(-20.0) and np.degrees(g[:, 1]).max() == pytest.approx(15.0)
    same = torch.ones(25, 192)
    assert B.spread(same) == 0.0
    t = torch.zeros(4, 3)
    t[:, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert B.spread(t) == pytest.approx(float(torch.tensor([0.0, 1.0, 2.0, 3.0]).std()) / 3.0)
    views = dict(patches=np.stack([np.full((5, 768), k, dtype=np.float32) for k in range(3)]))
    assert B.input_spread(views) == pytest.approx(1.0)


def test_the_face_down_reading(B):
    assert B.head_facing_down(np.array([0.0, 0.0, 0.0, 1.0])) == pytest.approx(0.0)
    s = math.sin(math.radians(45.0))
    # turned 90 degrees about the world y axis, the local x axis points to world -z: face down
    assert B.head_facing_down(np.array([0.0, s, 0.0, s])) == pytest.approx(-1.0)
    assert B.head_facing_down(np.array([0.0, -s, 0.0, s])) == pytest.approx(1.0)


def test_the_vision_encoder_is_read_by_name(B, tmp_path):
    from safetensors.torch import save_file
    f = tmp_path / "model.safetensors"
    save_file({"encoder_registry.encoders.vision.pool.query": torch.ones(1, 4),
               "encoder_registry.encoders.vision.blocks.0.attn.in_proj_weight": torch.zeros(6, 2),
               "encoder_registry.encoders.motor.pool.query": torch.ones(1, 4)}, str(f))
    sd = B.vision_state_dict(f)
    assert set(sd) == {"pool.query", "blocks.0.attn.in_proj_weight"}
    save_file({"forward_model.x": torch.ones(1)}, str(tmp_path / "other.safetensors"))
    with pytest.raises(KeyError):
        B.vision_state_dict(tmp_path / "other.safetensors")


def table_for(enc, sources, b2, b0=0.4):
    """spread per (encoder, source, body): body 0 at b0; body 2 from b2(encoder, source)."""
    t = {}
    for e in enc:
        for s in sources:
            t[(e, s, 0)] = b0
            t[(e, s, 2)] = b2(e, s)
    return t


def test_the_dating_rule(B):
    enc = [75596, 79404, 83477, 87556, 92429, 96500, 123975, 411150]
    sources = {79404: dict(face_down=False), 83477: dict(face_down=True), 411150: dict(face_down=True)}
    # the encoder answers face-down views until 92,429 and is deaf to them from 96,500 on; face up always answered
    live_until = lambda e, s: 0.3 if (s == 79404 or e < 96500) else 0.05
    out = B.summarize(table_for(enc, sources, live_until), sources, enc)
    assert out["date"] == 96500 and out["views_not_encoder"] is False and not out["void"]
    # deaf again at one later encoder only after an answering one: the date is the start of the last deaf run
    flicker = lambda e, s: 0.05 if (s != 79404 and e in (87556, 96500, 411150)) else 0.3
    assert B.summarize(table_for(enc, sources, flicker), sources, enc)["date"] == 411150
    # even the encoders before 83,477 read the face-down views deaf: the views, not the encoder
    views = lambda e, s: 0.3 if s == 79404 else 0.05
    out = B.summarize(table_for(enc, sources, views), sources, enc)
    assert out["views_not_encoder"] is True and out["date"] == 75596
    # a body 0 set below 0.2 voids the date
    out = B.summarize(table_for(enc, sources, live_until, b0=0.1), sources, enc)
    assert out["void"] and out["date"] is None
    # body 2's own face-up views unanswered by the early encoders void it too
    out = B.summarize(table_for(enc, sources, lambda e, s: 0.05), sources, enc)
    assert out["void"] and not out["controls"]["body2_face_up_answered"]
