# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""--proprio-heading free (mimo_bridge/proprio_heading.py): the proprioceptive input without the crib heading.

What is pinned: turning the whole body about the vertical by any angle leaves the turned row unchanged (float32
rounding); gravity-relative orientation is kept (a tilt is not removed); the lower trunk's own heading becomes 0;
the gaze row's quaternion, fatigue and the row index are untouched; a full turn (every world quaternion back as
its negative) gives the same row; the
default build installs nothing and stamps nothing; with the flag every path through the encoder reads the turned
row (the hook); a checkpoint trained with it refuses a build without it, and a build with it on an old checkpoint
logs and stamps the switch; the trainer flag."""
import inspect
import math

import pytest
import torch

from mimo_bridge import proprio_heading as PH


def _qmul(a, b):
    """Hamilton product, (x, y, z, w)."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw,
                        aw * bw - ax * bx - ay * by - az * bz], -1)


def _yaw(alpha):
    return torch.tensor([0.0, 0.0, math.sin(alpha / 2), math.cos(alpha / 2)])


def _rotz(v, alpha):
    c, s = math.cos(alpha), math.sin(alpha)
    x, y, z = v.unbind(-1)
    return torch.stack([c * x - s * y, s * x + c * y, z], -1)


def _body(seed=0, n=5):
    """n rows of a 26-bone body in the input's layout, with unit quaternions and a lying trunk."""
    g = torch.Generator().manual_seed(seed)
    r = torch.zeros(n, 26, 12)
    r[..., 0:4] = torch.nn.functional.normalize(torch.randn(n, 26, 4, generator=g), dim=-1)
    r[..., 4:7] = 0.2 * torch.randn(n, 26, 3, generator=g)
    r[..., 4:7] -= r[:, 0:1, 4:7]                                     # relative to row 0 (the left wrist)
    r[..., 7:10] = 0.03 * torch.randn(n, 26, 3, generator=g)
    r[..., 10] = torch.rand(n, 26, generator=g)
    r[..., 11] = torch.arange(26) / 25.0
    return r.double()


def _turn_world(r, alpha):
    """The same body turned by alpha about the world vertical (what lying another way round in the crib does)."""
    out = r.clone()
    q = _yaw(alpha).double().expand_as(r[..., 0:4])
    rows = [i for i in range(26) if i != PH.GAZE_ROW]                # the gaze row is head-relative
    out[:, rows, 0:4] = _qmul(q[:, rows], r[:, rows, 0:4])
    out[..., 4:7] = _rotz(r[..., 4:7], alpha)
    out[..., 7:10] = _rotz(r[..., 7:10], alpha)
    return out


@pytest.mark.parametrize("alpha", [0.3, -1.9, math.pi, 2.7])
def test_turning_the_body_about_the_vertical_leaves_the_row_unchanged(alpha):
    r = _body()
    a = PH.heading_free(r.reshape(5, -1))
    b = PH.heading_free(_turn_world(r, alpha).reshape(5, -1))
    qa, qb = a.view(5, 26, 12)[..., 0:4], b.view(5, 26, 12)[..., 0:4]
    # a quaternion and its negative are one rotation: compare up to sign, row by row
    sign = torch.sign((qa * qb).sum(-1, keepdim=True))
    assert torch.allclose(qa, qb * sign, atol=1e-9)
    assert torch.allclose(a.view(5, 26, 12)[..., 4:], b.view(5, 26, 12)[..., 4:], atol=1e-9)
    # float32, the model's precision
    a32 = PH.heading_free(r.float().reshape(5, -1))
    b32 = PH.heading_free(_turn_world(r, alpha).float().reshape(5, -1))
    q32a, q32b = a32.view(5, 26, 12)[..., 0:4], b32.view(5, 26, 12)[..., 0:4]
    s32 = torch.sign((q32a * q32b).sum(-1, keepdim=True))
    assert float((q32a - q32b * s32).abs().max()) < 1e-5
    assert float((a32.view(5, 26, 12)[..., 4:] - b32.view(5, 26, 12)[..., 4:]).abs().max()) < 1e-5


def test_the_trunk_ends_at_heading_zero_and_tilt_is_kept():
    r = _body()
    out = PH.heading_free(r.reshape(5, -1)).view(5, 26, 12)
    assert torch.allclose(PH.heading(out[:, PH.TRUNK_ROW, 0:4]), torch.zeros(5, dtype=r.dtype), atol=1e-9)
    # a tilt about a horizontal axis is not a heading: the world-up component of every bone's axes is kept
    def up(q):
        x, y, z, w = q.unbind(-1)
        return torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
    assert torch.allclose(up(out[..., 0:4]), up(r[..., 0:4]), atol=1e-9)
    # vectors keep their size and their vertical part
    assert torch.allclose(out[..., 4:7].norm(dim=-1), r[..., 4:7].norm(dim=-1), atol=1e-9)
    assert torch.allclose(out[..., 6], r[..., 6]) and torch.allclose(out[..., 9], r[..., 9])


def test_the_gaze_quaternion_fatigue_and_row_index_are_untouched_and_sign_does_not_matter():
    r = _body()
    out = PH.heading_free(r.reshape(5, -1)).view(5, 26, 12)
    assert torch.equal(out[:, PH.GAZE_ROW, 0:4], r[:, PH.GAZE_ROW, 0:4])
    assert torch.equal(out[..., 10:], r[..., 10:])
    # a full turn about the vertical brings every world quaternion back as its negative (the double cover),
    # while the head-relative gaze row is unchanged: the turned row must come back exactly
    flipped = r.clone()
    rows = [i for i in range(26) if i != PH.GAZE_ROW]
    flipped[:, rows, 0:4] *= -1
    o2 = PH.heading_free(flipped.reshape(5, -1)).view(5, 26, 12)
    assert torch.allclose(o2, out, atol=1e-12)
    assert torch.allclose(PH.heading_free(_turn_world(r, 2 * math.pi).reshape(5, -1)).view(5, 26, 12), out,
                          atol=1e-9)
    with pytest.raises(ValueError):
        PH.heading_free(torch.zeros(3, 24 * 12))                     # the legacy body has no trunk row


def _lite(**kw):
    torch.set_num_threads(1)
    from mimo_bridge.lem_model import LemLite
    return LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                   body_layout="trunk_v1", **kw)


def test_the_default_build_installs_nothing_and_the_flag_turns_every_encoder_call():
    off = _lite()
    on = _lite(proprio_heading="free")
    assert off.proprio_heading == "off" and off.proprio_heading_hook is None
    assert "proprio_heading" not in off.receipt and on.receipt["proprio_heading"]["version"] == PH.VERSION
    assert on.n_params == off.n_params, "no parameter"
    # the same weights: the flag's encoder output equals the plain encoder's on the turned row
    on.encoder_registry.load_state_dict(off.encoder_registry.state_dict())
    x = _body(n=4).float().reshape(4, -1)
    t_on = on.encoder_registry.encode({"proprioception": x}, {"proprioception": "real"})["proprioception"]
    t_off = off.encoder_registry.encode({"proprioception": PH.heading_free(x)},
                                        {"proprioception": "real"})["proprioception"]
    assert torch.equal(t_on, t_off)
    # and a body turned about the vertical gives the same token under the flag
    t_turn = on.encoder_registry.encode({"proprioception": _turn_world(_body(n=4), 1.3).float().reshape(4, -1)},
                                        {"proprioception": "real"})["proprioception"]
    assert float((t_turn - t_on).abs().max()) < 1e-4
    with pytest.raises(ValueError, match="proprio-heading"):
        _lite(proprio_heading="rotated")
    with pytest.raises(ValueError, match="trunk_v1"):
        from mimo_bridge.lem_model import LemLite
        LemLite(device="cpu", seed=0, imagination=False, no_vision=True, next_frame_loss=True,
                proprio_heading="free")
    # the hook handle is public: the episode companion never tries to carry it
    from mimo_bridge import continuation_state as CS
    assert "proprio_heading_hook" not in CS.capture_learner(on, include_replay=False)["data"]


def test_the_switch_is_logged_and_stamped_and_a_turned_checkpoint_refuses_the_plain_build(tmp_path, caplog):
    plain = _lite(checkpoint_dir=str(tmp_path / "plain"))
    p_plain = plain.save_checkpoint(3)
    on = _lite(proprio_heading="free", checkpoint_dir=str(tmp_path / "on"))
    with caplog.at_level("WARNING"):
        on.load_checkpoint(p_plain)
    assert on.proprio_heading_switch and "switched_at_load" in on.receipt["proprio_heading"]
    assert any("proprioception heading" in r.message for r in caplog.records)
    p_on = on.save_checkpoint(4)
    again = _lite(proprio_heading="free")
    again.load_checkpoint(p_on)
    assert not again.proprio_heading_switch, "a turned checkpoint loads into a turned build: no switch"
    with pytest.raises(ValueError, match="proprio-heading free"):
        _lite().load_checkpoint(p_on)


def test_the_trainer_flag_defaults_off_and_reaches_the_learner():
    from mimo_bridge import train_mimo as TM
    assert TM.parse_args(["--out", "x"]).proprio_heading == "off"
    b = TM.parse_args(["--out", "x", "--proprio-heading", "free"])
    assert b.proprio_heading == "free"
    src = inspect.getsource(TM)
    assert 'proprio_heading=getattr(args, "proprio_heading", "off") or "off"' in src
    cont = src.split("continuation_flags = (")[1].split(")")[0]
    assert "proprio_heading" not in cont
    assert "proprio_heading" in TM.acting_configuration(b), "it changes what the model reads"
