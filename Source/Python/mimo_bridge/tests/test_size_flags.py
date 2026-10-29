# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The model-size flags build the shapes they name, and no flags builds today's model.

Until 2026-09-17 every model size was a literal in LITE_OVERRIDES and no run
recorded its depth. size_overrides is the pure mapping from the six flags to
config overrides; these tests drive it through build_config and read the shapes
back off the resolved config and the built modules, on CPU.
"""
import types

import pytest

from mimo_bridge.lem_model import build_config
from mimo_bridge.train_mimo import size_overrides


def _args(**kw):
    base = dict(core_blocks=0, d_model=0, vision_swin_layers=0, vision_width=0,
                encoder_layers=0, encoder_width=0)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_no_flags_means_no_overrides_and_todays_model():
    assert size_overrides(_args()) == {}
    cfg = build_config({"model": {"context_length": 128}})
    assert cfg["model"]["d_model"] == 128
    assert cfg["model"]["forward_model"]["n_layers"] == 2
    assert cfg["modalities"]["vision"]["encoder"]["n_swin_layers"] == 2
    assert cfg["modalities"]["tactile"]["encoder"]["n_cross_attn_layers"] == 1


def test_the_proposed_configuration_resolves_to_the_measured_shapes():
    """core 8 x 256, vision 12 x 256, four sense encoders 4 x 256, intermind untouched -
    the configuration tools/capacity_plan.py measured at 32.3 M parameters."""
    ov = size_overrides(_args(core_blocks=8, d_model=256, vision_swin_layers=12,
                              vision_width=256, encoder_layers=4, encoder_width=256))
    ov.setdefault("model", {})["context_length"] = 128
    cfg = build_config(ov)
    assert cfg["model"]["d_model"] == 256
    assert cfg["model"]["forward_model"]["n_layers"] == 8
    v = cfg["modalities"]["vision"]["encoder"]
    assert (v["n_swin_layers"], v["d_internal"], v["angular_pos_hidden"]) == (12, 256, 256)
    for name in ("tactile", "motor"):
        e = cfg["modalities"][name]["encoder"]
        assert (e["n_cross_attn_layers"], e["d_internal"], e["per_patch_hidden"]) == (4, 256, 256)
    assert cfg["modalities"]["audio"]["encoder"]["conv_channels"] == [256] * 6
    assert cfg["modalities"]["proprioception"]["encoder"]["hidden_dims"] == [256] * 5
    # intermind is placeholder noise with loss weight 0.0; the flags must not grow it
    assert cfg["modalities"]["intermind"]["encoder"]["n_self_attn_layers"] == 1
    assert cfg["modalities"]["intermind"]["encoder"]["d_internal"] == 128


def test_the_built_core_has_the_requested_depth_and_width():
    from lem.model.forward_model import ForwardModel
    ov = size_overrides(_args(core_blocks=3, d_model=192))
    ov.setdefault("model", {})["context_length"] = 16
    fm = ForwardModel(build_config(ov))
    assert len(fm.blocks) == 3
    assert fm.modality_embedding.weight.shape[1] == 192


def test_each_flag_acts_alone():
    assert size_overrides(_args(core_blocks=4)) == {"model": {"forward_model": {"n_layers": 4}}}
    assert size_overrides(_args(vision_swin_layers=4)) == {"modalities": {"vision": {"encoder": {"n_swin_layers": 4}}}}
    ov = size_overrides(_args(encoder_width=256))
    assert ov["modalities"]["tactile"]["encoder"] == {"d_internal": 256, "per_patch_hidden": 256}
    assert ov["modalities"]["audio"]["encoder"]["conv_channels"] == [256] * 3      # layers left at 1
    assert ov["modalities"]["proprioception"]["encoder"]["hidden_dims"] == [256] * 2


def test_size_flags_are_part_of_the_acting_fingerprint():
    """A size flag changed on a resume must be refused, not silently applied: the
    fingerprint covers every flag not in train_mimo's `operational` set."""
    import inspect
    from mimo_bridge import train_mimo as TM
    src = inspect.getsource(TM)
    op_start = src.index("operational={")
    op_block = src[op_start:src.index("}", op_start)]
    for flag in ("core_blocks", "d_model", "vision_swin_layers", "vision_width",
                 "encoder_layers", "encoder_width"):
        assert flag not in op_block, f"{flag} is listed as operational and would escape the fingerprint"
