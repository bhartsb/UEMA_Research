# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A checkpoint whose core has a different block count or width must be refused
by name, not loaded with random weights and a warning.

Measured 2026-09-16 against the real build_config: a checkpoint saved with 2
core blocks loaded into a model built with 8 gave 150 missing keys and NO
exception - six blocks kept their fresh random weights and the only trace was one
warning line. Reversed, six trained blocks were silently dropped. A width change
re-randomised 52 of 54 core tensors the same way. The context length had a guard
beside this spot; block count and width did not.

core_shape_mismatch is a pure function over the checkpoint's key list and a
shape getter, so it is tested here without building a model or a checkpoint.
"""
import pytest

from mimo_bridge.lem_model import core_shape_mismatch


def _keys(n_blocks, d_model, ctx=128):
    """The forward_model tensor names a checkpoint carries, with their shapes."""
    shapes = {
        "forward_model.modality_embedding.weight": (6, d_model),
        "forward_model.temporal_embedding.weight": (ctx, d_model),
        "forward_model.final_norm.weight": (d_model,),
        "encoder_registry.encoders.vision.proj.weight": (d_model, 128),
    }
    for i in range(n_blocks):
        shapes[f"forward_model.blocks.{i}.cm_attn.out_proj.weight"] = (d_model, d_model)
        shapes[f"forward_model.blocks.{i}.ffn.3.weight"] = (d_model, 4 * d_model)
        shapes[f"forward_model.blocks.{i}.causal_mask"] = (ctx, ctx)
    return shapes


def test_matching_core_passes_silently():
    shapes = _keys(2, 128)
    assert core_shape_mismatch(shapes.keys(), shapes.__getitem__, n_blocks=2, d_model=128) is None


def test_more_blocks_in_the_model_than_the_checkpoint_is_refused():
    shapes = _keys(2, 128)
    msg = core_shape_mismatch(shapes.keys(), shapes.__getitem__, n_blocks=8, d_model=128)
    assert msg and "2 core blocks" in msg and "built with 8" in msg, msg
    # the message must say what would have happened, so the reader knows why it stopped
    assert "random" in msg


def test_fewer_blocks_in_the_model_than_the_checkpoint_is_refused():
    shapes = _keys(8, 128)
    msg = core_shape_mismatch(shapes.keys(), shapes.__getitem__, n_blocks=2, d_model=128)
    assert msg and "8 core blocks" in msg and "built with 2" in msg, msg
    assert "discard" in msg


def test_a_different_width_is_refused_by_name():
    shapes = _keys(2, 128)
    msg = core_shape_mismatch(shapes.keys(), shapes.__getitem__, n_blocks=2, d_model=256)
    assert msg and "128 wide" in msg and "built 256 wide" in msg, msg


def test_a_checkpoint_with_no_core_is_not_this_guard_s_problem():
    """Nothing to compare against: leave it to the loader's own missing-key handling."""
    shapes = {"encoder_registry.encoders.vision.proj.weight": (128, 128)}
    assert core_shape_mismatch(shapes.keys(), shapes.__getitem__, n_blocks=2, d_model=128) is None


def test_block_indices_are_counted_not_summed_from_key_count():
    """Each block carries many tensors; the guard must count DISTINCT block indices."""
    shapes = _keys(3, 128)
    # 3 blocks x 3 tensors each = 9 block keys, but 3 blocks
    assert sum(1 for k in shapes if ".blocks." in k) == 9
    assert core_shape_mismatch(shapes.keys(), shapes.__getitem__, n_blocks=3, d_model=128) is None
    assert core_shape_mismatch(shapes.keys(), shapes.__getitem__, n_blocks=9, d_model=128) is not None
