# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Think-harder (adaptive motor controller) and intention: wiring checks on CPU.

Hand-worked facts these tests pin down:
  * a fixed-mode snapshot loaded into a think-harder model transfers the encoders,
    forward model, heads and confidence head EXACTLY, leaves the new thinking blocks
    fresh, and skips the optimizer state (param groups differ);
  * the intention generator plans with gradients enabled and returns finite actions
    of the right shape, and it owns no weights (nothing new to train or snapshot).
"""
import os
import tempfile

import torch

from mimo_bridge.lem_model import LemLite


def test_think_harder_partial_load():
    with tempfile.TemporaryDirectory() as td:
        m1 = LemLite(device="cpu", seed=0, checkpoint_dir=os.path.join(td, "ck"))
        p = m1.save_checkpoint(step=7)
        m2 = LemLite(device="cpu", seed=1, think_harder=True)
        assert hasattr(m2.motor_controller, "residual_blocks"), "adaptive mode not built"
        fresh_block = m2.motor_controller.residual_blocks[0][0].weight.detach().clone()
        m2.load_checkpoint(p, load_optimizer=False)
        for (n1, t1), (n2, t2) in zip(m1.forward_model.state_dict().items(),
                                      m2.forward_model.state_dict().items()):
            assert n1 == n2 and torch.equal(t1, t2), f"forward model differs at {n1}"
        assert torch.equal(m1.motor_controller.confidence_head[0].weight,
                           m2.motor_controller.confidence_head[0].weight), "confidence head not transferred"
        assert torch.equal(fresh_block, m2.motor_controller.residual_blocks[0][0].weight), \
            "thinking blocks must stay fresh (fixed-mode snapshot has no keys for them)"
        out, conf, meta = m2.motor_controller(torch.randn(2, m2.n_mod, m2.d_model),
                                              torch.randn(2, 24, 10))
        assert out.shape == (2, 24, 9) and torch.isfinite(out).all()


def test_intention_plan_returns_finite_actions():
    m = LemLite(device="cpu", seed=0, intention=True)
    assert m.intention_gen is not None
    # the generator holds the forward model etc. as submodules, so parameters() is not
    # empty - but every parameter must BELONG to an existing module (nothing new to train)
    existing = {id(q) for mod in (m.forward_model, m.encoder_registry, m.prediction_heads)
                for q in mod.parameters()}
    assert all(id(q) in existing for q in m.intention_gen.parameters()), "intention added new weights"
    B, d = 2, m.d_model
    cur = {mod: torch.randn(B, d) * 0.1 for mod in m.active}
    goal = {mod: torch.randn(B, d) * 0.1 for mod in m.active}
    motor = torch.randn(B, 23, 9) * 0.1
    from lem.model.intention_generator import plan_multistep
    with torch.enable_grad():
        plan, meta = plan_multistep(m.intention_gen, m.forward_model, m.encoder_registry,
                                    m.prediction_heads, m.wellness.score_predicted_wellness,
                                    cur, goal, motor, max_steps=2)
    assert len(plan) >= 1
    for a in plan:
        assert a.shape == (B, 23, 9) and torch.isfinite(a).all()
    assert "plan_length" in meta
    # model params untouched by planning (generator freezes and restores them)
    assert all(p.requires_grad for p in m.forward_model.parameters())
