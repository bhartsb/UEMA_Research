# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Consequence through the command (config key consequence_from_command, Oct 1, 2026;
docs/gain_of_function_v4/agents/latent_neighbor_commands_2026-10-01.md). Each test names the defect it catches."""
import copy

import pytest
import torch

from lem.latent_body import checkpoint
from lem.latent_body.learning import ShadowLearner, command_inputs, validate_config
from lem.latent_body.losses import dynamics_loss
from lem.latent_body.tree import DIGITS, full_body_tree
from mimo_bridge.tests.test_latent_full_body_rows import NAMES, _example, config
from mimo_bridge.tools.latent_body_intervention import decode_with

DIGIT_PARTS = {f"{d}_{s}" for s in ("L", "R") for d in DIGITS}


def routed_config(groups=("digit",)):
    return dict(config(), consequence_from_command=list(groups))


def learner(cfg=None):
    return ShadowLearner(full_body_tree(NAMES), cfg or routed_config())


def batch(lrn, n=6):
    return lrn.batch([_example(seed) for seed in range(n)])


def test_absent_switch_keeps_the_single_decoder_and_its_parameters():
    """Defect: the default path changed (a consequence head or a different decoder size without the switch)."""
    plain = ShadowLearner(full_body_tree(NAMES), config())
    assert plain.model.routed == ()
    assert not any("consequence" in k for k in plain.model.state_dict())
    part = plain.model.parts["index_L"]
    assert part.command_inputs is None
    assert part.decoder[-1].out_features == sum(part.output_sizes)


def test_switch_routes_exactly_the_named_groups_and_reads_only_learned_command_numbers():
    """Defects: the switch applied to other parts; radius or strength (learned by nothing) fed to the consequence."""
    lrn = learner()
    assert set(lrn.model.routed) == DIGIT_PARTS
    inputs = command_inputs(lrn.tree, lrn.config)
    # under wrist_relative_joint_v1 a digit's position and turn are issued and executed; radius and strength are not
    assert all(v == [0, 1, 2, 3, 4, 5] for v in inputs.values())
    assert lrn.model.parts["hand_L"].command_inputs is None and lrn.model.parts["arm_L"].command_inputs is None


def test_declaration_is_checked():
    for bad in (["fingers"], [], "digit", ["digit", "digit"]):
        with pytest.raises(ValueError):
            validate_config(dict(config(), consequence_from_command=bad))
    no_rows = {k: v for k, v in routed_config().items() if k != "body_rows"}
    with pytest.raises(ValueError):
        validate_config(no_rows)
    validate_config(routed_config())


@torch.no_grad()
def test_the_code_moves_the_predicted_movement_only_through_its_command():
    """Defect: the consequence head reads the code (the decoupling this switch removes)."""
    lrn = learner()
    b = batch(lrn)
    part = lrn.model.parts["index_L"]
    ctx = lrn.model.context(b.state)["index_L"]
    s, h, c = b.state["index_L"], b.history["index_L"], ctx
    command = torch.randn(len(s), 8)
    za, zb = torch.zeros(len(s), lrn.model.latent_dim), torch.ones(len(s), lrn.model.latent_dim) * .9
    pa, pb = part.predict(s, h, c, za, command=command), part.predict(s, h, c, zb, command=command)
    assert torch.equal(pa.change, pb.change) and torch.equal(pa.contact_logits, pb.contact_logits)
    assert not torch.equal(pa.action, pb.action)
    # without a supplied command it reads the command its code decodes
    own = part.predict(s, h, c, za)
    assert torch.allclose(own.change, part.predict(s, h, c, za, command=own.action).change)
    # radius and strength do not move the prediction
    other = command.clone()
    other[:, 6:] += 5
    assert torch.equal(part.predict(s, h, c, za, command=other).change, pa.change)


@torch.no_grad()
def test_hindsight_route_reads_the_recorded_command_and_the_planning_route_the_decoded_one():
    """Defects: the hindsight route reading the decoded command; the planning route reading the recorded one."""
    lrn = learner()
    b = batch(lrn)
    z = lrn.model.encode(b)
    local = lrn.model.reconstruct(b, z)
    ctx = lrn.model.context(b.state)
    part = lrn.model.parts["middle_R"]
    recorded = part.predict(b.state["middle_R"], b.history["middle_R"], ctx["middle_R"], z["middle_R"],
                            command=b.action["middle_R"])
    decoded = part.predict(b.state["middle_R"], b.history["middle_R"], ctx["middle_R"], z["middle_R"])
    assert torch.equal(local["middle_R"].change, recorded.change)
    assert not torch.allclose(local["middle_R"].change, decoded.change)
    planned, _ = lrn.model.decode(b.state, b.history, {r: z[r] for r in lrn.tree.roots})
    again = decode_with(lrn.model, b.state, b.history, {r: z[r] for r in lrn.tree.roots}, {})
    for name in DIGIT_PARTS:
        assert torch.equal(planned[name].change, again[name].change)
        cmd = planned[name].action
        ref = lrn.model.parts[name].predict(b.state[name], b.history[name], ctx[name],
                                            torch.zeros_like(z[name]), command=cmd)
        assert torch.allclose(planned[name].change, ref.change, atol=1e-6)


def test_consequence_lessons_never_reshape_the_decoded_command():
    """Defect: the decoded command not detached, so the future lesson pulls the command off the recorded order."""
    lrn = learner()
    b = batch(lrn)
    z = {k: v.detach() for k, v in lrn.model.encode(b).items()}
    planned, _ = lrn.model.decode(b.state, b.history, {r: z[r] for r in lrn.tree.roots})
    lrn.model.zero_grad(set_to_none=True)
    (planned["ring_L"].change.square().sum() + planned["ring_L"].contact_logits.sum()).backward()
    head = lrn.model.parts["ring_L"].decoder
    assert all(p.grad is None or not p.grad.any() for p in head.parameters())
    assert any(p.grad is not None and p.grad.any() for p in lrn.model.parts["ring_L"].consequence.parameters())


def test_learning_and_the_checkpoint_round_trip_with_the_switch(tmp_path):
    """Defects: the switch lost on reload (the configuration fingerprint or the model shape differs)."""
    lrn = learner()
    examples = [_example(seed) for seed in range(4)]
    before = copy.deepcopy(lrn.model.parts["pinky_R"].consequence.state_dict())
    lrn.learn(examples)
    assert any(not torch.equal(before[k], v) for k, v in lrn.model.parts["pinky_R"].consequence.state_dict().items())
    checkpoint.save(tmp_path / "h.pt", lrn)
    from mimo_bridge.tools.latent_body import load_learner
    again = load_learner(tmp_path / "h.pt")
    assert set(again.model.routed) == DIGIT_PARTS
    b = again.batch(examples)
    with torch.no_grad():
        total_a, _, _ = dynamics_loss(lrn.model, b, lrn.config["weights"])
        total_b, _, _ = dynamics_loss(again.model, b, again.config["weights"])
    assert torch.allclose(total_a, total_b)
    plain = config()
    with pytest.raises(ValueError):
        ShadowLearner(full_body_tree(NAMES), plain).load_state_dict(lrn.state_dict())
