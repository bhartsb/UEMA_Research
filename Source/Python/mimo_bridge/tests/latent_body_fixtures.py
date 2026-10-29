# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Small synthetic fixtures. Their constants are not physical calibration evidence."""
import torch

from lem.latent_body.trajectory import Observation, Example
from lem.latent_body.tree import arm_hand_tree
from mimo_bridge.body_layout import names_for


def tree(levels=("arm", "hand", "index")):
    return arm_hand_tree(names_for("trunk_v1"), levels)


def config():
    return dict(architecture=dict(latent_dim=2, hidden=16, message_dim=2),
        scales=dict(state=[1.] * 12, change=[1.] * 6, action=[1.] * 8),
        weights=dict(future=1., contact=1., action=1., child=1., consistency=1., latent=.001),
        learning_rate=.003, gradient_clip=10., duration_s=.05, action_contract="world_v0",
        seed=7, measurement="synthetic unit fixture only", data_quality="plumbing", substrate_receipt="")


def observation(step, body=0, episode=0):
    bones = torch.zeros(26, 11)
    bones[:, 3] = 1
    bones[:, 10] = 1
    bones[:, 4] = torch.arange(26) * .1
    bones[2, 4] += step * .1
    bones[8, 4] -= step * .1
    contact = torch.zeros(26)
    contact[2] = step % 2
    return Observation(body, episode, step, step * .05, bones, contact)


def order(obs):
    action = torch.zeros(26, 9)
    action[:, :3] = obs.bones[:, 4:7]
    action[:, 3:7] = obs.bones[:, :4]
    action[:, 8] = .5
    action[2, 0] += .2
    return action


def example(step=1):
    history, start, end = observation(step - 1), observation(step), observation(step + 1)
    return Example(history, start, end, order(start), "world_v0")
