# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Top-down intentions, local consequences, and compact state-only peer messages."""
from __future__ import annotations

import torch
from torch import nn

from .model import PredictiveBodyModel

STATE_DIM = 12  # relative cm xyz, world quat xyzw, cm/s xyz, capacity, contact
CHANGE_DIM = 6  # world displacement cm xyz and rotation vector radians xyz
ACTION_DIM = 8  # target displacement xyz, turn vector xyz, radius, strength


class BodyHierarchy(nn.Module):
    def __init__(self, tree, *, latent_dim, hidden, message_dim, code_bound="none", command_inputs=None):
        """command_inputs: {part: indices of its learned command numbers} for the parts whose predicted consequence
        is read from their command, not their code (model.PredictiveBodyModel; learning.command_inputs). None or a
        part left out = the earlier single decoder."""
        super().__init__()
        if min(latent_dim, hidden, message_dim) < 1:
            raise ValueError("Architecture dimensions must be positive")
        self.tree = tree
        self.latent_dim = int(latent_dim)
        self.message_dim = int(message_dim)
        command_inputs = dict(command_inputs or {})
        if set(command_inputs) - set(tree.by_name):
            raise ValueError("Consequence-through-command parts must be parts of the tree")
        self.routed = tuple(sorted(command_inputs))
        self.parts = nn.ModuleDict()
        self.messages = nn.ModuleDict()
        for node in tree.nodes:
            n = len(tree.subtree[node.name])
            self.messages[node.name] = nn.Linear(n * STATE_DIM, message_dim)
            self.parts[node.name] = PredictiveBodyModel(
                n * STATE_DIM, len(tree.owned[node.name]) * ACTION_DIM, n * CHANGE_DIM, n,
                latent_dim=latent_dim, hidden=hidden, context_dim=len(node.peers) * message_dim,
                children={child: latent_dim for child in tree.children[node.name]}, code_bound=code_bound,
                command_inputs=command_inputs.get(node.name))

    def context(self, states):
        messages = {name: torch.tanh(layer(states[name])) for name, layer in self.messages.items()}
        return {n.name: torch.cat([messages[p] for p in n.peers], -1) if n.peers
                else states[n.name].new_empty((len(states[n.name]), 0)) for n in self.tree.nodes}

    def encode(self, batch):
        contexts = self.context(batch.state)
        return {n.name: self.parts[n.name].encode(batch.state[n.name], batch.history[n.name],
                    contexts[n.name], batch.action[n.name], batch.change[n.name], batch.contact[n.name])
                for n in self.tree.nodes}

    def decode(self, state, history, roots):
        if set(roots) != set(self.tree.roots):
            raise ValueError("One intention is required for each root")
        contexts = self.context(state)
        intentions, predictions = dict(roots), {}
        for node in self.tree.nodes:
            pred = self.parts[node.name].predict(state[node.name], history[node.name],
                                               contexts[node.name], intentions[node.name])
            predictions[node.name] = pred
            intentions.update(pred.children)
        return predictions, intentions

    def reconstruct(self, batch, latents):
        """The hindsight route. A part whose consequence is read from its command reads the RECORDED command here
        (the executed cause, known before the step); the decoded route (decode) reads the command its code decodes,
        as the planner does."""
        contexts = self.context(batch.state)
        return {n.name: self.parts[n.name].predict(batch.state[n.name], batch.history[n.name],
                    contexts[n.name], latents[n.name],
                    **({"command": batch.action[n.name]} if n.name in self.routed else {}))
                for n in self.tree.nodes}
