# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A consequence-conditioned encoder and a predictive local decoder."""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LocalPrediction:
    change: torch.Tensor
    contact_logits: torch.Tensor
    action: torch.Tensor
    children: dict[str, torch.Tensor]


def mlp(inputs, hidden, outputs):
    # SMOOTH LOCAL DYNAMICS (2026-09-26). SiLU keeps a derivative on negative
    # inputs; no token normalisation divides physical variation by a pose offset.
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class PredictiveBodyModel(nn.Module):
    """command_inputs (CONSEQUENCE THROUGH THE COMMAND, Oct 1, 2026; None = the earlier single decoder): the indices of
    the action numbers that are learned commands. With them the decoder splits in two. The command head reads the
    part's state, recent change, peers' state messages and code and gives the command and the children's codes; the
    consequence head reads the same state inputs and the command numbers only, never the code, and gives the predicted
    change and contact. A code then moves the predicted movement only through the command it decodes.
    Measured on the whole-body fit of Oct 1 (agents/latent_neighbor_commands_2026-10-01.md): with one decoder, codes in
    the causal test's range (a box of 2 standard deviations per dimension) lie a median 4.9-26.6 Mahalanobis units from
    the codes the encoder produces (2.2-2.5 for those codes), and there the predicted digit movement and the decoded
    command agree 0.21-0.47 (0.43-0.71 on the encoder's own codes)."""
    def __init__(self, state_dim, action_dim, outcome_dim, contact_dim, *, latent_dim,
                 hidden, context_dim, children, code_bound="none", command_inputs=None):
        super().__init__()
        if code_bound not in ("none", "tanh"):
            raise ValueError("Unknown code bound")
        self.code_bound = code_bound
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.children_dims = dict(children)
        conditioned = 2 * state_dim + context_dim
        self.encoder = mlp(conditioned + action_dim + outcome_dim + contact_dim, hidden, latent_dim)
        self.output_sizes = (outcome_dim, contact_dim, action_dim)
        self.command_inputs = None
        if command_inputs is None:
            self.decoder = mlp(conditioned + latent_dim, hidden,
                               outcome_dim + contact_dim + action_dim + sum(children.values()))
            return
        command_inputs = sorted({int(i) for i in command_inputs})
        if not command_inputs or command_inputs[0] < 0 or command_inputs[-1] >= action_dim:
            raise ValueError("The consequence head needs at least one learned command number of this part")
        self.command_inputs = command_inputs
        self.decoder = mlp(conditioned + latent_dim, hidden, action_dim + sum(children.values()))
        self.consequence = mlp(conditioned + len(command_inputs), hidden, outcome_dim + contact_dim)

    def encode(self, state, history, context, action, change, contact):
        code = self.encoder(torch.cat((state, history, context, action, change, contact), -1))
        # BOUNDED CODES (review of Sep 30, 2026): unbounded, H84's codes grew to a typical size of 43 per
        # dimension and the parent-predicts-child term made up 97.8% of the loss; tanh keeps every code in (-1, 1).
        return torch.tanh(code) if self.code_bound == "tanh" else code

    def predict(self, state, history, context, latent, command=None):
        """command (consequence head only): the normalized command row the consequence head reads, given by the
        caller (the recorded command in the hindsight lesson, hierarchy.reconstruct); None = the command this code
        decodes, detached so the consequence lessons never reshape the command (the planner's path)."""
        if self.command_inputs is None:
            if command is not None:
                raise ValueError("This part predicts its consequence from its code; it reads no command")
            values = self.decoder(torch.cat((state, history, context, latent), -1))
            sections = values.split((*self.output_sizes, *self.children_dims.values()), -1)
            return LocalPrediction(*sections[:3], dict(zip(self.children_dims, sections[3:])))
        outcome_dim, contact_dim, action_dim = self.output_sizes
        values = self.decoder(torch.cat((state, history, context, latent), -1))
        sections = values.split((action_dim, *self.children_dims.values()), -1)
        action = sections[0]
        source = action.detach() if command is None else command
        if source.shape != action.shape:
            raise ValueError("The consequence head reads this part's whole command row")
        change, contact = self.consequence(torch.cat((state, history, context, source[..., self.command_inputs]),
                                                     -1)).split((outcome_dim, contact_dim), -1)
        return LocalPrediction(change, contact, action, dict(zip(self.children_dims, sections[1:])))
