# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A CPU-only shadow optimizer; never part of the production optimizer or graph."""
from __future__ import annotations

import copy
import math

import torch

from .checkpoint import VERSION, fingerprint, validate_header
from .hierarchy import BodyHierarchy
from .losses import dynamics_loss, validate_weights
from .trajectory import TrajectoryExtractor, check_body_tree, make_batch, row_options, validate_scales


def finite_state(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_state(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def validate_config(config):
    from mimo_bridge.transition import known_action_contract
    required = {"architecture", "scales", "weights", "learning_rate", "gradient_clip",
                "duration_s", "action_contract", "seed", "measurement", "data_quality",
                "substrate_receipt"}
    # OPTIONAL SWITCHES (review of Sep 30, 2026): absent = the earlier meaning, so an existing hierarchy's
    # configuration and fingerprint are unchanged. code_bound "tanh" bounds the codes; hand_action_frame
    # "actual_wrist" learns finger commands on the wrist the body actually had (trajectory.move_digits).
    # digit_frame "wrist" and digit_scales: the index-level repair of Sep 30, 2026 (trajectory.DIGIT_FRAMES).
    # body_rows: the whole-body row convention of Oct 1, 2026 (trajectory.BodyRows).
    # horizon_steps: the hold horizon of Oct 1, 2026 (trajectory.hold_example): each example's end lies that many
    # model steps after its start; absent = 1.
    # consequence_from_command: part groups (trajectory.PART_GROUPS) whose parts predict their consequence from their
    # command, not their code (command_inputs; model.PredictiveBodyModel); absent = none, the earlier meaning.
    optional = {"code_bound", "hand_action_frame", "digit_frame", "digit_scales", "body_rows", "horizon_steps",
                "consequence_from_command"}
    if not required <= set(config) or set(config) - required - optional:
        raise ValueError(f"Hierarchy configuration requires {sorted(required)}, optionally {sorted(optional)}")
    if config.get("code_bound", "none") not in ("none", "tanh"):
        raise ValueError("code_bound must be none or tanh")
    from .trajectory import check_digit_frame, check_hand_frame
    check_hand_frame(config.get("hand_action_frame", "recorded"), config["action_contract"])
    check_digit_frame(config.get("digit_frame", "world"), config.get("hand_action_frame", "recorded"),
                      config.get("digit_scales"))
    from .trajectory import body_rows, check_horizon
    body_rows(config)
    check_horizon(config.get("horizon_steps", 1))
    check_consequence_groups(config)
    if not config["measurement"] or config["data_quality"] not in ("plumbing", "post_repair"):
        raise ValueError("Declare measurement provenance and data quality")
    if config["data_quality"] == "post_repair" and not config["substrate_receipt"]:
        raise ValueError("Post-repair data requires its substrate qualification receipt")
    if not known_action_contract(config["action_contract"]):
        raise ValueError("Unknown executed action contract")
    architecture = config["architecture"]
    if set(architecture) != {"latent_dim", "hidden", "message_dim"}:
        raise ValueError("Declare all architecture dimensions")
    if any(type(v) is not int or v <= 0 for v in architecture.values()):
        raise ValueError("Architecture dimensions must be positive integers")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**63:
        raise ValueError("Invalid private seed")
    for name in ("learning_rate", "gradient_clip", "duration_s"):
        value = float(config[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be measured, positive and finite")
    validate_scales(config["scales"])
    validate_weights(config["weights"])


def check_consequence_groups(config):
    groups = config.get("consequence_from_command")
    if groups is None:
        return
    from .trajectory import PART_GROUPS
    if (not isinstance(groups, list) or not groups or len(set(groups)) != len(groups)
            or not set(groups) <= set(PART_GROUPS)):
        raise ValueError(f"consequence_from_command lists part groups of {sorted(PART_GROUPS)}")
    if config.get("body_rows") is None:
        raise ValueError("consequence_from_command needs the whole-body rows (body_rows), which name the learned "
                         "command numbers")


def command_inputs(tree, config):
    """{part: indices of its learned command numbers (the action weights of trajectory.body_lesson_weights)} for every
    part all of whose handles are in a group of consequence_from_command; {} when the key is absent."""
    groups = config.get("consequence_from_command")
    if not groups:
        return {}
    from .trajectory import GROUP_OF, body_lesson_weights, body_rows
    weights = body_lesson_weights(tree, body_rows(config))
    out = {}
    for node in tree.nodes:
        if node.handles and all(GROUP_OF.get(h) in groups for h in node.handles):
            out[node.name] = [int(i) for i in weights[node.name][1].nonzero().flatten()]
    if not out:
        raise ValueError("consequence_from_command names no part of this tree")
    return out


class ShadowLearner:
    def __init__(self, tree, config):
        validate_config(config)
        self.rows = row_options(config)
        check_body_tree(tree, self.rows["body"])
        self.tree, self.config = tree, copy.deepcopy(config)
        # PRIVATE RANDOMNESS (2026-09-26). Do not call torch.manual_seed here: it
        # seeds CUDA too. CPU construction is bracketed; subsequent samples use
        # this learner's explicit CPU generator. Production's streams are untouched.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config["seed"])
            self.model = BodyHierarchy(tree, **config["architecture"],
                                       code_bound=config.get("code_bound", "none"),
                                       command_inputs=command_inputs(tree, config)).cpu()
        self.generator = torch.Generator(device="cpu").manual_seed(config["seed"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.extractor = TrajectoryExtractor(tree, duration_s=config["duration_s"],
                                             action_contract=config["action_contract"])
        self.updates = 0
        self.examples = 0

    def batch(self, examples):
        return make_batch(self.tree, examples, scales=self.config["scales"],
                          duration_s=self.config["duration_s"], action_contract=self.config["action_contract"],
                          hand_frame=self.config.get("hand_action_frame", "recorded"),
                          horizon_steps=self.config.get("horizon_steps", 1), **self.rows)

    def learn(self, examples, *, train=True):
        batch = self.batch(examples)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss, terms, _ = dynamics_loss(self.model, batch, self.config["weights"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite hierarchy loss; no update applied")
            metrics = {name: float(value.detach()) for name, value in terms.items()}
            if train:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clip"],
                                                     error_if_nonfinite=True)
                # A failed update keeps its last finite model AND Adam state.
                before = copy.deepcopy((self.model.state_dict(), self.optimizer.state_dict()))
                try:
                    self.optimizer.step()
                    if not finite_state((self.model.state_dict(), self.optimizer.state_dict())):
                        raise FloatingPointError("Non-finite hierarchy update")
                except Exception as error:
                    self.model.load_state_dict(before[0])
                    self.optimizer.load_state_dict(before[1])
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("Failed hierarchy update; restored last finite state") from error
                self.updates += 1
                self.examples += len(examples)
                metrics["gradient_norm_before_clip"] = float(norm)
        self.optimizer.zero_grad(set_to_none=True)
        return {"type": "latent_body_lesson", "authority": "none", "updates": self.updates,
                "examples": self.examples, "batch_examples": len(examples),
                "data_quality": self.config["data_quality"], "loss_before_update": metrics,
                "interpretation": "hindsight reconstruction; physical intervention is scored separately"}

    def state_dict(self):
        return copy.deepcopy(dict(version=VERSION, tree=self.tree.fingerprint,
                    tree_manifest=self.tree.manifest(),
                    configuration=fingerprint(self.config), config=self.config,
                    model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                    generator=self.generator.get_state(), extractor=self.extractor.state_dict(),
                    updates=self.updates, examples=self.examples))

    def load_state_dict(self, state):
        validate_header(state, self)
        # Validate on a separate object before touching any earned state.
        candidate = ShadowLearner(self.tree, self.config)
        candidate.model.load_state_dict(state["model"], strict=True)
        candidate.optimizer.load_state_dict(state["optimizer"])
        candidate.generator.set_state(state["generator"])
        candidate.extractor.load_state_dict(state["extractor"])
        if not finite_state((candidate.model.state_dict(), candidate.optimizer.state_dict())):
            raise ValueError("Non-finite hierarchy checkpoint")
        candidate.updates, candidate.examples = int(state["updates"]), int(state["examples"])
        if min(candidate.updates, candidate.examples) < 0:
            raise ValueError("Negative saved experience")
        self.model, self.optimizer = candidate.model, candidate.optimizer
        self.generator, self.extractor = candidate.generator, candidate.extractor
        self.updates, self.examples = candidate.updates, candidate.examples
