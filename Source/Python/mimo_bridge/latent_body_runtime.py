# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Physical observation adapter and independent shadow lifecycle."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import traceback

import torch

from lem.latent_body import checkpoint
from lem.latent_body.learning import ShadowLearner
from lem.latent_body.trajectory import Observation, predicted_rows, row_options, row_scales, state_inputs, tensor
from lem.latent_body.tree import body_tree, skin_owner

log = logging.getLogger("mimo_bridge.latent_body")


def observed_contract(frame):
    from mimo_bridge.limb_target_control import action_contract_parts
    hand = frame.meta.get("hand_target_contract", "world_v0")
    if hand not in ("world_v0", "wrist_relative_joint_v1"):
        raise ValueError("Unknown physical hand command contract")
    parts = ([] if hand == "world_v0" else [hand]) + action_contract_parts(frame.meta)
    return "+".join(parts) if parts else "world_v0"


def observation(frame, tree, step, episode_id):
    from lem.model.contact_head import PATCH_NAMES
    states = (frame.meta.get("task") or {}).get("contact_patch_state")
    if states is None or len(states) != len(PATCH_NAMES):
        raise ValueError("Physical contact observations are missing or use another skin layout")
    contacts = torch.zeros(len(tree.handle_names))
    for name, bits in zip(PATCH_NAMES, states):
        if type(bits) is not int or bits not in (0, 1, 2, 3):
            raise ValueError("Invalid physical contact bit field")
        owner = skin_owner(name)
        if owner in tree.owners and bits:
            contacts[tree.handle_names.index(owner)] = 1
    obs = Observation(int(frame.env_index), int(episode_id), int(step), float(frame.sim_time),
                      tensor(frame.bones), contacts)
    obs.validate(len(tree.handle_names))
    return obs


def jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def example_record(example):
    return dict(history=example.history.state_dict(), start=example.start.state_dict(),
                end=example.end.state_dict(), executed=example.executed,
                action_contract=example.action_contract)


class ShadowRuntime:
    def __init__(self, args, handles):
        self.args = args
        self.directory = Path(args.latent_body_output).resolve()
        self.tree = body_tree(handles, args.latent_body_levels)
        config = json.loads(Path(args.latent_body_config).read_text(encoding="utf-8"))
        if config.get("horizon_steps", 1) != 1:
            # the shadow's TrajectoryExtractor builds one-step examples; a hold-horizon hierarchy is fitted offline from
            # recorded holds (trajectory.HoldRuns, latent_body_split --horizon-steps)
            raise ValueError("The live shadow learns one-step examples; a hierarchy with horizon_steps %s is offline only"
                             % config.get("horizon_steps"))
        self.learner = ShadowLearner(self.tree, config)
        body = row_options(config)["body"]
        if body is not None and body.eye_row_convention != getattr(args, "eye_row_convention", "legacy"):
            # the eyes' command rows mean another rotation under the other convention (trajectory.eye_row_physical)
            raise ValueError("The hierarchy's eye row convention %s differs from the trainer's %s"
                             % (body.eye_row_convention, getattr(args, "eye_row_convention", "legacy")))
        self.latest = self.directory / "latest.pt"
        resume = args.latent_body_resume
        if resume == "auto":
            # a supervisor relaunch repeats the launch command: continue this directory's own experience
            resume = str(self.latest) if self.latest.exists() else ""
        if self.latest.exists() and not resume:
            raise ValueError("Existing shadow experience requires explicit --latent-body-resume")
        if resume:
            checkpoint.load(resume, self.learner)
        self.log_bytes = int(float(getattr(args, "latent_body_log_mb", 0.0) or 0.0) * 2 ** 20)
        self.goal = None
        if args.latent_planner == "shadow":
            self.goal = json.loads(Path(args.latent_body_goal).read_text(encoding="utf-8"))
            if set(self.goal) != set(self.tree.roots):
                raise ValueError("Shadow goal must give physical change targets for each root")
            for root, values in self.goal.items():
                target = tensor(values)
                if target.shape != (len(self.tree.subtree[root]), 6) or not torch.isfinite(target).all():
                    raise ValueError("Goals require per-handle cm displacement and radian turn vectors")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.write("events.jsonl", {"type": "started", "authority": "none", "tree": self.tree.manifest(),
                   "config": config, "resume": args.latent_body_resume or None})
        self.saved_updates = self.learner.updates
        self.first_step = None          # the first model step this shadow observes (observe sets it)

    def write(self, filename, value):
        path = self.directory / filename
        cap = getattr(self, "log_bytes", 0)
        if cap and path.exists() and path.stat().st_size >= cap:
            # --latent-body-log-mb: rotate, so a record never holds more than twice the cap
            path.replace(path.with_name(path.name + ".1"))
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, default=jsonable, allow_nan=False) + "\n")

    def observe(self, frames, transitions, step, episode_id, *, train, synergy=None):
        records = {tr.env_id: tr for tr in transitions}
        if len(records) != len(transitions):
            raise ValueError("Duplicate body in completed transition batch")
        if self.first_step is None:
            self.first_step = int(step)
        declared = self.learner.config["action_contract"]
        examples, observed, excluded = [], [], []
        for frame in frames:
            obs = observation(frame, self.tree, step, episode_id)
            tr = records.get(obs.env_id)
            if tr is not None:
                tr.validate()
                if tr.episode_id != episode_id or tr.frame_t1 != step:
                    raise ValueError("Completed transition and physical outcome disagree")
                if tr.action_contract != observed_contract(frame):
                    raise ValueError("Transition and physical command contracts differ")
                if abs(tr.duration_s - self.learner.config["duration_s"]) > 1e-9:
                    raise ValueError("Completed transition uses a different model-step duration")
                for source in (*tr.source_mask_t.values(), *tr.source_mask_t1.values()):
                    if str(getattr(source, "value", source)) != "real":
                        raise ValueError("Synthetic or stale data cannot become a physical training example")
                if tr.action_contract != declared and tr.frame_t < self.first_step:
                    # THE RESUME BOUNDARY (Oct 10, 2026; LATENT_SHADOW_REJECTS_REPAIRED_FINGER_CONTRACT). A launch that
                    # resumes a checkpoint closes, at its first model step, the open transitions the episode companion
                    # restores: executed before this shadow existed, under the contract the checkpoint's run wrote
                    # (wrist_relative_joint_v1 at 481,800, where the configuration declares the repaired servo's). Its
                    # command is not learned and nothing is relabelled: the transition is named in one
                    # excluded_transitions record and its outcome stays physical history, as for a transition with no
                    # command. Any later transition of another contract still stops the shadow (the extractor).
                    excluded.append(dict(env_id=int(tr.env_id), episode_id=int(tr.episode_id), frame_t=int(tr.frame_t),
                                         frame_t1=int(tr.frame_t1), order_id=tr.order_id,
                                         action_contract=tr.action_contract))
                    tr = None
            ex = self.learner.extractor.observe(obs, None if tr is None else tr.executed_action,
                    action_contract=None if tr is None else tr.action_contract,
                    frame_t=None if tr is None else tr.frame_t)
            observed.append((obs, tr))
            if ex is not None:
                examples.append(ex)
                # KEEP THE PHYSICAL LEDGER (2026-09-26). The learned action is the
                # issued canonical goal under this contract. Expanded substeps and
                # joint goals are retained for auditing what physics actually got.
                self.write("trajectory.jsonl", {**example_record(ex), "ledger": dict(
                    motor_substeps=tr.motor_substeps, hand_joint_goal=tr.hand_joint_goal,
                    decoded_ctrl=tr.decoded_ctrl, order_id=tr.order_id,
                    source_mask_t=tr.source_mask_t, source_mask_t1=tr.source_mask_t1)})
        if excluded:
            self.write("events.jsonl", dict(
                type="excluded_transitions", authority="none", step=int(step), first_step=self.first_step,
                declared_contract=declared, transitions=excluded,
                reason="executed before this shadow's first model step under another command contract (the resume "
                       "boundary): the command is not learned, the outcome is kept as physical history"))
        if examples:
            metrics = self.learner.learn(examples, train=train)
            self.write("events.jsonl", {**metrics, "step": int(step),
                       "skipped_history_boundaries": self.learner.extractor.dropped})
        if self.goal is not None and step % max(1, self.args.report_every) == 0:
            self.plan(observed, step, synergy)
        # SEPARATE SHADOW SAVE (2026-09-26). Never register new parameters, optimizer
        # entries or history on LemLite. Production checkpoints remain unchanged.
        # Every completed observation is saved, including warm-up and private RNG.
        # --latent-body-save-every N (Oct 1, 2026; default 1 = every observation, as before): the whole-body shadow's
        # latest.pt is 19.8 MB and takes 0.080-0.101 s to write (cost.py), against 5.7 MB and 0.023-0.032 s for the pilot;
        # N > 1 writes it once N updates have accumulated, so a crash loses at most N - 1 updates of shadow experience.
        every = int(getattr(self.args, "latent_body_save_every", 1) or 1)
        if every <= 1 or self.learner.updates - self.saved_updates >= every:
            checkpoint.save(self.latest, self.learner)
            self.saved_updates = self.learner.updates

    def plan(self, observed, step, synergy):
        from lem.planner.latent_candidates import search
        from lem.planner.latent_rollout import realize
        scales = self.learner.config["scales"]
        rows = row_options(self.learner.config)
        targets = {r: (tensor(v) / row_scales(self.tree.handle_names, self.tree.subtree[r], scales, "change",
                                              rows["digit_scales"], rows["body"])).flatten()
                   for r, v in self.goal.items()}
        for obs, tr in observed:
            prior = self.learner.extractor.previous[obs.env_id]
            if len(prior) < 2 or tr is None:
                continue
            state, history = state_inputs(self.tree, prior[0], obs, scales, **rows)
            score = lambda predictions: sum((predicted_rows(self.tree, predictions, r, rows["body"]).flatten(-2)
                                              - targets[r]).square().mean(-1) for r in self.tree.roots)
            choice = search(self.learner.model, state, history, score,
                            candidates=self.args.latent_body_candidates, radius=self.args.latent_body_radius,
                            generator=self.learner.generator)
            command = realize(choice, self.tree, obs, tr.executed_action, scales, synergy=synergy,
                              hand_frame=self.learner.config.get("hand_action_frame", "recorded"), **rows)
            self.write("plans.jsonl", dict(step=int(step), env_id=obs.env_id, authority="none",
                decoder="synergy" if synergy is not None else "handle_targets", candidates=choice.candidates,
                selected=choice.index, predicted_scores=choice.scores, roots=choice.roots,
                command_ignored=command, physical_execution=False))

    def fail(self):
        reason = traceback.format_exc()
        try:
            self.write("events.jsonl", dict(type="failed", authority="none", reason=reason,
                saved_updates=self.saved_updates, next_test="C1 reviews failure and reruns the declared shadow job"))
        except OSError:
            log.exception("Cannot write hierarchy failure receipt; production log retains the failure")
