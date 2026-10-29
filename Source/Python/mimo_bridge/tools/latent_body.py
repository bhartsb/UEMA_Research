# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Offline hierarchy learning and matched clone execution; C1 launches these jobs.

No command starts production training or changes a supervisor/control file.
Clone inputs are the baseline planners' issued commands bound to a saved start.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lem.latent_body import checkpoint
from lem.latent_body.learning import ShadowLearner
from lem.latent_body.trajectory import Example, Observation, end_positions, row_options, row_scales, state_inputs, tensor
from lem.latent_body.tree import arm_hand_tree, bind_mujoco, body_tree, from_manifest
from mimo_bridge.body_layout import names_for
from mimo_bridge.latent_body_runtime import jsonable, observation, observed_contract


def output_directory(path):
    path = Path(path).resolve()
    if "saved" in {part.lower() for part in path.parts}:
        raise ValueError("Qualification outputs must stay outside Saved")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, default=jsonable, indent=2, allow_nan=False), encoding="utf-8")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(Path(path).resolve()), sha256=digest.hexdigest())


def read_examples(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                yield Example(*(Observation.from_dict(row[k]) for k in ("history", "start", "end")),
                              torch.tensor(row["executed"]), row["action_contract"])
            except (KeyError, ValueError, TypeError) as error:
                raise ValueError(f"Invalid trajectory row {number} in {path}: {error}") from error


def training_examples(args, seed):
    """The training file, --epochs passes, each drawn through a --shuffle-buffer of examples (0 = file order).

    The shadow's trajectory is written in time order, eight bodies per model step; batches of consecutive rows
    are one moment seen eight times (review of Sep 30, 2026). A buffer of K rows mixes moments K/8 steps apart.
    """
    rng = np.random.default_rng(seed)
    size = int(getattr(args, "shuffle_buffer", 0) or 0)
    for _ in range(int(getattr(args, "epochs", 1) or 1)):
        buffer = []
        for ex in read_examples(args.data):
            if size <= 1:
                yield ex
                continue
            buffer.append(ex)
            if len(buffer) >= size:
                i = int(rng.integers(len(buffer)))
                buffer[i], buffer[-1] = buffer[-1], buffer[i]
                yield buffer.pop()
        for i in rng.permutation(len(buffer)):
            yield buffer[i]


def load_learner(path):
    state = torch.load(path, weights_only=True, map_location="cpu")
    learner = ShadowLearner(from_manifest(state["tree_manifest"]), state["config"])
    learner.load_state_dict(state)
    return learner


def fit(args):
    out = output_directory(args.output)
    if (out / "hierarchy.pt").exists():
        raise ValueError("Use a new output directory; preserve earlier hierarchy checkpoints")
    if args.resume:
        learner = load_learner(args.resume)
        supplied = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if checkpoint.fingerprint(supplied) != checkpoint.fingerprint(learner.config):
            raise ValueError("Resume configuration differs from the saved hierarchy")
        if learner.tree.fingerprint != body_tree(names_for(args.layout), args.levels).fingerprint:
            raise ValueError("Resume body tree differs from the declared layout/levels")
    else:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        learner = ShadowLearner(body_tree(names_for(args.layout), args.levels), cfg)
    if args.updates < 1 or args.batch_size < 1:
        raise ValueError("Declare a positive update budget and batch size")
    if Path(args.data).resolve() == Path(args.holdout).resolve():
        raise ValueError("Training and held-out data must be separate files")
    # Episode/body identifiers bind the split. Repeated frames from the same
    # body episode cannot be presented as held-out generalization.
    held_keys = set()
    for ex in read_examples(args.holdout):
        learner.batch([ex])
        held_keys.add((ex.start.env_id, ex.start.episode_id))
    if not held_keys:
        raise ValueError("Empty held-out set")
    batch, completed, training_keys = [], 0, set()
    with (out / "lessons.jsonl").open("w", encoding="utf-8") as stream:
        for ex in training_examples(args, learner.config["seed"]):
            if (ex.start.env_id, ex.start.episode_id) in held_keys:
                raise ValueError("Train/holdout body episodes overlap")
            training_keys.add((ex.start.env_id, ex.start.episode_id))
            batch.append(ex)
            if len(batch) == args.batch_size:
                record = learner.learn(batch)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                completed += 1
                batch = []
                if completed % max(1, int(getattr(args, "save_every", 1) or 1)) == 0 or completed == args.updates:
                    checkpoint.save(out / "hierarchy.pt", learner)
                if completed == args.updates:
                    break
    if completed != args.updates:
        raise ValueError(f"Only {completed} complete batches available for {args.updates} declared updates")
    from lem.latent_body.evaluation import representation_probe
    totals, samples, controls, probes, still = {}, 0, [], 0, 0.
    with (out / "representation.jsonl").open("w", encoding="utf-8") as stream:
        for ex in read_examples(args.holdout):
            result = learner.learn([ex], train=False)
            # the "nothing moves" guess on the same scale as the future loss (mean over parts of mean squares; under
            # the whole-body rows over the numbers the future lesson weighs, trajectory.body_lesson_weights)
            from lem.latent_body.losses import weighted_mse
            one = learner.batch([ex])
            weights = one.change_weight or {}
            graded = [weighted_mse(v, torch.zeros_like(v), weights.get(name)) for name, v in one.change.items()
                      if name not in weights or weights[name].sum() > 0]
            still += float(sum(graded)) / len(graded)
            for key, value in result["loss_before_update"].items():
                totals[key] = totals.get(key, 0.) + value
            samples += 1
            controls.append(ex)
            if len(controls) == max(2, args.batch_size):
                stream.write(json.dumps(representation_probe(learner, controls), allow_nan=False) + "\n")
                probes += 1
                controls = []
        if len(controls) >= 2:
            stream.write(json.dumps(representation_probe(learner, controls), allow_nan=False) + "\n")
            probes += 1
    receipt = dict(completed=True, authority="none", updates=completed, held_out_examples=samples,
        held_out_body_episodes=sorted(held_keys), data_quality=learner.config["data_quality"],
        training_body_episodes=sorted(training_keys), representation_control_batches=probes,
        data=file_digest(args.data), holdout=file_digest(args.holdout), checkpoint=file_digest(out / "hierarchy.pt"),
        held_out_hindsight_losses={key: value / samples for key, value in totals.items()},
        held_out_future_if_nothing_moves=still / samples,
        order=dict(epochs=int(getattr(args, "epochs", 1) or 1), shuffle_buffer=int(getattr(args, "shuffle_buffer", 0) or 0)),
        physical_qualification=False, next_test="C1 runs latent interventions and the matched clone job")
    write_json(out / "receipt.json", receipt)
    return receipt


def start_digest(session):
    """Bind supplied baseline candidates to positions, velocities, actuators and time.

    Checkpoint, body and physical time are recorded beside this digest. A digest
    alone does not qualify the candidate-generating planner or a physical goal.
    """
    digest = hashlib.sha256()
    for values in (session.env.data.qpos, session.env.data.qvel, session.env.data.act):
        digest.update(np.asarray(values, dtype="<f8").tobytes())
    digest.update(np.asarray([session.env.data.time], dtype="<f8").tobytes())
    return digest.hexdigest()


@contextmanager
def saved_session(args, out):
    from mimo_bridge.tools import gof_probe_harness as H
    # H's default is Saved. Its scene root is rebound only in this isolated job;
    # load_episode strips live caregiver tapes before any clone is constructed.
    old_root = H.PROBES_ROOT
    H.PROBES_ROOT = out / "scenes"
    session = None
    try:
        frozen = H.load_frozen(args.checkpoint)
        episode, reason = H.load_episode(frozen)
        if episode is None:
            raise ValueError(reason)
        session = H.Session(frozen, episode=episode, body=args.body)
        session.observe()
        yield session
    finally:
        if session is not None:
            session.close()
        H.PROBES_ROOT = old_root


def warm_start(session, steps, tree=None):
    """The observations before and at the start, read with the hierarchy's own tree (its contact bits cover the
    handles it owns: latent_body_runtime.observation); the pilot tree when none is given."""
    if type(steps) is not int or steps < 1:
        raise ValueError("At least one declared policy step supplies real physical history")
    tree = arm_hand_tree(session.model.handle_names) if tree is None else tree
    history = None
    for step in range(steps):
        history = observation(session.frames[0], tree, step, 0)
        session.policy_act()
        session.observe()
    return history, observation(session.frames[0], tree, steps, 0)


def prepare(args):
    """Export a real start for C1's baseline planners; issue no hierarchy command."""
    from lem.model.contact_head import PATCH_NAMES
    out = output_directory(args.output)
    with saved_session(args, out) as session:
        history, current = warm_start(session, args.warmup_steps)
        packet = dict(completed=True, authority="test_world_only", production_control=False,
            checkpoint=str(Path(args.checkpoint).resolve()), body=args.body,
            warmup_steps=args.warmup_steps, start_digest=start_digest(session),
            history=history.state_dict(), current=current.state_dict(),
            hold_command=session.hold_order(),
            mapping=bind_mujoco(arm_hand_tree(session.model.handle_names), session.env.model, PATCH_NAMES),
            next_test="C1 binds direct/synergy planner candidates and measured trial settings to this start")
        write_json(out / "start.json", packet)
        return packet


def latent_command(session, scenario, history, current, target, selected):
    from lem.planner.latent_candidates import search
    from lem.planner.latent_rollout import realize
    spec = scenario["latent_search"]
    learner = load_learner(spec["hierarchy"])
    if learner.tree.handle_names != tuple(session.model.handle_names):
        raise ValueError("Hierarchy uses a different body layout")
    if learner.config["data_quality"] != scenario["data_quality"]:
        raise ValueError("Hierarchy and clone data quality declarations differ")
    if learner.config["action_contract"] != observed_contract(session.frames[0]):
        raise ValueError("Hierarchy and clone use different command meanings")
    # Validate actual history and the prediction horizon, without feeding future
    # observations to the planner. This first model predicts one model step.
    dt = current.sim_time - history.sim_time
    if not np.isclose(dt, learner.config["duration_s"], rtol=1e-6, atol=1e-9):
        raise ValueError("Clone duration differs from the trained prediction horizon")
    if scenario["horizon_steps"] != 1:
        raise ValueError("This hierarchy predicts one model step; longer horizons need a rollout model")
    rows = row_options(learner.config)
    state, recent = state_inputs(learner.tree, history, current, learner.config["scales"], **rows)
    owned = {i for ids in learner.tree.owned.values() for i in ids}
    if not set(selected.tolist()).issubset(owned):
        raise ValueError("The pilot planner cannot target handles outside its subtrees")
    def score(predictions):
        errors = []
        for root in learner.tree.roots:
            ids = learner.tree.subtree[root]
            indices = [i for i, h in enumerate(ids) if h in selected]
            if not indices:
                continue
            scale = row_scales(learner.tree.handle_names, ids, learner.config["scales"], "change", rows["digit_scales"],
                               rows["body"])
            change = predictions[root].change.reshape(spec["candidates"], len(ids), 6) * scale
            # world end positions; a wrist-frame digit row is carried by its wrist row (trajectory.end_positions)
            predicted = end_positions(current, ids, change, learner.tree.handle_names, rows["digit_frame"], rows["body"])
            errors.append((predicted[:, indices] - tensor(target[list(ids)])[indices]).square().flatten(1))
        return torch.cat(errors, -1).mean(-1)
    choice = search(learner.model, state, recent, score, candidates=spec["candidates"],
                    radius=spec["radius"], generator=learner.generator)
    if spec["decoder"] not in ("synergy", "handles"):
        raise ValueError("Declare synergy or handles as the realization path")
    synergy = getattr(session.model, "synergy", None) if spec["decoder"] == "synergy" else None
    if spec["decoder"] == "synergy" and synergy is None:
        raise ValueError("The baseline checkpoint has no learned synergy to reuse")
    command = realize(choice, learner.tree, current, session.hold_order()[0],
                      learner.config["scales"], synergy=synergy,
                      hand_frame=learner.config.get("hand_action_frame", "recorded"), **rows)
    return command.unsqueeze(0).numpy(), float(choice.scores[choice.index]), dict(
        selected=choice.index, roots=choice.roots, predicted_scores=choice.scores,
        decoder=spec["decoder"], hierarchy=str(Path(spec["hierarchy"]).resolve()),
        updates=learner.updates, candidates=choice.candidates)


def physical_measurements(command, outcome, *, start, target, selected, lines, wellness_fields):
    """Endpoint following within the declared horizon; not an impossibility proof."""
    from lem.latent_body.trajectory import rotation_delta
    position = np.asarray(outcome["pos_cm"], dtype=float)
    error_cm = np.linalg.norm(position[selected] - command[0, selected, :3], axis=-1)
    turns = rotation_delta(tensor(command[0, selected, 3:7]), tensor(np.asarray(outcome["quat"])[selected]))
    error_rad = turns.norm(dim=-1).numpy()
    if any(not np.isfinite(lines[k]) or lines[k] <= 0 for k in ("following_cm", "following_rad")):
        raise ValueError("Following lines must be measured positive distances/angles")
    reached = (error_cm <= lines["following_cm"]) & (error_rad <= lines["following_rad"])
    initial = float(np.square(start[selected] - target[selected]).mean())
    final = float(np.square(position[selected] - target[selected]).mean())
    wellness = {}
    for name, spec in wellness_fields.items():
        value = outcome
        for key in spec["path"]:
            value = value[key]
        value = float(value)
        if not np.isfinite(value) or not spec["units"] or not spec["measurement"]:
            raise ValueError("Wellness readings require finite values, units and a measurement receipt")
        wellness[name] = dict(value=value, units=spec["units"], measurement=spec["measurement"])
    return dict(followed_handles=int(reached.sum()), tested_handles=len(selected),
        following_rate=float(reached.mean()), following_position_error_cm=error_cm,
        following_angle_error_rad=error_rad, goal_improvement_cm2=initial - final,
        wellness=wellness, wellness_measured=bool(wellness),
        interpretation="endpoint following under this horizon and declared lines; not unreachable-target proof")


def clone(args):
    from mimo_bridge.latent_body_clone import compare
    out = output_directory(args.output)
    scenario = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    with saved_session(args, out) as session:
        history, current = warm_start(session, scenario["warmup_steps"], load_learner(scenario["latent_search"]["hierarchy"]).tree)
        if scenario["start_digest"] != start_digest(session):
            raise ValueError("Baseline candidates belong to a different physical start")
        if str(Path(scenario["checkpoint"]).resolve()) != str(Path(args.checkpoint).resolve()) or scenario["body"] != args.body:
            raise ValueError("Baseline checkpoint/body binding differs")
        target = np.asarray(scenario["goal_positions_cm"], dtype=float)
        if target.shape != (session.n_handles, 3) or not np.isfinite(target).all():
            raise ValueError("Declare finite physical goal positions for every handle")
        selected = np.asarray(scenario["goal_handles"], dtype=int)
        if selected.ndim != 1 or len(selected) == 0 or len(set(selected.tolist())) != len(selected) or np.any(selected < 0) or np.any(selected >= session.n_handles):
            raise ValueError("Declare unique valid goal handles")
        if "latent" in scenario["commands"]:
            raise ValueError("The clone job generates its latent candidate from the saved hierarchy")
        command, predicted, search_receipt = latent_command(session, scenario, history, current, target, selected)
        commands = {**scenario["commands"], "latent": command}
        scores = {**scenario["predicted_scores"], "latent": predicted}
        counts = {**scenario["candidate_counts"], "latent": search_receipt["candidates"]}
        start = np.asarray(session.readout()["pos_cm"])
        score = lambda row: float(np.square(np.asarray(row["pos_cm"])[selected] - target[selected]).mean())
        measure = lambda command, row: physical_measurements(command, row, start=start,
            target=target, selected=selected, lines=scenario["lines"], wellness_fields=scenario["wellness_fields"])
        result = compare(session, commands, score, steps=scenario["horizon_steps"],
            predicted_scores=scores, repeat_noise=float(scenario["repeat_noise"]),
            candidate_counts=counts, group=scenario["held_out_group"],
            quality=scenario["data_quality"], measure=measure)
        # Physical readouts contain numpy arrays and token tensors. The receipt
        # keeps scalar comparisons; raw outcomes remain in a separate tensor file.
        torch.save(result, out / "clone_outcomes.pt")
        for row in result["candidates"].values():
            row.pop("outcome")
            row.pop("replay")
            row["measurements"]["improvement_cm2_per_candidate"] = (
                row["measurements"]["goal_improvement_cm2"] / row["candidates"])
        result.update(completed=True, start_digest=scenario["start_digest"],
                      baseline_receipt=scenario["baseline_receipt"],
                      score_units="mean squared centimetres on the declared handles",
                      latent_search=search_receipt, hierarchy_receipt=scenario["hierarchy_receipt"],
                      physical_qualification=False)
        write_json(out / "receipt.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit_parser = sub.add_parser("fit")
    for flag in ("config", "data", "holdout", "output"):
        fit_parser.add_argument("--" + flag, required=True)
    fit_parser.add_argument("--resume", default="")
    fit_parser.add_argument("--layout", choices=("legacy", "trunk_v1"), default="trunk_v1")
    fit_parser.add_argument("--levels", default="arm,hand,index", help="arm,hand,index[,digits] or full_body")
    fit_parser.add_argument("--updates", type=int)
    fit_parser.add_argument("--batch-size", type=int)
    fit_parser.add_argument("--budget", help="JSON with measured updates, batch_size and measurement receipt")
    fit_parser.add_argument("--epochs", type=int, default=1, help="Passes over the training file")
    fit_parser.add_argument("--shuffle-buffer", type=int, default=0, help="Examples held for random draws; 0 = file order")
    fit_parser.add_argument("--save-every", type=int, default=1, help="Updates between checkpoint saves (and the last)")
    fit_parser.set_defaults(function=fit)
    clone_parser = sub.add_parser("clone")
    for flag in ("checkpoint", "scenario", "output"):
        clone_parser.add_argument("--" + flag, required=True)
    clone_parser.add_argument("--body", type=int, required=True)
    clone_parser.set_defaults(function=clone)
    prepare_parser = sub.add_parser("prepare")
    for flag in ("checkpoint", "output"):
        prepare_parser.add_argument("--" + flag, required=True)
    prepare_parser.add_argument("--body", type=int, required=True)
    prepare_parser.add_argument("--warmup-steps", type=int, required=True)
    prepare_parser.set_defaults(function=prepare)
    args = parser.parse_args(argv)
    try:
        if args.command == "fit":
            if args.budget:
                if args.updates is not None or args.batch_size is not None:
                    raise ValueError("Use either an explicit fit budget or a measured budget file")
                budget = json.loads(Path(args.budget).read_text(encoding="utf-8"))
                if not budget["measurement"]:
                    raise ValueError("Fit budget requires its measurement receipt")
                args.updates, args.batch_size = budget["updates"], budget["batch_size"]
            if any(type(v) is not int or v < 1 for v in (args.updates, args.batch_size)):
                raise ValueError("Declare positive integer updates and batch_size")
        args.function(args)
    except Exception as error:
        out = Path(args.output).resolve()
        if out.is_dir() and "saved" not in {part.lower() for part in out.parts}:
            write_json(out / "failure.json", dict(completed=False, error=repr(error),
                authority="none", command=args.command,
                next_test="C1 attributes this failure before rerunning; retain all partial checkpoints"))
        raise


if __name__ == "__main__":
    main()
