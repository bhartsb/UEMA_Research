# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Controlled learning of measured action effects on a frozen sensory vocabulary.

Collect paired physical outcomes, then compare equal-exposure next-frame learning,
paired effect learning, and shuffled-action learning from identical weights. World
seeds are split before fitting. This experiment never writes a production checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODS = ("tactile", "proprioception")


def raw_candidates(raw, achieved, bones, rng):
    from mimo_bridge.tools.sensor_action_contract_check import hold_current
    from mimo_bridge.tools.first_action_benefit import matched_control
    hold = torch.as_tensor(hold_current(bones), dtype=raw.dtype)
    hold[..., :3] /= 200.0
    candidates = [raw.detach().clone(), hold]
    for _ in range(2):
        # Both arguments in normalized position units; execution converts to cm later.
        candidates.append(torch.as_tensor(matched_control(
            raw.detach().numpy(), achieved[..., :3].numpy(), rng), dtype=raw.dtype))
    for c in candidates:
        c[:, 23] = raw[:, 23]
    return candidates


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def paired_differences(x):
    """All distinct unordered action pairs, keeping the event axis intact."""
    i, j = torch.triu_indices(x.shape[1], x.shape[1], offset=1)
    return x[:, i] - x[:, j]


def physical_fields(frame):
    """Retain actual measurements so future evaluations need not invert a token."""
    return {
        "achieved": torch.as_tensor(frame.tokenized.motor_handles).detach().cpu().clone(),
        "proprioception": torch.as_tensor(frame.tokenized.proprio).detach().cpu().clone(),
        "patches": torch.as_tensor(frame.patches).detach().cpu().clone(),
        "bones": torch.as_tensor(frame.bones).detach().cpu().clone(),
    }


def measured_scales(rows):
    scales = {}
    for m in MODS:
        y = torch.stack([r["target"][m] for r in rows])
        current = torch.stack([r["tokens"][m] for r in rows])
        point = float((y - current[:, None]).square().mean())
        effect = float(paired_differences(y).square().mean())
        floor = float(torch.stack([r["replay_mse"][m] for r in rows]).mean())
        if not np.isfinite([point, effect, floor]).all() or effect <= max(10 * floor, 1e-12):
            raise ValueError(f"{m}: physical action effects unresolved: {effect=} {floor=}")
        scales[m] = {"point": point, "effect": effect, "replay_floor": floor}
    return scales


def learning_loss(pred, target, scales, effect=False):
    terms = []
    for m in MODS:
        terms.append(F.mse_loss(pred[m], target[m]) / scales[m]["point"])
        if effect:
            terms.append(F.mse_loss(paired_differences(pred[m]),
                                    paired_differences(target[m])) / scales[m]["effect"])
    return torch.stack(terms).mean()


def model(ckpt, intention=False):
    from mimo_bridge.tools.causal_test import _build_model
    # The observer is dark: removing its counters cannot affect chosen actions.
    m, _, _, _ = _build_model(str(ckpt), "cpu", 9173, intention=intention,
                              arg_overrides={"hierarchical_self": "off"})
    # Disable dropout for paired comparisons; gradients still work while fitting.
    for part in (m.encoder_registry, m.forward_model, m.prediction_heads):
        part.eval()
    return m


class CommandResidual(torch.nn.Module):
    """An experimental learned path retaining the complete outgoing command.
    n_handles is the model's own row count (24 legacy, 26 trunk_v1): until 2026-09-23 it
    was a fixed 24, which refused every order from a trunk_v1 model (every H84 segment)."""
    def __init__(self, base, width, n_handles):
        super().__init__()
        self.base = base
        self.n_handles = int(n_handles)
        self.command_projection = torch.nn.Linear(self.n_handles * 9, width, bias=False)
        torch.nn.init.zeros_(self.command_projection.weight)

    @property
    def patch_mlp(self):
        # build_motor_input reads this contract to distinguish 19 fields from legacy 10.
        return self.base.patch_mlp

    def forward(self, rows):
        if rows.shape[-2:] != (self.n_handles, 19):
            raise ValueError("Command residual requires achieved(10) + command(9) for "
                             f"{self.n_handles} handles, got {tuple(rows.shape[-2:])}")
        return self.base(rows) + self.command_projection(rows[..., 10:19].flatten(1)).unsqueeze(1)


def add_command_residual(m):
    # This is a checkpoint-copy experiment. The production encoder is unchanged.
    m.encoder_registry.encoders["motor"] = CommandResidual(m.encoder_registry.encoders["motor"], m.d_model,
                                                           len(m.handle_names))


def collection_steps(clock, seconds):
    """Observation horizon must not change the policy's physical clock."""
    return clock.steps_for(seconds), clock.action_repeat


def collect(ckpt, out, worlds, events, warm=80, stride=10, resume=False,
            seconds=0.05, seed_offset=0, body_only=False,
            history_steps=0, rest="off", evaluation_only=False, independent_streams=False,
            component_paths=False):
    from mimo_bridge.tools.causal_test import _limits_for_run, _cfg_from_run
    from mimo_bridge.tools.sensor_action_contract_check import resolved_env_kwargs
    from mimo_bridge.tools.first_action_benefit import (
        physics_state, restore_physics, encode_frame, state_wellness, free_ram_gb)
    from mimo_bridge.tools.probe_clock import ProbeClock
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.lem_model import LemLite

    if worlds < 6:
        raise ValueError("Need at least three training and three held-out worlds")
    out.mkdir(parents=True, exist_ok=resume)
    a, _, _ = _cfg_from_run(str(ckpt))
    phase = json.loads((ckpt / "meta.json").read_text())["curriculum_state"]["phase"]
    kw, _ = resolved_env_kwargs(str(ckpt.parent.parent), phase_override=phase)
    # Body-action intervention: the same voluntary eye command in every arm;
    # no stochastic eye override can change which command the motor token means.
    kw.update(gaze_reflex=False, spontaneous_saccades=False)
    declaration = {
        "checkpoint": str(ckpt.resolve()), "checkpoint_sha256": hashlib.sha256(
            (ckpt / "model.safetensors").read_bytes()).hexdigest(),
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "worlds": worlds, "events_per_world": events, "warm_model_steps": warm,
        "stride_model_steps": stride, "seconds": seconds, "actions": 4,
        "seed_offset": seed_offset, "trajectory_clock": "action_repeat physics steps per model step",
        "body_only": body_only,
        # 2026-09-10 sequence study: each decision row carries the STRICTLY
        # PAST policy history (executed command + resulting raw body/touch
        # fields per model step, recorded before the candidate action - never
        # from its future), and the trajectory alternates the trainer's
        # active/rest schedule so resting starting states are sampled too
        "history_steps": history_steps, "rest_schedule": rest,
        "history_clock_version": 1, "evaluation_only": evaluation_only,
        "component_paths": component_paths,
        "independent_streams": independent_streams,
        "train_worlds": [] if evaluation_only else list(range(worlds - 3)),
        "heldout_worlds": list(range(worlds)) if evaluation_only else list(range(worlds - 3, worlds)),
        "overrides": {"hierarchical_self": "off (dark observer only)",
                      "gaze_reflex": False, "spontaneous_saccades": False},
        "method": "Each raw candidate passes the production execution transform and smoother. "
                  "All arms restore full simulator state. Targets are raw frozen encoder outputs.",
        "acceptance": "Finite losses; repeatability floor resolves physical action effects; "
                      "held-out world effect error below the zero-effect predictor in every world "
                      "and below matched next-frame and shuffled-action controls. No behavioral "
                      "or developmental milestone is inferred from this experiment.",
        "fit_limit": "Fixed 12 epochs, no tuning on held-out worlds. Each condition has the same "
                     "batches and initial weights. Failure remains a result, never an automatic promotion.",
        "scale": "Effect error: 1 = predict no difference between actions; 0 = predict each difference exactly.",
    }
    if body_only:
        declaration["method"] = ("Frozen active-policy trajectories at their original cadence. "
            "Each candidate is clamped and smoothed once, then held across its observation branch. "
            "Full state restored per arm; raw body paths and duplicate replay retained every model step.")
        declaration["fit_limit"] = "No token fit. Separate physical-predictor declaration controls fitting and acceptance."
        declaration["acceptance"] = "Collection only; no capability or fitted-model pass follows from collection."
    if not resume:
        write_json(out / "declaration.json", declaration)
    else:
        original = json.loads((out / "declaration.json").read_text())
        for k in ("checkpoint_sha256", "worlds", "events_per_world", "train_worlds", "heldout_worlds",
                  "seconds", "seed_offset", "trajectory_clock", "warm_model_steps", "stride_model_steps", "body_only",
                  "history_steps", "rest_schedule", "history_clock_version", "evaluation_only", "independent_streams"):
            if declaration[k] != original[k]:
                raise ValueError(f"Resume changed declared {k}")
        write_json(out / "collection_amendment.json", {
            "reason": "An arrived goal can legitimately produce no plan. Such an event now uses "
                      "the controller's raw candidate, with the same physical constraints. "
                      "Remaining worlds each reload the checkpoint before collection.",
            "collector_sha256": declaration["collector_sha256"],
            "prior_worlds": "See the study declaration for inherited data. Remaining worlds "
                            "each start with the identical checkpoint and random seed.",
            "sampling": "Active trajectories; no externally scheduled rests. Eye overrides disabled.",
            "gate_changed": False, "fitting_started": False})
    meter = model(ckpt)
    limits, _ = _limits_for_run(str(ckpt), meter)
    dev = dict(limits, exploration_noise=float(a.get("babble_noise") or 0))
    rows = torch.load(out / "paired_frames.pt", weights_only=False) if resume else []
    for world in range(worlds):
        existing = [r for r in rows if r["world"] == world]
        if existing:
            if len(existing) != events:
                raise ValueError("Can resume only after a complete world")
            continue
        if free_ram_gb() < 2.0:
            raise MemoryError("Leaving 2 GB for the live trainer; collection stopped")
        live = model(ckpt, True)
        src = SingleEnvSource(**dict(kw, seed=7000 * (world + 11 + seed_offset)))
        clock = ProbeClock(float(src.env.dt), int(a["action_repeat"]), int(a["motor_hold"]))
        steps, trajectory_steps = collection_steps(clock, seconds)
        if not np.isclose(clock.model_step_seconds, .05):
            raise ValueError("This experiment requires the checkpoint's 50 ms model step")
        stream = world + (seed_offset if independent_streams else 0)
        rng = np.random.default_rng(441 + stream)
        from collections import deque
        past = deque(maxlen=max(history_steps, 1))
        bout_rng = np.random.default_rng(4600 + stream)
        bout_on, bout_until = True, 0
        try:
            live.reset_transition_state()
            frames = src.reset()
            held, previous = None, None
            count = 0
            for t in range(warm + stride * events):
                if rest == "on" and t >= bout_until:
                    # the trainer's own bout distribution (2-5 s active,
                    # 1-3 s rest) at the fixed 50 ms model-step clock
                    bout_on = not bout_on
                    dur_s = (bout_rng.uniform(2.0, 5.0) if bout_on
                             else bout_rng.uniform(1.0, 3.0))
                    bout_until = t + max(1, int(dur_s / clock.model_step_seconds))
                with torch.no_grad():
                    result = live.step(frames, dev, train=False, step_index=t, executed_action=previous)
                if t % clock.motor_hold == 0 or held is None:
                    held = torch.as_tensor(result.motor_for_action).detach().clone()
                if (t >= warm and (t - warm) % stride == 0
                        and (not history_steps or len(past) == history_steps)):
                    pi = live._last_plan_inputs
                    if pi is None:
                        raise ValueError("No decision state at sample time")
                    decision = pi["decision"]
                    # Require this very frame's decision, not a retained older plan.
                    if not torch.equal(decision["achieved"], live.last_achieved):
                        raise ValueError("Stale planner decision would misalign action and state")
                    raw = (pi["plan0"] if pi.get("plan0") is not None else pi["motor"]).detach().clone()
                    ach = decision["achieved"].detach().clone()
                    candidates = raw_candidates(raw, ach, frames[0].bones, rng)
                    with torch.no_grad():
                        commands = [LemLite.planned_execution(c, decision["dev"],
                            handle_state=ach, previous=decision["smooth_prev"],
                            smoothing=decision["smoothing"]).detach() for c in candidates]
                    state = physics_state(src.env)
                    start = float(frames[0].sim_time)
                    physical_start = physical_fields(frames[0])
                    outcomes, wellness, physical_end, physical_paths = {}, {}, {}, {}
                    for arm in rng.permutation(5):
                        restore_physics(src.env, state)
                        cmd = commands[min(int(arm), 3)] if arm < 4 else commands[0]
                        gaze = live._gaze_from_handle(cmd)
                        path = []
                        for k in range(steps):
                            end = src.step(cmd.numpy(), gaze, None, render=(not body_only and k == steps - 1))
                            if (k + 1) % clock.action_repeat == 0:
                                entry = physical_fields(end[0])
                                if component_paths:
                                    # 2026-09-11 wellness-choice study: the
                                    # grounded component wellness at EVERY
                                    # model step of the branch, from the same
                                    # state_wellness the end score uses - an
                                    # uncomfortable movement cannot hide
                                    # behind a comfortable ending
                                    entry["wellness"] = state_wellness(meter, end[0])
                                path.append(entry)
                        clock.verify_elapsed(start, float(end[0].sim_time), seconds)
                        if not body_only:
                            outcomes[int(arm)] = {m: v.reshape(-1).clone() for m, v in
                                                 encode_frame(meter, end, dict(limits, exploration_noise=0)).items()}
                        # parity BY CONSTRUCTION: the stored end wellness IS the
                        # final path entry's when component paths are on
                        wellness[int(arm)] = (path[-1]["wellness"] if component_paths
                                              else state_wellness(meter, end[0]))
                        physical_end[int(arm)] = physical_fields(end[0])
                        physical_paths[int(arm)] = path
                    restore_physics(src.env, state)
                    row = {
                        "world": world, "event": count, "sim_time": start,
                        "candidate_source": "planner" if pi.get("plan0") is not None else "controller (empty plan)",
                        "tokens": {m: v.detach().reshape(-1).clone() for m, v in live.last_tokens.items()},
                        "context": decision["context"].detach().squeeze(0).clone(),
                        "achieved": ach.squeeze(0), "commands": torch.cat(commands),
                        "raw_commands": torch.cat(candidates),
                        "source_mask": dict(decision["source_mask"]),
                        "target": {} if body_only else {m: torch.stack([outcomes[k][m] for k in range(4)]) for m in MODS},
                        "replay_mse": {} if body_only else {m: (outcomes[0][m] - outcomes[4][m]).square().mean() for m in MODS},
                        "wellness": [wellness[k] for k in range(4)],
                        "physical_start": physical_start,
                        "physical_end": [physical_end[k] for k in range(4)],
                        "physical_replay": physical_end[4],
                        "physical_paths": physical_paths,
                        # STRICTLY PAST: each entry is the command executed at
                        # an earlier model step and the raw fields measured
                        # AFTER it - nothing from the candidate's future
                        "history": ([{"command": c, **f} for c, f in past]
                                    if history_steps else None),
                        "resting_start": ((not bout_on) if rest == "on" else None),
                    }
                    rows.append(row)
                    count += 1
                send = held
                if rest == "on" and not bout_on and live.last_achieved is not None:
                    # the trainer's rest: an ACTIVE hold of the current pose
                    # (positions x200 cm, achieved quaternion; strength from
                    # the held order) - train_mimo's 'babble-bouts real' arithmetic
                    send = held.detach().clone()
                    _ach = live.last_achieved.detach().float()
                    n_h = min(send.shape[1], _ach.shape[1])
                    send[:, :n_h, 0:3] = _ach[:, :n_h, 0:3] * 200.0
                    _q = _ach[:, :n_h, 3:7]
                    _qn = _q.norm(dim=-1, keepdim=True)
                    send[:, :n_h, 3:7] = torch.where(_qn > 1e-6,
                                                     _q / _qn.clamp_min(1e-6),
                                                     send[:, :n_h, 3:7])
                cmd = send.detach().numpy()
                gaze = live._gaze_from_handle(send)
                trajectory_start = float(frames[0].sim_time)
                for k in range(trajectory_steps):
                    frames = src.step(cmd, gaze, None, render=(k == trajectory_steps - 1))
                clock.verify_elapsed(trajectory_start, float(frames[0].sim_time), clock.model_step_seconds)
                previous = send.detach().clone()
                if history_steps:
                    history_frame = physical_fields(frames[0])
                    history_frame.update(command_start_time=trajectory_start,
                                         outcome_time=float(frames[0].sim_time))
                    past.append((send.detach().clone(),history_frame))
            print(json.dumps({"world_collected": world, "events": count, "free_ram_gb": free_ram_gb()}), flush=True)
            torch.save(rows, out / "paired_frames.pt")
        finally:
            src.close()
            del live
    if not body_only:
        write_json(out / "scales.json", measured_scales([r for r in rows if r["world"] in declaration["train_worlds"]]))


def predict(m, rows, shuffle=False, generator=None, canonical=False):
    from lem.model.encoder_registry import build_motor_input, normalize_executed_command
    n, actions = len(rows), rows[0]["commands"].shape[0]
    commands = torch.stack([r["commands"] for r in rows])
    if shuffle:
        # Independently shuffle WITHIN each state: preserve all state/outcome statistics.
        commands = torch.stack([c[torch.randperm(actions, generator=generator)] for c in commands])
    achieved = torch.stack([r["achieved"] for r in rows]).repeat_interleave(actions, dim=0)
    motor = build_motor_input(m.encoder_registry, normalize_executed_command(
        commands.flatten(0, 1)), achieved=achieved)
    zm = m.encoder_registry.encode({"motor": motor}, {"motor": "real"})["motor"]
    current = {k: torch.stack([r["tokens"][k] for r in rows]).repeat_interleave(actions, dim=0)
               for k in rows[0]["tokens"]}
    tokens = {k: (zm if k == "motor" else v).reshape(n * actions, 1, -1) for k, v in current.items()}
    ctx = torch.stack([r["context"] for r in rows]).repeat_interleave(actions, dim=0)
    _, last, _ = m.forward_model(tokens, ctx)
    pred = m.prediction_heads(last, current=current, who="paired_action_effect_learning")
    if canonical:
        pred = {k: m.encoder_registry.output_norm(v) for k, v in pred.items()}
    return {k: pred[k].reshape(n, actions, -1) for k in MODS}


def evaluate(m, rows, scales):
    m.forward_model.invalidate_embedding_cache()
    results = {}
    for world in sorted({r["world"] for r in rows}):
        rr = [r for r in rows if r["world"] == world]
        result = {"physical_resolution": measured_scales(rr)}
        for canonical in (False, True):
            with torch.no_grad():
                batches = [predict(m, rr[i:i+2], canonical=canonical) for i in range(0, len(rr), 2)]
                pred = {k: torch.cat([b[k] for b in batches]) for k in MODS}
            result["normalized_heads" if canonical else "raw_heads"] = {}
            for k in MODS:
                y = torch.stack([r["target"][k] for r in rr])
                current = torch.stack([r["tokens"][k] for r in rr])[:, None]
                de = paired_differences(y)
                mse = float((paired_differences(pred[k]) - de).square().mean())
                result["normalized_heads" if canonical else "raw_heads"][k] = {
                    "effect_error_ratio": mse / float(de.square().mean()),
                    "next_frame_error_vs_persistence": float((pred[k] - y).square().mean()) /
                                                        float((current - y).square().mean()),
                }
        results[world] = result
    return results


def fit(ckpt, out, epochs=12, resume=False):
    declaration = json.loads((out / "declaration.json").read_text())
    if hashlib.sha256((ckpt / "model.safetensors").read_bytes()).hexdigest() != declaration["checkpoint_sha256"]:
        raise ValueError("Fitting checkpoint differs from the frozen target encoder")
    if epochs != 12:
        raise ValueError("Changing declared epochs requires a separate experiment")
    rows = torch.load(out / "paired_frames.pt", map_location="cpu", weights_only=False)
    train = [r for r in rows if r["world"] in declaration["train_worlds"]]
    test = [r for r in rows if r["world"] in declaration["heldout_worlds"]]
    if set(declaration["train_worlds"]) & set(declaration["heldout_worlds"]):
        raise ValueError("World split leaks held-out outcomes into training")
    if len(rows) != declaration["worlds"] * declaration["events_per_world"]:
        raise ValueError("Incomplete collection cannot be treated as the declared experiment")
    scales = measured_scales(train)
    results = {"scales_from_training_worlds_only": scales, "conditions": {},
               "evaluation_inputs": "Raw encoder tokens. The actual planner's top-down blend and "
                                    "wellness judge are outside this one-step learning test.",
               "fit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if resume:
        results = json.loads((out / "learning_results.json").read_text())
        results.setdefault("resume_source_sha256", []).append(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    conditions = declaration.get("conditions", ["next_frame", "paired_effect", "shuffled_action"])
    for condition in conditions:
        if condition in results["conditions"]:
            continue
        m = model(ckpt)
        if condition == "command_residual":
            add_command_residual(m)
        for p in m.params:
            p.requires_grad_(False)
        parts = [m.forward_model, m.prediction_heads, m.encoder_registry.encoders["motor"]]
        # Stored context includes modality embeddings. Keep their coordinate system
        # fixed, together with the sensory vocabulary, throughout this offline fit.
        fixed = {id(p) for emb in (m.forward_model.modality_embedding,
                                   m.forward_model.temporal_embedding) for p in emb.parameters()}
        params = []
        for part in parts:
            for p in part.parameters():
                if id(p) in fixed:
                    continue
                p.requires_grad_(True)
                params.append(p)
        # Same architecture, raw target vocabulary, initial weights, and optimizer in every arm.
        lr = m.optimizer.param_groups[0]["lr"]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=m.optimizer.param_groups[0]["weight_decay"])
        rng = torch.Generator().manual_seed(8162)
        shuffle_rng = torch.Generator().manual_seed(8831)
        if "before" not in results:
            results["before"] = evaluate(m, test, scales)
            write_json(out / "learning_results.json", results)
        m.forward_model.train()
        losses = []
        for epoch in range(epochs):
            order = torch.randperm(len(train), generator=rng).tolist()
            epoch_losses = []
            for offset in range(0, len(order), 2):
                batch = [train[i] for i in order[offset:offset+2]]
                pred = predict(m, batch, shuffle=condition == "shuffled_action", generator=shuffle_rng)
                target = {k: torch.stack([r["target"][k] for r in batch]) for k in MODS}
                loss = learning_loss(pred, target, scales, effect=condition != "next_frame")
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite learning loss; no weights promoted")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, m.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
            losses.append(float(np.mean(epoch_losses)))
            print(json.dumps({"condition": condition, "epoch": epoch + 1, "loss": losses[-1]}), flush=True)
        results["conditions"][condition] = {"lr": lr, "optimizer": "fresh, identical in all offline arms",
            "trainable_parameters": sum(p.numel() for p in params),
            "command_projection_norm": (float(m.encoder_registry.encoders["motor"].command_projection.weight.detach().norm())
                                        if condition == "command_residual" else None),
            "losses": losses, "heldout": evaluate(m, test, scales), "train": evaluate(m, train, scales)}
        torch.save({"forward_model": m.forward_model.state_dict(), "prediction_heads": m.prediction_heads.state_dict(),
                    "motor_encoder": m.encoder_registry.encoders["motor"].state_dict()}, out / f"{condition}_experimental_weights.pt")
        write_json(out / "learning_results.json", results)
        del m, optimizer, params, parts, pred, loss
    checks = []
    for w in declaration["heldout_worlds"]:
        for route in ("raw_heads", "normalized_heads"):
            for sense in MODS:
                def score(c):
                    worlds = results["conditions"][c]["heldout"]
                    world = worlds[w] if w in worlds else worlds[str(w)]
                    return world[route][sense]["effect_error_ratio"]
                candidate = "command_residual" if "command_residual" in conditions else "paired_effect"
                controls = [score(c) for c in conditions if c != candidate]
                checks.append(score(candidate) < min(1.0, *controls))
    results["declared_gate_passed"] = bool(all(checks))
    results["production_changed"] = False
    write_json(out / "learning_results.json", results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("collect", "fit"))
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worlds", type=int, default=6)
    p.add_argument("--events", type=int, default=12)
    p.add_argument("--resume-collection", action="store_true")
    p.add_argument("--resume-fit", action="store_true")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--body-only", action="store_true")
    p.add_argument("--seconds", type=float, default=0.05,
                   help="outcome horizon in seconds; independent of the unchanged model and command clocks")
    p.add_argument("--history-steps", type=int, default=0,
                   help="retain this many STRICTLY PAST model steps of executed "
                        "command + raw body/touch fields per decision (2026-09-10 "
                        "sequence study); 0 = off, unchanged behavior")
    p.add_argument("--component-paths", action="store_true",
                   help="store grounded component wellness (tactile/capacity/posture) "
                        "at every path frame; the stored end wellness is then the final "
                        "path entry - parity by construction (2026-09-11 choice study)")
    p.add_argument("--rest", choices=("off", "on"), default="off",
                   help="alternate the trainer's active/rest bout schedule during "
                        "the trajectory so resting starting states are sampled")
    p.add_argument("--evaluation-only", action="store_true")
    p.add_argument("--independent-streams", action="store_true",
                   help="Also offset candidate and rest RNG streams for independent confirmation")
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.mode == "collect":
        collect(a.ckpt, a.out, a.worlds, a.events, resume=a.resume_collection,
                seconds=a.seconds, seed_offset=a.seed_offset, body_only=a.body_only,
                history_steps=a.history_steps, rest=a.rest,
                evaluation_only=a.evaluation_only, independent_streams=a.independent_streams,
                component_paths=a.component_paths)
    else:
        fit(a.ckpt, a.out, resume=a.resume_fit)


if __name__ == "__main__":
    main()
