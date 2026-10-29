# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Causal-control test of the latent body hierarchy (the operator's rule of Sep 27, 2026): from one saved moment,
change only one part's code, execute the decoded command for one model step, and check that the body's actual
change differs with the code and that the hierarchy predicted the change.

    python -m mimo_bridge.tools.latent_body_intervention --checkpoint CK --hierarchy LATEST_PT --trajectory TRAJ
        --output OUT [--bodies 1,4,5,7] [--codes 16] [--warmup-steps 3] [--radius-rows 2000] [--seed 0]
        [--code-sampling auto|box|encoded|gaussian]

Per body, for every part of the tree (arms with their children decoded from the arm's code; hands and index fingers
with the roots at their centre codes and only the part's own code replaced): K codes, the centre code and K-1 drawn
uniformly within 2 standard deviations per dimension around it, both measured on the codes the hierarchy's own encoder
gives the newest real examples (--radius-rows rows of the shadow's trajectory.jsonl). --centre mean (the default since
Sep 30, 2026) centres on the average real code; --centre zero on the zero code, as the Sep 30 morning run did, when the
live shadow's average code sat about half a spread from zero. Each code is decoded top-down, the owned handles' predicted
commands become handle targets (lem.planner.latent_rollout.handle_targets; every other handle holds), and one model
step runs from the same saved moment (gof_probe_harness.Session.run_arm). The actual change of the part's subtree is
read from the next frame's bones, in the units and row convention the hierarchy trains on (cm and rotation-vector
radians; under digit_frame "wrist", the index-level repair of Sep 30, 2026, a finger row is its movement relative to
its wrist, trajectory.change_rows). spread_world_cm is the spread in world coordinates, as runs before that repair read.

Readings per part (displacement = the subtree's cm xyz; turn = rotation vector):
  spread_cm   mean distance between each code's actual subtree displacement and the centre code's: does the code move
              the body at all (0 = every code gives the same movement)
  repeat_cm   the centre code run twice from the snapshot: a physical replay must be exact (0)
  hold_cm     the subtree's displacement under the hold order (what the body does with no new order)
  match       correlation over codes of predicted and actual displacement (flattened, each coordinate centred over
              codes): 1 = the hierarchy predicts which way each code moves the part, 0 = no relation
  shuffled    the same correlation with each prediction paired to another code's outcome, 200 shuffles: the chance
              distribution; match_percentile is match's rank in it
  CONTROLLABLE when spread_cm > repeat_cm and match exceeds the shuffles' 95th percentile. A part whose codes do not
  change its movement, or whose predictions do not track the movement they cause, fails.
WHOLE BODY (Oct 1, 2026; a hierarchy of the full_body tree, trajectory.BodyRows). Every part of the tree is tested.
The eyes' command reaches the eye servo (Session.run_arm with gaze_rad from the decoded eye row, as
lem_model._gaze_from_handle reads it) when the tree owns the eyes; otherwise run_arm holds them. A part whose subtree is
only the head, the eyes or a foot (TURN_GRADED) is graded on its turn (rotation vectors, radians: spread_rad,
repeat_rad, match_turn and its shuffle line), because its movement is a turn about hinges at or near its body origin
(MIMo_modelv2.xml: the head hinges 0-2.45 cm from the head origin, the ankle hinges 1.5 cm from the foot origin) and the
eyes' rows are head-relative (no displacement); every other part on its displacement: in the world for a whole-body
hierarchy (the predicted world displacement of each handle from the part that owns it, placed through its parents'
predicted poses: trajectory.predicted_world_steps; carried and relative rows hold no displacement of their own), in the
hierarchy's rows for the pilot, as before. Both readings are recorded for every part; graded_on names the one that
decides. summary.by_kind counts parts by name without side.
HOLD HORIZON (Oct 1, 2026; config key horizon_steps, trajectory.hold_example). A hierarchy that predicts the change
over H model steps (one held order) is tested over H steps: each decoded order is held H model steps from the saved
moment and the change from the start to the H-th step is compared with the hierarchy's own H-step prediction. The hold
runs through Session.physics_only with render=True (the body follows the held order; the model is not stepped, its
outputs do not reach the body during a given order; the last env step of every model step is rendered, as run_arm
renders it, so the eye reflex, which observes rendered frames only, acts as in the recorded holds), once per code;
hold_execution_check records, for the first part of each body, the centre code's end bones through physics_only against
Session.run_arm with the model observing every step (0 = the model's observing does not change the physics). A one-step
hierarchy runs exactly as before (run_arm, one step).
CODE SAMPLING (--code-sampling, Oct 1, 2026; lem.latent_body.code_range). "box" is the earlier mode above (the K-1
codes drawn per dimension within 2 standard deviations). Those codes lay a median 4.9-26.6 Mahalanobis units from the
whole-body fit's own codes (2.2-2.5 for those), where the decoder has learned nothing. "encoded": the K-1 codes are
codes the hierarchy's encoder gave the --radius-rows real examples, only those within the 95th percentile of the
encoder's own Mahalanobis distances (the range). "gaussian": drawn from those codes' mean and covariance inside the
range. "auto" (the default): encoded for a whole-body hierarchy, box for the pilot. The centre code is the first code
in every mode. The result records the mode, the range of every part (code_range) and, per part, its trial codes'
distances against the range (code_range in each part's reading).
Nothing is trained; the production checkpoint and the hierarchy file are read only (the hierarchy is copied first).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
import zlib
from pathlib import Path

import numpy as np
import torch


def tail_rows(path, count):
    """The last `count` lines of a large JSONL file, read from its end."""
    size, block, data = os.path.getsize(path), 1 << 22, b""
    with open(path, "rb") as f:
        pos = size
        while pos > 0 and data.count(b"\n") <= count:
            pos = max(0, pos - block)
            f.seek(pos)
            data = f.read(size - pos)
    lines = [l for l in data.splitlines() if l.strip()]
    if pos > 0:
        lines = lines[1:]                     # the first line may be cut
    return [json.loads(l) for l in lines[-count:]]


def real_codes(learner, rows):
    """{part: [N, latent]} the encoder's codes of the usable real rows, and N."""
    from lem.latent_body.trajectory import Example, Observation
    examples = []
    for row in rows:
        try:
            ex = Example(*(Observation.from_dict(row[k]) for k in ("history", "start", "end")),
                         torch.tensor(row["executed"]), row["action_contract"])
            learner.batch([ex])
            examples.append(ex)
        except Exception:
            continue
    if len(examples) < 50:
        raise ValueError("Fewer than 50 usable real examples to measure the code range")
    with torch.no_grad():
        return learner.model.encode(learner.batch(examples)), len(examples)


def code_scales(learner, rows, codes=None):
    if codes is None:
        codes, count = real_codes(learner, rows)
    else:
        count = len(next(iter(codes.values())))
    # how far the zero code (the reference and the planner's always-included candidate) sits from the real codes:
    # the mean code's distance from zero in units of the codes' spread, averaged over dimensions (review of Sep 30)
    offsets = {name: float((z.mean(0).abs() / z.std(0).clamp_min(1e-6)).mean()) for name, z in codes.items()}
    return ({name: z.std(0).clamp_min(1e-6) for name, z in codes.items()}, count, offsets,
            {name: z.mean(0, keepdim=True) for name, z in codes.items()})


def decode_with(model, state, recent, roots, override):
    """hierarchy.decode, with one part's code replaced after its parent produced it."""
    contexts = model.context(state)
    intentions, predictions = dict(roots), {}
    for node in model.tree.nodes:
        z = override.get(node.name, intentions[node.name])
        pred = model.parts[node.name].predict(state[node.name], recent[node.name], contexts[node.name], z)
        predictions[node.name] = pred
        intentions.update(pred.children)
    return predictions


def actual_change(tree, before, after, ids, digit_frame="world", body=None):
    """The subtree's one-step change in the hierarchy's own row convention (digit rows relative to their wrist under
    digit_frame "wrist"; trajectory.change_rows), the units its predictions use."""
    from lem.latent_body.trajectory import change_rows
    return change_rows(before, after, ids, tree.handle_names, digit_frame, body)


TURN_GRADED = frozenset({"head", "gaze", "foot_L", "foot_R"})


def graded_on(tree, node):
    """"turn" when every handle of the part's subtree is in TURN_GRADED, else "move"."""
    names = {tree.handle_names[i] for i in tree.subtree[node]}
    return "turn" if names <= TURN_GRADED else "move"


def corr(x, y):
    x, y = x - x.mean(0), y - y.mean(0)
    den = float(x.norm() * y.norm())
    return float((x * y).sum()) / den if den > 0 else 0.0


def test_body(session, learner, scales_z, args, rng, centres=None, code_range=None):
    """code_range (lem.latent_body.code_range.CodeRange): required unless args.code_sampling is "box" (absent = box)."""
    from mimo_bridge.latent_body_runtime import observation, observed_contract
    from mimo_bridge.tools.latent_body import warm_start
    from lem.latent_body.trajectory import predicted_rows, predicted_world_steps, row_options, row_scales, state_inputs
    from lem.planner.latent_rollout import handle_targets
    tree = learner.tree
    if tuple(session.model.handle_names) != tree.handle_names:
        raise ValueError("Hierarchy and checkpoint use different body layouts")
    if learner.config["action_contract"] != observed_contract(session.frames[0]):
        raise ValueError("Hierarchy and world use different command meanings")
    history, current = warm_start(session, args.warmup_steps, tree)
    snap = session.snapshot()
    template = np.asarray(session.hold_order()[0], dtype=np.float32)
    scale = learner.config["scales"]
    rows = row_options(learner.config)
    state, recent = state_inputs(tree, history, current, scale, **rows)
    frame, body = rows["digit_frame"], rows["body"]
    eyes = "gaze" in tree.owners
    horizon = int(learner.config.get("horizon_steps", 1))
    sampling = getattr(args, "code_sampling", "box")
    if sampling != "box" and code_range is None:
        raise ValueError(f"Code sampling {sampling!r} needs the encoder's measured code range")
    # encoded and gaussian codes use their own generator, seeded from the test's; the box keeps the earlier draws
    generator = None if sampling == "box" else torch.Generator().manual_seed(int(rng.integers(2 ** 31)))
    check = {}

    def execute(order, observed=None):
        # the eyes follow the decoded eye row only when the tree owns them (otherwise run_arm holds them)
        gaze = None
        if eyes:
            gaze = np.asarray(session.model._gaze_from_handle(
                torch.from_numpy(np.asarray(order, dtype=np.float32)))).reshape(-1)[:2]
        kw = {} if gaze is None else dict(gaze_rad=gaze)
        if horizon == 1 or observed:
            session.run_arm(snap, order.astype(np.float32), steps=horizon, **kw)
        else:
            session.physics_only(snap, order.astype(np.float32), horizon, render=True, **kw)
        return observation(session.frames[0], tree, args.warmup_steps + horizon, 0)

    hold_after = execute(template[None])
    if centres is None or getattr(args, "centre", "mean") == "zero":
        centres = {n.name: torch.zeros(1, learner.model.latent_dim) for n in tree.nodes}
    centre_roots = {r: centres[r] for r in tree.roots}
    out = {}
    for node in tree.nodes:
        ids = tree.subtree[node.name]
        std = scales_z[node.name]
        codes = [centres[node.name].clone()]
        if sampling == "box":
            for _ in range(args.codes - 1):
                u = torch.from_numpy(rng.uniform(-2.0, 2.0, size=(1, learner.model.latent_dim)).astype(np.float32))
                codes.append(centres[node.name] + u * std)
        else:
            codes += list(code_range.sample(node.name, args.codes - 1, sampling, generator)[:, None])
        trial_range = None if code_range is None else code_range.report(node.name, torch.cat(codes[1:]))
        predicted, actual, world, predicted_world = [], [], [], []
        for k, z in enumerate(codes):
            if node.parent is None:
                roots, override = dict(centre_roots, **{node.name: z}), {}
            else:
                roots, override = centre_roots, {node.name: z}
            with torch.no_grad():
                preds = decode_with(learner.model, state, recent, roots, override)
            order = handle_targets(tree, preds, current, template, scale,
                                   learner.config.get("hand_action_frame", "recorded"), **rows).numpy()[None]
            after = execute(order)
            if horizon > 1 and k == 0 and not check:
                observed = execute(order, observed=True)
                check.update(part=node.name, horizon_steps=horizon,
                             max_bone_difference=float((observed.bones - after.bones).abs().max()))
            actual.append(actual_change(tree, current, after, ids, frame, body))
            world.append(actual_change(tree, current, after, ids))
            # under future_rows "own" each row is predicted by the part that owns its handle (trajectory.predicted_rows)
            predicted.append(predicted_rows(tree, preds, node.name, body)[0]
                             * row_scales(tree.handle_names, ids, scale, "change", rows["digit_scales"], body))
            if body is not None:
                steps = predicted_world_steps(tree, preds, current, scale, frame, rows["digit_scales"], body)
                predicted_world.append(torch.stack([steps[i][0] for i in ids]))
            if k == 0:
                again = execute(order)
                repeat = actual_change(tree, current, again, ids, frame, body)
                repeat_world = actual_change(tree, current, again, ids)
        A, P = torch.stack(actual), torch.stack(predicted)
        # displacement: in the hierarchy's rows (the pilot), or in the world under the whole-body rows, where carried
        # and relative rows hold no displacement of their own (trajectory.predicted_world_steps)
        D = torch.stack(world)[..., :3] if body is not None else A[..., :3]
        PD = torch.stack(predicted_world) if body is not None else P[..., :3]
        act = D.reshape(len(codes), -1)
        pred = PD.reshape(len(codes), -1)
        spread = float((D[1:] - D[0]).norm(dim=-1).mean())
        repeat_cm = float(((repeat_world if body is not None else repeat)[:, :3] - D[0]).norm(dim=-1).mean())
        hold_cm = float(actual_change(tree, current, hold_after, ids, frame, body)[:, :3].norm(dim=-1).mean())
        # the same spread in world coordinates, comparable with runs made before the wrist digit frame
        spread_world = float((torch.stack(world)[1:, :, :3] - torch.stack(world)[0, :, :3]).norm(dim=-1).mean())
        match = corr(pred, act)
        shuffled = []
        for _ in range(200):
            perm = rng.permutation(len(codes))
            shuffled.append(corr(pred, act[perm]))
        p95 = float(np.quantile(shuffled, 0.95))
        # the turn reading (rotation vectors, radians); its shuffles use their own generator, so the trial codes and
        # displacement shuffles draw the same random numbers as before this reading existed
        turn_rng = np.random.default_rng([int(getattr(args, "seed", 0)), zlib.crc32(node.name.encode())])
        act_t, pred_t = A[..., 3:6].reshape(len(codes), -1), P[..., 3:6].reshape(len(codes), -1)
        spread_rad = float((A[1:, :, 3:6] - A[0, :, 3:6]).norm(dim=-1).mean())
        repeat_rad = float((repeat[:, 3:6] - A[0, :, 3:6]).norm(dim=-1).mean())
        match_turn = corr(pred_t, act_t)
        p95_turn = float(np.quantile([corr(pred_t, act_t[turn_rng.permutation(len(codes))]) for _ in range(200)], 0.95))
        grade = graded_on(tree, node.name)
        decided = (spread > repeat_cm and match > p95) if grade == "move" else (
            spread_rad > repeat_rad and match_turn > p95_turn)
        out[node.name] = dict(
            level="root" if node.parent is None else "child of " + node.parent, handles=list(ids), codes=len(codes),
            spread_cm=spread, spread_world_cm=spread_world, digit_frame=frame, repeat_cm=repeat_cm, hold_cm=hold_cm,
            match=match,
            shuffled_mean=float(np.mean(shuffled)), shuffled_p95=p95,
            match_percentile=float(np.mean(np.asarray(shuffled) < match)),
            spread_rad=spread_rad, repeat_rad=repeat_rad, match_turn=match_turn, shuffled_p95_turn=p95_turn,
            graded_on=grade,
            predicted_step_cm=float(pred.reshape(len(codes), len(ids), 3).norm(dim=-1).mean()),
            actual_step_cm=float(act.reshape(len(codes), len(ids), 3).norm(dim=-1).mean()),
            horizon_steps=horizon, controllable=bool(decided), code_sampling=sampling, code_range=trial_range)
    if check:
        out["_hold_execution_check"] = check
    return out


def part_kinds(results):
    """{kind: tested, controllable} with a part's kind its name without the side (index_L -> index)."""
    out = {}
    for body in results.values():
        for name, p in body.items():
            if name.startswith("_"):
                continue
            kind = name[:-2] if name[-2:] in ("_L", "_R") else name
            row = out.setdefault(kind, dict(tested=0, controllable=0))
            row["tested"] += 1
            row["controllable"] += int(bool(p["controllable"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for flag in ("checkpoint", "hierarchy", "trajectory", "output"):
        ap.add_argument("--" + flag, required=True)
    ap.add_argument("--bodies", default="1,4,5,7")
    ap.add_argument("--codes", type=int, default=16)
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--radius-rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--centre", choices=("mean", "zero"), default="mean",
                    help="Centre of the trial codes: the average real code (default) or the zero code")
    ap.add_argument("--code-sampling", choices=("auto", "box", "encoded", "gaussian"), default="auto",
                    help="Trial codes: auto (encoded for a whole-body hierarchy, box for the pilot), box, encoded, gaussian")
    a = ap.parse_args(argv)
    if a.codes < 4:
        raise SystemExit("Declare at least 4 codes per part")
    out = Path(a.output).resolve()
    if "saved" in {p.lower() for p in out.parts}:
        raise SystemExit("Keep test outputs outside Saved")
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    hier = out / "hierarchy_used.pt"
    shutil.copy2(a.hierarchy, hier)                # the live shadow rewrites latest.pt every update
    from mimo_bridge.tools.latent_body import load_learner
    from mimo_bridge.tools import gof_probe_harness as H
    learner = load_learner(hier)
    learner.model.eval()
    from lem.latent_body.code_range import CodeRange, resolve_sampling, summarize_reports
    requested = a.code_sampling
    a.code_sampling = resolve_sampling(requested, learner.config)
    codes, _ = real_codes(learner, tail_rows(a.trajectory, a.radius_rows))
    scales_z, n_rows, offsets, centres = code_scales(learner, None, codes)
    code_range = CodeRange(codes)
    if a.centre == "zero" and a.code_sampling != "box":
        raise SystemExit("The zero centre is the box mode's (Sep 30 morning); encoded and gaussian codes centre on the "
                         "encoder's mean code")
    rng = np.random.default_rng(a.seed)
    old_root = H.PROBES_ROOT
    H.PROBES_ROOT = out / "scenes"
    results = {}
    try:
        frozen = H.load_frozen(Path(a.checkpoint))
        episode, reason = H.load_episode(frozen)
        if episode is None:
            raise ValueError(reason)
        for b in [int(x) for x in a.bodies.split(",")]:
            session = H.Session(frozen, episode=copy.deepcopy(episode), body=b)
            try:
                session.observe()
                results[str(b)] = test_body(session, learner, scales_z, a, rng, centres, code_range)
            finally:
                session.close()
            (out / "partial.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    finally:
        H.PROBES_ROOT = old_root
    parts = [v for body in results.values() for k, v in body.items() if not k.startswith("_")]
    summary = dict(parts_tested=len(parts), controllable=sum(p["controllable"] for p in parts),
                   by_level={lvl: dict(tested=sum(1 for p in parts if p["level"].startswith(lvl)),
                                       controllable=sum(1 for p in parts if p["level"].startswith(lvl) and p["controllable"]))
                             for lvl in ("root", "child")},
                   by_kind=part_kinds(results))
    result = dict(test="latent_body_intervention_v1", checkpoint=str(Path(a.checkpoint).resolve()),
                  hierarchy=str(Path(a.hierarchy).resolve()), hierarchy_updates=learner.updates,
                  horizon_steps=int(learner.config.get("horizon_steps", 1)),
                  hold_execution_checks={b: v["_hold_execution_check"] for b, v in results.items()
                                         if "_hold_execution_check" in v},
                  code_range_rows=n_rows, zero_code_offset_in_spreads=offsets, centre=a.centre, codes_per_part=a.codes, warmup_steps=a.warmup_steps, seed=a.seed,
                  code_sampling=a.code_sampling, code_sampling_requested=requested, code_range=code_range.receipt(),
                  code_range_summary=summarize_reports([v["code_range"] for body in results.values()
                                                        for k, v in body.items() if not k.startswith("_")]),
                  data_quality=learner.config["data_quality"], summary=summary, bodies=results,
                  elapsed_s=time.time() - started, authority="test_world_only", production_control=False)
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
