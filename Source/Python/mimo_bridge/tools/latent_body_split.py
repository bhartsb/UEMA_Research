# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Split the shadow's recorded examples by body and measure the hierarchy's scales on them (review of Sep 30, 2026).

    python -m mimo_bridge.tools.latent_body_split --trajectory T.1 --trajectory T --holdout-body 7
        --base-config CFG --output DIR [--scale-rows 20000] [--seed 0]

Writes DIR/train.jsonl (every other body) and DIR/holdout.jsonl (--holdout-body), without the audit ledger, and
DIR/config.json: the base configuration with
  scales             the root mean square of every state, change and action channel over --scale-rows training
                     examples drawn at random, pooled over the parts the way the losses see them. With these scales
                     a future loss of 1.0 is what "nothing moves" scores and 0 is a perfect prediction; an action
                     loss of 1.0 is what "command zero" scores.
  code_bound         "tanh" (codes stay inside -1..1)
  hand_action_frame  "actual_wrist" (finger commands on the wrist the body had, trajectory.move_digits)
  digit_frame        --digit-frame wrist: digit rows relative to their wrist (index-level repair of Sep 30, 2026;
                     trajectory.DIGIT_FRAMES); absent (world) keeps the earlier meaning
  digit_scales       --digit-scales: the digit rows' own root mean squares, the other rows' scales then measured
                     without them (trajectory.row_scales); absent = one pooled scale for every row
  body_rows          --levels full_body (Oct 1, 2026; trajectory.BodyRows): the whole-body row convention with
                     part_scales, the root mean squares of each part group's rows (trajectory.PART_GROUPS: trunk, head,
                     gaze, arm, hand, digit, leg) in the frames the lessons use; --eye-row-convention (the trainer's
                     args.eye_row_convention when the rows were recorded); --contact-recorded-levels (the tree the
                     recording shadow used: its observations set contact bits only for the handles that tree owned;
                     omit when the rows were recorded by a full_body shadow); --future-rows (own: each part's future
                     lesson grades the rows of its own handles). The eyes' position change is not graded (the eye is
                     rigid in the head), so its scale is set to 1 cm instead of the root mean square of float noise;
                     a group whose contact the rows never recorded keeps a contact input scale of 1 (a binary event).
  horizon_steps      --horizon-steps H (Oct 1, 2026; trajectory.HoldRuns): every example spans one held order, the H
                     consecutive rows of one body that executed it (the recorded eye row averaged over the hold), so
                     the hierarchy learns the change over the whole 0.25 s hold instead of one 50 ms step; the scales
                     are measured on these examples. Absent (1): one example per row, as before. The seg41 recording
                     yields 4,352 hold examples per body from 21,782 rows (every complete run is 5 rows).
                     With H > 1 the split also writes DIR/witness_bank.pt: every hold example of every body (start and
                     end bones, the order, frames), the recorded endpoints the planner comparison builds its goals from
                     (mimo_bridge.tools.latent_body_compare, goal design recorded_v2).
DIR/split_receipt.json records the rows, bodies, examples and measured scales. Nothing is trained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trajectory", action="append", required=True, help="Oldest first; repeat for rotated files")
    ap.add_argument("--holdout-body", type=int, required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--layout", default="trunk_v1")
    ap.add_argument("--levels", default="arm,hand,index")
    ap.add_argument("--scale-rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--digit-frame", choices=("world", "wrist"), default="world")
    ap.add_argument("--digit-scales", action="store_true", help="Measure the digit rows' scales on their own")
    ap.add_argument("--eye-row-convention", choices=("legacy", "physical"), default=None)
    ap.add_argument("--contact-recorded-levels", default="",
                    help="Levels of the tree whose handles' contact the recording shadow observed (e.g. arm,hand,index)")
    ap.add_argument("--future-rows", choices=("own", "subtree"), default="own")
    ap.add_argument("--consequence-from-command", default="",
                    help="Part groups (e.g. digit) whose parts predict their consequence from their command, not their "
                         "code (learning.command_inputs; full_body only); empty = none")
    ap.add_argument("--horizon-steps", type=int, default=1,
                    help="Model steps per example: 1 = one row; H > 1 = one held order of H rows (trajectory.HoldRuns)")
    a = ap.parse_args(argv)
    full = a.levels == "full_body"
    if a.consequence_from_command and not full:
        raise SystemExit("--consequence-from-command needs --levels full_body (the whole-body rows name the commands)")
    if full and (a.digit_scales or a.eye_row_convention is None):
        raise SystemExit("--levels full_body measures part_scales (not --digit-scales) and needs --eye-row-convention")
    from lem.latent_body.learning import ShadowLearner, validate_config
    from lem.latent_body.trajectory import Example, HoldRuns, Observation, digit_wrists, make_batch
    from lem.latent_body.tree import arm_hand_tree, body_tree
    from mimo_bridge.body_layout import names_for
    from mimo_bridge.latent_body_runtime import example_record, jsonable
    if a.horizon_steps < 1:
        raise SystemExit("--horizon-steps must be a positive number of model steps")
    out = Path(a.output).resolve()
    if "saved" in {p.lower() for p in out.parts}:
        raise SystemExit("Keep hierarchy experiments outside Saved")
    out.mkdir(parents=True, exist_ok=False)
    base = json.loads(Path(a.base_config).read_text(encoding="utf-8"))
    if a.horizon_steps > 1:
        base = dict(base, horizon_steps=a.horizon_steps)
    tree = body_tree(names_for(a.layout), a.levels)
    rng = np.random.default_rng(a.seed)
    holds = HoldRuns(tree.handle_names, a.horizon_steps) if a.horizon_steps > 1 else None
    bank = []
    sample, seen, counts, examples, trained = [], 0, {}, {}, 0
    with (out / "train.jsonl").open("w", encoding="utf-8") as train, \
            (out / "holdout.jsonl").open("w", encoding="utf-8") as held:
        for path in a.trajectory:
            with open(path, encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row.pop("ledger", None)
                    body = int(row["start"]["env_id"])
                    counts[body] = counts.get(body, 0) + 1
                    if body != a.holdout_body:
                        seen += 1
                    ex = None
                    if holds is not None:
                        ex = holds.add(Example(*(Observation.from_dict(row[k]) for k in ("history", "start", "end")),
                                               torch.tensor(row["executed"]), row["action_contract"]))
                        if ex is None:
                            continue
                        bank.append((ex.start.env_id, ex.start.episode_id, ex.start.frame, ex.start.bones,
                                     ex.end.bones, ex.executed))
                        row = example_record(ex)
                    examples[body] = examples.get(body, 0) + 1
                    text = json.dumps(row, default=jsonable) + "\n"
                    if body == a.holdout_body:
                        held.write(text)
                        continue
                    train.write(text)
                    trained += 1
                    # reservoir sample of the training examples for the scale measurement
                    j = trained - 1 if len(sample) < a.scale_rows else int(rng.integers(trained))
                    if j < a.scale_rows:
                        if ex is None:
                            ex = Example(*(Observation.from_dict(row[k]) for k in ("history", "start", "end")),
                                         torch.tensor(row["executed"]), row["action_contract"])
                        if len(sample) < a.scale_rows:
                            sample.append(ex)
                        else:
                            sample[j] = ex
    runs = holds.finish() if holds is not None else None
    if a.holdout_body not in counts or len(sample) < min(1000, a.scale_rows):
        raise SystemExit("Too few rows: %s" % counts)
    hold = dict(horizon_steps=a.horizon_steps, examples_by_body={str(k): v for k, v in sorted(examples.items())},
                train_examples=trained, holdout_examples=examples.get(a.holdout_body, 0))
    if runs is not None:
        hold["run_lengths_by_body"] = {str(b): {str(k): v for k, v in r.items()} for b, r in runs.items()}
        hold["continuity_breaks_by_body"] = {str(b): v for b, v in sorted(holds.breaks.items())}
        torch.save(dict(env_id=torch.tensor([b[0] for b in bank]), episode_id=torch.tensor([b[1] for b in bank]),
                        frame=torch.tensor([b[2] for b in bank]), start=torch.stack([b[3] for b in bank]),
                        end=torch.stack([b[4] for b in bank]), executed=torch.stack([b[5] for b in bank]),
                        handle_names=list(tree.handle_names), horizon_steps=a.horizon_steps,
                        duration_s=float(base["duration_s"]), holdout_body=a.holdout_body,
                        trajectory=[str(p) for p in a.trajectory], action_contract=str(base["action_contract"])),
                   out / "witness_bank.pt")
        hold["witness_bank"] = str(out / "witness_bank.pt")
    unit = dict(state=[1.] * 12, change=[1.] * 6, action=[1.] * 8)
    if full:
        return full_body(a, base, tree, sample, seen, counts, out, unit, hold)
    groups = ("other", "digit") if a.digit_scales else ("other",)
    sums = {g: {k: torch.zeros(len(v), dtype=torch.float64) for k, v in unit.items()} for g in groups}
    rows = {g: {k: 0 for k in unit} for g in groups}
    digits = digit_wrists(tree.handle_names)
    ids = {"state": tree.subtree, "change": tree.subtree, "action": tree.owned}
    for i in range(0, len(sample), 512):
        b = make_batch(tree, sample[i:i + 512], scales=unit, duration_s=base["duration_s"],
                       action_contract=base["action_contract"], hand_frame="actual_wrist", digit_frame=a.digit_frame,
                       horizon_steps=a.horizon_steps)
        for key in unit:
            for name, v in getattr(b, key).items():
                flat = v.double().reshape(len(v), -1, len(unit[key]))
                is_digit = torch.tensor([h in digits for h in ids[key][name]])
                for g in groups:
                    pick = flat[:, is_digit] if g == "digit" else flat[:, ~is_digit] if a.digit_scales else flat
                    pick = pick.reshape(-1, len(unit[key]))
                    sums[g][key] += pick.square().sum(0)
                    rows[g][key] += len(pick)
    measured = {g: {k: [max(float(x), 1e-4) for x in (sums[g][k] / rows[g][k]).sqrt()] for k in unit} for g in groups}
    scales = measured["other"]
    extra = dict(digit_frame=a.digit_frame) if a.digit_frame != "world" else {}
    if a.digit_scales:
        extra["digit_scales"] = measured["digit"]
    config = dict(base, scales=scales, code_bound="tanh", hand_action_frame="actual_wrist", **extra,
                  measurement="Scales measured Sep 30, 2026 (mimo_bridge.tools.latent_body_split): root mean square "
                  "of each channel over %d random training examples of %s, finger commands on the actual wrist; "
                  "future loss 1.0 = 'nothing moves', action loss 1.0 = 'command zero'. Codes bounded by tanh. "
                  "Other settings from %s.%s" % (len(sample), ", ".join(a.trajectory), a.base_config, (
                      " Digit rows relative to their wrist (digit_frame wrist)." if a.digit_frame == "wrist" else "") + (
                      " Digit rows scaled on their own (digit_scales); scales measured on the other rows."
                      if a.digit_scales else "")))
    validate_config(config)
    ShadowLearner(tree, config)
    (out / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    receipt = dict(rows_by_body={str(k): v for k, v in sorted(counts.items())}, holdout_body=a.holdout_body,
                   train_rows=seen, scale_rows=len(sample), scales=scales, trajectory=a.trajectory,
                   digit_frame=a.digit_frame, digit_scales=measured.get("digit"), **hold)
    (out / "split_receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps(receipt))
    return 0


def full_body(a, base, tree, sample, seen, counts, out, unit, hold):
    """--levels full_body: part_scales per group (each handle's state and change rows read once, in the root that
    holds it; its command rows in the part that owns it) and the pooled scales over every row for reference."""
    from lem.latent_body.learning import ShadowLearner, validate_config
    from lem.latent_body.trajectory import GROUP_OF, PART_GROUPS, body_rows, make_batch
    from lem.latent_body.tree import arm_hand_tree
    names = tree.handle_names
    contract = str(base["action_contract"])
    wrist = "wrist_relative_joint_v1" in contract.split("+")
    recorded = (sorted(arm_hand_tree(names, tuple(a.contact_recorded_levels.split(","))).owners)
                if a.contact_recorded_levels else None)
    spec = dict(version="contract_v1", eye_row_convention=a.eye_row_convention, future_rows=a.future_rows,
                part_scales={g: unit for g in PART_GROUPS})
    if recorded is not None:
        spec["contact_recorded"] = recorded
    frames = dict(hand_action_frame="actual_wrist", digit_frame="wrist") if wrist else {}
    probe = dict(base, **frames, body_rows=spec)
    body = body_rows(probe)
    sums = {g: {k: torch.zeros(len(v), dtype=torch.float64) for k, v in unit.items()} for g in (*PART_GROUPS, "all")}
    rows = {g: {k: 0 for k in unit} for g in sums}
    for i in range(0, len(sample), 512):
        b = make_batch(tree, sample[i:i + 512], scales=unit, duration_s=base["duration_s"],
                       action_contract=contract, hand_frame=probe.get("hand_action_frame", "recorded"),
                       digit_frame=probe.get("digit_frame", "world"), body=body, horizon_steps=a.horizon_steps)
        for key in unit:
            dim = len(unit[key])
            parts = tree.roots if key in ("state", "change") else [n.name for n in tree.nodes]
            for name in parts:
                ids = tree.subtree[name] if key in ("state", "change") else tree.owned[name]
                values = getattr(b, key)[name].double().reshape(-1, len(ids), dim)
                for k, h in enumerate(ids):
                    for g in (GROUP_OF[names[h]], "all"):
                        sums[g][key] += values[:, k].square().sum(0)
                        rows[g][key] += len(values)
    measured = {g: {k: [max(float(x), 1e-4) for x in (sums[g][k] / rows[g][k]).sqrt()] for k in unit} for g in sums}
    measured["gaze"]["change"][:3] = [1.0, 1.0, 1.0]
    for g in sums:
        # a group whose contact the rows never recorded reads 0 on every row; a later recording that does observe it
        # (a full_body shadow) must not divide a touch by the 0.0001 floor: a binary event keeps the scale 1
        if measured[g]["state"][11] <= 1e-4:
            measured[g]["state"][11] = 1.0
    spec["part_scales"] = {g: measured[g] for g in PART_GROUPS}
    config = dict(base, **dict(frames, hand_action_frame="actual_wrist"), scales=measured["all"], code_bound="tanh",
                  body_rows=spec,
                  measurement="Whole-body scales measured %s (mimo_bridge.tools.latent_body_split --levels full_body): "
                  "root mean square of each channel per part group over %d random training examples of %s, in the "
                  "frames of the command contract (trajectory.BodyRows); 'scales' is the same measurement pooled over "
                  "every row, for reference (row_scales uses part_scales). Future lesson per part over its own "
                  "handles (future_rows %s). Codes bounded by tanh. Other settings from %s." % (
                      __import__("time").strftime("%b %d, %Y"), len(sample), ", ".join(a.trajectory), a.future_rows,
                      a.base_config))
    config.pop("digit_scales", None)
    if a.consequence_from_command:
        config["consequence_from_command"] = a.consequence_from_command.split(",")
    validate_config(config)
    ShadowLearner(tree, config)
    (out / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    receipt = dict(rows_by_body={str(k): v for k, v in sorted(counts.items())}, holdout_body=a.holdout_body,
                   train_rows=seen, scale_rows=len(sample), levels=a.levels, part_scales=spec["part_scales"],
                   pooled_scales=measured["all"], contact_recorded=recorded, eye_row_convention=a.eye_row_convention,
                   future_rows=a.future_rows, trajectory=a.trajectory, **hold)
    (out / "split_receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
