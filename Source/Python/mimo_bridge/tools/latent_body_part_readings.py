# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Offline readings of every part of a latent body hierarchy on held-out rows (whole-body extension, Oct 1, 2026).

    python -m mimo_bridge.tools.latent_body_part_readings --hierarchy H.pt --holdout HOLDOUT.jsonl --output OUT.json
        [--rows 3000] [--every 30] [--codes 16] [--seed 0]

CPU only; no checkpoint of the production model and no world. The index-level readings
(latent_body_index_readings.py) for every part of the tree. Per part P (its owned handles O):
  graded_on                 "move" when the position of any handle of O is a learned command (an executor reads it:
                            trajectory.BodyRows.position_read; every position without the whole-body rows), else
                            "turn"; the readings below use those numbers (cm or radians)
  code_reads_parent_step_r2 (non-root parts) how much of the parent part's first handle's one-step WORLD displacement
                            a linear read-out of P's code recovers (R^2 on the second half of the rows, fitted on the
                            first half): the carry P's code stores (index, Sep 30 refit: 0.62 / 0.47)
  code_reads_own_step_r2    the same for P's own handles' one-step change in the hierarchy's rows
  direction_agreement       over the causal test's code range (the mean code and codes-1 codes within 2 standard
                            deviations per dimension; roots at their mean codes; P's code replaced), the correlation of
                            P's predicted change of its own handles with P's decoded own command offset, median over
                            rows (1 = the predicted movement points where the command points)
  command_miss / command_miss_if_zero   P's own command decoded from the row's own code against the recorded command,
                            median distance; and the same for "command zero" (the handle's own place or pose)
  decoded_target_spread / predicted_move_spread   mean distance of each trial code's decoded own command / predicted
                            own change from the mean code's, median over rows
  wrist_move_agreement      (arm parts) the predicted wrist movement (the arm part's own row for the pilot; the
                            hand's, placed in the world, for the whole body) against the hand child's decoded
                            wrist target over the arm code range (the arm-level control reading of Oct 1, 2026; refit
                            full fit 0.73 / 0.65, index repair 0.33 / 0.02)
Units are the hierarchy's own row convention.
CODE SAMPLING (--code-sampling, Oct 1, 2026; lem.latent_body.code_range): the trial codes of direction_agreement,
the spreads and wrist_move_agreement. "box" (the earlier mode): the mean code and codes-1 codes drawn uniformly within
2 standard deviations per dimension. "encoded": the mean code and codes-1 codes the encoder produced for OTHER held-out
rows, only those within the 95th percentile of the encoder's own Mahalanobis distances (the range). "gaussian": drawn
from the fitted mean and covariance inside the range. "auto" (the default): encoded for a whole-body hierarchy, box for
the pilot. Box codes lay a median 4.9-26.6 Mahalanobis units from the whole-body fit's own codes (2.2-2.5 for those);
agents/latent_code_range_2026-10-01.md. Every result records the mode (code_sampling) and, per part, code_range: the
trial codes' distances against the range. The trial codes of one row are decoded in one batch.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch


def _r2(x, y):
    x = torch.cat((x, torch.ones(len(x), 1)), 1).double()
    y = y.double()
    n = len(x) // 2
    w = torch.linalg.lstsq(x[:n], y[:n]).solution
    res = y[n:] - x[n:] @ w
    return float(1 - res.square().sum() / (y[n:] - y[n:].mean(0)).square().sum().clamp_min(1e-12))


def _corr(x, y):
    x, y = x - x.mean(0), y - y.mean(0)
    den = float(x.norm() * y.norm())
    return float((x * y).sum()) / den if den > 0 else 0.0


def graded_columns(tree, part, body):
    names = tree.handle_names
    own = [names[i] for i in tree.owned[part]]
    read = (lambda h: True) if body is None else (lambda h: h in body.position_read)
    return ("move", slice(0, 3)) if any(read(h) for h in own) else ("turn", slice(3, 6))


def trial_codes(mode, k, centre, spread, rng, generator, code_range, part, row, latent_dim):
    """[k, latent] codes for one row: the mean code first, then k-1 codes of the declared mode. The box draws use the
    tool's own numpy generator in the earlier order, so a box reading repeats the earlier numbers."""
    if mode == "box":
        draws = [torch.zeros(1, latent_dim)] + [torch.from_numpy(rng.uniform(-2, 2, size=(1, latent_dim)).astype(
            np.float32)) for _ in range(k - 1)]
        return centre + torch.cat(draws) * spread
    return torch.cat((centre, code_range.sample(part, k - 1, mode, generator, exclude=row)))


@torch.no_grad()
def part_readings(learner, examples, *, every=30, codes=16, seed=0, parts=None, code_sampling="auto"):
    import zlib
    from lem.latent_body.code_range import CodeRange, resolve_sampling, summarize_reports
    from lem.latent_body.trajectory import change_rows, predicted_world_steps, row_options, row_scales
    from mimo_bridge.tools.latent_body_intervention import decode_with
    if len(examples) < 4:
        raise ValueError("Readings need at least 4 held-out rows")
    tree, cfg, model = learner.tree, learner.config, learner.model
    names, rows = tree.handle_names, row_options(cfg)
    body = rows["body"]
    mode = resolve_sampling(code_sampling, cfg)
    batch = learner.batch(examples)
    z = model.encode(batch)
    code_range = CodeRange(z) if mode != "box" or len(examples) >= 50 else None
    roots = {r: z[r].mean(0, keepdim=True) for r in tree.roots}
    out = {}
    for part in parts or [n.name for n in tree.nodes]:
        rng = np.random.default_rng(seed)
        generator = torch.Generator().manual_seed(int(seed) * 1_000_003 + zlib.crc32(part.encode()))
        node = tree.by_name[part]
        own, sub = list(tree.owned[part]), list(tree.subtree[part])
        kind, cols = graded_columns(tree, part, body)
        sa = row_scales(names, own, cfg["scales"], "action", rows["digit_scales"], body)
        sc = row_scales(names, sub, cfg["scales"], "change", rows["digit_scales"], body)
        k_own = [sub.index(i) for i in own]
        reading = dict(graded_on=kind, handles=[names[i] for i in own])
        own_step = torch.stack([change_rows(ex.start, ex.end, own, names, rows["digit_frame"], body)[:, cols].flatten()
                                for ex in examples])
        reading["code_reads_own_step_r2"] = _r2(z[part], own_step)
        if node.parent is not None:
            p = tree.owned[node.parent][0]
            step = torch.stack([ex.end.bones[p, 4:7] - ex.start.bones[p, 4:7] for ex in examples])
            reading["code_reads_parent_step_r2"] = _r2(z[part], step)
        centre, spread = z[part].mean(0, keepdim=True), z[part].std(0, keepdim=True)
        arm_hand = "hand_" + part[-1] if part.startswith("arm_") and "hand_" + part[-1] in tree.by_name else None
        agree, targets, moves, miss, miss0, wrist_agree, drawn = [], [], [], [], [], [], []
        for r in range(0, len(examples), every):
            state = {k: v[r][None].expand(codes, -1) for k, v in batch.state.items()}
            recent = {k: v[r][None].expand(codes, -1) for k, v in batch.history.items()}
            code = trial_codes(mode, codes, centre, spread, rng, generator, code_range, part, r, model.latent_dim)
            drawn.append(code[1:])
            many = {k: v.expand(codes, -1) for k, v in roots.items()}
            preds = (decode_with(model, state, recent, dict(many, **{part: code}), {}) if node.parent is None
                     else decode_with(model, state, recent, many, {part: code}))
            change = preds[part].change.reshape(codes, len(sub), 6) * sc
            action = preds[part].action.reshape(codes, len(own), 8) * sa
            pc = change[:, k_own][..., cols].reshape(codes, -1)
            tg = action[..., cols].reshape(codes, -1)
            agree.append(_corr(pc, tg))
            targets.append(float((tg[1:] - tg[0]).reshape(codes - 1, -1, 3).norm(dim=-1).mean()))
            moves.append(float((pc[1:] - pc[0]).reshape(codes - 1, -1, 3).norm(dim=-1).mean()))
            if arm_hand is not None:
                w = names.index("wrist_" + part[-1])
                hand_own = list(tree.owned[arm_hand])
                sw = row_scales(names, [w], cfg["scales"], "action", rows["digit_scales"], body)[0]
                wt = preds[arm_hand].action.reshape(codes, len(hand_own), 8)[:, hand_own.index(w), :3] * sw[:3]
                # whole body: the wrist's predicted world step by its owner, the hand (predicted_world_steps)
                wm = (change[:, sub.index(w), :3] if body is None else predicted_world_steps(
                    tree, preds, examples[r].start, cfg["scales"], rows["digit_frame"], rows["digit_scales"],
                    body)[w])
                wrist_agree.append(_corr(wm, wt))
            one = {k: v[:1] for k, v in state.items()}
            posterior = model.parts[part].predict(one[part], recent[part][:1], model.context(one)[part], z[part][r][None])
            recorded = (batch.action[part][r].reshape(len(own), 8) * sa)[:, cols]
            decoded = (posterior.action.reshape(len(own), 8) * sa)[:, cols]
            miss.append(float((decoded - recorded).norm(dim=-1).mean()))
            miss0.append(float(recorded.norm(dim=-1).mean()))
        # the median's row-sampling noise: its standard deviation over 2,000 resamplings of the rows (own generator)
        boot = np.random.default_rng(0)
        reading["direction_agreement_noise"] = float(np.std([np.median(np.asarray(agree)[boot.integers(
            len(agree), size=len(agree))]) for _ in range(2000)]))
        reading.update(direction_agreement=float(np.median(agree)), command_miss=float(np.median(miss)),
                       command_miss_if_zero=float(np.median(miss0)),
                       decoded_target_spread=float(np.median(targets)), predicted_move_spread=float(np.median(moves)),
                       units="cm" if kind == "move" else "rad", rows_sampled=len(agree), code_sampling=mode,
                       code_range=None if code_range is None else code_range.report(part, torch.cat(drawn)))
        if wrist_agree:
            reading["wrist_move_agreement"] = float(np.median(wrist_agree))
            reading["wrist_move_agreement_noise"] = float(np.std([np.median(np.asarray(wrist_agree)[boot.integers(
                len(wrist_agree), size=len(wrist_agree))]) for _ in range(2000)]))
        out[part] = reading
    return dict(digit_frame=rows["digit_frame"], body_rows=body is not None, examples=len(examples), codes=codes,
                seed=seed, code_sampling=mode, code_sampling_requested=code_sampling,
                code_range_summary=summarize_reports([p["code_range"] for p in out.values()]), parts=out)


def main(argv=None):
    from mimo_bridge.tools.latent_body import load_learner, read_examples
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for flag in ("hierarchy", "holdout", "output"):
        ap.add_argument("--" + flag, required=True)
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--codes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--code-sampling", choices=("auto", "box", "encoded", "gaussian"), default="auto",
                    help="Trial codes: auto (encoded for a whole-body hierarchy, box for the pilot), box, encoded, gaussian")
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    learner = load_learner(a.hierarchy)
    learner.model.eval()
    examples = list(itertools.islice(read_examples(a.holdout), a.rows))
    result = dict(part_readings(learner, examples, every=a.every, codes=a.codes, seed=a.seed,
                                code_sampling=a.code_sampling),
                  hierarchy=str(Path(a.hierarchy).resolve()), holdout=str(Path(a.holdout).resolve()),
                  updates=learner.updates, authority="none")
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result["parts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
