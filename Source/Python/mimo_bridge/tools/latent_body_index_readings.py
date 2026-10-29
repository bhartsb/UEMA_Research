# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Offline readings of the latent hierarchy's digit level on held-out rows (index-level repair, Sep 30, 2026).

    python -m mimo_bridge.tools.latent_body_index_readings --hierarchy H.pt --holdout HOLDOUT.jsonl --output OUT.json
        [--rows 3000] [--every 30] [--codes 16] [--seed 0]

CPU only; no checkpoint of the production model and no world. Per index part (index_L, index_R):
  code_reads_wrist_step_r2   how much of the WRIST's own one-step world displacement a linear read-out of the index
                             code recovers (R^2 on the second half of the rows, fitted on the first half). The wrist is
                             moved by the arm and hand levels; an index code that carries it has learned movement its
                             finger command cannot make (Sep 30 refit: 0.62 left, 0.47 right).
  direction_agreement        over the causal test's code range (the mean code and 15 codes within 2 standard
                             deviations per dimension, every other part at its mean code), the correlation between the
                             index level's predicted finger movement and its own decoded finger command offset, median
                             over rows: 1 = the predicted movement points where the command points, 0 = unrelated
                             (Sep 30 refit: -0.23 left, -0.02 right).
  command_miss_cm            distance of the index command decoded from the row's own code from the recorded command,
                             median; command_miss_if_zero_cm: the same for "command zero" (the finger's own place).
  decoded_target_spread_cm / predicted_move_spread_cm   mean distance of each trial code's decoded finger target /
                             predicted finger movement from the mean code's, median over rows.
Units are the hierarchy's own row convention (world, or relative to the wrist under digit_frame "wrist").
CODE SAMPLING (--code-sampling, Oct 1, 2026): as latent_body_part_readings ("auto": encoded codes, the encoder's own
codes of other held-out rows inside its 95th-percentile Mahalanobis range, for a whole-body hierarchy; the earlier box
for the pilot). Every result records the mode (code_sampling) and each part's trial-code distances (code_range).
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


@torch.no_grad()
def index_readings(learner, examples, *, every=30, codes=16, seed=0, parts=("index_L", "index_R"), code_sampling="auto"):
    import zlib
    from lem.latent_body.code_range import CodeRange, resolve_sampling
    from lem.latent_body.trajectory import row_options, row_scales
    from mimo_bridge.tools.latent_body_intervention import decode_with
    from mimo_bridge.tools.latent_body_part_readings import trial_codes
    if len(examples) < 4:
        raise ValueError("Readings need at least 4 held-out rows")
    tree, cfg, model = learner.tree, learner.config, learner.model
    names, rows = tree.handle_names, row_options(cfg)
    mode = resolve_sampling(code_sampling, cfg)
    batch = learner.batch(examples)
    z = model.encode(batch)
    code_range = CodeRange(z) if mode != "box" or len(examples) >= 50 else None
    roots = {r: z[r].mean(0, keepdim=True).expand(codes, -1) for r in tree.roots}
    rng = np.random.default_rng(seed)
    out, ranges = {}, {}
    for part in parts:
        generator = torch.Generator().manual_seed(int(seed) * 1_000_003 + zlib.crc32(part.encode()))
        handle = names.index(part)
        wrist = names.index("wrist_" + part[-1])
        step = torch.stack([ex.end.bones[wrist, 4:7] - ex.start.bones[wrist, 4:7] for ex in examples])
        centre, spread = z[part].mean(0, keepdim=True), z[part].std(0, keepdim=True)
        sa = row_scales(names, [handle], cfg["scales"], "action", rows["digit_scales"])[0, :3]
        sc = row_scales(names, [handle], cfg["scales"], "change", rows["digit_scales"])[0, :3]
        agree, targets, moves, miss, miss0, drawn = [], [], [], [], [], []
        for r in range(0, len(examples), every):
            state = {k: v[r][None].expand(codes, -1) for k, v in batch.state.items()}
            recent = {k: v[r][None].expand(codes, -1) for k, v in batch.history.items()}
            code = trial_codes(mode, codes, centre, spread, rng, generator, code_range, part, r, model.latent_dim)
            drawn.append(code[1:])
            p = decode_with(model, state, recent, roots, {part: code})[part]
            pc = p.change.reshape(codes, -1)[:, :3] * sc
            tg = p.action.reshape(codes, -1)[:, :3] * sa
            x, y = pc - pc.mean(0), tg - tg.mean(0)
            den = float(x.norm() * y.norm())
            agree.append(float((x * y).sum()) / den if den > 0 else 0.0)
            targets.append(float((tg[1:] - tg[0]).norm(dim=-1).mean()))
            moves.append(float((pc[1:] - pc[0]).norm(dim=-1).mean()))
            one = {k: v[:1] for k, v in state.items()}
            own = model.parts[part].predict(one[part], recent[part][:1], model.context(one)[part], z[part][r][None])
            recorded = batch.action[part][r].reshape(-1)[:3] * sa
            miss.append(float((own.action.reshape(-1)[:3] * sa - recorded).norm()))
            miss0.append(float(recorded.norm()))
        out[part] = dict(code_reads_wrist_step_r2=_r2(z[part], step),
                         direction_agreement=float(np.median(agree)),
                         command_miss_cm=float(np.median(miss)), command_miss_if_zero_cm=float(np.median(miss0)),
                         decoded_target_spread_cm=float(np.median(targets)),
                         predicted_move_spread_cm=float(np.median(moves)), rows_sampled=len(agree))
        ranges[part] = None if code_range is None else code_range.report(part, torch.cat(drawn))
    return dict(digit_frame=rows["digit_frame"], digit_scales=rows["digit_scales"] is not None,
                examples=len(examples), codes=codes, seed=seed, code_sampling=mode,
                code_sampling_requested=code_sampling, code_range=ranges, parts=out)


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
    learner = load_learner(a.hierarchy)
    learner.model.eval()
    examples = list(itertools.islice(read_examples(a.holdout), a.rows))
    result = dict(index_readings(learner, examples, every=a.every, codes=a.codes, seed=a.seed,
                                 code_sampling=a.code_sampling),
                  hierarchy=str(Path(a.hierarchy).resolve()), holdout=str(Path(a.holdout).resolve()),
                  updates=learner.updates, authority="none")
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result["parts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
