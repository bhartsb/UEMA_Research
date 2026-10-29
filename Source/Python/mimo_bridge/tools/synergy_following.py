# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Does the body follow an order the motor synergy wrote? Physics only, beside the two anchors.

Why this exists (2026-09-22, DESIGN.md D11). tools/whole_body_following.py measured on 2026-09-19,
on physics alone, that a whole-body order which is a posture the body really reached is followed at 0.625
(-1 against, 0 unrelated, +1 exact) and one made of 25 independent 20-degree turns at 0.365. The
motor synergy (lem/model/motor_synergy.py) is meant to write orders of the first kind from a few
numbers z. This tool issues orders the synergy wrote from sampled z and scores them on the same
gauge, in the same session and body, next to the same two arms:

  CONSISTENT   every handle asked for its pose in a posture the body reached a moment earlier
  INDEPENDENT  every handle turned by angle_deg about its own random axis
  SYNERGY      z drawn from N(0, I); every body handle asked for S'(z, current pose) applied to
               its current pose (turns on all 25, moves on the 14 position handles), then the
               turns scaled so their mean equals angle_deg (the independent arm's size), again
               scaled to the CONSISTENT arm's own mean commanded turn so far in this session
               (`synergy_matched`, the two compared at one size), and at the synergy's own size
               (unscaled, `synergy_native`)

Method (whole_body_following's): one in-process body in the crib, vision off, mechanical_v4, the
trainer's 25-tick latch; settle at rest; each trial issues ONE order to all handles at strength
0.5 (eyes at 0) and holds it 25 ticks; scored on the environment's own net 250 ms gauge
(cmd_response_slow.cos) plus the cosine weighted by each handle's commanded turn; 60 rest ticks
between trials. The synergy is read from a checkpoint folder (model.safetensors, keys
motor_synergy.*, with its runtime_state.pt centre), a .pt state dict, or "fresh" (a newly built,
untrained synergy: the before-learning reference).

MEASURED 2026-09-22/23 (b5-synergy, 20 trials per arm, kp 6, strength 0.5; -1 against, 0 unrelated,
+1 exact). The anchors first: the README's 0.625 / 0.365 (2026-09-19) predate the finger-armature
repair; in these sessions the same arms read 0.78-0.86 (a reachable posture, mean commanded turn
8.2-8.5 degrees) and 0.54-0.56 (independent 20-degree turns). Orders written by:
  a FRESH synergy (untrained)                            0.42 / 0.45 matched / 0.41 own size
  a synergy FITTED on the live replay diary's 9,592      0.55 at 20.7 deg / 0.53 matched (9.1 deg) /
    orders (k 32, three passes of the lesson)            0.58 at its own size (11.8 deg)
  the RECORDED live changes themselves (the fitted       0.55 at 20.6 deg / 0.53 matched / 0.54
    synergy's own targets), as a control                 at their own size (14.4 deg)
So the fitted synergy writes orders the body follows as well as the changes it learned from, and
those recorded changes - taken from other poses and moments - are followed like independent turns
from this rest pose, not like a posture reached from it. Raw results: the b5 scratch folder
(synfollow_*.json; docs/gain_of_function_v3/agents/b5-synergy.md).

Run:  PYTHONPATH=Source/Python .venv_mimo/Scripts/python.exe -m mimo_bridge.tools.synergy_following \
          --synergy <ckpt folder, .pt or fresh> [--trials 100] [--angle-deg 20]
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from mimo_bridge.tools.command_following import _turn, rest_order
from mimo_bridge.tools.whole_body_following import DRIVERS, _weighted_cos


def load_synergy(path, names, gaze_idx, k=None):
    """Build lem/model/motor_synergy.MotorSynergy and load its weights (and pose centre)."""
    import torch
    from lem.model.motor_synergy import MotorSynergy
    from mimo_bridge.action_decoder import POSITION_HANDLES
    if str(path) == "fresh":
        from lem.model import motor_synergy as MS
        syn = MotorSynergy(names, gaze_idx, sorted(POSITION_HANDLES), int(k or MS.SYNERGY_DIM))
        syn.eval()
        return syn
    if os.path.isdir(path):
        from safetensors.torch import load_file
        sd = {k2[len("motor_synergy."):]: v for k2, v in load_file(os.path.join(path, "model.safetensors")).items()
              if k2.startswith("motor_synergy.")}
        rt = os.path.join(path, "runtime_state.pt")
        centre = (torch.load(rt, weights_only=False, map_location="cpu").get("motor_synergy")
                  if os.path.isfile(rt) else None)
    else:
        blob = torch.load(path, weights_only=False, map_location="cpu")
        sd, centre = blob["state_dict"], blob.get("centre")
    if not sd:
        raise ValueError(f"no motor synergy weights in {path}")
    k = int(k or sd["E3.weight"].shape[0])
    syn = MotorSynergy(names, gaze_idx, sorted(POSITION_HANDLES), k)
    syn.load_state_dict(sd, strict=True)
    if centre:
        syn.centre_mean.copy_(centre["centre_mean"])
        syn.centre_weight = float(centre["centre_weight"])
    syn.eval()
    return syn


def synergy_order(env, syn, names, rng, angle_deg=None):
    """One order the synergy wrote for the body as it is now: z ~ N(0, I), S'(z, pose) applied to
    every body handle's current pose. With angle_deg the turns are scaled so their mean over the
    25 body handles is angle_deg (moves scaled alike). Returns (order [n, 9] execution form, mean
    turn in degrees before any scaling)."""
    import torch
    n = len(names)
    order = rest_order(env, n)
    order[:, 0:3] = order[:, 0:3] / 200.0          # the raw form: positions in units of 200 cm
    ach = torch.zeros(1, n, 10)
    ach[0, :, 0:7] = torch.as_tensor(order[:, 0:7], dtype=torch.float32)
    z = torch.as_tensor(rng.normal(size=(1, syn.k)), dtype=torch.float32)
    with torch.no_grad():
        rot, pos = syn.delta(z, ach)
        native = float(rot.norm(dim=-1).mean()) * 180.0 / math.pi
        if angle_deg is not None and native > 1e-9:
            f = angle_deg / native
            rot, pos = rot * f, pos * f
        raw = syn.order_from_delta(rot, pos, ach, torch.as_tensor(order[None], dtype=torch.float32))
    out = raw[0].numpy().astype(np.float64)
    out[:, 0:3] = out[:, 0:3] * 200.0
    return out, native


def run(synergy_path, kp=6.0, strength=0.5, trials=60, angle_deg=20.0, seed=0, hold=25, quiet=False):
    from mimo_bridge.env import BridgeEnv
    from mimo_bridge.body_layout import names_for
    import torch
    names = list(names_for("trunk_v1"))
    gaze = names.index("gaze")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        syn = load_synergy(synergy_path, names, gaze)
    rng = np.random.default_rng(seed)
    env = BridgeEnv(scene="crib", run_dir=os.path.join("Saved", "mimo_bridge", "scenes_test"), vision=False,
                    tag=f"_synergyfollow_{os.getpid()}_{kp:g}", seed=seed, age=24, contact_profile="mechanical_v4",
                    caregiver_seconds=0, kp=kp, body_layout="trunk_v1", cmd_horizon=hold)
    kinds = ("consistent", "independent", "synergy", "synergy_matched", "synergy_native")
    res = {k: [] for k in kinds}
    try:
        env.reset()
        n = len(names)
        for _ in range(20):
            env.step(rest_order(env, n), None, None, render=False)
        for t in range(trials):
            kind = kinds[t % len(kinds)]
            native = None
            if kind == "consistent":
                drive = rest_order(env, n)
                for group in DRIVERS:
                    h = group[int(rng.integers(len(group)))]
                    i = names.index(h)
                    drive[i, 3:7] = _turn(drive[i, 3:7], rng.normal(size=3), math.radians(angle_deg))
                    drive[i, 8] = strength
                for _ in range(40):
                    env.step(drive, None, None, render=False)
                target = rest_order(env, n)
                for _ in range(60):
                    env.step(rest_order(env, n), None, None, render=False)
                order = target.copy()
            elif kind == "independent":
                order = rest_order(env, n)
                for i in range(n):
                    if i != gaze:
                        order[i, 3:7] = _turn(order[i, 3:7], rng.normal(size=3), math.radians(angle_deg))
            else:
                size = {"synergy": angle_deg, "synergy_native": None,
                        "synergy_matched": (float(np.mean([r["mean_turn_deg"] for r in res["consistent"]]))
                                            if res["consistent"] else angle_deg)}[kind]
                order, native = synergy_order(env, syn, names, rng, size)
            order[:, 8] = strength
            order[gaze, 8] = 0.0
            xq0 = np.array([env.data.xquat[b] for b in env.proprio.body_ids[:n]])
            cos_t = []
            # read on the hold's last tick, the order's whole tenure (until 2026-09-23: ticks 26-27)
            for k in range(hold):
                out = env.step(order, None, None, render=False)
                cr = out.info.get("cmd_response_slow")
                if cr and cr.get("cos") is not None and k == hold - 1:
                    cos_t.append((float(cr["cos"]), float(cr.get("ratio", 0.0)), int(cr.get("n", 0))))
            wcos, mag = _weighted_cos(env, order, xq0, names)
            import torch
            from lem.model.motor_synergy import following_counts
            before = np.zeros((1, n, 10))
            after = np.zeros_like(before)
            before[0, :, 3:7] = xq0[:, [1, 2, 3, 0]]
            xq1 = np.array([env.data.xquat[b] for b in env.proprio.body_ids[:n]])
            after[0, :, 3:7] = xq1[:, [1, 2, 3, 0]]
            counts = following_counts(torch.from_numpy(order[None]), torch.from_numpy(before),
                                      torch.from_numpy(after), [i for i in range(n) if i != gaze])
            res[kind].append(dict(cos=float(np.mean([c[0] for c in cos_t])) if cos_t else None,
                                      ratio=float(np.mean([c[1] for c in cos_t])) if cos_t else None,
                                      handles=int(np.mean([c[2] for c in cos_t])) if cos_t else 0, weighted_cos=wcos,
                                      mean_turn_deg=math.degrees(mag), native_turn_deg=native,
                                      part_orders=int(counts['closer'].numel()),
                                      closer=int(counts['closer'].sum()),
                                      chance_count=float(counts['chance'].sum()),
                                      moved_deg=counts['moved_deg'].tolist()))
            for _ in range(60):
                env.step(rest_order(env, n), None, None, render=False)
    finally:
        env.close()
    out = dict(kp=kp, strength=strength, angle_deg=angle_deg, hold_ticks=hold, synergy=str(synergy_path), seed=seed,
               k=int(syn.k), metric_version='following_counts_v1',
               chance_control='Uniform random rotation axis at the same ordered angle, evaluated '
                              'against each observed turn; a geometric control, not a physics replay')
    for kind, rows in res.items():
        if not rows:
            continue
        wc = [r["weighted_cos"] for r in rows if r["weighted_cos"] is not None]
        directions = [r['cos'] for r in rows if r['cos'] is not None]
        ratios = [r['ratio'] for r in rows if r['ratio'] is not None]
        out[kind] = dict(trials=len(rows),
                         part_orders=sum(r['part_orders'] for r in rows),
                         closer=sum(r['closer'] for r in rows),
                         chance_count=sum(r['chance_count'] for r in rows),
                         direction_match_mean=float(np.mean(directions)) if directions else None,
                         direction_match_p25=float(np.percentile(directions, 25)) if directions else None,
                         direction_match_weighted_by_turn=float(np.mean(wc)) if wc else None,
                         rotation_achieved_share_mean=float(np.mean(ratios)) if ratios else None,
                         commanded_turn_deg_mean=float(np.mean([r["mean_turn_deg"] for r in rows])),
                         native_turn_deg_mean=(float(np.mean([r["native_turn_deg"] for r in rows]))
                                               if rows[0]["native_turn_deg"] is not None else None))
        o = out[kind]
        n_part = o['part_orders']
        o['closer_per_100'] = 100 * o['closer'] / n_part if n_part else None
        o['chance_per_100'] = 100 * o['chance_count'] / n_part if n_part else None
        o['above_chance_per_100'] = (o['closer_per_100'] - o['chance_per_100']) if n_part else None
        moved = [x for r in rows for x in r['moved_deg']]
        o['median_realized_deg'] = float(np.median(moved)) if moved else None
        # Whole orders are the sampling units: neighboring joints are not independent trials.
        if n_part and len(rows) >= 2:
            draw = np.random.default_rng(seed + 9127).integers(len(rows), size=(2000, len(rows)))
            totals = np.asarray([r['part_orders'] for r in rows])[draw].sum(1)
            gains = np.asarray([r['closer'] - r['chance_count'] for r in rows])[draw].sum(1)
            boot = 100 * gains[totals > 0] / totals[totals > 0]
            o['above_chance_per_100_ci95'] = np.quantile(boot, [.025, .975]).tolist()
        else:
            o['above_chance_per_100_ci95'] = None
        if not quiet:
            if n_part:
                print(f"{kind:>14}: {o['closer_per_100']:.1f} of 100 part-orders ended closer; "
                      f"{o['chance_per_100']:.1f} expected by random-axis control; "
                      f"median realized turn {o['median_realized_deg']:.1f} degrees "
                      f"({n_part} part-orders, {o['trials']} trials)", flush=True)
            else:
                print(f"{kind:>14}: no scorable part-orders ({o['trials']} trials)", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synergy", required=True, help="checkpoint folder with motor_synergy weights, or a .pt")
    ap.add_argument("--kp", type=float, default=6.0)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--angle-deg", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the result as JSON here")
    a = ap.parse_args(argv)
    out = run(a.synergy, a.kp, a.strength, a.trials, a.angle_deg, a.seed)
    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2)
    return out


if __name__ == "__main__":
    main()
