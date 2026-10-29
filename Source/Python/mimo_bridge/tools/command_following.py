# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Does the body do what it is told, and at which decoder gain? Physics only, no model.

Why this exists (2026-09-18, gain-of-function tree X6/X13): in H83 the direction the body
turned matched the direction it was ordered at 0.067 (+1 = exactly, 0 = sideways) and
2.3% of the asked rotation was achieved per 10 ms tick. Every ability test scores an
executed movement, so until an order produces a measurable physical difference nothing
can be measured. The decoder gain kp was 2.0 in H83 (tuned so one shoulder command was
met; the CLI default is 8.0, which saturated 55.8% of demands before the rescale fix).
Nothing measured command following across gains and strengths on the body alone. This
tool does, on one in-process body, and writes the result to scale_anchors.json as the
motor anchor the page said was missing.

Method. For each (kp, strength): N trials. Each trial starts from the settled rest pose,
picks ONE handle (15 cycle: head, the six arm segments, the six leg segments, chest,
lower trunk), turns its current orientation by a fixed angle about a random axis, sets
the position target to the current position (hands: plus a 5 cm step in a random
direction), strength as given, and holds the order for one latch (5 model steps = 25
ticks of 10 ms). Scored on the world's own NET gauge (env.py cmd_response_slow): the
cosine between the rotation asked at the start of the hold and the rotation that had
happened 250 ms later (direction match), and the ratio of their sizes (how much of it
happened), plus the fraction of decoded controls sitting on the rail. A trial whose reading
scored no part (n = 0) has no direction match: it is counted per part as unscored and left
out of every mean, never averaged in as 0. The position match is read only on the trials
that order a position step (the wrists); env.py reports 0 for a trial without one. Two
readings a person can picture are taken on the same tick: how many of 100 ordered parts
ended the hold nearer the ordered orientation than they began it, and the share of the
ordered turn's degrees closed (both added 2026-09-23).

Run:  PYTHONPATH=Source/Python .venv_mimo/Scripts/python.exe -m mimo_bridge.tools.command_following
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time

import numpy as np

# One handle per trial. Turning a whole arm's three handles by the same world rotation
# is a conflicting ask (each segment would have to turn about its own origin), and it
# measured -0.02 / -0.09 direction match on the arms while single handles read 0.7-0.9;
# the gauge here is "does ONE order to ONE handle get followed", which is what the
# decoder was tuned on (a 28.6-degree shoulder command).
HANDLE_GROUPS = {name: (name,) for name in (
    "head", "upperarm_L", "lowerarm_L", "wrist_L", "upperarm_R", "lowerarm_R", "wrist_R",
    "thigh_L", "calf_L", "foot_L", "thigh_R", "calf_R", "foot_R", "chest", "lower_trunk")}


def _quat_mul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw,
                     aw*bw - ax*bx - ay*by - az*bz])


def _turn(q_xyzw, axis, angle):
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    s = math.sin(angle / 2.0)
    r = np.array([axis[0]*s, axis[1]*s, axis[2]*s, math.cos(angle / 2.0)])
    q = _quat_mul(r, q_xyzw)
    return q / max(np.linalg.norm(q), 1e-9)


def _deg_to(q_wxyz, target_xyzw):
    """Degrees between a body's orientation (MuJoCo wxyz) and an ordered one (xyzw)."""
    q = np.asarray(q_wxyz, dtype=np.float64); t = np.asarray(target_xyzw, dtype=np.float64)
    t = np.array([t[3], t[0], t[1], t[2]])
    d = abs(float(np.dot(q, t))) / max(float(np.linalg.norm(q) * np.linalg.norm(t)), 1e-12)
    return math.degrees(2.0 * math.acos(min(1.0, d)))


def rest_order(env, n_handles):
    """An order that asks for exactly the current pose at strength 0 (nothing pushes):
    position in world cm and orientation as xyzw, read from MuJoCo for each handle body."""
    order = np.zeros((n_handles, 9))
    for i, body in enumerate(env.proprio.body_ids[:n_handles]):
        order[i, 0:3] = np.asarray(env.data.xpos[body], dtype=np.float64) * 100.0
        q = np.asarray(env.data.xquat[body], dtype=np.float64)        # MuJoCo wxyz
        order[i, 3:7] = [q[1], q[2], q[3], q[0]]
    order[:, 7] = 0.5
    return order


def run(kp, strength, trials, angle_deg, rng, scene, age, hold_steps, quiet=False):
    from mimo_bridge.env import BridgeEnv
    from mimo_bridge.body_layout import names_for
    names = list(names_for("trunk_v1"))
    run_dir = os.path.join("Saved", "mimo_bridge", "scenes_test")
    env = BridgeEnv(scene=scene, run_dir=run_dir, vision=False, tag=f"_cmdfollow_{kp:g}_{strength:g}",
                    age=age, contact_profile="mechanical_v4", caregiver_seconds=0, kp=kp,
                    body_layout="trunk_v1", cmd_horizon=hold_steps)   # the trainer passes action_repeat x motor_hold = 25
    cos_all, ratio_all, pos_all, rail_all = [], [], [], []
    closer_all, gap_before, gap_after = [], [], []
    per_group = {g: [] for g in HANDLE_GROUPS}
    unscored = {g: 0 for g in HANDLE_GROUPS}      # trials the gauge scored no part for
    try:
        env.reset()
        n = len(names)
        for _ in range(10):                                   # settle
            env.step(rest_order(env, n), None, None, render=False)
        for t in range(trials):
            group = list(HANDLE_GROUPS)[t % len(HANDLE_GROUPS)]
            order = rest_order(env, n)
            axis = rng.normal(size=3)
            for h in HANDLE_GROUPS[group]:
                i = names.index(h)
                order[i, 3:7] = _turn(order[i, 3:7], axis, math.radians(angle_deg))
                order[i, 8] = strength
                if h.startswith("wrist"):
                    d = rng.normal(size=3); d = d / np.linalg.norm(d)
                    order[i, 0:3] = order[i, 0:3] + 5.0 * d          # 5 cm step
            cos_t, ratio_t, pos_t, rail_t = [], [], [], []
            unscored_t = 0
            q_start = {h: np.array(env.data.xquat[env.proprio.body_ids[names.index(h)]], dtype=np.float64)
                       for h in HANDLE_GROUPS[group]}
            # the gauge is the env's own NET 250 ms comparison (cmd_response_slow): the
            # rotation asked at tick 0 against the rotation that had happened by tick 25.
            # The per-tick gauge averaged over the hold penalises a limb that overshoots and
            # comes back (measured: direction fell from 0.58 to 0.45 as strength rose while
            # the achieved share doubled), which is not "did it go where it was told".
            # Read on the hold's last tick, where the order issued on tick 0 comes of age
            # (until 2026-09-23 the order was held 27 ticks and read on ticks 26 and 27).
            for k in range(hold_steps):
                out = env.step(order, None, None, render=False)
                cr = out.info.get("cmd_response_slow")
                if cr and cr.get("cos") is not None and k == hold_steps - 1:
                    if not cr.get("n", 0):
                        # the gauge scored no part: env._command_response returns cos 0.0 with
                        # n 0, which is no reading, not a direction match of 0. Until 2026-09-23
                        # it was kept as 0 - every chest trial, while the gauge took the chest
                        # (trunk_v1's last row) for the eyes' row and never scored it
                        unscored_t += 1
                    else:
                        cos_t.append(float(cr["cos"])); ratio_t.append(float(cr.get("ratio", 0.0)))
                        # env's pos_cos is 0.0 when no part was ordered to MOVE (only the wrists
                        # get a 5 cm step here): no reading, not a match of 0. Until 2026-09-23 the
                        # 26 of every 30 trials without a step went into position_match_mean as 0
                        if cr.get("pos_cos") is not None and any(h.startswith("wrist") for h in HANDLE_GROUPS[group]):
                            pos_t.append(float(cr["pos_cos"]))
                    # parts ending closer to the ordered orientation than they started, and the
                    # degrees left to go (a count and a real unit; DESIGN.md section 11)
                    for h in HANDLE_GROUPS[group]:
                        i = names.index(h)
                        before = _deg_to(q_start[h], order[i, 3:7])
                        after = _deg_to(env.data.xquat[env.proprio.body_ids[i]], order[i, 3:7])
                        closer_all.append(after < before); gap_before.append(before); gap_after.append(after)
                ctrl = out.info.get("decoded_ctrl")
                if ctrl is not None:
                    c = np.asarray(ctrl, dtype=np.float64)
                    rail_t.append(float(np.mean(np.abs(c) >= 0.999)))
            if cos_t:
                cos_all.append(float(np.mean(cos_t))); ratio_all.append(float(np.mean(ratio_t)))
                per_group[group].append(float(np.mean(cos_t)))
                if pos_t: pos_all.append(float(np.mean(pos_t)))
                if rail_t: rail_all.append(float(np.mean(rail_t)))
            elif unscored_t:
                unscored[group] += 1
            for _ in range(15):                               # return to rest
                env.step(rest_order(env, n), None, None, render=False)
    finally:
        env.close()
    out = dict(kp=kp, strength=strength, trials=len(cos_all),
               direction_match_mean=float(np.mean(cos_all)) if cos_all else None,
               direction_match_p25=float(np.percentile(cos_all, 25)) if cos_all else None,
               direction_match_ci95=([float(np.mean(cos_all) - 1.96 * np.std(cos_all, ddof=1) / math.sqrt(len(cos_all))),
                                      float(np.mean(cos_all) + 1.96 * np.std(cos_all, ddof=1) / math.sqrt(len(cos_all)))]
                                     if len(cos_all) > 1 else None),
               closer_out_of_100=(100.0 * float(np.mean(closer_all))) if closer_all else None,
               closer_parts=len(closer_all),
               gap_closed_pct=(100.0 * (1.0 - float(np.sum(gap_after)) / float(np.sum(gap_before))))
                              if gap_before and np.sum(gap_before) > 0 else None,
               rotation_achieved_share_mean=float(np.mean(ratio_all)) if ratio_all else None,
               position_match_mean=float(np.mean(pos_all)) if pos_all else None,
               rail_fraction_mean=float(np.mean(rail_all)) if rail_all else None,
               per_group={g: (float(np.mean(v)) if v else None) for g, v in per_group.items()},
               trials_unscored=int(sum(unscored.values())), per_group_unscored=dict(unscored))
    if not quiet and out["trials_unscored"]:
        print(f"kp {kp:4.1f} strength {strength:.2f}: {out['trials_unscored']} of "
              f"{out['trials_unscored'] + out['trials']} trials had no part the gauge could score and are "
              "left out of every mean: " + ", ".join(f"{g} {c}" for g, c in unscored.items() if c), flush=True)
    if not quiet and out["direction_match_mean"] is not None:
        pm = out['position_match_mean']
        print(f"kp {kp:4.1f} strength {strength:.2f}: direction {out['direction_match_mean']:.3f} "
              f"(p25 {out['direction_match_p25']:.3f}) achieved {out['rotation_achieved_share_mean']:.3f} "
              f"position {'-' if pm is None else round(pm, 3)} rail {out['rail_fraction_mean']:.3f} "
              f"closer {out['closer_out_of_100']:.0f}/100 gap closed {out['gap_closed_pct']:.0f}%  groups " +
              " ".join(f"{g}={v:.2f}" for g, v in out['per_group'].items() if v is not None), flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kps", default="2,3,4,6,8")
    ap.add_argument("--strengths", default="0.3,0.5,0.8,1.0")
    ap.add_argument("--trials", type=int, default=30, help="trials per cell; 15 handles cycle")
    ap.add_argument("--angle-deg", type=float, default=20.0)
    ap.add_argument("--hold-steps", type=int, default=25, help="ticks the order is held: 25 = one 250 ms latch")
    ap.add_argument("--scene", default="crib")
    ap.add_argument("--age", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-anchor", action="store_true", help="store the table in Saved/mimo_bridge/scale_anchors.json")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    kps = [float(x) for x in args.kps.split(",")]
    strengths = [float(x) for x in args.strengths.split(",")]
    t0 = time.perf_counter()
    table = []
    for kp in kps:
        for s in strengths:
            table.append(run(kp, s, args.trials, args.angle_deg, rng, args.scene, args.age, args.hold_steps))
    # the pick: highest direction match with under 20% of controls on the rail; ties to the
    # higher achieved share. Stated as a rule before the numbers were read.
    ok = [r for r in table if r["direction_match_mean"] is not None and (r["rail_fraction_mean"] or 0) < 0.20]
    best = max(ok or table, key=lambda r: (round(r["direction_match_mean"], 2), r["rotation_achieved_share_mean"]))
    print(f"\nbest cell under the 20% rail rule: kp {best['kp']} strength {best['strength']} "
          f"direction {best['direction_match_mean']:.3f} achieved {best['rotation_achieved_share_mean']:.3f} "
          f"({time.perf_counter()-t0:.0f} s)")
    result = dict(measured=datetime.datetime.now().astimezone().isoformat(), scene=args.scene, age_months=args.age,
                  angle_deg=args.angle_deg, hold_ticks=args.hold_steps, trials_per_cell=args.trials,
                  gauge="env.py cmd_response_slow, the net 250 ms comparison: cos = direction match of the rotation asked "
                        "at the start of the hold against the rotation that had happened 25 ticks later (+1 exact, 0 sideways, "
                        "-1 against); achieved share = their size ratio; rail = share of decoded controls at |1| during the hold",
                  selection_rule="highest direction match with under 20% of controls on the rail, ties to the higher achieved share",
                  unscored_rule="a trial whose gauge reading scored no part (n = 0) has no direction match: it is counted in "
                                "trials_unscored and per_group_unscored and left out of every mean; until 2026-09-23 it was "
                                "counted as a direction match of 0",
                  closer_rule="closer_out_of_100: of 100 ordered parts, how many ended the hold nearer the ordered "
                              "orientation than they began it (0 = none, 100 = every one); gap_closed_pct: the share of "
                              "the ordered turn's degrees closed, summed over the cell's parts (0% = never moved, 100% = "
                              "arrived; negative = ended farther away); position_match_mean is read only on trials that "
                              "ordered a position step (the wrists); direction_match_ci95 = mean +- 1.96 standard errors",
                  table=table, best=dict(kp=best["kp"], strength=best["strength"]),
                  reference_H83=dict(cmd_response_cos_mean=0.067, decoder_kp=2.0, strength_mean=0.474,
                                     source="H83 replay_training_stats.jsonl, 137 windows"))
    if args.write_anchor:
        path = os.path.join("Saved", "mimo_bridge", "scale_anchors.json")
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        anchors = json.loads(text)
        prev = anchors.get("command_following")
        if prev is not None:       # the table it replaces stays on record, newest first
            result["superseded"] = [{k: v for k, v in prev.items() if k != "superseded"}] + list(prev.get("superseded", []))
        # only this entry's text is rewritten: other anchors are laid out by hand (inline lists),
        # and a whole-file re-dump would re-lay them all (it did until 2026-09-23)
        text = _replace_entry(text, "command_following", result)
        if set(json.loads(text)) != set(anchors) | {"command_following"}:
            raise RuntimeError("scale_anchors.json: rewriting the command_following entry changed the set of anchors")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
        print(f"written to {path} under command_following")
    return result


def _replace_entry(text, key, value):
    """`text` is a JSON object; return it with only the top-level entry `key` rewritten (two-space
    indents, nested one level) and every other byte kept. An absent entry is added at the end."""
    dec = json.JSONDecoder()

    def skip(i):
        while text[i] in " \t\r\n":
            i += 1
        return i

    block = json.dumps(value, indent=2).replace("\n", "\n  ")
    i, last_end = skip(text.index("{") + 1), None
    while text[i] != "}":
        k, i = dec.raw_decode(text, i)
        i = skip(i)
        if text[i] != ":":
            raise ValueError(f"expected ':' after {k!r} at character {i}")
        i = skip(i + 1)
        _, end = dec.raw_decode(text, i)
        if k == key:
            return text[:i] + block + text[end:]
        last_end = end
        i = skip(end)
        if text[i] == ",":
            i = skip(i + 1)
    entry = json.dumps(key) + ": " + block
    if last_end is None:
        return text[:i] + "\n  " + entry + "\n" + text[i:]
    return text[:last_end] + ",\n  " + entry + text[last_end:]


if __name__ == "__main__":
    main()
