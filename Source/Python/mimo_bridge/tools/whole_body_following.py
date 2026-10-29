# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Does the body follow a WHOLE-BODY order, and does it matter whether the order is consistent?

Why this exists (2026-09-19). tools/command_following.py measured one order to one handle
on physics alone: direction match 0.716 at decoder gain 6 on the 250 ms gauge (a figure that
counted its chest trials as unrelated, because the gauge never scored the chest; re-measured
2026-09-23 on the repaired gauge: 75% of the way from unrelated to exact, 95% interval 65% to
85%, scale_anchors.json command_following). Live, the
same gauge reads 0.069 (H83) to 0.15 (H84): every handle is ordered at once. Two readings
of that gap lead to different work, so this tool separates them on physics alone, with no
model:

  CONSISTENT order   every handle is asked for the pose it had in a posture the body really
                     reached a moment earlier (so the 25 targets agree with one another);
  INDEPENDENT order  every handle is asked for its current orientation turned by the same
                     angle about its own random axis (so chained handles disagree).

If the consistent order is followed and the independent one is not, the decoder and the
gain are doing their job and the live reading says the orders the model emits are not
reachable postures - a learning problem. If neither is followed, execution is still broken.

Method. One in-process body in the crib, vision off, mechanical_v4, the trainer's 25-tick
latch. Each trial: settle at rest (posture A); drive three unchained handles (head, one
thigh, one upper arm) with single orders for 40 ticks to reach posture B and record every
handle's world pose there; return to rest for 60 ticks; then issue ONE order to all handles
and hold it 25 ticks. Scored on the env's own net 250 ms gauge (cmd_response_slow.cos, the
mean over handles with strength above 0 and a commanded turn above 1e-4 rad), plus the same
cosine weighted by the size of each handle's commanded turn, and the mean commanded turn.

Fixed 2026-09-23 (docs/gain_of_function_v3/agents/fix-following-gauge.md): the eyes' row is found
by name (row 23). Before, this tool and the env gauge took the LAST row as the eyes, which in
trunk_v1 is the chest: the independent order turned the eyes and never the chest, and both
gauges scored the eyes instead of the chest. The gauge is now read on the hold's last tick (the
order's 25-tick tenure), not on ticks 26 and 27. The 2026-09-19 anchor predates both fixes.

Run:  PYTHONPATH=Source/Python .venv_mimo/Scripts/python.exe -m mimo_bridge.tools.whole_body_following
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os

import numpy as np

from mimo_bridge.tools.command_following import _turn, rest_order

DRIVERS = (("head",), ("thigh_L", "thigh_R"), ("upperarm_L", "upperarm_R"))


def _weighted_cos(env, order, xq_before, names):
    """The gauge's per-handle cosines weighted by the commanded turn's size."""
    from mimo_bridge.action_decoder import quat_xyzw_to_mat, rotmat_log
    num = den = 0.0
    mags = []
    gaze = list(names).index("gaze")          # the eyes' row by name (the last row is the chest in trunk_v1)
    for i, b in enumerate(env.proprio.body_ids[:len(names)]):
        if i == gaze or order[i, 8] <= 0:
            continue
        q0 = xq_before[i]
        R0 = quat_xyzw_to_mat([q0[1], q0[2], q0[3], q0[0]])
        q1 = env.data.xquat[b]
        R1 = quat_xyzw_to_mat([q1[1], q1[2], q1[3], q1[0]])
        w_cmd = rotmat_log(quat_xyzw_to_mat(order[i, 3:7]) @ R0.T)
        w_got = rotmat_log(R1 @ R0.T)
        nc, ng = np.linalg.norm(w_cmd), np.linalg.norm(w_got)
        if nc < 1e-4:
            continue
        c = float(np.dot(w_cmd, w_got) / (nc * max(ng, 1e-9))) if ng > 1e-6 else 0.0
        num += nc * c
        den += nc
        mags.append(nc)
    return (num / den if den else None), (float(np.mean(mags)) if mags else 0.0)


def run(kp, strength, trials, angle_deg, seed, hold=25, quiet=False):
    from mimo_bridge.env import BridgeEnv
    from mimo_bridge.body_layout import names_for
    names = list(names_for("trunk_v1"))
    rng = np.random.default_rng(seed)
    env = BridgeEnv(scene="crib", run_dir=os.path.join("Saved", "mimo_bridge", "scenes_test"), vision=False,
                    tag=f"_wholebody_{kp:g}_{strength:g}", age=24, contact_profile="mechanical_v4",
                    caregiver_seconds=0, kp=kp, body_layout="trunk_v1", cmd_horizon=hold)
    res = {"consistent": [], "independent": []}
    try:
        env.reset()
        n = len(names)
        gaze = names.index("gaze")            # row 23; until 2026-09-23 n - 1, which is the chest
        for _ in range(20):
            env.step(rest_order(env, n), None, None, render=False)
        for t in range(trials):
            kind = "consistent" if t % 2 == 0 else "independent"
            if kind == "consistent":
                # reach posture B with three unchained single orders, record it, come back
                drive = rest_order(env, n)
                for group in DRIVERS:
                    h = group[int(rng.integers(len(group)))]
                    i = names.index(h)
                    drive[i, 3:7] = _turn(drive[i, 3:7], rng.normal(size=3), math.radians(angle_deg))
                    drive[i, 8] = strength
                for _ in range(40):
                    env.step(drive, None, None, render=False)
                target = rest_order(env, n)                       # every handle's pose in posture B
                for _ in range(60):
                    env.step(rest_order(env, n), None, None, render=False)
                order = target.copy()
            else:
                order = rest_order(env, n)
                for i in range(n):
                    if i != gaze:
                        order[i, 3:7] = _turn(order[i, 3:7], rng.normal(size=3), math.radians(angle_deg))
            order[:, 8] = strength
            order[gaze, 8] = 0.0
            xq0 = np.array([env.data.xquat[b] for b in env.proprio.body_ids[:n]])
            cos_t = []
            # the gauge for the order issued on tick 0 comes of age on tick `hold`, its whole
            # tenure (env.py cmd_response_slow); read there, and the weighted cosine from the
            # same pose. Until 2026-09-23 this read ticks 26 and 27 of 27.
            for k in range(hold):
                out = env.step(order, None, None, render=False)
                cr = out.info.get("cmd_response_slow")
                if cr and cr.get("cos") is not None and k == hold - 1:
                    cos_t.append((float(cr["cos"]), float(cr.get("ratio", 0.0)), int(cr.get("n", 0))))
            wcos, mag = _weighted_cos(env, order, xq0, names)
            if cos_t:
                res[kind].append(dict(cos=float(np.mean([c[0] for c in cos_t])), ratio=float(np.mean([c[1] for c in cos_t])),
                                      handles=int(np.mean([c[2] for c in cos_t])), weighted_cos=wcos, mean_turn_deg=math.degrees(mag)))
            for _ in range(60):
                env.step(rest_order(env, n), None, None, render=False)
    finally:
        env.close()
    out = dict(kp=kp, strength=strength, angle_deg=angle_deg, hold_ticks=hold)
    for kind, rows in res.items():
        if not rows:
            continue
        out[kind] = dict(trials=len(rows),
                         direction_match_mean=float(np.mean([r["cos"] for r in rows])),
                         direction_match_p25=float(np.percentile([r["cos"] for r in rows], 25)),
                         direction_match_weighted_by_turn=float(np.mean([r["weighted_cos"] for r in rows if r["weighted_cos"] is not None])),
                         rotation_achieved_share_mean=float(np.mean([r["ratio"] for r in rows])),
                         handles_scored_mean=float(np.mean([r["handles"] for r in rows])),
                         commanded_turn_deg_mean=float(np.mean([r["mean_turn_deg"] for r in rows])))
        if not quiet:
            o = out[kind]
            print(f"kp {kp:g} strength {strength:g} {kind:>11}: direction {o['direction_match_mean']:.3f} "
                  f"(p25 {o['direction_match_p25']:.3f}; weighted by turn size {o['direction_match_weighted_by_turn']:.3f}) "
                  f"achieved {o['rotation_achieved_share_mean']:.3f}  handles scored {o['handles_scored_mean']:.1f}  "
                  f"mean commanded turn {o['commanded_turn_deg_mean']:.1f} deg  ({o['trials']} trials)", flush=True)
    return out


def _blend(new, prev, a):
    """lem_model.LemLite.smooth_blend in numpy: the executed order is a x new + (1 - a) x
    previous, quaternions hemisphere-aligned and renormalised (the live run uses a = 0.35)."""
    if prev is None or a >= 1.0:
        return new
    out = new.copy()
    out[:, 0:3] = a * new[:, 0:3] + (1 - a) * prev[:, 0:3]
    pq = prev[:, 3:7].copy()
    flip = (new[:, 3:7] * pq).sum(-1) < 0
    pq[flip] = -pq[flip]
    q = a * new[:, 3:7] + (1 - a) * pq
    out[:, 3:7] = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-6)
    out[:, 8] = a * new[:, 8] + (1 - a) * prev[:, 8]
    return out


def run_sequence(kp, strength, latches, angle_deg, seed, blend, hold=25, quiet=False):
    """Orders arrive BACK TO BACK, as they do live: a new whole-body order every 25 ticks with
    no return to rest, the body still moving from the last one, and the executed order
    blended with the previous executed order. Each new order asks every handle for its
    CURRENT orientation turned by angle_deg about its own random axis."""
    from mimo_bridge.env import BridgeEnv
    from mimo_bridge.body_layout import names_for
    names = list(names_for("trunk_v1"))
    rng = np.random.default_rng(seed)
    env = BridgeEnv(scene="crib", run_dir=os.path.join("Saved", "mimo_bridge", "scenes_test"), vision=False,
                    tag=f"_wholebodyseq_{kp:g}_{blend:g}", age=24, contact_profile="mechanical_v4",
                    caregiver_seconds=0, kp=kp, body_layout="trunk_v1", cmd_horizon=hold)
    rows, prev = [], None
    try:
        env.reset()
        n = len(names)
        gaze = names.index("gaze")            # row 23; until 2026-09-23 n - 1, which is the chest
        for _ in range(20):
            env.step(rest_order(env, n), None, None, render=False)
        for t in range(latches):
            new = rest_order(env, n)
            for i in range(n):
                if i != gaze:
                    new[i, 3:7] = _turn(new[i, 3:7], rng.normal(size=3), math.radians(angle_deg))
            new[:, 8] = strength
            new[gaze, 8] = 0.0
            order = _blend(new, prev, blend)
            prev = order
            xq0 = np.array([env.data.xquat[b] for b in env.proprio.body_ids[:n]])
            cos_t = []
            for k in range(hold):
                out = env.step(order, None, None, render=False)
            # the gauge for the order issued at the START of this latch comes of age on the
            # latch's last tick, its whole tenure (until 2026-09-23: on one extra tick added here)
            cr = out.info.get("cmd_response_slow")
            wcos, mag = _weighted_cos(env, order, xq0, names)
            if cr and cr.get("cos") is not None and t >= 2:            # skip the first two latches (from rest)
                rows.append(dict(cos=float(cr["cos"]), ratio=float(cr.get("ratio", 0.0)), weighted_cos=wcos,
                                 mean_turn_deg=math.degrees(mag)))
    finally:
        env.close()
    out = dict(kp=kp, strength=strength, angle_deg=angle_deg, blend=blend, latches=len(rows),
               direction_match_mean=float(np.mean([r["cos"] for r in rows])),
               direction_match_p25=float(np.percentile([r["cos"] for r in rows], 25)),
               direction_match_weighted_by_turn=float(np.mean([r["weighted_cos"] for r in rows if r["weighted_cos"] is not None])),
               rotation_achieved_share_mean=float(np.mean([r["ratio"] for r in rows])),
               executed_turn_deg_mean=float(np.mean([r["mean_turn_deg"] for r in rows])))
    if not quiet:
        print(f"kp {kp:g} strength {strength:g} back-to-back, blend {blend:g}: direction {out['direction_match_mean']:.3f} "
              f"(p25 {out['direction_match_p25']:.3f}; weighted by turn size {out['direction_match_weighted_by_turn']:.3f}) "
              f"achieved {out['rotation_achieved_share_mean']:.3f}  mean executed turn {out['executed_turn_deg_mean']:.1f} deg  "
              f"({out['latches']} latches)", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kp", type=float, default=6.0)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--angle-deg", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-anchor", action="store_true")
    ap.add_argument("--sequence", type=int, default=0, help="also run this many back-to-back latches, unblended and at the live blend 0.35")
    a = ap.parse_args(argv)
    out = run(a.kp, a.strength, a.trials, a.angle_deg, a.seed)
    if a.sequence:
        out["back_to_back"] = [run_sequence(a.kp, a.strength, a.sequence, a.angle_deg, a.seed, blend) for blend in (1.0, 0.35)]
    if a.write_anchor:
        path = os.path.join("Saved", "mimo_bridge", "scale_anchors.json")
        anchors = json.load(open(path, encoding="utf-8"))
        anchors["whole_body_following"] = dict(out, measured=datetime.datetime.now().astimezone().isoformat(timespec="minutes"),
                                               tool="mimo_bridge/tools/whole_body_following.py",
                                               gauge="env.py cmd_response_slow.cos over a 25-tick hold (-1 against, 0 unrelated, +1 exact)")
        json.dump(anchors, open(path, "w", encoding="utf-8"), indent=2)
        print("anchor written:", path)


if __name__ == "__main__":
    main()
