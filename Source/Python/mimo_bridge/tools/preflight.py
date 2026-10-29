# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Effect probes that run BEFORE a training run is trusted.

    python -m mimo_bridge.tools.preflight --args <run_dir>/args.json --out <run_dir>/preflight.json

Why. The overseers watch process health - alive, stepping, RAM, loss falling - and every
serious bug of 2026-08-23/24 was invisible to all of that. The run was healthy while:
the eyes were mechanically frozen (armature x101), the body lay face-DOWN in a standing
posture, the scene was the welded rig instead of the crib, the mobile hung where it could
not be seen, the reflex fired without moving the eyes, and every eye mechanism died at the
first episode reset. Each was found by a one-off probe written after the damage. This file
is those probes made permanent: each tests an EFFECT against physics, never a trigger
count, on a throwaway body built from the run's own args - so a bad flag combination is
caught exactly as launched, in about a minute, before hours of training land on it.

Every check prints value, band, and PASS/FAIL. The overseer treats a missing or failing
preflight as an alarm on the run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run_env_kwargs(args_path: str) -> dict:
    """The world settings the trainer builds for this run: train_mimo.env_kwargs_for on the run's
    args.json laid over the trainer parser's own defaults, at the run's latest logged curriculum
    phase (the start phase before any is logged). The trainer passes the settings it built itself
    (run_probes env_kw); this is the command-line path."""
    from mimo_bridge import train_mimo as tm
    from mimo_bridge.tools.sensor_action_contract_check import latest_phase
    a = json.load(open(args_path))
    run_dir = os.path.dirname(os.path.abspath(args_path))
    ns = tm.parse_args(["--out", run_dir])
    for k, v in a.items():
        setattr(ns, k, v)
    phase = latest_phase(run_dir)
    if phase is None and getattr(ns, "curriculum", "off") == "on":
        phase = ns.start_phase
    return tm.env_kwargs_for(ns, phase, run_dir)


def run_probes(args_path: str, env_kw: dict | None = None) -> dict:
    """env_kw: the BridgeEnv settings of the world the run builds (train_mimo passes its own);
    None rebuilds them from args.json (run_env_kwargs)."""
    from mimo_bridge.env import BridgeEnv
    a = json.load(open(args_path))
    run_dir = os.path.dirname(os.path.abspath(args_path))
    # THE RUN'S OWN BODY. Until 2026-09-23 this built its own keyword set from args.json and never
    # passed body_layout or contact_profile, so every trunk_v1 run (all of H84) was certified on
    # the 24-row legacy body with no lower-trunk or chest handle (fix-rows-r1-preflight).
    kw = dict(run_env_kwargs(args_path) if env_kw is None else env_kw)
    if a.get("scene_binding") == "phase":
        kw["scene"] = "crib"               # every phase maps to crib
    # a throwaway world: its scene files beside the run, a fixed seed, eyes on for the image
    # checks, and no caregiver recording into the lineage's shared event store
    kw.update(run_dir=os.path.join(run_dir, "scenes"), tag="_preflight", seed=1234, vision=True,
              caregiver_record_dir="")
    scene = kw.get("scene", "crib")
    env = BridgeEnv(**kw)
    env.reset()
    m, d = env.model, env.data
    checks = []

    def check(name, value, lo, hi, note="", **extra):
        ok = (lo is None or value >= lo) and (hi is None or value <= hi)
        checks.append(dict(name=name, value=round(float(value), 4),
                           band=[lo, hi], ok=bool(ok), note=note, **extra))
        return ok

    # one order row per body part, picked by NAME: legacy has 24 rows, trunk_v1 26 (the lower
    # trunk and chest come after the eyes' row)
    names = list(env.proprio.handle_names)
    n_rows = len(names)
    zero = np.zeros((n_rows, 9))

    # ---- 1. posture: which way is the face pointing? (the plank/floor bug) ----------
    if scene == "crib":
        cid = m.camera("eye_left").id
        pitch = float(np.degrees(np.arcsin(np.clip(
            -np.asarray(d.cam_xmat[cid]).reshape(3, 3)[2, 2], -1, 1))))
        check("face up: eye pitch deg", pitch, 45.0, None,
              "was -48 (looking at the floor) for seven runs; supine should read +80ish")

        # ---- 2. the mobile is where the eyes are (the invisible-ball bug) ------------
        if env._target_id is not None:
            ball = np.asarray(d.mocap_pos[env._target_id])
            best = 180.0
            for c in ("eye_left", "eye_right"):
                ci = m.camera(c).id
                f = -np.asarray(d.cam_xmat[ci]).reshape(3, 3)[:, 2]
                v = ball - np.asarray(d.cam_xpos[ci])
                n = np.linalg.norm(v)
                if n > 1e-6:
                    best = min(best, float(np.degrees(np.arccos(
                        np.clip(np.dot(f, v / n), -1, 1)))))
            check("ball inside the wide field: deg off gaze", best, None, 30.0,
                  "was 60-120 for every run before the placement fix")

    # ---- 3. the eyes RESPOND to a command (the armature-froze-them bug) --------------
    q_adr = int(m.jnt_qposadr[m.joint("robot:left_eye_horizontal").id])
    for _ in range(30):
        env.step(zero, (np.radians(30.0), 0.0), render=False)
    reached = float(np.degrees(d.qpos[q_adr]))
    check("eye reaches a 30 deg command in 0.3 s: deg", reached, 10.0, None,
          "0.69 when the anti-buzz armature froze the eyeball (H10..H17)")

    # ---- 4. eyes MOVE on their own, and SURVIVE a reset (trigger vs effect) ----------
    if env.saccades is not None:
        env.reset()
        yaws = []
        for i in range(300):
            env.step(zero, (0.0, 0.0), render=(i % 5 == 4))
            yaws.append(float(np.degrees(d.qpos[q_adr])))
        span1 = max(yaws) - min(yaws)
        s1 = env.saccades.n_saccades
        check("eye excursion from spontaneous movement: deg span", span1, 5.0, None,
              "the reflex once counted snaps while the eyes moved 0.004 deg/step")
        env.reset()
        s_mark = env.saccades.n_saccades
        for i in range(300):
            env.step(zero, (0.0, 0.0), render=(i % 5 == 4))
        s2 = env.saccades.n_saccades - s_mark
        check("saccades after an episode reset (vs before)", s2, max(1.0, 0.4 * s1), None,
              "both eye mechanisms died at the first reset until 93aea0ec")

    # ---- 5. the body follows a limb command at a horizon where it can ---------------
    from mimo_bridge.action_decoder import quat_xyzw_to_mat, rotmat_log

    def follows(part, about_spine=False):
        """Order `part` 25 degrees about the vertical (about_spine: about the line from the lower
        trunk to the chest) and every other row to hold its pose, for 250 ms from a reset. Returns
        (direction match: 1 = turned exactly the way ordered, 0 = no match, or no turn at all;
        degrees turned)."""
        env.reset()
        hi_ = names.index(part)
        body = env.proprio.body_ids[hi_]
        axis = np.array([0.0, 0.0, 1.0])
        if about_spine:
            axis = (np.asarray(d.xpos[env.proprio.body_ids[names.index("chest")]])
                    - np.asarray(d.xpos[env.proprio.body_ids[names.index("lower_trunk")]]))
            axis = axis / np.linalg.norm(axis)
        cmd = np.zeros((n_rows, 9))
        for i, b in enumerate(env.proprio.body_ids):
            cmd[i, 0:3] = np.asarray(d.xpos[b]) * 100.0
            q = d.xquat[b]
            cmd[i, 3:7] = np.array([q[1], q[2], q[3], q[0]])
        cmd[:, 8] = 1.0
        q0 = np.asarray(d.xquat[body]).copy()
        R0 = quat_xyzw_to_mat([q0[1], q0[2], q0[3], q0[0]])
        r = np.radians(25.0) / 2
        turn = np.array([axis[0] * np.sin(r), axis[1] * np.sin(r), axis[2] * np.sin(r), np.cos(r)])
        x1, y1, z1, w1 = turn
        x2, y2, z2, w2 = cmd[hi_, 3:7]
        cmd[hi_, 3:7] = np.array([w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2,
                                  w1*z2 + x1*y2 - y1*x2 + z1*w2, w1*w2 - x1*x2 - y1*y2 - z1*z2])
        for _ in range(25):
            env.step(cmd, (0.0, 0.0), render=False)
        q1 = np.asarray(d.xquat[body])
        R1 = quat_xyzw_to_mat([q1[1], q1[2], q1[3], q1[0]])
        Rt = quat_xyzw_to_mat(cmd[hi_, 3:7])
        wc, wg = rotmat_log(Rt @ R0.T), rotmat_log(R1 @ R0.T)
        nc, ng = np.linalg.norm(wc), np.linalg.norm(wg)
        cosv = float(np.dot(wc, wg) / (nc * max(ng, 1e-9))) if ng > 1e-6 else 0.0
        return cosv, float(np.degrees(ng))

    cosv, turned = follows("upperarm_R")
    check("limb follows a command over 250 ms: cosine", cosv, 0.4, None,
          "one-handle baseline measured +0.9ish; near zero = the pathway is broken",
          part="upperarm_R", degrees_turned=round(turned, 1), degrees_ordered=25.0)
    # THE TRUNK ROWS (trunk_v1 only), same 25 degrees, 250 ms and pass line as the arm. A trunk
    # decoder chain or servo that moves nothing reads 0 here, a reversed one below 0; the
    # legacy-body gate could not see either. The chest is turned about the vertical like the arm.
    # The LOWER TRUNK is twisted about the spine instead: turned about the vertical it cannot
    # follow while the body lies on its back (measured 2026-09-23 on segment 31's settings: 1.5 of
    # 25 degrees, direction match 0.18 - its lean joint moves the ordered way but the free pelvis
    # and legs swing back against it; about its bend axis the pelvis swings further than the joint
    # moves and the part turns the opposite way, docs/gain_of_function_v3/agents/
    # fix-rows-r1-preflight.py.md). Twisted about the spine it turned 8.8 of 25 degrees, direction
    # match 1.00; the chest about the vertical 4.6 of 25, direction match 0.99.
    if "chest" in names:
        cosv, turned = follows("chest")
        check("chest follows a command over 250 ms: cosine", cosv, 0.4, None,
              "1 = turned exactly the way it was ordered, 0 = no match or no turn, below 0 = "
              "turned the opposite way: the chest order does not reach the chest's joints",
              part="chest", degrees_turned=round(turned, 1), degrees_ordered=25.0)
    if "lower_trunk" in names and "chest" in names:
        cosv, turned = follows("lower_trunk", about_spine=True)
        check("lower trunk twists about the spine on command over 250 ms: cosine", cosv, 0.4, None,
              "1 = twisted exactly the way it was ordered, 0 = no match or no turn, below 0 = "
              "twisted the opposite way: the lower-trunk order does not reach its joints",
              part="lower_trunk", degrees_turned=round(turned, 1), degrees_ordered=25.0)

    # ---- 6. vision is fresh on the frames the model reads ---------------------------
    ar = int(a.get("action_repeat") or 5)
    fresh = 0
    for i in range(ar * 10):
        raw = env.step(zero, (0.0, 0.0), render=(i % ar == ar - 1))
        if i % ar == ar - 1:
            fresh += raw.eye_left is not None
    check("fresh image on the frames the model reads: of 10", fresh, 10, 10,
          "half the frames were silent repeats before the render-cadence fix")

    body_layout = getattr(env, "body_layout", kw.get("body_layout", "legacy"))
    contact_profile = getattr(env, "contact_profile", kw.get("contact_profile", "legacy"))
    env.close()
    ok = all(c["ok"] for c in checks)
    return dict(type="preflight", ok=ok, timestamp=time.time(),
                scene=scene, args=os.path.abspath(args_path), body_layout=body_layout,
                contact_profile=contact_profile, n_handles=n_rows, checks=checks)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--args", required=True, help="the run's args.json")
    ap.add_argument("--out", default=None, help="default: preflight.json beside args.json")
    a = ap.parse_args(argv)
    rep = run_probes(a.args)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.args)), "preflight.json")
    json.dump(rep, open(out, "w"), indent=2)
    print(f"  body tested: {rep['body_layout']}, {rep['n_handles']} body parts ordered, "
          f"contact {rep['contact_profile']}")
    for c in rep["checks"]:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['value']}  band {c['band']}")
        if "degrees_turned" in c:
            print(f"         turned {c['degrees_turned']} of the {c['degrees_ordered']:.0f} degrees ordered")
        if not c["ok"]:
            print(f"         {c['note']}")
    print(("PREFLIGHT PASS" if rep["ok"] else "PREFLIGHT FAIL") + f" -> {out}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
