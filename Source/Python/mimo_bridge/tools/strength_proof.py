# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Can the actuators, at current strength, flip the body and crawl? (operator order)

Scripted synergies on the RAW actuator rails - no model, no learning, no reward.
The flip tries every lead-leg/sign combination (signs discovered, never assumed)
and measures trunk roll from the torso's own frame. The crawl starts the body
PRONE (root quaternion rotated) and drives an alternating commando-crawl cycle,
measuring net horizontal displacement. Numbers, not impressions.

Usage: python -m mimo_bridge.tools.strength_proof
"""
from __future__ import annotations

import tempfile

import numpy as np


def _act_ids(m, *subs, exclude=()):
    import mujoco
    out = []
    for a in range(m.nu):
        j = int(m.actuator_trnid[a, 0])
        nm = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").lower()
        if any(s in nm for s in subs) and not any(x in nm for x in exclude):
            out.append((a, nm))
    return out


def _roll_deg(m, d):
    """Trunk roll: how far the torso's dorsal axis has rotated from world-up.
    0 = supine (belly up), 180 = prone (belly down)."""
    R = np.asarray(d.xmat[m.body("upper_body").id]).reshape(3, 3)
    up_world = R[:, 0]      # supine at rest: body x-axis points up (measured below)
    return float(np.degrees(np.arccos(np.clip(up_world[2], -1.0, 1.0))))


def main():
    from mimo_bridge.env import BridgeEnv
    e = BridgeEnv(scene="crib", run_dir=tempfile.mkdtemp(), vision=False,
                  render_every=10**9, seed=51, tag="_str", mobile=False)
    m, d = e.model, e.data
    e.reset()

    # calibrate the roll reference: whichever torso axis points most nearly UP at
    # supine rest is the dorsal reference (measured, not assumed)
    R0 = np.asarray(d.xmat[m.body("upper_body").id]).reshape(3, 3)
    axis = int(np.argmax(np.abs(R0[2, :])))
    sign = 1.0 if R0[2, axis] > 0 else -1.0

    def roll_deg():
        R = np.asarray(d.xmat[m.body("upper_body").id]).reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(sign * R[2, axis], -1.0, 1.0))))

    hipsL = _act_ids(m, "left_hip")
    hipsR = _act_ids(m, "right_hip")
    kneeL = _act_ids(m, "left_knee")
    kneeR = _act_ids(m, "right_knee")
    # the trunk, by its REAL names (discovered: the first run matched zero and
    # flipped with legs only): chest lean/rotation + the hip-bend/lean articulation
    trunk = _act_ids(m, "chest_lean", "chest_rot", "hip_bend", "hip_lean")
    shL = _act_ids(m, "left_shoulder")
    shR = _act_ids(m, "right_shoulder")
    elbL = _act_ids(m, "left_elbow")
    elbR = _act_ids(m, "right_elbow")
    print(f"groups: hipL {len(hipsL)} hipR {len(hipsR)} kneeL {len(kneeL)} "
          f"kneeR {len(kneeR)} trunk {len(trunk)} shL {len(shL)} shR {len(shR)} "
          f"elbL {len(elbL)} elbR {len(elbR)}")

    snap = e.save_state() if hasattr(e, "save_state") else None
    q0, v0 = np.array(d.qpos), np.array(d.qvel)

    def reset_pose():
        d.qpos[:] = q0
        d.qvel[:] = v0
        e._mj.mj_forward(m, d)

    def rails(groups_signs, seconds):
        ctrl = np.zeros(m.nu)
        peak = 0.0
        end = 0.0
        for i in range(int(seconds / e.dt)):
            for grp, s in groups_signs:
                for a, _ in grp:
                    lo, hi = m.actuator_ctrlrange[a]
                    ctrl[a] = hi if s > 0 else lo
            e.step(None, None, None, ctrl=ctrl, render=False)
            r = roll_deg()
            peak = max(peak, r)
            end = r
        return peak, end

    # ---------------- FLIP: every lead-leg and sign combination
    print("\nFLIP attempts (peak / settled roll, deg; supine=0, prone=180):")
    best = (0.0, 0.0, "")
    for lead, hips, knee, sh in (("right", hipsR, kneeR, shR), ("left", hipsL, kneeL, shL)):
        for s_hip in (1, -1):
            for s_trunk in (1, -1):
                reset_pose()
                combo = [(hips, s_hip), (knee, 1), (trunk, s_trunk), (sh, s_hip)]
                peak, end = rails(combo, 3.5)
                tag = f"lead={lead} hip{'+' if s_hip>0 else '-'} trunk{'+' if s_trunk>0 else '-'}"
                print(f"  {tag}: peak {peak:6.1f}  settled {end:6.1f}")
                if end > best[1]:
                    best = (peak, end, tag)
    # constant rails are a STATIC push; a real roll is a MOMENTUM move. Phase
    # sequence: both legs up, hard side-swing with trunk twist, release.
    print("\nFLIP momentum attempts (legs up -> side swing -> release):")
    for s_side in (1, -1):
        for s_tw in (1, -1):
            reset_pose()
            ctrl = np.zeros(m.nu)
            peak = end = 0.0
            for i in range(int(4.0 / e.dt)):
                tt = i * e.dt
                ctrl[:] = 0.0
                if tt < 0.6:            # tuck: both legs up
                    for grp, sg in ((hipsL, -1), (hipsR, -1), (kneeL, 1), (kneeR, 1)):
                        for a2, _ in grp:
                            lo, hi = m.actuator_ctrlrange[a2]
                            ctrl[a2] = hi if sg > 0 else lo
                elif tt < 1.5:          # swing: legs and trunk hard to one side
                    for grp, sg in ((hipsL, s_side), (hipsR, s_side),
                                    (trunk, s_tw), (shL, s_side), (shR, s_side)):
                        for a2, _ in grp:
                            lo, hi = m.actuator_ctrlrange[a2]
                            ctrl[a2] = hi if sg > 0 else lo
                # else: release, let momentum carry
                e.step(None, None, None, ctrl=ctrl, render=False)
                r = roll_deg()
                peak = max(peak, r)
                end = r
            print(f"  swing {'+' if s_side>0 else '-'} twist {'+' if s_tw>0 else '-'}: "
                  f"peak {peak:6.1f}  settled {end:6.1f}")
            if end > best[1]:
                best = (peak, end, f"momentum swing{s_side:+d} twist{s_tw:+d}")
    print(f"FLIP verdict: best settled roll {best[1]:.1f} deg ({best[0]:.1f} peak) via {best[2]}"
          f" - {'FULL FLIP: actuators sufficient' if best[1] > 150 else 'PARTIAL: see numbers'}")

    # ---------------- CRAWL: start prone, alternate a commando-crawl cycle
    print("\nCRAWL (prone start, alternating hip/knee/shoulder cycle):")
    reset_pose()
    # rotate the root free joint 180 deg about the world x axis -> prone
    import mujoco
    root_j = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    qa = int(m.jnt_qposadr[root_j])
    quat = np.array(d.qpos[qa + 3: qa + 7])
    flip_q = np.array([0.0, 1.0, 0.0, 0.0])
    w1, x1, y1, z1 = flip_q
    w2, x2, y2, z2 = quat
    d.qpos[qa + 3: qa + 7] = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                              w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                              w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                              w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
    d.qpos[qa + 2] += 0.05
    d.qvel[:] = 0
    e._mj.mj_forward(m, d)
    for _ in range(100):                      # settle onto the belly
        e.step(None, None, None, ctrl=np.zeros(m.nu), render=False)
    print(f"  prone check: roll {roll_deg():.0f} deg (want ~180)")
    start = np.array(d.xpos[m.body("upper_body").id])
    best_crawl = 0.0
    for s in (1, -1):
        pos0 = np.array(d.xpos[m.body("upper_body").id])
        ctrl = np.zeros(m.nu)
        for i in range(int(10.0 / e.dt)):     # 10 s of cycling
            phase = (i * e.dt) % 1.2
            lead_left = phase < 0.6
            push = (phase % 0.6) < 0.3
            for grp, sg in ((hipsL if lead_left else hipsR, s if push else -s),
                            (kneeL if lead_left else kneeR, 1 if push else -1),
                            (shR if lead_left else shL, s if push else -s),
                            (elbR if lead_left else elbL, 1 if push else -1)):
                for a, _ in grp:
                    lo, hi = m.actuator_ctrlrange[a]
                    ctrl[a] = hi if sg > 0 else lo
            e.step(None, None, None, ctrl=ctrl, render=False)
        moved = float(np.linalg.norm((np.array(d.xpos[m.body("upper_body").id]) - pos0)[:2]))
        print(f"  cycle sign {s:+d}: net horizontal displacement {100*moved:.1f} cm in 10 s")
        best_crawl = max(best_crawl, moved)
    print(f"CRAWL verdict: best {100*best_crawl:.1f} cm / 10 s - "
          f"{'locomotion-capable' if best_crawl > 0.15 else 'insufficient with this script: see numbers'}")
    e.close()


if __name__ == "__main__":
    main()
