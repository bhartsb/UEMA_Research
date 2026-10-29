"""The approximate evaluator's (Judge B's) error on DIFFERENCES between near-identical orders, from
snapshot-restored twins (lane-b1-efference, 2026-09-23; review of b1, MAJOR 3).

ckpt_step_344400 on the CPU, as trained (no new flag), learning off. Bodies of the checkpoint's own
episode companion are restored one at a time (the probe harness's Session). At several moments of
each body's own behaviour the world and the model are snapshotted, and five candidate orders are
formed the way the planner's alternatives are (DESIGN D13): the model's own order, "hold" (every body
handle's target where it is now), and three perturbations of the model's order at the run's babble
sizes (5 cm per position axis inside the reach, 0.5 rad per rotation axis, strength + |N(0, 0.05)|).
The eyes are held in every candidate (their row is the command that keeps them where they are), so
the twins differ only in the body's order.

For each candidate: the PREDICTED five-term wellness sum one moment later, Judge B on the state
transition() predicts (the planner's scorer on the learning route), and the REALISED five-term sum:
the snapshot restored, the order sent for one model step, the model's grounded formulas on the frame
that came (causal_chain.grounded_five). For every pair of candidates from the same snapshot the
error on the difference is (P_i - P_j) - (R_i - R_j).

Writes <out>/tie_band_twins.json. The JSON beside this script (tie_band_twins_ckpt_step_344400.json)
was written on 2026-09-23 04:38-04:48 with 8 bodies x 3 moments; mimo_bridge/causal_chain.py TIE_BAND
is its 90th percentile. Usage: python tie_band_twins.py <out> [bodies] [snaps_per_body]
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[3] / "Source" / "Python"))
import itertools
import json
import os
import time

if __name__ == "__main__":  # only when run as a program: importing never changes the devices (2026-09-26)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import torch

CKPT = (r"D:/UEMA_Research/Saved/mimo_bridge/experiments/"
        r"H84_fresh_drivers_20260918_seg31_toy_and_push/checkpoints/ckpt_step_344400")


def perturb(order, cur_cm, reach, gen, gi):
    """The run's babble sizes on an execution-form order [1, H, 9] (lem_model babble block)."""
    o = np.array(order, dtype=np.float32, copy=True)
    H = o.shape[1]
    body = [i for i in range(H) if i != gi]
    n = gen.normal(0.0, 0.05, size=(H, 9))
    o[0, body, 0:3] = cur_cm[body] + np.clip(o[0, body, 0:3] + n[body, 0:3] * 100.0 - cur_cm[body],
                                             -reach, reach)
    o[0, body, 8] = np.clip(o[0, body, 8] + np.abs(n[body, 8]), 0.0, 1.0)
    rv = gen.normal(0.0, 0.5, size=(H, 3))
    for i in body:
        ang = max(np.linalg.norm(rv[i]), 1e-9)
        dq = np.r_[rv[i] / ang * np.sin(ang / 2), np.cos(ang / 2)]
        q = o[0, i, 3:7]
        x1, y1, z1, w1 = dq
        x2, y2, z2, w2 = q
        o[0, i, 3:7] = [w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2]
    return o


def main():
    out = sys.argv[1]
    n_bodies = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    n_snaps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    os.makedirs(out, exist_ok=True)
    from mimo_bridge.tools import gof_probe_harness as GH
    from mimo_bridge.causal_chain import grounded_five
    import mimo_bridge.lem_model as LM
    print("lem_model:", LM.__file__, flush=True)
    t0 = time.time()
    frozen = GH.load_frozen(CKPT, seed=0)
    ep, note = GH.load_episode(frozen)
    print(note, "load s", round(time.time() - t0, 1), flush=True)
    model = frozen.model
    gi = model.GAZE_IDX
    gen = np.random.default_rng(344400)
    rows = []
    for b in range(min(n_bodies, len(ep["worlds"]) if ep else 1)):
        s = GH.Session(frozen, episode=ep, body=b,
                       babble=float(getattr(frozen.args, "babble_noise", 0.05)))
        try:
            snaps = s.collect_starts(n_snaps, 10, max_steps=10 * n_snaps + 5)
            for k, snap in enumerate(snaps):
                s.restore(snap)
                mom = s.moment()
                cur_cm = model.last_achieved[0, :, 0:3].float().numpy() * 200.0
                reach = 40.0 * float(s.dev.get("rom_limit", 1.0))
                own = s.exec_order(snap["res"].motor_for_action)
                cands = {"own": own, "hold": s.hold_order()}
                for j in range(3):
                    cands["perturb_%d" % (j + 1)] = s.exec_order(perturb(own, cur_cm, reach, gen, gi))
                pred, real = {}, {}
                for name, order in cands.items():
                    pred[name] = GH.predicted_wellness(model, mom, order)
                for name, order in cands.items():
                    s.run_arm(snap, order, steps=1)
                    g = grounded_five(model)
                    real[name] = None if g is None else float(g.float().mean())
                rows.append({"body": b, "snap": k, "step_offset": snap.get("step_offset"),
                             "pred": pred, "real": real})
                print("body", b, "snap", k, {n: (round(pred[n], 4), round(real[n], 4)) for n in pred},
                      flush=True)
        finally:
            s.close()
    errs, pd_all, rd_all, agree = [], [], [], []
    for r in rows:
        for a_, b_ in itertools.combinations(sorted(r["pred"]), 2):
            if None in (r["pred"][a_], r["pred"][b_], r["real"][a_], r["real"][b_]):
                continue
            pd = r["pred"][a_] - r["pred"][b_]
            rd = r["real"][a_] - r["real"][b_]
            errs.append(pd - rd)
            pd_all.append(pd)
            rd_all.append(rd)
            if abs(rd) > 1e-9 and abs(pd) > 1e-9:
                agree.append(1.0 if np.sign(pd) == np.sign(rd) else 0.0)
    e = np.abs(np.asarray(errs))
    q = lambda x, p: None if len(x) == 0 else round(float(np.quantile(x, p)), 6)
    summary = {
        "checkpoint": CKPT, "bodies": n_bodies, "snapshots": len(rows), "pairs": len(errs),
        "abs_error_on_difference": {"median": q(e, 0.5), "p75": q(e, 0.75), "p90": q(e, 0.9),
                                    "p95": q(e, 0.95), "max": q(e, 1.0)},
        "abs_predicted_difference": {"median": q(np.abs(pd_all), 0.5), "p90": q(np.abs(pd_all), 0.9)},
        "abs_realised_difference": {"median": q(np.abs(rd_all), 0.5), "p90": q(np.abs(rd_all), 0.9)},
        "sign_agreement": (None if not agree else round(float(np.mean(agree)), 4)),
        "sign_agreement_pairs": len(agree),
        "seconds": round(time.time() - t0, 1),
        "rows": rows,
    }
    with open(os.path.join(out, "tie_band_twins.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
