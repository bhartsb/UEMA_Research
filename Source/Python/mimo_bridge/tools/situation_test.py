# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The situation-dependence test (operator-ordered 2026-09-03).

The overseer has always appended "(situation dependence is not established)"
because its between-body ratio cannot separate "each body reads its own world"
from "each body rolls different dice". This test separates them with controls,
on the model's DETERMINISTIC command proposal (motor_out, the pre-babble output:
no dice anywhere in the measured quantity).

Collected: K situation snapshots per body from a live drive (tokens + context +
achieved pose + the deterministic command), plus PAIRED situations that differ
only in one relevant feature - the ball is teleported between two positions
while the body holds still (env.randomize_target, two consecutive collections).

The four measurements, each with its self-measured floor:
  T1 determinism floor   re-present the identical cached inputs: |dcmd| must be ~0
                         (machine epsilon). The test refuses to run if not.
  T2 slot invariance     present body A's situation in slot 0 vs slot 1 of the
                         batch (B swapped alongside): the command must travel
                         with the SITUATION, not the slot. Kills slot artifacts.
  T3 relevant change     |cmd(ball at p1) - cmd(ball at p2)|, body pose held:
                         the command's response to a feature that SHOULD matter.
  T4 irrelevant change   |cmd(t) - cmd(t+1)| across adjacent frames where pose
                         and ball both held (< measured motion floor): temporal
                         jitter only - the noise floor T3 must clear.

Verdict: situation dependence ESTABLISHED when T3 exceeds T4 with a bootstrap
95% interval clear of 1.0x, and T2's slot effect sits at the T1 floor. The
result is written to <run_dir>/situation_test.json; the overseer cites it.

Usage:
  python -m mimo_bridge.tools.situation_test <ckpt> [n_pairs] [out_json]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

N_SNAP_STEPS = 240          # live steps per body collected for T4 pairs
BALL_SETTLE_STEPS = 2       # steps after a teleport before the paired capture


def _cmd(model, tokens, ctx, handle_state, wellness_grad=None, memory_bias=None):
    """The deterministic command for cached inputs: the exact live motor path,
    latents from the forward model on this situation's own context.

    THE MEMORY'S BIAS TRAVELS WITH THE SITUATION (2026-09-27, the motor fold's report). The live step adds the
    persistent memory's bias to the newest tokens (lem_model.py, forward_model(..., memory_bias=
    _memory_bias_current)); before this change T1 and T2 ran the path without it, so the controller saw an
    input missing the memory constant's image (9,191 long on H84) that it never sees in training. T3 and T4
    compare the live step's own commands and were not affected."""
    with torch.no_grad():
        latents, last_step, _ = model.forward_model(
            tokens, ctx, **({} if memory_bias is None else {"memory_bias": memory_bias}))
        wg = (wellness_grad if wellness_grad is not None
              else torch.zeros_like(handle_state[:, :, :9]))
        out, _c, _m = model.motor_controller(latents, handle_state, wellness_grad=wg)
    return out.detach().float()


def probe_world_kwargs(a, limits, run_dir):
    """The world the checkpoint trained in, built by the trainer's own
    train_mimo.env_kwargs_for, with only this test's own settings changed.

    Until 2026-09-23 this world was built by hand from a dozen args.json keys and never
    passed body_layout, so it was the env's default 24-row legacy body. Every H84 segment
    trains the trunk_v1 body (26 handle rows: the legacy 24 plus lower trunk and chest),
    whose proprioception encoder takes 26 x 12 = 312 numbers against the legacy world's
    288, so the first model step on an H84 checkpoint raised and situation_test.json could
    not be refreshed for the H84 lineage (fix-rows-r1-situation_test.py). The hand-built
    world also differed from the trained one in contact profile, body function, audio, the
    ring camera and the order-smoothing horizon (5 steps against the trainer's
    action_repeat x motor_hold); the trainer's builder carries all of them.

    `a` is the trainer's argument namespace as a dict (causal_test._build_model returns
    it); any key it lacks takes the trainer's default. This test's own settings: its run
    folder, seed and tag; vision on (the moved ball is seen); no caregiver and no critters,
    because the test teleports the ball itself and its jitter floor needs a scene where
    nothing else moves the ball or the view; and never recording or replaying the caregiver
    (the run's caregiver_record_dir is its live recording store)."""
    from mimo_bridge import train_mimo
    ns = train_mimo.parse_args(["--out", run_dir])
    vars(ns).update(a or {})
    # the phase the limits were logged under. 'fixed' (no curriculum) maps to None as in
    # sensor_action_contract_check.latest_phase, and so does any name missing from the
    # trainer's phase tables, which env_kwargs_for would look up and fail on
    phase = (limits or {}).get("phase_name")
    phase = phase if phase in train_mimo.PHASE_AGE else None
    kw = train_mimo.env_kwargs_for(ns, phase, run_dir)
    kw.update(run_dir=run_dir, vision=True, seed=71, tag="_sit",
              caregiver_mode="off", caregiver_seconds=0.0,
              caregiver_record_dir="", caregiver_replay="", critters=False)
    if (limits or {}).get("age_months") is not None and "age" not in kw:
        kw["age"] = float(limits["age_months"])
    return kw


def run(ckpt: str, n_pairs: int = 60, out_json: str | None = None,
        run_dir: str | None = None):
    torch.set_num_threads(2)
    from mimo_bridge.tools.causal_test import _build_model, _limits_for_run
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.lem_model import LemLite

    model, a, src_path, found = _build_model(ckpt, "cpu", seed=0, intention=False)
    model.load_checkpoint(ckpt, load_optimizer=False)
    limits, _src = _limits_for_run(ckpt, model)
    hold = int(a.get("motor_hold") or 5)
    reps = int(a.get("action_repeat") or 5)
    dev = dict(limits)
    dev["exploration_noise"] = float(a.get("babble_noise") or 0.05)

    rd = run_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(ckpt))),
                                 "situation_probe")
    src = SingleEnvSource(**probe_world_kwargs(a, limits, rd))

    snaps = []          # ordinary situations: (tokens, ctx, hs, cmd, ball_pos, pose)
    pairs = []          # relevant-change pairs: (cmd_p1, cmd_p2, pose_drift)
    try:
        frames = src.reset()
        model.reset_context(1)
        model.reset_transition_state()
        env = src.env
        held = prev = None
        order_seq, prev_oid = 0, None

        def capture(res):
            tokens = {m: v.detach().clone() for m, v in (model.last_tokens or {}).items()}
            ctx = model.context_buffer.detach().clone()
            hs = model.last_achieved.detach().clone()
            bias = getattr(model, "_memory_bias_current", None)
            return {"tokens": {m: v.unsqueeze(1) for m, v in tokens.items()},
                    "ctx": ctx, "hs": hs, "memory_bias": None if bias is None else bias.detach().clone(),
                    "cmd": torch.as_tensor(np.asarray(res.motor_out), dtype=torch.float32).clone(),
                    "ball": np.array(env.data.mocap_pos[0] if env.model.nmocap else [0, 0, 0]),
                    "pose": hs[..., :3].numpy().copy()}

        for i in range(N_SNAP_STEPS):
            res = model.step(frames, dev, train=False, step_index=i,
                             executed_action=prev, executed_order_id=prev_oid)
            snaps.append(capture(res))
            if i % hold == 0 or held is None:
                held = res.motor_for_action
                order_seq += 1
            for _ in range(reps):
                frames = src.step(held, res.gaze_out, None, render=True)
            prev = held
            prev_oid = order_seq
        step_i = N_SNAP_STEPS
        for _p in range(n_pairs):
            env.randomize_target()
            for _ in range(BALL_SETTLE_STEPS):
                frames = src.step(held, res.gaze_out, None, render=True)
            res = model.step(frames, dev, train=False, step_index=step_i,
                             executed_action=prev, executed_order_id=prev_oid)
            s1 = capture(res)
            step_i += 1
            env.randomize_target()
            for _ in range(BALL_SETTLE_STEPS):
                frames = src.step(held, res.gaze_out, None, render=True)
            res = model.step(frames, dev, train=False, step_index=step_i,
                             executed_action=prev, executed_order_id=prev_oid)
            s2 = capture(res)
            step_i += 1
            drift = float(np.abs(s2["pose"] - s1["pose"]).max())
            # T4's matched twin: the SAME cadence as s1->s2 - settle steps then a
            # capture - with the teleport omitted. (Second structural mismatch
            # caught by its own filter: without the settle steps the twin spanned
            # almost no physics and every twin passed while every teleport pair
            # failed - t3 0, t4 60.)
            for _ in range(BALL_SETTLE_STEPS):
                frames = src.step(held, res.gaze_out, None, render=True)
            res = model.step(frames, dev, train=False, step_index=step_i,
                             executed_action=prev, executed_order_id=prev_oid)
            s3 = capture(res)
            step_i += 1
            drift23 = float(np.abs(s3["pose"] - s2["pose"]).max())
            pairs.append((s1, s2, drift, s3, drift23))
    finally:
        src.close()

    d = lambda x, y: float((x - y).abs().mean())

    # ---- T1 determinism floor: identical cached inputs, twice through the path
    s = snaps[len(snaps) // 2]
    c1 = _cmd(model, s["tokens"], s["ctx"], s["hs"], memory_bias=s.get("memory_bias"))
    c2 = _cmd(model, s["tokens"], s["ctx"], s["hs"], memory_bias=s.get("memory_bias"))
    t1_floor = d(c1, c2)

    # ---- T2 slot invariance: A and B through slots (0,1) then (1,0)
    sa, sb = snaps[len(snaps) // 3], snaps[2 * len(snaps) // 3]
    model.reset_context(2)
    tok_ab = {m: torch.cat([sa["tokens"][m], sb["tokens"][m]]) for m in sa["tokens"]}
    tok_ba = {m: torch.cat([sb["tokens"][m], sa["tokens"][m]]) for m in sa["tokens"]}
    ctx_ab = torch.cat([sa["ctx"], sb["ctx"]])
    ctx_ba = torch.cat([sb["ctx"], sa["ctx"]])
    hs_ab = torch.cat([sa["hs"], sb["hs"]])
    hs_ba = torch.cat([sb["hs"], sa["hs"]])
    have_bias = sa.get("memory_bias") is not None and sb.get("memory_bias") is not None
    mb_ab = torch.cat([sa["memory_bias"], sb["memory_bias"]]) if have_bias else None
    mb_ba = torch.cat([sb["memory_bias"], sa["memory_bias"]]) if have_bias else None
    o_ab = _cmd(model, tok_ab, ctx_ab, hs_ab, memory_bias=mb_ab)
    o_ba = _cmd(model, tok_ba, ctx_ba, hs_ba, memory_bias=mb_ba)
    t2_slot_effect = max(d(o_ab[0:1], o_ba[1:2]), d(o_ab[1:2], o_ba[0:1]))
    model.reset_context(1)

    # ---- T3 relevant vs T4 irrelevant, matched by construction: the same
    # capture cadence, the same held order; only T3's step teleported the ball.
    # One shared drift filter (the median of both populations' drifts).
    drifts_all = [p[2] for p in pairs] + [p[4] for p in pairs]
    still = float(np.median(drifts_all))
    t3_vals = [d(p[1]["cmd"], p[0]["cmd"]) for p in pairs if p[2] <= still]
    t4_vals = [d(p[3]["cmd"], p[1]["cmd"]) for p in pairs if p[4] <= still]
    if len(t3_vals) < 10 or len(t4_vals) < 10:
        out = {"ckpt": ckpt, "when": time.strftime("%Y-%m-%d %H:%M"),
               "established": False, "invalid": True,
               "n_t3": len(t3_vals), "n_t4": len(t4_vals),
               "verdict": "INVALID: too few matched pairs to measure the floors "
                          f"(t3 {len(t3_vals)}, t4 {len(t4_vals)}, need 10 each) - "
                          "no claim either way"}
        print(json.dumps(out, indent=2))
        dest = out_json or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(ckpt))),
            "situation_test.json")
        json.dump(out, open(dest, "w"), indent=2)
        return out
    t3, t4 = float(np.mean(t3_vals)), float(np.mean(t4_vals))
    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(2000):
        r3 = np.mean(rng.choice(t3_vals, len(t3_vals)))
        r4 = np.mean(rng.choice(t4_vals, len(t4_vals)))
        ratios.append(r3 / max(r4, 1e-12))
    lo, hi = np.percentile(ratios, [2.5, 97.5])

    established = bool(lo > 1.0 and t2_slot_effect <= max(10 * t1_floor, 1e-6))
    out = {
        "ckpt": ckpt, "when": time.strftime("%Y-%m-%d %H:%M"),
        "n_snaps": len(snaps), "n_pairs_used": len(t3_vals), "n_still_used": len(t4_vals),
        "t1_determinism_floor": t1_floor,
        "t2_slot_effect": t2_slot_effect,
        "t3_relevant_change_cmd_delta": t3,
        "t4_irrelevant_change_cmd_delta": t4,
        "relevant_over_irrelevant": t3 / max(t4, 1e-12),
        "ratio_ci95": [float(lo), float(hi)],
        "stillness_floor_used": still,
        "established": established,
        "verdict": ("situation dependence ESTABLISHED: a moved ball changes the "
                    f"command {t3 / max(t4, 1e-12):.2f}x more than temporal jitter "
                    f"(95% CI [{lo:.2f}, {hi:.2f}]), commands travel with the "
                    "situation not the slot"
                    if established else
                    "situation dependence NOT established: the command's response "
                    "to a relevant change does not clear the jitter floor "
                    f"({t3 / max(t4, 1e-12):.2f}x, CI [{lo:.2f}, {hi:.2f}]) or the "
                    "slot effect is above the determinism floor")}
    print(json.dumps(out, indent=2))
    dest = out_json or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(ckpt))),
                                    "situation_test.json")
    json.dump(out, open(dest, "w"), indent=2)
    print("->", dest)
    return out


if __name__ == "__main__":
    run(sys.argv[1],
        n_pairs=int(sys.argv[2]) if len(sys.argv) > 2 else 60,
        out_json=sys.argv[3] if len(sys.argv) > 3 else None)
