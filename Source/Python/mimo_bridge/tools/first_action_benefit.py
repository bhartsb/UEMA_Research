# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""First-action benefit probe (spec 10.6 upstream gate; external review finding 4
"confirm or kill"): does the planner's CHOSEN action close more of its own goal's
distance than a matched control action executed from the SAME physics state?

Method (hardened per the 2026-09-08 code-verification review): run the model's
normal acting loop on a frozen checkpoint. At each probe event where the model
just planned, record the planner's ACTUAL goal (the held committed goal under
locality, else the global argmax - source recorded), save full physics state,
then branch in a COUNTERBALANCED random order:
  chosen    the action the planner produced
  replay    the identical action again - the |chosen-replay| difference is the
            repeatability floor the un-restored helper state imposes, and any
            steering gap must clear it
  matched   each handle's own commanded displacement rotated randomly about its
            current achieved position - same magnitude, same anchor, same
            feasibility class (the earlier row-permutation control reassigned
            absolute positions across handles: possibly infeasible, unfair)
  null      hold the current achieved pose
Each branch restores the physics state, executes for one held-order interval,
and the endpoint is encoded by a SEPARATE frozen copy of the model (context
reset per measurement; identical treatment keeps the comparison paired).
Distance = anchor-sense token MSE to the recorded goal.

Reports paired deltas and sign counts. NO thresholds: this tool measures; the
declared criteria of whatever attempt uses it live in the ledger.

State restoration (follow-up review 2026-09-08, upgraded from physics-only):
branches restore env.save_full_state() - physics + mocap + sim clock + the env
RNG stream + touch pooler history + gaze-reflex/saccade internals + audio
transients + critter/mobile state + the command-response delay queue. The
replay branch stays in as the PROOF the restoration works: |chosen-replay| is
the measured residual floor, and the artifact records it. Remaining stated
limit: the measuring copy's context is 1 frame deep at measurement - its
encoding is a per-frame reading, not the live model's full-context one.

python -m mimo_bridge.tools.first_action_benefit --ckpt <ckpt_dir>
    [--events 24] [--warm 120] [--stride 40]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SAVED = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "Saved", "mimo_bridge")

from mimo_bridge.tools.causal_test import _build_model, _limits_for_run  # noqa: E402
from mimo_bridge.tools.sensor_action_contract_check import (  # noqa: E402
    resolved_env_kwargs, hold_current)
from mimo_bridge.lem_model import LemLite  # noqa: E402

ANCHOR_MODS = ("vision", "tactile", "proprioception")
GAZE_HANDLE = 23


def free_ram_gb() -> float:
    import ctypes.wintypes as wt

    class M(ctypes.Structure):
        _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD)] + \
                   [(n, ctypes.c_uint64) for n in ("tp", "ap", "tpf", "apf", "tv", "av", "aev")]
    m = M(); m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ap / 1e9


def physics_state(env):
    """Full env state for paired branching: physics + helper/RNG/clock state
    (env.save_full_state; upgraded 2026-09-08 from mjSTATE_FULLPHYSICS-only,
    which left clocks, RNG streams and touch history diverging across branches)."""
    return env.save_full_state()


def restore_physics(env, snap):
    env.restore_full_state(snap)


def encode_frame(meter, frames, meter_dev):
    """Per-frame token reading from the frozen measuring copy: reset context,
    one eval step under the run's own developmental limits (a bare dict lacks
    required keys - rom_limit KeyError on first run), noise zeroed. Identical
    treatment for every branch keeps the comparison paired."""
    meter.reset_transition_state()
    with torch.no_grad():
        meter.step(frames, meter_dev, train=False, step_index=0)
    return {m: t.detach().float() for m, t in meter.last_tokens.items()}


def tok_dist(a, b, mods):
    ds = [F.mse_loss(a[m][0], b[m][0]).item() for m in mods if m in a and m in b]
    return float(np.mean(ds)) if ds else float("nan")


def matched_control(action: np.ndarray, achieved_cm: np.ndarray, rng) -> np.ndarray:
    """Physically matched comparator (code-verification review 2026-09-08: the
    earlier row permutation reassigned ABSOLUTE positions across handles - a
    possibly infeasible or harder command, an unfair control). This keeps every
    handle anchored to its own current achieved position and rotates only its
    commanded DISPLACEMENT by a random rotation: same per-handle displacement
    magnitude, same rotation/radius/strength columns, same feasibility class."""
    out = np.array(action, copy=True)
    for h in range(out.shape[1]):
        if h == GAZE_HANDLE:
            continue
        disp = action[0, h, 0:3] - achieved_cm[0, h]
        # uniform random rotation via normalized quaternion
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([[1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                      [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                      [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])
        out[0, h, 0:3] = achieved_cm[0, h] + R @ disp
    return out


def state_wellness(meter, frame) -> float:
    """Realized STATE wellness of one endpoint frame, through the canonical
    formulas (2026-09-09 prospective-wellness plan, phase 1): tactile comfort
    from the raw patch features, capability from the tokenized capacity
    features, pose-boundary from the achieved positions. Prediction-success is
    deliberately absent - it needs targets, and using it here would let 'was
    predictable' masquerade as 'felt good' (the reviewer's pain-anticipation
    point). Range: 0 (crushed, exhausted, jammed at limits) to 3 (gentle
    contact, fresh, mid-range pose)."""
    dev = torch.device("cpu")
    w = meter.wellness
    # NORMALIZED touch fields through THE SAME preparation training uses:
    # tokenizer normalization (fields / _FIELD_NORMS) and then the
    # strongest-32 point-cloud selection (touch_point_cloud), because the
    # training wellness call feeds the formula frame["tactile"][..., 3:11] -
    # the point cloud's feature columns. 09-09 reviews: raw Pa scored every
    # contact 0; and skipping the strongest-32 selection read 0.242 where
    # training reads 0.017 on a 53-contact frame (dense-contact parity).
    from tokenizer.modalities.touch import _FIELD_NORMS
    from mimo_bridge.lem_model import touch_point_cloud
    normed = torch.as_tensor(np.asarray(frame.patches)[:, 0:8]
                             / _FIELD_NORMS[:8], dtype=torch.float32)
    cloud = touch_point_cloud(normed, None)          # [32, 12], features 3:11
    tac = float(w._formula_tactile_comfort(cloud[None, :, 3:11], 1, dev))
    if getattr(meter, 'contact_profile', 'legacy') in ('mechanical_v2','mechanical_v2_local','mechanical_v3'):
        # Use the same complete local fields and history as the live wellness path.
        from lem.model.contact_wellness import contact_response
        _, _, _, extras = meter._batch_frames([frame])
        raw = extras['raw_tactile']
        tac = float(contact_response(raw['features'], raw.get('sensitivity'), raw.get('valid'),
                                    profile=raw['profile'], history=raw['history'])['comfort'])
    cap_t = getattr(frame.tokenized, "motor_capability", None)
    cap = float(w._formula_capability(
        None if cap_t is None else torch.as_tensor(np.asarray(cap_t),
                                                   dtype=torch.float32),
        1, dev))
    # BODY-RELATIVE posture, mirroring training's preparation exactly
    # (lem_model ~1237: handles minus their centroid; 09-09 review counterexample:
    # the uncentered world-coordinate version changed 2.5 -> 2.167 when the same
    # posture was translated 100 cm, while training's score stayed 1)
    ach = torch.as_tensor(frame.bones[:, 4:7] / 200.0,
                          dtype=torch.float32).unsqueeze(0)
    ach = ach - ach.mean(dim=1, keepdim=True)
    bnd = float(w._formula_motor_state(ach, 1, dev))
    return {"tactile": tac, "capability": cap, "posture": bnd,
            "total": tac + cap + bnd}


def _wellness_verdict(rows, key_a="chosen", key_b="matched") -> dict:
    """key_a-vs-key_b on realized state wellness (higher = felt better), gated on
    contact-bearing events, and the replay floor is ENFORCED (09-09 review: a
    benefit smaller than the disagreement between identical actions was being
    labeled detected). Also reports the worst moment during execution, so a
    comfortable endpoint after painful contact cannot read as comfort."""
    wa = np.array([r["w_" + key_a] for r in rows])
    wb = np.array([r["w_" + key_b] for r in rows])
    wr = np.array([r["w_replay"] for r in rows])
    wbase = np.array([r["w_" + ("off" if "w_off" in rows[0] else "chosen")] for r in rows])
    n_contact = int(sum(1 for r in rows if r.get("contact_bearing")))
    diff = wa - wb
    n = len(diff)
    wfloor = np.abs(wbase - wr)
    out = {"comparison": f"{key_a}-vs-{key_b}",
           "contact_bearing_events": n_contact, "events": n,
           "mean_w_delta": {key_a: float(wa.mean()), key_b: float(wb.mean())},
           "mean_worst_moment": {key_a: float(np.mean([r["wmin_" + key_a] for r in rows])),
                                 key_b: float(np.mean([r["wmin_" + key_b] for r in rows]))},
           "wellness_repeatability_floor": {
               "mean_abs": float(wfloor.mean()),
               "p90_abs": float(np.percentile(wfloor, 90))}}
    if n < 8 or n_contact < 8:
        out["verdict"] = "not_evaluated"
        out["reason"] = (f"{n_contact} contact-bearing events of {n}; needs >= 8 "
                         "of each - touch is the only state-wellness channel "
                         "that moves at this horizon")
        return out
    sem = float(diff.std(ddof=1) / np.sqrt(n))
    mean = float(diff.mean())
    line = max(3.0 * sem, float(np.percentile(wfloor, 90)))
    # WORST-MOMENT GATE (09-09 feasibility review: the minimum was printed but
    # never used, so a comfortable endpoint after a painful transient still
    # counted as steering). Same arithmetic as the endpoint: paired difference
    # of per-event minima against its own 3xSEM-or-floor line.
    dmin = (np.array([r["wmin_" + key_a] for r in rows])
            - np.array([r["wmin_" + key_b] for r in rows]))
    sem_min = float(dmin.std(ddof=1) / np.sqrt(n))
    mean_min = float(dmin.mean())
    # the MINIMUM's own replay floor, not the endpoint's (09-09 follow-up
    # review counterexample: an exactly repeatable 0.1 minimum loss passed as
    # steering because the endpoint replay error 0.5 was used as its floor)
    base_key = "off" if "w_off" in rows[0] else "chosen"
    wminfloor = np.abs(np.array([r["wmin_" + base_key] for r in rows])
                       - np.array([r["wmin_replay"] for r in rows]))
    line_min = max(3.0 * sem_min, float(np.percentile(wminfloor, 90)))
    if mean > line and mean_min < -line_min:
        verdict = "endpoint_gain_with_transient_harm"   # NOT a pass
    elif mean > line:
        verdict = "wellness_steering_detected"
    elif mean < -line or mean_min < -line_min:
        verdict = "ANTI_wellness_steering_detected"
    else:
        verdict = "no_effect_above_instrument_resolution"
    out.update(paired_mean=mean, paired_sem=sem,
               detection_line=line,
               detection_line_note="max(3x paired SEM, replay-floor p90) - the "
                                   "floor is part of the bar, not a footnote",
               worst_moment_paired_mean=mean_min,
               worst_moment_detection_line=line_min,
               worst_moment_replay_floor_p90=float(np.percentile(wminfloor, 90)),
               worst_moment_note="a steering pass additionally requires the "
                                 "worst moment NOT be measurably worse; its "
                                 "line uses the MINIMUM's own replay floor",
               verdict=verdict)
    return out


def _measurement_verdict(ch: np.ndarray, mt: np.ndarray,
                         comparison: str = "chosen-vs-matched",
                         preserve_margin: float | None = None,
                         replay_floor: np.ndarray | None = None) -> dict:
    """What this instrument can say about steering, stated by the instrument
    itself (follow-up review 2026-09-08: raw measurements were being over-read).
    Detection line = 3x the paired standard error, which already contains the
    un-restored helper-state noise the replay floor measures.

    SIGN: these are goal-DISTANCE deltas - NEGATIVE mean = the first branch ended
    CLOSER to its goal (better). 09-09 feasibility review: a +0.0045 mean was
    reported as 'goal pursuit unchanged'; positive means WORSE approach, and
    'unchanged' additionally requires a declared margin the CI excludes."""
    diff = ch - mt
    n = len(diff)
    if n < 8:
        return {"comparison": comparison, "verdict": "not_evaluated",
                "reason": f"only {n} events; needs >= 8"}
    sem = float(diff.std(ddof=1) / np.sqrt(n))
    resolution = 3.0 * sem
    # replay floor ENFORCED here too (09-09 feasibility review: the wellness
    # verdict enforced it while this goal verdict quietly did not)
    if replay_floor is not None and len(replay_floor):
        resolution = max(resolution, float(np.percentile(np.abs(replay_floor), 90)))
    mean = float(diff.mean())
    if abs(mean) > resolution:
        v = ("steering_effect_detected" if mean < 0 else
             "ANTI_steering_effect_detected")
    else:
        v = "no_effect_above_instrument_resolution"
    out = {"comparison": comparison, "verdict": v,
           "paired_mean": round(mean, 5), "paired_sem": round(sem, 5),
           "detection_line_3sem": round(resolution, 5),
           "detection_line_includes_replay_floor": replay_floor is not None,
           "sign_note": "negative mean = first branch closer to goal (better)",
           "meaning": "a true per-event steering benefit larger than the detection "
                      "line would have been seen; below it this instrument cannot "
                      "distinguish steering from zero - it does NOT certify that "
                      "the true benefit is exactly zero"}
    # A PRESERVATION claim ('goal pursuit unchanged') is only made against a
    # DECLARED acceptable-harm margin, with the 95% confidence bound inside it
    # AND the replay floor inside it (09-09 follow-up review counterexample:
    # 'preserved within 0.01' was claimed while identical actions disagreed by
    # 0.2 - an instrument that cannot repeat to the margin cannot certify it).
    if preserve_margin is not None:
        harm_hi = mean + 1.96 * sem     # upper bound on harm (positive = worse)
        fl = (float(np.percentile(np.abs(replay_floor), 90))
              if replay_floor is not None and len(replay_floor) else 0.0)
        bound = max(harm_hi, fl)
        out["preservation"] = {
            "declared_margin": float(preserve_margin),
            "harm_upper_ci95": round(harm_hi, 5),
            "replay_floor_p90": round(fl, 5),
            "verdict": ("preserved_within_declared_margin"
                        if bound < float(preserve_margin)
                        else "preservation_NOT_established")}
    else:
        out["preservation"] = "unmeasured (no declared margin supplied)"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--events", type=int, default=24)
    ap.add_argument("--warm", type=int, default=120)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prospective-wellness", choices=["run", "off", "add"],
                    default="run", help="A/B override of the planner term; \"run\" = checkpoint's own setting")
    ap.add_argument("--prospective-wellness-w", type=float, default=0.15)
    ap.add_argument("--pw-paired", type=float, default=None,
                    help="same-decision paired A/B: at every planning event, replan with weight 0 and this weight from identical inputs and compare "
                         "(09-09 review requirement 3)")
    ap.add_argument("--preserve-margin", type=float, default=None,
                    help="declared acceptable goal-approach harm for a preservation "
                         "claim; without it the artifact says preservation is "
                         "unmeasured (09-09 feasibility review)")
    ap.add_argument('--output', help='Separate artifact path; preserve earlier experiments')
    args = ap.parse_args(argv)

    if free_ram_gb() < 8.0:
        print(f"REFUSED: {free_ram_gb():.1f} GB free RAM < 8.0 launch floor "
              f"(standing rule after the 2026-09-08 memory incidents)")
        return 2

    # A/B override (2026-09-09 prospective-wellness plan phase 3): force the
    # planner term on/off regardless of the checkpoint's own args, recorded in
    # the artifact. "run" = the checkpoint's recorded setting, untouched.
    ov = None
    if args.prospective_wellness != "run":
        ov = {"prospective_wellness": args.prospective_wellness,
              "prospective_wellness_w": args.prospective_wellness_w}
    live, a, _s, _f = _build_model(args.ckpt, "cpu", args.seed, intention=True,
                                   arg_overrides=ov)
    meter, _a2, _s2, _f2 = _build_model(args.ckpt, "cpu", args.seed, intention=False)
    for mdl in (live, meter):
        if a.get("audio") == "real":
            mdl.config["modalities"].setdefault("audio", {})["data_source"] = "real"
        ape = getattr(mdl, "_agency_pay_ema", None)
        if ape is not None and torch.is_tensor(ape) and ape.numel() != 1:
            mdl._agency_pay_ema = ape.mean().reshape(1)
    if a.get("intention_every") is not None:
        live.intention_every = int(a["intention_every"])
    limits, _ = _limits_for_run(args.ckpt, live)
    from mimo_bridge.datasource import SingleEnvSource
    reps = int(a.get("action_repeat") or 5)
    hold = int(a.get("motor_hold") or 5)
    # the trainer's own resolved env construction (age, delays, seed, phase);
    # the earlier local helper dropped those (2026-09-08 review, environment gap).
    # Phase comes from THE CHECKPOINT's own recorded curriculum state, not from
    # the run's latest stats (follow-up review: a probe of an old checkpoint must
    # rebuild that checkpoint's world, not the run's current one).
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt)))
    ckpt_phase = None
    try:
        cs = json.load(open(os.path.join(args.ckpt, "meta.json"))).get("curriculum_state") or {}
        ckpt_phase = cs.get("phase")
        if ckpt_phase == "fixed":
            ckpt_phase = None
    except OSError:
        pass
    env_kw, env_phase = resolved_env_kwargs(run_dir, phase_override=ckpt_phase)
    src = SingleEnvSource(**env_kw)
    rng = np.random.default_rng(9900 + args.seed)

    rows = []
    t0 = time.time()
    try:
        live.reset_transition_state()
        frames = src.reset()
        dev = dict(limits)
        dev["exploration_noise"] = float(a.get("babble_noise") or 0.0)
        meter_dev = dict(limits, exploration_noise=0.0)
        held, prev_exec = None, None
        i, events_done, rejected_pairings = 0, 0, 0
        while events_done < args.events and i < args.warm + args.events * args.stride * 3:
            with torch.no_grad():
                res = live.step(frames, dev, train=False, step_index=i,
                                executed_action=prev_exec)
            if i == 20:
                per = (time.time() - t0) / 21.0
                print(f"[timing] {per:.2f}s/step", flush=True)
            if i % hold == 0 or held is None:
                held = res.motor_for_action

            li = live.last_intention
            probe_now = (i >= args.warm and i % args.stride == 0
                         and li is not None and len(live.goal_ring) >= 20)
            if probe_now:
                # THE PLANNER'S ACTUAL GOAL, not a reconstruction (review 2026-09-08):
                # under goal-locality the live policy holds a committed goal that a
                # global argmax would misreport; record which source was used.
                with torch.no_grad():
                    if getattr(live, "goal_locality", False) and live._gl_held_tok is not None:
                        goal = {m: v.detach().float().clone()
                                for m, v in live._gl_held_tok.items()}
                        goal_src = "held_local"
                    else:
                        wl = torch.stack([w for _, w in live.goal_ring])
                        bt = int(wl.argmax(dim=0)[0])
                        goal = {m: torch.stack([t[m] for t, _ in live.goal_ring])[bt]
                                .detach().float() for m in live.active}
                        goal_src = "global_argmax"
                state0 = physics_state(src.env)
                f_now = frames
                d_before = tok_dist(encode_frame(meter, f_now, meter_dev), goal, ANCHOR_MODS)
                ach_cm = (live.last_achieved[:, :, 0:3].detach().float().cpu().numpy()
                          * 200.0)   # verified: achieved world cm (motor_to_echo)
                if args.pw_paired is not None:
                    # SAME-DECISION paired A/B (09-09 review requirement 3): both
                    # planner variants re-run from the IDENTICAL stored decision
                    # inputs. Weight 0 must reproduce the executed choice (the
                    # per-event pairing check below proves it); the difference at
                    # weight w is attributable to the term alone.
                    a_off_raw, _m0 = live.replan_last(0.0)
                    a_add_raw, _m1 = live.replan_last(float(args.pw_paired))
                    plan0 = (live._last_plan_inputs or {}).get("plan0")
                    # PAIRING ANCHOR = the RAW stored first action of the executed
                    # plan (first paired run compared against the EXECUTED command,
                    # which had passed the execution transform and exploration
                    # noise - read 38.7 apart and correctly failed its own check)
                    pairing_dev = (float((a_off_raw - plan0).abs().max())
                                   if a_off_raw is not None and plan0 is not None
                                   else None)
                    # ENFORCED pairing (09-09 feasibility review: failure was
                    # reported, not rejected; and the old `continue` skipped the
                    # loop counter - an infinite loop when replanning failed).
                    # Tolerance basis: measured 0.0e+00 on ckpt_step_3458821;
                    # 1e-6 is ~500x below the smallest normalized command
                    # quantum that moves the body (0.1 cm of 200 cm = 5e-4).
                    if (a_add_raw is None or pairing_dev is None
                            or not np.isfinite(pairing_dev)
                            or pairing_dev > 1e-6):
                        # `not isfinite` because NaN > 1e-6 is False - a NaN
                        # pairing sailed through the numeric guard (09-09
                        # follow-up review)
                        rejected_pairings += 1
                        branches = None
                    else:
                        # both arms go to the body through the SAME executed-
                        # order pipeline as the live path: execution transform
                        # WITH the body-relative reach clamp, then the SHARED
                        # command smoother with the live step's own previous-
                        # command state (09-09 follow-up review: unsmoothed
                        # branches compared orders the body never receives -
                        # at smooth_actions 0.35 only 35% of a new command
                        # appears in the first executed order). No exploration
                        # noise on either arm (planner actions get none live).
                        ach_t = live.last_achieved.detach().float()
                        sm_prev = getattr(live, "_last_smooth_prev", None)
                        with torch.no_grad():
                            a_off = LemLite.smooth_blend(
                                live.execution_transform(
                                    a_off_raw.to(live.device), dev,
                                    handle_state=ach_t),
                                sm_prev, live.smooth_actions).cpu().numpy()
                            a_add = LemLite.smooth_blend(
                                live.execution_transform(
                                    a_add_raw.to(live.device), dev,
                                    handle_state=ach_t),
                                sm_prev, live.smooth_actions).cpu().numpy()
                        branches = {"add": a_add, "off": a_off, "replay": a_off,
                                    "matched": matched_control(a_off, ach_cm, rng)}
                    key_a, key_b = "add", "off"
                else:
                    branches = {"chosen": np.asarray(held),
                                "replay": np.asarray(held),   # identical action twice:
                                # the repeatability floor (review requirement)
                                "matched": matched_control(np.asarray(held), ach_cm, rng),
                                "null": hold_current(f_now[0].bones)[0]}
                    key_a, key_b = "chosen", "matched"
                    pairing_dev = None
                if branches is not None:
                    # counterbalanced execution order per event (review: fixed order
                    # let branch position masquerade as branch identity)
                    order = list(rng.permutation(list(branches.keys())))
                    deltas = {}
                    wb = state_wellness(meter, f_now[0])
                    w_end, w_min, contact_any = {}, {}, False
                    for name in order:
                        act = branches[name]
                        restore_physics(src.env, state0)
                        bf = f_now
                        cmd = act.reshape(1, -1, 9)
                        # EACH ARM'S OWN gaze (09-09 feasibility review: every
                        # branch executed with the ORIGINAL live step's gaze, so
                        # a variant's planned eye command never reached the env)
                        arm_gaze = live._gaze_from_handle(torch.as_tensor(
                            cmd, dtype=torch.float32, device=live.device))
                        wmin = float("inf")
                        for r in range(hold * reps):
                            bf = src.step(cmd, arm_gaze, None,
                                          render=(r == hold * reps - 1))
                            # INTERVENING pain counts (09-09 review: a comfortable
                            # endpoint after painful contact must not read as comfort)
                            wmin = min(wmin, state_wellness(meter, bf[0])["total"])
                        d_after = tok_dist(encode_frame(meter, bf, meter_dev), goal, ANCHOR_MODS)
                        deltas[name] = d_after - d_before
                        w_end[name] = state_wellness(meter, bf[0])
                        w_min[name] = wmin
                        if float(np.asarray(bf[0].patches)[:, 0].max()) > 0:
                            contact_any = True
                    restore_physics(src.env, state0)
                    # UNROUNDED (09-09 review: rows were rounded before summaries)
                    row = {"step": i, "d_before": d_before,
                           "goal_source": goal_src, "order": order,
                           "w_before": wb, "contact_bearing": bool(contact_any)}
                    if pairing_dev is not None:
                        row["pairing_dev"] = pairing_dev
                    for k, v in deltas.items():
                        row[k] = v
                    for k, v in w_end.items():
                        row["w_" + k] = v["total"] - wb["total"]
                        row["wc_" + k] = v            # per-component endpoint scores
                        row["wmin_" + k] = w_min[k]   # worst moment during execution
                    rows.append(row)
                    events_done += 1
                    print(f"event {events_done}/{args.events} step {i}: "
                          f"{key_a} {deltas[key_a]:+.4f}  {key_b} {deltas[key_b]:+.4f}  "
                          f"| wellness {key_a} {w_end[key_a]['total'] - wb['total']:+.4f} "
                          f"{key_b} {w_end[key_b]['total'] - wb['total']:+.4f}"
                          + (f"  pairing_dev {pairing_dev:.2e}" if pairing_dev is not None else "")
                          + (" contact" if contact_any else ""),
                          flush=True)

            gz = []
            for r in range(reps):
                frames = src.step(held, res.gaze_out, None, render=(r == reps - 1))
                gz.append(np.stack([np.asarray(f.meta.get("gaze_cmd") or (0.0, 0.0),
                                               dtype=np.float32) for f in frames]))
            prev_exec = LemLite.gaze_to_handle(held, np.mean(np.stack(gz), axis=0))
            i += 1
    finally:
        src.close()

    paired = args.pw_paired is not None
    key_a, key_b = ("add", "off") if paired else ("chosen", "matched")
    if not rows:
        # a valid NOT-EVALUATED result, never a crash (09-09 follow-up review:
        # all-events-rejected raised IndexError on the empty summary arrays)
        out = {"ckpt": os.path.basename(os.path.normpath(args.ckpt)),
               "ckpt_path": os.path.abspath(args.ckpt), "seed": args.seed,
               "mode": "paired_same_decision" if paired else "chosen_vs_matched",
               "events": 0, "verdict": "not_evaluated",
               "reason": ("every planning event was rejected or none occurred; "
                          f"rejected_pairings={rejected_pairings}"),
               "pw_paired_weight": args.pw_paired,
               "source_sha256": __import__("hashlib").sha256(
                   open(os.path.abspath(__file__), "rb").read()).hexdigest()}
        os.makedirs(os.path.join(SAVED, "first_action"), exist_ok=True)
        p = os.path.join(SAVED, "first_action",
                         f"{out['ckpt']}_s{args.seed}_not_evaluated.json")
        json.dump(out, open(p, "w"), indent=1)
        print(json.dumps(out, indent=1))
        print("written:", p)
        return 0
    A = np.array([r[key_a] for r in rows])
    B_ = np.array([r[key_b] for r in rows])
    rp = np.array([r["replay"] for r in rows])
    base = np.array([r["off" if paired else "chosen"] for r in rows])
    floor = np.abs(base - rp)        # identical action twice: pure helper-state noise
    out = {"ckpt": os.path.basename(os.path.normpath(args.ckpt)),
           "ckpt_path": os.path.abspath(args.ckpt), "seed": args.seed,
           # 09-09 review: EVERY effective setting inside the artifact
           "prospective_wellness_arm": args.prospective_wellness,
           "prospective_wellness_w": args.prospective_wellness_w,
           "pw_paired_weight": args.pw_paired,
           "mode": "paired_same_decision" if paired else "chosen_vs_matched",
           "source_sha256": __import__("hashlib").sha256(
               open(os.path.abspath(__file__), "rb").read()).hexdigest(),
           "events": len(rows),
           "curriculum_phase": env_phase,
           "resolved_env_kwargs": {k: v for k, v in env_kw.items() if k != "run_dir"},
           "env_parity_notes": "seed is the run's BASE seed; the trainer offsets the "
                               "env seed by resume step per segment - this probe world "
                               "is a fresh sample of the same distribution, not a "
                               "bit-identical replay of the checkpoint's segment",
           "mean_goal_delta": {k: float(np.mean([r[k] for r in rows]))
                               for k in rows[0] if k in
                               ("chosen", "replay", "matched", "null", "add", "off")},
           "repeatability_floor": {"mean_abs": float(floor.mean()),
                                   "p90_abs": float(np.percentile(floor, 90)),
                                   "meaning": "identical-action replay difference under "
                                              "FULL state restoration; any claimed gap "
                                              "must clear this floor - ENFORCED in the "
                                              "verdicts since the 09-09 review"},
           "goal_verdict": _measurement_verdict(A, B_, comparison=f"{key_a}-vs-{key_b}",
                                                preserve_margin=args.preserve_margin,
                                                replay_floor=floor),
           "realized_wellness": _wellness_verdict(rows, key_a, key_b),
           "rows": rows,
           "note": "goal delta = goal-token distance change over one held order "
                   "(negative = closed distance; internal scale, not toy arrival). "
                   "Wellness = tactile+capability+posture on 0-3, posture "
                   "body-centered like training. Order counterbalanced; replay "
                   "measures the floor; verdicts enforce it."}
    if paired:
        pd = [r["pairing_dev"] for r in rows if "pairing_dev" in r]
        out["pairing_check"] = {
            "max_abs_dev_off_vs_executed": float(max(pd)) if pd else None,
            "rejected_events": rejected_pairings,
            "tolerance": 1e-6,
            "meaning": "weight-0 replan must reproduce the executed action; "
                       "events exceeding the tolerance are REJECTED from the "
                       "comparison (enforced since the 09-09 feasibility "
                       "review), and the rejection count is part of the result"}
    os.makedirs(os.path.join(SAVED, "first_action"), exist_ok=True)
    arm = ("" if args.prospective_wellness == "run" else
           f"_pw{args.prospective_wellness}"
           + (f"{args.prospective_wellness_w:g}" if args.prospective_wellness == "add" else ""))
    if paired:
        arm += f"_paired{args.pw_paired:g}"
    p = args.output or os.path.join(SAVED, "first_action", f"{out['ckpt']}_s{args.seed}{arm}.json")
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    tmp = p + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1, allow_nan=False)
    os.replace(tmp, p)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print("written:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
