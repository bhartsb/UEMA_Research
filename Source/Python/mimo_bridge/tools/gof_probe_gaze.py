# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Gaze probes (addendum v2 section 6, build item 7).

    python -m mimo_bridge.tools.gof_probe_gaze --ckpt <checkpoint folder> [--states 4]

Seven measurements, one JSON (Saved/mimo_bridge/gof_probes/<run>/<ckpt>/gaze.json):

  step_response        physics only, no model: the eye servo at the run's gaze_kp/gaze_kd told to
                       jump 5, 10 and 20 degrees. Rise time, settle time, peak speed. Settles the
                       two figures in the code (env.py: "110 deg/s, 130 ms"; gaze_reflex.py: the
                       eye "can actually reach 1.235 rad/s" = 70.8 deg/s) and says whether a
                       100-150 ms prediction horizon covers a saccade.
  counterfactual       from the same restored moment, ONLY the gaze command changes (left, right,
                       up, down, hold; the body holds its pose). What the model predicted the eyes
                       would see against what they then saw, at 50 ms on BOTH routes - the trained
                       route (the prediction the next-frame lesson grades, pred["preds"]) and the
                       imagination route the planner uses (predictive_state.transition) - and at
                       100 and 150 ms on the imagination route. Every arm records the command it
                       asked for and the command it sent (clipped to the eye's range, flagged).
  prediction_before    every prediction is computed from the moment BEFORE the order is sent
                       (restored snapshot); the record carries both, in that order.
  repeatability        the same command from the same restored moment three times (an
                       instrument check: a frozen model on a deterministic body must repeat
                       exactly), and the same command from different moments (spread of the
                       eye's movement and of the prediction's direction).
  generalization       the head turned 10 degrees first, then the same commands.
  ablation             the same predictions with the gaze row replaced by "keep the eyes where
                       they are": how much worse the prediction gets without the gaze command.
  wellness_selection   two imagined gaze candidates (left, right) scored by the approximate
                       wellness evaluator; does the planner's own chosen gaze from that moment go
                       toward the better-scored one?

Headline: the share of the real change captured at 150 ms - when only the gaze command changes,
how far the imagined picture moves ALONG the way the real picture moved, as a share of the real
move (0 = the prediction ignores the command, 1 = it foresees the whole change, negative = the
wrong way). Pair discrimination (0.5 = guessing) is reported beside it ONLY for pairs whose
predicted difference stands above the pair's measured noise line: twice the spread of that
pair's predicted difference over the main moment and PAIR_NOISE_STARTS nearby eye starts (the
same moment, the eyes first held one degree away). A pair under its line is "undecided". Measured
on ckpt_step_344400 in the first smoke run (2026-09-22): the predictions of two commands differed
by a millionth of the real difference, so which command a pair was matched to was decided by that
millionth, and 4 such pairs were counted as "told apart correctly".
"""
from __future__ import annotations

import os
if __name__ == "__main__":
    # RUN AS A PROGRAM, A PROBE WORKS ON THE CPU: the card is hidden here, before torch starts, as it always was.
    # Importing the module never changes the process's devices (2026-09-26): the hide used to run at import, so a
    # test process that collected any probe test lost the card and skipped the card continuation test. A program
    # that imports a probe and wants the CPU hides the card at its own entry point (record:
    # docs/gain_of_function_v4/agents/probe_gpu_hiding_2026-09-26.md).
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import argparse
import itertools
import json
import math
import time

import numpy as np
import torch

from mimo_bridge.tools import gof_probe_harness as H

PROBE, ITEM = "gaze", 7
# Gaze offsets for the counterfactual arms, degrees (azimuth, elevation). 10 and 8 degrees sit
# inside the range of the run's own eye moves: 99 in 100 one-moment (50 ms) eye moves on
# H84 segment 31's world stayed under 8.3 degrees and 99 in 100 150 ms moves under 22.1
# (b1-causal-efference measurement, lem/model/efference.py EFFERENCE_SCALES, 2026-09-22), and
# both are within the eye's joint range (+/-45 azimuth, -47 to +33 elevation,
# gaze_reflex.AZ_RANGE/EL_RANGE) from any resting eye position within 12 degrees of centre. The
# eyes do not always rest there: two baseline starts on ckpt_step_344400 had them at -43.6 and
# -44.2 degrees of elevation, where the 8 degree "down" command is clipped to 3.3 and 2.8 degrees.
# Every arm records the command it asked for and the one it sent, and clipped arms are counted.
OFFSETS = {"left": (10.0, 0.0), "right": (-10.0, 0.0), "up": (0.0, 8.0), "down": (0.0, -8.0)}
HEAD_TURN_DEG = 10.0          # generalization: a nearby head pose, well inside the neck swivel range (+/-111)
AMPLITUDES = (5.0, 10.0, 20.0)
# THE PAIR NOISE LINE (review of the probes, 2026-09-23). The eye starts the same moment one
# degree away from where it was (azimuth, elevation), held there for one model step: an eighth of
# the smallest command offset (8 degrees), so every command still points the same way from each
# start. Three starts plus the main one give four readings of a pair's predicted difference; the
# line is twice their spread (the two-standard-error rule the wellness and contact gates use).
PAIR_NOISE_STARTS = ((1.0, 0.0), (0.0, 1.0), (-1.0, -1.0))
PAIR_NOISE_SPREADS = 2.0


# ============================================================================ physics only
def step_response(frozen: H.Frozen, *, amplitudes=AMPLITUDES, seconds: float = 0.40,
                  settle_s: float = 0.5, seed: int = 7) -> dict:
    """The eye servo's response to a gaze jump, sampled every physics substep (5 ms)."""
    from mimo_bridge.datasource import SingleEnvSource
    from mimo_bridge.tools.motor_reachability_probe import pose
    kw = H.world_kwargs(frozen, seed=seed, vision=False)
    src = SingleEnvSource(**kw)
    e = src.env
    try:
        src.reset()
        strength = float(frozen.limits.get("strength_limit", 1.0))
        body = pose(e, strength)
        for _ in range(int(round(settle_s / e.dt))):
            src.step(body, np.zeros((1, 2)), None, render=False)
        start = src.save_full_state()
        joints = {"az": "robot:left_eye_horizontal", "el": "robot:left_eye_vertical"}
        results = {}
        rows = []
        orig = getattr(e.env, "_bridge_contact_observer", None)

        def observer(before):
            if orig is not None:
                orig(before)
            rows.append((float(e.data.time), float(e.data.qpos[qadr]), float(e.data.qvel[dadr])))

        for axis, jname in joints.items():
            jid = e.model.joint(jname).id
            qadr, dadr = int(e.model.jnt_qposadr[jid]), int(e.model.jnt_dofadr[jid])
            for amp in amplitudes:
                src.restore_full_state(start)
                q0 = float(e.data.qpos[qadr])
                target = [0.0, 0.0]
                target[0 if axis == "az" else 1] = math.radians(amp)
                # the eye's rest angle is where it settled under a (0, 0) command; the jump is
                # measured from there to the commanded angle
                rows.clear()
                t0 = float(e.data.time)
                e.env._bridge_contact_observer = observer
                try:
                    for _ in range(int(round(seconds / e.dt))):
                        src.step(body, np.asarray(target).reshape(1, 2), None, render=False)
                finally:
                    e.env._bridge_contact_observer = orig
                t = np.array([r[0] for r in rows]) - t0
                q = np.degrees(np.array([r[1] for r in rows]))
                qd = np.degrees(np.array([r[2] for r in rows]))
                goal = math.degrees(target[0 if axis == "az" else 1])
                start_deg = math.degrees(q0)
                span = goal - start_deg
                frac = (q - start_deg) / span if abs(span) > 1e-9 else np.zeros_like(q)
                err = np.abs(goal - q)

                def first(mask):
                    idx = np.nonzero(mask)[0]
                    return round(float(t[idx[0]] * 1000.0), 1) if len(idx) else None

                def settle(tol):
                    outside = np.nonzero(err > tol)[0]
                    if not len(outside):
                        return 0.0
                    if outside[-1] == len(t) - 1:
                        return None                  # never settled inside the window
                    return round(float(t[outside[-1] + 1] * 1000.0), 1)

                at = {}
                for ms in (50, 100, 150):
                    k = int(np.argmin(np.abs(t * 1000.0 - ms)))
                    at[f"{ms}ms"] = float(frac[k])
                results[f"{axis}_{int(amp)}"] = dict(
                    commanded_deg=goal, start_deg=start_deg, jump_deg=span,
                    rise_to_90pct_ms=first(frac >= 0.9),
                    settle_within_5pct_ms=settle(0.05 * abs(span)),
                    settle_within_2pct_ms=settle(0.02 * abs(span)),
                    peak_speed_deg_s=float(np.max(np.abs(qd))),
                    overshoot_pct=float(max(0.0, np.max((q - goal) * np.sign(span))) / abs(span) * 100.0)
                    if abs(span) > 1e-9 else 0.0,
                    final_error_deg=float(err[-1]), share_of_jump_done=at,
                    samples=len(t), sample_ms=float(np.median(np.diff(t)) * 1000.0) if len(t) > 1 else None)
        peaks = [r["peak_speed_deg_s"] for r in results.values()]
        settles = [r["settle_within_5pct_ms"] for r in results.values() if r["settle_within_5pct_ms"] is not None]
        summary = dict(
            gaze_kp=float(getattr(frozen.args, "gaze_kp", 80.0)), gaze_kd=float(getattr(frozen.args, "gaze_kd", 3.0)),
            peak_speed_deg_s_max=float(max(peaks)) if peaks else None,
            settle_5pct_ms_max=float(max(settles)) if settles else None,
            claims_checked={"env.py: 110 deg/s peak, 130 ms": None, "gaze_reflex.py: reaches 70.8 deg/s": None},
            horizon_100ms_covers_every_jump=bool(settles) and len(settles) == len(results) and max(settles) <= 100.0,
            horizon_150ms_covers_every_jump=bool(settles) and len(settles) == len(results) and max(settles) <= 150.0)
        summary["claims_checked"] = {
            "env.py: 110 deg/s peak, 130 ms": "measured peak %.1f deg/s, slowest settle %s ms"
            % (summary["peak_speed_deg_s_max"] or float("nan"),
               "%.0f" % summary["settle_5pct_ms_max"] if summary["settle_5pct_ms_max"] is not None else "not within the window"),
            "gaze_reflex.py: reaches 70.8 deg/s": "measured peak %.1f deg/s" % (summary["peak_speed_deg_s_max"] or float("nan"))}
        return dict(per_jump=results, summary=summary, sampled_every="physics substep (%.1f ms)"
                    % (float(e.model.opt.timestep) * 1000.0),
                    body="held in its resting pose at the strength limit; reflex and saccades off")
    finally:
        src.close()


# ============================================================================ with the model
def _eye_targets(eyes0_rad):
    """{name: (requested, sent)} in radians: the eyes' start plus each offset, and the same
    clipped to the eye's joint range (the command actually sent)."""
    from mimo_bridge.gaze_reflex import AZ_RANGE, EL_RANGE
    out = {"hold": (eyes0_rad.copy(), eyes0_rad.copy())}
    for name, (daz, del_) in OFFSETS.items():
        req = eyes0_rad + np.radians([daz, del_])
        g = req.copy()
        g[0] = float(np.clip(g[0], *AZ_RANGE))
        g[1] = float(np.clip(g[1], *EL_RANGE))
        out[name] = (req, g)
    return out


def _arms_from(session: H.Session, snap: dict, depth: int, senses, *, names=None) -> dict:
    """Predicted and actual outcomes of every gaze candidate from one restored moment."""
    model = session.model
    session.restore(snap)
    mom = session.moment()
    eyes0 = session.eyes_rad()
    body = session.hold_order()
    body, hand_goal = session._canonical_hand_order(body)
    targets = _eye_targets(eyes0)
    if names is not None:
        targets = {k: v for k, v in targets.items() if k in names}
    arms = {}
    pathway = H.command_pathway(model, mom, {n: session.exec_order(body, g, hand_joint_goal=hand_goal)
                                            for n, (_r, g) in targets.items()})
    for name, (req, g) in targets.items():
        session.restore(snap)
        order_exec = session.exec_order(body, g, hand_joint_goal=hand_goal)
        # the "no gaze command" ablation: the same order with the eyes told to stay put
        order_ablated = session.exec_order(body, eyes0, hand_joint_goal=hand_goal)
        pred = H.predict(model, mom, order_exec)                     # before the order is sent
        roll = H.rollout(model, mom, order_exec, depth)
        roll_ab = H.rollout(model, mom, order_ablated, depth)
        pred_ab = H.predict(model, mom, order_ablated, want_eff=False)
        t_pred = time.time()
        actual = session.run_arm(snap, body, gaze_rad=g, steps=depth, hand_joint_goal=hand_goal)
        sent = np.degrees(np.asarray(session.last_gaze_sent, dtype=np.float64)).tolist()
        clipped = bool(np.max(np.abs(np.degrees(req - g))) > 1e-6)
        arms[name] = dict(target_deg=np.degrees(g).tolist(), requested_deg=np.degrees(req).tolist(),
                          sent_deg=sent, clipped=clipped,
                          clipped_by_deg=np.degrees(req - g).tolist() if clipped else None,
                          pred=pred, roll=roll, roll_ablated=roll_ab,
                          pred_ablated=pred_ab, actual=actual, predicted_before_sent=True,
                          predicted_at=t_pred)
    return dict(mom=mom, eyes0=eyes0, arms=arms, pathway=pathway)


NOISE_SENSES = ("vision", "proprioception")


def _pair_differences(session: H.Session, mom, eyes0, body, depth: int) -> dict:
    """For every pair of offset commands from this moment and eye start: the difference between
    their predicted tokens, on the trained route at 50 ms and the imagination route at every
    depth. {(route_key, sense): {(x, y): tensor}}."""
    model = session.model
    preds, rolls = {}, {}
    for name, (_req, g) in _eye_targets(eyes0).items():
        if name == "hold":
            continue
        o = session.exec_order(body, g)
        preds[name] = H.predict(model, mom, o, want_eff=False)["preds"]
        rolls[name] = H.rollout(model, mom, o, depth)
    out = {}
    for x, y in itertools.combinations([n for n in OFFSETS if n in preds], 2):
        for m in NOISE_SENSES:
            if preds[x].get(m) is not None and preds[y].get(m) is not None:
                out.setdefault(("50ms_trained", m), {})[(x, y)] = (preds[x][m] - preds[y][m]).reshape(-1).double()
            for k in range(depth):
                a, b = rolls[x][k].get(m), rolls[y][k].get(m)
                if a is not None and b is not None:
                    out.setdefault((f"{50 * (k + 1)}ms", m), {})[(x, y)] = (a - b).reshape(-1).double()
    return out


def _pair_noise(session: H.Session, snap: dict, depth: int) -> dict:
    """The noise line of every pair's predicted difference (PAIR_NOISE_STARTS): the main
    moment's difference and the same pair's difference from each nearby eye start.
    {route_key: {sense: {"x|y": {main_norm, spread, line, decided, readings}}}}."""
    session.restore(snap)
    mom = session.moment()
    eyes0 = session.eyes_rad()
    body = session.hold_order()
    readings = [_pair_differences(session, mom, eyes0, body, depth)]
    for daz, del_ in PAIR_NOISE_STARTS:
        session.restore(snap)
        start = eyes0 + np.radians([daz, del_])
        session.act(session.hold_order(), gaze_rad=start, new_order=True)
        session.observe()
        m2 = session.moment()
        readings.append(_pair_differences(session, m2, session.eyes_rad(), session.hold_order(), depth))
    out = {}
    for (key, m), pairs in readings[0].items():
        for pair, d0 in pairs.items():
            ds = [r.get((key, m), {}).get(pair) for r in readings]
            ds = [d for d in ds if d is not None]
            if len(ds) < 2:
                continue
            st = torch.stack(ds)
            mean = st.mean(dim=0)
            # the spread of the difference VECTOR over the readings (root of the summed squared
            # deviations over n - 1), against the length of the main moment's difference
            spread = float(((st - mean) ** 2).sum(dim=-1).sum().div(len(ds) - 1).sqrt())
            main = float(d0.norm())
            line = PAIR_NOISE_SPREADS * spread
            out.setdefault(key, {}).setdefault(m, {})["|".join(pair)] = dict(
                main_norm=main, spread=spread, line=line, decided=bool(main > line), readings=len(ds))
    session.restore(snap)
    return out


def _score_state(res: dict, senses, depth: int) -> dict:
    """Numbers for one moment: eye movement, prediction error, sensitivity, direction,
    discrimination, ablation, efference."""
    mom, arms = res["mom"], res["arms"]
    eyes0 = np.degrees(res["eyes0"])
    out = dict(eyes0_deg=eyes0.tolist(), arms={}, command_pathway=res.get("pathway"))
    for name, a in arms.items():
        row = dict(target_deg=a["target_deg"], requested_deg=a.get("requested_deg"), sent_deg=a.get("sent_deg"),
                   clipped=a.get("clipped", False), clipped_by_deg=a.get("clipped_by_deg"),
                   eye_change_deg=[(r["eyes_deg"] - eyes0).tolist() for r in a["actual"]],
                   lesson_50ms=H.token_scores(a["pred"]["preds"], a["actual"][0]["tokens"], mom.tokens, senses),
                   lesson_50ms_ablated=H.token_scores(a["pred_ablated"]["preds"], a["actual"][0]["tokens"],
                                                      mom.tokens, senses))
        row["imagined"] = {}
        for k in range(depth):
            row["imagined"][f"{50 * (k + 1)}ms"] = H.token_scores(a["roll"][k], a["actual"][k]["tokens"],
                                                                 mom.tokens, senses)
            row["imagined"][f"{50 * (k + 1)}ms_ablated"] = H.token_scores(
                a["roll_ablated"][k], a["actual"][k]["tokens"], mom.tokens, senses)
        if "eff" in a["pred"] and "gaze_deg" in a["pred"]["eff"]:
            row["efference_gaze_deg"] = a["pred"]["eff"]["gaze_deg"].tolist()
        out["arms"][name] = row
    # sensitivity and direction of each command against holding the eyes, per horizon and route:
    # "50ms_trained" = the trained route (pred["preds"], the prediction the next-frame lesson
    # grades); "50ms", "100ms", "150ms" = the imagination route (predictive_state.transition)
    offs = [n for n in arms if n != "hold"]
    routes = [("50ms_trained", lambda a, k: a["pred"]["preds"], 0)]
    routes += [(f"{50 * (k + 1)}ms", (lambda a, k: a["roll"][k]), k) for k in range(depth)]
    sens = {}
    for key, get, k in routes:
        per = {}
        for n in offs:
            for m in ("vision", "proprioception"):
                if m not in senses:
                    continue
                pn, ph = get(arms[n], k).get(m), get(arms["hold"], k).get(m)
                if pn is None or ph is None:
                    continue
                pd = pn - ph
                ad = arms[n]["actual"][k]["tokens"][m] - arms["hold"]["actual"][k]["tokens"][m]
                pdv, adv = pd.reshape(-1).double(), ad.reshape(-1).double()
                per.setdefault(m, []).append(dict(
                    arm=n, predicted_difference=H.l2(pd), actual_difference=H.l2(ad),
                    direction=H.cosine(pd, ad), captured_share=H.captured_share(pd, ad),
                    dot_pa=float(pdv @ adv), dot_aa=float(adv @ adv), dot_pp=float(pdv @ pdv)))
        sens[key] = per
    out["sensitivity_vs_hold"] = sens
    # two-way discrimination between every pair of different commands, counted only where the
    # pair's predicted difference stands above its measured noise line (_pair_noise).
    # "_ties" counts pairs whose two predictions are EXACTLY equal; every other pair under its
    # noise line, or above it but not resolved by pairing_correct (float32 cannot tell the two
    # assignments apart, or the two real outcomes were the same), is "_undecided". Corrected
    # 2026-09-23 (verify-fix-probes): the float tie test ran first and its count was reported as
    # "gave identical predictions", while on ckpt_step_344400 21 of those 24 pairs' predictions
    # differed (by 0.4 to 1.9 millionths of a token length) and only 3 were exactly equal.
    noise = res.get("noise") or {}
    disc = {}
    for key, get, k in routes:
        for m in ("vision", "proprioception"):
            if m not in senses:
                continue
            ok, undecided, ties = [], 0, 0
            for x, y in itertools.combinations(offs, 2):
                px, py = get(arms[x], k).get(m), get(arms[y], k).get(m)
                if px is None or py is None:
                    continue
                line = ((noise.get(key) or {}).get(m) or {}).get(f"{x}|{y}")
                if torch.equal(torch.as_tensor(px).reshape(-1), torch.as_tensor(py).reshape(-1)):
                    ties += 1                # the two commands gave exactly the same prediction
                    continue
                if not (line and line.get("decided")):
                    undecided += 1           # no noise line, or the difference sits under it
                    continue
                v = H.pairing_correct(px, py, arms[x]["actual"][k]["tokens"][m], arms[y]["actual"][k]["tokens"][m])
                if v is None:
                    undecided += 1           # above the line, but the pairing cannot be resolved
                else:
                    ok.append(v)
            disc.setdefault(key, {})[m] = ok
            disc.setdefault(key + "_ties", {})[m] = ties
            disc.setdefault(key + "_undecided", {})[m] = undecided
    out["discrimination"] = disc
    out["pair_noise"] = noise
    return out


def counterfactual(frozen: H.Frozen, *, states: int = 4, warm: int = 300, spacing: int = 10,
                   depth: int = 3, seed: int = 11, repeats: int = 3) -> dict:
    """The model-side gaze measurements from `states` start states (bodies of the checkpoint's
    own episode when it has one), reflex and spontaneous saccades off so only the probe moves
    the eyes.

    Each start is a moment where the planner made a plan, which only the wellness-selection
    readout uses (the planner's own gaze against the better-scored candidate). A body whose
    planner makes no plan in its search window starts at its first planning event without one
    (harness plan_fallback, START_WITHOUT_A_PLAN): measured on checkpoint 400,350, body 2's goal
    was its present moment at every planning event, so its planner never made a plan and the
    body used to be dropped without a word. Every state records how its start was chosen
    ('start_planner')."""
    senses = H.real_senses(frozen.model)
    per_state, gen_states, wsel = [], [], []
    instrument, cfg = None, None
    for si, (session, snap) in enumerate(H.start_states(frozen, states, want_plan=True, warm=warm,
                                                         spacing=spacing, seed=seed, plan_fallback=True,
                                                         gaze_reflex=False, saccades=False)):
        if cfg is None:
            cfg = H.frozen_config(frozen, session)
        res = _arms_from(session, snap, depth, senses)
        res["noise"] = _pair_noise(session, snap, depth)
        scored = _score_state(res, senses, depth)
        scored["start"] = session.start
        scored["body"] = (getattr(session, "episode_body", None) or {}).get("body")
        scored["step_offset"] = snap.get("step_offset")
        scored["start_planner"] = snap.get("planner")
        per_state.append(scored)
        # wellness selection: two imagined candidates, and the planner's own direction
        wsel.append(_wellness_selection(session, snap, res))
        # instrument repeatability, first moment only
        if si == 0 and repeats > 1:
            instrument = _instrument_repeat(session, snap, res, repeats)
        # generalization: the head turned first
        gen_states.append(_head_turned(session, snap, depth, senses))
        del res
    return dict(senses=senses, depth=depth, states=per_state, wellness_selection=wsel,
                instrument_repeat=instrument, generalization=gen_states, config=cfg or {})


def _instrument_repeat(session, snap, res, repeats):
    body_eyes = res["arms"]["left"]["target_deg"]
    g = np.radians(body_eyes)
    runs = []
    for _ in range(int(repeats)):
        session.restore(snap)
        body = session.hold_order()
        body, hand_goal = session._canonical_hand_order(body)
        mom = session.moment()
        pred = H.predict(session.model, mom, session.exec_order(body, g, hand_joint_goal=hand_goal), want_eff=False)
        act = session.run_arm(snap, body, gaze_rad=g, steps=1, hand_joint_goal=hand_goal)[0]
        runs.append((pred["preds"], act))
    eye_spread = max(float(np.max(np.abs(r[1]["eyes_deg"] - runs[0][1]["eyes_deg"]))) for r in runs)
    tok_spread = max(max(H.l2(r[1]["tokens"][m] - runs[0][1]["tokens"][m]) for m in runs[0][1]["tokens"]) for r in runs)
    pred_spread = max(max(H.l2(r[0][m] - runs[0][0][m]) for m in runs[0][0]) for r in runs)
    return dict(repeats=int(repeats), eye_max_difference_deg=eye_spread, token_max_difference=tok_spread,
                prediction_max_difference=pred_spread,
                exact=bool(eye_spread == 0.0 and tok_spread == 0.0 and pred_spread == 0.0))


def _head_turned(session, snap, depth, senses):
    """Turn the head HEAD_TURN_DEG about the neck swivel axis (the eyes told to stay in the head),
    let one order tenure pass, then run left/right/hold from there."""
    session.restore(snap)
    axes = session.joint_axes("head")
    swivel = [a for a in axes if a["joint"].endswith("head_swivel")]
    if not swivel:
        return dict(status="unmeasured: no head swivel joint in the decoder's head chain")
    base = session.hold_order()
    turned = session.rotated(base, "head", swivel[0]["axis"], HEAD_TURN_DEG)
    eyes = session.eyes_rad()
    hi = session.handle_names.index("head")
    q_start = session.pose()[1][hi]
    sw0 = _joint_deg(session, "robot:head_swivel")
    for k in range(session.hold):
        session.act(turned, gaze_rad=eyes, new_order=(k == 0))
        session.observe()
    # what the head really did before the commands (review of the probes, 2026-09-23: the order
    # was recorded as if it were the head; a 15 degree head order achieved 0.6 to 4.3 degrees)
    achieved = dict(head_turn_deg=H.rotation_deg_xyzw(q_start, session.pose()[1][hi]),
                    swivel_change_deg=(None if sw0 is None else _joint_deg(session, "robot:head_swivel") - sw0))
    snap2 = session.snapshot()
    res = _arms_from(session, snap2, depth, senses, names=("hold", "left", "right", "up", "down"))
    res["noise"] = _pair_noise(session, snap2, depth)
    scored = _score_state(res, senses, depth)
    del snap2
    return dict(head_turn_ordered_deg=HEAD_TURN_DEG, head_turn_achieved=achieved, scored=scored)


def _joint_deg(session, joint: str):
    """A hinge joint's angle in degrees, or None when the body has no such joint."""
    e = session.env
    try:
        jid = e.model.joint(joint).id
    except Exception:
        return None
    return math.degrees(float(e.data.qpos[int(e.model.jnt_qposadr[jid])]))


def _wellness_selection(session, snap, res):
    model = session.model
    mom, eyes0 = res["mom"], res["eyes0"]
    session.restore(snap)
    body = session.hold_order()
    scores = {}
    for name in ("left", "right", "up", "down"):
        g = np.radians(res["arms"][name]["target_deg"])
        scores[name] = H.predicted_wellness(model, mom, session.exec_order(body, g))
    out = dict(predicted_wellness=scores)
    if any(v is None for v in scores.values()):
        out["status"] = "unmeasured: no approximate wellness evaluator"
        return out
    best_h = H.better_of("left", scores["left"], "right", scores["right"])
    best_v = H.better_of("up", scores["up"], "down", scores["down"])
    out.update(better_horizontal=best_h, better_vertical=best_v,
               margin_horizontal=abs(scores["left"] - scores["right"]),
               margin_vertical=abs(scores["up"] - scores["down"]))
    plan0 = snap.get("plan0")
    if plan0 is None:
        out["planner"] = "no plan at this moment (planner off in this build, or no plan made)"
        decision = snap.get("planner") or {}
        if decision.get("consulted"):
            # the start's own record (harness planner_decision): why the planner gave no order
            out["planner"] = ("no plan at this moment: the planner stopped as %s, its goal %s from the present "
                              "moment (under 0.01 it stops without an order)"
                              % (decision.get("stopped_by"), "unmeasured" if decision.get("goal_distance_now") is None
                                 else "%.4g" % decision["goal_distance_now"]))
        return out
    g_plan = np.asarray(model._gaze_from_handle(plan0.to(model.device))).reshape(-1)[:2]
    d = np.degrees(g_plan - eyes0)
    out.update(planner_gaze_change_deg=d.tolist(),
               planner_horizontal=("left" if d[0] > 0 else "right") if abs(d[0]) > 1e-6 else "none",
               planner_vertical=("up" if d[1] > 0 else "down") if abs(d[1]) > 1e-6 else "none")
    # a tie (the evaluator scores both candidates the same) or no planner movement is not a choice
    out["agrees_horizontal"] = (None if out["planner_horizontal"] == "none" or best_h == "tie"
                                else out["planner_horizontal"] == best_h)
    out["agrees_vertical"] = (None if out["planner_vertical"] == "none" or best_v == "tie"
                              else out["planner_vertical"] == best_v)
    return out


def eff_present(states) -> bool:
    return any(a.get("efference_gaze_deg") is not None for s in states for a in s["arms"].values())


def summarize(sr: dict, cf: dict, frozen: H.Frozen) -> dict:
    senses = cf["senses"]
    states = cf["states"]

    def disc(states_, key, m):
        vals = [v for s in states_ for v in s["discrimination"].get(key, {}).get(m, [])]
        count = lambda suffix: int(sum(s["discrimination"].get(key + suffix, {}).get(m, 0) for s in states_))
        ties_, undecided = count("_ties"), count("_undecided")
        return dict(value=(float(np.mean(vals)) if vals else None), decided_pairs=len(vals),
                    correct=int(sum(1 for v in vals if v)), tied_pairs=ties_, undecided_pairs=undecided,
                    pairs=len(vals) + ties_ + undecided,
                    rule="a pair counts only when its predicted difference exceeds its noise line (twice the "
                         "spread of that difference over the main moment and %d nearby eye starts); "
                         "tied_pairs = the two predictions exactly equal; undecided_pairs = every other "
                         "pair not counted" % len(PAIR_NOISE_STARTS))

    findings = {}
    keys = ("50ms_trained", "50ms", "100ms", "150ms")
    for key in keys:
        for m in ("vision", "proprioception"):
            findings[f"discrimination_{m}_{key}"] = disc(states, key, m)
    # direction and size of the predicted change against the actual, per route and horizon; the
    # size is pooled over arms (root of summed squared predicted differences over summed squared
    # real ones), never a mean of per-arm ratios, and each difference is taken against holding
    # the eyes still from the same moment
    for key in keys:
        for m in ("vision", "proprioception"):
            rows = [e for s in states for e in s["sensitivity_vs_hold"].get(key, {}).get(m, [])]
            findings[f"{m}_captured_share_{key}"] = H.pooled_share(rows, "dot_pa", "dot_aa")
            findings[f"{m}_direction_{key}"] = H.mean_or_none([e["direction"] for e in rows])
            aa = sum(e["dot_aa"] for e in rows if e.get("dot_aa") is not None)
            pp = sum(e["dot_pp"] for e in rows if e.get("dot_pp") is not None)
            findings[f"{m}_size_against_hold_{key}"] = (math.sqrt(pp / aa) if aa > 0 else None)
    clipped = [dict(state=i, arm=n, requested_deg=a.get("requested_deg"), target_deg=a.get("target_deg"),
                    sent_deg=a.get("sent_deg"))
               for i, s in enumerate(states) for n, a in s["arms"].items() if a.get("clipped")]
    findings["clipped_commands"] = dict(
        n=len(clipped), arms=clipped, of=sum(len(s["arms"]) - 1 for s in states),
        note="a command beyond the eye's joint range is clipped to the range before it is sent; the arm is "
             "scored with the command it sent")
    # ablation: how much of the prediction error the gaze command removes
    abl = {}
    for key in ("50ms", "150ms"):
        e_real, e_ab = [], []
        for s in states:
            for n, a in s["arms"].items():
                if n == "hold":
                    continue
                sc = a["imagined"].get(key, {}).get("vision")
                sa = a["imagined"].get(key + "_ablated", {}).get("vision")
                if sc and sa:
                    e_real.append(sc["pred_err"])
                    e_ab.append(sa["pred_err"])
        abl[key] = dict(error_with_gaze=H.mean_or_none(e_real), error_without_gaze=H.mean_or_none(e_ab),
                        share_of_error_removed=(1.0 - np.mean(e_real) / np.mean(e_ab)) if e_ab and np.mean(e_ab) > 0 else None)
    findings["ablation_vision"] = abl
    # eye movement actually produced (degrees) at 150 ms by the 10-degree commands
    moved = [abs(a["eye_change_deg"][-1][0]) for s in states for n, a in s["arms"].items() if n in ("left", "right")]
    findings["eye_moved_deg_150ms_for_10deg_command"] = dict(mean=H.mean_or_none(moved),
                                                             spread=float(np.std(moved)) if moved else None)
    # generalization
    gen = [g["scored"] for g in cf["generalization"] if "scored" in g]
    findings["generalization_head_turned_discrimination_vision_150ms"] = disc(gen, "150ms", "vision")
    findings["generalization_head_turned_captured_share_vision_150ms"] = H.pooled_share(
        [e for g in gen for e in g["sensitivity_vs_hold"].get("150ms", {}).get("vision", [])], "dot_pa", "dot_aa")
    findings["generalization_head_turn_achieved"] = [g.get("head_turn_achieved") for g in cf["generalization"]]
    # wellness selection
    ws = cf["wellness_selection"]
    agree = [w.get("agrees_horizontal") for w in ws if w.get("agrees_horizontal") is not None]
    agree += [w.get("agrees_vertical") for w in ws if w.get("agrees_vertical") is not None]
    ties = sum(1 for w in ws for k in ("better_horizontal", "better_vertical") if w.get(k) == "tie")
    findings["planner_toward_better_predicted_wellness"] = dict(
        share=(float(np.mean(agree)) if agree else None), n=len(agree), candidate_pairs_tied=ties,
        candidate_pairs=2 * len(ws),
        # the weight is written as the launch setting it is ("--prospective-wellness-w 0.15"), never
        # as a bare score decimal (DESIGN section 11; fix-readout-explainer 2026-09-23)
        planner_objective=("distance to a remembered goal; predicted wellness enters its choice only "
                           "through --prospective-wellness (%s in this run, --prospective-wellness-w %s)"
                           % (getattr(frozen.args, "prospective_wellness", None),
                              getattr(frozen.args, "prospective_wellness_w", None))))
    findings["instrument_repeat"] = cf.get("instrument_repeat")
    findings["prediction_before_action"] = dict(
        every_prediction_made_before_the_order_was_sent=True,
        how=("each prediction is computed from the restored moment, then the same moment is restored "
             "and the order sent; the arm records hold both"),
        eye_pose_prediction=("efference gaze head, degrees at 50/100/150 ms" if eff_present(states)
                             else "no readable eye-pose output in this build (efference heads absent); the eye's "
                                  "pose sits inside the proprioception token, scored as proprioception above"))
    gz = [v for s in states for k, v in ((s.get("command_pathway") or {}).get("token_change_share") or {}).items()
          if v is not None]
    findings["gaze_command_moves_motor_token_share"] = H.mean_or_none(gz)
    findings["step_response"] = sr["summary"]
    # efference gaze head, when present
    eff = [(a.get("efference_gaze_deg"), a["eye_change_deg"]) for s in states for a in s["arms"].values()
           if a.get("efference_gaze_deg") is not None]
    if eff:
        errs = []
        for pred, act in eff:
            for k in range(min(3, len(act))):
                errs.append(float(np.linalg.norm(np.asarray(pred[k]) - np.asarray(act[k]))))
        findings["efference_gaze_error_deg"] = H.mean_or_none(errs)
    else:
        findings["efference_gaze_error_deg"] = H.absent("efference")
    return findings


def lived_sentence(findings: dict) -> str:
    """The operator's sentence (DESIGN section 11: percentages between named ends, counts)."""
    value = findings.get("vision_captured_share_150ms")
    v50 = findings.get("vision_captured_share_50ms_trained")
    size = findings.get("vision_size_against_hold_150ms")
    d = findings.get("discrimination_vision_150ms") or {}
    clipped = (findings.get("clipped_commands") or {}).get("n", 0)
    sent = (findings.get("clipped_commands") or {}).get("of", 0)
    s = ("When only the eye command changed (10 degrees left or right, 8 up or down, each against holding "
         "the eyes still from the same moment), the picture the model imagined 150 ms ahead, on the "
         "imagination route the planner uses, moved %s of the way the real picture moved; on the trained "
         "route (the prediction its next-moment lesson grades), 50 ms ahead, %s. 0%% = the prediction "
         "ignores where the eyes are sent, 100%% = it foresees the whole change. The imagined picture for a "
         "command differed from the imagined picture for holding the eyes still by %s of how much the real "
         "pictures differed. Of %d pairs of different commands, %d stood above their noise line (twice the "
         "spread of the pair's predicted difference over %d nearby eye starts) and %d of those were told "
         "apart correctly; %d were undecided (their predictions differed by less than that line) and %d gave "
         "exactly the same prediction for both commands. Distance to goal unmeasured."
         % (H.share_text(value), H.share_text(v50), H.pct(size), int(d.get("pairs") or 0),
            int(d.get("decided_pairs") or 0), len(PAIR_NOISE_STARTS), int(d.get("correct") or 0),
            int(d.get("undecided_pairs") or 0), int(d.get("tied_pairs") or 0)))
    if clipped:
        s += (" %d of the %d commands %s clipped to the eye's range before %s sent (the eyes started near the "
              "edge of it); each is scored with the command it sent."
              % (clipped, sent, "was" if clipped == 1 else "were", "it was" if clipped == 1 else "they were"))
    return s


def run(ckpt, *, states: int = 4, warm: int = 300, seed: int = 11, write: bool = True, root=None) -> dict:
    t0 = time.time()
    H.reset_routes()
    frozen = H.load_frozen(ckpt)
    sr = step_response(frozen)
    cf = counterfactual(frozen, states=states, warm=warm, seed=seed)
    findings = summarize(sr, cf, frozen)
    value = findings["vision_captured_share_150ms"]
    lived = lived_sentence(findings)
    body = dict(config=dict(cf["config"], states=states, warm=warm, seed=seed, offsets_deg=OFFSETS,
                            head_turn_deg=HEAD_TURN_DEG, amplitudes_deg=AMPLITUDES,
                            pair_noise_starts_deg=PAIR_NOISE_STARTS, pair_noise_spreads=PAIR_NOISE_SPREADS,
                            route=H.route_record()),
                arms=dict(step_response=sr, counterfactual_states=cf["states"],
                          generalization=cf["generalization"], wellness_selection=cf["wellness_selection"]),
                controls=dict(persistence="each sense's 'nothing changes' error is in every arm's scores",
                              hold="every command is compared with holding the eyes still from the same moment",
                              ablation=findings["ablation_vision"],
                              instrument_repeat=cf.get("instrument_repeat")),
                findings=findings,
                headline=H.headline("gaze_captured_share_vision_150ms", value,
                                    "share of the real change in what the eyes saw, 150 ms after an eye "
                                    "command, that the imagined picture foresaw: from none (ignored the "
                                    "command) to all (foresaw the whole change)", 0.0, None, lived),
                status="measured" if value is not None else "partial: no gaze command changed the picture",
                elapsed_s=round(time.time() - t0, 1))
    if write:
        body["path"] = str(H.write_result(frozen.info, PROBE, ITEM, body, root=root))
    return body


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--states", type=int, default=4)
    ap.add_argument("--warm", type=int, default=300, help="fresh-world warm-up steps, used only when the checkpoint has no episode companion")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(argv)
    out = run(a.ckpt, states=a.states, warm=a.warm, seed=a.seed)
    print(json.dumps(dict(headline=out["headline"], path=out.get("path"), elapsed_s=out["elapsed_s"]), indent=1))


if __name__ == "__main__":
    main()
