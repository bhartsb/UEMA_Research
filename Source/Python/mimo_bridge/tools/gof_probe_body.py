# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Body-control probes (addendum v2 section 7, build item 8).

    python -m mimo_bridge.tools.gof_probe_body --ckpt <checkpoint folder> [--states 3]

From identical restored moments, ONE limb's order changes and everything else holds its pose:
finger flexion (right index knuckle), wrist (right hand1), elbow (right), shoulder (right
horizontal) and leg (right knee), each turned 20 degrees two ways. Each order is held for one
order tenure (motor_hold model steps, 250 ms on H84), the way the trainer latches an order.

What the model expected is read BEFORE the order is sent:
  - the tenure pose head (tenure_heads['pose']): each handle's predicted 250 ms change of place,
    in cm (its output times the tokenizer's 200 cm), the one readable body prediction the model
    has had since 2026-09-01;
  - the efference body head, when the build has it: place in cm and turn in degrees, next moment
    and over the tenure;
  - the proprioception token 50 ms ahead (the trained next-moment route).
What happened is read from MuJoCo: every handle's place (cm) and turn (degrees).

Controls: holding still from the same moment; a random order (each handle turned about a random
axis by an angle drawn with the run's own tenure-long turn size); "nothing changes"
(persistence); the tenure head's own do-nothing order; and the model's earliest loadable
ancestor checkpoint (every ancestor on disk tried, oldest first, until one loads with nothing
reinitialised) from its own saved moment (or the reason none loads).
Also: repeatability (the same order three times from one moment: an instrument check; the same
order from different moments: the spread of what the body did), generalization (the shoulder
turned 10 degrees first - the turn it really made is recorded - then the same orders),
perturbation (a torque on the right forearm about the elbow's hinge axis, against the elbow's own
turn, sized by search until the loaded turn is at most half the unloaded one; "unmeasured: the
load did not change the motion" when the search cannot get there) and in-context adaptation (how
the model's 50 ms proprioception error changes over the five loaded steps), a predicted-wellness
choice (two limb orders scored by the approximate wellness evaluator from the same moment, and
which of the two the planner's own plan at that moment leaned toward), and the body milestone
ladder of section 7 (rungs 1-3: the order's effect and its prediction measured, repeatable
INTENDED movement unmeasured; the rest reported where they are measured or why they cannot be).

Headline: the share of the real 250 ms change the tenure head foresaw (the efference body head
when the build has it, with that head's hidden-unit live share beside it) - for each limb order,
the predicted change of every handle's place relative to holding still, projected on the real
change relative to holding still, over the real change's size (0 = the prediction ignores the
order, 1 = it foresees the whole change).
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
import json
import math
import time
from typing import Optional

import numpy as np
import torch

from mimo_bridge.tools import gof_probe_harness as H

PROBE, ITEM = "body", 8
# (name, handle, joint whose world axis the handle's target is turned about)
LIMBS = (("finger", "index_R", "robot:right_ff_knuckle"),
         ("wrist", "wrist_R", "robot:right_hand1"),
         ("elbow", "lowerarm_R", "robot:right_elbow"),
         ("shoulder", "upperarm_R", "robot:right_shoulder_horizontal"),
         ("leg", "calf_R", "robot:right_knee"))
# 20 degrees: about two of the run's own order-long turns. Measured on H84 segment 31's world
# (b1-causal-efference, lem/model/efference.py EFFERENCE_SCALES, 2026-09-22): a handle's turn over
# one order tenure has a root-mean-square of 11.0 degrees per axis and 99 in 100 stay under 42.2.
TURN_DEG = 20.0
RANDOM_TURN_RMS_DEG = 11.0         # the same measurement: the random order turns each handle this much
EDGE_MARGIN_DEG = 5.0              # a counter-turn must stay this far inside the joint range
PRE_TURN_DEG = 10.0                # generalization: the shoulder turned first by half an order
LOAD_BODY = "right_lower_arm"      # the body the load's torque is applied to
ELBOW_JOINT = "robot:right_elbow"
# THE LOAD LINE (review of the probes, 2026-09-23). The first version pushed the forearm down
# with its own and the hand's weight (2.49 N) and the elbow turned 11.82 degrees against 11.71
# unloaded - 0.9% of the turn, so the "adaptation" readings compared two runs of the same motion.
# The load is now sized by search until the loaded 250 ms turn is at most (1 - LOAD_TURN_SHARE)
# of the unloaded one. Half the turn: on ckpt_step_344400 the same 20 degree elbow order turned
# the elbow 10.9 degrees with a spread of 2.7 degrees over three start moments (baseline body
# probe, achieved_turn_deg), so a change of half the turn (5.4 degrees) is twice what the elbow
# varies from one moment to the next with no load; the run records its own spread beside it.
LOAD_TURN_SHARE = 0.5
LOAD_TORQUE_START_NM = 0.05        # the search doubles from here (11 doublings reach the cap);
LOAD_TORQUE_MAX_NM = 102.4         # the cap is where the search gives up and says so
LOAD_BISECTIONS = 6                # then halves the bracket 6 times: within 1/64 of it


def _plans(session: H.Session):
    """The two turns per limb: toward the middle of the joint's range, and back the other way
    when that stays inside the range (else half the first turn)."""
    out = {}
    for name, handle, joint in LIMBS:
        axes = [a for a in session.joint_axes(handle) if a["joint"] == joint]
        if not axes:
            out[name] = dict(status=f"unmeasured: {joint} is not in the decoder's chain for {handle}")
            continue
        a = axes[0]
        mid = 0.5 * (a["lo_deg"] + a["hi_deg"])
        first = TURN_DEG if a["q_deg"] <= mid else -TURN_DEG
        second = -first if (a["lo_deg"] + EDGE_MARGIN_DEG <= a["q_deg"] - first <= a["hi_deg"] - EDGE_MARGIN_DEG) else first / 2
        out[name] = dict(handle=handle, joint=joint, axis=a["axis"], q_deg=a["q_deg"], range_deg=[a["lo_deg"], a["hi_deg"]],
                         turns_deg=[first, second])
    return out


def _random_order(session: H.Session, base, rng):
    o = np.array(base, copy=True)
    for i, name in enumerate(session.handle_names):
        if name == "gaze":
            continue
        axis = rng.normal(size=3)
        o = session.rotated(o, name, axis, float(rng.normal(0.0, RANDOM_TURN_RMS_DEG)))
    return o


def _arm(session: H.Session, snap, order, mom, start, senses, *, before_step=None):
    """Prediction before, then the order held for one tenure; the physical change and scores."""
    model = session.model
    pred = H.predict(model, mom, order)
    rs = session.run_arm(snap, order, steps=session.hold, before_step=before_step)
    change = H.body_change(start, rs[-1])
    change1 = H.body_change(start, rs[0])
    return dict(pred=pred, readouts=rs, change=change, change_50ms=change1,
                lesson_50ms=H.token_scores(pred["preds"], rs[0]["tokens"], mom.tokens, senses))


def _state(session: H.Session, snap: dict, senses, rng, *, plans=None, limbs=None, with_random=True,
           before_step=None) -> dict:
    model = session.model
    session.restore(snap)
    mom = session.moment()
    start = session.readout()
    base = session.hold_order()
    plans = plans or _plans(session)
    null_pred = H.predict(model, mom, session.null_order(), want_eff=False)
    orders = {"hold": base}
    for name, plan in plans.items():
        if limbs is not None and name not in limbs:
            continue
        if "turns_deg" not in plan:
            continue
        for k, deg in enumerate(plan["turns_deg"]):
            orders[f"{name}_{'a' if k == 0 else 'b'}"] = session.rotated(base, plan["handle"], plan["axis"], deg)
    if with_random:
        orders["random"] = _random_order(session, base, rng)
    pathway = H.command_pathway(model, mom, orders)
    arms = {k: _arm(session, snap, o, mom, start, senses, before_step=before_step) for k, o in orders.items()}
    choice = _wellness_choice(session, snap, mom, plans, orders)
    return dict(mom=mom, start=start, plans=plans, null_pred=null_pred, arms=arms, pathway=pathway,
                wellness_choice=choice)


def _wellness_choice(session: H.Session, snap: dict, mom, plans: dict, orders: dict) -> dict:
    """The addendum's section 7 "predicted-wellness choice" for body motion: the two turns of each
    limb scored by the approximate wellness evaluator (the planner's scorer, on the planner's
    route) from the same moment, and which of the two the planner's own first action at that
    moment was nearer to (its target for that limb's handle, degrees from each order's target).
    A tie in predicted wellness, or no plan at this moment, is not a choice."""
    model = session.model
    plan0 = snap.get("plan0")
    out = {}
    for name, plan in plans.items():
        a, b = orders.get(f"{name}_a"), orders.get(f"{name}_b")
        if a is None or b is None:
            continue
        wa, wb = H.predicted_wellness(model, mom, a), H.predicted_wellness(model, mom, b)
        row = dict(predicted_wellness_a=wa, predicted_wellness_b=wb)
        if wa is None or wb is None:
            row["status"] = "unmeasured: no approximate wellness evaluator"
            out[name] = row
            continue
        best = H.better_of("a", wa, "b", wb)
        row["better"] = best
        if plan0 is None:
            row["planner"] = "no plan at this moment (planner off in this build, or no plan made)"
        else:
            hi = session.handle_names.index(plan["handle"])
            q_plan = np.asarray(plan0.reshape(-1, session.n_handles, 9)[0, hi, 3:7].cpu().numpy(), dtype=np.float64)
            da = H.rotation_deg_xyzw(q_plan, a[0, hi, 3:7])
            db = H.rotation_deg_xyzw(q_plan, b[0, hi, 3:7])
            toward = "a" if da < db else ("b" if db < da else "none")
            row.update(planner_deg_from_a=da, planner_deg_from_b=db, planner_toward=toward,
                       planner_turn_from_hold_deg=H.rotation_deg_xyzw(orders["hold"][0, hi, 3:7], q_plan),
                       agrees=(None if best == "tie" or toward == "none" else toward == best))
        out[name] = row
    return out


def _score(res: dict, handle_names) -> dict:
    """Per arm: what the commanded handle did, what the tenure head expected, the share of the
    real change foreseen (relative to holding still), and each control."""
    arms, plans = res["arms"], res["plans"]
    hold = arms["hold"]
    tenure_hold = hold["pred"].get("tenure_cm")
    null_t = res["null_pred"].get("tenure_cm")
    out = {}
    for key, a in arms.items():
        limb = key.rsplit("_", 1)[0] if key not in ("hold", "random") else key
        plan = plans.get(limb, {})
        hi = handle_names.index(plan["handle"]) if "handle" in plan else None
        row = dict(moved_cm_all_handles=float(np.linalg.norm(a["change"]["cm"])),
                   lesson_50ms=a["lesson_50ms"])
        if hi is not None:
            end, end_hold = a["readouts"][-1], hold["readouts"][-1]
            row.update(handle=plan["handle"], commanded_turn_deg=plan["turns_deg"][0 if key.endswith("_a") else 1],
                       achieved_turn_deg=float(a["change"]["deg"][hi]),
                       achieved_move_cm=float(a["change"]["cm_norm"][hi]),
                       # the order's own effect: where the part ended against where holding still left it
                       turn_vs_hold_deg=H.rotation_deg_xyzw(end["quat"][hi], end_hold["quat"][hi]),
                       move_vs_hold_cm=float(np.linalg.norm(np.asarray(end["pos_cm"][hi]) - np.asarray(end_hold["pos_cm"][hi]))))
        t = a["pred"].get("tenure_cm")
        if a["pred"].get("tenure_hidden"):
            row["tenure_hidden"] = a["pred"]["tenure_hidden"]
        if t is not None:
            actual = a["change"]["cm"]
            row["tenure_error_cm"] = float(np.linalg.norm(t - actual, axis=-1).mean())
            row["do_nothing_error_cm"] = float(np.linalg.norm(actual, axis=-1).mean())      # predicting no move
            if null_t is not None:
                row["null_order_error_cm"] = float(np.linalg.norm(null_t - actual, axis=-1).mean())
            if hi is not None:
                row["tenure_pred_cm_commanded_handle"] = t[hi].tolist()
                row["actual_cm_commanded_handle"] = actual[hi].tolist()
            if key != "hold" and tenure_hold is not None:
                pdv = (t - tenure_hold).reshape(-1)
                adv = (actual - hold["change"]["cm"]).reshape(-1)
                row["captured_share_tenure"] = H.captured_share(pdv, adv)
                row["dot_pred_actual_cm2"] = float(np.dot(pdv, adv))
                row["dot_actual_actual_cm2"] = float(np.dot(adv, adv))
                row["predicted_difference_cm"] = float(np.linalg.norm(pdv))
                row["actual_difference_cm"] = float(np.linalg.norm(adv))
        if key != "hold" and "proprioception" in a["readouts"][0]["tokens"]:
            pd = (a["pred"]["preds"]["proprioception"] - hold["pred"]["preds"]["proprioception"]).reshape(-1)
            ad = (a["readouts"][0]["tokens"]["proprioception"] - hold["readouts"][0]["tokens"]["proprioception"]).reshape(-1)
            row["captured_share_proprio_50ms"] = H.captured_share(pd, ad)
            row["dot_pred_actual_proprio"] = float(pd @ ad)
            row["dot_actual_actual_proprio"] = float(ad @ ad)
        eff = a["pred"].get("eff")
        if eff is not None and "body_tenure_cm" in eff:
            row["efference_tenure_cm_error"] = float(np.linalg.norm(eff["body_tenure_cm"] - a["change"]["cm"], axis=-1).mean())
            if hi is not None:
                row["efference_tenure_deg_commanded_handle"] = float(np.linalg.norm(eff["body_tenure_deg"][hi]))
            eff_hold = hold["pred"].get("eff")
            if key != "hold" and eff_hold is not None and "body_tenure_cm" in eff_hold:
                pe = (np.asarray(eff["body_tenure_cm"]) - np.asarray(eff_hold["body_tenure_cm"])).reshape(-1)
                ae = (a["change"]["cm"] - hold["change"]["cm"]).reshape(-1)
                row["dot_eff_pred_actual_cm2"] = float(np.dot(pe, ae))
                row["dot_eff_actual_actual_cm2"] = float(np.dot(ae, ae))
        out[key] = row
    # two-way discrimination per limb (a vs b), tenure head and proprioception token
    disc = {}
    for name in plans:
        a, b = arms.get(f"{name}_a"), arms.get(f"{name}_b")
        if a is None or b is None:
            continue
        d = {}
        if a["pred"].get("tenure_cm") is not None:
            d["tenure"] = H.pairing_correct(a["pred"]["tenure_cm"], b["pred"]["tenure_cm"],
                                            a["change"]["cm"], b["change"]["cm"])
        d["proprio_50ms"] = H.pairing_correct(a["pred"]["preds"]["proprioception"], b["pred"]["preds"]["proprioception"],
                                              a["readouts"][0]["tokens"]["proprioception"],
                                              b["readouts"][0]["tokens"]["proprioception"])
        disc[name] = d
    return dict(arms=out, discrimination=disc)


def _joint_deg(session: H.Session, joint: str) -> Optional[float]:
    e = session.env
    try:
        jid = e.model.joint(joint).id
    except Exception:
        return None
    return math.degrees(float(e.data.qpos[int(e.model.jnt_qposadr[jid])]))


def _elbow_turn(session: H.Session, snap: dict, order, torque_vec, fid: int) -> float:
    """Physics only: the elbow joint's signed turn (degrees) over one order tenure, with
    torque_vec (N m, world frame) applied to the forearm during every env step."""
    session.restore(snap)
    q0 = _joint_deg(session, ELBOW_JOINT)
    e = session.env
    for k in range(session.hold):
        e.data.xfrc_applied[fid, 3:6] = torque_vec
        session.act(order, new_order=(k == 0), render=False)
        e.data.xfrc_applied[fid, :] = 0.0
    return _joint_deg(session, ELBOW_JOINT) - q0


def size_load(turn_of, unloaded_deg: float, share: float = LOAD_TURN_SHARE, start: float = LOAD_TORQUE_START_NM,
              cap: float = LOAD_TORQUE_MAX_NM, bisections: int = LOAD_BISECTIONS) -> dict:
    """The smallest torque (to 1/2**bisections of its bracket) whose loaded turn is at most
    (1 - share) of the unloaded turn in the unloaded turn's direction. turn_of(torque) -> signed
    turn in degrees. Returns {torque_nm or None, loaded_deg, search: [[torque, turn], ...]}."""
    s = 1.0 if unloaded_deg >= 0 else -1.0
    target = (1.0 - float(share)) * abs(unloaded_deg)
    search, lo, hi, got_hi = [], 0.0, None, None
    t = float(start)
    while t <= cap * (1 + 1e-9):
        got = turn_of(t)
        search.append([t, got])
        if s * got <= target:
            hi, got_hi = t, got
            break
        lo, t = t, t * 2.0
    if hi is None:
        return dict(torque_nm=None, loaded_deg=None, search=search)
    for _ in range(int(bisections)):
        mid = 0.5 * (lo + hi)
        got = turn_of(mid)
        search.append([mid, got])
        if s * got <= target:
            hi, got_hi = mid, got
        else:
            lo = mid
    return dict(torque_nm=hi, loaded_deg=got_hi, search=search)


def _perturbation(session, snap, senses, plans, spread_deg=None):
    """The elbow order with and without a load against the elbow's own turn; the model's 50 ms
    proprioception error over the loaded tenure against the unloaded one (in-context adaptation).
    The load is a torque on the forearm about the elbow's hinge axis, opposing the unloaded turn,
    sized by size_load; when no torque up to the cap halves the turn the arm reports it
    unmeasured and runs no adaptation reading on a load that did not change the motion."""
    plan = plans.get("elbow")
    if not plan or "turns_deg" not in plan:
        return dict(status="unmeasured: no elbow plan")
    e = session.env
    fid = e.model.body(LOAD_BODY).id
    session.restore(snap)
    base = session.hold_order()
    order = session.rotated(base, plan["handle"], plan["axis"], plan["turns_deg"][0])
    hi = session.handle_names.index(plan["handle"])
    axis = next((np.asarray(a["axis"], dtype=np.float64) for a in session.joint_axes(plan["handle"])
                 if a["joint"] == ELBOW_JOINT), None)
    if axis is None or _joint_deg(session, ELBOW_JOINT) is None:
        return dict(status="unmeasured: no elbow hinge in this body")
    zero = np.zeros(3)
    unloaded = _elbow_turn(session, snap, order, zero, fid)
    sign = 1.0 if unloaded >= 0 else -1.0
    sized = size_load(lambda t: _elbow_turn(session, snap, order, -sign * axis * t, fid), unloaded)
    out = dict(load="torque on the %s about the elbow's hinge axis, against the elbow's own turn" % LOAD_BODY,
               line=dict(share_of_turn_removed=LOAD_TURN_SHARE,
                         why="half the turn: twice the elbow's own between-moment spread for the same order on "
                             "ckpt_step_344400 (10.9 degrees turned, 2.7 degrees spread over 3 moments)",
                         this_run_between_moment_spread_deg=spread_deg),
               unloaded_turn_deg=unloaded, search=sized["search"])
    if sized["torque_nm"] is None:
        largest = sized["search"][-1] if sized["search"] else [None, None]
        out.update(status="unmeasured: the load did not change the motion",
                   why=("no torque up to %.1f N m reduced the elbow's 250 ms turn of %.1f degrees by half (the "
                        "largest tried, %.1f N m, left %.1f degrees)" % (LOAD_TORQUE_MAX_NM, abs(unloaded),
                                                                        largest[0] or 0.0, abs(largest[1] or 0.0))))
        return out
    torque_vec = -sign * axis * sized["torque_nm"]

    def run(loaded):
        session.restore(snap)
        start = session.readout()
        q0 = _joint_deg(session, ELBOW_JOINT)
        errs = []
        rows = []
        for k in range(session.hold):
            mom = session.moment()
            pred = H.predict(session.model, mom, session.exec_order(order), want_eff=False)
            if loaded:
                e.data.xfrc_applied[fid, 3:6] = torque_vec
            session.act(order, new_order=(k == 0))
            e.data.xfrc_applied[fid, :] = 0.0
            session.observe()
            r = session.readout()
            rows.append(r)
            sc = H.token_scores(pred["preds"], r["tokens"], mom.tokens, ["proprioception"]).get("proprioception", {})
            errs.append(sc.get("pred_err"))
        ch = H.body_change(start, rows[-1])
        return dict(elbow_turn_deg=_joint_deg(session, ELBOW_JOINT) - q0, handle_turn_deg=float(ch["deg"][hi]),
                    move_cm=float(ch["cm_norm"][hi]), proprio_err_by_step=errs)

    free, loaded = run(False), run(True)
    # a step whose order stays the same: after the first step the model's context holds the load
    ratio = lambda r: (r["proprio_err_by_step"][-1] / r["proprio_err_by_step"][1]) if r["proprio_err_by_step"][1] else None
    out.update(status="measured", torque_nm=sized["torque_nm"], free=free, loaded=loaded,
               load_reduced_turn_deg=abs(free["elbow_turn_deg"]) - abs(loaded["elbow_turn_deg"]),
               error_trend_free=ratio(free), error_trend_loaded=ratio(loaded),
               note="error_trend = the 50 ms proprioception error on the tenure's last step over its second "
                    "(1 = no change; under 1 = the model's prediction improved as the load persisted). The "
                    "model is frozen: any improvement comes from its context, not from learning.")
    return out


def _repeat(session, snap, plans, repeats=3):
    plan = plans.get("elbow")
    if not plan or "turns_deg" not in plan:
        return dict(status="unmeasured: no elbow plan")
    hi = session.handle_names.index(plan["handle"])
    outs = []
    for _ in range(int(repeats)):
        session.restore(snap)
        start = session.readout()
        base = session.hold_order()
        order = session.rotated(base, plan["handle"], plan["axis"], plan["turns_deg"][0])
        pred = H.predict(session.model, session.moment(), order, want_eff=False)
        rs = session.run_arm(snap, order, steps=session.hold)
        outs.append((H.body_change(start, rs[-1]), pred.get("tenure_cm")))
    turn = [o[0]["deg"][hi] for o in outs]
    t0 = outs[0][1]
    pred_spread = (max(float(np.abs(o[1] - t0).max()) for o in outs) if t0 is not None else None)
    return dict(repeats=int(repeats), turn_deg=[float(x) for x in turn],
                turn_spread_deg=float(np.max(turn) - np.min(turn)), prediction_spread_cm=pred_spread,
                exact=bool(np.max(turn) - np.min(turn) == 0.0 and (pred_spread in (None, 0.0))))


def _pre_turned(session, snap, senses, rng, plans):
    """Generalization: the shoulder held PRE_TURN_DEG away for one tenure, then the limb orders."""
    session.restore(snap)
    sh = plans.get("shoulder")
    if not sh or "turns_deg" not in sh:
        return dict(status="unmeasured: no shoulder plan")
    base = session.hold_order()
    pre = session.rotated(base, sh["handle"], sh["axis"], math.copysign(PRE_TURN_DEG, sh["turns_deg"][0]))
    hi = session.handle_names.index(sh["handle"])
    q_start = session.pose()[1][hi]
    j0 = _joint_deg(session, sh["joint"])
    for k in range(session.hold):
        session.act(pre, new_order=(k == 0))
        session.observe()
    # what the shoulder really did before the limb orders (review of the probes, 2026-09-23: the
    # ordered pre-turn was recorded as if it were the body's)
    achieved = dict(shoulder_turn_deg=H.rotation_deg_xyzw(q_start, session.pose()[1][hi]),
                    joint_change_deg=(None if j0 is None else _joint_deg(session, sh["joint"]) - j0),
                    joint=sh["joint"])
    snap2 = session.snapshot()
    snap2["plan0"] = None                 # the planner's plan belongs to the first moment, not this one
    res = _state(session, snap2, senses, rng, limbs=("finger", "wrist", "elbow", "leg"), with_random=False)
    return dict(pre_turn_ordered_deg=PRE_TURN_DEG, pre_turn_achieved=achieved,
                scored=_score(res, session.handle_names))


def run_states(frozen: H.Frozen, *, states: int, warm: int, spacing: int, seed: int, world_of=None,
               extras: bool = True) -> dict:
    senses = H.real_senses(frozen.model)
    rng = np.random.default_rng(seed + 17)
    scored, extras_out, cfg = [], {}, None
    eff_max = {}
    # want_plan: each start is a moment where the planner made a plan (its first action is the
    # planner's side of the predicted-wellness choice); ignored by a build without a planner. A
    # body whose planner makes no plan in its window starts at its first planning event without
    # one, and its wellness choice says "no plan" (harness START_WITHOUT_A_PLAN: such a body used
    # to be dropped without a word)
    for si, (session, snap) in enumerate(H.start_states(frozen, states, warm=warm, spacing=spacing, seed=seed,
                                                         want_plan=True, plan_fallback=True, world_of=world_of,
                                                         gaze_reflex=False, saccades=False)):
        if cfg is None:
            cfg = H.frozen_config(frozen, session)
        res = _state(session, snap, senses, rng)
        for a in res["arms"].values():
            H.merge_max(eff_max, a["pred"].get("eff_hidden_max"))
        sc = _score(res, session.handle_names)
        sc["plans"] = {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
                       for k, v in res["plans"].items()}
        sc["start"] = session.start
        sc["step_offset"] = snap.get("step_offset")
        sc["start_planner"] = snap.get("planner")
        sc["command_pathway"] = res["pathway"]
        sc["wellness_choice"] = res["wellness_choice"]
        scored.append(sc)
        if extras and si == 0:
            extras_out["instrument_repeat"] = _repeat(session, snap, res["plans"])
            extras_out["perturbation"] = _perturbation(session, snap, senses, res["plans"])
            extras_out["generalization"] = _pre_turned(session, snap, senses, rng, res["plans"])
        del res
    return dict(senses=senses, states=scored, extras=extras_out, config=cfg or {},
                efference_live=H.live_share_of(eff_max))


def _limb_summary(states, limb):
    rows = [s["arms"].get(f"{limb}_{x}") for s in states for x in ("a", "b")]
    rows = [r for r in rows if r]
    if not rows:
        return None
    a_rows = [s["arms"].get(f"{limb}_a") for s in states if s["arms"].get(f"{limb}_a")]
    turns = [r["achieved_turn_deg"] for r in a_rows if "achieved_turn_deg" in r]
    return dict(captured_share_tenure=H.pooled_share(rows, "dot_pred_actual_cm2", "dot_actual_actual_cm2"),
                captured_share_proprio_50ms=H.pooled_share(rows, "dot_pred_actual_proprio", "dot_actual_actual_proprio"),
                turn_vs_hold_deg=H.mean_or_none(r.get("turn_vs_hold_deg") for r in rows),
                move_vs_hold_cm=H.mean_or_none(r.get("move_vs_hold_cm") for r in rows),
                tenure_error_cm=H.mean_or_none(r.get("tenure_error_cm") for r in rows),
                do_nothing_error_cm=H.mean_or_none(r.get("do_nothing_error_cm") for r in rows),
                achieved_turn_deg=dict(mean=H.mean_or_none(turns),
                                       spread=float(np.std(turns)) if len(turns) > 1 else None,
                                       n=len(turns)),
                commanded_turn_deg=H.mean_or_none(abs(r.get("commanded_turn_deg", 0.0)) for r in a_rows),
                discrimination_tenure=H.mean_or_none([s["discrimination"].get(limb, {}).get("tenure") for s in states]),
                discrimination_proprio_50ms=H.mean_or_none([s["discrimination"].get(limb, {}).get("proprio_50ms") for s in states]))


def _liveness_summary(states) -> dict:
    rows = [a.get("tenure_hidden") for s in states for a in s["arms"].values() if a.get("tenure_hidden")]
    rows = [r for r in rows if "live_share" in r]
    if not rows:
        return H.absent("tenure_pose_head")
    live = H.mean_or_none(r["live_share"] for r in rows)
    lo, hi = min(r["input_min"] for r in rows), max(r["input_max"] for r in rows)
    units = max(int(r.get("units") or 0) for r in rows)
    # the operator's sentence says what the line means and how near the nearest unit came, never a raw
    # input value (verify-fix-probes 2026-09-23: "inputs from -32.9 to -7.1" is a number a person
    # cannot picture; fix-readout-explainer 2026-09-23: so is "rises above -3; the highest was -7").
    # The raw values stay in input_min / input_max / live_input_line for a reader of the file.
    return dict(live_share=live, input_min=lo, input_max=hi, units=units, arms=len(rows),
                live_input_line=H.GELU_LIVE_INPUT, live_follow_share=H.LIVE_FOLLOW,
                lived=liveness_lived(units, live, len(rows), hi))


def liveness_lived(units: int, live, orders: int, input_max) -> str:
    """The tenure pose head's liveness sentence from its numbers (also used to re-write a stored
    sentence whose numbers are unchanged)."""
    return ("Of the tenure pose head's %d hidden units, %s responded, averaged over the probe's %d "
            "orders (0%% = the head returns the same number whatever the state and order, 100%% = "
            "every unit responds). %s %s"
            % (units, H.pct(live), orders, H.live_rule_text(), H.nearest_unit_text(input_max)))


def dead_head_lived(units: int, input_max) -> str:
    """The sentences the headline adds when none of the tenure pose head's units responded."""
    return (" The tenure pose head gave the same output for every state and order: none of its %d hidden "
            "units responded. %s %s" % (units, H.live_rule_text(), H.nearest_unit_text(input_max)))


def _efference_liveness(live: Optional[dict]) -> dict:
    """The efference body head's hidden layers, the share of units whose input rose above the
    GELU_LIVE_INPUT line on any of this probe's predictions (DESIGN section 10; review of the probes,
    2026-09-23: the headline switches to this head, and a dead head returns its bias whatever
    the order, as the tenure head did)."""
    if not live:
        return dict(status="unmeasured: no efference body head prediction was made, or its layers are "
                           "not a Linear-GELU sequence")
    worst = min(live.values())
    return dict(layers=live, live_input_line=H.GELU_LIVE_INPUT, live_follow_share=H.LIVE_FOLLOW,
                live_share_line=H.LIVE_SHARE_LINE, all_layers_live=bool(worst >= H.LIVE_SHARE_LINE),
                lived=("The efference body head's hidden layers: %s of units responded; a layer under %s is "
                       "not responding. %s"
                       % (" and ".join(H.pct(v) for v in live.values()), H.pct(H.LIVE_SHARE_LINE),
                          H.live_rule_text())))


def _pathway_summary(states) -> dict:
    shares = [v for s in states for k, v in (s.get("command_pathway", {}).get("token_change_share") or {}).items()
              if v is not None and k != "random"]
    rnd = [s.get("command_pathway", {}).get("token_change_share", {}).get("random") for s in states]
    w = next((s.get("command_pathway", {}).get("encoder_first_layer_mean_abs_weight") for s in states), None)
    return dict(limb_order_moves_motor_token_share=H.mean_or_none(shares),
                random_order_moves_motor_token_share=H.mean_or_none(rnd),
                encoder_first_layer_mean_abs_weight=w,
                lived=("One limb turned 20 degrees moved the motor token by %s of its size; a random order moving "
                       "every part by %s. The motor token is the only way an order reaches the model's predictions."
                       % (H.pct(H.mean_or_none(shares)), H.pct(H.mean_or_none(rnd)))))


def _choice_summary(states) -> dict:
    rows = [r for s in states for r in (s.get("wellness_choice") or {}).values()]
    scored = [r for r in rows if r.get("better") is not None]
    ties = sum(1 for r in scored if r["better"] == "tie")
    agree = [r["agrees"] for r in scored if r.get("agrees") is not None]
    planned = sum(1 for r in scored if "planner_toward" in r)
    return dict(limb_pairs=len(rows), scored=len(scored), tied=ties, with_plan=planned,
                planner_agreed=int(sum(agree)), planner_compared=len(agree),
                lived=("Of %d pairs of limb orders scored by the model's predicted wellness from the same moment, "
                       "%d scored the same for both orders; the planner's own plan leaned toward the better-scored "
                       "order in %d of the %d pairs it could be compared on (a pair with a tie or no plan cannot be)."
                       % (len(scored), ties, int(sum(agree)), len(agree))))


def ladder(limbs: dict, components: dict) -> list:
    """The addendum's section 7 milestone ladder. The rungs this probe touches are marked by
    what it measured: physics following an order the PROBE scripted, and the model's prediction
    of it. Whether the model makes the movement from its own intention, repeatedly, is not tested
    here (findings.repeatability is the simulator-determinism check), so no rung is "measured"
    as a whole (review of the probes, 2026-09-23)."""
    scripted = "the order's effect and its prediction measured; repeatable intended movement unmeasured"

    def rung(n, name, key=None, status=None, where=None):
        r = dict(rung=n, milestone=name)
        if key is not None and limbs.get(key):
            r.update(status=scripted, measured=limbs[key])
        else:
            r["status"] = status or "not measurable yet"
            if where:
                r["measured_in"] = where
        return r
    vest = components.get("vestibular", {}).get("present")
    touch = ("partly: the binding probe tests whether a scripted touch is reachable and how well its onset is "
             "foreseen; the model making the touch from its own intention is unmeasured")
    return [
        rung(1, "Repeatable finger flexion/extension", "finger"),
        rung(2, "Repeatable wrist movement", "wrist"),
        rung(3, "Repeatable elbow movement", "elbow"),
        rung(4, "Finger-to-finger self-touch", status=touch, where="binding.json"),
        rung(5, "Hand-to-face or hand-to-torso touch", status=touch, where="binding.json"),
        rung(6, "Raise the arm into view and predict where it will appear",
             status="partly: the binding probe sends the hand into view with a scripted order and scores the "
                    "prediction against holding still; the model raising it from its own intention is unmeasured",
             where="binding.json"),
        rung(7, "Push against the ground or support surface",
             status=("not measurable yet: no model output predicts support force; the vestibular sense "
                     "carries measured support shares but the model predicts only its token"
                     if vest else "not measurable yet: no model output predicts support force and the "
                     "vestibular sense (support shares) is absent from this build")),
        rung(8, "Shift weight while predicting proprioceptive and vestibular effects",
             status="see the posture probe's weight-shift arm: it counts only when its order rolled the chest "
                    "at least half as far as asked, and says unmeasured otherwise"
                    + ("" if vest else "; the vestibular part is absent from this build"),
             where="posture.json"),
        rung(9, "Lift part of the trunk",
             status="partly: the posture probe lifts the head (its achieved tilt and the neck's torque and power "
                    "are measured); lifting the trunk itself is unmeasured", where="posture.json"),
        rung(10, "Combine limbs and trunk into rolling, propping, crawling-like postures",
             status="not measurable yet: needs a multi-step intended-consequence test; "
                    "tools/motor_reachability_probe.py measures only whether a roll is physically reachable"),
    ]


def run(ckpt, *, states: int = 3, warm: int = 300, spacing: int = 10, seed: int = 21, write: bool = True,
        root=None, earlier: bool = True) -> dict:
    t0 = time.time()
    H.reset_routes()
    frozen = H.load_frozen(ckpt)
    main_res = run_states(frozen, states=states, warm=warm, spacing=spacing, seed=seed)
    route = H.route_record()               # before the earlier checkpoint's own moments are counted
    limbs = {name: _limb_summary(main_res["states"], name) for name, _, _ in LIMBS}
    pert = main_res["extras"].get("perturbation")
    if isinstance(pert, dict) and isinstance(pert.get("line"), dict) and limbs.get("elbow"):
        pert["line"]["this_run_between_moment_spread_deg"] = limbs["elbow"]["achieved_turn_deg"]["spread"]
    random_rows = [s["arms"].get("random") for s in main_res["states"] if s["arms"].get("random")]
    controls = dict(
        hold="every limb order is compared with holding still from the same moment",
        persistence="each sense's 'nothing changes' error is in every arm's lesson_50ms scores",
        do_nothing="do_nothing_error_cm = the error of predicting that nothing moves; null_order_error_cm = "
                   "the tenure head's own prediction for its do-nothing order",
        random_order=dict(captured_share_tenure=H.pooled_share(random_rows, "dot_pred_actual_cm2", "dot_actual_actual_cm2"),
                          tenure_error_cm=H.mean_or_none(r.get("tenure_error_cm") for r in random_rows),
                          do_nothing_error_cm=H.mean_or_none(r.get("do_nothing_error_cm") for r in random_rows)))
    earlier_out = None
    if earlier:
        earlier_out = _earlier(frozen, seed=seed, warm=warm, spacing=spacing)
    controls["earlier_checkpoint"] = earlier_out
    comps = H.components(frozen.model)
    limb_rows = [s["arms"][k] for s in main_res["states"] for k in s["arms"] if k not in ("hold", "random")]
    # DESIGN.md (probes): the efference body head in cm when the build has it, else the tenure head
    use_eff = comps["efference"]["present"]
    if use_eff:
        value = H.pooled_share(limb_rows, "dot_eff_pred_actual_cm2", "dot_eff_actual_actual_cm2")
        reader, metric = "efference body head", "body_captured_share_efference_cm"
    else:
        value = H.pooled_share(limb_rows, "dot_pred_actual_cm2", "dot_actual_actual_cm2")
        reader, metric = "tenure pose head", "body_captured_share_tenure_cm"
    tenure_value = H.pooled_share(limb_rows, "dot_pred_actual_cm2", "dot_actual_actual_cm2")
    moved = [limbs[n]["turn_vs_hold_deg"] for n in limbs if limbs.get(n) and limbs[n]["turn_vs_hold_deg"] is not None]
    lived = ("When one limb was ordered to turn %.0f degrees (finger, wrist, elbow, shoulder, knee) and the rest "
             "held still, the commanded part ended %s degrees away from where holding still left it, after 250 ms; "
             "the model's 250 ms prediction of where every body part would be (its %s) foresaw %s of the real "
             "difference the order made (0%% = it ignores the order, 100%% = it foresees the whole change). "
             "Distance to goal unmeasured."
             % (TURN_DEG, "%.1f" % float(np.mean(moved)) if moved else "an unmeasured number of", reader,
                H.share_text(value)))
    live = _liveness_summary(main_res["states"])
    eff_live = _efference_liveness(main_res.get("efference_live")) if use_eff else H.absent("efference")
    if use_eff:
        lived += " " + eff_live.get("lived", eff_live.get("status", ""))
    elif isinstance(live, dict) and live.get("live_share") == 0.0:
        lived += dead_head_lived(live.get("units") or 0, live["input_max"])
    if not comps["tenure_pose_head"]["present"] and not use_eff:
        lived = "unmeasured: the build has no tenure pose head and no efference body head to read a body prediction from"
    body = dict(config=dict(main_res["config"], states=states, warm=warm, spacing=spacing, seed=seed,
                            limbs=[dict(name=n, handle=h, joint=j) for n, h, j in LIMBS], turn_deg=TURN_DEG,
                            random_turn_rms_deg=RANDOM_TURN_RMS_DEG, route=route,
                            load_turn_share=LOAD_TURN_SHARE),
                arms=dict(states=main_res["states"]), controls=controls,
                findings=dict(limbs=limbs, repeatability=main_res["extras"].get("instrument_repeat"),
                              command_pathway=_pathway_summary(main_res["states"]),
                              tenure_head_liveness=live,
                              efference_head_liveness=eff_live,
                              wellness_choice=_choice_summary(main_res["states"]),
                              perturbation=pert,
                              generalization=main_res["extras"].get("generalization"),
                              efference=(dict(captured_share_cm=value, error_cm=H.mean_or_none(
                                  r.get("efference_tenure_cm_error") for r in limb_rows))
                                  if use_eff else H.absent("efference")),
                              tenure_captured_share_cm=tenure_value,
                              ladder=ladder(limbs, comps)),
                headline=dict(H.headline(metric, value,
                                         "share of the real 250 ms change in where every body part ended up that "
                                         "the prediction foresaw: from none (ignored the order) to all (foresaw "
                                         "the whole change)", 0.0, None, lived),
                              reader_liveness=(eff_live if use_eff else live)),
                status=("measured" if value is not None else "partial: no body prediction to score"),
                elapsed_s=round(time.time() - t0, 1))
    if write:
        body["path"] = str(H.write_result(frozen.info, PROBE, ITEM, body, root=root))
    return body


def _earlier(frozen: H.Frozen, *, seed, warm, spacing) -> dict:
    """The model's own earlier checkpoint: the OLDEST ancestor on disk (lineage.json resumed_from
    chain) that loads with nothing reinitialised, trying every one oldest first. Each runs from its own
    episode companion (its own lived moment, body and world at its save), or a warmed fresh world
    when it has none, and the same single-limb orders are scored the same way."""
    import gc
    cands = H.ancestors(frozen.info)
    if not cands:
        return dict(status="unmeasured: no ancestor checkpoint found on disk", chain=H.earlier_checkpoint(frozen.info))
    tried = []
    # every ancestor, OLDEST FIRST, until one loads with nothing reinitialised and runs (review of
    # the probes, 2026-09-23: only the oldest and the nearest were tried, so the control used a
    # checkpoint 4.7% of the lineage back while 23 others were never tried)
    for path in cands:
        gc.collect()
        try:
            old = H.load_frozen(path)
            res = run_states(old, states=1, warm=warm, spacing=spacing, seed=seed, extras=False)
            limbs = {name: _limb_summary(res["states"], name) for name, _, _ in LIMBS}
            rows = [st["arms"][k] for st in res["states"] for k in st["arms"] if k not in ("hold", "random")]
            out = dict(checkpoint=old.info["checkpoint"], lineage_step=old.info["lineage_step"],
                       captured_share_tenure=H.pooled_share(rows, "dot_pred_actual_cm2", "dot_actual_actual_cm2"),
                       captured_share_proprio_50ms=H.pooled_share(rows, "dot_pred_actual_proprio",
                                                                  "dot_actual_actual_proprio"),
                       limbs=limbs, start=res["config"].get("start"), ancestors_on_disk=len(cands),
                       tried=tried, tried_count=len(tried) + 1, tenure_head_liveness=_liveness_summary(res["states"]),
                       command_pathway=_pathway_summary(res["states"]))
            del old
            return out
        except Exception as error:          # an ancestor that cannot run is reported, not hidden
            tried.append(dict(checkpoint=path, refused=f"{type(error).__name__}: {error}"[:400]))
    return dict(status="unmeasured: no ancestor on disk loaded with nothing reinitialised and ran", tried=tried,
                tried_count=len(tried), ancestors_on_disk=len(cands))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--states", type=int, default=3)
    ap.add_argument("--warm", type=int, default=300, help="fresh-world warm-up steps, used only when the checkpoint has no episode companion")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--no-earlier", action="store_true")
    a = ap.parse_args(argv)
    out = run(a.ckpt, states=a.states, warm=a.warm, seed=a.seed, earlier=not a.no_earlier)
    print(json.dumps(dict(headline=out["headline"], path=out.get("path"), elapsed_s=out["elapsed_s"]), indent=1))


if __name__ == "__main__":
    main()
