# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Posture and vestibular probes (addendum v2 sections 7, 8 and 14, build item 10).

    python -m mimo_bridge.tools.gof_probe_posture --ckpt <checkpoint folder> [--states 2]

NECK COUNTERFACTUAL. From a restored moment only the head's order changes (turned 15 degrees
about the neck's swivel axis either way, or tilted), the rest holds. Predicted against actual,
each relative to holding still from the same moment: the vestibular token when the build has
the vestibular input (--vestibular-sense on); otherwise the proprioception token, labelled as a proxy (the head's row
cannot be separated inside the one proprioception token); and the tenure pose head's predicted
250 ms move of the head (cm).

REAFFERENCE. The same head turn made two ways: commanded (the body's own order), and imposed
by a torque from outside (MuJoCo xfrc_applied on the head, about the neck swivel axis) while the
body is told to hold still, the torque sized by search so the SIGNED swivel-joint angle turns as
far as the self-made turn did (within MATCH_TOL). The two turns' directions are compared as well
(the angle between their rotation vectors). A model that uses its own orders predicts the
self-made turn better than the imposed one: per start moment, the imposed turn's next-moment
error over the self-made one's (1 = no difference). A difference is reported only when every
start moment points the same way; otherwise "no consistent difference".

WEIGHT SHIFT. The trunk and the right leg ordered to shift the body's weight, first by leaning
the trunk joints and then by rolling the trunk about the body's length axis (WEIGHT_SHIFT_ORDERS,
smallest first), until the chest rolls at least
WEIGHT_SHIFT_SHARE of the roll the order asked for (and more than twice the holding arm's own
roll drift). The achieved chest roll is always reported; when no order gets there the arm is
"unmeasured: the order did not roll the chest". Predicted against actual chest move (tenure head,
cm); a predicted roll needs the efference body head.

HEAD LIFT EFFORT. The head ordered to tilt up for two tenures: the tilt achieved (degrees) and
the neck's effort - the joint torque MuJoCo applies through the neck actuators (qfrc_actuator at
the three neck joints, N m) and the mechanical power (that torque times the joint's speed, W) -
against holding still. The model has no effort output, so predicted effort is reported
unmeasured. (Review of the probes, 2026-09-23: the first version summed the actuators' control
force, which is torque divided by the gear of 1.97 to 3.62, and labelled it N m: 1.06 against a
joint torque of 3.63 N m for the same lift.)

Headline: the share of the real change foreseen when only the head's order changes (vestibular
token when present, else the proprioception token), relative to holding still.
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

import numpy as np
import torch

from mimo_bridge.tools import gof_probe_harness as H

PROBE, ITEM = "posture", 10
# 15 degrees: inside the neck's ranges (swivel +/-111, tilt -70..81, measured on the crib body at
# 24 months, understand-senses-proprio-vestibular.md) and inside the run's own order-long turns
# (root-mean-square 11.0 degrees per axis, 99 in 100 under 42.2; lem/model/efference.py).
NECK_DEG = 15.0
LIFT_DEG = 30.0
# Weight shift orders, smallest first. "lean": the lower trunk and chest turned about their
# hip_lean1 and chest_lean joint axes (degrees; hip_lean1's range is +/-14 and chest_lean's +/-9,
# so the joint limit clamps the larger asks, as it would clamp the model's own). "roll": both
# targets turned about the body's length axis, the turn chest_orientation reads as roll (0 = on
# the back, 90 = on the side): 20 and 45 degrees are a fifth and a half of the way to the side.
# Each adds the right hip's flexion ("leg", degrees; right_hip1's range is -133 to +20, resting
# near -50 on the crib body, so 90 degrees stays inside it). Measured on ckpt_step_344400 (smoke
# run 2026-09-23, one start moment): the three lean orders asked the chest for a roll of only 0.9
# to 1.2 degrees (a lean is mostly not a roll) and got under 0.1 degrees against holding still;
# the roll orders asked for 20 and 45 degrees and got 1.1 and 1.5 degrees, against a holding
# drift of 1.1 degrees - no order reached the line, and the arm reported itself unmeasured.
WEIGHT_SHIFT_ORDERS = (dict(kind="lean", hip=12.0, chest=12.0, leg=30.0),
                       dict(kind="lean", hip=14.0, chest=9.0, leg=60.0),
                       dict(kind="lean", hip=14.0, chest=9.0, leg=90.0),
                       dict(kind="roll", roll=20.0, leg=60.0),
                       dict(kind="roll", roll=45.0, leg=90.0))
# The chest must roll at least this share of the roll the order asked for (the same declared line
# as the body probe's load: most of the motion asked for must happen before a prediction of it
# says anything about THAT motion), and more than twice the holding arm's own roll drift over the
# same steps (the control measured in the same run). On ckpt_step_344400 the first order rolled
# the chest 0.02 and -0.64 degrees against holding still (baseline posture probe, 2026-09-22).
WEIGHT_SHIFT_SHARE = 0.5
MATCH_TOL = 0.10               # reafference: imposed turn within 10% of the self-made one
NECK_JOINTS = ("robot:head_swivel", "robot:head_tilt", "robot:head_tilt_side")
SWIVEL = "robot:head_swivel"


def _joint(session, handle, joint):
    return next((a for a in session.joint_axes(handle) if a["joint"] == joint), None)


def _joint_deg(session, joint: str):
    e = session.env
    try:
        jid = e.model.joint(joint).id
    except Exception:
        return None
    return math.degrees(float(e.data.qpos[int(e.model.jnt_qposadr[jid])]))


def _sense_meaning(sense):
    """The sense's plain meaning, in the words both pages define it with (mimo_bridge.explainer.schema
    SENSE_DEFINITIONS, up to its first colon)."""
    from mimo_bridge.explainer.schema import SENSE_DEFINITIONS
    return SENSE_DEFINITIONS[sense].split(":")[0]


def headline_unit(sense):
    """The headline's unit text for the sense the neck arm scored, in the model's own name with its plain
    meaning (tools/lineage_series.probe_words reads it: 'vestibular' marks a direct reading, 'standing
    in' and 'proprioception' a stand-in)."""
    return ("share of the real change in what the body sensed one moment after a head order (%s), against "
            "holding still, that the prediction foresaw: from none (ignored the order) to all (foresaw the "
            "whole change)"
            % (f"the vestibular input, {_sense_meaning('vestibular')}" if sense == "vestibular"
               else f"the proprioception input, {_sense_meaning('proprioception')}, standing in for the "
                    f"vestibular input"))


def _token_sense(senses):
    """(the sense scored, its name for a sentence: the model's own name with its plain meaning).
    Results written before 2026-09-25 named the two senses in other words; the dashboard still reads
    those (tools/lineage_series.probe_words)."""
    return ("vestibular", f"vestibular input ({_sense_meaning('vestibular')})") if "vestibular" in senses else (
        "proprioception", f"proprioception input ({_sense_meaning('proprioception')}), standing in for the "
                          f"vestibular input this build lacks; the head's part cannot be separated inside it")


def _rotvec_deg(q0, q1) -> np.ndarray:
    """The turn from q0 to q1 (x y z w) as a rotation vector in degrees (world frame)."""
    q0 = np.asarray(q0, float) / max(np.linalg.norm(q0), 1e-12)
    q1 = np.asarray(q1, float) / max(np.linalg.norm(q1), 1e-12)
    r = H.quat_mul_xyzw(q1, np.r_[-q0[:3], q0[3]])
    if r[3] < 0:
        r = -r
    n = float(np.linalg.norm(r[:3]))
    if n < 1e-12:
        return np.zeros(3)
    return r[:3] / n * math.degrees(2.0 * math.atan2(n, r[3]))


def _angle_between_deg(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b)) / (na * nb)))))


def _arm_scores(session, snap, order, mom, sense, steps, *, before_step=None, after_step=None):
    """Predict each step from the moment before it (with the eyes' row act() will send), then
    step; returns per-step scores and the final readout."""
    model = session.model
    session.restore(snap)
    rows = []
    for k in range(int(steps)):
        m_now = session.moment()
        pred = H.predict(model, m_now, session.exec_order(order), want_eff=False)
        if before_step is not None:
            before_step(session, k)
        session.act(order, new_order=(k == 0))
        if after_step is not None:
            after_step(session, k)
        session.observe()
        r = session.readout()
        sc = H.token_scores(pred["preds"], r["tokens"], m_now.tokens, [sense]).get(sense, {})
        pd = (pred["preds"][sense] - m_now.tokens[sense].reshape(1, -1)).reshape(-1) if sense in pred["preds"] else None
        ad = (r["tokens"][sense] - m_now.tokens[sense]).reshape(-1) if sense in r["tokens"] else None
        rows.append(dict(step=k + 1, pred_err=sc.get("pred_err"), persist_err=sc.get("persist_err"),
                         dot_pa=None if pd is None else float(pd.double() @ ad.double()),
                         dot_aa=None if ad is None else float(ad.double() @ ad.double()), readout=r))
    return rows


def neck(session, snap, senses) -> dict:
    sense, label = _token_sense(senses)
    session.restore(snap)
    mom = session.moment()
    start = session.readout()
    base = session.hold_order()
    hi = session.handle_names.index("head")
    orders = {"hold": base}
    for jn, degs in ((SWIVEL, (NECK_DEG, -NECK_DEG)), ("robot:head_tilt", (NECK_DEG,))):
        j = _joint(session, "head", jn)
        if j is None:
            continue
        for d in degs:
            orders[f"{jn.split(':')[1]}_{'+' if d > 0 else '-'}{abs(d):.0f}"] = session.rotated(base, "head", j["axis"], d)
    pathway = H.command_pathway(session.model, mom, orders)
    preds = {k: H.predict(session.model, mom, o, want_eff=False) for k, o in orders.items()}
    acts = {k: session.run_arm(snap, o, steps=session.hold) for k, o in orders.items()}
    rows = {}
    for k in orders:
        if k == "hold":
            continue
        a_end, h_end = acts[k][-1], acts["hold"][-1]
        pd = (preds[k]["preds"][sense] - preds["hold"]["preds"][sense]).reshape(-1)
        ad = (acts[k][0]["tokens"][sense] - acts["hold"][0]["tokens"][sense]).reshape(-1)
        row = dict(head_turn_vs_hold_deg=H.rotation_deg_xyzw(a_end["quat"][hi], h_end["quat"][hi]),
                   head_move_vs_hold_cm=float(np.linalg.norm(np.asarray(a_end["pos_cm"][hi]) - np.asarray(h_end["pos_cm"][hi]))),
                   share_50ms=H.captured_share(pd, ad), dot_pa=float(pd.double() @ ad.double()),
                   dot_aa=float(ad.double() @ ad.double()))
        if "tenure_cm" in preds[k]:
            pt = (preds[k]["tenure_cm"] - preds["hold"]["tenure_cm"]).reshape(-1)
            at = ((np.asarray(a_end["pos_cm"]) - np.asarray(start["pos_cm"])) -
                  (np.asarray(h_end["pos_cm"]) - np.asarray(start["pos_cm"]))).reshape(-1)
            row.update(tenure_share=H.captured_share(pt, at), dot_pa_tenure=float(np.dot(pt, at)),
                       dot_aa_tenure=float(np.dot(at, at)),
                       predicted_head_move_cm=float(np.linalg.norm(preds[k]["tenure_cm"][hi] - preds["hold"]["tenure_cm"][hi])))
        rows[k] = row
    return dict(sense=sense, sense_label=label, arms=rows, command_pathway=pathway,
                pooled_share_50ms=H.pooled_share(list(rows.values()), "dot_pa", "dot_aa"),
                pooled_share_tenure=H.pooled_share(list(rows.values()), "dot_pa_tenure", "dot_aa_tenure"))


def reafference(session, snap, senses) -> dict:
    sense, label = _token_sense(senses)
    # every joint axis is read at the restored start: the world the previous arm left behind is
    # in another pose (found 2026-09-23: the head-lift order read its tilt axis there, and the same
    # probe gave 1.82 and 1.81 N m depending on which arm ran before it)
    session.restore(snap)
    j = _joint(session, "head", SWIVEL)
    if j is None or _joint_deg(session, SWIVEL) is None:
        return dict(status="unmeasured: no head swivel joint")
    base = session.hold_order()
    hi = session.handle_names.index("head")
    turn = session.rotated(base, "head", j["axis"], NECK_DEG)
    e = session.env
    head_body = e.model.body("head").id
    axis = np.asarray(j["axis"], dtype=np.float64)

    session.restore(snap)
    start = session.readout()
    sw0 = _joint_deg(session, SWIVEL)
    self_rs = session.run_arm(snap, turn, steps=session.hold)
    target = _joint_deg(session, SWIVEL) - sw0           # the self-made turn, signed swivel degrees
    self_rot = _rotvec_deg(start["quat"][hi], self_rs[-1]["quat"][hi])
    sign = 1.0 if target >= 0 else -1.0

    def imposed(torque):
        def push(s, k):
            s.env.data.xfrc_applied[head_body, 3:6] = sign * axis * torque

        def release(s, k):
            s.env.data.xfrc_applied[head_body, :] = 0.0
        return push, release

    def imposed_turn(torque):
        push, release = imposed(torque)
        session.restore(snap)
        for k in range(session.hold):
            push(session, k)
            session.act(base, new_order=(k == 0), render=False)
            release(session, k)
        return _joint_deg(session, SWIVEL) - sw0, _rotvec_deg(start["quat"][hi], session.pose()[1][hi])

    # search the torque (N m) that turns the swivel joint as far as the order did, the same way:
    # double until past, then bisect
    lo, hi_t, tried = 0.0, 0.01, []
    for _ in range(12):
        got, rot = imposed_turn(hi_t)
        tried.append((hi_t, got, rot))
        if sign * got >= abs(target):
            break
        lo, hi_t = hi_t, hi_t * 2.0
    best = min(tried, key=lambda x: abs(x[1] - target)) if tried else (0.0, 0.0, np.zeros(3))
    for _ in range(10):
        if abs(best[1] - target) <= MATCH_TOL * max(abs(target), 1e-9):
            break
        mid = 0.5 * (lo + hi_t)
        got, rot = imposed_turn(mid)
        tried.append((mid, got, rot))
        if sign * got < abs(target):
            lo = mid
        else:
            hi_t = mid
        best = min(tried, key=lambda x: abs(x[1] - target))
    torque = best[0]
    matched = abs(best[1] - target) <= MATCH_TOL * max(abs(target), 1e-9)
    session.restore(snap)
    mom = session.moment()
    self_rows = _arm_scores(session, snap, turn, mom, sense, session.hold)
    push, release = imposed(torque)
    imp_rows = _arm_scores(session, snap, base, mom, sense, session.hold, before_step=push, after_step=release)
    e_self = H.mean_or_none(r["pred_err"] for r in self_rows)
    e_imp = H.mean_or_none(r["pred_err"] for r in imp_rows)
    return dict(sense=sense, sense_label=label, self_swivel_turn_deg=target, imposed_swivel_turn_deg=best[1],
                self_turn_rotvec_deg=self_rot.tolist(), imposed_turn_rotvec_deg=np.asarray(best[2]).tolist(),
                angle_between_turns_deg=_angle_between_deg(self_rot, best[2]),
                imposed_torque_nm=torque, matched_within=MATCH_TOL, matched=bool(matched),
                search=[dict(torque_nm=t, swivel_turn_deg=g) for t, g, _r in tried],
                error_self=e_self, error_imposed=e_imp,
                persistence_self=H.mean_or_none(r["persist_err"] for r in self_rows),
                persistence_imposed=H.mean_or_none(r["persist_err"] for r in imp_rows),
                ratio_imposed_over_self=(e_imp / e_self) if e_self else None,
                share_self=H.pooled_share(self_rows, "dot_pa", "dot_aa"),
                share_imposed=H.pooled_share(imp_rows, "dot_pa", "dot_aa"))


def _wrap_deg(d):
    return None if d is None else float((d + 180.0) % 360.0 - 180.0)


def weight_shift(session, snap, senses) -> dict:
    from mimo_bridge.body_layout import chest_orientation
    from mimo_bridge.action_decoder import quat_xyzw_to_mat
    session.restore(snap)
    mom = session.moment()
    start = session.readout()
    base = session.hold_order()
    ci = session.handle_names.index("chest")
    axes = getattr(session.env, "_body_orientation_axes", None)

    def roll_of(quat_xyzw):
        return chest_orientation(np.asarray(quat_xyzw_to_mat(quat_xyzw)), axes).get("roll_degrees")

    joints = {jn: _joint(session, handle, jn) for handle, jn in (("lower_trunk", "robot:hip_lean1"),
                                                                ("chest", "robot:chest_lean"),
                                                                ("thigh_R", "robot:right_hip1"))}
    steps = 2 * session.hold
    rh = session.run_arm(snap, base, steps=steps)
    r0, rh1 = roll_of(start["quat"][ci]), roll_of(rh[-1]["quat"][ci])
    hold_drift = None if r0 is None or rh1 is None else abs(_wrap_deg(rh1 - r0))
    tried, chosen = [], None
    # the body's length axis in the world, from the chest's start orientation (chest_orientation's
    # first axis is the length axis in the chest's own frame)
    along = None
    if axes is not None:
        along = np.asarray(quat_xyzw_to_mat(start["quat"][ci])) @ np.asarray(axes)[0]
    for spec in WEIGHT_SHIFT_ORDERS:
        o = base.copy()
        turned = []
        if spec["kind"] == "lean":
            parts = (("lower_trunk", "robot:hip_lean1", spec["hip"]), ("chest", "robot:chest_lean", spec["chest"]))
            for handle, jn, deg in parts:
                j = joints.get(jn)
                if j is not None:
                    o = session.rotated(o, handle, j["axis"], deg)
                    turned.append(dict(handle=handle, joint=jn, deg=deg))
        elif along is not None:
            for handle in ("lower_trunk", "chest"):
                o = session.rotated(o, handle, along, spec["roll"])
                turned.append(dict(handle=handle, about="the body's length axis", deg=spec["roll"]))
        if not turned:
            continue
        j = joints.get("robot:right_hip1")
        if j is not None:
            o = session.rotated(o, "thigh_R", j["axis"], -spec["leg"])
            turned.append(dict(handle="thigh_R", joint="robot:right_hip1", deg=-spec["leg"]))
        o = session.exec_order(o)
        asked = _wrap_deg(None if r0 is None or roll_of(o[0, ci, 3:7]) is None else roll_of(o[0, ci, 3:7]) - r0)
        rs = session.run_arm(snap, o, steps=steps)
        r1 = roll_of(rs[-1]["quat"][ci])
        achieved = None if r1 is None or rh1 is None else _wrap_deg(r1 - rh1)
        ok = (asked is not None and achieved is not None and abs(asked) > 0
              and achieved * np.sign(asked) >= WEIGHT_SHIFT_SHARE * abs(asked)
              and (hold_drift is None or abs(achieved) > 2.0 * hold_drift))
        tried.append(dict(orders=turned, chest_roll_asked_deg=asked, chest_roll_vs_hold_deg=achieved, reached_line=bool(ok)))
        if ok:
            chosen = (o, rs, tried[-1])
            break
    out = dict(steps=steps, tried=tried, chest_roll_start_deg=r0, chest_roll_end_holding_deg=rh1,
               holding_roll_drift_deg=hold_drift, line=dict(share_of_asked_roll=WEIGHT_SHIFT_SHARE,
                                                            over_holding_drift_times=2.0))
    if not tried:
        out["status"] = "unmeasured: no trunk joints in the decoder's chains and no body axes to roll about"
        return out
    if chosen is None:
        best = max(tried, key=lambda t: abs(t["chest_roll_vs_hold_deg"] or 0.0))
        out.update(status="unmeasured: the order did not roll the chest",
                   why=("the order that rolled the chest furthest asked for %s degrees and got %s degrees against "
                        "holding still (the line: half the roll asked, and more than twice the %s degrees the chest "
                        "drifted while holding still)"
                        % ("%.1f" % abs(best["chest_roll_asked_deg"]) if best["chest_roll_asked_deg"] is not None else "an unknown number of",
                           "%.1f" % abs(best["chest_roll_vs_hold_deg"]) if best["chest_roll_vs_hold_deg"] is not None else "an unknown number of",
                           "%.1f" % hold_drift if hold_drift is not None else "unmeasured")))
        return out
    o, rs, row = chosen
    out.update(status="measured", orders=row["orders"], chest_roll_asked_deg=row["chest_roll_asked_deg"],
               chest_roll_change_vs_hold_deg=row["chest_roll_vs_hold_deg"])
    session.restore(snap)                      # predictions from the start moment itself
    mom = session.moment()
    p_o = H.predict(session.model, mom, o, want_eff=False)
    p_h = H.predict(session.model, mom, base, want_eff=False)
    if "tenure_cm" in p_o:
        first = session.hold - 1
        real = np.asarray(rs[first]["pos_cm"][ci]) - np.asarray(rh[first]["pos_cm"][ci])
        pred = np.asarray(p_o["tenure_cm"][ci]) - np.asarray(p_h["tenure_cm"][ci])
        out.update(chest_move_vs_hold_cm_first_tenure=float(np.linalg.norm(real)),
                   predicted_chest_move_vs_hold_cm=float(np.linalg.norm(pred)),
                   chest_share=H.captured_share(pred, real))
    out["predicted_roll"] = H.absent("efference") if getattr(session.model, "efference", None) is None else \
        dict(status="unmeasured: the efference body head predicts turns as rotation vectors; roll reading not wired")
    return out


def head_lift(session, snap) -> dict:
    session.restore(snap)                       # the tilt axis at the restored start, not where the last arm left it
    j = _joint(session, "head", "robot:head_tilt")
    if j is None:
        return dict(status="unmeasured: no head tilt joint")
    e = session.env
    base = session.hold_order()
    # lift = the tilt direction that raises the face from the floor for a body on its back: the
    # sign that increases the head's height, found by trying both
    heights = {}
    for sign in (1.0, -1.0):
        o = session.rotated(base, "head", j["axis"], sign * LIFT_DEG)
        session.physics_only(snap, o, session.hold)
        heights[sign] = float(e.data.xpos[e.model.body("head").id][2])
    sign = max(heights, key=heights.get)
    lift = session.rotated(base, "head", j["axis"], sign * LIFT_DEG)
    neck_dofs = [int(e.model.jnt_dofadr[e.model.joint(n).id]) for n in NECK_JOINTS
                 if _joint_deg(session, n) is not None]
    tilt_q = int(e.model.jnt_qposadr[e.model.joint("robot:head_tilt").id])

    def effort(order):
        session.restore(snap)
        q0 = float(e.data.qpos[tilt_q])
        z0 = float(e.data.xpos[e.model.body("head").id][2])
        tau, power = [], []

        def watch(s):
            # the joint torque the neck actuators apply (MuJoCo qfrc_actuator = actuator force x gear,
            # in N m on a hinge) and the mechanical power it does (torque x joint speed, W)
            t = np.asarray(e.data.qfrc_actuator)[neck_dofs]
            w = np.asarray(e.data.qvel)[neck_dofs]
            tau.append(float(np.abs(t).sum()))
            power.append(float(np.abs(t * w).sum()))
        session.physics_only(snap, order, 2 * session.hold, watch=watch)
        return dict(tilt_change_deg=math.degrees(float(e.data.qpos[tilt_q]) - q0),
                    head_rise_cm=100.0 * (float(e.data.xpos[e.model.body("head").id][2]) - z0),
                    neck_torque_mean_nm=float(np.mean(tau)) if tau else None,
                    neck_power_mean_w=float(np.mean(power)) if power else None)

    return dict(lift_sign=sign, lifting=effort(lift), holding=effort(base),
                effort_source="qfrc_actuator at the neck joints (swivel, tilt, side tilt): joint torque in N m; "
                              "power = |torque x joint speed| summed over the three, W",
                predicted_effort=dict(status="unmeasured: the model has no effort output"),
                note="effort sampled once per model step (50 ms) over two order tenures (500 ms)")


def reafference_sentence(reaf: list) -> str:
    ratios = [r["ratio_imposed_over_self"] for r in reaf if r.get("ratio_imposed_over_self") is not None]
    if not ratios:
        return "A self-made and a pushed head turn could not be compared (unmeasured)."
    pct_diff = [100.0 * (x - 1.0) for x in ratios]
    listed = ", ".join(("%+.1f%%" % d) for d in pct_diff)
    if all(x > 1.0 for x in ratios):
        verdict = "its own turn was predicted better at every start moment"
    elif all(x < 1.0 for x in ratios):
        verdict = "the pushed turn was predicted better at every start moment"
    else:
        verdict = "no consistent difference over the start moments"
    return ("A head turn the body made itself against the same turn pushed from outside: the pushed turn's "
            "next-moment error was larger than the self-made turn's by %s (one per start moment; 0%% = no "
            "difference, above 0%% = its own turn predicted better): %s." % (listed, verdict))


def lived_sentence(turns, label, value, reaf, shifted, n_shift, lifts) -> str:
    s = ("When only the head was ordered to turn or tilt 15 degrees, it ended %s degrees away from where "
         "holding still left it; the model's next-moment prediction of its %s foresaw %s of the real "
         "difference (0%% = it ignores the order, 100%% = it foresees the whole change). Distance to goal "
         "unmeasured. %s The weight-shift orders rolled the chest at least half as far as asked in %d of %d "
         "start moments."
         % ("%.1f" % float(np.mean(turns)) if turns else "an unmeasured number of", label, H.share_text(value),
            reafference_sentence(reaf), shifted, n_shift))
    up = [l["lifting"].get("neck_torque_mean_nm") for l in lifts if isinstance(l.get("lifting"), dict)]
    still = [l["holding"].get("neck_torque_mean_nm") for l in lifts if isinstance(l.get("holding"), dict)]
    up, still = [v for v in up if v is not None], [v for v in still if v is not None]
    if up and still:
        s += (" Lifting the head took %.2f N m of neck torque on average against %.2f N m holding still; the "
              "model has no effort prediction to compare." % (float(np.mean(up)), float(np.mean(still))))
    return s


def run(ckpt, *, states: int = 2, warm: int = 300, seed: int = 41, write: bool = True, root=None) -> dict:
    t0 = time.time()
    H.reset_routes()
    frozen = H.load_frozen(ckpt)
    senses = H.real_senses(frozen.model)
    comps = H.components(frozen.model)
    per_state, cfg = [], None
    for session, snap in H.start_states(frozen, states, warm=warm, seed=seed, gaze_reflex=False, saccades=False):
        if cfg is None:
            cfg = H.frozen_config(frozen, session)
        per_state.append(dict(start=session.start, neck=neck(session, snap, senses),
                              reafference=reafference(session, snap, senses),
                              weight_shift=weight_shift(session, snap, senses),
                              head_lift=head_lift(session, snap)))
    necks = [s["neck"] for s in per_state]
    value = H.pooled_share([r for n in necks for r in n["arms"].values()], "dot_pa", "dot_aa")
    sense, label = _token_sense(senses)
    reaf = [s["reafference"] for s in per_state if s["reafference"].get("ratio_imposed_over_self") is not None]
    ratio = H.mean_or_none(r["ratio_imposed_over_self"] for r in reaf)
    turns = [r["head_turn_vs_hold_deg"] for n in necks for r in n["arms"].values()]
    shifts = [s["weight_shift"] for s in per_state]
    shifted = sum(1 for w in shifts if w.get("status") == "measured")
    lived = lived_sentence(turns, label, value, reaf, shifted, len(shifts), [s["head_lift"] for s in per_state])
    body = dict(config=dict(cfg or {}, states=states, seed=seed, neck_deg=NECK_DEG, lift_deg=LIFT_DEG,
                            weight_shift_orders=WEIGHT_SHIFT_ORDERS, weight_shift_share=WEIGHT_SHIFT_SHARE,
                            match_tol=MATCH_TOL, route=H.route_record()),
                arms=dict(states=per_state),
                controls=dict(holding="every head order is compared with holding still from the same moment",
                              imposed="the reafference arm's control is the same turn imposed from outside"),
                findings=dict(neck_share_50ms=value, neck_sense=label,
                              neck_share_tenure_cm=H.pooled_share([r for n in necks for r in n["arms"].values()],
                                                                  "dot_pa_tenure", "dot_aa_tenure"),
                              reafference_ratio=ratio,
                              reafference_ratios=[r["ratio_imposed_over_self"] for r in reaf],
                              reafference_matched=[r.get("matched") for r in reaf],
                              reafference_angle_between_turns_deg=[r.get("angle_between_turns_deg") for r in reaf],
                              vestibular=(None if comps["vestibular"]["present"] else H.absent("vestibular")),
                              weight_shift=shifts, weight_shift_measured=dict(n=shifted, of=len(shifts)),
                              head_lift=[s["head_lift"] for s in per_state]),
                headline=H.headline("posture_neck_share_50ms", value, headline_unit(sense), 0.0, None, lived),
                status="measured" if value is not None else "partial: no head order changed the senses",
                elapsed_s=round(time.time() - t0, 1))
    if write:
        body["path"] = str(H.write_result(frozen.info, PROBE, ITEM, body, root=root))
    return body


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--states", type=int, default=2)
    ap.add_argument("--seed", type=int, default=41)
    a = ap.parse_args(argv)
    out = run(a.ckpt, states=a.states, seed=a.seed)
    print(json.dumps(dict(headline=out["headline"], path=out.get("path"), elapsed_s=out["elapsed_s"]), indent=1))


if __name__ == "__main__":
    main()
