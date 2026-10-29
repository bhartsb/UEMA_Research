# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Binding probes: vision, proprioception and touch (addendum v2 sections 2 and 7, build item 9).

    python -m mimo_bridge.tools.gof_probe_binding --ckpt <checkpoint folder> [--states 2]

ARM INTO VIEW. From a restored moment the right hand is ordered to a point in front of the eyes
(the wrist is a position handle: the decoder moves the arm to put the hand there). Which point
brings the hand into the eyes' wide field (hand centre within 30 degrees of an eye's axis, the
env's own hand_in_view test) is first SEARCHED on physics alone. Then, with the model watching,
at every 50 ms step up to the step the hand enters the view, the model predicts the next moment
TWICE from the same moment - for continuing the order and for stopping the hand where it is (the
hold order) - and the physics is branched one step each way from that moment. The binding score
is the counterfactual share:

    (pred_order - pred_hold) . (actual_order - actual_hold) / |actual_order - actual_hold|^2

0 = the prediction does not depend on the order, 1 = it foresees the whole difference the order
makes, negative = the wrong way. A model that ignores the order and only extends the hand's
motion from its 128-moment context scores 0 here, which is what "knowing nothing about the order"
scores; the continuation share (pred - current) . (actual - current) / |actual - current|^2 that
the first version reported is kept beside it, labelled, because such a model scores well on it
(review of the probes, 2026-09-23). Also the tenure pose head's predicted 250 ms move of the hand
(cm) against the real move, and a control arm of the same steps with the body holding still.

SELF-TOUCH. Three scripted touches through the decoder: right thumb to right index finger,
hand to hand, right hand to the face. Each is first tested for being physically achievable at
all (contact between the two named body groups beginning within one second, from no contact).
For an achievable touch, the same counterfactual (touching order against the hold order, physics
branched one step each way) on the pooled touch token at the onset step; the holding arm's order
is built from the START pose, before the touching arm runs. When the build has a contact head,
BOTH SIDES: from the moment before the onset, the head's probability that an own part touches
each patch of each side (LemLite.predict_contacts) with the touching order and with the hold
order, scored on the patches the world reported own-touched at the onset.

Headline: the counterfactual share of the real visual change at the step the hand enters the
view (floor 0 = the order makes no difference to the prediction).
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
import time
from typing import List, Optional

import numpy as np
import torch

from mimo_bridge.tools import gof_probe_harness as H

PROBE, ITEM = "binding", 9
# Points in front of the left eye for the hand to go to: distance along the eye's axis, offset to
# the eye's right and offset up (cm). A 24-month MIMo arm reaches 42.85 cm from the wrist handle's
# chain root (measured on the crib body, commit 1c6ec9a6f), so every point is inside that reach.
# First smoke run (2026-09-22, body 0 of ckpt_step_344400's episode): with 500 ms and points on the
# axis only, the hand came no nearer than 52 degrees to the eye's axis from 89.5 at the start; the
# search now allows one second and points around the axis as well.
VIEW_POINTS_CM = ((15.0, 0.0, 0.0), (20.0, 0.0, 0.0), (25.0, 0.0, 0.0), (30.0, 0.0, 0.0),
                  (20.0, 5.0, 0.0), (20.0, -5.0, 0.0), (20.0, 0.0, 5.0), (20.0, 0.0, -5.0), (25.0, 5.0, -5.0))
IN_VIEW_DEG = 30.0                  # env.py hand_in_view: the 60-degree wide field's half-angle
VIEW_STEPS = 20                     # four order tenures (1 s)
TOUCH_STEPS = 20                    # one second
TOUCH_KINDS = ("finger_to_finger", "hand_to_hand", "hand_to_face")


def _prefix_bodies(session: H.Session, prefix: str) -> set:
    m = session.env.model
    return {b for b in range(m.nbody) if (m.body(b).name or "").startswith(prefix)}


def _groups(session: H.Session, kind: str):
    if kind == "finger_to_finger":
        return _prefix_bodies(session, "right_th"), _prefix_bodies(session, "right_ff")
    if kind == "hand_to_hand":
        return session.body_ids_under("left_hand"), session.body_ids_under("right_hand")
    if kind == "hand_to_face":
        return session.body_ids_under("right_hand"), {session.env.model.body("head").id}
    raise ValueError(kind)


def _touch_order(session: H.Session, kind: str):
    """The scripted order: position targets for the hand handles involved, full strength on
    them, everything else held."""
    pos, _ = session.pose()
    names = session.handle_names
    o = session.hold_order()
    strength = float(session.dev.get("strength_limit", 1.0))

    def put(handle, target_cm):
        i = names.index(handle)
        o[0, i, 0:3] = np.asarray(target_cm, dtype=np.float32)
        o[0, i, 8] = strength

    if kind == "finger_to_finger":
        th, ix = names.index("thumb_R"), names.index("index_R")
        put("thumb_R", pos[ix])
        put("index_R", pos[th])
    elif kind == "hand_to_hand":
        mid = 0.5 * (pos[names.index("wrist_L")] + pos[names.index("wrist_R")])
        put("wrist_L", mid)
        put("wrist_R", mid)
    elif kind == "hand_to_face":
        put("wrist_R", pos[names.index("head")])
    return o


def _view_order(session: H.Session, dist_cm: float, right_cm: float, up_cm: float = 0.0):
    e = session.env
    cid = e.model.camera("eye_left").id
    R = np.asarray(e.data.cam_xmat[cid]).reshape(3, 3)
    origin = np.asarray(e.data.cam_xpos[cid]) * 100.0
    target = origin + (-R[:, 2]) * dist_cm + R[:, 0] * right_cm + R[:, 1] * up_cm
    o = session.hold_order()
    i = session.handle_names.index("wrist_R")
    o[0, i, 0:3] = target
    o[0, i, 8] = float(session.dev.get("strength_limit", 1.0))
    return o, target


def _hand_view(session: H.Session):
    t = session.frames[0].meta.get("task") or {}
    return t.get("hand_view_deg")


def _dots(pd, ad) -> dict:
    pd, ad = pd.reshape(-1).double(), ad.reshape(-1).double()
    return dict(captured=H.captured_share(pd, ad), dot_pa=float(pd @ ad), dot_aa=float(ad @ ad))


def _patch_state(session: H.Session):
    """The skin's per-patch contact state at the current moment, as the model holds it (its own
    previous-moment record), else the world's last report; None when neither exists."""
    m = session.model
    ps = getattr(m, "_contact_prev_state", None)
    if ps is not None:
        return np.asarray(ps.detach().cpu().numpy() if torch.is_tensor(ps) else ps, dtype=np.int64).reshape(1, -1)
    t = session.frames[0].meta.get("task") or {}
    s = t.get("contact_patch_state")
    return None if s is None else np.asarray(s, dtype=np.int64).reshape(1, -1)


def _stepwise(session: H.Session, snap: dict, order, steps: int, senses, watch=None,
              counterfactual_until: int = 0, contacts_at: Optional[int] = None) -> List:
    """Restore, then at each step: the model's next-moment prediction for `order` from the
    current moment, the step, and the scores of that prediction against what came.

    counterfactual_until: for steps 1..this, also predict for the HOLD order (stay where the body
    is at that step) from the same moment and branch the physics one step with it (restore the
    step's snapshot afterwards and continue with `order`): row["cf"][sense] is the counterfactual
    share. contacts_at: at this step (1-based), read the contact head's per-patch probabilities
    from the moment before it, for `order` and for the hold order."""
    model = session.model
    session.restore(snap)
    rows = []
    for k in range(int(steps)):
        mom = session.moment()
        o_exec = session.exec_order(order)
        pred = H.predict(model, mom, o_exec, want_eff=False)
        row = dict(step=k + 1, share={}, cf={})
        if k + 1 <= int(counterfactual_until) or (contacts_at is not None and k + 1 == int(contacts_at)):
            hold_k = session.hold_order()
            snap_k = session.snapshot()
            if contacts_at is not None and k + 1 == int(contacts_at):
                ps = _patch_state(session)
                row["contacts"] = dict(order=H.predict_contacts(model, mom, o_exec, ps),
                                       hold=H.predict_contacts(model, mom, hold_k, ps),
                                       patch_state_before=None if ps is None else ps.reshape(-1).tolist())
            if k + 1 <= int(counterfactual_until):
                pred_h = H.predict(model, mom, hold_k, want_eff=False)
                session.act(hold_k, new_order=True)
                session.observe()
                r_hold = session.readout()
                session.restore(snap_k)
                row["_hold"] = (pred_h, r_hold)
            del snap_k
        session.act(order, new_order=(k == 0))
        session.observe()
        r = session.readout()
        row["scores"] = H.token_scores(pred["preds"], r["tokens"], mom.tokens, senses)
        for m in senses:
            if m in pred["preds"] and m in r["tokens"]:
                # the continuation share: how far the prediction moved along the real change
                row["share"][m] = _dots(pred["preds"][m] - mom.tokens[m].reshape(1, -1), r["tokens"][m] - mom.tokens[m])
        if "_hold" in row:
            pred_h, r_hold = row.pop("_hold")
            for m in senses:
                if m in pred["preds"] and m in pred_h["preds"] and m in r["tokens"] and m in r_hold["tokens"]:
                    row["cf"][m] = _dots(pred["preds"][m] - pred_h["preds"][m], r["tokens"][m] - r_hold["tokens"][m])
        row["world_task"] = {k2: v for k2, v in (session.frames[0].meta.get("task") or {}).items()
                             if k2 in ("self_contact_onsets", "contact_patch_state_burst", "contact_patch_state")}
        if watch is not None:
            row["watch"] = watch(session)
        rows.append(row)
    return rows


def arm_into_view(session: H.Session, snap: dict, senses) -> dict:
    session.restore(snap)
    start_view = _hand_view(session)
    tried = []
    best = None
    for point in VIEW_POINTS_CM:
        session.restore(snap)
        order, target = _view_order(session, *point)
        views = session.physics_only(snap, order, VIEW_STEPS, watch=_hand_view)
        first = next((i + 1 for i, v in enumerate(views) if v is not None and v < IN_VIEW_DEG), None)
        vals = [v for v in views if v is not None]
        mn = min(vals) if vals else None
        tried.append(dict(point_cm=list(point), min_hand_view_deg=mn, first_in_view_step=first))
        if mn is not None and (best is None or mn < best[0]):
            best = (mn, point, first)
    out = dict(start_hand_view_deg=start_view, search=tried, in_view_deg=IN_VIEW_DEG)
    if best is None or best[0] >= IN_VIEW_DEG:
        out["status"] = "not achievable from this moment: no searched point brought the hand within %.0f degrees" % IN_VIEW_DEG
        return out
    if start_view is not None and start_view < IN_VIEW_DEG:
        out["note"] = "the hand was already in view at the start; 'entering' is the first step inside the field"
    session.restore(snap)
    order, target = _view_order(session, *best[1])
    session.restore(snap)
    mom0 = session.moment()
    start = session.readout()
    hold = session.hold_order()                    # built from the START pose, before any arm runs
    pred0 = H.predict(session.model, mom0, order, want_eff=False)
    pred_hold = H.predict(session.model, mom0, hold, want_eff=False)
    moving = _stepwise(session, snap, order, VIEW_STEPS, senses, watch=_hand_view,
                       counterfactual_until=int(best[2] or 0))
    held = _stepwise(session, snap, hold, VIEW_STEPS, senses, watch=_hand_view)
    wi = session.handle_names.index("wrist_R")
    entering = next((r for r in moving if r["watch"] is not None and r["watch"] < IN_VIEW_DEG), None)
    k_enter = entering["step"] if entering else None
    held_same = held[k_enter - 1] if k_enter else None
    first_tenure_end = min(session.hold, len(moving))
    # the tenure head's predicted 250 ms move of the hand, against the real move over the first tenure
    rs = session.run_arm(snap, order, steps=first_tenure_end)
    real_move = np.asarray(rs[-1]["pos_cm"][wi]) - np.asarray(start["pos_cm"][wi])
    tenure = {}
    if "tenure_cm" in pred0:
        pm = np.asarray(pred0["tenure_cm"][wi])
        tenure = dict(predicted_hand_move_cm=pm.tolist(), real_hand_move_cm=real_move.tolist(),
                      error_cm=float(np.linalg.norm(pm - real_move)),
                      do_nothing_error_cm=float(np.linalg.norm(real_move)),
                      predicted_vs_hold_cm=float(np.linalg.norm(pm - np.asarray(pred_hold["tenure_cm"][wi]))))
    slim = lambda r: None if r is None else {k: v for k, v in r.items() if k not in ("world_task",)}
    out.update(status="measured", point_cm=list(best[1]), target_cm=target.tolist(), entering_step=k_enter,
               entering_step_physics_search=best[2],
               entering_ms=(k_enter * session.clock.model_step_seconds * 1000.0) if k_enter else None,
               at_entering=dict(moving=slim(entering), holding=slim(held_same)),
               counterfactual_at_entering=(entering or {}).get("cf") or None,
               pooled_counterfactual={m: H.pooled_share([r["cf"].get(m) for r in moving if r.get("cf")], "dot_pa", "dot_aa")
                                      for m in senses},
               pooled_share={m: H.pooled_share([r["share"].get(m) for r in moving], "dot_pa", "dot_aa") for m in senses},
               pooled_share_holding={m: H.pooled_share([r["share"].get(m) for r in held], "dot_pa", "dot_aa") for m in senses},
               tenure_head=tenure or H.absent("tenure_pose_head"),
               steps=[dict(step=r["step"], hand_view_deg=r["watch"],
                           vision_counterfactual=(r.get("cf") or {}).get("vision", {}).get("captured"),
                           vision_continuation=r["share"].get("vision", {}).get("captured"),
                           proprio_continuation=r["share"].get("proprioception", {}).get("captured")) for r in moving])
    return out


def _side_patches(session: H.Session, group: set, patch_names) -> List[int]:
    names = {session.env.model.body(b).name for b in group}
    return [i for i, n in enumerate(patch_names) if n in names]


def _both_sides(session: H.Session, row: Optional[dict], ga: set, gb: set) -> dict:
    """At the onset: each side's own-part probability (contact head) with the touching order and
    with the hold order, on the patches of that side the world reported touched by an own part
    in the onset's burst."""
    if not row or not row.get("contacts"):
        return dict(status="unmeasured: no contact prediction at the onset step")
    c = row["contacts"]
    if not c.get("order") or not c.get("hold"):
        return dict(status="unmeasured: the contact head returned nothing (no skin report at that moment)")
    names = list(c["order"].get("patch_names") or [])
    state = (row.get("world_task") or {}).get("contact_patch_state_burst") or \
        (row.get("world_task") or {}).get("contact_patch_state")
    if state is None or len(state) != len(names):
        return dict(status="unmeasured: the world gave no per-patch report of the onset")
    own = (np.asarray(state, dtype=np.int64) & 2) > 0
    before = c.get("patch_state_before")
    fresh = own if before is None else own & ~((np.asarray(before, dtype=np.int64) & 2) > 0)
    sides = {}
    for side, group in (("a", ga), ("b", gb)):
        patches = _side_patches(session, group, names)
        hit = [p for p in patches if fresh[p]]
        if not patches:
            sides[side] = dict(status="unmeasured: no skin patch on this side")
            continue
        if not hit:
            sides[side] = dict(patches=[names[p] for p in patches],
                               status="the world reported no new own-part touch on this side's patches")
            continue
        po = float(np.mean([c["order"]["p_self"][p] for p in hit]))
        ph = float(np.mean([c["hold"]["p_self"][p] for p in hit]))
        # a difference float32 cannot resolve is a tie, never foresight (verify-fix-probes 2026-09-23:
        # a bare po > ph let a head whose answer does not depend on the order score a side on
        # rounding; the gaze pairs' finding in the review of the probes is the same class)
        verdict = H.better_of("order", po, "hold", ph)
        sides[side] = dict(patches_touched=[names[p] for p in hit], p_own_touch_order=po, p_own_touch_hold=ph,
                           difference_per_100=100.0 * (po - ph),
                           order_above_hold=(None if verdict == "tie" else verdict == "order"))
    scored = [s for s in sides.values() if "p_own_touch_order" in s]
    return dict(sides=sides, sides_scored=len(scored),
                sides_tied=sum(1 for s in scored if s["order_above_hold"] is None),
                both_sides_foreseen=(None if len(scored) < 2 else all(s["order_above_hold"] is True for s in scored)),
                rule="a side counts as foreseen when the touching order's own-part probability on the side's newly "
                     "touched patches is above the hold order's by more than float32 can resolve (a tie is not "
                     "foreseen), both read from the moment before the onset; a head whose answer does not depend "
                     "on the order ties on every side and scores no onset, and a coin flip for each side would "
                     "score both sides on 1 of every 4 onsets")


def self_touch(session: H.Session, snap: dict, senses, comps) -> dict:
    out = {}
    for kind in TOUCH_KINDS:
        session.restore(snap)
        ga, gb = _groups(session, kind)
        if not ga or not gb:
            out[kind] = dict(status="unmeasured: the body groups were not found in this body")
            continue
        start_force = session.contact_between(ga, gb)
        # the holding arm's order is the START pose, built before the touching arm runs (review of
        # the probes, 2026-09-23: built after it, it ordered the body to the touching pose)
        hold = session.hold_order()
        order = _touch_order(session, kind)
        forces = session.physics_only(snap, order, TOUCH_STEPS, watch=lambda s: s.contact_between(ga, gb))
        onset = next((i + 1 for i, f in enumerate(forces) if f > 0.0), None)
        rec = dict(start_contact_n=start_force, max_contact_n=float(max(forces)) if forces else 0.0,
                   onset_step=onset, onset_ms=(onset * session.clock.model_step_seconds * 1000.0) if onset else None,
                   achievable=bool(onset is not None and start_force == 0.0))
        if start_force > 0.0:
            rec["status"] = "already touching at the start; no onset to predict from this moment"
            out[kind] = rec
            continue
        if onset is None:
            rec["status"] = "not achievable within %d ms from this moment by this scripted order" % int(
                TOUCH_STEPS * session.clock.model_step_seconds * 1000)
            out[kind] = rec
            continue
        has_head = comps["contact_head"]["present"]
        moving = _stepwise(session, snap, order, onset + 1, senses, watch=lambda s: s.contact_between(ga, gb),
                           counterfactual_until=onset, contacts_at=(onset if has_head else None))
        held = _stepwise(session, snap, hold, onset + 1, senses)
        at = next((r for r in moving if r["watch"] and r["watch"] > 0.0), None)
        k = at["step"] if at else None
        rec.update(status="measured", hold_order_is_start_pose=True,
                   touch_at_onset=dict(moving=(at or {}).get("scores", {}).get("tactile"),
                                       counterfactual=(at or {}).get("cf", {}).get("tactile"),
                                       moving_share=(at or {}).get("share", {}).get("tactile"),
                                       holding=held[k - 1]["scores"].get("tactile") if k else None,
                                       holding_share=held[k - 1]["share"].get("tactile") if k else None),
                   both_sides=(H.absent("contact_head") if not has_head else _both_sides(session, at, ga, gb)),
                   scored_instead=(None if has_head else "the pooled touch token (one vector for the whole skin)"))
        out[kind] = rec
    return out


def lived_sentence(n_views: int, n_states: int, value, continuation, touch_cf, both) -> str:
    s = ("The right hand was sent into view in %d of %d start moments. At the step it entered the eyes' field "
         "the model's next-moment prediction for continuing the order, against its prediction for stopping the "
         "hand where it was, moved %s of the way the real picture differed between continuing and stopping "
         "(0%% = the prediction does not depend on the order, 100%% = it foresees the whole difference). "
         "Measured the first way - how far the prediction moved along the real change, which a model that only "
         "extends the hand's motion also scores - it read %s. Distance to goal unmeasured."
         % (n_views, n_states, H.share_text(value), H.share_text(continuation)))
    if touch_cf is not None:
        s += (" At the onset of a scripted self-touch, the touch prediction for the touching order against "
              "holding still moved %s of the way the real touch differed." % H.share_text(touch_cf))
    if both is not None:
        s += (" Both sides of a self-touch foreseen above the hold order (contact head): %d of %d onsets. A head "
              "whose answer does not depend on the order scores none of them, a coin flip for each side would "
              "score 1 of every 4, and a head that foresees every touch scores all of them." % (both[0], both[1]))
    return s


def run(ckpt, *, states: int = 2, warm: int = 300, seed: int = 31, write: bool = True, root=None) -> dict:
    t0 = time.time()
    H.reset_routes()
    frozen = H.load_frozen(ckpt)
    senses = H.real_senses(frozen.model)
    comps = H.components(frozen.model)
    per_state, cfg = [], None
    for session, snap in H.start_states(frozen, states, warm=warm, seed=seed, gaze_reflex=False, saccades=False):
        if cfg is None:
            cfg = H.frozen_config(frozen, session)
        per_state.append(dict(start=session.start, arm_into_view=arm_into_view(session, snap, senses),
                              self_touch=self_touch(session, snap, senses, comps)))
    views = [s["arm_into_view"] for s in per_state if s["arm_into_view"].get("status") == "measured"]
    enter_cf = [(v.get("counterfactual_at_entering") or {}).get("vision") for v in views]
    value = H.pooled_share(enter_cf, "dot_pa", "dot_aa")
    enter_vision = [v["at_entering"]["moving"]["share"].get("vision") for v in views
                    if v.get("at_entering", {}).get("moving")]
    continuation = H.pooled_share(enter_vision, "dot_pa", "dot_aa")
    hold_vision = [v["at_entering"]["holding"]["share"].get("vision") for v in views
                   if v.get("at_entering", {}).get("holding")]
    touch = {k: dict(achievable=sum(1 for s in per_state if s["self_touch"].get(k, {}).get("achievable")),
                     of=len(per_state)) for k in TOUCH_KINDS}
    onsets = [s["self_touch"][k] for s in per_state for k in TOUCH_KINDS
              if s["self_touch"].get(k, {}).get("status") == "measured"]
    touch_cf = H.pooled_share([o["touch_at_onset"].get("counterfactual") for o in onsets], "dot_pa", "dot_aa")
    both = None
    if comps["contact_head"]["present"]:
        judged = [o["both_sides"].get("both_sides_foreseen") for o in onsets
                  if isinstance(o.get("both_sides"), dict) and o["both_sides"].get("both_sides_foreseen") is not None]
        both = (int(sum(judged)), len(judged))
    lived = lived_sentence(len(views), len(per_state), value, continuation, touch_cf, both)
    findings = dict(arm_into_view_achievable=dict(n=len(views), of=len(per_state)),
                    vision_counterfactual_entering=value,
                    vision_share_entering=continuation,
                    vision_share_entering_note=("the continuation share (prediction moved along the real change); "
                                                "a model that ignores the order and extends the motion scores on it"),
                    vision_share_entering_holding=H.pooled_share(hold_vision, "dot_pa", "dot_aa"),
                    vision_counterfactual_pooled_steps=H.mean_or_none(
                        v["pooled_counterfactual"].get("vision") for v in views),
                    self_touch_achievable=touch,
                    contact_head=None if comps["contact_head"]["present"] else H.absent("contact_head"),
                    both_sides=(None if both is None else dict(foreseen=both[0], judged=both[1])))
    findings["touch_counterfactual_at_onset"] = touch_cf
    findings["touch_share_at_onset"] = H.pooled_share(
        [o["touch_at_onset"].get("moving_share") for o in onsets], "dot_pa", "dot_aa")
    findings["touch_share_at_onset_holding"] = H.pooled_share(
        [o["touch_at_onset"].get("holding_share") for o in onsets], "dot_pa", "dot_aa")
    findings["touch_onsets_measured"] = len(onsets)
    findings["proprio_share_moving"] = H.mean_or_none(v["pooled_share"].get("proprioception") for v in views)
    findings["tenure_hand_error_cm"] = H.mean_or_none(v["tenure_head"].get("error_cm") for v in views
                                                      if isinstance(v.get("tenure_head"), dict))
    findings["tenure_hand_do_nothing_error_cm"] = H.mean_or_none(v["tenure_head"].get("do_nothing_error_cm") for v in views
                                                                 if isinstance(v.get("tenure_head"), dict))
    body = dict(config=dict(cfg or {}, states=states, seed=seed, view_points_cm=VIEW_POINTS_CM,
                            in_view_deg=IN_VIEW_DEG, view_steps=VIEW_STEPS, touch_steps=TOUCH_STEPS,
                            route=H.route_record()),
                arms=dict(states=per_state),
                controls=dict(counterfactual=("every scored step is predicted twice from the same moment, for the "
                                              "order and for holding still, and the physics is branched one step "
                                              "each way; the headline is the share of that real difference foreseen"),
                              holding="every measured step is also compared with the same step while the body holds still",
                              persistence="each sense's 'nothing changes' error is in every step's scores"),
                findings=findings,
                headline=H.headline("binding_vision_counterfactual_hand_entering_view", value,
                                    "share of the difference in what the eyes saw between continuing the order and "
                                    "stopping the hand, at the step the hand entered the view, that the prediction "
                                    "foresaw: from none (the order made no difference to it) to all", 0.0, None, lived),
                status="measured" if value is not None else "partial: the hand was not brought into view",
                elapsed_s=round(time.time() - t0, 1))
    if write:
        body["path"] = str(H.write_result(frozen.info, PROBE, ITEM, body, root=root))
    return body


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--states", type=int, default=2)
    ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args(argv)
    out = run(a.ckpt, states=a.states, seed=a.seed)
    print(json.dumps(dict(headline=out["headline"], path=out.get("path"), elapsed_s=out["elapsed_s"]), indent=1))


if __name__ == "__main__":
    main()
