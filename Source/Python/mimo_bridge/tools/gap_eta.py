# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The plan-gap clock: how far plans get, whether the gap is closing, when it would
arrive at the present learning rate, and what - if anything - could close it faster.

Operator order 2026-09-06. Everything here is log analytics (no model, no env, no
GPU): safe inside the overseer's loop, run at most hourly. Output:
  Saved/mimo_bridge/gap_eta.json          latest reading (dashboard reads this)
  Saved/mimo_bridge/gap_eta_history.jsonl one line per run, the clock's own record

The gap: each plan's ending distance to the goal it chose, in mean-over-senses token
error - anchored by scale_anchors.json (one instant of world-change, two unrelated
moments, the arrived line). The ladder below names where we stand in lived terms;
the ETA extrapolates the trend of per-bucket medians (Theil-Sen, quartile range);
the analysis ranks measured causes and names which levers are armed vs operator
decisions. A positive trend prints "not closing at the present rate" - that is a
finding, not a failure of the clock.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAVED = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "Saved", "mimo_bridge")
EXP = os.path.join(SAVED, "experiments")

# the ten rungs, by gap as % of the distance between two unrelated moments.
# Band 3 keeps the operator's own sentence at the position it was coined for.
LADDER = [
    (95.0, "the plan's endpoint and its goal are as unrelated as two random moments - "
           "planning is aiming at something it cannot reach at all"),
    (85.0, "the goal might as well be somewhere else entirely; plans close only a "
           "few percent of the distance (0% = never moved, 100% = arrived)"),
    (70.0, "the goal remains almost as far away as a completely different moment of "
           "experience - the planner nudges, it does not travel"),
    (55.0, "real travel has begun: plans cover a visible fraction of the way, but the "
           "destination is still another scene"),
    (40.0, "plans get about halfway to what they want - the destination is "
           "recognizable from where they stop"),
    (25.0, "most of the journey happens; plans stop in the goal's neighborhood"),
    (15.0, "plans arrive next door to their goal - the remaining gap is a few seconds "
           "of world-change"),
    (8.0,  "plans stop within arm's reach of their goal; fine positioning is all "
           "that is missing"),
    (3.0,  "plans essentially arrive - the gap is a couple of instants of "
           "world-change, polish rather than distance"),
    (0.0,  "plans ARRIVE: the chosen state and the reached state are the same moment "
           "for practical purposes - deliberate multi-step action is real"),
]


# Measured era boundaries (step, what changed) - behavior-changing fixes/config
# landed at these steps; a trend fit ACROSS one measures the change, not learning.
# Append when a semantics-changing commit goes live (dated from replay logs).
ERAS = [
    (3103950, "bee-lock fix (critters alive again)"),
    (3111775, "reflex sign + license-clamp + planner-context fixes: the first real "
              "multi-step planner - no earlier point describes this planner"),
    (3140775, "touch slip fix (touch stream statistics changed)"),
    (3157525, "imagination chain 3->6 steps, weight 0.5->0.8 (hail-mary attempt: "
              "chain-loss units and plan horizon changed here)"),
]
PLANNER_ERA = 3111775      # final_mse describes a multi-step planner only from here
CHAIN_ERA = 3157525        # imagination_loss units are only comparable from here
MIN_TREND_BUCKETS = 6      # fewer bucket-medians than this = no statable rate


def _theil_sen(xs, ys):
    """Median pairwise slope + quartiles (a robust trend with its own range)."""
    slopes = [(ys[j] - ys[i]) / (xs[j] - xs[i])
              for i in range(len(xs)) for j in range(i + 1, len(xs)) if xs[j] > xs[i]]
    if not slopes:
        return None, None, None
    return (float(np.median(slopes)), float(np.percentile(slopes, 25)),
            float(np.percentile(slopes, 75)))


# Discarded attempt BRANCHES (reverted hail-maries / operator attempts). Their step
# ranges OVERLAP the mainline's (each restarted from a mainline snapshot), so a
# step-sorted sweep interleaves two different brains and the trend reads the
# discarded semantics as learning - measured 2026-09-06 late: with _r included the
# clock reported rung 6 / goals-at-35% off the reverted locality brain while the
# live brain read rung 2 / 87%. Append here whenever an attempt is ledgered
# discarded (the ledger entry is the record; this list is the filter).
DISCARDED_BRANCHES = {"H67_gaze_world_p", "H67_gaze_world_r",
                      "H67_gaze_world_w"}   # _w: 8 rag-doll minutes, discarded


def _collect(prefix="H67_gaze_world"):
    rows, aux = [], []
    for run in sorted(glob.glob(os.path.join(EXP, prefix + "*"))):
        if os.path.basename(run) in DISCARDED_BRANCHES:
            continue
        p = os.path.join(run, "replay_training_stats.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8", errors="replace"):
            if '"mimo"' not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            it = r.get("intention") or {}
            if it.get("final_mse") is not None and it["final_mse"] == it["final_mse"]:
                rows.append((int(r["step"]), float(it["final_mse"]),
                             it.get("gap_closed_pct"), it.get("start_mse")))
            aux.append((int(r["step"]), r.get("imagination_loss"),
                        r.get("task_hand_fov"),
                        (r.get("agency") or {}).get("vis_adv"),
                        r.get("model_steps_per_s"),
                        (r.get("plan_license") or {}).get("win_emas")))
    rows.sort(); aux.sort()
    return rows, aux


def _bucket_med(pairs, width=2000):
    out = {}
    for s, v in pairs:
        out.setdefault(s // width, []).append(v)
    return [(k * width + width / 2, float(np.median(v))) for k, v in sorted(out.items())]


def compute(prefix="H67_gaze_world"):
    anchors = json.load(open(os.path.join(SAVED, "scale_anchors.json")))
    one, far, arrived = (anchors["one_instant"], anchors["half_minute_apart"],
                         anchors["goal_arrival"])
    rows, aux = _collect(prefix)
    if len(rows) < 30:
        return {"error": f"only {len(rows)} plan records - not enough history"}
    # era discipline: the trend is fit ONLY inside the multi-step-planner era -
    # before PLANNER_ERA a clamp bug held every plan at depth 1, so earlier points
    # describe a different planner and a fit across them measures the fix, not
    # learning. Boundaries crossed inside the fit window are named in the output.
    b_all = _bucket_med([(s, v) for s, v, _g, _s0 in rows])
    b = [(x, y) for x, y in b_all if x >= PLANNER_ERA][-40:]
    gap = float(np.median([v for _s, v in (b if b else b_all)[-3:]]))
    sps = float(np.median([a[4] for a in aux[-50:] if a[4]])) or 1.0
    era_hours = ((rows[-1][0] - PLANNER_ERA) / sps / 3600.0) if rows else 0.0
    inner = [f"step {s}: {w}" for s, w in ERAS if b and b[0][0] < s < b[-1][0]]
    if len(b) >= MIN_TREND_BUCKETS:
        xs, ys = [x for x, _ in b], [y for _, y in b]
        slope, s25, s75 = _theil_sen(xs, ys)      # mse per step (negative = closing)
    else:
        slope = s25 = s75 = None

    def eta_days(sl):
        if sl is None or sl >= 0:
            return None
        return ((gap - arrived) / -sl) / sps / 86400.0

    eta_med, eta_fast, eta_slow = eta_days(slope), eta_days(s25), eta_days(s75)
    gap_pct = 100.0 * gap / far
    rung_i, rung = next((i, t) for i, (lo, t) in enumerate(LADDER) if gap_pct >= lo)
    trend_100k = slope * 1e5 if slope is not None else None
    if len(b) < MIN_TREND_BUCKETS:
        eta_txt = (f"no statable rate yet: the multi-step planner is only "
                   f"{era_hours:.0f} hours old ({len(b)} trend buckets of the "
                   f"{MIN_TREND_BUCKETS} needed) - a rate becomes statable after "
                   f"~{(MIN_TREND_BUCKETS - len(b)) * 2000 / sps / 3600.0:.0f} more "
                   f"hours of training")
    elif eta_med is not None:
        eta_txt = (f"at the present learning rate the gap NUMBER closes in roughly "
                   f"{eta_med:.0f} days (anywhere between {eta_fast:.0f} days and "
                   f"{'never' if eta_slow is None else f'{eta_slow:.0f} days'}, from "
                   f"the trend's own spread) - but this extrapolates the distance "
                   f"number only: completing journeys is separately capped by how "
                   f"far goals are CHOSEN (the one behavior-carrying constant; the "
                   f"plan-trim rule and the arrival line shape reporting only, since "
                   f"just the first action of any plan executes - adversarial review "
                   f"2026-09-06 eve) and by imagination blur, so this is NOT an "
                   f"arrival date")
    else:
        eta_txt = (f"NOT closing at the present rate (trend "
                   f"{trend_100k:+.3f} per 100K steps; a negative trend is required)")
    if inner:
        eta_txt += ("; CAUTION - semantics changed inside the fit window ("
                    + "; ".join(inner) + "), so part of this trend is those changes")

    # ---- the hourly analysis: is there something that can be done? Measured causes
    # ranked; every lever labeled armed (already training) or operator decision.
    an = []
    isl, iml = None, None
    # chain-loss units changed at CHAIN_ERA (3->6 steps: a deeper chain is
    # mechanically larger) - only same-era points may be trended together
    im = _bucket_med([(a[0], a[1]) for a in aux
                      if a[1] and a[0] >= CHAIN_ERA])[-40:]
    if len(im) >= MIN_TREND_BUCKETS:
        isl, _q1, _q3 = _theil_sen([x for x, _ in im], [y for _, y in im])
        iml = im[-1][1]
        if isl is not None and isl >= 0:
            an.append(f"imagination chain error (how far the imagined future lands "
                      f"from the real one) is not improving ({iml:.2f}, trend "
                      f"{isl*1e5:+.3f} per 100K steps) - plans go blind however deep "
                      f"they look; an imagination depth/weight change is an OPERATOR "
                      f"DECISION")
        else:
            an.append(f"imagination chain error (how far the imagined future lands "
                      f"from the real one) is improving ({iml:.2f}, "
                      f"{isl*1e5:+.3f} per 100K steps) - the imagination training "
                      f"lesson is working on it")
    elif im:
        an.append(f"the 6-step imagination chain is {len(im)*2000/sps/3600.0:.0f} "
                  f"hours old - too young to trend ({len(im)} buckets of "
                  f"{MIN_TREND_BUCKETS} needed); the hail-mary hour-3 criterion is "
                  f"the active judge of whether depth is learnable")
    g0 = [s0 for _s, _v, _g, s0 in rows[-200:] if s0 is not None and s0 == s0]
    if g0:
        st = float(np.median(g0))
        an.append(f"goals are CHOSEN {100*st/far:.0f}% of the unrelated-moments "
                  f"distance from the start ({st/one:.0f} instants away) - "
                  + ("teleport-class targets; making goal choice favor nearer felt "
                     "states is an OPERATOR DECISION (goal-ring locality)"
                     if st >= 0.8 * far else "within reachable-looking range"))
    else:
        an.append("how far goals are CHOSEN from the start becomes measurable from "
                  "the first records after the current relaunch (start distance is "
                  "now recorded per plan)")
    hf = [a[2] for a in aux[-30:] if a[2] is not None]
    va = [a[3] for a in aux[-30:] if a[3] is not None]
    if hf:
        an.append(f"eyes-on-own-hand {float(np.median(hf)):.3f} (success line 0.03, "
                  f"wander-only 0.01-0.02): imagination can only learn hand "
                  f"consequences it watches, and this feeds rung 2 (the eyes "
                  f"recognizing the model's own hand); a stronger hand-dwell pull is "
                  f"QUEUED as the next attempt after the current one's verdict"
                  + (f"; vision self-advantage {float(np.median(va)):.4f} against its "
                     f"0.011 noise" if va else ""))
    win = next((a[5] for a in reversed(aux) if a[5]), None)
    if win:
        an.append(f"each imagined step beats freeze-frame persistence "
                  f"({'/'.join(f'{w:.2f}' for w in win)} vs the 0.50 line) - an "
                  f"ADMISSION GATE only: the freeze-frame baseline gets easier the "
                  f"deeper the step (the world drifts from any still frame), so these "
                  f"wins license depth but do not measure aim; journeys still do not "
                  f"complete")

    # ---- the hail mary (operator 2026-09-06): given the prevailing conditions, the
    # boldest intervention with a measured basis - its risk, the mitigation, and a
    # verification clock. ELIGIBLE only when its declared success metric can answer
    # within ~3 hours; an attempted hail mary that verifies is KEPT, one that fails
    # is REVERTED to the pre-attempt snapshot and ledgered (never looped). Bold moves
    # that change behavior semantics are surfaced as OPERATOR DECISIONS, not
    # auto-attempted.
    ledger_p = os.path.join(SAVED, "hail_mary_ledger.jsonl")
    tried, in_flight = set(), {}
    if os.path.exists(ledger_p):
        for line in open(ledger_p, encoding="utf-8", errors="replace"):
            try:
                e = json.loads(line)
                if e.get("outcome") in ("discarded", "attempting"):
                    tried.add(e.get("name"))
                # the LAST entry for a name decides whether it is still in flight
                if e.get("name"):
                    in_flight[e["name"]] = e.get("outcome") == "attempting"
            except ValueError:
                pass
    live_attempt = next((n for n, v in in_flight.items() if v), None)
    blur_binding = len(im) >= MIN_TREND_BUCKETS and isl is not None and isl >= 0
    hm = None
    if live_attempt is not None:
        # ONE attempt at a time (operator doctrine): while one is unresolved,
        # nothing else is eligible - the queue waits for the verdict
        hm = {"name": live_attempt, "eligible": False,
              "what": f"attempt IN FLIGHT: {live_attempt} - its declared criterion "
                      f"decides kept-or-reverted before anything else runs",
              "risk": "attempting two levers at once makes neither verifiable",
              "mitigation": "an auto-revert criterion is set: at its declared hour "
                            "the attempt is kept or reverted per "
                            "hail_mary_ledger.jsonl, and the queue resumes after "
                            "that decision",
              "verify": "see the ledger's declared_success/declared_failure for "
                        "this attempt", "verify_hours": None}
    elif blur_binding and "imagination_depth" not in tried:
        hm = {"name": "imagination_depth",
              "what": "raise the imagination chain from 3 to 6 orders and its training "
                      "weight from 0.5 to 0.8 - attack the aiming blur directly, since "
                      "it is measured as the binding wall and nothing armed trains it "
                      "harder",
              "risk": "chained self-prediction can destabilize training (each imagined "
                      "step feeds the next; errors compound - the telephone-game "
                      "failure), and the extra loss weight can crowd other lessons",
              "mitigation": "snapshot before the attempt (revert costs ~1 min and zero "
                            "steps); the standing collapse and obedience kill-teeth "
                            "stay armed; auto-revert on verification failure",
              "verify": f"multi-step imagination error's per-bucket median must sit "
                        f"below its pre-attempt level ({iml:.2f}) by hour 3 of the "
                        f"attempt, else revert",
              "verify_hours": 3, "eligible": True}
    elif hf and float(np.median(hf)) < 0.03 and "dwell_boost" not in tried:
        hm = {"name": "dwell_boost",
              "what": "strengthen the curiosity pull that holds the fovea on the "
                      "hand (the rung-2 supply line is under its 0.03 success line)",
              "risk": "a stronger gaze force once broke command-following in an "
                      "earlier era",
              "mitigation": "the STOP RULE on bout obedience is armed (follow_bout "
                            "below -0.038 two hourly checks running stops the run; "
                            "-0.038 = its measured mean 0.006 minus 3x its 0.015 "
                            "spread); snapshot before, auto-revert on failure",
              "verify": "eyes-on-hand above 0.03 within 3 hours, else revert",
              "verify_hours": 3, "eligible": True}
    if hm is None:
        hm = {"name": "goal_locality", "eligible": False,
              "what": "make goal choice favor NEARER felt-good states over the "
                      "globally best-felt (goals are currently chosen unrelated-"
                      "moment far, so no local skill can arrive)",
              "risk": "changes what the planner wants, not just how it gets there - "
                      "a behavior-semantics change against the spec's felt-best rule",
              "mitigation": "none that preserves the spec unreviewed",
              "verify": "OPERATOR DECISION - not auto-attempted at any verify speed",
              "verify_hours": None}

    # ---- sanity section (operator 2026-09-06): the standing question ledger plus
    # the freshest motion census. Resolved questions are not re-asked.
    sanity = {"open_questions": [], "census": None}
    try:
        ql = json.load(open(os.path.join(SAVED, "overseer_questions.json")))
        sanity["open_questions"] = [
            {"id": q["id"], "question": q["question"],
             "resolves_by": q.get("resolves_by")}
            for q in ql.get("questions", [])
            if q.get("status") in ("open", "in_work")]
    except Exception:
        pass
    try:
        mc = json.load(open(os.path.join(SAVED, "motion_census.json")))
        sanity["census"] = {"computed_at": mc.get("computed_at"),
                            "verdicts": mc.get("verdicts"),
                            "goal_service": mc.get("goal_service")}
    except Exception:
        pass

    # ---- the fourteen-step question (operator 2026-09-06): how long until the
    # planner completes a fourteen-instant journey? Assembled from the latest
    # depth_ladder_probe results (forced-probe measurements on frozen checkpoints -
    # license unearned at depths beyond the flag) plus the live license state.
    # The first fact is a constant, not a rate: lookahead = min(1 + earned license,
    # the --imagination-steps flag). No learning crosses a config cap.
    fourteen = {"status": "unmeasured - run depth_ladder_probe on a checkpoint"}
    probes = sorted(glob.glob(os.path.join(SAVED, "depth_ladder", "*.json")),
                    key=os.path.getmtime)
    if probes:
        try:
            pr = json.load(open(probes[-1]))
            fid = pr.get("fidelity", {})
            disp = {int(k): v for k, v in pr.get("displacement_curve", {}).items()
                    if v is not None}
            jn = pr.get("journey", {})
            anc = pr.get("anchors_this_ckpt", {})
            oi = anc.get("one_instant_raw")
            # invert the gap through the measured curve: how many steps of
            # continuous world-change equal the current gap? (token distance
            # saturates - linear division by one_instant overstates far distances)
            gap_steps = next((k for k in sorted(disp) if disp[k] >= gap), None)
            # the curve's highest point and the DEPTH IT OCCURS AT (argmax, not the
            # deepest k - the curve is noisy near its top)
            curve_top_k = (max(disp, key=disp.get) if disp else None)
            curve_top = (disp[curve_top_k] if disp else None)
            d14 = fid.get("14") or {}
            d1 = fid.get("1") or {}
            lic = len([w for w in (win or []) if w > 0.5])
            fourteen = {
                "status": "measured",
                "probe": os.path.basename(probes[-1]),
                "gate0_config_ceiling": (
                    f"lookahead is capped at the --imagination-steps flag "
                    f"(a constant): no learning rate reaches 14 until the operator "
                    f"raises it; the license has earned {lic} of the offered depths"),
                "license_speed": (
                    "when depth is offered and fidelity supports it, the license "
                    "earns it within hours (all 6 offered depths re-earned in under "
                    "4K steps after the deepening resume)"),
                "gap_in_worldchange_steps": gap_steps,
                "gap_beyond_curve": (gap_steps is None and curve_top is not None
                                     and gap > curve_top),
                "curve_top": curve_top, "curve_top_k": curve_top_k,
                "depth14_chain_err": d14.get("err_anchor_median"),
                "depth14_vs_prev_frame": (
                    None if not (d14.get("err_anchor_median")
                                 and d14.get("prev_frame_anchor_median"))
                    else d14["err_anchor_median"] / d14["prev_frame_anchor_median"]),
                "depth1_chain_err": d1.get("err_anchor_median"),
                "one_instant_this_ckpt": oi,
                "goal_persistence_events": jn.get("goal_persistence_events_median"),
                "closest_approach_median": jn.get("closest_approach_median"),
                "reached_code_line_pct": jn.get("reached_code_line_pct"),
                "executed_travel_per_event": jn.get(
                    "executed_travel_per_event_median")}
        except (ValueError, OSError, KeyError) as e:
            fourteen = {"status": f"probe unreadable: {e}"}

    out = {"computed_at": time.strftime("%Y-%m-%d %H:%M"),
           "gap": gap, "gap_instants": gap / one, "gap_pct_of_unrelated": gap_pct,
           "rung": rung_i + 1, "rung_text": rung, "trend_per_100k_steps": trend_100k,
           "eta_days_median": eta_med, "eta_days_range": [eta_fast, eta_slow],
           "eta_text": eta_txt, "analysis": an, "hail_mary": hm, "sanity": sanity,
           "planner_era_hours": era_hours, "fourteen_step": fourteen,
           "n_plan_records": len(rows), "anchors": anchors}
    # atomic write (the dashboard re-reads this file every 30 s; a mid-write read
    # would silently blank the whole panel for one refresh), and refuse nonfinite
    # values instead of poisoning the file with NaN tokens strict parsers reject
    _tmp = os.path.join(SAVED, "gap_eta.json.tmp")
    with open(_tmp, "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    os.replace(_tmp, os.path.join(SAVED, "gap_eta.json"))
    with open(os.path.join(SAVED, "gap_eta_history.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({k: out[k] for k in
                            ("computed_at", "gap", "gap_pct_of_unrelated", "rung",
                             "trend_per_100k_steps", "eta_days_median")}) + "\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="H67_gaze_world")
    a = ap.parse_args(argv)
    out = compute(a.prefix)
    if "error" in out:
        print(out["error"]); return 1
    print(f"rung {out['rung']}/10: {out['rung_text']}")
    print(f"gap {out['gap']:.3f} = {out['gap_instants']:.0f} instants of world-change "
          f"= {out['gap_pct_of_unrelated']:.0f}% of the unrelated-moments distance")
    print(out["eta_text"])
    for line in out["analysis"]:
        print(" - " + line)
    hm = out["hail_mary"]
    print(f"HAIL MARY [{'ELIGIBLE' if hm.get('eligible') else 'operator decision'}] "
          f"{hm['name']}: {hm['what']}")
    print(f"  risk: {hm['risk']}  |  mitigation: {hm['mitigation']}")
    print(f"  verify: {hm['verify']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
