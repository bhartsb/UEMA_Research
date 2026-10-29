# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The checkpoint probes: the same requested movements tried at each checkpoint, as charts over dates and as pictures.

THE SCHEDULED PROBE (ability recheck body_and_synergy, docs/gain_of_function_v3/compare_hand_body_goals_2026-09-24.py,
outputs Saved/mimo_bridge/ability_rechecks/body_and_synergy_*): at a checkpoint, from each of the 8 bodies' saved
states, five goals (curl_witness, extension_witness, wrist_reach_witness, supported_trunk_turn_witness,
elbow_flex_witness). Each goal is the endpoint a reference command physically reached from that start (the WITNESS, a
whole saved pose: qpos, the 26 handle poses, the 44 finger joint angles); three arms then try to reach it in 250 ms:
the controller's own command, and two planner searches (direct, synergy). Saved per trial: the witness and each arm's
qpos, handle poses and finger joint angles, and the start's handle poses (initial_pose: position / 2 m, orientation x y
z w) and finger joint angles (initial_hand_joint_angles_rad); not the start's whole qpos.

THE FROZEN-START PROBE (the movement viewer's part 7, a mode built beside the scheduled probe): the same start and the
same witness poses reused at every checkpoint, so results from different dates compare like with like; its trials also
save initial_qpos and the compiled model used.

SCORING (the probe's own rule, request_shares below): a try's share is its endpoint minus the ZERO POINT (the endpoint
of the controller's command at the checkpoint that made the requested poses), projected on the executed requested
change and divided by that change's squared size: 0 = no further than the zero point, 1 = the whole executed change.
The witness is the endpoint of the controller's own command with ONE change (the probe's recipe: "other command fields
retain the controller"), so only the part each goal asks for is scored: both hands' finger joints (degrees) for closing
and opening the fingers, the left wrist's position (cm) for the reach, both trunk orientations (degrees) for the turn,
and the left elbow joint's own angle (degrees, flexion positive) for the elbow bend. A trial counts only when it is
informative (the probe's WITNESS_RULE: the reference command moved the part beyond the controller's endpoint, and
further along the request than the same change reversed did).
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np

from . import INDEX_DIR, REPO, part_path, pictures_dir, replace_file
from . import commands as C

PROBE_GLOBS = (str(REPO / "Saved" / "mimo_bridge" / "ability_rechecks" / "body_and_synergy_*"),
               str(REPO / "Saved" / "mimo_bridge" / "ability_rechecks" / "fixed_start_body_and_synergy_*"),
               str(REPO / "Saved" / "mimo_bridge" / "fixed_probe_set" / "*"))
MODE_NAMES = {"frozen_start": "frozen start", "frozen_start_anchored": "frozen start, search anchored"}
ARMS = ("controller", "direct", "synergy")
GOAL_NAMES = {"curl_witness": "close the fingers", "extension_witness": "open the fingers",
              "wrist_reach_witness": "reach with the left wrist", "supported_trunk_turn_witness": "turn the trunk",
              "elbow_flex_witness": "bend the left elbow"}
PROBE_INDEX_VERSION = "probe_index_v6"   # v6: each trial row carries the probe's own two shares (own_share, mirror_share)
# goal -> (the part it asks for, unit, the handles or the hands it is measured on)
GOAL_PART = {"curl_witness": ("finger angles of both hands", "degrees", "fingers"),
             "extension_witness": ("finger angles of both hands", "degrees", "fingers"),
             "wrist_reach_witness": ("left wrist place", "cm", ("wrist_L",)),
             "supported_trunk_turn_witness": ("trunk direction", "degrees", ("lower_trunk", "chest")),
             "elbow_flex_witness": ("left elbow joint angle", "degrees", ("lowerarm_L",))}


def _read(path) -> Optional[dict]:
    try:
        with open(str(path), encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def run_id(folder: str) -> str:
    return hashlib.sha1(("probe|" + os.path.basename(os.path.normpath(folder))).encode()).hexdigest()[:12]


def scan() -> List[dict]:
    """Every probe run folder with trials: id, label, mode, checkpoint step, time, trial count, folder."""
    out = []
    for pattern in PROBE_GLOBS:
        for folder in sorted(glob.glob(pattern)):
            trials = os.path.join(folder, "trials.jsonl")
            if not os.path.isfile(trials) or os.path.getsize(trials) == 0:
                continue
            dec = _read(os.path.join(folder, "declaration.json")) or {}
            st = _read(os.path.join(folder, "status.json")) or {}
            ck = str(dec.get("checkpoint") or "")
            try:
                step = int(os.path.basename(ck.rstrip("\\/")).split("_")[-1])
            except ValueError:
                step = None
            try:
                bodies = int(dec.get("bodies") or 8)
            except (TypeError, ValueError):
                bodies = 8
            if bodies < 8:
                continue                      # an equality check on a subset of the bodies, not a reading of its own
            mode = MODE_NAMES.get(str(dec.get("mode") or ""))
            if mode is None:
                mode = "scheduled" if "ability_rechecks" in folder.replace("\\", "/") else "normal mode"
            with open(trials, encoding="utf-8") as f:
                n = sum(1 for line in f if line.strip())
            t = float(st.get("started_at") or os.path.getmtime(trials))
            # the date the checkpoint was saved (its meta.json's file time), where the checkpoint is still on disk: a run
            # made later on an earlier checkpoint is drawn at the date of that checkpoint's training
            meta = os.path.join(ck, "meta.json") if ck else ""
            saved = os.path.getmtime(meta) if meta and os.path.isfile(meta) else None
            out.append(dict(id=run_id(folder), folder=folder.replace("\\", "/"), checkpoint=ck, checkpoint_step=step,
                            mode=mode, time=saved if saved is not None else t, run_time=t, checkpoint_time=saved,
                            trials=n, complete=bool(st.get("state") == "completed"),
                            label="checkpoint %s%s" % (format(step, ",") if step else "?",
                                                      "" if mode == "scheduled" else " (%s)" % mode),
                            model=_saved_model(folder)))
    out.sort(key=lambda r: r["time"])
    # two runs of one checkpoint and mode (a repeat made by another route) are told apart by their folder names
    seen: Dict[str, int] = {}
    for r in out:
        seen[r["label"]] = seen.get(r["label"], 0) + 1
    for r in out:
        if seen[r["label"]] > 1:
            r["label"] += " · " + os.path.basename(r["folder"])
    return out


def _saved_model(folder: str) -> Optional[str]:
    for p in glob.glob(os.path.join(folder, "*.mjb.gz")) + glob.glob(os.path.join(folder, "model", "*.mjb.gz")):
        return p.replace("\\", "/")
    return None


def by_id(pid: str) -> Optional[dict]:
    return next((r for r in scan() if r["id"] == pid), None)


def trials(run: dict) -> List[dict]:
    with open(os.path.join(run["folder"], "trials.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# THE PROBE'S OWN RULE FOR "HOW MUCH OF THE REQUEST HAPPENED" (compare_hand_body_goals_2026-09-24.py witness_check,
# WITNESS_PARTS and FINGER_JOINTS, copied so the viewer does not import the probe's learner stack): the requested change
# is the witness's executed request minus the controller's, over the goal's parts; a try's share is its endpoint minus
# the controller's endpoint, projected on that change and divided by the change's squared size (1 = the whole executed
# request happened, 0 = the try ended where the controller's own command did). tests/test_movement_viewer.py checks
# that this reproduces the probe's stored achieved_share of every witness.
WITNESS_KIND = {"curl_witness": ("joints", ()), "extension_witness": ("joints", ()),
                "wrist_reach_witness": ("position", ("wrist_L",)),
                "supported_trunk_turn_witness": ("rotation", ("lower_trunk", "chest")),
                "elbow_flex_witness": ("elbow", ("lowerarm_L",))}
FINGER_JOINT_ENDS = ("_knuckle", "_middle", "_distal")
ELBOW_JOINT = "robot:left_elbow"


def _rotvec_deg(target_xyzw, base_xyzw) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return np.degrees((Rotation.from_quat(np.asarray(target_xyzw, dtype=np.float64))
                       * Rotation.from_quat(np.asarray(base_xyzw, dtype=np.float64)).inv()).as_rotvec())


def request_shares(tr: dict, geometry: dict, elbow_qpos: Optional[int],
                   reference: Optional[dict] = None) -> Dict[str, Optional[float]]:
    """{witness, controller, direct, synergy: the share of the requested change each endpoint made beyond the
    baseline controller's endpoint} for one trial (the rule above); None where the trial lacks what the rule needs.
    The baseline is the trial's own controller try; for a frozen-start trial at another checkpoint it is the
    REFERENCE trial's controller (the same body and goal at the checkpoint whose controller command made the requested
    poses), so every checkpoint is measured against the same request from the same endpoint."""
    kind, handles = WITNESS_KIND.get(str(tr.get("goal")), (None, ()))
    w = tr.get("witness") or {}
    outs = tr.get("outcomes") or {}
    c = ((reference or {}).get("outcomes") or {}).get("controller") if reference else outs.get("controller")
    c = c or {}
    ends = dict(witness=w, controller=outs.get("controller") if reference else None, direct=outs.get("direct"),
                synergy=outs.get("synergy"))
    out: Dict[str, Optional[float]] = {k: None for k in ends}
    try:
        if kind == "joints":
            parts = [i for i, j in enumerate(geometry["joints"]) if str(j["name"]).endswith(FINGER_JOINT_ENDS)]
            req = np.degrees(np.asarray(w["hand_joint_goal"][0], dtype=np.float64)[parts]
                             - np.asarray(c["hand_joint_goal"][0], dtype=np.float64)[parts])
            moved = lambda x: np.degrees(np.asarray(x["joint_angles_rad"][0], dtype=np.float64)[parts]
                                         - np.asarray(c["joint_angles_rad"][0], dtype=np.float64)[parts])
        elif kind in ("position", "rotation"):
            names = list(geometry["handle_names"])
            parts = [names.index(n) for n in handles]
            cw, cc = np.asarray(w["canonical_command"][0], np.float64), np.asarray(c["canonical_command"][0], np.float64)
            pc = np.asarray(c["pose"][0], np.float64)
            if kind == "position":
                req = np.concatenate([cw[i, :3] - cc[i, :3] for i in parts])                  # cm
                moved = lambda x: np.concatenate([(np.asarray(x["pose"][0], np.float64)[i, :3] - pc[i, :3]) * 200.0
                                                  for i in parts])                         # the pose stores cm / 200
            else:
                req = np.concatenate([_rotvec_deg(cw[i, 3:7], cc[i, 3:7]) for i in parts])
                moved = lambda x: np.concatenate([_rotvec_deg(np.asarray(x["pose"][0], np.float64)[i, 3:7], pc[i, 3:7])
                                                  for i in parts])
        elif kind == "elbow" and elbow_qpos is not None:
            wc = tr.get("witness_check") or {}
            size = float(wc.get("requested_executed") or 0.0)
            if size <= 0:
                return out
            qc = float(c["qpos"][elbow_qpos])
            # the probe divides by the SIGNED request (flexion positive); the trial saves the request's size and the
            # witness's share, so the request's sign is the sign of the witness's flexion times its share's sign (a
            # flexion request when either is missing or zero)
            share_w = wc.get("achieved_share")
            flex_w = math.degrees(qc - float(w["qpos"][elbow_qpos])) if w.get("qpos") is not None else 0.0
            signed = -size if (share_w is not None and float(share_w) * flex_w < 0) else size
            for k, x in ends.items():
                if x and x.get("qpos") is not None:
                    out[k] = float(np.degrees(qc - float(x["qpos"][elbow_qpos])) / signed)  # flexion lowers the angle
            if reference is None:
                out["controller"] = 0.0
            return out
        else:
            return out
        den = float(np.dot(req, req))
        if den <= 0:
            return out
        for k, x in ends.items():
            if x:
                out[k] = float(np.dot(moved(x), req) / den)
        if reference is None:
            out["controller"] = 0.0                  # its own endpoint is the baseline
    except (KeyError, IndexError, TypeError, ValueError):
        return out
    return out


def _r(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, 4) if math.isfinite(x) else None


def _elbow_qpos() -> Optional[int]:
    """The left elbow's qpos address, from an observer recording's header (every H84 world has the same layout)."""
    from mimo_bridge.observer_record import read_header
    from . import sources as S
    for r in S.load().get("recordings") or []:
        for body in r["bodies"].values():
            for h in body.get("headers") or []:
                for j in read_header(h).get("joints") or []:
                    if j.get("name") == ELBOW_JOINT:
                        return int(j["qpos"])
    return None


def shares_of_run(run: dict, runs: Optional[List[dict]] = None) -> List[dict]:
    """Per trial of a run: body, goal, informative, checked (the trial carries the probe's witness check; the earliest
    scheduled runs were saved before the probe had one), the shares of request_shares, the size of the requested change
    (witness_check.requested_executed), and zero_point_step: the checkpoint whose controller endpoint is the share's 0
    (the run's own checkpoint, where the controller's own share is 0 by definition, or the reference checkpoint of a
    frozen start at another checkpoint); cached beside the other probe measures."""
    cache = INDEX_DIR / "probes" / ("%s.shares.json" % run["id"])
    src = os.path.join(run["folder"], "trials.jsonl")
    old = _read(cache)
    if old and old.get("version") == PROBE_INDEX_VERSION and old.get("trials_size") == os.path.getsize(src):
        return old["rows"]
    geo = _read(os.path.join(run["folder"], "hand_geometry.json")) or {}
    eq = _elbow_qpos()
    ref_rows = {}
    zero = run["checkpoint_step"]
    if run["mode"].startswith("frozen"):
        src_step = goals_from(run)
        if src_step is not None and src_step != run["checkpoint_step"]:
            pool = [r for r in (runs or scan()) if r["checkpoint_step"] == src_step and r["complete"]
                    and r["mode"] in ("normal mode", "frozen start", "scheduled")]
            if pool:
                ref_rows = {(int(t["body"]), str(t["goal"])): t for t in trials(pool[0])}
                zero = src_step
    rows = []
    for tr in trials(run):
        ref = ref_rows.get((int(tr.get("body", -1)), str(tr.get("goal")))) if ref_rows else None
        if ref_rows and ref is None:
            continue
        sh = request_shares(tr, geo, eq, reference=ref)
        wc = tr.get("witness_check") or {}
        rows.append(dict(body=int(tr.get("body", -1)), goal=str(tr.get("goal")), informative=bool(wc.get("informative")),
                         checked=bool(wc), own_share=_r(wc.get("achieved_share")), mirror_share=_r(wc.get("mirror_share")),
                         change=_r(wc.get("requested_executed")), unit=wc.get("unit"), zero_point_step=zero,
                         **{k: _r(v) for k, v in sh.items()}))
    out = dict(version=PROBE_INDEX_VERSION, run=run["id"], trials_size=os.path.getsize(src), rows=rows)
    os.makedirs(str(cache.parent), exist_ok=True)
    tmp = part_path(cache)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f)
    replace_file(tmp, cache)
    return rows


def series(goal: str, body="all") -> dict:
    """Per complete probe run: the median, over the goal's INFORMATIVE trials (8 bodies, or one), of the share of the
    requested change each try made beyond the controller's own endpoint (request_shares; the probe's own measure of a
    request). The independent review of Oct 18, 2026 (finding 1): a distance from the requested pose credits the
    controller by construction, since the requested pose is the controller's own endpoint plus one change; the share is
    the probe's like-for-like comparison: 0 = where the controller's own command ended, 1 = the whole executed request.
    The reference command's own share (the witness) shows how much of the request the body can make at all.
    Where a run's own controller endpoint is the zero point (own_baseline), its controller share is 0 by definition and
    is left out of `tries`; it is kept where the zero point is another checkpoint's controller (a frozen start).
    `not_drawn`: every run with no point for this goal, and why. `replaced_by` (the second review of Oct 18, 2026,
    finding 3): a frozen start at another checkpoint whose planners searched around that checkpoint's own controller
    command names the run at the same checkpoint with the search anchored at the frozen set's command; the page draws
    that run in its place."""
    if goal not in GOAL_PART:
        raise ValueError("unknown goal")
    runs = scan()
    found = gains()
    out, not_drawn = [], []
    for run in runs:
        if not run["complete"]:
            st = _read(os.path.join(run["folder"], "status.json")) or {}
            not_drawn.append(dict(id=run["id"], label=run["label"], mode=run["mode"], time=run["time"],
                                  why="not completed: its status file reads %s with %d of %s trials saved" % (
                                      st.get("state") or "nothing", run["trials"], st.get("total") or "?")))
            continue
        every = shares_of_run(run, runs)
        rows = [r for r in every if r["goal"] == goal and (body == "all" or r["body"] == int(body))]
        good = [r for r in rows if r["informative"]]
        zero = rows[0].get("zero_point_step") if rows else None
        own = zero == run["checkpoint_step"]
        tries = {}
        for k in ("witness", "controller", "direct", "synergy"):
            if k == "controller" and own:
                continue                      # its own endpoint is the zero point: 0 by definition, not a measurement
            v = [r[k] for r in good if r.get(k) is not None]
            if v:
                tries[k] = dict(share=_r(np.median(v)), n=len(v))
        if tries:
            ch = [r["change"] for r in good if r.get("change") is not None]
            out.append(dict(id=run["id"], label=run["label"], mode=run["mode"], time=run["time"],
                            run_time=run["run_time"], checkpoint_time=run["checkpoint_time"],
                            checkpoint_step=run["checkpoint_step"], trials=run["trials"], tries=tries,
                            change_size=_r(np.median(ch)) if ch else None,
                            informative=len(good), bodies=len(rows), goals_from=goals_from(run),
                            zero_point_step=zero, own_baseline=own, replaced_by=None, gain=found.get(_folder_key(run["folder"]))))
        else:
            tried = {r["goal"] for r in every}
            if not rows:
                why = "it did not try this movement" + (" (it tried only closing and opening the fingers)"
                                                        if tried == {"curl_witness", "extension_witness"} else "")
            elif not any(r.get("checked") for r in rows):
                why = ("saved before the probe checked whether each request moved the part (its trials carry no such "
                       "check), so no trial counts")
            else:
                why = "no trial of this movement counts (none moved the part past the starting line farther than the same change reversed did)"
            not_drawn.append(dict(id=run["id"], label=run["label"], mode=run["mode"], time=run["time"], why=why))
    anchored = MODE_NAMES["frozen_start_anchored"]
    for x in out:
        if x["mode"] == MODE_NAMES["frozen_start"] and not x["own_baseline"]:
            y = next((y for y in out if y["mode"] == anchored and y["checkpoint_step"] == x["checkpoint_step"]), None)
            x["replaced_by"] = y["id"] if y else None
    part, unit, _ = GOAL_PART[goal]
    return dict(goal=goal, goal_name=GOAL_NAMES[goal], part=part, unit=unit, goals=GOAL_NAMES, runs=out,
                not_drawn=not_drawn, tries=["witness", "controller", "direct", "synergy"])


# THE IMPROVEMENT LINE OF THE FIXED-START PROBE (proxy decision movement-viewer-fixed-probe-decision-20261018,
# under the operator's Oct 13, 2026 delegation): each planner arm at a checkpoint against the SAME arm at the reference
# checkpoint 494,025, from the frozen starts toward the frozen goals; DIRECT decides (met when the paired 95% interval of
# the physical goal error difference lies below 0), synergy is reported beside it. fixed_probe_set_2026-10-18.py writes
# it as a top-level "gain" object: into a comparison file gain_*.json (recomputed on saved runs) or into a scheduled
# round's result.json. The page shows it beside the run it names (this_run, or the result's own folder).
GAIN_GLOBS = (str(REPO / "Saved" / "mimo_bridge" / "fixed_probe_set" / "*" / "gain_*.json"),
              str(REPO / "Saved" / "mimo_bridge" / "ability_rechecks" / "fixed_start_body_and_synergy_*" / "result.json"))
_GAINS: Dict[str, tuple] = {}


def _folder_key(folder: str) -> str:
    # a gain file names its run as recorded (inside D:\UEMA_Research); the run list names the folder where it is
    from mimo_bridge.repo_paths import localize
    return os.path.normcase(os.path.normpath(str(localize(str(folder)))))


def _gain_summary(d: dict) -> Optional[dict]:
    """The page's part of a gain object: status, the line, and each arm's composite, wrist and hand-joint readings."""
    g = d.get("gain") if isinstance(d, dict) else None
    if not isinstance(g, dict) or not g.get("rule_version"):
        return None

    def reading(x):
        x = x if isinstance(x, dict) else {}
        return dict(difference_mean=_r(x.get("difference_mean")),
                    interval=[_r(v) for v in (x.get("paired_bootstrap95") or [])],
                    this_mean=_r(x.get("this_mean")), reference_mean=_r(x.get("reference_mean")),
                    bodies_lower_here=x.get("bodies_lower_here"), reading=x.get("interval_reading"))

    arms = {}
    for k, a in (g.get("arms") or {}).items():
        a = a if isinstance(a, dict) else {}
        arms[k] = dict(reading(a), bodies=a.get("bodies"), line_met=a.get("line_met"),
                       wrist_cm=reading(a.get("wrist_goal_error_cm")), hand_deg=reading(a.get("hand_joint_error_deg")))
    ref, this = g.get("reference") or {}, g.get("this") or {}
    return dict(rule_version=g.get("rule_version"), status=g.get("status"), invalid_reasons=list(g.get("invalid_reasons") or []),
                primary_arm=g.get("primary_arm"), line_met=g.get("line_met"), reference_step=ref.get("checkpoint_step"),
                this_step=this.get("checkpoint_step"), label=g.get("label"), made_at=d.get("made_at"), arms=arms)


def gains() -> Dict[str, dict]:
    """{probe run folder (normalized): the newest gain summary that names it}, read again only when a file changed."""
    newest: Dict[str, tuple] = {}
    for pattern in GAIN_GLOBS:
        for p in glob.glob(pattern):
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            hit = _GAINS.get(p)
            if hit is None or hit[0] != mt:
                d = _read(p)
                s = _gain_summary(d) if d else None
                hit = _GAINS[p] = (mt, s, (d or {}).get("this_run") or os.path.dirname(p))
            _, s, run = hit
            if s:
                key = _folder_key(run)
                if key not in newest or mt > newest[key][0]:
                    newest[key] = (mt, s)
    return {k: v[1] for k, v in newest.items()}


def goals_from(run: dict) -> Optional[int]:
    """The checkpoint whose controller command made the run's requested poses: its own for a scheduled or normal run,
    the frozen set's reference checkpoint for a frozen-start run (read from the first trial's witness_source)."""
    if not run["mode"].startswith("frozen"):
        return run["checkpoint_step"]
    try:
        with open(os.path.join(run["folder"], "trials.jsonl"), encoding="utf-8") as f:
            first = json.loads(f.readline())
        src = first.get("witness_source")
        txt = json.dumps(src)
        import re as _re
        m = _re.search(r"ckpt_step_(\d+)", txt) or _re.search(r"(?<!\d)(4\d{5})(?!\d)", txt)
        return int(m.group(1)) if m else None
    except (OSError, ValueError):
        return None


def key_for(pid: str, body: int, goal: str, arm: str) -> str:
    from .render import RENDER_VERSION
    return hashlib.sha1(("probe|%s|%d|%s|%s|%s" % (pid, int(body), goal, arm, RENDER_VERSION)).encode()).hexdigest()[:20]


def _drawing_model(run: dict):
    """(model path, recorder header, how it was chosen): the probe's own saved model, else the saved model of the
    newest observer recording with the same number of joint positions."""
    from mimo_bridge.observer_record import read_header
    from . import sources as S
    recs = S.load().get("recordings") or []
    for r in sorted(recs, key=lambda r: -r["last_time"]):
        for b, body in r["bodies"].items():
            if body.get("model") and body.get("headers"):
                h = read_header(body["headers"][0])
                if run.get("model"):
                    return run["model"], h, "the probe's own saved model"
                return body["model"], h, "the saved model of recording %s (the probe saved none)" % r["label"]
    raise C.Refused("no saved model to draw with")


def _starting_line_trial(run: dict, tr: dict) -> Optional[dict]:
    """The trial whose motor controller command's end is this trial's starting line (0%): the trial itself, or for a
    frozen start at another checkpoint the reference run's trial of the same body and goal (shares_of_run's rule)."""
    if run["mode"].startswith("frozen"):
        src = goals_from(run)
        if src is not None and src != run["checkpoint_step"]:
            pool = [r for r in scan() if r["checkpoint_step"] == src and r["complete"]
                    and r["mode"] in ("normal mode", "frozen start", "scheduled")]
            if not pool:
                return None
            return next((t for t in trials(pool[0]) if int(t.get("body", -1)) == int(tr.get("body", -1))
                         and t.get("goal") == tr.get("goal")), None)
    return tr


def render_trial(pid: str, body: int, goal: str, arm: str, out_root: Optional[str] = None) -> dict:
    """The pictures of one probe trial and arm: start (hands from the saved start angles; the whole body only when the
    probe saved its start qpos), the witness pose as the requested pose, and the arm's pose after 250 ms."""
    from . import render as R
    t0 = time.perf_counter()
    mem0, gpu0 = R.process_memory_mb(), R.gpu_memory_mb()
    run = by_id(pid)
    if run is None:
        raise C.Refused("unknown probe run %r" % pid)
    tr = next((t for t in trials(run) if int(t.get("body", -1)) == int(body) and t.get("goal") == goal), None)
    if tr is None:
        raise C.Refused("no trial for body %d, goal %s" % (body, goal))
    o = (tr.get("outcomes") or {}).get(arm)
    if not o:
        raise C.Refused("the trial has no %s arm" % arm)
    key = key_for(pid, body, goal, arm)
    out_dir = os.path.join(str(out_root or pictures_dir()), key)
    os.makedirs(out_dir, exist_ok=True)
    model_path, header, how = _drawing_model(run)
    own = tr.get("world_model") or {}
    if own.get("file") and os.path.isfile(os.path.join(run["folder"], own["file"])):
        model_path = os.path.join(run["folder"], own["file"]).replace("\\", "/")
        how = "the world model this trial ran on (saved by the probe)"
        bm = C.BodyModel(model_path, header, check_fingerprint=False)
        if own.get("structure_fingerprint") and bm.fingerprint != own["structure_fingerprint"]:
            raise C.Refused("the trial's saved model does not have the structure fingerprint the trial recorded")
    else:
        bm = C.BodyModel(model_path, header, check_fingerprint=not run.get("model"))
    q_w = np.asarray(tr["witness"]["qpos"], dtype=np.float64)
    q_a = np.asarray(o["qpos"], dtype=np.float64)
    if q_w.size != bm.m.nq or q_a.size != bm.m.nq:
        raise C.Refused("the trial's joint positions (%d) do not fit the drawing model (%d)" % (q_w.size, bm.m.nq))
    q_start_full = np.asarray(tr["initial_qpos"], dtype=np.float64) if tr.get("initial_qpos") is not None else None
    # the hands depend only on their 44 finger joints once the wrist is held still (render.py), so the saved start
    # angles on the witness pose draw the starting hands exactly
    q_start_hands = q_w.copy()
    q_start_hands[np.asarray(bm.qadr)] = np.asarray(tr["initial_hand_joint_angles_rad"], dtype=np.float64)
    if q_start_full is not None:
        q_start_hands = q_start_full
    images: Dict[str, dict] = {}
    hands = R._Hands(model_path)
    try:
        for side in ("left", "right"):
            images.update(hands.draw(side, [("start", q_start_hands), ("target", q_w), ("t250", q_a)], out_dir,
                                     ghost=q_w, ghost_on=("t250",)))
    finally:
        hands.close()
    # the whole body as in a command's drawing (render.py): the infant alone in its upper body's coordinates at the
    # start (the witness pose where the probe saved no start), three cameras; the requested pose is the witness, a pose
    # a reference command reached, drawn pale over the try's pose; the probe saved no objects, so none is drawn
    body_ = R._Body(model_path, header)
    try:
        ref = q_start_full if q_start_full is not None else q_w
        body_.set_frame(ref)
        posture = body_.posture()
        weight = body_.weight_n()
        ghost = body_.own_geoms(q_w)
        sets = [ghost, body_.own_geoms(q_a)] + ([body_.own_geoms(q_start_full)] if q_start_full is not None else [])
        dots, views, wrist_marks = [], [v for v, _, _ in R.BODY_VIEWS], None
        if goal == "wrist_reach_witness":
            # the asked-for move is small (0.6 cm in all on 494,025's trials): the test pose's left wrist place (the
            # left hand body's origin, the middle of the palm) is a pink dot (R.MARK_COLOR; the arm's own blue hid it),
            # joined to the wrist where it is in each picture; the same point where the motor controller's own command
            # ended (the starting line, 0%) is a grey dot, since most of the move from the start is that command's own
            # (the independent review of Oct 19, 2026, round 4); a close-up camera 0.32 m from the pink dot, from the
            # front, draws them larger
            hand = bm.m.body("left_hand").id

            def palm(q):
                body_.place(q)
                return np.array(body_.d.xpos[hand])
            place = palm(q_w)
            dots = [(place, R.MARK_COLOR, hand)]
            line_tr = _starting_line_trial(run, tr)
            q_line = ((line_tr or {}).get("outcomes") or {}).get("controller", {}).get("qpos")
            p_line = palm(np.asarray(q_line, dtype=np.float64)) if q_line is not None else None
            if p_line is not None:
                dots.append((p_line, R.LINE_COLOR, None))
            p_got = palm(q_a)
            p_start = palm(q_start_full) if q_start_full is not None else None
            cm = lambda a_, b_: _r(float(np.linalg.norm(np.asarray(a_) - np.asarray(b_))) * 100.0) \
                if a_ is not None and b_ is not None else None
            wrist_marks = dict(start_to_test_cm=cm(p_start, place), start_to_line_cm=cm(p_start, p_line),
                               line_to_test_cm=cm(p_line, place), got_to_test_cm=cm(p_got, place))
            body_.place(ref)
            views.append("wrist")
        body_.fit(sets, [p for p, _, _ in dots])
        if dots:
            # the close-up looks across the line from the grey dot to the pink one (its direction at right angles to
            # that line, from the front side of the body), so the two dots, 0.4 cm apart on 483,411's body 0, do not
            # hide each other; from the front when there is no grey dot
            look = body_.Rf.T @ (dots[0][0] - body_.o)
            az, el = 180.0, -8.0
            if len(dots) > 1:
                v = body_.Rf.T @ (dots[0][0] - dots[1][0])
                f = np.cross(v, [0.0, 0.0, 1.0])
                if float(np.linalg.norm(f)) > 1e-9:
                    f = f / np.linalg.norm(f)
                    if f[0] > 0:
                        f = -f                    # from the front: the camera on the belly's side, looking back
                    az, el = float(np.degrees(np.arctan2(f[1], f[0]))), 0.0
            body_.cams["wrist"] = (look, 0.22, az, el)
        for view in views:
            r_dot = 0.0025 if view == "wrist" else 0.014    # the close-up shows two dots 0.4 cm apart
            if q_start_full is not None:
                images[body_.draw("start", view, q_start_full, None, None, dots=dots, out_dir=out_dir,
                                  dot_radius=r_dot)] = dict(view=view, state="start")
            images[body_.draw("target", view, q_w, None, None, dots=dots, out_dir=out_dir,
                              dot_radius=r_dot)] = dict(view=view, state="target")
            images[body_.draw("t250", view, q_a, None, None, ghost=ghost, dots=dots, out_dir=out_dir,
                              dot_radius=r_dot)] = dict(view=view, state="t250")
    finally:
        body_.close()
    for name in images:
        images[name]["sha256"] = R._sha(os.path.join(out_dir, name))
    share_row = next((r for r in shares_of_run(run) if r["body"] == int(body) and r["goal"] == goal), None) or {}
    part, unit, _ = GOAL_PART.get(goal, ("the requested part", "", ""))
    request = dict(part=part, unit=unit, change=share_row.get("change"), this_try=share_row.get(arm),
                   reference_command=share_row.get("witness"), informative=share_row.get("informative"),
                   zero_point_step=share_row.get("zero_point_step"),
                   goals_from=goals_from(run))
    gpu1, mem1 = R.gpu_memory_mb(), R.process_memory_mb()
    manifest = dict(
        version=R.RENDER_VERSION, key=key, probe=pid, label="probe at " + run["label"], kind="probe", run=run["folder"],
        body=int(body), goal=goal, goal_name=GOAL_NAMES.get(goal, goal), arm=arm, step=run["checkpoint_step"],
        time=run["time"], rendered_at=time.time(), samples_ms=[250], all_samples_ms=[250], has_start=True,
        has_hand_goal=True, posture=posture, posture_from="start" if q_start_full is not None else "test pose",
        body_views=views, body_weight_n=_r(weight), body_start_saved=q_start_full is not None, wrist_marks=wrist_marks,
        informative=(tr.get("witness_check") or {}).get("informative"),
        notes=(["This trial " + ("counts: the request put straight into the command moved the part past the starting "
                                 "line, farther than the same change reversed did"
                                 if (tr.get("witness_check") or {}).get("informative")
                                 else "does not count: the request put straight into the command did not move the part "
                                      "past the starting line farther than the same change reversed did")]
               + ([] if q_start_full is not None else
               ["Whole body at the start: not saved by this probe version (it saved the start's hand angles and handle "
                "poses only)"]) + ["Objects: not saved by the probe; the body is drawn alone",
                                   "Test pose: the pose the request put straight into the command reached from the same "
                                   "start"]),
        images=images, request=request,
        measures={},
        handles=[], contacts=[], context={},
        provenance=dict(chunks={os.path.join(run["folder"], "trials.jsonl").replace("\\", "/"):
                                R._sha(os.path.join(run["folder"], "trials.jsonl"))},
                        headers={}, model=model_path, model_gzip_sha256=R._sha(model_path),
                        model_structure_fingerprint=bm.fingerprint, renderer=R.RENDER_VERSION,
                        selection="probe run %s, body %d, goal %s, arm %s; drawn with %s" % (run["label"], body, goal,
                                                                                            arm, how)),
        cost=dict(seconds=round(time.perf_counter() - t0, 3), process_memory_mb_before=_r(mem0),
                  process_memory_mb_after=_r(mem1), gpu_memory_mb_before=_r(gpu0), gpu_memory_mb_after=_r(gpu1),
                  pictures=len(images), bytes=sum(os.path.getsize(os.path.join(out_dir, n)) for n in images)))
    tmp = part_path(os.path.join(out_dir, "manifest.json"))
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    replace_file(tmp, os.path.join(out_dir, "manifest.json"))
    return manifest
