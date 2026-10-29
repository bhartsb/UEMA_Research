# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Sensor/action contract commissioning checks (spec 20.6, executable on MIMo).

The three UE transfer contracts, run against the MIMo bridge so the checks
themselves are proven working code before any UE build runs them:

  touch    the COMMANDED hand's own skin patches (selected by body name) must
           show a pressure ONSET over their resting baseline while pressed and
           a RELEASE after the press ends; pressure on any other body never
           vouches for hand sensing, and collision counts alone never vouch
           for pressure sensing
  action   requested and achieved are distinct quantities from distinct
           sources: a free limb's achieved movement follows its command
           direction; a limb commanded through the floor shows a large command
           with small achieved movement; the joint-space control MuJoCo was
           HANDED must respond to the command (a source that ignores commands
           fails); and the learner's motor tensor must carry the achieved
           pose, not an echo of the command
  vision   a commanded gaze deflection must move the MEASURED eye orientation
           (eye-joint degrees) beyond the drift seen in equal no-command
           windows, the rendered image must change across that eye command,
           and every frame must be fresh (frame number and sim time strictly
           advancing, reproduced bit-equal in the learner's frame metadata)

Environment fidelity: the env is built by the trainer's own env_kwargs_for
(train_mimo), fed by the run's args.json overlaid on the trainer parser's
defaults, with the run's recorded curriculum phase - never by a partial local
re-implementation (2026-09-08 review: the local copy dropped age binding,
delays, seed and no_vision, and forced vision on).

Learner path: raw adapter numbers are cross-checked against the tokenized
frame the model actually consumes - touch pressure column, motor achieved
pose (/ position_range_cm), gaze metadata, frame number and timestamp.

Negative controls are BUILT IN: `--self-test` drives these predicates with
fake sources (unrelated stale pressure + ignored commands + frozen frames;
collision-only; command-echo achieved state) and requires every fake to FAIL.
The 2026-09-08 review fixtures (docs/code_verification_2026-09-08/
sensor_probe_fixtures.py) are the precedent; their environment_fixture
section pins the superseded env_kwargs_from_args helper, which this revision
removes in favour of the trainer's own constructor.

python -m mimo_bridge.tools.sensor_action_contract_check --run <run_dir>
python -m mimo_bridge.tools.sensor_action_contract_check --self-test
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import tempfile
import time
import types

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SAVED = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "Saved", "mimo_bridge")

RIGHT_WRIST = 6        # HANDLE_NAMES[6] == "wrist_R" (0 is wrist_L), in both body layouts
GAZE_HANDLE = 23       # "gaze" in both layouts: trunk_v1 adds the lower trunk and chest AFTER it
# the same body-name keys train_mimo uses to find hand skin patches (env.py:1828),
# prefixed with the side so the LEFT hand can never vouch for a RIGHT-hand press
HAND_PATCH_KEYS = ("hand", "distal", "middle", "knuckle", "metacarpal", "thbase", "thhub")


def _np(x):
    """torch tensor (tokenized frames) or ndarray -> ndarray; None passes through."""
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def free_ram_gb() -> float:
    """Physical RAM currently free (GlobalMemoryStatusEx.ullAvailPhys)."""
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                   [(n, ctypes.c_ulonglong) for n in
                    ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                     "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPhys / 2**30


def _sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# The tail sizes latest_phase reads, in bytes, tried in turn until a record that carries a phase is found. One
# size of 262,144 held a whole 'mimo' record while records stayed under it (H84's last 59 were 182,761 to
# 234,927 bytes on Oct 3, 2026); the swap test's added readings and controllability's shadow gate add 36,215
# bytes, which puts 3 of those 59 past it, and the phase was then read as absent.
PHASE_TAILS = (262144, 1048576, 4194304)


def latest_phase(run_dir: str):
    """The run's most recently logged curriculum phase, from the tail of its stats
    file (train_mimo writes it into every 'mimo' record). 'fixed' means no
    curriculum and maps to None for env_kwargs_for. The tail grows through PHASE_TAILS until a
    record with a phase is found or the whole file was read."""
    p = os.path.join(run_dir, "replay_training_stats.jsonl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        for tail in PHASE_TAILS:
            f.seek(max(0, size - tail))
            lines = f.read().decode("utf-8", "replace").splitlines()
            for ln in reversed(lines):
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                ph = r.get("phase") if isinstance(r, dict) else None
                if ph:
                    return None if ph == "fixed" else str(ph)
            if tail >= size:
                break
    return None


def resolved_env_kwargs(run_dir: str, phase_override=None):
    """Build env kwargs exactly the way training does: train_mimo.env_kwargs_for on
    the run's args.json overlaid on the trainer parser's own defaults. Scene files
    go to a scratch dir, never into the live run dir."""
    a = json.load(open(os.path.join(run_dir, "args.json")))
    from mimo_bridge import train_mimo as tm
    scratch = tempfile.mkdtemp(prefix="contract_")
    ns = tm.parse_args(["--out", scratch])
    for k, v in a.items():
        setattr(ns, k, v)
    phase = phase_override if phase_override is not None else latest_phase(run_dir)
    kw = tm.env_kwargs_for(ns, phase, scratch)
    kw["tag"] = "contract"
    return kw, phase


def right_hand_rows(patch_names):
    """Patch row indices belonging to the RIGHT hand, by body name. Empty when the
    source exposes no patch identity - which is itself a touch-contract failure."""
    if not patch_names:
        return []
    return [i for i, nm in enumerate(patch_names)
            if "right" in nm and any(k in nm for k in HAND_PATCH_KEYS)]


def hold_current(bones: np.ndarray) -> np.ndarray:
    """A command that asks every handle to stay where it is, full strength.
    bones layout VERIFIED at datasource.motor_to_echo (2026-09-08 second pass):
    [quat(0:4), pos_world_cm(4:7), ...] - the first version indexed quaternion
    components as positions and its action-phase numbers were invalid.
    One row per body row, so the order has the world's own layout: until 2026-09-23 this was a
    fixed 24 rows, which a trunk_v1 world (26 rows) refuses (fix-following-gauge)."""
    n = int(np.asarray(bones).shape[0])
    cmd = np.zeros((1, n, 9), dtype=np.float64)
    cmd[0, :, 0:3] = bones[:, 4:7]              # already cm
    cmd[0, :, 3:7] = bones[:, 0:4]
    cmd[0, :, 7] = 0.5
    cmd[0, :, 8] = 1.0
    return cmd


def run_phase(src, cmd, steps, rh_rows, gaze=None, render_every=5):
    """Drive one command for `steps` env steps; collect per-step observations,
    including the learner-side (tokenized) copies of the same quantities."""
    g = np.zeros((1, 2)) if gaze is None else gaze
    rows = []
    f = None
    for i in range(steps):
        f = src.step(cmd, g, None, render=(i % render_every == render_every - 1))[0]
        patches = np.asarray(f.patches)
        rh = rh_rows if rh_rows else []
        other = [r for r in range(patches.shape[0]) if r not in rh]
        tok = f.tokenized
        tok_imp = _np(getattr(tok, "touch_impulses", None))
        md = getattr(tok, "metadata", None)
        rows.append(dict(
            hand_pos=f.bones[RIGHT_WRIST, 4:7].copy(),
            frame=int(f.frame), sim_time=float(f.sim_time),
            n_contacts=int(f.n_contacts),
            rh_press_pa=float(patches[rh, 0].max()) if rh else 0.0,
            other_press_pa=float(patches[other, 0].max()) if other else 0.0,
            gaze_az=float(f.gaze_deg[0]), gaze_el=float(f.gaze_deg[1]),
            tok_frame=None if md is None else int(md.frame_number),
            tok_time=None if md is None else float(md.timestamp),
            tok_gaze_az=None if md is None else float(md.gaze_azimuth_deg),
            tok_valid=tuple(getattr(tok, "touch_valid_counts", (0, 0))),
            tok_rh_press=float(tok_imp[rh, 0].max()) if (rh and tok_imp is not None) else 0.0,
            handed_ctrl=None if not isinstance(f.meta, dict) else _np(f.meta.get("handed_ctrl")),
            vision=None if f.vision_patches is None
            else _np(f.vision_patches).astype(np.float32),
        ))
    return rows, f


def _med(rows, key, last=10):
    return float(np.median([r[key] for r in rows[-last:]]))


def echo_learner_check(f, rng_cm):
    """The learner's motor tensor must carry the ACHIEVED pose: tokenized
    motor_handles pos * position_range_cm == bones world cm, quat == bones quat."""
    mh = _np(getattr(f.tokenized, "motor_handles", None))
    if mh is None or len(mh) != len(f.bones):
        return False, False
    # every body row, the eyes' row left out BY NAME: until 2026-09-23 this read rows 0-22,
    # which in trunk_v1 also left out the lower trunk and the chest (fix-following-gauge)
    from mimo_bridge.body_layout import names_for_rows
    rows = [i for i, nm in enumerate(names_for_rows(len(f.bones))) if nm != "gaze"]
    ok_pos = bool(np.allclose(mh[rows, 0:3] * rng_cm, f.bones[rows, 4:7], atol=0.5))
    ok_quat = bool(np.allclose(mh[rows, 3:7], f.bones[rows, 0:4], atol=1e-2))
    return ok_pos, ok_quat


def run_contracts(src, patch_names, expected_patches, settle_steps, phase_steps,
                  run_name, rng_cm=200.0):
    """The assay itself, on any frame source. Returns the result dict (no I/O)."""
    out = {"run": run_name, "stamp": time.strftime("%Y-%m-%d %H:%M:%S"), "contracts": {}}
    rh_rows = right_hand_rows(patch_names)
    out["right_hand_patch_rows"] = {int(i): patch_names[i] for i in rh_rows} if patch_names else {}
    all_rows = []

    # Phase order matters physically (probed 2026-09-08 on the live run config):
    # a 40 cm press launched from the RAISED free-phase pose stalls 4 cm above the
    # mattress with zero hand contact, while the same press from the settled pose
    # produces a 145 kPa thumb-tip press. So: settle -> press -> release -> free.
    f0 = src.reset()[0]
    settle, f0 = run_phase(src, hold_current(f0.bones), settle_steps, rh_rows)
    all_rows += settle
    rest_ok_pos, rest_ok_quat = echo_learner_check(f0, rng_cm)

    # ---- ACTION contract: blocked limb (pressed from the settled pose) ----------
    # press runs 2x phase_steps: measured on the live config, hand-skin contact
    # first appears ~50 steps into the press (the hand pivots onto the thumb tip)
    start_b = f0.bones[RIGHT_WRIST, 4:7].copy() / 100.0
    blocked_cmd = hold_current(f0.bones)
    blocked_cmd[0, RIGHT_WRIST, 0:3] = start_b * 100.0 + np.array([0.0, 0.0, -40.0])
    blocked_rows, f2 = run_phase(src, blocked_cmd, 2 * phase_steps, rh_rows)
    all_rows += blocked_rows

    # commands must be ACCEPTED: the joint-space control MuJoCo was handed has to
    # respond to the changed command (a source with no applied-command record, or
    # one that ignores commands, fails here - 2026-09-08 review fake did both)
    hc_settle, hc_blocked = settle[-1]["handed_ctrl"], blocked_rows[-1]["handed_ctrl"]
    ctrl_recorded = hc_settle is not None and hc_blocked is not None
    ctrl_responded = bool(ctrl_recorded and float(np.abs(hc_blocked - hc_settle).mean()) > 0.0)
    achieved_drop_cm = float((start_b[2] - f2.bones[RIGHT_WRIST, 6] / 100.0) * 100.0)
    commanded_drop_cm = 40.0
    # the executable distinction: achieved (physics) vs requested (the command),
    # checked at the raw bones AND at the learner's motor tensor
    echo_equals_achieved = bool(np.allclose(f2.bones[RIGHT_WRIST, 4:7],
                                            blocked_cmd[0, RIGHT_WRIST, 0:3], atol=1.0))
    mh2 = _np(getattr(f2.tokenized, "motor_handles", None))
    if mh2 is None:
        tok_cmd_dist_cm, tok_ach_dist_cm = -1.0, -1.0
    else:
        wrist_tok_cm = mh2[RIGHT_WRIST, 0:3] * rng_cm
        tok_cmd_dist_cm = float(np.linalg.norm(wrist_tok_cm - blocked_cmd[0, RIGHT_WRIST, 0:3]))
        tok_ach_dist_cm = float(np.linalg.norm(wrist_tok_cm - f2.bones[RIGHT_WRIST, 4:7]))
    # ---- RELEASE phase (hand back up: press must END) ---------------------------
    release_cmd = hold_current(f2.bones)
    release_cmd[0, RIGHT_WRIST, 0:3] = f2.bones[RIGHT_WRIST, 4:7] + np.array([0.0, 0.0, 20.0])
    release_rows, f3 = run_phase(src, release_cmd, phase_steps, rh_rows)
    all_rows += release_rows

    # ---- ACTION contract: free-limb follow (from the lifted pose) ---------------
    start = f3.bones[RIGHT_WRIST, 4:7].copy() / 100.0   # cm -> m
    free_cmd = hold_current(f3.bones)
    target_delta_cm = np.array([8.0, 0.0, 6.0])      # up-and-out, away from floor
    free_cmd[0, RIGHT_WRIST, 0:3] = start * 100.0 + target_delta_cm
    free_rows, f1 = run_phase(src, free_cmd, phase_steps, rh_rows)
    all_rows += free_rows
    disp_m = f1.bones[RIGHT_WRIST, 4:7] / 100.0 - start
    cmd_dir = target_delta_cm / np.linalg.norm(target_delta_cm)
    moved_cm = float(np.linalg.norm(disp_m) * 100.0)
    follow_cos = float(np.dot(disp_m / max(np.linalg.norm(disp_m), 1e-9), cmd_dir))

    out["contracts"]["action"] = {
        "free_moved_cm": round(moved_cm, 2),
        "free_direction_cos": round(follow_cos, 3),
        "blocked_commanded_drop_cm": commanded_drop_cm,
        "blocked_achieved_drop_cm": round(achieved_drop_cm, 2),
        "blocked_ratio": round(achieved_drop_cm / commanded_drop_cm, 3),
        "achieved_is_command_echo": echo_equals_achieved,
        "applied_ctrl_recorded": bool(ctrl_recorded),
        "applied_ctrl_responded": ctrl_responded,
        "learner_motor_is_achieved_at_rest": bool(rest_ok_pos and rest_ok_quat),
        "learner_wrist_to_command_cm": round(tok_cmd_dist_cm, 2),
        "learner_wrist_to_achieved_cm": round(tok_ach_dist_cm, 2),
        "pass": bool(moved_cm > 1.0 and follow_cos > 0.0
                     and achieved_drop_cm < 0.5 * commanded_drop_cm
                     and not echo_equals_achieved
                     and ctrl_recorded and ctrl_responded
                     and rest_ok_pos and rest_ok_quat
                     and tok_ach_dist_cm >= 0.0 and tok_ach_dist_cm < 0.5
                     and tok_cmd_dist_cm > 2.0),
        "meaning": "a free limb moves with its command; a limb commanded 40 cm "
                   "through the floor must NOT achieve it; the control actually "
                   "handed to physics must respond to the command; and the "
                   "learner's motor tensor must carry the achieved pose (within "
                   "0.5 cm) while sitting >2 cm from the blocked command target",
    }

    # ---- TOUCH contract (identity + onset + release) ----------------------------
    # PEAK statistics, not medians: measured on the live config, pressing the
    # lying hand into the mattress produces brief tap contacts (368 kPa peaks
    # separated by zero-pressure steps), and the hand lifted clear still brushes
    # the body at ~1.5 kPa (self-touch is real touch). Median-of-press read 0 on
    # a working sensor. Comparator = the FREE phase (hand commanded up into the
    # air); onset demands the press peak exceed 10x the lifted peak - the
    # measured margin is 245x, so 10x is well inside it.
    lifted_peak = max(r["rh_press_pa"] for r in free_rows)
    pressed_peak = max(r["rh_press_pa"] for r in blocked_rows)
    pressed_steps = sum(1 for r in blocked_rows if r["rh_press_pa"] > 0)
    released_peak = max(r["rh_press_pa"] for r in release_rows[-phase_steps // 2:])
    pressed_other = max(r["other_press_pa"] for r in blocked_rows)
    tok_lifted_peak = max(r["tok_rh_press"] for r in free_rows)
    tok_pressed_peak = max(r["tok_rh_press"] for r in blocked_rows)
    valid_counts = blocked_rows[-1]["tok_valid"]
    out["contracts"]["touch"] = {
        "right_hand_rows_found": len(rh_rows),
        "settled_rh_peak_pa": round(max(r["rh_press_pa"] for r in settle), 1),
        "lifted_rh_peak_pa": round(lifted_peak, 1),
        "pressed_rh_peak_pa": round(pressed_peak, 1),
        "pressed_rh_contact_steps": pressed_steps,
        "late_release_rh_peak_pa": round(released_peak, 1),
        "pressed_other_body_peak_pa": round(pressed_other, 1),
        "resting_contacts": _med(settle, "n_contacts"),
        "pressed_contacts": _med(blocked_rows, "n_contacts"),
        "learner_lifted_rh_peak": round(tok_lifted_peak, 6),
        "learner_pressed_rh_peak": round(tok_pressed_peak, 6),
        "learner_valid_patch_count": int(valid_counts[0]),
        "expected_patch_count": int(expected_patches),
        "pass": bool(len(rh_rows) > 0
                     and pressed_peak > 10.0 * lifted_peak and pressed_peak > 0.0
                     and released_peak < 0.1 * pressed_peak
                     and tok_pressed_peak > 10.0 * tok_lifted_peak
                     and tok_pressed_peak > 0.0
                     and int(valid_counts[0]) == int(expected_patches)),
        "meaning": "the pressed RIGHT hand's own skin rows (picked by body name) "
                   "must peak above 10x their in-air peak, and fall back below "
                   "a tenth of the press peak once lifted - pressure on any "
                   "other body counts for nothing - and the same onset must "
                   "appear in the learner's tokenized pressure column with "
                   "every patch row marked valid",
    }

    # ---- VISION contract: eye command vs null windows ---------------------------
    # The eye sub-assay runs with the run's STOCHASTIC eye drivers disabled
    # (gaze_reflex, spontaneous_saccades) via env rebuild, and the artifact
    # records that isolation. Measured on the live config with them ON, the
    # reflex chases the moving mobile and no-command windows wander 46 deg by
    # the same estimator - the commanded response (30 deg) is real but cannot
    # be attributed inside that noise. Everything else stays the run's own
    # settings. Window is 40 steps: measured eye slew is ~1.8 deg/step, so a
    # sign reversal needs ~35 steps to cross.
    out["eye_assay_isolation"] = {"gaze_reflex": False, "spontaneous_saccades": False}
    seg_break = len(all_rows)      # rebuild resets frame counter and sim time
    src.rebuild(gaze_reflex=False, spontaneous_saccades=False)
    fe = src.reset()[0]
    eye_settle, fe = run_phase(src, hold_current(fe.bones), 30, rh_rows,
                               render_every=30)
    all_rows += eye_settle
    img_before = next((r["vision"] for r in reversed(eye_settle)
                       if r["vision"] is not None), None)
    img_after = None
    plus_az, minus_az, null_az = [], [], []
    W = 40
    for rep in range(4):
        sign = 1.0 if rep % 2 == 0 else -1.0
        g = np.zeros((1, 2))
        g[0, 0] = 0.8 * sign                     # radians: eye-joint target
        eye_rows, fe = run_phase(src, hold_current(fe.bones), W, rh_rows,
                                 gaze=g, render_every=W)
        all_rows += eye_rows
        win_az = float(np.mean([r["gaze_az"] for r in eye_rows[-5:]]))
        (plus_az if sign > 0 else minus_az).append(win_az)
        if img_after is None and eye_rows[-1]["vision"] is not None:
            img_after = eye_rows[-1]["vision"]
        null_rows, fe = run_phase(src, hold_current(fe.bones), W, rh_rows,
                                  gaze=np.zeros((1, 2)), render_every=W)
        all_rows += null_rows
        null_az.append(float(np.mean([r["gaze_az"] for r in null_rows[-5:]])))
    eye_sep_deg = float(abs(np.median(plus_az) - np.median(minus_az)))
    null_sep_deg = float(abs(np.median(null_az[0::2]) - np.median(null_az[1::2])))
    img_delta = (None if (img_before is None or img_after is None)
                 else float(np.mean(np.abs(img_after - img_before))))
    tok_gaze_err = max((abs(r["tok_gaze_az"] - r["gaze_az"]) for r in all_rows
                        if r["tok_gaze_az"] is not None), default=None)
    out["contracts"]["vision"] = {
        "eye_response_separation_deg": round(eye_sep_deg, 3),
        "null_windows_same_estimator_deg": round(null_sep_deg, 3),
        "image_delta_across_eye_command": None if img_delta is None else round(img_delta, 5),
        "learner_gaze_metadata_max_err_deg": None if tok_gaze_err is None
        else round(float(tok_gaze_err), 4),
        "pass": bool(eye_sep_deg > 3.0 * null_sep_deg and eye_sep_deg > 0.0
                     and img_delta is not None and img_delta > 0.0
                     and tok_gaze_err is not None and tok_gaze_err < 0.1),
        "meaning": "opposite gaze commands must separate the MEASURED eye "
                   "orientation (eye-joint degrees) by more than 3x the same "
                   "separation estimator applied to no-command windows; the "
                   "rendered image must change across the eye command; the "
                   "learner's gaze metadata must match the measured eye "
                   "orientation within 0.1 degree",
    }

    # ---- FRESHNESS contract (kills stale/frozen frames everywhere) --------------
    # judged per contiguous segment: the eye assay's env rebuild legitimately
    # resets frame counter and sim time, so the sequence restarts there
    frame_advances, time_advances = True, True
    for seg in (all_rows[:seg_break], all_rows[seg_break:]):
        frames = [r["frame"] for r in seg]
        times = [r["sim_time"] for r in seg]
        frame_advances &= bool(all(b > a for a, b in zip(frames, frames[1:])))
        time_advances &= bool(all(b > a for a, b in zip(times, times[1:])))
    frame_advances, time_advances = bool(frame_advances), bool(time_advances)
    tok_matches = bool(all(r["tok_frame"] == r["frame"]
                           and abs(r["tok_time"] - r["sim_time"]) < 1e-5
                           for r in all_rows if r["tok_frame"] is not None)
                       and any(r["tok_frame"] is not None for r in all_rows))
    out["contracts"]["freshness"] = {
        "steps_observed": len(all_rows),
        "frame_number_strictly_advances": frame_advances,
        "sim_time_strictly_advances": time_advances,
        "learner_metadata_matches_source": tok_matches,
        "pass": bool(frame_advances and time_advances and tok_matches),
        "meaning": "every observed step must carry a strictly increasing frame "
                   "number and sim time, reproduced bit-equal in the learner's "
                   "frame metadata - a re-served old frame fails here",
    }

    out["all_pass"] = all(c["pass"] for c in out["contracts"].values())
    return out


def write_artifact(out) -> str:
    os.makedirs(os.path.join(SAVED, "contract_checks"), exist_ok=True)
    p = os.path.join(SAVED, "contract_checks",
                     f"{out['run']}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    tmp = p + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1, allow_nan=False)
    os.replace(tmp, p)
    return p


# ============================================================================ self-test
# Negative controls as executable code: every fake below models a real defect
# class from the 2026-09-08 review and MUST fail its named contract. A checker
# revision that lets any of them pass is itself broken.

def _fake_tokenized(bones, press_rows_256, frame, sim_time, gaze_az, motor_handles):
    md = types.SimpleNamespace(frame_number=frame, timestamp=sim_time,
                               gaze_azimuth_deg=gaze_az, gaze_elevation_deg=0.0)
    return types.SimpleNamespace(metadata=md, touch_impulses=press_rows_256,
                                 touch_valid_counts=(2, 1), motor_handles=motor_handles)


class _FakeBase:
    """Two patches: row 0 = right_hand, row 1 = left_foot. `n_rows` body rows: 24 is the
    legacy layout, 26 is trunk_v1 (every H84 segment), whose lower trunk and chest rows
    come after the eyes' row."""
    patch_names = ["right_hand", "left_foot"]
    n_patches = 2

    def __init__(self, n_rows=24):
        self.i = 0
        self.n_rows = int(n_rows)
        self.env = types.SimpleNamespace(patch_names=lambda: self.patch_names)

    def _order(self, motor):
        """The order as datasource.SingleEnvSource.step reads it: an order sized for the
        other body layout is refused, as the real world refuses it."""
        return np.asarray(motor).reshape(-1, self.n_rows, 9)[0]

    def bones(self):
        b = np.zeros((self.n_rows, 11))
        b[:, 3] = 1.0
        b[:, 4:7] = [20.0, 30.0, 40.0]
        return b

    def frame_obj(self, render):
        b = self.bones()
        patches = self.patches()
        imp = np.zeros((256, 8), dtype=np.float32)
        imp[:2, 0] = patches[:2, 0] / 100000.0
        mh = np.zeros((len(b), 10), dtype=np.float32)
        mh[:, 0:3] = b[:, 4:7] / 200.0
        mh[:, 3:7] = b[:, 0:4]
        fr, tm_ = self.stamp()
        return types.SimpleNamespace(
            bones=b, patches=patches, n_contacts=1, frame=fr, sim_time=tm_,
            gaze_deg=(0.0, 0.0), eye_angles=((0.0, 0.0), (0.0, 0.0)),
            tokenized=_fake_tokenized(b, imp, fr, tm_, 0.0, mh),
            meta=self.meta(),
            vision_patches=(np.full((2, 768), 0.5 if self.i > 20 else 0.0,
                                    dtype=np.float32) if render else None))

    def patches(self):
        return np.zeros((2, 10))

    def stamp(self):
        return self.i, self.i * 0.01

    def meta(self):
        return {"handed_ctrl": np.full(90, 0.1 + 0.001 * self.i)}

    def reset(self):
        return [self.frame_obj(True)]

    def step(self, motor, gaze, mask, render=None):
        self._order(motor)
        self.i += 1
        return [self.frame_obj(bool(render))]

    def rebuild(self, **changes):
        pass

    def close(self):
        pass


class StaleUnrelatedFake(_FakeBase):
    """Review fake: constant 42 Pa on an UNRELATED body (left foot), commands
    ignored with no applied-command record, frame number and sim time frozen,
    body never moves. Must fail touch, action, vision and freshness."""

    def patches(self):
        p = np.zeros((2, 10))
        p[1, 0] = 42.0
        return p

    def stamp(self):
        return 0, 0.0

    def meta(self):
        return {}


class CollisionOnlyFake(_FakeBase):
    """Contacts counted, zero pressure everywhere. Must fail touch."""


class StaleImageFake(_FakeBase):
    """Camera images frozen while EYE FEEDBACK WORKS (gaze_deg answers commands,
    frames advance). Isolates the stale-image failure mode: advancing frame
    metadata alone must never certify fresh captures. Must fail vision."""

    def __init__(self, n_rows=24):
        super().__init__(n_rows)
        self.gz = 0.0

    def step(self, motor, gaze, mask, render=None):
        self._order(motor)
        self.i += 1
        if gaze is not None:
            g = float(np.asarray(gaze).reshape(-1)[0])
            self.gz = 30.0 * (1 if g > 0 else (-1 if g < 0 else 0))
        return [self.frame_obj(bool(render))]

    def frame_obj(self, render):
        f = super().frame_obj(render)
        f.gaze_deg = (self.gz, 0.0)
        f.tokenized.metadata.gaze_azimuth_deg = self.gz
        if f.vision_patches is not None:
            f.vision_patches = np.full((2, 768), 0.5, dtype=np.float32)  # frozen image
        return f


class CommandEchoFake(_FakeBase):
    """Achieved state is a verbatim echo of the last command (the UE export
    defect: command targets where Python expects achieved movement). Must fail
    the action contract even though everything 'moves'."""

    def __init__(self, n_rows=24):
        super().__init__(n_rows)
        self.last_cmd = None

    def step(self, motor, gaze, mask, render=None):
        self.i += 1
        self.last_cmd = self._order(motor).copy()
        return [self.frame_obj(bool(render))]

    def bones(self):
        b = super().bones()
        if self.last_cmd is not None:
            b[:, 4:7] = self.last_cmd[:, 0:3]
            b[:, 0:4] = self.last_cmd[:, 3:7]
        return b


def self_test() -> int:
    """Run every negative control; each must FAIL its named contract. Every fake runs in
    both body layouts: 24 rows (legacy) and 26 rows (trunk_v1, every H84 segment). Until
    2026-09-23 every fake had 24 rows, so the self-test passed while the checker could not
    drive a trunk_v1 world at all."""
    failures = []
    for n_rows in (24, 26):
        for fake_cls, must_fail in ((StaleUnrelatedFake, ("touch", "action", "vision", "freshness")),
                                    (CollisionOnlyFake, ("touch",)),
                                    (StaleImageFake, ("vision",)),
                                    (CommandEchoFake, ("action",))):
            label = f"{fake_cls.__name__} ({n_rows} body rows)"
            out = run_contracts(fake_cls(n_rows), fake_cls.patch_names, fake_cls.n_patches,
                                settle_steps=10, phase_steps=10, run_name=label)
            if out["all_pass"]:
                failures.append(f"{label}: reported all_pass=true")
            for c in must_fail:
                if out["contracts"][c]["pass"]:
                    failures.append(f"{label}: contract '{c}' passed but must fail")
    if failures:
        print("SELF-TEST FAILURE - the checker accepted a defective source:")
        for msg in failures:
            print("  " + msg)
        return 1
    print("SELF-TEST PASS: all negative controls (unrelated stale pressure + ignored "
          "commands + frozen frames; collision-only; frozen images with working eye "
          "feedback; command-echo achieved state) fail as required, in the 24-row "
          "legacy body and the 26-row trunk_v1 body")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run dir whose args.json + recorded phase shape the env")
    ap.add_argument("--settle", type=int, default=60)
    ap.add_argument("--phase-steps", type=int, default=50)
    ap.add_argument("--phase", default=None, help="curriculum phase override")
    ap.add_argument("--min-free-ram-gb", type=float, default=6.0)
    ap.add_argument("--self-test", action="store_true",
                    help="run built-in negative controls; every fake must FAIL")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.run:
        ap.error("--run is required (or use --self-test)")
    # QUALIFICATION IS MANDATORY (follow-up review 2026-09-08: "the sensor guard
    # is optional" - it is not any more): every assay run first proves the
    # checker itself against the negative-control fakes, and the verdict is
    # embedded in the artifact. A checker that cannot fail the fakes produces
    # no PASS for anything.
    if self_test() != 0:
        print("REFUSED: checker failed its own negative controls - no assay run")
        return 3
    free = free_ram_gb()
    if free < args.min_free_ram_gb:
        print(f"REFUSED: {free:.1f} GB free RAM < {args.min_free_ram_gb} GB floor "
              "(training is the priority workload)")
        return 2

    kw, phase = resolved_env_kwargs(args.run, args.phase)
    from tokenizer.config import TokenizerConfig
    rng_cm = float(TokenizerConfig().position_range_cm)
    from mimo_bridge.datasource import SingleEnvSource
    src = SingleEnvSource(**kw)
    try:
        out = run_contracts(src, src.env.patch_names(), src.n_patches,
                            args.settle, args.phase_steps,
                            os.path.basename(os.path.normpath(args.run)), rng_cm=rng_cm)
    finally:
        src.close()
    out["qualification"] = ("self_test_pass: all negative-control fakes failed as "
                            "required, run in this process before the assay")
    out["curriculum_phase"] = phase
    out["resolved_env_kwargs"] = {k: v for k, v in kw.items() if k != "run_dir"}
    out["env_parity_notes"] = ("seed is the run's BASE seed: the trainer offsets the "
                               "env seed by resume step per segment (train_mimo "
                               "env-seed note), so this probe world is a valid fresh "
                               "sample of the same distribution, not a bit-identical "
                               "replay of any one segment")
    src_dir = os.path.join(ROOT, "mimo_bridge")
    out["source_sha256"] = {
        "checker": _sha256(os.path.abspath(__file__)),
        "env.py": _sha256(os.path.join(src_dir, "env.py")),
        "datasource.py": _sha256(os.path.join(src_dir, "datasource.py")),
        "touch_adapter.py": _sha256(os.path.join(src_dir, "touch_adapter.py")),
        "frame_encoder.py": _sha256(os.path.join(src_dir, "frame_encoder.py")),
    }
    p = write_artifact(out)
    print(json.dumps(out, indent=1))
    print(("ALL CONTRACTS PASS" if out["all_pass"] else "CONTRACT FAILURE") + f" -> {p}")
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
