# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The overseer: watches every training run and steps in, wearing three hats.

    python -m mimo_bridge.tools.overseer            # loop every 3 minutes
    python -m mimo_bridge.tools.overseer --once

The hats and what each one checks:
  CTO                 Is the project on track and the machine safe? Free RAM and disk,
                      snapshots actually happening, run not silently dead, cost (speed).
  AI researcher       Is the model gaining function or fooling us? NaN, encoder collapse
                      (token spread), prediction error direction per sense, curiosity stuck
                      at zero, situational awareness (motor discrimination) trend.
  Simulation designer Is the world worth learning from? Are there any touches at all, is
                      the ball ever reached, are the boxes ever touched, did the body get
                      flung out of the pen (physics blow-up), is the sim speed sane.

What it can DO (not just say): it writes commands into <run>/control.json which the
trainer executes BETWEEN steps, so a stop can never interrupt a snapshot:
  snapshot            save all weights now
  snapshot_and_stop   save, then stop cleanly (used for NaN / collapse / low RAM)
  video               record a 30 s clip right now
Commands are only sent to runs that advertise <run>/control_supported (older runs are
observe-only). Every finding and action goes to <run>/overseer_log.jsonl and the
dashboard shows the latest ones. Hard interventions are rare on purpose: the overseer
prefers a note over a stop unless data is being corrupted (NaN), learning is provably
dead (collapse), or the machine is about to fall over (RAM).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
SAVED = os.path.join(REPO, "Saved", "mimo_bridge")

WINDOW = 30          # look at the last N "mimo" records
# Defaults; the META-OVERSEER retunes these with evidence via overseer_directives.json
COLLAPSE_STD = 0.02  # token temporal spread below this = encoder went flat
MIN_RAM_GB = 2.5
MIN_DISK_GB = 8.0
_DIRECTIVES_PATH = os.path.join(SAVED, "overseer_directives.json")


def apply_directives():
    """Read the meta-overseer's current thresholds (bounded there, trusted here)."""
    global COLLAPSE_STD, MIN_RAM_GB, MIN_DISK_GB, STALL_MIN
    try:
        d = json.load(open(_DIRECTIVES_PATH))
    except Exception:
        return None
    COLLAPSE_STD = float(d.get("COLLAPSE_STD", COLLAPSE_STD))
    MIN_RAM_GB = float(d.get("MIN_RAM_GB", MIN_RAM_GB))
    MIN_DISK_GB = float(d.get("MIN_DISK_GB", MIN_DISK_GB))
    try:
        STALL_MIN = float(d.get("STALL_MIN", STALL_MIN))
    except NameError:
        pass
    return d.get("updated")


def read_tail(path, kind, n=WINDOW):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if f'"{kind}"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("type") == kind:
                    out.append(r)
    except FileNotFoundError:
        pass
    return out[-n:]


def free_gb():
    try:
        import psutil
        ram = psutil.virtual_memory().available / 1e9
    except Exception:
        ram = 99.0
    disk = shutil.disk_usage("C:\\").free / 1e9
    return ram, disk


STALL_MIN = 10.0     # a run whose stats file has not moved for this long is stuck
RAM_GRACE_MIN = 6.0  # no RAM stop until a run is this old (16 workers take minutes to spawn)


def run_age_min(run) -> float:
    """Minutes since this run started, from its args.json (written once, at launch)."""
    try:
        return (time.time() - os.path.getmtime(os.path.join(run, "args.json"))) / 60.0
    except OSError:
        return 1e9


def run_processes(run):
    """python processes belonging to this run (matched by the run folder name in the command line)."""
    import psutil
    name = os.path.basename(run)
    procs = []
    for pr in psutil.process_iter(["name", "cmdline"]):
        try:
            if pr.info["name"] and "python" in pr.info["name"].lower():
                cmd = " ".join(pr.info["cmdline"] or [])
                if name in cmd and "train_mimo" in cmd:
                    procs.append(pr)
        except Exception:
            continue
    return procs


def stalled_runs():
    """Runs with no summary, a stale stats file = stuck (hung trainer or silent death)."""
    out = []
    for base in (os.path.join(SAVED, "experiments"), os.path.join(SAVED, "runs")):
        for d in glob.glob(os.path.join(base, "*")):
            st = os.path.join(d, "replay_training_stats.jsonl")
            if not (os.path.isdir(d) and os.path.exists(st)):
                continue
            if os.path.exists(os.path.join(d, "summary.json")) and                     os.path.getmtime(os.path.join(d, "summary.json")) > os.path.getmtime(st):
                continue          # summary newer than the stats = genuinely done
            age_min = (time.time() - os.path.getmtime(st)) / 60
            if age_min > STALL_MIN:
                out.append((d, age_min))
    return out


# a run may be auto-restarted at most this many times, and no more often than this, so a
# genuinely broken configuration cannot spin forever
MAX_RESTARTS = 4
RESTART_COOLDOWN_MIN = 15.0


def restart_dead(run, age_min, note):
    """Relaunch a dead run from its newest checkpoint, using the args it was started with.

    A run that dies SILENTLY - no traceback, no summary, the whole process tree gone - was
    invisible to every alarm we had: the log watcher matches crash text and a silent death
    writes none, so silence looked exactly like success (H13 leg 2, 2026-08-23: dead at
    00:04, noticed at 03:21). The overseer could see it and say so, but had no power to act.
    It does now: same output directory, same flags, resumed from the newest snapshot, so at
    most one hour of training is lost and the experiment's continuity is preserved.
    """
    import subprocess, sys
    args_p = os.path.join(run, "args.json")
    state_p = os.path.join(run, "restarts.json")
    try:
        args = json.load(open(args_p))
    except Exception as ex:
        note["actions"].append(f"cannot restart (no args.json): {ex}")
        return False
    state = {"count": 0, "last": 0.0}
    try:
        state.update(json.load(open(state_p)))
    except Exception:
        pass
    if state["count"] >= MAX_RESTARTS:
        note["actions"].append(f"NOT restarting: already restarted {state['count']}x "
                               f"(cap {MAX_RESTARTS}) - this needs a human")
        return False
    if (time.time() - state["last"]) / 60 < RESTART_COOLDOWN_MIN:
        note["actions"].append("NOT restarting yet: inside the cooldown window")
        return False
    ckpts = sorted(glob.glob(os.path.join(run, "checkpoints", "ckpt_step_*")),
                   key=lambda p: int(p.rsplit("_", 1)[-1]))
    if not ckpts:
        note["actions"].append("cannot restart: no checkpoint to resume from")
        return False
    newest = ckpts[-1]
    # rebuild the command line from args.json, forcing the resume and dropping the elapsed time
    cmd = [sys.executable, "-m", "mimo_bridge.train_mimo"]
    skip = {"resume"}
    for k, v in args.items():
        if k in skip or v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            cmd.append(flag)
        else:
            cmd += [flag, str(v)]
    cmd += ["--resume", newest]
    src = os.path.abspath(os.path.join(HERE, "..", ".."))
    log = os.path.join(run, "restart_%d.log" % int(time.time()))
    try:
        with open(log, "w") as lf:
            subprocess.Popen(cmd, cwd=src, stdout=lf, stderr=subprocess.STDOUT,
                             creationflags=0x00000008)      # DETACHED_PROCESS
        state["count"] += 1
        state["last"] = time.time()
        json.dump(state, open(state_p, "w"), indent=1)
        note["actions"].append(f"RESTARTED from {os.path.basename(newest)} "
                               f"(restart {state['count']} of {MAX_RESTARTS}); log {os.path.basename(log)}")
        return True
    except Exception as ex:
        note["actions"].append(f"restart failed: {ex}")
        return False


def put_down_stalled(reports):
    """A hung trainer ignores its control file, so it is killed with its workers. Hourly
    snapshots bound the loss to at most one hour; the chain script then moves on to the
    next stage instead of waiting forever."""
    for run, age_min in stalled_runs():
        procs = run_processes(run)
        note = {"type": "overseer", "timestamp": time.time(), "run": os.path.basename(run),
                "findings": {"cto": [f"run STALLED: no stats for {age_min:.0f} min"], "researcher": [], "designer": []},
                "actions": []}
        if procs:
            for pr in procs:
                try:
                    for ch in pr.children(recursive=True):
                        ch.kill()
                    pr.kill()
                    note["actions"].append(f"killed hung trainer pid {pr.pid} and its workers")
                except Exception as ex:
                    note["actions"].append(f"kill failed: {ex}")
        else:
            note["findings"]["cto"].append("no trainer process found: it died without writing a summary")
        # bring it back before writing it off: a silent death should cost one hour, not a night
        revived = restart_dead(run, age_min, note)
        if revived:
            with open(os.path.join(run, "overseer_log.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(note) + "\n")
            reports[os.path.basename(run) + ":restarted"] = note
            continue
        # ...and record that death, so the run stops being listed as a live stall forever
        # (two E5 runs sat in the dashboard for two days as 'stopped, no news for N min')
        try:
            if not os.path.exists(os.path.join(run, "summary.json")):
                json.dump({"ok": False, "abandoned": True,
                           "note": f"no stats for {age_min:.0f} min and no trainer process: marked abandoned by the overseer",
                           "silent_min": round(age_min)},
                          open(os.path.join(run, "summary.json"), "w"), indent=2)
                note["actions"].append("marked abandoned (summary written)")
        except OSError:
            pass
        with open(os.path.join(run, "overseer_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(note) + "\n")
        reports[os.path.basename(run) + ":stalled"] = note


def keep_dashboard_alive():
    """The overseer also restarts the dashboard loop if it has gone away."""
    import psutil, subprocess, sys
    for pr in psutil.process_iter(["name", "cmdline"]):
        try:
            if pr.info["name"] and "python" in pr.info["name"].lower() and "tools.dashboard" in " ".join(pr.info["cmdline"] or []):
                return False
        except Exception:
            continue
    src = os.path.abspath(os.path.join(HERE, "..", ".."))
    subprocess.Popen([sys.executable, "-m", "mimo_bridge.tools.dashboard"], cwd=src,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=0x00000008)   # DETACHED_PROCESS
    return True


def active_runs():
    runs = []
    for base in (os.path.join(SAVED, "experiments"), os.path.join(SAVED, "runs")):
        for d in glob.glob(os.path.join(base, "*")):
            st = os.path.join(d, "replay_training_stats.jsonl")
            # freshness decides, not the presence of a summary: a resumed run reuses its
            # folder and would otherwise go unsupervised behind the old summary
            if os.path.isdir(d) and os.path.exists(st) and time.time() - os.path.getmtime(st) < 240:
                runs.append(d)
    return runs


def send(run, cmd):
    """One command at a time; only to trainers that support the channel; never overwrite a
    command that has not been picked up yet."""
    if not os.path.exists(os.path.join(run, "control_supported")):
        return "run does not support commands (observe-only)"
    cp = os.path.join(run, "control.json")
    if os.path.exists(cp):
        return "previous command still pending"
    json.dump({"cmd": cmd, "from": "overseer", "time": time.time()}, open(cp, "w"))
    return f"sent {cmd}"


def slope(xs, ys):
    if len(xs) < 6:
        return 0.0
    a = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)[0]
    return float(a)


def inspect(run) -> dict:
    recs = read_tail(os.path.join(run, "replay_training_stats.jsonl"), "mimo")
    findings = {"cto": [], "researcher": [], "designer": []}
    actions = []
    # Semantic vitals, not just process health. Every serious bug of 2026-08-23/24 -
    # frozen eyes, face-down posture, dead saccades, unlogged instruments - happened in
    # runs this overseer called healthy, because it only ever watched liveness and RAM.
    # vitals.py band-checks the CONTENT of the stats; a missing preflight.json means the
    # run was launched without its effect probes.
    try:
        from mimo_bridge.tools import vitals as V
        for p in V.problems(V.evaluate(os.path.join(run, "replay_training_stats.jsonl"))):
            findings["cto"].append("VITALS: " + p)
    except Exception as ex:
        findings["cto"].append(f"vitals check itself failed: {ex}")
    if not os.path.exists(os.path.join(run, "preflight.json")):
        findings["cto"].append("VITALS: no preflight.json - run launched without effect probes")
    else:
        try:
            pf = json.load(open(os.path.join(run, "preflight.json")))
            if not pf.get("ok"):
                bad = [c["name"] for c in pf.get("checks", []) if not c.get("ok")]
                findings["cto"].append(f"VITALS: preflight FAILED on {bad}")
        except Exception:
            findings["cto"].append("VITALS: preflight.json unreadable")
    if not recs:
        findings["cto"].append("no records yet")
        return {"findings": findings, "actions": actions}
    last = recs[-1]
    steps = [r["step"] for r in recs]

    # ---------------- CTO hat
    ram, disk = free_gb()
    if ram < MIN_RAM_GB:
        # launch transients dip RAM for well under a minute (16 workers spawning); a run
        # is only stopped if the shortage PERSISTS across a re-sample - two healthy runs
        # were killed at startup by single-sample readings
        time.sleep(45)
        ram2, _ = free_gb()
        age = run_age_min(run)
        if ram2 >= MIN_RAM_GB:
            findings["cto"].append(f"RAM dipped to {ram:.1f} GB but recovered to {ram2:.1f} GB (launch transient): no action")
        elif age < RAM_GRACE_MIN:
            # A 45 s re-sample is not enough: 16 workers take minutes to finish spawning,
            # so BOTH samples can land inside the startup transient. That is how H21_eyes
            # was stopped 166 s after launch - the meta-overseer caught it afterwards as a
            # false kill, but the guard itself had no notion of run AGE. Report, never stop.
            findings["cto"].append(
                f"RAM low ({ram:.1f} then {ram2:.1f} GB) but the run is only {age:.1f} min old "
                f"(grace {RAM_GRACE_MIN:.0f} min): startup transient, NOT stopping")
        else:
            findings["cto"].append(f"RAM nearly gone ({ram:.1f} then {ram2:.1f} GB free, persistent): saving weights and stopping before a crash loses work")
            actions.append(send(run, "snapshot_and_stop"))
    if disk < MIN_DISK_GB:
        findings["cto"].append(f"disk low ({disk:.0f} GB free): old checkpoints/videos should be pruned soon")
    ck = os.path.join(run, "checkpoints.json")
    try:
        cks = json.load(open(ck))
        age_min = (time.time() - os.path.getmtime(ck)) / 60
        if age_min > 75:
            findings["cto"].append(f"last snapshot {age_min:.0f} min ago (should be hourly): requesting one")
            actions.append(send(run, "snapshot"))
    except Exception:
        if (time.time() - os.path.getctime(run)) / 60 > 75:
            findings["cto"].append("no snapshot yet after 75 min: requesting one")
            actions.append(send(run, "snapshot"))
    sps = [r.get("model_steps_per_s") for r in recs if r.get("model_steps_per_s")]
    if len(sps) > 10 and sps[-1] < 0.4 * float(np.median(sps)):
        findings["cto"].append(f"speed collapsed: {sps[-1]:.1f} steps/s vs usual {np.median(sps):.1f}")

    # ---------------- AI researcher hat
    # CAN THE MODEL READ ITS OWN ACTION? Checked on the input, per checkpoint, because
    # weight x input is what reaches the network: from H28 to H33 the command columns
    # carried live weights (|w| 0.083) while 99.6% of the motor row's input energy was a
    # position block in centimetres, and no causal test in that era could resolve the
    # action. That is the cheapest check for the most expensive failure this project has
    # had, so the watchdog makes it rather than waiting for a person.
    try:
        from mimo_bridge.tools.probes import (TURN_DEG, action_readability, per_handle_readability,
                                              synthetic_motor_rows)
        from mimo_bridge.tools.dashboard import HANDLE_READ_LINE, load_motor_encoder, readability_rows
        from mimo_bridge.tools.plain_numbers import pct, per
        cks = sorted(glob.glob(os.path.join(run, "checkpoints", "ckpt_step_*", "model.safetensors")),
                     key=lambda q: (int(re.search(r"ckpt_step_(\d+)", q).group(1)), os.path.getmtime(q)))
        if cks and getattr(inspect, "_read_ck", None) != cks[-1]:
            inspect._read_ck = cks[-1]
            # Only the motor encoder, built at the run's own body layout (the checkpoint's
            # receipt) and size (args.json). This built a default LemLite - the 24-row legacy
            # body at the lite width - and strict=False does not forgive a size mismatch, so on
            # every H84 checkpoint (26 rows, 192 wide) it printed "could not run" (2026-09-23).
            # None = the checkpoint predates the 10 -> 19 widening: nothing to read.
            loaded = load_motor_encoder(cks[-1], run)
            if loaded is not None:
                enc, layout, names = loaded
                n = len(names)
                rr = action_readability(enc, synthetic_motor_rows(n=n))
                inspect._read_val = rr
                at_once = (f"changing all {n} body parts' orders at once moves the motor token "
                           f"{pct(rr['diff_action'] / max(rr['pos_block'], 1e-9))} as far as zeroing "
                           f"the orders' position block does (0% = the encoder cannot see the orders; "
                           f"under 20% counts as drowned)")
                if not rr["readable"]:
                    findings["researcher"].append(
                        f"the model CANNOT READ ITS OWN ACTION: {at_once}. No causal test can resolve the "
                        f"action in this state - check the units at the encoder before spending more hours")
                else:
                    # one body part at a time (reviews/dashboards.md M9): the all-at-once reading
                    # passed on ckpt_step_344400 while 10 of its 26 parts could not be read at all
                    rows = readability_rows(layout, n)
                    if rows is None:
                        findings["researcher"].append(
                            f"action readable with every order changed together - {at_once}; "
                            f"per-body-part readability unmeasured (no recorded rows "
                            f"for the '{layout}' body layout)")
                    else:
                        from mimo_bridge.tools.development_dashboard import handle_words
                        share = per_handle_readability(enc, rows)["share"]
                        below = [nm for nm, s in zip(names, share) if s < HANDLE_READ_LINE]
                        line = per(HANDLE_READ_LINE, "parts of its size")
                        findings["researcher"].append(
                            (f"action UNREAD for {len(below)} of {n} body parts: a {TURN_DEG:.0f}-degree "
                             f"turn of one of them moves the motor token under {line} ("
                             + ", ".join(handle_words(x) for x in below) + ")"
                             if below else
                             f"action readable for all {n} body parts: each one's {TURN_DEG:.0f}-degree "
                             f"turn moves the motor token above {line}")
                            + f"; {at_once}")
    except Exception as _ex:
        findings["researcher"].append(
            f"action-readability check could not run: {type(_ex).__name__}: {_ex}")

    if sum(r.get("nan_steps", 0) for r in recs) > 0:
        findings["researcher"].append("NaN appeared in training: weights are being corrupted, saving and stopping")
        actions.append(send(run, "snapshot_and_stop"))
    stds = (last.get("token_temporal_std") or {})
    flat = [m for m, v in stds.items() if m in ("vision", "tactile", "proprioception") and v is not None and v < COLLAPSE_STD]
    if flat:
        findings["researcher"].append(f"encoder collapse in {flat} (token spread < {COLLAPSE_STD}): learning is dead, saving and stopping")
        actions.append(send(run, "snapshot_and_stop"))
    for m in ("vision", "tactile", "proprioception"):
        # next_pe is the prediction. `pe` is predicted change (heads vs the present) and
        # rises when the model starts predicting movement - not a regression.
        ys = [(r.get("next_pe") or {}).get(m) for r in recs if (r.get("next_pe") or {}).get(m) is not None]
        if len(ys) > 10:
            sl = slope(steps[-len(ys):], ys)
            if sl > 0 and ys[-1] > 2 * min(ys) and ys[-1] > 0.05:   # absolute floor: tiny wobbles are not regressions
                findings["researcher"].append(f"prediction of {m} is getting WORSE (now {ys[-1]:.3g}, best was {min(ys):.3g})")
    drives = [r.get("curiosity_drive") for r in recs if r.get("curiosity_drive") is not None]
    if len(drives) > 10 and max(drives) == 0.0:
        pe_now = np.mean([(last.get("next_pe") or last.get("pe") or {}).get(m, 0) for m in ("vision", "tactile", "proprioception")])
        if pe_now < 0.1:
            # Advance ONLY with patience (fixed 23:54: it advanced twice in 3 minutes because
            # prediction error re-converges right after each change): the current phase must
            # have run 20+ min and no advance in the last 30 min.
            phase_now = last.get("phase")
            same = [r for r in recs if r.get("phase") == phase_now]
            span_min = (same[-1]["timestamp"] - same[0]["timestamp"]) / 60 if len(same) > 2 else 0.0
            cmds = read_tail(os.path.join(run, "replay_training_stats.jsonl"), "overseer_cmd", n=20)
            last_adv = max((c["timestamp"] for c in cmds if c.get("cmd") == "advance_phase"), default=0)
            cool_min = (time.time() - last_adv) / 60
            if phase_now == "mature":
                # final stage: there is nothing to advance to; sending advance_phase here is
                # a dead command that only pollutes the meta-overseer's intervention audit
                findings["researcher"].append("bored at the final ('mature') stage: curriculum complete, nothing to advance to; boredom here means the world needs richer tasks, not a phase change")
            elif span_min >= 20 and cool_min >= 30:
                findings["researcher"].append(f"bored of the '{phase_now}' stage for {span_min:.0f} min straight: advancing the growth phase")
                actions.append(send(run, "advance_phase"))
            else:
                findings["researcher"].append(f"bored of '{phase_now}' but being patient (stage ran {span_min:.0f} min, last advance {cool_min:.0f} min ago; needs 20/30): prediction alone is not stage mastery")
        else:
            findings["researcher"].append("curiosity reads 'overwhelmed' (prediction error too high across the board); if this persists the world or curriculum is too hard for this stage")
    dr = [r.get("motor_discrimination_ratio") for r in recs if r.get("motor_discrimination_ratio") is not None]
    if dr:
        # renamed 2026-08-26 with the dashboard: the ratio is an uncalibrated between-body
        # proxy, and "target > 1.5" was an invented threshold - report the value plainly
        findings["researcher"].append(
            f"parallel-body action-signature ratio (uncalibrated proxy) now {dr[-1]:.2f}")

    # ---------------- simulation designer hat
    contacts = [r.get("contacts_per_env", 0) for r in recs]
    if len(contacts) > 10 and float(np.mean(contacts)) < 0.2:
        findings["designer"].append("almost no touching is happening: this world is giving the skin nothing to learn from")
    bd = [r.get("task_hand_ball_dist_m") for r in recs if r.get("task_hand_ball_dist_m") is not None]
    br = [r.get("task_ball_touch_rate") or 0 for r in recs]
    if bd:
        trend = "closing in" if len(bd) > 5 and bd[-1] < bd[0] - 0.01 else ("drifting away" if len(bd) > 5 and bd[-1] > bd[0] + 0.01 else "holding")
        findings["designer"].append(f"ball: {bd[-1]*100:.0f} cm from the hand ({trend}); touched in {100*float(np.mean(br)):.0f}% of checks")
        if len(bd) > 20 and max(br) == 0 and min(bd) > 0.45:
            findings["designer"].append("the ball has been out of reach for the whole window: placement should be tightened")
    pr = [r.get("task_pressing") for r in recs
          if isinstance(r.get("task_pressing"), (int, float))]
    if len(pr) >= 6:
        now, then = float(np.mean(pr[-len(pr) // 3:])), float(np.mean(pr[:len(pr) // 3]))
        # Random babble presses on 90% of steps (commit 2f39bf4b), so 90% is a floor to
        # sit AT, not a score to celebrate - and a FALL is worth saying out loud. H28
        # went 0.90 -> 0.49 and this hat mentioned pressing zero times in 213 entries,
        # because it watched the ball and the boxes and never the hands.
        note = ("at the random-babble baseline" if now > 0.85 else
                f"DOWN from {then:.0%} earlier in this window" if now < then - 0.15
                else "steady")
        findings["designer"].append(
            f"pressing on things: {now:.0%} of steps, {note} "
            "(random babble presses on 90%, so that is the floor, not an achievement)")
    boxes = last.get("task_box_touch_rate") or {}
    if boxes:
        findings["designer"].append("boxes touched (L/M/H): " + "/".join(f"{100*boxes.get(k,0):.0f}%" for k in ("light", "medium", "heavy")))
    if last.get("touch_mean_pressure_pa", 0) > 5e6:
        findings["designer"].append("skin pressures look explosive (physics may be unstable)")
    rolls = [r.get("roll_deg_max") for r in recs if r.get("roll_deg_max") is not None]
    if rolls:
        rf = last.get("rolled_over_fraction") or 0
        if max(rolls) > 120:
            findings["designer"].append(f"MILESTONE: rolling over is happening (max trunk rotation {max(rolls):.0f} deg; {100*rf:.0f}% of bodies rolled over this episode)")
        elif max(rolls) > 60:
            findings["designer"].append(f"the babies get onto their sides (max trunk rotation {max(rolls):.0f} deg) but have not rolled over yet")
        else:
            findings["designer"].append(f"no rolling yet (max trunk rotation {max(rolls):.0f} deg); watch whether this improves as strength grows")

    # ------------- the verdict: one plain paragraph that actually takes a position
    goal_bits = []
    # Scientific progress uses the action-conditioned next-state error.  The `pe`
    # field is current-frame reconstruction and can become tiny on a still body without
    # learning action -> consequence.  Never silently fall back from causal PE to it.
    # A SINGLE DIVERGING SENSE IS INVISIBLE IN AN AVERAGE. Measured on H32 while the
    # verdict below read "strong": tactile next-state error +68% over the run's
    # windows while vision fell 63% and proprioception 65%. The aggregate stayed
    # under every band, so nothing said one sense was going the wrong way. Split
    # the overseer's own window in half per modality and report any sense whose
    # error rose >20% while another fell >20% - a direction split, not a threshold
    # on the level.
    # A DRIFT IS SLOWER THAN THE MAIN WINDOW. WINDOW is 30 records, about 15 minutes at
    # this step rate, and a sense can go the wrong way for hours while every 15-minute
    # view of it looks flat - which is what H35 did. This check reads its own longer tail.
    _long = read_tail(os.path.join(run, "replay_training_stats.jsonl"), "mimo", n=240)
    if len(_long) >= 40:
        half = len(_long) // 2
        _chg = {}
        for _m in ("vision", "tactile", "proprioception"):
            a = [r["next_pe"][_m] for r in _long[:half]
                 if isinstance(r.get("next_pe"), dict) and r["next_pe"].get(_m) is not None]
            b = [r["next_pe"][_m] for r in _long[half:]
                 if isinstance(r.get("next_pe"), dict) and r["next_pe"].get(_m) is not None]
            if a and b and np.mean(a) > 0:
                _chg[_m] = 100.0 * (np.mean(b) - np.mean(a)) / np.mean(a)
        rising = [m for m, c in _chg.items() if c > 20]
        falling = [m for m, c in _chg.items() if c < -20]
        # A SENSE CAN DIVERGE ALONE. The rule was written for H32, where touch rose
        # 68% while vision fell 63%, and required BOTH - so on H35, where touch rose
        # 80% while vision and body position stayed flat, it never fired and the
        # overseer reported ball distances for six passes while the drift ran. Going
        # the wrong way while the others do not is the same finding; the peers do not
        # have to improve for one sense to be in trouble.
        if len(rising) == 1 and not falling:
            findings["researcher"].append(
                f"{rising[0]} is going the wrong way on its own: "
                + ", ".join(f"{m} {_chg[m]:+.0f}%" for m in sorted(_chg))
                + " across the two halves of the last %d records. The others are flat, so" % len(_long)
                + " this is not the world changing under all of them.")
        if rising and falling:
            findings["researcher"].append(
                "one sense is diverging from the rest: "
                + ", ".join(f"{m} {_chg[m]:+.0f}%" for m in rising)
                + " while " + ", ".join(f"{m} {_chg[m]:+.0f}%" for m in falling)
                + " - an aggregate verdict cannot see this split")

    # follow-up review 2026-09-08: "the overseer still selects developmental-
    # sounding wording from a recent prediction-error threshold". The verdict now
    # keeps THREE labeled evidence classes apart: PREDICTION (measured error
    # levels - never ability language), ABILITY (behavioral test artifacts ONLY),
    # and vitals. A prediction-error level can never produce an ability claim.
    _causal = last.get("next_pe") or {}
    pe_real = [_causal.get(m) for m in ("vision", "tactile", "proprioception")]
    pe_real = [v for v in pe_real if v is not None and np.isfinite(v)]
    if pe_real:
        lvl = float(np.mean(pe_real))
        goal_bits.append("PREDICTION: causal next-state error is "
                         + ("low" if lvl < 0.05 else
                            ("mid" if lvl < 0.3 else "high"))
                         + f" ({lvl:.3f}; prediction quality only - says nothing "
                           "about acquired ability)")
    else:
        goal_bits.append("PREDICTION: no valid causal next-state data")
    ability_bits = []
    # behavioral acquisition evidence: the skill battery's frozen-checkpoint task
    # comparison (tools/skill_battery.py), cited when a result exists, else the
    # absence is said out loud
    try:
        import glob as _glob
        _sb = sorted(_glob.glob(os.path.join(os.path.dirname(os.path.dirname(
            os.path.normpath(run))), "skill_battery", "*.json")))
        if _sb:
            _b = json.load(open(_sb[-1]))
            ability_bits.append("latest skill-battery result %s: %s" % (
                os.path.basename(_sb[-1]), _b.get("verdict", "no verdict field")))
    except Exception:
        pass
    if not ability_bits:
        ability_bits.append("no behavioral acquisition result on record - "
                            "prediction improvement is NOT ability evidence "
                            "(run tools/skill_battery.py on frozen checkpoints)")
    if rolls and max(rolls) > 120:
        ability_bits.append("trunk rotation exceeded 120 degrees; purposeful "
                            "rolling has not been established")
    dr_last = (dr[-1] if dr else None)
    if dr_last is not None:
        # 2026-09-03: the controlled situation test (tools/situation_test.py) can
        # now ESTABLISH what this ratio alone never could; cite its verdict when
        # a result exists for this run, otherwise say untested, never assert.
        _sit = None
        try:
            import json as _json
            _sp = os.path.join(run, "situation_test.json")
            if os.path.exists(_sp):
                _sit = _json.load(open(_sp))
        except Exception:
            _sit = None
        _sig = ("parallel bodies have different action signatures" if dr_last > 1.5
                else "parallel-body action signatures are not yet separated")
        if _sit is not None:
            ability_bits.append(_sig + " - controlled test %s: %s" % (
                _sit.get("when", "?"),
                "situation dependence ESTABLISHED (%.2fx over jitter)" % _sit.get("relevant_over_irrelevant", 0)
                if _sit.get("established") else
                "situation dependence not established (%.2fx over jitter, needs CI > 1)" % _sit.get("relevant_over_irrelevant", 0)))
        else:
            ability_bits.append(_sig + " (situation dependence untested for this run - "
                                "run tools/situation_test.py on a checkpoint)")
    goal_bits.append("ABILITY (behavioral evidence only): " + "; ".join(ability_bits))
    verdict = " | ".join(goal_bits) if goal_bits else "not enough data yet"
    hard = [a for a in actions if a and a.startswith("sent")]
    # The verdict must not contradict this pass's own vitals. Measured on H28: the CTO
    # hat reported "limb follows commands LOW at -0.1108 (band [0.02, 1.0])" and "body
    # speed LOW at 0.6494 (band [5.0, 120.0])" in the same entry whose verdict read
    # "parallel bodies have different action signatures (situation dependence is not established). No intervention needed right now."
    # Two failing vitals and a clean bill of health in one breath is how a report gets
    # read as reassurance. No automatic action being warranted is not the same as
    # nothing being wrong, and the verdict is the line people actually read.
    failing = [f[len("VITALS:"):].split("(band")[0].strip().rstrip("-").strip()
               for f in findings.get("cto", []) if f.startswith("VITALS:")]
    if hard:
        tail = f" Acting: {', '.join(hard)}."
    elif failing:
        tail = (f" NOT a clean bill of health: {len(failing)} vital(s) out of band"
                f" ({'; '.join(failing)}). No automatic action was warranted, which is"
                f" not the same as nothing being wrong.")
    else:
        tail = " No intervention needed right now."
    # renamed from "Gain-of-function verdict": that label let prediction numbers
    # read as function gained (follow-up review 2026-09-08)
    findings["verdict"] = [f"Verdict (prediction and ability kept separate): {verdict}." + tail]
    return {"findings": findings, "actions": [a for a in actions if a]}


def pass_once():
    reports = {}
    applied = apply_directives()
    if applied:
        reports["_directives"] = f"meta-overseer thresholds applied (updated {applied})"
    for run in active_runs():
        rep = inspect(run)
        rep.update(type="overseer", timestamp=time.time(), run=os.path.basename(run),
                   has_vitals=True)   # proof to the meta-overseer this code knows the bands
        # the writing hat (reading gate, the style guide 2026-09-05): the dashboard shows
        # notes_plain; findings stay raw for the meta-overseer's audits. The hat can
        # never block the log write.
        try:
            from mimo_bridge.tools.writing_hat import gate_report
            rep["notes_plain"], rep["writing_hat"] = gate_report(rep.get("findings") or {})
        except Exception as _wh_err:
            rep["writing_hat"] = {"error": str(_wh_err)}
        with open(os.path.join(run, "overseer_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rep) + "\n")
        reports[os.path.basename(run)] = rep
    put_down_stalled(reports)
    if keep_dashboard_alive():
        reports["_dashboard"] = "was down, restarted"
    # the plan-gap clock (operator 2026-09-06): hourly, pure log analytics. It answers
    # "how far do plans get, when would they arrive at this learning rate, and is
    # there something that can be done" - the dashboard shows it, the history file
    # keeps the clock's own record.
    try:
        from mimo_bridge.tools import gap_eta as _ge
        _gp = os.path.join(os.path.dirname(_ge.EXP), "gap_eta.json")
        if (not os.path.exists(_gp)) or (time.time() - os.path.getmtime(_gp)) >= 3600:
            _out = _ge.compute()
            if "error" not in _out:
                reports["_plan_gap"] = (f"rung {_out['rung']}/10, "
                                        f"{_out['gap_instants']:.0f} instants short; "
                                        + _out["eta_text"])
    except Exception as _ge_err:
        reports["_plan_gap"] = f"gap clock failed: {_ge_err}"
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=float, default=180.0)
    a = ap.parse_args()
    from mimo_bridge.tools.selfreload import stamp as _stamp, exec_if_changed as _reexec
    import sys as _sys
    _t0 = _stamp()
    while True:
        _reexec(_t0, "mimo_bridge.tools.overseer", _sys.argv[1:], log=print)
        reps = pass_once()
        if a.once:
            print(json.dumps(reps, indent=1))
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()

