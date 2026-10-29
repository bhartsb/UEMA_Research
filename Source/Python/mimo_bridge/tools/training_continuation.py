# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Continue a training lineage without relying on an LLM notification.

Attach to one existing trainer; preserve its exact argument vector. Resume scheduled
exits and crashes from the newest complete checkpoint. Never override a repair stop,
another trainer, or supervisor_control.json {"cmd": "pause"}. No trainer is killed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def argument(command, flag):
    return command[command.index(flag) + 1]


def replace_argument(command, flag, value):
    command = list(command)
    if flag not in command:
        return command + [flag, str(value)]
    command[command.index(flag) + 1] = str(value)
    return command


def is_watcher_command(command):
    # Inspect the invoked script, not a --watcher argument of this supervisor.
    # Matching any argv element would classify the supervisor as its own child.
    script_index = 2 if len(command) > 1 and command[1] == "-u" else 1
    return len(command) > script_index and Path(command[script_index]).name == "hourly_watch.py"


def processes(kind, run=None):
    matches = []
    for p in psutil.process_iter(["pid", "ppid", "name", "cmdline", "create_time"]):
        try:
            c = p.info["cmdline"] or []
            if "python" not in (p.info["name"] or "").lower():
                continue
            if kind == "trainer":
                hit = "mimo_bridge.train_mimo" in c
                if hit:
                    from .native_proof_role import is_isolated
                    hit = not is_isolated(c)
            else:
                hit = is_watcher_command(c)
            if hit and (run is None or any(Path(x).resolve() == Path(run).resolve() for x in c[1:])):
                matches.append(p)
        except (OSError, psutil.Error):
            pass
    # Windows venv python is a launcher + child, not two trainers.
    ids = {p.pid for p in matches}
    return [p for p in matches if p.info["ppid"] not in ids]


class OwnerLock:
    def __init__(self, path):
        self.file = open(path, "a+b")
        self.file.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def close(self):
        self.file.close()


def complete_candidates(run):
    candidates = []
    folder=Path(run)/'checkpoints'
    args_path=Path(run)/'args.json'
    if args_path.exists():
        try:
            storage=json.loads(args_path.read_text(encoding='utf-8')).get('checkpoint_storage')
            if storage:folder=Path(storage)
        except (OSError,ValueError):return []
    for path in folder.glob("ckpt_step_*"):
        try:
            step = int(path.name.removeprefix("ckpt_step_"))
            meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
            required = ("model.safetensors", "optimizer.pt", "runtime_state.pt", "model_receipt.json")
            if meta["step"] == step and all((path / f).stat().st_size > 0 for f in required):
                receipt=json.loads((path/'model_receipt.json').read_text(encoding='utf-8'))
                if receipt.get('episode_continuation') and not (path/'episode_state.json').exists():
                    continue
                candidates.append((step, path))
        except (ValueError, KeyError, OSError):
            continue
    return [path for _,path in sorted(candidates,reverse=True)]


def latest_complete(run):
    return next(iter(complete_candidates(run)),None)


def write_stage_done(run,*,ok,step,resumed_from,started_at,exception):
    """Record the actual configured checkpoint storage, including another drive."""
    run=Path(run);saved=latest_complete(run);now=time.time()
    record=dict(run=run.name,ok=ok,steps=step,resumed_from=resumed_from,
        final_checkpoint=None if saved is None else str(saved),finished_at=now,
        finished_at_human=time.strftime('%Y-%m-%d %H:%M:%S'),elapsed_min=(now-started_at)/60,
        exception=exception)
    atomic_json(run/'stage_done.json',record)
    return record


def newer_checkpoint(run, last_resume):
    """Whether `run` saved a complete checkpoint past the one its trainer resumed from.

    A START THAT SAVED A NEWER CHECKPOINT DID NOT FAIL (2026-09-24 9:29 PM PDT). The failed-start
    count reset only after a trainer ran 10 minutes, so planned stops coming faster than that read
    as failed starts: relaunches at 8:34, 9:06 and 9:24 PM were each stopped within 5 minutes, with
    a new checkpoint, by the ability scheduler's comparison windows and coordinated evaluation
    pauses, and after the 9:28 PM stop the supervisor refused to resume ("three failed starts").
    Progress, not wall time, separates a start that failed from one that was stopped: a trainer
    that crashes early saves nothing new (checkpoints are hourly unless a save is asked for), and
    the same-checkpoint guard still catches a crash that repeats."""
    if not last_resume:
        return False
    try:
        base = int(Path(str(last_resume)).name.removeprefix("ckpt_step_"))
    except ValueError:
        return False
    for path in complete_candidates(run):
        try:
            if int(path.name.removeprefix("ckpt_step_")) > base:
                return True
        except ValueError:
            continue
    return False


def validate_checkpoint(path):
    # Check finite model tensors and optimizer presence before any relaunch.
    import torch
    from safetensors import safe_open
    with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as f:
        if not f.keys() or not all(torch.isfinite(f.get_tensor(k)).all() for k in f.keys()):
            raise ValueError("non-finite or empty model checkpoint")
    opt = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False)["optimizer"]
    if not opt.get("state") or not opt.get("param_groups"):
        raise ValueError("checkpoint lacks earned optimizer state")
    for state in opt["state"].values():
        for v in state.values():
            if torch.is_tensor(v) and not torch.isfinite(v).all():
                raise ValueError("checkpoint optimizer contains non-finite values")
    torch.load(path / "runtime_state.pt", map_location="cpu", weights_only=False)
    from mimo_bridge.continuation_state import load
    episode=load(path)
    if episode is not None and episode['trainer']['step']!=int(path.name.removeprefix('ckpt_step_')):
        raise ValueError('physical episode and model checkpoint steps differ')


def resolved(path):
    """A checkpoint path the trainer can open: the path itself when it exists, else where the
    move records say the archiver took it, proven by the files' hashes on reading (review
    ops-storage finding 11, 2026-09-23). train_mimo opens --resume with
    model.load_checkpoint BEFORE anything consults the move records, so a supervisor that relaunched
    with the old C: path of a moved checkpoint would fail; it now passes the D: path."""
    from mimo_bridge.storage_locations import locate
    return locate(Path(path), verify=False)


def resume_checkpoint(run, command):
    saved = latest_complete(run)
    if saved is None and "--resume" in command:
        # Includes a later crash before the first periodic snapshot, not just startup.
        try:
            saved = resolved(argument(command, "--resume"))
        except OSError:                  # a move row whose copy is missing: named, never used
            saved = Path(argument(command, "--resume"))
    return saved


def validated_resume_checkpoint(run,command):
    """A truncated newest snapshot cannot prevent using earlier intact learning."""
    candidates=complete_candidates(run)
    rejected=[]
    if '--resume' in command:
        raw=argument(command,'--resume')
        try:parent=resolved(raw)
        except OSError as error:
            # the move records name a copy that is missing or not the same: rejected with the
            # reason, never a crash of the supervisor's loop
            parent=None;rejected.append(dict(checkpoint=str(raw),error=str(error)))
        if parent is not None and parent not in candidates:candidates.append(parent)
    for candidate in candidates:
        try:validate_checkpoint(candidate)
        except Exception as error:
            rejected.append(dict(checkpoint=str(candidate),error=str(error)))
            continue
        return candidate,rejected
    return None,rejected


def report_state(path,state):
    try:atomic_json(path,state)
    except OSError as error:
        # Keeping the owner process alive matters when its own log drive fills.
        # Do not turn a reporting failure into hours without a supervisor.
        print(f'supervisor state write failed; continuing supervision: {error}',file=sys.stderr,flush=True)


def pause_reason(root, run):
    control = root / "supervisor_control.json"
    if control.exists():
        try:
            c = json.loads(control.read_text(encoding="utf-8-sig"))
            if not isinstance(c, dict) or ("cmd" not in c and "archiver" not in c):
                raise ValueError("no cmd")
            if c.get("cmd") == "pause":
                return "explicit supervisor pause"
            # AN ARCHIVER SETTING WITHOUT A cmd KEEPS TRAINING PAUSED (verify-fix-ops 2026-09-23).
            # The pause and the archiver switch share this file, so writing {"archiver": "off"} over
            # {"cmd": "pause"} would otherwise lift the pause: on 2026-09-23 training was paused on
            # purpose for the build, and a relaunch beside the build's tests is what exhausted the
            # commit limit on 2026-09-21. A file without cmd read as unreadable (paused) before the
            # switch existed; it still pauses, with its reason. Run with the archiver off by
            # writing both: {"cmd": "run", "archiver": "off"}.
            if "cmd" not in c:
                return ("supervisor_control.json carries an archiver setting and no cmd; training "
                        "stays paused until a cmd is written ({\"cmd\": \"run\", \"archiver\": "
                        "\"off\"} runs it with the archiver off)")
            if c.get("cmd") != "run":
                return "unknown supervisor command; explicit run or pause required"
        except (OSError, KeyError, ValueError):
            return "unreadable supervisor control"
    reason = Path(run) / "stop_reason.json"
    if reason.exists():
        try:
            r = json.loads(reason.read_text(encoding="utf-8"))
            if not isinstance(r, dict):
                return "unreadable watcher stop reason"
            if r.get("resume_policy") == "repair_required":
                explanation = r.get("reason")
                return (explanation if isinstance(explanation, str) and explanation.strip()
                        else "watcher requires repair")
        except (OSError, ValueError):
            return "unreadable watcher stop reason"
    # The trainer retains consumed controls, so a deliberate stop is seen even
    # if it was consumed between supervisor polls. Scheduled exits have no such command.
    for consumed in Path(run).glob("control_done_*.json"):
        try:
            r = json.loads(consumed.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, ValueError):
            return "unreadable consumed control; inspect before resuming"
        if not isinstance(r, dict):
            return "unreadable consumed control; inspect before resuming"
        if r.get("cmd") in ("stop", "snapshot_and_stop") and r.get("auto_resume") is not True:
            return "deliberate stop recorded; explicit continuation required"
    return None


def repair_run_pause(root, run):
    """True when supervisor_control.json is the pause the relaunch script writes for a REPAIR RUN in `run` (the
    training preflight's repair run, the operator's rule of Oct 5, 2026): {"cmd": "pause", "repair_run": <run folder>}.
    pause_reason reads it as a pause, so nothing is relaunched when the trainer ends at its bound or crashes: the run
    cannot continue into normal training by itself. The run's hourly watcher still runs while its trainer lives (it
    judges the run's stop lines in <run>/declared_criteria.json and the standing stop rules); a run whose watcher
    already wrote its stop (stop_reason.json) gets no new one."""
    try:
        c = json.loads((Path(root) / "supervisor_control.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    if not isinstance(c, dict) or c.get("cmd") != "pause" or not c.get("repair_run"):
        return False
    try:
        same = os.path.normcase(str(Path(str(c["repair_run"])).resolve())) == os.path.normcase(str(Path(run).resolve()))
    except (OSError, ValueError):
        return False
    return same and not (Path(run) / "stop_reason.json").exists()


def launch(command, cwd, log):
    env = dict(os.environ, PYTHONPATH=str(cwd))
    with open(log, "ab", buffering=0) as stream:
        return subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


def resume_launch_refusal(command):
    """An automatic resume obeys the same source/configuration admission as a manual launch.

    Keeping argument strings does not keep code unchanged. Reuse the recorded
    gates; this performs no new physical experiment and changes no authorization.
    """
    try:
        from .training_preflight import launch_refusal
        return launch_refusal(command)
    except Exception as error:
        return "could not verify continuation preflight: %s" % error


# THE CHECKPOINT ARCHIVER. Each hourly save costs about 1.2 GB on C: and nothing but
# tools/archive_checkpoints.py moves older ones off (C: stopped H84 on 2026-09-21 at 3.1 GB free
# before it existed). Measured 2026-09-22 (storage survey): with the archiver running C: grows
# 38 MB an hour, without it 1.46-1.49 GB an hour, 2.3 days of runway. So the supervisor keeps
# exactly one running: it starts one only when no archiver process exists AND nobody holds the
# archiver's lock, and it never stops or replaces one that runs. When none runs after a start,
# the next start waits ARCHIVER_RETRY_S, so a crashing archiver cannot become a launch loop.
ARCHIVER_COMMAND = ("-u", "-m", "mimo_bridge.tools.archive_checkpoints", "--all", "--watch",
                    "--interval", "900")
ARCHIVER_RETRY_S = 600.0
# THE ARCHIVER FOLLOWS THE RUN'S LINEAGE, AND CAN BE SWITCHED OFF (review ops-storage finding 9,
# 2026-09-23). The command carried no --lineage, so every supervisor started H84's archiver
# (archive_checkpoints' default) whatever run it held: a supervisor on another lineage would have
# left that lineage's checkpoints on C: at about 1.2 GB an hour behind a fresh heartbeat. And it
# ran on every sweep, paused or not, so an archiver stopped by hand while D: was being replaced
# came back within 600 s. Now the lineage is the resume chain's first run (the folder name the
# lineage's other runs start with, as archive_checkpoints.inventory globs it), and
# {"archiver": "off"} in supervisor_control.json stops the supervisor from starting one (it never
# stops a running archiver: that stays a person's act). Write it beside the cmd: {"cmd": "pause",
# "archiver": "off"} or {"cmd": "run", "archiver": "off"}; alone it keeps training paused
# (pause_reason), so writing it can never lift a pause.
ARCHIVER_OFF = "off"


def lineage_of(run):
    """The lineage a run belongs to: the name of the first run of its resume chain (every later
    segment's folder starts with it). None when the run's args.json cannot be read."""
    try:
        from mimo_bridge.tools.offload_storage import resume_chain
        chain = resume_chain(Path(run))
    except Exception:                   # never a reason to keep no archiver: the default lineage
        return None
    if not chain or not (Path(chain[-1]) / "args.json").exists():
        return None
    return Path(chain[-1]).name


def archiver_command(lineage=None):
    return list(ARCHIVER_COMMAND) + (["--lineage", str(lineage)] if lineage else [])


def archiver_switch(root):
    """'off' when supervisor_control.json carries {"archiver": "off"}, else 'on'."""
    try:
        c = json.loads((Path(root) / "supervisor_control.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "on"
    return ARCHIVER_OFF if isinstance(c, dict) and c.get("archiver") == ARCHIVER_OFF else "on"


def archiver_pids():
    from mimo_bridge.tools.archive_checkpoints import archiver_pids as scan
    return scan()


def _running_archiver_lineage(root):
    """The lineage the running archiver last reported in its status file (None if unknown)."""
    from mimo_bridge.tools.archive_checkpoints import STATUS_NAME
    try:
        return json.loads((Path(root) / STATUS_NAME).read_text(encoding="utf-8")).get("lineage")
    except (OSError, ValueError, AttributeError):
        return None


def ensure_archiver(root, cwd, last_start=None, now=None, lineage=None):
    """Exactly one checkpoint archiver, for `lineage`. Returns a status dict; 'started_at' is set
    when this call launched one (the caller passes it back as last_start)."""
    from mimo_bridge.tools.archive_checkpoints import LOCK_NAME
    now = time.time() if now is None else now
    pids = archiver_pids()
    if pids:
        out = {"status": "running", "pids": pids, "lineage": lineage}
        reported = _running_archiver_lineage(root)
        if lineage and reported and reported != lineage:
            out["status"] = ("running, but for lineage %s while this run's lineage is %s; the "
                             "supervisor never stops a running archiver - stop it and the next "
                             "sweep starts one for %s" % (reported, lineage, lineage))
        return out
    if archiver_switch(root) == ARCHIVER_OFF:
        return {"status": "off: supervisor_control.json carries {\"archiver\": \"off\"}; none "
                          "started", "lineage": lineage}
    try:
        OwnerLock(Path(root) / LOCK_NAME).close()
    except OSError:
        return {"status": "no archiver process found, but its lock is held; not starting a second"}
    if last_start is not None and now - last_start < ARCHIVER_RETRY_S:
        return {"status": "started %.0f s ago and not running; next start allowed after %.0f s"
                          % (now - last_start, ARCHIVER_RETRY_S)}
    launch([sys.executable, *archiver_command(lineage)], cwd,
           Path(root) / "checkpoint_archive_stdout.log")
    return {"status": "started (none was running)", "started_at": now, "lineage": lineage}


def stop_watchers(run):
    watchers = processes("watcher", run)
    for p in watchers:
        try:
            for child in p.children(recursive=True):
                child.terminate()
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(watchers, timeout=5)


def ensure_ability_rechecks(root, cwd, last_start=None, now=None):
    """Keep one lightweight comparison scheduler alive; it honors training pauses."""
    now=time.time() if now is None else now
    repo=Path(cwd).resolve().parents[1]
    manifest=repo/'docs/gain_of_function_v3/ability_rechecks.json'
    if not manifest.is_file():return {'status':'no declared ability tests'}
    module='mimo_bridge.tools.ability_rechecks'
    pids=[]
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            command=process.info['cmdline'] or []
            if (module in command and '--root' in command
                    and Path(argument(command,'--root')).resolve()==repo):pids.append(process.pid)
        except (psutil.Error,OSError,ValueError):continue
    if pids:return {'status':'running','pids':pids}
    folder=Path(root)/'ability_rechecks'
    folder.mkdir(parents=True,exist_ok=True)
    try:OwnerLock(folder/'owner.lock').close()
    except OSError:return {'status':'existing scheduler holds its lock'}
    if last_start is not None and now-last_start<600:
        return {'status':'restart delayed after scheduler exit'}
    process=launch([sys.executable,'-u','-m',module,'--root',str(repo),
                    '--manifest',str(manifest)],cwd,folder/'worker.log')
    return {'status':'started','pid':process.pid,'started_at':now}


OBSERVER_STALL_S = 1800.0


def ensure_observer(root, cwd, last_start=None, now=None):
    """Keep one developmental-observer digester alive (mimo_bridge/tools/observer, addendum v4
    Part 6). It reads the active run's observer recordings in its own process and never touches
    training; if none ran, the recorders would fill their disk budget and then drop steps."""
    now = time.time() if now is None else now
    module = 'mimo_bridge.tools.observer'
    pids = []
    for process in psutil.process_iter(['pid', 'cmdline', 'create_time']):
        try:
            command = process.info['cmdline'] or []
            if module in command and 'watch' in command:
                pids.append(process)
        except (psutil.Error, OSError):
            continue
    folder = Path(root) / 'observer'
    if pids:
        # a digester that stopped reporting for OBSERVER_STALL_S is stuck (a detector loop that
        # never ends looks alive to a process check): it is stopped, recorded, and restarted below
        reported = None
        for name in ('daemon_status.json', 'daemon_heartbeat.json'):   # a pass's end, a window's end
            try:
                t = json.loads((folder / name).read_text(encoding='utf-8')).get('time')
                reported = max(float(t), reported or 0.0) if t is not None else reported
            except (OSError, ValueError, AttributeError, TypeError):
                continue
        started = min(p.info.get('create_time') or now for p in pids) if all(
            hasattr(p, 'info') for p in pids) else now
        last = max(float(reported or 0), float(started or 0))
        if now - last < OBSERVER_STALL_S:
            return {'status': 'running', 'pids': [p.pid for p in pids]}
        for p in pids:
            try:
                p.kill()
            except (psutil.Error, OSError):
                pass
        try:
            atomic_json(folder / 'stalled_digester.json',
                        {'stopped_pids': [p.pid for p in pids], 'last_report': reported, 'at': now})
        except OSError:
            pass
        return {'status': 'stalled for %.0f s; stopped, the next sweep restarts it' % (now - last)}
    folder.mkdir(parents=True, exist_ok=True)
    try:
        OwnerLock(folder / 'owner.lock').close()
    except OSError:
        return {'status': 'existing digester holds its lock'}
    if last_start is not None and now - last_start < 600:
        return {'status': 'restart delayed after digester exit'}
    process = launch([sys.executable, '-u', '-m', module, 'watch'], cwd, folder / 'worker.log')
    return {'status': 'started', 'pid': process.pid, 'started_at': now}


def supervise(run, root, watcher, replace_watcher=False, poll=15):
    root.mkdir(parents=True, exist_ok=True)
    lock = OwnerLock(root / "training_continuation.lock")
    state_file = root / "training_continuation.json"
    repo = Path(__file__).resolve().parents[4]
    cwd = repo / "Source" / "Python"
    running = processes("trainer")
    matching = processes("trainer", run)
    if len(running) != 1 or len(matching) != 1:
        raise RuntimeError("Attach requires exactly one live trainer for the supplied run")
    command = list(matching[0].info["cmdline"])
    # Use the venv path rather than the child interpreter's base executable.
    command[0] = sys.executable
    if replace_watcher:
        stop_watchers(run)
    attempts = 0
    restarted_at = 0.0
    # The SAME resume point twice means the last attempt died without reaching a new
    # checkpoint: the crash is in the weights or the code, not in the weather. Measured
    # 2026-09-20: H84 crashed on a runaway self-percept and this supervisor resumed it 18
    # times from ckpt_step_231473, every attempt living about 20 minutes - long enough to
    # clear the 10-minute healthy-start reset below - for 418 steps in 6.4 hours.
    last_resume = None
    repeat_resumes = 0
    origin = run.name
    state = {"supervisor_pid": os.getpid(), "origin": str(run), "active_run": str(run),
             "command": command, "started": time.time()}
    archiver_started = None
    ability_started = None
    observer_started = None
    lineage_for = {}
    try:
        while True:
            now = time.time()
            # attached: keep exactly one checkpoint archiver (every sweep, so one that dies is
            # replaced; never a second)
            try:
                if lineage_for.get("run") != str(run):      # walked once per held run
                    lineage_for.update(run=str(run), lineage=lineage_of(run))
                state["archiver"] = ensure_archiver(root, cwd, archiver_started, now,
                                                    lineage=lineage_for["lineage"])
                archiver_started = state["archiver"].get("started_at", archiver_started)
            except Exception as error:          # the archiver must never cost the supervisor
                state["archiver"] = {"status": "check failed: %s" % error}
            try:
                state['ability_rechecks']=ensure_ability_rechecks(root,cwd,ability_started,now)
                ability_started=state['ability_rechecks'].get('started_at',ability_started)
            except Exception as error:
                state['ability_rechecks']={'status':f'check failed: {error}'}
            try:
                state['observer'] = ensure_observer(root, cwd, observer_started, now)
                observer_started = state['observer'].get('started_at', observer_started)
            except Exception as error:          # the observer must never cost the supervisor
                state['observer'] = {'status': f'check failed: {error}'}
            reason = pause_reason(root, run)
            running = processes("trainer")
            matching = processes("trainer", run)
            state.update(timestamp=now, active_run=str(run), command=command,
                         trainer_pids=[p.pid for p in matching], pause_reason=reason)
            if matching:
                if len(running) != 1:
                    state["status"] = "multiple trainers detected; no launch permitted"
                else:
                    watches = processes("watcher", run)
                    if not watches and (reason is None or repair_run_pause(root, run)):
                        launch([sys.executable, "-u", str(watcher), str(run), "8760"],
                               cwd, run / "hourly_watch.log")
                    stats = run / "replay_training_stats.jsonl"
                    age = now - stats.stat().st_mtime if stats.exists() else None
                    state["stats_age_seconds"] = age
                    state["status"] = "training" if age is not None and age < 600 else "training stats stale; inspect trainer"
                    if restarted_at and now - restarted_at > 600 and age is not None and age < 600:
                        attempts = 0
            elif reason:
                state["status"] = "paused for repair or explicit instruction"
            elif running:
                state["status"] = "another trainer owns execution; no relaunch"
            elif attempts >= 3 and not newer_checkpoint(run, last_resume):
                state["status"] = "paused after three failed starts; repair required"
            elif repeat_resumes >= 1:
                state["status"] = ("paused: two attempts resumed from the same checkpoint "
                                   f"({Path(last_resume).name}), so the last one reached no new "
                                   "checkpoint; this crash repeats and requires repair")
            elif now - restarted_at >= 60:
                checkpoint,rejected = validated_resume_checkpoint(run, command)
                state['rejected_checkpoints']=rejected
                if checkpoint is None:
                    state["status"] = "no complete checkpoint; cannot resume without losing evidence"
                elif str(checkpoint) == last_resume and (repeat_resumes := repeat_resumes + 1):
                    # counted here, judged at the top of the next sweep, so the reason is
                    # reported before anything is launched again
                    state["status"] = ("paused: this checkpoint was already resumed from once "
                                       "and the attempt reached no new one; repair required")
                else:
                    new_run = run.parent / f"{origin}_resume_{time.strftime('%Y%m%d_%H%M%S')}"
                    next_command = replace_argument(replace_argument(command, "--resume", checkpoint), "--out", new_run)
                    if '--checkpoint-storage' in command and argument(command,'--checkpoint-storage'):
                        storage=Path(argument(command,'--checkpoint-storage'))
                        # A new segment owns its own checkpoint ancestry/args.
                        next_storage=storage.parent.with_name(new_run.name)/storage.name
                        next_command=replace_argument(next_command,'--checkpoint-storage',next_storage)
                    refused = resume_launch_refusal(next_command)
                    if refused:
                        state["status"] = "continuation preflight refused"
                        state["preflight_refusal"] = refused
                        report_state(state_file, state)
                        time.sleep(poll)
                        continue
                    state.pop("preflight_refusal", None)
                    try:new_run.mkdir(exist_ok=False)
                    except OSError as error:
                        state['status']=f'cannot create continuation run: {error}'
                        report_state(state_file,state);time.sleep(poll);continue
                    try:
                        atomic_json(new_run / "continuation_receipt.json", {
                            "parent_run": str(run), "checkpoint": str(checkpoint), "command": next_command,
                            "timestamp": now, "reason": "previous trainer exited; preserving checkpoint and exact flags"})
                    except OSError as error:
                        state['status']=f'cannot record continuation before launch: {error}'
                        report_state(state_file,state);time.sleep(poll);continue
                    # Recheck immediately before launch; another operator may have resumed meanwhile.
                    if not processes("trainer") and not pause_reason(root, run):
                        if newer_checkpoint(run, last_resume):
                            attempts = 0            # the last start made progress (newer_checkpoint)
                        stop_watchers(run)
                        launch(next_command, cwd, new_run / "trainer.log")
                        run, command = new_run, next_command
                        last_resume = str(checkpoint)
                        attempts += 1
                        restarted_at = now
                        state["status"] = "resumed from complete checkpoint"
                        state["active_run"] = str(run)
            report_state(state_file, state)
            time.sleep(poll)
    finally:
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--watcher", type=Path, required=True)
    p.add_argument("--replace-watcher", action="store_true")
    a = p.parse_args()
    supervise(a.run.resolve(), a.root.resolve(), a.watcher.resolve(), a.replace_watcher)


if __name__ == "__main__":
    main()
