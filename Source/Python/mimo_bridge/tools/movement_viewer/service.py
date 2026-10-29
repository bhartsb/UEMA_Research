# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The viewer service: the one process that draws the movement pictures, keeps the index current and copies new records.

Started by the dashboard (development_dashboard.main and api.ensure_service); one at a time (an owner lock on
Saved/mimo_bridge/movement_viewer/service.lock: a second start exits at once). Single-threaded, so drawings never
overlap. Each loop:
  1. DRAWING REQUESTS from the dashboard's Render button (requests/<key>.json), oldest first, each after the memory
     guard allows it (mimo_bridge.tools.commit_guard --need-gb NEED_GB; a refusal leaves the request waiting and is
     shown on the panel with the guard's reading). Result: the pictures and manifest.json in pictures_dir()/<key>/, and
     requests/<key>.done.json or .error.json.
  2. Every BACKGROUND_S, after the memory guard allows it: the recordings scanned again (sources.py), each recording's
     index brought up to date (index.py), the raw chunks of production runs copied to the archive
     (tools/observer/archive.py), the automatic picks recomputed (auto.py), and up to BACKGROUND_DRAWINGS of the picks not
     drawn yet drawn in the smaller LITE picture set (render.py), newest first, stopping early for a drawing request,
     with the memory guard read again every GUARD_EVERY drawings. A pause (below) stops the pass between two
     recordings, two probe runs or two drawings.
  3. The heartbeat and the status (service_status.json) the panel reads.
After IDLE_RELEASE_S without a drawing the loaded model is freed (commands.release_models; 380 MB of the MjData arena).
Saved/mimo_bridge/movement_viewer/service_control.json {"paused": true} holds every drawing and background pass;
requests wait. So does a declared heavy job: while a runner holds the memory guard's heavy-job lock with a live process
(commit_guard.HEAVY_LOCK, written by the pause runners for measurements in a training pause), the service does nothing
but its heartbeat. The process runs below normal priority, opted out of Windows' power throttling.

NEED_GB: a drawing took its process from 760 MB to 1,481 MB (measured Oct 18, 2026, render.py's first drawing in a fresh
process: model, the 380 MB data arena, two renderers); rounded up to 1.0 GB.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
import traceback
from typing import Dict, List, Optional

from . import CONTROL_FILE, HOME, LOCK_FILE, REQUESTS_DIR, STATUS_FILE, part_path, pictures_dir, replace_file
from . import auto as A
from . import commands as C
from . import index as I
from . import render as R
from . import sources as S

NEED_GB = 1.0
BACKGROUND_S = 300
BACKGROUND_DRAWINGS = 40
GUARD_EVERY = 10                     # automatic drawings between two readings of the memory guard
IDLE_RELEASE_S = 600
POLL_S = 1.0
PYTHON = sys.executable


def _atomic(path, value) -> None:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    tmp = part_path(path)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f)
    replace_file(tmp, path)


def _read(path) -> Optional[dict]:
    try:
        with open(str(path), encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def guard(need_gb: float = NEED_GB) -> dict:
    """The memory guard's answer for one drawing: {allowed, reading}."""
    try:
        out = subprocess.run([PYTHON, "-B", "-m", "mimo_bridge.tools.commit_guard", "--need-gb", str(need_gb)],
                             capture_output=True, text=True, timeout=60, cwd=str(C_SOURCE),
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return dict(allowed=out.returncode == 0, reading=(out.stdout or out.stderr).strip()[:400], at=time.time())
    except Exception as ex:
        return dict(allowed=False, reading="the guard could not be run: %r" % (ex,), at=time.time())


C_SOURCE = os.path.join(str(S.REPO), "Source", "Python")


def _own_process_settings() -> dict:
    out = {}
    try:
        import psutil
        p = psutil.Process()
        if os.name == "nt":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(10)
        out["priority"] = "below normal"
    except Exception as ex:
        out["priority"] = "not set: %r" % (ex,)
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class _State(ctypes.Structure):
                _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
            state = _State(1, 0x1, 0)          # ProcessPowerThrottling: EXECUTION_SPEED off
            ok = bool(k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state)))
            out["power_throttling"] = "opted out" if ok else "opt-out failed (%d)" % ctypes.get_last_error()
        except Exception as ex:
            out["power_throttling"] = "opt-out failed: %r" % (ex,)
    return out


class Service:
    def __init__(self):
        self.started = time.time()
        self.last_background = 0.0
        self.last_drawing = 0.0
        self.status: Dict[str, object] = dict(pid=os.getpid(), started_at=self.started, errors=[])
        self.status.update(_own_process_settings())
        self.recordings: List[dict] = []

    # ------------------------------------------------------------------ drawing
    def pending_requests(self) -> List[dict]:
        out = []
        for p in glob.glob(str(REQUESTS_DIR / "*.json")):
            if p.endswith((".done.json", ".error.json", ".part")):
                continue
            r = _read(p)
            if r is not None:
                r["_path"] = p
                out.append(r)
        return sorted(out, key=lambda r: r.get("requested_at") or 0)

    def draw(self, rid: str, body: int, step: int, why: str, lite: bool = False) -> dict:
        rec = next((r for r in self.recordings if r["id"] == rid), None) or S.by_id(rid)
        man = R.render_command(rid, body, step, recording=rec, lite=lite)
        self.last_drawing = time.time()
        self.status["last_drawing"] = dict(key=man["key"], why=why, at=time.time(), reused=man.get("reused"),
                                           cost=man.get("cost"))
        return man

    def serve_requests(self) -> bool:
        """Draw every waiting request the guard allows. True when a request was handled."""
        reqs = self.pending_requests()
        self.status["queue"] = len(reqs)
        if not reqs:
            return False
        for n, r in enumerate(reqs):
            if n and self.paused():
                break
            g = guard()                        # one reading per drawing: memory can change between two of them
            self.status["guard"] = g
            if not g["allowed"]:
                return n > 0
            key = r.get("key")
            try:
                if r.get("probe"):
                    from . import probes as P
                    man = P.render_trial(r["probe"], int(r["body"]), str(r["goal"]), str(r["arm"]))
                    self.last_drawing = time.time()
                    self.status["last_drawing"] = dict(key=man["key"], why="requested probe trial", at=time.time(),
                                                       cost=man.get("cost"))
                else:
                    man = self.draw(r["recording"], int(r["body"]), int(r["step"]), "requested")
                _atomic(REQUESTS_DIR / ("%s.done.json" % key), dict(key=man["key"], done_at=time.time()))
            except Exception as ex:
                _atomic(REQUESTS_DIR / ("%s.error.json" % key), dict(key=key, error=str(ex)[:600], at=time.time()))
                self._error("drawing %s: %s" % (key, ex))
            try:
                os.remove(r["_path"])
            except OSError:
                pass
        return True

    # ------------------------------------------------------------------ background
    def background(self) -> None:
        t0 = time.time()
        found = S.scan()
        S.write(found)
        self.recordings = found
        updated, archived = [], []
        for rec in found:
            if self.paused():
                return                        # a declared heavy job or the pause file: stop between recordings
            res = I.update_recording(rec)
            changed = [x for x in res if x.get("done") not in ("unchanged",)]
            if changed:
                updated.append(dict(recording=rec["id"], label=rec["label"], bodies=changed))
            for copy in rec["copies"]:
                if "/Saved/mimo_bridge/experiments/" in copy.replace("\\", "/"):
                    from mimo_bridge.tools.observer.archive import Archive
                    rep = Archive.configured().run_pass(os.path.dirname(os.path.dirname(copy)))
                    if rep.get("copied") or rep.get("failed") or rep.get("error"):
                        archived.append(rep)
        picks = A.compute(found)
        A.write(picks)
        # the checkpoint probes' scores, cached so the page reads them at once (probes.py)
        from . import probes as P
        runs = P.scan()
        for run in runs:
            if self.paused():
                return                        # the pass stops for the pause; it runs again in full after it
            if run["complete"]:
                try:
                    P.shares_of_run(run, runs)
                except Exception as ex:
                    self._error("probe scores of %s: %s" % (run["label"], ex))
        self.status["background"] = dict(at=t0, seconds=round(time.time() - t0, 2), recordings=len(found),
                                         index_updates=updated[:20], archive=archived[:20],
                                         picks=len(picks["picks"]))
        self.last_background = time.time()

    def draw_picks(self) -> int:
        picks = A.load().get("picks") or []
        root = pictures_dir()
        waiting = [p for p in picks
                   if not os.path.isfile(os.path.join(str(root), R.key_for(p["recording"], p["body"], p["step"], True),
                                                      "manifest.json"))]
        self.status["picks_waiting"] = len(waiting)
        self.status["picks_total"] = len(picks)
        if not waiting:
            return 0
        g = guard()
        self.status["guard"] = g
        if not g["allowed"]:
            return 0
        done = 0
        for n, p in enumerate(waiting[:BACKGROUND_DRAWINGS]):
            if self.pending_requests():
                break                          # a person is waiting: their drawing comes first
            if self.paused():
                break
            if n and n % GUARD_EVERY == 0:
                g = guard()                    # the memory guard read again every GUARD_EVERY drawings
                self.status["guard"] = g
                if not g["allowed"]:
                    break
            try:
                self.draw(p["recording"], int(p["body"]), int(p["step"]), "automatic: " + "; ".join(p["reasons"]),
                          lite=True)
                done += 1
            except Exception as ex:
                self._error("automatic drawing %s body %s step %s: %s" % (p["recording"], p["body"], p["step"], ex))
                _mark_failed(p, str(ex))
        self.status["picks_waiting"] = len(waiting) - done
        return done

    # ------------------------------------------------------------------ loop
    def paused(self) -> bool:
        """True while the pause file says so or a declared heavy job runs (module doc); the reason is kept in the
        status."""
        c = _read(CONTROL_FILE) or {}
        if c.get("paused"):
            self.status["paused_by"] = "service_control.json"
            return True
        try:
            from mimo_bridge.tools.commit_guard import heavy_job_extra_gb
            _, holder = heavy_job_extra_gb()
        except Exception:
            holder = None
        if holder is not None:
            self.status["paused_by"] = "a declared heavy job holds the memory guard's heavy-job lock (%s)" % (
                str(holder.get("name") or holder.get("job") or holder.get("row") or "unnamed")[:120])
            return True
        self.status["paused_by"] = None
        return False

    def _error(self, text: str) -> None:
        errs = list(self.status.get("errors") or [])
        errs.append(dict(at=time.time(), text=text[:800]))
        self.status["errors"] = errs[-12:]

    def write_status(self) -> None:
        self.status["heartbeat"] = time.time()
        self.status["paused"] = self.paused()
        self.status["pictures_dir"] = str(pictures_dir())
        self.status["models_loaded"] = len(C._LOADED)
        try:
            import psutil
            mi = psutil.Process().memory_info()
            self.status["memory_mb"] = round(float(getattr(mi, "private", mi.rss)) / 2 ** 20, 1)
        except Exception:
            pass
        _atomic(STATUS_FILE, self.status)

    def run(self, once: bool = False) -> None:
        while True:
            try:
                if self.paused():
                    self.status["state"] = "paused: %s" % self.status.get("paused_by")
                else:
                    self.status["state"] = "running"
                    self.serve_requests()
                    if time.time() - self.last_background >= BACKGROUND_S:
                        g = guard()
                        self.status["guard"] = g
                        if g["allowed"]:
                            self.background()
                            self.write_status()
                            self.draw_picks()
                        else:
                            self.last_background = time.time()     # try again at the next pass
                            self.status["background_held"] = dict(at=time.time(), guard=g["reading"])
                    if C._LOADED and time.time() - self.last_drawing > IDLE_RELEASE_S:
                        C.release_models()
            except Exception:
                self._error(traceback.format_exc()[-1500:])
            try:
                self.write_status()
            except Exception:
                # a status that cannot be written this pass is written at the next one; the service keeps running
                self._error(traceback.format_exc()[-1500:])
            if once:
                return
            time.sleep(POLL_S)


def _mark_failed(p: dict, why: str) -> None:
    """A pick that cannot be drawn gets a manifest saying why, so it is not retried every pass."""
    out = os.path.join(str(pictures_dir()), R.key_for(p["recording"], p["body"], p["step"], True))
    os.makedirs(out, exist_ok=True)
    _atomic(os.path.join(out, "manifest.json"), dict(version=R.RENDER_VERSION, refused=why[:600],
                                                     recording=p["recording"], body=p["body"], step=p["step"],
                                                     at=time.time(), images={}))


def main(once: bool = False) -> int:
    os.makedirs(str(HOME), exist_ok=True)
    os.makedirs(str(REQUESTS_DIR), exist_ok=True)
    from mimo_bridge.tools.training_continuation import OwnerLock
    try:
        lock = OwnerLock(str(LOCK_FILE))
    except OSError:
        print("another viewer service holds %s; exiting" % LOCK_FILE, flush=True)
        return 0
    try:
        Service().run(once=once)
    finally:
        lock.close()
    return 0
