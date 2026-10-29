# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Move heavy folders from C: to D: and back, reversibly, with a record every reader can follow.

    python -m mimo_bridge.tools.offload_storage plan [--out plan.json]
    python -m mimo_bridge.tools.offload_storage move <repo-relative folder> [...]
    python -m mimo_bridge.tools.offload_storage restore <repo-relative folder> [...]

WHY. D16 of docs/gain_of_function_v3/DESIGN.md: C: holds an 89-day runway with the checkpoint
archiver running, and D: is the binding disk. This tool moves nothing on its own: `plan` only lists
what COULD move, with sizes and the reason each other folder must stay. Which folders move, and
when, is decided by a person. What it guarantees is that a move is knowable and reversible:

  - the layout is prefix-preserving: <repo>/<path> goes to D:/UEMA_Research/external/D/UEMA_recordings/C_offload/<path>;
  - every file is hashed (SHA-256) on C:, copied, and hashed again on D:; any difference removes
    the copy and leaves C: untouched;
  - rows `planned` (before the copy), `moved` (after the verified copy) and `restored` go to
    Saved/mimo_bridge/storage_moves.jsonl on C: and to D:/UEMA_Research/external/D/UEMA_recordings/C_offload/
    storage_moves.jsonl; mimo_bridge.storage_locations reads them, so a reader that asks for the
    old path through locate() is given the D: copy, proven by hash;
  - a pointer <folder>.moved.json is left where the folder was, for a person looking at C:;
  - `restore` is the same procedure in reverse.

WHAT NEVER MOVES (each unit that stays carries its reasons in the plan):
  - a run folder itself, or any folder holding a run's args.json or replay_training_stats.jsonl:
    the dashboards and the lifetime sweep find runs by those files (lifetime.py sweeps every
    experiments/*/replay_training_stats.jsonl; dashboard.py lists every experiments/* with an
    args.json; development_dashboard.py walks ancestry through each run's args.json). Only a
    run's SUBFOLDERS are units;
  - the active lineage's checkpoints and caregiver records: the checkpoint archiver moves those,
    and never its newest 3 checkpoints or the supervisor's resume source; a second mover on the
    same files would race it;
  - anything modified in the last 6 hours;
  - anything a running process reads: a process whose command line names a path inside the
    folder (or the folder lies inside a path it names - build_explainer --run <run>), or a
    process holding a file inside it open. Measured 2026-09-22: the open-file scan takes 16 s
    and cannot read 175 of 372 processes (other users' and system services); the plan says how
    many it could not read;
  - the videos of the first three runs of the active run's resume chain: the development
    dashboard shows videos from those (development_dashboard.videos_for), and from every experiment
    run copied into experiment_mirror (a LIVE_SYSTEM folder, from Oct 13, 2026);
  - the active lineage's scenes folders: a resumed world writes its scene files into its saved
    run_dir, the lineage's first run's scenes folder;
  - every subfolder of the active run itself (from 2026-09-23): the development dashboard reads its
    live frames and the explainer rebuilds from its snapshots, and after a quiet night nothing else
    keeps them;
  - every path the active run's resume chain names in its args.json files, and every path the
    saved worlds' env settings name (from 2026-09-23, review ops-storage finding 3). Measured
    2026-09-22: the plan listed Saved/mimo_bridge/caregiver_speech_exchange_v1_final as movable;
    segment 31's args.json names it as caregiver_speech_dir, every saved H84 world's env settings
    carry it, and speech.SpeechBank opens it by path at every launch, so the deploy's resume would
    have stopped with FileNotFoundError. The saved worlds' settings are built by
    train_mimo.env_kwargs_for from the args of the run that started the episode (the chain's
    root); their path-valued settings are that run's caregiver_speech_dir, caregiver_replay and
    caregiver_record_dir and <out>/scenes, later changed only by a resume that extends the speech
    library (continuation_state.extend_caregiver_speech, to a later run's caregiver_speech_dir).
    So the chain's args name every one of them; `plan --companion` also reads the resume
    checkpoint's episode companion and adds its worlds' settings exactly (it loads about 600 MB);
  - the live system's own folders (LIVE_SYSTEM): gof_probes (its index.jsonl is the master record
    of "What was measured, as numbers" in docs/source_of_truth.json, read by the dashboards' ten-
    item panel and the probe watcher), lineage_series (the development dashboard's cache),
    dashboard_history (the dashboards' history), caregiver_speech (the trainer's default
    --caregiver-speech-dir, opened by path by tools and tests) and any *.lock folder (held by a
    running program; developmental_queue.lock was a candidate on 2026-09-22);
  - any folder holding files the repository tracks (git ls-files; from 2026-09-23): code and tests
    read them by path, and a move reads to git as deleted files.
A probe or proof run's checkpoints (archive_checkpoints.NOT_LINEAGE) are NOT kept "for the
archiver": the archiver leaves them out of the lineage, so keeping them here too would leave a
1.2 GB end checkpoint that neither tool could ever move (review ops-storage finding 10).

AN INTERRUPTED MOVE OR RESTORE IS FINISHED, NEVER LEFT STUCK (finding 12). `move` writes its
`moved` row before it takes the C: folder away, and `restore` renames the copy into place before it
writes `restored`. A process that dies between the two leaves both copies whole and the latest row
`moved`. Both commands prove that state (C: and D: each hash to the record) and finish what was
asked: `move` takes the C: folder away and leaves the pointer, `restore` writes `restored` and
removes the D: copy.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from mimo_bridge import storage_locations as SL

REPO = SL.REPO
SAVED = SL.SAVED
OFFLOAD_ROOT = SL.OFFLOAD_ROOT
MIN_AGE_HOURS = 6.0
POINTER_SUFFIX = ".moved.json"
RUN_RECORDS = ("args.json", "replay_training_stats.jsonl")
# top-level Saved/mimo_bridge folders that are not units themselves
NOT_UNITS = {"experiments", "caregiver_records"}
# the live system's own folders under Saved/mimo_bridge, each with who reads it (see the docstring)
LIVE_SYSTEM = {
    "gof_probes": "the probe index (index.jsonl), the master record of what was measured "
                  "(docs/source_of_truth.json); the dashboards and the probe watcher read it",
    "lineage_series": "the development dashboard's lineage cache",
    "dashboard_history": "the dashboards' history",
    # found by the 2026-09-23 dry plan (candidate, nothing names it in the H84 chain's args): the
    # trainer's default --caregiver-speech-dir, read by path by developmental_trial,
    # developmental_confirm, verify_continuous_world and three test files
    "caregiver_speech": "train_mimo's default --caregiver-speech-dir; tools and tests open it by path",
    # Oct 13, 2026: experiment runs outside the project, copied for the dashboard (tools/experiment_mirror.py); moved
    # to D:, their clips would leave the recordings list, which serves only files inside the project
    "experiment_mirror": "the development dashboard's recordings list and experiment view read every experiment "
                         "run's copies here (development_dashboard.videos_for, experiment_mirror.live_view)",
}
LOCK_SUFFIX = ".lock"
# args.json keys that name the run folder itself: its subfolders are the units, and each is judged
# by its own rules (a key naming the whole run would keep every one of them)
RUN_SELF_KEYS = frozenset({"out"})
# where the trainer runs (the supervisor launches it with this working folder), so a relative path
# in its args.json is read from here
TRAINER_CWD = REPO / "Source" / "Python"


# ------------------------------------------------------------------ what exists on C:
def _walk(folder: Path):
    """(bytes, newest mtime, file count, names of run-record files found) for a folder tree."""
    total, newest, count, records = 0, os.stat(folder).st_mtime, 0, set()
    for root, dirs, files in os.walk(folder):
        try:
            newest = max(newest, os.stat(root).st_mtime)
        except OSError:
            pass
        for f in files:
            try:
                st = os.stat(os.path.join(root, f))
            except OSError:
                continue
            total += st.st_size
            newest = max(newest, st.st_mtime)
            count += 1
            if f in RUN_RECORDS:
                records.add(f)
    return total, newest, count, records


def _is_run(folder: Path) -> bool:
    return any((folder / r).is_file() for r in RUN_RECORDS)


def units(saved: Path = SAVED):
    """Every folder the plan considers, as (path, category)."""
    out = []
    exp = saved / "experiments"
    if exp.is_dir():
        for run in sorted(exp.iterdir()):
            if run.is_dir():
                out += [(s, "run subfolder") for s in sorted(run.iterdir()) if s.is_dir()]
    cr = saved / "caregiver_records"
    if cr.is_dir():
        out += [(f, "caregiver records") for f in sorted(cr.iterdir()) if f.is_dir()]
    for top in sorted(saved.iterdir()) if saved.is_dir() else []:
        if not top.is_dir() or top.name in NOT_UNITS:
            continue
        if _is_run(top):                      # a run outside experiments/: its subfolders only
            out += [(s, "run subfolder") for s in sorted(top.iterdir()) if s.is_dir()]
        else:
            out.append((top, "study folder"))
    return out


# ------------------------------------------------------------------ what must stay
def resume_chain(active_run, limit=500):
    """The active run, then the run each one resumed from (args.json "resume" -> the
    checkpoint's run folder, or the run its lineage.json names when the checkpoint moved)."""
    chain, run = [], Path(active_run) if active_run else None
    for _ in range(limit):
        if run is None or run in chain:
            break
        chain.append(run)
        try:
            resume = json.loads((run / "args.json").read_text(encoding="utf-8")).get("resume")
        except (OSError, ValueError):
            break
        if not resume:
            break
        ck = Path(resume)
        parent = ck.parent.parent if ck.parent.name == "checkpoints" else None
        if parent is None or not (parent / "args.json").exists():
            try:
                parent = SL.resolve_run_args(ck).parent
            except (OSError, ValueError):
                parent = parent if parent is not None and parent.exists() else None
        run = parent
    return chain


def process_paths():
    """dict(pid, name, named: paths on the command line, open: open files, readable) for every
    process. A relative argument is taken against the process's own working folder."""
    import psutil
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = p.info["cmdline"] or []
        except (psutil.Error, OSError):
            cmd = []
        try:
            cwd = p.cwd()
        except (psutil.Error, OSError):
            cwd = None
        named = []
        for arg in cmd[1:]:
            piece = arg.split("=", 1)[-1]
            if not ("/" in piece or "\\" in piece):
                continue                      # a flag, a number or a module name
            if not os.path.isabs(piece):
                if cwd is None:
                    continue
                piece = os.path.join(cwd, piece)
            named.append(piece)
        try:
            files = [f.path for f in p.open_files()]
            readable = True
        except (psutil.Error, OSError):
            files, readable = [], False
        out.append(dict(pid=p.info["pid"], name=p.info["name"], named=named, open=files,
                        readable=readable))
    return out


def _inside(child, parent) -> bool:
    c, p = SL._key(child), SL._key(parent)
    return c == p or c.startswith(p.rstrip(os.sep) + os.sep)


def _covers(named, unit: Path) -> bool:
    """Whether a path a process names means the process may read inside this folder: the path
    lies inside it, or it is a run folder (or a folder of a run) holding it - build_explainer
    --run <run> reads the run's explain/ folder. A broad root does not count: the supervisor
    names --root Saved/mimo_bridge and reads two files there, not every study folder."""
    if _inside(named, unit):
        return True
    # units are a run's direct subfolders, so a named folder that holds one is the run itself
    return _inside(unit, named) and _is_run(Path(named))


def _path_like(value) -> bool:
    return isinstance(value, str) and ("/" in value or "\\" in value)


def _absolute(value, cwd: Path = None) -> str:
    return value if os.path.isabs(value) else os.path.join(str(cwd or TRAINER_CWD), value)


def args_paths(args: dict, cwd: Path = None):
    """[(key, absolute path)] for every path-valued setting of one run's args.json, lists and
    nested settings included, except the keys that name the run folder itself (RUN_SELF_KEYS).
    The scenes folder the run's worlds write to (env_kwargs_for: run_dir = <out>/scenes) is
    added as 'run_dir'."""
    out = []

    def walk(value, key):
        if _path_like(value):
            out.append((key, _absolute(value, cwd)))
        elif isinstance(value, dict):
            for k, v in value.items():
                walk(v, "%s.%s" % (key, k) if key else str(k))
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                walk(v, "%s[%d]" % (key, i))
    for k, v in (args or {}).items():
        if k not in RUN_SELF_KEYS:
            walk(v, k)
    if _path_like(args.get("out")):
        out.append(("run_dir", os.path.join(_absolute(args["out"], cwd), "scenes")))
    return out


def chain_named_paths(chain, cwd: Path = None):
    """[(path, what names it)] over every run of the resume chain (the active run first): what
    each run's args.json names, which covers the saved worlds' env settings (module docstring)."""
    named = []
    for run in chain:
        try:
            args = json.loads((Path(run) / "args.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        named += [(p, "%s's args.json (%s)" % (Path(run).name, k)) for k, p in args_paths(args, cwd)]
    return named


def companion_world_paths(checkpoint):
    """[(path, what names it)] from the saved worlds' env settings in a checkpoint's episode
    companion (continuation_state.load: reads the whole companion, about 600 MB at H84 size)."""
    from mimo_bridge import continuation_state as CS
    state = CS.load(str(checkpoint))
    out = []
    for i, world in enumerate((state or {}).get("worlds") or []):
        for k, v in (world.get("env_kwargs") or {}).items():
            if _path_like(v):
                out.append((_absolute(v), "saved world %d's env setting %s" % (i, k)))
    return out


def tracked_files(repo: Path):
    """Repository-relative paths ('/'-separated) of every file git tracks under Saved/; empty when
    git or the repository cannot be read (the plan then says so)."""
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(repo), "ls-files", "-z", "--", "Saved"],
                             capture_output=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return [p for p in out.decode("utf-8", "replace").split("\0") if p]


def context(saved: Path = SAVED, lineage: str = None, procs=None, now: float = None,
            companion: bool = False, tracked=False):
    """Everything the exclusion rules read, gathered once. companion: also read the resume
    checkpoint's episode companion for its worlds' env settings (companion_world_paths)."""
    from mimo_bridge.tools import archive_checkpoints as AC
    now = time.time() if now is None else now
    try:
        continuation = json.loads((saved / "training_continuation.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        continuation = {}
    chain = resume_chain(continuation.get("active_run"))
    if lineage is None:
        lineage = chain[-1].name if chain else None
    protected = {}
    if lineage:
        protected = AC.protected(AC.inventory(lineage, saved), continuation,
                                 keep_list=AC.read_keep_list(saved))
    named = chain_named_paths(chain)
    if companion:
        cmd = continuation.get("command") or []
        if "--resume" in cmd:
            named += companion_world_paths(cmd[cmd.index("--resume") + 1])
    # FILES THE REPOSITORY TRACKS (2026-09-23): a folder holding them is code's and tests' by path
    # (the speech libraries, scale_anchors' neighbours); moving it would also read to git as 1,000+
    # deleted files, as the lane worktree's accident of 00:31 that day showed. tracked=False reads
    # git; tests pass a list (or None = unreadable)
    if tracked is False:
        tracked = tracked_files(Path(saved).parents[1])
    tracked_dirs = None
    if tracked is not None:
        tracked_dirs = {}
        for rel in tracked:
            d = os.path.dirname(os.path.join(str(Path(saved).parents[1]), rel))
            while True:                       # every folder that holds the file counts it once
                k = SL._key(d)
                tracked_dirs[k] = tracked_dirs.get(k, 0) + 1
                parent = os.path.dirname(d)
                if parent == d or len(parent) < 4:
                    break
                d = parent
    return dict(now=now, lineage=lineage, chain=chain, dashboard_runs=chain[:3],
                protected=protected, continuation=continuation, named=named,
                tracked_dirs=tracked_dirs,
                procs=process_paths() if procs is None else procs)


def reasons_to_stay(unit: Path, category: str, walk, ctx):
    """Every reason this folder must stay on C: (empty = it may move)."""
    total, newest, count, records = walk
    why = []
    run = unit.parent
    if records:
        why.append("holds a run's %s (the dashboards and the lifetime sweep find runs by it)"
                   % " and ".join(sorted(records)))
    lineage = ctx["lineage"]
    from mimo_bridge.tools.archive_checkpoints import NOT_LINEAGE
    if lineage and (category == "run subfolder" and run.name.startswith(lineage)
                    and unit.name == "checkpoints" and not NOT_LINEAGE.search(run.name)):
        why.append("the active lineage's checkpoints: the checkpoint archiver moves them")
    if category == "study folder" and unit.name in LIVE_SYSTEM:
        why.append("the live system's own folder: %s" % LIVE_SYSTEM[unit.name])
    if unit.name.endswith(LOCK_SUFFIX):
        why.append("a lock folder a running program holds")
    n_tracked = (ctx.get("tracked_dirs") or {}).get(SL._key(unit), 0)
    if n_tracked:
        why.append("holds %d file%s the repository tracks (code and tests read them by path; git "
                   "would read them as deleted)" % (n_tracked, "" if n_tracked == 1 else "s"))
    # every path the chain's args (or the saved worlds) name inside this folder: one reason per
    # path, naming the newest run that names it and how many others do
    by_path = {}
    for path, who in ctx.get("named") or ():
        if _inside(path, unit):
            by_path.setdefault(SL._key(path), [path, []])[1].append(who)
    for path, whos in by_path.values():
        more = len(whos) - 1
        why.append("%s names %s%s" % (whos[0], path, "" if not more else
                                      " (and %d more of the chain's records)" % more))
    if lineage and category == "caregiver records" and unit.name.startswith(lineage):
        why.append("the active lineage's companions and live recordings: the archiver moves them")
    # a resumed world writes its scene files into its SAVED run_dir, and every H84 world's
    # names the lineage's first run's scenes folder (checked 2026-09-22 on ckpt_step_344400);
    # the largest of these folders holds 29 MB, so keeping them costs nothing
    if lineage and category == "run subfolder" and run.name.startswith(lineage) \
            and unit.name == "scenes":
        why.append("the active lineage's worlds write their scene files here")
    for ck, kept in ctx["protected"].items():
        if _inside(ck, unit):
            why.append("holds %s (%s)" % (Path(ck).name, "; ".join(kept)))
    age_h = (ctx["now"] - newest) / 3600.0
    if age_h < MIN_AGE_HOURS:
        why.append("modified %.1f h ago (under %.0f h)" % (max(age_h, 0.0), MIN_AGE_HOURS))
    # THE ACTIVE RUN'S OWN SUBFOLDERS (verify-fix-ops 2026-09-23, the class of review finding 3:
    # kept only by the 6-hour rule and the process scan, so candidates after a quiet night - on the
    # 2026-09-23 plan with training paused since 17:15 the day before, segment 31's live, explain and
    # views folders were). The development dashboard reads <active run>/live for its Live tab and
    # opens <active run>/live/viewer.log for writing (development_dashboard.py), and the explainer
    # rebuilds from <active run>/explain. Measured cost of keeping them on segment 31: explain
    # 3.3 MB, live 0.08 MB, views 15 MB.
    chain = ctx.get("chain") or []
    if category == "run subfolder" and chain and SL._key(run) == SL._key(chain[0]):
        why.append("the active run's own folder: the development dashboard reads its live frames and "
                   "the explainer rebuilds from its snapshots")
    if unit.name == "videos" and any(SL._key(run) == SL._key(r) for r in ctx["dashboard_runs"]):
        why.append("its videos are shown by the development dashboard (first 3 runs of the "
                   "active resume chain)")
    for p in ctx["procs"]:
        hit = next((x for x in p["named"] if _covers(x, unit)), None)
        if hit:
            why.append("process %s (%s) names %s" % (p["pid"], p["name"], hit))
            continue
        f = next((x for x in p["open"] if _inside(x, unit)), None)
        if f:
            why.append("process %s (%s) has %s open" % (p["pid"], p["name"], f))
    if (unit.parent / (unit.name + POINTER_SUFFIX)).exists():
        why.append("a pointer from an earlier move sits beside it; restore that move first")
    return why


def plan(saved: Path = SAVED, lineage: str = None, procs=None, now: float = None, log=print,
         companion: bool = False, tracked=False):
    ctx = context(saved, lineage, procs, now, companion=companion, tracked=tracked)
    repo = Path(saved).parents[1]             # <repo>/Saved/mimo_bridge
    rows = []
    for unit, category in units(saved):
        try:
            walk = _walk(unit)
        except OSError as error:
            rows.append(dict(path=_rel(unit, repo), category=category, bytes=None, files=None,
                             stay=["could not be read: %s" % error], candidate=False))
            continue
        why = reasons_to_stay(unit, category, walk, ctx)
        rows.append(dict(path=_rel(unit, repo), category=category, bytes=walk[0], files=walk[2],
                         newest=walk[1], stay=why, candidate=not why))
    unreadable = sum(1 for p in ctx["procs"] if not p["readable"])
    summary = dict(time=ctx["now"], lineage=ctx["lineage"],
                   dashboard_runs=[r.name for r in ctx["dashboard_runs"]],
                   chain_runs=len(ctx["chain"]),
                   chain_root=ctx["chain"][-1].name if ctx["chain"] else None,
                   named_paths=len({SL._key(p) for p, _ in ctx.get("named") or ()}),
                   companion_read=bool(companion),
                   tracked_read=ctx.get("tracked_dirs") is not None,
                   processes=len(ctx["procs"]), processes_open_files_unreadable=unreadable,
                   candidates=sum(r["candidate"] for r in rows),
                   candidate_bytes=sum(r["bytes"] or 0 for r in rows if r["candidate"]),
                   units=len(rows))
    return dict(summary=summary, units=rows)


def _rel(path: Path, repo: Path = None) -> str:
    """The repo-relative path, '/'-separated; repo defaults to the one holding this module."""
    try:
        return Path(path).resolve().relative_to(Path(repo or REPO).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


# ------------------------------------------------------------------ moving
def destination_for(path: Path, repo: Path = REPO, root: Path = OFFLOAD_ROOT) -> Path:
    return root / Path(path).resolve().relative_to(Path(repo).resolve())


def _hashes(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): SL.sha256_file(p)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def _record(row, c_manifest: Path, d_manifest: Path, log):
    """The C: manifest first (storage_locations reads it first); then the D: copy."""
    SL.append_row(c_manifest, row)
    try:
        SL.append_row(d_manifest, dict(row, mirror_of=str(c_manifest)))
    except OSError as error:
        log("the D: copy of this row could not be written (%s); %s holds it" % (error, c_manifest))


def move(path, *, repo: Path = REPO, root: Path = OFFLOAD_ROOT, c_manifest: Path = None,
         d_manifest: Path = None, saved: Path = SAVED, ctx=None, log=print) -> bool:
    """Move one unit C: -> D:, verified. True only when the copy is proven and C: freed."""
    src = Path(path)
    if not src.is_absolute():
        src = Path(repo) / src
    c_manifest = c_manifest or SL.C_MANIFEST
    d_manifest = d_manifest or (Path(root) / "storage_moves.jsonl")
    if not src.is_dir():
        log("%s is not a folder on C:; nothing moved" % src)
        return False
    category = next((c for u, c in units(saved) if SL._key(u) == SL._key(src)), None)
    if category is None:
        log("%s is not a unit this tool moves (a run's subfolder, a study folder or a caregiver "
            "record folder under %s); nothing moved" % (src, saved))
        return False
    ctx = ctx or context(saved)
    walk = _walk(src)
    why = reasons_to_stay(src, category, walk, ctx)
    if why:
        log("%s stays: %s" % (_rel(src), "; ".join(why)))
        return False
    dst = destination_for(src, repo, root)
    earlier = SL.moves_in(c_manifest).get(SL._key(src)) or SL.moves_in(d_manifest).get(SL._key(src))
    if earlier is not None and Path(earlier["destination"]).exists():
        if _both_copies_proven(src, Path(earlier["destination"]), earlier.get("files")):
            # the one interrupted state this command can prove: its `moved` row was written and
            # the process died before the C: folder was taken away (finding 12)
            log("%s: an earlier move stopped after its copy was proven and recorded; finishing it"
                % _rel(src))
            return _take_c_away(src, Path(earlier["destination"]), earlier["files"],
                                walk[0], c_manifest, d_manifest, log)
        log("%s: an earlier move of this path is recorded and its copy is on D: (%s); restore or "
            "remove it first - nothing moved, nothing deleted" % (_rel(src), earlier["destination"]))
        return False
    files = _hashes(src)
    total = walk[0]
    if shutil.disk_usage(Path(root).anchor or root).free < total * 1.1:
        log("%s: D: has too little room for %.2f GB; nothing moved" % (_rel(src), total / 1e9))
        return False
    base = dict(kind="folder", source=str(src), destination=str(dst), files=files, bytes=total,
                tool="offload_storage")
    _record(dict(base, event="planned", time=time.time()), c_manifest, d_manifest, log)
    if dst.exists():
        shutil.rmtree(dst)                    # a partial copy from an interrupted move (no
                                              # moved row names it: checked above)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    if _hashes(dst) != files:
        shutil.rmtree(dst, ignore_errors=True)
        log("%s: a copied file did not match its original; C: left untouched" % _rel(src))
        return False
    _record(dict(base, event="moved", time=time.time()), c_manifest, d_manifest, log)
    return _take_c_away(src, dst, files, total, c_manifest, d_manifest, log)


def _both_copies_proven(src: Path, dst: Path, files) -> bool:
    """Whether the C: folder and its D: copy each hash exactly to the recorded files."""
    if not files or not Path(src).is_dir() or not Path(dst).is_dir():
        return False
    return _hashes(src) == files and _hashes(dst) == files


def _take_c_away(src: Path, dst: Path, files: dict, total, c_manifest: Path, d_manifest: Path,
                 log) -> bool:
    """The last step of a move, after the `moved` row: the C: folder goes, the pointer stays."""
    trash = src.with_name(src.name + ".offloading_%d" % os.getpid())
    try:
        src.rename(trash)                     # atomic on C:: the old path is gone all at once
    except OSError as error:
        _record(dict(kind="folder", source=str(src), destination=str(dst), files=files,
                     bytes=total, tool="offload_storage", event="restored", time=time.time(),
                     note="the C: folder could not be taken away (%s); the D: copy is removed"
                          % error), c_manifest, d_manifest, log)
        shutil.rmtree(dst, ignore_errors=True)
        log("%s is in use (%s); nothing moved" % (_rel(src), error))
        return False
    pointer = dict(moved_to=str(dst), manifest=str(c_manifest), files=len(files), bytes=total,
                   time=time.time(), restore="python -m mimo_bridge.tools.offload_storage "
                                             "restore %s" % _rel(src))
    src.with_name(src.name + POINTER_SUFFIX).write_text(json.dumps(pointer, indent=2),
                                                        encoding="utf-8")
    shutil.rmtree(trash, ignore_errors=True)
    if trash.exists():
        log("%s: moved, but %s could not be fully removed; delete it by hand" % (_rel(src), trash))
    log("%s moved to %s, %.2f GB, %d files" % (_rel(src), dst, (total or 0) / 1e9, len(files)))
    return True


def _finish_restore(src: Path, dst: Path, files: dict, total, c_manifest: Path, d_manifest: Path,
                    log) -> bool:
    """The last steps of a restore, after the verified copy is in place on C:.

    The pointer goes BEFORE the `restored` row (verify-fix-ops 2026-09-23). In the other order a
    process that died between the row and the pointer left a folder both commands refused for good:
    `restore` read the row and said it was already on C:, and the pointer kept `move` (and the plan)
    saying "restore that move first". In this order a death before the row leaves the state `restore`
    proves and finishes (C: and D: both match the `moved` row), and a death after it leaves only a
    spare D: copy, which the next `move` of the folder replaces."""
    pointer = src.with_name(src.name + POINTER_SUFFIX)
    if pointer.exists():
        pointer.unlink()
    _record(dict(kind="folder", source=str(src), destination=str(dst), files=files,
                 bytes=total, tool="offload_storage", event="restored",
                 time=time.time()), c_manifest, d_manifest, log)
    shutil.rmtree(dst, ignore_errors=True)
    log("%s restored from %s" % (_rel(src), dst))
    return True


def restore(path, *, repo: Path = REPO, root: Path = OFFLOAD_ROOT, c_manifest: Path = None,
            d_manifest: Path = None, log=print) -> bool:
    """Bring one moved unit back to its C: path, verified, and record it."""
    src = Path(path)
    if not src.is_absolute():
        src = Path(repo) / src
    c_manifest = c_manifest or SL.C_MANIFEST
    d_manifest = d_manifest or (Path(root) / "storage_moves.jsonl")
    # the C: record first; the D: mirror when the C: file was lost
    move_row = (SL.moves_in(c_manifest).get(SL._key(src))
                or SL.moves_in(d_manifest).get(SL._key(src)))
    if src.exists():
        if (move_row is not None and move_row["kind"] == "folder"
                and _both_copies_proven(src, Path(move_row["destination"]), move_row.get("files"))):
            # the interrupted state (finding 12): a restore renamed its copy into place and died
            # before writing `restored`, or a move wrote `moved` and died before taking C: away.
            # Both copies are proven, so the restore asked for is finished
            log("%s: on C: and on D:, both proven against the record; finishing the restore" % src)
            return _finish_restore(src, Path(move_row["destination"]), move_row["files"],
                                   move_row.get("bytes"), c_manifest, d_manifest, log)
        log("%s is already on C:; nothing restored" % src)
        return False
    if move_row is None or move_row["kind"] != "folder":
        log("no move of %s is recorded in %s; nothing restored" % (src, c_manifest))
        return False
    dst, files = Path(move_row["destination"]), move_row["files"]
    if not dst.is_dir() or _hashes(dst) != files:
        log("the D: copy %s is missing or differs from its record; nothing restored" % dst)
        return False
    if shutil.disk_usage(src.anchor).free < sum(p.stat().st_size for p in dst.rglob("*")
                                                 if p.is_file()) * 1.1:
        log("C: has too little room to restore %s" % src)
        return False
    staging = src.with_name(src.name + ".restoring_%d" % os.getpid())
    if staging.exists():
        shutil.rmtree(staging)                # left by an interrupted restore
    shutil.copytree(dst, staging)
    if _hashes(staging) != files:
        shutil.rmtree(staging, ignore_errors=True)
        log("a restored file did not match its record; D: copy kept, C: unchanged")
        return False
    staging.rename(src)
    return _finish_restore(src, dst, files, move_row.get("bytes"), c_manifest, d_manifest, log)


# ------------------------------------------------------------------ command line
def _table(report, log=print, top=40):
    s = report["summary"]
    cands = sorted((u for u in report["units"] if u["candidate"]),
                   key=lambda u: -(u["bytes"] or 0))
    log("OFFLOAD PLAN (nothing moved). %d folders considered; %d could move, %.2f GB. Active "
        "lineage %s; dashboard video runs %s. Open files of %d of %d processes could not be "
        "read, so a file only those processes hold open is not seen."
        % (s["units"], s["candidates"], s["candidate_bytes"] / 1e9, s["lineage"],
           ", ".join(s["dashboard_runs"]) or "none", s["processes_open_files_unreadable"],
           s["processes"]))
    by_cat = {}
    for u in cands:
        c = by_cat.setdefault(u["category"], [0, 0])
        c[0] += 1
        c[1] += u["bytes"] or 0
    for cat, (n, b) in sorted(by_cat.items(), key=lambda kv: -kv[1][1]):
        log("  %-18s %4d folders  %7.2f GB" % (cat, n, b / 1e9))
    for u in cands[:top]:
        log("  %8.3f GB  %s" % ((u["bytes"] or 0) / 1e9, u["path"]))
    reasons = {}
    for u in report["units"]:
        for r in u["stay"]:
            key = re.sub(r"\d[\d.]*", "N", r.split(" (")[0])[:70]
            reasons[key] = reasons.get(key, 0) + 1
    log("reasons folders stay (count):")
    for r, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        log("  %4d  %s" % (n, r))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan", help="list what could move; moves nothing, writes only --out")
    pl.add_argument("--out", type=Path, default=None)
    pl.add_argument("--lineage", default=None)
    pl.add_argument("--companion", action="store_true",
                    help="also read the resume checkpoint's episode companion (about 600 MB) for "
                         "the saved worlds' env settings; without it they are taken from the "
                         "resume chain's args.json files, which name every one of them")
    for name in ("move", "restore"):
        sp = sub.add_parser(name)
        sp.add_argument("paths", nargs="+")
    a = p.parse_args(argv)
    if a.cmd == "plan":
        report = plan(lineage=a.lineage, companion=a.companion)
        _table(report)
        if a.out:
            a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        return 0
    if a.cmd == "move":
        ctx = context()
        ok = [move(x, ctx=ctx) for x in a.paths]
    else:
        ok = [restore(x) for x in a.paths]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
