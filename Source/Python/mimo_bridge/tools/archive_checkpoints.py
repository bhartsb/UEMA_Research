# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Move a lineage's older checkpoints from C: to the D: archive, verified, so C: never fills.

WHY THIS EXISTS. On 2026-09-21 at 11:03 the H84 lineage stopped because C: had 3.1 GB free,
under the watcher's 8 GB line. Each hourly checkpoint costs about 1.2 GB on C: - 606 MB of
model, optimiser and runtime state in the run folder, and a 595 MB caregiver companion under
caregiver_records - and nothing removed any of them: the retention policy
(docs/Checkpoint_Retention_2026-09-13.md) enrolls only folders on D:. Sixty-six checkpoints
and forty-four companions had reached 68 GB. The last checkpoint was written with 1.71 GB
free against its 594 MB companion. The lineage then sat stopped for almost seven hours.

WHAT IT DOES, the same procedure a session used by hand on 2026-09-20 for 43 checkpoints, and
it appends to that session's manifest in the same format so the archive has one record:
  1. every file of the checkpoint folder is hashed (SHA-256) and the companion's hash is
     checked against the hash the checkpoint itself recorded when it was written,
  2. both are copied to D:/UEMA_Research/external/D/UEMA_recordings/H84_checkpoint_archive,
  3. every copy is hashed again and compared,
  4. only when every hash matches are the C: originals removed and one line appended to
     move_manifest.jsonl. Any mismatch removes the partial copy and leaves C: untouched.

WHAT IT NEVER MOVES, following the retention document: the latest three complete checkpoints
of the lineage, the checkpoint the supervisor's continuation file names as its resume source,
the newest checkpoint of the run the supervisor holds, and any checkpoint written in the last
30 minutes, so a checkpoint still being published is never touched. A checkpoint folder
without its lineage record is incomplete and is left where it is. And every checkpoint the keep
list Saved/mimo_bridge/archive_keep.json names, with the reason it gives (from 2026-09-23: the
build's proofs load DESIGN D3's baseline ckpt_step_344400 by its C: path and skip when it is
gone, so a move would turn every such proof into a skip that reads like a pass; review
ops-storage finding 4). A keep list that exists but cannot be read stops the pass: nothing moves
until it is repaired, because a list silently ignored would move what it was written to keep.

A checkpoint moved here can still be resumed: its companion is at the path the manifest line
records under companion_destination, and the checkpoint's own episode_state.json keeps the
hash that proves it is the same file. continuation_state.load finds it there through
mimo_bridge.storage_locations, which reads these rows.

EVERY ROW IS ALSO WRITTEN TO C: (Saved/mimo_bridge/storage_moves.jsonl, from 2026-09-22), with
mirror_of naming the D: manifest, so the record of where things went survives the D: disk being
absent, full or replaced; storage_locations reads the C: copy first.

ONE ARCHIVER AT A TIME. A pass that runs twice at once can copy the same checkpoint twice and
remove it under the other's copy. main() holds Saved/mimo_bridge/checkpoint_archiver.lock and
refuses to start while another archiver process runs (tools/training_continuation.py starts
one only when none runs).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from mimo_bridge.storage_locations import append_row, long_path

REPO = Path(__file__).resolve().parents[4]
SAVED = REPO / "Saved" / "mimo_bridge"
ARCHIVE = Path("D:/UEMA_Research/external/D/UEMA_recordings/H84_checkpoint_archive")
KEEP_LATEST = 3
MIN_AGE_MINUTES = 30.0
MIRROR_NAME = "storage_moves.jsonl"          # under SAVED: the C: copy of every manifest row
LOCK_NAME = "checkpoint_archiver.lock"       # under SAVED: held by the one running archiver
STATUS_NAME = "checkpoint_archive_status.json"   # under SAVED: written after every pass
KEEP_NAME = "archive_keep.json"              # under SAVED: checkpoints that stay on C:, each with why
# NOT THE LINEAGE, though the folder carries its name. explain_probe writes a fresh
# experiments/<run>_explainprobe_<time> folder holding an end checkpoint and companion (~1.2 GB
# at H84 size); deploy scripts prove new code in a _deploy_proof_ folder. Their checkpoints are
# not the lineage's history: counted, a probe's end checkpoint at the newest step would take one
# of the three places the latest real checkpoints keep on C:. The probe outputs of DESIGN.md
# section 9 (Saved/mimo_bridge/gof_probes) are excluded by path as well.
NOT_LINEAGE = re.compile(r"_explainprobe_|_deploy_proof", re.IGNORECASE)
PROBE_ROOT_NAME = "gof_probes"
# A TAPE MANIFEST WRITTEN IN THE LAST 10 s IS NOT READ THIS PASS. The live recorder rewrites a
# tape's manifest by replacing the file, and Windows refuses that replace while a reader holds
# the file open (at 11:46:25 on 2026-09-22 this pass's read made the trainer's recorder fail
# with "[WinError 5] Access is denied" and drop 143 caregiver events). The writer now retries
# (event_tape._json_file); this pass also stays off a manifest that is being written. Skipping
# one is safe: the recorder writes a tape's newest chunk in the same flush as its manifest, so
# that chunk is seconds old and far inside the 6-hour idle rule that keeps chunks on C:; a tape
# closed without new data has no open chunk. A skipped manifest is read on the next pass.
# From the measured intake (3.5-4.9 GB a day over 8 recorders, archive_tapes below) each
# recorder fills its 64 KB flush every 9-13 s, so a live manifest is inside this window most of
# the time; it is never moved anyway (it is under 6 hours old). The window therefore protects
# little against a writer on that period - the writer's retry is the fix - and costs nothing.
MANIFEST_QUIET_S = 10.0


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def folder_hashes(folder: Path) -> dict:
    return {str(p.relative_to(folder)).replace("\\", "/"): sha256(p)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def folder_bytes(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def free_bytes(path) -> int:
    return shutil.disk_usage(str(path)).free


def _step(path) -> int:
    m = re.search(r"ckpt_step_(\d+)", str(path))
    return int(m.group(1)) if m else -1


def _inside(path: Path, folder: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(folder).resolve())
        return True
    except ValueError:
        return False


def inventory(lineage: str, saved: Path = SAVED):
    """Every checkpoint of the lineage on C:, with its companion if it has one. Probe and proof
    folders that share the lineage's name are not the lineage (NOT_LINEAGE)."""
    out = []
    for d in sorted(glob.glob(str(saved / "experiments" / (lineage + "*") / "checkpoints" /
                                  "ckpt_step_*"))):
        d = Path(d)
        if not d.is_dir():
            continue
        run = d.parent.parent.name
        if NOT_LINEAGE.search(run) or _inside(d, saved / PROBE_ROOT_NAME):
            continue
        receipt = d / "episode_state.json"
        companion, recorded = None, None
        if receipt.exists():
            try:
                r = json.loads(receipt.read_text(encoding="utf-8"))
                companion = Path(r["storage_path"]) if r.get("storage_path") else None
                recorded = r.get("sha256")
            except (ValueError, KeyError, OSError):
                companion = None
        out.append(dict(run=run, step=_step(d), path=d,
                        complete=(d / "lineage.json").exists(),
                        mtime=max((p.stat().st_mtime for p in d.rglob("*") if p.is_file()),
                                  default=d.stat().st_mtime),
                        companion=companion, companion_sha256=recorded))
    return out


class KeepListError(ValueError):
    """The keep list exists but cannot be read: no pass may move anything until it is repaired."""


def read_keep_list(saved: Path = SAVED) -> list:
    """[{path, reason}] from <saved>/archive_keep.json; [] when the file does not exist.

    Format: {"keep": [{"path": "<checkpoint folder>", "reason": "<why it stays on C:>"}, ...]}.
    Raises KeepListError when the file exists but is not that shape."""
    f = Path(saved) / KEEP_NAME
    if not f.exists():
        return []
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        rows = data["keep"]
        out = [dict(path=str(r["path"]), reason=str(r["reason"])) for r in rows]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise KeepListError("the keep list %s cannot be read (%s: %s); nothing moves until it is "
                            "repaired" % (f, type(error).__name__, error)) from error
    return out


def protected(items, continuation: dict, keep_latest: int = KEEP_LATEST,
              min_age_minutes: float = MIN_AGE_MINUTES, now: float = None, keep_list=None):
    """The checkpoints that stay on C:, each with the reason it stays.

    keep_list: [{path, reason}] (read_keep_list); None reads Saved/mimo_bridge/archive_keep.json."""
    now = time.time() if now is None else now
    keep = {}
    listed = read_keep_list() if keep_list is None else keep_list
    if listed:
        from mimo_bridge.storage_locations import _key
        wanted = {_key(r["path"]): r["reason"] for r in listed}
        for i in items:
            why = wanted.get(_key(i["path"]))
            if why is not None:
                keep.setdefault(str(i["path"]), []).append("named in %s: %s" % (KEEP_NAME, why))
    complete = sorted((i for i in items if i["complete"]), key=lambda i: i["step"])
    for i in complete[-keep_latest:] if keep_latest > 0 else []:
        keep.setdefault(str(i["path"]), []).append("one of the latest %d" % keep_latest)
    cmd = continuation.get("command") or []
    if "--resume" in cmd:
        resume = Path(cmd[cmd.index("--resume") + 1])
        for i in items:
            if i["path"].resolve() == resume.resolve():
                keep.setdefault(str(i["path"]), []).append("the supervisor's resume source")
    active = continuation.get("active_run")
    if active:
        mine = [i for i in items if i["path"].parent.parent.resolve() == Path(active).resolve()]
        if mine:
            newest = max(mine, key=lambda i: i["step"])
            keep.setdefault(str(newest["path"]), []).append("the held run's newest checkpoint")
    for i in items:
        if now - i["mtime"] < min_age_minutes * 60.0:
            keep.setdefault(str(i["path"]), []).append(
                "written under %.0f minutes ago" % min_age_minutes)
        if not i["complete"]:
            keep.setdefault(str(i["path"]), []).append("incomplete: no lineage record")
    return keep


def _copy_verified(src: Path, dst: Path, want: dict) -> bool:
    """Copy a folder and check every file's hash on arrival. Removes the copy on a mismatch."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    got = folder_hashes(dst)
    if got != want:
        shutil.rmtree(dst, ignore_errors=True)
        return False
    return True


def record_move(manifest: Path, row: dict, mirror: Path = None, log=print):
    """One row into the D: manifest, then the same row (with mirror_of) into the C: mirror.

    The D: row is written first and must succeed: it is the archive's own record, and nothing is
    removed from C: without it. A failed mirror write is reported and does not stop the move:
    storage_locations still finds the row in the D: manifest."""
    append_row(manifest, row)
    if mirror is None:
        return
    try:
        append_row(mirror, dict(row, mirror_of=str(manifest)))
    except OSError as error:
        log("  the C: copy of the move record could not be written (%s); the D: manifest %s "
            "holds it" % (error, manifest))


def _row_identity(row: dict) -> str:
    return json.dumps({k: v for k, v in row.items()
                       if k not in ("mirror_of", "mirror_backfilled_at")}, sort_keys=True)


def backfill_mirror(saved: Path = SAVED, archive: Path = ARCHIVE, tape_archive: Path = None,
                    log=print, now: float = None) -> dict:
    """Append to the C: mirror every row of the two D: archive manifests it does not hold yet.

    WHY. The C: mirror exists so the record of where things went survives the D: disk being
    absent (above), but it started empty on 2026-09-22: the rows written before it existed - 130
    checkpoints and 1,842 recording files by the review's count that day - were on D: only
    (review ops-storage finding 7). Each copied row is the D: row unchanged plus mirror_of (the
    D: manifest) and mirror_backfilled_at (when it was copied), so a reader can tell a row copied
    later from one mirrored at its move. Rows the mirror already holds (same fields apart from
    those two) are skipped, so running it twice appends nothing. Nothing is moved or deleted."""
    from mimo_bridge.storage_locations import _rows
    if tape_archive is None:
        from mimo_bridge.event_tape import TAPE_ARCHIVE as tape_archive
    now = time.time() if now is None else now
    mirror = Path(saved) / MIRROR_NAME
    have = {_row_identity(r) for r in _rows(mirror)}
    out = dict(appended=0, already=0, unreadable=[])
    for manifest in (Path(archive) / "move_manifest.jsonl",
                     Path(tape_archive) / "tape_move_manifest.jsonl"):
        if not manifest.exists():
            out["unreadable"].append(str(manifest))
            log("backfill: %s cannot be read; its rows are not copied" % manifest)
            continue
        for row in _rows(manifest):
            ident = _row_identity(row)
            if ident in have:
                out["already"] += 1
                continue
            append_row(mirror, dict(row, mirror_of=str(manifest), mirror_backfilled_at=now))
            have.add(ident)
            out["appended"] += 1
    log("backfill: %d rows copied to %s, %d already there" % (out["appended"], mirror,
                                                             out["already"]))
    return out


def archive_one(item, archive: Path, manifest: Path, log=print, still_needed=(),
                mirror: Path = None) -> bool:
    """Move one checkpoint and its companion. True only when both arrived verified and the C:
    originals were removed.

    ONE COMPANION CAN BELONG TO SEVERAL CHECKPOINTS. A checkpoint folder copied by hand -
    ckpt_step_220675 and ckpt_step_220675_entity_self_carried in seg10_toys are one example -
    names the same companion file as the folder it was copied from. So the companion is
    COPIED to the archive and removed from C: only when no checkpoint left on C: names it
    (still_needed holds the companion paths the remaining checkpoints name). A companion a
    sibling has already taken is found in the archive, checked against the hash this
    checkpoint recorded, and used from there."""
    src, run = item["path"], item["run"]
    dst = archive / run / "checkpoints" / src.name
    files = folder_hashes(src)
    total = folder_bytes(src)
    comp_src, comp_dst, comp_hash = item["companion"], None, None
    # EVERY OPERATION ON THE COMPANION GOES THROUGH THE EXTENDED-LENGTH FORM (2026-09-24). A
    # companion's path repeats the run name, and a supervisor resume folder's
    # (<segment>_resume_<time>) put it at 262-268 characters, past Windows' 260: Path.exists()
    # said False for 23 of 26 companions that were there, this pass moved nothing all day, C: fell
    # to 0.8 GB free at 1:29 PM PDT and the watcher stopped H84, whose own final save then failed
    # for lack of room. storage_locations.long_path is what the writer and every reader use.
    comp_io = long_path(comp_src) if comp_src is not None else None
    comp_on_c = comp_io is not None and os.path.isfile(comp_io)
    if comp_src is not None:
        comp_dst = archive / "checkpoint_state" / comp_src.parent.name / comp_src.name
        if comp_on_c:
            comp_hash = sha256(comp_io)
        elif comp_dst.exists():
            comp_hash = sha256(comp_dst)          # a sibling already archived it
        else:
            log("  step %d: its companion %s is neither on C: nor in the archive; "
                "left where it is" % (item["step"], comp_src.name))
            return False
        if item["companion_sha256"] and comp_hash != item["companion_sha256"]:
            log("  step %d: the companion's hash %s is not the %s its checkpoint recorded; "
                "left where it is" % (item["step"], comp_hash[:12], item["companion_sha256"][:12]))
            return False
        if comp_on_c:
            total += os.stat(comp_io).st_size
    if free_bytes(archive.anchor if archive.anchor else archive) < total * 1.1:
        log("  step %d: the archive disk has too little room; stopping" % item["step"])
        return False
    if not _copy_verified(src, dst, files):
        log("  step %d: a copied file did not match its original; C: left untouched"
            % item["step"])
        return False
    if comp_on_c and not (comp_dst.exists() and sha256(comp_dst) == comp_hash):
        comp_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(comp_io, comp_dst)
        if sha256(comp_dst) != comp_hash:
            comp_dst.unlink(missing_ok=True)
            shutil.rmtree(dst, ignore_errors=True)
            log("  step %d: the copied companion did not match; C: left untouched"
                % item["step"])
            return False
    # every copy is verified: the originals can go, and the move is recorded first
    row = dict(event="moved", step=item["step"], run=run, source=str(src),
               destination=str(dst), companion_source=str(comp_src) if comp_src else None,
               companion_destination=str(comp_dst) if comp_dst else None,
               companion_sha256=comp_hash, files=files, bytes=total, time=time.time())
    record_move(manifest, row, mirror, log)
    shutil.rmtree(src)
    kept = ""
    if comp_on_c:
        if str(comp_src) in {str(x) for x in still_needed}:
            kept = ", its companion kept on C: for a checkpoint that still names it"
        else:
            os.unlink(comp_io)
    log("  step %7d  %-48s moved, %.2f GB%s" % (item["step"], run, total / 1e9, kept))
    return True


def run_once(lineage: str, target_free_gb: float, dry_run: bool, saved: Path = SAVED,
             archive: Path = ARCHIVE, keep_latest: int = KEEP_LATEST,
             min_age_minutes: float = MIN_AGE_MINUTES, move_all: bool = False, log=print) -> dict:
    cont_path = saved / "training_continuation.json"
    continuation = (json.loads(cont_path.read_text(encoding="utf-8"))
                    if cont_path.exists() else {})
    items = inventory(lineage, saved)
    free0 = free_bytes(saved)
    try:
        listed = read_keep_list(saved)
    except KeepListError as error:
        log(str(error))
        return dict(moved=0, eligible=0, free_gb=free0 / 1e9, error=str(error))
    keep = protected(items, continuation, keep_latest, min_age_minutes, keep_list=listed)
    eligible = sorted((i for i in items if str(i["path"]) not in keep), key=lambda i: i["step"])
    log("%s: %d checkpoints on C:, %d stay, %d can move; C: has %.1f GB free, target %.1f GB"
        % (lineage, len(items), len(keep), len(eligible), free0 / 1e9, target_free_gb))
    for p, why in sorted(keep.items(), key=lambda kv: _step(kv[0])):
        log("  stays: step %7d  %s" % (_step(p), "; ".join(why)))
    moved = 0
    if dry_run:
        for i in eligible:
            log("  would move: step %7d  %s%s" % (i["step"], i["run"],
                                                 "" if i["companion"] else " (no companion)"))
        return dict(moved=0, eligible=len(eligible), free_gb=free0 / 1e9)
    try:
        archive.mkdir(parents=True, exist_ok=True)
        probe = archive / (".write_probe_%d" % os.getpid())
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as error:
        log("the archive %s cannot be written (%s); nothing moved" % (archive, error))
        return dict(moved=0, eligible=len(eligible), free_gb=free0 / 1e9, error=str(error))
    remaining = list(items)
    for i in eligible:
        if not move_all and free_bytes(saved) >= target_free_gb * 1e9:
            break
        remaining = [r for r in remaining if r["path"] != i["path"]]
        needed = {str(r["companion"]) for r in remaining if r["companion"] is not None}
        if archive_one(i, archive, archive / "move_manifest.jsonl", log=log,
                       still_needed=needed, mirror=saved / MIRROR_NAME):
            moved += 1
        else:
            break
    free1 = free_bytes(saved)
    log("moved %d; C: now has %.1f GB free" % (moved, free1 / 1e9))
    return dict(moved=moved, eligible=len(eligible), free_gb=free1 / 1e9)


def archive_tapes(lineage: str, saved: Path = SAVED, archive_root: Path = None,
                  min_age_hours: float = 6.0, log=print, now: float = None,
                  dry_run: bool = False) -> dict:
    """Move the caregiver's CLOSED recording chunks to D:, verified, so they stop filling C:.

    WHY. The checkpoints above were the only thing moved. The caregiver's event recording kept
    growing on C: beside them: on 2026-09-22 caregiver_records/H84_fresh_drivers_20260918 held
    953 chunk files, 17.4 GB, written at 3.5 to 4.9 GB a day, and nothing ever moved one.

    WHAT IS CLOSED. A tape's writer appends only to the LAST chunk its manifest lists; a new
    chunk starts at the 64 MB limit, and a tape restored from a checkpoint or copied at a
    restore starts a new chunk before it writes (EventTape.__deepcopy__ clears the open chunk;
    a restored tape is built without one). So every chunk but the last of a tape is closed for
    good. A chunk moves when nothing has written it for min_age_hours and it is not the last
    chunk of any tape written in that time - a live caregiver rewrites its manifest on every
    flush, so a tape idle that long belongs to a process that has ended. The live episode's
    tapes list its whole history back to the episode's start, so this is what lets the
    growing part move; its readers are replay and export only, never training, and
    event_tape.chunk_path finds a moved chunk in the archive.

    A tape manifest moves when it has been idle min_age_hours and every chunk it lists is
    moving in this pass or already in the archive, so a moved manifest never names a chunk
    left behind on C:.

    Each file is copied, hashed on both sides, recorded in tape_move_manifest.jsonl, and only
    then removed from C:. A mismatch removes the copy and stops the pass."""
    from mimo_bridge.event_tape import TAPE_ARCHIVE
    archive_root = TAPE_ARCHIVE if archive_root is None else Path(archive_root)
    now = time.time() if now is None else now
    src_dir = saved / "caregiver_records" / lineage
    if not src_dir.is_dir():
        return dict(moved_files=0, moved_bytes=0)
    dst_dir = archive_root / src_dir.name
    cutoff = now - min_age_hours * 3600.0
    tapes, being_written = {}, []
    for t in src_dir.glob("*.tape.json"):
        try:
            mtime = t.stat().st_mtime
        except OSError:
            continue                          # replaced or moved between the listing and now
        if now - mtime < MANIFEST_QUIET_S:
            being_written.append(t.name)      # see MANIFEST_QUIET_S: never read mid-write
            continue
        try:
            spans = json.loads(t.read_text(encoding="utf-8")).get("spans") or []
        except (OSError, ValueError):
            log("tapes: %s could not be read, so no chunk can be proven closed; nothing moved"
                % t.name)
            return dict(moved_files=0, moved_bytes=0)
        tapes[t] = (mtime, spans)
    if being_written:
        log("tapes: %d manifests written in the last %.0f s were not read this pass (a live "
            "recorder's; its newest chunk is seconds old and stays on C: by the idle rule)"
            % (len(being_written), MANIFEST_QUIET_S))
    open_chunks = {spans[-1]["file"] for mtime, spans in tapes.values()
                   if spans and mtime >= cutoff}
    chunks = [c for c in src_dir.glob("*.jsonl")
              if c.name not in open_chunks and c.stat().st_mtime < cutoff]
    closed = {c.name for c in chunks}
    manifests = [t for t, (mtime, spans) in tapes.items()
                 if mtime < cutoff and all(s["file"] in closed or (dst_dir / s["file"]).exists()
                                           for s in spans)]
    todo = chunks + manifests
    total = sum(p.stat().st_size for p in todo)
    log("tapes: %d chunk files and %d manifests on C: are closed (idle since %s), %.2f GB"
        % (len(chunks), len(manifests), time.strftime("%Y-%m-%d %H:%M", time.localtime(cutoff)),
           total / 1e9))
    if not todo or dry_run:
        return dict(moved_files=0, moved_bytes=0, closed_files=len(todo), closed_bytes=total)
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        log("tapes: the archive %s cannot be written (%s); nothing moved" % (dst_dir, error))
        return dict(moved_files=0, moved_bytes=0, error=str(error))
    if free_bytes(dst_dir) < total * 1.1:
        log("tapes: the archive disk has too little room; nothing moved")
        return dict(moved_files=0, moved_bytes=0)
    manifest = archive_root / "tape_move_manifest.jsonl"
    moved, moved_bytes = 0, 0
    for src in todo:                          # chunks first: a manifest never outruns its data
        dst = dst_dir / src.name
        want = sha256(src)
        shutil.copy2(src, dst)
        if sha256(dst) != want:
            dst.unlink(missing_ok=True)
            log("tapes: %s did not copy intact; C: left untouched, pass stopped" % src.name)
            break
        size = src.stat().st_size
        record_move(manifest, dict(event="moved", source=str(src), destination=str(dst),
                                   sha256=want, bytes=size, time=time.time()),
                    saved / MIRROR_NAME, log)
        src.unlink()
        moved, moved_bytes = moved + 1, moved_bytes + size
    log("tapes: moved %d files, %.2f GB; C: now has %.1f GB free"
        % (moved, moved_bytes / 1e9, free_bytes(saved) / 1e9))
    return dict(moved_files=moved, moved_bytes=moved_bytes)


def is_archiver_command(cmdline) -> bool:
    """A process that moves checkpoints: this module run with -m, or this file run as a script.
    A --dry-run only lists and a --backfill-mirror only copies records; neither counts."""
    c = list(cmdline or [])
    named = ("mimo_bridge.tools.archive_checkpoints" in c
             or any(Path(x).name == "archive_checkpoints.py" for x in c))
    return named and "--dry-run" not in c and "--backfill-mirror" not in c


def archiver_pids(exclude=()):
    """PIDs of running archivers, one per archiver: Windows' venv python.exe is a launcher with
    a child running the same command line, so a match whose parent also matches is dropped."""
    import psutil
    found = {}
    for p in psutil.process_iter(["pid", "ppid", "name", "cmdline"]):
        try:
            if "python" not in (p.info["name"] or "").lower():
                continue
            if p.info["pid"] in exclude or not is_archiver_command(p.info["cmdline"]):
                continue
            found[p.info["pid"]] = p.info["ppid"]
        except (psutil.Error, OSError):
            continue
    return sorted(pid for pid, parent in found.items() if parent not in found)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--lineage", default="H84_fresh_drivers_20260918")
    p.add_argument("--target-free-gb", type=float, default=60.0,
                   help="stop moving once C: has this much free (ignored with --all)")
    p.add_argument("--all", action="store_true", help="move every checkpoint that may move")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-tapes", action="store_true",
                   help="leave the caregiver's closed recording chunks on C: (see archive_tapes)")
    p.add_argument("--watch", action="store_true", help="repeat every --interval seconds")
    p.add_argument("--interval", type=float, default=900.0)
    p.add_argument("--report", type=Path, default=None,
                   help="status file (default %s). A --dry-run writes none unless this is given: "
                        "the hourly watcher reads the file's age as the archiver's heartbeat, "
                        "and a listing must not stand in for a pass that moves" % STATUS_NAME)
    p.add_argument("--lock", type=Path, default=SAVED / LOCK_NAME,
                   help="held while this archiver runs; a second archiver refuses to start")
    p.add_argument("--backfill-mirror", action="store_true",
                   help="copy every D: manifest row the C: mirror (%s) lacks, then exit; moves "
                        "nothing (see backfill_mirror)" % MIRROR_NAME)
    a = p.parse_args(argv)
    if a.backfill_mirror:
        backfill_mirror()
        return 0
    if a.report is None and not a.dry_run:
        a.report = SAVED / STATUS_NAME
    lock = None
    if not a.dry_run:
        # never a second archiver: one found by its command line (an archiver started on code
        # older than this lock holds no lock), or one holding the lock
        others = archiver_pids(exclude={os.getpid(), os.getppid()})
        if others:
            print("another checkpoint archiver is running (pid %s); this one does not start"
                  % ", ".join(map(str, others)), flush=True)
            return 2
        from mimo_bridge.tools.training_continuation import OwnerLock
        try:
            a.lock.parent.mkdir(parents=True, exist_ok=True)
            lock = OwnerLock(a.lock)
        except OSError:
            print("another checkpoint archiver holds %s; this one does not start" % a.lock,
                  flush=True)
            return 2
    try:
        return _passes(a)
    finally:
        if lock is not None:
            lock.close()


def _passes(a):
    while True:
        lines = []

        def log(msg):
            lines.append(msg)
            print(msg, flush=True)
        result = run_once(a.lineage, a.target_free_gb, a.dry_run, move_all=a.all, log=log)
        if not a.no_tapes:
            try:
                result["tapes"] = archive_tapes(a.lineage, log=log, dry_run=a.dry_run)
            except OSError as error:
                log("tapes: pass failed (%s); C: originals are removed only after a verified copy"
                    % error)
                # in the status too: the hourly watcher quotes a status error as an ALERT, and a
                # failure only in the printed lines would leave a fresh heartbeat reading as well
                result["tapes"] = dict(moved_files=0, moved_bytes=0, error=str(error))
        if a.report is not None:
            try:
                a.report.write_text(json.dumps(dict(result, time=time.time(), lines=lines[-40:],
                                                    pid=os.getpid(), interval_s=a.interval,
                                                    watch=a.watch, dry_run=a.dry_run,
                                                    lineage=a.lineage, archive=str(ARCHIVE)),
                                               indent=2), encoding="utf-8")
            except OSError:
                pass
        if not a.watch:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
