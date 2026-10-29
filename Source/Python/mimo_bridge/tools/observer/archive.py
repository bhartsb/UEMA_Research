# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Keep every observer movement record: copy a run's raw chunks to the archive disk before anything deletes them.

The operator's request of October 18, 2026 (the movement viewer, docs/H84_MOVEMENT_VIEWER_PLAN_2026-10-18.md, part 1):
pictures of any saved command, in training and in tests, on any date. The digester removes a digested raw chunk 72 hours
after its file time (daemon.RAW_KEEP_S) or while its body folder is past four fifths of the recorder's budget, and a
picture of a command cannot be drawn without the chunk that holds it. So every raw chunk is first copied here:

    <archive root>/<run folder name>/observer/world/b<body>/chunk_*.npz   (and the body's header_*.json, gaps.jsonl)
    <archive root>/<run folder name>/observer/world/models/<fingerprint>.mjb.gz   (the saved model the headers name)
    <archive root>/<run folder name>/observer/model/chunk_*.npz, header_*.json   (the learner lane)
    <archive root>/<run folder name>/args.json   (the run's launch arguments)

with the same file names and file times (shutil.copy2), so an archived run folder reads like the original one. A copy
counts only when its size and SHA-256 equal the original's. The digester's housekeeping removes a chunk only after
ensure() says its copy checks out; a missing archive disk or a failed copy keeps the chunk (every doubt keeps records,
as with the pins in pins.py) and the housekeeping report counts it.

The archive root is ARCHIVE_ROOT unless the environment variable UEMA_OBSERVER_ARCHIVE names another folder. Cost: about
1.8 KB per body per model step in all, the world records about 1.3 KB of it (measured Oct 18, 2026 on seg41r;
docs/H84_MOVEMENT_VIEWER_PLAN_2026-10-18.md, part 1).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import time
from typing import Dict, List, Optional, Tuple

ARCHIVE_ROOT = "D:/UEMA_archive/observer_raw"
ENV = "UEMA_OBSERVER_ARCHIVE"
MANIFEST = "archive_manifest.json"


def _lp(path: str) -> str:
    """The extended-length form of a long Windows path (observer_record._lp)."""
    if os.name == "nt" and len(path) >= 240 and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + os.path.abspath(path)
    return path


def _replace(tmp: str, dst: str, tries: int = 100, wait_s: float = 0.02) -> None:
    """os.replace, retried for up to tries x wait_s = 2 s: on Windows it fails with PermissionError while another
    process has `dst` open (the movement viewer reads archived files); after the last try the error is raised."""
    for i in range(tries):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(wait_s)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(_lp(path), "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Archive:
    """One archive root. ensure() for one file before it is deleted; run_pass() for a whole run folder."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)

    @classmethod
    def configured(cls) -> "Archive":
        return cls(os.environ.get(ENV) or ARCHIVE_ROOT)

    def available(self) -> Optional[str]:
        """None when the archive can be written; otherwise why not (the disk or its parent folder is missing)."""
        drive, _ = os.path.splitdrive(self.root)
        anchor = (drive + os.sep) if drive else os.path.dirname(self.root)
        if not os.path.isdir(anchor):
            return "the archive disk %s is not available" % anchor
        try:
            os.makedirs(self.root, exist_ok=True)
        except OSError as ex:
            return "the archive folder %s cannot be made (%s)" % (self.root, ex)
        return None

    def run_folder(self, run_dir: str) -> str:
        return os.path.join(self.root, os.path.basename(os.path.abspath(run_dir)))

    def target(self, run_dir: str, path: str) -> str:
        """Where `path` (a file inside run_dir) is kept in the archive."""
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(run_dir))
        if rel.startswith(".."):
            raise ValueError("%s is not inside the run folder %s" % (path, run_dir))
        return os.path.join(self.run_folder(run_dir), rel)

    @staticmethod
    def same(original: str, copy: str) -> bool:
        """The copy exists with the original's size and SHA-256."""
        try:
            if os.path.getsize(_lp(copy)) != os.path.getsize(_lp(original)):
                return False
        except OSError:
            return False
        return sha256(copy) == sha256(original)

    @staticmethod
    def _copy(src: str, dst: str) -> None:
        os.makedirs(os.path.dirname(_lp(dst)), exist_ok=True)
        tmp = "%s.%d.part" % (dst, os.getpid())      # the digester and the viewer service may copy one chunk
        shutil.copy2(_lp(src), _lp(tmp))
        _replace(_lp(tmp), _lp(dst))

    def _companions(self, run_dir: str, path: str) -> List[str]:
        """The files a chunk cannot be read without: its folder's headers and, for a world chunk, the saved models
        those headers name; the body's gaps.jsonl (the steps its recorder dropped) and the run's args.json."""
        folder = os.path.dirname(os.path.abspath(path))
        out = sorted(glob.glob(os.path.join(folder, "header_*.json")))
        gaps = os.path.join(folder, "gaps.jsonl")
        if os.path.isfile(gaps):
            out.append(gaps)
        if os.path.basename(os.path.dirname(folder)) == "world":
            world = os.path.dirname(folder)
            for h in out[:]:
                if not h.endswith(".json"):
                    continue
                try:
                    with open(_lp(h), encoding="utf-8") as f:
                        model = json.load(f).get("model_file")
                except (OSError, ValueError):
                    model = None
                if model:
                    p = os.path.join(world, model)
                    if os.path.isfile(_lp(p)) and p not in out:
                        out.append(p)
        args = os.path.join(os.path.abspath(run_dir), "args.json")
        if os.path.isfile(args):
            out.append(args)
        return out

    def ensure(self, run_dir: str, path: str) -> Tuple[bool, str]:
        """(True, the archived path) when an archived copy of `path` with its size and SHA-256 exists or was just made,
        with the files it cannot be read without; (False, why) otherwise. Never raises."""
        why = self.available()
        if why:
            return False, why
        try:
            for comp in self._companions(run_dir, path):
                dst = self.target(run_dir, comp)
                if not self.same(comp, dst):
                    self._copy(comp, dst)
                    if not self.same(comp, dst):
                        # a companion still being appended to (gaps.jsonl) is copied again at the next pass
                        return False, "the archived copy of %s differs from the original after copying" % (
                            os.path.basename(comp))
            dst = self.target(run_dir, path)
            if self.same(path, dst):
                return True, dst
            self._copy(path, dst)
            if self.same(path, dst):
                return True, dst
            return False, "the archived copy of %s differs from the original after copying" % os.path.basename(path)
        except (OSError, ValueError) as ex:
            return False, "archiving %s failed: %s" % (os.path.basename(path), ex)

    def run_pass(self, run_dir: str, limit: Optional[int] = None) -> Dict[str, object]:
        """Copy every raw chunk of a run not yet archived (world chunks of every body, learner-lane chunks) with its
        companions. A file whose size and modification time equal the ones this archive recorded for it, and whose
        copy is present with that size, is skipped without reading it. Returns counts and the first failure."""
        why = self.available()
        report: Dict[str, object] = dict(run=os.path.basename(os.path.abspath(run_dir)), copied=0, skipped=0,
                                         failed=0, bytes=0, error=why)
        if why:
            return report
        man_path = os.path.join(self.run_folder(run_dir), MANIFEST)
        try:
            with open(_lp(man_path), encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, ValueError):
            manifest = {}
        obs = os.path.join(os.path.abspath(run_dir), "observer")
        files = sorted(glob.glob(os.path.join(obs, "world", "b*", "chunk_*.npz"))
                       + glob.glob(os.path.join(obs, "model", "chunk_*.npz")))
        files = [p for p in files if not p.endswith(".part.npz")]
        done = 0
        for p in files:
            if limit is not None and done >= limit:
                break
            rel = os.path.relpath(p, os.path.abspath(run_dir)).replace("\\", "/")
            try:
                st = os.stat(_lp(p))
            except OSError:
                continue                      # removed since the listing
            rec = manifest.get(rel)
            dst = self.target(run_dir, p)
            if rec and rec[0] == st.st_size and rec[1] == st.st_mtime_ns and os.path.isfile(_lp(dst)) \
                    and os.path.getsize(_lp(dst)) == st.st_size:
                report["skipped"] = int(report["skipped"]) + 1
                continue
            ok, info = self.ensure(run_dir, p)
            done += 1
            if ok:
                manifest[rel] = [st.st_size, st.st_mtime_ns, sha256(dst)]
                report["copied"] = int(report["copied"]) + 1
                report["bytes"] = int(report["bytes"]) + st.st_size
            else:
                report["failed"] = int(report["failed"]) + 1
                report["error"] = report["error"] or info
        if done:
            os.makedirs(os.path.dirname(_lp(man_path)), exist_ok=True)
            tmp = "%s.%d.part" % (man_path, os.getpid())
            with open(_lp(tmp), "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            _replace(_lp(tmp), _lp(man_path))
        report["checked_at"] = time.time()
        return report
