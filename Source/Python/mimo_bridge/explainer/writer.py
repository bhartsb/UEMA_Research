# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Snapshot writer: atomic files, a self-healing index, bounded disk.

The trainer captures synchronously (the tensors mutate on the next step) but the
JSON encode + PNG payloads + file write run on ONE daemon worker with a depth-1
queue: if a write is still in flight when the next snapshot arrives, the new one is
DROPPED with a log line. A dropped snapshot costs a data point; a stalled step costs
every body its action deadline (the worker pipes feed N envs from this process).

Disk policy (the checkpoint rule's cousin - train_mimo keeps 48 snapshots ~1.8 GB
against ~27 GB free): keep the newest `keep_newest` outright; while the directory
exceeds `keep_mb`, delete every second file of the OLDER remainder (evenly thinning
history rather than truncating it). index.json is rebuilt from a directory scan on
every write, so deletions, crashes and torn writes self-heal.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional

log = logging.getLogger("mimo_bridge.explainer")

_STEP_RE = re.compile(r"^step_(\d+)\.json$")


class ExplainWriter:
    def __init__(self, run_dir: str, keep_mb: float = 500.0, keep_newest: int = 50):
        self.dir = os.path.join(os.path.abspath(run_dir), "explain")
        os.makedirs(self.dir, exist_ok=True)
        self.keep_mb = float(keep_mb)
        self.keep_newest = int(keep_newest)
        self._busy = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.dropped = 0
        self.written = 0

    # ------------------------------------------------------------ public
    def submit(self, snap: Dict[str, Any]) -> bool:
        """Queue one snapshot for writing. Returns False (and counts a drop) when the
        previous write is still in flight - never blocks the training step."""
        if self._busy.is_set():
            self.dropped += 1
            log.warning("explain snapshot at step %s dropped: previous write still "
                        "in flight (%d dropped so far)",
                        snap.get("header", {}).get("step"), self.dropped)
            return False
        self._busy.set()
        self._thread = threading.Thread(target=self._write, args=(snap,), daemon=True)
        self._thread.start()
        return True

    def flush(self, timeout: float = 30.0) -> None:
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout)

    # ------------------------------------------------------------ internals
    def _write(self, snap: Dict[str, Any]) -> None:
        try:
            step = int(snap["header"]["step"])
            path = os.path.join(self.dir, f"step_{step}.json")
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(snap, fh, separators=(",", ":"))
            os.replace(tmp, path)
            self.written += 1
            size_kb = os.path.getsize(path) / 1024.0
            if size_kb > 500:
                log.warning("explain snapshot step %d is %.0f KB (warn line 500 KB - "
                            "check the attention block's quantization)", step, size_kb)
            self._retention()
            self._rebuild_index()
        except Exception:
            log.exception("explain snapshot write failed (training unaffected)")
        finally:
            self._busy.clear()

    def _entries(self) -> List[Dict[str, Any]]:
        out = []
        try:
            names = os.listdir(self.dir)
        except OSError:
            return out
        for n in names:
            m = _STEP_RE.match(n)
            if not m:
                continue
            p = os.path.join(self.dir, n)
            try:
                st = os.stat(p)
            except OSError:
                continue
            out.append({"step": int(m.group(1)), "file": n,
                        "bytes": int(st.st_size), "mtime": st.st_mtime})
        out.sort(key=lambda e: e["step"])
        return out

    def _retention(self) -> None:
        entries = self._entries()
        total_mb = sum(e["bytes"] for e in entries) / 1e6
        guard = 0
        while total_mb > self.keep_mb and len(entries) > self.keep_newest and guard < 32:
            old = entries[:-self.keep_newest]
            victims = old[::2] if len(old) > 1 else old      # thin evenly, oldest half
            for e in victims:
                try:
                    os.remove(os.path.join(self.dir, e["file"]))
                except OSError:
                    pass
            entries = self._entries()
            total_mb = sum(e["bytes"] for e in entries) / 1e6
            guard += 1

    def _rebuild_index(self) -> None:
        idx = {"snapshots": self._entries(), "written": self.written,
               "dropped": self.dropped, "keep_mb": self.keep_mb}
        tmp = os.path.join(self.dir, "index.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(idx, fh, indent=1)
        os.replace(tmp, os.path.join(self.dir, "index.json"))
