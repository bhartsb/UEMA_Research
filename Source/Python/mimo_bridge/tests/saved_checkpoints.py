# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Where a test finds a saved checkpoint: through mimo_bridge.storage_locations, never one hand-built path.

WHY (2026-09-26). Eight test files looked for DESIGN D3's baseline checkpoint 344,400 only at its C:
path and skipped when it was not there. The checkpoint archiver moved it to D: on Sept 23 at
5:22 PM PDT, and from then on those files reported skips instead of running their checkpoint proofs.

HOW. SavedCheckpoint(relative) looks where storage_locations.find_in_trees looks: an override
from an environment variable, this tree, the main working tree (a git worktree holds no Saved/
data), and where a move record says the path went. At import it only reads the move records (no
file is hashed while pytest collects). A test that needs the checkpoint calls require(), which
proves a moved copy by the SHA-256 of every file its move record lists, once per process.

A test may skip only when the checkpoint is nowhere, and the skip names every place searched. A
move record whose copy is missing or differs FAILS the test: a damaged archive is never "not on this
machine".
"""
import os
from pathlib import Path

import pytest

from mimo_bridge import storage_locations as SL

# DESIGN D3's baseline: H84 segment 31's checkpoint at step 344,400
H84_344400 = Path("Saved/mimo_bridge/experiments/H84_fresh_drivers_20260918_seg31_toy_and_push/"
                  "checkpoints/ckpt_step_344400")
# the overrides the test files already honoured before this lookup existed
H84_344400_ENV = ("UEMA_CKPT_344400", "UEMA_H84_CKPT")


class SavedCheckpoint:
    def __init__(self, relative, env=()):
        self.relative = Path(relative)
        self.also = [os.environ[name] for name in env if os.environ.get(name)]
        self.error = None
        self._proven = None
        try:
            self.path, self.looked = SL.find_in_trees(self.relative, also=self.also, verify=False)
        except SL.MovedFileError as exc:
            self.path, self.looked, self.error = None, [], str(exc)

    @property
    def nowhere(self) -> bool:
        return self.path is None and self.error is None

    @property
    def reason(self) -> str:
        if self.error:
            return "%s: a move record names it, but the copy fails its proof: %s" % (
                self.relative.name, self.error)
        return "%s is nowhere on this machine; looked at %s" % (self.relative.name,
                                                               "; ".join(self.looked))

    def mark(self):
        """Skip only when the checkpoint is nowhere. A damaged move is not skipped: require() fails."""
        return pytest.mark.skipif(self.nowhere, reason=self.reason)

    def require(self) -> Path:
        """The checkpoint's folder, proven: fails on a damaged move, skips only when nowhere."""
        if self.error:
            pytest.fail(self.reason)
        if self.path is None:
            pytest.skip(self.reason)
        if self._proven is None:
            try:
                found, _ = SL.find_in_trees(self.relative, also=self.also, verify=True)
            except SL.MovedFileError as exc:
                self.error = str(exc)
                pytest.fail(self.reason)
            if found is None or SL._key(found) != SL._key(self.path):
                pytest.fail("%s: found at %s while collecting, at %s when proven"
                            % (self.relative.name, self.path, found))
            self._proven = found
        return self._proven


def h84_344400() -> SavedCheckpoint:
    return SavedCheckpoint(H84_344400, env=H84_344400_ENV)
