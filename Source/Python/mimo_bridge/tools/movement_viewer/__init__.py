# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The movement viewer: requested versus actual movement for every saved command, over dates.

The operator's request of October 18, 2026 (docs/H84_MOVEMENT_VIEWER_PLAN_2026-10-18.md; the specification
docs/H84_DASHBOARD_MOVEMENT_VIEWER_SPEC_2026-10-17.html). Read-only on every recording: it reads the observer's saved
chunks (mimo_bridge/observer_record.py) and writes only under its own folders.

  sources.py   every saved recording (training runs, tests, archived copies, checkpoint probes), each with an id
  commands.py  one recording body's commands and their measures (start, end and after-the-next-command errors)
  index.py     the stored per-command measures and the dashboard's chart series
  render.py    the pictures of one command: left hand, right hand, whole body; start, requested, actual
  probes.py    the scheduled checkpoint probes' trials as commands
  auto.py      the automatic picks (per body and hour, events) and their pictures
  service.py   the one process that draws, indexes and archives (started by the dashboard)
  api.py       what the dashboard's /api/movement/ routes return

A command is the 26 body handles x 9 numbers sent to the body (target position in cm in world coordinates, target
orientation as a quaternion x y z w, radius, strength 0 to 1), the handle mask and the 44 finger joint target angles,
as the observer saved them (order_motor, order_mask, order_hand_goal).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
SAVED = REPO / "Saved" / "mimo_bridge"
HOME = SAVED / "movement_viewer"                  # small files: sources, index, status, requests
INDEX_DIR = HOME / "index"
REQUESTS_DIR = HOME / "requests"
STATUS_FILE = HOME / "service_status.json"
CONTROL_FILE = HOME / "service_control.json"       # {"paused": true} holds the service's background work
LOCK_FILE = HOME / "service.lock"
# the pictures: on the large disk when it is present (each drawing is about 1 MB), else beside the index
PICTURES_ENV = "UEMA_MOVEMENT_PICTURES"
PICTURES_DEFAULT = "D:/UEMA_Research/external/D/UEMA_movement_viewer/pictures"


def pictures_dir() -> Path:
    named = os.environ.get(PICTURES_ENV)
    if named:
        return Path(named)
    drive = os.path.splitdrive(PICTURES_DEFAULT)[0]
    if drive and os.path.isdir(drive + os.sep):
        return Path(PICTURES_DEFAULT)
    return HOME / "pictures"


def part_path(dst) -> str:
    """A temporary name beside `dst` that no other process writes: the dashboard and the viewer service can write the
    same file (a probe cache, an archived chunk), and one shared name would mix their bytes."""
    return "%s.%d.part" % (dst, os.getpid())


REPLACE_TRIES = 100
REPLACE_WAIT_S = 0.02


def replace_file(tmp, dst) -> None:
    """os.replace(tmp, dst), retried for up to REPLACE_TRIES x REPLACE_WAIT_S = 2 s. On Windows a replace fails with
    PermissionError (WinError 5) while another process has `dst` open, as when the dashboard reads the status file at
    that instant: the viewer service ended that way on Oct 18, 2026, 6:35 PM PDT. A reader holds a file for the time of
    one read, so a later try succeeds; after the last try the error is raised."""
    for i in range(REPLACE_TRIES):
        try:
            os.replace(str(tmp), str(dst))
            return
        except PermissionError:
            if i == REPLACE_TRIES - 1:
                raise
            time.sleep(REPLACE_WAIT_S)
