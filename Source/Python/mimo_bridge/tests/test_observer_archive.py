# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The observer archive (tools/observer/archive.py; the movement viewer of Oct 18, 2026, part 1): the digester's
housekeeping removes a raw chunk only after a copy with the original's size and SHA-256 is on the archive disk.

Each test fails on the code before the change, which removed digested chunks past 72 hours with no copy:
  * housekeeping with an archive copies each chunk, its headers, its saved model and the run's arguments, keeps file
    names and file times, and only then removes the original;
  * an archive whose disk is missing keeps every chunk and says why;
  * a stale copy with other bytes is replaced before the original goes;
  * a whole-run pass copies what is not yet archived and skips what it already copied.
"""
import json
import os
import time

import numpy as np

from mimo_bridge.tools.observer import archive as AR
from mimo_bridge.tools.observer import daemon as DM

OLD = time.time() - 4 * 24 * 3600          # four days: past the 72 hours of daemon.RAW_KEEP_S


def _run(tmp_path, name="H99_seg01"):
    """A sealed run: body 0 with two digested four-day-old world chunks, a header naming a saved model, a learner lane
    with one chunk, args.json."""
    run = tmp_path / "experiments" / name
    b0 = run / "observer" / "world" / "b0"
    b0.mkdir(parents=True)
    models = run / "observer" / "world" / "models"
    models.mkdir()
    (models / "abc.mjb.gz").write_bytes(b"model bytes")
    (b0 / "header_11_1000.json").write_text(json.dumps(dict(budget_bytes=10 ** 12, model_file="models/abc.mjb.gz")),
                                           encoding="utf-8")
    names = []
    for k in range(2):
        p = b0 / ("chunk_%09d_11_%05d.npz" % (1000 + 100 * k, k))
        np.savez(str(p), step=np.arange(1000 + 100 * k, 1100 + 100 * k))
        os.utime(p, (OLD, OLD))
        names.append(p.name)
    lane = run / "observer" / "model"
    lane.mkdir()
    (lane / "header_99_1000.json").write_text("{}", encoding="utf-8")
    lp = lane / "chunk_000001000_99_00000.npz"
    np.savez(str(lp), step=np.arange(1000, 1100))
    os.utime(lp, (OLD, OLD))
    (run / "args.json").write_text(json.dumps(dict(observer="record")), encoding="utf-8")
    dg = DM.RunDigest(str(run))
    dg.state["bodies"]["0"] = dict(done={n: dict(pid=11) for n in names}, final={"11": 10 ** 6})
    dg.state["sealed"] = time.time()
    return run, dg, names


def test_housekeeping_archives_each_chunk_before_removing_it(tmp_path):
    run, dg, names = _run(tmp_path)
    originals = {n: (run / "observer/world/b0" / n).read_bytes() for n in names}
    arc = AR.Archive(str(tmp_path / "archive"))
    rep = dg.housekeeping(archive=arc)
    assert rep["removed"] == 3 and rep["archive_kept"] == 0
    kept = tmp_path / "archive" / "H99_seg01"
    for n in names:
        copy = kept / "observer/world/b0" / n
        assert copy.read_bytes() == originals[n]
        assert abs(os.path.getmtime(copy) - OLD) < 2, "the copy keeps the chunk's file time"
        assert not (run / "observer/world/b0" / n).exists()
    assert (kept / "observer/world/b0/header_11_1000.json").is_file()
    assert (kept / "observer/world/models/abc.mjb.gz").read_bytes() == b"model bytes"
    assert (kept / "observer/model/chunk_000001000_99_00000.npz").is_file()
    assert (kept / "args.json").is_file()


def test_a_missing_archive_disk_keeps_every_chunk(tmp_path):
    run, dg, names = _run(tmp_path)
    arc = AR.Archive("Q:/no_such_disk/observer_raw" if os.name == "nt" else str(tmp_path / "missing" / "x" / "y"))
    if os.name != "nt":
        arc.available = lambda: "the archive disk is not available"
    rep = dg.housekeeping(archive=arc)
    assert rep["removed"] == 0 and rep["archive_kept"] == 3 and "not available" in rep["archive_why"]
    assert sorted(n for n in os.listdir(run / "observer/world/b0") if n.startswith("chunk_")) == names


def test_a_stale_copy_is_replaced_before_the_original_goes(tmp_path):
    run, dg, names = _run(tmp_path)
    arc = AR.Archive(str(tmp_path / "archive"))
    stale = tmp_path / "archive" / "H99_seg01" / "observer/world/b0" / names[0]
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"other bytes")
    original = (run / "observer/world/b0" / names[0]).read_bytes()
    dg.housekeeping(archive=arc)
    assert stale.read_bytes() == original


def test_without_an_archive_housekeeping_removes_as_before(tmp_path):
    run, dg, names = _run(tmp_path)
    rep = dg.housekeeping()
    assert rep["removed"] == 3 and not (tmp_path / "archive").exists()


def test_a_run_pass_copies_new_chunks_and_skips_copied_ones(tmp_path):
    run, dg, names = _run(tmp_path)
    arc = AR.Archive(str(tmp_path / "archive"))
    first = arc.run_pass(str(run))
    assert first["copied"] == 3 and first["failed"] == 0 and first["error"] is None
    again = arc.run_pass(str(run))
    assert again["copied"] == 0 and again["skipped"] == 3
    p = run / "observer/world/b0" / ("chunk_%09d_11_%05d.npz" % (1200, 2))
    np.savez(str(p), step=np.arange(1200, 1300))
    third = arc.run_pass(str(run))
    assert third["copied"] == 1 and third["skipped"] == 3
    manifest = json.loads((tmp_path / "archive" / "H99_seg01" / AR.MANIFEST).read_text(encoding="utf-8"))
    assert len(manifest) == 4 and all(len(v) == 3 for v in manifest.values())


def test_a_copy_waits_for_a_reader_of_the_archived_file(tmp_path):
    """A stale archived copy that a reader holds open (the movement viewer reads archived chunks) is replaced once the
    reader closes it, not left stale with a PermissionError (WinError 5 on Windows); the temporary name carries the
    process id, so the digester and the viewer service never write one temporary file."""
    import threading
    src, dst = tmp_path / "chunk_1.npz", tmp_path / "copy" / "chunk_1.npz"
    src.write_bytes(b"new bytes")
    dst.parent.mkdir()
    dst.write_bytes(b"old")
    reader = open(dst, "rb")
    threading.Timer(0.3, reader.close).start()
    AR.Archive._copy(str(src), str(dst))
    assert dst.read_bytes() == b"new bytes"
    assert not list(dst.parent.glob("*.part"))
