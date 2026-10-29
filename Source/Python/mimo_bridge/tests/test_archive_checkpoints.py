# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Moving a lineage's older checkpoints to the archive disk, verified, without losing any.

Everything here runs in temporary folders standing in for C: and D:. No real checkpoint is
read or written.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from mimo_bridge.tools import archive_checkpoints as AC

LINEAGE = "H99_test_lineage"


def _make(saved: Path, run: str, step: int, companion=True, complete=True, age_s=7200.0):
    ck = saved / "experiments" / run / "checkpoints" / ("ckpt_step_%d" % step)
    ck.mkdir(parents=True)
    (ck / "model.safetensors").write_bytes(os.urandom(2048))
    (ck / "runtime_state.pt").write_bytes(os.urandom(4096))
    if complete:
        (ck / "lineage.json").write_text("{}", encoding="utf-8")
    comp = None
    if companion:
        comp = (saved / "caregiver_records" / LINEAGE / "checkpoint_state" / run /
                ("ckpt_step_%d_abc.pt.gz" % step))
        comp.parent.mkdir(parents=True, exist_ok=True)
        comp.write_bytes(os.urandom(3000))
        digest = hashlib.sha256(comp.read_bytes()).hexdigest()
        (ck / "episode_state.json").write_text(json.dumps(
            {"storage_path": str(comp), "sha256": digest, "step": step}), encoding="utf-8")
    old = time.time() - age_s
    for p in list(ck.rglob("*")) + [ck]:
        os.utime(p, (old, old))
    return ck, comp


def _continuation(saved: Path, active_run: str, resume: Path):
    (saved / "training_continuation.json").write_text(json.dumps({
        "active_run": str(saved / "experiments" / active_run),
        "command": ["python", "--resume", str(resume)]}), encoding="utf-8")


@pytest.fixture
def world(tmp_path):
    saved, archive = tmp_path / "C" / "Saved", tmp_path / "D" / "archive"
    made = {}
    for run, step in ((LINEAGE, 100), (LINEAGE, 200), (LINEAGE + "_seg2", 300),
                      (LINEAGE + "_seg2", 400), (LINEAGE + "_seg3", 500),
                      (LINEAGE + "_seg3", 600), (LINEAGE + "_seg4", 700)):
        made[step] = _make(saved, run, step)
    _continuation(saved, LINEAGE + "_seg4", made[400][0])
    return saved, archive, made


def test_the_latest_three_the_resume_source_and_the_held_runs_newest_stay(world):
    saved, archive, made = world
    cont = json.loads((saved / "training_continuation.json").read_text(encoding="utf-8"))
    keep = AC.protected(AC.inventory(LINEAGE, saved), cont)
    kept_steps = sorted(AC._step(p) for p in keep)
    # latest three: 500, 600, 700; the resume source: 400; the held run's newest: 700
    assert kept_steps == [400, 500, 600, 700], kept_steps
    assert any("resume source" in w for w in keep[str(made[400][0])])


def test_a_move_verifies_every_file_and_removes_the_originals_only_after(world):
    saved, archive, made = world
    before = {s: AC.folder_hashes(made[s][0]) for s in (100, 200, 300)}
    r = AC.run_once(LINEAGE, target_free_gb=0.0, dry_run=False, saved=saved, archive=archive,
                    move_all=True, log=lambda m: None)
    assert r["moved"] == 3, r
    for s, run in ((100, LINEAGE), (200, LINEAGE), (300, LINEAGE + "_seg2")):
        assert not made[s][0].exists() and not made[s][1].exists(), "an original was left behind"
        dst = archive / run / "checkpoints" / ("ckpt_step_%d" % s)
        assert AC.folder_hashes(dst) == before[s], "an archived file differs from its original"
        comp = archive / "checkpoint_state" / run / made[s][1].name
        assert comp.exists()
    rows = [json.loads(l) for l in (archive / "move_manifest.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert [r["step"] for r in rows] == [100, 200, 300]
    assert set(rows[0]) >= {"event", "step", "run", "source", "destination", "companion_source",
                            "companion_destination", "companion_sha256", "files", "bytes",
                            "time"}, "the manifest line is not in the archive's existing format"
    # the protected four are all still on C:
    for s in (400, 500, 600, 700):
        assert made[s][0].exists() and made[s][1].exists()


def test_a_companion_that_does_not_match_the_hash_its_checkpoint_recorded_stays_put(world):
    saved, archive, made = world
    made[100][1].write_bytes(b"not what was written")
    r = AC.run_once(LINEAGE, target_free_gb=0.0, dry_run=False, saved=saved, archive=archive,
                    move_all=True, log=lambda m: None)
    assert r["moved"] == 0, "a companion that failed its own recorded hash was moved"
    assert made[100][0].exists() and made[100][1].exists()


def test_a_copy_that_arrives_different_leaves_c_untouched(world, monkeypatch):
    saved, archive, made = world
    real = AC.folder_hashes
    calls = {"n": 0}

    def lying(folder):
        out = real(folder)
        if str(archive) in str(folder):         # the copy on the archive disk reads wrong
            calls["n"] += 1
            out = {k: "0" * 64 for k in out}
        return out
    monkeypatch.setattr(AC, "folder_hashes", lying)
    r = AC.run_once(LINEAGE, target_free_gb=0.0, dry_run=False, saved=saved, archive=archive,
                    move_all=True, log=lambda m: None)
    assert calls["n"] >= 1 and r["moved"] == 0
    assert made[100][0].exists() and made[100][1].exists()
    assert not (archive / LINEAGE / "checkpoints" / "ckpt_step_100").exists(), (
        "the partial copy was left on the archive disk")


def test_a_companion_past_260_characters_moves_and_is_not_called_missing(world):
    """A resume folder's companion path passes Windows' 260 characters (262-268 on H84,
    2026-09-24); the archiver must find, copy and remove it through the extended-length form."""
    from mimo_bridge.storage_locations import long_path
    saved, archive, made = world
    ck, comp = made[100]
    deep = comp.parent / ("r" * 100) / ("s" * 100)
    far = deep / comp.name
    assert len(str(far)) > 260
    os.makedirs(long_path(deep), exist_ok=True)
    with open(long_path(far), "wb") as f:
        f.write(comp.read_bytes())
    comp.unlink()
    receipt = json.loads((ck / "episode_state.json").read_text(encoding="utf-8"))
    receipt["storage_path"] = str(far)
    (ck / "episode_state.json").write_text(json.dumps(receipt), encoding="utf-8")
    old = time.time() - 7200.0
    os.utime(ck / "episode_state.json", (old, old))
    if os.name == "nt":
        assert not far.exists(), "the plain form reports the file missing: the defect this pins"
    r = AC.run_once(LINEAGE, target_free_gb=0.0, dry_run=False, saved=saved, archive=archive,
                    move_all=True, log=lambda m: None)
    assert r["moved"] == 3, r
    assert not os.path.exists(long_path(far)), "the moved companion was left on C:"
    assert (archive / "checkpoint_state" / deep.name / far.name).exists()


def test_a_dry_run_moves_nothing(world):
    saved, archive, made = world
    r = AC.run_once(LINEAGE, target_free_gb=0.0, dry_run=True, saved=saved, archive=archive,
                    move_all=True, log=lambda m: None)
    assert r["moved"] == 0 and r["eligible"] == 3
    assert all(made[s][0].exists() for s in made)


def test_a_checkpoint_written_in_the_last_half_hour_is_never_touched(tmp_path):
    saved, archive = tmp_path / "C" / "Saved", tmp_path / "D" / "archive"
    for step in (10, 20, 30, 40):
        _make(saved, LINEAGE, step)
    fresh, _ = _make(saved, LINEAGE + "_x", 5, age_s=60.0)     # old step, written a minute ago
    _continuation(saved, LINEAGE, saved / "nowhere")
    AC.run_once(LINEAGE, target_free_gb=0.0, dry_run=False, saved=saved, archive=archive,
                move_all=True, log=lambda m: None)
    assert fresh.exists(), "a checkpoint still being published was moved"


def test_an_incomplete_checkpoint_is_left_where_it_is(tmp_path):
    saved, archive = tmp_path / "C" / "Saved", tmp_path / "D" / "archive"
    for step in (10, 20, 30, 40):
        _make(saved, LINEAGE, step)
    broken, _ = _make(saved, LINEAGE + "_x", 5, complete=False)
    _continuation(saved, LINEAGE, saved / "nowhere")
    AC.run_once(LINEAGE, target_free_gb=0.0, dry_run=False, saved=saved, archive=archive,
                move_all=True, log=lambda m: None)
    assert broken.exists()


def test_it_stops_once_the_free_space_target_is_met(world, monkeypatch):
    saved, archive, made = world
    monkeypatch.setattr(AC, "free_bytes", lambda p: 10 ** 15)   # plenty of room already
    r = AC.run_once(LINEAGE, target_free_gb=60.0, dry_run=False, saved=saved, archive=archive,
                    log=lambda m: None)
    assert r["moved"] == 0


def _sibling(saved: Path, original: Path, suffix: str, age_s=7200.0):
    """A checkpoint folder copied by hand from another, naming the SAME companion file."""
    import shutil
    twin = original.parent / (original.name + suffix)
    shutil.copytree(original, twin)
    old = time.time() - age_s
    for p in list(twin.rglob("*")) + [twin]:
        os.utime(p, (old, old))
    return twin


def test_two_checkpoints_sharing_one_companion_both_move_and_the_companion_moves_once(world):
    """Found on the real lineage: seg10_toys holds ckpt_step_220675 and a hand-made copy,
    ckpt_step_220675_entity_self_carried, that names the same companion. The first cut moved
    the companion with the first folder and then stopped on the second."""
    saved, archive, made = world
    twin = _sibling(saved, made[100][0], "_carried")
    r = AC.run_once(LINEAGE, target_free_gb=0.0, dry_run=False, saved=saved, archive=archive,
                    move_all=True, log=lambda m: None)
    assert r["moved"] == 4, r                      # 100, its twin, 200, 300
    assert not twin.exists() and not made[100][0].exists() and not made[100][1].exists()
    comp = archive / "checkpoint_state" / LINEAGE / made[100][1].name
    assert comp.exists()


def test_a_companion_a_protected_checkpoint_still_names_is_never_removed_from_c(world):
    """The hazard the shared case hides: if a checkpoint that must stay on C: shares its
    companion with one that moves, removing the companion would leave the protected one
    unable to resume. The companion is copied and kept on C:."""
    saved, archive, made = world
    # make the resume source (400, protected) share the companion of 300 (eligible)
    shared = made[300][1]
    rec = json.loads((made[400][0] / "episode_state.json").read_text(encoding="utf-8"))
    rec["storage_path"] = str(shared)
    rec["sha256"] = hashlib.sha256(shared.read_bytes()).hexdigest()
    (made[400][0] / "episode_state.json").write_text(json.dumps(rec), encoding="utf-8")
    old = time.time() - 7200
    os.utime(made[400][0] / "episode_state.json", (old, old))
    AC.run_once(LINEAGE, target_free_gb=0.0, dry_run=False, saved=saved, archive=archive,
                move_all=True, log=lambda m: None)
    assert not made[300][0].exists(), "the eligible checkpoint did not move"
    assert shared.exists(), "a companion the resume source still names was removed from C:"
    assert (archive / "checkpoint_state" / (LINEAGE + "_seg2") / shared.name).exists()


def test_a_checkpoint_whose_companion_is_nowhere_is_left_and_nothing_else_breaks(world):
    saved, archive, made = world
    made[100][1].unlink()                          # gone from C:, never archived
    r = AC.run_once(LINEAGE, target_free_gb=0.0, dry_run=False, saved=saved, archive=archive,
                    move_all=True, log=lambda m: None)
    assert made[100][0].exists(), "a checkpoint whose companion is lost was moved anyway"
    assert r["moved"] == 0


# ------------------------------------------------------------ the caregiver's cold recordings
def _tape(folder: Path, name: str, chunks, age_s: float):
    """A tape manifest naming these chunk files (created if missing), all aged age_s."""
    folder.mkdir(parents=True, exist_ok=True)
    spans = []
    for c in chunks:
        p = folder / c
        if not p.exists():
            p.write_bytes(b'{"time":0}\n')
        spans.append(dict(file=c, bytes=p.stat().st_size, count=1))
    m = folder / (name + ".tape.json")
    m.write_text(json.dumps(dict(format="caregiver_tape_v1", count=len(spans), dropped=0,
                                 spans=spans)), encoding="utf-8")
    old = time.time() - age_s
    for p in [m] + [folder / c for c in chunks]:
        if os.stat(p).st_mtime > old:
            os.utime(p, (old, old))
    return m


def test_closed_chunks_move_verified_and_a_live_tapes_open_chunk_stays(tmp_path):
    saved, archive = tmp_path / "C" / "Saved", tmp_path / "D" / "tapes"
    rec = saved / "caregiver_records" / LINEAGE
    DAY = 86400.0
    _tape(rec, "old", ["a.jsonl", "shared.jsonl"], 3 * DAY)      # a process that has ended
    # the live episode: its history back to the start, and the chunk it is writing now,
    # which is also old - a quiet caregiver - but must stay because it is the last one
    _tape(rec, "live", ["shared.jsonl", "b.jsonl", "writing.jsonl"], 3 * DAY)
    # rewritten a minute ago. (Until 2026-09-22 this read "just now"; a manifest written in the
    # last 10 s is no longer read at all - see the test below and MANIFEST_QUIET_S.)
    os.utime(rec / "live.tape.json", (time.time() - 60.0,) * 2)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in rec.glob("*.jsonl")}
    r = AC.archive_tapes(LINEAGE, saved, archive, log=lambda m: None)
    moved = {p.name for p in (archive / LINEAGE).iterdir()}
    assert moved == {"a.jsonl", "shared.jsonl", "b.jsonl", "old.tape.json"}
    assert {p.name for p in rec.iterdir()} == {"writing.jsonl", "live.tape.json"}
    for name in ("a.jsonl", "shared.jsonl", "b.jsonl"):
        assert hashlib.sha256((archive / LINEAGE / name).read_bytes()).hexdigest() == before[name]
    rows = [json.loads(x) for x in (archive / "tape_move_manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 4 == r["moved_files"]


def test_a_recently_written_chunk_stays_and_so_does_the_manifest_naming_it(tmp_path):
    saved, archive = tmp_path / "C" / "Saved", tmp_path / "D" / "tapes"
    rec = saved / "caregiver_records" / LINEAGE
    _tape(rec, "ended", ["x.jsonl", "y.jsonl"], 3 * 86400.0)
    os.utime(rec / "y.jsonl", None)
    AC.archive_tapes(LINEAGE, saved, archive, log=lambda m: None)
    assert not (rec / "x.jsonl").exists() and (rec / "y.jsonl").exists()
    assert (rec / "ended.tape.json").exists()          # it still names a chunk left on C:
    os.utime(rec / "y.jsonl", (time.time() - 3 * 86400.0,) * 2)
    AC.archive_tapes(LINEAGE, saved, archive, log=lambda m: None)
    assert not any(rec.iterdir())                      # x already there, y and the manifest now


def test_an_unreadable_manifest_moves_nothing(tmp_path):
    saved, archive = tmp_path / "C" / "Saved", tmp_path / "D" / "tapes"
    rec = saved / "caregiver_records" / LINEAGE
    _tape(rec, "gone", ["c.jsonl"], 3 * 86400.0)
    (rec / "broken.tape.json").write_text("{not json", encoding="utf-8")
    # corrupt for a minute, not caught mid-write: a manifest written in the last 10 s is not
    # read at all (test_a_manifest_being_written_is_not_read_and_its_newest_chunk_stays)
    os.utime(rec / "broken.tape.json", (time.time() - 60.0,) * 2)
    assert AC.archive_tapes(LINEAGE, saved, archive, log=lambda m: None)["moved_files"] == 0
    assert (rec / "c.jsonl").exists()


def test_a_copy_that_arrives_different_leaves_the_tape_on_c(tmp_path, monkeypatch):
    saved, archive = tmp_path / "C" / "Saved", tmp_path / "D" / "tapes"
    rec = saved / "caregiver_records" / LINEAGE
    _tape(rec, "gone", ["c.jsonl"], 3 * 86400.0)
    real = AC.shutil.copy2

    def corrupt(src, dst, *a, **k):
        real(src, dst, *a, **k)
        Path(dst).write_bytes(b"different")
    monkeypatch.setattr(AC.shutil, "copy2", corrupt)
    assert AC.archive_tapes(LINEAGE, saved, archive, log=lambda m: None)["moved_files"] == 0
    assert (rec / "c.jsonl").exists() and not (archive / LINEAGE / "c.jsonl").exists()


def test_a_manifest_being_written_is_not_read_and_its_newest_chunk_stays(tmp_path):
    """2026-09-22 11:46:25: this pass read a live manifest while the trainer's recorder was
    replacing it, the replace failed with WinError 5 and 143 caregiver events were dropped. A
    manifest written in the last 10 s is now left unread. The recorder writes its newest chunk
    in the same flush, so that chunk is seconds old and stays by the 6-hour idle rule; the
    older, closed chunks of every other tape still move."""
    saved, archive = tmp_path / "C" / "Saved", tmp_path / "D" / "tapes"
    rec = saved / "caregiver_records" / LINEAGE
    DAY = 86400.0
    _tape(rec, "ended", ["a.jsonl"], 3 * DAY)
    _tape(rec, "live", ["b.jsonl", "writing.jsonl"], 3 * DAY)
    # the recorder's flush: append to the newest chunk, then replace the manifest - both now.
    # The manifest's bytes here are not even valid JSON, as if caught mid-write: if the pass
    # read it, it would stop and move nothing.
    (rec / "live.tape.json").write_text('{"format": "caregiver_tape_v1", "spa', encoding="utf-8")
    with open(rec / "writing.jsonl", "ab") as fh:
        fh.write(b'{"time":1}\n')
    lines = []
    r = AC.archive_tapes(LINEAGE, saved, archive, log=lines.append)
    assert (rec / "writing.jsonl").exists(), "the chunk being written left C:"
    assert (rec / "live.tape.json").exists()
    assert not (rec / "a.jsonl").exists() and not (rec / "ended.tape.json").exists()
    assert r["moved_files"] >= 2
    assert any("not read this pass" in m for m in lines), lines


def test_every_move_record_is_mirrored_on_c_and_the_resolver_reads_it(world):
    """The D: manifest is the archive's own record; the same rows on C: survive the D: disk
    being absent, and storage_locations finds a moved checkpoint through them alone."""
    from mimo_bridge import storage_locations as SL
    saved, archive, made = world
    AC.run_once(LINEAGE, target_free_gb=0.0, dry_run=False, saved=saved, archive=archive,
                move_all=True, log=lambda m: None)
    rec = saved / "caregiver_records" / LINEAGE
    _tape(rec, "gone", ["c.jsonl"], 3 * 86400.0)
    AC.archive_tapes(LINEAGE, saved, archive / "tapes", log=lambda m: None)
    d_rows = [json.loads(l) for l in (archive / "move_manifest.jsonl").read_text().splitlines()]
    d_rows += [json.loads(l) for l in (archive / "tapes" / "tape_move_manifest.jsonl")
               .read_text().splitlines()]
    c_rows = [json.loads(l) for l in (saved / "storage_moves.jsonl").read_text().splitlines()]
    assert len(c_rows) == len(d_rows) == 5                    # 3 checkpoints, 1 chunk, 1 manifest
    for c, d in zip(c_rows, d_rows):
        assert {k: v for k, v in c.items() if k != "mirror_of"} == d
        assert c["mirror_of"].endswith(("move_manifest.jsonl", "tape_move_manifest.jsonl"))
    mirror_only = [saved / "storage_moves.jsonl"]
    assert SL.locate(made[100][0], manifests=mirror_only) == (
        archive / LINEAGE / "checkpoints" / "ckpt_step_100")
    assert SL.locate(made[100][1], manifests=mirror_only) == (
        archive / "checkpoint_state" / LINEAGE / made[100][1].name)
    assert SL.locate(rec / "c.jsonl", manifests=mirror_only) == archive / "tapes" / LINEAGE / "c.jsonl"


def test_a_failed_mirror_write_does_not_stop_the_move(world, monkeypatch):
    saved, archive, made = world
    (saved / "storage_moves.jsonl").mkdir(parents=True)        # a folder where the file goes
    lines = []
    r = AC.run_once(LINEAGE, target_free_gb=0.0, dry_run=False, saved=saved, archive=archive,
                    move_all=True, log=lines.append)
    assert r["moved"] == 3
    assert any("could not be written" in m for m in lines)
    assert len((archive / "move_manifest.jsonl").read_text().splitlines()) == 3


def test_probe_and_proof_folders_are_not_the_lineage(world):
    """explain_probe writes experiments/<run>_explainprobe_<time> with an end checkpoint, and
    deploy proofs write _deploy_proof_ folders. At the newest steps they would take places
    among the latest three that the real checkpoints keep on C:, and be archived as history."""
    saved, archive, made = world
    probe, _ = _make(saved, LINEAGE + "_seg4_explainprobe_20260922-190000", 9000)
    proof, _ = _make(saved, LINEAGE + "_deploy_proof_seg5", 9500)
    gof = saved / "gof_probes" / "run" / "ckpt"            # outside experiments/: never globbed
    gof.mkdir(parents=True)
    items = AC.inventory(LINEAGE, saved)
    assert {i["step"] for i in items} == {100, 200, 300, 400, 500, 600, 700}
    cont = json.loads((saved / "training_continuation.json").read_text(encoding="utf-8"))
    kept = sorted(AC._step(p) for p in AC.protected(items, cont))
    assert kept == [400, 500, 600, 700]
    AC.run_once(LINEAGE, target_free_gb=0.0, dry_run=False, saved=saved, archive=archive,
                move_all=True, log=lambda m: None)
    assert probe.exists() and proof.exists(), "a probe or proof checkpoint was archived"


def test_a_replay_reads_a_moved_tape_from_its_old_path(tmp_path, monkeypatch):
    from mimo_bridge import event_tape as ET
    saved, archive = tmp_path / "C" / "Saved", tmp_path / "D" / "tapes"
    rec = saved / "caregiver_records" / LINEAGE
    tape = ET.EventTape(rec, tmp_path / "status.json", flush_bytes=1)
    for t in range(5):
        tape.append({"time": float(t)})
    tape.close()
    old = time.time() - 3 * 86400.0
    for p in rec.iterdir():
        os.utime(p, (old, old))
    monkeypatch.setattr(ET, "TAPE_ARCHIVE", archive)
    AC.archive_tapes(LINEAGE, saved, archive, log=lambda m: None)
    assert not any(rec.iterdir())
    replay = ET.TapeReplay(tape.manifest)                  # the C: path it was written at
    assert [replay[i]["time"] for i in range(len(replay))] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [e["time"] for e in tape] == [0.0, 1.0, 2.0, 3.0, 4.0]   # the restored-tape path


# ------------------------------------------------------------------ one archiver at a time
def test_an_archiver_is_recognised_by_its_command_and_a_listing_is_not():
    assert AC.is_archiver_command(["python", "-u", "-m", "mimo_bridge.tools.archive_checkpoints",
                                   "--all", "--watch", "--interval", "900"])
    assert AC.is_archiver_command(["python", "C:/repo/tools/archive_checkpoints.py", "--all"])
    assert not AC.is_archiver_command(["python", "-m", "mimo_bridge.tools.archive_checkpoints",
                                       "--dry-run"])
    assert not AC.is_archiver_command(["python", "-m", "mimo_bridge.tools.training_continuation"])


def test_the_venv_launcher_and_its_child_count_as_one_archiver(monkeypatch):
    from types import SimpleNamespace
    import psutil
    cmd = ["python", "-u", "-m", "mimo_bridge.tools.archive_checkpoints", "--all", "--watch"]
    procs = [SimpleNamespace(info=dict(pid=20808, ppid=11180, name="python.exe", cmdline=cmd)),
             SimpleNamespace(info=dict(pid=36900, ppid=20808, name="python.exe", cmdline=cmd)),
             SimpleNamespace(info=dict(pid=77, ppid=1, name="python.exe",
                                       cmdline=cmd[:4] + ["--dry-run"])),
             SimpleNamespace(info=dict(pid=78, ppid=1, name="notepad.exe", cmdline=cmd))]
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: iter(procs))
    assert AC.archiver_pids() == [20808]
    assert AC.archiver_pids(exclude={20808, 36900}) == []


def test_a_second_archiver_refuses_to_start(tmp_path, monkeypatch):
    def no_pass(*a, **k):
        raise AssertionError("a refused archiver ran a pass")
    monkeypatch.setattr(AC, "run_once", no_pass)
    lock = tmp_path / "archiver.lock"
    # another archiver found by its command line (one started on code older than the lock)
    monkeypatch.setattr(AC, "archiver_pids", lambda exclude=(): [20808])
    assert AC.main(["--lock", str(lock), "--report", str(tmp_path / "s.json")]) == 2
    # or one holding the lock
    monkeypatch.setattr(AC, "archiver_pids", lambda exclude=(): [])
    from mimo_bridge.tools.training_continuation import OwnerLock
    held = OwnerLock(lock)
    try:
        assert AC.main(["--lock", str(lock), "--report", str(tmp_path / "s.json")]) == 2
    finally:
        held.close()
    assert not (tmp_path / "s.json").exists()


def test_a_lone_archiver_runs_its_pass_holds_the_lock_and_reports(tmp_path, monkeypatch):
    from mimo_bridge.tools.training_continuation import OwnerLock
    lock = tmp_path / "archiver.lock"
    seen = {}

    def one_pass(*a, **k):
        with pytest.raises(OSError):             # held while the pass runs
            OwnerLock(lock)
        seen["ran"] = True
        return dict(moved=0, eligible=0, free_gb=1.0)
    monkeypatch.setattr(AC, "archiver_pids", lambda exclude=(): [])
    monkeypatch.setattr(AC, "run_once", one_pass)
    monkeypatch.setattr(AC, "archive_tapes", lambda *a, **k: dict(moved_files=0, moved_bytes=0))
    report = tmp_path / "status.json"
    assert AC.main(["--lock", str(lock), "--report", str(report)]) == 0 and seen["ran"]
    status = json.loads(report.read_text(encoding="utf-8"))
    assert status["dry_run"] is False and status["interval_s"] == 900.0 and "time" in status
    OwnerLock(lock).close()                      # released on exit


def test_a_dry_run_writes_no_heartbeat_unless_asked(tmp_path, monkeypatch):
    monkeypatch.setattr(AC, "SAVED", tmp_path)
    monkeypatch.setattr(AC, "run_once", lambda *a, **k: dict(moved=0, eligible=0, free_gb=1.0))
    monkeypatch.setattr(AC, "archive_tapes", lambda *a, **k: dict(moved_files=0, moved_bytes=0))
    assert AC.main(["--dry-run"]) == 0
    assert not (tmp_path / AC.STATUS_NAME).exists()


# ------------------------------------------------------------ the keep list (finding 4)
def _keep(saved: Path, *paths, reason="DESIGN D3's baseline"):
    (saved / AC.KEEP_NAME).write_text(json.dumps(
        {"keep": [dict(path=str(p), reason=reason) for p in paths]}), encoding="utf-8")


def test_a_checkpoint_the_keep_list_names_stays_with_its_reason(world):
    """Review ops-storage finding 4: the build's proofs load ckpt_step_344400 by its C: path and
    skip when it is gone. A checkpoint the keep list names stays on C:, whatever its age."""
    saved, archive, made = world
    _keep(saved, made[100][0])
    lines = []
    r = AC.run_once(LINEAGE, target_free_gb=0.0, dry_run=False, saved=saved, archive=archive,
                    move_all=True, log=lines.append)
    assert made[100][0].exists() and made[100][1].exists(), "a kept checkpoint was moved"
    assert r["moved"] == 2                                   # 200 and 300 still move
    assert any("named in archive_keep.json: DESIGN D3's baseline" in m for m in lines), lines


def test_the_keep_list_is_matched_whatever_the_spelling_of_the_path(world):
    saved, archive, made = world
    spelled = str(made[200][0]).upper() if os.name == "nt" else str(made[200][0])
    items = AC.inventory(LINEAGE, saved)
    kept = AC.protected(items, {}, keep_latest=0, keep_list=[dict(path=spelled, reason="x")])
    assert list(kept) == [str(made[200][0])]
    # the control: without the list the same checkpoint may move
    assert str(made[200][0]) not in AC.protected(items, {}, keep_latest=0, keep_list=[])


def test_an_unreadable_keep_list_moves_nothing_and_says_why(world):
    saved, archive, made = world
    (saved / AC.KEEP_NAME).write_text('{"keep": [{"path": ', encoding="utf-8")
    r = AC.run_once(LINEAGE, target_free_gb=0.0, dry_run=False, saved=saved, archive=archive,
                    move_all=True, log=lambda m: None)
    assert r["moved"] == 0 and "keep list" in r["error"]
    assert all(made[s][0].exists() for s in made)


def test_the_real_keep_list_names_design_d3s_baseline_with_a_reason():
    """The file this build wrote; if it is removed or loses the baseline, this says so."""
    rows = AC.read_keep_list(AC.SAVED)
    named = [r for r in rows if Path(r["path"]).name == "ckpt_step_344400"
             and "H84_fresh_drivers_20260918_seg31_toy_and_push" in r["path"]]
    assert len(named) == 1 and "D3" in named[0]["reason"]


# ------------------------------------------------------------ a failed pass is in the status
def test_a_failed_tape_pass_is_written_into_the_status(tmp_path, monkeypatch):
    """Review ops-storage finding 2: a pass that fails still writes a fresh status, so the
    failure must be IN it for the hourly watcher to quote."""
    monkeypatch.setattr(AC, "archiver_pids", lambda exclude=(): [])
    monkeypatch.setattr(AC, "run_once", lambda *a, **k: dict(moved=0, eligible=0, free_gb=1.0))

    def fails(*a, **k):
        raise OSError("[WinError 3] The system cannot find the path specified")
    monkeypatch.setattr(AC, "archive_tapes", fails)
    report = tmp_path / "status.json"
    assert AC.main(["--lock", str(tmp_path / "l.lock"), "--report", str(report)]) == 0
    st = json.loads(report.read_text(encoding="utf-8"))
    assert "WinError 3" in st["tapes"]["error"]
    assert st["lineage"] == "H84_fresh_drivers_20260918" and st["watch"] is False


def test_an_unwritable_archive_is_an_error_in_the_pass_result(world):
    saved, archive, made = world
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text("a file where the archive folder goes", encoding="utf-8")
    r = AC.run_once(LINEAGE, target_free_gb=0.0, dry_run=False, saved=saved, archive=archive,
                    move_all=True, log=lambda m: None)
    assert r["moved"] == 0 and r.get("error")
    assert all(made[s][0].exists() for s in made)


# ------------------------------------------------------------ backfilling the C: mirror (finding 7)
def test_the_backfill_copies_every_d_row_once_with_mirror_of(tmp_path):
    saved, archive, tapes = tmp_path / "C", tmp_path / "D" / "archive", tmp_path / "D" / "tapes"
    saved.mkdir(parents=True)
    rows = [dict(event="moved", step=s, source="C:/x/%d" % s, destination="D:/x/%d" % s,
                 files={"a": "0" * 64}, bytes=1, time=float(s)) for s in (1, 2, 3)]
    for r in rows:
        from mimo_bridge.storage_locations import append_row
        append_row(archive / "move_manifest.jsonl", r)
    tape_row = dict(event="moved", source="C:/t/c.jsonl", destination="D:/t/c.jsonl",
                    sha256="1" * 64, bytes=5, time=9.0)
    append_row(tapes / "tape_move_manifest.jsonl", tape_row)
    # row 3 was already mirrored at its move (the new archiver's record_move)
    append_row(saved / AC.MIRROR_NAME, dict(rows[2], mirror_of=str(archive / "move_manifest.jsonl")))
    out = AC.backfill_mirror(saved, archive, tapes, log=lambda m: None, now=123.0)
    assert out["appended"] == 3 and out["already"] == 1
    got = [json.loads(l) for l in (saved / AC.MIRROR_NAME).read_text().splitlines()]
    assert len(got) == 4
    backfilled = [g for g in got if "mirror_backfilled_at" in g]
    assert {g["source"] for g in backfilled} == {"C:/x/1", "C:/x/2", "C:/t/c.jsonl"}
    for g in backfilled:
        assert g["mirror_backfilled_at"] == 123.0 and g["mirror_of"].endswith(".jsonl")
        assert {k: v for k, v in g.items() if k not in ("mirror_of", "mirror_backfilled_at")} in \
            rows + [tape_row]
    again = AC.backfill_mirror(saved, archive, tapes, log=lambda m: None)
    assert again["appended"] == 0 and again["already"] == 4
    from mimo_bridge import storage_locations as SL
    assert SL.find_move(Path("C:/x/1"), manifests=[saved / AC.MIRROR_NAME]) is not None


def test_a_backfill_is_not_an_archiver():
    assert not AC.is_archiver_command(["python", "-m", "mimo_bridge.tools.archive_checkpoints",
                                       "--backfill-mirror"])
