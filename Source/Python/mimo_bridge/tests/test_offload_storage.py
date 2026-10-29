# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Reversible C: -> D: moves of heavy folders (tools/offload_storage.py), in temporary folders.

No real Saved folder, process list or drive is touched: the process list is passed in, and the
C: and D: sides are folders under tmp_path.
"""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import pytest

from mimo_bridge import storage_locations as SL
from mimo_bridge.tools import offload_storage as OS

LINEAGE = "H99_live"
OLD = time.time() - 7 * 3600.0          # older than the 6-hour rule


def _files(folder: Path, n=3, size=700):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / ("f%d.bin" % i)).write_bytes(os.urandom(size))


def _run(saved, name, resume=None):
    run = saved / "experiments" / name
    run.mkdir(parents=True)
    (run / "args.json").write_text(json.dumps({"resume": resume} if resume else {}), encoding="utf-8")
    (run / "replay_training_stats.jsonl").write_text("{}\n", encoding="utf-8")
    return run


def _ckpt(run, step):
    ck = run / "checkpoints" / ("ckpt_step_%d" % step)
    _files(ck, 1)
    (ck / "lineage.json").write_text(json.dumps({"out": str(run)}), encoding="utf-8")
    return ck


def _age(root: Path, when=OLD):
    for p in list(root.rglob("*")) + [root]:
        os.utime(p, (when, when))


@pytest.fixture
def repo(tmp_path):
    """An active three-segment lineage, an old lineage, a study folder, caregiver records."""
    repo = tmp_path / "repo"
    saved = repo / "Saved" / "mimo_bridge"
    s1 = _run(saved, LINEAGE)
    c1 = _ckpt(s1, 100)
    _files(s1 / "videos")
    s2 = _run(saved, LINEAGE + "_seg2", resume=str(c1))
    c2 = _ckpt(s2, 200)
    _files(s2 / "videos")
    s3 = _run(saved, LINEAGE + "_seg3", resume=str(c2))
    _ckpt(s3, 300)
    _files(s3 / "videos")
    s4 = _run(saved, LINEAGE + "_seg4", resume=str(s3 / "checkpoints" / "ckpt_step_300"))
    _ckpt(s4, 400)
    _files(s4 / "videos")
    old = _run(saved, "H50_old")
    _ckpt(old, 5)
    _files(old / "videos")
    _files(saved / "study_2026-09-10")
    _files(saved / "caregiver_records" / LINEAGE / "checkpoint_state")
    _files(saved / "caregiver_records" / "E5_old_tapes")
    (saved / "training_continuation.json").write_text(json.dumps({
        "active_run": str(s4), "command": ["python", "--resume", str(s3 / "checkpoints" / "ckpt_step_300")]}),
        encoding="utf-8")
    _age(repo)
    return repo, saved, dict(s1=s1, s2=s2, s3=s3, s4=s4, old=old)


def _plan(saved, procs=(), now=None):
    return OS.plan(saved, procs=list(procs), now=now)


def _by_path(report):
    return {u["path"].split("mimo_bridge/")[-1]: u for u in report["units"]}


def test_the_plan_lists_what_may_move_and_why_the_rest_stays_and_writes_nothing(repo):
    root, saved, runs = repo
    before = sorted((p.relative_to(root).as_posix(), p.stat().st_mtime) for p in root.rglob("*"))
    report = _plan(saved)
    after = sorted((p.relative_to(root).as_posix(), p.stat().st_mtime) for p in root.rglob("*"))
    assert before == after, "the plan changed something on C:"
    u = _by_path(report)
    assert report["summary"]["lineage"] == LINEAGE
    assert report["summary"]["dashboard_runs"] == [LINEAGE + "_seg4", LINEAGE + "_seg3", LINEAGE + "_seg2"]
    may = {k for k, v in u.items() if v["candidate"]}
    assert may == {"experiments/H50_old/checkpoints", "experiments/H50_old/videos",
                   "experiments/%s/videos" % LINEAGE, "study_2026-09-10",
                   "caregiver_records/E5_old_tapes"}, may
    assert report["summary"]["candidate_bytes"] == sum(u[k]["bytes"] for k in may)
    def why(k):
        return " | ".join(u[k]["stay"])
    assert "the checkpoint archiver moves them" in why("experiments/%s_seg4/checkpoints" % LINEAGE)
    assert "one of the latest 3" in why("experiments/%s_seg4/checkpoints" % LINEAGE)
    assert "resume source" in why("experiments/%s_seg3/checkpoints" % LINEAGE)
    assert "development dashboard" in why("experiments/%s_seg2/videos" % LINEAGE)
    assert "live recordings" in why("caregiver_records/%s" % LINEAGE)


def test_the_active_lineages_scene_folders_stay_even_when_old(repo):
    """Every resumed H84 world writes its scene files into its saved run_dir, the lineage's
    first run's scenes folder; an old scenes folder of an ended lineage may move."""
    root, saved, runs = repo
    _files(runs["s1"] / "scenes", 1)
    _files(runs["old"] / "scenes", 1)
    _age(root)
    u = _by_path(_plan(saved))
    assert "write their scene files here" in " ".join(u["experiments/%s/scenes" % LINEAGE]["stay"])
    assert u["experiments/H50_old/scenes"]["candidate"]


def test_a_folder_holding_a_runs_records_never_moves(repo):
    root, saved, runs = repo
    nested = runs["old"] / "videos" / "inner_run"
    nested.mkdir()
    (nested / "args.json").write_text("{}", encoding="utf-8")
    top_run = saved / "study_is_a_run"          # a run outside experiments/: subfolders only
    _files(top_run / "videos")
    (top_run / "replay_training_stats.jsonl").write_text("{}\n", encoding="utf-8")
    _age(root)
    u = _by_path(_plan(saved))
    assert "args.json" in " ".join(u["experiments/H50_old/videos"]["stay"])
    assert "study_is_a_run" not in u and u["study_is_a_run/videos"]["candidate"]


def test_anything_modified_in_the_last_six_hours_stays(repo):
    root, saved, runs = repo
    (runs["old"] / "videos" / "f0.bin").write_bytes(b"fresh")
    u = _by_path(_plan(saved))
    assert "modified 0.0 h ago" in " ".join(u["experiments/H50_old/videos"]["stay"])


def test_what_a_running_process_reads_stays_and_a_broad_root_does_not_count(repo):
    root, saved, runs = repo
    procs = [dict(pid=1, name="python.exe", named=[str(runs["old"])], open=[], readable=True),
             dict(pid=2, name="python.exe", named=[str(saved)], open=[], readable=True),
             dict(pid=3, name="vlc.exe", named=[], readable=True,
                  open=[str(saved / "study_2026-09-10" / "f1.bin")]),
             dict(pid=4, name="System", named=[], open=[], readable=False)]
    report = _plan(saved, procs)
    u = _by_path(report)
    assert "process 1" in " ".join(u["experiments/H50_old/videos"]["stay"])      # --run <run>
    assert "process 1" in " ".join(u["experiments/H50_old/checkpoints"]["stay"])
    assert "process 3" in " ".join(u["study_2026-09-10"]["stay"])                # holds a file open
    assert u["caregiver_records/E5_old_tapes"]["candidate"], "a broad --root blocked everything"
    assert report["summary"]["processes_open_files_unreadable"] == 1


def _move(root, saved, rel, tmp_path, procs=()):
    ctx = OS.context(saved, procs=list(procs))
    return OS.move(rel, repo=root, root=tmp_path / "D" / "C_offload", saved=saved, ctx=ctx,
                   c_manifest=saved / "storage_moves.jsonl",
                   d_manifest=tmp_path / "D" / "C_offload" / "storage_moves.jsonl", log=lambda m: None)


def test_a_move_is_verified_recorded_on_both_disks_pointed_to_and_found_by_the_resolver(repo, tmp_path):
    root, saved, runs = repo
    src = runs["old"] / "videos"
    want = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir()}
    assert _move(root, saved, "Saved/mimo_bridge/experiments/H50_old/videos", tmp_path)
    dst = tmp_path / "D" / "C_offload" / "Saved" / "mimo_bridge" / "experiments" / "H50_old" / "videos"
    assert not src.exists() and dst.is_dir()
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.iterdir()} == want
    c_rows = [json.loads(l) for l in (saved / "storage_moves.jsonl").read_text().splitlines()]
    d_rows = [json.loads(l) for l in (tmp_path / "D" / "C_offload" / "storage_moves.jsonl").read_text().splitlines()]
    assert [r["event"] for r in c_rows] == ["planned", "moved"] == [r["event"] for r in d_rows]
    assert c_rows[1]["files"] == want and d_rows[1]["mirror_of"].endswith("storage_moves.jsonl")
    pointer = json.loads((runs["old"] / "videos.moved.json").read_text(encoding="utf-8"))
    assert Path(pointer["moved_to"]) == dst and "restore" in pointer["restore"]
    c_only = [saved / "storage_moves.jsonl"]
    assert SL.locate(src, manifests=c_only) == dst
    assert SL.locate(src / "f1.bin", manifests=c_only) == dst / "f1.bin"
    # the run's own records never moved
    assert (runs["old"] / "args.json").exists() and (runs["old"] / "replay_training_stats.jsonl").exists()


def test_a_copy_that_arrives_different_leaves_c_untouched(repo, tmp_path, monkeypatch):
    root, saved, runs = repo
    real = shutil.copytree

    def corrupting(a, b, *x, **k):
        out = real(a, b, *x, **k)
        (Path(b) / "f0.bin").write_bytes(b"bit rot")
        return out
    monkeypatch.setattr(OS.shutil, "copytree", corrupting)
    assert not _move(root, saved, "Saved/mimo_bridge/study_2026-09-10", tmp_path)
    assert (saved / "study_2026-09-10" / "f0.bin").exists()
    assert not (tmp_path / "D" / "C_offload" / "Saved" / "mimo_bridge" / "study_2026-09-10").exists()
    events = [json.loads(l)["event"] for l in (saved / "storage_moves.jsonl").read_text().splitlines()]
    assert events == ["planned"], "a move that failed its hash check was recorded as moved"
    assert SL.locate(saved / "study_2026-09-10", manifests=[saved / "storage_moves.jsonl"]) == (
        saved / "study_2026-09-10")


def test_a_move_refuses_what_must_stay_and_what_is_not_a_unit(repo, tmp_path):
    root, saved, runs = repo
    assert not _move(root, saved, "Saved/mimo_bridge/experiments/%s_seg4/checkpoints" % LINEAGE, tmp_path)
    assert not _move(root, saved, "Saved/mimo_bridge/experiments/H50_old", tmp_path)   # a run folder
    assert (runs["s4"] / "checkpoints").exists() and (runs["old"] / "args.json").exists()
    assert not (saved / "storage_moves.jsonl").exists()


def test_an_earlier_moved_copy_on_d_is_never_overwritten(repo, tmp_path):
    """A folder re-created at a path whose earlier move is recorded, with its pointer gone:
    moving it again would delete the only copy of the first folder."""
    root, saved, runs = repo
    assert _move(root, saved, "Saved/mimo_bridge/study_2026-09-10", tmp_path)
    (saved / "study_2026-09-10.moved.json").unlink()
    _files(saved / "study_2026-09-10", 1)
    _age(root)
    assert not _move(root, saved, "Saved/mimo_bridge/study_2026-09-10", tmp_path)
    dst = tmp_path / "D" / "C_offload" / "Saved" / "mimo_bridge" / "study_2026-09-10"
    assert sorted(p.name for p in dst.iterdir()) == ["f0.bin", "f1.bin", "f2.bin"]


def test_restore_brings_it_back_identical_and_the_record_says_so(repo, tmp_path):
    root, saved, runs = repo
    src = saved / "caregiver_records" / "E5_old_tapes"
    want = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir()}
    assert _move(root, saved, "Saved/mimo_bridge/caregiver_records/E5_old_tapes", tmp_path)
    assert not src.exists()
    assert OS.restore("Saved/mimo_bridge/caregiver_records/E5_old_tapes", repo=root,
                      root=tmp_path / "D" / "C_offload", c_manifest=saved / "storage_moves.jsonl",
                      d_manifest=tmp_path / "D" / "C_offload" / "storage_moves.jsonl", log=lambda m: None)
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir()} == want
    assert not (src.parent / "E5_old_tapes.moved.json").exists()
    dst = tmp_path / "D" / "C_offload" / "Saved" / "mimo_bridge" / "caregiver_records" / "E5_old_tapes"
    assert not dst.exists(), "the D: copy was left after a verified restore"
    events = [json.loads(l)["event"] for l in (saved / "storage_moves.jsonl").read_text().splitlines()]
    assert events == ["planned", "moved", "restored"]
    assert SL.find_move(src, manifests=[saved / "storage_moves.jsonl"]) is None
    # and it can move again: the plan sees it as an ordinary folder
    _age(src)
    assert _by_path(_plan(saved))["caregiver_records/E5_old_tapes"]["candidate"]


# ------------------------------------------------------------ what the live lineage reads by path
REAL_SAVED = Path(__file__).resolve().parents[4] / "Saved" / "mimo_bridge"
SEG31_ARGS = REAL_SAVED / "experiments" / "H84_fresh_drivers_20260918_seg31_toy_and_push" / "args.json"


def _real_args():
    if not SEG31_ARGS.exists():
        pytest.fail("segment 31's args.json is gone from %s; this test reads the real file "
                    "(storage_locations.resolve_run_args finds it if it moved)" % SEG31_ARGS)
    return json.loads(SEG31_ARGS.read_text(encoding="utf-8"))


def test_every_path_segment_31s_args_names_keeps_its_folder(tmp_path):
    """Review ops-storage finding 3: the plan listed the caregiver speech library as movable while
    segment 31's args.json names it and every H84 launch opens it by path. Built from the real
    args.json, with its Saved folder mapped onto a temporary one."""
    real = _real_args()
    repo = tmp_path / "repo"
    saved = repo / "Saved" / "mimo_bridge"
    text = json.dumps(real).replace(json.dumps(str(REAL_SAVED))[1:-1],
                                    json.dumps(str(saved))[1:-1])
    args = json.loads(text)
    run = Path(args["out"])
    run.mkdir(parents=True)
    (run / "args.json").write_text(json.dumps(args), encoding="utf-8")
    speech = Path(args["caregiver_speech_dir"])
    _files(speech, 2)
    _files(saved / "caregiver_speech_exchange_v1", 1)          # the older library: nothing names it
    _files(run / "explain", 1)          # not the dashboard's videos: judged by the other rules alone
    (saved / "training_continuation.json").write_text(json.dumps({"active_run": str(run)}),
                                                      encoding="utf-8")
    _age(repo)
    report = OS.plan(saved, procs=[], lineage="H84_fresh_drivers_20260918")
    u = _by_path(report)
    why = " | ".join(u[speech.name]["stay"])
    assert "args.json (caregiver_speech_dir)" in why, why
    assert u["caregiver_speech_exchange_v1"]["candidate"], "a folder nothing names was kept"
    # the control that the args rule does not name the whole run folder ('out'): no args.json reason
    # keeps the run's explain folder. (Rewritten by verify-fix-ops 2026-09-23: the folder now stays
    # by the active-run rule, so "it is a candidate" no longer tests the args rule.)
    explain = u["experiments/%s/explain" % run.name]["stay"]
    assert not any("args.json" in w for w in explain), (
        "the run folder named by 'out' kept every subfolder of the run: %s" % explain)
    assert any("the active run's own folder" in w for w in explain), explain
    assert report["summary"]["named_paths"] >= 4


def test_the_args_rule_covers_every_path_the_saved_worlds_env_settings_carry():
    """The control on the rule's reach: train_mimo builds each world's env settings from the run's
    args (env_kwargs_for); every path-valued setting it produces must be one args_paths yields,
    or a world's folder could be offered for moving while the lineage reads it."""
    import argparse
    from mimo_bridge.train_mimo import env_kwargs_for
    real = _real_args()
    kw = env_kwargs_for(argparse.Namespace(**real), None, real["out"])
    world_paths = {SL._key(v) for v in kw.values() if OS._path_like(v)}
    assert world_paths, "the real settings carry no path at all: the check proves nothing"
    covered = {SL._key(p) for _, p in OS.args_paths(real)}
    assert world_paths <= covered, sorted(world_paths - covered)


def test_the_live_systems_folders_and_lock_folders_stay(repo):
    root, saved, runs = repo
    for name in ("gof_probes", "lineage_series", "dashboard_history", "developmental_queue.lock"):
        _files(saved / name, 1)
    _age(root)
    u = _by_path(_plan(saved))
    for name in ("gof_probes", "lineage_series", "dashboard_history"):
        assert "the live system's own folder" in " ".join(u[name]["stay"]), name
    assert "lock folder" in " ".join(u["developmental_queue.lock"]["stay"])
    assert u["study_2026-09-10"]["candidate"], "the control: an ordinary study folder may move"


def test_the_active_runs_own_subfolders_stay_after_a_quiet_night(repo):
    """verify-fix-ops 2026-09-23: with training paused more than 6 hours, the plan listed segment
    31's live, explain and views folders as movable; the development dashboard reads the active
    run's live frames (and writes live/viewer.log) and the explainer rebuilds from its snapshots."""
    root, saved, runs = repo
    for name in ("live", "explain", "views"):
        _files(runs["s4"] / name, 1)
    _files(runs["s3"] / "explain", 1)                  # the control: an earlier segment's may move
    _age(root)
    u = _by_path(_plan(saved))
    for name in ("live", "explain", "views"):
        row = u["experiments/%s_seg4/%s" % (LINEAGE, name)]
        assert not row["candidate"] and "the active run's own folder" in " ".join(row["stay"]), name
    assert u["experiments/%s_seg3/explain" % LINEAGE]["candidate"]


def test_a_probe_runs_checkpoints_are_not_kept_for_the_archiver(repo):
    """Finding 10: the archiver leaves <lineage>_explainprobe_* out of the lineage, so keeping its
    checkpoints here for the archiver left them where neither tool could move them."""
    root, saved, runs = repo
    probe = _run(saved, LINEAGE + "_seg4_explainprobe_20260922-190000")
    _ckpt(probe, 9000)
    _age(root)
    u = _by_path(_plan(saved))
    row = u["experiments/%s/checkpoints" % probe.name]
    assert not any("archiver moves them" in w for w in row["stay"]), row["stay"]
    assert row["candidate"]
    assert "archiver moves them" in " ".join(
        u["experiments/%s_seg4/checkpoints" % LINEAGE]["stay"]), "the lineage's own still stay"


# ------------------------------------------------------------ interrupted moves (finding 12)
def _kwargs(saved, tmp_path):
    return dict(repo=saved.parents[1], root=tmp_path / "D" / "C_offload",
                c_manifest=saved / "storage_moves.jsonl",
                d_manifest=tmp_path / "D" / "C_offload" / "storage_moves.jsonl", log=lambda m: None)


class _Died(BaseException):
    """Stands in for the process dying between two steps."""


def _die_on_rename_to_trash(monkeypatch):
    real_rename = Path.rename

    def dies(self, target):
        if ".offloading_" in str(target):
            raise _Died()
        return real_rename(self, target)
    monkeypatch.setattr(Path, "rename", dies)
    return real_rename


def test_a_move_that_died_before_taking_c_away_is_finished_by_move(repo, tmp_path, monkeypatch):
    root, saved, runs = repo
    src = saved / "study_2026-09-10"
    real_rename = _die_on_rename_to_trash(monkeypatch)
    with pytest.raises(_Died):
        _move(root, saved, "Saved/mimo_bridge/study_2026-09-10", tmp_path)
    monkeypatch.setattr(Path, "rename", real_rename)
    dst = tmp_path / "D" / "C_offload" / "Saved" / "mimo_bridge" / "study_2026-09-10"
    assert src.is_dir() and dst.is_dir()                    # both copies whole, the row says moved
    assert _move(root, saved, "Saved/mimo_bridge/study_2026-09-10", tmp_path)
    assert not src.exists() and dst.is_dir()
    assert (saved / "study_2026-09-10.moved.json").exists()
    assert SL.locate(src, manifests=[saved / "storage_moves.jsonl"]) == dst


def test_a_restore_that_died_before_its_record_is_finished_by_restore(repo, tmp_path, monkeypatch):
    root, saved, runs = repo
    src = saved / "caregiver_records" / "E5_old_tapes"
    assert _move(root, saved, "Saved/mimo_bridge/caregiver_records/E5_old_tapes", tmp_path)
    real_record = OS._record

    def dies(row, *a, **k):
        if row.get("event") == "restored":
            raise _Died()
        return real_record(row, *a, **k)
    monkeypatch.setattr(OS, "_record", dies)
    with pytest.raises(_Died):
        OS.restore("Saved/mimo_bridge/caregiver_records/E5_old_tapes", **_kwargs(saved, tmp_path))
    monkeypatch.setattr(OS, "_record", real_record)
    dst = tmp_path / "D" / "C_offload" / "Saved" / "mimo_bridge" / "caregiver_records" / "E5_old_tapes"
    assert src.is_dir() and dst.is_dir()
    assert OS.restore("Saved/mimo_bridge/caregiver_records/E5_old_tapes", **_kwargs(saved, tmp_path))
    assert src.is_dir() and not dst.exists()
    assert not (src.parent / "E5_old_tapes.moved.json").exists()
    events = [json.loads(l)["event"] for l in (saved / "storage_moves.jsonl").read_text().splitlines()]
    assert events == ["planned", "moved", "restored"]


def test_a_restore_that_died_after_its_record_leaves_nothing_stuck(repo, tmp_path, monkeypatch):
    """verify-fix-ops 2026-09-23: the third window of finding 12. A restore that died after writing
    `restored` and before removing the pointer left a folder that `restore` called already on C: and
    that the pointer kept from ever moving again ("restore that move first"). The pointer now goes
    first, so the same death leaves a folder that moves again."""
    root, saved, runs = repo
    rel = "Saved/mimo_bridge/caregiver_records/E5_old_tapes"
    src = saved / "caregiver_records" / "E5_old_tapes"
    assert _move(root, saved, rel, tmp_path)
    real_record = OS._record

    def dies_after(row, *a, **k):
        real_record(row, *a, **k)
        if row.get("event") == "restored":
            raise _Died()
    monkeypatch.setattr(OS, "_record", dies_after)
    with pytest.raises(_Died):
        OS.restore(rel, **_kwargs(saved, tmp_path))
    monkeypatch.setattr(OS, "_record", real_record)
    assert src.is_dir()
    assert not (src.parent / "E5_old_tapes.moved.json").exists(), "a stale pointer was left beside it"
    _age(root)
    row = _by_path(_plan(saved))["caregiver_records/E5_old_tapes"]
    assert row["candidate"], row["stay"]
    assert _move(root, saved, rel, tmp_path), "the folder could never move again"


def test_a_c_folder_that_differs_from_the_record_is_never_finished(repo, tmp_path, monkeypatch):
    """The control: the repair runs only on the state it can prove. A C: folder that changed
    since the move was recorded is left alone by both commands."""
    root, saved, runs = repo
    src = saved / "study_2026-09-10"
    real_rename = _die_on_rename_to_trash(monkeypatch)
    with pytest.raises(_Died):
        _move(root, saved, "Saved/mimo_bridge/study_2026-09-10", tmp_path)
    monkeypatch.setattr(Path, "rename", real_rename)
    (src / "f0.bin").write_bytes(b"edited after the move was recorded")
    _age(root)
    assert not _move(root, saved, "Saved/mimo_bridge/study_2026-09-10", tmp_path)
    assert not OS.restore("Saved/mimo_bridge/study_2026-09-10", **_kwargs(saved, tmp_path))
    dst = tmp_path / "D" / "C_offload" / "Saved" / "mimo_bridge" / "study_2026-09-10"
    assert src.is_dir() and dst.is_dir()


def test_a_folder_holding_files_the_repository_tracks_stays(repo):
    """Found by the 2026-09-23 dry plan: the trainer's default speech library (read by path by
    tools and tests) was a candidate. Any folder holding git-tracked files stays."""
    root, saved, runs = repo
    _files(saved / "caregiver_speech", 1)
    _age(root)
    u = _by_path(OS.plan(saved, procs=[], tracked=["Saved/mimo_bridge/study_2026-09-10/f0.bin",
                                                   "Saved/mimo_bridge/study_2026-09-10/f1.bin"]))
    assert "holds 2 files the repository tracks" in " ".join(u["study_2026-09-10"]["stay"])
    assert "default --caregiver-speech-dir" in " ".join(u["caregiver_speech"]["stay"])
    # the control: with no tracked files the same study folder may move
    assert _by_path(OS.plan(saved, procs=[], tracked=[]))["study_2026-09-10"]["candidate"]
