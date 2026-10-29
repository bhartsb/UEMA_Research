# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Finding a moved checkpoint, companion or recording through the move manifests, proven by hash.

Temporary folders stand in for C: and D:. No real checkpoint, manifest or drive is read.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest

from mimo_bridge import storage_locations as SL


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _append(manifest: Path, row: dict):
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _archived_checkpoint(tmp_path):
    """A checkpoint and its companion moved C: -> D: exactly as the archiver does it, with the
    archiver's manifest row. Returns (c_ckpt, d_ckpt, c_comp, d_comp, manifest, run_dir)."""
    run = tmp_path / "C" / "experiments" / "H99_seg1"
    (run / "checkpoints").mkdir(parents=True)
    (run / "args.json").write_text('{"scene": "crib"}', encoding="utf-8")
    c_ck = run / "checkpoints" / "ckpt_step_500"
    d_ck = tmp_path / "D" / "archive" / "H99_seg1" / "checkpoints" / "ckpt_step_500"
    d_ck.mkdir(parents=True)
    (d_ck / "Model.Safetensors").write_bytes(os.urandom(512))    # mixed case on purpose
    (d_ck / "lineage.json").write_text(json.dumps({"out": str(run)}), encoding="utf-8")
    (d_ck / "sub").mkdir()
    (d_ck / "sub" / "part.bin").write_bytes(b"part")
    files = {str(p.relative_to(d_ck)).replace("\\", "/"): _sha(p)
             for p in sorted(d_ck.rglob("*")) if p.is_file()}
    c_comp = tmp_path / "C" / "caregiver_records" / "H99" / "ckpt_step_500_abc.pt.gz"
    d_comp = tmp_path / "D" / "archive" / "checkpoint_state" / "H99_seg1" / c_comp.name
    d_comp.parent.mkdir(parents=True)
    d_comp.write_bytes(os.urandom(1024))
    manifest = tmp_path / "D" / "archive" / "move_manifest.jsonl"
    _append(manifest, dict(event="moved", step=500, run="H99_seg1", source=str(c_ck),
                           destination=str(d_ck), companion_source=str(c_comp),
                           companion_destination=str(d_comp), companion_sha256=_sha(d_comp),
                           files=files, bytes=1, time=1.0))
    return c_ck, d_ck, c_comp, d_comp, manifest, run


def test_an_existing_path_is_returned_unchanged_and_no_manifest_is_read(tmp_path):
    here = tmp_path / "file.bin"
    here.write_bytes(b"x")
    missing_manifest = tmp_path / "no" / "such.jsonl"
    assert SL.locate(here, manifests=[missing_manifest]) == here


def test_a_path_no_manifest_knows_is_returned_unchanged(tmp_path):
    gone = tmp_path / "never_moved.bin"
    assert SL.locate(gone, manifests=[tmp_path / "empty.jsonl"]) == gone


def test_a_moved_companion_is_found_and_proven_against_the_receipts_hash(tmp_path):
    c_ck, d_ck, c_comp, d_comp, manifest, run = _archived_checkpoint(tmp_path)
    assert SL.locate(c_comp, manifests=[manifest]) == d_comp
    assert SL.locate(c_comp, expected_sha256=_sha(d_comp), manifests=[manifest]) == d_comp
    with pytest.raises(SL.MovedFileError, match="not the same file"):
        SL.locate(c_comp, expected_sha256="0" * 64, manifests=[manifest])
    d_comp.write_bytes(b"changed on the archive disk")
    SL._VERIFIED.clear()
    with pytest.raises(SL.MovedFileError, match="not the"):
        SL.locate(c_comp, manifests=[manifest])


def test_a_moved_checkpoint_folder_and_a_file_and_folder_inside_it_are_found(tmp_path):
    c_ck, d_ck, c_comp, d_comp, manifest, run = _archived_checkpoint(tmp_path)
    assert SL.locate(c_ck, manifests=[manifest]) == d_ck
    # the manifest's file names keep their case; the caller's spelling of the folder does not matter
    inside = Path(str(c_ck).upper() if os.name == "nt" else str(c_ck)) / "Model.Safetensors"
    assert SL.locate(inside, manifests=[manifest]) == d_ck / "Model.Safetensors"
    assert SL.locate(c_ck / "sub", manifests=[manifest]) == d_ck / "sub"
    (d_ck / "sub" / "part.bin").write_bytes(b"tampered")
    SL._VERIFIED.clear()
    with pytest.raises(SL.MovedFileError, match="part.bin"):
        SL.locate(c_ck, manifests=[manifest])


def test_a_relative_spelling_of_a_moved_path_is_found(tmp_path, monkeypatch):
    """Review ops-storage finding 8: `--ckpt Saved/...` from the repository root was returned
    unchanged for a checkpoint the archiver had moved; the absolute spelling found the D: copy."""
    c_ck, d_ck, c_comp, d_comp, manifest, run = _archived_checkpoint(tmp_path)
    monkeypatch.chdir(tmp_path)
    rel = Path(os.path.relpath(c_ck, tmp_path))
    assert not rel.is_absolute()
    assert SL.locate(rel, manifests=[manifest]) == d_ck
    assert SL.locate(rel / "Model.Safetensors", manifests=[manifest]) == d_ck / "Model.Safetensors"
    # the control: from another folder the same relative spelling names another place
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert SL.locate(rel, manifests=[manifest]) == rel


def test_a_row_appended_after_a_lookup_is_seen_by_the_next_one(tmp_path):
    """The parsed manifest is cached until the file changes; an append must never be missed."""
    src, dst = tmp_path / "C" / "a.bin", tmp_path / "D" / "a.bin"
    dst.parent.mkdir(parents=True)
    dst.write_bytes(b"payload")
    m = tmp_path / "moves.jsonl"
    SL.append_row(m, dict(event="moved", source=str(tmp_path / "C" / "other.bin"),
                          destination=str(dst), sha256=_sha(dst)))
    assert SL.locate(src, manifests=[m]) == src                  # not known yet
    SL.append_row(m, dict(event="moved", source=str(src), destination=str(dst), sha256=_sha(dst)))
    assert SL.locate(src, manifests=[m]) == dst


def test_a_moved_folder_whose_copy_is_gone_is_refused_not_used(tmp_path):
    c_ck, d_ck, c_comp, d_comp, manifest, run = _archived_checkpoint(tmp_path)
    import shutil
    shutil.rmtree(d_ck)
    with pytest.raises(SL.MovedFileError, match="nothing is there"):
        SL.locate(c_ck, manifests=[manifest])


def test_the_c_manifest_answers_before_the_d_manifests(tmp_path):
    src = tmp_path / "C" / "chunk.jsonl"
    first, second = tmp_path / "D1" / "chunk.jsonl", tmp_path / "D2" / "chunk.jsonl"
    for p, body in ((first, b"one"), (second, b"two")):
        p.parent.mkdir(parents=True)
        p.write_bytes(body)
    c_manifest, d_manifest = tmp_path / "c.jsonl", tmp_path / "d.jsonl"
    _append(c_manifest, dict(event="moved", source=str(src), destination=str(first),
                             sha256=_sha(first)))
    _append(d_manifest, dict(event="moved", source=str(src), destination=str(second),
                             sha256=_sha(second)))
    assert SL.locate(src, manifests=[c_manifest, d_manifest]) == first
    assert SL.locate(src, manifests=[d_manifest, c_manifest]) == second


def test_a_restore_cancels_the_move_and_a_torn_last_line_hides_nothing(tmp_path):
    src = tmp_path / "C" / "videos"
    dst = tmp_path / "D" / "videos"
    dst.mkdir(parents=True)
    (dst / "a.mp4").write_bytes(b"video")
    m = tmp_path / "moves.jsonl"
    row = dict(event="moved", kind="folder", source=str(src), destination=str(dst),
               files={"a.mp4": _sha(dst / "a.mp4")})
    _append(m, row)
    with open(m, "a", encoding="utf-8") as fh:
        fh.write('{"event": "moved", "source": "torn')         # a writer died mid-line
    assert SL.locate(src, manifests=[m]) == dst
    SL.append_row(m, dict(row, event="restored"))        # starts a fresh line after the fragment
    assert SL.locate(src, manifests=[m]) == src, "a restored folder is looked for where it was"


def test_a_file_moved_twice_is_followed_to_its_second_place(tmp_path):
    a, b, c = tmp_path / "A" / "x.bin", tmp_path / "B" / "x.bin", tmp_path / "C" / "x.bin"
    c.parent.mkdir(parents=True)
    c.write_bytes(b"payload")
    m = tmp_path / "moves.jsonl"
    _append(m, dict(event="moved", source=str(a), destination=str(b), sha256=_sha(c)))
    _append(m, dict(event="moved", source=str(b), destination=str(c), sha256=_sha(c)))
    assert SL.locate(a, manifests=[m]) == c


def test_run_args_are_found_beside_the_run_and_through_lineage_for_an_archived_checkpoint(tmp_path):
    c_ck, d_ck, c_comp, d_comp, manifest, run = _archived_checkpoint(tmp_path)
    # the archived copy, named by its D: path: the archive holds no args.json
    assert SL.resolve_run_args(d_ck, manifests=[manifest]) == run / "args.json"
    # the same checkpoint named by the C: path the archiver removed
    assert SL.resolve_run_args(c_ck, manifests=[manifest]) == run / "args.json"
    # a checkpoint still on C: finds args.json beside its run folder
    live = run / "checkpoints" / "ckpt_step_900"
    live.mkdir()
    assert SL.resolve_run_args(live, manifests=[manifest]) == run / "args.json"
    (run / "args.json").unlink()
    with pytest.raises(FileNotFoundError, match="no args.json"):
        SL.resolve_run_args(d_ck, manifests=[manifest])


def _worktree(tmp_path):
    """A main tree and a git worktree of it, laid out as `git worktree add` lays them out."""
    main, wt = tmp_path / "main", tmp_path / "wt"
    admin = main / ".git" / "worktrees" / "wt"
    admin.mkdir(parents=True)
    (admin / "commondir").write_text("../..\n", encoding="utf-8")
    wt.mkdir()
    (wt / ".git").write_text("gitdir: %s\n" % admin, encoding="utf-8")
    return main, wt


def test_main_tree_is_found_from_a_worktree_and_is_itself_from_the_main_tree(tmp_path):
    main, wt = _worktree(tmp_path)
    assert SL._key(SL.main_tree(wt)) == SL._key(main)
    assert SL.main_tree(main) == main
    lone = tmp_path / "no_git"
    lone.mkdir()
    assert SL.main_tree(lone) == lone


REL = Path("Saved/mimo_bridge/experiments/H99_seg1/checkpoints/ckpt_step_500")


def test_find_in_trees_reads_the_main_trees_copy_from_a_worktree(tmp_path):
    """2026-09-26: a worktree's Saved/ holds no checkpoints; the eight checkpoint proofs looked only
    at one hand-built path and skipped."""
    main, wt = _worktree(tmp_path)
    (main / REL).mkdir(parents=True)
    found, looked = SL.find_in_trees(REL, repo=wt, manifests=[])
    assert found == main / REL and looked == [str(wt / REL), str(main / REL)]
    # an override answers first
    other = tmp_path / "elsewhere" / "ckpt_step_500"
    other.mkdir(parents=True)
    assert SL.find_in_trees(REL, also=[str(other), None], repo=wt, manifests=[])[0] == other


def test_find_in_trees_follows_a_move_of_the_main_trees_copy_proven_by_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(SL, "C_MANIFEST", tmp_path / "no_c_mirror.jsonl")      # no real manifest read
    monkeypatch.setattr(SL, "D_MANIFESTS", (tmp_path / "no_d_manifest.jsonl",))
    main, wt = _worktree(tmp_path)
    d_ck = tmp_path / "D" / "archive" / "ckpt_step_500"
    d_ck.mkdir(parents=True)
    (d_ck / "model.safetensors").write_bytes(os.urandom(256))
    manifest = tmp_path / "D" / "move_manifest.jsonl"
    _append(manifest, dict(event="moved", source=str(main / REL), destination=str(d_ck),
                           files={"model.safetensors": _sha(d_ck / "model.safetensors")}))
    assert SL.find_in_trees(REL, repo=wt, manifests=[manifest])[0] == d_ck
    # the main tree's C: mirror is read by default, beside the D: manifests
    mirror = main / "Saved" / "mimo_bridge" / "storage_moves.jsonl"
    _append(mirror, dict(event="moved", source=str(main / REL), destination=str(d_ck),
                         files={"model.safetensors": _sha(d_ck / "model.safetensors")}))
    assert SL.find_in_trees(REL, repo=wt)[0] == d_ck
    # a copy that differs from its record is a damaged archive, never "not on this machine"
    (d_ck / "model.safetensors").write_bytes(b"changed")
    SL._VERIFIED.clear()
    with pytest.raises(SL.MovedFileError, match="model.safetensors"):
        SL.find_in_trees(REL, repo=wt, manifests=[manifest])
    # without the proof the path is still returned (the caller hashes it later)
    assert SL.find_in_trees(REL, repo=wt, manifests=[manifest], verify=False)[0] == d_ck


def test_find_in_trees_names_every_place_it_searched_when_the_path_is_nowhere(tmp_path):
    main, wt = _worktree(tmp_path)
    manifest = tmp_path / "D" / "move_manifest.jsonl"
    found, looked = SL.find_in_trees(REL, also=[str(tmp_path / "env_named")], repo=wt,
                                     manifests=[manifest])
    assert found is None
    assert looked[:3] == [str(tmp_path / "env_named"), str(wt / REL), str(main / REL)]
    assert str(manifest) in looked[-1] and "none of which names it" in looked[-1]


def test_the_default_manifests_are_where_the_archiver_and_the_tape_reader_write(monkeypatch):
    """A control on the constants: if the archiver's D: folder or the tape archive move, the
    resolver must move with them or it silently stops finding anything."""
    from mimo_bridge.tools import archive_checkpoints as AC
    from mimo_bridge import event_tape as ET
    paths = [SL._key(p) for p in SL.default_manifests()]
    assert paths[0] == SL._key(AC.SAVED / "storage_moves.jsonl")
    assert SL._key(AC.ARCHIVE / "move_manifest.jsonl") in paths
    assert SL._key(ET.TAPE_ARCHIVE / "tape_move_manifest.jsonl") in paths
