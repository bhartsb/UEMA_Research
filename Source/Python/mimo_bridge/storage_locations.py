# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Find a file or folder that a storage move took off its original path, and prove it is the same.

WHY THIS EXISTS. Checkpoints, their episode companions and the caregiver's recording chunks leave C:
for D: (tools/archive_checkpoints.py), and whole heavy folders can too (tools/offload_storage.py).
Every mover writes a manifest row naming the old path, the new path and each file's SHA-256, but
until 2026-09-22 nothing read those rows back. Measured that day: the archived checkpoint
D:/.../seg31_toy_and_push/checkpoints/ckpt_step_340526 keeps, in its episode_state.json, the C:
companion path the archiver deleted, so continuation_state.load failed on it; and a probe of an
archived checkpoint could not find its run's args.json, because the archive holds checkpoints only.

WHERE THE ROWS ARE, read in this order (the first file that knows a path answers):
  1. Saved/mimo_bridge/storage_moves.jsonl on C:. The archiver mirrors every row it writes here
     (from 2026-09-22) and offload_storage writes its own rows here.
  2. The D: manifests, for moves made before the C: mirror existed or a C: file that was lost:
     H84_checkpoint_archive/move_manifest.jsonl (a checkpoint folder and its companion per row),
     H84_checkpoint_archive/caregiver_tapes/tape_move_manifest.jsonl (one recording file per row)
     and C_offload/storage_moves.jsonl (offload_storage's mirror).

A path that exists is returned unchanged and unread, so every caller behaves exactly as before
whenever nothing has moved. A path no row knows is also returned unchanged, so the caller's own
"file not found" still says what it always said. A path a row knows but whose new copy is missing
or hashes differently raises MovedFileError naming both places: a moved file is never used
unproven.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SAVED = REPO / "Saved" / "mimo_bridge"
C_MANIFEST = SAVED / "storage_moves.jsonl"
# The D: roots. tools/archive_checkpoints.ARCHIVE and event_tape.TAPE_ARCHIVE name the same two
# folders; tests/test_storage_locations.py fails if they drift apart.
ARCHIVE_ROOT = Path("D:/UEMA_Research/external/D/UEMA_recordings/H84_checkpoint_archive")
OFFLOAD_ROOT = Path("D:/UEMA_Research/external/D/UEMA_recordings/C_offload")
D_MANIFESTS = (ARCHIVE_ROOT / "move_manifest.jsonl",
               ARCHIVE_ROOT / "caregiver_tapes" / "tape_move_manifest.jsonl",
               OFFLOAD_ROOT / "storage_moves.jsonl")


class MovedFileError(FileNotFoundError):
    """A manifest says where the path went, and what is there is missing or not the same file."""


def long_path(path):
    """Windows refuses paths past 260 characters unless they carry the extended-length
    prefix, and reports the refusal as "file not found" on a folder that exists. The
    companion file's path repeats the run name (run directory, then the caregiver
    storage) and crossed the line in the H84 launch smokes of 2026-09-18 at the end
    save. Every open, rename, stat and mkdir on a companion goes through here; short
    paths and other platforms pass unchanged."""
    p = os.fspath(path)
    if os.name == 'nt' and len(p) >= 240 and not p.startswith('\\\\?\\'):
        p = '\\\\?\\' + os.path.abspath(p)
    return p


def _exists(path) -> bool:
    return os.path.exists(long_path(path))


def sha256_file(path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(long_path(path), 'rb') as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _key(path) -> str:
    """One spelling per place: case, slash direction, '..' and a relative spelling do not make two
    paths. Every manifest row names absolute paths; a relative path is taken against the current
    folder (review ops-storage finding 8, 2026-09-23: `--ckpt Saved/...` from the repository root
    was returned unchanged for a checkpoint the archiver had moved)."""
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def default_manifests():
    return (C_MANIFEST,) + tuple(D_MANIFESTS)


def append_row(manifest, row: dict):
    """Append one manifest row on a line of its own. A writer that died mid-line leaves a torn
    fragment with no newline; appending straight after it would glue the next row onto the
    fragment and lose that row too, so a missing final newline is written first."""
    manifest = Path(manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(long_path(manifest), 'a+b') as fh:
        fh.seek(0, os.SEEK_END)
        if fh.tell() > 0:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b'\n':
                fh.seek(0, os.SEEK_END)
                fh.write(b'\n')
        fh.seek(0, os.SEEK_END)
        fh.write((json.dumps(row) + '\n').encode('utf-8'))
        fh.flush()


def _rows(manifest: Path):
    """Every parseable row of one manifest, in file order. A torn last line (a writer that
    died mid-append) is skipped, never allowed to hide the rows before it."""
    try:
        text = Path(manifest).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


# (manifest key) -> (size, mtime_ns, table): a manifest is parsed once per change, not once per
# lookup. Measured 2026-09-23 after the C: mirror was backfilled (2,036 rows, 1.36 MB beside the
# D: manifests' 1.04 MB): an unknown path took 29-36 ms to resolve, read in full every call.
_TABLES = {}


def moves_in(manifest: Path) -> dict:
    """{source key: move} for one manifest, the latest event per source winning.

    A move is dict(source, destination, kind 'file'|'folder', sha256 (file) or files
    {relative path: sha256} (folder), manifest). The archiver's checkpoint row carries two moves:
    the checkpoint folder and its companion file. A 'restored' row (offload_storage) cancels the
    move before it; a 'planned' row moves nothing and is ignored. The table is cached until the
    file's size or modification time changes (every writer appends, which changes both)."""
    try:
        st = os.stat(long_path(manifest))
        stamp = (st.st_size, st.st_mtime_ns)
    except OSError:
        return {}
    key = _key(manifest)
    hit = _TABLES.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    table = _moves_in(manifest)
    _TABLES[key] = (stamp, table)
    return table


def _moves_in(manifest: Path) -> dict:
    found = {}
    for row in _rows(manifest):
        event = row.get('event')
        if event not in ('moved', 'restored'):
            continue
        pairs = []
        if row.get('source') and row.get('destination'):
            if isinstance(row.get('files'), dict):
                pairs.append(dict(source=row['source'], destination=row['destination'],
                                  kind='folder', files=row['files']))
            else:
                pairs.append(dict(source=row['source'], destination=row['destination'],
                                  kind=row.get('kind', 'file'), sha256=row.get('sha256'),
                                  files=row.get('files')))
        if row.get('companion_source') and row.get('companion_destination'):
            pairs.append(dict(source=row['companion_source'],
                              destination=row['companion_destination'], kind='file',
                              sha256=row.get('companion_sha256')))
        for move in pairs:
            move['manifest'] = str(manifest)
            if event == 'restored':
                found.pop(_key(move['source']), None)
            else:
                found[_key(move['source'])] = move
    return found


def find_move(path, manifests=None):
    """The move that took `path` (or a folder holding it) away, and the path's new place.

    Returns (move, new_path, relative) or None. `relative` is the path inside a moved folder
    ('' when the path itself was moved)."""
    want = _key(path)
    for manifest in (default_manifests() if manifests is None else manifests):
        table = moves_in(manifest)
        if not table:
            continue
        move = table.get(want)
        if move is not None:
            return move, Path(move['destination']), ''
        # inside a moved folder: the longest moved folder that holds the path answers
        best = None
        for key, candidate in table.items():
            if candidate['kind'] == 'folder' and want.startswith(key.rstrip(os.sep) + os.sep):
                if best is None or len(key) > len(best[0]):
                    best = (key.rstrip(os.sep), candidate)
        if best is not None:
            # cut from the caller's own spelling, not the case-folded key: the manifest's file
            # names keep their case ("model.safetensors"). normcase changes no lengths; the
            # spelling is made absolute first, as the key was.
            relative = os.path.abspath(os.fspath(path))[len(best[0]) + 1:].replace('\\', '/')
            return best[1], Path(best[1]['destination']) / relative, relative
    return None


# (path, size, mtime_ns) -> sha256, so a reader that locates the same chunk twice (a tape replay
# validates its spans, then reads them) hashes it once per process
_VERIFIED = {}


def _hash_cached(path) -> str:
    st = os.stat(long_path(path))
    key = (_key(path), st.st_size, st.st_mtime_ns)
    if key not in _VERIFIED:
        _VERIFIED[key] = sha256_file(path)
    return _VERIFIED[key]


def _recorded_hash(move, relative):
    if move['kind'] == 'folder':
        return (move.get('files') or {}).get(relative) if relative else None
    return move.get('sha256')


def locate(path, *, expected_sha256=None, verify=True, manifests=None, _depth=0) -> Path:
    """The path itself when it exists; otherwise where a manifest says it went, proven.

    expected_sha256: a hash the caller already holds (a checkpoint's episode_state.json receipt).
        It must equal the manifest's recorded hash, or the move is refused.
    verify: hash the new copy against the manifest's recorded hash (a moved folder: every file
        the row lists). Pass False only when the caller hashes the file itself right after
        (continuation_state.load does, against the same receipt), so it is read once, not twice.
    """
    p = Path(path)
    if _exists(p):
        return p
    hit = find_move(p, manifests)
    if hit is None:
        return p
    move, new, relative = hit
    recorded = _recorded_hash(move, relative)
    if expected_sha256 is not None and recorded is not None and recorded != expected_sha256:
        raise MovedFileError(
            "%s moved to %s (manifest %s), but the manifest records SHA-256 %s and the caller "
            "expects %s: not the same file" % (p, new, move['manifest'], recorded[:12],
                                               expected_sha256[:12]))
    if not _exists(new):
        if _depth < 4:                       # moved again from its first destination
            again = locate(new, expected_sha256=expected_sha256, verify=verify,
                           manifests=manifests, _depth=_depth + 1)
            if _exists(again):
                return again
        raise MovedFileError("%s moved to %s (manifest %s), and nothing is there"
                             % (p, new, move['manifest']))
    if verify:
        _verify(new, move, relative, expected_sha256)
    return new


def _verify(new: Path, move, relative, expected_sha256):
    files = move.get('files') or {}
    if move['kind'] == 'folder' and relative not in files:
        # the moved folder itself, or a folder inside it: every file the row lists under it
        prefix = relative + '/' if relative else ''
        wanted = {rel[len(prefix):]: digest for rel, digest in files.items()
                  if rel.startswith(prefix)}
        if not wanted:
            raise MovedFileError("%s: the manifest %s lists no file hashes to prove the folder"
                                 % (new, move['manifest']))
        for rel, digest in wanted.items():
            f = new / rel
            if not _exists(f) or _hash_cached(f) != digest:
                raise MovedFileError("%s: %s is missing or differs from the SHA-256 the manifest "
                                     "%s recorded" % (new, rel, move['manifest']))
        return
    want = expected_sha256 or _recorded_hash(move, relative)
    if want is None:
        raise MovedFileError("%s: the manifest %s records no SHA-256 for it, so the copy cannot "
                             "be proven" % (new, move['manifest']))
    if Path(new).is_dir():
        raise MovedFileError("%s is a folder where the manifest %s recorded a file"
                             % (new, move['manifest']))
    got = _hash_cached(new)
    if got != want:
        raise MovedFileError("%s hashes %s, not the %s the manifest %s recorded"
                             % (new, got[:12], want[:12], move['manifest']))


def resolve_run_args(checkpoint, manifests=None) -> Path:
    """The args.json of the run that wrote `checkpoint`, wherever the checkpoint now lives.

    Looked for, in order: beside the checkpoint's run folder (checkpoints/<ckpt>/../../args.json,
    true for every checkpoint still on C:); then in the run folder the checkpoint's own
    lineage.json names under "out" (an archived checkpoint: the archive holds checkpoints only,
    while the run folder with its args.json stays on C:). Raises FileNotFoundError naming every
    place it looked."""
    ck = locate(checkpoint, verify=False, manifests=manifests)
    looked = []
    if ck.parent.name == 'checkpoints':
        here = ck.parent.parent / 'args.json'
        looked.append(here)
        if _exists(here):
            return here
    lineage = ck / 'lineage.json'
    if _exists(lineage):
        try:
            out = json.loads(Path(long_path(lineage)).read_text(encoding='utf-8')).get('out')
        except (OSError, ValueError):
            out = None
        if out:
            there = locate(Path(out) / 'args.json', manifests=manifests)
            looked.append(there)
            if _exists(there):
                return there
    else:
        looked.append(lineage)
    raise FileNotFoundError("no args.json for checkpoint %s; looked at %s"
                            % (checkpoint, ", ".join(str(x) for x in looked)))


def main_tree(repo=None) -> Path:
    """The main working tree of the repository that holds `repo` (default: this file's).

    Agents work in git worktrees whose Saved/ holds no data. A worktree's .git is a file
    "gitdir: <main>/.git/worktrees/<name>", and that folder's commondir file names the main .git.
    Returns `repo` itself for the main tree, or when that layout cannot be read."""
    repo = Path(REPO if repo is None else repo)
    marker = repo / '.git'
    try:
        if marker.is_dir():
            return repo
        text = marker.read_text(encoding='utf-8').strip()
    except OSError:
        return repo
    if not text.startswith('gitdir:'):
        return repo
    gitdir = Path(text[len('gitdir:'):].strip())
    if not gitdir.is_absolute():
        gitdir = repo / gitdir
    try:
        common = Path((gitdir / 'commondir').read_text(encoding='utf-8').strip())
    except OSError:
        return repo
    main = Path(os.path.abspath(common if common.is_absolute() else gitdir / common)).parent
    return main if (main / '.git').is_dir() else repo


def find_in_trees(relative, *, also=(), verify=True, repo=None, manifests=None):
    """Find a repository-relative path (a checkpoint under Saved/) wherever it is now.

    WHY (2026-09-26): the archiver moved DESIGN D3's baseline checkpoint 344,400 to D: on Sept 23,
    and eight test files that looked for it only at its C: path skipped their checkpoint proofs
    from then on, silently. Places looked, in order: each path in `also` (a caller's override,
    such as an environment variable), this tree's copy, the main working tree's copy (a worktree
    holds no Saved/ data), then where a storage move's manifest says one of those spellings went
    (locate(), proven by hash unless verify=False). The manifests read are this tree's and the main
    tree's C: mirror and the D: manifests, unless `manifests` is given.

    Returns (path, looked). `path` is None when the path is nowhere; `looked` names every place and
    manifest searched, for the caller's "not found" message. A manifest that knows the path but
    whose copy is missing or differs raises MovedFileError: a damaged archive, never "absent"."""
    repo = Path(REPO if repo is None else repo)
    trees = [repo]
    main = main_tree(repo)
    if _key(main) != _key(repo):
        trees.append(main)
    candidates = [Path(a) for a in also if a] + [t / relative for t in trees]
    if manifests is None:
        manifests = []
        for m in ([t / 'Saved' / 'mimo_bridge' / 'storage_moves.jsonl' for t in trees]
                  + list(default_manifests())):
            if _key(m) not in {_key(x) for x in manifests}:
                manifests.append(m)
    looked = [str(p) for p in candidates]
    for p in candidates:
        if _exists(p):
            return p, looked
    for p in candidates:
        found = locate(p, verify=verify, manifests=manifests)
        if _key(found) != _key(p) and _exists(found):
            return found, looked
    looked.append("the move records %s, none of which names it"
                  % ", ".join(str(m) for m in manifests))
    return None, looked
