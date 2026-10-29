# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""H84 file ownership, launch admission, close-out and read-only scratch inventory.

No operation in this module deletes, moves, or executes an inventoried file.
The registry remains the run authority. This module records where output belongs.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import tempfile
import time

REPO = Path(__file__).resolve().parents[4]
RECORDS = REPO / "docs/gain_of_function_v4/coordinator/work_items"
REGISTRY = REPO / "docs/gain_of_function_v4/test_registry.json"
STORE = REPO / "Saved/gof_coordination/file_lifecycle"
ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,119}$")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


# Windows refuses to replace a file another process holds open without delete sharing (WinError 5; 32 is the sharing
# violation); a plain reader such as read() opens a record that way for milliseconds. Run u's test runner got WinError
# 5 at its start claim on Oct 25, 2026; a probe reproduced that error while a reader held the target, a possible cause.
# The process that held the record then is unknown.
REPLACE_SHARING_ERRORS = (5, 32)
REPLACE_WAIT_S = 2.0


def _replace(temporary, path, wait_s=None):
    """os.replace: the first attempt at once; retried only on a Windows sharing refusal, each retry begun only while
    the bound (wait_s, at most REPLACE_WAIT_S) has not passed; then the last error is raised, so a lasting denial is
    not hidden. The target keeps its previous contents until a replace succeeds."""
    end = time.monotonic() + (REPLACE_WAIT_S if wait_s is None else wait_s)
    delay = .02
    while True:
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in REPLACE_SHARING_ERRORS or time.monotonic() + delay > end:
                raise
            time.sleep(delay)
            if time.monotonic() > end:          # a late wake-up begins no replacement after the bound
                raise
            delay = min(delay * 2, .25)


def write(path, data, replace_wait_s=None):
    """replace_wait_s: the caller's remaining time for a refused replace; at most REPLACE_WAIT_S is used."""
    if replace_wait_s is not None:
        replace_wait_s = max(0.0, min(float(replace_wait_s), REPLACE_WAIT_S))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        _replace(temporary, path, replace_wait_s)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)  # Only our own incomplete metadata write.


@contextlib.contextmanager
def record_lock(path):
    """A process lock, released by the OS on crash; no stale PID lock to erase."""
    path = Path(path).with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        end = time.monotonic() + 10
        while True:
            try:
                f.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > end:
                    raise TimeoutError("file record is being updated: " + str(path))
                time.sleep(.05)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def valid_id(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError("work and attempt IDs use letters, numbers, hyphens and underscores only")
    return value


def link(path):
    try:
        s = Path(path).lstat()
        return stat.S_ISLNK(s.st_mode) or bool(getattr(s, "st_file_attributes", 0) & 0x400)
    except FileNotFoundError:
        return False


def clean_absolute(value):
    p = Path(value)
    if not p.is_absolute() or ".." in p.parts:
        raise ValueError("output must be an absolute path without '..': " + str(value))
    for ancestor in [p, *p.parents]:
        if link(ancestor):
            raise ValueError("links and junctions cannot be output paths: " + str(ancestor))
    return p.resolve()


def inside(path, parent):
    try:
        clean_absolute(path).relative_to(clean_absolute(parent))
        return True
    except (ValueError, OSError):
        return False


def locations(policy, record):
    root = clean_absolute(record["root"])
    allowed = [clean_absolute(p) for p in policy["managed_roots"]]
    if root not in allowed:
        raise ValueError("output root is not one of the registered managed roots")
    base = root / valid_id(record["work_id"]) / "attempts" / valid_id(record["attempt_id"])
    evidence, temporary = base / "evidence", base / "tmp"
    for key, expected in [("evidence_dir", evidence), ("tmp_dir", temporary)]:
        if clean_absolute(record[key]) != clean_absolute(expected):
            raise ValueError(key + " must be exactly " + str(expected))
    return base, evidence, temporary


def manifest_path(policy, work_id):
    directory = clean_absolute(policy["manifest_dir"])
    path = directory / (valid_id(work_id) + ".json")
    clean_absolute(path)
    return path


def admission(row, policy, *, fresh=True):
    """Return a plain refusal reason, never create output directories at admission."""
    if not policy:
        return None
    try:
        if not isinstance(policy, dict) or policy.get("version") != 1:
            raise ValueError("missing or unsupported file-lifecycle policy version")
        record = row.get("file_lifecycle")
        if not isinstance(record, dict):
            raise ValueError("register its work/attempt and output folders with file_lifecycle prepare")
        base, evidence, temporary = locations(policy, record)
        manifest = read(manifest_path(policy, record["work_id"]))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("managed_attempts"), list):
            raise ValueError("work record must be an object with a managed_attempts list")
        if not all(isinstance(a, dict) for a in manifest["managed_attempts"]):
            raise ValueError("every attempt record must be an object")
        for field in ["purpose", "owner"]:
            if not str(manifest.get(field, "")).strip():
                raise ValueError("work record lacks " + field)
        attempts = [a for a in manifest.get("managed_attempts", [])
                    if a.get("attempt_id") == record["attempt_id"]]
        if len(attempts) != 1 or attempts[0].get("row_id") != row["id"]:
            raise ValueError("attempt is missing, duplicated or belongs to another row")
        attempt = attempts[0]
        if attempt["declaration"] != record:
            raise ValueError("row and work record disagree about the output declaration")
        if not isinstance(record.get("generator"), list) or not record["generator"] or not all(
                isinstance(x, str) and x.strip() for x in record["generator"]):
            raise ValueError("record the generator as a nonempty command argument list")
        if not str(record.get("retention", "")).strip():
            raise ValueError("record why evidence is kept and when temporary output is reviewed")
        result = row.get("result_source", {}).get("file")
        if not result or not inside(result, evidence):
            raise ValueError("the row's result file must be inside its evidence folder")
        if fresh and (attempt.get("state") != "planned" or base.exists()):
            raise ValueError("attempt already used; register a new attempt ID without overwriting evidence")
        return None
    except (AttributeError, KeyError, TypeError, ValueError, OSError) as exc:
        return "file lifecycle: " + str(exc)


def runner_refusal(row, data):
    if "file_lifecycle_policy" in data and (not isinstance(data["file_lifecycle_policy"], dict)
                                           or data["file_lifecycle_policy"].get("version") != 1):
        return "file lifecycle: configured policy is malformed; restore its declared version and roots"
    return admission(row, data.get("file_lifecycle_policy"))


def claim_start(row, policy, run_id, command):
    """Atomically claim an unused attempt and bind the actual argv before execution."""
    if not policy:
        return
    why = admission(row, policy)
    if why:
        raise ValueError(why)
    record = row["file_lifecycle"]
    if command != record["generator"]:
        raise ValueError("actual command differs from the registered generator argument list")
    path = manifest_path(policy, record["work_id"])
    with record_lock(path):
        why = admission(row, policy)
        if why:
            raise ValueError(why)
        manifest = read(path)
        attempt = next(a for a in manifest["managed_attempts"] if a["attempt_id"] == record["attempt_id"])
        attempt.update(state="claimed", run_id=run_id, claimed_utc=now())
        write(path, manifest)


def close_from_registry(row, policy):
    """Idempotent metadata close-out after the registry has recorded the result."""
    last = row.get("last_run", {})
    binding = last.get("file_lifecycle")
    if not policy or not binding:
        return False
    if not last.get("run_id") or not last.get("ended"):
        raise ValueError("registry has no completed run identity")
    completed_row = dict(row, file_lifecycle=binding["declaration"], result_source=binding["result_source"])
    why = admission(completed_row, policy, fresh=False)
    if why:
        raise ValueError(why)
    record = binding["declaration"]
    path = manifest_path(policy, record["work_id"])
    with record_lock(path):
        manifest = read(path)
        attempt = next(a for a in manifest["managed_attempts"] if a["attempt_id"] == record["attempt_id"])
        if attempt.get("run_id") != last["run_id"]:
            raise ValueError("completed run does not match the attempt's start claim")
        attempt.update(state="closed", run_id=last["run_id"], ended=last["ended"],
                       outcome=row.get("outcome"), result_source=binding["result_source"],
                       source_identity={k:last.get(k) for k in ["commit", "commit_dirty", "checkpoint", "result_file_record"]},
                       temporary_review="Owner checks dependencies and cleanup receipt at work closure; no deletion authorized.")
        manifest["updated_utc"] = now()
        write(path, manifest)
    return True


def failed_start(row, policy, run_id, reason):
    if not policy:
        return
    record = row["file_lifecycle"]
    path = manifest_path(policy, record["work_id"])
    with record_lock(path):
        manifest = read(path)
        attempt = next(a for a in manifest["managed_attempts"] if a["attempt_id"] == record["attempt_id"])
        if attempt.get("run_id") == run_id and attempt.get("state") == "claimed":
            attempt.update(state="start_record_failed", reason=reason, ended_utc=now())
            write(path, manifest)


def prepare(registry, work_id, attempt_id, row_id, purpose, owner, generator, root):
    """Register metadata only; the admitted runner creates evidence/tmp directories."""
    from mimo_bridge.tools import test_registry
    data = read(registry)
    policy = data["file_lifecycle_policy"]
    valid_id(work_id)
    valid_id(attempt_id)
    base = clean_absolute(root) / work_id / "attempts" / attempt_id
    record = dict(work_id=work_id, attempt_id=attempt_id, root=str(clean_absolute(root)),
                  evidence_dir=str(base / "evidence"), tmp_dir=str(base / "tmp"), generator=generator,
                  retention="Keep result evidence. Review regenerable temporary files at work closure; no automatic deletion.")
    locations(policy, record)
    if base.exists():
        raise ValueError("attempt path already exists; select a new attempt ID")
    rows = [r for r in data["rows"] if r.get("id") == row_id]
    if len(rows) != 1:
        raise ValueError("exactly one registered test row must already exist")
    row = rows[0]
    if row.get("status") == "running":
        raise ValueError("cannot replace a running attempt")
    if not inside(row.get("result_source", {}).get("file", ""), base / "evidence"):
        raise ValueError("first set the row's result_source.file inside " + str(base / "evidence"))
    path = manifest_path(policy, work_id)
    with record_lock(path):
        manifest = read(path) if path.exists() else dict(schema_version=1, work_id=work_id,
                        purpose=purpose, owner=owner, source_worktree=str(REPO), registry=str(registry), registry_ids=[], path_groups=[])
        if not purpose.strip() or not owner.strip():
            raise ValueError("purpose and owner are required")
        if row_id not in manifest.setdefault("registry_ids", []):
            manifest["registry_ids"].append(row_id)
        attempts = manifest.setdefault("managed_attempts", [])
        if any(a["attempt_id"] == attempt_id for a in attempts):
            raise ValueError("attempt ID is already registered; do not replace its declaration")
        attempts.append(dict(attempt_id=attempt_id, row_id=row_id, state="planned", purpose=purpose,
                             owner=owner, created_utc=now(), declaration=record))
        manifest["updated_utc"] = now()
        write(path, manifest)
    # A partial failure leaves a planned record, not permission to run unregistered output.
    if not test_registry.update_row(row_id, {"file_lifecycle": record}, path=registry):
        raise ValueError("test row changed during registration; reconcile the planned record")
    return record


def documented_groups():
    groups, issues = [], []
    for path in sorted(RECORDS.glob("*.json")):
        try:
            doc = read(path)
            for group in doc.get("path_groups", []):
                raw = group.get("path", "")
                # Narrative expressions are evidence for humans, not path authority.
                if not isinstance(raw, str) or " and " in raw or " (" in raw:
                    issues.append(str(path.name) + ": narrative path needs explicit paths: " + str(raw))
                    continue
                p = Path(raw) if Path(raw).is_absolute() else REPO / raw
                if not inside(p, Path("D:/UEMA_Research/external/D/UEMA_scratch")):
                    continue
                groups.append((str(p.resolve()).casefold(), group.get("class", "documented_group"),
                    group.get("purpose", ""), group.get("owner", doc.get("owner", "")),
                    str(path.relative_to(REPO)), group.get("retention", "Keep pending review")))
            for attempt in doc.get("managed_attempts", []):
                a = attempt["declaration"]
                for key, kind in [("evidence_dir", "evidence"), ("tmp_dir", "temporary")]:
                    groups.append((str(Path(a[key]).resolve()).casefold(), kind, attempt["purpose"],
                        attempt["owner"], str(path.relative_to(REPO)), a["retention"]))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(str(path) + ": " + str(exc))
    return sorted(groups, key=lambda x: len(x[0]), reverse=True), issues


def inventory(root, store=STORE, retry_errors=False):
    """One metadata scan; no junction traversal, file execution or bulk content reads."""
    root = clean_absolute(root)
    store = clean_absolute(store)
    if inside(store, root):
        raise ValueError("inventory storage must be outside the scanned tree")
    store.mkdir(parents=True, exist_ok=True)
    groups, issues = documented_groups()
    source = REPO / "docs/h84_tool_accounting_2026-10-17_data.js"
    text = source.read_text(encoding="utf-8-sig")
    accounting = json.loads(text[text.index("{"):].rstrip(";\r\n "))
    old = {str(Path(p).resolve()).casefold() for p in accounting["uncovered_scratch"]["paths"]}
    sampled = {str(Path(x["path"]).resolve()).casefold(): x for x in
               read(REPO / "Saved/gof_coordination/reviews/scratch_sample_20261020.json")["files"]}
    known_tools = {str(Path(f["path"]).resolve()).casefold(): f for f in accounting["files"]
                   if str(f["path"]).lower().startswith("d:")}
    fd, temp = tempfile.mkstemp(prefix="inventory-", suffix=".sqlite", dir=store)
    os.close(fd)
    connection = sqlite3.connect(temp)
    errors, count, total = [], 0, 0
    try:
        stack = [root]
        if retry_errors:
            previous = read(store / "summary.json")
            if clean_absolute(previous["root"]) != root:
                raise ValueError("retry root does not match existing inventory")
            stack = [Path(e["path"]) for e in previous["scan_errors"]]
            if any(not inside(p, root) for p in stack):
                raise ValueError("retry path is outside inventory root or is a link")
            old_db = sqlite3.connect(store / "inventory.sqlite")
            try:
                old_db.backup(connection)
            finally:
                old_db.close()
        connection.execute("CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, bytes INTEGER, modified_utc TEXT, created_utc TEXT, folder TEXT, category TEXT, purpose TEXT, owner TEXT, record TEXT, retention TEXT, old_uncovered INTEGER) WITHOUT ROWID")
        while stack:
            directory = stack.pop()
            try:
                native = "\\\\?\\" + str(directory) if os.name == "nt" else str(directory)
                entries = list(os.scandir(native))
            except OSError as exc:
                errors.append({"path": str(directory), "error": str(exc)})
                continue
            for entry in entries:
                p = directory / entry.name
                try:
                    s = entry.stat(follow_symlinks=False)
                    if stat.S_ISLNK(s.st_mode) or getattr(s, "st_file_attributes", 0) & 0x400:
                        errors.append({"path": str(p), "error": "link/junction skipped"})
                        continue
                    if stat.S_ISDIR(s.st_mode):
                        stack.append(p)
                        continue
                    key = str(p.resolve()).casefold()
                    category, purpose, owner, record, retention = (
                        "unidentified", "Purpose not yet verified", "the builder",
                        "Legacy inventory: this record", "Keep in place; identify before reuse or any removal")
                    for prefix, kind, why, who, ref, keep in groups:
                        if key == prefix or key.startswith(prefix + os.sep):
                            category, purpose, owner, record, retention = "documented_" + kind, why, who, ref, keep
                            if kind == "protected_state":
                                category = "protected_state"
                            break
                    if key in known_tools and category != "protected_state":
                        f = known_tools[key]
                        tool = accounting["tools"].get(f.get("tool"), {})
                        category, purpose, record = "accounted_tool", tool.get("does", f.get("note", "Listed tool")), str(source.relative_to(REPO))
                    if key in sampled and category != "protected_state":
                        sample = sampled[key]
                        if s.st_size == sample["bytes"] and hashlib.sha256(p.read_bytes()).hexdigest() == sample["sha256"]:
                            category, purpose, record = "inspected_sample", sample["purpose"], "Saved/gof_coordination/reviews/scratch_sample_20261020.json"
                    # Checkpoints and recordings never become temporary merely from parent names.
                    if any(part.casefold() in ("checkpoints", "checkpoint_state", "recordings") for part in p.parts) or p.name in ("model.safetensors", "optimizer.pt", "runtime_state.pt"):
                        category, retention = "protected_state", "Preserve learned state/world companions; no cleanup eligibility"
                    iso = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat()
                    connection.execute("INSERT OR REPLACE INTO files VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
                        str(p), s.st_size, iso(s.st_mtime), iso(s.st_ctime), p.relative_to(root).parts[0],
                        category, purpose, owner, record, retention, int(key in old)))
                    count += 1
                    total += s.st_size
                except (OSError, ValueError) as exc:
                    errors.append({"path": str(p), "error": str(exc)})
        connection.commit()
        count, total = connection.execute("SELECT count(*),coalesce(sum(bytes),0) FROM files").fetchone()
        categories = dict(connection.execute("SELECT category,count(*) FROM files GROUP BY category"))
        old_present = connection.execute("SELECT count(*) FROM files WHERE old_uncovered=1").fetchone()[0]
        folders = [dict(zip(["folder", "files", "bytes", "unidentified", "old_uncovered"], row)) for row in
            connection.execute("SELECT folder,count(*),sum(bytes),sum(category='unidentified'),sum(old_uncovered) FROM files GROUP BY folder ORDER BY folder")]
        summary = dict(scanned_utc=now(), root=str(root), files=count, bytes=total, complete=not errors,
                       scan_errors=errors, manifest_issues=issues, categories=categories, folders=folders,
                       old_uncovered_count=len(old), old_uncovered_present=old_present,
                       old_uncovered_not_observed=len(old)-old_present,
                       rule="Purpose by cited records, not filename guesses. Unidentified files stay in place. Dates never authorize deletion.")
        if retry_errors:
            summary['full_scan_utc'] = previous.get('full_scan_utc', previous['scanned_utc'])
            summary['refresh_scope'] = 'Retried only previous error directories; other rows retain full-scan dates.'
        with (store / "inventory.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            # The requested legacy list, not a second copy of all half-million rows.
            cursor = connection.execute("SELECT * FROM files WHERE old_uncovered=1 ORDER BY path")
            writer.writerow([c[0] for c in cursor.description])
            writer.writerows(cursor)
        connection.close()
        os.replace(temp, store / "inventory.sqlite")
        write(store / "summary.json", summary)
        return summary
    finally:
        connection.close()
        if os.path.exists(temp):
            os.unlink(temp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for key in ["work", "attempt", "row", "purpose", "owner", "root", "generator_json"]:
        p.add_argument("--" + key.replace("_", "-"), required=True)
    p.add_argument("--registry", default=str(REGISTRY))
    p = sub.add_parser("check")
    p.add_argument("--row", required=True)
    p.add_argument("--registry", default=str(REGISTRY))
    p = sub.add_parser("reconcile")
    p.add_argument("--registry", default=str(REGISTRY))
    p = sub.add_parser("inventory")
    p.add_argument("--root", default="D:/UEMA_Research/external/D/UEMA_scratch")
    p.add_argument("--retry-errors", action="store_true")
    p = sub.add_parser("find")
    p.add_argument("text", help="part of an existing path, for example a folder or filename")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.registry, args.work, args.attempt, args.row, args.purpose,
                                 args.owner, json.loads(args.generator_json), args.root), indent=2))
    elif args.command == "check":
        data = read(args.registry)
        rows = [r for r in data["rows"] if r["id"] == args.row]
        why = "row missing or duplicated" if len(rows) != 1 else runner_refusal(rows[0], data)
        print(why or "PASS: output ownership, purpose, fresh attempt and result path verified")
        return 1 if why else 0
    elif args.command == "reconcile":
        data = read(args.registry)
        errors, closed = [], 0
        for row in data["rows"]:
            if row.get("file_lifecycle") and row.get("last_run", {}).get("ended"):
                try:
                    closed += int(close_from_registry(row, data.get("file_lifecycle_policy")))
                except Exception as exc:
                    errors.append({"row": row["id"], "error": str(exc)})
        print(json.dumps({"closed": closed, "errors": errors}))
        return 1 if errors else 0
    elif args.command == "find":
        connection = sqlite3.connect(STORE / "inventory.sqlite")
        try:
            connection.row_factory = sqlite3.Row
            found = connection.execute("SELECT * FROM files WHERE path LIKE ? ORDER BY path LIMIT 21", ('%'+args.text+'%',)).fetchall()
            print(json.dumps({"matches": [dict(r) for r in found[:20]], "more_than_twenty": len(found)>20}, indent=2))
        finally:
            connection.close()
    else:
        result = inventory(args.root, retry_errors=args.retry_errors)
        print(json.dumps({k: result[k] for k in ["files", "bytes", "complete", "categories", "old_uncovered_present"]}))
        return 0 if result["complete"] else 1
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(REPO / "Source/Python"))
    raise SystemExit(main())
