# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The observer validation's pass/fail lines, locked: only the operator changes them.

Operator, 2026-09-25: "lock the pass/fail lines so the agent building a feature cannot edit them.
Right now the same agents that change the observer's detectors also rewrite its file of expected
answers, golden.json." Until then `validate --write-golden` overwrote golden.json with whatever the
detectors of the moment produced, and each case's expected-present / expected-absent lines and the
KNOWN_DEFECTS register (which turns a failure into a "known" one) lived in validate.py beside the
detector code, so one agent could change a detector and the answer it is judged against together.

THE LOCKED SET (docs/gain_of_function_v4/observer_validation/locked/):
  expectations.json  for every approved case: expected present, expected absent, the scripted truth
                     values, and the case's recorded-trajectory fingerprint (qpos_hash: a changed
                     scenario, such as a box moved closer to make a case pass, is not the same case);
                     plus the KNOWN_DEFECTS and BOUNDARY registers
  golden.json        the detections each approved case's recording produced when it was approved
  lock.json          the SHA-256 of both files (of their canonical JSON, so line endings and
                     indentation do not matter), who approved them and when, and the history
Validation reads its pass/fail lines ONLY from this set, after checking both files against
lock.json (verify). A case in validate.CASES that the locked set lacks is run and reported as NOT
APPROVED and never counts; a locked case missing from CASES fails.

WHO CHANGES WHAT
  builders (agents)  change detectors and case scripts, run validation, and write PROPOSALS
                     (propose(): docs/gain_of_function_v4/observer_validation/proposals/<id>/, never
                     the locked folder; `validate --write-golden` now proposes).
  the operator       approves a proposal from their OWN terminal:
                         python -m mimo_bridge.tools.observer.validation_lock approve <proposal folder>
                     (or `approve --current` for the set as it stands). approve() refuses unless stdin
                     and stdout are an interactive console, and asks for the proposal id and the
                     approver's name typed in. It writes the locked set and lock.json and commits them.
Enforcement besides this code is the operator's own: editor settings that deny file edits under the
locked folder and a shell guard that refuses commands writing there or running `approve`.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / "docs" / "gain_of_function_v4" / "observer_validation"
LOCKED = BASE / "locked"
PROPOSALS = BASE / "proposals"
FILES = ("expectations.json", "golden.json")
SCHEMA = "observer_validation_lock_v1"


class LockError(RuntimeError):
    """The locked pass/fail lines are missing or differ from their lock record."""


def canonical_sha256(obj) -> str:
    """SHA-256 of an object's canonical JSON (sorted keys, no whitespace): line endings, indentation
    and key order do not change it."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def verify(folder: Path = LOCKED) -> dict:
    """The locked set, checked against lock.json: {'expectations', 'golden', 'lock'}. Raises
    LockError when a file is missing or its canonical SHA-256 differs from the recorded one."""
    folder = Path(folder)
    lock_path = folder / "lock.json"
    if not lock_path.is_file():
        raise LockError("no locked pass/fail lines: %s is missing" % lock_path)
    try:
        lock = _read(lock_path)
    except (OSError, ValueError) as error:
        raise LockError("lock.json cannot be read (%s): the pass/fail lines fail closed" % error)
    if lock.get("schema") != SCHEMA:
        raise LockError("lock.json has schema %r, not %r" % (lock.get("schema"), SCHEMA))
    out = dict(lock=lock)
    for name in FILES:
        path = folder / name
        if not path.is_file():
            raise LockError("locked file missing: %s" % path)
        try:
            obj = _read(path)
        except (OSError, ValueError) as error:
            raise LockError("%s cannot be read (%s): the pass/fail lines fail closed" % (name, error))
        want = (lock.get("files") or {}).get(name)
        got = canonical_sha256(obj)
        if want != got:
            raise LockError("%s differs from its lock record (recorded %s, now %s): the pass/fail lines "
                            "were changed without the operator's approval" % (name, str(want)[:16], got[:16]))
        out[name.split(".")[0]] = obj
    return out


def uncommitted(folder: Path = LOCKED) -> Optional[List[str]]:
    """The locked folder's files that differ from the last commit (git status lines), [] when none,
    None when git cannot tell (no repository). An approval commits the locked set at once, so a
    difference means someone wrote there outside approve() - even with lock.json rewritten to match."""
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", str(Path(folder))],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode:
        return None
    return [line for line in r.stdout.splitlines() if line.strip()]


def expectations_from_code(cases, known_defects: Dict[str, str], boundary: Dict[str, tuple],
                           qpos_hashes: Optional[Dict[str, str]] = None) -> dict:
    """The expectations a builder's code holds now (validate.CASES and its registers), in the locked
    file's form: what a proposal would lock."""
    out = dict(cases={}, known_defects=dict(known_defects),
               boundary={k: dict(detectors=list(v[0]), what=v[1]) for k, v in boundary.items()})
    for name, _build, present, absent, truth in cases:
        row = dict(present=list(present), absent=list(absent), truth=dict(truth))
        if qpos_hashes and name in qpos_hashes:
            row["qpos_hash"] = qpos_hashes[name]
        out["cases"][name] = row
    return out


def differences(old: dict, new: dict) -> dict:
    """What a proposal changes against the locked set, case by case."""
    oc, nc = (old or {}).get("cases", {}), (new or {}).get("cases", {})
    changed = {}
    for name in sorted(set(oc) & set(nc)):
        fields = [k for k in ("present", "absent", "truth", "qpos_hash") if oc[name].get(k) != nc[name].get(k)]
        if fields:
            changed[name] = {k: dict(locked=oc[name].get(k), proposed=nc[name].get(k)) for k in fields}
    return dict(added=sorted(set(nc) - set(oc)), removed=sorted(set(oc) - set(nc)), changed=changed,
                known_defects=dict(locked=(old or {}).get("known_defects", {}),
                                   proposed=(new or {}).get("known_defects", {}))
                if (old or {}).get("known_defects") != (new or {}).get("known_defects") else None)


def golden_differences(old: dict, new: dict) -> dict:
    oc, nc = (old or {}).get("cases", {}), (new or {}).get("cases", {})
    changed = {}
    for name in sorted(set(oc) & set(nc)):
        a, b = oc[name], nc[name]
        if a.get("qpos_hash") != b.get("qpos_hash"):
            changed[name] = dict(change="a different recording (the scenario or the simulator changed)")
        elif a.get("detections") != b.get("detections"):
            changed[name] = dict(change="same recording, different detections",
                                 only_locked=sorted(set(a["detections"]) - set(b["detections"])),
                                 only_proposed=sorted(set(b["detections"]) - set(a["detections"])))
    return dict(added=sorted(set(nc) - set(oc)), removed=sorted(set(oc) - set(nc)), changed=changed)


def propose(expectations: Optional[dict], golden: Optional[dict], why: str, by: str,
            proposals: Path = PROPOSALS, locked: Path = LOCKED) -> Path:
    """Write a proposal beside the locked set, never into it: the proposed files (a missing one keeps
    the locked file) and proposal.json with what changes and why. Returns its folder."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    pid = "%s_%s" % (stamp, hashlib.sha256(("%s|%s|%f" % (why, by, time.time())).encode()).hexdigest()[:6])
    folder = Path(proposals) / pid
    folder.mkdir(parents=True, exist_ok=False)
    try:
        current = verify(locked)
    except LockError:
        current = dict(expectations=None, golden=None)
    summary = dict(id=pid, why=why, proposed_by=by, made=time.time(),
                   made_clock=time.strftime("%Y-%m-%d %I:%M %p"))
    if expectations is not None:
        _write(folder / "expectations.json", expectations)
        summary["expectations"] = differences(current.get("expectations"), expectations)
    if golden is not None:
        _write(folder / "golden.json", golden)
        summary["golden"] = golden_differences(current.get("golden"), golden)
    _write(folder / "proposal.json", summary)
    return folder


def _interactive() -> bool:
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def _commit(paths: List[Path], message: str) -> Optional[str]:
    """Commit the locked files alone, race-safe (another session may commit at the same time): a
    private index built from HEAD, commit-tree, update-ref only if HEAD did not move."""
    def git(*args, env=None):
        r = subprocess.run(["git", *args], cwd=str(ROOT), env=env, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("git %s: %s" % (" ".join(args[:2]), (r.stderr or r.stdout).strip()))
        return r.stdout.strip()
    index = LOCKED.parent / ".lock_index"
    env = dict(os.environ, GIT_INDEX_FILE=str(index))
    try:
        head = git("rev-parse", "HEAD")
        git("read-tree", head, env=env)
        git("add", "--", *[str(p.relative_to(ROOT)) for p in paths], env=env)
        tree = git("write-tree", env=env)
        new = git("commit-tree", tree, "-p", head, "-m", message)
        git("update-ref", "HEAD", new, head)
        git("reset", "-q", "--", *[str(p.relative_to(ROOT)) for p in paths])
        return new
    finally:
        if index.exists():
            index.unlink()


def approve(proposal: Optional[Path] = None, current: bool = False, commit: bool = True,
            ask=input) -> dict:
    """Make a proposal the locked set (or approve the locked set as it stands, current=True). Only
    from an interactive console: the id and the approver's name are typed in."""
    if not _interactive():
        raise LockError("approval needs the operator's own interactive console (stdin and stdout are not "
                        "a terminal here); run it yourself in a terminal: python -m "
                        "mimo_bridge.tools.observer.validation_lock approve ...")
    if current:
        state = verify(LOCKED)
        pid = "current_" + state["lock"]["files"]["expectations.json"][:8]
        print("Approving the locked set as it stands: %d cases, recorded %s by %s." % (
            len(state["expectations"]["cases"]), state["lock"].get("created_clock"), state["lock"].get("created_by")))
        files = {n: LOCKED / n for n in FILES}
    else:
        proposal = Path(proposal)
        info = _read(proposal / "proposal.json")
        pid = info["id"]
        if (proposal / "approved.json").is_file():
            raise LockError("proposal %s was approved already" % pid)
        files = {n: (proposal / n if (proposal / n).is_file() else LOCKED / n) for n in FILES}
        try:
            now_locked = verify(LOCKED)
        except LockError as error:
            if any(src == LOCKED / n for n, src in files.items()):
                raise LockError("the proposal keeps a locked file that fails its check (%s); a proposal "
                                "carrying both files is needed" % error)
            now_locked = dict(expectations=None, golden=None)
        # the change against the locked set as it is NOW: a proposal made before the last approval
        # would otherwise undo that approval without showing it
        change = dict(id=pid, why=info.get("why"), proposed_by=info.get("proposed_by"),
                      made_clock=info.get("made_clock"))
        if (proposal / "expectations.json").is_file():
            change["expectations_against_the_locked_set_now"] = differences(
                now_locked.get("expectations"), _read(proposal / "expectations.json"))
        if (proposal / "golden.json").is_file():
            change["golden_against_the_locked_set_now"] = golden_differences(
                now_locked.get("golden"), _read(proposal / "golden.json"))
        print(json.dumps(change, indent=1)[:20000])
    typed = ask("Type the id %s to approve it (anything else refuses): " % pid).strip()
    if typed != pid:
        raise LockError("not approved: the typed id did not match")
    name = ask("Your name: ").strip()
    if not name:
        raise LockError("not approved: no approver name")
    try:
        previous = _read(LOCKED / "lock.json")
    except (OSError, ValueError):
        previous = {}
    LOCKED.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for n, src in files.items():
        obj = _read(src)
        if Path(src).resolve() != (LOCKED / n).resolve():
            _write(LOCKED / n, obj)
        hashes[n] = canonical_sha256(obj)
    now = time.time()
    record = dict(approved_by=name, approved_at=now, approved_clock=time.strftime("%Y-%m-%d %I:%M %p"),
                  proposal=pid, files=hashes)
    lock = dict(schema=SCHEMA, files=hashes, approved=True, approved_by=name, approved_at=now,
                approved_clock=record["approved_clock"], proposal=pid,
                created_by=previous.get("created_by"), created_clock=previous.get("created_clock"),
                history=(previous.get("history") or []) + [record])
    _write(LOCKED / "lock.json", lock)
    if proposal is not None:
        _write(Path(proposal) / "approved.json", record)
    commit_id = None
    if commit:
        try:
            commit_id = _commit([LOCKED / n for n in FILES] + [LOCKED / "lock.json"],
                                "Operator %s approves observer validation's pass/fail lines (%s)" % (name, pid))
        except RuntimeError as error:
            raise LockError("the locked files are written but the commit failed (%s); validation fails "
                            "closed until they are committed: run approve --current again" % error)
    return dict(approved=pid, by=name, commit=commit_id)


def init_from(expectations: dict, golden: dict, by: str, source: str, folder: Path = LOCKED) -> dict:
    """The first locked set, NOT approved: written once, when no lock exists, from the state the
    caller names. Validation reports it as awaiting the operator's approval until approve --current."""
    folder = Path(folder)
    if (folder / "lock.json").exists():
        raise LockError("a locked set exists already; changes go through a proposal and approve")
    _write(folder / "expectations.json", expectations)
    _write(folder / "golden.json", golden)
    lock = dict(schema=SCHEMA, files={"expectations.json": canonical_sha256(expectations),
                                      "golden.json": canonical_sha256(golden)},
                approved=False, approved_by=None, approved_at=None, proposal=None,
                created_by=by, created_at=time.time(), created_clock=time.strftime("%Y-%m-%d %I:%M %p"),
                source=source, history=[])
    _write(folder / "lock.json", lock)
    return lock


def status() -> dict:
    try:
        state = verify(LOCKED)
        lock = state["lock"]
        out = dict(intact=True, approved=bool(lock.get("approved")), approved_by=lock.get("approved_by"),
                   approved_clock=lock.get("approved_clock"), cases=len(state["expectations"]["cases"]),
                   known_defects=len(state["expectations"].get("known_defects") or {}))
    except LockError as error:
        out = dict(intact=False, error=str(error))
    out["uncommitted"] = uncommitted()
    out["pending_proposals"] = sorted(p.name for p in PROPOSALS.glob("*") if (p / "proposal.json").is_file()
                                      and not (p / "approved.json").is_file()) if PROPOSALS.is_dir() else []
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mimo_bridge.tools.observer.validation_lock",
                                 description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    a = sub.add_parser("approve")
    a.add_argument("proposal", nargs="?")
    a.add_argument("--current", action="store_true", help="approve the locked set as it stands")
    args = ap.parse_args(argv)
    if args.cmd == "status":
        print(json.dumps(status(), indent=1))
        return 0
    if not args.current and not args.proposal:
        ap.error("approve needs a proposal folder or --current")
    try:
        out = approve(Path(args.proposal) if args.proposal else None, current=args.current)
    except LockError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
