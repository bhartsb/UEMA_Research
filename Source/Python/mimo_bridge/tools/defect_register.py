# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The defect register: every defect that can affect gain of function, in one file, with its age computed.

The operator's recovery directive of Oct 5, 2026, sections 4 and 5: "Every defect that can affect gain of function
must have these fields in the single source of truth", and the age rule: "any GOF-BLOCKER with no meaningful work for
12 hours moves automatically to the top of the dashboard. At 24 hours it requires a coordinator explanation and a named
next action before lower-priority work starts." The file is docs/gain_of_function_v4/defect_register.json:

    {"about": ..., "schema": 1, "severity_rule": ..., "defects": [defect, ...]}

A DEFECT, typed (each entry says in "sources" where its lines were read and in "unverified" what could not be read):
    id               the canonical defect id, upper case with underscores
    title            one sentence: what is wrong, as measured
    component        the id of the component manifest the defect sits in
    repairs          [{"component": a repair's id, "state": one of STATES, "note"}]: each repair built for it
    stage            the scoreboard stage it blocks (gof_scoreboard.STAGES), or None
    found_at         "Oct 4, 2026 2:45 PM PDT", or a day alone ("Sep 25, 2026") when no time of day is on record
    severity         GOF-BLOCKER, TRAINING-RISK, MEASUREMENT-RISK, NONBLOCKING or CURRENT EFFECT UNKNOWN (a past
                     measurement whose effect on the current model is not measured), with severity_basis
    owner, backup_owner
    cause_status     what is known of the cause, starting with FOUND or CAUSE KNOWN
    next_action      the exact next action
    waits_on         what that action waits on
    branch, commit   where the repair's code is
    state            where the defect stands, one of STATES (the contract's section 4)
    closing_test     {"row": a test registry id or None, "text": what must be shown}
    tests            the test registry ids that serve it
    last_work_stated {"at", "written_at", "what", "by", "source"}: meaningful work that no commit and no test run
                     shows; "at" is the time the work was done (no later than the time it was written, written_at)
    withdrawn_work   [{"statement", "by", "why", "at"}]: stated work taken back (a restated finding is not work)
    explanation      {"text", "next_action", "by", "at"}: the coordinator's explanation the 24-hour rule asks for
    closed           {"at", "by", "reason", "evidence"?: [row ids, files, commits]} when the entry is no longer a
                     defect: it takes effect only with the state FAILED/SUPERSEDED or VERIFIED, and needs evidence
                     (CLOSED ENTRIES below)

COMPUTED on every read (view()), never typed:
    age              since found_at
    last meaningful work   the newest of: the newest commit of the entry's branch whose message names the entry's id;
                     the newest start or end of a run of a test whose registry row names the entry (its "defect"
                     field: the id, or a list holding it); last_work_stated.at. Each idle time says which kind it
                     rests on (the dashboard review of Oct 5, 2026, finding F2: the newest commit of a shared branch,
                     and any run of a linked row, counted as this defect's work)
    idle             the time since that
    the 12-hour rule an open GOF-BLOCKER idle for 12 hours or more is listed first ("top")
    the 24-hour rule an open GOF-BLOCKER idle for 24 hours or more needs an explanation with a named next action,
                     written after its last meaningful work and no older than 24 hours; while it has none,
                     runner_gate() makes the test runner refuse every test that serves no open GOF-BLOCKER
    state checks     IN TRAINING needs one of the entry's repairs switched on in the active run's launch settings;
                     BUILT/OFF, REVIEWED and QUALIFIED/OFF need the repair of that state switched off; REVIEWED
                     needs an independent review on record whose reviewed code is the code now (no QA STALE);
                     VERIFIED needs the closing test's row at outcome PASS written by a runner. A state that fails
                     its check is listed under "problems": the page shows it as an error.
    CLOSED ENTRIES   (the dashboard review of Oct 5, 2026, finding L7) a "closed" entry takes effect only with the state
                     FAILED/SUPERSEDED or VERIFIED: with any other state the entry stays open and its close is a
                     problem; a close needs evidence: closed.evidence, or a test registry row id, a file path or a
                     commit named in closed.reason, or a run that ended of a test the entry links (tests or its
                     closing test); without one it is a problem.
    PAGE TEXTS       the texts of an entry that the pages print (PAGE_FIELDS and the stated work's "what") are held to
                     DESIGN.md section 11: one holding a bare score decimal is a problem (the dashboard review's
                     re-check of Oct 5, 2026, finding N1: a title's cosines printed as decimals on the Now page).
Only VERIFIED may be called fixed (directive section 4).

    python -m mimo_bridge.tools.defect_register status            one line per open defect: ID: STATE
    python -m mimo_bridge.tools.defect_register check             the register's problems; exit code 1 on any
    python -m mimo_bridge.tools.defect_register explain ID --text "..." --next-action "..." --by NAME
    python -m mimo_bridge.tools.defect_register work ID --what "..." --by NAME [--source "..."] [--at "TIME"]
    python -m mimo_bridge.tools.defect_register withdraw-work ID --why "..." --by NAME
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

from . import test_registry as tr

REPO = Path(__file__).resolve().parents[4]
REGISTER_RELATIVE = Path("docs/gain_of_function_v4/defect_register.json")
# CURRENT EFFECT UNKNOWN: the operator, Oct 5, 2026: "the defect register should contain every known unresolved defect,
# but the GOF-BLOCKER label should mean it is demonstrated to affect the current gain-of-function path".
SEVERITIES = ("GOF-BLOCKER", "TRAINING-RISK", "MEASUREMENT-RISK", "NONBLOCKING", "CURRENT EFFECT UNKNOWN")
# the states of a repair (directive section 4, the contract's section 4), in their order
STATES = ("FOUND", "CAUSE KNOWN", "BUILT/OFF", "REVIEWED", "QUALIFIED/OFF", "IN TRAINING", "VERIFIED",
          "FAILED/SUPERSEDED")
STATE_MEANINGS = {
    "FOUND": "the defect is measured; its cause may still be unknown",
    "CAUSE KNOWN": "the mechanism is supported by evidence",
    "BUILT/OFF": "the repair's code exists; production's behaviour is unchanged",
    "REVIEWED": "an independent review is completed; its findings are resolved or explicitly open",
    "QUALIFIED/OFF": "the repair passed the test that measures the real target",
    "IN TRAINING": "the repair's switch is on in the current run",
    "VERIFIED": "the target behaviour is improved in the full training loop",
    "FAILED/SUPERSEDED": "not active; the reason and the successor are recorded",
}
OFF_STATES = ("BUILT/OFF", "REVIEWED", "QUALIFIED/OFF")
TOP_AFTER_S = 12 * 3600.0
EXPLAIN_AFTER_S = 24 * 3600.0
EXPLANATION_LASTS_S = 24 * 3600.0
REQUIRED = ("id", "title", "component", "found_at", "severity", "owner", "backup_owner", "cause_status",
            "next_action", "waits_on", "state", "closing_test")
ID = re.compile(r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$")
_DAY = re.compile(r"\b([A-Z][a-z]{2})[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b")


class RegisterError(Exception):
    """The defect register cannot be read or written as the register."""


def register_path(root=None) -> Path:
    return Path(root or REPO) / REGISTER_RELATIVE


def load(root=None, path=None) -> dict:
    path = Path(path or register_path(root))
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise RegisterError("the defect register %s does not exist" % path) from None
    except (OSError, ValueError) as exc:
        raise RegisterError("the defect register %s cannot be read: %s" % (path, exc)) from exc
    if not isinstance(data, dict) or not isinstance(data.get("defects"), list):
        raise RegisterError("the defect register %s holds no 'defects' list" % path)
    return data


_SEVERITY_IN_RULE = re.compile(r"\b(%s) = (.+?)(?=; (?:%s) = |\. [A-Z]|\.?$)"
                               % ("|".join(map(re.escape, SEVERITIES)), "|".join(map(re.escape, SEVERITIES))))


def severity_meanings(root=None, rule=None) -> list:
    """[{name, definition}] for the severities, read from the register's own severity_rule, where each is written
    "NAME = definition" and the next one follows after "; " (the Terms and definitions tab defines them from here, so a
    change to the rule reaches the page). A severity the rule does not write in that form gets the sentence that it is
    not defined there; an unreadable register gives that sentence for all of them."""
    if rule is None:
        try:
            rule = str(load(root).get("severity_rule") or "")
        except RegisterError:
            rule = ""
    found = {m.group(1): m.group(2).strip().rstrip(".") for m in _SEVERITY_IN_RULE.finditer(rule)}
    return [dict(name=s, definition=found.get(s) or "not defined in the form 'NAME = definition' by the defect "
                                                     "register's severity rule") for s in SEVERITIES]


def save(data, root=None, path=None):
    path = Path(path or register_path(root))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".%d.tmp" % os.getpid())
    with open(tmp, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def when(text):
    """(seconds since 1970, precision) for a time as the register writes it: ("minute") for a full clock time
    ("Oct 4, 2026 2:45 PM PDT"), ("day") for a day alone ("Sep 25, 2026": the day's first second, Pacific time);
    (None, None) for anything else."""
    full = tr.parse_clock(text)
    if full is not None:
        return full, "minute"
    match = _DAY.search(str(text or ""))
    if match:
        day = tr.parse_clock("%s %s, %s 12:00 AM PDT" % match.groups())
        if day is not None:
            return day, "day"
    return None, None


def span_text(seconds) -> str:
    """A span as the page prints it: days and hours from 48 hours up, else hours and minutes."""
    seconds = max(0, int(seconds))
    if seconds >= 48 * 3600:
        days, hours = seconds // 86400, seconds % 86400 // 3600
        return "%d days %d %s" % (days, hours, "hour" if hours == 1 else "hours")
    hours, minutes = seconds // 3600, seconds % 3600 // 60
    if hours:
        return "%d %s %d %s" % (hours, "hour" if hours == 1 else "hours", minutes,
                                "minute" if minutes == 1 else "minutes")
    return "%d %s" % (minutes, "minute" if minutes == 1 else "minutes")


CLOSING_STATES = ("FAILED/SUPERSEDED", "VERIFIED")


def is_open(defect) -> bool:
    """False for VERIFIED, and for an entry closed with the state FAILED/SUPERSEDED (CLOSED ENTRIES in the module
    text: a "closed" typed beside any other state does not close the entry)."""
    if defect.get("state") == "VERIFIED":
        return False
    return not (defect.get("closed") and defect.get("state") in CLOSING_STATES)


# ------------------------------------------------------------------------------------------ last meaningful work
_NAMING = {}


def _branch_time(code_root, branch, defect_id=None):
    """(seconds, short hash) of the newest commit of `branch` whose message names `defect_id` as a whole word (merge
    commits left out: they carry other work), or (None, None). The dashboard review of Oct 5, 2026, finding F2: the
    newest commit of a branch counted as the work of every defect that names the branch. Kept per (repository, the
    branch's tip, id)."""
    if not branch or not defect_id:
        return None, None
    from . import component_manifest
    tip = component_manifest.resolve(code_root, branch)
    if tip is None:
        return None, None
    key = (str(code_root), tip, defect_id)
    if key in _NAMING:
        return _NAMING[key]
    out = component_manifest.git(code_root, "log", "--no-merges", "-F", "--grep=" + defect_id, "-n", "200",
                                 "--format=%x1e%H%x1f%ct%x1f%B", tip) or ""
    answer = (None, None)
    word = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(defect_id) + r"(?![A-Za-z0-9_])")
    for record in out.split("\x1e"):
        parts = record.strip("\n").split("\x1f", 2)
        if len(parts) == 3 and word.search(parts[2]):
            try:
                answer = (float(parts[1]), parts[0][:9])
            except ValueError:
                continue
            break
    if len(_NAMING) > 512:
        _NAMING.clear()
    _NAMING[key] = answer
    return answer


def names_defect(row, defect_id) -> bool:
    """True when a test registry row names the defect: its "defect" field is the id, or a list holding it."""
    named = row.get("defect") if isinstance(row, dict) else None
    return named == defect_id or (isinstance(named, list) and defect_id in named)


def _run_times(row):
    last = row.get("last_run") if isinstance(row.get("last_run"), dict) else {}
    out = []
    for key_s, key_text in (("ended_at", "ended"), ("started_at", "started")):
        value = last.get(key_s)
        if value is None:
            value = tr.parse_clock(last.get(key_text))
        if value is not None:
            out.append((float(value), "ended" if key_s == "ended_at" else "started"))
    return out


BASIS_TEXT = {"branch commit": "a commit whose message names the defect",
              "test run": "a run of a test whose registry row names the defect",
              "stated": "work stated in the register"}
NO_WORK_TEXT = ("no meaningful work is on record: no commit of its branch names it, no test whose row names it has "
                "run, and no work is stated")
# what an idle time counts from, printed beside each idle time on the page (the operator, Oct 5, 2026: "last worked"
# means when substantive work happened, not when somebody edited the register)
IDLE_MEANING = ("counted from when the last meaningful work was done (a commit naming the defect, a run of a test "
                "naming it, or work stated with the time it was done), not from an edit of the register")
# the same, in the few words the page prints beside each idle time (the full sentence is in their title)
IDLE_SHORT_WORK = "since the work was done, not a register edit"
IDLE_SHORT_FOUND = "since it was found: no work on record"


def last_work(defect, code_root=None, rows=None, branch_time=_branch_time) -> dict:
    """LAST MEANINGFUL WORK of a defect (the module text): {at_s, at_text, what, basis, basis_text} with basis "branch
    commit" (a commit of the entry's branch whose message names its id), "test run" (a run of a test whose registry
    row names the defect) or "stated"; at_s None when none of the three is on record (the entry is then idle since it
    was found)."""
    best = None

    def offer(at_s, what, basis):
        nonlocal best
        if at_s is not None and (best is None or at_s > best["at_s"]):
            best = dict(at_s=float(at_s), what=what, basis=basis)
    at_s, commit = branch_time(code_root or REPO, defect.get("branch"), defect.get("id"))
    if at_s is not None:
        offer(at_s, "commit %s of branch %s, whose message names %s" % (commit, defect.get("branch"), defect.get("id")),
              "branch commit")
    for row in rows or []:
        if isinstance(row, dict) and isinstance(row.get("id"), str) and names_defect(row, defect.get("id")):
            for at, kind in _run_times(row):
                offer(at, "test %s %s (its row names %s)" % (row["id"], kind, defect.get("id")), "test run")
    stated = defect.get("last_work_stated") if isinstance(defect.get("last_work_stated"), dict) else {}
    stated_s, _ = when(stated.get("at"))
    if stated_s is not None:
        what = str(stated.get("what") or "work stated in the register")
        by = stated.get("by") or "a writer not named"
        # the time the work was done and the time it was written, side by side (finding N2); a statement written
        # before Oct 5, 2026 kept only the time it was written
        offer(stated_s, ("%s (done %s; stated by %s at %s)" % (what, tr.clock(stated_s), by, stated["written_at"])
                         if not tr.empty(stated.get("written_at")) else "%s (stated by %s)" % (what, by)), "stated")
    if best is None:
        return dict(at_s=None, at_text=None, what=None, basis=None, basis_text=NO_WORK_TEXT)
    best["at_text"] = tr.clock(best["at_s"])
    best["basis_text"] = BASIS_TEXT[best["basis"]]
    return best


def rule_state(defect, work, now) -> dict:
    """The age rule for one defect: {idle_s (since its last meaningful work, or since it was found when it has
    none), top (an open GOF-BLOCKER idle 12 hours or more), needs_explanation (idle 24 hours or more without an
    explanation in force), explanation_in_force}."""
    found_s, _ = when(defect.get("found_at"))
    since = work.get("at_s") if work.get("at_s") is not None else found_s
    idle = None if since is None else max(0.0, now - since)
    blocker = is_open(defect) and defect.get("severity") == "GOF-BLOCKER"
    explanation = defect.get("explanation") if isinstance(defect.get("explanation"), dict) else {}
    written_s, _ = when(explanation.get("at"))
    in_force = bool(written_s is not None and not tr.empty(explanation.get("text"))
                    and not tr.empty(explanation.get("next_action")) and now - written_s < EXPLANATION_LASTS_S
                    and (since is None or written_s >= since))
    # an entry with no time on record at all cannot be shown to have had work: the rule treats it as idle
    over_12 = idle is None or idle >= TOP_AFTER_S
    over_24 = idle is None or idle >= EXPLAIN_AFTER_S
    return dict(idle_s=idle, top=bool(blocker and over_12), needs_explanation=bool(blocker and over_24 and
                                                                                 not in_force),
                explanation_in_force=in_force)


# ------------------------------------------------------------------------------------------ state checks
def state_problems(defect, manifest_status=None, rows=None) -> list:
    """Plain sentences for a typed state its check does not support (STATE CHECKS in the module text). Without
    `manifest_status` (component_manifest.status()'s components by id) only the checks that need no switch are
    made."""
    out = []
    name = defect.get("id")
    components = manifest_status or {}
    repairs = [r for r in defect.get("repairs") or [] if isinstance(r, dict)]
    for repair in repairs:
        state = repair.get("state")
        entry = components.get(repair.get("component"))
        if state not in STATES:
            out.append("%s: the state of %s is not one of %s" % (name, repair.get("component"), ", ".join(STATES)))
            continue
        if entry is None:
            if manifest_status is not None:
                out.append("%s: its repair %s is not an id of the component manifest" % (name,
                                                                                         repair.get("component")))
            continue
        on = (entry.get("switch") or {}).get("repair_on")
        if state == "IN TRAINING" and on is False:
            out.append("%s: %s is typed IN TRAINING and its switch is off in the active run (%s)"
                       % (name, repair["component"], (entry.get("switch") or {}).get("sentence")))
        if state in OFF_STATES and on is True:
            out.append("%s: %s is typed %s and its switch is on in the active run (%s)"
                       % (name, repair["component"], state, (entry.get("switch") or {}).get("sentence")))
        if state in ("REVIEWED", "QUALIFIED/OFF"):
            if entry.get("last_qa") is None:
                out.append("%s: %s is typed %s and no independent review of it is on record"
                           % (name, repair["component"], state))
            elif entry.get("qa_stale"):
                out.append("%s: %s is typed %s and its code changed after the reviewed commit %s (%s)"
                           % (name, repair["component"], state, entry["last_qa"].get("reviewed_commit"),
                              entry.get("qa_stale_text")))
    state = defect.get("state")
    if state not in STATES:
        out.append("%s: its state is not one of %s" % (name, ", ".join(STATES)))
    if state == "IN TRAINING" and repairs and manifest_status is not None and not any(
            (components.get(r.get("component")) or {}).get("switch", {}).get("repair_on") for r in repairs):
        out.append("%s is typed IN TRAINING and none of its repairs is switched on in the active run" % name)
    if state == "VERIFIED":
        closing = defect.get("closing_test") if isinstance(defect.get("closing_test"), dict) else {}
        row = next((r for r in rows or [] if isinstance(r, dict) and r.get("id") == closing.get("row")), None)
        if row is None or tr.verified_outcome(row) != "PASS":
            out.append("%s is typed VERIFIED and its closing test %s has no PASS written by a runner"
                       % (name, closing.get("row") or "(no row named)"))
    out += closed_problems(defect, rows)
    return out


_PATHLIKE = re.compile(r"[\w.-]+/[\w./-]+\.[A-Za-z0-9]{1,6}\b")
_COMMIT = re.compile(r"\b(?=[0-9a-f]*[0-9])[0-9a-f]{7,40}\b")       # a commit hash holds a digit


def closed_problems(defect, rows=None) -> list:
    """CLOSED ENTRIES (the module text; the dashboard review of Oct 5, 2026, finding L7): a "closed" beside a state other
    than FAILED/SUPERSEDED or VERIFIED, and a close with no evidence (closed.evidence; a registry row id, a file path or
    a commit named in closed.reason; or a run that ended of a test the entry links)."""
    closed = defect.get("closed")
    if not closed:
        return []
    name = defect.get("id")
    if not isinstance(closed, dict):
        return ["%s: closed is not {at, by, reason}" % name]
    out = []
    if defect.get("state") not in CLOSING_STATES:
        out.append("%s is closed with the state %s: a close takes effect only with FAILED/SUPERSEDED or VERIFIED, so the "
                   "entry stays open" % (name, defect.get("state")))
    for field in ("at", "by", "reason"):
        if tr.empty(closed.get(field)):
            out.append("%s: its close has no %s" % (name, field))
    reason = str(closed.get("reason") or "")
    ids = {r.get("id") for r in rows or [] if isinstance(r, dict) and isinstance(r.get("id"), str)}
    linked = set(str(i) for i in defect.get("tests") or [])
    closing = defect.get("closing_test") if isinstance(defect.get("closing_test"), dict) else {}
    if closing.get("row"):
        linked.add(str(closing["row"]))
    ran = [r.get("id") for r in rows or [] if isinstance(r, dict) and r.get("id") in linked
           and any(kind == "ended" for _at, kind in _run_times(r))]
    named_row = any(re.search(r"(?<![A-Za-z0-9_])" + re.escape(i) + r"(?![A-Za-z0-9_])", reason) for i in ids)
    if not (not tr.empty(closed.get("evidence")) or named_row or _PATHLIKE.search(reason)
            or _COMMIT.search(reason) or ran):
        out.append("%s is closed with no evidence: closed.evidence is empty, its reason names no test registry row, "
                   "file or commit, and no test it links has a run that ended" % name)
    return out


def structure_problems(data, component_ids=None) -> list:
    out, seen = [], set()
    for position, d in enumerate(data.get("defects", [])):
        if not isinstance(d, dict):
            out.append("defect %d is not an object" % position)
            continue
        name = d.get("id") if isinstance(d.get("id"), str) else "defect %d" % position
        for field in REQUIRED:
            if tr.empty(d.get(field)):
                out.append("%s: %s is missing" % (name, field.replace("_", " ")))
        if not isinstance(d.get("id"), str) or not ID.match(d.get("id") or ""):
            out.append("%s: the id is not upper case words joined by underscores" % name)
        if name in seen:
            out.append("%s: two defects have this id" % name)
        seen.add(name)
        if d.get("severity") not in SEVERITIES:
            out.append("%s: severity is not one of %s" % (name, ", ".join(SEVERITIES)))
        if when(d.get("found_at"))[0] is None and "found_at" not in (d.get("unverified") or {}):
            out.append("%s: found_at is not a time or a day, and the entry does not list it as unverified" % name)
        if component_ids is not None and d.get("component") not in component_ids:
            out.append("%s: its component %s is not an id of the component manifest" % (name, d.get("component")))
        closing = d.get("closing_test")
        if not isinstance(closing, dict) or tr.empty(closing.get("text")):
            out.append("%s: the closing test does not say what must be shown" % name)
        out += text_problems(d)
    return out


# the texts of an entry the pages print: the Now page's top GOF-BLOCKERs (recovery_api: title, next action, waits on,
# owner, and the idle text that carries the stated work's "what") and the Architecture tab's open defects (title)
PAGE_FIELDS = ("title", "next_action", "waits_on", "owner")


def text_problems(defect) -> list:
    """PAGE TEXTS (the module text): one sentence per number in a printed text of the entry that DESIGN.md section 11
    bars from a page (check_no_bare_decimals.readout_problems: a bare score decimal, an unanchored score, multiplier or
    unit input)."""
    from .check_no_bare_decimals import readout_problems
    name = defect.get("id")
    texts = [(field.replace("_", " "), defect.get(field)) for field in PAGE_FIELDS]
    stated = defect.get("last_work_stated")
    if isinstance(stated, dict):
        texts += [("stated work", stated.get("what")), ("stated work's writer", stated.get("by"))]
    out = []
    for field, text in texts:
        if isinstance(text, str):
            for number, rule, _context in readout_problems(text):
                out.append("%s: its %s, which the pages print, holds %s (%s): write a count, a real unit or a "
                           "percentage between named ends (DESIGN.md section 11)" % (name, field, number, rule))
    return out


def stated_problems(defect, now) -> list:
    """Stated work whose time of work cannot be read, is later than now, or is later than the time it was written
    (finding N2: stated work counts from the time the work was done, never from a time after it)."""
    stated = defect.get("last_work_stated")
    if not stated:
        return []
    name = defect.get("id")
    if not isinstance(stated, dict):
        return ["%s: its stated work is not {at, written_at, what, by, source}" % name]
    at_s, _ = when(stated.get("at"))
    if at_s is None:
        return ["%s: the time of its stated work (%s) is not a time as the register writes one" % (name,
                                                                                                  stated.get("at"))]
    out = []
    if at_s > now:
        out.append("%s: its stated work is dated %s, later than now (%s)" % (name, stated.get("at"), tr.clock(now)))
    written_s = tr.parse_clock(stated.get("written_at")) if not tr.empty(stated.get("written_at")) else None
    if written_s is not None and at_s > written_s + 59:
        out.append("%s: its stated work is dated %s, later than the time it was written (%s)"
                   % (name, stated.get("at"), stated.get("written_at")))
    return out


# ------------------------------------------------------------------------------------------ the runner's gate
def serves(row, defect) -> bool:
    """True when a test registry row serves a defect: the row's id is among the defect's tests or is its closing
    test, the row names the defect ("defect"), or the row's component is the defect's or one of its repairs'."""
    if not isinstance(row, dict):
        return False
    closing = defect.get("closing_test") if isinstance(defect.get("closing_test"), dict) else {}
    if row.get("id") in (defect.get("tests") or []) or (closing.get("row") and row.get("id") == closing["row"]):
        return True
    if row.get("defect") and row.get("defect") == defect.get("id"):
        return True
    component = row.get("component")
    return bool(component) and (component == defect.get("component") or component in [
        r.get("component") for r in defect.get("repairs") or [] if isinstance(r, dict)])


def unexplained(register, now=None, code_root=None, rows=None, branch_time=_branch_time) -> list:
    """The open GOF-BLOCKERs the 24-hour rule holds: [(defect, idle_s)] idle 24 hours or more with no explanation in
    force."""
    now = time.time() if now is None else float(now)
    out = []
    for d in register.get("defects", []):
        if isinstance(d, dict) and is_open(d) and d.get("severity") == "GOF-BLOCKER":
            rule = rule_state(d, last_work(d, code_root, rows, branch_time), now)
            if rule["needs_explanation"]:
                out.append((d, rule["idle_s"]))
    return out


def runner_gate(row, register=None, now=None, code_root=None, rows=None, branch_time=_branch_time):
    """THE 24-HOUR RULE AT THE RUNNER: why the test of `row` may not start now, or None. While an open GOF-BLOCKER has
    had no meaningful work for 24 hours and carries no explanation in force, a row that serves no open GOF-BLOCKER is
    refused. A row that serves any open GOF-BLOCKER starts."""
    register = load(root=code_root) if register is None else register
    held = unexplained(register, now, code_root, rows, branch_time)
    if not held:
        return None
    for d in register.get("defects", []):
        if isinstance(d, dict) and is_open(d) and d.get("severity") == "GOF-BLOCKER" and serves(row, d):
            return None
    worst, idle = max(held, key=lambda item: item[1] if item[1] is not None else float("inf"))
    return ("the defect register holds back every test that serves no open GOF-BLOCKER: %s (%s) has had no "
            "meaningful work for %s and carries no explanation with a named next action written in the last 24 "
            "hours%s. The coordinator writes one with `python -m mimo_bridge.tools.defect_register explain %s`"
            % (worst.get("id"), worst.get("title"), "a time not on record" if idle is None else span_text(idle),
               "" if len(held) == 1 else " (and %d more such defects)" % (len(held) - 1), worst.get("id")))


# ------------------------------------------------------------------------------------------ the page's view
def view(root=None, data_root=None, now=None, with_manifest=True, branch_time=_branch_time) -> dict:
    """GET /api/defects: the register with every computed field, ordered for the page: first the open GOF-BLOCKERs
    idle 12 hours or more (the longest idle first), then the other open defects by severity and idle time, then the
    closed ones. Keys: defects [entry + open, age_text, last_work, idle_text, top, needs_explanation, problems],
    top [ids], holding [ids the 24-hour rule holds], problems [sentences], status_lines ["ID: STATE", ...],
    sentence. `root` holds the register and the manifest; `data_root` the test registry and Saved/."""
    root = Path(root or REPO)
    data_root = Path(data_root or root)
    now = time.time() if now is None else float(now)
    out = dict(schema=1, register=REGISTER_RELATIVE.as_posix(), read_at_text=tr.clock(now), states=list(STATES),
               state_meanings=STATE_MEANINGS, severities=list(SEVERITIES), idle_meaning=IDLE_MEANING,
               rule=("An open GOF-BLOCKER with no meaningful work for 12 hours is listed first. At 24 hours the test "
                     "runner refuses every test that serves no open GOF-BLOCKER until the coordinator writes an "
                     "explanation with a named next action into the entry; an explanation holds for 24 hours."))
    try:
        data = load(root)
    except RegisterError as exc:
        out.update(available=False, error=str(exc), defects=[], top=[], holding=[], problems=[str(exc)],
                   status_lines=[], sentence="The defect register cannot be read: %s." % exc)
        return out
    try:
        rows = tr.load(tr.registry_path(data_root))["rows"]
    except tr.RegistryError as exc:
        rows = []
        out["tests_error"] = str(exc)
    manifest_status, component_ids = None, None
    if with_manifest:
        try:
            from . import component_manifest
            manifest = component_manifest.load(root)
            component_ids = set(component_manifest.components(manifest))
            run_name, args = component_manifest.active_args(data_root)
            entries = component_manifest.qa_entries(root)
            wanted = {r.get("component") for d in data["defects"] if isinstance(d, dict)
                      for r in d.get("repairs") or [] if isinstance(r, dict)}
            manifest_status = {}
            for cid, entry in component_manifest.components(manifest).items():
                if cid in wanted:
                    qa = component_manifest.last_qa(root, entry, entries)
                    manifest_status[cid] = dict(switch=component_manifest.switch_state(entry, run_name, args),
                                                last_qa=qa, qa_stale=bool(qa and qa["stale"]),
                                                qa_stale_text=qa["stale_text"] if qa else None)
        except Exception as exc:
            out["manifest_error"] = "%s: %s" % (type(exc).__name__, exc)
    problems = structure_problems(data, component_ids)
    views = []
    for d in data["defects"]:
        if not isinstance(d, dict):
            continue
        work = last_work(d, root, rows, branch_time)
        rule = rule_state(d, work, now)
        found_s, precision = when(d.get("found_at"))
        mine = state_problems(d, manifest_status, rows) + stated_problems(d, now)
        problems += mine
        entry = dict(d)
        entry.update(open=is_open(d), found_precision=precision,
                     age_s=None if found_s is None else max(0.0, now - found_s),
                     age_text=("found date not on record" if found_s is None else
                               "%s since it was found%s" % (span_text(now - found_s),
                                                            " (a day without its time of day)"
                                                            if precision == "day" else "")),
                     last_work=work,
                     idle_s=rule["idle_s"],
                     idle_text=("no time on record" if rule["idle_s"] is None else
                                "%s since its last meaningful work, %s: %s" % (span_text(rule["idle_s"]),
                                                                               work["basis_text"], work["what"])
                                if work.get("at_s") is not None else
                                "%s since it was found; %s" % (span_text(rule["idle_s"]), NO_WORK_TEXT)),
                     idle_short=(None if rule["idle_s"] is None else IDLE_SHORT_WORK if work.get("at_s") is not None
                                 else IDLE_SHORT_FOUND),
                     top=rule["top"], needs_explanation=rule["needs_explanation"],
                     explanation_in_force=rule["explanation_in_force"], problems=mine,
                     status_line="%s: %s" % (d.get("id"), d.get("state")),
                     linked_tests=[dict(id=r.get("id"), status=r.get("status"), outcome=tr.recorded_outcome(r))
                                   for r in rows if isinstance(r, dict) and serves(r, d)
                                   and (r.get("id") in (d.get("tests") or []) or
                                        r.get("id") == (d.get("closing_test") or {}).get("row"))])
        if manifest_status is not None:
            entry["repair_switches"] = {r.get("component"): (manifest_status.get(r.get("component")) or {}).get(
                "switch", {}).get("sentence") for r in d.get("repairs") or [] if isinstance(r, dict)}
        views.append(entry)
    rank = {s: i for i, s in enumerate(SEVERITIES)}

    def order(v):
        idle = v["idle_s"] if v["idle_s"] is not None else float("inf")
        return (0 if v["open"] else 1, 0 if v["top"] else 1, rank.get(v.get("severity"), len(SEVERITIES)), -idle)
    views.sort(key=order)
    top = [v["id"] for v in views if v["top"]]
    holding = [v["id"] for v in views if v["needs_explanation"]]
    open_count = sum(1 for v in views if v["open"])
    blockers = sum(1 for v in views if v["open"] and v.get("severity") == "GOF-BLOCKER")
    sentence = "%d open %s, %d of them GOF-BLOCKER." % (open_count, "defect" if open_count == 1 else "defects",
                                                       blockers)
    if top:
        sentence += " %d GOF-BLOCKER %s had no meaningful work for 12 hours or more." % (
            len(top), "has" if len(top) == 1 else "have")
    if holding:
        sentence += (" The 24-hour rule holds back every test that serves no open GOF-BLOCKER until an explanation "
                     "is written for: %s." % ", ".join(holding))
    out.update(available=True, defects=views, top=top, holding=holding, problems=problems,
               status_lines=[v["status_line"] for v in views if v["open"]], sentence=sentence,
               severity_rule=data.get("severity_rule"))
    return out


# ------------------------------------------------------------------------------------------ writes
def _change(defect_id, fields, root=None):
    data = load(root)
    found = [d for d in data["defects"] if isinstance(d, dict) and d.get("id") == defect_id]
    if len(found) != 1:
        raise RegisterError("no single defect has id %s" % defect_id)
    found[0].update(fields)
    save(data, root)
    return found[0]


def explain(defect_id, text, next_action, by, root=None):
    """Write the coordinator's explanation the 24-hour rule asks for: why the defect has had no meaningful work, and
    the named next action. It holds for 24 hours from the clock time written here."""
    if tr.empty(text) or tr.empty(next_action) or tr.empty(by):
        raise RegisterError("an explanation needs its text, a named next action and who writes it")
    return _change(defect_id, dict(explanation=dict(text=str(text), next_action=str(next_action), by=str(by),
                                                    at=tr.clock())), root)


def state_work(defect_id, what, by, source=None, root=None, at=None, now=None):
    """State meaningful work that no commit of the entry's branch and no run of a linked test shows. `at` is the time
    the work was done, written as the register writes times ("Oct 5, 2026 1:45 PM PDT"), no later than now; without
    it, the clock's time now. The time the statement is written is kept beside it (written_at) and the page prints
    both. The dashboard review's re-check of Oct 5, 2026, finding N2: work stated at 1:48 PM PDT counted from 1:48 PM,
    hours after the runs it named had ended."""
    if tr.empty(what) or tr.empty(by):
        raise RegisterError("stated work needs what was done and who states it")
    now = time.time() if now is None else float(now)
    if tr.empty(at):
        at_s = now
    else:
        at_s = tr.parse_clock(at)
        if at_s is None:
            raise RegisterError("the time the work was done, %r, is not a time as the register writes one "
                                "(\"Oct 5, 2026 1:45 PM PDT\")" % (at,))
        if at_s > now:
            raise RegisterError("the time the work was done, %s, is later than now (%s)" % (at, tr.clock(now)))
    return _change(defect_id, dict(last_work_stated=dict(at=tr.clock(at_s), written_at=tr.clock(now), what=str(what),
                                                         by=str(by), source=None if source is None else str(source))),
                   root)


def withdraw_work(defect_id, by, why, root=None):
    """Take back an entry's stated work (a restated finding is not work: finding N2). The statement leaves
    last_work_stated and is kept, with who took it back, why and when, in withdrawn_work."""
    if tr.empty(by) or tr.empty(why):
        raise RegisterError("taking back stated work needs who does it and why")
    data = load(root)
    found = [d for d in data["defects"] if isinstance(d, dict) and d.get("id") == defect_id]
    if len(found) != 1:
        raise RegisterError("no single defect has id %s" % defect_id)
    if not found[0].get("last_work_stated"):
        raise RegisterError("%s has no stated work to take back" % defect_id)
    statement = found[0].pop("last_work_stated")
    found[0].setdefault("withdrawn_work", []).append(dict(statement=statement, by=str(by), why=str(why),
                                                          at=tr.clock()))
    save(data, root)
    return found[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="defect_register", description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["status", "check", "explain", "work", "withdraw-work", "view"])
    parser.add_argument("id", nargs="?")
    parser.add_argument("--root")
    parser.add_argument("--data-root")
    parser.add_argument("--text")
    parser.add_argument("--next-action")
    parser.add_argument("--what")
    parser.add_argument("--source")
    parser.add_argument("--by")
    parser.add_argument("--at", help="work: the time the work was done, as the register writes times; no later than "
                                     "now (default: now)")
    parser.add_argument("--why", help="withdraw-work: why the stated work is taken back")
    a = parser.parse_args(argv)
    root = Path(a.root) if a.root else REPO
    try:
        if a.action == "explain":
            explain(a.id, a.text, a.next_action, a.by, root)
            print("explanation written into %s at %s" % (a.id, tr.clock()))
            return 0
        if a.action == "work":
            stated = state_work(a.id, a.what, a.by, a.source, root, at=a.at)["last_work_stated"]
            print("work stated in %s: done %s, written %s" % (a.id, stated["at"], stated["written_at"]))
            return 0
        if a.action == "withdraw-work":
            withdraw_work(a.id, a.by, a.why, root)
            print("stated work of %s taken back at %s" % (a.id, tr.clock()))
            return 0
        found = view(root, a.data_root)
        if a.action == "view":
            print(json.dumps(found, indent=1, ensure_ascii=False, default=str))
            return 0
        if a.action == "status":
            print(found["sentence"])
            for v in found["defects"]:
                if v["open"]:
                    print("%s%s | %s | %s" % ("TOP " if v["top"] else "", v["status_line"], v.get("severity"),
                                             v["idle_text"]))
            return 0
        print("%d %s in the defect register." % (len(found["problems"]),
                                                "problem" if len(found["problems"]) == 1 else "problems"))
        for line in found["problems"]:
            print("-", line)
        return 1 if found["problems"] else 0
    except RegisterError as exc:
        print("refused: %s" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
