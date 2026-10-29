# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""ROOT-CAUSE INVESTIGATIONS AS LIVE PROCESS-OF-ELIMINATION DIAGRAMS (the style guide, "EVERY ROOT-CAUSE INVESTIGATION HAS A
LIVE PROCESS-OF-ELIMINATION DIAGRAM"; the operator, Oct 5, 2026: "This should be part of the dashboard rules, not a
one-off diagram."). The record of this work: docs/gain_of_function_v4/dashboard_recovery/ELIMINATION_DIAGRAMS_2026-10-05.md.

THE FORMAT. One JSON file per investigation in docs/gain_of_function_v4/investigations/ (INVESTIGATIONS) of the code
checkout. The file DECLARES the path under suspicion and nothing that a test measures:

    {"schema": 1, "id", "title", "question", "about"?, "written_at", "written_by",
     "input": {"name", "what", "component"},          what enters the path, drawn first
     "stages": [STAGE, ...]}                          the path's stages, in the sequence the signal passes them
    STAGE  = {"key", "name", "question", "component",
              "components"?: [ids],                   every component between the previous stage's reading and this
                                                      stage's, when more than one (the box links and names each)
              "sources": [SOURCE, ...],               readings that answer this stage's question
              "related": [SOURCE, ...],               readings of a neighbouring question: shown, never deciding
              "defects": [defect id, ...],            entries of the defect register this stage concerns
              "to_decide": {"text", "defect"?, "row"?},  the test that would decide the stage while nothing does
              "branches": {"PASS": BRANCH, "FAIL": BRANCH, "UNKNOWN": BRANCH}}
    BRANCH = {"text", "next": a stage key, or null where the walk along the branches ends}
    SOURCE = {"row": a test registry row id, "in": "result" | "result_file", ...}   a registry row's reading, or
             {"file": a path relative to a checkout, ...}                           a file that is no registry row
             with "key": [the path to the value inside the JSON], "kind": "verdict" | "failures" | "reading" |
             "compare", "label" (what the value is, with its scale's two ends), "means" (what a met or an unmet reading
             says), for "compare": "against" [the path to the value it is compared with, in the same place] and
             "met_when" ("at_or_below", "below", "at_or_above", "above": the line the value meets against it),
             "of"? [the path to every item, for "failures"], "when_met"? "PASS" | "FAIL" (default PASS),
             "form"? how a "reading" prints ("share", "count", "unit", "list", "text"), "unit"?, "time_key"? [the
             path to a file's own time], "show"? [{"key", "label", "form"?, "in"?, "of"?, "unit"?}] more values of the
             same row or file, printed beside the reading.
check() refuses a file holding any other field, names a computed field typed by hand (a verdict, a time, a commit, an
evidence value) as such, and refuses texts that name a verdict (PASS, FAIL, UNKNOWN in capitals) or print a bare score
decimal (DESIGN.md section 11). view() draws no box of a refused file and lists why.

THE COMPUTATION (view(), on every read; nothing is kept between reads but parsed files whose size and time are
unchanged). For each stage:
- each SOURCE is read: a registry row through its runner record and result (the test registry module's rules: a result
  counts only for the row's last run; a run that is running, never ran, was cancelled or ended INVALID or STOPPED gives
  no verdict; a result file counts only when it was written during the row's last run), a file through its own JSON.
  kind "verdict": the value is a verdict word or true/false; "failures": the value is the list of items that fail the
  source's own line, so an empty list meets it; "reading": a value with no line for this stage's question; "compare":
  the value against another value of the same place, met when the declared relation holds.
- A BOX IS DECIDED ONLY BY A SEALED RUNNER RECORD AND A RESULT WRITTEN BY THAT RUN (the dashboard review of Oct 5, 2026,
  finding B2; the operator's rule "nobody types a box's verdict by hand"): a row source decides only when the row's
  last run carries the runner's verified seal and the value comes from the result the runner sealed at the run's end
  (test_registry.runner_result: its digest unchanged) or from the result file the runner recorded at the run's end,
  unchanged since (test_registry.runner_file). A value typed into the row's result is shown as history, marked typed
  into the row; a row of a runner of the older kind decides nothing: its reading is history. Old tests are not run
  again to fill a box.
- the VERDICT is that of the newest registry row source that decides; met gives when_met, unmet the other. No such
  source: UNKNOWN. A file that is no registry row never decides a stage: its reading is shown as history with its file
  as evidence. Every other source is history, newest first.
- every value carries the two judgements the directive asks of a number (section 7A): the row's own statistical line
  (how a difference is told from run-to-run noise) and its material line for gain of function, or the words "GOF
  threshold not yet defined" where the row states none (the dashboard review's finding F12).
- the box's fields: the test (the deciding row), its date and time (Pacific, 12-hour), its commit (the runner's record,
  else the result file's code.commit or commit) and the checkpoint the run measured (the runner's record; the
  dashboard review's finding L5), its evidence (the reading, its value and where it was read), the Architecture link
  of each of the stage's components, and the next unresolved branch. A field that cannot be filled is named in
  "missing" with the reason.
- THE WALK starts at the first stage and follows each stage's branch for its verdict. It ends where a branch names no
  next stage; a stage reading UNKNOWN whose branch ends the walk is the NEXT BRANCH to test; when the walk ends at the
  stage a fault is located at, the first stage reading UNKNOWN inside the located span is the next branch. Every
  UNKNOWN stage is an unresolved branch. A box's next unresolved branch is the first UNKNOWN stage its own branches lead to (itself when it
  reads UNKNOWN).
- LOCATED BY ELIMINATION: the first stage on the walk that reads FAIL and is not followed on the walk by a stage that
  reads PASS; the fault lies after the last stage before it that reads PASS and at or before it. A stage that reads
  FAIL and is followed by a PASS is listed as gone around (a later stage carries the signal past it).

    python -m mimo_bridge.tools.elimination check [FILE ...]        the format check, exit 1 on any problem
    python -m mimo_bridge.tools.elimination show [ID] [--root DATA] the computed boxes as text
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import test_registry as tr

INVESTIGATIONS = Path("docs/gain_of_function_v4/investigations")
SCHEMA = 1
VERDICTS = ("PASS", "FAIL", "UNKNOWN")
KINDS = ("verdict", "failures", "reading", "compare")
RELATIONS = {"at_or_below": "at or below", "below": "below", "at_or_above": "at or above", "above": "above"}
FORMS = ("share", "count", "unit", "list", "text")
PLACES = ("result", "result_file")
TOP_FIELDS = ("schema", "id", "title", "question", "about", "written_at", "written_by", "input", "stages")
INPUT_FIELDS = ("name", "what", "component")
STAGE_FIELDS = ("key", "name", "question", "component", "components", "sources", "related", "defects", "to_decide",
                "branches")
SOURCE_FIELDS = ("row", "file", "in", "key", "kind", "label", "means", "of", "when_met", "form", "unit", "time_key",
                 "show", "against", "met_when")
SHOW_FIELDS = ("in", "key", "label", "form", "of", "unit")
TO_DECIDE_FIELDS = ("text", "defect", "row")
BRANCH_FIELDS = ("text", "next")
# what the diagram computes; written into the file by hand, each is refused by name
COMPUTED = ("verdict", "verdicts", "outcome", "state", "status", "result", "results", "passed", "failed", "pass",
            "fail", "met", "commit", "time", "date", "when", "started", "ended", "evidence", "value", "values",
            "reading_value", "located", "next_branch", "unresolved", "conclusion", "decided_by", "history")
REQUIRED_TOP = ("schema", "id", "title", "question", "written_at", "written_by", "input", "stages")
REQUIRED_STAGE = ("key", "name", "question", "component", "branches", "to_decide")
REQUIRED_SOURCE = ("key", "kind", "label", "means")
# a declared text that names a verdict: the verdict of a box is computed, never typed
VERDICT_WORD = re.compile(r"(?<![A-Za-z_])(PASS|FAIL|UNKNOWN|PASSED|FAILED)(?![A-Za-z_])")
RESULT_SLACK_S = 5.0          # a job's result file is written before its process ends; the runner then writes the end
COMMIT_KEYS = (("code", "commit"), ("commit",), ("git_commit",), ("code_commit",))
NO_GOF_LINE = "GOF threshold not yet defined"
NAMES = (
    ("PASS (a stage)", "the newest test registry row that decides the stage's question met the line that question is "
                       "read against; a row decides only with its runner's sealed record and a value of that run's own "
                       "result, as the runner sealed it at the run's end"),
    ("FAIL (a stage)", "the newest test registry row that decides the stage's question did not meet that line"),
    ("UNKNOWN (a stage)", "no test registry row has decided the stage's question: none exists, its test has not run, "
                          "is running, ended INVALID or STOPPED, holds a reading with no line, was run by a runner of "
                          "the older kind, or holds only values typed into the row; the stage is an unresolved branch "
                          "and names the test that would decide it"),
    ("Typed into the row", "a value of a row's result that differs from the result its runner sealed at the run's end, "
                           "or a result file whose content changed after the run: shown as history, never deciding"),
    ("A runner record of the older kind", "a run written by a runner without the seal of Oct 5, 2026: its reading is "
                                          "shown as history and decides nothing; the test is not run again only to "
                                          "fill a box"),
    ("Statistically clear? (a value)", "the row's own statistical line: how its difference is told from run-to-run "
                                       "noise; a row written before Oct 5, 2026 states none"),
    ("Large enough to matter for gain of function? (a value)", "the row's material line for gain of function, or "
                                                               "GOF threshold not yet defined where it states none"),
    ("Next unresolved branch", "the first stage reading UNKNOWN that the branches of the verdicts lead to from a box; "
                               "for the whole investigation, the stage where the walk along the branches stops, or, "
                               "when the walk ends at the stage a fault is located at, the first undecided stage "
                               "inside that span"),
    ("Located by elimination", "the first stage on the walk that reads FAIL with no stage reading PASS after it: the "
                               "fault lies after the last stage before it that reads PASS, and at or before it"),
    ("History", "older readings of the same question, and readings of files that are no registry row: shown with "
                "their own date, never deciding the box"),
    ("Related", "readings of a neighbouring question, shown beside a stage with their own lines; they never decide it"),
    ("Input (a diagram)", "what enters the path under suspicion, drawn first, with its component"),
    ("Test (a box)", "the test registry row whose reading decides the stage: the newest row that answers the stage's "
                     "question with a verdict"),
    ("Date and time (a box)", "the end of that row's last run, Pacific time; for a file that is no registry row, its "
                              "own time, else the time of its last commit"),
    ("Commit (a box)", "the commit of the code that produced the reading: from the runner's record, else from the "
                       "result file; for a file that is no registry row, the last commit that changed it"),
    ("Evidence (a box)", "the value that decides the stage, what it is, and where it was read (the row's result or "
                         "its result file, and the key inside it)"),
    ("Architecture (a box)", "the stage's component in the component manifest; the link opens that component in the "
                             "Architecture tab"),
    ("Missing", "a field of a box that the records do not fill, with the reason"),
    ("Defects (a box)", "the entries of the defect register the stage concerns, each as NAME: STATE"),
)


class InvestigationError(Exception):
    """An investigation file cannot be read as JSON."""


# ------------------------------------------------------------------------------------------ reading files
_JSON = {}


def _read_json(path):
    """The parsed JSON of `path`, kept while its size and modification time are unchanged. Raises OSError or
    ValueError."""
    path = Path(path)
    st = path.stat()
    stamp = (st.st_size, st.st_mtime_ns)
    kept = _JSON.get(str(path))
    if kept and kept[0] == stamp:
        return kept[1]
    if st.st_size > 64 * 2 ** 20:
        raise ValueError("%s is larger than 64 MB" % path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if len(_JSON) > 128:
        _JSON.clear()
    _JSON[str(path)] = (stamp, data)
    return data


def folder(code_root=None) -> Path:
    """The investigation files' folder: UEMA_INVESTIGATIONS_DIR when set (a test's fixture), else INVESTIGATIONS of
    the code checkout."""
    if os.environ.get("UEMA_INVESTIGATIONS_DIR"):
        return Path(os.environ["UEMA_INVESTIGATIONS_DIR"])
    return Path(code_root or tr.REPO) / INVESTIGATIONS


def files(code_root=None) -> list:
    """The investigation files of the code checkout, sorted by name."""
    base = folder(code_root)
    try:
        return sorted(p for p in base.glob("*.json") if p.is_file())
    except OSError:
        return []


def load(path) -> dict:
    """The investigation file as JSON. Raises InvestigationError."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise InvestigationError("%s cannot be read as JSON: %s" % (path, exc)) from None
    if not isinstance(data, dict):
        raise InvestigationError("%s does not hold a JSON object" % path)
    return data


# ------------------------------------------------------------------------------------------ the format check
def _texts(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _texts(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _texts(v)


def _fields(obj, allowed, where, out):
    """Refuse every field of `obj` outside `allowed`; a field the diagram computes is named as such."""
    for key in obj:
        if key in allowed:
            continue
        if str(key).lower() in COMPUTED:
            out.append("%s: '%s' is computed from the test registry and may not be typed into the file" % (where, key))
        else:
            out.append("%s: '%s' is not a field of the format" % (where, key))


def _path_problems(value, where, out, name="key"):
    if not (isinstance(value, list) and value and all(isinstance(p, (str, int)) and not isinstance(p, bool)
                                                      for p in value)):
        out.append("%s: %s is not a list of keys (text) and positions (whole numbers)" % (where, name))


def _text_problems(text, where, out, allow_verdict_words=False):
    if not isinstance(text, str) or not text.strip():
        out.append("%s is empty or not text" % where)
        return
    if not allow_verdict_words:
        m = VERDICT_WORD.search(text)
        if m:
            out.append("%s names the verdict %s: a verdict is computed from the test registry, never typed"
                       % (where, m.group(1)))
    try:
        from .check_no_bare_decimals import readout_problems
        for number, rule, around in readout_problems(text):
            out.append("%s prints %s (%s) where a reader cannot place it: '...%s...' (DESIGN.md section 11)"
                       % (where, number, rule, around))
    except ImportError:
        pass


def _source_problems(src, where, out, keys):
    if not isinstance(src, dict):
        out.append("%s is not an object" % where)
        return
    _fields(src, SOURCE_FIELDS, where, out)
    has_row, has_file = isinstance(src.get("row"), str) and src["row"].strip(), isinstance(src.get("file"), str) \
        and src["file"].strip()
    if bool(has_row) == bool(has_file):
        out.append("%s names neither a registry row nor a file, or both" % where)
    if has_row and src.get("in") not in PLACES:
        out.append("%s: 'in' is not one of %s" % (where, ", ".join(PLACES)))
    if has_file and "in" in src:
        out.append("%s: a file source has no 'in'" % where)
    for name in REQUIRED_SOURCE:
        if name not in src:
            out.append("%s: '%s' is missing" % (where, name))
    if "key" in src:
        _path_problems(src["key"], where, out)
    for name in ("of", "time_key"):
        if name in src:
            _path_problems(src[name], where, out, name)
    if src.get("kind") not in KINDS:
        out.append("%s: kind is not one of %s" % (where, ", ".join(KINDS)))
    if src.get("kind") in ("reading", "compare") and src.get("form") not in FORMS:
        out.append("%s: a %s names its form, one of %s" % (where, src.get("kind"), ", ".join(FORMS)))
    if src.get("kind") == "compare":
        if "against" not in src:
            out.append("%s: a compare names the value it is compared with ('against')" % where)
        else:
            _path_problems(src["against"], where, out, "against")
        if src.get("met_when") not in RELATIONS:
            out.append("%s: met_when is not one of %s" % (where, ", ".join(RELATIONS)))
    elif "against" in src or "met_when" in src:
        out.append("%s: 'against' and 'met_when' belong to a compare only" % where)
    if "form" in src and src.get("form") not in FORMS:
        out.append("%s: form is not one of %s" % (where, ", ".join(FORMS)))
    if src.get("form") == "unit" and not (isinstance(src.get("unit"), str) and src["unit"].strip()):
        out.append("%s: a form 'unit' names its unit" % where)
    if "when_met" in src and src["when_met"] not in ("PASS", "FAIL"):
        out.append("%s: when_met is not PASS or FAIL" % where)
    for name in ("label", "means"):
        if name in src:
            _text_problems(src[name], "%s: %s" % (where, name), out)
    shows = src.get("show", [])
    if not isinstance(shows, list):
        out.append("%s: show is not a list" % where)
        shows = []
    for j, item in enumerate(shows):
        w = "%s, show %d" % (where, j + 1)
        if not isinstance(item, dict):
            out.append("%s is not an object" % w)
            continue
        _fields(item, SHOW_FIELDS, w, out)
        _path_problems(item.get("key"), w, out)
        if "of" in item:
            _path_problems(item["of"], w, out, "of")
        if item.get("form", "text") not in FORMS:
            out.append("%s: form is not one of %s" % (w, ", ".join(FORMS)))
        if item.get("form") == "unit" and not (isinstance(item.get("unit"), str) and item["unit"].strip()):
            out.append("%s: a form 'unit' names its unit" % w)
        if has_row and item.get("in", src.get("in")) not in PLACES:
            out.append("%s: 'in' is not one of %s" % (w, ", ".join(PLACES)))
        if has_file and "in" in item:
            out.append("%s: a file source has no 'in'" % w)
        _text_problems(item.get("label"), "%s: label" % w, out)


def check(data) -> list:
    """Every problem of an investigation's declaration, as sentences; [] when the file may be drawn. A file with any
    problem is refused whole: a box drawn from a part of a declaration could hide the part that was refused."""
    out = []
    if not isinstance(data, dict):
        return ["the file does not hold a JSON object"]
    _fields(data, TOP_FIELDS, "the file", out)
    for name in REQUIRED_TOP:
        if name not in data:
            out.append("the file: '%s' is missing" % name)
    if data.get("schema") != SCHEMA:
        out.append("the file: schema is not %d" % SCHEMA)
    if not (isinstance(data.get("id"), str) and re.fullmatch(r"[a-z0-9_]+", data["id"] or "")):
        out.append("the file: id is not lower-case letters, digits and underscores")
    for name in ("title", "question", "about"):
        if name in data:
            _text_problems(data[name], "the file: %s" % name, out)
    if tr.parse_clock(data.get("written_at")) is None:
        out.append("the file: written_at is not a clock time such as 'Oct 5, 2026 10:30 AM PDT'")
    inp = data.get("input")
    if not isinstance(inp, dict):
        out.append("the file: input is not an object")
    else:
        _fields(inp, INPUT_FIELDS, "the input", out)
        for name in INPUT_FIELDS:
            _text_problems(inp.get(name), "the input: %s" % name, out)
    stages = data.get("stages")
    if not (isinstance(stages, list) and stages):
        out.append("the file: stages is not a list of stages")
        return out
    keys = [s.get("key") for s in stages if isinstance(s, dict)]
    for k in sorted({k for k in keys if keys.count(k) > 1}, key=str):
        out.append("stage key '%s' is used more than once" % k)
    for i, st in enumerate(stages):
        where = "stage %d" % (i + 1)
        if not isinstance(st, dict):
            out.append("%s is not an object" % where)
            continue
        if isinstance(st.get("key"), str):
            where = "stage '%s'" % st["key"]
        _fields(st, STAGE_FIELDS, where, out)
        for name in REQUIRED_STAGE:
            if name not in st:
                out.append("%s: '%s' is missing" % (where, name))
        if not (isinstance(st.get("key"), str) and re.fullmatch(r"[a-z0-9_]+", st.get("key") or "")):
            out.append("%s: key is not lower-case letters, digits and underscores" % where)
        for name in ("name", "question", "component"):
            _text_problems(st.get(name), "%s: %s" % (where, name), out)
        if "components" in st:
            comps = st["components"]
            if not (isinstance(comps, list) and len(comps) >= 2 and all(isinstance(c, str) and c.strip() for c in comps)
                    and len(set(comps)) == len(comps)):
                out.append("%s: components is not a list of two or more distinct component ids" % where)
            elif st.get("component") not in comps:
                out.append("%s: components does not hold the stage's own component %s" % (where, st.get("component")))
        for group in ("sources", "related"):
            items = st.get(group, [])
            if not isinstance(items, list):
                out.append("%s: %s is not a list" % (where, group))
                continue
            for j, src in enumerate(items):
                _source_problems(src, "%s, %s %d" % (where, group[:-1] if group == "sources" else "related", j + 1),
                                 out, keys)
        defects = st.get("defects", [])
        if not (isinstance(defects, list) and all(isinstance(d, str) and d.strip() for d in defects)):
            out.append("%s: defects is not a list of defect ids" % where)
        todo = st.get("to_decide")
        if isinstance(todo, dict):
            _fields(todo, TO_DECIDE_FIELDS, "%s: to_decide" % where, out)
            _text_problems(todo.get("text"), "%s: to_decide: text" % where, out)
        elif "to_decide" in st:
            out.append("%s: to_decide is not an object" % where)
        branches = st.get("branches")
        if not isinstance(branches, dict):
            if "branches" in st:
                out.append("%s: branches is not an object" % where)
            continue
        for verdict in branches:
            if verdict not in VERDICTS:
                out.append("%s: branches holds '%s', which is not one of %s" % (where, verdict, ", ".join(VERDICTS)))
        for verdict in VERDICTS:
            b = branches.get(verdict)
            w = "%s: the %s branch" % (where, verdict)
            if not isinstance(b, dict):
                out.append("%s is missing" % w)
                continue
            _fields(b, BRANCH_FIELDS, w, out)
            if "next" not in b:
                out.append("%s names no next stage (null where the walk ends)" % w)
            elif b["next"] is not None and b["next"] not in keys:
                out.append("%s leads to '%s', which is not a stage of this file" % (w, b["next"]))
            elif b["next"] == st.get("key"):
                out.append("%s leads to its own stage" % w)
            _text_problems(b.get("text"), "%s: text" % w, out)
    return out


# ------------------------------------------------------------------------------------------ values as a reader reads them
def _num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _plain_number(n) -> str:
    """A number with at most 2 decimals and no exponent; thousands separated from 1,000 on."""
    if abs(n) >= 1000:
        return "{:,.0f}".format(n)
    if abs(n) >= 100:
        return "%.0f" % n
    return ("%.2f" % n).rstrip("0").rstrip(".") or "0"


def share_text(v) -> str:
    """A share (1 = all of it) as a count a reader can picture: "74.7 of every 100", "1.2 in every 10,000"."""
    if not _num(v):
        return "not a number (%s)" % json.dumps(v)
    if v == 0:
        return "0 of every 100"
    if abs(v) >= 0.01:
        return "%s of every 100" % _plain_number(v * 100)
    base = 1000
    while abs(v) * base < 1 and base < 10 ** 15:
        base *= 10
    return "%s in every %s" % (_plain_number(v * base), "{:,}".format(base))


# Sentences end at a full stop and a space. A semicolon does not end one: a statistical line puts semicolons inside its
# parentheses ("(0 where it reads below 0; ...)"), and splitting there left half a parenthesis on the page (Oct 15).
_SENTENCE_END = re.compile(r"(?<=\.)\s+")

# PLAIN REWRITES of essential sentences that print bare decimals, keyed by (registry row id, the original sentence with
# its white space collapsed). The original stays unchanged in the test registry, which the page names as the full
# source; a sentence that changes there no longer matches and is withheld again, with its notice. Oct 15, 2026
# (explainer-formatting-and-current-run-20261015): "Withholding a sentence must not conceal a failure, decision,
# comparison direction or essential statistical definition. Rewrite such essential text plainly and retain access to
# its full numerical source." Numbers checked against D:/UEMA_Research/external/D/UEMA_scratch/finger_investigation/probe_path/verify/
# training_bands.json on Oct 15: rho 0 to 0.577 and k 3.726 to 5.952 over the 55 bands; 0.05 / (2 x 55) = 1 in 2,200;
# 0.05 / 55 = 1 in 1,100. Wording of the rho and combined-chance rewrites corrected the same day (correlation
# measures a straight-line relationship, not all information; 5 in 100 is an upper bound on any band's crossing, not
# the combined chance under every dependence; explainer-b1k-binding-confirmed-20261015).
PLAIN_REWRITES = {
    ("probe_path_learning_off_match_481800",
     "s_D = the root mean square of D; rho = D's correlation from one matched window to the next (0 where it reads below "
     "0; 0.00 to 0.58 over the 55)."):
        "s_D = the root mean square of D; rho = D's correlation from one matched window to the next, a number from 0 to "
        "1 in the n_eff and degrees-of-freedom formulas below: it measures how adjacent windows' D values move together "
        "along a straight line. Negative readings are set to 0 for those formulas; the values used range from 0 to 58 in "
        "100 (rho at most 58 / 100) over the 55 summaries.",
    ("probe_path_learning_off_match_481800",
     "k = the Student t quantile at 1 - 0.05 / (2 x 55) with n x (1 - rho) / (1 + rho) degrees of freedom, n the "
     "matched windows (3.726 to 5.952)."):
        "k = the Student t quantile with n x (1 - rho) / (1 + rho) degrees of freedom, n the matched windows, that a "
        "draw exceeds with a chance of 1 in 2,200 (a 5 in 100 chance shared over both sides of the 55 bands); over the "
        "55 bands k is 3.726 to 5.952 multiples of the average's own spread.",
    ("probe_path_learning_off_match_481800",
     "If the probe path's windows are drawn like a training run's, its 16-window average minus control run 2's has "
     "the spread s_D / sqrt(n_eff), so each band is breached with a chance of about 0.05 / 55 and one of the 55 with a "
     "chance of about 5 in 100 at the measured correlations, whatever the summaries' dependence."):
        "If the probe path's windows are drawn like a training run's, its 16-window average minus control run 2's has "
        "the spread s_D / sqrt(n_eff), so each band is crossed with a nominal chance of about 1 in 1,100 (5 in 100 "
        "shared over the 55 bands). The bands aim to keep the chance of any of the 55 being crossed to no more than 5 in "
        "100, under the approximations listed below, even when the summaries depend on one another.",
}


def plain_text(text, source="its source file", row=None):
    """A text written elsewhere (a result file's sentence, a registry row's statistical line), as a page may print it:
    scientific numbers rewritten; a sentence that still prints a bare score decimal (DESIGN.md section 11) is replaced
    by its plain rewrite where PLAIN_REWRITES holds one for this row, else withheld; a notice gives the counts and where
    the original is. The page check found such sentences printed whole on the explainer's architecture tab on Oct 15,
    2026 (26 bare decimals, 3 explainer tests failing)."""
    from .check_no_bare_decimals import bare_decimals
    from .plain_numbers import sci_to_fixed
    if not isinstance(text, str):
        return text
    kept, rewritten, withheld = [], 0, 0
    for sentence in _SENTENCE_END.split(sci_to_fixed(text)):
        if not bare_decimals(sentence):
            kept.append(sentence)
            continue
        plain = PLAIN_REWRITES.get((row, " ".join(sentence.split())))
        if plain and not bare_decimals(plain):
            kept.append(plain)
            rewritten += 1
        else:
            withheld += 1
    if rewritten:
        kept.append("(%s rewritten here in plain numbers on Oct 15, 2026; %s holds the original wording.)" % (
            "1 sentence" if rewritten == 1 else "%d sentences" % rewritten, source))
    if withheld:
        kept.append("(%s withheld here: %s a score as a bare decimal, which DESIGN.md section 11 bars from pages; "
                    "%s holds the full text.)" % ("1 sentence" if withheld == 1 else "%d sentences" % withheld,
                                                  "it prints" if withheld == 1 else "they print", source))
    return " ".join(kept)


# A band record of a statistics stage: [quantity, summary, value, band low end, band high end]. A quantity named with
# one of these endings prints its values in that unit (radians per second converted to degrees per second); a
# prediction ratio prints as the share of the no-change error (0 = exact, 100 = no better than predicting no change).
# Every other quantity (a squared prediction error, a no-change sum) has no unit a reader can picture, so only its
# place against its band prints and the values stay in the result file.
_BAND_UNITS = (("_deg", "degrees", 1.0), ("_rads", "degrees per second", 180.0 / math.pi), ("_n", "N", 1.0),
               ("_cm", "cm", 1.0), ("_ms2", "m per second squared", 1.0))


def _band_record(x):
    return (isinstance(x, (list, tuple)) and len(x) == 5 and isinstance(x[0], str) and isinstance(x[1], str)
            and all(_num(v) for v in x[2:]))


def band_text(record) -> str:
    """One band record as a reader reads it: the quantity and summary, its value and band in a unit a reader can
    picture where there is one, and how far outside the band it lies, counted in hundredths of the band's width."""
    quantity, summary, value, low, high = record
    where = "inside its band"
    width = high - low
    if value < low or value > high:
        side, gap = ("below", low - value) if value < low else ("above", value - high)
        if width > 0:
            share = gap / width
            far = ("%s in 100 of the band's width" % "{:,.0f}".format(share * 100) if share >= 0.01
                   else share_text(share).replace(" in every ", " in ").replace(" of every ", " in ")
                   + " of the band's width")
            where = "%s its band by %s" % (side, far)
        else:
            where = "%s its band, which has no width" % side
    for ending, name, scale in _BAND_UNITS:
        if quantity.endswith(ending):
            return "%s %s: %s %s (band %s %s to %s %s), %s" % (
                quantity, summary, _plain_number(value * scale), name, _plain_number(low * scale), name,
                _plain_number(high * scale), name, where)
    if quantity.startswith("prediction_ratio"):
        # whole numbers, unless the value would print equal to a band end: then one decimal for all three (a decimal
        # of 1 or more is no bare score decimal), so "70 (band 53 to 70)" never hides that 70.42 lies above 69.89
        fmt = "{:,.0f}"
        if fmt.format(value * 100) in (fmt.format(low * 100), fmt.format(high * 100)):
            fmt = "{:,.1f}"
        return ("%s %s: a prediction error of %s in 100 of the no-change error (band %s to %s in 100), %s" % (
            quantity, summary, fmt.format(value * 100), fmt.format(low * 100), fmt.format(high * 100), where))
    return "%s %s: %s (no unit a reader can picture; its value and band are in the result file)" % (
        quantity, summary, where)


def _item_text(x) -> str:
    if _band_record(x):
        return band_text(x)
    if _num(x):
        return _plain_number(x)
    if isinstance(x, (list, tuple)):
        return "[%s]" % ", ".join(_item_text(v) for v in x)
    return str(x)


def value_text(value, form="text", unit=None, total=None) -> str:
    """How a value prints in its declared form."""
    if form == "share":
        return share_text(value)
    if form == "count":
        return "{:,}".format(int(value)) if _num(value) else json.dumps(value)
    if form == "unit":
        return "%s %s" % (_plain_number(value), unit) if _num(value) else json.dumps(value)
    if form == "list":
        if not isinstance(value, (list, tuple)):
            return json.dumps(value)
        of = " of the %d" % total if isinstance(total, int) else ""
        if not value:
            return "none%s" % of
        # an item that still prints a bare decimal is withheld alone, and the notice counts it; items that are records
        # themselves are set apart by semicolons
        from .check_no_bare_decimals import bare_decimals
        items = [_item_text(x) for x in value]
        shown = [s for s in items if not bare_decimals(s)]
        sep = "; " if any(isinstance(x, (list, tuple)) for x in value) else ", "
        text = "%d%s: %s" % (len(value), of, sep.join(shown))
        if len(shown) < len(items):
            n = len(items) - len(shown)
            text += (" (%s withheld here: %s a score as a bare decimal, which DESIGN.md section 11 bars from pages; the "
                     "result file holds %s.)" % ("1 item" if n == 1 else "%d items" % n,
                                                 "it prints" if n == 1 else "they print", "it" if n == 1 else "them"))
        return text
    if isinstance(value, str):
        return plain_text(value, "the result file")
    return json.dumps(value)


_MET_WORDS = {"pass": True, "passed": True, "yes": True, "met": True, "true": True,
              "fail": False, "failed": False, "no": False, "not met": False, "false": False}


def met_of(value):
    """True for a verdict word or value that meets its line (true, pass, passed, yes, met), False for one that does not
    (false, fail, failed, no, not met), None for anything else."""
    if value is True or value is False:
        return value
    if isinstance(value, str):
        return _MET_WORDS.get(value.strip().lower())
    return None


def resolve(data, path):
    """The value at `path` (keys and positions) inside `data`. Raises KeyError naming the first step not found."""
    cur = data
    for part in path:
        if isinstance(cur, dict) and isinstance(part, str) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and isinstance(part, int) and not isinstance(part, bool) and -len(cur) <= part < len(cur):
            cur = cur[part]
        else:
            raise KeyError(part)
    return cur


def path_text(path) -> str:
    return " > ".join(str(p) for p in path)


# ------------------------------------------------------------------------------------------ git
_GIT = {}


def git_last_commit(root, rel):
    """(short commit, its time in seconds) of the last commit that changed `rel` in the repository at `root`, or
    (None, None). Kept per file while the file's modification time is unchanged."""
    try:
        mtime = (Path(root) / rel).stat().st_mtime_ns
    except OSError:
        return None, None
    key = (str(root), str(rel), mtime)
    if key in _GIT:
        return _GIT[key]
    try:
        done = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%h %ct", "--", str(rel)],
                              capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        parts = done.stdout.split() if done.returncode == 0 else []
        answer = (parts[0], float(parts[1])) if len(parts) == 2 else (None, None)
    except (OSError, subprocess.SubprocessError, ValueError):
        answer = (None, None)
    if len(_GIT) > 256:
        _GIT.clear()
    _GIT[key] = answer
    return answer


# ------------------------------------------------------------------------------------------ reading one source
class Context:
    """What the sources of one view are read against: the data checkout (the registry, Saved/, result files given
    relative to it), the code checkout (the manifest, the register, the investigation files), the registry's rows by
    id, the manifest's entries and the register's entries."""

    def __init__(self, data_root=None, code_root=None, now=None):
        self.data_root = Path(data_root or tr.REPO)
        self.code_root = Path(code_root or tr.REPO)
        self.now = time.time() if now is None else float(now)
        self.registry_path = tr.registry_path(self.data_root)
        self.registry_error = None
        self.rows = {}
        self.registry_modified = None
        try:
            data = tr.load(self.registry_path)
            for row in data["rows"]:
                if isinstance(row, dict) and isinstance(row.get("id"), str):
                    self.rows.setdefault(row["id"], []).append(row)
            stamp = tr._stamp(self.registry_path)
            self.registry_modified = tr.clock(stamp[0] / 1e9) if stamp else None
        except tr.RegistryError as exc:
            self.registry_error = str(exc)
        self.components = tr._manifest_entries(self.code_root)
        self.defects, self.register_error = {}, None
        try:
            from . import defect_register
            reg = defect_register.load(root=self.code_root)
            self.defects = {d["id"]: d for d in reg.get("defects") or [] if isinstance(d, dict) and d.get("id")}
            self._is_open = defect_register.is_open
        except Exception as exc:                  # the register's own page reports a register that cannot be read
            self.register_error = "%s: %s" % (type(exc).__name__, exc)
            self._is_open = lambda d: not d.get("closed")

    def file_path(self, rel):
        """A file named relative to a checkout: the data checkout's first, then the code checkout's; absolute paths
        as they are."""
        p = Path(rel)
        if p.is_absolute():
            return p, None
        for root in (self.data_root, self.code_root):
            if (root / p).is_file():
                return root / p, root
        return self.data_root / p, None


def _run_span(last):
    """(earliest start, latest end) in seconds of a runner record: exact numbers when the runner wrote them, else the
    minute its clock text names."""
    start, _ = tr._time_span(last.get("started_at"), last.get("started"))
    _, end = tr._time_span(last.get("ended_at"), last.get("ended"))
    return start, end


def _commit_in(data):
    for path in COMMIT_KEYS:
        try:
            value = resolve(data, path)
        except KeyError:
            continue
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{7,40}", value.strip()):
            return value.strip()[:9], path_text(path)
    return None, None


def _blank(src, group):
    out = dict(group=group, kind=src.get("kind"), label=src.get("label"), means=src.get("means"),
               key=list(src.get("key") or []), key_text=path_text(src.get("key") or []), row=src.get("row"),
               file=src.get("file"), registry=bool(src.get("row")), state="no verdict", met=None, verdict=None,
               why=None, value=None, value_text=None, shown=[], time_s=None, time_text=None, time_basis=None,
               commit=None, commit_basis=None, commit_missing=None, read_from=None, provenance=None, run_text=None,
               when_met=src.get("when_met", "PASS"), decides=False, undecidable=None, typed=False, checkpoint=None,
               checkpoint_basis=None, runner=None, runner_version=None, statistical_line=None, gof_line=NO_GOF_LINE)
    return out


def checkpoint_text(value) -> str:
    """A checkpoint as a box prints it: "ckpt_step_455800 of run H84_..." for a path ending in RUN/checkpoints/NAME, the
    last part of any other path, the value itself otherwise."""
    text = str(value).strip().replace("\\", "/").rstrip("/")
    parts = [p for p in text.split("/") if p]
    if len(parts) >= 3 and parts[-2] == "checkpoints":
        return "%s of run %s" % (parts[-1], parts[-3])
    return parts[-1] if parts else text


def _checkpoint_in(data):
    """(the checkpoint a result names at its top-level key 'checkpoint', as checkpoint_text prints it) or None."""
    value = data.get("checkpoint") if isinstance(data, dict) else None
    return checkpoint_text(value) if isinstance(value, (str, int)) and str(value).strip() else None


def judgement_lines(row):
    """(statistical line, material line) of a registry row for every value read from it (the directive's section 7A
    asks both of every number; the dashboard review's finding F12): the row's own statistical_line, or the sentence
    that it states none; its material_gof_line, or NO_GOF_LINE where it states none or says "not yet defined"."""
    stat = row.get("statistical_line") if isinstance(row, dict) else None
    stat = (plain_text(" ".join(str(stat).split()), "the row's statistical_line in the test registry", row.get("id"))
            if isinstance(stat, str) and stat.strip() else
            "the row states no statistical line (a row written before Oct 5, 2026 1:23 AM PDT does not owe one)")
    material = row.get("material_gof_line") if isinstance(row, dict) else None
    material = " ".join(str(material).split()) if isinstance(material, str) else ""
    if not material or material.lower().startswith(tr.NOT_YET_DEFINED):
        material = NO_GOF_LINE
    return stat, material


def _judge(out, src, value, data):
    """Fill met, verdict and value_text from the value at the source's key."""
    kind = src.get("kind")
    total = None
    if src.get("of"):
        try:
            whole = resolve(data, src["of"])
            total = len(whole) if isinstance(whole, (list, dict)) else None
        except KeyError:
            total = None
    out["value"] = value
    if kind == "verdict":
        met = met_of(value)
        out["value_text"] = value_text(value, "text")
        if met is None:
            out.update(state="no verdict", why="the value at %s is %s, not a verdict word or true/false"
                                               % (out["key_text"], json.dumps(value)[:80]))
            return
    elif kind == "failures":
        if not isinstance(value, list):
            out.update(state="no verdict", why="the value at %s is not a list of the items that fail the line"
                                               % out["key_text"])
            return
        met = not value
        out["value_text"] = value_text(value, "list", total=total)
    elif kind == "compare":
        # THE VALUE AGAINST ANOTHER VALUE OF THE SAME PLACE (the dashboard review of Oct 5, 2026, finding F11: a stage
        # of a direction loss is decided by a direction reading against the real changes' own, not by a size line)
        form, unit = src.get("form", "text"), src.get("unit")
        try:
            other = resolve(data, src["against"])
        except KeyError as exc:
            out.update(state="no verdict", why="%s is not in %s (no step '%s')"
                                               % (path_text(src["against"]), out.get("read_from") or "the place read",
                                                  exc.args[0]))
            return
        if not (_num(value) and _num(other)):
            out.update(state="no verdict", why="the values at %s and %s are not both numbers"
                                               % (out["key_text"], path_text(src["against"])))
            return
        relation = src.get("met_when")
        met = {"at_or_below": value <= other, "below": value < other, "at_or_above": value >= other,
               "above": value > other}[relation]
        out["value_text"] = "%s; the line: %s %s (read at %s)" % (
            value_text(value, form, unit), RELATIONS[relation], value_text(other, form, unit),
            path_text(src["against"]))
    else:
        out.update(state="reading", value_text=value_text(value, src.get("form", "text"), src.get("unit"), total),
                   why="a reading with no line for this question")
        return
    other_verdict = "FAIL" if out["when_met"] == "PASS" else "PASS"
    out.update(state="decided", met=met, verdict=out["when_met"] if met else other_verdict)


def _shown(out, src, places, ctx, distrust=None):
    """The values the source's 'show' list prints beside its reading; places = {'result': data or why not,
    'result_file': data or why not} for a row, {None: data} for a file. `distrust` = {place: why its values are not the
    runner's} marks each value read from such a place."""
    distrust = distrust or {}
    for item in src.get("show") or []:
        where = item.get("in", src.get("in"))
        data = places.get(where)
        entry = dict(label=item.get("label"), key_text=path_text(item.get("key") or []), value_text=None, why=None,
                     not_the_runners=distrust.get(where))
        if isinstance(data, str) or data is None:
            entry["why"] = data or "not read"
        else:
            try:
                value = resolve(data, item["key"])
                total = None
                if item.get("of"):
                    try:
                        whole = resolve(data, item["of"])
                        total = len(whole) if isinstance(whole, (list, dict)) else None
                    except KeyError:
                        total = None
                entry["value_text"] = value_text(value, item.get("form", "text"), item.get("unit"), total)
            except KeyError as exc:
                entry["why"] = "%s is not in %s (no step '%s')" % (entry["key_text"], "the file" if where is None
                                                                  else where.replace("_", " "), exc.args[0])
        out["shown"].append(entry)


def _typed(why) -> bool:
    return bool(why) and (why.startswith("typed into the row") or why.startswith("written into the file after"))


def read_row_source(src, ctx, group="sources") -> dict:
    """One registry row's reading (see the module's docstring). It DECIDES (out["decides"]) only when the row's last run
    carries the runner's verified seal and the value is that run's own: the result the runner sealed at the run's end,
    or the result file it recorded then, unchanged since (the dashboard review of Oct 5, 2026, finding B2). Otherwise
    the reading is shown as history with the reason (out["undecidable"]; out["typed"] when a value was typed into the
    row or written into the file after the run)."""
    out = _blank(src, group)
    row_id = src["row"]
    if ctx.registry_error:
        out["why"] = "the test registry could not be read: %s" % ctx.registry_error
        return out
    rows = ctx.rows.get(row_id) or []
    if not rows:
        out.update(state="missing", why="row %s is not in the test registry" % row_id)
        return out
    if len(rows) > 1:
        out["why"] = "%d rows of the test registry have the id %s" % (len(rows), row_id)
        return out
    row = rows[0]
    last = row.get("last_run") if isinstance(row.get("last_run"), dict) else {}
    status = row.get("status")
    sealed = tr.sealed(row)
    state = tr.run_state(row)
    out["title"] = row.get("title")
    out["component_of_row"] = row.get("component")
    out["statistical_line"], out["gof_line"] = judgement_lines(row)
    out["checkpoint"] = str(last["checkpoint"]) if last.get("checkpoint") else None
    out["checkpoint_basis"] = "the runner's record" if last.get("checkpoint") else None
    # THE RUNNER AND ITS VERSION (the operator's rule of Oct 5, 2026: an older measurement is shown with them): the
    # runner the run record names, and the code checkout's commit it recorded when it ran (else the one its result
    # names, below)
    if last:
        out["runner"] = str(last.get("runner") or "a runner the run record does not name")
        out["runner_version"] = ("commit %s of the code checkout when it ran%s" % (
            str(last["commit"])[:9], " (with changes not committed)" if last.get("commit_dirty") else "")
            if last.get("commit") else None)
    out["provenance"] = ("the runner's sealed record, run %s" % last.get("run_id") if sealed else
                         "a runner record of the older kind (%s), not sealed: it decides nothing"
                         % (last.get("runner") or "runner not named") if last else "no runner record")
    started, ended = last.get("started"), last.get("ended")
    out["run_text"] = ("ran %s to %s" % (started, ended) if started and ended else
                       "started %s, no end recorded" % started if started else "no run recorded")
    # the time of this source: the end of the row's last run, else when its result was read, else when it was written
    t = last.get("ended_at") if tr._number(last.get("ended_at")) else tr.parse_clock(last.get("ended"))
    basis = "the end of the row's last run"
    result, current = tr.current_result(row)
    if t is None and isinstance(result, dict) and tr.parse_clock(result.get("read_at")) is not None:
        t, basis = tr.parse_clock(result.get("read_at")), "the time the row's result was read"
    if t is None and tr.parse_clock(row.get("written_at")) is not None:
        t, basis = tr.parse_clock(row.get("written_at")), "the time the row was written (it has no run)"
    out.update(time_s=t, time_text=None if t is None else tr.clock(t), time_basis=basis)
    if last.get("commit"):
        out.update(commit=str(last["commit"])[:9] + (" (with changes not committed)" if last.get("commit_dirty") else ""),
                   commit_basis="the runner's record")
    pending = None
    if status == "cancelled":
        pending = ("no verdict", "the row was cancelled")
    elif status in ("written", "queued"):
        pending = ("pending", "the row is %s and its test has not run" % status)
    elif status == "running" or state == "running":
        pending = ("pending", "its run started %s and has not ended" % (started or "at a time not recorded"))
    elif state in ("INVALID", "STOPPED"):
        pending = ("no verdict", "its last run ended %s: no judgement" % state)
    elif status not in ("done", "failed"):
        pending = ("no verdict", "its status '%s' is not one a run ends with" % status)
    # the two places a reading is read from, and for each whether its values are the runner's own (distrust: why not)
    places, distrust = {}, {}
    if isinstance(result, dict) and current:
        places["result"] = result
        _runners, distrust["result"] = tr.runner_result(row)
    else:
        places["result"] = ("the row's result belongs to an earlier run than its last one" if not current else
                            "the row holds no result")
    target = tr.result_file_of(row, ctx.data_root)
    if target is None:
        source = row.get("result_source") if isinstance(row.get("result_source"), dict) else {}
        places["result_file"] = "the row's result source names no single existing file (%s)" % (
            source.get("file") or "no file named")
    else:
        try:
            mtime = target.stat().st_mtime
            lo, hi = _run_span(last)
            if lo is not None and mtime < lo - 2.0:
                places["result_file"] = "%s was last written %s, before the row's last run started %s" % (
                    target, tr.clock(mtime), started)
            elif hi is not None and mtime > hi + RESULT_SLACK_S and status != "running":
                places["result_file"] = "%s was last written %s, after the row's last run ended %s" % (
                    target, tr.clock(mtime), ended)
            else:
                places["result_file"] = _read_json(target)
                distrust["result_file"] = tr.runner_file(row, target)
        except (OSError, ValueError) as exc:
            places["result_file"] = "%s could not be read: %s" % (target, exc)
    if out["checkpoint"] is None:
        for place in ("result_file", "result"):
            named = _checkpoint_in(places.get(place))
            if named:
                out.update(checkpoint=named, checkpoint_basis="the 'checkpoint' %s names" % (
                    "the result file" if place == "result_file" else "the row's result"))
                break
    if out["commit"] is None:
        for place in ("result_file", "result"):
            data = places.get(place)
            if isinstance(data, dict):
                commit, where = _commit_in(data)
                if commit:
                    out.update(commit=commit, commit_basis="%s, %s" % ("the result file" if place == "result_file"
                                                                       else "the row's result", where))
                    break
    if last and out["runner_version"] is None:
        out["runner_version"] = ("not in the run record; commit %s (%s)" % (out["commit"], out["commit_basis"])
                                 if out["commit"] else "not on record: neither the run record nor its result names a "
                                                       "commit")
    if out["commit"] is None:
        data = places.get("result_file")
        sha = data.get("script_sha256") if isinstance(data, dict) else None
        out["commit_missing"] = ("not recorded: %s holds no commit, and %s" % (
            "the runner's record" if last else "the row (it has no runner record)",
            "the result file holds none" if isinstance(data, dict) else "no result file of the run was read")
            + (" (the result file names its script by SHA-256 %s)" % sha[:12] if isinstance(sha, str) else ""))
    where = src.get("in")
    data = places.get(where)
    out["read_from"] = (str(target) if where == "result_file" and target is not None else
                        "the row's result in the test registry" if where == "result" else None)
    _shown(out, src, places, ctx, distrust)
    if pending and (src.get("kind") != "reading" or not isinstance(data, dict)):
        out.update(state=pending[0], why=pending[1])
        return out
    if isinstance(data, str):
        out["why"] = data
        return out
    try:
        value = resolve(data, src["key"])
    except KeyError as exc:
        out["why"] = "%s is not in %s (no step '%s')" % (out["key_text"], out["read_from"], exc.args[0])
        return out
    _judge(out, src, value, data)
    if pending and out["state"] == "reading":
        out["why"] = "a reading with no line for this question; %s" % pending[1]
    # A BOX IS DECIDED ONLY BY A SEALED RUNNER RECORD AND A RESULT WRITTEN BY THAT RUN (finding B2)
    not_the_runners = distrust.get(where)
    if out["state"] in ("decided", "reading") and not_the_runners:
        out.update(undecidable=not_the_runners, typed=_typed(not_the_runners))
    out["decides"] = out["state"] == "decided" and not not_the_runners
    # a sealed run's own verdict and the verdict word of its result must agree
    if out["decides"] and where == "result" and list(src["key"]) == ["verdict"] \
            and state in ("PASS", "FAIL") and out["met"] != (state == "PASS"):
        out.update(state="no verdict", met=None, verdict=None, decides=False,
                   why="the row's result says %s and the runner's sealed record says %s" % (json.dumps(value), state))
    return out


def read_file_source(src, ctx, group="sources") -> dict:
    """A file that is no registry row: its reading and its own verdict are shown as history; it never decides."""
    out = _blank(src, group)
    path, root = ctx.file_path(src["file"])
    out["read_from"] = str(path)
    out["provenance"] = "a file that is no registry row"
    out["statistical_line"] = "no statistical line: a file that is no registry row states none"
    out["undecidable"] = "a file that is no registry row never decides a stage"
    try:
        data = _read_json(path)
    except (OSError, ValueError) as exc:
        out.update(state="missing", why="%s could not be read: %s" % (path, exc))
        return out
    # its commit: git runs in the code checkout only (as every route does); the code checkout's history applies when
    # its copy of the file holds the same bytes as the copy read
    rel = None
    if root is not None:
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = None
    commit = commit_t = None
    why_no_commit = "not recorded: the file is outside both checkouts, so no commit of it is read"
    if rel:
        own = ctx.code_root / rel
        try:
            same = own.is_file() and (own == path or own.read_bytes() == path.read_bytes())
        except OSError:
            same = False
        if same:
            commit, commit_t = git_last_commit(ctx.code_root, rel)
            if not commit:
                why_no_commit = "not recorded: no commit of the code checkout holds the file"
        else:
            why_no_commit = ("not recorded: the code checkout's copy of the file differs from the one read (or is "
                             "missing), and git runs in the code checkout only")
    if commit:
        out.update(commit=commit, commit_basis="the last commit that changed the file")
    else:
        out["commit_missing"] = why_no_commit
    t, basis = None, None
    if src.get("time_key"):
        try:
            t = tr.parse_clock(resolve(data, src["time_key"]))
            basis = "the file's own time (%s)" % path_text(src["time_key"])
        except KeyError:
            t = None
    if t is None and commit_t is not None:
        t, basis = commit_t, "the time of the last commit that changed the file"
    if t is None:
        try:
            t, basis = path.stat().st_mtime, "the file's modification time"
        except OSError:
            t = None
    out.update(time_s=t, time_text=None if t is None else tr.clock(t), time_basis=basis)
    named = _checkpoint_in(data)
    if named:
        out.update(checkpoint=named, checkpoint_basis="the 'checkpoint' the file names")
    _shown(out, src, {None: data}, ctx)
    try:
        value = resolve(data, src["key"])
    except KeyError as exc:
        out["why"] = "%s is not in %s (no step '%s')" % (out["key_text"], path, exc.args[0])
        return out
    _judge(out, src, value, data)
    return out


def read_source(src, ctx, group="sources") -> dict:
    return read_row_source(src, ctx, group) if src.get("row") else read_file_source(src, ctx, group)


# ------------------------------------------------------------------------------------------ one investigation
def _architecture(component, ctx):
    entry = (ctx.components or {}).get(component)
    known = None if ctx.components is None else entry is not None
    return dict(component=component, known=known, name=entry.get("name") if entry else None,
                kind=entry.get("kind") if entry else None, blocks=list(entry.get("blocks") or []) if entry else [],
                href="/files/Saved/mimo_bridge/explainer.html#t=arch&component=%s" % component)


def _defect_line(defect_id, ctx):
    d = ctx.defects.get(defect_id)
    if d is None:
        why = ("the defect register could not be read (%s)" % ctx.register_error if ctx.register_error else
               "not in the defect register")
        return dict(id=defect_id, known=False, status_line="%s: %s" % (defect_id, why))
    closed = d.get("closed") if isinstance(d.get("closed"), dict) else None
    line = "%s: %s" % (defect_id, d.get("state") or "no state written")
    if closed:
        line += " (closed %s)" % (closed.get("at") or "at a time not recorded")
    return dict(id=defect_id, known=True, state=d.get("state"), severity=d.get("severity"), open=ctx._is_open(d),
                closed_at=closed.get("at") if closed else None, status_line=line,
                next_action=d.get("next_action"))


def _to_decide(stage, ctx, sources):
    todo = stage.get("to_decide") or {}
    parts = [todo.get("text")] if todo.get("text") else []
    if todo.get("defect"):
        d = ctx.defects.get(todo["defect"])
        if d and d.get("next_action"):
            parts.append("%s, its next action in the defect register: %s" % (todo["defect"], d["next_action"]))
        else:
            parts.append("%s has no next action in the defect register" % todo["defect"])
    if todo.get("row"):
        rows = ctx.rows.get(todo["row"]) or []
        parts.append("Its registry row %s: %s" % (todo["row"], rows[0].get("status") if rows else "not written yet"))
    for s in sources:
        if s["registry"] and s["state"] == "pending":
            parts.append("Row %s answers it when its run ends: %s" % (s["row"], s["why"]))
    return " ".join(p.rstrip(".") + "." for p in parts if p)


def decide(sources):
    """(the deciding source or None, the verdict): the newest registry row source that DECIDES (read_row_source: a
    verdict from the runner's sealed record and that run's own result) decides; with none, UNKNOWN. A file that is no
    registry row, a row of a runner of the older kind and a value typed into a row never decide."""
    deciding = [s for s in sources if s["registry"] and s["state"] == "decided" and s.get("decides")]
    deciding.sort(key=lambda s: -(s["time_s"] if s["time_s"] is not None else -1e18))
    decided = deciding[0] if deciding else None
    return decided, (decided["verdict"] if decided else "UNKNOWN")


def missing_fields(box, decided, sources) -> dict:
    """{field: why it is not filled} for the six fields of a box that the records leave empty (the next unresolved
    branch is named by the walk, next_unresolved_of)."""
    missing = {}
    if decided:
        if box["time_text"] is None:
            missing["time"] = "the deciding row records no time of its run, its result or its writing"
        if box["commit"] is None:
            missing["commit"] = decided["commit_missing"] or "not recorded"
    else:
        missing["test"] = ("no source answers this stage" if not sources else
                           "no registry row source decides it: " + "; ".join(
                               "%s: %s" % (s["row"] or s["file"], s.get("undecidable") or s["why"] or s["state"])
                               for s in sources))
        missing["time"] = "no run has decided this stage"
        missing["commit"] = "no run has decided this stage"
        missing["evidence"] = ("no reading decides this stage; the readings that answer it without a line, and files "
                               "that are no registry row, are listed under History" if sources else
                               "no reading answers this stage")
    unknown = [a["component"] for a in box.get("architectures") or [box["architecture"]] if a["known"] is False]
    if unknown:
        missing["architecture"] = "%s %s not an id of the component manifest" % (
            " and ".join(unknown), "is" if len(unknown) == 1 else "are")
    elif any(a["known"] is None for a in box.get("architectures") or [box["architecture"]]):
        missing["architecture"] = "the component manifest could not be read"
    return missing


def walk_of(boxes, order):
    """(the stage keys the walk passes, the key of the next branch or None): from the first stage, each stage's branch
    for its verdict; the walk ends where a branch names no next stage, and a stage reading UNKNOWN whose branch ends
    it is the next branch to test."""
    walk, seen, key, next_branch = [], set(), order[0], None
    while key and key not in seen:
        seen.add(key)
        walk.append(key)
        box = boxes[key]
        if box["verdict"] == "UNKNOWN" and box["branch"]["next"] is None:
            next_branch = key
            break
        key = box["branch"]["next"]
    return walk, next_branch


def unresolved_of(boxes, order) -> list:
    """Every stage that reads UNKNOWN, in the declared sequence: each is an unresolved branch."""
    return [k for k in order if boxes[k]["verdict"] == "UNKNOWN"]


def next_unresolved_of(boxes, key):
    """(the first stage reading UNKNOWN that the branches lead to from stage `key`, itself when it reads UNKNOWN, or
    None; the last stage visited)."""
    k, seen, last_k = key, set(), key
    while k and k not in seen:
        seen.add(k)
        last_k = k
        if boxes[k]["verdict"] == "UNKNOWN":
            return k, last_k
        k = boxes[k]["branch"]["next"]
    return None, last_k


def next_inside_span(walk, boxes, located):
    """The first stage reading UNKNOWN on the walk between the last stage reading PASS before the located stage and the
    located stage itself, or None (the dashboard review of Oct 5, 2026, finding F11)."""
    start = walk.index(located["after"]) + 1 if located.get("after") else 0
    inside = [k for k in walk[start:walk.index(located["at_or_before"])] if boxes[k]["verdict"] == "UNKNOWN"]
    return inside[0] if inside else None


def compute(data, ctx, path=None) -> dict:
    """The computed view of one investigation whose declaration passed check()."""
    stages = data["stages"]
    by_key = {s["key"]: s for s in stages}
    order = [s["key"] for s in stages]
    boxes = {}
    for st in stages:
        sources = [read_source(s, ctx, "sources") for s in st.get("sources") or []]
        related = [read_source(s, ctx, "related") for s in st.get("related") or []]
        decided, verdict = decide(sources)
        history = [s for s in sources if s is not decided]
        history.sort(key=lambda s: -(s["time_s"] if s["time_s"] is not None else -1e18))
        components = list(st.get("components") or [st["component"]])
        box = dict(
            key=st["key"], name=st["name"], question=st["question"], component=st["component"],
            components=components, architectures=[_architecture(c, ctx) for c in components],
            architecture=_architecture(st["component"], ctx), verdict=verdict, test=None, time_text=None, commit=None,
            evidence=None, missing={}, history=history, related=related,
            defects=[_defect_line(d, ctx) for d in st.get("defects") or []],
            to_decide=_to_decide(st, ctx, sources) if verdict == "UNKNOWN" else None,
            branch=None, next_unresolved=None, on_walk=False, position=order.index(st["key"]) + 1)
        if decided:
            box.update(test=dict(row=decided["row"], title=decided.get("title"), provenance=decided["provenance"],
                                 run_text=decided["run_text"], checkpoint=decided.get("checkpoint")),
                       time_text=decided["time_text"], commit=decided["commit"],
                       evidence=dict(label=decided["label"], means=decided["means"], value_text=decided["value_text"],
                                     read_from=decided["read_from"], key_text=decided["key_text"],
                                     shown=decided["shown"], when_met=decided["when_met"], met=decided["met"],
                                     statistical_line=decided["statistical_line"], gof_line=decided["gof_line"]))
        box["missing"] = missing_fields(box, decided, sources)
        boxes[st["key"]] = box
    # each box's branch for its verdict
    for key, box in boxes.items():
        b = by_key[key]["branches"][box["verdict"]]
        nxt = b.get("next")
        box["branch"] = dict(verdict=box["verdict"], text=b.get("text"), next=nxt,
                             next_name=by_key[nxt]["name"] if nxt else None)
    walk, next_branch = walk_of(boxes, order)
    for key in walk:
        boxes[key]["on_walk"] = True
    unresolved = unresolved_of(boxes, order)
    # each box's next unresolved branch: itself when UNKNOWN, else the first UNKNOWN its branches lead to
    for key, box in boxes.items():
        found, last_k = next_unresolved_of(boxes, key)
        if found:
            box["next_unresolved"] = dict(key=found, name=boxes[found]["name"], itself=found == key,
                                          position=boxes[found]["position"], test=boxes[found]["to_decide"])
        else:
            box["missing"]["next_unresolved"] = ("none: the branches from this stage reach no stage that reads "
                                                 "UNKNOWN; they end at '%s' with: %s"
                                                 % (boxes[last_k]["name"], boxes[last_k]["branch"]["text"]))
    # located by elimination
    seq = [(k, boxes[k]["verdict"]) for k in walk]
    located = dict(found=False, after=None, at_or_before=None, gone_around=[], sentence=None)
    for i, (k, v) in enumerate(seq):
        if v != "FAIL":
            continue
        if any(v2 == "PASS" for _, v2 in seq[i + 1:]):
            located["gone_around"].append(k)
            continue
        before = [k2 for k2, v2 in seq[:i] if v2 == "PASS"]
        located.update(found=True, at_or_before=k, after=before[-1] if before else None)
        break
    # A LOCATED SPAN WITH AN UNDECIDED STAGE INSIDE IT: when the walk ends at the stage the fault is located at, the
    # first stage reading UNKNOWN between the last stage reading PASS and that stage is the next branch: deciding it
    # splits the span the fault lies in (the dashboard review of Oct 5, 2026, finding F11: the 26-handle diagram's
    # reading at the predicted next proprioception is taken after three components)
    if located["found"] and next_branch is None:
        next_branch = next_inside_span(walk, boxes, located)

    def label(k):
        comps = boxes[k]["components"]
        named = comps[0] if len(comps) == 1 else "%s and %s" % (", ".join(comps[:-1]), comps[-1])
        return "'%s' (%s, %s)" % (boxes[k]["name"], named, boxes[k]["verdict"])
    if located["found"]:
        located["sentence"] = (("The fault lies after %s and at or before %s." % (label(located["after"]),
                                                                                   label(located["at_or_before"])))
                               if located["after"] else
                               "The fault lies at or before %s, the first stage of the walk that reads it."
                               % label(located["at_or_before"]))
    elif next_branch:
        located["sentence"] = ("Not located: the walk stops at %s, which no test has decided." % label(next_branch))
    else:
        located["sentence"] = "Not located: no stage on the walk reads FAIL."
    if located["gone_around"]:
        located["sentence"] += " Gone around: %s, each followed on the walk by a stage that reads PASS." % ", ".join(
            label(k) for k in located["gone_around"])
    nb = boxes[next_branch] if next_branch else None
    inp = data["input"]
    shown_path = None
    if path:
        try:
            shown_path = Path(path).resolve().relative_to(ctx.code_root.resolve()).as_posix()
        except ValueError:
            shown_path = str(path)
    return dict(
        id=data["id"], file=shown_path, title=data["title"], question=data["question"],
        about=data.get("about"), written_at=data["written_at"], written_by=data["written_by"], refused=False,
        problems=[], input=dict(name=inp["name"], what=inp["what"], component=inp["component"],
                                architecture=_architecture(inp["component"], ctx)),
        stages=[boxes[k] for k in order], walk=walk,
        next_branch=dict(key=nb["key"], name=nb["name"], test=nb["to_decide"]) if nb else None,
        unresolved=[dict(key=k, name=boxes[k]["name"], test=boxes[k]["to_decide"], on_walk=boxes[k]["on_walk"])
                    for k in unresolved],
        located=located,
        errors=[dict(stage=b["key"], text="%s: %s" % (b["key"], b["missing"]["architecture"]))
                for b in boxes.values() if "architecture" in b["missing"]
                and any(a["known"] is False for a in b["architectures"])]
        + [dict(stage=b["key"], text="%s: %s" % (b["key"], d["status_line"])) for b in boxes.values()
           for d in b["defects"] if not d["known"]])


def view(data_root=None, code_root=None, only=None, now=None) -> dict:
    """GET /api/investigations: every investigation file of the code checkout, each checked and computed on this read
    against the data checkout's test registry; `only` limits it to one id."""
    ctx = Context(data_root, code_root, now)
    out = dict(schema=1, read_at_text=tr.clock(ctx.now), folder=INVESTIGATIONS.as_posix(),
               registry=tr.REGISTRY_RELATIVE.as_posix(), registry_modified_text=ctx.registry_modified,
               registry_error=ctx.registry_error, register_error=ctx.register_error,
               names=[dict(name=n, definition=d) for n, d in NAMES], investigations=[], errors=[])
    for path in files(ctx.code_root):
        try:
            data = load(path)
        except InvestigationError as exc:
            out["investigations"].append(dict(id=path.stem, file=str(path), refused=True, problems=[str(exc)]))
            out["errors"].append(dict(investigation=path.stem, text=str(exc)))
            continue
        if only and data.get("id") != only:
            continue
        problems = check(data)
        if problems:
            out["investigations"].append(dict(id=data.get("id") or path.stem, file=str(path), refused=True,
                                              title=data.get("title"), problems=problems))
            out["errors"].append(dict(investigation=data.get("id") or path.stem,
                                      text="Investigation %s is refused: %s" % (data.get("id") or path.stem,
                                                                               "; ".join(problems))))
            continue
        inv = compute(data, ctx, path)
        out["investigations"].append(inv)
        for e in inv["errors"]:
            out["errors"].append(dict(investigation=inv["id"], text=e["text"]))
    if ctx.registry_error:
        out["errors"].append(dict(investigation=None, text="The test registry could not be read: %s; every stage "
                                                           "read from it is UNKNOWN." % ctx.registry_error))
    return out


# ------------------------------------------------------------------------------------------ command line
def _print_view(v):
    for inv in v["investigations"]:
        print("=" * 100)
        print(inv["id"], "REFUSED" if inv.get("refused") else "")
        if inv.get("refused"):
            for p in inv["problems"]:
                print("  problem:", p)
            continue
        print(inv["title"])
        for b in inv["stages"]:
            print("-" * 100)
            print("%d. %s [%s] %s" % (b["position"], b["name"], b["component"], b["verdict"]))
            if b["test"]:
                print("   test:", b["test"]["row"], "|", b["test"]["provenance"], "| checkpoint",
                      b["test"].get("checkpoint"))
            print("   time:", b["time_text"], "| commit:", b["commit"])
            if b["evidence"]:
                print("   evidence:", b["evidence"]["label"], "=>", b["evidence"]["value_text"])
                for s in b["evidence"]["shown"]:
                    print("      shown:", s["label"], "=>", s["value_text"] or s["why"])
            for k, why in b["missing"].items():
                print("   missing %s: %s" % (k, why))
            print("   branch:", b["branch"]["text"], "->", b["branch"]["next_name"])
            if b["next_unresolved"]:
                print("   next unresolved:", b["next_unresolved"]["name"])
            for h in b["history"]:
                print("   history:", h["row"] or h["file"], h["state"], h["verdict"], h["time_text"],
                      h["value_text"], "|", h.get("undecidable") or h["why"], "| runner:", h["runner"],
                      "| its version:", h["runner_version"], "| checkpoint:", h["checkpoint"])
            for r in b["related"]:
                print("   related:", r["row"] or r["file"], r["state"], r["verdict"], r["time_text"], r["value_text"],
                      "|", r["why"])
            for d in b["defects"]:
                print("   defect:", d["status_line"])
        print("-" * 100)
        print("walk:", " -> ".join(inv["walk"]))
        print("next branch:", inv["next_branch"])
        print("unresolved:", [u["key"] for u in inv["unresolved"]])
        print("located:", inv["located"]["sentence"])
    for e in v["errors"]:
        print("ERROR:", e["text"])


def main(argv=None) -> int:
    import argparse
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("files", nargs="*")
    s = sub.add_parser("show")
    s.add_argument("id", nargs="?")
    s.add_argument("--root", default=None, help="the data checkout (its test registry); this checkout when absent")
    s.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "check":
        bad = 0
        for p in [Path(f) for f in args.files] or files():
            try:
                problems = check(load(p))
            except InvestigationError as exc:
                problems = [str(exc)]
            print(p, "OK" if not problems else "REFUSED")
            for q in problems:
                print("   ", q)
            bad += bool(problems)
        return 1 if bad else 0
    v = view(data_root=args.root, only=args.id)
    if args.json:
        print(json.dumps(v, indent=1, ensure_ascii=False))
    else:
        _print_view(v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
