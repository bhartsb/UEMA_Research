# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The test registry: one row per test, written BEFORE the test runs.

the style guide "NO TEST RUNS WITHOUT ITS ROW ON THE TESTS PAGE" (operator 2026-10-02: "This entire testing process
seems to be spiraling out of control"). The file is docs/gain_of_function_v4/test_registry.json:

    {"about": ..., "required": [field names], "fields": {field name: its format}, "rows": [row, ...]}

A row's "id" is exactly the pause job's name, or the ability scheduler's task id. This module is the one place
that reads, judges and writes rows:

- run_in_training_pause.py and coordinator/run_jobs_in_held_pause.py call refusals() before anything stops
  training (one job without a complete row refuses its whole jobs file, recorded in a receipt), refusal() again
  right before each job starts (a row cancelled or emptied during a pause refuses that job), and job_started() /
  job_finished() to write the row's status and last_run back;
- tools/ability_rechecks.py calls refusals() every cycle and holds a task without a complete row as
  'repair needed' with the reason;
- tools/development_dashboard.py serves dashboard_view() as GET /api/tests, the Tests page.

A ROW IS COMPLETE when every field the file lists in "required" is present and not empty (None, "", only spaces,
[] or {} or a list/object holding only empty values), status is one of STATUSES, cost holds training_stopped
(true/false), memory_gb (a number) and where (text), and result_source holds file (text) and keys. A complete row
whose status is "cancelled" is refused too: running it would contradict the page. Two rows with one id are
refused: a job could not be matched to one row.

A ROW'S TIME LIMIT is cost.time_limit_s, in seconds (not a required key). For a pause job it is the job's "timeout_s"
in its jobs file: the pause runner ends the job's whole process tree when that many seconds have passed. The Tests
page prints it in the row's Cost section (time_limit_line): "Stops by itself after N seconds" when the row states a
number above 0, "No time limit stated" when it does not.

WRITES never touch another field: update_row() takes a lock file beside the registry, reads the file again, changes
only the named fields of the one row with that id, and replaces the file with os.replace only when its size and
modification time still equal those of that read (another writer's edit in between restarts the read, up to 5
times). The text keeps the file's layout (dumps): one-space indent, cost, result_source, code and short lists on
one line.

ADDED OCT 5, 2026 (the operator's recovery directive, sections 6 to 8; the record is
docs/gain_of_function_v4/dashboard_recovery/DASHBOARD_DATA_LAYER_2026-10-05.md). Nothing below changes what an older
copy of this module reads: "status" keeps its six values, and every new fact sits in an added field that older code
skips.

- A ROW WRITTEN AT OR AFTER ADDED_FROM also fills ADDED_FIELDS: component (an id of the component manifest), gof_link,
  target_metric, baseline, statistical_line, material_gof_line (or the words "not yet defined"), a diagram of four
  short phrases {input, operation, measurement, decision}, and cost.time_limit_s as a number. row_problems() judges
  them, so every runner refuses such a row without them. A row written before that time is judged as before; the
  Tests page lists what it lacks (view key "gaps").
- THE RUNNER'S RECORD. job_started() writes status "running" and last_run {state "running", started, commit,
  checkpoint, owner, resource_class, pid and pid_created of the job's process, runner_pid, run_id, ...}.
  job_finished() reads the outcome (read_outcome: the result file's verdict key, else the exit code), writes
  last_run.state PASS, FAIL, INVALID or STOPPED, the row's "outcome" (a copy of that state), status "done" for PASS and
  "failed" for the other three, the end time, the result file, and "next_action": the row's on_pass after PASS, its
  on_fail after FAIL, and after INVALID or STOPPED the sentence that a follow-up and its owner are owed
  (write_follow_up writes them). Every last_run these two functions write carries a seal (seal_of: a hash of the row
  id and the last_run's own fields), so a last_run typed by hand does not verify.
- A HAND WRITE MAY NOT OVERRIDE THE RUNNER'S STATE. update_row() refuses status running, done or failed and the fields
  outcome, last_run and next_action unless a runner function of this module calls it. By hand a row can be queued,
  blocked (block_row: a reason and what it waits on), unblocked, cancelled, and given its result, decision and
  follow-up. test_runner.audit() reports a runner status that carries no verified seal.
- THE TESTS PAGE'S SECTIONS (section_of) are computed from status, the outcome and the block: QUEUED, RUNNING, BLOCKED,
  FAILED, INVALID, STOPPED, DONE.
- A RESULT'S READINGS (result.readings, a list): each {name, value, unit, meaning, baseline, statistically_clear
  ("yes", "no" or "not judged"), statistical_basis, gof_material ("yes", "no" or "GOF threshold not yet defined")}.
  reading_problems() judges one reading.
"""
from __future__ import annotations

from datetime import datetime
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

REPO = Path(__file__).resolve().parents[4]
REGISTRY_RELATIVE = Path("docs/gain_of_function_v4/test_registry.json")
DEFAULT_REQUIRED = ("id", "title", "question", "why", "inputs", "steps", "measures", "pass_line",
                    "pass_line_basis", "cost", "result_source", "on_pass", "on_fail", "status", "written_at")
# the registry's own list ("written | queued | running | done | cancelled") plus "failed", which the pause runners
# write when a job ends with an exit code other than 0, times out or never starts
STATUSES = ("written", "queued", "running", "done", "failed", "cancelled")
REFUSED_STATUSES = ("cancelled",)
STATUS_LABELS = {"running": "Running", "queued": "Queued", "written": "Written, not yet queued", "failed": "Failed",
                 "done": "Done", "cancelled": "Cancelled"}
# the Tests page's group order; rows with a missing or unknown status come last
GROUP_ORDER = ("running", "queued", "written", "failed", "done", "cancelled")
# plain labels for the page and the refusal sentences (the JSON key is given beside it where a fix needs it)
LABELS = {"id": "job name", "title": "title", "question": "question", "why": "why it matters", "inputs": "inputs",
          "steps": "steps", "measures": "measures", "pass_line": "pass line",
          "pass_line_basis": "basis of the pass line", "cost": "cost", "result_source": "result source",
          "on_pass": "on pass", "on_fail": "on fail", "status": "status", "written_at": "written at",
          "result": "result", "decision": "decision", "code": "code files",
          "component": "component", "gof_link": "gain-of-function link", "target_metric": "direct target metric",
          "baseline": "baseline", "statistical_line": "statistical line",
          "material_gof_line": "material gain-of-function line", "diagram": "diagram"}
LOCK_WAIT_S = 15.0
LOCK_STALE_S = 30.0
INLINE_TEXT = 40          # a list of plain values each at most this long stays on one line when written
INLINE_KEYS = ("required", "cost", "result_source", "code")    # written on one line, as the registry's writers do

# ------------------------------------------------------------------------------------------ added Oct 5, 2026
# THE OUTCOME OF ONE RUN, written by job_finished() and by nothing else (directive section 8).
OUTCOMES = ("PASS", "FAIL", "INVALID", "STOPPED")
OUTCOME_MEANINGS = {
    "PASS": "the test ran to its end and its pass line was met",
    "FAIL": "the test ran to its end and its pass line was not met",
    "INVALID": "the test ended without a judgement: it could not start, it ended in an error, or its result could "
               "not be read; a follow-up and its owner are owed",
    "STOPPED": "the test was ended before its own end: its time limit, a stop request, a trainer process that "
               "started, or a process that was gone when the audit looked; a follow-up and its owner are owed",
}
# what an older copy of this module reads for each outcome: it knows "done" and "failed" only
STATUS_OF_OUTCOME = {"PASS": "done", "FAIL": "failed", "INVALID": "failed", "STOPPED": "failed"}
RUNNER_STATUSES = ("running", "done", "failed")       # written only by a runner function of this module
HAND_STATUSES = ("written", "queued", "cancelled")    # written by hand
RUNNER_FIELDS = ("outcome", "last_run", "next_action")
# the Tests page's sections, in the page's order (directive sections 7A and 8); only the first is open at first load
SECTIONS = ("QUEUED", "RUNNING", "BLOCKED", "FAILED", "INVALID", "STOPPED", "DONE")
SECTION_MEANINGS = {
    "QUEUED": "written and able to start: its row is complete and nothing blocks it",
    "RUNNING": "a runner started it and has not written its end",
    "BLOCKED": "written and not able to start now: a block written into the row, an incomplete row, or the "
               "24-hour rule of the defect register",
    "FAILED": "its last run ended FAIL (or, for a run before Oct 5, 2026, with status failed)",
    "INVALID": "its last run ended INVALID: no judgement was produced",
    "STOPPED": "its last run ended STOPPED, or the row was cancelled by hand",
    "DONE": "its last run ended PASS (or, for a run before Oct 5, 2026, with status done)",
}
SECTION_OF_OUTCOME = {"PASS": "DONE", "FAIL": "FAILED", "INVALID": "INVALID", "STOPPED": "STOPPED"}
OPEN_BY_DEFAULT = ("QUEUED",)               # the sections the Tests page shows expanded on its first load (7A)
# the fields of the contract of Oct 5, 2026 (section 6) and the directive (section 6), required of every row written
# at or after ADDED_FROM; LABELS below gives their plain names
ADDED_FIELDS = ("component", "gof_link", "target_metric", "baseline", "statistical_line", "material_gof_line",
                "diagram")
ADDED_FROM = "Oct 5, 2026 1:23 AM PDT"      # the contract's own time: the first rows that owe the added fields
DIAGRAM_KEYS = ("input", "operation", "measurement", "decision")
DIAGRAM_MAX = 120                           # characters of one diagram phrase: it must fit a box on the Tests page
NOT_YET_DEFINED = "not yet defined"
# the machine resource class of a run (directive sections 8 and 12): which machine mode the run needs
RESOURCE_CLASSES = {
    "BESIDE_TRAINING": "declared able to run while the production trainer runs (cost.training_stopped is false)",
    "TEST_WINDOW": "needs production training stopped (cost.training_stopped is true)",
    "PRODUCTION_SEGMENT": "the run is a segment of production training itself: its process is the trainer",
}
STATISTICAL_ANSWERS = ("yes", "no", "not judged")
GOF_ANSWERS = ("yes", "no", "GOF threshold not yet defined")
READING_KEYS = ("name", "value", "unit", "meaning", "baseline", "statistically_clear", "statistical_basis",
                "gof_material")
DEFAULT_FAIL_EXIT_CODES = (1,)
MANIFEST_RELATIVE = Path("docs/gain_of_function_v4/component_manifest.json")


class RegistryError(Exception):
    """The registry file cannot be read or written as the registry."""


# ------------------------------------------------------------------------------------------ clock times
def _pacific():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("America/Los_Angeles")
    except Exception:            # no time zone data: the machine's own zone (Pacific on this machine)
        return None


PACIFIC = _pacific()


def clock(seconds=None) -> str:
    """A clock time as the operator reads it: "Oct 2, 2026 5:08 PM PDT" (AM/PM, Pacific)."""
    seconds = time.time() if seconds is None else float(seconds)
    if PACIFIC is not None:
        moment = datetime.fromtimestamp(seconds, PACIFIC)
        zone = moment.tzname() or ""
    else:
        moment = datetime.fromtimestamp(seconds)
        name = time.tzname[time.localtime(seconds).tm_isdst > 0]
        zone = "".join(w[0] for w in name.split()) if " " in name else name
    hour = moment.hour % 12 or 12
    return "%s %d, %d %d:%02d %s %s" % (moment.strftime("%b"), moment.day, moment.year, hour, moment.minute,
                                        "AM" if moment.hour < 12 else "PM", zone)


_CLOCK = re.compile(r"\b([A-Z][a-z]{2,8})\.?\s+(\d{1,2}),?\s+(\d{4}),?\s+(?:at\s+)?(\d{1,2}):(\d{2})(?::(\d{2}))?"
                    r"\s*([AaPp])\.?\s*[Mm]\b")
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")


def parse_clock(text):
    """Seconds since 1970 for a clock time written like "Oct 2, 2026 5:08 PM PDT" (read as Pacific time) or an
    ISO time, else None."""
    text = str(text or "")
    match = _CLOCK.search(text)
    if match:
        month = match.group(1)[:3].lower()
        if month not in _MONTHS:
            return None
        hour = int(match.group(4)) % 12 + (12 if match.group(7).lower() == "p" else 0)
        try:
            moment = datetime(int(match.group(3)), _MONTHS.index(month) + 1, int(match.group(2)), hour,
                              int(match.group(5)), int(match.group(6) or 0), tzinfo=PACIFIC)
        except ValueError:
            return None
        return moment.timestamp()
    try:
        moment = datetime.fromisoformat(text.strip())
    except ValueError:
        return None
    if moment.tzinfo is None and PACIFIC is not None:
        moment = moment.replace(tzinfo=PACIFIC)
    return moment.timestamp()


# ------------------------------------------------------------------------------------------ reading
def registry_path(root=None) -> Path:
    """The registry file under repository `root` (production's when None)."""
    return Path(root or REPO) / REGISTRY_RELATIVE


def load(path=None, attempts=3, wait_s=0.4) -> dict:
    """The registry as read from the file now (never a cached copy). A file that cannot be parsed is read again
    up to `attempts` times, `wait_s` apart: another writer may be replacing it. Raises RegistryError."""
    path = Path(path or registry_path())
    last = None
    for attempt in range(attempts):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            raise RegistryError("the registry file %s does not exist" % path) from None
        except (OSError, ValueError) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(wait_s)
            continue
        if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
            raise RegistryError("the registry file %s holds no 'rows' list" % path)
        return data
    raise RegistryError("the registry file %s cannot be read: %s" % (path, last))


def required_fields(data) -> list:
    """The fields every row must fill: the file's own "required" list (DEFAULT_REQUIRED when it has none), with the
    id always included, since jobs are matched to rows by it."""
    declared = data.get("required") if isinstance(data, dict) else None
    if isinstance(declared, list) and declared and all(isinstance(x, str) and x for x in declared):
        fields = list(declared)
    else:
        fields = list(DEFAULT_REQUIRED)
    if "id" not in fields:
        fields.insert(0, "id")
    return fields


def empty(value) -> bool:
    """None, text of only spaces, and a list or object with nothing but empty values; a number or true/false is a
    value."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple)):
        return all(empty(v) for v in value)
    if isinstance(value, dict):
        return all(empty(v) for v in value.values())
    return False


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def row_problems(row, required) -> list:
    """[(key, plain sentence)] for every required field of `row` that is missing, empty or not in its declared
    form; empty when the row is complete."""
    if not isinstance(row, dict):
        return [("row", "the row is not an object")]
    out = []
    for name in required:
        label = LABELS.get(name, name.replace("_", " "))
        if name not in row:
            out.append((name, "%s is missing" % label))
            continue
        value = row[name]
        if empty(value):
            out.append((name, "%s is empty" % label))
        elif name == "id" and not isinstance(value, str):
            out.append((name, "the job name is not text"))
        elif name == "status" and value not in STATUSES:
            out.append((name, "status '%s' is not one of %s" % (value, ", ".join(STATUSES))))
        elif name == "cost":
            if not isinstance(value, dict):
                out.append((name, "cost is not {training_stopped, memory_gb, where}"))
                continue
            if not isinstance(value.get("training_stopped"), bool):
                out.append(("cost.training_stopped", "cost does not say whether training is stopped (true or false)"))
            if not _number(value.get("memory_gb")) or value["memory_gb"] < 0:
                out.append(("cost.memory_gb", "cost gives no memory in GB"))
            if empty(value.get("where")) or not isinstance(value.get("where"), str):
                out.append(("cost.where", "cost does not say where it runs"))
        elif name == "result_source":
            if not isinstance(value, dict):
                out.append((name, "the result source is not {file, keys}"))
                continue
            if empty(value.get("file")) or not isinstance(value.get("file"), str):
                out.append(("result_source.file", "the result source names no file"))
            if empty(value.get("keys")):
                out.append(("result_source.keys", "the result source names no keys"))
    return out + added_problems(row)


_COMPONENT_IDS = {}


def component_ids(root=None):
    """The ids of the component manifest (docs/gain_of_function_v4/component_manifest.json under `root`, this module's
    own checkout when None) as a set, read again when the file changes; None when the file is missing or cannot be
    read, and then no row is judged on its component's spelling."""
    path = Path(root or REPO) / MANIFEST_RELATIVE
    stamp = _stamp(path)
    if stamp is None:
        return None
    cached = _COMPONENT_IDS.get(str(path))
    if cached and cached[0] == stamp:
        return cached[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        ids = {c["id"] for c in data["components"] if isinstance(c, dict) and isinstance(c.get("id"), str)}
    except (OSError, ValueError, KeyError, TypeError):
        return None
    _COMPONENT_IDS[str(path)] = (stamp, ids)
    return ids


def owes_added_fields(row) -> bool:
    """True for a row written at or after ADDED_FROM (the contract of Oct 5, 2026): it owes ADDED_FIELDS. A row whose
    written_at cannot be read as a time owes them too: only a readable earlier time exempts a row."""
    written = parse_clock(row.get("written_at")) if isinstance(row, dict) else None
    return written is None or written >= parse_clock(ADDED_FROM)


def diagram_problems(diagram) -> list:
    """[(key, plain sentence)] for a diagram that is not four short phrases {input, operation, measurement,
    decision}; empty when it is."""
    if not isinstance(diagram, dict):
        return [("diagram", "the diagram is not {input, operation, measurement, decision}")]
    out = []
    for key in DIAGRAM_KEYS:
        phrase = diagram.get(key)
        if not isinstance(phrase, str) or not phrase.strip():
            out.append(("diagram." + key, "the diagram has no %s phrase" % key))
        elif len(phrase.strip()) > DIAGRAM_MAX:
            out.append(("diagram." + key, "the diagram's %s phrase is %d characters, above the %d a box holds"
                        % (key, len(phrase.strip()), DIAGRAM_MAX)))
    return out


def added_problems(row) -> list:
    """[(key, plain sentence)] for the fields of Oct 5, 2026 that a row written at or after ADDED_FROM lacks; empty
    for an earlier row. The diagram is not asked of a row whose status is done (directive section 7A: a done test
    may omit it)."""
    if not isinstance(row, dict) or not owes_added_fields(row):
        return []
    out = []
    for name in ADDED_FIELDS:
        if name == "diagram" and row.get("status") == "done" and empty(row.get("diagram")):
            continue
        label = LABELS.get(name, name.replace("_", " "))
        if name not in row:
            out.append((name, "%s is missing" % label))
        elif empty(row[name]):
            out.append((name, "%s is empty" % label))
        elif name == "diagram":
            out += diagram_problems(row[name])
        elif not isinstance(row[name], str):
            out.append((name, "%s is not text" % label))
        elif name == "component":
            known = component_ids()
            if known is not None and row[name].strip() not in known:
                out.append((name, "component '%s' is not an id of the component manifest" % row[name].strip()))
    cost = row.get("cost")
    if isinstance(cost, dict) and not (_number(cost.get("time_limit_s")) and cost["time_limit_s"] > 0):
        out.append(("cost.time_limit_s", "cost gives no time limit in seconds"))
    return out


def gaps(row) -> list:
    """The plain names of the directive's section 6 fields a row does not fill, whenever it was written: the Tests
    page prints them for a row written before ADDED_FROM, which the runners do not refuse for them."""
    if not isinstance(row, dict):
        return []
    out = [LABELS.get(name, name) for name in ADDED_FIELDS
           if empty(row.get(name)) or (name == "diagram" and diagram_problems(row.get(name)))]
    cost = row.get("cost")
    if not (isinstance(cost, dict) and _number(cost.get("time_limit_s")) and cost["time_limit_s"] > 0):
        out.append("time limit in seconds")
    return out


def reading_problems(reading) -> list:
    """Plain sentences for a result reading that is not {name, value, unit, meaning, baseline, statistically_clear,
    statistical_basis, gof_material} in its declared form; empty when it is. value may be a number or text; unit may
    be empty text for a count; the two judgements are from STATISTICAL_ANSWERS and GOF_ANSWERS; a statistical answer
    of yes or no needs its basis."""
    if not isinstance(reading, dict):
        return ["the reading is not an object"]
    out = []
    for key in ("name", "meaning", "baseline"):
        if not isinstance(reading.get(key), str) or not reading[key].strip():
            out.append("the reading has no %s" % key)
    if empty(reading.get("value")) and reading.get("value") != 0:
        out.append("the reading has no value")
    if not isinstance(reading.get("unit"), str):
        out.append("the reading has no unit (empty text for a plain count)")
    if reading.get("statistically_clear") not in STATISTICAL_ANSWERS:
        out.append("statistically clear is not one of %s" % ", ".join(STATISTICAL_ANSWERS))
    elif reading["statistically_clear"] != "not judged" and empty(reading.get("statistical_basis")):
        out.append("statistically clear says %s without its basis" % reading["statistically_clear"])
    if reading.get("gof_material") not in GOF_ANSWERS:
        out.append("large enough to matter for gain of function is not one of %s" % ", ".join(GOF_ANSWERS))
    return out


def reading_lines(reading) -> list:
    """One reading as the Tests page prints it: the value with its unit and meaning, the baseline, then the two plain
    judgements (directive section 7A)."""
    if not isinstance(reading, dict):
        return []
    unit = str(reading.get("unit") or "").strip()
    value = reading.get("value")
    shown = _number_text(value) if _number(value) else "; ".join(_lines(value)) or "no value"
    head = "%s: %s%s" % (str(reading.get("name") or "a reading").strip(), shown, (" " + unit) if unit else "")
    if not empty(reading.get("meaning")):
        head += " (%s)" % str(reading["meaning"]).strip()
    out = [head]
    if not empty(reading.get("baseline")):
        out.append("Baseline: %s" % str(reading["baseline"]).strip())
    clear = reading.get("statistically_clear")
    basis = str(reading.get("statistical_basis") or "").strip()
    out.append("Statistically clear: %s%s" % (clear if clear in STATISTICAL_ANSWERS else "not judged",
                                              (" (%s)" % basis) if basis else ""))
    material = reading.get("gof_material")
    out.append("Large enough to matter for gain of function: %s"
               % (material if material in GOF_ANSWERS else GOF_ANSWERS[2]))
    return out


def rows_with_id(data, job_id) -> list:
    return [r for r in data.get("rows", []) if isinstance(r, dict) and r.get("id") == job_id]


def _near_id(data, job_id):
    ids = [r.get("id") for r in data.get("rows", []) if isinstance(r, dict) and isinstance(r.get("id"), str)]
    folded = [i for i in ids if i.strip().casefold() == str(job_id).strip().casefold()]
    if folded:
        return folded[0]
    near = difflib.get_close_matches(str(job_id), ids, n=1, cutoff=0.8)
    return near[0] if near else None


def judge(job_id, data):
    """None when exactly one row has id `job_id`, it is complete and not cancelled; else (kind, detail): ("missing",
    a near id or None), ("duplicate", row count), ("incomplete", [(key, sentence)]) or ("cancelled", None)."""
    rows = rows_with_id(data, job_id)
    if not rows:
        return "missing", _near_id(data, job_id)
    if len(rows) > 1:
        return "duplicate", len(rows)
    problems = row_problems(rows[0], required_fields(data))
    if problems:
        return "incomplete", problems
    if rows[0].get("status") in REFUSED_STATUSES:
        return "cancelled", None
    return None


def block_refusal(row):
    """Why a row's block holds its job back, or None (block_row writes the block; unblock_row removes it)."""
    block = is_blocked(row)
    if not block:
        return None
    return ("its row is blocked: %s; it waits on: %s (test_runner unblock removes the block)"
            % (str(block["reason"]).rstrip("."), block.get("waits_on") or "not stated"))


def rule_refusal(row, data):
    """Why the defect register's 24-hour rule holds the row's job back now, or None (defect_register.runner_gate). A
    register that cannot be read refuses every job, as an unreadable test registry does: a rule that a missing or broken
    file switches off is no rule."""
    try:
        from . import defect_register
        return defect_register.runner_gate(row, rows=data.get("rows") if isinstance(data, dict) else None)
    except Exception as exc:
        return ("the defect register's 24-hour rule could not be applied (%s: %s); every test is refused until the "
                "register can be read" % (type(exc).__name__, exc))


def refusal(job_id, data=None, path=None, check_rule=True, lifecycle_supported=False, preserve_production=False):
    """Why job `job_id` may not run, as one sentence naming the keys to fix, or None when it may: no single complete
    row that is not cancelled has the id (judge); the row carries a block (block_refusal); the defect register's
    24-hour rule holds it (rule_refusal; with check_rule false this one is not read). Every runner reads this one
    function before a job starts: test_runner run, the two pause runners and the ability scheduler (the dashboard
    review of Oct 5, 2026, finding F3: the block and the rule bound test_runner alone). Reads the file when `data` is
    not given."""
    if data is None:
        try:
            data = load(path)
        except RegistryError as exc:
            return "the test registry cannot be read (%s); every job is refused until it can" % exc
    verdict = judge(job_id, data)
    if verdict is None:
        row = rows_with_id(data, job_id)[0]
        try:
            from . import file_lifecycle
            files_why = None if preserve_production else file_lifecycle.runner_refusal(row, data)
            if not files_why and data.get("file_lifecycle_policy") and not lifecycle_supported and not preserve_production:
                files_why = "file lifecycle: this runner must use test_runner.run_command for managed output and close-out"
        except Exception as exc:
            files_why = "file lifecycle check failed: %s: %s" % (type(exc).__name__, exc)
        return block_refusal(row) or files_why or (rule_refusal(row, data) if check_rule else None)
    kind, detail = verdict
    if kind == "missing":
        return ("no row in the test registry has id '%s'%s; write its row before the job runs"
                % (job_id, " (a row has the near id '%s': the id must equal the job name exactly)" % detail if detail else ""))
    if kind == "duplicate":
        return "%d rows in the test registry have id '%s'; keep one row per test" % (detail, job_id)
    if kind == "incomplete":
        return "its row in the test registry is incomplete: " + "; ".join("%s (%s)" % (text, key) for key, text in detail)
    return "its row in the test registry says cancelled; set the status to queued to run it"


def plain_refusal(job_id, data):
    """judge() as a plain sentence for the Tests page (no key names), or None."""
    verdict = judge(job_id, data)
    if verdict is None:
        return None
    kind, detail = verdict
    if kind == "missing":
        return "No row yet."
    if kind == "duplicate":
        return "%d rows share its job name." % detail
    if kind == "incomplete":
        return "Its row is incomplete: " + "; ".join(text for _, text in detail) + "."
    return "Its row says cancelled."


def refusals(job_ids, path=None, check_rule=True) -> dict:
    """{job id: reason} for every job of `job_ids` that may not run (refusal: the row, its block, the 24-hour rule),
    from one read of the file; {} when all may."""
    try:
        data = load(path)
    except RegistryError as exc:
        return {j: "the test registry cannot be read (%s); every job is refused until it can" % exc for j in job_ids}
    out = {}
    for job_id in job_ids:
        why = refusal(job_id, data=data, path=path, check_rule=check_rule)
        if why:
            out[job_id] = why
    return out


# ------------------------------------------------------------------------------------------ writing
def _short_plain(value) -> bool:
    return isinstance(value, (list, tuple)) and all(
        v is None or isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) <= INLINE_TEXT) for v in value)


def dumps(value, level=0, key=None) -> str:
    """JSON text in the registry's own layout (it reproduces the file of Oct 2, 2026 5:27 PM byte for byte): a
    one-space indent per level; the values of INLINE_KEYS, and a list of plain values each at most INLINE_TEXT
    characters, on one line; every other object or list one item per line."""
    pad, close = " " * (level + 1), " " * level
    if isinstance(value, (dict, list, tuple)) and value and (key in INLINE_KEYS or _short_plain(value)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [pad + json.dumps(str(k), ensure_ascii=False) + ": " + dumps(v, level + 1, str(k))
                 for k, v in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + close + "}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        return "[\n" + ",\n".join(pad + dumps(v, level + 1) for v in value) + "\n" + close + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _stamp(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


class _Lock:
    """registry.json.lock beside the file: one writer of this module at a time (O_CREAT | O_EXCL). A lock older than
    LOCK_STALE_S is a crashed writer's and is removed."""

    def __init__(self, path):
        self.path = Path(str(path) + ".lock")

    def __enter__(self):
        deadline = time.monotonic() + LOCK_WAIT_S
        while True:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > LOCK_STALE_S:
                        self.path.unlink(missing_ok=True)
                        continue
                except OSError:
                    continue
                if time.monotonic() > deadline:
                    raise RegistryError("the registry's lock file %s stayed taken for %d s" % (self.path, LOCK_WAIT_S))
                time.sleep(0.05)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(str(os.getpid()))
            return self

    def __exit__(self, *exc):
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _replace(path, text):
    tmp = Path(path).with_name(Path(path).name + ".%d.tmp" % os.getpid())
    with open(tmp, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    for attempt in range(40):          # Windows refuses the swap while another process holds the file open
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 39:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.05)


_RUNNER = object()          # passed by this module's runner functions; a hand write does not hold it


def hand_refusal(fields):
    """Why a hand write of `fields` is refused, or None: status running, done or failed and the fields outcome,
    last_run and next_action are the runner's (directive section 8: no manually written status may override the
    runner's state)."""
    if not isinstance(fields, dict):
        return "the fields are not an object"
    if "status" in fields and fields["status"] in RUNNER_STATUSES:
        return ("status '%s' is written only by a runner function (job_started, job_finished); by hand a row can be "
                "set to %s" % (fields["status"], ", ".join(HAND_STATUSES)))
    taken = [name for name in RUNNER_FIELDS if name in fields]
    if taken:
        return ("%s %s written only by a runner function (job_started, job_finished)"
                % (", ".join(taken), "is" if len(taken) == 1 else "are"))
    return None


def update_row(row_id, fields, path=None, runner=None) -> bool:
    """Write `fields` (a dict) into the one row whose id is `row_id`, touching nothing else (see the module text).
    True when written; False when no row, or more than one, has that id. Raises RegistryError when the file cannot
    be read or written, and when a caller that is not one of this module's runner functions names a field of the
    runner (hand_refusal). `fields` may be a function of the row as read under the lock, returning the dict to write
    (or None to write nothing and return False): a writer that decides from the row's present state."""
    if runner is not _RUNNER and not callable(fields):
        why = hand_refusal(fields)
        if why:
            raise RegistryError(why)
    path = Path(path or registry_path())
    with _Lock(path):
        for _ in range(5):
            before = _stamp(path)
            data = load(path)
            rows = rows_with_id(data, row_id)
            if len(rows) != 1:
                return False
            new = fields(rows[0]) if callable(fields) else fields
            if new is None:
                return False
            if runner is not _RUNNER:
                why = hand_refusal(new)
                if why:
                    raise RegistryError(why)
            rows[0].update(new)
            try:
                text = dumps(data) + "\n"
            except ValueError as exc:
                raise RegistryError("the registry holds a value JSON cannot write: %s" % exc) from exc
            if _stamp(path) != before:
                continue                      # another writer replaced the file after the read: read it again
            try:
                _replace(path, text)
            except OSError as exc:
                raise RegistryError("the registry file %s could not be replaced: %s" % (path, exc)) from exc
            return True
    raise RegistryError("the registry file %s changed during 5 attempts to write row %s" % (path, row_id))


def _write(row_id, fields, path, runner=None):
    try:
        if update_row(row_id, fields, path=path, runner=runner):
            return None
        return "no single row has id '%s'" % row_id
    except RegistryError as exc:
        return str(exc)
    except Exception as exc:              # a write-back never stops a runner
        return "%s: %s" % (type(exc).__name__, exc)


# ------------------------------------------------------------------------------------------ the runner's record
def seal_of(row_id, last_run) -> str:
    """The seal of a last_run written by job_started() or job_finished(): the first 24 hex characters of the SHA-256
    of the JSON text of [row id, the last_run without its "seal" key] with sorted keys. It needs no secret, so it
    does not stop a deliberate forgery by someone who calls this function; it makes a last_run typed or edited by
    hand fail sealed(), which test_runner.audit() reports."""
    body = {k: v for k, v in (last_run or {}).items() if k != "seal"}
    text = json.dumps([str(row_id), body], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def sealed(row) -> bool:
    """True when the row's last_run carries the seal its own fields give."""
    last = row.get("last_run") if isinstance(row, dict) else None
    return (isinstance(last, dict) and isinstance(last.get("seal"), str)
            and last["seal"] == seal_of(row.get("id"), last))


# THE RUNNER'S OWN RESULT (the dashboard review of Oct 5, 2026, finding B2: "a box is decided only by a sealed runner
# record and a result file written during that run"). The row's "result" field can be written by hand (update_row):
# a person or an agent adds readings, a decision, a value. finish_job() therefore seals, inside the run's last_run,
#   result_digest       the digest of the result it wrote (result_digest below), and
#   result_file_record  {path, sha256, size, mtime} of the row's result file when that file was written during the run
#                       (its modification time at or after the run's start, less 2 s), read at the run's end.
# runner_result() and runner_file() answer whether a value read now is still the runner's: a result whose digest
# differs, or a file whose SHA-256 differs, holds something written after the run ended.
FILE_RECORD_MAX = 64 * 2 ** 20          # a larger result file is not hashed at the run's end; it decides nothing


def result_digest(result) -> str:
    """The first 24 hexadecimal characters of the SHA-256 of a result's JSON text with sorted keys."""
    text = json.dumps(result, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path):
    """{path, sha256, size, mtime} of a file as it is now; None when it cannot be read or is larger than
    FILE_RECORD_MAX."""
    try:
        stat = os.stat(path)
        if stat.st_size > FILE_RECORD_MAX:
            return None
        return dict(path=str(path), sha256=_file_sha256(path), size=stat.st_size, mtime=stat.st_mtime)
    except OSError:
        return None


def runner_result(row):
    """(the row's result, None) when it is the result the runner wrote at the end of the row's last run: the last run
    carries a verified seal, its result_digest equals the digest of the row's result now, and the result's run id is
    that run's. Else (None, why not): a runner record of the older kind, no runner record, a run sealed without a
    result digest, or a result changed after the run ended (a value typed into the row)."""
    last = row.get("last_run") if isinstance(row, dict) and isinstance(row.get("last_run"), dict) else None
    if not last:
        return None, "the row has no runner record"
    if not sealed(row):
        return None, ("a runner record of the older kind (%s), not sealed: it decides nothing"
                      % (last.get("runner") or "its runner is not named"))
    result = row.get("result")
    if not isinstance(result, dict):
        return None, "the row holds no result"
    if result.get("run_id") != last.get("run_id"):
        return None, "the row's result belongs to another run than its last one"
    digest = last.get("result_digest")
    if not digest:
        return None, ("the runner's record of this run holds no digest of its result (a run ended before Oct 5, "
                      "2026's rule), so its result cannot be told from a value typed into the row")
    if digest != result_digest(result):
        return None, "typed into the row: the row's result differs from the result the runner sealed at the run's end"
    return result, None


def runner_file(row, path):
    """None when `path` is the result file the runner recorded at the end of the row's sealed last run and its content
    is unchanged since (the same SHA-256); else why not."""
    last = row.get("last_run") if isinstance(row, dict) and isinstance(row.get("last_run"), dict) else None
    if not last:
        return "the row has no runner record"
    if not sealed(row):
        return ("a runner record of the older kind (%s), not sealed: it decides nothing"
                % (last.get("runner") or "its runner is not named"))
    record = last.get("result_file_record") if isinstance(last.get("result_file_record"), dict) else None
    if record is None:
        return ("the runner recorded no result file written during this run (the row's result file was not "
                "written during the run, was larger than 64 MB, or the run ended before Oct 5, 2026's rule)")
    try:
        same_path = Path(record.get("path") or "").resolve() == Path(path).resolve()
    except (OSError, ValueError):
        same_path = False
    if not same_path:
        return "the runner recorded another result file for this run (%s)" % record.get("path")
    try:
        now = _file_sha256(path)
    except OSError as exc:
        return "the result file could not be read (%s)" % exc
    if now != record.get("sha256"):
        return ("written into the file after the run ended: its content differs from the content the runner "
                "recorded at the run's end")
    return None


def run_state(row):
    """The state the runner last wrote for the row: "running" or one of OUTCOMES (last_run.state), or None for a row
    no runner of Oct 5, 2026 or later has written."""
    last = row.get("last_run") if isinstance(row, dict) else None
    state = last.get("state") if isinstance(last, dict) else None
    return state if state == "running" or state in OUTCOMES else None


def recorded_outcome(row):
    """The outcome the runner's own record holds for the row's last run: last_run.state when it is one of OUTCOMES,
    else None (a row no runner of Oct 5, 2026 or later has finished; a row whose last run was written afterwards by a
    runner of the older kind, which replaces last_run). The row's "outcome" field is a copy for a reader of the
    file: every computed view reads this function, so a copy typed by hand, or left by an earlier run, decides
    nothing."""
    state = run_state(row)
    return state if state in OUTCOMES else None


def verified_outcome(row):
    """recorded_outcome() when the row's last_run carries the seal its fields give, else None: the outcome a runner
    function wrote and nobody edited afterwards."""
    return recorded_outcome(row) if sealed(row) else None


def _time_span(seconds, text):
    """(earliest, latest) seconds of a time a row records: an exact number gives (t, t); a clock text, which holds the
    minute only, gives the minute's first and last second; (None, None) when neither can be read."""
    if _number(seconds):
        return float(seconds), float(seconds)
    at = parse_clock(text)
    return (None, None) if at is None else (at, at + 59.999)


def current_follow_up(row):
    """The row's follow-up {text, owner, by, at, at_s} when it belongs to the row's last run, else None.

    A follow-up answers a run that has ended, so it belongs to the last run when it was written after that run's end
    (while no end is recorded, after its start). The rule compares the latest second the run can have ended with the
    earliest second the follow-up can have been written: a runner of the older kind records its times to the minute,
    and a follow-up written in that same minute cannot be placed after its run, so it is not shown for that run. A row
    that never ran, and a follow-up with no readable time, keep the follow-up."""
    follow = row.get("follow_up") if isinstance(row, dict) and isinstance(row.get("follow_up"), dict) else None
    if not follow or empty(follow.get("text")):
        return None
    last = row.get("last_run") if isinstance(row.get("last_run"), dict) else {}
    _, run_latest = _time_span(last.get("ended_at"), last.get("ended"))
    if run_latest is None:
        _, run_latest = _time_span(last.get("started_at"), last.get("started"))
    written_earliest, _ = _time_span(follow.get("at_s"), follow.get("at"))
    if run_latest is not None and written_earliest is not None and written_earliest < run_latest:
        return None
    return follow


def current_result(row):
    """(the row's result, True) when it is the result of the row's last run or a result typed without a run id (a row
    written before Oct 5, 2026); (the result, False) when it carries the run id of an earlier run."""
    result = row.get("result") if isinstance(row, dict) else None
    last = row.get("last_run") if isinstance(row, dict) and isinstance(row.get("last_run"), dict) else {}
    if isinstance(result, dict) and result.get("run_id") and result.get("run_id") != last.get("run_id"):
        return result, False
    return result, True


def process_created(pid):
    """The creation time of process `pid` (seconds since 1970), or None when it has ended or cannot be read."""
    try:
        import psutil
        return psutil.Process(int(pid)).create_time()
    except Exception:
        return None


_CODE_STATE = {}


def code_state(cwd=None, keep_s=30.0):
    """(short commit hash, True when tracked files differ from it, branch name) of the git repository holding `cwd`
    (this process's working directory when None); (None, None, None) outside a repository or without git. One
    answer per folder is kept for `keep_s` seconds: a runner asks once per job."""
    folder = str(Path(cwd or os.getcwd()).resolve())
    kept = _CODE_STATE.get(folder)
    if kept and time.time() - kept[0] < keep_s:
        return kept[1]
    import subprocess

    def git(*arguments):
        try:
            done = subprocess.run(["git", "-C", folder, *arguments], capture_output=True, text=True, timeout=60,
                                  stdin=subprocess.DEVNULL,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None
    head = git("rev-parse", "--short=9", "HEAD")
    if not head:
        answer = (None, None, None)
    else:
        changed = git("status", "--porcelain", "--untracked-files=no")
        answer = (head, None if changed is None else bool(changed), git("branch", "--show-current") or None)
    if len(_CODE_STATE) > 64:
        _CODE_STATE.clear()
    _CODE_STATE[folder] = (time.time(), answer)
    return answer


def default_resource_class(row) -> str:
    """The resource class a row's own cost declares: TEST_WINDOW when cost.training_stopped is true, else
    BESIDE_TRAINING (PRODUCTION_SEGMENT is never a default: the caller that launches a trainer states it)."""
    cost = row.get("cost") if isinstance(row, dict) else None
    return "TEST_WINDOW" if isinstance(cost, dict) and cost.get("training_stopped") is True else "BESIDE_TRAINING"


def new_run_id(row_id, started_at) -> str:
    """An id for one run of one row: its start as YYYYmmdd_HHMMSS, the runner's process id and 4 hex characters of
    the row id's hash."""
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(float(started_at)))
    return "%s_%d_%s" % (stamp, os.getpid(), hashlib.sha256(str(row_id).encode("utf-8")).hexdigest()[:4])


def _text_or_none(value):
    return None if value is None else str(value)


def job_started(row_id, started=None, log=None, runner=None, receipt=None, path=None, *, pid=None, pid_created=None,
                runner_pid=None, commit=None, commit_dirty=None, checkpoint=None, owner=None, resource_class=None,
                device=None, command=None, run_id=None, production_run=None, cwd=None, time_limit_s=None):
    """A RUNNER FUNCTION. Status "running", outcome and next_action cleared, and a new sealed last_run:
    {state "running", started, started_at, ended null, exit_code null, log, outcome "running", runner, receipt,
    run_id, commit, commit_dirty, checkpoint, owner, resource_class, device, pid, pid_created, runner_pid,
    runner_pid_created, command, production_run, time_limit_s (when the runner states the limit the run used)}.

    pid is the job's own process once it exists (job_process() writes it after the start); runner_pid is the calling
    process unless given. commit, the dirty flag and the branch come from the git repository of `cwd` (code_state)
    unless `commit` is given. The owner is the argument, else the environment's UEMA_TEST_OWNER, else the row's own
    "owner", else that branch, else "not recorded". The resource class is the argument, else the row's own cost
    (default_resource_class). None when written, else why not (never raises)."""
    try:
        data = load(path)
        lifecycle_policy = (None if resource_class == "PRODUCTION_SEGMENT" and production_run and pid
                            else data.get("file_lifecycle_policy"))
        from . import file_lifecycle
        rows = rows_with_id(data, row_id)
        if len(rows) == 1:
            files_why = (file_lifecycle.runner_refusal(rows[0], data) if lifecycle_policy is not None
                         or not (resource_class == "PRODUCTION_SEGMENT" and production_run and pid) else None)
            if files_why:
                return files_why
    except Exception as exc:
        return "file lifecycle start check failed: %s: %s" % (type(exc).__name__, exc)
    started_at = time.time() if started is None else float(started)
    lifecycle_run_id = run_id or new_run_id(row_id, started_at)
    claimed_row = json.loads(json.dumps(rows[0])) if len(rows) == 1 else None
    try:
        if len(rows) == 1:
            file_lifecycle.claim_start(claimed_row, lifecycle_policy, lifecycle_run_id, command)
    except Exception as exc:
        return "file lifecycle start claim failed: %s: %s" % (type(exc).__name__, exc)
    runner_pid = os.getpid() if runner_pid is None else int(runner_pid)
    branch = None
    if commit is None:
        commit, commit_dirty, branch = code_state(cwd)
    if pid is not None and pid_created is None:
        pid_created = process_created(pid)

    def fields(row):
        if lifecycle_policy and (row.get("file_lifecycle") != claimed_row.get("file_lifecycle")
                                                  or row.get("result_source") != claimed_row.get("result_source")):
            raise RegistryError("output declaration changed during the start claim; no job started")
        last = dict(
            state="running", started=clock(started_at), started_at=started_at, ended=None, ended_at=None,
            exit_code=None, log=_text_or_none(log), outcome="running", runner=runner, receipt=_text_or_none(receipt),
            run_id=lifecycle_run_id, commit=commit, commit_dirty=commit_dirty,
            checkpoint=_text_or_none(checkpoint if checkpoint is not None else row.get("checkpoint")),
            owner=str(owner or os.environ.get("UEMA_TEST_OWNER") or row.get("owner") or branch or "not recorded"),
            resource_class=resource_class if resource_class in RESOURCE_CLASSES else default_resource_class(row),
            device=str(device or "cpu"), pid=None if pid is None else int(pid), pid_created=pid_created,
            runner_pid=runner_pid, runner_pid_created=process_created(runner_pid),
            command=None if command is None else [str(x) for x in command][:60],
            production_run=_text_or_none(production_run))
        if _number(time_limit_s):
            last["time_limit_s"] = float(time_limit_s)      # the limit the run used (a runner that states one)
        if lifecycle_policy:
            last["file_lifecycle"] = {"declaration": claimed_row["file_lifecycle"], "result_source": claimed_row["result_source"]}
        last["seal"] = seal_of(row_id, last)
        return dict(status="running", outcome=None, next_action=None, follow_up=None, last_run=last)
    why = _write(row_id, fields, path, runner=_RUNNER)
    if why and claimed_row:
        try:
            file_lifecycle.failed_start(claimed_row, lifecycle_policy, lifecycle_run_id, why)
        except Exception as exc:
            why += "; start claim needs reconciliation: " + str(exc)
    return why


def job_process(row_id, pid, path=None, run_id=None, pid_created=None):
    """A RUNNER FUNCTION. Write the started job's process id and creation time into the running last_run (of run
    `run_id` when given) and seal it again. None when written, else why not (never raises)."""
    created = process_created(pid) if pid_created is None else pid_created

    def fields(row):
        last = row.get("last_run")
        if not isinstance(last, dict) or last.get("state") != "running" or (run_id and last.get("run_id") != run_id):
            return None
        last = dict(last, pid=int(pid), pid_created=created)
        last["seal"] = seal_of(row_id, last)
        return dict(last_run=last)
    why = _write(row_id, fields, path, runner=_RUNNER)
    return "the row holds no running last_run of this run" if why and why.startswith("no single row") else why


_VERDICT_WORDS = {"pass": "PASS", "passed": "PASS", "fail": "FAIL", "failed": "FAIL", "invalid": "INVALID"}


def _verdict_of(value):
    """PASS, FAIL or INVALID for a result file's verdict value (true/false, or one of _VERDICT_WORDS in any case);
    None for anything else."""
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    if isinstance(value, str):
        return _VERDICT_WORDS.get(value.strip().lower())
    return None


def result_file_of(row, root=None):
    """The row's result_source.file as a path when it names one file that exists now (absolute, or relative to
    `root`), else None: many rows describe their source in words."""
    source = row.get("result_source") if isinstance(row, dict) else None
    name = source.get("file") if isinstance(source, dict) else None
    if not isinstance(name, str) or not name.strip() or any(c in name for c in "<>*?|"):
        return None
    target = Path(name.strip())
    if not target.is_absolute():
        target = Path(root or REPO) / target
    try:
        return target if target.is_file() else None
    except OSError:
        return None


def read_outcome(row, exit_code, started_at=None, *, stopped=None, not_started=None, root=None):
    """HOW A RUN'S OUTCOME IS READ: (state, why, result file or None, readings or None).

    1. not_started (text: why the job never started) gives INVALID; stopped (text: who or what ended the job before
       its end) gives STOPPED.
    2. Else the row's result file (result_source.file, when it names one existing JSON file written at or after the
       start of the run, at most 50 MB) is read for its verdict key: result_source.verdict_key when the row names
       one, else the first of "verdict" and "passed" the file holds. true, "pass" or "passed" give PASS; false,
       "fail" or "failed" give FAIL; "invalid" gives INVALID. A PASS verdict beside an exit code other than 0 gives
       INVALID: the file and the process disagree. The file's "readings" list, when it holds one, is returned.
    3. Else, when the row sets result_source.verdict_required to true: INVALID.
    4. Else the exit code: 0 gives PASS; a code in result_source.fail_exit_codes (DEFAULT_FAIL_EXIT_CODES, 1, when
       the row names none) gives FAIL; every other code, and no code, gives INVALID.
    An uncaught Python error also exits with code 1, so step 4 reads it as FAIL: a row whose job can end that way
    names a verdict key, or sets fail_exit_codes to []."""
    if not_started:
        return "INVALID", str(not_started), None, None
    if stopped:
        return "STOPPED", str(stopped), None, None
    source = row.get("result_source") if isinstance(row.get("result_source"), dict) else {}
    target = result_file_of(row, root)
    missing = "the row names no single existing result file"
    verdict = key = readings = None
    if target is not None:
        try:
            stat = target.stat()
            if started_at is not None and stat.st_mtime < float(started_at) - 2.0:
                missing = "%s was last written before this run started" % target
            elif target.suffix.lower() != ".json" or stat.st_size > 50 * 2 ** 20:
                missing = "%s is not a JSON file of at most 50 MB" % target
            else:
                payload = json.loads(target.read_text(encoding="utf-8-sig"))
                if isinstance(payload, dict):
                    names = [source["verdict_key"]] if isinstance(source.get("verdict_key"), str) else ["verdict",
                                                                                                         "passed"]
                    key = next((n for n in names if n in payload), None)
                    verdict = _verdict_of(payload.get(key)) if key else None
                    if isinstance(payload.get("readings"), list):
                        readings = payload["readings"]
                    missing = ("%s holds no key %s" % (target, " or ".join(names)) if key is None else
                               "key '%s' of %s holds no verdict word" % (key, target))
                else:
                    missing = "%s does not hold a JSON object" % target
        except (OSError, ValueError) as exc:
            missing = "%s could not be read (%s)" % (target, exc)
    if verdict is not None:
        if verdict == "PASS" and exit_code not in (0, None):
            return ("INVALID", "key '%s' of %s says PASS but the job ended with exit code %s" % (key, target, exit_code),
                    str(target), readings)
        return verdict, "key '%s' of %s" % (key, target), str(target), readings
    if source.get("verdict_required") is True:
        return "INVALID", "the row requires a verdict key and none was read: %s" % missing, None, None
    codes = source.get("fail_exit_codes")
    fail_codes = tuple(codes) if isinstance(codes, list) else DEFAULT_FAIL_EXIT_CODES
    note = "exit code %s (no verdict key was read: %s)" % (exit_code, missing)
    if exit_code == 0:
        return "PASS", note, None, None
    if exit_code is not None and exit_code in fail_codes:
        return "FAIL", note, None, None
    return "INVALID", note, None, None


def follow_up_owed_text(state, why, owner) -> str:
    return ("FOLLOW-UP OWED: this run ended %s (%s). Its owner (%s) writes what happens next and who does it into "
            "the row (test_registry.write_follow_up)." % (state, why, owner))


def finish_job(row_id, exit_code=None, outcome=None, started=None, ended=None, log=None, runner=None, receipt=None,
               path=None, *, stopped=None, not_started=None, run_id=None, root=None, state=None, judged_by=None,
               result_file=None):
    """A RUNNER FUNCTION, and the one place a run's end is written. Reads the outcome (read_outcome; `state` with
    `judged_by` replaces the reading for a run no wrapper owns, such as a production segment) and writes: status
    "done" for PASS and "failed" for FAIL, INVALID and STOPPED; "outcome" (the state); "next_action" (the row's
    on_pass after PASS, its on_fail after FAIL, the sentence that a follow-up is owed after INVALID and STOPPED);
    "result" {read_at, verdict, judged_by, file, readings} with the row's earlier result kept as previous_result;
    and the sealed last_run with ended, ended_at, exit_code, state, judged_by, result_file, next_action and
    next_action_owner, keeping the fields job_started() wrote for the same run.

    `outcome` as text is kept as last_run.outcome (the sentence an older page prints); text that starts "not
    started" or "timed out" or "stopped" also sets not_started or stopped, which is how callers written before
    Oct 5, 2026 say it. Returns (why the write failed or None, the state written or None); never raises."""
    ended_at = time.time() if ended is None else float(ended)
    text = outcome.strip() if isinstance(outcome, str) and outcome.strip() else None
    if text and not_started is None and text.lower().startswith("not started"):
        not_started = text
    if text and stopped is None and text.lower().startswith(("timed out", "stopped")):
        stopped = text
    registry = Path(path or registry_path())
    try:
        default_root = registry.resolve().parents[2]
    except IndexError:
        default_root = registry.parent
    written = {}

    def fields(row):
        previous = row.get("last_run") if isinstance(row.get("last_run"), dict) else {}
        same_run = previous.get("state") == "running" and (run_id is None or previous.get("run_id") == run_id)
        if previous.get("file_lifecycle"):
            if run_id is not None and previous.get("run_id") != run_id:
                raise RegistryError("completed run identity does not match the managed start")
            if previous.get("ended"):
                written["state"] = previous.get("state")
                return {}  # Idempotent close; preserve the sealed binding and original result.
        last = {k: v for k, v in previous.items() if k != "seal"} if same_run else {}
        if last.get("file_lifecycle"):
            row = dict(row, result_source=last["file_lifecycle"]["result_source"])
            if result_file is not None and Path(result_file).resolve() != Path(row["result_source"]["file"]).resolve():
                raise RegistryError("result override differs from the managed run's bound result file")
        started_at = float(started) if started is not None else last.get("started_at")
        if state in OUTCOMES:
            verdict = (state, str(judged_by or "stated by the caller"), _text_or_none(result_file), None)
        else:
            verdict = read_outcome(row, exit_code, started_at, stopped=stopped, not_started=not_started,
                                   root=root or default_root)
        final, why, file, readings = verdict
        owner = str(last.get("owner") or os.environ.get("UEMA_TEST_OWNER") or row.get("owner") or "not recorded")
        if final == "PASS":
            action = "; ".join(_lines(row.get("on_pass"))) or None
        elif final == "FAIL":
            action = "; ".join(_lines(row.get("on_fail"))) or None
        else:
            action = follow_up_owed_text(final, why, owner)
        last.update(
            state=final, started=None if started_at is None else clock(started_at), started_at=started_at,
            ended=clock(ended_at), ended_at=ended_at, exit_code=exit_code,
            log=_text_or_none(log) if log is not None else last.get("log"),
            outcome=text or "%s: %s" % (final, why), judged_by=why,
            runner=runner if runner is not None else last.get("runner"),
            receipt=_text_or_none(receipt) if receipt is not None else last.get("receipt"),
            result_file=file, next_action=action, next_action_owner=owner,
            follow_up_owed=final in ("INVALID", "STOPPED"))
        last.setdefault("run_id", run_id or new_run_id(row_id, started_at if started_at is not None else ended_at))
        result = dict(read_at=clock(ended_at), verdict=final, judged_by=why, file=file)
        if isinstance(readings, list):
            result["readings"] = [r for r in readings if not reading_problems(r)]
            refused = ["; ".join(reading_problems(r)) for r in readings if reading_problems(r)]
            if refused:
                result["readings_refused"] = refused
        result["run_id"] = last["run_id"]
        # THE RUNNER'S OWN RESULT, sealed with the run (runner_result, runner_file): the digest of the result written
        # here, and the row's result file when it was written during the run
        last["result_digest"] = result_digest(result)
        target = result_file_of(row, root or default_root)
        record = file_record(target) if target is not None else None
        last["result_file_record"] = (record if record is not None and started_at is not None
                                      and record["mtime"] >= float(started_at) - 2.0 else None)
        last["seal"] = seal_of(row_id, last)
        out = dict(status=STATUS_OF_OUTCOME[final], outcome=final, next_action=action, last_run=last, result=result)
        if not empty(row.get("result")) and not (isinstance(row["result"], dict)
                                                 and row["result"].get("run_id") == last["run_id"]):
            out["previous_result"] = row["result"]
        written["state"] = final
        return out
    why_not = _write(row_id, fields, path, runner=_RUNNER)
    if not why_not:
        # Do not write outside the registry from its retrying update callback.
        # Keep a valid experiment result even if file close-out needs reconciliation.
        try:
            from . import file_lifecycle
            data = load(path)
            row = rows_with_id(data, row_id)[0]
            closed = file_lifecycle.close_from_registry(row, data.get("file_lifecycle_policy"))
            if closed:
                _write(row_id, {"file_lifecycle_close": {"state": "recorded", "run_id": row["last_run"]["run_id"]}}, path, runner=_RUNNER)
        except Exception as exc:
            _write(row_id, {"file_lifecycle_close": {"state": "reconcile_required", "reason": str(exc)}}, path, runner=_RUNNER)
    return why_not, (None if why_not else written.get("state"))


def job_finished(row_id, exit_code=None, outcome=None, started=None, ended=None, log=None, runner=None, receipt=None,
                 path=None, **more):
    """A RUNNER FUNCTION: finish_job() for callers that read only whether the write happened. None when written,
    else why not (never raises)."""
    return finish_job(row_id, exit_code=exit_code, outcome=outcome, started=started, ended=ended, log=log,
                      runner=runner, receipt=receipt, path=path, **more)[0]


# ------------------------------------------------------------------------------------------ writes by hand
def _hand(row_id, change, path):
    """Write what `change(row)` returns into the row, as a hand write: never a field of the runner, and never a
    status while the row's test is running. None when written, else why not (never raises)."""
    def fields(row):
        new = change(row)
        if new is not None and "status" in new and row.get("status") == "running":
            raise RegistryError("the row's test is running: its status is the runner's until the run ends "
                                "(test_runner stop ends it; the runner then writes STOPPED)")
        return new
    return _write(row_id, fields, path)


def queue_row(row_id, by, path=None):
    """By hand: status "queued" and no block. `by` names who queued it. None when written, else why not."""
    return _hand(row_id, lambda row: dict(status="queued", blocked=None, queued=dict(by=str(by), at=clock())), path)


def block_row(row_id, reason, waits_on, by, path=None):
    """By hand: the row's block {reason, waits_on, by, at}. The row keeps its status; the Tests page shows it under
    BLOCKED and the runner refuses to start it. None when written, else why not."""
    if empty(reason) or empty(waits_on):
        return "a block needs its reason and what it waits on"
    return _hand(row_id, lambda row: dict(blocked=dict(reason=str(reason), waits_on=str(waits_on), by=str(by),
                                                       at=clock())), path)


def unblock_row(row_id, by, path=None):
    """By hand: the row's block removed. None when written, else why not."""
    return _hand(row_id, lambda row: dict(blocked=None, unblocked=dict(by=str(by), at=clock())), path)


def cancel_row(row_id, reason, by, path=None):
    """By hand: status "cancelled" with {reason, by, at}. None when written, else why not."""
    if empty(reason):
        return "a cancellation needs its reason"
    return _hand(row_id, lambda row: dict(status="cancelled", cancelled=dict(reason=str(reason), by=str(by),
                                                                             at=clock())), path)


def write_follow_up(row_id, text, owner, by, path=None):
    """By hand: the follow-up of a run that ended INVALID or STOPPED (or of a cancelled row): {text, owner, by, at}.
    None when written, else why not."""
    if empty(text) or empty(owner):
        return "a follow-up needs what happens next and its owner"
    now = time.time()
    return _hand(row_id, lambda row: dict(follow_up=dict(text=str(text), owner=str(owner), by=str(by), at=clock(now),
                                                         at_s=now)), path)


def is_blocked(row):
    """The row's block {reason, waits_on, ...} when one is written into it, else None."""
    block = row.get("blocked") if isinstance(row, dict) else None
    return block if isinstance(block, dict) and not empty(block.get("reason")) else None


def section_of(row, required=None, rule_block=None) -> str:
    """The Tests page's section of a row, one of SECTIONS, computed and never typed: RUNNING for status running; for
    status done or failed the section of the outcome in the runner's record (recorded_outcome: PASS: DONE, FAIL:
    FAILED, INVALID, STOPPED), and DONE or FAILED by the status alone for a run whose record holds no outcome (a run
    that ended before outcomes existed, or under a runner of the older kind); STOPPED for a cancelled row; for a
    written or queued row BLOCKED when
    it carries a block, is incomplete (row_problems with `required`), or `rule_block` (the defect register's
    24-hour rule) names a reason, else QUEUED. A row with a missing or unknown status is BLOCKED: the runners refuse
    it."""
    if not isinstance(row, dict):
        return "BLOCKED"
    status = row.get("status")
    outcome = recorded_outcome(row)
    if status == "running":
        return "RUNNING"
    if status == "done":
        return SECTION_OF_OUTCOME.get(outcome, "DONE")
    if status == "failed":
        return "FAILED" if outcome in (None, "PASS") else SECTION_OF_OUTCOME[outcome]
    if status == "cancelled":
        return "STOPPED"
    if status in ("queued", "written"):
        blocked = is_blocked(row) or rule_block or (required is not None and row_problems(row, required))
        return "BLOCKED" if blocked else "QUEUED"
    return "BLOCKED"


# ------------------------------------------------------------------------------------------ the Tests page
def _number_text(value) -> str:
    """A number as written, without an exponent: 37 -> "37", 0.325 -> "0.325", 12000.0 -> "12000"."""
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return str(value)


def _lines(value) -> list:
    """A field as plain lines: text as one line, a list item by item, an object as "key: value" lines."""
    if empty(value):
        return []
    if isinstance(value, str):
        return [value.strip()]
    if isinstance(value, bool):
        return ["yes" if value else "no"]
    if isinstance(value, (int, float)):
        return [_number_text(value)]
    if isinstance(value, dict):
        return ["%s: %s" % (str(k).replace("_", " "), "; ".join(_lines(v)) or "none")
                for k, v in value.items() if not empty(v)]
    out = []
    for item in value:
        if isinstance(item, dict):
            out.append("; ".join(_lines(item)))
        elif not empty(item):
            out.extend(_lines(item))
    return [line for line in out if line]


def _section(label, lines, ordered=False, code=False):
    return dict(label=label, lines=lines, ordered=ordered, code=code) if lines else None


def _duration_text(seconds) -> str:
    """Whole seconds written as hours, minutes and seconds, leaving out the parts that are 0: 7500 -> "2 hours
    5 minutes", 120 -> "2 minutes", 3661 -> "1 hour 1 minute 1 second"."""
    seconds = int(round(seconds))
    parts = [(seconds // 3600, "hour"), (seconds % 3600 // 60, "minute"), (seconds % 60, "second")]
    return " ".join("%d %s%s" % (n, name, "" if n == 1 else "s") for n, name in parts if n)


def time_limit_line(cost) -> str:
    """The Cost section's line for a row's time limit, cost.time_limit_s: the seconds after which the program that runs
    the test ends it. For a pause job it is the job's "timeout_s" in its jobs file, where
    run_in_training_pause.run_jobs ends the job's whole process tree.

    A number above 0 gives "Stops by itself after N seconds", with the same span in hours and minutes from 60 seconds
    up. A missing key, None, or anything that is not a number above 0 gives "No time limit stated"; text the row wrote
    in that place is printed after it, so nothing the row says is dropped."""
    value = cost.get("time_limit_s") if isinstance(cost, dict) else None
    if _number(value) and value > 0:
        whole = float(value).is_integer()
        count = "{:,}".format(int(value)) if whole else _number_text(value)
        line = "Stops by itself after %s %s" % (count, "second" if value == 1 else "seconds")
        return line + (" (%s)" % _duration_text(value) if whole and value >= 60 else "")
    if empty(value):
        return "No time limit stated"
    return "No time limit stated in seconds (the row says: %s)" % "; ".join(_lines(value))


def _cost_lines(cost):
    """The Cost section: training stopped or not, memory, where, then the time limit (time_limit_line: every row gets
    one of its two sentences, a row without a cost too), then the cost's other keys as written."""
    if not isinstance(cost, dict):
        return _lines(cost) + [time_limit_line(cost)]
    out = []
    if isinstance(cost.get("training_stopped"), bool):
        out.append("Training stopped: " + ("yes" if cost["training_stopped"] else "no"))
    if _number(cost.get("memory_gb")):
        out.append("Memory: %s GB" % _number_text(cost["memory_gb"]))
    if not empty(cost.get("where")):
        out.append("Where: %s" % "; ".join(_lines(cost["where"])))
    out.append(time_limit_line(cost))
    out += ["%s: %s" % (str(k).replace("_", " "), "; ".join(_lines(v)))
            for k, v in cost.items()
            if k not in ("training_stopped", "memory_gb", "where", "time_limit_s") and not empty(v)]
    return out


def _result_file(root, source):
    """{path, exists, modified} for the row's result_source file (a path relative to the repository or absolute)."""
    if not isinstance(source, dict) or empty(source.get("file")) or not isinstance(source.get("file"), str):
        return None
    name = source["file"].strip()
    target = Path(name) if Path(name).is_absolute() else Path(root) / name
    try:
        from mimo_bridge.storage_locations import long_path   # paths past 260 characters (Windows long paths off)
    except ImportError:
        long_path = os.fspath
    try:
        stat = os.stat(long_path(target))
    except (OSError, ValueError):
        return dict(path=name, exists=False)
    return dict(path=name, exists=True, modified=clock(stat.st_mtime), modified_at=stat.st_mtime)


def _last_run_text(last):
    if not isinstance(last, dict) or empty(last):
        return None
    parts = []
    if last.get("started"):
        parts.append("started %s" % last["started"])
    parts.append("ended %s" % last["ended"] if last.get("ended") else "no end recorded yet")
    if last.get("exit_code") is not None:
        parts.append("exit code %s" % last["exit_code"])
    text = "Last run: " + ", ".join(parts) + "."
    if last.get("outcome") and last.get("outcome") not in ("running", "exit code %s" % last.get("exit_code")):
        text += " " + str(last["outcome"]).rstrip(".") + "."
    return text


def row_view(row, position, required, counts, root, now=None, rule_block=None):
    """One row of the Tests page: its sentences and expandable sections, why the runners refuse it, and the keys of
    Oct 5, 2026 (added_view). `rule_block` is the defect register's 24-hour reason for this row, when it has one."""
    if not isinstance(row, dict):
        return dict(position=position, title="A row that is not an object", status="", status_label="Unknown",
                    incomplete_text="Incomplete - the runners will refuse it: the row is not an object.", sections=[],
                    section="BLOCKED", section_reason="The row is not an object.", component=None, diagram=None,
                    diagram_missing=True, component_missing=True, outcome=None, run=None, next_action=None,
                    follow_up_owed=False, readings=[], unjudged_result=False, gaps=[], age_s=None, age_text=None,
                    card=dict(question="A row that is not an object", component=None, state="BLOCKED", age=None,
                              direct_target=None, conclusion="No result recorded yet.", next_action=None))
    status = row.get("status") if row.get("status") in STATUSES else ""
    problems = row_problems(row, required)
    view = dict(position=position, id=row.get("id") if isinstance(row.get("id"), str) else None,
                title="; ".join(_lines(row.get("title"))) or "Untitled row",
                question="; ".join(_lines(row.get("question"))), why="; ".join(_lines(row.get("why"))),
                status=status, status_label=STATUS_LABELS.get(status, "Status missing or unknown"),
                written_at="; ".join(_lines(row.get("written_at"))), written=parse_clock(row.get("written_at")))
    if problems:
        view["incomplete"] = [text for _, text in problems]
        view["incomplete_text"] = "Incomplete - the runners will refuse it: " + "; ".join(view["incomplete"]) + "."
    if view["id"] and counts.get(view["id"], 0) > 1:
        view["refused_text"] = ("%d rows share this job name: the runners refuse the job until one row remains."
                                % counts[view["id"]])
    elif status in REFUSED_STATUSES and not problems:
        view["refused_text"] = "Cancelled: the runners refuse this job while its status says cancelled."
    view["last_run_text"] = _last_run_text(row.get("last_run"))
    source = row.get("result_source")
    view["result_file"] = _result_file(root, source)
    if view["result_file"] and view["result_file"].get("exists"):
        view["result_file_text"] = "Result file written %s." % view["result_file"]["modified"]
    pass_lines = _lines(row.get("pass_line"))
    if _lines(row.get("pass_line_basis")):
        pass_lines.append("Why this line: " + "; ".join(_lines(row.get("pass_line_basis"))))
    outcome_lines = (["On pass: " + "; ".join(_lines(row.get("on_pass")))] if _lines(row.get("on_pass")) else []) + \
                    (["On fail: " + "; ".join(_lines(row.get("on_fail")))] if _lines(row.get("on_fail")) else [])
    result, result_is_current = current_result(row)
    result_lines = []
    if not result_is_current:
        result_lines.append("These lines are from an earlier run (run id %s); the row's last run wrote no result."
                            % result.get("run_id"))
    if isinstance(result, dict):
        if result.get("read_at"):
            result_lines.append("Read %s" % result["read_at"])
        result_lines += _lines(result.get("values"))
        for reading in result.get("readings") or []:
            result_lines += reading_lines(reading)
        if not empty(result.get("verdict")):
            result_lines.append("Verdict: %s" % "; ".join(_lines(result["verdict"])))
        result_lines += ["%s: %s" % (str(k).replace("_", " "), "; ".join(_lines(v)))
                         for k, v in result.items()
                         if k not in ("read_at", "values", "verdict", "readings", "run_id") and not empty(v)]
    else:
        result_lines = _lines(result)
    source_lines = []
    if view["result_file"]:
        source_lines.append("Read from %s%s" % (view["result_file"]["path"], "" if view["result_file"].get("exists")
                                                 else " (not written yet)"))
        if isinstance(source, dict) and not empty(source.get("keys")):
            source_lines.append("Keys read: " + ", ".join(_lines(source["keys"])))
    code = _lines(row.get("code"))
    if view["id"]:
        code.append("job name: " + view["id"])
    # the directive's section 6 fields of how a result is judged (rows written from Oct 5, 2026 on carry them; an older
    # row has none and so no such section): its link to gain of function, what "no improvement" reads, how a
    # difference is told from run-to-run noise, and how large it must be to matter
    judged = ["%s: %s" % (LABELS[key][:1].upper() + LABELS[key][1:], "; ".join(_lines(row.get(key))))
              for key in ("gof_link", "baseline", "statistical_line", "material_gof_line") if _lines(row.get(key))]
    sections = [_section("Inputs", _lines(row.get("inputs"))),
                _section("Steps", _lines(row.get("steps")), ordered=True),
                _section("Measures", _lines(row.get("measures"))),
                _section("Pass line and its basis", pass_lines),
                _section("How a result is judged", judged),
                _section("Cost", _cost_lines(row.get("cost"))),
                _section("On pass / on fail", outcome_lines),
                _section("Result", (result_lines or ["No result recorded yet."]) + source_lines),
                _section("Decision", _lines(row.get("decision")) or ["No decision recorded yet."]),
                _section("Written at", _lines(row.get("written_at"))),
                _section("Code files", code, code=True)]
    view["sections"] = [s for s in sections if s]
    view.update(added_view(row, required, now=now, rule_block=rule_block))
    return view


def _age_text(seconds) -> str:
    """A span as the page prints it: whole days and hours from one day up, hours and minutes from one hour up, else
    minutes; "under 1 minute" below 60 seconds."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return "under 1 minute"
    days, hours, minutes = seconds // 86400, seconds % 86400 // 3600, seconds % 3600 // 60
    if days:
        return "%d %s %d %s" % (days, "day" if days == 1 else "days", hours, "hour" if hours == 1 else "hours")
    if hours:
        return "%d %s %d %s" % (hours, "hour" if hours == 1 else "hours", minutes,
                                "minute" if minutes == 1 else "minutes")
    return "%d %s" % (minutes, "minute" if minutes == 1 else "minutes")


UNJUDGED_NUMBERS = ("The full verdict is under Details: its numbers carry no judgement of statistical clarity or of "
                    "size for gain of function.")


def card_conclusion(outcome, verdict, section) -> str:
    """The conclusion on a row's default card (directive section 7A: raw numbers without the two judgements stay
    behind Details). With an outcome in the runner's record: that outcome's meaning. Else, for a verdict typed into
    the row: the verdict itself when it is at most 60 characters and holds no digit; otherwise its label (its words
    before a first colon, when those are at most 60 characters without a digit) followed by UNJUDGED_NUMBERS. A
    sentence is never cut inside: a cut sentence can say the opposite of the whole one."""
    if outcome:
        return OUTCOME_MEANINGS[outcome].capitalize() + "."
    short = str(verdict or "").strip().rstrip(".")
    if short:
        if len(short) <= 60 and not re.search(r"\d", short):
            return "Verdict as written: %s." % short
        label = re.match(r"([^:]{1,60}):", short)       # a verdict written as "pass: ..." keeps its label
        head = label.group(1).strip() if label else ""
        lead = "Verdict as written: %s. " % head if head and not re.search(r"\d", head) else ""
        return lead + UNJUDGED_NUMBERS
    return ("No result recorded yet." if section in ("QUEUED", "BLOCKED", "RUNNING") else
            "No verdict is written in the row.")


def added_view(row, required, now=None, rule_block=None) -> dict:
    """The keys of Oct 5, 2026 a row adds to its view on the Tests page (directive sections 7A and 8), every one
    computed from the row:

    section            one of SECTIONS (section_of)
    section_reason     for BLOCKED and STOPPED: the block's reason, the incomplete fields, the 24-hour rule's reason, or
                       "cancelled by hand"
    component          the row's component id, or None: the page links it to the Architecture tab
    diagram            {input, operation, measurement, decision} or None; diagram_missing is true for a row that is
                       not in section DONE and has none (a page error)
    outcome            PASS, FAIL, INVALID, STOPPED or None: the outcome in the runner's record (recorded_outcome)
    run                the runner's last record in plain fields: state, started, ended, commit, checkpoint, owner,
                       resource_class, device, pid, runner, sealed (true, or false for a record no runner of this
                       module wrote)
    next_action        the text to act on: the follow-up written for the last run when there is one, else the next
                       action the runner wrote at that run's end; None when the row holds neither for its last run
    follow_up_owed     true while a run that ended INVALID or STOPPED has no follow-up written for it
    readings           the structured readings of the last run's result, each with its lines (reading_lines)
    result_from_earlier_run   true when the row's result carries the run id of an earlier run than its last one: the
                       last run (a runner of the older kind) wrote no result, so no reading and no conclusion of the
                       earlier result is shown for it
    unjudged_result    true when the row holds result values that are not structured readings: numbers without the
                       two plain judgements stay behind Details
    gaps               the directive's section 6 fields the row does not fill
    age_s, age_text    since the run started (RUNNING), since it ended (after a run), else since written_at
    card               the default card (directive section 7A): question, component, state, age, direct_target,
                       conclusion (card_conclusion: never a number without its two judgements) and next_action (the
                       row's next_action above, else one sentence for its section: waits to be run, the block's
                       reason, the runner writes it at the run's end, a decision is under Details, or none is written)
    """
    now = time.time() if now is None else float(now)
    problems = row_problems(row, required)
    section = section_of(row, required, rule_block)
    last = row.get("last_run") if isinstance(row.get("last_run"), dict) else {}
    outcome = recorded_outcome(row)
    block = is_blocked(row)
    reason = None
    if section == "BLOCKED":
        reason = ("Blocked: %s. It waits on: %s." % (str(block["reason"]).rstrip("."),
                                                     str(block.get("waits_on") or "not stated").rstrip("."))
                  if block else "Blocked by the 24-hour rule: %s" % rule_block if rule_block else
                  "Its row is incomplete: " + "; ".join(text for _, text in problems) + "." if problems else
                  "Its status is missing or unknown.")
    elif section == "STOPPED" and row.get("status") == "cancelled":
        cancelled = row.get("cancelled") if isinstance(row.get("cancelled"), dict) else {}
        reason = "Cancelled by hand" + (": %s." % str(cancelled["reason"]).rstrip(".")
                                        if not empty(cancelled.get("reason")) else ".")
    follow = current_follow_up(row)
    owed = section in ("INVALID", "STOPPED") and follow is None
    if follow:
        action = "%s (owner: %s)" % (str(follow["text"]).rstrip("."), follow.get("owner") or "not named")
    elif section in ("INVALID", "STOPPED") and row.get("status") == "cancelled":
        action = "No follow-up is written for this cancelled row."
    elif outcome is not None and last.get("next_action") == row.get("next_action"):
        action = "; ".join(_lines(row.get("next_action"))) or None
    else:
        action = None                  # a next action left in the row by an earlier run is not this run's
    whole_result, result_is_current = current_result(row)
    result = whole_result if isinstance(whole_result, dict) and result_is_current else {}
    readings = [dict(r, lines=reading_lines(r), problems=reading_problems(r))
                for r in (result.get("readings") or []) if isinstance(r, dict)]
    unjudged = (result_is_current and not empty(whole_result) and not readings and
                not (isinstance(whole_result, dict) and set(whole_result) <= {"read_at", "verdict", "judged_by",
                                                                              "file", "run_id"}))
    started_at, ended_at = last.get("started_at"), last.get("ended_at")
    if started_at is None:
        started_at = parse_clock(last.get("started"))
    if ended_at is None:
        ended_at = parse_clock(last.get("ended"))
    written = parse_clock(row.get("written_at"))
    since, since_what = ((started_at, "running for") if section == "RUNNING" and started_at is not None else
                         (ended_at, "ended") if ended_at is not None and section in ("DONE", "FAILED", "INVALID",
                                                                                     "STOPPED") else
                         (written, "written"))
    age = None if since is None else max(0.0, now - since)
    age_text = (None if age is None else "running for %s" % _age_text(age) if since_what == "running for" else
                "%s %s ago" % (since_what, _age_text(age)))
    diagram = row.get("diagram") if isinstance(row.get("diagram"), dict) and not diagram_problems(row["diagram"]) \
        else None
    component = row["component"].strip() if isinstance(row.get("component"), str) and row["component"].strip() else None
    verdict = "; ".join(_lines(result.get("verdict"))) if result else ""
    conclusion = card_conclusion(outcome, verdict, section)
    decision = "; ".join(_lines(row.get("decision")))
    card_action = action or (
        "The runner writes the outcome and the next action when the run ends." if section == "RUNNING" else
        reason if section == "BLOCKED" else
        "Waits to be run." if section == "QUEUED" else
        "A decision is written in the row (under Details)." if decision else
        "No next action is written in the row.")
    state_text = (section if outcome is None or SECTION_OF_OUTCOME.get(outcome) == section else
                  "%s (last run: %s)" % (section, outcome))
    run = None
    if last:
        run = dict(state=run_state(row), started=last.get("started"), ended=last.get("ended"),
                   exit_code=last.get("exit_code"), commit=last.get("commit"), commit_dirty=last.get("commit_dirty"),
                   checkpoint=last.get("checkpoint"), owner=last.get("owner"),
                   resource_class=last.get("resource_class"), device=last.get("device"), pid=last.get("pid"),
                   runner=last.get("runner"), run_id=last.get("run_id"), judged_by=last.get("judged_by"),
                   result_file=last.get("result_file"), log=last.get("log"), sealed=sealed(row))
    return dict(section=section, section_reason=reason, component=component, diagram=diagram,
                diagram_missing=section != "DONE" and diagram is None,
                component_missing=section != "DONE" and component is None,
                outcome=outcome, run=run, next_action=action, follow_up_owed=bool(owed), readings=readings,
                unjudged_result=bool(unjudged), result_from_earlier_run=not result_is_current,
                gaps=gaps(row), age_s=age, age_text=age_text,
                card=dict(question="; ".join(_lines(row.get("question"))) or "; ".join(_lines(row.get("title"))),
                          component=component, state=state_text, age=age_text,
                          direct_target="; ".join(_lines(row.get("target_metric"))) or None,
                          conclusion=conclusion, next_action=card_action))


def scheduler_gaps(root, data):
    """The ability scheduler's declared tasks (docs/gain_of_function_v3/ability_rechecks.json) that it refuses for their
    rows: [{id, title, text}]."""
    manifest = Path(root) / "docs/gain_of_function_v3/ability_rechecks.json"
    try:
        tasks = json.loads(manifest.read_text(encoding="utf-8")).get("tasks") or []
    except (OSError, ValueError, AttributeError):
        return []
    out = []
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("id"), str):
            continue
        text = plain_refusal(task["id"], data)
        if text:
            out.append(dict(id=task["id"], title=str(task.get("title") or task["id"]), text=text))
    return out


def scheduler_applies_check(root):
    """True when the running ability scheduler (the pid in its status.json) started after the last change to its code
    file, so it runs the registry check; False when it started before; None when no scheduler process is found."""
    try:
        pid = json.loads((Path(root) / "Saved/mimo_bridge/ability_rechecks/status.json").read_text(encoding="utf-8"))["pid"]
        import psutil
        process = psutil.Process(int(pid))
        if "mimo_bridge.tools.ability_rechecks" not in " ".join(process.cmdline()):
            return None
        started = process.create_time()
        return started >= (Path(root) / "Source/Python/mimo_bridge/tools/ability_rechecks.py").stat().st_mtime
    except Exception:
        return None


def _rule_blocks(data, code_root=None) -> dict:
    """{row id: reason} for the written and queued rows the defect register's 24-hour rule holds back now
    (defect_register.runner_gate); {} when the register cannot be read (the register's own page reports that)."""
    try:
        from . import defect_register
        register = defect_register.load(root=code_root)
    except Exception:
        return {}
    out = {}
    for row in data.get("rows", []):
        if isinstance(row, dict) and isinstance(row.get("id"), str) and row.get("status") in ("queued", "written"):
            try:
                why = defect_register.runner_gate(row, register=register, code_root=code_root, rows=data["rows"])
            except Exception:
                why = None
            if why:
                out[row["id"]] = why
    return out


def _manifest_entries(code_root=None):
    """{component id: its manifest entry}, or None when the component manifest cannot be read."""
    try:
        from . import component_manifest
        return {c["id"]: c for c in component_manifest.load(root=code_root)["components"]
                if isinstance(c, dict) and isinstance(c.get("id"), str)}
    except Exception:
        return None


def architecture_link(component, entries) -> dict:
    """What the Tests page needs to link a row to the Architecture tab (directive section 7A): {component, known
    (true when the id is in the component manifest; None when the manifest cannot be read), name, kind, blocks (the
    Architecture boxes the component is drawn in; empty for a tool and for a part with no box)}."""
    entry = (entries or {}).get(component)
    return dict(component=component, known=None if entries is None else entry is not None,
                name=entry.get("name") if entry else None, kind=entry.get("kind") if entry else None,
                blocks=list(entry.get("blocks") or []) if entry else [])


PROCESS_STATES = {
    "RUNNING": "a live process of the row's run was found",
    "HELD": "a production segment's row while no production trainer process is alive: training is held, so the "
            "segment's process does not exist; the hold's reason is shown",
    "LOST": "the row says running and no process of its run is alive, and no hold explains it: the run ended without "
            "its end being written",
}


def apply_process_states(views, audit_view) -> None:
    """Give every row whose status is running its process state from the audit (test_runner.audit running_rows):
    view["process"] = {state, how, hold, held_text}. A HELD or LOST row is shown under BLOCKED with that state as its
    reason and in its card's state: the RUNNING section holds only rows with a live process, and a row whose process
    does not exist never reads as running. Without an audit (it could not run) the rows stay in RUNNING and the page
    shows the audit's own error."""
    entries = {r.get("id"): r for r in (audit_view or {}).get("running_rows") or [] if isinstance(r, dict)}
    for v in views:
        if v.get("section") != "RUNNING":
            continue
        entry = entries.get(v.get("id"))
        if entry is None:
            continue
        state = entry.get("state") or ("RUNNING" if entry.get("alive") else "LOST")
        v["process"] = dict(state=state, how=entry.get("how"), hold=entry.get("hold"),
                            held_text=entry.get("held_text"))
        if state == "RUNNING":
            continue
        if state == "HELD":
            reason = "HELD: %s" % (entry.get("held_text") or "no production trainer process is alive.")
            reason += " The row's file still says running for run %s." % entry.get("production_run")
        else:
            reason = ("LOST: the row says running and no process of its run is alive; the run ended without its end "
                      "being written (test_runner close writes STOPPED).")
        v["section"] = "BLOCKED"
        v["section_reason"] = reason
        card = v.get("card")
        if isinstance(card, dict):
            card["state"] = "BLOCKED (%s: the row says running and no process of it is alive)" % state
            card["next_action"] = reason


def lacking(view) -> list:
    """What a row that is not DONE lacks of the directive's section 7A ("Each test names the exact Architecture
    component", "Every non-DONE test gets a diagram"): a sublist of ["a component", "a diagram"]; [] for a DONE row.
    Each item is an error on the Tests page."""
    if view.get("section") == "DONE":
        return []
    return [w for w, missing in (("a component", view.get("component_missing")),
                                 ("a diagram", view.get("diagram_missing"))) if missing]


def _names_used(views, code_root=None) -> list:
    """"The names used on this page" (directive section 7A): [{name, kind, definition}] for the sections, outcomes and
    resource classes that at least one row of this view shows, and for every component a row names (its plain name
    and its one sentence from the component manifest)."""
    out = []
    for key in SECTIONS:
        if any(v.get("section") == key for v in views):
            out.append(dict(name=key, kind="section", definition=SECTION_MEANINGS[key]))
    for key in OUTCOMES:
        if any(v.get("outcome") == key for v in views):
            out.append(dict(name=key, kind="outcome", definition=OUTCOME_MEANINGS[key]))
    for key, meaning in RESOURCE_CLASSES.items():
        if any((v.get("run") or {}).get("resource_class") == key for v in views):
            out.append(dict(name=key, kind="resource class", definition=meaning))
    for key in ("HELD", "LOST"):
        if any((v.get("process") or {}).get("state") == key for v in views):
            out.append(dict(name=key, kind="process state", definition=PROCESS_STATES[key]))
    named = sorted({v["component"] for v in views if v.get("component")})
    known = _manifest_entries(code_root) or {}
    for key in named:
        entry = known.get(key)
        out.append(dict(name=key, kind="component",
                        definition=("%s: %s" % (entry.get("name"), entry.get("does")) if entry else
                                    "not an id of the component manifest")))
    return out


def dashboard_view(root=None, now=None, audit=True, code_root=None, rows_in="groups") -> dict:
    """GET /api/tests: every row of the registry, read from the file on this call, grouped by status in GROUP_ORDER
    (unknown last), newest written_at first within a group; each row's refusal reasons; the scheduler's tasks without
    a complete row. Every sentence the page prints is built here.

    Schema 2 (Oct 5, 2026) keeps every key of schema 1 and adds: on each row the keys of added_view(); "sections", the
    seven SECTIONS in the page's order, each {key, label, meaning, open_by_default (true for QUEUED only), count,
    positions (the rows' "position" numbers, newest first)}; "audit", test_runner.audit() (every live test process
    against the running rows, both directions); "errors", the list the page must show as errors [{kind, row, text}];
    "warnings", the development runs the audit lists [{kind, row, text}];
    "names_used" (_names_used); on each row "architecture" (architecture_link: the component's name, kind and
    Architecture boxes, or None for a row without a component). With rows_in="sections" each section carries its
    full rows under "rows" and "groups"
    is left out. `root` holds the registry and the runs; `code_root` (this module's checkout when None) holds the
    component manifest and the defect register."""
    root = Path(root or REPO)
    path = registry_path(root)
    now = time.time() if now is None else float(now)
    out = dict(schema=2, registry=REGISTRY_RELATIVE.as_posix(), registry_url="/files/" + REGISTRY_RELATIVE.as_posix(),
               read_at=now, read_at_text=clock(now))
    stamp = _stamp(path)
    if stamp:
        out["modified_text"] = clock(stamp[0] / 1e9)
    try:
        data = load(path)
    except RegistryError as exc:
        out.update(available=False, error=str(exc),
                   sentence="The test registry could not be read: %s. While it cannot be read, the pause runners and "
                            "the ability scheduler refuse every test." % exc)
        return out
    required = required_fields(data)
    counts = {}
    for row in data["rows"]:
        if isinstance(row, dict) and isinstance(row.get("id"), str):
            counts[row["id"]] = counts.get(row["id"], 0) + 1
    rule = _rule_blocks(data, code_root)
    views = [row_view(row, i, required, counts, root, now=now,
                      rule_block=rule.get(row.get("id")) if isinstance(row, dict) else None)
             for i, row in enumerate(data["rows"])]
    groups = []
    for key in GROUP_ORDER + ("",):
        members = [v for v in views if v["status"] == key]
        if not members:
            continue
        members.sort(key=lambda v: (-(v["written"] if v["written"] is not None else -1e18), -v["position"]))
        groups.append(dict(status=key, label=STATUS_LABELS.get(key, "Status missing or unknown"), rows=members))
    refused = sum(1 for v in views if v.get("incomplete_text") or v.get("refused_text"))
    if views:
        tally = ", ".join("%d %s" % (len(g["rows"]), (g["label"].split(",")[0]).lower()) for g in groups)
        sentence = "%d %s: %s." % (len(views), "row" if len(views) == 1 else "rows", tally)
        if refused:
            sentence += (" %d %s refused: the pause runners and the ability scheduler will not run %s."
                         % (refused, "row is" if refused == 1 else "rows are", "its job" if refused == 1 else "their jobs"))
    else:
        sentence = ("No rows yet. Every test needs its row here before it runs: the pause runners and the ability "
                    "scheduler refuse a job without one.")
    task_gaps = scheduler_gaps(root, data)
    if task_gaps:
        applies = scheduler_applies_check(root)
        out["scheduler_gaps_title"] = "Scheduled ability checks without a complete row"
        out["scheduler_gaps_note"] = (
            "The running ability scheduler holds each of these as 'repair needed' until its row is complete."
            if applies else
            "The running ability scheduler started before the last change to its code file, so it runs older code. "
            "After its restart it holds each of these as 'repair needed' until its row is complete."
            if applies is False else
            "No running ability scheduler was found. Once started, it holds each of these as 'repair needed' until "
            "its row is complete.")
    audit_view = None
    if audit:
        try:
            from . import test_runner
            audit_view = test_runner.audit(path=path, data=data, root=root, now=now, code_root=code_root)
        except Exception as exc:          # the audit that cannot run is itself an error on the page
            audit_view = dict(available=False, ok=False, mismatches=[], running_rows=[],
                              error="%s: %s" % (type(exc).__name__, exc),
                              sentence="The audit of test processes against running rows could not run: %s: %s."
                                       % (type(exc).__name__, exc))
        apply_process_states(views, audit_view)
    sections = []
    for key in SECTIONS:
        members = [v for v in views if v.get("section") == key]
        members.sort(key=lambda v: (-(v["written"] if v.get("written") is not None else -1e18), -v["position"]))
        entry = dict(key=key, label=key.capitalize(), meaning=SECTION_MEANINGS[key],
                     open_by_default=key in OPEN_BY_DEFAULT,
                     count=len(members), positions=[v["position"] for v in members])
        if rows_in == "sections":
            entry["rows"] = members
        sections.append(entry)
    errors = []
    entries = _manifest_entries(code_root)
    for v in views:
        name = v.get("id") or "row %d" % v["position"]
        v["architecture"] = architecture_link(v["component"], entries) if v.get("component") else None
        if v.get("section") != "DONE" and v["architecture"] and v["architecture"]["known"] is False:
            errors.append(dict(kind="component not in the manifest", row=name,
                               text="Test %s names the component %s, which is not an id of the component manifest."
                                    % (name, v["component"])))
        lacks = lacking(v)
        if lacks:
            errors.append(dict(kind="no component or diagram", row=name,
                               text="Test %s is not done and has no %s." % (name, " and no ".join(
                                   w.split(" ", 1)[1] for w in lacks))))
        if v.get("follow_up_owed"):
            how = ("was cancelled" if v.get("status") == "cancelled" else
                   "ended %s" % (v.get("outcome") or v.get("section")))
            errors.append(dict(kind="follow-up owed", row=name,
                               text="Test %s %s and no follow-up with an owner is written in its row (test_runner "
                                    "follow-up %s --text TEXT --owner NAME --by NAME writes it)." % (name, how, name)))
    warnings = []
    if audit_view is not None:
        if not audit_view.get("available", True):
            errors.append(dict(kind="audit unavailable", row=None, text=audit_view["sentence"]))
        for found in audit_view.get("mismatches") or []:
            errors.append(dict(kind=found.get("kind"), row=found.get("row"), text=found.get("text")))
        # a development run is no test of a row: listed as a warning with its owner and purpose (finding F10)
        warnings = [dict(kind=w.get("kind"), row=w.get("row"), text=w.get("text"))
                    for w in audit_view.get("warnings") or []]
    out.update(available=True, sentence=sentence, required=[LABELS.get(f, f.replace("_", " ")) for f in required],
               scheduler_gaps=task_gaps, sections=sections, section_order=list(SECTIONS), audit=audit_view,
               errors=errors, warnings=warnings, names_used=_names_used(views, code_root),
               added_fields=[LABELS.get(f, f) for f in ADDED_FIELDS], added_from=ADDED_FROM,
               note=("The page reads the registry file on every load and every 30 seconds while this tab is open. "
                     "A runner writes each row's status (running, then done or failed), its outcome (PASS, FAIL, "
                     "INVALID or STOPPED), its last run, its result and its next action. A result's readings with "
                     "their two judgements, the decision and a follow-up are written into the row by the person or "
                     "agent who reads the result."))
    if rows_in != "sections":
        out["groups"] = groups
    return out
