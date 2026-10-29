# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The component manifest: one structured file shared by the Architecture page and the test registry.

The operator's recovery directive of Oct 5, 2026, sections 7C and 9: "Every architecture component has a stable
canonical component ID shared with the test registry", "mapped model code cannot merge without updating the component
manifest", "changing mapped code automatically makes that component's QA stale". The file is
docs/gain_of_function_v4/component_manifest.json:

    {"about": ..., "schema": 1,
     "scope": {"entry": "mimo_bridge.train_mimo", "also": [paths], "outside": [{"path", "why"}]},
     "components": [component, ...]}

A COMPONENT, typed by a person or an agent:
    id          the canonical name, upper case with underscores; used in rows, statuses, pages and commits
    kind        "model part", "world" (the simulated world, which is not part of the model), "repair" or "tool"
    name        one plain-English name
    does        one sentence: what goes in, what is computed, what comes out
    paths       [{"file": path from the repository root, "functions": [names]}]; "functions" is given where one file
                holds several parts (a function, a class, or Class.method); without it the whole file is the part.
                The name "<rest>" (REST) is the text of the file outside every definition the manifest maps: a file
                mapped by functions only has exactly one component that owns it (the dashboard review of Oct 5, 2026,
                finding F13: an edit of a checkpoint migration or of a module-level constant left every entry fresh)
    wiring      for a repair: [needles]: every definition of the model code that holds one of these texts (the
                repair's attribute or switch name) joins the repair's parts, for the check and for QA STALE (finding
                F14: a change of the repair's wiring inside LemLite.step left its QA current)
    ref         a branch name when the code is not on this branch yet (a repair that is built and not merged)
    blocks      the Architecture tab's boxes it corresponds to (keys of mimo_bridge/explainer/schema.py DOC, "node.*")
    flags       the launch flags that switch it
    switch      for a repair: {"flag", "on": [values that mean the repair is on]}
    aliases     other names people have used for it: for search only, refused in operator-facing text
    confirmed   {"at", "by", "commit", "content", "basis"}: when the entry was last checked against its code, by whom,
                on which commit, the hash of the mapped code at that time, and what the check was

GENERATED, never typed (status()):
    last code change    the newest commit that changed the mapped code (for "functions": the text of those functions),
                        with its time in PDT and its short hash; read from git over the mapped paths
    last independent QA the newest entry of docs/gain_of_function_v4/component_qa_log.jsonl for the component, with
                        the reviewed commit; "no independent QA on record" when the log holds none
    QA STALE            true when the mapped code now differs from the mapped code at the reviewed commit
    switch state        each flag's value in the active run's args.json
    open defects        from the defect register, with their age, closing test {row, text} and the note on this
                        component's repair in each
    linked tests        the test registry's rows that name the component and are not done

THE CHECK (check(); `python -m mimo_bridge.tools.component_manifest check`, exit code 1 on any finding) fails when:
    (i)  a model code file is mapped to no component. MODEL CODE is every .py file under Source/Python/lem and
         Source/Python/mimo_bridge that the trainer imports, directly or through other imports (the import statements
         of the files, followed from scope.entry; an import built from text at run time is not seen), plus the files
         of scope.also. A file listed in scope.outside with its reason is exempt.
    (ii) the mapped code of a component differs from the code its entry was confirmed against (confirmed.content, the
         SHA-256 of the mapped files, or of the mapped functions' own text). The finding names the parts that differ
         from confirmed.commit. `confirm` re-stamps an entry after its text was checked against the changed code.
    (iii) the manifest is malformed: a missing field, a mapped file or function that does not exist, two components
         with one id, an alias that is another component's id or name, a block the explainer does not have, a file
         mapped by functions only whose text outside them ("<rest>") no single component owns, a wiring list that is
         not a list of texts.
It is wired into the unit tests (tests/test_component_manifest.py), the coordinator's relaunch script (a refusal before
a relaunch) and GET /api/components (an error on the page).

    python -m mimo_bridge.tools.component_manifest check
    python -m mimo_bridge.tools.component_manifest confirm ID [ID ...] --by NAME --basis "TEXT"
    python -m mimo_bridge.tools.component_manifest status [ID]
    python -m mimo_bridge.tools.component_manifest files
    python -m mimo_bridge.tools.component_manifest record-qa ID --reviewed-commit C --review-ref BRANCH
           --review-record PATH --verdict "TEXT" --by NAME [--reviewed-ref BRANCH]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from . import test_registry as tr

REPO = Path(__file__).resolve().parents[4]
MANIFEST_RELATIVE = tr.MANIFEST_RELATIVE
QA_LOG_RELATIVE = Path("docs/gain_of_function_v4/component_qa_log.jsonl")
SOURCE = "Source/Python"
PACKAGES = ("lem", "mimo_bridge")
KINDS = ("model part", "world", "repair", "tool")
ID = re.compile(r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$")
REQUIRED = ("id", "kind", "name", "does", "paths", "blocks", "flags", "aliases", "confirmed")
NO_QA = "no independent QA on record"
HISTORY_LIMIT = 600                 # commits of one file read for a function's last change
REF_KEEP_S = 15.0                   # how long a branch name's commit is kept before git is asked again
_SHA = re.compile(r"[0-9a-f]{40}")


class ManifestError(Exception):
    """The manifest file cannot be read as the manifest."""


# ------------------------------------------------------------------------------------------ git
def git(root, *arguments, text=True):
    """git's standard output (text, or bytes with text=False), or None when git cannot answer (never raises)."""
    try:
        done = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, timeout=120,
                              stdin=subprocess.DEVNULL,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.decode("utf-8", errors="replace") if text else done.stdout


_REFS, _TIMES, _TEXTS, _NOW = {}, {}, {}, {}


def forget():
    """Drop everything this module keeps in memory (a test that rewrites a repository under one path calls it)."""
    for kept in (_REFS, _TIMES, _TEXTS, _NOW, _OUTLINES, _CLOSURE, _DIRTY):
        kept.clear()


def resolve(root, ref):
    """The full commit hash of `ref` (a branch, a hash, HEAD), or None. A branch name's answer is kept REF_KEEP_S
    seconds; a full hash that exists is kept."""
    if not ref:
        return None
    key = (str(root), str(ref))
    kept = _REFS.get(key)
    if kept and (kept[0] is None or time.time() - kept[0] < REF_KEEP_S):
        return kept[1]
    out = git(root, "rev-parse", "--verify", "--quiet", "%s^{commit}" % ref)
    sha = out.strip() if out and _SHA.fullmatch(out.strip()) else None
    if len(_REFS) > 512:
        _REFS.clear()
    _REFS[key] = (None if sha is not None and str(ref) == sha else time.time(), sha)
    return sha


def short(root, commit):
    """The first 9 characters of a commit's hash."""
    sha = commit if _SHA.fullmatch(str(commit or "")) else resolve(root, commit)
    return sha[:9] if sha else str(commit)[:9]


def commit_time(root, commit):
    sha = resolve(root, commit)
    if sha is None:
        return None
    if (str(root), sha) not in _TIMES:
        out = git(root, "show", "-s", "--format=%ct", sha)
        try:
            _TIMES[(str(root), sha)] = float(out.strip())
        except (AttributeError, ValueError):
            return None
    return _TIMES[(str(root), sha)]


def file_at(root, commit, path):
    """The text of `path` at `commit` with line ends as \\n, or None when the commit does not hold it. A commit's
    file never changes, so the newest 48 answers are kept."""
    sha = resolve(root, commit)
    if sha is None:
        return None
    key = (str(root), sha, path)
    if key not in _TEXTS:
        blob = git(root, "show", "%s:%s" % (sha, path), text=False)
        if len(_TEXTS) >= 48:
            _TEXTS.pop(next(iter(_TEXTS)))
        _TEXTS[key] = None if blob is None else blob.decode("utf-8", errors="replace").replace("\r\n", "\n")
    return _TEXTS[key]


def file_now(root, path):
    """The text of `path` in the working tree with line ends as \\n, or None when it is not there; read again when
    the file's size or modification time changes."""
    target = Path(root) / path
    try:
        stat = target.stat()
    except OSError:
        return None
    key = str(target)
    kept = _NOW.get(key)
    if kept and kept[0] == (stat.st_mtime_ns, stat.st_size):
        return kept[1]
    try:
        text = target.read_bytes().decode("utf-8", errors="replace").replace("\r\n", "\n")
    except OSError:
        return None
    if len(_NOW) > 512:
        _NOW.clear()
    _NOW[key] = ((stat.st_mtime_ns, stat.st_size), text)
    return text


# ------------------------------------------------------------------------------------------ the mapped code's text
_DEF = re.compile(r"^(?P<indent>[ \t]*)(?:async[ \t]+def|def|class)[ \t]+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b")
_TOKEN = re.compile(r"""#[^\n]*|\"\"\"|'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|[()\[\]{}]""")
_OUTLINES = {}


def logical_lines(text):
    """(the text's lines, for each line whether it CONTINUES an earlier one): a line continues when it starts inside
    a bracket that an earlier line opened, inside a triple-quoted string, or after a line ending in a backslash. A
    definition's extent is judged on the other lines only: a closing parenthesis of a signature, or a string's text,
    may sit at any indentation without ending the definition. One pass of a token pattern (comments, strings,
    brackets) over the text."""
    lines = text.split("\n")
    starts, at = [], 0
    for line in lines:
        starts.append(at)
        at += len(line) + 1
    marks = []                              # (offset, bracket depth after it, inside a triple-quoted string after it)
    depth, pos, size = 0, 0, len(text)
    while pos < size:
        match = _TOKEN.search(text, pos)
        if match is None:
            break
        token = match.group(0)
        pos = match.end()
        if token in ('"""', "'''"):
            end = text.find(token, pos)
            while end > 0 and text[end - 1] == "\\" and (end < 2 or text[end - 2] != "\\"):
                end = text.find(token, end + 1)
            end = size if end < 0 else end + 3
            marks.append((match.start(), depth, True))
            marks.append((end, depth, False))
            pos = end
        elif token in "([{":
            depth += 1
            marks.append((match.end(), depth, False))
        elif token in ")]}":
            depth = max(0, depth - 1)
            marks.append((match.end(), depth, False))
    continues = [False] * len(lines)
    index, state_depth, in_string = 0, 0, False
    for i, start in enumerate(starts):
        while index < len(marks) and marks[index][0] <= start:
            state_depth, in_string = marks[index][1], marks[index][2]
            index += 1
        continues[i] = state_depth > 0 or in_string or (i > 0 and lines[i - 1].rstrip().endswith("\\")
                                                        and not lines[i - 1].lstrip().startswith("#"))
    return lines, continues


def outline(text):
    """(lines, {qualified name: [(first line, line after the last)]}) for every definition of `text` at the top level
    ("name") and directly inside a top-level class ("Class.method"). A definition runs from its first decorator line
    to the last line before the next line that is not blank, not a comment, does not continue an earlier line
    (logical_lines) and is indented no deeper than the definition's own `def` or `class` line. A name defined twice
    has both spans. Read by indentation in one pass, without a full parse: hundreds of old versions of a
    660,000-character file are read this way for one function's history. Kept for the newest 24 texts."""
    key = (len(text), hash(text))
    kept = _OUTLINES.get(key)
    if kept is not None:
        return kept
    lines, continues = logical_lines(text)
    count = len(lines)
    indents = []
    for line in lines:
        expanded = line.expandtabs()
        indents.append(len(expanded) - len(expanded.lstrip()))

    def is_code(i):
        return bool(lines[i].strip()) and not continues[i] and not lines[i].lstrip().startswith("#")

    def end_of(i, indent, last):
        end = i + 1
        while end < last and not (is_code(end) and indents[end] <= indent):
            end += 1
        while end > i + 1 and (not lines[end - 1].strip() or (
                lines[end - 1].lstrip().startswith("#") and indents[end - 1] <= indent and not continues[end - 1])):
            end -= 1
        return end

    def start_of(i, indent, first):
        start = i
        while start > first and not continues[start - 1] and lines[start - 1].lstrip().startswith("@") \
                and indents[start - 1] == indent:
            start -= 1
        return start
    spans = {}
    i = 0
    while i < count:
        match = None if continues[i] else _DEF.match(lines[i])
        if match is None or match.group("indent") != "":
            i += 1
            continue
        end = end_of(i, 0, count)
        name = match.group("name")
        spans.setdefault(name, []).append((start_of(i, 0, 0), end))
        if lines[i].lstrip().startswith("class"):
            inner, nearest = [], None
            j = i + 1
            while j < end:
                sub = None if continues[j] else _DEF.match(lines[j])
                if sub is not None and indents[j] > 0 and (nearest is None or indents[j] <= nearest):
                    nearest = indents[j] if nearest is None else min(nearest, indents[j])
                    sub_end = end_of(j, indents[j], end)
                    inner.append((sub.group("name"), indents[j], start_of(j, indents[j], i + 1), sub_end))
                    j = sub_end
                    continue
                j += 1
            for sub_name, indent, a, b in inner:
                if indent == nearest:                   # the class's own definitions, not one nested deeper
                    spans.setdefault(name + "." + sub_name, []).append((a, b))
        i = end
    if len(_OUTLINES) >= 24:
        _OUTLINES.pop(next(iter(_OUTLINES)))
    _OUTLINES[key] = (lines, spans)
    return lines, spans


def blocks_named(text, qualname) -> list:
    """The source text of every definition named `qualname` in `text` (outline): "name" for a function or class at
    the top level, "Class.method" for a definition directly inside a top-level class."""
    lines, spans = outline(text)
    return ["\n".join(lines[a:b]) for a, b in spans.get(qualname, [])]


def part_keys(component) -> list:
    """[(file, function name or None)] for every mapped part of a component, in the manifest's order."""
    out = []
    for path in component.get("paths") or []:
        functions = path.get("functions") or []
        out += [(path["file"], f) for f in functions] if functions else [(path["file"], None)]
    return out


REST = "<rest>"


def rest_text(text, names) -> str:
    """The text of a file outside the definitions `names` (the definitions the manifest maps in it): its module-level
    lines and every definition no component maps."""
    lines, spans = outline(text)
    covered = set()
    for name in names or ():
        for a, b in spans.get(name, []):
            covered.update(range(a, b))
    return "\n".join(line for i, line in enumerate(lines) if i not in covered)


def part_text(text, function, rest_names=None):
    """The text of one mapped part inside a file's text: the whole file, the named definitions joined, or for REST the
    text outside `rest_names`; None when the file is missing or holds no such definition."""
    if text is None:
        return None
    if function is None:
        return text
    if function == REST:
        return rest_text(text, rest_names)
    found = blocks_named(text, function)
    return "\n\n".join(found) if found else None


def mapped_definitions(data) -> dict:
    """{file: set of the definitions any component on this branch maps in it} (REST left out)."""
    out = {}
    for c in (data or {}).get("components", []):
        if isinstance(c, dict) and not c.get("ref"):
            for path in c.get("paths") or []:
                if isinstance(path, dict) and isinstance(path.get("file"), str):
                    out.setdefault(path["file"], set()).update(f for f in path.get("functions") or [] if f != REST)
    return out


_MANIFEST_NOW = {}


def manifest_now(root) -> dict:
    """The manifest of the checkout `root` as its file is now, kept while the file's size and time are unchanged; {}
    when it cannot be read."""
    path = manifest_path(root)
    stamp = tr._stamp(path)
    kept = _MANIFEST_NOW.get(str(path))
    if kept and kept[0] == stamp:
        return kept[1]
    try:
        data = load(root)
    except ManifestError:
        data = {}
    _MANIFEST_NOW[str(path)] = (stamp, data)
    return data


_WIRING = {}
# the trainer's own code, as the training preflight fingerprints it: the import closure without the tools, the tests and
# the explainer's page text (tools/training_preflight.py CODE_SKIP)
WIRING_SKIP = ("Source/Python/mimo_bridge/tools/", "Source/Python/mimo_bridge/tests/",
               "Source/Python/mimo_bridge/explainer/")


def wiring_parts(root, component, commit=None, data=None) -> list:
    """WIRING (the module text; finding F14): [(file, definition)] for every definition of the trainer's own code (the
    import closure of scope.entry without WIRING_SKIP, and the component's own files) that holds one of the component's
    "wiring" texts and is not one of its mapped parts, read in the working tree, or at `commit` (a hash or a branch; a
    component with "ref" and no commit is read at that branch's tip). A class whose methods are outlined counts by its
    methods. [] for a component without wiring."""
    needles = [n for n in component.get("wiring") or [] if isinstance(n, str) and n.strip()]
    if not needles:
        return []
    data = manifest_now(root) if data is None else data
    where = commit or component.get("ref")
    sha = resolve(root, where) if where else None
    if where and sha is None:
        return []
    scope = data.get("scope") if isinstance(data.get("scope"), dict) else {}
    closure = [f for f in import_closure(Path(root), scope.get("entry") or "mimo_bridge.train_mimo")
               if not f.startswith(WIRING_SKIP)]
    files = sorted(set(closure) | {p["file"] for p in component.get("paths") or []
                                   if isinstance(p, dict) and isinstance(p.get("file"), str)})
    files = [f for f in files if f.endswith(".py")]
    key = (str(root), sha or "worktree", tuple(needles), tuple(files)) if sha else None
    if key and key in _WIRING:
        return _WIRING[key]
    if sha:
        args = ["grep", "-l", "-F"] + [x for n in needles for x in ("-e", n)] + [sha, "--"] + files
        out = git(root, *args) or ""
        holding = [line.split(":", 1)[1] for line in out.splitlines() if ":" in line]
    else:
        holding = [f for f in files if any(n in (file_now(root, f) or "") for n in needles)]
    own = set(part_keys(component))
    found = []
    for f in holding:
        text = file_at(root, sha, f) if sha else file_now(root, f)
        if text is None:
            continue
        _lines, spans = outline(text)
        names = [n for n in spans if not ("." not in n and any(k.startswith(n + ".") for k in spans))]
        for name in names:
            body = "\n".join("\n".join(_lines[a:b]) for a, b in spans[name])
            if any(n in body for n in needles) and (f, name) not in own and (f, None) not in own:
                found.append((f, name))
    found.sort()
    if key:
        if len(_WIRING) > 256:
            _WIRING.clear()
        _WIRING[key] = found
    return found


def all_part_keys(root, component, commit=None, data=None) -> list:
    """The mapped parts of a component (part_keys) and its wiring parts at `commit` (wiring_parts)."""
    return part_keys(component) + wiring_parts(root, component, commit, data)


def _digest(texts) -> str:
    h = hashlib.sha256()
    for key, text in texts:
        h.update(("%s::%s\n" % key).encode("utf-8"))
        h.update(b"<missing>" if text is None else text.encode("utf-8"))
        h.update(b"\n--\n")
    return h.hexdigest()


def component_parts(root, component, commit=None, data=None, keys=None) -> list:
    """[((file, function), text or None)] for a component's mapped code and its wiring (all_part_keys, or `keys`): in
    the working tree, or at `commit` (a hash or a branch) when given. A component with "ref" and no `commit` is read
    at that branch's tip. REST is read against the definitions the manifest maps now (`data`, the manifest file's own
    when None)."""
    where = commit or component.get("ref")
    keys = all_part_keys(root, component, commit, data) if keys is None else keys
    rest = mapped_definitions(manifest_now(root) if data is None else data) if any(fn == REST for _f, fn in keys) \
        else {}
    return [((f, fn), part_text(file_at(root, where, f) if where else file_now(root, f), fn, rest.get(f)))
            for f, fn in keys]


def content_hash(root, component, commit=None, data=None) -> str:
    """The SHA-256 of a component's mapped code and wiring (component_parts): what `confirmed.content` stores."""
    return _digest(component_parts(root, component, commit, data))


def changed_parts(root, component, since_commit, data=None) -> list:
    """The parts (mapped and wiring, of the code now or at `since_commit`) whose text now differs from their text at
    `since_commit`, as "file" or "file::function"."""
    keys = list(dict.fromkeys(all_part_keys(root, component, None, data) +
                              all_part_keys(root, component, since_commit, data)))
    now = dict(component_parts(root, component, None, data, keys))
    then = dict(component_parts(root, component, since_commit, data, keys))
    return ["%s::%s" % key if key[1] else key[0] for key in keys if now.get(key) != then.get(key)]


# ------------------------------------------------------------------------------------------ the manifest file
def manifest_path(root=None) -> Path:
    return Path(root or REPO) / MANIFEST_RELATIVE


def load(root=None, path=None) -> dict:
    """The manifest as read from the file now. Raises ManifestError when it cannot be read or holds no components
    list."""
    path = Path(path or manifest_path(root))
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise ManifestError("the component manifest %s does not exist" % path) from None
    except (OSError, ValueError) as exc:
        raise ManifestError("the component manifest %s cannot be read: %s" % (path, exc)) from exc
    if not isinstance(data, dict) or not isinstance(data.get("components"), list):
        raise ManifestError("the component manifest %s holds no 'components' list" % path)
    return data


def dumps(data) -> str:
    """The manifest's own layout: one-space indent, each path object and each short list on one line."""
    def one_line(value):
        return json.dumps(value, ensure_ascii=False)

    def write(value, level):
        pad = " " * (level + 1)
        if isinstance(value, dict):
            if not value:
                return "{}"
            if set(value) <= {"file", "functions"} or set(value) <= {"path", "why"} or set(value) <= {"flag", "on"}:
                return one_line(value)
            items = [pad + json.dumps(k, ensure_ascii=False) + ": " + write(v, level + 1) for k, v in value.items()]
            return "{\n" + ",\n".join(items) + "\n" + " " * level + "}"
        if isinstance(value, list):
            if not value:
                return "[]"
            if all(isinstance(v, str) for v in value) and sum(len(v) for v in value) < 100:
                return one_line(value)
            return "[\n" + ",\n".join(pad + write(v, level + 1) for v in value) + "\n" + " " * level + "]"
        return one_line(value)
    return write(data, 0) + "\n"


def save(data, root=None, path=None):
    path = Path(path or manifest_path(root))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".%d.tmp" % os.getpid())
    with open(tmp, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(dumps(data))
    os.replace(tmp, path)


def components(data) -> dict:
    return {c["id"]: c for c in data.get("components", []) if isinstance(c, dict) and isinstance(c.get("id"), str)}


def aliases(root=None) -> list:
    """[(alias in lower case, [component ids it has named])] for every alias of the manifest, longest first; [] when
    the manifest cannot be read. The reply check and the document check refuse these in operator-facing text."""
    try:
        data = load(root)
    except ManifestError:
        return []
    found = {}
    for c in data["components"]:
        if isinstance(c, dict) and isinstance(c.get("id"), str):
            for alias in c.get("aliases") or []:
                if isinstance(alias, str) and alias.strip():
                    found.setdefault(" ".join(alias.lower().split()), []).append(c["id"])
    return sorted(found.items(), key=lambda item: -len(item[0]))


# ------------------------------------------------------------------------------------------ the model code's files
def _module_file(source, name):
    base = source / Path(*name.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _module_name(source, path):
    parts = list(path.relative_to(source).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imports(source, path) -> set:
    try:
        tree = ast.parse(path.read_bytes().decode("utf-8-sig", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return set()
    name = _module_name(source, path)
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".") if package else []
                parts = parts[:len(parts) - (node.level - 1)]
                base = ".".join(parts + ([base] if base else []))
            out.add(base)
            out.update(base + "." + a.name for a in node.names)
    return out


_CLOSURE = {}


def import_closure(root, entry) -> list:
    """Every .py file under Source/Python/lem and Source/Python/mimo_bridge that module `entry` imports, directly or
    through other imports, as paths from the repository root (the import statements of each file, read with ast; the
    packages' __init__ files on the way are included). Kept until one of the files changes."""
    source = Path(root) / SOURCE
    kept = _CLOSURE.get((str(source), entry))
    if kept:
        try:
            if all(Path(root, f).stat().st_mtime_ns == stamp for f, stamp in kept[0].items()):
                return kept[1]
        except OSError:
            pass
    seen, todo = {}, [entry]
    while todo:
        name = todo.pop()
        path = _module_file(source, name)
        if path is None or path in seen:
            continue
        seen[path] = name
        parts = name.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts))]
        todo += [imp for imp in _imports(source, path) if imp.split(".")[0] in PACKAGES]
    files = sorted(p.relative_to(root).as_posix() for p in seen)
    stamps = {}
    for f in files:
        try:
            stamps[f] = Path(root, f).stat().st_mtime_ns
        except OSError:
            pass
    if len(_CLOSURE) > 8:
        _CLOSURE.clear()
    _CLOSURE[(str(source), entry)] = (stamps, files)
    return files


def model_code_files(root, data) -> list:
    """MODEL CODE (the module text): the import closure of scope.entry plus scope.also, as sorted paths."""
    scope = data.get("scope") if isinstance(data.get("scope"), dict) else {}
    files = set(import_closure(Path(root), scope.get("entry") or "mimo_bridge.train_mimo"))
    files.update(str(p) for p in scope.get("also") or [] if isinstance(p, str))
    return sorted(files)


def mapped_files(data) -> dict:
    """{file: [component ids that map it]} over every component without a "ref" (code on this branch)."""
    out = {}
    for c in data.get("components", []):
        if isinstance(c, dict) and not c.get("ref"):
            for path in c.get("paths") or []:
                if isinstance(path, dict) and isinstance(path.get("file"), str):
                    out.setdefault(path["file"], []).append(c.get("id"))
    return out


# ------------------------------------------------------------------------------------------ the check
def _explainer_blocks():
    """The Architecture tab's box keys (schema.DOC keys that start "node."), or None when the explainer's schema
    cannot be imported here."""
    try:
        from mimo_bridge.explainer import schema
        return {k for k in schema.DOC if k.startswith("node.")}
    except Exception:
        return None


def structure_problems(root, data) -> list:
    """Finding (iii) of the check: plain sentences for what is malformed in the manifest itself."""
    out = []
    seen = {}
    blocks = _explainer_blocks()
    names = {}
    for position, c in enumerate(data.get("components", [])):
        if not isinstance(c, dict):
            out.append("component %d is not an object" % position)
            continue
        cid = c.get("id")
        label = cid if isinstance(cid, str) and cid else "component %d" % position
        for field in REQUIRED:
            if field not in c or (field in ("id", "kind", "name", "does") and not str(c.get(field) or "").strip()):
                out.append("%s: %s is missing" % (label, field))
        if not isinstance(cid, str) or not ID.match(cid or ""):
            out.append("%s: the id is not upper case words joined by underscores" % label)
        elif cid in seen:
            out.append("%s: two components have this id" % cid)
        seen[cid] = True
        if c.get("kind") not in KINDS:
            out.append("%s: kind is not one of %s" % (label, ", ".join(KINDS)))
        if isinstance(c.get("name"), str):
            names.setdefault(c["name"].strip().lower(), []).append(label)
        paths = c.get("paths")
        if not isinstance(paths, list) or (not paths and c.get("kind") != "repair"):
            out.append("%s: no code path is mapped" % label)
        commit = resolve(root, c["ref"]) if c.get("ref") else None
        if c.get("ref") and commit is None:
            out.append("%s: its branch %s is not in this repository" % (label, c.get("ref")))
            continue
        if isinstance(paths, list) and all(isinstance(p, dict) and isinstance(p.get("file"), str) for p in paths):
            where = " on branch %s" % c["ref"] if c.get("ref") else ""
            exists = {}
            for (f, fn), text in component_parts(root, c):
                if text is not None:
                    continue
                if f not in exists:
                    exists[f] = (file_at(root, commit, f) if commit else file_now(root, f)) is not None
                if fn and exists[f]:
                    out.append("%s: %s holds no definition named %s%s" % (label, f, fn, where))
                else:
                    out.append("%s: the mapped file %s does not exist%s" % (label, f, where))
        for block in c.get("blocks") or []:
            if blocks is not None and block not in blocks:
                out.append("%s: the explainer has no box %s" % (label, block))
        if "wiring" in c and not (isinstance(c["wiring"], list) and all(isinstance(n, str) and len(n.strip()) >= 4
                                                                         for n in c["wiring"])):
            out.append("%s: wiring is not a list of texts of 4 characters or more" % label)
        confirmed = c.get("confirmed")
        if not isinstance(confirmed, dict) or not all(str(confirmed.get(k) or "").strip() for k in
                                                      ("at", "by", "commit", "content", "basis")):
            out.append("%s: confirmed does not hold at, by, commit, content and basis" % label)
        if c.get("kind") == "repair" and not (isinstance(c.get("switch"), dict) and c["switch"].get("flag")
                                              and isinstance(c["switch"].get("on"), list)):
            out.append("%s: a repair names its switch {flag, on}" % label)
    out += rest_problems(data)
    ids = {c.get("id") for c in data.get("components", []) if isinstance(c, dict)}
    lowered = {str(i).lower().replace("_", " ") for i in ids}
    for c in data.get("components", []):
        if not isinstance(c, dict):
            continue
        for alias in c.get("aliases") or []:
            low = " ".join(str(alias).lower().split())
            if low in lowered or low in names:
                out.append("%s: its alias '%s' is a canonical name" % (c.get("id"), alias))
    for name, owners in names.items():
        if len(owners) > 1:
            out.append("the plain name '%s' is used by %s" % (name, " and ".join(owners)))
    return out


def rest_problems(data) -> list:
    """Finding (iii) for REST (finding F13): each file that components on this branch map by functions only must have
    exactly one component owning its "<rest>"; a "<rest>" of a file mapped whole is a problem too."""
    whole, by_function, owners = set(), set(), {}
    for c in data.get("components", []):
        if not isinstance(c, dict) or c.get("ref"):
            continue
        for path in c.get("paths") or []:
            if not (isinstance(path, dict) and isinstance(path.get("file"), str)):
                continue
            functions = path.get("functions") or []
            if not functions:
                whole.add(path["file"])
            elif any(f != REST for f in functions):
                by_function.add(path["file"])
            if REST in functions:
                owners.setdefault(path["file"], []).append(c.get("id"))
    out = []
    for f in sorted(by_function - whole):
        mine = owners.get(f, [])
        if len(mine) != 1:
            out.append("the text of %s outside its mapped definitions (%s) is owned by %s" % (
                f, REST, "no component" if not mine else "%d components (%s)" % (len(mine), ", ".join(mine))))
    for f in sorted(set(owners) & whole):
        out.append("%s is mapped whole, so a %s of it (%s) is not needed" % (f, REST, ", ".join(owners[f])))
    return out


def unmapped_files(root, data) -> list:
    """Finding (i): the model code files no component maps and scope.outside does not exempt."""
    scope = data.get("scope") if isinstance(data.get("scope"), dict) else {}
    outside = {o.get("path") for o in scope.get("outside") or [] if isinstance(o, dict)}
    mapped = mapped_files(data)
    return [f for f in model_code_files(root, data) if f not in mapped and f not in outside]


def freshness(root, component, data=None) -> dict:
    """Finding (ii) for one component: {fresh, confirmed_commit, changed (the parts that differ from the confirmed
    commit, when that commit is in the repository), sentence}."""
    confirmed = component.get("confirmed") if isinstance(component.get("confirmed"), dict) else {}
    now = content_hash(root, component, data=data)
    if confirmed.get("content") == now:
        return dict(fresh=True, confirmed_commit=confirmed.get("commit"), changed=[], sentence=None)
    since = resolve(root, confirmed["commit"]) if confirmed.get("commit") else None
    changed = changed_parts(root, component, since, data) if since else []
    where = ", ".join(changed[:6]) + (" and %d more" % (len(changed) - 6) if len(changed) > 6 else "")
    sentence = ("%s: its mapped code changed after its entry was confirmed (%s, on commit %s)%s. Read the changed "
                "code, correct the entry, then run `component_manifest confirm %s`."
                % (component.get("id"), confirmed.get("at") or "at no recorded time", confirmed.get("commit") or
                   "none", ": " + where if where else "", component.get("id")))
    return dict(fresh=False, confirmed_commit=confirmed.get("commit"), changed=changed, sentence=sentence)


def check(root=None, data=None) -> dict:
    """THE CHECK (the module text): {ok, problems [plain sentences], unmapped [files], stale [component ids],
    structure [sentences], sentence}. Never raises: a manifest that cannot be read is one problem."""
    root = Path(root or REPO)
    try:
        data = load(root) if data is None else data
    except ManifestError as exc:
        return dict(ok=False, problems=[str(exc)], unmapped=[], stale=[], structure=[str(exc)],
                    sentence="The component manifest cannot be read: %s." % exc)
    structure = structure_problems(root, data)
    unmapped = unmapped_files(root, data)
    stale, sentences = [], []
    for c in data["components"]:
        if isinstance(c, dict) and isinstance(c.get("confirmed"), dict) and isinstance(c.get("paths"), list):
            fresh = freshness(root, c, data)
            if not fresh["fresh"]:
                stale.append(c.get("id"))
                sentences.append(fresh["sentence"])
    problems = (structure + ["model code file %s is mapped to no component" % f for f in unmapped] + sentences)
    sentence = ("Every model code file is mapped and every entry matches its code (%d components)."
                % len(data["components"]) if not problems else
                "The component manifest is not fresh: %d %s." % (len(problems),
                                                                 "finding" if len(problems) == 1 else "findings"))
    return dict(ok=not problems, problems=problems, unmapped=unmapped, stale=stale, structure=structure,
                sentence=sentence)


def refusal(root=None):
    """One sentence for a caller that must refuse on a manifest that is not fresh (the relaunch script), or None."""
    found = check(root)
    if found["ok"]:
        return None
    return ("the component manifest is not fresh (%d %s): %s%s. Run `python -m mimo_bridge.tools.component_manifest "
            "check` for all of them" % (len(found["problems"]), "finding" if len(found["problems"]) == 1 else
                                        "findings", found["problems"][0],
                                        "" if len(found["problems"]) == 1 else "; and %d more"
                                        % (len(found["problems"]) - 1)))


def confirm(ids, by, basis, root=None) -> list:
    """Stamp the named entries as checked against their code now: confirmed {at (the clock), by, commit (HEAD, or the
    tip of the entry's branch), content (the mapped code's hash now), basis}. Returns the ids stamped. Raises
    ManifestError for an unknown id or an empty `by` or `basis`."""
    root = Path(root or REPO)
    if not str(by or "").strip() or not str(basis or "").strip():
        raise ManifestError("a confirmation names who checked the entry and what the check was")
    data = load(root)
    known = components(data)
    missing = [i for i in ids if i not in known]
    if missing:
        raise ManifestError("no component has id %s" % ", ".join(missing))
    head = resolve(root, "HEAD")
    for cid in ids:
        c = known[cid]
        tip = resolve(root, c["ref"]) if c.get("ref") else head
        c["confirmed"] = dict(at=tr.clock(), by=str(by), commit=short(root, tip) if tip else "no commit",
                              content=content_hash(root, c, data=data), basis=str(basis))
    save(data, root)
    return list(ids)


# ------------------------------------------------------------------------------------------ last code change
def _cache_file(root) -> Path:
    folder = Path(os.environ.get("UEMA_COMPONENT_CACHE") or Path(tempfile.gettempdir()) / "uema_component_manifest")
    return folder / ("history_%s.json" % hashlib.sha256(str(Path(root).resolve()).lower().encode()).hexdigest()[:12])


def read_cache(root) -> dict:
    """The history cache of a repository: {"blobs": {blob hash: {definition: digest of its text}}, "files": {"tip|
    file": {"searched": commits read, "changes": {definition or "<file>": {commit, time, subject} or None}}}}. A
    blob's definitions never change and a tip's history never changes, so nothing in it goes stale; a cache that
    cannot be read is an empty one."""
    try:
        data = json.loads(_cache_file(root).read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("blobs"), dict) and isinstance(data.get("files"), dict):
            return data
    except (OSError, ValueError):
        pass
    return dict(blobs={}, files={})


def write_cache(root, cache):
    try:
        if len(cache.get("files", {})) > 4000:          # old tips' answers: the blobs' digests are what cost time
            cache["files"] = {}
        path = _cache_file(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".%d.tmp" % os.getpid())
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


class _Blobs:
    """Blob texts from one `git cat-file --batch` process (hundreds of old versions of one file)."""

    def __init__(self, root):
        self.root, self.proc = root, None

    def text(self, blob):
        if not blob or set(blob) == {"0"}:
            return None
        if self.proc is None:
            self.proc = subprocess.Popen(["git", "-C", str(self.root), "cat-file", "--batch"], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self.proc.stdin.write((blob + "\n").encode("ascii"))
        self.proc.stdin.flush()
        header = self.proc.stdout.readline().split()
        if len(header) < 3 or header[1] != b"blob":
            return None
        size = int(header[2])
        data = b""
        while len(data) < size:
            chunk = self.proc.stdout.read(size - len(data))
            if not chunk:
                break
            data += chunk
        self.proc.stdout.read(1)
        return data.decode("utf-8", errors="replace").replace("\r\n", "\n")

    def close(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def file_history(root, tip, path) -> list:
    """The commits reachable from `tip` that changed `path`, newest first: [{commit, time, subject, parents (the
    file's blob in each parent), blob (the file's blob after the commit)}]. A merge is listed when its file differs
    from every parent's (git's combined form)."""
    out = git(root, "log", "--format=%x01%H%x09%ct%x09%s", "--raw", "--no-abbrev", "-c", "--no-renames", tip, "--",
              path)
    history = []
    for chunk in (out or "").split("\x01")[1:]:
        lines = chunk.split("\n")
        head = lines[0].split("\t", 2)
        if len(head) < 2:
            continue
        for line in lines[1:]:
            if line.startswith(":"):
                colons = len(line) - len(line.lstrip(":"))
                fields = line[colons:].split("\t")[0].split()
                # <mode> x (colons+1), then <blob> x (colons+1): the parents' blobs, then the result's
                blobs = fields[colons + 1:2 * colons + 2]
                if len(blobs) == colons + 1:
                    history.append(dict(commit=head[0], time=float(head[1]), subject=head[2] if len(head) > 2 else "",
                                        parents=blobs[:-1], blob=blobs[-1]))
                break
    return history


def _rest_key(rest_names) -> str:
    return "%s|%s" % (REST, hashlib.sha256("\n".join(sorted(rest_names or ())).encode("utf-8")).hexdigest()[:12])


def _blob_digests(cache, blobs, blob, rest_names=None) -> dict:
    """{definition: the first 16 hex characters of the SHA-256 of its text} for every definition of a blob's
    outline, from the cache or read once from git; with `rest_names`, also the digest of the text outside them under
    _rest_key(rest_names)."""
    if not blob or set(blob) == {"0"}:
        return {}
    found = cache["blobs"].get(blob)
    rest_key = _rest_key(rest_names) if rest_names is not None else None
    if found is None or (rest_key and rest_key not in found):
        text = blobs.text(blob)
        found = dict(found or {})
        if text is not None:
            lines, spans = outline(text)
            for name, where in spans.items():
                joined = "\n\n".join("\n".join(lines[a:b]) for a, b in where)
                found[name] = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
            if rest_key:
                found[rest_key] = hashlib.sha256(rest_text(text, rest_names).encode("utf-8")).hexdigest()[:16]
        cache["blobs"][blob] = found
    return found


def file_changes(root, tip, path, functions, cache, rest_names=None) -> dict:
    """{function name, or "<file>" for the whole file: {commit, time, subject} or None} for the newest commit
    reachable from `tip` that changed `path` (for "<file>"), or the text of that definition (a commit changes a
    definition when its text after the commit differs from its text in every parent). None when no commit within
    HISTORY_LIMIT did. Also "<searched>": how many commits of the file were read. One walk of the file's history
    serves every definition asked for; answers and blob digests go into `cache` (read_cache)."""
    key = "%s|%s" % (tip, path)
    entry = cache["files"].setdefault(key, dict(searched=None, changes={}))
    rest_key = _rest_key(rest_names) if REST in functions else None
    wanted = ["<file>" if fn is None else rest_key if fn == REST else fn for fn in functions]
    missing = [w for w in wanted if w not in entry["changes"]]
    if missing:
        history = file_history(root, tip, path)[:HISTORY_LIMIT]
        entry["searched"] = len(history)
        if "<file>" in missing:
            entry["changes"]["<file>"] = (dict(commit=history[0]["commit"], time=history[0]["time"],
                                               subject=history[0]["subject"]) if history else None)
        named = set(m for m in missing if m != "<file>")
        if named:
            blobs = _Blobs(root)
            try:
                for item in history:
                    after = _blob_digests(cache, blobs, item["blob"], rest_names if rest_key else None)
                    before = [_blob_digests(cache, blobs, b, rest_names if rest_key else None) for b in item["parents"]]
                    for fn in list(named):
                        if all(after.get(fn) != b.get(fn) for b in before):
                            entry["changes"][fn] = dict(commit=item["commit"], time=item["time"],
                                                        subject=item["subject"])
                            named.discard(fn)
                    if not named:
                        break
            finally:
                blobs.close()
            for fn in named:
                entry["changes"][fn] = None
    out = {w: entry["changes"].get(w) for w in wanted}
    if rest_key:
        out[REST] = out.get(rest_key)
    out["<searched>"] = entry.get("searched")
    return out


_DIRTY = {}


def dirty_files(root, keep_s=5.0) -> set:
    """The paths git reports as changed or new in the working tree against HEAD (`git status --porcelain`), kept
    `keep_s` seconds; an empty set when git cannot answer."""
    key = str(root)
    kept = _DIRTY.get(key)
    if kept and time.time() - kept[0] < keep_s:
        return kept[1]
    out = git(root, "status", "--porcelain", "--untracked-files=all") or ""
    found = set()
    for line in out.splitlines():
        path = line[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path:
            found.add(path)
    _DIRTY[key] = (time.time(), found)
    return found


def last_changes(root, component, cache=None) -> dict:
    """LAST CODE CHANGE of a component, read from git: the newest commit (reachable from HEAD, or from the tip of the
    entry's branch) that changed a mapped file, or for a part with "functions" the text of those functions.

    Returns {commit (short), time (seconds), time_text (PDT, AM/PM), subject, part ("file" or "file::function"),
    uncommitted (true when the working tree's mapped code differs from the tip's; None for an entry on another
    branch), searched}; commit is None when the entry maps no code, or when no commit within HISTORY_LIMIT changed
    the mapped code (searched says which)."""
    root = Path(root)
    tip = resolve(root, component.get("ref") or "HEAD")
    if tip is None:
        return dict(commit=None, time=None, time_text=None, subject=None, part=None, uncommitted=None,
                    searched="the branch %s is not in this repository" % (component.get("ref") or "HEAD"))
    if not part_keys(component):
        return dict(commit=None, time=None, time_text=None, subject=None, part=None, uncommitted=None,
                    searched="the entry maps no code yet")
    own_cache = cache is None
    cache = read_cache(root) if own_cache else cache
    by_file = {}
    keys = all_part_keys(root, component)
    rest = mapped_definitions(manifest_now(root)) if any(fn == REST for _f, fn in keys) else {}
    for f, fn in keys:
        by_file.setdefault(f, []).append(fn)
    best, searched = None, []
    for f, functions in by_file.items():
        found = file_changes(root, tip, f, functions, cache, rest_names=rest.get(f))
        searched.append("%s commits of %s" % (found.get("<searched>"), f))
        for fn in functions:
            hit = found.get("<file>" if fn is None else fn)
            if hit and (best is None or hit["time"] > best["time"]):
                best = dict(hit, part="%s::%s" % (f, fn) if fn else f)
    if own_cache:
        write_cache(root, cache)
    uncommitted = None
    if not component.get("ref"):
        # only the files git reports as changed are read at the tip: a clean file's parts equal the tip's
        dirty = dirty_files(root)
        uncommitted = any(f in dirty and part_text(file_now(root, f), fn, rest.get(f)) !=
                          part_text(file_at(root, tip, f), fn, rest.get(f)) for f, fn in keys)
    if best is None:
        return dict(commit=None, time=None, time_text=None, subject=None, part=None, uncommitted=uncommitted,
                    searched="no commit among %s changed the mapped code" % "; ".join(searched))
    return dict(commit=best["commit"][:9], time=best["time"], time_text=tr.clock(best["time"]),
                subject=best["subject"], part=best["part"], uncommitted=uncommitted, searched="; ".join(searched))


# ------------------------------------------------------------------------------------------ independent QA
def qa_log_path(root=None) -> Path:
    return Path(root or REPO) / QA_LOG_RELATIVE


def qa_entries(root=None) -> list:
    """Every entry of the append-only QA log, oldest first; [] when the file is missing. A line that cannot be read
    is skipped."""
    try:
        lines = qa_log_path(root).read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and isinstance(entry.get("component"), str):
            out.append(entry)
    return out


def record_qa(component_id, reviewed_commit, review_ref, review_record, verdict, by, root=None, reviewed_ref=None,
              reviewer=None, reviewed_at=None) -> dict:
    """Append one independent review to the QA log, after checking that it is on record with its commit: the reviewed
    commit exists in this repository, the review's record file exists at the tip of `review_ref`, and the record's
    text holds the reviewed commit's hash (its first 7 characters or more). Raises ManifestError when one of the
    three does not hold. `reviewed_at` defaults to the commit time of the review record's newest commit."""
    root = Path(root or REPO)
    known = components(load(root))
    if component_id not in known:
        raise ManifestError("no component has id %s" % component_id)
    full = resolve(root, reviewed_commit)
    if full is None:
        raise ManifestError("the reviewed commit %s is not in this repository" % reviewed_commit)
    tip = resolve(root, review_ref)
    if tip is None:
        raise ManifestError("the review's branch %s is not in this repository" % review_ref)
    text = file_at(root, tip, review_record)
    if text is None:
        raise ManifestError("%s is not a file at the tip of %s" % (review_record, review_ref))
    if full[:7] not in text:
        raise ManifestError("the review record %s does not name the reviewed commit %s" % (review_record, full[:9]))
    last = git(root, "log", "-1", "--format=%H%x09%ct", tip, "--", review_record) or ""
    when = float(last.strip().split("\t")[1]) if "\t" in last else commit_time(root, tip)
    entry = dict(component=component_id, reviewed_commit=full[:9], reviewed_ref=reviewed_ref,
                 review_ref=review_ref, review_commit=last.split("\t")[0][:9] if "\t" in last else tip[:9],
                 review_record=review_record,
                 reviewed_at=reviewed_at or tr.clock(when), reviewed_at_s=when,
                 reviewer=reviewer or "an independent reviewer on branch %s" % review_ref, verdict=str(verdict),
                 recorded_at=tr.clock(), recorded_by=str(by))
    path = qa_log_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def last_qa(root, component, entries=None):
    """The newest QA log entry of a component with QA STALE computed, or None (no independent QA on record):
    {reviewed_commit, reviewed_at, verdict, review_record, review_ref, stale, changed (the parts that differ from the
    reviewed commit), stale_text}."""
    entries = qa_entries(root) if entries is None else entries
    mine = [e for e in entries if e.get("component") == component.get("id")]
    if not mine:
        return None
    entry = max(mine, key=lambda e: e.get("reviewed_at_s") or 0)
    reviewed = resolve(root, entry.get("reviewed_commit") or "")
    if reviewed is None:
        changed, stale = [], True
        text = "QA STALE: the reviewed commit %s is not in this repository." % entry.get("reviewed_commit")
    else:
        changed = changed_parts(root, component, reviewed)
        stale = bool(changed)
        text = ("QA STALE: %s changed after the reviewed commit %s." % (", ".join(changed[:4]) + (
            " and %d more" % (len(changed) - 4) if len(changed) > 4 else ""), entry.get("reviewed_commit"))
                if stale else None)
    return dict(reviewed_commit=entry.get("reviewed_commit"), reviewed_at=entry.get("reviewed_at"),
                reviewed_at_s=entry.get("reviewed_at_s"), verdict=entry.get("verdict"),
                review_record=entry.get("review_record"), review_ref=entry.get("review_ref"),
                reviewer=entry.get("reviewer"), stale=stale, changed=changed, stale_text=text)


# ------------------------------------------------------------------------------------------ the switch state
def active_args(data_root):
    """(the active run's name, its args.json as a dict) from Saved/mimo_bridge/training_continuation.json under
    `data_root`; (None, None) when they cannot be read."""
    try:
        saved = Path(data_root) / "Saved/mimo_bridge"
        run = Path(json.loads((saved / "training_continuation.json").read_text(encoding="utf-8"))["active_run"])
        args = json.loads((run / "args.json").read_text(encoding="utf-8"))
        return run.name, args if isinstance(args, dict) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None, None


def switch_state(component, run_name, args) -> dict:
    """The deployed switch state of a component, from the active run's launch settings: {run, flags [{flag, value,
    text}], repair_on (true, false or None for a component that is not a repair), sentence}."""
    flags = []
    for flag in component.get("flags") or []:
        key = flag.lstrip("-").replace("-", "_")
        if args is None:
            flags.append(dict(flag=flag, value=None, text="the active run's launch settings could not be read"))
        elif key not in args:
            flags.append(dict(flag=flag, value=None, present=False,
                              text="not present at the launch of %s (its code is not in that run, or its default "
                                   "applies)" % run_name))
        else:
            value = args[key]
            flags.append(dict(flag=flag, value=value, present=True,
                              text="%s in %s" % ("on" if value is True else "off" if value is False else
                                                 "not set" if value is None else value, run_name)))
    repair_on = None
    switch = component.get("switch") if isinstance(component.get("switch"), dict) else None
    if switch and args is not None:
        key = str(switch.get("flag") or "").lstrip("-").replace("-", "_")
        repair_on = key in args and str(args[key]) in [str(v) for v in switch.get("on") or []]
    sentence = None
    if switch:
        sentence = ("The launch settings of the active run could not be read." if args is None else
                    "%s %s is %s in the active run %s." % (
                        switch.get("flag"), "(%s)" % " or ".join(str(v) for v in switch.get("on") or []),
                        "ON" if repair_on else "OFF", run_name))
    return dict(run=run_name, flags=flags, repair_on=repair_on, sentence=sentence)


# ------------------------------------------------------------------------------------------ the page's view
def closing_test_of(defect) -> dict:
    """What ends a defect, as the defect register types it: {row (a test registry id, or None while the row is not
    written), text (what must be shown, or None)}. The explainer's list of the parts being worked on prints it beside
    each repair's state (the decided design of the Architecture tab, Oct 7, 2026, D-E2). A register entry without the
    field gives both as None."""
    closing = defect.get("closing_test") if isinstance(defect.get("closing_test"), dict) else {}
    row, text = closing.get("row"), closing.get("text")
    return dict(row=row if isinstance(row, str) and row.strip() else None,
                text=text if isinstance(text, str) and text.strip() else None)


def status(root=None, data_root=None, only=None, with_history=True) -> dict:
    """GET /api/components: the manifest with every generated field (the module text), the check's result, and
    by_block {Architecture box: [component ids]}. `root` holds the manifest, the QA log, the defect register and the
    code's git history; `data_root` holds the test registry and Saved/ (the same checkout when None). `only` limits
    the components to one id. Never raises: what cannot be read is said in the view."""
    root = Path(root or REPO)
    data_root = Path(data_root or root)
    out = dict(schema=1, manifest=MANIFEST_RELATIVE.as_posix(), read_at_text=tr.clock(), no_qa_text=NO_QA)
    try:
        data = load(root)
    except ManifestError as exc:
        out.update(available=False, error=str(exc), components=[], by_block={},
                   check=dict(ok=False, problems=[str(exc)], sentence="The component manifest cannot be read."))
        return out
    found = check(root, data)
    run_name, args = active_args(data_root)
    entries = qa_entries(root)
    try:
        from . import defect_register
        defects = defect_register.view(root=root, data_root=data_root).get("defects", [])
    except Exception as exc:
        defects, out["defects_error"] = [], "%s: %s" % (type(exc).__name__, exc)
    tests = {}
    try:
        registry = tr.load(tr.registry_path(data_root))
        required = tr.required_fields(registry)
        for row in registry["rows"]:
            if isinstance(row, dict) and isinstance(row.get("component"), str):
                section = tr.section_of(row, required)
                if section != "DONE":
                    tests.setdefault(row["component"], []).append(dict(
                        id=row.get("id"), title="; ".join(tr._lines(row.get("title"))), section=section,
                        outcome=tr.recorded_outcome(row)))
    except tr.RegistryError as exc:
        out["tests_error"] = str(exc)
    cache = read_cache(root) if with_history else None
    views, by_block = [], {}
    for c in data["components"]:
        if not isinstance(c, dict) or (only and c.get("id") != only):
            continue
        fresh = freshness(root, c) if isinstance(c.get("confirmed"), dict) else dict(fresh=False, changed=[],
                                                                                    sentence="never confirmed")
        qa = last_qa(root, c, entries)
        change = last_changes(root, c, cache) if with_history else None
        mine = [d for d in defects if d.get("component") == c.get("id") or c.get("id") in [
            r.get("component") for r in d.get("repairs") or [] if isinstance(r, dict)]]
        view = dict(
            id=c.get("id"), kind=c.get("kind"), name=c.get("name"), does=c.get("does"), paths=c.get("paths"),
            ref=c.get("ref"), blocks=c.get("blocks") or [], flags=c.get("flags") or [],
            aliases=c.get("aliases") or [], confirmed=c.get("confirmed"),
            entry_fresh=fresh["fresh"], entry_stale_text=fresh["sentence"], entry_changed=fresh["changed"],
            last_code_change=change,
            last_code_change_text=(None if change is None else
                                   "%s, commit %s" % (change["time_text"], change["commit"]) if change.get("commit")
                                   else "none: %s" % change.get("searched")),
            last_qa=qa, last_qa_text=(NO_QA if qa is None else "%s, reviewed commit %s"
                                      % (qa["reviewed_at"], qa["reviewed_commit"])),
            qa_stale=bool(qa and qa["stale"]), qa_stale_text=qa["stale_text"] if qa else None,
            switch=switch_state(c, run_name, args),
            open_defects=[dict(id=d.get("id"), title=d.get("title"), severity=d.get("severity"), state=d.get("state"),
                               repair_state=next((r.get("state") for r in d.get("repairs") or []
                                                  if isinstance(r, dict) and r.get("component") == c.get("id")), None),
                               # the register's note on this component's repair in that defect (what the state rests
                               # on: "Purpose not met.", the branch, the review), printed beside its state by the
                               # explainer's list of the parts being worked on
                               repair_note=next((r.get("note") for r in d.get("repairs") or []
                                                 if isinstance(r, dict) and r.get("component") == c.get("id")
                                                 and isinstance(r.get("note"), str)), None),
                               age_text=d.get("age_text"), idle_text=d.get("idle_text"),
                               closing_test=closing_test_of(d)) for d in mine
                          if d.get("open")],
            linked_tests=tests.get(c.get("id"), []))
        views.append(view)
        for block in view["blocks"]:
            by_block.setdefault(block, []).append(view["id"])
    if with_history and cache is not None:
        write_cache(root, cache)
    out.update(available=True, components=views, by_block=by_block, check=found, active_run=run_name,
               qa_log=QA_LOG_RELATIVE.as_posix(), kinds=list(KINDS),
               note=("Last code change, last independent QA, QA STALE, the switch state, the open defects and the "
                     "linked tests are computed on every request from git, the QA log, the active run's launch "
                     "settings, the defect register and the test registry. Name, sentence, code paths, boxes, flags "
                     "and aliases are the manifest's own text, confirmed against the code at the time each entry "
                     "states."))
    return out


# ------------------------------------------------------------------------------------------ the command line
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="component_manifest", description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["check", "confirm", "status", "record-qa", "files"])
    parser.add_argument("ids", nargs="*")
    parser.add_argument("--root")
    parser.add_argument("--data-root")
    parser.add_argument("--by")
    parser.add_argument("--basis")
    parser.add_argument("--reviewed-commit")
    parser.add_argument("--reviewed-ref")
    parser.add_argument("--review-ref")
    parser.add_argument("--review-record")
    parser.add_argument("--verdict")
    a = parser.parse_args(argv)
    root = Path(a.root) if a.root else REPO
    try:
        if a.action == "check":
            found = check(root)
            print(found["sentence"])
            for problem in found["problems"]:
                print("-", problem)
            return 0 if found["ok"] else 1
        if a.action == "confirm":
            done = confirm(a.ids, a.by, a.basis, root)
            print("confirmed at %s: %s" % (tr.clock(), ", ".join(done)))
            return 0
        if a.action == "files":
            data = load(root)
            mapped = mapped_files(data)
            for f in model_code_files(root, data):
                print("%s\t%s" % (f, ", ".join(mapped.get(f, [])) or "UNMAPPED"))
            return 0
        if a.action == "record-qa":
            entry = record_qa(a.ids[0], a.reviewed_commit, a.review_ref, a.review_record, a.verdict, a.by, root,
                              reviewed_ref=a.reviewed_ref)
            print(json.dumps(entry, indent=1, ensure_ascii=False))
            return 0
        view = status(root, a.data_root, only=a.ids[0] if a.ids else None)
        print(json.dumps(view, indent=1, ensure_ascii=False, default=str))
        return 0
    except ManifestError as exc:
        print("refused: %s" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
