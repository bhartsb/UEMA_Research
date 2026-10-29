"""Check the assistant's reply against the operator's rules before the turn ends (the builder Stop hook).

Wired in .agent/settings.json as a Stop hook. It reads the session transcript, takes every piece of
text the assistant wrote since the operator's last message, and checks:

  1. NUMBERS TRACE TO THIS TURN. Every number in the reply must appear in a tool result or in the
     operator's message of this turn (rounding allowed; a share may appear as a count per 100). A number
     worked out in the head - a clock time from felt duration, an order rate, an angle from an average -
     is refused: compute it with a tool so the tool result carries it. (the style guide: "Every factual claim
     ... traces to a file read AT THAT MOMENT".) Integers 0-10 are exempt as labels ("item 5"), except
     next to a time unit ("3 hours").
  2. NO BARE SCORE DECIMALS such as 0.14 (DESIGN.md section 11): counts, real units, or a percentage
     between named ends.
  3. NO BANNED WORDING from the style guide: honest/honestly/truthful, vague size words, meta-filler phrases,
     body/machine metaphors for stop rules.
  4. NO TIME ESTIMATES about future work (the style guide: "NEVER give time estimates").
  5. NO INFORMATION HIDING (the style guide "MECHANISM, NEVER METAPHOR", operator 2026-10-02): software described
     as a person ("asks the model", "the planner decides") and process verbs that hide the operation
     ("nudges", "works out", "moves closer"). The same list checks written documents (worklog_time_check.py).

On problems it blocks the stop ONCE (decision "block" with the list), so the assistant must send a
corrected reply; when the builder reports stop_hook_active (the assistant is already answering this
hook) it lets the stop through and only logs, so it can never loop. Every run is logged to
.agent/hooks/reply_lint.log.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

LOG = Path(__file__).with_name("reply_lint.log")
TAIL_BYTES = 8_000_000

BANNED = [
    (r"\bhonest(ly)?\b", "the word 'honest' (the style guide bans it)"),
    (r"\btruthful\b", "the word 'truthful'"),
    (r"\bno-cheat\b|\bfair-and-square\b", "banned wording"),
    (r"\bbarely\b", "vague size word 'barely' - give the number and its scale"),
    (r"\bslightly\b", "vague size word 'slightly'"),
    (r"\broughly\b", "vague size word 'roughly'"),
    (r"\ba bit\b", "vague size word 'a bit'"),
    (r"\bsignificantly\b", "vague size word 'significantly'"),
    (r"\bsubstantially\b", "vague size word 'substantially'"),
    (r"\bto be clear\b|\bin other words\b|\bnote that\b|\bimportantly\b|\bsimply put\b|\bsaid plainly\b",
     "meta-filler phrase (the style guide bans it)"),
    (r"\bteeth\b|\btooth\b|\btripped\b|\bbites?\b", "body/machine metaphor for a stop rule or watcher"),
]

# MECHANISM, NEVER METAPHOR (the style guide, operator 2026-10-02): software described as a person, and process
# verbs that hide the operation. Shared with worklog_time_check.py, which applies them to written documents.
_AGENTS = (r"(model|models|planner|network|forward model|transformer|heads?|controller|forecaster|encoders?|"
           r"optimi[sz]er|gate|channel|system|algorithm|copy|scorer|evaluator)")
HIDING = [
    (r"\b(asks?|asking|asked|tells?|telling|told) (the |its |a )?(model|planner|network|body|forward model|"
     r"controller|transformer|heads?)\b", "software described as a person ('asks/tells the model')"),
    (r"\b" + _AGENTS + r" (thinks?|knows?|wants?|believes?|tries|try|figures?|realizes?|understands?|sees?|"
     r"decides?|guesses?|imagines?|feels?|hopes?|expects?|looks at|notices?|remembers?|chooses?|wonders?)\b",
     "software described as a person ('the model thinks/knows/decides')"),
    (r"\bnudg(e|es|ed|ing)\b", "process verb that hides the operation ('nudges') - state the computation"),
    (r"\btweak(s|ed|ing)?\b", "process verb that hides the operation ('tweaks')"),
    (r"\b(works?|worked|working) out\b", "process verb that hides the operation ('works out')"),
    (r"\bfigur(e|es|ed|ing) out\b", "process verb that hides the operation ('figures out')"),
    (r"\bmov(e|es|ed|ing) closer\b", "process phrase that hides the operation ('moves closer') - say what "
     "distance, measured how, changed by how much"),
    (r"\bdeal(s|t|ing)? with\b", "process phrase that hides the operation ('deals with')"),
    (r"\bsomehow\b|\bkind of\b|\bsort of\b|\bunder the hood\b|\bbehind the scenes\b|\bmagic(ally)?\b",
     "hiding phrase - state the mechanism"),
    # the style guide "NEVER NAME A CHECK, A CONDITION OR A WAIT WITHOUT ITS FACTS" (operator 2026-10-04)
    (r"\bbriefly\b|\bshortly\b|\bharmless\b|\bfor a (short |little )?(while|time)\b|\bminor (issue|condition|problem)\b",
     "vague word that hides the fact - state the count, the condition's own words, the trigger that ends the "
     "wait, or the measured run time"),
]

# ONE WORD PER THING (the style guide, operator 2026-10-02: "which is it a step or a moment?"). Synonyms of the
# canonical names; code names in backticks are stripped before this check.
SYNONYMS = [
    (r"\bmoments?\b", "synonym 'moment' - the canonical name is 'model step'"),
    (r"\bticks?\b", "synonym 'tick' - the canonical name is 'model step'"),
    (r"(?<!\bin )\borders?\b(?! to\b)", "synonym 'order' - the canonical name is 'command'"),
    (r"\brewards?\b", "synonym 'reward' - the canonical name is 'wellness channel'"),
    # the senses by the registry's names (the style guide; tests/test_sense_names.py OLD_NAMES)
    (r"\bbody[- ]position\b|\bbody[- ]sense\b", "sense synonym - the registry's name is 'proprioception'"),
    (r"\bbalance[- ]sense\b|\bvestibular sense\b|\bbalance\b(?= (?:and|,|is|reads|token|prediction|sense))",
     "sense synonym - the registry's name is 'vestibular'"),
    (r"\bhearing\b", "sense synonym - the registry's name is 'audio'"),
]
HIDING = HIDING + SYNONYMS

# ONE COMPONENT, ONE NAME (the operator's recovery directive of Oct 5, 2026, section 10: "Ambiguous names such as
# 'head repair' are forbidden", "Aliases/synonyms are stored only in the glossary for search; they may not appear in
# status output", "Every chat status states both name and state: HEAD_TILT_REPAIR - BUILT/OFF").
#   INPUT      docs/gain_of_function_v4/component_manifest.json (each component's "id" and its "aliases") and
#              docs/gain_of_function_v4/defect_register.json (each defect's "id"), read on every check.
#   OPERATION  (1) every alias, as whole words in any letter case with spaces or hyphens between its words and an
#              optional plural s, is searched in the text outside code, backticks and quoted mentions whose speaker
#              or naming word stands before them in the sentence ("the operator wrote", "the name"); a quote with
#              neither is the text's own wording and is searched too;
#              (2) every "ID: WORDS IN CAPITALS" (also with a dash) is read, and the capitals must begin with one of
#              REPAIR_STATES; every "ID: word" whose first word, in any letter case, is a state word or a status word
#              (STATUS_WORDS: "fixed", "built", "in training", ...) must begin with one of REPAIR_STATES in capitals;
#              (3) "fixed" said of a component or defect id ("ID is fixed", "ID fixed the ...") is refused unless the
#              defect register holds that id (or a repair of that component) at VERIFIED: only VERIFIED may be called
#              fixed.
#              The dashboard review of Oct 5, 2026, finding F6: "PREDICTOR_REPAIR: fixed." and a quoted alias passed.
#              A page that marks an alias it prints from data with its canonical ids ("head repair [alias of
#              HEAD_TILT_REPAIR or PREDICTOR_REPAIR]") passes the check called with marked_ok (finding F7).
#   OUTPUT     one problem per alias found (naming the canonical ids), per status whose state is not a repair state,
#              and per "fixed" beside an id that is not VERIFIED.
# A MISSING OR MALFORMED FILE GIVES NO NAMES AND NO PROBLEM: this hook runs on every reply of the coordinator, so
# canonical_names() never raises; the manifest's own check (tools/component_manifest.py) reports a broken manifest.
MANIFEST = Path(__file__).resolve().parents[2] / "docs" / "gain_of_function_v4" / "component_manifest.json"
DEFECTS = Path(__file__).resolve().parents[2] / "docs" / "gain_of_function_v4" / "defect_register.json"
REPAIR_STATES = ("FOUND", "CAUSE KNOWN", "BUILT/OFF", "REVIEWED", "QUALIFIED/OFF", "IN TRAINING", "VERIFIED",
                 "FAILED/SUPERSEDED")
# a state word is a whole word: "HEAD_TILT_REPAIR_1 is typed IN TRAINING" after a defect's id and a colon (the defect
# register's problem sentences) names a repair, and its "HEAD" is not a state (found by the pages' acceptance test,
# Oct 5, 2026 5:20 AM PDT)
_CAPS = r"[A-Z]{2,}(?:/[A-Z]{2,})?(?![A-Za-z0-9_])"
# a status word in any letter case right after "ID:" (finding F6): the eight states' words and the words used for a
# status that is not one of them; each makes the text after the colon a status, which then needs a state in capitals
STATUS_WORDS = ("found", "cause known", "built", "built/off", "reviewed", "qualified", "qualified/off", "in training",
                "verified", "failed", "failed/superseded", "superseded", "fixed", "done", "ready", "complete",
                "completed", "working", "broken", "passed", "pass", "fail", "ok", "okay", "stale", "qa stale", "solved",
                "resolved", "repaired", "deployed", "merged", "shipped", "closed", "finished", "healthy", "green",
                "bench passed")
# a quote is a mention, not the text's own wording, when a speaker or a naming word stands before it in its sentence
_MENTION = re.compile(r"\b(said|says|say|wrote|writes|write|asked|asks|ask|called|calls|call|named|names|name|"
                      r"alias|aliases|word|words|term|terms|quote|quoted|quotes|wording|phrase|phrases|spelled|"
                      r"labelled|labeled|reads|read)\b[^.!?\n]{0,60}$", re.I)
_MARK = re.compile(r"\s*\[alias of ([A-Z][A-Z0-9_]*(?:(?:, | or | and )[A-Z][A-Z0-9_]*)*)\]")


def canonical_names(manifest=None, defects=None):
    """(aliases, ids): aliases is [(the alias in lower case with single spaces, [the ids it has named])], longest
    first; ids is the set of component ids and defect ids. ([], set()) for files that are missing, unreadable or not
    in their form. Never raises."""
    aliases, ids = {}, set()
    try:
        data = json.loads(Path(manifest or MANIFEST).read_text(encoding="utf-8-sig"))
        for component in data.get("components", []) if isinstance(data, dict) else []:
            if not isinstance(component, dict) or not isinstance(component.get("id"), str):
                continue
            ids.add(component["id"])
            listed = component.get("aliases")
            for alias in listed if isinstance(listed, list) else []:
                if isinstance(alias, str) and len(alias.strip()) >= 4:
                    aliases.setdefault(" ".join(alias.lower().replace("-", " ").split()), []).append(component["id"])
    except Exception:
        aliases, ids = {}, set()
    try:
        data = json.loads(Path(defects or DEFECTS).read_text(encoding="utf-8-sig"))
        for defect in data.get("defects", []) if isinstance(data, dict) else []:
            if isinstance(defect, dict) and isinstance(defect.get("id"), str):
                ids.add(defect["id"])
    except Exception:
        pass
    return sorted(aliases.items(), key=lambda item: -len(item[0])), ids


def _mentions_removed(text: str) -> str:
    """The text without code fences, backticked code and short quoted MENTIONS: a quote of up to 40 characters whose
    sentence names a speaker or a naming word before it (_MENTION: "the operator wrote", "the alias"). A quote with
    neither is the text's own wording and stays (finding F6: 'Status: the "head repair" is switched on.' passed)."""
    text = re.sub(r"```.*?```", " ", text or "", flags=re.S)
    text = re.sub(r"`[^`\n]*`", " ", text)
    text = re.sub(r"<(code|pre)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)

    def mention(match):
        before = text[max(0, match.start() - 80):match.start()]
        return " " if _MENTION.search(before) else match.group(0)
    return re.sub(r"[\"“‘][^\"”’\n]{1,40}[\"”’]", mention, text)


def alias_problems(text: str, manifest=None, defects=None, marked_ok=False):
    """Problems for every forbidden alias of the component manifest used in `text` (see ONE COMPONENT, ONE NAME). With
    marked_ok, an alias followed by its mark "[alias of ID ...]" naming every id the alias has named passes: a page
    that prints a row's own text marks each alias in it (finding F7)."""
    aliases, _ids = canonical_names(manifest, defects)
    if not aliases:
        return []
    body = _mentions_removed(text)
    out, taken = [], []
    for alias, owners in aliases:
        # not inside a word, and not inside a command-line flag ("--vestibular-encoder-repair" is the code's own name
        # of a switch, not an alias in prose)
        pattern = r"(?<![A-Za-z0-9_\-])" + r"[\s-]+".join(re.escape(word) for word in alias.split()) + \
                  r"s?(?![A-Za-z0-9_])"
        for match in re.finditer(pattern, body, flags=re.I):
            if any(a <= match.start() < b for a, b in taken):
                continue                       # inside a longer alias already reported
            taken.append((match.start(), match.end()))
            if marked_ok:
                mark = _MARK.match(body, match.end())
                if mark and set(owners) <= set(re.split(r", | or | and ", mark.group(1))):
                    continue                   # marked with its canonical ids where it is printed
            out.append("forbidden alias '%s' - write the canonical name %s (directive of Oct 5, section 10: one "
                       "component, one name): '...%s...'"
                       % (alias, " or ".join(sorted(set(owners))),
                          body[max(0, match.start() - 30):match.end() + 30].replace("\n", " ").strip()))
            break
    return out


def verified_ids(defects=None) -> set:
    """The ids the defect register holds at VERIFIED: each defect whose state is VERIFIED, and each repair component an
    entry lists with the state VERIFIED. An empty set when the register cannot be read. Never raises."""
    out = set()
    try:
        data = json.loads(Path(defects or DEFECTS).read_text(encoding="utf-8-sig"))
        for defect in data.get("defects", []) if isinstance(data, dict) else []:
            if not isinstance(defect, dict):
                continue
            if defect.get("state") == "VERIFIED" and isinstance(defect.get("id"), str):
                out.add(defect["id"])
            for repair in defect.get("repairs") or []:
                if isinstance(repair, dict) and repair.get("state") == "VERIFIED" and \
                        isinstance(repair.get("component"), str):
                    out.add(repair["component"])
    except Exception:
        return set()
    return out


def status_problems(text: str, manifest=None, defects=None):
    """Problems for every "ID: STATE" (or "ID - STATE") whose words in capitals do not begin with one of
    REPAIR_STATES; for every "ID: word" whose first words, in any letter case, are a state word or a status word
    (STATUS_WORDS) and do not begin with a repair state in capitals; and for "fixed" after an id in the same sentence
    while the register does not hold that id at VERIFIED (only VERIFIED may be called fixed). For the ids of the
    component manifest and the defect register."""
    _aliases, ids = canonical_names(manifest, defects)
    if not ids:
        return []
    body = _mentions_removed(text)
    out = []
    alternation = "|".join(sorted((re.escape(i) for i in ids), key=len, reverse=True))
    pattern = (r"(?<![A-Za-z0-9_])(" + alternation +
               r")(?![A-Za-z0-9_])\s*(?::|—|–|-)\s*(" + _CAPS + r"(?:[ \t]+" + _CAPS + r")*)")
    reported = set()
    for match in re.finditer(pattern, body):
        stated = " ".join(match.group(2).split())
        reported.add(match.start())
        if not any(stated == s or stated.startswith(s + " ") for s in REPAIR_STATES):
            out.append("status '%s: %s' - the state is not one of %s (directive of Oct 5, section 4)"
                       % (match.group(1), stated, ", ".join(REPAIR_STATES)))
    # a state or status word in any letter case (finding F6: "PREDICTOR_REPAIR: fixed." passed)
    words = r"[A-Za-z]+(?:/[A-Za-z]+)?(?:[ \t]+[A-Za-z]+(?:/[A-Za-z]+)?)?"
    loose = (r"(?<![A-Za-z0-9_])(" + alternation + r")(?![A-Za-z0-9_])\s*(?::|—|–|-)[ \t]*(" + words + r")")
    for match in re.finditer(loose, body):
        if match.start() in reported:
            continue
        said = match.group(2)
        low = " ".join(said.lower().split())
        first = low.split(" ")[0]
        if not (low in STATUS_WORDS or first in STATUS_WORDS):
            continue
        if any(said.startswith(s) for s in REPAIR_STATES):
            continue
        shown = said if low in STATUS_WORDS else said.split()[0]
        out.append("status '%s: %s' - the state is not one of %s in capitals (directive of Oct 5, section 4)"
                   % (match.group(1), shown, ", ".join(REPAIR_STATES)))
    # "fixed" said of an id: only VERIFIED may be called fixed (directive section 4; finding F6). The id is matched in
    # its own capitals; "fixed" is its predicate ("ID is fixed", "ID was now fixed", "ID fixed the ..."), so a technical
    # "held fixed" or "a fixed vector" beside an id is not a status (found by this branch's live check, Oct 5, 2026,
    # 1:12 PM PDT: rows' text "the transformer held fixed")
    verified = verified_ids(defects)
    fixed = (r"(?<![A-Za-z0-9_])(" + alternation + r")(?![A-Za-z0-9_])[ \t]+"
             r"(?:(?:is|are|was|were|has been|have been|had been|got|gets|now|already|finally|been)[ \t]+){0,3}"
             r"(?i:fixed)\b")
    for match in re.finditer(fixed, body):
        if match.group(1) in verified:
            continue
        if any(problem.lower().startswith("status '%s: fixed" % match.group(1).lower()) for problem in out):
            continue                           # already reported as a status (in any letter case)
        out.append("'%s ... fixed' - only VERIFIED may be called fixed, and the defect register does not hold %s at "
                   "VERIFIED (directive of Oct 5, section 4): '...%s...'"
                   % (match.group(1), match.group(1), body[match.start():match.end()].replace("\n", " ")))
    return out


TIME_ESTIMATES = [
    r"\bwill take\b", r"\bshould take\b", r"\btakes about\b", r"\bexpected to take\b", r"\bETA\b",
    r"\bwithin (a few |\d+ )?(minutes?|hours?|days?)\b", r"\bin (a few|about|around) (minutes?|hours?)\b",
    r"\babout an? (hour|minute)\b", r"~\s*\d+\s*(min|minutes|h|hours)\b", r"\bshortly\b",
]

# Each unit ends at a word boundary (the explainer-part2-rule-confirmed-20261015, Oct 15, 2026): the earlier prefix
# match let "0.00001 perturbation" pass as "per", and "sec", "second", "day", "hour" and "cm" also matched the start of
# section, secondary, daylight, hourly and cmd. The same expression as UNIT_AFTER_STRICT in
# Source/Python/mimo_bridge/tools/check_no_bare_decimals.py and S11_UNIT_STRICT in the explainer's viewer.js
# (test_reply_lint.py keeps this one equal to the Python one).
UNIT_AFTER = re.compile(
    r"^\s*(%|°|per ?cent\b|cm\b|mm\b|m\b|km\b|s\b|ms\b|secs?\b|seconds?\b|minutes?\b|min\b|hours?\b|h\b|days?\b|"
    r"GB\b|GiB\b|MB\b|MiB\b|KB\b|kB\b|kg\b|g\b|N\b|W\b|Hz\b|kPa\b|Pa\b|steps?\b|model steps\b|frames?\b|per\b|x\b|"
    r"times\b|fps\b|bytes\b|degrees?\b)", re.I)
# Whole words too (the explainer-commit-block-and-evidence-followup-20261015): "hours?" alone also matched "hourly",
# "days?" "daylight", "seconds?" "secondary" and "minutes?" "minutely".
TIME_UNIT_AFTER = re.compile(r"^\s*(hours?\b|h\b|minutes?\b|min\b|days?\b|seconds?\b|s\b)", re.I)
NUM = re.compile(r"(?<![\w.\-/\\#])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![\w/\\]|\.\d)")
CLOCK = re.compile(r"(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?(?![\d:])")


def _text_of(content):
    if isinstance(content, str):
        return content
    out = []
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                out.append(b.get("text", ""))
            elif b.get("type") == "tool_result":
                out.append(_text_of(b.get("content")))
    return "\n".join(out)


def _is_prompt(rec) -> bool:
    """A message the operator (or the harness on the operator's behalf) sent - not a tool result."""
    if rec.get("type") != "user" or rec.get("isMeta"):
        return False                     # skill instructions and other harness insertions are "meta"
    content = (rec.get("message") or {}).get("content")
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        kinds = {b.get("type") for b in content if isinstance(b, dict)}
        return "tool_result" not in kinds and bool(kinds & {"text", "image"})
    return False


def _is_hook_feedback(rec) -> bool:
    """This hook's own refusal as the harness hands it back (a meta user record "Stop hook feedback: ...")."""
    if rec.get("type") != "user" or not rec.get("isMeta"):
        return False
    content = (rec.get("message") or {}).get("content")
    text = content if isinstance(content, str) else _text_of(content)
    return isinstance(text, str) and text.lstrip().startswith("Stop hook feedback")


def load_turn(path: Path):
    """(assistant_text, corpus_text) since the last operator message; the assistant text only from this hook's
    latest feedback on, when there is one in the turn."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - TAIL_BYTES))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return "", ""
    recs = []
    for ln in lines:
        try:
            recs.append(json.loads(ln))
        except ValueError:
            continue
    start = 0
    for i in range(len(recs) - 1, -1, -1):
        if _is_prompt(recs[i]):
            start = i
            break
    # Text written before this hook's latest feedback was checked then and answered by a corrected reply; checking
    # it again flagged a corrected number in every later reply of the same operator turn (Oct 5, 2026: a bare
    # decimal corrected at 7:4x AM was refused again at 8:4x AM). The corpus still runs from the operator's message.
    reply_from = start
    for i in range(len(recs) - 1, start - 1, -1):
        if _is_hook_feedback(recs[i]):
            reply_from = i
            break
    reply, corpus = [], []
    for k, r in enumerate(recs[start:], start):
        m = r.get("message") or {}
        if r.get("type") == "assistant":
            if k < reply_from:
                continue
            for b in m.get("content") or []:
                if isinstance(b, dict) and b.get("type") == "text":
                    reply.append(b.get("text", ""))
                elif isinstance(b, dict) and b.get("type") == "tool_use":
                    # what the assistant typed into a tool is not evidence, but its OUTPUT is
                    pass
        elif r.get("type") == "user":
            corpus.append(_text_of(m.get("content")))
            tur = r.get("toolUseResult")
            if isinstance(tur, (dict, list, str)):
                corpus.append(json.dumps(tur) if not isinstance(tur, str) else tur)
        elif r.get("type") == "attachment":
            corpus.append(json.dumps(r.get("attachment") or r)[:200_000])
    return "\n\n".join(reply), "\n".join(corpus)


def strip_code(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    # a short quoted fragment is a MENTION (naming a banned word or an example), not a use:
    # found 2026-09-22 when the checker flagged its own description of what it refuses
    text = re.sub(r"[\"“][^\"”\n]{1,30}[\"”]", " ", text)
    text = re.sub(r"`[^`\n]*`", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", lambda m: m.group(0).split("](")[0] + "]", text)  # keep link text only
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", text)                                   # dates
    text = re.sub(r"\bv\d+(\.\d+)+\b", " ", text)                                         # versions
    text = re.sub(r"\b[0-9a-f]{7,40}\b", " ", text)                                       # commit ids
    text = re.sub(r"\S*[\\/]\S*", " ", text)                                              # paths
    text = re.sub(r"\b\w+\.(py|js|json|jsonl|md|html|txt|pt|npz|css|yaml|log)\b", " ", text)
    return text


def corpus_numbers(corpus: str):
    vals = set()
    for m in re.finditer(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?", corpus):
        s = m.group(0).replace(",", "")
        try:
            v = float(s)
        except ValueError:
            continue
        if math.isfinite(v):
            vals.add(abs(v))
    return vals


def _matches(value: float, decimals: int, pool) -> bool:
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    for y in pool:
        if abs(y - value) <= tol or abs(round(y * 100, max(decimals, 0)) - value) <= tol \
                or abs(y * 100 - value) <= 0.5 + 1e-9 and decimals == 0:
            return True
        if value >= 1000 and y > 0 and abs(y - value) / value < 0.005:
            return True
    return False


# Examples and illustrations must be REAL (operator 2026-09-22: "Descriptions must be real", after the
# assistant invented "the hand is told to go left while the elbow is told to turn the other way" and
# "one of its numbers might mean 'bring both hands toward the middle'"). A sentence that offers an
# example or a hypothetical must carry its source: a file path in backticks or a markdown link, or at
# least 4 consecutive words that also appear in a tool result or operator message of this turn.
EXAMPLE_MARKERS = re.compile(
    r"\b(for example|for instance|e\.g\.|imagine|suppose|let'?s say|might mean|could mean|"
    r"picture (this|it)|as if|such as when)\b", re.I)
SOURCE_MARK = re.compile(r"`[^`\n]*[\\/.][^`\n]*`|\]\([^)]+\)")


def _sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _shares_four_words(sentence: str, corpus_low: str) -> bool:
    words = re.findall(r"[a-z0-9']+", sentence.lower())
    for i in range(len(words) - 3):
        if " ".join(words[i:i + 4]) in corpus_low:
            return True
    return False


def unreal_examples(reply: str, corpus: str):
    """Sentences that offer an example or hypothetical without a traceable source."""
    text = re.sub(r"```.*?```", " ", reply, flags=re.S)
    corpus_low = " ".join(re.findall(r"[a-z0-9']+", corpus.lower()))
    out = []
    for s in _sentences(text):
        if EXAMPLE_MARKERS.search(s) and not SOURCE_MARK.search(s) and not _shares_four_words(s, corpus_low):
            out.append("example or description with no source - it may be invented; cite the file it comes "
                       "from or quote what was read, or remove it: '%s'" % s[:160])
    return out


# CLAIM LEDGER (the style guide "CLAIM LEDGER BEFORE A SUBSTANTIVE ANSWER", operator 2026-10-04: "I ask you why something
# failed. you answer. I ask you to adversarily review the answer and you find that you made more errors."). The
# errors of that day sat in sentences without a number: an absolute ("nobody had measured", "nothing to do with")
# or a cause stated from recall. Two checks: (1) a sentence with an absolute or a causal word must be tied to this
# turn's tool results (4 consecutive words shared, or a source mark); (2) a reply of LEDGER_MIN_SENTENCES or more
# needs a claim ledger run this turn (.agent/hooks/claim_check.py) with every entry verified.
CLAIM_MARKERS = re.compile(
    r"\b(nobody|no one|never|nothing|none of|always|every time|at no point|because|caused by|the cause|"
    r"is why|was why|due to|the reason)\b", re.I)
LEDGER_LINE = re.compile(r"CLAIM LEDGER: (\d+) of (\d+) verified")
LEDGER_MIN_SENTENCES = 8


def unsupported_claims(reply: str, corpus: str):
    text = re.sub(r"```.*?```", " ", reply, flags=re.S)
    corpus_low = " ".join(re.findall(r"[a-z0-9']+", corpus.lower()))
    out = []
    for s in _sentences(text):
        if CLAIM_MARKERS.search(strip_code(s)) and not SOURCE_MARK.search(s) \
                and not _shares_four_words(s, corpus_low):
            out.append("absolute or causal claim with no tie to a tool result of this turn - read the file, put "
                       "the claim in the claim ledger with its quote (python .agent/hooks/claim_check.py), and "
                       "word the sentence as the ledger's claim; or remove it: '%s'" % s[:160])
    full = [s for s in _sentences(strip_code(text)) if len(s.split()) >= 4]
    ledgers = LEDGER_LINE.findall(corpus)
    if len(full) >= LEDGER_MIN_SENTENCES and not ledgers:
        out.append("a reply of %d sentences with no claim ledger this turn - list each factual claim with its "
                   "file and a literal quote and run python .agent/hooks/claim_check.py LEDGER.json before "
                   "answering" % len(full))
    elif ledgers and ledgers[-1][0] != ledgers[-1][1]:
        out.append("the newest claim ledger of this turn holds a NOT FOUND entry (%s of %s verified) - correct "
                   "the claim or its quote and run it again" % ledgers[-1])
    return out[:8]


def name_problems(text: str, manifest=None, defects=None, marked_ok=False):
    """The naming problems of a reply or a document: forbidden aliases, statuses with a state that is not a repair
    state, and "fixed" beside an id that is not VERIFIED. marked_ok passes an alias marked with its canonical ids (a
    page's data text, alias_problems). Never raises: a fault in this check must not stop a reply (it is logged by
    main())."""
    try:
        return alias_problems(text, manifest, defects, marked_ok) + status_problems(text, manifest, defects)
    except Exception:
        return []


def check(reply: str, corpus: str):
    problems = list(unreal_examples(reply, corpus)) + unsupported_claims(reply, corpus)
    problems += name_problems(reply)
    body = strip_code(reply)
    low = body.lower()
    for pat, why in BANNED:
        for m in re.finditer(pat, low):
            problems.append("%s: '...%s...'" % (why, body[max(0, m.start() - 30):m.end() + 30].strip()))
            break
    for pat, why in HIDING:
        m = re.search(pat, low)
        if m:
            problems.append("%s - give the input, the computation and the output (the style guide 'MECHANISM, NEVER "
                            "METAPHOR'): '...%s...'" % (why, body[max(0, m.start() - 40):m.end() + 40].strip()))
    for pat in TIME_ESTIMATES:
        m = re.search(pat, body, flags=re.I)
        if m:
            problems.append("time estimate about future work (the style guide bans them): '...%s...'"
                            % body[max(0, m.start() - 30):m.end() + 30].strip())
    pool = corpus_numbers(corpus)
    corpus_clocks = {(int(h), int(mi)) for h, mi, _ in CLOCK.findall(corpus)}
    seen = set()
    for m in CLOCK.finditer(body):
        h, mi = int(m.group(1)), int(m.group(2))
        if (h, mi) not in corpus_clocks and (h, mi) not in seen:
            seen.add((h, mi))
            problems.append("clock time %02d:%02d is not in any tool result this turn - read it with "
                            "`date` or a file timestamp" % (h, mi))
    body_noclock = CLOCK.sub(" ", body)
    for m in NUM.finditer(body_noclock):
        whole, frac = m.group(1), m.group(2)
        raw = m.group(0)
        value = float(whole.replace(",", "") + ("." + frac if frac else ""))
        decimals = len(frac) if frac else 0
        after = body_noclock[m.end():m.end() + 14]
        if frac and value < 1 and not UNIT_AFTER.match(after):
            problems.append("bare score decimal %s - use a count out of 100, a real unit, or a percentage "
                            "between named ends (DESIGN.md section 11)" % raw)
            continue
        if not frac and value <= 10 and not TIME_UNIT_AFTER.match(after):
            continue
        if 2000 <= value <= 2100 and not frac:
            continue
        before = body_noclock[max(0, m.start() - 12):m.start()].lower()
        if value in (100.0, 1000.0) and re.search(r"(out of|of every|of|per|in)\s+$", before):
            continue                                 # the denominator of a count: "32 out of 100"
        key = (raw, after[:6])
        if key in seen:
            continue
        seen.add(key)
        if not _matches(value, decimals, pool):
            ctx = body_noclock[max(0, m.start() - 35):m.end() + 25].replace("\n", " ").strip()
            problems.append("number %s does not appear in any tool result or operator message this turn - "
                            "read it or compute it with a tool first: '...%s...'" % (raw, ctx))
    return problems[:20]


def main():
    try:
        data = json.load(sys.stdin)
    except ValueError:
        data = {}
    path = Path(data.get("transcript_path") or "")
    reply, corpus = load_turn(path) if path.is_file() else ("", "")
    problems = check(reply, corpus) if reply.strip() else []
    active = bool(data.get("stop_hook_active"))
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "session": data.get("session_id"),
                                "active": active, "problems": problems}) + "\n")
    except OSError:
        pass
    if problems and not active:
        reason = ("REPLY CHECK (.agent/hooks/reply_lint.py) found problems in what you just told the "
                  "operator. Send a corrected reply now that fixes every item; do not repeat the flawed "
                  "text. Compute or read every number with a tool in this turn before stating it.\n- "
                  + "\n- ".join(problems))
        print(json.dumps({"decision": "block", "reason": reason}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
