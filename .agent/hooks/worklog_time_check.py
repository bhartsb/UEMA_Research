"""Checks on files as they are written (the builder PostToolUse hook on Write|Edit).

Wired in .agent/settings.json on Write|Edit. Two checks:

1. WORKLOG TIMES. When the edited file is a WORKLOG.md, every line of the new text that starts "- HH:MM" is
   compared with the machine's clock; a time more than 2 minutes ahead (and less than 12 hours ahead, so a log
   that crossed midnight is not refused) means the time was guessed from felt duration instead of read from
   `date` or a file timestamp - it happened three times on 2026-09-22.
2. NO INFORMATION HIDING IN DOCUMENTS (the style guide "MECHANISM, NEVER METAPHOR", operator 2026-10-02: "Not here, not
   in a document you create!"). For every .md or .html file, and the explainer's text sources
   (mimo_bridge/explainer/schema.py, walkthrough.py, viewer.js), the new text is checked against reply_lint.HIDING:
   software described as a person ("asks the model", "the planner decides") and process verbs that hide the
   operation ("nudges", "works out", "moves closer"). Text inside code fences and backticks is not checked.

Either check returns decision "block" with the offending lines, so the text is corrected.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ENTRY = re.compile(r"^\s*-\s*([01]?\d|2[0-3]):([0-5]\d)\b", re.M)
DOC_SUFFIXES = (".md", ".html")
EXPLAINER_TEXT = ("mimo_bridge/explainer/schema.py", "mimo_bridge/explainer/walkthrough.py",
                  "mimo_bridge/explainer/viewer.js",
                  # the Tests page's rows (the style guide "NO TEST RUNS WITHOUT ITS ROW")
                  "docs/gain_of_function_v4/test_registry.json")


def future_entries(text: str, now_min: int):
    bad = []
    for m in ENTRY.finditer(text or ""):
        t = int(m.group(1)) * 60 + int(m.group(2))
        ahead = t - now_min
        if 2 < ahead < 12 * 60:
            bad.append("%s:%s (the clock reads %02d:%02d)" % (m.group(1), m.group(2), now_min // 60, now_min % 60))
    return bad


def hiding_lines(text: str):
    """Lines of new document text that describe software as a person or hide an operation."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from reply_lint import HIDING
    text = re.sub(r"```.*?```", " ", text or "", flags=re.S)
    text = re.sub(r"`[^`\n]*`", " ", text)
    text = re.sub(r"<(code|pre)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
    out = []
    for line in text.splitlines():
        low = re.sub(r"<[^>]+>", " ", line).lower()
        for pat, why in HIDING:
            m = re.search(pat, low)
            if m:
                out.append("%s: '...%s...'" % (why, low[max(0, m.start() - 40):m.end() + 40].strip()))
                break
    return out[:15]


def name_lines(text: str):
    """The naming problems of new document text (reply_lint.name_problems: a forbidden alias of the component
    manifest, a status whose state is not a repair state). [] when reply_lint cannot be loaded: this hook must keep
    accepting documents then."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from reply_lint import name_problems
        return name_problems(text or "")[:15]
    except Exception:
        return []


def main():
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return 0
    inp = data.get("tool_input") or {}
    path = str(inp.get("file_path") or "").replace("\\", "/")
    text = inp.get("content") if "content" in inp else inp.get("new_string")
    reasons = []
    if path.endswith("WORKLOG.md"):
        lt = time.localtime()
        bad = future_entries(text or "", lt.tm_hour * 60 + lt.tm_min)
        if bad:
            reasons.append("WORKLOG time check: these entries are stamped later than the clock, so they were "
                           "guessed, not read. Correct them from `date` or file timestamps now: " + "; ".join(bad))
    if path.lower().endswith(DOC_SUFFIXES) or any(path.endswith(p) for p in EXPLAINER_TEXT):
        bad = hiding_lines(text or "")
        if bad:
            reasons.append("DOCUMENT CHECK (the style guide 'MECHANISM, NEVER METAPHOR'): this text hides the mechanism. "
                           "Rewrite each line to state the input, the computation and the output:\n- "
                           + "\n- ".join(bad))
        named = name_lines(text or "")
        if named:
            reasons.append("NAME CHECK (the recovery directive of Oct 5, 2026, sections 4 and 10: one component, one "
                           "name; a status is NAME: STATE with one of the eight repair states):\n- "
                           + "\n- ".join(named))
    if reasons:
        print(json.dumps({"decision": "block", "reason": "\n\n".join(reasons)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
