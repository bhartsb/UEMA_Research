"""Claim ledger: every claim of a reply tied to a literal quote in a named file, checked before the reply is sent.

    python .agent/hooks/claim_check.py LEDGER.json

LEDGER.json is a list of entries {"claim": "...", "file": "path", "quote": "text that is in that file"}.
For each entry the file is read now and the quote must occur in it (case and runs of white space ignored; in
.json and .jsonl files the JSON string escapes are undone first; in .html files the tags are removed first).
A quote must hold at least 3 words and 12 characters. Prints one line per entry and one closing line:

    VERIFIED | <claim> | <file>
    NOT FOUND | <claim> | <file> | quote: <start of the quote>
    CLAIM LEDGER: k of n verified

The reply check (.agent/hooks/reply_lint.py) reads these lines from the turn's tool results: a reply of 8 or more
sentences without a ledger run this turn is refused, and so is one whose newest ledger holds a NOT FOUND.

WHAT THIS DOES NOT DO. It tests that the quote is in the file. It cannot tell whether the quote supports the
claim, and it cannot catch a field whose meaning was misread. Those two need a reader other than the writer.
(the style guide "CLAIM LEDGER BEFORE A SUBSTANTIVE ANSWER", operator 2026-10-04.)
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAIL_BYTES = 30_000_000          # a larger file (the session transcript) is read from its end only


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def file_text(path: Path) -> str:
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - TAIL_BYTES))
        text = f.read().decode("utf-8", errors="replace")
    forms = [text]
    low = path.suffix.lower()
    if low in (".json", ".jsonl"):
        forms.append(text.replace('\\"', '"').replace("\\n", " ").replace("\\\\", "\\"))
    if low in (".html", ".htm"):
        forms.append(html.unescape(re.sub(r"<[^>]+>", " ", text)))
    return " \n ".join(norm(t) for t in forms)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print(__doc__)
        return 2
    entries = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    ok = 0
    cache = {}
    for e in entries:
        claim, quote = str(e.get("claim", "")).strip(), str(e.get("quote", "")).strip()
        p = Path(str(e.get("file", "")))
        p = p if p.is_absolute() else ROOT / p
        why = ""
        if len(quote) < 12 or len(quote.split()) < 3:
            why = "the quote is under 3 words or 12 characters"
        elif not p.is_file():
            why = "no such file"
        else:
            if p not in cache:
                cache[p] = file_text(p)
            if norm(quote) not in cache[p]:
                why = "quote: " + quote[:80]
        if why:
            print("NOT FOUND | %s | %s | %s" % (claim, e.get("file"), why))
        else:
            ok += 1
            print("VERIFIED | %s | %s" % (claim, e.get("file")))
    print("CLAIM LEDGER: %d of %d verified" % (ok, len(entries)))
    return 0 if ok == len(entries) else 1


if __name__ == "__main__":
    sys.exit(main())
