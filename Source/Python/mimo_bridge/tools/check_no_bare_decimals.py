# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Find bare score decimals in text a person reads (DESIGN.md section 11).

A bare score decimal is a number below 1 with a fractional part and no unit after it ("0.14",
"0.0047"), or any number in scientific notation ("7.3e-06"). Section 11 allows only counts ("32 out
of 100 orders"), real units ("0.8 cm", "0.25 s") and percentages between two named ends. A unit must
be a whole word (UNIT_AFTER, since Oct 15, 2026): "0.25 s later" and "0.5 per cent" pass, "0.00001
perturbation" does not. The unit list is the same one the chat reply check uses,
so a page and a chat reply are held to one rule.

    python -m mimo_bridge.tools.check_no_bare_decimals FILE_OR_URL [...]

prints every hit with its context and exits 1 when any is found. Page builders and tests call
`bare_decimals(text)` on the text they render.

A number of 1 or more escapes that rule, so three more rules (added 2026-09-23 by fix-scanner, after
the verifiers found "error ratio 1.003", "ungraded-channel drift x1.0" and "inputs ranged from -32.9
to -7.1" on pages that passed every test) are in `unanchored_numbers(text)`:
  score       a number with a fractional part, of any size, a few words after a score word (ratio,
              score, cosine, correlation, entropy, loss, margin, share, drift, advantage, gain,
              spread, error) with no unit after it or after the range it opens: "error ratio
              1.003", "the share reads 1.0". A number that defines a scale's end ("1.0 = all your
              weight on one spot") and a count ("63.6 in 100") pass.
  multiplier  "x1.0", "1.3x" or "1.3 times" with nothing after it naming what it multiplies:
              "drift x1.0" fails, "4.2 times its first value" and "47x next-state error" pass, and
              so does one whose sentence already named what is multiplied. A whole number after "x"
              is a repeat count ("feed back, ×3") unless a score word stands just before it.
  unit input  a raw input value of a hidden unit ("their inputs ranged from -32.9 to -7.1", "its
              input rises above -3"): a number a reader cannot picture, whatever its size.
`readout_problems(text)` returns the bare decimals and these together; the command line reports both.
Page builders use `bare_decimals` to decide which archived sentences to withhold (plain_numbers.clean),
so a change of its unit list changes what pages print: the switch to whole-word units of Oct 15, 2026
newly flagged 0 sentences on the explainer pages of B1K, H84 segment 41r and the layout test's fixture,
0 in the test registry and 1 in the defect register (docs/explainer_inputs_2026-10-15/
strict_unit_scan_output.txt); other page builders' texts were not scanned.
"""
from __future__ import annotations

import html
import re
import sys
import urllib.request
from pathlib import Path
from typing import List, Tuple

# THE UNIT AFTER A NUMBER: each unit a whole word (the second recheck of the explainer, Oct 15, 2026: "the 0.00001
# perturbation" passed the list used until then, UNIT_AFTER_PREFIX, because several of its units match the start of a
# longer word: "per" in perturbation, perception and period, "sec" and "second" in section and secondary, "day" in
# daylight, "hour" in hourly, "cm" in cmd). The same units, each ending at a word boundary, with their plural forms and
# "percent" / "per cent" named. Switched on for every caller on Oct 15, 2026 (the decision
# explainer-part2-rule-confirmed-20261015); the reply check
# holds the same pattern.
UNIT_AFTER_STRICT = re.compile(
    r"^\s*(%|°|per ?cent\b|cm\b|mm\b|m\b|km\b|s\b|ms\b|secs?\b|seconds?\b|minutes?\b|min\b|hours?\b|h\b|days?\b|"
    r"GB\b|GiB\b|MB\b|MiB\b|KB\b|kB\b|kg\b|g\b|N\b|W\b|Hz\b|kPa\b|Pa\b|steps?\b|model steps\b|frames?\b|per\b|x\b|"
    r"times\b|fps\b|bytes\b|degrees?\b)", re.I)
UNIT_AFTER = UNIT_AFTER_STRICT
# the list used until Oct 15, 2026, each unit matching the start of any word; read by no rule, kept only so the scan of
# what the switch newly flagged (docs/explainer_inputs_2026-10-15/strict_unit_scan.py) can be repeated
UNIT_AFTER_PREFIX = re.compile(
    r"^\s*(%|°|cm|mm|m\b|km|s\b|ms\b|sec|second|minute|min\b|hour|h\b|day|GB|GiB|MB|MiB|KB|kB|kg|g\b|N\b|W\b|"
    r"Hz|kPa|Pa\b|steps?|model steps|frames?|per|x\b|times|fps|bytes|degrees?)", re.I)
DECIMAL = re.compile(r"(?<![\w.\-/\\#:])(-?\d*\.\d+)(?![\w/\\]|\.\d)")
SCIENTIFIC = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?[eE][-+]?\d+)(?![\w.])")
# A commit's short hash can hold only digits around one "e" ("2e3746163"; the old dashboard's chip "HEAD 2e374616"). It
# is a name, not a number: git prints at least 7 characters of a hash, while a number in exponent form that a page
# prints carries the exponent's sign (Python's "7.3e-06", "1e+20"), a decimal point, or is shorter ("1e5", "3e10").
# Found Oct 5, 2026 by the dashboard fix agent: the registered unit run of 1:19 PM PDT at commit 2e3746163 failed on the
# old dashboard's chip, and the page cleaner (plain_numbers.sci_to_fixed) rewrote such a hash into "inf" ("commit
# 2e3746163" printed as "commit inf"). Both use commit_like.
COMMIT_LIKE = re.compile(r"\d+[eE]\d+")
COMMIT_MIN_CHARS = 7


def commit_like(token: str) -> bool:
    """True for a token of at least 7 characters made of digits around one "e", no sign and no point: a commit's
    short hash, never a number in exponent form a page prints."""
    return len(token) >= COMMIT_MIN_CHARS and COMMIT_LIKE.fullmatch(token) is not None
TAGS = re.compile(r"<(script|style)\b.*?</\1>|<[^>]+>", re.S | re.I)
# A command-line flag and the value typed after it ("--babble-noise 0.05", "--finger-armature=0.0004")
# is a setting as the operator types it, not a score; so is the text of a <code> element. Neither is
# scanned (added 2026-09-23 by fix-dashboards: the dashboards list launch settings and what changed).
FLAG_LITERAL = re.compile(r"--[A-Za-z][\w-]*(?:[ \t=][^\s,;()]+)?")
CODE = re.compile(r"<code\b[^>]*>.*?</code>", re.S | re.I)


def page_text(markup: str) -> str:
    """The text a reader sees in static HTML: scripts, styles and tags removed, entities decoded.
    <code> literals are dropped (see FLAG_LITERAL)."""
    return html.unescape(TAGS.sub(" ", CODE.sub(" ", markup)))


def bare_decimals(text: str, context: int = 40) -> List[Tuple[str, str]]:
    """Every bare score decimal in `text` as (the number, the text around it). The unit after a number is read with
    UNIT_AFTER (each unit a whole word)."""
    hits: List[Tuple[str, str]] = []
    text = FLAG_LITERAL.sub(" ", text)
    for m in SCIENTIFIC.finditer(text):
        if commit_like(m.group(1)):
            continue                       # a commit's short hash, not a number
        hits.append((m.group(1), text[max(0, m.start() - context):m.end() + context].strip()))
    cleaned = SCIENTIFIC.sub(" ", text)
    for m in DECIMAL.finditer(cleaned):
        raw = m.group(1)
        try:
            value = abs(float(raw))
        except ValueError:
            continue
        if value >= 1 or UNIT_AFTER.match(cleaned[m.end():m.end() + 14]):
            continue
        hits.append((raw, cleaned[max(0, m.start() - context):m.end() + context].strip()))
    return hits


# ---------------------------------------------------------------- numbers of any size (fix-scanner)
# A number as a page prints it: optional minus, digits with thousands commas, optional fraction. Not
# part of a word, a path, a flag, a clock time ("3:44") or a version ("v3.6", "3.6.1"). The minus
# may be the typeset one (U+2212) the addendum prints.
_BEFORE = r"(?<![\w.\-−/\\#:])"
_AFTER = r"(?![\w/\\:]|\.\d)"
_DIGITS = r"[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)"      # "1,234", never the "17," of "-17, far below"
_NUM = _BEFORE + r"(?P<num>" + _DIGITS + r"(?:\.\d+)?)" + _AFTER
_FRACTION = _BEFORE + r"(?P<num>" + _DIGITS + r"\.\d+)" + _AFTER
SCORE_WORDS = ("ratio", "score", "cosine", "correlation", "entropy", "loss", "margin", "share", "drift",
               "advantage", "gain", "spread", "error")
# the score word, then at most three more words (or a sign such as "=", "<", ":"), then the number
SCORE_NUMBER = re.compile(
    r"\b(?P<word>(?:%s)(?:s|es)?)\b(?:[ \t]*[:=<>≤≥~]?[ \t]*[A-Za-z][\w'-]*){0,3}[ \t]*[:=<>≤≥~]?[ \t]*%s"
    % ("|".join(SCORE_WORDS), _FRACTION), re.I)
# "= meaning" or "means ..." after a number: the number is a scale's named end, not a reading
SCALE_END_AFTER = re.compile(r"^\s*(=|means\b|is (?:the )?(?:top|bottom|end|floor|ceiling)\b)", re.I)
# a count: "1.5 of every 100", "3 out of 4", "7 in every billion" (never "1.0 in every record")
COUNT_AFTER = re.compile(r"^\s*(?:(?:of|in) every|out of|of|in)\s+(?:\d[\d,]*|ten\b|hundred|thousand|million|billion)",
                         re.I)
MULT_PREFIX = re.compile(r"(?<![\w.])[x×](?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?!\w|\.\d)")
MULT_SUFFIX = re.compile(_BEFORE + r"(?P<num>" + _DIGITS + r"(?:\.\d+)?)(?:x|[ \t]?×)(?!\w|[ \t]*\d)")
MULT_TIMES = re.compile(_BEFORE + r"(?P<num>" + _DIGITS + r"\.\d+)[ \t]+times\b")
# what follows an anchored multiplier: one space (or an opening bracket), then a word naming what is
# multiplied ("its first value", "the shuffled error", "next-state error", "further back"). Two or
# more spaces are an element boundary in page_text (a tag became a space), so no anchor crosses them.
ANCHOR_AFTER = re.compile(r"^ ?\(?(?P<word>[A-Za-z][\w'-]*)")
NOT_ANCHOR = frozenset("""and or but nor in on at for with while when where to from by is was are were be
been now then so after before during since until which who it if because ago here there per yet also
still only just each every""".split())
# a sentence that says what is multiplied before the number: "how good it felt multiplied by (1 + the
# curiosity drive) - 1x for a moment outside the band, 2x at the band's midpoint"
SENTENCE_ANCHOR = re.compile(r"\bmultipl(?:y|ied|ies|ying)\b|\bscaled by\b|\bfactor\b", re.I)
# a whole number after "x" is a repeat count when no score word precedes it: "predict, feed back, ×3",
# "vibration x4", "loop ×20"; "drift x1" is still a multiplier
SCORE_BEFORE = re.compile(r"\b(?:%s)(?:s|es)?\b(?:\W+\w+){0,2}\W*$" % "|".join(SCORE_WORDS), re.I)
# "their inputs ranged from -32.9 to -7.1", "its input rises above -3", "all 256 units sit at -29 to -5.5"
UNIT_INPUT = re.compile(r"\b(?:inputs?|pre-?activations?|units?\b[^.;:\n]{0,20}?\b(?:sit|sits|sat|lie|lies|lay|range|"
                        r"ranges|ranged|ranging))\b(?P<mid>[^.;:\n]{0,50}?)" + _NUM, re.I)
UNIT_WORD = re.compile(r"\b(units?|neurons?|hidden|respond\w*)\b", re.I)


# the second end of a range: "4.8–11.4 months", "1.16 to 2.45 cm"
RANGE_AFTER = re.compile(r"^\s*(?:–|—|-|−|to)\s*[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _unit_follows(text: str, end: int) -> bool:
    """Whether a unit follows the number ending at `end`, directly or after the range it opens."""
    if UNIT_AFTER.match(text[end:end + 14]):
        return True
    r = RANGE_AFTER.match(text[end:end + 30])
    return bool(r) and bool(UNIT_AFTER.match(text[end + r.end():end + r.end() + 14]))


def _anchored(text: str, end: int) -> bool:
    """Whether the words right after a multiplier name what it multiplies."""
    a = ANCHOR_AFTER.match(text[end:end + 40])
    return bool(a) and a.group("word").lower() not in NOT_ANCHOR


def _around(text: str, start: int, end: int, context: int) -> str:
    return " ".join(text[max(0, start - context):end + context].split())


def _is_bare_decimal(text: str, m) -> bool:
    """Whether bare_decimals already reports the number at this match (below 1, a fraction, no unit)."""
    raw = m.group("num")
    try:
        value = abs(float(raw.replace(",", "").replace("−", "-")))
    except ValueError:
        return False
    return "." in raw and value < 1 and not UNIT_AFTER.match(text[m.end():m.end() + 14])


def unanchored_numbers(text: str, context: int = 40) -> List[Tuple[str, str, str]]:
    """Every unanchored score, multiplier or hidden-unit input value in `text`, as (the number,
    the rule, the text around it). The rules are in the module docstring."""
    text = FLAG_LITERAL.sub(" ", text)
    hits: List[Tuple[int, str, str, str]] = []
    for m in SCORE_NUMBER.finditer(text):
        raw = m.group("num")
        try:
            value = abs(float(raw.replace(",", "").replace("−", "-")))
        except ValueError:
            continue
        after = text[m.end():m.end() + 30]
        if value < 1 or _unit_follows(text, m.end()) or SCALE_END_AFTER.match(after) or COUNT_AFTER.match(after):
            continue
        hits.append((m.start("num"), raw, "score", _around(text, m.start("num"), m.end(), context)))
    mults = []
    for rx, form in ((MULT_PREFIX, "x%s"), (MULT_SUFFIX, "%sx"), (MULT_TIMES, "%s times")):
        for m in rx.finditer(text):
            if rx is MULT_PREFIX and "." not in m.group("num") and not SCORE_BEFORE.search(
                    text[max(0, m.start() - 60):m.start()]):
                continue                          # "feed back, ×3": a repeat count
            if rx is MULT_TIMES and COUNT_AFTER.match(text[m.end():m.end() + 30]):
                continue                          # "99.8 times in 100": a count
            mults.append((m.start(), m.end(), form % m.group("num"), _anchored(text, m.end())))
    mults.sort()
    for i, (start, end, shown, anchored) in enumerate(mults):
        if anchored:
            continue
        # an earlier multiplier of the same sentence that names what it multiplies anchors this one
        # ("It sits at 2.3 times its next-state error ... undamaged runs measured 1.16 to 2.45 times;"),
        # and so does a sentence that says what is multiplied ("multiplied by ... - 1x ..., 2x ...")
        sentence_start = max((s.end() for s in SENTENCE_END.finditer(text, 0, start)), default=0)
        if any(a and sentence_start <= s < start for s, _e, _n, a in mults[:i]):
            continue
        if SENTENCE_ANCHOR.search(text, sentence_start, start):
            continue
        hits.append((start, shown, "multiplier", _around(text, start, end, context)))
    for m in UNIT_INPUT.finditer(text):
        after = text[m.end():m.end() + 30]
        sentence = text[max(0, m.start() - 150):m.end() + 150]
        if not UNIT_WORD.search(sentence) or _is_bare_decimal(text, m):
            continue
        if _unit_follows(text, m.end()) or COUNT_AFTER.match(after) or SCALE_END_AFTER.match(after):
            continue
        word = re.match(r"^\s*([A-Za-z][\w'-]*)", after)
        if word and word.group(1).lower() not in NOT_ANCHOR:
            continue                                  # "inputs from 26 handles": a count of things
        hits.append((m.start("num"), m.group("num"), "unit input",
                     _around(text, m.start(), m.end(), context)))
    seen, out = set(), []
    for start, number, rule, around in sorted(hits):
        if (start, rule) not in seen:
            seen.add((start, rule))
            out.append((number, rule, around))
    return out


def readout_problems(text: str, context: int = 40) -> List[Tuple[str, str, str]]:
    """The bare decimals and the unanchored numbers of `text`, as (the number, the rule, the text
    around it); the rule of a bare decimal is 'bare decimal'."""
    return ([(n, "bare decimal", " ".join(c.split())) for n, c in bare_decimals(text, context)]
            + unanchored_numbers(text, context))


LONG_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*\.\d+")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def long_numbers(text: str, max_significant: int = 7) -> List[str]:
    """Numbers printed with more than `max_significant` significant digits ("0.9999999268563718",
    "1.0000000000000002"): raw floats that reached the page unformatted, whatever their size. A
    small value written with few significant digits ("0.0000073") is not one."""
    out = []
    for m in LONG_NUMBER.finditer(text):
        digits = m.group(0).lstrip("-").replace(",", "").replace(".", "").lstrip("0")
        if len(digits) > max_significant:
            out.append(m.group(0))
    return out


def doubled_sentences(text: str, min_words: int = 6) -> List[str]:
    """Sentences of at least `min_words` words that appear twice in the same text, and doubled full
    stops (".." not part of an ellipsis): the same paragraph printed twice, or "Scale: <sentence>.."."""
    out = []
    for m in re.finditer(r"(?<!\.)\.\.(?!\.)", text):
        out.append(text[max(0, m.start() - 40):m.end() + 5].strip())
    seen = set()
    for raw in SENTENCE_END.split(text):
        s = " ".join(raw.split()).strip()
        if len(s.split()) < min_words:
            continue
        key = s.rstrip(".").lower()
        if key in seen:
            out.append(s)
        seen.add(key)
    return out


def _read(source: str) -> str:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=30) as r:
            return r.read().decode("utf-8", "replace")
    return Path(source).read_text(encoding="utf-8", errors="replace")


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = list(sys.argv[1:] if argv is None else argv)
    found = 0
    for source in argv:
        raw = _read(source)
        text = page_text(raw) if source.endswith((".html", ".htm")) or "<html" in raw[:2000].lower() else raw
        for number, rule, around in readout_problems(text):
            found += 1
            print("%s: %s %s in '...%s...'" % (source, RULE_WORDS[rule], number, around))
    print("%d number%s found that a reader cannot place (DESIGN.md section 11)" % (found, "" if found == 1 else "s"))
    return 1 if found else 0


RULE_WORDS = {"bare decimal": "bare score decimal", "score": "score with no unit",
              "multiplier": "multiplier with nothing named after it", "unit input": "raw hidden-unit input value"}


if __name__ == "__main__":
    sys.exit(main())
