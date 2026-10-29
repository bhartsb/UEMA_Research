# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The writing hat: every overseer note passes the reading gate before the dashboard.

The gate (the style guide, 2026-09-05, operator): notes are written in terms of the goals,
and scored adversarially 1-10 for cognitive load. 1-2 passes. 3 or more means rewrite
and re-score until it passes.

One backend, deterministic: the score is computed from measurable load features
(unexplained jargon tokens, numbers per sentence, sentence length, clause chains) and
the rewrite substitutes plain real terms from the glossary, rounds long decimals, and
splits clause chains. Content is never deleted: the raw findings stay in the log for
audits; only the dashboard's copy is rewritten.

The hat returns both scores so the log shows what the gate did.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

MAX_ROUNDS = 3


# Real plain phrasings for the metric names the hats emit. Real terms, never
# metaphors: each right side is the thing itself in words, not a stand-in.
GLOSSARY = [
    (r"\bcausal[_ ]pe\b", "action-conditioned next-state error"),
    (r"\bnext[_ ]pe\b", "next-frame prediction error"),
    (r"\bvis[_ ]adv\b", "vision's action-advantage"),
    (r"\btouch[_ ]adv\b", "touch's action-advantage"),
    (r"\bprior[_ ]w\b", "the endowed agency prior's weight"),
    (r"\brank1\b", "action-matching accuracy"),
    (r"\btd[_ ]gate\b", "the expectation gate"),
    (r"\bpe\b", "prediction error"),
    (r"\bqpos\b", "joint position"),
    (r"\bctrl\b", "motor control value"),
    (r"\bEMA\b", "running average"),
    (r"\bROM\b", "range of motion"),
]
# tokens that count as unexplained jargon if they survive the glossary
JARGON = re.compile(r"\b(?:[a-z]+_[a-z_]+|q[a-z]{3,}|vicreg|fovea_[a-z]+)\b")
NUM = re.compile(r"-?\d+\.?\d*(?:e-?\d+)?")
LONG_FLOAT = re.compile(r"(-?\d+\.\d{4,})")


def _sentences(text: str) -> List[str]:
    # a newline is a point boundary on the dashboard: four short labeled lines are
    # four points, not one run-on sentence (measured: the designer's 4-line note
    # scored 4 as one blob and 1 as lines - the lines were the truth)
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def score(text: str) -> int:
    """Cognitive load 1-10 from measurable features. 1-2 = passes the gate."""
    s = 1
    jargon_types = set(JARGON.findall(text))
    s += min(3, len(jargon_types))
    for sent in _sentences(text):
        if len(NUM.findall(sent)) > 3:
            s += 2
        if len(sent.split()) > 35:
            s += 1
        if sent.count(";") > 2:
            s += 1
    if LONG_FLOAT.search(text):
        s += 1
    return max(1, min(10, s))


def _det_rewrite(text: str) -> str:
    out = text
    for pat, plain in GLOSSARY:
        out = re.sub(pat, plain, out, flags=re.IGNORECASE)
    # long decimals carry no extra decision content on a dashboard note
    out = LONG_FLOAT.sub(lambda m: f"{float(m.group(1)):.3g}", out)
    # a chain of three or more clauses becomes sentences
    parts = out.split("; ")
    if len(parts) > 3:
        out = ". ".join(p[0].upper() + p[1:] if p else p for p in parts)
    return out


def gate(lines: List[str]) -> Tuple[List[str], Dict]:
    """Score and rewrite one hat's note lines. Returns (final_lines, record)."""
    text = "\n".join(lines)
    first = score(text)
    if first <= 2:
        return lines, {"first_score": first, "final_score": first,
                       "rounds": 0, "backend": "none-needed"}
    cur, rounds = text, 0
    while score(cur) > 2 and rounds < MAX_ROUNDS:
        nxt = _det_rewrite(cur)
        rounds += 1
        if nxt == cur:
            break
        cur = nxt
    return cur.split("\n"), {"first_score": first, "final_score": score(cur),
                             "rounds": rounds, "backend": "deterministic"}


def gate_report(findings: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], Dict]:
    """Gate every hat's lines. Returns (notes_plain, hat_record)."""
    notes, recs = {}, {}
    for hat, lines in findings.items():
        if not lines:
            continue
        notes[hat], recs[hat] = gate(list(lines))
    worst_first = max((r["first_score"] for r in recs.values()), default=1)
    worst_final = max((r["final_score"] for r in recs.values()), default=1)
    return notes, {"per_hat": recs, "worst_first": worst_first,
                   "worst_final": worst_final}
