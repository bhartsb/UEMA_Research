# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Numbers in the forms DESIGN.md section 11 allows, for every page a person reads.

Section 11 (operator 2026-09-22): no bare score decimals where a person reads. A number shown is
a COUNT the reader can picture ("32 of every 100 orders"), a REAL UNIT with a lived anchor
("0.8 cm", "12 degrees", "per hour"), or a PERCENTAGE BETWEEN TWO NAMED ENDS ("20% of the way from
'the order made no difference' to 'predicted the move exactly'"), and never scientific notation.
tools/check_no_bare_decimals.py is the scanner that holds every page to that; these are the
formatters the pages build their sentences with, so a sentence passes the scanner by construction.
"""
from __future__ import annotations

import math
import re
from typing import Optional

_SCI = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?[eE][-+]?\d+)(?![\w.])")


def finite(value) -> Optional[float]:
    """float(value) when it is a finite real number (not a bool), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def fixed(x: float, sig: int = 2) -> str:
    """x with `sig` significant digits in fixed notation, never scientific: 7.3e-06 -> "0.0000073",
    0.0014 -> "0.0014", 12.4 -> "12", 1234.5 -> "1,235". Values of 10 and above are whole numbers,
    1 to 10 keep one decimal (a trailing .0 dropped)."""
    x = float(x)
    if x == 0 or not math.isfinite(x):
        return "0" if x == 0 else str(x)
    a = abs(x)
    if a >= 10:
        return f"{x:,.0f}"
    if a >= 1:
        text = f"{x:.1f}"
        return text[:-2] if text.endswith(".0") else text
    digits = min(20, max(0, sig - 1 - int(math.floor(math.log10(a)))))
    text = f"{x:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text not in ("-0", "") else "0"


def pct(share: float, sig: int = 2) -> str:
    """A share (1 = all) as a percentage with significant digits: 0.0000000731 -> "0.0000073%"."""
    return fixed(100.0 * float(share), sig) + "%"


def pct_points(value: float, sig: int = 2) -> str:
    """A value already in percent: 54.2 -> "54%", 0.043 -> "0.043%"."""
    return fixed(float(value), sig) + "%"


def per(share: float, noun: str, sig_min: int = 1) -> str:
    """A share as a count per 100 (or per 1,000, 100,000 ... when under 1 in 100):
    0.745 -> "75 of every 100 <noun>", 0.004 -> "4 of every 1,000 <noun>", 0 -> "0 of every 100"."""
    share = float(share)
    if share == 0:
        return f"0 of every 100 {noun}".rstrip()
    base = 100
    while abs(share) * base < 1 and base < 10 ** 12:
        base *= 10
    n = share * base
    count = f"{n:,.0f}" if abs(n) >= 10 or base > 100 else fixed(n, 2)
    return f"{count} of every {base:,} {noun}".rstrip()


def per100(share: float) -> str:
    """The count part only: 0.745 -> "75", used where the sentence names 'of every 100'."""
    return f"{100.0 * float(share):,.0f}"


def between(value: float, lo: float, hi: float, lo_name: str, hi_name: str, sig: int = 2) -> str:
    """Where a value sits between two named ends, as a percentage of the way:
    "3% of the way from 'unrelated' to 'exact'". Below the low end reads as a negative share."""
    span = float(hi) - float(lo)
    share = (float(value) - float(lo)) / span if span else 0.0
    return f"{pct(share, sig)} of the way from '{lo_name}' to '{hi_name}'"


def share_between(value: float, lo: float, hi: float) -> float:
    span = float(hi) - float(lo)
    return (float(value) - float(lo)) / span if span else 0.0


def unit(value: float, name: str, sig: int = 2) -> str:
    """A value in a real unit: unit(0.83, "cm") -> "0.83 cm"; large values get separators."""
    return f"{fixed(value, sig)} {name}"


def times(ratio: float, sig: int = 2) -> str:
    """A multiple: 52.49 -> "52 times", 1.26 -> "1.3 times"."""
    return f"{fixed(ratio, sig)} times"


def count(n, noun: str, plural: Optional[str] = None) -> str:
    n = int(round(float(n)))
    word = noun if n == 1 else (plural or noun + "s")
    return f"{n:,} {word}"


def sci_to_fixed(text: str, sig: int = 2) -> str:
    """Rewrite every scientific-notation number in a sentence written elsewhere into fixed
    notation with the same significant digits ("7.3e-06%" -> "0.0000073%"). The sentence's words
    are left as they are, and so is a commit's short hash made of digits around one "e" ("2e3746163" was printed
    as "inf" until Oct 5, 2026; check_no_bare_decimals.commit_like)."""
    from mimo_bridge.tools.check_no_bare_decimals import commit_like

    def rep(m):
        raw = m.group(1)
        if commit_like(raw):
            return raw
        mant = raw.lower().split("e")[0].lstrip("-")
        digits = len(mant.replace(".", "").lstrip("0")) or 1
        return fixed(float(raw), max(sig, digits))
    return _SCI.sub(rep, str(text))


WITHHELD = ("(Withheld here: this archived text prints scores in a form DESIGN.md section 11 bars from the "
            "page; its source file keeps it word for word.)")
_LITERAL_KEYS = {"url", "href", "path", "paths", "subject", "run_dir", "checkpoint", "id", "sha256", "receipts",
                 "experiment_id", "review", "source", "sources", "links", "label", "date", "state", "kind",
                 "outcome", "lineage", "validation", "capability", "abilities", "model_identity"}


def plain_tree(obj, key=None):
    """A copy of a JSON-like record with every sentence cleaned for the page: scientific numbers
    rewritten, and a sentence that still carries a bare score decimal replaced by WITHHELD (archived
    texts written before section 11; the page links their source files). Identifiers, paths and
    links are left as they are."""
    if isinstance(obj, dict):
        return {k: (v if k in _LITERAL_KEYS else plain_tree(v, k)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [plain_tree(v, key) for v in obj]
    if isinstance(obj, str):
        out = clean(obj)
        return WITHHELD if out is None else out
    return obj


def clean(text: str) -> Optional[str]:
    """A sentence written elsewhere, shown only when it passes section 11 after the scientific
    numbers are rewritten; None when a bare score decimal remains (the caller says so instead)."""
    from mimo_bridge.tools.check_no_bare_decimals import bare_decimals
    if text is None:
        return None
    out = sci_to_fixed(str(text))
    return None if bare_decimals(out) else out
