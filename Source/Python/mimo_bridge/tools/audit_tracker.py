# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Keep docs/Explainer_Audit_Resolution_Tracker.html consistent with itself.

A problem's state is shown in three places on that page - the all-problems table, the
problem's own card, and the header counts - and until 2026-09-19 each was edited by hand.
Measured that day: the table said 76 problems were unsettled, the cards said 90 (14 cards
still read OPEN or OPERATOR DECISION for problems the table had closed on September 16-17),
and 37 problems settled on September 18 appeared in none of the three. This tool makes the
table the one source, rewrites the cards and the header from it, and applies a dated batch
of updates with its evidence paragraph.

    python -m mimo_bridge.tools.audit_tracker --check
    python -m mimo_bridge.tools.audit_tracker --apply docs/explainer_audit_updates/<file>.json

An update file: {"title": "Closed 2026-09-19, fifth pass", "items": [{"id": "P118",
"state": "FIXED", "ticks": {"rechecked": true, "fixed": true, "tested": true},
"note": "one or two sentences of evidence: what changed, the commit, the test"}]}
"""
from __future__ import annotations

import argparse
import collections
import html as _html
import json
import re
from pathlib import Path

TRACKER = Path(__file__).resolve().parents[4] / "docs" / "Explainer_Audit_Resolution_Tracker.html"
CHIP = {"FIXED + VERIFIED": "ok", "FIXED": "ok", "OPEN": "unk", "NOT REACHED": "unk",
        "OPERATOR DECISION": "warn", "NOT A BUG": "bad", "WITHDRAWN BY THE AUDIT": "bad"}
SETTLED = ("FIXED + VERIFIED", "FIXED", "NOT A BUG", "WITHDRAWN BY THE AUDIT")
TICKS = ("rechecked", "fixed", "tested", "seen")            # the table's four tick columns, in order
_STATE_CHIP = r'<span class="chip (?:ok|warn|unk|bad)">([^<]*)</span>'
_ROW = r"(<tr><td><a href='#%s'>.*?</tr>)"
_CARD_HEAD = r"(<div class='card' id='%s'>\s*<h3[^>]*>)(.*?)(</h3>)"


def _table_segment(page: str):
    i, j = page.index("id='all'"), page.index("id='detail'")
    return i, j


def table_states(page: str) -> dict:
    i, j = _table_segment(page)
    out = {}
    for m in re.finditer(r"<tr><td><a href='#(P\d{3})'>(.*?)</tr>", page[i:j], flags=re.S):
        chips = re.findall(_STATE_CHIP, m.group(2))
        out[m.group(1)] = chips[-1] if chips else "?"
    return out


def card_states(page: str) -> dict:
    out = {}
    for m in re.finditer(r"<div class='card' id='(P\d{3})'>\s*<h3[^>]*>(.*?)</h3>", page, flags=re.S):
        chips = re.findall(_STATE_CHIP, m.group(2))
        out[m.group(1)] = chips[-1] if chips else "?"
    return out


def _chip(state: str) -> str:
    return f'<span class="chip {CHIP[state]}">{state}</span>'


def set_state(page: str, pid: str, state: str, ticks: dict | None = None) -> str:
    """Write one problem's state into its table row and its card; set any ticks given."""
    if state not in CHIP:
        raise ValueError(f"unknown state {state!r}")
    i, j = _table_segment(page)
    seg = page[i:j]
    m = re.search(_ROW % pid, seg, flags=re.S)
    if not m:
        raise KeyError(pid)
    row = m.group(1)
    cells = re.findall(r"<td class='tick [yn]'>[^<]*</td>", row)
    if len(cells) != 4:
        raise ValueError(f"{pid}: expected 4 tick cells, found {len(cells)}")
    for k, name in enumerate(TICKS):
        if ticks and name in ticks:
            new = "<td class='tick y'>&#10003;</td>" if ticks[name] else "<td class='tick n'>&middot;</td>"
            cells[k] = new
    parts = re.split(r"<td class='tick [yn]'>[^<]*</td>", row)
    row2 = parts[0] + "".join(c + p for c, p in zip(cells, parts[1:]))
    chips = list(re.finditer(_STATE_CHIP, row2))
    row2 = row2[:chips[-1].start()] + _chip(state) + row2[chips[-1].end():]
    seg = seg[:m.start(1)] + row2 + seg[m.end(1):]
    page = page[:i] + seg + page[j:]
    return _set_card(page, pid, state)


def _set_card(page: str, pid: str, state: str) -> str:
    m = re.search(_CARD_HEAD % pid, page, flags=re.S)
    if not m:
        raise KeyError(f"no card for {pid}")
    head = m.group(2)
    chips = list(re.finditer(_STATE_CHIP, head))
    if not chips:
        raise ValueError(f"{pid}: card has no state chip")
    head = head[:chips[-1].start()] + _chip(state) + head[chips[-1].end():]
    return page[:m.start(2)] + head + page[m.end(2):]


def sync_cards(page: str) -> tuple[str, list]:
    """The table is the source: rewrite every card whose state differs from its row."""
    table, cards = table_states(page), card_states(page)
    changed = [p for p in table if cards.get(p) != table[p]]
    for p in changed:
        page = _set_card(page, p, table[p])
    return page, changed


def add_closure(page: str, title: str, items: list) -> str:
    """Append a dated evidence paragraph after the last 'Closed ...' paragraph of section 0."""
    body = " ".join(f"<b>{it['id']}</b> &rarr; {it['state']} - {_html.escape(it['note'], quote=False)}" for it in items)
    para = f"\n<p class='sub'><b>{_html.escape(title)}:</b> {body}</p>"
    last = None
    for m in re.finditer(r"<p class='sub'><b>Closed[^<]*</b>.*?</p>", page, flags=re.S):
        last = m
    if last is None:
        raise ValueError("no earlier closure paragraph to append after")
    return page[:last.end()] + para + page[last.end():]


def refresh_header(page: str) -> str:
    states = table_states(page)
    n = len(states)
    c = collections.Counter(states.values())
    fixed = c["FIXED + VERIFIED"] + c["FIXED"]
    dismissed = c["NOT A BUG"] + c["WITHDRAWN BY THE AUDIT"]
    rest = n - fixed - dismissed
    i, j = _table_segment(page)
    rechecked = len(re.findall(r"<tr><td><a href='#P\d{3}'>(?:(?!</tr>).)*?</code></td><td class='tick y'>", page[i:j], flags=re.S))
    page = re.sub(r"<div class='bar'><i class='f' style='width:[\d.]+%'></i><i class='n' style='width:[\d.]+%'></i>"
                  r"<i class='o' style='width:[\d.]+%'></i></div>",
                  f"<div class='bar'><i class='f' style='width:{100 * fixed / n:.2f}%'></i>"
                  f"<i class='n' style='width:{100 * dismissed / n:.2f}%'></i>"
                  f"<i class='o' style='width:{100 * rest / n:.2f}%'></i></div>", page, count=1)
    line = (f"<p class='small'>{fixed} fixed &middot; {dismissed} dismissed as not real &middot; {c['OPEN']} still open, "
            f"plus {c['OPERATOR DECISION']} waiting on an operator decision and {c['NOT REACHED']} never reached "
            f"&middot; {100 * (fixed + dismissed) / n:.1f}% settled of {n}</p>")
    page, k = re.subn(r"<p class='small'>\d+ fixed &middot; .*? settled of \d+</p>", line, page, count=1, flags=re.S)
    if k != 1:
        raise ValueError("header summary line not found")
    tiles = {"problems": n, "fixed + verified": c["FIXED + VERIFIED"], "fixed": c["FIXED"], "open": c["OPEN"],
             "audit was wrong": c["NOT A BUG"], "withdrawn in audit": c["WITHDRAWN BY THE AUDIT"],
             "operator decision": c["OPERATOR DECISION"], "not reached": c["NOT REACHED"], "re-checked": rechecked}
    for label, value in tiles.items():
        page, k = re.subn(r"<div class='count'><div class='n'>\d+</div><div class='t'>%s</div></div>" % re.escape(label),
                          f"<div class='count'><div class='n'>{value}</div><div class='t'>{label}</div></div>", page, count=1)
        if k != 1:
            raise ValueError(f"count tile {label!r} not found")
    return page


def header_counts(page: str) -> dict:
    return {t: int(v) for v, t in re.findall(r"<div class='count'><div class='n'>(\d+)</div><div class='t'>([^<]+)</div></div>", page)}


def check(page: str) -> list:
    """Every inconsistency between the table, the cards and the header, as sentences."""
    problems = []
    table, cards = table_states(page), card_states(page)
    for p in sorted(table):
        if cards.get(p) != table[p]:
            problems.append(f"{p}: the table says {table[p]}, its card says {cards.get(p)}")
    c = collections.Counter(table.values())
    head = header_counts(page)
    want = {"problems": len(table), "fixed + verified": c["FIXED + VERIFIED"], "fixed": c["FIXED"], "open": c["OPEN"],
            "audit was wrong": c["NOT A BUG"], "withdrawn in audit": c["WITHDRAWN BY THE AUDIT"],
            "operator decision": c["OPERATOR DECISION"], "not reached": c["NOT REACHED"]}
    for k, v in want.items():
        if head.get(k) != v:
            problems.append(f"header tile {k!r} reads {head.get(k)}, the table counts {v}")
    return problems


def apply(page: str, update: dict) -> str:
    page, _ = sync_cards(page)
    for it in update["items"]:
        page = set_state(page, it["id"], it["state"], it.get("ticks"))
    page = add_closure(page, update["title"], update["items"])
    return refresh_header(page)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--apply", default="")
    ap.add_argument("--tracker", default=str(TRACKER))
    a = ap.parse_args(argv)
    path = Path(a.tracker)
    page = path.read_text(encoding="utf-8")
    if a.apply:
        update = json.loads(Path(a.apply).read_text(encoding="utf-8"))
        page = apply(page, update)
        path.write_text(page, encoding="utf-8", newline="")
        print(f"applied {len(update['items'])} updates: {update['title']}")
    problems = check(page)
    c = collections.Counter(table_states(page).values())
    print(dict(c))
    for p in problems:
        print("INCONSISTENT:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
