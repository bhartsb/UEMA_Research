# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The audit tracker page shows each problem's state in three places; they must agree.

Measured 2026-09-19 before tools/audit_tracker.py existed: the all-problems table counted 76
unsettled problems, the cards 90 (14 cards still read OPEN or OPERATOR DECISION for problems
the table had closed), and 37 problems settled on September 18 appeared nowhere. The first
test reads the real page; the rest drive the tool on a three-problem page built here.
"""
import pytest

from mimo_bridge.tools import audit_tracker as AT


def _page():
    def row(pid, state, ticks="nnnn"):
        cells = "".join("<td class='tick y'>&#10003;</td>" if t == "y" else "<td class='tick n'>&middot;</td>" for t in ticks)
        return (f"<tr><td><a href='#{pid}'><code>{pid}</code></a></td><td><span class=\"chip sev-high\">HIG</span></td>"
                f"<td class='small'>code bug</td><td>claim of {pid}</td><td class='small'><code>file.py:1</code></td>"
                f"{cells}<td><span class=\"chip {AT.CHIP[state]}\">{state}</span></td></tr>")

    def card(pid, state):
        return (f"<div class='card' id='{pid}'>\n<h3 style='margin-top:0'><code>{pid}</code> "
                f"<span class=\"chip sev-high\">HIGH</span> <span class=\"chip kind\">code bug</span> "
                f"<span class=\"chip {AT.CHIP[state]}\">{state}</span> claim of {pid}</h3>\n<p>detail</p>\n</div>\n")

    tiles = "".join(f"<div class='count'><div class='n'>0</div><div class='t'>{t}</div></div>" for t in (
        "problems", "fixed + verified", "fixed", "open", "audit was wrong", "withdrawn in audit",
        "operator decision", "not reached", "re-checked", "audit corrections"))
    return ("<body><div class='bar'><i class='f' style='width:0.00%'></i><i class='n' style='width:0.00%'></i>"
            "<i class='o' style='width:100.00%'></i></div>"
            "<p class='small'>0 fixed &middot; 0 dismissed as not real &middot; 0 still open, plus 0 waiting on an operator "
            "decision and 0 never reached &middot; 0.0% settled of 3</p>"
            f"<div class='counts'>{tiles}</div>"
            "<h2 id='later'>0.</h2><p class='sub'><b>Closed 2026-09-16, third pass:</b> <b>P001</b> &rarr; FIXED - x</p>"
            "<h2 id='all'>3. All problems</h2><table><tbody>"
            + row("P001", "FIXED", "ynnn") + row("P002", "OPEN") + row("P003", "OPERATOR DECISION", "ynnn")
            + "</tbody></table><h2 id='detail'>4. Detail</h2>"
            + card("P001", "OPEN") + card("P002", "OPEN") + card("P003", "OPERATOR DECISION") + "</body>")


def test_the_real_tracker_agrees_with_itself():
    page = AT.TRACKER.read_text(encoding="utf-8")
    states = AT.table_states(page)
    assert len(states) == 451 and set(states.values()) <= set(AT.CHIP)
    assert AT.check(page) == [], "the tracker's table, cards and header counts disagree; run tools/audit_tracker.py"


def test_a_stale_card_is_reported_and_synced_from_the_table():
    page = _page()
    assert any("P001: the table says FIXED, its card says OPEN" in p for p in AT.check(page))
    page, changed = AT.sync_cards(page)
    assert changed == ["P001"] and AT.card_states(page)["P001"] == "FIXED"


def test_an_update_reaches_the_row_its_ticks_the_card_the_header_and_the_evidence_paragraph():
    update = {"title": "Closed 2026-09-19, fifth pass", "items": [
        {"id": "P002", "state": "FIXED", "ticks": {"rechecked": True, "fixed": True, "tested": True},
         "note": "bounded at its source; 580 of 1,560 <before>"}]}
    page = AT.apply(_page(), update)
    assert AT.table_states(page) == {"P001": "FIXED", "P002": "FIXED", "P003": "OPERATOR DECISION"}
    assert AT.card_states(page) == AT.table_states(page)
    row = page[page.index("<a href='#P002'>"):page.index("<a href='#P003'>")]
    assert row.count("tick y") == 3 and row.count("tick n") == 1
    head = AT.header_counts(page)
    assert (head["problems"], head["fixed"], head["open"], head["operator decision"], head["re-checked"]) == (3, 2, 0, 1, 3)
    assert "2 fixed &middot; 0 dismissed as not real &middot; 0 still open, plus 1 waiting" in page
    assert "width:66.67%" in page
    assert "<b>Closed 2026-09-19, fifth pass:</b> <b>P002</b> &rarr; FIXED - bounded at its source; 580 of 1,560 &lt;before&gt;" in page
    assert page.index("fifth pass") > page.index("third pass")
    assert AT.check(page) == []


def test_an_unknown_state_or_problem_is_refused():
    with pytest.raises(ValueError):
        AT.set_state(_page(), "P002", "DONE")
    with pytest.raises(KeyError):
        AT.set_state(_page(), "P999", "FIXED")
