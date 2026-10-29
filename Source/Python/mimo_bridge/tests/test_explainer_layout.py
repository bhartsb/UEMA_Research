# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The explainer's diagrams, measured in a real browser, and the one place each box opens.

the style guide "DIAGRAM LAYOUT ON A GRID" (operator 2026-10-02): no label may overlap a box or run past the
drawing's edge. The page carries its own check: opened with #layoutcheck in its address it measures the
architecture diagram of tab 1 (at one unit per pixel), every detail diagram and the learning circle, every
text and every box with getBBox (in each drawing's own viewBox units), and writes each problem into the
page (viewer.js layoutCheck): a box's own text closer than 3 units to its box's left or right edge or 1
unit to its top or bottom, any other text over a box, two texts or a text and a numbered circle
overlapping, a text past the drawing's edge, a text or box across a block frame, a line through a label
or a box. tools/check_explainer_page.py check_layout opens a copy of a built page in headless Edge with
--dump-dom and reads that report. Planted problems prove the check can fail: an overlap in a detail
diagram, and a caption wider than its box plus a caption-band sentence past the right edge in the
architecture diagram. Measured with the final check on 2026-10-02: the architecture diagram as commit
d1aedf111 drew it had 145 problems on H84's page and 94 on the fixture page; the re-laid-out one has 0.

The operator's other complaint of the same day ("REDUNDANT POPUP"): a box with a detail diagram opened
the side popup AND the panel under the picture with the same text. It now opens the panel only, and
the panel carries the whole entry, the live values and the diagram (check_explainer_page's exerciser
counts both).

Every test that needs the browser skips cleanly where no Chromium-family browser is installed.
"""
import json
import os
import re

import pytest

from mimo_bridge.tools import build_explainer as BX

try:                                          # pytest's prepend import mode puts this folder on the path
    import test_build_explainer as TB
except ImportError:                           # pragma: no cover - an import from the package instead
    from mimo_bridge.tests import test_build_explainer as TB

# every diagram the page draws for a run with every feature the fixture can switch on: the architecture
# diagram of tab 1, the 12 detail diagrams a box opens below the picture, and the learning circle
DIAGRAMS = {"architecture", "intention", "curiosity", "imagination", "replay", "curriculum", "memory",
            "self_state", "voice", "hself", "sense_tactile", "wellness", "caregiver", "learning circle",
            # the model diagram with every gated part's switch on, and with every one off (Oct 15, 2026)
            "architecture, every switch on", "architecture, every switch off"}


def _browser_or_skip():
    from mimo_bridge.tools import check_explainer_page as CK
    exe = CK.find_browser()
    if not exe:
        pytest.skip("no Edge, Chrome or Chromium on this machine (set EXPLAINER_BROWSER)")
    return CK, exe


def _fixture_page(tmp_path):
    """The build fixture with the planner's production settings (H84's): finger targets, the second
    finger search, the normalised ranking, the top-down gate, the 35:65 blend."""
    run = TB._fixture_run(tmp_path)
    TB._set_args(run, top_down_gate="on", hierarchical_self="dark", hself_slots=64,
                 caregiver_mode="on", persistent_memory="on", self_state="learn",
                 intention="on", intention_every=10, motor_hold=5, imagination="on",
                 imagination_steps=3, hand_target_mode="wrist_relative_v1", hand_goals="memory",
                 wellness_rank="normalized", curiosity_goal="on", curiosity_mode="additive",
                 smooth_actions=0.35, plan_horizon="earned", planner_effort_per_body="on")
    page = tmp_path / "page.html"
    page.write_text(BX.render(run), encoding="utf-8", newline="")
    return page


def _said(rep):
    return "\n".join("%s: %s: '%s' / '%s' %s" % (p.get("diagram"), p.get("kind"), p.get("what"),
                                                p.get("with"), p.get("overlap") or "")
                     for p in rep.get("problems") or [])


def test_every_detail_diagram_of_the_fixture_page_is_free_of_overlaps(tmp_path):
    CK, exe = _browser_or_skip()
    rep = CK.check_layout(str(_fixture_page(tmp_path)), exe)
    assert rep["ran"], rep.get("why")
    drawn = {d["name"] for d in rep["diagrams"]}
    assert DIAGRAMS <= drawn, f"diagrams the check never measured: {sorted(DIAGRAMS - drawn)}"
    assert all(d["boxes"] and d["texts"] for d in rep["diagrams"]), rep["diagrams"]
    assert not rep["problems"], _said(rep)


def test_every_detail_diagram_of_the_newest_real_run_page_is_free_of_overlaps(tmp_path):
    """The same check on the newest real run under Saved, whose labels carry its own counts and run
    name (the footnotes that ran past the right edge did so on real data). Skips where there is no
    run data."""
    CK, exe = _browser_or_skip()
    run = BX.newest_run()
    if not run or not os.path.isdir(os.path.join(run, "explain")):
        pytest.skip("no run with snapshots under Saved/mimo_bridge on this checkout")
    page = tmp_path / "real.html"
    page.write_text(BX.render(run), encoding="utf-8", newline="")
    rep = CK.check_layout(str(page), exe)
    assert rep["ran"], rep.get("why")
    assert {"architecture", "intention", "learning circle"} <= {d["name"] for d in rep["diagrams"]}
    assert not rep["problems"], _said(rep)


def test_the_layout_check_catches_a_planted_overlap(tmp_path):
    """The control: a diagram planted into a good page with a label laid over a box, a footnote-free
    label running past the right edge, and a line drawn through a label. Each must be reported, under
    the planted diagram's name, and nothing else on the page may be."""
    CK, exe = _browser_or_skip()
    good = _fixture_page(tmp_path).read_text(encoding="utf-8")
    tail = "</script>\n</body></html>"
    assert good.endswith(tail)
    planted = (';DETAIL.planted = () => flowDiagram(300, 120, ['
               '{x: 10, y: 10, w: 120, h: 40, l1: "a box"}, {x: 180, y: 70, w: 100, h: 40, l1: "another"}], ['
               '{pts: [[130, 30], [180, 90]], label: "a label laid over the box", lx: 70, ly: 34},'
               '{pts: [[230, 70], [230, 40]], label: "a label that runs far past the right edge", lx: 240, ly: 30, anchor: "start"},'
               '{pts: [[10, 95], [170, 95]], label: "crossed", lx: 90, ly: 98}'
               '], [], false);')
    page = tmp_path / "planted.html"
    page.write_text(good[:-len(tail)] + planted + tail, encoding="utf-8", newline="")
    rep = CK.check_layout(str(page), exe)
    assert rep["ran"], rep.get("why")
    kinds = {p["kind"] for p in rep["problems"] if p["diagram"] == "planted"}
    assert {"over a box", "past the edge", "line through a label"} <= kinds, _said(rep)
    assert not [p for p in rep["problems"] if p["diagram"] != "planted"], _said(rep)


def test_the_layout_check_catches_an_overflow_planted_in_the_architecture_diagram(tmp_path):
    """The control for the architecture diagram of tab 1. After boot draws it, a band 70 units tall is
    added under its caption band holding a box 80 units wide whose 11-unit caption is wider than the
    box, and a caption-band sentence that starts 120 units left of the drawing's right edge and is longer
    than that. The first must be reported as past its box and the second as past the edge, both under the
    architecture diagram, and nothing else on the page may be."""
    CK, exe = _browser_or_skip()
    good = _fixture_page(tmp_path).read_text(encoding="utf-8")
    tail = "</script>\n</body></html>"
    assert good.endswith(tail)
    planted = (';(function () { const drawn = buildDiagram; buildDiagram = function () { drawn();'
               ' const svg = document.querySelector("#diagram svg"), vb = svg.viewBox.baseVal;'
               ' const W = vb.width, H = vb.height;'
               ' svg.setAttribute("viewBox", "0 0 " + W + " " + (H + 70));'
               ' svg.appendChild(el("g", { class: "dg-node", id: "nd-planted" },'
               '   el("rect", { x: 260, y: H + 10, width: 80, height: 44 }),'
               '   el("text", { x: 300, y: H + 30, "text-anchor": "middle",'
               '                text: "a planted caption far wider than its box" })));'
               ' svg.appendChild(el("text", { x: W - 120, y: H + 64, class: "sublabel",'
               '   text: "a planted caption-band sentence that runs on past the right edge" }));'
               ' }; })();')
    page = tmp_path / "planted_architecture.html"
    page.write_text(good[:-len(tail)] + planted + tail, encoding="utf-8", newline="")
    rep = CK.check_layout(str(page), exe)
    assert rep["ran"], rep.get("why")
    found = {(p["kind"], p["what"]) for p in rep["problems"] if p["diagram"] == "architecture"}
    assert ("past its box", "a planted caption far wider than its box") in found, _said(rep)
    assert ("past the edge", "a planted caption-band sentence that runs on past the right edge") in found, _said(rep)
    assert all("planted" in p["what"] for p in rep["problems"]), _said(rep)


def test_the_layout_check_catches_a_line_through_an_arrowhead(tmp_path):
    """The control of the kind "line crosses an arrowhead" (the independent review of Oct 15, 2026, S3: the persistent
    memory -> replay line crossed two arrows into self perception 3 units above their arrowheads' bases). Planted under
    the architecture diagram: a line ending in a downward arrow, a second line crossing it 3 units above its tip, and a
    third crossing a second arrow 20 units above its tip. The first crossing must be reported and the second not, and
    nothing else on the page may be."""
    CK, exe = _browser_or_skip()
    good = _fixture_page(tmp_path).read_text(encoding="utf-8")
    tail = "</script>\n</body></html>"
    assert good.endswith(tail)
    planted = (';(function () { const drawn = buildDiagram; buildDiagram = function () { drawn();'
               ' const svg = document.querySelector("#diagram svg"), vb = svg.viewBox.baseVal;'
               ' const W = vb.width, H = vb.height;'
               ' svg.setAttribute("viewBox", "0 0 " + W + " " + (H + 70));'
               ' const line = (a, b, d) => svg.appendChild(el("path", { class: "dg-edge", d, "data-a": a, "data-b": b }));'
               ' line("planted_src", "planted_box", "M200 " + (H + 10) + " L200 " + (H + 60));'
               ' line("planted_cross", "planted_far", "M180 " + (H + 57) + " L240 " + (H + 57));'
               ' line("planted_src2", "planted_box2", "M400 " + (H + 10) + " L400 " + (H + 60));'
               ' line("planted_clear", "planted_far2", "M380 " + (H + 40) + " L440 " + (H + 40));'
               ' }; })();')
    page = tmp_path / "planted_arrowhead.html"
    page.write_text(good[:-len(tail)] + planted + tail, encoding="utf-8", newline="")
    rep = CK.check_layout(str(page), exe)
    assert rep["ran"], rep.get("why")
    # the check also redraws the diagram with every switch on and every switch off, and the planted lines are drawn
    # there too: each of the three drawings reports the one planted crossing and nothing else
    seen = [(p["diagram"], p["kind"], p["what"], p["with"].split(" (")[0]) for p in rep["problems"]]
    assert sorted(seen) == sorted(
        (d, "line crosses an arrowhead", "planted_cross -> planted_far", "planted_src -> planted_box")
        for d in ("architecture", "architecture, every switch on", "architecture, every switch off")), _said(rep)


def test_the_architecture_diagram_lays_its_text_out_from_measured_widths():
    """Source-level: the box text of the architecture diagram is placed from measured widths (node()
    wraps a sub-caption with boxSubLines and grows the box with boxHeight, which counts the box's extra
    lines too: a repair path built inside the box and the attention cap), the in-box title, sub-caption
    and count keep their declared sizes (a class and an element in ".dg-node text" outranked ".sublabel"
    and ".paramtag", so both were drawn at 11px), the caption band is wrapped with wrapRuns, and the
    layout check measures the drawing and the learning circle at one unit per pixel. The sizes are one
    step larger since Oct 7, 2026 (the decided design of the Architecture tab, D-E6): titles 12, sub-
    captions and the drawing's captions 10, corner counts 9 units, where this test asserted 9 and 8."""
    js = open(os.path.join(BX.PKG, "viewer.js"), encoding="utf-8").read()
    css = open(os.path.join(BX.PKG, "viewer.css"), encoding="utf-8").read()
    diagram = js[js.index("function buildDiagram()"):js.index("function buildLearnLoop()")]
    node = diagram[diagram.index("  function node(id, x, y, label, sub, w, h) {"):]
    node = node[:node.index("\n  }\n")]
    assert "boxSubLines(sub, ww)" in node and "boxHeight(sub, ww, h, extra.length)" in node
    assert "x + ww - BOX_PAD - 1" in node, "the count keeps BOX_PAD units inside the box's right edge"
    assert "wrapRuns(runs, SUB_PX, W - 32)" in diagram, "the caption band is wrapped to the drawing's width"
    assert re.search(r"const SUB_PX = 10, SUB_STEP = 14;", js), "the sub-caption's size and line step"
    assert 'svg.setAttribute("viewBox", `0 0 ${W} ${fy - 7}`)' in diagram
    assert "capY(" not in diagram, "a caption line placed on a fixed line again"
    assert re.search(r"\.dg-node text\s*\{\s*font-size:\s*12px;", css)
    assert re.search(r"\.dg-node text\.sublabel\s*\{\s*font-size:\s*10px;", css)
    assert re.search(r"\.dg-node text\.paramtag\s*\{\s*font-size:\s*9px;", css)
    assert re.search(r"#diagram \.sublabel\s*\{\s*font-size:\s*10px;", css)
    check = js[js.index("function layoutCheck()"):js.index("// ------------------------------------------------------------------ live reload")]
    assert 'atFullSize("architecture", arch)' in check and 'atFullSize("learning circle", loop)' in check
    assert '"width:" + svg.viewBox.baseVal.width + "px' in check


def test_a_box_with_a_diagram_opens_one_place_and_it_carries_the_whole_entry(tmp_path):
    """The exerciser clicks every drawn box: a box with a detail diagram must leave no popup open and
    must draw the entry's text (a dl) above its diagram; every other box opens its popup. The planner's
    panel also carries the 20 updates' own record for the newest snapshot that has one."""
    CK, exe = _browser_or_skip()
    rep = CK.check_page(str(_fixture_page(tmp_path)), exe)
    assert rep["ran"], rep.get("why")
    assert rep["detail_boxes"] >= 10, rep["detail_boxes"]
    assert rep["detail_box_popups"] == [] and rep["detail_box_without_text"] == []
    assert not rep["problems"], "\n".join(rep["problems"])


def test_the_detail_panel_shows_the_popups_text_and_no_nested_planner_drawing():
    """Source-level: nodeClicked sends a box with a diagram to renderDetail only; renderDetail draws the
    four entry fields, the run note and the live values before the diagram; the popup no longer draws
    the planner diagram inside itself."""
    js = open(os.path.join(BX.PKG, "viewer.js"), encoding="utf-8").read()
    start = js.index("function nodeClicked(")
    clicked = js[start:js.index("\n}\n", start)]
    assert "if (DETAIL[id]) {" in clicked and "renderDetail(id);" in clicked
    assert clicked.index("return;") < clicked.index("showDoc(docKeyFor(id)")
    detail = js[js.index("function renderDetail(id)"):js.index("function liveValuesFor(")]
    for part in ("docFieldList(d)", "docRunNote(d)", "liveValuesFor(id).concat(livenessRows(id))",
                 "DETAIL[id]()"):
        assert part in detail, part
    assert detail.index("liveValuesFor(id)") < detail.index("DETAIL[id]()")
    panel = js[js.index("function renderDocPanel()"):js.index("// ------------------------------------------------ text measured")]
    assert "intentionMini" not in panel


def test_the_old_planner_keys_open_their_renamed_entries():
    """term.ring-moment and term.candidate-order carried a word the canonical names replace (the style guide
    "ONE WORD PER THING"), and since 2026-10-03 so do term.moment, term.executed-order and
    term.shuffled-order-control. Their entries live under the new keys; viewer.js DOC_ALIAS opens them from a
    link or a saved #doc= address written with an old key."""
    doc = TB._page_doc()
    js = open(os.path.join(BX.PKG, "viewer.js"), encoding="utf-8").read()
    alias = dict(re.findall(r"\[`(term\.[a-z-]+)`\]:\s*\"(term\.[a-z-]+)\"", js))
    assert alias == {"term.ring-moment": "term.goal-ring-entry",
                     "term.candidate-order": "term.candidate-command",
                     "term.moment": "term.model-step",
                     "term.executed-order": "term.executed-command",
                     "term.shuffled-order-control": "term.shuffled-command-control"}
    for old, new in alias.items():
        assert old not in doc and new in doc, (old, new)
    assert "key = docKey(key);" in js[js.index("function showDoc("):js.index("function docFieldList(")]
    flat = json.dumps(doc)
    assert "[[ring-moment" not in flat and "[[candidate-order" not in flat
    for old_link in ("[[moment", "[[executed-order", "[[shuffled-order-control"):
        assert old_link not in flat, old_link
    # the model step's entry gives the code's own name once, in backticks, and no entry says the old word
    assert doc["term.model-step"]["t"] == "model step" and "`step`" in doc["term.model-step"]["w"]
    assert doc["term.executed-command"]["t"] == "executed command"
    assert "`executed_action`" in doc["term.executed-command"]["w"]
