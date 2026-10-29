# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Open a BUILT explainer page in a headless browser and report what broke.

    python -m mimo_bridge.tools.check_explainer_page PAGE.html [--browser PATH] [--json] [--layout]

--layout runs the page's own layout check instead (check_layout below): the architecture diagram's,
every detail diagram's and the learning circle's texts against their boxes, each other, the lines and
the drawing's edge.

Why this exists: viewer.js is about 3,100 lines that no test ever EXECUTED. esprima proves
the file parses; it cannot see a ReferenceError, a TypeError on a run's real data, or a
field the model renamed (H84's plan row printed the word "undefined" on all 14 snapshots
and nothing raised). node is not installed on this machine; a Chromium-family browser is
(Microsoft Edge ships with Windows 11), and its headless mode needs no display.

What it does, on a COPY of the page in a temp folder - the page itself is never touched:
  1. injects an error listener ahead of the page's own script (window 'error' and
     'unhandledrejection'), so an uncaught error anywhere in viewer.js is recorded;
  2. injects an exerciser after it that, once the page has loaded, switches to each of
     the page's tabs (the list the page itself declares, TABS), opens every DOC popup, clicks
     every drawn box (a popup with live values, or - for a box with a sub-diagram - the panel
     under the picture with the whole entry, the live values and the diagram, and no popup),
     renders every sub-diagram, every tour slide and every embedded
     snapshot in the inspector - each inside try/catch - and scans the text it produced
     for "undefined", "[object Object]", a bare NaN and unfilled {placeholders}; it also
     compares the popup entries the page's script holds with the map the build embedded,
     since an entry the script adds or rewrites at load is read by no check of that map;
  3. writes the findings as JSON into <pre id="explainer-check-report"> and lets
     --dump-dom hand the finished DOM back.
The browser is started with --headless=new --disable-gpu: the GPU belongs to the training
run. It gets its own throwaway profile directory, so it shares nothing with a browser the
operator has open.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

BROWSER_ENV = "EXPLAINER_BROWSER"
_WINDOWS_PATHS = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)
_ON_PATH = ("msedge", "microsoft-edge", "google-chrome", "chrome", "chromium",
            "chromium-browser")
REPORT_ID = "explainer-check-report"

ERROR_LISTENER = """<script>
window.__explainerErrors = [];
window.addEventListener("error", function (e) {
  window.__explainerErrors.push({kind: "uncaught error", message: String(e.message),
                                 line: e.lineno, column: e.colno});
});
window.addEventListener("unhandledrejection", function (e) {
  window.__explainerErrors.push({kind: "unhandled rejection", message: String(e.reason)});
});
</script>"""

EXERCISER = """<script>
window.addEventListener("load", function () {
  var R = {errors: window.__explainerErrors, stray: [], booted: true};
  function attempt(where, fn) {
    try { fn(); } catch (e) {
      R.errors.push({kind: "exception", where: where, message: String((e && e.message) || e),
                     stack: String((e && e.stack) || "").slice(0, 700)});
    }
  }
  // text the page should never show a reader
  var STRAY = [["undefined", /\\bundefined\\b/], ["[object Object]", /\\[object Object\\]/],
               ["NaN", /\\bNaN\\b/], ["unfilled placeholder", /\\{[a-z][a-z0-9_]*\\}/]];
  // ONE TEXT NODE AT A TIME, never an element's whole textContent: the inspector draws
  // <b>plan sent to the body</b><span>undefined - ...</span>, textContent joins that into
  // "bodyundefined", and a word-boundary match then finds nothing. Measured 2026-09-19 on
  // H84's page with the viewer.js of commit 483347e77 spliced in: the joined scan reported
  // 0 problems and this one 8 - the row at boot and on each of the 7 embedded snapshots.
  var seen = {};
  // DESIGN.md section 11 (operator 2026-09-22): no bare score decimal anywhere a person reads.
  // Every text node the exercise renders is HANDED BACK (R.texts, one entry per place and text)
  // and Python scans it with tools/check_no_bare_decimals.readout_problems (bare_decimals and, since
  // 2026-09-23, the scanner's three rules for numbers of any size). An earlier version scanned here in
  // JavaScript with a looser unit list of its own ("0.25 of every 100" passed), so the page and
  // the rest of the build were held to two different rules.
  R.texts = [];
  var textSeen = {};
  // the panel a text node sits in (its h3's own text), so a count names the panel that shows it
  function panelOf(node) {
    var p = node && node.parentNode && node.parentNode.closest ? node.parentNode.closest(".panel") : null;
    var h = p ? p.querySelector("h3") : null;
    return h && h.firstChild && h.firstChild.nodeValue ? " / " + h.firstChild.nodeValue.slice(0, 70) : "";
  }
  function keepText(where, text) {
    if (!/\\d/.test(text)) return;          // a text with no digit can hold no decimal
    var key = where + "|" + text;
    if (textSeen[key]) return;
    textSeen[key] = true;
    R.texts.push([where, text]);
  }
  function scan(where, node) {
    if (!node) return;
    var walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT, null), t;
    while ((t = walker.nextNode())) {
      var tag = t.parentNode ? t.parentNode.nodeName : "";
      if (tag === "SCRIPT" || tag === "STYLE") continue;
      var text = t.nodeValue || "";
      keepText(where + panelOf(t), text);
      for (var i = 0; i < STRAY.length; i++) {
        var m = STRAY[i][1].exec(text);
        if (!m) continue;
        var context = text.slice(Math.max(0, m.index - 70), m.index + 60);
        var key = STRAY[i][0] + "|" + where + "|" + context;
        if (seen[key] || R.stray.length >= 60) continue;
        seen[key] = true;
        R.stray.push({what: STRAY[i][0], where: where, context: context});
      }
    }
  }
  var byId = function (id) { return document.getElementById(id); };
  // 1. what the boot sequence rendered, before anything is clicked
  // THE PAGE'S OWN TAB LIST (TABS in viewer.js), never a typed one: the probes tab of
  // 2026-09-22 would otherwise go unchecked
  var tabIds = [];
  attempt("read the page's tab list", function () {
    tabIds = TABS.map(function (t) { return t[0]; });
  });
  R.tabs = tabIds.map(function (t) {
    return {id: t, button: !!byId("btn-" + t), section: !!byId("tab-" + t)};
  });
  R.diagram_boxes = document.querySelectorAll("#diagram .dg-node").length;
  R.diagram_lines = document.querySelectorAll("#diagram .dg-edge").length;
  R.learning_loop_drawn = document.querySelectorAll("#learnloop svg").length;
  R.inspector_panels = document.querySelectorAll("#inspector .panel").length;
  R.run_charts = document.querySelectorAll("#runtab .chart-block").length;
  R.run_tiles = document.querySelectorAll("#runtab .tile").length;
  R.probe_panels = document.querySelectorAll("#probestab .panel").length;
  R.probes_tab_drawn = !!(byId("probestab") && byId("probestab").childNodes.length);
  attempt("read the page's own data", function () {
    R.snapshots = SNAPS.length; R.tour_slides = TOUR.length;
    R.doc_entries = Object.keys(DOC).length; R.run = META.run;
    // a page built without a model receipt draws only what the launch flags name: it must say so
    // where the diagram starts (review of 2026-09-25, finding 14)
    R.receipt_missing = !(META.receipt && Object.keys(META.receipt).length);
    var note = byId("receipt-note");
    R.receipt_note = note ? note.textContent : null;
  });
  // THE POPUPS A READER OPENS ARE THE MAP THE BUILD EMBEDDED. The suite checks every popup's text
  // through that map (schema.py DOC and build_explainer.py DOC_CHARTS); an entry the page's own
  // script adds, rewrites or deletes at load is shown to a reader and read by no check, and a
  // rewrite does not even change the count of entries. Found 2026-09-25: viewer.js rewrote
  // node.efference and added edge.efference.heads at load (commit aebef83c1).
  // ...and AGAIN AT THE END, after every tab, popup, box and tour slide has been opened: a popup
  // rewritten when it opens (in renderDocPanel, say) passed the load-time comparison alone (review of
  // 2026-09-25, finding 6)
  function docDiff() {
    var shipped = JSON.parse(byId("data-doc").textContent);
    var own = Object.prototype.hasOwnProperty, out = [];
    Object.keys(DOC).forEach(function (k) {
      if (!own.call(shipped, k)) out.push(k + " added");
      else if (JSON.stringify(DOC[k]) !== JSON.stringify(shipped[k])) out.push(k + " rewritten");
    });
    Object.keys(shipped).forEach(function (k) {
      if (!own.call(DOC, k)) out.push(k + " removed");
    });
    return out;
  }
  R.doc_changed = [];
  attempt("compare the page's doc map with the embedded one", function () { R.doc_changed = docDiff(); });
  scan("architecture tab", byId("tab-arch"));
  scan("inspector tab", byId("tab-inspect"));
  scan("run tab", byId("tab-run"));
  scan("probes tab", byId("tab-probes"));
  // 2. every tab
  tabIds.forEach(function (t) {
    attempt("switch to tab " + t, function () {
      switchTab(t);
      if (byId("tab-" + t).classList.contains("hidden")) throw new Error("tab stayed hidden");
    });
  });
  // 3. every popup the page ships
  R.popups_opened = 0;
  attempt("list the DOC entries", function () {
    Object.keys(DOC).forEach(function (key) {
      attempt("popup " + key, function () {
        showDoc(key, key);
        if (!byId("docpanel")) throw new Error("no popup was drawn");
        scan("popup " + key, byId("docpanel"));
        R.popups_opened++;
      });
    });
  });
  // 4. every drawn box: its popup with this run's live values - or, for a box with a detail
  // diagram, its ONE place: the panel under the picture with the whole entry, the live values
  // and the diagram, and no popup (operator 2026-10-02: "REDUNDANT POPUP")
  R.boxes_clicked = 0;
  R.detail_boxes = 0;
  R.detail_box_popups = [];
  R.detail_box_without_text = [];
  Array.prototype.forEach.call(document.querySelectorAll("#diagram .dg-node"), function (g) {
    var id = g.id.replace(/^nd-/, "");
    attempt("click box " + id, function () {
      nodeClicked(id, id);
      if (DETAIL[id]) {
        R.detail_boxes++;
        if (byId("docpanel")) R.detail_box_popups.push(id);
        if (!document.querySelector("#detail #detail-text dl")) R.detail_box_without_text.push(id);
        scan("box " + id, byId("detail"));
      } else {
        scan("box " + id, byId("docpanel"));
      }
      R.boxes_clicked++;
    });
  });
  R.sub_diagrams = 0;
  attempt("list the sub-diagrams", function () {
    Object.keys(DETAIL).forEach(function (id) {
      attempt("sub-diagram " + id, function () {
        renderDetail(id);
        scan("sub-diagram " + id, byId("detail"));
        R.sub_diagrams++;
      });
    });
  });
  // 5. every tour slide, each one READ as it is drawn: the tour box holds one slide at a time,
  // and scanning it once after the walk read the last slide only (found 2026-09-23: the
  // earlier slides' text had never been scanned)
  R.tour_rendered = 0;
  attempt("walk the tour", function () {
    for (var i = 0; i < TOUR.length; i++) {
      (function (n) {
        attempt("tour slide " + n, function () {
          tourIdx = n; renderTour(); R.tour_rendered++;
          scan("tour slide " + (TOUR[n] && TOUR[n].id ? TOUR[n].id : n), byId("tour"));
        });
      })(i);
    }
  });
  // 6. every embedded snapshot in the inspector
  R.snapshots_rendered = 0;
  attempt("page through the snapshots", function () {
    SNAPS.forEach(function (s) {
      attempt("inspector at step " + s.header.step, function () {
        state.s = String(s.header.step);
        renderInspector();
        scan("inspector at step " + s.header.step, byId("inspector"));
        R.snapshots_rendered++;
      });
    });
  });
  attempt("re-render the run tab", function () { renderRun(); scan("run tab", byId("runtab")); });
  // WHAT A HOVER SHOWS IS READ TOO: the run tab's readout lists every series at the step under
  // the pointer, and a heatmap prints the cell under the pointer. Neither exists until a pointer
  // moves, so the exercise moves one to the middle of each and reads what appeared.
  function hover(node) {
    var b = node.getBoundingClientRect();
    node.dispatchEvent(new MouseEvent("mousemove", {bubbles: true, clientX: b.left + b.width / 2,
                                                     clientY: b.top + b.height / 2}));
  }
  R.hovered = 0;
  attempt("hover the run tab's charts", function () {
    switchTab("run");
    var svgs = document.querySelectorAll("#runtab svg.chart");
    for (var i = 0; i < svgs.length; i++) {
      hover(svgs[i]);
      scan("run tab readout", document.querySelector("#runtab .readout"));
      R.hovered++;
    }
  });
  attempt("hover the inspector's heatmaps", function () {
    switchTab("inspect");
    SNAPS.forEach(function (s) {
      state.s = String(s.header.step);
      renderInspector();
      var maps = document.querySelectorAll("#inspector canvas.hm");
      for (var i = 0; i < maps.length; i++) {
        hover(maps[i]);
        var tip = maps[i].parentNode && maps[i].parentNode.querySelector(".hmtip");
        if (tip) { scan("heatmap hover at step " + s.header.step, tip); R.hovered++; }
      }
    });
  });
  attempt("re-render the probes tab", function () { renderProbes(); scan("probes tab", byId("probestab")); });
  attempt("compare the doc map again after the whole exercise", function () {
    docDiff().forEach(function (d) {
      if (R.doc_changed.indexOf(d) < 0) R.doc_changed.push(d + " after load");
    });
  });
  var out = document.createElement("pre");
  out.id = "%s";
  out.textContent = JSON.stringify(R);
  document.body.appendChild(out);
});
</script>""" % REPORT_ID


# A LAUNCH SETTING WRITTEN THE WAY THE COMMAND LINE WRITES IT ("--efference-weight 0.1") is the
# literal text the operator types, not a score: rewriting it would misquote the command. It is the
# one form excused from the section 11 scan, and every excused one is counted and listed apart.
SETTING = re.compile(r"(--[a-z][a-z0-9-]*[ =])(-?\d*\.\d+)")


def _where_group(where: str) -> str:
    """The place a text was shown, with the snapshot step and the popup key taken off, so a count
    by place stays short: 'inspector at step 343139 / The order' -> 'inspector / The order'."""
    head, sep, panel_title = where.partition(" / ")
    head = re.sub(r" at step \d+$", "", head)
    head = re.sub(r"^(popup|box|sub-diagram) .*", r"\1s", head)
    return head + (sep + panel_title if sep else "")


def scan_rendered_texts(texts: List[Any]) -> Dict[str, Any]:
    """Every text the exercise rendered, scanned with check_no_bare_decimals.readout_problems (DESIGN
    section 11): a bare score decimal ("0.14"), and since 2026-09-23 (fix-readout-explainer, after
    the extended scanner of fix-scanner) a score of 1 or more with no unit ("error ratio 1.003"), a
    multiplier with nothing named after it ("drift x1.0") and a raw hidden-unit input value ("its
    input rises above -3"). Returns {count, by_where, by_rule, examples (the first 40), hits (every
    one: where, number, rule, context), settings_excused, settings_examples}; the key keeps its
    older name, bare_decimals, in the report. A hit is counted once per place and context, so the
    same sentence drawn on five snapshots counts five times only when it reads differently on each."""
    from mimo_bridge.tools.check_no_bare_decimals import readout_problems
    out: Dict[str, Any] = {"count": 0, "by_where": {}, "by_rule": {}, "examples": [], "hits": [],
                           "settings_excused": 0, "settings_examples": []}
    seen = set()
    for entry in texts:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        where, text = str(entry[0]), str(entry[1])
        for sm in SETTING.finditer(text):
            key = ("setting", sm.group(0))
            if key not in seen:
                seen.add(key)
                out["settings_excused"] += 1
                if len(out["settings_examples"]) < 40:
                    out["settings_examples"].append({"where": where, "setting": sm.group(0)})
        cleaned = SETTING.sub(lambda mm: mm.group(1) + "<setting>", text)
        group = _where_group(where)
        for number, rule, context in readout_problems(cleaned):
            ctx = " ".join(context.split())
            key = (group, ctx, number, rule)
            if key in seen:
                continue
            seen.add(key)
            out["count"] += 1
            out["by_where"][group] = out["by_where"].get(group, 0) + 1
            out["by_rule"][rule] = out["by_rule"].get(rule, 0) + 1
            hit = {"where": where, "number": number, "rule": rule, "context": ctx}
            out["hits"].append(hit)
            if len(out["examples"]) < 40:
                out["examples"].append({"where": where, "rule": rule, "context": ctx})
    return out


def find_browser(explicit: Optional[str] = None) -> Optional[str]:
    """A Chromium-family browser: --browser, then $EXPLAINER_BROWSER, then the places
    Edge and Chrome install to on Windows, then PATH. None when there is none."""
    for cand in (explicit, os.environ.get(BROWSER_ENV)):
        if cand and os.path.exists(cand):
            return cand
    for cand in _WINDOWS_PATHS:
        if os.path.exists(cand):
            return cand
    for name in _ON_PATH:
        hit = shutil.which(name)
        if hit:
            return hit
    return None


def instrument(page: str) -> str:
    """The page with the error listener ahead of its script and the exerciser after it."""
    if "<body>" not in page or "</body>" not in page:
        raise ValueError("not an explainer page: no <body> element")
    page = page.replace("<body>", "<body>" + ERROR_LISTENER, 1)
    head, sep, tail = page.rpartition("</body>")
    return head + EXERCISER + sep + tail


def check_page(page_path: str, browser: Optional[str] = None,
               timeout_s: float = 300.0) -> Dict[str, Any]:
    """Run the check. Always returns a dict:
      ran       False when no browser was found or it produced no report ('why' says which)
      errors    uncaught errors and exceptions, each with where it happened
      stray     text no reader should see ("undefined", "[object Object]", NaN, {placeholder})
      bare_decimals  every rendered text (hovers included) scanned with
                check_no_bare_decimals.readout_problems (DESIGN.md section 11: bare score decimals,
                scores of 1 or more with no unit, multipliers with nothing named after them, raw
                hidden-unit input values): count, by_where, by_rule, the first 40 examples, every
                hit, and the launch settings excused - a measurement, not in 'problems'
      doc_changed  popup entries the page's script added, rewrote or removed after load
      problems  errors, stray text, missing parts and changed popup entries, as plain
                sentences - empty = passed
    plus the counts the exerciser took (tabs, boxes, popups opened, snapshots rendered)."""
    exe = find_browser(browser)
    if not exe:
        return {"ran": False, "why": "no Edge, Chrome or Chromium found (set %s)" % BROWSER_ENV,
                "errors": [], "stray": [], "problems": []}
    src = Path(page_path).read_text(encoding="utf-8")
    work = tempfile.mkdtemp(prefix="explainer_check_")
    try:
        copy = Path(work) / "page_under_check.html"
        copy.write_text(instrument(src), encoding="utf-8", newline="")
        cmd = [exe, "--headless=new", "--disable-gpu", "--no-first-run",
               "--no-default-browser-check", "--disable-extensions",
               "--disable-background-networking", "--disable-sync",
               "--user-data-dir=" + str(Path(work) / "profile"),
               # under 20 s: a --watch page reloads itself after 20 s without input
               "--virtual-time-budget=10000",
               "--enable-logging=stderr", "--v=0",
               "--dump-dom", copy.as_uri()]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return {"ran": False, "why": "the browser did not finish inside %.0f s" % timeout_s,
                    "browser": exe, "errors": [], "stray": [], "problems": []}
        dom = proc.stdout.decode("utf-8", "replace")
        console = [ln.strip() for ln in proc.stderr.decode("utf-8", "replace").splitlines()
                   if "CONSOLE" in ln]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    m = re.search(r'<pre id="%s">(.*?)</pre>' % REPORT_ID, dom, re.S)
    if not m:
        # the exerciser never finished: a syntax error in the page's script stops the
        # script outright, and the listener's list never reaches the DOM - so say what the
        # browser's own console said
        return {"ran": False, "browser": exe, "console": console[:20],
                "why": "the page produced no report - its script did not run to the end "
                       "of the load handler (browser exit code %s)" % proc.returncode,
                "errors": [], "stray": [], "problems": []}
    rep: Dict[str, Any] = json.loads(html.unescape(m.group(1)))
    rep.update(ran=True, browser=exe, console=console[:20],
               page_mb=round(len(src.encode("utf-8")) / 1e6, 2))
    rep["bare_decimals"] = scan_rendered_texts(rep.pop("texts", None) or [])
    problems: List[str] = []
    for e in rep.get("errors") or []:
        problems.append("%s%s: %s" % (e.get("kind"), " in " + e["where"] if e.get("where") else "",
                                      e.get("message")))
    for s in rep.get("stray") or []:
        problems.append("%s shown in %s: ...%s..." % (s["what"], s["where"], s["context"]))
    for t in rep.get("tabs") or []:
        if not (t["button"] and t["section"]):
            problems.append("tab %s was not rendered" % t["id"])
    if not rep.get("tabs"):
        problems.append("the page declares no tab list (TABS) the check could read")
    if not rep.get("diagram_boxes"):
        problems.append("the architecture diagram drew no box")
    if any(t["id"] == "probes" for t in rep.get("tabs") or []) and not rep.get("probes_tab_drawn"):
        problems.append("the probes tab drew nothing")
    if not rep.get("learning_loop_drawn"):
        problems.append("the learning-loop circle was not drawn")
    if rep.get("doc_entries") and rep.get("popups_opened") != rep.get("doc_entries"):
        problems.append("%s of %s popups opened" % (rep.get("popups_opened"), rep.get("doc_entries")))
    if rep.get("receipt_missing") and not rep.get("receipt_note"):
        problems.append("the page has no model receipt and does not say so: the diagram draws only what "
                        "the launch flags name")
    if rep.get("doc_changed"):
        problems.append("the page's script changed %d popup entries after the build embedded them, "
                        "so a reader sees text no check of the doc map reads: %s"
                        % (len(rep["doc_changed"]), ", ".join(rep["doc_changed"][:20])))
    if rep.get("snapshots") and rep.get("snapshots_rendered") != rep.get("snapshots"):
        problems.append("%s of %s snapshots rendered in the inspector"
                        % (rep.get("snapshots_rendered"), rep.get("snapshots")))
    if rep.get("snapshots") and not rep.get("inspector_panels"):
        problems.append("the inspector drew no panel although the page embeds snapshots")
    if rep.get("detail_box_popups"):
        problems.append("%d boxes with a detail diagram opened a popup as well as their panel: %s"
                        % (len(rep["detail_box_popups"]), ", ".join(rep["detail_box_popups"])))
    if rep.get("detail_box_without_text"):
        problems.append("%d detail panels drew no entry text above their diagram: %s"
                        % (len(rep["detail_box_without_text"]), ", ".join(rep["detail_box_without_text"])))
    rep["problems"] = problems
    return rep


LAYOUT_ID = "layout-check"


def check_layout(page_path: str, browser: Optional[str] = None,
                 timeout_s: float = 300.0) -> Dict[str, Any]:
    """The page's own layout check (viewer.js layoutCheck, opened with #layoutcheck in the address),
    read from a headless browser's dump of the page. It measures the architecture diagram of tab 1
    (at one unit per pixel), every detail diagram and the learning circle, every text and box with
    getBBox in the drawing's own units, and lists each problem: a box's own text closer than 3 units
    to its box's left or right edge or 1 unit to its top or bottom (or outside it), any other text
    over a box, a text past the drawing's edge, two texts (a numbered circle counts as one) or two
    boxes overlapping, a text or box across a block frame, a line through a label or through a box
    (the style guide "DIAGRAM LAYOUT ON A GRID"). The page is opened from a COPY in a temp folder, with a
    throwaway profile. Returns {ran, why?, diagrams, problems, counts, total, browser}; 'problems' is
    empty on a page whose diagrams pass."""
    exe = find_browser(browser)
    if not exe:
        return {"ran": False, "why": "no Edge, Chrome or Chromium found (set %s)" % BROWSER_ENV,
                "problems": [], "diagrams": []}
    work = tempfile.mkdtemp(prefix="explainer_layout_")
    try:
        copy = Path(work) / "page_under_layout_check.html"
        shutil.copyfile(page_path, copy)
        cmd = [exe, "--headless=new", "--disable-gpu", "--no-first-run",
               "--no-default-browser-check", "--disable-extensions",
               "--disable-background-networking", "--disable-sync",
               "--user-data-dir=" + str(Path(work) / "profile"),
               "--virtual-time-budget=10000", "--dump-dom", copy.as_uri() + "#layoutcheck"]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return {"ran": False, "why": "the browser did not finish inside %.0f s" % timeout_s,
                    "browser": exe, "problems": [], "diagrams": []}
        dom = proc.stdout.decode("utf-8", "replace")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    # the LAST such element: the page's own script text is dumped too and must never be read as one
    found = re.findall(r'<pre id="%s">(.*?)</pre>' % LAYOUT_ID, dom, re.S)
    if not found:
        return {"ran": False, "browser": exe, "problems": [], "diagrams": [],
                "why": "the page wrote no layout report - its script did not reach the end of its "
                       "boot handler (browser exit code %s)" % proc.returncode}
    rep: Dict[str, Any] = json.loads(html.unescape(found[-1]))
    rep.update(ran=True, browser=exe)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("page", help="a built explainer.html")
    ap.add_argument("--browser", default=None, help="path to msedge / chrome / chromium")
    ap.add_argument("--json", action="store_true", help="print the whole report as JSON")
    ap.add_argument("--layout", action="store_true",
                    help="run the page's layout check (#layoutcheck) instead: the architecture "
                         "diagram's, every detail diagram's and the learning circle's texts against "
                         "their boxes, each other, the lines and the drawing's edge")
    a = ap.parse_args(argv)
    if a.layout:
        rep = check_layout(a.page, a.browser)
        if a.json:
            print(json.dumps(rep, indent=1))
        elif not rep["ran"]:
            print("NOT RUN: " + rep["why"])
        else:
            for d in rep.get("diagrams") or []:
                print("  %-18s %4s x %-4s %3d boxes %3d texts %3d problems"
                      % (d["name"], d["width"], d["height"], d["boxes"], d["texts"], d["problems"]))
            for p in rep.get("problems") or []:
                print("  - %s: %s: '%s' / '%s'" % (p["diagram"], p["kind"], p["what"], p["with"]))
            print("  problems             %s" % rep.get("total", 0))
        return 0 if rep["ran"] and not rep.get("problems") else (2 if not rep["ran"] else 1)
    rep = check_page(a.page, a.browser)
    if a.json:
        print(json.dumps(rep, indent=1))
    elif not rep["ran"]:
        print("NOT RUN: " + rep["why"])
        for ln in rep.get("console") or []:
            print("  console: " + ln)
    else:
        print("page %s (%.2f MB) in %s" % (rep.get("run"), rep["page_mb"],
                                           os.path.basename(rep["browser"])))
        print("  tabs rendered        %s of %s" % (sum(1 for t in rep["tabs"] if t["button"] and t["section"]),
                                                  len(rep["tabs"])))
        print("  diagram              %s boxes, %s lines" % (rep.get("diagram_boxes"), rep.get("diagram_lines")))
        print("  run tab              %s charts, %s tiles" % (rep.get("run_charts"), rep.get("run_tiles")))
        print("  probes tab           %s checkpoint panels" % rep.get("probe_panels"))
        print("  inspector panels     %s" % rep.get("inspector_panels"))
        print("  popups opened        %s of %s" % (rep.get("popups_opened"), rep.get("doc_entries")))
        print("  boxes clicked        %s" % rep.get("boxes_clicked"))
        print("  sub-diagrams drawn   %s" % rep.get("sub_diagrams"))
        print("  tour slides          %s of %s" % (rep.get("tour_rendered"), rep.get("tour_slides")))
        print("  snapshots rendered   %s of %s" % (rep.get("snapshots_rendered"), rep.get("snapshots")))
        bd = rep.get("bare_decimals") or {}
        print("  unplaceable numbers  %s shown (DESIGN.md section 11: counts, real units or a "
              "percentage between named ends; scanned with check_no_bare_decimals.readout_problems)"
              % bd.get("count", "not measured"))
        for rule, n in sorted((bd.get("by_rule") or {}).items(), key=lambda kv: -kv[1]):
            print("    %5d  %s" % (n, rule))
        for where, n in sorted((bd.get("by_where") or {}).items(), key=lambda kv: -kv[1]):
            print("    %5d  %s" % (n, where))
        print("  launch settings      %s written as the command line writes them (--flag 0.1), "
              "excused from the scan" % bd.get("settings_excused", 0))
        print("  hovers read          %s (the run tab's readout, the heatmaps' cells)" % rep.get("hovered"))
        print("  problems             %s" % len(rep["problems"]))
        for p in rep["problems"]:
            print("    - " + p)
    return 0 if rep["ran"] and not rep["problems"] else (2 if not rep["ran"] else 1)


if __name__ == "__main__":
    sys.exit(main())
