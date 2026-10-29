/* Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved. */
/* Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file. */
/* THE PROCESS-OF-ELIMINATION DIAGRAM of a root-cause investigation (the style guide, "EVERY ROOT-CAUSE INVESTIGATION HAS A
   LIVE PROCESS-OF-ELIMINATION DIAGRAM", the operator's rule of Oct 5, 2026). One renderer for both places the rule
   names: the dashboard's Tests page (development_dashboard.js) and the Architecture tab of the explainer
   (explainer/viewer.js; tools/build_explainer.py puts this file into the page).
   INPUT     one investigation of GET /api/investigations (tools/elimination.py view): the server computes every verdict,
             time, commit, evidence value, branch and sentence from the test registry's rows; this file prints them and
             adds only the fixed labels of the boxes.
   OPERATION one column of nodes on a grid: the input, each stage of the path in its declared sequence, and the box
             "Located by elimination" after the stage the fault is located at. Between two nodes an arrow row of its own
             holds the branch the upper stage's verdict takes: the arrow sits in the rail column, its label in the
             content column, so no label lies on a box and none runs past the drawing. A stage box is a grid of a
             verdict rail and its content; its six fields (test, date and time, commit, evidence, Architecture link,
             next unresolved branch) are a grid of label and value columns, and a field the server could not fill
             prints "Missing:" with the reason. At 700 px or less the rail sits above the content and the fields take
             one column. Under the stage's name and question its CURRENT VERDICT is stated in words, and directly
             under it every older measurement of the stage stays in view as a block of its own, never inside a
             closed details element (the operator's rule of Oct 5, 2026): "HISTORICAL — not qualifying under current
             runner" with its date and time (PDT), its runner and that runner's version, the checkpoint and the
             result. Once a run of the current runner decides the box, that run is the box's evidence and the older
             measurements stay there as history.
   OUTPUT    DOM nodes inside the container; nothing is fetched or written here.
   opts      href(componentId) -> the address of that component in the Architecture tab; onComponent(id, event)
             optional, called on a click of that link; onRow(rowId) optional, a button beside a row id that opens the
             row's card; rowHref(rowId) optional, a link for a row id where no button applies; sourceText, a sentence
             saying where and when the data was read. */
(function (root) {
  "use strict";
  var LABELS = { test: "Test", time: "Date and time", commit: "Commit", evidence: "Evidence",
                 architecture: "Architecture", next_unresolved: "Next unresolved branch" };
  var FIELD_ORDER = ["test", "time", "commit", "evidence", "architecture", "next_unresolved"];

  function h(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null && text !== "") n.textContent = String(text);
    return n;
  }
  function code(text) { return h("code", "", text); }
  function verdictClass(v) { return v === "PASS" ? "pass" : v === "FAIL" ? "fail" : "unknown"; }

  function componentLink(arch, opts) {
    var id = (arch && arch.component) || "";
    if (!arch || arch.known === false) {
      var s = h("span", "elim-component unknown", id || "no component named");
      s.title = id + " is not an id of the component manifest";
      return s;
    }
    var a = h("a", "elim-component");
    a.href = opts.href ? opts.href(id) : (arch.href || "#");
    a.dataset.component = id;
    a.appendChild(h("span", "elim-component-id", id));
    if (arch.name) a.appendChild(h("span", "elim-component-name", arch.name));
    a.title = "Opens " + id + " in the Architecture tab";
    if (opts.onComponent) a.addEventListener("click", function (e) { opts.onComponent(id, e); });
    return a;
  }

  function rowRef(row, opts) {
    var span = h("span", "elim-row");
    span.appendChild(code(row));
    if (opts.onRow) {
      var b = h("button", "elim-row-button", "Show its card");
      b.type = "button";
      b.addEventListener("click", function () { opts.onRow(row); });
      span.appendChild(document.createTextNode(" "));
      span.appendChild(b);
    } else if (opts.rowHref) {
      var a = h("a", "elim-row-link", "its card on the Tests page");
      a.href = opts.rowHref(row);
      span.appendChild(document.createTextNode(" "));
      span.appendChild(a);
    }
    return span;
  }

  function missing(text) {
    var d = h("span", "elim-missing");
    d.appendChild(h("strong", "", "Missing: "));
    d.appendChild(document.createTextNode(text));
    return d;
  }

  // THE TWO JUDGEMENTS OF EVERY NUMBER (the directive's section 7A; the dashboard review of Oct 5, 2026, finding F12):
  // the row's own statistical line and its material line for gain of function, or "GOF threshold not yet defined"
  function judgements(s) {
    var wrap = h("div", "elim-judgements");
    var a = h("p", "elim-judgement");
    a.appendChild(h("strong", "", "Statistically clear? "));
    a.appendChild(document.createTextNode(s.statistical_line || "no statistical line is stated"));
    var b = h("p", "elim-judgement");
    b.appendChild(h("strong", "", "Large enough to matter for gain of function? "));
    b.appendChild(document.createTextNode(s.gof_line || "GOF threshold not yet defined"));
    wrap.appendChild(a);
    wrap.appendChild(b);
    return wrap;
  }

  // why a reading decides nothing: a runner record of the older kind, a file that is no registry row, or a value typed
  // into the row (the dashboard review of Oct 5, 2026, finding B2)
  function notDeciding(s) {
    if (!s.undecidable) return "";
    return (s.typed ? "typed into the row, so it decides nothing: " : "decides nothing: ") + s.undecidable;
  }

  // what was read, where and how: a source of the stage (deciding, history or related)
  function sourceLines(s, opts, decided) {
    var wrap = h("div", "elim-source-lines");
    var head = h("p", "elim-source-head");
    if (s.row) head.appendChild(rowRef(s.row, opts));
    else head.appendChild(code(s.file || "a file"));
    var when = s.time_text ? s.time_text : "no time on record";
    var state = s.state === "decided"
      ? (s.group === "related" ? "its own line " + (s.met ? "met" : "not met")
        : (decided ? "decides this stage: " : "reads ") + s.verdict + (s.registry ? "" : " (a file that is no registry row: history only)"))
      : s.state === "pending" ? "pending" : s.state === "reading" ? "a reading with no line for this question"
      : s.state === "missing" ? "not found" : "no verdict";
    head.appendChild(document.createTextNode(" - " + when + " - " + state));
    wrap.appendChild(head);
    if (s.registry && s.undecidable && (s.state === "decided" || s.state === "reading"))
      wrap.appendChild(h("p", "elim-source-why" + (s.typed ? " typed" : ""), notDeciding(s)));
    if (s.label) wrap.appendChild(h("p", "elim-source-label", s.label));
    if (s.value_text !== null && s.value_text !== undefined) {
      var v = h("p", "elim-source-value");
      v.appendChild(h("strong", "", s.value_text));
      wrap.appendChild(v);
    }
    if (s.why && s.state !== "decided") wrap.appendChild(h("p", "elim-source-why", s.why));
    for (var i = 0; i < (s.shown || []).length; i++) {
      var sh = s.shown[i], p = h("p", "elim-source-shown");
      p.appendChild(document.createTextNode((sh.label || sh.key_text) + ": "));
      p.appendChild(h("strong", "", sh.value_text !== null && sh.value_text !== undefined ? sh.value_text : (sh.why || "not read")));
      // a value typed into the row, or written into the file after the run, is marked where it is printed; a record of
      // the older kind is said once, in the source's own line above
      if (sh.not_the_runners && /^(typed into the row|written into the file after)/.test(sh.not_the_runners))
        p.appendChild(document.createTextNode(" (" + notDeciding({ undecidable: sh.not_the_runners, typed: true }) + ")"));
      wrap.appendChild(p);
    }
    if ((s.value_text !== null && s.value_text !== undefined) || (s.shown || []).length) wrap.appendChild(judgements(s));
    if (s.means) wrap.appendChild(h("p", "elim-source-means", s.means));
    var where = [];
    if (s.read_from) where.push("read from " + s.read_from);
    if (s.key_text) where.push("at " + s.key_text);
    if (s.time_basis) where.push("its time: " + s.time_basis);
    if (s.checkpoint) where.push("checkpoint " + s.checkpoint);
    if (s.commit) where.push("commit " + s.commit + (s.commit_basis ? " (" + s.commit_basis + ")" : ""));
    else if (s.commit_missing) where.push("commit " + s.commit_missing);
    if (s.provenance) where.push(s.provenance);
    if (where.length) wrap.appendChild(h("p", "elim-source-where", where.join("; ") + "."));
    return wrap;
  }

  // THE OLDER MEASUREMENT STAYS IN VIEW (the operator's rule of Oct 5, 2026): each source of the stage that does not
  // decide the box is printed as a visible block directly under the current verdict, with the facts a reader needs to
  // judge it; a run of the current runner that a newer run replaced is said to be that
  var HISTORICAL = "HISTORICAL \u2014 not qualifying under current runner";
  function historyHead(s) {
    if (s.decides) return "EARLIER \u2014 a run of the current runner; a newer run of it decides this box";
    if ((s.state === "decided" || s.state === "reading") && (s.undecidable || !s.registry)) return HISTORICAL;
    if (s.state === "reading") return "READING \u2014 a run of the current runner, with no line for this question";
    return "NOT MEASURED \u2014 " + (s.state === "pending" ? "pending" : s.state === "missing" ? "not found" : "no verdict");
  }
  function fact(label, text) {
    var p = h("p", "elim-historical-fact");
    p.appendChild(h("strong", "", label + ": "));
    p.appendChild(document.createTextNode(text));
    return p;
  }
  function historicalBlock(s, opts) {
    var d = h("div", "elim-historical");
    d.dataset.source = s.row || s.file || "";
    d.appendChild(h("p", "elim-historical-head", historyHead(s)));
    var src = h("p", "elim-historical-source");
    if (s.row) src.appendChild(rowRef(s.row, opts));
    else src.appendChild(code(s.file || "a file"));
    if (s.label) src.appendChild(document.createTextNode(" - " + s.label));
    d.appendChild(src);
    d.appendChild(fact("Date and time", s.time_text ? s.time_text + (s.time_basis ? " (" + s.time_basis + ")" : "") : "not on record"));
    d.appendChild(fact("Runner", s.registry ? (s.runner || "no run record") + "; its version: " + (s.runner_version || "not on record")
      : "none: a file that is no registry row, written outside the test runner"));
    d.appendChild(fact("Checkpoint", s.checkpoint ? s.checkpoint + (s.checkpoint_basis ? " (" + s.checkpoint_basis + ")" : "") : "not on record"));
    d.appendChild(fact("Result", s.state === "decided" ? s.verdict + (s.value_text ? ": " + s.value_text : "")
      : s.state === "reading" ? "a reading with no line for this question: " + s.value_text
      : (s.why || s.state)));
    if (s.undecidable) d.appendChild(h("p", "elim-historical-why" + (s.typed ? " typed" : ""), "Why it does not decide: " + notDeciding(s)));
    for (var i = 0; i < (s.shown || []).length; i++) {
      var sh = s.shown[i], p = h("p", "elim-source-shown");
      p.appendChild(document.createTextNode((sh.label || sh.key_text) + ": "));
      p.appendChild(h("strong", "", sh.value_text !== null && sh.value_text !== undefined ? sh.value_text : (sh.why || "not read")));
      if (sh.not_the_runners && /^(typed into the row|written into the file after)/.test(sh.not_the_runners))
        p.appendChild(document.createTextNode(" (" + notDeciding({ undecidable: sh.not_the_runners, typed: true }) + ")"));
      d.appendChild(p);
    }
    if ((s.value_text !== null && s.value_text !== undefined) || (s.shown || []).length) d.appendChild(judgements(s));
    if (s.means) d.appendChild(h("p", "elim-source-means", s.means));
    var where = [];
    if (s.read_from) where.push("read from " + s.read_from);
    if (s.key_text) where.push("at " + s.key_text);
    if (s.commit) where.push("commit " + s.commit + (s.commit_basis ? " (" + s.commit_basis + ")" : ""));
    else if (s.commit_missing) where.push("commit " + s.commit_missing);
    if (s.provenance) where.push(s.provenance);
    if (where.length) d.appendChild(h("p", "elim-source-where", where.join("; ") + "."));
    return d;
  }

  function fieldValue(box, key, opts, inv) {
    var dd = h("dd", "elim-field");
    dd.dataset.field = key;
    var miss = (box.missing || {})[key];
    if (key === "test" && box.test) {
      dd.appendChild(rowRef(box.test.row, opts));
      dd.appendChild(h("span", "elim-sub", box.test.provenance + "; " + box.test.run_text + "; " +
        (box.test.checkpoint ? "checkpoint measured: " + box.test.checkpoint : "the runner's record names no checkpoint")));
    } else if (key === "time" && box.time_text) {
      dd.appendChild(document.createTextNode(box.time_text));
    } else if (key === "commit" && box.commit) {
      dd.appendChild(code(box.commit));
    } else if (key === "evidence" && box.evidence) {
      var e = box.evidence;
      var p = h("span", "elim-evidence-value");
      p.appendChild(h("strong", "", e.value_text));
      dd.appendChild(p);
      dd.appendChild(h("span", "elim-sub", e.label));
      dd.appendChild(judgements(e));
      var more = h("details", "elim-how");
      more.appendChild(h("summary", "", "How it was read"));
      if (e.means) more.appendChild(h("p", "", e.means));
      for (var i = 0; i < (e.shown || []).length; i++) {
        var sh = e.shown[i], q = h("p", "");
        q.appendChild(document.createTextNode((sh.label || sh.key_text) + ": "));
        q.appendChild(h("strong", "", sh.value_text !== null && sh.value_text !== undefined ? sh.value_text : (sh.why || "not read")));
        more.appendChild(q);
      }
      var w = [];
      if (e.read_from) w.push("read from " + e.read_from);
      if (e.key_text) w.push("at " + e.key_text);
      if (w.length) more.appendChild(h("p", "elim-source-where", w.join("; ") + "."));
      dd.appendChild(more);
    } else if (key === "architecture" && box.architecture && !(box.architectures || [box.architecture]).some(function (a) { return a.known === false; })) {
      // every component between the previous stage's reading and this one (finding F11), each with its link
      var archs = box.architectures || [box.architecture];
      for (var ai = 0; ai < archs.length; ai++) {
        if (ai) dd.appendChild(document.createTextNode(" "));
        dd.appendChild(componentLink(archs[ai], opts));
      }
      if (archs.length > 1) dd.appendChild(h("span", "elim-sub", "This stage's reading is taken after all " + archs.length + " components, so it cannot tell them apart."));
    } else if (key === "next_unresolved" && box.next_unresolved) {
      // the test that would decide it is printed once, in that stage's own box (and in the summary above the drawing)
      var nu = box.next_unresolved;
      if (nu.itself) {
        dd.appendChild(h("strong", "", "this stage"));
        if (nu.test) dd.appendChild(h("span", "elim-sub", "The test that would decide it: " + nu.test));
      } else {
        dd.appendChild(h("strong", "", nu.name));
        var go = h("button", "elim-row-button", "Show that stage");
        go.type = "button";
        go.addEventListener("click", function () {
          var target = document.getElementById("elim-" + inv.id + "-" + nu.key);
          if (target && target.scrollIntoView) target.scrollIntoView({ block: "start" });
        });
        dd.appendChild(document.createTextNode(" "));
        dd.appendChild(go);
        dd.appendChild(h("span", "elim-sub", "Stage " + (nu.position || "") + " reads UNKNOWN; its box names the test that would decide it."));
      }
    } else {
      dd.classList.add("is-missing");
      dd.appendChild(missing(miss || "not computed"));
    }
    return dd;
  }

  function sourcesBlock(list, summary, cls, opts) {
    var d = h("details", "elim-list " + cls);
    d.appendChild(h("summary", "", summary));
    for (var i = 0; i < list.length; i++) d.appendChild(sourceLines(list[i], opts, false));
    return d;
  }

  function stageBox(box, inv, opts) {
    var v = box.verdict || "UNKNOWN";
    var node = h("article", "elim-node elim-stage verdict-" + verdictClass(v) + (box.on_walk ? "" : " off-walk"));
    node.dataset.stage = box.key;
    node.dataset.verdict = v;
    node.id = "elim-" + inv.id + "-" + box.key;
    var rail = h("div", "elim-rail");
    rail.appendChild(h("span", "elim-verdict", v));
    rail.appendChild(h("span", "elim-pos", "Stage " + box.position));
    node.appendChild(rail);
    var body = h("div", "elim-body");
    var title = h("h4", "elim-name", box.name);
    title.dataset.plain = "";
    body.appendChild(title);
    body.appendChild(h("p", "elim-question", box.question));
    if (!box.on_walk) body.appendChild(h("p", "elim-offwalk", "Not on the walk: with the verdicts as they read now, no branch above leads to this stage."));
    // the current verdict in words, and directly under it every older measurement (the operator's rule of Oct 5, 2026)
    var cur = h("p", "elim-current");
    cur.appendChild(h("strong", "", "Current verdict: " + v));
    cur.appendChild(document.createTextNode(box.test
      ? " - measured by the current test runner: row " + box.test.row + ", " + (box.time_text || "its time not on record") + "."
      : " - no run of the current test runner has measured this stage" + ((box.history || []).length ? "; the older measurements stay below." : ".")));
    body.appendChild(cur);
    var histHost = body;
    for (var hi = 0; hi < (box.history || []).length; hi++) histHost.appendChild(historicalBlock(box.history[hi], opts));
    var dl = h("dl", "elim-fields");
    for (var i = 0; i < FIELD_ORDER.length; i++) {
      var key = FIELD_ORDER[i];
      dl.appendChild(h("dt", "", LABELS[key]));
      dl.appendChild(fieldValue(box, key, opts, inv));
    }
    body.appendChild(dl);
    if ((box.defects || []).length) {
      var dp = h("p", "elim-defects");
      dp.appendChild(h("strong", "", "Defects: "));
      dp.appendChild(document.createTextNode(box.defects.map(function (d) { return d.status_line; }).join("; ")));
      body.appendChild(dp);
    }
    if ((box.related || []).length)
      body.appendChild(sourcesBlock(box.related, "Related (" + box.related.length + "): readings of a neighbouring question; they never decide this stage", "elim-related", opts));
    node.appendChild(body);
    return node;
  }

  function inputBox(inv, opts) {
    var node = h("div", "elim-node elim-input");
    var rail = h("div", "elim-rail");
    rail.appendChild(h("span", "elim-verdict input", "Input"));
    node.appendChild(rail);
    var body = h("div", "elim-body");
    var t = h("h4", "elim-name", inv.input.name);
    t.dataset.plain = "";
    body.appendChild(t);
    body.appendChild(h("p", "elim-question", inv.input.what));
    var p = h("p", "elim-input-component");
    p.appendChild(document.createTextNode("Architecture: "));
    p.appendChild(componentLink(inv.input.architecture, opts));
    body.appendChild(p);
    node.appendChild(body);
    return node;
  }

  function locatedBox(inv) {
    var node = h("div", "elim-node elim-located" + (inv.located.found ? " found" : ""));
    var rail = h("div", "elim-rail");
    rail.appendChild(h("span", "elim-verdict located", "Located"));
    node.appendChild(rail);
    var body = h("div", "elim-body");
    var t = h("h4", "elim-name", "Located by elimination");
    t.dataset.plain = "";
    body.appendChild(t);
    body.appendChild(h("p", "elim-question", inv.located.sentence));
    node.appendChild(body);
    return node;
  }

  function arrow(label, cls, from, to) {
    var row = h("div", "elim-arrow " + (cls || ""));
    row.dataset.from = from || "";
    row.dataset.to = to || "";
    var g = h("span", "elim-arrow-glyph", cls === "stop" ? "■" : "↓");
    g.setAttribute("aria-hidden", "true");
    row.appendChild(g);
    row.appendChild(h("span", "elim-arrow-label", label));
    return row;
  }

  // the drawn sequence: input, the stages, and the located box right after the stage the fault is located at
  function drawn(inv) {
    var out = [{ node: "input" }];
    for (var i = 0; i < inv.stages.length; i++) {
      out.push({ node: "stage", box: inv.stages[i] });
      if (inv.located.found && inv.located.at_or_before === inv.stages[i].key) out.push({ node: "located" });
    }
    if (!inv.located.found) out.push({ node: "located" });
    return out;
  }

  function branchLabel(box, nextKey, byKey) {
    var b = box.branch || {};
    var head = box.verdict + ": " + (b.text || "");
    if (b.next === null || b.next === undefined) return { text: head + " The walk along the branches ends here.", cls: "stop" };
    if (b.next !== nextKey) return { text: head + " Leads to '" + (byKey[b.next] ? byKey[b.next].name : b.next) + "'.", cls: "jump" };
    return { text: head, cls: "" };
  }

  function render(container, inv, opts) {
    opts = opts || {};
    var sec = h("section", "elim");
    sec.dataset.investigation = inv.id;
    sec.id = "investigation-" + inv.id;
    // the dashboard's glossary adds no term buttons here: its "Stage" is another meaning; the diagram's own terms are
    // defined in "The names used on this page" (tools/elimination.py NAMES)
    sec.dataset.plain = "";
    var title = h("h3", "elim-title", inv.title || inv.id);
    title.dataset.plain = "";
    sec.appendChild(title);
    if (inv.refused) {
      var r = h("div", "elim-refused");
      r.appendChild(h("p", "", "This investigation file is refused, so no box is drawn from it: " + (inv.file || inv.id)));
      var ul = h("ul", "");
      for (var i = 0; i < (inv.problems || []).length; i++) ul.appendChild(h("li", "", inv.problems[i]));
      r.appendChild(ul);
      sec.appendChild(r);
      container.appendChild(sec);
      return sec;
    }
    sec.appendChild(h("p", "elim-q-main", inv.question));
    var sum = h("div", "elim-summary");
    var p1 = h("p", "elim-sum-located");
    p1.appendChild(h("strong", "", "Located by elimination: "));
    p1.appendChild(document.createTextNode(inv.located.sentence));
    sum.appendChild(p1);
    var p2 = h("p", "elim-sum-next");
    p2.appendChild(h("strong", "", "Next unresolved branch: "));
    p2.appendChild(document.createTextNode(inv.next_branch ? inv.next_branch.name + ". The test that would decide it: " + (inv.next_branch.test || "not named")
      : "none: the walk along the branches ends at a stage that a test has decided."));
    sum.appendChild(p2);
    var p3 = h("p", "elim-sum-unresolved");
    p3.appendChild(h("strong", "", "Unresolved branches: "));
    p3.appendChild(document.createTextNode((inv.unresolved || []).length
      ? inv.unresolved.map(function (u) { return "'" + u.name + "'" + (u.on_walk ? "" : " (not on the walk)"); }).join(", ") + "."
      : "none: every stage is decided."));
    sum.appendChild(p3);
    sec.appendChild(sum);
    var chain = h("div", "elim-chain");
    chain.setAttribute("role", "group");
    chain.setAttribute("aria-label", "Process-of-elimination diagram: " + (inv.title || inv.id));
    var byKey = {};
    for (var k = 0; k < inv.stages.length; k++) byKey[inv.stages[k].key] = inv.stages[k];
    var order = drawn(inv);
    for (var j = 0; j < order.length; j++) {
      var item = order[j], next = order[j + 1];
      if (item.node === "input") chain.appendChild(inputBox(inv, opts));
      else if (item.node === "located") chain.appendChild(locatedBox(inv));
      else chain.appendChild(stageBox(item.box, inv, opts));
      if (!next) break;
      var nextKey = next.node === "stage" ? next.box.key : null;
      if (item.node === "input") chain.appendChild(arrow("The command enters the path.", "", "input", nextKey));
      else if (item.node === "located") {
        var lb = byKey[inv.located.at_or_before] && byKey[inv.located.at_or_before].branch;
        chain.appendChild(arrow(lb && lb.next === nextKey ? "Its branch leads on to '" + next.box.name + "'."
          : "The walk along the branches does not lead to '" + (next.box ? next.box.name : "") + "'.",
          lb && lb.next === nextKey ? "" : "stop", "located", nextKey));
      } else {
        var target = next.node === "located" ? (item.box.branch || {}).next : nextKey;
        var lab = branchLabel(item.box, target, byKey);
        chain.appendChild(arrow(lab.text, lab.cls, item.box.key, next.node === "located" ? "located" : nextKey));
      }
    }
    sec.appendChild(chain);
    var src = h("p", "elim-source");
    src.appendChild(document.createTextNode((opts.sourceText ? opts.sourceText + " " : "")
      + "Declared in " + (inv.file || inv.id) + " (" + inv.written_at + ", " + inv.written_by + "); every verdict, time, commit and evidence value is computed from the test registry's rows and their result files when this was read."));
    sec.appendChild(src);
    container.appendChild(sec);
    return sec;
  }

  function renderAll(container, view, opts) {
    var list = (view && view.investigations) || [];
    for (var i = 0; i < list.length; i++) render(container, list[i], opts);
    return list.length;
  }

  root.EliminationDiagram = { render: render, renderAll: renderAll, LABELS: LABELS };
})(typeof window !== "undefined" ? window : this);
