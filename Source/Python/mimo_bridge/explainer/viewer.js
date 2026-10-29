/* Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved. */
/* Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file. */
"use strict";
/* explainer viewer - inlined by build_explainer.py. No external libraries: canvas
   heatmaps, hand-rolled SVG charts, hash-preserved state. Data contracts live in
   build_explainer.py (the JSON script blocks) and explainer/schema.py. */

// ------------------------------------------------------------------ data + state
function jload(id, fallback) {
  const n = document.getElementById(id);
  if (!n) return fallback;
  try { return JSON.parse(n.textContent); } catch (e) { return fallback; }
}
const META = jload("data-meta", {});
const SNAPS = jload("data-snapshots", []);
const SERIES = jload("data-series", { charts: [], seams: [], phases: [], snapmarks: [] });
const TOUR = jload("data-tour", []);
const DOC = jload("data-doc", {});
// DOC IS THE BUILD'S MAP AND NOTHING HERE WRITES TO IT. Every popup's text lives in schema.py DOC
// (or build_explainer.py DOC_CHARTS), where the suite checks it; an entry this script added or
// replaced at load was shown to a reader and read by no check. node.efference and
// edge.efference.heads were written here from 2026-09-23 to 2026-09-25 and now live in schema.py;
// tools/check_explainer_page.py fails a page whose script changes the embedded map.
// the run's own flag values, so a number drawn in a sub-diagram is never a constant
// typed into the picture; every use falls back to the constant it replaced
const ARGS = META.args || {};
// ONE source for the model step's length. build_explainer.py publishes META.clock from the
// receipt's MEASURED env step and model step (train_mimo.py records frames[0].dt and
// frames[0].dt x --action-repeat there); nothing below types the env step again. (P406)
const CLOCK = META.clock || {};
const ENV_STEP_S = Number(CLOCK.env_step_s), MOMENT_S = Number(CLOCK.model_step_s);
// ONE source for the model's size. build_explainer.py publishes META.shape from the run's
// own model receipt (schema.model_shape): token width, core block count, context length
// and the sizes derived from them. The size is a launch setting (--d-model,
// --core-blocks, ...), so nothing below types a token width or a block count. (H84 sync)
const SHAPE = META.shape || {};
const D_MODEL = Number(SHAPE.d_model), N_BLOCKS = Number(SHAPE.core_blocks);

const state = {};
(function readHash() {
  const h = location.hash.replace(/^#/, "");
  for (const kv of h.split("&")) {
    const [k, v] = kv.split("=");
    if (k) state[k] = decodeURIComponent(v || "");
  }
})();
// #layoutcheck: after the page has booted, draw every detail diagram and measure it (layoutCheck,
// near the end of this file). Read here, before the first writeHash rewrites the address.
const LAYOUT_CHECK = /(^|&)layoutcheck(=|&|$)/.test(location.hash.replace(/^#/, ""));
function writeHash() {
  const parts = [];
  for (const k of Object.keys(state)) {
    if (state[k] !== undefined && state[k] !== "") parts.push(k + "=" + encodeURIComponent(state[k]));
  }
  history.replaceState(null, "", "#" + parts.join("&"));
}
window.addEventListener("hashchange", () => {
  const investigation = new URLSearchParams(location.hash.slice(1)).get("investigation");
  if (investigation && investigation !== state.investigation) {
    switchTab("arch");
    showInvestigation(investigation);
    return;
  }
  const component = new URLSearchParams(location.hash.slice(1)).get("component");
  if (component && component !== state.component) {
    switchTab("arch");
    showComponent(component);
    return;
  }
  const asked = new URLSearchParams(location.hash.slice(1)).get("doc");
  const key = asked && docKey(asked);
  if (key && Object.prototype.hasOwnProperty.call(DOC, key)) {
    state.doc = key;
    openDocOrDetail(key);
  }
});
// an address naming a box's own entry opens that box's ONE place: its detail panel when it has one
function openDocOrDetail(key) {
  const did = detailIdForKey(key);
  if (did) renderDetail(did);
  else showDoc(key, DOC[key].t);
}

// ------------------------------------------------------------------ tiny helpers
function el(tag, attrs, ...kids) {
  const svgTags = { svg: 1, g: 1, rect: 1, text: 1, tspan: 1, path: 1, line: 1, circle: 1, defs: 1, marker: 1, polyline: 1, title: 1 };
  const n = svgTags[tag] ? document.createElementNS("http://www.w3.org/2000/svg", tag)
                         : document.createElement(tag);
  for (const k in (attrs || {})) {
    if (k === "onclick" || k.startsWith("on")) n.addEventListener(k.slice(2), attrs[k]);
    else if (k === "text") n.textContent = attrs[k];
    else n.setAttribute(k, attrs[k]);
  }
  for (const kid of kids) {
    if (kid == null) continue;
    n.appendChild(typeof kid === "string" ? document.createTextNode(kid) : kid);
  }
  return n;
}
// fmt() is for a number that carries its REAL UNIT in the same string ("12.5 cm", "40 ms"):
// DESIGN.md section 11 lets a person read a count, a real unit or a percentage between two named
// ends, and nothing else, so every other number on the page goes through the readers further down
// (per100, pct, wayBetween, tokErr, picture, ...). It never writes exponent form - "1.0e-5" reads
// as nothing at all - and a nonzero value too small for its decimals says "under" that step.
function fmt(x, d) {
  if (x === null || x === undefined || x !== x) return "—";
  if (typeof x !== "number") return String(x);
  const dd = d === undefined ? 3 : d;
  if (x !== 0 && Math.abs(x) < 0.5 * Math.pow(10, -dd))
    return (x < 0 ? "-" : "") + "under " + Math.pow(10, -dd).toFixed(dd);
  return x.toFixed(dd);
}
function isObj(x) { return x && typeof x === "object" && !Array.isArray(x); }
// what a raw internal says where it has no count form yet (DESIGN.md section 11)
const NOT_COUNT = "not yet measured as a count";
// the page's own words for the named-things reader's shut gate, shown when the model's reason
// (mimo_bridge/self_value_reader.py, the model lane's text) carries a bare decimal
const READER_WHY = "its gate is shut: the real reader does not yet track wellness by the line it "
  + "must reach on its own before its shuffled copies are compared (a correlation: " + NOT_COUNT + ")";
// the same for the wellness channels' evidence gates (lem/model/wellness_channels.py and
// lem/model/contact_head.py write their reasons with raw numbers; the model lane owns that text)
const CONTROL_GATE_WHY = "its readings do not yet beat their control by twice their own standard "
  + "error over the last 32 (an agreement of expected and achieved change, averaged: " + NOT_COUNT + ")";
const CONTACT_GATE_WHY = "foreseeing with this body's own command does not yet beat foreseeing with "
  + "another body's command by twice the readings' standard error over the last 32";
// a decimal a person would read with no real unit after it - the page's own guard for sentences
// the model writes (a gate's reason, the reader's reason); tools/check_no_bare_decimals.py is the
// rule the built page is tested against
const BARE_DECIMAL = /(^|[^\w.\/\\#:-])-?\d*\.\d+(?!\d)(?!\s*(%|°|cm\b|mm\b|m\b|km\b|s\b|ms\b|sec|seconds?\b|minutes?\b|hours?\b|h\b|days?\b|GB\b|MB\b|KB\b|kg\b|g\b|N\b|W\b|Hz\b|kPa\b|Pa\b|steps?\b|frames?\b|per\b|x\b|times\b|degrees?\b))/i;

// dequantize schema.py's uint8 blocks -> {rows, cols, planes, get(i,j,k)}
function decodeQ(q) {
  if (!q || !q.u8) return null;
  const bin = atob(q.u8);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  const f = new Float32Array(u8.length);
  const lo = q.min, span = (q.max - q.min);
  for (let i = 0; i < u8.length; i++) f[i] = lo + (u8[i] / 255) * span;
  return { shape: q.shape, data: f };
}
function slice2d(dq, plane) {  // [P, R, C] -> plane p as {rows, cols, at}
  const [P, R, C] = dq.shape.length === 3 ? dq.shape : [1, dq.shape[0], dq.shape[1]];
  const off = (dq.shape.length === 3 ? plane : 0) * R * C;
  return { rows: R, cols: C, at: (r, c) => dq.data[off + r * C + c] };
}

// color scales
function seqColor(v, vmax) {           // 0..vmax -> white -> deep blue
  const t = Math.max(0, Math.min(1, vmax ? v / vmax : 0));
  const r = Math.round(248 - t * 210), g = Math.round(250 - t * 175), b = Math.round(252 - t * 90);
  return `rgb(${r},${g},${b})`;
}
function divColor(v, vmax) {           // -vmax..vmax -> red..white..blue
  const t = Math.max(-1, Math.min(1, vmax ? v / vmax : 0));
  if (t >= 0) return `rgb(${Math.round(245 - t * 170)},${Math.round(245 - t * 120)},245)`;
  return `rgb(245,${Math.round(245 + t * 150)},${Math.round(245 + t * 160)})`;
}

// ------------------------------------------------------------------ heatmap
// opts.say turns a cell's value into the words the hover shows (DESIGN.md section 11): by default a
// cell is an attention weight, and a row's weights add to 1, so the hover says how many of every
// 100 parts of that row's budget the cell holds
function heatmap(mat, opts) {
  opts = opts || {};
  const say = opts.say || (v => Math.round(100 * v) + " of every 100 parts of this row's attention");
  const cell = opts.cell || 14;
  const wrap = el("div", { class: "hmwrap" });
  const cv = el("canvas", { class: "hm", width: mat.cols * cell, height: mat.rows * cell });
  if (opts.cssWidth) cv.style.width = opts.cssWidth + "px";
  const ctx = cv.getContext("2d");
  let vmax = 0;
  for (let r = 0; r < mat.rows; r++) for (let c = 0; c < mat.cols; c++)
    vmax = Math.max(vmax, mat.at(r, c));
  vmax = opts.vmax || vmax || 1;
  for (let r = 0; r < mat.rows; r++)
    for (let c = 0; c < mat.cols; c++) {
      ctx.fillStyle = seqColor(mat.at(r, c), vmax);
      ctx.fillRect(c * cell, r * cell, cell, cell);
    }
  const tip = el("div", { class: "hmtip" });
  cv.addEventListener("mousemove", ev => {
    const b = cv.getBoundingClientRect();
    const scale = cv.width / b.width;
    const c = Math.floor((ev.clientX - b.left) * scale / cell);
    const r = Math.floor((ev.clientY - b.top) * scale / cell);
    if (r < 0 || c < 0 || r >= mat.rows || c >= mat.cols) { tip.style.display = "none"; return; }
    const rl = opts.ylabels ? opts.ylabels[r] : r, clb = opts.xlabels ? opts.xlabels[c] : c;
    tip.textContent = `${rl} → ${clb}: ${say(mat.at(r, c))}`;
    tip.style.display = "block";
    tip.style.left = (ev.clientX - b.left + 12) + "px";
    tip.style.top = (ev.clientY - b.top - 22) + "px";
  });
  cv.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  wrap.appendChild(cv); wrap.appendChild(tip);
  return wrap;
}

// prediction-vs-actual strip: 3 rows (predicted, actual, |difference|) x d columns
function pairStrip(pred, act) {
  const d = Math.min(pred.length, act.length), cell = 3, rows = 3;
  const cv = el("canvas", { class: "hm", width: d * cell, height: rows * cell * 3 });
  const ctx = cv.getContext("2d");
  let vmax = 0.3;
  for (let i = 0; i < d; i++) vmax = Math.max(vmax, Math.abs(pred[i]), Math.abs(act[i]));
  const rowsData = [pred, act, pred.map((p, i) => Math.abs(p - act[i]))];
  for (let r = 0; r < rows; r++)
    for (let i = 0; i < d; i++) {
      const v = rowsData[r][i];
      ctx.fillStyle = r < 2 ? divColor(v, vmax) : seqColor(v, vmax);
      ctx.fillRect(i * cell, r * cell * 3, cell, cell * 3 - 1);
    }
  return cv;
}

// ------------------------------------------------------------------ line charts
const PALETTE = ["#2a5db0", "#b5574a", "#3a8a5a", "#8a5ab0", "#c98a2a", "#4aa0b5",
                 "#b04a86", "#6b7280", "#95651a", "#4a4ab0"];
function niceTicks(lo, hi, n) {
  if (!(hi > lo)) hi = lo + 1;
  const span = hi - lo, step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
  let step = step0;
  for (const m of [1, 2, 5, 10]) { if (span / (step0 * m) <= n) { step = step0 * m; break; } }
  const t = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) t.push(v);
  return t;
}
// A CHART'S VIEW OF ITS DATA (DESIGN.md section 11). The series carry the run's own logged values;
// a chart says how a person reads them: `scale` (a share drawn per 100, an error as a percentage of
// a named reference), `relative` (each line as a percentage of its own first point: 100% = where
// this log began, 0% = none of it left), and `unit`, printed on every axis mark and in the hover
// readout. A chart of a raw internal with no count form yet - a cosine - carries `ends` instead:
// [value, "what that value means"] pairs drawn as named reference lines, and NO numeric axis mark.
function chartView(chart) {
  const k = Number(chart.scale) || 1;
  return chart.series.map(s => {
    let base = 1;
    if (chart.relative) {
      const first = s.points.find(p => p[1] != null && p[1] === p[1] && p[1] !== 0);
      base = first ? Math.abs(first[1]) / 100 : null;
    }
    return base === null ? [] : s.points.map(p => [p[0], p[1] == null ? p[1] : p[1] * k / base]);
  });
}
// an axis mark's number: as many decimals as the spacing of the marks needs and no more, never exponent form
function tickNumber(v, step) {
  const d = Math.max(0, Math.min(6, Math.ceil(-Math.log10(step || 1) - 1e-9)));
  const s = v.toFixed(d);
  return /^-0(\.0+)?$/.test(s) ? s.slice(1) : s;
}
function logTickNumber(tv) {
  const v = Math.pow(10, tv);
  return tv >= 0 ? Math.round(v).toLocaleString() : v.toFixed(Math.min(8, Math.round(-tv)));
}
// one plotted value in the chart's own terms, for the hover readout
function sayChartValue(chart, v) {
  if (v === null || v === undefined || v !== v) return "—";
  if (Array.isArray(chart.ends)) return nearestEnd(v, chart.ends);
  const u = chart.unit;
  if (chart.integer) return Math.round(v).toLocaleString() + (u ? " " + u : "");
  if (u === "%") return pctNum(v);
  if (u === "/100") return Math.round(v) + " of 100";
  if (u === "°") return sig3(v) + "°";
  if (u) return sig3(v) + " " + u;
  return "a raw internal - " + NOT_COUNT;
}
function lineChart(chart, width) {
  const W = width || 640, H = 150, ml = Array.isArray(chart.ends) ? 150 : 52, mr = 10, mt = 8, mb = 20;
  const svg = el("svg", { class: "chart", width: W, height: H, viewBox: `0 0 ${W} ${H}` });
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  const log = !!chart.log;
  const ends = Array.isArray(chart.ends) ? chart.ends : null;
  const view = chartView(chart);
  for (const pts of view) for (const [x, y] of pts) {
    if (y == null || y !== y) continue;
    if (log && y <= 0) continue;
    x0 = Math.min(x0, x); x1 = Math.max(x1, x);
    const yy = log ? Math.log10(y) : y;
    y0 = Math.min(y0, yy); y1 = Math.max(y1, yy);
  }
  if (x0 === Infinity) { svg.appendChild(el("text", { x: 10, y: 20, text: "no data" })); return { svg }; }
  // the named ends a chart declares are always on the axis, so a line is read against them
  if (ends) for (const [v] of ends) if (!chart.endsInside || (v >= y0 && v <= y1)) { y0 = Math.min(y0, v); y1 = Math.max(y1, v); }
  if (y0 === y1) { y0 -= 0.5; y1 += 0.5; }
  const px = x => ml + (x - x0) / (x1 - x0 || 1) * (W - ml - mr);
  const py = y => { const yy = log ? Math.log10(Math.max(y, 1e-12)) : y;
                    return mt + (1 - (yy - y0) / (y1 - y0)) * (H - mt - mb); };
  // a chart that declares its unit prints it on every axis mark (DESIGN.md section 11): "%" and "/100"
  // and "°" attach to the number, a word unit follows a space
  const U = chart.unit ? (["%", "/100", "°"].indexOf(chart.unit) >= 0 ? chart.unit : " " + chart.unit) : "";
  if (ends) {
    for (const [tv, name] of ends) {
      const y = mt + (1 - (tv - y0) / (y1 - y0)) * (H - mt - mb);
      svg.appendChild(el("line", { x1: ml, x2: W - mr, y1: y, y2: y, stroke: "#d6d9df", "stroke-dasharray": "3 3" }));
      svg.appendChild(el("text", { x: ml - 4, y: y + 3, "text-anchor": "end", class: "axis-label", text: name }));
    }
  } else {
    let marks = niceTicks(y0, y1, 4);
    if (chart.integer) marks = marks.filter(t => Math.abs(t - Math.round(t)) < 1e-9);
    const step = marks.length > 1 ? marks[1] - marks[0] : 1;
    for (const tv of marks) {
      const y = mt + (1 - (tv - y0) / (y1 - y0)) * (H - mt - mb);
      svg.appendChild(el("line", { x1: ml, x2: W - mr, y1: y, y2: y, stroke: "#eee" }));
      const lab = log ? logTickNumber(tv) : tickNumber(tv, step);
      svg.appendChild(el("text", { x: ml - 4, y: y + 3, "text-anchor": "end", class: "axis-label", text: lab + U }));
    }
  }
  for (const tv of niceTicks(x0, x1, 6)) {
    svg.appendChild(el("text", { x: px(tv), y: H - 6, "text-anchor": "middle", class: "axis-label",
                                 text: tv >= 1e6 ? (tv / 1e6).toFixed(1) + "M" : tv >= 1000 ? Math.round(tv / 1000) + "k" : Math.round(tv) }));
  }
  // markers: resume seams (red), phase changes (green), snapshots (blue, clickable).
  // They are positions on a TRAINING-STEP axis; a chart whose x axis is anything else
  // (the stage-3 time-distance chart) passes noStepMarkers and gets none.
  const stepAxis = !chart.noStepMarkers;
  for (const sstep of (stepAxis ? SERIES.seams : null) || []) if (sstep >= x0 && sstep <= x1)
    svg.appendChild(el("line", { x1: px(sstep), x2: px(sstep), y1: mt, y2: H - mb,
                                 stroke: "#c05050", "stroke-dasharray": "4 3" }));
  for (const ph of (stepAxis ? SERIES.phases : null) || []) if (ph.step >= x0 && ph.step <= x1)
    svg.appendChild(el("line", { x1: px(ph.step), x2: px(ph.step), y1: mt, y2: H - mb,
                                 stroke: "#3a8a5a", "stroke-dasharray": "2 3" }));
  chart.series.forEach((s, si) => {
    const col = s.color || PALETTE[si % PALETTE.length];
    // A LOG AXIS CANNOT DRAW ZERO, and zero is exactly what the collapse watch
    // exists to catch. Dropping those points used to join a straight line across
    // the gap, hiding the one reading the chart is there to show. The line now
    // BREAKS at every undrawable point, and a point that is a real number at or
    // below zero gets a hollow dot on the chart's floor.
    let d = "", open = false;
    const floored = [];
    for (const p of view[si]) {
      const missing = p[1] == null || p[1] !== p[1];
      if (missing || (log && p[1] <= 0)) {
        if (!missing) floored.push(p[0]);
        open = false;
        continue;
      }
      d += (open ? "L" : "M") + px(p[0]).toFixed(1) + " " + py(p[1]).toFixed(1);
      open = true;
    }
    if (d) svg.appendChild(el("path", { d, fill: "none", stroke: col, "stroke-width": 1.4 }));
    for (const fx of floored) {
      const dot = el("circle", { cx: px(fx), cy: H - mb, r: 3, fill: "none", stroke: col });
      dot.appendChild(el("title", { text: s.label + " was at or below zero at step "
        + fx + " - a log axis cannot draw it, so it sits on the floor" }));
      svg.appendChild(dot);
    }
  });
  for (const ms of (stepAxis ? SERIES.snapmarks : null) || []) if (ms >= x0 && ms <= x1) {
    const c = el("circle", { cx: px(ms), cy: mt + 4, r: 4, fill: "#2a5db0", stroke: "#fff",
                             style: "cursor:pointer" });
    c.addEventListener("click", () => { openSnapshot(ms); });
    c.appendChild(el("title", { text: "snapshot at step " + ms + " - click to inspect" }));
    svg.appendChild(c);
  }
  const cursor = el("line", { x1: -10, x2: -10, y1: mt, y2: H - mb, stroke: "#555" });
  svg.appendChild(cursor);
  return { svg, px, py, x0, x1, cursor, view, ml };
}

// ------------------------------------------------------------------ DOC popover
// Popups can nest: [[term]] markers in any DOC text become clickable words that
// open that term's own popup, with a back button returning to the parent.
function docBtn(key, title) {
  const b = el("button", { class: "doc-btn", title: "what is this?", text: "i" });
  b.addEventListener("click", ev => { ev.stopPropagation(); showDoc(key, title); });
  return b;
}
// Popup text is plain sentences with four light marks: [[term]] or [[term|shown words]]
// opens that glossary entry, **words** are bold, a name between two backticks is the code's own
// name for a thing (the style guide "ONE WORD PER THING": given once beside the canonical name) and is
// set in code type, a blank line starts a new paragraph, and lines that start with "- " are a
// bullet list. Text with none of these renders as before.
function renderDocInline(container, text) {
  const re = /\[\[([a-z0-9_.-]+)(?:\|([^\]]+))?\]\]|\*\*([^*]+)\*\*|\x60([^\x60\n]+)\x60/gi;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) container.appendChild(document.createTextNode(text.slice(last, m.index)));
    if (m[4] !== undefined) {
      container.appendChild(el("code", { text: m[4] }));
    } else if (m[3] !== undefined) {
      container.appendChild(el("b", { text: m[3] }));
    } else {
      const key = "term." + m[1].toLowerCase(), shown = m[2] || m[1].replace(/-/g, " ");
      const a = el("a", { class: "term", href: "#", text: shown,
                          title: "click for a definition" });
      a.addEventListener("click", ev => {
        ev.preventDefault(); ev.stopPropagation();
        showDoc(key, shown, null, true);
      });
      container.appendChild(a);
    }
    last = re.lastIndex;
  }
  if (last < text.length) container.appendChild(document.createTextNode(text.slice(last)));
}
function renderDocText(container, text) {
  text = String(text);
  const blocks = text.split(/\n\s*\n/);
  // a <p> cannot hold paragraphs or a list: flatten for it
  if (container.tagName === "P" || (blocks.length === 1 && !/^- /m.test(text))) {
    renderDocInline(container, text.replace(/\n- /g, " \u2022 ").replace(/^- /, "\u2022 ").replace(/\s*\n\s*/g, " "));
    return;
  }
  for (const blk of blocks) {
    const lines = blk.split("\n").filter(l => l.trim());
    if (!lines.length) continue;
    if (lines.every(l => /^- /.test(l))) {
      const ul = el("ul", { class: "doclist" });
      for (const l of lines) { const li = el("li"); renderDocInline(li, l.slice(2)); ul.appendChild(li); }
      container.appendChild(ul);
    } else if (lines.every(l => /^[0-9]+[.] /.test(l))) {
      const ol = el("ol", { class: "doclist" });
      for (const l of lines) { const li = el("li"); renderDocInline(li, l.replace(/^[0-9]+[.] /, "")); ol.appendChild(li); }
      container.appendChild(ol);
    } else {
      const p = el("div", { class: "docp" });
      renderDocInline(p, lines.join(" "));
      container.appendChild(p);
    }
  }
}
// OLD KEYS KEPT AS ALIASES (the style guide "ONE WORD PER THING", 2026-10-02; three more on 2026-10-03). Five
// glossary keys carried a word the canonical names replace; their entries moved to correctly named keys,
// and a link or a saved #doc= address written with the old key still opens the moved entry. The old keys
// are code names, so they are written in backticks.
const DOC_ALIAS = { [`term.ring-moment`]: "term.goal-ring-entry",
                    [`term.candidate-order`]: "term.candidate-command",
                    [`term.moment`]: "term.model-step",
                    [`term.executed-order`]: "term.executed-command",
                    [`term.shuffled-order-control`]: "term.shuffled-command-control" };
const docKey = key => (DOC_ALIAS[key] && DOC[DOC_ALIAS[key]]) ? DOC_ALIAS[key] : key;
let docTrail = [];
function showDoc(key, title, extraKv, nested) {
  key = docKey(key);
  if (!nested) docTrail = [];
  docTrail.push({ key, title, extraKv });
  renderDocPanel();
}
// the four fields every entry may carry, as the popup and the detail panel both show them
function docFieldList(d) {
  if (d.plain) {
    const wrap = el("div", {class: "plain-part-explanation"});
    const intro = el("div"); renderDocText(intro, d.plain); wrap.appendChild(intro);
    if (d === DOC["node.self_state"]) {
      const mode = (META.receipt || {}).self_state || ARGS.self_state || "unknown";
      const choice = ARGS.self_choice || "off";
      const run = el("p", {class: "note"});
      run.textContent = "Recorded run: " + META.run + ". Self perception: " + mode + "; self-choice: " + choice + ". "
        + (mode === "perceive" ? "Its body estimate also reaches the forward model. " : "Its body estimate is not added to the forward model. ")
        + (choice === "off" ? "The separate self-choice route is off. " : choice === "shadow" ? "Self-choice reports comparisons without sending commands. " : "Self-choice can supply commands only when its qualification and execution checks allow them. ")
        + (ARGS.eval_only ? "This recorded run evaluates without learning." : "During training, the teaching score updates this part’s networks.");
      wrap.appendChild(run);
    }
    if (d === DOC["node.intention"]) {
      const mode = ARGS.prospective_wellness || "off";
      const note = el("p", {class: "note", text: "Recorded run: " + META.run + ". Prospective wellness: " + mode + ". "
        + (mode === "off" ? "Here wellness chooses the remembered goal; this extra predicted-wellness score is off."
          : "This run enables the optional predicted-wellness term; see exact calculations for its weight and conditions.")});
      wrap.appendChild(note);
    }
    const more = el("details"); more.appendChild(el("summary", {text: "Exact calculations and source references"}));
    const technical = Object.assign({}, d); delete technical.plain;
    more.appendChild(docFieldList(technical)); wrap.appendChild(more); return wrap;
  }
  const dl = el("dl");
  const add = (label, txt) => {
    if (!txt) return;
    dl.appendChild(el("dt", { text: label }));
    const dd = el("dd");
    renderDocText(dd, txt);
    dl.appendChild(dd);
  };
  add("what it is", d.w);
  add("when it runs", d.when);
  add(d.h_label || "how it is computed", d.h);
  add(d.f_label || "what would fool it", d.f);
  return dl;
}
// DOC IS WRITTEN ONCE AND SHIPPED ON EVERY PAGE. Its "this run", "this probe" and
// "the audited run" mean the run the text was written about, which is the run on
// this page only some of the time - so every popup that uses the phrase says whose.
function docRunNote(d) {
  const wf = META.doc_written_for;
  const said = [d.w, d.when, d.h, d.f].filter(Boolean).join(" ");
  if (!(wf && /this run|this probe|the audited run|this page's run|this frozen run/i.test(said))) return null;
  return el("p", { class: "note", text:
      "Where this text says 'this run', 'this probe' or 'the audited run', it means "
      + wf.run + " (" + wf.what + ")"
      + (wf.run === META.run
         ? ", which is the run on this page."
         : " - NOT " + META.run + ", the run on this page. A measurement quoted in this "
           + "text can differ on this page's run. What the page fills in from its own run "
           + "when it is built - sizes, counts, launch-flag values, model step lengths - is "
           + META.run + "'s own, and a sentence that names " + META.run + " is about that "
           + "run; the live values below and the run tab are this page's own.") });
}
function renderDocPanel() {
  const old = document.getElementById("docpanel");
  if (old) old.remove();
  if (!docTrail.length) return;
  const { key, title, extraKv, component } = docTrail[docTrail.length - 1];
  const d = DOC[key] || {};
  const p = el("div", { id: "docpanel" });
  const bar = el("div", { class: "docbar" });
  if (docTrail.length > 1)
    bar.appendChild(el("button", { class: "back", text: "◀ back",
      onclick: () => { docTrail.pop(); renderDocPanel(); } }));
  bar.appendChild(el("button", { class: "close", text: "×",
    onclick: () => { docTrail = []; p.remove(); delete state.doc; delete state.component; markComponentBoxes(null);
                     writeHash(); } }));
  p.appendChild(bar);
  p.appendChild(el("h4", { text: d.t || title || key }));
  p.appendChild(docFieldList(d));
  const note = docRunNote(d);
  if (note) p.appendChild(note);
  if (extraKv) {
    const kv = el("div", { class: "kv" });
    for (const [k, v] of extraKv) {
      kv.appendChild(el("b", { text: k }));
      kv.appendChild(v instanceof Node ? el("span", {}, v) : el("span", { text: String(v) }));
    }
    p.appendChild(el("h4", { text: "live values (newest snapshot)" }));
    p.appendChild(kv);
  }
  // the components of the component manifest: the one this popup was opened for, or those drawn in this box
  const ids = component ? [component] : componentIdsOfKey(key);
  if (ids.length) p.appendChild(componentsSection(ids, component ? "the component and its code and QA record"
                                                                 : componentsHeading(ids.length)));
  document.body.appendChild(p);
}

// ------------------------------------------------ text measured before it is placed
// A drawing's labels are laid out from their MEASURED width (the style guide "DIAGRAM LAYOUT ON A GRID"):
// a canvas measures a string in the page's own font at the size the drawing uses. In an SVG the
// font size is in the drawing's own units, so the width comes back in those units too. Without a
// canvas the width is estimated at 0.55 of the font size per character.
const MEASURE = { ctx: null };
function textWidth(s, px) {
  try {
    if (!MEASURE.ctx) MEASURE.ctx = document.createElement("canvas").getContext("2d");
    MEASURE.ctx.font = px + 'px "Segoe UI", system-ui, sans-serif';
    return MEASURE.ctx.measureText(String(s)).width;
  } catch (e) {
    return String(s).length * px * 0.55;
  }
}
// s broken at spaces into lines no wider than maxW units at font size px
function wrapText(s, px, maxW) {
  const out = [];
  let cur = "";
  for (const w of String(s).split(/\s+/).filter(Boolean)) {
    const t = cur ? cur + " " + w : w;
    if (cur && textWidth(t, px) > maxW) { out.push(cur); cur = w; } else cur = t;
  }
  if (cur) out.push(cur);
  return out.length ? out : [""];
}
// a sentence made of runs - [words, class, click handler] - broken at its spaces into lines no wider
// than maxW units at font size px; a run keeps its class and its handler on every line it reaches
function wrapRuns(runs, px, maxW) {
  const lines = [[]];
  let w = 0;
  for (const [text, cls, on] of runs) {
    for (const part of String(text).split(/(?= )/)) {
      let piece = part;
      if (w > 0 && part[0] === " " && w + textWidth(part, px) > maxW) { lines.push([]); w = 0; piece = part.slice(1); }
      if (!piece) continue;
      const line = lines[lines.length - 1], last = line[line.length - 1];
      if (last && last[1] === cls && last[2] === on) last[0] += piece;
      else line.push([piece, cls, on]);
      w += textWidth(piece, px);
    }
  }
  return lines.filter(l => l.length);
}

// ------------------------------------------------ small clickable flow diagrams
// Every box and every arrow can carry a DOC key: clicking it opens that popup
// (nested when the diagram itself lives inside a popup). Arrows carry a LABEL
// naming the data that flows - the contract, not just a line.
function flowDiagram(W, H, boxes, arrows, footnotes, nested) {
  // The footnotes are drawn INSIDE this svg, stacked upward from its bottom edge
  // (see the emit below). So a caller's H has to leave room for them, and getting
  // that arithmetic wrong silently prints the notes THROUGH the lowest boxes -
  // which is what the contact-concentration diagram did: boxes down to y=266 with
  // the first of 8 notes landing at y=203. Grow H here instead of trusting every
  // caller to redo the sum whenever a note is added.
  const lowest = Math.max(
    0,
    ...boxes.map(b => b.y + b.h),
    ...(arrows || []).flatMap(a => [...(a.pts || []).map(pt => pt[1]),
                                    ...(a.ly ? [a.ly + 11 * (Array.isArray(a.label) ? a.label.length - 1 : 0)] : [])]));
  // A FOOTNOTE IS WRAPPED TO THE DRAWING'S WIDTH from its measured text: one unwrapped line ran
  // past the right edge in six drawings (the layout check of 2026-10-02). Lines of one footnote
  // are 12 units apart, footnotes 3 units more.
  const FOOT_PX = 9, FOOT_LINE = 12, FOOT_GAP = 3;
  const footLines = (footnotes || []).map(f => wrapText(f, FOOT_PX, (W - 20) * 0.96));
  const nLines = footLines.reduce((a, l) => a + l.length, 0);
  const footH = nLines ? nLines * FOOT_LINE + (footLines.length - 1) * FOOT_GAP : 0;
  H = Math.max(H, nLines ? Math.ceil(lowest + 12 + footH + 6) : Math.ceil(lowest + 6));
  const svg = el("svg", { class: "mini", viewBox: `0 0 ${W} ${H}` });
  svg.appendChild(el("defs", {},
    el("marker", { id: "miniarrow", viewBox: "0 0 10 10", refX: 9, refY: 5,
                   markerWidth: 7, markerHeight: 7, orient: "auto" },
      el("path", { d: "M0 0 L10 5 L0 10 z", fill: "#6b7280" }))));
  // A box or a line inside the detail panel whose key is the panel's OWN entry opens no popup: the
  // panel above the drawing already shows that entry's text, so the click scrolls to it instead
  // (one place per box, operator 2026-10-02: "REDUNDANT POPUP")
  const open = (key, title) => {
    if (!key) return;
    if (!nested && key === detailOwnKey) { showDetailText(); return; }
    showDoc(key, title, null, !!nested);
  };
  // a label may be one string or an array of lines; each line is 11 units below the one before, all
  // in ONE text element (tspans), so the layout check measures a wrapped label as one block
  const lines = s => Array.isArray(s) ? s : [s];
  const flat = s => lines(s).join(" ");
  const multi = (attrs, s) => {
    const t = el("text", attrs);
    lines(s).forEach((ln, i) => t.appendChild(el("tspan", { x: attrs.x, dy: i ? 11 : 0, text: ln })));
    return t;
  };
  function drawArrow(a) {
    const d = "M" + a.pts.map(p => p.join(" ")).join(" L");
    const attrs = { class: "mini-edge" + (a.dash ? " dash" : "") + (a.plain ? " plainline" : ""), d };
    if (!a.plain) attrs["marker-end"] = "url(#miniarrow)";
    svg.appendChild(el("path", attrs));
    if (a.key) {
      const hit = el("path", { class: "dg-hit", d });
      hit.addEventListener("click", ev => { ev.stopPropagation(); open(a.key, flat(a.label || "")); });
      svg.appendChild(hit);
    }
    if (a.label) {
      const t = multi({ x: a.lx, y: a.ly, "text-anchor": a.anchor || "middle",
                        class: "flowlabel" + (a.key ? " clicky" : "") }, a.label);
      // a plain line is a curve drawn INSIDE a box, so its label belongs to that box too: the
      // layout check (#layoutcheck) reads data-in-box and holds the label inside its box
      if (a.plain) t.setAttribute("data-in-box", "1");
      if (a.key) t.addEventListener("click", ev => { ev.stopPropagation(); open(a.key, flat(a.label)); });
      svg.appendChild(t);
    }
  }
  // Connecting arrows go UNDER the boxes; a plain line is a plotted curve drawn
  // INSIDE a box, so it goes over. SVG has no z-index - the drawing sequence is everything, and
  // .mini-box is an opaque fill, so a curve emitted before its own panel is
  // painted out. The curiosity drive curve was invisible for exactly that reason.
  arrows.filter(a => !a.plain).forEach(drawArrow);
  for (const b of boxes) {
    const g = el("g", { class: "mini-node" + (b.key ? " clicky" : "") });
    g.appendChild(el("rect", { x: b.x, y: b.y, width: b.w, height: b.h, rx: 5, class: "mini-box" }));
    g.appendChild(el("text", { x: b.x + b.w / 2, y: b.y + (b.l2 ? 16 : b.h / 2 + 4),
                               "text-anchor": "middle", text: b.l1 }));
    // l2 may be one line or several (an array), 11 units apart
    if (b.l2) g.appendChild(multi({ x: b.x + b.w / 2, y: b.y + 30, "text-anchor": "middle",
                                    class: "sublabel" }, b.l2));
    if (b.key) g.addEventListener("click", ev => { ev.stopPropagation(); open(b.key, b.l1); });
    svg.appendChild(g);
  }
  arrows.filter(a => a.plain).forEach(drawArrow);
  let fy = H - 6 - (footH - FOOT_LINE);          // the first footnote's first baseline
  for (const ls of footLines) {
    const t = el("text", { x: 10, y: fy, class: "sublabel" });
    ls.forEach((ln, i) => t.appendChild(el("tspan", { x: 10, dy: i ? FOOT_LINE : 0, text: ln })));
    svg.appendChild(t);
    fy += ls.length * FOOT_LINE + FOOT_GAP;
  }
  return svg;
}

// THE PLAN RECORD IN ONE SENTENCE (P125; rewritten 2026-10-02 to the quantities the code computes).
// start_mse, final_mse and mse_by_depth are LemLite._plan_chain's distance: for every sense measured
// at that model step except motor, the mean over the token's numbers of (tokens - goal tokens)^2,
// ADDED over those senses and averaged over the bodies. Two unrelated tokens differ by 2 per sense
// on that scale (tokens are held at size 1), so it is written as a percentage of 2 x that many
// senses: 0% = at the goal tokens, 100% = as unlike them as unrelated tokens. mse_by_depth[0] is
// this model step's distance; entry k is the predicted distance after the plan's first k commands,
// and the distance after the LAST command of a plan is never measured - for a one-command plan
// none is. The descent rows (descentRows) carry the predicted distance before and after the 20
// updates.
function planLine(it, nCmp) {
  if (!it) return "no plan at this model step (the planner is off, or this model step is not a planner step)";
  const d = it.mse_by_depth || [];
  const top = 2 * (nCmp || 1);
  const pc = x => { const v = num(x); return v === null ? "—" : pctNum(100 * v / top); };
  const why = { diverged: "the predicted distance after a command grew more than 5% past the smallest so far",
                converged: "the distance fell under one hundredth, a sum of squared differences",
                scale_gate: "a predicted token's size left half to twice the size of this model step's token, so the whole plan was dropped",
                low_predicted_wellness: "the learned wellness copy scored a predicted model step under 5% of its 0-to-1 range",
                horizon: "it reached its depth limit" }[it.stopped_by || "horizon"];
  const len = it.plan_length;
  const handLimited = ARGS.hand_target_mode === "wrist_relative_v1";
  let s = len + " command" + (len === 1 ? "" : "s") + " long; licensed depth " + (it.horizon === undefined ? "?" : it.horizon)
    + (handLimited ? " (--hand-target-mode wrist_relative_v1 limits every plan to one command)" : "")
    + "; this model step's tokens sat " + pc(it.start_mse) + " of the way from the goal tokens to unrelated tokens"
    + (nCmp ? " (" + nCmp + " senses compared)" : "");
  s += d.length > 1
    ? "; predicted after the first " + (d.length - 1) + " command" + (d.length > 2 ? "s" : "") + ": " + d.slice(1).map(pc).join(" -> ")
    : "; no predicted distance after the command is recorded for a one-command plan: the descent rows give the score before and after the 20 updates";
  return s + (it.stopped_by ? "; it stopped because " + why : "");
}
// THE 20 UPDATES OF ONE PLANNING EVENT (generate_action's own record, kept per depth in
// last_intention.descent): the score's two parts for the starting command and for the kept one,
// averaged over the bodies, and the update each body kept. The goal part is written against two
// unrelated tokens per compared sense, the strength part against its largest value (every handle
// at full strength: one hundredth of the handle count, times the body count under
// --planner-effort-per-body off).
function descentRows(it, nCmp) {
  const ds = (it && it.descent) || [];
  if (!it) return [];
  if (!ds.length) return [["the 20 updates", "not in this snapshot's plan record (a capture from before the planner recorded them)"]];
  const top = 2 * (nCmp || 1);
  const nH = ((META.receipt || {}).handle_names || []).length || 24;
  const effTop = nH / 100 * (ARGS.planner_effort_per_body === "on" ? 1 : (Number(ARGS.n_envs) || 1));
  const ch = (a, b) => (a !== null && b !== null && a > 0) ? (b <= a ? "down " : "up ") + pctNum(100 * Math.abs(b - a) / a) : "—";
  const rows = [];
  ds.forEach((d, i) => {
    const tag = ds.length > 1 ? " (depth " + (i + 1) + ")" : "";
    const g0 = num(d.first_goal_distance), g1 = num(d.last_goal_distance);
    const e0 = num(d.first_effort), e1 = num(d.last_effort);
    rows.push(["goal part of the score" + tag + ": starting command -> kept command",
      (g0 === null ? "—" : pctNum(100 * g0 / top)) + " -> " + (g1 === null ? "—" : pctNum(100 * g1 / top))
      + " of the way from the goal tokens to unrelated tokens (" + ch(g0, g1) + "), averaged over the bodies"]);
    rows.push(["strength part of the score" + tag + ": starting command -> kept command",
      (e0 === null ? "—" : pctNum(100 * e0 / effTop)) + " -> " + (e1 === null ? "—" : pctNum(100 * e1 / effTop))
      + " of its largest value, every handle at full strength (" + ch(e0, e1) + ")"]);
    const sel = Array.isArray(d.selected_iteration) ? d.selected_iteration : [];
    if (sel.length) {
      const count = {};
      sel.forEach(k => { count[k] = (count[k] || 0) + 1; });
      rows.push(["update kept" + tag, Object.keys(count).map(Number).sort((a, b) => b - a)
        .map(k => (k === 0 ? "the starting command" : "update " + k + " of " + (d.n_steps || 20)) + " on " + count[k] + " of " + sel.length + " bodies").join("; ")
        + (d.stopped_by && d.stopped_by !== "step_budget" ? "; the updates stopped early: " + String(d.stopped_by).replace(/_/g, " ") : "")]);
    }
  });
  return rows;
}

// THE GOAL ENTRY OF THE NEWEST PLANNING EVENT (review b3-wellness, minor, 2026-09-22: "the explainer
// names the wrong goal under the normalised ranking"). The capture used to name the goal-ring entry
// with the largest RAW stored value at the snapshot as the goal. The model now records the ring index
// and age of the entry it ranked highest (last_intention.goal_ring_index / goal_age_steps), and
// capture.py follows it through the ring to the snapshot (intention.goal_source "recorded"); without
// that record the page says so and names no goal. A snapshot taken before the fix carries the older
// capture's pick, shown as what it is: the ring's highest stored value at the snapshot. Stored values
// are sums of wellness terms, written against the most an entry can store without the curiosity term.
// The stored value divided by its curiosity factor (1 + the drive under --curiosity-mode multiply; 1
// under additive, where the drive is added instead) is the capture's goal_felt_value.
function goalRows(it, snap) {
  const ceiling = num(it.stored_value_ceiling_no_curiosity);
  const val = x => ceiling ? wayBetween(x, 0, ceiling, "every term at its worst",
                                         "the most an entry can store without the curiosity term") : wUnits(x);
  const rows = [];
  const lastStep = (it.last || {}).step, stepNow = (snap.header || {}).step;
  const when = (typeof lastStep === "number" && typeof stepNow === "number")
    ? "planned at step " + lastStep.toLocaleString() + ", " + (stepNow - lastStep) + " model steps before this snapshot" : "";
  const factor = f => Number(f) > 1 ? ", divided by its curiosity factor (raised " + pctNum(100 * (Number(f) - 1)) + " by the curiosity drive) "
                                    : ", with no curiosity factor, ";
  if (it.goal_source === "recorded" && it.goal_felt_value !== undefined) {
    rows.push(["the goal entry (ranked highest at the planning event)", (it.goal_age_now !== undefined ? it.goal_age_now + " model steps old at this snapshot" : "")
      + (when ? " (" + when + ", when it was " + it.goal_age_at_plan + " model steps old)" : "")]);
    rows.push(["its stored value", val(it.goal_stored_value)
      + (num(it.goal_curiosity_factor) !== null ? factor(it.goal_curiosity_factor) + val(it.goal_felt_value) : "")]);
    rows.push(["is that also the ring's highest value without the curiosity factor", (it.goal_wins_without_curiosity ? "yes" : "no")
      + " (the ring's highest such value " + val(it.best_felt_value) + ")"]);
    if (ARGS.wellness_rank === "normalized")
      rows.push(["how it was ranked", "by the normalised ranking (--wellness-rank normalized), not by the stored value: each channel's share of the ranking number is in the causal chain panel"]);
  } else if (it.goal_source !== undefined) {
    rows.push(["the goal entry", "not named: " + (it.goal_index_absent || "not recorded") + (when ? " (" + when + ")" : "")]);
  } else {
    rows.push(["the goal entry", "not recorded by this snapshot's capture (taken before 2026-09-23): the snapshot carries no ring index for the entry ranked highest" + (when ? "; " + when : "")]);
    if (it.goal_felt_value !== undefined)
      rows.push(["the ring's highest stored value at this snapshot (what the older capture took for the goal - not necessarily the planned goal)",
                 val(it.best_remembered_wellness) + (num(it.goal_curiosity_factor) !== null ? factor(it.goal_curiosity_factor) + val(it.goal_felt_value) : "")]);
  }
  if (it.goal_source !== undefined || it.goal_felt_value === undefined)
    rows.push(["the ring's highest stored value at this snapshot", it.best_remembered_wellness == null ? "—"
      : val(it.best_remembered_wellness) + " - under --wellness-rank normalized, and once the ring has moved on since the planning event, not necessarily the goal"]);
  return rows;
}

// THE PLANNER, DRAWN ON A GRID (the style guide "DIAGRAM LAYOUT ON A GRID", operator 2026-10-02: the
// old drawing's line labels ran under its boxes and past its gaps). Five columns of 168 units with
// 84-unit gutters, three rows of 60 units with 58-unit corridors between them. Every box sits in
// one cell. A horizontal line's label sits in the gutter it crosses, wrapped from its measured
// width to the gutter; a line that runs through a corridor carries its label in that corridor.
// The boxes follow the sequence of the computation (lem/model/intention_generator.py
// generate_action, mimo_bridge/lem_model.py LemLite._plan_chain):
//   row 0  this model step's input, 1 goal ring, 2 rank every entry, 3 goal tokens
//   row 1  4 starting command, 5 execution filter, 6 forward model, 7 score, 8 Adam step - and the
//          line from 8 back to 5, which runs 20 times
//   row 2  9 keep the lowest of 21, 10 finger targets, 11 size check, 12 command to the body
// Every count comes from the run: the senses and the token width from the receipt, the live
// senses from the newest snapshot's source labels, the cadence, the hold, the blend and the hand
// settings from its launch flags.
function intentionMini() {
  const A = META.args || {}, R = META.receipt || {};
  const sm = ((SNAPS[SNAPS.length - 1] || {}).header || {}).source_mask || {};
  const nTok = (R.active_modalities || Object.keys(sm)).length || 6;
  // the score's senses: measured this model step (source label real) and not motor - the rule in
  // generate_action that builds _obj
  const nCmp = Object.keys(sm).filter(m => m !== "motor" && sm[m] === "real").length;
  const nH = (R.handle_names || []).length || 24;
  const ctxPrev = Math.max(0, Number(SHAPE.context_length) - 1);
  const hold = Number(A.motor_hold) || 5;
  const holdMs = Math.round(1000 * MOMENT_S * hold);
  const hand = A.hand_target_mode === "wrist_relative_v1", handGoals = A.hand_goals === "memory";
  const normalized = A.wellness_rank === "normalized";
  const smooth = Number(A.smooth_actions) || 0;
  const newParts = Math.round(100 * smooth), oldParts = 100 - newParts;
  // the grid
  const X0 = 12, CW = 168, GX = 84, Y0 = 14, RH = 60, GY = 58, W = X0 * 2 + 5 * CW + 4 * GX;
  const cx = c => X0 + c * (CW + GX);                   // a column's left edge
  const mid = c => cx(c) + CW / 2;                      // a column's middle
  const ry = r => Y0 + r * (RH + GY);                   // a row's top edge
  const box = (r, c, l1, l2, key) => ({ x: cx(c), y: ry(r), w: CW, h: RH, l1, l2, key });
  // a horizontal line through the gutter between columns a and b of row r, its label wrapped to
  // the gutter's width (or broken where the array breaks it) and stacked upward from just above
  // the line
  const across = (r, a, b, label, key) => {
    const y = ry(r) + RH / 2;
    const ls = Array.isArray(label) ? label : wrapText(label, 9.5, GX - 10);
    const left = Math.min(a, b), gx = cx(left) + CW + GX / 2;
    return { pts: [[a < b ? cx(a) + CW : cx(a), y], [a < b ? cx(b) : cx(b) + CW, y]], key,
             label: ls, lx: gx, ly: y - 6 - 11 * (ls.length - 1) };
  };
  const boxes = [
    box(0, 0, "this model step", [nTok + " tokens × " + D_MODEL + " numbers",
                                  "+ the " + ctxPrev + " model steps before"], "node.intention"),
    box(0, 1, "1 · goal ring", ["last 200 model steps, per body:", "tokens + wellness values"], "term.goal-ring"),
    normalized
      ? box(0, 2, "2 · rank every entry", ["Σ weight × (value − mean) ÷ spread;",
                                           "over the harm line: only if all are"], "term.wellness-ranking")
      : box(0, 2, "2 · rank every entry", ["by its stored value,", "the wellness sum"], "term.goal-ring-entry"),
    box(0, 3, "3 · goal tokens", ["the top entry's tokens,", "one entry per body"], "term.goal-ring-entry"),
    box(1, 0, "4 · starting command", ["the motor controller's", "command: " + nH + " handles × 9"], "term.candidate-command"),
    box(1, 1, "5 · execution filter", ["reach and strength limits,",
      smooth > 0 ? newParts + ":" + oldParts + " blend with the last; ÷ 200" : "no blend; positions ÷ 200"], "term.execution-filter"),
    box(1, 2, "6 · forward model, frozen", ["motor token → transformer", "→ the next model step's tokens"], "term.forward-model"),
    box(1, 3, "7 · score", ["Σ " + (nCmp || "the live") + " senses: mean (pred − goal)²", "+ 1/100 × Σ strength²"], "term.intention-distance"),
    box(1, 4, "8 · Adam step", ["derivative by backpropagation,", "step size 1/10; weights fixed"], "term.adam"),
    box(2, 4, "9 · keep the lowest", ["of 21 scored commands", "(start + 20 updates), per body"], "term.candidate-command"),
    box(2, 3, "10 · finger targets", handGoals
      ? ["fitted 250 ms hand response", "aims fingers at the goal's pose"]
      : ["skipped: --hand-goals " + (A.hand_goals || "off")], "term.hand-goal"),
    box(2, 2, "11 · size check", ["forward model once more: RMS", "½ to 2× the real, or no plan"], "term.scale-check"),
    box(2, 1, "12 · command to the body", ["latched every " + hold + " model steps,", "held " + holdMs + " ms"], "term.motor-hold"),
  ];
  const corridor0 = ry(0) + RH + GY / 2, corridor1 = ry(1) + RH + GY / 2;
  const arrows = [
    // this model step's tokens and context into the forward model, through the first corridor
    { pts: [[mid(0), ry(0) + RH], [mid(0), corridor0], [mid(2), corridor0], [mid(2), ry(1)]],
      label: "this model step's tokens + context, " + nTok + " × " + D_MODEL + " each",
      lx: mid(1), ly: corridor0 - 5, key: "term.forward-model" },
    across(0, 1, 2, "200 entries", "term.goal-ring-entry"),
    across(0, 2, 3, "the top entry", normalized ? "term.wellness-ranking" : "term.goal-ring-entry"),
    // the goal tokens down into the score
    { pts: [[mid(3), ry(0) + RH], [mid(3), ry(1)]], label: ["goal: " + nTok + " × " + D_MODEL, "per body"],
      lx: mid(3) + 6, ly: corridor0 - 5, anchor: "start", key: "term.goal-ring-entry" },
    across(1, 0, 1, nH + " × 9", "term.candidate-command"),
    across(1, 1, 2, "the filtered command", "term.execution-filter"),
    across(1, 2, 3, ["predicted", nTok + " × " + D_MODEL], "term.predicted-outcome"),
    across(1, 3, 4, ["one number", "per body"], "term.gradient"),
    // the loop: the updated command back through the filter, under row 1
    { pts: [[cx(4) + CW / 4, ry(1) + RH], [cx(4) + CW / 4, corridor1], [mid(1), corridor1], [mid(1), ry(1) + RH]],
      label: "the updated command, back through the filter: 20 updates",
      lx: (mid(1) + cx(4) + CW / 4) / 2, ly: corridor1 - 5, key: "term.adam" },
    { pts: [[cx(4) + CW * 3 / 4, ry(1) + RH], [cx(4) + CW * 3 / 4, ry(2)]], label: ["21 scored", "commands"],
      lx: cx(4) + CW * 3 / 4 - 6, ly: corridor1 - 5, anchor: "end", key: "term.intention-distance" },
    across(2, 4, 3, "the kept command", "term.candidate-command"),
    across(2, 3, 2, hand ? "+ finger angles" : "the kept command", "term.hand-goal"),
    across(2, 2, 1, "the plan's first command", "term.scale-check"),
  ];
  const depth = hand
    ? "plan depth: one command, because --hand-target-mode is wrist_relative_v1"
    : A.physical_goals === "memory"
      ? "plan depth: one command for a goal with a recorded pose (--physical-goals memory); otherwise as below"
      : A.plan_horizon === "earned"
        ? "plan depth (--plan-horizon earned): 1 + the leading depths the imagination chain has licensed, at most --imagination-steps; each further depth repeats 4 to 11 from the predicted tokens"
        : "plan depth: one command (--plan-horizon " + (A.plan_horizon || "fixed") + ")";
  return flowDiagram(W, ry(2) + RH + 8, boxes, arrows, [
    "the model's weights take no gradient during the search and are restored after it; only the command numbers"
      + (hand ? " and the finger joint angles" : "") + " change",
    "the score's strength part reads the command's strength before the filter's limit; one hundredth of the sum over the "
      + nH + " handles of strength squared" + (A.planner_effort_per_body === "on" ? ", per body" : ", times the number of bodies under --planner-effort-per-body off"),
    depth,
    "no plan kept (the goal ring under 20 entries, or the size check failed): the body gets the motor controller's command, with babble noise when the noise is on",
  ], false);
}

// ------------------------------------------------------------------ verdict tiles
// The page's hierarchy: the questions below, in this sequence, ARE the project's
// goals. Every verdict is structural - a comparison to computed chance, to zero,
// or a first-vs-last-quarter-window trend (the dashboard's own convention,
// including its ±2% "about the same" band) - never an invented threshold.
function seriesByKey(key) {
  for (const ch of SERIES.charts || [])
    for (const s of ch.series || [])
      if (s.key === key) return s.points;
  return null;
}
function afterLastSeam(pts) {
  // the dashboard's rule: a trend window must not straddle a process restart -
  // one log can hold several configurations (H30's imagination flip lesson)
  const seams = SERIES.seams || [];
  if (!pts || !seams.length) return pts;
  const cut = seams[seams.length - 1];
  const f = pts.filter(p => p[0] >= cut);
  return f.length >= 2 ? f : pts;
}
function winStats(ptsRaw) {
  const pts = afterLastSeam(ptsRaw);
  if (!pts || pts.length < 2) return null;
  const k = Math.max(1, Math.min(100, Math.floor(pts.length / 4)));
  const mean = a => a.reduce((x, p) => x + p[1], 0) / a.length;
  return { a: mean(pts.slice(0, k)), b: mean(pts.slice(-k)), k, last: pts[pts.length - 1][1] };
}
function verdict(a, b, dir, res) {   // dir "error": smaller is better; "higher": bigger is; res = the series' own resolution
  if (res && Math.abs(a) < res) return { txt: "AT ZERO - TOO SMALL TO CALL", cls: "same", pct: null };
  const pct = 100 * (b - a) / (Math.abs(a) || 1e-9);
  if (Math.abs(pct) <= 2) return { txt: "ABOUT THE SAME", cls: "same", pct };
  const improving = dir === "error" ? pct < 0 : pct > 0;
  return { txt: improving ? "IMPROVING" : "GETTING WORSE", cls: improving ? "good" : "bad", pct };
}
function tile(status, title, big, sub, pic) {
  return el("div", { class: "tile " + status },
    el("div", { class: "tile-title", text: title }),
    el("div", { class: "big", text: big }),
    pic || null,
    el("div", { class: "tile-sub", text: sub }));
}
function renderTiles(selSnap) {
  const wrap = el("div", {});
  // PROVENANCE BELONGS WITH THE TILES, not only on the run tab. The tiles read
  // SERIES, which is built from META.stats_source; when that is a different run
  // from META.run the tiles describe the TRAINING run, not the snapshots below
  // them - and this strip renders on the inspector tab too.
  const otherRun = META.stats_source && META.stats_source !== META.run;
  const seams = SERIES.seams || [];
  const note = el("p", { class: "note" });
  note.appendChild(document.createTextNode(
    "The questions that matter, in order of importance. "
    + (otherRun
       ? "Every verdict below is computed from the TRAINING run " + META.stats_source
         + "'s own stats - the history these weights lived - and NOT from "
         + META.run + ", the run whose snapshots the inspector shows. The one exception "
         + "is the last sentence of tile 5, which reads a snapshot of " + META.run
         + " and names it. "
       : "Every verdict below is computed from this run's own stats. ")
    + "Trends compare the first and last quarter-windows of the current process "
    + (seams.length
       ? "segment, which starts after the last restart at step " + seams[seams.length - 1] + "."
       : "segment; this log records no restart, so that segment is the whole log "
         + "and a configuration change inside it is averaged away.")));
  note.appendChild(docBtn("tiles", "the verdict tiles"));
  wrap.appendChild(note);
  const row = el("div", { class: "tiles" });

  const r1 = winStats(seriesByKey("tenure_tok.rank1"));
  // THE CHART PUBLISHES THE CHANCE LINE, 1/(1+decoys), under its own key. This tile
  // used to look up "tenure_tok.n_decoys" and treat what it found as a decoy COUNT;
  // once the chart plotted the chance line under that key the tile computed
  // 1/(1 + 0.0667) = 0.94 and printed "~0 decoys. Guessing would score 94%".
  const ch = winStats(seriesByKey("tenure_tok.chance"));
  if (r1 && ch && ch.b > 0) {
    const chance = ch.b;
    const decoys = 1 / chance - 1;
    const above = r1.b > chance;
    row.appendChild(tile(above ? "good" : "warn",
      "1 · Does the command explain the outcome?", Math.round(r1.b * 100) + "%",
      `of command-endings, the model picks its OWN outcome out of ~${Math.round(decoys)} decoys. ` +
      `Guessing would score ${Math.round(chance * 100)}%. ` +
      (above ? "Above guessing - some self-recognition is measured."
             : "At or below guessing - no self-recognition measured yet. This is the " +
               "number the project exists to move.")));
  } else {
    row.appendChild(tile("na", "1 · Does the command explain the outcome?", "—",
      "No command-to-outcome events in this run's log. An eval probe grades none; " +
      "open the TRAINING run's page for this number - it is the one the project " +
      "exists to move."));
  }

  const cz = winStats(seriesByKey("causal_pe"));
  const sp = winStats(seriesByKey("movement.local_speed"));
  if (cz) {
    // causal_pe is a mean of squares against tokens held at RMS 1, so it is written as a
    // percentage of the error an answer of all zeros makes (DESIGN.md section 11)
    const v = verdict(cz.a, cz.b, "error");
    row.appendChild(tile(v.cls, "2 · Next-model-step error", pctNum(100 * cz.b),
      `${v.txt}: ${pctNum(100 * cz.a)} → ${pctNum(100 * cz.b)} of the error an answer of all zeros makes ` +
      `(${v.pct > 0 ? "+" : ""}${Math.round(v.pct)}% change; 0% = exact, 100% = no better than answering zeros). ` +
      "This is an error - smaller is better: how wrong the model was about each " +
      "model step, given the last model step and its own command." +
      (sp ? ` Body speed went ${fmt(sp.a, 1)} cm/s → ${fmt(sp.b, 1)} cm/s; a stiller body ` +
            "is easier to predict, so read the two together." : "")));
  }

  const ex = winStats(seriesByKey("exec_frac"));
  const cc = winStats(seriesByKey("cmd_response_cos_held")) || winStats(seriesByKey("cmd_response_cos"));
  // the chart publishes the commanded turn in DEGREES (build_explainer._deg); a page built
  // before that carries the radian series, converted here
  const cmDeg = winStats(seriesByKey("cmd_mag_deg"));
  const cmRad = cmDeg ? null : winStats(seriesByKey("cmd_mag_rad"));
  const cm = cmDeg || (cmRad ? { a: cmRad.a * 180 / Math.PI, b: cmRad.b * 180 / Math.PI,
                                 k: cmRad.k, last: cmRad.last * 180 / Math.PI } : null);
  if (ex || cc || cm) {
    // THE SCALE HAS THREE ENDS, not two: this is a cosine and it goes negative. A cosine is
    // never printed (DESIGN.md section 11): the tile draws where it sits between its three named
    // ends and says which end it is nearest.
    const CMD_ENDS = [[-1, "exactly against the command"], [0, "unrelated to the command"], [1, "exactly as commanded"]];
    row.appendChild(tile(cc && cc.b < 0 ? "warn" : "info", "3 · Is the action pathway alive?",
      cc ? (cc.b < 0 ? "against the command" : "direction match")
         : (ex ? Math.round(ex.last * 100) + " of 100" : "—"),
      (cc ? "Where the limbs moved against where the command pointed, averaged over the window: " +
            "the marker sits between 'exactly against the command', 'unrelated to the command' " +
            "(the middle) and 'exactly as commanded', " + nearestEnd(cc.b, CMD_ENDS) + ". " +
            (cc.b < 0 ? "This is BELOW THE MIDDLE: on average the limbs are moving against " +
                        "the direction the command pointed, which is what a wrong sign " +
                        "convention in the action decoder, or a command written in a " +
                        "different frame from the pose that comes back, looks like. " +
                        "It is not a weak version of obeying. " : "") : "") +
      (cm ? `Commands ask for ${Math.round(cm.b)}° of joint turn on average ` +
            "(commands under ~3° once produced no visible movement at all). " : "") +
      (ex ? `And ${Math.round(ex.last * 100)}% of steps handed their executed command ` +
            "back to the model - the ledger contract, complete at 100%." : ""),
      cc ? picture(cc.b, -1, 1, "against ←", "→ as commanded") : null));
  }

  // THE QUIETEST SENSE MEANS EVERY SENSE. A literal four-sense list here could
  // not see audio or intermind, and intermind is the quietest encoder in the run.
  let minV = null, minM = null, synthV = null, synthM = null;
  const tileSrc = SNAPS.length ? (SNAPS[SNAPS.length - 1].header.source_mask || {}) : {};
  // every collapse series the page CARRIES - not the receipt's list with a four-sense
  // fallback, which is all a run has until its first checkpoint writes a receipt
  const encMods = [];
  for (const chx of SERIES.charts || []) for (const sx of chx.series || [])
    if (sx.key && sx.key.indexOf("token_temporal_std.") === 0)
      encMods.push(sx.key.slice("token_temporal_std.".length));
  for (const m of encMods) {
    const w = winStats(seriesByKey("token_temporal_std." + m));
    if (!w) continue;
    if (tileSrc[m] === "synthetic") {
      if (synthV === null || w.b < synthV) { synthV = w.b; synthM = m; }
    } else if (minV === null || w.b < minV) { minV = w.b; minM = m; }
  }
  if (minV !== null)
    row.appendChild(tile("info", "4 · Are the encoders alive?", pctNum(100 * minV),
      `the quietest MEASURED sense's (${minM}) token variation over time, as a percentage of ` +
      "a real token's size. 0% would " +
      "mean that encoder emits the same token every step regardless of what the body " +
      "senses - collapse, the failure the ring watches for." +
      (synthM ? ` The ${synthM} channel sits lower still at ${pctNum(100 * synthV)}: its encoder ` +
                "learned to emit one fixed token for a placeholder draw whose average never " +
                "changes (measured on the H83 encoder: the token moves 0.06% of a token's " +
                "size, where an untrained encoder on the same draw moves 59.5%), so it " +
                "is left out of this verdict on that stated ground, not left out of the search." : "")));

  // TILE 5'S SNAPSHOT SENTENCE FOLLOWS THE VIEW IT SITS IN. On the inspector tab this
  // row is rendered under the snapshot picker, so the sentence describes the snapshot
  // the reader selected (renderInspector passes it in); on the run tab there is no
  // picker and it describes the newest one. It always names the step, and when the
  // charts come from another run it names the run the snapshot belongs to - the speed
  // beside it is the chart run's, and nothing used to tell the two apart.
  const sN = selSnap || (SNAPS.length ? SNAPS[SNAPS.length - 1] : null);
  const sWhich = sN ? (selSnap ? "The snapshot selected above" : "Newest snapshot")
    + " (step " + sN.header.step + (otherRun ? " of " + META.run : "") + ")" : "";
  if (sp || sN)
    row.appendChild(tile("info", "5 · Is the body doing anything?",
      sp ? fmt(sp.b, 1) + " cm/s" : "—",
      (sp ? "recent body speed" + (otherRun ? " in " + META.stats_source : "") + ". " : "") +
      (sN ? `${sWhich}: ${sN.senses.contacts} skin contacts, phase ` +
            `${sN.header.phase}, age ${fmt(sN.header.age_months, 0)} months.` : "")));

  wrap.appendChild(row);
  return wrap;
}

// ------------------------------------------------------------------ tabs
const TABS = [["arch", "1 · Architecture"], ["inspect", "2 · Instance inspector"], ["run", "3 · Run replay"],
              ["probes", "4 · Probes and liveness"]];
function switchTab(t) {
  state.t = t; writeHash();
  for (const [id] of TABS) {
    document.getElementById("tab-" + id).classList.toggle("hidden", id !== t);
    document.getElementById("btn-" + id).classList.toggle("active", id === t);
  }
}
function openSnapshot(step) {
  let best = null, bd = Infinity;
  SNAPS.forEach((s, i) => { const d = Math.abs(s.header.step - step); if (d < bd) { bd = d; best = i; } });
  if (best === null) return;
  state.s = String(SNAPS[best].header.step); writeHash();
  renderInspector();
  switchTab("inspect");
}

// ================================================================== TAB 1: diagram
// Layout is designed on a grid of fixed columns and reserved corridors ("lanes"),
// so no line ever crosses a box: solid edges flow left-to-right between adjacent
// columns; every between-step (dashed) edge is hand-routed through the empty
// gutter under the core and the channels between the strip boxes.
// 50 holds the caption (12 units), one line of sub-caption (10 units) and the corner count (9 units), each on
// a line of its own with a unit or more between their measured text boxes and 1.5 units or more to the
// border (node() below). One size step larger than the 11, 9 and 8 units of a 44-unit box until Oct 7, 2026
// (the decided design of the Architecture tab, D-E6).
const NW = 118, NH = 50;
// a box's text keeps this many units inside its left and right edges (the layout check's least is 3)
const BOX_PAD = 4;
// the sub-caption's font size and the distance between its lines, in the drawing's units: measured in headless Edge at
// one unit per pixel (Oct 7, 2026), a 10-unit line's text box reaches 11 above its baseline and 3 below it (14 tall),
// so lines drawn as text elements of their own (a box's extra lines) sit 14 apart and their boxes do not overlap; at
// 13 apart the layout check measured an overlap of 1 unit and a rounding residue, over its slack of 1
const SUB_PX = 10, SUB_STEP = 14;
// the lines a box's sub-caption takes at SUB_PX units in a box ww units wide (an array of strings breaks
// where the array breaks, and each piece is wrapped again if it is still too wide), and the height
// the box needs; `extra` counts the lines a box adds under its sub-caption (buildDiagram's boxExtras)
function boxSubLines(sub, ww) {
  return sub ? [].concat(...[].concat(sub).map(s => wrapText(s, SUB_PX, ww - 2 * BOX_PAD))) : [];
}
function boxHeight(sub, ww, h, extra) {
  const n = boxSubLines(sub, ww).length + (extra || 0);
  return Math.max(h || NH, n ? NH + SUB_STEP * (n - 1) : 0);
}
// a block's name as the defect register and the code write it (blocks.0 is "block 0"), with its place in the
// stack: "block 0 (1st of 4)". The frames printed "block 1 / 4" and the inspector "layer 1" for the same block
// until Oct 7, 2026 (the review's finding F11).
function ordinalWord(n) {
  const t = n % 100, e = n % 10;
  return n + (t >= 11 && t <= 13 ? "th" : e === 1 ? "st" : e === 2 ? "nd" : e === 3 ? "rd" : "th");
}
function blockName(i, n) {
  return "block " + i + " (" + ordinalWord(i + 1) + (n ? " of " + n : "") + ")";
}
const NODE_PARAM = { heads: "prediction_heads", recon: "recon_decoders",
                     motor: "motor_controller", wellness: "wellness",
                     memory: "persistent_memory", self_state: "self_state",
                     td_gate: "td_gates", voice: "voice_head",
                     // the gain-of-function build's trainable side modules (2026-09-22)
                     efference: "efference_heads", contact_head: "contact_head",
                     synergy: "motor_synergy",
                     // The shared output LayerNorm holds learned numbers only when the run
                     // set encoder_defaults.output_layernorm_affine (a gain and a bias, two
                     // numbers per unit of token width). The model receipt files them under
                     // enc_shared; the build's count of a checkpoint's saved tensors files
                     // them under encoder_registry. A list means: whichever is present.
                     ln_out: ["enc_shared", "encoder_registry"] };
// REPAIR PATHS DRAWN INSIDE THEIR HOST BOX (the decided design of the Architecture tab, Oct 7, 2026, D-E4): box id ->
// the component_params keys of the repair paths a run builds inside that box. The host box claims each key it finds
// in the run's receipt (so the not-drawn list drops it) and prints it as a line of its own under its sub-caption, with
// the path's count, opening the path's own entry (HOSTED_DOC: the words the line starts with, the entry's key). The
// box's corner count stays the box's own module (NODE_PARAM).
const HOSTED_PARAM = { enc_motor: ["motor_asked_change"], heads: ["heads_all_slot_path"] };
const HOSTED_DOC = { motor_asked_change: ["+ asked-change path", "path.motor_asked_change"],
                     heads_all_slot_path: ["+ all-slot path", "path.heads_all_slot_path"] };
// A box whose weights are real but never move: the corner count is true, the
// lesson is absent, so the box draws plain and the count says why. The voice head is
// called every model step with voice_target=None (mimo_bridge/lem_model.py) and its
// configured loss weight is 0.0, so loss_manager adds no term at all; with a target
// forced in, every gradient into the head still measures exactly 0.0. Measured on
// this lineage's checkpoint: the head's LayerNorm gain is still exactly 1.0 and its
// bias exactly 0.0, and every weight is still inside its initialization bound.
const NO_LESSON = { voice: "never updated" };
// Boxes that own part of a LARGER component's count. token assembly holds the
// modality and model step embeddings and final LayerNorm holds its own gain and
// bias, both inside forward_model, so neither appeared in component_params and
// both drew plain while the legend promised bold for anything carrying weights.
// [component in the receipt's inventory, tensor-name prefixes this box owns]
const SUB_PARAM = {
  assembly:   ["forward_model", ["modality_embedding", "temporal_embedding"]],
  final_norm: ["forward_model", ["final_norm"]],
};
// The same for every stage box inside a block: block 0's cross-modal stage owns the
// tensors named blocks.0.cm_*, and so on. Without these eight entries the stage boxes
// drew thin and countless under a legend promising bold for anything carrying weights,
// and the whole core's figure had nowhere to go but block 1's frame.
const BLOCK_SUBPARAM = { s1: "cm_", s2: "ct_", s3: "cmb_", ffn: "ffn" };
function subParamSpec(id) {
  const b = id.match(/^block(\d+)_(s1|s2|s3|ffn)$/);
  if (b) return ["forward_model", ["blocks." + b[1] + "." + BLOCK_SUBPARAM[b[2]]]];
  return SUB_PARAM[id];
}
function pfmt(n) {
  return n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1000 ? Math.round(n / 1000) + "k" : String(n);
}
// ------------------------------------------------------------------ the gated parts' switches (Oct 15, 2026)
// The operator, Oct 15, 2026: "Things that are behind gates should have a switch to show them on/off with the default
// being whatever is current in the code." A gated part is a box or a line the model diagram draws only when a launch
// setting (META.args, the run's args.json) or a module the run built (META.receipt, the run's model receipt) says so.
// Each has a switch above the drawing. Its default is this run's value as this checkout's code read it when the page
// was built; a part shown although this run has it off is drawn faint and dashed (class offrun), so it is never read as
// live, and a part switched off is not drawn. The defaults follow the run and the code through the page's builder:
// build_explainer --watch rebuilds the page whenever the run's settings or the explainer's own code change, and the
// panel's "load the newest build" button reloads the page.
//   id, the switch's words, the setting that decides it in a run, the boxes it draws, the lines it draws
const GATES = [
  ["td_gate", "top-down gate", "--top-down-gate", ["td_gate"], []],
  ["memory", "persistent memory", "--persistent-memory", ["memory"], []],
  ["self_state", "self perception", "--self-state", ["self_state", "self_learning", "self_report"], []],
  ["voice", "voice head", "the receipt's voice_head.enabled", ["voice"], []],
  ["hself", "legacy self tracker", "--hierarchical-self", ["hself"], []],
  ["entity_self", "named hierarchical self", "--entity-self", ["entity_self"], []],
  ["caregiver", "caregiver", "--caregiver-mode", ["caregiver"], []],
  ["efference", "efference heads", "--efference-heads (the receipt's efference_heads)", ["efference"], []],
  ["eff_gaze_copy", "eye command into the vision prediction (gaze copy)",
   "the efference stamp's gaze_copy_into_vision", [], [["efference", "heads"]]],
  ["eff_goal", "efference heads into the goal ranking",
   "--wellness-rank normalized with --controllability or --learning-progress on", [], [["efference", "wellness"]]],
  ["eff_search", "body head in the planner's search", "--physical-goals memory", [], [["efference", "intention"]]],
  ["contact", "contact head", "--contact-head (the receipt's contact_head)", ["contact_head"], []],
  ["synergy", "motor synergy and its completed movements", "--motor-synergy (the receipt's motor_synergy)",
   ["synergy", "synergy_examples"], []],
  ["syn_plan", "synergy as the planner's action space", "--synergy-planner", [], [["synergy", "intention"]]],
  ["syn_babble", "synergy babble into the world", "--synergy-babble", [], [["synergy", "world"]]],
  ["imagination", "imagination chain", "--imagination", [],
   [["ln_out", "imagination"], ["motor", "imagination"], ["td_gate", "imagination"]]],
  ["imag_depth", "imagination sets the planner's depth", "--hand-target-mode world", [], [["imagination", "intention"]]],
  ["curriculum_world", "phase change rebuilds the world's body",
   "--start-phase, --age-binding and --delay-binding (a run that starts in the final phase with its age fixed and its "
   + "delays off can never change phase)", [], [["curriculum", "world"]]],
  ["attention_cap", "attention cap (its line inside the stage-1 and motor-encoder boxes)", "--attention-cap", [], []],
  // the input check of Oct 15, 2026: with --residual-prediction on, each prediction head's output is added to the raw
  // token of its sense (lem/model/prediction_heads.py PredictionHeads.forward), so the shared LayerNorm's tokens reach
  // the heads on a line of their own
  ["residual", "raw tokens added to the prediction heads' output (residual)", "--residual-prediction", [],
   [["ln_out", "heads"]]],
  // the independent review of Oct 15, 2026 (S1, m12): the planner's hand stage (and its physical goal) reads the chosen
  // goal ring entry's stored handle pose, and the goal ring's value adds the world's count of consequence kinds
  ["hand_goal_pose", "the motor sense's stored pose into the planner's goal", "--hand-goals memory or --physical-goals memory",
   [], [["sense_motor", "intention"]]],
  ["consequence", "the world's consequence count into the goal ring's value", "--consequence-w above 0", [],
   [["world", "intention"]]],
];
const GATE_SET = {};      // the switches the reader changed: id -> true or false
const GATE_RUN = {};      // this run's value of each gate, recorded as the diagram is drawn
function gate(id, runOn) {
  GATE_RUN[id] = !!runOn;
  return Object.prototype.hasOwnProperty.call(GATE_SET, id) ? GATE_SET[id] : !!runOn;
}
// the panel of switches between the tour and the drawing, written again after every change
function buildGatePanel() {
  const diagram = document.getElementById("diagram");
  if (!diagram || !diagram.parentNode) return;
  let box = document.getElementById("gate-panel");
  const wasOpen = !!(box && box.open);
  if (!box) {
    box = el("details", { id: "gate-panel", class: "gate-panel" });
    diagram.parentNode.insertBefore(box, diagram);
  }
  box.textContent = "";
  box.open = wasOpen;
  const rows = GATES.filter(g => Object.prototype.hasOwnProperty.call(GATE_RUN, g[0]));
  // a switch counts as changed only when it differs from this run's value
  const differs = id => Object.prototype.hasOwnProperty.call(GATE_SET, id) && GATE_SET[id] !== GATE_RUN[id];
  const changed = rows.filter(g => differs(g[0])).length;
  box.appendChild(el("summary", { text: "switches: " + rows.length + " parts drawn only when a setting is on"
    + (changed ? " · " + changed + " set differently from this run" : " · all as in this run") }));
  box.appendChild(el("p", { class: "gate-note", text: "Each switch starts at this run's value, as this checkout's code "
    + "read it from the run's settings when the page was built (" + (META.generated_label || META.generated_at || "build "
    + "time not recorded") + ", run " + (META.run || "not recorded") + "). A part shown although this run has it off is "
    + "drawn faint and dashed. The page's builder, run with --watch, rebuilds the page when the run's settings or the "
    + "explainer's code change; \"load the newest build\" reloads it." }));
  const table = el("table", { class: "gate-table" });
  table.appendChild(el("tr", {}, el("th", { text: "drawn" }), el("th", { text: "part" }),
                       el("th", { text: "in this run" }), el("th", { text: "decided by" })));
  for (const [id, words, setting] of rows) {
    const on = Object.prototype.hasOwnProperty.call(GATE_SET, id) ? GATE_SET[id] : GATE_RUN[id];
    const cb = el("input", { type: "checkbox", id: "gate-" + id });
    cb.checked = on;
    cb.addEventListener("change", () => {
      if (cb.checked === GATE_RUN[id]) delete GATE_SET[id]; else GATE_SET[id] = cb.checked;
      redrawWithGates();
    });
    const tr = el("tr", { class: differs(id) ? "gate-changed" : "" },
                  el("td", {}, cb), el("td", {}, el("label", { for: "gate-" + id, text: words })),
                  el("td", { text: GATE_RUN[id] ? "on" : "off" }), el("td", { text: setting }));
    table.appendChild(tr);
  }
  box.appendChild(table);
  const reset = el("button", { type: "button", text: "reset to this run's settings" });
  reset.addEventListener("click", () => { Object.keys(GATE_SET).forEach(k => delete GATE_SET[k]); redrawWithGates(); });
  const reload = el("button", { type: "button", text: "load the newest build" });
  reload.addEventListener("click", () => location.reload());
  box.appendChild(el("div", { class: "gate-buttons" }, reset, reload));
}
function redrawWithGates() {
  buildDiagram();
  buildGatePanel();
  if (typeof renderTour === "function") renderTour();
}

// The reviewed source map supplements omitted launch fields only for its named run.
function selfReviewedSetting(name) {
  if (ARGS[name] !== undefined) return ARGS[name];
  const d = DOC["guide.self_architecture"] || {};
  return d.reviewed_run === META.run ? (d.reviewed_args || {})[name] : undefined;
}
function mountSelfGuide() {
  if (document.getElementById("self-guide")) return;
  const d = DOC["guide.self_architecture"];
  const anchor = document.getElementById("learnloop") || document.getElementById("model-diagram");
  if (!d || !anchor) return;
  const holder = document.createElement("div"); holder.innerHTML = d.html;
  anchor.parentNode.insertBefore(holder, anchor);
  initSelfGuide(holder);
}
function initSelfGuide(root) {
  root.querySelectorAll('.sg-table-scroll table').forEach(table => {
    const labels = [...table.querySelectorAll('thead th')].map(th => th.textContent);
    table.querySelectorAll('tbody tr').forEach(row => [...row.children].forEach((td, i) => td.dataset.label = labels[i] || ''));
  });
  root.querySelectorAll('[data-spec-link]').forEach(a => a.href = location.pathname.replace(/\\/g, '/').includes('/Saved/mimo_bridge/') ? '../../docs/UEMA/UEMA_Spec_v3_6.html' : 'UEMA/UEMA_Spec_v3_6.html');
  root.querySelectorAll("[data-self-topic]").forEach(button => {
    button.addEventListener("click", () => {
      root.querySelectorAll("[data-self-topic]").forEach(b => b.setAttribute("aria-pressed", String(b === button)));
      root.querySelectorAll("[data-self-panel]").forEach(p => p.hidden = p.dataset.selfPanel !== button.dataset.selfTopic);
    });
  });
  const note = root.querySelector("[data-self-run-summary]");
  if (note) note.textContent = "Displayed saved run: " + META.run + ". Self perception: "
    + ((META.receipt || {}).self_state || ARGS.self_state || "not recorded")
    + "; self-choice: " + (ARGS.self_choice || "off")
    + ". Latent-body: " + (selfReviewedSetting("latent_body_hierarchy") || "not recorded")
    + "; latent planner: " + (selfReviewedSetting("latent_planner") || "not recorded")
    + ". This describes saved settings, not a running process.";
}

function buildDiagram() {
  mountSelfGuide();
  const root = document.getElementById("diagram");
  root.textContent = "";
  const rec = META.receipt || {}, fm = rec.forward_model || {};
  const CP = META.component_params || {};
  const mods = rec.active_modalities || [];
  const nL = N_BLOCKS;
  const argsm = META.args || {};
  const snap = SNAPS.length ? SNAPS[SNAPS.length - 1] : null;
  const src = snap ? (snap.header.source_mask || {}) : {};
  // A run that starts in the final phase with its age fixed and its delays off can never change phase (train_mimo's
  // defaults: --age-binding curriculum, --delay-binding on), so its curriculum draws no line to the world
  const finalFixed = argsm.start_phase === "mature" && /^fixed/.test(argsm.age_binding || "")
                     && (argsm.delay_binding || "on") === "off";
  // THE GATED PARTS (GATES, above): this run's value of each, or the reader's switch where one is set
  const EFFR = META.efference_routes || {};
  const G = {
    td_gate: gate("td_gate", (argsm.top_down_gate || "off") !== "off"),
    memory: gate("memory", (rec.persistent_memory || argsm.persistent_memory) === "on"),
    self_state: gate("self_state", (rec.self_state || argsm.self_state || "off") !== "off"),
    voice: gate("voice", !!(rec.voice_head || {}).enabled),
    hself: gate("hself", !!argsm.hierarchical_self && argsm.hierarchical_self !== "off"),
    entity_self: gate("entity_self", !!argsm.entity_self && argsm.entity_self !== "off"),
    caregiver: gate("caregiver", !!argsm.caregiver_mode && argsm.caregiver_mode !== "off"),
    efference: gate("efference", !!rec.efference_heads),
    eff_gaze_copy: gate("eff_gaze_copy", !!(rec.efference_heads || {}).gaze_copy_into_vision),
    eff_goal: gate("eff_goal", !!rec.efference_heads && (EFFR.goal_channels || []).length > 0),
    eff_search: gate("eff_search", !!rec.efference_heads && !!EFFR.action_search),
    contact: gate("contact", !!rec.contact_head),
    synergy: gate("synergy", !!rec.motor_synergy),
    syn_plan: gate("syn_plan", !!rec.motor_synergy && rec.motor_synergy.planner !== false),
    syn_babble: gate("syn_babble", !!(rec.motor_synergy || {}).babble),
    imagination: gate("imagination", imagOn()),
    imag_depth: gate("imag_depth", (argsm.hand_target_mode || "world") === "world"),
    curriculum_world: gate("curriculum_world", !finalFixed),
    attention_cap: gate("attention_cap", (argsm.attention_cap || "off") === "on"),
    // train_mimo --residual-prediction, default "off" (train_mimo.py sets model.residual_prediction from it)
    residual: gate("residual", (argsm.residual_prediction || "off") === "on"),
    // lem_model.py: the planner's decision carries the selected entry's pose under either flag (physical_goal,
    // hand_goal); the goal ring's value adds consequence_weight x consequence_kinds (train_mimo --consequence-w, 0.0
    // by default)
    hand_goal_pose: gate("hand_goal_pose", argsm.hand_goals === "memory" || argsm.physical_goals === "memory"),
    consequence: gate("consequence", (Number(argsm.consequence_w) || 0) > 0),
  };
  // What each sense hands its encoder, short enough for the picture to carry. The
  // full contract (unit, range, rate, whether gradient crosses) is the edge.* entry
  // the line itself opens; a diagram that names no contract at all was the 2026-09-15
  // audit's finding. Counts come from the receipt so they follow the run.
  const nBody = (snap && snap.header && snap.header.n_envs) || argsm.n_envs || 1;
  const nHandles = (rec.handle_names || []).length || 24;
  const wideP = Math.pow(256 / 16, 2), fovP = argsm.fovea_size ? Math.pow(argsm.fovea_size / 16, 2) : 0;
  // the tactile row's width is the capture's own column legend (senses.touch_cloud_doc), never a
  // typed number: it has been widened twice (12 -> 14 -> 19) and the caption said 12 until
  // 2026-09-22
  const touchW = (() => {
    const s = SNAPS.map(x => ((x || {}).senses || {}).touch_cloud_doc).filter(d => typeof d === "string").slice(-1)[0];
    const lg = s ? touchLegend(s) : [];
    return lg.length ? lg[lg.length - 1].start + lg[lg.length - 1].span : null;
  })();
  const vstRec = rec.vestibular_sense || {};
  const WIRE = {
    vision: (2 * (wideP + fovP)) + " × 772: pixels 0-1 plus 4 tags",
    tactile: "32 rows × " + (touchW || "the capture's") + " scaled fields, hip-relative",
    audio: "514: 256 bands × 2 ears + 2 differences",
    proprioception: nHandles + " rows × 12, wrist-relative positions",
    motor: nHandles + " rows × 19: achieved 10 + command 9, world positions",
    intermind: "5 rows × " + SHAPE.intermind_row + " placeholder noise",
    vestibular: (vstRec.dim || 29) + ": balance, neck, support, effort, clipped ±" + (vstRec.input_bound || 4),
  };
  // THE GRID (the style guide "DIAGRAM LAYOUT ON A GRID", 2026-10-02; relaid out on Oct 7, 2026 for the decided design of
  // the Architecture tab, D-E4 to D-E6). Columns, left to right: the senses (x 32, 96 wide) and their encoders (x 138,
  // 138 wide) with each line's contract centred under the pair (WIRE_W wide, from x 32 to 276); the shared LayerNorm
  // and the top-down gate (x 300, 106 wide); the token assembly (x 434, 130 wide); the block frames (x 590, 200 wide,
  // the boxes and cells inside at x 602, 184 wide, the line between two blocks down the gutter at x 595); the final
  // LayerNorm (x 806, 118 wide); the outputs (x 972, 150 wide); the side heads (x 1290, 150 wide); under the core the
  // line corridors and the two rows of strip boxes (x 60 + 240 per box, 210 wide). Rows: a sense row is its box (the
  // taller of the sense box and its encoder box), its contract's lines and room above the next row's numbered circles;
  // block 0's frame holds its two label lines, its four boxes 10 apart and its closing caption; each later block's frame
  // holds its two label lines and one row of four cells. Every box's text is laid out from its measured width in node().
  // the senses (x 32, SENSE_W wide) and their encoders (x 138, ENC_W wide): an encoder's title is a code name as long
  // as "cross_handle_attention" (121 units at 12), and its numbered circle reaches 7 units into the box's top-left
  // corner, so the box keeps 8 units on each side of the title; the line's contract is centred under the pair
  // The sense boxes start at x 32, 96 wide (Oct 15, 2026). The world's trunk leaves the world box's bottom edge at
  // x 20 and runs down to a 12-unit line into each sense box's left edge. LEFT of the trunk run two corridors, x 4
  // (the comb "tactile-raw") and x 12 (the comb "motor-raw"), each carrying one sense's raw readings to the boxes that
  // read them without an encoder (wellness, the motor controller, persistent memory, self perception, the planner,
  // replay): each crosses the trunk itself and no arrow of the world's lines (the layout check's "line crosses an
  // arrowhead"), and each branch shows its own sense (the independent recheck of Oct 15, 2026, P2-M1: one shared
  // corridor carried both senses' branches, and a reader could not tell their sources apart). The sense box keeps the
  // 96 units "proprioception" needs beside its numbered circle; the 10-unit gap to the encoder holds a straight line at
  // the sense box's middle. The contract under a row is centred under the pair from x 32 to the encoder's right edge, so
  // no corridor runs through its text
  const SENSE_X = 32, SENSE_W = 96, ENC_W = 138, WIRE_W = 138 + ENC_W - SENSE_X, WIRE_X = SENSE_X + WIRE_W / 2;
  const TRUNK_X = 20, X_TACT = 4, X_MOTS = 12;
  // one line when the contract and its rate fit WIRE_W, else the rate on a line of its own
  const wireLines = m => {
    if (!WIRE[m]) return [];
    const one = wrapText(WIRE[m] + " · every model step", SUB_PX, WIRE_W);
    return one.length === 1 ? one : wrapText(WIRE[m], SUB_PX, WIRE_W).concat(["every model step"]);
  };
  // LINES A BOX ADDS UNDER ITS SUB-CAPTION (D-E4): a repair path this run built inside the box (HOSTED_PARAM, with its
  // own count and its own entry), and the attention cap's bound when this run was launched with the cap on - the cap
  // bounds the scores of stage 1 (cross-modal attention) in every block and of the motor encoder's attention
  // (lem/model/attention_cap.py), so block 0's stage-1 box and the motor encoder's box say so; the compact cells of the
  // later blocks have no room for it, and their popups carry it (capRows)
  const capOn = G.attention_cap;
  const capB = argsm.attention_cap_bound !== null && argsm.attention_cap_bound !== undefined
    ? Number(argsm.attention_cap_bound) : (rec.attention_cap || {}).bound;
  const capText = "attention cap on, " + (capB === null || capB === undefined || capB !== capB ? "bound not recorded"
    : Number.isInteger(capB) ? "bound " + capB : "--attention-cap-bound " + capB);
  const CAPPED_BOX = { enc_motor: true, block0_s1: true };
  function boxExtras(id) {
    const out = [];
    for (const k of HOSTED_PARAM[id] || [])
      if (CP[k]) out.push({ text: HOSTED_DOC[k][0] + ", " + pfmt(CP[k]), key: HOSTED_DOC[k][1], claims: k });
    if (capOn && CAPPED_BOX[id]) out.push({ text: capText, key: null, claims: null });
    return out;
  }
  // each extra line wrapped to the box's measured width, like a sub-caption
  function extraLines(id, ww) {
    return [].concat(...boxExtras(id).map(e => wrapText(e.text, SUB_PX, ww - 2 * BOX_PAD).map(t => Object.assign({}, e, { text: t }))));
  }
  const boxH = (id, sub, ww, h) => boxHeight(sub, ww, h, extraLines(id, ww).length);
  const encBuilt = rec.encoders_built || {};
  const encSub = m => "→ [" + nBody + " bodies × 1 × " + ((encBuilt[m] || {}).d_output || D_MODEL) + "]";
  // THE SENSE ROWS: each row's top, from a 74-unit top band for the world box. A row is as tall as its taller box; its
  // contract's first line sits 13 under the box, each further line 12 lower, and the next row starts 10 units under the
  // last line's text (a numbered circle reaches 9 above its box), so a row whose encoder box grows (the motor encoder
  // with its repair path and the cap) moves the rows under it down
  const rowH = mods.map(m => Math.max(boxH("sense_" + m, src[m] || "", SENSE_W), boxH("enc_" + m, encSub(m), ENC_W)));
  const rowTop = [];
  // THE TOP BAND above the world box (Oct 15, 2026): one lane, 10 units apart, for each line into the world (topIn, top
  // lane first) and then each line out of it (topOut), every one down into or up out of the world box's top edge, so
  // the caregiver box can sit beside the world box with its two short lines. Every row below moves down by TB.
  const topIn = ["motor", "gaze"].concat(G.curriculum_world ? ["curriculum"] : [])
    .concat(G.synergy && G.syn_babble ? ["synergy"] : []);
  // the lines out of the world's top edge (Oct 15, 2026: the world's contact report to the contact head joins the
  // completed movements), each up to a lane of its own below the lines in; and, lowest, the lane of the raw tokens'
  // line into the prediction heads (--residual-prediction on), which runs over the caregiver and the caption
  const topOut = (G.synergy ? ["synergy_examples"] : []).concat(G.contact ? ["contact"] : []);
  const topLow = G.residual ? ["residual"] : [];
  const TOPY = k => 6 + 10 * k;
  const topLane = name => {
    const k = topIn.concat(topOut, topLow).indexOf(name);
    if (k < 0) throw new Error("buildDiagram: no top lane named " + name);
    return TOPY(k);
  };
  const TB = 6 + 10 * (topIn.length + topOut.length + topLow.length) + 6 - 8;
  let rowNext = 74 + TB;
  mods.forEach((m, i) => {
    rowTop.push(rowNext);
    const n = wireLines(m).length;
    rowNext += rowH[i] + (n ? 26 + SUB_STEP * (n - 1) : 20);
  });
  const senseY = i => (rowTop[i] !== undefined ? rowTop[i] : 74);
  // THE CORE COLUMN (D-E5): block 0 at full size - its stage boxes and FFN, where the open GOF-BLOCKERs of the
  // transformer sit - and blocks 1 to N-1 as one compact row of cells each, every block in a frame of its own (its own
  // weights) and joined in series by the lines between them. Block tops (blockY) and frame heights:
  //   block 0  label lines at +4 and +16, boxes from +30, 10 units apart, closing caption 14 under the last box, frame
  //            from -8 to 20 under the last box;
  //   later    label lines at +4 and +16, cells from +26, CELL_H tall, frame from -8 to +56 (64 tall);
  //   between  12 units from one frame's bottom to the next frame's top (the corridor of the line between them).
  const FRAME_X = 590, FRAME_W = 200, STAGE_X = 602, STAGE_W = 184, CELL_H = 24, CELL_GAP = 8;
  const stageOn = k => (fm.stages || {})[k] !== false;
  // [id, box title, sub-caption, the cell's name] for each part a block has: the three stages this run built (a receipt
  // without the stages key is an older one and means all three) and the FFN
  const parts = [["s1", "stage 1 cross-modal", "senses ↔ senses", "stage 1", stageOn("cross_modal")],
                 ["s2", "stage 2 temporal", "causal mask, per sense", "stage 2", stageOn("causal_temporal")],
                 ["s3", "stage 3 bridge", "all × all, no mask", "stage 3", stageOn("cross_modal_bridge")],
                 ["ffn", "FFN", "×" + (fm.ffn_expansion || 4) + " expansion", "FFN", true]].filter(p => p[4]);
  const partH = parts.map(([sid, , sub]) => boxH("block0_" + sid, sub, STAGE_W));
  const block0Last = 30 + partH.reduce((a, h) => a + h, 0) + 10 * (parts.length - 1);
  const frameH = [], frameTop = [];
  for (let i = 0, top = 70 + TB; i < nL; i++) {
    frameTop.push(top);
    frameH.push(i === 0 ? block0Last + 28 : 64);
    top += frameH[i] + 12;
  }
  const blockY = i => frameTop[i];
  // the whole core's figure under the last frame; the corridors under it keep clear of its text
  const coreLabelY = nL ? frameTop[nL - 1] - 8 + frameH[nL - 1] + 14 : 70 + TB;
  // 32 under the label: the imagination corridor (coreH - 20) keeps 9 units clear of the label's text
  const coreH = Math.max(rowNext, coreLabelY + 32);
  // where the efference heads' readings reach the planner (schema.efference_routes, the same rule the tour step uses):
  // the goal-ranking channels, and the action search under --physical-goals memory, whose line takes a lane of its own
  // under the core (the lanes are defined with the bottom row, below)
  const EFF_ROUTES = META.efference_routes || {};
  const effSearch = G.efference && G.eff_search;
  let effGoalPts = null;           // the goal-ranking line's corners: drawn last, over a halo (below)
  const synPlan = G.synergy && G.syn_plan;
  const strip2 = [];
  if (G.memory)
    strip2.push(["memory", "persistent memory", "events beyond the window"]);
  // ONE RULE ON BOTH SIDES. walkthrough.py's self perception slide falls back to the
  // launch flag when the receipt carries no self_state field (a receipt written before
  // the field existed, or a page built before the run's first checkpoint). Reading the
  // receipt alone here drew no box for a slide that names one, and the tour skips a
  // missing box without a word.
  const selfMode = rec.self_state || argsm.self_state || "off";
  if (G.self_state)
    strip2.push(["self_state", "self perception (" + selfMode + ")", "learned body-condition estimate"]);
  if (G.self_state)
    strip2.push(["self_learning", "self learning", argsm.eval_only ? "measure error; weights fixed" : "update self-network weights"]);
  if (G.voice)
    strip2.push(["voice", "voice head", "80 numbers out · no lesson reaches it"]);
  if (G.hself)
    strip2.push(["hself", "legacy self tracker (dark)",
                 (argsm.hself_slots ? argsm.hself_slots + " slots, " : "") + "observe-only"]);
  // the named-entity self and its reader (--entity-self dark): drawn before the caregiver so
  // it stays in the second row, where its line up to the planner crosses no box
  if (G.entity_self)
    strip2.push(["entity_self", "named self: levels 2/3",
                 "teacher: " + (argsm.entity_self_teacher || "wellness")]);
  // Keep the omitted/off parts visible; their edges still follow the actual mode.
  if (!G.hself && !GATE_RUN.hself) strip2.push(["hself", "legacy self tracker (off)", "older levels 2/3 tracker; not built"]);
  strip2.push(["latent_body", "latent-body hierarchy", "learned networks; " + (selfReviewedSetting("latent_body_hierarchy") || "not recorded") + "; latent planner " + (selfReviewedSetting("latent_planner") || "not recorded")]);
  if (G.self_state) strip2.push(["self_report", "self report", "measurements to the researcher"]);
  if ((argsm.self_choice || "off") === "off") strip2.push(["self_choice", "self-choice (off)", "chooser not built; open for modes"]);
  // the caregiver runs inside the world's own physics step (env.py, BridgeEnv.step), so its box sits in the top band
  // beside the world box, with the two lines it has: what it reads of the world and what it changes in it
  const careTop = G.caregiver;
  // THE LANES UNDER THE CORE (Oct 15, 2026): one lane per line, 10 units apart, listed top to bottom below, each only
  // when its line is drawn - so no two lines share a lane and the gutter is as tall as this run's lines need
  const has = id => strip2.some(s => s[0] === id);
  // the input check of Oct 15, 2026 adds, top to bottom: "gated" (the top-down gate's tokens, or the shared
  // LayerNorm's with the gate off, into the planner and the imagination chain; first, so its corridor at x 412 ends
  // above the wellness line's corridor at x 405), "exec" (the executed command into the motor encoder, persistent
  // memory and self perception; above "ln_self", so its corridor at x 296 ends above that line's at x 290),
  // "final_self" (the core's output into self perception, one model step later), and last the two raw-reading lanes
  // of the tactile and motor senses
  const senseOn = m => mods.includes(m);
  const laneNames = ["gated", "ln_wellness", "intention_motor", "motor_intention", "wellness_intention"]
    .concat(has("memory") ? ["memory_final_norm"] : [])
    .concat(has("voice") ? ["final_norm_voice"] : [])
    .concat(["heads_curiosity"])
    .concat(["exec"])
    .concat(has("self_state") ? ["final_self"] : [])
    .concat(has("memory") ? ["ln_memory"] : [])
    .concat(has("self_state") ? ["ln_self"] : [])
    .concat(has("self_state") && selfMode === "perceive" ? ["self_final"] : [])
    .concat(has("entity_self") ? ["wellness_entity"] : [])
    .concat(has("hself") ? ["wellness_hself", "ln_hself"] : [])
    .concat(synPlan ? ["synergy_intention"] : [])
    .concat(effSearch ? ["efference_intention"] : [])
    .concat(senseOn("tactile") ? ["tactile_raw"] : [])
    .concat(senseOn("motor") ? ["motor_raw"] : []);
  const lane = name => {
    const k = laneNames.indexOf(name);
    if (k < 0) throw new Error("buildDiagram: no lane named " + name);
    return coreH + 10 + k * 10;
  };
  const stripY = coreH + 10 + 10 * laneNames.length + 6;
  // THE BUILD'S SIDE HEADS (2026-09-22) - drawn only when this run's receipt built them, in
  // a column of their own right of the outputs, so no existing line has to move for them. The
  // last number is the gap above the box: the column is stacked from the efference heads' top
  // (level with the motor controller) downward, 10 units between boxes and 20 before the
  // completed movements
  const sideDefs = [];
  // the code's own name (lem/model/efference.py EfferenceHeads), the one the box's popup, the
  // tour slide and the run tab's charts use; "view" only when the heads' stamp records the gaze
  // copy into the vision prediction
  if (G.efference)
    sideDefs.push(["efference", "efference heads", (G.eff_gaze_copy
      ? "command → cm, degrees, view" : "command → cm, degrees"), 0]);
  if (G.contact)
    sideDefs.push(["contact_head", "contact head", "53 patches: touched? by what?", 10]);
  if (G.synergy) {
    sideDefs.push(["synergy", "motor synergy", G.syn_plan ? "planner control on"
      : G.syn_babble ? "exploration on" : "learning; control off", 10]);
    sideDefs.push(["synergy_examples", "completed movements", "before / after body poses", 20]);
  }
  const RM = 1260;                          // the right margin every older route is drawn to
  const sideX = RM + 30, sideW = 150;
  // The caption band under the picture: its sentences are wrapped to the drawing's width from
  // their measured text (one ran past the right edge on H84's page, the layout check of
  // 2026-10-02) and the drawing's height is set once they are placed, at the end of this function.
  // Two of them exist only when this run has a voice head or a hierarchical self.
  const vhRec = rec.voice_head || {};
  const capVoice = !!(vhRec.enabled && vhRec.params && rec.total_params
                      && !SNAPS.some(s => (((s.losses || {}).breakdown) || {}).voice !== undefined));
  // only a run whose table exists: H84 carries --hself-slots 6400 with --hierarchical-self off
  // since segment 9, and the line claimed 7,372,800 stored numbers for a table never built
  const capHself = !!(argsm.hself_slots && mods.length && argsm.hierarchical_self
                      && argsm.hierarchical_self !== "off");
  const idleSelf = (((META.idle_params || {}).self_state) || {}).elements || 0;   // P204
  // the bottom rows wrap at five boxes, 70 units apart, like the first two
  // THE STRIP BOXES' CAPTIONS, read from this run's flags (the code check of Oct 15, 2026): the planner's search looks
  // one model step ahead whenever --hand-target-mode is not "world" (lem_model.py _plan_chain sets max_steps = 1), so the
  // imagination chain sets no plan depth then; a run that starts in the final phase with its age fixed and its delays off
  // can never change phase, so the curriculum only holds the final phase's limits
  const planDepthOne = (argsm.hand_target_mode || "world") !== "world";
  const rankNorm = (argsm.wellness_rank || (rec.wellness_channels || {}).rank) === "normalized";
  const curSub = (argsm.curiosity_goal || "off") !== "on"
    ? "not read by the planner: --curiosity-goal " + (argsm.curiosity_goal || "off")
    : rankNorm ? "one of the goal ranking's channels"
    : argsm.curiosity_mode === "additive" ? "goal-ring value + drive × its weight" : "goal-ring value × (1 + drive)";
  const handStage = argsm.hand_goals === "memory";
  const stripDefs = [["curiosity", "curiosity", curSub],
                     ["intention", "intention planner", "uses learned predictions; every " + (argsm.intention_every || 10)
                       + " model steps: goal ring, 20 Adam updates" + (handStage ? ", hand stage" : "")
                       + (planDepthOne ? "; 1 model step ahead" : "")],
                     ["imagination", "imagination", imagOn()
                       ? "predict, feed back, ×" + imagDepth() + (planDepthOne ? "; sets no plan depth this run" : "")
                       : "off this run - no multi-step lesson trained"],
                     ["replay", "replay memory", argsm.replay_every
                       ? "re-study every " + argsm.replay_every + " model steps" : "re-study of remembered model steps"],
                     ["curriculum", "curriculum", finalFixed ? "final phase (mature): fixed reach and strength limits"
                       : "6 phases, growth + delays"]];
  // each strip row is as tall as its tallest box. THE GUTTER between the two rows (Oct 15, 2026) holds one lane, 10
  // units apart, for each group of lines listed here, and the second row starts 20 units under the last lane:
  //   gut_w  the world's consequence count -> planner (x 20 to 340, up into the planner's bottom edge)
  //   gut_a  named things -> planner (x 430 to the named things' middle) and planner -> persistent memory (x 230 to 320)
  //   gut_b  tactile -> self perception (x 4 to 330) and the core's output -> self perception (x 470 to 757)
  //   gut_c  persistent memory -> replay memory (x 250 to 900)
  const row1H = Math.max(...stripDefs.map(([id, , sub]) => boxH(id, sub, 210, NH)));
  const row2H = strip2.length ? Math.max(...strip2.map(([id, , sub]) => boxH(id, sub, 210, NH))) : 0;
  const esOn = G.entity_self;
  const gutNames = [].concat(G.consequence ? ["gut_w"] : []).concat(esOn || has("memory") ? ["gut_a"] : [])
    .concat(has("self_state") ? ["gut_b"] : []).concat(has("memory") ? ["gut_c"] : []);
  const gut = name => {
    const k = gutNames.indexOf(name);
    if (k < 0) throw new Error("buildDiagram: no gutter lane named " + name);
    return stripY + row1H + 10 + 10 * k;
  };
  // the second row 20 units under the lowest lane (the independent review of Oct 15, 2026, S3): a line along the
  // lowest lane then crosses the arrows into the second row's top edges 13 units above their arrowheads' bases
  const row2Y = stripY + row1H + 20 + 10 * Math.max(1, gutNames.length);
  const rows2 = Math.ceil(strip2.length / 5);
  const rowsBottom = strip2.length ? row2Y + (row2H + 20) * (rows2 - 1) + row2H : stripY + row1H;
  // the world's lines to the named things, to persistent memory and to self perception run along a corridor 12 units
  // under the bottom row (the world's trunk, one of the drawing's combs)
  const underRow = esOn || has("memory") || has("self_state");
  const bodyBottom = rowsBottom + (underRow ? 22 : 0);
  // the side column keeps three corridors right of it (the contact head's trunk, the world -> completed movements
  // line and the synergy's babble line to the world), 12 units apart, and a fourth for the world's contact report
  // into the contact head when the run built one
  const W = sideDefs.length ? sideX + sideW + (G.contact ? 62 : 50) : RM;
  const svg = el("svg", { viewBox: `0 0 ${W} ${bodyBottom + 20}` });
  svg.appendChild(el("defs", {},
    el("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" },
      el("path", { d: "M0 0 L10 5 L0 10 z", fill: "#9aa5b8" })),
    // the arrowhead of the 8-unit links between a compact block row's cells: 4 x the line's width, so it fits the link
    el("marker", { id: "arrow-s", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 4, markerHeight: 4, orient: "auto" },
      el("path", { d: "M0 0 L10 5 L0 10 z", fill: "#9aa5b8" }))));
  const nodes = {};
  // every component_params key some DRAWN box accounts for, filled in by node() as the
  // boxes are made; the footer's not-drawn list is whatever is left over
  const drawnComps = new Set();
  // a box's own share of a larger component's count, read from the receipt's
  // per-tensor inventory; 0 when an older receipt carries no inventory, and the box
  // then draws exactly as it did before
  function subParam(id) {
    const spec = subParamSpec(id);
    if (!spec) return 0;
    const tens = (((rec.checkpoint_inventory || {}).components || {})[spec[0]] || {}).tensors || {};
    let n = 0;
    for (const k in tens)
      if (tens[k].kind !== "buffer" && spec[1].some(p => k.indexOf(p) === 0))
        n += tens[k].elements || 0;
    return n;
  }
  // A BOX'S TEXT IS LAID OUT FROM ITS MEASURED WIDTH (the style guide "DIAGRAM LAYOUT ON A GRID"; the
  // layout check of 2026-10-02 found captions running edge to edge and counts printed over the
  // border). The caption is drawn at 12 units, the sub-caption at 10 and the corner count at 9
  // (viewer.css; one size step larger since Oct 7, 2026, D-E6), each at least BOX_PAD units inside the
  // box's left and right edges. A sub-caption wider than that is broken at its spaces onto further lines
  // SUB_STEP units apart (boxSubLines), and the box grows SUB_STEP units per added line (boxHeight), so a
  // caller can place a box below a taller one before drawing it (boxH counts the box's extra lines too).
  // Under the sub-caption come the box's extra lines (boxExtras: a repair path the run built inside the box,
  // which opens its own entry, and the attention cap's bound). The corner count has a line of its own under
  // them. Baselines: the caption 16 units under the box's top, the sub-caption's and the extra lines 31,
  // 43, ... and the count 4 above the bottom: at one unit per pixel a 12-unit caption's text box reaches
  // about 13 above its baseline and 3 below it, a 10-unit line's 11 and 3, a 9-unit count's 10 and 3.
  function node(id, x, y, label, sub, w, h) {
    // the keys this box's own corner count may come from, in order; an encoder box's
    // key is its own id. A box that owns part of a larger component (token assembly,
    // the stage boxes, the final LayerNorm) accounts for that component too.
    const direct = [].concat(NODE_PARAM[id] || [], id);
    const np = direct.map(k => CP[k]).find(v => v) || subParam(id);
    direct.forEach(k => drawnComps.add(k));
    const owner = subParamSpec(id);
    if (owner) drawnComps.add(owner[0]);
    const ww = w || NW, subLines = boxSubLines(sub, ww), extra = extraLines(id, ww);
    // a repair path drawn as a line of this box accounts for its own component_params key (HOSTED_PARAM)
    extra.forEach(e => { if (e.claims) drawnComps.add(e.claims); });
    const hh = boxHeight(sub, ww, h, extra.length);
    const g = el("g", { class: "dg-node" + (np ? (NO_LESSON[id] ? " nolesson" : " learned") : ""), id: "nd-" + id });
    g.appendChild(el("rect", { x, y, width: ww, height: hh, rx: 4 }));
    g.appendChild(el("text", { x: x + ww / 2, y: y + (sub || extra.length ? 16 : hh / 2 + 4.5),
      "text-anchor": "middle", text: label }));
    if (sub) {
      const t = el("text", { x: x + ww / 2, y: y + 31, "text-anchor": "middle", class: "sublabel" });
      subLines.forEach((ln, i) => t.appendChild(el("tspan", { x: x + ww / 2, dy: i ? SUB_STEP : 0, text: ln })));
      g.appendChild(t);
    }
    extra.forEach((e, k) => {
      const t = el("text", { x: x + ww / 2, y: y + 31 + SUB_STEP * (subLines.length + k), "text-anchor": "middle",
                             class: "sublabel" + (e.key ? " hosted" : ""), text: e.text });
      if (e.key) t.addEventListener("click", ev => { ev.stopPropagation(); showDoc(e.key, (DOC[e.key] || {}).t || e.text); });
      g.appendChild(t);
    });
    if (np) g.appendChild(el("text", { x: x + ww - BOX_PAD - 1, y: y + hh - 4,
      "text-anchor": "end", class: "paramtag" + (NO_LESSON[id] ? " nolesson" : ""),
      text: pfmt(np) + (NO_LESSON[id] ? " · " + NO_LESSON[id] : "") }));
    g.addEventListener("click", () => nodeClicked(id, label));
    svg.appendChild(g);
    nodes[id] = { x, y, w: ww, h: hh };
    return g;
  }
  // A COMPACT CELL (D-E5): one part of a block after block 0, drawn small in its block's row. It keeps the box's own id
  // (block1_s1, ...), so the tour, the edge mirrors, the component map and the address of a box find it, and a click
  // opens the same entry the full-size box opens. Its name is drawn at 10 units, centred; it carries no corner count
  // (its frame's label carries the block's own count), and the bold outline when the receipt counts weights in it.
  function cell(id, x, y, w, h, name, title) {
    const np = subParam(id);
    const owner = subParamSpec(id);
    if (owner) drawnComps.add(owner[0]);
    const g = el("g", { class: "dg-node" + (np ? " learned" : ""), id: "nd-" + id });
    g.appendChild(el("rect", { x, y, width: w, height: h, rx: 4 }));
    g.appendChild(el("text", { x: x + w / 2, y: y + h / 2 + 3.5, "text-anchor": "middle", class: "cell", text: name }));
    g.addEventListener("click", () => nodeClicked(id, title));
    svg.appendChild(g);
    nodes[id] = { x, y, w, h };
    return g;
  }
  // the DOC key for a line's data contract; block edges share one entry per
  // stage pair, encoder->norm lines share one, sense->encoder stays per sense.
  // The two lines around the top-down gate follow the same rule as every other
  // line - edge.ln_out.td_gate and edge.td_gate.assembly carry their own contracts
  // (shape, units, whether gradient crosses). An earlier override pointed them at
  // tokens.raw and tokens.fed instead, which left both line entries shipped in the
  // page with no click able to open them; tokens.raw and tokens.fed open from the
  // inspector's gate panel, where the two are measured against each other.
  function edgeKey(a, b) {
    let x = a.replace(/^block\d+_/, "block_"), y = b.replace(/^block\d+_/, "block_");
    if (/^enc_/.test(x) && y === "ln_out") x = "enc";
    if (/^sense_/.test(x) && /^enc_/.test(y)) y = "enc";
    return "edge." + x + "." + y;
  }
  // bus: the comb a line belongs to (data-bus) - the only lines the layout check lets share a stretch are the lines of
  // one comb: the world's trunk to the senses, the encoders into the shared LayerNorm, the final LayerNorm's three
  // outputs (operator, Oct 15, 2026: no two lines drawn on top of each other)
  function draw(a, b, d, cls, bus) {
    const attrs = { class: "dg-edge" + (cls ? " " + cls : ""),
      d, "data-a": a, "data-b": b, "marker-end": cls === "short" ? "url(#arrow-s)" : "url(#arrow)" };
    if (bus) attrs["data-bus"] = bus;
    const vis = el("path", attrs);
    svg.appendChild(vis);
    // every line is clickable: a wide invisible twin opens the data contract
    const hit = el("path", { class: "dg-hit", d });
    hit.addEventListener("click", ev => {
      ev.stopPropagation();
      showDoc(edgeKey(a, b), "data on this line");
    });
    hit.addEventListener("mouseenter", () => vis.classList.add("hot"));
    hit.addEventListener("mouseleave", () => vis.classList.remove("hot"));
    svg.appendChild(hit);
  }
  // solid flow edge between horizontally adjacent boxes (never crosses columns); cls "short" draws the small
  // arrowhead of the 8-unit links between the cells of a compact block row
  function edge(a, b, cls, bus) {
    const A = nodes[a], B = nodes[b];
    if (!A || !B) return;
    if (A.x + A.w <= B.x) {
      const sy = A.y + A.h / 2, ty = B.y + B.h / 2, sx = A.x + A.w, tx = B.x,
            mx = Math.round((sx + tx) / 2);
      draw(a, b, sy === ty ? `M${sx} ${sy} L${tx} ${ty}`
                           : `M${sx} ${sy} L${mx} ${sy} L${mx} ${ty} L${tx} ${ty}`, cls, bus);
    } else {                              // stacked in one column
      const cx = Math.round((Math.max(A.x, B.x) + Math.min(A.x + A.w, B.x + B.w)) / 2);
      draw(a, b, A.y < B.y ? `M${cx} ${A.y + A.h} L${cx} ${B.y}`
                           : `M${cx} ${A.y} L${cx} ${B.y + B.h}`, cls, bus);
    }
  }
  // hand-routed dashed edge: explicit corridor waypoints, no box crossings
  function route(a, b, pts, bus) {
    const d = "M" + pts.map(p => p.join(" ")).join(" L");
    draw(a, b, d, "loopback", bus);
  }
  // the same routing for a read that happens INSIDE the step, drawn solid: the
  // legend gives dashed the meaning "influence between steps", so a same-step
  // read that needs a corridor must not borrow the dashed style
  function routeSolid(a, b, pts, bus) {
    draw(a, b, "M" + pts.map(p => p.join(" ")).join(" L"), undefined, bus);
  }
  // data-for and data-n name the box a circle numbers, for the layout check (#layoutcheck)
  function badge(id, n) {
    const N = nodes[id];
    if (!N) return;
    svg.appendChild(el("circle", { cx: N.x - 1, cy: N.y - 1, r: 8, class: "dg-badge", "data-for": id, "data-n": n }));
    svg.appendChild(el("text", { x: N.x - 1, y: N.y + 2.5, "text-anchor": "middle",
                                 class: "dg-badge-num", text: String(n) }));
  }
  // the world: where commands act and the sensor readings come from - NOT part of the model
  node("world", 16, 8 + TB, "the world", "MuJoCo crib: body, caregiver, toys", 254, 30);
  if (careTop) {
    node("caregiver", 300, 8 + TB, "caregiver (world)", "speaks, brings toys, comes and goes", 210, NH);
    // same physics step, both solid: it reads where the infant looks, the toy's contact and movement, and the hands
    // gripping the toy; it moves the adult and the toy, plays its speech and sets the mobile's height
    const wB0 = nodes.world;
    routeSolid("world", "caregiver", [[wB0.x + wB0.w, wB0.y + 14], [300, wB0.y + 14]]);
    routeSolid("caregiver", "world", [[300, wB0.y + 36], [wB0.x + wB0.w, wB0.y + 36]]);
  }
  // each sense row: the sense, its encoder, and under the taller of the two the line's contract (WIRE), wrapped
  // to WIRE_W from its measured width and centred under the pair
  mods.forEach((m, i) => {
    node("sense_" + m, SENSE_X, senseY(i), m, src[m] || "", SENSE_W);
    node("enc_" + m, 138, senseY(i), (encBuilt[m] || {}).type || "encoder", encSub(m), ENC_W);
    // straight at the sense box's middle into the encoder's left edge (the encoder box is as tall or taller): the
    // 10-unit gap between the two boxes holds no bend
    {
      const A = nodes["sense_" + m], B = nodes["enc_" + m], y = A.y + A.h / 2;
      if (y >= B.y + 4 && y <= B.y + B.h - 4) draw("sense_" + m, "enc_" + m, `M${A.x + A.w} ${y} L${B.x} ${y}`);
      else edge("sense_" + m, "enc_" + m);
    }
    const wl = wireLines(m);
    if (wl.length) {
      const t = el("text", { x: WIRE_X, y: senseY(i) + rowH[i] + 13, "text-anchor": "middle", class: "sublabel" });
      wl.forEach((ln, k) => t.appendChild(el("tspan", { x: WIRE_X, dy: k ? SUB_STEP : 0, text: ln })));
      svg.appendChild(t);
    }
  });
  // the world feeds EVERY sense - one shared trunk down the left margin, one
  // branch per sense box (drawing only vision made five senses look sourceless).
  // Solid: these readings are carried into THIS step's encoders, so drawing them
  // dashed said "between steps" about six lines that are not.
  mods.forEach((m, i) => routeSolid("world", "sense_" + m,
        [[TRUNK_X, nodes.world.y + nodes.world.h], [TRUNK_X, senseY(i) + NH / 2], [SENSE_X, senseY(i) + NH / 2]],
        "world-trunk"));
  // the middle of the sense column, from the first row's top to the last row's box bottom
  const lastRow = Math.max(mods.length - 1, 0);
  const midY = Math.round((senseY(0) + senseY(lastRow) + (rowH[lastRow] || NH)) / 2 - NH / 2);
  // "RMS = 1" holds only while the norm has no learned gain and bias; a run that
  // switched them on gets the count at the corner and a caption that says so
  node("ln_out", 300, midY, "shared LayerNorm",
       (CP.enc_shared || CP.encoder_registry) ? "learned gain + bias" : "RMS = 1", 106);
  mods.forEach(m => edge("enc_" + m, "ln_out", undefined, "enc-norm"));
  const ctx = (fm._seq_length || 0) / (fm._n_modalities || 1);
  node("assembly", 434, midY, "token assembly", `${fm._n_modalities || 6} senses × ${ctx} model steps`, 130);
  // --top-down-gate blends up to half of the model's own prediction of NOW into the
  // token the core reads. One unbroken line from the norm to the assembly said the
  // core reads the encoders, which is false for any sense whose gate is open, and the
  // gate appeared nowhere in the picture that claims to show how the model works.
  // The gate's box sits ABOVE the norm: the four lines that carry the norm's RAW tokens down to
  // wellness, re-study, imagination and the hierarchical self leave the norm's bottom edge, and
  // with the box below the norm they ran through it (the layout check of 2026-10-02), as if those
  // four read the gated tokens.
  if (G.td_gate) {
    const tdSub = "up to half the model's guess of now";
    node("td_gate", 300, midY - 22 - boxHeight(tdSub, 106), "top-down gate", tdSub, 106);
    edge("ln_out", "td_gate");
    edge("td_gate", "assembly");
  } else edge("ln_out", "assembly");
  // THE BLOCK FRAMES (D-E5), top to bottom of each: two label lines (baselines 4 and 16 under the block's top, from
  // x 600: the block's name as the defect register writes it, and its own count of learned numbers), then block 0's
  // stage boxes and FFN from 30 under its top, 10 units apart, and its closing caption 14 under the last box; or, for
  // each later block, one row of cells from 26 under its top. Block 0's second label line ends 2 units above stage 1's
  // numbered circle, and the line from one block to the next runs down the frames' left gutter at x 595, left of
  // every label and caption.
  for (let i = 0; i < nL; i++) {
    const gy = blockY(i), full = i === 0;
    svg.appendChild(el("rect", { x: FRAME_X, y: gy - 8, width: FRAME_W, height: frameH[i], fill: "none", class: "dg-frame",
                                 "data-name": "the frame of " + blockName(i, nL),
                                 stroke: "#bcc4d4", "stroke-dasharray": "3 3", rx: 6 }));
    // THE FIGURE ON A FRAME IS WHAT THAT FRAME ENCLOSES. The whole core's 712k used to
    // be printed on block 1's frame, which holds 331k of it - 2.05 times the enclosed
    // block. Each frame now carries its own four parts added up ("its own": every block
    // has weights of its own), and the label opens the transformer's definition like
    // every other labelled thing on the model diagram.
    const blockP = ["s1", "s2", "s3", "ffn"].reduce((a, sg) => a + subParam(`block${i}_${sg}`), 0);
    const frameLab = el("text", { x: FRAME_X + 10, y: gy + 4, class: "sublabel" + (blockP ? " clicky" : "") });
    [blockName(i, nL)].concat(blockP ? ["its own " + pfmt(blockP) + " learned numbers"] : []).forEach((ln, k) =>
      frameLab.appendChild(el("tspan", { x: FRAME_X + 10, dy: k ? SUB_STEP : 0, text: ln })));
    if (blockP) frameLab.addEventListener("click", () => showDoc("node.core", "the core and its two size figures"));
    svg.appendChild(frameLab);
    // WHICH stages exist and how wide the FFN is are the RUN's, not this file's (parts, above): the
    // three boxes and "×4 expansion" used to be fixed strings, so an ablation with a
    // stage off would have drawn a diagram the model did not have.
    let prev = null;
    if (full) {
      let y = gy + 30;
      parts.forEach(([sid, lab, sub], k) => {
        node(`block${i}_${sid}`, STAGE_X, y, lab, sub, STAGE_W);
        if (prev) edge(`block${i}_${prev}`, `block${i}_${sid}`);
        prev = sid; y += partH[k] + 10;
      });
      svg.appendChild(el("text", { x: FRAME_X + 10, y: gy + block0Last + 14, class: "sublabel",
        text: "norm → attend or MLP → add back" }));
    } else {
      // the same parts as block 0, as cells left to right, 8 units apart
      const cw = (STAGE_W - CELL_GAP * (parts.length - 1)) / parts.length;
      parts.forEach(([sid, lab, , name], k) => {
        cell(`block${i}_${sid}`, STAGE_X + k * (cw + CELL_GAP), gy + 26, cw, CELL_H, name, lab);
        if (prev) edge(`block${i}_${prev}`, `block${i}_${sid}`, "short");
        prev = sid;
      });
    }
    const firstBox = parts[0][0];
    if (i === 0) edge("assembly", `block0_${firstBox}`);
    else {
      // from block 0's FFN box: out of its left side and down the gutter at x 595. From a compact row's FFN cell, at
      // the row's right end: out of its bottom, left along the corridor between the two frames (6 units above this
      // frame), then down the gutter. Either way into the first part's left side: a straight line from box to box ran
      // through the caption above and this frame's label
      const A = nodes[`block${i - 1}_ffn`], B = nodes[`block${i}_${firstBox}`], by = B.y + B.h / 2, gut = FRAME_X + 5;
      draw(`block${i - 1}_ffn`, `block${i}_${firstBox}`, i === 1
        ? `M${A.x} ${A.y + A.h / 2} L${gut} ${A.y + A.h / 2} L${gut} ${by} L${B.x} ${by}`
        : `M${A.x + A.w / 2} ${A.y + A.h} L${A.x + A.w / 2} ${gy - 14} L${gut} ${gy - 14} L${gut} ${by} L${B.x} ${by}`);
    }
  }
  // the whole core's figure, under every block frame where it cannot be read as one
  // block's size, and clickable like every other labelled thing on the model diagram
  if (CP.forward_model) {
    const coreLab = el("text", { x: FRAME_X, y: coreLabelY, class: "sublabel clicky",
      text: "the whole core: " + pfmt(CP.forward_model) + " saved numbers" });
    coreLab.addEventListener("click", () => showDoc("node.core", "the core and its two size figures"));
    svg.appendChild(coreLab);
  }
  node("final_norm", 806, midY, "final LayerNorm", "learned gain + bias", 118);
  edge(`block${nL - 1}_ffn`, "final_norm");
  // the output column, 48 units right of the final LayerNorm: the norm's three lines into it share the corridor at x 948,
  // which ends at the norm's middle; the controller's command to the planner drops 8 units left of the column (x 964,
  // 16 units right of that corridor, out of the controller's own bottom edge), and the planner's line into the
  // controller rises 16 units left of it (x 956) and ends in an arrowhead on the controller's left edge (the independent
  // recheck of Oct 15, 2026, P2-m1: the command to the planner ran down x 948 and read as the final LayerNorm's lines
  // continuing)
  const outX = 972;
  const handles = rec.handle_names || [];
  const gzi = handles.indexOf("gaze");
  // the output column is stacked on the motor controller, whose top stays at midY - 20 where the
  // lines into its sides are drawn: 10 units between boxes, each as tall as its measured text needs
  // (the prediction heads' sub-caption takes two lines, and a third when the run built their all-slot path)
  const headsSub = ["next model step,", "every sense but motor"], reconSub = "the present, as a guard";
  const motorY = midY - 20, reconY = motorY - 10 - boxH("recon", reconSub, 150);
  node("heads", outX, reconY - 10 - boxH("heads", headsSub, 150), "prediction heads", headsSub, 150);
  node("recon", outX, reconY, "recon decoders", reconSub, 150);
  node("motor", outX, motorY, "motor controller",
       (handles.length || 24) + " handles × "
       + ((rec.motor_controller || {}).output_per_handle || 9), 150, 66);
  node("gaze", outX, nodes.motor.y + nodes.motor.h + 10, "gaze = handle " + (gzi >= 0 ? gzi : 23),
       "one command, both eyes", 150);
  // 6 fixed formulas score a LIVED model step (the sixth, self_touch, joined on 2026-09-21);
  // the learned copy that scores an IMAGINED model step has one output per name in
  // lem/model/wellness.py APPROXIMATE_TERMS, which is five of them
  node("wellness", outX, nodes.gaze.y + nodes.gaze.h + 10, "wellness", "6 fixed + a learned copy of 5", 150);
  ["heads", "recon", "motor"].forEach(t => edge("final_norm", t, undefined, "norm-out"));
  edge("motor", "gaze");
  // the strip boxes (their captions are read from this run's flags above, with the row heights)
  stripDefs.forEach(([id, label, sub], i) => node(id, 60 + i * 240, stripY, label, sub, 210, NH));
  strip2.forEach(([id, label, sub], i) =>
    node(id, 60 + (i % 5) * 240, row2Y + (row2H + 20) * Math.floor(i / 5), label, sub, 210, NH));
  // the side column, stacked from the efference heads' top (level with the motor controller)
  let sideTop = midY - 20;
  sideDefs.forEach(([id, label, sub, gap]) => {
    node(id, sideX, sideTop + gap, label, sub, sideW, NH);
    sideTop = nodes[id].y + nodes[id].h;
  });
  // HAND-ROUTED LINES (relaid out on Oct 15, 2026, from that day's code check of every box's inputs and outputs in
  // lem_model.py, train_mimo.py and env.py, and the operator's rule of the same day: no two lines on top of each
  // other). DASHED is reserved for a line whose data carries from one model step into the NEXT; every line built and
  // used inside one step is solid. No two lines share a stretch, except inside the three combs the drawing declares
  // (data-bus: the world's trunk to the senses, the encoders into the shared LayerNorm, the final LayerNorm into its
  // three outputs), and two lines keep 8 units or more apart (the layout check's "lines run together"). Every line has
  // its own exit, its own corridor and its own entry:
  //   the motor controller's right edge  pW out to the world, pCur in from the curriculum, pSide out to the efference
  //                                      heads, pCon out to the contact head, pImag out to imagination, pSlope in from
  //                                      wellness; its left edge: the final LayerNorm's line at the middle, the
  //                                      planner's 10 units above the bottom
  //   the wellness box's right edge      qHeads in from the prediction heads, qSlope out to the motor controller, qEff
  //                                      in from the efference heads, qCon in from the contact head
  //   the shared LayerNorm's bottom      one exit per line, 12 units apart: imagination, replay, wellness, memory,
  //                                      self perception, hierarchical self
  //   the final LayerNorm's top          efference heads (x - 9 from its right edge), contact head (x - 21), each
  //                                      over the output column on a corridor of its own (12 and 24 above the heads)
  //   vertical corridors right of the    X_HW, X_SL, X_CU, X_HC (12 to 48 units right of the column), X_EW, X_CW, X_EI,
  //   output column                      X_IM, X_WD, X_GW, X_CWD, X_EH, X_SB; right of the side column the contact
  //                                      head's trunk, the world -> completed movements line and the synergy's babble
  //                                      line, 12 units apart
  //   the top band                       one lane per line into or out of the world (TOPY), each line down into the
  //                                      world box's top at an x of its own
  //   the lanes under the core           lane(name), one per line, listed where the lanes are defined above
  const lnB = nodes.ln_out, fnB = nodes.final_norm, hdB = nodes.heads, moB = nodes.motor, gzB = nodes.gaze,
        wlB = nodes.wellness, wB = nodes.world, iB = nodes.intention;
  const lnBot = lnB.y + lnB.h, fnBot = fnB.y + fnB.h, wellBot = wlB.y + wlB.h;
  const lnExit = k => lnB.x + 26 + 12 * k;     // k: 0 imagination, 1 replay, 2 wellness, 3 memory, 4 self, 5 hself
  const OR = outX + 150;
  const M0 = moB.y, pW = M0 + 6, pCur = M0 + 16, pSide = M0 + 25, pCon = M0 + 36, pImag = M0 + 46, pSlope = M0 + 56;
  const W0 = wlB.y, qHeads = W0 + 8, qSlope = W0 + 20, qEff = W0 + 32, qCon = W0 + 44;
  const X_HW = OR + 12, X_SL = OR + 24, X_CU = OR + 36, X_HC = OR + 48, X_EW = 1184, X_CW = 1196, X_EI = 1208,
        X_IM = 1220, X_WD = 1232, X_GW = 1244, X_CWD = 1256, X_EH = 1266, X_SB = 1278;
  const X_CT = sideX + sideW + 12, X_SE = sideX + sideW + 24, X_BB = sideX + sideW + 36, X_WC = sideX + sideW + 48;
  // THE INPUT CHECK OF OCT 15, 2026 (lem_model.py, train_mimo.py, env.py, memory_integration.py,
  // self_state_integration.py, prediction_heads.py, motor_controller.py and wellness.py read line by line): its lines'
  // corridors, each 8 units or more from its neighbours
  //   X_MW           the motor sense's raw readings up to the motor controller's left edge, 36 units left of the
  //                  output column; the final LayerNorm's comb runs at 24, the controller's command down to the planner
  //                  at 16 and the planner's line up at 8. The two senses' lines into wellness rise from their lanes
  //                  into wellness's bottom edge, 12 and 24 units from its left end
  //   X_GT           the top-down gate's tokens (the shared LayerNorm's with the gate off) down to the planner and the
  //                  imagination chain, 6 units right of the shared LayerNorm, 8 left of the gate -> assembly corridor
  //   X_RS, X_RU     the raw tokens' line into the prediction heads: out of the shared LayerNorm's top 14 units from
  //                  its right end, and up past the caregiver's right edge to the top band
  //   X_EM          the executed command up into the motor encoder's right edge, 8 right of the encoders' comb (the
  //                  same command's corridors into persistent memory and self perception sit 23 units left of the
  //                  memory box and 23 right of the self-perception box, 8 from the voice line)
  //   X_FS           the core's output out of the final LayerNorm's bottom, 12 units from the voice line and from
  //                  re-study's line
  const X_MW = outX - 36, X_GT = lnB.x + lnB.w + 6, X_RS = lnB.x + lnB.w - 14, X_RU = 520,
        X_EM = 296, X_FS = fnB.x + 62;
  // the world box's top edge, right to left: the lines out of it (topOut, one slot each, the last furthest right), then
  // the lines into it (topIn; the line in the top lane enters furthest left); 14 units apart
  const nOutSlots = Math.max(1, topOut.length);
  const outX0 = name => wB.x + wB.w - 16 - 14 * (topOut.length - 1 - topOut.indexOf(name));
  // a line into the world: right to its own corridor, up it to its own lane of the top band, left to its own x and
  // down into the world box's top
  const toWorld = (name, x, y, xv) => {
    const k = topIn.indexOf(name), xe = wB.x + wB.w - 16 - 14 * (nOutSlots - 1 + topIn.length - k);
    return [[x, y], [xv, y], [xv, TOPY(k)], [xe, TOPY(k)], [xe, wB.y]];
  };
  // WELLNESS IS NOT DOWNSTREAM OF THE TRANSFORMER ON THIS PATH. lem_model.py hands wellness.py canonical_tokens = the
  // six RAW encoder tokens, detached, and the learned scorer uses the five sensory ones; `latents` is read only for the
  // batch size and the device. So the line starts at the shared LayerNorm.
  routeSolid("ln_out", "wellness", [[lnExit(2), lnBot], [lnExit(2), lane("ln_wellness")], [wlB.x + 128, lane("ln_wellness")],
                                    [wlB.x + 128, wellBot]]);
  // the wellness slope - one input number per command number of the motor controller
  routeSolid("wellness", "motor", [[OR, qSlope], [X_SL, qSlope], [X_SL, pSlope], [OR, pSlope]]);
  // THE PREDICTION HEADS' OUTPUT (the code check of Oct 15, 2026). The delayed re-run of the encoders, the core and the
  // heads on the previous model step's readings, with the executed command in, predicts THIS model step's tokens
  // (lem_model.py self._nf_pred). Three parts read that prediction inside the same step, each on a line of its own: the
  // top-down gate blends up to half of it into the tokens the core reads (out of the heads' top, over the block
  // frames), wellness scores it against the raw tokens (the prediction-error and learning-progress channels), and the
  // curiosity drive is computed from its error (out of the heads' right side, down X_HC and along its own lane). The
  // live pass's own prediction (the line into the box) feeds no command and no loss in a run whose live-head loss
  // weight is 0.
  // the heads' right edge, from its middle down: the eye command in (efference heads), the line out to curiosity 13
  // units lower, the line out to wellness 26 units lower (at most 4 above the box's bottom)
  const hMid = hdB.y + hdB.h / 2, hCur = hMid + 13, hWel = Math.min(hMid + 26, hdB.y + hdB.h - 4);
  routeSolid("heads", "wellness", [[OR, hWel], [X_HW, hWel], [X_HW, qHeads], [OR, qHeads]]);
  if (nodes.td_gate) {
    const tg = nodes.td_gate, tgx = tg.x + tg.w / 2, above = tg.y - 14, hTopX = hdB.x + 28, overF = frameTop[0] - 18;
    routeSolid("heads", "td_gate", [[hTopX, hdB.y], [hTopX, overF], [572, overF], [572, above], [tgx, above], [tgx, tg.y]]);
  }
  routeSolid("heads", "curiosity", [[OR, hCur], [X_HC, hCur], [X_HC, lane("heads_curiosity")],
                                    [nodes.curiosity.x + 105, lane("heads_curiosity")], [nodes.curiosity.x + 105, stripY]]);
  // the planner's command replaces the controller's: up its own lane and a corridor 16 units left of the output column,
  // into the motor controller's left edge 10 units above its bottom
  routeSolid("intention", "motor", [[iB.x + 160, stripY], [iB.x + 160, lane("intention_motor")],
                                    [outX - 16, lane("intention_motor")], [outX - 16, M0 + moB.h - 10], [outX, M0 + moB.h - 10]]);
  // the ranked goal-ring values reach the planner
  routeSolid("wellness", "intention", [[wlB.x + 40, wellBot], [wlB.x + 40, lane("wellness_intention")],
                                       [iB.x + 105, lane("wellness_intention")], [iB.x + 105, stripY]]);
  // the executed command and the eye command into the world (dashed: they act over the next model step's physics),
  // each on its own corridor and its own lane of the top band
  route("motor", "world", toWorld("motor", OR, pW, X_WD));
  route("gaze", "world", toWorld("gaze", OR, gzB.y + 12, X_GW));
  // THE CURRICULUM: the phase's reach and strength limits clamp every command the body receives, inside the step
  // (solid); a phase change rebuilds the world's body at the new age and delays (dashed), which a run that starts in
  // the final phase with its age fixed and its delays off can never do, so such a run draws no line to the world
  if (nodes.curriculum) {
    const cuB = nodes.curriculum;
    routeSolid("curriculum", "motor", [[X_CU, stripY], [X_CU, pCur], [OR, pCur]]);
    if (topIn.includes("curriculum"))
      route("curriculum", "world", toWorld("curriculum", cuB.x + cuB.w, cuB.y + cuB.h / 2, X_CWD));
  }
  // THE SIDE HEADS. Each is handed the command executed over the interval that just ended (lem_model.py exec_t, from the
  // trainer's prev_executed): dashed, each from its own port. Each reads the delayed re-run's core output inside the
  // step: solid, each up from its own exit on the final LayerNorm's top and over the output column on its own corridor
  if (nodes.efference) {
    const e = nodes.efference, fx = fnB.x + fnB.w - 9, over = hdB.y - 12;
    route("motor", "efference", [[OR, pSide], [e.x, pSide]]);
    routeSolid("final_norm", "efference", [[fx, fnB.y], [fx, over], [e.x + e.w / 2, over], [e.x + e.w / 2, e.y]]);
    if (G.eff_gaze_copy) {
      const eyeY = hdB.y + hdB.h / 2;
      routeSolid("efference", "heads", [[e.x, e.y + 8], [X_EH, e.y + 8], [X_EH, eyeY], [OR, eyeY]]);
      const label = el("text", { x: (OR + X_WD) / 2, y: eyeY - 7,
        "text-anchor": "middle", class: "sublabel clicky", text: "eye command → vision" });
      label.addEventListener("click", () => showDoc("edge.efference.heads"));
      svg.appendChild(label);
    }
    // THE HEADS' READINGS IN THE GOAL RANKING (EFF_ROUTES.goal_channels): the controllability and learning-progress
    // channels are stored with every goal ring entry and ranked when the planner's goal is picked. Dashed: the
    // expectation is formed when a command is sent and judged when it ends. Drawn LAST, over a white halo (below)
    if (G.eff_goal && nodes.wellness) {
      const gap = e.y + e.h + 5;
      effGoalPts = [[e.x + 12, e.y + e.h], [e.x + 12, gap], [X_EW, gap], [X_EW, qEff], [OR, qEff]];
    }
    // THE BODY HEAD IN THE ACTION SEARCH (EFF_ROUTES.action_search, --physical-goals memory): called for every
    // candidate command inside the planning step, so solid; its own lane to the planner's top edge
    if (effSearch && iB) {
      routeSolid("efference", "intention", [[e.x + 24, e.y], [e.x + 24, e.y - 8], [X_EI, e.y - 8],
                                            [X_EI, lane("efference_intention")], [iB.x + 195, lane("efference_intention")],
                                            [iB.x + 195, stripY]]);
    }
  }
  if (nodes.contact_head) {
    const c = nodes.contact_head, fx = fnB.x + fnB.w - 21, over2 = hdB.y - 24;
    route("motor", "contact_head", [[OR, pCon], [X_SB, pCon], [X_SB, c.y + 20], [c.x, c.y + 20]]);
    routeSolid("final_norm", "contact_head", [[fx, fnB.y], [fx, over2], [X_CT, over2], [X_CT, c.y + 20],
                                              [c.x + c.w, c.y + 20]]);
    // the contact-control channel (lem_model.py _contact_control_value): payments for touch onsets the head predicted
    // from the previous model step, stored with each goal ring entry and ranked - dashed, like the efference heads'
    route("contact_head", "wellness", [[c.x, c.y + 40], [X_CW, c.y + 40], [X_CW, qCon], [OR, qCon]]);
  }
  // The hindsight lesson remains connected while both control switches are off. It uses completed physical movements,
  // not the motor encoder's token or the general replay ring. Out of the world box's top at its own x, along the lowest
  // lane of the top band.
  if (nodes.synergy_examples) {
    const ex = nodes.synergy_examples, xo = outX0("synergy_examples"), yo = topLane("synergy_examples");
    route("world", "synergy_examples", [[xo, wB.y], [xo, yo], [X_SE, yo], [X_SE, ex.y + ex.h / 2],
                                        [ex.x + ex.w, ex.y + ex.h / 2]]);
    edge("synergy_examples", "synergy");
  }
  // the synergy as the planner's action space: the planner adjusts its few numbers and the synergy turns them into the
  // candidate command - its own lane to the planner's top edge
  if (nodes.synergy && synPlan) {
    const sB = nodes.synergy;
    routeSolid("synergy", "intention", [[sB.x, sB.y + sB.h / 2], [sB.x - 10, sB.y + sB.h / 2],
                                        [sB.x - 10, lane("synergy_intention")], [iB.x + 180, lane("synergy_intention")],
                                        [iB.x + 180, stripY]]);
  }
  if (nodes.synergy && topIn.includes("synergy")) {
    const sB = nodes.synergy;
    route("synergy", "world", toWorld("synergy", sB.x + sB.w, sB.y + sB.h / 2, X_BB));
  }
  // THE SECOND LEARNING CHANNEL: every model step writes a row into the replay ring (solid, along the corridor
  // coreH - 8), and every --replay-every model steps one extra AdamW update comes back out (dashed). The update reaches
  // the motor encoder, the core, the final LayerNorm, the prediction heads and the motor controller (the replay box's
  // entry lists them); the line ends at the final LayerNorm, the core's output
  routeSolid("ln_out", "replay", [[lnExit(1), lnBot], [lnExit(1), coreH - 8], [nodes.replay.x + 10, coreH - 8],
                                  [nodes.replay.x + 10, stripY]]);
  route("replay", "final_norm", [[nodes.replay.x + 100, stripY], [nodes.replay.x + 100, fnBot]]);
  // THE IMAGINATION CHAIN (dashed): it starts from the model step 3 back and replays the commands executed then. Its
  // win rate sets the planner's depth only when --hand-target-mode is "world"; otherwise the search looks one model
  // step ahead and no line runs to the planner
  if (G.imagination && nodes.imagination) {
    const imB = nodes.imagination;
    route("ln_out", "imagination", [[lnExit(0), lnBot], [lnExit(0), coreH - 20], [imB.x + 20, coreH - 20],
                                    [imB.x + 20, stripY]]);
    // one of the executed command's comb (data-bus "exec", below)
    route("motor", "imagination", [[OR, pImag], [X_IM, pImag], [X_IM, coreH - 20], [imB.x + 160, coreH - 20],
                                   [imB.x + 160, stripY]], "exec");
    if (G.imag_depth) route("imagination", "intention", [[imB.x, stripY + 20], [iB.x + iB.w, stripY + 20]]);
  }
  // the drive reaches the PLANNER, not wellness: under --wellness-rank normalized it is one goal-ranking channel
  routeSolid("curiosity", "intention", [[nodes.curiosity.x + nodes.curiosity.w, stripY + 20], [iB.x, stripY + 20]]);
  // THE BOTTOM ROW's lines enter each box from the gap beside it (30 units wide), each along its own lane: the
  // persistent memory's recall bias is added AFTER the core's last LayerNorm to the newest tokens only; the memory, self
  // perception and the hierarchical self read this step's raw tokens off the shared LayerNorm; the voice head reads the
  // core's output - all solid
  if (nodes.memory) {
    const mB = nodes.memory;
    routeSolid("memory", "final_norm", [[mB.x + mB.w, mB.y + 12], [mB.x + mB.w + 8, mB.y + 12],
                                        [mB.x + mB.w + 8, lane("memory_final_norm")], [fnB.x + 20, lane("memory_final_norm")],
                                        [fnB.x + 20, fnBot]]);
    // into the left edge 32 units down: the tactile and the motor sense's raw readings enter 12 and 22 down, the executed
    // command 42 down (the input check of Oct 15, 2026, below)
    routeSolid("ln_out", "memory", [[lnExit(3), lnBot], [lnExit(3), lane("ln_memory")], [mB.x - 15, lane("ln_memory")],
                                    [mB.x - 15, mB.y + 32], [mB.x, mB.y + 32]]);
  }
  if (nodes.self_state) {
    const ssB = nodes.self_state;
    // down the gap between the curiosity and planner boxes 14 units left of self perception: the curiosity -> planner
    // arrow crosses that gap, and a corridor at 10 units met its arrowhead 10 units from the tip (the layout check's
    // "line crosses an arrowhead", Oct 15, 2026); 8 units right of the memory -> final LayerNorm corridor
    routeSolid("ln_out", "self_state", [[lnExit(4), lnBot], [lnExit(4), lane("ln_self")], [ssB.x - 14, lane("ln_self")],
                                        [ssB.x - 14, ssB.y + 30], [ssB.x, ssB.y + 30]]);
  }
  if (nodes.self_state && nodes.self_learning) {
    edge("self_state", "self_learning");
    const sl = nodes.self_learning, ss = nodes.self_state;
    if (!argsm.eval_only) route("self_learning", "self_state", [[sl.x + 28, sl.y + sl.h], [sl.x + 28, sl.y + sl.h + 8], [ss.x + ss.w - 28, sl.y + sl.h + 8], [ss.x + ss.w - 28, ss.y + ss.h]]);
    if (nodes.self_report) {
      const sr = nodes.self_report;
      routeSolid("self_learning", "self_report", [[sl.x + sl.w - 24, sl.y + sl.h], [sl.x + sl.w - 24, sr.y - 6], [sr.x + sr.w - 24, sr.y - 6], [sr.x + sr.w - 24, sr.y]]);
    }
    if (selfMode === "perceive") {
      const ss = nodes.self_state;
      routeSolid("self_state", "final_norm", [[ss.x + ss.w - 12, ss.y], [ss.x + ss.w - 12, lane("self_final")],
          [fnB.x + 32, lane("self_final")], [fnB.x + 32, fnBot]]);
    }
  }
  if (nodes.voice) {
    const vB = nodes.voice;
    routeSolid("final_norm", "voice", [[fnB.x + 50, fnBot], [fnB.x + 50, lane("final_norm_voice")],
                                       [vB.x - 15, lane("final_norm_voice")], [vB.x - 15, vB.y + 20], [vB.x, vB.y + 20]]);
  }
  // both of the hierarchical self's inputs are read inside the SAME step() call as the forward pass (LemLite.step hands
  // the observer this step's raw tokens and this step's wellness), so neither line is dashed
  if (nodes.hself && G.hself) {
    const hsB = nodes.hself, xr = hsB.x + hsB.w + 15;
    routeSolid("wellness", "hself", [[wlB.x + 56, wellBot], [wlB.x + 56, lane("wellness_hself")],
                                     [xr, lane("wellness_hself")], [xr, hsB.y + 20], [hsB.x + hsB.w, hsB.y + 20]]);
    routeSolid("ln_out", "hself", [[lnExit(5), lnBot], [lnExit(5), lane("ln_hself")], [hsB.x - 14, lane("ln_hself")],
                                   [hsB.x - 14, hsB.y + 14], [hsB.x, hsB.y + 14]]);
  }
  // THE NAMED THINGS AND THE READER (--entity-self dark). Its teacher is each body's wellness, read inside the same
  // step; the reader adds its value to the model step stored in the goal ring, also inside the step; and the world's
  // frame report of who and what is present (lem_model.py entity presence, the caregiver's report among it) arrives
  // with the readings, on the world's trunk (one of the drawing's three combs) - all solid. The wellness line enters
  // from the gap on its left, or on its right when the hierarchical self's line takes the left gap
  if (nodes.entity_self) {
    const esB = nodes.entity_self;
    const leftTaken = !!nodes.hself && nodes.hself.x + nodes.hself.w + 30 === esB.x;
    const ex = leftTaken ? esB.x + esB.w + 15 : esB.x - 15, ein = leftTaken ? esB.x + esB.w : esB.x;
    routeSolid("wellness", "entity_self", [[wlB.x + 72, wellBot], [wlB.x + 72, lane("wellness_entity")],
                                           [ex, lane("wellness_entity")], [ex, esB.y + 20], [ein, esB.y + 20]]);
    const gutY = gut("gut_a");
    routeSolid("entity_self", "intention", [[esB.x + esB.w / 2, esB.y], [esB.x + esB.w / 2, gutY], [iB.x + 130, gutY],
                                            [iB.x + 130, iB.y + iB.h]]);
    const yb = rowsBottom + 12;
    routeSolid("world", "entity_self", [[TRUNK_X, wB.y + wB.h], [TRUNK_X, yb], [esB.x + esB.w / 2, yb],
                                        [esB.x + esB.w / 2, esB.y + esB.h]], "world-trunk");
  }
  // THE INPUT CHECK OF OCT 15, 2026: the inputs the drawing lacked. Each was read in the code before it was drawn
  // (docs/explainer_inputs_2026-10-15: the claim ledger quotes each code line), and each line opens its own entry.
  const sT = nodes.sense_tactile, sM = nodes.sense_motor, mB2 = nodes.memory, ssB2 = nodes.self_state,
        imB2 = nodes.imagination, rpB = nodes.replay, eM = nodes.enc_motor;
  // THE TACTILE SENSE'S RAW READINGS, read without the encoder (comb "tactile-raw"): out of the box's left edge 11
  // units above its bottom, across the world's trunk and down x 4, to wellness's skin-comfort and self-contact scores (lem_model.py hands
  // wellness extras["raw_tactile"] and frame["tactile"]), to persistent memory's measured needs
  // (memory_integration.measured_needs) and to self perception's per-site skin fields
  // (self_state_integration.batch_observation)
  if (sT) {
    const ey = sT.y + sT.h - 11, head = [[sT.x, ey], [X_TACT, ey]];
    routeSolid("sense_tactile", "wellness", head.concat([[X_TACT, lane("tactile_raw")], [wlB.x + 12, lane("tactile_raw")],
                                                         [wlB.x + 12, wellBot]]), "tactile-raw");
    if (ssB2) routeSolid("sense_tactile", "self_state", head.concat([[X_TACT, gut("gut_b")], [ssB2.x + 30, gut("gut_b")],
                                                                    [ssB2.x + 30, ssB2.y]]), "tactile-raw");
    if (mB2) routeSolid("sense_tactile", "memory", head.concat([[X_TACT, mB2.y + 12], [mB2.x, mB2.y + 12]]),
                        "tactile-raw");
  }
  // THE MOTOR SENSE'S RAW READINGS, read without the encoder (comb "motor-raw"): out of the box's left edge, across the
  // world's trunk and down x 12, to the motor controller (the handle state it concatenates, motor_controller.py), to wellness (the
  // handle-position score and the capability score), to persistent memory's capacity need, to the replay ring (each
  // row's achieved pose) and, under --hand-goals memory or --physical-goals memory, to the planner (the goal ring's
  // stored poses and this model step's, lem_model.py _goal_ring_pose and selected_pose); the last two drop from the
  // motor-raw lane into the replay box's and the planner's top edges
  if (sM) {
    const ey = sM.y + sM.h - 11, head = [[sM.x, ey], [X_MOTS, ey]], lnM = lane("motor_raw"),
          base = head.concat([[X_MOTS, lnM]]), trunk = base.concat([[X_MW, lnM]]);
    routeSolid("sense_motor", "motor", trunk.concat([[X_MW, M0 + 14], [outX, M0 + 14]]), "motor-raw");
    routeSolid("sense_motor", "wellness", base.concat([[wlB.x + 24, lnM], [wlB.x + 24, wellBot]]), "motor-raw");
    if (mB2) routeSolid("sense_motor", "memory", head.concat([[X_MOTS, mB2.y + 22], [mB2.x, mB2.y + 22]]), "motor-raw");
    routeSolid("sense_motor", "replay", base.concat([[rpB.x + 30, lnM], [rpB.x + 30, stripY]]), "motor-raw");
    if (G.hand_goal_pose)
      routeSolid("sense_motor", "intention", base.concat([[iB.x + 60, lnM], [iB.x + 60, stripY]]), "motor-raw");
  }
  // THE PLANNER'S STARTING COMMAND (solid: this model step's motor controller output, lem_model.py motor_out handed to
  // _plan_chain as current_motor; each search starts from it): out of the controller's bottom edge into the 10-unit
  // gap above the gaze box, left to its own corridor 8 units left of the output column (16 units right of the final
  // LayerNorm's comb corridor), down it to its own lane, and into the planner's top edge
  {
    const moBot = M0 + moB.h, yMI = Math.round((moBot + gzB.y) / 2), X_MI = outX - 8;
    routeSolid("motor", "intention", [[outX + 10, moBot], [outX + 10, yMI], [X_MI, yMI], [X_MI, lane("motor_intention")],
                                      [iB.x + 120, lane("motor_intention")], [iB.x + 120, stripY]]);
  }
  // THE EXECUTED COMMAND (comb "exec", dashed: the command of the interval that just ended, train_mimo's
  // prev_executed, arrives as lem_model.py exec_t one model step after the controller wrote it): out of the motor
  // controller's port beside the imagination chain's line, down X_IM to its lane, then into the motor encoder (the
  // command half of the motor row, re-encoded by the delayed next-model-step lesson and the context back-fill), into
  // persistent memory (the event's command columns) and into self perception (the forecast's command sequence)
  {
    const exA = [[OR, pImag], [X_IM, pImag], [X_IM, lane("exec")]];
    if (eM) route("motor", "enc_motor", exA.concat([[X_EM, lane("exec")], [X_EM, eM.y + 10], [eM.x + eM.w, eM.y + 10]]),
                  "exec");
    if (mB2) route("motor", "memory", exA.concat([[mB2.x - 23, lane("exec")], [mB2.x - 23, mB2.y + 42],
                                                  [mB2.x, mB2.y + 42]]), "exec");
    if (ssB2) route("motor", "self_state", exA.concat([[ssB2.x + ssB2.w + 23, lane("exec")],
                                                      [ssB2.x + ssB2.w + 23, ssB2.y + 12],
                                                      [ssB2.x + ssB2.w, ssB2.y + 12]]), "exec");
    // the replay ring's row: the executed command is back-filled into the row one model step later (replay.add)
    route("motor", "replay", exA.concat([[rpB.x + 50, lane("exec")], [rpB.x + 50, stripY]]), "exec");
  }
  // THE CORE'S OUTPUT INTO SELF PERCEPTION, dashed: lem_model.py stores the final LayerNorm's newest tokens (memory
  // bias added), averaged over the senses, as self_history.internal, and the next model step's perceive() reads it.
  // Down the gap right of the imagination box (7 units from it, 8 from the wellness -> named things corridor), along the
  // gutter's lane gut_b and into self perception's top edge: the gap left of the imagination box is crossed by the
  // imagination -> planner arrow, whose arrowhead a corridor there met 7 units from its tip
  if (ssB2)
    route("final_norm", "self_state", [[X_FS, fnBot], [X_FS, lane("final_self")],
                                       [imB2.x + imB2.w + 7, lane("final_self")], [imB2.x + imB2.w + 7, gut("gut_b")],
                                       [ssB2.x + 170, gut("gut_b")], [ssB2.x + 170, ssB2.y]]);
  // THE WORLD'S OWN READINGS, on the world's trunk: the frame's simulated time into persistent memory (each event's
  // time stamp), and the body-function, actuator and skin-site readings into self perception
  const ybW = rowsBottom + 12;
  if (mB2) routeSolid("world", "memory", [[TRUNK_X, wB.y + wB.h], [TRUNK_X, ybW], [mB2.x + mB2.w / 2, ybW],
                                          [mB2.x + mB2.w / 2, mB2.y + mB2.h]], "world-trunk");
  if (ssB2) routeSolid("world", "self_state", [[TRUNK_X, wB.y + wB.h], [TRUNK_X, ybW], [ssB2.x + ssB2.w / 2, ybW],
                                              [ssB2.x + ssB2.w / 2, ssB2.y + ssB2.h]], "world-trunk");
  if (nodes.latent_body && selfReviewedSetting("latent_body_hierarchy") === "shadow") {
    const lb = nodes.latent_body;
    route("world", "latent_body", [[TRUNK_X, wB.y + wB.h], [TRUNK_X, ybW], [lb.x + lb.w / 2, ybW], [lb.x + lb.w / 2, lb.y + lb.h]], "world-trunk");
  }
  if (nodes.hself && !GATE_RUN.hself) svg.querySelector("#nd-hself").classList.add("offrun");
  if (nodes.latent_body) {
    const lbMode = selfReviewedSetting("latent_body_hierarchy");
    if (lbMode && lbMode !== "off") svg.querySelector("#nd-latent_body").classList.add("learned");
    else svg.querySelector("#nd-latent_body").classList.add("offrun");
  }
  if (nodes.self_choice && (argsm.self_choice || "off") === "off") svg.querySelector("#nd-self_choice").classList.add("offrun");
  // THE WORLD'S CONSEQUENCE COUNT into the goal ring's value (--consequence-w above 0): lem_model.py adds
  // consequence_weight x consequence_kinds(the frame's task report) to each stored value; on the world's trunk down to
  // the gutter's first lane and up into the planner's bottom edge
  if (G.consequence)
    routeSolid("world", "intention", [[TRUNK_X, wB.y + wB.h], [TRUNK_X, gut("gut_w")], [iB.x + 40, gut("gut_w")],
                                      [iB.x + 40, iB.y + iB.h]], "world-trunk");
  // THE WORLD'S CONTACT REPORT into the contact head (env.py _contact_report: 53 patch states per body): out of the
  // world box's top at its own slot, along its own lane of the top band, down the corridor right of the side column
  if (nodes.contact_head) {
    const c = nodes.contact_head, xo = outX0("contact"), yo = topLane("contact");
    routeSolid("world", "contact_head", [[xo, wB.y], [xo, yo], [X_WC, yo], [X_WC, c.y + 34], [c.x + c.w, c.y + 34]]);
  }
  // THE TOKENS THE PLANNER AND THE IMAGINATION CHAIN READ (comb "gated"): the goal ring stores this model step's
  // tokens after the top-down gate (lem_model.py goal_ring.append and current_state), and the imagination chain starts
  // from the same set three model steps later (enc_history); with the gate off they are the shared LayerNorm's tokens
  if (nodes.td_gate) {
    const tg = nodes.td_gate, gy = tg.y + tg.h - 10,
          gHead = [[tg.x + tg.w, gy], [X_GT, gy], [X_GT, lane("gated")]];
    // the dashed branch first, so the solid line drawn over it shows the shared stretch solid (the planner reads the
    // set inside the step) and only the branch into the imagination chain reads dashed
    if (G.imagination && imB2)
      route("td_gate", "imagination", gHead.concat([[imB2.x + 40, lane("gated")], [imB2.x + 40, stripY]]), "gated");
    routeSolid("td_gate", "intention", gHead.concat([[iB.x + 140, lane("gated")], [iB.x + 140, stripY]]), "gated");
  } else {
    const gy = lnB.y + lnB.h - 8;
    routeSolid("ln_out", "intention", [[lnB.x + lnB.w, gy], [X_GT, gy], [X_GT, lane("gated")],
                                       [iB.x + 140, lane("gated")], [iB.x + 140, stripY]]);
  }
  // THE RAW TOKENS INTO THE PREDICTION HEADS (--residual-prediction on): each head's output is added to its sense's
  // raw token (prediction_heads.py), the live pass's from this model step and the delayed lesson's re-encoded from the
  // previous one. Out of the shared LayerNorm's top edge 14 units from its right end, right along the 22-unit gap under
  // the top-down gate (7 units under it, 6 over the token assembly's numbered circle), up past the caregiver to the
  // lowest lane of the top band, and down into the heads' top edge
  if (G.residual) {
    const ry = lnB.y - 15, hx = hdB.x + hdB.w - 14, ty = topLane("residual");
    routeSolid("ln_out", "heads", [[X_RS, lnB.y], [X_RS, ry], [X_RU, ry], [X_RU, ty], [hx, ty], [hx, hdB.y]]);
  }
  // PERSISTENT MEMORY INTO REPLAY (solid: each model step's memory state is written into the replay ring with its row,
  // and re-study turns it back into the memory bias), and THE PLANNER'S LAST GOAL INTO PERSISTENT MEMORY (dashed: the
  // goal tokens of the last planning step, averaged over the senses, are an event's goal columns), both along the
  // gutter between the two strip rows
  if (mB2) {
    routeSolid("memory", "replay", [[mB2.x + 190, mB2.y], [mB2.x + 190, gut("gut_c")], [rpB.x + 120, gut("gut_c")],
                                    [rpB.x + 120, rpB.y + rpB.h]]);
    route("intention", "memory", [[iB.x + 20, iB.y + iB.h], [iB.x + 20, gut("gut_a")], [mB2.x + 170, gut("gut_a")],
                                  [mB2.x + 170, mB2.y]]);
  }
  // the efference heads' goal-ranking line, over the lines it must cross (see its route above): the
  // halo stops short of both boxes, so it never cuts their borders
  if (effGoalPts) {
    const n = effGoalPts.length, [lx, ly] = effGoalPts[n - 1], [px, py] = effGoalPts[n - 2];
    const run = Math.hypot(lx - px, ly - py) || 1;
    const halo = effGoalPts.map(p => p.slice());
    halo[0] = [halo[0][0], halo[0][1] + 3];
    halo[n - 1] = [lx - (lx - px) / run * 9, ly - (ly - py) / run * 9];
    svg.appendChild(el("path", { class: "dg-halo", "data-for": "efference-wellness",
                                 d: "M" + halo.map(p => p.join(" ")).join(" L") }));
    route("efference", "wellness", effGoalPts);
  }
  // Numbered circles: the sequence of operations in THIS model step's live pass, numbered
  // where the program does the work.
  //  - The shared LayerNorm gets no number: it runs INSIDE every encoder call
  //    (encoder_registry._encode_one applies output_norm before returning), so
  //    numbering it made "run the encoder" carry two step numbers for one operation.
  //  - All six senses carry the first two numbers: one call encodes them together, so
  //    numbering vision alone made five senses look as if they joined later.
  //  - The persistent memory observes between the encoders and the forward model
  //    (LemLite.step in lem_model.py calls memory_integration.observe before the
  //    top-down gate block), so it takes its own number.
  //  - Wellness (LemLite.step's wellness evaluation) runs after the prediction heads and before the motor controller,
  //    which reads its slope, so it takes its own number between them (the code check of Oct 15, 2026).
  //  - The motor controller (the self.motor_controller call in LemLite.step) writes
  //    the command the world executes;
  //    it carried no number while the caption handed off from the prediction heads,
  //    which output next-model-step tokens and never touch the body.
  let bn = 0;
  for (const ids of [mods.map(m => "sense_" + m), mods.map(m => "enc_" + m),
                     ["memory"], ["td_gate"], ["assembly"], ["block0_s1"],
                     ["final_norm"], ["heads"], ["wellness"], ["motor"], ["world"]]) {
    const present = ids.filter(i => nodes[i]);
    if (!present.length) continue;
    bn++;
    present.forEach(i => badge(i, bn));
  }
  // the last numbered step is not part of the model, and the caption saying so is the
  // page's own account of it - so it opens that account instead of ending the chain
  // in text nothing can question
  // right of the caregiver box, and right of the raw tokens' line up to the top band (x 520) when it is drawn
  const capX = Math.max(careTop ? nodes.caregiver.x + nodes.caregiver.w + 20 : 300, G.residual ? X_RU + 14 : 0);
  const worldCap = el("text", { x: capX, y: nodes.world.y + 14, class: "sublabel clicky" });
  wrapText(bn + " · the final executed command acts in the world; the next sensor readings "
           + "come back through the senses, where the losses compare them with the predictions", SUB_PX, 560)
    .forEach((ln, k) => worldCap.appendChild(el("tspan", { x: capX, dy: k ? SUB_STEP : 0, text: ln })));
  worldCap.addEventListener("click",
    () => showDoc("diagram.world_step", "the loop closes in the world"));
  svg.appendChild(worldCap);
  // THE CAPTION BAND under the picture. Each sentence is a list of runs - [words, class, click
  // handler] - wrapped as one block to the drawing's width from its measured text (wrapRuns): lines
  // 12 units apart, 3 more between sentences, and the drawing's height set to hold the last line.
  // "numbered circles" opens what the circles LEAVE OUT: the passes that run over the same boxes
  // without a number (the delayed next-frame lesson, imagination, re-study).
  const foot = [
    [["numbered circles", "docword", () => showDoc("diagram.numbered_chain", "the numbered chain")],
     [" = the sequence of operations in this model step's live pass · bold blue box = weights a lesson "
      + "reaches, count at the corner · grey count = weights no lesson reaches · click any BOX or LINE for "
      + "its data", "", null]],
    [["a hand-routed line is dashed when its data carries from one model step into the NEXT (the executed command "
      + "into the world, the side heads, the motor encoder, persistent memory and self perception, re-study's update, "
      + "a phase change, the imagination chain, the core's output into self perception, the planner's last goal into "
      + "persistent memory, the side heads' goal-ranking channels) and solid when it is built and used inside one step",
      "", null]],
    [["lines share a stretch only inside a comb, one source fanning out to several boxes or several sources into one: "
      + "the world's trunk, the tactile sense's raw readings and, beside them, the motor sense's (each read without an "
      + "encoder, each down its own corridor left of the world's trunk), the executed "
      + "command, the tokens the planner and the imagination chain read, the encoders into the shared LayerNorm and "
      + "the final LayerNorm's three outputs", "", null]],
    [["the prediction heads' three lines out (to the top-down gate, wellness and curiosity) carry the delayed "
      + "re-run's prediction of this model step, made inside this step from the previous model step's readings",
      "", null]],
    [["one pass per model step is what the circles number, not what runs: the delayed next-frame lesson "
      + "re-runs the encoders, the core and the heads on the PREVIOUS model step's readings before this "
      + "pass, the imagination chain runs the core again once per predicted model step, and the motor "
      + "controller runs twice", "", null]],
  ];
  if (nodes.latent_body && selfReviewedSetting("latent_body_hierarchy") === "shadow")
    foot.push([["Latent-body learns separate saved networks beside this model. Their weights are outside the main model receipt's totals below.", "", null]]);
  if (Object.keys(CP).length) {
    // NOT DRAWN = what no drawn box accounted for (drawnComps, filled in by node()).
    // This used to be a second, literal list of eight names plus one per sense, and the
    // shared LayerNorm's own numbers - drawn as the ln_out box - were on neither list,
    // so a run with the norm's gain and bias switched on printed them here as not drawn.
    // What each not-drawn part IS, so a bare count is never read as evidence of
    // learning, and a small count is never read as a count of PARTS. gaze_head's
    // 83,202 numbers are registered in the optimizer and nothing ever calls the
    // module (grep for "self.gaze_head(" finds no call site), so they are inside the
    // optimizer's count and can never move. td_gates is five 3-number gates, one per
    // predicted sense, and the motor sense has none - "td gates 15" read as fifteen.
    const gz = (((rec.checkpoint_inventory || {}).components || {}).gaze_head || {}).parameters;
    // WITH --efference-heads on the GazeHead is not idle (review b1, minor 5): its attention and
    // layers are the efference gaze head's, whose 50 ms eye-turn lesson trains them - unless the
    // receipt says the heads were built with a gaze pool of their own (gaze_head_repurposed false;
    // schema.gaze_head_reused reads the same field)
    const effStamp = rec.efference_heads;
    const effOn = !!effStamp && (typeof effStamp !== "object" || effStamp.gaze_head_repurposed !== false);
    const nGate = (rec.prediction_head_modalities || []).length;
    const NOTE = {
      gaze_head: effOn
        ? " (reused as the efference gaze head, whose eye-turn lesson trains"
          + (gz ? " those " + gz.toLocaleString() : " those") + " numbers)"
        : " (in the optimizer's list and never called, so"
          + (gz ? " those " + gz.toLocaleString() : " those")
          + " numbers can never move)",
      td_gates: " numbers" + (nGate ? " - " + nGate + " gates of 3 numbers, one per predicted"
                                      + " sense; the motor sense has none" : ""),
    };
    // a not-drawn part whose name opens its own popup: three words in a footer list
    // is not an account of a 375,758-number part of the model
    const DOCKEY = { tenure_heads: "node.tenure_heads", gaze_head: "node.gaze",
                     td_gates: "node.td_gate",
                     // the largest part no box draws (2,014,180 numbers on H84's receipt of Oct 7, 2026): it puts its
                     // units inside many layers, so it opens an entry of its own instead (D-E4)
                     liveness_repair: "part.liveness_repair" };
    const rest = Object.entries(CP).filter(([k]) => !drawnComps.has(k));
    const sum = Object.values(CP).reduce((a, b) => a + b, 0);
    // total_params counts every SAVED parameter; the optimizer's own figure is two
    // keys away in the receipt's inventory. They agree only while every parameter
    // tensor happens to be registered, and a frozen parameter breaks that silently.
    const inv = rec.checkpoint_inventory || {};
    const opt = inv.optimizer_parameters || rec.total_params;
    // the leftovers are not "calibration" of anything: the biggest block of them, by
    // far, is the causal masks - fixed true/false triangles saved into every checkpoint
    const masks = ((fm.stages || {}).causal_temporal === false) ? 0 : nL * ctx * ctx;
    const untrained = (opt && sum > opt) ? sum - opt : 0;
    const notDrawn = [["not drawn: ", "", null]];
    if (!rest.length) notDrawn.push(["nothing", "", null]);
    rest.forEach(([k, v], i) => {
      if (i) notDrawn.push([", ", "", null]);
      notDrawn.push([k.replace(/_/g, " ") + " " + pfmt(v), DOCKEY[k] ? "docword" : "",
                     DOCKEY[k] ? () => showDoc(DOCKEY[k], k.replace(/_/g, " ")) : null]);
      if (NOTE[k]) notDrawn.push([NOTE[k], "", null]);
    });
    foot.push(notDrawn);
    // The optimizer's count is printed WHENEVER the receipt carries one. It used to be
    // printed only when it differed from the parts sum, so a newer receipt - whose
    // component_params counts parameters only and therefore matches - would have
    // deleted the page's only statement of how much of the model the optimizer holds.
    foot.push([["all parts sum to " + sum.toLocaleString() + " " + (META.component_params_basis || "")
            + (opt ? " · the optimizer holds " + opt.toLocaleString() + " of them" : "")
            + (untrained ? "; the other " + untrained.toLocaleString() + " are saved but never trained"
               + (masks ? ", and " + masks.toLocaleString() + " of those are the " + nL
                  + " true/false masks that stop stage 2 reading the future" : "") : ""), "", null]]);
    // Those two bins - in the optimizer, or saved outside it - hide a third: numbers
    // the optimizer holds that no loss term reaches. Taken from the receipt, and the
    // line withdraws itself as soon as a snapshot shows a voice lesson in its losses.
    // The gaze head is the SAME case and sits in the same figure: its numbers are in
    // the optimizer's list and nothing ever calls the module, so subtracting only the
    // voice head's left this line claiming 83,202 numbers can learn while the
    // not-drawn line above it says those same numbers can never move.
    if (capVoice || idleSelf) {
      const gzIdle = effOn ? 0 : (gz || 0);
      const idle = (capVoice ? vhRec.params + gzIdle : 0) + idleSelf;
      foot.push([["of those " + opt.toLocaleString() + " optimizer numbers, "
            + (capVoice ? vhRec.params.toLocaleString() + " are the voice head's (no target is given, its loss weight is zero)"
                          + (gzIdle ? " and " + gzIdle.toLocaleString() + " are the gaze head's (nothing calls it)" : "") : "")
            // WHICH parts are idle is the run's own list (META.idle_params.groups, decided
            // by its --self-state and --self-prediction flags): under 'perceive' only the
            // legacy head is idle, and the sentence used to name both whatever the flags
            + (idleSelf ? (capVoice ? " and " : "") + idleSelf.toLocaleString()
                          + " are inside self perception ("
                          + (((((META.idle_params || {}).self_state) || {}).groups || [])
                               .map(g => String(g.why).split(" - ")[0]).join(" and ")
                             || "weights its flags leave without a lesson")
                          + ", switched off by this run's flags)" : "")
            + ", and no lesson reaches them, so the part of the model that can learn is " + (opt - idle).toLocaleString() + " numbers",
        "", null]]);
    }
    // The hierarchical self's slot table is in NEITHER figure: it holds no learned
    // weights, so component_params has no entry for it and the optimizer total
    // excludes it. Without this line a reader takes the gap between the two numbers
    // above for every stored number that sits outside the optimizer.
    if (capHself) {
      const hsN = argsm.hself_slots * mods.length
                  * D_MODEL;
      foot.push([["outside both of those counts: the hierarchical self's " + argsm.hself_slots.toLocaleString()
            + " slot centres hold " + hsN.toLocaleString() + " stored numbers, none of them learned "
            + "and none of them in the optimizer", "", null]]);
    }
  }
  // the caption band at the sub-caption's size (SUB_PX units), its lines 13 units apart
  let fy = bodyBottom + 17;                // the first sentence's first baseline
  for (const runs of foot) {
    const lines = wrapRuns(runs, SUB_PX, W - 32);
    const t = el("text", { x: 16, y: fy, class: "sublabel" });
    lines.forEach((ln, i) => ln.forEach(([s, cls, on], j) => {
      const a = { text: s };
      if (j === 0) { a.x = 16; a.dy = i ? 13 : 0; }
      if (cls) a.class = cls;
      const sp = el("tspan", a);
      if (on) sp.addEventListener("click", on);
      t.appendChild(sp);
    }));
    svg.appendChild(t);
    fy += 13 * lines.length + 3;
  }
  svg.setAttribute("viewBox", `0 0 ${W} ${fy - 7}`);   // the last line's descent and 6 units under it
  // the parts the reader switched on although this run has them off (GATES): faint and dashed, boxes and lines
  for (const [id, , , gBoxes, gLines] of GATES) {
    if (!(Object.prototype.hasOwnProperty.call(GATE_SET, id) && GATE_SET[id] && !GATE_RUN[id])) continue;
    gBoxes.forEach(b => { const n = svg.querySelector("#nd-" + b); if (n) n.classList.add("offrun"); });
    svg.querySelectorAll("path.dg-edge").forEach(p => {
      const a = p.getAttribute("data-a"), b = p.getAttribute("data-b");
      if (gBoxes.includes(a) || gBoxes.includes(b) || gLines.some(([x, y]) => x === a && y === b)) p.classList.add("offrun");
    });
  }
  root.appendChild(svg);
  if (["shadow", "on"].includes(argsm.self_choice)) {
    const actorRoute = el("div", {class: "sg-route", id: "self-choice-actor-route"});
    actorRoute.appendChild(el("h3", {text: "Self-choice " + argsm.self_choice + ": separate frozen actor"}));
    actorRoute.appendChild(el("p", {text: "Current world readings → frozen perception and command proposal → frozen self forecasts of timed commands → self-choice comparison."}));
    actorRoute.appendChild(el("p", {text: argsm.self_choice === "shadow"
      ? "Selected trial commands → report and discard. The ordinary actor supplies the body's commands."
      : "Selected commands → execution path → body. This separate qualified actor supplies the body's commands; the learner drawn above can update separately."}));
    actorRoute.appendChild(el("p", {text: "Frozen means that this actor's network weights remain fixed. Its self forecasts come from its own copy, not the updating learner's self state."}));
    root.appendChild(actorRoute);
  }

  // the orange rings of the parts being worked on, in their own layer behind the boxes (D-E2)
  drawWorkRings();
}

// ---------------------------------------------- goal, command and outcome
function buildLearnLoop() {
  const root = document.getElementById("learnloop");
  if (!root) return;
  root.textContent = "";
  const a = META.args || {};
  const planning = a.intention === "on", executing = planning && !a.no_intention_execute;
  const futureValue = a.prospective_wellness === "add" && Number(a.prospective_wellness_w) > 0;
  const comparison = a.causal_chain === "on", normalized = a.wellness_rank === "normalized";
  const h = el("h3", {}, "How a remembered goal becomes a new command");
  h.appendChild(docBtn("loop.overview", "Goal, command and outcome"));
  root.appendChild(h);
  root.appendChild(el("p", {class: "goal-lead"},
    "Wellness ranks remembered states. The planner searches for a new command from the body's current state. The outcome is measured again."));
  const flags = el("p", {class: "goal-run-state"},
    planning ? (executing ? "Planner execution on" : "Planner computes commands; execution off") : "Planner off: the planning path below is inactive",
    " · ", a.eval_only === true ? "Learner weights frozen" : a.eval_only === false ? "Learner weight updates enabled" : "Training setting not recorded",
    " · ", futureValue ? "Future wellness included in command search" : "Future wellness not an objective of command search");
  root.appendChild(flags);
  const cards = el("ol", {class: "goal-flow", "aria-label": "Goal selection and outcome, in order"});
  const data = [
    ["loop.goal", "Choose a remembered state",
     "Keep recent memories of what the body sensed and its wellness. Pick a valued memory; its sensory readings become the goal.",
     "The goal is a state to approach, not an old command to repeat."],
    ["loop.search", "Search for a new command",
     "Start from the current state and controller command. Adjust the command to make predicted senses match the goal, with a cost for effort.",
     a.hand_goals === "memory" ? "Then a separate hand search aims at the remembered hand pose." : "The remembered-hand search is off."],
    ["loop.execute", "Execute the final command",
     executing ? "The planner can supply the next command. Other commands can come from the controller, with exploration when enabled. Smoothing and a command hold affect what reaches the body."
               : "Planner commands do not reach the body in this configuration. The controller and execution settings determine the command.",
     "Learning receives the command actually executed."],
    ["loop.outcome", "Measure what happened",
     "Read the resulting senses and calculate wellness again. " + (comparison
       ? "Record predicted versus measured change for the five wellness terms that have predictions."
       : "Predicted-versus-measured wellness recording is off in this run."),
     "Reaching the remembered state need not bring back its old wellness."]
  ];
  data.forEach(([key, title, text, note], i) => {
    const head = el("h4", {}, el("span", {class: "goal-number", "aria-hidden": "true"}, String(i + 1)), title);
    head.appendChild(docBtn(key, title));
    cards.appendChild(el("li", {class: "goal-card", "data-goal-card": key}, head,
      el("p", {}, text), el("p", {class: "goal-card-note"}, note)));
  });
  root.appendChild(cards);
  const feedback = el("div", {class: "goal-feedback", "aria-label": "Two paths from the measured outcome"});
  for (const [key, title, text] of [
    ["loop.memory", "Outcome → future goal choices",
     "Store the new sensory readings and measured wellness in the goal ring. " + (normalized
       ? "Updated channel statistics affect the next ranking." : "Stored or revalued scores determine later goal choices.")],
    ["loop.learning", "Outcome → lessons for the model",
     a.eval_only === true ? "This run keeps weights frozen. It measures prediction errors without a weight update."
       : a.eval_only === false ? "Use previous senses and the executed command to predict the arriving senses. Prediction errors train the learner; the learned wellness copy has its own matching lesson."
       : "This run does not record whether training was enabled. During training, prediction errors from the executed command update the learner's weights."]
  ]) {
    const heading = el("h4", {}, title); heading.appendChild(docBtn(key, title));
    feedback.appendChild(el("div", {class: "goal-return"}, heading, el("p", {}, text)));
  }
  root.appendChild(feedback);
  root.appendChild(el("p", {class: "goal-caution"}, el("strong", {}, "Does the command bring wellness back? "),
    "That must be measured. Fatigue, contact and recent learning can change its value. " + (comparison
      ? "The recorded wellness comparison does not directly train a command-selection policy from success or failure."
      : "The wellness comparison recorder is off; new sensory experience can still inform learning and memory.")));
  const details = el("details", {class: "goal-details"});
  details.appendChild(el("summary", {}, "Timing, scoring and weight updates"));
  const ms = Number.isFinite(MOMENT_S) ? Math.round(1000 * MOMENT_S) : null;
  details.appendChild(el("p", {}, "An arriving sensory reading is the result of the previous command. During training, its prediction error updates weights before the next command is selected."));
  if (ms && a.intention_every && a.motor_hold)
    details.appendChild(el("p", {}, "This run: senses every " + ms + " ms; planning every "
      + (ms * a.intention_every) + " ms after goal-memory warm-up; body command held for "
      + (ms * a.motor_hold) + " ms. Gaze is handled separately."));
  details.appendChild(el("p", {}, a.wellness_rank === "normalized"
    ? "Goal ranking uses weighted wellness channels relative to recent history, with a harm restriction. Its score is different from the five-term wellness prediction shown in outcome records."
    : "Goal ranking uses this run's stored or revalued wellness scores. Its score is different from the five-term wellness prediction shown in outcome records."));
  if (futureValue) details.appendChild(el("p", {}, "This run also includes predicted future wellness in the command-search objective."));
  root.appendChild(details);
  buildWeightLoop(details);
}

// Prediction-error learning is a different loop from ranking remembered goals.
function buildWeightLoop(root) {
  const h = el("h3", {}, "How prediction errors change weights");
  h.appendChild(docBtn("bp.loop", "how learning flows"));
  root.appendChild(h);
  // The circle describes a TRAINING run. lem_model.py gates the backward, the
  // clipping and the optimizer step on train=True, and an --eval-only launch
  // never switches replay on at all - so on such a run stations 3, 4 and 5 do
  // not happen. The run's own args decide which of the two this page shows.
  // Every number drawn here is read off THIS run's args, receipt and snapshots. A
  // page built for a run launched with --action-repeat 1 must not print a
  // five-repeat model step length, and a run in which no weight moved must not be drawn
  // as a run that learned.
  const lArgs = META.args || {}, lRec = META.receipt || {};
  const lSnap = SNAPS.length ? SNAPS[SNAPS.length - 1] : null;
  const frozen = lArgs.eval_only === true;
  const trainingKnown = typeof lArgs.eval_only === "boolean";
  const replayOn = lArgs.eval_only === false && lArgs.replay === "on";
  if (!trainingKnown) root.appendChild(el("p", {},
    "This run does not record whether training was enabled. The diagram describes the training mechanism; it does not establish that these updates ran."));
  if (lSnap && lSnap.header && lSnap.header.weights === "frozen" && !frozen)
    root.appendChild(el("p", {}, "The latest snapshot records frozen weights. A frozen actor can provide these snapshots while a separate learner trains; the run settings below describe the learner."));
  const reps = lArgs.action_repeat || 5;           // printed beside the model step's measured length
  const momentMs = Math.round(1000 * MOMENT_S);
  const nEnv = lArgs.n_envs || (lSnap ? lSnap.header.n_envs : 0) || 1;
  const rEvery = lArgs.replay_every || 4, rBatch = lArgs.replay_batch || 48;
  const rFloor = Math.max(2, Math.floor(rBatch / 4));   // replay.py skips the pass below this
  const headM = lRec.prediction_head_modalities || [];
  const actM = lRec.active_modalities || [];
  const gradedM = (lSnap && lSnap.predictions)
    ? headM.filter(m => (lSnap.predictions.error_of_now || {})[m] !== undefined) : [];
  const band = frozen ? 46 : 0;
  const W = 920, H = band + 351;
  const svg = el("svg", { class: "mini", viewBox: `0 0 ${W} ${H}` });
  svg.appendChild(el("defs", {},
    el("marker", { id: "loopar", viewBox: "0 0 10 10", refX: 9, refY: 5,
                   markerWidth: 7, markerHeight: 7, orient: "auto" },
      el("path", { d: "M0 0 L10 5 L0 10 z", fill: "#6b7280" }))));
  if (frozen) {
    svg.appendChild(el("rect", { x: 8, y: 5, width: W - 16, height: 35, rx: 5,
                                 fill: "#fdf3d9", stroke: "#d8b45e" }));
    svg.appendChild(el("text", { x: 16, y: 20, class: "flowlabel", text:
      "EVALUATION ONLY: no training backward pass, weight update or replay update runs. "
      + "Control and planning can still calculate gradients." }));
    svg.appendChild(el("text", { x: 16, y: 33, class: "flowlabel", text:
      "The circle below is what a TRAINING run does. These weights carry "
      + (META.trained_step ? META.trained_step.toLocaleString() : "an unknown number of")
      + " training steps taken in another run"
      + (META.stats_source && META.stats_source !== META.run
         ? "; the Run tab's charts come from a third run." : ".") }));
  }
  const cx = 430, cy = band + 132, rx = 310, ry = 88;
  const pt = deg => [cx + rx * Math.cos(deg * Math.PI / 180),
                     cy + ry * Math.sin(deg * Math.PI / 180)];
  // one text element per label, its lines as tspans 11 units apart, so the layout check measures
  // the label as one block
  function arcLabel(key, title, lines, x, y, anchor) {
    const t = el("text", { x, y, "text-anchor": anchor || "middle", class: "flowlabel clicky" });
    lines.forEach((ln, i) => t.appendChild(el("tspan", { x, dy: i ? 11 : 0, text: ln })));
    t.addEventListener("click", ev => { ev.stopPropagation(); showDoc(key, title); });
    svg.appendChild(t);
  }
  // Clockwise quarter arcs. The trim at each end is 28 degrees, not 8: the station
  // boxes reach 14 to 22 degrees in and are drawn AFTER the arcs over an opaque fill,
  // so at 8 degrees all eight endpoints sat inside a box and not one of the four
  // arrowheads was ever visible. Each arc gets a label and a wide invisible twin
  // opening its data contract.
  const ARCS = [
    [-62, -28, "edge.bp.forward.grade", "1 \u2192 2: what flows into the grading",
     ["predictions from the previous state", "and the command actually executed"],
     598, band + 22, "start"],
    [28, 62, "edge.bp.grade.backward", "2 \u2192 3: what flows into backpropagation",
     ["ONE number: the lessons added up",
      lSnap && lSnap.losses ? "(no lived unit: the loss panel shows each lesson's share)"
                            : "(one loss for all bodies together)"], 598, band + 232, "start"],
    [118, 152, "edge.bp.backward.update", "3 \u2192 4: what flows into the weight update",
     ["main gradients limited together", "see popup for other gradients"],
     287, band + 228, "end"],
    [208, 242, "edge.bp.update.forward", "4 \u2192 1: what carries into the next circle",
     ["next command runs; its outcome returns", "at the next model step"],
     292, band + 22, "end"],
  ];
  // each label sits OUTSIDE its arc, clear of the arc and of the station boxes (placed from the
  // layout check of 2026-10-02, which samples the arcs: every label used to straddle its arc)
  for (const [a, b, key, title, lines, lx, ly, anchor] of ARCS) {
    const [x1, y1] = pt(a), [x2, y2] = pt(b);
    const d = `M${x1.toFixed(1)} ${y1.toFixed(1)} A ${rx} ${ry} 0 0 1 ${x2.toFixed(1)} ${y2.toFixed(1)}`;
    svg.appendChild(el("path", { class: "mini-edge", "marker-end": "url(#loopar)", d }));
    const hit = el("path", { class: "dg-hit", d });
    hit.addEventListener("click", ev => { ev.stopPropagation(); showDoc(key, title); });
    svg.appendChild(hit);
    arcLabel(key, title, lines, lx, ly, anchor);
  }
  // Each station hands showDoc this run's own numbers, the way every box on the
  // architecture diagram does; without them the circle was the only clickable thing
  // on the page whose popup was static prose.
  function station(x, y, w, l1, l2, key, muted, kv) {
    const g = el("g", { class: "mini-node clicky" });
    const r = el("rect", { x: x - w / 2, y: y - 21, width: w, height: 42, rx: 6, class: "mini-box" });
    if (muted) r.setAttribute("style", "fill:#f2f2ef;stroke:#bbbbbb");
    g.appendChild(r);
    g.appendChild(el("text", { x, y: y - 4, "text-anchor": "middle", text: l1 }));
    g.appendChild(el("text", { x, y: y + 11, "text-anchor": "middle", class: "sublabel", text: l2 }));
    g.addEventListener("click", () => showDoc(key, l1, kv && kv.length ? kv : null));
    svg.appendChild(g);
  }
  const snapRow = lSnap
    ? [["from snapshot", "step " + lSnap.header.step + " \u00b7 weights " + lSnap.header.weights]] : [];
  const kv1 = lSnap && lSnap.predictions
    ? snapRow.concat(Object.entries(lSnap.predictions.error_of_now || {})
        .map(([k, v]) => ["one-step error \u00b7 " + k, tokErr(v)]),
      [["causal_pe (mean over the senses graded)", tokErr(lSnap.predictions.causal_pe)]]) : [];
  const kv2 = lSnap && lSnap.losses
    ? snapRow.concat([["the one number backpropagation is given",
                       "a sum of lessons with no lived unit - the inspector's loss panel shows each lesson's share of it"],
        ["senses guessed / graded this step",
         headM.length + " guessed of " + actM.length + " carried; " + gradedM.length
         + " got an error (" + (gradedM.join(", ") || "none") + ")"]]) : [];
  const kv34 = lSnap && lSnap.losses
    ? snapRow.concat([["gradient size, before clipping", lSnap.losses.grad_norm === 0
        ? "reported main gradient norm: zero; this alone does not establish that weights stayed fixed"
        : clipShare(lSnap.losses.grad_norm) + " (over 100% = it was scaled down to the line)"]]) : [];
  const kv5 = lSnap && lSnap.losses && lSnap.losses.replay
    ? [["ring filled", per100(lSnap.losses.replay.fill_frac) + " slots"],
       ["re-study error", "a lesson value with no lived unit - " + NOT_COUNT + "; the run tab charts it against its own start"]]
    : [["re-study passes recorded in this run", frozen
        ? "0 - launched --eval-only, so the ring is never built"
        : "none in this snapshot - replay off, or the ring is still filling"]];
  station(cx, cy - ry, 250, "1 \u00b7 read outcome; predict it again",
          "previous state + executed command", "bp.forward", false, kv1);
  station(cx + rx, cy, 220, "2 \u00b7 compare prediction with senses",
          lSnap ? actM.length + " senses in, " + headM.length + " guessed, "
                  + gradedM.length + " graded"
                : "losses grade each sense that was guessed", "bp.grade", false, kv2);
  station(cx, cy + ry, 270, "3 \u00b7 the error flows BACKWARD",
          frozen ? "did NOT run: the weights are frozen" : "backpropagation, step by step",
          "bp.backward", frozen, kv34);
  // NOT "moves every weight": the checkpoint's Adam step counters differ from
  // parameter to parameter, so a weight moves only on the circles where it got one
  station(cx - rx, cy, 230, "4 \u00b7 AdamW moves the weights",
          frozen ? "did NOT run: the weights are frozen" : "only those that got a gradient",
          "bp.update", frozen, kv34);
  svg.appendChild(el("text", { x: cx, y: cy - 13, "text-anchor": "middle", class: "sublabel",
    text: "one circle per model step (" + momentMs + " ms of body time at --action-repeat "
        + reps + ")" + (frozen ? " IN A TRAINING RUN;" : ";") }));
  svg.appendChild(el("text", { x: cx, y: cy + 1, "text-anchor": "middle", class: "sublabel",
    text: nEnv > 1 ? "every circle carries all " + nEnv + " bodies: one weight update mixes them"
                   : "this run carries one body per circle" }));
  svg.appendChild(el("text", { x: cx, y: cy + 15, "text-anchor": "middle", class: "sublabel",
    text: frozen ? "here stations 3, 4 and 5 did not run and no weight changed"
                 : "the next circle starts with changed weights" }));
  // re-study: its own loss, its own backward, its own optimizer step. The rate, the
  // batch and its floor come from the run's flags: 48 is what the sampler ASKS for,
  // and replay.py throws the whole pass away if fewer than a quarter of them survive
  // the episode-boundary test.
  const g5 = el("g", { class: "mini-node clicky" });
  const r5 = el("rect", { x: 28, y: band + 248, width: 240, height: 46, rx: 6, class: "mini-box" });
  if (!replayOn) r5.setAttribute("style", "fill:#f2f2ef;stroke:#bbbbbb");
  g5.appendChild(r5);
  g5.appendChild(el("text", { x: 148, y: band + 262, "text-anchor": "middle",
    text: !replayOn ? "5 \u00b7 re-study disabled in this run"
                 : "5 \u00b7 re-study (1 model step in " + rEvery + ")" }));
  g5.appendChild(el("text", { x: 148, y: band + 275, "text-anchor": "middle", class: "sublabel",
    text: !replayOn ? (frozen ? "--eval-only: no replay updates" : "--replay off")
                 : rBatch + " windows asked for, " + rFloor + " the fewest that runs" }));
  g5.appendChild(el("text", { x: 148, y: band + 288, "text-anchor": "middle", class: "sublabel",
    text: !replayOn ? "no weight updates from replay"
                 : "the motor encoder trains here; the other 5 do not" }));
  g5.addEventListener("click", () => showDoc("bp.replay", "re-study", kv5.length ? kv5 : null));
  const d5 = `M120 ${band + 248} L120 ${cy + 24}`;
  const t5title = "5 \u2192 4: re-study's own optimizer step";
  svg.appendChild(el("path", { class: "mini-edge dash", "marker-end": "url(#loopar)", d: d5 }));
  const hit5 = el("path", { class: "dg-hit", d: d5 });
  hit5.addEventListener("click", ev => { ev.stopPropagation(); showDoc("edge.bp.replay.update", t5title); });
  svg.appendChild(hit5);
  // left of the dashed line, below station 4 and clear of the 3 -> 4 arc (it used to sit on the arc)
  const t5 = el("text", { x: 114, y: band + 196, "text-anchor": "end", class: "flowlabel clicky" });
  ["its own loss,", "its own optimizer step"].forEach((ln, i) =>
    t5.appendChild(el("tspan", { x: 114, dy: i ? 11 : 0, text: ln })));
  t5.addEventListener("click", ev => { ev.stopPropagation(); showDoc("edge.bp.replay.update", t5title); });
  svg.appendChild(t5);
  svg.appendChild(g5);
  // TWO NUMBERED SEQUENCES SHARE THIS TAB and the same numbered-circle idiom: these
  // five number the stages of learning, the picture below numbers the sequence of
  // operations inside one model step. "1" is the whole forward pass here and the first
  // sense box there. Nothing said so, and this one is read first.
  const foot = ["these 1-5 number the STAGES OF LEARNING; the numbered circles on the model diagram "
    + "below number the SEQUENCE OF OPERATIONS inside one model step - two different countings"];
  // "the next circle starts with changed weights" is a claim about direction. It is
  // measured here with the page's own trend rule, on the page's own charted series,
  // instead of asserted - and the size word it replaced ("slightly better") carried
  // no number at all.
  // A page whose charts are its OWN --eval-only probe measures a trend on FROZEN
  // weights. No circle moved one, so that trend cannot be a circle's improvement -
  // and this very page says so four lines up ("no gradient was computed, no weight
  // moved"). Scoring it anyway put both claims on one picture. Report the trend and
  // say plainly what it is not, rather than attributing it to learning that did not
  // happen. When the charts come from a DIFFERENT run (stats_source), that run did
  // train and the measured verdict stands.
  const probeCharts = frozen && (!META.stats_source || META.stats_source === META.run);
  const czw = winStats(seriesByKey("causal_pe"));
  foot.push((probeCharts && czw)
      ? "is a circle an improvement? not answerable from this page: no weight moved in this run, so its one-step prediction error trend is not a circle's doing"
      : czw ? "is a circle an improvement? measured, not assumed, on the run these charts come from:"
            : "is a circle an improvement? this page carries no one-step prediction error series to measure it on, so the direction is unmeasured here");
  if (czw)
    foot.push((META.stats_source || META.run) + "'s one-step prediction error runs " + pctNum(100 * czw.a)
          + " of the all-zeros answer's error in its first quarter-window and " + pctNum(100 * czw.b)
          + " in its last" + (probeCharts
              ? " - that is the frozen model meeting a changing world, not a weight update"
              : " - the page's own trend rule calls that " + verdict(czw.a, czw.b, "error").txt));
  // each sentence wrapped to the drawing's width from its measured text (one of them, carrying a
  // run name, ran past the right edge in the layout check of 2026-10-02); the drawing grows to fit
  let fy = band + 312;
  for (const s of foot) {
    const ls = wrapText(s, 9, (W - 32) * 0.96);
    const t = el("text", { x: 16, y: fy, class: "sublabel" });
    ls.forEach((ln, i) => t.appendChild(el("tspan", { x: 16, dy: i ? 12 : 0, text: ln })));
    svg.appendChild(t);
    fy += 12 * ls.length + 1;
  }
  svg.setAttribute("viewBox", `0 0 ${W} ${Math.max(H, fy)}`);
  root.appendChild(svg);
}

// ---------------------------------------------- per-subsystem detail diagrams
// Clicking a bottom-row box also renders its own diagram BELOW the main one,
// with the data named on every arrow. Built from the same verified facts as
// the popups; every box and label opens the matching popup.
const DETAIL = {};
function detailFlow(id, W, H, boxes, arrows, foot) {
  DETAIL[id] = () => flowDiagram(W, H, boxes, arrows, foot, false);
}
DETAIL.intention = () => intentionMini(false);   // reuses the planner flow, full size

// The curiosity curve is drawn FROM THE FUNCTION the code runs (lem/model/curiosity.py)
// over the band THIS run was launched with, so the picture cannot drift from the code:
// zero at each band edge, straight up to 1.0 at the midpoint, straight back down.
const CURB = (() => {
  const p = String((META.args || {}).curiosity_band || "").split(":").map(Number);
  return (p.length === 2 && p.every(v => isFinite(v) && v > 0) && p[1] > p[0])
    ? { lo: p[0], hi: p[1], src: "this run's --curiosity-band " + p[0] + ":" + p[1] }
    : { lo: 0.1, hi: 0.4, src: "the built-in band of 10% to 40% of the reference error - this page's run recorded no --curiosity-band" };
})();
const CURMID = (CURB.lo + CURB.hi) / 2, CURAX = CURB.hi * 1.25;
// the error is causal_pe, written as a percentage of the reference error (DESIGN.md section 11)
const CURN = v => pctNum(100 * v);
const CURX = pe => Math.round(262 + 276 * pe / CURAX);   // error -> x (error 0 at x 262)
const CURY = d => Math.round(112 - 90 * d);              // drive -> y (0 at 112, 1.0 at 22)
const CURTICK = pe => ({ pts: [[CURX(pe), 112], [CURX(pe), 118]], plain: true,
                         label: CURN(pe), lx: CURX(pe), ly: 126 });
const CURUSE = (() => {
  const a = META.args || {};
  return (a.curiosity_goal && a.wellness_mode)
    ? " (here: curiosity-goal " + a.curiosity_goal + ", wellness-mode " + a.wellness_mode + ")" : "";
})();

detailFlow("curiosity", 920, 225, [
  { x: 10, y: 40, w: 175, h: 40, l1: "next-model-step error", l2: "causal_pe: mean recorded sense error", key: "term.causal_pe" },
  { x: 250, y: 12, w: 300, h: 118, l1: "", l2: "", key: "node.curiosity" },
  { x: 610, y: 40, w: 115, h: 40, l1: "drive", l2: "0 to 1", key: "node.curiosity" },
  // what the drive does to a goal-ring entry under this run's flags (schema.py edge.curiosity.intention)
  { x: 610, y: 130, w: 240, h: 44, l1: "goal-ring entry", key: "edge.curiosity.intention",
    l2: (ARGS.curiosity_goal || "off") !== "on" ? "not read: --curiosity-goal " + (ARGS.curiosity_goal || "off")
        : ARGS.wellness_rank === "normalized" ? "separately scaled and weighted channel"
        : ARGS.curiosity_mode === "additive" ? "stored value + drive × its weight" : "stored value × (1 + drive)" },
], [
  { pts: [[185, 60], [250, 60]], label: "error in", lx: 217, ly: 52, key: "term.causal_pe" },
  { pts: [[550, 60], [610, 60]], label: "drive out", lx: 580, ly: 52, key: "node.curiosity" },
  { pts: [[667, 80], [667, 130]], anchor: "start", lx: 674, ly: 108, key: "edge.curiosity.intention",
    label: "[" + ((META.args || {}).n_envs || 2) + " bodies] float32, every model step" },
  // the curve, computed from the band above - never drawn by eye
  { pts: [[262, CURY(0)], [CURX(CURB.lo), CURY(0)], [CURX(CURMID), CURY(1)],
          [CURX(CURB.hi), CURY(0)], [538, CURY(0)]], plain: true },
  CURTICK(CURB.lo), CURTICK(CURMID), CURTICK(CURB.hi),
], [
  "the curve is computed from the code's own formula over " + CURB.src + ", never drawn by eye",
  "no drive at an error of " + CURN(CURB.lo) + " of the reference error · straight up to the full drive at " + CURN(CURMID)
    + " · straight back down to none at " + CURN(CURB.hi) + " · none outside the configured band",
  "axes: error runs from 0% at the left edge to " + CURN(CURAX) + " of the reference error at the right; drive runs from none at the baseline to the full drive at the peak",
  "causal_pe averages the recorded delayed-prediction errors for non-synthetic senses; the fixed prediction-accuracy wellness score has its own exclusions",
  "audio carried 74% to 90% of that summed error in H81, H82 and H83 (measured 2026-09-14), so the drive mostly follows how surprising AUDIO was",
  "goal choice reads the drive when --curiosity-goal is on; only --wellness-mode direct or forward also scales the wellness lesson by it" + CURUSE,
]);

// The imagination chain reads its own settings off the run instead of being drawn
// in: the depth from the launch flags (clamped the way the model clamps it), the
// weight measured from the run's own loss records - the launch may leave
// --imagination-weight unset, and the page never reads the config file the default
// lives in. imagDepthExported() says which of the two happened.
function imagOn() { return ((META.args || {}).imagination || "config") !== "off"; }
function imagDepthExported() { return typeof (META.args || {}).imagination_steps === "number"; }
function imagDepth() {
  const a = META.args || {};
  let d = (imagDepthExported() && a.imagination_steps > 0) ? a.imagination_steps : 3;
  const ctx = a.context_length;          // lem_model clamps the chain to ctx-1 when
  if (typeof ctx === "number" && ctx > 0 && ctx < d + 1) d = Math.max(1, ctx - 1);
  return d;                              // the context window cannot hold depth+1
}
function imagWeight() {                  // measured: weighted term / raw summed error
  for (let i = SNAPS.length - 1; i >= 0; i--) {
    const L = SNAPS[i].losses || {}, w = (L.breakdown || {}).imagination;
    if (L.imagination_loss && w != null) return w / L.imagination_loss;
  }
  const a = (META.args || {}).imagination_weight;
  return (typeof a === "number") ? a : null;
}

(function () {
  const D = imagDepth(), x0 = 250, span = Math.min(200, Math.floor(640 / D)), bw = span - 50;
  const gx = i => x0 + i * span, gw = gx(D - 1) + bw - x0;
  const R = META.receipt || {}, A = META.args || {};
  const carried = (R.active_modalities || []).filter(m => m !== "motor").length || 5;
  const nB = A.n_envs || 1;
  const dOut = D_MODEL;
  const sN = SNAPS.length ? SNAPS[SNAPS.length - 1] : null;
  const graded = sN ? (R.prediction_head_modalities || []).filter(
      m => (sN.header.source_mask || {})[m] === "real").length : carried;
  const w = imagWeight();
  const boxes = [
    // the chain is FED the tokens the core consumed, not the raw ones: the box used
    // to say only "sense tokens", and the grading box below says "raw tokens", so the
    // two ends of the same picture read as one type of token when they are two
    { x: 10, y: 30, w: 190, h: 40, l1: "the model step " + D + " steps back",
      l2: carried + " tokens as the core read them", key: "edge.ln_out.imagination" },
  ];
  for (let i = 0; i < D; i++)
    boxes.push({ x: gx(i), y: 30, w: bw, h: 40, l1: "guess +" + (i + 1),
                 l2: i ? "measured senses fed back" : "predicted tokens",
                 key: "node.imagination" });
  boxes.push({ x: x0, y: 130, w: gw, h: 40, l1: "graded against what really arrived",
               l2: "each depth, raw tokens · " + (D * graded) + " errors summed, then "
                   + (w === null ? "weighted once" : "weighted at " + pctNum(100 * w) + " of full weight"),
               key: "bp.grade" });
  const w1 = Math.round(gw * 0.45), w2 = gw - w1 - 40;
  boxes.push({ x: x0, y: 200, w: w1, h: 40, l1: "joins the step's loss",
               l2: "the one number AdamW descends", key: "bp.update" });
  // the win rate sets the planner's depth only under --hand-target-mode world (lem_model.py _plan_chain sets max_steps
  // = 1 otherwise): the code check of Oct 15, 2026
  const depthOne = (A.hand_target_mode || "world") !== "world";
  boxes.push({ x: x0 + w1 + 40, y: 200, w: w2, h: 40, l1: "did it beat doing nothing?",
               l2: depthOne ? "computed; no plan depth this run" : "sets how far ahead the planner may aim",
               key: "edge.imagination.intention" });
  const arrows = [
    { pts: [[200, 50], [x0, 50]], label: "predict", lx: (200 + x0) / 2, ly: 42,
      key: "edge.motor.imagination" },
  ];
  for (let i = 1; i < D; i++)
    arrows.push({ pts: [[gx(i - 1) + bw, 50], [gx(i), 50]], label: "feed back",
                  lx: gx(i) - 25, ly: 42, key: "node.imagination" });
  for (let i = 0; i < D; i++)
    arrows.push({ pts: [[gx(i) + bw / 2, 70], [gx(i) + bw / 2, 130]], dash: true, key: "bp.grade" });
  arrows.push({ pts: [[x0 + w1 / 2, 170], [x0 + w1 / 2, 200]], dash: true, key: "bp.update" });
  arrows.push({ pts: [[x0 + w1 + 40 + w2 / 2, 170], [x0 + w1 + 40 + w2 / 2, 200]], dash: true,
                key: "edge.imagination.intention" });
  const foot = [
    "no command leaves this chain - it puts no force on the body itself",
    depthOne
      ? "it decides nothing about behaviour in this run either: under --hand-target-mode " + A.hand_target_mode
        + " the planner's search looks one model step ahead, so the depth this chain earns is not used"
      : "what it does decide: how often each depth beats assuming nothing changed sets how many model steps ahead the planner may aim",
    "only senses measured that model step are graded and fed back; one that was not measured is put back as it was observed, so its noise cannot compound",
    "each hop is given the command the body really executed then; the body pose inside that command token stays the real one from the chain's start",
    "it is FED the tokens the core consumed (up to half the model's own guess of now blended in) and GRADED on the raw arriving tokens - two different token sets, one at each end",
  ];
  // what that blend actually measured on this page's own snapshots, so "gate-blended"
  // is never a word without a size next to it
  const gband = {};
  for (const s of SNAPS) {
    const g = (s.drives || {}).td_gate;
    if (!g) continue;
    for (const k in g) {
      if (!gband[k]) gband[k] = [g[k], g[k]];
      else { gband[k][0] = Math.min(gband[k][0], g[k]); gband[k][1] = Math.max(gband[k][1], g[k]); }
    }
  }
  const gKeys = Object.keys(gband);
  if (gKeys.length) {
    const shut = gKeys.filter(k => gband[k][1] === 0), open = gKeys.filter(k => gband[k][1] > 0);
    foot.push("how big that blend was here, in parts of the token out of 100 (0 = the raw token, 50 = the cap): "
      + (shut.length ? shut.join(", ") + " stayed at 0" : "")
      + (shut.length && open.length ? "; " : "")
      + open.map(k => k + " " + Math.round(100 * gband[k][0]) + " to " + Math.round(100 * gband[k][1]) + " in 100").join(", "));
  }
  if (!imagDepthExported())
    foot.push("the launch did not record --imagination-steps, so the depth drawn is the config file's default - the page reads launch flags, not the config file");
  if (w === null)
    foot.push("no snapshot carries both the summed error and its weighted share, so this run's imagination weight cannot be read off the page");
  detailFlow("imagination", 900, 246 + 13 * foot.length, boxes, arrows, foot);
})();

// The replay memory, drawn. Every number is the RUN's: the ring size, the rate, the
// batch and the window length come from META.args where the build exports them, and
// the fallbacks are the launcher's own defaults. What the picture used to state as
// facts about this run - a rate, a guaranteed batch of 48, "encoders are excluded" -
// was wrong in three ways: this run never ran one pass, 48 is what the sampler asks
// for, and the motor encoder does train here.
(function () {
  const rsz = ARGS.replay_size || 6000, rbt = ARGS.replay_batch || 48;
  const rev = ARGS.replay_every || 4, rct = Number(SHAPE.context_length);
  const rbod = ARGS.n_envs || 1, rsec = MOMENT_S;
  const rfloor = Math.max(2, Math.floor(rbt / 4));      // replay.py drops the pass below this
  const rneed = Math.max(4 * rbt, 8 * rct);             // train_mimo.py's --fill-ring-first gate
  const replayEnabled = ARGS.eval_only === false && ARGS.replay === "on";
  const foot = [
    replayEnabled ? "replay enabled: a pass is scheduled every " + rev + " live model steps, after " + rneed.toLocaleString()
          + " stored model steps carry the command that left them (" + Math.round(rneed * rsec)
          + " s of body time) when --fill-ring-first is set"
        : ARGS.eval_only === true ? "evaluation only: no replay sampler or replay weight updates"
        : ARGS.replay === "off" ? "replay is switched off: no replay sampler or replay weight updates"
        : "the saved settings do not establish whether replay training was enabled",
    rbt + " windows is what the sampler ASKS for - it rejects any window crossing an "
      + "episode reset and throws the whole pass away if fewer than " + rfloor + " survive",
    "each window is " + (rct + 1) + " consecutive model steps from ONE body and ONE episode, so no lesson pairs one body's cause with another's effect",
    "that filling gate is checked only on the model steps a pass would run, it stays passed for the rest of the process once it passes, and a model step counts as filled if ANY body carried an outgoing command",
    "the motor encoder DOES learn here"
      + (((META.component_params || {}).enc_motor)
         ? " (" + META.component_params.enc_motor.toLocaleString() + " weights)" : "")
      + ": the step-from motor token is re-encoded live from [achieved pose, executed command]",
    "the other five encoders do not - their tokens were stored detached; the persistent memory's bias projection is trained here too",
    "the motor lesson teaches 7 of the 9 command columns: radius gets exactly zero gradient and strength is only pushed down toward the phase's strength limit",
  ];
  detailFlow("replay", 900, 120 + 13 * foot.length, [
    { x: 10, y: 30, w: 250, h: 44, l1: "the ring",
      l2: rsz.toLocaleString() + " model steps × " + rbod + (rbod === 1 ? " body" : " bodies"),
      key: "node.replay" },
    { x: 360, y: 30, w: 210, h: 44, l1: "its OWN AdamW step",
      l2: "up to " + rbt + " windows, at least " + rfloor, key: "edge.bp.replay.update" },
    { x: 650, y: 30, w: 240, h: 44, l1: "forward model · heads · motor",
      l2: "· memory bias · the motor encoder", key: "node.replay" },
  ], [
    // the label is wider than the 100-unit gap between the two boxes, so it sits above both
    { pts: [[260, 52], [360, 52]], label: "each window: " + (rct + 1) + " model steps + the command executed at each",
      // the glossary key is the code's own name, so it is written in backticks (the style guide "ONE
      // WORD PER THING": a code name that differs is given once in backticks)
      lx: 310, ly: 21, key: "term.executed-command" },
    { pts: [[570, 52], [650, 52]], label: "gradients", lx: 610, ly: 44, key: "bp.backward" },
  ], foot);
})();

// The phase boxes are built from META.phases, which build_explainer.py reads out of
// lem/training/curriculum.py and the trainer's age and delay tables. Writing the list
// here a second time is what produced three boxes the code has never had - "infant",
// "sitter" and "crawler" against the real neonate / early_infant / late_infant /
// toddler / child / mature.
const CURR_PH = META.phases || [];
const CURR_SPP = (META.args || {}).steps_per_phase;
const CURR_HERE = (SNAPS.length ? SNAPS[SNAPS.length - 1].header.phase : "")
                  || (META.args || {}).start_phase || "";
const CURR_TSEC = MOMENT_S;
const CURR_W = CURR_PH.length ? Math.floor((880 - 20 * (CURR_PH.length - 1)) / CURR_PH.length) : 880;
const CURR_NAME = x => String(x || "").replace(/_/g, " ");
const CURR_BOXES = CURR_PH.length ? CURR_PH.map((p, i) => ({
  x: 10 + i * (CURR_W + 20), y: 34, w: CURR_W, h: 44,
  l1: CURR_NAME(p.name) + (p.name === CURR_HERE ? " (now)" : ""),
  l2: "reach " + Math.round(40 * p.rom_limit) + " cm · "
      + Math.round(100 * p.strength_limit) + "% push",
  key: "node.curriculum" })) : [
  { x: 10, y: 34, w: 880, h: 44, l1: "this page carries no phase table",
    l2: "rebuild it with the current build_explainer.py", key: "node.curriculum" }];
const CURR_ARROWS = CURR_PH.slice(1).map((p, i) => ({
  pts: [[10 + i * (CURR_W + 20) + CURR_W, 56], [10 + (i + 1) * (CURR_W + 20), 56]],
  key: "node.curriculum" }));
if (CURR_ARROWS.length) {        // one label for all of them: the gaps are 20 px wide
  const mid = CURR_ARROWS[CURR_ARROWS.length >> 1];
  mid.label = "each change: the world workers rebuild the body at the new age and motor delay; the model drops what spanned it";
  mid.lx = 450; mid.ly = 22;
}
const CURR_FOOT = [];
if (CURR_PH.length) {
  const a = CURR_PH[0], z = CURR_PH[CURR_PH.length - 1];
  CURR_FOOT.push("left to right, strength and reach rise together: " + CURR_NAME(a.name)
    + " may push at " + Math.round(100 * a.strength_limit)
    + "% of full force and move a handle " + Math.round(40 * a.rom_limit)
    + " cm per command, " + CURR_NAME(z.name) + " at " + Math.round(100 * z.strength_limit)
    + "% and " + Math.round(40 * z.rom_limit) + " cm");
  // Only the MOTOR wait is real. A sensory delay reached nothing in this bridge -
  // MIMo applies it in _get_obs and the env reads live MuJoCo sensors - so the column
  // is 0 at every phase and BridgeEnv refuses anything else. Printing a "sense arrives
  // 40 ms late" line for a number that never touched a sense is the page lying.
  if (a.motor_delay_ms !== null && a.motor_delay_ms !== undefined)
    CURR_FOOT.push("the wait shrinks: a command acts " + a.motor_delay_ms
      + " ms late at " + CURR_NAME(a.name) + " and " + z.motor_delay_ms + " ms at "
      + CURR_NAME(z.name) + " (one model step is " + Math.round(CURR_TSEC * 1000)
      + " ms of body time); senses are never delayed at any phase");
  if (a.age_months !== null && a.age_months !== undefined)
    CURR_FOOT.push("the body itself is regrown at each phase's age, " + a.age_months + " to "
      + z.age_months + " months"
      + (a.scene ? "; the scene is the " + a.scene + " at every phase" : ""));
  CURR_FOOT.push(CURR_PH.every(p => p.exploration_noise === a.exploration_noise)
    ? "babble noise does NOT follow the phases: the table falls from a random offset of "
      + Math.round(100 * Number(a.table_exploration_noise)) + " cm to "
      + Math.round(100 * Number(z.table_exploration_noise)) + " cm and a floor lifts every phase to a random offset of "
      + Math.round(100 * Number(a.exploration_noise)) + " cm - the same in all " + CURR_PH.length
    : "babble noise per phase, as the random offset it adds to a command's position: " + CURR_PH.map(p => Math.round(100 * p.exploration_noise) + " cm").join(", "));
  CURR_FOOT.push("what that floor means: it adds a random offset of " + Math.round(100 * a.exploration_noise) + " cm of position and "
    + Math.round(10 * a.exploration_noise * 180 / Math.PI)
    + " degrees of rotation to a command, one standard deviation");
  CURR_FOOT.push((CURR_SPP ? CURR_SPP.toLocaleString() + " model steps per phase in this run = "
      + Math.round(CURR_SPP * CURR_TSEC) + " s of body time per phase"
      : "this page's args.json does not record --steps-per-phase, so the pace is not shown")
    + (CURR_HERE ? "; this page's model is at " + CURR_NAME(CURR_HERE) : ""));
  if (CURR_SPP && META.trained_step)
    CURR_FOOT.push("a lineage that ran this schedule from " + CURR_NAME(a.name) + " reaches "
      + CURR_NAME(z.name) + " at step " + (CURR_SPP * (CURR_PH.length - 1)).toLocaleString()
      + " - " + (100 * CURR_SPP * (CURR_PH.length - 1) / META.trained_step).toFixed(2)
      + "% of this model's " + META.trained_step.toLocaleString()
      + " trained steps; every step after that is spent in the last box");
}
CURR_FOOT.push("a phase change that changes the body - its age, the motor delay, the scene - rebuilds the worlds and drops the context window, the pending lessons and the goal ring");
CURR_FOOT.push("there is no sensory delay at any phase: MIMo applies one inside _get_obs and this bridge reads live MuJoCo sensors, so it reached nothing - measured, 4 steps against 0 left tactile, proprioception and vestibular identical on 12 of 12 steps at a maximum difference of exactly zero - and the env now refuses a nonzero setting rather than keep a knob that does nothing");
CURR_FOOT.push("so no single lesson pairs one body's cause with another body's effect; model steps already in the replay ring are NOT dropped and keep being re-studied after the rebuild");
detailFlow("curriculum", 900, 215, CURR_BOXES, CURR_ARROWS, CURR_FOOT);

// The ring's reach in seconds is what this component exists for, so both windows are
// computed from the run's own numbers: the ring's size and sampling period ride on the
// newest snapshot (lem_model.py last_memory), the core's window comes from the
// receipt's _seq_length and --action-repeat. (P181)
const MEM_SNAP = SNAPS.filter(s => isObj((s || {}).memory)).slice(-1)[0] || null;
const MEM_LAST = MEM_SNAP ? MEM_SNAP.memory : {};
// THE RECALLED BIAS AGAINST THE TOKEN IT LANDS ON. "a third to a half the size of that
// token" was drawn on every run's page and was H81's measurement (0.19 to 0.28 against
// 0.59); H84's snapshots read 3.0 to 4.0 for the bias. The label now carries what THIS
// page's newest snapshot measured. A snapshot taken by the capture of 2026-09-19 or later
// records the token's own size too (landing_token_rms) and the ratio is computed from it;
// an older one records the bias alone, and the label says the comparison cannot be made.
const MEM_STEP = MEM_SNAP ? Number(MEM_SNAP.header.step).toLocaleString() : "";
const MEM_BIAS = typeof MEM_LAST.bias_rms_body === "number" ? MEM_LAST.bias_rms_body
               : (typeof MEM_LAST.bias_rms === "number" ? MEM_LAST.bias_rms : null);
const MEM_TOKEN = typeof MEM_LAST.landing_token_rms === "number" && MEM_LAST.landing_token_rms > 0
                ? MEM_LAST.landing_token_rms : null;
const MEM_BIAS_L2 = "[" + SHAPE.n_senses + " × " + D_MODEL + "]"
  + (MEM_BIAS === null ? ""
     : MEM_TOKEN !== null
       ? ", " + pctNum(100 * MEM_BIAS / MEM_TOKEN) + " of the size of the token it lands on (step " + MEM_STEP + ")"
       : ", " + tokSize(MEM_BIAS) + " at step " + MEM_STEP + "; that token's size is not in this snapshot");
const MEM_BIAS_NOTE = MEM_BIAS === null
  ? "the recalled bias's size is not in this page's snapshots; on H81's probe it measured a third to a half of the size of the token it lands on"
  : MEM_TOKEN !== null
    ? "at step " + MEM_STEP + " the recalled bias measures " + tokSize(MEM_BIAS) + " and the token it lands on " + tokSize(MEM_TOKEN) + " (RMS, 0% = all zeros) - the bias is " + pctNum(100 * MEM_BIAS / MEM_TOKEN) + " of that token's size"
    : "at step " + MEM_STEP + " the recalled bias measures " + tokSize(MEM_BIAS) + " (RMS, 0% = adds nothing); this snapshot does not record the size of the token it lands on";
const MEM_BIAS_NOTE2 = MEM_TOKEN !== null ? null
  : "so no measured share of that token can be stated here. On H81's probe the bias measured a third to a half of that token's size - that run's figure, not this one's";
// WITHOUT THE MEASUREMENT, AN ESTIMATE - and the word estimate stays on it. The token is the
// final LayerNorm's output; build_explainer.landing_token_estimate reads that layer's gain
// and bias from the run's newest checkpoint and gives sqrt(mean gain^2 + mean bias^2), which
// assumes the normalised token is unrelated to them from number to number. On H81, where
// the measurement exists, it gave 0.589 against 0.59 measured.
const MEM_EST = META.memory_token_estimate;
const MEM_BIAS_NOTE3 = (MEM_TOKEN !== null || MEM_BIAS === null || !MEM_EST || !(MEM_EST.estimate > 0)) ? null
  : "an ESTIMATE from the final LayerNorm's gain and bias in the checkpoint at step " + Number(MEM_EST.step).toLocaleString()
    + " puts that token at " + tokSize(MEM_EST.estimate) + ", which would make the bias " + pctNum(100 * MEM_BIAS / MEM_EST.estimate)
    + " of its size (the same estimate came within 0.2% of the size measured on H81)";
const MEM_CAP = MEM_LAST.capacity || 128;
const MEM_WIN = MEM_LAST.window_seconds || MEM_CAP * (MEM_LAST.interval_seconds || 0.5);
const MEM_FMR = (META.receipt || {}).forward_model || {};
const MEM_MOM = (MEM_FMR._seq_length || 0) / (MEM_FMR._n_modalities || 1);
const CORE_WIN = MEM_MOM * MOMENT_S;
detailFlow("memory", 900, 300, [
  { x: 10, y: 14, w: 210, h: 44, l1: "one model step of the body", l2: "[" + SHAPE.memory_event + "] numbers, detached", key: "edge.ln_out.memory" },
  { x: 10, y: 96, w: 210, h: 44, l1: "striking events", l2: "the biggest change each 0.5 s", key: "node.memory" },
  { x: 270, y: 96, w: 175, h: 44, l1: "event ring", l2: `up to ${MEM_CAP} kept = ${MEM_WIN.toFixed(0)} s of body time`, key: "node.memory" },
  { x: 495, y: 96, w: 175, h: 44, l1: "GRU summary", l2: `all ${MEM_CAP + 2} positions re-read in order`, key: "term.gru" },
  { x: 715, y: 96, w: 175, h: 44, l1: "lookup", l2: "the present queries it", key: "node.memory" },
  { x: 10, y: 196, w: 400, h: 44, l1: "needs forecast: the body's next 5 needs", l2: "5 numbers 0 to 1 · graded at a tenth of full lesson weight", key: "edge.memory.needs" },
  { x: 460, y: 196, w: 430, h: 44, l1: "bias on the core's newest-model-step OUTPUT", l2: MEM_BIAS_L2, key: "edge.memory.final_norm" },
], [
  { pts: [[115, 58], [115, 96]], label: "every " + MOMENT_S.toFixed(2) + " s", lx: 170, ly: 80, key: "edge.ln_out.memory" },
  { pts: [[220, 118], [270, 118]], key: "node.memory" },
  { pts: [[445, 118], [495, 118]], label: "each event + 2 time numbers", lx: 470, ly: 90, key: "edge.memory.ring_gru" },
  { pts: [[670, 118], [715, 118]], key: "node.memory" },
  { pts: [[802, 140], [802, 196]], label: "final summary (96) + weighted recall (96)", lx: 795, ly: 160, anchor: "end", key: "edge.memory.final_norm" },
  { pts: [[560, 140], [560, 172], [210, 172], [210, 196]], label: "that same state + the command just executed", lx: 215, ly: 166, anchor: "start", key: "edge.memory.needs" },
], ["one event = " + SHAPE.token_row + " raw sense-token numbers (what the senses reported, before the top-down gate) + 5 bodily needs + 5 validity flags + " + SHAPE.order_numbers + " command numbers + " + D_MODEL + " goal numbers",
    MEM_BIAS_NOTE,
    MEM_BIAS_NOTE2,
    MEM_BIAS_NOTE3,
    "the first entry after any episode reset is simply the first frame - there is no earlier frame to compare it with",
    "events pushed out of the ring survive only as a carried summary, and that carry is detached: no lesson ever reaches an evicted event",
    "the whole window is re-read from its first position on every call, and a run with the delayed next-frame lesson on makes that call twice per model step - once on the previous model step's stored window, once on this model step's",
    (() => {   // P414: the denominators come from the page's own data, never a written-in run
      const thr = (SERIES.charts || []).flatMap(c => c.series || []).find(s => s.key === "model_steps_per_s");
      const tot = SNAPS.map(s => ((s.losses || {}).timings_ms || {}).total).filter(x => typeof x === "number");
      const steps = SNAPS.map(s => (s.header || {}).step);
      const gap = steps.length > 1 ? steps[1] - steps[0] : null;
      const out = ["re-reading the whole window measured 1.0 to 1.3 ms on this machine's GPU (explainer probe, batch 2, 130 positions)"];
      if (thr && thr.points && thr.points.length) {
        const ms = thr.points.map(pt => 1000 / pt[1]).filter(x => isFinite(x) && x > 0);
        if (ms.length) out.push("against " + ((META.stats_source && META.stats_source !== META.run) ? META.stats_source + "'s" : "this run's")
          + " logged model step of " + Math.round(Math.min.apply(null, ms)) + " to " + Math.round(Math.max.apply(null, ms))
          + " ms, which counts every step (model compute plus the world exchange)");
        else out.push("this page carries no throughput record, so its share of a model step is not shown here");
      } else {
        out.push("this page carries no throughput record, so its share of a model step is not shown here");
      }
      if (tot.length) {
        out.push("the " + tot.length + " snapshot steps you can page through are not typical steps"
          + (gap ? " - one every " + gap + " steps, so each also carries a re-study batch and a planning pass" : "")
          + " - their own in-step total ran " + Math.round(Math.min.apply(null, tot)) + " to " + Math.round(Math.max.apply(null, tot)) + " ms");
      }
      return out.join("; ");
    })(),
    `with the ring full the recurrent network reads ${MEM_CAP + 2} positions - the ${MEM_CAP} kept events, the biggest change so far inside the open half second, and the present model step; ${MEM_CAP + 1} on the step right after a half second closes`,
    CORE_WIN > 0 ? `the transformer itself reads ${MEM_MOM} model steps = ${CORE_WIN.toFixed(1)} s of body time, so this ring reaches ${(MEM_WIN / CORE_WIN).toFixed(0)}x further back - that reach is the whole reason this component exists` : "this page's receipt records no sequence length for the core, so the ring's reach against it is not shown"].filter(x => x));

detailFlow("self_state", 900, 350, [
  { x: 10, y: 34, w: 230, h: 44, l1: "skin readings", l2: "pressure, sensing quality and load", key: "edge.world.self_state" },
  { x: 10, y: 102, w: 230, h: 44, l1: "motor capacity readings", l2: "fraction of motor capacity available", key: "edge.world.self_state" },
  { x: 300, y: 68, w: 180, h: 44, l1: "learned body estimate", l2: "remembers recent body readings", key: "term.gru" },
  { x: 300, y: 136, w: 180, h: 44, l1: "movement history", l2: "commands in the order they happened", key: "edge.motor.self_state" },
  { x: 530, y: 68, w: 360, h: 44, l1: "predict body changes from past commands", l2: "check against the later body measurement", key: "node.self_state" },
  { x: 530, y: 196, w: 360, h: 44, l1: "compare prediction with measurement", l2: "how clearly skin senses; available motor capacity", key: "bp.grade.self_state" },
  { x: 300, y: 196, w: 200, h: 44, l1: "teaching score", l2: "updates self networks in training", key: "edge.self_state.loss" },
], [
  { pts: [[240, 56], [270, 56], [270, 84], [300, 84]], key: "edge.world.self_state" },
  { pts: [[240, 124], [270, 124], [270, 96], [300, 96]], key: "edge.world.self_state" },
  { pts: [[60, 10], [390, 10], [390, 68]], dash: true, label: "sensory information and earlier core output", lx: 210, ly: 24, key: "edge.ln_out.self_state" },
  { pts: [[480, 90], [530, 90]], label: "estimate", lx: 505, ly: 82, key: "node.self_state" },
  { pts: [[480, 158], [508, 158], [508, 102], [530, 102]], label: "executed commands, in order", lx: 512, ly: 132, anchor: "start", key: "edge.motor.self_state" },
  { pts: [[710, 112], [710, 196]], dash: true, key: "bp.grade.self_state" },
  { pts: [[530, 218], [500, 218]], key: "edge.self_state.loss" },
  ...(!ARGS.eval_only ? [
  { pts: [[300, 218], [282, 218], [282, 104], [300, 104]], dash: true, key: "edge.self_learning.self_state" },
  { pts: [[400, 240], [400, 256], [898, 256], [898, 90], [890, 90]], dash: true, key: "edge.self_learning.self_state" },
  ] : []),
], [
  (((META.receipt || {}).self_state || ARGS.self_state) === "perceive"
    ? "Perceive mode also passes the body estimate to the forward model."
    : "Learn mode does not pass the body estimate into the forward model."),
  "Self-choice: " + (ARGS.self_choice || "off") + ". Off builds no chooser; shadow discards its commands; on uses a qualified frozen actor. The overview explains the hand-goal dependency.",
  ((META.args || {}).eval_only
    ? "this page's run was launched --eval-only: every lesson here was computed and reported and no weight moved"
    : "these lessons update this module's own weights every model step while the run is training"),
  "the inspector's Self perception panel prints, beside each horizon, what 'nothing changed at all' scores on the same window - the floor this forecast has to beat",
]);

// the data on the voice head's two lines, read from the run's own receipt and args
// instead of written into the picture: bodies, senses and token width from the
// receipt, the model step's length from --action-repeat, the live band from the
// snapshots this page embeds.
const VOICE = (() => {
  const r = META.receipt || {}, a = META.args || {};
  const n = (r.forward_model || {})._n_modalities || 6;
  const b = a.n_envs || 1;
  const d = D_MODEL;
  const vs = SNAPS.map(s => (s || {}).voice).filter(isObj);
  const band = vs.length ? pctNum(100 * Math.min(...vs.map(v => v.min))) + " to "
                         + pctNum(100 * Math.max(...vs.map(v => v.max))) + " of the range" : "";
  return { inLabel: "[" + b + " bodies × " + n + " senses × " + d + "], one model step ("
                    + Math.round(1000 * MOMENT_S) + " ms)",
           outLabel: "[" + b + " bodies × " + ((r.voice_head || {}).output_dim || 80)
                     + "] - read only by this page",
           band: "bound 0 to 1" + (band ? "; live " + band : "") };
})();
detailFlow("voice", 900, 176, [
  { x: 10, y: 30, w: 220, h: 44, l1: "newest frame's tokens", l2: "the final LayerNorm's 6 rows", key: "edge.final_norm.voice" },
  { x: 310, y: 30, w: 200, h: 44, l1: "one query reads all 6", l2: "4 heads → MLP " + D_MODEL + " → 80 → sigmoid", key: "node.voice" },
  { x: 590, y: 30, w: 300, h: 44, l1: "80 numbers out, 0 to 1", l2: VOICE.band, key: "node.voice" },
], [
  { pts: [[230, 52], [310, 52]], label: VOICE.inLabel, lx: 270, ly: 96, key: "edge.final_norm.voice" },
  { pts: [[510, 52], [590, 52]], label: VOICE.outLabel, lx: 550, ly: 96, key: "node.voice" },
], ["80 is the width of the action slot the shared-memory buffer reserves (float32[80] at offset 856), not a vocal tract's control count: the repo's only vocal-tract parameter set is 56 numbers - 11 named parameters, a wrapped synth clock and 44 tract diameters - so 24 of the 80 map onto nothing named in the tree",
    "no lesson reaches this head: no target is given and its loss weight is zero, so its numbers still hold the values they were first given",
    "the LayerNorm here acts on the QUERY, not on the 6 sense rows - the query is one learned constant, the same " + D_MODEL + " numbers for every body and every step",
    "no sound in this world yet: nothing turns the controls into audio until BioFold (the physical vocal tract) is done",
    "so the model cannot hear itself; the caregiver has no ears either - it reads camera pose, toy positions and contacts, never an ear signal - so a working voice would still not be answered"]);

// this run's own shape: the slot count is a launch flag (the trainer's default is
// 64) and the fused width is the sense count times the encoder width, so neither
// belongs in the drawing as a constant
const HS_REC = META.receipt || {}, HS_ARGS = META.args || {};
const HS_MODS = (HS_REC.active_modalities || []).length || 6;
const HS_D = D_MODEL;
const HS_SLOTS = (HS_ARGS.hself_slots || 0).toLocaleString();
detailFlow("hself", 900, 220, [
  { x: 10, y: 30, w: 230, h: 44, l1: "the model step, fused",
    l2: "all " + HS_MODS + " tokens end to end: " + (HS_MODS * HS_D) + " numbers", key: "node.hself" },
  { x: 310, y: 30, w: 230, h: 44, l1: "every slot gets a share",
    l2: HS_SLOTS + " centres, each a running average", key: "node.hself" },
  { x: 610, y: 30, w: 280, h: 44, l1: "presence ↔ wellness link",
    l2: "windows of 10,000 and 100,000 model steps", key: "node.hself" },
  { x: 610, y: 140, w: 280, h: 44, l1: "evidence rank test",
    l2: "beat its own 8 sign-flipped controls", key: "node.hself" },
], [
  { pts: [[240, 52], [310, 52]], label: "one vector", lx: 275, ly: 44, key: "node.hself" },
  { pts: [[540, 52], [610, 52]], label: "a share each", lx: 575, ly: 44, key: "node.hself" },
  { pts: [[750, 74], [750, 140]], key: "node.hself" },
], [
  "every slot gets a share of every model step and no slot wins one outright - the biggest share one slot holds and the number of slots a model step spreads over are live rows in the inspector panel (on run H81's stored table, 4.7 in 100 and 2,913 of 6,400)",
  "dark = watching only: no loss, no goal, no wellness term flows back, and none of its numbers is in the optimizer",
  "no slot can be counted before 27 windows on a 6,400-slot table (2,700,000 model steps at the hours level, 270,000 at the minutes level; 16 windows on a 64-slot table): 2 for the controls to warm up, then the rank average must clear a line set so that a whole table with no link would miscount a slot 135 times in 100,000",
  (() => {   // P225: when the table filled and what the novelty rule has said since, from the newest snapshot
    const hs = (SNAPS.map(s => (s || {}).hself).filter(isObj).slice(-1)[0]) || {};
    if (hs.slot_age_newest_frames === undefined || !hs.admission) return "when the table filled and the novelty rule's share since are not in this page's snapshots (older capture)";
    const filled = (hs.frames || 0) - hs.slot_age_newest_frames;
    const share = hs.admission.share_new;
    return "the last slot opened at model step " + filled.toLocaleString() + " (" + hs.slot_age_newest_frames.toLocaleString() + " model steps ago); since the counters began, the novelty rule has called "
      + (hs.admission.judged_new || 0).toLocaleString() + " of " + (hs.admission.checked || 0).toLocaleString() + " examined body-frames new"
      + (share === null || share === undefined ? "" : " (" + per100(share) + ")")
      + " - at nearly 100 of every 100 the refused counters cannot size a bigger table until the rule is tightened";
  })(),
]);

// How a tactile number is built, and why the divisor is the surprising half. Drawn
// because every pressure claim in this project rests on it: the numerator comes
// from MuJoCo's contact solver and the denominator from MIMo's skin grid, and
// nothing ties the two together.
detailFlow("sense_tactile", 920, 300, [
  { x: 10, y: 30, w: 250, h: 50, l1: "MuJoCo contact solver",
    l2: "20 contacts while resting, each a force in newtons", key: "term.contact-concentration" },
  { x: 10, y: 130, w: 250, h: 50, l1: "MIMo skin grid",
    l2: "4,683 points on 53 body parts at this age", key: "term.contact-concentration" },
  { x: 340, y: 30, w: 230, h: 50, l1: "add the forces up",
    l2: "40.3 N on the head, resting", key: "term.contact-concentration" },
  { x: 340, y: 130, w: 230, h: 50, l1: "count the points that felt it",
    l2: "14 of the head's 73", key: "term.contact-concentration" },
  { x: 315, y: 222, w: 280, h: 44, l1: "times one point's own size",
    l2: "169 mm2 on the head, 4 mm2 on a fingertip", key: "term.contact-concentration" },
  { x: 655, y: 78, w: 255, h: 56, l1: "force divided by that area",
    l2: "40.3 N / 2,366 mm2 = 17,036 Pa", key: "term.touch-pressure-scale" },
  { x: 655, y: 190, w: 255, h: 66, l1: "land on a FINGERTIP instead",
    l2: "266 N / 72 mm2 = 3,701,230 Pa", key: "term.contact-concentration" },
], [
  { pts: [[260, 55], [340, 55]], label: "newtons", lx: 300, ly: 47, key: "term.contact-concentration" },
  { pts: [[260, 155], [338, 155]], label: "which fired", lx: 299, ly: 147, key: "term.contact-concentration" },
  // each label above or below its slanted line, clear of it (the layout check found both crossed)
  { pts: [[570, 55], [655, 95]], label: "numerator", lx: 606, ly: 57, key: "term.contact-concentration" },
  { pts: [[570, 155], [650, 120]], label: "denominator", lx: 603, ly: 168, key: "term.contact-concentration" },
  { pts: [[455, 180], [455, 222]], key: "term.contact-concentration" },
  // RIGHT of the line on two lines: on one line at x 790 running right it ran past the 920-wide
  // viewBox, and anchored left of the line it sat against the "denominator" label and read as one
  // phrase with it
  { pts: [[782, 134], [782, 190]], label: ["same body,", "217x the reading"], lx: 788, ly: 160,
    anchor: "start", key: "term.contact-concentration" },
], ["the two halves come from DIFFERENT places: the force from the physics solver, the area from the skin grid, and nothing keeps them in step",
    "the divisor is not the footprint the body rests on - it is the points that fired times their OWN size, and one point runs 4 mm2 on a fingertip against 1,444 mm2 at the widest, a 361-fold range",
    "that 361-fold range is MIMo's own deliberate sensor spacing - a fingertip is sampled finely on purpose - so this is not a MIMo defect, it is a defect in READING the result as a pressure",
    "split the spread exactly (pressure is force over area, so the two shares sum to 100%): which part happened to touch accounts for 120.5%, how hard the landing was for MINUS 20.5%",
    "so the number does not measure impact weakly, it points the WRONG WAY: across 30 drop peaks force and this reading correlate BELOW ZERO (a correlation: not yet measured as a count)",
    "the worst reading in the set, 3,730,075 Pa, was a fingertip carrying 254 N; the biggest force in the set, 3,221 N on the head, read 1,361,526 Pa - less than half as much",
    "a 1,861 N hip landing from 0.10 m read 99,275 Pa, thirty-seven times LESS than that fingertip",
    "on the mechanical_v2_local path wellness reads this signal, which is why a fall does not grade there; v3 and v4 divide by the resolved contact footprint instead",
    "this page's run: " + (ARGS.contact_profile ? "--contact-profile " + ARGS.contact_profile : "its launch flags do not record --contact-profile") + " - see the contact-profile entry"]);

// THE WELLNESS SUB-DIAGRAM (2026-09-22). There was none: the box that decides what the planner
// aims at opened a popup and nothing drawn. It shows the two things wellness feeds and how they
// differ - the raw six-term sum (the motor gradient and the entity teacher, never changed by the
// build) and, under --wellness-rank normalized, the channels ranking of a remembered model step.
(function () {
  const wcr = (META.receipt || {}).wellness_channels || {};
  const mode = wcr.rank || ARGS.wellness_rank || "raw";
  const chans = (wcr.channels || []).filter(c => c !== "pain" && c !== "harm");
  const w = wcr.weights || {};
  const foot = [
    "Measured scores: six fixed formulas per body. Their sum teaches the named-things observer; its derivative supplies a separate input to the motor controller. Only the position formula contributes to that derivative.",
    "Predicted scores: a learned network estimates five fixed scores. It does not predict the extra goal-ranking channels. Command search uses those estimates as an objective only when prospective wellness is enabled with positive weight.",
    mode === "normalized"
      ? "this run ranks by channels (--wellness-rank normalized), each weighted as a share of the top priority: "
        + (chans.length ? (() => { const top = Math.max(...chans.map(c => Number(w[c]) || 0));
            return chans.map(c => c + " " + (w[c] !== undefined && top > 0 ? Math.round(100 * Number(w[c]) / top) + "%" : "?")).join(", "); })()
          : "channel list not in this receipt")
      : "this run ranks goals by the raw stored value (--wellness-rank " + (ARGS.wellness_rank || "raw, the default") + "): the channel boxes below describe the normalized mode",
    "Stored channel value = current reading minus that body's recent average. Contribution = weight x (stored value - running centre) / max(running spread, minimum scale). A memory whose harm is over "
      + (wcr.gate_line === undefined ? "the gate line" : Math.round(100 * wcr.gate_line) + "% of the way from no modelled aversion to an extreme load")
      + " is never chosen while any remembered model step of that body is under it; when every one is over it, the least harmful is chosen",
    "Controllability and contact control use 32 controlled readings: their gates open above three standard errors and close below one. A contact gate shuts if its newest controlled reading is more than 200 model steps old.",
  ];
  detailFlow("wellness", 1060, 540, [
    { x: 10, y: 20, w: 260, h: 44, l1: "six measured scores", l2: "fixed formulas, one set per body", key: "term.wellness" },
    { x: 400, y: 20, w: 260, h: 44, l1: "add the measured scores", l2: "raw sum, separate from goal rank", key: "term.wellness" },
    { x: 790, y: 20, w: 260, h: 44, l1: "observer target; motor derivative", l2: "only positions give a motor derivative", key: "term.wellness-gradient" },
    { x: 10, y: 130, w: 260, h: 44, l1: "first five measured scores", l2: "training targets; self-touch excluded", key: "term.approximate-evaluator" },
    { x: 400, y: 130, w: 260, h: 44, l1: "train the wellness predictor", l2: "squared score errors update its weights", key: "term.approximate-evaluator" },
    { x: 790, y: 130, w: 260, h: 44, l1: "current encoded readings", l2: "vision, tactile, audio, proprioception, motor", key: "edge.ln_out.wellness" },
    { x: 10, y: 240, w: 260, h: 44, l1: "predicted sensory lists", l2: "+ encoded candidate command", key: "term.approximate-evaluator" },
    { x: 400, y: 240, w: 260, h: 44, l1: "use the learned predictor", l2: "estimate five scores for the candidate", key: "term.approximate-evaluator" },
    { x: 790, y: 240, w: 260, h: 44, l1: "five predicted scores", l2: "optional command-search objective", key: "term.planner-blind" },
    { x: 10, y: 355, w: 260, h: 44, l1: "enabled ranking channels", l2: "selected fixed scores + extra measurements", key: "term.wellness-ranking" },
    { x: 400, y: 355, w: 260, h: 44, l1: "subtract history; centre and scale", l2: "stored changes compared per channel", key: "term.wellness-ranking" },
    { x: 790, y: 355, w: 260, h: 44, l1: "weight, add; apply harm rule", l2: "pain and harm are unranked gate inputs", key: "term.integrity" },
    { x: 790, y: 465, w: 260, h: 44, l1: "remembered sensory goal", l2: "highest eligible rank", key: "term.goal-ring" },
  ], [
    { pts: [[270, 42], [400, 42]], label: "six scores", lx: 335, ly: 34, key: "term.wellness" },
    { pts: [[660, 42], [790, 42]], label: "sum; derivative", lx: 725, ly: 34, key: "term.wellness-gradient" },
    { pts: [[140, 64], [140, 130]], label: "select five", lx: 180, ly: 101, key: "term.approximate-evaluator" },
    { pts: [[270, 152], [400, 152]], label: "targets", lx: 335, ly: 144, key: "term.approximate-evaluator" },
    { pts: [[790, 152], [660, 152]], label: "inputs", lx: 725, ly: 144, key: "edge.ln_out.wellness" },
    { pts: [[530, 174], [530, 240]], label: "learned weights", lx: 589, ly: 211, key: "term.approximate-evaluator" },
    { pts: [[270, 262], [400, 262]], label: "inputs", lx: 335, ly: 254, key: "term.approximate-evaluator" },
    { pts: [[660, 262], [790, 262]], label: "estimates", lx: 725, ly: 254, key: "term.planner-blind" },
    { pts: [[270, 377], [400, 377]], label: "measurements", lx: 335, ly: 369, key: "term.wellness-ranking" },
    { pts: [[660, 377], [790, 377]], label: "scaled changes", lx: 725, ly: 369, key: "term.wellness-ranking" },
    { pts: [[920, 399], [920, 465]], label: "select memory", lx: 976, ly: 436, key: "term.goal-ring" },
  ], foot);
})();

detailFlow("caregiver", 900, 263, [
  // the toy's spell and the gap are LAUNCH FLAGS; drawing 60 and 15 as constants put
  // the picture and the popup that opens from the same box on two different sources
  { x: 10, y: 44, w: 300, h: 44, l1: "adult here 180-300 s",
    l2: "a toy for " + (ARGS.caregiver_present_seconds || "?") + " s, then a "
        + (ARGS.caregiver_absent_seconds || "?") + " s gap - both launch flags",
    key: "node.caregiver" },
  { x: 400, y: 44, w: 260, h: 44, l1: "adult gone 67-97 s", l2: "walks out, waits 30-60 s, walks back", key: "node.caregiver" },
  { x: 10, y: 144, w: 440, h: 50, l1: "answers what the baby DOES", l2: "a 0.4 s look within 15 deg · any touch of the toy · a watched toy shifts 1.5 cm", key: "node.caregiver" },
  { x: 540, y: 144, w: 350, h: 50, l1: "speech to the ears", l2: "text-to-speech lines, 0.96 s to 3.04 s, quieter and later in the far ear", key: "edge.world.sense_audio" },
], [
  { pts: [[310, 66], [400, 66]], label: "leaves", lx: 355, ly: 58, key: "node.caregiver" },
  { pts: [[530, 44], [530, 22], [220, 22], [220, 44]], label: "comes back - and answers again", lx: 375, ly: 18, key: "node.caregiver" },
  { pts: [[100, 88], [100, 144]], label: "the four tests, every 0.01 s", lx: 106, ly: 104, anchor: "start", key: "node.caregiver" },
  { pts: [[450, 169], [540, 169]], label: "one line at a time", lx: 495, ly: 161, key: "edge.world.sense_audio" },
  { pts: [[890, 144], [890, 116], [270, 116], [270, 88]], label: "and the adult is still there", lx: 580, ly: 112, key: "node.caregiver" },
], ["the world picks these comings and goings at random, not the model - never read an absence as the model's choice",
    "while the adult is away nothing the baby does is answered, and the toy and the mobile go up out of the crib with it",
    "a look or a moving toy waits out the adult's turn; a touch is dropped if the adult is still speaking 2.0 s after the contact starts",
    "the toy is a mocap body, so a push cannot move it; the adult carries it (up to 6 cm/s, a fixed offset from the eye)",
    "with --toy-hand-off follow a hand that grips it for 0.1 s carries it until it lets go; this run: " + (ARGS.toy_hand_off || "not recorded"),
    "run H84, 8 bodies, 688 such lines: 78 of 100 followed the baby moving its own head over 1.5 cm in 0.5 s, 11 of 100 the adult carrying the toy in or out",
    "the adult has no ears at all: it cannot answer the baby's sounds, and the model has no voice in this world yet either"]);

let detailId = null;
// the entry key of the open detail panel: a click inside its diagram on a box or line with this key
// scrolls to the panel's text instead of opening the same text in a popup (flowDiagram's open)
let detailOwnKey = null;
// the entry key a detail panel shows: sense_/enc_/block boxes carry a dotted doc key, so it is
// resolved the way a click resolves it rather than assuming "node." + the raw id
function detailKeyFor(id) { return DOC[docKeyFor(id)] ? docKeyFor(id) : "node." + id; }
// the detail panel whose own entry is this key, or null
function detailIdForKey(key) { return Object.keys(DETAIL).find(id => detailKeyFor(id) === key) || null; }
// the panel's own text, scrolled into view and marked for 0.9 s: what a click inside the drawing
// on a box whose entry IS this panel's does, instead of opening the same text in a popup
function showDetailText() {
  const t = document.getElementById("detail-text");
  if (!t) return;
  if (t.scrollIntoView) t.scrollIntoView({ behavior: "smooth", block: "start" });
  t.classList.add("flash");
  setTimeout(() => t.classList.remove("flash"), 900);
}
// ONE PLACE PER BOX (operator 2026-10-02: "REDUNDANT POPUP"). A box that has a detail diagram
// opens this panel and no popup. The panel carries everything the popup carried for that box -
// what it is, when it runs, how it is computed, what would fool it, the note saying whose run
// "this run" means, the live values from the newest snapshot and the liveness sweep - and then
// the diagram. A box or line INSIDE the diagram still opens its own entry as a popup.
function renderDetail(id) {
  const root = document.getElementById("detail");
  if (!root || !DETAIL[id]) return;
  root.textContent = "";
  root.hidden = false;
  detailId = id; state.d = id; writeHash();
  const key = detailKeyFor(id);
  detailOwnKey = key;
  const d = DOC[key] || {};
  const h = el("h3", {}, d.t || id);
  h.appendChild(el("button", { class: "close", text: "×", onclick: () => {
    root.hidden = true; root.textContent = ""; detailId = null; detailOwnKey = null;
    state.d = ""; writeHash(); } }));
  root.appendChild(h);
  const text = el("div", { id: "detail-text" });
  root.appendChild(docFieldList(d));
  const measurements = el("details");
  measurements.appendChild(el("summary", {text: "Recorded measurements and component inventory"}));
  const note = docRunNote(d);
  if (note) text.appendChild(note);
  const rows = liveValuesFor(id).concat(livenessRows(id));
  if (rows.length) {
    text.appendChild(el("h4", { text: "live values (newest snapshot)" }));
    text.appendChild(kvGrid(rows));
  }
  const ids = componentIdsOfKey(key);
  if (ids.length) text.appendChild(componentsSection(ids, componentsHeading(ids.length)));
  measurements.appendChild(text); root.appendChild(measurements);
  root.appendChild(id === "wellness"
    ? el("div", { class: "wellness-detail-scroll" }, DETAIL[id]())
    : DETAIL[id]());
  root.appendChild(el("p", { class: "note", text: "Every box and labeled line in this detail diagram opens the "
    + "entry for the data it names; a box whose entry is this panel's own scrolls back to the text above." }));
}

// a box's live rows: its own (liveValuesBase), then the rows of the repairs the run built in it (D-E4): a repair path
// drawn as a line of the box, and the attention cap, which bounds stage 1 in every block (block 0's box says so in its
// sub-caption; the cells of the later blocks have no room for it) and the motor encoder's attention
function liveValuesFor(id) {
  return liveValuesBase(id).concat(repairRows(id));
}
function repairRows(id) {
  const out = [], cp = META.component_params || {}, argsm = META.args || {};
  for (const k of HOSTED_PARAM[id] || [])
    if (cp[k]) out.push([HOSTED_DOC[k][0].replace(/^\+ /, "") + " built in this box",
                         cp[k].toLocaleString() + " learned numbers in " + META.run + "'s receipt; its own entry opens from the "
                         + "box's line '" + HOSTED_DOC[k][0] + "'"]);
  if ((/^block\d+_s1$/.test(id) || id === "enc_motor") && (argsm.attention_cap || "off") === "on") {
    const b = Number(argsm.attention_cap_bound), rec = META.receipt || {};
    // with n inputs and scores held in [-B, B], one input takes at most 1 / (1 + (n - 1) exp(-2B)) of a head's weight
    // (lem/model/attention_cap.py largest_weight)
    const most = n => (100 / (1 + (n - 1) * Math.exp(-2 * b))).toFixed(1);
    const nIn = id === "enc_motor" ? (rec.handle_names || []).length : (rec.active_modalities || []).length;
    out.push(["attention cap in " + META.run, "on" + (b === b && argsm.attention_cap_bound !== null
        ? ", --attention-cap-bound " + argsm.attention_cap_bound : "")
      + ": each attention score of stage 1 in all " + (N_BLOCKS || "the") + " blocks and of the motor encoder's attention "
      + "is the bound times the cosine of query and key" + (b === b && b > 0 && nIn > 1
        ? ", so one of the " + nIn + (id === "enc_motor" ? " handle rows" : " sense slots") + " takes at most " + most(nIn)
          + " of every 100 parts of a head's weight" : "")
      + " (lem/model/attention_cap.py; ATTENTION_CAP)"]);
  }
  return out;
}
function liveValuesBase(id) {
  const rec = META.receipt || {}, argsm = META.args || {};
  const cp = META.component_params || {};
  const sN = SNAPS.length ? SNAPS[SNAPS.length - 1] : null;
  if (id === "world") return [
    ["what it is", "the simulated crib - not part of the model, no weights"],
    ["world clock", sN ? fmt(sN.senses.sim_time, 1) + " s of body time" : "—"],
    ["skin contacts now", sN ? sN.senses.contacts : "—"]];
  if (id === "memory") return [
    ["learned parameters", cp.persistent_memory ? cp.persistent_memory.toLocaleString() : "?"],
    ["state", rec.persistent_memory || "?"],
    ["live numbers", "in the inspector's 'Memory beyond the window' panel"]];
  if (id === "self_state") {
    // P204: "learned parameters" overstated it - on a page without component_params the
    // figure counted 468 buffer values, and on H83 (128 wide) 262,022 of self perception's
    // 912,143 weights got no gradient under its flags (the retired legacy forecast head and
    // the belief bias). The rows below print THIS page's run's own counts, from its receipt
    const IDLE = META.idle_params || {};
    const tot = cp.self_state || 0, id0 = ((IDLE.self_state || {}).elements || 0), rows = [];
    rows.push(["saved numbers", tot ? tot.toLocaleString() : "?"]);
    if (id0 && tot) {
      rows.push(["of those, reached by a lesson", (tot - id0).toLocaleString()
        + " - the other " + id0.toLocaleString() + " get no gradient in this run"]);
      // g.measured is what this run's own checkpoints showed when the page was built
      // (build_explainer.measure_idle_groups), or a sentence saying it is unmeasured
      ((IDLE.self_state || {}).groups || []).forEach(g =>
        rows.push(["  " + g.elements.toLocaleString() + " of them",
                   g.why + (g.measured ? " - " + g.measured : "")]));
    }
    rows.push(["mode", rec.self_state || argsm.self_state || "?"]);
    rows.push(["prediction mode", rec.self_prediction || "?"]);
    rows.push(["live numbers", "in the inspector's 'Self perception' panel"]);
    return rows;
  }
  if (id === "voice") {
    const vh = rec.voice_head || {}, v = (sN || {}).voice;
    const rows = [["parameters", (vh.params !== undefined ? vh.params.toLocaleString() : "?")
                   + " - in the optimizer's list, reached by no lesson"],
                  ["outputs", (vh.output_dim || 80)
                   + " numbers, each held between 0 and 1 by a sigmoid - the reserved width of the shared-memory action slot; 56 of them could map onto the repo's vocal-tract parameter set (11 parameters, a synth clock, 44 tract diameters), the other 24 map onto nothing named in the tree"]];
    if (sN) rows.unshift(["from snapshot", "step " + sN.header.step + (META.trained_step
      ? " of this probe (the rest of the model trained for " + META.trained_step.toLocaleString()
        + " steps; these weights never moved)" : "")]);
    if (isObj(v)) rows.push(
      ["output level (RMS of the 80 numbers)", score(v.rms) + " of the 0-to-1 range - a readout no lesson has touched sits at the middle, 50%"],
      ["smallest / largest control", score(v.min) + " / " + score(v.max) + " of the range"],
      ["effect on the world", String(v.world_effect)]);
    return rows; }
  if (id === "hself") {
    const hm = (rec.active_modalities || []).length,
          hd = D_MODEL,
          hn = argsm.hself_slots || 0;
    return [
      ["learned parameters", "none - running averages only, never in the optimizer"],
      ["slots", hn || "?"],
      ["what each slot is a centre of", hm ? hm + " tokens laid end to end = "
        + (hm * hd) + " numbers, scaled to length 1" : "?"],
      ["stored numbers", hn && hm ? (hn * hm * hd).toLocaleString()
        + " in the slot centres alone - in neither the parts sum nor the optimizer "
        + "total under the diagram" : "?"],
      ["mode", argsm.hierarchical_self || "?"],
      ["live numbers", "in the inspector's 'Hierarchical self' panel"]];
  }
  if (id === "caregiver") return [
    ["mode", argsm.caregiver_mode || "?"],
    ["a toy stays / then a gap", (argsm.caregiver_present_seconds || "?") + " s / " +
                                 (argsm.caregiver_absent_seconds || "?") + " s"],
    ["the adult's own spells", "here 180-300 s, gone 67-97 s - drawn at random each time, not a launch flag"],
    ["the toy hand-off (launch flag)", argsm.toy_hand_off ? "--toy-hand-off " + argsm.toy_hand_off
       : "not recorded by this run's launch (the flag arrived on 2026-09-21)"],
    ["the toy follows a grip (the world's own report)", (() => {
       const cg = (sN || {}).caregiver;
       return isObj(cg) && cg.object_follows_grip !== undefined
         ? (cg.object_follows_grip ? "yes" : "no") + (cg.object_grips !== undefined
            ? " - " + cg.object_grips + " grips so far, held by the infant now: "
              + (cg.object_held_by_infant ? "yes" : "no") : "")
         : "not in the newest snapshot (a capture from before the hand-off existed)"; })()]];
  // THE BUILD'S PARTS (2026-09-22). Each reads its own block of the newest snapshot, or says
  // which launch flag leaves it off.
  if (id === "efference") {
    const E = (sN || {}).efference, st = rec.efference_heads || {};
    // where the heads reach the planner (META.efference_routes, schema.efference_routes - the rule
    // the diagram's lines and the tour step use): the goal-ranking channels under --wellness-rank
    // normalized, and the action search under --physical-goals memory (lem_model.LemLite._plan_chain
    // builds PhysicalGoalCost from nothing else); the gaze copy is added in every imagined step
    // whenever the heads' stamp records it
    const pg = argsm.physical_goals, er = META.efference_routes || {};
    const chWords = { controllability: "controllability", learning_progress: "learning progress" };
    const rows = [["learned parameters", (cp.efference_heads || 0).toLocaleString()],
                  ["gaze copy (eye command into the vision prediction)", st.gaze_copy_into_vision
                     ? "built: added to vision's next-model-step prediction in the lessons and in every imagined step, the planner's action search included"
                     : "not recorded in this run's receipt, so no gaze copy line is drawn"],
                  ["the heads in the goal ranking", (er.goal_channels || []).length
                     ? "read by " + er.goal_channels.map(c => chWords[c] || c).join(" and ")
                       + " (--wellness-rank normalized); controllability counts only while its evidence gate is open"
                     : "not read: this run does not rank goals by controllability or learning progress (they rank goals only under --wellness-rank normalized)"],
                  ["body head in the planner's action search", er.action_search
                     ? "read: --physical-goals memory scores each candidate command by how far the body head's command-long prediction lands from the pose stored with the remembered goal"
                     : (pg === "memory" ? "not read: --physical-goals memory, but this run's planner is off (--intention off)"
                       : pg ? "not read: under --physical-goals " + pg + " the action search never calls the body head"
                       : "not read: this run's launch does not record --physical-goals (the flag arrived on 2026-09-24)")
                       // the heads still reach the choice of command through the goal (the row above, the
                       // dashed line): review of 2026-09-25, finding 1
                       + ((er.goal_channels || []).length ? "; the heads still reach the choice of command through the goal ranking (the row above)" : "")],
                  ["lesson weight (each head)", st.weight !== undefined ? "--efference-weight " + st.weight : "?"],
                  ["measured spreads the loss is scaled by", st.scales
                     ? Object.entries(st.scales).map(([k, v]) => k.replace(/_(cm|deg)$/, "") + " "
                         + (/_cm$/.test(k) ? fmt(v, 2) + " cm" : /_deg(?:_|$)/.test(k) ? fmt(v, 2) + " degrees" : "(unit not recorded)")).join(", ") : "?"]];
    if (isObj(E) && isObj(E.window_so_far)) {
      const w = E.window_so_far;
      const pr = (a, b, u) => (w[a] === undefined ? "—" : fmt(w[a], 2) + " " + u) + " against "
        + (w[b] === undefined ? "—" : fmt(w[b], 2) + " " + u);
      rows.push(["next 50 ms: head's error against doing nothing", pr("body_cm_err_next", "body_cm_do_nothing_next", "cm")]);
      rows.push(["command's 250 ms: head's error against doing nothing", pr("body_cm_err_latch", "body_cm_do_nothing_latch", "cm")]);
      rows.push(["eyes at 50 ms: head's error against doing nothing", pr("gaze_deg_err_1", "gaze_deg_do_nothing_1", "degrees")]);
    } else rows.push(["this snapshot", String(E === undefined ? "carries no efference block (a capture from before the build's schema v3)" : E)]);
    return rows;
  }
  if (id === "contact_head") {
    const C = (sN || {}).contact_control, st = rec.contact_head || {};
    const rows = [["learned parameters", (cp.contact_head || 0).toLocaleString()],
                  ["patches", (st.patch_names || []).length || 53],
                  ["foreseen when the probability beats the base rate of starting by",
                   st.margin ? Math.round(100 * st.margin.self) + " in 100 for an own part, " + Math.round(100 * st.margin.world)
                     + " in 100 for an outside thing" : "?"],
                  ["contact_control's weight in the goal ranking", st.contact_control_weight !== undefined
                     ? "--contact-control-wellness " + st.contact_control_weight : "?"]];
    if (isObj(C)) rows.push(["evidence gate, this model step", C.gate_open ? "open: foreseen touches pay"
      : "shut: foreseen touches pay 0" + (C.gate_why ? " - " + plainOr(C.gate_why, CONTACT_GATE_WHY) : "")]);
    return rows;
  }
  if (id === "synergy") {
    const st = rec.motor_synergy || {};
    return [["learned parameters", (cp.motor_synergy || 0).toLocaleString()],
            ["size of z", st.k !== undefined ? String(st.k) : st.dim !== undefined ? String(st.dim) : ((sN || {}).synergy || {}).dim || "?"],
            ["used by the planner", st.planner === false ? "no: learning from the controller's movements first" : "yes"],
            ["used for exploration", st.babble ? "yes" : "no"],
            ["what it trains on", "the posture changes the body really made over each command, weighted by following x movement"]];
  }
  if (id === "entity_self") {
    const es = (sN || {}).entity_self;
    if (!isObj(es) || !es.names) return [["this snapshot", "carries no named-things block"]];
    const rd = es.reader || {};
    return [["named things", es.names.length + ": " + es.names.join(", ")],
            ["teacher", es.teacher || "?"],
            ["minutes level: things with a proven link", es.l2_counted === -1 ? "-1: cannot be measured yet" : String(es.l2_counted)],
            ["hours level: things with a proven link", es.l3_counted === -1 ? "-1: cannot be measured yet" : String(es.l3_counted)],
            ["the reader", rd.gate_open ? "ON, adding " + wUnits(rd.applied_weight) + " per unit of link x presence"
                                        : "off, adding nothing" + (rd.why ? " - " + plainOr(rd.why, READER_WHY) : "")],
            ["live numbers", "in the inspector's 'Named things' panel"]];
  }
  const s = sN;
  if (!s) return [["snapshots", "none yet - run the probe or enable --explain-every-min"]];
  const out = [["from snapshot", "step " + s.header.step + (META.trained_step
    ? " of this probe (the weights were trained for " + META.trained_step.toLocaleString() + " steps)" : "")]];
  // NODE_PARAM may name more than one key for a box (the shared LayerNorm); take the
  // first one this page carries a count for
  const pk = [].concat(NODE_PARAM[id] || [], id.startsWith("enc_") ? [id] : []).find(k => cp[k]);
  if (pk) out.push(["learned parameters", cp[pk].toLocaleString()
    + (id === "ln_out" ? " - this run built the shared LayerNorm with a learned gain and "
      + "bias, so this layer no longer holds a token's size at 1.0" : "")]);
  else if (/^block\d+_/.test(id) || id === "assembly" || id === "final_norm") {
    if (cp.forward_model) {
      // when the counts come from the checkpoint's tensors they include the causal
      // masks: fixed true/false triangles, not weights, and not "learned parameters"
      const fmr = rec.forward_model || {};
      const ctxL = (fmr._seq_length || 0) / (fmr._n_modalities || 1);
      const saved = /saved numbers/.test(META.component_params_basis || "");
      const msk = (saved && (fmr.stages || {}).causal_temporal !== false)
        ? N_BLOCKS * ctxL * ctxL : 0;
      out.push([saved ? "saved numbers" : "learned parameters",
        "inside the core's " + cp.forward_model.toLocaleString()
        + (msk ? " - of which " + (cp.forward_model - msk).toLocaleString()
                 + " are learned weights and " + msk.toLocaleString()
                 + " are true/false attention masks no optimizer touches" : "")]);
    }
  } else if (id === "gaze")
    out.push(["learned parameters",
      "none in use - gaze is the motor controller's own gaze handle"
      + (cp.gaze_head ? "; an unused gaze head (" + cp.gaze_head.toLocaleString()
         + " saved numbers) rides in the checkpoint so old checkpoints load - "
         + "nothing reads it" : "")]);
  else if (id.startsWith("sense_") || ["curiosity", "curriculum", "replay", "imagination", "ln_out"].includes(id))
    out.push(["learned parameters", "none - not a learned module"]);
  const m = id.replace(/^(sense_|enc_)/, "");
  if (id.startsWith("sense_")) {
    out.push(["how it arrived", (s.header.source_mask || {})[m] || "?"]);
    const std = (s.tokens.token_temporal_std || {})[m];
    if (std !== undefined) out.push(["token variation over time (0% = collapsed)", tokSize(std)]);
  } else if (id.startsWith("enc_")) {
    out.push(["token size after the shared LayerNorm",
      "a real token's full size (RMS 1) while this encoder's own spread stays above about "
      + "two hundredths; below that the norm under-divides and the token leaves smaller (a "
      + "spread of one thousandth leaves it at 30% of full size)"]);
    const e = (s.predictions.error_of_now || {})[m];
    if (e !== undefined) out.push(["one-step prediction error, this sense", tokErr(e)]);
  } else if (id === "td_gate") {
    // the corner tag reads "15", which is a count of NUMBERS, not of gates: one
    // nn.Linear(2,1) - 2 weights and a bias - per predicted sense
    if (cp.td_gates) {
      const nPh = (rec.prediction_head_modalities || []).length;
      out.push(["what those " + cp.td_gates + " numbers are", nPh
        ? nPh + " gates of 3 (2 weights and a bias), one for each sense that has a "
          + "prediction head - " + rec.prediction_head_modalities.join(", ")
          + "; the motor sense has no gate at all"
        : "one gate of 3 numbers per sense that has a prediction head"]);
    }
    out.push(["most the gate may blend in",
      "50 parts of the token in 100 - it is half a sigmoid (0 = the core reads the encoder alone, "
      + "50 = half the token is the model's own guess of NOW)"]);
    // solved from the page's own data: fed = raw + g x (prediction of now - raw)
    const rawT = s.tokens.raw || {}, fedT = s.tokens.fed, nowT = s.predictions.of_now;
    if (fedT && typeof fedT === "object" && nowT && typeof nowT === "object") {
      for (const k of Object.keys(rawT)) {
        if (!Array.isArray(fedT[k]) || !Array.isArray(nowT[k])) continue;
        let acc = 0, n = 0;
        for (let i = 0; i < rawT[k].length; i++) {
          const gap = nowT[k][i] - rawT[k][i];
          if (Math.abs(gap) > 1e-3) { acc += (fedT[k][i] - rawT[k][i]) / gap; n++; }
        }
        out.push(["blend used for " + k + " (parts of the token out of 100; 0 to 50)", n ? blend(acc / n) : "0 in 100"]);
      }
    } else out.push(["this snapshot", fedT === undefined
      // a snapshot written before the capture recorded the tokens the core consumed has
      // no fed field at all; String(undefined) printed the word "undefined" here, which
      // the headless-browser check found (the inspector's gate panel already said this)
      ? "predates the capture of the tokens the core consumed, so the blend cannot be solved from it"
      : String(fedT)]);
  } else if (id === "heads") {
    for (const [k, v] of Object.entries(s.predictions.error_of_now || {}))
      out.push(["one-step error · " + k, tokErr(v)]);
    out.push(["causal_pe (mean one-step error, real senses)", tokErr(s.predictions.causal_pe)]);
  } else if (id === "recon") {
    out.push(["the scale these sit on", "encoder tokens are held at RMS 1.0, so a "
      + "decoder that always output zeros scores 100% of the all-zeros answer's error and a perfect one 0%"]);
    for (const [k, v] of Object.entries(s.predictions.recon_dec_error || {}))
      out.push([k + (v > 1 ? " - worse than saying nothing" : ""), tokErr(v)]);
  } else if (id === "motor" || id === "gaze") {
    const hold = (s.header.flags || {}).motor_hold || 5;
    out.push(["confidence", score(s.motor.confidence) + " (0% = none, 100% = full)"]);
    out.push(["command age", s.motor.order_age + " of 0-" + (hold - 1)
      + " (0 = the command was latched on this step, " + (hold - 1)
      + " = its last step before the next one). A snapshot cadence that is a "
      + "multiple of " + hold + " lands on latch steps only, so this reads 0 every time."]);
    // gap_closed_pct compares the start with the closest PREDICTED depth, so a one-command plan
    // (every plan under --hand-target-mode wrist_relative_v1) has no predicted depth and its 0% is
    // a definition, not a measurement: the row is shown only for a plan of two commands or more
    const li = (s.intention || {}).last || {};
    const gp = li.gap_closed_pct;
    if (gp !== null && gp !== undefined && Number(li.plan_length) > 1)
      out.push(["the newest plan's closest predicted depth", pctNum(gp)
        + " of the way from this model step's distance to the goal tokens down to zero (0% = no predicted depth closer, 100% = at the goal tokens)"]);
    // DEGREES, never radians alone (the eye servo is sent radians; the page reads degrees)
    if (s.motor.gaze_az_el) out.push(["gaze command (degrees right / up)",
      s.motor.gaze_az_el.map(x => fmt(x * 180 / Math.PI, 1) + "°").join(" / ")]);
  } else if (id === "wellness") {
    // one row per score the snapshot carries, and the sum of exactly those rows: the
    // grounded set gained self_touch on 2026-09-21, so a typed term count would go stale
    // here the way "5 grounded" did on the box above
    let sum = 0, nTerms = 0;
    for (const [k, v] of Object.entries(s.drives.wellness_body || {})) {
      out.push([k + (k === "tactile_comfort" ? " (its span is 0% to 85%, not 0% to 100%)"
                     : k === "self_touch" ? " (its span is 0 to the --self-touch-wellness weight, "
                       + ((argsm.self_touch_wellness !== undefined && argsm.self_touch_wellness !== null)
                          ? argsm.self_touch_wellness + " in this run" : "which this page's build did not publish")
                       + "; the planner's copy of wellness has no output for it, so an imagined model step is never scored on it)"
                     : ""), score(v) + " of the 0-to-1 range"]);
      sum += Number(v) || 0; nTerms++;
    }
    out.push(["the " + nTerms + "-term sum the motor gradient is taken of", wUnits(sum)]);
    if (s.drives.prediction_error_source)
      out.push(["what prediction_error was measured against", s.drives.prediction_error_source]);
    if (s.drives.coherence_source)
      out.push(["which reading of the body map the coherence score carried",
        s.drives.coherence_source + (s.drives.coherence_source === "body_schema_extent"
          ? " - the mean of the map's co-movement numbers, its 26 handles' and the outside things' "
            + "(a handle's number counts how often its command's change and proprioception's change "
            + "fall in the same model step; it does not say the command caused the change)"
          : s.drives.coherence_source === "body_schema_certainty"
          ? " - how far the handles' co-movement numbers sit from the middle of their range, which "
            + "rises as those numbers fall"
          : " - no body map ran, so this is the cosine likeness of prediction to arrival")]);
    out.push(["curiosity drive at this model step",
      score(s.drives.curiosity_drive) + " of full drive; its effect on goal value depends on the enabled ranking mode"]);
  } else if (id === "curiosity") {
    out.push(["drive", score(s.drives.curiosity_drive) + " of the full drive (zero at and outside the configured error-band edges; maximum at its middle)"]);
    out.push(["zone", s.drives.curiosity_zone || "?"]);
  } else if (id === "intention") {
    // the newest planning event: the plan record (planLine) and the 20 updates' own record of the
    // score's two parts (descentRows), with the senses the score compared read off this snapshot
    const it = s.intention.last;
    const smI = s.header.source_mask || {};
    const nCmp = Object.keys(smI).filter(m => m !== "motor" && smI[m] === "real").length;
    if (it && typeof it.step === "number")
      out.push(["the newest planning event", "step " + it.step.toLocaleString() + ", "
        + (s.header.step - it.step) + " model steps before this snapshot"]);
    out.push(["the plan", it ? planLine(it, nCmp) : "none in this snapshot"]);
    descentRows(it, nCmp).forEach(r => out.push(r));
    out.push(["goal ring", s.intention.goal_ring_size + " entries (200 = full)"]);
    out.push(["the planner runs", argsm.intention_every
      ? "on 1 model step in " + argsm.intention_every + " (--intention-every " + argsm.intention_every + ")"
      : "on the --intention-every cadence; this page's build did not publish the flag"]);
  } else if (id === "imagination") {
    // a bare sum could not be read: it grows with the chain's depth and with how
    // many senses happened to be measured that model step, and nothing placed it
    const L = s.losses || {}, bd = L.breakdown || {}, IL = L.imagination_loss;
    const hm = (META.receipt || {}).prediction_head_modalities || [];
    const gr = hm.filter(m => (s.header.source_mask || {})[m] === "real");
    const nT = imagDepth() * gr.length;
    out.push(["how far ahead it guessed", imagDepth() + " model steps deep"]);
    out.push(["senses graded this model step", gr.length + " of " + hm.length + " ("
      + (gr.join(", ") || "none") + ") - a sense that was not measured is neither "
      + "graded nor fed back"]);
    if (IL != null && nT)
      out.push(["error per guess", tokErr(IL / nT)
        + " (0% = the guess was exact; two tokens with nothing to do with each other "
        + "score 200%)"]);
    if (s.predictions && s.predictions.causal_pe != null)
      out.push(["the one-step guess, same model step, same scale",
                tokErr(s.predictions.causal_pe) + " per sense"]);
    out.push(["summed over every depth and sense, BEFORE the lesson weight",
      tokErr(IL) + " summed over " + nT + " guesses - the number the run tab charts"]);
    // the loss panel one tab away prints the same lesson after its weight, and
    // nothing said the two sizes were one lesson
    if (bd.imagination != null)
      out.push(["the same lesson after its weight, in the loss panel",
        (IL ? pctNum(100 * bd.imagination / IL) + " of the number above - the "
                + "imagination weight; the loss panel's bar is this one" : "a lesson value with no lived unit")]);
    if (bd.imagination != null && num(bd.next_frame) > 0)
      out.push(["its size beside the other lessons, this same step", pctNum(100 * bd.imagination / bd.next_frame)
        + " of the one-step lesson"
        + (num(bd.motor) > 0 ? " and " + pctNum(100 * bd.imagination / bd.motor) + " of the body-command lesson" : "")]);
    if (s.header.weights === "frozen")
      out.push(["was anything learned from it here?", "no - this snapshot's weights are "
        + "frozen, so the number is a measurement, not a lesson being applied"]);
  } else if (id === "replay") {
    const r = s.losses.replay;
    if (r) { out.push(["memory filled", per100(r.fill_frac) + " slots"]); out.push(["re-study error", "a lesson value with no lived unit - " + NOT_COUNT + "; the run tab charts it against its own start"]); }
    else if ((s.header.flags || {}).eval_only || (META.args || {}).eval_only)
      out.push(["re-study passes in this run", "0 - launched --eval-only, so the ring "
                + "is never built and no weight moved from memory"]);
    else out.push(["re-study", "no record this step - replay off, or the ring is still filling"]);
  } else if (id === "curriculum") {
    // One row per field of the phase's limits, each shown for what it is.
    // phase_name is skipped: it repeats the phase row above it. A count is printed as
    // a count and a true/false as on/off - running every field through a 3-decimal
    // number format printed "3.000" for a chain depth of 3. And context_length is a
    // phase setting the MIMo trainer never applies, so showing it bare put a 32 two
    // columns from a box saying 128 model steps.
    const dl = s.header.dev_limits || {};
    const fmr2 = rec.forward_model || {};
    const liveCtx = s.header.ctx_len !== undefined ? s.header.ctx_len
                    : (fmr2._seq_length || 0) / (fmr2._n_modalities || 1);
    const NAMED = { rom_limit: "reach limit", strength_limit: "strength limit",
                    exploration_noise: "babble noise" };
    out.push(["phase", s.header.phase || dl.phase_name || "?"]);
    out.push(["age", fmt(s.header.age_months, 1) + " months"]);
    if (dl.rom_limit !== undefined)
      out.push(["reach limit", "one command may move a handle "
                + Math.round(40 * dl.rom_limit)
                + " cm (0 cm = pinned in place, 40 cm = the full reach)"]);
    if (dl.strength_limit !== undefined)
      out.push(["strength limit",
                Math.round(100 * dl.strength_limit) + "% of full push (0% = limp, 100% = full)"]);
    if (dl.exploration_noise !== undefined) {
      // the snapshot's value is what the trainer HANDED THE MODEL this model step, not
      // the phase's own number: --babble-noise replaces it outright and --babble-bouts
      // switches it off on whole stretches, so a 0.000 here is not a phase setting
      const en = Number(dl.exploration_noise) || 0;
      out.push(["babble noise on this model step", (en > 0 ? "a random offset of " + Math.round(100 * en) + " cm of position and "
                            + Math.round(10 * en * 180 / Math.PI)
                            + " degrees of rotation added to a command, one standard deviation"
                          : "no random offset was added on this model step")
                + (argsm.babble_noise !== null && argsm.babble_noise !== undefined
                   ? "; this run's --babble-noise " + argsm.babble_noise
                     + " replaces the phase's own value, and a bout schedule can switch it "
                     + "off for stretches of model steps" : "")]);
      // THE SCHEDULE IS NOT THE EFFECT. Babble is added only on a model step where the
      // planner produced no plan; the planner runs every --intention-every steps and
      // the body latches a new command every --motor-hold steps, so the share of
      // executed commands that can carry babble is 1 - gcd(hold, every)/every.
      const ie = argsm.intention_every, mh = argsm.motor_hold;
      if (argsm.intention !== "off" && ie > 0 && mh > 0) {
        const g2 = (a, b) => b ? g2(b, a % b) : a;
        const share = g2(mh, ie) / ie;
        out.push(["does that babble reach the body?", share >= 1
          ? "no - --intention-every " + ie + " divides --motor-hold " + mh
            + ", so every latched command is a planner command and the babble branch is skipped on all of them"
          : "on " + Math.round(100 * (1 - share)) + " of every 100 latched commands - the planner writes the other "
            + Math.round(100 * share) + " (--intention-every " + ie + ", --motor-hold " + mh + ")"]);
      }
    }
    for (const [k, v] of Object.entries(dl)) {
      if (k === "phase_name" || NAMED[k]) continue;
      if (k === "context_length")
        out.push(["context_length (this phase's definition - the MIMo trainer does not apply it)",
                  v + "; the window actually running is " + liveCtx + " model steps"]);
      else if (typeof v === "boolean") out.push([k, v ? "on" : "off"]);
      else if (typeof v === "number")
        out.push([k, Number.isInteger(v) ? String(v) : "a phase setting with no lived unit - " + NOT_COUNT]);
      else out.push([k, String(v)]);
    }
    out.push(["model steps the model actually runs", String(liveCtx)]);
    // capture.py should also write header.ctx_len = int(model.ctx_len) so the two
    // numbers on this page come from one source and cannot disagree.
  } else if (id === "td_gate") {
    const g = s.drives.td_gate;
    if (g) for (const [k, v] of Object.entries(g))
      out.push(["blend into " + k + " (parts of the token out of 100: 0 = measured sense only, 50 = the cap)", blend(v)]);
    else out.push(["blend now", "no gate reading in this snapshot (closed or off)"]);
  } else if (/block\d+_s\d/.test(id)) {
    out.push(["maps", "see the instance inspector tab"]);
  }
  return out;
}
// every diagram box maps to a "node.*" entry in schema.py's DOC
function docKeyFor(id) {
  const b = id.match(/^block\d+_(s\d|ffn)$/);
  if (b) return "node.block_" + b[1];
  if (/^(sense|enc)_/.test(id)) return "node." + id.replace("_", ".");
  return "node." + id;
}
// WHAT THE LIVENESS SWEEP MEASURED FOR THIS BOX (docs/gain_of_function_v3/measurements/,
// read into META.liveness when the page was built): a dead layer or a locked attention pool
// inside the part the box draws, so no box reads as working because it is wired
const LIVE_PARTS = {
  motor: ["motor_controller."], wellness: ["wellness.approximate."], recon: ["recon_decoders."],
  memory: ["persistent_memory."], enc_motor: ["encoder_registry.encoders.motor."],
  enc_tactile: ["encoder_registry.encoders.tactile."],
  block0_s1: ["forward_model.blocks.0.cm_attn"], block0_s2: ["forward_model.blocks.0.ct_attn"],
  block0_s3: ["forward_model.blocks.0.cmb_attn"],
};
// a measured part by its name with the code path beside it (build_explainer.liveness_part_name: the
// development dashboard's words for the part), never the bare path
function livenessName(k) {
  const names = ((META.liveness || {}).sweep || {}).names || {};
  return names[k] || k;
}
function livenessRows(id) {
  const L = META.liveness || {}, sw = L.sweep || {};
  const pre = LIVE_PARTS[id];
  if (!pre) return [];
  const rows = [];
  const hit = k => pre.some(p => k.indexOf(p) === 0);
  // counts, never a bare decimal (DESIGN.md section 11)
  for (const [k, v] of Object.entries(sw.dead_layers || {}))
    if (hit(k)) rows.push([livenessName(k) + ", in the liveness sweep " + (sw.file || ""),
      per100(v) + " of its units ever respond (0 = dead: it outputs its bias for every input, 100 = every unit responds)"]);
  for (const [k, a] of Object.entries(sw.locked_attention || {}))
    if (hit(k)) rows.push([livenessName(k) + ", attention in the liveness sweep", "locked: its single largest input takes "
      + per100(isObj(a) ? a.top_share : null) + " parts of its attention"
      + (isObj(a) && a.keys ? ", of " + a.keys + " inputs it could read" : "")]);
  return rows;
}
function nodeClicked(id, label) {
  // a box with a detail diagram opens ONE place, the panel under the picture, and closes any open
  // popup; every other box opens its popup with this run's live values
  if (DETAIL[id]) {
    docTrail = [];
    renderDocPanel();
    renderDetail(id);
    const root = document.getElementById("detail");
    if (root && root.scrollIntoView) root.scrollIntoView({ behavior: "smooth", block: "start" });
    return;
  }
  showDoc(docKeyFor(id), label, liveValuesFor(id).concat(livenessRows(id)));
  readComponents(redrawComponentViews);
}

// ================================================================== components: the code and QA map
// THE ARCHITECTURE TAB'S CODE AND QA MAP (the recovery directive of Oct 5, 2026, sections 7C and 9).
// INPUT     the component manifest's view (tools/component_manifest.py status): the copy the build put into this
//           page (data-components, with the time it was read) and, when the development dashboard serves the page,
//           GET /api/components, read when the page loads and again when a box is clicked 60 seconds or more after
//           the last read.
// OPERATION every box of the drawing has an entry key ("node.intention"); the view's by_block lists the components
//           that name that key in the manifest. A box's popup, or its detail panel, lists each of them; the address
//           #component=ID (the Tests page's links) opens the popup of one component and outlines the boxes it is drawn
//           in; the list under the drawing opens any component, a component without a box too.
// OUTPUT    per component: its canonical id and plain name; one sentence of what it does; its switch state in the
//           active run; its last code change (Pacific time, 12-hour clock, short commit hash); its last independent
//           QA with the reviewed commit, or the sentence that none is on record; a red QA STALE label when its mapped
//           code changed after the reviewed commit; its open defects with their age; the tests that name it and are
//           not done, with their section; and, as an error, a manifest entry that no longer matches its code.
let COMPONENTS = jload("data-components", null);
let componentsLive = false, componentsReadAt = 0;
function readComponents(after) {
  if (!/^https?:$/.test(location.protocol) || Date.now() - componentsReadAt < 60000) return;
  componentsReadAt = Date.now();
  fetch("/api/components", { cache: "no-store" })
    .then(r => (r.ok ? r.json() : null))
    .then(d => {
      if (d && Array.isArray(d.components)) {
        COMPONENTS = d; componentsLive = true;
        if (after) after();
      }
    })
    .catch(() => {});
}
function componentById(id) {
  return ((COMPONENTS || {}).components || []).find(c => c && c.id === id) || null;
}
function componentIdsOfKey(key) {
  const ids = ((COMPONENTS || {}).by_block || {})[key];
  return Array.isArray(ids) ? ids : [];
}
// the drawn boxes of a component: every box whose entry key is one of the component's blocks
function boxesOfComponent(c) {
  const keys = new Set((c && c.blocks) || []);
  return [...document.querySelectorAll("#diagram .dg-node")].filter(g => keys.has(docKeyFor(g.id.replace(/^nd-/, ""))));
}
function componentsSourceText() {
  if (!COMPONENTS) return "This page holds no view of the component manifest: it was built before the manifest "
    + "existed, or the manifest could not be read when it was built.";
  return (componentsLive ? "Read from the development dashboard " : "Read when this page was built, ")
    + (COMPONENTS.read_at_text || "at a time that is not recorded")
    + (componentsLive ? "." : "; the page reads it again when the development dashboard serves it.");
}
// a repair's switch sentence ("--x (on) is OFF in the active run R."), else each launch flag's value in the active run,
// the run named once
function switchText(c) {
  const s = c.switch || {};
  if (s.sentence) return s.sentence;
  const flags = s.flags || [];
  if (!flags.length) return "no launch flag switches it";
  if (!s.run) return flags.map(f => f.flag + ": " + (f.text || "not read")).join("; ");
  return "In the active run " + s.run + ": " + flags.map(f => f.flag + " " + (f.present === false ? "absent (its default applies)"
    : f.value === true ? "on" : f.value === false ? "off" : f.value === null || f.value === undefined ? "not set"
    : String(f.value))).join(", ") + ".";
}
function listOrNone(items, none) {
  if (!items.length) return none;
  const ul = el("ul", { class: "comp-list" });
  for (const t of items) ul.appendChild(el("li", { text: t }));
  return ul;
}
function componentCard(c) {
  const card = el("div", { class: "comp-card" + (c.qa_stale ? " stale" : ""), "data-component": c.id });
  const head = el("div", { class: "comp-head" },
    el("code", { class: "comp-id", text: c.id }), el("span", { class: "comp-name", text: c.name || "" }));
  if (c.qa_stale) head.appendChild(el("span", { class: "qa-stale", text: "QA STALE" }));
  // the orange mark's chip: the repair state's own word (workMarks, the rule the rings and the list use)
  const worked = workMarkOf(c.id);
  if (worked) head.appendChild(stateChip(worked));
  card.appendChild(head);
  if (c.does) card.appendChild(el("p", { class: "comp-does", text: c.does }));
  const boxes = (c.blocks || []).map(k => (DOC[k] && DOC[k].t) || k);
  const rows = [
    ["switch state", switchText(c)],
    ["last code change", c.last_code_change_text || "not read for this page"],
    ["last independent QA", c.last_qa_text || (COMPONENTS || {}).no_qa_text || "no independent QA on record"],
  ];
  if (c.qa_stale) rows.push(["QA STALE", String(c.qa_stale_text || "its mapped code changed after the reviewed commit")
    .replace(/^QA STALE:\s*/, "")]);
  rows.push(["open defects", listOrNone((c.open_defects || []).map(d => d.id + " (" + [d.severity, d.repair_state || d.state]
    .filter(Boolean).join(", ") + "): " + (d.age_text || "its age is not on record")), "none in the defect register")]);
  rows.push(["linked tests not done", listOrNone((c.linked_tests || []).map(t => t.id + ": " + (t.section || "section not read")
    + (t.outcome ? ", last run " + t.outcome : "")), "none in the test registry")]);
  rows.push(["investigation stages", listOrNone(stagesOfComponent(c.id), INVESTIGATIONS
    ? "no stage of a root-cause investigation names it" : investigationsSourceText())]);
  rows.push(["drawn in", boxes.length ? boxes.join("; ") : "no box of the model diagram"]);
  const conf = c.confirmed || {};
  // the row's title names the check, not its result: on 51 of 81 cards of Oct 7, 2026 the value reads "carried over from
  // the explainer's text, not checked against the code" (the coordinator's change of that day)
  rows.push(["check of this entry against the code", (conf.at ? conf.at + (conf.commit ? ", commit " + conf.commit : "") + ": " : "")
    + (conf.basis || "not recorded")]);
  card.appendChild(kvGrid(rows));
  const problems = c.entry_fresh === false && c.entry_stale_text ? [c.entry_stale_text]
    : (((COMPONENTS || {}).check || {}).problems || []).filter(p => String(p).indexOf(c.id + ":") === 0);
  for (const p of problems) card.appendChild(el("p", { class: "comp-error", text: "Error: " + p }));
  return card;
}
function componentsSection(ids, heading) {
  const wrap = el("div", { class: "comp-section" });
  wrap.appendChild(el("h4", { text: heading }));
  if (COMPONENTS) {
    for (const id of ids) {
      const c = componentById(id);
      wrap.appendChild(c ? componentCard(c)
        : el("p", { class: "comp-error", text: "Error: " + id + " is not an id of the component manifest." }));
    }
  }
  wrap.appendChild(el("p", { class: "note", text: componentsSourceText() }));
  return wrap;
}
function componentsHeading(n) {
  return n === 1 ? "the component of the component manifest drawn in this box"
    : "the " + n + " components of the component manifest drawn in this box";
}
function markComponentBoxes(c) {
  document.querySelectorAll(".dg-node.comp-hl").forEach(n => n.classList.remove("comp-hl"));
  const boxes = c ? boxesOfComponent(c) : [];
  boxes.forEach(g => g.classList.add("comp-hl"));
  return boxes;
}
function showComponent(id) {
  docTrail = [{ key: "component." + id, title: id, component: id }];
  state.component = id; writeHash();
  renderDocPanel();
  const boxes = markComponentBoxes(componentById(id));
  if (boxes.length && boxes[0].scrollIntoView) boxes[0].scrollIntoView({ block: "center" });
}
function buildComponentIndex() {
  const arch = document.getElementById("tab-arch");
  if (!arch) return;
  const old = document.getElementById("components-index");
  const wasOpen = !!(old && old.open);
  if (old) old.remove();
  const comps = (COMPONENTS || {}).components || [];
  const box = el("details", { id: "components-index", class: "comp-index" });
  box.open = wasOpen;
  box.appendChild(el("summary", { text: "Every component of the component manifest (" + comps.length
    + "), with its code change, its independent QA and QA STALE: each opens its popup" }));
  const check = (COMPONENTS || {}).check || {};
  if (check.sentence) box.appendChild(el("p", { class: check.ok ? "note" : "comp-error", text: "The manifest check: " + check.sentence }));
  if (!check.ok && (check.problems || []).length) box.appendChild(listOrNone(check.problems.map(p => "Error: " + p), ""));
  for (const group of ["model part", "world", "repair", "tool"]) {
    const mine = comps.filter(c => c.kind === group);
    if (!mine.length) continue;
    box.appendChild(el("h4", { text: group + " (" + mine.length + ")" }));
    const list = el("div", { class: "comp-index-list" });
    for (const c of mine) {
      const b = el("button", { class: "comp-index-item" + (c.qa_stale ? " stale" : ""), "data-component": c.id,
        onclick: () => showComponent(c.id) }, el("code", { text: c.id }), " " + (c.name || ""));
      if (c.qa_stale) b.appendChild(el("span", { class: "qa-stale", text: "QA STALE" }));
      const worked = workMarkOf(c.id);
      if (worked) b.appendChild(stateChip(worked));
      list.appendChild(b);
    }
    box.appendChild(list);
  }
  box.appendChild(el("p", { class: "note", text: componentsSourceText() }));
  arch.appendChild(box);
}
// ================================================================== investigations: process-of-elimination diagrams
// THE EXPLAINER VIEW OF THE ROOT-CAUSE INVESTIGATIONS (the style guide, "EVERY ROOT-CAUSE INVESTIGATION HAS A LIVE
// PROCESS-OF-ELIMINATION DIAGRAM", the operator's rule of Oct 5, 2026).
// INPUT     tools/elimination.py's view: the copy the build put into this page (data-investigations, with the time it
//           was read) and, when the development dashboard serves the page, GET /api/investigations, read when the page
//           loads and again when the panel is opened 60 seconds or more after the last read.
// OPERATION the panel under the drawing draws each investigation with tools/elimination_diagram.js (the build puts it
//           into this page), the renderer of the dashboard's Tests page: the same boxes from the same data. A box's
//           component link opens that component's popup here; the address #investigation=ID opens the panel at that
//           investigation; each component's card lists the stages that name it, with their verdicts.
// OUTPUT    the panel #investigations-index and the row "investigation stages" of each component card.
let INVESTIGATIONS = jload("data-investigations", null);
let investigationsLive = false, investigationsReadAt = 0;
function readInvestigations(after) {
  if (!/^https?:$/.test(location.protocol) || Date.now() - investigationsReadAt < 60000) return;
  investigationsReadAt = Date.now();
  fetch("/api/investigations", { cache: "no-store" })
    .then(r => (r.ok ? r.json() : null))
    .then(d => {
      if (d && Array.isArray(d.investigations)) {
        INVESTIGATIONS = d; investigationsLive = true;
        if (after) after();
      }
    })
    .catch(() => {});
}
function investigationsSourceText() {
  if (!INVESTIGATIONS) return "This page holds no view of the investigations: it was built before they existed, or "
    + "they could not be read when it was built.";
  return (investigationsLive ? "Read from the development dashboard " : "Read when this page was built, ")
    + (INVESTIGATIONS.read_at_text || "at a time that is not recorded")
    + (investigationsLive ? "." : "; the page reads them again when the development dashboard serves it.");
}
// the stages of every investigation that name a component, as the component's card lists them
function stagesOfComponent(id) {
  const out = [];
  for (const inv of (INVESTIGATIONS || {}).investigations || []) {
    if (inv.refused) continue;
    if (inv.input && inv.input.component === id) out.push(inv.title + ": its input, '" + inv.input.name + "'");
    for (const b of inv.stages || []) {
      if (b.component === id) out.push(inv.title + ": stage " + b.position + ", '" + b.name + "', reads " + b.verdict
        + (b.time_text ? " (" + b.time_text + ")" : ""));
    }
  }
  return out;
}
function buildInvestigationsIndex() {
  const arch = document.getElementById("tab-arch");
  if (!arch) return;
  const old = document.getElementById("investigations-index");
  const wasOpen = old ? old.open : !!state.investigation;
  if (old) old.remove();
  const list = (INVESTIGATIONS || {}).investigations || [];
  const box = el("details", { id: "investigations-index", class: "comp-index inv-index" });
  box.open = wasOpen;
  box.appendChild(el("summary", { text: "Root-cause investigations (" + list.length + "): each a live "
    + "process-of-elimination diagram computed from the test registry, the same boxes as on the dashboard's Tests page" }));
  box.addEventListener("toggle", () => { if (box.open) readInvestigations(redrawInvestigationViews); });
  if (window.EliminationDiagram) {
    const served = /^https?:$/.test(location.protocol);
    const opts = { href: id => "#t=arch&component=" + encodeURIComponent(id),
                   onComponent: (id, e) => { if (e) e.preventDefault(); showComponent(id); },
                   rowHref: served ? (() => "/#tests") : null,
                   sourceText: investigationsSourceText() };
    for (const inv of list) window.EliminationDiagram.render(box, inv, opts);
  } else {
    box.appendChild(el("p", { class: "comp-error", text: "Error: the diagram renderer (elimination_diagram.js) is not in this page." }));
  }
  if (!list.length) box.appendChild(el("p", { class: "note", text: investigationsSourceText() }));
  const comp = document.getElementById("components-index");
  if (comp) arch.insertBefore(box, comp); else arch.appendChild(box);
}
function showInvestigation(id) {
  state.investigation = id; writeHash();
  const box = document.getElementById("investigations-index");
  if (!box) return;
  box.open = true;
  const sec = document.getElementById("investigation-" + id);
  if (sec && sec.scrollIntoView) sec.scrollIntoView({ block: "start" });
}
function redrawInvestigationViews() {
  buildInvestigationsIndex();
  if (docTrail.length) renderDocPanel();
  if (state.investigation) {
    const sec = document.getElementById("investigation-" + state.investigation);
    const box = document.getElementById("investigations-index");
    if (box && sec) box.open = true;
  }
}

// a newer view arrived: what shows components is drawn again from it, the orange marks of the model diagram too
function redrawComponentViews() {
  WORK_KEPT = null;
  drawWorkRings();
  buildWorkList();
  buildComponentIndex();
  if (docTrail.length) renderDocPanel();
  const open = document.querySelector("#detail .comp-section");
  if (open && detailId) {
    const ids = componentIdsOfKey(detailKeyFor(detailId));
    open.replaceWith(componentsSection(ids, componentsHeading(ids.length)));
  }
  if (state.component) markComponentBoxes(componentById(state.component));
}

// ================================================================== the parts being worked on: orange marks
// THE ORANGE MARKS (the decided design of the Architecture tab, Oct 7, 2026, D-E2), computed, never typed.
// INPUT     the component manifest's view (COMPONENTS: each component's kind and blocks; its open defects, each with
//           this component's repair state in it and the defect's closing test {row, text}; its linked registry rows with
//           their section) and META.repair_switches (each repair's switch in THIS page's run, read from the run's
//           args.json when the page was built; the view's own switch, which describes the active run, stands in when
//           the page has none and the active run is this page's run).
// OPERATION per component, one of:
//           solid   a repair whose open defects carry IN TRAINING and whose switch is on in this page's run, or a
//                   component drawn in one box only that has a linked registry row in section RUNNING;
//           dashed  a repair whose open defects carry BUILT/OFF, REVIEWED or QUALIFIED/OFF (built, not switched on);
//           listed  a repair whose state and switch disagree: FAILED/SUPERSEDED but switched on in this page's run, or
//                   IN TRAINING but switched off in it - a line in the list and no ring.
//           A row in section QUEUED marks nothing. A box drawing components of both kinds takes the solid ring.
// OUTPUT    workMarks(): [{c, mark ("solid", "dashed", or null for a listed item), word (the chip's word: the repair
//           state that made the mark, or RUNNING for a part that is not a repair), states, boxes, sw, running}], read by
//           the rings (drawWorkRings), the list under the model diagram (buildWorkList) and the chips of the component
//           list and the component cards (stateChip): one function, so the three cannot disagree.
const WORK_DASHED = ["BUILT/OFF", "REVIEWED", "QUALIFIED/OFF"];
let WORK_KEPT = null;
function drawnSwitch(c) {
  const own = (META.repair_switches || {})[c.id];
  if (own && typeof own.on === "boolean") return own;
  const s = c.switch || {}, f = (s.flags || [])[0] || {};
  if (s.run && s.run === META.run && typeof s.repair_on === "boolean")
    return { on: s.repair_on, flag: f.flag, value: f.value, present: f.present !== false };
  return null;
}
function workMarks() {
  if (WORK_KEPT && WORK_KEPT.view === COMPONENTS) return WORK_KEPT.items;
  const out = [];
  for (const c of (COMPONENTS || {}).components || []) {
    if (!c || !c.id) continue;
    const states = [...new Set((c.open_defects || []).map(d => d && d.repair_state).filter(Boolean))];
    const repair = c.kind === "repair";
    const sw = repair ? drawnSwitch(c) : null;
    const running = (c.linked_tests || []).filter(t => t && t.section === "RUNNING");
    const boxes = boxesOfComponent(c);
    const built = states.find(s => WORK_DASHED.includes(s));
    let mark = null, word = null;
    if (repair && sw && sw.on && states.includes("IN TRAINING")) { mark = "solid"; word = "IN TRAINING"; }
    else if (running.length && boxes.length === 1) { mark = "solid"; word = repair && states.length ? (built || states[0]) : "RUNNING"; }
    else if (repair && built) { mark = "dashed"; word = built; }
    else if (repair && sw && sw.on && states.includes("FAILED/SUPERSEDED")) word = "FAILED/SUPERSEDED";
    else if (repair && sw && !sw.on && states.includes("IN TRAINING")) word = "IN TRAINING";
    if (word) out.push({ c, mark, word, states, boxes, sw, running });
  }
  // kept for this view only after the drawing exists: the boxes are read from it
  if (document.querySelector("#diagram svg")) WORK_KEPT = { view: COMPONENTS, items: out };
  return out;
}
function workMarkOf(id) {
  return workMarks().find(it => it.c.id === id) || null;
}
// the chip of a marked component: the state's own word, solid orange, dashed orange, or grey for a listed item
function stateChip(it) {
  return el("span", { class: "state-chip " + (it.mark || "plain"), text: it.word,
    title: it.mark === "solid" ? "solid orange ring on its boxes: being tested now"
      : it.mark === "dashed" ? "dashed orange ring on its boxes: built, not switched on"
      : "listed under the model diagram without a ring: its state and its switch in this page's run disagree" });
}
// the rings, in their own layer drawn right after the drawing's defs, so every box and line is drawn over them
function drawWorkRings() {
  const svg = document.querySelector("#diagram svg");
  if (!svg) return;
  svg.querySelectorAll("g.dg-rings").forEach(n => n.remove());
  const best = {};
  for (const it of workMarks()) {
    if (!it.mark) continue;
    for (const g of it.boxes) {
      const id = g.id.replace(/^nd-/, "");
      if (best[id] !== "solid") best[id] = it.mark;
    }
  }
  const layer = el("g", { class: "dg-rings" });
  for (const [id, mark] of Object.entries(best)) {
    const g = document.getElementById("nd-" + id), r = g && g.querySelector("rect");
    if (!r) continue;
    const x = Number(r.getAttribute("x")), y = Number(r.getAttribute("y"));
    const w = Number(r.getAttribute("width")), h = Number(r.getAttribute("height"));
    layer.appendChild(el("rect", { class: "dg-ring " + mark, "data-for": id, rx: 6,
                                   x: x - 3, y: y - 3, width: w + 6, height: h + 6 }));
  }
  const defs = svg.querySelector("defs");
  svg.insertBefore(layer, defs ? defs.nextSibling : svg.firstChild);
}
// THE SECTION 11 RULE FOR A TEXT THE PAGE PRINTS FROM A RECORD (DESIGN.md section 11; tools/check_no_bare_decimals.py
// bare_decimals, ported): a number below 1 written with a decimal point and no unit after it ("0.25 or more"), or a
// number in exponent form, is a bare score the page does not print; a launch setting as the command line writes it
// ("--babble-noise 0.05") is excused, and so is a commit's short hash. The defect register's closing texts are typed by
// people and its own page check (defect_register.PAGE_FIELDS) does not read them, so the list under the model diagram
// tests each one here. The "before" sides of the Python patterns are checked in code: the suite parses this file with
// esprima, and no regular expression here uses a lookbehind.
const S11_FLAG = /--[A-Za-z][\w-]*(?:[ \t=][^\s,;()]+)?/g;
// The unit after a number: tools/check_no_bare_decimals.py UNIT_AFTER_STRICT, ported, each unit a whole word ("per" no
// longer excuses "0.00001 perturbation"; "percent" and "per cent" are named). The only unit list here, switched on with
// the Python rule on Oct 15, 2026 (the decision explainer-part2-rule-confirmed-20261015); the list used until then matched the start of any word.
const S11_UNIT_STRICT = /^\s*(%|°|per ?cent\b|cm\b|mm\b|m\b|km\b|s\b|ms\b|secs?\b|seconds?\b|minutes?\b|min\b|hours?\b|h\b|days?\b|GB\b|GiB\b|MB\b|MiB\b|KB\b|kB\b|kg\b|g\b|N\b|W\b|Hz\b|kPa\b|Pa\b|steps?\b|model steps\b|frames?\b|per\b|x\b|times\b|fps\b|bytes\b|degrees?\b)/i;
function hasBareDecimal(text) {
  const unit = S11_UNIT_STRICT;
  let t = String(text || "").replace(S11_FLAG, " ");
  const sci = [];
  for (const m of t.matchAll(/-?\d+(?:\.\d+)?[eE][-+]?\d+/g)) {
    const end = m.index + m[0].length;
    if ((m.index && /[\w.]/.test(t[m.index - 1])) || /^[\w.]/.test(t.slice(end))) continue;
    if (m[0].length >= 7 && /^\d+[eE]\d+$/.test(m[0])) continue;      // a commit's short hash
    sci.push([m.index, end]);
  }
  if (sci.length) return true;
  for (const m of t.matchAll(/-?\d*\.\d+/g)) {
    const end = m.index + m[0].length;
    if (m.index && /[\w.\-\/\\#:]/.test(t[m.index - 1])) continue;
    if (/^([\w\/\\]|\.\d)/.test(t.slice(end))) continue;
    if (Math.abs(parseFloat(m[0])) >= 1 || unit.test(t.slice(end, end + 14))) continue;
    return true;
  }
  return false;
}
// THE DEFECT REGISTER'S SENTENCES AS THE PAGE PRINTS THEM (the independent review of Oct 15, 2026, finding M2;
// explainer-formatting-and-current-run-20261015: "Withholding a sentence must not conceal a failure, decision,
// comparison direction or essential statistical definition"). A closing text or a repair's note is split into
// sentences at a full stop and white space, as tools/elimination.py plain_text splits a registry text; a sentence that
// the table WORK_PLAIN_REWRITES holds a rewrite for (that defect, field and exact sentence) is replaced by its rewrite
// whether or not the rule above flags it (the review of Oct 15, 2026, second recheck: "the 0.00001 perturbation"
// passed the unit list of that time, which matched the start of a word, "per" in "perturbation"); any other sentence
// that prints no bare decimal (hasBareDecimal) is printed as written, and one that does is withheld; a notice counts
// each and names the defect register as the source of the original. A sentence edited in the register no longer
// matches its key: it is printed as written when the rule passes it and withheld, with the notice, when it does not
// (tests/test_build_explainer.py test_every_plain_rewrite_key_is_still_a_sentence_of_the_live_register fails on such
// a key). The rewrites' numbers and scales are read from the rows the sentences name:
//   T1, head tilt read from the vestibular token (registry row vestibular_input_path_production_segment, measures):
//     the trainer's probe vestibular.repair.balance_probe.r2.head_tilt, R squared of a ridge read-out of each body's
//     change of head tilt from its token change, 0 = no better than predicting no change, 1 = every scored change
//     predicted; line 0.25 -> 25 in 100;
//   the whole-model check (row head_tilt_repair_whole_model_check_481800, result): head tilt on the 500 unread model
//     steps at 0.44 to 0.50 at the 9 readings from update 2,000 on, the same R squared -> 44 to 50 in 100; verdict
//     fail, on vision's line P2 only;
//   the seed-0 rerun (row head_tilt_repair_on_seed0_rerun_481800, result readings): the median over 11 readings of
//     ln(this arm's vision error ratio / the control arm's), 0 = the same error as without the switch; 0.0089 against
//     the line 0.0342 (read set) and 0.0069 against 0.0195 (unread set), each line the largest such ln ratio between
//     the two runs without the switch -> 89, 342, 69 and 195 in 10,000, the same values on the same quantity; the
//     anchor printed beside them is arithmetic on that definition: an ln ratio of 0.10 (1,000 in 10,000) is an error
//     ratio of e to the 0.10 = 1.105, an error 10.5 in 100 above the arm without the switch;
//   the planner search's perturbation (PLANNER_SEARCH_UNSTABLE's closing test; registry row
//     planner_weights_review_search_noise_479350, inputs): "per draw and per sense 192 Gaussian numbers of standard
//     deviation 0.00001" added to the five senses' present tokens handed to the search, "the vision token differs by
//     up to 0.00001 between two drawings of one scene on this machine's graphics driver" -> 1 in 100,000; the tokens'
//     own size: the shared LayerNorm (lem/model/encoder_registry.py output_norm, elementwise_affine False by default)
//     leaves each raw token at a root mean square of 1 per number, and the top-down gate blends at most half of the
//     core's prediction into it.
// Every rewrite keeps the original's values (only the unit of the scale moves: 0.25 of 1 is 25 in 100) and its
// meaning; the quantity and its two ends come from the rows named, never typed from memory.
const WORK_PLAIN_REWRITES = {
  ["VESTIBULAR_TOKEN_NO_HEAD_TILT|closing test|With the path on in production, the trainer's own probe reads each "
   + "body's change of head tilt from the vestibular token at 0.25 or more in every window of 1,000 model steps from "
   + "2,001 model steps after the switch, with no stop line crossed through 10,000 model steps."]:
    "With the path on in production, the trainer's own probe reads each body's change of head tilt from the "
    + "vestibular token at 25 in 100 or more (R squared of the probe's read-out: 0 = no better than predicting no "
    + "change, 100 = every scored change predicted) in every window of 1,000 model steps from 2,001 model steps after "
    + "the switch, with no stop line crossed through 10,000 model steps.",
  ["VESTIBULAR_TOKEN_NO_HEAD_TILT|HEAD_TILT_REPAIR|Its whole-model check ran on checkpoint 481,800 (Oct 5): head tilt "
   + "in the token at 0.44 to 0.50 (line 0.25), vision's cost above the seed line."]:
    "Its whole-model check ran on checkpoint 481,800 (Oct 5): head tilt read from the token at 44 to 50 in 100 (R "
    + "squared: 0 = no better than predicting no change, 100 = every scored change predicted; line 25 in 100), and "
    + "vision's cost above the seed line, so that check's verdict was fail (registry row "
    + "head_tilt_repair_whole_model_check_481800).",
  ["VESTIBULAR_TOKEN_NO_HEAD_TILT|HEAD_TILT_REPAIR|Two further runs of on did not repeat the vision cost: seed 1 (Oct "
   + "5) and seed 0 with the first run's replay draws (Oct 8, PASS: median ln ratio 0.0089 against 0.0342 read, 0.0069 "
   + "against 0.0195 unread)."]:
    "Two further runs of on did not repeat the vision cost: seed 1 (Oct 5) and seed 0 with the first run's replay "
    + "draws (Oct 8, PASS: the median of ln(vision's error ratio in the arm with the switch / in the arm without it), "
    + "where 0 = the same error and 1,000 in 10,000 = an error 10.5 in 100 above the arm without the switch, at 89 in "
    + "10,000 against the line 342 in 10,000 on the read set and 69 in 10,000 against 195 in 10,000 on the unread set; "
    + "each line is the largest such ln ratio between two runs without the switch).",
  ["PLANNER_SEARCH_UNSTABLE|closing test|The search selects nearly the same command from nearly the same input "
   + "(same-state repeatability and the 0.00001 perturbation), and the sense weights are then tested again with a "
   + "paired gate (checklist item 4g)."]:
    "The search selects nearly the same command from nearly the same input (same-state repeatability, and the "
    + "perturbation of registry row planner_weights_review_search_noise_479350: Gaussian noise of standard deviation 1 "
    + "in 100,000 added to each of the 192 numbers of each of the five senses' present tokens, tokens the shared "
    + "LayerNorm leaves at a root mean square of 1 per number before the top-down gate; that noise equals the largest "
    + "difference in the vision token between two drawings of one scene on this machine's graphics driver), and the "
    + "sense weights are then tested again with a paired gate (checklist item 4g).",
};
// sentences end at a full stop followed by white space, outside parentheses and quotation marks: a quoted sentence or
// a parenthesis with a full stop inside stays whole with its sentence, and every character is kept
function registerSentences(text) {
  const s = String(text), out = [];
  let depth = 0, quote = null, start = 0;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (quote) { if ((quote === "\"" && ch === "\"") || (quote === "“" && ch === "”")) quote = null; continue; }
    if (ch === "\"" || ch === "“") { quote = ch; continue; }
    if (ch === "(") depth++;
    else if (ch === ")") depth = Math.max(0, depth - 1);
    else if (ch === "." && depth === 0 && i + 1 < s.length && /\s/.test(s[i + 1])) {
      out.push(s.slice(start, i + 1).trim());
      start = i + 1;
    }
  }
  if (s.slice(start).trim()) out.push(s.slice(start).trim());
  return out;
}
function registerText(text, key, where) {
  const kept = [];
  let rewritten = 0, withheld = 0;
  for (const sentence of registerSentences(text)) {
    const plain = WORK_PLAIN_REWRITES[key + "|" + sentence.split(/\s+/).join(" ").trim()];
    if (typeof plain === "string" && !hasBareDecimal(plain)) { kept.push(plain); rewritten++; }
    else if (!hasBareDecimal(sentence)) kept.push(sentence);
    else withheld++;
  }
  const n = k => (k === 1 ? "1 sentence" : k + " sentences");
  if (rewritten) kept.push("(" + n(rewritten) + " rewritten here in plain numbers on Oct 15, 2026; the defect register, "
    + where + ", holds the original wording.)");
  if (withheld) kept.push("(" + n(withheld) + " withheld here: " + (withheld === 1 ? "it prints" : "they print")
    + " a score as a bare decimal, which DESIGN.md section 11 bars from pages; the defect register, " + where
    + ", holds the full text.)");
  return kept.join(" ");
}
// one line per marked or listed component: NAME: STATE, its switch in this page's run, the boxes it is drawn in, and
// what ends it - each open defect's closing test as the defect register types it
function closingLine(d, word) {
  const ct = d.closing_test;
  let said;
  if (!ct || typeof ct !== "object") said = "closing test not in the view this page read (the dashboard's view carries "
    + "it once the dashboard has restarted on this code)";
  else if (!ct.row && !ct.text) said = "closing test not written";
  else {
    // sentence by sentence (registerText): a sentence holding a bare decimal is rewritten or withheld, with its count
    const text = !ct.text ? "" : registerText(ct.text, d.id + "|closing test",
                                              "docs/gain_of_function_v4/defect_register.json, " + d.id + "'s closing test");
    said = "closing test: " + (ct.row ? "registry row " + ct.row : "registry row not written yet") + (text ? ". " + text : "");
  }
  const other = d.repair_state && d.repair_state !== word ? " (this repair's state in that defect: " + d.repair_state + ")" : "";
  return d.id + other + ": " + said;
}
function switchFact(it) {
  const sw = it.sw;
  if (it.c.kind !== "repair") return null;
  if (!sw) return "its switch in this page's run was not read";
  const flag = sw.flag || "its switch";
  const cap = flag === "--attention-cap" && sw.on && ARGS.attention_cap_bound !== null && ARGS.attention_cap_bound !== undefined
    ? ", bound " + ARGS.attention_cap_bound : "";
  return (sw.on ? "switched on" : "switched off") + " in this page's run, " + META.run + " (" + flag + " "
    + (sw.present === false ? "absent" : sw.value === true ? "on" : sw.value === false ? "off" : String(sw.value)) + cap + ")";
}
function workLine(it) {
  const c = it.c;
  const li = el("li", { "data-component": c.id, "data-mark": it.mark || "listed" });
  li.appendChild(el("button", { class: "worked-name", title: "open its card and outline its boxes in red",
                                onclick: () => showComponent(c.id) }, el("code", { text: c.id })));
  const head = [];
  head.push(c.kind === "repair" && it.states.length ? ": " + it.word : "");
  const sw = switchFact(it);
  if (sw) head.push(" · " + sw);
  const boxes = (c.blocks || []).map(k => (DOC[k] && DOC[k].t) || k);
  head.push(" · drawn in: " + (boxes.length ? boxes.join("; ") : "no box of the model diagram"));
  for (const t of it.running) head.push(" · registry row " + t.id + " is RUNNING" + (t.title ? " (" + t.title + ")" : ""));
  li.appendChild(document.createTextNode(head.join("")));
  const ends = (c.open_defects || []).filter(d => d && d.repair_state);
  if (ends.length) {
    const ul = el("ul", { class: "worked-ends" });
    for (const d of ends) ul.appendChild(defectItem(d, it.word, c.id));
    li.appendChild(ul);
  }
  return li;
}
// one open defect of a listed component: its closing test, then the defect register's note on this repair in that
// defect (what its state rests on, such as "Purpose not met."): a note of up to 240 characters in the line, a longer
// one folded under the line; a sentence of the note holding a score written as a bare decimal is rewritten or
// withheld alone, with its count, like a closing text's (registerText), so the rest of the note stays readable
function defectItem(d, word, cid) {
  const li = el("li", { text: closingLine(d, word) });
  const note = typeof d.repair_note === "string" ? d.repair_note.trim() : "";
  if (note) {
    const said = registerText(note, d.id + "|" + cid, "docs/gain_of_function_v4/defect_register.json, " + d.id
                              + "'s note on " + cid);
    if (said.length <= 240) li.appendChild(document.createTextNode(" · the register's note on this repair: " + said));
    else li.appendChild(el("details", { class: "worked-note" },
      el("summary", { text: "the register's note on this repair" }), el("p", { text: said })));
  }
  return li;
}
function buildWorkList() {
  const panel = document.getElementById("model-diagram");
  if (!panel) return;
  let box = document.getElementById("worked-on");
  if (!box) { box = el("div", { id: "worked-on" }); panel.appendChild(box); }
  box.textContent = "";
  box.appendChild(el("h4", { text: "Orange: the parts being worked on" }));
  box.appendChild(el("p", { class: "worked-rule" },
    el("span", { class: "ring-key" }),
    "solid ring: a repair drawn in the box is at IN TRAINING and switched on in this page's run, or a test of a part "
    + "drawn in that box alone is RUNNING. ",
    el("span", { class: "ring-key dashed" }),
    "dashed ring: a repair drawn in the box is at BUILT/OFF, REVIEWED or QUALIFIED/OFF (built, not switched on). A test "
    + "that is QUEUED marks nothing; a repair whose state and switch disagree is listed without a ring. Each line opens "
    + "its card and outlines its boxes. Computed from the defect register's repair states and closing tests, each "
    + "repair's launch setting in this run's args.json and the test registry's sections. " + componentsSourceText()));
  if (!COMPONENTS) return;
  const items = workMarks();
  if (!items.length) {
    box.appendChild(el("p", { class: "note", text: "Nothing is marked: no repair drawn in a box is at IN TRAINING, "
      + "BUILT/OFF, REVIEWED or QUALIFIED/OFF, and no test of a part drawn in one box is RUNNING." }));
    return;
  }
  const ul = el("ul", { class: "worked-list" });
  for (const it of items) ul.appendChild(workLine(it));
  box.appendChild(ul);
}
// THE MODEL DIAGRAM'S HEADING (D-E1): the drawing's one name, printed over its tour and the drawing
function buildModelHead() {
  const panel = document.getElementById("model-diagram");
  if (!panel || panel.querySelector("h3")) return;
  const h = el("h3", {}, "The model diagram: what runs in one model step, from the senses to the command");
  h.appendChild(docBtn("diagram.numbered_chain", "the numbered chain, and what it leaves out"));
  panel.insertBefore(h, panel.firstChild);
}

// THE TOUR OF THE MODEL DIAGRAM (the decided design of the Architecture tab, Oct 7, 2026, D-E1 and D-E3). It sits in
// the model diagram's own panel, between the panel's heading and the drawing. Its items are slides ("slide 3 of 28"),
// counted from the tour's own length (walkthrough.py builds a different number of slides for different launch flags).
// A slide fills its boxes violet, draws a violet outline over each of them as an element of its own (rect.dg-tour-mark
// in the layer g.dg-tour, so the bold blue outline of a box a lesson reaches cannot hide it) and animates its own lines.
// "end the tour" clears the fill, the outlines, the moving lines and the address's tr= part.
let tourIdx = state.tr ? parseInt(state.tr, 10) : -1;
if (!(tourIdx >= 0 && tourIdx < TOUR.length)) tourIdx = -1;
function endTour() {
  tourIdx = -1;
  delete state.tr; writeHash();
  renderTour();
}
function renderTour() {
  const box = document.getElementById("tour");
  box.textContent = "";
  document.querySelectorAll(".dg-node.hl").forEach(n => n.classList.remove("hl"));
  document.querySelectorAll("#diagram g.dg-tour").forEach(n => n.remove());
  const started = tourIdx >= 0 && !!TOUR[tourIdx];
  const bar = el("div", { class: "tour-bar" });
  if (started)
    bar.appendChild(el("button", { text: "◀ previous", onclick: () => { tourIdx = Math.max(0, tourIdx - 1); renderTour(); } }));
  bar.appendChild(el("button", { text: started ? "next ▶" : "start the tour ▶",
                                 onclick: () => { tourIdx = Math.min(TOUR.length - 1, tourIdx + 1); renderTour(); } }));
  if (started) bar.appendChild(el("button", { text: "end the tour", onclick: endTour }));
  bar.appendChild(el("span", { class: "tour-pos", text: started
    ? `slide ${tourIdx + 1} of ${TOUR.length} · the tour of the model diagram`
    : `the tour of the model diagram: ${TOUR.length} slides, each filling its boxes violet and moving its lines`
      + " · click any box for its live numbers" }));
  box.appendChild(bar);
  document.querySelectorAll(".dg-edge.flow").forEach(e => e.classList.remove("flow"));
  if (started) {
    const st = TOUR[tourIdx];
    box.appendChild(el("div", { class: "tour-body" },
      el("span", { class: "tour-title", text: st.title }), el("br"), st.body));
    const hl = new Set(st.nodes || []);
    const undrawn = [];
    const svg = document.querySelector("#diagram svg");
    const layer = svg ? el("g", { class: "dg-tour" }) : null;
    for (const nid of hl) {
      const n = document.getElementById("nd-" + nid);
      if (n) {
        n.classList.add("hl");
        const r = n.querySelector("rect");
        if (layer && r) layer.appendChild(el("rect", { class: "dg-tour-mark", "data-for": nid, rx: 4,
          x: r.getAttribute("x"), y: r.getAttribute("y"), width: r.getAttribute("width"), height: r.getAttribute("height") }));
      }
      else undrawn.push(nid);
    }
    if (layer) svg.appendChild(layer);
    // a slide that names a box this run's picture does not have says so, where it used
    // to highlight nothing in silence
    if (undrawn.length)
      box.appendChild(el("div", { class: "warn", text:
        "This slide names a box that is not drawn for this run: " + undrawn.join(", ") + "." }));
    // animate EXACTLY the lines this slide declares. The old rule - light every
    // line touching a highlighted box - marched side paths (memory->assembly,
    // planner->motor) through slides they had no part in, while the slide's own
    // path went dark wherever it crossed an unhighlighted box (2026-09-14).
    const flows = st.edges ? new Set(st.edges.map(p => p[0] + ">" + p[1])) : null;
    document.querySelectorAll(".dg-edge").forEach(e => {
      if (flows ? flows.has(e.dataset.a + ">" + e.dataset.b)
                : (hl.has(e.dataset.a) || hl.has(e.dataset.b))) e.classList.add("flow");
    });
    state.tr = String(tourIdx); writeHash();
  }
}

// ================================================================== TAB 2: inspector
function snapByStep(stepStr) {
  if (!SNAPS.length) return null;
  const want = stepStr ? parseInt(stepStr, 10) : SNAPS[SNAPS.length - 1].header.step;
  let best = SNAPS[SNAPS.length - 1], bd = Infinity;
  for (const s of SNAPS) { const d = Math.abs(s.header.step - want); if (d < bd) { bd = d; best = s; } }
  return best;
}
function panel(title, docKey, caption, ...kids) {
  const h = el("h3", {}, title);
  if (docKey) h.appendChild(docBtn(docKey, title));
  const p = el("div", { class: "panel" }, h);
  if (caption) p.appendChild(el("p", { class: "caption", text: caption }));
  for (const k of kids) if (k != null) p.appendChild(k);
  return p;
}
// a value may be a string or a drawn element (a picture of a raw internal, DESIGN.md section 11)
function kvGrid(pairs) {
  const g = el("div", { class: "kv" });
  for (const [k, v] of pairs) {
    g.appendChild(el("b", { text: k }));
    g.appendChild(v instanceof Node ? el("span", {}, v) : el("span", { text: v == null ? "—" : String(v) }));
  }
  return g;
}
// THE TACTILE ROW'S COLUMNS, READ FROM THE LEGEND capture.py WRITES BESIDE THE ROWS
// (senses.touch_cloud_doc). Each comma-separated entry of that legend is one column,
// except an entry ending in "xN" - "vib x4", "push direction x3" - which is N columns.
// The push columns have ridden every tactile token since 2026-09-21 and this panel drew
// none of them, because the column numbers were typed into the drawing code and the
// legend was read by nothing. Reading it is what keeps the two in step.
function touchLegend(doc) {
  if (typeof doc !== "string") return [];
  const out = []; let col = 0;
  for (const raw of doc.replace(/^\s*\[/, "").replace(/\]\s*$/, "").split(",")) {
    const label = raw.trim();
    if (!label) continue;
    const m = label.match(/\bx(\d+)$/);
    const span = m ? parseInt(m[1], 10) : 1;
    out.push({ label: label, start: col, span: span });
    col += span;
  }
  return out;
}
// the first column of the entry whose label matches, or the fallback for a snapshot
// written before the legend carried that entry
function touchCol(legend, re, fallback) {
  for (const e of legend) if (re.test(e.label)) return e.start;
  return fallback;
}
// what the stored number must be MULTIPLIED BY to read in the legend's own unit: the
// legend writes the divisor at the end of the label ("depth cm/10", "arrival speed
// cm/s/500"), so the page never carries a second copy of it
function touchScale(legend, re, fallback) {
  for (const e of legend) if (re.test(e.label)) {
    const m = e.label.match(/\/(\d+(?:\.\d+)?)$/);
    if (m) return parseFloat(m[1]);
  }
  return fallback;
}
// The hierarchical self's rows, grouped. Every label says "slot" - the code's own word
// for one line of this part's table - and names both ends of its scale.
// The two ends of the body map's co-movement scale, from the map's own sigmoid(5 x (score - 0.5)):
// a score of 0 reads the first, a score of 1 the second. (The function names keep the word the page
// used until 2026-10-03, "belonging"; the page's word is co-movement.)
const INCORP_FLOOR = 1 / (1 + Math.exp(2.5)), INCORP_TOP = 1 / (1 + Math.exp(-2.5));
// A CO-MOVEMENT NUMBER AS A PERCENTAGE OF THE WAY BETWEEN THE MAP'S OWN TWO ENDS (DESIGN.md
// section 11): 0% = a score of 0, 100% = a score of 1
const belong = x => wayBetween(x, INCORP_FLOOR, INCORP_TOP, "a score of 0", "a score of 1");
const belongPct = x => { const v = num(x); return v === null ? "—" : pctNum(100 * (v - INCORP_FLOOR) / (INCORP_TOP - INCORP_FLOOR)); };
// the certainty row: the average distance from the middle, 0 at the middle and 2 x (top - 0.5) at an end
const certainty = x => wayBetween(x, 0, 2 * (INCORP_TOP - 0.5), "every handle on the middle", "every handle at one end");
// an outside thing's match, control and earned scores: 0 = no match, 1 = carried exactly with the hand
const carried = x => { const v = num(x); return v === null ? "—" : pctNum(100 * v); };
// One row per outside thing: what it scored, what the same score reads with the hand
// moving the opposite way, the difference, the belonging number that difference becomes,
// and whether the world lets the body move the thing at all.
function incorporationTable(inc, cg) {
  const wrap = el("div", {});
  wrap.appendChild(kvGrid([
    ["handles: the mean co-movement number of the body's handles",
      belong(inc.handles_mine) + " (a score of 0 = no handle's command change ever fell in the same "
      + "model step as a proprioception change; a score of 1 = the two change sizes matched on every "
      + "model step; co-movement, not proof that the command caused the change)"],
    ["extent: the mean co-movement number of EVERYTHING the map watches",
      belong(inc.extent) + " (the body's handles and the outside things together)"],
    ["certainty: how far the handles' numbers sit from the middle of their range",
      certainty(inc.certainty) + " - a number that RISES as the handles' co-movement numbers fall"],
    ["which of extent and certainty the coherence term of wellness carries",
      inc.carried_by_wellness ? inc.carried_by_wellness + " (the --wellness-coherence flag)"
                              : "not recorded by this run's capture"]]));
  const objs = inc.objects || {};
  const names = Object.keys(objs);
  if (!names.length) {
    wrap.appendChild(el("p", { class: "note", text:
      "This run's map watches no outside thing, so there is nothing here but the body." }));
    return wrap;
  }
  const ev = inc.objects_evidence || {}, phys = inc.objects_physics_driven || {};
  // the caregiver's own report says whether the toy's "yes" is the hand-off
  const cgRep = (typeof cg === "object" && cg) ? cg : {};
  const follows = !!cgRep.object_follows_grip, held = cgRep.object_held_by_infant;
  const haveEv = Object.keys(ev).length > 0, havePhys = Object.keys(phys).length > 0;
  const head = ["the thing", "its co-movement number (0% = never travelled with a hand, 100% = travelled exactly with a hand)"];
  if (haveEv) head.push("what it scored (0% = no match, 100% = carried exactly with the hand)",
                        "the same score with the hand moving the OPPOSITE way",
                        "the difference, which is what counts");
  if (havePhys) head.push("can the body move it at all?");
  const rows = [head];
  for (const nm of names) {
    const r = [nm, belongPct(objs[nm])];
    if (haveEv) {
      const e = ev[nm] || {};
      r.push(carried(e.match), carried(e.control), carried(e.earned));
    }
    // three different truths: the mobile swings on a joint; the toy is a mocap body that a
    // gripping hand carries only under the toy hand-off; any other mocap thing never moves
    if (havePhys) r.push(!phys[nm]
      ? "NO - the world writes its place from outside every step and physics never moves "
        + "it, so the infant's hand cannot displace it at all"
      : (nm === "toy" && follows
         ? "only by gripping it - it is still a mocap body, but under the toy hand-off it goes "
           + "where a hand that grips it for 0.1 s goes"
           + (held === undefined ? "" : (held ? " (held by the infant now)" : " (not held now)"))
         : "yes - it moves under physics, so a push can displace it"));
    rows.push(r);
  }
  const tb = el("table", { class: "kvtable" });
  rows.forEach((r, ri) => {
    const tr = el("tr", {});
    r.forEach(c => tr.appendChild(c instanceof Node ? el(ri ? "td" : "th", {}, c) : el(ri ? "td" : "th", { text: c })));
    tb.appendChild(tr);
  });
  wrap.appendChild(tb);
  if (havePhys && names.some(nm => !phys[nm]))
    wrap.appendChild(el("p", { class: "warn", text:
      "The things marked NO are MuJoCo mocap bodies. MuJoCo writes a mocap body's place "
      + "from outside on every step and never integrates it, so no amount of reaching moves "
      + "one and its whole path is the adult's. A score for such a thing can only ever be "
      + "the adult happening to carry it the way a hand was going - never the infant "
      + "carrying it. The toy hand-off (--toy-hand-off follow) changes that for the toy: "
      + "a gripping hand carries it." }));
  else if (!havePhys)
    wrap.appendChild(el("p", { class: "note", text:
      "This snapshot does not record whether the world lets the body move each thing "
      + "(snapshots written before the capture carried it), so a score here cannot be told "
      + "apart from the adult carrying the thing the way a hand happened to go." }));
  if (!haveEv)
    wrap.appendChild(el("p", { class: "note", text:
      "This snapshot does not record each thing's own control - the same score with the "
      + "hand moving the opposite way - so a reading here cannot be told from its own noise." }));
  const per = inc.objects_per_body;
  if (isObj(per)) {
    const det = el("details", {}, el("summary", { text: "each thing, body by body (0% = never travelled with a hand, 100% = travelled exactly with a hand)" }));
    det.appendChild(kvGrid(Object.entries(per).map(([nm, v]) =>
      [nm, (Array.isArray(v) ? v : [v]).map(belongPct).join(", ")])));
    wrap.appendChild(det);
  }
  return wrap;
}

// One row per named thing, BOTH levels, and the reader's own line under them. The panel's
// blurb promises "the same two levels" and says a count of -1 means "cannot be measured
// yet"; until 2026-09-21 this table drew the minutes level only and printed no count at
// all, while every snapshot carried l3_ticks, l3_link, l3_rank, l3_line, l3_counted,
// l3_counted_names, l3_entity_windows and l3_steps_to_first_evidence - and the same five
// unread fields for the minutes level.
const ENTITY_LEVELS = [["l2", "minutes"], ["l3", "hours"]];
function entitySelfTable(es) {
  const wrap = el("div", {});
  const names = es.names || [], pres = es.presence_mean || [];
  const head = ["what it is", "how much of the time it is there (0% = never, 100% = always)"];
  const counted = {}, reachable = {};
  for (const [tag, word] of ENTITY_LEVELS) {
    head.push(word + " link with wellness (a correlation, drawn: " + NOT_COUNT + ")",
              word + " windows of its own", word + " counted?");
    counted[tag] = new Set(es[tag + "_counted_names"] || []);
    // the level's own count says whether ANY thing could be counted yet: -1 means too few
    // windows have passed for even a perfect link, so "not yet" would read as a verdict
    reachable[tag] = typeof es[tag + "_counted"] === "number" && es[tag + "_counted"] >= 0;
  }
  const rows = [head];
  names.forEach((nm, i) => {
    const r = [nm, score(pres[i])];
    for (const [tag] of ENTITY_LEVELS)
      r.push(linkPicture((es[tag + "_link"] || [])[i], "opposite", null, "together"),
             String((es[tag + "_entity_windows"] || [])[i]),
             counted[tag].has(nm) ? "yes"
               : reachable[tag] ? "not yet" : "cannot be measured yet");
    rows.push(r);
  });
  const tb = el("table", { class: "kvtable" });
  rows.forEach((r, ri) => {
    const tr = el("tr", {});
    r.forEach(c => tr.appendChild(c instanceof Node ? el(ri ? "td" : "th", {}, c) : el(ri ? "td" : "th", { text: c })));
    tb.appendChild(tr);
  });
  wrap.appendChild(tb);
  for (const [tag, word] of ENTITY_LEVELS) {
    const c = es[tag + "_counted"], togo = es[tag + "_steps_to_first_evidence"];
    const ranks = (es[tag + "_rank"] || []).filter(v => typeof v === "number");
    const best = ranks.length ? Math.max.apply(null, ranks) : null;
    const line = es[tag + "_line"];
    wrap.appendChild(el("h4", { class: "kvgroup", text: "Evidence at the " + word
      + " level (windows of " + (es[tag + "_window_moments"] || 0).toLocaleString() + " model steps)" }));
    wrap.appendChild(kvGrid([
      ["named things with a proven wellness link",
        typeof c === "number" && c >= 0 ? c + " of " + names.length
          : "-1: cannot be measured yet - too few windows have passed for even a perfect link to count"],
      ["windows lived", (es[tag + "_ticks"] === undefined ? "—" : es[tag + "_ticks"])
        + " of the " + es.windows_needed + " a link needs"],
      ["model steps to go before anything here could count",
        togo === 0 ? "0 - they can count now"
          : (typeof togo === "number" ? togo.toLocaleString() : "—")],
      // a RUNNING AVERAGE of the share of its 8 controls beaten each window, started at one half
      // (mimo_bridge/entity_self.py rank_ema) - not a count of comparisons won: a thing that beat
      // all 8 in every one of its first 3 windows reads 64 in 100 (verify-fix-explainer)
      ["best standing against its own 8 controls",
        (best === null ? "—" : "a standing of " + Math.round(100 * best) + " in 100")
        + " (a running average of how many of its 8 controls it beat each window, out of every 100;"
        + " it starts at 50, which is no better than a control, and moves a tenth of the way to each"
        + " window's result, so 100 is reached only by beating all 8 window after window)"
        + (typeof line === "number" ? ", against a pass mark of " + Math.round(100 * line) + " in 100"
           : "; the pass mark cannot be reached yet, so none is shown")]]));
  }
  wrap.appendChild(el("p", { class: "note", text:
    "learning from " + (es.teacher === "ability"
      ? "how many kinds of consequence the body caused (a contact, an object moved, a view gained)"
      : "the body's wellness")
    + " · " + es.frames + " model steps watched across " + es.bodies + " bodies" }));
  const rd = es.reader || {};
  wrap.appendChild(el("p", { class: rd.gate_open ? "warn" : "note", text:
    "the reader: " + (rd.gate_open
      ? "ON, adding " + wUnits(rd.applied_weight) + " per unit of link times presence to a "
        + "remembered model step's value, so the planner aims at model steps that held these things"
      : "off, adding nothing. " + (rd.why ? plainOr(rd.why, READER_WHY) : "not built for this run")) }));
  return wrap;
}

function hselfGroups() {
  const occS = Math.round(1000 * MOMENT_S);
  // each row: [label, reader] - the reader turns the field into a count per 100, a real unit, a
  // percentage between named ends or a picture (DESIGN.md section 11); a bare label is a count
  // a slot's standing is a RUNNING AVERAGE of the share of its 8 controls it beat each window,
  // started at one half (mimo_bridge/hierarchical_self.py rank_ema, momentum 0.9) - never a count of
  // comparisons won: a slot that beat all 8 in its first 3 windows reads 64 in 100 (verify-fix-explainer)
  const standing = x => { const v = num(x); return v === null ? "—" : "a standing of " + Math.round(100 * v) + " in 100"; };
  const per100of = what => x => { const v = num(x); return v === null ? "—" : Math.round(100 * v) + " in 100 " + what; };
  const cosPic = x => picture(x, 0, 1, "nothing in common", "the same pattern");
  const links = x => Array.isArray(x)
    ? el("span", {}, ...x.map(v => linkPicture(v, "opposite", null, "together")))
    : linkPicture(x, "opposite", null, "together");
  const count = x => { const v = num(x); return v === null ? "—" : Number.isInteger(v) ? v.toLocaleString() : sig3(v); };
  const level = (tag, name, win) => [
    "Evidence at the " + name + " level (windows of " + win + " model steps)", {
    [tag + "_sig"]: "slots that count as evidence (-1 = cannot be measured yet: too few windows have passed for even a perfect slot to count)",
    [tag + "_ticks"]: "windows lived so far",
    [tag + "_ticks_needed"]: "windows a perfect slot needs on a table this size before it can count",
    [tag + "_steps_to_first_evidence"]: "model steps to go until the oldest slots could count (0 = they can now)",
    [tag + "_slots_eligible"]: "slots old enough to count: their OWN windows have reached the windows-needed row (0 = none yet, so a 0 in the first row would be forced)",
    [tag + "_slot_ticks_min"]: ["windows lived by the youngest slot (each slot is judged on its own windows)", count],
    [tag + "_slot_ticks_median"]: ["windows lived by the middle slot", count],
    [tag + "_abs_mean"]: ["average size of the wellness link over all slots. On its own this cannot show a link is real, because controls score almost as high",
                          x => picture(x, 0, 1, "no link", "always move together or opposite")],
    [tag + "_top"]: ["the three strongest wellness links (a correlation each, drawn between always opposite and always together)", links],
    [tag + "_rank_top"]: ["best slot's standing against its own 8 controls (50 = no better than a control, 100 = beat all 8 in every window)", standing],
    [tag + "_rank_ceiling"]: ["highest standing any slot could have after this many windows (it starts at 50 in 100 and closes a tenth of the gap to 100 each window)", standing],
    [tag + "_line"]: ["standing a slot must beat to count - set so that a table with no real link shows a counted slot 135 times in 100,000 (blank until it can be reached)", standing],
    [tag + "_line_empirical"]: ["the table's own middle standing plus its spread - shown for comparison, not used as the bar", standing],
    [tag + "_rank_mean"]: ["middle standing of all slots (50 = no better than a control)", standing],
    [tag + "_rank_sd"]: ["spread of the slots' standings (one standard deviation)", per100of("of standing")],
    [tag + "_expected_false"]: ["counted slots a table with no real link would show by luck", per100of("tables would show one")],
    [tag + "_sig_1slot"]: "the older rule's count (each slot judged alone against 50 in 100 plus three standard deviations)",
    [tag + "_line_1slot"]: ["the older rule's bar", standing],
    [tag + "_expected_false_1slot"]: ["counted slots the older rule would show by luck (the slot count times 135 in 100,000)", per100of("tables would show one")] }];
  return [
    ["Is the table about today's model steps?", {
      moments_with_a_slot: ["model steps that belong to a slot - every sense a close match (a cosine of nine tenths or more) to that slot's stored pattern (0 = none: no stored pattern is being refined and the table is about other model steps; 100 = every model step belongs to a slot)", per100of("model steps")],
      moments_with_a_row: ["model steps that belong to a slot (this snapshot was written under the key's first name; 0 = none, 100 = every model step)", per100of("model steps")],
      best_slot_cos: ["how alike a model step and its nearest slot's pattern are, averaged over about 1,000 model steps (blank for the first 100 model steps after a start)", cosPic],
      best_row_cos: ["how alike a model step and its nearest slot's pattern are (this snapshot was written under the key's first name)", cosPic],
      frames: "model steps observed so far",
      slot_age_newest_frames: "model steps since the newest slot was opened (far above 0 = no new pattern has been stored since then)",
      slot_age_median_frames: "age of the middle slot, in model steps",
      slot_age_oldest_frames: "age of the oldest slot, in model steps" }],
    ["The slots", {
      slots_filled: "slots in use. This counts slots, not kinds of model step: two slots can store the same pattern (next rows)",
      slots_distinct: "different patterns stored: slots in use, with slots that store the same pattern (alike to within 1 part in 10,000) counted once",
      slots_twinned: "slots whose pattern is a copy of another slot's (0 = none)",
      twin_groups: "how many different patterns those copies are copies of",
      twin_occupancy: ["share of all presence held by slots that are copies (0 = none, 100 = all of it)", per100of("parts")],
      nearest_other_cos_median: ["how alike the middle slot's pattern and its nearest other slot's pattern are (a model step needs a close match - nine tenths - on every sense to belong to a slot)", cosPic],
      admitted: "how the slots were opened - single_sense: at least one sense matched no stored pattern; new_combination: every sense matched some slot, but no one slot matched them all",
      admission: ["model steps checked for a new slot, and how many the rule called new (100 of every 100 = every checked model step looked new, so the next row counts model steps, not missing kinds of model step)",
                  a => isObj(a) ? Object.entries(a).map(([k, v]) => k === "share_new" ? per100(v) + " called new"
                                                                    : k.replace(/_/g, " ") + " " + count(v)).join(", ") : rawDisp(a)],
      shadow: "model steps that looked new after the table was full, so no slot could be opened - distinct: different patterns among them, kept as a record; sightings: repeats of those; overflow: no room left even to record" }],
    ["Presence: how one model step is shared among the slots", {
      presence_top1: ["biggest share of one model step that any one slot holds (every slot equal = one part in the slots-in-use count; 100% = one slot holds the whole model step; blank = still warming up)", x => pct(x) + " of the model step"],
      presence_eff: ["how many slots one model step is spread over (1 = one slot; the slots-in-use count = spread evenly over all)", x => count(x) + " slots"],
      occupancy_effective_slots: ["how many slots the presence of the last 1,000 model steps is spread over (1 = one slot holds it all; the slots-in-use count = spread evenly)", x => count(x) + " slots"],
      occ_entropy: ["evenness of that spread (an entropy, drawn)", x => picture(x, 0, 1, "one slot holds it all", "perfectly even")],
      slots_live: "slots holding more than a tenth of an even share of the last 1,000 model steps (" + occS + " s of body time)",
      slots_live_margin: ["how far the quietest slot clears that line (100% = exactly on it; near 100% means every slot passes and the row above is at its ceiling)", x => pct(x) + " of the line"] }],
    level("l2", "minutes", "10,000"),
    level("l3", "hours", "100,000"),
    ["How much the stored patterns are moving", {
      center_drift: ["how far a stored pattern turns in one change, averaged over the slots that changed. A pattern changes only when a model step belongs to its slot, so read the first group before this",
                     x => zeroOr(x, "not moving", "turning")],
      center_drift_max: ["the same for the slot that turned the most", x => zeroOr(x, "not moving", "turning")],
      center_step: ["how big one change is, as the part of the way toward the model step it moves. It moves only when some pattern changes, so it can hold an old value",
                    x => { const v = num(x); return v === null || v <= 0 ? "—" : "one part in " + Math.round(1 / v).toLocaleString(); }],
      center_quarter_turn_moments_fastest: ["model steps the fastest-turning pattern would need for a quarter turn (90 degrees) at its current rate - compare with the windows, 10,000 and 100,000 model steps; blank = not turning", x => count(x) + " model steps"],
      center_quarter_turn_moments_mean: ["the same for the average pattern", x => count(x) + " model steps"] }]];
}

// A FIELD NO LABEL NAMES (the "raw fields" row): an integer is a count and is printed; a string, a
// yes or no is printed; any other number is a raw internal whose meaning this page has not written
// down, so it gets the words "not yet measured as a count" and never a bare decimal (DESIGN.md
// section 11). Its value is in the snapshot file itself (explain/step_N.json).
function rawDisp(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString()
                                                        : "a raw number with no lived unit (" + NOT_COUNT + ")";
  if (Array.isArray(v)) return v.every(x => typeof x !== "number" || Number.isInteger(x))
    ? v.map(rawDisp).join(", ") : "a list of " + v.length + " raw numbers with no lived unit (" + NOT_COUNT + ")";
  if (isObj(v)) return Object.entries(v).map(([a, b]) => a + " " + rawDisp(b)).join(", ");
  return String(v);
}
// A row's spec: a plain label (the value is a count or words, printed by rawDisp) or [label,
// reader] - the reader returns the words or the drawn element the row shows.
function specRow(spec, v) {
  const [label, reader] = Array.isArray(spec) ? spec : [spec, null];
  return [label, v == null ? "—" : (reader ? reader(v) : rawDisp(v))];
}
// The same renderer with the rows sorted under plain headings, in the sequence given:
// groups = [[heading, {key: spec, ...}], ...]. A key no group names folds into the
// "raw fields" row, as in labeledKv.
function groupedKv(obj, groups) {
  if (!isObj(obj)) return el("p", { class: "note", text: String(obj) });
  const wrap = el("div", {}), used = new Set();
  for (const [heading, labels] of groups) {
    const rows = [];
    for (const [k, spec] of Object.entries(labels))
      if (k in obj) { rows.push(specRow(spec, obj[k])); used.add(k); }
    if (!rows.length) continue;
    wrap.appendChild(el("h4", { class: "kvgroup", text: heading }));
    wrap.appendChild(kvGrid(rows));
  }
  const rest = Object.entries(obj).filter(([k]) => !used.has(k)).map(([k, v]) => [k, rawDisp(v)]);
  if (rest.length) {
    const det = el("details", {}, el("summary", { text: "raw fields (code names)" }));
    det.appendChild(kvGrid(rest));
    wrap.appendChild(det);
  }
  return wrap;
}
// scalar-dict renderer with plain labels (a label, or [label, reader]); unmapped fields fold into a
// details row. A field the run wrote as None arrives here as null: print the dash every other
// empty value on this page gets, never the word "null" (v == null ? "—" in specRow and rawDisp)
function labeledKv(obj, labels) {
  if (!isObj(obj)) return el("p", { class: "note", text: String(obj) });
  const main = [], rest = [];
  for (const [k, v] of Object.entries(obj))
    labels[k] ? main.push(specRow(labels[k], v)) : rest.push([k, v == null ? "—" : rawDisp(v)]);
  const wrap = el("div", {});
  wrap.appendChild(kvGrid(main));
  if (rest.length) {
    const det = el("details", {}, el("summary", { text: "raw fields (code names)" }));
    det.appendChild(kvGrid(rest));
    wrap.appendChild(det);
  }
  return wrap;
}
// ------------------------------------------------ the gain-of-function build's panels
// DESIGN.md SECTION 11 (operator 2026-09-22): no bare score decimal anywhere a person reads. Every
// number these panels print is a COUNT ("35 of every 100"), a REAL UNIT (cm, degrees, W, g), or a
// PERCENTAGE between two named ends; a cosine or a correlation is described and marked "not yet
// measured as a count" instead of printed.
function num(x) { const n = Number(x); return x === null || x === undefined || x === "" || !isFinite(n) ? null : n; }
// a per-cent number: whole numbers from 10 up, one decimal below, and a tiny non-zero one said as
// "under 0.001" rather than printed in exponent form
function pctNum(v) {
  if (v === null || v === undefined || v !== v) return "—";
  if (v !== 0 && Math.abs(v) < 0.001) return (v < 0 ? "-" : "") + "under 0.001%";
  const a = Math.abs(v);
  // under 1% two significant figures, so 0.04% is never printed as 0%
  return (a >= 10 ? Math.round(v) : a >= 1 ? Math.round(v * 10) / 10 : Number(v.toPrecision(2))) + "%";
}
// a share (0 to 1) as a count out of 100
function per100(x) { const v = num(x); return v === null ? "—" : Math.round(v * 100) + " of every 100"; }
// a share or ratio (1 = all / the reference) as a per-cent number
function pct(x) { const v = num(x); return v === null ? "—" : pctNum(v * 100); }
// a value between two named ends, as the percentage of the way from one to the other
function wayBetween(x, lo, hi, loName, hiName) {
  const v = num(x);
  if (v === null || hi === lo) return "—";
  return pctNum((v - lo) / (hi - lo) * 100) + " of the way from " + loName + " to " + hiName;
}
function withUnit(x, d, unit) { const v = num(x); return v === null ? "—" : v.toFixed(d) + " " + unit; }

// ---- THE PAGE'S READERS FOR THE OLDER PANELS (DESIGN.md section 11, 2026-09-23). The panels built
// before the gain-of-function build printed about 2,000 bare decimals on segment 31's page. Each
// type of number now has one reader that says what it is against named ends (NOT_COUNT and
// BARE_DECIMAL are defined at the top of the file: top-level code that runs at load reads them).
// three significant figures, never exponent form (a number that carries its unit after it)
function sig3(v) {
  const a = Math.abs(v);
  if (a >= 100) return Math.round(v).toLocaleString();
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  if (a === 0) return "0";
  if (a < 1e-6) return (v < 0 ? "-" : "") + "under 0.000001";
  return v.toFixed(Math.min(8, Math.ceil(-Math.log10(a)) + 2));
}
// a token-space error - a mean of squares against a token held at RMS 1 - as a percentage of the
// error an answer of all zeros makes: 0% = exact, 100% = no better than zeros, 200% = an unrelated token
function tokErr(x) { const v = num(x); return v === null ? "—" : pctNum(100 * v) + " of the all-zeros answer's error"; }
// a token-space size (an RMS, a spread) against the size a real token leaves the encoders at, 1.0
function tokSize(x) { const v = num(x); return v === null ? "—" : pctNum(100 * v) + " of a real token's size"; }
// a 0-to-1 score as a percentage of its range
function score(x) { const v = num(x); return v === null ? "—" : pctNum(100 * v); }
// a mean of squares on a 0-to-1 scale as its typical miss (its square root), in per cent of the range
function typMiss(x) { const v = num(x); return v === null || v < 0 ? "—" : "a typical miss of " + pctNum(100 * Math.sqrt(v)) + " of the range"; }
// the top-down gate's blend: parts of the token out of 100 that are the model's own guess (0 = the
// raw token, 50 = the cap)
function blend(x) { const v = num(x); return v === null ? "—" : Math.round(100 * v) + " in 100"; }
// a wellness sum or payment in units of one evaluator's full 0-to-1 range
function wUnits(x) { const v = num(x); return v === null ? "—" : pctNum(100 * v) + " of one evaluator's range"; }
// the planner's plan distance, a mean of squares over the carried senses
function planDist(x) { return wayBetween(x, 0, 2, "at the goal", "as unlike it as two unrelated tokens"); }
// the gradient's size against the clipping line (1.0)
function clipShare(x) { const v = num(x); return v === null ? "—" : pctNum(100 * v) + " of the clipping size"; }
// a size with no lived unit and no named ends: say whether it is zero, and nothing more
function zeroOr(x, zeroText, moreText) {
  const v = num(x);
  return v === null ? "—" : v === 0 ? zeroText : moreText + " (its size has no lived unit: " + NOT_COUNT + ")";
}
// the named end a value sits nearest, in words - for a raw internal (a cosine) shown in a sentence
function nearestEnd(v, ends) {
  let best = null, bd = Infinity;
  for (const e of ends) { const d = Math.abs(v - e[0]); if (d < bd) { bd = d; best = e; } }
  return best ? "nearest '" + best[1] + "' (" + NOT_COUNT + ")" : "—";
}
// A PICTURE, NOT A NUMBER: a raw internal - a cosine, a correlation - drawn as a marker on a track
// between its named ends, followed by the words "not yet measured as a count". No digit of the
// value is printed, and the marker's place is set in its style, never in text.
function picture(x, lo, hi, loName, hiName) {
  const v = num(x);
  const wrap = el("span", { class: "pic" });
  wrap.appendChild(el("span", { class: "picend", text: loName }));
  const track = el("span", { class: "pictrack" });
  if (v !== null && hi !== lo) {
    const m = el("i", {});
    m.style.left = (100 * Math.max(0, Math.min(1, (v - lo) / (hi - lo)))).toFixed(1) + "%";
    track.appendChild(m);
  } else track.appendChild(el("span", { class: "picnone", text: "—" }));
  wrap.appendChild(track);
  wrap.appendChild(el("span", { class: "picend", text: hiName }));
  wrap.appendChild(el("span", { class: "picnote", text: NOT_COUNT }));
  return wrap;
}
// a -1..+1 link (a correlation or a cosine) as a picture with its three named ends
function linkPicture(x, loName, midName, hiName) {
  return picture(x, -1, 1, (loName || "always opposite") + " ←", "→ " + (hiName || "always together"));
}
// a model-written sentence the page cannot restate is shown only when it carries no bare decimal;
// otherwise the page's own words stand in for it
function plainOr(text, fallback) { return (typeof text === "string" && !BARE_DECIMAL.test(text)) ? text : fallback; }

// SIGNED BARS: the wellness channels' contributions run below and above zero, and the drives
// panel's 0-to-1 bars could not draw one below zero at all. The zero line sits in the middle; a bar
// grows left (red) for a negative value and right (blue) for a positive one, scaled to the largest
// size in the set. `say` turns the value into the words beside the bar (never a bare decimal).
function signedBars(rows, say) {
  const wrap = el("div", { class: "sbars" });
  const vmax = Math.max(1e-12, ...rows.map(r => Math.abs(Number(r[1]) || 0)));
  for (const [label, v, note] of rows) {
    wrap.appendChild(el("span", { text: label }));
    const track = el("div", { class: "strack" });
    const n = Number(v);
    if (v !== null && v !== undefined && isFinite(n) && n !== 0) {
      const bar = el("div", { class: "sbar" + (n < 0 ? " neg" : "") });
      const w = Math.max(1, Math.round(Math.abs(n) / vmax * 100));
      bar.style.width = w + "px";
      bar.style.marginLeft = n < 0 ? (100 - w) + "px" : "100px";
      track.appendChild(bar);
    }
    wrap.appendChild(track);
    wrap.appendChild(el("span", { text: (v === null || v === undefined ? "—" : (say ? say(n) : pct(n)))
                                        + (note ? " " + note : "") }));
  }
  return wrap;
}
// each signed value as its share of the whole set's total size, in per cent, with its sign: the
// wellness channels' pull on a goal's value is a sum of weighted spreads, which has no lived unit,
// but each channel's part of the total pull does
function shareOfPull(rows) {
  const tot = rows.reduce((a, r) => a + Math.abs(Number(r[1]) || 0), 0);
  return rows.map(r => [r[0], tot > 0 && num(r[1]) !== null ? Number(r[1]) / tot : null, r[2]]);
}
function table(rows) {
  const tb = el("table", { class: "kvtable" });
  rows.forEach((r, ri) => {
    const tr = el("tr", {});
    r.forEach(c => tr.appendChild(c instanceof Node ? el(ri ? "td" : "th", {}, c)
                                                    : el(ri ? "td" : "th", { text: c == null ? "—" : String(c) })));
    tb.appendChild(tr);
  });
  return tb;
}
// the model registry's own sense names, as schema.py SENSE_WORDS (one word per sense on both pages)
const SENSE_WORD = { vision: "vision", tactile: "tactile", audio: "audio", proprioception: "proprioception",
                     motor: "motor", intermind: "intermind", vestibular: "vestibular" };

// THE CAUSAL CHAIN, one planning event: intention -> prediction -> predicted wellness -> action
// -> result. The first four come from the snapshot (the model's own record at that model step); the
// result from the run's causal_chain.jsonl, read when the page was built, for the newest commands
// whose end had been graded.
function causalChainPanel(snap) {
  const C = snap.causal_chain;
  const kids = [];
  const WELL5 = x => wayBetween(x, 0, 5, "all five scores at their worst", "all five at their best");
  if (!isObj(C)) {
    kids.push(el("p", { class: "note", text: C === undefined
      ? "This snapshot predates the causal-chain capture (schema v3, 2026-09-22)."
      : String(C) + ". The run's launch flag: " + (ARGS.causal_chain ? "--causal-chain " + ARGS.causal_chain
        : "not recorded (the flag arrived on 2026-09-22)") + "." }));
  } else if (C.error) {
    kids.push(el("p", { class: "warn", text: "The planning record at step " + C.step + " failed: " + C.error }));
  } else {
    const g = C.goal || {}, pw = C.pred_wellness || {}, alts = C.alternatives || [];
    // the model keeps its newest planning record until the next plan replaces it (review b1,
    // minor 8), so the record can be older than the snapshot: the heading says by how much
    const ago = (typeof C.step === "number" && typeof (snap.header || {}).step === "number")
      ? snap.header.step - C.step : null;
    kids.push(el("h4", { class: "kvgroup", text: "1 · the intention (planning event at step " + C.step + ", body " + C.body
      + (ago === null ? "" : ago === 0 ? ", this snapshot's own step" : ", " + ago + " model steps before this snapshot") + ")" }));
    const stored = num(g.value_stored);
    const part = x => (stored && stored > 0 && num(x) !== null) ? pctNum(Number(x) / stored * 100) : "—";
    kids.push(kvGrid([
      ["the remembered model step aimed at", (g.age_steps == null ? "—" : g.age_steps + " model steps old")
        + " of a goal ring of " + (g.ring_size == null ? "—" : g.ring_size + " model steps")],
      ["why it was chosen", W_RANK_WORDS[(ARGS.wellness_rank || "raw")] || "the highest stored value in the ring"],
      ["what its raw stored value is made of", "the six formulas " + part(g.bodily) + ", consequence "
        + part(g.consequence) + ", other drives " + part(g.other_drives) + " of it"],
      ["curiosity", num(g.curiosity_factor) === null ? "—" : Number(g.curiosity_factor) === 1 ? "no boost"
        : "raised its value by " + pctNum((Number(g.curiosity_factor) - 1) * 100)]]));
    kids.push(el("h4", { class: "kvgroup", text: "2 · the prediction (made before the command was sent)" }));
    kids.push(kvGrid(Object.entries(C.pred_change || {}).map(([m, v]) =>
      ["expected change of the " + (SENSE_WORD[m] || m) + " token", wayBetween(v, 0, 1, "nothing changes", "a whole token's size")])));
    kids.push(el("h4", { class: "kvgroup", text: "3 · the predicted wellness (the learned copy's five-term sum, 0 to 5)" }));
    const rank = C.chosen_rank;
    kids.push(signedBars([["the chosen plan", pw.chosen, rank != null ? "(rank " + rank + " of " + (alts.length + 1) + ")" : ""]]
      .concat(alts.map(a => [a.replace(/_/g, " "), pw[a], ""])), WELL5));
    kids.push(el("p", { class: "note", text: "Rank 1 = no alternative was predicted higher; picking at random among "
      + (alts.length + 1) + " puts the chosen one first 1 time in " + (alts.length + 1) + ". The planner aims at "
      + "distance to its goal, not at predicted wellness, so this shows whether the two agree." }));
    if (isObj(C.wellness_rank) && isObj(C.wellness_rank.contrib_chosen)) {
      kids.push(el("p", { class: "note", text: "each channel's part of the pull on the chosen goal's value (per cent of the "
        + "total pull; above 0 pulled it up, below 0 pushed it down):" }));
      kids.push(signedBars(shareOfPull(Object.entries(C.wellness_rank.contrib_chosen).map(([k, v]) => [k, v, ""]))));
    }
    kids.push(el("h4", { class: "kvgroup", text: "4 · the action" }));
    const p0 = C.plan0 || {};
    kids.push(kvGrid([
      ["the plan's first command", "moves the body handles " + withUnit(p0.pos_step_cm, 1, "cm") + " on average at "
        + pct(p0.strength_mean) + " of full strength, eyes to " + withUnit(p0.gaze_az_deg, 1, "degrees") + " right / "
        + withUnit(p0.gaze_el_deg, 1, "degrees") + " up"],
      ["did it reach the body?", C.reached_body === true ? "yes: it became the body's command on this model step"
        : C.reached_body === false ? "no: it was computed and not sent (not a latch model step)"
        : "not stamped (the recorder stamps it at the latch)"],
      // THE DISTANCE, NOT THE SETTING (review b1, minor 2): the plan's share of the executed command
      // is the smoother's weight by construction; the recorder now writes how far, in cm, the
      // command the body got sat from the command the planner imagined it would get
      num(C.plan_exec_distance_cm) !== null
        ? ["how far the command sent sat from the command the planner imagined", withUnit(C.plan_exec_distance_cm, 2, "cm")
            + " per handle on average (0 cm = the body got exactly what was imagined)"]
        : ["how much of it survived the smoother", C.plan_share_of_executed == null ? "—"
            : wayBetween(C.plan_share_of_executed, 0, 1, "the body's previous command", "the plan itself")
              + " - the smoother's own weight, a setting rather than a measurement (this snapshot's record "
              + "carries no plan_exec_distance_cm)"]]));
  }
  const EV = (META.causal_events || {}).events || [];
  kids.push(el("h4", { class: "kvgroup", text: "5 · the result (the run's newest resolved commands, read when the page was built)" }));
  if (!EV.length) {
    kids.push(el("p", { class: "note", text: ((META.causal_events || {}).missing
      || "no resolved command is recorded") + "." }));
  } else {
    const rows = [["step", "command by", "reached the body", "sense", "error one model step later (per cent of 'nothing changes')",
                   "at the command's end", "wellness change, predicted / measured (change divided by five, times 100)"]];
    for (const ev of EV.slice().reverse()) {
      const nx = ev.next || {}, la = ev.latch || {};
      const senses = Object.keys(nx.pred_err || {});
      senses.forEach((m, i) => {
        const r1 = (nx.persist_err || {})[m] > 0 ? nx.pred_err[m] / nx.persist_err[m] : null;
        const r2 = (la.persist_err || {})[m] > 0 && (la.pred_err || {})[m] != null ? la.pred_err[m] / la.persist_err[m] : null;
        const w5 = x => num(x) === null ? "—" : pctNum(Number(x) / 5 * 100);
        rows.push([i ? "" : ev.step, i ? "" : (ev.source || "?"),
                   i ? "" : (ev.reached_body === undefined ? "—" : (ev.reached_body ? "yes" : "no")),
                   SENSE_WORD[m] || m, pct(r1), pct(r2),
                   i ? "" : w5(la.predicted_wellness_change) + " / " + w5(la.realized_wellness_change)]);
      });
    }
    kids.push(table(rows));
    kids.push(el("p", { class: "note", text: "100% = no better than predicting that nothing changes, 0% = exact, "
      + "above 100% = worse than nothing changes. Each command's prediction was made before it was sent." }));
  }
  return panel("The causal chain: intention to result", "causal_chain", null, ...kids);
}
const W_RANK_WORDS = {
  raw: "the highest raw stored value in the goal ring (--wellness-rank raw)",
  normalized: "the highest ranking value: each wellness channel measured against its own usual level, "
    + "weighted by priority, with a model step over the integrity gate line never chosen while any remembered model step "
    + "is under it (--wellness-rank normalized)" };

// the efference heads' newest graded command: expected against achieved, per handle
function efferencePanel(snap) {
  const E = snap.efference;
  if (!isObj(E)) return panel("What the command was expected to do", "efference", null,
    el("p", { class: "note", text: E === undefined ? "This snapshot predates the efference capture (schema v3)."
      : String(E) + "." }));
  const kids = [];
  const T = E.tenure;
  const cm = x => withUnit(x, 2, "cm"), deg = x => withUnit(x, 1, "degrees");
  if (isObj(T)) {
    const names = E.handle_names || [];
    const len = v => Array.isArray(v) ? Math.sqrt(v.reduce((a, x) => a + x * x, 0)) : null;
    const rows = [["handle", "expected move", "achieved move", "expected turn", "achieved turn"]];
    (T.expected_cm || []).forEach((v, i) => {
      if (names[i] === "gaze") return;
      rows.push([names[i] || "handle " + i, cm(len(v)), cm(len((T.achieved_cm || [])[i])),
                 deg(len((T.expected_deg || [])[i])), deg(len((T.achieved_deg || [])[i]))]);
    });
    kids.push(el("p", { class: "note", text: "The command that began at step " + T.order_began + " and was graded at step "
      + T.step + ". Lengths of the change vectors; the expectation was formed where the command began." }));
    const det = el("details", {}, el("summary", { text: "every handle" }), table(rows));
    kids.push(det);
  } else kids.push(el("p", { class: "note", text: String(T) + "." }));
  const w = E.window_so_far || {};
  if (Object.keys(w).length) kids.push(kvGrid([
    ["next 50 ms per handle: head / doing nothing / another body's command, size-matched / its absolute command",
     [w.body_cm_err_next, w.body_cm_do_nothing_next, w.body_cm_size_matched_next,
      w.body_cm_shuffled_next].map(cm).join(" / ")],
    ["the command's 250 ms per handle: head / doing nothing / another body's command, size-matched / its absolute command",
     [w.body_cm_err_latch, w.body_cm_do_nothing_latch, w.body_cm_size_matched_latch,
      w.body_cm_shuffled_latch].map(cm).join(" / ")],
    ["eyes at 50 / 100 / 150 ms: head's error",
     [w.gaze_deg_err_1, w.gaze_deg_err_2, w.gaze_deg_err_3].map(deg).join(" / ")],
    ["the same, doing nothing (the eye's own turn)",
     [w.gaze_deg_do_nothing_1, w.gaze_deg_do_nothing_2, w.gaze_deg_do_nothing_3].map(deg).join(" / ")]]));
  kids.push(el("p", { class: "note", text: "An error below doing nothing foresees some of the change; equal to "
    + "another body's size-matched command (that body's asked moves and turns, measured from its own pose, applied "
    + "to this body's pose), the head ignores the command it is handed; 0 cm and 0 degrees = it foresees the whole "
    + "change. The absolute command (that body's targets as they stand, kept for one segment) asks this body for "
    + "changes chosen for another pose. These are this report window's means so far." }));
  return panel("What the command was expected to do, and what it did", "efference", null, ...kids);
}

// the vestibular vector, field by field in its lived unit
function vestibularWords(f) {
  const v = num(f.lived);
  if (v === null) return "—";
  const u = f.unit || "";
  // the total support in body weights, as a percentage of the body's own weight (DESIGN.md section 11)
  if (u === "body weights") return pctNum(100 * v) + " of the body's weight";
  if (/^(g|degrees\/s|W|N m)$/.test(u)) return v.toFixed(2) + " " + u;
  if (u.indexOf("of full strength") === 0) return pctNum(v * 100) + " of full strength (negative = the other way)";
  if (u.indexOf("of the joint's range") === 0) return pctNum(v * 100) + " of the way from the middle of its range to its end (negative = the other end)";
  if (u.indexOf("unit vector") === 0) return pctNum(v * 100) + " along this axis of the chest (100% = straight along it, -100% = straight against it)";
  if (u.indexOf("share of outside support") === 0) return per100(v) + " parts of the outside support";
  if (u.indexOf("share of joints") === 0) return per100(v) + " joints within 2 degrees of a limit";
  if (u.indexOf("(1 = fresh") === 0) return pctNum(v * 100) + " of full capacity (100% = fresh, 0% = spent)";
  if (f.name === "valid") return v > 0.5 ? "yes: a real read" : "no: a padded legacy read";
  return v.toFixed(2) + " " + u;
}
function vestibularPanel(snap) {
  const V = snap.vestibular;
  if (!isObj(V)) return panel("Balance, neck, support and effort", "vestibular", null,
    el("p", { class: "note", text: V === undefined ? "This snapshot predates the vestibular capture (schema v3)." : String(V) + "." }));
  const rows = [["field", "this model step"]].concat((V.fields || []).map(f => [f.name.replace(/_/g, " "), vestibularWords(f)]));
  return panel("Balance, neck, support and effort", "vestibular",
    (V.read_by_model ? "The model reads this vector (--vestibular-sense on): it is the sixth input token."
                     : "The world reports this vector on every run; this run's model does not read it (--vestibular-sense off).")
    + (V.valid ? "" : " This frame is a padded legacy read."),
    el("details", {}, el("summary", { text: "all " + (V.fields || []).length + " fields" }), table(rows)));
}

// the wellness channels of this model step and their signed contributions to its value as a goal
function wellnessChannelsPanel(snap) {
  const W = snap.wellness_channels;
  if (!isObj(W)) return panel("Wellness channels and the goal ranking", "wellness_channels", null,
    el("p", { class: "note", text: W === undefined ? "This snapshot predates the wellness-channel capture (schema v3)." : String(W) + "." }));
  const kids = [];
  const pain = x => wayBetween(x, 0, 1, "no modelled aversion", "an extreme load");
  kids.push(kvGrid([["ranking", W.mode === "normalized" ? "changes from recent history, scaled per channel, multiplied by weights and added"
                                  : "the raw stored value"],
                    ["curiosity", W.mode === "normalized" || W.curiosity_mode === "additive" ? "added as a channel of its own when enabled"
                                  : "multiplies the stored value by one plus the drive"],
                    ["consequence's hand-press kind", W.consequence_world_only ? "counts pressing on the world only" : "counts self-contact too"],
                    ["skin pain this model step", pain(W.pain)]]
    // the model lane's harm row (2026-09-23): the gate reads the larger of skin pain and a limb
    // pressed into the end of its reach; a snapshot without it gates on pain alone
    .concat(W.harm !== undefined ? [["harm (larger of skin aversion and the average handle-position penalty)",
                                     pain(W.harm)],
                                    ["the harm line (over it = never a goal while any remembered model step is under it)", pain(W.gate_line)]]
                                 : [["the pain line (over it = never a goal while any remembered model step is under it)", pain(W.gate_line)]])
    .concat(W.stored_as ? [["what the goal ring stores per channel", W.stored_as]] : [])));
  if (Array.isArray(W.channels)) {
    kids.push(el("p", { class: "note", text: "Each bar shows a channel's signed contribution divided by the sum of all absolute contributions. "
      + "Positive raises the ranking score; negative lowers it. The contribution is the stored change minus the running centre, "
      + "divided by its scale and multiplied by its weight." }));
    kids.push(signedBars(shareOfPull(W.channels.map(c => [c.channel, c.contribution,
      c.weight_in_force !== c.weight ? "(gate shut: contribution zero)" : ""]))));
    const spreads = c => {
      const v = num(c.value), m = num(c.centre), s = num(c.scale), fl = num(c.floor);
      const den = s === null ? null : Math.max(s, fl === null ? 0 : fl);
      if (v === null || m === null || !den) return "—";
      const k = (v - m) / den;
      return Math.abs(k) < 0.05 ? "at its usual level" : withUnit(Math.abs(k), 1, "times its usual spread") + (k > 0 ? " above" : " below");
    };
    const rel = c => { const w = num(c.weight), top = Math.max(...W.channels.map(x => num(x.weight) || 0));
                       return w === null || !top ? "—" : pctNum(w / top * 100) + " of the top priority"; };
    const plannerSees = c => c.channel === "integrity"
      && String(c.planner_sees).includes("tactile_comfort") && String(c.planner_sees).includes("motor_state")
      ? "Partly: the learned network estimates tactile comfort and the handle-position boundary score. It has no integrity output."
      : c.planner_sees || "—";
    const det = el("details", {}, el("summary", { text: "where each channel sits, and its priority" }),
      table([["channel", "this model step against its usual level", "priority weight", "scored in an imagined model step (the planner's learned copy has an output for it)"]]
        .concat(W.channels.map(c => [c.channel, spreads(c), rel(c), plannerSees(c)]))));
    kids.push(det);
  } else kids.push(el("p", { class: "note", text: String(W.channels) + "." }));
  if (isObj(W.controllability)) {
    const c = W.controllability;
    kids.push(kvGrid([["controllability now", "predicted versus measured handle change: direction agreement times smaller/larger change size, averaged over qualified commands"],
                      ["its evidence gate", (c.gate_open ? "OPEN" : "shut") + (c.gate_why ? " - " + plainOr(c.gate_why, CONTROL_GATE_WHY) : "")]]));
  }
  if (isObj(W.learning_progress)) {
    const lp = W.learning_progress;
    kids.push(el("p", { class: "note", text: "Learning progress compares the fast (200 model steps) model-error/no-change-error ratio with the slow (2,000 model steps) ratio. +100% means the fast ratio is zero; 0% means equal ratios; -100% means at least twice the slow ratio." }));
    kids.push(signedBars(Object.entries(lp.per_sense || {}).map(([k, v]) => [SENSE_WORD[k] || k, (v || {}).progress, ""])
      .concat([["the channel (mean)", lp.channel, ""]])));
  }
  return panel("Wellness channels and the goal ranking", "wellness_channels", null, ...kids);
}

// foreseen touches: the contact head's scoring of this model step
function contactPanel(snap) {
  const C = snap.contact_control;
  if (!isObj(C)) return panel("Foreseen touches", "contact_control", null,
    el("p", { class: "note", text: C === undefined ? "This snapshot predates the contact capture (schema v3)." : String(C) + "." }));
  const sp = C.self_predicted || [], wp = C.world_predicted || [];
  const hab = x => num(x) === null ? "" : " (paid at " + pct(x) + " of full: habituation)";
  const would = num(C.would_pay), paid = num(C.pay);
  return panel("Foreseen touches", "contact_control",
    "Touches that began at this model step AND were foreseen by the contact head the model step before. A foreseen "
    + "self-touch pays twice what an outside touch pays; a pair's second time within 10 s pays half.",
    kvGrid([["self-touches foreseen", sp.length ? sp.length + ": " + sp.map(p => p[0] + " + " + p[1] + hab(p[2])).join(", ") : "none this model step"],
            ["outside touches foreseen", wp.length ? wp.length + ": " + wp.map(p => p[0] + hab(p[1])).join(", ") : "none this model step"],
            ["what the channel carried", would === null ? "—" : !would ? "nothing (no foreseen touch)"
              : pctNum((paid || 0) / would * 100) + " of what these touches would pay with the gate open"],
            ["the evidence gate", (C.gate_open ? "OPEN" : "shut: foreseen touches pay nothing yet") + (C.gate_why ? " - " + plainOr(C.gate_why, CONTACT_GATE_WHY) : "")],
            ["contact_control's priority", C.channel_weight ? "set at launch by --contact-control-wellness; its place "
              + "beside the other channels is in the wellness-channels panel" : "off (--contact-control-wellness 0)"]]));
}

// the two sides of every self-contact that began this model step, from the world's own report
function selfContactPairs(snap) {
  const task = (snap.senses || {}).task || {};
  const ons = task.self_contact_onsets;
  if (!Array.isArray(ons)) return el("p", { class: "note", text: "This snapshot's world report carries no self-contact onsets (a world from before 2026-09-22)." });
  if (!ons.length) return el("p", { class: "note", text: "No self-contact began at this model step (both sides of every new touch are listed here when one does)." });
  const fore = new Set(((snap.contact_control || {}).self_predicted || []).map(p => p[0] + "|" + p[1]));
  const rows = [["one side", "the other side", "regions", "touching at rest anyway", "within 30° of an eye", "foreseen by the contact head"]];
  for (const o of ons) {
    const p = o.pair || [], r = o.regions || [];
    rows.push([p[0], p[1], (r[0] || "?") + " / " + (r[1] || "?"), o.structural ? "yes (never scored)" : "no",
               o.in_view ? "yes" : "no",
               isObj(snap.contact_control) ? (fore.has(p[0] + "|" + p[1]) || fore.has(p[1] + "|" + p[0]) ? "yes" : "no") : "no contact head"]);
  }
  return el("div", {}, el("p", { class: "note", text: "Self-contacts that BEGAN at this model step, both sides named (the world's report):" }), table(rows));
}

function renderInspector() {
  const root = document.getElementById("inspector");
  root.textContent = "";
  if (!SNAPS.length) {
    root.appendChild(el("p", { class: "warn", text:
      "No snapshots in this run's explain/ directory yet. Produce some: live - echo '{\"cmd\": \"explain\"}' > <run>/control.json, or launch with --explain-every-min 5; from a checkpoint - python -m mimo_bridge.tools.explain_probe --ckpt <run>/checkpoints/ckpt_step_N. Then rebuild this page." }));
    return;
  }
  const snap = snapByStep(state.s);
  state.s = String(snap.header.step); writeHash();
  const hdr = snap.header;

  const sel = el("div", { class: "selrow" });
  const pick = el("select");
  SNAPS.forEach(s => pick.appendChild(el("option", { value: s.header.step,
    text: `step ${s.header.step} (${s.header.phase || "?"})` })));
  pick.value = hdr.step;
  pick.addEventListener("change", () => { state.s = pick.value; writeHash(); renderInspector(); });
  const idx = SNAPS.indexOf(snap);
  sel.appendChild(el("button", { text: "◀", onclick: () => { if (idx > 0) { state.s = String(SNAPS[idx - 1].header.step); renderInspector(); } } }));
  sel.appendChild(pick);
  sel.appendChild(el("button", { text: "▶", onclick: () => { if (idx < SNAPS.length - 1) { state.s = String(SNAPS[idx + 1].header.step); renderInspector(); } } }));
  sel.appendChild(el("span", { class: "note",
    text: `body ${hdr.body} of ${hdr.n_envs} · weights: ${hdr.weights} · capture ${fmt(hdr.capture_ms, 1)} ms`
      + (META.trained_step ? ` · steps count from 0 in this probe; the weights carry `
        + META.trained_step.toLocaleString() + ` training steps` : "") }));
  sel.appendChild(docBtn("header.weights", "weights label"));
  root.appendChild(sel);
  // the picker lists only the snapshots this page embeds; the builder thins them to keep
  // the page inside its size budget, so say how many exist and which ones are here
  const SN = META.snapshots;
  if (SN && SN.on_disk > SN.embedded)
    root.appendChild(el("p", { class: "note", text:
      "This page embeds " + SN.embedded + " of the run's " + SN.on_disk + " snapshots - the "
      + "first, the newest, and the rest spread evenly between them - "
      + (SN.limited_by === "budget"
         ? "as many as fit the page's " + SN.budget_mb + " MB size budget."
         : "the --max-snapshots cap of " + SN.max_snapshots + ".")
      + " The others are in the run folder's explain/ directory." }));

  // the tiles sit under the picker row; tile 5's snapshot sentence follows the pick
  root.appendChild(renderTiles(snap));
  const grid = el("div", { class: "panelgrid" });
  root.appendChild(grid);

  // --- senses
  const chips = el("div", { class: "chips" });
  for (const [m, v] of Object.entries(hdr.source_mask || {}))
    chips.appendChild(el("span", { class: v, text: `${m}: ${v}` }));
  const eyesP = panel("What the eyes saw", "senses.eyes",
    "The exact pixels the vision encoder consumed this step. Wide = the broad " +
    "periphery, fovea = the sharp centre; one image per eye. The chips say how " +
    "each sense arrived: real (measured now), stale (previous value re-used), " +
    "synthetic (placeholder, teaches nothing).", chips);
  if (isObj(snap.senses.eyes)) {
    const eg = el("div", { class: "eyegrid" });
    for (const [srcName, eyes] of Object.entries(snap.senses.eyes))
      for (const [eye, b64] of Object.entries(eyes)) {
        eg.appendChild(el("figure", {}, el("img", { src: "data:image/png;base64," + b64 }),
                          el("figcaption", { text: `${srcName} · ${eye}` })));
      }
    eyesP.appendChild(eg);
  } else eyesP.appendChild(el("p", { class: "note", text: String(snap.senses.eyes) }));

  // --- touch
  const touchP = panel(`Tactile (${snap.senses.contacts} raw contacts)`, "node.sense.tactile",
    "Each dot is one tactile row - up to 32 of the body's 53 skin parts, highest " +
    "pressure first. A filled dot is a part pressed now (dot size = pressure). A " +
    "hollow dot is a part that is NOT pressed now: its touch ended within the last " +
    "second and only its vibration numbers are still nonzero. A GREEN dot is the body " +
    "touching ITSELF: the force on that part came from another part of the same body, " +
    "which the floor and the adult can never produce. Position is relative " +
    "to the hip, so the pattern moves with the pose. A RED STROKE out of a dot is the " +
    "direction the push came from - the felt numbers say how hard and how much it " +
    "rubbed, and say nothing about direction; the table under the picture gives that " +
    "direction, how deep the contact is in centimetres, and how fast it arrived in " +
    "centimetres a second.");
  if (Array.isArray(snap.senses.touch_cloud)) {
    // every column index and every divisor below comes from the legend the capture writes
    const lg = touchLegend(snap.senses.touch_cloud_doc);
    const cPress = touchCol(lg, /^pressure$/, 3), cRub = touchCol(lg, /^shear$/, 4);
    const cSlip = touchCol(lg, /^slip$/, 5), cArea = touchCol(lg, /^area$/, 6);
    const cSelf = touchCol(lg, /self-touch share/, 11);
    const cPush = touchCol(lg, /push direction/, 13);
    const cDeep = touchCol(lg, /^depth/, 16), cSpd = touchCol(lg, /^arrival speed/, 17);
    const deepCm = touchScale(lg, /^depth/, 10), spdCms = touchScale(lg, /^arrival speed/, 500);
    const svg = el("svg", { class: "chart", width: 220, height: 220, viewBox: "-1.1 -1.1 2.2 2.2" });
    svg.appendChild(el("rect", { x: -1.1, y: -1.1, width: 2.2, height: 2.2, fill: "#fcfcfa" }));
    svg.appendChild(el("circle", { cx: 0, cy: 0, r: 0.02, fill: "#999" }));
    let active = 0, leftover = 0, selfRows = 0, pushRows = 0;
    // DESIGN.md section 11: the place in whole centimetres from the hip (the row stores it divided
    // by half a metre), the direction as each axis's share of a unit length (-100% to 100%)
    const hipCm = v => Math.round(50 * (Number(v) || 0));
    const pushTable = [["where it is (x / y / z, cm from the hip)",
                        "which way the push came from (x / y / z, each axis's part of one unit direction into the body)",
                        "how deep", "how fast it arrived"]];
    for (const row of snap.senses.touch_cloud) {
      if (!row[row.length - 1]) continue;   // validity is always the LAST column
      active++;
      // pressure, rub, slip, area all zero means nothing touches this part NOW
      const pressedNow = row[cPress] > 0 || row[cRub] > 0 || row[cSlip] > 0 || row[cArea] > 0;
      if (!pressedNow) leftover++;
      // the share of this contact that came from the body's OWN parts
      const isSelf = row[cSelf] > 0; if (isSelf) selfRows++;
      const cx = row[0], cy = -row[1];
      svg.appendChild(el("circle", { cx: cx, cy: cy, r: Math.min(0.14, 0.03 + Math.abs(row[cPress]) * 0.02),
        fill: pressedNow ? (isSelf ? "rgba(31,111,67,.60)" : "rgba(42,93,176,.55)") : "none",
        stroke: isSelf ? "#1f6f43" : "#2a5db0", "stroke-width": 0.008 }));
      const px = row[cPush] || 0, py = row[cPush + 1] || 0, pz = row[cPush + 2] || 0;
      const deep = (row[cDeep] || 0) * deepCm, spd = (row[cSpd] || 0) * spdCms;
      if (px || py || pz || deep || spd) {
        pushRows++;
        // one fixed-length stroke out of the dot: the DIRECTION only. How deep and how
        // fast are numbers, and sit in the table below, because a length carrying three
        // quantities at once can be read as none of them.
        if (px || py)
          svg.appendChild(el("line", { x1: cx, y1: cy, x2: cx + px * 0.16, y2: cy - py * 0.16,
            stroke: "#b8442a", "stroke-width": 0.012 }));
        if (pushTable.length <= 8)
          pushTable.push([[row[0], row[1], row[2]].map(hipCm).join(" / ") + " cm",
                          [px, py, pz].map(v => Math.round(100 * v) + "%").join(" / "),
                          fmt(deep, 2) + " cm", fmt(spd, 1) + " cm/s"]);
      }
    }
    touchP.appendChild(svg);
    touchP.appendChild(el("div", { class: "note",
      text: `${active} tactile rows (of ${snap.senses.touch_cloud.length} slots): ${active - leftover} pressed now, ${leftover} holding only the vibration of a touch that ended within the last second · ${selfRows} green: the body touching ITSELF · ${pushRows} carry a push (the red stroke out of the dot, one fixed length, is the direction the push came from) · x/y root-relative, radius ~ pressure` }));
    if (pushTable.length > 1) {
      const pt = el("table", { class: "kvtable" });
      pushTable.forEach((r, ri) => {
        const tr = el("tr", {});
        r.forEach(c => tr.appendChild(c instanceof Node ? el(ri ? "td" : "th", {}, c) : el(ri ? "td" : "th", { text: c })));
        pt.appendChild(tr);
      });
      touchP.appendChild(pt);
      touchP.appendChild(el("p", { class: "note", text:
        "The push columns say WHICH WAY and HOW SUDDEN; the pressure and rub columns above "
        + "say how hard. " + (pushRows > 8 ? "The " + Math.min(8, pushRows) + " hardest-pressed of "
        + pushRows + " pushed rows are listed; " : "") + "a row with no push carries three zeros, "
        + "no depth and no arrival speed, and is left out of this table." }));
    } else {
      touchP.appendChild(el("p", { class: "note", text:
        "No row this step carries a push: every valid row has a push direction of three "
        + "zeros, no depth and no arrival speed. The world fills these from its collision "
        + "entries, so a step with no fresh collision reports none." }));
    }
    if (typeof snap.senses.touch_cloud_doc === "string")
      touchP.appendChild(el("p", { class: "note", text:
        "the row's columns, in order: " + snap.senses.touch_cloud_doc }));
    else
      touchP.appendChild(el("p", { class: "note", text:
        "This snapshot carries no column legend (snapshots written before the capture "
        + "wrote one), so the columns above were read at the positions this run's tactile "
        + "token had when the page was built." }));
  } else touchP.appendChild(el("p", { class: "note", text: String(snap.senses.touch_cloud) }));
  // THE TWO SIDES OF A SELF-CONTACT, PAIRED (2026-09-22): the world reports every own-part pair
  // that came together this model step; both sides are named in one row, beside whether the contact
  // head foresaw it
  touchP.appendChild(selfContactPairs(snap));

  // --- attention
  const att = snap.attention;
  const attP = panel("Where the core looked", "attention",
    "The transformer's attention weights while computing this step. Bright = " +
    "weight. Every ROW is one token's budget of looking, spent across the " +
    "columns (rows always sum to 1). Stage 1: which senses consulted which, " +
    "this model step. Stage 2: how far back each sense looked. Stage 3: everything " +
    "at once.");
  attP.appendChild(el("p", { class: "note",
    text: "re-run on the exact tokens and context the live step consumed; attention heads averaged" }));
  const mods = att.modalities;
  const layerSel = el("select");
  // each core block by the name the model diagram's frames and the defect register use ("block 0 (1st of 4)")
  att.layers.forEach((_, i) => layerSel.appendChild(el("option", { value: i, text: blockName(i, att.layers.length) })));
  layerSel.value = state.ly || 0;
  const modSel = el("select");
  mods.forEach((m, i) => modSel.appendChild(el("option", { value: i, text: m })));
  modSel.value = state.md || 0;
  const tSlider = el("input", { type: "range", min: 0, max: att.ctx - 1, value: state.ts !== undefined ? state.ts : att.ctx - 1 });
  const tLabel = el("span", { class: "note" });
  const attBody = el("div");
  function renderAtt() {
    state.ly = layerSel.value; state.md = modSel.value; state.ts = tSlider.value; writeHash();
    attBody.textContent = "";
    const L = att.layers[parseInt(layerSel.value, 10)] || {};
    const t = parseInt(tSlider.value, 10), mi = parseInt(modSel.value, 10);
    tLabel.textContent = `model step ${t + 1}/${att.ctx}` + (t === att.ctx - 1 ? " (current)" : "");
    const row = el("div", { class: "panelgrid" });
    const s1 = L.stage1 ? decodeQ(L.stage1) : null;
    if (s1) row.appendChild(el("div", {},
      el("h3", {}, "stage 1: senses ↔ senses", docBtn("attention.stage1")),
      heatmap(slice2d(s1, t), { cell: 26, xlabels: mods, ylabels: mods }),
      el("div", { class: "note", text: "row = looking sense, column = looked-at, at the slider's model step" })));
    const s2 = L.stage2 ? decodeQ(L.stage2) : null;
    if (s2) row.appendChild(el("div", {},
      el("h3", {}, `stage 2: ${mods[mi]} through time`, docBtn("attention.stage2")),
      heatmap(slice2d(s2, mi), { cell: Math.max(3, Math.floor(320 / att.ctx)) }),
      el("div", { class: "note", text: "row = model step doing the looking (newest at bottom), column = model step looked at" })));
    if (L.stage3_rows) {
      const s3 = decodeQ(L.stage3_rows);
      const wrap = el("div", {},
        el("h3", {}, "stage 3: the current frame across everything", docBtn("attention.stage3")),
        heatmap({ rows: s3.shape[0], cols: s3.shape[1], at: (r, c) => s3.data[r * s3.shape[1] + c] },
                { cell: 3, cssWidth: 560, ylabels: mods }),
        el("div", { class: "note", text: `${att.seq} columns = ${mods.length} senses × ${att.ctx} model steps, oldest first` }));
      if (L.stage3_pair) {
        wrap.appendChild(el("h3", { text: "sense-pair mean (whole sequence)" }));
        wrap.appendChild(heatmap({ rows: mods.length, cols: mods.length, at: (r, c) => L.stage3_pair[r][c] },
                                 { cell: 22, xlabels: mods, ylabels: mods }));
      }
      if (L.stage3_lag) {
        // Stage 3 passes NO causal mask, so it attends forward in time as freely
        // as backward. This chart plotted only the backward curve under the title
        // "attention vs how far back", which hid close to half the budget. Both
        // curves are drawn now, and the share below them is the summary to read:
        // the curves are means over different numbers of cells, so they are NOT
        // shares and summing them answers nothing.
        const both = Array.isArray(L.stage3_lag_forward);
        const series = [{ label: "looking back: mean weight on the model step that many steps EARLIER (parts of a row's budget per 100)",
                          points: L.stage3_lag.map((v, i) => [i, v]) }];
        if (both)
          series.push({ label: "looking forward: mean weight on the model step that many steps LATER",
                        points: L.stage3_lag_forward.map((v, i) => [i, v]) });
        // noStepMarkers: this x axis counts model steps of distance, not training steps, so
        // the run's restart, phase and snapshot markers do not belong on it
        const lc = lineChart({ log: false, series, noStepMarkers: true, scale: 100, unit: "/100" }, 340);
        // the heading says what was DRAWN: a snapshot written before the capture
        // recorded the forward half carries the backward curve alone
        wrap.appendChild(el("h3", { text: both
          ? "attention vs time distance, both directions"
          : "attention vs how far BACK - this snapshot carries no forward curve" }));
        // the two lines need a key: lineChart draws none of its own
        const lg = el("div", { class: "legend" });
        series.forEach((s, i) => {
          const sw = el("i"); sw.style.background = PALETTE[i % PALETTE.length];
          lg.appendChild(el("span", {}, sw, s.label));
        });
        wrap.appendChild(lg);
        wrap.appendChild(lc.svg);
        if (!both)
          wrap.appendChild(el("div", { class: "note", text:
            "This snapshot was written before the capture recorded stage 3's forward half. "
            + "Stage 3 carries no mask, so it also reads LATER model steps; that part of its "
            + "attention is not on this chart, and its size cannot be read from this "
            + "snapshot. A snapshot written by the current capture carries both curves and "
            + "the three-way share." }));
        const sh = L.stage3_direction_share;
        if (sh && sh.backward != null) {
          const pc = v => (100 * v).toFixed(1) + "%";
          wrap.appendChild(el("div", { class: "note", text:
            "of all this layer's attention: " + pc(sh.backward) + " on earlier model steps, "
            + pc(sh.same_frame) + " on the same model step, " + pc(sh.forward)
            + " on LATER ones. These three are shares of one total and add to 100%; "
            + "the curves above are not shares." }));
        }
      }
      row.appendChild(wrap);
    }
    attBody.appendChild(row);
  }
  layerSel.addEventListener("change", renderAtt);
  modSel.addEventListener("change", renderAtt);
  tSlider.addEventListener("input", renderAtt);
  attP.appendChild(el("div", { class: "selrow" }, layerSel, modSel, tSlider, tLabel));
  attP.appendChild(attBody);
  renderAtt();
  // how big each sense's token is in the remembered model steps the core reads: captured
  // in every snapshot and never drawn until now
  const cr = snap.tokens.context_rms;
  if (Array.isArray(cr) && cr.length && Array.isArray(cr[0])) {
    attP.appendChild(el("h3", {}, "token size in the remembered model steps"));
    attP.appendChild(heatmap({ rows: cr.length, cols: cr[0].length, at: (r, c) => cr[r][c] },
                             { cell: 4, cssWidth: 420, xlabels: mods, say: tokSize }));
    attP.appendChild(el("div", { class: "note",
      text: "row = remembered model step (oldest first), column = sense; colour = that "
          + "token's overall size against the size the encoders hand out for every real "
          + "token (hover a cell for it). No colour means a remembered slot this run has not filled yet." }));
  }

  // --- predictions
  const predP = panel("Predicted vs arrived - the core test", "predictions.of_now",
    "One step ago, knowing its own command, the model predicted THIS model step. " +
    "Rows: what it predicted / what arrived / the difference, one bar per token " +
    "number. The per-sense error is the ingredient of causal_pe - the number " +
    "the dashboard treats as the real one - written as a percentage of the error an " +
    "answer of all zeros makes (0% = exact, 100% = no better than answering zeros). " +
    "Smaller is better.");
  if (isObj(snap.predictions.of_now)) {
    for (const [m, pv] of Object.entries(snap.predictions.of_now)) {
      const act = (snap.tokens.raw || {})[m];
      if (!act) continue;
      predP.appendChild(el("div", { class: "note",
        text: `${m} · delayed-lesson error ${tokErr((snap.predictions.error_of_now || {})[m])} (rows: predicted, arrived, |difference|)` }));
      predP.appendChild(pairStrip(pv.flat ? pv.flat() : pv, act.flat ? act.flat() : act));
    }
  } else predP.appendChild(el("p", { class: "note", text: String(snap.predictions.of_now) }));
  predP.appendChild(kvGrid([["causal_pe (the real one)", tokErr(snap.predictions.causal_pe)],
                            ["recon_pe (predicted change)", tokErr(snap.predictions.recon_pe)]]));
  // the guess of the NEXT model step: in every snapshot, drawn nowhere until now. Nothing
  // has arrived to grade it, so it is shown against THIS model step's token and the
  // difference is the predicted change, not an error.
  if (isObj(snap.predictions.of_next)) {
    const nx = el("p", { class: "note" }, "the guess of the NEXT model step, made just "
      + "now - rows: guess of next, this model step's arrived token, |difference| (that "
      + "difference is the change it expects, not an error) ");
    nx.appendChild(docBtn("predictions.of_next", "prediction of the next frame"));
    predP.appendChild(nx);
    for (const [m, pv] of Object.entries(snap.predictions.of_next)) {
      const now = (snap.tokens.raw || {})[m];
      if (!now || !Array.isArray(pv)) continue;
      predP.appendChild(el("div", { class: "note", text: m }));
      predP.appendChild(pairStrip(pv, now));
    }
  }
  const rd = snap.predictions.recon_dec_error || {};
  if (Object.keys(rd).length)
    predP.appendChild(kvGrid(Object.entries(rd).map(([k, v]) => ["recon decoder · " + k, tokErr(v)])));

  // --- what the core actually read (the top-down gate's measured effect)
  const gateP = panel("What the core actually read", "tokens.fed",
    "Every lesson grades the RAW encoder tokens. The transformer reads a blend: " +
    "where the top-down gate is open, part of the model's own guess of NOW stands " +
    "in for measured sense. The first row is how big that change was next to the " +
    "token itself - 0 means the core read the measured token exactly.");
  if (isObj(snap.tokens.fed) && isObj(snap.tokens.raw)) {
    const rows = [];
    for (const [m, fedv] of Object.entries(snap.tokens.fed)) {
      const rawv = snap.tokens.raw[m];
      if (!Array.isArray(rawv) || !Array.isArray(fedv)) continue;
      const n = Math.min(rawv.length, fedv.length);
      let sd = 0, sr = 0;
      for (let i = 0; i < n; i++) { const d = fedv[i] - rawv[i]; sd += d * d; sr += rawv[i] * rawv[i]; }
      rows.push([m + ": size of the change, against the token's own size", pct(sr ? Math.sqrt(sd / sr) : 0) + " of the token's size"]);
    }
    gateP.appendChild(kvGrid(rows.length ? rows
      : [["every sense", "0% - the core read exactly the raw tokens"]]));
  } else gateP.appendChild(el("p", { class: "note",
    text: snap.tokens.fed === undefined
      ? "this snapshot predates the capture of the tokens the core consumed"
      : String(snap.tokens.fed) }));
  gateP.appendChild(kvGrid([["how far the gate opened, per sense (parts of the token out of 100: 0 = closed, 50 = the cap)",
    snap.drives.td_gate
      ? Object.entries(snap.drives.td_gate).map(([k, v]) => `${k} ${blend(v)}`).join(", ")
      : "closed or off"]]));
  // the other half of this comparison - the encoder output every lesson is graded
  // against - gets its own 'i' button here, the one place the page holds the two
  // side by side
  const rawNote = el("p", { class: "note" }, "the tokens these are compared against ");
  rawNote.appendChild(docBtn("tokens.raw", "raw encoder tokens"));
  gateP.appendChild(rawNote);

  // --- motor
  const mo = snap.motor;
  const motP = panel("The command", "motor.held_order",
    "What the body was told to do. Positions in whole centimetres, strength as a percentage " +
    "of full strength. " +
    "'Prev executed' is the command in force during the PREVIOUS interval - the " +
    "cause the model was just graded against. 'Held' is the command about to be " +
    "issued (one command stays in force for several steps; a limb needs ~250 ms " +
    "to obey).");
  motP.appendChild(kvGrid([
    ["command about to be issued", `#${mo.order_seq}, age ${mo.order_age} steps`],
    ["confidence", score(mo.confidence) + " (0% = none, 100% = full)"],
    ["gaze direction (degrees right / up)", mo.gaze_az_el ? mo.gaze_az_el.map(x => Math.round(x * 57.3) + "°").join(" / ") : "—"],
    ["wellness slope reaching the controller", zeroOr(mo.wellness_grad_norm, "zero: wellness is silent on this step", "present")]]));
  const tbl = el("table", { class: "motor" });
  tbl.appendChild(el("tr", {}, el("th", { text: "handle" }),
    el("th", { text: "held: x / y / z (cm)" }), el("th", { text: "held strength" }),
    el("th", { text: "prev executed: x / y / z (cm)" }), el("th", { text: "prev strength" }),
    el("th", { text: "proposed now: x / y / z (cm; the controller's whole range is 200 cm either side of the world origin)" }),
    el("th", { text: "sent now: x / y / z (cm)" })));
  const held = Array.isArray(mo.held_order) ? mo.held_order : null;
  const prev = Array.isArray(mo.prev_executed) ? mo.prev_executed : null;
  // row count comes from the COMMAND itself; names prefer the receipt (older
  // snapshots stamped the 24-name base list while a trunk body's command holds 26 handles)
  const rcptNames = (META.receipt || {}).handle_names || [];
  const snapNames = mo.handle_names || [];
  const hNames = rcptNames.length >= snapNames.length ? rcptNames : snapNames;
  const nHandles = (held || prev || hNames).length;
  const prop = Array.isArray(mo.motor_out) ? mo.motor_out : null;
  const sent = Array.isArray(mo.motor_for_action) ? mo.motor_for_action : null;
  // whole centimetres (DESIGN.md section 11); the proposal is stored in the controller's -1..1
  // range, which is 200 cm either side of the world origin
  const xyz = r => Array.isArray(r) ? r.slice(0, 3).map(x => Math.round(Number(x) || 0)).join(" / ") + " cm" : "—";
  const xyzProp = r => Array.isArray(r) ? r.slice(0, 3).map(x => Math.round(200 * (Number(x) || 0))).join(" / ") + " cm" : "—";
  const strength = r => Array.isArray(r) ? Math.round(100 * (Number(r[8]) || 0)) + "%" : "—";
  // audit P118: the proposal's position numbers are bounded to -1..1 at the controller;
  // a number AT the edge is one the network asked to be larger than the body can be told
  if (prop) {
    let pinned = 0, total = 0;
    for (const r of prop) if (Array.isArray(r)) for (const x of r.slice(0, 3)) { total++; if (Math.abs(x) >= 0.99995) pinned++; }
    motP.appendChild(kvGrid([["position numbers pinned at the edge of the body's range, proposed now",
      `${pinned} of ${total} (0 = every number chosen inside the range, ${total} = every number pinned; snapshots taken before September 19, 2026 show the unbounded inner number instead)`]]));
  }
  for (let i = 0; i < nHandles; i++) {
    const h = held ? held[i] : null, p = prev ? prev[i] : null;
    tbl.appendChild(el("tr", {}, el("td", { class: "name", text: hNames[i] || "handle " + i }),
      el("td", { text: xyz(h) }),
      el("td", { text: strength(h) }),
      el("td", { text: xyz(p) }),
      el("td", { text: strength(p) }),
      el("td", { text: xyzProp(prop ? prop[i] : null) }),
      el("td", { text: xyz(sent ? sent[i] : null) })));
  }
  const det = el("details", {}, el("summary", { text: "all " + nHandles + " handles" }), tbl);
  motP.appendChild(det);
  const prevNote = el("p", { class: "note" }, "the 'prev executed' columns are the "
    + "command that was in force while these frames were produced - the cause the "
    + "prediction was just graded against ");
  prevNote.appendChild(docBtn("motor.prev_executed", "previously executed command"));
  motP.appendChild(prevNote);
  // THE 'PROPOSED NOW' COLUMN IS NOT WHAT THE MOTOR LESSON GRADES. lem_model.py runs the
  // controller a second time on the PREVIOUS model step's stored inputs and grades that
  // output against the pose reached now (motor_for_loss); this column is the new
  // proposal made from this model step's inputs. On H81's step 100 the column's largest
  // position number was 1.28 while the motor bar read 61,248, and nothing on the page
  // said they were two different outputs.
  motP.appendChild(el("p", { class: "note", text:
    "The 'proposed now' column is the motor controller's new proposal, made from THIS "
    + "model step's inputs. The motor bar in the loss panel is not computed from it: that "
    + "lesson runs the controller a second time on the PREVIOUS model step's stored inputs "
    + "(that model step's core output, handle state and wellness slope) and compares the "
    + "result with the pose the handles have reached by now. The two are different "
    + "outputs, so the bar's size cannot be computed from this column." }));
  if (typeof mo.prev_executed === "string")
    motP.appendChild(el("p", { class: "note", text: mo.prev_executed }));

  // --- drives
  const dr = snap.drives;
  const drP = panel("Measured wellness and curiosity", "drives.wellness",
    "Fixed-formula scores for this body at this recorded model step. Click each name for its inputs, calculation and scale. " +
    "Values are shown as score times 100: tactile comfort cannot exceed 85%, and self-touch is limited by its configured weight. " +
    "Remembered-goal ranking and learned predictions of these scores are separate calculations. Curiosity below is this model step's drive, not a window average.");
  const wb = dr.wellness_body || {};
  const bars = el("div", { class: "bars" });
  // one scale for every bar: 0 to the largest of 1 and the terms, so a term above 1 is not
  // clipped to look like 1, and a negative value draws red to the left of nothing
  const wbMax = Math.max(1, ...Object.values(wb).map(v => Math.abs(Number(v) || 0)));
  for (const [k, v] of Object.entries(wb)) {
    bars.appendChild(docBtn("term.eval-" + k.replaceAll("_", "-"), k));
    const b = el("div", { class: "bar" + (v < 0 ? " neg" : "") });
    b.style.width = Math.round(Math.min(1, Math.abs(Number(v) || 0) / wbMax) * 140) + "px";
    bars.appendChild(el("div", {}, b));
    bars.appendChild(el("span", { text: score(v) }));
  }
  drP.appendChild(bars);
  if (dr.prediction_error_source) {
    const pes = el("p", { class: "note",
      text: "the prediction-error bar on this step was measured as: " + dr.prediction_error_source + " " });
    pes.appendChild(docBtn("drives.prediction_error_source", "which pairing produced the prediction-error bar"));
    drP.appendChild(pes);
  }
  drP.appendChild(kvGrid([
    ["curiosity", `${score(dr.curiosity_drive)} of the full drive (${dr.curiosity_zone || "?"})`],
    ["agency payment", dr.agency ? wUnits(dr.agency.pay) : "off"],
    ["tenure rank1 (last event): the command picked its own outcome", dr.tenure && dr.tenure.tok && dr.tenure.tok.rank1 !== undefined
      ? per100(dr.tenure.tok.rank1) + " command endings" + (dr.tenure.tok.n_decoys ? " (guessing among " + Math.round(dr.tenure.tok.n_decoys) + " decoys scores " + Math.round(100 / (1 + Number(dr.tenure.tok.n_decoys))) + ")" : "") : "—"],
    ["tenure gap (last event): the do-nothing error the command removed", (() => {
      const g = num(((dr.tenure || {}).gap || {}).pose), e = num(((dr.tenure || {}).pe || {}).pose);
      return g === null || e === null || !(g + e > 0) ? "—" : pctNum(100 * g / (g + e)) + " of it (0% = the command explains nothing, 100% = exact)"; })()],
    ["top-down gate (parts of the token out of 100)", dr.td_gate ? Object.entries(dr.td_gate).map(([k, v]) => `${k} ${blend(v)}`).join(", ") : "closed/off"],
    ["movement", dr.movement ? `speed ${fmt(dr.movement.local_speed, 2)} cm/s, stillness debt ${score(dr.movement.stillness_debt)} of full (100% = still for 150 s or more)` : "off"]]));

  // --- intention + losses
  const it = snap.intention;
  const feltNow = Object.values(wb).reduce((a, v) => a + (v || 0), 0)
    + ((dr.movement || {}).movement_wellness || 0) + ((dr.agency || {}).pay || 0);
  const hz = SNAPS.map(x => ((x.intention || {}).last || {}).horizon)
    .filter(h => typeof h === "number");
  const nRep = SNAPS.filter(x => (x.intention || {}).scale_report).length;
  // the senses the planner's score compared at this snapshot: measured (source label real), not motor
  const smP = hdr.source_mask || {};
  const nCmpP = Object.keys(smP).filter(m => m !== "motor" && smP[m] === "real").length;
  const itP = panel("The plan", "intention",
    "On a planner step - 1 model step in " + (((META.args || {}).intention_every) || "the --intention-every count") +
    ", with --intention on - the planner ranks the goal-ring entries, takes the top entry's tokens " +
    "as the goal and runs 20 Adam updates on a command, scoring every command with the forward " +
    "model's prediction of the next model step (the i button gives the whole computation). The plan " +
    "row gives the distance from this model step's tokens to the goal tokens and the predicted " +
    "distance after each command but the last; the descent rows give the score's two parts for the " +
    "starting command and the kept one. 'No plan at this model step' is normal: most model steps " +
    "are not planner steps.");
  itP.appendChild(kvGrid([
    ["this snapshot's plan", planLine(it.last, nCmpP)],
    ...descentRows(it.last, nCmpP),
    ["plans on this page of two or more commands whose predicted distance never fell below the start",
     (() => { const E = SNAPS.map(x => ((x.intention || {}).last)).filter(e => e && e.plan_length > 1 && e.start_mse !== undefined);
              const n = E.filter(e => !(e.final_mse < e.start_mse)).length;
              return E.length ? n + " of " + E.length : "no plan of two or more commands on this page"; })()],
    // THE FIELD WAS RENAMED IN THE MODEL (lem_model.py writes intention_execute_flag since
    // the P125 work) and this row went on reading `executed`: all 14 of H84's snapshots
    // printed "undefined" here. An older snapshot still carries `executed`.
    ["plan sent to the body", it.last
      ? `${it.last.intention_execute_flag !== undefined ? it.last.intention_execute_flag : it.last.executed} - this is the --intention-execute launch setting, not a check that the command was latched; `
        + (it.last.reached_body === true ? "the causal-chain recorder stamped reached_body TRUE: the plan's first command became the body's command at this model step"
           : it.last.reached_body === false ? "the causal-chain recorder stamped reached_body FALSE: the plan was computed and not sent"
           : "reached_body is empty - the causal-chain recorder, which stamps it at the latch, runs only under --causal-chain on")
      : "—"],
    ["goal ring", `${it.goal_ring_size} entries (200 = full)`],
    ...goalRows(it, snap),
    ["the most an entry can store without the curiosity term", it.stored_value_ceiling_no_curiosity === undefined ? "not recorded by this run's capture"
      : isObj(it.stored_value_ceiling_parts)
        // the capture's own parts (schema v3): the reader and an added curiosity are in them
        ? wUnits(it.stored_value_ceiling_no_curiosity) + " = " + Object.entries(it.stored_value_ceiling_parts)
            .filter(([, v]) => v).map(([k, v]) => k.replace(/_/g, " ") + " " + pctNum(100 * v)).join(" + ")
          + " - a stored value above this can only come from the curiosity factor"
        : wUnits(it.stored_value_ceiling_no_curiosity) + " (five evaluators 500% + self-touch up to "
          + pctNum(100 * (((META.args || {}).self_touch_wellness) || 0)) + ", its --self-touch-wellness weight + movement "
          + pctNum(100 * (0.5 + (((META.args || {}).movement_bonus) || 0) * Math.tanh(3.0))) + " + agency "
          + pctNum(100 * ((((META.args || {}).agency_weight) || 0) + (((META.args || {}).agency_prior) || 0)))
          + ((META.args || {}).consequence_w ? " + consequence up to " + pctNum(300 * (META.args || {}).consequence_w) : "")
          + ") - this older capture leaves the entity reader out, and once the reader's gate is open it can add to a stored value too"],
    [`licensed depth over this page's ${SNAPS.length} snapshots`, hz.length ? `${hz[0]} at the earliest plan on this page, ${hz[hz.length - 1]} at the newest, ${Math.min.apply(null, hz)} to ${Math.max.apply(null, hz)} across them - each depth is licensed on its own record, and the first to lose its licence closes the ones above it` : "not recorded"],
    // the report is written on a drop and never cleared (lem_model.py initialises it to None and
    // assigns it only when the size check drops a plan), so an empty one is a statement about the
    // whole process up to the snapshot, not about this model step
    ["size check (predicted token size against this model step's token)", nRep ? `${nRep} of ${SNAPS.length} snapshots carry a report: a plan's predicted tokens grew past twice or shrank past half the size of this model step's tokens, and the whole plan was dropped` : `no report in any of this page's ${SNAPS.length} snapshots; the model keeps the newest report until its process ends, so the size check dropped no plan in this process up to the newest snapshot`]]));
  if ((snap.header || {}).weights === "frozen")
    itP.appendChild(el("p", { class: "note", text:
      "Weights frozen in this run (--eval-only): the planner planned and the " +
      "earned-depth win averages still moved, but no optimizer step ran, so " +
      "nothing on this page was learned." }));

  const lo = snap.losses;
  const loP = panel("This step's teaching signal", "losses",
    "The loss, split by term - each bar is one lesson's size, written as its share of the " +
    "positive lessons' total (a loss has no lived unit, so the page never prints its raw " +
    "value). A term at zero taught nothing this step. Live step only: " +
    "the replay memory runs its own separate updates.");
  if ((snap.header || {}).weights === "frozen")
    loP.appendChild(el("p", { class: "note", text:
      "This snapshot records frozen weights: the displayed model had no training weight update. " +
      "That can describe an evaluation-only run or a frozen actor while a separate learner trains. " +
      "The label alone does not identify which configuration was used." }));
  // Only LOSS TERMS get a bar, and each at the value it actually contributed.
  // Before this, every breakdown entry was drawn: on step_200 the two longest
  // bars were a count of 21 remembered events and an age of 10 seconds, and the
  // scale every other bar was measured against was that count.
  const NOT_A_LESSON = /^(recon_weight$|persistent_memory\.(events|oldest_retained_age_seconds|has_recurrent_prefix|bias_rms)$)/;
  const bd = Object.entries(lo.breakdown || {})
    .filter(([k]) => !NOT_A_LESSON.test(k))
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const lbars = el("div", { class: "bars" });
  const bmax = bd.length ? Math.max(...bd.map(([, v]) => Math.abs(v))) : 1;
  // each lesson against the positive lessons' total: a count per 100 of that total, and a negative
  // lesson (self perception's likelihood) as how many times the positive total it is, below zero
  const posTot = bd.reduce((a, [, v]) => a + (v > 0 ? v : 0), 0);
  const share = v => !(posTot > 0) ? "—" : v >= 0 ? pctNum(100 * v / posTot) + " of the positive total"
    : "below zero, " + (Math.abs(v) / posTot >= 1 ? fmt(Math.abs(v) / posTot, 1) + " times" : pctNum(100 * Math.abs(v) / posTot) + " of")
      + " the positive total";
  for (const [k, v] of bd.slice(0, 14)) {
    lbars.appendChild(el("span", { text: k }));
    const b = el("div", { class: "bar" + (v < 0 ? " neg" : "") });
    b.style.width = Math.max(1, Math.round(Math.abs(v) / (bmax || 1) * 140)) + "px";
    lbars.appendChild(el("div", {}, b));
    lbars.appendChild(el("span", { text: share(v) }));
  }
  loP.appendChild(kvGrid([["total", num(lo.loss) === null ? "—" : (lo.loss < 0 ? "below zero" : "above zero")
                            + " - a sum of lessons with no lived unit (" + NOT_COUNT + ")"],
    ["gradient size, before clipping", clipShare(lo.grad_norm) + " (over 100% = it was scaled down to the line)"],
    ["scope", lo.scope]]));
  // an --eval-only run reports grad norm 0.0 at every snapshot: say so here rather
  // than let the panel's "share of this step's weight update" stand unqualified
  if (lo.grad_norm === 0)
    loP.appendChild(el("div", { class: "note",
      text: "Gradient norm is 0 on this snapshot: no weight moved on this step, so every "
          + "bar below is what that lesson WOULD have taught, not a change that happened." }));
  loP.appendChild(lbars);
  if ((lo.breakdown || {}).motor !== undefined)
    loP.appendChild(el("div", { class: "note", text:
      "The motor bar grades the command the controller gives when it is run a second time "
      + "on the PREVIOUS model step's stored inputs, against the pose the handles have "
      + "reached by now. It is not computed from the command panel's 'proposed now' column, "
      + "which is this model step's new proposal. On the first step after a reset there is no "
      + "previous model step, and the new proposal is graded in its place." }));
  if (lo.replay) loP.appendChild(kvGrid([["replay ring fill", per100(lo.replay.fill_frac) + " slots"],
    ["replay loss (window)", "a lesson value with no lived unit - " + NOT_COUNT + "; the run tab charts it against its own start"]]));
  // timings_ms.backward is (total - encode - forward) whether or not a backward
  // ran (LemLite.step assigns timings["backward"] after the `if train:` block, at
  // the function's own indentation): in an eval probe it times the skipped branch,
  // not backpropagation
  const froze = hdr.weights === "frozen";
  loP.appendChild(kvGrid(Object.entries(lo.timings_ms || {}).map(([k, v]) =>
    [k !== "backward" ? "t \u00b7 " + k
       : froze ? "t \u00b7 after the forward pass (no backpropagation ran - the weights are frozen)"
               : "t \u00b7 backpropagation + the weight update",
     fmt(v, 1) + " ms"])));

  // newer components (schema v2 snapshots carry them; older snapshots do not)
  const extraPanels = [];
  if (snap.memory !== undefined) {
    const MM = isObj(snap.memory) ? snap.memory : {};
    const mCap = MM.capacity || 128;
    const mWin = MM.window_seconds || mCap * (MM.interval_seconds || 0.5);
    extraPanels.push(panel("Memory beyond the window", "node.memory",
      "What the persistent memory holds right now, and how large the recall it " +
      "adds to the newest model step's six tokens is. The transformer never reads " +
      "that addition - the prediction heads, the motor controller and the voice " +
      "head do (the wellness scorer reads the raw tokens, not this addition).",
      labeledKv(snap.memory, {
        events: `events remembered (of ${mCap} - the ring's full size)`,
        oldest_retained_age_seconds: [`oldest event's age (${mWin.toFixed(0)} s when the ring is full)`, x => fmt(x, 1) + " s of body time"],
        encoded_positions: "positions the recurrent network read this step (the kept events + the open half second's best + now)",
        capacity: "the ring's full size, in events",
        interval_seconds: ["one event kept per", x => fmt(x, 2) + " s of body time"],
        window_seconds: ["how far back a full ring reaches", x => fmt(x, 0) + " s of body time"],
        has_recurrent_prefix: "even older events kept as a summary",
        bias_rms: ["recalled bias size, over every body (0% = adds nothing). Read it against the landing-token row: on the H81 lineage that token measured 59% of a real token's size, and that is H81's figure", tokSize],
        bias_rms_body: ["recalled bias size, this body (0% = adds nothing)",
                        x => tokSize(x) + (num(MM.landing_token_rms) > 0 ? " - " + pctNum(100 * x / MM.landing_token_rms) + " of the token it lands on" : "")],
        landing_token_rms: ["size of the token the bias lands on, this body, before the bias is added (0% = all zeros)", tokSize],
        next_bodily_state_error: ["needs-forecast error (0% = all 5 needs came out exactly as forecast, 100% = each off by its whole 0-to-1 range)", typMiss] })));
  }
  if (snap.self !== undefined) {
    const ordS = MOMENT_S;   // one command = one model step
    const back = n => `${n} command${n === 1 ? "" : "s"} back (${(n * ordS).toFixed(2)} s)`;
    const selfShown = isObj(snap.self) ? Object.assign({}, snap.self) : snap.self;
    if (isObj(selfShown)) delete selfShown.consequence_error;
    extraPanels.push(panel("Self perception", "node.self_state",
      "How well the model forecasts its own body's local condition, beside what " +
      "assuming nothing changed at all scores on the same window. Each error is a " +
      "mean of squares on a 0-to-1 scale, shown as its square root, the typical per-row " +
      "miss: 0% = the forecast names the measured value exactly, 100% = wrong by the " +
      "full range. The rollout starts at the measurement that many commands " +
      "ago and is graded against measurements that have already arrived; one command " +
      "is one model step, " + ordS.toFixed(2) + " s of simulated body time.",
      el("p", { class: "note", text:
        "A forecast under 100% of the floor's error beat the do-nothing floor on that " +
        "window; above 100% it did worse than assuming the body did not change at " +
        "all. That floor is this architecture's own zero output, so beating it is " +
        "the admission gate, not the goal." }),
      el("p", { class: "note", text:
        "The stated-spread rows are what the module says its own uncertainty is. " +
        "Compare each with the typical miss above it: they " +
        "agree only if the stated uncertainty is calibrated. The spread is clamped " +
        "to a floor of 1.11% of the range and the clamp zeroes the " +
        "gradient there, so a row reading 1.11% is the code's limit, not the " +
        "module's judgement. self_state_loss in the loss panel is a different " +
        "quantity again - click the line out of the self-perception diagram." }),
      // consequence_error is an alias the stats writer keeps for older readers: the same
      // float as consequence_error_1 (self_state_integration.py), and None with no
      // 1-command row at all before the first executed command. One measurement, one row.
      labeledKv(selfShown, {
        consequence_error_1: ["forecast error, " + back(1), typMiss],
        persistence_error_1: ["floor - if nothing changed, " + back(1), typMiss],
        forecast_vs_persistence_1: ["forecast against the floor, " + back(1) + " (under 100% = beat it)", x => pct(x) + " of the floor's error"],
        forecast_stated_std_1: ["its own stated spread, " + back(1) + " (clamp floor 1.11% of the range)", x => score(x) + " of the range"],
        consequence_error_5: ["forecast error, " + back(5), typMiss],
        persistence_error_5: ["floor - if nothing changed, " + back(5), typMiss],
        forecast_vs_persistence_5: ["forecast against the floor, " + back(5) + " (under 100% = beat it)", x => pct(x) + " of the floor's error"],
        forecast_stated_std_5: ["its own stated spread, " + back(5) + " (clamp floor 1.11% of the range)", x => score(x) + " of the range"],
        consequence_error_20: ["forecast error, " + back(20), typMiss],
        persistence_error_20: ["floor - if nothing changed, " + back(20), typMiss],
        forecast_vs_persistence_20: ["forecast against the floor, " + back(20) + " (under 100% = beat it)", x => pct(x) + " of the floor's error"],
        forecast_stated_std_20: ["its own stated spread, " + back(20) + " (clamp floor 1.11% of the range)", x => score(x) + " of the range"],
        consequence_error_100: ["forecast error, " + back(100), typMiss],
        persistence_error_100: ["floor - if nothing changed, " + back(100), typMiss],
        forecast_vs_persistence_100: ["forecast against the floor, " + back(100) + " (under 100% = beat it)", x => pct(x) + " of the floor's error"],
        forecast_stated_std_100: ["its own stated spread, " + back(100) + " (clamp floor 1.11% of the range)", x => score(x) + " of the range"],
        rehearsal_loss: ["re-study lesson value (a lesson, not an error)", () => "a lesson value with no lived unit - " + NOT_COUNT],
        retained_windows: "stored practice windows",
        learned_present_worst_site_error: ["belief vs measured now, worst single row (0% = names it exactly, 100% = wrong by the full range)", x => score(x) + " of the range"],
        learned_present_function_worst_error: ["belief vs measured sensing resolution, worst row", x => score(x) + " of the range"],
        present_error: ["belief vs measured now, over all rows", typMiss],
        present_training_loss: ["present-belief lesson value", () => "a lesson value with no lived unit - " + NOT_COUNT],
        resolution_min: ["worst skin site's sensing resolution now (100% = the pressure reading is returned untouched, 0% = 2 bits of a 100 kPa range)", x => score(x) + " of full resolution"],
        motor_capacity_min: ["worst actuator group's available force now (100% = the body's native force, 0% = none)", x => score(x) + " of native force"],
        valid_sites: "skin sites observed, all bodies added together (10 actuator rows join them inside the module)",
        relation_events: "remembered relations",
        prediction_mode: "prediction mode",
        present_source: "present read from",
        spatial_regions: "body regions tracked",
        belief_bounded_count: "times its picture of the body was rescaled to stay inside its limit (0 when no runaway happened; above 0 names the runaway that stopped the run of 2026-09-20)",
        // self_state_integration.py (2026-09-26): the live picture's size before the rescale, the
        // times --self-percept-release started a body's stored pictures again from zero, and the
        // per-step counts over the whole report (a record's other rows are its last step only)
        belief_size: ["size of its picture of the body at the report's last step, largest body, before the rescale (H84 healthy 1.4 to 2.0; the limit is 8; 16 to 25 while a body was held at the limit on 2026-09-25)",
                      x => { const v = num(x); return v === null ? "—" : v.toFixed(2) + " (limit 8)"; }],
        self_percept_releases: "times a body's stored pictures were set to zero because its picture ran over the limit, since this segment started (always 0 with --self-percept-release off)",
        window_steps: "steps counted in this report",
        window_live_over_limit: "steps in this report whose picture ran over the limit of 8 (0 when healthy; every step while a body was held at the limit)",
        window_releases: "steps in this report on which a body's stored pictures were set to zero",
        window_body_releases: "bodies whose stored pictures were set to zero in this report",
        window_restored_bodies: "bodies whose restored stored pictures sat at the limit and were set to zero before the first step of this segment",
        window_bounded_calls: "times in this report any picture, live or recomputed for a lesson, ran over the limit",
        window_belief_size_max: ["largest picture size on any step of this report, before the rescale (H84 healthy 1.4 to 2.0; the limit is 8)",
                                 x => { const v = num(x); return v === null ? "—" : v.toFixed(2) + " (limit 8)"; }],
        window_attention_peak_max: ["largest share of a body's recall carried by one stored picture in this report, before the cap (all 128 equal = 0.8%; healthy 0.9% to 3.3%; a stuck body 100%; the cap is 5%)",
                                    x => { const v = num(x); return v === null ? "—" : (100 * v).toFixed(1) + "% of the recall"; }],
        window_attention_capped_steps: "steps in this report on which the 5% cap spread a body's recall",
        window_cleared_rehearsal_windows: "stored practice windows cleared of a released body's pictures in this report",
        window_cleared_events: "saved recent steps cleared of a released body's pictures in this report" })));
  }
  // UEMA 11.2 object incorporation. The trainer has written this into every stats record
  // since the body map grew object columns; nothing drew it, so the toy's number, its own
  // control and whether the world lets the body move the toy at all were readable only by
  // opening the log by hand.
  if (snap.incorporation !== undefined)
    extraPanels.push(isObj(snap.incorporation)
      ? panel("The body map's co-movement numbers", "incorporation",
          "The body map keeps one co-movement number per thing it watches. For each of the body's " +
          "handles: how often the size of that handle's command change and the size of " +
          "proprioception's change are large in the same model step - co-movement, not proof that " +
          "the command caused the change. For each outside thing on its list: how closely the thing " +
          "travelled with a hand, minus the same score with the hand's direction reversed. THE TWO " +
          "ENDS OF THIS SCALE ARE NOT 0 AND 1 - the map runs each score through a sigmoid, " +
          "so a score of 0 reads " + pctNum(100 * INCORP_FLOOR) + " of the 0-to-1 range and a " +
          "score of 1 reads " + pctNum(100 * INCORP_TOP) + ". Every number below is " +
          "written as a percentage of the way between those two ends (0% = a score of 0, 100% = a " +
          "score of 1); for an outside thing, past the middle (50%) " +
          "means the thing kept more than half the hand's distance in the hand's own " +
          "direction.",
          incorporationTable(snap.incorporation, snap.caregiver))
      : panel("The body map's co-movement numbers", "incorporation",
          "This run does not update the body map, so there is no co-movement number for the " +
          "body's handles or for the toy and the mobile. Click the i for what this measures " +
          "when the map is running.",
          el("p", { class: "note", text: String(snap.incorporation) })));
  if (snap.entity_self && snap.entity_self.names)
    extraPanels.push(panel("Named things the self-model watches (dark)", "node.entity_self",
      "The same two levels as the panel below - minutes and hours - over things that can be " +
      "named rather than whole model steps. Each thing has its own presence signal read from the " +
      "world, so it survives the encoders changing. Each level's own block gives its count of " +
      "named things with a proven wellness link; a count of -1 means 'cannot be measured yet', " +
      "not 'none'. It only watches; the reader at the end says whether anything is being " +
      "acted on.",
      entitySelfTable(snap.entity_self)));
  if (snap.hself !== undefined)
    // A run with --hierarchical-self off records the string "absent: hierarchical self off".
    // Titling that "watching only" would tell the reader it is running when it is not.
    extraPanels.push(isObj(snap.hself)
      ? panel("Hierarchical self (dark - watching only)", "node.hself",
          "This part is meant to learn which kinds of model step are good to have around. A slot is one " +
          "stored pattern: what all six senses reported at one model step. It only watches; nothing here " +
          "changes what the model does. Read the first group first: if no model step belongs to a slot, " +
          "every number below is about old model steps. -1 means 'cannot be measured yet'. Click the i " +
          "for the full explanation.",
          groupedKv(snap.hself, hselfGroups()))
      : panel("Hierarchical self (switched off on this run)", "node.hself",
          "The table of whole model steps is not running here, so it stores nothing and produces no " +
          "numbers. The named-things panel above is what watches on this run. Click the i for what " +
          "this part does when it is switched on.",
          groupedKv(snap.hself, hselfGroups())));
  if (snap.voice !== undefined) {
    const vs = SNAPS.map(s => (s || {}).voice).filter(isObj);
    const lo = vs.length ? Math.min(...vs.map(v => v.min)) : null;
    const hi = vs.length ? Math.max(...vs.map(v => v.max)) : null;
    const band = vs.length ? " Across this page's " + vs.length + " snapshots they stay between "
      + pctNum(100 * lo) + " and " + pctNum(100 * hi) + " of the 0-to-1 range they are free to use, a slice "
      + pctNum(100 * (hi - lo)) + " of it wide." : "";
    // P255: bfloat16 carries a tenth of a percent of measurement here - 0.00195 apart below
    // 0.5, 0.00391 at and above it - so the rows print percentages to one decimal and the caption
    // names both spacings (DESIGN.md section 11: as percentages of the range); the step count
    // comes from the page's own band, never a written-in number
    const steps = (lo != null && hi != null)
      ? Math.round((Math.min(hi, 0.5) - Math.min(lo, 0.5)) / 0.001953125
                 + (Math.max(hi, 0.5) - Math.max(lo, 0.5)) / 0.00390625) : null;
    const prec = (snap.voice.dtype === "bfloat16")
      ? " They are carried in bfloat16, a 16-bit number format: neighbouring values are "
        + "0.2% of the range apart below the middle (50%), where the smallest control sits, and 0.4% apart at "
        + "the middle and above, where the largest sits. A tenth of a percent is the finest step that "
        + "carries a measurement" + (steps ? ", and the whole band above is " + steps
        + " of those steps wide." : ".")
      : (snap.voice.dtype ? " They are carried in " + snap.voice.dtype + "."
         : " This snapshot does not record the number format the controls were carried in, so the rows above are shown to a tenth of a percent and no finer.");
    const vshow = Object.assign({}, snap.voice, {
      rms: score(snap.voice.rms) + " of the range", min: score(snap.voice.min) + " of the range",
      max: score(snap.voice.max) + " of the range" });
    extraPanels.push(panel("Voice head", "node.voice",
      "The 80 numbers the voice head set this step. 80 is the reserved width of the "
      + "shared-memory action slot; the repo's vocal-tract parameter set is 56 numbers "
      + "wide, so 24 of these map onto nothing named in the tree. A readout that no "
      + "lesson has ever touched sits at the middle of its range (50%) by construction, and that is what these "
      + "are - they move only when the head is given a lesson, which it has not been."
      + band + prec,
      labeledKv(vshow, {
        rms: "output level (RMS of the 80 numbers)",
        min: "smallest control", max: "largest control",
        n_controls: "numbers out (the reserved action-slot width)",
        dtype: "number format they are carried in",
        world_effect: "effect on the world" })));
  }
  if (snap.caregiver !== undefined)
    extraPanels.push(panel("Caregiver (part of the world)", "node.caregiver",
      "Where the caregiver is and what they are doing this step.",
      labeledKv(snap.caregiver, {
        mode: "mode",
        adult_present: "caregiver present now",
        adult_phase: "what they are doing",
        last_utterance: ["last thing they said", u => isObj(u) ? "\u201c" + (u.text || u.key || "?") + "\u201d"
                           + (num(u.time) !== null ? " at " + fmt(u.time, 1) + " s of world time" : "")
                           + (u.reason ? " (" + String(u.reason).replace(/_/g, " ") + ")" : "") : rawDisp(u)],
        replies: "replies to the baby so far",
        adult_departures: "times they left",
        adult_returns: "times they came back",
        adult_visibility: ["how much of the adult is visible", x => score(x)],
        mouth_opening: ["how far the adult's mouth is open", x => score(x) + " of fully open"],
        object_present: "toy within reach",
        mobile_present: "mobile hanging",
        present_seconds: ["a toy stays in reach", x => fmt(x, 0) + " s"],
        absent_seconds: ["gap before the next toy", x => fmt(x, 0) + " s"],
        head_horizontal_distance_m: ["their head's distance", x => fmt(100 * x, 0) + " cm"],
        head_height_m: ["their head above the infant's eyes", x => fmt(100 * x, 0) + " cm"],
        object_position: ["the toy's place (x / y / z from the world origin)", p => Array.isArray(p) ? p.map(v => Math.round(100 * v)).join(" / ") + " cm" : rawDisp(p)],
        mobile_position: ["the mobile's place (x / y / z from the world origin)", p => Array.isArray(p) ? p.map(v => Math.round(100 * v)).join(" / ") + " cm" : rawDisp(p)],
        recording: ["the caregiver tape", r => isObj(r) ? (r.error ? "error: " + r.error + "; " : "")
                      + (num(r.free_bytes) !== null ? fmt(r.free_bytes / 1e9, 0) + " GB free on its disk" : "")
                      + (num(r.dropped) !== null ? ", " + r.dropped + " lines dropped" : "") : rawDisp(r)],
        object_disappearances: "toys taken away so far",
        simulation_seconds: ["world clock", x => fmt(x, 1) + " s"] })));

  // the gain-of-function build's panels (schema v3): each says so when its snapshot or its
  // run carries none of it
  // A block the snapshot does not carry as an object (a capture from before schema v3, or a
  // feature this run was launched without) is listed once in a summary panel, with its reason,
  // instead of five near-empty panels
  const GOF = [["causal_chain", "the causal chain", causalChainPanel],
               ["efference", "the efference heads", efferencePanel],
               ["wellness_channels", "the wellness channels", wellnessChannelsPanel],
               ["contact_control", "foreseen touches", contactPanel],
               ["vestibular", "the vestibular vector", vestibularPanel]];
  const gofPanels = [], gofAbsent = [];
  for (const [key, name, fn] of GOF) {
    if (isObj(snap[key])) gofPanels.push(fn(snap));
    else gofAbsent.push([name, snap[key] === undefined
      ? "this snapshot was captured before the build's capture existed (schema " + (hdr.schema_version || "?") + ")"
      : String(snap[key]).replace(/^absent: /, "")]);
  }
  if (gofAbsent.length) {
    const lst = el("div", {});
    for (const [name, why] of gofAbsent) {
      const row = el("p", { class: "note" }, name + ": " + why + " ");
      row.appendChild(docBtn(GOF.find(g => g[1] === name)[0], name));
      lst.appendChild(row);
    }
    gofPanels.push(panel("The gain-of-function build's panels not drawn for this snapshot", null,
      "Each would draw a part of the 2026-09-22 build for this model step; the reason each is missing:", lst));
  }
  // panels appended in order of importance: the causal test first, then where the
  // core looked, the command, then the raw senses, then motivation and bookkeeping
  for (const p of [predP, ...gofPanels, gateP, attP, motP, eyesP, touchP, drP,
                   ...extraPanels, itP, loP]) grid.appendChild(p);
}

// ================================================================== TAB 3: run replay
function renderRun() {
  const root = document.getElementById("runtab");
  root.textContent = "";
  if (!SERIES.charts || !SERIES.charts.length) {
    root.appendChild(el("p", { class: "warn", text: "No stats series found (replay_training_stats.jsonl empty or missing)." }));
    return;
  }
  root.appendChild(renderTiles());
  if (META.stats_source && META.stats_source !== META.run) {
    // every flag value printed on the architecture tab comes from META.args - the
    // --run directory's args.json - while these charts come from another run. Name
    // the values where the two disagree, so no reader carries one run's setting onto
    // the other's history.
    const a1 = META.args || {}, a2 = META.stats_args || {};
    const diff = Object.keys(a1).filter(k => k in a2
      && JSON.stringify(a1[k]) !== JSON.stringify(a2[k]));
    root.appendChild(el("p", { class: "note", text:
      "These charts show the TRAINING run " + META.stats_source + " - the history the "
      + "snapshots' weights actually lived. The inspector's snapshots are from this "
      + "probe, whose own steps count from 0, so no snapshot dots appear here."
      + (diff.length ? " Every flag value printed on the architecture tab is the "
         + "PROBE's; these differ between the two runs: "
         + diff.map(k => k + " " + JSON.stringify(a1[k]) + " here, "
                   + JSON.stringify(a2[k]) + " in " + META.stats_source).join("; ")
         + "." : "") }));
    // TWO RUNS, TWO DIFFERENT BODIES. --intention-every against --motor-hold decides
    // who writes the commands the body executes: the planner writes a latched command
    // whenever the two step sets meet, which is gcd(hold, every) / every of them, and
    // exploration noise is added only on the model steps it does not. A reader carrying a
    // number from the architecture tab to these charts is not reading one body.
    const gcd2 = (x, y) => y ? gcd2(y, x % y) : x;
    const rule = "when --intention-every divides --motor-hold, every executed command is "
      + "a planner command and exploration noise reaches the body on none of them; "
      + "otherwise the planner writes gcd of the two divided by --intention-every of "
      + "them and noise reaches the rest.";
    const pair = o => (typeof o.intention_every === "number" && typeof o.motor_hold === "number")
      ? "--intention-every " + o.intention_every + ", --motor-hold " + o.motor_hold
        + " (planner writes " + Math.round(100 * gcd2(o.motor_hold, o.intention_every)
                                           / o.intention_every) + " of every 100 latched commands)"
      : null;
    const p1 = pair(a1), p2 = pair(a2);
    root.appendChild(el("p", { class: "warn", text:
      "Two runs, two different bodies. The boxes on the Architecture tab read snapshots of "
      + META.run + (p1 ? " (" + p1 + ")" : "") + "; these charts read " + META.stats_source
      + (p2 ? " (" + p2 + ")" : " - this page does not carry that run's launch flags, so the "
              + "same pair cannot be printed for it") + ". That pair decides who writes the "
      + "commands the body executes: " + rule }));
  }
  // THE LEGEND MUST DESCRIBE WHAT WAS DRAWN. All three marker kinds were named
  // unconditionally while this page carries none of them.
  const drawn = [], absent = [];
  ((SERIES.seams || []).length ? drawn : absent).push(
    (SERIES.seams || []).length ? "red dashed = process restart (resume seam)"
                               : "no process restart is marked: this stats log holds none");
  ((SERIES.phases || []).length ? drawn : absent).push(
    (SERIES.phases || []).length ? "green dashed = phase change"
                                 : "no phase change is marked: this stats log holds none");
  ((SERIES.snapmarks || []).length ? drawn : absent).push(
    (SERIES.snapmarks || []).length
      ? "blue dots = snapshots (click to inspect)"
      : (META.stats_source && META.stats_source !== META.run
         ? "no snapshot dots: these charts are " + META.stats_source + "'s and the "
           + "snapshots are " + META.run + "'s, whose steps count from 0 - two "
           + "different step axes, so a dot would land on nonsense"
         : "no snapshot dots: this run wrote no snapshot on this axis"));
  root.appendChild(el("p", { class: "note", text:
    drawn.concat(absent).join(" · ") + (SERIES.thinning ? " · " + SERIES.thinning : "") }));
  // a chart whose feature this run was launched without is named apart from a chart the log
  // is too thin for: the first says the feature is off, the second that the data are missing
  if ((SERIES.off || []).length)
    root.appendChild(el("p", { class: "note", text:
      "Not drawn, because this run was launched without the feature they read (their block is in "
      + "no record of the log): " + SERIES.off.join(" · ") + "." }));
  // a chart the builder declares and cannot fill is NAMED, never left out in silence
  if ((SERIES.dropped || []).length)
    root.appendChild(el("p", { class: "note", text:
      "Not drawn, because this stats log carries fewer than two points for every field "
      + "they read: " + SERIES.dropped.join(" · ") + "." }));
  const wrap = el("div", { style: "display:flex; gap:1rem; align-items:flex-start;" });
  const col = el("div", { style: "flex:1 1 auto; min-width:0;" });
  const readout = el("div", { class: "readout" });
  readout.appendChild(el("b", { text: "hover a chart" }));
  wrap.appendChild(col); wrap.appendChild(readout);
  root.appendChild(wrap);
  const charts = [];
  for (const ch of SERIES.charts) {
    const block = el("div", { class: "chart-block" });
    const h = el("h3", {}, ch.label);
    if (ch.doc) h.appendChild(docBtn(ch.doc, ch.label));
    if (ch.trend) {
      const key = ch.trend.key;
      const w = winStats((ch.series.find(s => s.key === key) || {}).points);
      if (w) {
        const v = verdict(w.a, w.b, ch.trend.dir);
        h.appendChild(el("span", { class: "chip " + v.cls,
          text: v.txt + (ch.trend.dir === "error" ? " — this is an error, smaller is better" : "") }));
      }
    }
    block.appendChild(h);
    if (ch.caption) block.appendChild(el("p", { class: "caption", text: ch.caption }));
    const leg = el("div", { class: "legend" });
    ch.series.forEach((s, i) => {
      const sw = el("i"); sw.style.background = PALETTE[i % PALETTE.length];
      leg.appendChild(el("span", {}, sw, s.label));
    });
    block.appendChild(leg);
    const c = lineChart(ch, 760);
    charts.push({ def: ch, r: c });
    block.appendChild(c.svg);
    col.appendChild(block);
  }
  function scrub(step) {
    readout.textContent = "";
    readout.appendChild(el("b", { text: "step " + Math.round(step) }));
    const t = el("table");
    // each value in its chart's own terms (DESIGN.md section 11), read off the chart's view of
    // its data - never the logged raw number
    for (const { def, r } of charts)
      def.series.forEach((s, si) => {
        let best = null;
        for (const p of ((r.view || [])[si] || s.points)) { if (p[0] <= step) best = p; else break; }
        if (best) t.appendChild(el("tr", {}, el("td", { text: def.label + " · " + s.label }),
                                  el("td", { text: sayChartValue(def, best[1]) })));
      });
    readout.appendChild(t);
    for (const { r } of charts) {
      if (!r.px) continue;
      const x = r.px(step);
      r.cursor.setAttribute("x1", x); r.cursor.setAttribute("x2", x);
    }
    state.x = String(Math.round(step)); writeHash();
  }
  for (const { r } of charts) {
    if (!r.px) continue;
    r.svg.addEventListener("mousemove", ev => {
      const b = r.svg.getBoundingClientRect();
      const frac = (ev.clientX - b.left) / b.width;
      const vb = r.svg.viewBox.baseVal;
      const xpix = frac * vb.width;
      const lm = r.ml || 52;
      const step = r.x0 + (xpix - lm) / (vb.width - lm - 10) * (r.x1 - r.x0);
      if (step >= r.x0 && step <= r.x1) scrub(step);
    });
  }
  if (state.x) scrub(parseFloat(state.x));
}

// ================================================================== TAB 4: probes + liveness
// The gain-of-function probes (DESIGN.md section 9) and the liveness measurements, both read when
// the page was built (META.probes, META.liveness). Every probe number is placed between its floor
// (what knowing nothing scores) and its goal (what working control would score).
const PROBE_ORDER = [["gaze", 7], ["body", 8], ["binding", 9], ["posture", 10]];
function probeBar(h) {
  const v = Number(h.value), lo = Number(h.floor), hi = h.goal === null || h.goal === undefined ? null : Number(h.goal);
  const wrap = el("div", { class: "pbar" });
  if (hi === null || !isFinite(v) || !isFinite(lo) || hi === lo) {
    wrap.appendChild(el("span", { class: "note", text: "goal unmeasured: the distance to it cannot be drawn" }));
    return wrap;
  }
  const frac = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
  const track = el("div", { class: "ptrack" });
  const fill = el("div", { class: "pfill" }); fill.style.width = Math.max(1, Math.round(frac * 240)) + "px";
  track.appendChild(fill);
  wrap.appendChild(el("span", { class: "note", text: "0%: knowing nothing" }));
  wrap.appendChild(track);
  wrap.appendChild(el("span", { class: "note", text: "100%: working control" }));
  return wrap;
}
// A probe headline as a percentage of the way from its floor (what knowing nothing scores) to its
// goal (what working control would score) - DESIGN.md section 11; the probe's own metric name and
// raw value stay in its result file, named beside it.
// A PROBE'S OWN SENTENCE is shown as the probe wrote it, with one change: a number the probe wrote
// in exponent form before a unit ("7.3e-06%") is written out in plain digits ("0.0000073%") -
// the same value, in the one notation a reader can picture (DESIGN.md section 11)
function plainExponents(text) {
  return String(text).replace(/(-?\d+(?:\.(\d+))?)[eE]([-+]?\d+)(?=\s*(%|cm\b|degrees?\b|°|s\b|ms\b))/g,
    (m, mant, frac, ex) => {
      const v = Number(mant) * Math.pow(10, Number(ex));
      const d = Math.min(12, Math.max(0, -Number(ex) + (frac || "").length));
      return isFinite(v) ? v.toFixed(d) : m;
    });
}
function probeHeadline(hl) {
  const v = num(hl.value), lo = num(hl.floor), hi = num(hl.goal);
  if (v === null) return "not measured";
  if (lo === null || hi === null || hi === lo)
    return "goal unmeasured: the probe's " + (hl.metric || "metric") + " has no goal line to place it against";
  return wayBetween(v, lo, hi, "what knowing nothing scores", "what working control would score");
}
function renderProbes() {
  const root = document.getElementById("probestab");
  if (!root) return;
  root.textContent = "";
  const P = META.probes || {};
  const h = el("h3", {}, "The gain-of-function probes (items 7 to 10)");
  h.appendChild(docBtn("probes", "the gain-of-function probes"));
  root.appendChild(h);
  const cks = P.checkpoints || [];
  root.appendChild(el("p", { class: "note", text: P.missing ? "Not shown: " + P.missing + "."
    : "Lineage " + (P.lineage || "?") + ": " + cks.length + " probed checkpoint" + (cks.length === 1 ? "" : "s")
      + (P.checkpoints_total > cks.length ? " of " + P.checkpoints_total : "") + ", newest first, read from "
      + "Saved/mimo_bridge/gof_probes/index.jsonl when this page was built. Each result changes ONLY one command "
      + "from the checkpoint's own lived model steps and asks whether the model's prediction moves the way the real "
      + "outcome does." }));
  const grid = el("div", { class: "panelgrid" });
  for (const c of cks) {
    const kids = [];
    for (const [name, item] of PROBE_ORDER) {
      const d = (c.probes || {})[name];
      if (!d) { kids.push(el("p", { class: "note", text: "item " + item + " (" + name + "): not probed on this checkpoint" })); continue; }
      const hl = d.headline || {};
      kids.push(el("h4", { class: "kvgroup", text: "item " + item + " · " + name + " - " + (d.status || "?") + ", " + (d.time || "") }));
      kids.push(kvGrid([
        ["where it stands", probeHeadline(hl)],
        ["what was measured", (hl.unit || "?") + " (the probe's metric " + (hl.metric || "?") + ")"]]));
      kids.push(probeBar(hl));
      if (hl.lived) kids.push(el("p", { class: "caption", text: plainExponents(hl.lived) }));
      if ((d.controls || []).length) {
        const ul = el("ul", { class: "doclist" });
        d.controls.forEach(t => ul.appendChild(el("li", { text: t })));
        kids.push(el("div", { class: "note", text: "controls:" }), ul);
      }
    }
    grid.appendChild(panel("Checkpoint " + c.checkpoint + " (lineage step " + (c.lineage_step == null ? "?" : Number(c.lineage_step).toLocaleString())
      + ", run " + (c.run || "?") + ")", "term.captured-share", null, ...kids));
  }
  root.appendChild(grid);
  // ---- the liveness measurements
  const L = META.liveness || {};
  const h2 = el("h3", {}, "Dead and locked parts of the network");
  h2.appendChild(docBtn("liveness", "dead and locked parts"));
  root.appendChild(h2);
  if (L.missing) { root.appendChild(el("p", { class: "note", text: "Not shown: " + L.missing + "." })); return; }
  root.appendChild(el("p", { class: "caption", text: "Measured by the lead on 2026-09-22 on the lineage's own checkpoints "
    + "(CPU, 24 lived model steps each). A unit counts as live when, on some model step, its input left the flat low end of "
    + "its activation function (GELU, the curve that turns a unit's input into its output). In that flat end a change "
    + "in the input moves the output by less than 1.2% of the change, so the unit outputs about nothing and the "
    + "lesson's correction reaches it shrunk by the same share; for a large input the output moves as much as the "
    + "input (100%)." }));
  const live = (L.columns || []).map((c, i) => [c, i]).filter(([c]) => !/pool|cross/.test(c.key));
  const locked = (L.columns || []).filter(c => /pool|cross/.test(c.key));
  if (live.length) {
    const cell = v => { const n = num(String(v).replace("*", ""));
      return n === null ? String(v).replace("(fresh)", "rebuilt fresh at this step")
        : Math.round(n * 100) + " of 100" + (String(v).indexOf("*") >= 0 ? " (out of line: not trusted)" : ""); };
    const rows = [["checkpoint step"].concat(live.map(([c]) => c.name + ": units live"))]
      .concat((L.rows || []).map(r => [Number(r.step).toLocaleString()].concat(live.map(([, i]) => cell(r.values[i])))));
    root.appendChild(table(rows));
  }
  if (locked.length) root.appendChild(el("p", { class: "note", text: "The attention columns of the same table ("
    + locked.map(c => c.name).join("; ") + ") are entropies, which this page does not print: not yet measured as a "
    + "count over the lineage. The newest checkpoint's count form is below." }));
  const sw = L.sweep || {};
  if (Object.keys(sw.dead_layers || {}).length) {
    root.appendChild(el("h4", { class: "kvgroup", text: "the newest sweep (" + sw.file + ", " + sw[`moments`] + " lived model steps): layers where fewer than half the units ever respond" }));
    root.appendChild(signedBars(Object.entries(sw.dead_layers).map(([k, v]) => [livenessName(k), v, ""]),
      v => Math.round(v * 100) + " of every 100 units live"));
  }
  if (Object.keys(sw.locked_attention || {}).length) {
    root.appendChild(el("h4", { class: "kvgroup", text: "attention locked on one input: out of every 100 parts of its attention, how many its single largest input takes" }));
    root.appendChild(signedBars(Object.entries(sw.locked_attention).map(([k, a]) => [livenessName(k), isObj(a) ? a.top_share : a,
      isObj(a) && a.keys ? "(of " + a.keys + " inputs it could read)" : ""]),
      v => Math.round(v * 100) + " of 100"));
  }
}

// ------------------------------------------------------------------ layout check (#layoutcheck)
// DIAGRAM LAYOUT ON A GRID (the style guide, operator 2026-10-02): no label may overlap a box or run past
// the drawing's edge. With #layoutcheck in the address the page measures the architecture diagram
// (tab 1, as boot drew it), every detail diagram and the learning circle: each detail diagram is drawn
// into a holder 1,200 pixels wide and the other two are set to one pixel per unit while they are
// measured, every text and every box is measured with getBBox in the drawing's own viewBox units (the
// browser lays text out in whole pixels, so a drawing shrunk below one pixel per unit reads its text
// boxes larger; see layoutCheck), and what it found is written as JSON into a pre element whose id is
// layout-check (the tag is not spelled out here: this comment is dumped with the page). tools/check_explainer_page.py check_layout reads it from a headless browser's dump of the
// page (tests/test_explainer_layout.py). What counts (LAYOUT_SLACK units of slack wherever two things
// may not overlap, so a hairline touch is not an overlap):
//   past the edge         a text or a numbered circle outside the viewBox
//   past its box          a box's own text closer to its box's edge than LAYOUT_MARGIN, or outside
//                         it: 3 units at the left and right, 1 unit at the top and bottom (getBBox
//                         already holds the font's full ascent and descent, which reach past the
//                         letters themselves)
//   over a box            any other text inside a box (a plain curve's label belongs to the box the
//                         curve is drawn in; a numbered circle belongs to the box it numbers)
//   labels overlap        two texts' boxes intersect; a numbered circle counts as a text
//   boxes overlap         two boxes intersect
//   across a frame        a text or a box partly inside the dashed frame of one transformer block:
//                         each sits wholly inside a frame or wholly outside it
//   line through a label  a straight segment of a line crosses a text that is not a box's own
//   line through a box    a straight segment of a line crosses the inside of a box (a line that
//                         starts or ends at a box touches only its edge)
//   line crosses an arrowhead  a segment of one line meets the last 12 units of another line's last segment (its
//                         7-unit arrowhead and 5 units before it), the two not of one comb (Oct 15, 2026)
// The architecture diagram's boxes are g.dg-node, its lines path.dg-edge, its block frames
// rect.dg-frame and its numbered circles circle.dg-badge; a detail diagram's boxes are g.mini-node and
// its lines path.mini-edge.
const LAYOUT_SLACK = 1;
const LAYOUT_MARGIN = { x: 3, y: 1 };
function layoutRect(b) { return { x: b.x, y: b.y, w: b.width, h: b.height }; }
function layoutOverlap(a, b, s) {
  const w = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
  const h = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
  return (w > s && h > s) ? [Math.round(w * 10) / 10, Math.round(h * 10) / 10] : null;
}
function layoutShrink(r, dx, dy) { return { x: r.x + dx, y: r.y + dy, w: r.w - 2 * dx, h: r.h - 2 * dy }; }
// whether the segment p -> q passes through the inside of the rectangle r (Liang-Barsky clipping)
function segmentCrosses(p, q, r) {
  if (r.w <= 0 || r.h <= 0) return false;
  let t0 = 0, t1 = 1;
  const dx = q[0] - p[0], dy = q[1] - p[1];
  for (const [pp, qq] of [[-dx, p[0] - r.x], [dx, r.x + r.w - p[0]], [-dy, p[1] - r.y], [dy, r.y + r.h - p[1]]]) {
    if (pp === 0) { if (qq < 0) return false; continue; }
    const t = qq / pp;
    if (pp < 0) { if (t > t1) return false; if (t > t0) t0 = t; }
    else { if (t < t0) return false; if (t < t1) t1 = t; }
  }
  return t1 - t0 > 1e-6;
}
// the straight segments of a path written "M x y L x y ..." (a path with an arc gives none)
function pathSegments(d) {
  if (!/^\s*M[\d\s.,L-]*$/.test(d || "")) return [];
  const v = (d.match(/-?\d+(\.\d+)?/g) || []).map(Number), out = [];
  for (let i = 0; i + 3 < v.length; i += 2) out.push([[v[i], v[i + 1]], [v[i + 2], v[i + 3]]]);
  return out;
}
function layoutProblems(name, svg) {
  const vb = svg.viewBox.baseVal, s = LAYOUT_SLACK, M = LAYOUT_MARGIN, found = [];
  const words = t => (t.textContent || "").slice(0, 70);
  const r1 = v => Math.round(v * 10) / 10;
  const boxes = Array.from(svg.querySelectorAll("g.mini-node, g.dg-node")).map(g => {
    const r = g.querySelector("rect");
    return { g, r: layoutRect(r.getBBox()), name: words(g.querySelector("text") || r) || "an unlabelled box" };
  });
  const frames = Array.from(svg.querySelectorAll("rect.dg-frame")).map(f =>
    ({ r: layoutRect(f.getBBox()), name: f.getAttribute("data-name") || "a block frame" }));
  // every text, and every numbered circle as a text of its own whose home is the box it numbers (its
  // number is drawn inside the circle, so the circle stands for both)
  const texts = Array.from(svg.querySelectorAll("text"))
    .filter(t => (t.textContent || "").trim() && !t.classList.contains("dg-badge-num"))
    .map(t => ({ name: words(t), r: layoutRect(t.getBBox()), box: boxes.find(b => b.g.contains(t)) || null,
                 inBox: t.getAttribute("data-in-box") === "1", badge: false, badgeOf: null }));
  for (const c of svg.querySelectorAll("circle.dg-badge"))
    texts.push({ name: "numbered circle " + (c.getAttribute("data-n") || "") + " on " + (c.getAttribute("data-for") || "?"),
                 r: layoutRect(c.getBBox()), box: null, inBox: false, badge: true,
                 badgeOf: boxes.find(b => b.g.id === "nd-" + c.getAttribute("data-for")) || null });
  const add = (kind, what, other, ov) =>
    found.push({ diagram: name, kind, what, with: other || "", overlap: ov || null });
  const inside = (r, b) => r.x + r.w / 2 >= b.x && r.x + r.w / 2 <= b.x + b.w
                        && r.y + r.h / 2 >= b.y && r.y + r.h / 2 <= b.y + b.h;
  const within = (r, f) => r.x >= f.x - s && r.y >= f.y - s && r.x + r.w <= f.x + f.w + s && r.y + r.h <= f.y + f.h + s;
  for (const x of texts) {
    const r = x.r;
    if (r.x < -s || r.y < -s || r.x + r.w > vb.width + s || r.y + r.h > vb.height + s)
      add("past the edge", x.name, "the drawing's edge (" + vb.width + " x " + vb.height + ")");
    if (x.box) {
      // the room left between the text and its box's left, right, top and bottom edges
      const b = x.box.r, gaps = [r.x - b.x, b.x + b.w - r.x - r.w, r.y - b.y, b.y + b.h - r.y - r.h];
      if (gaps[0] < M.x || gaps[1] < M.x || gaps[2] < M.y || gaps[3] < M.y)
        add("past its box", x.name, x.box.name + " (room left, right, above, below: " + gaps.map(r1).join(", ")
            + "; the least allowed is " + M.x + " at the sides and " + M.y + " above and below)", gaps.map(r1));
    }
    const home = x.box || x.badgeOf || (x.inBox ? boxes.find(b => inside(r, b.r)) : null);
    for (const b of boxes) {
      if (b === home) continue;
      const ov = layoutOverlap(r, b.r, s);
      if (ov) add("over a box", x.name, b.name, ov);
    }
    for (const f of frames)
      if (layoutOverlap(r, f.r, s) && !within(r, f.r)) add("across a frame", x.name, f.name);
  }
  for (const b of boxes)
    for (const f of frames)
      if (layoutOverlap(b.r, f.r, s) && !within(b.r, f.r)) add("across a frame", b.name, f.name);
  for (let i = 0; i < texts.length; i++)
    for (let j = i + 1; j < texts.length; j++) {
      const ov = layoutOverlap(texts[i].r, texts[j].r, s);
      if (ov) add("labels overlap", texts[i].name, texts[j].name, ov);
    }
  for (let i = 0; i < boxes.length; i++)
    for (let j = i + 1; j < boxes.length; j++) {
      const ov = layoutOverlap(boxes[i].r, boxes[j].r, s);
      if (ov) add("boxes overlap", boxes[i].name, boxes[j].name, ov);
    }
  // every drawn line except a plain curve inside a box: straight segments are clipped exactly;
  // a path with an arc (the learning circle) is sampled every 2 units along its length
  for (const pth of svg.querySelectorAll("path.mini-edge:not(.plainline), path.dg-edge")) {
    const d = pth.getAttribute("d") || "";
    const segs = pathSegments(d);
    let pts = null;
    if (!segs.length && pth.getTotalLength) {
      const len = pth.getTotalLength();
      pts = [];
      for (let u = 0; u <= len; u += 2) { const pt = pth.getPointAtLength(u); pts.push([pt.x, pt.y]); }
    }
    const crosses = r => segs.length ? segs.some(([p, q]) => segmentCrosses(p, q, r))
      : pts.some(([x, y]) => x > r.x && x < r.x + r.w && y > r.y && y < r.y + r.h);
    const seg = "the line " + d.slice(0, 40);
    // a numbered circle is drawn over the lines on purpose, opaque, so a line under it is not counted
    for (const x of texts)
      if (!x.box && !x.badge && crosses(layoutShrink(x.r, 1, 2))) add("line through a label", x.name, seg);
    for (const b of boxes)
      if (crosses(layoutShrink(b.r, 3, 3))) add("line through a box", b.name, seg);
  }
  // LINES ON TOP OF EACH OTHER (operator, Oct 15, 2026: "arrows and lines are on top of each other"). Two lines of the
  // architecture diagram (path.dg-edge, each named by its data-a and data-b boxes) may share a stretch only when both
  // belong to one of the drawing's declared combs (data-bus: the world's trunk to the senses, the encoders into the
  // shared LayerNorm, the final LayerNorm's three outputs), and then only exactly on top of each other. Counted:
  //   lines run together  a horizontal (or vertical) segment of one lies less than LINE_GAP units from a parallel
  //                       segment of the other, over more than 2 units of length (a bus's two lines: closer than
  //                       LINE_GAP but not exactly on top of each other). The diagram is read at about 0.6 pixels per
  //                       unit on a 1,000-pixel-wide panel, where 8 units are 5 pixels: closer reads as one line;
  //   arrowheads meet     their end points lie less than ARROW_GAP units apart (the arrowhead is 7 units long) and
  //                       the two lines are not of one comb.
  const LINE_GAP = 8, ARROW_GAP = 9;
  const lines = Array.from(svg.querySelectorAll("path.dg-edge")).map(p => ({ a: p.getAttribute("data-a") || "",
    b: p.getAttribute("data-b") || "", bus: p.getAttribute("data-bus") || "", d: p.getAttribute("d") || "",
    segs: pathSegments(p.getAttribute("d") || "") }));
  const spanOverlap = (a0, a1, b0, b1) =>
    Math.min(Math.max(a0, a1), Math.max(b0, b1)) - Math.max(Math.min(a0, a1), Math.min(b0, b1));
  for (let i = 0; i < lines.length; i++)
    for (let j = i + 1; j < lines.length; j++) {
      const P = lines[i], Q = lines[j], bus = !!P.bus && P.bus === Q.bus;
      const pn = P.a + " -> " + P.b, qn = Q.a + " -> " + Q.b;
      let together = null;
      for (const [p0, p1] of P.segs) for (const [q0, q1] of Q.segs) {
        if (together) break;
        const ph = p0[1] === p1[1], qh = q0[1] === q1[1], pv = p0[0] === p1[0], qv = q0[0] === q1[0];
        if (ph && qh && Math.abs(p0[1] - q0[1]) < LINE_GAP && !(bus && p0[1] === q0[1])) {
          const o = spanOverlap(p0[0], p1[0], q0[0], q1[0]);
          if (o > 2) together = "along y " + p0[1] + " and " + q0[1] + " for " + r1(o) + " units";
        } else if (pv && qv && Math.abs(p0[0] - q0[0]) < LINE_GAP && !(bus && p0[0] === q0[0])) {
          const o = spanOverlap(p0[1], p1[1], q0[1], q1[1]);
          if (o > 2) together = "along x " + p0[0] + " and " + q0[0] + " for " + r1(o) + " units";
        }
      }
      if (together) add("lines run together", pn, qn + " (" + together + ")");
      const pe = P.segs.length ? P.segs[P.segs.length - 1][1] : null, qe = Q.segs.length ? Q.segs[Q.segs.length - 1][1] : null;
      if (pe && qe && !bus && Math.hypot(pe[0] - qe[0], pe[1] - qe[1]) < ARROW_GAP)
        add("arrowheads meet", pn, qn + " (at " + pe.join(",") + " and " + qe.join(",") + ")");
    }
  // A LINE THROUGH ANOTHER LINE'S ARROWHEAD (the independent review of Oct 15, 2026, S3: the persistent memory -> replay
  // line crossed two arrows into self perception's top edge 3 units above their arrowheads' bases, so it read as
  // feeding self perception). Counted: a segment of one line that meets the last TAIL_GAP units of another line's last
  // segment (the 7-unit arrowhead and 5 units before it), at a point other than that line's tip, the two lines not of
  // one comb. A crossing exactly TAIL_GAP units from the tip is outside.
  const TAIL_GAP = 12;
  const hitPoint = (p0, p1, q0, q1) => {
    const d = (p1[0] - p0[0]) * (q1[1] - q0[1]) - (p1[1] - p0[1]) * (q1[0] - q0[0]);
    if (Math.abs(d) < 1e-9) return null;              // parallel: "lines run together" judges these
    const t = ((q0[0] - p0[0]) * (q1[1] - q0[1]) - (q0[1] - p0[1]) * (q1[0] - q0[0])) / d;
    const u = ((q0[0] - p0[0]) * (p1[1] - p0[1]) - (q0[1] - p0[1]) * (p1[0] - p0[0])) / d;
    if (t < -1e-9 || t > 1 + 1e-9 || u < -1e-9 || u > 1 + 1e-9) return null;
    return [p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])];
  };
  for (const Q of lines) {
    if (!Q.segs.length) continue;
    const [q0, q1] = Q.segs[Q.segs.length - 1], len = Math.hypot(q1[0] - q0[0], q1[1] - q0[1]);
    if (!len) continue;
    const k = Math.min(TAIL_GAP, len) / len, t0 = [q1[0] - (q1[0] - q0[0]) * k, q1[1] - (q1[1] - q0[1]) * k];
    for (const P of lines) {
      if (P === Q || (P.bus && P.bus === Q.bus)) continue;
      let at = null;
      for (const [p0, p1] of P.segs) {
        const h = hitPoint(p0, p1, t0, q1);
        const dist = h ? Math.hypot(h[0] - q1[0], h[1] - q1[1]) : null;
        if (h && dist > 0.5 && dist < TAIL_GAP - 1e-6) { at = h; break; }
      }
      if (at) add("line crosses an arrowhead", P.a + " -> " + P.b,
                  Q.a + " -> " + Q.b + " (at " + at.map(r1).join(",") + ", within " + TAIL_GAP + " units of its tip)");
    }
  }
  return { boxes: boxes.length, texts: texts.length, found };
}
function layoutCheck() {
  const out = { diagrams: [], problems: [], counts: {} };
  switchTab("arch");
  // the architecture diagram as boot drew it, in its own tab
  const arch = document.querySelector("#diagram svg");
  const holder = el("div", { id: "layout-holder", style: "width:1200px" });
  document.body.appendChild(holder);
  const measure = (name, svg) => {
    const r = layoutProblems(name, svg);
    out.diagrams.push({ name, width: svg.viewBox.baseVal.width, height: svg.viewBox.baseVal.height,
                        boxes: r.boxes, texts: r.texts, problems: r.found.length });
    out.problems.push(...r.found);
  };
  for (const id of Object.keys(DETAIL)) {
    try {
      const svg = DETAIL[id]();
      holder.appendChild(svg);
      measure(id, svg);
      holder.removeChild(svg);
    } catch (e) {
      out.problems.push({ diagram: id, kind: "not drawn", what: String((e && e.message) || e), with: "" });
    }
  }
  // the learning circle and the architecture diagram are measured where boot drew them, at one unit
  // per pixel - the size a 1,600-pixel-wide window draws the architecture diagram at. The browser lays
  // text out in whole device pixels, so in the check's own headless window (754 pixels wide, the
  // architecture diagram at 0.48 pixels per unit) a caption's and its sub-caption's text boxes
  // overlapped by 2.6 units where at one unit per pixel they overlap by 1.0 (measured 2026-10-02)
  const atFullSize = (name, svg) => {
    const keep = svg.getAttribute("style");
    svg.setAttribute("style", "width:" + svg.viewBox.baseVal.width + "px;max-width:none;min-width:0");
    measure(name, svg);
    if (keep === null) svg.removeAttribute("style"); else svg.setAttribute("style", keep);
  };
  const loop = document.querySelector("#learnloop svg");
  if (loop) {
    const detail = loop.closest("details"), wasOpen = detail && detail.open;
    if (detail) detail.open = true;
    atFullSize("learning circle", loop);
    if (detail) detail.open = wasOpen;
  }
  if (arch) atFullSize("architecture", arch);
  else out.problems.push({ diagram: "architecture", kind: "not drawn", what: "the diagram holder #diagram has no drawing", with: "" });
  // the same diagram with every gated part switched on, then every one switched off (the switches of Oct 15, 2026),
  // then back to this run's settings
  if (arch && typeof GATES !== "undefined") {
    for (const [label, on] of [["architecture, every switch on", true], ["architecture, every switch off", false]]) {
      GATES.forEach(g => { GATE_SET[g[0]] = on; });
      buildDiagram();
      const s = document.querySelector("#diagram svg");
      if (s) atFullSize(label, s);
    }
    Object.keys(GATE_SET).forEach(k => delete GATE_SET[k]);
    buildDiagram();
  }
  holder.remove();
  for (const p of out.problems) out.counts[p.kind] = (out.counts[p.kind] || 0) + 1;
  out.total = out.problems.length;
  document.body.appendChild(el("pre", { id: "layout-check", text: JSON.stringify(out) }));
  return out;
}

// ------------------------------------------------------------------ live reload
(function liveReload() {
  if (!META.watch) return;
  let lastTouch = Date.now();
  ["mousemove", "keydown", "pointerdown", "wheel"].forEach(evn =>
    window.addEventListener(evn, () => { lastTouch = Date.now(); }, { passive: true }));
  const every = Math.max(10, META.every || 30) * 1000;
  setInterval(() => {
    if (Date.now() - lastTouch > 20000) { writeHash(); location.reload(); }
  }, every);
})();

// ------------------------------------------------------------------ boot
window.addEventListener("DOMContentLoaded", () => {
  const nav = document.getElementById("tabs");
  for (const [id, label] of TABS) {
    nav.appendChild(el("button", { id: "btn-" + id, text: label, onclick: () => switchTab(id) }));
  }
  buildLearnLoop();
  // the model diagram's panel: its heading, the drawing (with its orange rings), the list of the parts being worked on
  // under it, and the tour between the heading and the drawing (D-E1, D-E2)
  buildModelHead();
  buildDiagram();
  buildGatePanel();
  buildWorkList();
  renderTour();
  renderInspector();
  renderRun();
  renderProbes();
  if (state.d && DETAIL[state.d]) renderDetail(state.d);
  switchTab(state.t && TABS.some(([id]) => id === state.t) ? state.t : "arch");
  // an address carrying tr= opens on that slide: the tour is scrolled into view, unless the address also names a
  // component, an investigation, an entry or a detail panel, which open where they are
  if (tourIdx >= 0 && (!state.t || state.t === "arch")
      && !state.component && !state.investigation && !state.doc && !state.d) {
    const panel = document.getElementById("model-diagram");
    if (panel && panel.scrollIntoView) panel.scrollIntoView({ block: "start" });
  }
  if (state.doc && Object.prototype.hasOwnProperty.call(DOC, docKey(state.doc))) {
    state.doc = docKey(state.doc);
    openDocOrDetail(state.doc);
  }
  // the code and QA map: the list of every component under the drawing, the component an address names
  // (#t=arch&component=ID, the Tests page's links), and the dashboard's newer view when it serves the page
  buildComponentIndex();
  // the root-cause investigations (the same boxes as the Tests page), and the one an address names
  // (#t=arch&investigation=ID, the Tests page's links)
  buildInvestigationsIndex();
  if (state.investigation) { switchTab("arch"); showInvestigation(state.investigation); }
  if (state.component) { switchTab("arch"); showComponent(state.component); }
  readComponents(redrawComponentViews);
  readInvestigations(redrawInvestigationViews);
  if (LAYOUT_CHECK) layoutCheck();
});
