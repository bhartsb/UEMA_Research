/* Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved. */
/* Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file. */
/* The movement viewer on the Now tab (tools/movement_viewer; the operator's request of Oct 18, 2026, revised after the
   operator's review of Oct 19, 2026 and the independent review that followed: one bar per run instead of a chart over
   dates, plain names for the choices and the runs, the whole body drawn from three sides like the hands, per-finger and
   per-part bars, "Pin this drawing to compare", and bars for the checkpoint probes). Reads /api/movement/*: the saved
   recordings, the per-command scores, one command at a time, the scheduled checkpoint probes, and the pictures the
   viewer service draws on request. */
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  // one color per group of control settings: runs with other settings are separate examples, never one series
  const SETTING_COLORS = { 'servo_off|arm_off': '#a0522d', 'servo_on|arm_off': '#12645e', 'servo_on|arm_memory': '#6a4c93' };
  const SETTING_WORDS = {
    'servo_off|arm_off': 'finger joint servo off: each finger is moved toward an asked-for fingertip place and direction',
    'servo_on|arm_off': 'finger joint servo on: each finger joint is turned toward its own asked-for angle',
    'servo_on|arm_memory': "servo on and arm stage memory: in addition, the planner's first command copies both arms' asked-for places and directions from the pose stored with the goal it picked",
  };
  const SPARE_COLORS = ['#275e90', '#b5651d', '#3c8d2f', '#8e3b46', '#5d6d7e'];
  const HAND_COLORS = [['index finger', '#1f7ade'], ['middle finger', '#f5941f'], ['ring finger', '#9e5cd9'], ['little finger', '#1fad8c'],
    ['thumb', '#e64d4f'], ['palm', '#b8c2d1']];
  const PART_COLORS = { 'left arm and hand': '#2970db', 'right arm and hand': '#e8871f', 'left leg': '#2e9e5c', 'right leg': '#12a3b5',
    head: '#8f45ad', trunk: '#8c6e63' };
  const FINGER_COLOR = { thumb: '#e64d4f', 'index finger': '#1f7ade', 'middle finger': '#f5941f', 'ring finger': '#9e5cd9', 'little finger': '#1fad8c' };
  const VIEW_WORDS = { front: 'From the front', left: 'From its left side', right: 'From its right side', wrist: 'Close-up of the left wrist' };
  const MEASURE_WORDS = {
    fingers_L: ['Left-hand fingers', 'how many degrees each of the left hand\'s 22 finger joints is from the angle the command asked for, averaged over the 22'],
    fingers_R: ['Right-hand fingers', 'how many degrees each of the right hand\'s 22 finger joints is from the angle the command asked for, averaged over the 22'],
    wrist_L: ['Left wrist', 'how many cm the middle of the left palm (the point the command calls the left wrist) is from the place the command asked for'],
    wrist_R: ['Right wrist', 'how many cm the middle of the right palm (the point the command calls the right wrist) is from the place the command asked for'],
    body_turn: ['Whole body', 'how many degrees each powered body part (arms, legs, feet, head, chest, hips: 15 parts) points away from the direction the command asked for, averaged over the parts'],
  };
  // the code names of the rows of a command, in words (the table of every powered part)
  const ROW_WORDS = { gaze: 'eyes (where to look)', thumb_L: 'left thumb tip (a place)', index_L: 'left index fingertip (a place)',
    middle_L: 'left middle fingertip (a place)', ring_L: 'left ring fingertip (a place)', pinky_L: 'left little fingertip (a place)',
    thumb_R: 'right thumb tip (a place)', index_R: 'right index fingertip (a place)', middle_R: 'right middle fingertip (a place)',
    ring_R: 'right ring fingertip (a place)', pinky_R: 'right little fingertip (a place)' };
  const TRY_COLOR = { witness: '#c58b2c', controller: '#7c8a90', direct: '#12645e', synergy: '#3a74b0' };
  const TRY_NAME = { witness: 'the request put straight into the command (its end is the test pose)', controller: "the motor controller's own command",
    direct: 'planner search, direct', synergy: 'planner search, synergy' };
  // how each probe run started, in words (probes.MODE_NAMES and the scheduled check)
  const MODE_WORDS = { scheduled: 'scheduled check', 'normal mode': 'run by hand, its own start',
    'frozen start': 'same start each time, searches from its own command', 'frozen start, search anchored': 'same start each time' };
  function probeLabel(x) { return dayLabel(x.time) + ', checkpoint ' + int(x.checkpoint_step) + ' (' + (MODE_WORDS[x.mode] || x.mode) + ')'; }
  const S = { sources: null, series: null, full: null, t0: null, t1: null, count: 0, index: 0, cmd: null, current: null,
    keep: null, timers: {}, playing: {}, seqSeries: 0, seqCmd: 0, userEpoch: 0, userPending: 0, rowMap: null,
    pendingCards: [], probes: null, igeom: null };
  let store = null;
  try { store = window.localStorage; } catch (e) { store = null; }

  // ------------------------------------------------------------------ small helpers
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
  function num(x, d) { return (x == null || !isFinite(x)) ? '–' : Number(x).toLocaleString('en-US', { maximumFractionDigits: d == null ? 1 : d, minimumFractionDigits: d == null ? 1 : d }); }
  function int(x) { return (x == null || !isFinite(x)) ? '–' : Math.round(x).toLocaleString('en-US'); }
  function pct(x) { return (x == null || !isFinite(x)) ? '–' : Math.round(x * 100) + '%'; }
  function sgn(x, d) { return (x == null || !isFinite(x)) ? '–' : (x > 0 ? '+' : '') + num(x, d); }
  function unitSign(u) { return u === 'degrees' ? '°' : ' ' + u; }
  function words(s) { return String(s == null ? '' : s).replace(/_/g, ' '); }
  function debounce(fn, ms) { let t = null; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
  function saved(k, v) { if (!store) return null; try { if (v === undefined) return store.getItem('mv.' + k); store.setItem('mv.' + k, v); } catch (e) { return null; } return null; }
  async function getJSON(url, opts) {
    const r = await fetch(url, Object.assign({ cache: 'no-store' }, opts || {}));
    let body = null;
    try { body = await r.json(); } catch (e) { body = null; }
    if (!r.ok) throw new Error((body && body.error) || ('the dashboard answered ' + r.status));
    return body;
  }
  // the share of a distance closed, in words: "25% of the way", or "moved farther away (the distance grew 10%)"
  function wayText(share) {
    if (share == null || !isFinite(share)) return '–';
    return share >= 0 ? pct(share) + ' of the way' : 'moved farther away (the distance grew ' + pct(-share) + ')';
  }
  // a small bar with a zero mark at ZERO_AT of its width: a share closed grows right from the mark (100% reaches the
  // right end); a distance that grew is drawn red, growing left from the mark (a growth of 100% or more reaches the left end)
  const ZERO_AT = 22;
  function miniBar(share, color) {
    if (share == null || !isFinite(share)) return '<span class="mv-mini none"><b style="left:' + ZERO_AT + '%"></b></span>';
    const w = share >= 0 ? Math.min(1, share) * (100 - ZERO_AT) : Math.min(1, -share) * ZERO_AT;
    const left = share >= 0 ? ZERO_AT : ZERO_AT - w;
    return '<span class="mv-mini"><i class="' + (share < 0 ? 'away' : '') + '" style="left:' + left.toFixed(1) + '%;width:' + w.toFixed(1) + '%' + (share >= 0 && color ? ';background:' + color : '') + '"></i><b style="left:' + ZERO_AT + '%"></b></span>';
  }
  const BAR_KEY = '<span><i class="mv-key-bar"></i>bar to the right of the mark: share of the way made (the mark = no closer, the right end = got there)</span><span><i class="mv-key-bar away"></i>red bar to the left of the mark: moved farther away</span>';
  // Pacific time
  const TZ = { timeZone: 'America/Los_Angeles' };
  function when(t, withYear) {
    if (t == null) return '–';
    return new Date(t * 1000).toLocaleString('en-US', Object.assign({ month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', timeZoneName: 'short' }, withYear ? { year: 'numeric' } : {}, TZ));
  }
  function dayLabel(t) { return new Date(t * 1000).toLocaleString('en-US', Object.assign({ month: 'short', day: 'numeric' }, TZ)); }
  function clockLabel(t) { return new Date(t * 1000).toLocaleString('en-US', Object.assign({ hour: 'numeric', minute: '2-digit' }, TZ)); }

  // ------------------------------------------------------------------ choices
  function filters() {
    return { measure: $('mv-measure').value, body: $('mv-body').value, author: $('mv-author').value, kind: $('mv-kind').value, touch: $('mv-touch').checked ? 'hand' : 'any' };
  }
  function query(extra) {
    const f = filters();
    const one = f.kind.startsWith('rec:');
    const q = new URLSearchParams({ measure: f.measure, body: f.body, author: f.author, kind: one ? 'all' : f.kind, touch: f.touch });
    if (one) q.set('rec', f.kind.slice(4));
    if (S.t0 != null) q.set('t0', String(S.t0));
    if (S.t1 != null) q.set('t1', String(S.t1));
    Object.entries(extra || {}).forEach(([k, v]) => { if (v != null) q.set(k, String(v)); });
    return q.toString();
  }
  function measureInfo() { const m = S.sources && S.sources.measures[$('mv-measure').value]; return m || { name: '', unit: '' }; }
  function measureWords() { return MEASURE_WORDS[$('mv-measure').value] || [measureInfo().name, measureInfo().definition || '']; }
  function settingsColor(key) {
    if (SETTING_COLORS[key]) return SETTING_COLORS[key];
    const keys = [...new Set(((S.sources && S.sources.recordings) || []).map((r) => (r.settings || {}).key))].filter((k) => !SETTING_COLORS[k]).sort();
    return SPARE_COLORS[Math.max(0, keys.indexOf(key)) % SPARE_COLORS.length];
  }
  function settingsWords(key, label) { return SETTING_WORDS[key] || label || 'settings not recorded'; }
  function recordingsOfKind() {
    const kind = $('mv-kind').value;
    const recs = S.sources ? S.sources.recordings : [];
    if (kind.startsWith('rec:')) return recs.filter((r) => r.id === kind.slice(4));
    return recs.filter((r) => kind === 'all' || r.kind === kind);
  }
  function recordingById(id) { return ((S.sources && S.sources.recordings) || []).find((r) => r.id === id) || null; }
  // a run's name and description in words (sources.about_of: training runs by segment, tests with their registry title)
  function runName(id, label) { const r = recordingById(id); return (r && r.about && r.about.name) || label || ''; }
  function runAbout(id) { const r = recordingById(id); return (r && r.about && r.about.words) || ''; }
  // a command's source in words, from its author code (a drawing's manifest) or its author name (a scored command)
  function authorName(x) {
    const n = /^\d+$/.test(String(x)) ? ((S.sources && S.sources.authors && S.sources.authors[String(x)]) || 'source ' + x) : String(x || 'an unknown source');
    return { planner: 'the planner', reflex: 'the motor controller', 'babble noise': 'the motor controller with exploration noise' }[n] || n;
  }
  function kindWord(k) { return k === 'training' ? 'training run' : 'test run'; }
  // a run as a reader names it: "training segment 41u from 4:34 PM" or "test run POSE_A_off_494025"
  function runTitle(id, label, kind) { const n = runName(id, label); return kind === 'training' ? n : 'test run ' + n; }

  // ------------------------------------------------------------------ status and dates
  function statusLine() {
    const el = $('mv-status');
    const recs = (S.sources && S.sources.recordings) || [];
    if (!recs.length) { el.textContent = 'No saved recording was found.'; return; }
    const tr = recs.filter((r) => r.kind === 'training').length;
    const cmds = recs.reduce((a, r) => a + (r.commands || 0), 0);
    const t0 = Math.min(...recs.map((r) => r.first_time)), t1 = Math.max(...recs.map((r) => r.last_time));
    const sv = S.sources.service || {};
    let svText;
    if (sv.alive) {
      const bg = sv.background || {};
      svText = 'The drawing service is running' + (bg.at ? '; it last looked for new records ' + when(bg.at) : '') + '.';
      if (sv.paused) svText = 'The drawing service is paused (' + (sv.paused_by || 'service_control.json') + ').';
      if (sv.guard && sv.guard.allowed === false) svText += ' Drawings wait: the memory guard refused (' + sv.guard.reading + ').';
    } else {
      svText = 'The drawing service is not answering; a drawing request starts it.';
    }
    el.textContent = recs.length + ' saved runs (' + tr + ' training runs, ' + (recs.length - tr) + ' tests) from ' + dayLabel(t0) + ' to ' + dayLabel(t1) + ': ' + int(cmds) + ' commands scored. ' + svText;
    el.classList.toggle('warn', !sv.alive);
  }
  function fullRange() {
    const recs = recordingsOfKind();
    if (!recs.length) return null;
    return [Math.floor(Math.min(...recs.map((r) => r.first_time))), Math.ceil(Math.max(...recs.map((r) => r.last_time)))];
  }
  function setRange(full) {
    S.full = full;
    const from = $('mv-from'), to = $('mv-to');
    // the handles move in 60 s steps from the first command; the top of the scale is rounded up onto that grid, so the
    // end handle can return to its end (a handle at an end sends no bound)
    const top = full[0] + Math.ceil((full[1] - full[0]) / 60) * 60;
    [from, to].forEach((el) => { el.min = String(full[0]); el.max = String(top); el.step = '60'; });
    from.value = String(S.t0 != null ? Math.max(full[0], S.t0) : full[0]);
    to.value = String(S.t1 != null ? Math.min(top, S.t1) : top);
    rangeText();
  }
  function rangeText() {
    const b = Number($('mv-to').value);
    $('mv-range-text').textContent = when(Number($('mv-from').value)) + ' to ' + when(S.full && b >= Number($('mv-to').max) ? S.full[1] : b);
  }

  // ------------------------------------------------------------------ one bar per run
  function measureLine() {
    const [name, words_] = measureWords();
    // the three pick buttons sort by this body part: say so above them
    const part = name.toLowerCase();
    $('mv-pick-by').textContent = 'These buttons sort the commands chosen above by the ' + part + (part.endsWith('s') ? '\'' : '\'s') +
      ' distance from what was asked for at each command\'s end (set by "Body part" at the top). Change "Body part" to sort by another part.';
    $('mv-measure-def').innerHTML = '<b>' + esc(name) + '</b>: the distance is ' + esc(words_) + '. Each bar shows how much of that distance the run\'s commands closed during the command (250 ms): <b>0%</b> = the body ended no closer to what was asked for than it began, <b>100%</b> = it got exactly there. What share counts as working has not been measured: no test has set the share a working controller reaches in 250 ms.';
  }
  function bars() {
    const box = $('mv-bars');
    const ser = ((S.series && S.series.series) || []).slice().sort((a, b) => a.bins[0].t - b.bins[0].t);
    const shown = new Set(ser.map((s) => s.recording));
    const missing = $('mv-kind').value.startsWith('rec:') ? [] : recordingsOfKind().filter((r) => !shown.has(r.id));
    $('mv-missing-runs').innerHTML = missing.length ? 'Not shown: ' + missing.map((r) => esc(runName(r.id, r.label)) + ' (' + dayLabel(r.first_time) + ')').join(', ') + ': none of ' + (missing.length === 1 ? 'its' : 'their') + ' commands matches these choices.' : '';
    if (!ser.length) { box.innerHTML = '<p class="mv-empty">No scored commands match these choices.</p>'; return; }
    const vals = ser.map((s) => s.summary.gap_closed).filter((v) => v != null && isFinite(v));
    const lo = Math.min(0, ...vals);
    const span = 1 - lo;
    const z = (-lo / span) * 100;
    const X = (v) => ((v - lo) / span) * 100;
    const ticks = [0, 0.25, 0.5, 0.75, 1].concat(lo < -0.2 ? [-0.25] : []);
    let h = '<div class="mv-bar-head"><span></span><span class="mv-axis-row">' + ticks.map((t) => '<b style="left:' + X(t).toFixed(1) + '%">' + Math.round(t * 100) + '%</b>').join('') + '</span><span></span></div>';
    const one = ser.length === 1;
    h += ser.map((s) => {
      const v = s.summary.gap_closed;
      const c = settingsColor(s.settings_key);
      const left = v == null ? z : v >= 0 ? z : X(v);
      const width = v == null ? 0 : Math.abs(X(v) - z);
      const t = s.bins[0].t;
      const about = runAbout(s.recording);
      return '<button type="button" class="mv-bar-row' + (one ? ' only' : '') + '" data-rec="' + esc(s.recording) + '" title="' + esc(one ? 'This run is shown alone' : 'Look inside this run') + '">' +
        '<span class="mv-bar-label"><b>' + esc(dayLabel(t)) + '</b> ' + esc(runName(s.recording, s.label)) + ' <em>' + (s.kind === 'training' ? 'training' : 'test') + '</em>' + (about ? '<small>' + esc(about) + '</small>' : '') + '</span>' +
        '<span class="mv-track">' + ticks.map((tk) => '<i class="grid" style="left:' + X(tk).toFixed(1) + '%"></i>').join('') +
        '<i class="bar' + (v != null && v < 0 ? ' away' : '') + '" style="left:' + left.toFixed(2) + '%;width:' + width.toFixed(2) + '%;' + (v != null && v >= 0 ? 'background:' + c : '') + '"></i></span>' +
        '<span class="mv-bar-value">' + (v == null ? '–' : pct(v)) + '<small>' + int(s.summary.with_start) + ' commands</small></span></button>';
    }).join('');
    box.innerHTML = h;
  }
  function reading() {
    const el = $('mv-reading');
    const s = S.series && S.series.summary;
    if (!s || !s.commands) { el.textContent = ''; return; }
    const u = unitSign(measureInfo().unit);
    const groups = new Map();
    ((S.series && S.series.series) || []).forEach((x) => { if (!groups.has(x.settings_key)) groups.set(x.settings_key, []); groups.get(x.settings_key).push(x); });
    const sets = S.series.summary_by_settings || [];
    const who = $('mv-author').value === 'planner' ? 'chosen by the planner' : 'from every source';
    el.innerHTML = '<p>Read each color on its own: runs with different settings are not one series.</p><ul>' +
      sets.map((g) => {
        const runs = (groups.get(g.settings_key) || []).map((x) => x.summary.gap_closed).filter((v) => v != null);
        const range = runs.length > 1 ? pct(Math.min(...runs)) + ' to ' + pct(Math.max(...runs)) + ' of the way in its ' + runs.length + ' runs' : pct(runs[0]) + ' of the way in its run';
        return '<li><i class="mv-swatch" style="background:' + settingsColor(g.settings_key) + '"></i><b>' + esc(g.settings) + '</b>: ' + range +
          '. A typical command: ' + num(g.summary.median_start) + u + ' from what was asked for before it, ' + num(g.summary.median_end) + u + ' after it.</li>';
      }).join('') + '</ul><p class="small muted">Counted: ' + int(s.with_start) + ' commands ' + esc(who) + '. Each body\'s first command in a run has no saved start and is left out.</p>';
  }
  function legend() {
    const ser = (S.series && S.series.series) || [];
    const groups = new Map();
    ser.forEach((s) => groups.set(s.settings_key, s.settings));
    $('mv-legend').innerHTML = '<span class="mv-legend-head">Bar colors (the run\'s control setting):</span>' + [...groups.entries()].map(([k, label]) => '<span><i class="mv-swatch" style="background:' + settingsColor(k) + '"></i>' + esc(settingsWords(k, label)) + '</span>').join('') +
      '<span>Runs are listed by date, oldest first. Click a run to look inside it.</span>';
  }

  // ------------------------------------------------------------------ inside one run: the share over time
  function inside() {
    const wrap = $('mv-inside');
    const ser = (S.series && S.series.series) || [];
    if (ser.length !== 1) { wrap.hidden = true; S.igeom = null; return; }
    wrap.hidden = false;
    const s = ser[0];
    const bins = s.bins.filter((b) => b.closed != null);
    const per = bins.length ? Math.round(bins.reduce((a, b) => a + b.with_start, 0) / bins.length) : 0;
    $('mv-inside-title').textContent = 'Inside ' + runName(s.recording, s.label) + ' (' + kindWord(s.kind) + '): share of the way over time';
    const svg = $('mv-inside-chart');
    const W = svg.clientWidth || 900, H = svg.clientHeight || 220;
    const m = { l: 44, r: 14, t: 12, b: 30 };
    if (!bins.length) { svg.innerHTML = '<text x="' + W / 2 + '" y="' + H / 2 + '" text-anchor="middle">No command of this run has a start pose.</text>'; S.igeom = null; return; }
    const lo = Math.min(0, ...bins.map((b) => b.closed));
    const tA = bins[0].t0, tB = Math.max(bins[bins.length - 1].t1, tA + 60);
    const X = (t) => m.l + (W - m.l - m.r) * (t - tA) / (tB - tA);
    const Y = (v) => m.t + (H - m.t - m.b) * (1 - (v - lo) / (1 - lo));
    const parts = [];
    [0, 0.25, 0.5, 0.75, 1].concat(lo < -0.2 ? [-0.25] : []).forEach((v) => {
      parts.push('<line class="' + (v === 0 ? 'mv-axis' : 'mv-grid') + '" x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(v) + '" y2="' + Y(v) + '"/><text x="' + (m.l - 6) + '" y="' + (Y(v) + 4) + '" text-anchor="end">' + Math.round(v * 100) + '%</text>');
    });
    const nTicks = Math.max(2, Math.min(6, Math.floor((W - m.l - m.r) / 110)));
    for (let k = 0; k <= nTicks; k++) {
      const t = tA + (tB - tA) * k / nTicks;
      parts.push('<text x="' + X(t) + '" y="' + (H - 9) + '" text-anchor="' + (k === 0 ? 'start' : k === nTicks ? 'end' : 'middle') + '">' + esc(k === 0 ? dayLabel(t) + ' ' + clockLabel(t) : clockLabel(t)) + '</text>');
    }
    const c = settingsColor(s.settings_key);
    parts.push('<polyline class="mv-line" stroke="' + c + '" points="' + bins.map((b) => X(b.t) + ',' + Y(b.closed)).join(' ') + '"/>');
    const pts = [];
    bins.forEach((b) => { parts.push('<circle fill="' + c + '" r="3.4" cx="' + X(b.t) + '" cy="' + Y(b.closed) + '"/>'); pts.push({ x: X(b.t), b: b }); });
    // the observer's events inside this run, numbered and named below the chart
    S.markShown = [];
    ((S.series.markers) || []).filter((mk) => mk.time != null && mk.time >= tA && mk.time <= tB).forEach((mk) => {
      S.markShown.push(mk);
      parts.push('<line class="mv-marker" x1="' + X(mk.time) + '" x2="' + X(mk.time) + '" y1="' + m.t + '" y2="' + (H - m.b) + '"/><text class="mv-marker-label" x="' + (X(mk.time) + 3) + '" y="' + (m.t + 10) + '">' + S.markShown.length + '</text>');
    });
    const picked = S.cmd && S.cmd.recording === s.recording;
    if (picked) parts.push('<line class="mv-sel" x1="' + X(S.cmd.time) + '" x2="' + X(S.cmd.time) + '" y1="' + m.t + '" y2="' + (H - m.b) + '"/>');
    svg.innerHTML = '<title>' + esc($('mv-inside-title').textContent) + '</title>' + parts.join('');
    S.igeom = { pts: pts, W: W, tA: tA, tB: tB, m: m };
    const recs = recordingsOfKind();
    const undated = ((S.series && S.series.markers) || []).filter((mk) => mk.time == null && recs.some((r) => r.first_step <= mk.step && mk.step <= r.last_step));
    $('mv-inside-notes').innerHTML = '<p>Each dot: ' + int(per) + ' commands in a row, read as on the bars (0% = no closer, 100% = got there). Click the chart to pick the command nearest that time.' + (picked ? ' The black line marks the command picked below.' : '') + (S.markShown.length ? ' Dashed lines: the observer\'s events, numbered: ' + S.markShown.map((mk, i) => (i + 1) + ' = ' + esc(mk.name) + ', ' + esc(mk.stage_name || mk.stage) + ' (model step ' + int(mk.step) + ')').join('; ') + '.' : '') + '</p>' +
      (undated.length ? '<p>Observer events inside this run\'s model steps that cannot be dated here: ' + undated.map((mk) => esc(mk.name) + ', ' + esc(mk.stage_name || mk.stage) + ' (model step ' + int(mk.step) + '): ' + esc(mk.why || 'no date')).join('; ') + '.</p>' : '');
  }
  function insideTip(evt) {
    const g = S.igeom, el = $('mv-itip');
    if (!g) { el.hidden = true; return; }
    const box = $('mv-inside-chart').getBoundingClientRect();
    const x = (evt.clientX - box.left) * (g.W / box.width);
    let best = null, d = Infinity;
    g.pts.forEach((p) => { const dd = Math.abs(p.x - x); if (dd < d) { d = dd; best = p; } });
    if (!best || d > 30) { el.hidden = true; return; }
    const b = best.b, u = unitSign(measureInfo().unit);
    el.innerHTML = '<b>' + esc(when(b.t0)) + ' to ' + esc(when(b.t1)) + '</b><br>' + int(b.with_start) + ' commands counted, model steps ' + int(b.step0) + ' to ' + int(b.step1) + '<br><b>' + wayText(b.closed) + '</b><br>a typical command: ' + num(b.start) + u + ' from what was asked for at the start, ' + num(b.end) + u + ' at the end<br><span style="opacity:.8">Click to pick the command nearest this time.</span>';
    const wrap = $('mv-inside-chart').parentElement.getBoundingClientRect();
    el.style.left = Math.max(0, Math.min(box.width - 300, (best.x / g.W) * box.width + 12)) + 'px';
    el.style.top = Math.max(0, Math.min(evt.clientY - wrap.top - 20, wrap.height - 120)) + 'px';
    el.hidden = false;
  }

  // ------------------------------------------------------------------ choosing a command
  function readout() {
    const el = $('mv-readout');
    const c = S.cmd;
    const u = unitSign(measureInfo().unit);
    if (!c) { el.textContent = S.count ? '' : 'No command matches these choices.'; $('mv-pos').textContent = ''; return; }
    if (S.count) $('mv-pos').textContent = posText(S.index);
    const share = (c.start != null && c.start > 0) ? (c.start - c.end) / c.start : null;
    const name = measureWords()[0];
    el.innerHTML = '<b>' + esc(when(c.time, true)) + '</b> · ' + esc(runTitle(c.recording, c.label, c.kind)) + ' · body ' + c.body + ' · model step ' + int(c.step) + ' · chosen by ' + esc(authorName(c.author)) + '.<br>' +
      esc(name) + ': ' + (c.start != null ? num(c.start) + u + ' from what was asked for before the command, ' : 'no saved start pose (the first command of its run); ') + '<b>' + num(c.end) + u + ' at its end</b>' + (share != null ? ' — <b>' + wayText(share) + '</b>' : '') +
      (c.next != null ? '; ' + num(c.next) + u + ' from this command\'s asked-for values when the next command ended, 250 ms later (each command keeps 65% of the one before, so this shows whether the body kept the change)' : '') + '.' +
      (c.hand_touch ? ' A hand touched an object during it' + (c.lift_cm != null ? '; the touched object rose ' + num(c.lift_cm) + ' cm' : '') + '.' : '') +
      (c.drawn ? ' <span class="muted">Already drawn.</span>' : '');
  }
  // the picker's position, with the commands without a saved start named (the reading counts only the others)
  function posText(i) {
    const sum = (S.series && S.series.summary) || {};
    const none = sum.commands != null && sum.with_start != null ? sum.commands - sum.with_start : 0;
    return int(i + 1) + ' of ' + int(S.count) + ', sorted by time' + (none > 0 && sum.commands === S.count ? ' (' + int(none) + ' of them, each body\'s first command in a run, have no saved start and are not counted in the shares above)' : '');
  }
  async function loadCommand(extra, byPerson) {
    const my = ++S.seqCmd;
    if (byPerson) { S.userEpoch++; S.userPending++; }
    try {
      const r = await getJSON('/api/movement/command?' + query(extra));
      if (my !== S.seqCmd) return;
      S.count = r.count || 0;
      S.cmd = r.command || null;
      S.index = r.index || 0;
      const sl = $('mv-slider');
      sl.max = String(Math.max(0, S.count - 1));
      sl.value = String(S.index);
      sl.disabled = !S.count;
      $('mv-render').disabled = !S.cmd;
      readout();
      inside();
    } catch (e) {
      $('mv-readout').textContent = 'The command could not be read: ' + e.message;
    } finally {
      if (byPerson) S.userPending--;
    }
  }
  async function loadSeries(keepCommand) {
    const my = ++S.seqSeries;
    const epoch = S.userEpoch;
    $('mv-reading').textContent = 'Reading the scores.';
    measureLine();
    try {
      const r = await getJSON('/api/movement/series?' + query());
      if (my !== S.seqSeries) return;
      S.series = r;
      bars();
      reading();
      legend();
      inside();
      $('mv-all-runs').hidden = !$('mv-kind').value.startsWith('rec:');
      if (epoch !== S.userEpoch || S.userPending > 0) { loadAuto(); return; }
      const inRange = S.cmd && (S.t0 == null || S.cmd.time >= S.t0) && (S.t1 == null || S.cmd.time <= S.t1);
      if (keepCommand && inRange) await loadCommand({ keep_rec: S.cmd.recording, keep_body: S.cmd.body, keep_step: S.cmd.step, at: S.cmd.time });
      else await loadCommand({ index: 0, pick: 'median' });
      loadAuto();
    } catch (e) {
      $('mv-reading').textContent = 'The scores could not be read: ' + e.message;
    }
  }
  function chooseRun(rid) {
    const sel = $('mv-kind');
    if (![...sel.options].some((o) => o.value === 'rec:' + rid)) return;
    sel.value = 'rec:' + rid;
    sel.dispatchEvent(new Event('change'));
  }

  // ------------------------------------------------------------------ the drawings
  function img(key, name, alt, cls, lazy) { return '<img class="' + (cls || '') + '"' + (lazy ? ' loading="lazy"' : '') + ' alt="' + esc(alt) + '" src="/api/movement/image?key=' + encodeURIComponent(key) + '&name=' + encodeURIComponent(name) + '">'; }
  function missing(text, cls) { return '<div class="mv-missing ' + (cls || '') + '">' + esc(text) + '</div>'; }
  function sampleName(ms) { return 't' + String(ms).padStart(3, '0'); }
  // the index of a drawn sample in the manifest's per-sample lists (an automatic drawing draws only the last one)
  function sampleIndex(man, ms) { const all = man.all_samples_ms || man.samples_ms || []; const k = all.indexOf(ms); return k < 0 ? all.length - 1 : k; }
  // a share needs an asked-for change of at least ASKED_MIN degrees: a joint asked to move less than that gives
  // shares like "the distance grew 1,366%" from a fraction of a degree (the independent review of Oct 19, 2026)
  const ASKED_MIN = 2;
  // joints' share at a sample: (sum of |start - asked-for| - sum of |now - asked-for|) / sum of |start - asked-for|
  function shareOf(joints, k) {
    let s = 0, e = 0, n = 0;
    joints.forEach((j) => {
      const now = (j.samples || [])[k];
      if (j.start == null || j.target == null || now == null) return;
      s += Math.abs(j.start - j.target);
      e += Math.abs(now - j.target);
      n += 1;
    });
    return n ? { share: s >= ASKED_MIN ? (s - e) / s : null, start: s / n, now: e / n, asked: s } : null;
  }
  function fingerRows(man, side, k, ms) {
    const per = ((man.fingers || {})[side]) || [];
    if (!per.length) return '';
    const all = [].concat(...per.map((f) => f.joints));
    const whole = shareOf(all, k);
    const rows = per.map((f) => {
      const r = shareOf(f.joints, k);
      const ok = r && r.share != null;
      return '<li><div class="mv-line1"><span class="mv-fname"><i class="mv-swatch" style="background:' + FINGER_COLOR[f.finger] + '"></i>' + esc(f.finger) + '</span>' +
        (ok ? miniBar(r.share, FINGER_COLOR[f.finger]) + '<span class="mv-fval">' + wayText(r.share) + '</span>' : miniBar(null) + '<span class="mv-fval muted">asked to move under ' + ASKED_MIN + '° in all, not scored</span>') + '</div></li>';
    }).join('');
    const table = '<details class="mv-joints"><summary>Each finger joint (degrees)</summary><div class="mv-table-wrap"><table class="mv-table"><thead><tr><th>Joint</th><th>start</th><th>asked for</th><th>after ' + ms + ' ms</th><th>share</th></tr></thead><tbody>' +
      per.map((f) => f.joints.map((j, i) => {
        const now = (j.samples || [])[k];
        const d0 = j.start != null && j.target != null ? Math.abs(j.start - j.target) : null;
        const sh = d0 != null && now != null && d0 >= ASKED_MIN ? (d0 - Math.abs(now - j.target)) / d0 : null;
        return '<tr><td>' + (i === 0 ? '<b>' + esc(f.finger) + '</b> ' : '') + esc(j.name) + '</td><td>' + num(j.start) + '°</td><td>' + num(j.target) + '°</td><td>' + num(now) + '°</td><td>' + (sh == null ? (d0 != null && d0 < ASKED_MIN ? 'under ' + ASKED_MIN + '° asked' : '–') : (sh >= 0 ? pct(sh) : '<span class="mv-away">farther, ' + pct(-sh) + '</span>')) + '</td></tr>';
      }).join('')).join('') + '</tbody></table></div><p class="small muted">Share: the share of the way the joint made, (how far it began from the asked-for angle - how far it is now) / how far it began; "farther" = it moved away by that share of its starting distance; "under 2° asked" = not scored. Positive angles bend a finger toward the palm.</p></details>';
    return '<p class="mv-hand-sum">' + (whole && whole.share != null ? 'All fingers together: <b>' + wayText(whole.share) + '</b> (on average ' + num(whole.start) + '° from the asked-for angles at the start, ' + num(whole.now) + '° after ' + ms + ' ms)' : 'No finger change asked for in this command') + '</p><ul class="mv-fingers">' + rows + '</ul>' + table;
  }
  function partRows(man, k, ms) {
    const keep = (man.handles || []).filter((h) => h.powered && h.scored && h.turn_deg && h.turn_deg.samples);
    if (!keep.length) return '';
    const order = ['trunk', 'head', 'left arm and hand', 'right arm and hand', 'left leg', 'right leg'];
    keep.sort((a, b) => order.indexOf(a.group) - order.indexOf(b.group));
    let s0 = 0, s1 = 0;
    keep.forEach((h) => { s0 += h.turn_deg.start || 0; s1 += h.turn_deg.samples[k] || 0; });
    const whole = s0 > 0 ? (s0 - s1) / s0 : null;
    const rows = keep.map((h) => {
      const a = h.turn_deg.start, b = h.turn_deg.samples[k];
      const sh = a != null && b != null && a >= ASKED_MIN ? (a - b) / a : null;
      let place = '';
      if (h.position_row && h.position_cm && h.position_cm.samples) {
        const pa = h.position_cm.start, pb = h.position_cm.samples[k];
        const ps = pa != null && pb != null && pa > 0 ? (pa - pb) / pa : null;
        place = '<div class="mv-line2">its place (the dot): ' + num(pa) + ' → ' + num(pb) + ' cm away, ' + wayText(ps) + '</div>';
      }
      return '<li><div class="mv-line1"><span class="mv-fname"><i class="mv-swatch" style="background:' + (PART_COLORS[h.group] || '#888') + '"></i>' + esc(h.part || words(h.name)) + '</span>' + miniBar(sh, PART_COLORS[h.group]) +
        '<span class="mv-fval">direction: ' + num(a) + '° → ' + num(b) + '° away, ' + (sh == null ? 'asked to turn under ' + ASKED_MIN + '°' : wayText(sh)) + '</span></div>' + place + '</li>';
    }).join('');
    return '<p class="mv-hand-sum">' + keep.length + ' body parts were asked to point in a direction: together <b>' + wayText(whole) + '</b> (on average ' + num(s0 / keep.length) + '° away from the asked-for direction at the start, ' + num(s1 / keep.length) + '° after ' + ms + ' ms).</p>' +
      '<ul class="mv-fingers mv-parts">' + rows + '</ul><p class="small muted">Direction: the angle between where the part points and where the command asked it to point, at the start → after ' + ms + ' ms. Place: the distance from the part to the dot the command asked it to be at.</p>';
  }
  // the asked-for body (directions only) against the asked-for places (the dots), in this drawing
  function placesText(man) {
    const pl = man.asked_for_places || [];
    if (!pl.length) return '';
    return ' Here the directions and the places do not fit together: in the asked-for body, ' + pl.map((x) => esc(x.part || words(x.name)) + ' ' + num(x.asked_for_body_cm) + ' cm' + (x.start_cm != null ? ' (' + num(x.start_cm) + ' cm at the start)' : '')).join(', ') + ' from its dot.';
  }
  function resultHTML(man, slot) {
    const id = 'mvr' + slot;
    const probe = man.kind === 'probe';
    const samples = man.samples_ms || [];
    const view = (!man.lite && S['view' + slot]) || 'front';
    const ti = S['t' + slot] != null ? Math.min(S['t' + slot], samples.length - 1) : samples.length - 1;
    const ms = samples[ti];
    const k = sampleIndex(man, ms);
    const stepAt = (man.sample_steps || [])[ti];
    const ims = man.images || {};
    const has = (n) => Object.prototype.hasOwnProperty.call(ims, n);
    const key = man.key;
    const req = man.request || null;
    const cell = (name, alt, cls, missText, caption) => '<figure>' + (has(name) ? img(key, name, alt, cls) : missing(missText || 'Not saved', cls)) + (caption ? '<figcaption>' + caption + '</figcaption>' : '') + '</figure>';
    const handBlock = (side, label) => {
      const st = 'hand_' + side + '_' + view + '_';
      return '<div class="mv-hand"><h4>' + label + '</h4><div class="mv-trio">' +
        cell(st + 'start.png', label + ' at the start', '', 'Start not saved', 'Start') +
        cell(st + 'target.png', label + (probe ? ' in the test pose' : ' as asked for'), '', man.has_start ? 'This command carried no finger angles' : 'Start not saved', probe ? 'Test pose' : 'Asked for') +
        cell(st + sampleName(ms) + '.png', label + ' after ' + ms + ' ms, the asked-for shape pale over it', '', 'Not saved', 'Got to, ' + ms + ' ms') + '</div>' +
        (probe ? '' : fingerRows(man, side, k, ms)) + '</div>';
    };
    const views = man.body_views || ['front'];
    const noBodyStart = probe && man.body_start_saved === false;
    const bodyCols = noBodyStart ? ['target', 'got'] : ['start', 'target', 'got'];
    const colName = { start: 'Start', target: probe ? 'Test pose' : 'Asked for <small>(directions)</small>', got: 'Got to, ' + ms + ' ms <small>(asked for: pale)</small>' };
    const bodyCell = (v, c) => {
      const w = (VIEW_WORDS[v] || v).toLowerCase();
      if (c === 'start') return cell('body_' + v + '_start.png', 'Whole body at the start, ' + w, 'body', 'Start not saved');
      if (c === 'target') return cell('body_' + v + '_target.png', 'Whole body as asked for, ' + w, 'body', 'Start not saved');
      return cell('body_' + v + '_' + sampleName(ms) + '.png', 'Whole body after ' + ms + ' ms, the asked-for body pale over it, ' + w, 'body', 'Not saved');
    };
    const bodyBlock = '<div class="mv-body-grid cols' + bodyCols.length + '"><span></span>' + bodyCols.map((c) => '<span class="mv-colhead">' + colName[c] + '</span>').join('') +
      views.map((v) => '<div class="mv-view-label">' + esc(VIEW_WORDS[v] || v) + '</div>' + bodyCols.map((c) => bodyCell(v, c)).join('')).join('') + '</div>';
    const where = man.posture_from === 'start' ? 'at the start' : man.posture_from === 'test pose' ? 'in the test pose (the probe saved no start pose for the whole body)' : 'at the command\'s first saved model step (no start pose was saved)';
    const posture = '<li>Drawn alone and upright, chest toward the front camera' + (man.posture ? ' (it was ' + esc(man.posture) + ' ' + where + ')' : '') + '. The caregiver, the floor and objects no hand touched are left out.</li>';
    const bodyText = '<ul class="mv-bodynotes">' + (probe ? posture + '<li>Test pose: the pose the request put straight into the command reached from the same start.</li>' + (man.goal === 'wrist_reach_witness' ? '<li>The pink dot: the middle of the left palm in the test pose, joined by a line to the palm; the grey dot: the same point at the starting line (where the motor controller\'s own command ended). The close-up shows them larger.</li>' : '') :
      posture + '<li>The command asks for a direction for each of ' + int((man.handles || []).filter((h) => h.powered && h.scored).length) + ' body parts, and a place for the wrists, the head and the hips (the dots).</li><li>"Asked for" turns each part to its asked-for direction, joined to the part before it. The body cannot always reach this pose.</li>' + (placesText(man) ? '<li>' + placesText(man).trim() + '</li>' : '')) + '</ul>';
    const handles = man.handles || [];
    const weight = man.body_weight_n;
    const table = '<div class="mv-table-wrap"><table class="mv-table"><thead><tr><th>Body part</th><th>strength (0% = off, 100% = full)</th><th>place: start</th><th>end</th><th>after next</th><th>direction: start</th><th>end</th><th>after next</th></tr></thead><tbody>' +
      handles.filter((h) => h.powered).map((h) => !h.scored ?
        '<tr><td>' + esc(ROW_WORDS[h.name] || words(h.name)) + '</td><td>' + pct(h.strength) + '</td><td colspan="6" style="text-align:left">not scored: ' + (h.name === 'gaze' ? 'this row moves no joint (the eyes follow the separate gaze command)' : 'the hand control replaces this fingertip place before the controller reads it' + (String(man.settings || '').indexOf('servo on') >= 0 ? ', and the finger joint servo follows the 44 asked-for joint angles instead' : '')) + '</td></tr>' :
        '<tr><td>' + esc(h.part || words(h.name)) + '</td><td>' + pct(h.strength) + '</td><td>' + (h.position_row ? num(h.position_cm.start) + ' cm' : 'no place asked') + '</td><td>' + (h.position_row ? num(h.position_cm.end) + ' cm' : '') + '</td><td>' + (h.position_row ? num(h.position_cm.after_next) + ' cm' : '') + '</td><td>' + num(h.turn_deg.start) + '°</td><td>' + num(h.turn_deg.end) + '°</td><td>' + num(h.turn_deg.after_next) + '°</td></tr>').join('') +
      '</tbody></table></div><p class="small muted">Place: the distance from the body part to the place the command asked for. Direction: the angle between where the part points and where the command asked it to point. "After next": when the following command ended, 250 ms later. Only parts the command powered (mask on, strength above 0) are listed.</p>';
    const ctx = man.context || {};
    const objects = '<p>' + (ctx.hand_touch_left || ctx.hand_touch_right ? 'A ' + (ctx.hand_touch_left && ctx.hand_touch_right ? 'left and a right' : ctx.hand_touch_left ? 'left' : 'right') + ' hand part touched ' + int(ctx.objects_touched_by_a_hand) + ' object(s); the touched object rose ' + num(ctx.lift_cm) + ' cm from the start to the end (saved object positions).' : 'No hand part touched an object during this command (saved contacts).') + (ctx.toy_held ? ' The caregiver script carried the toy with the infant\'s hand at this time; that carry is the script\'s, not a grip.' : '') + '</p>';
    const touched = (man.objects_touched || []).length ? '<ul>' + man.objects_touched.map((o) => '<li><b>' + esc(words(o.name)) + '</b>: touched by the ' + esc(o.hands) + ' hand from model step ' + int(o.first_step) + ' to ' + int(o.last_step) + '; ' +
      (o.touched_at_end ? 'in contact at the command\'s end' + (o.touched_in_next === true ? ' and during the next command' : o.touched_in_next === false ? ', not during the next command' : '') : 'not in contact at the command\'s end (the contact ended during the command)') +
      (o.free === false ? '; the world\'s scripts place this object (a mount or a mocap body), so its height says nothing about a lift.' : '; the object rose ' + num(o.rise_cm) + ' cm from the start to the end.') + '</li>').join('') + '</ul>' : '';
    const contacts = (man.contacts || []).length ? '<ul>' + man.contacts.map((c) => '<li>' + esc(words(c.body_part)) + ' against ' + esc(words(c.other)) + ' (' + esc(c.other_kind) + '), model steps ' + int(c.first_step) + ' to ' + int(c.last_step) + ', largest force ' + num(c.peak_force_n, 1) + ' N' + (weight ? ' (' + pct(c.peak_force_n / weight) + ' of the infant\'s weight of ' + num(weight, 1) + ' N)' : '') + '</li>').join('') + '</ul>' : '<p>No saved contact between the body and anything else during this command.</p>';
    const cost = man.cost || {};
    const prov = man.provenance || {};
    const notes = (man.notes || []).length ? '<p class="small">' + man.notes.map(esc).join('. ') + '.</p>' : '';
    const zeroStep = req ? (req.zero_point_step != null ? req.zero_point_step : req.goals_from) : null;
    const ownZero = req ? zeroStep === man.step : false;
    const wm = man.wrist_marks || {};
    const reach = man.goal === 'wrist_reach_witness' && req && req.change != null ? ' The pink dot marks the middle of the left palm (the point the command calls the left wrist) in the test pose; the grey dot marks the same point where the motor controller\'s own command ended (the starting line, 0%).' +
      (wm.start_to_test_cm != null ? ' From the start, that point lies ' + num(wm.start_to_test_cm, 1) + ' cm from the pink dot and ' + num(wm.start_to_line_cm, 1) + ' cm from the grey dot: most of the move is the motor controller\'s own command.' : '') +
      (wm.line_to_test_cm != null ? ' The pink dot lies ' + num(wm.line_to_test_cm, 1) + ' cm from the grey dot: that is what the request put straight into the command added, of the ' + num(req.change, 1) + ' cm it asked for.' : '') +
      (wm.got_to_test_cm != null ? ' This try ended ' + num(wm.got_to_test_cm, 1) + ' cm from the pink dot.' : '') + ' The close-up of the left wrist shows the dots larger.' : '';
    const requestLine = req ? '<p class="mv-request">The test movement asks for a change of <b>' + esc(req.part) + '</b> (' + esc(changeWords(req.change, req.unit, man.goal)) + ').' + reach + ' ' +
      (man.arm === 'controller' && ownZero ? 'This try is the motor controller\'s own command: the starting line (0%) the other tries are measured from' : 'This try made <b>' + pct(req.this_try) + '</b> of it beyond where ' + (ownZero ? 'this checkpoint\'s' : 'checkpoint ' + int(zeroStep) + '\'s') + ' motor controller command ended') +
      '; the request put straight into the command made ' + pct(req.reference_command) + ' (its end is the test pose, the end the planner searches are scored against). The test poses were made at checkpoint ' + int(req.goals_from) + '.</p>' : '';
    const playing = !!S.playing[slot];
    const fingerTest = probe && (man.goal === 'curl_witness' || man.goal === 'extension_witness');
    const bodySection = '<div class="mv-key">' + Object.entries(PART_COLORS).map(([kk, c]) => '<span><i style="background:' + c + '"></i>' + esc(kk) + '</span>').join('') + '<span><i class="pale"></i>pale: the body asked for</span>' + (probe ? '' : '<span><i class="dot"></i>dot and line: an asked-for place and its part (for a wrist: the middle of the palm)</span>') + '</div>' +
      bodyBlock + (probe ? '' : partRows(man, k, ms) + '<div class="mv-key">' + BAR_KEY + '</div>');
    return '<div class="mv-result" id="' + id + '">' +
      (slot === 'A' ? '<p class="mv-pinned">Pinned for comparison</p>' : '') +
      (probe ?
        '<h3>Test movement · ' + esc(man.goal_name) + ' · body ' + man.body + ' · ' + esc(TRY_NAME[man.arm] || man.arm) + '</h3>' +
        '<div class="mv-id">' + esc(man.label) + ', run ' + esc(when(man.time, true)) + ' · pictures made ' + esc(when(man.rendered_at)) + '</div>' :
        '<h3>Body ' + man.body + ' · model step ' + int(man.step) + ' · ' + esc(runName(man.recording, man.label)) + '</h3>' +
        '<div class="mv-id">Recorded ' + esc(when(man.time, true)) + ' · ' + esc(kindWord(man.kind)) + (runAbout(man.recording) ? ' (' + esc(runAbout(man.recording)) + ')' : '') + (man.settings ? ' · ' + esc(man.settings) : '') + ' · chosen by ' + esc(authorName(man.author)) + ' · held ' + int((man.last_step || man.step) - man.step + 1) + ' model steps (' + int(((man.last_step || man.step) - man.step + 1) * 50) + ' ms) · pictures made ' + esc(when(man.rendered_at)) + '</div>') +
      requestLine +
      '<div class="mv-time">' + (samples.length > 1 ? '<button type="button" data-act="play" data-slot="' + slot + '">' + (playing ? 'Pause' : 'Play the 250 ms') + '</button><input type="range" min="0" max="' + Math.max(0, samples.length - 1) + '" value="' + ti + '" data-act="time" data-slot="' + slot + '" aria-label="Time within the command">' : '') +
      '<span>"Got to" shows the body <b>' + ms + ' ms</b> after the command began' + (stepAt != null ? ' (model step ' + int(stepAt) + ')' : '') + '</span></div>' +
      '<div class="mv-actions">' + (slot === 'B' && !S.keep ? '<button type="button" class="mv-keep" data-act="keep" title="Keeps this drawing on the page; the next command you draw appears beside it">Pin this drawing to compare</button><span class="small muted">Pin it, then pick and draw another command: the two show side by side.</span>' : slot === 'A' && !S.current ? '<button type="button" class="mv-keep" data-act="unkeep">Unpin</button>' : '') +
      '<a class="mv-keep mv-export" href="/api/movement/export?key=' + encodeURIComponent(key) + '" download="movement_' + esc(key) + '.html">Save this drawing as one file</a></div>' +
      notes +
      '<section class="mv-sec"><div class="mv-sec-head"><h4>Hands</h4>' + (man.lite ? '' : '<span class="mv-views" role="group" aria-label="Hand view"><button type="button" data-act="view" data-view="front" data-slot="' + slot + '" aria-pressed="' + (view === 'front') + '">front view</button><button type="button" data-act="view" data-view="side" data-slot="' + slot + '" aria-pressed="' + (view === 'side') + '">side view</button></span>') + '</div>' +
      '<div class="mv-key">' + HAND_COLORS.map((c) => '<span><i style="background:' + c[1] + '"></i>' + c[0] + '</span>').join('') + '<span><i class="pale"></i>pale: the shape asked for</span></div>' +
      '<div class="mv-hands">' + handBlock('left', 'Left hand') + handBlock('right', 'Right hand') + '</div>' +
      (probe ? '' : '<div class="mv-key">' + BAR_KEY + '</div>' + (String(man.settings || '').indexOf('servo on') >= 0 ? '<div class="mv-why small"><b>Why a finger can look unchanged</b><ul><li>Fingertip joints move little in 250 ms: 13.7% to 17.8% of an asked-for change (knuckles: 41% to 54%).</li><li>Straightening is weaker than curling: at most 34.5% of the curling force on the 16 bending joints of each hand (its 6 joints that move sideways or twist are equally strong both ways).</li><li>The left and right hands are alike.</li></ul><span class="muted">Measured Oct 19, 2026 over the 3,952 commands of test run POSE_A_off_494025 (finger joint servo on); each joint kind pooled over a hand\'s four fingers, curling and straightening apart, asked-for changes of 5° or more. <a href="/files/docs/H84_MOVEMENT_VIEWER_PLAN_2026-10-18.md">The measurements &#8599;</a></span></div>' : '')) + '</section>' +
      (fingerTest ? '<details class="mv-sec"><summary><b>Whole body</b> (this test asks only for a finger change)</summary>' + bodyText + bodySection + '</details>' :
        '<section class="mv-sec"><div class="mv-sec-head"><h4>Whole body</h4></div>' + bodyText + bodySection + '</section>') +
      (probe ? '<p class="small muted">The probe saved no per-part command and no contacts; the numbers above are the asked-for part only.</p>' :
        '<details><summary>Every powered body part: asked-for places and directions, and the distances at the start, the end and after the next command</summary>' + table + '</details>' +
        '<details><summary>Objects and contacts during the command</summary>' + objects + touched + contacts + '</details>') +
      '<details><summary>Where these pictures come from, and their cost</summary><p class="small">Drawn from the saved records by forward kinematics of the saved model; no training, no new movement. ' +
      esc(String(prov.selection || '').replace(/^./, (ch) => ch.toUpperCase())) + '. Drawing took ' + num(cost.seconds, 1) + ' s; the drawing process used ' + int(cost.process_memory_mb_after) + ' MB of memory afterwards (' + int(cost.process_memory_mb_before) + ' MB before); the graphics card had ' + int(cost.gpu_memory_mb_before) + ' MB in use before and ' + int(cost.gpu_memory_mb_after) + ' MB after; ' + int(cost.pictures) + ' pictures, ' + num((cost.bytes || 0) / 1048576, 1) + ' MB.' + (man.reused ? ' These pictures were drawn earlier and reused.' : '') + '</p>' +
      '<div class="mv-prov">Model: ' + esc(prov.model) + ' (structure fingerprint ' + esc(String(prov.model_structure_fingerprint || '').slice(0, 16)) + '…)<br>' + Object.entries(prov.chunks || {}).map(([p, h]) => esc(p) + ' · SHA-256 ' + esc(String(h).slice(0, 16)) + '…').join('<br>') + '</div></details></div>';
  }
  // the comparison of a pinned drawing (A) and the current one (B): the same pictures side by side, row by row, with
  // each drawing's shares beside its pictures; each full drawing folded below (the independent review of Oct 19, 2026:
  // two full drawings one above the other put matching pictures about 2,900 px apart)
  function compareHTML(A, B) {
    const side = (man, slot) => {
      const samples = man.samples_ms || [];
      const ti = S['t' + slot] != null ? Math.min(S['t' + slot], samples.length - 1) : samples.length - 1;
      const ms = samples[ti];
      return { man: man, ms: ms, k: sampleIndex(man, ms), name: man.kind === 'probe' ? 'Test movement, ' + man.goal_name + ', body ' + man.body : 'Body ' + man.body + ', model step ' + int(man.step) + ', ' + runName(man.recording, man.label) };
    };
    const a = side(A, 'A'), b = side(B, 'B');
    const setA = A.settings || '', setB = B.settings || '';
    const sets = setA && setB && setA !== setB ? '<p class="mv-cmp-warn">These two runs used different control settings (' + esc(setA) + '; ' + esc(setB) + '): read them as separate examples, not as before and after.</p>' : '';
    const pic = (x, name, alt) => (x.man.images || {})[name] ? img(x.man.key, name, alt, name.startsWith('body_') ? 'body' : '') : missing('Not saved', name.startsWith('body_') ? 'body' : '');
    const trio = (x, pre, isBody) => '<p class="mv-cmp-tag">' + (x === a ? 'Pinned' : 'Drawn now') + '</p><div class="mv-trio">' + ['start', 'target', sampleName(x.ms)].map((st, i) => '<figure>' + pic(x, pre + st + '.png', pre + st) + '<figcaption>' + ['Start', x.man.kind === 'probe' ? 'Test pose' : isBody ? 'Asked for (directions)' : 'Asked for', 'Got to, ' + x.ms + ' ms'][i] + '</figcaption></figure>').join('') + '</div>';
    const handShare = (x, s) => { const per = ((x.man.fingers || {})[s]) || []; const r = shareOf([].concat(...per.map((f) => f.joints)), x.k); return r && r.share != null ? 'all fingers: ' + wayText(r.share) : ''; };
    const bodyShare = (x) => { const keep = (x.man.handles || []).filter((h) => h.powered && h.scored && h.turn_deg && h.turn_deg.samples); let s0 = 0, s1 = 0; keep.forEach((h) => { s0 += h.turn_deg.start || 0; s1 += h.turn_deg.samples[x.k] || 0; }); return s0 > 0 ? keep.length + ' parts\' directions: ' + wayText((s0 - s1) / s0) : ''; };
    const row = (title, fa, fb) => '<div class="mv-cmp-row"><h4>' + title + '</h4><div class="mv-cmp-pair"><div>' + fa + '</div><div>' + fb + '</div></div></div>';
    return '<div class="mv-result mv-compare"><h3>Comparing two drawings</h3><div class="mv-cmp-pair mv-cmp-names"><div><p class="mv-pinned">Pinned</p><b>' + esc(a.name) + '</b>' + (setA ? '<span class="small muted">' + esc(setA) + '</span>' : '') + '</div><div><p class="mv-pinned now">Drawn now</p><b>' + esc(b.name) + '</b>' + (setB ? '<span class="small muted">' + esc(setB) + '</span>' : '') + '</div></div>' + sets +
      '<div class="mv-key">' + HAND_COLORS.map((c) => '<span><i style="background:' + c[1] + '"></i>' + c[0] + '</span>').join('') + '<span><i class="pale"></i>pale: what was asked for</span></div>' +
      row('Left hand, front view', trio(a, 'hand_left_front_') + '<p class="small">' + handShare(a, 'left') + '</p>', trio(b, 'hand_left_front_') + '<p class="small">' + handShare(b, 'left') + '</p>') +
      row('Right hand, front view', trio(a, 'hand_right_front_') + '<p class="small">' + handShare(a, 'right') + '</p>', trio(b, 'hand_right_front_') + '<p class="small">' + handShare(b, 'right') + '</p>') +
      row('Whole body, from the front', trio(a, 'body_front_', true) + '<p class="small">' + bodyShare(a) + '</p>', trio(b, 'body_front_', true) + '<p class="small">' + bodyShare(b) + '</p>') +
      '<p class="small muted">"Asked for (directions)": every powered body part turned to its asked-for direction, each joined to the part before it; it is not a pose the body can necessarily reach (the full drawings below say how far its parts are from their asked-for places).</p>' +
      '<div class="mv-actions"><button type="button" class="mv-keep" data-act="unkeep">Unpin</button><span class="small muted">The full drawings, with their numbers and the side views, are folded below.</span></div></div>' +
      '<details class="mv-full" data-fold="full-A"><summary>Full drawing: pinned</summary>' + nameFolds(resultHTML(A, 'A'), 'A') + '</details>' +
      '<details class="mv-full" data-fold="full-B"><summary>Full drawing: drawn now</summary>' + nameFolds(resultHTML(B, 'B'), 'B') + '</details>';
  }
  // each fold (a details element) of a drawing gets a name made of its drawing's slot and its place in the drawing, so
  // a re-draw (the time slider, Play, the hand view) opens again every fold that was open (the independent review of
  // Oct 19, 2026, round 3, finding 1: in a comparison every control inside a full drawing closed it)
  function nameFolds(html, prefix) {
    let n = 0;
    return html.replace(/<details(?![^>]*data-fold)/g, () => '<details data-fold="' + prefix + '-' + (n++) + '"');
  }
  function showResults() {
    const box = $('mv-results');
    const open = new Set([...box.querySelectorAll('details[data-fold]')].filter((d) => d.open).map((d) => d.dataset.fold));
    let html;
    if (S.keep && S.current) html = compareHTML(S.keep, S.current);
    else {
      const list = [];
      if (S.keep) list.push(['A', S.keep]);
      if (S.current) list.push(['B', S.current]);
      html = list.map(([slot, man]) => nameFolds(resultHTML(man, slot), slot)).join('') +
        (S.keep && !S.current ? '<p class="mv-pin-hint">Pinned. Pick another command above and draw it: the two show side by side.</p>' : '');
    }
    box.innerHTML = html;
    box.querySelectorAll('details[data-fold]').forEach((d) => { if (open.has(d.dataset.fold)) d.open = true; });
  }
  async function pollDrawing(r, status) {
    const t0 = Date.now();
    while (r.state === 'queued') {
      const sv = r.service || {};
      let msg = 'Drawing: waiting for the drawing service' + (r.position ? ' (number ' + r.position + ' in its queue)' : '') + ', ' + Math.round((Date.now() - t0) / 1000) + ' s so far.';
      if (sv.paused) msg += ' The service is paused (' + (sv.paused_by || 'service_control.json') + ').';
      if (sv.guard && sv.guard.allowed === false) msg += ' The memory guard refused the drawing: ' + sv.guard.reading;
      if (sv.alive === false) msg += ' The service is starting.';
      status(msg);
      await new Promise((res) => setTimeout(res, 1000));
      r = await getJSON('/api/movement/render?key=' + encodeURIComponent(r.key));
    }
    return r;
  }
  async function draw() {
    const c = S.cmd;
    if (!c) return;
    const btn = $('mv-render');
    btn.disabled = true;
    const status = (t) => { $('mv-draw-status').textContent = t; };
    status('Asking the drawing service to draw this command.');
    try {
      let r = await getJSON('/api/movement/render', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ recording: c.recording, body: c.body, step: c.step }) });
      r = await pollDrawing(r, status);
      if (r.state === 'ready') {
        S.current = r; S.tB = undefined; S.playing.B = false;
        status('');
        showResults();
        readout();
        $('mv-results').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      } else {
        status('The command could not be drawn: ' + (r.error || r.state));
      }
    } catch (e) {
      status('The drawing request failed: ' + e.message);
    } finally {
      btn.disabled = false;
    }
  }
  function stopPlay(slot) { clearInterval(S.timers[slot]); S.playing[slot] = false; }
  function onResultClick(evt) {
    const b = evt.target.closest('[data-act]');
    if (!b) return;
    const slot = b.dataset.slot;
    if (b.dataset.act === 'view') { S['view' + slot] = b.dataset.view; showResults(); }
    if (b.dataset.act === 'keep') { stopPlay('B'); S.keep = S.current; S.current = null; S.tA = S.tB; S.viewA = S.viewB; showResults(); }
    if (b.dataset.act === 'unkeep') { stopPlay('A'); S.keep = null; showResults(); }
    if (b.dataset.act === 'play') {
      const man = slot === 'A' ? S.keep : S.current;
      const n = (man.samples_ms || []).length;
      if (S.playing[slot]) { stopPlay(slot); showResults(); return; }
      S.playing[slot] = true;
      let i = (S['t' + slot] != null && S['t' + slot] < n - 1) ? S['t' + slot] : -1;
      const advance = () => {
        i += 1;
        if (i >= n) { stopPlay(slot); showResults(); return; }
        S['t' + slot] = i;
        showResults();
      };
      advance();
      S.timers[slot] = setInterval(advance, 600);
    }
  }
  function onResultInput(evt) {
    const b = evt.target.closest('[data-act="time"]');
    if (!b) return;
    stopPlay(b.dataset.slot);
    S['t' + b.dataset.slot] = Number(b.value);
    showResults();
    const again = document.querySelector('#mvr' + b.dataset.slot + ' [data-act="time"]');
    if (again) again.focus();
  }

  // ------------------------------------------------------------------ automatic pictures
  async function loadAuto() {
    try {
      const r = await getJSON('/api/movement/auto?' + query({ limit: 48 }));
      $('mv-auto-summary').textContent = 'Pictures drawn automatically in these dates: ' + int(r.total) + ' chosen, ' + int(r.drawn) + ' drawn, ' + int(r.waiting) + ' waiting for the drawing service';
      const rules = Array.isArray(r.rules) ? r.rules : [];
      $('mv-auto-rules').innerHTML = 'Chosen from the scores, not at fixed times, and drawn in a smaller set (front views at the start, as asked for and at the end). Each picture shows where the body or the hands got to after the command, with what was asked for drawn pale over it, and names the rule that chose it:<ul>' + rules.map((x) => '<li>' + esc(x) + '</li>').join('') + '</ul>';
      $('mv-gallery').innerHTML = (r.picks || []).map((p) => '<button type="button" data-key="' + esc(p.key) + '" data-rec="' + esc(p.recording) + '" data-body="' + p.body + '" data-step="' + p.step + '">' +
        (p.state === 'drawn' && (p.thumbs || p.thumb) ? (p.thumbs ? '<span class="mv-thumbs">' + p.thumbs.map((t, i) => img(p.key, t, (i ? 'Right' : 'Left') + ' hand after the command', '', true)).join('') + '</span><small class="muted">left and right hand, got to (asked for: pale)</small>' : img(p.key, p.thumb, 'Whole body after the command, from the front', '', true) + '<small class="muted">whole body, got to (asked for: pale)</small>') : '<span class="mv-wait">' + (p.state === 'refused' ? 'cannot be drawn' : 'not drawn yet') + '</span>') +
        '<b>' + esc(when(p.time)) + '</b><br>' + esc(runName(p.recording, p.label)) + ', body ' + p.body + ', model step ' + int(p.step) + '<br>' + esc(p.reasons.join('; ')) + '</button>').join('') ||
        '<p class="small muted">No automatic pictures in these dates.</p>';
    } catch (e) {
      $('mv-auto-summary').textContent = 'Automatic pictures could not be read: ' + e.message;
    }
  }
  async function openPick(evt) {
    const b = evt.target.closest('button[data-step]');
    if (!b) return;
    S.userEpoch++;
    try {
      const r = await getJSON('/api/movement/command?' + new URLSearchParams({ rec: b.dataset.rec, body: b.dataset.body, step: b.dataset.step, measure: $('mv-measure').value, author: 'all' }).toString());
      if (r.command) { S.cmd = r.command; readout(); $('mv-pos').textContent = 'an automatic pick'; inside(); }
      const d = await getJSON('/api/movement/render?key=' + encodeURIComponent(b.dataset.key));
      if (d.state === 'ready') {
        S.current = d; S.tB = undefined; S.playing.B = false;
        showResults();
        $('mv-draw-status').textContent = 'Showing the automatic drawing; "Draw this command" draws every view and every 50 ms sample.';
        $('mv-results').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      } else {
        draw();
      }
    } catch (e) {
      $('mv-draw-status').textContent = 'That pick could not be opened: ' + e.message;
    }
  }

  // ------------------------------------------------------------------ checkpoint probes
  // the size of a test movement in words: the probe saves the length of the change divided by the square root of the
  // number of parts it asks of (compare_hand_body_goals_2026-09-24.py: size = |change| / sqrt(len(parts))), the parts
  // being the 28 bending finger joints, the one left wrist, the two trunk parts or the one elbow joint; so the wrist's
  // size is its whole move and the trunk's a typical turn per part (the independent review of Oct 19, 2026, round 5)
  function changeWords(size, unit, goal) {
    const u = unitSign(unit === 'cm' ? 'cm' : 'degrees');
    if (goal === 'curl_witness' || goal === 'extension_witness') return 'a typical change of ' + num(size) + u + ' per joint, over the 28 bending joints of both hands\' fingers';
    if (goal === 'supported_trunk_turn_witness') return 'a typical turn of ' + num(size) + u + ' per trunk part, over its 2 parts';
    if (goal === 'elbow_flex_witness') return 'a bend of ' + num(size) + u + ' at the left elbow joint';
    return 'a move of ' + num(size) + u + ' in all';
  }
  function zeroText(own, step) { return own ? 'where this checkpoint\'s motor controller command ended' : 'where checkpoint ' + int(step) + '\'s motor controller command ended'; }
  // the fixed-start improvement line (proxy decision of Oct 18, 2026): each planner search at this checkpoint
  // against the same search at the reference checkpoint, from the same frozen starts toward the same frozen goals
  function gainText(g, brief) {
    if (!g) return '';
    if (g.status !== 'valid') return 'improvement check not valid: ' + esc((g.invalid_reasons || []).join('; ') || 'no reason recorded') + '.';
    const d = (g.arms || {}).direct || {}, s = (g.arms || {}).synergy || {};
    // the distance to the goals is a score, so its change is shown as a share of the reference's own distance (DESIGN.md
    // section 11): 0% = no change, -100% = every body part on its goal; real units (cm, degrees) stay as they are
    const rel = (a, x) => (a.reference_mean ? 100 * x / a.reference_mean : null);
    const pctOf = (a, x) => { const v = rel(a, x); return v == null || !isFinite(v) ? '–' : Math.abs(v).toFixed(1) + '% ' + (v > 0 ? 'farther' : 'nearer'); };
    const ivp = (a) => pctOf(a, a.interval && a.interval[0]) + ' to ' + pctOf(a, a.interval && a.interval[1]);
    const ivu = (a, dd, unit) => sgn(a.interval && a.interval[0], dd) + unit + ' to ' + sgn(a.interval && a.interval[1], dd) + unit;
    const met = g.line_met ? 'met' : 'not met';
    if (brief) return 'improvement check ' + met + ': the direct search ended ' + pctOf(d, d.difference_mean) + ' from its goals than at checkpoint ' + int(g.reference_step);
    const same = g.this_step === g.reference_step ? ' (the reference compared with itself: no change by construction)' : '';
    const said = (r) => { const w = String(r || ''); return /worsen/.test(w) ? 'higher here than at the reference' : /improv|gain/.test(w) ? 'lower here than at the reference' : /no clear/.test(w) ? 'no clear difference' : w; };
    return '<b>Improvement check: ' + met + '.</b> The direct planner search ended ' + pctOf(d, d.difference_mean) + ' from its goals here than at checkpoint ' + int(g.reference_step) + same + ' (same 8 starts, same test poses). To be met it must be nearer with 95% confidence: the whole 95% range must read "nearer"; here it runs from ' + ivp(d) + ' (the rule of Oct 18, 2026). ' +
      '<details class="mv-gain-more"><summary>More of this comparison</summary><p class="small">' + int(d.bodies_lower_here) + ' of ' + int(d.bodies) + ' bodies ended nearer the goals here. The direct search\'s wrist distance: ' + sgn(d.wrist_cm && d.wrist_cm.difference_mean, 2) + ' cm (95% range ' + ivu(d.wrist_cm || {}, 2, ' cm') + ', ' + esc(said((d.wrist_cm || {}).reading)) + '); its hand joint distance: ' + sgn(d.hand_deg && d.hand_deg.difference_mean, 2) + '° (' + ivu(d.hand_deg || {}, 2, '°') + ', ' + esc(said((d.hand_deg || {}).reading)) + '). The synergy search, reported and not deciding: ' + pctOf(s, s.difference_mean) + ' (95% range ' + ivp(s) + '). The strength and eye command inside each search are its own checkpoint\'s, so the check reads the whole planner and controller at that checkpoint, not one planner change.</p></details>';
  }
  async function loadProbes() {
    const goal = $('mv-goal').value || 'curl_witness';
    try {
      const r = await getJSON('/api/movement/probes?' + new URLSearchParams({ goal: goal, body: $('mv-pbody').value }).toString());
      S.probes = r;
      if (!$('mv-goal').options.length) {
        $('mv-goal').innerHTML = Object.entries(r.goals).map(([k, v]) => '<option value="' + esc(k) + '">' + esc(v) + ' (' + esc(r.parts[k].part) + ')</option>').join('');
        $('mv-goal').value = goal;
      }
      const runs = r.runs || [];
      const keep = $('mv-prun').value;
      // one entry per bar row (runs of one checkpoint and start with the same result share a row and an entry), plus
      // each run that is not drawn; newest first
      const opts = probeRows(runs.filter((x) => !x.replaced_by)).map((row) => [row.x, probeLabel(row.x) + (row.runs.length > 1 ? ' (' + row.runs.length + ' runs, same result)' : '')])
        .concat(runs.filter((x) => x.replaced_by).map((x) => [x, probeLabel(x) + ', not drawn: its searches started from its own command']))
        .sort((u, v) => v[0].time - u[0].time);
      $('mv-prun').innerHTML = opts.map(([x, t]) => '<option value="' + esc(x.id) + '">' + esc(t) + '</option>').join('');
      if (keep && runs.some((x) => x.id === keep)) $('mv-prun').value = keep;
      probeBars();
      probeReadout();
    } catch (e) {
      $('mv-preadout').textContent = 'The probe results could not be read: ' + e.message;
    }
  }
  // runs at one checkpoint with the same start kind and the same shares read as one row ("4 runs, same result")
  function probeRows(runs) {
    const rows = [];
    runs.forEach((x) => {
      const frozen = x.mode.startsWith('frozen');
      const sig = x.checkpoint_step + '|' + frozen + '|' + Object.entries(x.tries).map(([k, a]) => k + ':' + a.share).sort().join(',');
      const hit = rows.find((r) => r.sig === sig);
      if (hit) { hit.runs.push(x); return; }
      rows.push({ sig: sig, x: x, runs: [x], frozen: frozen });
    });
    return rows;
  }
  function probeBars() {
    const box = $('mv-probe-bars');
    const r = S.probes;
    const runs = ((r && r.runs) || []).filter((x) => !x.replaced_by);
    $('mv-probe-key').innerHTML = ['witness', 'controller', 'direct', 'synergy'].map((k) => '<span><i class="mv-swatch" style="background:' + TRY_COLOR[k] + '"></i>' + esc(TRY_NAME[k]) +
      (k === 'controller' ? ' (drawn only where another checkpoint\'s controller is the starting line)' : '') + '</span>').join('') +
      '<span>Each bar: the middle value over the trials that count, of the 8 bodies (the probe\'s own rule: a trial counts when the request put straight into the command moved the part past the starting line, and farther along the asked-for change than the same change reversed did).</span>' +
      '<span>A planner bar longer than the orange one went past the test pose; a bar left of 0% moved the part the opposite way from the asked-for change. A grey bar\'s length includes the distance between the two checkpoints\' motor controller commands, not only the request.</span>';
    if (!runs.length) { box.innerHTML = '<p class="mv-empty">No probe runs for this movement.</p>'; $('mv-probe-legend').innerHTML = notDrawn(r); $('mv-probe-reading').textContent = ''; return; }
    const rows = probeRows(runs);
    let lo = 0, hi = 1;
    runs.forEach((x) => Object.values(x.tries).forEach((a) => { lo = Math.min(lo, a.share); hi = Math.max(hi, a.share); }));
    const span = hi - lo;
    const X = (v) => ((v - lo) / span) * 100;
    const z = X(0);
    const ticks = [0, 0.25, 0.5, 0.75, 1].concat(lo < -0.2 ? [-0.25] : []);
    const sel = $('mv-prun').value;
    let h = '<div class="mv-bar-head"><span></span><span class="mv-axis-row">' + ticks.map((t) => '<b style="left:' + X(t).toFixed(1) + '%">' + Math.round(t * 100) + '%</b>').join('') + '</span><span></span></div>';
    h += rows.map((row) => {
      const x = row.x;
      const label = '<b>' + esc(dayLabel(x.time)) + '</b> checkpoint ' + int(x.checkpoint_step) + (row.frozen ? ' <em>same start each time</em>' : '') + (row.runs.length > 1 ? ' <em>' + row.runs.length + ' runs, same result</em>' : '');
      const lines = ['witness', 'controller', 'direct', 'synergy'].filter((k) => x.tries[k]).map((k) => {
        const v = x.tries[k].share;
        const left = v >= 0 ? z : X(v), width = Math.abs(X(v) - z);
        return '<span class="mv-pline"><span class="mv-track thin">' + ticks.map((tk) => '<i class="grid" style="left:' + X(tk).toFixed(1) + '%"></i>').join('') + '<i class="bar" data-try="' + k + '" style="left:' + left.toFixed(2) + '%;width:' + width.toFixed(2) + '%;background:' + TRY_COLOR[k] + '"></i></span><span class="mv-pval">' + pct(v) + '</span></span>';
      }).join('');
      return '<button type="button" class="mv-prow' + (row.runs.some((y) => y.id === sel) ? ' sel' : '') + '" data-run="' + esc(x.id) + '"><span class="mv-bar-label">' + label + '</span><span class="mv-plines">' + lines + '</span></button>';
    }).join('');
    box.innerHTML = h;
    // the reading above the chart: the ranges over the rows
    const range = (k) => { const v = runs.map((x) => x.tries[k] && x.tries[k].share).filter((a) => a != null); return v.length ? (v.length > 1 ? pct(Math.min(...v)) + ' to ' + pct(Math.max(...v)) : pct(v[0])) : null; };
    const w = range('witness'), d = range('direct'), s = range('synergy');
    $('mv-probe-reading').innerHTML = '<b>Reading:</b> at the ' + rows.length + ' checkpoints drawn, putting the request straight into the command made ' + (w || '–') + ' of the movement; that is how far the test pose lies, and so how long a planner bar would be if the planner reached the test pose. The planner searches made ' + (d || '–') + ' (direct) and ' + (s || '–') + ' (synergy).';
    $('mv-probe-legend').innerHTML = '<span>"Same start each time": the same saved bodies and test poses at every checkpoint (made at checkpoint 494,025), so those rows compare like with like.</span>' + notDrawn(r);
  }
  // every probe run with no bar for this movement, and why
  function notDrawn(r) {
    const rows = ((r && r.runs) || []).filter((x) => x.replaced_by).map((x) => [x, "its planner searches started from this checkpoint's own motor controller command, which lies outside the planners' search region around the command the test poses were made from (checkpoint 494,025's), so its shares measure the distance between those two commands; the run whose searches start from 494,025's command is drawn in its place, and this run's trials stay in the checkpoint list"])
      .concat(((r && r.not_drawn) || []).map((x) => [x, x.why]));
    if (!rows.length) return '';
    return '<details class="mv-notdrawn"><summary>Not drawn: ' + rows.length + (rows.length === 1 ? ' probe run' : ' probe runs') + '</summary><ul>' +
      rows.map(([x, why]) => '<li>' + esc(x.checkpoint_step != null ? probeLabel(x) : dayLabel(x.time) + ', ' + x.label) + ': ' + esc(why) + '.</li>').join('') + '</ul></details>';
  }
  async function probeReadout() {
    const pid = $('mv-prun').value;
    if (!pid) return;
    try {
      const r = await getJSON('/api/movement/probe_trials?' + new URLSearchParams({ probe: pid, goal: $('mv-goal').value }).toString());
      const bodies = [...new Set(r.trials.map((t) => t.body))].sort((a, b) => a - b);
      const keep = $('mv-ptbody').value;
      $('mv-ptbody').innerHTML = bodies.map((b) => '<option value="' + b + '">Body ' + b + '</option>').join('');
      if (keep && bodies.includes(Number(keep))) $('mv-ptbody').value = keep;
      const b = Number($('mv-ptbody').value), arm = $('mv-parm').value;
      const t = r.trials.find((x) => x.body === b);
      if (!t) { $('mv-preadout').textContent = 'This checkpoint has no such trial.'; return; }
      const goal = $('mv-goal').value;
      $('mv-preadout').innerHTML = esc(S.probes.goal_name) + ', body ' + b + ', ' + esc(TRY_NAME[arm]) + ': ' +
        (arm === 'controller' && t.own_baseline
          ? 'the starting line. Every other try is measured from where this command ended (0% = there, 100% = the whole asked-for change: ' + esc(changeWords(t.change, t.unit, goal)) + '): the request put straight into the command made ' + pct(t.witness) + ', the direct search ' + pct(t.direct) + ', the synergy search ' + pct(t.synergy) + '.'
          : 'made <b>' + pct(t[arm]) + '</b> of the asked-for change (' + esc(changeWords(t.change, t.unit, goal)) + ') beyond ' + zeroText(t.own_baseline, t.zero_point_step) + '; the request put straight into the command made ' + pct(t.witness) + '.' +
            (arm === 'controller' ? ' This checkpoint\'s own controller command differs from checkpoint ' + int(t.zero_point_step) + '\'s, so this share includes the distance between the two commands, not only the request.' : '')) +
        (t.informative ? ' This trial counts: the request put straight into the command moved the part past the starting line (' + pct(t.own_share) + '), farther than the same change reversed did (' + pct(t.mirror_share) + ').' :
          ' <b>This trial does not count</b> (the bars leave it out): ' + (t.own_share == null ? 'it carries no check of its request.' : !(t.own_share > 0) ? 'the request put straight into the command did not move the part past the starting line (' + pct(t.own_share) + ').' : t.mirror_share == null ? 'the same change reversed was not run for it.' : 'the same change reversed moved the part as far or farther along the asked-for change (' + pct(t.mirror_share) + ') than the request did (' + pct(t.own_share) + '), so the movement is not the request\'s doing.') + ' These two shares are measured from this checkpoint\'s own motor controller command.') +
        (((S.probes.runs || []).find((x) => x.id === pid) || {}).replaced_by ? ' <b>Not drawn:</b> this run\'s planner searches started from this checkpoint\'s own controller command, outside their search region around the command the test poses were made from (checkpoint 494,025\'s), so their shares here measure the distance between those two commands; the run whose searches start from 494,025\'s command is drawn instead.' : '') +
        (t.drawn && t.drawn[arm] ? ' <span class="muted">Already drawn.</span>' : '') +
        (r.gain ? '<div class="mv-gain">' + gainText(r.gain, false) + '</div>' : '');
    } catch (e) {
      $('mv-preadout').textContent = 'The trials could not be read: ' + e.message;
    }
  }
  async function drawProbe() {
    const pid = $('mv-prun').value;
    if (!pid) return;
    const req = { probe: pid, body: Number($('mv-ptbody').value), goal: $('mv-goal').value, arm: $('mv-parm').value };
    $('mv-pdraw').disabled = true;
    try {
      let r = await getJSON('/api/movement/render', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(req) });
      r = await pollDrawing(r, (t) => { $('mv-preadout').textContent = t; });
      if (r.state === 'ready') { S.current = r; S.tB = undefined; showResults(); probeReadout(); $('mv-results').scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }
      else $('mv-preadout').textContent = 'The try could not be drawn: ' + (r.error || r.state);
    } catch (e) {
      $('mv-preadout').textContent = 'The drawing request failed: ' + e.message;
    } finally {
      $('mv-pdraw').disabled = false;
    }
  }

  // ------------------------------------------------------------------ the Tests page
  function decorate(card, row) {
    const recs = S.rowMap && S.rowMap.get(row);
    if (!recs || !recs.length || card.querySelector('.mv-test-link')) return;
    const box = document.createElement('div');
    box.className = 'mv-test-link';
    box.append('Saved movements of the recordings this row names: ');
    recs.forEach((r, i) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'text-button';
      b.textContent = r.label + ' (' + dayLabel(r.first_time) + ')';
      b.addEventListener('click', () => openRecording(r.id));
      if (i) box.append(' · ');
      box.append(b);
    });
    card.append(box);
  }
  function openRecording(rid) {
    const tab = document.getElementById('tab-now');
    if (tab) tab.click();
    chooseRun(rid);
    setTimeout(() => $('movement-viewer').scrollIntoView({ block: 'start' }), 50);
  }
  window.movementViewer = {
    testButton(card, row) {
      if (!S.sources) { S.pendingCards.push([card, row]); return; }
      decorate(card, row);
    },
    open: openRecording,
  };

  // ------------------------------------------------------------------ the definitions
  function about() {
    $('mv-about').innerHTML =
      '<p><b>A command</b> is what the model sends to the body: for each of 26 body handles, 9 numbers (an asked-for place in cm, an asked-for direction, a radius, a strength from 0 to 1), which handles are switched on, and 44 asked-for finger joint angles. On H84 each command is held for 5 model steps (a model step is 50 ms, so 250 ms). The observer saves each command with the body\'s joint angles at every model step.</p>' +
      '<p><b>Who chose the command.</b> The saved commands come from three sources. The planner picks a goal from the last 200 model steps and searches for the command the model predicts reaches it. The motor controller is the model\'s own command when the planner does not choose one. The motor controller with exploration noise is that command with random numbers added to it (the run setting babble_noise).</p>' +
      '<p><b>The distances.</b> For each command the viewer measures how far the body was from what the command asked for: just before it began, at its last model step, and when the following command ended. Fingers: per hand, the average over its 22 finger joints of how many degrees each joint is from its asked-for angle. Wrist: how many cm the wrist is from its asked-for place. Whole body: the average over the powered body parts (both wrists, forearms, upper arms, thighs, lower legs and feet, the head, the chest and the hips: 15 parts) of the angle between where the part points and where it was asked to point. The body\'s positions come from the saved joint angles by forward kinematics of the saved model; nothing is simulated again. The 10 fingertip rows and the eyes\' row are not scored: the hand control replaces the fingertip places before the controller reads them, and the eyes\' row moves no joint.</p>' +
      '<p><b>Share of the way</b> = (sum of the distances at the start - sum of the distances at the end) / sum of the distances at the start, over the commands counted: 0% = they ended no closer than they began, 100% = every command got exactly there, below 0% = the body moved farther away. A command with no saved start (a body\'s first command in a run) is not counted. Each command sent is 35% of the new request plus 65% of the previous command sent (the command smoother: the run setting smooth_actions), so the next command\'s asked-for values lie close to this one\'s.</p>' +
      '<p><b>Control settings.</b> Runs differ in the finger joint servo (off in 6 runs, on in the others) and in the arm stage; the finger share separates by the servo setting, so runs are compared only within one setting.</p>' +
      '<p><b>Pictures.</b> Hands: only the hand, held still at the wrist so the shape of the fingers shows; "asked for" sets the 44 finger joints to their asked-for angles. Whole body: the infant alone, drawn upright and turned so its chest faces the front camera, from three sides; "asked for" turns every powered body part to its asked-for direction, each joined to the part before it, and the dots mark the asked-for places of the wrists, head and hips. In "got to", what was asked for is drawn pale over the body. A pale part that lies where the drawn part lies is left out (within 1.5 mm and 1.5° on the hands, 4 mm and 3° on the whole body).</p>' +
      '<p><b>Dates.</b> A command is dated between the file time of the chunk before its own and its own chunk\'s file time, in proportion to its place in the chunk (a chunk is written when its 100 model steps end).</p>' +
      '<p><b>Test movements at each checkpoint.</b> A checkpoint is a saved copy of the model\'s learned numbers at one training model step. Each try is scored by the probe\'s own rule: its end minus the starting line, measured along the asked-for change and divided by that change\'s squared size (0% = at the starting line, 100% = the whole change). The starting line is where the motor controller\'s command ended at the checkpoint that made the test poses. The planner search, direct: the planner changes the command\'s own numbers step by step toward the command the model predicts ends nearest the test pose. The planner search, synergy: the same search over a smaller set of numbers that a learned map (the motor synergy) turns into a whole command. <b>The improvement check</b> of the fixed-start probe (the decision of Oct 18, 2026) compares each planner search at a checkpoint with the same search at checkpoint 494,025, from the same saved starts (the same start each time) toward the same test poses: the direct search decides (met when the whole 95% range of its distance difference is "nearer"), the synergy search is reported beside it. A trial counts only when the request put straight into the command moved the part beyond the starting line, and further than the same change reversed did.</p>' +
      '<p><b>Automatic pictures</b> are chosen from the scores, per body and per hour of recording, and at events, and drawn in a smaller set. The drawing service draws them one at a time, below normal priority, after the memory guard (commit_guard) allows each pass; it pauses while a declared heavy job holds the memory guard\'s heavy-job lock, and on {"paused": true} in Saved/mimo_bridge/movement_viewer/service_control.json. A drawing request from this page goes first.</p>' +
      '<p><b>Where the recordings come from:</b> ' + esc(((S.sources && S.sources.roots) || []).join('; ')) + '.</p>';
  }

  // ------------------------------------------------------------------ wiring
  async function init() {
    if (!$('movement-viewer')) return;
    try {
      S.sources = await getJSON('/api/movement/sources');
    } catch (e) {
      $('mv-status').textContent = 'The saved recordings could not be read: ' + e.message;
      return;
    }
    $('mv-measure').innerHTML = Object.keys(S.sources.measures).map((k) => '<option value="' + esc(k) + '">' + esc((MEASURE_WORDS[k] || [S.sources.measures[k].name])[0]) + ' (' + esc(S.sources.measures[k].unit) + ')</option>').join('');
    const bodies = new Set();
    S.sources.recordings.forEach((r) => r.bodies.forEach((b) => bodies.add(b)));
    $('mv-body').innerHTML = '<option value="all">All ' + bodies.size + ' bodies</option>' + [...bodies].sort((a, b) => a - b).map((b) => '<option value="' + b + '">Body ' + b + '</option>').join('');
    const kindSel = $('mv-kind');
    const group = document.createElement('optgroup');
    group.label = 'One run';
    S.sources.recordings.slice().sort((a, b) => a.first_time - b.first_time).forEach((r) => {
      const o = document.createElement('option');
      o.value = 'rec:' + r.id;
      const ab = r.about || {};
      o.textContent = dayLabel(r.first_time) + ', ' + (ab.name || r.label) + (ab.words ? ': ' + ab.words : ' (' + (r.kind === 'training' ? 'training' : 'test') + ')');
      group.appendChild(o);
    });
    kindSel.appendChild(group);
    S.rowMap = new Map();
    S.sources.recordings.forEach((r) => (r.rows || []).forEach((row) => { if (!S.rowMap.has(row)) S.rowMap.set(row, []); S.rowMap.get(row).push(r); }));
    S.pendingCards.splice(0).forEach(([card, row]) => decorate(card, row));
    ['measure', 'body', 'author'].forEach((k) => { const v = saved(k); if (v && [...$('mv-' + k).options].some((o) => o.value === v)) $('mv-' + k).value = v; });
    { const v = saved('kind'); if (v && !v.startsWith('rec:') && [...kindSel.options].some((o) => o.value === v)) kindSel.value = v; }
    $('mv-touch').checked = saved('touch') === 'hand';
    statusLine();
    about();
    const fr = fullRange();
    if (!fr) { $('mv-reading').textContent = 'No saved recordings were found.'; return; }
    setRange(fr);
    const reload = debounce(() => loadSeries(true), 250);
    ['mv-measure', 'mv-body', 'mv-author'].forEach((id) => $(id).addEventListener('change', () => { saved(id.slice(3), $(id).value); loadSeries(true); }));
    $('mv-touch').addEventListener('change', () => { saved('touch', $('mv-touch').checked ? 'hand' : 'any'); loadSeries(true); });
    kindSel.addEventListener('change', () => { saved('kind', kindSel.value); S.t0 = null; S.t1 = null; const f = fullRange(); if (f) setRange(f); loadSeries(false); });
    $('mv-all-runs').addEventListener('click', () => { kindSel.value = 'all'; kindSel.dispatchEvent(new Event('change')); });
    const onRange = (which) => () => {
      let a = Number($('mv-from').value), b = Number($('mv-to').value);
      if (a > b - 600) { if (which === 'from') { a = b - 600; $('mv-from').value = String(a); } else { b = a + 600; $('mv-to').value = String(b); } }
      // a handle at its end sends no bound, so no command at the edge of the range is left out
      S.t0 = a <= Number($('mv-from').min) ? null : a;
      S.t1 = b >= Number($('mv-to').max) ? null : b;
      rangeText();
      reload();
    };
    $('mv-from').addEventListener('input', onRange('from'));
    $('mv-to').addEventListener('input', onRange('to'));
    $('mv-range-all').addEventListener('click', () => { const f = fullRange(); if (f) { S.t0 = null; S.t1 = null; setRange(f); loadSeries(true); } });
    $('mv-bars').addEventListener('click', (evt) => { const b = evt.target.closest('[data-rec]'); if (b && !b.classList.contains('only')) chooseRun(b.dataset.rec); });
    const slide = debounce(() => loadCommand({ index: Number($('mv-slider').value) }, true), 140);
    $('mv-slider').addEventListener('input', () => { S.userEpoch++; $('mv-pos').textContent = posText(Number($('mv-slider').value)); slide(); });
    document.querySelectorAll('#movement-viewer [data-pick]').forEach((b) => b.addEventListener('click', () => loadCommand({ pick: b.dataset.pick }, true)));
    $('mv-render').addEventListener('click', draw);
    const isvg = $('mv-inside-chart');
    isvg.addEventListener('mousemove', insideTip);
    isvg.addEventListener('mouseleave', () => { $('mv-itip').hidden = true; });
    isvg.addEventListener('click', (evt) => {
      const g = S.igeom;
      if (!g) return;
      const box = isvg.getBoundingClientRect();
      const x = (evt.clientX - box.left) * (g.W / box.width);
      const t = g.tA + (g.tB - g.tA) * (x - g.m.l) / (g.W - g.m.l - g.m.r);
      loadCommand({ at: t }, true);
    });
    $('mv-results').addEventListener('click', onResultClick);
    $('mv-results').addEventListener('input', onResultInput);
    $('mv-gallery').addEventListener('click', openPick);
    const pb = $('mv-pbody');
    [...bodies].sort((a, b) => a - b).forEach((b) => { const o = document.createElement('option'); o.value = String(b); o.textContent = 'Body ' + b; pb.appendChild(o); });
    $('mv-goal').addEventListener('change', loadProbes);
    $('mv-pbody').addEventListener('change', loadProbes);
    $('mv-prun').addEventListener('change', () => { probeBars(); probeReadout(); });
    $('mv-ptbody').addEventListener('change', probeReadout);
    $('mv-parm').addEventListener('change', probeReadout);
    $('mv-pdraw').addEventListener('click', drawProbe);
    $('mv-probe-bars').addEventListener('click', (evt) => { const b = evt.target.closest('[data-run]'); if (b) { $('mv-prun').value = b.dataset.run; probeBars(); probeReadout(); } });
    window.addEventListener('resize', debounce(() => { inside(); }, 150));
    loadProbes();
    await loadSeries(false);
    setInterval(async () => { try { S.sources = await getJSON('/api/movement/sources'); statusLine(); } catch (e) { /* the next read tries again */ } }, 60000);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
