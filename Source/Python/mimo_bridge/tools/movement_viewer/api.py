# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""What the dashboard's /api/movement/ routes return (development_dashboard.Handler calls these).

Read-only on every recording. A request names a recording by its id from sources.json, a body number and a model step;
any other value is refused (no file path is ever taken from a request). A drawing request is written to
Saved/mimo_bridge/movement_viewer/requests/ for the viewer service (service.py), which this module starts when its
heartbeat is missing or older than SERVICE_STALE_S.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import HOME, REQUESTS_DIR, STATUS_FILE, part_path, pictures_dir, replace_file
from . import auto as A
from . import commands as C
from . import index as I
from . import probes as P
from . import render as R
from . import sources as S

SERVICE_STALE_S = 90
_KEY = re.compile(r"[0-9a-f]{20}")
_IMAGE = re.compile(r"(hand_(left|right)_(front|side)|body_(front|left|right|wrist))_(start|target|t\d{3,5})\.png")
_LAST_START = [0.0]


def _q(query: Dict[str, list], name: str, default=None):
    v = query.get(name)
    return v[0] if v else default


def _float(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None


def _time(query: Dict[str, list], name: str) -> Optional[float]:
    """A date bound (seconds since 1970) or None when the query has none; a bound that is not a number is refused
    instead of being dropped (the second review of Oct 18, 2026: a malformed bound read as no bound)."""
    v = _q(query, name)
    if v is None or v == "":
        return None
    x = _float(v)
    if x is None:
        raise ValueError("%s must be a time in seconds since 1970" % name)
    return x


def _read(path) -> Optional[dict]:
    try:
        with open(str(path), encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _running(pid) -> bool:
    """True while process `pid` runs the viewer service (its command line names the package)."""
    try:
        import psutil
        return "movement_viewer" in " ".join(psutil.Process(int(pid)).cmdline())
    except Exception:
        return False


def service_state() -> dict:
    st = _read(STATUS_FILE) or {}
    beat = st.get("heartbeat")
    alive = bool(beat and time.time() - float(beat) < SERVICE_STALE_S and _running(st.get("pid")))
    return dict(alive=alive, heartbeat=beat, pid=st.get("pid"), state=st.get("state"), queue=st.get("queue"),
                paused_by=st.get("paused_by"),
                guard=st.get("guard"), last_drawing=st.get("last_drawing"), background=st.get("background"),
                picks_total=st.get("picks_total"), picks_waiting=st.get("picks_waiting"), errors=st.get("errors"),
                memory_mb=st.get("memory_mb"), priority=st.get("priority"), power_throttling=st.get("power_throttling"),
                pictures_dir=st.get("pictures_dir"), paused=st.get("paused"))


def ensure_service() -> dict:
    """Start the viewer service when no live one answers (its own owner lock refuses a second)."""
    state = service_state()
    if state["alive"] or time.time() - _LAST_START[0] < SERVICE_STALE_S:
        return state
    _LAST_START[0] = time.time()
    os.makedirs(str(HOME), exist_ok=True)
    flags = 0
    if os.name == "nt":
        flags = (getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                 | getattr(subprocess, "CREATE_NO_WINDOW", 0))
    out = open(str(HOME / "service.log"), "a", encoding="utf-8")
    err = open(str(HOME / "service.err.log"), "a", encoding="utf-8")
    try:
        subprocess.Popen([sys.executable, "-B", "-m", "mimo_bridge.tools.movement_viewer", "service"],
                         cwd=str(Path(S.REPO) / "Source" / "Python"), stdout=out, stderr=err,
                         stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
    finally:
        out.close()
        err.close()
    state["starting"] = True
    return state


def _recordings(query: Dict[str, list]) -> List[dict]:
    """The recordings a query selects: rec=<id,id,...> or kind=training|test|all (default all)."""
    recs = S.load().get("recordings") or []
    ids = [x for x in (_q(query, "rec", "") or "").split(",") if x]
    if ids:
        known = {r["id"]: r for r in recs}
        bad = [x for x in ids if x not in known]
        if bad:
            raise ValueError("unknown recording id %s" % bad[0])
        return [known[x] for x in ids]
    kind = _q(query, "kind", "all")
    if kind not in ("all", "training", "test"):
        raise ValueError("kind must be all, training or test")
    return [r for r in recs if kind == "all" or r["kind"] == kind]


def _filters(query: Dict[str, list]):
    body = _q(query, "body", "all")
    if body != "all":
        body = int(body)
        if not 0 <= body < 64:
            raise ValueError("body out of range")
    measure = _q(query, "measure", "fingers_L")
    if measure not in C.MEASURES:
        raise ValueError("unknown measure")
    author = _q(query, "author", "planner")
    if author not in ("planner", "all"):
        author = str(int(author))
    touch = _q(query, "touch", "any")
    if touch not in ("any", "hand"):
        raise ValueError("touch must be any or hand")
    return body, measure, author, _time(query, "t0"), _time(query, "t1"), touch


def sources_view(query: Dict[str, list]) -> dict:
    data = S.load()
    recs = []
    for r in data.get("recordings") or []:
        metas = [I.read_meta(r["id"], int(b)) for b in r["bodies"]]
        times = [(m or {}).get("first_time") for m in metas] + [(m or {}).get("last_time") for m in metas]
        times = [t for t in times if t is not None]
        first = min(times) if times else r["first_time"]
        recs.append(dict(id=r["id"], label=r["label"], run=r["run"], kind=r["kind"], rows=r.get("rows") or [],
                         rows_total=r.get("rows_total", 0), first_step=r["first_step"], last_step=r["last_step"],
                         first_time=first,
                         last_time=max(times) if times else r["last_time"], chunks=r["chunks"],
                         settings=r.get("settings") or {},
                         bodies=sorted(int(b) for b in r["bodies"]), where=r.get("where"),
                         commands=sum(int((m or {}).get("commands") or 0) for m in metas),
                         indexed=all(m is not None for m in metas), args=r.get("args") or {},
                         about=S.about_of(dict(r, first_time=first))))
    return dict(scanned_at=data.get("scanned_at"), roots=data.get("roots"), recordings=recs,
                measures={k: dict(name=v[0], unit=v[1], definition=v[2]) for k, v in C.MEASURES.items()},
                authors=I.AUTHOR_NAMES, service=ensure_service(), pictures_dir=str(pictures_dir()))


def series_view(query: Dict[str, list]) -> dict:
    recs = _recordings(query)
    body, measure, author, t0, t1, touch = _filters(query)
    out = I.series(recs, body, measure, author, t0, t1, touch=touch)
    out["markers"] = markers(recs)
    return out


def command_view(query: Dict[str, list]) -> dict:
    """One command of the selection (index, pick or nearest to a time), or, with step (and one rec and body), exactly
    the command that starts at that model step."""
    recs = _recordings(query)
    body, measure, author, t0, t1, touch = _filters(query)
    pick = _q(query, "pick")
    if pick not in (None, "best", "median", "worst"):
        raise ValueError("pick must be best, median or worst")
    step = _q(query, "step")
    if step is not None:
        if len(recs) != 1 or body == "all":
            raise ValueError("an exact command needs one recording and one body")
        sel = I.selection(recs, body, measure, author, None, None, touch)
        hit = [i for i in range(len(sel["step"])) if int(sel["step"][i]) == int(step)]
        if not hit:
            raise ValueError("no scored command starts at model step %s" % step)
        out = dict(count=len(sel["step"]), index=hit[0], command=I._ref(sel, hit[0], recs))
    elif _q(query, "keep_rec") is not None:
        # the command already chosen, found again in a new selection (another date range): its new position, or the
        # command nearest its time when the new selection no longer holds it
        sel = I.selection(recs, body, measure, author, t0, t1, touch)
        ids = [r["id"] for r in recs]
        rid, kb, ks = _q(query, "keep_rec"), int(_q(query, "keep_body", "-1")), int(_q(query, "keep_step", "-1"))
        hit = [i for i in range(len(sel["step"])) if int(sel["step"][i]) == ks and int(sel["body"][i]) == kb
               and ids[int(sel["rec"][i])] == rid]
        if hit:
            out = dict(count=len(sel["step"]), index=hit[0], command=I._ref(sel, hit[0], recs))
        else:
            out = I.command_at(recs, body, measure, author, t0, t1, at_time=_float(_q(query, "at")), touch=touch)
    else:
        idx = _q(query, "index")
        out = I.command_at(recs, body, measure, author, t0, t1, index=int(idx) if idx is not None else None,
                           pick=pick, at_time=_float(_q(query, "at")), touch=touch)
    cmd = out.get("command")
    if cmd:
        cmd["key"] = R.key_for(cmd["recording"], cmd["body"], cmd["step"])
        cmd["drawn"] = os.path.isfile(os.path.join(str(pictures_dir()), cmd["key"], "manifest.json"))
    return out


def _request_path(key: str) -> Path:
    return REQUESTS_DIR / ("%s.json" % key)


def render_request(body: dict) -> dict:
    if body.get("probe"):
        return _probe_request(body)
    rid = str(body.get("recording") or "")
    rec = S.by_id(rid)
    if rec is None:
        raise ValueError("unknown recording id")
    b = int(body.get("body"))
    step = int(body.get("step"))
    if str(b) not in rec["bodies"]:
        raise ValueError("that recording has no body %d" % b)
    key = R.key_for(rid, b, step)
    st = render_status({"key": [key]})
    if st["state"] in ("ready", "queued"):
        return st
    os.makedirs(str(REQUESTS_DIR), exist_ok=True)
    for p in (REQUESTS_DIR / ("%s.error.json" % key), REQUESTS_DIR / ("%s.done.json" % key)):
        try:
            os.remove(str(p))
        except OSError:
            pass
    tmp = part_path(_request_path(key))
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(key=key, recording=rid, body=b, step=step, requested_at=time.time()), f)
    replace_file(tmp, _request_path(key))
    service = ensure_service()
    return dict(state="queued", key=key, service=service)


def _probe_request(body: dict) -> dict:
    pid = str(body.get("probe") or "")
    run = P.by_id(pid)
    if run is None:
        raise ValueError("unknown probe run")
    b, goal, arm = int(body.get("body")), str(body.get("goal")), str(body.get("arm"))
    if goal not in P.GOAL_NAMES or arm not in P.ARMS or not 0 <= b < 64:
        raise ValueError("unknown goal, arm or body")
    key = P.key_for(pid, b, goal, arm)
    st = render_status({"key": [key]})
    if st["state"] in ("ready", "queued"):
        return st
    os.makedirs(str(REQUESTS_DIR), exist_ok=True)
    for p_ in (REQUESTS_DIR / ("%s.error.json" % key), REQUESTS_DIR / ("%s.done.json" % key)):
        try:
            os.remove(str(p_))
        except OSError:
            pass
    tmp = part_path(_request_path(key))
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(key=key, probe=pid, body=b, goal=goal, arm=arm, requested_at=time.time()), f)
    replace_file(tmp, _request_path(key))
    return dict(state="queued", key=key, service=ensure_service())


def probes_view(query: Dict[str, list]) -> dict:
    goal = _q(query, "goal", "curl_witness")
    body = _q(query, "body", "all")
    if body != "all":
        body = int(body)
    out = P.series(goal, body)
    out["parts"] = {g: dict(part=v[0], unit=v[1]) for g, v in P.GOAL_PART.items()}
    return out


def probe_trials_view(query: Dict[str, list]) -> dict:
    """The trials of one probe run (and goal): per body, the share of the requested change each try made beyond the
    zero point (probes.request_shares: the endpoint of the controller's command at zero_point_step), the size of that
    change and whether the trial is informative; own_baseline when that is the run's own controller, whose share is
    then 0 by definition; per try, whether its pictures are drawn."""
    pid = _q(query, "probe", "")
    goal = _q(query, "goal")
    run = P.by_id(pid)
    if run is None:
        raise ValueError("unknown probe run")
    rows = [dict(r) for r in P.shares_of_run(run) if goal in (None, r["goal"])]
    for r in rows:
        r["own_baseline"] = r.get("zero_point_step") == run["checkpoint_step"]
        r["drawn"] = {arm: os.path.isfile(os.path.join(str(pictures_dir()), P.key_for(pid, r["body"], r["goal"], arm),
                                                       "manifest.json")) for arm in P.ARMS}
    return dict(probe=pid, trials=rows, goals_from=P.goals_from(run), mode=run["mode"],
                checkpoint_step=run["checkpoint_step"], gain=P.gains().get(P._folder_key(run["folder"])))


def render_status(query: Dict[str, list]) -> dict:
    key = _q(query, "key", "")
    if not _KEY.fullmatch(key or ""):
        raise ValueError("not a drawing key")
    man = _read(os.path.join(str(pictures_dir()), key, "manifest.json"))
    if man is not None and not man.get("refused"):
        man["state"] = "ready"
        return man
    if man is not None:
        return dict(state="error", key=key, error=man.get("refused"))
    err = _read(REQUESTS_DIR / ("%s.error.json" % key))
    if err is not None:
        return dict(state="error", key=key, error=err.get("error"))
    if _request_path(key).is_file():
        waiting = sorted(p for p in os.listdir(str(REQUESTS_DIR)) if p.endswith(".json")
                         and not p.endswith((".done.json", ".error.json")))
        return dict(state="queued", key=key, position=waiting.index(key + ".json") + 1 if key + ".json" in waiting
                    else None, service=service_state())
    return dict(state="none", key=key)


def image_path(query: Dict[str, list]) -> Path:
    key, name = _q(query, "key", ""), _q(query, "name", "")
    if not _KEY.fullmatch(key or "") or not _IMAGE.fullmatch(name or ""):
        raise ValueError("not a picture of the viewer")
    return Path(pictures_dir()) / key / name


def held_steps(man: dict) -> int:
    """How many model steps the drawn command was held: its last record's model step minus its first, plus one (the
    second review of Oct 18, 2026, finding 7: the count of drawn samples gave 1 for an automatic drawing, which draws
    only the end)."""
    first = int(man.get("step") or 0)
    return max(1, int(man.get("last_step") or first) - first + 1)


def export_path(query: Dict[str, list]) -> Path:
    """One drawing as a single HTML file (pictures inside it), with its dates, sources and measures: written beside the
    pictures as export.html when missing or older than the manifest, and returned for the dashboard to send."""
    import base64
    import html
    key = _q(query, "key", "")
    if not _KEY.fullmatch(key or ""):
        raise ValueError("not a drawing key")
    folder = Path(pictures_dir()) / key
    man_path = folder / "manifest.json"
    man = _read(man_path)
    if man is None or man.get("refused"):
        raise ValueError("no drawing with that key")
    out = folder / "export.html"
    if out.is_file() and out.stat().st_mtime >= man_path.stat().st_mtime:
        return out
    esc = html.escape
    samples = man.get("samples_ms") or [250]
    last = "t%03d" % int(samples[-1])

    def when(t):
        if t is None:
            return "not recorded"
        lt = time.localtime(float(t))
        return time.strftime("%B %d, %Y, %I:%M %p ", lt).replace(" 0", " ") + ("PDT" if lt.tm_isdst else "PST")

    def pic(name, alt):
        p = folder / name
        if not p.is_file():
            return '<div class="miss">%s: not saved</div>' % esc(alt)
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        return '<img alt="%s" src="data:image/png;base64,%s">' % (esc(alt), data)

    def arrow(m):
        if not m:
            return ""
        unit = "°" if m.get("unit") == "degrees" else " cm"
        parts = []
        if m.get("start") is not None:
            parts.append("%.1f%s at the start" % (m["start"], unit))
        if m.get("end") is not None:
            parts.append("%.1f%s at the end" % (m["end"], unit))
        if m.get("after_next") is not None:
            parts.append("%.1f%s after the next command" % (m["after_next"], unit))
        return "; ".join(parts)
    M = man.get("measures") or {}
    rows = []
    for side, label, fk, wk in (("left", "Left hand", "fingers_L", "wrist_L"), ("right", "Right hand", "fingers_R", "wrist_R")):
        rows.append('<tr><th>%s<small>finger angles: %s</small><small>wrist place: %s</small></th><td>%s</td><td>%s</td><td>%s</td></tr>' % (
            label, esc(arrow(M.get(fk))), esc(arrow(M.get(wk))),
            pic("hand_%s_front_start.png" % side, label + " at the start"),
            pic("hand_%s_front_target.png" % side, label + " as asked for"),
            pic("hand_%s_front_%s.png" % (side, last), label + " at the end, the asked-for shape pale over it")))
    places = "; ".join("%s %.1f cm from its dot" % (x.get("part") or x.get("name"), float(x.get("asked_for_body_cm") or 0))
                       for x in man.get("asked_for_places") or [] if x.get("asked_for_body_cm") is not None)
    target_caption = ("the pose a reference command reached from the same start (the test pose)" if man.get("kind") == "probe"
                      else "the asked-for directions only: every body part the command powers turned to the direction it "
                      "asks for, each joined to the part before it; not a pose the body can necessarily reach. Dots: "
                      "the asked-for places of the wrists, head and hips" + (" (in this body: %s)" % places if places else ""))
    for view in man.get("body_views") or ["front"]:
        words = R.BODY_VIEW_NAMES.get(view, view)
        rows.append('<tr><th>Whole body, %s<small>directions: %s</small></th><td>%s</td><td>%s<small>%s</small></td>'
                    '<td>%s</td></tr>' % (esc(words), esc(arrow(M.get("body_turn"))),
                                          pic("body_%s_start.png" % view, "Whole body at the start, " + words),
                                          pic("body_%s_target.png" % view, "Whole body as asked for, " + words),
                                          esc(target_caption),
                                          pic("body_%s_%s.png" % (view, last), "Whole body at the end, " + words)))
    prov = man.get("provenance") or {}
    files = "".join("<li>%s, SHA-256 %s</li>" % (esc(p), esc(h)) for p, h in (prov.get("chunks") or {}).items())
    if man.get("kind") == "probe":
        title = "Probe: %s, body %s, %s" % (man.get("goal_name"), man.get("body"), man.get("arm"))
        ident = "%s; probe run %s" % (man.get("label"), when(man.get("time")))
    else:
        title = "Body %s, model step %s, %s" % (man.get("body"), format(int(man.get("step") or 0), ","), man.get("label"))
        held = held_steps(man)
        ident = "%s run; recorded %s; held %d model steps (%d ms)" % (
            "Training" if man.get("kind") == "training" else "Test", when(man.get("time")), held, held * 50)
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>%s</title><style>body{font:15px/1.5 "Segoe UI",Arial,sans-serif;color:#162a32;background:#fff;margin:24px;max-width:1100px}
h1{font-size:22px;margin:0 0 4px}p{margin:6px 0}table{border-collapse:collapse;width:100%%}th{text-align:left;vertical-align:top;width:180px;padding:6px 8px 6px 0;font-size:14px}
th small,td small{display:block;font-weight:400;color:#536770;font-size:12px;margin-top:4px}td{padding:4px;vertical-align:top;width:30%%}
img{width:100%%;height:auto;border-radius:6px;background:#f0f3f2}.miss{border:1px dashed #c9d4d6;border-radius:6px;padding:30px 8px;text-align:center;color:#536770;font-size:12px}
.muted{color:#536770;font-size:12px}li{font-size:12px;word-break:break-all}</style></head><body>
<h1>%s</h1><p class="muted">%s. Pictures made %s. Exported %s from the movement viewer of the UEMA development dashboard.</p>
<p class="muted">Columns: start (before the command), asked for, got to (after %d ms, with the asked-for shape drawn pale
over it). The whole body is drawn alone and upright, turned so its chest faces the front camera (%s at the start). Drawn
from the saved records by forward kinematics of the recording's saved model; no training, no new movement.</p>
<table><tr><th></th><th>Start</th><th>Asked for</th><th>Got to</th></tr>%s</table>
<p class="muted">%s</p><p class="muted">Sources: %s; model %s (structure fingerprint %s).</p><ul>%s</ul></body></html>""" % (
        esc(title), esc(title), esc(ident), esc(when(man.get("rendered_at"))), esc(when(time.time())), int(samples[-1]),
        esc(man.get("posture") or "posture not recorded"), "".join(rows), esc(". ".join(man.get("notes") or [])), esc(prov.get("selection") or ""), esc(str(prov.get("model"))),
        esc(str(prov.get("model_structure_fingerprint"))[:16]), files)
    tmp = part_path(out)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(page)
    replace_file(tmp, out)
    return out


def auto_view(query: Dict[str, list]) -> dict:
    recs = _recordings(query)
    ids = {r["id"] for r in recs}
    body = _q(query, "body", "all")
    t0, t1 = _time(query, "t0"), _time(query, "t1")
    data = A.load()
    root = str(pictures_dir())
    out = []
    for p in data.get("picks") or []:
        if p["recording"] not in ids or (body != "all" and int(body) != int(p["body"])):
            continue
        if (t0 is not None and p["time"] < t0) or (t1 is not None and p["time"] > t1):
            continue
        key = R.key_for(p["recording"], p["body"], p["step"], True)       # automatic picks are drawn LITE
        man = os.path.join(root, key, "manifest.json")
        out.append(dict(p, key=key, state="drawn" if os.path.isfile(man) else "waiting", _man=man))
    limit = int(_q(query, "limit", "60"))
    shown = out[:limit]
    for p in shown:
        # the thumbnail: the whole body from the front at the command's last saved sample
        m = _read(p["_man"]) if p["state"] == "drawn" else None
        if m is not None and m.get("refused"):
            p["state"] = "refused"
        elif m is not None and m.get("samples_ms"):
            last = int(m["samples_ms"][-1])
            p["thumb"] = "body_front_t%03d.png" % last
            # a pick chosen for its hands error is shown by its hands (the independent review of Oct 19, 2026: the
            # fingers cannot be seen on a whole body that small)
            if any(("hands distance" in str(x) or "hands error" in str(x)) for x in p.get("reasons") or []):
                p["thumbs"] = ["hand_left_front_t%03d.png" % last, "hand_right_front_t%03d.png" % last]
    for p in out:
        p.pop("_man", None)
    return dict(computed_at=data.get("computed_at"), rules=data.get("rules"), total=len(out), picks=shown,
                drawn=sum(1 for p in out if p["state"] == "drawn"),
                waiting=sum(1 for p in out if p["state"] == "waiting"), service=service_state())


STAGE_NAMES = {"observed_once": "observed once", "repeated": "repeated", "generalized": "generalized",
               "predicted_before_action": "predicted before action"}


def markers(recs: List[dict]) -> List[dict]:
    """The observer's timeline events (development.json: model step, ability, stage). The timeline names no run, and
    runs resumed from one checkpoint cover the same model steps, so an event is dated only when exactly ONE run the
    observer digested (development.json's runs) covers its model step, from that run's first indexed command at or after
    the step, and only when that run is among the selected recordings (the second review of Oct 18, 2026, finding 2:
    events were dated on test runs the observer never read). Every other event comes back without a time and with
    the reason (`why`)."""
    dev = _read(Path(S.REPO) / "Saved" / "mimo_bridge" / "observer" / "H84_fresh_drivers_20260918" / "development.json")
    digested = set((dev or {}).get("runs") or [])
    every = S.load().get("recordings") or []
    selected = {r["id"] for r in recs}
    out = []
    for ev in (dev or {}).get("timeline") or []:
        step = int(ev.get("step", -1))
        covering = [r for r in every if r["run"] in digested and r["first_step"] <= step <= r["last_step"]]
        runs = sorted({r["run"] for r in covering})
        when, why, run = None, None, None
        if not runs:
            why = "no saved recording of a run the observer digested covers model step %s" % format(step, ",")
        elif len(runs) > 1:
            why = ("%d runs the observer digested cover model step %s (runs resumed from one checkpoint), and the "
                   "observer's timeline names none of them" % (len(runs), format(step, ",")))
        else:
            run = runs[0]
            r = next((x for x in covering if x["id"] in selected), None)
            if r is None:
                why = "the run that covers it (%s) is not among the recordings shown" % run
            else:
                cols = I.body_cols(r["id"], min(int(b) for b in r["bodies"]))
                if cols is not None and len(cols["step"]) and bool((cols["step"] >= step).any()):
                    when = float(cols["time"][int((cols["step"] >= step).argmax())])
                else:
                    why = "its run has no indexed command at or after model step %s" % format(step, ",")
        out.append(dict(step=step, capability=ev.get("cap"), name=ev.get("name"), stage=ev.get("stage"),
                        stage_name=STAGE_NAMES.get(str(ev.get("stage")), str(ev.get("stage")).replace("_", " ")),
                        time=when, run=run, why=why))
    return out
