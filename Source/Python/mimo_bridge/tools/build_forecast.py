# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Forecast when the build's remaining work ends, from the MEASURED run times of agents that finished.

Operator 2026-09-23: "You need to write a program to give me time estimates for these agents
remaining". This is that program. It predicts nothing by judgement: every duration it uses is a
measured one.

- How long a job takes: every finished workflow agent of this repository (the journal holds its
  result) gives one measured run time, from the first to the last timestamp in its transcript. Jobs
  are grouped by kind from their label (build, fix, verify, refute, review, find, survey, other), because a
  model build and a re-check take different times.
- A running agent's time left: drawn from the finished agents of its kind that ran LONGER than it has
  been running so far, minus what it has already run. When it has run longer than every finished
  agent of its kind, there is no measured basis; its time left counts as 0 and the forecast is marked
  a lower bound.
- An agent waiting its turn, or one the plan says has not started: a whole run time drawn from its
  kind.
- Steps that are not agents (a training checkpoint, the probe suite) use the measured duration the
  plan names with its source.
- The plan, docs/gain_of_function_v3/build_plan.json, lists the stages to 100% done and what each
  waits on; a stage that is a workflow reads its agents live from build_status's records.
- The spread: 2,000 draws; the forecast gives the 10th, 50th and 90th percentile finish times, so the
  real finish lands between the first and the last in 8 of 10 cases IF the future jobs are like the
  measured ones. Not included: pauses when the usage limit stops agents (one did at about 02:00 on
  2026-09-23), or work the plan does not list.
- Agents started one at a time (the Agent tool, the build's way since 2026-09-25) count as well: their
  transcripts (<project>/<session>/subagents/agent-<id>.jsonl, the description in agent-<id>.meta.json)
  hold one or more JOBS, each begun by a prompt (the first task, or a message the coordinator sent
  after the agent finished). A job the usage limit stopped continues when the agent is resumed and
  counts as one job, without the stopped gap. Finished jobs give measured run times: an agent's first
  job by its description's first word ("Review ...", "Build ...", "Fix ...", "Find ..."), a later job
  as a fix, since the coordinator resumes an agent to act on a review of its work. A plan item names
  a live agent by its description ("match") and the time its job began ("since").
- Training-pause jobs (a stage with "pause"): each job not yet done takes a run time measured for a
  job of the same name in an earlier pause's receipt; a job never measured counts as 0 and marks the
  forecast a lower bound.
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PLAN = Path(__file__).resolve().parents[4] / "docs" / "gain_of_function_v3" / "build_plan.json"
DRAWS = 2000
MIN_POOL = 3            # a kind with fewer measured run times borrows a wider pool
RECENT_H = 36           # agents of this build: started in the last 36 h (the build page's window)
KINDS = (("build", r"^(b\d-|lane-|env-senses$|ops-storage$|probes$|dashboards$|explainer-update$)"),
         ("fix", r"^(fix-|measure-|addendum|merge)"),
         ("refute", r"^verify-r\d"),        # the sweep's one-finding refutation checks (minutes)
         ("verify", r"^verify"),                # re-checking a whole fix agent's work
         ("review", r"^review"),
         ("find", r"^find-"),
         ("survey", r"^survey"))
_TS = re.compile(rb'"timestamp"\s*:\s*"([0-9T:.\-]+)Z?"')
_SPAN_CACHE: dict = {}


def kind_of(label: str) -> str:
    for kind, pat in KINDS:
        if re.search(pat, label or ""):
            return kind
    return "other"


def _parse(ts: bytes):
    try:
        return datetime.fromisoformat(ts.decode().rstrip("Z") + "+00:00").timestamp()
    except ValueError:
        return None


def transcript_span(path: Path):
    """(first, last) timestamp in an agent transcript, in epoch seconds, or (None, None)."""
    try:
        st = path.stat()
    except OSError:
        return None, None
    key = str(path)
    hit = _SPAN_CACHE.get(key)
    if hit and hit[0] == (st.st_size, st.st_mtime):
        return hit[1]
    first = last = None
    try:
        with open(path, "rb") as f:
            m = _TS.search(f.read(262_144))
            first = _parse(m.group(1)) if m else None
            f.seek(max(0, st.st_size - 262_144))
            ms = _TS.findall(f.read())
            last = _parse(ms[-1]) if ms else None
    except OSError:
        pass
    _SPAN_CACHE[key] = ((st.st_size, st.st_mtime), (first, last))
    return first, last


def measured_durations(project_dirs, now: float = None) -> dict:
    """{kind: [minutes, ...]} over every finished workflow agent of the repository, plus 'all'; the
    same for this build's agents alone (started in the last RECENT_H hours) under 'recent:<kind>'."""
    now = time.time() if now is None else now
    pools = defaultdict(list)
    for proj in project_dirs:
        for jp in glob.glob(str(Path(proj) / "*" / "subagents" / "workflows" / "*" / "journal.jsonl")):
            label, done = {}, set()
            try:
                with open(jp, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        if not isinstance(rec, dict):
                            continue
                        if rec.get("type") == "started":
                            label[rec.get("agentId")] = rec.get("label") or ""
                        elif rec.get("type") == "result":
                            done.add(rec.get("agentId"))
            except OSError:
                continue
            for aid in done:
                first, last = transcript_span(Path(jp).parent / ("agent-%s.jsonl" % aid))
                if first is None or last is None or last <= first:
                    continue
                minutes = (last - first) / 60.0
                kind = kind_of(label.get(aid, ""))
                pools[kind].append(minutes)
                pools["all"].append(minutes)
                if now - first < RECENT_H * 3600:
                    pools["recent:" + kind].append(minutes)
                    pools["recent:all"].append(minutes)
    _add_subagent_durations(pools, project_dirs, now)
    return dict(pools)


# ------------------------------------------------------------------ agents started one at a time
SUB_KINDS = (("review", r"^(review|re-review|independent review|second review|third review)\b"),
             ("build", r"^(build|implement|add|make|write|create)\b"),
             ("fix", r"^(fix|repair|correct|act on|merge|finish)\b"),
             ("find", r"^(find|diagnose|investigate|measure|check|confirm|survey)\b"))
RUNNING_S = 1800        # a job with a record in the last 30 minutes is running; older and open, silent
_LIMIT_TEXT = re.compile(r"^\s*(You've hit your \w+ limit|API Error)", re.I)
_JOBS_CACHE: dict = {}


def kind_of_description(text: str) -> str:
    for kind, pat in SUB_KINDS:
        if re.search(pat, (text or "").strip().lower()):
            return kind
    return "other"


def _record(line: str):
    """(time, is_prompt, assistant_state) of one transcript line; assistant_state is None for a
    record that is not the agent's, 'error' for the API's limit or error message, 'end' for a turn
    that ended in text without a tool call, else 'working'."""
    try:
        rec = json.loads(line)
    except ValueError:
        return None
    if not isinstance(rec, dict):
        return None
    ts = rec.get("timestamp")
    t = _parse(ts.encode()) if isinstance(ts, str) else None
    if t is None:
        return None
    msg = rec.get("message") if isinstance(rec.get("message"), dict) else {}
    content = msg.get("content")
    parts = [p for p in content if isinstance(p, dict)] if isinstance(content, list) else []
    if rec.get("type") == "user":
        if rec.get("toolEndsTurn"):
            # the result of a tool that ends the agent's turn (SubagentHandback, Oct 2026): the agent
            # delivered its report through a tool call, so no text-only end_turn record follows
            return t, False, "end"
        if isinstance(content, str):
            text = content
        elif parts and not any(p.get("type") == "tool_result" for p in parts):
            text = " ".join(str(p.get("text") or "") for p in parts)
        else:
            return t, False, None
        text = text.lstrip()
        prompt = bool(text) and not text.startswith("<system-reminder>") and not text.startswith(
            "This session is being continued")
        return t, prompt, None
    if rec.get("type") == "assistant":
        tools = any(p.get("type") == "tool_use" for p in parts)
        text = " ".join(str(p.get("text") or "") for p in parts if p.get("type") == "text")
        if rec.get("isApiErrorMessage") or (not tools and _LIMIT_TEXT.search(text)):
            return t, False, "error"
        if not tools and msg.get("stop_reason") == "end_turn":
            return t, False, "end"
        return t, False, "working"
    return t, False, None


def _prompt_text(line: str) -> str:
    try:
        content = (json.loads(line).get("message") or {}).get("content")
    except (ValueError, AttributeError):
        return ""
    if isinstance(content, list):
        content = " ".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
    return str(content or "")[:4000]


def job_kind(description: str, index: int, prompt: str) -> str:
    """An agent's first job by its description; a later one is a fix when its prompt names a review
    (the coordinator resumes an author to act on one), else a follow-up: on 2026-09-25 those were
    answers to a question or a small addition, 0.7 to 9.4 minutes, and counted as fixes they pulled
    the fixes' median from 37-75 minutes down to 7.8."""
    if index == 0:
        return kind_of_description(description)
    return "fix" if re.search(r"\breview", prompt or "", re.I) else "follow-up"


def subagent_jobs(path: Path, now: float = None) -> list:
    """The jobs in one agent transcript, oldest first: {start, end, minutes, status}. status is
    'finished', 'limit' (stopped by the usage limit, waiting to be resumed), 'running' (a record in the
    last RUNNING_S) or 'silent' (open, no record for longer). A prompt that arrives while a job is
    still open, or after the limit stopped it, continues that job; the gap after a limit stop is not
    counted as working time."""
    now = time.time() if now is None else now
    try:
        st = path.stat()
    except OSError:
        return []
    key = str(path)
    hit = _JOBS_CACHE.get(key)
    if hit and hit[0] == (st.st_size, st.st_mtime):
        segs = hit[1]
    else:
        segs = []
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    r = _record(line)
                    if r is None:
                        continue
                    t, prompt, state = r
                    if prompt or not segs:
                        segs.append({"start": t, "last": t, "state": None,
                                     "prompt": _prompt_text(line) if prompt else ""})
                    seg = segs[-1]
                    seg["last"] = max(seg["last"], t)
                    if state is not None:
                        seg["state"] = state
        except OSError:
            return []
        _JOBS_CACHE[key] = ((st.st_size, st.st_mtime), segs)
    jobs = []
    for seg in segs:
        ended = {"end": "finished", "error": "limit"}.get(seg["state"], "open")
        if jobs and jobs[-1]["status"] in ("limit", "open"):
            job = jobs[-1]
            # an open job kept working through the gap; a stopped one did not
            job["minutes"] += ((seg["last"] - seg["start"]) if job["status"] == "limit"
                               else (seg["last"] - job["end"])) / 60.0
            job.update(end=seg["last"], status=ended)
        else:
            jobs.append({"start": seg["start"], "end": seg["last"], "minutes": (seg["last"] - seg["start"]) / 60.0,
                         "status": ended, "prompt": seg.get("prompt", "")})
    if jobs and jobs[-1]["status"] == "open":
        jobs[-1]["status"] = "running" if now - jobs[-1]["end"] < RUNNING_S else "silent"
    return jobs


def subagents(project_dirs, now: float = None) -> dict:
    """{description: [jobs]} for the agents of the last RECENT_H hours started one at a time; when two
    share a description, the newest transcript wins."""
    now = time.time() if now is None else now
    found = {}
    for proj in project_dirs:
        for mp in glob.glob(str(Path(proj) / "*" / "subagents" / "agent-*.meta.json")):
            tp = Path(mp[:-len(".meta.json")] + ".jsonl")
            try:
                mtime = tp.stat().st_mtime
                desc = json.loads(Path(mp).read_text(encoding="utf-8")).get("description") or ""
            except (OSError, ValueError, AttributeError):
                continue
            if now - mtime > RECENT_H * 3600 or not desc:
                continue
            if desc not in found or found[desc][0] < mtime:
                found[desc] = (mtime, tp)
    return {d: subagent_jobs(tp, now) for d, (_m, tp) in found.items()}


def _add_subagent_durations(pools, project_dirs, now):
    for proj in project_dirs:
        for mp in glob.glob(str(Path(proj) / "*" / "subagents" / "agent-*.meta.json")):
            tp = Path(mp[:-len(".meta.json")] + ".jsonl")
            try:
                desc = json.loads(Path(mp).read_text(encoding="utf-8")).get("description") or ""
            except (OSError, ValueError, AttributeError):
                continue
            for i, job in enumerate(subagent_jobs(tp, now)):
                if job["status"] != "finished" or job["minutes"] <= 0:
                    continue
                kind = job_kind(desc, i, job.get("prompt"))
                pools[kind].append(job["minutes"])
                pools["all"].append(job["minutes"])
                if now - job["start"] < RECENT_H * 3600:
                    pools["recent:" + kind].append(job["minutes"])
                    pools["recent:all"].append(job["minutes"])


def _epoch(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return _parse(str(value).encode())


def _matched_job(item: dict, agents: dict):
    """The live agent's job a plan item names: the first job of the agent with that description that
    began at or after the item's 'since' (one minute's tolerance), else None (not started)."""
    jobs = (agents or {}).get(item.get("match") or "") or []
    since = _epoch(item.get("since"))
    for job in jobs:
        if since is None or job["start"] >= since - 60:
            return job
    return None


# ------------------------------------------------------------------ training-pause jobs
PAUSES = Path(__file__).resolve().parents[4] / "Saved" / "gof_observer_20260924" / "pauses"


def pause_history(folder: Path = PAUSES) -> dict:
    """{job name: [minutes, ...]} over every pause receipt's finished jobs."""
    out = defaultdict(list)
    for rp in glob.glob(str(Path(folder) / "*" / "receipt.json")):
        try:
            r = json.loads(Path(rp).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for j in r.get("jobs") or []:
            if j.get("status") == "done" and isinstance(j.get("seconds"), (int, float)):
                out[j["name"]].append(j["seconds"] / 60.0)
    return dict(out)


def _pause_items(stage: dict, history: dict, now: float, root: Path):
    """[(label, state, elapsed minutes or None, measured minutes)] for the jobs of a pause stage not yet
    done; the receipt (when the pause is running) says which are done and which one runs."""
    spec = stage.get("pause") or {}
    try:
        jobs = json.loads((root / spec["jobs"]).read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError):
        return []
    receipt = {}
    if spec.get("receipt"):
        try:
            receipt = json.loads((root / spec["receipt"]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            receipt = {}
    if receipt.get("finished"):
        return []
    done = [j for j in receipt.get("jobs") or []]
    names = {j.get("name") for j in done}
    since = max([receipt.get("stopped") or 0.0] + [(j.get("started") or 0.0) + (j.get("seconds") or 0.0) for j in done])
    out, first = [], bool(receipt)
    for job in jobs:
        if job.get("name") in names:
            continue
        elapsed = (now - since) / 60.0 if first and since else None
        out.append((job["name"], "running" if elapsed is not None else "planned", elapsed,
                    history.get(job["name"]) or []))
        first = False
    return out


def _pool(pools: dict, kind: str):
    """This build's own agents of the kind; else every agent of the kind; else this build's agents
    of every kind; else all."""
    for key in ("recent:" + kind, kind, "recent:all", "all"):
        p = pools.get(key) or []
        if len(p) >= MIN_POOL:
            return p
    return []


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def _zone(t) -> str:
    """'PDT' from Windows' 'Pacific Daylight Time' (the initials), or the zone name as given."""
    z = time.localtime(t).tm_zone or ""
    return "".join(w[0] for w in z.split()) if " " in z else z


def _clock(t, now):
    """A time the operator reads: '1:44 PM PDT', with the weekday when it is not today (operator
    2026-09-23: "DOn't use 24 hour time use am pm PDT")."""
    if t is None:
        return None
    lt, ln = time.localtime(t), time.localtime(now)
    day = "" if (lt.tm_yday, lt.tm_year) == (ln.tm_yday, ln.tm_year) else time.strftime("%A ", lt)
    return day + time.strftime("%I:%M %p", lt).lstrip("0") + " " + _zone(t)


def _live_agents(wf: dict, now: float) -> list:
    out = []
    for a in wf.get("agents", []):
        if a.get("state") not in ("running", "queued"):
            continue
        elapsed = None
        if a["state"] == "running":
            first = transcript_span(Path(a["transcript"]))[0] if a.get("transcript") else None
            if first is None:
                first = a.get("started")
            elapsed = None if first is None else max(0.0, now - first) / 60.0
        out.append({"label": a["label"], "state": a["state"], "kind": kind_of(a["label"]), "elapsed": elapsed})
    return out


def forecast(snapshot: dict, plan: dict, pools: dict, now: float = None, draws: int = DRAWS,
             seed: int = 0, agents: dict = None, pause_hist: dict = None, root: Path = None) -> dict:
    """Finish-time percentiles for every stage of the plan and for the whole plan.

    snapshot: build_status's dict (its workflows' agents with their states and transcript paths).
    agents: subagents() for plan items that name a live agent ('match', 'since'); pause_hist:
    pause_history() for stages with 'pause'; root: the folder a pause's files are named from."""
    now = time.time() if now is None else now
    root = Path(root) if root is not None else PLAN.parents[2]
    rng = random.Random(seed)
    wfs = {w["workflow"]: w for w in snapshot.get("workflows", [])}
    stages = plan.get("stages", [])
    names = [s["name"] for s in stages]
    live, flags = {}, defaultdict(list)
    lower_bound, gates = set(), []
    agent_lb, pause_lb = set(), set()     # which reason made a stage's end the earliest it can be
    pause_jobs = {}
    for s in stages:
        wid = s.get("workflow")
        if wid and wid in wfs:
            live[s["name"]] = [a for a in _live_agents(wfs[wid], now)
                               if not s.get("labels") or a["label"] in s["labels"]]
        elif wid:
            live[s["name"]] = []
            flags[s["name"]].append("its workflow has no records in the window; counted as finished")
        if s.get("pause"):
            pause_jobs[s["name"]] = _pause_items(s, pause_hist or {}, now, root)
            unmeasured = [n for n, _st, _el, hist in pause_jobs[s["name"]] if not hist]
            if unmeasured:
                lower_bound.add(s["name"])
                flags[s["name"]].append("%d of its %d jobs never ran in an earlier pause, so each counts as 0 "
                                        "(%s): this is the earliest it can end"
                                        % (len(unmeasured), len(pause_jobs[s["name"]]), ", ".join(unmeasured)))
        if s.get("operator"):
            gates.append(s["name"])
            lower_bound.add(s["name"])
            flags[s["name"]].append("waits on the operator (%s); that wait is not included" % s["operator"])
        if s.get("unmeasured"):
            # a step whose length nobody has declared or measured yet counts as 0, and says so
            lower_bound.add(s["name"])
            flags[s["name"]].append("not timed: %s; counted as 0" % s["unmeasured"])
        for p in s.get("planned", []) + s.get("agents", []):
            job = _matched_job(p, agents) if p.get("match") else None
            if job is not None and job["status"] in ("limit", "silent"):
                flags[s["name"]].append("%s: %s" % (p["label"], "stopped by the usage limit, counted as running"
                                                    if job["status"] == "limit" else
                                                    "no record for %d minutes, counted as running"
                                                    % round((now - job["end"]) / 60)))
    finishes = {n: [] for n in names}
    for _ in range(draws):
        end = {}
        for s in stages:
            start = max([now] + [end[d] for d in s.get("after", []) if d in end])
            items = []            # (label, minutes, after_label)
            for a in live.get(s["name"], []):
                pool = _pool(pools, a["kind"])
                if a["state"] == "running" and a["elapsed"] is not None:
                    longer = [d for d in pool if d > a["elapsed"]]
                    if longer:
                        items.append((a["label"], rng.choice(longer) - a["elapsed"], None, "running"))
                    else:
                        items.append((a["label"], 0.0, None, "running"))
                        lower_bound.add(s["name"])
                        agent_lb.add(s["name"])
                else:
                    items.append((a["label"], rng.choice(pool) if pool else 0.0, None, a["state"]))
            seen = {a["label"] for a in live.get(s["name"], [])}
            for p in s.get("planned", []) + s.get("agents", []):
                if p["label"] in seen:
                    continue
                pool = _pool(pools, p.get("kind") or kind_of(p["label"]))
                job = _matched_job(p, agents) if p.get("match") else None
                if job is not None and job["status"] == "finished":
                    continue
                if job is not None:
                    # a live agent started one at a time: its time left, as for a workflow's agent
                    longer = [d for d in pool if d > job["minutes"]]
                    if longer:
                        items.append((p["label"], rng.choice(longer) - job["minutes"], None, "running"))
                    else:
                        items.append((p["label"], 0.0, None, "running"))
                        lower_bound.add(s["name"])
                        agent_lb.add(s["name"])
                    continue
                items.append((p["label"], rng.choice(pool) if pool else 0.0, p.get("after_label"), "planned"))
            for name, _st, elapsed, hist in pause_jobs.get(s["name"], []):
                # a pause runs its jobs one after another
                if elapsed is not None:
                    longer = [h for h in hist if h > elapsed]
                    items.append((name, (rng.choice(longer) - elapsed) if longer else 0.0, None, "pause"))
                    if not longer and hist:
                        lower_bound.add(s["name"])
                        pause_lb.add(s["name"])
                else:
                    items.append((name, rng.choice(hist) if hist else 0.0, None, "pause"))
            if s.get("wait_minutes") is not None:
                items.append(("wait", float(s["wait_minutes"]), None, "planned"))
            if s.get("order") == "parallel":
                done_at = {}
                for label, minutes, after_label, _st in items:
                    t0 = done_at.get(after_label, start) if after_label else start
                    done_at[label] = t0 + minutes * 60.0
                end[s["name"]] = max(done_at.values()) if done_at else start
            else:
                # the usual shape of a workflow here: the agents running now run side by side, and
                # the ones waiting or not yet started follow one after another
                run = [m for _l, m, _a, st in items if st == "running"]
                rest = [m for _l, m, _a, st in items if st != "running"]
                end[s["name"]] = start + 60.0 * (max(run, default=0.0) + sum(rest))
        for n in names:
            finishes[n].append(end[n])
    out_stages = []
    for s in stages:
        f = finishes[s["name"]]
        items = live.get(s["name"], [])
        running = sum(1 for a in items if a["state"] == "running")
        queued = sum(1 for a in items if a["state"] == "queued")
        planned = 0
        for p in s.get("planned", []) + s.get("agents", []):
            if p["label"] in {a["label"] for a in items}:
                continue
            job = _matched_job(p, agents) if p.get("match") else None
            if job is None:
                planned += 1
            elif job["status"] != "finished":
                running += 1
        pending = pause_jobs.get(s["name"], [])
        running += sum(1 for _n, st, _e, _h in pending if st == "running")
        planned += sum(1 for _n, st, _e, _h in pending if st != "running")
        state = (("running" if running else "waiting")
                 if (running or queued or planned or s.get("wait_minutes") or s.get("operator")
                     or s.get("unmeasured")) else "done")
        if s["name"] in agent_lb:
            flags[s["name"]].append("an agent here has run longer than every finished agent of its kind: "
                                    "its time left counts as 0, so this is the earliest it can end")
        if s["name"] in pause_lb:
            flags[s["name"]].append("the job running now has run longer than every earlier run of it: its time "
                                    "left counts as 0, so this is the earliest it can end")
        out_stages.append({
            "name": s["name"], "what": s.get("what"), "after": s.get("after", []), "state": state,
            "running": running, "waiting": queued, "not_started": planned,
            "wait_minutes": s.get("wait_minutes"), "wait_source": s.get("source"),
            "p10": _pct(f, 0.1), "p50": _pct(f, 0.5), "p90": _pct(f, 0.9),
            "p10_clock": _clock(_pct(f, 0.1), now), "p50_clock": _clock(_pct(f, 0.5), now),
            "p90_clock": _clock(_pct(f, 0.9), now),
            "lower_bound": s["name"] in lower_bound, "notes": flags.get(s["name"], []) + s.get("notes", [])})
    last = names[-1] if names else None
    allf = [max(finishes[n][i] for n in names) for i in range(draws)] if names else []
    basis = {k: {"agents": len(v), "p10_min": round(_pct(v, 0.1), 1), "p50_min": round(_pct(v, 0.5), 1),
                 "p90_min": round(_pct(v, 0.9), 1)} for k, v in sorted(pools.items()) if v}
    return {"made": now, "draws": draws, "plan_updated": plan.get("updated"), "last_stage": last,
            "all_done": {"p10": _pct(allf, 0.1), "p50": _pct(allf, 0.5), "p90": _pct(allf, 0.9),
                         "p10_clock": _clock(_pct(allf, 0.1), now), "p50_clock": _clock(_pct(allf, 0.5), now),
                         "p90_clock": _clock(_pct(allf, 0.9), now),
                         "lower_bound": bool(lower_bound)},
            "stages": out_stages, "basis": basis, "operator_gates": gates,
            "not_included": ["pauses when the usage limit stops agents",
                             "work the plan does not list (new findings, repairs after a failed check)"]
                            + (["the operator's answers the plan waits on: "
                                + "; ".join(next(s.get("operator") for s in stages if s["name"] == g) for g in gates)]
                               if gates else [])}


def load_plan(path: Path = PLAN) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"stages": []}
