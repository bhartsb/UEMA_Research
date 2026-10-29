# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The build forecast (tools/build_forecast.py) turns finished agents' measured run times into finish
times for the remaining stages, and only measured times: operator 2026-09-23, "You need to write a
program to give me time estimates for these fucking agents remaining"."""
import json
import os
import time
from datetime import datetime, timezone

from mimo_bridge.tools import build_forecast as BF

NOW = 1_790_150_000.0


def test_jobs_are_grouped_by_kind_from_their_label():
    assert [BF.kind_of(x) for x in ("b6-loss-balance", "lane-b3-wellness", "fix-probes", "verify-fix-ops",
                                    "verify-r1-3", "review-b5", "find-literal-r1", "survey:x", "rows-check")] == \
        ["build", "build", "fix", "verify", "refute", "review", "find", "survey", "other"]


def _ts(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def test_run_times_come_from_the_transcripts_own_timestamps(tmp_path):
    wf = tmp_path / "proj" / "sess" / "subagents" / "workflows" / "wf_x"
    wf.mkdir(parents=True)
    (wf / "journal.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"type": "launched"},
        {"type": "started", "agentId": "a", "label": "fix-one"},
        {"type": "result", "agentId": "a"},
        {"type": "started", "agentId": "b", "label": "fix-two"},
        {"type": "failed", "agentId": "b"}]) + "\n", encoding="utf-8")
    (wf / "agent-a.jsonl").write_text(json.dumps({"timestamp": _ts(NOW - 3000)}) + "\n" +
                                      json.dumps({"timestamp": _ts(NOW - 600)}) + "\n", encoding="utf-8")
    (wf / "agent-b.jsonl").write_text(json.dumps({"timestamp": _ts(NOW - 100)}) + "\n", encoding="utf-8")
    pools = BF.measured_durations([tmp_path / "proj"], now=NOW)
    assert pools["fix"] == [40.0] and pools["recent:fix"] == [40.0]    # the failed agent gives no time


def _snap(*agents):
    return {"workflows": [{"workflow": "wf_a", "agents": list(agents)}]}


def test_stages_add_up_in_order_and_a_running_agent_uses_only_longer_finished_ones():
    pools = {"recent:build": [60.0, 60.0, 60.0], "recent:verify": [30.0, 30.0, 30.0],
             "recent:fix": [10.0, 10.0, 10.0]}
    snap = _snap({"label": "lane-x", "agent_id": "1", "state": "running", "started": NOW - 20 * 60},
                 {"label": "verify-lane-x", "agent_id": "2", "state": "queued"})
    plan = {"stages": [{"name": "lane", "workflow": "wf_a", "order": "sequential"},
                       {"name": "merge", "after": ["lane"], "agents": [{"label": "merge-lanes", "kind": "fix"}]},
                       {"name": "checkpoint", "after": ["merge"], "wait_minutes": 60}]}
    f = BF.forecast(snap, plan, pools, now=NOW, draws=50)
    ends = {s["name"]: (s["p50"] - NOW) / 60 for s in f["stages"]}
    # 40 min left of a 60-min build, then a 30-min re-check; then a 10-min merge; then 60 min
    assert abs(ends["lane"] - 70) < 1e-6 and abs(ends["merge"] - 80) < 1e-6 and abs(ends["checkpoint"] - 140) < 1e-6
    assert abs((f["all_done"]["p50"] - NOW) / 60 - 140) < 1e-6 and not f["all_done"]["lower_bound"]


def test_an_agent_past_every_measured_run_is_a_lower_bound_not_a_guess():
    pools = {"recent:build": [30.0, 40.0, 50.0]}
    snap = _snap({"label": "lane-x", "agent_id": "1", "state": "running", "started": NOW - 90 * 60})
    f = BF.forecast(snap, {"stages": [{"name": "lane", "workflow": "wf_a"}]}, pools, now=NOW, draws=20)
    s = f["stages"][0]
    assert s["lower_bound"] and abs(s["p50"] - NOW) < 1e-6 and f["all_done"]["lower_bound"]
    assert any("longer than every finished agent" in n for n in s["notes"])


def test_parallel_stages_wait_only_for_the_job_they_follow():
    pools = {"recent:fix": [30.0] * 3, "recent:verify": [20.0] * 3, "recent:review": [50.0] * 3}
    plan = {"stages": [{"name": "p", "order": "parallel",
                        "agents": [{"label": "fix-a", "kind": "fix"},
                                   {"label": "verify-fix-a", "kind": "verify", "after_label": "fix-a"},
                                   {"label": "review-b", "kind": "review"}]}]}
    f = BF.forecast({"workflows": []}, plan, pools, now=NOW, draws=10)
    assert abs((f["stages"][0]["p50"] - NOW) / 60 - 50) < 1e-6          # max(30 + 20, 50)


# Agents started one at a time (the Agent tool, the build's way since 2026-09-25) and training-pause
# jobs: the operator asked for a finish time on 2026-09-25 at 2:0x PM PDT, and the forecast read only
# workflow agents, none of which were running.
def _line(t, kind, text="", stop=None, tools=False, error=False):
    if kind == "user":
        rec = {"type": "user", "message": {"role": "user", "content": text}}
    else:
        content = [{"type": "tool_use", "name": "Bash"}] if tools else [{"type": "text", "text": text}]
        rec = {"type": "assistant", "message": {"role": "assistant", "content": content, "stop_reason": stop}}
        if error:
            rec["isApiErrorMessage"] = True
    rec["timestamp"] = _ts(t)
    return json.dumps(rec)


def _transcript(folder, aid, description, lines):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ("agent-%s.jsonl" % aid)).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / ("agent-%s.meta.json" % aid)).write_text(json.dumps({"description": description}), encoding="utf-8")
    return folder / ("agent-%s.jsonl" % aid)


def test_an_agent_started_alone_splits_into_jobs_at_finished_turns_and_joins_across_a_limit_stop(tmp_path):
    t0 = NOW - 7200
    path = _transcript(tmp_path / "proj" / "sess" / "subagents", "a1", "Review the repair", [
        _line(t0, "user", "Review the repair."),
        _line(t0 + 600, "assistant", tools=True),
        _line(t0 + 1800, "assistant", "Verdict: two findings.", stop="end_turn"),
        _line(t0 + 3600, "user", "The coordinator sent a message: fix them."),
        _line(t0 + 3900, "assistant", tools=True),
        _line(t0 + 4200, "assistant", "You've hit your session limit · resets 1:50pm", error=True),
        _line(t0 + 4800, "user", "The coordinator sent a message: the limit reset, continue."),
        _line(t0 + 6000, "assistant", tools=True)])
    jobs = BF.subagent_jobs(path, now=t0 + 6060)
    assert [j["status"] for j in jobs] == ["finished", "running"]
    assert abs(jobs[0]["minutes"] - 30) < 1e-6
    assert abs(jobs[1]["minutes"] - 30) < 1e-6, "10 minutes before the limit stop and 20 after; the gap is not work"
    pools = BF.measured_durations([tmp_path / "proj"], now=t0 + 6060)
    assert pools["review"] == [30.0] and "fix" not in pools, "only finished jobs are measured; a first job by its description"


def test_an_agent_that_hands_back_its_report_through_a_tool_is_finished(tmp_path):
    # Oct 5, 2026: the dashboard data agent ended with a SubagentHandback tool call; the forecast read it
    # as running for 161.7 minutes, because no text-only end_turn record follows such a call.
    t0 = NOW - 7200
    hand_back = json.dumps({"type": "user", "toolEndsTurn": True, "timestamp": _ts(t0 + 2400), "message": {
        "role": "user", "content": [{"type": "tool_result", "content": "Report delivered."}]}})
    path = _transcript(tmp_path / "proj" / "sess" / "subagents", "a2", "Build the data layer", [
        _line(t0, "user", "Build the data layer."),
        _line(t0 + 600, "assistant", tools=True),
        _line(t0 + 2390, "assistant", tools=True),
        hand_back])
    jobs = BF.subagent_jobs(path, now=t0 + 2460)
    assert [j["status"] for j in jobs] == ["finished"]
    assert abs(jobs[0]["minutes"] - 40) < 1e-6


def test_a_plan_item_naming_a_live_agent_uses_its_time_left_and_a_finished_one_is_done():
    pools = {"recent:fix": [60.0, 60.0, 60.0], "recent:review": [45.0] * 3}
    agents = {"Fix the watcher": [{"start": NOW - 5 * 3600, "end": NOW - 4 * 3600, "minutes": 55.0, "status": "finished"},
                                  {"start": NOW - 1200, "end": NOW - 30, "minutes": 20.0, "status": "running"}],
              "Review the repair": [{"start": NOW - 3000, "end": NOW - 600, "minutes": 40.0, "status": "finished"}]}
    plan = {"stages": [
        {"name": "fixes", "agents": [{"label": "fix-watcher", "kind": "fix", "match": "Fix the watcher", "since": NOW - 1500}]},
        {"name": "review", "agents": [{"label": "review-repair", "kind": "review", "match": "Review the repair",
                                        "since": NOW - 3100}]},
        {"name": "re-review", "after": ["fixes"], "agents": [{"label": "review-fixes", "kind": "review",
                                                               "match": "Review the fixes", "since": NOW}]}]}
    f = BF.forecast({"workflows": []}, plan, pools, now=NOW, draws=20, agents=agents)
    ends = {s["name"]: (s["p50"] - NOW) / 60 for s in f["stages"]}
    assert abs(ends["fixes"] - 40) < 1e-6, "60-minute fixes, 20 minutes run: the job begun after 'since', not the older one"
    assert abs(ends["review"]) < 1e-6 and f["stages"][1]["state"] == "done"
    assert abs(ends["re-review"] - 85) < 1e-6, "not started: a whole review after the fixes"


def test_pause_jobs_take_their_measured_times_and_a_never_measured_job_marks_the_earliest_end(tmp_path):
    (tmp_path / "jobs.json").write_text(json.dumps([{"name": "a"}, {"name": "b"}, {"name": "c"}]), encoding="utf-8")
    (tmp_path / "receipt.json").write_text(json.dumps({"started": NOW - 700, "stopped": NOW - 600, "jobs": [
        {"name": "a", "status": "done", "started": NOW - 600, "seconds": 120}]}), encoding="utf-8")
    plan = {"stages": [{"name": "pause", "pause": {"jobs": "jobs.json", "receipt": "receipt.json"}}]}
    f = BF.forecast({"workflows": []}, plan, {}, now=NOW, draws=40, pause_hist={"b": [20.0, 20.0, 20.0]},
                    root=tmp_path)
    s = f["stages"][0]
    assert abs((s["p50"] - NOW) / 60 - 12) < 1e-6, "b has run 8 of its measured 20 minutes; c never ran: 0"
    assert s["lower_bound"] and any("never ran in an earlier pause" in n and "c" in n for n in s["notes"])
    assert s["running"] == 1 and s["not_started"] == 1


def test_a_later_job_is_a_fix_only_when_it_acts_on_a_review():
    assert BF.job_kind("Review the repair", 0, "") == "review"
    assert BF.job_kind("Build the skin fix", 0, "") == "build"
    assert BF.job_kind("Build the skin fix", 1, "The independent review found three defects: fix them") == "fix"
    assert BF.job_kind("Build the skin fix", 2, "What was the friction you measured?") == "follow-up"


def test_a_stage_nobody_has_timed_counts_as_zero_and_says_so():
    f = BF.forecast({"workflows": []}, {"stages": [{"name": "vision", "unmeasured": "depends on the pause"}]}, {},
                    now=NOW, draws=5)
    s = f["stages"][0]
    assert s["state"] == "waiting" and s["lower_bound"] and any("depends on the pause" in n for n in s["notes"])


def test_a_stage_waiting_on_the_operator_is_named_and_not_timed():
    plan = {"stages": [{"name": "skin answers", "operator": "the skin questions"},
                       {"name": "deploy", "after": ["skin answers"], "wait_minutes": 60, "source": "measured"}]}
    f = BF.forecast({"workflows": []}, plan, {}, now=NOW, draws=5)
    assert f["operator_gates"] == ["skin answers"] and f["all_done"]["lower_bound"]
    assert abs((f["all_done"]["p50"] - NOW) / 60 - 60) < 1e-6
    assert any("the skin questions" in x for x in f["not_included"])


def test_the_plan_file_names_only_stages_that_exist():
    plan = BF.load_plan()
    names = [s["name"] for s in plan["stages"]]
    assert names and len(names) == len(set(names))
    for s in plan["stages"]:
        assert all(d in names[:names.index(s["name"])] for d in s.get("after", [])), s["name"]
        if s.get("wait_minutes") is not None:
            assert s.get("source"), s["name"]
