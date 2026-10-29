"""Run the self-percept acceptance check every 5 minutes beside H84 training, and stop the run on a declared stop line.

    python self_percept_watch_loop.py                 (ends when Saved/gof_takeover_20260927/self_percept_watch/stop exists)
    python self_percept_watch_loop.py --once --no-act [--judge-anyway]     one pass, nothing written into the run

The check is mimo_bridge.tools.self_percept_watch: the declared lines of
docs/gain_of_function_v4/agents/self_percept_runaway_2026-09-26.md ("Next executable test"), judged from the per-step
counts in the stats block self_state over the steps since the run's resume. Each pass judges the supervisor's active
run, and only when that run was launched with --self-percept-release on (its args.json): with the switch off the
lines are not declared and nothing is judged.

On verdict "stop" (2 or more live model steps above the limit of 8 within any 1,000 model steps, or more than 25
perceive calls above the limit in a report of 100 model steps) this program does what the hourly watcher's stop does:
it pauses the supervisor when the judged run is the supervisor's active run, writes stop_reason.json (repair
required) and snapshot_and_stop into the run's control.json, once per run. The revert is the coordinator's: resume
the stopped checkpoint with --self-percept-release off, as the record declares. A pass that raises is logged and the
next pass runs: the watcher must never cost training.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SAVED = ROOT / "Saved" / "mimo_bridge"
OUT = ROOT / "Saved/gof_takeover_20260927/self_percept_watch"
EVERY_S = 300
sys.path.insert(0, str(ROOT / "Source" / "Python"))


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def stop_run(run, why):
    """The hourly watcher's stop (docs/h84_launch_2026-09-18/hourly_watch.py, stop_run), for `run`."""
    try:
        state = read_json(SAVED / "training_continuation.json") or {}
        if Path(state.get("active_run", "")).resolve() == Path(run).resolve():
            path = SAVED / "supervisor_control.json"
            control = read_json(path) or {}
            control.update(cmd="pause", reason=why, timestamp=time.time())
            tmp = path.with_suffix(".selfpercept.tmp")
            tmp.write_text(json.dumps(control), encoding="utf-8")
            tmp.replace(path)
    except (OSError, ValueError, TypeError):
        pass  # stop_reason.json below still forbids automatic relaunch
    with open(os.path.join(run, "stop_reason.json"), "w", encoding="utf-8") as f:
        json.dump({"reason": why, "resume_policy": "repair_required", "timestamp": time.time()}, f)
    with open(os.path.join(run, "control.json"), "w") as f:
        json.dump({"cmd": "snapshot_and_stop"}, f)


def one_pass(act=True, judge_anyway=False):
    from mimo_bridge.tools import self_percept_watch as W
    state = read_json(SAVED / "training_continuation.json") or {}
    run = state.get("active_run")
    line = dict(at=time.strftime("%b %d, %Y, %I:%M:%S %p PDT"), run=Path(run).name if run else None)
    if not run or not Path(run).is_dir():
        return dict(line, state="no active run")
    switch = (read_json(Path(run) / "args.json") or {}).get("self_percept_release")
    line["switch"] = switch
    if switch != "on" and not judge_anyway:
        return dict(line, state="switch off: not judged")
    result = W.check(W.read_run(run))
    lines = result.get("lines") or {}
    line.update(state=result["verdict"],
                live_steps_over_limit_worst_1000=(lines.get("sticking_again") or {}).get("value"),
                live_steps_over_limit_total=(lines.get("sticking_again") or {}).get("total"),
                bounded_calls_per_100_worst=(lines.get("recomputed_calls") or {}).get("value_per_100"),
                restored_bodies=(lines.get("restore") or {}).get("bodies"),
                recovery_median_largest_size=(lines.get("recovery") or {}).get("median_largest_size_last_10"),
                capped_share=(lines.get("attention") or {}).get("capped_share"),
                steps_since_resume=lines.get("steps_since_resume"))
    if result["verdict"] == "stop":
        already = (Path(run) / "stop_reason.json").exists()
        line["stop_written"] = False
        if act and switch == "on" and not already:
            s, c = lines["sticking_again"], lines["recomputed_calls"]
            why = ("Self-percept stop line with --self-percept-release on: %d live model steps above the limit of 8 "
                   "within 1,000 model steps (stop at 2) and at most %.1f perceive calls above the limit per 100 "
                   "model steps in a report (stop above 25). Save, then resume with --self-percept-release off "
                   "(docs/gain_of_function_v4/agents/self_percept_runaway_2026-09-26.md)."
                   % (s["value"], c["value_per_100"]))
            stop_run(run, why)
            line.update(stop_written=True, why=why)
        elif already:
            line["note"] = "stop_reason.json already present"
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-act", action="store_true")
    ap.add_argument("--judge-anyway", action="store_true", help="judge a run whose switch is off (never acts on it)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            line = one_pass(act=not a.no_act, judge_anyway=a.judge_anyway)
        except Exception as exc:                      # the watcher must never cost training
            line = dict(at=time.strftime("%b %d, %Y, %I:%M:%S %p PDT"), error=repr(exc)[:500])
        if a.once:
            print(json.dumps(line, indent=1))
            return
        with (OUT / "loop.log").open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        if (OUT / "stop").exists():
            return
        time.sleep(EVERY_S)


if __name__ == "__main__":
    main()
