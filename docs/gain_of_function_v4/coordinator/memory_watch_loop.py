"""Run the memory repair's watcher (persistent_memory_deploy_watch.py --act) every 5 minutes beside H84 training.

    python memory_watch_loop.py            (stops when Saved/gof_takeover_20260927/memory_watch/stop exists)

The watcher itself judges the active run's ancestry and writes snapshot_and_stop only on a declared STOP line; this
loop only schedules it and keeps one line per pass in memory_watch/loop.log. A pass that raises is logged and the
next pass runs: the watcher must never cost training.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "Saved/gof_takeover_20260927/memory_watch"
WATCH = ROOT / "docs/gain_of_function_v4/persistent_memory_deploy_watch.py"
EVERY_S = 300


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(__import__("os").environ, PYTHONPATH=str(ROOT / "Source/Python"))
    while not (OUT / "stop").exists():
        started = time.strftime("%b %d, %Y, %I:%M:%S %p PDT")
        try:
            r = subprocess.run([sys.executable, "-B", str(WATCH), "--act", "--out", str(OUT / "verdict.json")],
                               cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
            v = json.loads((OUT / "verdict.json").read_text(encoding="utf-8")) if (OUT / "verdict.json").exists() else {}
            line = dict(at=started, exit=r.returncode, state=v.get("state"), stops=v.get("stops"),
                        alerts=len(v.get("alerts") or []), reports=(v.get("readings") or {}).get("reports"))
            if r.returncode:
                line["stderr"] = r.stderr[-500:]
        except Exception as exc:
            line = dict(at=started, error=repr(exc)[:500])
        with (OUT / "loop.log").open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        time.sleep(EVERY_S)


if __name__ == "__main__":
    main()
