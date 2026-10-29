# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Wait for a training stage to genuinely finish - and to have finished the RIGHT work.

    python -m mimo_bridge.tools.wait_for_stage <run_dir> [--from-ckpt ckpt_step_N]
                                               [--timeout-min 360] [--min-steps N]

Why this exists (2026-08-23). Chain scripts used "does summary.json exist?" as the test for
"did that stage finish?". Two failures in one night came from that:

  * H13 leg 2 died silently at 00:04; the chain marched on and ran the END battery against
    the HOUR-4 brain, then wrote results into directories named for hour 8;
  * the follow-on chain waited on one of those stale files and would have launched its
    successor against the wrong checkpoint.

A filename cannot say which brain produced a result. stage_done.json can: train_mimo writes
it at the end of every run with the checkpoint it resumed FROM, the checkpoint it produced,
the step count and the outcome. This waiter blocks until that marker exists, is fresh, says
ok, and (optionally) confirms the stage resumed from the checkpoint the chain intended.

Exit codes: 0 the stage finished the expected work | 2 finished but from the WRONG
checkpoint (the failure mode above) | 3 finished with ok=false | 4 timed out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

MARKER = "stage_done.json"


def read_marker(run_dir: str):
    try:
        with open(os.path.join(run_dir, MARKER), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--from-ckpt", default=None,
                    help="basename of the checkpoint this stage MUST have resumed from")
    ap.add_argument("--min-steps", type=int, default=0,
                    help="reject a marker whose run did fewer steps than this")
    ap.add_argument("--timeout-min", type=float, default=360.0)
    ap.add_argument("--poll-s", type=float, default=30.0)
    ap.add_argument("--since", type=float, default=None,
                    help="only accept a marker written after this unix time (default: now)")
    a = ap.parse_args(argv)
    since = a.since if a.since is not None else time.time()
    deadline = time.time() + a.timeout_min * 60

    while time.time() < deadline:
        m = read_marker(a.run_dir)
        if m and m.get("finished_at", 0) >= since - 1:
            if not m.get("ok"):
                print(f"stage finished but FAILED: {m.get('exception', '')[:200]}")
                return 3
            if a.from_ckpt and os.path.basename(m.get("resumed_from") or "") != a.from_ckpt:
                print(f"stage finished but resumed from {m.get('resumed_from')!r}, "
                      f"expected {a.from_ckpt!r} - refusing to treat this as the right stage")
                return 2
            if m.get("steps", 0) < a.min_steps:
                print(f"stage finished with only {m.get('steps')} steps (< {a.min_steps})")
                return 3
            print(f"stage OK: {m.get('steps')} steps, produced "
                  f"{os.path.basename(m.get('final_checkpoint') or '?')}, "
                  f"resumed from {os.path.basename(m.get('resumed_from') or 'scratch')}")
            return 0
        time.sleep(a.poll_s)
    print(f"TIMED OUT after {a.timeout_min:.0f} min waiting for {MARKER} in {a.run_dir}")
    return 4


if __name__ == "__main__":
    sys.exit(main())
