# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Acceptance check for --self-percept-release: the self-percept on every step since the resume.

    python -m mimo_bridge.tools.self_percept_watch <run_dir> [--json]

Reads the run's replay_training_stats.jsonl (records of type "mimo") and judges the declared lines of
docs/gain_of_function_v4/agents/self_percept_runaway_2026-09-26.md ("Next executable test") from the
self_state block's window_* fields, which count every step of each report (self_state_integration.new_window).
Exit code 0 = pass or still waiting for readings, 3 = a stop line was crossed (snapshot_and_stop, then resume
with the switch off), 2 = the records carry no per-step counts (the run is not on this code).

Lines, over the steps since the resume:
  sticking again (stop line)  live steps above the limit of 8. With the switch on every such step is a release;
                              2 or more within any 1,000 steps is a body that keeps getting stuck again (the
                              review's planted body, stuck again every 144 steps, gives 7). Replays of H84's
                              recorded steps with the switch on gave 0 in 1,024 steps at all four stuck
                              checkpoints and on the 415,075 weights with other memories (attention_cap_sweep).
  recomputed calls (stop line) perceive() calls above the limit per report, the live one and the recomputed
                              ones; above 25 in a report of 100 steps stops the run (the first round's line).
                              Expected 0: the release clears the saved inputs and held rehearsal windows.
  restore (reading)           bodies released before the first step because their restored stored percepts
                              sat at the limit (4 on the memory saved at 415,075).
  recovery (reading)          median of each report's largest live percept size over the last 10 reports:
                              healthy top 2.0; the resume weights frozen, after the restore, settle at 2.52;
                              above 4.3 (the first round's post-release reading) calls for attribution work.
  attention (reading)         share of steps on which the 5% cap spread a body's recall; it falls as the
                              weights relearn to spread their own attention.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

LIMIT = 8.0
STICK_WINDOW = 1000
STICK_STOP = 2
BOUNDED_STOP = 25
RECOVERY_GOAL = 2.0
RECOVERY_FROZEN = 2.52
RECOVERY_INVESTIGATE = 4.3


def check(records):
    """records: [{'step': int, 'self_state': {...}}] from one segment, in step order."""
    rows = [(int(r['step']), r.get('self_state') or {}) for r in records]
    rows = [(s, b) for s, b in rows if 'window_steps' in b]
    if not rows:
        return dict(verdict='no_counts', lines={}, reason='no record carries window_* per-step counts')
    start = rows[0][0] - int(rows[0][1]['window_steps'])
    over = [(s, int(b['window_live_over_limit'])) for s, b in rows]
    worst = 0
    for i, (s, _) in enumerate(over):
        inside = sum(n for t, n in over if s - STICK_WINDOW < t <= s)
        worst = max(worst, inside)
    bounded = max(int(b['window_bounded_calls']) * 100 / max(1, int(b['window_steps'])) for _, b in rows)
    sizes = [b['window_belief_size_max'] for _, b in rows[-10:] if b.get('window_belief_size_max') is not None]
    med = sorted(sizes)[len(sizes) // 2] if sizes else None
    capped = sum(int(b.get('window_attention_capped_steps', 0)) for _, b in rows)
    steps = sum(int(b['window_steps']) for _, b in rows)
    lines = dict(
        sticking_again=dict(value=worst, stop_at=STICK_STOP, window_steps=STICK_WINDOW,
                            total=sum(n for _, n in over), crossed=worst >= STICK_STOP),
        recomputed_calls=dict(value_per_100=round(bounded, 2), stop_above=BOUNDED_STOP, crossed=bounded > BOUNDED_STOP),
        restore=dict(bodies=int(rows[0][1].get('window_restored_bodies', 0))),
        recovery=dict(median_largest_size_last_10=None if med is None else round(med, 3), goal=RECOVERY_GOAL,
                      frozen_resume_weights=RECOVERY_FROZEN, investigate_above=RECOVERY_INVESTIGATE,
                      investigate=med is not None and med > RECOVERY_INVESTIGATE),
        attention=dict(capped_share=round(capped / max(1, steps), 4)),
        steps_since_resume=rows[-1][0] - start, first_step=start)
    crossed = lines['sticking_again']['crossed'] or lines['recomputed_calls']['crossed']
    verdict = 'stop' if crossed else ('pass' if lines['steps_since_resume'] >= STICK_WINDOW else 'waiting')
    return dict(verdict=verdict, lines=lines)


def read_run(run_dir):
    records = []
    path = Path(run_dir) / 'replay_training_stats.jsonl'
    mimo = re.compile(r'"type": "mimo"')
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if not mimo.search(line) or 'window_steps' not in line:
                continue
            r = json.loads(line)
            records.append(dict(step=r['step'], self_state=r.get('self_state')))
    return records


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    result = check(read_run(a.run_dir))
    print(json.dumps(result, indent=None if a.json else 1))
    return {'stop': 3, 'no_counts': 2}.get(result['verdict'], 0)


if __name__ == '__main__':
    sys.exit(main())
