# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""What do the body workers cost by themselves, with no model attached?

    python -m mimo_bridge.tools.pool_step_cost [--workers 8] [--ticks 30]

WHY. A profile of the real training step showed the main process waiting on the
8 body workers for 151-265 ms per model step, while one body stepped in-process
costs 9.6 ms per 10 ms tick with vision off and 10.7 ms with it rendered every
second tick. Five ticks should be ~54 ms, and the bodies run in parallel on a
32-core machine, so contention is not the answer. This isolates the pool: the
same step, once through SingleEnvSource in-process and once through PoolSource,
timed with nothing else running. The difference is what the pool itself costs -
frame packing, the 393 KB per body per tick through Windows pipes, and the
gather.

Multiprocessing spawn re-imports __main__, so this must be a real module.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run(src, ticks, label):
    frames = src.reset()
    ne = len(frames)
    nh = len(frames[0].bones)
    m = np.zeros((ne, nh, 9), np.float32)
    m[..., 6] = 1.0
    for _ in range(3):
        src.step(m)
    t = time.perf_counter()
    for i in range(ticks):
        src.step(m, render=(i % 2 == 0) and not getattr(src, "_no_vision", False))
    dt = (time.perf_counter() - t) / ticks
    print(f"  {label:26s} {1000*dt:7.1f} ms per tick of 10 ms body time, {ne} bodies"
          f"  -> x5 ticks = {5000*dt:6.0f} ms per model step")
    return dt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ticks", type=int, default=30)
    ap.add_argument("--no-vision", action="store_true",
                    help="no eye rendering: isolates the GPU the workers share from the physics")
    args = ap.parse_args()
    from mimo_bridge.datasource import PoolSource, SingleEnvSource
    kw = dict(scene="crib", vision=not args.no_vision, seed=3, age=24,
              contact_profile="mechanical_v4", caregiver_seconds=0)

    t0 = time.perf_counter()
    s1 = SingleEnvSource(**kw)
    print(f"single env built in {time.perf_counter()-t0:.1f} s")
    d1 = run(s1, args.ticks, "1 body, in-process")
    s1.close()

    t0 = time.perf_counter()
    pool = PoolSource(args.workers, kw)
    print(f"{args.workers}-worker pool built in {time.perf_counter()-t0:.1f} s")
    d8 = run(pool, args.ticks, f"{args.workers} bodies, {args.workers} workers")
    pool.close()

    print()
    print(f"  pool cost beyond one in-process body: {1000*(d8-d1):.1f} ms per tick, "
          f"{5000*(d8-d1):.0f} ms per model step")
    print(f"  if the workers were free, {args.workers} bodies would cost what 1 does: "
          f"{5000*d1:.0f} ms per model step. They cost {5000*d8:.0f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
