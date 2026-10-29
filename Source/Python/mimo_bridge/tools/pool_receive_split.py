# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Where does the main process spend its per-body receive time: pipe or decode?

    python -m mimo_bridge.tools.pool_receive_split [--workers 24] [--ticks 30]

WHY. The pool costs about 8.5 ms per added body per model step. A threaded
receive was tried on the assumption that the cost was WAITING on pipes one after
another; it did not help. This splits the main process's per-body cost into the
pipe read (conn.poll + conn.recv, which unpickles) and the frame decode
(_payload_to_frame), timed per call with nothing else running, for the serial and
the threaded receive. If the decode dominates and it holds the GIL, threads cannot
help and the cost has to move out of the main process instead.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--ticks", type=int, default=30)
    args = ap.parse_args()
    from mimo_bridge import datasource as DS

    recv_t, dec_t = [], []
    real_recv, real_dec = DS.PoolSource._recv, DS._payload_to_frame

    def timed_recv(self, i):
        t = time.perf_counter(); r = real_recv(self, i); recv_t.append(time.perf_counter() - t); return r

    def timed_dec(p, i):
        t = time.perf_counter(); r = real_dec(p, i); dec_t.append(time.perf_counter() - t); return r

    DS.PoolSource._recv = timed_recv
    DS._payload_to_frame = timed_dec

    kw = dict(scene="crib", vision=True, seed=3, age=24, contact_profile="mechanical_v4",
              caregiver_seconds=0)
    for mode in ("serial", "threaded"):
        if mode == "serial":
            os.environ["MIMO_POOL_SERIAL_RECV"] = "1"
        else:
            os.environ.pop("MIMO_POOL_SERIAL_RECV", None)
        pool = DS.PoolSource(args.workers, kw)
        frames = pool.reset()
        nh = len(frames[0].bones)
        m = np.zeros((args.workers, nh, 9), np.float32); m[..., 6] = 1.0
        for _ in range(3):
            pool.step(m)
        recv_t.clear(); dec_t.clear()
        t0 = time.perf_counter()
        for i in range(args.ticks):
            pool.step(m, render=(i % 2 == 0))
        wall = (time.perf_counter() - t0) / args.ticks
        pool.close()
        n = args.workers * args.ticks
        print(f"{mode:9s} {args.workers} workers: {1000*wall:6.1f} ms per tick wall")
        print(f"           pipe read+unpickle  median {1000*statistics.median(recv_t):6.2f} ms per body, "
              f"sum per tick {1000*sum(recv_t)/args.ticks:6.1f} ms  ({len(recv_t)} calls)")
        print(f"           frame decode        median {1000*statistics.median(dec_t):6.2f} ms per body, "
              f"sum per tick {1000*sum(dec_t)/args.ticks:6.1f} ms  ({len(dec_t)} calls)")
        print(f"           (sums exceed wall under threads when calls overlap; under serial they add up)")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
