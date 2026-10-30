# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Run once after cloning: restore every file's original modification time, and check the weights arrived.

    python setup_checkout.py

Git does not keep modification times: a fresh clone stamps every file with the time of the checkout. Some checks of
the dashboard read those times, for example whether a test's result file was written during its own run (the
investigation boxes on the Tests page decide only on a result written during the run) and which run folder holds the
newest records. checkout_times.json holds each tracked file's modification time in the release (nanoseconds since
1970); this script writes them back. It also reports whether the saved weights are here: they come from the private
Hugging Face repository through `python fetch_weights.py`, not with the clone.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def main() -> int:
    record = json.loads((REPO / "checkout_times.json").read_text(encoding="utf-8"))
    restored = missing = 0
    for rel, ns in record["files"].items():
        p = REPO / rel
        try:
            st = p.stat()
        except OSError:
            missing += 1
            continue
        os.utime(p, ns=(st.st_atime_ns, int(ns)))
        restored += 1
    print("modification times restored: %d files; %d listed files are not in this checkout" % (restored, missing))
    sys.path.insert(0, str(REPO))
    from fetch_weights import BUNDLE, FILES
    absent = [f for f in FILES if not (REPO / BUNDLE / f).is_file() or (REPO / BUNDLE / f).stat().st_size < 1024]
    if absent:
        print("the saved weights are not here yet (%d of %d files): run `python fetch_weights.py --accept-license`"
              % (len(absent), len(FILES)))
        return 1
    print("saved weights: %d files present (fetch_weights.py checks their SHA-256)" % len(FILES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
