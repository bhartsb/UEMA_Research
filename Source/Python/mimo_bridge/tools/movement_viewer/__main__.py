# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""python -m mimo_bridge.tools.movement_viewer <command>

  service                    run the viewer service (service.py; the dashboard starts it)
  service --once             one loop of it, then exit
  sources                    scan the recordings and write sources.json
  index [--rec ID]           bring the index up to date (every recording, or one)
  auto                       recompute the automatic picks (auto.json)
  render ID BODY STEP        draw one command now and print its manifest's key, measures and cost
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="movement_viewer")
    ap.add_argument("command", choices=["service", "sources", "index", "auto", "render"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--rec")
    a = ap.parse_args(argv)
    from . import sources as S
    if a.command == "service":
        from . import service
        return service.main(once=a.once)
    if a.command == "sources":
        recs = S.scan()
        S.write(recs)
        print(json.dumps([dict(id=r["id"], label=r["label"], kind=r["kind"], chunks=r["chunks"]) for r in recs],
                         indent=1))
        return 0
    if a.command == "index":
        from . import index as I
        recs = S.load()["recordings"] or S.scan()
        for r in recs:
            if a.rec and r["id"] != a.rec:
                continue
            res = I.update_recording(r)
            print(r["label"], [(x["body"], x["done"], x.get("commands")) for x in res], flush=True)
        return 0
    if a.command == "auto":
        from . import auto as A
        res = A.compute(S.load()["recordings"])
        A.write(res)
        print(len(res["picks"]), "picks")
        return 0
    if a.command == "render":
        from . import render as R
        rid, body, step = a.args[0], int(a.args[1]), int(a.args[2])
        m = R.render_command(rid, body, step)
        print(json.dumps(dict(key=m["key"], measures=m["measures"], cost=m["cost"], notes=m.get("notes")), indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
