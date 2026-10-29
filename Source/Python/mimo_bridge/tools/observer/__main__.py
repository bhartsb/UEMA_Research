# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""python -m mimo_bridge.tools.observer <command>

  watch     [--run DIR] [--interval S] [--no-replays]   digest and report continuously (the
            active run from training_continuation.json unless --run is given)
  digest    --run DIR [--final]                         one digestion pass and the lineage report
  report    --run DIR                                   the lineage report only
  baseline  --like DIR --kind random_orders|random_ctrl|limp|yoked|yoked_sequence [--bodies N] [--steps N]
            [--workers N]
            [--vision like|on|off]      the eyes' images as in the run (default) or forced
            [--from-checkpoint CKPT]    start body i from production body i's saved world in CKPT
                                        (--like optional); becomes the comparison the tracker reads,
                                        the previous one of another checkpoint kept as <kind>_ckpt_<step>
            [--yoke-from RUN [RUN ...]] yoked, yoked_sequence: the run folders whose recorded orders body i
                                        replays, shuffled (yoked) or in their recorded order from a random
                                        first order (yoked_sequence) (default: the checkpoint's run folder)
            [--resummarize SIGNATURE]   re-digest a kept recording with the current detectors
                                        (--kind may name one kept aside: random_orders_reset_world)
            [--compare SIGNATURE]       each capability's rate in <kind>.json against
                                        <kind>_reset_world.json, or --other NAME (no world)
  validate  --like DIR [--write-golden]                 the scripted validation worlds, judged by the
                                                        LOCKED pass/fail lines (validation_lock);
                                                        --write-golden PROPOSES, the operator approves
  validate  --replay [--write-golden]                   re-digest the kept case recordings and compare
                                                        with the locked golden detections (no simulation)
  rebuild-history --lineage NAME --at "YYYY-mm-dd HH:MM" [...]
                                                        a past Development report from the observer's own
                                                        records under today's rules (rebuild.py), written as
                                                        a "_rebuilt" dated copy to the history folder; each
                                                        time moves back to the latest moment no run was
                                                        recording. No model, no world.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mimo_bridge.tools.observer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch")
    w.add_argument("--run")
    w.add_argument("--interval", type=float, default=60.0)
    w.add_argument("--no-replays", action="store_true")
    w.add_argument("--once", action="store_true")
    d = sub.add_parser("digest")
    d.add_argument("--run", required=True)
    d.add_argument("--final", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--run", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--like", help="a training run folder whose args.json sets the world (not with --resummarize)")
    b.add_argument("--kind", default="random_orders",
                   help="random_orders, random_ctrl, limp, yoked or yoked_sequence to record; with "
                        "--resummarize or --compare also a kept baseline's name (random_orders_reset_world)")
    b.add_argument("--bodies", type=int, default=8)
    b.add_argument("--steps", type=int, default=5000)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--resummarize", metavar="SIGNATURE")
    b.add_argument("--vision", choices=["like", "on", "off"], default="like",
                   help="the eyes' images: 'like' the run's own setting (default; the eyes' motion "
                        "reflex then runs as in training), or forced on/off")
    b.add_argument("--from-checkpoint", metavar="CKPT",
                   help="a production checkpoint folder (<run>/checkpoints/ckpt_step_N): body i starts "
                        "from production body i's saved world, restored as train_mimo restores it on "
                        "resume, with the saved worlds' own vision; the recording becomes the comparison "
                        "the tracker reads and a reset-world one moves aside to <kind>_reset_world")
    b.add_argument("--compare", metavar="SIGNATURE",
                   help="print each capability's chance rate in <kind>.json against "
                        "<kind>_reset_world.json of that world (or --other), with the two-sided test of the counts")
    b.add_argument("--other", metavar="NAME",
                   help="with --compare: the kept baseline to compare <kind>.json with (random_orders_ckpt_397125)")
    b.add_argument("--yoke-from", nargs="+", metavar="RUN",
                   help="with --kind yoked or yoked_sequence: the production run folders whose recorded orders "
                        "are replayed (default: the checkpoint's own run folder)")
    v = sub.add_parser("validate")
    v.add_argument("--like")
    v.add_argument("--replay", action="store_true")
    v.add_argument("--scratch")
    v.add_argument("--write-golden", action="store_true")
    h = sub.add_parser("rebuild-history")
    h.add_argument("--lineage", required=True)
    h.add_argument("--at", nargs="+", required=True, help='local times, "2026-09-29 12:00"')
    h.add_argument("--root", help="history folder (default: history.history_root())")
    h.add_argument("--saved", help="the Saved/mimo_bridge folder to read (default: this checkout's)")
    a = ap.parse_args(argv)
    if a.cmd == "rebuild-history":
        import time
        from .daemon import SAVED as DEFAULT_SAVED
        from . import rebuild as RB
        SAVED = a.saved or DEFAULT_SAVED
        for when in a.at:
            t = RB.best_moment(SAVED, a.lineage, time.mktime(time.strptime(when, "%Y-%m-%d %H:%M")))
            if t is None:
                print("%s: no moment without a recording run before it" % when)
                continue
            print(when, "->", RB.write_rebuilt(SAVED, a.lineage, t, root=a.root))
        return 0
    if a.cmd == "watch":
        from .daemon import serve
        serve(a.run, a.interval, a.once, render=not a.no_replays)
        return 0
    if a.cmd in ("digest", "report"):
        from .daemon import SAVED, RunDigest, active_run, build_report, lineage_runs, report_path_for
        from mimo_bridge.tools.training_continuation import OwnerLock
        try:                                   # never beside the service: two writers double-count
            lock = OwnerLock(os.path.join(SAVED, "observer", "owner.lock"))
        except OSError:
            print("the observer service holds Saved/mimo_bridge/observer/owner.lock; stop it first", flush=True)
            return 3
        try:
            runs = lineage_runs(a.run)
            live = active_run()
            digests = {}
            if a.cmd == "digest":
                for run in runs:
                    dg = digests[run] = RunDigest(run)
                    is_live = live is not None and os.path.abspath(run) == os.path.abspath(live)
                    # the training's active run is never finalized: its future is still coming
                    final = (a.final or os.path.abspath(run) != os.path.abspath(a.run)) and not is_live
                    print(json.dumps(dg.digest(final=final)))
            out = report_path_for(a.run)
            rep = build_report(runs, out, digests, current_run=a.run)
            print(out, rep["counts"])
        finally:
            lock.close()
        return 0
    if a.cmd == "baseline":
        from .baseline import KINDS, YOKED_KINDS, compare, resummarize, run
        if a.compare:
            print(json.dumps(compare(a.compare, a.kind, a.other), indent=1))
        elif a.resummarize:
            print(resummarize(a.resummarize, a.kind))
        elif a.kind not in KINDS:
            ap.error("--kind must be one of %s to record; a kept baseline's name only with --resummarize "
                     "or --compare" % ", ".join(KINDS))
        elif not a.like and not a.from_checkpoint:
            ap.error("baseline needs --like DIR or --from-checkpoint CKPT (or --resummarize SIGNATURE)")
        elif a.kind in YOKED_KINDS and not a.from_checkpoint:
            ap.error("--kind %s replays production's orders in its saved worlds: it needs --from-checkpoint"
                     % a.kind)
        else:
            print(run(a.like, a.kind, a.bodies, a.steps, a.workers, vision=a.vision,
                      from_checkpoint=a.from_checkpoint,
                      **({"yoke_from": a.yoke_from} if a.yoke_from else {})))
        return 0
    if a.cmd == "validate":
        from .validate import replay_check, run
        from .catalog import ROOT
        out = os.path.join(str(ROOT), "docs", "gain_of_function_v4", "observer_validation")
        scratch = a.scratch or os.path.join(str(ROOT), "Saved", "mimo_bridge", "observer", "validation_scratch")
        if a.replay:
            rep = replay_check(scratch, None, out, write=a.write_golden)
            print(json.dumps(dict(ok=rep["ok"], lock=rep["lock"], same=rep["same"], cases=rep["cases"],
                                  differences=rep["differences"], expectations=rep["expectations"],
                                  golden_proposed=rep.get("golden_proposed")), indent=1))
            return 0 if rep["ok"] and not rep["expectations"]["failed"] else 1
        if not a.like:
            ap.error("validate needs --like DIR (or --replay)")
        rep = run(a.like, scratch, out, None, a.write_golden)
        print(json.dumps(dict(ok=rep["ok"], lock=rep["lock"], cases=[(c["name"], c["ok"], c.get("approved"))
                                                                   for c in rep["cases"]],
                              not_approved=rep.get("not_approved"), missing_locked_cases=rep.get("missing_locked_cases"),
                              checks={k: v.get("ok") for k, v in rep["checks"].items()},
                              replay=rep.get("replay"), proposal=rep.get("proposal")), indent=1))
        return 0 if rep["ok"] else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
