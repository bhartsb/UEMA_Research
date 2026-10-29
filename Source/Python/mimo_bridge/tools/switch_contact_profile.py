# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Carry a saved run onto a different contact profile, at an explicit boundary.

    python -m mimo_bridge.tools.switch_contact_profile --checkpoint <ckpt dir>
        [--target mechanical_v4] [--out <dir>] [--apply]

Without --apply this only reports: it loads the companion, says what the switch
would change, and writes nothing.

WHY A BOUNDARY IS NEEDED. A contact profile decides WHERE the pressure fed to the
wellness formula comes from. mechanical_v2 and mechanical_v2_local divide the
contact force by the ACTIVE SENSOR AREA, which env.py itself labels 'not resolved
tissue footprint'. mechanical_v3 and mechanical_v4 divide it by the contact
footprint resolved from the touching geometry, and they write a different history
contract. The single frame sitting at the switch still carries the old contract,
so lem_model refuses it unless the episode is annotated to say this exact switch
happened at this exact sim_time. That annotation is what this tool writes.

WHAT THE RUN WILL FEEL AFTERWARDS, measured 2026-09-16 in the crib at age 24
months (Saved/mimo_bridge/scale_anchors.json):

                         lying still          a 30 cm fall     a 90 cm fall
  mechanical_v2_local    comfort 0.7081       pain 0.9991      pain 0.9164
  mechanical_v4          comfort 0.5461       pain 0.7025      pain 0.9134

Lying still scores 0.1620 lower for doing nothing different, because the resolved
footprint is smaller than the sensor support area and the same rest reads 73,844
Pa instead of 17,024 Pa. In exchange the fall response stops running backwards:
under v4 pain rises at every rung of the ladder - 0.1466, 0.1624, 0.7025, 0.8711,
0.8808, 0.9134 across 5, 10, 30, 45, 66 and 90 cm - with no two rungs tied.

WHAT THIS TOOL DOES NOT DO. It does not requalify a decorated shadow actor. If the
run carries one and any of its bound module sources changed, the resume still
refuses with 'decorated acting execution changed'. The explicit operator action
for that is --retire-acting-snapshot on the resume, which drops the saved frozen
actor and qualifies a fresh candidate under the current code; this tool names it
in its output rather than doing it silently.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="checkpoint directory holding episode_state.json")
    ap.add_argument("--target", default="mechanical_v4")
    ap.add_argument("--out", default="",
                    help="where the switched checkpoint is written; "
                         "default: <checkpoint>_<target>")
    ap.add_argument("--apply", action="store_true",
                    help="actually write; without it this only reports")
    args = ap.parse_args()

    from mimo_bridge import continuation_state
    from mimo_bridge.virtual_skin_transition import (
        activate_local_skin, TRANSFERABLE_SOURCES, LOCAL_SKIN_TARGETS)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_dir():
        print(f"no such checkpoint directory: {ckpt}")
        return 2
    if args.target not in LOCAL_SKIN_TARGETS:
        print(f"target must be one of {', '.join(LOCAL_SKIN_TARGETS)}")
        return 2

    state = continuation_state.load(str(ckpt))
    if state is None:
        print(f"{ckpt} has no episode companion - nothing to carry across. A run "
              f"started fresh on --contact-profile {args.target} needs no boundary.")
        return 2

    source = state["config"].get("contact_profile")
    step = state["trainer"]["step"]
    print(f"checkpoint {ckpt}")
    print(f"  step                 {step:,}")
    print(f"  contact profile now  {source}")
    print(f"  switching to         {args.target}")
    print(f"  worlds in episode    {len(state['worlds'])}")
    print()

    if source == args.target:
        print(f"already on {args.target}; nothing to do")
        return 0
    if source not in TRANSFERABLE_SOURCES:
        print(f"cannot carry an episode off {source}: only "
              f"{' and '.join(TRANSFERABLE_SOURCES)} write the history contract that "
              f"the boundary frame needs. {source} already writes the resolved-geometry "
              f"contract, so a run on it can be relaunched on the other resolved-geometry "
              f"profile directly, with no boundary and no rewrite.")
        return 2

    actor = state["trainer"].get("acting_snapshot")
    if actor:
        print("  NOTE this run carries a decorated shadow actor. Its manifest hashes the")
        print("  SOURCE of the modules it acts through, and this branch has changed several")
        print("  of them, so the resume will refuse with 'decorated acting execution changed'")
        print("  no matter what this tool writes. There is no IMPLICIT path -")
        print("  prepare_decorated_shadow.py refuses an already-decorated shadow by design -")
        print("  but there is an explicit one: resume with --retire-acting-snapshot, which")
        print("  drops the saved frozen actor and qualifies a fresh candidate under the")
        print("  current code. The retired actor keeps its recorded decisions in the old")
        print("  run stats. That flag exists for exactly this: changed bound code on a")
        print("  shadow lineage.")
        print()

    out = Path(args.out) if args.out else ckpt.with_name(ckpt.name + "_" + args.target)
    print("WHAT CHANGES")
    print(f"  every world's env_kwargs contact_profile: {source} -> {args.target}")
    print(f"  every world gets a contact_profile_transition marker at its own sim_time")
    print(f"  the saved observations, pressures, q and prediction history are UNTOUCHED")
    print(f"  the learner state - weights, optimizer, replay - is copied byte for byte")
    print()
    print("WHAT THE BODY WILL FEEL (measured, crib, age 24 months)")
    print("                        lying still      30 cm fall     90 cm fall")
    print("  mechanical_v2_local   comfort 0.7081   pain 0.9991    pain 0.9164")
    print("  mechanical_v4         comfort 0.5461   pain 0.7025    pain 0.9134")
    print("  Resting scores 0.1620 lower for doing nothing different. The fall")
    print("  ladder stops running backwards and rises at every rung.")
    print()

    if not args.apply:
        print(f"REPORT ONLY - nothing written. Re-run with --apply to write to {out}")
        return 0

    if out.exists():
        print(f"refusing to overwrite an existing directory: {out}")
        return 2

    upgraded = activate_local_skin(state, target=args.target)
    shutil.copytree(ckpt, out, ignore=shutil.ignore_patterns(
        "episode_state.pt.gz", "episode_state.json", "episode_state.json.tmp"))
    receipt = continuation_state.write_episode(out, upgraded)

    note = dict(version="contact_profile_switch_v1", source=source, target=args.target,
                step=int(step), from_checkpoint=str(ckpt.resolve()),
                measured="Saved/mimo_bridge/scale_anchors.json",
                resting_comfort_before=0.7081, resting_comfort_after=0.5461,
                why="the v2_local fall response runs backwards above 30 cm; v4 "
                    "resolves contact area from geometry and keeps the aversion "
                    "curve resolving at the pressures falls produce",
                decorated_actor_present=bool(actor))
    (out / "contact_profile_switch.json").write_text(
        json.dumps(note, indent=2), encoding="utf-8")

    print(f"written to {out}")
    print(f"  episode receipt step {receipt['step']:,}, {receipt['bytes']:,} bytes")
    resume = f"--resume {out} --contact-profile {args.target}"
    if actor:
        resume += " --retire-acting-snapshot"
    print(f"  resume with {resume}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
