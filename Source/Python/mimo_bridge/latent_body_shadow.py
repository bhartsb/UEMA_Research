# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Default-OFF trainer hook. This module imports no model or numerical library."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import weakref

_RUNTIMES = weakref.WeakKeyDictionary()
log = logging.getLogger("mimo_bridge.latent_body")


def add_arguments(parser):
    parser.add_argument("--latent-body-hierarchy", choices=("off", "shadow", "control"), default="off")
    parser.add_argument("--latent-body-levels", default="arm,hand,index",
                        help="arm,hand,index[,thumb,middle,ring,pinky]: the arm-and-hand pilot (lem.latent_body.tree."
                             "arm_hand_tree); full_body: every handle of the layout (tree.full_body_tree), which needs "
                             "the whole-body row convention (body_rows) in --latent-body-config")
    parser.add_argument("--latent-planner", choices=("off", "shadow", "clone", "control"), default="off")
    parser.add_argument("--latent-body-config", default="", help="Measured hierarchy configuration JSON")
    parser.add_argument("--latent-body-output", default="", help="Separate shadow records/checkpoints directory")
    parser.add_argument("--latent-body-resume", default="",
                        help="Explicit independent hierarchy checkpoint, or 'auto': the output directory's latest.pt "
                             "when it exists, else a fresh start (a supervisor relaunch keeps the same command)")
    parser.add_argument("--latent-body-log-mb", type=float, default=0.0,
                        help="Largest size of each shadow record file before it is rotated to <name>.1 (the "
                             "previous .1 is replaced), so each record keeps at most twice this; 0 = no cap")
    parser.add_argument("--latent-body-save-every", type=int, default=1,
                        help="Write the shadow's latest.pt once this many updates have accumulated (1 = after every "
                             "observation, the earlier behaviour)")
    parser.add_argument("--latent-body-goal", default="", help="Local physical change goals for shadow search")
    parser.add_argument("--latent-body-candidates", type=int, default=None)
    parser.add_argument("--latent-body-radius", type=float, default=None)


def validate_arguments(args):
    mode, planner = args.latent_body_hierarchy, args.latent_planner
    if mode == "control" or planner == "control":
        raise ValueError("Hierarchy control authority is unavailable before cloned-world qualification")
    if planner == "clone":
        raise ValueError("Clone execution belongs in mimo_bridge.tools.latent_body, never in the trainer")
    if mode == "off":
        if planner != "off":
            raise ValueError("The latent planner requires the shadow hierarchy")
        return
    if not args.latent_body_config or not args.latent_body_output:
        raise ValueError("Shadow learning requires measured configuration and a separate output directory")
    output, production = Path(args.latent_body_output).resolve(), Path(args.out).resolve()
    if "saved" in {part.lower() for part in output.parts}:
        raise ValueError("Keep the independent hierarchy experiment outside Saved")
    if output == production or production in output.parents or output in production.parents:
        raise ValueError("Shadow output must be separate from production output")
    if planner == "shadow" and (not args.latent_body_goal or args.latent_body_candidates is None
                                or args.latent_body_radius is None):
        raise ValueError("Shadow search requires declared goals, candidate budget and measured latent radius")
    if planner == "shadow" and (args.latent_body_candidates < 2 or not math.isfinite(args.latent_body_radius)
                                or args.latent_body_radius <= 0):
        raise ValueError("Shadow search requires >=2 candidates and a finite positive radius")
    from lem.latent_body.learning import validate_config
    config = json.loads(Path(args.latent_body_config).read_text(encoding="utf-8"))
    validate_config(config)
    if args.latent_body_levels == "full_body" and "body_rows" not in config:
        raise ValueError("--latent-body-levels full_body needs the whole-body row convention (body_rows) in its config")


def observe_shadow(args, model, frames, transitions, step, episode_id):
    """Return nothing: no route from hierarchy output to production commands.

    One call site owns creation, physical recording, lessons and independent
    checkpointing. OFF returns before imports, allocations, RNG draws or I/O.
    A shadow failure is recorded and disables only this experiment.
    """
    if getattr(args, "latent_body_hierarchy", "off") == "off":
        return
    runtime = _RUNTIMES.get(model)
    if runtime is False:
        return
    try:
        if runtime is None:
            validate_arguments(args)
            from mimo_bridge.latent_body_runtime import ShadowRuntime
            runtime = ShadowRuntime(args, model.handle_names)
            _RUNTIMES[model] = runtime
        runtime.observe(frames, transitions, step, episode_id, train=not args.eval_only,
                        synergy=getattr(model, "synergy", None))
    except Exception:
        log.exception("Latent body shadow failed; its output remains unused and this experiment is disabled")
        _RUNTIMES[model] = False
        if runtime is not None:
            try:
                runtime.fail()
            except Exception:
                log.exception("Hierarchy failure receipt could not be written")
