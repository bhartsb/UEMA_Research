# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Measure the lived reference points of token-distance space, so every distance the
operator reads has both ends of its scale (rule in the style guide).

Anchors, in the same mean-over-senses MSE units the planner's goal distance uses:
  one_instant    the world's change across ONE model step (50 ms) of ordinary
                 experience (an ORDER is 5 model steps = 250 ms; the imagination
                 chain's depth unit is the model step, same as this anchor)
  half_minute    the distance between moments ~30+ orders apart in the same world -
                 the "unrelated moments" far end (same crib, different situation)
  goal_arrival   the planner's arrived line (0.01, fixed in code)

Writes Saved/mimo_bridge/scale_anchors.json; the dashboard reads it to translate
distances into "instants of world-change" and "% of a full-scale gap".
python -m mimo_bridge.tools.scale_anchors --ckpt <ckpt_dir> [--device cpu]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mimo_bridge.tools.causal_test import _build_model, _limits_for_run, _collect  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "Saved", "mimo_bridge",
                   "scale_anchors.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=70)
    ap.add_argument("--device", default="cpu")   # CPU: runs beside a live trainer
    args = ap.parse_args(argv)
    t0 = time.time()
    model, a, _s, _f = _build_model(args.ckpt, args.device, 0, intention=False)
    if a.get("audio") == "real":
        model.config["modalities"].setdefault("audio", {})["data_source"] = "real"
    limits, _ = _limits_for_run(args.ckpt, model)
    limits["_action_repeat"] = int(a.get("action_repeat") or 5)
    from mimo_bridge.datasource import SingleEnvSource
    import tempfile
    env_kw = dict(scene="crib", run_dir=tempfile.mkdtemp(prefix="anch_"), vision=True,
                  fovea_size=int(a.get("fovea_size") or 288),
                  fovea_fov=float(a.get("fovea_fov") or 24.0))
    if a.get("audio") is not None:
        env_kw["audio_real"] = a.get("audio") == "real"
    for k, on in (("critters", "critters"), ("mobile", "mobile")):
        if a.get(on) is not None:
            env_kw[k] = a.get(on) == "on"
    src = SingleEnvSource(tag="_anch", seed=170, **env_kw)
    try:
        trs = _collect(model, src, args.n, int(a.get("motor_hold") or 5),
                       limits=dict(limits), babble=float(a.get("babble_noise") or 0.05))
    finally:
        src.close()
    mods = [m for m in ("vision", "tactile", "audio", "proprioception")
            if m in trs[0]["t"]]

    def dist(ta, tb):
        return float(np.mean([F.mse_loss(ta[m].float(), tb[m].float()).item()
                              for m in mods]))

    one = [dist(tr["t"], tr["t1"]) for tr in trs]
    far = [dist(trs[i]["t"], trs[j]["t"])
           for i, j in itertools.combinations(range(len(trs)), 2) if j - i >= 30]
    anchors = {"one_instant": float(np.median(one)),
               "half_minute_apart": float(np.median(far)),
               "goal_arrival": 0.01,
               "mods": mods, "n_transitions": len(trs),
               "ckpt": os.path.abspath(args.ckpt),
               "measured": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(anchors, open(OUT, "w"), indent=1)
    print(f"one instant of world-change: {anchors['one_instant']:.4f}")
    print(f"moments half a minute apart: {anchors['half_minute_apart']:.4f}")
    print(f"planner's arrived line     : {anchors['goal_arrival']:.4f}")
    print(f"written {OUT}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
