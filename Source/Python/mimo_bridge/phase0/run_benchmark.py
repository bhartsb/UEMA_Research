# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Phase 0: run MIMo's own benchmark.py headless on this laptop.
Writes a tab-separated results file (MIMo's format) and a JSON summary."""
import os, sys, json, copy, time, platform
os.environ.setdefault("MUJOCO_GL", "glfw")
import mimo_infant.simulation  # registers envs
from mimo_infant.simulation.benchmark import benchmark, load_benchmark_file
import gymnasium as gym
# MIMo's benchmark.py calls env.dt on the gym.make() result; with gymnasium 1.2.3 that
# result is a TimeLimit wrapper without .dt. We unwrap here, in OUR process only, so MIMo
# itself stays unchanged. (Guess written down: unwrapping only removes the 6000-step
# TimeLimit; the benchmark loop resets on done anyway.)
_orig_make = gym.make
gym.make = lambda *a, **k: _orig_make(*a, **k).unwrapped
from mimo_infant.simulation.envs.mimo_env import DEFAULT_TOUCH_PARAMS_V2, DEFAULT_VISION_PARAMS

SIM_SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
out_dir = os.path.dirname(os.path.abspath(__file__))
res_file = os.path.join(out_dir, "benchmark_results.txt")

vision_256 = copy.deepcopy(DEFAULT_VISION_PARAMS)
touch_v2 = copy.deepcopy(DEFAULT_TOUCH_PARAMS_V2)
configs = [
    ("V2 default (touch v2 + vision 256 + vestibular)", "MIMoBenchV2-v0", {}, SIM_SECONDS),
    ("V2 no vision", "MIMoBenchV2-v0", {"vision_params": None}, SIM_SECONDS),
    ("V2 no touch", "MIMoBenchV2-v0", {"touch_params": None}, SIM_SECONDS),
    ("V2 physics only", "MIMoBenchV2-v0", {"touch_params": None, "vision_params": None, "vestibular_params": None}, SIM_SECONDS),
]
t0 = time.time()
benchmark(configs, res_file)
data = load_benchmark_file(res_file)
summary = {"machine": platform.node(), "python": sys.version.split()[0], "sim_seconds_per_config": SIM_SECONDS,
           "wall_total_s": time.time() - t0, "configs": {}}
for name, d in data.items():
    steps = d["n_steps"]; rt = d["Runtime"] - d["Init."]
    summary["configs"][name] = {
        "env_steps": int(steps),
        "sim_time_s": d["Simtime"],
        "init_s": round(d["Init."], 2),
        "run_s_excl_init": round(rt, 2),
        "env_steps_per_s": round(steps / rt, 1) if rt > 0 else None,
        "realtime_factor": round(d["Simtime"] / rt, 3) if rt > 0 else None,
        "split_s": {k: round(d[k], 2) for k in ("Physics", "Muscle", "Touch", "Vision", "Proprioception", "Vestibular", "Other")},
        "split_pct_of_run": {k: round(100 * d[k] / rt, 1) for k in ("Physics", "Touch", "Vision", "Proprioception", "Vestibular", "Other")} if rt > 0 else {},
    }
json.dump(summary, open(os.path.join(out_dir, "benchmark_summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
