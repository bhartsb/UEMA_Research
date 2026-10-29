# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Phase 3 runner: launches the experiment runs (E1..E5) as subprocesses and analyzes them.

    python -m mimo_bridge.experiments.run_all --stage e5_throughput
    python -m mimo_bridge.experiments.run_all --stage A        # E1 + E3 in parallel
    python -m mimo_bridge.experiments.run_all --stage B        # E2, 10 runs, 3 at a time
    python -m mimo_bridge.experiments.run_all --stage C        # E4, 3 runs in parallel
    python -m mimo_bridge.experiments.run_all --stage D --minutes 90   # E5 stability
    python -m mimo_bridge.experiments.run_all --stage analyze

All outputs live under Saved/mimo_bridge/experiments/<run name>/. results.json + PNGs go to
Saved/mimo_bridge/experiments/results/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(SRC, "..", ".."))
PY = os.path.join(REPO, ".venv_mimo", "Scripts", "python.exe")
EXP = os.path.join(REPO, "Saved", "mimo_bridge", "experiments")
RESULTS = os.path.join(EXP, "results")


def run_spec(name: str, args: list) -> dict:
    return {"name": name, "out": os.path.join(EXP, name), "args": args}


def launch(spec: dict) -> subprocess.Popen:
    os.makedirs(spec["out"], exist_ok=True)
    cmd = [PY, "-m", "mimo_bridge.train_mimo", "--out", spec["out"]] + [str(a) for a in spec["args"]]
    log = open(os.path.join(spec["out"], "stdout.log"), "a")
    log.write("CMD: " + " ".join(cmd) + "\n"); log.flush()
    print(f"[launch] {spec['name']}: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, cwd=SRC, stdout=log, stderr=subprocess.STDOUT)


def run_pool(specs: list, max_parallel: int):
    pending = list(specs)
    running = []
    results = {}
    while pending or running:
        while pending and len(running) < max_parallel:
            s = pending.pop(0)
            running.append((s, launch(s), time.time()))
        time.sleep(5)
        still = []
        for s, p, t0 in running:
            rc = p.poll()
            if rc is None:
                still.append((s, p, t0))
            else:
                results[s["name"]] = {"rc": rc, "minutes": (time.time() - t0) / 60}
                print(f"[done] {s['name']} rc={rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
        running = still
    return results


def stage_e5_throughput(minutes: float):
    specs = [run_spec(f"E5_throughput_n{n}", ["--n-envs", n, "--minutes", minutes, "--curriculum", "off",
                                                "--report-every", 50, "--checkpoint-every-min", 1e9, "--episode-steps", 0])
             for n in (1, 4, 16)]
    return run_pool(specs, 1)   # one at a time: clean numbers


def stage_a(e1_minutes: float, e3_minutes: float):
    specs = [
        run_spec("E1_reach_1env", ["--n-envs", 1, "--minutes", e1_minutes, "--scene", "reach", "--curriculum", "off",
                                   "--report-every", 100, "--checkpoint-every-min", 60]),
        run_spec("E3_selfbody_touch", ["--n-envs", 4, "--minutes", e3_minutes, "--scene", "selfbody", "--no-vision",
                                       "--curriculum", "off", "--probe-touch", "--probe-every", 10, "--report-every", 100,
                                       "--episode-steps", 1500, "--checkpoint-every-min", 60]),
        # E1b: same as E1 but 4 bodies (batch 4 -> VICReg alive), to show the batch-1 collapse is a batch effect
        run_spec("E1b_reach_4env", ["--n-envs", 4, "--minutes", min(30.0, e1_minutes), "--scene", "reach", "--curriculum", "off",
                                    "--report-every", 100, "--checkpoint-every-min", 60]),
    ]
    return run_pool(specs, 3)


def stage_b(minutes: float, seeds=(0, 1, 2, 3, 4), steps_per_phase: int = 700, parallel: int = 3):
    specs = []
    for s in seeds:
        specs.append(run_spec(f"E2_curr_on_seed{s}", ["--n-envs", 4, "--minutes", minutes, "--seed", s, "--curriculum", "on",
                                                       "--steps-per-phase", steps_per_phase, "--age-binding", "curriculum",
                                                       "--delay-binding", "on", "--report-every", 50, "--checkpoint-every-min", 1e9]))
        specs.append(run_spec(f"E2_curr_off_seed{s}", ["--n-envs", 4, "--minutes", minutes, "--seed", s, "--curriculum", "off",
                                                        "--age-binding", "fixed:18", "--delay-binding", "off",
                                                        "--report-every", 50, "--checkpoint-every-min", 1e9]))
    return run_pool(specs, parallel)


def stage_c(minutes: float):
    # E4 needs a body that can actually move: the model's static table keeps the first 8000 steps
    # at neonate limits (strength 0.02), so E4 runs in the fixed "child" phase (rom 1.0,
    # strength 0.9, all handles, age 18 months) via the curriculum manager with a huge phase length.
    common = ["--n-envs", 4, "--minutes", minutes, "--scene", "reach", "--curriculum", "on", "--start-phase", "child",
              "--steps-per-phase", 100000000, "--report-every", 50, "--checkpoint-every-min", 1e9, "--wellness-mode", "forward"]
    specs = [
        run_spec("E4_model_imag_on", common + ["--imagination", "on"]),
        run_spec("E4_model_imag_off", common + ["--imagination", "off"]),
        run_spec("E4_random_actions", common + ["--imagination", "on", "--random-actions"]),
        # E3b: self-touch body map again, but with a body that can move (child phase) and vision off
        run_spec("E3b_selfbody_touch_child", ["--n-envs", 4, "--minutes", minutes + 5, "--scene", "selfbody", "--no-vision",
                                              "--curriculum", "on", "--start-phase", "child", "--steps-per-phase", 100000000,
                                              "--probe-touch", "--probe-every", 10, "--report-every", 100,
                                              "--episode-steps", 1500, "--checkpoint-every-min", 1e9]),
    ]
    return run_pool(specs, 4)


def stage_d(minutes: float, n_envs: int = 8):
    specs = [run_spec("E5_stability", ["--n-envs", n_envs, "--minutes", minutes, "--curriculum", "on", "--steps-per-phase", 3000,
                                       "--report-every", 100, "--checkpoint-every-min", 60])]
    return run_pool(specs, 1)


def stage_analyze():
    from mimo_bridge.experiments import analyze as A
    os.makedirs(RESULTS, exist_ok=True)
    res = {}
    if os.path.isdir(os.path.join(EXP, "E1_reach_1env")):
        res["E1"] = A.analyze_e1(os.path.join(EXP, "E1_reach_1env"), RESULTS)
    e2 = {"curriculum_on": {}, "curriculum_off": {}}
    for d in os.listdir(EXP) if os.path.isdir(EXP) else []:
        if d.startswith("E2_curr_on_seed"):
            e2["curriculum_on"][int(d.rsplit("seed", 1)[1])] = os.path.join(EXP, d)
        if d.startswith("E2_curr_off_seed"):
            e2["curriculum_off"][int(d.rsplit("seed", 1)[1])] = os.path.join(EXP, d)
    if e2["curriculum_on"] or e2["curriculum_off"]:
        res["E2"] = A.analyze_e2(e2, RESULTS)
    if os.path.isdir(os.path.join(EXP, "E3_selfbody_touch")):
        res["E3"] = A.analyze_e3(os.path.join(EXP, "E3_selfbody_touch"), RESULTS)
    if os.path.isdir(os.path.join(EXP, "E3b_selfbody_touch_child")):
        res["E3b"] = A.analyze_e3(os.path.join(EXP, "E3b_selfbody_touch_child"), RESULTS)
    e4 = {k: os.path.join(EXP, v) for k, v in (("model_imag_on", "E4_model_imag_on"), ("model_imag_off", "E4_model_imag_off"),
                                               ("random_actions", "E4_random_actions")) if os.path.isdir(os.path.join(EXP, v))}
    if e4:
        res["E4"] = A.analyze_e4(e4, RESULTS)
    e5 = {n: os.path.join(EXP, f"E5_throughput_n{n}") for n in (1, 4, 16) if os.path.isdir(os.path.join(EXP, f"E5_throughput_n{n}"))}
    stab = os.path.join(EXP, "E5_stability") if os.path.isdir(os.path.join(EXP, "E5_stability")) else None
    if e5 or stab:
        res["E5"] = A.analyze_e5(e5, stab, RESULTS)
    # ---- re-runs after the fixes (2026-08-19 morning) ----
    if os.path.isdir(os.path.join(EXP, "E1_reach_1env_v2")):
        res["E1_v2"] = A.analyze_e1(os.path.join(EXP, "E1_reach_1env_v2"), RESULTS)
        if res["E1_v2"].get("figure"):
            os.replace(res["E1_v2"]["figure"], os.path.join(RESULTS, "E1_v2_pe.png")); res["E1_v2"]["figure"] = os.path.join(RESULTS, "E1_v2_pe.png")
    if os.path.isdir(os.path.join(EXP, "E3c_selfbody_touch_posemb")):
        res["E3c"] = A.analyze_e3(os.path.join(EXP, "E3c_selfbody_touch_posemb"), RESULTS)
        if res["E3c"].get("figure"):
            os.replace(res["E3c"]["figure"], os.path.join(RESULTS, "E3c_touch_pca.png")); res["E3c"]["figure"] = os.path.join(RESULTS, "E3c_touch_pca.png")
    e4b = {k: os.path.join(EXP, v) for k, v in (("model_imag_on", "E4_model_imag_on_v3"), ("model_imag_off", "E4_model_imag_off_v3"),
                                                ("random_actions", "E4_random_actions_v3")) if os.path.isdir(os.path.join(EXP, v))}
    if e4b:
        res["E4_v3"] = A.analyze_e4(e4b, RESULTS)
        if res["E4_v3"].get("figure"):
            os.replace(res["E4_v3"]["figure"], os.path.join(RESULTS, "E4_v3_motor_loss.png")); res["E4_v3"]["figure"] = os.path.join(RESULTS, "E4_v3_motor_loss.png")
    longrun = os.path.join(EXP, "E5_longrun_8h")
    if os.path.isdir(longrun):
        res["E5_longrun"] = A.analyze_e5({}, longrun, RESULTS)
        res["E5_longrun"].pop("figure", None)
    try1 = os.path.join(EXP, "E5_longrun_8h_try1")
    if os.path.isdir(try1):
        res["E5_longrun_try1"] = A.analyze_e5({}, try1, RESULTS)
        res["E5_longrun_try1"].pop("figure", None)
    json.dump(res, open(os.path.join(RESULTS, "results.json"), "w"), indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["e5_throughput", "A", "B", "C", "D", "analyze"])
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--e3-minutes", type=float, default=25)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--parallel", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(EXP, exist_ok=True)
    if a.stage == "e5_throughput":
        r = stage_e5_throughput(a.minutes or 2.5)
    elif a.stage == "A":
        r = stage_a(a.minutes or 60, a.e3_minutes)
    elif a.stage == "B":
        r = stage_b(a.minutes or 12, tuple(int(s) for s in a.seeds.split(",")), parallel=a.parallel)
    elif a.stage == "C":
        r = stage_c(a.minutes or 15)
    elif a.stage == "D":
        r = stage_d(a.minutes or 90)
    else:
        r = stage_analyze()
    json.dump(r, open(os.path.join(EXP, f"stage_{a.stage}_result.json"), "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
