# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Continue training from the saved state in weights/ on this machine.

    python resume.py --out FOLDER [--steps N] [--print]

Reads weights/b1_continue_B1K_skip13_492800/run/command.json, the exact command that produced the saved state at
model step 494,025, and changes only what belongs to the machine it ran on:
  - the Python interpreter: the one running this script (use the environment the README's Install section builds);
  - the saved state to continue from: this repository's weights/ folder (--resume, --latent-body-resume);
  - where the run writes: FOLDER (--out FOLDER/run, --checkpoint-storage FOLDER/run/checkpoints,
    --latent-body-output FOLDER/latent_body, --caregiver-record-dir FOLDER/caregiver_records);
  - nothing about the input files (calibrations, latent-body settings, caregiver speech): the command keeps naming
    them inside the development repository, as the saved state does (the trainer compares those names with the saved
    ones), and mimo_bridge/repo_paths.py reads each from this repository's copy at the same relative path; this
    script checks that every one of them is here;
  - the step to stop at: --max-steps 494,025 + N (default N = 100 model steps; one model step = 50 ms of each of the
    eight bodies' simulated time).
Every other setting stays as it was. It runs the trainer from Source/Python and returns its exit code. --print shows
the command without running it.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
BUNDLE = REPO / "weights" / "b1_continue_B1K_skip13_492800"
CHECKPOINT = BUNDLE / "run" / "checkpoints" / "ckpt_step_494025"
START_STEP = 494025
DEV_ROOT = r"C:\Users\bhart\Documents\UEMA_devA_private"   # the development repository the command names


def command(out: Path, steps: int) -> list:
    saved = json.loads((BUNDLE / "run" / "command.json").read_text(encoding="utf-8"))
    if saved[1:4] != ["-u", "-m", "mimo_bridge.train_mimo"]:
        raise SystemExit("unexpected saved command: %s" % saved[:4])
    out_of = {
        "--resume": CHECKPOINT,
        "--latent-body-resume": BUNDLE / "latent_body" / "latest.pt",
        "--out": out / "run",
        "--checkpoint-storage": out / "run" / "checkpoints",
        "--latent-body-output": out / "latent_body",
        "--caregiver-record-dir": out / "caregiver_records",
    }
    args = [sys.executable] + saved[1:]
    problems = []
    for i in range(4, len(args)):
        flag = args[i - 1]
        if flag in out_of:
            args[i] = str(out_of[flag])
        elif flag == "--max-steps":
            args[i] = str(START_STEP + steps)
        elif os.path.normcase(args[i]).startswith(os.path.normcase(DEV_ROOT) + os.sep):
            local = REPO / Path(args[i][len(DEV_ROOT) + 1:].replace("\\", "/"))
            if not local.exists():
                problems.append("%s %s: not in this repository (%s)" % (flag, args[i], local))
        elif re.match(r"^[A-Za-z]:[\\/]", args[i]):
            problems.append("%s %s: a path of the development machine this script does not map" % (flag, args[i]))
    if "--max-steps" not in args:
        args += ["--max-steps", str(START_STEP + steps)]
    if problems:
        raise SystemExit("cannot build the command:\n  " + "\n  ".join(problems))
    return args


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="a new folder for this run's records, checkpoints and latent-body state")
    ap.add_argument("--steps", type=int, default=100, help="model steps to train after 494,025 (default 100)")
    ap.add_argument("--print", action="store_true", help="show the command and stop")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if a.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if not CHECKPOINT.joinpath("model.safetensors").is_file() or CHECKPOINT.joinpath("model.safetensors").stat().st_size < 1_000_000:
        raise SystemExit("the saved weights are missing or are Git LFS pointer files: run 'git lfs pull' in %s" % REPO)
    args = command(out, a.steps)
    if a.print:
        print(subprocess.list2cmdline(args))
        return 0
    if out.exists() and any(out.iterdir()):
        raise SystemExit("%s is not empty: give a new folder" % out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resume_command.json").write_text(json.dumps(args, indent=1), encoding="utf-8")
    return subprocess.call(args, cwd=REPO / "Source" / "Python")


if __name__ == "__main__":
    sys.exit(main())
