# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Continue training from the saved state in weights/ on this machine.

    python resume.py --out FOLDER [--steps N] [--device NAME] [--print]

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
    eight bodies' simulated time);
  - the device the model runs on: --device NAME replaces the saved "--device cuda:0" (default NAME = auto: the trainer
    takes cuda:0 when CUDA is available, else mps, the Metal device, else cpu; a named device that is not
    available stops the trainer with a message naming it; mimo_bridge/device_resolve.py).
Every other setting stays as it was. It runs the trainer from Source/Python and returns its exit code. --print shows
the command without running it.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent
BUNDLE = REPO / "weights" / "b1_continue_B1K_skip13_492800"
CHECKPOINT = BUNDLE / "run" / "checkpoints" / "ckpt_step_494025"
START_STEP = 494025
DEV_ROOT = r"C:\Users\bhart\Documents\UEMA_devA_private"   # the development repository the command names


class PathMapper:
    """Maps a path the saved command names inside the development repository onto this checkout.

    Inputs: dev_root, the development repository the saved command names (a Windows path written with backslashes,
    DEV_ROOT above); repo, this checkout's folder. Operation: a path and dev_root are compared with every backslash
    replaced by "/" and in lower case (the development machine's file system compares paths without case), as
    mimo_bridge/repo_paths.py does, so the comparison gives the same answer whatever os.sep is ("\\" on Windows,
    "/" on macOS and Linux; joining os.sep onto the backslash dev_root matched nothing off Windows). Outputs:
    local(path) is the path re-rooted at repo (the part after dev_root, split on "/", joined onto repo) or None when
    the path does not lie inside dev_root; foreign(path) is True when the path starts with a drive letter, a path of
    the development machine this script does not map.
    """

    def __init__(self, dev_root: str, repo: Path) -> None:
        self.dev_root = dev_root
        self.repo = repo
        self._prefix = self._norm(dev_root).rstrip("/") + "/"

    @staticmethod
    def _norm(text: str) -> str:
        return text.replace("\\", "/").lower()

    def local(self, path: str) -> Optional[Path]:
        """The checkout path for `path` when it lies inside dev_root, else None."""
        if not self._norm(path).startswith(self._prefix):
            return None
        rel = path.replace("\\", "/")[len(self._prefix):]
        return self.repo.joinpath(*[p for p in rel.split("/") if p])

    @staticmethod
    def foreign(path: str) -> bool:
        """True when `path` starts with a drive letter (a path of the development machine)."""
        return re.match(r"^[A-Za-z]:[\\/]", path) is not None


def command(out: Path, steps: int, mapper: PathMapper, device: str = "auto") -> list:
    """The trainer command for this machine.

    Inputs: out, the folder the run writes to; steps, model steps to train after START_STEP; mapper, the PathMapper
    that re-roots the saved command's development-repository inputs at this checkout; device, the --device value the
    trainer resolves ("auto", "cpu", "cuda", "cuda:N" or "mps"). Operation: reads the saved command, replaces the
    interpreter, the output paths and the device, sets --max-steps, and checks that every input the saved command
    names inside the development repository is in this checkout (an input it names elsewhere on the development
    machine is refused). Output: the argument list, or SystemExit naming every missing or unmapped input.
    """
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
        "--device": device,
    }
    args = [sys.executable] + saved[1:]
    problems = []
    for i in range(4, len(args)):
        flag = args[i - 1]
        if flag in out_of:
            args[i] = str(out_of[flag])
        elif flag == "--max-steps":
            args[i] = str(START_STEP + steps)
        else:
            local = mapper.local(args[i])
            if local is not None:
                if not local.exists():
                    problems.append("%s %s: not in this repository (%s)" % (flag, args[i], local))
            elif mapper.foreign(args[i]):
                problems.append("%s %s: a path of the development machine this script does not map" % (flag, args[i]))
    if "--max-steps" not in args:
        args += ["--max-steps", str(START_STEP + steps)]
    if "--device" not in args:
        args += ["--device", device]
    if problems:
        raise SystemExit("cannot build the command:\n  " + "\n  ".join(problems))
    return args


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="a new folder for this run's records, checkpoints and latent-body state")
    ap.add_argument("--steps", type=int, default=100, help="model steps to train after 494,025 (default 100)")
    ap.add_argument("--device", default="auto",
                    help="the torch device the trainer runs the model on: auto (default; cuda:0, else mps, else cpu), "
                         "cpu, cuda, cuda:N or mps; an unavailable name stops the trainer with a message naming it")
    ap.add_argument("--print", action="store_true", help="show the command and stop")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if a.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if not CHECKPOINT.joinpath("model.safetensors").is_file() or CHECKPOINT.joinpath("model.safetensors").stat().st_size < 1_000_000:
        raise SystemExit("the saved weights are missing or are Git LFS pointer files: run 'git lfs pull' in %s" % REPO)
    args = command(out, a.steps, PathMapper(DEV_ROOT, REPO), a.device)
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
