# UEMA - Unified Embodied Mind Architecture (research release)

Model code of the UEMA research program of Mind Machine Learning, Inc., running on the
MIMo infant body in the MuJoCo physics simulator.

Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved. Licensed under the Mind Machine
Learning Noncommercial License 1.0 (`LICENSE`), a research license: use, changes and sharing for
noncommercial purposes only; it grants no patent license. `NOTICE` states what it covers, the patents and
the third-party parts (MIMo keeps its MIT License).

## Quick start for reviewers

1. Clone: `git clone https://github.com/bhartsb/UEMA_Research.git`
2. In the clone: `python setup_checkout.py`. Git does not keep file modification times and some checks
   read them; this restores them. Run it once, before anything else.
3. Build the Python environment as in **Install** below.
4. `python fetch_weights.py`. The saved weights (1.68 GiB) are in a private Hugging Face repository,
   `mml-user/uema-research-weights`; the script reads them with the token in `hf_reviewer_token.txt`, which can
   only read that repository, and checks every file's SHA-256. No Hugging Face account is needed. Do not reuse
   or share the token: Mind Machine Learning revokes it to end access.
5. The dashboard (tests, investigations, components, the movement viewer) and the explainer, from
   `Source/Python`: `python -m mimo_bridge.tools.development_dashboard --port 8766`, then open
   http://127.0.0.1:8766/ ; the explainer page is http://127.0.0.1:8766/files/Saved/mimo_bridge/explainer.html .
   How to rebuild the explainer and stop both: **Starting the dashboard and the explainer** below.
6. Continue training from the saved state (model step 494,025), from the repository root:
   `python resume.py --out <a new folder> --steps 100` (see **Resuming training** below).

The clone can sit in any folder. The records name files inside `D:\UEMA_Research`, the folder the
release was built in; `Source/Python/mimo_bridge/repo_paths.py` reads every such path from the clone.

## What is in this repository

| Folder | What it holds |
| --- | --- |
| `Source/Python/tokenizer` | Turns each sense's raw frame (vision, tactile, audio, proprioception, motor, vestibular, intermind) into the token arrays the model reads; `config.py` fixes every array size. |
| `Source/Python/lem` | The Learning Embodied Model: one encoder per sense, the forward model (a transformer that predicts the next model step's tokens from the current ones and the command), the wellness terms, the planner that searches a command against the forward model, the training losses and checkpoints. |
| `Source/Python/mimo_bridge` | Runs the model on the MIMo body: the environment (`env.py`), the trainer (`train_mimo.py`), the command decoder, probes, the explainer and the dashboards, and their tests. |
| `Saved/mimo_bridge/scale_anchors.json` | Measured reference points (with their dates and commands) for the scales the tools report. |
| `third_party/MIMo` | The MIMo infant body this code runs on (`mimo/`, MIT License), the patch this code applied to it, and MIMo's license. |

Not included: the Unreal Engine avatar and the Python side of its frame pipeline (shared-memory
frame transport, Direct3D 12 / CUDA interop, process-separated replay training). One module of
that side, `lem/live_test.py`, is present because `mimo_bridge/lem_model.py` imports two
functions from it; it needs the Unreal pipeline to run as a program.

## Requirements

- Python 3.13 (developed with 3.13.3 on Windows 11).
- An NVIDIA graphics card with a CUDA 12.4 capable driver for the pinned `torch==2.6.0+cu124`
  build. Training was developed on an 8 GB card.
- Measured on Oct 22, 2026 by resuming a clone for 100 model steps on a laptop RTX 4070 (8 GB): the graphics
  card's memory peaked at 4,287 MB; Windows' committed memory (RAM plus page file) rose from 44.0 GB to 73.8 GB
  while it ran, about 30 GB for the trainer and its eight world processes; it trained 0.41 to 0.46 model steps
  per second (one model step = 50 ms of each of the eight bodies' simulated time); the whole run, loading
  included, took 376 s.
- MIMo, the infant body, is included in `third_party/MIMo/mimo`: MIMo (https://github.com/trieschlab/MIMo,
  MIT License) at commit `f2be26b` of its `v3` branch with the patch `third_party/MIMo/MIMo_bridge_work.patch`
  applied (five files; what it changes is listed in `third_party/MIMo/README.md`).

## Install

From the repository root:

```
python -m venv .venv
.venv\Scripts\activate            (Windows)   or   source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e third_party/MIMo/mimo --no-deps     (MIMo pins older versions of numpy and torch; this code
                                                    was developed with the versions in requirements.txt)
```

## Run

All commands run from `Source/Python`.

```
python -m pytest lem/tests tokenizer/tests mimo_bridge/tests -m "not slow"
python -m mimo_bridge.train_mimo --help
python -m mimo_bridge.tools.dashboard --help
```

Tests marked `slow` build a MuJoCo body.

## Resuming training from the included checkpoint

`weights/b1_continue_B1K_skip13_492800/` holds the complete saved state at model step 494,025
(weights, optimizer, runtime state, the eight worlds' episode companion and the latent-body state).
`python resume.py --out <a new folder> --steps N` (from the repository root) runs the exact command that
produced this state (`weights/.../run/command.json`) with only the machine's parts changed: this Python, this
repository's saved state and input files, the new folder for everything the run writes, and the stop step
(494,025 + N). `--print` shows the command without running it; `weights/b1_continue_B1K_skip13_492800/RESUME.md`
explains the values. Its six largest files (weights, optimizer, runtime state, episode companion, latent-body
state and trajectory; 6 to 621 MB) are not in this Git repository: `python fetch_weights.py` downloads them from
the private Hugging Face repository `mml-user/uema-research-weights` (step 4 of the quick start).

## Starting the dashboard and the explainer

Before the first start, do steps 1 to 4 of the quick start (clone, `python setup_checkout.py`, the Python
environment, `python fetch_weights.py`).

**Start the dashboard.** In a terminal, with the Python environment activated:

```
cd Source/Python
python -m mimo_bridge.tools.development_dashboard --port 8766
```

When it prints `Development dashboard: http://127.0.0.1:8766/`, open that address in a browser. It shows the
test registry and the investigations, the components, the movement viewer and the training records. Another
port: change `--port`.

**Open the explainer.** With the dashboard running, open
http://127.0.0.1:8766/files/Saved/mimo_bridge/explainer.html . Open it through the dashboard, not as a file from
disk: the page asks the dashboard for the components and the investigations (`/api/components`,
`/api/investigations`), and a page opened from disk cannot ask.

**Rebuild the explainer** (for example after the records change). From `Source/Python`, with the dashboard
running or not:

```
python -m mimo_bridge.tools.build_explainer --run ../../weights/b1_continue_B1K_skip13_492800/run --no-run-copy
```

It rewrites `Saved/mimo_bridge/explainer.html`; reload the page.

**Stop.** Ctrl+C in the dashboard's terminal stops the dashboard. The dashboard also starts the movement viewer's
service (it draws the movement pictures) as a separate background process, which keeps running; stop it with

```
# Windows PowerShell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'movement_viewer service' } | ForEach-Object { Stop-Process -Id $_.ProcessId }
# Linux or macOS
pkill -f "mimo_bridge.tools.movement_viewer service"
```

**What they write.** The dashboard writes its own status files under `Saved/mimo_bridge` (a launcher page, its
service records, the older summary pages) and the movement pictures under `external/D/UEMA_movement_viewer/pictures`;
`git status` lists them afterwards. They are local; `git checkout -- Saved` discards the changed ones before a pull.

**Where the records come from.** The records both read are included: `docs/` (the test registry, the component
manifest, the defect register, the investigations), `Saved/mimo_bridge/` (training-run statistics, ability
rechecks, the movement viewer's index, probe results), and the files they read from drive D: under `external/D/`.
The records name those files by absolute path inside `D:\UEMA_Research`; `mimo_bridge/repo_paths.py` reads each
such path from the clone, wherever it sits.

Some records carry SHA-256 fingerprints of other records: a test run's seal and its result's digest in the test
registry, the fingerprint of the run's result file, the registry audit's list of rows from before the seal, and the
ability evidence receipts. Removing vendor names and pointing paths into this release changed the text those
fingerprints cover, so each one was recomputed for this copy, only where it matched in the development repository
and the copy is exactly that record with the same text rewrite. Every recomputed fingerprint, with its old and new
value, is listed in `docs/release_reseal_receipt.json`.

## Work in progress on other branches

`branches/` holds one patch per development branch that changed this code after the release's
source commit and is not part of it, with a table in `branches/README.md`. They are not applied.

## Known limits of this release

- Many tools under `mimo_bridge/tools` and many tests under `mimo_bridge/tests` carry paths
  of the development machine (`C:\Users\...\UEMA_devA_private`, `D:\UEMA_...`). Some of those
  tests skip when the path is absent; others fail until the path is edited. Tools that default to
  such a path need it given on the command line.
- The training preflight checker and the test runner read records of the development repository that
  are not part of this release. The dashboard and the explainer read only files this release holds
  (checked on Oct 22, 2026 by recording every file they open from a clone in another folder).
- The included checkpoint's run records (`weights/.../run/args.json`, `command.json`, logs) carry
  the development machine's absolute paths.
