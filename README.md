# UEMA - Unified Embodied Mind Architecture (research release)

UEMA is Mind Machine Learning's research program to build conscious AI: a mind that learns through a
body, develops a sense of self, and comes to understand its world through experience. This repository is
that mind's code, running on the MIMo infant body in the MuJoCo physics simulator, with a saved model you
can download, run, change and continue training. It is here for people who want to work on that question
with the machinery in their hands.

**What you get.** The model code (one encoder per sense, the forward model, the wellness terms, the
planner, the training losses); the MIMo body and the bridge that runs the model on it, with three scenes
(`reach`: MIMo hangs with a small ball floating in front of the right hand; `selfbody`: MIMo sits with both
hands free; `floor`: MIMo stands free, falls and rolls); the complete saved state of a model at model step
494,025 (one model step is 50 ms of simulated time: 6.9 hours of each of eight bodies' lives), 1.68 GiB; a
dashboard with the test registry, the investigations and a movement viewer; an explainer that walks the
architecture component by component against the code; and the specifications the design comes from.

**How the model works, in one paragraph.** Each of seven senses (vision, tactile, audio, proprioception,
motor, vestibular, intermind) is encoded into tokens by its own encoder. A transformer, the forward model,
reads this model step's tokens and the command and predicts the next model step's tokens. A command is 9
numbers for each of 26 body handles (target position, rotation, radius, strength); handle 23 is the eyes:
gaze is a limb here, not a separate system. Wellness terms score a predicted body state from the body's
own senses. The planner searches for the command whose predicted outcome scores best; the body executes
it; the difference between the prediction and what the senses then report is the training signal for
every part. There is no policy network and no dataset: the model trains on its own stream while the body
lives. The specification's first section states why the design has no policy network.

**Where the model stands.** The saved model is small (18,131,745 parameters: 192-number tokens, 4
transformer blocks; the specification's full configuration is about 271M at d_model 768) and early. It is
a common starting point for experiments, not a demonstration of a finished ability. What it can and cannot
do is measured, with dates, pass lines and results, in the dashboard's test registry
(`docs/gain_of_function_v4/test_registry.json`, 643 rows) and the ability rechecks
(`Saved/mimo_bridge/ability_rechecks`); read those before quoting an ability. Training was developed on an
8 GB graphics card; Requirements gives a measured 100-model-step resume on a laptop.

Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved. Licensed under the Mind Machine
Learning Noncommercial License 1.0 (`LICENSE`), a research license: use, changes and sharing for
noncommercial purposes only; it grants no patent license. `NOTICE` states what it covers, the patents and
the third-party parts (MIMo keeps its MIT License).

## Quick start

1. Clone: `git clone https://github.com/bhartsb/UEMA_Research.git`
2. In the clone: `python setup_checkout.py`. Git does not keep file modification times and some checks
   read them; this restores them. Run it once, before anything else.
3. Build the Python environment as in **Install** below.
4. `python fetch_weights.py --accept-license`. The flag states your agreement to the license, which covers the
   weights and whatever is trained with them (`LICENSE`, Trained Models and Other Outputs); without it the script
   downloads nothing. The saved weights (1.68 GiB) are in the public Hugging Face repository
   `mml-user/uema-research-weights`; the script downloads them without an account and checks every file's
   SHA-256.
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

## Experiments we hope you run

Each experiment here has one form: the same saved state, one change, the same scenario run with the
mechanism under test switched off as the control, and a measurement named before the run. The plan these
come from is `docs/UEMA_Specs/fastGainOfFunction-4.docx` (sections 2, 5, 8 and 9); the invitation page
`docs/CONSCIOUS_BEINGS_RESEARCH_INVITATION_2026-10-28.html` puts them in context.

1. **One movement, traced end to end** (start here). Follow one movement from the camera and tactile
   readings to the predicted outcome and the resulting contact. Then ask whether one more piece of
   sensory information improves the next prediction. The explainer and the movement viewer are the tools.
2. **Does the tactile sense tell a secure grip from a slip?** Compare access to the tactile tokens from
   the same saved state; measure prediction error on contact and whether the object stays in the hand
   across repeated trials.
3. **Sense ablation.** Vision only; vision + tactile; vision + tactile + proprioception; all senses. Same
   saved state, same compute. Which reaches a discrimination first, and how many model steps each needs.
4. **Self-touch discovery.** In the `selfbody` scene: does prediction error for self-contact events drop
   across trials, and does the self-model's coherence score (the plan's section 7, metric 6) rise?
5. **Body-schema extension.** Does a grasped object enter the self-model's boundary, and leave it when
   the object is removed? Needs a graspable object and a scripted removal, which the release does not
   yet build.
6. **Weight-encoded temporal memory.** An event that recurs at a fixed period: does prediction error at
   the event's time fall across cycles? Controls: skip one cycle, shift the phase, move the event's
   location. Needs a scene the release does not yet build.
7. **Scaling.** The released 18M model against the specification's 271M configuration at equal compute:
   prediction-error convergence and probe accuracy (a small linear classifier read from the model's
   tokens, labels taken from the simulator's state).
8. **The hard questions.** How should a model learn a continuous movement? How does it keep useful
   abilities while acquiring new ones? What distinguishes learning about the world from learning about
   itself? Does a learned distinction survive a new situation? The ability the program is working toward
   is a reach: a ball just out of reach, the eyes following it, the body shifting its weight, the hand
   adjusting on contact.

Items 5 and 6 need environment work before the measurement; building that scene is itself a contribution.

## Read first

- `docs/CONSCIOUS_BEINGS_RESEARCH_INVITATION_2026-10-28.html` - why this project exists and what it asks of you.
- `docs/UEMA_Specs/UEMA_Spec_v3_5.docx` (also `.txt`) - the architecture specification: the thesis, the
  encoders, the forward model, wellness, intention as model inversion, the self-model, the training streams.
- `docs/UEMA_Specs/fastGainOfFunction-4.docx` - the plan: three headline demonstrations, the developmental
  ladder, the experiment harness, the key experiments.
- `docs/UEMA_Specs/ManyMinds_Spec_v2.md` - the many-minds extension (several minds sharing one body).
- The explainer (quick start, item 5) - the code's components, each against the specification's section.

## Requirements

- Python 3.13 (developed with 3.13.3 on Windows 11).
- An NVIDIA graphics card with a CUDA 12.4 capable driver for the pinned `torch==2.6.0+cu124`
  build. Training was developed on an 8 GB card. A Mac with Apple silicon: **Running on macOS (Apple
  silicon)** below (its own requirements file and device).
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
state and trajectory; 6 to 621 MB) are not in this Git repository: `python fetch_weights.py --accept-license` downloads them from
the Hugging Face repository `mml-user/uema-research-weights` (step 4 of the quick start).

## Running on macOS (Apple silicon)

The macOS port is work in progress on the branch `feature/apple-port-2026-11-04`. What it must do is stated, as
requirements with pass lines, in [UEMA on Apple platforms: specification](docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html) (target A, section 3); each
change is one work item of [UEMA on Apple platforms: implementation plan](docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html), whose section 9 (Status) states
which items have landed on the branch. Nothing in this section was run on a Mac: every Python change is verified
on Windows with the CPU device standing in for the Apple GPU, with one world, never eight (the plan's section 3).

**Install.** Use a native arm64 Python 3.13 (`python3 -c "import platform; print(platform.machine())"` prints
`arm64`; MuJoCo refuses an x86_64 Python under Rosetta at import). From the repository root:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-macos.txt
pip install -e third_party/MIMo/mimo --no-deps
```

`requirements-macos.txt` holds the 21 pins of `requirements.txt` plus torch 2.6.0 (22 in all), torch from the
default index since the macOS wheel has no CUDA; its header names the macOS wheels of torch, mujoco, opencv, scipy and
scikit-learn and the install floor, macOS 13.0 (set by the opencv wheel), measured on Nov 4, 2026 with `pip download`
(one wheel per pin, 22 wheels).

**Device.** The trainer runs on the Apple GPU through torch's `mps` device or on the CPU. `--device auto` (cuda,
then mps, then cpu; a named device that is unavailable stops with a message naming it) is the plan's work item WI-2
and is pending until that item lands: the plan's section 9 records it. On `mps`, torch has no float64, so the
float64 sites compute on the CPU and hand the model float32 (the plan's WI-5), and the model runs in float32 without
autocast (WI-3); both are pending until they land (section 9). The macOS floor for `mps` on torch 2.6.0 is unverified
(the header of `requirements-macos.txt` gives the two published floors).

**Resuming the released checkpoint.** The saved episode companion (`episode_state.pt.gz`, 621,243,950 bytes) was
written by a run on `cuda:0` and carries that device's random-number state. A resume on `cpu` or `mps` cannot
restore it and stops with `exact continuation requires the saved execution device`. `--rng-continuation reseed`
(the plan's work item WI-11, pending until it lands; section 9) drops that state, logs `cuda_rng dropped`, and calls
`torch.manual_seed` with the first 8 bytes of the SHA-256 of the dropped bytes: the resume is repeatable from the same
companion, and every later random draw differs from the saved run's. The port's verification runs use one world
(`--n-envs 1`, the plan's WI-24); eight worlds on a Mac are unmeasured.

**The live viewer.** The dashboard's Live tab opens a window on a training body through MuJoCo's passive viewer
(`mimo_bridge.tools.live_viewer`). On macOS that program must be started with `mjpython`, the launcher the mujoco
wheel installs beside the venv's python (`.venv/bin/mjpython`), because the viewer draws on the main thread; the
Live tab's copy-and-paste line names it on macOS. Training, the dashboard and the explainer need no `mjpython`.

**Memory and speed.** On Windows the trainer and its eight world processes added about 30 GB of committed memory
(44.0 GB to 73.8 GB; Requirements above) at 0.41 to 0.46 model steps per second. The Mac figures are unmeasured:
no release file holds a Mac run; the plan's section 10 states that the Mac's speed is unmeasured and that no work item
measures it, and no work item measures the trainer's memory on a Mac.

## Starting the dashboard and the explainer

Before the first start, do steps 1 to 4 of the quick start (clone, `python setup_checkout.py`, the Python
environment, `python fetch_weights.py --accept-license`).

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

## Contributing

Pull requests are not merged yet: `CONTRIBUTING.md` states the contributor agreement that must come first.
Welcome now: issues, questions, experiment reports and reviews. An experiment report is an issue with: the
question, the saved state it started from, the one change, the control, the measurement named before the
run, the result, and the commit of the code that produced it.

## Work in progress on other branches

`branches/` holds one patch per development branch that changed this code after the release's
source commit and is not part of it, with a table in `branches/README.md`. They are not applied.

## Apple Swift port (apple/)

`apple/` holds UEMAKit, a Swift package, and UEMA Mind, a SwiftUI app for macOS 14 and iOS 17 or later. Together
they run one MIMo body in MuJoCo with the released forward model and planner, without Python and without learning:
the weights stay frozen and the only gradient is the planner's with respect to the command. The requirements are
target B of [UEMA on Apple platforms: specification](docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html) (section 4); the work items, their states and the
sequence to build it on a Mac are in [UEMA on Apple platforms: implementation plan](docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html) (sections 5 to 9).
Nothing under `apple/` was compiled: this repository was built on Windows 11 with no Swift toolchain and no Xcode,
so the highest state a Swift item reaches here is BUILT (uncompiled) until its first build on a Mac. `apple/README.md`
is the package's own page.

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
- The macOS port (**Running on macOS** above) is verified on Windows with the CPU device standing in for
  `mps`, one world; no release file holds a measurement from a Mac (speed, memory, which operators `mps` lacks,
  whether bf16 autocast on the Apple GPU rounds as CUDA bf16 does). `mps` runs float32 without autocast (the plan's
  WI-3, pending until it lands).
- The released episode companion holds a CUDA random-number state: a resume on `cpu` or `mps` needs
  `--rng-continuation reseed` once it lands, after which later random draws differ from the saved run's; exact
  continuation off CUDA is out of scope (the specification's section 6).
- Nothing under `apple/` was compiled (no Swift toolchain on the build machine). MuJoCo 3.3.0 has no official iOS
  build; whether it compiles with the plan's flags is unverified until the first build on a Mac (the plan's WI-18).
