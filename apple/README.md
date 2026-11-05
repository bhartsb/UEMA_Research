# UEMAKit and UEMA Mind: the Apple Swift port

**Continuing this work:** start with [docs/apple_port/HANDOFF_2026-11-04.md](../docs/apple_port/HANDOFF_2026-11-04.md), the state of every work item and the open issues as of November 4, 2026. Then [FIRST_BUILD_CHECKLIST.md](FIRST_BUILD_CHECKLIST.md) for the Mac build, and [linux_check/](linux_check/README.md) for the tests that run without a Mac.

This folder holds target B of the Apple port, a Swift package and a SwiftUI app that run one MIMo body
with the released UEMA weights on macOS and iOS without Python, and under `apple/scripts/` the Python
export scripts of target A that write the package's resources (plan page, section 4). The requirements
are in `docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html` (section 4, REQ-B-1 to REQ-B-18) and the
work items, their states and the Mac first-build checklist are in
`docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html` (sections 5 to 9). Every fact below is
copied from those two pages and the file:line sources are theirs, except the one measured grep result
in the last section, which was run here on November 4, 2026.

## Nothing under apple/ was compiled on Windows

The build machine is Windows 11 with no Swift toolchain and no Xcode (plan page, section 3, fact 1).
Every Swift file under `apple/` was written to compile on first try and has not been compiled by
anyone yet. The Swift work items are in state BUILT (uncompiled) until the founder (the project's
owner) runs the sequence below on a Mac; only then can a Swift item reach VERIFIED (plan page,
section 9). The Python scripts under `apple/scripts/` are target A items (WI-12 to WI-15, WI-34),
which the plan page verifies by running here on Windows with a test beside each; the shell script
`apple/scripts/build_mujoco_xcframework.sh` (WI-18) runs only on the Mac, in step 2 below.

## What the package is

`apple/UEMAKit` is a Swift package (a folder with a `Package.swift` manifest). Its modules follow the
nine stages of one model step (one 50 ms model update) in the sequence the numbers flow (plan page,
section 2):

| Module | Stage | Input | Output |
| --- | --- | --- | --- |
| `UEMAPhysics` | 1 | 90 motor controls, each -1 to 1 | joint positions and velocities, 168 sensor values (56 sensors), body poses, contacts after 10 physics steps of 5 ms |
| `UEMASenses` | 2 | the physics arrays | raw arrays per sense (vision, tactile, audio, proprioception, motor, vestibular) |
| `UEMATokenizer` | 3 | raw arrays per sense | encoder inputs (fixed divisors; vision cut into 16 x 16 x 3 patches) |
| `UEMAModel` | 4, 5, 6, 7 | encoder inputs | one 192-number token per sense; the forward model's state over 6 tokens x 128 model steps = 768 tokens; the predicted next token per sense; the motor controller's command |
| `UEMAPlanner` | 8 | the goal (the highest-wellness entry of the goal ring, the memory of the last 200 model steps) and the predicted tokens | one command refined by Adam (learning rate 0.1) for 20 updates through the frozen forward model, every 10th model step once the goal ring holds 20 entries |
| `UEMACommand` | 9 | the command (26 body handles x 9 numbers) | 90 motor controls, to stage 1 |

Three more modules carry no stage of their own: `CMuJoCo`, the C module over the MuJoCo XCFramework
(WI-19); `UEMACore`, the value types (Command, Token, SenseName, ModelStep) and the protocols every
other module depends on (WI-20); and `UEMAScene`, the shipped scene bundle and skin layout (WI-26). The
target sequence in `Package.swift` is UEMACore, UEMAPhysics, UEMAScene, UEMASenses, UEMATokenizer,
UEMAModel, UEMAPlanner, UEMACommand, each with one XCTest target (WI-17).

MuJoCo, the physics program written in C, enters the package as an XCFramework (one compiled library
per Apple platform) built from MuJoCo 3.3.0 source by `apple/scripts/build_mujoco_xcframework.sh`
(specification REQ-B-1). The encoders, forward model, prediction heads, motor controller and planner
are MLX Swift modules (Apple's array and neural-network library for Swift, with gradients and Adam;
specification REQ-B-9 to REQ-B-12). The weights come from `model.safetensors`, 73,119,988 bytes
(MANIFEST.json:120 by way of the plan page, section 1), the checkpoint of model step 494,025 with
18,131,745 parameters (README.md:27); the file is not in git and is fetched in step 1 below.
No learning happens on the device: the weights are frozen and the only gradient is the planner's with
respect to the command (specification REQ-B-17).

## What the app is

UEMA Mind (`apple/UEMAMind/`, an Xcode project written from `apple/project.yml` by XcodeGen; WI-33,
WI-38) is a SwiftUI app with
two schemes, `UEMAMind-macOS` and `UEMAMind-iOS`, for macOS 14 and iOS 17 or later. It runs one body
with the package above, inference only, and shows (specification REQ-B-14): the scene view with
play, pause and "one model step", the model-step counter and the 5 eye images; the sense panel (per
sense the input's shape, minimum and maximum, the token as a 12 x 16 heat map, the prediction error);
the plan panel (goal ring fill from 0 to 200, the goal's wellness and rank, the planner's 21 scores,
the command as a 26 x 9 table); and settings (renderer, compute device, babble off by default,
planner on or off, memory readout).

## Prerequisites (plan page, section 8, step 1)

A Mac with Apple silicon, Xcode with the macOS 14 and iOS 17 SDKs, CMake 3.16 or newer, XcodeGen,
network access. Tool versions are unverified here: no Mac was available when this file was written.

## The Mac first-build sequence (plan page, section 8)

Run the steps in this sequence. Each step names its pass line and the work item it verifies.

1. `git checkout feature/apple-port-2026-11-04` in a clone of the repository, then
   `python fetch_weights.py --accept-license` for `model.safetensors` (73,119,988 bytes; not in git).
2. `bash apple/scripts/build_mujoco_xcframework.sh`: exit 0; three slices (ios-arm64,
   ios-arm64-simulator, macos-arm64); `xcrun lipo -info` names arm64 for each (WI-18).
3. `cd apple/UEMAKit && swift package resolve`: exit 0 (WI-17).
4. `xcodebuild -scheme UEMAKit-Package test -destination 'platform=macOS'`: every XCTest target
   passes; the tests of WI-28, WI-29, WI-35 and WI-36 print their measured differences, which go into
   the plan page's section 9 as pass lines.
5. `cd apple && xcodegen generate && xcodebuild -scheme UEMAMind-macOS build`: exit 0 (WI-33, WI-38).
6. Run UEMAMind-macOS; pick `model.safetensors`; press "one model step" ten times: the counter reads
   10 and the five eye images change (WI-38, WI-40).
7. `xcodebuild -scheme UEMAMind-iOS -destination 'generic/platform=iOS' build`: exit 0 (needs a
   signing team; unverified until done); a device run records peak memory from MemoryView.
8. Write `apple/FIRST_BUILD_RECORD.md` (WI-41) and update the plan page's section 9.

MLX Swift's Metal shaders are not built by command-line SwiftPM, so steps 4 to 7 go through
`xcodebuild` or Xcode (specification REQ-B-10; research1 torch-apple).

## Writing rules for files under apple/

`docs/apple_port/banned_words.txt` lists the writing rules' seven banned phrases, one per line:
the one word the rules forbid outright and the six imprecise magnitude words. The check on this file is
`grep -c -i -f docs/apple_port/banned_words.txt apple/README.md`, which must print 0 (the count of
matching lines; grep exits 1 when that count is 0). On the Windows build machine the shell's GNU grep
3.0 aborts with exit 134 on `-i` with two or more patterns under its default locale (measured
November 4, 2026, on a one-line control file too), so there the check runs as
`LC_ALL=C.UTF-8 grep -c -i -f docs/apple_port/banned_words.txt apple/README.md`: that form printed 0 on
this file and 7 on a control file of seven planted lines, one per phrase, in mixed case. Every number in
a file under `apple/` carries its unit and its source; no duration estimates; the canonical names
(model step, command, wellness channel; the senses vision, tactile, audio, proprioception, motor,
vestibular, intermind).
