# Apple port: first-build checklist

Written 08:15 PM PST, November 4, 2026, after the cross-module consistency pass (item 1 of
[docs/apple_port/HANDOFF_2026-11-04.md](../docs/apple_port/HANDOFF_2026-11-04.md)). Section 1 is the
Mac sequence of the plan page's section 8 (`docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html#firstbuild`),
one pass line per step. Section 2 is what already compiled and passed on Linux, so a Mac failure in
one of those parts points at a platform difference. Section 3 lists the mismatches that remain.

## 1. The Mac sequence

Record each command, its exit code and the clock time (AM/PM Pacific) in `apple/FIRST_BUILD_RECORD.md`
(WI-41; the specification's REQ-B-18 names the same record `apple/records/first_build.md`).

1. **Prerequisites.** A Mac with Apple silicon; Xcode with the macOS 14 and iOS 17 SDKs; CMake 3.16 or
   newer; XcodeGen; network access. Record `sw_vers -productVersion`, `xcodebuild -version` and
   `swift --version` (the Swift version decides the mlx-swift tag: 6.3 or newer resolves 0.33.0, 6.0
   to 6.2 resolves 0.31.4; `apple/UEMAKit/Package.swift`, "THE mlx-swift VERSION").
2. **Checkout and weights.** `git checkout feature/apple-port-2026-11-04`, then
   `python fetch_weights.py --accept-license` (model.safetensors, 73,119,988 bytes, not in git).
3. **Target A on the Mac.** Native arm64 Python 3.13, `pip install -r requirements-macos.txt`,
   `PYTORCH_ENABLE_MPS_FALLBACK` unset; the resume runs of plan section 8 step 3. Pass line: each run
   reaches 100 model steps with a finite loss; records under `apple/records/mac_python_<device>_<worlds>.json`.
4. **Physics wheel divergence.** `python apple/scripts/export_fixture.py --replay-physics Tests/Fixtures/run200`
   with the macOS arm64 mujoco 3.3.0 wheel; the maximum qpos difference goes to
   `apple/records/physics_wheel_divergence.json`.
5. **MuJoCo XCFramework (WI-18).** `bash apple/scripts/build_mujoco_xcframework.sh`: exit 0; three
   slices; `xcrun lipo -info` names arm64 for each. Never run yet.
6. **Resolve (WI-17).** `cd apple/UEMAKit && swift package resolve`: exit 0; commit `Package.resolved`.
7. **Package tests on macOS.** `xcodebuild -scheme UEMAKit test -destination 'platform=macOS'`:
   every test target passes. Test functions per target, counted in the files on November 4, 2026:
   CMuJoCoTests 12, UEMACoreTests 31, UEMAPhysicsTests 24, UEMASceneTests 27, UEMASensesTests 103
   (86 of them already passed on Linux, section 2; the other 17 are GeomMirrorTests 10 and
   SceneKitEyeRendererTests 7), UEMATokenizerTests 30, UEMACommandTests 35, UEMAModelTests 54,
   UEMAPlannerTests 36, UEMAIntegrationTests 10; no machine has compiled the last three. Each
   target's header lists its own "First build on a Mac" steps.
8. **Package tests on the iOS Simulator.** The same with `-destination 'platform=iOS Simulator,name=<an installed iPhone>'`.
9. **The app (WI-33, WI-38).** `cd apple && xcodegen generate && xcodebuild -scheme UEMAMind-macOS build`: exit 0.
10. **The app runs.** Run UEMAMind-macOS, pick model.safetensors, press "one model step" ten times: the
    counter reads 10 and the eye images change (WI-38, WI-40). This needs the real stack (section 3, item 1).
11. **iPhone.** `xcodebuild -scheme UEMAMind-iOS -destination 'generic/platform=iOS' build` (needs a
    signing team), then WI-40's tests on a connected iPhone with the increased-memory-limit entitlement,
    peak memory against the 3,000 MB line.
12. **Records.** Write `apple/FIRST_BUILD_RECORD.md` and move each section 9 row of the plan page to
    VERIFIED or FAILED with the commit.

## 2. Already compiled and passed on Linux

`bash apple/linux_check/run.sh` (README beside it) compiles a second manifest over the same source folders
with Swift 6.2 for Ubuntu 24.04 and the MuJoCo 3.3.0 Linux release. Run at 08:10 PM PST, November 4,
2026: 242 tests, 0 failures.

| Part | Tests that passed | Measured |
| --- | --- | --- |
| CMuJoCo (WI-19) | CMuJoCoSymbolTests 8, CMuJoCoPackageTests 1 | mj_versionString() "3.3.0"; the importer forms the header doubted compile |
| UEMACore (WI-20) | 31 | |
| UEMAPhysics (WI-25) | 24, the acceptance test among them | sizes equal `PhysicsSizes.released`: nq 146, nv 136, nu 90, nbody 76, njnt 104, nsensordata 168, ngeom 188, ncam 6 |
| UEMAScene (WI-26) | 27 | |
| UEMASenses (WI-27, WI-28, part of WI-29, the new readers) | BodySenseReaderTests 22, tactile 42, CameraSpecTests 13, VisionReaderTests 4, AudioReaderTests 3, package 2 | every WI-28 measured difference 0.0; the audio render equals the Python `CribAudio` within 3.4e-16 (spectrum) and 1.7e-15 (metadata) |
| UEMATokenizer (WI-30) | 30 | N = 1,160 on the 3 WI-14 states, angles within 3.4e-16 rad, patch and angle digests equal |
| UEMACommand (WI-37) | 35 | released decoder's reach equals the receipt within 0.01 cm; held commands move qpos by 0.93 to 2.06; body function modulate runs 50 env steps |
| UEMAMind view models (WI-32) | compiled, with OpenCombine for Combine; RunViewModelTests compiled, not run | |

Not covered on Linux: UEMAModel, UEMAPlanner, their tests and UEMAIntegrationTests (mlx-swift does not
build on Linux: 0.31.4 stops at MLXNN's MLXFast calls, 0.32.3 at cuda.h, 0.33.0 at Metal/Metal.hpp,
all measured November 4, 2026); GeomMirror.swift and SceneKitEyeRenderer.swift (SceneKit, Metal); the
app's views and UEMAMindApp.swift (SwiftUI). For those files a script compared every type name they use
with the declarations in `apple/`: all resolve except the four of section 3, item 1.

## 2a. The first Mac build and the fixes after it

The first build ran on November 4, 2026 (`apple/FIRST_BUILD_RECORD.md`: MacBook Air M5, macOS 26.7.2,
Xcode 27.2, Swift 6.4.2, mlx-swift 0.33.0). Everything compiled; 372 tests ran with 20 assertion
failures in three places. What was found and changed after it, for the second build:

1. **ModelPassTests, 11 tolerance failures.** The three encoders with real matrix products (vision,
   tactile, motor; 1,160, 32 and 26 rows) came out 2e-3 to 2.7e-3 relative off the fixtures' pooled
   vectors, the batch-1 MLP encoders within 1e-5, the transformer's latents 0.05 to 0.08 off. The same
   arithmetic run in float32 numpy (`Fixtures/prediction_heads/check_model_pass_numpy.py` with
   float32 instead of float64, November 4, 2026) is within 1.3e-6 on every token and 0.023 on every
   pooled vector, so the line of 1e-3 is reachable and stays. The cause is mlx's Metal matmul: on a
   GPU with tensor units (the M5) it routes float32 products through them when `MLX_ENABLE_TF32` is
   1, its default, at about 10 mantissa bits (mlx/utils.h, mlx/backend/metal/matmul.cpp). Fix:
   `MLXFloat32Precision.enforce()` (TransformerCore.swift) sets `MLX_ENABLE_TF32=0` before the first
   matmul; `ForwardModel.released`, `EncoderSet.released`, `SafetensorsLoader.load` and the test's
   `setUp` call it. The app must call it at launch once it wires the model. If the second build
   still shows the 2e-3, read the mlx checkout under Xcode's DerivedData `SourcePackages/checkouts`
   for the variable's name in mlx-swift 0.33.0, or set the variable in the scheme's environment.
2. **ModelPassTests, synthetic weights.** The vision token's root mean square on synthetic weights was
   0.176 against a guessed floor of 0.5; the test now checks the LayerNorm relation
   sqrt(v / (v + 1e-5)) against the measured pooled variance instead of guessing the variance.
3. **SceneKitEyeRendererTests, 8 failures, every pixel 0.** The renderer chose its attachment
   formats (.rgba8Unorm, .depth32Float); SceneKit builds its pipelines for the formats it reports
   (colorPixelFormat, depthPixelFormat, stencilPixelFormat), and a pass with other attachments draws
   nothing. The renderer now takes the reported formats, attaches the stencil plane when the depth
   format carries one, keeps SceneKit's own depth direction (clearing to match `usesReverseZ` as
   read), calls `prepare` before the pass, and converts the read-back to linear RGBA (BGRA order and
   sRGB decoding when the format says so). `lastDiagnostics` and the tests print the formats, the
   depth direction, the read-back's mean alpha (255 = the pass reached the texture) and the drawn
   count, so a third build, if needed, says which stage fails.
4. **Scheme name.** The workspace scheme is `UEMAKit`, not `UEMAKit-Package`; every file now says so.

## 3. Mismatches left

1. **The real stack is not wired.** `ReleasedStackComposer` (`Tests/UEMAIntegrationTests/OneModelStepTests.swift`,
   compiled only with `-D UEMA_REAL_STACK`) and the app's `RealStackSessionBuilder` need:
   - a type over `ForwardModel` that conforms to UEMACore's `ForwardModeling`: `ForwardModel.step` returns
     `UEMAModel.ForwardPass`, while `ForwardModeling.step` returns `ModelOutput` and also runs the motor
     controller and wellness (both in UEMAModel: `MotorController.swift`, `Wellness.swift`);
   - a conformance of the forward model to UEMAPlanner's `PlanTransitionProviding` (the execution
     transform, positions divided by 200, the encoders, transformer and heads per candidate;
     planner_loss.md section 3);
   - `ReleasedSenseReaders.make` (the six readers on one physics, now including `VisionReader` and
     `AudioReader`, and the shared `BodyStateTracker`), `ReleasedVestibularNormaliser.make`,
     `ReleasedHandDescent.make`, `CandidateExecution` (lem_model.py:5664-5727). None is on disk.
   The body-function lines of the composer were corrected to the adapters `ReleasedSceneCommandTests` runs.
2. **Audio sources.** `AudioReader` hears only what `AudioSourceProviding` gives it; nothing yet turns
   contacts into impacts (env.py:3523-3527), detects the mobile's chime or the kick panel's squeak, or
   runs the bee and dragonfly visitors. The caregiver's speech and the dragonfly's recorded timbre are
   not ported. The bee's wobble uses a different Gaussian stream from numpy's `default_rng(7)`.
3. **Weights.** `SafetensorsFileReader` (the F64 path; `MLX.loadArrays` refuses the 49 F64 tensors)
   has not compiled anywhere.
4. **Reviews.** WI-33 (`apple/project.yml`) and WI-35 (`UEMAModel/ForwardModel.swift` and its
   neighbours) have no adversarial review. `project.yml` was read in this pass: the app targets take
   `UEMAMind/` without `Tests/`, the test targets take `UEMAMind/Tests`, and every target depends on the
   package product `UEMAKit`, which holds every library module the app imports (only UEMACore today).
5. **The first-build record's path.** The plan names `apple/FIRST_BUILD_RECORD.md`, the specification
   `apple/records/first_build.md`; pick one.
6. **The reviewers' other open items.** The 192 `left` entries in
   `docs/apple_port/evidence/build_records_2026-11-04.json` not closed above (the handoff prints them).
