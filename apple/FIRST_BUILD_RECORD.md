# First Mac build record

Mac: MacBook Air (Mac17,3, Apple M5), macOS 26.7.2; Xcode 27.2 (27B52); Swift 6.4.2 (swiftlang-6.4.2.1.9); cmake 4.4.5 (Homebrew)
Branch and commit: claude/awesome-pasteur-w4d2f4 at 12037fd Apple port: consistency pass, Linux check and first-build checklist
(the build ran on 12037fd plus the four fixes listed under "Changes made during this build", committed with this record)

## Before step 3
- `feature/apple-port-2026-11-04` did not yet hold `apple/FIRST_BUILD_CHECKLIST.md`; cloned `claude/awesome-pasteur-w4d2f4` instead.
- Xcode needed `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`, `sudo xcodebuild -license accept`
  and `sudo xcodebuild -runFirstLaunch` (without it, `xcodebuild -create-xcframework` failed loading
  IDESimulatorFoundation: CoreSimulator.framework missing). Metal toolchain downloaded with
  `xcodebuild -downloadComponent MetalToolchain` before step 5.
- `fetch_weights.py` needs `huggingface_hub`; Homebrew's python3.13 is externally managed, so it ran from a venv
  (`python3.13 -m venv .venv && .venv/bin/pip install huggingface_hub`, huggingface_hub 2.4.0). All six files matched MANIFEST.json.

## Step 3, MuJoCo XCFramework
Command: bash apple/scripts/build_mujoco_xcframework.sh --clean
Result: passed (third attempt)      Time: 11:18 PM Pacific, 2026-11-04
Slices: ios-arm64, ios-arm64-simulator, macos-arm64; each libmujoco.a is arm64 (lipo -info).
First errors (earlier attempts), both from CMake 4.x:
1. qhull's `cmake_minimum_required` < 3.5: "Compatibility with CMake < 3.5 has been removed from CMake."
   Fix: `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` added to COMMON_CMAKE_FLAGS.
2. With mujoco patched to STATIC: `install(EXPORT "mujoco" ...) includes target "mujoco" which requires target
   "lodepng" that is not in any export set.` (CMake Generate step failed).
   Fix: a third source patch, `if(NOT (APPLE AND MUJOCO_BUILD_MACOS_FRAMEWORKS))` -> `if(FALSE)` on MuJoCo's
   install block; the script merges the archives with libtool and never installs.

## Step 4, swift package resolve
Result: passed      Time: 11:19 PM Pacific, 2026-11-04      mlx-swift version in Package.resolved: 0.33.0 (swift-numerics 1.1.1)

## Step 5, package build and test
Scheme: the workspace lists only `UEMAKit` (no `UEMAKit-Package`); used `-scheme UEMAKit`. The guide, the
checklist and the Package.swift header comment name `UEMAKit-Package`.
Build result: passed (after fix 3)      Time: 11:21 PM Pacific, 2026-11-04
First error (before fix 3):
  Sources/UEMAModel/SafetensorsLoader.swift:904:61: error: cannot find type 'VerifyUpdate' in scope
  In mlx-swift 0.33.0 it is nested: `Module.VerifyUpdate` (Source/MLXNN/Module.swift:435).

Test result: Executed 372 tests, with 20 failures (1 skipped)      Time: 11:23 PM Pacific, 2026-11-04
First error (before fix 4): UEMAPhysicsTests, UEMASensesTests and UEMATokenizerTests failed to link: undefined
  std::__1 symbols (basic_iostream, typeinfo for std::exception, ...) from the C++ libmujoco.a.

| Target | Executed | Failures | Expected count |
|---|---|---|---|
| CMuJoCoTests | 12 | 0 | 12 |
| UEMACoreTests | 31 | 0 | 31 |
| UEMAPhysicsTests | 24 | 0 | 24 |
| UEMASceneTests | 27 | 0 | 27 |
| UEMASensesTests | 103 | 8 | 103 |
| UEMATokenizerTests | 30 | 0 | 30 |
| UEMACommandTests | 35 | 0 | 35 |
| UEMAModelTests | 54 | 12 | 54 |
| UEMAPlannerTests | 36 | 0 | 36 |
| UEMAIntegrationTests | 10 (1 skipped) | 0 | 10 |

Failures (assertion counts; see apple/records/uema_step5_test.log):
- UEMAModelTests.ModelPassTests.testFixturePassesMatchWithinTheDeclaredTolerance (11): max abs error 1.40e-3 to
  2.89e-3 against tolerance 1e-3, tokens and predictions for vision, tactile and motor at steps 492800, 493674, 493980
  (ModelPassTests.swift:354 and :361). Motor token error is 0.002887683 at all three steps.
- UEMAModelTests.ModelPassTests.testOneModelStepOnSyntheticWeightsHasTheShapes (1): ModelPassTests.swift:208,
  "0.17587008" is not greater than "0.5" - vision.
- UEMASensesTests.SceneKitEyeRendererTests (8): every rendered pixel reads 0: testEmissiveBoxAhead (:77),
  testImageOrientation (:93, :98), testHeadlightShadingMatchesFixedFunctionValue (:111-113, expected 64 +/- 6),
  testNearBoxHidesFarBox (:130, :136).
- Skipped: UEMAIntegrationTests.OneModelStepTests.testReleasedStackRunsTwentyModelSteps (compiled without UEMA_REAL_STACK, by design).
No OpenGL linker errors.

## Step 6, app build
Result: passed (xcodegen generate, then xcodebuild -scheme UEMAMind-macOS, BUILD SUCCEEDED)      Time: 11:24 PM Pacific, 2026-11-04
First errors (if any): none

## Changes made during this build
1. apple/scripts/build_mujoco_xcframework.sh: `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` in COMMON_CMAKE_FLAGS (CMake 4).
2. apple/scripts/build_mujoco_xcframework.sh: patch MuJoCo's install block off (STATIC + install(EXPORT) under CMake 4).
3. apple/UEMAKit/Sources/UEMAModel/SafetensorsLoader.swift:904: `VerifyUpdate` -> `Module.VerifyUpdate`.
4. apple/UEMAKit/Package.swift: CMuJoCo `linkerSettings: [.linkedLibrary("c++")]`.

## For the next session
- Fix the 2 ModelPassTests failures (tolerance vs MLX numerics, or a real mismatch) and the 4 SceneKitEyeRenderer
  tests (black frames on macOS: offscreen render setup).
- Change the scheme name `UEMAKit-Package` to `UEMAKit` in the guide, the checklist and the Package.swift header.

# Second Mac build (after the fixes in "Apple port: fixes after the first Mac build")

Same Mac and toolchain. Branch and commit: feature/apple-port-2026-11-04 at 6130a55 (the session branch's four
commits rebased onto 789d22d; trees otherwise identical). Logs: apple/records/uema_build2_step5_*.log and
apple/records/uema_build2_step6_app.log. Steps 3 and 4 not repeated (no change to the script or Package.resolved).

## Step 5, package build and test
Build result: passed      Test result: Executed 372 tests, with 0 failures (1 skipped)      Time: 1:06 AM Pacific, 2026-11-05
Per target: CMuJoCoTests 12, UEMACoreTests 31, UEMAPhysicsTests 24, UEMASceneTests 27, UEMASensesTests 103,
UEMATokenizerTests 30, UEMACommandTests 35, UEMAModelTests 54, UEMAPlannerTests 36, UEMAIntegrationTests 10
(skipped: testReleasedStackRunsTwentyModelSteps, compiled without UEMA_REAL_STACK, by design). All match the
expected counts. The ModelPassTests (MLX_ENABLE_TF32=0) and SceneKitEyeRenderer (reported pixel formats) fixes hold.

## Step 6, app build
Result: passed (xcodegen generate, then xcodebuild -scheme UEMAMind-macOS, BUILD SUCCEEDED)      Time: 1:06 AM Pacific, 2026-11-05
