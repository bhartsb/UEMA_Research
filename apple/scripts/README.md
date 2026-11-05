# apple/scripts: the MuJoCo XCFramework build and the export scripts

This folder holds the scripts of the Apple port that run outside the Swift package. The
requirements are in `docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html` (section 4) and the work
items in `docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html` (sections 5 and 6).

| File | Work item | What it does |
| --- | --- | --- |
| `build_mujoco_xcframework.sh` | WI-18 | builds MuJoCo 3.3.0 from source into `apple/UEMAKit/Binaries/mujoco.xcframework` (this page) |
| `dump_weight_manifest.py` | WI-12 | writes every tensor's key, shape and dtype of `model.safetensors` to `apple/UEMAKit/Sources/UEMAModel/Resources/weight_manifest.json` |
| `export_skin_layout.py` | WI-13 | writes the skin points of the age-24 trunk_v1 body to `apple/UEMAKit/Sources/UEMAScene/Resources/skin_layout_age24_trunk_v1.json` |
| `export_eye_fixtures.py` | WI-14 | writes the five camera images and patch files of three recorded states under `Tests/UEMASensesTests/Fixtures/eyes/` |
| `export_scene_bundle.py` | WI-15 | writes the crib scene files and `bundle.json` under `Sources/UEMAScene/Resources/crib_age24_trunk_v1/` |
| `export_model_pass_fixtures.py` | WI-34 | writes the released model's acting pass on the WI-14 states to `Tests/UEMAModelTests/Fixtures/model_pass/<state>.safetensors` |

The five Python scripts belong to work items WI-12 to WI-15 and WI-34; each documents its inputs, operation
and outputs in its own docstring, has its tests beside it (`tests/`, `test_export_scene_bundle.py`)
and is accepted by the command its plan row names. The shell script is the one file of this folder
that needs a Mac: it has not run anywhere yet (the build machine has no Xcode; plan page, section 3,
fact 1), so WI-18 is in state BUILT (uncompiled) until the founder runs it. Its shell functions have a
unit test beside the Python tests, `tests/test_build_mujoco_xcframework.sh`, which runs here without
Xcode or network (last section of this page).

## build_mujoco_xcframework.sh

### What it produces

One XCFramework (Apple's bundle of one compiled library per platform) holding MuJoCo, the physics
program written in C, as three static libraries plus its 14 public headers and a module map:

| Slice | For | Built with |
| --- | --- | --- |
| `macos-arm64` | macOS on Apple silicon | MuJoCo's own CMake, `-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0` |
| `ios-arm64` | iPhone and iPad devices | the ios-cmake toolchain, `-DPLATFORM=OS64` |
| `ios-arm64-simulator` | the iOS Simulator on an Apple silicon Mac | the ios-cmake toolchain, `-DPLATFORM=SIMULATORARM64` |

The MuJoCo version is the Python pin, read from `requirements.txt` (line 9: `mujoco==3.3.0`), so the
Swift side steps the same physics code as the trainer (specification REQ-B-1: after 1,000 physics
steps every `qpos` entry within 1e-6 of Python's). WI-17's `Package.swift` names the result as the
binary target `mujoco`; WI-19's `CMuJoCo` module calls it.

REQ-B-1 asks for "simulate, tests and renderers off". Simulate and tests are off by flag. The classic
OpenGL renderer (`src/render`, added to the library by `CMakeLists.txt` line 94) has no off switch at
tag 3.3.0, so its source is compiled into every slice; it loads OpenGL at run time, through its bundled
glad loader, only when an `mjr_` function is called, and UEMAKit calls none (WI-19), which is the
condition the requirement itself quotes: the library "can be used on systems without OpenGL support,
as long as mjr_ functions are not called".

### Run it (on a Mac)

```
bash apple/scripts/build_mujoco_xcframework.sh --dry-run    # prints the command plan only
bash apple/scripts/build_mujoco_xcframework.sh              # clones, builds, verifies
```

Needs: macOS, Xcode with the macOS 14 and iOS 17 SDKs, git, CMake 3.16 or newer (MuJoCo's
`CMakeLists.txt` line 15 at tag 3.3.0), network access to github.com. Exit code 0 means every check
below passed; 1 means a check or a build step failed (the last log line names the line of the script
and the command that failed); 2 is a usage error. `--help` lists the options (`--slices`, `--build-dir`,
`--output`, `--records-dir`, `--jobs`, `--mujoco-tag`, `--ios-deployment-target`,
`--macos-deployment-target`, `--clean`, `--dry-run`) and the one environment variable,
`UEMA_MUJOCO_CXX_FLAGS` (extra compiler flags appended after MuJoCo's own through the CMake
variable `MUJOCO_CXX_FLAGS`, `CMakeLists.txt` line 108; several flags are separated by `;`).

### What it does, in sequence

1. Checks the host: `uname -s` is Darwin; git, cmake, xcodebuild, xcrun and PlistBuddy are present;
   CMake is 3.16 or newer; `xcrun --sdk <macosx|iphoneos|iphonesimulator> --show-sdk-path` finds
   each SDK a requested slice needs.
2. Clones `google-deepmind/mujoco` at tag 3.3.0 and `leetal/ios-cmake` at tag 4.6.1 (its latest
   release on 2026-11-04, published 2026-09-24) into `apple/build/mujoco/`, then compares each
   clone's HEAD with the commit the tag pointed to on 2026-11-04 (MuJoCo
   `8de9b4e4e373caf01035c6c13efd363347d89032`, ios-cmake `b36d5bce6f9a2d11d9f144bd5882ba31cfdca420`).
3. Patches two lines of the MuJoCo checkout (next section) and saves `git diff` as
   `apple/records/mujoco_xcframework/mujoco_source_patch.diff`.
4. For each slice: `cmake` configure with the flags below; `cmake --build --target mujoco` (only
   the library, not the plugins, samples, simulate or tests); `xcrun libtool -static` merges every
   static archive the build wrote (MuJoCo's own plus the five libraries the `mujoco` target links,
   `CMakeLists.txt` 117-124: ccd, lodepng, qhullstatic_r, tinyobjloader, tinyxml2; SdfLib is fetched
   at configure time, `cmake/MujocoDependencies.cmake` 190-204, but linked only by the sdf plugin,
   `plugin/sdf/CMakeLists.txt` line 40, which `--target mujoco` never builds, and MarchingCubeCpp is
   header-only, `cmake/MujocoDependencies.cmake` 123) into one `libmujoco.a`; then
   `xcrun lipo -info` must name arm64 and `xcrun nm -g` must list `_mj_step` and
   `_mj_versionString` (the API reference: `mj_step` "Advance simulation, use control callback to
   obtain external force and control"; `mj_versionString` "Return the current version of MuJoCo as
   a null-terminated string").
5. Copies `include/mujoco/*.h` (14 files at tag 3.3.0, `CMakeLists.txt` lines 55-70) into a
   staging folder with a module map (`module mujoco`, umbrella header `mujoco/mujoco.h`), so a
   consumer writes `#include <mujoco/mujoco.h>` or `import mujoco`.
6. `xcodebuild -create-xcframework -library <slice>/libmujoco.a -headers <staging>` for each slice,
   `-output apple/UEMAKit/Binaries/mujoco.xcframework` (the output is deleted first; Apple's page
   on binary framework bundles names `-library` with `-headers` for static libraries built by
   "an alternative build system"). Then PlistBuddy counts the `LibraryIdentifier` entries of the
   XCFramework's `Info.plist`: the count must equal the number of slices built, 3 for a full run.
7. Writes `apple/records/mujoco_xcframework/build_record.json` (host, tool versions, tags, commits,
   flags, per-slice bytes, lipo and nm results, library identifiers) beside the logs.

### The flags, and why each one

Every slice gets these; the line numbers are in the 3.3.0 checkout (read 2026-11-04).

| Flag | Why |
| --- | --- |
| `-DMUJOCO_BUILD_SIMULATE=OFF` | no simulate library; it links Cocoa and GLFW, which iOS has not (`CMakeLists.txt` 42, 176-178) |
| `-DMUJOCO_BUILD_TESTS=OFF` | no tests, and no abseil, googletest or benchmark fetch (`CMakeLists.txt` 184-190; `cmake/MujocoDependencies.cmake` 239-328) |
| `-DMUJOCO_BUILD_EXAMPLES=OFF` | no sample programs and no `enable_language(OBJC)` (`CMakeLists.txt` 41, 46-49, 180-182) |
| `-DMUJOCO_TEST_PYTHON_UTIL=OFF` | with it on, Eigen3 is fetched from gitlab.com even when tests are off (`cmake/MujocoDependencies.cmake` 330-353) |
| `-DMUJOCO_ENABLE_AVX=OFF`, `-DMUJOCO_ENABLE_AVX_INTRINSICS=OFF` | no `-mavx` check and no `mjUSEPLATFORMSIMD` define, so the x86 AVX code of `engine_util_sparse_avx.h` is not compiled; AVX is a build option (`cmake/MujocoOptions.cmake` 69-79, `CMakeLists.txt` 99-101); the radcli14/swift-mujoco fork leaves the same define off |
| `-DMUJOCO_USE_FILAMENT=OFF` | required by WI-18; at tag 3.3.0 the option does not exist, so CMake prints "Manually-specified variables were not used by the project: MUJOCO_USE_FILAMENT" and the exit code stays 0 |
| `-DMUJOCO_BUILD_MACOS_FRAMEWORKS=OFF` | the dylib framework branch stays off (`cmake/MujocoOptions.cmake` 81; `CMakeLists.txt` 132-165) |
| `-DCMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE=OFF` | `cmake/MujocoOptions.cmake` 107-109 turns link-time optimization on for every non-Debug build and a `-D` of the generic variable cannot stop it; the per-configuration property "overrides the generic property for the named configuration" (CMake documentation), so the archives hold machine code that lipo, nm and a later Xcode can read |
| `-DCMAKE_BUILD_TYPE=Release`, `-G "Unix Makefiles"` | the build type MuJoCo would default to, stated; one generator for all slices (ios-cmake's README names Makefiles as supported) |
| iOS slices: `-DDEPLOYMENT_TARGET=17.0 -DENABLE_BITCODE=OFF` | the specification's iOS floor (section 1: "iOS 17 and macOS 14 or later"); bitcode is "dead since Xcode 14" (ios-cmake README) |
| macOS slice: `-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0` | the specification's macOS floor (section 1). MuJoCo's own objects carry `-mmacosx-version-min=11` (`cmake/MujocoMacOS.cmake` 25); without this variable the five fetched dependencies would carry the SDK's own default, and an app linked for macOS 14 would get one ld warning "object file was built for newer 'macOS' version" per dependency object |

### The two source patches

MuJoCo's CMake cannot build the library as a static archive or for iOS without these; the script
applies them once (a second run finds them in place) and refuses a checkout where the lines are
not exactly where tag 3.3.0 has them.

1. `CMakeLists.txt` line 79: `add_library(mujoco SHARED ${MUJOCO_RESOURCE_FILES})` becomes
   `add_library(mujoco STATIC ...)`. The library type is written into the file, so no `-D`
   changes it, and iOS takes static libraries or `.framework` bundles, not plain dylibs (Apple:
   "Dynamic linking on iOS ... requires the XCFramework to contain .framework bundles"). The
   private dependencies are static already (`cmake/MujocoDependencies.cmake` 81-91).
2. `cmake/MujocoMacOS.cmake` lines 15 and 37: `if(APPLE)` becomes `if(APPLE AND NOT IOS)`. Line 15
   opens the block that sets `-mmacosx-version-min=11` and two macOS availability `-Werror` flags
   (lines 25-26), which go to the `mujoco` target (`CMakeLists.txt` 106) and to each fetched
   dependency (`cmake/MujocoDependencies.cmake` 107, 149, 168, 205, 225); line 37 is the same test
   inside `enforce_mujoco_macosx_min_version`, called only by simulate and the samples. ios-cmake
   sets `APPLE` and `IOS` on (`ios.toolchain.cmake` 849-861) and compiles with
   `-target arm64-apple-ios17.0`, which that macOS flag contradicts. CMake sets `IOS` "when the
   target system (CMAKE_SYSTEM_NAME) is iOS" (since 3.14), so the macOS slice keeps MuJoCo's own
   flags.

### The checks the script makes, against the acceptance line

WI-18's acceptance: the script exits 0; `xcrun lipo -info` names arm64 per slice; the
XCFramework's `Info.plist` lists 3 libraries. The script performs all three itself and exits 1 on
any miss, so a printed `RESULT: exit 0; every check passed` after the line
`XCFramework libraries (3): ios-arm64 ios-arm64-simulator macos-arm64` is the pass. The nm check
and the `-flto` count (expected 0 lines in `CMakeFiles/mujoco.dir/flags.make`) are extra evidence
written to the record.

### Unverified until the first run on a Mac

- No official MuJoCo build for iOS exists (the programming guide names Windows, Linux and macOS).
  The two iOS slices rest on reported precedents: the App Store app iMuJoCo, and the
  radcli14/swift-mujoco fork's README (MuJoCo 3.9.0 for iOS device and simulator, no build log).
- Whether the 3.3.0 sources compile under the iOS 17 SDK with MuJoCo's `-Werror -Wall -Wpedantic`
  set (`cmake/MujocoOptions.cmake` 87-98). The slice logs answer it.
- Whether SdfLib and the other fetched dependencies configure under the ios-cmake toolchain. The
  configure logs answer it.
- Tool versions on the founder's Mac (plan page, section 8, step 1).

### When a step fails

| Last log line says | Cause | Next step |
| --- | --- | --- |
| `this script runs on macOS` | run on another system | run it on the Mac; `--dry-run` works anywhere |
| `CMake x.y.z is older than 3.16` | old CMake | install CMake 3.16 or newer |
| `SDK 'iphoneos' not found by xcrun` | Xcode without the iOS SDK | install the iOS 17 SDK in Xcode, or `--slices macos-arm64` for the Mac only |
| `... is at <commit>; tag 3.3.0 was commit 8de9b4e4...` | the tag moved or the clone is another checkout | `--clean`, then rerun |
| `holds '...' on N line(s), expected M` | the checkout is not tag 3.3.0 as read on 2026-11-04 | check `--mujoco-tag`; a new tag needs its lines re-read |
| `FAILED at line N of ...: <command>` | that command exited non-zero (a configure, a build, the merge or `xcodebuild`) | read the log named on the same line; a configure failure of a fetched dependency under the ios-cmake toolchain is a first-build finding |
| an error line ending in `[-Werror,-W...]` in `<slice>_build.log` | a warning the iOS SDK or a newer clang raises that MuJoCo's `-Werror` turns into an error | rerun with `UEMA_MUJOCO_CXX_FLAGS=-Wno-error`, record the warning in `apple/records/first_build.md` |
| `nm lists _mj_step 0 time(s)` | the merge missed MuJoCo's own archive | read `<slice>_build.log` for the list of merged archives |
| `Info.plist lists N libraries, expected 3` | a slice is missing from the bundle | read `logs/xcframework.log` |

Other failures are first-build findings for `apple/records/first_build.md` (specification REQ-B-18,
item 4).

### The checks run here, on Windows, on 2026-11-04

No Mac was available, so three checks ran, from `D:/UEMA_Research` in Git Bash:

- `bash -n apple/scripts/build_mujoco_xcframework.sh` printed nothing and exited 0.
- `bash apple/scripts/build_mujoco_xcframework.sh --dry-run` exited 0 and printed the full command
  plan: the two `git clone` lines, the two patches, and for each of the 3 slices one `cmake`
  configure line with the flags above, one `cmake --build ... --target mujoco` line, the merge and
  the verify lines, then the `xcodebuild -create-xcframework` line with 3 `-library` and 3
  `-headers` pairs.
- `bash apple/scripts/tests/test_build_mujoco_xcframework.sh` (the unit test; its header lists the
  8 groups of checks) printed `PASS: 82 checks, 0 failed` and exited 0. It sources the script (the
  script runs `main` only when executed) and checks `version_at_least`, `needed_sdks`, `json_str`,
  `quoted`, `abs_path`, `valid_slice`, `slice_flags`, `parse_args`, `resolve_mujoco_tag`,
  `check_host` (its dry-run branch) and the `--dry-run` and `--help` runs. `patch_line` is checked
  on files written by the test with the exact lines of tag 3.3.0 (`add_library(mujoco SHARED
  ${MUJOCO_RESOURCE_FILES})` once; `if(APPLE)` twice): 1 and 2 lines change, every other line is
  byte-identical (`cmp`), a second run reports "already patched", and a wrong expected count exits 1
  and leaves the file unchanged. The line numbers cited on this page and in the script header were
  compared the same day with the files of the 3.3.0 tag and of ios-cmake 4.6.1 fetched from GitHub.

## Writing rules for files in this folder

The rules of `apple/README.md` apply: no duration estimates, every number with its unit and
source, the canonical names (model step, command, wellness channel; vision, tactile, audio,
proprioception, motor, vestibular, intermind). The banned-words check on this page and on the shell
script is `grep -c -i -E "$(paste -sd '|' docs/apple_port/banned_words.txt)" <file>`, which must
print 0 for each (the `grep -i -f docs/apple_port/banned_words.txt` form of `apple/README.md`
aborts in this machine's GNU grep 3.0 with "Aborted", exit 134, on any file; the alternation
form matches the same seven lines).
