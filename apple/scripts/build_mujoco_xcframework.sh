#!/usr/bin/env bash
# =============================================================================
# apple/scripts/build_mujoco_xcframework.sh
#
# Builds MuJoCo (the physics program written in C) from source at the tag the
# Python side pins (requirements.txt line 9: mujoco==3.3.0) into one static
# XCFramework with three slices, for the Swift package apple/UEMAKit
# (specification REQ-B-1; implementation plan work item WI-18):
#
#     macos-arm64            macOS on Apple silicon
#     ios-arm64              iPhone and iPad devices
#     ios-arm64-simulator    the iOS Simulator on an Apple silicon Mac
#
# INPUT
#   - network access to github.com: two git clones (MuJoCo, ios-cmake) and the
#     dependencies MuJoCo's CMake fetches itself with FetchContent (lodepng,
#     MarchingCubeCpp, qhull, tinyxml2, tinyobjloader, SdfLib, ccd; the pinned
#     commits are in cmake/MujocoDependencies.cmake of the 3.3.0 checkout)
#   - a Mac with Xcode (xcodebuild, xcrun, lipo, libtool, nm, PlistBuddy), git
#     and CMake 3.16 or newer (CMakeLists.txt line 15 at tag 3.3.0:
#     cmake_minimum_required(VERSION 3.16)); a C++17 compiler comes with Xcode
#     (cmake/MujocoOptions.cmake line 18: CMAKE_CXX_STANDARD 17)
#   - requirements.txt in the repository root, read for the mujoco== pin
#
# OPERATION (one function per step, run in this sequence by main)
#   1. check_host         host is macOS; tools present; CMake >= 3.16; the SDKs
#                         the requested slices need are found by xcrun
#   2. fetch_sources      git clone google-deepmind/mujoco at tag 3.3.0 (commit
#                         8de9b4e4e373caf01035c6c13efd363347d89032, read from the
#                         GitHub tag reference on 2026-11-04) and leetal/ios-cmake
#                         at tag 4.6.1 (commit b36d5bce6f9a2d11d9f144bd5882ba31cfdca420,
#                         the latest release on 2026-11-04, published 2026-09-24);
#                         every clone's HEAD commit is compared with the pin
#   3. patch_sources      two literal one-line edits of the MuJoCo checkout (PATCHES)
#   4. per slice          cmake configure with FLAGS, cmake --build --target mujoco,
#                         merge every static archive the build wrote into one
#                         libmujoco.a with libtool -static, then verify with
#                         xcrun lipo -info (must name arm64) and xcrun nm (must
#                         list _mj_step and _mj_versionString)
#   5. stage_headers      copy include/mujoco/*.h and write a module map
#   6. create_xcframework xcodebuild -create-xcframework, then count the
#                         LibraryIdentifier entries of its Info.plist
#   7. write_record       a JSON record beside the logs
#
# OUTPUT
#   apple/UEMAKit/Binaries/mujoco.xcframework      WI-17's binaryTarget "mujoco"
#   apple/records/mujoco_xcframework/build_record.json
#   apple/records/mujoco_xcframework/mujoco_source_patch.diff
#   apple/records/mujoco_xcframework/logs/<slice>_configure.log, <slice>_build.log,
#                                         <slice>_cmake_args.txt, xcframework.log
#   apple/build/mujoco/                            clones, build trees, staging
#   Exit code 0 means every check above passed; 1 means a check or a build step
#   failed (the last log line names it); 2 means a usage error.
#
# FLAGS passed to every slice, each with the line of the 3.3.0 checkout it acts on
#   -DCMAKE_BUILD_TYPE=Release          cmake/MujocoOptions.cmake 24-31 would default
#                                       to Release; passed so the log states it
#   -DMUJOCO_BUILD_EXAMPLES=OFF         CMakeLists.txt 41, 46-49, 180-182: no sample
#                                       programs, and no enable_language(OBJC)
#   -DMUJOCO_BUILD_SIMULATE=OFF         CMakeLists.txt 42, 176-178: no simulate
#                                       library (it links Cocoa and GLFW, see
#                                       simulate/CMakeLists.txt; not an iOS target)
#   -DMUJOCO_BUILD_TESTS=OFF            CMakeLists.txt 43, 184-190 and
#                                       cmake/MujocoDependencies.cmake 239-328: no tests,
#                                       and no abseil, googletest or benchmark fetch
#   -DMUJOCO_TEST_PYTHON_UTIL=OFF       cmake/MujocoDependencies.cmake 330-353: with it
#                                       ON (CMakeLists.txt 44) Eigen3 is fetched from
#                                       gitlab.com even when tests are off
#   -DMUJOCO_ENABLE_AVX=OFF             cmake/MujocoOptions.cmake 69, 74-79: skips the
#                                       -mavx compile check (cmake/CheckAvxSupport.cmake)
#   -DMUJOCO_ENABLE_AVX_INTRINSICS=OFF  CMakeLists.txt 99-101: mjUSEPLATFORMSIMD is not
#                                       defined, so engine_util_sparse_avx.h's x86 AVX
#                                       code is not compiled (the radcli14/swift-mujoco
#                                       Package.swift leaves the same define off)
#   -DMUJOCO_USE_FILAMENT=OFF           required by WI-18; at tag 3.3.0 no such option
#                                       exists in CMakeLists.txt or cmake/MujocoOptions.cmake
#                                       (read 2026-11-04), so CMake prints the warning
#                                       "Manually-specified variables were not used by
#                                       the project: MUJOCO_USE_FILAMENT"; the exit code
#                                       stays 0
#   -DMUJOCO_BUILD_MACOS_FRAMEWORKS=OFF cmake/MujocoOptions.cmake 81 and CMakeLists.txt
#                                       132-165: the dylib framework branch stays off
#   -DCMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE=OFF
#                                       cmake/MujocoOptions.cmake 107-109 turns the
#                                       generic CMAKE_INTERPROCEDURAL_OPTIMIZATION on for
#                                       every non-Debug build, and a -D of the generic
#                                       variable cannot stop it (the if() tests NOT of
#                                       its value); the per-configuration property
#                                       "overrides the generic property for the named
#                                       configuration" (CMake documentation,
#                                       prop_tgt/INTERPROCEDURAL_OPTIMIZATION_CONFIG),
#                                       so the archives hold machine code, not LLVM
#                                       bitcode, and lipo, nm and a later Xcode read them
#   -G "Unix Makefiles"                 one generator for all three slices; ios-cmake's
#                                       README names Makefiles as supported
#   macos-arm64 only:   -DCMAKE_OSX_ARCHITECTURES=arm64
#                       -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 (the specification's macOS
#                       floor, section 1: "iOS 17 and macOS 14 or later"; MuJoCo's own
#                       objects carry -mmacosx-version-min=11 from
#                       cmake/MujocoMacOS.cmake 25, and without this variable the five
#                       fetched dependencies would carry the SDK's own default, so an
#                       app linked for macOS 14 would get one ld warning "object file
#                       was built for newer 'macOS' version" per dependency object)
#   ios-arm64 only:     -DCMAKE_TOOLCHAIN_FILE=<ios-cmake>/ios.toolchain.cmake
#                       -DPLATFORM=OS64 -DDEPLOYMENT_TARGET=17.0 -DENABLE_BITCODE=OFF
#   ios-arm64-simulator: the same with -DPLATFORM=SIMULATORARM64
#                       (ios-cmake README: "OS64 | iOS devices | arm64",
#                       "SIMULATORARM64 | iOS Simulator on Apple Silicon | arm64";
#                       17.0 is the specification's iOS floor, section 1: "iOS 17 and macOS 14 or later")
#
# PATCHES (applied once to the checkout; the diff is saved beside the record)
#   CMakeLists.txt line 79:  add_library(mujoco SHARED ${MUJOCO_RESOURCE_FILES})
#                        ->  add_library(mujoco STATIC ${MUJOCO_RESOURCE_FILES})
#       The library type is written into the file, so no -D can change it, and
#       iOS apps take static libraries or .framework bundles, not plain dylibs
#       (Apple, "Creating a multiplatform binary framework bundle": "Dynamic
#       linking on iOS ... requires the XCFramework to contain .framework
#       bundles"). The private dependencies (CMakeLists.txt 117-124) are static
#       already (cmake/MujocoDependencies.cmake 81-91), so the merge step joins
#       them into the one archive a consumer links.
#   cmake/MujocoMacOS.cmake lines 15 and 37:  if(APPLE)  ->  if(APPLE AND NOT IOS)
#       Line 15 opens the block that sets -mmacosx-version-min=11 and two macOS
#       availability -Werror flags (lines 25-26), given to the mujoco target
#       (CMakeLists.txt 106) and to each fetched dependency
#       (cmake/MujocoDependencies.cmake 107, 149, 168, 205, 225); line 37 is the
#       same test inside enforce_mujoco_macosx_min_version, which only simulate
#       and the samples call. ios-cmake sets APPLE ON and IOS ON
#       (ios.toolchain.cmake 4.6.1, lines 849-861) and compiles with
#       -target arm64-apple-ios17.0, which that macOS flag contradicts. CMake
#       defines IOS "when the target system (CMAKE_SYSTEM_NAME) is iOS" (CMake
#       documentation, variable/IOS, since 3.14), so the macOS slice is unchanged.
#
# WHAT THE SLICES LEAVE OUT, AND WHY IT IS SAFE
#   Only the target "mujoco" is built. The plugin libraries (plugin/elasticity,
#   actuator, sensor, sdf; each add_library(... SHARED)) are not built: the crib
#   scene's XML and the bridge's environment code contain no <plugin> or
#   <extension> element (grep of Source/Python/mimo_bridge/env.py, mimo_env.py and
#   third_party/MIMo/**/*.xml on 2026-11-04 returned no line). So the archives the
#   merge step finds are libmujoco.a and the five libraries the mujoco target
#   links (CMakeLists.txt 117-124: ccd, lodepng, qhullstatic_r, tinyobjloader,
#   tinyxml2); SdfLib is fetched at configure time (cmake/MujocoDependencies.cmake
#   190-204) but linked only by the sdf plugin (plugin/sdf/CMakeLists.txt line 40),
#   so cmake --build --target mujoco never compiles it, and MarchingCubeCpp is
#   header-only (cmake/MujocoDependencies.cmake 123). The classic OpenGL
#   renderer (src/render, compiled into the library by CMakeLists.txt line 94)
#   loads OpenGL at run time through its bundled glad loader and only when an
#   mjr_ function is called; UEMAKit calls none (WI-19).
#
# UNVERIFIED UNTIL THIS SCRIPT RUNS ON A MAC
#   No official MuJoCo build for iOS exists (the programming guide names Windows,
#   Linux and macOS). The two iOS slices rest on reported precedents only: the
#   App Store app iMuJoCo and the radcli14/swift-mujoco fork (MuJoCo 3.9.0 for
#   iOS device and simulator, a README statement with no build log). Whether the
#   3.3.0 sources compile under the iOS SDK with MuJoCo's -Werror set
#   (cmake/MujocoOptions.cmake 87-98) is answered by the slice logs; a failure on
#   a warning is retried with UEMA_MUJOCO_CXX_FLAGS=-Wno-error (see usage).
#
# SOURCES (read 2026-11-04)
#   https://raw.githubusercontent.com/google-deepmind/mujoco/3.3.0/CMakeLists.txt
#   https://raw.githubusercontent.com/google-deepmind/mujoco/3.3.0/cmake/MujocoOptions.cmake
#   https://raw.githubusercontent.com/google-deepmind/mujoco/3.3.0/cmake/MujocoMacOS.cmake
#   https://raw.githubusercontent.com/google-deepmind/mujoco/3.3.0/cmake/MujocoDependencies.cmake
#   https://raw.githubusercontent.com/google-deepmind/mujoco/3.3.0/include/mujoco/mjexport.h
#   https://raw.githubusercontent.com/google-deepmind/mujoco/3.3.0/dist/module.modulemap
#   https://mujoco.readthedocs.io/en/stable/programming/index.html ("Building from source")
#   https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html
#       (mj_step: "Advance simulation, use control callback to obtain external
#       force and control"; mj_versionString: "Return the current version of
#       MuJoCo as a null-terminated string"; both MJAPI in include/mujoco/mujoco.h
#       at 3.3.0, mj_step on line 131 and mj_versionString on line 546, and the
#       two symbols the nm check looks for)
#   https://github.com/leetal/ios-cmake (README, 4.6.1 toolchain file)
#   https://developer.apple.com/documentation/xcode/creating-a-multi-platform-binary-framework-bundle
#   https://cmake.org/cmake/help/latest/prop_tgt/INTERPROCEDURAL_OPTIMIZATION_CONFIG.html
#   https://cmake.org/cmake/help/latest/variable/IOS.html
#   research1.json entries mujoco-ios and mujoco-macos (scratchpad), requirements.txt:9
#
# FIRST BUILD ON A MAC (the founder's checklist; the plan's section 8, step 2)
#   [ ] 1. macOS 14 or later, Xcode with the macOS 14 and iOS 17 SDKs, CMake >= 3.16,
#          git, network access; check: xcodebuild -version; cmake --version;
#          xcrun --sdk iphoneos --show-sdk-version
#   [ ] 2. git checkout feature/apple-port-2026-11-04 in a clone of the repository
#   [ ] 3. bash apple/scripts/build_mujoco_xcframework.sh --dry-run
#          (prints the command plan; nothing is cloned or built); the unit test
#          of the shell functions is bash apple/scripts/tests/test_build_mujoco_xcframework.sh
#          (no network, no Xcode; pass line: its last line is "PASS: N checks, 0 failed")
#   [ ] 4. bash apple/scripts/build_mujoco_xcframework.sh
#          pass line: exit 0; the summary names arm64 for each of the 3 slices;
#          "XCFramework libraries (3): ios-arm64 ios-arm64-simulator macos-arm64"
#   [ ] 5. if a slice fails: read apple/records/mujoco_xcframework/logs/<slice>_build.log;
#          for an error line ending in "[-Werror,-W...]", rerun step 4 with
#          UEMA_MUJOCO_CXX_FLAGS=-Wno-error; any other error is a first-build
#          finding for apple/records/first_build.md (REQ-B-18, item 4)
#   [ ] 6. keep apple/records/mujoco_xcframework/ (REQ-B-18, item 2: "keep the
#          CMake log"); the build directory and the XCFramework are build output
#   [ ] 7. cd apple/UEMAKit && swift package resolve (WI-17), then WI-19's test:
#          mj_versionString() equals "3.3.0"
# =============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MUJOCO_REPO="https://github.com/google-deepmind/mujoco.git"
MUJOCO_TAG=""                                  # empty: read from requirements.txt
MUJOCO_TAG_3_3_0_COMMIT="8de9b4e4e373caf01035c6c13efd363347d89032"
MUJOCO_EXPECTED_COMMIT=""
IOS_CMAKE_REPO="https://github.com/leetal/ios-cmake.git"
IOS_CMAKE_TAG="4.6.1"
IOS_CMAKE_COMMIT="b36d5bce6f9a2d11d9f144bd5882ba31cfdca420"
IOS_DEPLOYMENT_TARGET="17.0"
MACOS_DEPLOYMENT_TARGET="14.0"
ALL_SLICES="macos-arm64 ios-arm64 ios-arm64-simulator"
SLICES="$ALL_SLICES"
BUILD_ROOT="$REPO_ROOT/apple/build/mujoco"
OUTPUT="$REPO_ROOT/apple/UEMAKit/Binaries/mujoco.xcframework"
RECORDS_DIR="$REPO_ROOT/apple/records/mujoco_xcframework"
JOBS=""
DRY_RUN=0
CLEAN=0
EXTRA_CXX_FLAGS="${UEMA_MUJOCO_CXX_FLAGS:-}"
MUJOCO_SRC=""
IOS_CMAKE_SRC=""
XCF_IDS=""
XCF_COUNT=0

COMMON_CMAKE_FLAGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DMUJOCO_BUILD_EXAMPLES=OFF
  -DMUJOCO_BUILD_SIMULATE=OFF
  -DMUJOCO_BUILD_TESTS=OFF
  -DMUJOCO_TEST_PYTHON_UTIL=OFF
  -DMUJOCO_ENABLE_AVX=OFF
  -DMUJOCO_ENABLE_AVX_INTRINSICS=OFF
  -DMUJOCO_USE_FILAMENT=OFF
  -DMUJOCO_BUILD_MACOS_FRAMEWORKS=OFF
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE=OFF
)

# ----------------------------------------------------------------------------- output
usage() {
  cat <<'USAGE'
Usage: bash apple/scripts/build_mujoco_xcframework.sh [options]

Builds MuJoCo at the requirements.txt pin (mujoco==3.3.0) into
apple/UEMAKit/Binaries/mujoco.xcframework with three static slices:
macos-arm64, ios-arm64, ios-arm64-simulator.

Options
  --slices LIST              comma-separated subset of
                             macos-arm64,ios-arm64,ios-arm64-simulator
                             (default: all three; the acceptance line needs all three)
  --build-dir DIR            clones, build trees and staging
                             (default: apple/build/mujoco)
  --output PATH              the XCFramework to write
                             (default: apple/UEMAKit/Binaries/mujoco.xcframework)
  --records-dir DIR          record, patch diff and logs
                             (default: apple/records/mujoco_xcframework)
  --jobs N                   parallel compile jobs (default: sysctl -n hw.ncpu)
  --mujoco-tag TAG           MuJoCo git tag (default: the mujoco== line of
                             requirements.txt; only 3.3.0 has its commit pinned here)
  --ios-deployment-target V  minimum iOS version of the two iOS slices (default: 17.0)
  --macos-deployment-target V
                             minimum macOS version of the macos-arm64 slice (default: 14.0)
  --clean                    delete the build directory first (clones included)
  --dry-run                  print every command without running it; skips the
                             host checks, so it also runs on a machine without Xcode
  -h, --help                 this text

Environment
  UEMA_MUJOCO_CXX_FLAGS      extra compiler flags appended after MuJoCo's own
                             (CMakeLists.txt line 108, MUJOCO_CXX_FLAGS); separate
                             several flags with ';'. For a slice that fails on a
                             warning turned into an error, set -Wno-error.

Exit codes: 0 built and every check passed; 1 a check or build step failed;
2 a usage error.
USAGE
}

log() { printf '[%s] %s\n' "$(date '+%I:%M:%S %p')" "$*" >&2; }

die() {
  log "ERROR: $1"
  exit "${2:-1}"
}

on_error() {  # on_error LINE COMMAND
  log "FAILED at line $1 of ${BASH_SOURCE[0]}: $2; the logs are under $RECORDS_DIR/logs"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

# quoted CMD...: the command as one shell-quoted line (printf %q), so a logged
# command can be pasted back into a shell.
quoted() {
  local out="" arg
  for arg in "$@"; do out="$out$(printf '%q' "$arg") "; done
  printf '%s' "${out% }"
}

# run CMD...: print the command, then run it unless --dry-run.
run() {
  log "+ $(quoted "$@")"
  if [[ "$DRY_RUN" -eq 1 ]]; then return 0; fi
  "$@"
}

# run_logged LOGFILE CMD...: like run, with stdout and stderr appended to LOGFILE.
run_logged() {
  local logfile="$1"
  shift
  log "+ $(quoted "$@") (log: $logfile)"
  if [[ "$DRY_RUN" -eq 1 ]]; then return 0; fi
  "$@" 2>&1 | tee -a "$logfile"
}

# ----------------------------------------------------------------------------- arguments
need_value() {  # need_value OPTION VALUE
  [[ -n "$2" ]] || die "option $1 needs a value" 2
}

abs_path() {  # abs_path PATH: absolute form of PATH (the path need not exist yet)
  case "$1" in
    /*) printf '%s' "$1" ;;
    *) printf '%s/%s' "$PWD" "$1" ;;
  esac
}

valid_slice() {  # valid_slice NAME
  case "$1" in
    macos-arm64|ios-arm64|ios-arm64-simulator) return 0 ;;
    *) return 1 ;;
  esac
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --slices) need_value "$1" "${2:-}"; SLICES="$(printf '%s' "$2" | tr ',' ' ')"; shift 2 ;;
      --build-dir) need_value "$1" "${2:-}"; BUILD_ROOT="$(abs_path "$2")"; shift 2 ;;
      --output) need_value "$1" "${2:-}"; OUTPUT="$(abs_path "$2")"; shift 2 ;;
      --records-dir) need_value "$1" "${2:-}"; RECORDS_DIR="$(abs_path "$2")"; shift 2 ;;
      --jobs) need_value "$1" "${2:-}"; JOBS="$2"; shift 2 ;;
      --mujoco-tag) need_value "$1" "${2:-}"; MUJOCO_TAG="$2"; shift 2 ;;
      --ios-deployment-target) need_value "$1" "${2:-}"; IOS_DEPLOYMENT_TARGET="$2"; shift 2 ;;
      --macos-deployment-target) need_value "$1" "${2:-}"; MACOS_DEPLOYMENT_TARGET="$2"; shift 2 ;;
      --clean) CLEAN=1; shift ;;
      --dry-run) DRY_RUN=1; shift ;;
      -h|--help) usage; exit 0 ;;
      *) usage >&2; die "unknown option: $1" 2 ;;
    esac
  done
  local slice seen=" "
  for slice in $SLICES; do
    valid_slice "$slice" || die "unknown slice '$slice' (valid: $ALL_SLICES)" 2
    [[ "$seen" != *" $slice "* ]] || die "slice '$slice' is named twice in --slices" 2
    seen="$seen$slice "
  done
  [[ -n "$SLICES" ]] || die "--slices names no slice" 2
  if [[ -n "$JOBS" && ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then die "--jobs needs a whole number of 1 or more, got '$JOBS'" 2; fi
}

# ----------------------------------------------------------------------------- step 1: host
version_at_least() {  # version_at_least X.Y[.Z] MAJOR MINOR
  local major minor
  major="$(printf '%s' "$1" | cut -d. -f1)"
  minor="$(printf '%s' "$1" | cut -d. -f2 | tr -cd '0-9')"
  [[ "$major" -gt "$2" ]] || { [[ "$major" -eq "$2" ]] && [[ "${minor:-0}" -ge "$3" ]]; }
}

needed_sdks() {  # the xcrun SDK names the requested slices need, one per line
  local slice
  for slice in $SLICES; do
    case "$slice" in
      macos-arm64) printf '%s\n' macosx ;;
      ios-arm64) printf '%s\n' iphoneos ;;
      ios-arm64-simulator) printf '%s\n' iphonesimulator ;;
    esac
  done | sort -u
}

check_host() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry run: host checks skipped (this host: $(uname -s) $(uname -m); a real run needs Darwin)"
    JOBS="${JOBS:-<hw.ncpu>}"
    return 0
  fi
  local host
  host="$(uname -s)"
  [[ "$host" == "Darwin" ]] || die "this script runs on macOS: uname -s must print Darwin, it printed $host"
  local tool
  for tool in git cmake xcodebuild xcrun /usr/libexec/PlistBuddy; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool not found: $tool"
  done
  local cmake_version
  cmake_version="$(cmake --version | head -n 1 | awk '{print $3}')"
  version_at_least "$cmake_version" 3 16 \
    || die "CMake $cmake_version is older than 3.16, the minimum in CMakeLists.txt line 15 at tag $MUJOCO_TAG"
  log "host: macOS $(sw_vers -productVersion) on $(uname -m); cmake $cmake_version; $(xcodebuild -version | tr '\n' ' ')"
  local sdk path
  for sdk in $(needed_sdks); do
    path="$(xcrun --sdk "$sdk" --show-sdk-path 2>/dev/null)" \
      || die "SDK '$sdk' not found by xcrun; the iOS slices need Xcode with the iOS 17 SDK installed"
    log "SDK $sdk $(xcrun --sdk "$sdk" --show-sdk-version): $path"
  done
  if [[ -z "$JOBS" ]]; then JOBS="$(sysctl -n hw.ncpu)"; fi
}

# ----------------------------------------------------------------------------- step 2: sources
resolve_mujoco_tag() {
  local req="$REPO_ROOT/requirements.txt"
  if [[ -z "$MUJOCO_TAG" ]]; then
    [[ -f "$req" ]] || die "requirements.txt not found at $req; it carries the mujoco== pin"
    MUJOCO_TAG="$( (grep -E '^mujoco==' "$req" || true) | head -n 1 | sed 's/^mujoco==//' | tr -d '[:space:]')"
    [[ -n "$MUJOCO_TAG" ]] || die "no 'mujoco==' line in $req"
  fi
  case "$MUJOCO_TAG" in
    3.3.0) MUJOCO_EXPECTED_COMMIT="$MUJOCO_TAG_3_3_0_COMMIT" ;;
    *) MUJOCO_EXPECTED_COMMIT=""
       log "tag $MUJOCO_TAG has no commit pinned in this script; the clone's commit is recorded, not checked" ;;
  esac
  log "MuJoCo tag $MUJOCO_TAG (requirements.txt line: $(grep -E '^mujoco==' "$req" | head -n 1 || printf 'none'))"
}

clone_at_tag() {  # clone_at_tag REPOSITORY TAG DIRECTORY EXPECTED_COMMIT
  local repo="$1" tag="$2" dir="$3" expected="$4"
  if [[ -d "$dir/.git" ]]; then
    log "reusing the clone at $dir (--clean deletes it)"
  else
    run git clone --depth 1 --branch "$tag" "$repo" "$dir"
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "would compare HEAD of $dir with ${expected:-<not pinned>}"
    return 0
  fi
  local head
  head="$(git -C "$dir" rev-parse HEAD)"
  log "$dir is at commit $head"
  if [[ -n "$expected" && "$head" != "$expected" ]]; then
    die "$dir is at $head; tag $tag was commit $expected on 2026-11-04 (rerun with --clean, or pass --mujoco-tag)"
  fi
}

fetch_sources() {
  clone_at_tag "$MUJOCO_REPO" "$MUJOCO_TAG" "$MUJOCO_SRC" "$MUJOCO_EXPECTED_COMMIT"
  if needs_ios; then
    clone_at_tag "$IOS_CMAKE_REPO" "$IOS_CMAKE_TAG" "$IOS_CMAKE_SRC" "$IOS_CMAKE_COMMIT"
  fi
}

needs_ios() {
  local slice
  for slice in $SLICES; do
    case "$slice" in ios-*) return 0 ;; esac
  done
  return 1
}

# ----------------------------------------------------------------------------- step 3: patches
# patch_line FILE OLD NEW COUNT: replace the literal text OLD by NEW on every line
# that holds it; FILE must hold OLD on exactly COUNT lines before (or NEW on COUNT
# lines already, in which case nothing changes).
patch_line() {
  local file="$1" old="$2" new="$3" expected="$4"
  local before after tmp line
  before="$(grep -cF -- "$old" "$file" || true)"
  after="$(grep -cF -- "$new" "$file" || true)"
  if [[ "$before" -eq 0 && "$after" -eq "$expected" ]]; then
    log "already patched: $file"
    return 0
  fi
  [[ "$before" -eq "$expected" ]] \
    || die "$file holds '$old' on $before line(s), expected $expected; the checkout differs from tag $MUJOCO_TAG as read on 2026-11-04"
  tmp="$file.patch.tmp"
  : > "$tmp"
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == *"$old"* ]]; then line="${line//"$old"/$new}"; fi
    printf '%s\n' "$line" >> "$tmp"
  done < "$file"
  mv "$tmp" "$file"
  after="$(grep -cF -- "$new" "$file" || true)"
  [[ "$after" -eq "$expected" ]] || die "$file: the patch left $after line(s) with '$new', expected $expected"
  log "patched $file: $expected line(s) '$old' -> '$new'"
}

patch_sources() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "would patch $MUJOCO_SRC/CMakeLists.txt: 'add_library(mujoco SHARED' -> 'add_library(mujoco STATIC' (1 line)"
    log "would patch $MUJOCO_SRC/cmake/MujocoMacOS.cmake: 'if(APPLE)' -> 'if(APPLE AND NOT IOS)' (2 lines)"
    log "would write $RECORDS_DIR/mujoco_source_patch.diff"
    return 0
  fi
  patch_line "$MUJOCO_SRC/CMakeLists.txt" \
    'add_library(mujoco SHARED ${MUJOCO_RESOURCE_FILES})' \
    'add_library(mujoco STATIC ${MUJOCO_RESOURCE_FILES})' 1
  patch_line "$MUJOCO_SRC/cmake/MujocoMacOS.cmake" 'if(APPLE)' 'if(APPLE AND NOT IOS)' 2
  git -C "$MUJOCO_SRC" diff > "$RECORDS_DIR/mujoco_source_patch.diff"
  log "patch diff: $RECORDS_DIR/mujoco_source_patch.diff ($(wc -l < "$RECORDS_DIR/mujoco_source_patch.diff" | tr -d ' ') lines)"
}

# ----------------------------------------------------------------------------- step 4: slices
slice_build_dir() { printf '%s/build-%s' "$BUILD_ROOT" "$1"; }
slice_archive() { printf '%s/out/%s/libmujoco.a' "$BUILD_ROOT" "$1"; }

slice_flags() {  # slice_flags SLICE: the slice-specific cmake flags, one per line
  case "$1" in
    macos-arm64)
      printf '%s\n' "-DCMAKE_OSX_ARCHITECTURES=arm64" "-DCMAKE_OSX_DEPLOYMENT_TARGET=$MACOS_DEPLOYMENT_TARGET" ;;
    ios-arm64)
      printf '%s\n' "-DCMAKE_TOOLCHAIN_FILE=$IOS_CMAKE_SRC/ios.toolchain.cmake" "-DPLATFORM=OS64" \
        "-DDEPLOYMENT_TARGET=$IOS_DEPLOYMENT_TARGET" "-DENABLE_BITCODE=OFF" ;;
    ios-arm64-simulator)
      printf '%s\n' "-DCMAKE_TOOLCHAIN_FILE=$IOS_CMAKE_SRC/ios.toolchain.cmake" "-DPLATFORM=SIMULATORARM64" \
        "-DDEPLOYMENT_TARGET=$IOS_DEPLOYMENT_TARGET" "-DENABLE_BITCODE=OFF" ;;
  esac
}

configure_slice() {  # configure_slice SLICE
  local slice="$1" build_dir log_file args_file line
  build_dir="$(slice_build_dir "$slice")"
  log_file="$RECORDS_DIR/logs/${slice}_configure.log"
  args_file="$RECORDS_DIR/logs/${slice}_cmake_args.txt"
  local -a args
  args=(-S "$MUJOCO_SRC" -B "$build_dir" -G "Unix Makefiles" "${COMMON_CMAKE_FLAGS[@]}")
  if [[ -n "$EXTRA_CXX_FLAGS" ]]; then args+=("-DMUJOCO_CXX_FLAGS=$EXTRA_CXX_FLAGS"); fi
  while IFS= read -r line; do args+=("$line"); done < <(slice_flags "$slice")
  if [[ "$DRY_RUN" -eq 0 ]]; then printf '%s\n' "${args[@]}" > "$args_file"; fi
  run_logged "$log_file" cmake "${args[@]}"
}

build_slice() {  # build_slice SLICE
  local slice="$1"
  run_logged "$RECORDS_DIR/logs/${slice}_build.log" \
    cmake --build "$(slice_build_dir "$slice")" --target mujoco --config Release -j "$JOBS"
}

merge_slice() {  # merge_slice SLICE: every static archive the build wrote -> one libmujoco.a
  local slice="$1" build_dir out
  build_dir="$(slice_build_dir "$slice")"
  out="$(slice_archive "$slice")"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "would merge every *.a under $build_dir (CMakeFiles/ excluded) into $out with xcrun libtool -static -no_warning_for_no_symbols"
    return 0
  fi
  mkdir -p "$(dirname "$out")"
  local -a archives
  archives=()
  local f
  while IFS= read -r f; do archives+=("$f"); done \
    < <(find "$build_dir" -type f -name '*.a' -not -path '*/CMakeFiles/*' | sort)
  [[ "${#archives[@]}" -ge 1 ]] || die "slice $slice: no static archive under $build_dir after the build"
  log "slice $slice: merging ${#archives[@]} archive(s):"
  printf '    %s\n' "${archives[@]}" >&2
  rm -f "$out"
  xcrun libtool -static -no_warning_for_no_symbols -o "$out" "${archives[@]}" 2>&1 \
    | tee -a "$RECORDS_DIR/logs/${slice}_build.log"
  [[ -f "$out" ]] || die "slice $slice: libtool did not write $out"
}

verify_slice() {  # verify_slice SLICE: lipo names arm64; nm lists _mj_step and _mj_versionString
  local slice="$1" out
  out="$(slice_archive "$slice")"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "would verify $out: xcrun lipo -info names arm64; xcrun nm -g lists ' T _mj_step' and ' T _mj_versionString'"
    return 0
  fi
  local lipo_out n_step n_ver n_lto bytes
  lipo_out="$(xcrun lipo -info "$out")"
  log "slice $slice: lipo -info: $lipo_out"
  [[ "$lipo_out" == *arm64* ]] || die "slice $slice: lipo -info does not name arm64"
  n_step="$(xcrun nm -g "$out" | grep -c ' T _mj_step$' || true)"
  n_ver="$(xcrun nm -g "$out" | grep -c ' T _mj_versionString$' || true)"
  [[ "$n_step" -ge 1 && "$n_ver" -ge 1 ]] \
    || die "slice $slice: nm lists _mj_step $n_step time(s) and _mj_versionString $n_ver time(s); each must be at least 1"
  n_lto="$(grep -c -- '-flto' "$(slice_build_dir "$slice")/CMakeFiles/mujoco.dir/flags.make" 2>/dev/null || true)"
  bytes="$(wc -c < "$out" | tr -d ' ')"
  log "slice $slice: $bytes bytes; _mj_step x$n_step; _mj_versionString x$n_ver; '-flto' lines in flags.make: ${n_lto:-0} (0 expected)"
  if [[ "${n_lto:-0}" -ne 0 ]]; then log "WARNING: slice $slice was compiled with -flto; the archive holds bitcode objects"; fi
  write_slice_fragment "$slice" "$bytes" "$lipo_out" "$n_step" "$n_ver" "${n_lto:-0}"
}

# ----------------------------------------------------------------------------- step 5: headers
stage_headers() {
  local stage="$BUILD_ROOT/stage/include" n
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "would copy $MUJOCO_SRC/include/mujoco/*.h to $stage/mujoco/ and write $stage/module.modulemap (module mujoco, umbrella header mujoco/mujoco.h)"
    return 0
  fi
  rm -rf "$stage"
  mkdir -p "$stage/mujoco"
  cp "$MUJOCO_SRC"/include/mujoco/*.h "$stage/mujoco/"
  n="$(find "$stage/mujoco" -name '*.h' | wc -l | tr -d ' ')"
  cat > "$stage/module.modulemap" <<'MODULEMAP'
// Module map for the static MuJoCo slices of mujoco.xcframework, written by
// apple/scripts/build_mujoco_xcframework.sh from MuJoCo's own dist/module.modulemap
// (tag 3.3.0), changed from a framework module to a library module because the
// slices are static libraries with a Headers folder: Headers/module.modulemap and
// Headers/mujoco/*.h, so that #include <mujoco/mujoco.h> resolves as it does for
// the Python wheel's headers.
module mujoco {
  umbrella header "mujoco/mujoco.h"
  export *
  module * { export * }
}
MODULEMAP
  log "staged $n header(s) (14 expected at tag 3.3.0: MUJOCO_HEADERS, CMakeLists.txt lines 55-70) and module.modulemap"
  if [[ "$n" -ne 14 ]]; then log "WARNING: header count $n differs from the 14 of tag 3.3.0"; fi
}

# ----------------------------------------------------------------------------- step 6: xcframework
create_xcframework() {
  local stage="$BUILD_ROOT/stage/include" slice
  local -a args
  args=(-create-xcframework)
  for slice in $SLICES; do
    args+=(-library "$(slice_archive "$slice")" -headers "$stage")
  done
  args+=(-output "$OUTPUT")
  run rm -rf "$OUTPUT"
  run mkdir -p "$(dirname "$OUTPUT")"
  run_logged "$RECORDS_DIR/logs/xcframework.log" xcodebuild "${args[@]}"
}

verify_xcframework() {
  local plist="$OUTPUT/Info.plist" n_slices=0 slice ids n
  for slice in $SLICES; do n_slices=$((n_slices + 1)); done
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "would count the LibraryIdentifier entries of $plist (PlistBuddy Print :AvailableLibraries): $n_slices expected"
    return 0
  fi
  [[ -f "$plist" ]] || die "$plist is missing after xcodebuild -create-xcframework"
  ids="$(/usr/libexec/PlistBuddy -c 'Print :AvailableLibraries' "$plist" | (grep 'LibraryIdentifier' || true) | awk '{print $3}' | sort | tr '\n' ' ')"
  ids="${ids% }"
  n="$(printf '%s' "$ids" | wc -w | tr -d ' ')"
  log "XCFramework libraries ($n): $ids"
  [[ "$n" -eq "$n_slices" ]] || die "Info.plist lists $n libraries, expected $n_slices"
  if [[ "$n" -ne 3 ]]; then log "NOTE: the acceptance line needs 3 libraries; this run built $n (--slices)"; fi
  XCF_IDS="$ids"
  XCF_COUNT="$n"
}

# ----------------------------------------------------------------------------- step 7: record
json_str() {  # json_str TEXT: TEXT as a JSON string body (backslash and quote escaped; a line break becomes '; ')
  printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' | awk 'NR > 1 { printf "; " } { printf "%s", $0 }'
}

write_slice_fragment() {  # SLICE BYTES LIPO N_STEP N_VER N_LTO
  local f="$RECORDS_DIR/.slice_$1.json"
  {
    printf '    {"slice": "%s", "archive": "%s", "bytes": %s, "lipo_info": "%s",\n' \
      "$1" "$(json_str "$(slice_archive "$1")")" "$2" "$(json_str "$3")"
    printf '     "nm_mj_step": %s, "nm_mj_versionString": %s, "flto_lines_in_flags_make": %s,\n' "$4" "$5" "$6"
    printf '     "cmake_args_file": "%s", "configure_log": "%s", "build_log": "%s"}' \
      "$(json_str "$RECORDS_DIR/logs/$1_cmake_args.txt")" \
      "$(json_str "$RECORDS_DIR/logs/$1_configure.log")" \
      "$(json_str "$RECORDS_DIR/logs/$1_build.log")"
  } > "$f"
}

write_record() {
  local record="$RECORDS_DIR/build_record.json" slice first=1 flag
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "would write $record"
    return 0
  fi
  {
    printf '{\n'
    printf '  "script": "apple/scripts/build_mujoco_xcframework.sh",\n'
    printf '  "finished_at": "%s",\n' "$(date '+%Y-%m-%d %I:%M %p %Z')"
    printf '  "host": {"macos": "%s", "machine": "%s", "xcodebuild_version": "%s", "cmake_version": "%s"},\n' \
      "$(sw_vers -productVersion)" "$(uname -m)" "$(json_str "$(xcodebuild -version)")" \
      "$(cmake --version | head -n 1 | awk '{print $3}')"
    printf '  "mujoco": {"repository": "%s", "tag": "%s", "commit": "%s", "requirements_pin": "%s"},\n' \
      "$MUJOCO_REPO" "$MUJOCO_TAG" "$(git -C "$MUJOCO_SRC" rev-parse HEAD)" \
      "$(json_str "$(grep -E '^mujoco==' "$REPO_ROOT/requirements.txt" | head -n 1 || true)")"
    if needs_ios; then
      printf '  "ios_cmake": {"repository": "%s", "tag": "%s", "commit": "%s"},\n' \
        "$IOS_CMAKE_REPO" "$IOS_CMAKE_TAG" "$(git -C "$IOS_CMAKE_SRC" rev-parse HEAD)"
    fi
    printf '  "ios_deployment_target": "%s",\n' "$IOS_DEPLOYMENT_TARGET"
    printf '  "macos_deployment_target": "%s",\n' "$MACOS_DEPLOYMENT_TARGET"
    printf '  "source_patch_diff": "%s",\n' "$(json_str "$RECORDS_DIR/mujoco_source_patch.diff")"
    printf '  "common_cmake_flags": ['
    for flag in "${COMMON_CMAKE_FLAGS[@]}"; do
      if [[ "$first" -eq 1 ]]; then first=0; else printf ', '; fi
      printf '"%s"' "$(json_str "$flag")"
    done
    printf '],\n'
    printf '  "extra_cxx_flags": "%s",\n' "$(json_str "$EXTRA_CXX_FLAGS")"
    printf '  "slices": [\n'
    first=1
    for slice in $SLICES; do
      if [[ "$first" -eq 1 ]]; then first=0; else printf ',\n'; fi
      cat "$RECORDS_DIR/.slice_$slice.json"
    done
    printf '\n  ],\n'
    printf '  "xcframework": {"path": "%s", "library_identifiers": "%s", "library_count": %s},\n' \
      "$(json_str "$OUTPUT")" "$(json_str "$XCF_IDS")" "$XCF_COUNT"
    printf '  "logs_dir": "%s",\n' "$(json_str "$RECORDS_DIR/logs")"
    printf '  "exit_status": 0\n'
    printf '}\n'
  } > "$record"
  rm -f "$RECORDS_DIR"/.slice_*.json
  log "record: $record"
}

print_summary() {
  local slice
  log "SUMMARY"
  log "  MuJoCo tag $MUJOCO_TAG; slices: $SLICES"
  for slice in $SLICES; do
    log "  $slice -> $(slice_archive "$slice")"
  done
  log "  XCFramework: $OUTPUT"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "  dry run: nothing was cloned, built or written"
  else
    log "  XCFramework libraries ($XCF_COUNT): $XCF_IDS"
    log "  record and logs: $RECORDS_DIR"
    log "RESULT: exit 0; every check passed"
  fi
}

# ----------------------------------------------------------------------------- main
main() {
  parse_args "$@"
  MUJOCO_SRC="$BUILD_ROOT/mujoco"
  IOS_CMAKE_SRC="$BUILD_ROOT/ios-cmake"
  log "repository root: $REPO_ROOT"
  if [[ "$CLEAN" -eq 1 ]]; then run rm -rf "$BUILD_ROOT"; fi
  run mkdir -p "$BUILD_ROOT" "$RECORDS_DIR/logs"
  resolve_mujoco_tag
  check_host
  fetch_sources
  patch_sources
  local slice
  for slice in $SLICES; do
    log "=== slice $slice ==="
    configure_slice "$slice"
    build_slice "$slice"
    merge_slice "$slice"
    verify_slice "$slice"
  done
  stage_headers
  create_xcframework
  verify_xcframework
  write_record
  print_summary
}

# When sourced (apple/scripts/tests/test_build_mujoco_xcframework.sh), only the
# functions are defined; main runs when the file is executed.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
