#!/usr/bin/env bash
# =============================================================================
# apple/scripts/tests/test_build_mujoco_xcframework.sh
#
# Unit test of the shell functions of apple/scripts/build_mujoco_xcframework.sh
# (work item WI-18). It needs bash, awk, sed, grep, cmp and mktemp; it needs no
# network, no Xcode and no CMake, so it runs on the Windows build machine (Git
# Bash) and on a Mac alike.
#
# INPUT    the script, sourced (its last lines run main only when executed);
#          a temporary directory from mktemp -d, deleted at exit
# OPERATION
#   1. bash -n on the script
#   2. version_at_least, needed_sdks, json_str, quoted, abs_path, valid_slice,
#      slice_flags: each on fixed inputs with the expected text written here
#   3. patch_line on files written here with the exact lines of tag 3.3.0
#      (CMakeLists.txt line 79; cmake/MujocoMacOS.cmake lines 15 and 37): the
#      line counts after the patch, the other lines unchanged (cmp), the second
#      run's "already patched", and a wrong expected count exiting 1 with the
#      file unchanged
#   4. parse_args: a duplicate slice, an unknown slice, an unknown option and a
#      non-numeric --jobs each exit 2; a valid line sets the globals
#   5. resolve_mujoco_tag against a requirements.txt written here
#   6. check_host in dry-run mode (the only branch that runs off a Mac)
#   7. the script executed with --dry-run into the temporary directory: exit 0,
#      3 cmake configure lines, 3 cmake --build lines, one xcodebuild line with
#      3 -library pairs, the slice-specific flags, and no directory created
#   8. --help exits 0 and prints "Usage:"
# OUTPUT   one line per check ("ok" or "FAIL" with what was expected and found);
#          the last line is "PASS: N checks, 0 failed" (exit 0) or
#          "FAIL: N checks, M failed" (exit 1)
#
# Run:  bash apple/scripts/tests/test_build_mujoco_xcframework.sh
# =============================================================================

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/../build_mujoco_xcframework.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# shellcheck source=../build_mujoco_xcframework.sh
source "$SCRIPT"
set +e
trap - ERR

PASSED=0
FAILED=0

check_eq() {  # check_eq NAME EXPECTED FOUND
  if [[ "$2" == "$3" ]]; then
    PASSED=$((PASSED + 1)); printf 'ok   %s\n' "$1"
  else
    FAILED=$((FAILED + 1)); printf 'FAIL %s: expected [%s], found [%s]\n' "$1" "$2" "$3"
  fi
}

check_contains() {  # check_contains NAME TEXT NEEDLE
  if [[ "$2" == *"$3"* ]]; then
    PASSED=$((PASSED + 1)); printf 'ok   %s\n' "$1"
  else
    FAILED=$((FAILED + 1)); printf 'FAIL %s: [%s] not found in the text\n' "$1" "$3"
  fi
}

count_lines() {  # count_lines TEXT NEEDLE: lines of TEXT that hold NEEDLE (fixed string)
  printf '%s\n' "$1" | grep -cF -- "$2"
}

# --- 1. syntax ---------------------------------------------------------------
bash -n "$SCRIPT"; check_eq "bash -n exits 0" 0 "$?"

# --- 2. small functions ------------------------------------------------------
version_at_least "3.16.0" 3 16; check_eq "version_at_least 3.16.0 >= 3.16" 0 "$?"
version_at_least "3.28.1" 3 16; check_eq "version_at_least 3.28.1 >= 3.16" 0 "$?"
version_at_least "4.0" 3 16;    check_eq "version_at_least 4.0 >= 3.16" 0 "$?"
version_at_least "3.15.7" 3 16; check_eq "version_at_least 3.15.7 < 3.16" 1 "$?"
version_at_least "2.8.12" 3 16; check_eq "version_at_least 2.8.12 < 3.16" 1 "$?"

SLICES="ios-arm64 ios-arm64-simulator"
check_eq "needed_sdks for the two iOS slices" "iphoneos iphonesimulator" "$(needed_sdks | tr '\n' ' ' | sed 's/ $//')"
SLICES="macos-arm64"
check_eq "needed_sdks for the macOS slice" "macosx" "$(needed_sdks)"
SLICES="$ALL_SLICES"
check_eq "needed_sdks for all three slices" "iphoneos iphonesimulator macosx" "$(needed_sdks | tr '\n' ' ' | sed 's/ $//')"

check_eq "json_str escapes quote and backslash" 'a\"b\\c' "$(json_str 'a"b\c')"
check_eq "json_str joins lines with '; '" "x; y" "$(json_str "$(printf 'x\ny')")"
check_eq "json_str keeps a ';' as written" "-Wno-error;-Wall" "$(json_str '-Wno-error;-Wall')"
check_eq "json_str of an empty string" "" "$(json_str '')"

check_eq "quoted keeps a space-holding argument as one word" 'cmake -G Unix\ Makefiles' "$(quoted cmake -G "Unix Makefiles")"

check_eq "abs_path leaves an absolute path alone" "/a/b" "$(abs_path /a/b)"
check_eq "abs_path roots a relative path at PWD" "$PWD/x/y" "$(abs_path x/y)"

valid_slice macos-arm64; check_eq "valid_slice macos-arm64" 0 "$?"
valid_slice ios-arm64-simulator; check_eq "valid_slice ios-arm64-simulator" 0 "$?"
valid_slice ios-x86_64; check_eq "valid_slice rejects ios-x86_64" 1 "$?"

IOS_CMAKE_SRC="$TMP/ios-cmake"
flags="$(slice_flags macos-arm64 | tr '\n' ' ')"
check_contains "slice_flags macos-arm64 names arm64" "$flags" "-DCMAKE_OSX_ARCHITECTURES=arm64"
check_contains "slice_flags macos-arm64 names the macOS 14.0 floor" "$flags" "-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0"
flags="$(slice_flags ios-arm64 | tr '\n' ' ')"
check_contains "slice_flags ios-arm64 names OS64" "$flags" "-DPLATFORM=OS64 "
check_contains "slice_flags ios-arm64 names the iOS 17.0 floor" "$flags" "-DDEPLOYMENT_TARGET=17.0"
check_contains "slice_flags ios-arm64 names the toolchain file" "$flags" "-DCMAKE_TOOLCHAIN_FILE=$TMP/ios-cmake/ios.toolchain.cmake"
flags="$(slice_flags ios-arm64-simulator | tr '\n' ' ')"
check_contains "slice_flags ios-arm64-simulator names SIMULATORARM64" "$flags" "-DPLATFORM=SIMULATORARM64"

# --- 3. patch_line -----------------------------------------------------------
MUJOCO_TAG="3.3.0"
cmakelists="$TMP/CMakeLists.txt"
printf '%s\n' \
  'if(WIN32)' \
  '  set(MUJOCO_RESOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/dist/mujoco.rc)' \
  'else()' \
  '  set(MUJOCO_RESOURCE_FILES "")' \
  'endif()' \
  '' \
  'add_library(mujoco SHARED ${MUJOCO_RESOURCE_FILES})' \
  'target_include_directories(' > "$cmakelists"
sed 's/add_library(mujoco SHARED/add_library(mujoco STATIC/' "$cmakelists" > "$TMP/CMakeLists.expected"
out="$(patch_line "$cmakelists" 'add_library(mujoco SHARED ${MUJOCO_RESOURCE_FILES})' 'add_library(mujoco STATIC ${MUJOCO_RESOURCE_FILES})' 1 2>&1)"
check_eq "patch_line CMakeLists.txt exits 0" 0 "$?"
check_eq "patch_line CMakeLists.txt: 1 STATIC line" 1 "$(grep -cF 'add_library(mujoco STATIC ${MUJOCO_RESOURCE_FILES})' "$cmakelists")"
check_eq "patch_line CMakeLists.txt: 0 SHARED lines" 0 "$(grep -cF 'add_library(mujoco SHARED' "$cmakelists")"
cmp -s "$cmakelists" "$TMP/CMakeLists.expected"; check_eq "patch_line CMakeLists.txt: the other 7 lines unchanged" 0 "$?"
check_contains "patch_line CMakeLists.txt logs the change" "$out" "1 line(s)"
out="$(patch_line "$cmakelists" 'add_library(mujoco SHARED ${MUJOCO_RESOURCE_FILES})' 'add_library(mujoco STATIC ${MUJOCO_RESOURCE_FILES})' 1 2>&1)"
check_eq "patch_line second run exits 0" 0 "$?"
check_contains "patch_line second run says already patched" "$out" "already patched"
cmp -s "$cmakelists" "$TMP/CMakeLists.expected"; check_eq "patch_line second run changes nothing" 0 "$?"

macos_cmake="$TMP/MujocoMacOS.cmake"
printf '%s\n' \
  'if(APPLE)' \
  '  set(MUJOCO_MACOSX_VERSION_MIN 11)' \
  'else()' \
  '  set(MUJOCO_MACOS_COMPILE_OPTIONS "")' \
  'endif()' \
  'function(enforce_mujoco_macosx_min_version)' \
  '  if(APPLE)' \
  '    add_compile_options(${MUJOCO_MACOS_COMPILE_OPTIONS})' \
  '  endif()' \
  'endfunction()' > "$macos_cmake"
cp "$macos_cmake" "$TMP/MujocoMacOS.original"
out="$(patch_line "$macos_cmake" 'if(APPLE)' 'if(APPLE AND NOT IOS)' 3 2>&1)"
check_eq "patch_line with a wrong expected count (3) exits 1" 1 "$?"
check_contains "patch_line with a wrong count names the counts" "$out" "on 2 line(s), expected 3"
cmp -s "$macos_cmake" "$TMP/MujocoMacOS.original"; check_eq "patch_line with a wrong count leaves the file unchanged" 0 "$?"
out="$(patch_line "$macos_cmake" 'if(APPLE)' 'if(APPLE AND NOT IOS)' 2 2>&1)"
check_eq "patch_line MujocoMacOS.cmake exits 0" 0 "$?"
check_eq "patch_line MujocoMacOS.cmake: 2 patched lines" 2 "$(grep -cF 'if(APPLE AND NOT IOS)' "$macos_cmake")"
check_eq "patch_line MujocoMacOS.cmake: line 1 is the patched if" 'if(APPLE AND NOT IOS)' "$(sed -n 1p "$macos_cmake")"
check_eq "patch_line MujocoMacOS.cmake: line 7 is the patched if" '  if(APPLE AND NOT IOS)' "$(sed -n 7p "$macos_cmake")"
check_eq "patch_line MujocoMacOS.cmake: 10 lines as before" 10 "$(wc -l < "$macos_cmake" | tr -d ' ')"

# --- 4. parse_args -----------------------------------------------------------
(parse_args --slices macos-arm64,macos-arm64 >/dev/null 2>&1); check_eq "parse_args rejects a duplicate slice with exit 2" 2 "$?"
(parse_args --slices ios-x86_64 >/dev/null 2>&1); check_eq "parse_args rejects an unknown slice with exit 2" 2 "$?"
(parse_args --bogus >/dev/null 2>&1); check_eq "parse_args rejects an unknown option with exit 2" 2 "$?"
(parse_args --jobs abc >/dev/null 2>&1); check_eq "parse_args rejects --jobs abc with exit 2" 2 "$?"
(parse_args --jobs 0 >/dev/null 2>&1); check_eq "parse_args rejects --jobs 0 with exit 2" 2 "$?"
(parse_args --slices >/dev/null 2>&1); check_eq "parse_args rejects --slices without a value with exit 2" 2 "$?"
parse_args --slices ios-arm64,macos-arm64 --jobs 4 --macos-deployment-target 14.2 --ios-deployment-target 17.4 --mujoco-tag 3.3.0 --dry-run
check_eq "parse_args valid line exits 0" 0 "$?"
check_eq "parse_args --slices sets SLICES" "ios-arm64 macos-arm64" "$SLICES"
check_eq "parse_args --jobs sets JOBS" 4 "$JOBS"
check_eq "parse_args --macos-deployment-target sets the floor" "14.2" "$MACOS_DEPLOYMENT_TARGET"
check_eq "parse_args --ios-deployment-target sets the floor" "17.4" "$IOS_DEPLOYMENT_TARGET"
check_eq "parse_args --dry-run sets DRY_RUN" 1 "$DRY_RUN"
needs_ios; check_eq "needs_ios with an iOS slice" 0 "$?"
SLICES="macos-arm64"; needs_ios; check_eq "needs_ios without an iOS slice" 1 "$?"

# --- 5. resolve_mujoco_tag ---------------------------------------------------
mkdir -p "$TMP/repo"
printf '%s\n' 'numpy==2.2.0' 'mujoco==3.3.0' 'torch==2.6.0' > "$TMP/repo/requirements.txt"
REPO_ROOT="$TMP/repo"
MUJOCO_TAG=""
resolve_mujoco_tag 2>/dev/null
check_eq "resolve_mujoco_tag reads 3.3.0 from requirements.txt" "3.3.0" "$MUJOCO_TAG"
check_eq "resolve_mujoco_tag pins the 3.3.0 commit" "8de9b4e4e373caf01035c6c13efd363347d89032" "$MUJOCO_EXPECTED_COMMIT"
MUJOCO_TAG="3.3.1"
resolve_mujoco_tag 2> "$TMP/resolve.log"
out="$(cat "$TMP/resolve.log")"
check_eq "resolve_mujoco_tag with --mujoco-tag 3.3.1 keeps the tag" "3.3.1" "$MUJOCO_TAG"
check_eq "resolve_mujoco_tag with 3.3.1 pins no commit" "" "$MUJOCO_EXPECTED_COMMIT"
check_contains "resolve_mujoco_tag with 3.3.1 says so" "$out" "no commit pinned"
printf '%s\n' 'numpy==2.2.0' > "$TMP/repo/requirements.txt"
MUJOCO_TAG=""
out="$( (resolve_mujoco_tag) 2>&1 )"; status=$?
check_eq "resolve_mujoco_tag without a mujoco== line exits 1" 1 "$status"
check_contains "resolve_mujoco_tag without a mujoco== line names the line" "$out" "no 'mujoco==' line"

# --- 6. check_host, dry-run branch -------------------------------------------
DRY_RUN=1; JOBS=""
check_host 2> "$TMP/check_host.log"
check_eq "check_host in dry run exits 0 off a Mac" 0 "$?"
out="$(cat "$TMP/check_host.log")"
check_contains "check_host in dry run says the checks were skipped" "$out" "host checks skipped"
check_eq "check_host in dry run leaves JOBS as the hw.ncpu placeholder" "<hw.ncpu>" "$JOBS"

# --- 7. the script executed with --dry-run -----------------------------------
out="$(bash "$SCRIPT" --dry-run --build-dir "$TMP/build" --output "$TMP/out/mujoco.xcframework" --records-dir "$TMP/records" 2>&1)"
check_eq "--dry-run exits 0" 0 "$?"
check_eq "--dry-run prints 3 cmake configure lines" 3 "$(count_lines "$out" '+ cmake -S ')"
check_eq "--dry-run prints 3 cmake --build lines" 3 "$(count_lines "$out" '+ cmake --build ')"
check_eq "--dry-run prints 1 xcodebuild line" 1 "$(count_lines "$out" '+ xcodebuild -create-xcframework')"
xcf_line="$(printf '%s\n' "$out" | grep -F '+ xcodebuild -create-xcframework')"
check_eq "--dry-run xcodebuild line has 3 -library pairs" 3 "$(printf '%s' "$xcf_line" | grep -o -- ' -library ' | wc -l | tr -d ' ')"
check_eq "--dry-run xcodebuild line has 3 -headers pairs" 3 "$(printf '%s' "$xcf_line" | grep -o -- ' -headers ' | wc -l | tr -d ' ')"
check_eq "--dry-run: 1 configure line with the macOS 14.0 floor" 1 "$(count_lines "$out" '-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0')"
check_eq "--dry-run: 1 configure line with -DPLATFORM=OS64" 1 "$(count_lines "$out" '-DPLATFORM=OS64 ')"
check_eq "--dry-run: 1 configure line with -DPLATFORM=SIMULATORARM64" 1 "$(count_lines "$out" '-DPLATFORM=SIMULATORARM64')"
check_eq "--dry-run: 3 configure lines with -DMUJOCO_ENABLE_AVX_INTRINSICS=OFF" 3 "$(count_lines "$out" '-DMUJOCO_ENABLE_AVX_INTRINSICS=OFF')"
check_eq "--dry-run: 3 configure lines with -DCMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE=OFF" 3 "$(count_lines "$out" '-DCMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE=OFF')"
check_eq "--dry-run: 2 git clone lines" 2 "$(count_lines "$out" '+ git clone --depth 1 --branch ')"
check_eq "--dry-run: the last line says nothing was written" 1 "$(printf '%s\n' "$out" | tail -n 1 | grep -c 'nothing was cloned, built or written')"
[[ -e "$TMP/build" || -e "$TMP/records" || -e "$TMP/out" ]]; check_eq "--dry-run creates no directory" 1 "$?"

# --- 8. --help ---------------------------------------------------------------
out="$(bash "$SCRIPT" --help 2>&1)"
check_eq "--help exits 0" 0 "$?"
check_contains "--help prints the usage" "$out" "Usage: bash apple/scripts/build_mujoco_xcframework.sh"
check_contains "--help lists --macos-deployment-target" "$out" "--macos-deployment-target"

# --- result ------------------------------------------------------------------
total=$((PASSED + FAILED))
if [[ "$FAILED" -eq 0 ]]; then
  printf 'PASS: %d checks, 0 failed\n' "$total"
  exit 0
fi
printf 'FAIL: %d checks, %d failed\n' "$total" "$FAILED"
exit 1
