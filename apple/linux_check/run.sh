#!/usr/bin/env bash
# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
#
# The Linux check of the Apple port (README.md beside this file). Fetches Swift 6.2 for Ubuntu
# 24.04 and the MuJoCo 3.3.0 Linux release into $UEMA_LINUX_CACHE (default
# ~/.cache/uema_linux_check) when absent, writes the VisionPatcher copy, builds and runs the
# tests, and greps the package for the MuJoCo renderer prefix. Needs curl, tar, python3 with
# numpy and Pillow, and network access to download.swift.org and github.com.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cache="${UEMA_LINUX_CACHE:-$HOME/.cache/uema_linux_check}"
mkdir -p "$cache"
swift_dir="$cache/swift-6.2-RELEASE-ubuntu24.04"
if [ ! -x "$swift_dir/usr/bin/swift" ]; then
  curl -sSL https://download.swift.org/swift-6.2-release/ubuntu2404/swift-6.2-RELEASE/swift-6.2-RELEASE-ubuntu24.04.tar.gz | tar xz -C "$cache"
fi
export MUJOCO_DIR="$cache/mujoco-3.3.0"
if [ ! -f "$MUJOCO_DIR/lib/libmujoco.so" ]; then
  curl -sSL https://github.com/google-deepmind/mujoco/releases/download/3.3.0/mujoco-3.3.0-linux-x86_64.tar.gz | tar xz -C "$cache"
fi
export PATH="$swift_dir/usr/bin:$PATH"
swift --version
python3 "$here/generate_vision_copy.py"
cd "$here"
swift build --build-tests
swift test --skip-build 2>&1 | tee "$here/.build/linux_check_test.log" | grep -E "error:|Executed [0-9]+ tests|WI-30|WI-37|audio" || true
# CMuJoCoRendererFreeTests' check, by grep: no code line under Sources or Tests calls mjr_.
if grep -rn "mjr_" ../UEMAKit/Sources ../UEMAKit/Tests ../UEMAKit/Package.swift --include=*.swift --include=*.h --include=*.c \
    | grep -v "//.*mjr_" | grep -v "CMuJoCoRendererFreeTests.swift"; then
  echo "renderer prefix found in code"; exit 1
fi
echo "renderer prefix: none in code"
grep -q "with 0 failures" <(grep -E "Executed [0-9]+ tests" "$here/.build/linux_check_test.log" | tail -1)
