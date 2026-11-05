# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The XcodeGen spec apple/project.yml (work item WI-33) names what the plan row asks for.

Inputs: apple/project.yml, apple/UEMAKit/Package.swift (WI-17) and apple/.gitignore (WI-17), read from
disk relative to this file. Operation: parse the YAML with PyYAML and compare its keys and values with the
plan row (docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html, section 6, WI-33: "targets UEMAMind-macOS
and UEMAMind-iOS (SwiftUI), depending on the local package apple/UEMAKit; deployment targets macOS 14, iOS 17")
and with the keys of XcodeGen's project spec (Docs/ProjectSpec.md, read November 4, 2026). Outputs: pass or
fail per check; no file is written. XcodeGen and Xcode do not run here (no Mac): what this file cannot check,
`xcodegen generate` and `xcodebuild -scheme UEMAMind-macOS build` check at the first build (plan page,
section 8, step 6). Run from Source/Python:
    python -m pytest ../../apple/scripts/tests/test_project_yml.py -p no:cacheprovider -q
"""
import os
import re

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
APPLE = os.path.dirname(os.path.dirname(HERE))
PROJECT_YML = os.path.join(APPLE, "project.yml")
PACKAGE_SWIFT = os.path.join(APPLE, "UEMAKit", "Package.swift")
GITIGNORE = os.path.join(APPLE, ".gitignore")
BANNED_WORDS = os.path.join(os.path.dirname(APPLE), "docs", "apple_port", "banned_words.txt")

APPS = {"UEMAMind-macOS": ("macOS", "14.0"), "UEMAMind-iOS": ("iOS", "17.0")}
TESTS = {"UEMAMind-macOS": "UEMAMindTests-macOS", "UEMAMind-iOS": "UEMAMindTests-iOS"}


@pytest.fixture(scope="module")
def spec():
    with open(PROJECT_YML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_project_name_and_platform_floors(spec):
    # The project the acceptance line names (xcodegen generate writes UEMAMind.xcodeproj).
    assert spec["name"] == "UEMAMind"
    # macOS 14, iOS 17 project-wide: the same floors as Package.swift's .macOS(.v14) and .iOS(.v17).
    assert spec["options"]["deploymentTarget"] == {"macOS": "14.0", "iOS": "17.0"}
    with open(PACKAGE_SWIFT, encoding="utf-8") as f:
        manifest = f.read()
    assert ".macOS(.v14)" in manifest and ".iOS(.v17)" in manifest
    assert spec["settings"]["base"]["SWIFT_VERSION"] == "5.0"


def test_local_package_is_uemakit_with_its_library_product(spec):
    package = spec["packages"]["UEMAKit"]
    assert package == {"path": "UEMAKit"}
    assert os.path.isfile(os.path.join(APPLE, package["path"], "Package.swift"))
    with open(PACKAGE_SWIFT, encoding="utf-8") as f:
        manifest = f.read()
    # The one library product every target links by name.
    assert re.search(r'\.library\(\s*name:\s*"UEMAKit"', manifest)
    for target in spec["targets"].values():
        assert {"package": "UEMAKit", "product": "UEMAKit"} in target["dependencies"]


def test_two_app_targets_as_the_plan_row_names_them(spec):
    targets = spec["targets"]
    assert set(targets) == set(APPS) | set(TESTS.values())
    for name, (platform, floor) in APPS.items():
        app = targets[name]
        assert app["type"] == "application"
        assert app["platform"] == platform
        assert app["deploymentTarget"] == floor
        # The shared SwiftUI sources of WI-38 and WI-32, without the test folder.
        assert app["sources"] == [{"path": "UEMAMind", "excludes": ["Tests/**"]}]
        # A scheme block makes XcodeGen write a scheme named after the target (SchemeGenerator.swift),
        # the name the acceptance line passes to xcodebuild -scheme.
        assert app["scheme"]["testTargets"] == [TESTS[name]]
        assert app["info"]["path"].startswith("Generated/")
        assert app["info"]["properties"]["CFBundleDisplayName"] == "UEMA Mind"
        assert app["settings"]["base"]["CODE_SIGN_STYLE"] == "Automatic"


def test_each_app_has_one_hosted_unit_test_bundle_with_optional_sources(spec):
    targets = spec["targets"]
    for app_name, test_name in TESTS.items():
        bundle = targets[test_name]
        platform, floor = APPS[app_name]
        assert bundle["type"] == "bundle.unit-test"
        assert bundle["platform"] == platform and bundle["deploymentTarget"] == floor
        # Optional: xcodegen generate passes before WI-32 writes apple/UEMAMind/Tests.
        assert bundle["sources"] == [{"path": "UEMAMind/Tests", "optional": True}]
        # The host app, from which XcodeGen derives TEST_HOST.
        assert {"target": app_name} in bundle["dependencies"]


def test_macos_app_signs_ad_hoc_without_sandbox_or_hardened_runtime(spec):
    base = spec["targets"]["UEMAMind-macOS"]["settings"]["base"]
    assert base["CODE_SIGN_IDENTITY"] == "-"
    # Strings, not YAML booleans, so XcodeGen writes the literal NO.
    assert base["ENABLE_APP_SANDBOX"] == "NO" and base["ENABLE_HARDENED_RUNTIME"] == "NO"
    assert "entitlements" not in spec["targets"]["UEMAMind-macOS"]
    info = spec["targets"]["UEMAMind-macOS"]["info"]["properties"]
    assert info["LSMinimumSystemVersion"] == "$(MACOSX_DEPLOYMENT_TARGET)"
    assert info["NSPrincipalClass"] == "NSApplication"
    assert info["NSHighResolutionCapable"] is True


def test_ios_app_carries_the_memory_entitlement_and_no_team(spec):
    ios = spec["targets"]["UEMAMind-iOS"]
    assert ios["entitlements"]["path"].startswith("Generated/")
    # REQ-B-15: the peak footprint is read "with the entitlement".
    assert ios["entitlements"]["properties"] == {"com.apple.developer.kernel.increased-memory-limit": True}
    assert ios["info"]["properties"]["UILaunchScreen"] == {}
    assert ios["info"]["properties"]["UIApplicationSceneManifest"] == {"UIApplicationSupportsMultipleScenes": False}
    assert ios["settings"]["base"]["TARGETED_DEVICE_FAMILY"] == "1,2"
    # The founder's team is passed on the xcodebuild line or written in by hand; never committed here.
    assert "DEVELOPMENT_TEAM" not in ios["settings"]["base"]


def test_no_direct_framework_entry_for_the_static_xcframework(spec):
    # MuJoCo reaches the app through the package's binary target (static slices, WI-18); a second
    # mention as a framework dependency is what the header of project.yml rules out.
    for target in spec["targets"].values():
        assert not any("framework" in dep for dep in target["dependencies"])


def test_generated_project_is_ignored_by_git():
    with open(GITIGNORE, encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    assert "UEMAMind.xcodeproj/" in lines


def test_no_banned_word_in_the_spec():
    if not os.path.isfile(BANNED_WORDS):
        pytest.skip("no %s (WI-21 writes it)" % BANNED_WORDS)
    with open(BANNED_WORDS, encoding="utf-8") as f:
        banned = [w.strip().lower() for w in f if w.strip()]
    with open(PROJECT_YML, encoding="utf-8") as f:
        text = f.read().lower()
    hits = [w for w in banned if re.search(r"\b%s\b" % re.escape(w), text)]
    assert hits == []
