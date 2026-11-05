// swift-tools-version: 5.9
// Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
// Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
//
// The Linux check of the Apple port: a second manifest over the same source folders (Sources,
// Tests and UEMAMind here are links to ../UEMAKit/Sources, ../UEMAKit/Tests and ../UEMAMind)
// that compiles and tests every part with no Apple framework or MLX in it, on a Linux Swift
// toolchain with the MuJoCo 3.3.0 Linux release. run.sh fetches both and runs it; README.md
// says what it covers. It does not replace the Mac build of ../UEMAKit/Package.swift.
//
// Left out, with the reason: UEMAModel, UEMAPlanner and their tests and UEMAIntegrationTests
// (MLX: mlx-swift 0.31.4 fails on Linux at MLXNN's MLXFast calls, 0.33.0 at Metal/Metal.hpp,
// both measured November 4, 2026); Vision/GeomMirror.swift, Vision/SceneKitEyeRenderer.swift
// and their tests (SceneKit, Metal); the app's views and UEMAMindApp.swift (SwiftUI);
// CMuJoCoRendererFreeTests (it walks the source tree from #filePath, which here resolves
// through the links; run.sh greps for the renderer prefix instead). VisionPatcherTests reads
// its PNGs with ImageIO, so run.sh writes a copy under Generated/ that reads the same pixels
// decoded by Python's PIL.

import Foundation
import PackageDescription

let mujoco = ProcessInfo.processInfo.environment["MUJOCO_DIR"] ?? "/opt/mujoco-3.3.0"
let mujocoInclude: [String] = ["-I", mujoco + "/include"]
let mujocoLink: [String] = ["-L", mujoco + "/lib", "-lmujoco", "-Xlinker", "-rpath", "-Xlinker", mujoco + "/lib"]
let swiftMujoco: [SwiftSetting] = [.unsafeFlags(["-Xcc", "-I", "-Xcc", mujoco + "/include"])]

var targets: [Target] = []
targets.append(.target(name: "CryptoKit", dependencies: [.product(name: "Crypto", package: "swift-crypto")],
                       path: "Shims/CryptoKit"))
targets.append(.target(name: "Combine", dependencies: [
    .product(name: "OpenCombine", package: "OpenCombine"),
    .product(name: "OpenCombineFoundation", package: "OpenCombine"),
], path: "Shims/Combine"))
targets.append(.target(name: "SwiftUI", dependencies: ["Combine"], path: "Shims/SwiftUI"))
targets.append(.target(name: "os", path: "Shims/os"))

targets.append(.target(name: "CMuJoCo", path: "Sources/CMuJoCo", publicHeadersPath: ".",
                       cSettings: [.unsafeFlags(mujocoInclude)], linkerSettings: [.unsafeFlags(mujocoLink)]))
targets.append(.target(name: "UEMACore", path: "Sources/UEMACore"))
targets.append(.target(name: "UEMAPhysics", dependencies: ["CMuJoCo", "UEMACore"], path: "Sources/UEMAPhysics",
                       swiftSettings: swiftMujoco))
targets.append(.target(name: "UEMAScene", dependencies: ["UEMACore", "CryptoKit"], path: "Sources/UEMAScene",
                       resources: [.copy("Resources")]))
targets.append(.target(name: "UEMASenses", dependencies: ["UEMACore", "UEMAPhysics", "UEMAScene", "CryptoKit"],
                       path: "Sources/UEMASenses",
                       exclude: ["Vision/GeomMirror.swift", "Vision/SceneKitEyeRenderer.swift"]))
targets.append(.target(name: "UEMATokenizer", dependencies: ["UEMACore"], path: "Sources/UEMATokenizer"))
targets.append(.target(name: "UEMACommand", dependencies: ["UEMACore", "UEMAPhysics", "UEMASenses"],
                       path: "Sources/UEMACommand"))
// The app's view models, under the module name the app's iOS tests import.
targets.append(.target(name: "UEMAMind_iOS", dependencies: ["UEMACore", "Combine", "SwiftUI", "os"],
                       path: "UEMAMind/ViewModels"))

targets.append(.testTarget(name: "CMuJoCoTests", dependencies: ["CMuJoCo"], path: "Tests/CMuJoCoTests",
                           exclude: ["CMuJoCoRendererFreeTests.swift"], swiftSettings: swiftMujoco))
targets.append(.testTarget(name: "UEMACoreTests", dependencies: ["UEMACore"], path: "Tests/UEMACoreTests"))
targets.append(.testTarget(name: "UEMAPhysicsTests", dependencies: ["UEMACore", "UEMAPhysics"],
                           path: "Tests/UEMAPhysicsTests"))
targets.append(.testTarget(name: "UEMASceneTests", dependencies: ["UEMACore", "UEMAScene"], path: "Tests/UEMASceneTests"))
targets.append(.testTarget(name: "UEMASensesTests", dependencies: ["UEMACore", "UEMAPhysics", "UEMAScene", "UEMASenses"],
                           path: "Tests/UEMASensesTests",
                           exclude: ["Vision/GeomMirrorTests.swift", "Vision/SceneKitEyeRendererTests.swift"],
                           resources: [.copy("Fixtures")]))
targets.append(.testTarget(name: "UEMATokenizerTests", dependencies: ["UEMACore", "UEMASenses", "UEMATokenizer"],
                           path: "Tests/UEMATokenizerTests", exclude: ["VisionPatcherTests.swift"]))
targets.append(.testTarget(name: "VisionPatcherLinuxTests", dependencies: ["UEMACore", "UEMATokenizer", "CryptoKit"],
                           path: "Generated/VisionPatcherLinux"))
targets.append(.testTarget(name: "UEMACommandTests", dependencies: ["UEMACore", "UEMAPhysics", "UEMASenses", "UEMACommand"],
                           path: "Tests/UEMACommandTests"))

let package = Package(
    name: "UEMALinuxCheck",
    dependencies: [
        .package(url: "https://github.com/apple/swift-crypto", from: "3.9.0"),
        .package(url: "https://github.com/OpenCombine/OpenCombine", from: "0.14.0"),
    ],
    targets: targets
)
