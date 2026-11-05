// swift-tools-version: 5.9
// UEMAKit: the Swift package of the UEMA Apple port (target B of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html, work item WI-17).
//
// WHAT THIS FILE IS. The manifest of one Swift package named UEMAKit. It declares the
// platforms, one package dependency (mlx-swift, Apple's array and neural-network library
// for Swift), one binary target (the MuJoCo 3.3.0 XCFramework that
// apple/scripts/build_mujoco_xcframework.sh writes, work item WI-18), nine library
// modules in dependency sequence, one XCTest target per module and one integration test
// target (work item WI-40). Nothing in this package was compiled on the machine that
// wrote it (Windows 11, no Swift toolchain, no Xcode: plan section 3, fact 1).
//
// MODULE SEQUENCE. Each module depends on every module before it, in this order
// (plan row WI-17: "UEMACore <- UEMAPhysics <- UEMAScene <- UEMASenses <- UEMATokenizer
// <- UEMAModel <- UEMAPlanner <- UEMACommand"); CMuJoCo, the C module over the binary
// target, sits before UEMAPhysics:
//   mujoco (binary) <- CMuJoCo (C) <- UEMACore <- UEMAPhysics <- UEMAScene <- UEMASenses
//   <- UEMATokenizer <- UEMAModel <- UEMAPlanner <- UEMACommand
// Every dependency is declared directly (not left to transitivity) so that an `import`
// of any earlier module compiles in any later one.
//
// WHICH MODULE GETS MLX. UEMATokenizer (MLX: its outputs are the encoders' inputs),
// UEMAModel (MLX, MLXNN, MLXRandom: encoders, transformer, heads, motor controller,
// wellness) and UEMAPlanner (MLX, MLXNN, MLXOptimizers, MLXRandom: Adam on the command
// through the frozen forward model, plan row WI-39). The other modules hold plain Swift
// value types.
//
// CMuJoCo IS A C-FAMILY TARGET. publicHeadersPath "." makes Sources/CMuJoCo itself the
// public headers folder, so work item WI-19's module.modulemap and shim.h sit flat in
// that folder as its row names them. SwiftPM's ModuleMapGenerator.swift (read November
// 4, 2026): "When a custom module map exists in the header directory, it is used
// as-is."; without one it writes an umbrella-directory map over the folder in both of
// its remaining branches ("if 'PublicHeadersDir' contains only header files and no
// subdirectories, use it as the umbrella directory" and, for a folder with other
// content such as this one with CMuJoCo.c in it, "an umbrella directory is still created
// for them"; each branch ends in `return .umbrellaDirectory(publicHeadersDir)`). So
// before WI-19 lands, `import CMuJoCo` compiles against an umbrella directory holding
// zero headers (an empty module); after it lands, against WI-19's module.modulemap. At
// swift-tools-version 5.9 SwiftPM stops on a folder holding both .c and .swift files
// (TargetSourcesBuilder.swift: `if sources.containsMixedLanguage && toolsVersion < .v6_5
// { throw Module.Error.mixedSources(targetPath) }`; the message, in PackageBuilder.swift:
// "contains mixed language source files; mixed language targets require a tools version
// of 6.5 or later"), so no .swift file may be added to Sources/CMuJoCo, although the
// plan's WI-19 row lists one there (CMuJoCo.swift): Swift-side accessors over the C
// symbols belong in UEMAPhysics. The placeholder CMuJoCo.c keeps the target non-empty
// (SwiftPM emits its targetHasNoSources diagnostic for a target without compiled
// sources).
//
// RESOURCES. UEMAScene bundles Sources/UEMAScene/Resources (the skin layout JSON of
// WI-13 and the crib scene bundle folder of WI-15); UEMAModel bundles
// Sources/UEMAModel/Resources (weight_manifest.json of WI-12); UEMASensesTests bundles
// Tests/UEMASensesTests/Fixtures (the eye fixtures of WI-14); UEMAModelTests bundles
// Tests/UEMAModelTests/Fixtures (the model-pass fixtures of WI-34). Each folder is
// copied as is (the `copy` rule) and read through `Bundle.module` with
// subdirectory "Resources" or "Fixtures". Each folder exists in git with at least one
// file: for a declared resource path it cannot find, SwiftPM warns (TargetSourcesBuilder.swift:
// "Invalid Resource '<path>': File not found.") and builds no resource bundle for the
// target, and without a bundle `Bundle.module` does not exist and the code that reads
// it fails to compile.
//
// THE mlx-swift VERSION. `from: "0.30.6"` is a floor, not the pin: SwiftPM takes the
// newest tag whose manifest its toolchain can read (SourceControlPackageContainer.swift:
// toolsVersionsAppropriateVersionsDescending keeps a version only when
// isToolsVersionCompatible returns true, and drops the others with no diagnostic) and
// writes that tag into Package.resolved at the first `swift package resolve`; the founder commits
// Package.resolved, and that commit is the pin the work item's instructions call "tag
// pinned at first build" (the plan page's row reads "dependency mlx-swift"). Tags and
// their commit dates read on November 4, 2026 (https://api.github.com/repos/ml-explore/mlx-swift/tags
// and each tag's commit): 0.33.0 (October 27, 2026, the newest), 0.32.3 (September 30,
// 2026), 0.31.6 (July 2, 2026), 0.30.6 (February 10, 2026). The first line of the raw
// Package.swift of each tag at or above 0.30.6, read the same day: "// swift-tools-version:
// 6.3;(experimentalCGen)" for 0.33.0, 0.32.3, 0.32.2, 0.31.6 and 0.31.5;
// "// swift-tools-version: 5.12" for 0.31.4, 0.31.3, 0.31.2, 0.31.1 and 0.30.6 (and for
// 0.30.1 to 0.30.3, below the floor). SwiftPM compares tools versions as numbers, so a
// Swift 6.3 or newer toolchain resolves 0.33.0, a Swift 6.0 to 6.2 toolchain resolves
// 0.31.4 (the newest tag whose manifest it can read), and a Swift 5.9 or 5.10 toolchain
// reads none of these manifests, so its resolution fails.
//
// SOURCES READ BEFORE WRITING (all fetched November 4, 2026), with the fact each gave:
// - https://raw.githubusercontent.com/ml-explore/mlx-swift/0.33.0/Package.swift :
//   platforms ".macOS("14.0"), .iOS(.v17), .tvOS(.v17), .visionOS(.v1)"; library
//   products "MLX", "MLXRandom", "MLXNN", "MLXOptimizers", "MLXFFT", "MLXLinalg",
//   "MLXFast"; the same seven products and platforms in the 0.32.3, 0.31.6 and 0.30.6
//   manifests.
// - https://github.com/ml-explore/mlx-swift/blob/main/README.md : the dependency line
//   '.package(url: "https://github.com/ml-explore/mlx-swift", from: "0.10.0")'; the
//   product lines '.product(name: "MLX", package: "mlx-swift")' and the same for
//   "MLXNN" and "MLXOptimizers"; "SwiftPM (command line) cannot build the Metal shaders
//   so the ultimate build has to be done via Xcode."; "Although `SwiftPM` (command line)
//   cannot build the Metal shaders, `xcodebuild` can and it can be used to do command
//   line builds".
// - https://ml-explore.github.io/mlx/build/html/install.html : "MLX is only available on
//   devices running macOS >= 14.0 and higher."; build from source needs "Xcode >= 15.0
//   and macOS SDK >= 14.0".
// - https://developer.apple.com/documentation/packagedescription/target/binarytarget(name:path:) :
//   "static func binaryTarget(name: String, path: String) -> Target"; path: "The path
//   to the binary artifact. This path can point directly to a binary artifact or to an
//   archive file that contains the binary artifact at its root."; "Binary targets are
//   only available on Apple platforms."
// - https://developer.apple.com/documentation/packagedescription/resource/copy(_:) :
//   "static func copy(_ path: String) -> Resource"; "It copies resources at the given
//   `path`, as is, to the top level in the package's resource bundle. If the given path
//   represents a directory, Swift Package Manager preserves its structure."
// - https://developer.apple.com/documentation/packagedescription/target/target(name:dependencies:path:exclude:sources:resources:publicheaderspath:csettings:cxxsettings:swiftsettings:linkersettings:plugins:) :
//   publicHeadersPath: "The directory containing public headers of a C-family library
//   target."; dependencies: "A dependency can be another target in the package or a
//   product from a package dependency."
// - https://developer.apple.com/documentation/packagedescription/target/testtarget(name:dependencies:path:exclude:sources:resources:csettings:cxxsettings:swiftsettings:linkersettings:plugins:) :
//   "static func testTarget(name: String, dependencies: [Target.Dependency] = [],
//   path: String? = nil, exclude: [String] = [], sources: [String]? = nil,
//   resources: [Resource]? = nil, ...) -> Target".
// - https://developer.apple.com/documentation/packagedescription/supportedplatform/macosversion :
//   v14 "The value that represents macOS 14.0."
// - https://developer.apple.com/documentation/packagedescription/supportedplatform/iosversion :
//   v17 "The value that represents iOS 17.0."
// - https://raw.githubusercontent.com/swiftlang/swift-package-manager/main/Sources/PackageLoading/PackageBuilder.swift :
//   the mixed-language and missing-artifact messages quoted above; a target without
//   sources gets the targetHasNoSources diagnostic.
// - https://raw.githubusercontent.com/swiftlang/swift-package-manager/main/Sources/PackageLoading/ModuleMapGenerator.swift :
//   the custom-module-map and umbrella-directory rules quoted above.
// - https://raw.githubusercontent.com/swiftlang/swift-package-manager/main/Sources/PackageLoading/TargetSourcesBuilder.swift :
//   "Invalid Resource '\(resource.path)': \(message)." with message "File not found";
//   the mixed-language check quoted above; header files inside a Swift target's
//   sources are classified as headers with no diagnostic.
// - https://raw.githubusercontent.com/swiftlang/swift-package-manager/main/Sources/Workspace/PackageContainer/SourceControlPackageContainer.swift :
//   "let isValid = await self.isToolsVersionCompatible(at: version)" / "if isValid {
//   appropriateVersions.append(version) }"; a tools version that fails validation
//   "return false" with no diagnostic.
//
// FIRST BUILD ON A MAC (this work item's acceptance; plan section 8, steps 1 to 5).
//  1. Prerequisites: a Mac with Apple silicon; Xcode with the macOS 14 and iOS 17 SDKs;
//     `swift --version` prints the toolchain's Swift version, which decides the
//     mlx-swift tag step 3 resolves (see THE mlx-swift VERSION above); network access
//     for the mlx-swift clone.
//  2. `bash apple/scripts/build_mujoco_xcframework.sh` (WI-18) has written
//     apple/UEMAKit/Binaries/mujoco.xcframework. This manifest names that path as a
//     binary target, and SwiftPM stops loading a package whose binary artifact is
//     missing (PackageBuilder.swift: "binary target 'mujoco' could not be mapped to an
//     artifact with expected name ..."), so step 3 cannot pass before step 2.
//  3. `cd apple/UEMAKit && swift package resolve` : exit code 0; the output names
//     mlx-swift and the tag it took; Package.resolved appears. Commit Package.resolved.
//     If resolution fails with no mlx-swift version in the range 0.30.6 ..< 1.0.0, the
//     toolchain reads none of the tags' manifests (SwiftPM drops a tag whose tools
//     version it cannot validate, silently; see THE mlx-swift VERSION above): update
//     Xcode and resolve again.
//  4. `xcodebuild -scheme UEMAKit-Package build -destination 'platform=macOS'` : exit
//     code 0 (Xcode's build, not `swift build`: the README line above). The scheme
//     UEMAKit-Package is the one Xcode generates for a package named UEMAKit.
//     If CMuJoCo fails with "contains mixed language source files", a .swift file was
//     added to Sources/CMuJoCo: move it to Sources/UEMAPhysics. If linking fails on OpenGL
//     symbols from the MuJoCo static library's renderer objects, add
//     `linkerSettings: [.linkedFramework("OpenGL", .when(platforms: [.macOS]))]` to
//     CMuJoCo and record the symbol names in apple/FIRST_BUILD_RECORD.md.
//  5. `xcodebuild -scheme UEMAKit-Package test -destination 'platform=macOS'` : every
//     placeholder test passes (one per test target: each Swift module's test checks its
//     module's name constant; CMuJoCoTests only runs until WI-19's mj_versionString()
//     test replaces it; UEMASceneTests, UEMASensesTests and UEMAModelTests also check
//     that one bundled file is found).
//  6. Write each step's command, exit code and clock time (AM/PM Pacific) into
//     apple/FIRST_BUILD_RECORD.md (the plan's WI-41; the specification's REQ-B-18 names
//     the same record apple/records/first_build.md).

import PackageDescription

/// The package identity of mlx-swift, the last path component of its repository URL;
/// every `.product(name:package:)` below names it.
let mlxPackage = "mlx-swift"

/// Dependency on one mlx-swift library product.
func mlx(_ product: String) -> Target.Dependency {
    .product(name: product, package: mlxPackage)
}

let package = Package(
    name: "UEMAKit",
    platforms: [
        // MLX runs on macOS 14.0 or newer (MLX install page); mlx-swift declares iOS 17.
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        // One library product that the UEMAMind app (apple/project.yml, WI-33) links.
        .library(
            name: "UEMAKit",
            targets: [
                "UEMACore",
                "UEMAPhysics",
                "UEMAScene",
                "UEMASenses",
                "UEMATokenizer",
                "UEMAModel",
                "UEMAPlanner",
                "UEMACommand",
            ]
        ),
    ],
    dependencies: [
        // The floor; Package.resolved written at the first resolve is the pin.
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.6"),
    ],
    targets: [
        // MuJoCo 3.3.0 as a static library per platform, written by
        // apple/scripts/build_mujoco_xcframework.sh (WI-18): slices macos-arm64,
        // ios-arm64 and ios-arm64-simulator. Not in git (apple/.gitignore).
        .binaryTarget(
            name: "mujoco",
            path: "Binaries/mujoco.xcframework"
        ),

        // The C module over the binary target: module.modulemap and shim.h (WI-19) sit
        // flat in Sources/CMuJoCo; CMuJoCo.c is the placeholder compiled source.
        .target(
            name: "CMuJoCo",
            dependencies: ["mujoco"],
            publicHeadersPath: "."
        ),

        // Value types and protocols shared by every module (WI-20).
        .target(
            name: "UEMACore"
        ),

        // MuJoCoPhysics over CMuJoCo (WI-25).
        .target(
            name: "UEMAPhysics",
            dependencies: ["CMuJoCo", "UEMACore"]
        ),

        // The exported scene bundle and skin layout, found in the module bundle (WI-26).
        .target(
            name: "UEMAScene",
            dependencies: ["CMuJoCo", "UEMACore", "UEMAPhysics"],
            resources: [.copy("Resources")]
        ),

        // The six sense readers (WI-27, WI-28, WI-29).
        .target(
            name: "UEMASenses",
            dependencies: ["CMuJoCo", "UEMACore", "UEMAPhysics", "UEMAScene"]
        ),

        // Raw sense arrays into encoder inputs (WI-30).
        .target(
            name: "UEMATokenizer",
            dependencies: [
                "CMuJoCo", "UEMACore", "UEMAPhysics", "UEMAScene", "UEMASenses",
                mlx("MLX"),
            ]
        ),

        // Weights loader, encoders, forward model, heads, motor controller, wellness
        // (WI-31, WI-35, WI-36).
        .target(
            name: "UEMAModel",
            dependencies: [
                "CMuJoCo", "UEMACore", "UEMAPhysics", "UEMAScene", "UEMASenses",
                "UEMATokenizer",
                mlx("MLX"), mlx("MLXNN"), mlx("MLXRandom"),
            ],
            resources: [.copy("Resources")]
        ),

        // Goal ring, planner, hand descent (WI-39).
        .target(
            name: "UEMAPlanner",
            dependencies: [
                "CMuJoCo", "UEMACore", "UEMAPhysics", "UEMAScene", "UEMASenses",
                "UEMATokenizer", "UEMAModel",
                mlx("MLX"), mlx("MLXNN"), mlx("MLXOptimizers"), mlx("MLXRandom"),
            ]
        ),

        // Command decoder and motor function (WI-37).
        .target(
            name: "UEMACommand",
            dependencies: [
                "CMuJoCo", "UEMACore", "UEMAPhysics", "UEMAScene", "UEMASenses",
                "UEMATokenizer", "UEMAModel", "UEMAPlanner",
            ]
        ),

        // One XCTest target per module.
        .testTarget(
            name: "CMuJoCoTests",
            dependencies: ["CMuJoCo"]
        ),
        .testTarget(
            name: "UEMACoreTests",
            dependencies: ["UEMACore"]
        ),
        .testTarget(
            name: "UEMAPhysicsTests",
            dependencies: ["CMuJoCo", "UEMACore", "UEMAPhysics"]
        ),
        .testTarget(
            name: "UEMASceneTests",
            dependencies: ["CMuJoCo", "UEMACore", "UEMAPhysics", "UEMAScene"]
        ),
        .testTarget(
            name: "UEMASensesTests",
            dependencies: ["CMuJoCo", "UEMACore", "UEMAPhysics", "UEMAScene", "UEMASenses"],
            resources: [.copy("Fixtures")]
        ),
        .testTarget(
            name: "UEMATokenizerTests",
            dependencies: [
                "CMuJoCo", "UEMACore", "UEMAPhysics", "UEMAScene", "UEMASenses",
                "UEMATokenizer",
                mlx("MLX"),
            ]
        ),
        .testTarget(
            name: "UEMAModelTests",
            dependencies: [
                "CMuJoCo", "UEMACore", "UEMAPhysics", "UEMAScene", "UEMASenses",
                "UEMATokenizer", "UEMAModel",
                mlx("MLX"), mlx("MLXNN"), mlx("MLXRandom"),
            ],
            resources: [.copy("Fixtures")]
        ),
        .testTarget(
            name: "UEMAPlannerTests",
            dependencies: [
                "CMuJoCo", "UEMACore", "UEMAPhysics", "UEMAScene", "UEMASenses",
                "UEMATokenizer", "UEMAModel", "UEMAPlanner",
                mlx("MLX"), mlx("MLXNN"), mlx("MLXOptimizers"), mlx("MLXRandom"),
            ]
        ),
        .testTarget(
            name: "UEMACommandTests",
            dependencies: [
                "CMuJoCo", "UEMACore", "UEMAPhysics", "UEMAScene", "UEMASenses",
                "UEMATokenizer", "UEMAModel", "UEMAPlanner", "UEMACommand",
            ]
        ),

        // The real stack for 20 model steps (WI-40, wave 5) lands here.
        .testTarget(
            name: "UEMAIntegrationTests",
            dependencies: [
                "CMuJoCo", "UEMACore", "UEMAPhysics", "UEMAScene", "UEMASenses",
                "UEMATokenizer", "UEMAModel", "UEMAPlanner", "UEMACommand",
                mlx("MLX"), mlx("MLXNN"), mlx("MLXOptimizers"), mlx("MLXRandom"),
            ]
        ),
    ]
)
