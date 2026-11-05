// SceneBundle.swift: the shipped crib scene bundle, found in this module's resource bundle and
// checked file by file against its manifest (work item WI-26 of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-2 and the
// SceneBundle row of its section 5).
//
// WHAT THIS FILE IS. The scene side of the UEMAScene module: ResourceSceneBundle locates the
// folder Resources/crib_age24_trunk_v1 that apple/scripts/export_scene_bundle.py (WI-15) wrote,
// reads its manifest bundle.json, finds the skin layout JSON of WI-13 beside it, and hands out
// file URLs. It conforms to UEMACore's SceneLoading (Sources/UEMACore/Protocols.swift:467-476)
// through this file's SceneBundle protocol, which refines it. Parsing the skin layout is
// SkinLayout.swift's.
// Nothing in this file was compiled on the machine that wrote it (no Swift toolchain; plan
// section 3, fact 1).
//
// INPUT. The module's resource bundle (Bundle.module, exposed as UEMASceneModule.resourceBundle in
// UEMAScene.swift), in which Package.swift copies Sources/UEMAScene/Resources as is (the copy rule),
// so the folder sits at Resources/crib_age24_trunk_v1 inside the resource bundle. The folder holds,
// per its manifest written at export time: scene.xml (the MuJoCo scene, 2 <include> elements and 6
// <texture file="..."> attributes, every one a bare file name), MIMo_metav2_lite.xml and
// MIMo_modelv2_grown24_fovea.xml (the two include files), play_mat.png, ceiling.png and
// pen_wall_n/s/e/w.png (the six textures) and bundle.json (for every file its name, bytes and
// SHA-256; the model counts nq 146, nv 136, nu 90, nmocap 8, nbody 76 the export compared with
// the released run's observer header: export_scene_bundle.py, docstring OPERATION step 5).
//
// OPERATION. init(name:skinLayoutName:in:) calls Bundle.url(forResource:withExtension:subdirectory:)
// for bundle.json under "Resources/<name>" and takes the folder from that URL, then calls it for
// skin_layout_age24_trunk_v1.json under "Resources";
// init(folderURL:skinLayoutURL:) decodes the manifest with JSONDecoder and refuses a manifest that
// does not list its own scene file. checkFiles() reads every listed file, counts its
// bytes and computes its SHA-256 with CryptoKit; verify() throws the first failure by file name
// (missing, byte count, hash), which is REQ-B-2's "the app checks every hash at start", then
// throws when the skin layout file cannot be read (bundle.json lists no hash for it; its identity
// check is SkinLayout.checkShippedIdentity(), the layout digest 0581069917fae467).
//
// OUTPUT. sceneXMLURL (SceneLoading's name; sceneURL is the same URL under its earlier name) and
// scenePath (the path string mj_loadXML takes; MuJoCoPhysics of WI-25 is the consumer),
// skinLayoutURL (the tactile reader of WI-28 and SkinLayout.load(from:) take it), includeURLs and
// textureURLs (the eye renderer of WI-29 reads the floor texture from
// textureURLs), the decoded manifest, and the per-file check records.
//
// SOURCES READ (all fetched November 4, 2026 through the documentation JSON at
// developer.apple.com/tutorials/data/documentation/...):
// - https://developer.apple.com/documentation/foundation/bundle/url(forresource:withextension:subdirectory:) :
//   "func url(forResource name: String?, withExtension ext: String?, subdirectory subpath:
//   String?) -> URL?"; returns "The file URL for the resource file or nil if the file could not
//   be located."; macOS 10.6+, iOS 4.0+.
// - https://developer.apple.com/documentation/xcode/bundling-resources-with-a-swift-package :
//   "Always use Bundle.module when you access resources."; the copy rule has "Xcode copy the
//   resource as is to the top level of the resource bundle. If you pass a directory path to the
//   copy rule, Xcode retains the directory's structure."
// - https://developer.apple.com/documentation/foundation/data/init(contentsof:options:) :
//   "init(contentsOf url: URL, options: Data.ReadingOptions = []) throws"; throws "An error in
//   the Cocoa domain, if url cannot be read."; macOS 10.10+, iOS 8.0+.
// - https://developer.apple.com/documentation/foundation/jsondecoder/decode(_:from:) :
//   "func decode<T>(_ type: T.Type, from data: Data) throws -> T where T : Decodable".
// - https://developer.apple.com/documentation/cryptokit/sha256 : "struct SHA256", "An
//   implementation of Secure Hashing Algorithm 2 (SHA-2) hashing with a 256-bit digest.";
//   macOS 10.15+, iOS 13.0+ (both below this package's macOS 14 and iOS 17 floors).
// - https://developer.apple.com/documentation/cryptokit/hashfunction/hash(data:) : "static func
//   hash<D>(data: D) -> Self.Digest where D : DataProtocol"; "Computes the digest of the bytes in
//   the given data instance and returns the computed digest."
// - https://developer.apple.com/documentation/cryptokit/sha256digest : "struct SHA256Digest"
//   conforms to Digest, Sequence (of UInt8) and ContiguousBytes; "static var byteCount: Int".
// - apple/scripts/export_scene_bundle.py (read entire): file_record() writes {"name", "bytes",
//   "sha256"} per file; export() writes the manifest keys bundle, created, generator,
//   configuration, sources, not_shipped, files, scene_file, includes, textures,
//   play_mat_size_px, model_counts, observer_header, observer_header_match, reference_check and
//   fresh_folder_load. Only the keys this file reads are decoded; the rest stay in the JSON.
//
// FIRST BUILD ON A MAC (this work item's acceptance: "scene.xml and the skin JSON found; 4,683
// points parsed"; plan section 8, step 4).
//  1. Package.swift's steps 1 to 4 (the XCFramework of WI-18, `swift package resolve`, the
//     xcodebuild build) have passed; this module adds no dependency.
//  2. `xcodebuild -scheme UEMAKit-Package test -destination 'platform=macOS'
//     -only-testing:UEMASceneTests` : exit code 0. The tests of Tests/UEMASceneTests/
//     SceneBundleTests.swift (14 tests) check that the bundle folder is found, that scene.xml
//     exists at sceneXMLURL, that the manifest lists 9 files, 2 includes and 6 textures, that
//     every file's byte count and SHA-256 equal the manifest's, that the model counts read nq
//     146, that a copy with one changed byte is refused by file name, that the bundle serves
//     UEMACore's SceneLoading (scene.xml and skin_layout_age24_trunk_v1.json found and present),
//     that the JSON at skinLayoutURL parses to 4,683 points with digest 0581069917fae467, and
//     that an absent skin layout file is reported by name; SkinLayoutTests.swift checks the skin
//     side (its pass lines are in SkinLayout.swift's header).
//  2a. If step 2 fails to compile in this file: the constructs to check first are the protocol
//     refinement `protocol SceneBundle: SceneLoading` (its three inherited members are
//     implemented by ResourceSceneBundle's `sceneXMLURL`, `skinLayoutURL` and
//     `url(forFile:)`), and `@discardableResult` on `verify()`, which the protocol declares
//     without the attribute. A compile error naming a SceneLoading requirement means this file
//     is out of date with Protocols.swift, and Protocols.swift wins.
//  3. If the bundle folder or the skin JSON is not found: the test's failure message names the
//     file and the subdirectory asked for ("bundle.json" under "Resources/crib_age24_trunk_v1",
//     or "skin_layout_age24_trunk_v1.json" under "Resources"). Check that Package.swift still declares
//     `resources: [.copy("Resources")]` on UEMAScene and that the folder is in git with its 10
//     files; if the resource bundle lays the folder out elsewhere, print
//     `UEMASceneModule.resourceBundle.resourceURL` in the failing test and record the layout in
//     apple/FIRST_BUILD_RECORD.md (WI-41).
//  4. If a hash check fails on the Mac but not here: git's text conversion is the first suspect
//     (`git ls-files --eol` on November 4, 2026 read i/lf w/lf attr/-text for the three XML
//     files: the -text attribute turns line-ending conversion off; the export measured 22,906 bytes for
//     scene.xml on November 4, 2026). Run `git ls-files --eol` on the folder and compare the
//     working-tree byte counts with bundle.json's "bytes" values before changing any code.
//  5. Record the step's command, exit code and clock time (AM/PM Pacific) in
//     apple/FIRST_BUILD_RECORD.md.

import CryptoKit
import Foundation
import UEMACore

/// The decoded part of bundle.json, the manifest apple/scripts/export_scene_bundle.py writes
/// beside the scene files (its export() builds the dictionary; file_record() the per-file rows).
/// Keys the module does not read (generator, configuration, sources, not_shipped,
/// play_mat_size_px, observer_header, observer_header_match, reference_check,
/// fresh_folder_load) are left undecoded.
public struct SceneBundleManifest: Decodable, Equatable {
    /// One file of the bundle: its name, its size in bytes and its SHA-256 as 64 lowercase hex
    /// characters (export_scene_bundle.py, file_record()).
    public struct FileRecord: Decodable, Equatable {
        /// The file name inside the bundle folder, for example "scene.xml".
        public let name: String
        /// The file's size in bytes at export time.
        public let bytes: Int
        /// The SHA-256 of the file's bytes at export time, 64 lowercase hex characters.
        public let sha256: String

        /// Creates one record (tests build manifests by hand with it).
        public init(name: String, bytes: Int, sha256: String) {
            self.name = name
            self.bytes = bytes
            self.sha256 = sha256
        }
    }

    /// The MuJoCo model's size fields the export read after loading the bundle from a fresh
    /// folder (export_scene_bundle.py, model_counts()): nq generalized coordinates, nv degrees of
    /// freedom, nu actuators, nmocap mocap bodies, nbody bodies, njnt joints, ngeom geoms, ncam
    /// cameras, ntex textures, nmat materials. The released run's observer header holds nq 146,
    /// nv 136, nu 90, nmocap 8, nbody 76 (export docstring, OPERATION step 5).
    public struct ModelCounts: Decodable, Equatable {
        public let nq: Int
        public let nv: Int
        public let nu: Int
        public let nmocap: Int
        public let nbody: Int
        public let njnt: Int
        public let ngeom: Int
        public let ncam: Int
        public let ntex: Int
        public let nmat: Int
    }

    /// The bundle's name, "crib_age24_trunk_v1" for the shipped folder (BUNDLE_NAME in the export).
    public let bundle: String
    /// The export's clock time, as the export wrote it (time.strftime("%Y-%m-%dT%H:%M:%S%z")).
    public let created: String
    /// The scene file mj_loadXML opens, "scene.xml" (SCENE_NAME in the export).
    public let sceneFile: String
    /// The two include files scene.xml names, in the export's order (META_NAME, BODY_NAME).
    public let includes: [String]
    /// The six texture files scene.xml names, in the export's order (TEXTURE_NAMES).
    public let textures: [String]
    /// Every shipped file except the manifest itself: 3 XML files then 6 PNG files.
    public let files: [FileRecord]
    /// The model counts measured on the fresh-folder load.
    public let modelCounts: ModelCounts

    private enum CodingKeys: String, CodingKey {
        case bundle
        case created
        case sceneFile = "scene_file"
        case includes
        case textures
        case files
        case modelCounts = "model_counts"
    }
}

/// Why a scene bundle could not be located, read or trusted. Every case names the file or
/// folder concerned so a first-build failure message says which one.
public enum SceneBundleError: Error, Equatable, CustomStringConvertible {
    /// `Bundle.url(forResource:withExtension:subdirectory:)` returned nil for the manifest.
    case resourceNotFound(resource: String, subdirectory: String)
    /// bundle.json could not be read or decoded; `reason` is the underlying error's text.
    case manifestUnreadable(path: String, reason: String)
    /// The manifest's "files" list does not contain its own "scene_file".
    case sceneFileNotListed(sceneFile: String)
    /// A file the manifest lists is not in the folder (or not readable).
    case fileMissing(name: String, path: String)
    /// A listed file's byte count differs from the manifest's.
    case byteCountMismatch(name: String, manifestBytes: Int, fileBytes: Int)
    /// A listed file's SHA-256 differs from the manifest's.
    case sha256Mismatch(name: String, manifestSHA256: String, fileSHA256: String)

    public var description: String {
        switch self {
        case let .resourceNotFound(resource, subdirectory):
            return "scene bundle: \(resource) not found under \(subdirectory) in the module's resource bundle"
        case let .manifestUnreadable(path, reason):
            return "scene bundle: manifest \(path) unreadable: \(reason)"
        case let .sceneFileNotListed(sceneFile):
            return "scene bundle: the manifest's files list does not contain its scene file \(sceneFile)"
        case let .fileMissing(name, path):
            return "scene bundle: listed file \(name) missing at \(path)"
        case let .byteCountMismatch(name, manifestBytes, fileBytes):
            return "scene bundle: \(name) holds \(fileBytes) bytes, the manifest says \(manifestBytes)"
        case let .sha256Mismatch(name, manifestSHA256, fileSHA256):
            return "scene bundle: \(name) has SHA-256 \(fileSHA256), the manifest says \(manifestSHA256)"
        }
    }
}

/// The result of checking one listed file against its manifest record.
public struct SceneFileCheck: Equatable {
    /// The file name from the manifest.
    public let name: String
    /// The byte count the manifest records.
    public let manifestBytes: Int
    /// The byte count read from the folder, or nil when the file could not be read.
    public let fileBytes: Int?
    /// The SHA-256 the manifest records, lowercase hex.
    public let manifestSHA256: String
    /// The SHA-256 computed from the folder's file, lowercase hex, or nil when it could not be read.
    public let fileSHA256: String?

    /// True when the file was read and both its byte count and its SHA-256 equal the manifest's.
    public var passes: Bool {
        fileBytes == manifestBytes && fileSHA256 == manifestSHA256.lowercased()
    }
}

/// What the physics (MuJoCoPhysics, WI-25), the tactile reader (WI-28) and the eye renderer
/// (WI-29) take from a scene bundle: file URLs by name and the manifest check (specification
/// section 5, row "SceneBundle: Files by name with hash check (REQ-B-2); ResourceSceneBundle;
/// injected into MuJoCoPhysics, SceneRenderer"). It refines UEMACore's `SceneLoading`
/// (Sources/UEMACore/Protocols.swift:467-476: `sceneXMLURL`, `skinLayoutURL`,
/// `url(forFile:)`), so a user that needs only those three members takes a `SceneLoading`
/// and one that also needs the manifest, the include and texture URLs or the start-up check
/// takes a `SceneBundle`. Users are injected with a value of either protocol type, never with a
/// folder path they resolve themselves.
public protocol SceneBundle: SceneLoading {
    /// The bundle's name, "crib_age24_trunk_v1" for the shipped folder.
    var name: String { get }
    /// The folder that holds the scene files and the manifest.
    var folderURL: URL { get }
    /// `sceneXMLURL.path`, the C string argument of mj_loadXML.
    var scenePath: String { get }
    /// The include files scene.xml names, in the manifest's order.
    var includeURLs: [URL] { get }
    /// The texture files scene.xml names, in the manifest's order (the floor texture first).
    var textureURLs: [URL] { get }
    /// The decoded manifest: every file's name, bytes and SHA-256, and the model counts.
    var manifest: SceneBundleManifest { get }
    /// The REQ-B-2 start-up check: every listed file present with the manifest's byte count
    /// and SHA-256, and the skin layout file present; throws the first failure by file name.
    func verify() throws -> [SceneFileCheck]
}

/// The shipped scene bundle folder, located in the module's resource bundle and described by
/// its manifest, plus the URL of the skin layout JSON beside it. One responsibility: locate the
/// folder, its files and the skin layout file and check them against the manifest. It opens no
/// MuJoCo model (that is MuJoCoPhysics, WI-25), parses no skin layout (SkinLayout.swift) and
/// renders nothing (WI-29); each takes the URLs this type gives it.
public struct ResourceSceneBundle: SceneBundle, Equatable {
    /// The shipped folder's name (export_scene_bundle.py BUNDLE_NAME): crib scene, body age 24
    /// months, body layout trunk_v1, the released run's configuration.
    public static let shippedName = "crib_age24_trunk_v1"
    /// The top-level folder the copy rule places in the resource bundle (Package.swift,
    /// `resources: [.copy("Resources")]` on UEMAScene).
    public static let resourcesSubdirectory = "Resources"
    /// The manifest's file name (export_scene_bundle.py MANIFEST_NAME).
    public static let manifestFileName = "bundle.json"

    /// The folder that holds the scene files and the manifest.
    public let folderURL: URL
    /// The decoded manifest.
    public let manifest: SceneBundleManifest
    /// The skin layout JSON of 4,683 points (WI-13), the file the tactile reader (WI-28) and
    /// `SkinLayout.load(from:)` take (`SceneLoading.skinLayoutURL`). In the module's resource
    /// bundle it sits beside the scene folder, at Resources/skin_layout_age24_trunk_v1.json;
    /// bundle.json does not list it (apple/scripts/export_skin_layout.py writes no manifest row
    /// for it), so its checks are presence in `verify()` and the layout digest in
    /// `SkinLayout.checkShippedIdentity()`.
    public let skinLayoutURL: URL

    /// The bundle's name, read from the manifest.
    public var name: String { manifest.bundle }

    /// The scene file mj_loadXML opens (`SceneLoading.sceneXMLURL`); its <include> and <texture>
    /// file attributes are bare names, resolved by MuJoCo against this file's folder.
    public var sceneXMLURL: URL {
        folderURL.appendingPathComponent(manifest.sceneFile, isDirectory: false)
    }

    /// The same URL as `sceneXMLURL`, under the name this type gave it before UEMACore's
    /// `SceneLoading` existed; kept because Tests/UEMAIntegrationTests/OneModelStepTests.swift
    /// (line 712) calls `physics.load(sceneXML: scene.sceneURL)`.
    public var sceneURL: URL { sceneXMLURL }

    /// `sceneXMLURL.path`, the C string argument of mj_loadXML (MuJoCoPhysics, WI-25).
    public var scenePath: String { sceneXMLURL.path }

    /// The two include files, in the manifest's order.
    public var includeURLs: [URL] {
        manifest.includes.map { folderURL.appendingPathComponent($0, isDirectory: false) }
    }

    /// The six texture files, in the manifest's order (play_mat.png first: the floor texture the
    /// eye renderer of WI-29 draws).
    public var textureURLs: [URL] {
        manifest.textures.map { folderURL.appendingPathComponent($0, isDirectory: false) }
    }

    /// The manifest file itself.
    public var manifestURL: URL {
        folderURL.appendingPathComponent(ResourceSceneBundle.manifestFileName, isDirectory: false)
    }

    /// Locates the folder `Resources/<name>` in a resource bundle by asking for its bundle.json,
    /// and the skin layout JSON `Resources/<skinLayoutName>.json` beside it.
    /// - Parameters:
    ///   - name: the bundle folder's name; the shipped one by default.
    ///   - skinLayoutName: the skin layout file's name without its ".json" extension; the shipped
    ///     one (`SkinLayout.shippedResourceName`) by default.
    ///   - bundle: the resource bundle to search; this module's by default.
    /// - Throws: `SceneBundleError.resourceNotFound` when the bundle has no such manifest (checked
    ///   first) or no such skin layout file, and the errors of `init(folderURL:skinLayoutURL:)`.
    public init(name: String = ResourceSceneBundle.shippedName,
                skinLayoutName: String = SkinLayout.shippedResourceName,
                in bundle: Bundle = UEMASceneModule.resourceBundle) throws {
        let subdirectory = "\(ResourceSceneBundle.resourcesSubdirectory)/\(name)"
        guard let manifestURL = bundle.url(forResource: "bundle", withExtension: "json",
                                           subdirectory: subdirectory) else {
            throw SceneBundleError.resourceNotFound(resource: ResourceSceneBundle.manifestFileName,
                                                    subdirectory: subdirectory)
        }
        guard let skinURL = bundle.url(forResource: skinLayoutName, withExtension: "json",
                                       subdirectory: ResourceSceneBundle.resourcesSubdirectory) else {
            throw SceneBundleError.resourceNotFound(resource: "\(skinLayoutName).json",
                                                    subdirectory: ResourceSceneBundle.resourcesSubdirectory)
        }
        try self.init(folderURL: manifestURL.deletingLastPathComponent(), skinLayoutURL: skinURL)
    }

    /// Opens a bundle folder anywhere on disk (tests copy the shipped folder and change a byte).
    /// - Parameters:
    ///   - folderURL: a folder holding bundle.json and the files it lists.
    ///   - skinLayoutURL: the skin layout JSON this bundle hands out as `skinLayoutURL`; its
    ///     presence is checked by `verify()`, not here.
    /// - Throws: `SceneBundleError.manifestUnreadable` when bundle.json cannot be read or
    ///   decoded; `SceneBundleError.sceneFileNotListed` when the manifest does not list its own
    ///   scene file.
    public init(folderURL: URL, skinLayoutURL: URL) throws {
        let manifestURL = folderURL.appendingPathComponent(ResourceSceneBundle.manifestFileName,
                                                           isDirectory: false)
        let data: Data
        do {
            data = try Data(contentsOf: manifestURL)
        } catch {
            throw SceneBundleError.manifestUnreadable(path: manifestURL.path,
                                                      reason: String(describing: error))
        }
        let decoded: SceneBundleManifest
        do {
            decoded = try JSONDecoder().decode(SceneBundleManifest.self, from: data)
        } catch {
            throw SceneBundleError.manifestUnreadable(path: manifestURL.path,
                                                      reason: String(describing: error))
        }
        guard decoded.files.contains(where: { $0.name == decoded.sceneFile }) else {
            throw SceneBundleError.sceneFileNotListed(sceneFile: decoded.sceneFile)
        }
        self.folderURL = folderURL.standardizedFileURL
        self.manifest = decoded
        self.skinLayoutURL = skinLayoutURL.standardizedFileURL
    }

    /// The URL of one file this bundle hands out (`SceneLoading.url(forFile:)`): a file the
    /// manifest lists, inside the bundle folder, or the skin layout JSON by its file name
    /// ("skin_layout_age24_trunk_v1.json" shipped), which gives `skinLayoutURL`.
    /// - Throws: `SceneBundleError.fileMissing` when `name` is neither.
    public func url(forFile name: String) throws -> URL {
        if manifest.files.contains(where: { $0.name == name }) {
            return folderURL.appendingPathComponent(name, isDirectory: false)
        }
        if name == skinLayoutURL.lastPathComponent {
            return skinLayoutURL
        }
        throw SceneBundleError.fileMissing(
            name: name, path: folderURL.appendingPathComponent(name, isDirectory: false).path)
    }

    /// Reads every listed file and compares its byte count and SHA-256 with the manifest.
    /// Never throws: a file that cannot be read gives a record with nil `fileBytes` and
    /// `fileSHA256`. Cost: one read of each file (the shipped nine total 239,628 bytes, the sum
    /// of the manifest's "bytes" values in the bundle.json created 2026-11-04T17:35:23-0800).
    public func checkFiles() -> [SceneFileCheck] {
        manifest.files.map { (record: SceneBundleManifest.FileRecord) -> SceneFileCheck in
            let url = folderURL.appendingPathComponent(record.name, isDirectory: false)
            guard let data = try? Data(contentsOf: url) else {
                return SceneFileCheck(name: record.name, manifestBytes: record.bytes, fileBytes: nil,
                                      manifestSHA256: record.sha256, fileSHA256: nil)
            }
            return SceneFileCheck(name: record.name, manifestBytes: record.bytes,
                                  fileBytes: data.count, manifestSHA256: record.sha256,
                                  fileSHA256: ResourceSceneBundle.sha256Hex(of: data))
        }
    }

    /// REQ-B-2's start-up check: every listed file present with the manifest's byte count and
    /// SHA-256, then the skin layout file readable at `skinLayoutURL`. Returns the per-file
    /// records of the listed files (9 shipped) when all pass.
    /// - Throws: the first failure in manifest order, as `SceneBundleError.fileMissing`,
    ///   `.byteCountMismatch` or `.sha256Mismatch`, each naming the file; then
    ///   `.fileMissing` naming the skin layout file when it cannot be read.
    @discardableResult
    public func verify() throws -> [SceneFileCheck] {
        let checks = checkFiles()
        for check in checks {
            guard let fileBytes = check.fileBytes, let fileSHA256 = check.fileSHA256 else {
                throw SceneBundleError.fileMissing(
                    name: check.name,
                    path: folderURL.appendingPathComponent(check.name, isDirectory: false).path)
            }
            if fileBytes != check.manifestBytes {
                throw SceneBundleError.byteCountMismatch(name: check.name,
                                                         manifestBytes: check.manifestBytes,
                                                         fileBytes: fileBytes)
            }
            if fileSHA256 != check.manifestSHA256.lowercased() {
                throw SceneBundleError.sha256Mismatch(name: check.name,
                                                      manifestSHA256: check.manifestSHA256.lowercased(),
                                                      fileSHA256: fileSHA256)
            }
        }
        guard FileManager.default.isReadableFile(atPath: skinLayoutURL.path) else {
            throw SceneBundleError.fileMissing(name: skinLayoutURL.lastPathComponent,
                                               path: skinLayoutURL.path)
        }
        return checks
    }

    /// The SHA-256 of `data` as 64 lowercase hex characters, the form Python's
    /// hashlib.sha256(...).hexdigest() writes into the manifest.
    public static func sha256Hex(of data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
