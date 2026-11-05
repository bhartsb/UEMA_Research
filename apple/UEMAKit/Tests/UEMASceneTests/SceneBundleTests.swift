// SceneBundleTests.swift: the tests of ResourceSceneBundle (work item WI-26; specification
// REQ-B-2 and T-B-2 "Bundle hashes: all equal"). Pass lines in the file header of
// Sources/UEMAScene/SceneBundle.swift. Nothing here was compiled on the machine that wrote it.
//
// The expected values below are read from the shipped manifest
// Sources/UEMAScene/Resources/crib_age24_trunk_v1/bundle.json (created 2026-11-04T17:35:23-0800)
// and from apple/scripts/export_scene_bundle.py's constants (SCENE_NAME, META_NAME, BODY_NAME,
// TEXTURE_NAMES, HEADER_COUNT_FIELDS); the model counts are the released run's observer header
// values the export compared against (nq 146, nv 136, nu 90, nmocap 8, nbody 76).

import Foundation
import XCTest
import UEMACore
import UEMAScene

final class SceneBundleTests: XCTestCase {
    /// The scene file and its 8 companions named by the export (export_scene_bundle.py).
    private let expectedIncludes = ["MIMo_metav2_lite.xml", "MIMo_modelv2_grown24_fovea.xml"]
    private let expectedTextures = ["play_mat.png", "ceiling.png", "pen_wall_n.png", "pen_wall_s.png",
                                    "pen_wall_e.png", "pen_wall_w.png"]

    /// The shipped folder is found in the module's resource bundle and scene.xml exists at
    /// sceneXMLURL.
    func testShippedBundleIsFoundAndSceneXMLExists() throws {
        let bundle = try ResourceSceneBundle()
        XCTAssertEqual(bundle.name, ResourceSceneBundle.shippedName)
        XCTAssertEqual(bundle.manifest.sceneFile, "scene.xml")
        XCTAssertTrue(FileManager.default.fileExists(atPath: bundle.scenePath),
                      "scene.xml not at \(bundle.scenePath)")
        XCTAssertTrue(bundle.scenePath.hasSuffix("crib_age24_trunk_v1/scene.xml"), bundle.scenePath)
        XCTAssertTrue(FileManager.default.fileExists(atPath: bundle.manifestURL.path))
    }

    /// The manifest lists 9 files (3 XML, 6 PNG), the 2 includes and the 6 textures, by name.
    func testManifestListsNineFilesTwoIncludesSixTextures() throws {
        let bundle = try ResourceSceneBundle()
        let names = bundle.manifest.files.map { $0.name }
        XCTAssertEqual(names.count, 9, "\(names)")
        XCTAssertEqual(Set(names), Set(["scene.xml"] + expectedIncludes + expectedTextures))
        XCTAssertEqual(bundle.manifest.includes, expectedIncludes)
        XCTAssertEqual(bundle.manifest.textures, expectedTextures)
        XCTAssertEqual(bundle.includeURLs.map { $0.lastPathComponent }, expectedIncludes)
        XCTAssertEqual(bundle.textureURLs.map { $0.lastPathComponent }, expectedTextures)
        for name in names {
            XCTAssertTrue(FileManager.default.fileExists(atPath: try bundle.url(forFile: name).path),
                          "\(name) missing from \(bundle.folderURL.path)")
        }
        XCTAssertThrowsError(try bundle.url(forFile: "not_in_the_manifest.png"))
    }

    /// Every listed file's byte count and SHA-256 equal the manifest's (T-B-2).
    func testEveryFileHashAndByteCountMatchTheManifest() throws {
        let bundle = try ResourceSceneBundle()
        let checks = try bundle.verify()
        XCTAssertEqual(checks.count, 9)
        for check in checks {
            XCTAssertTrue(check.passes, "\(check.name): bytes \(String(describing: check.fileBytes)) vs \(check.manifestBytes), sha256 \(String(describing: check.fileSHA256)) vs \(check.manifestSHA256)")
            XCTAssertEqual(check.manifestSHA256.count, 64, check.name)
        }
    }

    /// The model counts the export measured on a fresh-folder load equal the released run's
    /// observer header (export_scene_bundle.py HEADER_COUNT_FIELDS; WI-15's pass line nq 146).
    func testModelCountsEqualTheObserverHeader() throws {
        let counts = try ResourceSceneBundle().manifest.modelCounts
        XCTAssertEqual(counts.nq, 146)
        XCTAssertEqual(counts.nv, 136)
        XCTAssertEqual(counts.nu, 90)
        XCTAssertEqual(counts.nmocap, 8)
        XCTAssertEqual(counts.nbody, 76)
    }

    /// scene.xml names its 2 include files and 6 texture files by bare name: no drive letter, no
    /// slash, no parent step (the export's OPERATION step 4 and check 5).
    func testSceneXMLNamesIncludesAndTexturesByBareName() throws {
        let bundle = try ResourceSceneBundle()
        let xml = try String(contentsOf: bundle.sceneXMLURL, encoding: .utf8)
        let values = fileAttributeValues(in: xml)
        XCTAssertEqual(values.count, 8, "\(values)")
        XCTAssertEqual(Set(values), Set(expectedIncludes + expectedTextures))
        for value in values {
            XCTAssertFalse(value.contains("/") || value.contains("\\") || value.contains(":"), value)
        }
        XCTAssertTrue(xml.contains("<mujoco model=\"MIMo_bridge_crib\">"), "scene root element not found")
    }

    /// A copy of the folder with one byte of pen_wall_s.png changed is refused by that name.
    func testCopyWithOneChangedByteIsRefusedByName() throws {
        let copy = try copyOfShippedFolder()
        defer { try? FileManager.default.removeItem(at: copy) }
        let target = copy.appendingPathComponent("pen_wall_s.png", isDirectory: false)
        var data = try Data(contentsOf: target)
        let last = data.index(before: data.endIndex)
        data[last] = data[last] ^ 0xFF
        try data.write(to: target)
        let tampered = try ResourceSceneBundle(folderURL: copy, skinLayoutURL: shippedSkinURL())
        XCTAssertThrowsError(try tampered.verify()) { error in
            guard let bundleError = error as? SceneBundleError,
                  case let .sha256Mismatch(name, manifestSHA256, fileSHA256) = bundleError else {
                return XCTFail("expected sha256Mismatch, got \(error)")
            }
            XCTAssertEqual(name, "pen_wall_s.png")
            XCTAssertNotEqual(manifestSHA256, fileSHA256)
        }
        let failing = tampered.checkFiles().filter { !$0.passes }.map { $0.name }
        XCTAssertEqual(failing, ["pen_wall_s.png"])
    }

    /// A copy with scene.xml cut to 100 bytes is refused as a byte-count mismatch; a copy without
    /// ceiling.png is refused as a missing file.
    func testTruncatedAndMissingFilesAreRefusedByName() throws {
        let copy = try copyOfShippedFolder()
        defer { try? FileManager.default.removeItem(at: copy) }
        let scene = copy.appendingPathComponent("scene.xml", isDirectory: false)
        let data = try Data(contentsOf: scene)
        try data.prefix(100).write(to: scene)
        XCTAssertThrowsError(try ResourceSceneBundle(folderURL: copy, skinLayoutURL: shippedSkinURL()).verify()) { error in
            guard let bundleError = error as? SceneBundleError,
                  case let .byteCountMismatch(name, manifestBytes, fileBytes) = bundleError else {
                return XCTFail("expected byteCountMismatch, got \(error)")
            }
            XCTAssertEqual(name, "scene.xml")
            XCTAssertEqual(fileBytes, 100)
            XCTAssertGreaterThan(manifestBytes, 100)
        }
        try data.write(to: scene)
        XCTAssertNoThrow(try ResourceSceneBundle(folderURL: copy, skinLayoutURL: shippedSkinURL()).verify(), "the restored copy must pass")
        try FileManager.default.removeItem(at: copy.appendingPathComponent("ceiling.png", isDirectory: false))
        XCTAssertThrowsError(try ResourceSceneBundle(folderURL: copy, skinLayoutURL: shippedSkinURL()).verify()) { error in
            guard let bundleError = error as? SceneBundleError,
                  case let .fileMissing(name, _) = bundleError else {
                return XCTFail("expected fileMissing, got \(error)")
            }
            XCTAssertEqual(name, "ceiling.png")
        }
    }

    /// A manifest whose files list lacks its own scene file is refused at open.
    func testManifestWithoutItsSceneFileIsRefused() throws {
        let folder = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let json = """
        {"bundle": "x", "created": "t", "scene_file": "scene.xml", "includes": [], "textures": [],
         "files": [{"name": "other.xml", "bytes": 1, "sha256": "00"}],
         "model_counts": {"nq": 1, "nv": 1, "nu": 1, "nmocap": 0, "nbody": 1, "njnt": 1, "ngeom": 1,
                          "ncam": 0, "ntex": 0, "nmat": 0}}
        """
        try Data(json.utf8).write(to: folder.appendingPathComponent("bundle.json", isDirectory: false))
        XCTAssertThrowsError(try ResourceSceneBundle(folderURL: folder, skinLayoutURL: shippedSkinURL())) { error in
            guard let bundleError = error as? SceneBundleError,
                  case let .sceneFileNotListed(sceneFile) = bundleError else {
                return XCTFail("expected sceneFileNotListed, got \(error)")
            }
            XCTAssertEqual(sceneFile, "scene.xml")
        }
        XCTAssertThrowsError(try ResourceSceneBundle(folderURL: folder.appendingPathComponent("absent"),
                                                     skinLayoutURL: shippedSkinURL())) { error in
            guard let bundleError = error as? SceneBundleError,
                  case .manifestUnreadable = bundleError else {
                return XCTFail("expected manifestUnreadable, got \(error)")
            }
        }
    }

    /// The hex form equals Python's hashlib.sha256(...).hexdigest() on the FIPS 180-2 example
    /// message "abc" (appendix B.1 of the standard).
    func testSHA256HexMatchesTheStandardExample() {
        XCTAssertEqual(ResourceSceneBundle.sha256Hex(of: Data("abc".utf8)),
                       "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }

    /// The users of the bundle take it through the SceneBundle protocol (SOLID: dependency
    /// inversion); the shipped type satisfies every requirement.
    func testShippedBundleServesTheSceneBundleProtocol() throws {
        let bundle: SceneBundle = try ResourceSceneBundle()
        XCTAssertEqual(bundle.name, "crib_age24_trunk_v1")
        XCTAssertEqual(bundle.sceneXMLURL.lastPathComponent, "scene.xml")
        XCTAssertEqual(bundle.scenePath, bundle.sceneXMLURL.path)
        XCTAssertEqual(bundle.includeURLs.count, 2)
        XCTAssertEqual(bundle.textureURLs.count, 6)
        XCTAssertEqual(try bundle.url(forFile: "scene.xml"), bundle.sceneXMLURL)
        XCTAssertEqual(try bundle.verify().count, 9)
    }

    /// The physics (WI-25) and the tactile reader (WI-28) take the bundle through UEMACore's
    /// SceneLoading (Protocols.swift:467-476): scene.xml and the skin layout JSON are found,
    /// both exist on disk, and url(forFile:) gives the same two URLs by file name.
    func testShippedBundleServesTheSceneLoadingProtocol() throws {
        let scene: SceneLoading = try ResourceSceneBundle()
        XCTAssertEqual(scene.sceneXMLURL.lastPathComponent, "scene.xml")
        XCTAssertEqual(scene.skinLayoutURL.lastPathComponent, "skin_layout_age24_trunk_v1.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: scene.sceneXMLURL.path),
                      "scene.xml not at \(scene.sceneXMLURL.path)")
        XCTAssertTrue(FileManager.default.fileExists(atPath: scene.skinLayoutURL.path),
                      "skin layout JSON not at \(scene.skinLayoutURL.path)")
        XCTAssertEqual(try scene.url(forFile: "scene.xml"), scene.sceneXMLURL)
        XCTAssertEqual(try scene.url(forFile: "skin_layout_age24_trunk_v1.json"), scene.skinLayoutURL)
        XCTAssertThrowsError(try scene.url(forFile: "not_in_the_manifest.png"))
    }

    /// The work item's acceptance line, end to end: scene.xml and the skin JSON found through
    /// the bundle, and the JSON at skinLayoutURL parses to 4,683 points with the shipped digest
    /// 0581069917fae467 (touch_inside_rays.py:24-25).
    func testSkinLayoutURLParsesFourThousandSixHundredEightyThreePoints() throws {
        let scene = try ResourceSceneBundle()
        XCTAssertEqual(scene.skinLayoutURL, try SkinLayout.shippedURL().standardizedFileURL)
        let layout = try SkinLayout.load(from: scene.skinLayoutURL)
        XCTAssertEqual(layout.pointCount, 4683)
        XCTAssertEqual(layout.pointPositions.count, 4683)
        XCTAssertNoThrow(try layout.checkShippedIdentity())
    }

    /// A skin layout name the resource bundle does not hold is reported by file name; a bundle
    /// opened with an absent skin layout file passes the 9 manifest files and then fails
    /// verify() naming the skin file.
    func testAbsentSkinLayoutIsReportedByName() throws {
        XCTAssertThrowsError(try ResourceSceneBundle(skinLayoutName: "no_such_skin")) { error in
            guard let bundleError = error as? SceneBundleError,
                  case let .resourceNotFound(resource, subdirectory) = bundleError else {
                return XCTFail("expected resourceNotFound, got \(error)")
            }
            XCTAssertEqual(resource, "no_such_skin.json")
            XCTAssertEqual(subdirectory, "Resources")
        }
        let shipped = try ResourceSceneBundle()
        let absentSkin = FileManager.default.temporaryDirectory
            .appendingPathComponent("UEMASceneTests-absent-\(UUID().uuidString).json", isDirectory: false)
        let opened = try ResourceSceneBundle(folderURL: shipped.folderURL, skinLayoutURL: absentSkin)
        XCTAssertTrue(opened.checkFiles().allSatisfy { $0.passes })
        XCTAssertThrowsError(try opened.verify()) { error in
            guard let bundleError = error as? SceneBundleError,
                  case let .fileMissing(name, _) = bundleError else {
                return XCTFail("expected fileMissing, got \(error)")
            }
            XCTAssertEqual(name, absentSkin.lastPathComponent)
        }
    }

    /// A name the bundle does not hold is reported with the subdirectory asked for.
    func testUnknownBundleNameIsReportedWithItsSubdirectory() {
        XCTAssertThrowsError(try ResourceSceneBundle(name: "no_such_bundle")) { error in
            guard let bundleError = error as? SceneBundleError,
                  case let .resourceNotFound(resource, subdirectory) = bundleError else {
                return XCTFail("expected resourceNotFound, got \(error)")
            }
            XCTAssertEqual(resource, "bundle.json")
            XCTAssertEqual(subdirectory, "Resources/no_such_bundle")
        }
    }

    // MARK: - Helpers

    /// The shipped skin layout JSON's URL in the module's resource bundle.
    private func shippedSkinURL() throws -> URL {
        try SkinLayout.shippedURL()
    }

    /// A new empty folder under the temporary directory.
    private func temporaryFolder() throws -> URL {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("UEMASceneTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder
    }

    /// A copy of the shipped folder's 10 files in a new temporary folder.
    private func copyOfShippedFolder() throws -> URL {
        let shipped = try ResourceSceneBundle()
        let folder = try temporaryFolder()
        let names = shipped.manifest.files.map { $0.name } + [ResourceSceneBundle.manifestFileName]
        for name in names {
            try FileManager.default.copyItem(
                at: shipped.folderURL.appendingPathComponent(name, isDirectory: false),
                to: folder.appendingPathComponent(name, isDirectory: false))
        }
        return folder
    }

    /// Every value of a `file="..."` attribute in an XML text, in document order.
    private func fileAttributeValues(in xml: String) -> [String] {
        var values: [String] = []
        var rest = Substring(xml)
        while let range = rest.range(of: "file=\"") {
            let afterQuote = rest[range.upperBound...]
            guard let close = afterQuote.firstIndex(of: "\"") else { break }
            values.append(String(afterQuote[..<close]))
            rest = afterQuote[close...]
        }
        return values
    }
}
