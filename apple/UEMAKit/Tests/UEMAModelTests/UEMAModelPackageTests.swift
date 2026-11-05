// UEMAModelPackageTests.swift: the placeholder test of the UEMAModelTests
// target (work item WI-17). The work item that fills the module adds its own tests
// beside this file.

import XCTest
import Foundation
import UEMAModel

/// Checks that the test target links its module (and finds its bundled files where
/// the manifest declares resources).
final class UEMAModelPackageTests: XCTestCase {
    /// The module's name constant equals its SwiftPM target name.
    func testModuleNameEqualsTargetName() {
        XCTAssertEqual(UEMAModelModule.name, "UEMAModel")
    }

    /// The weight manifest of WI-12 is found in the module's resource bundle.
    func testWeightManifestResourceIsBundled() {
        let url = UEMAModelModule.resourceBundle.url(
            forResource: "weight_manifest",
            withExtension: "json",
            subdirectory: "Resources"
        )
        XCTAssertNotNil(url)
    }

    /// The Fixtures folder is bundled with the test target (its placeholder file is found).
    func testFixturesFolderIsBundled() {
        let url = Bundle.module.url(
            forResource: "FIXTURES",
            withExtension: "txt",
            subdirectory: "Fixtures"
        )
        XCTAssertNotNil(url)
    }
}
