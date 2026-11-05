// UEMAScenePackageTests.swift: the placeholder test of the UEMASceneTests
// target (work item WI-17). The work item that fills the module adds its own tests
// beside this file.

import XCTest
import UEMAScene

/// Checks that the test target links its module (and finds its bundled files where
/// the manifest declares resources).
final class UEMAScenePackageTests: XCTestCase {
    /// The module's name constant equals its SwiftPM target name.
    func testModuleNameEqualsTargetName() {
        XCTAssertEqual(UEMASceneModule.name, "UEMAScene")
    }

    /// The skin layout JSON of WI-13 is found in the module's resource bundle.
    func testSkinLayoutResourceIsBundled() {
        let url = UEMASceneModule.resourceBundle.url(
            forResource: "skin_layout_age24_trunk_v1",
            withExtension: "json",
            subdirectory: "Resources"
        )
        XCTAssertNotNil(url)
    }
}
