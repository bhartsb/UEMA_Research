// UEMASensesPackageTests.swift: the placeholder test of the UEMASensesTests
// target (work item WI-17). The work item that fills the module adds its own tests
// beside this file.

import XCTest
import Foundation
import UEMASenses

/// Checks that the test target links its module (and finds its bundled files where
/// the manifest declares resources).
final class UEMASensesPackageTests: XCTestCase {
    /// The module's name constant equals its SwiftPM target name.
    func testModuleNameEqualsTargetName() {
        XCTAssertEqual(UEMASensesModule.name, "UEMASenses")
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
