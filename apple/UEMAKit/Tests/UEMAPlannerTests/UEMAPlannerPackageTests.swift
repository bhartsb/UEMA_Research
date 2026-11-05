// UEMAPlannerPackageTests.swift: the placeholder test of the UEMAPlannerTests
// target (work item WI-17). The work item that fills the module adds its own tests
// beside this file.

import XCTest
import UEMAPlanner

/// Checks that the test target links its module (and finds its bundled files where
/// the manifest declares resources).
final class UEMAPlannerPackageTests: XCTestCase {
    /// The module's name constant equals its SwiftPM target name.
    func testModuleNameEqualsTargetName() {
        XCTAssertEqual(UEMAPlannerModule.name, "UEMAPlanner")
    }
}
