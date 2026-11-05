// UEMAPhysicsPackageTests.swift: the placeholder test of the UEMAPhysicsTests
// target (work item WI-17). The work item that fills the module adds its own tests
// beside this file.

import XCTest
import UEMAPhysics

/// Checks that the test target links its module (and finds its bundled files where
/// the manifest declares resources).
final class UEMAPhysicsPackageTests: XCTestCase {
    /// The module's name constant equals its SwiftPM target name.
    func testModuleNameEqualsTargetName() {
        XCTAssertEqual(UEMAPhysicsModule.name, "UEMAPhysics")
    }
}
