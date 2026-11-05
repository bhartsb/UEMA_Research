// CMuJoCoPackageTests.swift: the first test of the CMuJoCoTests target. Work item
// WI-17 wrote it as a placeholder; work item WI-19 replaced the placeholder assertion
// with the header-against-library version check below. The other WI-19 tests sit in
// CMuJoCoSymbolTests.swift and CMuJoCoRendererFreeTests.swift beside this file.
//
// Nothing here was compiled: the machine that wrote it has no Swift toolchain. Swift's
// C importer gives the integer macro mjVERSION_HEADER the type Int32, the same type as
// the return value of int mj_version(void).

import CMuJoCo
import XCTest

/// Checks that the test target imports and links its module.
final class CMuJoCoPackageTests: XCTestCase {
    /// The header the module imported and the static library linked are the same
    /// MuJoCo release: mjVERSION_HEADER (a macro of mujoco.h, 330 at tag 3.3.0) equals
    /// mj_version() (a function compiled into libmujoco.a). A mismatch means the
    /// XCFramework's Headers folder and its library come from different builds.
    /// WI-19's acceptance test, which calls mj_versionString() and expects "3.3.0", is
    /// testVersionStringIs330 in CMuJoCoSymbolTests.swift beside this file.
    func testHeaderAndLibraryVersionsAgree() {
        XCTAssertEqual(mjVERSION_HEADER, mj_version())
    }
}
