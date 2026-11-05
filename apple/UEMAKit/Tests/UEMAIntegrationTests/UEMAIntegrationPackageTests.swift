// UEMAIntegrationPackageTests.swift: the placeholder test of the UEMAIntegrationTests
// target (work item WI-17). Work item WI-40 adds OneModelStepTests.swift beside it: the
// real stack for 20 model steps on macOS, recording peak resident memory.

import XCTest
import UEMACore
import UEMACommand

/// Checks that the integration test target links the first and last modules of the
/// package's dependency sequence.
final class UEMAIntegrationPackageTests: XCTestCase {
    /// Both ends of the module sequence link.
    func testSequenceEndsLink() {
        XCTAssertEqual(UEMACoreModule.name, "UEMACore")
        XCTAssertEqual(UEMACommandModule.name, "UEMACommand")
    }
}
