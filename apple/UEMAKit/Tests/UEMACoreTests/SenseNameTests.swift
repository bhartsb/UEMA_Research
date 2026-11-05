// SenseNameTests.swift: the tests of SenseName.swift (work item WI-20): six active senses in
// the registry's sequence, intermind inactive, 192 numbers per token, the token set helpers.
// Not compiled on the machine that wrote it (no Swift toolchain on Windows).

import Foundation
import XCTest
import UEMACore

final class SenseNameTests: XCTestCase {
    /// The acceptance line of WI-20: six active senses, in the released run's sequence.
    func testSixActiveSensesInTheRegistrySequence() {
        XCTAssertEqual(SenseName.active, [.vision, .tactile, .audio, .proprioception, .motor, .vestibular])
        XCTAssertEqual(SenseName.active.count, 6)
        XCTAssertEqual(SenseName.activeCount, 6)
        XCTAssertEqual(SenseName.allCases.count, 7)
        XCTAssertEqual(SenseName.allCases.filter { $0.isActive }.count, 6)
        XCTAssertFalse(SenseName.intermind.isActive)
        XCTAssertNil(SenseName.intermind.tokenIndex)
        XCTAssertEqual(SenseName.vision.tokenIndex, 0)
        XCTAssertEqual(SenseName.vestibular.tokenIndex, 5)
        XCTAssertEqual(SenseName.goalScored.count, 5)
        XCTAssertFalse(SenseName.goalScored.contains(.motor))
        XCTAssertFalse(SenseName.goalScored.contains(.intermind))
        XCTAssertEqual(SenseName.vision.rawValue, "vision")
        XCTAssertEqual(SenseName(rawValue: "proprioception"), .proprioception)
    }

    /// The acceptance line of WI-20: 192 numbers per token.
    func testTokenHolds192Numbers() throws {
        XCTAssertEqual(Token.count, 192)
        XCTAssertEqual(Token.zero.values.count, 192)
        XCTAssertThrowsError(try Token(values: [1, 2, 3]))
        XCTAssertThrowsError(try Token(values: Array(repeating: 0, count: 191)))
        let ones = Token(repeating: 1)
        XCTAssertEqual(ones.values.count, 192)
        XCTAssertEqual(ones[191], 1)
        XCTAssertEqual(ones.rootMeanSquare, 1, accuracy: 1e-6)
        XCTAssertEqual(Token.zero.rootMeanSquare, 0, accuracy: 1e-6)
        XCTAssertEqual(ones.meanSquaredDifference(to: Token.zero), 1, accuracy: 1e-6)
        XCTAssertEqual(ones.meanSquaredDifference(to: ones), 0, accuracy: 1e-6)
        let exact = try Token(values: Array(repeating: 0.5, count: 192))
        XCTAssertEqual(exact.meanSquaredDifference(to: Token.zero), 0.25, accuracy: 1e-6)
    }

    /// A token set names its missing active senses and lists the six tokens in sequence.
    func testTokenSetHelpers() throws {
        var set: TokenSet = [:]
        XCTAssertEqual(set.missingActiveSenses, SenseName.active)
        XCTAssertThrowsError(try set.activeSequence())
        for sense in SenseName.active {
            set[sense] = Token(repeating: Float(sense.tokenIndex ?? -1))
        }
        XCTAssertTrue(set.missingActiveSenses.isEmpty)
        let sequence = try set.activeSequence()
        XCTAssertEqual(sequence.count, 6)
        XCTAssertEqual(sequence[0][0], 0)
        XCTAssertEqual(sequence[5][0], 5)
        set[.audio] = nil
        XCTAssertEqual(set.missingActiveSenses, [.audio])
    }

    /// Encoding then decoding keeps a token; a decoded token of the wrong count throws.
    func testTokenCodableKeepsTheCountCheck() throws {
        let token = Token(repeating: 0.25)
        let data = try JSONEncoder().encode(token)
        XCTAssertEqual(try JSONDecoder().decode(Token.self, from: data), token)
        let bad = Data(#"{"values":[1,2,3]}"#.utf8)
        XCTAssertThrowsError(try JSONDecoder().decode(Token.self, from: bad))
        // A sense encodes as the registry's name, the spelling of the manifest and fixtures.
        XCTAssertEqual(try JSONEncoder().encode([SenseName.vestibular]), Data(#"["vestibular"]"#.utf8))
        XCTAssertThrowsError(try JSONDecoder().decode([SenseName].self, from: Data(#"["balance"]"#.utf8)))
    }
}
