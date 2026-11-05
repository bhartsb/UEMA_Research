// TokenizerBankTests.swift: TokenizerBank routes each reading to its sense's tokenizer (work item
// WI-30).

import XCTest
import UEMACore
import UEMATokenizer

final class TokenizerBankTests: XCTestCase {
    func testReleasedBankRoutesBySense() throws {
        let bank = TokenizerBank.released(vestibular: .cold)
        XCTAssertEqual(Set(bank.tokenizers.keys), Set(SenseName.active))
        let readings = [
            SenseReading(sense: .audio, primary: ShapedArray(zerosOfShape: [256, 2])),
            SenseReading(sense: .motor, primary: ShapedArray(zerosOfShape: [26, 10])),
            SenseReading(sense: .proprioception, primary: ShapedArray(zerosOfShape: [26, 11])),
        ]
        let inputs = try bank.tokenize(readings)
        XCTAssertEqual(inputs[.audio]?.primary.shape, [256, 2])
        XCTAssertEqual(inputs[.motor]?.primary.shape, [26, 19])
        XCTAssertEqual(inputs[.proprioception]?.primary.shape, [312])
    }

    func testDuplicateAndMissingSensesThrow() throws {
        XCTAssertThrowsError(try TokenizerBank([AudioTokenizer(), AudioTokenizer()])) { error in
            XCTAssertEqual(error as? TokenizerError, .duplicateSense(.audio))
        }
        let bank = try TokenizerBank([AudioTokenizer()])
        XCTAssertThrowsError(try bank.tokenize(
            [SenseReading(sense: .intermind, primary: ShapedArray(zerosOfShape: [3]))])) { error in
            XCTAssertEqual(error as? UEMACoreError, .missingSense(.intermind))
        }
    }
}
