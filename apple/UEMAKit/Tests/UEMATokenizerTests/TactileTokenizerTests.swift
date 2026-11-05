// TactileTokenizerTests.swift: one hand-computed token of TactileTokenizer (work item WI-30).
// Rules: stretch columns minus 1.0, every field over _FIELD_NORMS [100000, 50000, 200, 20000,
// 10, 10, 10, 10, 0.5, 0.5] (touch.py:75-76, :116-117); active patches sorted by normalised
// pressure, positions over 0.5 m, valid flag last (lem_model.py:224-264).

import XCTest
import UEMACore
import UEMATokenizer

final class TactileTokenizerTests: XCTestCase {
    /// Patch 0: 15,000 Pa, 20,000 Pa, 200 mm/s, 200 mm^2, vibration 4.0 N^2 at 5 Hz, stretches
    /// 1.2 and 0.9. Patch 1: 30,000 Pa, area 400 mm^2, stretches at rest. Patch 2: at rest
    /// everywhere (inactive after the re-zeroing).
    private func reading(withSides: Bool) throws -> SenseReading {
        let fields: [Float] = [
            15_000, 20_000, 200, 200, 4, 0, 0, 0, 1.2, 0.9,
            30_000, 0, 0, 400, 0, 0, 0, 0, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        ]
        let sides: [Float] = [
            0.1, -0.05, 0.2, 0.5, 0.25, 0, 0, 1, 0.3, 0.6,
            0, 0.25, -0.1, 0, 0, 1, 0, 0, 0.1, 0.2,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]
        let primary = try ShapedArray(shape: [3, 10], values: fields)
        var auxiliary: ShapedArray?
        if withSides {
            auxiliary = try ShapedArray(shape: [3, 10], values: sides)
        }
        return SenseReading(sense: .tactile, primary: primary, auxiliary: auxiliary)
    }

    private func assertRow(_ token: [Float], _ row: Int, _ expected: [Float],
                           file: StaticString = #filePath, line: UInt = #line) {
        for (index, value) in expected.enumerated() {
            XCTAssertEqual(token[row * 19 + index], value, accuracy: 1e-6, "row \(row) column \(index)",
                           file: file, line: line)
        }
    }

    func testTwoActivePatchesSortByPressure() throws {
        let input = try TactileTokenizer().tokenize(try reading(withSides: true))
        XCTAssertEqual(input.primary.shape, [32, 19])
        let token = input.primary.values
        // Patch 1 first (pressure 30,000 / 100,000 = 0.3), then patch 0 (0.15).
        assertRow(token, 0, [0, 0.5, -0.2, 0.3, 0, 0, 0.02, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.1, 0.2, 1])
        assertRow(token, 1, [0.2, -0.1, 0.4, 0.15, 0.4, 1, 0.01, 0.4, 0, 0, 0, 0.5, 0.25, 0, 0, 1, 0.3, 0.6, 1])
        assertRow(token, 2, [Float](repeating: 0, count: 19))
    }

    func testWithoutSideColumnsPositionsAreZeroAndRowsStillValid() throws {
        let token = try TactileTokenizer().tokenize(try reading(withSides: false)).primary.values
        assertRow(token, 0, [0, 0, 0, 0.3, 0, 0, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    }

    func testNormalisedSlotsKeepTheStretchColumns() throws {
        let slots = try TactileTokenizer.normalisedSlots(try reading(withSides: false).primary)
        XCTAssertEqual(slots.count, 256)
        XCTAssertEqual(slots[0][8], 0.4, accuracy: 1e-6)
        XCTAssertEqual(slots[0][9], -0.2, accuracy: 1e-6)
        XCTAssertEqual(slots[2], [Float](repeating: 0, count: 10))
    }

    /// 53 patches at rest (stretch ratios 1.0, every other field 0): every normalised field is 0,
    /// so no patch is active and the token is 32 zero rows.
    func testNoContactGives32ZeroRowsAndAFinishedTokenPassesThrough() throws {
        var rest = [Float](repeating: 0, count: 53 * 10)
        for r in 0 ..< 53 {
            rest[r * 10 + 8] = 1
            rest[r * 10 + 9] = 1
        }
        let empty = SenseReading(sense: .tactile, primary: try ShapedArray(shape: [53, 10], values: rest))
        let token = try TactileTokenizer().tokenize(empty)
        XCTAssertEqual(token.primary, ShapedArray(zerosOfShape: [32, 19]))
        var built = [Float](repeating: 0, count: 32 * 19)
        built[18] = 1
        let finished = SenseReading(sense: .tactile, primary: try ShapedArray(shape: [32, 19], values: built))
        XCTAssertEqual(try TactileTokenizer().tokenize(finished).primary.values, built)
    }

    /// The WI-28 reader's layout (UEMASenses/Tactile/TactileSense.swift): primary [256, 10] with
    /// the 3 patch rows of `reading(withSides:)` first and 253 zero slots, auxiliary [256, 11]
    /// with the same side columns and the patch-row flag in column 10. The zero slots are not
    /// re-zeroed as patches (touch.py:114-115), so the token equals the [3, 10] reading's.
    func testReaderLayoutSkipsTheUnusedSlots() throws {
        let small = try reading(withSides: true)
        var primary = [Float](repeating: 0, count: 256 * 10)
        primary.replaceSubrange(0 ..< 30, with: small.primary.values)
        var auxiliary = [Float](repeating: 0, count: 256 * 11)
        let sides = try XCTUnwrap(small.auxiliary).values
        for r in 0 ..< 3 {
            for c in 0 ..< 10 {
                auxiliary[r * 11 + c] = sides[r * 10 + c]
            }
            auxiliary[r * 11 + 10] = 1
        }
        let padded = SenseReading(
            sense: .tactile,
            primary: try ShapedArray(shape: [256, 10], values: primary),
            auxiliary: try ShapedArray(shape: [256, 11], values: auxiliary))
        let token = try TactileTokenizer().tokenize(padded).primary
        XCTAssertEqual(token, try TactileTokenizer().tokenize(small).primary)
        // Without the flag column the 253 zero slots would read -2 in both stretch columns and
        // fill rows 2 to 31; here row 2 stays zero.
        assertRow(token.values, 2, [Float](repeating: 0, count: 19))
    }

    func testWrongAuxiliaryShapeThrows() throws {
        let bad = SenseReading(
            sense: .tactile,
            primary: try reading(withSides: false).primary,
            auxiliary: ShapedArray(zerosOfShape: [3, 7]))
        XCTAssertThrowsError(try TactileTokenizer().tokenize(bad))
        let tooMany = SenseReading(sense: .tactile, primary: ShapedArray(zerosOfShape: [257, 10]))
        XCTAssertThrowsError(try TactileTokenizer().tokenize(tooMany))
    }
}
