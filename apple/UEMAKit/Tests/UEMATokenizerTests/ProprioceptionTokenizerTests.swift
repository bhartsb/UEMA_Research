// ProprioceptionTokenizerTests.swift: one hand-computed row of ProprioceptionTokenizer (work item
// WI-30). Rule (datasource.py:127-141): position minus row 0 over 200 cm, velocity over 500 cm/s,
// row index over 25, quaternion and fatigue copied.

import XCTest
import UEMACore
import UEMATokenizer

final class ProprioceptionTokenizerTests: XCTestCase {
    /// Row i: quaternion (0, 0, 0, 1), position (10 i, 20, 30 + i) cm, velocity 0, fatigue 0.75;
    /// row 3 has quaternion (0.1, 0.2, 0.3, 0.9) and velocity (100, -250, 50) cm/s.
    private func reading(columns: Int) throws -> SenseReading {
        var values = [Float](repeating: 0, count: 26 * columns)
        for i in 0 ..< 26 {
            let s = i * columns
            values[s + 3] = 1
            values[s + 4] = Float(10 * i)
            values[s + 5] = 20
            values[s + 6] = Float(30 + i)
            values[s + 10] = 0.75
            if columns == 12 {
                values[s + 11] = 99
            }
        }
        values[3 * columns + 0] = 0.1
        values[3 * columns + 1] = 0.2
        values[3 * columns + 2] = 0.3
        values[3 * columns + 3] = 0.9
        values[3 * columns + 7] = 100
        values[3 * columns + 8] = -250
        values[3 * columns + 9] = 50
        return SenseReading(sense: .proprioception, primary: try ShapedArray(shape: [26, columns], values: values))
    }

    func testRowThreeMatchesTheHandComputedRow() throws {
        for columns in [11, 12] {
            let input = try ProprioceptionTokenizer().tokenize(try reading(columns: columns))
            XCTAssertEqual(input.primary.shape, [312])
            let row = Array(input.primary.values[36 ..< 48])
            // (30 - 0) / 200 = 0.15; (20 - 20) / 200 = 0; (33 - 30) / 200 = 0.015; 100 / 500 = 0.2;
            // -250 / 500 = -0.5; 50 / 500 = 0.1; 3 / 25 = 0.12.
            let expected: [Float] = [0.1, 0.2, 0.3, 0.9, 0.15, 0, 0.015, 0.2, -0.5, 0.1, 0.75, 0.12]
            for (index, value) in expected.enumerated() {
                XCTAssertEqual(row[index], value, accuracy: 1e-6, "columns \(columns), number \(index)")
            }
            // Row 0 is the origin; row 25's index column is 25 / 25 = 1.
            XCTAssertEqual(input.primary.values[4], 0)
            XCTAssertEqual(input.primary.values[25 * 12 + 11], 1, accuracy: 1e-7)
        }
    }

    func testOtherShapesAndSensesThrow() throws {
        let tokenizer = ProprioceptionTokenizer()
        XCTAssertThrowsError(try tokenizer.tokenize(
            SenseReading(sense: .proprioception, primary: ShapedArray(zerosOfShape: [24, 11]))))
        XCTAssertThrowsError(try tokenizer.tokenize(
            SenseReading(sense: .motor, primary: ShapedArray(zerosOfShape: [26, 11])))) { error in
            XCTAssertEqual(error as? TokenizerError, .wrongSense(expected: .proprioception, found: .motor))
        }
    }
}
