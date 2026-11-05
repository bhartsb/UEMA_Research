// VestibularNormaliserTests.swift: one hand-computed row of VestibularNormaliser (work item
// WI-30). Rule (vestibular_sense.py:435-465): x = clip(field, -4, 4); z = (x - mean) /
// sqrt(max(variance, 1e-6)); a missing or non-finite z is 0; z clipped to -4..4.

import XCTest
import UEMACore
import UEMATokenizer

final class VestibularNormaliserTests: XCTestCase {
    /// mean 1.0 for field 0 and 0 elsewhere; variance 0.25 for field 0 (spread 0.5), 0 for
    /// field 1 (spread floored to 0.001), 1 elsewhere.
    private func normaliser() throws -> VestibularNormaliser {
        var mean = [Double](repeating: 0, count: 28)
        mean[0] = 1
        var variance = [Double](repeating: 1, count: 28)
        variance[0] = 0.25
        variance[1] = 0
        return try VestibularNormaliser(mean: mean, variance: variance, weight: 0.9)
    }

    private func fields(field3: Float) -> [Float] {
        var values = [Float](repeating: 0, count: 29)
        values[0] = 2        // (2 - 1) / 0.5 = 2
        values[1] = 0.01     // 0.01 / 0.001 = 10, clipped to 4
        values[2] = 9        // raw clipped to 4, then 4 / 1 = 4
        values[3] = field3
        values[4] = -0.5     // -0.5 / 1
        values[28] = 1       // valid
        return values
    }

    func testHandComputedRow() throws {
        let result = try normaliser().normalise(fields(field3: 0.25))
        XCTAssertTrue(result.isRealRead)
        XCTAssertEqual(result.values.count, 28)
        let expected: [Float] = [2, 4, 4, 0.25, -0.5, 0]
        for (index, value) in expected.enumerated() {
            XCTAssertEqual(result.values[index], value, accuracy: 1e-6, "field \(index)")
        }
    }

    func testNonFiniteFieldReadsZeroAndMarksTheReadStale() throws {
        let reading = SenseReading(
            sense: .vestibular, primary: try ShapedArray(shape: [29], values: fields(field3: .nan)))
        let input = try normaliser().tokenize(reading)
        XCTAssertEqual(input.primary.shape, [28])
        XCTAssertEqual(input.primary.values[3], 0)
        XCTAssertEqual(input.primary.values[0], 2, accuracy: 1e-6)
        XCTAssertFalse(input.isFresh)
    }

    func testColdNormaliserGivesZerosAndWrongLengthsThrow() throws {
        let result = try VestibularNormaliser.cold.normalise(fields(field3: 0.25))
        XCTAssertEqual(result.values, [Float](repeating: 0, count: 28))
        XCTAssertThrowsError(try VestibularNormaliser(mean: [0], variance: [1], weight: 1))
        XCTAssertThrowsError(try normaliser().normalise([Float](repeating: 0, count: 28)))
        let spreadForm = try VestibularNormaliser(
            mean: [Double](repeating: 0, count: 28), spread: [Double](repeating: 2, count: 28), weight: 1)
        XCTAssertEqual(try spreadForm.normalise(fields(field3: 0.25)).values[4], -0.25, accuracy: 1e-7)
    }
}
