// MotorTokenizerTests.swift: one hand-computed row of MotorTokenizer and MotorEnergyScaler (work
// item WI-30). Rules: achieved positions over 200 cm (datasource.py:150); command columns zero on
// the live model step and the executed command with positions over 200 on a context row
// (lem_model.py:139-149, :2640-2661); energy and accumulated effort over 1,000 (motor.py:85-108).

import XCTest
import UEMACore
import UEMATokenizer

final class MotorTokenizerTests: XCTestCase {
    /// Row 5: position (40, -20, 100) cm, quaternion (0, 0, 0.6, 0.8), radius 0.3, strength 0.6,
    /// mask 1; every other row 0.
    private func achieved() throws -> ShapedArray {
        var values = [Float](repeating: 0, count: 26 * 10)
        let row: [Float] = [40, -20, 100, 0, 0, 0.6, 0.8, 0.3, 0.6, 1]
        values.replaceSubrange(50 ..< 60, with: row)
        return try ShapedArray(shape: [26, 10], values: values)
    }

    func testLiveRowHasZeroCommandColumns() throws {
        let input = try MotorTokenizer().tokenize(SenseReading(sense: .motor, primary: try achieved()))
        XCTAssertEqual(input.primary.shape, [26, 19])
        let row = Array(input.primary.values[5 * 19 ..< 6 * 19])
        let expected: [Float] = [0.2, -0.1, 0.5, 0, 0, 0.6, 0.8, 0.3, 0.6, 1] + [Float](repeating: 0, count: 9)
        for (index, value) in expected.enumerated() {
            XCTAssertEqual(row[index], value, accuracy: 1e-6, "number \(index)")
        }
    }

    func testContextRowCarriesTheExecutedCommandOver200() throws {
        var command = Command.zero(scale: .centimetres)
        guard let handle = BodyHandle(rawValue: 5) else {
            return XCTFail("handle 5 exists in the 26-row layout")
        }
        try command.setRow(handle, to: [100, 0, -50, 0, 0, 0, 1, 0.5, 0.7])
        let input = try MotorTokenizer().encoderInput(achieved: try achieved(), executedCommand: command)
        let row = Array(input.values[5 * 19 + 10 ..< 6 * 19])
        let expected: [Float] = [0.5, 0, -0.25, 0, 0, 0, 1, 0.5, 0.7]
        for (index, value) in expected.enumerated() {
            XCTAssertEqual(row[index], value, accuracy: 1e-6, "command number \(index)")
        }
        // A command already at the normalized scale is not divided again.
        let again = try MotorTokenizer().encoderInput(achieved: try achieved(), executedCommand: command.normalized())
        XCTAssertEqual(again, input)
    }

    func testEnergyAndCapabilityOver1000() throws {
        let scaler = MotorEnergyScaler()
        let energy = scaler.energy(physicalWork: 250, holding: 50, total: 300)
        XCTAssertEqual(energy.count, 3)
        XCTAssertEqual(energy[0], 0.25, accuracy: 1e-7)
        XCTAssertEqual(energy[1], 0.05, accuracy: 1e-7)
        XCTAssertEqual(energy[2], 0.3, accuracy: 1e-7)
        let capacities = [Float](repeating: 0.9, count: 10)
        let accumulated = [Float](repeating: 120, count: 10)
        let capability = try scaler.capability(capacities: capacities, accumulated: accumulated, totalEnergy: 300)
        XCTAssertEqual(capability.count, 21)
        XCTAssertEqual(capability[0], 0.9)
        XCTAssertEqual(capability[10], 0.12, accuracy: 1e-7)
        XCTAssertEqual(capability[20], 0.3, accuracy: 1e-7)
        XCTAssertThrowsError(try scaler.capability(capacities: [0.9], accumulated: accumulated, totalEnergy: 0))
    }

    /// The motor reader's auxiliary [24] (UEMASenses/MotorReader.swift): energy (250, 50, 300),
    /// capacities 0.9, accumulated 120, total 300 gives the same numbers as the two functions.
    func testReaderAuxiliaryScalesLikeTheTwoFunctions() throws {
        let values: [Float] = [250, 50, 300] + [Float](repeating: 0.9, count: 10)
            + [Float](repeating: 120, count: 10) + [300]
        let scaled = try MotorEnergyScaler().scaled(readingAuxiliary: try ShapedArray(shape: [24], values: values))
        XCTAssertEqual(scaled.energy, MotorEnergyScaler().energy(physicalWork: 250, holding: 50, total: 300))
        XCTAssertEqual(scaled.capability.count, 21)
        XCTAssertEqual(scaled.capability[9], 0.9)
        XCTAssertEqual(scaled.capability[19], 0.12, accuracy: 1e-7)
        XCTAssertEqual(scaled.capability[20], 0.3, accuracy: 1e-7)
        XCTAssertThrowsError(try MotorEnergyScaler().scaled(readingAuxiliary: ShapedArray(zerosOfShape: [21])))
    }
}
