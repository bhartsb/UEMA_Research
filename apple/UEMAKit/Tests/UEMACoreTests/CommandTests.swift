// CommandTests.swift: the tests of Command.swift (work item WI-20): the command holds 234
// numbers (26 handles x 9 columns), the handle and column names, the scale conversion, the
// gaze round trip, the handle mask, the finger joint goals and the 92 motor controls.
// Not compiled on the machine that wrote it (no Swift toolchain on Windows).

import Foundation
import XCTest
import UEMACore

final class CommandTests: XCTestCase {
    /// The acceptance line of WI-20: 234 numbers per command.
    func testCommandHolds234Numbers() throws {
        XCTAssertEqual(Command.count, 234)
        XCTAssertEqual(Command.handleCount, 26)
        XCTAssertEqual(Command.numbersPerHandle, 9)
        XCTAssertEqual(Command.handleCount * Command.numbersPerHandle, Command.count)
        let zero = Command.zero(scale: .centimetres)
        XCTAssertEqual(zero.values.count, 234)
        XCTAssertEqual(zero.scale, .centimetres)
        XCTAssertTrue(zero.values.allSatisfy { $0 == 0 })
        let ones = try Command(values: Array(repeating: 1, count: 234), scale: .normalized)
        XCTAssertEqual(ones.values.count, 234)
        XCTAssertThrowsError(try Command(values: Array(repeating: 0, count: 233), scale: .centimetres))
        XCTAssertThrowsError(try Command(values: Array(repeating: 0, count: 235), scale: .centimetres))
    }

    /// The 26 handle rows follow HANDLE_NAMES then lower_trunk and chest; row 23 is the eyes.
    func testHandleRowsAndColumns() {
        XCTAssertEqual(BodyHandle.allCases.count, 26)
        XCTAssertEqual(BodyHandle.count, 26)
        XCTAssertEqual(BodyHandle.pythonNames.count, 26)
        XCTAssertEqual(BodyHandle.gaze.rawValue, 23)
        XCTAssertEqual(Command.gazeHandle, .gaze)
        XCTAssertEqual(BodyHandle.pythonNames[0], "wrist_L")
        XCTAssertEqual(BodyHandle.pythonNames[12], "head")
        XCTAssertEqual(BodyHandle.pythonNames[23], "gaze")
        XCTAssertEqual(BodyHandle.pythonNames[24], "lower_trunk")
        XCTAssertEqual(BodyHandle.pythonNames[25], "chest")
        XCTAssertEqual(BodyHandle.chest.pythonName, "chest")
        XCTAssertEqual(BodyHandle(pythonName: "pinky_R"), .pinkyR)
        XCTAssertNil(BodyHandle(pythonName: "tail"))
        for handle in BodyHandle.allCases {
            XCTAssertEqual(BodyHandle(pythonName: handle.pythonName), handle)
        }
        XCTAssertEqual(BodyHandle.fingers.count, 10)
        XCTAssertTrue(BodyHandle.thumbL.isFinger)
        XCTAssertFalse(BodyHandle.wristL.isFinger)
        XCTAssertFalse(BodyHandle.head.isFinger)
        XCTAssertEqual(BodyHandle.positionHandles.count, 14)
        XCTAssertEqual(BodyHandle.positionHandles.map { $0.pythonName }, [
            "wrist_L", "thumb_L", "index_L", "middle_L", "ring_L", "pinky_L",
            "wrist_R", "thumb_R", "index_R", "middle_R", "ring_R", "pinky_R",
            "head", "lower_trunk",
        ])
        XCTAssertEqual(BodyHandle.fingers.map { $0.rawValue }, [1, 2, 3, 4, 5, 7, 8, 9, 10, 11])
        XCTAssertEqual(CommandColumn.allCases.count, 9)
        XCTAssertEqual(CommandColumn.count, 9)
        XCTAssertEqual(CommandColumn.position.count, 3)
        XCTAssertEqual(CommandColumn.quaternion.count, 4)
        XCTAssertEqual(Command.index(handle: .gaze, column: .quaternionW), 23 * 9 + 6)
        XCTAssertEqual(Command.index(handle: .chest, column: .strength), 233)
    }

    /// Subscript, row read and row write address the numbers row-major by handle then column.
    func testSubscriptAndRows() throws {
        var command = Command.zero(scale: .normalized)
        command[.head, .strength] = 0.5
        XCTAssertEqual(command.strength(of: .head), 0.5)
        XCTAssertEqual(command.values[12 * 9 + 8], 0.5)
        try command.setRow(.wristR, to: [1, 2, 3, 0, 0, 0, 1, 0.25, 0.75])
        XCTAssertEqual(command.row(.wristR), [1, 2, 3, 0, 0, 0, 1, 0.25, 0.75])
        XCTAssertEqual(command.position(of: .wristR), [1, 2, 3])
        XCTAssertEqual(command.quaternion(of: .wristR), [0, 0, 0, 1])
        XCTAssertEqual(command.radius(of: .wristR), 0.25)
        XCTAssertEqual(command.strength(of: .wristR), 0.75)
        XCTAssertEqual(command.row(.wristL), Array(repeating: 0, count: 9))
        XCTAssertThrowsError(try command.setRow(.wristR, to: [1, 2]))
    }

    /// normalized() divides the three position columns by 200 and copies the other six.
    func testNormalizedDividesPositionsBy200() throws {
        var command = Command.zero(scale: .centimetres)
        try command.setRow(.wristL, to: [100, -50, 20, 0.1, 0.2, 0.3, 0.9, 0.4, 0.6])
        let normalized = command.normalized()
        XCTAssertEqual(normalized.scale, .normalized)
        XCTAssertEqual(normalized.position(of: .wristL), [0.5, -0.25, 0.1])
        XCTAssertEqual(normalized.quaternion(of: .wristL), [0.1, 0.2, 0.3, 0.9])
        XCTAssertEqual(normalized.radius(of: .wristL), 0.4)
        XCTAssertEqual(normalized.strength(of: .wristL), 0.6)
        XCTAssertEqual(normalized.normalized(), normalized)
        XCTAssertEqual(Command.positionRangeCentimetres, 200)
    }

    /// setGaze then gazeAngles returns the angles; a zero row reads straight ahead; angles
    /// beyond the eye ranges clamp (azimuth 0.785, elevation -0.820).
    func testGazeRoundTrip() {
        var command = Command.zero(scale: .centimetres)
        command.setGaze(GazeAngles(azimuth: 0.3, elevation: -0.2))
        let quaternion = command.quaternion(of: .gaze)
        // The numpy values of gaze_to_handle for az 0.3, el -0.2: x, y, z, w.
        XCTAssertEqual(Double(quaternion[0]), 0.01491892, accuracy: 1e-6)
        XCTAssertEqual(Double(quaternion[1]), -0.09871239, accuracy: 1e-6)
        XCTAssertEqual(Double(quaternion[2]), 0.14869156, accuracy: 1e-6)
        XCTAssertEqual(Double(quaternion[3]), 0.98383134, accuracy: 1e-6)
        let read = command.gazeAngles
        XCTAssertEqual(read.azimuth, 0.3, accuracy: 1e-5)
        XCTAssertEqual(read.elevation, -0.2, accuracy: 1e-5)
        XCTAssertEqual(command.position(of: .gaze), [0, 0, 0])

        let straight = Command.zero(scale: .centimetres).gazeAngles
        XCTAssertEqual(straight.azimuth, 0, accuracy: 1e-12)
        XCTAssertEqual(straight.elevation, 0, accuracy: 1e-12)

        command.setGaze(GazeAngles(azimuth: 1.5, elevation: 0.3))
        XCTAssertEqual(command.gazeAngles.azimuth, 0.785, accuracy: 1e-5)
        XCTAssertEqual(command.gazeAngles.elevation, 0.3, accuracy: 1e-5)
        command.setGaze(GazeAngles(azimuth: 0.2, elevation: -1.2))
        XCTAssertEqual(command.gazeAngles.azimuth, 0.2, accuracy: 1e-5)
        XCTAssertEqual(command.gazeAngles.elevation, -0.820, accuracy: 1e-5)

        XCTAssertEqual(GazeAngles(azimuth: 2, elevation: 2).clamped(), GazeAngles(azimuth: 0.785, elevation: 0.576))
        XCTAssertEqual(GazeAngles.straightAhead, GazeAngles(azimuth: 0, elevation: 0))
    }

    /// 92 controls; clipped() brings every control into -1 to 1.
    func testMotorControlsHold92Numbers() throws {
        XCTAssertEqual(MotorControls.count, 92)
        XCTAssertEqual(MotorControls.zero.values.count, 92)
        XCTAssertTrue(MotorControls.zero.isWithinRange)
        XCTAssertThrowsError(try MotorControls(values: [0, 1]))
        var values = Array(repeating: 0.0, count: 92)
        values[0] = 2.5
        values[91] = -3
        let controls = try MotorControls(values: values)
        XCTAssertFalse(controls.isWithinRange)
        let clipped = controls.clipped()
        XCTAssertEqual(clipped.values[0], 1)
        XCTAssertEqual(clipped.values[91], -1)
        XCTAssertEqual(clipped.values[1], 0)
        XCTAssertTrue(clipped.isWithinRange)
        XCTAssertEqual(MotorControls.range, (-1.0) ... (1.0))
    }

    /// The handle mask holds 26 booleans (all true in the released run); the finger joint
    /// goals hold 44 angles.
    func testHandleMaskAndFingerGoals() throws {
        XCTAssertEqual(HandleMask.count, 26)
        XCTAssertEqual(HandleMask.allTrue.values.count, 26)
        XCTAssertTrue(HandleMask.allTrue[.gaze])
        XCTAssertEqual(HandleMask.allTrue.asFloats, Array(repeating: 1, count: 26))
        XCTAssertEqual(HandleMask(repeating: false).asFloats, Array(repeating: 0, count: 26))
        XCTAssertThrowsError(try HandleMask(values: [true]))
        XCTAssertEqual(FingerJointGoals.count, 44)
        XCTAssertThrowsError(try FingerJointGoals(angles: [0]))
        let goals = try FingerJointGoals(angles: Array(repeating: 0.1, count: 44))
        XCTAssertEqual(goals.angles.count, 44)
        XCTAssertEqual(try JSONDecoder().decode(FingerJointGoals.self, from: JSONEncoder().encode(goals)), goals)
        XCTAssertThrowsError(try JSONDecoder().decode(FingerJointGoals.self, from: Data(#"{"angles":[0.1]}"#.utf8)))
        let mask = try HandleMask(values: (0 ..< 26).map { $0 != 23 })
        XCTAssertFalse(mask[.gaze])
        XCTAssertTrue(mask[.chest])
        XCTAssertEqual(try JSONDecoder().decode(HandleMask.self, from: JSONEncoder().encode(mask)), mask)
        XCTAssertThrowsError(try JSONDecoder().decode(HandleMask.self, from: Data(#"{"values":[true]}"#.utf8)))
    }

    /// Encoding then decoding keeps the command; a decoded command of the wrong count throws.
    func testCodableKeepsTheCountCheck() throws {
        var command = Command.zero(scale: .normalized)
        command[.footL, .positionZ] = -0.125
        let data = try JSONEncoder().encode(command)
        let back = try JSONDecoder().decode(Command.self, from: data)
        XCTAssertEqual(back, command)
        XCTAssertEqual(back[.footL, .positionZ], -0.125)
        let bad = Data(#"{"scale":"normalized","values":[1,2,3]}"#.utf8)
        XCTAssertThrowsError(try JSONDecoder().decode(Command.self, from: bad))
        let badControls = Data(#"{"values":[1,2,3]}"#.utf8)
        XCTAssertThrowsError(try JSONDecoder().decode(MotorControls.self, from: badControls))
    }
}
