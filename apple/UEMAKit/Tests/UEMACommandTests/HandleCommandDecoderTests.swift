// HandleCommandDecoderTests.swift: 14 tests of HandleCommandDecoder, the array-level port of
// Source/Python/mimo_bridge/action_decoder.py decode (lines 402-520), on hand-built bodies
// (CommandTestBodies.swift). Each expected number is worked from the formulas of
// apple/UEMAKit/Specs/command_decoder.md section 3 in the test's comment: kp 6.0, kd 0.3,
// damping 1e-3, position weight 8.0, gaze kp 80.0 (DecoderGains.released). Nothing here was
// compiled on the machine that wrote it (Windows 11, no Swift toolchain).

import XCTest
import UEMACore
import UEMACommand

final class HandleCommandDecoderTests: XCTestCase {
    private let tolerance = 1e-9

    // One hinge about z on body 1, the handle "head" on that body.
    private func singleHinge(limited: Bool = false, low: Double = 0, high: Double = 0) throws -> HandleCommandDecoder {
        let tables = try TestBodies.tables(
            bodies: [("world", 0), ("head", 0)],
            joints: [TestJoint(name: "neck", body: 1, axis: Vector3(0, 0, 1), limited: limited, low: low, high: high)])
        return try HandleCommandDecoder(tables: tables,
                                        layout: TestBodies.oneHandleLayout(bone: "head", chain: ["head"]),
                                        reach: FixedReach(metres: 0.5))
    }

    private func singleHingeState(angle: Double = 0, speed: Double = 0) -> FixedBodyState {
        FixedBodyState(qpos: [angle], qvel: [speed],
                       positions: [.zero, .zero],
                       rotations: [.identity, TestBodies.rotationZ(angle)],
                       axes: [Vector3(0, 0, 1)], anchors: [.zero])
    }

    private func zRow(_ angle: Double, strength: Double) -> [Double] {
        TestBodies.row(quaternion: TestBodies.quaternion(rotationVector: Vector3(0, 0, angle)), strength: strength)
    }

    /// A target turned +0.3 rad about the hinge gives a positive control: dq = 0.3 / 1.001,
    /// u = 0.1 x 6.0 x dq = 0.17982018.
    func testTurnAboutHingeGivesControlInTheAskedDirection() throws {
        let decoder = try singleHinge()
        let up = try decoder.decode(command: zRow(0.3, strength: 0.1), gaze: nil, handleMask: nil,
                                    fingerGoals: nil, state: singleHingeState())
        XCTAssertEqual(up[0], 0.1 * 6.0 * 0.3 / 1.001, accuracy: tolerance)
        let down = try decoder.decode(command: zRow(-0.3, strength: 0.1), gaze: nil, handleMask: nil,
                                      fingerGoals: nil, state: singleHingeState())
        XCTAssertEqual(down[0], -0.1 * 6.0 * 0.3 / 1.001, accuracy: tolerance)
    }

    /// The target already reached leaves the speed term alone: u = 0.5 x (0 - 0.3 x 1.0) = -0.15.
    func testSpeedTermBrakesTheHinge() throws {
        let decoder = try singleHinge()
        let ctrl = try decoder.decode(command: zRow(0.2, strength: 0.5), gaze: nil, handleMask: nil,
                                      fingerGoals: nil, state: singleHingeState(angle: 0.2, speed: 1.0))
        XCTAssertEqual(ctrl[0], -0.15, accuracy: 1e-7)
    }

    /// A limited hinge (-0.1..0.1 rad) asked for 0.3 rad steps only to its limit:
    /// dq = 0.1, u = 0.5 x 6.0 x 0.1 = 0.3.
    func testJointLimitCapsTheStep() throws {
        let decoder = try singleHinge(limited: true, low: -0.1, high: 0.1)
        let ctrl = try decoder.decode(command: zRow(0.3, strength: 0.5), gaze: nil, handleMask: nil,
                                      fingerGoals: nil, state: singleHingeState())
        XCTAssertEqual(ctrl[0], 0.3, accuracy: tolerance)
    }

    /// Strength 0 and a false mask entry give zero controls; strength 2 is clipped to 1.
    func testStrengthAndMaskGateTheHandle() throws {
        let decoder = try singleHinge()
        let state = singleHingeState()
        let zero = try decoder.decode(command: zRow(0.3, strength: 0), gaze: nil, handleMask: nil, fingerGoals: nil, state: state)
        XCTAssertEqual(zero, [0])
        let masked = try decoder.decode(command: zRow(0.3, strength: 1), gaze: nil, handleMask: [false], fingerGoals: nil, state: state)
        XCTAssertEqual(masked, [0])
        let one = try decoder.decode(command: zRow(0.01, strength: 1), gaze: nil, handleMask: nil, fingerGoals: nil, state: state)
        let two = try decoder.decode(command: zRow(0.01, strength: 2), gaze: nil, handleMask: nil, fingerGoals: nil, state: state)
        XCTAssertEqual(one[0], two[0], accuracy: tolerance)
        XCTAssertEqual(one[0], 6.0 * 0.01 / 1.001, accuracy: 1e-8)
    }

    /// Two hinges on one body (about z and about x) asked for the rotation vector (0.5, 0, 1.0)
    /// at strength 1: raw u = (5.994, 2.997); the chain is scaled as a unit to (1.0, 0.5), so the
    /// ratio 0.5 between the two survives (action_decoder.py:483-506).
    func testSaturatingChainKeepsItsRatio() throws {
        let tables = try TestBodies.tables(
            bodies: [("world", 0), ("head", 0)],
            joints: [TestJoint(name: "yaw", body: 1, axis: Vector3(0, 0, 1)),
                     TestJoint(name: "roll", body: 1, axis: Vector3(1, 0, 0))])
        let decoder = try HandleCommandDecoder(tables: tables,
                                               layout: TestBodies.oneHandleLayout(bone: "head", chain: ["head"]),
                                               reach: FixedReach(metres: 0.5))
        let state = FixedBodyState(qpos: [0, 0], qvel: [0, 0], positions: [.zero, .zero],
                                   rotations: [.identity, .identity],
                                   axes: [Vector3(0, 0, 1), Vector3(1, 0, 0)], anchors: [.zero, .zero])
        let row = TestBodies.row(quaternion: TestBodies.quaternion(rotationVector: Vector3(0.5, 0, 1.0)), strength: 1)
        let ctrl = try decoder.decode(command: row, gaze: nil, handleMask: nil, fingerGoals: nil, state: state)
        XCTAssertEqual(ctrl[0], 1.0, accuracy: 1e-9)
        XCTAssertEqual(ctrl[1], 0.5, accuracy: 1e-9)
    }

    /// A position handle 10 cm from its target: hinge about z at the origin, body at
    /// (0.1, 0, 0) m, target (0, 10, 0) cm. e = (-0.1, 0.1, 0) m; J = z x (0.1, 0, 0) =
    /// (0, 0.1, 0); stacked [A; 8 J] dq = [0; 8 e] gives dq = 0.64 / 1.641; u = 0.1 x 6.0 x dq =
    /// 0.23400366, positive: the hinge turns the body toward +y, where the target is.
    func testPositionRowTurnsTheBodyTowardTheTarget() throws {
        let tables = try TestBodies.tables(bodies: [("world", 0), ("left_hand", 0)],
                                           joints: [TestJoint(name: "wrist", body: 1, axis: Vector3(0, 0, 1))])
        let layout = TestBodies.oneHandleLayout(name: "wrist_L", bone: "left_hand", chain: ["left_hand"], position: true)
        let decoder = try HandleCommandDecoder(tables: tables, layout: layout, reach: FixedReach(metres: 0.5))
        let state = FixedBodyState(qpos: [0], qvel: [0], positions: [.zero, Vector3(0.1, 0, 0)],
                                   rotations: [.identity, .identity], axes: [Vector3(0, 0, 1)], anchors: [.zero])
        let ctrl = try decoder.decode(command: TestBodies.row(position: [0, 10, 0], strength: 0.1),
                                      gaze: nil, handleMask: nil, fingerGoals: nil, state: state)
        XCTAssertEqual(ctrl[0], 0.1 * 6.0 * 0.64 / 1.641, accuracy: tolerance)
    }

    // The left arm: bodies upper arm (1), lower arm (2, child of 1), hand (3, child of 2), one
    // hinge about z on each (joints 0 hand, 1 lower arm, 2 upper arm, the wrist chain's
    // sequence hand, lower, upper).
    private func leftArm() throws -> HandleCommandDecoder {
        let tables = try TestBodies.tables(
            bodies: [("world", 0), ("left_upper_arm", 0), ("left_lower_arm", 1), ("left_hand", 2)],
            joints: [TestJoint(name: "hand_z", body: 3, axis: Vector3(0, 0, 1)),
                     TestJoint(name: "elbow_z", body: 2, axis: Vector3(0, 0, 1)),
                     TestJoint(name: "shoulder_z", body: 1, axis: Vector3(0, 0, 1))])
        let layout = HandleLayout(
            names: ["wrist_L", "lowerarm_L", "upperarm_L"],
            bones: ["wrist_L": "left_hand", "lowerarm_L": "left_lower_arm", "upperarm_L": "left_upper_arm"],
            chains: ["wrist_L": ["left_hand", "left_lower_arm", "left_upper_arm"],
                     "lowerarm_L": ["left_lower_arm"], "upperarm_L": ["left_upper_arm"]],
            positionHandles: [], fingerDigits: [], eyeJoints: [])
        return try HandleCommandDecoder(tables: tables, layout: layout, reach: FixedReach(metres: 0.5))
    }

    private func leftArmState() -> FixedBodyState {
        FixedBodyState(qpos: [0, 0, 0], qvel: [0, 0, 0], positions: Array(repeating: .zero, count: 4),
                       rotations: Array(repeating: .identity, count: 4),
                       axes: Array(repeating: Vector3(0, 0, 1), count: 3), anchors: Array(repeating: .zero, count: 3))
    }

    /// The arm group's owners and columns (action_decoder.py:312-345): hand joint -> wrist,
    /// elbow -> lowerarm, shoulder -> upperarm; the lower arm's rotation reads the elbow and the
    /// shoulder columns, the upper arm's the shoulder column only.
    func testArmGroupOwnersAndColumns() throws {
        let decoder = try leftArm()
        XCTAssertEqual(decoder.armGroups.count, 1)
        let group = decoder.armGroups[0]
        XCTAssertEqual(group.members, [0, 1, 2])
        XCTAssertEqual(group.owners, [0, 1, 2])
        XCTAssertEqual(group.columns[1], [1, 2])
        XCTAssertEqual(group.columns[2], [2])
        XCTAssertEqual(group.columns[0], [0, 1, 2])
    }

    /// Only the upper arm active, +0.3 rad about z at strength 0.1: one damped solve over the
    /// three wrist-chain columns with only the shoulder column filled: dq_shoulder =
    /// 0.01 x 0.3 / (0.01 + 0.001) = 0.27272727, u = 0.1 x 6.0 x dq = 0.16363636; hand and elbow 0.
    /// Only the lower arm active instead: its two columns share the turn: dq = 0.003 / 0.021 =
    /// 0.14285714 each, u = 0.08571429 on the elbow and on the shoulder (whose owner is not
    /// active, so the largest active strength 0.1 gates it, action_decoder.py:480-482).
    func testArmGroupSolvesTheArmAsOneSystem() throws {
        let decoder = try leftArm()
        let q = TestBodies.quaternion(rotationVector: Vector3(0, 0, 0.3))
        let idle = TestBodies.row(strength: 0)
        let turn = TestBodies.row(quaternion: q, strength: 0.1)
        let upper = try decoder.decode(command: idle + idle + turn, gaze: nil, handleMask: nil,
                                       fingerGoals: nil, state: leftArmState())
        XCTAssertEqual(upper[0], 0, accuracy: tolerance)
        XCTAssertEqual(upper[1], 0, accuracy: tolerance)
        XCTAssertEqual(upper[2], 0.6 * 0.003 / 0.011, accuracy: tolerance)
        let lower = try decoder.decode(command: idle + turn + idle, gaze: nil, handleMask: nil,
                                       fingerGoals: nil, state: leftArmState())
        XCTAssertEqual(lower[0], 0, accuracy: tolerance)
        XCTAssertEqual(lower[1], 0.6 * 0.003 / 0.021, accuracy: tolerance)
        XCTAssertEqual(lower[2], 0.6 * 0.003 / 0.021, accuracy: tolerance)
    }

    /// The eye servo is SET from the gaze pair: azimuth 0.005 rad, eyes at 0: left horizontal
    /// u = 80.0 x 0.005 = 0.4, right horizontal (sign -1) -0.4; with no gaze pair the eye
    /// controls stay 0 (action_decoder.py:508-518).
    func testGazeServoSetsTheEyeMotors() throws {
        let tables = try TestBodies.tables(
            bodies: [("world", 0), ("left_eye", 0)],
            joints: [TestJoint(name: "robot:left_eye_horizontal", body: 1, axis: Vector3(0, 0, 1)),
                     TestJoint(name: "robot:right_eye_horizontal", body: 1, axis: Vector3(0, 0, -1))])
        let layout = TestBodies.oneHandleLayout(
            name: "gaze", bone: "left_eye", chain: [],
            eyeJoints: [HandleLayout.EyeJoint(jointName: "robot:left_eye_horizontal", gazeComponent: 0, sign: 1),
                        HandleLayout.EyeJoint(jointName: "robot:right_eye_horizontal", gazeComponent: 0, sign: -1)])
        let decoder = try HandleCommandDecoder(tables: tables, layout: layout, reach: FixedReach(metres: 0.5))
        XCTAssertEqual(decoder.eyes.count, 2)
        let state = FixedBodyState(qpos: [0, 0], qvel: [0, 0], positions: [.zero, .zero],
                                   rotations: [.identity, .identity],
                                   axes: [Vector3(0, 0, 1), Vector3(0, 0, -1)], anchors: [.zero, .zero])
        let row = TestBodies.row(strength: 0)
        let ctrl = try decoder.decode(command: row, gaze: GazeAngles(azimuth: 0.005, elevation: 0),
                                      handleMask: nil, fingerGoals: nil, state: state)
        XCTAssertEqual(ctrl[0], 0.4, accuracy: tolerance)
        XCTAssertEqual(ctrl[1], -0.4, accuracy: tolerance)
        let none = try decoder.decode(command: row, gaze: nil, handleMask: nil, fingerGoals: nil, state: state)
        XCTAssertEqual(none, [0, 0])
    }

    /// A finger handle with joint goals steps toward the goal angle, not the row's pose:
    /// dq = 0.05 - 0, u = 1.0 x 6.0 x 0.05 = 0.3; a hinge without a goal throws, naming it.
    func testFingerGoalsDriveTheFingerChain() throws {
        let tables = try TestBodies.tables(bodies: [("world", 0), ("left_ffdistal", 0)],
                                           joints: [TestJoint(name: "robot:left_ff_distal", body: 1, axis: Vector3(0, 0, 1))])
        let layout = TestBodies.oneHandleLayout(name: "index_L", bone: "left_ffdistal", chain: ["left_ffdistal"])
        let decoder = try HandleCommandDecoder(tables: tables, layout: layout, reach: FixedReach(metres: 0.5))
        XCTAssertEqual(decoder.fingerHandles, [0])
        let state = singleHingeState()
        let ctrl = try decoder.decode(command: zRow(-0.3, strength: 1), gaze: nil, handleMask: nil,
                                      fingerGoals: [0: 0.05], state: state)
        XCTAssertEqual(ctrl[0], 0.3, accuracy: tolerance)
        XCTAssertThrowsError(try decoder.decode(command: zRow(0, strength: 1), gaze: nil, handleMask: nil,
                                                fingerGoals: [7: 0.05], state: state)) { error in
            XCTAssertEqual(error as? CommandDecoderError, .missingFingerGoal([0]))
        }
    }

    /// 200 commands of numbers drawn from a fixed linear congruential sequence (positions
    /// -100..100 cm, quaternion entries -1..1, strengths -0.5..1.5, speeds -5..5 rad/s) on the
    /// arm: every control lies within -1..1.
    func testEveryControlWithinMinusOneToOne() throws {
        let decoder = try leftArm()
        var seed: UInt64 = 20261104
        func next() -> Double {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Double(seed >> 11) / Double(UInt64(1) << 53)
        }
        for _ in 0..<200 {
            var command: [Double] = []
            for _ in 0..<3 {
                command += [next() * 200 - 100, next() * 200 - 100, next() * 200 - 100]
                command += [next() * 2 - 1, next() * 2 - 1, next() * 2 - 1, next() * 2 - 1, 0, next() * 2 - 0.5]
            }
            var state = leftArmState()
            state.qvel = [next() * 10 - 5, next() * 10 - 5, next() * 10 - 5]
            let ctrl = try decoder.decode(command: command, gaze: nil, handleMask: nil, fingerGoals: nil, state: state)
            for u in ctrl {
                XCTAssertTrue(u >= -1.0 && u <= 1.0, "control \(u) outside -1..1")
            }
        }
    }

    /// The control bounds narrow -1..1 to the force range (-0.65..1, the left knee's in
    /// MIMo_metav2.xml:339, actuation.py:136-138), cast to float32 as MIMo's action space is
    /// (-0.65 becomes -0.6499999761581421): a negative demand of -3 ends at that bound.
    func testForceRangeNarrowsTheBounds() throws {
        let tables = try TestBodies.tables(
            bodies: [("world", 0), ("head", 0)],
            joints: [TestJoint(name: "neck", body: 1, axis: Vector3(0, 0, 1))], forceLow: -0.65, forceHigh: 1)
        let decoder = try HandleCommandDecoder(tables: tables,
                                               layout: TestBodies.oneHandleLayout(bone: "head", chain: ["head"]),
                                               reach: FixedReach(metres: 0.5))
        XCTAssertEqual(decoder.controlLow, [Double(Float(-0.65))])
        XCTAssertEqual(decoder.controlLow[0], -0.6499999761581421, accuracy: 1e-15)
        XCTAssertEqual(decoder.controlHigh, [1])
        let ctrl = try decoder.decode(command: zRow(-0.5, strength: 1), gaze: nil, handleMask: nil,
                                      fingerGoals: nil, state: singleHingeState())
        XCTAssertEqual(ctrl[0], Double(Float(-0.65)), accuracy: tolerance)
    }

    /// A command of the wrong count throws, naming both counts; the contract string is the
    /// Python world's (env.py:3576-3583).
    func testCountCheckAndContract() throws {
        let decoder = try singleHinge()
        XCTAssertThrowsError(try decoder.decode(command: [0, 0, 0], gaze: nil, handleMask: nil,
                                                fingerGoals: nil, state: singleHingeState())) { error in
            XCTAssertEqual(error as? CommandDecoderError, .commandCount(expected: 9, found: 3))
        }
        XCTAssertEqual(decoder.motorContract, "ancestor_arm_axes_signed_limits_v3+finger_joint_servo_v1")
    }

    /// The rotation logarithm returns the rotation vector it was built from (angle 1.118 rad)
    /// and handles the angle-pi branch (a half turn about x gives (pi, 0, 0) up to sign).
    func testRotationLogarithm() {
        let v = Vector3(0.5, 0, 1.0)
        let r = Matrix3(quaternionXYZW: TestBodies.quaternion(rotationVector: v)).logarithm()
        XCTAssertEqual(r.x, 0.5, accuracy: 1e-9)
        XCTAssertEqual(r.y, 0.0, accuracy: 1e-9)
        XCTAssertEqual(r.z, 1.0, accuracy: 1e-9)
        let half = Matrix3(rowMajor: [1, 0, 0, 0, -1, 0, 0, 0, -1]).logarithm()
        XCTAssertEqual(abs(half.x), Double.pi, accuracy: 1e-9)
        XCTAssertEqual(half.y, 0.0, accuracy: 1e-9)
        XCTAssertEqual(half.z, 0.0, accuracy: 1e-9)
    }
}
