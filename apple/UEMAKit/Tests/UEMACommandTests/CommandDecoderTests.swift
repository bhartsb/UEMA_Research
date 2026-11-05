// CommandDecoderTests.swift: 10 tests of CommandDecoder (UEMACore's CommandDecoding),
// MotorCommandWriter over PlantPhysics (the 90-motor test plant of CommandTestBodies.swift) and
// the adapters over UEMAPhysics.
// The held-command test follows the pattern of Source/Python/mimo_bridge/tests/
// test_command_moves_body.py (hold one command for 25 env steps, check the named body part
// moved in the asked direction, check every control within -1..1); the body is the test plant,
// not MIMo, so the moved angle is a property of the plant's inertia 0.1 (a test constant).
// Nothing here was compiled on the machine that wrote it (Windows 11, no Swift toolchain).

import XCTest
import UEMACore
import UEMAPhysics
import UEMACommand

final class CommandDecoderTests: XCTestCase {
    private func makeDecoder(physics: PlantPhysics) throws -> CommandDecoder {
        let engine = try HandleCommandDecoder(tables: PlantPhysics.tables(), layout: PlantPhysics.layout(),
                                              reach: FixedReach(metres: 0.5))
        return try CommandDecoder(engine: engine, physics: physics, fingerJointIDs: Array(1...44))
    }

    /// The head row turned `angle` rad about z at `strength`, every other row strength 0, at
    /// `.centimetres` scale.
    private func headCommand(angle: Double, strength: Float) throws -> Command {
        var command = Command.zero(scale: .centimetres)
        let q = TestBodies.quaternion(rotationVector: Vector3(0, 0, angle)).map { Float($0) }
        try command.setRow(.head, to: [0, 0, 0] + q + [0, strength])
        return command
    }

    /// A held command turns the named body part in the asked direction: 25 env steps (5 model
    /// steps) of the head row turned +0.3 rad move the plant's hinge by more than 1e-3 rad in
    /// the positive direction, -0.3 rad in the negative direction, and every control of every
    /// env step lies within -1..1 (test_command_moves_body.py's pattern; WI-37 acceptance).
    func testHeldCommandMovesTheBodyPartInTheAskedDirection() throws {
        for (angle, sign) in [(0.3, 1.0), (-0.3, -1.0)] {
            let physics = PlantPhysics()
            let writer = MotorCommandWriter(decoder: try makeDecoder(physics: physics), physics: physics, motorFunction: nil)
            let command = try headCommand(angle: angle, strength: 0.5)
            for _ in 0..<25 {
                let controls = try writer.stepEnvStep(command, mask: .allTrue, gaze: .straightAhead, fingerGoals: nil) { _ in }
                XCTAssertTrue(controls.isWithinRange)
            }
            XCTAssertGreaterThan(sign * physics.angle, 1e-3, "hinge angle \(physics.angle) for a target of \(angle) rad")
            XCTAssertEqual(physics.writtenControls.count, 25)
            XCTAssertEqual(physics.time, 25 * ModelStep.envStepSeconds, accuracy: 1e-12)
        }
    }

    /// Half the Arm_L capacity halves the plant's gain (MotorFunction.swift), so the same held
    /// command moves the hinge less; the modulation was written once per env step.
    func testModulatedBodyMovesLessThanTheRestedOne() throws {
        func run(armCapacity: Double) throws -> (angle: Double, writes: Int) {
            let physics = PlantPhysics()
            var capacities = Array(repeating: 1.0, count: MotorBodySet.count)
            capacities[MotorBodySet.armL.rawValue] = armCapacity
            let applier = MotorFunctionApplier(modulator: try MotorFunctionModulator(tables: PlantPhysics.tables()),
                                               mode: .modulate,
                                               capacities: FixedCapacities(capacities),
                                               writer: physics)
            let writer = MotorCommandWriter(decoder: try makeDecoder(physics: physics), physics: physics, motorFunction: applier)
            let command = try headCommand(angle: 0.3, strength: 0.5)
            for _ in 0..<25 {
                try writer.stepEnvStep(command, mask: .allTrue, gaze: nil, fingerGoals: nil) { _ in }
            }
            XCTAssertEqual(physics.gain[0], armCapacity, accuracy: 1e-12)
            return (physics.angle, physics.strengthWrites)
        }
        let rested = try run(armCapacity: 1.0)
        let tired = try run(armCapacity: 0.5)
        XCTAssertGreaterThan(rested.angle, 1e-3)
        XCTAssertGreaterThan(tired.angle, 1e-3)
        XCTAssertLessThan(tired.angle, rested.angle)
        XCTAssertEqual(tired.writes, 25)
    }

    /// The decoder refuses a `.normalized` command (UEMACore's contract).
    func testNormalizedCommandIsRefused() throws {
        let decoder = try makeDecoder(physics: PlantPhysics())
        XCTAssertThrowsError(try decoder.decode(.zero(scale: .normalized), mask: .allTrue, gaze: nil, fingerGoals: nil)) { error in
            XCTAssertEqual(error as? UEMACoreError, .wrongScale(expected: .centimetres, found: .normalized))
        }
    }

    /// A false mask entry for the head leaves all 90 controls at 0; the result is the decoder's
    /// lastControls.
    func testMaskSilencesTheHandle() throws {
        let decoder = try makeDecoder(physics: PlantPhysics())
        var flags = Array(repeating: true, count: HandleMask.count)
        flags[BodyHandle.head.rawValue] = false
        let controls = try decoder.decode(try headCommand(angle: 0.3, strength: 1), mask: try HandleMask(values: flags),
                                          gaze: nil, fingerGoals: nil)
        XCTAssertEqual(controls, MotorControls.zero)
        XCTAssertEqual(decoder.lastControls, controls)
    }

    /// The decoder refuses an engine that does not make 90 controls, and a finger id list
    /// that is not 44 long.
    func testShapeChecksAtInit() throws {
        let small = try HandleCommandDecoder(
            tables: TestBodies.tables(bodies: [("world", 0), ("head", 0)],
                                      joints: [TestJoint(name: "neck", body: 1, axis: Vector3(0, 0, 1))]),
            layout: TestBodies.oneHandleLayout(bone: "head", chain: ["head"]),
            reach: FixedReach(metres: 0.5))
        XCTAssertThrowsError(try CommandDecoder(engine: small, physics: PlantPhysics(), fingerJointIDs: Array(0..<44)))
        let full = try HandleCommandDecoder(tables: PlantPhysics.tables(), layout: PlantPhysics.layout(),
                                            reach: FixedReach(metres: 0.5))
        XCTAssertThrowsError(try CommandDecoder(engine: full, physics: PlantPhysics(), fingerJointIDs: [1, 2]))
    }

    /// The 44 finger joint names are the receipt's hand_response.joint_names (44, the last
    /// "robot:right_th_distal"), each once; a table without them throws naming the first.
    func testFingerJointNames() throws {
        XCTAssertEqual(CommandDecoder.fingerJointNames.count, FingerJointGoals.count)
        XCTAssertEqual(Set(CommandDecoder.fingerJointNames).count, 44)
        XCTAssertEqual(CommandDecoder.fingerJointNames.first, "robot:left_ff_side")
        XCTAssertEqual(CommandDecoder.fingerJointNames.last, "robot:right_th_distal")
        XCTAssertThrowsError(try CommandDecoder.fingerJointIDs(in: PlantPhysics.tables())) { error in
            XCTAssertEqual(error as? CommandDecoderError, .missingBody("finger joint robot:left_ff_side"))
        }
    }

    /// ActuatedBodyTables(modelTables:) passes every field through: tables with a limited
    /// joint (-0.5..0.7 rad) and an asymmetric force range (-0.65..1) come back equal.
    func testTablesFromPhysicsModelTables() throws {
        let tables = try TestBodies.tables(
            bodies: [("world", 0), ("head", 0)],
            joints: [TestJoint(name: "neck", body: 1, axis: Vector3(0, 0, 1), limited: true, low: -0.5, high: 0.7)],
            forceLow: -0.65, forceHigh: 1, gain: 2)
        let mirrored = try ActuatedBodyTables(modelTables: PhysicsModelTables.mirroring(tables, nq: 1))
        XCTAssertEqual(mirrored, tables)
    }

    /// ScratchForwardKinematics returns the scratch query's 3 numbers as a Vector3 and refuses
    /// another count; CornerSweepReach over a body that never moves gives the 0.01 m floor
    /// (action_decoder.py:241); ActuatorParameterStrengthWriter writes the gain, then the force
    /// range, unchanged.
    func testUEMAPhysicsAdapters() throws {
        let side = FakeModelSide()
        let kinematics = ScratchForwardKinematics(query: side)
        XCTAssertEqual(try kinematics.bodyPosition(ofBody: 1, withJointAngles: [0: 0.2]), Vector3(0.1, 0.2, 0.3))
        let tables = try TestBodies.tables(bodies: [("world", 0), ("head", 0)],
                                           joints: [TestJoint(name: "neck", body: 1, axis: Vector3(0, 0, 1))])
        XCTAssertEqual(try CornerSweepReach(kinematics: kinematics, tables: tables).reachMetres(body: 1, jointIDs: [0]),
                       0.01, accuracy: 1e-15)
        side.position = [1, 2]
        XCTAssertThrowsError(try kinematics.bodyPosition(ofBody: 1, withJointAngles: [:])) { error in
            XCTAssertEqual(error as? CommandDecoderError,
                           .inconsistentTables("scratch position of body 1 has 2 numbers, expected 3"))
        }
        let strength = ActuatorStrength(gain: [0.8], forceLow: [-0.3], forceHigh: [0.4])
        try ActuatorParameterStrengthWriter(side).setActuatorStrength(strength)
        XCTAssertEqual(side.calls, ["gain", "forceRange"])
        XCTAssertEqual(side.gain, [0.8])
        XCTAssertEqual(side.forceLow, [-0.3])
        XCTAssertEqual(side.forceHigh, [0.4])
    }

    /// CommandDecoder.released reads the model tables and builds layout trunk_v1, so a body
    /// without the MIMo bones is refused naming the first missing one (wrist_L's left_hand).
    func testReleasedNeedsTheMimoBones() throws {
        XCTAssertThrowsError(try CommandDecoder.released(physics: PlantPhysics())) { error in
            XCTAssertEqual(error as? CommandDecoderError, .missingBody("handle wrist_L: bone left_hand"))
        }
    }

    /// The decoder's copy of the body state checks every array against the physics's own
    /// sizes: the plant passes; a physics whose xmat holds 9 numbers for 2 bodies is refused,
    /// naming xmat (a short array would otherwise read as the identity rotation).
    func testBodyStateCountsAreChecked() throws {
        XCTAssertNoThrow(try PhysicsBodyState(validating: PlantPhysics()))
        XCTAssertThrowsError(try PhysicsBodyState(validating: ShortXmatPhysics())) { error in
            XCTAssertEqual(error as? CommandDecoderError,
                           .stateTooShort("xmat has 9 numbers, the physics sizes give 18"))
        }
    }
}

/// A two-body, one-joint physics whose xmat holds one body's 9 numbers instead of 18.
private final class ShortXmatPhysics: PhysicsReading {
    var sizes: PhysicsSizes { PhysicsSizes(nq: 1, nv: 1, nu: 1, nbody: 2, njnt: 1, nsensordata: 0, ngeom: 0, ncam: 0) }
    var time: Double { 0 }
    var qpos: [Double] { [0] }
    var qvel: [Double] { [0] }
    var sensordata: [Double] { [] }
    var xpos: [Double] { Array(repeating: 0, count: 6) }
    var xquat: [Double] { [1, 0, 0, 0, 1, 0, 0, 0] }
    var xmat: [Double] { Matrix3.identity.m }
    var xaxis: [Double] { [0, 0, 1] }
    var xanchor: [Double] { [0, 0, 0] }
    var actuatorForce: [Double] { [0] }
    var control: [Double] { [0] }
    var contacts: [Contact] { [] }
    func id(of kind: MuJoCoObjectKind, named name: String) -> Int? { nil }
    func bodyVelocity(bodyId: Int) -> [Double] { Array(repeating: 0, count: 6) }
}
