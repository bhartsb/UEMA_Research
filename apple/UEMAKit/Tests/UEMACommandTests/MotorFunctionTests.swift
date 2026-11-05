// MotorFunctionTests.swift: 6 tests of MotorFunction.swift, the port of `_apply_motor_function`
// (Source/Python/mimo_bridge/env.py:2832-2845) and `body_set_for_mimo_body`
// (Source/Python/mimo_bridge/proprio_adapter.py:46-68). Nothing here was compiled on the
// machine that wrote it (Windows 11, no Swift toolchain).

import XCTest
import UEMACore
import UEMACommand

/// Records every strength write.
private final class RecordingStrengthWriter: ActuatorStrengthWriting {
    var writes: [ActuatorStrength] = []

    func setActuatorStrength(_ strength: ActuatorStrength) throws {
        writes.append(strength)
    }
}

final class MotorFunctionTests: XCTestCase {
    /// MIMo body names map to the 10 body sets as proprio_adapter.py:46-68 maps them.
    func testBodySetOfMimoBodies() {
        let expected: [(String, MotorBodySet?)] = [
            ("left_upper_arm", .armL), ("right_lower_arm", .armR),
            ("left_hand", .handL), ("right_ffdistal", .handR), ("left_lfmetacarpal", .handL),
            ("right_thbase", .handR), ("left_thhub", .handL),
            ("hip", .spine), ("lower_body", .spine), ("upper_body", .spine), ("chest", .spine),
            ("head", .head), ("left_eye", .head), ("right_eye", .head),
            ("left_upper_leg", .legL), ("right_lower_leg", .legR),
            ("left_foot", .footL), ("right_toes", .footR), ("left_big_toe", .footL),
            ("world", nil), ("crib", nil), ("left_fingertip", nil),
        ]
        for (name, set) in expected {
            XCTAssertEqual(MotorBodySet(mimoBodyName: name), set, name)
        }
        XCTAssertEqual(MotorBodySet.allCases.map { $0.rawValue }, Array(0..<10))
    }

    // Bodies: world, left_upper_arm (Arm_L), right_foot (Foot_R), crib (no set); one hinge
    // motor on each of the last three, gain 2.0, force range -0.75..1.
    private func tables(gainType: Int = 0) throws -> ActuatedBodyTables {
        try TestBodies.tables(
            bodies: [("world", 0), ("left_upper_arm", 0), ("right_foot", 0), ("crib", 0)],
            joints: [TestJoint(name: "shoulder", body: 1, axis: Vector3(0, 0, 1)),
                     TestJoint(name: "ankle", body: 2, axis: Vector3(0, 0, 1)),
                     TestJoint(name: "crib_hinge", body: 3, axis: Vector3(0, 0, 1))],
            forceLow: -0.75, forceHigh: 1, gain: 2, gainType: gainType)
    }

    /// factor[a] = capacity of a's body set, 1.0 with no set (env.py:2840-2842).
    func testFactorsFollowTheBodySetOfEachMotor() throws {
        let modulator = try MotorFunctionModulator(tables: tables())
        XCTAssertEqual(modulator.actuatorBodySet, [.armL, .footR, nil])
        var capacities = Array(repeating: 1.0, count: 10)
        capacities[MotorBodySet.armL.rawValue] = 0.4
        capacities[MotorBodySet.footR.rawValue] = 0.9
        XCTAssertEqual(try modulator.factors(capacities: capacities), [0.4, 0.9, 1.0])
    }

    /// gain0 and both force-range columns are the native values times the factor
    /// (env.py:2843-2845): 2.0 x 0.4 = 0.8; -0.75 x 0.4 = -0.3; 1 x 0.4 = 0.4.
    func testModulatedGainAndForceRange() throws {
        let modulator = try MotorFunctionModulator(tables: tables())
        var capacities = Array(repeating: 1.0, count: 10)
        capacities[MotorBodySet.armL.rawValue] = 0.4
        let strength = try modulator.modulated(capacities: capacities)
        XCTAssertEqual(strength.gain[0], 0.8, accuracy: 1e-12)
        XCTAssertEqual(strength.forceLow[0], -0.3, accuracy: 1e-12)
        XCTAssertEqual(strength.forceHigh[0], 0.4, accuracy: 1e-12)
        XCTAssertEqual(strength.gain[2], 2.0, accuracy: 1e-12)
        XCTAssertEqual(strength.forceLow[2], -0.75, accuracy: 1e-12)
    }

    /// The modulation always starts from the native tables, never from its last output:
    /// two calls at capacity 0.5 give the same 1.0 gain, not 0.5 (env.py:2843 restores the
    /// native gain before scaling).
    func testModulationStartsFromNativeValues() throws {
        let modulator = try MotorFunctionModulator(tables: tables())
        var capacities = Array(repeating: 1.0, count: 10)
        capacities[MotorBodySet.armL.rawValue] = 0.5
        let first = try modulator.modulated(capacities: capacities)
        let second = try modulator.modulated(capacities: capacities)
        XCTAssertEqual(first, second)
        XCTAssertEqual(second.gain[0], 1.0, accuracy: 1e-12)
    }

    /// A gain type other than fixed is refused (env.py:2835-2836), naming every actuator; a
    /// capacity array of 9 numbers is refused.
    func testRefusals() throws {
        XCTAssertThrowsError(try MotorFunctionModulator(tables: tables(gainType: 1))) { error in
            XCTAssertEqual(error as? MotorFunctionError, .notFixedGainBiasFree([0, 1, 2]))
        }
        let modulator = try MotorFunctionModulator(tables: tables())
        XCTAssertThrowsError(try modulator.factors(capacities: Array(repeating: 1.0, count: 9))) { error in
            XCTAssertEqual(error as? MotorFunctionError, .capacityCount(expected: 10, found: 9))
        }
    }

    /// The applier writes under `modulate` and writes nothing under `off`.
    func testApplierModes() throws {
        let modulator = try MotorFunctionModulator(tables: tables())
        let capacities = FixedCapacities(Array(repeating: 0.5, count: 10))
        let writer = RecordingStrengthWriter()
        let on = MotorFunctionApplier(modulator: modulator, mode: .modulate, capacities: capacities, writer: writer)
        try on.apply()
        XCTAssertEqual(writer.writes.count, 1)
        XCTAssertEqual(on.lastWritten?.gain[1] ?? -1, 1.0, accuracy: 1e-12)
        let off = MotorFunctionApplier(modulator: modulator, mode: .off, capacities: capacities, writer: writer)
        try off.apply()
        XCTAssertEqual(writer.writes.count, 1)
        XCTAssertNil(off.lastWritten)
    }
}
