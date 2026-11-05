// PhysicsStateTests.swift: the tests of the value types in Sources/UEMAPhysics/PhysicsState.swift
// (work item WI-25). They load no scene: PhysicsState and PhysicsModelTables are built by hand.
// 5 tests. Nothing here was compiled on the machine that wrote it (no Swift toolchain).

import Foundation
import UEMACore
import UEMAPhysics
import XCTest

final class PhysicsStateTests: XCTestCase {
    func makeState(qpos: [Double] = [0, 0, 0.1], time: Double = 1.0) -> PhysicsState {
        PhysicsState(time: time, qpos: qpos, qvel: [0, 0], control: [0.5], sensordata: [9.81],
                     xpos: [0, 0, 0], xquat: [1, 0, 0, 0], actuatorForce: [0.01], contactCount: 2)
    }

    /// A state of finite numbers reports no non-finite field.
    func testFiniteStateHasNoNonFiniteFields() {
        let state = makeState()
        XCTAssertTrue(state.isFinite)
        XCTAssertEqual(state.nonFiniteFields, [])
    }

    /// A NaN in qpos and an infinite time are both named, in declaration sequence.
    func testNonFiniteFieldsAreNamed() {
        let state = makeState(qpos: [0, .nan, 0], time: .infinity)
        XCTAssertFalse(state.isFinite)
        XCTAssertEqual(state.nonFiniteFields, ["time", "qpos"])
    }

    /// A state survives a JSON round trip unchanged.
    func testStateCodableRoundTrip() throws {
        let state = makeState()
        let decoded = try JSONDecoder().decode(PhysicsState.self, from: JSONEncoder().encode(state))
        XCTAssertEqual(decoded, state)
    }

    /// Model tables with a per-joint array of the wrong count throw wrongCount naming it.
    func testModelTablesRejectWrongCount() {
        XCTAssertThrowsError(try PhysicsModelTables(
            timestep: 0.005,
            bodyNames: ["world", "hip"], bodyParentID: [0, 0], bodyMass: [0, 1.2],
            jointNames: ["hip_bend"], jointType: [3, 3], jointBodyID: [1],
            jointQposAddress: [0], jointDofAddress: [0], jointLimited: [true],
            jointRangeLow: [-1], jointRangeHigh: [1],
            qpos0: [0],
            actuatorNames: ["act:hip_bend"], actuatorTransmissionType: [0], actuatorTransmissionID: [0],
            actuatorControlLimited: [true], actuatorControlRangeLow: [-1], actuatorControlRangeHigh: [1],
            actuatorForceLimited: [true], actuatorForceRangeLow: [-1], actuatorForceRangeHigh: [1],
            actuatorGainType: [0], actuatorBiasType: [0], actuatorGainParameter0: [1], actuatorGear0: [0.004],
            sensorNames: [], sensorAddress: [], sensorDimension: [],
            cameraNames: [])) { error in
            XCTAssertEqual(error as? PhysicsError, .wrongCount(name: "jointType", expected: 1, found: 2))
        }
    }

    /// bodyRootID may be empty (hand-built tables) or hold nbody entries; any other count throws.
    func testModelTablesBodyRootCount() throws {
        func tables(bodyRootID: [Int]) throws -> PhysicsModelTables {
            try PhysicsModelTables(
                timestep: 0.005,
                bodyNames: ["world", "hip"], bodyParentID: [0, 0], bodyMass: [0, 1.2],
                jointNames: [], jointType: [], jointBodyID: [],
                jointQposAddress: [], jointDofAddress: [], jointLimited: [],
                jointRangeLow: [], jointRangeHigh: [],
                qpos0: [],
                actuatorNames: [], actuatorTransmissionType: [], actuatorTransmissionID: [],
                actuatorControlLimited: [], actuatorControlRangeLow: [], actuatorControlRangeHigh: [],
                actuatorForceLimited: [], actuatorForceRangeLow: [], actuatorForceRangeHigh: [],
                actuatorGainType: [], actuatorBiasType: [], actuatorGainParameter0: [], actuatorGear0: [],
                sensorNames: [], sensorAddress: [], sensorDimension: [],
                cameraNames: [],
                geomBodyID: [1, 1, 0], bodyRootID: bodyRootID)
        }
        XCTAssertEqual(try tables(bodyRootID: []).geomCount, 3)
        XCTAssertEqual(try tables(bodyRootID: [0, 1]).bodyRootID, [0, 1])
        XCTAssertThrowsError(try tables(bodyRootID: [0])) { error in
            XCTAssertEqual(error as? PhysicsError, .wrongCount(name: "bodyRootID", expected: 2, found: 1))
        }
    }
}
