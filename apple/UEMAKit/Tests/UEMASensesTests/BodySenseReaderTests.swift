// BodySenseReaderTests.swift: hand-computed checks of the proprioception, motor and
// vestibular readers (work item WI-27; acceptance "on a fake Physics with known arrays the
// rows equal hand-computed rows; counts 26, 29, 26 x 10").
//
// THE FAKE BODY. Body b (1 ... 26) carries row b - 1. xpos[b] = (0.01 b, 0.02 b, -0.005 b) m,
// so row b - 1's position is (b, 2 b, -0.5 b) cm; the linear velocity of body b is
// (0.001 b, 0, -0.002 b) m/s, so (0.1 b, 0, -0.2 b) cm/s. Every quaternion is the identity
// except the head (body 13) w x y z = (0.6, 0, 0, 0.8) and the left eye (body 24)
// (0.8, 0, 0.6, 0). Four actuators: forces (2, -1, 5, 0.5) N on qvel indices 0 ... 3 holding
// (0.5, 3, 7, -2); body sets Hand_L, Head, none, Spine; gears (2, 3, 4, 5).
//   effort (no_gear)   Hand_L |2 x 0.5| = 1, Head |-1 x 3| = 3, Spine |0.5 x -2| = 1
//   one env step       cap = 1 + 0.01 (0 - 0.02 e): Hand_L 0.9998, Head 0.9994, Spine 0.9998
//   a second env step  Hand_L 0.9998 + 0.01 (0.05 x 0.0002 - 0.02) = 0.9996001
//   energy             ph_work 5 x 0.01 = 0.05; pcc_hold 8.5 x 0.01 x 0.01 = 0.00085; total
//                      0.05085
//   gaze row           conj(head) x eye in x y z w = (0.48, 0.36, -0.64, 0.48)
// THE VESTIBULAR CASE. The expected 29 numbers were computed by the Python reference
// `vestibular_from_raw` (Source/Python/mimo_bridge/vestibular_adapter.py:268-292) on the same
// raw values (November 4, 2026, numpy): see `testVestibularFieldsMatchPythonReference`.
// The fatigue numbers above were also recomputed with the FatigueModel formula of
// proprio_adapter.py:112-116 in numpy: 0.9998, 0.9994, 0.9996001, 0.9988003.
// THE TABLES CASE. A five-body `PhysicsModelTables` (world, hip, chest, head, crib) with
// body_parentid (0, 0, 1, 2, 0): by mjmodel.h's "ancestor that is direct child of world" the
// roots are (0, 1, 1, 1, 4), so the own bodies of hip's root are 1, 2 and 3.
// Nothing in this file was compiled on the machine that wrote it (no Swift toolchain).

import Foundation
import XCTest
import UEMACore
import UEMAPhysics
import UEMASenses

// MARK: - Fakes

/// A physics state of hand-written arrays.
final class FakeBodyPhysics: PhysicsReading {
    var sizes: PhysicsSizes
    var time: Double = 0.01
    var qpos: [Double] = []
    var qvel: [Double] = []
    var sensordata: [Double] = []
    var xpos: [Double] = []
    var xquat: [Double] = []
    var xmat: [Double] = []
    var xaxis: [Double] = []
    var xanchor: [Double] = []
    var actuatorForce: [Double] = []
    var control: [Double] = []
    var contacts: [Contact] = []
    var velocities: [[Double]] = []

    init(nbody: Int, nq: Int, nv: Int, nu: Int) {
        sizes = PhysicsSizes(nq: nq, nv: nv, nu: nu, nbody: nbody, njnt: 0, nsensordata: 0, ngeom: 0, ncam: 0)
        qpos = Array(repeating: 0, count: nq)
        qvel = Array(repeating: 0, count: nv)
        xpos = Array(repeating: 0, count: 3 * nbody)
        xmat = Array(repeating: 0, count: 9 * nbody)
        xquat = (0 ..< nbody).flatMap { _ in [1.0, 0.0, 0.0, 0.0] }
        actuatorForce = Array(repeating: 0, count: nu)
        control = Array(repeating: 0, count: nu)
        velocities = Array(repeating: [0, 0, 0, 0, 0, 0], count: nbody)
    }

    func id(of kind: MuJoCoObjectKind, named name: String) -> Int? { nil }

    func bodyVelocity(bodyId: Int) -> [Double] { velocities[bodyId] }

    /// The fake body of the header.
    static func proprioceptionBody() -> FakeBodyPhysics {
        let physics = FakeBodyPhysics(nbody: 28, nq: 4, nv: 4, nu: 4)
        for body in 0 ..< 28 {
            let b = Double(body)
            physics.xpos[3 * body] = 0.01 * b
            physics.xpos[3 * body + 1] = 0.02 * b
            physics.xpos[3 * body + 2] = -0.005 * b
            physics.velocities[body] = [9, 9, 9, 0.001 * b, 0, -0.002 * b]
        }
        physics.xquat.replaceSubrange(4 * 13 ..< 4 * 13 + 4, with: [0.6, 0, 0, 0.8])
        physics.xquat.replaceSubrange(4 * 24 ..< 4 * 24 + 4, with: [0.8, 0, 0.6, 0])
        physics.actuatorForce = [2, -1, 5, 0.5]
        physics.qvel = [0.5, 3, 7, -2]
        return physics
    }
}

/// A sampler that returns one fixed sample.
final class FixedSampler: ProprioceptionSampling {
    let fixed: ProprioceptionSample
    var fatigue = FatigueModel()
    var sampleCalls = 0

    init(_ fixed: ProprioceptionSample) {
        self.fixed = fixed
    }

    func sample(_ physics: any PhysicsReading) throws -> ProprioceptionSample {
        sampleCalls += 1
        return fixed
    }

    func restore(fatigue: FatigueModel) {
        self.fatigue = fatigue
    }

    func reset() {}
}

/// A support provider of fixed records.
final class FixedSupport: SupportContactProviding {
    let records: [SupportContact]

    init(_ records: [SupportContact]) {
        self.records = records
    }

    func supportContacts(_ physics: any PhysicsReading) throws -> [SupportContact] { records }
}

/// A model description of hand-written tables: bodies 0 world, 1 ... 26 the row bodies,
/// 27 hip; six joints; five actuators; two sensors.
struct FakeModelDescription: MuJoCoModelDescribing {
    let bodyNames: [String] = ["world"] + ProprioceptionLayout.bodyNames + ["hip"]
    let jointNames = ["robot:head_swivel", "robot:head_tilt", "robot:head_tilt_side",
                      "robot:left_eye_horizontal", "robot:left_ff_joint", "robot:hip_free"]
    let jointBodies = [13, 13, 13, 24, 3, 27]
    let jointTypes: [MuJoCoJointType] = [.hinge, .hinge, .hinge, .hinge, .hinge, .free]
    let jointRanges = [LimitPair(lower: -1, upper: 1), LimitPair(lower: -0.5, upper: 1.5),
                       LimitPair(lower: -2, upper: 2), LimitPair(lower: -0.7, upper: 0.7),
                       LimitPair(lower: 0, upper: 1), LimitPair(lower: 0, upper: 0)]
    let jointQpos = [7, 8, 9, 10, 11, 0]
    let jointDofs = [0, 1, 2, 3, 4, 5]
    let actuatorNames = ["act:head_swivel", "act:head_tilt", "act:head_tilt_side", "act:eye", "act:finger"]
    let actuatorJoints = [0, 1, 2, 3, 4]
    let gears = [1.0, 2.0, 0.5, 4.0, 3.0]
    let forceRanges = [LimitPair(lower: -2, upper: 4), LimitPair(lower: -5, upper: 0),
                       LimitPair(lower: 0, upper: 0), LimitPair(lower: -1, upper: 1),
                       LimitPair(lower: -1, upper: 1)]
    var transmissionTypes: [MuJoCoTransmissionType] = Array(repeating: .joint, count: 5)

    var sizes: PhysicsSizes {
        PhysicsSizes(nq: 12, nv: 6, nu: 5, nbody: 28, njnt: 6, nsensordata: 6, ngeom: 0, ncam: 0)
    }

    func id(of kind: MuJoCoObjectKind, named name: String) -> Int? {
        switch kind {
        case .body: return bodyNames.firstIndex(of: name)
        case .joint: return jointNames.firstIndex(of: name)
        case .actuator: return actuatorNames.firstIndex(of: name)
        case .sensor: return ["vestibular_acc", "vestibular_gyro"].firstIndex(of: name)
        case .geom, .camera: return nil
        }
    }

    func name(of kind: MuJoCoObjectKind, id: Int) -> String? {
        switch kind {
        case .body: return bodyNames[id]
        case .joint: return jointNames[id]
        case .actuator: return actuatorNames[id]
        default: return nil
        }
    }

    func jointBodyId(_ joint: Int) -> Int { jointBodies[joint] }
    func jointType(_ joint: Int) -> MuJoCoJointType { jointTypes[joint] }
    func jointIsLimited(_ joint: Int) -> Bool { jointTypes[joint] == .hinge }
    func jointRange(_ joint: Int) -> LimitPair { jointRanges[joint] }
    func jointQposAddress(_ joint: Int) -> Int { jointQpos[joint] }
    func jointDofAddress(_ joint: Int) -> Int { jointDofs[joint] }
    func actuatorTransmissionType(_ actuator: Int) -> MuJoCoTransmissionType { transmissionTypes[actuator] }
    func actuatorTransmissionTarget(_ actuator: Int) -> Int { actuatorJoints[actuator] }
    func actuatorGear(_ actuator: Int) -> Double { gears[actuator] }
    func actuatorForceRange(_ actuator: Int) -> LimitPair { forceRanges[actuator] }
    func bodyMass(_ body: Int) -> Double { 0.1 }
    func bodyRootId(_ body: Int) -> Int { body == 0 ? 0 : 27 }
    func geomBodyId(_ geom: Int) -> Int { 0 }
    func sensorAddress(_ sensor: Int) -> Int { 3 * sensor }
    func sensorDimension(_ sensor: Int) -> Int { 3 }
}

// MARK: - Tests

final class BodySenseReaderTests: XCTestCase {
    /// The layout of the fake body: row i on body i + 1.
    private func fakeLayout() throws -> ProprioceptionLayout {
        try ProprioceptionLayout(
            bodyIds: Array(1 ... 26),
            actuatorBodySets: [.handL, .head, nil, .spine],
            actuatorDofs: [0, 1, 2, 3],
            actuatorGears: [2, 3, 4, 5])
    }

    private func assertRow(_ values: [Float], row: Int, columns: Int, equals expected: [Float],
                           accuracy: Float = 1e-5, file: StaticString = #filePath, line: UInt = #line) {
        for (column, value) in expected.enumerated() {
            XCTAssertEqual(values[row * columns + column], value, accuracy: accuracy,
                           "row \(row) column \(column)", file: file, line: line)
        }
    }

    // MARK: Proprioception

    func testProprioceptionRowsCountAndHandComputedValues() throws {
        let physics = FakeBodyPhysics.proprioceptionBody()
        let reader = ProprioceptionReader(tracker: BodyStateTracker(layout: try fakeLayout()))
        try reader.observeEnvStep(physics)
        let reading = try reader.read(physics, at: ModelStep(count: 1))
        XCTAssertEqual(reading.sense, .proprioception)
        XCTAssertEqual(reading.primary.shape, [26, 12])
        XCTAssertNil(reading.auxiliary)
        let v = reading.primary.values
        // row 0 (body 1): Arm_L, untouched capacity 1
        assertRow(v, row: 0, columns: 12, equals: [0, 0, 0, 1, 1, 2, -0.5, 0.1, 0, -0.2, 1, 0])
        // row 1 (body 2): Hand_L
        assertRow(v, row: 1, columns: 12, equals: [0, 0, 0, 1, 2, 4, -1, 0.2, 0, -0.4, 0.9998, 1])
        // row 12 (body 13): the head, quaternion x y z w (0, 0, 0.8, 0.6)
        assertRow(v, row: 12, columns: 12, equals: [0, 0, 0.8, 0.6, 13, 26, -6.5, 1.3, 0, -2.6, 0.9994, 12])
        // row 23 (body 24): the gaze row, head-relative
        assertRow(v, row: 23, columns: 12, equals: [0.48, 0.36, -0.64, 0.48, 24, 48, -12, 2.4, 0, -4.8, 0.9994, 23])
        // row 25 (body 26): the chest, Spine
        assertRow(v, row: 25, columns: 12, equals: [0, 0, 0, 1, 26, 52, -13, 2.6, 0, -5.2, 0.9998, 25])
    }

    func testGazeRowProductAlone() {
        var bones = Array(repeating: Float(0), count: 26 * 11)
        bones.replaceSubrange(12 * 11 ..< 12 * 11 + 4, with: [0, 0, 0.8, 0.6])
        bones.replaceSubrange(23 * 11 ..< 23 * 11 + 4, with: [0, 0.6, 0, 0.8])
        BodyStateTracker.makeGazeRowHeadRelative(&bones)
        assertRow(bones, row: 23, columns: 11, equals: [0.48, 0.36, -0.64, 0.48], accuracy: 1e-6)
        // the head row is unchanged
        assertRow(bones, row: 12, columns: 11, equals: [0, 0, 0.8, 0.6], accuracy: 0)
    }

    func testFatigueAdvancesOncePerEnvStepWhateverReaderAsks() throws {
        let physics = FakeBodyPhysics.proprioceptionBody()
        let tracker = BodyStateTracker(layout: try fakeLayout())
        let proprioception = ProprioceptionReader(tracker: tracker)
        let motor = MotorReader(tracker: tracker, executed: ExecutedCommandRecord())
        try motor.observeEnvStep(physics)
        try proprioception.observeEnvStep(physics)
        try proprioception.observeEnvStep(physics)
        XCTAssertEqual(tracker.fatigue.capacities[BodySet.handL.rawValue], 0.9998, accuracy: 1e-12)
        XCTAssertEqual(tracker.fatigue.accumulated[BodySet.head.rawValue], 0.03, accuracy: 1e-12)
        physics.time = 0.02
        try proprioception.observeEnvStep(physics)
        XCTAssertEqual(tracker.fatigue.capacities[BodySet.handL.rawValue], 0.9996001, accuracy: 1e-12)
        XCTAssertEqual(tracker.fatigue.capacities[BodySet.head.rawValue], 0.9988003, accuracy: 1e-12)
        XCTAssertEqual(tracker.fatigue.accumulated[BodySet.spine.rawValue], 0.02, accuracy: 1e-12)
        proprioception.reset()
        XCTAssertEqual(tracker.fatigue, FatigueModel())
    }

    func testEnergyOfOneEnvStep() throws {
        let physics = FakeBodyPhysics.proprioceptionBody()
        let tracker = BodyStateTracker(layout: try fakeLayout())
        let sample = try tracker.sample(physics)
        XCTAssertEqual(sample.energy[0], 0.05, accuracy: 1e-7)
        XCTAssertEqual(sample.energy[1], 0.00085, accuracy: 1e-8)
        XCTAssertEqual(sample.energy[2], 0.05085, accuracy: 1e-7)
        XCTAssertEqual(sample.bodySets.count, 20)
        XCTAssertEqual(sample.bodySets[2 * BodySet.head.rawValue], 0.9994, accuracy: 1e-6)
        XCTAssertEqual(sample.bodySets[2 * BodySet.head.rawValue + 1], 0.03, accuracy: 1e-7)
    }

    func testWithGearEffortMultipliesByGear() throws {
        let physics = FakeBodyPhysics.proprioceptionBody()
        let tracker = BodyStateTracker(layout: try fakeLayout(), effortMode: .withGear)
        let effort = try tracker.effortPerSet(actuatorForce: physics.actuatorForce, qvel: physics.qvel)
        // Hand_L |2 x 2 x 0.5| = 2; Head |-1 x 3 x 3| = 9; Spine |0.5 x 5 x -2| = 5; the
        // actuator of no set (5 N on 7 rad/s) adds nothing
        XCTAssertEqual(effort, [0, 0, 2, 0, 5, 9, 0, 0, 0, 0])
    }

    func testFatigueModelClipsAtZeroAndRestores() throws {
        var model = FatigueModel()
        var effort = Array(repeating: 0.0, count: 10)
        effort[0] = 1e6
        try model.step(effort: effort, dt: 0.01)
        XCTAssertEqual(model.capacities[0], 0)
        XCTAssertEqual(model.accumulated[0], 1e4, accuracy: 1e-9)
        XCTAssertThrowsError(try model.step(effort: [1], dt: 0.01))
        let restored = try FatigueModel(capacities: Array(repeating: 0.5, count: 10), accumulated: Array(repeating: 2, count: 10))
        XCTAssertEqual(restored.capacities[9], 0.5)
        XCTAssertThrowsError(try FatigueModel(capacities: [1], accumulated: [0]))
    }

    // MARK: Motor

    func testMotorRowsCountAndHandComputedValues() throws {
        let physics = FakeBodyPhysics.proprioceptionBody()
        var command = Command.zero(scale: .centimetres)
        for handle in BodyHandle.allCases {
            command[handle, .radius] = 0.1 * Float(handle.rawValue)
            command[handle, .strength] = 0.5
        }
        var maskValues = Array(repeating: true, count: 26)
        maskValues[3] = false
        let record = ExecutedCommandRecord()
        record.record(command, mask: try HandleMask(values: maskValues))
        let reader = MotorReader(tracker: BodyStateTracker(layout: try fakeLayout()), executed: record)
        let reading = try reader.read(physics, at: ModelStep(count: 1))
        XCTAssertEqual(reading.sense, .motor)
        XCTAssertEqual(reading.primary.shape, [26, 10])
        let v = reading.primary.values
        assertRow(v, row: 3, columns: 10, equals: [4, 8, -2, 0, 0, 0, 1, 0.3, 0.5, 0])
        assertRow(v, row: 23, columns: 10, equals: [24, 48, -12, 0.48, 0.36, -0.64, 0.48, 2.3, 0.5, 1])
        assertRow(v, row: 25, columns: 10, equals: [26, 52, -13, 0, 0, 0, 1, 2.5, 0.5, 1])
    }

    func testMotorRowsBeforeAnyCommand() throws {
        let physics = FakeBodyPhysics.proprioceptionBody()
        let reader = MotorReader(tracker: BodyStateTracker(layout: try fakeLayout()), executed: ExecutedCommandRecord())
        let v = try reader.read(physics, at: .start).primary.values
        assertRow(v, row: 12, columns: 10, equals: [13, 26, -6.5, 0, 0, 0.8, 0.6, 0, 0, 1])
    }

    func testMotorAuxiliaryEnergyAndCapability() throws {
        let physics = FakeBodyPhysics.proprioceptionBody()
        let reader = MotorReader(tracker: BodyStateTracker(layout: try fakeLayout()), executed: ExecutedCommandRecord())
        let reading = try reader.read(physics, at: ModelStep(count: 1))
        let auxiliary = try XCTUnwrap(reading.auxiliary)
        XCTAssertEqual(auxiliary.shape, [24])
        let expected: [Float] = [
            0.05, 0.00085, 0.05085,
            1, 1, 0.9998, 1, 0.9998, 0.9994, 1, 1, 1, 1,
            0, 0, 0.01, 0, 0.01, 0.03, 0, 0, 0, 0,
            0.05085,
        ]
        for (index, value) in expected.enumerated() {
            XCTAssertEqual(auxiliary.values[index], value, accuracy: 1e-6, "auxiliary \(index)")
        }
        XCTAssertEqual(MotorReader.auxiliaryIndicesOverEnergyMax, [0, 1, 2, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    }

    // MARK: Vestibular

    /// The layout of the vestibular case (see the Python reference in the test below).
    private func vestibularLayout(supportGroups: [Int] = [-1, 0, 1, 2, 3, -1]) throws -> VestibularLayout {
        try VestibularLayout(
            accelerometerAddresses: [0, 1, 2], gyroAddresses: [3, 4, 5],
            neckQposAddresses: [7, 8, 9], neckDofAddresses: [0, 1, 2],
            neckRanges: [LimitPair(lower: -1, upper: 1), LimitPair(lower: -0.5, upper: 1.5), LimitPair(lower: 0, upper: 0)],
            neckActuators: [0, 1, 2],
            neckForceRanges: [LimitPair(lower: -2, upper: 4), LimitPair(lower: -5, upper: 0), LimitPair(lower: 0, upper: 0)],
            chestBodyId: 2,
            actuatorGears: [1, 2, 0.5, 4], actuatorDofs: [0, 1, 2, 3],
            limitQposAddresses: [7, 10, 11],
            limitRanges: [LimitPair(lower: -1, upper: 1), LimitPair(lower: 0, upper: 1), LimitPair(lower: -1, upper: 1)],
            weightNewtons: 2 * 9.81,
            supportGroups: supportGroups)
    }

    private func vestibularPhysics() -> FakeBodyPhysics {
        let physics = FakeBodyPhysics(nbody: 6, nq: 12, nv: 4, nu: 4)
        physics.sensordata = [0, 0, 9.81, 0.4, -1.0, 2.0]
        physics.qpos[7] = 0.5
        physics.qpos[8] = 1.6
        physics.qpos[9] = 0
        physics.qpos[10] = 0.98
        physics.qpos[11] = 0
        physics.qvel = [0.75, -1.5, 3.0, 0.2]
        physics.actuatorForce = [2, -1, -3, 1]
        physics.xmat.replaceSubrange(18 ..< 27, with: [1, 0, 0, 0, 1, 0, 0.1, 0.2, 0.97])
        return physics
    }

    private let vestibularContacts = [
        SupportContact(bodyId: 1, normalForce: 3, otherIsSelf: false),
        SupportContact(bodyId: 2, normalForce: 5, otherIsSelf: false),
        SupportContact(bodyId: 3, normalForce: 2, otherIsSelf: true),
        SupportContact(bodyId: 4, normalForce: 2, otherIsSelf: false),
        SupportContact(bodyId: 5, normalForce: 10, otherIsSelf: false),
        SupportContact(bodyId: 4, normalForce: -1, otherIsSelf: false),
    ]

    /// The 29 numbers `vestibular_from_raw` gave on the same raw values (float32, numpy,
    /// November 4, 2026).
    private let pythonVestibular: [Float] = [
        0.0, 0.0, 1.0, 0.20000000298023224, -0.5, 1.0, 0.5, 1.0, -1.0, 0.5, -1.0, 2.0, 0.5,
        -0.20000000298023224, -3.0, 0.10000000149011612, 0.20000000298023224, 0.9700000286102295,
        0.30000001192092896, 0.5, 0.0, 0.20000000298023224, 0.5096839666366577, 0.14848485589027405,
        0.15079365670681, 0.3333333432674408, 0.5, 0.9200000166893005, 1.0,
    ]

    private let vestibularCapacities = [1, 0.5, 0.9, 1, 1, 1, 1, 1, 0.8, 1.0]

    func testSupportForcesSkipSelfAndUngroupedBodies() throws {
        let forces = VestibularReader.supportForces(layout: try vestibularLayout(), contacts: vestibularContacts)
        XCTAssertEqual(forces, [3, 5, 0, 2])
    }

    func testVestibularFieldsMatchPythonReference() throws {
        let layout = try vestibularLayout()
        let physics = vestibularPhysics()
        let values = try VestibularReader.fields(
            layout: layout, sensordata: physics.sensordata, qpos: physics.qpos, qvel: physics.qvel,
            actuatorForce: physics.actuatorForce, xmat: physics.xmat,
            supportForces: VestibularReader.supportForces(layout: layout, contacts: vestibularContacts),
            capacities: vestibularCapacities)
        XCTAssertEqual(values.count, 29)
        for (index, value) in pythonVestibular.enumerated() {
            XCTAssertEqual(values[index], value, accuracy: 1e-6, VestibularReader.fieldNames[index])
        }
    }

    func testVestibularReaderReads29FreshNumbers() throws {
        let sample = try ProprioceptionSample(
            time: 0.01, bones: Array(repeating: 0, count: 286), energy: [0, 0, 0],
            capacities: vestibularCapacities, accumulated: Array(repeating: 0, count: 10))
        let sampler = FixedSampler(sample)
        let reader = VestibularReader(layout: try vestibularLayout(), tracker: sampler, support: FixedSupport(vestibularContacts))
        try reader.observeEnvStep(vestibularPhysics())
        let reading = try reader.read(vestibularPhysics(), at: ModelStep(count: 1))
        XCTAssertEqual(reading.sense, .vestibular)
        XCTAssertEqual(reading.primary.shape, [29])
        XCTAssertTrue(reading.isFresh)
        XCTAssertEqual(VestibularReader.fieldNames.count, 29)
        XCTAssertEqual(reading.primary.values[22], 0.5096839666366577, accuracy: 1e-6)
        XCTAssertEqual(reading.primary.values[28], 1)
        XCTAssertEqual(sampler.sampleCalls, 2)
    }

    func testContactListSupportBuildsOneRecordPerSkinBody() throws {
        // geoms 0 ... 3 on bodies 1, 2, 3, 9; skin on bodies 1, 2, 3; own bodies 1, 2, 3
        let support = ContactListSupport(geomBodyIds: [1, 2, 3, 9], skinBodyIds: [1, 2, 3], ownBodyIds: [1, 2, 3])
        let physics = FakeBodyPhysics(nbody: 10, nq: 1, nv: 1, nu: 1)
        let frame = [1.0, 0, 0, 0, 1, 0, 0, 0, 1]
        physics.contacts = try [
            // self-touch between bodies 1 and 2, fn = -4 N: two records of 4 N, other is self
            Contact(geomIds: [0, 1], distance: -0.001, position: [0, 0, 0], frame: frame, force: [-4, 0, 0, 0, 0, 0]),
            // body 3 against the outside body 9, fn = 6 N: one record, other is not self
            Contact(geomIds: [3, 2], distance: -0.001, position: [0, 0, 0], frame: frame, force: [6, 1, 1, 0, 0, 0]),
            // below 1e-9 N: skipped
            Contact(geomIds: [2, 3], distance: 0, position: [0, 0, 0], frame: frame, force: [1e-10, 0, 0, 0, 0, 0]),
            // a flex geom (-1): skipped
            Contact(geomIds: [-1, 3], distance: 0, position: [0, 0, 0], frame: frame, force: [5, 0, 0, 0, 0, 0]),
        ]
        let records = try support.supportContacts(physics)
        XCTAssertEqual(records, [
            SupportContact(bodyId: 1, normalForce: 4, otherIsSelf: true),
            SupportContact(bodyId: 2, normalForce: 4, otherIsSelf: true),
            SupportContact(bodyId: 3, normalForce: 6, otherIsSelf: false),
        ])
    }

    // MARK: Names and layouts

    func testBodySetNameRule() {
        XCTAssertEqual(BodySet.of(mujocoBodyNamed: "left_hand"), .handL)
        XCTAssertEqual(BodySet.of(mujocoBodyNamed: "right_ffdistal"), .handR)
        XCTAssertEqual(BodySet.of(mujocoBodyNamed: "left_thbase"), .handL)
        XCTAssertEqual(BodySet.of(mujocoBodyNamed: "right_upper_arm"), .armR)
        XCTAssertEqual(BodySet.of(mujocoBodyNamed: "left_lower_leg"), .legL)
        XCTAssertEqual(BodySet.of(mujocoBodyNamed: "right_big_toe"), .footR)
        XCTAssertEqual(BodySet.of(mujocoBodyNamed: "left_eye"), .head)
        XCTAssertEqual(BodySet.of(mujocoBodyNamed: "head"), .head)
        XCTAssertEqual(BodySet.of(mujocoBodyNamed: "lower_body"), .spine)
        XCTAssertEqual(BodySet.of(mujocoBodyNamed: "chest"), .spine)
        XCTAssertNil(BodySet.of(mujocoBodyNamed: "left_shoulder"))
        XCTAssertNil(BodySet.of(mujocoBodyNamed: "crib_rail"))
        XCTAssertEqual(BodySet.ofRows.count, 26)
        XCTAssertEqual(ProprioceptionLayout.bodyNames.count, 26)
    }

    func testSupportGroupNames() {
        XCTAssertEqual(VestibularLayout.supportGroup(ofBodyNamed: "head"), 0)
        XCTAssertEqual(VestibularLayout.supportGroup(ofBodyNamed: "left_eye"), 0)
        XCTAssertEqual(VestibularLayout.supportGroup(ofBodyNamed: "upper_body"), 1)
        XCTAssertEqual(VestibularLayout.supportGroup(ofBodyNamed: "right_mfdistal"), 2)
        XCTAssertEqual(VestibularLayout.supportGroup(ofBodyNamed: "left_upper_arm"), 2)
        XCTAssertEqual(VestibularLayout.supportGroup(ofBodyNamed: "right_lower_leg"), 3)
        XCTAssertEqual(VestibularLayout.supportGroup(ofBodyNamed: "left_toes"), 3)
        XCTAssertEqual(VestibularLayout.supportGroup(ofBodyNamed: "floor"), -1)
    }

    func testProprioceptionLayoutFromModel() throws {
        let layout = try ProprioceptionLayout.make(from: FakeModelDescription())
        XCTAssertEqual(layout.bodyIds, Array(1 ... 26))
        // joints 0-2 on the head, joint 3 on left_eye (head set), joint 4 on left_ffdistal
        XCTAssertEqual(layout.actuatorBodySets, [.head, .head, .head, .head, .handL])
        XCTAssertEqual(layout.actuatorDofs, [0, 1, 2, 3, 4])
        XCTAssertEqual(layout.actuatorGears, [1, 2, 0.5, 4, 3])
    }

    func testVestibularLayoutFromModel() throws {
        let model = FakeModelDescription()
        // every body but the world (body 0) shares hip's root body 27
        XCTAssertEqual(VestibularLayout.ownBodyIds(of: model), Set(1 ... 27))
        let layout = try VestibularLayout.make(from: model, ownBodyIds: [1, 2, 3])
        XCTAssertEqual(layout.accelerometerAddresses, [0, 1, 2])
        XCTAssertEqual(layout.gyroAddresses, [3, 4, 5])
        XCTAssertEqual(layout.neckQposAddresses, [7, 8, 9])
        XCTAssertEqual(layout.neckDofAddresses, [0, 1, 2])
        XCTAssertEqual(layout.neckActuators, [0, 1, 2])
        XCTAssertEqual(layout.neckForceRanges, Array(model.forceRanges[0 ..< 3]))
        XCTAssertEqual(layout.chestBodyId, 26)
        // the eye joint (index 3, name containing "_eye_") is left out of limit_load
        XCTAssertEqual(layout.limitQposAddresses, [7, 8, 9, 11])
        XCTAssertEqual(layout.weightNewtons, 0.3 * 9.81, accuracy: 1e-12)
        XCTAssertEqual(layout.supportGroups.count, 28)
        XCTAssertEqual(layout.supportGroups[0], -1)
        XCTAssertEqual(layout.supportGroups[13], 0)
        XCTAssertEqual(layout.supportGroups[24], 0)
        XCTAssertEqual(layout.supportGroups[26], 1)
        XCTAssertEqual(layout.supportGroups[27], 1)
        XCTAssertEqual(layout.supportGroups[3], 2)
        XCTAssertEqual(layout.supportGroups[18], 3)
        let native = Array(repeating: LimitPair(lower: -9, upper: 9), count: 5)
        let withNative = try VestibularLayout.make(from: model, ownBodyIds: [], nativeForceRanges: native)
        XCTAssertEqual(withNative.neckForceRanges, Array(native[0 ..< 3]))
        XCTAssertEqual(withNative.weightNewtons, 1.0)
        XCTAssertThrowsError(try VestibularLayout.make(from: model, ownBodyIds: [], nativeForceRanges: [native[0]]))
        XCTAssertThrowsError(try VestibularLayout.make(from: model, ownBodyIds: [99])) { error in
            XCTAssertEqual(error as? BodySenseError, .arrayTooShort(array: "body_mass", index: 99, count: 28))
        }
    }

    // MARK: Errors

    func testThrownReadLeavesFatigueUnchanged() throws {
        let physics = FakeBodyPhysics.proprioceptionBody()
        let fullPositions = physics.xpos
        // bodies 0 ... 9 only: row 9 (body 10) needs xpos index 32
        physics.xpos = Array(fullPositions.prefix(30))
        let tracker = BodyStateTracker(layout: try fakeLayout())
        XCTAssertThrowsError(try tracker.sample(physics)) { error in
            XCTAssertEqual(error as? BodySenseError, .arrayTooShort(array: "xpos", index: 32, count: 30))
        }
        XCTAssertEqual(tracker.fatigue, FatigueModel())
        physics.xpos = fullPositions
        _ = try tracker.sample(physics)
        XCTAssertEqual(tracker.fatigue.capacities[BodySet.handL.rawValue], 0.9998, accuracy: 1e-12)
    }

    func testCountAndActuatorErrors() throws {
        XCTAssertThrowsError(try ProprioceptionLayout(bodyIds: [1], actuatorBodySets: [], actuatorDofs: [], actuatorGears: [])) { error in
            XCTAssertEqual(error as? BodySenseError, .wrongCount(type: "ProprioceptionLayout.bodyIds", expected: 26, found: 1))
        }
        XCTAssertThrowsError(try ProprioceptionSample(
            time: 0, bones: Array(repeating: 0, count: 286), energy: [0, 0, 0],
            capacities: Array(repeating: 1, count: 10), accumulated: [0])) { error in
            XCTAssertEqual(error as? BodySenseError, .wrongCount(type: "ProprioceptionSample.accumulated", expected: 10, found: 1))
        }
        // actuator 4 on a tendon transmission has no single joint speed to multiply
        var tendonModel = FakeModelDescription()
        tendonModel.transmissionTypes[4] = .tendon
        XCTAssertThrowsError(try ProprioceptionLayout.make(from: tendonModel)) { error in
            XCTAssertEqual(error as? BodySenseError, .unsupportedActuator(actuator: 4))
        }
        XCTAssertThrowsError(try VestibularLayout.make(from: tendonModel, ownBodyIds: [])) { error in
            XCTAssertEqual(error as? BodySenseError, .unsupportedActuator(actuator: 4))
        }
    }

    func testSupportForcesIgnoreGroupsOutsideTheFour() throws {
        let layout = try vestibularLayout(supportGroups: [-1, 7, 2])
        let contacts = [
            SupportContact(bodyId: 1, normalForce: 4, otherIsSelf: false),
            SupportContact(bodyId: 2, normalForce: 3, otherIsSelf: false),
            SupportContact(bodyId: 9, normalForce: 5, otherIsSelf: false),
        ]
        XCTAssertEqual(VestibularReader.supportForces(layout: layout, contacts: contacts), [0, 0, 3, 0])
    }

    // MARK: The model description over UEMAPhysics's tables

    /// The tables case of the header: five bodies, three joints, two actuators, two sensors.
    private func physicsTables(jointType: [Int] = [3, 3, 2], parents: [Int] = [0, 0, 1, 2, 0]) throws -> PhysicsModelTables {
        try PhysicsModelTables(
            timestep: 0.005,
            bodyNames: ["world", "hip", "chest", "head", "crib"], bodyParentID: parents,
            bodyMass: [0, 1, 2, 0.5, 9],
            jointNames: ["robot:head_swivel", "robot:left_eye_horizontal", "crib_slide"],
            jointType: jointType, jointBodyID: [3, 3, 4],
            jointQposAddress: [0, 1, 2], jointDofAddress: [0, 1, 2], jointLimited: [true, true, false],
            jointRangeLow: [-1, -0.7, 0], jointRangeHigh: [1, 0.7, 0],
            qpos0: [0, 0, 0],
            actuatorNames: ["act:head_swivel", "act:eye"], actuatorTransmissionType: [0, 0],
            actuatorTransmissionID: [0, 1],
            actuatorControlLimited: [true, true], actuatorControlRangeLow: [-1, -1],
            actuatorControlRangeHigh: [1, 1], actuatorForceLimited: [true, true],
            actuatorForceRangeLow: [-2, -1], actuatorForceRangeHigh: [4, 1],
            actuatorGainType: [0, 0], actuatorBiasType: [0, 0], actuatorGainParameter0: [1, 1],
            actuatorGear0: [1.5, 0.25],
            sensorNames: ["vestibular_acc", "vestibular_gyro"], sensorAddress: [0, 3], sensorDimension: [3, 3],
            cameraNames: ["eye_left"])
    }

    private let tablesSizes = PhysicsSizes(nq: 3, nv: 3, nu: 2, nbody: 5, njnt: 3, nsensordata: 6, ngeom: 2, ncam: 1)

    func testModelTablesDescriptionLookupsAndRoots() throws {
        let model = try ModelTablesDescription(tables: try physicsTables(), sizes: tablesSizes, geomBodyIds: [3, 4])
        XCTAssertEqual((0 ..< 5).map { model.bodyRootId($0) }, [0, 1, 1, 1, 4])
        XCTAssertEqual(VestibularLayout.ownBodyIds(of: model), [1, 2, 3])
        XCTAssertEqual(model.id(of: .body, named: "chest"), 2)
        XCTAssertEqual(model.id(of: .camera, named: "eye_left"), 0)
        XCTAssertEqual(model.id(of: .sensor, named: "vestibular_gyro"), 1)
        XCTAssertNil(model.id(of: .geom, named: "any"))
        XCTAssertNil(model.id(of: .body, named: ""))
        XCTAssertEqual(model.name(of: .joint, id: 2), "crib_slide")
        XCTAssertNil(model.name(of: .body, id: 9))
        XCTAssertEqual(model.jointType(2), .slide)
        XCTAssertEqual(model.jointRange(1), LimitPair(lower: -0.7, upper: 0.7))
        XCTAssertEqual(model.actuatorTransmissionType(1), .joint)
        XCTAssertEqual(model.actuatorForceRange(0), LimitPair(lower: -2, upper: 4))
        XCTAssertEqual(model.actuatorGear(1), 0.25)
        XCTAssertEqual(model.geomBodyId(1), 4)
        XCTAssertEqual(model.geomBodyId(-1), -1)
        XCTAssertEqual(model.sensorAddress(1), 3)
        let eye = try model.scalarJoint(ofActuator: 1)
        XCTAssertEqual(eye.joint, 1)
        XCTAssertEqual(eye.dof, 1)
        // the tables lack the 26 row bodies and two of the neck joints: the layouts name them
        XCTAssertThrowsError(try ProprioceptionLayout.make(from: model)) { error in
            XCTAssertEqual(error as? BodySenseError, .missingName(kind: .body, name: "left_hand"))
        }
        XCTAssertThrowsError(try VestibularLayout.make(from: model, ownBodyIds: [1, 2, 3])) { error in
            XCTAssertEqual(error as? BodySenseError, .missingName(kind: .joint, name: "robot:head_tilt"))
        }
    }

    func testModelTablesDescriptionRejectsBadTables() throws {
        let tables = try physicsTables()
        let fourBodies = PhysicsSizes(nq: 3, nv: 3, nu: 2, nbody: 4, njnt: 3, nsensordata: 6, ngeom: 2, ncam: 1)
        XCTAssertThrowsError(try ModelTablesDescription(tables: tables, sizes: fourBodies)) { error in
            XCTAssertEqual(error as? BodySenseError, .wrongCount(type: "PhysicsModelTables.bodyNames", expected: 4, found: 5))
        }
        XCTAssertThrowsError(try ModelTablesDescription(tables: tables, sizes: tablesSizes, geomBodyIds: [1])) { error in
            XCTAssertEqual(error as? BodySenseError, .wrongCount(type: "ModelTablesDescription.geomBodyIds", expected: 2, found: 1))
        }
        XCTAssertThrowsError(try ModelTablesDescription(tables: try physicsTables(jointType: [3, 3, 7]), sizes: tablesSizes)) { error in
            XCTAssertEqual(error as? BodySenseError, .invalidTableValue(table: "jnt_type", index: 2, value: 7))
        }
        XCTAssertThrowsError(try ModelTablesDescription.rootIds(parents: [0, 5])) { error in
            XCTAssertEqual(error as? BodySenseError, .invalidTableValue(table: "body_parentid", index: 1, value: 5))
        }
        // bodies 1 and 2 name each other as parent: the chain never reaches the world body
        XCTAssertThrowsError(try ModelTablesDescription.rootIds(parents: [0, 2, 1]))
    }
}
