// CommandTestBodies.swift: small hand-built bodies, body states and fakes for the tests of the
// UEMACommand module (work item WI-37). No MuJoCo: every table is written out here, every
// state is a fixed value, and the one moving body is a test plant (a single hinge with an
// inertia chosen for the test, not a property of the MIMo body). Nothing here was compiled on
// the machine that wrote it (Windows 11, no Swift toolchain).

import Foundation
import XCTest
import UEMACore
import UEMAPhysics
import UEMACommand

/// One hinge joint of a test body.
struct TestJoint {
    var name: String
    var body: Int
    var axis: Vector3
    var anchor: Vector3 = .zero
    var limited: Bool = false
    var low: Double = 0
    var high: Double = 0
}

enum TestBodies {
    /// Tables of a body with the given bodies (name, parent id) and hinge joints, one motor per
    /// joint in joint sequence, qpos and qvel address = joint id, control range -1..1 (limited),
    /// force range `forceLow`..`forceHigh`, fixed gain `gain`, no bias.
    static func tables(bodies: [(String, Int)],
                       joints: [TestJoint],
                       forceLow: Double = -1,
                       forceHigh: Double = 1,
                       gain: Double = 1,
                       gainType: Int = 0) throws -> ActuatedBodyTables {
        let n = joints.count
        return try ActuatedBodyTables(
            actuatorCount: n,
            bodyNames: bodies.map { $0.0 },
            bodyParentID: bodies.map { $0.1 },
            jointNames: joints.map { $0.name },
            jointType: Array(repeating: ActuatedBodyTables.hingeJointType, count: n),
            jointBodyID: joints.map { $0.body },
            jointQposAddress: Array(0..<n),
            jointDofAddress: Array(0..<n),
            jointLimited: joints.map { $0.limited },
            jointRangeLow: joints.map { $0.low },
            jointRangeHigh: joints.map { $0.high },
            actuatorTransmissionType: Array(repeating: ActuatedBodyTables.jointTransmission, count: n),
            actuatorTransmissionID: Array(0..<n),
            actuatorControlLimited: Array(repeating: true, count: n),
            actuatorControlRangeLow: Array(repeating: -1, count: n),
            actuatorControlRangeHigh: Array(repeating: 1, count: n),
            actuatorForceRangeLow: Array(repeating: forceLow, count: n),
            actuatorForceRangeHigh: Array(repeating: forceHigh, count: n),
            actuatorGainType: Array(repeating: gainType, count: n),
            actuatorBiasType: Array(repeating: 0, count: n),
            actuatorGainParameter0: Array(repeating: gain, count: n))
    }

    /// A one-handle layout: handle `name` on body `bone`, its chain `chain`.
    static func oneHandleLayout(name: String = "head",
                                bone: String,
                                chain: [String],
                                position: Bool = false,
                                eyeJoints: [HandleLayout.EyeJoint] = []) -> HandleLayout {
        HandleLayout(names: [name],
                     bones: [name: bone],
                     chains: [name: chain],
                     positionHandles: position ? [name] : [],
                     fingerDigits: ["thumb", "index", "middle", "ring", "pinky"],
                     eyeJoints: eyeJoints)
    }

    /// The rotation about z by `angle` rad as a row-major matrix.
    static func rotationZ(_ angle: Double) -> Matrix3 {
        Matrix3(rowMajor: [cos(angle), -sin(angle), 0, sin(angle), cos(angle), 0, 0, 0, 1])
    }

    /// The quaternion x y z w of the rotation vector `v` (axis x angle, rad).
    static func quaternion(rotationVector v: Vector3) -> [Double] {
        let angle = v.norm
        if angle < 1e-12 {
            return [0, 0, 0, 1]
        }
        let s = sin(angle / 2) / angle
        return [v.x * s, v.y * s, v.z * s, cos(angle / 2)]
    }

    /// One command row: position (cm), quaternion x y z w, radius 0, strength.
    static func row(position: [Double] = [0, 0, 0], quaternion: [Double] = [0, 0, 0, 1], strength: Double) -> [Double] {
        position + quaternion + [0, strength]
    }
}

/// A fixed body state.
struct FixedBodyState: BodyStateReading {
    var qpos: [Double]
    var qvel: [Double]
    var positions: [Vector3]
    var rotations: [Matrix3]
    var axes: [Vector3]
    var anchors: [Vector3]

    func bodyPosition(_ body: Int) -> Vector3 { positions[body] }
    func bodyRotation(_ body: Int) -> Matrix3 { rotations[body] }
    func jointAxis(_ joint: Int) -> Vector3 { axes[joint] }
    func jointAnchor(_ joint: Int) -> Vector3 { anchors[joint] }
}

/// The 10 body-set capacities, settable.
final class FixedCapacities: BodySetCapacityReading {
    var bodySetCapacities: [Double]

    init(_ capacities: [Double]) {
        bodySetCapacities = capacities
    }
}

/// A test plant behind UEMACore's Physics: 90 hinge joints (MotorControls.count), one motor each;
/// joint 0 sits on body 1 ("left_upper_arm", body set Arm_L) and turns about z at the origin, joints 1-89 sit
/// on body 2 ("dummy") and never move. Per physics step of 0.005 s, joint 0's speed gains
/// gain0 x ctrl0 / inertia x 0.005 and its angle gains speed x 0.005 (semi-implicit Euler).
/// The inertia is a test constant (0.1, chosen so 25 env steps of a saturated control turn the
/// joint by 0.22 rad), not a property of the MIMo body.
final class PlantPhysics: Physics, ActuatorStrengthWriting {
    static let motorCount = MotorControls.count
    static let inertia = 0.1

    var angle = 0.0
    var speed = 0.0
    var gain: [Double] = Array(repeating: 1, count: PlantPhysics.motorCount)
    var forceLow: [Double] = Array(repeating: -1, count: PlantPhysics.motorCount)
    var forceHigh: [Double] = Array(repeating: 1, count: PlantPhysics.motorCount)
    var control: [Double] = Array(repeating: 0, count: PlantPhysics.motorCount)
    var writtenControls: [MotorControls] = []
    var strengthWrites = 0
    private(set) var time = 0.0
    private(set) var isLoaded = true

    var sizes: PhysicsSizes {
        PhysicsSizes(nq: PlantPhysics.motorCount, nv: PlantPhysics.motorCount, nu: PlantPhysics.motorCount,
                     nbody: 3, njnt: PlantPhysics.motorCount, nsensordata: 0, ngeom: 0, ncam: 0)
    }

    var qpos: [Double] {
        var q = Array(repeating: 0.0, count: PlantPhysics.motorCount)
        q[0] = angle
        return q
    }

    var qvel: [Double] {
        var v = Array(repeating: 0.0, count: PlantPhysics.motorCount)
        v[0] = speed
        return v
    }

    var sensordata: [Double] { [] }
    var xpos: [Double] { Array(repeating: 0, count: 9) }
    var xquat: [Double] { [1, 0, 0, 0, cos(angle / 2), 0, 0, sin(angle / 2), 1, 0, 0, 0] }
    var xmat: [Double] { Matrix3.identity.m + TestBodies.rotationZ(angle).m + Matrix3.identity.m }

    var xaxis: [Double] {
        var out: [Double] = []
        for _ in 0..<PlantPhysics.motorCount { out += [0, 0, 1] }
        return out
    }

    var xanchor: [Double] { Array(repeating: 0, count: 3 * PlantPhysics.motorCount) }
    var actuatorForce: [Double] { zip(gain, control).map { $0 * $1 } }
    var contacts: [Contact] { [] }

    func id(of kind: MuJoCoObjectKind, named name: String) -> Int? { nil }
    func bodyVelocity(bodyId: Int) -> [Double] { Array(repeating: 0, count: 6) }
    func load(sceneXML url: URL) throws { isLoaded = true }

    func reset() throws {
        angle = 0
        speed = 0
        time = 0
    }

    func setState(qpos: [Double], qvel: [Double]) throws {
        angle = qpos.first ?? 0
        speed = qvel.first ?? 0
    }

    func write(controls: MotorControls) throws {
        control = controls.clipped().values
        writtenControls.append(controls)
    }

    func stepPhysicsStep() throws {
        let dt = ModelStep.physicsTimestepSeconds
        speed += dt * gain[0] * control[0] / PlantPhysics.inertia
        angle += dt * speed
        time += dt
    }

    func setActuatorStrength(_ strength: ActuatorStrength) throws {
        gain = strength.gain
        forceLow = strength.forceLow
        forceHigh = strength.forceHigh
        strengthWrites += 1
    }

    /// The tables of this plant: bodies world, left_upper_arm, dummy; joint 0 named
    /// "plant_hinge" on body 1, joints 1-89 "dummy_<k>" on body 2.
    static func tables() throws -> ActuatedBodyTables {
        var joints = [TestJoint(name: "plant_hinge", body: 1, axis: Vector3(0, 0, 1))]
        for k in 1..<motorCount {
            joints.append(TestJoint(name: "dummy_\(k)", body: 2, axis: Vector3(0, 0, 1)))
        }
        return try TestBodies.tables(bodies: [("world", 0), ("left_upper_arm", 0), ("dummy", 0)], joints: joints)
    }

    /// The 26-handle layout over this plant: every handle on body left_upper_arm; only the head
    /// handle's chain holds a hinge (joint 0).
    static func layout() -> HandleLayout {
        var bones: [String: String] = [:]
        var chains: [String: [String]] = [:]
        for name in BodyHandle.pythonNames {
            bones[name] = "left_upper_arm"
            chains[name] = []
        }
        chains["head"] = ["left_upper_arm"]
        return HandleLayout(names: BodyHandle.pythonNames, bones: bones, chains: chains,
                            positionHandles: [], fingerDigits: ["thumb", "index", "middle", "ring", "pinky"],
                            eyeJoints: [])
    }
}

extension PhysicsModelTables {
    /// UEMAPhysics's model tables holding the same numbers as `t` (body mass 1 kg, gear 1,
    /// qpos0 all 0, no sensors, no cameras): the input of ActuatedBodyTables(modelTables:).
    static func mirroring(_ t: ActuatedBodyTables, nq: Int) throws -> PhysicsModelTables {
        try PhysicsModelTables(
            timestep: ModelStep.physicsTimestepSeconds,
            bodyNames: t.bodyNames, bodyParentID: t.bodyParentID,
            bodyMass: Array(repeating: 1, count: t.bodyCount),
            jointNames: t.jointNames, jointType: t.jointType, jointBodyID: t.jointBodyID,
            jointQposAddress: t.jointQposAddress, jointDofAddress: t.jointDofAddress,
            jointLimited: t.jointLimited,
            jointRangeLow: t.jointRangeLow, jointRangeHigh: t.jointRangeHigh,
            qpos0: Array(repeating: 0, count: nq),
            actuatorNames: (0..<t.actuatorCount).map { "act_\($0)" },
            actuatorTransmissionType: t.actuatorTransmissionType,
            actuatorTransmissionID: t.actuatorTransmissionID,
            actuatorControlLimited: t.actuatorControlLimited,
            actuatorControlRangeLow: t.actuatorControlRangeLow,
            actuatorControlRangeHigh: t.actuatorControlRangeHigh,
            actuatorForceLimited: Array(repeating: true, count: t.actuatorCount),
            actuatorForceRangeLow: t.actuatorForceRangeLow,
            actuatorForceRangeHigh: t.actuatorForceRangeHigh,
            actuatorGainType: t.actuatorGainType, actuatorBiasType: t.actuatorBiasType,
            actuatorGainParameter0: t.actuatorGainParameter0,
            actuatorGear0: Array(repeating: 1, count: t.actuatorCount),
            sensorNames: [], sensorAddress: [], sensorDimension: [], cameraNames: [])
    }
}

/// The test plant as UEMAPhysics's model description and scratch kinematics, so that
/// CommandDecoder.released(physics:) can be called on it. The scratch position is the origin
/// for every body and angle.
extension PlantPhysics: PhysicsModelDescribing, ScratchKinematicsQuerying {
    func modelTables() throws -> PhysicsModelTables {
        try PhysicsModelTables.mirroring(PlantPhysics.tables(), nq: PlantPhysics.motorCount)
    }

    func scratchBodyPosition(ofBody body: Int, withJointAngles jointAngles: [Int: Double]) throws -> [Double] {
        [0, 0, 0]
    }
}

/// A recording fake of UEMAPhysics's scratch kinematics and actuator parameter writes.
final class FakeModelSide: ScratchKinematicsQuerying, ActuatorParameterWriting {
    var position: [Double] = [0.1, 0.2, 0.3]
    var calls: [String] = []
    var gain: [Double] = []
    var forceLow: [Double] = []
    var forceHigh: [Double] = []

    func scratchBodyPosition(ofBody body: Int, withJointAngles jointAngles: [Int: Double]) throws -> [Double] {
        position
    }

    func setActuatorGainParameter0(_ values: [Double]) throws {
        calls.append("gain")
        gain = values
    }

    func setActuatorForceRange(low: [Double], high: [Double]) throws {
        calls.append("forceRange")
        forceLow = low
        forceHigh = high
    }
}
